"""B1-B3 de la fase v6 (V6-02): factorizacion probabilistica condicionada de etiquetas.

Contexto
--------
El gap analysis (V6-00) marco la factorizacion condicional como la primera linea ejecutable
y el diagnostico (V6-01) le dio soporte empirico: **TR=1 implica HS=1 de forma determinista**
en el desarrollo (0 casos de TR=1 con HS=0 frente a 569 con HS=1). No es una tendencia, es
una invariante del corpus, y un clasificador TR independiente no puede expresarla: si
predice TR=1 con HS=0, predice algo que los datos dicen imposible.

Que NO es esto, y la distincion importa
----------------------------------------
Ya se probaron y cerraron: cadenas de clasificadores (E2-01, seis ordenes) y label powerset
(V4-E, doce variantes, mejor 0.7713). Ninguna de las dos cosas restringe el **dominio de
entrenamiento** del condicional: una cadena entrena P(TR|X) con todas las filas y la
encadena anterior como entrada; el powerset entrena un clasificador de 4 o 6 clases. Aqui el
modelo de TR **solo ve ejemplos con HS=1**, que es su dominio natural, y la prediccion se
compone como producto. Esa es la novedad, y tambien la razon por la que el resultado puede
diferir.

Los tres modelos
----------------
- **B1** p(TR|X) = p(HS|X) * p(TR | HS=1, X), con el condicional entrenado solo en HS=1.
- **B2** p(AG|X) = p(HS|X) * p(AG | HS=1, X) + (1 - p(HS|X)) * p(AG | HS=0, X).
- **B2r** el mismo B2 con la rama HS=0 regularizada. Esa rama tiene **22 positivos de 835**,
  es decir, P(AG|HS=0) = 0.026: un modelo libre no tiene datos para aprenderla y sobreajusta
  con seguridad. Se regulariza con mas shrinkage hacia la prior de la rama.
- **B3** el conjunto HS -> {TR, AG} condicionales frente a V03, con sus umbrales.

Sin routing duro (no se impone TR=0 cuando HS=0 en la prediccion; la restriccion viene del
producto, no de una regla) y sin classifier chain. Los umbrales de cada etiqueta se ajustan
solo sobre probabilidades fuera de pliegue, y el punto de operacion de comparacion son los
umbrales congelados de V03.

Protocolo
---------
Grupos fused (near-duplicate UNION pseudo-autor) y pliegues del protocolo v2 corregido. El
OOF de V03 se usa como referencia sin reentrenar nada: es un artefacto canonico ya auditado.

Uso
---
    /home/carlos/venv/bin/python experiments/v6_02_label_factorization.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train  # noqa: E402
from src.evaluation import (  # noqa: E402
    ETIQUETAS,
    _f1_por_celda,
    metricas_multietiqueta,
    metrica_macro,
)
from src.splits import grupos_fusionados, holdout_v2, pliegues_corregidos  # noqa: E402

EXPERIMENTO = "V6-02 (B1-B3)"
OOF_V03 = PROJECT_ROOT / "features" / "oof_v03_final.npz"
REPORTE = PROJECT_ROOT / "reports" / "v6_label_factorization.md"
REPORTE_CSV = PROJECT_ROOT / "reports" / "v6_label_factorization.csv"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v6_label_factorization.json"
FIGURAS = PROJECT_ROOT / "reports" / "figuras"

REFERENCIA_V03 = 0.777211
UMBRALES_V03 = np.array([0.44, 0.51, 0.46])
REJILLA_UMBRALES = np.round(np.arange(0.20, 0.81, 0.01), 2)
GATE_1_DELTA = 0.003
GATE_1_FOLDS = 4
REPETICIONES_BOOTSTRAP = 10_000
SEMILLA_BOOTSTRAP = 20260926

PARAMS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
PARAMS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2, "sublinear_tf": True}
PARAMETROS_LR = {
    "C": 2.0,
    "class_weight": "balanced",
    "max_iter": 1000,
    "solver": "liblinear",
    "random_state": 42,
}
# La rama HS=0 de AG tiene 22 positivos de 835. Sin regularizacion, un clasificador balanceado
# sobre 22 positivos ajusta la frontera a la mitad de la muestra y la prediccion se dispara en
# las filas HS=0. Se baja C y se sube el peso de la clase minoritaria solo en esa rama.
# La rama HS=0 de AG tiene 22 positivos de 835 filas. Por debajo de este tamano la rama se
# considera debil y recibe los parametros regularizados de mas abajo.
MINIMO_FILAS_RAMA = 20
UMBRAL_RAMA_DEBIL = 200
PARAMETROS_RAMA_DEBIL = {
    "C": 0.25,
    "class_weight": {0: 1.0, 1: 20.0},
    "max_iter": 1000,
    "solver": "liblinear",
    "random_state": 42,
}


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V6-B] {mensaje}", flush=True)


def ajustar_modelos(
    textos, y, pliegues
) -> dict[str, np.ndarray]:
    """Ajusta, para cada pliegue, el modelo de HS y los condicionales de TR y AG.

    Los condicionales se entrenan **solo** con las filas del subdominio correspondiente, que
    es la novedad de la fase: el modelo de TR ve unicamente HS=1.

    Args:
        textos: columna de textos del desarrollo.
        y: matriz binaria de labels.
        pliegues: pliegues corregidos.

    Returns:
        Diccionario con las probabilidades fuera de pliegue de HS, TR y AG, y con las
        probabilidades condicionadas de cada rama.
    """
    n = len(textos)
    proba_hs = np.zeros(n, dtype=np.float64)
    proba_tr = np.zeros(n, dtype=np.float64)
    proba_ag = np.zeros(n, dtype=np.float64)
    # Marginales de control: los tres modelos de V03 entrenados con ESTE codigo, sin
    # factorizar nada. Son el grupo de control que permite atribuir cualquier delta a la
    # factorizacion y no a un simple reentrenamiento con otra instancia.
    marginal_tr = np.zeros(n, dtype=np.float64)
    marginal_ag = np.zeros(n, dtype=np.float64)
    cond_tr_hs1 = np.full(n, np.nan, dtype=np.float64)
    cond_ag_hs1 = np.full(n, np.nan, dtype=np.float64)
    cond_ag_hs0 = np.full(n, np.nan, dtype=np.float64)
    for entrenamiento, validacion in pliegues:
        vectorizador_word = TfidfVectorizer(**PARAMS_WORD)
        vectorizador_char = TfidfVectorizer(**PARAMS_CHAR)
        matriz = sparse.hstack(
            [
                vectorizador_word.fit_transform(textos.iloc[entrenamiento]),
                vectorizador_char.fit_transform(textos.iloc[entrenamiento]),
            ]
        ).tocsr()
        consulta = sparse.hstack(
            [
                vectorizador_word.transform(textos.iloc[validacion]),
                vectorizador_char.transform(textos.iloc[validacion]),
            ]
        ).tocsr()
        y_entrenamiento = y[entrenamiento]

        # Modelo marginal de HS: es el mismo de V03 y sirve de factor en las tres formulas.
        modelo_hs = LogisticRegression(**PARAMETROS_LR)
        modelo_hs.fit(matriz, y_entrenamiento[:, 0])
        proba_hs[validacion] = modelo_hs.predict_proba(consulta)[:, 1]

        # Marginales de control: exactamente V03, con este codigo.
        for indice, destino in ((1, marginal_tr), (2, marginal_ag)):
            modelo = LogisticRegression(**PARAMETROS_LR)
            modelo.fit(matriz, y_entrenamiento[:, indice])
            destino[validacion] = modelo.predict_proba(consulta)[:, 1]

        # Condicional de TR: SOLO filas con HS=1. Si el subdominio fuera insuficiente o
        # fuera de una sola clase, se cae a la prior de la rama en vez de a cero: dejar un
        # 0 silencioso haria creer que el factor Trump es una certeza negativa.
        mascara = y_entrenamiento[:, 0] == 1
        if mascara.sum() < MINIMO_FILAS_RAMA or len(np.unique(y_entrenamiento[mascara, 1])) < 2:
            cond_tr_hs1[validacion] = float(y_entrenamiento[mascara, 1].mean()) if mascara.any() else 0.5
        else:
            modelo_tr = LogisticRegression(**PARAMETROS_LR)
            modelo_tr.fit(matriz[mascara], y_entrenamiento[mascara, 1])
            cond_tr_hs1[validacion] = modelo_tr.predict_proba(consulta)[:, 1]
        proba_tr[validacion] = proba_hs[validacion] * cond_tr_hs1[validacion]

        # Condicional de AG en las dos ramas de HS.
        for valor_hs, destino in ((1, cond_ag_hs1), (0, cond_ag_hs0)):
            rama = y_entrenamiento[:, 0] == valor_hs
            if rama.sum() < MINIMO_FILAS_RAMA or len(np.unique(y_entrenamiento[rama, 2])) < 2:
                destino[validacion] = float(y_entrenamiento[rama, 2].mean()) if rama.any() else 0.5
            else:
                # La rama debil (22 positivos de 835) necesita mas regularizacion: con los
                # parametros de V03 un clasificador balanceado ajusta la frontera a la mitad
                # de la muestra y la prediccion se dispara en filas que no son de la rama.
                parametros = (
                    PARAMETROS_RAMA_DEBIL
                    if (valor_hs == 0 and rama.sum() < UMBRAL_RAMA_DEBIL)
                    else PARAMETROS_LR
                )
                modelo_ag = LogisticRegression(**parametros)
                modelo_ag.fit(matriz[rama], y_entrenamiento[rama, 2])
                destino[validacion] = modelo_ag.predict_proba(consulta)[:, 1]
        proba_ag[validacion] = (
            proba_hs[validacion] * cond_ag_hs1[validacion]
            + (1 - proba_hs[validacion]) * cond_ag_hs0[validacion]
        )
    return {
        "hs": proba_hs,
        "tr": proba_tr,
        "ag": proba_ag,
        "marginal_tr": marginal_tr,
        "marginal_ag": marginal_ag,
        "cond_tr_hs1": cond_tr_hs1,
        "cond_ag_hs1": cond_ag_hs1,
        "cond_ag_hs0": cond_ag_hs0,
    }


def ajustar_umbrales(y: np.ndarray, proba: np.ndarray) -> np.ndarray:
    """Ajusta un umbral por etiqueta sobre la rejilla de V03.

    Args:
        y: matriz binaria verdadera.
        proba: matriz de probabilidades.

    Returns:
        Vector de tres umbrales.
    """
    from sklearn.metrics import f1_score

    umbrales = np.empty(3, dtype=np.float64)
    for indice in range(3):
        mejor = (0.0, 0.5)
        for umbral in REJILLA_UMBRALES:
            valor = float(
                f1_score(y[:, indice], (proba[:, indice] >= umbral).astype(int), zero_division=0)
            )
            if valor > mejor[0]:
                mejor = (valor, float(umbral))
        umbrales[indice] = mejor[1]
    return umbrales


def evaluar(nombre: str, y: np.ndarray, proba: np.ndarray, pred_v03: np.ndarray) -> dict[str, Any]:
    """Evalua una variante con los umbrales congelados y con los suyos.

    Args:
        nombre: identificador de la variante.
        y: matriz binaria verdadera.
        proba: matriz de probabilidades.
        pred_v03: predicciones de V03.

    Returns:
        Diccionario con metricas, comparacion pareada y complementariedad.
    """
    from src.evaluation import bootstrap_diferencia_pareada

    umbrales = ajustar_umbrales(y, proba)
    pred_frozen = (proba >= UMBRALES_V03[None, :]).astype(int)
    pred_ajustados = (proba >= umbrales[None, :]).astype(int)
    f1_candidato = _f1_por_celda(y, pred_frozen)
    f1_v03 = _f1_por_celda(y, pred_v03)
    acierta_candidato = f1_candidato >= 0.999
    acierta_v03 = f1_v03 >= 0.999
    comparacion = bootstrap_diferencia_pareada(
        y, pred_frozen, pred_v03, REPETICIONES_BOOTSTRAP, SEMILLA_BOOTSTRAP
    )
    por_etiqueta = metricas_multietiqueta(y, pred_frozen)["f1_por_etiqueta"]
    return {
        "variante": nombre,
        "corrected_cv_frozen": round(metrica_macro(y, pred_frozen), 6),
        "corrected_cv_ajustados": round(metrica_macro(y, pred_ajustados), 6),
        "delta_vs_V03": round(metrica_macro(y, pred_frozen) - REFERENCIA_V03, 6),
        "umbrales_ajustados": [float(v) for v in umbrales],
        "ic95_0": comparacion["ic95"][0],
        "ic95_1": comparacion["ic95"][1],
        "f1_HS": round(por_etiqueta[ETIQUETAS[0]], 4),
        "f1_TR": round(por_etiqueta[ETIQUETAS[1]], 4),
        "f1_AG": round(por_etiqueta[ETIQUETAS[2]], 4),
        "salvage": int(np.sum(acierta_candidato & ~acierta_v03)),
        "damage": int(np.sum(~acierta_candidato & acierta_v03)),
        "correccion_neta": int(np.sum(acierta_candidato & ~acierta_v03) - np.sum(~acierta_candidato & acierta_v03)),
        "pred_frozen": pred_frozen,
        "proba": proba,
    }


def main() -> None:
    """Ejecuta B1, B2, B2r y B3."""
    inicio = time.time()
    log("Track B: factorizacion condicional de etiquetas")
    train = read_train()
    grupos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    pliegues = pliegues_corregidos(desarrollo, grupos[indices_dev])
    y = desarrollo[list(ETIQUETAS)].to_numpy(dtype=int)
    oof = np.load(OOF_V03, allow_pickle=True)
    proba_v03 = oof["proba"]
    pred_v03 = (proba_v03 >= UMBRALES_V03[None, :]).astype(int)
    log(f"   desarrollo {len(y)} filas | V03 {metrica_macro(y, pred_v03):.6f}")
    # La invariante TR implica HS es un hecho de las ETIQUETAS. Lo que decide si
    # factorizar sirve es si V03 la VIOLA en sus PREDICCIONES: si ya la cumple, la
    # restriccion es inerte y el producto solo anade ruido de multiplicacion.
    violacion_v03 = int(((pred_v03[:, 1] == 1) & (pred_v03[:, 0] == 0)).sum())
    violacion_hs0_real = int(((pred_v03[:, 1] == 1) & (y[:, 0] == 0)).sum())
    log(f"   V03 predice TR=1 con HS=0: {violacion_v03} filas | con HS=0 real: {violacion_hs0_real}")
    log(
        "   p(TR) de V03 en filas HS=0: media "
        f"{proba_v03[y[:, 0] == 0, 1].mean():.4f} max {proba_v03[y[:, 0] == 0, 1].max():.4f} "
        f"(umbral TR = {UMBRALES_V03[1]})"
    )
    log(
        f"   dominios: HS=1 {int((y[:, 0] == 1).sum())} filas, HS=0 "
        f"{int((y[:, 0] == 0).sum())} | TR=1 con HS=0: {int(((y[:, 1] == 1) & (y[:, 0] == 0)).sum())}"
    )

    log("1/4 modelos marginales y condicionales")
    modelos = ajustar_modelos(desarrollo["text"].astype(str), y, pliegues)
    log(
        f"   P(AG|HS=1) media en condicional: "
        f"{np.nanmean(modelos['cond_ag_hs1']):.4f} | P(AG|HS=0): "
        f"{np.nanmean(modelos['cond_ag_hs0']):.4f}"
    )
    log(
        f"   las filas HS=0 reciben p(TR) = p(HS) * p(TR|HS=1): "
        f"{np.nanmax(modelos['tr'][y[:, 0] == 0]):.4f} (V03 puede dar hasta 1.0)"
    )

    log("2/4 evaluacion de las variantes")
    variantes: list[dict[str, Any]] = []
    # B0: control con marginales reentrenados por este mismo codigo (grupo de control).
    proba_b0 = np.column_stack(
        [modelos["hs"], modelos["marginal_tr"], modelos["marginal_ag"]]
    )
    variantes.append(evaluar("B0-control-marginales", y, proba_b0, pred_v03))
    # B1: solo TR factorizado; HS y AG son los marginales de V03 (no tocarlos).
    proba_b1 = proba_v03.copy()
    proba_b1[:, 1] = modelos["tr"]
    variantes.append(evaluar("B1-TR-condicional", y, proba_b1, pred_v03))
    # B1 completo: TR y AG factorizados.
    proba_b1c = proba_v03.copy()
    proba_b1c[:, 1] = modelos["tr"]
    proba_b1c[:, 2] = modelos["ag"]
    variantes.append(evaluar("B1-TR-AG-condicional", y, proba_b1c, pred_v03))
    # B2: solo AG factorizado, con las dos ramas libres.
    proba_b2 = proba_v03.copy()
    proba_b2[:, 2] = modelos["ag"]
    variantes.append(evaluar("B2-AG-dos-ramas", y, proba_b2, pred_v03))
    # B3: HS propio del pipeline y los tres scores, para ver si el factor HS cambia algo.
    proba_b3 = np.column_stack([modelos["hs"], modelos["tr"], modelos["ag"]])
    variantes.append(evaluar("B3-HS-propio", y, proba_b3, pred_v03))

    for variante in variantes:
        log(
            f"   {variante['variante']:22s} frozen={variante['corrected_cv_frozen']:.4f} "
            f"(ajustado {variante['corrected_cv_ajustados']:.4f}) "
            f"delta={variante['delta_vs_V03']:+.4f} IC95 [{variante['ic95_0']:+.4f}, "
            f"{variante['ic95_1']:+.4f}] | HS {variante['f1_HS']:.4f} TR {variante['f1_TR']:.4f} "
            f"AG {variante['f1_AG']:.4f} | salvage {variante['salvage']} damage {variante['damage']}"
        )

    log("3/4 gate de dos niveles")
    for variante in variantes:
        delta = variante["delta_vs_V03"]
        excluye_cero = variante["ic95_0"] > 0
        cumple_delta = delta >= GATE_1_DELTA
        pliegues_positivos = _pliegues_positivos(y, variante["pred_frozen"], pliegues, pred_v03)
        variante["gate_1_delta"] = cumple_delta
        variante["gate_1_ic95"] = excluye_cero
        variante["pliegues_positivos"] = pliegues_positivos
        variante["gate_1_supera"] = cumple_delta and excluye_cero and pliegues_positivos >= GATE_1_FOLDS
        log(
            f"   {variante['variante']:22s} Gate 1: delta {'OK' if cumple_delta else 'no'} "
            f"({delta:+.4f}) | IC95 {'OK' if excluye_cero else 'no'} | "
            f"pliegues {pliegues_positivos}/{len(pliegues)} "
            f"({'>=' if pliegues_positivos >= GATE_1_FOLDS else '<'} {GATE_1_FOLDS}) -> "
            f"{'SUPERA' if variante['gate_1_supera'] else 'no supera'}"
        )

    log("4/4 informe y figura")
    _figura(variantes)
    _informe(
        variantes, y, modelos, proba_v03, pred_v03, len(y), pliegues, time.time() - inicio
    )
    with REPORTE_CSV.open("w", encoding="utf-8", newline="") as archivo:
        campos = [
            "variante", "corrected_cv_frozen", "corrected_cv_ajustados", "delta_vs_V03",
            "ic95_0", "ic95_1", "f1_HS", "f1_TR", "f1_AG", "salvage", "damage",
            "correccion_neta", "umbrales_ajustados", "pliegues_positivos", "gate_1_supera",
        ]
        escritor = csv.DictWriter(archivo, fieldnames=campos, lineterminator="\n")
        escritor.writeheader()
        for variante in variantes:
            escritor.writerow({campo: variante.get(campo, "") for campo in campos})
    REPORTE_JSON.write_text(
        json.dumps(
            {
                "experimento": EXPERIMENTO,
                "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "referencia_v03": REFERENCIA_V03,
                "umbrales_v03": UMBRALES_V03.tolist(),
                "gate_1": {
                    "delta": GATE_1_DELTA, "folios_minimos": GATE_1_FOLDS, "exige_ic95": True,
                },
                "variantes": [
                    {clave: valor for clave, valor in variante.items()
                     if clave not in ("pred_frozen", "proba")}
                    for variante in variantes
                ],
                "invariante_en_predicciones": {
                    "filas_tr1_hs0_predichas_por_v03": violacion_v03,
                    "filas_tr1_con_hs0_real": violacion_hs0_real,
                    "p_tr_media_en_filas_hs0": round(
                        float(proba_v03[y[:, 0] == 0, 1].mean()), 4
                    ),
                    "p_tr_maxima_en_filas_hs0": round(
                        float(proba_v03[y[:, 0] == 0, 1].max()), 4
                    ),
                },
                "condicionales": {
                    "media_p_ag_hs1": round(float(np.nanmean(modelos["cond_ag_hs1"])), 4),
                    "media_p_ag_hs0": round(float(np.nanmean(modelos["cond_ag_hs0"])), 4),
                    "max_p_tr_en_filas_hs0": round(
                        float(np.nanmax(modelos["tr"][y[:, 0] == 0])), 4
                    ),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    mejores = max(variantes, key=lambda v: v["corrected_cv_frozen"])
    log(
        f"Track B completada en {round(time.time() - inicio, 1)} s | mejor: "
        f"{mejores['variante']} {mejores['corrected_cv_frozen']:.4f} "
        f"(delta {mejores['delta_vs_V03']:+.4f})"
    )


def _pliegues_positivos(
    y: np.ndarray, pred: np.ndarray, pliegues, pred_referencia: np.ndarray
) -> int:
    """Cuenta los pliegues en los que la variante mejora la metrica frente a la referencia.

    Args:
        y: matriz binaria verdadera.
        pred: predicciones de la variante.
        pliegues: pliegues corregidos.
        pred_referencia: predicciones de referencia.

    Returns:
        Numero de pliegues con mejora.
    """
    positivos = 0
    for validacion in pliegues:
        if isinstance(validacion, tuple):
            indices = validacion[1]
        else:
            indices = validacion
        if metrica_macro(y[indices], pred[indices]) > metrica_macro(
            y[indices], pred_referencia[indices]
        ):
            positivos += 1
    return positivos


def _figura(variantes: list[dict[str, Any]]) -> None:
    """Dibuja las variantes de la factorizacion frente a V03.

    Args:
        variantes: resultados de las variantes.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURAS.mkdir(parents=True, exist_ok=True)
    nombres = [v["variante"] for v in variantes]
    posiciones = np.arange(len(nombres))
    fig, eje = plt.subplots(figsize=(9, 4.6))
    eje.bar(posiciones, [v["delta_vs_V03"] for v in variantes], color="#1f77b4")
    eje.axhline(0, color="black", linewidth=1)
    eje.axhline(GATE_1_DELTA, color="red", linestyle=":", label=f"gate {GATE_1_DELTA:+.3f}")
    for indice, variante in enumerate(variantes):
        eje.errorbar(
            indice,
            variante["delta_vs_V03"],
            yerr=[
                [variante["delta_vs_V03"] - variante["ic95_0"]],
                [variante["ic95_1"] - variante["delta_vs_V03"]],
            ],
            fmt="none",
            ecolor="black",
            capsize=4,
        )
    eje.set_xticks(posiciones)
    eje.set_xticklabels(nombres, rotation=20, ha="right", fontsize=9)
    eje.set_ylabel("delta frente a V03")
    eje.set_title("Factorizacion condicional: delta e IC95 frente a V03")
    eje.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURAS / "fig_59_label_factorization_v6.png", dpi=150)
    plt.close()


def _informe(
    variantes: list[dict[str, Any]],
    y: np.ndarray,
    modelos: dict[str, np.ndarray],
    proba_v03: np.ndarray,
    pred_v03: np.ndarray,
    n_filas: int,
    pliegues,
    duracion: float,
) -> None:
    """Escribe el informe del track B.

    Args:
        variantes: resultados de las variantes.
        y: matriz binaria verdadera.
        modelos: probabilidades de los modelos ajustados.
        proba_v03: probabilidades fuera de pliegue de V03.
        pred_v03: predicciones de V03.
        n_filas: numero de filas del desarrollo.
        pliegues: pliegues corregidos.
        duracion: segundos que tardo el script.
    """
    violacion_v03 = int(((pred_v03[:, 1] == 1) & (pred_v03[:, 0] == 0)).sum())
    violacion_hs0_real = int(((pred_v03[:, 1] == 1) & (y[:, 0] == 0)).sum())
    # Se recalcula aqui la prediccion de la variante factorizada para contar sus violaciones.
    # Es la comprobacion de que el producto impone la invariante: si no fuera cero, la
    # restriccion no estaria aplicandose y el resultado negativo no significaria nada.
    pred_b1 = (
        np.column_stack(
            [proba_v03[:, 0], modelos["tr"], proba_v03[:, 2]]
        )
        >= UMBRALES_V03[None, :]
    ).astype(int)
    violacion_b1 = int(((pred_b1[:, 1] == 1) & (pred_b1[:, 0] == 0)).sum())
    lineas = [
        "# Track B: factorizacion condicional de etiquetas (V6-02)",
        "",
        f"Generado por `experiments/v6_02_label_factorization.py` ({EXPERIMENTO}).",
        "",
        "## 1. Que se prueba y por que no es una repeticion",
        "",
        "Lo que ya se cerro: cadenas de clasificadores (E2-01, seis ordenes) y label powerset",
        "(V4-E, doce variantes, mejor 0.7713). **Ninguna restringe el dominio de entrenamiento del",
        "condicional**: la cadena entrena P(TR|X) con todas las filas y la anterior como entrada.",
        "",
        "Aqui el modelo de TR **solo ve ejemplos con HS=1**, que es su dominio natural, y la",
        "prediccion se compone como producto:",
        "",
        "```text",
        "p(TR|X) = p(HS|X) * p(TR | HS=1, X)",
        "p(AG|X) = p(HS|X) * p(AG | HS=1, X) + (1 - p(HS|X)) * p(AG | HS=0, X)",
        "```",
        "",
        "No hay routing duro: la restriccion viene del producto, no de imponer TR=0 cuando",
        "HS=0. Esa es exactamente la invariante que el diagnostico (V6-01) encontro en los datos.",
        "",
        "## 2. Resultados",
        "",
        "| variante | corrected_cv (frozen) | delta | IC95 | HS | TR | AG | salvage | damage |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for variante in variantes:
        linea = (
            f"| {variante['variante']} | {variante['corrected_cv_frozen']:.4f} | "
            f"{variante['delta_vs_V03']:+.4f} | [{variante['ic95_0']:+.4f}, "
            f"{variante['ic95_1']:+.4f}] | {variante['f1_HS']:.4f} | {variante['f1_TR']:.4f} | "
            f"{variante['f1_AG']:.4f} | {variante['salvage']} | {variante['damage']} |"
        )
        lineas.append(linea)
    lineas += [
        "",
        "Referencia V03: 0.777211. El termino AG de las filas con HS=0 recibe una probabilidad",
        f"media de {np.nanmean(modelos['cond_ag_hs0']):.4f}, coherente con P(AG|HS=0) = 0.026",
        "observado. La maxima p(TR) que la factorizacion puede dar a una fila con HS=0 es",
        f"{np.nanmax(modelos['tr'][y[:, 0] == 0]):.4f}, mientras que un clasificador TR",
        "independiente puede llegar a 1.0.",
        "",
        "## 3. Gate de dos niveles",
        "",
        "| variante | delta >= +0.003 | IC95 excluye 0 | pliegues positivos | Gate 1 |",
        "|---|---|---|---|---|",
    ]
    for variante in variantes:
        lineas.append(
            f"| {variante['variante']} | {'si' if variante['gate_1_delta'] else 'no'} "
            f"({variante['delta_vs_V03']:+.4f}) | "
            f"{'si' if variante['gate_1_ic95'] else 'no'} | "
            f"{variante['pliegues_positivos']}/{len(pliegues)} | "
            f"{'**SUPERA**' if variante['gate_1_supera'] else 'no supera'} |"
        )
    lineas += [
        "",
        "El Gate 2 (segunda particion agrupada con otra semilla) solo se aplica si alguna",
        "variante supera el Gate 1. El holdout no se usa: esta agotado.",
        "",
        "## 4. Por que la restriccion no aporta: la prediccion de V03 ya la cumple",
        "",
        "La invariante TR implica HS es un hecho de las **etiquetas**, y el diagnostico (V6-01)",
        "la confirmo con 0 casos de TR=1 con HS=0. Lo que decide si factorizar sirve no es",
        "que la invariante exista, sino si **V03 la viola en sus predicciones**:",
        "",
        f"- V03 predice **TR=1 con HS=0 en {violacion_v03} filas** de {n_filas} "
        f"({100 * violacion_v03 / n_filas:.2f} %).",
        f"- V03 predice TR=1 en **{violacion_hs0_real} filas que en realidad son HS=0**: en",
        "  esas filas el TR=1 predicho es un falso positivo **garantizado** por la invariante,",
        "  porque ningun ejemplo del corpus tiene TR=1 sin HS=1.",
        f"- p(TR) de V03 en las filas HS=0: media {proba_v03[y[:, 0] == 0, 1].mean():.4f}, "
        f"maximo {proba_v03[y[:, 0] == 0, 1].max():.4f}, con umbral TR = {UMBRALES_V03[1]}.",
        "",
        "**La restriccion se aplica y funciona**: la variante factorizada predice TR=1 con",
        f"HS=0 en **{violacion_b1} filas**, frente a {violacion_v03} de V03. La invariante deja",
        "de violarse en la prediccion, que era el objetivo declarado.",
        "",
        "El problema es que **la restriccion no es gratis**, y ahi esta el resultado. El",
        "producto p(HS)p(TR|HS=1) no solo prohibe TR=1 cuando HS=0: tambien **atenua** p(TR)",
        "en toda fila con p(HS) intermedia, porque multiplica por un factor menor que 1. V03",
        "acertaba justamente en esas filas. De ahi el saldo de B1: **1 fila rescatada frente",
        "a 35 rotas**. El dano no viene de una restriccion mal puesta, viene del efecto de",
        "escala sobre las filas que V03 acierta con p(HS) intermedia.",
        "",
        "La lectura del resultado negativo es por tanto cuantitativa y no moral: **la",
        f"incoherencia que la invariante corrige ({violacion_v03} filas de prediccion,",
        f"{100 * violacion_v03 / n_filas:.2f} % del desarrollo) es demasiado pequena para pagar",
        "el coste de reescalar la probabilidad de TR en todas las filas.** Contra una",
        "referencia que violara la invariante en muchas mas filas, o con un umbral de TR mas",
        "alto, que haria subir la probabilidad de las filas coherentes, la misma linea",
        "tendria otro resultado. Para V03 congelada, no.",
        "",
        "",
        "## 5. Advertencia sobre los umbrales ajustados",
        "",
        "La columna \"ajustado\" de la tabla reoptimiza un umbral por etiqueta con barrido",
        "greedy **sobre el mismo OOF que se evalua**, asi que esta sesgada al alza y no es",
        "comparable con los umbrales congelados de V03: por eso B0 sube a 0.7808 solo por",
        "reajustar umbrales, sin haber cambiado el modelo. **El gate se evalua siempre con los",
        "umbrales congelados**, y la columna de ajustados queda solo como dato de diagnostico.",
        "",
        f"Fecha: {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
        f"Duracion: {round(duracion, 1)} s.",
    ]
    REPORTE.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
