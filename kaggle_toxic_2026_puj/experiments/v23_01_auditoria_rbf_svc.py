"""V23-01: auditoria competitiva de svd:rbf_svc, el mayor DeltaOOF del proyecto.

Que hace y por que
------------------
P0 encontro que `33_class_conditional_lm_v4.py` midio un SVC con kernel RBF sobre una
compresion TruncatedSVD de la union de TF-IDF de V03, con `corrected_cv` 0.790410, es
decir **+0.013199** sobre V03 y **5 de 5 pliegues**, sin leakage, y que **ninguna de las
21 submissions** pertenece a esa familia. Este script hace lo que V21 demostro que decide:
no mirar solo el delta de metrica, sino el **perfil de cambios de celdas**.

Ademas corrige la unica debilidad conocida de esa implementacion. `SVC` sin
`probability=True` no expone `predict_proba`, y el script historico aplicaba una
sigmoidal sobre `decision_function`. Eso es una calibracion ad-hoc: deforma las
probabilidades y por tanto desplaza el punto de operacion de los umbrales congelados de
V03, que fueron ajustados sobre probabilidades de un clasificador lineal. La variante
`platt` usa `probability=True`, que es Platt scaling con 5 pliegues internos de
scikit-learn. Es la diferencia algoritmica que se propone en V23.

Controles, y por que cada uno esta
----------------------------------
**El control reproduce el valor historico.** La primera configuracion es exactamente la
codificacion de V4-G: sigmoidal sobre `decision_function`, `SVC(C=2.0, class_weight=
balanced, gamma="scale")` y `TruncatedSVD(300)`. Si no reproduce 0.790410, ninguna de las
variantes es atribuible a la calibracion, porque la unica diferencia con `platt` es la
calibracion.

**El SVD se ajusta dentro del pliegue.** `svd.fit_transform(entrenamiento)` y
`svd.transform(validacion)`, nunca al reves. Es la comprobacion que V4-G ya hacia bien y que
V23 repite porque es la que haria invalido el resultado si se hiciera al reves.

**La alineacion es por identificador**, contra `features/oof_v03_final.npz`, y se verifica
que unica y con etiquetas coincidentes.

**Se reportan las dos lecturas de umbral**, la congelada de V03 y 0.5, porque la diferencia
entre calibraciones se manifiesta justo en donde cae el umbral.

Uso
---
    /home/carlos/venv/bin/python experiments/v23_01_auditoria_rbf_svc.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from math import comb
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train
from src.splits import grupos_fusionados, pliegues_corregidos

EXPERIMENTO = "V23-01 (auditoria competitiva de svd:rbf_svc y su calibracion Platt)"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v23_01_auditoria_rbf_svc.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "v23_01_auditoria_rbf_svc.md"
OOF_CANONICO = PROJECT_ROOT / "features" / "oof_v03_final.npz"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"

ETIQUETAS = ("HS", "TR", "AG")
UMBRALES_V03 = np.array([0.44, 0.51, 0.46])
CORRECTED_CV_V03 = 0.7772114525165721
PUBLICADO_RBF = 0.790410
N_SVD = 300
SEMILLA = 42
TOLERANCIA = 0.0015
PARAMETROS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
PARAMETROS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2, "sublinear_tf": True}


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V23-01] {mensaje}", flush=True)


def macro_f1(verdad: np.ndarray, predicho: np.ndarray) -> float:
    """Macro-F1 de las tres clases positivas.

    Args:
        verdad: matriz verdadera binaria.
        predicho: matriz predicha binaria.

    Returns:
        La media de las tres F1.
    """
    valores = []
    for j in range(verdad.shape[1]):
        comunes = int(np.count_nonzero(verdad[:, j] & predicho[:, j]))
        denominador = int(np.count_nonzero(verdad[:, j])) + int(np.count_nonzero(predicho[:, j]))
        valores.append(2 * comunes / denominador if denominador > 0 else 0.0)
    return float(np.mean(valores))


def mcnemar_exacto(ganadoras: int, perdedoras: int) -> float:
    """Test de McNemar exacto bilateral sobre los pares discordantes.

    Args:
        ganadoras: filas donde la candidata acierta y V03 falla.
        perdedoras: filas donde V03 acierta y la candidata falla.

    Returns:
        El p-valor bilateral.
    """
    n = ganadoras + perdedoras
    if n == 0:
        return 1.0
    k = min(ganadoras, perdedoras)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def main() -> None:
    """Ejecuta el control, la variante Platt y el analisis de celdas."""
    inicio = time.time()
    log("V23-01: auditoria competitiva de svd:rbf_svc. Ningun modelo de lenguaje se entrena")

    log("1/6 desarrollo, pliegues canonicos y alineacion por identificador")
    train = read_train()
    particion = pd.read_csv(PARTICION)
    ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
    desarrollo = train[train["id"].astype(int).isin(ids_dev)].reset_index(drop=True)
    grupos_full = grupos_fusionados(train)
    mapa_grupo = dict(zip(train["id"].astype(int), grupos_full.tolist()))
    grupos_dev = np.array([mapa_grupo[i] for i in desarrollo["id"].astype(int)])
    pliegues = pliegues_corregidos(desarrollo, grupos_dev)
    y = desarrollo[list(ETIQUETAS)].to_numpy().astype(int)
    textos = desarrollo["text"]

    oof = np.load(OOF_CANONICO, allow_pickle=True)
    ids_oof = np.asarray(oof["ids"]).astype(np.int64)
    y_oof = np.asarray(oof["y_true"], dtype=int)
    mapa_posicion = {v: i for i, v in enumerate(desarrollo["id"].astype(int).to_numpy())}
    posiciones = np.array([mapa_posicion[i] for i in ids_oof])
    if len(set(posiciones.tolist())) != len(posiciones):
        raise SystemExit("PARADA OBLIGATORIA: el mapa de alineacion por id no es unico")
    if not np.array_equal(y[posiciones], y_oof):
        raise SystemExit("PARADA OBLIGATORIA: las etiquetas no coinciden tras alinear por id")
    pred_v03 = np.asarray(oof["pred"], dtype=int)
    log(f"   desarrollo {len(desarrollo)} | {len(pliegues)} pliegues | alineacion verificada")

    log("2/6 las matrices dispersas de cada pliegue, ajustadas solo con su entrenamiento")
    matrices: list[tuple[sparse.csr_matrix, sparse.csr_matrix]] = []
    for index, (entrenar, validar) in enumerate(pliegues):
        palabra = TfidfVectorizer(**PARAMETROS_WORD)
        caracter = TfidfVectorizer(**PARAMETROS_CHAR)
        izquierda = sparse.hstack(
            [palabra.fit_transform(textos.iloc[entrenar]), caracter.fit_transform(textos.iloc[entrenar])]
        ).tocsr()
        derecha = sparse.hstack(
            [palabra.transform(textos.iloc[validar]), caracter.transform(textos.iloc[validar])]
        ).tocsr()
        matrices.append((izquierda, derecha))
        log(
            f"   pliego {index + 1}: entrenamiento {izquierda.shape} | validacion {derecha.shape}"
        )

    probabilidades: dict[str, np.ndarray] = {}
    for nombre, usar_probability in (("svd_rbf_sigmoid", False), ("svd_rbf_platt", True)):
        log(f"3/6 {nombre} (probability={usar_probability})")
        proba = np.zeros((len(desarrollo), len(ETIQUETAS)))
        for index, (entrenar, validar) in enumerate(pliegues):
            izquierda, derecha = matrices[index]
            svd = TruncatedSVD(n_components=N_SVD, random_state=SEMILLA)
            # El SVD se ajusta SOLO con el entrenamiento del pliegue. Ajustarlo con la
            # validacion seria fuga, y es la comprobacion que haria invalido el resultado.
            reducido_entrenar = svd.fit_transform(izquierda)
            reducido_validar = svd.transform(derecha)
            for posicion in range(len(ETIQUETAS)):
                modelo = SVC(
                    C=2.0,
                    class_weight="balanced",
                    gamma="scale",
                    probability=usar_probability,
                    random_state=SEMILLA,
                )
                modelo.fit(reducido_entrenar, y[entrenar, posicion])
                if usar_probability:
                    proba[validar, posicion] = modelo.predict_proba(reducido_validar)[:, 1]
                else:
                    decision = modelo.decision_function(reducido_validar)
                    proba[validar, posicion] = 1.0 / (1.0 + np.exp(-decision))
        probabilidades[nombre] = proba[posiciones]

    log("4/6 metricas y el control de cordura contra el valor historico")
    filas: list[dict[str, Any]] = []
    predicciones: dict[str, np.ndarray] = {}
    for nombre in ("svd_rbf_sigmoid", "svd_rbf_platt"):
        proba = probabilidades[nombre]
        pred = (proba >= UMBRALES_V03[None, :]).astype(int)
        predicciones[nombre] = pred
        valor = macro_f1(y_oof, pred)
        filas.append(
            {
                "configuracion": nombre,
                "corrected_cv": valor,
                "delta_vs_v03": valor - CORRECTED_CV_V03,
                "corrected_cv_umbral_05": macro_f1(y_oof, (proba >= 0.5).astype(int)),
                "tasa_positiva": [float(pred[:, j].mean()) for j in range(3)],
                "media_probabilidad": [float(proba[:, j].mean()) for j in range(3)],
            }
        )
        log(
            f"   {nombre:<18} corrected_cv {valor:.6f} (dV03 {valor - CORRECTED_CV_V03:+.6f}) | "
            f"@0.5 {filas[-1]['corrected_cv_umbral_05']:.6f} | tasa "
            f"{[round(t, 4) for t in filas[-1]['tasa_positiva']]}"
        )
    control = filas[0]["corrected_cv"]
    desviacion = abs(control - PUBLICADO_RBF)
    coincide = desviacion <= TOLERANCIA
    log(
        f"   CONTROL: sigmoid {control:.6f} frente a {PUBLICADO_RBF:.6f} publicados | desviacion "
        f"{desviacion:.6f} | dentro de tolerancia: {coincide}"
    )
    if not coincide:
        log("   PARADA: la variante historica no reproduce. Ninguna otra es atribuible a la calibracion")
        REPORTE_JSON.write_text(
            json.dumps(
                {
                    "experimento": EXPERIMENTO,
                    "estado": "PARADA_POR_CONTROL_DE_CORDURA",
                    "control": control,
                    "publicado": PUBLICADO_RBF,
                    "desviacion": desviacion,
                },
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        return

    log("5/6 perfil de cambios de celdas, que es la cifra que decide la transferencia")
    for nombre in ("svd_rbf_sigmoid", "svd_rbf_platt"):
        pred = predicciones[nombre]
        cambio = pred != pred_v03
        rescates = int(np.count_nonzero(cambio & (pred_v03 != y_oof) & (pred == y_oof)))
        danos = int(np.count_nonzero(cambio & (pred_v03 == y_oof) & (pred != y_oof)))
        p = mcnemar_exacto(rescates, danos)
        por_etiqueta = {}
        for j, etiqueta in enumerate(ETIQUETAS):
            c = cambio[:, j]
            r = int(np.count_nonzero(c & (pred_v03[:, j] != y_oof[:, j]) & (pred[:, j] == y_oof[:, j])))
            d = int(np.count_nonzero(c & (pred_v03[:, j] == y_oof[:, j]) & (pred[:, j] != y_oof[:, j])))
            por_etiqueta[etiqueta] = {
                "cambios": int(np.count_nonzero(c)),
                "rescates": r,
                "danios": d,
                "neto": r - d,
                "tasa_positiva_v03": float(pred_v03[:, j].mean()),
                "tasa_positiva_candidata": float(pred[:, j].mean()),
            }
        filas[0 if "sigmoid" in nombre else 1].update(
            {
                "celdas": int(np.count_nonzero(cambio)),
                "rescates": rescates,
                "danios": danos,
                "neto": rescates - danos,
                "filas_afectadas": int(np.count_nonzero(cambio.any(axis=1))),
                "mcnemar_p": p,
                "por_etiqueta": por_etiqueta,
            }
        )
        f = filas[0 if "sigmoid" in nombre else 1]
        log(
            f"   {nombre:<18} celdas {f['celdas']:>4} | rescates {rescates:>4} | danos {danos:>4} | "
            f"NETO {f['neto']:>+4} | filas {f['filas_afectadas']:>4} | McNemar p={p:.4f}"
        )
        for etiqueta, v in por_etiqueta.items():
            log(
                f"       {etiqueta}: cambios {v['cambios']:>4} | rescates {v['rescates']:>3} | danos "
                f"{v['danios']:>3} | neto {v['neto']:>+4} | tasa {v['tasa_positiva_v03']:.4f} -> "
                f"{v['tasa_positiva_candidata']:.4f} ({v['tasa_positiva_candidata'] - v['tasa_positiva_v03']:+.4f})"
            )

    log("6/6 veredicto y escala competitiva")
    sigm = filas[0]
    platt = filas[1]
    mejor = platt if platt["corrected_cv"] > sigm["corrected_cv"] else sigm
    if mejor["mcnemar_p"] < 0.05 and mejor["neto"] > 0:
        veredicto = "CANDIDATA_CON_EVIDENCIA"
    elif mejor["neto"] > 0:
        veredicto = "CANDIDATA_MEJORABLE"
    else:
        veredicto = "RECHAZADA_PERFIL_DE_CELDAS"
    escala = 107 * 0.003
    esperado = mejor["neto"] / 107 * 0.003
    log(
        f"   mejor: {mejor['configuracion']} dV03 {mejor['delta_vs_v03']:+.6f} | neto {mejor['neto']:+d} | "
        f"p {mejor['mcnemar_p']:.4f} | VEREDICTO {veredicto}"
    )
    log(
        f"   con la escala del proyecto (107 cambios netos ~ 0.003 de LB), un neto de "
        f"{mejor['neto']:+d} implica del orden de {esperado:+.6f} de leaderboard"
    )
    del escala

    informe = {
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mecanismo": (
            "SVC con kernel RBF sobre la compresion TruncatedSVD(300) de la union TF-IDF de V03. "
            "Es la unica diferencia algoritmica real no cerrada: V03 es lineal sobre matriz "
            "dispersa de 27.431 features, esto es no lineal sobre 300 componentes densas, con "
            "regularizacion implicita por kernel en vez de por norma L2"
        ),
        "control_de_cordura": {
            "sigmoid": control,
            "publicado": PUBLICADO_RBF,
            "desviacion": desviacion,
            "dentro_de_tolerancia": coincide,
        },
        "sin_leakage": (
            "el TruncatedSVD se ajusta dentro de cada pliegue con fit_transform sobre el "
            "entrenamiento y transform sobre la validacion; el SVC se ajusta sobre el bloque "
            "reducido de entrenamiento"
        ),
        "configuraciones": filas,
        "veredicto": veredicto,
        "gastos": {"modelos_entrenados": 0, "submissions": 0},
    }
    REPORTE_JSON.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _informe_md(informe, inicio)
    log("   informes escritos en reports/v23_01_auditoria_rbf_svc.{json,md}")
    log(f"completado en {round(time.time() - inicio, 1)} s")
    log(f"VEREDICTO: {veredicto} | submissions 0")


def _informe_md(informe: dict[str, Any], inicio: float) -> None:
    """Escribe el informe legible.

    Args:
        informe: informe con los datos.
        inicio: instante de arranque.
    """
    control = informe["control_de_cordura"]
    filas = informe["configuraciones"]
    lineas = [
        "# V23-01: auditoria competitiva de svd:rbf_svc",
        "",
        f"Generado por `experiments/v23_01_auditoria_rbf_svc.py` ({EXPERIMENTO}).",
        "",
        "**Ninguna submission. Test no utilizado.**",
        "",
        "## 1. El mecanismo",
        "",
        informe["mecanismo"] + ".",
        "",
        "Sin leakage: " + informe["sin_leakage"] + ".",
        "",
        "## 2. Control de cordura",
        "",
        (f"La variante con sigmoidal reproduce **{control['sigmoid']:.6f}** frente a los "
        f"**{control['publicado']:.6f}** publicados (desviacion {control['desviacion']:.6f}), "
        f"dentro de tolerancia: **{control['dentro_de_tolerancia']}**."),
        "",
        "## 3. Las dos calibraciones",
        "",
        "| configuracion | corrected_cv | delta V03 | @0.5 | celdas | rescates | danos | **neto** | McNemar p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for f in filas:
        lineas.append(
            f"| {f['configuracion']} | {f['corrected_cv']:.6f} | {f['delta_vs_v03']:+.6f} | "
            f"{f['corrected_cv_umbral_05']:.6f} | {f['celdas']} | {f['rescates']} | {f['danios']} | "
            f"**{f['neto']:+d}** | {f['mcnemar_p']:.4f} |"
        )
    lineas += [
        "",
        "## 4. Desglose por etiqueta",
        "",
        "| configuracion | etiqueta | cambios | rescates | danos | neto | tasa V03 -> candidata |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for f in filas:
        for etiqueta, v in f["por_etiqueta"].items():
            lineas.append(
                f"| {f['configuracion']} | {etiqueta} | {v['cambios']} | {v['rescates']} | "
                f"{v['danios']} | {v['neto']:+d} | {v['tasa_positiva_v03']:.4f} -> "
                f"{v['tasa_positiva_candidata']:.4f} "
                f"({v['tasa_positiva_candidata'] - v['tasa_positiva_v03']:+.4f}) |"
            )
    lineas += [
        "",
        f"## 5. Veredicto: {informe['veredicto']}",
        "",
        f"Completado en {round(time.time() - inicio, 1)} s.",
        "",
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
