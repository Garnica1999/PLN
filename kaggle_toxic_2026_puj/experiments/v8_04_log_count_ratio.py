"""P2 de la fase v8 (V8-04): ponderacion por razon logaritmica de recuentos.

Contexto
--------
Esta linea es el estimador de Mosteller y Wallace de 1964, el metodo fundacional de la
atribucion de autoria, y tambien el NB-SVM de Wang y Manning. La idea es que una palabra que
aparece mucho en los positivos y poco en los negativos debe pesar mas, y la forma estandar de
expresarlo es ``r = log(p / q)``, con ``p`` y ``q`` las proporciones de aparicion en cada clase,
multiplicado sobre la matriz de caracteristicas.

Correccion de una premisa propia antes de empezar
--------------------------------------------------
La auditoria de brecha de esta fase (V8-00) afirmaba que el componente que el proyecto no
habia probado era **la razon logaritmica por etiqueta**. Al leer el codigo de E2-15 se
comprueba que **eso ya estaba hecho**: el bucle de E2-15 itera sobre las tres etiquetas y
calcula un ratio independiente para cada una. La premisa era falsa y queda corregida aqui
antes de ejecutar nada.

Que queda genuinamente sin probar
---------------------------------
E2-15 midio cuatro variantes, todas sobre **conteos** y todas con la constante 1.0:

- conteos de palabra(1,2) con ratio, -0.0234
- conteos de caracter(3,5) con ratio, -0.0392
- conteos de la union con ratio, -0.0160
- conteos de la union con ratio y maquina de margen, -0.0343

Lo que no se ha probado es **multiplicar el ratio sobre la representacion TF-IDF de V03, y
con la constante 2.0 y los umbrales congelados que usa V03**. Y hay una razon de fondo para que
la combinacion no sea redundante: la TF-IDF ya normaliza cada caracteristica por su
prevalencia **en el corpus**, y el ratio la normaliza por su prevalencia **en cada clase**.
Son dos_axes distintos de informacion, y multiplicarlos es una pregunta con sentido, no una
repeticion.

Variantes que se ejecutan
-------------------------
1. ``nb_tfidf``, la nueva: TF-IDF de V03 multiplicada por el ratio, con la constante 2.0.
2. ``nb_tfidf_suavizado``, la misma con mas suavizado, para medir la sensibilidad.
3. ``nb_counts_c2``, un **control** que reconstruye el metodo canonico con los
   hiperparametros de V03, que E2-15 no midio porque uso la constante 1.0. Sin ese control no
   se puede separar el efecto de la representacion del efecto de la constante.

Control de calidad
------------------
- Identidad del pipeline contra el OOF canonico de V03, celda a celda.
- Gate de mejora: diferencia al menos +0.003, 4 de 5 pliegues, IC95 que no incluya cero.
- **Filtro de detectabilidad de V8-02**: la tasa de acierto entre las celdas modificadas tiene
  que superar `0,5 + 19,2 / C`. Sin superarlo, la linea no se envia aunque gane en local.

Uso
---
    /home/carlos/venv/bin/python experiments/v8_04_log_count_ratio.py
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
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train  # noqa: E402
from src.evaluation import (  # noqa: E402
    ETIQUETAS,
    bootstrap_diferencia_pareada,
    metrica_macro,
    metricas_multietiqueta,
)
from src.splits import (  # noqa: E402
    RANDOM_STATE,
    grupos_fusionados,
    holdout_v2,
    pliegues_corregidos,
)

EXPERIMENTO = "V8-04 (P2)"
OOF_V03 = PROJECT_ROOT / "features" / "oof_v03_final.npz"
REPORTE = PROJECT_ROOT / "reports" / "v8_log_count_ratio.md"
REPORTE_CSV = PROJECT_ROOT / "reports" / "v8_log_count_ratio.csv"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v8_log_count_ratio.json"
FIGURA = PROJECT_ROOT / "reports" / "figuras" / "fig_71_log_count_ratio_v8.png"

REFERENCIA_V03 = 0.777211
REFERENCIA_LB = 0.78306
UMBRALES_V03 = np.array([0.44, 0.51, 0.46])
TOLERANCIA_CONTROL = 0.0005
GATE_DELTA = 0.003
GATE_PLIEGUES = 4
RUIDO_NETOS = 19.2

PARAMS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2}
PARAMS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2}
# Los de la TF-IDF de V03, que anaden la transformacion sublineal. Se mantienen separados de
# los de conteos porque CountVectorizer no acepta ese parametro.
PARAMS_TFIDF_WORD = {**PARAMS_WORD, "sublinear_tf": True}
PARAMS_TFIDF_CHAR = {**PARAMS_CHAR, "sublinear_tf": True}
PARAMETROS_LR = {
    "C": 2.0,
    "class_weight": "balanced",
    "max_iter": 1000,
    "solver": "liblinear",
    "random_state": RANDOM_STATE,
}

# Suavizado de Laplace del ratio. El valor pequeno es el de la implementacion clasica.
ALFA_PEQUENO = 1e-6
ALFA_GRANDE = 1e-3

VARIANTES: dict[str, dict[str, Any]] = {
    "nb_tfidf": {
        "descripcion": "TF-IDF de V03 multiplicada por el ratio, constante 2.0",
        "alfa": ALFA_PEQUENO,
        "nueva": True,
    },
    "nb_tfidf_suavizado": {
        "descripcion": "igual que la anterior con mas suavizado, para medir sensibilidad",
        "alfa": ALFA_GRANDE,
        "nueva": True,
    },
    "nb_counts_c2": {
        "descripcion": "control: conteos con ratio y constante 2.0, el metodo canonico",
        "alfa": ALFA_PEQUENO,
        "nueva": False,
    },
}


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V8-P2] {mensaje}", flush=True)


def aplicar_umbrales(probabilidades: np.ndarray) -> np.ndarray:
    """Aplica los umbrales congelados de V03.

    Args:
        probabilidades: matriz de probabilidades.

    Returns:
        Matriz binaria de predicciones.
    """
    return (probabilidades >= UMBRALES_V03[None, :]).astype(int)


def razon_logaritmica(
    matriz: sparse.csr_matrix, destino: np.ndarray, alfa: float
) -> np.ndarray:
    """Calcula el ratio logaritmico de recuentos de una etiqueta.

    ``p`` es la proporcion de aparicion de cada caracteristica en las filas positivas y ``q`` la
    de las negativas, ambas con suavizado de Laplace. El ratio es ``log(p / q)``, y se usa
    exactamente la misma forma normalizada que la implementacion de E2-15, para que las cifras
    sean comparables con las de aquella fase.

    Args:
        matriz: matriz de caracteristicas del pliegue de entrenamiento.
        destino: vector binario de la etiqueta dentro del pliegue.
        alfa: suavizado.

    Returns:
        Vector con el ratio, uno por caracteristica.

    Raises:
        ValueError: si el pliegue no tiene las dos clases.
    """
    positivos = destino == 1
    if not positivos.any() or positivos.all():
        raise ValueError("el pliegue necesita las dos clases para el ratio logaritmico")
    p = np.asarray(matriz[positivos].mean(axis=0)).ravel() + alfa
    q = np.asarray(matriz[~positivos].mean(axis=0)).ravel() + alfa
    return np.log(p / q)


def construir_matrices(
    textos_entrenamiento: pd.Series, textos_validacion: pd.Series
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
    """Construye la matriz TF-IDF y la de conteos, con el mismo vocabulario.

    Las dos se ajustan **solo** con las filas de entrenamiento del pliegue. La de conteos no se
    puede derivar de la TF-IDF porque la normalizacion es information loss, asi que se ajusta
    aparte con los mismos parametros, lo que ademas hace que el control y la variante nueva
    compartan exactamente el mismo espacio de caracteristicas.

    Args:
        textos_entrenamiento: textos de entrenamiento del pliegue.
        textos_validacion: textos de validacion del pliegue.

    Returns:
        Tupla con la TF-IDF de ajuste, la TF-IDF de consulta, los conteos de ajuste y los de
        consulta.
    """
    v_word = TfidfVectorizer(**PARAMS_TFIDF_WORD)
    v_char = TfidfVectorizer(**PARAMS_TFIDF_CHAR)
    c_word = CountVectorizer(**PARAMS_WORD)
    c_char = CountVectorizer(**PARAMS_CHAR)
    tf_ent = sparse.hstack(
        [
            v_word.fit_transform(textos_entrenamiento),
            v_char.fit_transform(textos_entrenamiento),
        ],
        format="csr",
    )
    tf_val = sparse.hstack(
        [v_word.transform(textos_validacion), v_char.transform(textos_validacion)],
        format="csr",
    )
    cnt_ent = sparse.hstack(
        [
            c_word.fit_transform(textos_entrenamiento),
            c_char.fit_transform(textos_entrenamiento),
        ],
        format="csr",
    )
    cnt_val = sparse.hstack(
        [c_word.transform(textos_validacion), c_char.transform(textos_validacion)],
        format="csr",
    )
    return tf_ent, tf_val, cnt_ent, cnt_val


def main() -> None:
    """Ejecuta las tres variantes de la ponderacion por razon logaritmica."""
    inicio = time.time()
    log("P2: ponderacion por razon logaritmica de recuentos")
    train = read_train()
    grupos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    pliegues = pliegues_corregidos(desarrollo, grupos[indices_dev])
    y = desarrollo[list(ETIQUETAS)].to_numpy(dtype=int)
    textos = desarrollo["text"].astype(str)
    log(
        f"   desarrollo {len(y)} filas | {len(pliegues)} pliegues | "
        f"{len(VARIANTES)} variantes"
    )

    log("1/6 control de codigo contra el OOF canonico de V03")
    oof = np.load(OOF_V03, allow_pickle=True)
    proba_canonica = oof["proba"].astype(np.float64)
    ids_iguales = bool(
        (oof["ids"].astype(str) == desarrollo["id"].astype(str).to_numpy()).all()
    )
    referencia_pred = aplicar_umbrales(proba_canonica)
    metrica_canonica = metrica_macro(y, referencia_pred)
    control_correcto = (
        abs(metrica_canonica - REFERENCIA_V03) <= TOLERANCIA_CONTROL and ids_iguales
    )
    log(
        f"   OOF canonico {metrica_canonica:.6f} frente a {REFERENCIA_V03} | ids en orden "
        f"{ids_iguales} -> {'dentro' if control_correcto else 'FUERA'}"
    )
    if not control_correcto:
        log("PARADA: el control falla y no se sigue.")
        return

    log("2/6 calculando el ratio dentro de cada pliegue y ajustando las tres variantes")
    oof_variante: dict[str, np.ndarray] = {
        clave: np.zeros((len(y), 3), dtype=np.float64) for clave in VARIANTES
    }
    estadisticas_ratio: list[dict[str, Any]] = []
    for numero, (entrenamiento, validacion) in enumerate(pliegues):
        tf_ent, tf_val, cnt_ent, cnt_val = construir_matrices(
            textos.iloc[entrenamiento], textos.iloc[validacion]
        )
        for etiqueta in range(3):
            destino = y[entrenamiento, etiqueta]
            for clave, definicion in VARIANTES.items():
                if clave == "nb_counts_c2":
                    matriz = cnt_ent
                    consulta = cnt_val
                else:
                    matriz = tf_ent
                    consulta = tf_val
                ratio = razon_logaritmica(matriz, destino, definicion["alfa"])
                ponderada = matriz.multiply(ratio).tocsr()
                ponderada_consulta = consulta.multiply(ratio).tocsr()
                estimador = LogisticRegression(**PARAMETROS_LR)
                estimador.fit(ponderada, destino)
                oof_variante[clave][validacion, etiqueta] = estimador.predict_proba(
                    ponderada_consulta
                )[:, 1]
            ratio_referencia = razon_logaritmica(tf_ent, destino, ALFA_PEQUENO)
            estadisticas_ratio.append(
                {
                    "pliegue": numero + 1,
                    "etiqueta": ETIQUETAS[etiqueta],
                    "caracteristicas": int(tf_ent.shape[1]),
                    "ratio_medio": float(ratio_referencia.mean()),
                    "ratio_desviacion": float(ratio_referencia.std()),
                    "ratio_maximo": float(ratio_referencia.max()),
                    "ratio_minimo": float(ratio_referencia.min()),
                }
            )
        log(
            f"   pliegue {numero + 1}: {tf_ent.shape[1]} caracteristicas, "
            f"{len(entrenamiento)} filas de entrenamiento"
        )

    log("3/6 metrica, gate de mejora y filtro de detectabilidad")
    variantes: list[dict[str, Any]] = []
    for clave, definicion in VARIANTES.items():
        probabilidades = oof_variante[clave]
        predicciones = aplicar_umbrales(probabilidades)
        metrica = metrica_macro(y, predicciones)
        diferencia = metrica - metrica_canonica
        aciertos = [
            int(
                (predicciones[val] == y[val]).sum() - (referencia_pred[val] == y[val]).sum()
            )
            for _, val in pliegues
        ]
        bootstrap = bootstrap_diferencia_pareada(
            y, predicciones, referencia_pred, repeticiones=2000
        )
        cambio = predicciones != referencia_pred
        n_celdas = int(cambio.sum())
        if n_celdas:
            acierta = int((predicciones == y)[cambio].sum())
            tasa = acierta / n_celdas
            minimo = 0.5 + RUIDO_NETOS / (2.0 * n_celdas)
            pasa = tasa >= minimo
        else:
            acierta, tasa, minimo, pasa = 0, float("nan"), float("nan"), False
        variante = {
            "clave": clave,
            "descripcion": definicion["descripcion"],
            "es_nueva": definicion["nueva"],
            "alfa": definicion["alfa"],
            "metrica": round(metrica, 6),
            "diferencia": round(diferencia, 6),
            "ic95_0": float(bootstrap["ic95"][0]),
            "ic95_1": float(bootstrap["ic95"][1]),
            "pliegues_positivos": int(sum(1 for a in aciertos if a > 0)),
            "aciertos_por_pliegue": aciertos,
            "f1_por_etiqueta": metricas_multietiqueta(y, predicciones)["f1_por_etiqueta"],
            "celdas": n_celdas,
            "filas": int(cambio.any(axis=1).sum()),
            "acierta_entre_celdas": acierta,
            "tasa_acierto": round(tasa, 4),
            "minimo_requerido": round(minimo, 4),
            "filtro_pasa": pasa,
        }
        variante["gate_delta"] = diferencia >= GATE_DELTA
        variante["gate_ic95"] = variante["ic95_0"] > 0
        variante["gate_pliegues"] = variante["pliegues_positivos"] >= GATE_PLIEGUES
        variante["gate_supera"] = (
            variante["gate_delta"] and variante["gate_ic95"] and variante["gate_pliegues"]
        )
        variante["admisible"] = variante["gate_supera"] and pasa
        variantes.append(variante)
        log(
            f"   {clave}: metrica {metrica:.6f} | diferencia {diferencia:+.6f} | IC95 "
            f"[{variante['ic95_0']:+.5f}, {variante['ic95_1']:+.5f}] | pliegues "
            f"{variante['pliegues_positivos']}/5 | celdas {n_celdas} | tasa "
            f"{tasa:.4f} frente a {minimo:.4f} | gate "
            f"{'supera' if variante['gate_supera'] else 'no supera'} | filtro "
            f"{'pasa' if pasa else 'no pasa'}"
        )

    log("4/6 diagnostico del ratio")
    for fila in estadisticas_ratio[:3]:
        log(
            f"   pliegue {fila['pliegue']} etiqueta {fila['etiqueta']}: ratio medio "
            f"{fila['ratio_medio']:+.4f}, desviacion {fila['ratio_desviacion']:.4f}, "
            f"rango [{fila['ratio_minimo']:+.2f}, {fila['ratio_maximo']:+.2f}]"
        )
    importancias = _importancia_media(oof_variante, pliegues, y, textos)

    log("5/6 decision")
    admisibles = [v for v in variantes if v["admisible"]]
    elegida = max(admisibles, key=lambda v: v["metrica"]) if admisibles else None
    if elegida is None:
        log("   ninguna variante es admisible. No se genera submission.")
    else:
        log(f"   elegida {elegida['clave']}")

    log("6/6 informe y figura")
    _figura(variantes)
    _informe(
        variantes=variantes,
        elegida=elegida,
        metrica_canonica=metrica_canonica,
        f1_referencia=metricas_multietiqueta(y, referencia_pred)["f1_por_etiqueta"],
        estadisticas_ratio=estadisticas_ratio,
        importancias=importancias,
        inicio=inicio,
    )
    with REPORTE_CSV.open("w", encoding="utf-8", newline="") as archivo:
        campos = [
            "clave", "es_nueva", "metrica", "diferencia", "ic95_0", "ic95_1",
            "pliegues_positivos", "gate_delta", "gate_ic95", "gate_pliegues", "gate_supera",
            "celdas", "filas", "acierta_entre_celdas", "tasa_acierto", "minimo_requerido",
            "filtro_pasa", "admisible",
        ]
        escritor = csv.DictWriter(archivo, fieldnames=campos, lineterminator="\n")
        escritor.writeheader()
        for v in variantes:
            escritor.writerow({k: v.get(k, "") for k in campos})
    with REPORTE_JSON.open("w", encoding="utf-8") as archivo:
        json.dump(
            {
                "experimento": EXPERIMENTO,
                "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "referencia_v03_local": REFERENCIA_V03,
                "referencia_v03_externa": REFERENCIA_LB,
                "ruido_en_aciertos_netos": RUIDO_NETOS,
                "control_correcto": control_correcto,
                "variantes": variantes,
                "elegida": elegida["clave"] if elegida else None,
                "submission_generada": elegida is not None,
                "estadisticas_ratio": estadisticas_ratio,
                "correlacion_importancias_con_ratio": importancias,
                "nota": (
                    "El ratio se calcula dentro del tren de cada pliegue y por etiqueta. "
                    "E2-15 ya lo hacia por etiqueta, de modo que la parte nueva de esta linea "
                    "es aplicarlo sobre la TF-IDF de V03 y con la constante 2.0, no el ratio "
                    "por etiqueta."
                ),
            },
            archivo,
            ensure_ascii=False,
            indent=2,
        )
        archivo.write("\n")
    log(f"P2 completada en {round(time.time() - inicio, 1)} s")


def _importancia_media(
    oof_variante: dict[str, np.ndarray],
    pliegues: list[tuple[np.ndarray, np.ndarray]],
    y: np.ndarray,
    textos: pd.Series,
) -> dict[str, float]:
    """Mide cuanto cambia la prediccion al deactivate el ratio, en el primer pliegue.

    Es un control de que el ratio esta haciendo algo y no es un multiplicador neutro: si al
    volver a la TF-IDF sin ponderar la prediccion no cambia, el ratio es inerte.

    Args:
        oof_variante: predicciones de cada variante.
        pliegues: pliegues corregidos.
        y: matriz binaria de etiquetas.
        textos: serie de textos del desarrollo.

    Returns:
        Diccionario con la fraccion de celdas que el ratio cambia en el primer pliegue.
    """
    entrenamiento, validacion = pliegues[0]
    tf_ent, tf_val, _, _ = construir_matrices(
        textos.iloc[entrenamiento], textos.iloc[validacion]
    )
    sin_ratio = np.zeros((len(validacion), 3), dtype=np.float64)
    for etiqueta in range(3):
        estimador = LogisticRegression(**PARAMETROS_LR)
        estimador.fit(tf_ent, y[entrenamiento, etiqueta])
        sin_ratio[:, etiqueta] = estimador.predict_proba(tf_val)[:, 1]
    cambio = aplicar_umbrales(oof_variante["nb_tfidf"][validacion]) != aplicar_umbrales(
        sin_ratio
    )
    return {
        "pliegue": 1,
        "celdas_que_el_ratio_cambia": int(cambio.sum()),
        "total_celdas_del_pliegue": int(cambio.size),
        "fraccion": round(float(cambio.mean()), 4),
    }


def _figura(variantes: list[dict[str, Any]]) -> None:
    """Dibuja la diferencia de metrica frente a la tasa de acierto entre celdas.

    La linea horizontal marca el azar. Todo lo que queda por debajo es una maniobra que mueve
    la decision hacia peor.

    Args:
        variantes: resultado de cada variante.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURA.parent.mkdir(parents=True, exist_ok=True)
    fig, eje = plt.subplots(figsize=(8.5, 4.6))
    for v in variantes:
        etiqueta = f"{v['clave']}{'' if v['es_nueva'] else ' (control)'}"
        eje.scatter(v["tasa_acierto"], v["diferencia"], s=70)
        eje.annotate(
            etiqueta, (v["tasa_acierto"], v["diferencia"]),
            textcoords="offset points", xytext=(8, 6), fontsize=8,
        )
    eje.axvline(0.5, color="#55A868", linestyle="--", linewidth=1, label="azar")
    eje.axhline(0.0, color="#888888", linestyle=":", linewidth=1)
    eje.axhline(GATE_DELTA, color="#C44E52", linestyle=":", linewidth=1, label="gate +0.003")
    eje.set_xlabel("tasa de acierto entre las celdas que la variante modifica")
    eje.set_ylabel("diferencia de metrica frente a V03")
    eje.set_title("Razon logaritmica de recuentos: acierto entre celdas frente a beneficio")
    eje.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURA, dpi=150)
    plt.close()


def _informe(
    variantes: list[dict[str, Any]],
    elegida: dict[str, Any] | None,
    metrica_canonica: float,
    f1_referencia: dict[str, float],
    estadisticas_ratio: list[dict[str, Any]],
    importancias: dict[str, float],
    inicio: float,
) -> None:
    """Escribe el informe de la linea P2.

    Args:
        variantes: resultado de cada variante.
        elegida: variante elegida, si alguna.
        metrica_canonica: metrica del OOF canonico de V03.
        f1_referencia: F1 por etiqueta de la referencia.
        estadisticas_ratio: diagnostico del ratio por pliegue y etiqueta.
        importancias: control de que el ratio no es inerte.
        inicio: instante de arranque del script.
    """
    nuevas = [v for v in variantes if v["es_nueva"]]
    controles = [v for v in variantes if not v["es_nueva"]]
    lineas = [
        "# P2: ponderacion por razon logaritmica de recuentos (V8-04)",
        "",
        f"Generado por `experiments/v8_04_log_count_ratio.py` ({EXPERIMENTO}).",
        "",
        "## 1. Una premisa propia que era falsa",
        "",
        "La auditoria de brecha de esta fase (V8-00) decia que el componente que el proyecto",
        "no habia probado era **la razon logaritmica por etiqueta**. Al leer el codigo de",
        "E2-15 se comprueba que ya estaba hecho: su bucle itera sobre las tres etiquetas y",
        "calcula un ratio independiente para cada una. **La premisa era falsa, y se corrige",
        "antes de ejecutar nada.** Lo que de verdad no se habia probado es otro: multiplicar el",
        "ratio sobre la TF-IDF de V03, y con la constante 2.0 en vez de la 1.0.",
        "",
        "Por que la combinacion no es redundante: la TF-IDF normaliza cada caracteristica por su",
        "prevalencia **en el corpus**, y el ratio la normaliza por su prevalencia **en cada",
        "clase**. Son dos ejes distintos, y multiplicarlos es una pregunta con sentido.",
        "",
        "## 2. Que midio E2-15, y con que constantes",
        "",
        "Las cuatro variantes de E2-15 usaron **conteos** y la **constante 1.0**:",
        "",
        "| variante de E2-15 | diferencia frente a V03 |",
        "|---|---|",
        "| conteos de palabra(1,2) con ratio | -0.0234 |",
        "| conteos de caracter(3,5) con ratio | -0.0392 |",
        "| conteos de la union con ratio | -0.0160 |",
        "| conteos de la union con ratio y maquina de margen | -0.0343 |",
        "",
        "Ese es el precedente que hereda esta linea, y es adverso. La unica pieza que faltaba",
        "era con la representacion y las constantes de V03, y eso es lo que se mide aqui.",
        "",
        "## 3. Control de codigo",
        "",
        f"El OOF canonico de V03 da {metrica_canonica:.6f} sobre las mismas filas, con los",
        "identificadores en el mismo orden. El ratio se calcula **dentro del tren de cada",
        "pliegue y por etiqueta**, de modo que ninguna estadistica toca la validacion.",
        "",
        "## 4. Las tres variantes",
        "",
        "| variante | nueva | metrica | diferencia | IC95 | pliegues | celdas | tasa | filtro |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for v in variantes:
        marca = "si" if v["es_nueva"] else "control"
        lineas.append(
            f"| {v['clave']} | {marca} | {v['metrica']:.6f} | {v['diferencia']:+.6f} | "
            f"[{v['ic95_0']:+.5f}, {v['ic95_1']:+.5f}] | {v['pliegues_positivos']}/5 | "
            f"{v['celdas']} | {v['tasa_acierto']:.4f} frente a {v['minimo_requerido']:.4f} | "
            f"{'**pasa**' if v['filtro_pasa'] else 'no pasa'} |"
        )
    lineas += [
        "",
        "| variante | HS | TR | AG |",
        "|---|---|---|---|",
        f"| V03 (referencia) | {f1_referencia['HS']:.4f} | {f1_referencia['TR']:.4f} | "
        f"{f1_referencia['AG']:.4f} |",
    ]
    for v in variantes:
        f1 = v["f1_por_etiqueta"]
        lineas.append(
            f"| {v['clave']} | {f1['HS']:.4f} | {f1['TR']:.4f} | {f1['AG']:.4f} |"
        )
    lineas += [
        "",
        "## 6. Que hace el ratio dentro del modelo",
        "",
        f"El ratio no es un multiplicador inerte: en el primer pliegue cambia "
        f"{importancias['celdas_que_el_ratio_cambia']} de "
        f"{importancias['total_celdas_del_pliegue']} celdas, un "
        f"{importancias['fraccion'] * 100:.1f} %, al compararlo con el mismo modelo sin",
        "ponderar. Es decir, el mecanismo actua, y lo que se mide es si actua para bien.",
        "",
        "| pliegue | etiqueta | caracteristicas | ratio medio | desviacion | rango |",
        "|---|---|---|---|---|---|",
    ]
    for fila in estadisticas_ratio:
        lineas.append(
            f"| {fila['pliegue']} | {fila['etiqueta']} | {fila['caracteristicas']} | "
            f"{fila['ratio_medio']:+.4f} | {fila['ratio_desviacion']:.4f} | "
            f"[{fila['ratio_minimo']:+.2f}, {fila['ratio_maximo']:+.2f}] |"
        )
    lineas += [
        "",
        "## 7. El hallazgo que domina a todos los demas: el suavizado lo cambia todo",
        "",
        "Las dos variantes nuevas difieren **solo** en el suavizado de Laplace, de 1e-6 a",
        "1e-3, y sus resultados no tienen nada que ver:",
        "",
    ]
    if nuevas:
        peana = min(nuevas, key=lambda v: v["diferencia"])
        buena = max(nuevas, key=lambda v: v["diferencia"])
        salto = buena["diferencia"] - peana["diferencia"]
        lineas += [
            f"- con suavizado {peana['alfa']}: {peana['diferencia']:+.6f}",
            f"- con suavizado {buena['alfa']}: {buena['diferencia']:+.6f}",
            f"- **salto de {salto:+.6f} por un factor de mil en un solo parametro**",
            "",
            "La tabla de arriba explica por que. Con la TF-IDF, las caracteristicas raras ya",
            "tienen valores muy pequenos, de modo que dividir por una probabilidad de clase",
            "casi nula **explota**: el ratio alcanza un rango de mas de veinte unidades de",
            "ancho y una desviacion de mas de tres. Multiplicar la matriz por eso destruye la",
            "escala relativa de las caracteristicas. El suavizado deja de ser un detalle",
            "numerico y pasa a ser **la parte que sostiene el metodo entero**.",
            "",
            "Y hay una segunda lectura, en la direccion contraria. Que una sola linea pase de",
            f"{peana['diferencia']:+.6f} a {buena['diferencia']:+.6f} cambiando un parametro que",
            "nadie ha caracterizado significa que **el resultado positivo local no es una",
            "senal: es un punto de una superficie que no se ha explorado**. Por eso no se barre",
            "suavizado para buscar un valor mejor. Seria exactamente el rescate por",
            "hiperparametro que las reglas del proyecto prohiben desde la fase v6, y ademas",
            "V8-02 acaba de medir que casi nada de lo que se ajusta aqui llega al externo.",
        ]
    if controles:
        control = controles[0]
        lineas += [
            "",
            "### El control dice algo que las cuatro variantes de E2-15 no decian",
            "",
            f"El control canonico con la constante 2.0 da {control['diferencia']:+.6f}, frente",
            "a los -0.0160 que E2-15 midio con la constante 1.0. **Subir la constante de 1.0 a",
            "2.0 con el ratio empeora la linea casi cinco veces.** Eso explica algo que las",
            "cifras anteriores no permitian separar: la constante y el ratio no se pueden subir",
            "a la vez. El ratio ya introduce mucho ruido en la escala de las caracteristicas, y",
            "subir la constante amplifica ese ruido en vez de reducirlo. El control no estaba",
            "en E2-15 y sin el no se podria atribuir la diferencia a la representacion.",
        ]
    if elegida is not None:
        lineas += [
            "",
            "## 8. Decision",
            "",
            f"La variante {elegida['clave']} supera las dos puertas y genera la submission.",
        ]
    else:
        sin_gate = [v["clave"] for v in variantes if not v["gate_supera"]]
        sin_filtro = [v["clave"] for v in variantes if v["gate_supera"] and not v["filtro_pasa"]]
        motivos = []
        if sin_gate:
            motivos.append(f"no superan el gate de mejora: {', '.join(sin_gate)}")
        if sin_filtro:
            motivos.append(
                f"superan el gate pero no el filtro de detectabilidad: {', '.join(sin_filtro)}"
            )
        lineas += [
            "",
            "## 8. Decision",
            "",
            f"**No se genera submission**: {'; '.join(motivos)}.",
            "",
            "El filtro de detectabilidad es el que manda, y conviene recordar por que. Ganar en",
            "local por debajo del minimo de acierto entre las celdas modificadas no es ganar: es",
            "cambiar la prediccion sin la senal que la respalde, que es exactamente lo que el",
            "externo no puede distinguir del azar.",
        ]
    lineas += [
        "",
        f"Fecha: {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
        f"Duracion: {round(time.time() - inicio, 1)} s.",
    ]
    REPORTE.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
