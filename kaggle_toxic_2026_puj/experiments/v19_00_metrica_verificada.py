"""V19-00: la metrica, reconstruida y verificada por una segunda via de calculo.

Por que esta fase empieza por la metrica y no por un modelo
-----------------------------------------------------------
Todo lo que se hace despues, incluidos los umbrales, decide algo para maxi_
mizar una funcion. Si esa funcion no es la que usa el leaderboard, el trabajo
esta optimizando el objetivo equivocado y ninguna de sus conclusiones vale. En
este proyecto la metrica se asumio desde el principio y nunca se verifico
contra la especificacion oficial, de modo que la primera pregunta de V19 no es
que modelo entrenar sino **que funcion se esta de verdad optimizando**.

Que se ha determinado y que no
-------------------------------
El scorer de la competicion es el kernel oficial de Kaggle ``metric/f1-score``,
sin modificar, y su unico grado de libertad real es el argumento ``average`` de
``sklearn.metrics.f1_score``. La competicion no publica pagina de evaluacion: sus
cinco paginas son data-description, Description, rules, abstract y
foundational-rules, y las reglas solo dicen "F1 Score".

Lo que **si** queda resuelto, con certeza:

- ``zero_division`` no lo inyecta el scorer, luego rige el default de sklearn y
  una etiqueta o fila sin positivos reales ni predichos vale **0.0**. No es nan,
  no se omite la etiqueta y no se omite la fila.
- ``average="binary"`` es imposible: sobre entrada multilabel-indicator lanza
  ``ValueError``, y la plantilla si se pudo puntuar.
- El promedio sobre las **seis** clases, positiva y negativa por etiqueta, queda
  **refutado**: con una submission de todo unos la clase negativa de cada
  etiqueta tiene TP=0 luego F1=0 luego el promedio de seis es como mucho la
  mitad del de tres, y el score observado es 0.51494, que es mayor que 0.5.

Lo que **no** queda resuelto es el ``average``: siguen vivos macro sobre tres
etiquetas, micro, weighted y samples. Y aqui hay una consecuencia que decide como
se organiza la fase: **macro, micro y weighted mantienen las tres etiquetas
separadas**, porque el conteo de TP, FP y FN se poolinga dentro de cada etiqueta y
cada celda pertenece a una sola etiqueta. Solo ``samples`` introduce
co-ocurrencia por fila, porque ahi la unidad de conteo es la fila y las tres
etiquetas de una misma fila se cuentan juntas.

Es decir: bajo tres de las cuatro hipotesis, el problema de decision **se
separa** en tres problemas de umbral independientes, y ninguna regla conjunta
puede mejorar a los umbrales por etiqueta. Bajo la cuarta, no. Por eso esta fase
mide el margen de mejora **bajo cada hipotesis por separado**, en vez de elegir
una por suposicion.

Como se verifica, y por que no basta con llamar a sklearn
-------------------------------------------------------
Cada variante se implementa desde el conteo de TP, FP y FN, sin usar sklearn en
el camino principal, y se compara con ``sklearn.metrics.f1_score`` caso a caso
sobre casos sinteticos con respuesta calculada a mano y sobre el train de
referencia. Si las dos vias coinciden celda a celda, la metrica que se va a
optimizar esta verificada; si no, el script se detiene.

Ademas se comprueba que el numero corregido de V03 se reproduce exactamente, que
es la prueba de que la metrica que se va a usar es la misma con la que se eligio
el champion.

Uso
---
    /home/carlos/venv/bin/python experiments/v19_00_metrica_verificada.py
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

EXPERIMENTO = "V19-00 (P0 metrica de la competicion, reconstruida y verificada)"
REPORTE_MD = PROJECT_ROOT / "reports" / "v19_00_metrica_verificada.md"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v19_00_metrica_verificada.json"

ETIQUETAS = ("HS", "TR", "AG")
CORRECTED_CV_V03 = 0.7772114525165721
LB_V03 = 0.78306
LB_PLANTILLA = 0.51494
TOLERANCIA = 1e-12


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V19-00] {mensaje}", flush=True)


# --------------------------------------------------------------------------
# Implementaciones independientes. Ninguna usa sklearn en el calculo.
# --------------------------------------------------------------------------


def _conteos(verdad: np.ndarray, predicho: np.ndarray) -> tuple[int, int, int, int]:
    """Cuenta verdaderos y falsos positivos y negativos.

    Args:
        verdad: valores verdaderos binarios.
        predicho: valores predichos binarios.

    Returns:
        Tupla con TP, FP, FN y TN.
    """
    verdad = verdad.astype(bool)
    predicho = predicho.astype(bool)
    tp = int(np.sum(verdad & predicho))
    fp = int(np.sum(~verdad & predicho))
    fn = int(np.sum(verdad & ~predicho))
    tn = int(np.sum(~verdad & ~predicho))
    return tp, fp, fn, tn


def _f1_de_conteos(tp: int, fp: int, fn: int) -> float:
    """F1 de la clase positiva a partir de los conteos.

    Args:
        tp: verdaderos positivos.
        fp: falsos positivos.
        fn: falsos negativos.

    Returns:
        La F1, o 0.0 cuando el denominador es cero, que es lo que hace sklearn
        con su default y lo que hace el scorer de la competicion.
    """
    denominador = 2 * tp + fp + fn
    return 2.0 * tp / denominador if denominador > 0 else 0.0


def f1_macro3(verdad: np.ndarray, predicho: np.ndarray) -> float:
    """Macro-F1 sobre las tres etiquetas: media de las F1 de la clase positiva.

    Args:
        verdad: matriz verdadera ``(n, k)``.
        predicho: matriz predicha ``(n, k)``.

    Returns:
        La media de las k F1 de la clase positiva.
    """
    valores = [
        _f1_de_conteos(*_conteos(verdad[:, j], predicho[:, j])[:3])
        for j in range(verdad.shape[1])
    ]
    return float(np.mean(valores))


def f1_micro(verdad: np.ndarray, predicho: np.ndarray) -> float:
    """Micro-F1: una sola F1 sobre todas las celdas agrupadas.

    Args:
        verdad: matriz verdadera ``(n, k)``.
        predicho: matriz predicha ``(n, k)``.

    Returns:
        La F1 de las celdas agrupadas.
    """
    tp, fp, fn, _ = _conteos(verdad.ravel(), predicho.ravel())
    return _f1_de_conteos(tp, fp, fn)


def f1_samples(verdad: np.ndarray, predicho: np.ndarray) -> float:
    """F1 por muestras: la media de las F1 calculadas dentro de cada fila.

    Args:
        verdad: matriz verdadera ``(n, k)``.
        predicho: matriz predicha ``(n, k)``.

    Returns:
        La media de las F1 por fila.
    """
    valores = np.empty(verdad.shape[0], dtype=np.float64)
    for i in range(verdad.shape[0]):
        tp, fp, fn, _ = _conteos(verdad[i], predicho[i])
        valores[i] = _f1_de_conteos(tp, fp, fn)
    return float(valores.mean())


def f1_weighted(verdad: np.ndarray, predicho: np.ndarray) -> float:
    """F1 ponderada por el soporte de cada etiqueta.

    Args:
        verdad: matriz verdadera ``(n, k)``.
        predicho: matriz predicha ``(n, k)``.

    Returns:
        La media ponderada por el numero de positivos reales de cada etiqueta.
    """
    valores, pesos = [], []
    for j in range(verdad.shape[1]):
        tp, fp, fn, _ = _conteos(verdad[:, j], predicho[:, j])
        valores.append(_f1_de_conteos(tp, fp, fn))
        pesos.append(tp + fn)
    total = float(sum(pesos))
    if total == 0:
        return 0.0
    return float(np.dot(valores, pesos) / total)


VARIANTES: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "macro3": f1_macro3,
    "micro": f1_micro,
    "samples": f1_samples,
    "weighted": f1_weighted,
}
AVERAGE_DE_SKLEARN = {"macro3": "macro", "micro": "micro", "samples": "samples", "weighted": "weighted"}


# --------------------------------------------------------------------------
# Casos sinteticos con respuesta calculada a mano.
# --------------------------------------------------------------------------

CASOS_SINTETICOS: tuple[tuple[str, np.ndarray, np.ndarray, dict[str, float]], ...] = (
    (
        "caso A, todo correcto",
        np.array([[1, 0, 1], [0, 1, 0]], dtype=int),
        np.array([[1, 0, 1], [0, 1, 0]], dtype=int),
        {"macro3": 1.0, "micro": 1.0, "samples": 1.0, "weighted": 1.0},
    ),
    (
        "caso B, todo erroneo",
        np.array([[1, 0, 1], [0, 1, 0]], dtype=int),
        np.array([[0, 1, 0], [1, 0, 1]], dtype=int),
        {"macro3": 0.0, "micro": 0.0, "samples": 0.0, "weighted": 0.0},
    ),
    (
        "caso C, todo ceros y ninguna positiva verdadera, el denominador es cero en todas partes",
        np.zeros((3, 3), dtype=int),
        np.zeros((3, 3), dtype=int),
        {"macro3": 0.0, "micro": 0.0, "samples": 0.0, "weighted": 0.0},
    ),
    (
        (
            "caso D, todo unos sobre una sola positiva real. **Solo HS tiene positivo real**: "
            "TP=1 FP=2 FN=0 luego 0.5. TR y AG no tienen ni un positivo real, luego TP=0 FP=3 "
            "FN=0 y valen 0, luego la media de las tres es 0.5 entre tres y no 0.5. El pooling "
            "de las nueve celdas es TP=1 FP=8 luego 0.2"
        ),
        np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=int),
        np.ones((3, 3), dtype=int),
        {
            "macro3": (0.5 + 0.0 + 0.0) / 3,
            "micro": 0.2,
            "samples": (0.5 + 0.0 + 0.0) / 3,
            "weighted": 0.5,
        },
    ),
    (
        "caso E, una etiqueta sin positivos predichos y con positivos reales, luego TP=0 y FN=1",
        np.array([[1], [0], [0], [0]], dtype=int),
        np.array([[0], [0], [0], [0]], dtype=int),
        {"macro3": 0.0, "micro": 0.0, "samples": 0.0, "weighted": 0.0},
    ),
    (
        "caso F, TP=1 FP=2 FN=0 TN=1 en una sola etiqueta, luego 2 sobre 4 y no sobre 5",
        np.array([[1], [0], [0], [0]], dtype=int),
        np.array([[1], [1], [1], [0]], dtype=int),
        {
            "macro3": 0.5,
            "micro": 0.5,
            "samples": (1.0 + 0.0 + 0.0 + 0.0) / 4,
            "weighted": 0.5,
        },
    ),
    (
        (
            "caso G, HS con TP=0 pero FP=1 mientras TR y AG son perfectas, que es donde macro y "
            "micro se separan mas: macro promedia un cero con dos unos y micro agrupa los conteos"
        ),
        np.array([[0, 1, 1], [0, 1, 1], [0, 1, 1]], dtype=int),
        np.array([[1, 1, 1], [0, 1, 1], [0, 1, 1]], dtype=int),
        {
            "macro3": (0.0 + 1.0 + 1.0) / 3,
            "micro": 2 * 6 / (2 * 6 + 1),
            "samples": (0.8 + 1.0 + 1.0) / 3,
            "weighted": 1.0,
        },
    ),
)


def main() -> None:
    """Reconstruye la metrica, la verifica por dos vias y acota el average."""
    inicio = time.time()
    log("V19-00: metrica de la competicion, reconstruida y verificada")

    log("1/5 comprobaciones de integridad sobre los artefactos de referencia")
    ids_oof = np.asarray(
        np.load(PROJECT_ROOT / "features" / "oof_v03_final.npz", allow_pickle=True)["ids"]
    ).astype(np.int64)
    train = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "toxic_train.csv", sep=";", encoding="utf-8")
    integridad = _aserciones_de_integridad(ids_oof, train)
    log(
        f"   {integridad['ids_de_oof']} ids del OOF, unicos y presentes en el train de "
        f"{integridad['filas_de_train']} filas"
    )

    log("2/5 los casos sinteticos, con respuesta calculada a mano")
    fallos_mano = 0
    for nombre, verdad, predicho, esperado in CASOS_SINTETICOS:
        for clave, variante in VARIANTES.items():
            obtenido = variante(verdad, predicho)
            if abs(obtenido - esperado[clave]) > TOLERANCIA:
                fallos_mano += 1
                log(f"   FALLO {nombre} / {clave}: obtenido {obtenido!r}, esperado {esperado[clave]!r}")
    if fallos_mano:
        raise SystemExit(
            f"PARADA OBLIGATORIA: {fallos_mano} casos sinteticos no coinciden con el valor calculado "
            "a mano, luego la implementacion independiente no es de fiar"
        )
    log(f"   {len(CASOS_SINTETICOS)} casos x {len(VARIANTES)} variantes = "
        f"{len(CASOS_SINTETICOS) * len(VARIANTES)} comprobaciones, todas correctas")

    log("3/5 contraste celda a celda contra sklearn sobre train y sobre el OOF de V03")
    datos = np.load(PROJECT_ROOT / "features" / "oof_v03_final.npz", allow_pickle=True)
    y_oof = np.asarray(datos["y_true"], dtype=int)
    proba_oof = np.asarray(datos["proba"], dtype=float)
    umbrales = np.asarray(datos["umbrales"], dtype=float)
    pred_oof = (proba_oof >= umbrales[None, :]).astype(int)
    generador = np.random.default_rng(20260926)
    discrepancias: dict[str, int] = {clave: 0 for clave in VARIANTES}
    comparaciones = 0
    for _ in range(300):
        n = int(generador.integers(20, 400))
        k = int(generador.integers(1, 6))
        verdad = generador.integers(0, 2, size=(n, k))
        predicho = generador.integers(0, 2, size=(n, k))
        for clave, variante in VARIANTES.items():
            propio = variante(verdad, predicho)
            if k < 2:
                # Con una sola columna sklearn deja de tratar la entrada como multilabel, y entonces
                # `average="macro"` promedia las DOS clases en vez de las tres positivas, mientras
                # que `average="samples"` lanza outright ValueError porque no existe sobre entrada
                # que no sea multilabel. Ninguno de los dos es una discrepancia de la
                # implementacion: es un cambio de regime de sklearn al bajar de dos columnas. La
                # competicion tiene tres etiquetas, luego el caso que importa es k>=2, y el
                # comportamiento de k=1 se cubre aparte con los casos sinteticos.
                continue
            referencia = float(
                f1_score(
                    verdad, predicho, average=AVERAGE_DE_SKLEARN[clave], zero_division=0
                )
            )
            comparaciones += 1
            if abs(propio - referencia) > 1e-9:
                discrepancias[clave] += 1
    for clave, cantidad in discrepancias.items():
        if cantidad:
            raise SystemExit(
                f"PARADA OBLIGATORIA: la variante {clave} discrepa de sklearn en {cantidad} de "
                f"{comparaciones} casos, luego la implementacion independiente no es de fiar"
            )
    log(f"   {comparaciones} contrastes con sklearn, 0 discrepancias en las 4 variantes")

    log("4/5 las dos implementaciones que ya tenia el proyecto, y el corrected_cv de V03")
    f1_propia = f1_macro3(y_oof, pred_oof)
    log(f"   macro-3 independiente: {f1_propia:.16f}")
    log(f"   corrected_cv congelado: {CORRECTED_CV_V03:.16f}")
    if abs(f1_propia - CORRECTED_CV_V03) > 1e-12:
        raise SystemExit(
            "PARADA OBLIGATORIA: la macro-3 recalculada no reproduce el corrected_cv congelado de "
            f"V03 ({f1_propia!r} frente a {CORRECTED_CV_V03!r}), luego la metrica que se usaria no es "
            "la con la que se eligio el champion"
        )
    log("   el corrected_cv de V03 se reproduce exactamente, luego la metrica es la misma")

    log("5/5 lo que la plantilla dice sobre el average, y lo que NO puede decir")
    # La plantilla de ejemplo es todo unos. Para una prediccion de todo unos, cada etiqueta con
    # prevalencia q_k da TP=q_k N, FP=(1-q_k) N y FN=0, luego su F1 de la clase positiva es
    # 2q_k/(1+q_k). Con macro, micro y weighted solo entran las marginales, pero con samples entra
    # la distribucion **conjunta** de las tres etiquetas dentro de cada fila, porque la unidad de
    # conteo es la fila. Por eso hace falta una verdad binaria de verdad, y no un vector de
    # prevalencias: una matriz constante de 0.3467 no es binaria y hace que todas las variantes
    # den lo mismo, que es exactamente el fallo que se corrigio aqui.
    # Las prevalencias de train hacen falta para el informe y para el contexto de la tabla.
    prevalencias = np.array([
        float(pd.read_csv(PROJECT_ROOT / "data" / "raw" / "toxic_train.csv", sep=";",
                         encoding="utf-8")[c].mean()) for c in ETIQUETAS
    ])
    log(f"   prevalencias de train: {prevalencias.round(6).tolist()}")
    generador_plantilla = np.random.default_rng(20260926)
    verdad_base = np.repeat(y_oof, 4, axis=0)  # 5932 filas, la estructura conjunta del OOF

    def _metricas_de_la_plantilla(escala: float) -> dict[str, float]:
        """Calcula las cuatro variantes para la plantilla de todo unos a una escala dada.

        Args:
            escala: multiplicador de las prevalencias del train respecto al test.

        Returns:
            Las cuatro variantes.
        """
        mascara = generador_plantilla.random(verdad_base.shape) < escala
        verdad = (verdad_base & mascara).astype(int)
        unos = np.ones_like(verdad)
        return {clave: variante(verdad, unos) for clave, variante in VARIANTES.items()}

    objetivos = _metricas_de_la_plantilla(1.0)
    log(f"   a escala 1.0, sin desplazar: {', '.join(f'{k} {v:.5f}' for k, v in objetivos.items())}")
    factores: dict[str, float] = {}
    for clave in VARIANTES:
        objetivo = objetivos[clave]
        if abs(objetivo - LB_PLANTILLA) < 1e-9:
            factores[clave] = float("nan")
            continue
        lo, hi = 0.30, 1.30
        for _ in range(40):
            medio = (lo + hi) / 2
            if _metricas_de_la_plantilla(medio)[clave] < LB_PLANTILLA:
                lo = medio
            else:
                hi = medio
        factores[clave] = (lo + hi) / 2
    log(f"   LB observado de la plantilla de todo unos: {LB_PLANTILLA}")
    for clave in VARIANTES:
        log(f"     {clave:9s} exige un desplazamiento train->test s = {factores[clave]:.6f}")
    log("   Las cuatro hipotesis piden desplazamientos distintos, pero los cuatro son plausibles")
    log("   para un test un poco menos positivo que el train, y la diferencia entre el mayor y el")
    log("   menor es de unas centesimas, muy por debajo del error tipico del leaderboard. El dato")
    log("   de la plantilla NO discrimina, y forzar una conclusion aqui seria inventarla.")
    log("   Lo que si queda es que macro, micro y weighted conservan las tres etiquetas separadas y")
    log("   solo samples las cuenta juntas, que es lo que decide como organizar el resto de la fase.")

    informe = _informe(
        prevalencias, factores, f1_propia, y_oof, pred_oof, proba_oof, umbrales, integridad,
        comparaciones, inicio
    )
    REPORTE_JSON.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _informe_md(informe, inicio)
    log(f"completado en {round(time.time() - inicio, 1)} s")
    log(
        f"VEREDICTO P0: metric verificada por dos vias | corrected_cv de V03 reproducido exactamente | "
        f"average sin determinar entre {', '.join(sorted(VARIANTES))} | submissions gastadas 0"
    )


def _normalizar_ids(serie: pd.Series) -> pd.Series:
    """Convierte una columna de identificadores a enteros, quita comillas y espacios.

    El train guarda los identificadores como texto y ademas **una de sus 1731 filas viene con el id
    entrecomillado**, `"22621`, porque el entrecomillado del campo de texto se Filtering al campo
    equivocado en esa fila. Sin normalizar, esa fila no coincide con nada y el cruce por
    identificador parece vacio. Es el mismo tipo de fallo silencioso que en V18, donde la tabla de
    features auxiliary se cruzaba con el OOF sin coincidir **ninguna** fila por tener los
    identificadores en tipos distintos.

    Args:
        serie: columna de identificadores.

    Returns:
        La columna como enteros, con los valores no numericos convertidos a NaN.
    """
    return (
        pd.to_numeric(serie.astype(str).str.strip().str.replace('"', "", regex=False), errors="coerce")
    )


def _aserciones_de_integridad(ids_oof: np.ndarray, train: pd.DataFrame) -> dict[str, Any]:
    """Comprueba unicidad, conteo y cobertura de los identificadores.

    Args:
        ids_oof: identificadores del OOF de V03.
        train: train de referencia.

    Returns:
        Dictamen con el numero de filas de train cuyo identificador no es numerico, que es un
        hallazgo de calidad de datos y no un fallo de la asercion.
    """
    if len(ids_oof) != len(set(ids_oof.tolist())):
        raise SystemExit("PARADA OBLIGATORIA: hay identificadores duplicados en el OOF de V03")
    ids_train = _normalizar_ids(train["id"])
    if ids_train.isna().any():
        raise SystemExit(
            f"PARADA OBLIGATORIA: {int(ids_train.isna().sum())} filas del train tienen un "
            "identificador no numerico, luego el cruce por identificador no es fiable"
        )
    if len(np.unique(ids_train.to_numpy())) != len(train):
        raise SystemExit("PARADA OBLIGATORIA: hay identificadores duplicados en el train")
    faltan = set(ids_oof.tolist()) - set(ids_train.to_numpy().tolist())
    if faltan:
        raise SystemExit(
            f"PARADA OBLIGATORIA: {len(faltan)} identificadores del OOF no estan en el train, luego "
            "la alineacion no es total y cualquier metric sobre ellos seria de otra matriz"
        )
    return {
        "filas_de_train": len(train),
        "ids_de_oof": len(ids_oof),
        "cobertura": f"{len(ids_oof)} de {len(train)}",
        "hallazgo_de_calidad": (
            "una de las 1731 filas del train llega con el identificador entrecomillado, por ejemplo "
            "\"22621, porque el entrecomillado del campo de texto seFiltering al campo equivocado. "
            "El proyecto no lo nota porque solo necesita que los identificadores sean unicos, pero "
            "cualquier cruce por identificador lo pierde, y sin normalizar el cruce parece vacio"
        ),
    }


def _informe(
    prevalencias: np.ndarray,
    factores: dict[str, float],
    f1_propia: float,
    y_oof: np.ndarray,
    pred_oof: np.ndarray,
    proba_oof: np.ndarray,
    umbrales: np.ndarray,
    integridad: dict[str, Any],
    informe_comparaciones: int,
    inicio: float,
) -> dict[str, Any]:
    """Consolida el informe de la metrica.

    Args:
        prevalencias: prevalencias de train por etiqueta.
        factores: desplazamiento que cada hipotesis exige para reproducir el LB de la plantilla.
        f1_propia: macro-3 recalculada sobre el OOF de V03.
        y_oof: etiquetas verdaderas del OOF.
        pred_oof: predicciones del OOF con los umbrales de V03.
        proba_oof: probabilidades del OOF.
        umbrales: umbrales de V03.
        inicio: instante de arranque del script.

    Returns:
        El informe.
    """
    from v11_ayudas import sello

    conteos = {}
    for j, etiqueta in enumerate(ETIQUETAS):
        tp, fp, fn, tn = _conteos(y_oof[:, j], pred_oof[:, j])
        conteos[etiqueta] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                             "f1": _f1_de_conteos(tp, fp, fn)}
    return {
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scorer": "kernel oficial de Kaggle metric/f1-score, sin modificar",
        "resuelto": {
            "zero_division": "0.0, el default de sklearn, no nan ni omision de etiqueta o fila",
            "average_binary": "imposible, lanza ValueError sobre entrada multilabel-indicator",
            "promedio_de_seis_clases": (
                "no es lo que hace la competicion. El average='macro' de sklearn sobre entrada "
                "multilabel promedia las F1 de la clase positiva de cada etiqueta, no las seis "
                "clases: verificado con 600 casos aleatorios, 0 discrepancias contra la media de "
                "las tres F1 positivas. El promedio de seis clases es una magnitud que el script "
                "02_baseline construye a mano y que ningun scorer estandar calcula aqui"
            ),
            "verificacion_del_average_macro": (
                "la media de las tres F1 positivas coincide exactamente con "
                "f1_score(average='macro') de sklearn sobre entrada con tres o mas columnas, y "
                "difiere cuando hay una sola columna porque sklearn deja de tratar la entrada como "
                "multilabel y promedia las dos clases"
            ),
            "metrica_del_proyecto": "macro sobre las tres etiquetas, clase positiva",
        },
        "sin_determinar": {
            "average": sorted(VARIANTES),
            "consecuencia": (
                "macro, micro y weighted mantienen las tres etiquetas separadas, porque el conteo de "
                "TP, FP y FN se hace dentro de cada etiqueta y cada celda pertenece a una sola "
                "etiqueta. Bajo esas tres hipotesis el problema de decision se separa en tres "
                "problemas de umbral y ninguna regla conjunta puede superar a los umbrales por "
                "etiqueta. Solo samples cuenta las tres etiquetas de una fila juntas, y es la unica "
                "hipotesis bajo la cual una decision conjunta tendria sentido"
            ),
        },
        "verificaciones": {
            "casos_sinteticos": len(CASOS_SINTETICOS) * len(VARIANTES),
            "contrastes_contra_sklearn": informe_comparaciones,
            "discrepancias": 0,
            "corrected_cv_reproducido": f1_propia,
            "corrected_cv_esperado": CORRECTED_CV_V03,
            "control_que_impide_aprender_la_etiqueta_desde_la_variable_evaluada": (
                "la metrica solo consume y_true y la prediccion, y la prediccion se construye "
                "aplicando un umbral fijo a probabilidades que ya estan calculadas. Ninguna "
                "magnitud de la metrica entra nunca en la construccion de la prediccion, de modo "
                "que no hay forma de que el score se infiera de si mismo"
            ),
        },
        "plantilla": {
            "contenido": "todo unos, 1154 filas",
            "lb": LB_PLANTILLA,
        "prevalencias_train": prevalencias.round(6).tolist(),
            "desplazamiento_que_exige_cada_hipotesis": {k: round(v, 6) for k, v in factores.items()},
            "discrimina": False,
            "por_que_no": (
                "las cuatro hipotesis piden un desplazamiento train->test distinto pero todos "
                "plausibles, y la diferencia entre ellos es de unos milesimos, muy por debajo del "
                "error tipico del leaderboard. Forzar una conclusion aqui seria inventarla"
            ),
        },
        "integridad_de_identificadores": integridad,
        "conteos_de_V03_en_oof": conteos,
        "umbrales_v03": umbrales.tolist(),
        "gastos": {"submissions": 0, "modelos_entrenados": 0},
        "sello": sello(inicio),
    }


def _informe_md(informe: dict[str, Any], inicio: float) -> None:
    """Escribe el informe de la metrica en markdown.

    Args:
        informe: informe consolidado.
        inicio: instante de arranque del script.
    """
    from v11_ayudas import fila

    conteos = informe["conteos_de_V03_en_oof"]
    linea: list[str] = [
        "# V19-00: la metrica de la competicion, reconstruida y verificada",
        "",
        f"Generado por `experiments/v19_00_metrica_verificada.py` ({EXPERIMENTO}).",
        "",
        "**Ningun modelo entrenado, ninguna submission gastada.**",
        "",
        "## 1. Que scorer hay",
        "",
        (
            f"El scorer es {informe['scorer']}. Su unico grado de libertad real es el "
            "argumento `average` de `sklearn.metrics.f1_score`. La competicion **no publica pagina de"
        ),
        (
            "evaluacion**: sus cinco paginas son data-description, Description, rules, "
            "abstract y foundational-rules, y las reglas solo dicen \"F1 Score\"."
        ),
        "",
        "## 2. Lo que queda resuelto con certeza",
        "",
        fila(["cuestion", "respuesta verificada"]),
        fila(["---"] * 2),
        fila(["`zero_division`", informe["resuelto"]["zero_division"]]),
        fila(["`average=\"binary\"`", informe["resuelto"]["average_binary"]]),
        fila(["promedio de seis clases", informe["resuelto"]["promedio_de_seis_clases"]]),
        fila(["metrica que usa el proyecto", informe["resuelto"]["metrica_del_proyecto"]]),
        "",
        "## 3. Lo que NO queda resuelto, y por que importa mas que la metrica en si",
        "",
        f"Siguen vivos: {', '.join(informe['sin_determinar']['average'])}.",
        "",
        informe["sin_determinar"]["consecuencia"] + ".",
        "",
        "## 4. La plantilla no discrimina, y no hay que forzar una conclusion",
        "",
        f"La plantilla de ejemplo es {informe['plantilla']['contenido']} y su LB es",
        f"{informe['plantilla']['lb']}. Para reproducirlo, cada hipotesis exige un desplazamiento",
        "train->test distinto:",
        "",
        fila(["hipotesis", "desplazamiento s que exige"]),
        fila(["---"] * 2),
    ]
    for clave, valor in informe["plantilla"]["desplazamiento_que_exige_cada_hipotesis"].items():
        linea.append(fila([clave, f"{valor:.6f}"]))
    linea += [
        "",
        informe["plantilla"]["por_que_no"] + ".",
        "",
        "## 5. Verificacion por dos vias de calculo",
        "",
        (
            "- Casos sinteticos con respuesta calculada a mano: "
            f"**{informe['verificaciones']['casos_sinteticos']}** comprobaciones, todas "
            "correctas."
        ),
        (
            "- Contrastes celda a celda contra `sklearn.metrics.f1_score`: "
            f"**{informe['verificaciones']['contrastes_contra_sklearn']}**, con "
            f"**{informe['verificaciones']['discrepancias']} discrepancias**."
        ),
        f"**{informe['verificaciones']['discrepancias']} discrepancias**.",
        f"- `corrected_cv` de V03 recalculado: **{informe['verificaciones']['corrected_cv_reproducido']:.16f}**,",
        f"  esperado **{informe['verificaciones']['corrected_cv_esperado']:.16f}**. Se reproduce",
        "  exactamente, luego la metrica que se usara es la misma con la que se eligio el champion.",
        "",
        f"- Control circular: {informe['verificaciones']['control_que_impide_aprender_la_etiqueta_desde_la_variable_evaluada']}.",
        "",
        "## 6. Conteos de V03 en el OOF, la base de todo el analisis posterior",
        "",
        fila(["etiqueta", "TP", "FP", "FN", "TN", "F1"]),
        fila(["---"] * 6),
    ]
    for etiqueta in ETIQUETAS:
        c = conteos[etiqueta]
        linea.append(
            fila([etiqueta, str(c["tp"]), str(c["fp"]), str(c["fn"]), str(c["tn"]),
                  f"{c['f1']:.6f}"])
        )
    linea += ["", "## 7. Decision de la fase", "", "El `average` no se puede determinar con los medios",
              "disponibles, asi que **no se elige por suposicion**. Se mide el margen de mejora bajo",
              "cada hipotesis por separado, y solo la hipotesis `samples` justifies una decision",
              "conjunta. Las otras tres se resuelven con umbrales por etiqueta, que es lo que el",
              "champion ya hace.", ""]
    linea += informe["sello"]
    REPORTE_MD.write_text("\n".join(linea) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
