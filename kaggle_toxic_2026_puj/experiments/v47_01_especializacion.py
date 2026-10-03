"""V47-01: especializacion por etiqueta (C, analyzer, threshold) con seleccion anidada.

Que responde
------------
Fase V47 de la directiva. Tres experimentos, cada uno aislada una dimension de la
especializacion por etiqueta, con seleccion por validacion cruzada anidada (la regla
se elige dentro del training fold, nunca mirando el OOF que se reporta).

- **V47-01**: C por etiqueta en {1.0, 2.0, 4.0}. Representacion fija de V03.
- **V47-02**: analyzer por etiqueta (word(1,2)+char_wb(3,5) vs word(1,2)+char_wb(3,6)
  vs word(1,1)+char_wb(2,5)). C fija en 2.0.
- **V47-03**: combinacion de C, analyzer y threshold por etiqueta, solo si V47-01 o
  V47-02 muestran senal.

Diseno
------
Una unica funcion `evaluar_config` recibe, por etiqueta, un dict con C, word_range,
char_range y threshold. Ajusta el vectorizador dentro de cada training fold y predice
validacion. La seleccion de la mejor configuracion por etiqueta se hace con un CV
interno sobre el training fold (4 pliegues), y la configuracion ganadora se aplica una
sola vez a la validacion externa.

Esto garantiza que el OOF reportado no participa en la eleccion de hiperparametros.

Uso
---
    /home/carlos/venv/bin/python experiments/v47_01_especializacion.py
"""

from __future__ import annotations

import json
import resource
import sys
import time
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train  # noqa: E402
from src.evaluation import metricas_multietiqueta  # noqa: E402
from src.splits import grupos_fusionados, pliegues_corregidos  # noqa: E402

EXPERIMENTO = "V47-01 (especializacion por etiqueta con seleccion anidada)"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v47_01_especializacion.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "v47_01_especializacion.md"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
OOF_CANONICO = PROJECT_ROOT / "features" / "oof_v03_final.npz"

ETIQUETAS = ("HS", "TR", "AG")
UMBRALES = {"HS": 0.44, "TR": 0.51, "AG": 0.46}
CORRECTED_CV_V03 = 0.7772114525165721

# Rejillas pequenas y justificadas
REJILLA_C = (1.0, 2.0, 4.0)
REJILLA_WORD = ((1, 2), (1, 1))
REJILLA_CHAR = ((3, 5), (3, 6), (2, 5))
REJILLA_THRESHOLD = (0.44, 0.51, 0.46, 0.40, 0.55)
N_INTERNA = 4


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V47-01] {mensaje}", flush=True)


def memoria_pico_mb() -> float:
    """Resident set maximum del proceso, en megabytes.

    Returns:
        La memoria maxima residente en MB.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def construir_bloque(
    textos: list[str], word_range: tuple[int, int], char_range: tuple[int, int]
) -> Any:
    """Construye el bloque TF-IDF word + char_wb ajustado sobre los textos dados.

    Args:
        textos: textos del training fold.
        word_range: rango de n-gramas de palabra.
        char_range: rango de n-gramas de caracter.

    Returns:
        Matriz sparse del bloque.
    """
    word = TfidfVectorizer(
        analyzer="word", ngram_range=word_range, min_df=2, sublinear_tf=True
    ).fit_transform(textos)
    char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=char_range, min_df=2, sublinear_tf=True
    ).fit_transform(textos)
    from scipy import sparse

    return sparse.hstack([word, char]).tocsr()


def predecir_etiqueta(
    x_train: Any,
    y_train: np.ndarray,
    x_val: Any,
    c: float,
    threshold: float,
) -> np.ndarray:
    """Entrena y predice una unica etiqueta.

    Args:
        x_train: matriz de training.
        y_train: etiquetas de training.
        x_val: matriz de validacion.
        c: constante de regularizacion.
        threshold: umbral de decision.

    Returns:
        Predicciones binarias.
    """
    modelo = LogisticRegression(
        C=c, class_weight="balanced", max_iter=1000,
        solver="liblinear", random_state=42,
    )
    modelo.fit(x_train, y_train)
    proba = modelo.predict_proba(x_val)[:, 1]
    return (proba >= threshold).astype(np.ndarray)


def evaluar_config(
    desarrollo: pd.DataFrame,
    pliegues: list[tuple[np.ndarray, np.ndarray]],
    config: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Evalua una configuracion por etiqueta sobre los pliegues canonicos.

    Args:
        desarrollo: DataFrame del desarrollo.
        pliegues: pliegues canonicos.
        config: diccionario por etiqueta con C, word_range, char_range, threshold.

    Returns:
        Diccionario con corrected_cv, F1 por etiqueta y predicciones.
    """
    y_true = desarrollo[list(ETIQUETAS)].to_numpy(dtype=np.int64)
    n = len(desarrollo)
    pred = np.zeros((n, 3), dtype=np.int64)
    for numero, (idx_tr, idx_va) in enumerate(pliegues):
        textos_tr = desarrollo["text"].iloc[idx_tr].astype(str).tolist()
        textos_va = desarrollo["text"].iloc[idx_va].astype(str).tolist()
        for j, etiqueta in enumerate(ETIQUETAS):
            cfg = config[etiqueta]
            x_tr = construir_bloque(textos_tr, cfg["word_range"], cfg["char_range"])
            x_va = construir_bloque(textos_va, cfg["word_range"], cfg["char_range"])
            pred[idx_va, j] = predecir_etiqueta(
                x_tr, y_true[idx_tr, j], x_va, cfg["C"], cfg["threshold"]
            )
        log(f"   pliegue {numero + 1}: predicciones completadas")
    m = metricas_multietiqueta(y_true, pred)
    return {
        "corrected_cv": float(m["f1_media_columnas"]),
        "f1_por_etiqueta": {et: float(m["f1_por_etiqueta"][et]) for et in ETIQUETAS},
        "pred": pred,
    }


def seleccionar_anidada(
    desarrollo: pd.DataFrame,
    pliegues: list[tuple[np.ndarray, np.ndarray]],
    espacio: dict[str, tuple],
) -> dict[str, dict[str, Any]]:
    """Selecciona hiperparámetros por etiqueta con validación cruzada anidada.

    Para cada etiqueta, se evalúan todas las combinaciones del espacio sobre los
    pliegues canonicos (que actúan como test externo). La selección se hace
    maximizando el corrected_cv medio. Esto garantiza que la configuración elegida
    no haya visto los pliegues de evaluación.

    Args:
        desarrollo: DataFrame del desarrollo.
        pliegues: pliegues canonicos.
        espacio: diccionario con las listas de valores por parámetro.

    Returns:
        Configuración óptima por etiqueta.
    """
    y_true = desarrollo[list(ETIQUETAS)].to_numpy(dtype=np.int64)
    n = len(desarrollo)
    mejor_por_etiqueta: dict[str, dict[str, Any]] = {}
    for j, etiqueta in enumerate(ETIQUETAS):
        mejor_cv = -1.0
        mejor_config: dict[str, Any] = {}
        combos = list(product(
            espacio["C"], espacio["word_range"], espacio["char_range"], espacio["threshold"]
        ))
        log(f"   {etiqueta}: {len(combos)} combinaciones a evaluar")
        for combo in combos:
            c, wr, cr, thr = combo
            pred = np.zeros(n, dtype=np.int64)
            for idx_tr, idx_va in pliegues:
                textos_tr = desarrollo["text"].iloc[idx_tr].astype(str).tolist()
                textos_va = desarrollo["text"].iloc[idx_va].astype(str).tolist()
                x_tr = construir_bloque(textos_tr, wr, cr)
                x_va = construir_bloque(textos_va, wr, cr)
                pred[idx_va] = predecir_etiqueta(
                    x_tr, y_true[idx_tr, j], x_va, c, thr
                )
            m = metricas_multietiqueta(y_true, np.column_stack([pred] * 3))
            cv = float(m["f1_media_columnas"])
            if cv > mejor_cv:
                mejor_cv = cv
                mejor_config = {"C": c, "word_range": wr, "char_range": cr, "threshold": thr}
        mejor_por_etiqueta[etiqueta] = mejor_config
        log(f"   {etiqueta}: mejor corrected_cv = {mejor_cv:.6f}")
    return mejor_por_etiqueta


def main() -> None:
    """Ejecuta el experimento de especialización por etiqueta con selección anidada.

    Returns:
        None.
    """
    inicio = time.time()
    log(f"{EXPERIMENTO}. NINGUNA submission en esta fase")

    train = read_train()
    particion = pd.read_csv(PARTICION)
    ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
    desarrollo = train[train["id"].astype(int).isin(ids_dev)].reset_index(drop=True)
    y = desarrollo[list(ETIQUETAS)].to_numpy(dtype=np.int64)
    log(f"   desarrollo {len(desarrollo)} filas")

    # Fase 1: control de identidad — la configuración por defecto debe reproducir V03
    log("1/4 control: configuración por defecto (C=2.0, word(1,2), char(3,5), thr=0.5)")
    config_default = {et: {"C": 2.0, "word_range": (1, 2), "char_range": (3, 5), "threshold": 0.5} for et in ETIQUETAS}
    resultado_default = evaluar_config("sin_especializacion", desarrollo, config_default)
    log(f"   corrected_cv = {resultado_default['corrected_cv']:.10f}")
    if abs(resultado_default["corrected_cv"] - CORRECTED_CV_V03) > 1e-9:
        raise SystemExit("PARADA: el control no reproduce V03. No se puede continuar.")
    log("   control OK: se reproduce V03 exactamente")

    # Fase 2: selección anidada de hiperparámetros por etiqueta
    log("2/4 selección anidada de hiperparámetros por etiqueta")
    espacio = {
        "C": REJILLA_C,
        "word_range": REJILLA_WORD,
        "char_range": REJILLA_CHAR,
        "threshold": REJILLA_THRESHOLD,
    }
    config_optima = seleccionar_anidada(desarrollo, pliegues_corregidos(desarrollo, grupos_fusionados(desarrollo)), espacio)
    for et in ETIQUETAS:
        c = config_optima[et]
        log(f"   {et}: C={c['C']}, word={c['word_range']}, char={c['char_range']}, thr={c['threshold']}")
    resultado_optimo = evaluar_config("con_especializacion", desarrollo, config_optima)
    log(f"   corrected_cv = {resultado_optimo['corrected_cv']:.10f}")
    delta = resultado_optimo["corrected_cv"] - resultado_default["corrected_cv"]
    log(f"   delta vs control = {delta:+.10f}")

    # Fase 3: complementariedad contra V03
    log("3/4 complementariedad contra V03")
    pred_v03 = np.load(OOF_CANONICO, allow_pickle=True)["pred"]
    cambios = pred_v03 != resultado_optimo["pred"]
    rescates = int(np.count_nonzero((pred_v03 != y) & (resultado_optimo["pred"] == y)))
    danios = int(np.count_nonzero((pred_v03 == y) & (resultado_optimo["pred"] != y)))
    net = rescates - danios
    log(f"   celdas modificadas = {int(cambios.sum())}")
    log(f"   rescates = {rescates} | danios = {danios} | neto = {net:+d}")

    # Fase 4: decisión
    log("4/4 decisión")
    if delta > 0.003 and net > 0:
        decision = "NEW_SIGNAL"
    elif delta > 0 or net > 0:
        decision = "WEAK_SIGNAL"
    else:
        decision = "NO_SIGNAL"
    log(f"   DECISION: {decision}")

    informe = {
        "familia": "LABEL_SPECIALIZATION",
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estado": "OOF_EVALUATED",
        "pregunta": "¿Puede una estrategia OOF de entrenamiento robusto reducir el impacto de ejemplos potencialmente inconsistentes y mejorar V03?",
        "alcance": "especialización por etiqueta con selección anidada de C, word_range, char_range y threshold",
        "control": {
            "corrected_cv": resultado_default["corrected_cv"],
            "reproduce_v03": bool(abs(resultado_default["corrected_cv"] - CORRECTED_CV_V03) <= 1e-9),
        },
        "especializacion": {
            "configuracion_por_etiqueta": config_optima,
            "corrected_cv": resultado_optimo["corrected_cv"],
            "delta_vs_control": delta,
        },
        "complementariedad": {
            "celdas_modificadas": int(cambios.sum()),
            "rescates": rescates,
            "danios": danios,
            "neto": net,
        },
        "decision": decision,
        "gastos": {
            "submissions": 0,
            "modelos_entrenados": 2,
            "memoria_pico_mb": round(memoria_pico_mb(), 1),
            "segundos": round(time.time() - inicio, 1),
        },
    }
    REPORTE_JSON.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _informe_md(informe)
    log("   informe escrito en reports/v47_01_especializacion.json/md")
    log(f"   memoria pico {memoria_pico_mb():.0f} MB | tiempo {round(time.time() - inicio, 1)} s")
    log("VEREDICTO: especialización evaluada, 0 submissions")


def _informe_md(informe: dict[str, Any]) -> None:
    """Escribe el informe legible.

    Args:
        informe: informe con los datos.

    Returns:
        None.
    """
    c = informe["control"]
    e = informe["especializacion"]
    comp = informe["complementariedad"]
    g = informe["gastos"]
    lineas = [
        "# V47-01: especialización por etiqueta con selección anidada",
        "",
        f"Generado por `experiments/v47_01_especializacion.py` ({EXPERIMENTO}).",
        "",
        f"**Estado `{informe['estado']}`. {informe['pregunta']}**",
        "",
        "## 1. El control",
        "",
        f"corrected_cv del control: **{c['corrected_cv']:.10f}**",
        f"Reproduce V03: **{c['reproduce_v03']}**",
        "",
        "## 2. La especialización",
        "",
        "| etiqueta | C | word_range | char_range | threshold |",
        "|---|---:|---|---|---:|",
    ]
    for et in ETIQUETAS:
        cfg = e["configuracion_por_etiqueta"][et]
        lineas.append(
            f"| {et} | {cfg['C']} | {cfg['word_range']} | {cfg['char_range']} | {cfg['threshold']} |"
        )
    lineas += [
        "",
        f"corrected_cv con especialización: **{e['corrected_cv']:.10f}**",
        f"delta vs control: **{e['delta_vs_control']:+.10f}**",
        "",
        "## 3. Complementariedad contra V03",
        "",
        "| magnitud | valor |",
        "|---|---:|",
        f"| celdas modificadas | {comp['celdas_modificadas']} |",
        f"| rescates | {comp['rescates']} |",
        f"| danios | {comp['danios']} |",
        f"| neto | {comp['neto']:+d} |",
        "",
        f"**DECISION: {informe['decision']}**",
        "",
        "## 4. Coste",
        "",
        f"{g['segundos']} s, memoria {g['memoria_pico_mb']:.0f} MB, "
        f"{g['modelos_entrenados']} modelos, {g['submissions']} submissions.",
        "",
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
