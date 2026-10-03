"""V31-01: Fase 1 de STRUCTURAL_TEXT_FEATURES. Bloque URL estructural.

Que responde
------------
La pregunta de la fase, dicha con las palabras de la directiva:

> Existe informacion estructural del tweet que V03 no puede representar adecuadamente mediante
> sus n-grams aditivos, y que pueda incorporarse con pocas features contextuales?

Este script prueba el **primer bloque**, el de URL y dominio, y solo el primero. No se prueban
mencion, hashtag ni enfasis, y no se combinan bloques.

El control
----------
Se reproduce V03 exactamente (word TF-IDF (1,2) + char_wb TF-IDF (3,5), min_df=2,
sublinear_tf, LogisticRegression C=2.0 balanced liblinear, umbrales congelados 0.44/0.51/0.46)
sobre los pliegues canonicos. Si el corrected_cv no reproduce 0.7772114525165721, el script
se para: no se puede comparar contra una base que no se sostiene.

Las features
------------
Solo estructura textual presente en el tweet. No se consulta el contenido del dominio. El
dominio completo **no** se usa como categoria: un dominio con pocas observaciones es una tabla
de lookup, y la directiva lo prohibe. Se usan en cambio longitud del dominio, numero de
subdominios y TLD como categoria de cardinalidad controlada.

Las features estructurales son funciones deterministas del texto: no usan etiquetas ni
estadisticas del corpus, asi que pueden calcularse una vez para todo el desarrollo sin
introducir leakage. El vectorizador TF-IDF si se ajusta dentro de cada pliegue, como en V03.

Uso
---
    /home/carlos/venv/bin/python experiments/v31_01_url_estructural.py
"""

from __future__ import annotations

import json
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train
from src.evaluation import metricas_multietiqueta
from src.splits import grupos_fusionados, pliegues_corregidos
from src.structural_features import columnas, extraer

EXPERIMENTO = "V31-01 (Fase 1 de STRUCTURAL_TEXT_FEATURES: bloque URL estructural)"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v31_01_url_estructural.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "v31_01_url_estructural.md"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"

ETIQUETAS = ("HS", "TR", "AG")
UMBRALES_V03 = {"HS": 0.44, "TR": 0.51, "AG": 0.46}
CORRECTED_CV_V03 = 0.7772114525165721
PARAMS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
PARAMS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2, "sublinear_tf": True}
LR_PARAMS = {"C": 2.0, "class_weight": "balanced", "max_iter": 1000,
             "solver": "liblinear", "random_state": 42}
BLOQUE = "url"


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V31-01] {mensaje}", flush=True)


def memoria_pico_mb() -> float:
    """Resident set maximum del proceso, en megabytes.

    Returns:
        La memoria maxima residente en MB.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def matriz_v03(textos_train: pd.Series, textos_val: pd.Series) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Representacion de V03: word (1,2) + char_wb (3,5), ajustada dentro del pliegue.

    Args:
        textos_train: textos del pliegue de entrenamiento.
        textos_val: textos del pliegue de validacion.

    Returns:
        Tupla con las matrices de entrenamiento y validacion.
    """
    word = TfidfVectorizer(**PARAMS_WORD)
    char = TfidfVectorizer(**PARAMS_CHAR)
    train = sparse.hstack([word.fit_transform(textos_train), char.fit_transform(textos_train)]).tocsr()
    val = sparse.hstack([word.transform(textos_val), char.transform(textos_val)]).tocsr()
    return train, val


def evaluar(
    train_dev: pd.DataFrame,
    pliegues: list[tuple[np.ndarray, np.ndarray]],
    usar_bloque: bool,
) -> dict[str, Any]:
    """Evalua V03 solo o V03 mas el bloque estructural, sobre los pliegues canonicos.

    Args:
        train_dev: DataFrame del conjunto de desarrollo.
        pliegues: pliegues canonicos de V03.
        usar_bloque: si se anade el bloque estructural.

    Returns:
        Diccionario con probabilidades, predicciones y metricas por pliegue.
    """
    y_true = train_dev[list(ETIQUETAS)].to_numpy(dtype=np.int64)
    n = len(train_dev)
    proba = np.zeros((n, 3), dtype=float)
    # Las features estructurales son deterministas del texto: se calculan una vez.
    features = pd.DataFrame(extraer(BLOQUE, train_dev["text"].astype(str).tolist()))
    features = features[columnas(BLOQUE)]
    metricas_pliegues = []
    for numero, (idx_tr, idx_va) in enumerate(pliegues, start=1):
        train, val = matriz_v03(
            train_dev["text"].iloc[idx_tr], train_dev["text"].iloc[idx_va]
        )
        if usar_bloque:
            train = sparse.hstack([train, features.iloc[idx_tr].to_numpy()]).tocsr()
            val = sparse.hstack([val, features.iloc[idx_va].to_numpy()]).tocsr()
        for j, etiqueta in enumerate(ETIQUETAS):
            modelo = LogisticRegression(**LR_PARAMS)
            modelo.fit(train, train_dev[etiqueta].iloc[idx_tr])
            proba[idx_va, j] = modelo.predict_proba(val)[:, 1]
        pred_pliegue = np.column_stack(
            [(proba[idx_va, j] >= UMBRALES_V03[et]).astype(int) for j, et in enumerate(ETIQUETAS)]
        )
        metrica = metricas_multietiqueta(y_true[idx_va], pred_pliegue)
        metrica["pliegue"] = numero
        metricas_pliegues.append(metrica)
        log(f"   pliegue {numero}: F1 media {metrica['f1_media_columnas']:.4f}")
    pred = np.column_stack(
        [(proba[:, j] >= UMBRALES_V03[et]).astype(int) for j, et in enumerate(ETIQUETAS)]
    )
    return {"proba": proba, "pred": pred, "metricas_pliegues": metricas_pliegues,
            "y_true": y_true}


def comparar(y: np.ndarray, pred_base: np.ndarray, pred_nueva: np.ndarray) -> dict[str, Any]:
    """Rescates, danos y net entre dos predicciones.

    Args:
        y: etiquetas verdaderas.
        pred_base: predicciones de V03.
        pred_nueva: predicciones del challenger.

    Returns:
        Diccionario con el desglose.
    """
    cambios = pred_base != pred_nueva
    por_label: dict[str, Any] = {}
    rescates = dam = 0
    for j, et in enumerate(ETIQUETAS):
        bien_base = pred_base[:, j] == y[:, j]
        bien_nueva = pred_nueva[:, j] == y[:, j]
        r = int(np.count_nonzero(~bien_base & bien_nueva))
        d = int(np.count_nonzero(bien_base & ~bien_nueva))
        rescates += r
        dam += d
        por_label[et] = {
            "rescates": r, "danios": d, "neto": r - d,
            "celdas_modificadas": int(cambios[:, j].sum()),
        }
    return {
        "rescates": rescates, "danios": dam, "neto": rescates - dam,
        "celdas_modificadas": int(cambios.sum()),
        "cobertura_celdas": float(cambios.sum() / cambios.size),
        "por_label": por_label,
    }


def main() -> None:
    """Ejecuta la Fase 1 de STRUCTURAL_TEXT_FEATURES.

    Returns:
        None.
    """
    inicio = time.time()
    log(f"{EXPERIMENTO}. NINGUNA submission")

    train = read_train()
    particion = pd.read_csv(PARTICION)
    ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
    desarrollo = train[train["id"].astype(int).isin(ids_dev)].reset_index(drop=True)
    grupos = grupos_fusionados(desarrollo)
    pliegues = pliegues_corregidos(desarrollo, grupos)
    y = desarrollo[list(ETIQUETAS)].to_numpy(dtype=np.int64)
    log(f"   desarrollo {len(desarrollo)} filas | {len(pliegues)} pliegues canonicos")

    log("1/4 control: reproducir V03 exactamente")
    control = evaluar(desarrollo, pliegues, usar_bloque=False)
    m_control = metricas_multietiqueta(y, control["pred"])
    cv_control = float(m_control["f1_media_columnas"])
    log(f"   corrected_cv del control: {cv_control:.10f}")
    if abs(cv_control - CORRECTED_CV_V03) > 1e-9:
        raise SystemExit(
            f"PARADA: el control reproduce {cv_control:.10f} y no "
            f"{CORRECTED_CV_V03:.10f}. No se puede comparar contra una base que no se sostiene."
        )
    log("   control exacto confirmado")

    log("2/4 challenger: V03 + bloque URL estructural")
    challenger = evaluar(desarrollo, pliegues, usar_bloque=True)
    m_ch = metricas_multietiqueta(y, challenger["pred"])
    cv_ch = float(m_ch["f1_media_columnas"])
    log(f"   corrected_cv del challenger: {cv_ch:.10f}")
    log(f"   delta OOF: {cv_ch - cv_control:+.10f}")

    log("3/4 metricas completas y rescates/danos")
    comparacion = comparar(y, control["pred"], challenger["pred"])
    log(f"   rescates {comparacion['rescates']} | danios {comparacion['danios']} | "
        f"neto {comparacion['neto']:+d} | celdas {comparacion['celdas_modificadas']}")
    for et in ETIQUETAS:
        d = comparacion["por_label"][et]
        log(
            f"      {et}: rescates {d['rescates']} danios {d['danios']} "
            f"neto {d['neto']:+d} | celdas {d['celdas_modificadas']}"
        )
    # Train F1 y gap, para saber si el bloque aporta capacidad o solo ruido.
    train_f1 = _train_f1(desarrollo, pliegues, usar_bloque=True)
    log(f"   train F1 del challenger: {train_f1:.4f} | gap train->OOF {train_f1 - cv_ch:+.4f}")

    log("4/4 informe")
    informe = {
        "familia": "STRUCTURAL_TEXT_FEATURES",
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estado": "OOF_EVALUATED",
        "bloque": BLOQUE,
        "fase": "1_URL_ESTRUCTURAL",
        "pregunta": (
            "la estructura de URL y dominio aporta informacion OOF adicional a V03 mediante "
            "features estructurales aditivas"
        ),
        "alcance": (
            "solo el bloque URL. No se prueban mencion, hashtag ni enfasis, y no se combinan bloques"
        ),
        "features": columnas(BLOQUE),
        "n_features": len(columnas(BLOQUE)),
        "control": {
            "corrected_cv": cv_control,
            "reproduce_v03": bool(abs(cv_control - CORRECTED_CV_V03) <= 1e-9),
            "f1_por_etiqueta": {et: float(m_control["f1_por_etiqueta"][et]) for et in ETIQUETAS},
        },
        "challenger": {
            "corrected_cv": cv_ch,
            "delta_oof": cv_ch - cv_control,
            "f1_por_etiqueta": {et: float(m_ch["f1_por_etiqueta"][et]) for et in ETIQUETAS},
            "train_f1": train_f1,
            "gap_train_oof": train_f1 - cv_ch,
            "metricas_por_pliegue": challenger["metricas_pliegues"],
        },
        "comparacion": comparacion,
        "gastos": {
            "submissions": 0,
            "modelos_entrenados": 2,
            "memoria_pico_mb": round(memoria_pico_mb(), 1),
            "segundos": round(time.time() - inicio, 1),
        },
    }
    REPORTE_JSON.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _informe_md(informe)
    log("   informe escrito en reports/v31_01_url_estructural.{json,md}")
    log(f"   memoria pico {memoria_pico_mb():.0f} MB | tiempo {round(time.time() - inicio, 1)} s")
    log("VEREDICTO: Fase 1 evaluada, 0 submissions")


def _train_f1(
    train_dev: pd.DataFrame, pliegues: list[tuple[np.ndarray, np.ndarray]], usar_bloque: bool
) -> float:
    """F1 sobre el entrenamiento, para medir el gap.

    Args:
        train_dev: DataFrame del desarrollo.
        pliegues: pliegues canonicos.
        usar_bloque: si se anade el bloque.

    Returns:
        La media del F1 sobre los pliegues de entrenamiento.
    """
    y = train_dev[list(ETIQUETAS)].to_numpy(dtype=np.int64)
    features = pd.DataFrame(extraer(BLOQUE, train_dev["text"].astype(str).tolist()))
    features = features[columnas(BLOQUE)]
    puntuaciones = []
    for idx_tr, _ in pliegues:
        train, _ = matriz_v03(train_dev["text"].iloc[idx_tr], train_dev["text"].iloc[idx_tr])
        if usar_bloque:
            train = sparse.hstack([train, features.iloc[idx_tr].to_numpy()]).tocsr()
        pred = np.zeros((len(idx_tr), 3), dtype=np.int64)
        for j, etiqueta in enumerate(ETIQUETAS):
            modelo = LogisticRegression(**LR_PARAMS)
            modelo.fit(train, train_dev[etiqueta].iloc[idx_tr])
            proba = modelo.predict_proba(train)[:, 1]
            pred[:, j] = (proba >= UMBRALES_V03[etiqueta]).astype(int)
        puntuaciones.append(metricas_multietiqueta(y[idx_tr], pred)["f1_media_columnas"])
    return float(np.mean(puntuaciones))


def _informe_md(informe: dict[str, Any]) -> None:
    """Escribe el informe legible.

    Args:
        informe: informe con los datos.

    Returns:
        None.
    """
    c = informe["control"]
    ch = informe["challenger"]
    comp = informe["comparacion"]
    g = informe["gastos"]
    lineas = [
        "# V31-01: Fase 1 de STRUCTURAL_TEXT_FEATURES, bloque URL estructural",
        "",
        f"Generado por `experiments/v31_01_url_estructural.py` ({EXPERIMENTO}).",
        "",
        f"**Estado `{informe['estado']}`. Alcance: {informe['alcance']}**",
        "",
        "## 1. El control",
        "",
        f"corrected_cv del control: **{c['corrected_cv']:.10f}**",
        f"Reproduce V03: **{c['reproduce_v03']}**",
        "",
        "| etiqueta | F1 control |",
        "|---|---:|",
    ]
    for et in ETIQUETAS:
        lineas.append(f"| {et} | {c['f1_por_etiqueta'][et]:.4f} |")
    lineas += [
        "",
        "## 2. El challenger: V03 + bloque URL",
        "",
        f"{informe['n_features']} features, todas de estructura textual:",
        "",
    ]
    for f in informe["features"]:
        lineas.append(f"- `{f}`")
    lineas += [
        "",
        "El dominio completo **no** se usa como categoria: un dominio con pocas observaciones es",
        "una tabla de lookup. Se usan longitud del dominio, numero de subdominios y TLD.",
        "",
        "| etiqueta | F1 control | F1 challenger | delta |",
        "|---|---:|---:|---:|",
    ]
    for et in ETIQUETAS:
        lineas.append(
            f"| {et} | {c['f1_por_etiqueta'][et]:.4f} | {ch['f1_por_etiqueta'][et]:.4f} | "
            f"{ch['f1_por_etiqueta'][et] - c['f1_por_etiqueta'][et]:+.4f} |"
        )
    lineas += [
        "",
        (f"**corrected_cv {c['corrected_cv']:.10f} -> {ch['corrected_cv']:.10f} | "
        f"delta {ch['delta_oof']:+.10f}**"),
        "",
        f"train F1 del challenger {ch['train_f1']:.4f} | gap train->OOF {ch['gap_train_oof']:+.4f}",
        "",
        "## 3. Rescates y danos",
        "",
        "| label | rescates | danios | neto | celdas |",
        "|---|---:|---:|---:|---:|",
    ]
    for et in ETIQUETAS:
        d = comp["por_label"][et]
        lineas.append(
            f"| {et} | {d['rescates']} | {d['danios']} | {d['neto']:+d} | {d['celdas_modificadas']} |"
        )
    lineas += [
        "",
        (f"**rescates {comp['rescates']} | danios {comp['danios']} | neto {comp['neto']:+d} | "
        f"celdas {comp['celdas_modificadas']}**"),
        "",
        "## 4. Metricas por pliegue",
        "",
        "| pliegue | F1 media |",
        "|---:|---:|",
    ]
    for m in ch["metricas_por_pliegue"]:
        lineas.append(f"| {m['pliegue']} | {m['f1_media_columnas']:.4f} |")
    lineas += [
        "",
        "## 5. Lo que este script no hace",
        "",
        "- No prueba mencion, hashtag ni enfasis.",
        "- No combina bloques.",
        "- No usa interacciones: eso es el Control B, que solo se ejecuta si este muestra senal.",
        "- No genera submission.",
        "- No lee el test.",
        "",
        (f"Coste: {g['segundos']} s, memoria {g['memoria_pico_mb']:.0f} MB, "
        f"{g['modelos_entrenados']} modelos, {g['submissions']} submissions."),
        "",
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
