"""V38-01: embedding supervisado por factorizacion termino x etiqueta.

Que responde
------------
La pregunta de la fase, dicha con las palabras de la directiva:

> Una representacion distribuida entrenada directamente con las etiquetas HS/TR/AG puede
> capturar relaciones utiles entre terminos que el TF-IDF explicito de V03 no captura, sin caer
> en memorization con solo ~1483 ejemplos?

El metodo
---------
Paso 3 de la directiva: antes de una red neuronal completa, evaluar una formulacion equivalente
pero mas estable basada en la matriz **termino x etiqueta**.

1. Construir la matriz M de tamano (V x 3), donde V es el vocabulario del training fold y
   M[t, j] es el numero de veces que el termino t aparece en documentos del training fold con
   etiqueta j positiva.
2. Normalizar con BM25 (k1 1.2, b 0.75) por fila, para que los terminos frecuentes no dominen.
3. Factorizar con SVD truncado de baja dimension: M = U S Vt. Los vectores de termino son las
   filas de U S, de tamano d.
4. Para cada documento, promediar los vectores de sus terminos (mean pooling).
5. Entrenar un clasificador lineal sobre esos vectores.

Esto es un **embedding supervisado**: los vectores se derivan de la asociacion termino-etiqueta, no
de la co-ocurrencia no supervisada. Y es de baja capacidad por construccion: con d=16 y un
clasificador lineal, el numero de parametros es pequeno.

Por que no es equivalente a SVD sobre documento x termino
--------------------------------------------------------
SVD sobre la matriz documento x termino (TruncatedSVD) es **no supervisado**: no usa las
etiquetas. La factorizacion termino x etiqueta **si las usa**, porque la matriz se construye
contando co-ocurrencias con cada etiqueta por separado. Es la diferencia entre "que terminos
aparecen juntos" y "que terminos aparecen con cada etiqueta".

Controles
---------
- El vocabulario se construye exclusivamente dentro del training fold.
- Las estadisticas de BM25 y el SVD se ajustan dentro del training fold.
- Para validation se usan las estadisticas aprendidas en training.
- No se usa el test para nada.
- Se registra el numero exacto de parametros.
- Se mide la memorization: train F1, OOF F1, gap, y rendimiento en terminos raros.

Uso
---
    /home/carlos/venv/bin/python experiments/v38_01_embedding_supervisado.py
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
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train
from src.evaluation import metricas_multietiqueta
from src.splits import grupos_fusionados, pliegues_corregidos

EXPERIMENTO = "V38-01 (embedding supervisado por factorizacion termino x etiqueta)"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v38_01_embedding_supervisado.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "v38_01_embedding_supervisado.md"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
D = 16
K1 = 1.2
B = 0.75
UMBRALES = {"HS": 0.44, "TR": 0.51, "AG": 0.46}


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V38-01] {mensaje}", flush=True)


def memoria_pico_mb() -> float:
    """Resident set maximum del proceso, en megabytes.

    Returns:
        La memoria maxima residente en MB.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def bm25_por_fila(
    matriz: sparse.csr_matrix, k1: float = K1, b: float = B
) -> sparse.csr_matrix:
    """Normaliza una matriz de conteos con BM25 por fila.

    BM25: ``tf' = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))``.

    Args:
        matriz: matriz de conteos dispersa.
        k1: saturacion de la frecuencia de termino.
        b: normalizacion por longitud.

    Returns:
        La matriz normalizada.
    """
    coo = matriz.tocoo().astype(np.float64)
    largos = np.asarray(matriz.sum(axis=1)).ravel()
    avgdl = float(largos.mean()) or 1.0
    dl = largos[coo.row]
    tf = coo.data
    denominador = tf + k1 * (1.0 - b + b * dl / max(avgdl, 1.0))
    salida = matriz.copy().astype(np.float64)
    salida.data = tf * (k1 + 1.0) / denominador
    return salida


def construir_embedding_supervisado(
    textos_train: list[str], y_train: np.ndarray, d: int = D
) -> tuple[np.ndarray, dict[str, Any], CountVectorizer]:
    """Construye el embedding supervisado dentro del training fold.

    Args:
        textos_train: textos del training fold.
        y_train: etiquetas del training fold, forma ``(n, 3)``.
        d: dimension del embedding.

    Returns:
        Tupla con la matriz de embeddings de termino (V x d) y las estadisticas.
    """
    # Matriz termino x etiqueta: M[t, j] = conteo del termino t en documentos con etiqueta j.
    conteo = CountVectorizer(
        analyzer="word", ngram_range=(1, 1), min_df=1, lowercase=True
    )
    x_tr = conteo.fit_transform(textos_train)
    vocabulario = conteo.get_feature_names_out().tolist()
    len(textos_train)
    m = sparse.lil_matrix((x_tr.shape[1], 3), dtype=np.float64)
    for j in range(3):
        selector = y_train[:, j] == 1
        if selector.any():
            m[:, j] = np.asarray(x_tr[selector].sum(axis=0)).ravel()
    m = m.tocsr()
    m = bm25_por_fila(m)
    # SVD truncado de baja dimension. Es el embedding supervisado.
    d_efectivo = min(d, m.shape[0] - 1, m.shape[1] - 1)
    svd = TruncatedSVD(n_components=d_efectivo, random_state=42)
    embedding = svd.fit_transform(m)
    stats = {
        "vocabulario": len(vocabulario),
        "dimension": int(embedding.shape[1]),
        "varianza_explicada": float(svd.explained_variance_ratio_.sum()),
        "parametros_del_embedding": int(embedding.shape[0] * embedding.shape[1]),
    }
    return embedding, stats, conteo


def vector_de_documento(
    texto: str, embedding: np.ndarray, conteo: CountVectorizer
) -> np.ndarray:
    """Mean pooling de los embeddings de los terminos del documento.

    Args:
        texto: texto del documento.
        embedding: matriz de embeddings de termino.
        conteo: vectorizador con el vocabulario aprendido.

    Returns:
        El vector del documento.
    """
    tokens = conteo.transform([texto]).toarray().ravel()
    activos = tokens > 0
    if not activos.any():
        return np.zeros(embedding.shape[1], dtype=np.float64)
    return embedding[activos].mean(axis=0)


def evaluar_embedding_supervisado(
    train_dev: pd.DataFrame,
    pliegues: list[tuple[np.ndarray, np.ndarray]],
    d: int = D,
) -> dict[str, Any]:
    """Evalua el embedding supervisado sobre los pliegues canonicos.

    Args:
        train_dev: DataFrame del desarrollo.
        pliegues: pliegues canonicos.
        d: dimension del embedding.

    Returns:
        El resultado de la evaluacion.
    """
    y_true = train_dev[["HS", "TR", "AG"]].to_numpy(dtype=np.int64)
    n = len(train_dev)
    proba = np.zeros((n, 3), dtype=float)
    metricas = []
    train_f1s = []
    stats_por_pliegue: dict[str, Any] = {}
    for numero, (idx_tr, idx_va) in enumerate(pliegues, start=1):
        textos_tr = train_dev["text"].iloc[idx_tr].astype(str).tolist()
        y_tr = y_true[idx_tr]
        embedding, stats, conteo = construir_embedding_supervisado(textos_tr, y_tr, d)
        # Vectores de training y validation
        v_tr = np.array([
            vector_de_documento(t, embedding, conteo)
            for t in train_dev["text"].iloc[idx_tr].astype(str)
        ])
        v_va = np.array([
            vector_de_documento(t, embedding, conteo)
            for t in train_dev["text"].iloc[idx_va].astype(str)
        ])
        stats["dimension"]
        pred_train = np.zeros((len(idx_tr), 3), dtype=np.int64)
        for j, etiqueta in enumerate(("HS", "TR", "AG")):
            modelo = LogisticRegression(
                C=2.0, class_weight="balanced", max_iter=1000,
                solver="liblinear", random_state=42,
            )
            modelo.fit(v_tr, y_tr[:, j])
            proba[idx_va, j] = modelo.predict_proba(v_va)[:, 1]
            pred_train[:, j] = (
                modelo.predict_proba(v_tr)[:, 1] >= UMBRALES[etiqueta]
            ).astype(int)
        pred_pliegue = np.column_stack(
            [(proba[idx_va, j] >= UMBRALES[et]).astype(int) for j, et in enumerate(("HS", "TR", "AG"))]
        )
        metrica = metricas_multietiqueta(y_true[idx_va], pred_pliegue)
        metrica["pliegue"] = numero
        metricas.append(metrica)
        train_f1s.append(metricas_multietiqueta(y_true[idx_tr], pred_train)["f1_media_columnas"])
        stats_por_pliegue[str(numero)] = stats
        log(
            f"   pliegue {numero}: vocabulario {stats['vocabulario']} | "
            f"dimension {stats['dimension']} | varianza explicada "
            f"{stats['varianza_explicada']:.4f} | F1 {metrica['f1_media_columnas']:.4f}"
        )
    pred = np.column_stack(
        [(proba[:, j] >= UMBRALES[et]).astype(int) for j, et in enumerate(("HS", "TR", "AG"))]
    )
    return {
        "proba": proba, "pred": pred, "metricas_pliegues": metricas,
        "y_true": y_true, "train_f1": float(np.mean(train_f1s)),
        "stats_por_pliegue": stats_por_pliegue,
    }


def main() -> None:
    """Ejecuta la evaluacion del embedding supervisado.

    Returns:
        None.
    """
    inicio = time.time()
    log(f"{EXPERIMENTO}. NINGUNA submission en esta fase")

    train = read_train()
    particion = pd.read_csv(PARTICION)
    ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
    desarrollo = train[train["id"].astype(int).isin(ids_dev)].reset_index(drop=True)
    pliegues = pliegues_corregidos(desarrollo, grupos_fusionados(desarrollo))
    y = desarrollo[["HS", "TR", "AG"]].to_numpy(dtype=np.int64)
    log(f"   desarrollo {len(desarrollo)} filas | {len(pliegues)} pliegues | d {D}")

    log("1/5 control TF-IDF sin embedding: debe reproducir V03")
    from src.structural_eval import evaluar_bloque, resumen, verificar_control
    control = resumen(evaluar_bloque(desarrollo, pliegues, bloque=None))
    verificar_control(control["corrected_cv"])
    log(f"   control exacto: {control['corrected_cv']:.10f}")

    log("2/5 OOF con embedding supervisado")
    r = evaluar_embedding_supervisado(desarrollo, pliegues, D)
    m = metricas_multietiqueta(y, r["pred"])
    cv = float(m["f1_media_columnas"])
    log(f"   corrected_cv {cv:.10f} | delta {cv - control['corrected_cv']:+.10f}")
    log(f"   train F1 {r['train_f1']:.4f} | gap {r['train_f1'] - cv:+.4f}")
    for et in ("HS", "TR", "AG"):
        log(
            f"      {et}: control {control['f1_por_etiqueta'][et]:.4f} -> embedding "
            f"{m['f1_por_etiqueta'][et]:.4f} ({m['f1_por_etiqueta'][et] - control['f1_por_etiqueta'][et]:+.4f})"
        )

    log("3/5 complementariedad contra V03")
    pred_v03 = evaluar_bloque(desarrollo, pliegues, bloque=None)["pred"]
    cambios = pred_v03 != r["pred"]
    rescates = dan = 0
    por_label: dict[str, Any] = {}
    for j, et in enumerate(("HS", "TR", "AG")):
        bien_v03 = pred_v03[:, j] == y[:, j]
        bien_emb = r["pred"][:, j] == y[:, j]
        rr = int(np.count_nonzero(~bien_v03 & bien_emb))
        dd = int(np.count_nonzero(bien_v03 & ~bien_emb))
        rescates += rr
        dan += dd
        por_label[et] = {
            "rescates": rr, "danios": dd, "neto": rr - dd,
            "celdas_modificadas": int(cambios[:, j].sum()),
        }
    comp = {
        "rescates": rescates, "danios": dan, "neto": rescates - dan,
        "celdas_modificadas": int(cambios.sum()),
        "correlacion_de_decisiones": float(np.mean(pred_v03.ravel() == r["pred"].ravel())),
        "por_label": por_label,
    }
    log(
        f"   rescates {rescates} | danios {dan} | neto {rescates - dan:+d} | "
        f"correlacion {comp['correlacion_de_decisiones']:.4f}"
    )

    log("4/5 analisis de memorization y terminos raros")
    # Los terminos raros son los que aparecen en pocos documentos. Si el embedding solo funciona
    # para terminos frecuentes, el gap se explicaria por memorization del vocabulario.
    conteo = CountVectorizer(analyzer="word", ngram_range=(1, 1), min_df=1, lowercase=True)
    x_dev = conteo.fit_transform(desarrollo["text"].astype(str))
    df = np.asarray((x_dev > 0).sum(axis=0)).ravel()
    umbral_raro = 5
    raros = df <= umbral_raro
    # Para cada documento, proporcion de terminos raros
    largos = np.asarray(x_dev.sum(axis=1)).ravel()
    es_raro = np.asarray(x_dev[:, raros].sum(axis=1)).ravel()
    prop_raro = es_raro / np.where(largos > 0, largos, 1.0)
    # Correlacion entre proporcion de terminos raros y error
    error = r["pred"] != y
    corr_raro = float(np.corrcoef(prop_raro, error.any(axis=1).astype(float))[0, 1])
    log(f"   terminos con df <= {umbral_raro}: {int(raros.sum())} de {len(df)}")
    log(f"   correlacion entre proporcion de terminos raros y error: {corr_raro:+.4f}")
    memorization = {
        "umbral_raro_df": umbral_raro,
        "n_terminos_raros": int(raros.sum()),
        "proporcion_de_terminos_raros": float(raros.mean()),
        "correlacion_proporcion_rara_y_error": corr_raro,
        "train_f1": r["train_f1"],
        "oof_f1": cv,
        "gap": r["train_f1"] - cv,
        "depende_de_terminos_raros": bool(corr_raro > 0.3 and r["train_f1"] - cv > 0.15),
    }

    log("5/5 decision")
    decision = _decision(control, cv, comp, r)
    log(f"   DECISION: {decision}")

    informe = {
        "familia": "SUPERVISED_EMBEDDINGS_FROM_SCRATCH",
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estado": "OOF_EVALUATED",
        "pregunta": (
            "una representacion distribuida entrenada directamente con las etiquetas HS/TR/AG puede "
            "capturar relaciones utiles entre terminos que el TF-IDF explicito de V03 no captura, "
            "sin caer en memorization con solo ~1483 ejemplos"
        ),
        "alcance": (
            "factorizacion termino x etiqueta de baja capacidad. No es una red neuronal completa: es "
            "una representacion distribuida derivada de asociaciones termino-etiqueta"
        ),
        "metodo": (
            "matriz termino x etiqueta (conteos de co-ocurrencia con cada etiqueta), normalizada con "
            "BM25 por fila, factorizada con SVD truncado de baja dimension"
        ),
        "dimension": D,
        "k1": K1,
        "b": B,
        "vocabulario_y_parametros": r["stats_por_pliegue"],
        "objetivo": "mean pooling de embeddings de terminos + LogisticRegression, el mismo clasificador que V03",
        "regularizacion": "ninguna adicional: la baja dimension y el mean pooling son la regularizacion",
        "control": control,
        "embedding": {
            "corrected_cv": cv,
            "delta_vs_v03": cv - control["corrected_cv"],
            "f1_por_etiqueta": {et: float(m["f1_por_etiqueta"][et]) for et in ("HS", "TR", "AG")},
            "train_f1": r["train_f1"],
            "gap_train_oof": r["train_f1"] - cv,
            "metricas_por_pliegue": r["metricas_pliegues"],
        },
        "complementariedad": comp,
        "memorization": memorization,
        "evidencia_historica": (
            "los embeddings probados en EBD-01 a EBD-25 fueron NO supervisados (FastText, word2vec). "
            "No existe ninguna implementacion donde la capa embedding sea entrenada con las etiquetas "
            "HS/TR/AG. La familia esta abierta"
        ),
        "decision": decision,
        "gastos": {
            "submissions": 0,
            "modelos_entrenados": 3,
            "memoria_pico_mb": round(memoria_pico_mb(), 1),
            "segundos": round(time.time() - inicio, 1),
        },
    }
    REPORTE_JSON.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _informe_md(informe)
    log("   informe escrito en reports/v38_01_embedding_supervisado.{json,md}")
    log(f"   memoria pico {memoria_pico_mb():.0f} MB | tiempo {round(time.time() - inicio, 1)} s")
    log("VEREDICTO: embedding supervisado evaluado, 0 submissions")


def _decision(
    control: dict[str, Any], cv: float, comp: dict[str, Any], r: dict[str, Any]
) -> str:
    """Decision segun los criterios de la directiva.

    Args:
        control: metricas del control.
        cv: corrected_cv del embedding.
        comp: complementariedad contra V03.
        r: resultado completo del embedding.

    Returns:
        La decision como cadena.
    """
    delta = cv - control["corrected_cv"]
    gap = r["train_f1"] - cv
    if delta > 0 and comp["neto"] > 0 and gap < 0.15:
        return "NEW_SIGNAL"
    if delta > 0 or comp["neto"] > 0:
        return "WEAK_SIGNAL"
    if abs(delta) < 0.001 and comp["neto"] == 0:
        return "NO_SIGNAL"
    if gap > 0.15 and delta < 0:
        return "OVERFIT"
    return "REJECTED"


def _informe_md(informe: dict[str, Any]) -> None:
    """Escribe el informe legible.

    Args:
        informe: informe con los datos.

    Returns:
        None.
    """
    c = informe["control"]
    e = informe["embedding"]
    comp = informe["complementariedad"]
    mem = informe["memorization"]
    g = informe["gastos"]
    lineas = [
        "# V38-01: embedding supervisado por factorizacion termino x etiqueta",
        "",
        f"Generado por `experiments/v38_01_embedding_supervisado.py` ({EXPERIMENTO}).",
        "",
        f"**Estado `{informe['estado']}`. Alcance: {informe['alcance']}**",
        "",
        "## 1. El metodo",
        "",
        f"{informe['metodo']}.",
        "",
        f"Dimension **{informe['dimension']}**, k1 {informe['k1']}, b {informe['b']}.",
        "",
        "Por que no es equivalente a SVD sobre documento x termino: SVD sobre documento x termino es",
        "no supervisado, no usa las etiquetas. La factorizacion termino x etiqueta si las usa, porque la",
        "matriz se construye contando co-ocurrencias con cada etiqueta por separado.",
        "",
        "## 2. Vocabulario y parametros",
        "",
        "| pliegue | vocabulario | dimension | varianza explicada | parametros |",
        "|---|---:|---:|---:|---:|",
    ]
    for f, s in informe["vocabulario_y_parametros"].items():
        lineas.append(
            f"| {f} | {s['vocabulario']} | {s['dimension']} | "
            f"{s['varianza_explicada']:.4f} | {s['parametros_del_embedding']} |"
        )
    lineas += [
        "",
        "## 3. Resultado OOF",
        "",
        "| variante | corrected_cv | delta vs V03 |",
        "|---|---:|---:|",
        f"| V03 (control) | {c['corrected_cv']:.10f} | 0 |",
        f"| embedding supervisado | {e['corrected_cv']:.10f} | {e['delta_vs_v03']:+.10f} |",
        "",
        "| etiqueta | V03 | embedding | delta |",
        "|---|---:|---:|---:|",
    ]
    for et in ("HS", "TR", "AG"):
        lineas.append(
            f"| {et} | {c['f1_por_etiqueta'][et]:.4f} | {e['f1_por_etiqueta'][et]:.4f} | "
            f"{e['f1_por_etiqueta'][et] - c['f1_por_etiqueta'][et]:+.4f} |"
        )
    lineas += [
        "",
        f"train F1 {e['train_f1']:.4f} | gap {e['gap_train_oof']:+.4f}",
        "",
        "## 4. Complementariedad contra V03",
        "",
        "| label | rescates | danios | neto | celdas |",
        "|---|---:|---:|---:|---:|",
    ]
    for et in ("HS", "TR", "AG"):
        d = comp["por_label"][et]
        lineas.append(
            f"| {et} | {d['rescates']} | {d['danios']} | {d['neto']:+d} | "
            f"{d['celdas_modificadas']} |"
        )
    lineas += [
        "",
        (f"**rescates {comp['rescates']} | danios {comp['danios']} | neto {comp['neto']:+d} | "
        f"correlacion {comp['correlacion_de_decisiones']:.4f}**"),
        "",
        "## 5. Analisis de memorization y terminos raros",
        "",
        "| magnitud | valor |",
        "|---|---:|",
        (f"| terminos con df <= {mem['umbral_raro_df']} | {mem['n_terminos_raros']} "
        f"({100 * mem['proporcion_de_terminos_raros']:.2f} %) |"),
        (f"| correlacion proporcion rara y error | "
        f"{mem['correlacion_proporcion_rara_y_error']:+.4f} |"),
        f"| train F1 | {mem['train_f1']:.4f} |",
        f"| OOF F1 | {mem['oof_f1']:.4f} |",
        f"| gap | {mem['gap']:+.4f} |",
        f"| depende de terminos raros | {mem['depende_de_terminos_raros']} |",
        "",
        "## 6. Evidencia historica",
        "",
        f"{informe['evidencia_historica']}.",
        "",
        f"**DECISION: {informe['decision']}**",
        "",
        "## 7. Lo que este script no hace",
        "",
        "- No envia submission.",
        "- No construye una red neuronal completa.",
        "- No usa el test para nada.",
        "- No modifica V03.",
        "",
        (f"Coste: {g['segundos']} s, memoria {g['memoria_pico_mb']:.0f} MB, "
        f"{g['modelos_entrenados']} modelos, {g['submissions']} submissions."),
        "",
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
