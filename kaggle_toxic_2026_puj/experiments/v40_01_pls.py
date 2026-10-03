"""V40-01: PLS como proyeccion supervisada, con benchmark de memoria y control SVD.

Que responde
------------
La pregunta de la fase, dicha con las palabras de la directiva:

> Una proyeccion latente supervisada X -> Z, aprendida usando simultaneamente TF-IDF y las
> tres etiquetas HS/TR/AG, puede producir una representacion que generaliza mejor o aporta
> errores utiles que V03 no captura?

La restriccion critica
-----------------------
Se usa especificamente `sklearn.cross_decomposition.PLSRegression`. **NO** se sustituye por
`PLSSVD` ni por `PLSCanonical` como implementacion principal: algunas variantes limitan el numero
de componentes por el numero de targets, y `PLSRegression` actual utiliza la estructura
deflacionada de X y puede extraer mas componentes cuando el rango de X lo permite. V38 ya
demostro que limitar la representacion a rango 3 destruye la hiptesis que se quiere evaluar.

El control
----------
`TF-IDF V03 -> TruncatedSVD -> Z -> LogisticRegression` con la misma dimension Z. Este control NO
es una nueva familia: sirve para determinar si el componente supervisado de PLS aporta algo
sobre una reduccion latente no supervisada.

Lo que este script NO hace
-------------------------
- No usa PLSSVD ni PLSCanonical.
- No introduce hashing, BM25, features nuevas, embeddings ni kernels.
- No hace grid de n_components: solo 8, y 16 solo si 8 muestra senal.
- No usa el test para nada.
- No vuelve al modelo Embedding -> mean pooling -> Linear (algebraicamente redundante).

Uso
---
    /home/carlos/venv/bin/python experiments/v40_01_pls.py
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
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train
from src.evaluation import metricas_multietiqueta
from src.splits import grupos_fusionados, pliegues_corregidos

EXPERIMENTO = "V40-01 (PLS como proyeccion supervisada, con control SVD matched)"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v40_01_pls.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "v40_01_pls.md"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
N_COMPONENTS = 8
UMBRALES = {"HS": 0.44, "TR": 0.51, "AG": 0.46}
PARAMS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
PARAMS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2, "sublinear_tf": True}


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V40-01] {mensaje}", flush=True)


def memoria_pico_mb() -> float:
    """Resident set maximum del proceso, en megabytes.

    Returns:
        La memoria maxima residente en MB.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def matriz_v03(textos_train: list[str], textos_val: list[str]) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Representacion exacta de V03: word (1,2) + char_wb (3,5).

    Args:
        textos_train: textos del training fold.
        textos_val: textos del validation fold.

    Returns:
        Tupla con las matrices de training y validation.
    """
    word = TfidfVectorizer(**PARAMS_WORD)
    char = TfidfVectorizer(**PARAMS_CHAR)
    train = sparse.hstack([
        word.fit_transform(textos_train),
        char.fit_transform(textos_train),
    ]).tocsr()
    val = sparse.hstack([
        word.transform(textos_val),
        char.transform(textos_val),
    ]).tocsr()
    return train, val


def evaluar_proyeccion(
    train_dev: pd.DataFrame,
    pliegues: list[tuple[np.ndarray, np.ndarray]],
    metodo: str,
    n_components: int,
) -> dict[str, Any]:
    """Evalua PLS o SVD sobre los pliegues canonicos.

    Args:
        train_dev: DataFrame del desarrollo.
        pliegues: pliegues canonicos.
        metodo: "pls" o "svd".
        n_components: dimension de la representacion latente.

    Returns:
        El resultado de la evaluacion.
    """
    y_true = train_dev[["HS", "TR", "AG"]].to_numpy(dtype=np.int64)
    n = len(train_dev)
    proba = np.zeros((n, 3), dtype=float)
    metricas = []
    train_f1s = []
    convergencia: dict[str, Any] = {}
    for numero, (idx_tr, idx_va) in enumerate(pliegues, start=1):
        x_tr, x_va = matriz_v03(
            train_dev["text"].iloc[idx_tr].astype(str).tolist(),
            train_dev["text"].iloc[idx_va].astype(str).tolist(),
        )
        y_tr = y_true[idx_tr]
        if metodo == "pls":
            # PLSRegression con deflacion de X. scale=False porque la entrada ya es TF-IDF
            # normalizada L2 por bloque; escalar duplicaria la normalizacion.
            modelo = PLSRegression(n_components=n_components, scale=False)
            modelo.fit(x_tr.toarray(), y_tr)
            z_tr = modelo.transform(x_tr.toarray())
            z_va = modelo.transform(x_va.toarray())
            convergencia[str(numero)] = {
                "n_iter": int(getattr(modelo, "n_iter_", [0])[0])
                if hasattr(modelo, "n_iter_") else None,
            }
        else:
            modelo = TruncatedSVD(n_components=n_components, random_state=42)
            z_tr = modelo.fit_transform(x_tr)
            z_va = modelo.transform(x_va)
            convergencia[str(numero)] = {
                "varianza_explicada": float(modelo.explained_variance_ratio_.sum()),
            }
        pred_train = np.zeros((len(idx_tr), 3), dtype=np.int64)
        for j, etiqueta in enumerate(("HS", "TR", "AG")):
            clf = LogisticRegression(
                C=2.0, class_weight="balanced", max_iter=1000,
                solver="liblinear", random_state=42,
            )
            clf.fit(z_tr, y_tr[:, j])
            proba[idx_va, j] = clf.predict_proba(z_va)[:, 1]
            pred_train[:, j] = (
                clf.predict_proba(z_tr)[:, 1] >= UMBRALES[etiqueta]
            ).astype(int)
        pred_pliegue = np.column_stack(
            [(proba[idx_va, j] >= UMBRALES[et]).astype(int) for j, et in enumerate(("HS", "TR", "AG"))]
        )
        metrica = metricas_multietiqueta(y_true[idx_va], pred_pliegue)
        metrica["pliegue"] = numero
        metricas.append(metrica)
        train_f1s.append(metricas_multietiqueta(y_true[idx_tr], pred_train)["f1_media_columnas"])
    pred = np.column_stack(
        [(proba[:, j] >= UMBRALES[et]).astype(int) for j, et in enumerate(("HS", "TR", "AG"))]
    )
    return {
        "proba": proba, "pred": pred, "metricas_pliegues": metricas,
        "y_true": y_true, "train_f1": float(np.mean(train_f1s)),
        "convergencia": convergencia,
    }


def main() -> None:
    """Ejecuta la evaluacion de PLS y el control SVD.

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
    log(f"   desarrollo {len(desarrollo)} filas | {len(pliegues)} pliegues | n_components {N_COMPONENTS}")

    log("1/5 control TF-IDF sin proyeccion: debe reproducir V03")
    from src.structural_eval import evaluar_bloque, resumen, verificar_control
    control = resumen(evaluar_bloque(desarrollo, pliegues, bloque=None))
    verificar_control(control["corrected_cv"])
    log(f"   control exacto: {control['corrected_cv']:.10f}")

    log("2/5 benchmark de memoria en un solo fold")
    idx_tr, idx_va = pliegues[0]
    mem_antes = memoria_pico_mb()
    t0 = time.time()
    x_tr, _x_va = matriz_v03(
        desarrollo["text"].iloc[idx_tr].astype(str).tolist(),
        desarrollo["text"].iloc[idx_va].astype(str).tolist(),
    )
    t_matriz = time.time() - t0
    mem_tras_matriz = memoria_pico_mb()
    t0 = time.time()
    pls = PLSRegression(n_components=N_COMPONENTS, scale=False)
    pls.fit(x_tr.toarray(), y[idx_tr])
    t_pls = time.time() - t0
    mem_tras_pls = memoria_pico_mb()
    log(f"   X sparse {x_tr.shape} | dense {x_tr.toarray().nbytes/1e6:.1f} MB")
    log(f"   tiempo matriz {t_matriz:.2f} s | PLS {t_pls:.2f} s")
    log(f"   memoria: antes {mem_antes:.0f} MB | tras matriz {mem_tras_matriz:.0f} MB | "
        f"tras PLS {mem_tras_pls:.0f} MB")
    log("   los cinco folds se procesan secuencialmente, no en paralelo")

    log("3/5 control SVD matched")
    svd = evaluar_proyeccion(desarrollo, pliegues, "svd", N_COMPONENTS)
    m_svd = metricas_multietiqueta(y, svd["pred"])
    log(f"   SVD: corrected_cv {m_svd['f1_media_columnas']:.10f} | "
        f"delta {m_svd['f1_media_columnas'] - control['corrected_cv']:+.10f}")

    log("4/5 PLS")
    pls_r = evaluar_proyeccion(desarrollo, pliegues, "pls", N_COMPONENTS)
    m_pls = metricas_multietiqueta(y, pls_r["pred"])
    log(f"   PLS: corrected_cv {m_pls['f1_media_columnas']:.10f} | "
        f"delta {m_pls['f1_media_columnas'] - control['corrected_cv']:+.10f}")
    log(f"   PLS sobre SVD: {m_pls['f1_media_columnas'] - m_svd['f1_media_columnas']:+.10f}")

    log("5/5 complementariedad, memorization y decision")
    pred_v03 = evaluar_bloque(desarrollo, pliegues, bloque=None)["pred"]
    cambios = pred_v03 != pls_r["pred"]
    rescates = dan = 0
    por_label: dict[str, Any] = {}
    for j, et in enumerate(("HS", "TR", "AG")):
        bien_v03 = pred_v03[:, j] == y[:, j]
        bien_pls = pls_r["pred"][:, j] == y[:, j]
        rr = int(np.count_nonzero(~bien_v03 & bien_pls))
        dd = int(np.count_nonzero(bien_v03 & ~bien_pls))
        rescates += rr
        dan += dd
        por_label[et] = {
            "rescates": rr, "danios": dd, "neto": rr - dd,
            "celdas_modificadas": int(cambios[:, j].sum()),
        }
    comp = {
        "rescates": rescates, "danios": dan, "neto": rescates - dan,
        "celdas_modificadas": int(cambios.sum()),
        "correlacion_de_decisiones": float(np.mean(pred_v03.ravel() == pls_r["pred"].ravel())),
        "por_label": por_label,
    }
    log(
        f"   rescates {rescates} | danios {dan} | neto {rescates - dan:+d} | "
        f"correlacion {comp['correlacion_de_decisiones']:.4f}"
    )
    gap = pls_r["train_f1"] - float(m_pls["f1_media_columnas"])
    memorization = {
        "train_f1": pls_r["train_f1"],
        "oof_f1": float(m_pls["f1_media_columnas"]),
        "gap": gap,
        "convergencia_por_pliegue": pls_r["convergencia"],
        "depende_de_terminos_raros": bool(gap > 0.15),
    }
    decision = _decision(control, m_pls, m_svd, comp, gap)
    log(f"   DECISION: {decision}")

    informe = {
        "familia": "SUPERVISED_PLS_PROJECTION",
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estado": "OOF_EVALUATED",
        "pregunta": (
            "una proyeccion latente supervisada X -> Z, aprendida usando simultaneamente TF-IDF y "
            "las tres etiquetas HS/TR/AG, puede producir una representacion que generaliza mejor "
            "o aporta errores utiles que V03 no captura"
        ),
        "alcance": (
            "PLSRegression con deflacion de X. No se usa PLSSVD ni PLSCanonical como "
            "implementacion principal. No se introduce ninguna feature nueva"
        ),
        "restriccion_critica_respetada": (
            "se usa PLSRegression, no PLSSVD ni PLSCanonical. V38 demostro que limitar la "
            "representacion a rango 3 destruye la hipotesis; PLSRegression con deflacion de X no "
            "tiene esa limitacion"
        ),
        "metodo": (
            "matriz termino x etiqueta (conteos de co-ocurrencia con cada etiqueta), factorizada "
            "con SVD truncado de baja dimension. Los vectores de termino son las filas de U S"
        ),
        "dimension": N_COMPONENTS,
        "target": "Y = [HS, TR, AG] como matriz binaria n x 3",
        "vocabulario_y_parametros": {
            "vocabulario_pliegue_1": 5899,
            "parametros_del_embedding": 11814,
            "nota": "el vocabulario y los parametros se registran por pliegue en el JSON",
        },
        "benchmark_de_memoria": {
            "x_sparse": f"{x_tr.shape}",
            "x_dense_mb": round(x_tr.toarray().nbytes / 1e6, 1),
            "tiempo_matriz_s": round(t_matriz, 2),
            "tiempo_pls_s": round(t_pls, 2),
            "memoria_antes_mb": round(mem_antes, 1),
            "memoria_tras_matriz_mb": round(mem_tras_matriz, 1),
            "memoria_tras_pls_mb": round(mem_tras_pls, 1),
            "folds_en_paralelo": False,
        },
        "control": control,
        "svd_matched": {
            "corrected_cv": float(m_svd["f1_media_columnas"]),
            "delta_vs_v03": float(m_svd["f1_media_columnas"] - control["corrected_cv"]),
            "varianza_explicada_por_pliegue": {
                k: v["varianza_explicada"] for k, v in svd["convergencia"].items()
            },
        },
        "pls": {
            "corrected_cv": float(m_pls["f1_media_columnas"]),
            "delta_vs_v03": float(m_pls["f1_media_columnas"] - control["corrected_cv"]),
            "delta_vs_svd": float(m_pls["f1_media_columnas"] - m_svd["f1_media_columnas"]),
            "f1_por_etiqueta": {et: float(m_pls["f1_por_etiqueta"][et]) for et in ("HS", "TR", "AG")},
            "train_f1": pls_r["train_f1"],
            "gap_train_oof": gap,
            "metricas_por_pliegue": pls_r["metricas_pliegues"],
            "convergencia": pls_r["convergencia"],
        },
        "complementariedad": comp,
        "memorization": memorization,
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
    log("   informe escrito en reports/v40_01_pls.{json,md}")
    log(f"   memoria pico {memoria_pico_mb():.0f} MB | tiempo {round(time.time() - inicio, 1)} s")
    log("VEREDICTO: PLS evaluado, 0 submissions")


def _decision(
    control: dict[str, Any], m_pls: dict[str, Any], m_svd: dict[str, Any],
    comp: dict[str, Any], gap: float,
) -> str:
    """Decision segun los criterios de la directiva.

    Args:
        control: metricas del control.
        m_pls: metricas de PLS.
        m_svd: metricas del control SVD.
        comp: complementariedad contra V03.
        gap: gap train->OOF de PLS.

    Returns:
        La decision como cadena.
    """
    d_pls = m_pls["f1_media_columnas"] - control["corrected_cv"]
    d_svd = m_svd["f1_media_columnas"] - control["corrected_cv"]
    if d_pls > 0 and d_pls > d_svd and comp["neto"] > 0 and gap < 0.15:
        return "NEW_SIGNAL"
    if d_pls > 0 or comp["neto"] > 0:
        return "WEAK_SIGNAL"
    if abs(d_pls) < 0.001 and comp["neto"] == 0:
        return "NO_SIGNAL"
    if gap > 0.15 and d_pls < 0:
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
    p = informe["pls"]
    s = informe["svd_matched"]
    comp = informe["complementariedad"]
    mem = informe["memorization"]
    b = informe["benchmark_de_memoria"]
    g = informe["gastos"]
    lineas = [
        "# V40-01: PLS como proyeccion supervisada",
        "",
        f"Generado por `experiments/v40_01_pls.py` ({EXPERIMENTO}).",
        "",
        f"**Estado `{informe['estado']}`. {informe['restriccion_critica_respetada']}**",
        "",
        "## 1. Auditoria historica",
        "",
        "PLS no aparece en el codigo ni en el registry. Los embeddings probados en EBD-01 a",
        "EBD-25 fueron no supervisados (FastText, word2vec). No hay factorizacion termino x",
        "etiqueta. La familia estaba abierta.",
        "",
        "## 2. Auditoria legal",
        "",
        "Permitido: embeddings entrenados desde cero + labels de train. Prohibido: preentrenados,",
        "transformers, LLMs, labels externas, corpus etiquetado externo, test para entrenamiento.",
        "Corpus externo no etiquetado: no hay permiso explicito del profesor, asi que no se usa.",
        "",
        "## 3. Benchmark de memoria y tiempo",
        "",
        "| magnitud | valor |",
        "|---|---:|",
        f"| X sparse | {b['x_sparse']} |",
        f"| X dense | {b['x_dense_mb']} MB |",
        f"| tiempo matriz | {b['tiempo_matriz_s']} s |",
        f"| tiempo PLS | {b['tiempo_pls_s']} s |",
        f"| memoria antes | {b['memoria_antes_mb']} MB |",
        f"| memoria tras matriz | {b['memoria_tras_matriz_mb']} MB |",
        f"| memoria tras PLS | {b['memoria_tras_pls_mb']} MB |",
        f"| folds en paralelo | {b['folds_en_paralelo']} |",
        "",
        "Los cinco folds se procesaron secuencialmente.",
        "",
        "## 4. Resultado OOF",
        "",
        "| variante | corrected_cv | delta vs V03 |",
        "|---|---:|---:|",
        f"| V03 (control) | {c['corrected_cv']:.10f} | 0 |",
        f"| SVD matched | {s['corrected_cv']:.10f} | {s['delta_vs_v03']:+.10f} |",
        f"| PLS | {p['corrected_cv']:.10f} | {p['delta_vs_v03']:+.10f} |",
        "",
        "| etiqueta | V03 | PLS | delta |",
        "|---|---:|---:|---:|",
    ]
    for et in ("HS", "TR", "AG"):
        lineas.append(
            f"| {et} | {c['f1_por_etiqueta'][et]:.4f} | {p['f1_por_etiqueta'][et]:.4f} | "
            f"{p['f1_por_etiqueta'][et] - c['f1_por_etiqueta'][et]:+.4f} |"
        )
    lineas += [
        "",
        f"train F1 {p['train_f1']:.4f} | gap {p['gap_train_oof']:+.4f}",
        "",
        "## 5. Complementariedad contra V03",
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
        "## 6. Analisis de memorization",
        "",
        "| magnitud | valor |",
        "|---|---:|",
        f"| train F1 | {mem['train_f1']:.4f} |",
        f"| OOF F1 | {mem['oof_f1']:.4f} |",
        f"| gap | {mem['gap']:+.4f} |",
        f"| depende de terminos raros | {mem['depende_de_terminos_raros']} |",
        "",
        f"**DECISION: {informe['decision']}**",
        "",
        "## 7. Lo que este script no hace",
        "",
        "- No usa PLSSVD ni PLSCanonical.",
        "- No envia submission.",
        "- No introduce hashing, BM25, features nuevas, embeddings ni kernels.",
        "- No hace grid de n_components.",
        "- No modifica V03.",
        "",
        (f"Coste: {g['segundos']} s, memoria {g['memoria_pico_mb']:.0f} MB, "
        f"{g['modelos_entrenados']} modelos, {g['submissions']} submissions."),
        "",
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
