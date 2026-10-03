"""P1c: downstream. Embedding-only, fusion con V03, y complementariedad.

Que mide cada cosa
------------------
    embedding-only   mean pooling de los vectores -> LogisticRegression, umbrales de V03
    fusion           la representacion TF-IDF de V03 CONCATENADA con el pool -> LR
    complementariedad  quien acierta donde el otro falla

La pregunta del encargo no es si el embedding solo es bueno, es si APORTA. Un embedding puede
mejorar solo y no aportar nada a V03, o al reves. Por eso las tres metricas.

El protocolo
------------
Los embeddings vienen de corpus externos SIN etiquetas, luego no introducen fuga. El LR si usa
las etiquetas de train y se evalua con los cinco pliegues canonicos: se ajusta en cuatro y se
predice el quinto. Los umbrales son los congelados de V03 (0.44/0.51/0.46) en todo el calculo.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import V03_CCV, V03_SHA, V03_SUB

from src.evaluation import metricas_multietiqueta
from src.splits import (
    ETIQUETAS,
    grupos_fusionados,
    holdout_v2,
    pliegues_corregidos,
)

REPORTS = PROJECT_ROOT / "reports"
MODELOS = PROJECT_ROOT / "modelos" / "twitter_domain"
UMBRALES = np.array([0.44, 0.51, 0.46])
RE_TOKEN = re.compile(r"[#@]?\w+(?:['\u2019]\w+)*", re.UNICODE)
SEMILLA = 42


def sembrar(semilla: int) -> None:
    """Siembra todo lo que interviene en el downstream.

    Args:
        semilla: semilla.
    """
    import os
    import random as _random
    os.environ["PYTHONHASHSEED"] = str(semilla)
    _random.seed(semilla)
    np.random.seed(semilla)


def log(m: str) -> None:
    """Imprime una linea de progreso.

    Args:
        m: texto a mostrar.
    """
    print(f"[P71-05] {m}", flush=True)


def f1m(y: np.ndarray, pred: np.ndarray) -> float:
    """Macro F1 con la metrica del proyecto.

    Args:
        y: etiquetas.
        pred: predicciones.

    Returns:
        La media de las F1 por columna.
    """
    return float(metricas_multietiqueta(y, pred)["f1_media_columnas"])


def cargar_modelo(etiqueta: str) -> tuple[dict[str, int], np.ndarray]:
    """Carga un modelo entrenado.

    Args:
        etiqueta: nombre del corpus.

    Returns:
        Indice de palabras y matriz de embeddings.
    """
    d = np.load(MODELOS / f"w2v_{etiqueta}.npz", allow_pickle=True)
    return {w: i for i, w in enumerate(d["palabras"])}, d["W"]


def pool(textos, vocab: dict[str, int], W: np.ndarray) -> np.ndarray:
    """Mean pooling de los vectores de cada texto.

    Args:
        textos: serie de textos.
        vocab: indice de palabras.
        W: matriz de embeddings.

    Returns:
        Matriz (n, dim).
    """
    out = np.zeros((len(textos), W.shape[1]), dtype=np.float32)
    for i, t in enumerate(textos):
        ids = [vocab[w] for w in RE_TOKEN.findall(str(t)) if w in vocab]
        if ids:
            out[i] = W[ids].mean(axis=0)
    return out


def cv_embedding(X: np.ndarray, y: np.ndarray,
                 pliegues: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, list]:
    """OOF del LR sobre una representacion, con los pliegues canonicos.

    Args:
        X: matriz de caracteristicas.
        y: etiquetas.
        pliegues: pares entrenamiento-validacion.

    Returns:
        Probabilidades OOF y las F1 por pliegue.
    """
    proba = np.zeros((X.shape[0], len(ETIQUETAS)))
    f1s = []
    for tr, va in pliegues:
        for j in range(len(ETIQUETAS)):
            m = LogisticRegression(C=1.0, max_iter=2000, solver="liblinear",
                                   random_state=SEMILLA)
            m.fit(X[tr], y[tr, j])
            proba[va, j] = m.predict_proba(X[va])[:, 1]
        f1s.append(round(f1m(y[va], (proba[va] >= UMBRALES[None, :]).astype(int)), 6))
    return proba, f1s


def main() -> None:
    """Ejecuta el downstream completo de los dos embeddings.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    sembrar(SEMILLA)
    log("P71-05 P1c: downstream embedding-only, fusion y complementariedad")
    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler

    from src.data import read_train

    v = np.load(PROJECT_ROOT / "features" / "oof_v03_final.npz", allow_pickle=True)
    y = v["y_true"].astype(int)
    ids = v["ids"].astype(np.int64)
    p_v03 = v["proba"]
    pred_v03 = (p_v03 >= UMBRALES[None, :]).astype(int)
    cv_v03 = f1m(y, pred_v03)

    train = read_train()
    grupos = grupos_fusionados(train)
    idev, _ = holdout_v2(train, grupos)
    desarrollo = train.iloc[idev].reset_index(drop=True)
    if not np.array_equal(desarrollo["id"].to_numpy(dtype=np.int64), ids):
        raise SystemExit("PARADA: los ids no coinciden")
    pliegues = pliegues_corregidos(desarrollo, grupos[idev])
    textos = desarrollo["text"]
    log(f"   V03 de referencia: {cv_v03:.10f} | {len(y)} filas | 5 pliegues canonicos")

    # Representacion de V03, la misma familia, para la fusion
    pal = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    car = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                          sublinear_tf=True)
    X_v03 = sparse.hstack([pal.fit_transform(textos), car.fit_transform(textos)],
                          format="csr")
    log(f"   representacion de V03 para la fusion: {X_v03.shape[1]:,} dimensiones")

    res = {"v03": round(cv_v03, 10)}
    probas = {}
    for etiqueta in ("twitter", "general"):
        t0 = time.time()
        vocab, W = cargar_modelo(etiqueta)
        P = pool(textos, vocab, W)
        P = StandardScaler().fit_transform(P).astype(np.float32)
        no_cero = int((np.abs(P).sum(axis=1) > 0).sum())
        p_emb, f1s = cv_embedding(P, y, pliegues)
        cv_emb = f1m(y, (p_emb >= UMBRALES[None, :]).astype(int))
        X_fus = sparse.hstack([X_v03, sparse.csr_matrix(P)], format="csr")
        p_fus, f1f = cv_embedding(X_fus, y, pliegues)
        cv_fus = f1m(y, (p_fus >= UMBRALES[None, :]).astype(int))
        probas[etiqueta] = {"emb": p_emb, "fus": p_fus}
        res[etiqueta] = {
            "vocabulario": len(vocab), "dim": int(W.shape[1]),
            "filas_con_vector_no_cero": no_cero,
            "embedding_only_cv": round(cv_emb, 10),
            "embedding_only_delta": round(cv_emb - cv_v03, 10),
            "embedding_f1_pliegues": f1s,
            "fusion_cv": round(cv_fus, 10),
            "fusion_delta": round(cv_fus - cv_v03, 10),
            "fusion_f1_pliegues": f1f,
            "segundos": round(time.time() - t0, 1),
        }
        f = res[etiqueta]
        log("")
        log(f"   {etiqueta.upper()}")
        log(f"      embedding-only {cv_emb:.10f}  delta {cv_emb - cv_v03:+.10f}")
        log(f"      fusion         {cv_fus:.10f}  delta {cv_fus - cv_v03:+.10f}")
        log(f"      f1 por pliegue (fusion): {f1f}")

    # --- la comparacion que pide el encargo: twitter frente a general ---
    d_emb = res["twitter"]["embedding_only_cv"] - res["general"]["embedding_only_cv"]
    d_fus = res["twitter"]["fusion_cv"] - res["general"]["fusion_cv"]
    log("")
    log("   CONTROL DE DOMINIO: twitter frente a general")
    log(f"      embedding-only  twitter {res['twitter']['embedding_only_cv']:.6f} | "
        f"general {res['general']['embedding_only_cv']:.6f} -> {d_emb:+.6f}")
    log(f"      fusion          twitter {res['twitter']['fusion_cv']:.6f} | "
        f"general {res['general']['fusion_cv']:.6f} -> {d_fus:+.6f}")

    # --- complementariedad, la del mejor de los dos ---
    comp = {}
    for etiqueta in ("twitter", "general"):
        p = probas[etiqueta]["fus"]
        pr = (p >= UMBRALES[None, :]).astype(int)
        ar, ac = (pred_v03 == y), (pr == y)
        comp[etiqueta] = {
            "rescue": int((~ar & ac).sum()), "damage": int((ar & ~ac).sum()),
            "por_etiqueta": {ETIQUETAS[j]: {
                "rescue": int((~ar[:, j] & ac[:, j]).sum()),
                "damage": int((ar[:, j] & ~ac[:, j]).sum())} for j in range(3)},
        }
        comp[etiqueta]["net"] = comp[etiqueta]["rescue"] - comp[etiqueta]["damage"]
        log(f"   complementariedad {etiqueta}: rescue {comp[etiqueta]['rescue']} | "
            f"damage {comp[etiqueta]['damage']} | neto {comp[etiqueta]['net']:+d}")

    # --- el gate del encargo, sobre la fusion ---
    gates = {}
    for etiqueta in ("twitter", "general"):
        f = res[etiqueta]
        comp_ok = comp[etiqueta]["net"] > 0
        pos = sum(1 for a, b in zip(f["fusion_f1_pliegues"],
                                    [round(f1m(y[va], pred_v03[va]), 6) for _, va in pliegues])
                  if a > b)
        gates[etiqueta] = {
            "delta_fusion": f["fusion_delta"],
            "delta_mayor_igual_00015": bool(f["fusion_delta"] >= 0.0015),
            "pliegues_positivos": pos,
            "cuatro_de_cinco": bool(pos >= 4),
            "complementariedad_neta_positiva": bool(comp_ok),
            "pasa_gate": bool(f["fusion_delta"] >= 0.0015 and pos >= 4 and comp_ok),
        }
        log(f"   gate {etiqueta}: delta {f['fusion_delta']:+.6f} | pliegues {pos}/5 | "
            f"neto {comp[etiqueta]['net']:+d} | PASA {gates[etiqueta]['pasa_gate']}")

    tw = res["twitter"]
    dom = "TWITTER_MEJORA" if d_fus > 0 else ("GENERAL_MEJORA" if d_fus < 0 else "EMPATE")
    if tw["fusion_delta"] >= 0.0015 and gates["twitter"]["pasa_gate"]:
        veredicto = "PROMISING"
    elif d_fus <= 0 and tw["fusion_delta"] < 0.0015:
        veredicto = "CLOSED"
    else:
        veredicto = "MARGINAL"
    log("")
    log(f"   dominio: {dom} ({d_fus:+.6f})")
    log(f"   VEREDICTO: {veredicto}")

    salida = {
        "familia": "P71_TWITTER_WORD2VEC", "experiment_id": "P71-05", "fase": "P1c downstream",
        "fecha": inicio.isoformat(timespec="seconds"),
        "naturaleza": ("EXPERIMENTO. 2 corpus, 2 downstreams cada uno. 0 test, 0 holdout, "
                       "0 leaderboard"),
        "v03_referencia": round(cv_v03, 10),
        "resultados": res, "control_de_dominio": {"delta_embedding": round(d_emb, 10),
                                                  "delta_fusion": round(d_fus, 10),
                                                  "ganador": dom},
        "complementariedad": comp, "gates": gates, "veredicto": veredicto,
        "decision_del_encargo": {
            "A_twitter_mejora_claramente": bool(d_fus > 0 and tw["fusion_delta"] >= 0.0015),
            "B_mejora_embedding_pero_no_fusion": bool(
                res["twitter"]["embedding_only_delta"] > 0 and tw["fusion_delta"] < 0.0015),
            "C_no_mejora_al_general": bool(d_fus <= 0),
            "D_empeora": bool(tw["fusion_delta"] < 0),
            "resultado": veredicto,
        },
        "reproducibilidad": {"nivel": "R2", "hash_v03": hash_real, "corrected_cv_v03": V03_CCV,
                             "modelos": [res["twitter"]["vocabulario"],
                                         res["general"]["vocabulario"]]},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    (REPORTS / "twitter_word2vec_domain_results.json").write_text(
        json.dumps(salida, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    import csv as _csv
    with (REPORTS / "twitter_word2vec_domain_results.csv").open("w", encoding="ascii",
                                                                newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["corpus", "vocabulario", "embedding_only_cv", "embedding_delta",
                    "fusion_cv", "fusion_delta", "rescue", "damage", "net",
                    "f1_pliegues_fusion", "pasa_gate"])
        w.writerow(["V03", "", round(cv_v03, 10), 0.0, round(cv_v03, 10), 0.0, "", "", "",
                    "", ""])
        for e in ("twitter", "general"):
            f = res[e]
            w.writerow([e, f["vocabulario"], f["embedding_only_cv"], f["embedding_only_delta"],
                        f["fusion_cv"], f["fusion_delta"], comp[e]["rescue"],
                        comp[e]["damage"], comp[e]["net"],
                        "|".join(str(x) for x in f["fusion_f1_pliegues"]),
                        gates[e]["pasa_gate"]])
    with (REPORTS / "twitter_word2vec_complementarity.csv").open("w", encoding="ascii",
                                                                 newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["corpus", "etiqueta", "rescue", "damage", "net"])
        for e in ("twitter", "general"):
            for k, val in comp[e]["por_etiqueta"].items():
                w.writerow([e, k, val["rescue"], val["damage"],
                            val["rescue"] - val["damage"]])
            w.writerow([e, "GLOBAL", comp[e]["rescue"], comp[e]["damage"], comp[e]["net"]])

    log("")
    log("=" * 78)
    log(f"P71-05 OK | {veredicto} | dominio {dom}")
    log(f"   twitter fusion {res['twitter']['fusion_cv']:.6f} | "
        f"general fusion {res['general']['fusion_cv']:.6f}")
    log("=" * 78)


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
