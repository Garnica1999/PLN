"""P1: downstream de las tres dimensiones. Capacidad, transferencia y complementariedad.

Las cuatro cosas que el encargo pide separar
--------------------------------------------
    capacidad de la representacion    embedding-only, por dimension
    composicion del tweet             no se toca aqui: mean pooling en las tres, para que la
                                      dimension sea la unica variable
    transferencia al downstream       fusion con V03
    complementariedad                 quien acierta donde el otro falla

El protocolo, que es el de V03 verificado
-----------------------------------------
El vectorizador se ajusta DENTRO de cada pliegue y el clasificador usa C=2.0 con
`class_weight=balanced`. Con eso la reconstruccion reproduce V03 exactamente, 0,7772114525165721.

El bloque denso se escala para igualar la norma de fila del bloque disperso. Sin eso las 100, 200
o 300 columnas densas de varianza unidad dominan la penalizacion L2 y la comparacion entre
dimensiones mediria el desequilibrio, no la dimension.
"""

from __future__ import annotations

import ast
import csv as _csv
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import V03_CCV, V03_SHA, V03_SUB

from src.data import read_train
from src.evaluation import metricas_multietiqueta
from src.splits import ETIQUETAS, grupos_fusionados, holdout_v2, pliegues_corregidos

REPORTS = PROJECT_ROOT / "reports"
MODELOS = PROJECT_ROOT / "modelos" / "twitter_domain"
UMBRALES = np.array([0.44, 0.51, 0.46])
RE_TOKEN = re.compile(r"[#@]?\w+(?:['\u2019]\w+)*", re.UNICODE)
SEMILLA = 42
DIMS = (100, 200, 300)
ESCALAS = (0.05, 0.20)
BINS = (("CORTA", 0, 6), ("MEDIA", 7, 15), ("LARGA", 16, 30), ("MUY_LARGA", 31, 10**9))


def log(m: str) -> None:
    """Imprime una linea de progreso.

    Args:
        m: texto a mostrar.
    """
    print(f"[P72-02] {m}", flush=True)


def f1m(y: np.ndarray, pred: np.ndarray) -> float:
    """Macro F1 del proyecto.

    Args:
        y: etiquetas.
        pred: predicciones.

    Returns:
        La media de las F1 por columna.
    """
    return float(metricas_multietiqueta(y, pred)["f1_media_columnas"])


def cargar(dim: int):
    """Carga el modelo de una dimension.

    Args:
        dim: dimension.

    Returns:
        Indice y matriz.
    """
    nombre = "w2v_twitter_lrfix.npz" if dim == 200 else f"w2v_twitter_{dim}d.npz"
    d = np.load(MODELOS / nombre, allow_pickle=True)
    return {w: i for i, w in enumerate(d["palabras"])}, d["W"]


def pool(textos, vocab, W):
    """Mean pooling, con contadores de cobertura.

    Args:
        textos: textos.
        vocab: indice.
        W: embeddings.

    Returns:
        Matriz, tokens por fila y cubiertos.
    """
    out = np.zeros((len(textos), W.shape[1]), dtype=np.float32)
    n_tok = np.zeros(len(textos), dtype=int)
    n_cub = np.zeros(len(textos), dtype=int)
    for i, x in enumerate(textos):
        toks = RE_TOKEN.findall(str(x))
        n_tok[i] = len(toks)
        ii = [vocab[w] for w in toks if w in vocab]
        n_cub[i] = len(ii)
        if ii:
            out[i] = W[ii].mean(axis=0)
    return out, n_tok, n_cub


def cv(des, tx, y, pl, D=None, escala=0.0):
    """CV con el protocolo de V03 y bloque denso opcional.

    Args:
        des: DataFrame.
        tx: textos.
        y: etiquetas.
        pl: pliegues.
        D: bloque denso.
        escala: fraccion de la norma del bloque disperso.

    Returns:
        El CV, las probabilidades y las F1 por pliegue.
    """
    pr = np.zeros((len(y), len(ETIQUETAS)))
    f1s = []
    for a, b in pl:
        p2 = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        c2 = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                             sublinear_tf=True)
        Xi = sparse.hstack([p2.fit_transform(tx.iloc[a]), c2.fit_transform(tx.iloc[a])],
                           format="csr")
        Xv = sparse.hstack([p2.transform(tx.iloc[b]), c2.transform(tx.iloc[b])], format="csr")
        if D is not None and escala > 0:
            sc = StandardScaler().fit(D[a])
            Da, Db = sc.transform(D[a]), sc.transform(D[b])
            nd = float(np.sqrt((Da ** 2).sum(axis=1).mean()))
            nx = float(np.sqrt(Xi.multiply(Xi).sum(axis=1).mean()))
            k = escala * nx / max(nd, 1e-12)
            Xi = sparse.hstack([Xi, sparse.csr_matrix(Da * k)], format="csr")
            Xv = sparse.hstack([Xv, sparse.csr_matrix(Db * k)], format="csr")
        for j in range(len(ETIQUETAS)):
            m = LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000,
                                   solver="liblinear", random_state=SEMILLA)
            m.fit(Xi, y[a, j])
            pr[b, j] = m.predict_proba(Xv)[:, 1]
        f1s.append(round(f1m(y[b], (pr[b] >= UMBRALES[None, :]).astype(int)), 6))
    return f1m(y, (pr >= UMBRALES[None, :]).astype(int)), pr, f1s


def main() -> None:
    """Ejecuta el downstream de las tres dimensiones.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P72-02 P1: downstream de 100d, 200d y 300d")
    hash_v03 = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_v03 != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")

    v = np.load(PROJECT_ROOT / "features" / "oof_v03_final.npz", allow_pickle=True)
    y = v["y_true"].astype(int)
    pred_v03 = (v["proba"] >= UMBRALES[None, :]).astype(int)
    tr = read_train()
    g = grupos_fusionados(tr)
    idev, _ = holdout_v2(tr, g)
    des = tr.iloc[idev].reset_index(drop=True)
    pl = pliegues_corregidos(des, g[idev])
    tx = des["text"]

    base, _, _f1_base = cv(des, tx, y, pl)
    log(f"   V03 con el protocolo verificado: {base:.10f}")
    if abs(base - V03_CCV) > 1e-9:
        raise SystemExit("PARADA: la reconstruccion no reproduce V03")

    # Longitud, para el analisis por rebanada
    n_tok_txt = np.array([len(RE_TOKEN.findall(str(x))) for x in tx])
    rebanada = np.empty(len(y), dtype=object)
    for nombre, lo, hi in BINS:
        rebanada[(n_tok_txt >= lo) & (n_tok_txt <= hi)] = nombre

    res, comp, _probas = {}, {}, {}
    for dim in DIMS:
        vocab, W = cargar(dim)
        D, n_tok, n_cub = pool(tx, vocab, W)
        cv_emb, _p_emb, f1_emb = cv(des, tx, y, pl, D, 0.0) if False else cv(des, tx, y, pl)
        # embedding-only: el bloque denso ES la representacion, sin TF-IDF
        pr_emb = np.zeros((len(y), 3))
        for a, b in pl:
            sc = StandardScaler().fit(D[a])
            for j in range(3):
                m = LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000,
                                       solver="liblinear", random_state=SEMILLA)
                m.fit(sc.transform(D[a]), y[a, j])
                pr_emb[b, j] = m.predict_proba(sc.transform(D[b]))[:, 1]
        cv_emb = f1m(y, (pr_emb >= UMBRALES[None, :]).astype(int))

        variantes = {}
        for esc in ESCALAS:
            c2, p2, f2 = cv(des, tx, y, pl, D, esc)
            variantes[f"fusion_{esc:.2f}"] = {"cv": round(c2, 10),
                                              "delta": round(c2 - base, 10),
                                              "f1_pliegues": f2, "proba": p2}

        best = max(variantes.items(), key=lambda kv: kv[1]["cv"])
        pr_best = best[1]["proba"]
        prd = (pr_best >= UMBRALES[None, :]).astype(int)
        ar, ac = (pred_v03 == y), (prd == y)
        comp[dim] = {
            "rescue": int((~ar & ac).sum()), "damage": int((ar & ~ac).sum()),
            "net": int((~ar & ac).sum()) - int((ar & ~ac).sum()),
            "por_etiqueta": {ETIQUETAS[j]: {
                "V03_ok_cand_ko": int((ar[:, j] & ~ac[:, j]).sum()),
                "V03_ko_cand_ok": int((~ar[:, j] & ac[:, j]).sum()),
                "ambos_ok": int((ar[:, j] & ac[:, j]).sum()),
                "ambos_ko": int((~ar[:, j] & ~ac[:, j]).sum())} for j in range(3)}}
        por_reb = {}
        for nombre, _lo, _hi in BINS:
            m = rebanada == nombre
            if m.sum() < 10:
                continue
            por_reb[nombre] = {
                "filas": int(m.sum()),
                "v03": round(f1m(y[m], pred_v03[m]), 6),
                "embedding_only": round(f1m(y[m], (pr_emb[m] >= UMBRALES[None, :]).astype(int)),
                                        6),
                "fusion": round(f1m(y[m], prd[m]), 6),
                "delta_fusion": round(f1m(y[m], prd[m]) - f1m(y[m], pred_v03[m]), 6)}
        res[dim] = {
            "vocabulario": len(vocab), "dim": int(W.shape[1]),
            "cobertura_tokens": round(float(n_cub.sum() / max(n_tok.sum(), 1)), 6),
            "tokens_por_fila": round(float(n_tok.mean()), 2),
            "tokens_cubiertos": round(float(n_cub.mean()), 2),
            "filas_sin_token": int((n_cub == 0).sum()),
            "embedding_only_cv": round(cv_emb, 10),
            "embedding_only_delta": round(cv_emb - base, 10),
            "variantes": {k: {kk: vv for kk, vv in val.items() if kk != "proba"}
                          for k, val in variantes.items()},
            "mejor_variante": best[0],
            "por_rebanada": por_reb,
            "f1_pliegues_embedding": f1_emb,
        }
        log("")
        log(f"   {dim}d: vocab {len(vocab):,} | cobertura {100 * res[dim]['cobertura_tokens']:.1f}%"
            f" | {res[dim]['tokens_cubiertos']} de {res[dim]['tokens_por_fila']} tokens")
        log(f"      embedding-only {cv_emb:.6f}  delta {cv_emb - base:+.6f}")
        for k, val in variantes.items():
            log(f"      {k:14} {val['cv']:.6f}  delta {val['delta']:+.6f}")
        log(f"      complementariedad: rescue {comp[dim]['rescue']} | "
            f"damage {comp[dim]['damage']} | neto {comp[dim]['net']:+d}")

    log("")
    log("   LA PREGUNTA DE P1: EXISTE RELACION ENTRE DIMENSION Y TRANSFERENCIA?")
    log(f"      {'dim':>6}{'emb-only':>14}{'mejor fusion':>16}{'delta fusion':>16}")
    for dim in DIMS:
        r = res[dim]
        bv = r["variantes"][r["mejor_variante"]]
        log(f"      {dim:>5}d{r['embedding_only_cv']:>14.6f}{bv['cv']:>16.6f}"
            f"{bv['delta']:>+16.6f}")

    d_emb = max(res[d]["embedding_only_cv"] for d in DIMS) - \
        min(res[d]["embedding_only_cv"] for d in DIMS)
    d_fus = max(res[d]["variantes"][res[d]["mejor_variante"]]["cv"] for d in DIMS) - \
        min(res[d]["variantes"][res[d]["mejor_variante"]]["cv"] for d in DIMS)
    log("")
    log(f"      dispersion entre dimensiones: embedding-only {d_emb:.6f} | "
        f"mejor fusion {d_fus:.6f}")
    mejor_fus = max(res[d]["variantes"][res[d]["mejor_variante"]]["delta"] for d in DIMS)
    log(f"      mejor delta de fusion de las tres: {mejor_fus:+.6f}")
    if mejor_fus < 0.0015:
        veredicto = "DIMENSION_NO_LIMITANTE"
    elif d_emb > 0.0015:
        veredicto = "DIMENSION_RELEVANTE_CONGELAR"
    else:
        veredicto = "DIMENSION_NO_CONCLUYENTE"
    log(f"      VEREDICTO: {veredicto}")

    salida = {
        "familia": "P72_TWITTER_WORD2VEC_DIM", "experiment_id": "P72-02",
        "fase": "P1 downstream de las dimensiones",
        "fecha": inicio.isoformat(timespec="seconds"),
        "naturaleza": "EXPERIMENTO. 3 dimensiones, mean pooling en las tres",
        "v03_referencia": round(base, 10), "v03_reproducido": bool(abs(base - V03_CCV) < 1e-9),
        "resultados": res, "complementariedad": comp,
        "dispersion_embedding_only": round(d_emb, 10),
        "dispersion_mejor_fusion": round(d_fus, 10),
        "mejor_delta_fusion": round(mejor_fus, 10),
        "veredicto": veredicto,
        "reproducibilidad": {"nivel": "R2", "hash_v03": hash_v03, "corrected_cv_v03": V03_CCV},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    (REPORTS / "twitter_word2vec_dimension_results.json").write_text(
        json.dumps(salida, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with (REPORTS / "twitter_word2vec_dimension_results.csv").open(
            "w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["dim", "vocabulario", "cobertura", "embedding_only_cv",
                    "embedding_only_delta", "mejor_variante", "mejor_fusion_cv",
                    "mejor_fusion_delta", "rescue", "damage", "net"])
        w.writerow(["V03", "", "", round(base, 10), 0.0, "", round(base, 10), 0.0, "", "", ""])
        for dim in DIMS:
            r = res[dim]
            bv = r["variantes"][r["mejor_variante"]]
            w.writerow([f"{dim}d", r["vocabulario"], r["cobertura_tokens"],
                        r["embedding_only_cv"], r["embedding_only_delta"], r["mejor_variante"],
                        bv["cv"], bv["delta"], comp[dim]["rescue"], comp[dim]["damage"],
                        comp[dim]["net"]])
        w.writerow([])
        w.writerow(["rebanada", "dim", "filas", "v03", "embedding_only", "fusion",
                    "delta_fusion"])
        for dim in DIMS:
            for k, val in res[dim]["por_rebanada"].items():
                w.writerow([k, f"{dim}d", val["filas"], val["v03"], val["embedding_only"],
                            val["fusion"], val["delta_fusion"]])

    with (REPORTS / "twitter_word2vec_complementarity.csv").open(
            "w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["dim", "etiqueta", "V03_ok_cand_ko", "V03_ko_cand_ok", "ambos_ok",
                    "ambos_ko", "rescue", "damage", "net"])
        for dim in DIMS:
            for k, val in comp[dim]["por_etiqueta"].items():
                w.writerow([f"{dim}d", k, val["V03_ok_cand_ko"], val["V03_ko_cand_ok"],
                            val["ambos_ok"], val["ambos_ko"], val["V03_ko_cand_ok"],
                            val["V03_ok_cand_ko"],
                            val["V03_ko_cand_ok"] - val["V03_ok_cand_ko"]])
            w.writerow([f"{dim}d", "GLOBAL", "", "", "", "", comp[dim]["rescue"],
                        comp[dim]["damage"], comp[dim]["net"]])

    log("")
    log("=" * 78)
    log(f"P72-02 OK | {veredicto}")
    log(f"   mejor delta de fusion de las tres dimensiones: {mejor_fus:+.6f}")
    log("=" * 78)


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
