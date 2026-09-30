"""P1d: downstream con los modelos que SI entrenaron, y con la fusion corregida.

Dos cosas que cambian respecto del downstream invalidado
--------------------------------------------------------
UNO. Los embeddings. Los de la corrida anterior eran practicamente aleatorios porque el
learning rate no estaba escalado por el lote. Estos vienen de la corrida corregida, donde la
loss baja de 1,272 a 1,095 en twitter y de 1,254 a 1,056 en general.

DOS. La fusion. El downstream anterior concatenaba 200 columnas densas estandarizadas a un
bloque disperso de 24.646 dimensiones. Con `StandardScaler` las densas tienen varianza unidad y
las TF-IDF valores pequenos, asi que el bloque denso domina la penalizacion L2 y la regresion
se sobreajusta a el. Eso produjo la caida de 0,78 a 0,64, que no es un hallazgo sobre el
embedding sino un defecto del diseno.

Aqui se corre la fusion de DOS formas:

    estandarizada   la del informe anterior, para poder comparar
    normalizada     el producto concatenado se normaliza por filas, que pone los dos bloques
                    en pie de igualdad y es la practica estandar

La pregunta del encargo se responde con el CONTROL DE DOMINIO, que es la comparacion entre
twitter y general bajo el mismo pipeline. Esa comparacion es limpia en las dos formas.
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
from sklearn.preprocessing import StandardScaler, normalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import V03_CCV, V03_SHA, V03_SUB

from src.evaluation import metricas_multietiqueta
from src.splits import ETIQUETAS, grupos_fusionados, holdout_v2, pliegues_corregidos

REPORTS = PROJECT_ROOT / "reports"
MODELOS = PROJECT_ROOT / "modelos" / "twitter_domain"
UMBRALES = np.array([0.44, 0.51, 0.46])
RE_TOKEN = re.compile(r"[#@]?\w+(?:['\u2019]\w+)*", re.UNICODE)
SEMILLA = 42


def log(m: str) -> None:
    """Imprime una linea de progreso.

    Args:
        m: texto a mostrar.
    """
    print(f"[P71-09] {m}", flush=True)


def f1m(y: np.ndarray, pred: np.ndarray) -> float:
    """Macro F1 del proyecto.

    Args:
        y: etiquetas.
        pred: predicciones.

    Returns:
        La media de las F1 por columna.
    """
    return float(metricas_multietiqueta(y, pred)["f1_media_columnas"])


def cargar(etiqueta: str) -> tuple[dict[str, int], np.ndarray]:
    """Carga un modelo corregido.

    Args:
        etiqueta: nombre del corpus.

    Returns:
        Indice y matriz de embeddings.
    """
    d = np.load(MODELOS / f"w2v_{etiqueta}_lrfix.npz", allow_pickle=True)
    return {w: i for i, w in enumerate(d["palabras"])}, d["W"]


def pool(textos, vocab: dict[str, int], W: np.ndarray) -> np.ndarray:
    """Mean pooling.

    Args:
        textos: serie de textos.
        vocab: indice.
        W: embeddings.

    Returns:
        Matriz (n, dim).
    """
    out = np.zeros((len(textos), W.shape[1]), dtype=np.float32)
    for i, t in enumerate(textos):
        ids = [vocab[w] for w in RE_TOKEN.findall(str(t)) if w in vocab]
        if ids:
            out[i] = W[ids].mean(axis=0)
    return out


def cv(X, y, pliegues) -> tuple[np.ndarray, list[float]]:
    """OOF del LR con los pliegues canonicos.

    Args:
        X: caracteristicas.
        y: etiquetas.
        pliegues: pares entrenamiento-validacion.

    Returns:
        Probabilidades y F1 por pliegue.
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
    """Ejecuta el downstream con los modelos corregidos.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P71-09 P1d: downstream con los modelos corregidos y la fusion arreglada")
    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")

    from sklearn.feature_extraction.text import TfidfVectorizer

    from src.data import read_train

    v = np.load(PROJECT_ROOT / "features" / "oof_v03_final.npz", allow_pickle=True)
    y = v["y_true"].astype(int)
    ids = v["ids"].astype(np.int64)
    pred_v03 = (v["proba"] >= UMBRALES[None, :]).astype(int)
    cv_v03 = f1m(y, pred_v03)
    train = read_train()
    grupos = grupos_fusionados(train)
    idev, _ = holdout_v2(train, grupos)
    desarrollo = train.iloc[idev].reset_index(drop=True)
    if not np.array_equal(desarrollo["id"].to_numpy(dtype=np.int64), ids):
        raise SystemExit("PARADA: ids distintos")
    pliegues = pliegues_corregidos(desarrollo, grupos[idev])
    textos = desarrollo["text"]
    log(f"   V03: {cv_v03:.10f} | {len(y)} filas")

    pal = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    car = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
    X_v03 = sparse.hstack([pal.fit_transform(textos), car.fit_transform(textos)],
                          format="csr")
    log(f"   representacion de V03: {X_v03.shape[1]:,} dimensiones")

    res, comp, probas = {"v03": round(cv_v03, 10)}, {}, {}
    for etiqueta in ("twitter", "general"):
        t0 = time.time()
        vocab, W = cargar(etiqueta)
        P = pool(textos, vocab, W)
        no_cero = int((np.abs(P).sum(axis=1) > 0).sum())
        p_emb, _ = cv(P, y, pliegues)
        cv_emb = f1m(y, (p_emb >= UMBRALES[None, :]).astype(int))

        variantes = {}
        # forma anterior: densas estandarizadas y concatenadas sin mas
        Pe = StandardScaler().fit_transform(P).astype(np.float32)
        Xe = sparse.hstack([X_v03, sparse.csr_matrix(Pe)], format="csr")
        pe, f1e = cv(Xe, y, pliegues)
        variantes["fusion_estandarizada"] = {
            "cv": round(f1m(y, (pe >= UMBRALES[None, :]).astype(int)), 10),
            "f1_pliegues": f1e, "proba": pe}
        # forma corregida: el producto concatenado se normaliza por filas
        Xn = normalize(sparse.hstack([X_v03, sparse.csr_matrix(Pe)], format="csr"))
        pn, f1n = cv(Xn, y, pliegues)
        variantes["fusion_normalizada"] = {
            "cv": round(f1m(y, (pn >= UMBRALES[None, :]).astype(int)), 10),
            "f1_pliegues": f1n, "proba": pn}

        for k, val in variantes.items():
            val["delta"] = round(val["cv"] - cv_v03, 10)
        probas[etiqueta] = variantes["fusion_normalizada"]["proba"]
        res[etiqueta] = {"vocabulario": len(vocab), "filas_con_vector": no_cero,
                         "embedding_only_cv": round(cv_emb, 10),
                         "embedding_only_delta": round(cv_emb - cv_v03, 10),
                         "variantes": {k: {kk: vv for kk, vv in val.items() if kk != "proba"}
                                       for k, val in variantes.items()},
                         "segundos": round(time.time() - t0, 1)}
        log("")
        log(f"   {etiqueta.upper()}")
        log(f"      embedding-only    {cv_emb:.10f}  delta {cv_emb - cv_v03:+.10f}")
        for k, val in variantes.items():
            log(f"      {k:18} {val['cv']:.10f}  delta {val['delta']:+.10f}")
        log(f"      f1 pliegues (normalizada): {variantes['fusion_normalizada']['f1_pliegues']}")

    # --- control de dominio, bajo las dos formas de fusion ---
    dom = {}
    for k in ("embedding_only", "fusion_estandarizada", "fusion_normalizada"):
        if k == "embedding_only":
            a = res["twitter"]["embedding_only_cv"]
            b = res["general"]["embedding_only_cv"]
        else:
            a = res["twitter"]["variantes"][k]["cv"]
            b = res["general"]["variantes"][k]["cv"]
        dom[k] = {"twitter": a, "general": b, "delta": round(a - b, 10),
                  "ganador": "TWITTER" if a > b else "GENERAL"}
        log("")
        log(f"   dominio {k:20} twitter {a:.6f} | general {b:.6f} -> {a - b:+.6f} "
            f"({dom[k]['ganador']})")

    # --- complementariedad con la fusion normalizada ---
    for etiqueta in ("twitter", "general"):
        pr = (probas[etiqueta] >= UMBRALES[None, :]).astype(int)
        ar, ac = (pred_v03 == y), (pr == y)
        comp[etiqueta] = {
            "rescue": int((~ar & ac).sum()), "damage": int((ar & ~ac).sum()),
            "net": int((~ar & ac).sum()) - int((ar & ~ac).sum()),
            "por_etiqueta": {ETIQUETAS[j]: {
                "rescue": int((~ar[:, j] & ac[:, j]).sum()),
                "damage": int((ar[:, j] & ~ac[:, j]).sum())} for j in range(3)}}
        log(f"   complementariedad {etiqueta}: rescue {comp[etiqueta]['rescue']} | "
            f"damage {comp[etiqueta]['damage']} | neto {comp[etiqueta]['net']:+d}")

    # --- gate, con la fusion normalizada ---
    gates = {}
    ref_f1 = [round(f1m(y[va], pred_v03[va]), 6) for _, va in pliegues]
    for etiqueta in ("twitter", "general"):
        f1s = res[etiqueta]["variantes"]["fusion_normalizada"]["f1_pliegues"]
        pos = sum(1 for a, b in zip(f1s, ref_f1) if a > b)
        d = res[etiqueta]["variantes"]["fusion_normalizada"]["delta"]
        ok = bool(d >= 0.0015 and pos >= 4 and comp[etiqueta]["net"] > 0)
        gates[etiqueta] = {"delta": d, "pliegues_positivos": pos, "neto":
                           comp[etiqueta]["net"], "pasa": ok}
        log(f"   gate {etiqueta}: delta {d:+.6f} | pliegues {pos}/5 | "
            f"neto {comp[etiqueta]['net']:+d} | PASA {ok}")

    res["twitter"]
    dom_emb = dom["embedding_only"]["delta"]
    dom_fus = dom["fusion_normalizada"]["delta"]
    if gates["twitter"]["pasa"] and dom_fus > 0:
        veredicto = "PROMISING"
    elif dom_emb <= 0 and dom_fus <= 0:
        veredicto = "CLOSED"
    else:
        veredicto = "MARGINAL"
    log("")
    log(f"   VEREDICTO: {veredicto}")

    salida = {
        "familia": "P71_TWITTER_WORD2VEC", "experiment_id": "P71-09",
        "fase": "P1d downstream con modelos corregidos",
        "fecha": inicio.isoformat(timespec="seconds"),
        "naturaleza": ("EXPERIMENTO. Modelos con learning rate corregido. 0 test, 0 holdout, "
                       "0 leaderboard"),
        "por_que_se_reejecuta": ("los embeddings del downstream anterior eran casi aleatorios: "
                                 "el learning rate no estaba escalado por el lote"),
        "fusion_arreglada": ("se corre la fusion estandarizada de antes, para comparar, y una "
                             "fusion normalizada por filas que pone los dos bloques en pie de "
                             "igualdad"),
        "v03_referencia": round(cv_v03, 10),
        "resultados": res, "control_de_dominio": dom,
        "complementariedad": comp, "gates": gates, "veredicto": veredicto,
        "decision_del_encargo": {
            "A_twitter_mejora_claramente": bool(dom_fus > 0 and gates["twitter"]["pasa"]),
            "B_mejora_embedding_pero_no_fusion": bool(
                dom_emb > 0 and res["twitter"]["variantes"]["fusion_normalizada"]["delta"]
                < 0.0015),
            "C_no_mejora_al_general": bool(dom_emb <= 0 and dom_fus <= 0),
            "D_empeora": bool(res["twitter"]["variantes"]["fusion_normalizada"]["delta"] < 0),
            "resultado": veredicto},
        "reproducibilidad": {"nivel": "R2", "hash_v03": hash_real, "corrected_cv_v03": V03_CCV},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    (REPORTS / "twitter_word2vec_domain_results_lrfix.json").write_text(
        json.dumps(salida, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    import csv as _csv
    with (REPORTS / "twitter_word2vec_domain_results_lrfix.csv").open(
            "w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["corpus", "vocabulario", "embedding_only_cv", "embedding_delta",
                    "fusion_estandarizada_cv", "fusion_estandarizada_delta",
                    "fusion_normalizada_cv", "fusion_normalizada_delta",
                    "rescue", "damage", "net", "pasa_gate"])
        w.writerow(["V03", "", round(cv_v03, 10), 0.0, round(cv_v03, 10), 0.0,
                    round(cv_v03, 10), 0.0, "", "", "", ""])
        for e in ("twitter", "general"):
            r = res[e]
            w.writerow([e, r["vocabulario"], r["embedding_only_cv"], r["embedding_only_delta"],
                        r["variantes"]["fusion_estandarizada"]["cv"],
                        r["variantes"]["fusion_estandarizada"]["delta"],
                        r["variantes"]["fusion_normalizada"]["cv"],
                        r["variantes"]["fusion_normalizada"]["delta"],
                        comp[e]["rescue"], comp[e]["damage"], comp[e]["net"],
                        gates[e]["pasa"]])

    log("")
    log("=" * 78)
    log(f"P71-09 OK | {veredicto}")
    log(f"   dominio embedding-only {dom_emb:+.6f} | fusion normalizada {dom_fus:+.6f}")
    log("=" * 78)


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
