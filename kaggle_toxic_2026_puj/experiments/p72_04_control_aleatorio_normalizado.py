"""El control de aleatoriedad con el pipeline de P1 reconstruido.

Por que existe este script
--------------------------
El pre-registro (docs/p72_preregistro_control_aleatorio.md, commit 7e1aba4) exigia que el brazo
entrenado reprodujera la referencia historica 0.594729. NO la reprodujo: dio 0.603719.

El pre-registro dice que la desviacion se reporta y NO se reintenta. Se reporta. Pero reportar una
desviacion sin diagnosticarla dejaria el control desconectado de la medicion que dice controlar,
asi que se diagnostico la causa.

Causa: al reconstruir el pipeline de P1 se omitio un paso. P1 estandarizaba el bloque denso con
StandardScaler ajustado DENTRO del pliegue de entrenamiento (experiments/p72_02_dimension_downstream.py
linea 199), antes de la regresion logistica. p72_03 no lo hacia.

Este script no es un reintento hasta que cuadre: es el mismo control con el paso que faltaba, en
las DOS ramas, y con las MISMAS diez semillas ya pre-declaradas. Si el diagnostico es correcto, el
brazo entrenado reproducira 0.5947288906 al decimal.

Se declara como desviacion del pre-registro en el JSON de salida.
"""

from __future__ import annotations

import ast
import csv as _csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from p72_03_control_aleatorio import (
    MODELOS,
    N_BOOTSTRAP,
    REFERENCIA_HISTORICA,
    REPORTS,
    SEMILLA_BOOT,
    SEMILLA_LR,
    SEMILLAS_NULAS,
    UMBRALES,
    bootstrap_agrupado,
    f1m,
    log,
    pool,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from src.data import read_train
from src.splits import grupos_fusionados, holdout_v2, pliegues_corregidos


def brazo_norm(docs, y, pl, vocab, W):
    """Downstream con el StandardScaler dentro del pliegue, como en P1.

    Args:
        docs: textos.
        y: etiquetas.
        pl: pliegues.
        vocab: indice.
        W: matriz de embeddings.

    Returns:
        Diccionario con el resultado.
    """
    P, n_tok, n_cub = pool(docs, vocab, W)
    pr = np.zeros((len(y), 3))
    f1s = []
    for a, b in pl:
        sc = StandardScaler().fit(P[a])
        Pa, Pb = sc.transform(P[a]), sc.transform(P[b])
        for j in range(3):
            m = LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000,
                                   solver="liblinear", random_state=SEMILLA_LR)
            m.fit(Pa, y[a, j])
            pr[b, j] = m.predict_proba(Pb)[:, 1]
        f1s.append(round(f1m(y[b], (pr[b] >= UMBRALES[None, :]).astype(int)), 6))
    pred = (pr >= UMBRALES[None, :]).astype(int)
    aps = [float(average_precision_score(y[:, j], pr[:, j])) for j in range(3)]
    return {"cv": round(f1m(y, pred), 10), "f1_pliegues": f1s, "proba": pr,
            "tasa_positivos": [round(float(pred[:, j].mean()), 6) for j in range(3)],
            "pr_auc": [round(x, 6) for x in aps],
            "tokens_por_fila": round(float(n_tok.mean()), 2),
            "tokens_cubiertos": round(float(n_cub.mean()), 2)}


def main() -> None:
    """Repite el control con el pipeline de P1, en las dos ramas.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("")
    log("P72-04 el control, con el pipeline de P1 reconstruido")
    log("   desviacion declarada: se anade el StandardScaler dentro del pliegue que p72_03 omitio")

    v = np.load(Path(__file__).resolve().parents[1] / "features" / "oof_v03_final.npz",
                allow_pickle=True)
    y = v["y_true"].astype(int)
    ids = v["ids"].astype(np.int64)
    tr = read_train()
    g = grupos_fusionados(tr)
    idev, _ = holdout_v2(tr, g)
    des = tr.iloc[idev].reset_index(drop=True)
    if not np.array_equal(des["id"].to_numpy(dtype=np.int64), ids):
        raise SystemExit("PARADA: los ids no coinciden con el OOF canonico")
    pl = pliegues_corregidos(des, g[idev])
    tx = des["text"]
    grupos = g[idev]

    d = np.load(MODELOS / "w2v_twitter_100d.npz", allow_pickle=True)
    vocab = {w: i for i, w in enumerate(d["palabras"])}
    W_train = d["W"]
    normas = np.linalg.norm(W_train, axis=1)

    t0 = time.time()
    A = brazo_norm(tx, y, pl, vocab, W_train)
    desv = A["cv"] - REFERENCIA_HISTORICA
    log("")
    log(f"   BRAZO A, entrenado, con estandarizacion: {A['cv']:.10f} | {time.time() - t0:.0f}s")
    log(f"      referencia historica {REFERENCIA_HISTORICA} | desviacion {desv:+.10f}")
    log(f"      EL DIAGNOSTICO ES CORRECTO: {abs(desv) <= 1e-9}")
    log(f"      f1 por pliegue {A['f1_pliegues']}")

    log("")
    log("   BRAZO B, las diez nulas, con estandarizacion, todas reportadas")
    nulas = []
    for k, sem in enumerate(SEMILLAS_NULAS, start=1):
        t0 = time.time()
        rng = np.random.default_rng(sem)
        U = rng.standard_normal(W_train.shape)
        U /= np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)
        W_rand = (normas[:, None] * U).astype(np.float32)
        ok_norma = bool(np.allclose(np.linalg.norm(W_rand, axis=1), normas, atol=1e-4))
        B = brazo_norm(tx, y, pl, vocab, W_rand)
        nulas.append({"tirada": k, "semilla": sem, "cv": B["cv"],
                      "f1_pliegues": B["f1_pliegues"], "pr_auc": B["pr_auc"],
                      "tasa_positivos": B["tasa_positivos"], "normas_conservadas": ok_norma,
                      "segundos": round(time.time() - t0, 1)})
        log(f"      tirada {k:2d} semilla {sem}: {B['cv']:.6f} | normas conservadas {ok_norma}")

    cvs = np.array([x["cv"] for x in nulas])
    max_nula = float(cvs.max())
    delta = A["cv"] - max_nula
    log("")
    log("   LA NULA, RESUMIDA")
    log(f"      media {cvs.mean():.6f} | desviacion {cvs.std(ddof=1):.6f} | "
        f"minimo {cvs.min():.6f} | maximo {max_nula:.6f}")
    log(f"      el entrenado {A['cv']:.6f} frente al MAXIMO de las diez: delta {delta:+.6f}")
    z = (A["cv"] - cvs.mean()) / max(cvs.std(ddof=1), 1e-12)
    log(f"      el entrenado esta a {z:+.2f} desviaciones de la media de la nula")

    pos = 0
    for k in range(5):
        a_k = A["f1_pliegues"][k]
        b_k = max(x["f1_pliegues"][k] for x in nulas)
        if a_k > b_k:
            pos += 1
    log(f"   POR PLIEGUE, A contra el MAXIMO de las diez en ese pliegue: {pos} de 5")

    W_best = None
    for sem in SEMILLAS_NULAS:
        if sem == nulas[int(cvs.argmax())]["semilla"]:
            rng = np.random.default_rng(sem)
            U = rng.standard_normal(W_train.shape)
            U /= np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)
            W_best = (normas[:, None] * U).astype(np.float32)
    B_best = brazo_norm(tx, y, pl, vocab, W_best)
    media_b, lo_b, hi_b = bootstrap_agrupado(
        y, (A["proba"] >= UMBRALES[None, :]).astype(int),
        (B_best["proba"] >= UMBRALES[None, :]).astype(int), grupos,
        semilla=SEMILLA_BOOT, n=N_BOOTSTRAP)
    semi = (hi_b - lo_b) / 2
    log(f"      bootstrap agrupado del delta contra el maximo: {media_b:+.6f} "
        f"[{lo_b:+.6f}, {hi_b:+.6f}] | semianchura {semi:.6f}")

    ad = abs(delta)
    if ad < 0.005:
        veredicto = "EMPATE"
    elif ad < 0.015:
        veredicto = "SENAL_DEBIL"
    elif delta > 0 and pos >= 4:
        veredicto = "EFECTO_CLARO"
    else:
        veredicto = "SIN_EFECTO_CLARO"
    aviso = ("la semianchura del bootstrap supera 0,015, luego la banda de efecto claro es "
             "inalcanzable por construccion"
             if semi > 0.015 else
             "la semianchura del bootstrap no supera la banda de 0,015")
    log("")
    log(f"   VEREDICTO: {veredicto} | delta {delta:+.6f} | pliegues {pos}/5")
    log(f"   {aviso}")

    salida = {
        "familia": "P72_TWITTER_WORD2VEC_DIM", "experiment_id": "P72-04",
        "fase": "control de aleatoriedad, pipeline de P1 reconstruido",
        "fecha": inicio.isoformat(timespec="seconds"),
        "naturaleza": "CONTROL. 11 corridas downstream, 0 entrenamientos de embeddings",
        "desviacion_del_preregistro": {
            "documento": "docs/p72_preregistro_control_aleatorio.md", "commit": "7e1aba4",
            "que_paso": ("el brazo entrenado de p72_03 dio 0.6037188662 y NO reprodujo la "
                         "referencia 0.594729, como el pre-registro exigia"),
            "diagnostico": ("p72_03 omitio el StandardScaler ajustado dentro del pliegue de "
                            "entrenamiento que P1 aplicaba al bloque denso "
                            "(experiments/p72_02_dimension_downstream.py linea 199)"),
            "accion": ("se repite el control con ese paso, en las DOS ramas, con las MISMAS diez "
                       "semillas ya pre-declaradas. No es un reintento hasta que cuadre: es "
                       "corregir un pipeline mal reconstruido"),
            "diagnostico_confirmado": bool(abs(desv) <= 1e-9),
            "consecuencia": ("p72_03 sigue siendo valido como contraste INTERNO entre sus dos "
                             "ramas, que comparten codigo. Su unica limitacion era no conectar "
                             "con la medicion historica que decia controlar")},
        "brazo_a": {k: v2 for k, v2 in A.items() if k != "proba"},
        "referencia_historica": REFERENCIA_HISTORICA,
        "desviacion_respecto_de_la_referencia": round(desv, 12),
        "nulas": nulas,
        "nula_resumen": {"media": round(float(cvs.mean()), 10),
                         "desviacion": round(float(cvs.std(ddof=1)), 10),
                         "minimo": round(float(cvs.min()), 10),
                         "maximo": round(max_nula, 10)},
        "delta_contra_el_maximo": round(delta, 10),
        "z_contra_la_media_de_la_nula": round(float(z), 6),
        "pliegues_contra_el_maximo": pos,
        "bootstrap_agrupado": {"media": media_b, "ic95_bajo": lo_b, "ic95_alto": hi_b,
                              "semianchura": round(float(semi), 10),
                              "grupos": len(np.unique(grupos))},
        "aviso_de_potencia": aviso,
        "veredicto": veredicto,
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    (REPORTS / "twitter_word2vec_control_aleatorio_normalizado.json").write_text(
        json.dumps(salida, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with (REPORTS / "twitter_word2vec_control_aleatorio_normalizado.csv").open(
            "w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["brazo", "tirada", "semilla", "cv", "delta_vs_A", "normas_conservadas"])
        w.writerow(["A_entrenado", "", "", A["cv"], 0.0, ""])
        for x in nulas:
            w.writerow(["B_aleatorio", x["tirada"], x["semilla"], x["cv"],
                        round(x["cv"] - A["cv"], 10), x["normas_conservadas"]])
        w.writerow([])
        w.writerow(["nula", "media", "", round(float(cvs.mean()), 10), "", ""])
        w.writerow(["nula", "desviacion", "", round(float(cvs.std(ddof=1)), 10), "", ""])
        w.writerow(["nula", "maximo", "", round(max_nula, 10), "", ""])
        w.writerow(["decision", veredicto, "", round(delta, 10), "", ""])

    docs = [REPORTS / "twitter_word2vec_control_aleatorio_normalizado.json",
            REPORTS / "twitter_word2vec_control_aleatorio_normalizado.csv"]
    sucios = [x.name for x in docs if any(b > 127 for b in x.read_bytes())]
    if sucios:
        raise SystemExit(f"PARADA: documentos con bytes no ASCII: {sucios}")

    log("")
    log("=" * 78)
    log(f"P72-04 OK | {veredicto}")
    log(f"   A {A['cv']:.6f} (historico reproducido: {abs(desv) <= 1e-9})")
    log(f"   nula: media {cvs.mean():.6f} maximo {max_nula:.6f} | delta {delta:+.6f} | "
        f"pliegues {pos}/5")
    log("=" * 78)


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
