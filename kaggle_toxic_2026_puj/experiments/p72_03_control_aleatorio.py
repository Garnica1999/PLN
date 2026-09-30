"""El control de aleatoriedad: Word2Vec entrenado frente a vectores aleatorios congelados.

El pre-registro va ANTES, en docs/p72_preregistro_control_aleatorio.md, commiteado en 7e1aba4.
Este script lo ejecuta tal como se declaro alli, sin desviarse.

Lo que hay que tener presente al leer el resultado
--------------------------------------------------
Con mean pooling y cabeza lineal, las dos ramas son la MISMA FAMILIA FUNCIONAL: un clasificador
lineal sobre la bolsa de tokens normalizada, con coeficientes en un subespacio de dimension menor
o igual que 100. Lo unico que cambia es CUAL subespacio: el aprendido de las co-ocurrencias del
corpus, o uno aleatorio con la misma norma por token.

Luego un empate significa "la geometria aprendida no aporta a ESTE lector", no "no hay senal".

Y hay una asimetria que va a favor de la honestidad del control: como el clasificador reajusta
sus coeficientes, un subespacio aleatorio puede capturar bastante senal de frecuencia o palabra
suelta. La nula puede ser fuerte y el control puede fallar en detectar una contribucion real. Un
empate es evidencia DEBIL; una victoria clara del entrenado es evidencia FUERTE.
"""

from __future__ import annotations

import ast
import csv as _csv
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import V03_CCV, V03_SHA, V03_SUB

from src.data import read_train
from src.evaluation import metricas_multietiqueta
from src.splits import grupos_fusionados, holdout_v2, pliegues_corregidos

REPORTS = PROJECT_ROOT / "reports"
MODELOS = PROJECT_ROOT / "modelos" / "twitter_domain"
UMBRALES = np.array([0.44, 0.51, 0.46])
RE_TOKEN = re.compile(r"[#@]?\w+(?:['\u2019]\w+)*", re.UNICODE)
SEMILLA_LR = 42
SEMILLAS_NULAS = (20261020, 20261021, 20261022, 20261023, 20261024,
                  20261025, 20261026, 20261027, 20261028, 20261029)
N_BOOTSTRAP = 2000
SEMILLA_BOOT = 20261030
TOL = 1e-6
REFERENCIA_HISTORICA = 0.594729


def log(m: str) -> None:
    """Imprime una linea de progreso.

    Args:
        m: texto a mostrar.
    """
    print(f"[P72-03] {m}", flush=True)


def f1m(y: np.ndarray, pred: np.ndarray) -> float:
    """Macro F1 del proyecto.

    Args:
        y: etiquetas.
        pred: predicciones.

    Returns:
        La media de las F1 por columna.
    """
    return float(metricas_multietiqueta(y, pred)["f1_media_columnas"])


def pool(textos, vocab, W):
    """Mean pooling. El denominador son los tokens presentes en vocabulario.

    Args:
        textos: textos.
        vocab: indice.
        W: matriz de embeddings.

    Returns:
        Matriz y contadores.
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


def brazo(docs, y, pl, vocab, W):
    """Corre el downstream completo y devuelve todo lo que el pre-registro pide.

    Args:
        docs: textos.
        y: etiquetas.
        pl: pliegues.
        vocab: indice.
        W: matriz.

    Returns:
        Diccionario con el resultado.
    """
    P, n_tok, n_cub = pool(docs, vocab, W)
    pr = np.zeros((len(y), 3))
    f1s = []
    for a, b in pl:
        for j in range(3):
            m = LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000,
                                   solver="liblinear", random_state=SEMILLA_LR)
            m.fit(P[a], y[a, j])
            pr[b, j] = m.predict_proba(P[b])[:, 1]
        f1s.append(round(f1m(y[b], (pr[b] >= UMBRALES[None, :]).astype(int)), 6))
    pred = (pr >= UMBRALES[None, :]).astype(int)
    aps = [float(average_precision_score(y[:, j], pr[:, j])) for j in range(3)]
    return {"cv": round(f1m(y, pred), 10), "f1_pliegues": f1s, "proba": pr,
            "tasa_positivos": [round(float(pred[:, j].mean()), 6) for j in range(3)],
            "pr_auc": [round(x, 6) for x in aps],
            "norma_fila_media": round(float(np.linalg.norm(P, axis=1).mean()), 6),
            "tokens_por_fila": round(float(n_tok.mean()), 2),
            "tokens_cubiertos": round(float(n_cub.mean()), 2),
            "filas_sin_token": int((n_cub == 0).sum())}


def bootstrap_agrupado(y, pred_a, pred_b, grupos, semilla=SEMILLA_BOOT, n=N_BOOTSTRAP):
    """Bootstrap agrupado del delta, remuestreando grupos.

    Args:
        y: etiquetas.
        pred_a: predicciones del brazo A.
        pred_b: predicciones del brazo B.
        grupos: vector de grupo por fila.
        semilla: semilla.
        n: remuestreos.

    Returns:
        La media y los dos extremos del IC95.
    """
    u = np.unique(grupos)
    por_g = {g: np.where(grupos == g)[0] for g in u}
    rng = np.random.default_rng(semilla)
    d = np.empty(n)
    for k in range(n):
        idx = np.concatenate([por_g[x] for x in rng.choice(u, len(u), True)])
        d[k] = f1m(y[idx], pred_a[idx]) - f1m(y[idx], pred_b[idx])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return round(float(d.mean()), 10), round(float(lo), 10), round(float(hi), 10)


def main() -> None:
    """Ejecuta el control tal como se pre-registro.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P72-03 control de aleatoriedad: entrenado frente a 10 nulas")
    log("   pre-registro: docs/p72_preregistro_control_aleatorio.md, commit 7e1aba4")
    hash_v03 = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_v03 != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")

    v = np.load(PROJECT_ROOT / "features" / "oof_v03_final.npz", allow_pickle=True)
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
    log(f"   {len(y)} filas | {len(np.unique(grupos))} grupos | 5 pliegues")

    # ---------- el vocabulario y las normas, del corpus externo ----------
    d = np.load(MODELOS / "w2v_twitter_100d.npz", allow_pickle=True)
    vocab = {w: i for i, w in enumerate(d["palabras"])}
    W_train = d["W"]
    normas = np.linalg.norm(W_train, axis=1)
    sha_W = hashlib.sha256(W_train.tobytes()).hexdigest()[:32]
    log(f"   vocabulario {len(vocab):,} | dimension {W_train.shape[1]} | "
        f"sha de la matriz {sha_W}")

    # Controles de integridad que la auditoria ML/DL pidio
    cos = []
    sub = np.random.default_rng(1).choice(len(W_train), size=min(3000, len(W_train)),
                                          replace=False)
    Wn = W_train[sub] / np.maximum(np.linalg.norm(W_train[sub], axis=1, keepdims=True), 1e-12)
    C = Wn @ Wn.T
    cos = float((C.sum() - len(sub)) / (len(sub) * (len(sub) - 1)))
    log(f"   coseno medio entre vectores de token: {cos:+.4f}")
    log("   (si fuera muy alto habria anisotropia; la auditoria pedia medirlo)")

    # ---------- el brazo A: el entrenado, UNA sola corrida ----------
    t0 = time.time()
    A = brazo(tx, y, pl, vocab, W_train)
    log("")
    log(f"   BRAZO A, el entrenado: {A['cv']:.10f} | {time.time() - t0:.0f}s")
    desv = A["cv"] - REFERENCIA_HISTORICA
    log(f"      desviacion respecto de la referencia historica {REFERENCIA_HISTORICA}: "
        f"{desv:+.10f}")
    log(f"      reproduce dentro de {TOL}: {abs(desv) <= TOL}")
    log(f"      f1 por pliegue {A['f1_pliegues']}")
    log(f"      pr-auc por etiqueta {A['pr_auc']}")
    log(f"      tasa de positivos {A['tasa_positivos']}")

    # ---------- los 10 nulos ----------
    log("")
    log("   BRAZO B, LAS DIEZ NULAS, todas reportadas")
    nulas = []
    for k, sem in enumerate(SEMILLAS_NULAS, start=1):
        t0 = time.time()
        rng = np.random.default_rng(sem)
        U = rng.standard_normal(W_train.shape)
        U /= np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)
        W_rand = (normas[:, None] * U).astype(np.float32)
        # Controles de integridad de la matriz aleatoria
        ok_fro = float(np.linalg.norm(W_rand - W_train)) > 0
        ok_norma = bool(np.allclose(np.linalg.norm(W_rand, axis=1), normas, atol=1e-4))
        B = brazo(tx, y, pl, vocab, W_rand)
        nulas.append({"tirada": k, "semilla": sem, "cv": B["cv"],
                      "f1_pliegues": B["f1_pliegues"], "pr_auc": B["pr_auc"],
                      "tasa_positivos": B["tasa_positivos"],
                      "matriz_distinta": ok_fro, "normas_conservadas": ok_norma,
                      "segundos": round(time.time() - t0, 1)})
        log(f"      tirada {k:2d} semilla {sem}: {B['cv']:.6f} | "
            f"matriz distinta {ok_fro} | normas conservadas {ok_norma} | "
            f"{time.time() - t0:.0f}s")

    cvs = np.array([x["cv"] for x in nulas])
    max_nula = float(cvs.max())
    delta = A["cv"] - max_nula
    log("")
    log("   LA NULA, RESUMIDA")
    log(f"      media {cvs.mean():.6f} | desviacion {cvs.std(ddof=1):.6f} | "
        f"minimo {cvs.min():.6f} | maximo {max_nula:.6f}")
    log(f"      el entrenado {A['cv']:.6f} frente al MAXIMO de las diez: delta {delta:+.6f}")

    # ---------- las anclas ----------
    prev = y.mean(axis=0)
    piso = [float(2 * p / (1 + p)) for p in prev]
    rng = np.random.default_rng(SEMILLA_BOOT)
    uniforme = rng.integers(0, 2, size=y.shape)
    anclas = {"prevalencia": [round(float(p), 6) for p in prev],
              "piso_all_positivo_por_etiqueta": [round(x, 6) for x in piso],
              "piso_all_positivo_macro": round(float(np.mean(piso)), 6),
              "predicciones_uniformes_macro": round(f1m(y, uniforme), 6)}
    log("")
    log("   ANCLAS")
    for k, val in anclas.items():
        log(f"      {k:32} {val}")

    # ---------- el pliegue a pliegue, definido en el pre-registro ----------
    pos = 0
    for k in range(5):
        a_k = A["f1_pliegues"][k]
        b_k = max(x["f1_pliegues"][k] for x in nulas)
        if a_k > b_k:
            pos += 1
    log("")
    log(f"   POR PLIEGUE, A contra el MAXIMO de las diez en ese pliegue: {pos} de 5")

    # ---------- bootstrap agrupado ----------
    t0 = time.time()
    mejor = nulas[int(cvs.argmax())]
    rng = np.random.default_rng(mejor["semilla"])
    U = rng.standard_normal(W_train.shape)
    U /= np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)
    W_best = (normas[:, None] * U).astype(np.float32)
    B_best = brazo(tx, y, pl, vocab, W_best)
    media_b, lo_b, hi_b = bootstrap_agrupado(y, (A["proba"] >= UMBRALES[None, :]).astype(int),
                                             (B_best["proba"] >= UMBRALES[None, :]).astype(int),
                                             grupos)
    semi = (hi_b - lo_b) / 2
    log(f"      bootstrap agrupado del delta contra el maximo: {media_b:+.6f} "
        f"[{lo_b:+.6f}, {hi_b:+.6f}] | semianchura {semi:.6f} ({time.time() - t0:.0f}s)")

    # ---------- la decision, mecanica ----------
    ad = abs(delta)
    if ad < 0.005:
        veredicto = "EMPATE"
    elif ad < 0.015:
        veredicto = "SENAL_DEBIL"
    elif delta > 0 and pos >= 4:
        veredicto = "EFECTO_CLARO"
    else:
        veredicto = "SIN_EFECTO_CLARO"
    if semi > 0.015:
        aviso = ("la semianchura del bootstrap supera 0,015, luego la banda de efecto claro "
                 "es inalcanzable por construccion. Se declara para no presentar como hallazgo "
                 "lo que es limite de resolucion")
    else:
        aviso = "la semianchura del bootstrap no supera la banda de 0,015"
    log("")
    log(f"   VEREDICTO: {veredicto} | delta {delta:+.6f} | pliegues {pos}/5")
    log(f"   {aviso}")

    salida = {
        "familia": "P72_TWITTER_WORD2VEC_DIM", "experiment_id": "P72-03",
        "fase": "control de aleatoriedad",
        "fecha": inicio.isoformat(timespec="seconds"),
        "pre_registro": {"documento": "docs/p72_preregistro_control_aleatorio.md",
                         "commit": "7e1aba4", "anterior_a_la_corrida": True},
        "naturaleza": "CONTROL. 11 corridas downstream, 0 entrenamientos de embeddings",
        "las_dos_ramas": {"A": "Word2Vec entrenado", "B": "10 nulas con direccion aleatoria "
                                                          "y norma por token exacta"},
        "declaracion_de_vocabulario": (
            "las DIRECCIONES no provienen de datos; el SOPORTE (vocabulario) y las MAGNITUDES por "
            "token si vienen del corpus externo. El pipeline de vocabulario y normas NO consulta "
            "el dataset de la competencia"),
        "brazo_a": {k: v2 for k, v2 in A.items() if k != "proba"},
        "referencia_historica": REFERENCIA_HISTORICA,
        "desviacion_respecto_de_la_referencia": round(desv, 10),
        "reproduce_dentro_de_tolerancia": bool(abs(desv) <= TOL),
        "nulas": nulas,
        "nula_resumen": {"media": round(float(cvs.mean()), 10),
                         "desviacion": round(float(cvs.std(ddof=1)), 10),
                         "minimo": round(float(cvs.min()), 10),
                         "maximo": round(max_nula, 10)},
        "delta_contra_el_maximo": round(delta, 10),
        "pliegues_contra_el_maximo": pos,
        "anclas": anclas,
        "bootstrap_agrupado": {"media": media_b, "ic95_bajo": lo_b, "ic95_alto": hi_b,
                               "semianchura": round(float(semi), 10),
                               "grupos": len(np.unique(grupos))},
        "aviso_de_potencia": aviso,
        "veredicto": veredicto,
        "parrafo_de_alcance": (
            "El control mide la contribucion marginal de las DIRECCIONES APRENDIDAS bajo una "
            "unica lectura fija: mismo vocabulario, d=100, mismo mean pooling, misma regresion "
            "logistica, mismos umbrales congelados. Las dos ramas pertenecen a la MISMA FAMILIA "
            "FUNCIONAL -un clasificador lineal sobre la bolsa de tokens normalizada, con "
            "coeficientes en un subespacio de dimension menor o igual que 100- y lo unico que "
            "cambia es CUAL subespacio se usa. Un empate acota la conclusion a esta lectura; no "
            "prueba ni descarta senal dependiente de orden, composicion o no linealidad"),
        "que_no_se_puede_afirmar": [
            "Word2Vec no sirve, ni los embeddings no aportan, como afirmaciones generales",
            "no hay senal semantica en los tuits",
            "los vectores aleatorios son tan buenos como los entrenados, como conclusion global",
            "el modelo secuencial funcionara o no funcionara",
            "los vectores aleatorios no usan datos: falso, el vocabulario y las normas si",
            "cualquier afirmacion sobre test o leaderboard derivada del control"],
        "reproducibilidad": {"nivel": "R2", "hash_v03": hash_v03, "corrected_cv_v03": V03_CCV,
                             "sha_matriz_entrenada": sha_W},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    (REPORTS / "twitter_word2vec_control_aleatorio.json").write_text(
        json.dumps(salida, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with (REPORTS / "twitter_word2vec_control_aleatorio.csv").open(
            "w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["brazo", "tirada", "semilla", "cv", "delta_vs_A",
                    "matriz_distinta", "normas_conservadas", "segundos"])
        w.writerow(["A_entrenado", "", "", A["cv"], 0.0, "", "", ""])
        for x in nulas:
            w.writerow(["B_aleatorio", x["tirada"], x["semilla"], x["cv"],
                        round(x["cv"] - A["cv"], 10), x["matriz_distinta"],
                        x["normas_conservadas"], x["segundos"]])
        w.writerow([])
        for k, val in anclas.items():
            w.writerow(["ancla", k, "", val, "", "", "", ""])
        w.writerow(["nula", "media", "", round(float(cvs.mean()), 10), "", "", "", ""])
        w.writerow(["nula", "desviacion", "", round(float(cvs.std(ddof=1)), 10), "", "", "", ""])
        w.writerow(["nula", "maximo", "", round(max_nula, 10), "", "", "", ""])
        w.writerow(["decision", veredicto, "", round(delta, 10), "", "", "", ""])

    docs = [REPORTS / "twitter_word2vec_control_aleatorio.json",
            REPORTS / "twitter_word2vec_control_aleatorio.csv"]
    sucios = [x.name for x in docs if any(b > 127 for b in x.read_bytes())]
    if sucios:
        raise SystemExit(f"PARADA: documentos con bytes no ASCII: {sucios}")

    log("")
    log("=" * 78)
    log(f"P72-03 OK | {veredicto}")
    log(f"   A {A['cv']:.6f} | nula: media {cvs.mean():.6f} maximo {max_nula:.6f}")
    log(f"   delta contra el maximo {delta:+.6f} | pliegues {pos}/5")
    log("=" * 78)


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
