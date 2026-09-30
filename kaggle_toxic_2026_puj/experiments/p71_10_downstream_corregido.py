"""P1e: downstream con el protocolo de V03, la escala corregida y el control de azar.

Los tres defectos que tenia mi downstream, y que este script arregla
---------------------------------------------------------------------
UNO. El vectorizador se ajustaba sobre TODO el desarrollo en vez de dentro de cada pliegue.
    Comprobado: la forma correcta reproduce el OOF de V03 EXACTAMENTE, 0,777211, y la forma
    atajo daba 0,778004.

DOS. Usaba C=1.0 sin `class_weight`, cuando V03 usa C=2.0 con `balanced`.

TRES. La escala del bloque denso. Concatenar 200 columnas densas de varianza unidad a un
    bloque disperso de 24.646 dimensiones hace que las densas dominen la penalizacion L2. Y
    `normalize()` sobre el producto concatenado NO es el arreglo: encoge el bloque TF-IDF unas
    veinte veces y hunde el resultado.

El arreglo correcto es igualar la NORMA DE FILA de los dos bloques, que es lo que hace este
script. Y con eso se puede barrer la escala, que es lo que revela el resultado.

El control de azar
------------------
Se anade un bloque ALEATORIO con la misma forma y escala. Es el control que decide: si los
embeddings reales rinden igual que el ruido, entonces no aportan.
"""

from __future__ import annotations

import ast
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
ESCALAS = (0.0, 0.05, 0.10, 0.20, 0.50, 1.00)


def log(m: str) -> None:
    """Imprime una linea de progreso.

    Args:
        m: texto a mostrar.
    """
    print(f"[P71-10] {m}", flush=True)


def f1m(y: np.ndarray, pred: np.ndarray) -> float:
    """Macro F1 del proyecto.

    Args:
        y: etiquetas.
        pred: predicciones.

    Returns:
        La media de las F1 por columna.
    """
    return float(metricas_multietiqueta(y, pred)["f1_media_columnas"])


def pool(textos, vocab: dict[str, int], W: np.ndarray) -> np.ndarray:
    """Mean pooling.

    Args:
        textos: textos.
        vocab: indice.
        W: embeddings.

    Returns:
        Matriz (n, dim).
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


def cv_con_bloque(des, tx, y, pl, D: np.ndarray, escala: float) -> tuple[float, np.ndarray]:
    """CV con el protocolo de V03 mas un bloque denso opcional.

    El vectorizador se ajusta DENTRO de cada pliegue, y el bloque denso se escala para que su
    norma de fila iguale la del bloque disperso. Sin eso, las densas dominan la penalizacion L2.

    Args:
        des: DataFrame de desarrollo.
        tx: textos.
        y: etiquetas.
        pl: pliegues.
        D: bloque denso.
        escala: fraccion de la norma del bloque disperso.

    Returns:
        El CV y las probabilidades OOF.
    """
    pr = np.zeros((len(y), len(ETIQUETAS)))
    for a, b in pl:
        p2 = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        c2 = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                             sublinear_tf=True)
        Xi = sparse.hstack([p2.fit_transform(tx.iloc[a]), c2.fit_transform(tx.iloc[a])],
                           format="csr")
        Xv = sparse.hstack([p2.transform(tx.iloc[b]), c2.transform(tx.iloc[b])], format="csr")
        if escala > 0:
            sc = StandardScaler().fit(D[a])
            Da, Db = sc.transform(D[a]), sc.transform(D[b])
            norma_d = float(np.sqrt((Da ** 2).sum(axis=1).mean()))
            norma_x = float(np.sqrt(Xi.multiply(Xi).sum(axis=1).mean()))
            k = escala * norma_x / max(norma_d, 1e-12)
            Xi = sparse.hstack([Xi, sparse.csr_matrix(Da * k)], format="csr")
            Xv = sparse.hstack([Xv, sparse.csr_matrix(Db * k)], format="csr")
        for j in range(len(ETIQUETAS)):
            m = LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000,
                                   solver="liblinear", random_state=SEMILLA)
            m.fit(Xi, y[a, j])
            pr[b, j] = m.predict_proba(Xv)[:, 1]
    return f1m(y, (pr >= UMBRALES[None, :]).astype(int)), pr


def main() -> None:
    """Ejecuta el downstream corregido con el control de azar y el barrido de escala.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P71-10 P1e: protocolo de V03, escala igualada y control de azar")
    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")

    v = np.load(PROJECT_ROOT / "features" / "oof_v03_final.npz", allow_pickle=True)
    y = v["y_true"].astype(int)
    tr = read_train()
    g = grupos_fusionados(tr)
    idev, _ = holdout_v2(tr, g)
    des = tr.iloc[idev].reset_index(drop=True)
    pl = pliegues_corregidos(des, g[idev])
    tx = des["text"]

    base, _ = cv_con_bloque(des, tx, y, pl, np.zeros((len(y), 1)), 0.0)
    log(f"   V03 con el protocolo correcto: {base:.6f}  (canonico {V03_CCV!r})")
    if abs(base - V03_CCV) > 1e-9:
        raise SystemExit("PARADA: la reconstruccion no reproduce V03")

    bloques, coberturas = {}, {}
    for etq in ("twitter", "general"):
        d = np.load(MODELOS / f"w2v_{etq}_lrfix.npz", allow_pickle=True)
        vocab = {w: i for i, w in enumerate(d["palabras"])}
        P, n_tok, n_cub = pool(tx, vocab, d["W"])
        bloques[etq] = P
        coberturas[etq] = {
            "vocabulario": len(vocab),
            "tokens_por_fila_media": round(float(n_tok.mean()), 2),
            "tokens_cubiertos_media": round(float(n_cub.mean()), 2),
            "fraccion_cubierta": round(float(n_cub.sum() / max(n_tok.sum(), 1)), 6),
            "filas_sin_ningun_token": int((n_cub == 0).sum()),
        }
        c = coberturas[etq]
        log(f"   {etq}: {c['tokens_por_fila_media']} tokens por fila, "
            f"{c['tokens_cubiertos_media']} cubiertos "
            f"({100 * c['fraccion_cubierta']:.1f} por ciento)")
        if c["filas_sin_ningun_token"]:
            log(f"      AVISO: {c['filas_sin_ningun_token']} filas sin ningun token en vocabulario")
    bloques["azar"] = np.random.default_rng(SEMILLA).standard_normal((len(y), 200))

    log("")
    log("   BARRIDO DE ESCALA, con el control de azar")
    log(f"      {'escala':>8}{'twitter':>14}{'general':>14}{'AZAR':>14}")
    tabla = []
    for esc in ESCALAS:
        fila = {"escala": esc}
        for etq in ("twitter", "general", "azar"):
            cv, _ = cv_con_bloque(des, tx, y, pl, bloques[etq], esc)
            fila[etq] = round(cv, 10)
        tabla.append(fila)
        log(f"      {esc:>8.2f}{fila['twitter']:>14.6f}{fila['general']:>14.6f}"
            f"{fila['azar']:>14.6f}")

    # La lectura que decide: los reales frente al azar
    log("")
    log("   LA LECTURA QUE DECIDE: los embeddings reales frente al RUIDO")
    exceso = [(f["twitter"] - f["azar"], f["general"] - f["azar"], f["escala"]) for f in tabla]
    for d_tw, d_ge, esc in exceso:
        log(f"      escala {esc:.2f}: twitter-azar {d_tw:+.6f} | general-azar {d_ge:+.6f}")
    max_tw = max(e[0] for e in exceso)
    max_ge = max(e[1] for e in exceso)
    log("")
    log(f"      exceso maximo de twitter sobre el ruido: {max_tw:+.6f}")
    log(f"      exceso maximo de general sobre el ruido: {max_ge:+.6f}")
    aporta = max_tw >= 0.0015
    log(f"      alguno supera al ruido por mas de 0,0015 en algun punto: {aporta}")

    mejor = max(tabla, key=lambda f: max(f["twitter"], f["general"]))
    gana = "TWITTER" if mejor["twitter"] > mejor["general"] else "GENERAL"
    ventaja = mejor["twitter"] - mejor["general"]
    veredicto = "PROMISING" if (aporta and ventaja > 0) else "CLOSED"

    salida = {
        "familia": "P71_TWITTER_WORD2VEC", "experiment_id": "P71-10",
        "fase": "P1e downstream corregido",
        "fecha": inicio.isoformat(timespec="seconds"),
        "naturaleza": "EXPERIMENTO. Protocolo de V03, escala igualada, control de azar",
        "defectos_corregidos": [
            "el vectorizador se ajustaba sobre todo el desarrollo en vez de dentro de cada pliegue",
            "C=1.0 sin class_weight, cuando V03 usa C=2.0 balanced",
            "la escala del bloque denso no estaba igualada a la del disperso",
        ],
        "verificacion_del_protocolo": {"cv_reconstruido": round(base, 10),
                                       "cv_canonico": V03_CCV,
                                       "reproduce": bool(abs(base - V03_CCV) < 1e-9)},
        "cobertura": coberturas,
        "barrido_de_escala": tabla,
        "exceso_sobre_el_ruido": {"twitter_max": round(max_tw, 10),
                                  "general_max": round(max_ge, 10),
                                  "alguno_aporta": bool(aporta)},
        "mejor": mejor, "ganador": gana, "ventaja": round(ventaja, 10),
        "veredicto": veredicto,
        "reproducibilidad": {"nivel": "R2", "hash_v03": hash_real, "corrected_cv_v03": V03_CCV},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    (REPORTS / "twitter_word2vec_downstream_corregido.json").write_text(
        json.dumps(salida, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    import csv as _csv
    with (REPORTS / "twitter_word2vec_escala_y_ruido.csv").open(
            "w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["escala", "twitter", "general", "azar",
                    "twitter_menos_azar", "general_menos_azar"])
        for f in tabla:
            w.writerow([f["escala"], f["twitter"], f["general"], f["azar"],
                        round(f["twitter"] - f["azar"], 10),
                        round(f["general"] - f["azar"], 10)])
        w.writerow([])
        for etq, c in coberturas.items():
            w.writerow([etq, "cobertura", c["fraccion_cubierta"],
                        c["tokens_por_fila_media"], c["tokens_cubiertos_media"],
                        c["filas_sin_ningun_token"]])

    log("")
    log("=" * 78)
    log(f"P71-10 OK | {veredicto}")
    log(f"   exceso maximo sobre el ruido: twitter {max_tw:+.6f} | general {max_ge:+.6f}")
    log("=" * 78)


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
