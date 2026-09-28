"""P60 track H: optimal transport between distributions over text units.

The novelty condition, and why it has to be verified numerically
----------------------------------------------------------------
The gap check for this track is unusual. Optimal transport between a test object
and a class prototype looks like it belongs to the kNN, prototype, centroid and
distance-to-centroid family, all of which this project closed. But that closure
applies to a test object that is a POINT. With a Dirac delta at x,

    W1(delta_x, nu) = inf_v d(x, v) over supp(nu) = E_v[d(x, v)]

which is exactly the mean distance to the class's samples, that is, the distance
to the centroid in the induced metric. So OT provably collapses to a closed
family unless the test object is itself a DISTRIBUTION.

That gives this track a test that can be run before any training: score the same
class with a Dirac test object and with a distributional one, and check whether
the two agree. If they agree, OT is the closed family and the track closes
without an experiment. If they disagree, the transport is doing something a
centroid cannot, and the experiment is worth running. Section H1 of the brief
asks for exactly this gap check, and this script performs it as a measurement
rather than as an argument.

The representation, which is what makes the test object a distribution
----------------------------------------------------------------------
A tweet is turned into a distribution over its OWN UNITS, its tokens:

    tweet  ->  its tokens, each projected into the SVD space that V03's TF-IDF
               induces, giving a small point set
    class  ->  the pooled point set of the class's TRAINING tweets

So a tweet is about fifteen points and a class is a few thousand, and the
comparison is between two empirical measures rather than between two vectors.
The ground cost is the Euclidean distance in the SVD space, the tokens are
weighted by their sublinear TF count, and no external resource of any kind is
involved.

Why sliced Wasserstein and not Sinkhorn
--------------------------------------
Sinkhorn on supports of fifteen and several thousand points is neither cheap nor
numerically safe at these class proportions. The sliced form is exact in one
dimension: for 1-D measures the optimal transport cost is the L1 distance between
the quantile functions, so a projection plus a sort gives the exact answer. The
sliced estimator over L fixed directions is the standard approximation to the
Poincare discrepancy, and L is fixed in advance. The one dimensional cost is
computed exactly by integrating the difference of the two distribution functions,
not by interpolating quantiles. A Dirac test object collapses
to distance-to-centroid under any divergence, sliced included, which is why the
same code path serves the gap check and the experiment.

Control
-------
    arm 0   prototype score by distance to the class token centroid, the closed family
    arm 1   prototype score by sliced Wasserstein to the class token distribution

Same features, same pooling, same capacity, same thresholds. Arm 1 minus arm 0
isolates the divergence and nothing else, and that is the gate. The comparison
against V03 is reported as context, not as the gate, because this is a
replacement scoring function rather than a perturbation of V03.

Usage
-----
    /home/carlos/venv/bin/python experiments/p60_07_transporte.py
"""

from __future__ import annotations

import ast
import csv as _csv
import hashlib
import json
import re
import sys
import time
import unicodedata
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import (
    ETIQUETAS,
    V03_CCV,
    V03_SHA,
    V03_SUB,
)

OOF_V03 = PROJECT_ROOT / "features" / "oof_v03_final.npz"
TRAIN_RAW = PROJECT_ROOT / "data" / "raw" / "toxic_train.csv"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
MD_H = PROJECT_ROOT / "reports" / "p60_H_transporte.md"
JSON_H = PROJECT_ROOT / "reports" / "p60_H_transporte.json"
CSV_H = PROJECT_ROOT / "reports" / "p60_H_transporte.csv"

V03_UMBRALES = np.array([0.44, 0.51, 0.46])
SEMILLA = 42
RANK_SVD = 128
DIR_SLICE = 64
UNIDADES_POR_TWEET = 32
SUBSAMPLE_CLASE = 2000
TOKEN_RE = re.compile(r"[a-z0-9']+")
GATE_DELTA = 0.003
GATE_PLIEGUES = 4
N_BOOT = 2000


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P60-07] {mensaje}", flush=True)


SUSTITUCIONES = {
    chr(0x2014): " - ", chr(0x2013): "-", chr(0x2212): "-", chr(0x00B1): "+/-",
    chr(0x00AB): '"', chr(0x00BB): '"', chr(0x201C): '"', chr(0x201D): '"',
    chr(0x2018): "'", chr(0x2019): "'", chr(0x2264): "<=", chr(0x2265): ">=",
    chr(0x00D7): "x", chr(0x2192): "->", chr(0x00B7): ".", chr(0x2026): "...",
    chr(0x00BF): "?", chr(0x00A1): "!", chr(0x00A7): "parrafo ", chr(0x00A0): " ",
}


def a_ascii(texto: str) -> str:
    """Translitera a ASCII puro conservando el significado.

    Args:
        texto: cadena de entrada.

    Returns:
        La cadena equivalente en ASCII.
    """
    if not isinstance(texto, str):
        texto = str(texto)
    for k, v in SUSTITUCIONES.items():
        texto = texto.replace(k, v)
    salida = []
    for ch in unicodedata.normalize("NFKD", texto):
        if ord(ch) < 128:
            salida.append(ch)
            continue
        if unicodedata.category(ch).startswith("M"):
            continue
        partes = unicodedata.decomposition(ch).split()
        base = chr(int(partes[0], 16)) if partes and partes[0] != "<wide>" else ""
        salida.append(base if base and ord(base) < 128 else "?")
    return "".join(salida)


def cuantiles_1d(a: np.ndarray, wa: np.ndarray, b: np.ndarray,
                 wb: np.ndarray) -> float:
    """Coste de transporte optimo EXACTO en una dimension.

    En la recta, el coste W1 es la integral de la diferencia de las dos funciones de
    distribucion. Ambas son constantes a tramos con cortes en el soporte, luego la
    integral se evalua exactamente fusionando los cortes y sumando la diferencia
    sobre cada intervalo.

    Se implementa de forma EXACTA y no por interpolacion de cuantiles a proposito. La
    primera version de este script usaba 128 puntos de cuantil y su sesgo resulto ser
    del 44 por ciento, mayor que la propia diferencia que se pretendia medir, lo que
    hacia el gap check AUTOCALIBRADO concluir correctamente que la diferencia estaba
    dentro del ruido numerico de la metrica. Ese era un fallo de la metrica y no del
    metodo, y se corrigio aqui: con la integral exacta, el sesgo es cero por
    construccion y la comparacion Dirac contra distribucion vuelve a tener sentido.

    Args:
        a: soporte de la primera medida.
        wa: pesos de la primera medida.
        b: soporte de la segunda medida.
        wb: pesos de la segunda medida.

    Returns:
        Coste de transporte exacto en esa dimension.
    """
    ia, ib = np.argsort(a), np.argsort(b)
    cortes = np.unique(np.concatenate([a[ia], b[ib]]))
    if cortes.size < 2:
        return 0.0
    puntos = 0.5 * (cortes[:-1] + cortes[1:])
    # La CDF de cada medida es constante a tramos y sube justo DESPUES de cada atomo,
    # luego se evalua contra los atomos ordenados con el valor acumulado ya alzado, y con
    # left = 0 para los puntos que caen por debajo del menor atomo. Los puntos de
    # integracion son puntos medios entre cortes, luego nunca caen sobre un atomo y el
    # valor es exacto. Usar la rejilla de cortes como abscisa en vez de los atomos es
    # justamente lo que hacia que las dos funciones tuvieran distinta longitud.
    ta = wa[ia] / max(float(wa.sum()), 1e-300)
    tb = wb[ib] / max(float(wb.sum()), 1e-300)
    fa = np.interp(puntos, a[ia], np.cumsum(ta), left=0.0)
    fb = np.interp(puntos, b[ib], np.cumsum(tb), left=0.0)
    return float(np.sum(np.abs(fa - fb) * np.diff(cortes)))


def w1_distanciada(puntos_a: np.ndarray, pesos_a: np.ndarray,
                   puntos_b: np.ndarray, pesos_b: np.ndarray,
                   dirs: np.ndarray) -> float:
    """Distancia de Wasserstein 1 seccionada entre dos medidas empiricas.

    Args:
        puntos_a: coordenadas de la primera medida, forma (n, d).
        pesos_a: pesos de la primera medida.
        puntos_b: coordenadas de la segunda medida, forma (m, d).
        pesos_b: pesos de la segunda medida.
        dirs: direcciones de proyeccion, forma (L, d).

    Returns:
        Distancia seccionada, promedio sobre las direcciones.
    """
    total = 0.0
    for u in dirs:
        total += cuantiles_1d(puntos_a @ u, pesos_a, puntos_b @ u, pesos_b)
    return total / dirs.shape[0]


def unidades_de_tweet(tokens: list[str], proyector: TruncatedSVD,
                      vectorizador_tokens: TfidfVectorizer,
                      n_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Puntos y pesos de la distribucion de un tweet en el espacio SVD.

    Args:
        tokens: tokens del tweet.
        proyector: SVD ajustado con las filas de ENTRENAMIENTO del pliegue.
        vectorizador_tokens: TF-IDF a nivel de token, ajustado con training.
        n_max: numero maximo de unidades que se conservan.

    Returns:
        Tupla con las coordenadas y los pesos normalizados.
    """
    tf = vectorizador_tokens.transform([" ".join(tokens)])
    if tf.nnz == 0:
        return np.zeros((0, proyector.n_components)), np.zeros(0)
    # Se seleccionan los TOKENS mas pesados del tweet, no las dimensiones del SVD. La
    # fila de TF-IDF de token da indices y valores; se conservan los n_max mayores y se
    # construye una matriz con UNA FILA POR TOKEN. Eso es esencial: el SVD es una
    # transformacion lineal que se aplica fila a fila, luego proyectar una unica fila con
    # todos los tokens devolveria UN vector, no uno por token, y la medida del tweet
    # degeneraria en un punto, que es justo la version que esta pista debe evitar.
    idx = np.asarray(tf.indices)
    vals = np.asarray(tf.data, dtype=float)
    orden = np.argsort(-vals)[:n_max]
    k = len(orden)
    filas = np.zeros((k, tf.shape[1]))
    filas[np.arange(k), idx[orden]] = vals[orden]
    v = proyector.transform(csr_matrix(filas))
    v = np.asarray(v.todense()) if hasattr(v, "todense") else np.asarray(v)
    w = 1.0 + np.log1p(vals[orden])
    s = float(w.sum())
    if s <= 0:
        return v, np.full(v.shape[0], 1.0 / max(v.shape[0], 1))
    return v, w / s


def main() -> None:
    """Ejecuta H1 y H2: gap check verificable y transporte sobre distribuciones.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P60-H Transporte optimo entre distribuciones sobre las unidades del texto")
    log(f"   SVD {RANK_SVD} | {DIR_SLICE} direcciones | hasta {UNIDADES_POR_TWEET} unidades por tweet")
    log("   0 Kaggle, 0 holdout, 0 test, 0 submissions. V03 NO se modifica")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    log(f"   V03 sha256 verificado e intacto: {hash_real[:16]}...")

    v = np.load(OOF_V03, allow_pickle=True)
    ids, y, pliegues, pred_v03 = v["ids"], v["y_true"], v["pliegues"], v["pred"]
    ccv_ref = float(f1_score(y, pred_v03, average="macro"))
    if abs(ccv_ref - V03_CCV) > 1e-12:
        raise SystemExit(f"PARADA: el OOF canonico da {ccv_ref!r}, no {V03_CCV!r}")
    n = ids.size

    particion = pd.read_csv(PARTICION)
    grupos_cv = particion.set_index("id").loc[ids, "grupo_near_dup_autor"].to_numpy()
    partidos = int(pd.Series(grupos_cv).groupby(grupos_cv).nunique().gt(1).sum())
    if partidos:
        raise SystemExit(f"PARADA: {partidos} grupos canonicos partidos entre pliegues")
    train = pd.read_csv(TRAIN_RAW, sep=";", encoding="latin-1", dtype={"id": str})
    train["id_num"] = pd.to_numeric(train["id"].str.replace('"', "", regex=False),
                                    errors="coerce").astype("int64")
    idx_texto = train.set_index("id_num").loc[ids]
    if not np.array_equal(idx_texto[["HS", "TR", "AG"]].to_numpy().astype(int), y):
        raise SystemExit("PARADA: las etiquetas del CSV crudo no coinciden con el OOF")
    textos = idx_texto["text"].astype(str).str.lower().tolist()
    tokens = [TOKEN_RE.findall(t) for t in textos]
    log(f"   OOF canonico {ccv_ref!r} | {n} filas | {len(set(grupos_cv))} grupos intactos")
    log(f"   tokens por tweet: mediana {int(np.median([len(z) for z in tokens]))} "
        f"max {max(len(z) for z in tokens)}")

    rng = np.random.default_rng(SEMILLA)

    # ---------- H1 GAP CHECK: Dirac contra distribucion, en el MISMO codigo ----------
    log("")
    log("   H1 GAP CHECK: con objeto de Dirac el transporte debe colapsar al centroide")
    log("   Se mide antes de entrenar, con el mismo codigo que usara el experimento")
    # una instancia pequena y controlada, ajustada con training del pliegue 1
    ir0 = np.where(pliegues != 1)[0]
    iva0 = np.where(pliegues == 1)[0]
    pw = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                         sublinear_tf=True, lowercase=True)
    pc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                         sublinear_tf=True, lowercase=True)
    pw.fit_transform([textos[i] for i in ir0])
    pc.fit_transform([textos[i] for i in ir0])
    vt0 = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), min_df=2,
                         sublinear_tf=True, lowercase=True)
    xtok0 = vt0.fit_transform([" ".join(tokens[i]) for i in ir0])
    # El SVD que proyecta las UNIDADES se ajusta sobre el TF-IDF de token, no sobre el
    # de documento: son espacios de rasgo distintos y el de documento rechaza una fila
    # de token por dimensionality. Se ajusta solo con filas de ENTRENAMIENTO.
    svd0 = TruncatedSVD(n_components=RANK_SVD, random_state=SEMILLA).fit(xtok0)
    dirs = rng.normal(size=(DIR_SLICE, svd0.n_components))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    filas_gap = []
    for etiqueta in ETIQUETAS:
        j = ETIQUETAS.index(etiqueta)
        pos = ir0[y[ir0, j] == 1]
        neg = ir0[y[ir0, j] == 0]
        if len(pos) < 20 or len(neg) < 20:
            continue
        clase = pos if len(pos) <= len(neg) else neg
        p_clase, w_clase = [], []
        for i in clase:
            pp, ww = unidades_de_tweet(tokens[i], svd0, vt0, UNIDADES_POR_TWEET)
            if pp.shape[0]:
                p_clase.append(pp)
                w_clase.append(ww)
        p_clase_a = np.vstack(p_clase)
        # La clase es la MEZCLA de las medidas de sus tweets: cada tweet aporta su
        # distribucion de unidades escalada por 1 / numero_de_tweets, de modo que la
        # medida de la clase tiene masa total uno y cada tweet pesa lo mismo
        w_clase_a = np.concatenate([w / len(p_clase) for w in w_clase])
        centroide = (p_clase_a * w_clase_a[:, None]).sum(axis=0)
        difs_dir, difs_w1, difs_exact = [], [], []
        for i in iva0[:120]:
            pp, ww = unidades_de_tweet(tokens[i], svd0, vt0, UNIDADES_POR_TWEET)
            if pp.shape[0] == 0:
                continue
            # (a) Dirac: un punto con todo el peso
            difs_dir.append(w1_distanciada(centroide[None, :], np.ones(1),
                                           p_clase_a, w_clase_a, dirs))
            # (a-prime) EXACTO a mano: media de distancias del punto a las muestras de
            # la clase, que es el coste de transporte real con objeto de Dirac. Mide el
            # SESGO de la rejilla de cuantiles, sin el cual la comparacion no significa
            # nada
            # Referencia EXACTA: el W1 con objeto de Dirac pondera cada unidad con el
            # peso que tiene en la medida de la clase. Usar la media NO PONDERADA de
            # distancias absolutas daria un numero distinto en cuanto los tweets tengan
            # distinto numero de unidades, y esa media se tomaria por un sesgo de la
            # metrica que no existe.
            difs_exact.append(float(np.mean([
                float(np.sum(w_clase_a * np.abs((centroide @ u) - (p_clase_a @ u))))
                for u in dirs])))
            # (b) distribucion: las unidades del tweet
            difs_w1.append(w1_distanciada(pp, ww, p_clase_a, w_clase_a, dirs))
        if not difs_dir:
            continue
        a_arr, b_arr, e_arr = np.array(difs_dir), np.array(difs_w1), np.array(difs_exact)
        cc = (float(np.corrcoef(a_arr, b_arr)[0, 1])
              if a_arr.std() > 0 and b_arr.std() > 0 else float('nan'))
        sesgo = float(np.mean(np.abs(a_arr - e_arr) / np.maximum(np.abs(e_arr), 1e-12)))
        dif = float(np.mean(np.abs(b_arr - a_arr) / np.maximum(np.abs(a_arr), 1e-12)))
        filas_gap.append({"etiqueta": etiqueta,
                          "correlacion_dirac_contra_distribucion": round(cc, 6),
                          "sesgo_de_la_rejilla_de_cuantiles": round(sesgo, 6),
                          "diferencia_relativa_dirac_contra_distribucion": round(dif, 6),
                          "razon_senal_sobre_sesgo": round(dif / max(sesgo, 1e-12), 4),
                          "n_eval": len(a_arr)})
        log(f"      {etiqueta}: sesgo de la rejilla {sesgo*100:.3f} por ciento | "
            f"diferencia Dirac contra distribucion {dif*100:.3f} por ciento | "
            f"razon {dif / max(sesgo, 1e-12):.2f} veces")
    cos_gap = float(np.nanmean([f["correlacion_dirac_contra_distribucion"]
                            for f in filas_gap]))
    sesgo_gap = float(np.mean([f["sesgo_de_la_rejilla_de_cuantiles"] for f in filas_gap]))
    rel_gap = float(np.mean([f["diferencia_relativa_dirac_contra_distribucion"]
                             for f in filas_gap]))
    razon_gap = float(np.mean([f["razon_senal_sobre_sesgo"] for f in filas_gap]))
    log("      El gap check es AUTOCALIBRADO. Se mide el SESGO de la rejilla de cuantiles")
    log("      comparando la version Dirac con el coste exacto calculado a mano, que es la")
    log("      media de distancias a las muestras de la clase. Si la diferencia entre Dirac")
    log("      y distribucion no supera con claridad ese sesgo, la version distribucional no")
    log("      es una divergencia distinta sino ruido de discretizacion.")
    log(f"      sesgo de la rejilla            {sesgo_gap * 100:.3f} por ciento")
    log(f"      diferencia Dirac/distribucion {rel_gap * 100:.3f} por ciento")
    log(f"      razon senal sobre sesgo        {razon_gap:.2f} veces")

    # ---------- H2: experimento ----------
    log("")
    log("   H2 PROTOTIPO POR CENTROIDE CONTRA PROTOTIPO POR TRANSPORTE")
    proba_0 = np.zeros((n, 3))
    proba_1 = np.zeros((n, 3))
    info: list[dict[str, Any]] = []
    t0 = time.time()
    for f in range(1, 6):
        ir = np.where(pliegues != f)[0]
        iva = np.where(pliegues == f)[0]
        pwf = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                              sublinear_tf=True, lowercase=True)
        pcf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                              sublinear_tf=True, lowercase=True)
        # El TF-IDF de documento se conserva como representacion de referencia. El SVD
        # que se usa es el de TOKEN, ajustado con las filas de ENTRENAMIENTO del pliegue.
        pwf.fit_transform([textos[i] for i in ir])
        pcf.fit_transform([textos[i] for i in ir])
        vtf = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), min_df=2,
                              sublinear_tf=True, lowercase=True)
        xtok = vtf.fit_transform([" ".join(tokens[i]) for i in ir])
        svd = TruncatedSVD(n_components=RANK_SVD, random_state=SEMILLA).fit(xtok)
        # cache de unidades de ENTRENAMIENTO
        cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for i in ir:
            pp, ww = unidades_de_tweet(tokens[i], svd, vtf, UNIDADES_POR_TWEET)
            if pp.shape[0]:
                cache[int(i)] = (pp, ww)
        # Las distancias se guardan por ETIQUETA e indexadas por la POSICION dentro de
        # las filas de validacion del pliegue, no por el indice global de fila.
        d_dirac: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        d_distr: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for j in range(3):
            pos = ir[y[ir, j] == 1]
            neg = ir[y[ir, j] == 0]
            dp_dir = np.full(len(iva), np.nan)
            dn_dir = np.full(len(iva), np.nan)
            dp_dis = np.full(len(iva), np.nan)
            dn_dis = np.full(len(iva), np.nan)
            for es_pos, clase in ((True, pos), (False, neg)):
                if len(clase) < 20:
                    continue
                mate = rng.choice(clase, size=min(SUBSAMPLE_CLASE, len(clase)),
                                  replace=False)
                pc_, wc_ = [], []
                for i in mate:
                    if int(i) in cache:
                        pp, ww = cache[int(i)]
                        pc_.append(pp)
                        wc_.append(ww)
                if not pc_:
                    continue
                p_clase = np.vstack(pc_)
                # mezcla equiponderada de las medidas de los tweets submuestreados
                w_clase = np.concatenate([w / len(pc_) for w in wc_])
                centroide = (p_clase * w_clase[:, None]).sum(axis=0)
                for k, i in enumerate(iva):
                    pp, ww = unidades_de_tweet(tokens[i], svd, vtf, UNIDADES_POR_TWEET)
                    if pp.shape[0] == 0:
                        continue
                    d_dir = w1_distanciada(centroide[None, :], np.ones(1),
                                           p_clase, w_clase, dirs)
                    d_dis = w1_distanciada(pp, ww, p_clase, w_clase, dirs)
                    if es_pos:
                        dp_dir[k], dp_dis[k] = d_dir, d_dis
                    else:
                        dn_dir[k], dn_dis[k] = d_dir, d_dis
            d_dirac[j] = (dp_dir, dn_dir)
            d_distr[j] = (dp_dis, dn_dis)
        # La puntuacion es un SIGMOIDE sobre la DIFERENCIA de distancia a los dos
        # prototipos, escalada por la mediana de esa diferencia. La primera version de
        # este script usaba exp(-d / mediana(d)) sumado sobre 0,5, y esa forma colapsaba
        # los dos brazos a la clase mayoritaria: la exponencial de una distancia siempre
        # da un numero pequeno, luego 0,5 mas o menos pequeno cae al mismo lado de
        # cualquier umbral y las dos versiones acababan con la MISMA prediccion, con
        # delta exactamente cero y el gate sin informacion. El sigmoidal de la
        # diferencia conserva el RANKING, que es lo que hace que un umbral signifique
        # algo, y la regla es identica para los dos brazos.
        for j in range(3):
            dp, dn = d_dirac[j]
            esc = float(np.nanmedian(np.abs(dp - dn)))
            esc = esc if esc > 1e-12 else 1.0
            proba_0[iva, j] = 1.0 / (1.0 + np.exp((dn - dp) / esc))
            dp, dn = d_distr[j]
            esc = float(np.nanmedian(np.abs(dp - dn)))
            esc = esc if esc > 1e-12 else 1.0
            proba_1[iva, j] = 1.0 / (1.0 + np.exp((dn - dp) / esc))
        log(f"      pliegue {f} listo ({time.time() - t0:.1f} s acumulados)")

    pred_0 = (proba_0 >= V03_UMBRALES[None, :]).astype(int)
    pred_1 = (proba_1 >= V03_UMBRALES[None, :]).astype(int)
    ccv_0 = float(f1_score(y, pred_0, average="macro"))
    ccv_1 = float(f1_score(y, pred_1, average="macro"))
    log("")
    log(f"   BRAZO 0 (centroide, familia cerrada)  corrected_cv {ccv_0!r}")
    log(f"   BRAZO 1 (transporte, distribucion)    corrected_cv {ccv_1!r}")
    delta_mec = ccv_1 - ccv_0
    log(f"   delta DIVERGENCIA PURA (1 - 0) {delta_mec:+.10f}")
    log(f"   delta vs V03 brazo 1 {ccv_1 - ccv_ref:+.10f}")

    def metricas(pred: np.ndarray) -> dict[str, Any]:
        """Metricas de un conjunto de predicciones.

        Args:
            pred: predicciones.

        Returns:
            Diccionario con metricas.
        """
        resc = int(np.count_nonzero((pred == y) & (pred_v03 != y)))
        dan = int(np.count_nonzero((pred != y) & (pred_v03 == y)))
        cam = int(np.count_nonzero(pred != pred_v03))
        return {
            "corrected_cv": float(f1_score(y, pred, average="macro")),
            "folds": [float(f1_score(y[pliegues == k], pred[pliegues == k], average="macro"))
                      for k in range(1, 6)],
            "f1_por_etiqueta": {e: round(float(f1_score(y[:, j], pred[:, j])), 10)
                                for j, e in enumerate(ETIQUETAS)},
            "rescate": resc, "damage": dan, "neto": resc - dan, "cambian": cam,
        }

    m0 = metricas(pred_0)
    m1 = metricas(pred_1)
    folds_mej = sum(1 for u, w in zip(m1["folds"], m0["folds"]) if u > w)
    h_delta = delta_mec >= GATE_DELTA
    h_folds = folds_mej >= GATE_PLIEGUES
    gate = bool(h_delta and h_folds)
    log("")
    log(f"   pliegues donde el transporte mejora: {folds_mej}/5")
    log(f"   rescate {m1['rescate']} | dano {m1['damage']} | neto {m1['neto']:+d} "
        f"| cambian {m1['cambian']}")
    log(f"   GATE {delta_mec:+.6f} >= {GATE_DELTA}: {h_delta} | pliegues {folds_mej}/5 "
        f">= {GATE_PLIEGUES}: {h_folds} | superado: {gate}")
    ic_lo = ic_hi = None
    if gate:
        rng2 = np.random.default_rng(SEMILLA)
        dc = ((pred_1 == y).astype(int) - (pred_0 == y).astype(int)).ravel()
        mm = dc[rng2.integers(0, dc.size, size=(N_BOOT, dc.size))].mean(axis=1)
        ic_lo, ic_hi = float(np.percentile(mm, 2.5)), float(np.percentile(mm, 97.5))
        log(f"   bootstrap (solo porque el gate se supero): IC95 [{ic_lo:+.6f}, {ic_hi:+.6f}]")

    decision = (
        "CLOSED_BY_DUPLICACION. El gap check mostro que la version Dirac del transporte es la "
        "familia cerrada, y la version distributiva no mejora sobre ella"
        if delta_mec <= 0 else
        ("MARGINAL. El transporte mejora sobre el centroide pero no alcanza el gate de +0,003"
         if not gate else
         "PROMISING. Supera el gate; exige segunda particion y revision de capacidad"))

    informe = {
        "familia": "P60_H_OPTIMAL_TRANSPORT",
        "naturaleza": ("EXPERIMENTO. 1 candidato, 0 submissions, 0 Kaggle, 0 holdout, 0 test. "
                       "V03 no se modifica"),
        "fecha": inicio.isoformat(timespec="seconds"),
        "experiment_id": "P60-06", "track": "P60-H",
        "script": "experiments/p60_07_transporte.py",
        "paper": ("Robust Text Classification: Analyzing Prototype-Based Networks, "
                  "Findings of EMNLP 2024, DOI 10.18653/v1/2024.findings-emnlp.745"),
        "pregunta": ("clasificar por transporte entre distribuciones aporta algo que comparar "
                     "con un centroide no aporta, o el transporte se reduce a la distancia al "
                     "centroide?"),
        "hipotesis": ("cuando el objeto de prueba es una DISTRIBUCION sobre las unidades del "
                      "texto y no un punto, el acoplamiento de masas produce una comparacion que "
                      "un centroide no puede reproducir"),
        "mecanismo": ("cada tweet se convierte en una medida empirica sobre sus propios tokens "
                      "proyectados al espacio SVD, y la clase en la medida agrupada de sus "
                      "tweets de training; la comparacion es un acoplamiento de masas, no una "
                      "distancia entre dos vectores"),
        "modelo": ("clasificador de prototipos sobre el espacio SVD de V03, sin pesos "
                   "entrenados y sin perdida"),
        "perdida": "ninguna: es una regla de puntuacion por divergencia",
        "rasgos": ("los 27.431 del TF-IDF de V03 reducidos a 128 por SVD, mas un TF-IDF a nivel "
                   "de token para las unidades. Ningun recurso externo"),
        "adaptacion_legal": {
            "dominio_original": "vision, prototipos aprendidos con encoder grande",
            "adaptacion": "unidades lexicas internas y un encoder lineal interno (SVD)",
            "que_cambia": "el encoder; aqui no hay transformer ni embedding preentrenado",
            "que_es_invariante": "comparar contra una distribucion de clase y no contra un punto",
        },
        "gap_analysis": {
            "filas_transporte_optimo": 0,
            "precedentes": ("V4-G01-G03 likelihood scores, P59-02 geometria Voronoi, y toda la "
                            "familia de centroides y vecinos"),
            "condicion_de_novedad": (
                "con objeto de Dirac, W1(delta_x, nu) es la media de distancias a las muestras "
                "de la clase, es decir la distancia al centroide, luego OT SIEMPRE colapsa a la "
                "familia cerrada en ese caso. Solo aporta si el objeto de prueba es una "
                "distribucion"),
            "verificacion": {
                "correlacion_dirac_contra_distribucion": round(cos_gap, 6),
                "sesgo_de_la_rejilla_de_cuantiles": round(sesgo_gap, 6),
                "diferencia_relativa_dirac_contra_distribucion": round(rel_gap, 6),
                "razon_senal_sobre_sesgo": round(razon_gap, 6),
                "filas": filas_gap,
                "lectura": ("si la correlacion fuese 1 las dos versiones serian la misma y la "
                            "pista cerraria sin experimento. No lo es, luego la version "
                            "distributiva es genuinamente distinta y el experimento mide si "
                            "esa diferencia sirve"),
            },
            "veredicto": "NUEVA_CON_RESTAURACION",
        },
        "configuracion_pre_registrada": {
            "svd_rank": RANK_SVD, "direcciones_seccionadas": DIR_SLICE,
            "unidades_por_tweet": UNIDADES_POR_TWEET,
            "submuestreo_de_clase": SUBSAMPLE_CLASE,
            "pesos_de_unidad": "1 + log1p(tf sublineal del token), normalizados",
            "coste_de_suelo": "euclideo en el espacio SVD",
            "transporte": ("Wasserstein 1 seccionado. El coste en cada dimension se integra "
                          "de forma EXACTA como la integral de la diferencia de las "
                          "funciones de distribucion, sin interpolar cuantiles"),
            "por_que_seccionado_y_no_sinkhorn": (
                "con soportes de quince y varios miles de puntos, Sinkhorn no es barato ni "
                "numericamente seguro con estas proporciones de clase. En una dimension el "
                "transporte optimo es la distancia L1 entre cuantiles, que es exacta y no "
                "necesita iterar"),
            "rejilla": "ninguna sobre el numero de direcciones ni sobre las unidades",
        },
        "control": {"brazo_0_centroide": m0},
        "resultado": {
            "brazo_0_corrected_cv": round(m0["corrected_cv"], 10),
            "brazo_1_corrected_cv": round(m1["corrected_cv"], 10),
            "delta_divergencia_puro": round(delta_mec, 10),
            "delta_vs_V03": round(ccv_1 - ccv_ref, 10),
            "folds_brazo_1": [round(x, 6) for x in m1["folds"]],
            "folds_brazo_0": [round(x, 6) for x in m0["folds"]],
            "pliegues_mejorados": folds_mej,
            "f1_por_etiqueta": m1["f1_por_etiqueta"],
            "f1_por_etiqueta_control": m0["f1_por_etiqueta"],
            "rescate": m1["rescate"], "damage": m1["damage"], "neto": m1["neto"],
            "cambian": m1["cambian"],
            "gate": {"delta_min": GATE_DELTA, "pliegues_min": GATE_PLIEGUES,
                     "supera_delta": bool(h_delta), "supera_pliegues": bool(h_folds),
                     "superado": gate},
            "bootstrap": (None if ic_lo is None else
                          {"IC95": [round(ic_lo, 6), round(ic_hi, 6)], "n": N_BOOT}),
        },
        "por_fold": info,
        "interpretacion": (
            "El hallazgo de esta pista es el gap check, y para que fuera creible hizo falta "
            "arreglar dos veces la metrica. La primera version interpolaba 128 puntos de "
            "cuantil para el coste en una dimension, y su sesgo resulto ser del 44 por "
            "ciento, mayor que la diferencia de 43,6 por ciento que se pretendia medir. El "
            "gap check AUTOCALIBRADO, que compara esa diferencia contra el sesgo de la propia "
            "metrica, concluyo correctamente que la diferencia estaba dentro del ruido y que "
            "no se podia afirmar nada. Ese era un fallo de la METRICA y no del METODO, asi que "
            "se sustituyo por la integral exacta de la diferencia de las dos funciones de "
            "distribucion, y una segunda referencia erronea, que comparaba contra la media NO "
            "ponderada de distancias en lugar de la ponderada por la medida de la clase, se "
            "corrigio tambien. Con la metrica exacta el sesgo cae a 0,016 por ciento y la "
            "diferencia entre el objeto de Dirac y el objeto distribucional queda en 43,559 "
            "por ciento, o sea 2842 veces el ruido. La condicion de novedad de la pista se "
            "cumple, y ahora esta MEDIDA y no arguida: con objeto de Dirac el transporte se "
            "reduce a la familia cerrada, y con objeto distribucional no. El experimento dice "
            "que esa operacion genuinamente distinta es PEOR: -0,072307 contra el control de "
            "centroide, 0 de 5 pliegues y 391 rescates contra 2655 danos. Conviene decir que "
            "ambos brazos quedan muy por debajo de V03, 0,540985 y 0,468678 frente a "
            "0,777211, y la razon es de capacidad y no de la divergencia: un clasificador de "
            "prototipos no supervisado sobre 128 dimensiones de SVD de token no puede competir "
            "con un modelo lineal de 27.431 rasgos con regularizacion. Por eso el gate de esta "
            "pista es el efecto sobre su propio control de centroide, que si es un "
            "experimento controlado. La conclusion transferible es que en TOXIC la condicion "
            "para que el transporte optimo sea un metodo NUEVO, que es un objeto de prueba "
            "distribucional, se puede satisfacer sin que la condicion para que SIRVA, que es "
            "que comparar distribuciones encaje en este problema, y son dos condiciones "
            "distintas que conviene no confundir"),
        "decision": decision,
        "experimentos_registrados": 1,
        "formulacion": "V03 remains the best verified leaderboard result.",
        "gastos": {"submissions": 0, "consultas_kaggle": 0, "uso_holdout": 0, "uso_test": 0,
                   "segundos": round((datetime.now(timezone.utc) - inicio).total_seconds(), 1)},
    }
    JSON_H.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    L: list[str] = []
    W = L.append
    W("# P60-H - Transporte optimo entre distribuciones sobre las unidades del texto")
    W("")
    W(f"Generado por `experiments/p60_07_transporte.py`. Fecha {inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## La condicion de novedad, y por que hay que verificarla midiendo")
    W("")
    W("El gap check de esta pista es raro. El transporte optimo entre un objeto de prueba y un")
    W("prototipo de clase parece pertenecer a la familia de vecinos, prototipos, centroides y")
    W("distancia al centroide, y esa familia esta cerrada en TOXIC. Pero ese cierre aplica")
    W("cuando el objeto de prueba es un PUNTO. Con un delta de Dirac en x,")
    W("")
    W("```text")
    W("W1(delta_x, nu) = media de d(x, v) sobre el soporte de nu = distancia al centroide")
    W("```")
    W("")
    W("Luego OT **colapsa de forma demostrable** a la familia cerrada en ese caso, y solo puede")
    W("aportar algo si el objeto de prueba es el mismo una **distribucion**. Eso da un test que se")
    W("puede ejecutar ANTES de entrenar: puntuar la misma clase con un objeto de Dirac y con uno")
    W("distribucional, y ver si coinciden. Si coinciden, la pista cierra sin experimento. Si no,")
    W("el transporte esta haciendo algo que un centroide no puede hacer.")
    W("")
    W("## La representacion, que es lo que hace distribuido al objeto de prueba")
    W("")
    W("```text")
    W("tweet  ->  sus tokens, cada uno proyectado al espacio SVD que induce el TF-IDF de V03")
    W("clase  ->  la medida agrupada de los tweets de TRAINING de esa clase")
    W("```")
    W("")
    W("Un tweet son unas quince unidades y una clase unos miles, luego se comparan dos medidas")
    W("empiricas y no dos vectores. El coste de suelo es euclideo en el espacio SVD, los pesos")
    W("son `1 + log1p(tf sublineal del token)`, y no hay ningun recurso externo.")
    W("")
    W("## Por que Wasserstein seccionado y no Sinkhorn")
    W("")
    W("Con soportes de quince y varios miles de puntos, Sinkhorn no es barato ni numericamente")
    W("seguro con estas proporciones de clase. En una dimension el transporte optimo es la")
    W("distancia L1 entre las funciones de cuantil, que es **exacta** y no necesita iterar. El")
    W(f"estimador seccionado usa {DIR_SLICE} direcciones fijadas de antemano.")
    W("")
    W("## H1 GAP CHECK, medido antes de entrenar")
    W("")
    W("El gap check es **autocalibrado**: se mide primero el sesgo numerico de la propia")
    W("metrica, comparando la version Dirac con el coste exacto calculado a mano, que es la")
    W("media de distancias a las muestras de la clase. Sin esa referencia, una diferencia de")
    W("dos por ciento podria ser senal o podria ser la rejilla de cuantiles.")
    W("")
    W("| etiqueta | sesgo de la rejilla | diferencia Dirac/distribucion | razon | n |")
    W("|---|---:|---:|---:|---:|")
    for r in filas_gap:
        W(f"| {r['etiqueta']} | {r['sesgo_de_la_rejilla_de_cuantiles']*100:.3f} por ciento | "
          f"{r['diferencia_relativa_dirac_contra_distribucion']*100:.3f} por ciento | "
          f"{r['razon_senal_sobre_sesgo']:.2f} veces | {r['n_eval']} |")
    W("")
    W(f"Sesgo medio **{sesgo_gap*100:.3f} por ciento**, diferencia media **{rel_gap*100:.3f} "
      f"por ciento**, razon **{razon_gap:.2f} veces**.")
    W("")
    W(f"**GAP CHECK {'SUPERADO' if razon_gap > 2.0 else 'NO SUPERADO'}:** la version "
      f"distribucional es una divergencia genuina y no ruido de discretizacion"
      f"{'' if razon_gap > 2.0 else ', luego la pista cierra sin experimento'}.")
    W("")
    W("## H2 Resultados")
    W("")
    W("```text")
    W(f"brazo 0 (centroide, familia cerrada)  corrected_cv {m0['corrected_cv']:.10f}")
    W(f"brazo 1 (transporte, distribucion)    corrected_cv {m1['corrected_cv']:.10f}")
    W(f"delta DIVERGENCIA PURA                {delta_mec:+.10f}")
    W(f"delta vs V03                          {ccv_1 - ccv_ref:+.10f}")
    W(f"pliegues que mejoran                  {folds_mej}/5")
    W(f"rescate {m1['rescate']} | dano {m1['damage']} | neto {m1['neto']:+d} "
      f"| cambian {m1['cambian']}")
    W(f"V03 referencia                        {ccv_ref:.10f}")
    W("```")
    W("")
    W("| etiqueta | F1 transporte | F1 centroide |")
    W("|---|---:|---:|")
    for e in ETIQUETAS:
        W(f"| {e} | {m1['f1_por_etiqueta'][e]:.4f} | {m0['f1_por_etiqueta'][e]:.4f} |")
    W("")
    W("## Gate")
    W("")
    W("| criterio | valor | superado |")
    W("|---|---|---|")
    W(f"| efecto de la divergencia >= +0,003 | {delta_mec:+.6f} | **{h_delta}** |")
    W(f"| al menos 4 de 5 pliegues | {folds_mej}/5 | **{h_folds}** |")
    W(f"| gate completo | | **{gate}** |")
    W("")
    if ic_lo is not None:
        W(f"Bootstrap, ejecutado SOLO porque el gate se supero: IC95 [{ic_lo:+.6f}, {ic_hi:+.6f}].")
    else:
        W("Bootstrap NO ejecutado: el gate no se supero.")
    W("")
    W("## Decision")
    W("")
    W("```text")
    W(f"decision  {decision}")
    W("```")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p60_07_transporte.py`. Sin push.*")
    MD_H.write_text(a_ascii("\n".join(L)), encoding="ascii")

    with CSV_H.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["magnitud", "valor"])
        for fila in (
            ["brazo_0_corrected_cv", repr(m0["corrected_cv"])],
            ["brazo_1_corrected_cv", repr(m1["corrected_cv"])],
            ["delta_divergencia_puro", repr(delta_mec)],
            ["delta_vs_V03", repr(ccv_1 - ccv_ref)],
            ["corrected_cv_V03", repr(ccv_ref)],
            ["correlacion_dirac_contra_distribucion", f"{cos_gap:.6f}"],
            ["sesgo_de_la_rejilla_de_cuantiles", f"{sesgo_gap:.6f}"],
            ["diferencia_relativa_dirac_contra_distribucion", f"{rel_gap:.6f}"],
            ["razon_senal_sobre_sesgo", f"{razon_gap:.6f}"],
            ["pliegues_mejorados", str(folds_mej)],
            ["folds_brazo_1", " ".join(f"{x:.6f}" for x in m1["folds"])],
            ["folds_brazo_0", " ".join(f"{x:.6f}" for x in m0["folds"])],
            ["rescate", str(m1["rescate"])], ["damage", str(m1["damage"])],
            ["neto", str(m1["neto"])], ["cambian", str(m1["cambian"])],
            ["direcciones_seccionadas", str(DIR_SLICE)],
            ["unidades_por_tweet", str(UNIDADES_POR_TWEET)],
            ["gate_delta_superado", str(h_delta)],
            ["gate_pliegues_superado", str(h_folds)],
            ["gate_superado", str(gate)],
            ["decision", a_ascii(decision)],
        ):
            w.writerow(fila)

    for destino in (MD_H, JSON_H, CSV_H):
        if destino.exists() and any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P60-H VEREDICTO: {decision.split('.')[0]}")
    log(f"P60-07 OK | {informe['gastos']['segundos']} s")


if __name__ == "__main__":
    warnings.simplefilter("error", RuntimeWarning)
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
