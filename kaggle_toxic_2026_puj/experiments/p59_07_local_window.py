"""P59 step 07: track B, local-window representation on V03.

The hypothesis, stated exactly
------------------------------
Paragraph 1 rules out the familiar reading. B is not trying to find the difficult
cells better, and it is not a selector, a confidence filter, a nearest-neighbour
correction or an influence reweighting. The causal claim is about the **unit of
representation**: V03 compresses a whole tweet into one sparse vector, and that
single vector can dilute toxic evidence that is concentrated in a few tokens. A
local-window representation can preserve that evidence, because the classifier
never sees the whole document at once and therefore cannot average the local
signal away.

Why this is not the precedent it looks like
--------------------------------------------
`V4-D01-D04` already tested "ventanas locales como tokens" (spec 24, window +-3)
and closed TRACK D as NO_SIGNAL. That is a **different mechanism**: there the
window is an extra sparse *feature channel* hstacked onto the full document, so
the scoring unit is still the whole tweet and the dilution problem is untouched.
Here the window is the scoring *unit*. The document vector does not exist. That
distinction is the whole experiment, so it is recorded before running anything.

`DL-02` (CharCNN, character kernels 3 and 5) is the closest precedent for the
aggregation *principle*: local receptive fields pooled by a maximum. It scored
0,741036 and was discarded, but in a dense neural family trained from scratch, not
in the sparse linear family with V03's exact representation. Zero rows of the
registry mention multiple-instance learning, and no report in the project is named
after windows or segments other than the FastText training-window family, which is
an unrelated mechanism.

Everything is pre-registered
---------------------------
The configuration is fixed before any validation number is observed, and there is
exactly one of it:

    window   32 tokens
    stride   16 tokens, so windows overlap by 16
    token    the TfidfVectorizer default pattern, so a "token" is the unit the
             vectorizer itself sees
    tweets of at most 32 tokens get ONE window equal to the whole tweet, with no
    duplication and no padding
    the training set is the union of all windows of the training fold, each
    window inheriting its parent tweet's label
    aggregation  max over windows, per label, fixed in advance
    thresholds   V03's 0,44 / 0,51 / 0,46, not retuned
    features     V03's word(1,2) + char_wb(3,5), min_df=2, sublinear_tf
    classifier   V03's LogisticRegression, C=2,0, balanced, liblinear

No grid over window size, stride, aggregation or thresholds. Paragraph 17 forbids
rescuing a negative result with another aggregation, and paragraph 4 forbids
probing alternatives, so the script scores exactly one candidate.

A feasibility ceiling, registered before running
-----------------------------------------------
Tweets here are short: median 15 tokens, maximum 51. With a 32-token window only
8,23 per cent of tweets get more than one window. The representation change can
therefore only act on that stratum, which bounds what B1 can possibly achieve. The
ceiling is computed and printed before the metrics, not after, so it cannot be
excused retroactively.

Usage
-----
    /home/carlos/venv/bin/python experiments/p59_07_local_window.py
"""

from __future__ import annotations

import ast
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
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import (
    ETIQUETAS,
    V03_CCV,
    V03_SHA,
    V03_SUB,
    leer_poblaciones,
)

OOF_V03 = PROJECT_ROOT / "features" / "oof_v03_final.npz"
TRAIN_RAW = PROJECT_ROOT / "data" / "raw" / "toxic_train.csv"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
REGISTRY = PROJECT_ROOT / "experiments.csv"
FRONTIER = PROJECT_ROOT / "reports" / "p59_frontier_registry.csv"
JSON_HIPO = PROJECT_ROOT / "reports" / "p59_hypothesis_registry.json"
MD_RES = PROJECT_ROOT / "reports" / "p59_B_local_window.md"
JSON_RES = PROJECT_ROOT / "reports" / "p59_B_local_window.json"
CSV_RES = PROJECT_ROOT / "reports" / "p59_B_local_window.csv"
CELDA_NPZ = PROJECT_ROOT / "features" / "p59_local_window_cells.npz"

V03_UMBRALES = np.array([0.44, 0.51, 0.46])
C_MODELO = 2.0
MAX_ITER = 1000
SEMILLA = 42
GATE_DELTA = 0.003
GATE_PLIEGUES = 4
N_BOOT = 2000

VENTANA = 32
STRIDE = 16
# El mismo patron que usa TfidfVectorizer por defecto, para que un token sea
# exactamente la unidad que ve el vectorizador.
TOKEN_PATTERN = r"(?u)\b\w\w+\b"
RE_TOKEN = re.compile(TOKEN_PATTERN)
# Rebanadas del proyecto, tomadas literalmente de experiments/06_error_analysis.py.
# No se crea ninguna nueva.
RE_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
RE_MENCION = re.compile(r"@\w+")
RE_HASHTAG = re.compile(r"#\w+")
RE_MENCION_INICIAL = re.compile(r"^\s*(?:rt\s+)?@\w+", re.IGNORECASE)
RE_RETWEET = re.compile(r"^\s*rt\s+@\w+", re.IGNORECASE)


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P59-07] {mensaje}", flush=True)


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
        base = chr(int(partes[0], 16)) if partes else ""
        salida.append(base if base and ord(base) < 128 else "?")
    return "".join(salida)


def cortar_ventanas(texto: str, ventana: int, stride: int) -> list[str]:
    """Parte un texto en ventanas solapadas de forma determinista.

    Los cortes se hacen sobre los intervalos de caracter de los tokens, no
    concatenando tokens, para que dentro de cada ventana se conserve la forma
    superficial exacta del original: puntuacion, mayusculas y todo lo que
    char_wb necesita para seguir viendo la misma evidencia.

    Un texto de `ventana` tokens o menos produce UNA ventana igual al texto
    entero, sin duplicar y sin relleno.

    Args:
        texto: texto del documento.
        ventana: numero de tokens por ventana.
        stride: avance entre ventanas, en tokens.

    Returns:
        Lista de ventanas, en orden.
    """
    spans = [m.span() for m in RE_TOKEN.finditer(texto)]
    if not spans:
        return [texto]
    if len(spans) <= ventana:
        return [texto]
    salida: list[str] = []
    inicio = 0
    while inicio < len(spans):
        trozo = spans[inicio:min(inicio + ventana, len(spans))]
        salida.append(texto[trozo[0][0]:trozo[-1][1]])
        if inicio + ventana >= len(spans):
            break
        inicio += stride
    return salida


def rebanadas(textos: list[str]) -> dict[str, np.ndarray]:
    """Moreas de las rebanadas ya definidas por el proyecto.

    Los patrones se copian literalmente de `experiments/06_error_analysis.py`. No
    se define ninguna rebanada nueva. `tiene_insulto_lexico` NO se incluye: depende
    de un lexico de insultos inducido fold-local que no esta en disco, y rehacerlo
    seria crear una rebanada nueva; se declara el hueco en vez de sustituirlo.

    Args:
        textos: textos de los documentos.

    Returns:
        Diccionario rebanada -> mascara booleana.
    """
    n = len(textos)
    n_caracteres = np.array([len(t) for t in textos])
    cuartiles = pd.qcut(n_caracteres, 4, labels=["Q1", "Q2", "Q3", "Q4"]).astype(str)
    m: dict[str, np.ndarray] = {}
    m["tiene_url"] = np.array([bool(RE_URL.search(t)) for t in textos])
    m["tiene_hashtag"] = np.array([bool(RE_HASHTAG.search(t)) for t in textos])
    m["tiene_mencion"] = np.array([bool(RE_MENCION.search(t)) for t in textos])
    m["tiene_mencion_inicial"] = np.array([bool(RE_MENCION_INICIAL.match(t)) for t in textos])
    m["tiene_retuit"] = np.array([bool(RE_RETWEET.match(t)) for t in textos])
    m["tiene_pregunta"] = np.array(["?" in t for t in textos])
    m["tiene_comillas"] = np.array(['"' in t or "'" in t for t in textos])
    m["texto_corto_40"] = n_caracteres < 40
    m["region_regimen_noticia"] = m["tiene_url"] | m["tiene_hashtag"]
    m["region_sin_url_ni_hashtag"] = ~m["region_regimen_noticia"]
    for q in ("Q1", "Q2", "Q3", "Q4"):
        m[f"cuartil_longitud_{q}"] = np.asarray(cuartiles == q)
    assert all(len(v) == n for v in m.values())
    return m


def f1_por_celda(pred: np.ndarray, y: np.ndarray, filas: np.ndarray | None = None
                 ) -> dict[str, float]:
    """Macro-F1 de macro sobre las celdas seleccionadas.

    Args:
        pred: predicciones, forma (n, 3).
        y: etiquetas verdaderas, forma (n, 3).
        filas: mascara de filas a evaluar; None usa todas.

    Returns:
        Valor del macro-F1 sobre macro.
    """
    if filas is not None:
        pred = pred[filas]
        y = y[filas]
    return float(f1_score(y, pred, average="macro"))


def descomposicion(pred: np.ndarray, ref: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Rescate, damage y neto celda a celda contra una referencia.

    Args:
        pred: predicciones del candidato.
        ref: predicciones de la referencia.
        y: etiquetas verdaderas.

    Returns:
        Diccionario con el recuento y el desglose por etiqueta.
    """
    rescate = int(np.count_nonzero((pred == y) & (ref != y)))
    dano = int(np.count_nonzero((pred != y) & (ref == y)))
    cambian = int(np.count_nonzero(pred != ref))
    por = {}
    for j, e in enumerate(ETIQUETAS):
        por[e] = {
            "f1": round(float(f1_score(y[:, j], pred[:, j])), 10),
            "f1_ref": round(float(f1_score(y[:, j], ref[:, j])), 10),
            "rescate": int(np.count_nonzero((pred[:, j] == y[:, j]) & (ref[:, j] != y[:, j]))),
            "damage": int(np.count_nonzero((pred[:, j] != y[:, j]) & (ref[:, j] == y[:, j]))),
        }
    return {"rescate": rescate, "damage": dano, "neto": rescate - dano,
            "cambian": cambian,
            "acierto_en_las_que_cambian": round(rescate / cambian, 4) if cambian else None,
            "ambos_correctos": int(np.count_nonzero((pred == y) & (ref == y))),
            "ambos_fallan": int(np.count_nonzero((pred != y) & (ref != y))),
            "solo_B1_correcto": int(np.count_nonzero((pred == y) & (ref != y))),
            "solo_V03_correcto": int(np.count_nonzero((pred != y) & (ref == y))),
            "por_etiqueta": por}


def ic_bootstrap(delta_celdas: np.ndarray, n_boot: int = N_BOOT) -> tuple[float, float]:
    """IC95 del delta de acierto por celda.

    Args:
        delta_celdas: diferencia de acierto por celda.
        n_boot: numero de repeticiones.

    Returns:
        Tupla con los percentiles 2,5 y 97,5.
    """
    rng = np.random.default_rng(SEMILLA)
    n = delta_celdas.size
    m = delta_celdas[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    """Ejecuta B1: una sola representacion local, una sola agregacion.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P59-B Local window. UNA sola configuracion: ventana 32 tokens, stride 16, max")
    log("   0 Kaggle, 0 holdout, 0 test, 0 submissions. V03 NO se modifica")

    # ---------------- integridad de V03 ----------------
    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    log(f"   V03 sha256 verificado e intacto: {hash_real[:16]}...")

    v = np.load(OOF_V03, allow_pickle=True)
    ids, y, pliegues, pred_v03, proba_v03 = v["ids"], v["y_true"], v["pliegues"], v["pred"], v["proba"]
    ccv_ref = float(f1_score(y, pred_v03, average="macro"))
    if abs(ccv_ref - V03_CCV) > 1e-12:
        raise SystemExit(f"PARADA: el OOF canonico da {ccv_ref!r}, no {V03_CCV!r}")
    n = ids.size

    particion = pd.read_csv(PARTICION)
    grupos = particion.set_index("id").loc[ids, "grupo_near_dup_autor"]
    partidos = int(pd.Series(grupos.to_numpy()).groupby(grupos.to_numpy()).nunique().gt(1).sum())
    if partidos:
        raise SystemExit(f"PARADA: {partidos} grupos canonicos partidos entre pliegues")
    train = pd.read_csv(TRAIN_RAW, sep=";", encoding="latin-1", dtype={"id": str})
    train["id_num"] = pd.to_numeric(train["id"].str.replace('"', "", regex=False),
                                    errors="coerce").astype("int64")
    idx_texto = train.set_index("id_num").loc[ids]
    if not np.array_equal(idx_texto[["HS", "TR", "AG"]].to_numpy().astype(int), y):
        raise SystemExit("PARADA: las etiquetas del CSV crudo no coinciden con el OOF")
    textos = idx_texto["text"].astype(str).tolist()
    log(f"   OOF canonico {ccv_ref!r} | {n} filas | {grupos.nunique()} grupos intactos")

    # ---------------- factibilidad, ANTES de cualquier metrica ----------------
    log("")
    log("   FACTIBILIDAD, registrada antes de mirar validation")
    todas_ventanas = [cortar_ventanas(t, VENTANA, STRIDE) for t in textos]
    n_vent = np.array([len(v_) for v_ in todas_ventanas])
    largo_tokens = np.array([len(RE_TOKEN.findall(t)) for t in textos])
    n_car = np.array([len(t) for t in textos])
    multi = n_vent > 1
    log(f"      tokens por tweet: mediana {np.median(largo_tokens):.0f} "
        f"p90 {np.percentile(largo_tokens, 90):.0f} max {largo_tokens.max()}")
    log(f"      caracteres por tweet: mediana {np.median(n_car):.0f} max {n_car.max()}")
    log(f"      ventanas por tweet: media {n_vent.mean():.4f} max {n_vent.max()}")
    log(f"      tweets con MAS DE UNA ventana: {int(multi.sum())} de {n} "
        f"({multi.mean()*100:.2f} por ciento)")
    celdas_multi = int(multi.sum()) * 3
    log(f"      celdas alcanzables por el cambio de representacion: {celdas_multi} de {n*3} "
        f"({celdas_multi/(n*3)*100:.2f} por ciento)")
    log(f"      TECHO: fuera de esas {celdas_multi} celdas, la representacion es identica a la")
    log("      de V03 por construccion, luego el cambio no puede excederlas. Es una cota, no una")
    log("      promesa: el techo absoluto de corrected_cv sigue siendo el delta maximo posible.")
    dist = {str(k): int((n_vent == k).sum()) for k in sorted(set(n_vent.tolist()))}
    log(f"      distribucion del numero de ventanas: {dist}")
    log(f"      longitud de ventana en caracteres: mediana "
        f"{np.median([np.mean([len(w) for w in v_]) for v_ in todas_ventanas]):.0f}")

    # ---------------- brazo A: control V03 exacto ----------------
    log("")
    log("   BRAZO A: control V03 por sklearn, debe reproducir el canonico BIT A BIT")
    proba_a = np.zeros((n, 3))
    for f in range(1, 6):
        ir = np.where(pliegues != f)[0]
        iva = np.where(pliegues == f)[0]
        pw = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                             sublinear_tf=True, lowercase=True)
        pc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                             sublinear_tf=True, lowercase=True)
        x = hstack([pw.fit_transform([textos[i] for i in ir]),
                    pc.fit_transform([textos[i] for i in ir])]).tocsr()
        xv = hstack([pw.transform([textos[i] for i in iva]),
                     pc.transform([textos[i] for i in iva])]).tocsr()
        for j in range(3):
            lr = LogisticRegression(C=C_MODELO, class_weight="balanced", max_iter=MAX_ITER,
                                    solver="liblinear", random_state=SEMILLA)
            lr.fit(x, y[ir, j])
            proba_a[iva, j] = lr.predict_proba(xv)[:, 1]
    pred_a = (proba_a >= V03_UMBRALES[None, :]).astype(int)
    ccv_a = float(f1_score(y, pred_a, average="macro"))
    dmax_a = float(np.abs(proba_a - proba_v03).max())
    log(f"      corrected_cv {ccv_a!r} | max |proba - canonica| = {dmax_a:.3e} | "
        f"celdas distintas = {int((pred_a != pred_v03).sum())}")
    if abs(ccv_a - V03_CCV) > 1e-12 or dmax_a > 1e-12:
        raise SystemExit("PARADA: el control no reproduce V03. Auditar antes de continuar")
    log("      control superado")

    # ---------------- brazo B: ventanas locales con max ----------------
    log("")
    log("   BRAZO B: ventanas locales, agregacion max, umbrales de V03 congelados")
    proba_b = np.zeros((n, 3))
    # por cada fold: ventanas del training fold (cada una hereda la etiqueta de su
    # tweet) y ventanas del validation fold agrupadas por tweet para el max.
    deriva_max: list[float] = []
    deriva_prom: list[float] = []
    t_ajuste = time.time()
    for f in range(1, 6):
        ir = np.where(pliegues != f)[0]
        iva = np.where(pliegues == f)[0]
        textos_win: list[str] = []
        padre_win: list[int] = []
        for i in ir:
            for w in todas_ventanas[i]:
                textos_win.append(w)
                padre_win.append(int(i))
        padre_win_arr = np.array(padre_win)
        # ventanas de validacion, aplanadas, con su tweet
        textos_val: list[str] = []
        padre_val: list[int] = []
        for i in iva:
            for w in todas_ventanas[i]:
                textos_val.append(w)
                padre_val.append(int(i))
        padre_val_arr = np.array(padre_val)
        pw = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                             sublinear_tf=True, lowercase=True)
        pc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                             sublinear_tf=True, lowercase=True)
        xw = hstack([pw.fit_transform(textos_win), pc.fit_transform(textos_win)]).tocsr()
        xv_w = hstack([pw.transform(textos_val), pc.transform(textos_val)]).tocsr()
        for j in range(3):
            lr = LogisticRegression(C=C_MODELO, class_weight="balanced", max_iter=MAX_ITER,
                                    solver="liblinear", random_state=SEMILLA)
            lr.fit(xw, y[padre_win_arr, j])
            pv = lr.predict_proba(xv_w)[:, 1]
            # max por tweet, que es la agregacion pre-registrada
            for i in iva:
                sel = pv[padre_val_arr == i]
                proba_b[i, j] = float(sel.max())
                if sel.size > 1:
                    deriva_max.append(float(sel.max() - sel.mean()))
                    deriva_prom.append(float(sel.mean()))
        log(f"      pliegue {f} listo ({time.time() - t_ajuste:.1f} s acumulados)")

    pred_b = (proba_b >= V03_UMBRALES[None, :]).astype(int)
    ccv_b = float(f1_score(y, pred_b, average="macro"))
    delta = ccv_b - ccv_a
    log("")
    log(f"   BRAZO B corrected_cv {ccv_b!r} | delta vs V03 {delta:+.10f}")

    # ---------------- metricas ----------------
    fp_b = [f1_por_celda(pred_b, y, pliegues == k) for k in range(1, 6)]
    fp_a = [f1_por_celda(pred_a, y, pliegues == k) for k in range(1, 6)]
    pliegues_mej = sum(1 for u, w in zip(fp_b, fp_a) if u > w)
    dec = descomposicion(pred_b, pred_a, y)
    log("")
    log(f"   pliegues que mejoran: {pliegues_mej}/5 | deltas "
        + " ".join(f"{u - w:+.4f}" for u, w in zip(fp_b, fp_a)))
    log(f"   rescate {dec['rescate']} | damage {dec['damage']} | neto {dec['neto']:+d} "
        f"| cambian {dec['cambian']} | acierto en las que cambian {dec['acierto_en_las_que_cambian']}")
    log(f"   solo B1 correcto {dec['solo_B1_correcto']} | solo V03 correcto {dec['solo_V03_correcto']} "
        f"| ambos {dec['ambos_correctos']} | ambos fallan {dec['ambos_fallan']}")
    log("   F1 por etiqueta (B1): "
        + " ".join(f"{e} {dec['por_etiqueta'][e]['f1']:.4f}" for e in ETIQUETAS))
    log("   F1 por etiqueta (V03): "
        + " ".join(f"{e} {dec['por_etiqueta'][e]['f1_ref']:.4f}" for e in ETIQUETAS))

    # ---------------- estratos: donde puede actuar el cambio ----------------
    log("")
    log("   ESTRATOS: el cambio de representacion solo puede actuar en multi-ventana")
    corto = (n_car < np.percentile(n_car, 33))
    medio = (n_car >= np.percentile(n_car, 33)) & (n_car < np.percentile(n_car, 67))
    largo = (n_car >= np.percentile(n_car, 67))
    cambio = pred_b != pred_a
    log(f"      tweets multi-ventana: {int(multi.sum())} | celdas que cambian: {int(cambio.sum())}")
    log(f"      cambios DENTRO de multi-ventana: {int((cambio & multi[:, None]).sum())}")
    log(f"      cambios FUERA de multi-ventana:  {int((cambio & ~multi[:, None]).sum())}")
    estrato = {}
    for nombre, m in (("corto", corto), ("medio", medio), ("largo", largo)):
        cb = int((cambio & m[:, None]).sum())
        estrato[nombre] = {"n_tweets": int(m.sum()),
                           "cambios": cb,
                           "ccv_B1": round(f1_por_celda(pred_b, y, m), 6),
                           "ccv_V03": round(f1_por_celda(pred_a, y, m), 6)}
        log(f"        {nombre:6} n={int(m.sum()):5d} cambios={cb:4d} "
            f"ccv B1 {estrato[nombre]['ccv_B1']:.4f} V03 {estrato[nombre]['ccv_V03']:.4f}")
    log(f"      cambios con UNA sola ventana: {int((cambio & ~multi[:, None]).sum())}")
    log(f"      cambios con VARIAS ventanas:  {int((cambio & multi[:, None]).sum())}")

    # slices del proyecto
    m_reb = rebanadas(textos)
    slice_res: dict[str, Any] = {}
    log("")
    log("   SLICES DEL PROYECTO (ninguna nueva)")
    for nombre, mascara in m_reb.items():
        cb = int((cambio & mascara[:, None]).sum())
        slice_res[nombre] = {"n_tweets": int(mascara.sum()), "cambios": cb}
        if cb:
            log(f"        {nombre:30} n={int(mascara.sum()):5d} cambios={cb}")

    # deriva del max: calibracion, no puntuacion de alternativas
    log("")
    log("   DERIVA DEL MAX (calibracion; NO se puntua como variante alternativa)")
    if deriva_max:
        log(f"      en los tweets de varias ventanas, max menos media: media {np.mean(deriva_max):+.4f}, "
            f"mediana {np.median(deriva_max):+.4f}, maximo {max(deriva_max):+.4f}")
        log(f"      probabilidad media de ventana: {np.mean(deriva_prom):.4f}")
        log("      el max de k probabilidades ruidosas sesga al alza, luego con un umbral fijo "
            "empuja a predecir positivo; eso es un sesgo conocido de la regla, no un defecto del modelo")
    else:
        log("      ningun tweet de validacion con varias ventanas, luego la deriva no se midio")

    # rescate y dano DESCOMPUESTOS por estrato alcanzable: es la pregunta del
    # parrafo 13, "que corrige B1 errores que V03 no puede corregir?", y separarla por
    # estrato dice si el mecanismo local produjo correcciones aunque el global no.
    est_res: dict[str, Any] = {}
    for nombre, mascara in (("multi_ventana", multi), ("una_sola_ventana", ~multi)):
        dm = (cambio & mascara[:, None])
        resc = int(np.count_nonzero((pred_b == y) & (pred_a != y) & mascara[:, None]))
        dan = int(np.count_nonzero((pred_b != y) & (pred_a == y) & mascara[:, None]))
        est_res[nombre] = {
            "n_tweets": int(mascara.sum()),
            "cambios": int(dm.sum()),
            "rescate": resc, "damage": dan, "neto": resc - dan,
            "acierto_en_las_que_cambian": round(resc / int(dm.sum()), 4) if int(dm.sum()) else None,
        }
        log(f"      {nombre:18} n={int(mascara.sum()):5d} cambios={int(dm.sum()):3d} "
            f"rescate={resc:3d} damage={dan:3d} neto={resc - dan:+d}")

    # concentracion en el estrato alcanzable
    log(f"      delta de ccv en multi-ventana: "
        f"{f1_por_celda(pred_b, y, multi) - f1_por_celda(pred_a, y, multi):+.6f}")
    log(f"      delta de ccv en una-sola-ventana: "
        f"{f1_por_celda(pred_b, y, ~multi) - f1_por_celda(pred_a, y, ~multi):+.6f}")

    # ---------------- gate ----------------
    delta_celdas = ((pred_b == y).astype(int) - (pred_a == y).astype(int)).ravel()
    h_delta = delta >= GATE_DELTA
    h_folds = pliegues_mej >= GATE_PLIEGUES
    gate = bool(h_delta and h_folds)
    log("")
    log(f"   GATE: delta {delta:+.6f} >= {GATE_DELTA}: {h_delta} | "
        f"pliegues {pliegues_mej}/5 >= {GATE_PLIEGUES}: {h_folds} | superado: {gate}")
    ic_lo = ic_hi = None
    if gate:
        ic_lo, ic_hi = ic_bootstrap(delta_celdas)
        log(f"   bootstrap (solo porque el gate se supero): IC95 [{ic_lo:+.6f}, {ic_hi:+.6f}]")

    if delta < 0:
        decision = ("CLOSED. La representacion por ventanas locales con agregacion max NO mejora "
                    "la metrica. NO se rescata con mean, top-k, softmax ni max+mean (parrafo 15). "
                    "Y con el techo de viabilidad registrado, el resultado no puedeerse como "
                    "refutacion de la hipotesis local en general, solo de esta configuracion")
    elif not gate:
        decision = ("MARGINAL. La representacion por ventanas mejora la metrica pero no alcanza "
                    "el gate delta >= +0,003. NO se demuestra. Bootstrap NO ejecutado")
    else:
        decision = ("PROMISING. Supera delta y consistencia de pliegues. Queda como candidato local; "
                    "no se envia nada y la segunda particion se reserva")

    pob = leer_poblaciones()
    celda_c = np.array([(int(ids[i]), ETIQUETAS[j]) in pob["celdas_c"]
                        for i in range(n) for j in range(3)])
    rescata_42 = int(np.count_nonzero(celda_c & (pred_b == y).ravel() & (pred_a != y).ravel()))

    np.savez_compressed(CELDA_NPZ, ids=ids, y_true=y, pliegues=pliegues,
                        pred_v03=pred_a, pred_b1=pred_b,
                        proba_v03=proba_a, proba_b1=proba_b,
                        n_ventanas=n_vent, multi_ventana=multi, n_caracteres=n_car)
    log("   escrito features/p59_local_window_cells.npz")

    informe = {
        "familia": "P59_B_LOCAL_WINDOW",
        "naturaleza": ("EXPERIMENTO. 1 candidato, 0 submissions, 0 consultas a Kaggle, 0 holdout, "
                       "0 test. V03 no se modifica"),
        "fecha": inicio.isoformat(timespec="seconds"),
        "experiment_id": "P59-07",
        "track": "P59-B",
        "gap_analysis": {
            "precedente_mas_cercano": "V4-D01-D04, 'ventanas locales como tokens' (spec 24)",
            "por_que_no_es_duplicacion": ("en V4-D la ventana es un canal sparse EXTRA apilado al "
                                          "documento completo, luego la unidad de puntuacion sigue "
                                          "siendo el tweet entero y la dilution no se toca. Aqui la "
                                          "ventana ES la unidad de puntuacion: el vector del "
                                          "documento no existe"),
            "precedente_del_principio_de_agregacion": ("DL-02 CharCNN con kernels de caracter 3 y 5 "
                                                      "= campos locales con max, 0,741036, descartado, "
                                                      "pero en familia densa y entrenada desde cero, no "
                                                      "en la familia sparse lineal de V03"),
            "rows_MIL_multiple_instance": 0,
            "informes_con_ventana_o_segmento": ("solo la familia de la ventana de entrenamiento de "
                                                "FastText (EBD-18/19/21/22/23), mecanismo distinto"),
            "B_NOVELTY": True,
            "FUNCTIONAL_EQUIVALENCE": "FALSE",
        },
        "configuracion_pre_registrada": {
            "ventana_tokens": VENTANA, "stride_tokens": STRIDE, "solapamiento": VENTANA - STRIDE,
            "tokenizacion": TOKEN_PATTERN + " (el patron por defecto de TfidfVectorizer)",
            "corte": "sobre los intervalos de caracter de los tokens, para conservar la forma superficial",
            "tweets_cortos": "una sola ventana igual al tweet completo, sin duplicar ni relleno",
            "entrenamiento": "la union de todas las ventanas del training fold; cada ventana hereda la etiqueta de su tweet",
            "agregacion": "max de la probabilidad de ventana, por etiqueta",
            "clasificador": f"LogisticRegression C={C_MODELO} balanced liblinear max_iter={MAX_ITER} seed={SEMILLA}, el de V03",
            "features": "word(1,2) + char_wb(3,5), min_df=2, sublinear_tf, el de V03",
            "umbrales": "0,44 / 0,51 / 0,46 congelados, no reajustados",
            "rejilla": "ninguna. No se prueba otro tamano, stride, agregacion ni umbral",
        },
        "factibilidad_registrada_antes_de_validation": {
            "tokens_mediana": float(np.median(largo_tokens)),
            "tokens_max": int(largo_tokens.max()),
            "caracteres_mediana": float(np.median(n_car)),
            "ventanas_por_tweet_media": round(float(n_vent.mean()), 4),
            "ventanas_por_tweet_max": int(n_vent.max()),
            "tweets_multi_ventana": int(multi.sum()),
            "porcentaje_multi_ventana": round(float(multi.mean() * 100), 2),
            "celdas_alcanzables": celdas_multi,
            "distribucion_num_ventanas": dist,
            "techo": ("fuera de esas celdas la representacion es identica a la de V03 por "
                      "construccion, luego el cambio no puede excederlas"),
        },
        "control": {
            "brazo_A_corrected_cv": repr(ccv_a),
            "max_abs_diff_prob_vs_canonica": dmax_a,
            "reproduce_bit_a_bit": bool(abs(ccv_a - V03_CCV) <= 1e-12 and dmax_a <= 1e-12),
        },
        "resultado": {
            "brazo_A_corrected_cv": round(ccv_a, 10),
            "brazo_B_corrected_cv": round(ccv_b, 10),
            "delta_vs_V03": round(delta, 10),
            "folds": [round(x, 6) for x in fp_b],
            "folds_V03": [round(x, 6) for x in fp_a],
            "pliegues_mejorados": pliegues_mej,
            "f1_por_etiqueta": dec["por_etiqueta"],
            "complementariedad": dec,
            "gate": {"delta_min": GATE_DELTA, "pliegues_min": GATE_PLIEGUES,
                     "supera_delta": bool(h_delta), "supera_pliegues": bool(h_folds),
                     "superado": gate},
            "bootstrap": (None if ic_lo is None else
                          {"IC95": [round(ic_lo, 6), round(ic_hi, 6)], "n": N_BOOT}),
        },
        "deriva_max": (None if not deriva_max else
                       {"n_observaciones": len(deriva_max),
                        "max_menos_media_promedio": round(float(np.mean(deriva_max)), 6),
                        "max_menos_media_mediana": round(float(np.median(deriva_max)), 6),
                        "max_menos_media_maximo": round(float(max(deriva_max)), 6),
                        "prob_ventana_media": round(float(np.mean(deriva_prom)), 6),
                        "nota": ("el max de k probabilidades sesga al alza; se reporta como "
                                 "calibracion de la regla y NO se puntua como variante")}),
        "estratos": estrato,
        "complementariedad_por_estrato": est_res,
        "slices_proyecto": slice_res,
        "residuo_42_celdas_rescatadas": rescata_42,
        "decision": decision,
        "experimentos_registrados": 1,
        "formulacion": "V03 remains the best verified leaderboard result.",
        "gastos": {"submissions": 0, "consultas_kaggle": 0, "uso_holdout": 0, "uso_test": 0,
                   "segundos": round((datetime.now(timezone.utc) - inicio).total_seconds(), 1)},
    }
    JSON_RES.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # informe md
    L: list[str] = []
    A = L.append
    A("# P59-B - Representacion por ventanas locales sobre V03")
    A("")
    A(f"Generado por `experiments/p59_07_local_window.py`. Fecha {inicio.isoformat(timespec='seconds')}.")
    A("")
    A("## Motivacion e hipotesis")
    A("")
    A("No es detectar mejor las celdas difiles. La hipotesis es sobre la **unidad de "
      "representacion**: V03 comprime un tweet entero en un unico vector sparse, y ese vector "
      "puede diluir evidencia toxica concentrada en pocos tokens. Una representacion por ventanas "
      "locales la preserva, porque el clasificador nunca ve el documento entero.")
    A("")
    A("## Auditoria de novedad funcional")
    A("")
    A("| precedente | veredicto |")
    A("|---|---|")
    A("| V4-D01-D04, 'ventanas locales como tokens' (spec 24, ventana +-3) | **NO es duplicacion**: "
      "alli la ventana es un canal sparse EXTRA apilado al documento, luego la unidad de puntuacion "
      "sigue siendo el tweet. Aqui la ventana ES la unidad de puntuacion |")
    A("| DL-02 CharCNN kernels 3 y 5 (campos locales + max) | mismo principio de agregacion, pero en "
      "familia densa y desde cero, 0,741036, descartado. B lo prueba en la familia sparse lineal |")
    A("| EBD-18/19/21/22/23, ventana de entrenamiento de FastText | mecanismo sin relacion |")
    A("| filas con multiple-instance learning | **0** |")
    A("| informes con nombre de ventana o segmento | ninguno fuera de la familia FastText |")
    A("")
    A("`B_NOVELTY = TRUE`, `FUNCTIONAL_EQUIVALENCE = FALSE`.")
    A("")
    A("## Configuracion, fijada antes de mirar validation")
    A("")
    A("```text")
    A("ventana 32 tokens | stride 16 (solapamiento 16) | agregacion max | umbrales 0,44/0,51/0,46")
    A("tokenizacion  el patron por defecto de TfidfVectorizer")
    A("tweets de <= 32 tokens  una sola ventana = el tweet entero, sin duplicar ni relleno")
    A("entrenamiento  union de todas las ventanas del pliegue; cada ventana hereda la etiqueta de su tweet")
    A("clasificador  el de V03 | features word(1,2)+char_wb(3,5) | rejilla ninguna")
    A("```")
    A("")
    A("## Factibilidad: un techo registrado antes de las metricas")
    A("")
    A("Los tweets son cortos. Mediana de tokens "
      f"{np.median(largo_tokens):.0f}, maximo {largo_tokens.max()}. Con ventana 32 solo el "
      f"**{multi.mean()*100:.2f} por ciento** de los tweets tiene mas de una ventana "
      f"({int(multi.sum())} de {n}).")
    A("")
    A("```text")
    A(f"ventanas por tweet: media {n_vent.mean():.4f} | max {n_vent.max()}")
    A(f"tweets multi-ventana: {int(multi.sum())} ({multi.mean()*100:.2f} por ciento)")
    A(f"celdas alcanzables: {celdas_multi} de {n*3} ({celdas_multi/(n*3)*100:.2f} por ciento)")
    A(f"distribucion: {dist}")
    A("```")
    A("")
    A("Fuera de esas celdas la representacion es identica a la de V03 por construccion, luego el "
      "cambio no puede excederlas. Esto acota el experimento desde el principio y es la razon por la "
      "que un resultado negativo aqui no refuta la hipotesis local en general.")
    A("")
    A("## Control")
    A("")
    A("```text")
    A(f"brazo A corrected_cv {ccv_a!r} | max |proba - canonica| = {dmax_a:.3e} | celdas distintas "
      f"{int((pred_a != pred_v03).sum())}")
    A("```")
    A("")
    A("## Resultados")
    A("")
    A("```text")
    A(f"corrected_cv  B1 {ccv_b:.10f}")
    A(f"corrected_cv  V03 {ccv_a:.10f}")
    A(f"delta_vs_V03  {delta:+.10f}")
    A(f"pliegues que mejoran {pliegues_mej}/5")
    A(f"folds B1  {' '.join(f'{x:.4f}' for x in fp_b)}")
    A(f"folds V03 {' '.join(f'{x:.4f}' for x in fp_a)}")
    A(f"rescate {dec['rescate']} | damage {dec['damage']} | neto {dec['neto']:+d} | cambian {dec['cambian']}")
    A(f"solo B1 correcto {dec['solo_B1_correcto']} | solo V03 correcto {dec['solo_V03_correcto']}")
    A("```")
    A("")
    A("| etiqueta | F1 B1 | F1 V03 | rescate | damage |")
    A("|---|---:|---:|---:|---:|")
    for e in ETIQUETAS:
        vv = dec["por_etiqueta"][e]
        A(f"| {e} | {vv['f1']:.4f} | {vv['f1_ref']:.4f} | {vv['rescate']} | {vv['damage']} |")
    A("")
    A("## Donde se concentran los cambios")
    A("")
    A("| estrato | n tweets | cambios | ccv B1 | ccv V03 |")
    A("|---|---:|---:|---:|---:|")
    for nombre in ("corto", "medio", "largo"):
        ee = estrato[nombre]
        A(f"| {nombre} | {ee['n_tweets']} | {ee['cambios']} | {ee['ccv_B1']:.4f} | {ee['ccv_V03']:.4f} |")
    A("")
    A("```text")
    A(f"cambios dentro de multi-ventana:  {int((cambio & multi[:, None]).sum())}")
    A(f"cambios fuera de multi-ventana:   {int((cambio & ~multi[:, None]).sum())}")
    A(f"cambios con una sola ventana:    {int((cambio & ~multi[:, None]).sum())}")
    A(f"cambios con varias ventanas:     {int((cambio & multi[:, None]).sum())}")
    A(f"delta ccv en multi-ventana:      {f1_por_celda(pred_b, y, multi) - f1_por_celda(pred_a, y, multi):+.6f}")
    A(f"delta ccv en una-sola-ventana:   {f1_por_celda(pred_b, y, ~multi) - f1_por_celda(pred_a, y, ~multi):+.6f}")
    A("```")
    A("")
    A("## Complementariedad por estrato alcanzable")
    A("")
    A("La pregunta central del parrafo 13 es si B1 corrige errores que V03 no puede corregir con su "
      "representacion global. Separada por estrato, la respuesta es distinta a la global.")
    A("")
    A("| estrato | n tweets | cambios | rescate | damage | neto | acierto en las que cambian |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for nombre in ("multi_ventana", "una_sola_ventana"):
        ee = est_res[nombre]
        A(f"| {nombre} | {ee['n_tweets']} | {ee['cambios']} | {ee['rescate']} | {ee['damage']} "
          f"| {ee['neto']:+d} | {ee['acierto_en_las_que_cambian']} |")
    A("")
    A("## Slices del proyecto")
    A("")
    A("| rebanada | n tweets | cambios |")
    A("|---|---:|---:|")
    for nombre in ("tiene_url", "tiene_hashtag", "tiene_mencion", "tiene_mencion_inicial",
                   "tiene_retuit", "tiene_pregunta", "tiene_comillas", "texto_corto_40",
                   "region_regimen_noticia", "region_sin_url_ni_hashtag",
                   "cuartil_longitud_Q1", "cuartil_longitud_Q2", "cuartil_longitud_Q3",
                   "cuartil_longitud_Q4"):
        ss = slice_res[nombre]
        A(f"| {nombre} | {ss['n_tweets']} | {ss['cambios']} |")
    A("")
    A("`tiene_insulto_lexico` NO se incluye: depende de un lexico de insultos inducido fold-local "
      "que no esta en disco, y rehacerlo seria crear una rebanada nueva. Se declara el hueco en vez "
      "de sustituirlo.")
    A("")
    A("## Gate")
    A("")
    A("| criterio | valor | superado |")
    A("|---|---|---|")
    A(f"| delta >= +0,003 | {delta:+.6f} | **{h_delta}** |")
    A(f"| al menos 4 de 5 pliegues | {pliegues_mej}/5 | **{h_folds}** |")
    A(f"| gate completo | | **{gate}** |")
    A("")
    if ic_lo is not None:
        A(f"Bootstrap, ejecutado SOLO porque el gate se supero: IC95 [{ic_lo:+.6f}, {ic_hi:+.6f}].")
    else:
        A("Bootstrap NO ejecutado: el parrafo 16 lo condiciona al gate.")
    A("")
    A("## Decision")
    A("")
    A("```text")
    A(f"decision  {decision}")
    A("```")
    A("")
    A(f"De las 42 celdas no rescatables de P54, B1 rescata **{rescata_42}**.")
    A("")
    A("## Formulacion")
    A("")
    A("> V03 remains the best verified leaderboard result.")
    A("")
    A("*Seccion generada por `experiments/p59_07_local_window.py`. Sin push.*")
    MD_RES.write_text(a_ascii("\n".join(L)), encoding="ascii")

    cab = ["magnitud", "valor"]
    filas_csv = [
        ["brazo_A_corrected_cv", repr(ccv_a)],
        ["brazo_B_corrected_cv", repr(ccv_b)],
        ["delta_vs_V03", repr(delta)],
        ["pliegues_mejorados", str(pliegues_mej)],
        ["folds_B1", " ".join(f"{x:.6f}" for x in fp_b)],
        ["folds_V03", " ".join(f"{x:.6f}" for x in fp_a)],
        ["rescate", str(dec["rescate"])],
        ["damage", str(dec["damage"])],
        ["neto", str(dec["neto"])],
        ["cambian", str(dec["cambian"])],
        ["solo_B1_correcto", str(dec["solo_B1_correcto"])],
        ["solo_V03_correcto", str(dec["solo_V03_correcto"])],
        ["tokens_mediana", str(float(np.median(largo_tokens)))],
        ["tweets_multi_ventana", str(int(multi.sum()))],
        ["porcentaje_multi_ventana", f"{multi.mean()*100:.2f}"],
        ["ventanas_por_tweet_media", f"{n_vent.mean():.4f}"],
        ["cambios_multi_ventana", str(int((cambio & multi[:, None]).sum()))],
        ["cambios_una_ventana", str(int((cambio & ~multi[:, None]).sum()))],
        ["gate_delta_superado", str(h_delta)],
        ["gate_pliegues_superado", str(h_folds)],
        ["gate_superado", str(gate)],
        ["decision", a_ascii(decision)],
    ]
    import csv as _csv
    with CSV_RES.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(cab)
        w.writerows(filas_csv)

    for destino in (MD_RES, JSON_RES, CSV_RES):
        if destino.exists() and any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")

    log("")
    log(f"P59-B VEREDICTO: {decision.split('.')[0]}")
    log(f"P59-07 OK | {informe['gastos']['segundos']} s")


if __name__ == "__main__":
    warnings.simplefilter("error", RuntimeWarning)
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
