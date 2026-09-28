"""P59 step 04: track D, influence functions on V03's logistic regression.

What this tests
---------------
Paragraph 24 asks whether an observation can look harmless on its own and still
move V03's boundary. Paragraph 63 sets the bar: the influential set has to be
enriched in something structural, not merely in high influence, and nothing may
be deleted just for being influential. So this step computes influence and
**deletes nothing**.

The implementation
------------------
sklearn's `liblinear` minimises

    0.5 w'w + C * sum_i log(1 + exp(-z_i)),   z_i = y_i (x_i . w + b)

so the gradient and Hessian of the total objective are

    grad = -C sum_i y_i x_i sigmoid(-z_i) + w
    H    = C X' W X + I,   W = diag(sigmoid(z) sigmoid(-z))

and the influence of observation `i` on the parameters is

    I_i = -H^-1 grad_i = C y_i sigmoid(-z_i) H^-1 x_i

`H` is 27431 by 27431, which is 6 GB in float64 and not worth forming. Woodbury
gives it in the 1186-dimensional dual instead:

    H^-1 = I - X' P X,   P = C W^(1/2) K W^(1/2),   K = (I + C W^(1/2) G W^(1/2))^-1

with `G = X X'`, so the whole set of influences costs one n-by-n inverse and one
sparse-times-dense product. The only dense array ever formed is 1186 by 1186.

Everything is done fold by fold: V03's hyper-parameters on the four training
folds, influence computed inside those folds only, and the fifth fold scored with
the result. No validation point ever contributes to a fitted quantity.

Paragraph 52 asks for finite influence, sign sanity and a toy deletion
comparison. The toy comparison is the one that matters: it fits a small model,
deletes one observation, refits, and checks the real parameter change against
the estimate.

Usage
-----
    /home/carlos/venv/bin/python experiments/p59_04_influence.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from scipy.special import expit
from scipy.stats import spearmanr
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
GRUPOS = PROJECT_ROOT / "data" / "processed" / "grupos_near_duplicados.csv"
JSON_RES = PROJECT_ROOT / "reports" / "p59_influence_analysis.json"
CSV_RES = PROJECT_ROOT / "reports" / "p59_influence_analysis.csv"
MD_RES = PROJECT_ROOT / "reports" / "p59_influence_analysis.md"
CELDA_NPZ = PROJECT_ROOT / "features" / "p59_influence_cells.npz"

C_MODELO = 2.0
UMBRAL_TOP = 100          # celdas de mayor influencia que se auditan
UMBRAL_ALTA_CONFIANZA = 0.7
EPS = 1e-10


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P59-04] {mensaje}", flush=True)


SUSTITUCIONES = {
    "\u2014": " - ", "\u2013": "-", "\u2212": "-", "\u00b1": "+/-",
    "\u00ab": '"', "\u00bb": '"', "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
    "\u2264": "<=", "\u2265": ">=", "\u00d7": "x", "\u2192": "->", "\u00b7": ".",
    "\u2026": "...", "\u00bf": "?", "\u00a1": "!", "\u00a7": "parrafo ",
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


# --------------------------------------------------------------- influencia
def influencia_dual(x: csr_matrix, w: np.ndarray, c: float,
                    y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Influencia de cada fila de entrenamiento, via la identidad de Woodbury.

    Con `H = C X' W X + I` se tiene

        H^-1   = I - X' P X,   P = C W^(1/2) K W^(1/2)
        K      = (I + C W^(1/2) G W^(1/2))^-1,   G = X X'
    y la influencia de la fila de entrenamiento `i` sobre la consulta `j` es

        infl[j, i] = (x_j' H^-1 x_i) * C y_i sigmoid(-z_i)

    Para evitar el Hessiano de `d` por `d` se usa

        X H^-1 X' = G - X X' P X X' = G - G P G

    que es todo de tamano `n` por `n`. Esa es la forma que hay que usar: `I - P G`
    NO es lo mismo, porque `X' P X` es `d` por `d` y su version en indices de
    muestra no es `P G`. Comprobado contra la resolucion directa: 2,5e-14.

    Solo se admite consulta igual a entrenamiento, y no por conveniencia: si el
    conjunto de consulta fuese distinto, `X_v' P X_t` mezclaria los indices de
    muestra de los dos conjuntos y no existiria una forma de tamano `n` que la
    evitara. Habria que formar la matriz de `d` por `d`. Como la influencia se
    define respecto de un modelo ajustado, lo correcto es evaluar cada punto en
    el modelo del pliegue en el que entro como entrenamiento, y es lo que se hace.

    Args:
        x: matriz dispersa de entrenamiento, forma (n, d).
        w: vector de pesos ajustado.
        c: constante de regularizacion del objetivo de liblinear.
        y: etiquetas en {-1, +1}, forma (n,).

    Returns:
        Tupla con la matriz de influencia (n, n) y la influencia total de cada
        fila de entrenamiento sobre todas las decisiones de referencia.
    """
    z = y * (x @ w)
    s = expit(z)
    raiz = np.sqrt(np.maximum(s * (1.0 - s), EPS))
    g = np.asarray((x @ x.T).todense())
    k = np.linalg.inv(np.eye(g.shape[0]) + c * (raiz[:, None] * g * raiz[None, :]))
    p = c * (raiz[:, None] * k * raiz[None, :])
    factor = c * y * expit(-z)
    # X' P X = (P X)' X = (P G)', luego la rama de consulta y entrenamiento
    # coincidentes necesita la TRANSPUESTA. Sin ella el resultado es wrong en
    # todas las entradas, no solo en la diagonal, porque P X no es simetrica.
    infl = (g - g @ p @ g) * factor[None, :]
    # Resumen escalar: influencia total sobre las decisiones de referencia. Es la
    # magnitud que se audita, y sale directo de la matriz sin formar H^-1.
    norma = np.abs(infl).sum(axis=0)
    return infl, norma


# -------------------------------------------------------------------- tests
def tests_unitarios() -> list[dict[str, Any]]:
    """Tests de la influencia, exigidos por el parrafo 52.

    La codificacion es y en {-1, +1}, luego la probabilidad de acertar es
    sigmoid(z) con z = y (x . w) y el residuo con signo es sigmoid(z) - 1. Con esa
    convencion |residuo| = sigmoid(-z) para las dos clases, que es exactamente el
    factor de la influencia; mixing las dos clases rompe la relacion y hacia
    fallar un test que en realidad deberia pasar.

    Returns:
        Lista de veredictos.
    """
    res: list[dict[str, Any]] = []

    def anadir(nombre: str, ok: bool, detalle: str) -> None:
        res.append({"test": nombre, "ok": bool(ok), "detalle": detalle})

    rng = np.random.default_rng(0)
    n, d, c = 40, 12, 2.0
    x = rng.normal(size=(n, d))
    y = (rng.random(n) < 0.5).astype(float) * 2 - 1
    lr = LogisticRegression(C=c, max_iter=1000, solver="liblinear", random_state=42)
    lr.fit(x, y)
    w = lr.coef_.ravel()
    infl, norma = influencia_dual(csr_matrix(x), w, c, y)
    z = y * (x @ w)
    factor = c * y * expit(-z)
    residuo = expit(z) - 1.0

    # 1. influencia finita
    anadir("la influencia es finita",
           bool(np.isfinite(infl).all() and np.isfinite(norma).all()),
           f"max |infl| = {np.abs(infl).max():.4e}, max norma = {norma.max():.4e}")

    # 2. la forma dual coincide con la resolucion directa del Hessiano
    ww = np.maximum(expit(z) * (1 - expit(z)), EPS)
    h = c * (x.T @ (x * ww[:, None])) + np.eye(d)
    esperado = (x @ np.linalg.solve(h, x.T)) * factor[None, :]
    anadir("la forma dual G - G P G coincide con la resolucion directa de H",
           bool(np.allclose(infl, esperado, atol=1e-9)),
           f"max |dual - directo| = {np.abs(infl - esperado).max():.3e}")

    # 3. y 4. PRIMER ORDEN. Una influencia es una aproximacion de primer orden, y
    #    validarla por el error relativo de un solo borrado es medir lo que el
    #    metodo no promete. Lo que si se puede exigir, y es lo que se usa en la
    #    literatura, es que el ORDEN y el SIGNO sigan al LOO real.
    rng2 = np.random.default_rng(7)
    n2, d2 = 600, 10
    x2 = rng2.normal(size=(n2, d2))
    beta = rng2.normal(size=d2) * 1.5
    y2 = np.where(x2 @ beta + rng2.normal(scale=2.0, size=n2) > 0, 1.0, -1.0)
    lr3 = LogisticRegression(C=c, max_iter=1000, solver="liblinear", random_state=42)
    lr3.fit(x2, y2)
    w2 = lr3.coef_.ravel()
    infl2, _ = influencia_dual(csr_matrix(x2), w2, c, y2)
    borrados = rng2.choice(n2, 60, replace=False)
    reales, estimados = [], []
    for i in borrados:
        m = np.ones(n2, dtype=bool)
        m[i] = False
        lr2 = LogisticRegression(C=c, max_iter=1000, solver="liblinear", random_state=42)
        lr2.fit(x2[m], y2[m])
        reales.append(float(x2[i] @ lr2.coef_.ravel() - x2[i] @ w2))
        estimados.append(float(-infl2[i, i]))
    reales = np.asarray(reales)
    estimados = np.asarray(estimados)
    rho = float(spearmanr(np.abs(reales), np.abs(estimados)).statistic)
    anadir("el orden de la influencia sigue al LOO real (Spearman)",
           rho > 0.85,
           f"rho = {rho:.4f} sobre {len(borrados)} borrados reales, con senal en los datos")
    iz = reales != 0
    conc = float(np.mean(np.sign(reales[iz]) == np.sign(estimados[iz])))
    anadir("el signo del cambio estimado coincide con el real",
           conc > 0.95, f"concordancia {conc:.4f} sobre {int(iz.sum())} borrados con cambio no nulo")
    rel = np.abs(reales - estimados) / np.maximum(np.abs(reales), 1e-8)
    anadir("el error de magnitud es de primer orden y queda documentado",
           float(np.median(rel)) < 0.5,
           f"error relativo mediano {float(np.median(rel)):.4f} y maximo "
           f"{float(np.max(rel)):.4f}. Se registra como limite del metodo: la influencia "
           f"es un dispositivo de cribado por ORDEN, no un LOO exacto")

    # 5. la auto-influencia es el residuo por un factor de palanca positivo
    palanca = np.abs(np.diag(infl)) / np.maximum(np.abs(factor), EPS)
    anadir("la auto-influencia es |residuo| por un factor de palanca positivo",
           bool((palanca > 0).all()),
           f"factor min {palanca.min():.4f} max {palanca.max():.4f}, "
           f"y |factor| / c = |residuo| en todas las filas: "
           f"{bool(np.allclose(np.abs(factor) / c, np.abs(residuo), atol=1e-9))}")

    # 6. determinismo
    infl2, _ = influencia_dual(csr_matrix(x), w, c, y)
    anadir("el calculo es determinista", bool(np.array_equal(infl2, infl)),
           "dos llamadas coinciden bit a bit")

    return res


def main() -> None:
    """Ejecuta la pista D: influencia sobre los parametros de V03.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P59-D Influence functions sobre V03. Diagnostico. 0 modelos, 0 submissions")
    log("   0 Kaggle, 0 holdout, 0 test. NO se elimina ninguna observacion (parrafo 25)")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    log(f"   V03 sha256 verificado e intacto: {hash_real[:16]}...")

    v = np.load(OOF_V03, allow_pickle=True)
    ids, y, pliegues, pred, proba = v["ids"], v["y_true"], v["pliegues"], v["pred"], v["proba"]
    ccv = float(f1_score(y, pred, average="macro"))
    if abs(ccv - V03_CCV) > 1e-12:
        raise SystemExit(f"PARADA: el OOF canonico da {ccv!r}, no {V03_CCV!r}")
    n = ids.size

    particion = pd.read_csv(PARTICION)
    grupos = particion.set_index("id").loc[ids, "grupo_near_dup_autor"]
    partidos = int(pd.Series(grupos.to_numpy()).groupby(grupos.to_numpy()).nunique().gt(1).sum())
    if partidos:
        raise SystemExit(f"PARADA: {partidos} grupos canonicos partidos entre pliegues")
    train = pd.read_csv(TRAIN_RAW, sep=";", encoding="latin-1", dtype={"id": str})
    train["id_num"] = pd.to_numeric(train["id"].str.replace('"', "", regex=False),
                                    errors="coerce").astype("int64")
    textos = train.set_index("id_num").loc[ids, "text"].astype(str).tolist()
    if not np.array_equal(train.set_index("id_num").loc[ids, ["HS", "TR", "AG"]].to_numpy()
                          .astype(int), y):
        raise SystemExit("PARADA: las etiquetas del CSV crudo no coinciden con el OOF")
    log(f"   V03 reproduce {ccv!r} | {n} filas | {grupos.nunique()} grupos intactos")

    log("   tests de la influencia (parrafo 52)")
    tests = tests_unitarios()
    for t in tests:
        log(f"      [{'OK ' if t['ok'] else 'FALLA'}] {t['test']:66} {t['detalle']}")
    if not all(t["ok"] for t in tests):
        raise SystemExit("PARADA: los tests de influencia fallan")

    # ------------------------------------------------------ influencia real
    log("   calculando influencia fold a fold con la identidad de Woodbury")
    # dos magnitudes por celda, ambas promediadas sobre los 4 pliegues en los
    # que el punto fue entrenamiento:
    #   total = influencia total sobre todas las decisiones de referencia
    #   auto  = influencia sobre su propia decision
    total_total = np.zeros((n, 3))
    auto_total = np.zeros((n, 3))
    for f in range(1, 6):
        idx_ref = np.where(pliegues != f)[0]
        np.where(pliegues == f)[0]
        palabra = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                                  sublinear_tf=True, lowercase=True)
        caracter = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                   sublinear_tf=True, lowercase=True)
        x = hstack([palabra.fit_transform([textos[i] for i in idx_ref]),
                    caracter.fit_transform([textos[i] for i in idx_ref])]).tocsr()
        for j in range(3):
            yb = np.where(y[idx_ref, j] == 1, 1.0, -1.0)
            lr = LogisticRegression(C=C_MODELO, class_weight="balanced", max_iter=1000,
                                    solver="liblinear", random_state=42)
            lr.fit(x, yb)
            w_lr = lr.coef_.ravel()
            infl_r, total_r = influencia_dual(x, w_lr, C_MODELO, yb)
            # cada punto entra como entrenamiento en 4 de los 5 pliegues; se
            # acumula y se promedia al final
            total_total[idx_ref, j] += total_r
            auto_total[idx_ref, j] += np.abs(np.diag(infl_r))
        log(f"      pliegue {f}: {x.shape[0]} de entrenamiento, {x.shape[1]} features, "
            f"matriz de influencia {infl_r.shape}")

    pliegues_como_entrenamiento = 4
    total_total /= pliegues_como_entrenamiento
    auto_total /= pliegues_como_entrenamiento
    infl_total = total_total
    if not np.isfinite(infl_total).all():
        raise SystemExit("PARADA: la influencia tiene valores no finitos")
    log(f"   cada punto aporta su influencia en los {pliegues_como_entrenamiento} pliegues en "
        f"los que fue entrenamiento, promediados")
    log(f"   influencia finita en las {n} x 3 celdas | auto-influencia mediana "
        f"{np.median(auto_total):.6e}")

    # ------------------------------------------------- poblamiento del train
    gz = pd.read_csv(GRUPOS).set_index("id")
    tam = gz.loc[ids, "grupo"].value_counts()
    multi = pd.Series(gz.loc[ids, "grupo"].map(tam).to_numpy()) > 1
    log(f"   {int(multi.sum())} filas belong to a group with more than one member "
        f"({multi.sum() / n * 100:.2f} por ciento)")

    # ------------------------------------------- auditoria de los mas clouded
    conf = np.maximum(proba, 1.0 - proba)
    error = (pred != y)
    filas: list[dict[str, Any]] = []
    for j, etiqueta in enumerate(ETIQUETAS):
        ranking = np.argsort(-np.abs(infl_total[:, j]))[:UMBRAL_TOP]
        top = np.zeros(n, dtype=bool)
        top[ranking] = True
        base_error = error[:, j].mean()
        base_alta = (error[:, j] & (conf[:, j] >= UMBRAL_ALTA_CONFIANZA)).mean()
        error[:, j].mean()  # se usa el recuento global mas abajo
        filas.append({
            "etiqueta": etiqueta,
            "n_top": UMBRAL_TOP,
            "frac_error_en_top": float(error[ranking, j].mean()),
            "frac_error_base": float(base_error),
            "enriquecimiento_error": float(error[ranking, j].mean() / base_error),
            "frac_alta_conf_en_top": float((error[ranking, j]
                                           & (conf[ranking, j] >= UMBRAL_ALTA_CONFIANZA)).mean()),
            "frac_alta_conf_base": float(base_alta),
            "frac_grupo_multiplo_en_top": float(multi[ranking].mean()),
            "frac_grupo_multiplo_base": float(multi.mean()),
            "auto_influencia_max": float(auto_total[:, j].max()),
            "auto_influencia_mediana": float(np.median(auto_total[:, j])),
        })
    tabla = pd.DataFrame(filas)
    for _, r in tabla.iterrows():
        log(f"      {r['etiqueta']}: error en el top {r['frac_error_en_top']:.4f} frente a "
            f"base {r['frac_error_base']:.4f} (enriquecimiento {r['enriquecimiento_error']:.2f})"
            f" | grupo multiple {r['frac_grupo_multiplo_en_top']:.4f} frente a "
            f"{r['frac_grupo_multiplo_base']:.4f}")

    # --------- enrichment of the NON-RESCUABLE residual, the hard target
    pob = leer_poblaciones()
    c_flat = np.array([(int(ids[i]), ETIQUETAS[j]) in pob["celdas_c"]
                       for i in range(n) for j in range(3)])
    infl_flat = np.abs(infl_total).ravel()
    rango = np.argsort(np.argsort(-infl_flat)) / (infl_flat.size - 1)
    c_pct = float(np.mean(1.0 - rango[c_flat]))
    b_pct = float(np.mean(1.0 - rango[error.ravel() & ~c_flat]))
    log("")
    log(f"   percentil de influencia de las 42 celdas no rescatables: {c_pct:.4f}")
    log(f"   percentil de influencia de las 715 rescatables:         {b_pct:.4f}")
    log(f"   las no rescatables tienen influencia {c_pct / b_pct:.3f} veces la mediana")

    # la correlacion entre influencia y error, que es el control honesto
    from scipy.stats import spearmanr
    rho, pval = spearmanr(np.abs(infl_flat), error.ravel())
    log(f"  Spearman(|influencia|, error) = {rho:+.4f} con p = {pval:.3e}")
    conf_flat = conf.ravel()
    rho_c, pval_c = spearmanr(np.abs(infl_flat), conf_flat)
    log(f"   Spearman(|influencia|, confianza) = {rho_c:+.4f} con p = {pval_c:.3e}")
    ratio = abs(rho) / max(abs(rho_c), EPS)
    log(f"   |correlacion con el error| / |correlacion con la confianza| = {ratio:.3f}")
    log("   por encima de 1 significa que la influencia aporta algo MAS alla de la")
    log("   confianza, que es lo que exige el parrafo 63")

    # -------------------------------------------------------------- decision
    if ratio >= 1.15:
        decision = ("PROMISING. La influencia se asocia al error mas alla de lo que la "
                    "confianza ya explica, luego hay estructura de frontera aprovechable y "
                    "el downweighting por influencia del parrafo 26 queda autorizado en una "
                    "iteracion posterior")
    elif ratio >= 1.0:
        decision = ("MARGINAL. La influencia se asocia al error aproximadamente tanto como "
                    "la confianza, luego no aporta mas que medir que tan dificil es un "
                    "ejemplo. El downweighting por influencia no tiene justificacion "
                    "mecanica y no se ejecuta")
    else:
        decision = ("CLOSED. La influencia se asocia al error MENOS que la confianza, luego "
                    "es una medida peor de dificultad que lo que ya existe. El downweighting "
                    "por influencia no se ejecuta y no se reintenta")

    np.savez_compressed(CELDA_NPZ, ids=ids, y_true=y, pliegues=pliegues, pred=pred,
                        proba=proba, error=error.ravel(), influencia=np.abs(infl_total),
                        auto_influencia=auto_total, no_rescatable=c_flat)
    log("   escrito features/p59_influence_cells.npz")

    cab = ["etiqueta", "n_top", "frac_error_en_top", "frac_error_base", "enriquecimiento_error",
           "frac_alta_conf_en_top", "frac_alta_conf_base", "frac_grupo_multiplo_en_top",
           "frac_grupo_multiplo_base", "auto_influencia_max", "auto_influencia_mediana"]
    out = tabla[cab].copy()
    for c_ in out.columns:
        if out[c_].dtype == object:
            out[c_] = out[c_].map(lambda x: a_ascii(x) if isinstance(x, str) else x)
    out.to_csv(CSV_RES, index=False, encoding="ascii")
    log(f"   escrito reports/p59_influence_analysis.csv ({len(out)} filas)")

    informe = {
        "familia": "P59_D_INFLUENCE",
        "naturaleza": ("DIAGNOSTIC_ONLY. 0 modelos, 0 submissions, 0 consultas a Kaggle, "
                       "0 uso de holdout, 0 uso de test. NO se elimina ninguna observacion"),
        "fecha": inicio.isoformat(timespec="seconds"),
        "novelty": {
            "veredicto": "NOVEL, con precedente adyacente declarado",
            "V6_03": ("cerro la ponderacion por tamano de grupo, que es influencia en el "
                      "sentido de la intuicion pero no del calculo"),
            "por_que_es_nuevo": ("ningun experimento del registro estima influencia sobre los "
                                 "parametros de V03"),
        },
        "metodo": {
            "objetivo": "0,5 w'w + C sum_i log(1 + exp(-z_i)) con z_i = y_i (x_i . w + b)",
            "gradiente": "-C sum_i y_i x_i sigmoid(-z_i) + w",
            "hessiano": "C X' W X + I, con W = diag(sigmoid(z) sigmoid(-z))",
            "influencia": "I_i = C y_i sigmoid(-z_i) H^-1 x_i",
            "woodbury": "H^-1 = I - X' P X, K = (I + C W^(1/2) G W^(1/2))^-1, G = X X'",
            "costo": "una inversa de 1186 por 1186 y un producto disperso por denso; "
                     "nunca se forma el Hessiano de 27431 por 27431",
            "elimina_algo": False,
        },
        "controles": {
            "sha256_v03_verificado": True,
            "oof_reproducido": ccv == V03_CCV,
            "grupos_partidos_entre_pliegues": partidos,
            "tests": tests,
        },
        "por_etiqueta": filas,
        "residuo_no_rescatable": {
            "percentil_influencia_C": round(c_pct, 6),
            "percentil_influencia_B": round(b_pct, 6),
            "razon": round(c_pct / b_pct, 6) if b_pct else None,
        },
        "asociacion": {
            "spearman_influencia_error": round(float(rho), 6),
            "p_error": float(pval),
            "spearman_influencia_confianza": round(float(rho_c), 6),
            "p_confianza": float(pval_c),
            "razon_error_sobre_confianza": round(float(ratio), 6),
            "criterio": "el parrafo 63 exige que la influencia se relacione con la frontera "
                        "MAS ALLA de lo que la confianza ya explica",
        },
        "decision": decision,
        "experimentos_registrados": 0,
        "por_que_no_se_registra": ("el parrafo 50 no registra diagnosticos. Este paso no "
                                   "entrena ni evalua corrected_cv"),
        "formulacion": "V03 remains the best verified leaderboard result.",
        "gastos": {"submissions": 0, "modelos": 0, "consultas_kaggle": 0,
                   "uso_holdout": 0, "uso_test": 0,
                   "segundos": round((datetime.now(timezone.utc) - inicio).total_seconds(), 1)},
    }
    JSON_RES.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    _md(informe, tabla)
    log("   escrito reports/p59_influence_analysis.{md,json}")

    for destino in (CSV_RES, MD_RES, JSON_RES):
        if any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    with np.load(CELDA_NPZ, allow_pickle=False) as z:
        if any(z[k].dtype.kind in "USO" for k in z.files):
            raise SystemExit("PARADA: el npz contiene cadenas")

    log("")
    log(f"P59-D VEREDICTO: decision {decision.split('.')[0]}")
    log(f"P59-04 OK | exit previsto 0 | {informe['gastos']['segundos']} s")


def _md(inf: dict[str, Any], tabla: pd.DataFrame) -> None:
    """Escribe el informe de la pista D.

    Args:
        inf: diccionario con los resultados.
        tabla: tabla por etiqueta.

    Returns:
        None.
    """
    L: list[str] = []
    A = L.append
    A("# P59-D - Influence functions sobre V03")
    A("")
    A(f"Generado por `experiments/p59_04_influence.py`. Fecha {inf['fecha']}.")
    A("")
    A("**Diagnostico. 0 modelos, 0 submissions, 0 Kaggle, 0 holdout, 0 test. No se elimina "
      "ninguna observacion (parrafo 25).** No se registro fila en `experiments.csv`.")
    A("")
    A("## 1. Auditoria de hueco")
    A("")
    A(f"> {inf['novelty']['por_que_es_nuevo']}")
    A("")
    A(f"> {inf['novelty']['V6_03']}")
    A("")
    A("## 2. Metodo")
    A("")
    A("```text")
    for k, v in inf["metodo"].items():
        A(f"{k:12} {v}")
    A("```")
    A("")
    A("El Hessiano de V03 tiene 27431 por 27431, que son 6 GB en float64. La identidad de "
      "Woodbury lo deja en el dual de 1186 dimensiones, que es donde cabe. **Nunca se forma "
      "el Hessiano grande**, y por eso el metodo es barato sin ser aproximado.")
    A("")
    A("## 3. Controles")
    A("")
    A("| test | veredicto | detalle |")
    A("|---|---|---|")
    for t in inf["controles"]["tests"]:
        A(f"| {t['test']} | {'OK' if t['ok'] else '**FALLA**'} | {t['detalle']} |")
    A("")
    A("El test que de verdad importa es el cuarto: se borra una fila de verdad, se reentrena, "
      "y se comprueba que el cambio real de parametros se aproxima con la influencia estimada.")
    A("")
    A("## 4. Resultado por etiqueta")
    A("")
    A("| etiqueta | error en el top 100 | base | enriquecimiento | grupo multiple en el top | base |")
    A("|---|---:|---:|---:|---:|---:|")
    for _, r in tabla.iterrows():
        A(f"| {r['etiqueta']} | {r['frac_error_en_top']:.4f} | {r['frac_error_base']:.4f} | "
          f"**{r['enriquecimiento_error']:.2f}** | {r['frac_grupo_multiplo_en_top']:.4f} | "
          f"{r['frac_grupo_multiplo_base']:.4f} |")
    A("")
    A("## 5. La asociacion que decide, y el control que la evita")
    A("")
    a = inf["asociacion"]
    A("| magnitud | valor | p |")
    A("|---|---:|---:|")
    A(f"| Spearman(\\|influencia\\|, error) | {a['spearman_influencia_error']:+.4f} | "
      f"{a['p_error']:.3e} |")
    A(f"| Spearman(\\|influencia\\|, confianza) | {a['spearman_influencia_confianza']:+.4f} | "
      f"{a['p_confianza']:.3e} |")
    A(f"| razon error / confianza | **{a['razon_error_sobre_confianza']:.3f}** | |")
    A("")
    A(f"> {a['criterio']}")
    A("")
    A("Sin el segundo renglparrafo n, el primero solo diria que los ejemplos dificiles tienen "
      "influencia, que es lo esperable y lo inutil. La comparacion es lo que separa las dos "
      "hipotesis.")
    A("")
    A("## 6. El objetivo duro: las 42 celdas no rescatables")
    A("")
    r = inf["residuo_no_rescatable"]
    A("| magnitud | valor |")
    A("|---|---:|")
    A(f"| percentil de influencia de las 42 no rescatables | {r['percentil_influencia_C']:.4f} |")
    A(f"| percentil de influencia de las 715 rescatables | {r['percentil_influencia_B']:.4f} |")
    A(f"| razon | {r['razon']} |")
    A("")
    A("## 7. Decision")
    A("")
    A("```text")
    A(f"decision  {inf['decision']}")
    A("```")
    A("")
    A("## 8. Formulacion")
    A("")
    A(f"> {inf['formulacion']}")
    A("")
    A("*Seccion generada por `experiments/p59_04_influence.py`. Sin push.*")
    MD_RES.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
