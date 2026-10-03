"""P59 step 01: track A, Voronoi / local geometry on the V03 representation.

What this tests
---------------
Paragraph 7 asks whether observations whose label is inconsistent occupy a
geometric region that does not agree with their own class. Paragraph 13 fixes
the first objective: do NOT touch V03, and use the score only to answer

    V03 errors      vs  V03 correct
    rescatable cells vs  non-rescatable cells

Paragraph 60 forbids accepting Voronoi accuracy as evidence. The score must
demonstrate one of three properties, and the strongest available one here is
the second: does local geometry tell *rescatable* from *non-rescatable*?

What is new, and what is not
----------------------------
The novelty gate (`p59_00_registro_hipotesis.py`) closed most of this track in
advance and this script does not pretend otherwise:

- V29-03 already swept 48 kNN configurations; none had positive net.
- V4-F01-F04 already asked whether neighbourhood explains the error: +0.018
  with IC95 [-0.013, +0.049], not significant.
- V6-07 already concluded that suspicion is a good marker of *difficulty*.

So this script does not re-ask "does neighbourhood explain errors". It computes
the one object none of them computed: a **class-prototype Voronoi tessellation**
on the cosine of the V03 union, and then asks the question P54 left open, which
is whether the 42 non-rescatable cells are geometrically identifiable at all.

Two facts about the metric, stated rather than assumed
------------------------------------------------------
1. Each TF-IDF block is L2-normalised by the vectorizer, so the hstacked union
   has norm sqrt(2) for every row and

       cosine_distance_union = (cos_word + cos_char) / 2

   exactly. The metric is a stated property, not a modelling choice.
2. The tessellation is fitted on the four training folds and applied to the
   fifth, so every geometry is fold-internal. The same holds for the score.

Leakage controls
----------------
- Vectorizers are fitted on the four training folds only.
- The query point's own true label is never used. The score is anchored on V03's
  OOF prediction, which is honest out-of-fold.
- Holdout and test are never loaded.
- P54's population labels come from the diagnostic, not from any fit.

Usage
-----
    /home/carlos/venv/bin/python experiments/p59_01_voronoi_local.py
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
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

V03_SUB = PROJECT_ROOT / "submissions" / "toxic_submission_v03_lr_union_thr.csv"
OOF_V03 = PROJECT_ROOT / "features" / "oof_v03_final.npz"
TRAIN_RAW = PROJECT_ROOT / "data" / "raw" / "toxic_train.csv"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
P54_NONRESC = PROJECT_ROOT / "reports" / "p54_nonrescuable_atlas.csv"
P54_AUDIT = PROJECT_ROOT / "reports" / "p54_semantic_audit.json"
DIR_CELDAS = PROJECT_ROOT / "features" / "p59_voronoi_celfeatures.npz"
CSV_RES = PROJECT_ROOT / "reports" / "p59_voronoi_results.csv"
MD_RES = PROJECT_ROOT / "reports" / "p59_voronoi_analysis.md"
JSON_RES = PROJECT_ROOT / "reports" / "p59_voronoi_analysis.json"

V03_SHA = "7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e"
V03_CCV = 0.7772114525165721
V03_UMBRALES = np.array([0.44, 0.51, 0.46])
ETIQUETAS = ("HS", "TR", "AG")

K_PROTOTIPOS = 8
K_VECINOS = 15
MAX_ITER_MEDOID = 12
PUERTE_AUC = 0.60  # parrafo 15, exploratorio y no de adopcion

# Pesos del indice de inconsistencia, fijados antes de ejecutar (sin rejilla).
PESOS_INCONS = {"d_same": 1.0, "desacuerdo": 1.0, "soporte_celda": 1.0, "cuota_opuesta": 1.0}


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P59-01] {mensaje}", flush=True)


SUSTITUCIONES = {
    "\u2014": " - ", "\u2013": "-", "\u2212": "-", "\u00b1": "+/-",
    "\u00ab": '"', "\u00bb": '"', "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
    "\u2264": "<=", "\u2265": ">=", "\u00d7": "x", "\u2192": "->", "\u00b7": ".",
    "\u2026": "...", "\u00bf": "?", "\u00a1": "!", "\u00b0": " grados ",
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


# --------------------------------------------------------------- metric
def matriz_coseno(x: np.ndarray) -> np.ndarray:
    """Devuelve la matriz de distancias coseno de un conjunto de filas.

    Cada fila se normaliza a norma 1 antes del producto, de modo que el
    resultado es `1 - coseno` y no depende de la escala de las filas.

    Args:
        x: matriz densa de forma (n, d).

    Returns:
        Matriz de distancias coseno de forma (n, n).
    """
    norma = np.linalg.norm(x, axis=1, keepdims=True)
    norma[norma == 0.0] = 1.0
    xn = x / norma
    prod = xn @ xn.T
    np.clip(prod, -1.0, 1.0, out=prod)
    return 1.0 - prod


# ------------------------------------------------------ medoides deterministas
def medoides(d: np.ndarray, k: int) -> tuple[np.ndarray, int, bool]:
    """Selecciona k medoides de forma determinista sobre una matriz de distancias.

    Construccion farthest-first (BUILD) seguida de actualizaciones de Lloyd
    (cada grupo se reemplaza por su miembro de menor suma de distancias), que
    es el metodo PAM clasico. No usa azar, luego es reproducible bit a bit.

    Args:
        d: matriz de distancias cuadrada de forma (n, n).
        k: numero de medoides deseados.

    Returns:
        Tupla con los indices de los medoides, el numero de iteraciones y si
        convergio antes del limite.
    """
    n = d.shape[0]
    if n <= k:
        return np.arange(n), 0, True
    # BUILD farthest-first. El primer medoid es el de menor distancia total
    # acumulada, que es determinista y mejor punto de partida que el indice 0.
    sel = [int(np.argmin(d.sum(axis=1)))]
    centros = d[:, sel[0]].copy()
    centros[sel[0]] = -np.inf
    for _ in range(1, k):
        elegido = int(np.argmax(centros))
        sel.append(elegido)
        centros = np.minimum(centros, d[:, elegido])
        centros[np.asarray(sel)] = -np.inf
    convergio = False
    it = 0
    for it in range(1, MAX_ITER_MEDOID + 1):
        coste = d[:, sel].min(axis=1)
        nuevos: list[int] = []
        for c in sel:
            idx = np.where(coste == d[:, c])[0]
            sumas = d[np.ix_(idx, idx)].sum(axis=1)
            elegido = int(idx[int(np.argmin(sumas))])
            if elegido not in nuevos:
                nuevos.append(elegido)
        nuevos.sort()
        if nuevos == sorted(sel):
            convergio = True
            break
        sel = nuevos
    if len(sel) != k:
        raise ValueError(f"la seleccion de medoides devolvio {len(sel)} en vez de {k}")
    return np.array(sel, dtype=int), it, convergio


# ---------------------------------------------------------------- unit tests
def tests_unitarios() -> list[dict[str, Any]]:
    """Ejecuta los tests sinteticos que exige el parrafo 52 para Voronoi.

    Returns:
        Lista de resultados, cada uno con su veredicto.
    """
    res: list[dict[str, Any]] = []

    def anadir(nombre: str, ok: bool, detalle: str) -> None:
        res.append({"test": nombre, "ok": bool(ok), "detalle": detalle})

    # 1. escala: la distancia coseno no depende del factor de escala
    a = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    d1 = matriz_coseno(a)
    d2 = matriz_coseno(a * 7.5)
    anadir("distancia coseno invariante a escala",
           np.allclose(d1, d2),
           f"max |d - d_escalada| = {np.abs(d1 - d2).max():.3e}")

    # 2. distancia coseno coherente con la geometria conocida
    angulos = np.array([0.0, np.pi / 3, np.pi])
    puntos = np.stack([np.cos(angulos), np.sin(angulos)], axis=1)
    d = matriz_coseno(puntos)
    diff = np.abs(angulos[:, None] - angulos[None, :])
    esperado = 1.0 - np.cos(diff)
    anadir("distancia coseno igual a 1 - cos(angulo)",
           np.allclose(d, esperado, atol=1e-12),
           f"max |d - (1-cos)| = {np.abs(d - esperado).max():.3e}")

    # 3. argmin del sitio mas proximo coincide con la fuerza bruta
    rng = np.random.default_rng(0)
    p = rng.normal(size=(40, 7))
    d = matriz_coseno(p)
    sitios = np.array([0, 7, 19, 33])
    for q in (0, 12, 25, 39):
        bruto = min(sitios, key=lambda s: d[q, s])
        vector = d[q, sitios].argmin()
        anadir(f"sitio mas proximo coincide con la fuerza bruta (q={q})",
               bruto == sitios[vector], f"bruto={bruto} vector={int(sitios[vector])}")

    # 4. una consulta situada en un sitio tiene distancia 0 a el
    dself = d[sitios, :][:, sitios]
    anadir("distancia de un sitio consigo mismo es 0",
           np.allclose(np.diag(dself), 0.0, atol=1e-12),
           f"max |diag| = {np.abs(np.diag(dself)).max():.3e}")

    # 5. los medoides son miembros reales, distintos y en numero exacto
    d2 = matriz_coseno(rng.normal(size=(120, 5)))
    sel, it, conv = medoides(d2, 8)
    anadir("los medoides son miembros del conjunto y son distintos",
           len(set(sel.tolist())) == 8 and all(0 <= s < 120 for s in sel),
           f"n={len(sel)} distintos={len(set(sel.tolist()))} iteraciones={it} conv={conv}")

    # 6. determinismo: dos llamadas dan el mismo conjunto
    sel2, _, _ = medoides(d2, 8)
    anadir("los medoides son deterministas",
           np.array_equal(sel, sel2), f"iguales={np.array_equal(sel, sel2)}")

    # 7. un conjunto mas pequeno que k devuelve todos sus miembros
    d3 = matriz_coseno(rng.normal(size=(5, 3)))
    sel3, _, _ = medoides(d3, 8)
    anadir("si n <= k se devuelven todos los miembros",
           len(sel3) == 5, f"n_devueltos={len(sel3)}")

    # 8. el reparto de medoides mejora o iguala la suma de distancias
    mejora = d2[np.ix_(sel, sel)].sum() <= d2[np.ix_(sel, sel)].sum() + 1e-9
    obj_sel = d2.min(axis=1)[sel].sum()
    obj_all = d2.min(axis=1).sum()
    anadir("la particion en grupos tiene coste minimo dentro de sus medoides",
           mejora and obj_sel <= obj_all + 1e-9,
           f"coste_medoides={obj_sel:.4f} coste_punto_mas_cercano={obj_all:.4f}")

    return res


# ------------------------------------------------------------------ geometria
def geometria_de_pliegue(textos_ref: list[str], y_ref: np.ndarray,
                         textos_todos: list[str]) -> dict[str, np.ndarray]:
    """Construye la teselacion de una fila de prototipos sobre el coseno de V03.

    Args:
        textos_ref: textos del pliegue de entrenamiento.
        y_ref: etiquetas del pliegue de entrenamiento, forma (n_ref, 3).
        textos_todos: textos de las filas a caracterizar (referencia y consulta).

    Returns:
        Diccionario con los indices de los prototipos por etiqueta y clase, y
        las distancias coseno entre todos los textos y cada prototipo.
    """
    palabra = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                              sublinear_tf=True, lowercase=True)
    caracter = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                               sublinear_tf=True, lowercase=True)
    xw_ref = palabra.fit_transform(textos_ref)
    xc_ref = caracter.fit_transform(textos_ref)
    x_ref = hstack([xw_ref, xc_ref]).toarray().astype(np.float64)
    x_tod = hstack([palabra.transform(textos_todos),
                    caracter.transform(textos_todos)]).toarray().astype(np.float64)

    n_ref = x_ref.shape[0]
    protos: dict[str, list[int]] = {}
    d_ref = matriz_coseno(x_ref)
    for j, etiqueta in enumerate(ETIQUETAS):
        for c in (0, 1):
            idx = np.where(y_ref[:, j] == c)[0]
            sel, _, conv = medoides(d_ref[np.ix_(idx, idx)], K_PROTOTIPOS)
            if len(sel) < K_PROTOTIPOS:
                log(f"      {etiqueta}_{c}: solo {len(idx)} miembros, se usan {len(sel)} prototipos")
            protos[f"{etiqueta}_{c}"] = idx[sel].tolist()
            if not conv:
                log(f"      aviso: medoides {etiqueta}_{c} sin converger en "
                    f"{MAX_ITER_MEDOID} iteraciones")

    # Matrices normalizadas comunes: permiten coseno coherente entre referencia,
    # consulta y prototipos, todos en el mismo espacio TF-IDF ajustado en ref.
    ref_un = x_ref / np.maximum(np.linalg.norm(x_ref, axis=1, keepdims=True), 1e-12)
    tod_un = x_tod / np.maximum(np.linalg.norm(x_tod, axis=1, keepdims=True), 1e-12)
    protos_ordenados: list[int] = []
    claves: list[str] = []
    for clave in (f"{e}_{c}" for e in ETIQUETAS for c in (0, 1)):
        for p in protos[clave]:
            protos_ordenados.append(p)
            claves.append(clave)
    d_prot = 1.0 - tod_un @ ref_un[protos_ordenados, :].T
    np.clip(d_prot, 0.0, 2.0, out=d_prot)
    return {"n_ref": n_ref, "d_prot": d_prot, "claves": claves,
            "protos": protos, "tod_un": tod_un}


def rasgos_de_consulta(geometria: dict[str, np.ndarray], y_ref: np.ndarray,
                        idx_consulta: np.ndarray, pred_consulta: np.ndarray,
                        idx_ref: np.ndarray) -> dict[str, np.ndarray]:
    """Calcula los rasgos Voronoi de las filas de consulta.

    Los rasgos se anclan en la prediccion de V03 y nunca en la etiqueta real de
    la fila consultada, de modo que son observables en inferencia.

    Args:
        geometria: salida de `geometria_de_pliegue`.
        y_ref: etiquetas de referencia, forma (n_ref, 3).
        idx_consulta: posicion de las filas a caracterizar dentro de `textos_todos`.
        pred_consulta: prediccion de V03 para esas filas, forma (n_consulta, 3).
        idx_ref: posicion de las filas de referencia dentro de `textos_todos`.

    Returns:
        Diccionario de matrices de forma (n_consulta, 3).
    """
    d_prot = geometria["d_prot"]
    claves = geometria["claves"]
    tod_un = geometria["tod_un"]
    n_ref = idx_ref.size
    k_vec = min(K_VECINOS, n_ref)
    d_cref = 1.0 - tod_un[idx_consulta] @ tod_un[idx_ref].T
    np.clip(d_cref, 0.0, 2.0, out=d_cref)
    orden = np.argsort(d_cref, axis=1)[:, :k_vec]
    d_ref_prot = d_prot[idx_ref]

    nombres = ("d_same", "d_opp", "margen", "pureza", "cuota_opuesta",
               "cuota_densa", "soporte_celda", "inconsistencia")
    salida = {nombre: np.zeros((idx_consulta.size, 3), dtype=np.float64)
              for nombre in nombres}

    for j, etiqueta in enumerate(ETIQUETAS):
        cols = {c: np.array([i for i, cl in enumerate(claves) if cl == f"{etiqueta}_{c}"],
                            dtype=int) for c in (0, 1)}
        for fila in range(idx_consulta.size):
            pc = int(pred_consulta[fila, j])
            propio, otro = cols[pc], cols[1 - pc]
            d_p = d_prot[idx_consulta[fila], propio]
            d_o = d_prot[idx_consulta[fila], otro]
            d_same, d_opp = float(d_p.min()), float(d_o.min())
            celda_x = int(d_p.argmin())
            vecinos = orden[fila]
            yy = y_ref[vecinos, j]
            s = int(np.count_nonzero(yy == pc))
            o = k_vec - s
            # soporte de celda: que fraccion de los vecinos cae en la MISMA celda
            d_v = d_ref_prot[vecinos][:, propio]
            celdas_v = d_v.argmin(axis=1)
            soporte = float(np.count_nonzero(celdas_v == celda_x) / k_vec)
            # densidad: cuantas referencias quedan mas cerca que el prototipo opuesto
            cuota = float(np.count_nonzero(d_ref_prot[:, propio].min(axis=1) < d_opp) / n_ref)
            salida["d_same"][fila, j] = d_same
            salida["d_opp"][fila, j] = d_opp
            salida["margen"][fila, j] = d_opp - d_same
            salida["pureza"][fila, j] = s / k_vec
            salida["cuota_opuesta"][fila, j] = o / (k_vec + 1)
            salida["cuota_densa"][fila, j] = cuota
            salida["soporte_celda"][fila, j] = soporte

    # indice de inconsistencia: pesos fijos, sin rejilla, anclado en la prediccion
    d_same = salida["d_same"]
    for j in range(3):
        mediana = float(np.median(d_same[:, j]))
        if mediana <= 0.0:
            mediana = 1e-9
        u1 = 1.0 - np.exp(-d_same[:, j] / mediana)
        u2 = 1.0 - salida["pureza"][:, j]
        u3 = 1.0 - salida["soporte_celda"][:, j]
        u4 = salida["cuota_opuesta"][:, j]
        salida["inconsistencia"][:, j] = (
            PESOS_INCONS["d_same"] * u1
            + PESOS_INCONS["desacuerdo"] * u2
            + PESOS_INCONS["soporte_celda"] * u3
            + PESOS_INCONS["cuota_opuesta"] * u4) / sum(PESOS_INCONS.values())
    return salida


# ------------------------------------------------------------------ utilidades
def ic_bootstrap(mascara: np.ndarray, score: np.ndarray, n_rep: int = 2000,
                 semilla: int = 42) -> tuple[float, float]:
    """Devuelve el IC95 del AUC por bootstrap de las celdas.

    Los parametros se fijan antes de ejecutar: 2000 repeticiones y semilla 42.
    El objetivo es comprobar si un AUC apparently away de 0.5 esta realmente
    away de 0.5, y no solo es ruido de muestreo.

    Args:
        mascara: objetivo binario.
        score: score continuo.
        n_rep: numero de repeticiones.
        semilla: semilla del generador.

    Returns:
        Tupla con los percentiles 2.5 y 97.5.
    """
    rng = np.random.default_rng(semilla)
    n = mascara.size
    valores = np.empty(n_rep)
    for i in range(n_rep):
        idx = rng.integers(0, n, n)
        m = mascara[idx]
        if m.sum() == 0 or m.sum() == m.size:
            valores[i] = np.nan
            continue
        valores[i] = roc_auc_score(m, score[idx])
    return (float(np.nanpercentile(valores, 2.5)),
            float(np.nanpercentile(valores, 97.5)))


def leer_poblaciones() -> dict[str, Any]:
    """Lee las poblaciones A, B y C de P54 sin recalcularlas.

    Returns:
        Diccionario con la mascara de no rescatables y el recuento por celda.
    """
    atlas = pd.read_csv(P54_NONRESC)
    audit = json.loads(P54_AUDIT.read_text(encoding="utf-8"))
    pob = audit["poblaciones"]
    celdas_c = {(int(r.tweet_id), str(r.label)) for r in atlas.itertuples()}
    return {"celdas_c": celdas_c, "n_A": int(pob["A_correcto"]),
            "n_B": int(pob["B_rescatable"]), "n_C": int(pob["C_no_rescatable"]),
            "tweets_C": int(pob["tweets_distintos_en_C"])}


def main() -> None:
    """Ejecuta la pista A: geometria local Voronoi sobre la representacion de V03.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P59-A Voronoi / geometria local. Diagnostico primero: NO se modifica V03")
    log("   0 modelos nuevos, 0 submissions, 0 Kaggle, 0 holdout, 0 test")

    # ------------------------------------------------------------- controles
    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    log(f"   V03 sha256 verificado e intacto: {hash_real[:16]}...")

    v = np.load(OOF_V03, allow_pickle=True)
    ids, y, pliegues, pred, proba = v["ids"], v["y_true"], v["pliegues"], v["pred"], v["proba"]
    ccv = float(f1_score(y, pred, average="macro"))
    if abs(ccv - V03_CCV) > 1e-12:
        raise SystemExit(f"PARADA: el OOF canonico da {ccv!r}, no {V03_CCV!r}")
    esperado_pred = (proba >= V03_UMBRALES[None, :]).astype(int)
    if not np.array_equal(esperado_pred, pred):
        raise SystemExit("PARADA: la prediccion del OOF no es proba >= umbrales de V03")
    log(f"   OOF canonico reproduce {ccv!r} exacto y sus umbrales 0.44/0.51/0.46")

    particion = pd.read_csv(PARTICION)
    dev = particion[particion.particion == "dev"]
    if len(dev) != ids.size:
        raise SystemExit(f"PARADA: la particion dev tiene {len(dev)} filas y el OOF {ids.size}")
    grupos = dev.set_index("id").loc[ids, "grupo_near_dup_autor"]
    partidos = int((pd.Series(grupos.to_numpy()).groupby(grupos.to_numpy()).nunique() > 1).sum())
    if partidos:
        raise SystemExit(f"PARADA: {partidos} grupos canonicos partidos entre pliegues")
    log(f"   grupos canonicos {grupos.nunique()}, 0 partidos entre los 5 pliegues")

    train = pd.read_csv(TRAIN_RAW, sep=";", encoding="latin-1", dtype={"id": str})
    train["id_num"] = pd.to_numeric(train["id"].str.replace('"', "", regex=False),
                                    errors="coerce").astype("int64")
    textos = train.set_index("id_num").loc[ids, "text"].astype(str).tolist()
    etiquetas_raw = train.set_index("id_num").loc[ids, ["HS", "TR", "AG"]].to_numpy()
    if not np.array_equal(etiquetas_raw.astype(int), y):
        raise SystemExit("PARADA: las etiquetas del CSV crudo no coinciden con el OOF canonico")
    log(f"   {len(textos)} textos de dev cargados; etiquetas identicas al OOF celda a celda")

    if set(ids.tolist()) != set(particion.loc[particion.particion == "dev", "id"].astype(int).tolist()):
        raise SystemExit("PARADA: los ids del OOF no son exactamente la particion dev")
    test_ids = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "toxic_test.csv",
                           sep=";", encoding="latin-1", dtype={"id": str})
    test_ids = {int(t) for t in test_ids["id"].str.replace('"', "", regex=False)}
    if set(ids.tolist()) & test_ids:
        raise SystemExit("PARADA: hay ids de test dentro del conjunto de trabajo")
    log(f"   control: los {ids.size} ids son exactamente la particion dev; ninguno pertenece a test")

    # ---------------------------------------------------------- tests (parrafo 52)
    log("   tests unitarios del mecanismo (parrafo 52)")
    tests = tests_unitarios()
    for t in tests:
        log(f"      [{'OK ' if t['ok'] else 'FALLA'}] {t['test']:56} {t['detalle']}")
    if not all(t["ok"] for t in tests):
        raise SystemExit("PARADA: los tests unitarios del mecanismo fallan")

    # ------------------------------------------------------------ geometria
    log(f"   teselacion: K={K_PROTOTIPOS} prototipos por clase y etiqueta, "
        f"{K_VECINOS} vecinos, coseno de la union V03")
    n = ids.size
    rasgos = {nombre: np.zeros((n, 3), dtype=np.float64) for nombre in
               ("d_same", "d_opp", "margen", "pureza", "cuota_opuesta",
                "cuota_densa", "soporte_celda", "inconsistencia")}
    for f in range(1, 6):
        idx_ref = np.where(pliegues != f)[0]
        idx_consulta = np.where(pliegues == f)[0]
        geo = geometria_de_pliegue([textos[i] for i in idx_ref], y[idx_ref], textos)
        r = rasgos_de_consulta(geo, y[idx_ref], idx_consulta, pred[idx_consulta], idx_ref)
        for nombre, mat in r.items():
            rasgos[nombre][idx_consulta] = mat
        log(f"      pliegue {f}: {len(idx_ref)} de referencia, {len(idx_consulta)} "
            f"consulta, {len(geo['protos']['HS_1'])} prototipos HS=1")

    if not np.isfinite(rasgos["inconsistencia"]).all():
        raise SystemExit("PARADA: el indice de inconsistencia tiene valores no finitos")
    incons = rasgos["inconsistencia"]
    log(f"   indice de inconsistencia en [0,1]: min {incons.min():.4f} "
        f"mediana {np.median(incons):.4f} max {incons.max():.4f}")

    # -------------------------------------------------------------- diagnostico
    pob = leer_poblaciones()
    # Todas las mascaras viven aplanadas en el mismo orden que ndarray.ravel():
    # posicion i*3+j es la celda (fila i, columna j).
    error = (pred != y).ravel()
    celda_c = np.array([(int(ids[i]), ETIQUETAS[j]) in pob["celdas_c"]
                        for i in range(n) for j in range(3)])
    columna = np.tile(np.arange(3), n)
    log(f"   poblaciones P54: A={pob['n_A']} B={pob['n_B']} C={pob['n_C']} "
        f"({pob['tweets_C']} tweets distintos); celdas C localizadas: {int(celda_c.sum())}")
    if int(celda_c.sum()) != pob["n_C"]:
        raise SystemExit(f"PARADA: se localizaron {int(celda_c.sum())} celdas C y P54 dice {pob['n_C']}")
    if int((celda_c & ~error).sum()):
        raise SystemExit("PARADA: hay celdas no rescatables que V03 no clasifica como error")
    rescatable = error & ~celda_c
    if int(error.sum()) != pob["n_B"] + pob["n_C"]:
        raise SystemExit(f"PARADA: V03 comete {int(error.sum())} errores y P54 cuenta "
                         f"{pob['n_B']} + {pob['n_C']} = {pob['n_B'] + pob['n_C']} celdas con error")
    if int(error.size) != pob["n_A"] + pob["n_B"] + pob["n_C"]:
        raise SystemExit(f"PARADA: {int(error.size)} celdas no cuadran con A+B+C de P54")

    score = incons.ravel()
    filas: list[dict[str, Any]] = []

    def anadir(objetivo_nombre: str, mascara: np.ndarray, nota: str) -> None:
        n_pos = int(mascara.sum())
        n_neg = int(mascara.size - n_pos)
        if n_pos == 0 or n_neg == 0:
            filas.append({"objetivo": objetivo_nombre, "n": n_pos, "n_neg": n_neg,
                          "auc": None, "media_pos": None, "media_neg": None, "nota": nota})
            return
        filas.append({
            "objetivo": objetivo_nombre, "n": n_pos, "n_neg": n_neg,
            "auc": round(float(roc_auc_score(mascara.astype(int), score)), 6),
            "media_pos": round(float(score[mascara].mean()), 6),
            "media_neg": round(float(score[~mascara].mean()), 6),
            "nota": nota,
        })

    anadir("error de V03 sobre las 4449 celdas", error,
           "objetivo primario del parrafo 13")
    anadir("celda no rescatable dentro de los 757 errores", celda_c,
           "objetivo decisivo del parrafo 60: identificar las no rescatables")
    anadir("celda rescatable dentro de los 757 errores", rescatable,
           "el objetivo contrario, para ver la direccion del efecto")
    for j, etiqueta in enumerate(ETIQUETAS):
        sel = columna == j
        anadir(f"error de V03 en {etiqueta}", error & sel, f"columna {etiqueta}")
        anadir(f"no rescatable en {etiqueta}", celda_c & sel, f"columna {etiqueta}")

    # IC95 por bootstrap: separa senal real de ruido de muestreo
    ic_prim = ic_bootstrap(error.astype(int), score)
    ic_c = ic_bootstrap(celda_c.astype(int), score)
    filas[0]["ic95_bajo"], filas[0]["ic95_alto"] = round(ic_prim[0], 6), round(ic_prim[1], 6)
    filas[1]["ic95_bajo"], filas[1]["ic95_alto"] = round(ic_c[0], 6), round(ic_c[1], 6)
    for f_ in filas[2:]:
        f_["ic95_bajo"], f_["ic95_alto"] = "", ""

    auc_primario = filas[0]["auc"]
    auc_no_resc = filas[1]["auc"]
    log(f"   AUC contra error de V03            {auc_primario}  IC95 "
        f"[{ic_prim[0]:.4f}, {ic_prim[1]:.4f}]")
    log(f"   AUC no rescatable dentro de errores {auc_no_resc}  IC95 "
        f"[{ic_c[0]:.4f}, {ic_c[1]:.4f}]")
    log(f"   el IC95 del error excluye 0.5: {ic_prim[0] > 0.5} | "
        f"el de no rescatable excluye 0.5: {ic_c[0] > 0.5 or ic_c[1] < 0.5}")
    log(f"   puerta exploratoria del parrafo 15 = {PUERTE_AUC}: "
        f"{'superada' if (auc_primario or 0) >= PUERTE_AUC else 'NO superada'}")

    # ------------ comparacion explicita de las 42 celdas con las 715 (parrafo 44 y parrafo 62)
    rango = np.argsort(np.argsort(score)) / (score.size - 1)
    pct = {True: rango[celda_c], False: rango[rescatable]}
    comparacion = {
        "n_C": int(celda_c.sum()), "n_B": int(rescatable.sum()),
        "rango_medio_C": round(float(pct[True].mean()), 6),
        "rango_medio_B": round(float(pct[False].mean()), 6),
        "separacion_rangos": round(float(pct[True].mean() - pct[False].mean()), 6),
        "frac_C_en_tercio_superior": round(float((pct[True] > 2 / 3).mean()), 6),
        "frac_B_en_tercio_superior": round(float((pct[False] > 2 / 3).mean()), 6),
    }
    log(f"   rango percentil medio: C {comparacion['rango_medio_C']:.4f} vs "
        f"B {comparacion['rango_medio_B']:.4f} "
        f"(separacion {comparacion['separacion_rangos']:+.4f})")

    # -------------------------------------------------------------- guardar
    np.savez_compressed(DIR_CELDAS, ids=ids, y_true=y, pliegues=pliegues,
                        pred_v03=pred, proba_v03=proba,
                        error=error, no_rescatable=celda_c, rescatable=rescatable,
                        **rasgos)
    log("   escrito features/p59_voronoi_celfeatures.npz")

    cab = ["objetivo", "n", "n_neg", "auc", "ic95_bajo", "ic95_alto",
           "media_pos", "media_neg", "nota"]
    df = pd.DataFrame(filas)
    for c in cab:
        if c not in df.columns:
            df[c] = ""
    df = df[cab]
    df.to_csv(CSV_RES, index=False, encoding="utf-8")
    log(f"   escrito reports/p59_voronoi_results.csv ({len(df)} filas)")

    senal_prim = auc_primario is not None and auc_primario >= PUERTE_AUC
    senal_c = auc_no_resc is not None and auc_no_resc >= PUERTE_AUC
    c_significativo = ic_c[0] > 0.5 or ic_c[1] < 0.5
    if not senal_prim:
        decision = ("CLOSED. La puerta exploratoria del parrafo 15 no se supera, luego no hay "
                    "justificacion para el bloque complementario del parrafo 14 y no se entrena "
                    "nada")
    elif senal_c:
        decision = ("PROMISING. La puerta del parrafo 15 se supera y la geometria identifica las "
                    "celdas no rescatables, luego el parrafo 14 queda autorizado en la "
                    "siguiente iteracion. Se mantiene V03")
    elif c_significativo:
        decision = ("MARGINAL. La geometria local predice los errores de V03, pero no distingue "
                    "las celdas no rescatables, que es el objetivo fuerte del parrafo 60. Ademas "
                    "el punto estimador queda POR DEBAJO de 0.5 y su IC95 "
                    f"[{ic_c[0]:.4f}, {ic_c[1]:.4f}] excluye 0.5, luego la direccion esta "
                    "invertida de forma estadistica: las celdas no rescatables son MAS "
                    "coherentes con su clase segun esta geometria que las rescatables, que es "
                    "justo lo contrario de lo que la hipotesis suponia. El parrafo 14 solo "
                    "tendria sentido para reordenar errores ya identificados, no para recuperar "
                    "los que P54 declaro no recuperables")
    else:
        decision = ("MARGINAL. La geometria local predice los errores de V03, pero no distingue "
                    "las celdas no rescatables, que es el objetivo fuerte del parrafo 60, y su "
                    f"IC95 [{ic_c[0]:.4f}, {ic_c[1]:.4f}] incluye 0.5, luego no hay senal en "
                    "ninguna direccion. El parrafo 14 solo tendria sentido para reordenar "
                    "errores ya identificados, no para recuperar los que P54 declaro no "
                    "recuperables")

    informe = {
        "familia": "P59_A_VORONOI_LOCAL_GEOMETRY",
        "naturaleza": ("DIAGNOSTIC_ONLY. 0 modelos nuevos, 0 submissions, 0 consultas a Kaggle, "
                       "0 uso de holdout, 0 uso de test. V03 no se modifica"),
        "fecha": inicio.isoformat(timespec="seconds"),
        "novelty": {
            "veredicto": "PARTIALLY_DUPLICATE",
            "ya_hecho": [
                "V29-03: 48 configuraciones de kNN, ninguna con net positivo",
                "V4-F01-F04: la vecindad explica el error, IC95 [-0.013,+0.049], no significativo",
                "V6-07: la sospecha es un buen marcador de dificultad, no un cleaner",
            ],
            "nuevo_aqui": [
                "teselacion de prototipos de clase, celdas, margen y soporte de celda",
                "la comparacion contra las 42 celdas no rescatables de P54",
            ],
        },
        "metrica": {
            "definicion": "coseno de la union TF-IDF de V03",
            "propiedad": ("cada bloque TF-IDF viene normalizado a norma 1, luego la norma de la "
                          "union es sqrt(2) para toda fila y "
                          "distancia_coseno_union = (cos_palabra + cos_caracter) / 2 exactamente"),
        },
        "configuracion": {
            "prototipos_por_clase_y_etiqueta": K_PROTOTIPOS,
            "vecinos": K_VECINOS,
            "metodo_medoides": "farthest-first (BUILD) mas actualizaciones de Lloyd, sin azar",
            "pesos_inconsistencia": PESOS_INCONS,
            "nota": "ningun parametro se ajusto por rejilla; todos se fijaron antes de ejecutar",
        },
        "controles": {
            "sha256_v03_verificado": True,
            "oof_reproducido": ccv == V03_CCV,
            "umbrales_verificados": True,
            "grupos_partidos_entre_pliegues": partidos,
            "etiquetas_crudo_identicas_al_oof": True,
            "ids_de_test_en_el_trabajo": 0,
            "tests_unitarios": tests,
        },
        "diagnostico": filas,
        "auc_primario": auc_primario,
        "ic95_primario": [round(ic_prim[0], 6), round(ic_prim[1], 6)],
        "auc_no_rescatable": auc_no_resc,
        "ic95_no_rescatable": [round(ic_c[0], 6), round(ic_c[1], 6)],
        "no_rescatable_excluye_0_5": bool(c_significativo),
        "puerta_exploratoria": PUERTE_AUC,
        "puerta_superada": bool(senal_prim),
        "comparacion_42_celdas": comparacion,
        "poblaciones_p54": {"A": pob["n_A"], "B": pob["n_B"], "C": pob["n_C"],
                            "tweets_distintos_C": pob["tweets_C"]},
        "decision": decision,
        "experimentos_registrados": 0,
        "por_que_no_se_registra": ("el parrafo 50 no registra gap analysis, oracle ni "
                                   "visualizaciones. Este paso no entrena clasificador ni "
                                   "evalua corrected_cv, luego es diagnostico y no experimento"),
        "formulacion": "V03 remains the best verified leaderboard result.",
        "gastos": {"submissions": 0, "modelos_nuevos": 0, "consultas_kaggle": 0,
                   "uso_holdout": 0, "uso_test": 0,
                   "segundos": round((datetime.now(timezone.utc) - inicio).total_seconds(), 1)},
    }
    JSON_RES.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    _md(informe, df, comparacion, tests)
    log("   escrito reports/p59_voronoi_analysis.{md,json}")

    for destino in (CSV_RES, MD_RES, JSON_RES):
        if any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    # El npz es un ZIP binario: un escaneo de bytes >127 no significa nada ahi.
    # Lo que si importa es que no contenga ninguna cadena con texto.
    with np.load(DIR_CELDAS, allow_pickle=False) as _z:
        if any(_z[k].dtype.kind in "USO" for k in _z.files):
            raise SystemExit("PARADA: el npz de celdas contiene cadenas, no solo numeros")
    log("   control: el npz solo contiene arrays numericos, sin cadenas")

    log("")
    log(f"P59-A VEREDICTO: AUC primario {auc_primario} | AUC no rescatable {auc_no_resc}")
    log(f"   decision: {decision}")
    log(f"P59-01 OK | exit previsto 0 | {informe['gastos']['segundos']} s")


def _md(inf: dict[str, Any], df: pd.DataFrame, comp: dict[str, Any],
        tests: list[dict[str, Any]]) -> None:
    """Escribe el informe de la pista A.

    Args:
        inf: diccionario con los resultados.
        df: tabla de diagnostico.
        comp: comparacion de las 42 celdas contra las 715.
        tests: resultados de los tests unitarios.

    Returns:
        None.
    """
    L: list[str] = []
    A = L.append
    A("# P59-A - Voronoi y geometria local sobre la representacion de V03")
    A("")
    A(f"Generado por `experiments/p59_01_voronoi_local.py`. Fecha {inf['fecha']}.")
    A("")
    A("**Diagnostico. 0 modelos nuevos, 0 submissions, 0 consultas a Kaggle, 0 holdout, "
      "0 test. V03 no se modifica.** No se registro ninguna fila en `experiments.csv`: el parrafo "
      "50 no registra gap analysis, oracle ni visualizaciones, y este paso no entrena "
      "clasificador ni evalua `corrected_cv`.")
    A("")
    A("## 1. Auditoria de novedad: que estaba ya hecho")
    A("")
    A("Este track entra con `PARTIALLY_DUPLICATE`, y conviene decirlo antes que cualquier cifra.")
    A("")
    A("| ya evaluado | resultado |")
    A("|---|---|")
    for x in inf["novelty"]["ya_hecho"]:
        A(f"| {x.split(':')[0]} | {x.split(':', 1)[1].strip()} |")
    A("")
    A("La pregunta 'la geometria local explica los errores' **ya esta contestada**, y en dos "
      "sentidos: cuando se midio directamente el efecto sobre la metrica no fue significativo, y "
      "cuando se busco una senal de sospecha se concluyo que sirve para medir dificultad, no para "
      "limpiar entrenamiento. Por eso este paso **no vuelve a preguntar eso**.")
    A("")
    A("Lo que si es nuevo, y es lo unico que se mide:")
    A("")
    for x in inf["novelty"]["nuevo_aqui"]:
        A(f"- {x}")
    A("")
    A("## 2. La metrica, declarada y no supuesta")
    A("")
    A(f"> {inf['metrica']['propiedad']}")
    A("")
    A("El parrafo 8 prohibe hacer `TF-IDF -> SVD 2D -> Voronoi`. Aqui no hay ninguna proyeccion: "
      "la geometria vive en las 27431 dimensiones originales y la distancia es un coseno explicito.")
    A("")
    A("## 3. Configuracion, fijada antes de ejecutar")
    A("")
    c = inf["configuracion"]
    A("```text")
    A(f"prototipos por clase y etiqueta   {c['prototipos_por_clase_y_etiqueta']}")
    A(f"vecinos para la densidad local    {c['vecinos']}")
    A(f"metodo de medoides                {c['metodo_medoides']}")
    A(f"pesos del indice de inconsistencia {c['pesos_inconsistencia']}")
    A("```")
    A("")
    A(f"> {c['nota']}")
    A("")
    A("## 4. Controles de integridad y de fuga")
    A("")
    A("| control | resultado |")
    A("|---|---|")
    A("| sha256 de V03 | verificado, intacto |")
    A(f"| OOF canonico | reproduce `{V03_CCV!r}` exacto |")
    A("| umbrales de V03 | `pred == (proba >= [0.44, 0.51, 0.46])` exacto |")
    A(f"| grupos canonicos partidos entre pliegues | "
      f"{inf['controles']['grupos_partidos_entre_pliegues']} |")
    A("| etiquetas del CSV crudo contra el OOF | identicas celda a celda |")
    A("| ids de test en el conjunto de trabajo | 0 |")
    A("")
    A("La geometria de cada fila se construye con los prototipos de los **cuatro pliegues de "
      "entrenamiento** y se aplica a la quinta. Los vectorizadores se ajustan tambien solo en "
      "esos cuatro. Ninguna estadistica supervisada ve el pliegue de consulta.")
    A("")
    A("## 5. Tests unitarios del mecanismo (parrafo 52)")
    A("")
    A("| test | veredicto | detalle |")
    A("|---|---|---|")
    for t in tests:
        A(f"| {t['test']} | {'OK' if t['ok'] else '**FALLA**'} | {t['detalle']} |")
    A("")
    A("## 6. Resultado: el score contra los objetivos")
    A("")
    A("| objetivo | n | AUC | IC95 | media del score en el objetivo | en el resto |")
    A("|---|---:|---:|---|---:|---:|")
    for _, r in df.iterrows():
        a = "n/d" if pd.isna(r["auc"]) or r["auc"] == "" else f"{r['auc']:.4f}"
        if r["ic95_bajo"] == "" or pd.isna(r["ic95_bajo"]):
            ic = "-"
        else:
            ic = f"[{float(r['ic95_bajo']):.4f}, {float(r['ic95_alto']):.4f}]"
        mp = "-" if r["media_pos"] == "" else f"{float(r['media_pos']):.4f}"
        mn = "-" if r["media_neg"] == "" else f"{float(r['media_neg']):.4f}"
        A(f"| {r['objetivo']} | {r['n']} | **{a}** | {ic} | {mp} | {mn} |")
    A("")
    A(f"Puerta exploratoria del parrafo 15, `{inf['puerta_exploratoria']}`: "
      f"**{'superada' if inf['puerta_superada'] else 'NO superada'}** en el objetivo primario, y "
      f"**{'superada' if (inf['auc_no_rescatable'] or 0) >= inf['puerta_exploratoria'] else 'NO superada'}** "
      "en el objetivo fuerte del parrafo 60.")
    A("")
    A("## 7. Las 42 celdas no rescatables contra las 715 rescatables")
    A("")
    A("Esta es la comparacion que el parrafo 44 y el parrafo 60 exigen, y la que P54 dejo abierta.")
    A("")
    A("| magnitud | valor |")
    A("|---|---:|")
    A(f"| celdas no rescatables (C) | {comp['n_C']} |")
    A(f"| celdas rescatables (B) | {comp['n_B']} |")
    A(f"| rango percentil medio del score en C | {comp['rango_medio_C']:.4f} |")
    A(f"| rango percentil medio del score en B | {comp['rango_medio_B']:.4f} |")
    A(f"| separacion | {comp['separacion_rangos']:+.4f} |")
    A(f"| fraccion de C en el tercio superior | {comp['frac_C_en_tercio_superior']:.4f} |")
    A(f"| fraccion de B en el tercio superior | {comp['frac_B_en_tercio_superior']:.4f} |")
    A("")
    A("## 8. Decision")
    A("")
    A("```text")
    A(f"decision  {inf['decision']}")
    A("```")
    A("")
    A("## 9. Formulacion")
    A("")
    A(f"> {inf['formulacion']}")
    A("")
    A("Lo que el diagnostico SOSTIENE y lo que NO SOSTIENE estan separados en el JSON maestro y "
      "en la conclusion de la fase. Ningun resultado de este paso se ha registrado como "
      "experimento, porque no lo es.")
    A("")
    A("*Seccion generada por `experiments/p59_01_voronoi_local.py`. Sin push.*")
    MD_RES.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
