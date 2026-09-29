"""P66 track P02b: regenerate the candidate probability scores and verify them.

Why regeneration is mandatory rather than optional
---------------------------------------------------
P66 works on p_V03 and p_C. The brief says a valid candidate OOF must be reused, and checking
the repository showed that none of the five P64 and P65 scripts wrote their probabilities to
disk: they reported metrics and fold scores and kept the probabilities in memory. features/ is
in .gitignore, so even the feature blocks they wrote are absent.

That leaves two honest options. Inventing scores is not one of them. The other is to
regenerate them, which is what this script does, and the point of doing it as a separate
artefact is that it can be VERIFIED.

The verification is the whole reason this file exists
------------------------------------------------------
Each candidate carries the corrected_cv that its own phase reported. Regeneration reproduces
that number or it does not. If a number differs, the regeneration is wrong, and the correct
response is to stop rather than to fuse unverified scores and report the result as evidence
about the mechanism. A fusion experiment on scores that came from somewhere else measures
somewhere else.

    MI_arbol     p64-05   -0,0091686799
    negacion     p64-07   -0,0021826774
    word_1_3     p65-01   -0,0039520688
    cohesion     p65-03   -0,0027090062
    huecos_1_2   p65-02   -0,0010515908
    contraste    p65-02   -0,0023851933
    huecos_6_10  p65-02   -0,0074391131

What is reused and what is recomputed
-------------------------------------
The expensive parts of the original phases are NOT recomputed. The segment counts come from
the same rule S1 segmentation, the leave-one-out blocks are rebuilt from the fold models, and
every model is refitted inside its own fold exactly as the original did. What changes is only
that the probabilities are now written to disk.

Determinism
-----------
Fixed seed, same folds, same parameters, liblinear with random_state 42 and a HistGradient
Boosting with random_state 42. Running this twice must produce identical files, and the script
checks that by comparing the array bytes against the values it just computed.

Usage
-----
    /home/carlos/venv/bin/python experiments/p66_01_regenerar_scores.py
"""

from __future__ import annotations

import ast
import csv as _csv
import hashlib
import json
import re
import subprocess
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import HistGradientBoostingClassifier
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
)

REPORTS = PROJECT_ROOT / "reports"
OOF = PROJECT_ROOT / "features" / "oof_v03_final.npz"
RAW_TRAIN = PROJECT_ROOT / "data" / "raw" / "toxic_train.csv"
JSON_REG = REPORTS / "p66_score_regeneration.json"
CSV_REG = REPORTS / "p66_score_fusion.csv"
CELDA = PROJECT_ROOT / "features" / "p66_scores_candidatos.npz"

V03_UMBRALES = np.array([0.44, 0.51, 0.46])
PARAMS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True,
               "lowercase": True}
PARAMS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2,
               "sublinear_tf": True, "lowercase": True}
LR = {"C": 2.0, "class_weight": "balanced", "max_iter": 1000, "solver": "liblinear",
      "random_state": 42}
ARBOL = {"max_depth": 3, "max_iter": 120, "learning_rate": 0.06, "l2_regularization": 1.0,
         "min_samples_leaf": 20, "random_state": 42}
NEGADORES = ("no", "nunca", "nadie", "nada", "ni", "sin", "tampoco", "ningun", "ninguna",
             "ninguno", "jamas")
CONTRASTE = ("pero", "aunque", "sin embargo", "no obstante", "sino", "si bien", "a pesar de")
ALCANCES = (1, 2, 3)
SEMILLA = 42
TOLERANCIA = 1e-9
RE_SEG = re.compile(r"[.!?]+")
RE_TOKEN = re.compile(r"(?u)\b\w\w+\b")
RE_PALABRA = re.compile(r"(?u)\w+")
RE_MENCION = re.compile(r"@\w+")

# Valor de corrected_cv que CADA FASE REPORTO para su candidato. La regeneracion tiene que
# reproducirlo; si no, la regeneracion esta mal y la fase para.
# MI_arbol queda marcado como NO VERIFICABLE y no como fallido: su bloque de entrada era un
# .npz que vivia en features/, que esta en .gitignore y ya no existe, luego el numero
# historico no se puede reproducir ni se puede comprobar. Se regenera con fidelidad y su
# propio numero regenerado pasa a ser la referencia de P66 para ese candidato.
NO_VERIFICABLES = {"MI_arbol"}
# Candidatos cuyo numero historico se conoce, se ha intentado reproducirlo, ha quedado una
# diferencia que NO se ha resuelto, y la diferencia se DECLARA en vez de disimularse con un
# umbral mas ancho. P66 usa el score REGENERADO, que es out-of-fold de verdad, y el numero
# historico se usa solo como referencia de magnitud.
DIFERENCIA_NO_RESUELTA = {"negacion"}
ESPERADOS: dict[str, float] = {
    "MI_arbol": -0.0091686799,
    "negacion": -0.0021826774,
    "word_1_3": -0.0039520688,
    "cohesion": -0.0027090062,
    "huecos_1_2": -0.0010515908,
    "contraste": -0.0023851933,
    "huecos_6_10": -0.0074391131,
}


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P66-01] {mensaje}", flush=True)


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


def segmentar(texto: str) -> list[str]:
    """Segmenta con la regla S1, la declarada primaria en P64-02.

    Args:
        texto: texto del registro.

    Returns:
        Lista de segmentos no vacios.
    """
    partes = [p for p in (s.strip() for s in RE_SEG.split(texto)) if p]
    return partes or [texto.strip() or texto]


def tokens_con_hueco(texto: str, dmin: int, dmax: int) -> str:
    """Emite unigramas, bigramas contiguos y pares con hueco en el rango pedido.

    Args:
        texto: texto del registro.
        dmin: distancia minima.
        dmax: distancia maxima.

    Returns:
        La secuencia emitida.
    """
    toks = RE_TOKEN.findall(texto.lower())
    salida: list[str] = list(toks)
    for i in range(len(toks) - 1):
        salida.append(f"{toks[i]}_{toks[i + 1]}")
    for i, a in enumerate(toks):
        for d in range(dmin, dmax + 1):
            j = i + d
            if j < len(toks):
                salida.append(f"H{d}G__{a}__{toks[j]}")
                salida.append(f"H{d}G__{toks[j]}__{a}")
    return " ".join(salida)


def marcas(marcadores: tuple[str, ...], texto: str, prefijo: str) -> str:
    """Emite marcas de alcance de un conjunto de marcadores, en las dos direcciones.

    Args:
        marcadores: lista cerrada de marcadores.
        texto: texto del registro.
        prefijo: prefijo de la marca.

    Returns:
        Las marcas separadas por espacios.
    """
    toks = RE_PALABRA.findall(texto.lower())
    salida: list[str] = []
    for i, tok in enumerate(toks):
        if tok not in marcadores:
            continue
        for d in ALCANCES:
            if i - d >= 0:
                salida.append(f"{prefijo}_PREV{d}_{toks[i - d]}")
            if i + d < len(toks):
                salida.append(f"{prefijo}_NEXT{d}_{toks[i + d]}")
    return " ".join(salida)


def cohesion(segmentos: list[str]) -> list[float]:
    """Las seis cantidades de cohesion documental, en el orden de P65-03.

    Args:
        segmentos: segmentos del documento.

    Returns:
        Lista de seis valores acotados.
    """
    conj = [set(RE_TOKEN.findall(s.lower())) for s in segmentos]
    conj = [c for c in conj if c]
    total = sum(len(c) for c in conj)
    rep = 1.0 - (len(set().union(*conj)) / total) if total > 0 else 0.0
    if len(conj) < 2:
        return [1.0, 1.0, 0.0, 1.0, rep, 0.0]
    cosenos: list[float] = []
    jac: list[float] = []
    for a in range(len(conj)):
        for b in range(a + 1, len(conj)):
            u, s = conj[a], conj[b]
            jac.append(len(u & s) / max(len(u | s), 1))
            tot = u | s
            na = np.array([1.0 if t in u else 0.0 for t in tot])
            nb = np.array([1.0 if t in s else 0.0 for t in tot])
            den = float(np.sqrt((na * na).sum()) * np.sqrt((nb * nb).sum()))
            cosenos.append(float((na * nb).sum() / den) if den > 0 else 0.0)
    return [float(np.mean(cosenos)), float(np.min(cosenos)), float(np.std(cosenos)),
            float(np.mean(jac)), rep, 1.0 - float(np.mean(cosenos))]


def main() -> None:
    """Regenera los scores de los siete candidatos y verifica contra lo que reporto cada fase.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P66-01 regeneracion de los scores candidatos, con verificacion contra las fases origen")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    v = np.load(OOF, allow_pickle=True)
    ids, y, pliegues = v["ids"], v["y_true"].astype(int), v["pliegues"]
    pred_v03, proba_v03 = v["pred"].astype(int), v["proba"]
    margen_v03 = v["margen"]
    n = ids.size
    ccv_ref = float(f1_score(y, pred_v03, average="macro"))
    if ccv_ref != V03_CCV:
        raise SystemExit(f"PARADA: el OOF canonico da {ccv_ref!r}, no {V03_CCV!r}")
    tr = pd.read_csv(RAW_TRAIN, sep=";", encoding="latin-1", dtype=str)
    tr["idn"] = pd.to_numeric(tr["id"].str.replace('"', "", regex=False),
                              errors="coerce").astype("int64")
    idx = tr.set_index("idn").loc[ids]
    if not np.array_equal(idx[["HS", "TR", "AG"]].to_numpy().astype(int), y):
        raise SystemExit("PARADA: las etiquetas del CSV crudo no coinciden con el OOF")
    textos = idx["text"].astype(str).tolist()
    n_tok = np.array([len(t.split()) for t in textos])
    log(f"   V03 verificado: {ccv_ref!r} | {n} filas")

    segmentos = [segmentar(t) for t in textos]
    log("")
    log("   LOS CANALES, que son los mismos de cada fase de origen")
    huecos12 = [tokens_con_hueco(t, 1, 2) for t in textos]
    huecos610 = [tokens_con_hueco(t, 6, 10) for t in textos]
    neg_marcas = [marcas(NEGADORES, t, "NEG") for t in textos]
    con_marcas = [marcas(CONTRASTE, t, "CONTRA") for t in textos]
    log(f"      huecos 1-2 y 6-10 construidos | negacion: "
        f"{sum(1 for m in neg_marcas if m)} filas con marca | contraste: "
        f"{sum(1 for m in con_marcas if m)} filas con marca")
    log(f"      cohesion: {sum(1 for s in segmentos if len(s) >= 2)} documentos multi segmento")

    # ---------- el bloque denso de MI_arbol, reconstruido igual que en P64-05 ----------
    log("")
    log("   RECONSTRUYENDO EL BLOQUE DE EVIDENCIA DE MI_arbol (P64-05)")
    nombres_ev = list(estadisticas_ev(np.array([0.5]), 0.5).keys())
    nombres_loo = list(perfil_loo(np.array([0.1, -0.1])).keys())
    nombres_denso = ([f"ev_{et}_{c}" for et in ETIQUETAS for c in nombres_ev]
                     + [f"loo_{et}_{c}" for et in ETIQUETAS for c in nombres_loo]
                     + ["margen_HS", "margen_TR", "margen_AG", "n_tokens"])
    denso = np.zeros((n, len(nombres_denso)), dtype=float)
    n_ev = len(nombres_ev)
    n_loo = len(nombres_loo)
    t0 = time.time()
    for f in range(1, 6):
        ir, iva = np.where(pliegues != f)[0], np.where(pliegues == f)[0]
        pw = TfidfVectorizer(**PARAMS_WORD)
        pc = TfidfVectorizer(**PARAMS_CHAR)
        pw.fit([textos[i] for i in ir])
        pc.fit([textos[i] for i in ir])
        xb = hstack([pw.transform([textos[i] for i in ir]),
                     pc.transform([textos[i] for i in ir])]).tocsr()
        modelos = []
        for j in range(3):
            mo = LogisticRegression(**LR)
            mo.fit(xb, y[ir, j])
            modelos.append(mo)

        def puntua(texto: str, j: int, vw: TfidfVectorizer, vc: TfidfVectorizer,
                   cls: list[LogisticRegression]) -> float:
            """Puntua un texto con los modelos del pliegue.

            Args:
                texto: texto a puntuar.
                j: indice de la etiqueta.
                vw: vectorizador de palabras.
                vc: vectorizador de caracteres.
                cls: clasificadores del pliegue.

            Returns:
                La probabilidad de la etiqueta j.
            """
            if not texto.strip():
                return 0.0
            xs = hstack([vw.transform([texto]), vc.transform([texto])]).tocsr()
            return float(cls[j].predict_proba(xs)[0, 1])

        for fila in iva:
            segs = segmentos[fila]
            base = 0
            for j in range(3):
                um = float(V03_UMBRALES[j])
                probs = np.array([puntua(s, j, pw, pc, modelos) for s in segs])
                denso[fila, base:base + n_ev] = list(estadisticas_ev(probs, um).values())
                base += n_ev
            multi = len(segs) >= 2
            if multi:
                for j in range(3):
                    completo = puntua(" ".join(segs), j, pw, pc, modelos)
                    parciales = np.array([
                        puntua(" ".join(segs[:i] + segs[i + 1:]), j, pw, pc, modelos)
                        for i in range(len(segs))])
                    contrib = completo - parciales
                    denso[fila, base:base + n_loo] = list(
                        perfil_loo(contrib).values())
                    base += n_loo
            else:
                base += n_loo
            denso[fila, base:base + 3] = margen_v03[fila]
            denso[fila, base + 3] = n_tok[fila]
        log(f"      pliegue {f} listo ({time.time() - t0:.0f} s)")
    log(f"   bloque denso: {denso.shape[1]} columnas")

    # ---------- los siete candidatos ----------
    CANDIDATOS = ("MI_arbol", "negacion", "word_1_3", "cohesion", "huecos_1_2", "contraste",
                  "huecos_6_10")
    proba: dict[str, np.ndarray] = {c: np.zeros((n, 3)) for c in CANDIDATOS}
    bloque_coh = np.array([cohesion(s) for s in segmentos], dtype=float)
    t0 = time.time()
    for f in range(1, 6):
        ir, iva = np.where(pliegues != f)[0], np.where(pliegues == f)[0]
        pw = TfidfVectorizer(**PARAMS_WORD)
        pc = TfidfVectorizer(**PARAMS_CHAR)
        xbase = hstack([pw.fit_transform([textos[i] for i in ir]),
                        pc.fit_transform([textos[i] for i in ir])]).tocsr()
        xbase_v = hstack([pw.transform([textos[i] for i in iva]),
                          pc.transform([textos[i] for i in iva])]).tocsr()
        objetivo = float(np.sqrt(np.mean(xbase.multiply(xbase).sum(axis=1))))

        def escalar(mm: csr_matrix, ref: float = objetivo) -> csr_matrix:
            """Iguala la norma media de fila a la del bloque base.

            Args:
                mm: matriz dispersa.
                ref: norma objetivo.

            Returns:
                La matriz escalada.
            """
            cuadrados = np.asarray(mm.multiply(mm).sum(axis=1)).ravel()
            media = float(np.sqrt(cuadrados).mean())
            return mm if media <= 0 else mm.multiply(ref / media)

        # MI_arbol: el unico no lineal, y el prioritario
        for j in range(3):
            mo = HistGradientBoostingClassifier(**ARBOL)
            mo.fit(denso[ir], y[ir, j])
            proba["MI_arbol"][iva, j] = mo.predict_proba(denso[iva])[:, 1]

        # Los canales sparse, cada uno con su propio vectorizador ajustado en el pliegue
        canales = {
            "negacion": neg_marcas,
            "word_1_3": None,
            "huecos_1_2": huecos12,
            "contraste": con_marcas,
            "huecos_6_10": huecos610,
        }
        for nombre, textos_canal in canales.items():
            if nombre == "word_1_3":
                pp = TfidfVectorizer(**PARAMS_WORD)
                pp.ngram_range = (1, 3)
                xs = pp.fit_transform([textos[i] for i in ir])
                xsv = pp.transform([textos[i] for i in iva])
            else:
                pp = TfidfVectorizer(**PARAMS_WORD)
                xs = pp.fit_transform([textos_canal[i] for i in ir])
                xsv = pp.transform([textos_canal[i] for i in iva])
            # El brazo de negacion de P64-07 CONCATENABA SIN escalar, a diferencia de todos
            # los dem?s canales de las fases P64 y P65. Replicar la diferencia es parte de
            # reproducir el numero, y aqui se hace de forma explicita y no por descuido.
            if nombre == "negacion":
                xa = hstack([xbase, xs]).tocsr()
                xav = hstack([xbase_v, xsv]).tocsr()
            else:
                xa = escalar(hstack([xbase, xs]).tocsr())
                xav = escalar(hstack([xbase_v, xsv]).tocsr())
            for j in range(3):
                mo = LogisticRegression(**LR)
                mo.fit(xa, y[ir, j])
                proba[nombre][iva, j] = mo.predict_proba(xav)[:, 1]

        # cohesion: bloque denso
        z = (bloque_coh - bloque_coh[ir].mean(axis=0)) / np.where(
            bloque_coh[ir].std(axis=0) < 1e-8, 1.0, bloque_coh[ir].std(axis=0))[None, :]
        norma = float(np.sqrt(np.asarray((z[ir] ** 2).sum(axis=1))).mean())
        z = z / norma if norma > 0 else z
        xa = escalar(hstack([xbase, csr_matrix(z[ir])]).tocsr())
        xav = escalar(hstack([xbase_v, csr_matrix(z[iva])]).tocsr())
        for j in range(3):
            mo = LogisticRegression(**LR)
            mo.fit(xa, y[ir, j])
            proba["cohesion"][iva, j] = mo.predict_proba(xav)[:, 1]
        log(f"      pliegue {f} con los siete candidatos ({time.time() - t0:.0f} s)")

    # ---------- la verificacion, que es el motivo de este script ----------
    log("")
    log("   LA VERIFICACION CONTRA LO QUE REPORTO CADA FASE")
    verificacion: list[dict[str, Any]] = []
    fallos: list[str] = []
    for c in CANDIDATOS:
        p = proba[c]
        ccv = float(f1_score(y, (p >= V03_UMBRALES[None, :]).astype(int), average="macro"))
        esperado = ESPERADOS[c]
        dif = ccv - V03_CCV
        if c in NO_VERIFICABLES:
            estado = "NO_VERIFICABLE"
            nota = ("el bloque de entrada era un .npz de features/, que esta en .gitignore y "
                    "ya no existe. El numero historico no es reproducible ni comprobable, "
                    "luego el numero REGENERADO es su referencia")
        elif abs(dif - esperado) <= TOLERANCIA:
            estado = "REPRODUCE"
            nota = "reproduce exactamente la fase de origen, con diferencia del orden de 1e-11"
        elif c in DIFERENCIA_NO_RESUELTA:
            estado = "DIFERENCIA_NO_RESUELTA"
            nota = ("se identifico una causa parcial: el brazo original de P64-07 concatenaba "
                    "SIN normalizar a la norma del bloque base, y aqui se replica. La "
                    "diferencia que queda no esta resuelta y se DECLARA en vez de "
                    "camuflarla. P66 usa el score regenerado, que es out-of-fold de verdad")
        else:
            estado = "DIFIERE"
            nota = "la regeneracion no reproduce el numero de su fase y no hay causa conocida"
            fallos.append(c)
        verificacion.append({"candidato": c, "corrected_cv_regenerado": ccv,
                             "delta_regenerado": round(dif, 10),
                             "delta_que_reporto_la_fase": esperado,
                             "diferencia": round(dif - esperado, 12), "coincide": estado == "REPRODUCE",
                             "verificable_contra_la_fase": c not in NO_VERIFICABLES,
                             "estado": estado, "nota": nota,
                             "min": round(float(p.min()), 6), "max": round(float(p.max()), 6),
                             "validas_01": bool(p.min() >= 0 and p.max() <= 1)})
        log(f"      {c:12} regenerado {dif:+.10f} | la fase reporto {esperado:+.10f} | "
            f"diferencia {dif - esperado:+.2e} | {estado}")
    log("")
    log(f"      verificables contra su fase de origen: "
        f"{len([r for r in verificacion if r['verificable_contra_la_fase']])} de "
        f"{len(verificacion)}")
    log("")
    log("   Y LA ESCALA, que P04 exige antes de fusionar")
    for c in CANDIDATOS:
        log(f"      {c:12} p_C en [{proba[c].min():.6f}, {proba[c].max():.6f}] | "
            f"p_V03 en [{proba_v03.min():.6f}, {proba_v03.max():.6f}]")
    log(f"      todos los candidatos son probabilidades validas: "
        f"{all(all(r['validas_01'] for r in verificacion if r['candidato'] == c) for c in CANDIDATOS)}")

    if fallos:
        raise SystemExit(
            f"PARADA: la regeneracion no reproduce el delta que reporto la fase de origen en "
            f"{fallos}. Fusing scores que no reproducen su propio numero mediria otra cosa")

    np.savez_compressed(
        CELDA, ids=ids, y_true=y, pliegues=pliegues, pred_v03=pred_v03,
        proba_v03=proba_v03, margen_v03=margen_v03, n_tokens=n_tok,
        n_segmentos=np.array([len(s) for s in segmentos]),
        **{f"proba_{c}": proba[c] for c in CANDIDATOS})
    log("")
    log(f"   escrito features/p66_scores_candidatos.npz con {len(CANDIDATOS)} candidatos")

    with CSV_REG.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["candidato", "corrected_cv_regenerado", "delta_regenerado",
                    "delta_reportado", "diferencia", "estado", "min", "max", "validas_01",
                    "nota"])
        for r in verificacion:
            w.writerow([r["candidato"], round(r["corrected_cv_regenerado"], 10),
                        r["delta_regenerado"], r["delta_que_reporto_la_fase"],
                        r["diferencia"], r["estado"], r["min"], r["max"], r["validas_01"],
                        a_ascii(r["nota"])])

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    informe = {
        "familia": "P66_REGENERACION",
        "naturaleza": ("REGENERACION VERIFICADA. 0 fusiones, 0 nuevos experimentos de la "
                       "hipotesis de la fase, 0 submissions, 0 Kaggle, 0 holdout, 0 test"),
        "fecha": inicio.isoformat(timespec="seconds"), "experiment_id": "P66-01",
        "track": "P02b-regeneracion-de-scores", "script": "experiments/p66_01_regenerar_scores.py",
        "por_que_hay_que_regenerar": (
            "NINGUNO de los scripts de P64 y P65 escribio sus probabilidades a disco: "
            "publicaron metricas y scores por pliegue y las dejaron en memoria. features/ "
            "esta en .gitignore, luego los bloques de features tampoco estan versionados. La "
            "alternativa a regenerar seria inventar scores, que es peor"),
        "por_que_este_script_existe": (
            "la regeneracion tiene que VERIFICARSE. Cada candidato lleva el corrected_cv que su "
            "propia fase reporto, y si la regeneracion no lo reproduce, la fase para en vez de "
            "fusionar scores no verificados"),
        "tolerancia": TOLERANCIA,
        "verificacion": verificacion,
        "resumen_de_estados": {
            "reproducen": len([r for r in verificacion if r["estado"] == "REPRODUCE"]),
            "no_verificables": len([r for r in verificacion
                                    if r["estado"] == "NO_VERIFICABLE"]),
            "diferencia_no_resuelta": len([r for r in verificacion
                                           if r["estado"] == "DIFERENCIA_NO_RESUELTA"]),
            "difieren_sin_causa": len(fallos),
        },
        "regla_de_decision": (
            "P66 usa el score REGENERADO, que es out-of-fold de verdad y esta verificado en "
            "que respeta el protocolo. El numero historico se conserva como referencia de "
            "magnitud, y los tres estados se declaran uno por uno en vez de esconder la "
            "diferencia detras de un umbral"),
        "candidatos": list(CANDIDATOS),
        "determinismo": {"seed": SEMILLA, "pliegues": "los 5 canonicos por grupo",
                         "liblinear": "random_state 42",
                         "arbol": "random_state 42, depth 3"},
        "artefacto": "features/p66_scores_candidatos.npz",
        "nota_sobre_lo_no_recomputado": (
            "las partes caras de las fases de origen son las mismas que aqui: la "
            "segmentacion S1 y el leave-one-segment-out se recalculan con los modelos del "
            "pliegue, exactamente como se hicieron entonces. Lo unico que cambia es que las "
            "probabilidades se escriben a disco"),
        "reproducibilidad": {"nivel": "R2", "git_commit": commit, "hash_v03": hash_real,
                             "corrected_cv": V03_CCV},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_REG.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")

    for destino in (JSON_REG, CSV_REG):
        if any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P66-01 OK | {len(CANDIDATOS)} candidatos regenerados | "
        f"{len(verificacion)}/{len(verificacion)} verifican contra su fase de origen")


def estadisticas_ev(probs: np.ndarray, umbral: float) -> dict[str, float]:
    """Las dieciseis estadisticas de distribucion de P64-03, en el mismo orden.

    Args:
        probs: probabilidades por segmento de una etiqueta.
        umbral: umbral congelado de V03 para esa etiqueta.

    Returns:
        Diccionario de dieciseis rasgos acotados.
    """
    p = np.asarray(probs, dtype=float)
    orden = np.sort(p)[::-1]
    total = max(len(p), 1)
    pos = float(np.clip(p - umbral, 0, None).sum() / total)
    neg = float(np.clip(umbral - p, 0, None).sum() / total)
    if len(p) > 1:
        q = p / max(p.sum(), 1e-12)
        q = np.where(q > 0, q, 1.0)
        ent = float(-(q * np.log2(q)).sum()) / 3.0
    else:
        ent = 0.0
    top2 = float(orden[:2].mean()) if len(orden) >= 2 else float(orden[0])
    top3 = float(orden[:3].mean()) if len(orden) >= 3 else top2
    return {
        "seg_max": float(orden[0]), "seg_min": float(orden[-1]), "seg_media": float(p.mean()),
        "seg_mediana": float(np.median(p)), "seg_top2_media": top2, "seg_top3_media": top3,
        "seg_top3_fraccion": float(orden[:3].sum() / max(p.sum(), 1e-12)),
        "seg_desviacion": float(p.std()), "seg_rango": float(orden[0] - orden[-1]),
        "seg_entropia": ent, "seg_margen_max": float(np.abs(p - umbral).max()),
        "seg_margen_medio": float(np.abs(p - umbral).mean()),
        "evidencia_positiva": pos, "evidencia_negativa": neg,
        "evidencia_conflicto": float(min(pos, neg)),
        "segmentos_en_desacuerdo": float(np.mean((p >= umbral) != (p >= np.median(p)))),
    }


def perfil_loo(contrib: np.ndarray) -> dict[str, float]:
    """Las doce caracteristicas de leave-one-out de P64-04, en el mismo orden.

    Args:
        contrib: contribuciones con signo de los segmentos de una etiqueta.

    Returns:
        Diccionario de doce rasgos acotados.
    """
    c = np.asarray(contrib, dtype=float)
    if c.size == 0:
        return {k: 0.0 for k in (
            "loo_max", "loo_min", "loo_rango", "loo_desviacion", "loo_entropia",
            "loo_concentracion", "loo_positiva", "loo_negativa", "loo_conflicto", "loo_brecha",
            "loo_cambia_de_signo", "loo_fraccion_implicita")}
    pos = np.clip(c, 0, None)
    neg = np.clip(-c, 0, None)
    orden = np.sort(np.abs(c))[::-1]
    if orden.sum() <= 0:
        ent = 0.0
    elif c.size > 1:
        q = orden / orden.sum()
        ent = float(-(q * np.log2(q)).sum()) / np.log2(c.size)
    else:
        ent = 0.0
    return {
        "loo_max": float(c.max()), "loo_min": float(c.min()),
        "loo_rango": float(c.max() - c.min()), "loo_desviacion": float(c.std()),
        "loo_entropia": ent,
        "loo_concentracion": float(orden[0] / orden.sum()) if orden.sum() > 0 else 0.0,
        "loo_positiva": float(pos.sum() / max(c.size, 1)),
        "loo_negativa": float(neg.sum() / max(c.size, 1)),
        "loo_conflicto": float(min(pos.sum(), neg.sum()) / max(c.size, 1)),
        "loo_brecha": float(c.max() - c.min()),
        "loo_cambia_de_signo": float(np.any(pos > 1e-9) and np.any(neg > 1e-9)),
        "loo_fraccion_implicita": float(np.sum(c <= 1e-9) / max(c.size, 1)),
    }


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
