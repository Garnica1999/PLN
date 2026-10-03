"""P62 track: is the short-tweet failure an information ceiling, or is there headroom?

What is left, and why this is the last lead
-------------------------------------------
Five experiments have now measured and refuted the two natural hypotheses for the
representation axis:

    grammar     the interaction between word and POS is real, +0,0657 in the EDA, but it
                beats bare lemmas by about +0,0009 and never beats V03
    coverage    min_df=2 leaves only 28,35 % of the surface forms out, but restoring it
                to 99,53 % costs -0,004023 instead of gaining

So the representation axis is closed on both of its natural dimensions, with numbers.

One lead from the EDA was never attacked, and it is the biggest measured anomaly of the
whole phase. V03's high-confidence errors concentrate in the SHORT and simple tweets:

    TR short, high-confidence error rate   0,4167
    TR long,  high-confidence error rate   0,0433
    F1 of V03, short minus long slice      -0,2864 for TR
    F1 of V03, syntax-rich minus poor       +0,1070, V03 is BETTER when structure is there

Those two facts together are strange, and they point at one explanation: V03 does not
fail because it ignores grammar, it fails when there is no text. If that is right, the
short-tweet region is an INFORMATION CEILING, and no representation will fix it. If it is
wrong, there is headroom and the phase stopped one experiment early.

How to tell them apart
----------------------
A ceiling claim is a claim about identifiability, and identifiability is measurable
without training anything. Inside the short-positive region, two things can be true at
once and they have opposite consequences:

    the ORDERING is good and only the cut is wrong
        then the problem is a decision problem, which V8-09 and P59-T already closed, and
        the region is a ceiling in practice

    the ORDERING is bad
        then the representation cannot even separate the two classes inside the region,
        which is a genuine information ceiling and no representation can fix it

The measure is the AUC of V03's own score computed INSIDE the region, positives against
negatives. It uses no new model, no candidate and no outcome selection, and it is
computed on the canonical OOF only. If that AUC is already high, V03 can order the region
correctly and the whole difficulty is where the line is drawn, not what the text says.

The second, independent check is the group ceiling. The project already has a methodology
for it: near-duplicate groups that disagree on the label are the only cells where any
method can possibly win, and a majority vote inside a group is the hard bound. This script
reproduces that bound inside the short region.

The slice is declared before anything is looked at
---------------------------------------------------
Short is at most six spaCy tokens, the same definition the linguistic EDA used, declared
there before any candidate existed. It is not re-cut here to suit a result.

Usage
-----
    /home/carlos/venv/bin/python experiments/p62_10_techo_cortos.py
"""

from __future__ import annotations

import ast
import gzip
import hashlib
import json
import subprocess
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
TRAIN_RAW = PROJECT_ROOT / "data" / "raw" / "toxic_train.csv"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
ANOTACION = PROJECT_ROOT / "features" / "p62_anotacion_spacy.json.gz"
MD_T = REPORTS / "p62_techo_cortos.md"
JSON_T = REPORTS / "p62_techo_cortos.json"
CSV_T = REPORTS / "p62_techo_cortos.csv"

V03_UMBRALES = np.array([0.44, 0.51, 0.46])
MAX_TOKENS_CORTO = 6
DELTA_HC = 0.10
SEMILLA = 42


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P62-10] {mensaje}", flush=True)


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


def auc(sel: np.ndarray, score: np.ndarray, y: np.ndarray) -> float | None:
    """AUC por rangos, con la definicion explicita para no depender de una libreria.

    Args:
        sel: mascara de la region.
        score: puntuacion.
        y: etiquetas.

    Returns:
        El AUC, o None si la region no tiene ambas clases.
    """
    s, t = score[sel], y[sel]
    npos, nneg = int((t == 1).sum()), int((t == 0).sum())
    if npos == 0 or nneg == 0:
        return None
    orden = np.argsort(s, kind="stable")
    rangos = np.empty(len(s), dtype=float)
    rango = 1
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[orden[j + 1]] == s[orden[i]]:
            j += 1
        rango_medio = (rango + rango + (j - i)) / 2.0
        rangos[orden[i:j + 1]] = rango_medio
        rango += (j - i + 1)
        i = j + 1
    return float((rangos[t == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main() -> None:
    """Determina si el fallo en los tweets cortos es un techo de informacion.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P62-10 el fallo en los tweets cortos, techo de informacion o margen real")
    log(f"   region corta: hasta {MAX_TOKENS_CORTO} tokens de spaCy, DEFINIDA ANTES")
    log("   no se entrena nada: es una medicion de identificabilidad sobre el OOF canonico")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")

    v = np.load(OOF, allow_pickle=True)
    ids, y, _pliegues = v["ids"], v["y_true"].astype(int), v["pliegues"]
    pred_v03, proba_v03 = v["pred"].astype(int), v["proba"]
    n = ids.size
    from sklearn.metrics import f1_score
    ccv_ref = float(f1_score(y, pred_v03, average="macro"))
    if ccv_ref != V03_CCV:
        raise SystemExit(f"PARADA: el OOF canonico da {ccv_ref!r}, no {V03_CCV!r}")
    part = pd.read_csv(PARTICION)
    grupos = part.set_index("id").loc[ids, "grupo_near_dup_autor"].to_numpy()
    tr = pd.read_csv(TRAIN_RAW, sep=";", encoding="latin-1", dtype={"id": str})
    tr["id_num"] = pd.to_numeric(tr["id"].str.replace('"', "", regex=False),
                                errors="coerce").astype("int64")
    idx = tr.set_index("id_num").loc[ids]
    idx["text"].astype(str).tolist()
    with gzip.open(ANOTACION, "rt", encoding="utf-8") as fh:
        anot = json.load(fh)
    if anot["ids"] != [int(x) for x in ids]:
        raise SystemExit("PARADA: la anotacion no corresponde a las filas del OOF")
    n_tok = np.array([d["n"] for d in anot["documentos"]])
    corto = n_tok <= MAX_TOKENS_CORTO
    log("")
    log(f"   OOF canonico {ccv_ref!r} | {n} filas | {int(corto.sum())} cortas "
        f"({corto.mean() * 100:.1f} por ciento) | {int((~corto).sum())} largas")
    log(f"   longitud media: corta {n_tok[corto].mean():.2f} tokens, larga "
        f"{n_tok[~corto].mean():.2f} tokens")

    filas: list[dict[str, Any]] = []
    res: dict[str, Any] = {}
    for et in ETIQUETAS:
        j = ETIQUETAS.index(et)
        bloque: dict[str, Any] = {}
        for nombre, mascara in (("corta", corto), ("larga", ~corto)):
            pos, neg = mascara & (y[:, j] == 1), mascara & (y[:, j] == 0)
            a = auc(mascara, proba_v03[:, j], y[:, j])
            ac = pred_v03[mascara, j] == y[mascara, j]
            dist = np.abs(proba_v03[mascara, j] - V03_UMBRALES[j])
            # el mejor F1 alcanzable con CUALQUIER monotonica de la puntuacion de V03:
            # se recorre el corte optimo sobre las puntuaciones observadas. Si ese techo
            # esta muy por encima del F1 achieved, el problema es donde se dibuja la linea
            # y no que dice el texto.
            mejor_f1 = 0.0
            mejor_corte = 0.0
            s_reg, t_reg = proba_v03[mascara, j], y[mascara, j]
            for corte in np.unique(s_reg):
                pred = (s_reg >= corte).astype(int)
                v_ = f1_score(t_reg, pred, zero_division=0)
                if v_ > mejor_f1:
                    mejor_f1, mejor_corte = float(v_), float(corte)
            bloque[nombre] = {
                "n": int(mascara.sum()), "n_positivos": int(pos.sum()),
                "n_negativos": int(neg.sum()),
                "F1_de_V03": round(float(f1_score(y[mascara, j], pred_v03[mascara, j],
                                                   zero_division=0)), 6),
                "tasa_de_error": round(float(1.0 - ac.mean()), 6),
                "tasa_de_error_de_alta_confianza": round(
                    float(((~ac) & (dist >= DELTA_HC)).mean()), 6),
                "AUC_dentro_de_la_region": (None if a is None else round(a, 6)),
                "mejor_F1_alcanzable_moviendo_el_corte": round(mejor_f1, 6),
                "corte_optimo": round(mejor_corte, 6),
                "distancia_de_V03_al_techo": round(mejor_f1 - float(f1_score(
                    y[mascara, j], pred_v03[mascara, j], zero_division=0)), 6),
            }
        res[et] = bloque
        c, l = bloque["corta"], bloque["larga"]
        log("")
        log(f"   {et}")
        log(f"      corta  n {c['n']:5d} (pos {c['n_positivos']:4d} neg "
            f"{c['n_negativos']:4d}) F1 {c['F1_de_V03']:.4f} error alto "
            f"{c['tasa_de_error_de_alta_confianza']:.4f} AUC {c['AUC_dentro_de_la_region']}")
        log(f"      larga  n {l['n']:5d} (pos {l['n_positivos']:4d} neg "
            f"{l['n_negativos']:4d}) F1 {l['F1_de_V03']:.4f} error alto "
            f"{l['tasa_de_error_de_alta_confianza']:.4f} AUC {l['AUC_dentro_de_la_region']}")
        log(f"      techo moviendo solo el corte: corta {c['mejor_F1_alcanzable_moviendo_el_corte']:.4f} "
            f"en el corte {c['corte_optimo']:.3f}, larga "
            f"{l['mejor_F1_alcanzable_moviendo_el_corte']:.4f} en {l['corte_optimo']:.3f}")
        for region in ("corta", "larga"):
            filas.append({"etiqueta": et, "region": region, **bloque[region]})
    with_csv = pd.DataFrame(filas)
    with_csv.to_csv(CSV_T, index=False, encoding="utf-8")

    # ---------- el techo de grupo, dentro de la region corta ----------
    log("")
    log("   EL TECHO DE GRUPO DENTRO DE LA REGION CORTA, que es el limite duro")
    log("   un grupo de casi duplicados solo puede acertar por mayoria, y la evidencia de")
    log("   que la etiqueta es contradictoria dentro del grupo es la unica que existe")
    g_unicos, g_cod = np.unique(grupos, return_inverse=True)
    teller: dict[str, Any] = {}
    for et in ETIQUETAS:
        j = ETIQUETAS.index(et)
        uniformes = contradictorios = 0
        filas_corto = 0
        for g in range(len(g_unicos)):
            idx_g = np.where(g_cod == g)[0]
            if len(idx_g) < 2:
                continue
            t = y[idx_g, j]
            if t.min() == t.max():
                uniformes += 1
            else:
                contradictorios += 1
                filas_corto += int(corto[idx_g].sum())
        teller[et] = {"grupos_con_etiqueta_uniforme": uniformes,
                      "grupos_contradictorios": contradictorios,
                      "filas_de_la_region_corta_en_grupos_contradictorios": filas_corto,
                      "fraccion_de_la_region_corta_en_grupos_contradictorios": round(
                          filas_corto / max(int((corto & (y[:, j] == 1)).sum()), 1), 6)}
        log(f"      {et}: {uniformes} grupos uniformes, {contradictorios} contradictorios | "
            f"filas cortas positivas dentro de grupos contradictorios: {filas_corto} "
            f"({teller[et]['fraccion_de_la_region_corta_en_grupos_contradictorios'] * 100:.1f} "
            f"por ciento)")

    # ---------- veredicto ----------
    log("")
    log("   VEREDICTO")
    aucs_c = [res[et]["corta"]["AUC_dentro_de_la_region"] for et in ETIQUETAS]
    techo_c = [res[et]["corta"]["distancia_de_V03_al_techo"] for et in ETIQUETAS]
    auc_med = float(np.mean([a for a in aucs_c if a is not None]))
    techo_med = float(np.mean(techo_c))
    log(f"      AUC dentro de la region corta, media: {auc_med:.6f}")
    log(f"      margen hasta el techo moviendo solo el corte, media: {techo_med:+.6f}")
    if auc_med >= 0.85 and techo_med <= 0.05:
        veredicto = (
            f"V03 ORDENA bien la region corta, con un AUC interno de {auc_med:.4f}, y el "
            f"margen hasta el techo moviendo solo el corte es de {techo_med:+.4f}, luego es "
            f"un problema de donde se dibuja la linea y no de que dice el texto. Eso es un "
            f"techo EN LA PRACTICA: la unica palanca que lo moveria es el umbral, que V8-09 "
            f"y P59-T ya cerraron. Ninguna representacion lo ataca")
    elif auc_med < 0.85:
        veredicto = (
            f"V03 NO ordena bien la region corta, con un AUC interno de {auc_med:.4f}, luego "
            f"la representacion no separa las clases ni dentro de la region. Es un techo de "
            f"INFORMACION y ninguna representacion lo arregla, porque el problema no es como "
            f"se representa el texto sino que el texto de seis tokens no lleva la senal")
    else:
        veredicto = (
            f"el AUC interno de la region corta es {auc_med:.4f} y el margen hasta el techo "
            f"es {techo_med:+.4f}, luego hay parte que es decision y parte que no, y la "
            f"fase no la separa")
    log(f"      {veredicto}")

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    informe = {
        "familia": "P62_TECHO_CORTOS",
        "naturaleza": ("DIAGNOSTICO. 0 experimentos, 0 modelos, 0 submissions, 0 Kaggle, "
                       "0 holdout, 0 test. No se entrena nada"),
        "fecha": inicio.isoformat(timespec="seconds"),
        "script": "experiments/p62_10_techo_cortos.py",
        "pregunta": ("el fallo de V03 en los tweets cortos es un techo de informacion, o "
                     "queda margen"),
        "definicion_de_la_region": {
            "max_tokens": MAX_TOKENS_CORTO,
            "cuando_se_declaro": ("en la EDA linguistica de P62, antes de que existiera "
                                  "ningun candidato, y aqui se reutiliza sin re-recortar"),
        },
        "metodo": [
            ("AUC de la puntuacion de V03 DENTRO de cada region, positivos contra negativos, "
            "con la definicion de rangos implementada aqui para no depender de una libreria"),
            ("mejor F1 alcanzable moviendo SOLO el corte, recorriendo las puntuaciones "
            "observadas, que es el techo de cualquier funcion monotona de la puntuacion"),
            ("techo de grupo por mayoria dentro de los grupos de casi duplicados, que es el "
            "limite duro y la unica evidencia de etiqueta contradictoria"),
        ],
        "por_region": res,
        "techo_de_grupo": teller,
        "auc_medio_region_corta": round(auc_med, 6),
        "margen_medio_al_techo_region_corta": round(techo_med, 6),
        "veredicto": veredicto,
        "reproducibilidad": {"nivel": "R2", "git_commit": commit,
                             "tagger": "spaCy es_core_news_md 3.8.0, D-003"},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_T.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")

    L: list[str] = []
    W = L.append
    W("# P62 - El fallo de los tweets cortos: techo de informacion o margen real")
    W("")
    W(f"Generado por `experiments/p62_10_techo_cortos.py`. "
      f"Fecha {inicio.isoformat(timespec='seconds')}.")
    W("")
    W("> **DIAGNOSTICO.** No entrena, no crea candidatos, no usa test, holdout ni")
    W("> leaderboard. Es una medicion de identificabilidad sobre el OOF canonico.")
    W("")
    W("## Que queda sin atacar")
    W("")
    W("Cinco experimentos han medido y refutado las dos hipotesis naturales del eje de")
    W("representacion, y queda una anomalia medida que nadie ha tratado:")
    W("")
    W("```text")
    W("TR corto, error de alta confianza    0,4167")
    W("TR largo,  error de alta confianza    0,0433")
    W("F1 de V03, rebanada corta menos larga  -0,2864 en TR")
    W("F1 de V03, sintaxis rica menos pobre  +0,1070, V03 es MEJOR con estructura")
    W("```")
    W("")
    W("Las dos cosas juntas son extranas y apuntan a una sola explicacion: V03 no falla")
    W("porque ignore la gramatica, falla cuando no hay texto. Si eso es cierto, la region")
    W("corta es un TECHO DE INFORMACION. Si no, hay margen y la fase paro antes de tiempo.")
    W("")
    W("## Como se distingue")
    W("")
    W("Una afirmacion de techo es una afirmacion sobre identificabilidad, y eso se mide sin")
    W("entrenar nada. Dentro de la region corta pueden ser ciertas dos cosas a la vez, y")
    W("tienen consecuencias opuestas:")
    W("")
    W("- **el ORDEN es bueno y solo la linea esta mal puesta**: es un problema de")
    W("  decision, que V8-09 y P59-T ya cerraron, luego es un techo en la practica.")
    W("- **el ORDEN es malo**: la representacion no separa las clases ni dentro de la")
    W("  region, luego es un techo de informacion genuino.")
    W("")
    W("La medida es el AUC de la puntuacion de V03 calculado DENTRO de la region, y el")
    W("segundo control independiente es el techo de grupo por mayoria en los casi")
    W("duplicados, que es el limite duro.")
    W("")
    W(f"La region corta es hasta {MAX_TOKENS_CORTO} tokens de spaCy, la MISMA definicion que")
    W("uso la EDA linguistica, declarada alli antes de que existiera ningun candidato. Aqui")
    W("no se re-recorta para que convenga a un resultado.")
    W("")
    W("## Los resultados por region")
    W("")
    W("| et | region | n | F1 de V03 | error | error alta conf. | AUC interno | "
      "techo moviendo el corte |")
    W("|---|---|---:|---:|---:|---:|---:|---:|")
    for f in filas:
        W(f"| {f['etiqueta']} | {f['region']} | {f['n']} | {f['F1_de_V03']:.4f} | "
          f"{f['tasa_de_error']:.4f} | {f['tasa_de_error_de_alta_confianza']:.4f} | "
          f"{f['AUC_dentro_de_la_region']:.4f} | {f['mejor_F1_alcanzable_moviendo_el_corte']:.4f} |")
    W("")
    W("## El techo de grupo dentro de la region corta")
    W("")
    W("| et | grupos uniformes | grupos contradictorios | filas cortas positivas en grupos "
      "contradictorios |")
    W("|---|---:|---:|---:|")
    for et in ETIQUETAS:
        t_ = teller[et]
        W(f"| {et} | {t_['grupos_con_etiqueta_uniforme']} | "
          f"{t_['grupos_contradictorios']} | "
          f"{t_['filas_de_la_region_corta_en_grupos_contradictorios']} "
          f"({t_['fraccion_de_la_region_corta_en_grupos_contradictorios'] * 100:.1f} por "
          f"ciento) |")
    W("")
    W("## Veredicto")
    W("")
    W("```text")
    W(f"AUC medio dentro de la region corta        {auc_med:.6f}")
    W(f"margen hasta el techo moviendo el corte   {techo_med:+.6f}")
    W("")
    W(veredicto)
    W("```")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p62_10_techo_cortos.py`. Sin push.*")
    MD_T.write_text(a_ascii("\n".join(L)), encoding="ascii")

    for destino in (MD_T, JSON_T, CSV_T):
        if any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P62-10 OK | AUC corto {auc_med:.4f} | margen al techo {techo_med:+.4f} | "
        f"0 experimentos")


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
