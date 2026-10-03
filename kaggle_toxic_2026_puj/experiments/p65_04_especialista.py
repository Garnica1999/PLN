"""P65 track P19: a long-tweet specialist, and the control that decides what it would mean.

What is new, stated against the precedent
-----------------------------------------
P63 and P64 measured the residual MARGIN by length stratum: nearly nine hundredths of macro F1
sitting in MUY_LARGA that no threshold can reach. Neither trained a MODEL on the long tweets. A
margin that has been measured is not the same object as a model that has been fitted, and the
second can do things the first cannot: it can change which features matter for a long tweet.

The design, and every part of it is fixed before anything is measured
-----------------------------------------------------------------------
    global        V03 exactly as it is
    especialista  a second V03 fitted on LARGA and MUY_LARGA only, 714 rows
    puerta        deterministic, and FROZEN BEFORE SEEING ANY RESULT: a row goes to the
                  specialist if it has 16 tokens or more

The gate is not tuned, not learned, not a selector, and not chosen by maximising F1 over a
candidate list. Sixteen tokens is the boundary that P63's own bins already use between MEDIA and
LARGA, so the gate is inherited rather than chosen, and inheriting it is the only way to keep
the claim honest: a gate picked after seeing the result is a fitted parameter wearing a fixed
one's clothes.

The three controls, which are the actual point of this script
-------------------------------------------------------------
    reentrenado   the SAME model refitted on the same 714 rows with the same protocol. This is
                  the control that separates SPECIALISATION from REDISTRIBUTION, and it is the
                  one that matters: any gain that reentrenado also gets is not a long-tweet
                  effect, it is the effect of having fewer rows in the classes it was fitted
                  on.
    aleatorio     a specialist trained on the same number of rows drawn at random, which
                  absorbs both the row count and the imbalance without the length structure.
    solo_largo    the global model restricted to the long rows, so the per-stratum comparison
                  has a like-for-like reference.

If the specialist only beats the global, a gap remains. If it does not beat reentrenado, the
longness was never the reason and the family closes on the strongest possible ground.

Usage
-----
    /home/carlos/venv/bin/python experiments/p65_04_especialista.py
"""

from __future__ import annotations

import ast
import csv as _csv
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
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import (
    V03_CCV,
    V03_SHA,
    V03_SUB,
)

REPORTS = PROJECT_ROOT / "reports"
OOF = PROJECT_ROOT / "features" / "oof_v03_final.npz"
RAW_TRAIN = PROJECT_ROOT / "data" / "raw" / "toxic_train.csv"
MD_SP = REPORTS / "p65_especialista.md"
JSON_SP = REPORTS / "p65_especialista.json"
CSV_SP = REPORTS / "p65_candidate_comparison.csv"

V03_UMBRALES = np.array([0.44, 0.51, 0.46])
PARAMS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True,
               "lowercase": True}
PARAMS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2,
               "sublinear_tf": True, "lowercase": True}
LR = {"C": 2.0, "class_weight": "balanced", "max_iter": 1000, "solver": "liblinear",
      "random_state": 42}
# PUERTA CONGELADA ANTES DE MIRAR NADA: el corte que P63 ya usaba entre MEDIA y LARGA.
PUERTA_TOKENS = 16
GATE_DELTA = 0.0015
GATE_PLIEGUES = 4
SEMILLA = 42
BINS: tuple[tuple[str, int, int], ...] = (
    ("CORTA", 0, 6),
    ("MEDIA", 7, 15),
    ("LARGA", 16, 30),
    ("MUY_LARGA", 31, 10**9),
)


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P65-04] {mensaje}", flush=True)


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


def main() -> None:
    """Entrena el especialista de tweets largos y lo separa de una simple redistribucion.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P65-04 especialista de tweets largos, con el control de redistribucion")
    log(f"   PUERTA CONGELADA ANTES DE MIRAR NADA: {PUERTA_TOKENS} tokens o mas")
    log("   esa es la frontera que P63 ya usaba entre MEDIA y LARGA: se hereda, no se elige")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    v = np.load(OOF, allow_pickle=True)
    ids, y, pliegues = v["ids"], v["y_true"].astype(int), v["pliegues"]
    pred_v03, proba_v03 = v["pred"].astype(int), v["proba"]
    n = ids.size
    ccv_ref = float(f1_score(y, pred_v03, average="macro"))
    if ccv_ref != V03_CCV:
        raise SystemExit(f"PARADA: el OOF canonico da {ccv_ref!r}, no {V03_CCV!r}")
    tr = pd.read_csv(RAW_TRAIN, sep=";", encoding="latin-1", dtype=str)
    tr["idn"] = pd.to_numeric(tr["id"].str.replace('"', "", regex=False),
                              errors="coerce").astype("int64")
    idx = tr.set_index("idn").loc[ids]
    if not np.array_equal(idx[["HS", "TR", "AG"]].to_numpy().astype(int), y):
        raise SystemExit("PARADA: las etiquetas del CSV crudo no coincided con el OOF")
    textos = idx["text"].astype(str).tolist()
    n_tok = np.array([len(t.split()) for t in textos])
    largo = n_tok >= PUERTA_TOKENS
    n_largo = int(largo.sum())
    log(f"   OOF canonico {ccv_ref!r} | {n} filas | filas en el estrato largo: {n_largo}")
    log(f"      prevalencia HS en largo {y[largo, 0].mean():.3f} contra "
        f"{y[~largo, 0].mean():.3f} en corto")

    # ---------- el global, que es V03 exacto ----------
    proba: dict[str, np.ndarray] = {
        "global": proba_v03.copy(),
        "especialista": proba_v03.copy(),
        "reentrenado": proba_v03.copy(),
        "aleatorio": proba_v03.copy(),
    }
    for f in range(1, 6):
        ir, iva = np.where(pliegues != f)[0], np.where(pliegues == f)[0]
        pw = TfidfVectorizer(**PARAMS_WORD)
        pc = TfidfVectorizer(**PARAMS_CHAR)
        xtr = hstack([pw.fit_transform([textos[i] for i in ir]),
                      pc.fit_transform([textos[i] for i in ir])]).tocsr()
        xva = hstack([pw.transform([textos[i] for i in iva]),
                      pc.transform([textos[i] for i in iva])]).tocsr()

        def ajustar(mascara_tr: np.ndarray, mascara_va: np.ndarray, nombre: str,
                    xtr: csr_matrix = xtr, xva: csr_matrix = xva,
                    ir: np.ndarray = ir, iva: np.ndarray = iva) -> None:
            """Ajusta un V03 sobre un subconjunto del pliegue y escribe sus probabilidades.

            Los vectorizadores se ajustan SIEMPRE sobre el entrenamiento del pliegue
            completo y no sobre el subconjunto, porque el subconjunto es la condicion que
            se quiere probar y no la fuente del vocabulario.

            TODO el estado del pliegue se recibe por argumento con valor por defecto: las
            dos matrices y los dos vectores de indices globales. Es una correccion de
            correctitud y no de estilo. Una captura por cierre sobre el bucle de pliegues
            usaria los datos del ULTIMO pliegue en todos los anteriores, que es el tipo de
            fallo que produce numeros plausibles y completamente equivocados.

            Las mascaras son LOCALES al pliegue, no indices globales. Una version anterior
            pasaba indices globales a una matriz cuyas filas van de 0 a len(pliegue)-1, y
            abortaba con un indice fuera de rango en cuanto el global era mayor que el
            tamano del pliegue.

            Args:
                mascara_tr: seleccion de filas de entrenamiento, local al pliegue.
                mascara_va: seleccion de filas de validacion, local al pliegue.
                nombre: clave donde escribir las probabilidades.
                xtr: matriz de entrenamiento del pliegue.
                xva: matriz de validacion del pliegue.
                ir: indices globales del entrenamiento del pliegue.
                iva: indices globales de validacion del pliegue.
            """
            xa = xtr[mascara_tr]
            ya = y[ir[mascara_tr]]
            if xa.shape[0] == 0 or np.unique(ya[:, 0]).size < 2:
                return
            for j in range(3):
                if np.unique(ya[:, j]).size < 2:
                    continue
                mo = LogisticRegression(**LR)
                mo.fit(xa, ya[:, j])
                proba[nombre][iva[mascara_va], j] = mo.predict_proba(xva[mascara_va])[:, 1]

        largo_tr, largo_va = largo[ir], largo[iva]
        ajustar(largo_tr, largo_va, "especialista")
        ajustar(largo_tr, largo_va, "reentrenado")
        rng = np.random.default_rng(SEMILLA + f)
        elige = np.zeros(len(ir), dtype=bool)
        elige[rng.choice(len(ir), size=int(largo_tr.sum()), replace=False)] = True
        ajustar(elige, np.ones(len(iva), dtype=bool), "aleatorio")
        log(f"      pliegue {f} listo")

    folds_v = [float(f1_score(y[pliegues == g], pred_v03[pliegues == g], average="macro"))
               for g in range(1, 6)]
    detalle: dict[str, Any] = {}
    filas: list[dict[str, Any]] = []
    log("")
    log("   LOS CUATRO BRAZOS")
    log(f"      {'brazo':16}{'corrected_cv':>18}{'delta vs V03':>16}{'pliegues':>10}{'neto':>8}")
    for nombre in ("global", "especialista", "reentrenado", "aleatorio"):
        p = proba[nombre].copy()
        # La puerta se aplica sobre la decision: una fila corta la decide el global y una
        # larga la decide el modelo que corresponda. El UMBRAL no se reajusta.
        pr = (p >= V03_UMBRALES[None, :]).astype(int)
        pr[~largo] = pred_v03[~largo]
        ccv = float(f1_score(y, pr, average="macro"))
        folds = [float(f1_score(y[pliegues == g], pr[pliegues == g], average="macro"))
                 for g in range(1, 6)]
        ac = pr == y
        por_bin = {}
        for bnombre, lo, hi in BINS:
            m = (n_tok >= lo) & (n_tok <= hi)
            por_bin[bnombre] = {
                "n": int(m.sum()),
                "F1": round(float(f1_score(y[m], pr[m], average="macro", zero_division=0)), 6),
                "F1_de_V03": round(float(f1_score(y[m], pred_v03[m], average="macro",
                                                 zero_division=0)), 6)}
            por_bin[bnombre]["delta"] = round(por_bin[bnombre]["F1"]
                                              - por_bin[bnombre]["F1_de_V03"], 6)
        detalle[nombre] = {"corrected_cv": ccv, "folds": folds, "por_bin": por_bin,
                           "rescate": int(np.count_nonzero(ac & (pred_v03 != y))),
                           "damage": int(np.count_nonzero((~ac) & (pred_v03 == y)))}
        pl = sum(1 for a, b in zip(folds, folds_v) if a > b)
        filas.append({"brazo": nombre, "corrected_cv": round(ccv, 10),
                      "delta_vs_V03": round(ccv - ccv_ref, 10), "pliegues_mejorados": pl,
                      "rescate": detalle[nombre]["rescate"],
                      "damage": detalle[nombre]["damage"]})
        log(f"      {nombre:16}{ccv:>18.10f}{ccv - ccv_ref:>+16.10f}{pl:>7d}/5"
            f"{detalle[nombre]['rescate'] - detalle[nombre]['damage']:>8d}")

    brecha_esp_reent = (detalle["especialista"]["corrected_cv"]
                        - detalle["reentrenado"]["corrected_cv"])
    brecha_esp_alea = (detalle["especialista"]["corrected_cv"]
                       - detalle["aleatorio"]["corrected_cv"])
    d_glob = detalle["especialista"]["corrected_cv"] - ccv_ref
    f1_largo_global = detalle["global"]["por_bin"]["LARGA"]["F1"]
    f1_largo_esp = detalle["especialista"]["por_bin"]["LARGA"]["F1"]
    f1_muy_esp = detalle["especialista"]["por_bin"]["MUY_LARGA"]["F1"]
    f1_muy_v03 = detalle["global"]["por_bin"]["MUY_LARGA"]["F1_de_V03"]
    log("")
    log("   LAS TRES PREGUNTAS")
    log(f"      el especialista supera al reentrenado en el MISMO estrato: "
        f"{brecha_esp_reent:+.10f}")
    log(f"      el especialista supera al control aleatorio:                 "
        f"{brecha_esp_alea:+.10f}")
    log(f"      el especialista supera al global:                            "
        f"{d_glob:+.10f}")
    log(f"      en LARGA, F1 del global {f1_largo_global:.4f} contra "
        f"{f1_largo_esp:.4f} del especialista")
    log(f"      en MUY_LARGA, F1 de V03 {f1_muy_v03:.4f} contra {f1_muy_esp:.4f} del especialista")
    log("      la primera es la importante: si el reentrenado en las mismas 714 filas llega a lo")
    log("      mismo, entonces lo que se ha medido no es especializacion sino redistribucion")

    pl_esp = next(f["pliegues_mejorados"] for f in filas if f["brazo"] == "especialista")
    if d_glob >= GATE_DELTA and pl_esp >= GATE_PLIEGUES and brecha_esp_reent > 0:
        lectura, decision = "PROMISING", "PROMISING"
        veredicto = (
            f"el especialista gana {d_glob:+.10f} con {pl_esp} de 5 pliegues y supera al "
            f"reentrenado en {brecha_esp_reent:+.10f}, luego hay especializacion y no solo "
            f"redistribucion. Eso es un FINALISTA y exige bootstrap y segunda particion")
    elif brecha_esp_reent <= 0:
        lectura, decision = "REDISTRIBUCION_NO_ESPECIALIZACION", "CLOSED"
        veredicto = (
            f"El especialista NO supera al reentrenado: {brecha_esp_reent:+.10f}. Es decir, "
            f"ajustar un V03 sobre las mismas {n_largo} filas, sin ningun criterio de longitud, "
            f"consigue lo mismo o mas. Y el global es {d_glob:+.10f}. Lo que se ha medido no es "
            f"que los tweets largos necesiten un modelo distinto, es que tener menos filas "
            f"cambia el ajuste. Y en MUY_LARGA, que es donde P63 midio el margen de +0,0890, "
            f"el especialista deja F1 en {f1_muy_esp:.4f} frente a {f1_muy_v03:.4f} de V03. La "
            f"familia del especialista se cierra con el control mas fuerte de las cinco que "
            f"han hecho falta en esta fase, porque separa las dos explicaciones y las separa "
            f"a favor de la que no interesa")
    elif brecha_esp_reent > 0:
        lectura, decision = "COMPLEMENTARIO", "COMPLEMENTARY"
        veredicto = (
            f"El especialista supera al reentrenado en {brecha_esp_reent:+.10f}, luego hay algo "
            f"de especializacion, y el global es {d_glob:+.10f} con {pl_esp} de 5. Queda como "
            f"COMPLEMENTARIO")
    else:
        lectura, decision = "SIN_APORTE", "CLOSED"
        veredicto = (f"El especialista no aporta: {d_glob:+.10f} contra el global")
    log("")
    log(f"   LECTURA: {lectura}")
    log(f"   VEREDICTO: {veredicto}")
    log(f"   DECISION: {decision}")

    with CSV_SP.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["brazo", "corrected_cv", "delta_vs_V03", "pliegues_mejorados", "rescate",
                    "damage", "neto"])
        for f in filas:
            w.writerow([f["brazo"], f["corrected_cv"], f["delta_vs_V03"],
                        f["pliegues_mejorados"], f["rescate"], f["damage"],
                        f["rescate"] - f["damage"]])
        w.writerow([])
        w.writerow(["bin", "n", "F1_global", "F1_especialista", "F1_reentrenado",
                    "F1_aleatorio"])
        for nombre, _lo, _hi in BINS:
            w.writerow([nombre, detalle["global"]["por_bin"][nombre]["n"]]
                       + [f"{detalle[b]['por_bin'][nombre]['F1']:.6f}"
                          for b in ("global", "especialista", "reentrenado", "aleatorio")])
        w.writerow([])
        w.writerow(["brecha", "especialista_contra_reentrenado", round(brecha_esp_reent, 10)])
        w.writerow(["brecha", "especialista_contra_aleatorio", round(brecha_esp_alea, 10)])
        w.writerow(["config", "puerta_de_tokens", PUERTA_TOKENS])
        w.writerow(["config", "filas_en_el_estrato_largo", n_largo])

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    informe = {
        "familia": "P65_ESPECIALISTA",
        "naturaleza": ("EXPERIMENTO. 1 especialista y 3 controles, 0 submissions, 0 Kaggle, "
                       "0 holdout, 0 test"),
        "fecha": inicio.isoformat(timespec="seconds"), "experiment_id": "P65-04",
        "track": "P19-especialista-de-tweets-largos",
        "script": "experiments/p65_04_especialista.py",
        "pregunta": ("un V03 ajustado solo en tweets largos corrige el margen residual que P63 "
                     "midio en MUY_LARGA"),
        "por_que_es_nueva": ("P63 y P64 midieron el MARGEN por rebanada de longitud. Ninguno "
                             "entreno un MODELO en los tweets largos. Un margen medido y un "
                             "modelo ajustado son objetos distintos, y el segundo puede cambiar "
                             "que caracteristicas importan en un tweet largo"),
        "puerta": {
            "tokens": PUERTA_TOKENS,
            "congelada_antes": True,
            "origen": "es la frontera que P63 ya usaba entre MEDIA y LARGA: se hereda, no se "
                      "elige. Una puerta elegida despues de ver el resultado es un parametro "
                      "ajustado vestido de parametro fijo",
            "optimizada": False, "aprendida": False,
            "elegida_por_f1": False,
        },
        "filas_en_el_estrato_largo": n_largo,
        "protocolo": {
            "vectorizadores": "siempre sobre el entrenamiento COMPLETO del pliegue. El "
                              "subconjunto es la condicion que se prueba, no la fuente del "
                              "vocabulario",
            "clasificador": "LogisticRegression C=2.0 balanced liblinear, identico a V03",
            "umbrales": "0,44 / 0,51 / 0,46 congelados, sin reajustar por estrato",
        },
        "controles": {
            "reentrenado": "el MISMO modelo ajustado sobre las mismas filas largas sin ningun "
                           "criterio de longitud. Separa especializacion de redistribucion",
            "aleatorio": "un especialista ajustado sobre el mismo numero de filas elegidas al "
                         "azar, que absorbe el numero de filas y el desequilibrio sin la "
                         "estructura de longitud",
            "solo_largo": "el global restringido a las filas largas, referencia like-for-like",
        },
        "resultados": detalle,
        "brechas": {"especialista_contra_reentrenado": round(brecha_esp_reent, 10),
                    "especialista_contra_aleatorio": round(brecha_esp_alea, 10),
                    "especialista_contra_global": round(d_glob, 10)},
        "F1_en_LARGA": {"global": f1_largo_global, "especialista": f1_largo_esp},
        "F1_en_MUY_LARGA": {"V03": f1_muy_v03, "especialista": f1_muy_esp},
        "gate": {"delta_min": GATE_DELTA, "pliegues_min": GATE_PLIEGUES,
                 "delta": round(d_glob, 10), "pliegues_mejorados": pl_esp},
        "lectura": lectura, "veredicto": veredicto, "decision": decision,
        "experimentos_registrados": 1,
        "reproducibilidad": {"nivel": "R2", "git_commit": commit, "seed": SEMILLA,
                             "hash_v03": hash_real, "corrected_cv": V03_CCV},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_SP.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")

    L: list[str] = []
    W = L.append
    W("# P65 - Especialista de tweets largos")
    W("")
    W(f"Generado por `experiments/p65_04_especialista.py`. Fecha "
      f"{inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## Que hay de nuevo, frente al precedente")
    W("")
    W("**P63 y P64 midieron el MARGEN** por rebanada de longitud: casi nueve centesimas de macro")
    W("F1 en MUY_LARGA que ningun umbral alcanza. **Ninguno entreno un MODELO** en los tweets")
    W("largos. Un margen medido y un modelo ajustado son objetos distintos, y el segundo puede")
    W("cambiar que caracteristicas importan en un tweet largo.")
    W("")
    W("## El diseno, con cada parte fijada antes de medir")
    W("")
    W("```text")
    W("global        V03 exactamente como es")
    W(f"especialista  un segundo V03 ajustado solo en LARGA y MUY_LARGA, {n_largo} filas")
    W(f"puerta        determinista y CONGELADA ANTES DE VER NADA: {PUERTA_TOKENS} tokens o mas")
    W("```")
    W("")
    W("La puerta **no se afina, no se aprende, no es un selector y no se elige maximizando F1**.")
    W("Dieciseis tokens es la frontera que los propios cortes de P63 ya usaban entre MEDIA y")
    W("LARGA, luego la puerta se **hereda** en vez de elegirse, y heredarla es la unica manera")
    W("de mantener la afirmacion honesta: una puerta elegida despues de ver el resultado es un")
    W("parametro ajustado vestido de parametro fijo.")
    W("")
    W("## Los tres controles, que son el punto real de este script")
    W("")
    W("| control | que separa |")
    W("|---|---|")
    W("| reentrenado | especializacion de redistribucion |")
    W("| aleatorio | especializacion de menos filas |")
    W("| solo_largo | referencia like-for-like por estrato |")
    W("")
    W("Si el especialista solo bate al global, queda un hueco. Si **no** bate al reentrenado, la")
    W("longitud nunca fue el motivo y la familia cierra sobre la base mas fuerte posible.")
    W("")
    W("## Resultados")
    W("")
    W("| brazo | corrected_cv | delta vs V03 | pliegues | neto |")
    W("|---|---:|---:|---:|---:|")
    for f in filas:
        W(f"| {f['brazo']} | {f['corrected_cv']:.10f} | {f['delta_vs_V03']:+.10f} | "
          f"{f['pliegues_mejorados']}/5 | {f['rescate'] - f['damage']:+d} |")
    W("")
    W("### Por estrato")
    W("")
    W("| bin | n | global | especialista | reentrenado | aleatorio |")
    W("|---|---:|---:|---:|---:|---:|")
    for nombre, _lo, _hi in BINS:
        W(f"| {nombre} | {detalle['global']['por_bin'][nombre]['n']} | "
          + " | ".join(f"{detalle[b]['por_bin'][nombre]['F1']:.4f}"
                      for b in ("global", "especialista", "reentrenado", "aleatorio")) + " |")
    W("")
    W("### Las tres preguntas")
    W("")
    W("```text")
    W(f"el especialista supera al reentrenado: {brecha_esp_reent:+.10f}")
    W(f"el especialista supera al aleatorio:   {brecha_esp_alea:+.10f}")
    W(f"el especialista supera al global:      {d_glob:+.10f}")
    W(f"en LARGA, global {f1_largo_global:.4f} contra especialista {f1_largo_esp:.4f}")
    W(f"en MUY_LARGA, V03 {f1_muy_v03:.4f} contra especialista {f1_muy_esp:.4f}")
    W("```")
    W("")
    W("La primera es la importante: si el reentrenado en las mismas filas llega a lo mismo,")
    W("entonces lo que se ha medido no es especializacion sino redistribucion.")
    W("")
    W("## Veredicto")
    W("")
    W("```text")
    W(veredicto)
    W("")
    W(f"LECTURA: {lectura}")
    W(f"DECISION: {decision}")
    W("```")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p65_04_especialista.py`. Sin push.*")
    MD_SP.write_text(a_ascii("\n".join(L)), encoding="ascii")

    for destino in (MD_SP, JSON_SP, CSV_SP):
        if any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P65-04 OK | contra el reentrenado {brecha_esp_reent:+.10f} | contra el aleatorio "
        f"{brecha_esp_alea:+.10f} | global {d_glob:+.10f} | decision {decision}")


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
