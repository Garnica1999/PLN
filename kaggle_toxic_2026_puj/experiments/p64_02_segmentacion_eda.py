"""P64 track P2 and P3: deterministic segmentation, and the segment profile EDA.

Why the segmentation layer is built this way
---------------------------------------------
The gap analysis established that punctuation segmentation reaches 5,29 times more rows than
the fixed window P59-07 used, and that the error rate of V03 rises by 8,35 points between one
segment and five or more. The brief requires the segmentation to be deterministic, model-free
and built from rules on the text alone, and explicitly forbids an external parser and spaCy
for this first layer.

Four schemes are built, and they are four DIFFERENT hypotheses about where a boundary is, not
four parameters of one scheme:

    S1  strong punctuation. Split on any run of dot, exclamation or question mark. This is the
        simplest and it is the one whose coverage the gap analysis measured.

    S2  strong punctuation plus a newline, because a line break in a tweet is a boundary that
        costs no punctuation character.

    S3  structural delimiters. Only a run of TWO OR MORE of those marks, plus a semicolon or a
        colon. This makes a boundary rare and strong, and it is the scheme that should isolate
        genuinely separate statements rather than incidental interjections.

    S4  mention and quote aware. A mention or an opening quote also opens a new unit, because
        a quoted stretch in a tweet is a different speaker's text and the brief asks for
        quote-aware structure as its own track.

The choice between them is not a search over a grid. Each is reported separately with its own
coverage and its own error profile, and the reading is descriptive. If one scheme is clearly
better structurally, that is stated; if not, S1 stays as the pre-declared primary and the
others are reported as sensitivity.

What the profile is allowed to be
---------------------------------
Seven bounded quantities per row: number of segments, longest and shortest segment share,
mean and standard deviation of segment length, the share of tokens in the longest segment, and
the entropy of the segment length distribution. The brief says not to assume they work and to
look first, and this script does exactly that: it compares the profile across the four
strata of V03 behaviour that matter, namely all rows, the long and very long strata, the rows
V03 gets wrong, and the rows V03 gets wrong with high confidence.

The read that decides the next step
-----------------------------------
If the profile separates V03's errors from V03's correct rows, the structure carries
information and P4 should decompose the evidence with V03's own coefficients. If it does not
separate them, the profile is a dead end, but the phase still continues because the evidence
decomposition measures a different thing: not the shape of the segmentation but where the
model's evidence actually sits inside it.

Usage
-----
    /home/carlos/venv/bin/python experiments/p64_02_segmentacion_eda.py
"""

from __future__ import annotations

import ast
import csv as _csv
import hashlib
import json
import re
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
    V03_CCV,
    V03_SHA,
    V03_SUB,
)

REPORTS = PROJECT_ROOT / "reports"
OOF = PROJECT_ROOT / "features" / "oof_v03_final.npz"
RAW_TRAIN = PROJECT_ROOT / "data" / "raw" / "toxic_train.csv"
MD_E = REPORTS / "p64_segment_eda.md"
JSON_E = REPORTS / "p64_segment_eda.json"
CSV_E = REPORTS / "p64_segment_analysis.csv"
CELDA_E = PROJECT_ROOT / "features" / "p64_segment_profile.npz"

V03_UMBRALES = np.array([0.44, 0.51, 0.46])
DELTA_HC = 0.10
# Cortes de longitud CONGELADOS de P63.
BINS: tuple[tuple[str, int, int], ...] = (
    ("CORTA", 0, 6),
    ("MEDIA", 7, 15),
    ("LARGA", 16, 30),
    ("MUY_LARGA", 31, 10**9),
)
# El esquema S1 se declara aqui como PRIMARIO antes de mirar ningun resultado, y los otros
# tres se reportan como sensibilidad. No se elige despues.
PRIMARIO = "S1"

RE_S1 = re.compile(r"[.!?]+")
RE_S2 = re.compile(r"[.!?]+|\n+")
RE_S3 = re.compile(r"(?:[.!?]+){2,}|[;:]+")
RE_MENCION = re.compile(r"@\w+")
RE_URL = re.compile(r"https?://|www\.")
RE_HASHTAG = re.compile(r"#\w+")
RE_ESPACIO = re.compile(r"\s+")


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P64-02] {mensaje}", flush=True)


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


def segmentar(texto: str, esquema: str) -> list[str]:
    """Segmenta un texto con una de las cuatro reglas declaradas.

    Ninguna regla consulta una etiqueta, un lexicon ni un modelo, y todas son
    deterministas: el mismo texto produce siempre los mismos segmentos.

    Args:
        texto: texto del registro.
        esquema: S1, S2, S3 o S4.

    Returns:
        Lista de segmentos no vacios.

    Raises:
        ValueError: si el esquema no existe.
    """
    if esquema == "S1":
        partes = RE_S1.split(texto)
    elif esquema == "S2":
        partes = RE_S2.split(texto)
    elif esquema == "S3":
        partes = RE_S3.split(texto)
    elif esquema == "S4":
        # Una mencion o una comilla de apertura abren unidad propia, y la comilla de cierre
        # tambien. Es una regla de estructura, no una regla de identidad: no se extrae el
        # nombre de nadie.
        partes = re.split(r"(?=[@#])|(?=[.!?]+)|(?<=\")|\n+", texto)
    else:
        raise ValueError(f"esquema de segmentacion desconocido: {esquema}")
    salida = [p for p in (s.strip() for s in partes) if p]
    return salida or [texto.strip() or texto]


def perfil(segmentos: list[str], n_tok_total: int) -> dict[str, float]:
    """Perfil acotado de la segmentacion de una fila.

    Args:
        segmentos: segmentos ya cortados.
        n_tok_total: numero de tokens del texto completo.

    Returns:
        Diccionario de siete rasgos acotados.
    """
    longitudes = np.array([len(RE_ESPACIO.split(s)) for s in segmentos], dtype=float)
    longitudes = np.maximum(longitudes, 1.0)
    total = max(n_tok_total, 1)
    if len(longitudes) > 1:
        p = longitudes / longitudes.sum()
        ent = float(-(p * np.log2(p)).sum())
    else:
        ent = 0.0
    return {
        "n_segmentos": float(len(segmentos)),
        "log_n_segmentos": float(np.log1p(len(segmentos))),
        "es_multi_segmento": float(len(segmentos) > 1),
        "fraccion_del_segmento_mas_largo": float(longitudes.max() / total),
        "fraccion_del_segmento_mas_corto": float(longitudes.min() / total),
        "desviacion_de_la_longitud": float(longitudes.std()) / 10.0,
        "entropia_de_la_longitud": ent / 3.0,
    }


def main() -> None:
    """Construye las cuatro segmentaciones y hace la EDA del perfil de segmento.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P64-02 segmentacion determinista y EDA del perfil de segmento")
    log(f"   esquema primario declarado ANTES de mirar nada: {PRIMARIO}")
    log("   S1 puntuacion | S2 puntuacion y salto | S3 delimitadores fuertes | "
        "S4 mencion y cita")
    log("   0 experimentos, 0 modelos, no entrena nada")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    from sklearn.metrics import f1_score
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
        raise SystemExit("PARADA: las etiquetas del CSV crudo no coinciden con el OOF")
    textos = idx["text"].astype(str).tolist()
    n_tok = np.array([len(t.split()) for t in textos])
    log("")
    log(f"   OOF canonico {ccv_ref!r} | {n} filas")

    error = (pred_v03 != y)
    dist = np.abs(proba_v03 - V03_UMBRALES[None, :])
    error_alto = error & (dist >= DELTA_HC)

    # ---------- las cuatro segmentaciones ----------
    esquemas = ("S1", "S2", "S3", "S4")
    segmentos: dict[str, list[list[str]]] = {}
    perfiles: dict[str, np.ndarray] = {}
    nombres_perfil: list[str] = []
    log("")
    log("   LAS CUATRO SEGMENTACIONES, con su cobertura y su tasa de error")
    log(f"      {'esquema':>9}{'multi':>8}{'porc.':>8}{'seg. med.':>11}{'seg. max':>10}"
        f"{'error':>9}{'error multi':>13}")
    cov: dict[str, Any] = {}
    for k, esquema in enumerate(esquemas):
        segs = [segmentar(t, esquema) for t in textos]
        segmentos[esquema] = segs
        conteo = np.array([len(s) for s in segs])
        multi = conteo >= 2
        filas_perfil = [perfil(s, int(n_tok[i])) for i, s in enumerate(segs)]
        if k == 0:
            nombres_perfil = list(filas_perfil[0].keys())
        P = np.array([[f[c] for c in nombres_perfil] for f in filas_perfil], dtype=float)
        perfiles[esquema] = P
        cov[esquema] = {
            "n_multi_segmento": int(multi.sum()),
            "porcentaje_multi": round(float(multi.mean() * 100), 4),
            "segmentos_medianos": float(np.median(conteo)),
            "segmentos_maximos": int(conteo.max()),
            "tasa_de_error": round(float(error.mean()), 6),
            "tasa_de_error_multi": round(float(error[multi].mean()), 6) if multi.any() else None,
            "tasa_de_error_uni": round(float(error[~multi].mean()), 6),
            "vacio_o_uno": int((conteo == 0).sum()),
        }
        c = cov[esquema]
        log(f"      {esquema:>9}{c['n_multi_segmento']:>8}{c['porcentaje_multi']:>7.1f}%"
            f"{c['segmentos_medianos']:>11.0f}{c['segmentos_maximos']:>10}"
            f"{c['tasa_de_error'] * 100:>8.2f}%"
            f"{(c['tasa_de_error_multi'] or 0) * 100:>12.2f}%")
    log("")
    log("      S4 parte por mencion, por hashtag y por comilla, luego da mas unidades que")
    log("      S1. Eso NO es mejor: son unidades mas pequenas y mas homogeneas, y el")
    log("      objetivo de la fase es evidencia LOCAL que compita con la global, no trocear")
    log("      mas. Por eso el primario declarado sigue siendo S1 y S4 se reporta como")
    log("      sensibilidad para la linea de citas del encargo.")

    # ---------- la EDA del perfil ----------
    log("")
    log("   LA EDA DEL PERFIL, que es la pregunta de P3")
    log("   los tweets que V03 no puede resolver contienen una estructura segmental "
        "distinta?")
    P = perfiles[PRIMARIO]
    estratos: list[tuple[str, np.ndarray]] = [
        ("todas", np.ones(n, dtype=bool)),
    ]
    for nombre, lo, hi in BINS:
        estratos.append((nombre, (n_tok >= lo) & (n_tok <= hi)))
    estratos.append(("V03 correcto", ~error.any(axis=1)))
    estratos.append(("V03 incorrecto", error.any(axis=1)))
    estratos.append(("error de alta confianza", error_alto.any(axis=1)))

    log("")
    log(f"      {'estrato':>26}{'n':>7}" + "".join(f"{c[:16]:>18}" for c in nombres_perfil))
    filas_eda: list[dict[str, Any]] = []
    for nombre, m in estratos:
        if m.sum() < 5:
            continue
        medias = P[m].mean(axis=0)
        fila = {"estrato": nombre, "n": int(m.sum()),
                "tasa_de_error": round(float(error[m].mean()), 6)}
        for j, c in enumerate(nombres_perfil):
            fila[c] = round(float(medias[j]), 6)
        filas_eda.append(fila)
        log(f"      {nombre:>26}{int(m.sum()):>7}"
            + "".join(f"{medias[j]:>18.4f}" for j in range(len(nombres_perfil))))

    # ---------- la separacion, que es lo que decide si el perfil sirve ----------
    log("")
    log("   LA SEPARACION, que es la lectura que decide el paso siguiente")
    from sklearn.metrics import roc_auc_score
    {c: {e: filas_eda[[f["estrato"] for f in filas_eda].index(e)][c]
               for e in ("V03 correcto", "V03 incorrecto") if e in
               [f["estrato"] for f in filas_eda]} for c in nombres_perfil}
    aucs: dict[str, Any] = {}
    for j, c in enumerate(nombres_perfil):
        objetivo = error.any(axis=1).astype(int)
        if objetivo.sum() < 10 or (1 - objetivo).sum() < 10:
            continue
        a = float(roc_auc_score(objetivo, P[:, j]))
        aucs[c] = round(a, 6)
        log(f"      AUC de {c:34} para predecir si V03 falla: {a:.6f}")
    # El AUC de 0,5 es el azar, y un AUC de un rasgo estructural tiene que compararse con el
    # de un rasgo trivial de longitud, porque si solo mide longitud no aporta nada.
    auc_tokens = float(roc_auc_score(error.any(axis=1).astype(int), n_tok))
    aucs["n_tokens_como_referencia"] = round(auc_tokens, 6)
    log(f"      AUC de n_tokens, como REFERENCIA trivial: {auc_tokens:.6f}")
    mejores = {k: v for k, v in aucs.items() if k != "n_tokens_como_referencia"}
    clave = max(mejores, key=lambda k: abs(mejores[k] - 0.5)) if mejores else None
    ventaja = (abs(aucs[clave] - 0.5) - abs(auc_tokens - 0.5)) if clave else 0.0
    s_sep = {
        "auc_por_rasgo": aucs,
        "auc_de_referencia_por_longitud": round(auc_tokens, 6),
        "mejor_rasgo": clave,
        "ventaja_sobre_la_referencia_de_longitud": round(ventaja, 6),
    }
    log("")
    log(f"      el mejor rasgo es {clave}, con una ventaja de {ventaja:+.6f} sobre la")
    log("      referencia trivial de longitud")

    # ---------- veredicto de la EDA ----------
    if ventaja > 0.01:
        veredicto_eda = (
            f"el perfil segmental separa los errores de V03 mas alla de lo que la longitud "
            f"ya explica: {clave} con AUC {aucs[clave]:.6f} frente a {auc_tokens:.6f} de la "
            f"longitud, una ventaja de {ventaja:+.6f}. La estructura interna es un rasgo con "
            f"senal propia y P4 debe descomponer la evidencia con los coeficientes de V03")
        decision_eda = "PERFIL CON SEnAL. Continuar con descomposicion de evidencia"
    elif ventaja > -0.005:
        veredicto_eda = (
            f"el perfil segmental no separa los errores mas alla de la longitud: la ventaja "
            f"sobre la referencia trivial es {ventaja:+.6f}, dentro del ruido. El perfil como "
            f"rasgo queda cerrado, pero la fase continua porque P4 mide otra cosa: no la FORMA "
            f"de la segmentacion sino donde se situa la evidencia del modelo dentro de ella")
        decision_eda = "PERFIL SIN SEnAL PROPIA. Continuar con descomposicion de evidencia"
    else:
        veredicto_eda = (
            f"el perfil segmental es PEOR que la referencia trivial de longitud, con una "
            f"ventaja de {ventaja:+.6f}. Es una asociacion de la longitud disfrazada de "
            f"estructura, y queda cerrado como rasgo")
        decision_eda = "PERFIL CERRADO. Continuar con descomposicion de evidencia"
    log("")
    log(f"   VEREDICTO DE LA EDA: {veredicto_eda}")
    log(f"   DECISION: {decision_eda}")

    # ---------- artefactos ----------
    with CSV_E.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["tipo", "clave"] + nombres_perfil + ["tasa_de_error", "nota"])
        for f in filas_eda:
            w.writerow(["eda", f["estrato"], f["n"]] + [f[c] for c in nombres_perfil]
                       + [f["tasa_de_error"], ""])
        for esquema in esquemas:
            c = cov[esquema]
            w.writerow(["cobertura", esquema, c["n_multi_segmento"]]
                       + [c["segmentos_medianos"], c["segmentos_maximos"],
                          c["porcentaje_multi"] / 100, c["tasa_de_error"]]
                       + [f"tasa de error multi {c['tasa_de_error_multi']}"])
        for c, a in aucs.items():
            w.writerow(["auc", c, a, "", "", "", "",
                        "referencia trivial de longitud"
                        if c == "n_tokens_como_referencia" else ""])

    np.savez_compressed(CELDA_E, ids=ids, y_true=y, pliegues=pliegues, pred_v03=pred_v03,
                        n_tokens=n_tok, nombres=np.array(nombres_perfil),
                        **{f"perfil_{k}": v for k, v in perfiles.items()},
                        **{f"conteo_{k}": np.array([len(s) for s in segmentos[k]])
                           for k in esquemas})

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    informe = {
        "familia": "P64_SEGMENTACION_EDA",
        "naturaleza": ("DIAGNOSTICO. 0 experimentos, 0 modelos, 0 features, 0 submissions, "
                       "0 Kaggle, 0 holdout, 0 test"),
        "fecha": inicio.isoformat(timespec="seconds"), "experiment_id": "P64-01",
        "script": "experiments/p64_02_segmentacion_eda.py",
        "pregunta": ("los tweets que V03 no puede resolver contienen una estructura segmental "
                     "distinta?"),
        "esquemas": {
            "S1": "puntuacion fuerte, cualquier secuencia de punto, exclamacion o interrogacion",
            "S2": "puntuacion fuerte mas salto de linea",
            "S3": ("delimitadores fuertes: dos o mas de esos signos, o punto y coma, o dos "
                   "puntos"),
            "S4": "mencion, hashtag y comilla de apertura abren unidad propia",
        },
        "primario_declarado_antes": PRIMARIO,
        "porque_s1_sigue_primario": ("S4 da mas unidades pero son mas pequenas y mas "
                                     "homogeneas. El objetivo es evidencia local que compita "
                                     "con la global, no trocear mas. S4 se reporta como "
                                     "sensibilidad para la linea de citas del encargo"),
        "determinismo": ("las cuatro reglas son expresiones regulares sobre el texto, sin "
                         "modelo, sin lexicon, sin etiqueta y sin consulta a spaCy"),
        "cobertura": cov,
        "perfil_rasgos": nombres_perfil,
        "eda_por_estrato": filas_eda,
        "separacion": s_sep,
        "veredicto": veredicto_eda,
        "decision": decision_eda,
        "experimentos_registrados": 1,
        "reproducibilidad": {"nivel": "R2", "git_commit": commit, "hash_v03": hash_real,
                             "corrected_cv": V03_CCV},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_E.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")

    L: list[str] = []
    W = L.append
    W("# P64 - Segmentacion determinista y EDA del perfil de segmento")
    W("")
    W(f"Generado por `experiments/p64_02_segmentacion_eda.py`. Fecha "
      f"{inicio.isoformat(timespec='seconds')}.")
    W("")
    W("> **DIAGNOSTICO.** No entrena, no crea features, no usa test, holdout ni leaderboard.")
    W("")
    W("## Por que la segmentacion es de esta manera")
    W("")
    W("El gap analysis establecio que la segmentacion por puntuacion alcanza **5,29 veces mas**")
    W("filas que la ventana fija de P59-07, y que la tasa de error de V03 sube **8,35 puntos**")
    W("entre un segmento y cinco o mas. El encargo exige que la segmentacion sea determinista,")
    W("sin modelo, construida con reglas sobre el texto, y prohibe expresamente un parser")
    W("externo y spaCy en esta primera capa.")
    W("")
    W("Hay cuatro esquemas, y son **cuatro hipotesis distintas** sobre donde esta un limite,")
    W("no cuatro parametros del mismo:")
    W("")
    W("```text")
    W("S1  puntuacion fuerte, cualquier secuencia de . ! o ?")
    W("S2  puntuacion fuerte mas un salto de linea")
    W("S3  delimitadores fuertes: dos o mas de esos signos, o ; o :")
    W("S4  una mencion, un hashtag o una comilla de apertura abren unidad propia")
    W("```")
    W("")
    W("La eleccion entre ellos **no es una busqueda sobre una rejilla**. Cada uno se reporta")
    W("con su cobertura y su perfil de error, y la lectura es descriptiva.")
    W("")
    W(f"El primario se declara **antes de mirar nada**: **{PRIMARIO}**.")
    W("")
    W("## Las cuatro segmentaciones")
    W("")
    W("| esquema | filas multi | porcentaje | segmentos medianos | maximos | error | "
      "error en multi |")
    W("|---|---:|---:|---:|---:|---:|---:|")
    for e in esquemas:
        c = cov[e]
        W(f"| {e} | {c['n_multi_segmento']} | {c['porcentaje_multi']:.1f}% | "
          f"{c['segmentos_medianos']:.0f} | {c['segmentos_maximos']} | "
          f"{c['tasa_de_error'] * 100:.2f}% | "
          f"{(c['tasa_de_error_multi'] or 0) * 100:.2f}% |")
    W("")
    W("**S4 parte por mencion, hashtag y comilla, luego da mas unidades que S1. Eso no es**")
    W("**mejor**: son unidades mas pequenas y mas homogeneas, y el objetivo de la fase es")
    W("evidencia local que compita con la global, no trocear mas. El primario declarado sigue")
    W("siendo S1 y S4 se reporta como sensibilidad para la linea de citas del encargo.")
    W("")
    W("## La EDA del perfil")
    W("")
    W("Siete rasgos acotados por fila: numero de segmentos, su logaritmo, indicador de multi,")
    W("fraccion del segmento mas largo y mas corto, desviacion de la longitud y entropia de la")
    W("distribucion de longitudes.")
    W("")
    W("| estrato | n | " + " | ".join(f"`{c}`" for c in nombres_perfil) + " |")
    W("|---:|---:|" + "---:|" * len(nombres_perfil))
    for f in filas_eda:
        W(f"| {f['estrato']} | {f['n']} | "
          + " | ".join(f"{f[c]:.4f}" for c in nombres_perfil) + " |")
    W("")
    W("## La separacion, que decide el paso siguiente")
    W("")
    W("| rasgo | AUC para predecir si V03 falla |")
    W("|---|---:|")
    for c, a in aucs.items():
        W(f"| {c} | {a:.6f} |")
    W("")
    W(f"El mejor rasgo es **{clave}**, con una ventaja de **{ventaja:+.6f}** sobre la")
    W("referencia trivial de longitud. Esa referencia importa: un AUC de un rasgo estructural")
    W("que solo mide longitud no aporta nada, y P63 ya mostro que la longitud explica una")
    W("parte sustancial del error.")
    W("")
    W("## Veredicto")
    W("")
    W("```text")
    W(veredicto_eda)
    W("")
    W(decision_eda)
    W("```")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p64_02_segmentacion_eda.py`. Sin push.*")
    MD_E.write_text(a_ascii("\n".join(L)), encoding="ascii")

    for destino in (MD_E, JSON_E, CSV_E):
        if any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P64-02 OK | esquema primario {PRIMARIO} | cobertura "
        f"{cov[PRIMARIO]['n_multi_segmento']} filas | mejor rasgo {clave} | ventaja "
        f"{ventaja:+.6f} | 0 experimentos")


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
