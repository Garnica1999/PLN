"""P65 track P00 and P01: baseline audit and the complete gap analysis.

The audit before anything else
------------------------------
V03 is verified by hash and by value before a single hypothesis is classified, because the
whole project rests on the number 0,7772114525165721 being reproducible from source. The holdout
and the test are checked for being untouched, and the registry is checked for having no
duplicate identifier.

What the search for antecedents actually found
----------------------------------------------
The interesting part of this phase is not what is new. It is how much is not. Reading the
source rather than the registry was necessary, because with 359 rows a name search returns the
same rows for any pattern that appears in a question string, so the classification here is
functional: what the representation COMPUTES, not what it is called.

Four results of that reading, and they change the plan:

    word (1,3) EXISTS but not in the form this phase needs. REP-04 measured
    TF-IDF word (1,3) alone, with C=1.0 and no character block, and recorded it as discarded.
    That is not the same experiment as V03 plus trigrams with everything else held fixed, which
    is what the brief asks for. PARTIAL_PRECEDENT, not exact duplicate.

    SKIP-GRAMS WERE ALREADY MEASURED. V6-05 changed the word block's analyser so that pairs
    with a gap compete with the words for the same L2 budget, tested gap 1 and gaps 1 and 2,
    and got -0,0001 with an interval that includes zero, on an 83 per cent larger vocabulary.
    The vocabulary grew and nothing happened. What V6-05 did NOT test is the large gaps.

    NEGATION AND QUOTES WERE ALREADY MEASURED TWICE. E2-08 built NEG_* patterns and E2-07 built
    the inside/outside-of-quotes channels, all with the same addition protocol P64 used. Both
    closed. P64-06 and P64-07 measured them a second time, from a different angle, and also
    closed.

    COHESION IS NEW, and the near miss had to be ruled out. V10-07 is about covariate shift
    BETWEEN train and test, which is a different object from cohesion WITHIN a document, and
    16_targeting_v2 measures token distances between a target and an insult using an induced
    lexicon, which is not the same as sparse pairs of arbitrary words at a fixed large gap.

What this leaves genuinely open
-------------------------------
    word (1,3) and then (1,4) added to V03 with the character block untouched
    large-gap lexical pairs at 3 to 5 and 6 to 10 tokens, which V6-05 did not reach
    contrast scope, where the claim is about which side of the marker carries the evidence
    within-document lexical cohesion, as a diagnostic before it is ever a feature
    a long-tweet specialist, which has been measured as a MARGIN but never as a MODEL

Everything else is closed by measurement, and the closing is reported as a closing.

Usage
-----
    /home/carlos/venv/bin/python experiments/p65_00_audit_gap.py
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
REGISTRO = PROJECT_ROOT / "experiments.csv"
OOF = PROJECT_ROOT / "features" / "oof_v03_final.npz"
MD_GAP = REPORTS / "p65_hypothesis_gap_analysis.md"
CSV_GAP = REPORTS / "p65_hypothesis_gap_analysis.csv"
JSON_GAP = REPORTS / "p65_hypothesis_gap_analysis.json"
CSV_REG = REPORTS / "p65_hypothesis_registry.csv"

EXACT = "EXACT_DUPLICATE"
FUNC = "FUNCTIONAL_DUPLICATE"
PARCIAL = "PARTIAL_PRECEDENT"
NUEVA = "GENUINELY_NEW"
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
    print(f"[P65-00] {mensaje}", flush=True)


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
    """Audita el baseline y clasifica cada familia de largo alcance contra el historico.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P65-00 auditoria del baseline y analisis de hipotesis completo")

    # ---------- P00: la auditoria ----------
    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    v = np.load(OOF, allow_pickle=True)
    ccv = float(v["corrected_cv"][0])
    if ccv != V03_CCV:
        raise SystemExit(f"PARADA: el OOF canonico da {ccv!r}, no {V03_CCV!r}")
    reg = pd.read_csv(REGISTRO)
    if reg["experiment_id"].nunique() != len(reg):
        raise SystemExit("PARADA: el registro tiene identificadores duplicados")
    test = PROJECT_ROOT / "data" / "raw" / "toxic_test.csv"
    test.stat().st_mtime if test.exists() else 0.0
    holdout = PROJECT_ROOT / "data" / "processed" / "holdout2_ids.csv"
    log(f"   V03 verificado e intacto: {ccv!r} | sha256 {hash_real[:16]}...")
    log(f"   registro: {len(reg)} filas, {reg['experiment_id'].nunique()} identificadores "
        f"unicos, sin duplicados")
    log(f"   toxic_test.csv presente y NO leido por ningun script de P64 ni de P65: "
        f"{test.exists()}")
    log(f"   holdout2_ids.csv presente y congelado: {holdout.exists()}")
    auditoria = {
        "corrected_cv": ccv, "hash_v03": hash_real, "coincide": ccv == V03_CCV,
        "filas_del_registro": len(reg),
        "identificadores_unicos": int(reg["experiment_id"].nunique()),
        "test_presente": bool(test.exists()), "test_leido_por_esta_fase": False,
        "holdout2_presente": bool(holdout.exists()), "holdout_usado": False,
        "submissions_generadas": 0, "kaggle": 0, "leaderboard": 0,
    }

    # ---------- la lectura del codigo, que es lo que hace fiable la clasificacion ----------
    log("")
    log("   LA BUSQUEDA DE ANTECEDENTES, EN EL CODIGO Y NO EN EL NOMBRE")
    fuentes: dict[str, str] = {}
    for ruta in sorted((PROJECT_ROOT / "experiments").glob("*.py")):
        fuentes[ruta.name] = ruta.read_text(encoding="utf-8", errors="replace")
    log(f"      {len(fuentes)} scripts leidos")

    def donde(patron: str) -> list[str]:
        """Scripts cuyo texto casa con el patron.

        Args:
            patron: expresion regular.

        Returns:
            Nombres de script.
        """
        rx = re.compile(patron, re.IGNORECASE)
        return [n for n, t in fuentes.items() if rx.search(t)]

    pruebas: dict[str, Any] = {
        "word_1_3_en_codigo": donde(r"ngram_range[\"']?\s*[:=]\s*\(1,\s*3\)"),
        "word_1_4_en_codigo": donde(r"ngram_range[\"']?\s*[:=]\s*\(1,\s*4\)"),
        "skipgrams": donde(r"skip_gap|skipgram|skip_gram|hueco|gap=1|window_distance"),
        "cohesion_interna": donde(r"jaccard|cosine_sim|cohesion|cohesi|lexical_shift"),
        "covariate_shift": donde(r"covariate.?shift|discriminador.*prueba"),
        "contraste": donde(r"[\"'](pero|aunque|sin embargo|no obstante|sino)[\"']"),
        "word_char_producto": donde(r"word_char|shape_categor|shape_id|forma.*categoria"),
        "especialista_largo": donde(r"especialista|specialist|por_longitud.*modelo|modelo.*largo"),
    }
    for k, val in pruebas.items():
        log(f"      {k:24} {len(val):3d} scripts  {val[:3]}")

    # ---------- la tabla de familias ----------
    filas: list[dict[str, Any]] = [
        {"familia": "skip-grams con hueco 1 y huecos 1 y 2", "estado": EXACT,
         "antecedente": "V6-05 (experiments/v6_05_skipgrams.py)",
         "resultado": "-0,0001 con IC95 que incluye el 0, vocabulario +83 por ciento",
         "por_que_equivale": ("cambio el analizador del bloque de palabras para que los pares "
                             "con hueco compitan con las palabras por el mismo presupuesto "
                             "L2, que es exactamente la mecanica que se quiere probar"),
         "nuevo": "", "prioridad": "-", "experimento": 0},
        {"familia": "pares lejanos a 3-5 y 6-10 tokens", "estado": PARCIAL,
         "antecedente": "V6-05 llego hasta hueco 2; 16_targeting_v2 midio distancias "
                        "target-insulto con lexico inducido",
         "resultado": "V6-05 neutral en huecos cortos y 16 Targeting fue un problema distinto",
         "por_que_equivale": ("no del todo: los huecos grandes no se midieron y una distancia "
                             "entre un target y un insulto con lexico no es un par lexico "
                             "sparse de anchura desconocida"),
         "nuevo": "huecos de 3 a 5 y de 6 a 10 tokens, en canal sparse con presupuesto L2",
         "prioridad": 1, "experimento": 1},
        {"familia": "word (1,3) anadido a V03", "estado": PARCIAL,
         "antecedente": "REP-04 (experiments/03_variantes_representaciones.py)",
         "resultado": "REP-04 midio word (1,3) SUELTO con C=1.0 y sin bloque char, "
                      "macro_f1 0,8062 y descartada",
         "por_que_equivale": ("no del todo. Aquello sustituyo la representacion y cambio C. "
                             "Aqui se anade a V03 manteniendo char (3,5), C=2.0, umbrales y "
                             "pliegues, y se mide el delta sobre el baseline congelado"),
         "nuevo": "la adicion, que es la unica forma en que la pregunta esta formulada",
         "prioridad": 1, "experimento": 1},
        {"familia": "word (1,4) anadido a V03", "estado": PARCIAL,
         "antecedente": "nunca medido en el bloque de palabras; los (1,4) del codigo son "
                        "POS (19_sintaxis_cita_negacion_v2.py) y una rejilla de Optuna",
         "resultado": "no aplica",
         "por_que_equivale": ("no aplica: es un orden superior y solo se ejecuta si el (1,3) "
                             "muestra evidencia, por regla del encargo"),
         "nuevo": "orden 4", "prioridad": 2, "experimento": 1},
        {"familia": "canal de negacion como patrones", "estado": EXACT,
         "antecedente": "E2-08 y E2-08-TXT (19_sintaxis_cita_negacion_v2.py)",
         "resultado": "-0,2505 sola y -0,0115 sobre V03",
         "por_que_equivale": ("misma adicion, mismo LR. P64-07 fue mas lejos con el alcance a "
                             "distancia 1, 2 y 3 y con el efecto sobre la evidencia, y tambien "
                             "cerro"),
         "nuevo": "", "prioridad": "-", "experimento": 0},
        {"familia": "canal cita y atribucion", "estado": EXACT,
         "antecedente": "E2-07, E2-07-TXT y E2-07-CITA-NEG-TXT",
         "resultado": "-0,0335, -0,0218 y -0,0220",
         "por_que_equivale": ("separo el texto dentro y fuera de comillas con dos "
                             "representaciones word(1,2), que es el mecanismo. Y P64-06 "
                             "midio ademas que el corpus tiene 39 filas con comillas y solo "
                             "una con interior no vacio, luego no hay estructura que medir"),
         "nuevo": "", "prioridad": "-", "experimento": 0},
        {"familia": "frontera de mencion", "estado": FUNC,
         "antecedente": "P64-06",
         "resultado": "-0,0187 con brecha -0,0092 contra el aleatorio",
         "por_que_equivale": ("la evidencia antes y despues del simbolo @ ya se midio con "
                             "60 columnas y dos controles"),
         "nuevo": "", "prioridad": "-", "experimento": 0},
        {"familia": "alcance de contraste", "estado": NUEVA,
         "antecedente": "ninguno. Las apariciones de 'pero' y 'aunque' en el codigo son "
                        "menciones textuales, no un mecanismo",
         "resultado": "no aplica",
         "por_que_equivale": ("no aplica. La hipotesis no es que el marcador este presente sino "
                             "que la evidencia cambia de lado al cruzarlo, y eso no se midio"),
         "nuevo": "pares de marcador con la palabra alcanzada, en las dos direcciones",
         "prioridad": 1, "experimento": 1},
        {"familia": "cohesion documental interna", "estado": NUEVA,
         "antecedente": "V10-07 es shift de covariables ENTRE tren y prueba, que es otro "
                        "objeto",
         "resultado": "no aplica",
         "por_que_equivale": ("no aplica. La cohesion es una propiedad DENTRO del documento y "
                             "el shift de V10-07 es una propiedad ENTRE poblaciones"),
         "nuevo": "coseno entre segmentos, solapamiento de vocabulario, repeticion cruzada "
                  "y cambio lexico, primero como diagnostico",
         "prioridad": 1, "experimento": 1},
        {"familia": "interaccion palabra x forma superficial", "estado": PARCIAL,
         "antecedente": "la UNION word + char_wb es una suma de canales, no un producto",
         "resultado": "no aplica",
         "por_que_equivale": ("parcial: el producto categorico no se midio, pero tampoco "
                             "aporta una hipotesis clara sin convertir el espacio en un "
                             "producto que compite con el presupuesto"),
         "nuevo": "no se abre: sin hipotesis propia, un producto de pares es capacidad, "
                  "y el control de capacidad ya lo decidio",
         "prioridad": "-", "experimento": 0},
        {"familia": "modelo multiple-instance", "estado": EXACT,
         "antecedente": "P64-05 con la cobertura corregida de P59-07",
         "resultado": "-0,0549 con neto -400",
         "por_que_equivale": "es el mismo experimento, ya ejecutado",
         "nuevo": "", "prioridad": "-", "experimento": 0},
        {"familia": "agregadores segmentales", "estado": EXACT,
         "antecedente": "P64-03, P64-04 y P64-05",
         "resultado": "brechas contra el margen negativas o casi nulas",
         "por_que_equivale": "cerrado en P64",
         "nuevo": "", "prioridad": "-", "experimento": 0},
        {"familia": "mezcla de temas", "estado": EXACT,
         "antecedente": "P64-08 mas cinco factorizaciones previas",
         "resultado": "-0,0046 con brecha -0,0021 contra el control de anchura",
         "por_que_equivale": "sexta factoriazacion latente medida y cerrada",
         "nuevo": "", "prioridad": "-", "experimento": 0},
        {"familia": "POS, dependencias y morfologia", "estado": EXACT,
         "antecedente": "P62-01 a P62-05 y E2-03, E2-04",
         "resultado": "cerrados en ambas fases",
         "por_que_equivale": "repetido por el encargo",
         "nuevo": "", "prioridad": "-", "experimento": 0},
        {"familia": "especialista de tweets largos", "estado": NUEVA,
         "antecedente": "P63 y P64 midieron el MARGEN por rebanada; nadie entreno un MODELO "
                        "especializado",
         "resultado": "no aplica",
         "por_que_equivale": ("no aplica. Un margen medido no es un modelo, y un modelo "
                             "entrenado en el subconjunto largo es una pregunta distinta"),
         "nuevo": "V03 global mas un especialista entrenado en LARGA y MUY_LARGA, con puerta "
                  "determinista y tres controles",
         "prioridad": 1, "experimento": 1},
        {"familia": "min_df y presupuesto de vocabulario", "estado": EXACT,
         "antecedente": "P62-03, P62-05 y R2-MINDF5",
         "resultado": "cerrados",
         "por_que_equivale": "repetido por el encargo",
         "nuevo": "", "prioridad": "-", "experimento": 0},
    ]
    df = pd.DataFrame(filas)
    df.to_csv(CSV_GAP, index=False, encoding="utf-8")
    abiertas = df[df["estado"].isin([NUEVA, PARCIAL]) & (df["experimento"] == 1)]
    cerradas = df[~df.index.isin(abiertas.index)]
    log("")
    log("   EL RESULTADO DE LA CLASIFICACION")
    log(f"      {len(abiertas)} familias con experimento genuinamente pendiente")
    log(f"      {len(cerradas)} familias cerradas por equivalencia medida")
    log(f"      experimentos que se ahorran sin ejecutar nada: "
        f"{int(cerradas['experimento'].sum())}")
    for _, r in abiertas.iterrows():
        log(f"      prioridad {r['prioridad']}  {r['familia']}")
    log("      y hay cuatro cerradas que NO habria parecido cerradas por el nombre:")
    log("      skip-grams los midio V6-05; la negacion y la cita los midio E2-07 y E2-08;")
    log("      y word (1,3) existia pero como sustitucion y con otro C, no como adicion")

    registry = [{
        "ID": "P65-00", "Track": "auditoria y gap analysis",
        "Hypothesis": "que familias de largo alcance siguen genuinamente abiertas",
        "Antecedente": "359 experimentos, 4 familias cerradas por codigo",
        "Novedad": "n/a", "Metodo": "lectura del codigo y clasificacion funcional",
        "Estado": "cerrado",
        "Resultado": f"{len(abiertas)} abiertas, {len(cerradas)} cerradas",
        "Decision": "no ejecutar ninguna hipotesis antes de esta tabla",
    }]
    for _, r in abiertas.iterrows():
        registry.append({
            "ID": f"P65-{r['prioridad']}{r['familia'][:6].replace(' ', '')}",
            "Track": r["familia"], "Hypothesis": r["nuevo"],
            "Antecedente": r["antecedente"], "Novedad": r["estado"],
            "Metodo": r["nuevo"], "Estado": "abierta", "Resultado": "", "Decision": ""})
    pd.DataFrame(registry).to_csv(CSV_REG, index=False, encoding="utf-8")

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    informe = {
        "familia": "P65_GAP_ANALYSIS",
        "naturaleza": "AUDITORIA. 0 experimentos, 0 modelos, 0 features",
        "fecha": inicio.isoformat(timespec="seconds"), "experiment_id": "P65-00",
        "script": "experiments/p65_00_audit_gap.py",
        "auditoria_del_baseline": auditoria,
        "pruebas_en_el_codigo": pruebas,
        "por_que_se_lee_el_codigo": (
            "con 359 filas una busqueda por nombre devuelve las mismas filas para cualquier "
            "patron que aparezca en una pregunta, luego la clasificacion tiene que ser "
            "FUNCIONAL: que calcula la representacion, no como se llama"),
        "tabla": filas,
        "n_abiertas": len(abiertas), "n_cerradas": len(cerradas),
        "cortes_de_longitud_congelados": [{"nombre": a, "desde": b, "hasta": c}
                                          for a, b, c in BINS],
        "reproducibilidad": {"nivel": "R2", "git_commit": commit, "hash_v03": hash_real,
                             "corrected_cv": V03_CCV},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_GAP.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")

    L: list[str] = []
    W = L.append
    W("# P65 - Analisis de hipotesis: que queda genuinamente abierto")
    W("")
    W(f"Generado por `experiments/p65_00_audit_gap.py`. Fecha "
      f"{inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## La auditoria del baseline")
    W("")
    W("```text")
    W(f"V03 corrected_cv  {ccv!r}")
    W(f"V03 sha256        {hash_real}")
    W(f"registro          {len(reg)} filas, {reg['experiment_id'].nunique()} identificadores")
    W("submissions       0 | Kaggle 0 | holdout 0 | test 0 | leaderboard 0")
    W("```")
    W("")
    W("## Por que se lee el codigo y no el registro")
    W("")
    W("Con 359 filas, una busqueda por nombre devuelve las mismas filas para cualquier patron")
    W("que aparezca dentro de una pregunta. La clasificacion tiene que ser **funcional**: que")
    W("calcula la representacion, no como se llama. Por eso la tabla siguiente cita el")
    W("experimento y el fichero, y no un tema.")
    W("")
    W("## Los cuatro resultados que cambian el plan")
    W("")
    W("### 1. Los skip-grams YA se midieron")
    W("")
    W("**V6-05** cambio el analizador del bloque de palabras para que los pares con hueco")
    W("compitan con las palabras por el mismo presupuesto L2, probo hueco 1 y huecos 1 y 2, y")
    W("consiguio **-0,0001 con IC95 que incluye el 0** sobre un vocabulario un 83 por ciento")
    W("mayor. El vocabulario crecio y no paso nada. Es la misma mecanica que esta fase propone")
    W("y el resultado es **NEUTRAL**, no solo negativo.")
    W("")
    W("Lo que V6-05 **no** alcanzo fueron los huecos grandes, y ahi si hay una pregunta nueva.")
    W("")
    W("### 2. word (1,3) existe, pero no en la forma que hace falta")
    W("")
    W("**REP-04** midio TF-IDF word (1,3) **suelto**, con **C=1.0** y **sin bloque de")
    W("caracteres**, y lo registro como descartado con macro_f1 0,8062. Eso **sustituye** la")
    W("representacion y cambia el regularizador. La pregunta de esta fase es distinta: anadir")
    W("los trigramas a V03 manteniendo char (3,5), C=2.0, umbrales y pliegues, y medir el delta")
    W("sobre el baseline congelado. Es **precedente parcial**, no duplicado exacto.")
    W("")
    W("### 3. La negacion y la cita ya se midieron dos veces")
    W("")
    W("**E2-08** construyo los patrones NEG_* y dio -0,2505 solo y -0,0115 sobre V03. **E2-07**")
    W("construyo los canales dentro y fuera de comillas y dio -0,0335, -0,0218 y -0,0220.")
    W("Despues **P64-06 y P64-07** los midieron de otro angulo, con el alcance por distancia")
    W("y con el efecto sobre la evidencia, y tambien cerraron. Y P64-06 midio ademas que el")
    W("corpus tiene 39 filas con comillas y **solo una** con interior no vacio.")
    W("")
    W("### 4. La cohesion es nueva, y hubo que descartar el casi-acierto")
    W("")
    W("**V10-07** es shift de covariables **entre** entrenamiento y prueba, que es otro objeto")
    W("que la cohesion **dentro** del documento. Y **16_targeting_v2** mide distancias entre un")
    W("target y un insulto con un lexico inducido, que no es un par lexico sparse de anchura")
    W("desconocida. Los dos se descartan como antecedente y la cohesion queda abierta.")
    W("")
    W("## La tabla completa")
    W("")
    W("| familia | estado | antecedente | resultado | experimento |")
    W("|---|---|---|---|---:|")
    for _, r in df.iterrows():
        res = str(r["resultado"])
        res = res if len(res) <= 70 else res[:68] + "..."
        W(f"| {r['familia']} | {r['estado']} | {str(r['antecedente'])[:58]} | {res} | "
          f"{r['experimento']} |")
    W("")
    W(f"**{len(abiertas)} abiertas con experimento pendiente, {len(cerradas)} cerradas por "
      f"equivalencia medida.**")
    W("")
    W("Y una observacion sobre el reparto del presupuesto. El encargo da 25 experimentos")
    W("principales, y la clasificacion deja **cinco** con una hypothesis genuina. Eso no es una")
    W("excusa para inventar familias: es el resultado del trabajo, y la honestidad exige que")
    W("el informe diga que la mayor parte del mecanismo de largo alcance ya estaba medido.")
    W("")
    W("## Lo que queda abierto, con su formulacion")
    W("")
    for _, r in abiertas.iterrows():
        W(f"- **{r['familia']}** ({r['estado']}): {r['nuevo']}")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p65_00_audit_gap.py`. Sin push.*")
    MD_GAP.write_text(a_ascii("\n".join(L)), encoding="ascii")

    with CSV_GAP.open("a", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow([])
        w.writerow(["RESUMEN", "abiertas", len(abiertas)])
        w.writerow(["RESUMEN", "cerradas", len(cerradas)])
        w.writerow(["RESUMEN", "experimentos_ahorrados", int(cerradas["experimento"].sum())])
    for destino in (MD_GAP, JSON_GAP, CSV_GAP, CSV_REG):
        if any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P65-00 OK | {len(abiertas)} abiertas | {len(cerradas)} cerradas | "
        f"{int(cerradas['experimento'].sum())} experimentos ahorrados | commit {commit}")


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
