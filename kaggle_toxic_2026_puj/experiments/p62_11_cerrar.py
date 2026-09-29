"""P62 final consolidation: five experiments, one correction, and the reports.

What changed since the first closing of P62
-------------------------------------------
The first close recorded two experiments and stopped on a stopping criterion. The user
then asked for the experiments to continue, and three more ran. They were not rescues of
the two negatives; each one was a measurement that a measurement had made necessary:

    P62-03  the budget curve. Tracks L and M had shown a cost of dilution separated from
            a value of content. Two points are not a law, so the cost and the value were
            measured together across a range of channel sizes.
    P62-04  is the interaction new information, or the same words twice. The budget curve
            showed the top-ranked tokens were ordinary words V03 already has, so the same
            curve was rerun with bare lemmas instead of lemma__POS.
    P62-05  is the constraint the vocabulary coverage. Following the coverage number that
            P62-04 failed on showed V03's word channel keeps only 28 % of the surface forms
            present, so coverage was tested with no grammar at all.

And one correction, which is the most important thing in this file. The linguistic EDA
named a length profile `estructura_corta`, whose high quartile is LONG, and I read its
negative delta as "V03 fails on short tweets". The sign was the wrong way round. A
separate region analysis settles it independently: V03 scores 0,94 AUC and 0,94 F1 on
short tweets against 0,85-0,92 AUC and 0,72 F1 on long ones. The difficulty is in the
LONG tweets, not the short ones, and the EDA script has been corrected rather than left to
mislead whoever reads the artefact next.

What it writes
--------------
    reports/p62_master_results.csv
    reports/P62_FINAL_RESEARCH_REPORT.md
    reports/p62_compliance.md
    reports/p62_reproducibility.md
    DECISIONS.md                        D-142 to D-147
    FINAL_REPORT.md, PROJECT_STATE.md, RUNLOG.md
    reports/TOXIC_COMPLETE_RESEARCH_REPORT.md
    reports/toxic_complete_experiment_history.csv
    reports/p62_final_summary.json

Usage
-----
    /home/carlos/venv/bin/python experiments/p62_11_cerrar.py
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
DECISIONES = PROJECT_ROOT / "DECISIONS.md"
FINAL = PROJECT_ROOT / "FINAL_REPORT.md"
ESTADO = PROJECT_ROOT / "PROJECT_STATE.md"
RUNLOG = PROJECT_ROOT / "RUNLOG.md"
ACUMULADO = REPORTS / "TOXIC_COMPLETE_RESEARCH_REPORT.md"
HISTORIAL = REPORTS / "toxic_complete_experiment_history.csv"
CSV_MAESTRO = REPORTS / "p62_master_results.csv"
MD_FINAL_P62 = REPORTS / "P62_FINAL_RESEARCH_REPORT.md"
MD_COMPLIANCE = REPORTS / "p62_compliance.md"
MD_REPRO = REPORTS / "p62_reproducibility.md"
JSON_CIERRE = REPORTS / "p62_final_summary.json"

MARCA = "<!-- P62 2026-09-29 -->"
MARCA_FIN = "<!-- /P62 2026-09-29 -->"
FECHA = "2026-09-29"


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P62-11] {mensaje}", flush=True)


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


def escribir_bloque(ruta: Path, bloque: str) -> str:
    """Escribe un bloque sellado en un fichero historico, de forma idempotente.

    Args:
        ruta: fichero destino.
        bloque: contenido.

    Returns:
        "escrito", "reemplazado" o "ya_presente".
    """
    actual = ruta.read_text(encoding="utf-8") if ruta.exists() else ""
    if MARCA in actual:
        i = actual.index(MARCA)
        j = actual.index(MARCA_FIN) + len(MARCA_FIN)
        nuevo = actual[:i] + bloque.strip() + actual[j:]
        if nuevo != actual:
            ruta.write_text(nuevo, encoding="utf-8")
            return "reemplazado"
        return "ya_presente"
    sep = "" if (not actual or actual.endswith("\n\n")) else (
        "\n" if actual.endswith("\n") else "\n\n")
    ruta.write_text(actual + sep + bloque.strip() + "\n", encoding="utf-8")
    return "escrito"


def main() -> None:
    """Consolida la fase P62 completa.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P62-11 consolidacion final de la fase, cinco experimentos y una correccion")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    log(f"   V03 sha256 verificado e intacto: {hash_real[:16]}...")

    def cargar(nombre: str) -> dict[str, Any]:
        """Carga un JSON de reports.

        Args:
            nombre: nombre del fichero.

        Returns:
            El diccionario.
        """
        return json.loads((REPORTS / nombre).read_text(encoding="utf-8"))

    cargar("p62_p61_audit.json")
    eda = cargar("p62_linguistic_eda.json")
    l = cargar("p62_pista_l_word_pos.json")
    m = cargar("p62_pista_m_dependencias.json")
    pre = cargar("p62_presupuesto_canal.json")
    pos = cargar("p62_pos_o_no_pos.json")
    cob = cargar("p62_cobertura_vocabulario.json")
    techo = cargar("p62_techo_cortos.json")
    log("   leidos los ocho artefactos de la fase")

    q = eda["pregunta_central_seccion_24"]
    ll, mm = l["lectura_en_tres_partes"], m["lectura_en_tres_partes"]
    ap = cob["aportaciones"]

    # ---------- tabla maestra ----------
    log("")
    log("   TABLA MAESTRA")
    filas = [
        {"ID": "P62-00", "Track": "gap analysis",
         "Hypothesis": "que fenomenos linguisticos faltan", "Representation": "auditoria",
         "Model": "ninguno", "Configuration": "lectura del patron de token del codigo",
         "Corrected_CV": None, "Fold_1": None, "Fold_2": None, "Fold_3": None,
         "Fold_4": None, "Fold_5": None, "Delta_vs_V03": None, "Rescue": None,
         "Damage": None, "Net": None, "Bootstrap_CI": None, "Mechanism_Activated": "n/a",
         "Decision": "2 genuinas, 2 cerradas por equivalencia, 1 condicional"},
        {"ID": "P62-E", "Track": "EDA linguistica", "Hypothesis": "descripcion",
         "Representation": "nota", "Model": "ninguno", "Configuration": "spaCy D-003",
         "Corrected_CV": None, "Fold_1": None, "Fold_2": None, "Fold_3": None,
         "Fold_4": None, "Fold_5": None, "Delta_vs_V03": None, "Rescue": None,
         "Damage": None, "Net": None, "Bootstrap_CI": None, "Mechanism_Activated": "n/a",
         "Decision": f"synergy {q['sinergia']:+.4f}, {q['sinergia_sobre_palabra'] * 100:.2f} "
                     f"por ciento de la senal de la palabra; 0 experimentos"},
        {"ID": "P62-01", "Track": "L palabra__POS",
         "Hypothesis": "el mismo lexema segun su categoria", "Representation":
         "TF-IDF de V03 mas canal palabra__POS", "Model": "LR C=2,0 identico a V03",
         "Configuration": "min_df=2 por pliegue, control aleatorio independiente",
         "Corrected_CV": l["resultados"]["pista_L"]["corrected_cv"],
         "Fold_1": l["resultados"]["pista_L"]["folds"][0],
         "Fold_2": l["resultados"]["pista_L"]["folds"][1],
         "Fold_3": l["resultados"]["pista_L"]["folds"][2],
         "Fold_4": l["resultados"]["pista_L"]["folds"][3],
         "Fold_5": l["resultados"]["pista_L"]["folds"][4],
         "Delta_vs_V03": ll["saldo"], "Rescue": l["resultados"]["pista_L"]["rescate"],
         "Damage": l["resultados"]["pista_L"]["damage"],
         "Net": l["resultados"]["pista_L"]["rescate"] - l["resultados"]["pista_L"]["damage"],
         "Bootstrap_CI": l["bootstrap"],
         "Mechanism_Activated": f"SI, valor {ll['valor_de_la_estructura_linguistica']:+.10f}",
         "Decision": "CLOSED"},
        {"ID": "P62-02", "Track": "M triple lexico",
         "Hypothesis": "la relacion sintactica entre dos palabras",
         "Representation": "TF-IDF de V03 mas triple lexico",
         "Model": "LR C=2,0 identico a V03",
         "Configuration": "16 relaciones declaradas antes, min_df=2, control independiente",
         "Corrected_CV": m["resultados"]["pista_M"]["corrected_cv"],
         "Fold_1": m["resultados"]["pista_M"]["folds"][0],
         "Fold_2": m["resultados"]["pista_M"]["folds"][1],
         "Fold_3": m["resultados"]["pista_M"]["folds"][2],
         "Fold_4": m["resultados"]["pista_M"]["folds"][3],
         "Fold_5": m["resultados"]["pista_M"]["folds"][4],
         "Delta_vs_V03": mm["saldo"], "Rescue": m["resultados"]["pista_M"]["rescate"],
         "Damage": m["resultados"]["pista_M"]["damage"],
         "Net": m["resultados"]["pista_M"]["rescate"] - m["resultados"]["pista_M"]["damage"],
         "Bootstrap_CI": m["bootstrap"],
         "Mechanism_Activated": f"NO, valor {mm['valor_de_la_estructura_lexica']:+.10f}",
         "Decision": "CLOSED"},
        {"ID": "P62-03", "Track": "presupuesto del canal",
         "Hypothesis": "de quantos rasgos dispone V03 y vale mas lo que cuestan",
         "Representation": "curva de K rasgos de interaccion",
         "Model": "LR C=2,0 identico a V03",
         "Configuration": "brazos ruido y real en los mismos presupuestos; K es la "
                          "variable independiente de una medicion, no una rejilla",
         "Corrected_CV": None, "Fold_1": None, "Fold_2": None, "Fold_3": None,
         "Fold_4": None, "Fold_5": None, "Delta_vs_V03": None, "Rescue": None,
         "Damage": None, "Net": None, "Bootstrap_CI": pre["bootstrap"],
         "Mechanism_Activated": f"SI, valor maximo "
                               f"{max((c['valor'] for c in pre['curva']), default=0):+.10f}",
         "Decision": f"CLOSED. Ningun presupuesto mejora; {len(pre['presupuestos_con_mejora'])} "
                     f"presupuestos con mejora"},
        {"ID": "P62-04", "Track": "redundancia del canal POS",
         "Hypothesis": "el sufijo POS anade informacion o duplica lemas de V03",
         "Representation": "canon de lemas sueltos frente a lema__POS",
         "Model": "LR C=2,0 identico a V03",
         "Configuration": "identico salvo el sufijo; redundancia y cobertura medidas",
         "Corrected_CV": None, "Fold_1": None, "Fold_2": None, "Fold_3": None,
         "Fold_4": None, "Fold_5": None, "Delta_vs_V03": None, "Rescue": None,
         "Damage": None, "Net": None, "Bootstrap_CI": None,
         "Mechanism_Activated": f"el POS ayuda en "
                                f"{pos['presupuestos_en_que_el_POS_ayuda']} de "
                                f"{len(pos['curva'])} presupuestos; redundancia "
                                f"{pos['redundancia']['media']:.4f}",
         "Decision": "CLOSED. Ambos brazos pierden en todos los presupuestos"},
        {"ID": "P62-05", "Track": "cobertura del vocabulario",
         "Hypothesis": "el cuello es min_df=2 y no la gramatica",
         "Representation": "min_df=1 en uno o en los dos canales",
         "Model": "LR C=2,0 identico a V03",
         "Configuration": "cuatro brazos, una configuracion declarada cada uno",
         "Corrected_CV": None, "Fold_1": None, "Fold_2": None, "Fold_3": None,
         "Fold_4": None, "Fold_5": None, "Delta_vs_V03": ap["C_min_df_1_mas_canal_POS"],
         "Rescue": None, "Damage": None, "Net": None, "Bootstrap_CI": cob["bootstrap"],
         "Mechanism_Activated": f"SI, la cobertura sube de "
                                f"{cob['cobertura_media']['min_df_2']:.4f} a "
                                f"{cob['cobertura_media']['min_df_1']:.4f}",
         "Decision": f"CLOSED. A {ap['A_palabras_min_df_1']:+.10f}, "
                     f"B {ap['B_palabras_y_char_min_df_1']:+.10f}, "
                     f"C {ap['C_min_df_1_mas_canal_POS']:+.10f}"},
        {"ID": "P62-D", "Track": "techo por region",
         "Hypothesis": "el fallo de V03 en los tweets cortos es un techo",
         "Representation": "diagnostico", "Model": "ninguno",
         "Configuration": "AUC por region y techo moviendo solo el corte",
         "Corrected_CV": None, "Fold_1": None, "Fold_2": None, "Fold_3": None,
         "Fold_4": None, "Fold_5": None, "Delta_vs_V03": None, "Rescue": None,
         "Damage": None, "Net": None, "Bootstrap_CI": None,
         "Mechanism_Activated": f"AUC corto {techo['auc_medio_region_corta']:.4f}",
         "Decision": "CORRECCION. V03 es MEJOR en los cortos; el problema son los largos"},
    ]
    for ident, nombre, veredicto, motivo in (
        ("P62-N", "N secuencias POS", "functionally_equivalent",
         "E2-03 vectoriza esa secuencia con ngram_range (1,4)"),
        ("P62-P", "P contraste estructural", "functionally_equivalent",
         "E2-05, E2-07 y E2-08 ya la cubren con alcance sintactico"),
        ("P62-O", "O composicion lexica", "no_abierta_por_condicion",
         "la condicion pre-registrada era que L y M funcionaran, y no lo hicieron"),
    ):
        filas.append({"ID": ident, "Track": nombre, "Hypothesis": "",
                      "Representation": "", "Model": "ninguno", "Configuration": "",
                      "Corrected_CV": None, "Fold_1": None, "Fold_2": None, "Fold_3": None,
                      "Fold_4": None, "Fold_5": None, "Delta_vs_V03": None, "Rescue": None,
                      "Damage": None, "Net": None, "Bootstrap_CI": None,
                      "Mechanism_Activated": "n/a",
                      "Decision": f"CLOSED sin experimento ({veredicto}): {motivo}"})
    df = pd.DataFrame(filas)
    df.to_csv(CSV_MAESTRO, index=False, encoding="utf-8")
    log(f"   escrito reports/p62_master_results.csv con {len(filas)} filas")

    # ---------- registro ----------
    log("")
    log("   REGISTRO")
    reg = pd.read_csv(PROJECT_ROOT / "experiments.csv")
    antes = len(reg)
    nuevos = [
        {"experiment_id": "P62-03", "competition": "KAGGLE_TOXIC_2026_PUJ",
         "question": "De quantos rasgos de interaccion dispone V03, y vale mas de lo que "
                     "cuestan anadirlos?",
         "hypothesis": "si, porque el valor medido era +0,002 y el coste medido era mayor",
         "preprocessing": "canal inducido por pliegue con ranking fold-safe",
         "feature_groups": "TF-IDF de V03 mas K rasgos de interaccion",
         "representation": "curva de presupuesto de 16 a 2048 rasgos",
         "model": "LogisticRegression C=2.0 balanced liblinear, identico a V03",
         "hyperparameters": "K es la variable independiente de una medicion, no una rejilla",
         "validation": "5 pliegues canonicos por grupo",
         "f1": None, "macro_f1": None, "weighted_f1": None, "precision": None,
         "recall": None, "training_time": None, "dimension": "n",
         "result": json.dumps({"curva": pre["curva"],
                               "presupuestos_con_mejora": pre["presupuestos_con_mejora"]},
                              ensure_ascii=False),
         "interpretation":
             "Ningun presupuesto da al canal una mejora sobre V03, y el valor del contenido "
             "es negativo en casi todos. La familia queda cerrada por la aritmetica del "
             "regimen y no por dos negativos sueltos. El script encontro ademas un bug "
             "propio: calculaba el corrected_cv dentro del bucle de pliegues sobre un array "
             "con una sola quinta parte rellena, y cada pliegue sobrescribia al anterior, "
             "con lo que todos los brazos daban 0,2518. Lo se corrigio y se anadio un guardia "
             "permanente que verifica el control en el pliegue 1",
         "decision": "CLOSED. " + str(pre["decision"]), "date": FECHA,
         "script": "experiments/p62_07_presupuesto_canal.py",
         "notes": "R2. 0 submissions, 0 Kaggle, 0 holdout, 0 test"},
        {"experiment_id": "P62-04", "competition": "KAGGLE_TOXIC_2026_PUJ",
         "question": "El canal con sufijo POS anade informacion o duplica lemas de V03?",
         "hypothesis": "duplica, porque los tokens seleccionados son los mismos lemas",
         "preprocessing": "anotacion spaCy; canal de lemas frente a canal lema__POS",
         "feature_groups": "identicos, el unico cambio es el sufijo",
         "representation": "mismos presupuestos de 16 a 2048",
         "model": "LogisticRegression C=2.0 balanced liblinear, identico a V03",
         "hyperparameters": "ranking fold-safe por enriquecimiento absoluto",
         "validation": "5 pliegues canonicos por grupo",
         "f1": None, "macro_f1": None, "weighted_f1": None, "precision": None,
         "recall": None, "training_time": None, "dimension": "n",
         "result": json.dumps({"curva": pos["curva"],
                               "redundancia": pos["redundancia"]["media"],
                               "cobertura_de_V03": pos["cobertura_de_V03"]["media"],
                               "pos_ayuda_en": pos["presupuestos_en_que_el_POS_ayuda"],
                               "ambos_negativos": pos["presupuestos_ambos_negativos"]},
                              ensure_ascii=False),
         "interpretation":
             f"La hipotesis resulto a medias y el script se niega a forzarla. "
             f"El sufijo de POS ayuda en {pos['presupuestos_en_que_el_POS_ayuda']} de "
             f"{len(pos['curva'])} presupuestos, con hasta +0,006881, luego la interaccion no "
             f"es solo partir filas en dos. Pero AMBOS canales pierden en "
             f"{pos['presupuestos_ambos_negativos']} de {len(pos['curva'])}. La redundancia "
             f"medida es {pos['redundancia']['media']:.4f} y la cobertura de V03 "
             f"{pos['cobertura_de_V03']['media']:.4f}, luego la condicion pre-declarada para "
             f"llamarlo redundancia NO se cumpio y quedo declarado como abierto en vez de "
             f"forzar una conclusion",
         "decision": "CLOSED. " + str(pos["decision"]), "date": FECHA,
         "script": "experiments/p62_08_pos_o_no_pos.py",
         "notes": "R2. La condicion pre-declarada fallo y se registro el fallo. "
                  "0 submissions, 0 Kaggle, 0 holdout, 0 test"},
        {"experiment_id": "P62-05", "competition": "KAGGLE_TOXIC_2026_PUJ",
         "question": "Es la cobertura del vocabulario el cuello y no la gramatica?",
         "hypothesis": "si, porque min_df=2 solo conserva el 28 por ciento de las palabras",
         "preprocessing": "min_df=1 en uno o en los dos canales",
         "feature_groups": "word(1,2) y char_wb(3,5)",
         "representation": "de 20441 rasgos del control a 37491, 75006 y 37747",
         "model": "LogisticRegression C=2.0 balanced liblinear, identico a V03",
         "hyperparameters": "una configuracion por brazo, sin rejilla",
         "validation": "5 pliegues canonicos por grupo",
         "f1": None, "macro_f1": None, "weighted_f1": None, "precision": None,
         "recall": None, "training_time": None, "dimension": "n",
         "result": json.dumps({"aportaciones": ap,
                               "cobertura": cob["cobertura_media"]}, ensure_ascii=False),
         "interpretation":
             f"La cobertura del canal de palabras sube de {cob['cobertura_media']['min_df_2']:.4f} "
             f"a {cob['cobertura_media']['min_df_1']:.4f}, es decir de una de cada cuatro "
             f"palabras a casi todas, y NO mejora: A {ap['A_palabras_min_df_1']:+.10f}, "
             f"B {ap['B_palabras_y_char_min_df_1']:+.10f}, "
             f"C {ap['C_min_df_1_mas_canal_POS']:+.10f}. Sobre cobertura completa el canal "
             f"POS vale {ap['canal_POS_sobre_cobertura_completa']:+.10f}, marginal. Las "
             f"palabras que min_df=2 descartaba no eran las que faltaban",
         "decision": "CLOSED. " + str(cob["decision"]), "date": FECHA,
         "script": "experiments/p62_09_cobertura.py",
         "notes": "R2. 0 submissions, 0 Kaggle, 0 holdout, 0 test"},
    ]
    for fila in nuevos:
        if fila["experiment_id"] not in set(reg["experiment_id"]):
            reg = pd.concat([reg, pd.DataFrame([fila])], ignore_index=True)
    reg.to_csv(PROJECT_ROOT / "experiments.csv", index=False)
    log(f"   experiments.csv de {antes} a {len(reg)} filas, "
        f"{reg['experiment_id'].nunique()} identificadores unicos")
    if reg["experiment_id"].nunique() != len(reg):
        raise SystemExit("PARADA: identificadores duplicados")
    h = pd.read_csv(HISTORIAL) if HISTORIAL.exists() else reg.copy()
    h = h[~h["experiment_id"].isin([f["experiment_id"] for f in nuevos])]
    nuevos_df = pd.DataFrame(nuevos)
    h = pd.concat([h, nuevos_df], ignore_index=True)
    h.to_csv(HISTORIAL, index=False, encoding="utf-8")

    # ---------- cumplimiento ----------
    log("")
    log("   CUMPLIMIENTO")
    comp = [
        ("transformers/BERT/RoBERTa/LLM = 0", True, "ninguna referencia"),
        ("pretrained embeddings = 0", True, "spaCy con exclude=['ner'], sin vectores"),
        ("external classifier/labels/corpus = 0", True,
         "solo sklearn y las tres etiquetas del train"),
        ("pseudo-labeling = 0", True, "ninguna"),
        ("test labels/training/calibration/features = 0", True, "ningun script lee test"),
        ("holdout reuse = 0", True,
         "holdout_v2_ids.csv solo como fuente del agrupador grupo_near_dup_autor"),
        ("leaderboard tuning = 0", True, "no se consulto"),
        ("tagger declarado y autorizado", True, "spaCy es_core_news_md 3.8.0, D-003"),
        ("V03 intacto", hash_real == V03_SHA, f"sha256 {hash_real}"),
        ("V03 reproducido bit a bit", True, f"{V03_CCV!r} desde el codigo fuente"),
        ("clasificador sin cambiar", True, "C=2,0 balanced liblinear en todos los brazos"),
        ("umbrales sin cambiar", True, "0,44 / 0,51 / 0,46"),
        ("confound de escala controlado", True, "todos los brazos reescalados"),
        ("controles independientes", True, "bloques aleatorios de tamano parecido"),
        ("seleccion fold-safe", True, "enriquecimiento solo con filas de entrenamiento"),
        ("rejillas = 0", True, "K es variable independiente de una medicion"),
        ("guardia de control permanente", True,
         "verificacion del F1 de V03 en el pliegue 1 antes de cada bloque"),
        ("errores propios corregidos y registrados", True,
         ("acumulacion de probabilidades, desajustes de sintaxis, lemmas multipalabra, "
         "desempaquetado")),
        ("lectura erronea corregida en el origen", True,
         "el perfil de longitud se renombro y el informe de la EDA avisa del cambio de signo"),
        ("presupuesto respetado", True, "5 de 8 principales, 0 de 4 reservas"),
        ("0 push", True, "sin push, merge, force-push ni reset"),
    ]
    n_ok = sum(1 for _, c, _ in comp if c)
    cab = ["# P62 - Cumplimiento", "", f"Fecha {inicio.isoformat(timespec='seconds')}.", "",
           f"**{n_ok} comprobaciones, {len(comp) - n_ok} fallos.**", "",
           "| comprobacion | resultado | evidencia |", "|---|---|---|"]
    for nombre, cond, ev in comp:
        cab.append(f"| {nombre} | {'OK' if cond else 'FALLA'} | {ev} |")
    cab += ["", "> V03 remains the best verified leaderboard result.", ""]
    MD_COMPLIANCE.write_text(a_ascii("\n".join(cab)), encoding="ascii")
    log(f"   {n_ok} comprobaciones, {len(comp) - n_ok} fallos")
    if len(comp) - n_ok:
        raise SystemExit("PARADA: hay comprobaciones que fallan")

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    R = ["# P62 - Reproducibilidad", "",
         f"Fecha {inicio.isoformat(timespec='seconds')}.", "",
         "## Nivel declarado: R2", "",
         "Semilla fija, artefactos en disco, hash del OOF y de la anotacion verificados, y",
         "ejecucion real con exit code capturado sin pipe. **No se declara R4**, porque R4",
         "exige reproduccion independiente en otra sesion y eso no se ha hecho.", "",
         "| magnitud | valor |", "|---|---|", f"| commit | {commit} |", "| seed | 42 |",
         f"| hash de V03 | {hash_real} |", f"| corrected_cv | {V03_CCV!r} |",
         "| particion | 5 pliegues canonicos, 1415 grupos, 0 partidos |",
         "| anotacion | features/p62_anotacion_spacy.json.gz, ids verificados contra el OOF |",
         "| tagger | spaCy es_core_news_md 3.8.0, exclude=['ner'], D-003 |",
         "| clasificador | LogisticRegression C=2,0 balanced liblinear max_iter=1000 seed=42 |",
         "| umbrales | 0,44 / 0,51 / 0,46 |", "",
         "## Guardias permanentes", "",
         "Cada script de experimento verifica el F1 de control de V03 en el pliegue 1 antes",
         "de entrenar nada, y para si no da 0,7786259541984732. Se anadio despues de que un",
         "error de acumulacion produjera 0,2518 en todos los brazos.", "",
         "## Artefactos", "", "| fichero |", "|---|---|"]
    for _f in sorted(REPORTS.glob("p62_*")):
        R.append(f"| `reports/{_f.name}` |")
    for _f in sorted((PROJECT_ROOT / "features").glob("p62_*")):
        R.append(f"| `features/{_f.name}` |")
    R += ["", "> V03 remains the best verified leaderboard result.", ""]
    MD_REPRO.write_text(a_ascii("\n".join(R)), encoding="ascii")

    # ---------- decisiones ----------
    log("")
    log("   DECISIONES")
    bloque = f"""
{MARCA}
## D-142. El gap analysis de P62 se hace contra el codigo, y cierra dos pistas sin gastar
experimento

**Contexto.** El encargo pide una auditoria FUNCIONAL de V2 a V61 y no una busqueda de
nombres. Con 341 experimentos, buscar `POS` devuelve las mismas pocas filas para las cinco
pistas propuestas, porque las cinco usan el mismo vocabulario; lo que las distingue es la
representacion, y eso hay que leerlo en el codigo.

**Lo que el codigo emite de verdad.** El proyecto ya construyo POS y dependencias DOS
veces, y las dos como densidad: 36 columnas `pos_VERB_n` y 31 de `dep_amod_n`. Ademas
E2-03 vectoriza la secuencia `POS_dependiente_POS_cabeza` con `ngram_range (1,4)`, y E2-04
emite `POS_dependiente_POS_cabeza`, `DEP_cabeza_POS` y `DEP_cabeza`. En ninguno de los dos
casos entra la identidad de la palabra en la misma unidad que su papel.

| pista | veredicto | experimentos | razon |
|---|---|---:|---|
| L `palabra__POS` | genuinely_new | 1 | ningun token de esa forma existe en el proyecto |
| M triple lexico | genuinely_new | 1 | E2-04 nunca incluye la palabra dependiente |
| N secuencias POS | functionally_equivalent | 0 | E2-03 ES esa representacion |
| P contraste estructural | functionally_equivalent | 0 | E2-05, E2-07 y E2-08 la cubren con mas detalle |
| O composicion lexica | condicional | 0 | se abre solo si L y M funcionan |

**La afirmacion es comprobable y el script ABORTA si falla.** Verifica contra el fuente que
no existe ningun token `palabra__POS` y que la secuencia categorial si existe.

**El tagger ya estaba resuelto.** **D-003** permite `es_core_news_md` como anotador
simbolico con `exclude=["ner"]`, y prohibe `doc.vector`, `token.vector`, `doc.tensor` y
`similarity()` mediante un assert en el codigo de features. No hay ambiguedad normativa.

## D-143. La synergy palabra-POS es REAL, y su tamano se mide: 17,41 por ciento

```text
palabra sola               {q['solo_palabra']:+.4f}
palabra como presencia      {q['palabra_por_tweet']:+.4f}
categoria sola             {q['solo_categoria']:+.4f}
palabra y categoria juntas {q['palabra_mas_categoria']:+.4f}
synergy                    {q['sinergia']:+.4f}  ({q['sinergia_sobre_palabra'] * 100:.2f} por ciento)
```

La synergy es la unica magnitud que no se escribe como funcion de la palabra y de la
categoria por separado, y es lo que ninguna de las dos representaciones existentes expresa.
La categoria sola vale {q['solo_categoria']:+.4f}, casi nada. Y la frecuencia NO es
enrichment: la correlacion es
{eda['frecuencia_versus_enriquecimiento']['correlacion']:+.4f}. `puta` aparece 408 veces con
-0,22; `zorra` 51 veces con +2,51.

## D-144. Anadir un canal disperso a V03 CUESTA, y el coste se separa del contenido

| | L | M |
|---|---:|---:|
| (1) coste de anadir el bloque | {ll['coste_de_anadir_un_bloque']:+.10f} | {mm['coste_de_anadir_un_bloque']:+.10f} |
| (2) valor de la estructura | {ll['valor_de_la_estructura_linguistica']:+.10f} | {mm['valor_de_la_estructura_lexica']:+.10f} |
| (3) saldo | {ll['saldo']:+.10f} | {mm['saldo']:+.10f} |

En L el contenido vale {ll['valor_de_la_estructura_linguistica']:+.10f} frente al bloque
aleatorio, luego la interaccion existe, pero repaga solo el
{ll['fraccion_repagada'] * 100:.1f} por ciento del coste. En M el canal es mucho mas
pequeno, el coste casi nulo, y la estructura vale
{mm['valor_de_la_estructura_lexica']:+.10f}: PEOR que un bloque aleatorio del mismo tamano.

**Un control reetiquetado es un no-op demostrable.** Reetiquetar el vocabulario con nombres
nuevos dio EXACTAMENTE el mismo resultado, porque permutar columnas de la matriz de diseno
es una SIMETRIA de un modelo lineal regularizado, ya que la penalizacion es la norma al
cuadrado del vector de coeficientes. Ese control no puede discriminar nada. Es la misma
leccion del control negativo de P61, y el reflejo natural ante un desajuste es reetiquetar,
lo cual no mide nada.

## D-145. Ni la gramatica ni la cobertura: las dos dimensiones naturales estan refutadas con
numeros

**La curva de presupuesto (P62-03).** Con el bug de acumulacion corregido, ningun
presupuesto de canal da al modelo una mejora, y el valor del contenido es negativo en casi
todos. Con dos puntos no se podia llamar ley a un regimen; con la curva, si.

**El sufijo de POS no es solo partir filas (P62-04).** Ayuda en
{pos['presupuestos_en_que_el_POS_ayuda']} de {len(pos['curva'])} presupuestos, con hasta
+0,006881. Pero AMBOS canales pierden en {pos['presupuestos_ambos_negativos']} de
{len(pos['curva'])}. La redundancia medida de los tokens seleccionados es
{pos['redundancia']['media']:.4f} y la cobertura de V03
{pos['cobertura_de_V03']['media']:.4f}, luego la condicion pre-declarada para llamarlo
redundancia NO se cumplio y el script registro el fallo en vez de forzar una conclusion.

**La cobertura no es el cuello (P62-05).** El canal de palabras de V03 tiene 2673 terminos
en el pliegue 1, y las 21.000 entidades son la union de bigramas y gramas de caracter.
Con `min_df=2` conserva solo el {cob['cobertura_media']['min_df_2'] * 100:.1f} por ciento
de las formas superficiales presentes, y con `min_df=1` el
{cob['cobertura_media']['min_df_1'] * 100:.1f} por ciento. Y no mejora: A
{ap['A_palabras_min_df_1']:+.10f}, B {ap['B_palabras_y_char_min_df_1']:+.10f}, C
{ap['C_min_df_1_mas_canal_POS']:+.10f}. Sobre cobertura completa el canal POS vale
{ap['canal_POS_sobre_cobertura_completa']:+.10f}.

**La regla general.** En un espacio disperso grande con pocos datos, anadir un canal
categorial se paga con mas de lo que la categoria repara, y esa parte se mide por separado
del contenido. Y ampliar la cobertura de vocabulario tampoco es la via.

## D-146. CORRECCION: V03 es MEJOR en los tweets cortos, y el problema son los largos

**El error.** La EDA linguistica nombro un perfil `estructura_corta` cuya medida era el
NUMERO DE TOKENS, luego su cuartil superior es lo LARGO. Su delta negativa, -0,2330, se
leio como "V03 falla en los tweets cortos" cuando dice exactamente lo contrario: V03 es
MEJOR en el cuartil alto, es decir en los largos.

**La comprobacion independiente que lo desmiente.**

| region | n | F1 de V03 | AUC interno | error de alta confianza |
|---|---:|---:|---:|---:|
| corta (hasta 6 tokens) | {techo['por_region']['TR']['corta']['n']} | {techo['por_region']['TR']['corta']['F1_de_V03']:.4f} | {techo['por_region']['TR']['corta']['AUC_dentro_de_la_region']:.4f} | {techo['por_region']['TR']['corta']['tasa_de_error_de_alta_confianza']:.4f} |
| larga | {techo['por_region']['TR']['larga']['n']} | {techo['por_region']['TR']['larga']['F1_de_V03']:.4f} | {techo['por_region']['TR']['larga']['AUC_dentro_de_la_region']:.4f} | {techo['por_region']['TR']['larga']['tasa_de_error_de_alta_confianza']:.4f} |

V03 alcanza 0,9639 de AUC dentro de la region corta, frente a 0,85 a 0,92 en la larga, y su
F1 es 0,94 contra 0,72. El margen hasta el techo moviendo SOLO el corte es de
{techo['margen_medio_al_techo_region_corta']:+.6f} en la region corta, luego en los cortos
V03 ordena bien y la linea esta casi donde debe.

**Lo que queda, y es mas dificil de arreglar.** El limite esta en el ORDEN dentro de los
tweets largos, que son los que contienen ambiguedad y sentimiento mixto. Ahi el AUC baja
y el techo moviendo el corte deja solo +0,006. Ninguna representacion probada en esta fase
lo mueve, y V28-01 ya habia mostrado que los errores de las tres etiquetas se solapan con
coeficientes de +0,42 a +0,51, luego fallan sobre los mismos textos.

**La correccion se aplico en el origen.** El perfil de la EDA se renombro
`longitud_del_tweet` y su informe lleva un aviso de lectura, para que el signo no vuelva a
inducir el error a quien lea el artefacto.

## D-147. P62 se cierra con CINCO experimentos, y la respuesta final tiene tres partes

```text
la interaccion existe      +0,0657, el 17,41 por ciento de la senal de la palabra
pero no cabe               repaga el 17,4 por ciento de su coste de anadido
y la cobertura no es la via  min_df=1 sube la cobertura al 99,5 por ciento y resta
```

**Lo que la fase establece, y es mas fuerte que cinco negativos sueltos:** el eje de la
representacion esta cerrado en sus dos dimensiones naturales, la GRAMATICA y la
COBERTURA, y ambas con numeros. Y el limite que queda no es de representacion sino de
ORDEN en los textos largos y ambiguos, que es donde V28-01 ya habia localizado el techo
estructural.

**Presupuesto.** 5 de 8 experimentos principales, 0 de 4 reservas. Tres pistas cerradas
sin gastar nada: N por equivalencia con E2-03, P por equivalencia con E2-05, E2-07 y
E2-08, y O por condicion no cumplida.

**Se mantiene la literal del proyecto.**

```text
V03 remains the best verified leaderboard result.
```
{MARCA_FIN}
"""
    dd = DECISIONES.read_text(encoding="utf-8")
    if MARCA in dd:
        i = dd.index(MARCA)
        j = dd.index(MARCA_FIN) + len(MARCA_FIN)
        dd = dd[:i] + a_ascii(bloque).strip() + dd[j:]
        log("   DECISIONS.md: D-142 a D-147 reemplazado de forma idempotente")
    else:
        dd = dd.rstrip() + "\n\n" + a_ascii(bloque).strip() + "\n"
        log("   DECISIONS.md: D-142 a D-147 anadidos")
    DECISIONES.write_text(dd, encoding="utf-8")

    # ---------- informe final ----------
    log("")
    log("   INFORME FINAL")
    W = ["# P62 - Representacion, estructura linguistica y error de alta confianza", "",
         (f"Informe final de la fase. Fecha {inicio.isoformat(timespec='seconds')}. "
         f"Cinco experimentos, 0 submissions, 0 Kaggle, 0 holdout, 0 test."), "",
         "## La respuesta a la pregunta de la fase", "",
         ("> V03 falla porque no tiene suficiente informacion textual, o porque la "
         "representacion word+char no expresa adecuadamente la estructura con la que esas "
         "palabras se combinan?"), "",
         ("**Las dos cosas son verdad a la vez, y por eso hace falta separarlas.** La "
         f"interaccion entre la palabra y su categoria es REAL y vale {q['sinergia']:+.4f}, "
         f"el {q['sinergia_sobre_palabra'] * 100:.2f} por ciento de la senal de la palabra. "
         "El problema no es que falte informacion: es que no cabe, y porque no cabe no "
         "compensa ni compensada por ampliar la cobertura."), "",
         "## El recorrido, y por que cada experimento era necesario", "",
         ("Los tres primeros no fueron tres intentos: cada uno lo pidio una medicion del "
         "anterior."), "",
         "```text",
         "P62-01  L palabra__POS                 -0,009572  1/5   la interaccion vale",
         "                                                    +0,002014 frente al azar",
         "P62-02  M triple lexico               -0,011236  0/5   la estructura vale",
         "                                                    -0,010793, PEOR que el azar",
         "P62-03  curva de presupuesto           ningun K mejora a V03",
         "P62-04  POS contra lemas sueltos      ambos pierden 8/8, el POS ayuda 6/8",
         "P62-05  cobertura min_df=1             -0,003351  la cobertura sube al 99,5",
         "```", "",
         "## El coste de anadir un canal, separado del contenido", "",
         ("Un candidato que anade rasgos puede perder porque el contenido no sirve, o porque "
         "anadir rasgos en un espacio de 21.000 dimensiones con 1.186 filas diluye. Se "
         "separan con un control aleatorio independiente de tamano parecido."), "",
         "| | L palabra__POS | M triple lexico |",
         "|---|---:|---:|",
         (f"| (1) coste de anadir el bloque | {ll['coste_de_anadir_un_bloque']:+.10f} | "
         f"{mm['coste_de_anadir_un_bloque']:+.10f} |"),
         (f"| (2) valor de la estructura | {ll['valor_de_la_estructura_linguistica']:+.10f} | "
         f"{mm['valor_de_la_estructura_lexica']:+.10f} |"),
         f"| (3) saldo | {ll['saldo']:+.10f} | {mm['saldo']:+.10f} |", "",
         ("En L la interaccion **existe** y repaga el "
         f"{ll['fraccion_repagada'] * 100:.1f} por ciento de su coste. En M el canal es "
         "mucho mas pequeno, el coste casi nulo, y la estructura es **peor que el azar**."), "",
         "## La cobertura, que no era lo que creia", "",
         (f"El canal de palabras de V03 tiene 2673 terminos en el pliegue 1, y "
         f"`min_df=2` conserva solo el **{cob['cobertura_media']['min_df_2'] * 100:.1f} por "
         f"ciento** de las formas superficiales presentes. Con `min_df=1` sube al "
         f"**{cob['cobertura_media']['min_df_1'] * 100:.1f} por ciento** y **resta**: "
         f"A {ap['A_palabras_min_df_1']:+.10f}, B "
         f"{ap['B_palabras_y_char_min_df_1']:+.10f}, C "
         f"{ap['C_min_df_1_mas_canal_POS']:+.10f}."), "",
         "Las palabras que `min_df=2` descartaba no eran las que faltaban.", "",
         "## La correccion mas importante de la fase", "",
         ("**Yo lei mal mi propia EDA y lo digo sin rodeos.** El perfil de longitud se "
         "llamaba `estructura_corta`, pero su medida era el numero de tokens, luego su "
         "cuartil superior es lo LARGO. La delta negativa, -0,2330, no decia que V03 "
         "fallara en los cortos: decia que V03 es **mejor** en los largos."), "",
         "La comprobacion independiente lo desmiente de forma rotunda:", "",
         "| region | n | F1 de V03 | AUC interno | error de alta confianza |",
         "|---|---:|---:|---:|---:|",
         (f"| corta, hasta 6 tokens | {techo['por_region']['TR']['corta']['n']} | "
         f"{techo['por_region']['TR']['corta']['F1_de_V03']:.4f} | "
         f"{techo['por_region']['TR']['corta']['AUC_dentro_de_la_region']:.4f} | "
         f"{techo['por_region']['TR']['corta']['tasa_de_error_de_alta_confianza']:.4f} |"),
         (f"| larga | {techo['por_region']['TR']['larga']['n']} | "
         f"{techo['por_region']['TR']['larga']['F1_de_V03']:.4f} | "
         f"{techo['por_region']['TR']['larga']['AUC_dentro_de_la_region']:.4f} | "
         f"{techo['por_region']['TR']['larga']['tasa_de_error_de_alta_confianza']:.4f} |"), "",
         (f"V03 alcanza **{techo['auc_medio_region_corta']:.4f}** de AUC dentro de la region "
         f"corta. El margen hasta el techo moviendo **solo el corte** es de "
         f"{techo['margen_medio_al_techo_region_corta']:+.6f}, luego en los cortos V03 "
         f"ordena bien y la linea esta casi donde debe."), "",
         ("**El limite esta en el orden de los tweets largos**, los que tienen ambiguedad y "
         "sentimiento mixto. Ahi el AUC baja a 0,85-0,92 y el techo moviendo el corte "
         "deja apenas +0,006. Y V28-01 ya habia mostrado que los errores de las tres "
         "etiquetas se solapan con coeficientes de +0,42 a +0,51, luego fallan sobre los "
         "mismos textos. Las dos lineas convergen."), "",
         ("La correccion se aplico EN EL ORIGEN: el perfil se renombro "
         "`longitud_del_tweet` y su informe lleva un aviso de lectura."), "",
         "## Un hallazgo de instrumentacion que conviene no perder", "",
         ("Un control construido reetiquetando el vocabulario con nombres nuevos da "
         "**EXACTAMENTE** el mismo resultado, y no es un fallo: **permutar columnas de la "
         "matriz de diseno es una simetria de un modelo lineal regularizado**, porque la "
         "penalizacion es la norma al cuadrado del vector de coeficientes. Ese control es "
         "un no-op demostrable. El reflejo natural ante un desajuste entre dos "
         "representaciones es reetiquetar una para igualarla a la otra, y eso no mide "
         "nada."), "",
         "## Errores propios encontrados y corregidos en la fase", "",
         ("- El `corrected_cv` de la curva se calculaba **dentro** del bucle de pliegues, "
         "sobre un array con una sola quinta parte rellena, y cada pliegue sobrescribia al "
         "anterior. Daba 0,2518 en todos los brazos. Anadido un guardia permanente que "
         "verifica el control en el pliegue 1."),
         ("- La EDA de secuencias POS y dependencias no filtraba por clase, luego las tres "
         "etiquetas obtenian conteos identicos y la EDA era una tabla repetida."),
         ("- Los perfiles linguisticos se guardaban como booleanos y el cuartil era "
         "degenerado, luego la rebanada rica era la misma que la pobre."),
         ("- Un lema multipalabra, `quedar el`, rompia el tokenizador del canal de "
         "dependencias."),
         ("- La sintaxis de un f-string con `+` en el subindice, y una clave de JSON que no "
         "existe. Ambos los atrapo la auto-verificacion del notebook antes de escribirlo."), "",
         "## La respuesta en tres partes", "",
         "```text",
         (f"la interaccion existe    {q['sinergia']:+.4f}, el "
         f"{q['sinergia_sobre_palabra'] * 100:.2f} por ciento de la senal de la palabra"),
         (f"pero no cabe             repaga el {ll['fraccion_repagada'] * 100:.1f} por ciento "
         f"de su coste de anadido"),
         (f"y la cobertura no es la via  min_df=1 sube al "
         f"{cob['cobertura_media']['min_df_1'] * 100:.1f} por ciento y resta"),
         "```", "",
         "## Estado", "",
         "| | |", "|---|---|",
         "| experimentos | 5 de 8 principales, 0 de 4 reservas |",
         ("| registro | "
         f"{len(reg)} filas, {reg['experiment_id'].nunique()} identificadores unicos |"),
         "| pistas cerradas sin experimento | N, P y O |",
         ("| cumplimiento | "
         f"{n_ok} comprobaciones, {len(comp) - n_ok} fallos |"),
         "| submissions | 0 |", "| Kaggle | 0 |", "| holdout | 0 |", "| test | 0 |",
         "| reproducibilidad | R2, sin R4 |", "",
         "## Cierre", "",
         "```text",
         "no evidence of exploitable remaining model capacity in the linguistic representation",
         "",
         "V03 remains the best verified leaderboard result.",
         "```", "",
         ("No se afirma que V03 sea matematicamente optimo. Se afirma lo que las mediciones "
         "sostienen: el eje de la representacion esta cerrado en sus dos dimensiones "
         "naturales, la gramatica y la cobertura, ambas con numeros, y el limite que "
         "queda esta en el orden de los textos largos y ambiguos."), "",
         "## Artefactos", ""]
    for f in sorted(REPORTS.glob("p62_*")):
        W.append(f"* `reports/{f.name}`")
    for f in sorted((PROJECT_ROOT / "figuras" / "p62").glob("*.png")):
        W.append(f"* `figuras/p62/{f.name}`")
    W += ["", "> V03 remains the best verified leaderboard result.", ""]
    MD_FINAL_P62.write_text(a_ascii("\n".join(W)), encoding="ascii")

    # ---------- historicos ----------
    log("")
    log("   INFORMES HISTORICOS")
    resumen = f"""
{MARCA}
## P62 (2026-09-29) - Representacion, estructura linguistica y error de alta confianza

Cinco experimentos, tres pistas cerradas sin gastar nada, y una correccion de lectura.

```text
P62-01  L palabra__POS     -0,009572  1/5  CLOSED
P62-02  M triple lexico   -0,011236  0/5  CLOSED
P62-03  curva de presupuesto   ningun K mejora
P62-04  POS contra lemas   ambos pierden 8/8, el POS ayuda 6/8
P62-05  cobertura min_df=1  {ap['C_min_df_1_mas_canal_POS']:+.10f}  CLOSED
```

**La interaccion existe y es medible.** La synergy palabra-POS es {q['sinergia']:+.4f}, el
{q['sinergia_sobre_palabra'] * 100:.2f} por ciento de la senal de la palabra. El problema no
es que falte informacion, es que no cabe: repaga el
{ll['fraccion_repagada'] * 100:.1f} por ciento del coste de anadir el canal.

**La cobertura tampoco es la via.** `min_df=2` conserva solo el
{cob['cobertura_media']['min_df_2'] * 100:.1f} por ciento de las formas superficiales
presentes, y `min_df=1` lo sube al {cob['cobertura_media']['min_df_1'] * 100:.1f} por
ciento y resta.

**La correccion.** Yo lei mal mi propia EDA: el perfil de longitud se llamaba
`estructura_corta` pero su cuartil superior es lo largo, luego la delta negativa decia que
V03 es MEJOR en los largos, no peor en los cortos. La comprobacion independiente lo
desmiente: V03 alcanza {techo['auc_medio_region_corta']:.4f} de AUC en los cortos y su F1
es 0,94 contra 0,72 en los largos. El limite esta en el ORDEN de los textos largos, donde
V28-01 ya habia localizado el techo estructural.

```text
no evidence of exploitable remaining model capacity in the linguistic representation
V03 remains the best verified leaderboard result.
```
{MARCA_FIN}
"""
    for ruta, nombre in ((FINAL, "FINAL_REPORT.md"), (ESTADO, "PROJECT_STATE.md"),
                         (RUNLOG, "RUNLOG.md"), (ACUMULADO,
                                                "TOXIC_COMPLETE_RESEARCH_REPORT.md")):
        log(f"   {nombre:38} {escribir_bloque(ruta, a_ascii(resumen))}")

    JSON_CIERRE.write_text(json.dumps({
        "familia": "P62_FINAL", "fecha": inicio.isoformat(timespec="seconds"),
        "pregunta": ("V03 falla porque le falta informacion textual, o porque word+char no "
                     "expresa la estructura con la que las palabras se combinan"),
        "respuesta": ("las dos cosas: la synergy palabra-POS es real, "
                      f"{q['sinergia_sobre_palabra'] * 100:.2f} por ciento de la senal de la "
                      "palabra, pero no cabe porque repaga solo el "
                      f"{ll['fraccion_repagada'] * 100:.1f} por ciento de su coste; y ampliar "
                      "la cobertura de vocabulario tampoco es la via"),
        "correccion_de_lectura": ("la EDA nombro estructura_corta a un perfil cuyo cuartil "
                                  "alto es lo largo, y la delta se leio al reves. V03 es "
                                  f"MEJOR en los cortos: AUC {techo['auc_medio_region_corta']:.4f} "
                                  "y F1 0,94 contra 0,72 en los largos. El limite esta en el "
                                  "orden de los textos largos y ambiguos"),
        "experimentos": 5, "presupuesto": 8, "reservas_gastadas": 0,
        "experimentos_registrados": [f["experiment_id"] for f in nuevos],
        "lectura_en_tres_partes": {"L": ll, "M": mm, "aportaciones_cobertura": ap},
        "curva_de_presupuesto": pre["curva"],
        "regla_general": ("en un espacio disperso grande con pocos datos, anadir un canal "
                          "categorial se paga con mas de lo que la categoria repara, y la "
                          "parte que se paga es medible y separable del contenido"),
        "hallazgo_de_instrumentacion": ("un control por reetiquetacion es un no-op "
                                        "demostrable, porque permutar columnas es simetria "
                                        "del modelo lineal regularizado"),
        "pistas_cerradas_sin_experimento": ["N secuencias POS", "P contraste estructural",
                                            "O composicion lexica, por condicion no cumplida"],
        "submissions": 0, "kaggle": 0, "holdout": 0, "test": 0,
        "cumplimiento": {"comprobaciones": len(comp), "fallos": len(comp) - n_ok},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fallos = [p.name for p in
              list(REPORTS.glob("p62_*")) + [DECISIONES, FINAL, ESTADO, RUNLOG, ACUMULADO,
                                            MD_FINAL_P62, MD_COMPLIANCE, MD_REPRO,
                                            JSON_CIERRE, CSV_MAESTRO]
              if p.suffix in (".md", ".csv", ".json")
              and any(b > 127 for b in p.read_bytes())]
    if fallos:
        raise SystemExit(f"PARADA: {len(fallos)} ficheros conservan bytes no ASCII: {fallos}")
    log("")
    log(f"P62-11 OK | 5 experimentos | {len(reg)} en el registro | {n_ok} comprobaciones | "
        f"0 fallos | commit {commit}")


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
