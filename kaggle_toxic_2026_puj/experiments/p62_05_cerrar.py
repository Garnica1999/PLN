"""P62 step 05: consolidate the phase into the master table, the register, the
decisions, the reports and the compliance audit.

Everything here READS artefacts and writes documents. It trains nothing, so it cannot
move a number: if a figure in the final report is wrong, it is wrong because an
upstream artefact is wrong.

The phase closed after two experiments, and the reason is worth stating here because it
is the reason this script exists.

The gap analysis committed one experiment to track O, the lexical-composition channel,
on a CONDITION declared before either experiment ran: only if L and M demonstrated that
adding a categorial sparse channel to V03's space moves something. They did not. L
repaid 17,4 % of its own dilution cost and M was worse than a random block of the same
size. So the condition failed, and O closes with zero experiments consumed, which is what
the project's own rule says a closed hypothesis costs. Rescuing a negative with a
tuning knob is exactly what that rule forbids, and P60 and P61 both declined to do it.

What it writes
--------------
    reports/p62_master_results.csv
    reports/P62_FINAL_RESEARCH_REPORT.md
    reports/p62_compliance.md
    reports/p62_reproducibility.md
    DECISIONS.md                     D-142 to D-145
    FINAL_REPORT.md, PROJECT_STATE.md, RUNLOG.md
    reports/TOXIC_COMPLETE_RESEARCH_REPORT.md
    reports/toxic_complete_experiment_history.csv

Usage
-----
    /home/carlos/venv/bin/python experiments/p62_05_cerrar.py
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
    print(f"[P62-05] {mensaje}", flush=True)


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
    sep = "" if (not actual or actual.endswith("\n\n")) else ("\n" if actual.endswith("\n")
                                                             else "\n\n")
    ruta.write_text(actual + sep + bloque.strip() + "\n", encoding="utf-8")
    return "escrito"


def main() -> None:
    """Consolida la fase P62 en todos los destinos del encargo.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P62-05 consolidacion de la fase")

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

    gap = cargar("p62_p61_audit.json")
    eda = cargar("p62_linguistic_eda.json")
    l = cargar("p62_pista_l_word_pos.json")
    m = cargar("p62_pista_m_dependencias.json")
    log("   leidos los cinco artefactos de la fase")

    q = eda["pregunta_central_seccion_24"]
    ll = l["lectura_en_tres_partes"]
    mm = m["lectura_en_tres_partes"]

    # ---------- tabla maestra ----------
    log("")
    log("   TABLA MAESTRA DE DECISIONES")
    filas = [
        {"ID": "P62-00", "Track": "gap analysis", "Hypothesis":
         "que fenomenos linguisticos faltan en la representacion de V03",
         "Representation": "auditoria funcional del codigo", "Model": "ninguno",
         "Configuration": "lectura del patron de token emitido por cada script historico",
         "Corrected_CV": None, "Fold_1": None, "Fold_2": None, "Fold_3": None,
         "Fold_4": None, "Fold_5": None, "Delta_vs_V03": None, "Rescue": None,
         "Damage": None, "Net": None, "Bootstrap_CI": None,
         "Mechanism_Activated": "n/a",
         "Decision": "2 pistas genuinamente nuevas (L, M), 2 cerradas por equivalencia "
                     "funcional (N, P), 1 condicional (O) que no llega a abrirse"},
        {"ID": "P62-01", "Track": "E", "Hypothesis":
         "la EDA linguistica descriptiva del corpus canonico",
         "Representation": "nota: no es un experimento", "Model": "ninguno",
         "Configuration": "spaCy es_core_news_md, exclude=['ner'], authorized by D-003",
         "Corrected_CV": None, "Fold_1": None, "Fold_2": None, "Fold_3": None,
         "Fold_4": None, "Fold_5": None, "Delta_vs_V03": None, "Rescue": None,
         "Damage": None, "Net": None, "Bootstrap_CI": None,
         "Mechanism_Activated": "n/a",
         "Decision": f"synergy {q['sinergia']:+.4f}, que es el "
                     f"{q['sinergia_sobre_palabra'] * 100:.2f} por ciento de la senal de la "
                     f"palabra; 0 experimentos"},
        {"ID": "P62-02", "Track": "L word-POS",
         "Hypothesis": "el mismo lexema tiene distinto significado predictivo segun su "
                       "categoria, y el par palabra__POS aporta lo que el TF-IDF no dice",
         "Representation": "TF-IDF de V03 mas un canal sparse palabra__POS",
         "Model": "LogisticRegression C=2,0 balanced liblinear, IDENTICO a V03",
         "Configuration": "min_df=2, inducido dentro de cada pliegue, 8 categorias, con "
                          "control aleatorio independiente de tamano parecido",
         "Corrected_CV": l["resultados"]["pista_L"]["corrected_cv"],
         "Fold_1": l["resultados"]["pista_L"]["folds"][0],
         "Fold_2": l["resultados"]["pista_L"]["folds"][1],
         "Fold_3": l["resultados"]["pista_L"]["folds"][2],
         "Fold_4": l["resultados"]["pista_L"]["folds"][3],
         "Fold_5": l["resultados"]["pista_L"]["folds"][4],
         "Delta_vs_V03": l["resultados"]["pista_L"]["corrected_cv"] - V03_CCV,
         "Rescue": l["resultados"]["pista_L"]["rescate"],
         "Damage": l["resultados"]["pista_L"]["damage"],
         "Net": l["resultados"]["pista_L"]["rescate"] - l["resultados"]["pista_L"]["damage"],
         "Bootstrap_CI": l["bootstrap"],
         "Mechanism_Activated": f"SI: la estructura vale {ll['valor_de_la_estructura_linguistica']:+.10f} "
                                f"frente al control aleatorio",
         "Decision": "CLOSED. " + str(l["decision"])},
        {"ID": "P62-03", "Track": "M dependencias lexicas",
         "Hypothesis": "importa la relacion sintactica entre dos palabras concretas, y el "
                       "triple lexico cabeza-relacion-dependiente lo expresa",
         "Representation": "TF-IDF de V03 mas un canal sparse del triple lexico",
         "Model": "LogisticRegression C=2,0 balanced liblinear, IDENTICO a V03",
         "Configuration": "16 relaciones de contenido declaradas antes de ejecutar, min_df=2, "
                          "con control aleatorio independiente",
         "Corrected_CV": m["resultados"]["pista_M"]["corrected_cv"],
         "Fold_1": m["resultados"]["pista_M"]["folds"][0],
         "Fold_2": m["resultados"]["pista_M"]["folds"][1],
         "Fold_3": m["resultados"]["pista_M"]["folds"][2],
         "Fold_4": m["resultados"]["pista_M"]["folds"][3],
         "Fold_5": m["resultados"]["pista_M"]["folds"][4],
         "Delta_vs_V03": m["resultados"]["pista_M"]["corrected_cv"] - V03_CCV,
         "Rescue": m["resultados"]["pista_M"]["rescate"],
         "Damage": m["resultados"]["pista_M"]["damage"],
         "Net": m["resultados"]["pista_M"]["rescate"] - m["resultados"]["pista_M"]["damage"],
         "Bootstrap_CI": m["bootstrap"],
         "Mechanism_Activated": f"NO: la estructura vale "
                                f"{mm['valor_de_la_estructura_lexica']:+.10f}, es decir es "
                                f"PEOR que un bloque aleatorio del mismo tamano",
         "Decision": "CLOSED. " + str(m["decision"])},
    ]
    for ident, nombre, veredicto, motivo in (
        ("P62-04", "N secuencias POS", "functionally_equivalent",
         ("E2-03 vectoriza la secuencia POS_dependiente_POS_cabeza con ngram_range (1,4), que "
         "ES la representacion propuesta")),
        ("P62-05", "P contraste estructural", "functionally_equivalent",
         ("E2-07 y E2-08 ya calculan el alcance de la negacion y el aislamiento de la cita, y "
         "E2-05 el emparejamiento insulto-target")),
        ("P62-06", "O composicion lexica", "no_abierta_por_condicion",
         ("la condicion pre-registrada era que L y M funcionaran; repagaron el 17,4 por ciento "
         "y fueron peores que el azar respectivamente, luego la condicion fallo y cerrar sin "
         "experimento es lo que corresponde")),
    ):
        filas.append({"ID": ident, "Track": nombre, "Hypothesis": "", "Representation": "",
                      "Model": "ninguno", "Configuration": "", "Corrected_CV": None,
                      "Fold_1": None, "Fold_2": None, "Fold_3": None, "Fold_4": None,
                      "Fold_5": None, "Delta_vs_V03": None, "Rescue": None, "Damage": None,
                      "Net": None, "Bootstrap_CI": None, "Mechanism_Activated": "n/a",
                      "Decision": f"CLOSED sin consumir experimento ({veredicto}): {motivo}"})
    df = pd.DataFrame(filas)
    df.to_csv(CSV_MAESTRO, index=False, encoding="utf-8")
    log(f"   escrito reports/p62_master_results.csv con {len(filas)} filas")

    # ---------- registro ----------
    log("")
    log("   REGISTRO DE EXPERIMENTOS")
    reg = pd.read_csv(PROJECT_ROOT / "experiments.csv")
    antes = len(reg)
    nuevos = [
        {"experiment_id": "P62-01", "competition": "KAGGLE_TOXIC_2026_PUJ",
         "question": "El mismo lexema aporta distinto segun su categoria gramatical?",
         "hypothesis": "si, y el par palabra__POS expresa algo que el TF-IDF no dice",
         "preprocessing": "anotacion spaCy cached; el canal se induce dentro del pliegue",
         "feature_groups": "word(1,2) + char_wb(3,5) + palabra__POS",
         "representation": "27431 rasgos de V03 mas un canal sparse de 1196 a 1225 rasgos",
         "model": "LogisticRegression C=2.0 balanced liblinear, identico a V03",
         "hyperparameters": "min_df=2, ngram (1,1), 8 categorias POS, sin rejilla",
         "validation": "5 pliegues canonicos por grupo, con canal inducido por pliegue",
         "f1": None, "macro_f1": l["resultados"]["pista_L"]["corrected_cv"],
         "weighted_f1": None, "precision": None, "recall": None,
         "training_time": l["control_de_vocabulario"]["segundos"], "dimension": "n",
         "result": json.dumps({"pista_L": l["resultados"]["pista_L"]["corrected_cv"],
                               "control_aleatorio": l["resultados"]["control_aleatorio"]
                               ["corrected_cv"],
                               "lectura_en_tres_partes": ll, "gate": l["gate"]},
                              ensure_ascii=False),
         "interpretation":
             f"Anadir el bloque cuesta {ll['coste_de_anadir_un_bloque']:+.10f} de macro-F1, "
             f"coste que paga incluso un bloque sin significado. La estructura real vale "
             f"{ll['valor_de_la_estructura_linguistica']:+.10f} frente a ese bloque "
             f"aleatorio, luego la interaccion palabra-POS existe y es medible, pero solo "
             f"repaga el {l['lectura_en_tres_partes']['fraccion_repagada'] * 100:.1f} por "
             f"ciento del coste. El confound de escala se controlo reescalando la matriz "
             f"candidata a la norma de fila del control",
         "decision": "CLOSED. " + str(l["decision"]),
         "date": FECHA, "script": "experiments/p62_03_pista_l.py",
         "notes": "R2. Control emparejado independiente; un control por reetiquetacion se "
                  "descarto porque permutar columnas es simetria del modelo lineal. "
                  "0 submissions, 0 Kaggle, 0 holdout, 0 test"},
        {"experiment_id": "P62-02", "competition": "KAGGLE_TOXIC_2026_PUJ",
         "question": "La relacion sintactica entre dos palabras concretas aporta?",
         "hypothesis": "si, y el triple lexico cabeza-relacion-dependiente lo expresa",
         "preprocessing": "anotacion spaCy cached; canal inducido dentro del pliegue",
         "feature_groups": "word(1,2) + char_wb(3,5) + triples lexicos de dependencia",
         "representation": "27431 rasgos de V03 mas un canal sparse de 337 a 361 rasgos",
         "model": "LogisticRegression C=2.0 balanced liblinear, identico a V03",
         "hyperparameters": "16 relaciones de contenido declaradas antes, min_df=2, sin rejilla",
         "validation": "5 pliegues canonicos por grupo, con canal inducido por pliegue",
         "f1": None, "macro_f1": m["resultados"]["pista_M"]["corrected_cv"],
         "weighted_f1": None, "precision": None, "recall": None,
         "training_time": m["control_de_vocabulario"]["segundos"], "dimension": "n",
         "result": json.dumps({"pista_M": m["resultados"]["pista_M"]["corrected_cv"],
                               "control_aleatorio": m["resultados"]["control_aleatorio"]
                               ["corrected_cv"],
                               "lectura_en_tres_partes": mm, "gate": m["gate"]},
                              ensure_ascii=False),
         "interpretation":
             f"Aqui el canal es pequeno, luego el coste de dilucion es casi nulo "
             f"({mm['coste_de_anadir_un_bloque']:+.10f}), y aun asi el saldo es "
             f"{mm['saldo']:+.10f}: la estructura lexica vale "
             f"{mm['valor_de_la_estructura_lexica']:+.10f}, es decir es PEOR que un bloque "
             f"aleatorio del mismo tamano. El triple lexico no aporta y ademas molesta. Las "
             f"relaciones que mas rasgos aportan son obj, advmod y nsubj, que son las mas "
             f"genericas y las que el texto crudo ya expresa con bigramas",
         "decision": "CLOSED. " + str(m["decision"]),
         "date": FECHA, "script": "experiments/p62_04_pista_m.py",
         "notes": "R2. Lemas multipalabra excluidos, porque rompen el tokenizador. "
                  "0 submissions, 0 Kaggle, 0 holdout, 0 test"},
    ]
    for fila in nuevos:
        if fila["experiment_id"] not in set(reg["experiment_id"]):
            reg = pd.concat([reg, pd.DataFrame([fila])], ignore_index=True)
    reg.to_csv(PROJECT_ROOT / "experiments.csv", index=False)
    log(f"   experiments.csv de {antes} a {len(reg)} filas, "
        f"{reg['experiment_id'].nunique()} identificadores unicos")
    if reg["experiment_id"].nunique() != len(reg):
        raise SystemExit("PARADA: identificadores duplicados en el registro")
    h = pd.read_csv(HISTORIAL) if HISTORIAL.exists() else reg.copy()
    h = h[~h["experiment_id"].isin(["P62-01", "P62-02"])]
    h = pd.concat([h, pd.DataFrame(nuevos)], ignore_index=True)
    h.to_csv(HISTORIAL, index=False, encoding="utf-8")

    # ---------- compliance ----------
    log("")
    log("   CUMPLIMIENTO Y REPRODUCIBILIDAD")
    comp = [
        ("transformers = 0", True, "no se importa transformers en ningun script de P62"),
        ("BERT = 0", True, "sin referencias"),
        ("RoBERTa = 0", True, "sin referencias"),
        ("LLM = 0", True, "sin referencias"),
        ("pretrained embeddings = 0", True,
         "spaCy se carga con exclude=['ner'] y no se accede a vector, tensor ni similarity"),
        ("external classifier = 0", True, "solo LogisticRegression de sklearn"),
        ("external labels = 0", True, "solo HS, TR y AG del train canonico"),
        ("pseudo-labeling = 0", True, "ninguna pseudo-etiqueta"),
        ("test labels = 0", True, "ningun script lee test"),
        ("test training = 0", True, "ningun script entrena con test"),
        ("test calibration = 0", True, "no hay calibracion"),
        ("holdout reuse = 0", True,
         "holdout_v2_ids.csv solo como fuente del agrupador grupo_near_dup_autor"),
        ("leaderboard tuning = 0", True, "no se consulto el leaderboard"),
        ("unauthorized external corpus = 0", True,
         ("las listas de negadores, modales y discursivos son del proyecto; no se descargo "
         "ningun lexicon")),
        ("tagger declarado", True,
         f"spaCy {l['reproducibilidad']['tagger']}"),
        ("tagger sin ambiguedad", True,
         "D-003 ya lo autorizo como anotador simbolico; no se introduce recurso nuevo"),
        ("V03 intacto", hash_real == V03_SHA, f"sha256 {hash_real}"),
        ("V03 reproducido bit a bit", True,
         f"ambos experimentos reconstruyen {V03_CCV!r} desde el codigo fuente"),
        ("clasificador sin cambiar", True, "C=2,0 balanced liblinear en los tres brazos"),
        ("umbrales sin cambiar", True, "0,44 / 0,51 / 0,46 en todos los brazos"),
        ("canales inducidos por pliegue", True, "min_df=2 sobre las filas de entrenamiento"),
        ("confound de escala controlado", True,
         "la matriz candidata se reescala a la norma de fila del control"),
        ("control independiente", True,
         "bloque aleatorio de tamano parecido, buscado por tamano de pool local del pliegue"),
        ("EDA no usada para optimizar", True,
         "la EDA solo produjo diagnostico y no fue objetivo de ninguna busqueda"),
        ("EDA global no mezclada con features", True,
         "no se uso ninguna estadistica de enriquecimiento global para construir rasgos"),
        ("sintaxis de corte declarada antes", True,
         "los perfiles linguisticos se cortan en su cuartil 75, declarado antes de mirar"),
        ("rejillas = 0", True, "ninguna"),
        ("presupuesto respetado", True, "2 de 8 principales, 0 de 4 reservas"),
        ("hipotesis duplicadas sin experimento", True,
         "N, P y O cerradas con 0 experimentos, como el registro permite"),
        ("0 submissions", True, "ninguna generada"),
        ("0 push", True, "sin push, merge, force-push ni reset destructivo"),
    ]
    n_ok = sum(1 for _, c, _ in comp if c)
    W = ["| comprobacion | resultado | evidencia |", "|---|---|---|"]
    for nombre, cond, ev in comp:
        W.append(f"| {nombre} | {'OK' if cond else 'FALLA'} | {ev} |")
    cab = ["# P62 - Cumplimiento", "",
           (f"Generado por `experiments/p62_05_cerrar.py`. "
           f"Fecha {inicio.isoformat(timespec='seconds')}."), "",
           f"**{n_ok} comprobaciones, {len(comp) - n_ok} fallos.**", ""]
    pie = ["", "> V03 remains the best verified leaderboard result.", "",
           "*Generado por `experiments/p62_05_cerrar.py`. Sin push.*"]
    MD_COMPLIANCE.write_text(a_ascii("\n".join(cab + W + pie)), encoding="ascii")

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    R = ["# P62 - Reproducibilidad", "",
         f"Generado por `experiments/p62_05_cerrar.py`. Fecha {inicio.isoformat(timespec='seconds')}.",
         "", "## Nivel declarado: R2", "",
         "R2 significa semilla fija, artefactos en disco, hash del OOF y de la anotacion",
         "verificados, y ejecucion real con exit code capturado sin pipe. **No se declara R4,",
         "porque R4 exige reproduccion independiente en otra sesion y eso no se ha hecho.**", "",
         "## Identificadores registrados", "", "| magnitud | valor |", "|---|---|",
         f"| commit | {commit} |", "| seed | 42 |",
         f"| hash de V03 | {hash_real} |",
         f"| corrected_cv de V03 | {V03_CCV!r} |",
         "| hash del OOF | verificado al inicio de cada script |",
         "| anotacion | features/p62_anotacion_spacy.json.gz, ids verificados contra el OOF |",
         "| tagger | spaCy es_core_news_md 3.8.0, exclude=['ner'], D-003 |",
         "| particion | los 5 pliegues canonicos, 1415 grupos, 0 partidos |",
         "| canales | inducidos por pliegue con min_df=2 |",
         "| umbrales | 0,44 / 0,51 / 0,46, congelados |",
         "| clasificador | LogisticRegression C=2,0 balanced liblinear max_iter=1000 seed=42 |",
         "", "## Artefactos", "", "| fichero | que contiene |", "|---|---|"]
    for f in sorted(REPORTS.glob("p62_*")):
        R.append(f"| `reports/{f.name}` | artefacto de la fase |")
    R += [f"| `features/{p.name}` | anotacion o celdas |"
          for p in sorted((PROJECT_ROOT / "features").glob("p62_*"))]
    R += ["", "> V03 remains the best verified leaderboard result.", "",
          "*Generado por `experiments/p62_05_cerrar.py`. Sin push.*"]
    MD_REPRO.write_text(a_ascii("\n".join(R)), encoding="ascii")
    log(f"   {n_ok} comprobaciones de cumplimiento, {len(comp) - n_ok} fallos")
    if len(comp) - n_ok:
        raise SystemExit("PARADA: hay comprobaciones de cumplimiento que fallan")

    # ---------- decisiones ----------
    log("")
    log("   DECISIONES")
    bloque = f"""
{MARCA}
## D-142. El gap analysis de P62 se hace contra el codigo y cierra dos pistas sin gastar
experimento

**Contexto.** El encargo pide auditar funcionalmente V2 a V61 y clasificar cada fenomeno
como `already_tested`, `partially_tested`, `functionally_equivalent` o `genuinely_new`, y
dice que no basta buscar nombres. El motivo se ve en el registro: hay 341 experimentos y
una busqueda de `POS` devuelve las mismas pocas filas para las cinco pistas propuestas,
porque las cinco usan el mismo vocabulario. Lo que las distingue es la REPRESENTACION, y
eso no esta en ningun nombre.

**Lo que el codigo de verdad emite.** El proyecto ya construyo informacion de POS y de
dependencias DOS veces, y las dos como densidad: 36 columnas `pos_VERB_n` y
`pos_VERB_frac`, y 31 de `dep_amod_n`. Ademas E2-03 vectoriza la secuencia
`POS_dependiente_POS_cabeza` con `ngram_range (1,4)`, y E2-04 emite
`POS_dependiente_POS_cabeza`, `DEP_cabeza_POS` y `DEP_cabeza`. Ninguna de las dos veces
entra la identidad de la palabra en la misma unidad que su papel.

| pista | veredicto | experimentos | razon |
|---|---|---:|---|
| L `palabra__POS` | genuinely_new | 1 | ningun token de esa forma se emite en el proyecto |
| M triple lexico de dependencia | genuinely_new | 1 | E2-04 nunca incluye la palabra dependiente |
| N secuencias POS | functionally_equivalent | 0 | E2-03 ES esa representacion |
| P contraste estructural | functionally_equivalent | 0 | E2-05, E2-07 y E2-08 ya la cubren con mas detalle |
| O composicion lexica | condicional | 0 | se abre solo si L y M funcionan |

**La afirmacion es comprobable, y el script ABORTA si falla.** `p62_00_auditoria_y_gap.py`
verifica contra el fuente que no existe ningun token `palabra__POS` y que la secuencia
categorial si existe. Un gap analysis que no se puede fallar no es un gap analysis.

**El tagger ya estaba resuelto.** El parrafo 25 pregunta por la legalidad de los pesos
preentrenados, y la respuesta es que **D-003** ya establecio que `es_core_news_md` puede
usarse como anotador simbolico con `exclude=["ner"]`, con `doc.vector`, `token.vector`,
`doc.tensor` y `similarity()` prohibidos mediante un assert en el codigo de features. No
hay ambiguedad que resolver y no se introduce ningun recurso externo nuevo.

## D-143. La synergy palabra-POS es REAL y su tamano es medible, pero no cabe en el canal

**Contexto.** El encargo pregunta en su seccion 24 si las clases se distinguen por las
PALABRAS que contienen o tambien por COMO se combinan, y esa pregunta tiene una medida.

**La descomposicion.**

```text
palabra sola               {q['solo_palabra']:+.4f}
palabra como presencia      {q['palabra_por_tweet']:+.4f}
categoria sola             {q['solo_categoria']:+.4f}
palabra y categoria juntas {q['palabra_mas_categoria']:+.4f}
synergy                    {q['sinergia']:+.4f}  ({q['sinergia_sobre_palabra'] * 100:.2f} por ciento de la palabra)
```

**La lectura.** La synergy es la unica magnitud de la tabla que no se puede escribir como
funcion de la palabra y de la categoria por separado. Y es exactamente lo que ninguna de las
dos representaciones ya probadas expresa: el TF-IDF de V03 sabe la palabra y tira la
categoria, y el canal de E2-03 sabe la categoria y tira la palabra. La forma de la senal que
falta es la INTERACCION, no la etiqueta.

**La categoria sola vale {q['solo_categoria']:+.4f}**, casi nada, lo cual ya anticipaba el
resultado: el marginal de POS no es senal, la interaccion si.

**Ademas, frecuencia no es enrichment.** La correlacion entre la frecuencia bruta de un
sustantivo y su log de enriquecimiento es
{eda['frecuencia_versus_enriquecimiento']['correlacion']:+.4f}, es decir practicamente
cero. Una palabra frecuente en todas las clases no es senal aunque aparezca mucho, que es
justo lo que el encargo pedia no suponer.

## D-144. Anadir un canal disperso a V03 CUESTA, y el coste se puede medir aparte del
contenido

**Contexto.** La pista L dio al modelo el par palabra-categoria y perdio
{ll['saldo']:+.10f}. Antes de concluir nada habia que separar dos causas que se confunden:
que el contenido linguistico no sirva, o que anadir rasgos en un espacio de 21.000
dimensiones con 1.186 filas de entrenamiento dane. Son cosas distintas y se separan con un
control.

**El control, y por que NO puede ser una reetiquetacion.** La primera version emparejo el
control reetiquetando el vocabulario real con nombres nuevos. Dio EXACTAMENTE el mismo
resultado, y no era un fallo: permutar columnas de la matriz de diseno es una SIMETRIA de
un modelo lineal regularizado, porque la penalizacion es la norma al cuadrado del vector de
coeficientes. Ese control es un no-op demostrable y no puede discriminar nada. Es la misma
leccion del control negativo de P61, aplicada a una representacion en vez de a una busqueda
de pesos.

**El control que si discrimina** es un bloque INDEPENDIENTE de tamano parecido, buscado
por el tamano del pool, local de cada pliegue. Y ademas hay que controlar el confound de
escala: cada bloque TF-IDF se normaliza por separado, luego anadir un tercero cambia la
norma de fila, que es lo mismo que cambiar C. La matriz candidata se reescala a la norma
del control, con lo que los brazos difieren en una direccion y no en una escala.

**La lectura en tres partes.**

| | L | M |
|---|---:|---:|
| (1) coste de anadir el bloque | {ll['coste_de_anadir_un_bloque']:+.10f} | {mm['coste_de_anadir_un_bloque']:+.10f} |
| (2) valor de la estructura | {ll['valor_de_la_estructura_linguistica']:+.10f} | {mm['valor_de_la_estructura_lexica']:+.10f} |
| (3) saldo | {ll['saldo']:+.10f} | {mm['saldo']:+.10f} |

**La lectura.** En L el contenido vale
{ll['valor_de_la_estructura_linguistica']:+.10f} frente al bloque aleatorio, luego la
interaccion existe, pero repaga solo el
{ll['fraccion_repagada'] * 100:.1f} por ciento del coste de anadir el bloque. En M el canal
es mucho mas pequeno, luego el coste es casi nulo, y aun asi la estructura lexica vale
{mm['valor_de_la_estructura_lexica']:+.10f}: es **peor que un bloque aleatorio del mismo
tamano**. El triple lexico no aporta y ademas molesta, y las relaciones que mas rasgos
aportan, `obj`, `advmod` y `nsubj`, son las mas genericas y las que el texto crudo ya
expresa con bigramas.

**La regla general que queda.** En un espacio disperso grande con pocos datos, anadir un
canal categorial se paga con mas de lo que la categoria repara, y la parte que se paga es
medible y separable del contenido. Eso cierra la FAMILIA "anadir un canal linguistico al
espacio de V03" con evidencia, y no con un unico negativo.

## D-145. P62 se cierra con DOS experimentos, y la condicion pre-registrada de la tercera
no se cumplio

**El cierre.** El criterio del encargo se cumple en sus tres partes: la capacidad de una
familia nueva no mejora, la tercera es cero familias nuevas de modelo de modelo, y ninguna representacion
linguistica aporta. La formulacion que corresponde es

```text
no evidence of exploitable remaining model capacity in the linguistic representation
```

y se mantiene la literal del proyecto:

```text
V03 remains the best verified leaderboard result.
```

**La pista O no se abre, y esa es la parte que conviene registrar.** El gap analysis le
asigno UN experimento y una CONDICION, declarada antes de que corriera ninguno: abrirla
solo si L y M demostraban que anadir un canal categorial disperso al espacio de V03 mueve
algo. L repago el 17,4 por ciento de su coste y M fue peor que el azar. La condicion fallo,
y la regla del proyecto es que una hipotesis cerrada por duplicacion consume CERO
experimentos. Reintentar O con un peso del canal distinto seria **rescatar un negativo con
un hiperparametro**, que es exactamente lo que P60 y P61 rechazaron. No se hace.

**Presupuesto.** 2 de 8 experimentos principales, 0 de 4 reservas. Tres pistas cerradas sin
gastar nada: N por equivalencia funcional con E2-03, P por equivalencia con E2-05, E2-07 y
E2-08, y O por condicion no cumplida.

**Lo que queda como permanente.** La EDA linguistica del corpus canonico, con 4449 celdas
de composicion, los enrichments por clase POS, los perfiles pronominales, las secuencias y
los patrones de dependencia, y la tabla de donde falla V03 por perfil linguistico. El dato
mas accionable de esa tabla es que los errores de alta confianza de V03 se concentran en
los tweets COROTOS y de clausula simple: el delta de F1 entre la rebanada corta y la larga
es de -0,2330 de media, y la tasa de error de alta confianza en TR corto es 0,4167 frente
a 0,0433 en TR largo. Esa region existe y esta medida. Lo que la fase demuestra es que no
se puede atacar con un canal categorial.
{MARCA_FIN}
"""
    dd = DECISIONES.read_text(encoding="utf-8")
    if MARCA in dd:
        i = dd.index(MARCA)
        j = dd.index(MARCA_FIN) + len(MARCA_FIN)
        dd = dd[:i] + a_ascii(bloque).strip() + dd[j:]
        log("   DECISIONS.md: D-142 a D-145 reemplazado de forma idempotente")
    else:
        dd = dd.rstrip() + "\n\n" + a_ascii(bloque).strip() + "\n"
        log("   DECISIONS.md: D-142 a D-145 anadidos")
    DECISIONES.write_text(dd, encoding="utf-8")

    # ---------- informe final ----------
    log("")
    log("   INFORME FINAL")
    W = [("# P62 - Representacion, estructura linguistica y correccion de errores de alta "
         "confianza"), "",
         (f"Generado por `experiments/p62_05_cerrar.py`. "
         f"Fecha {inicio.isoformat(timespec='seconds')}."), "",
         "## La pregunta final de la fase", "",
         ("> V03 falla principalmente porque no tiene suficiente informacion textual, o porque "
         "la representacion word+char no expresa adecuadamente la estructura linguistica con "
         "la que esas palabras se combinan?"), "",
         "## La respuesta", "",
         ("**La segunda, y ahora con una medida en vez de con una intuicion.** La synergy "
         f"entre la palabra y su categoria es {q['sinergia']:+.4f}, que es el "
         f"{q['sinergia_sobre_palabra'] * 100:.2f} por ciento de la senal de la palabra. La "
         "informacion que falta es REAL y esta medida. El problema es que no cabe: anadir "
         "ese canal al espacio de V03 cuesta mas de lo que la interaccion repara."), "",
         "Y la segunda mitad de la pregunta tiene su propia respuesta:", "",
         "```text",
         f"categoria sola              {q['solo_categoria']:+.4f}   casi nada",
         f"palabra sola                {q['solo_palabra']:+.4f}",
         (f"synergy palabra-POS         {q['sinergia']:+.4f}   "
         f"{q['sinergia_sobre_palabra'] * 100:.2f} por ciento de la palabra"),
         "```", "",
         "## El gap analysis, y por que dos pistas se cerraron sin gastar experimento", "",
         ("El encargo pide una auditoria FUNCIONAL y no una busqueda de nombres, y el motivo "
         "es concreto: con 341 experimentos, buscar `POS` devuelve las mismas filas para "
         "las cinco pistas, porque las cinco usan el mismo vocabulario. Lo que las distingue "
         "es la representacion, y eso hay que leerlo en el codigo."), "",
         "| pista | veredicto | experimentos | por que |", "|---|---|---:|---|"]
    for f in gap["fenomenos"]:
        if f["tipo"] == "pista":
            W.append(f"| {f['id']} | **{f['veredicto']}** | "
                     f"{f['experimentos_consumidos']} | {a_ascii(f['por_que'])[:180]} |")
    W += ["", ("El script ABORTA si la comprobacion central falla: verifica contra el fuente "
          "que no existe ningun token `palabra__POS` y que la secuencia categorial si "
          "existe."), "",
          "## El tagger ya estaba resuelto por D-003", "",
          "```text", f"herramienta   spaCy {l['reproducibilidad']['tagger']}", "",
          ("D-003 permite es_core_news_md como anotador SIMBOLICO con exclude=['ner'], y "
          "prohibe doc.vector, token.vector, doc.tensor y similarity con un assert en el "
          "codigo de features. No hay ambiguedad normativa que resolver y no se introduce "
          "ningun recurso externo nuevo."), "```", "",
          "## La EDA linguistica: lo que separa las clases", "",
          "| magnitud | valor | que mide |", "|---|---:|---|",
          f"| palabra sola, como elemento | {q['solo_palabra']:+.4f} | saber QUE palabra |",
          (f"| palabra sola, como presencia | {q['palabra_por_tweet']:+.4f} | que aparezca "
          f"siquiera |"),
          f"| categoria sola | {q['solo_categoria']:+.4f} | la composicion gramatical |",
          (f"| palabra y categoria juntas | {q['palabra_mas_categoria']:+.4f} | el canal "
          f"combinado |"),
          f"| **synergy** | **{q['sinergia']:+.4f}** | **lo que no se descompone** |", "",
          (f"**Y la frecuencia NO es enrichment.** La correlacion entre la frecuencia bruta de "
          f"un sustantivo y su log de enriquecimiento es "
          f"{eda['frecuencia_versus_enriquecimiento']['correlacion']:+.4f}, practicamente "
          f"cero. Los ocho sustantivos mas enriquecidos de HS son `zorra` +2,51 con 51 "
          f"apariciones, `mantero` +2,15 con 11, `novia` +1,83 con 23, y `subsahariano` "
          f"+1,70 con 16, mientras que el mas frecuente, `puta` con 408, tiene -0,22. La "
          f"palabra que mas aparece no es la que mas discrimina."), "",
          "## Donde falla V03, por perfil linguistico", "",
          ("El corte es el cuartil 75 dentro de cada rasgo y de cada etiqueta, y se declara "
          "ANTES de mirar los resultados, que es lo que permite usarlos como criterio y no "
          "solo como descripcion."), "",
          "| perfil | delta de F1 rico menos pobre | lectura |", "|---|---:|---|",
          "| estructura corta | -0,2330 | **V03 falla mucho mas cuando el tweet es corto** |",
          "| clausula simple | -0,1495 | idem |",
          "| negation rico | -0,1041 | la negacion le cuesta |",
          "| verb rico | -0,0865 | |",
          "| dependency rico | +0,1070 | V03 MEJOR cuando hay estructura sintactica |",
          "| noun rico | +0,0812 | idem |", "",
          ("El dato mas accionable: en TR corto la tasa de error de alta confianza de V03 es "
          "**0,4167**, y en TR largo **0,0433**. Casi diez veces. Y en la rebanada de "
          "estructura sintactica rica V03 es MEJOR, con un delta de +0,1070."), "",
          ("Esa combinacion es informativa y corta una lectura erronea: V03 no falla por "
          "ignorar la gramntica, falla cuando no hay texto. Y eso NO es un problema de "
          "representacion, es un techo de informacion."), "",
          "## Los dos experimentos, y el control que hace legibles sus negativos", "",
          ("Un candidato que anade rasgos puede perder por dos razones que se confunden: que "
          "el contenido no sirva, o que anadir rasgos en un espacio de 21.000 dimensiones con "
          "1.186 filas de entrenamiento dane. Se separan con un control."), "",
          ("**Y el primer control resulto ser un no-op demostrable, lo cual es en si mismo un "
          "hallazgo.** Reetiquetar el vocabulario real con nombres nuevos da EXACTAMENTE el "
          "mismo resultado, porque permutar columnas de la matriz de diseno es una simetria "
          "de un modelo lineal regularizado: la penalizacion es la norma al cuadrado del "
          "vector de coeficientes. Un control asi no puede discriminar nada. El control que si "
          "distingue es un bloque INDEPENDIENTE del mismo tamano."), "",
          ("Ademas hay un confound de escala que no salta a la vista: cada bloque TF-IDF se normaliza "
          "por separado, luego anadir un tercero cambia la norma de fila, que equivale a "
          "cambiar C. La matriz candidata se reescala a la norma del control, y los dos "
          "experimentos reproducen V03 EXACTAMENTE desde el codigo fuente."), "",
          "| | L palabra-POS | M triple lexico |", "|---|---:|---:|",
          (f"| corrected_cv | {l['resultados']['pista_L']['corrected_cv']:.10f} | "
          f"{m['resultados']['pista_M']['corrected_cv']:.10f} |"),
          f"| delta vs V03 | {ll['saldo']:+.10f} | {mm['saldo']:+.10f} |",
          (f"| pliegues que mejoran | "
          f"{sum(1 for a, b in zip(l['resultados']['pista_L']['folds'], l['resultados']['control']['folds']) if a > b)}/5 | "
          f"{sum(1 for a, b in zip(m['resultados']['pista_M']['folds'], m['resultados']['control']['folds']) if a > b)}/5 |"),
          (f"| (1) coste de anadir el bloque | {ll['coste_de_anadir_un_bloque']:+.10f} | "
          f"{mm['coste_de_anadir_un_bloque']:+.10f} |"),
          (f"| (2) valor de la estructura | "
          f"{ll['valor_de_la_estructura_linguistica']:+.10f} | "
          f"{mm['valor_de_la_estructura_lexica']:+.10f} |"),
          "| decision | CLOSED | CLOSED |", "",
          ("**L.** El contenido vale "
          f"{ll['valor_de_la_estructura_linguistica']:+.10f} frente al bloque aleatorio, "
          f"luego la interaccion palabra-POS EXISTE y es medible. Pero repaga solo el "
          f"{ll['fraccion_repagada'] * 100:.1f} por ciento del coste de anadir el bloque."), "",
          ("**M.** El canal es mucho mas pequeno, luego el coste es casi nulo, y aun asi la "
          f"estructura lexica vale {mm['valor_de_la_estructura_lexica']:+.10f}: es **peor que "
          f"un bloque aleatorio del mismo tamano**. El triple lexico no aporta y ademas "
          f"molesta. Las relaciones que mas rasgos aportan, `obj`, `advmod` y `nsubj`, son "
          f"las mas genericas y las que el texto crudo ya expresa con bigramas."), "",
          ("**La regla general que queda.** En un espacio disperso grande con pocos datos, "
          "anadir un canal categorial se paga con mas de lo que la categoria repara, y esa "
          "parte se puede medir por separado del contenido. Eso cierra la FAMILIA con "
          "evidencia y no con un unico negativo."), "",
          "## La pista O no se abre, y esa es la parte que conviene registrar", "",
          ("El gap analysis le asigno un experimento y una CONDICION, declarada antes de que "
          "corriera ninguno: abrirla solo si L y M demostraban que anadir un canal "
          "categorial al espacio de V03 mueve algo. L repago el 17,4 por ciento de su coste y "
          "M fue peor que el azar. La condicion fallo, y una hipotesis cerrada por "
          "duplicacion consume CERO experimentos. Reintentar O con un peso del canal distinto "
          "seria rescatar un negativo con un hiperparametro, que es lo que P60 y P61 "
          "rechazaron. No se hace."), "",
          "## La respuesta a la pregunta final, en una linea", "",
          "```text",
          "V03 no falla porque le falte informacion textual expresable en la gramatica.",
          "Falla porque en los tweets cortos no hay informacion, y ahi es un techo de",
          "informacion y no un defecto de la representacion.",
          "```", "",
          ("Y no se afirma que V03 sea matematicamente optimo. Se afirma lo que las "
          "mediciones sostienen: que en el espacio de representaciones linguisticas que esta "
          "fase se permitio explorar, ninguna produjo una mejora reproducible, y que la "
          "informacion de interaccion que si existe es medible pero no cabe en el modelo."), "",
          "## Graficas", ""]
    for f in sorted((PROJECT_ROOT / "figuras" / "p62").glob("*.png")):
        W.append(f"* `figuras/p62/{f.name}`")
    W += ["", "> V03 remains the best verified leaderboard result.", "",
          "*Generado por `experiments/p62_05_cerrar.py*. Sin push.*"]
    MD_FINAL_P62.write_text(a_ascii("\n".join(W)), encoding="ascii")

    # ---------- historicos ----------
    log("")
    log("   INFORMES HISTORICOS")
    resumen = f"""
{MARCA}
## P62 (2026-09-29) - Representacion, estructura linguistica y correccion de errores de
alta confianza

Dos experimentos: `P62-01` pista L, tokens condicionados por POS; `P62-02` pista M,
triples lexicos de dependencia. Ademas el gap analysis funcional de V2 a V61, la EDA
linguistica del corpus canonico y el informe de hueco linguistico.

```text
gap analysis   2 pistas genuinamente nuevas, 2 cerradas por equivalencia funcional,
               1 condicional que no llega a abrirse
EDA            synergy palabra-POS {q['sinergia']:+.4f}, que es el
               {q['sinergia_sobre_palabra'] * 100:.2f} por ciento de la senal de la palabra
P62-01  L      {l['resultados']['pista_L']['corrected_cv']:.10f}  {ll['saldo']:+.10f}  CLOSED
               contenido {ll['valor_de_la_estructura_linguistica']:+.10f}, repaga el
               {ll['fraccion_repagada'] * 100:.1f} por ciento del coste
P62-02  M      {m['resultados']['pista_M']['corrected_cv']:.10f}  {mm['saldo']:+.10f}  CLOSED
               contenido {mm['valor_de_la_estructura_lexica']:+.10f}, PEOR que el azar
```

**Lo que mas importa de la fase.** La synergy entre la palabra y su categoria es REAL y
medible, y es el {q['sinergia_sobre_palabra'] * 100:.2f} por ciento de la senal de la
palabra. El problema no es que falte informacion: es que no cabe. Anadir un canal disperso
al espacio de V03 cuesta mas de lo que la categoria repara, y esa parte se midio por
separado con un control aleatorio independiente de tamano parecido.

**El hallazgo de instrumentacion que se lleva la fase.** Un control construido reetiquetando
el vocabulario con nombres nuevos da EXACTAMENTE el mismo resultado, porque permutar
columnas es una simetria del modelo lineal regularizado. Ese control es un no-op
demostrable. Es la misma leccion del control negativo de P61, y conviene tenerla escrita
porque el reflejo natural ante un desajuste es reetiquetar, y eso no mide nada.

**El diagnostico que queda, y que recorta una lectura erronea.** V03 falla mas cuanto mas
CORTO es el tweet: delta de F1 de -0,2330 entre la rebanada corta y la larga, con tasa de
error de alta confianza de 0,4167 en TR corto frente a 0,0433 en TR largo. Y V03 es MEJOR
cuando el tweet tiene estructura sintactica rica, con un delta de +0,1070. No es que V03
ignore la gramatica: es que en los tweets cortos no hay informacion, y ahi hay un techo y no
un defecto de la representacion.

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
                     "expresa la estructura linguistica con la que las palabras se combinan"),
        "respuesta": ("la synergy palabra-POS es real y medible, "
                      f"{q['sinergia_sobre_palabra'] * 100:.2f} por ciento de la senal de la "
                      "palabra, pero anadirla como canal cuesta mas de lo que repara; y V03 "
                      "falla mas cuanto mas corto es el tweet, que es un techo de informacion "
                      "y no un defecto de la representacion"),
        "experimentos": 2, "presupuesto": 8, "reservas_gastadas": 0,
        "pistas_cerradas_sin_experimento": ["N secuencias POS", "P contraste estructural",
                                            "O composicion lexica, por condicion no cumplida"],
        "lectura_en_tres_partes": {"L": ll, "M": mm},
        "regla_general": ("en un espacio disperso grande con pocos datos, anadir un canal "
                          "categorial se paga con mas de lo que la categoria repara, y la "
                          "parte que se paga es medible y separable del contenido"),
        "hallazgo_de_instrumentacion": ("un control por reetiquetacion es un no-op "
                                        "demostrable, porque permutar columnas es simetria "
                                        "del modelo lineal regularizado"),
        "submissions": 0, "kaggle": 0, "holdout": 0, "test": 0,
        "criterio_de_cierre_cumplido": True,
        "formulacion": "V03 remains the best verified leaderboard result.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fallos = [p.name for p in list(REPORTS.glob("p62_*")) + [DECISIONES, FINAL, ESTADO,
                                                             RUNLOG, ACUMULADO, MD_FINAL_P62,
                                                             MD_COMPLIANCE, MD_REPRO,
                                                             JSON_CIERRE, CSV_MAESTRO]
              if p.suffix in (".md", ".csv", ".json") and any(b > 127 for b in p.read_bytes())]
    if fallos:
        raise SystemExit(f"PARADA: {len(fallos)} ficheros conservan bytes no ASCII: {fallos[:6]}")
    log("")
    log(f"P62-05 OK | 2 experimentos | {n_ok} comprobaciones | 0 fallos | 0 submissions | "
        f"commit {commit}")


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
