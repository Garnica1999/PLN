"""Correccion del criterio de auditoria: la licencia no era una regla, era un campo.

El error, dicho sin adornos
---------------------------
En P0 clasifique `pysentimiento/spanish-tweets` como AMBIGUOUS poniendo la ausencia de licencia
como primer motivo. La licencia no es una regla de esta competencia, y tampoco era una
prohibicion del encargo.

La verificacion, sobre el texto verbatim de `docs/competition_rules.md`
-----------------------------------------------------------------------
Las reglas hablan de datos en exactamente tres frases:

    esta permitido el uso de librerias comunes de NLP y ML
    el dataset de prueba no debe ser usado para entrenamiento
    se pueden usar modelos de vectorizacion, pero deben ENTRENARSE DESDE CERO

Cero menciones de licencias. Cero menciones de corpus externos. Y la tercera frase no solo no
prohibe esta linea: la pide.

Lo que el encargo pedia de verdad
---------------------------------
    registrar `license/terms` como CAMPO de la auditoria
    prohibir los corpus cuya finalidad fuese un dataset ETIQUETADO de las seis categorias
    garantizar razonablemente que el texto del test no este dentro

Un campo que se registra no es un criterio que descalifica. Los tres criterios reales son
etiquetado, procedencia y solapamiento.

Que cambia con la informacion nueva
-----------------------------------
Carlos confirma que el dataset de TOXIC fue RECOPILADO para la competencia y es privado. Eso
alimenta directamente el tercer criterio: un corpus publico no puede contener un dataset que
nunca se publico, salvo coincidencia de tweets concretos, y la coincidencia posible esta acotada
por el periodo y por el tema del corpus externo.

La licencia sigue importando como RIESGO PRACTICO, y se sigue registrando. Lo que no puede es
ser la razon principal de una clasificacion.
"""

from __future__ import annotations

import ast
import csv as _csv
import hashlib
import json
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import V03_SHA, V03_SUB

REPORTS = PROJECT_ROOT / "reports"
CSV_REG = REPORTS / "twitter_word2vec_corpus_registry.csv"
MD_C = REPORTS / "twitter_word2vec_audit_criteria_correction.md"
JSON_C = REPORTS / "twitter_word2vec_audit_criteria.json"

ALLOWED, AMBIGUOUS, PROHIBITED, NA = "ALLOWED", "AMBIGUOUS", "PROHIBITED", "NOT_APPLICABLE"


def log(m: str) -> None:
    """Imprime una linea de progreso.

    Args:
        m: texto a mostrar.
    """
    print(f"[P71-02] {m}", flush=True)


SUST = {chr(0x2014): " - ", chr(0x2013): "-", chr(0x2212): "-", chr(0x00B1): "+/-",
        chr(0x00AB): '"', chr(0x00BB): '"', chr(0x201C): '"', chr(0x201D): '"',
        chr(0x2264): "<=", chr(0x2265): ">=", chr(0x00D7): "x", chr(0x2192): "->",
        chr(0x2026): "...", chr(0x00BF): "?", chr(0x00A1): "!", chr(0x00A0): " "}


def a_ascii(t: str) -> str:
    """Translitera a ASCII puro.

    Args:
        t: cadena.

    Returns:
        La cadena en ASCII.
    """
    if not isinstance(t, str):
        t = str(t)
    for k, v in SUST.items():
        t = t.replace(k, v)
    out = []
    for ch in unicodedata.normalize("NFKD", t):
        if ord(ch) < 128:
            out.append(ch)
        elif not unicodedata.category(ch).startswith("M"):
            p = unicodedata.decomposition(ch).split()
            b = chr(int(p[0], 16)) if p and p[0] != "<wide>" else ""
            out.append(b if b and ord(b) < 128 else "?")
    return "".join(out)


CRITERIOS = [
    {"criterio": "1_corpus_etiquetado", "es_regla": True,
     "fuente": "encargo, item 3, prohibicion explicita",
     "enunciado": ("no utilizar ningun corpus cuya finalidad original sea un dataset "
                   "etiquetado de toxicidad, hate speech, cyberbullying, sentiment, age o "
                   "gender, si las etiquetas forman parte de su definicion")},
    {"criterio": "2_solapamiento_con_el_test", "es_regla": True,
     "fuente": "encargo, item 22",
     "enunciado": ("si no se puede garantizar razonablemente que el texto del test no este en "
                   "el corpus, clasificar como AMBIGUOUS y no utilizarlo")},
    {"criterio": "3_contenido_utilizable", "es_regla": True,
     "fuente": "implicito: la hipotesis exige texto de tweets",
     "enunciado": "el corpus tiene que contener texto de tweets sobre el que aprender"},
    {"criterio": "4_licencia_declarada", "es_regla": False,
     "fuente": "la competencia NO lo exige; el encargo lo pide como CAMPO a registrar",
     "enunciado": ("se registra y se declara como riesgo practico, pero no descalifica. "
                   "Elevarlo a criterio fue un error mio")},
]

# Reclasificacion con los criterios correctos. Cambia el peso de cada motivo, no los hechos.
RECLASIFICACION = [
    {"candidato": "mmaguero/covid19-spanishtweets-earlylateapril2020",
     "antes": AMBIGUOUS, "por_que_antes": "solapamiento no garantizable",
     "etiquetado": False, "licencia": "CC BY-SA 4.0", "contenido": "texto de tweets",
     "periodo": "cerrado, 29 de marzo a 1 de mayo de 2020",
     "despues": ALLOWED,
     "razon": ("pasa los tres criterios: sin etiquetas, con texto, y con periodo cerrado y "
               "documentado anterior a la competicion. La procedencia privada del dataset de "
               "TOXIC refuerza el criterio de solapamiento"),
     "pendiente": ("la comprobacion empirica contra el texto de TOXIC train, que es legal y "
                   "aporta evidencia adicional")},
    {"candidato": "pysentimiento/spanish-tweets",
     "antes": AMBIGUOUS, "por_que_antes": "sin licencia declarada",
     "etiquetado": False, "licencia": "NO DECLARADA", "contenido": "texto de tweets",
     "periodo": "abierto: coleccion de mayo de 2019 mas timelines sin periodo acotado",
     "despues": AMBIGUOUS,
     "razon": ("la licencia deja de ser el motivo principal y pasa a ser un riesgo declarado. "
               "Lo que lo mantiene fuera es el criterio 2: la recoleccion por timelines "
               "completas no tiene periodo acotado, luego el solapamiento no es garantizable "
               "ni conociendo la procedencia de TOXIC"),
     "pendiente": ""},
    {"candidato": "soncco/historical-twitter-trends-spanish-locations-2020-2021",
     "antes": NA, "por_que_antes": "no contiene texto de tweets",
     "etiquetado": False, "licencia": "ODC-By", "contenido": "cadenas de tendencias",
     "periodo": "cerrado, 2020-2021", "despues": NA,
     "razon": "criterio 3: sin texto de tweets no hay corpus sobre el que aprender",
     "pendiente": ""},
    {"candidato": "sinai-uja/COST", "antes": PROHIBITED,
     "por_que_antes": "etiquetado de sentimiento",
     "etiquetado": "SI, 34634 con polaridad", "licencia": "CC BY-NC-SA 4.0",
     "contenido": "texto de tweets", "periodo": "2015 o anterior", "despues": PROHIBITED,
     "razon": "criterio 1, el unico que la licencia no puede salvar",
     "pendiente": ""},
]


def main() -> None:
    """Corrige el criterio, reclasifica y reabre la linea si procede.

    Returns:
        None.
    """
    datetime.now(timezone.utc)
    log("P71-02 correccion del criterio de auditoria y reclasificacion")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")

    log("")
    log("   EL ERROR, Y LA VERIFICACION")
    log("      reglas de la competencia: 0 menciones de licencias, 0 de corpus externos")
    log("      y la tercera frase que si existe PIDE esta linea:")
    log("      'se pueden usar modelos de vectorizacion, pero deben ENTRENARSE DESDE CERO'")
    log("")
    log("      el encargo pedia REGISTRAR license/terms como campo, no descalificar por el")
    log("      elevar un campo descriptivo a criterio eliminatorio fue mi error")

    log("")
    log("   LOS CUATRO CRITERIOS, CON SU FUENTE")
    for c in CRITERIOS:
        marca = "REGLA " if c["es_regla"] else "CAMPO "
        log(f"      {marca} {c['criterio']:34} {c['fuente'][:52]}")

    log("")
    log("   RECLASIFICACION")
    for r in RECLASIFICACION:
        log(f"      {r['candidato'][:48]:50}")
        log(f"         antes {r['antes']:16} despues {r['despues']}")
        log(f"         {a_ascii(r['razon'])[:104]}")

    procede = any(r["despues"] == ALLOWED for r in RECLASIFICACION)
    ganador = next((r for r in RECLASIFICACION if r["despues"] == ALLOWED), None)
    log("")
    log(f"   ALLOWED tras la correccion: {sum(1 for r in RECLASIFICACION if r['despues'] == ALLOWED)}")
    log(f"   P1 procede: {procede}")
    if ganador:
        log(f"   candidato: {ganador['candidato']}")

    decision = {
        "el_error_corregido": ("la ausencia de licencia se trato como criterio eliminatorio "
                               "cuando era un campo a registrar"),
        "verificacion": ("docs/competition_rules.md, texto verbatim: 0 menciones de licencias "
                         "y 0 de corpus externos"),
        "lo_que_la_competencia_si_dice": [
            "esta permitido el uso de librerias comunes de NLP y ML",
            "el dataset de prueba no debe ser usado para entrenamiento",
            "se pueden usar modelos de vectorizacion, pero deben entrenarse DESDE CERO",
        ],
        "nota_sobre_la_tercera_frase": ("no solo no prohibe esta linea, la pide"),
        "criterios": CRITERIOS,
        "reclasificacion": RECLASIFICACION,
        "procede_p1": bool(procede),
        "candidato": ganador["candidato"] if ganador else None,
        "informacion_nueva_de_carlos": (
            "el dataset de TOXIC fue RECOPILADO para la competencia y es privado. Eso alimenta "
            "el criterio 2: un corpus publico no puede contener un dataset que nunca se publico, "
            "salvo coincidencia de tweets concretos"),
        "pendiente_antes_de_usarlo": (
            "comprobacion empirica de solapamiento contra el texto de TOXIC TRAIN, que es legal "
            "porque el train no esta prohibido"),
    }
    JSON_C.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")

    # Reescribir el registro con la clasificacion corregida.
    previo = json.loads((REPORTS / "twitter_word2vec_corpus_audit.json")
                        .read_text(encoding="utf-8"))
    todos = previo["registro_de_corpus"]
    for c in todos:
        for r in RECLASIFICACION:
            if c["candidato"] == r["candidato"]:
                c["clasificacion_original"] = c["clasificacion"]
                c["clasificacion"] = r["despues"]
                c["razon_original"] = c.get("razon_principal", "")
                c["razon_principal"] = r["razon"]
                c["registro_corregido"] = ("la licencia pasa de criterio eliminatorio a campo "
                                           "declarado")
    conteo = {}
    for c in todos:
        conteo[c["clasificacion"]] = conteo.get(c["clasificacion"], 0) + 1
    previo["registro_de_corpus"] = todos
    previo["conteo"] = conteo
    previo["correccion_del_criterio"] = decision
    previo["decision"]["puede_ejecutarse_p1"] = bool(procede)
    previo["decision"]["clasificacion_de_la_linea"] = (
        "REOPENED_P1_PROCEDE" if procede else "CLOSED_AT_P0_BY_CORPUS_UNAVAILABILITY")
    (REPORTS / "twitter_word2vec_corpus_audit.json").write_text(
        json.dumps(previo, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    campos = list(todos[0].keys())
    with CSV_REG.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(campos)
        for c in todos:
            w.writerow([a_ascii(str(c.get(k, "")))[:400] for k in campos])
    log(f"   registro reescrito: {conteo}")
    log("")
    log("=" * 78)
    log(f"P71-02 OK | {'P1 PROCEDE' if procede else 'linea sigue cerrada'}")
    log(f"   {conteo}")
    log("=" * 78)


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
