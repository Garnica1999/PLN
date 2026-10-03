"""P0: provenance audit of the Spanish Twitter/X corpora, before downloading or training anything.

Why the line stops here, and why that is a result and not a failure
------------------------------------------------------------------
The hypothesis was that a Word2Vec trained from scratch on real Spanish Twitter text could
capture distributional information that Wikipedia and mC4 did not. That is a good question and it
stays unanswered, because no corpus survives the audit this project requires.

Four candidates were examined. Each one fails for a DIFFERENT and independently verifiable
reason, and none of them fails for a reason that could be fixed by trying harder:

    pysentimiento/spanish-tweets          AMBIGUOUS
      597M tweets, text/tweet_id/user_id, no labels, purpose is language modelling. Good
      corpus in principle. Two problems and either one is enough. It declares NO LICENSE: the
      HuggingFace API returns an empty license field and the card has zero mentions of the word
      while its own table of contents lists a "Licensing Information" section that does not
      exist. And its collection method, downloading complete user timelines, means the date
      range is arbitrary per user, so overlap with TOXIC's texts cannot be ruled out without
      comparing against the test, which the rules forbid.

    pysentimiento/spanish-tweets-small   AMBIGUOUS
      Same provenance, same two problems.

    JorgeLoera/spanish-dialect-tweets    AMBIGUOUS
      Has a real license (cc-by-4.0) and that is the only thing it has. No dataset card, no
      provenance, no collection period, no field list. The single data file is called
      "Datos_dialectos.parquet", which suggests a labelled dialect-classification dataset, and a
      labelled external dataset is prohibited regardless of its topic.

    santiagxf/spanish-marketing-tweets   PROHIBITED
      Declares "unlicense", which is maximally permissive, and it does not matter: the
      repository contains a README and a .gitattributes and no data files at all. There is
      nothing to train on.

What this script does NOT do
----------------------------
It does not download any corpus. It does not train anything. It does not touch the test set,
the holdout or the leaderboard. It does not filter any external corpus against the test texts,
because that would itself be a use of the test during resource construction.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import V03_CCV, V03_SHA, V03_SUB

REPORTS = PROJECT_ROOT / "reports"
CSV_REG = REPORTS / "twitter_word2vec_corpus_registry.csv"
MD_A = REPORTS / "twitter_word2vec_domain_analysis.md"
JSON_A = REPORTS / "twitter_word2vec_corpus_audit.json"

ALLOWED, AMBIGUOUS, PROHIBITED = "ALLOWED", "AMBIGUOUS", "PROHIBITED"
FECHA_CONSULTA = "2026-09-29"


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P71-00] {mensaje}", flush=True)


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


# Cada candidato con los campos que el encargo exige auditar. Los datos vienen de la API de
# HuggingFace y de la ficha del dataset, consultados el 2026-09-29. NO se ha descargado nada.
REGISTRO = [
    {"candidato": "pysentimiento/spanish-tweets", "prioridad": 1,
     "url": "https://huggingface.co/datasets/pysentimiento/spanish-tweets",
     "provider": "pysentimiento (grupo de PLN, UBA / DC-UBA)",
     "source": ("coleccion Spritzer de Archive.org de mayo de 2019, filtrada a espanol, mas la "
                "descarga de las timelines completas de los 432.000 usuarios detectados"),
     "collection_period": ("linea base: mayo 2019. Las timelines se descargaron despues, sin "
                           "periodo acotado ni publicado por usuario"),
     "number_of_tweets": 597433111,
     "splits": {"train": 597433111, "test": 6224733},
     "campos": ["text", "tweet_id", "user_id"],
     "language_filtering": "solo en la deteccion inicial; el corpus admite 7-8 por ciento de otras lenguas",
     "labels_existen": False,
     "metadata_existe": True,
     "license_terms": ("NO DECLARADA. La API de HuggingFace devuelve el campo de licencia "
                       "vacio, y la ficha no menciona la palabra ni una vez pese a que su "
                       "indice lista una seccion 'Licensing Information' que no existe"),
     "download_date": FECHA_CONSULTA,
     "hash": "no aplica: no se ha descargado",
     "test_overlap_garantizable": False,
     "clasificacion": AMBIGUOUS,
     "razon_principal": ("sin licencia declarada, y sin poder garantizar que el texto del test "
                         "no este en el corpus sin compararlo contra el test, que esta prohibido"),
     },
    {"candidato": "pysentimiento/spanish-tweets-small", "prioridad": 2,
     "url": "https://huggingface.co/datasets/pysentimiento/spanish-tweets-small",
     "provider": "pysentimiento", "source": "misma procedencia que el corpus grande",
     "collection_period": "identico al corpus grande",
     "number_of_tweets": "10M < n < 100M (categoria declarada)",
     "splits": {"train": "10M<n<100M"},
     "campos": ["text", "tweet_id", "user_id"],
     "language_filtering": "identico",
     "labels_existen": False, "metadata_existe": True,
     "license_terms": "NO DECLARADA, igual que el corpus grande",
     "download_date": FECHA_CONSULTA, "hash": "no aplica",
     "test_overlap_garantizable": False, "clasificacion": AMBIGUOUS,
     "razon_principal": "hereda los dos problemas del corpus grande",
     },
    {"candidato": "JorgeLoera/spanish-dialect-tweets", "prioridad": 3,
     "url": "https://huggingface.co/datasets/JorgeLoera/spanish-dialect-tweets",
     "provider": "usuario individual", "source": "NO DOCUMENTADA",
     "collection_period": "NO DOCUMENTADO",
     "number_of_tweets": "100K < n < 1M (categoria declarada)",
     "splits": {"train": "100K<n<1M"},
     "campos": "NO DOCUMENTADOS. El unico fichero es Datos_dialectos.parquet",
     "language_filtering": "NO DOCUMENTADO",
     "labels_existen": ("PROBABLEMENTE SI. El nombre del fichero, 'Datos_dialectos', apunta a "
                        "clasificacion de dialecto, y un dataset externo etiquetado esta "
                        "prohibido con independencia del tema"),
     "metadata_existe": "NO DOCUMENTADO",
     "license_terms": "cc-by-4.0, la unica linea de la ficha",
     "download_date": FECHA_CONSULTA, "hash": "no aplica",
     "test_overlap_garantizable": False, "clasificacion": AMBIGUOUS,
     "razon_principal": ("tiene licencia real, y es lo unico que tiene: sin ficha, sin "
                         "procedencia, sin periodo y con indicios de estar etiquetado"),
     },
    {"candidato": "santiagxf/spanish-marketing-tweets", "prioridad": 4,
     "url": "https://huggingface.co/datasets/santiagxf/spanish-marketing-tweets",
     "provider": "usuario individual", "source": "no aplica: no hay datos",
     "collection_period": "no aplica", "number_of_tweets": 0,
     "splits": {}, "campos": [], "language_filtering": "no aplica",
     "labels_existen": "no aplica", "metadata_existe": "no aplica",
     "license_terms": "unlicense, la mas permisiva posible",
     "download_date": FECHA_CONSULTA, "hash": "no aplica",
     "test_overlap_garantizable": "no aplica", "clasificacion": PROHIBITED,
     "razon_principal": ("el repositorio contiene un README y un .gitattributes y NINGUN "
                         "fichero de datos. No hay corpus que entrenar"),
     },
]


def main() -> None:
    """Ejecuta la auditoria de procedencia y cierra P0 con su decision.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P71-00 P0: auditoria de procedencia de los corpus espanoles de Twitter/X")
    log("   NO se descarga nada, NO se entrena nada")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")

    # ---------- las reglas del encargo, recordadas ----------
    reglas = {
        "solo_se_utiliza_ALLOWED": True,
        "prohibido_si_su_finalidad_era_un_dataset_etiquetado_de":
            ["toxicidad", "hate speech", "cyberbullying", "sentiment", "age", "gender"],
        "prohibido_si_no_se_puede_garantizar_test_texts_not_in_external_corpus": True,
        "prohibido_filtrar_el_corpus_contra_el_test": True,
        "el_corpus_externo_solo_aporta_texto_no_etiquetado": True,
        "prohibido_como_feature": ["user_id", "author metadata", "followers", "timestamp",
                                   "geolocation", "likes", "retweets", "engagement",
                                   "profiles", "labels"],
    }
    log("")
    log("   LAS REGLAS DEL ENCARGO")
    for k, val in reglas.items():
        log(f"      {k}: {str(val)[:70]}")

    log("")
    log("   LOS CUATRO CANDIDATOS AUDITADOS")
    for c in REGISTRO:
        log(f"      {c['candidato']:40} {c['clasificacion']}")
        log(f"         {a_ascii(c['razon_principal'])[:100]}")

    conteo = {ALLOWED: 0, AMBIGUOUS: 0, PROHIBITED: 0}
    for c in REGISTRO:
        conteo[c["clasificacion"]] += 1
    log("")
    log(f"      ALLOWED {conteo[ALLOWED]} | AMBIGUOUS {conteo[AMBIGUOUS]} | "
        f"PROHIBITED {conteo[PROHIBITED]}")

    puede_ejecutarse = conteo[ALLOWED] > 0
    log("")
    log("   LA DECISION DE P0")
    if not puede_ejecutarse:
        log("      NINGUN corpus pasa la auditoria. P1 NO se puede ejecutar.")
        log("      La linea de domain-specific Word2Vec sobre Twitter/X queda cerrada en P0,")
        log("      por indisponibilidad de recurso, y NO por un resultado experimental.")
    else:
        log("      hay corpus ALLOWED y P1 podria ejecutarse")

    decision = {
        "clasificacion_de_la_linea": ("CLOSED_AT_P0_BY_CORPUS_UNAVAILABILITY"
                                      if not puede_ejecutarse else "PROCEDE"),
        "puede_ejecutarse_p1": bool(puede_ejecutarse),
        "experimentos_gastados": 0,
        "por_que_no_es_un_fracaso": (
            "la pregunta sigue sin respuesta. Lo que se cierra es la posibilidad de "
            "responderla con los recursos disponibles, no la hipotesis"),
        "por_que_no_se_usa_el_corpus_principal": (
            "dos motivos independientes y cada uno suficiente: no declara licencia, y no se "
            "puede garantizar que el texto del test no este dentro sin compararlo contra el "
            "test, cosa que el encargo prohibe"),
        "por_que_no_se_busca_mas": (
            "los cuatro candidatos se agotaron con la busqueda dirigida del encargo. Ninguno "
            "falla por un motivo que insistir arregle"),
        "alternativas_que_se_descartaron": [
            {"via": "usar el corpus sin licencia", "por_que_no": "incumple la regla del propio encargo"},
            {"via": "deduplicar contra el test", "por_que_no": "seria usar el test para construir el recurso"},
            {"via": "usar un dataset etiquetado", "por_que_no": "prohibido con independencia del tema"},
            {"via": "sustituir Word2Vec por FastText", "por_que_no": "cambiaria dos variables a la vez"},
            {"via": "usar los corpus generales de v5", "por_que_no": ("serian el CONTROL, no el "
                                                                      "tratamiento, y sin "
                                                                      "tratamiento no hay "
                                                                      "comparacion")},
        ],
        "lo_que_queda_registrado": (
            "el corpus principal y sus dos problemas quedan documentados con la evidencia de la "
            "consulta, para que una fase futura no repita la auditoria desde cero"),
    }
    log("")
    log(f"      clasificacion: {decision['clasificacion_de_la_linea']}")
    log(f"      experimentos gastados: {decision['experimentos_gastados']}")

    # ---------- ficheros ----------
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    campos = ["candidato", "prioridad", "url", "provider", "source", "collection_period",
              "number_of_tweets", "language_filtering", "labels_existen", "metadata_existe",
              "license_terms", "download_date", "hash", "test_overlap_garantizable",
              "clasificacion", "razon_principal"]
    with CSV_REG.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(campos)
        for c in REGISTRO:
            w.writerow([c.get(k, "") for k in campos])

    salida = {
        "familia": "P71_TWITTER_WORD2VEC_DOMAIN", "experiment_id": "P71-00",
        "fase": "P0 auditoria de corpus y legalidad",
        "fecha": inicio.isoformat(timespec="seconds"),
        "naturaleza": ("AUDITORIA. 0 descargas, 0 entrenamientos, 0 test, 0 holdout, "
                       "0 leaderboard"),
        "hipotesis": ("un Word2Vec entrenado desde cero sobre texto real de Twitter/X puede "
                      "capturar informacion distribucional especifica del dominio que los "
                      "entrenados sobre Wikipedia y mC4 no capturaron"),
        "novedad_declarada": ["entrenamiento desde cero", "corpus Twitter/X",
                              "dominio linguistico cercano al problema"],
        "reglas_aplicadas": reglas,
        "registro_de_corpus": [{k: v for k, v in c.items()} for c in REGISTRO],
        "conteo": conteo,
        "decision": decision,
        "reproducibilidad": {"nivel": "R2", "git_commit": commit, "hash_v03": hash_real,
                             "corrected_cv_v03": V03_CCV,
                             "consulta_a_la_api": FECHA_CONSULTA},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_A.write_text(json.dumps(salida, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    A: list[str] = []
    W = A.append
    W("# Domain-specific Word2Vec sobre Twitter/X: auditoria de corpus")
    W("")
    W(f"Generado por `experiments/p71_00_auditoria_corpus.py`. Fecha "
      f"{inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## La hipotesis, que no se ha podido poner a prueba")
    W("")
    W("> Un Word2Vec entrenado desde cero sobre texto real de Twitter/X puede capturar")
    W("> informacion distribucional especifica del dominio que los entrenados sobre Wikipedia y")
    W("> mC4 no capturaron.")
    W("")
    W("La novedad no era `Word2Vec existe`. Era **entrenamiento desde cero + corpus Twitter/X +")
    W("dominio linguistico cercano al problema**.")
    W("")
    W("```text")
    W("la pregunta SIGUE SIN RESPUESTA.")
    W("lo que se cierra es la posibilidad de responderla con los recursos disponibles,")
    W("no la hipotesis.")
    W("```")
    W("")
    W("## La decision de P0")
    W("")
    W("```text")
    W(f"{decision['clasificacion_de_la_linea']}")
    W("")
    W(f"ALLOWED {conteo[ALLOWED]}    AMBIGUOUS {conteo[AMBIGUOUS]}    PROHIBITED {conteo[PROHIBITED]}")
    W("")
    W(f"experimentos gastados: {decision['experimentos_gastados']}")
    W("```")
    W("")
    W("**Ningun corpus pasa la auditoria.** P1, la comparacion Word2Vec Twitter frente a")
    W("Word2Vec general, no se puede ejecutar. No hay tratamiento contra el que comparar.")
    W("")
    W("## Los cuatro candidatos")
    W("")
    for c in REGISTRO:
        W(f"### `{c['candidato']}` - {c['clasificacion']}")
        W("")
        W(f"- **Provider**: {a_ascii(c['provider'])}")
        W(f"- **Origen**: {a_ascii(str(c['source']))}")
        W(f"- **Periodo de recoleccion**: {a_ascii(str(c['collection_period']))}")
        W(f"- **Tweets**: {a_ascii(str(c['number_of_tweets']))}")
        W(f"- **Campos**: {a_ascii(str(c['campos']))}")
        W(f"- **Labels**: {a_ascii(str(c['labels_existen']))}")
        W(f"- **Licencia**: {a_ascii(c['license_terms'])}")
        W(f"- **Se puede garantizar que el texto del test no esta dentro?**: "
          f"`{c['test_overlap_garantizable']}`")
        W(f"- **Razon principal**: {a_ascii(c['razon_principal'])}")
        W("")
    W("## El corpus que el encargo senalaba, y sus dos problemas")
    W("")
    W("`pysentimiento/spanish-tweets` es un corpus **bueno en principio**: 597 millones de")
    W("tweets, campos `text`/`tweet_id`/`user_id`, **sin labels**, y su finalidad declarada es")
    W("*language modelling* para preentrenar representaciones. Es exactamente el tipo de recurso")
    W("que la hipotesis pedia.")
    W("")
    W("### Problema 1: no declara licencia")
    W("")
    W("```text")
    W("API de HuggingFace   campo de licencia VACIO")
    W("ficha del dataset    0 menciones de la palabra 'license'")
    W("                      su indice SI lista 'Licensing Information'")
    W("                      y la seccion NO existe en el documento")
    W("```")
    W("")
    W("### Problema 2: no se puede garantizar que el texto del test no este dentro")
    W("")
    W("```text")
    W("recoleccion   coleccion Spritzer de Archive.org de mayo de 2019,")
    W("              mas la descarga de las TIMELINES COMPLETAS de 432.000 usuarios")
    W("              localizados a partir de ella")
    W("TOXIC         competencia de 2026, sin periodo de recoleccion documentado")
    W("")
    W("las timelines cubren rangos de fecha arbitrarios por usuario, sin periodo")
    W("acotado ni publicado. La unica forma de comprobar el solapamiento seria")
    W("comparar contra el test, y el encargo PROHIBE usar ids, textos o hashes")
    W("del test para construir o limpiar el corpus")
    W("```")
    W("")
    W("El encargo es explicito: *en caso de no poder garantizar razonablemente, clasificar el")
    W("corpus como `AMBIGUOUS` y no utilizarlo*.")
    W("")
    W("**Cada problema es suficiente por si solo.** Se declaran los dos.")
    W("")
    W("## Las alternativas que se descartaron, y por que")
    W("")
    W("| via | por que no |")
    W("|---|---|")
    for a in decision["alternativas_que_se_descartaron"]:
        W(f"| {a['via']} | {a['por_que_no']} |")
    W("")
    W("## Lo que no se hace")
    W("")
    W("```text")
    W("no se descarga ningun corpus")
    W("no se entrena ningun embedding")
    W("no se toca el test, ni el holdout, ni el leaderboard")
    W("no se filtra ningun corpus externo contra el test")
    W("no se cambia Word2Vec por FastText: cambiaria dos variables a la vez")
    W("no se usan los corpus generales de v5 como tratamiento: serian el CONTROL")
    W("```")
    W("")
    W("## Lo que queda registrado para una fase futura")
    W("")
    W("El corpus principal y sus dos problemas quedan documentados con la evidencia de la")
    W("consulta, en `reports/twitter_word2vec_corpus_registry.csv`. Una fase futura que")
    W("consiguiera un corpus con licencia explicita y con un periodo de recoleccion anterior y")
    W("verificable no tendria que repetir esta auditoria desde cero.")
    W("")
    W("## El estado del proyecto, que no cambia")
    W("")
    W("```text")
    W("V03                          FROZEN")
    W("A_cost_sensitive_downweight  PROMISING_UNCONFIRMED")
    W("research_status              EXPERIMENTAL_RESEARCH_CLOSED")
    W("")
    W("domain-specific Word2Vec     CLOSED_AT_P0_BY_CORPUS_UNAVAILABILITY")
    W("```")
    W("")
    W("Cerrada esta linea, el estado de la investigacion experimental no cambia: sigue cerrada,")
    W("y el candidato sigue congelado.")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p71_00_auditoria_corpus.py`. Sin push.*")
    MD_A.write_text(a_ascii("\n".join(A)), encoding="ascii")

    documentos = [CSV_REG, MD_A, JSON_A]
    sucios = [d.name for d in documentos if any(b > 127 for b in d.read_bytes())]
    if sucios:
        raise SystemExit(f"PARADA: documentos con bytes no ASCII: {sucios}")

    log("")
    log("=" * 78)
    log(f"P71-00 OK | {decision['clasificacion_de_la_linea']}")
    log(f"   ALLOWED {conteo[ALLOWED]} | AMBIGUOUS {conteo[AMBIGUOUS]} | "
        f"PROHIBITED {conteo[PROHIBITED]}")
    log(f"   P1 ejecutable: {puede_ejecutarse}")
    log("   0 descargas, 0 entrenamientos, 0 test, 0 holdout, 0 leaderboard")
    log("=" * 78)


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
