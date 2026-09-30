"""P0 ampliado: busqueda de corpus en HuggingFace, Kaggle y GitHub.

Que cambio respecto del primer P0
---------------------------------
El primer P0 audito cuatro candidatos de HuggingFace y los cuatro fallaron. Este los amplia a
tres plataformas y encuentra algo que el primero no encontro: un candidato donde la licencia y
el periodo son AMBOS correctos, y donde el unico hueco que queda es de otro tipo.

Eso no desbloquea la linea, pero cambia la forma del bloqueo, y merece quedar escrito porque
identifica exactamente que haria falta para desbloquearla.

Los tres hallazgos que importan
-------------------------------
    soncco/historical-twitter-trends-...   legal y con periodo documentado, pero NO CONTIENE
                                           TEXTO DE TWEETS: son cadenas de tendencias. El propio
                                           card lo dice: "it does not contain tweets"

    mmaguero/covid19-spanishtweets-...     LICENCIA EXPLICITA (CC BY-SA 4.0), PERIODO CERRADO Y
                                           DOCUMENTADO (29 marzo a 1 mayo de 2020), 2,27 millones
                                           de tweets espanoles con columna `text`, sin etiquetas
                                           de toxicidad. Es el candidato mas cercano a
                                           utilizable que ha aparecido

    sinai-uja/COST                         CC BY-NC-SA, pero ETIQUETADO DE SENTIMIENTO: 34.634
                                           tweets con polaridad. Prohibido por la regla del
                                           propio encargo, con independencia de la licencia

El bloqueo, que ha cambiado de naturaleza
-----------------------------------------
En el primer P0 los candidatos fallaban por motivos del CORPUS: sin licencia, o con recoleccion
abierta. En el candidato de Kaggle la licencia es real y el periodo esta cerrado. Lo que falta
es del otro lado:

    el periodo de recoleccion de TOXIC NO ESTA DOCUMENTADO EN NINGUNA PARTE

Las reglas oficiales de la competencia, leidas verbatim en `docs/competition_rules.md`, hablan
del formato y de que el test no se use para entrenar, y no dicen nada del periodo de los datos.

Sin ese dato, la no coincidencia con el test no se puede demostrar, y la regla del encargo es
conservadora.
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

from p59_01_voronoi_local import V03_SHA, V03_SUB

REPORTS = PROJECT_ROOT / "reports"
CSV_REG = REPORTS / "twitter_word2vec_corpus_registry.csv"
MD_A = REPORTS / "twitter_word2vec_domain_analysis.md"
JSON_A = REPORTS / "twitter_word2vec_corpus_audit.json"

ALLOWED, AMBIGUOUS, PROHIBITED, NA = "ALLOWED", "AMBIGUOUS", "PROHIBITED", "NOT_APPLICABLE"
FECHA = "2026-09-29"


def log(m: str) -> None:
    """Imprime una linea de progreso.

    Args:
        m: texto a mostrar.
    """
    print(f"[P71-01] {m}", flush=True)


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


# La busqueda ampliada. Campos exigidos por el encargo, con la evidencia de la consulta.
NUEVOS = [
    {"candidato": "soncco/historical-twitter-trends-spanish-locations-2020-2021",
     "plataforma": "HuggingFace", "prioridad": 5,
     "url": "https://huggingface.co/datasets/soncco/historical-twitter-trends-spanish-locations-2020-2021",
     "provider": "Braulio Andres Soncco Pimentel",
     "source": "columnas de tendencias de TweetDeck, recolector cada 15 minutos",
     "collection_period": "15 de julio de 2020 a 22 de septiembre de 2021",
     "number_of_tweets": "5.726.446 observaciones de tendencias, NO tweets",
     "language_filtering": "11 paises hispanohablantes",
     "labels_existen": False, "metadata_existe": True,
     "license_terms": "ODC-By 1.0, licencia real y explicita",
     "download_date": FECHA, "hash": "no aplica",
     "test_overlap_garantizable": "no aplica: no hay tweets que puedan solaparse",
     "clasificacion": NA,
     "razon_principal": ("el propio card lo dice: no contiene tuits, usuarios, noticias ni "
                         "etiquetas. Son cadenas de tendencias. Legal y bien documentado, pero "
                         "no es un corpus de texto de tweets")},
    {"candidato": "mmaguero/covid19-spanishtweets-earlylateapril2020",
     "plataforma": "Kaggle", "prioridad": 6,
     "url": "https://www.kaggle.com/datasets/mmaguero/covid19-spanishtweets-earlylateapril2020",
     "provider": "mmaguero",
     "source": ("tweets en espanol sobre COVID filtrados de otro dataset de Kaggle, con columna "
                "`text`, codigo de pais e idioma"),
     "collection_period": "29 de marzo de 2020 a 1 de mayo de 2020, CERRADO y documentado",
     "number_of_tweets": "2,27 millones de tweets en espanol",
     "language_filtering": "espanol declarado",
     "labels_existen": False,
     "metadata_existe": True,
     "license_terms": "CC BY-SA 4.0, licencia real y explicita",
     "download_date": FECHA, "hash": "no aplica: no se ha descargado",
     "test_overlap_garantizable": False,
     "clasificacion": AMBIGUOUS,
     "razon_principal": ("LICENCIA Y PERIODO CORRECTOS, y es el unico. Lo que falta es del otro "
                         "lado: el periodo de recoleccion de TOXIC no esta documentado en "
                         "ninguna parte, luego la no coincidencia con el test no se puede "
                         "demostrar")},
    {"candidato": "leadominguez/uruguayan-media-historical-tweets",
     "plataforma": "Kaggle", "prioridad": 7,
     "url": "https://www.kaggle.com/datasets/leadominguez/uruguayan-media-historical-tweets",
     "provider": "leadominguez", "source": "tweets de seis cuentas de MEDIOS uruguayos",
     "collection_period": "no documentado en la ficha",
     "number_of_tweets": "761 MB", "language_filtering": "espanol",
     "labels_existen": False, "metadata_existe": True,
     "license_terms": "CC0 1.0, dominio publico, la mas permisiva posible",
     "download_date": FECHA, "hash": "no aplica",
     "test_overlap_garantizable": False,
     "clasificacion": AMBIGUOUS,
     "razon_principal": ("periodo sin documentar, y el registro es PERIODISTICO, no "
                         "conversacional: un embedding de lenguaje de medios no es el dominio "
                         "de la hipotesis")},
    {"candidato": "hectorfr1984/spanish-trans-law-twitter-dataset",
     "plataforma": "Kaggle", "prioridad": 8,
     "url": "https://www.kaggle.com/datasets/hectorfr1984/spanish-trans-law-twitter-dataset",
     "provider": "hectorfr1984",
     "source": "1,5 millones de tweets alrededor del debate de la Ley Trans espanola",
     "collection_period": "el debate es de 2022-2023, sin periodo exacto en la ficha",
     "number_of_tweets": "1,5 millones", "language_filtering": "espanol",
     "labels_existen": False, "metadata_existe": True,
     "license_terms": "CC0 1.0",
     "download_date": FECHA, "hash": "no aplica",
     "test_overlap_garantizable": False,
     "clasificacion": AMBIGUOUS,
     "razon_principal": ("el tema del corpus se solapa con la tarea objetivo: el debate sobre "
                         "una ley de derechos de un grupo protegido es exactamente el espacio "
                         "donde vive AG. Entrenar la representacion con ese vocabulario "
                         "introduce senal de tema en el clasificador de toxicidad")},
    {"candidato": "sinai-uja/COST", "plataforma": "GitHub", "prioridad": 9,
     "url": "https://github.com/sinai-uja/COST",
     "provider": "SINAI, Universidad de Jaen",
     "source": "34.634 tweets en espanol con polaridad derivada de los emoticonos",
     "collection_period": "publicacion de 2015, tweets anteriores",
     "number_of_tweets": 34634, "language_filtering": "espanol",
     "labels_existen": True,
     "metadata_existe": False,
     "license_terms": "CC BY-NC-SA 4.0, mas solicitud por correo",
     "download_date": FECHA, "hash": "no aplica",
     "test_overlap_garantizable": False,
     "clasificacion": PROHIBITED,
     "razon_principal": ("ETIQUETADO DE SENTIMIENTO, 17.317 positivos y 17.317 negativos. "
                         "Prohibido por la regla del encargo con independencia de la licencia"),
     },
]


def main() -> None:
    """Funde la busqueda ampliada con la primera y reescribe la auditoria.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P71-01 P0 ampliado: HuggingFace, Kaggle y GitHub")
    log("   NO se descarga nada, NO se entrena nada")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    previo = json.loads((REPORTS / "twitter_word2vec_corpus_audit.json")
                        .read_text(encoding="utf-8"))
    todos = previo["registro_de_corpus"] + NUEVOS
    for c in todos:
        c.setdefault("plataforma", "HuggingFace")

    conteo = {ALLOWED: 0, AMBIGUOUS: 0, PROHIBITED: 0, NA: 0}
    for c in todos:
        conteo[c["clasificacion"]] = conteo.get(c["clasificacion"], 0) + 1
    log("")
    log(f"   {len(todos)} candidatos auditados en total")
    for c in todos:
        log(f"      {c['plataforma']:12} {c['candidato'][:46]:48} {c['clasificacion']}")
    log("")
    log(f"      ALLOWED {conteo[ALLOWED]} | AMBIGUOUS {conteo[AMBIGUOUS]} | "
        f"PROHIBITED {conteo[PROHIBITED]} | NOT_APPLICABLE {conteo[NA]}")

    # ---------- el bloqueo ha cambiado de naturaleza ----------
    bloqueo = {
        "en_el_primer_p0": ("los candidatos fallaban por motivos DEL CORPUS: sin licencia "
                            "declarada, o con recoleccion abierta y sin periodo acotado"),
        "en_la_busqueda_ampliada": (
            "el candidato de Kaggle tiene LICENCIA EXPLICITA y PERIODO CERRADO Y DOCUMENTADO. "
            "Lo que falta es del otro lado: el periodo de recoleccion de TOXIC no esta "
            "documentado en ninguna parte"),
        "donde_se_comprobo": "docs/competition_rules.md, texto verbatim del organizador",
        "que_dicen_las_reglas": ("hablan del formato de los ficheros y de que el dataset de "
                                 "prueba no se use para entrenamiento. No mencionan el periodo "
                                 "de recoleccion de los datos ni los corpus externos"),
        "consecuencia": ("sin el periodo de TOXIC, la no coincidencia con el test no se puede "
                         "demostrar, y la regla del encargo es conservadora"),
        "lo_que_desbloquearia_la_linea": [
            "que el organizador documente el periodo de recoleccion de TOXIC",
            "o una regla explicita sobre el uso de corpus externos anteriores",
            ("o un corpus cuya recoleccion fuese demostrablemente anterior al inicio de la "
            "competicion por una fuente independiente"),
        ],
        "no_es_un_fracaso": ("identifica exactamente que haria falta. Eso es mas util que "
                             "cerrar sin saber por que"),
    }
    log("")
    log("   EL BLOQUEO HA CAMBIADO DE NATURALEZA")
    log(f"      primer P0:      {bloqueo['en_el_primer_p0'][:66]}")
    log(f"      busqueda nueva: {bloqueo['en_la_busqueda_ampliada'][:66]}")
    log("")
    log("      lo que desbloquearia la linea:")
    for x in bloqueo["lo_que_desbloquearia_la_linea"]:
        log(f"         - {x}")

    decision = {
        "clasificacion_de_la_linea": "CLOSED_AT_P0_BY_CORPUS_UNAVAILABILITY",
        "puede_ejecutarse_p1": False,
        "experimentos_gastados": 0,
        "candidatos_auditados": len(todos),
        "plataformas": ["HuggingFace", "Kaggle", "GitHub"],
        "conteo": conteo,
        "el_candidato_mas_cercano": "mmaguero/covid19-spanishtweets-earlylateapril2020",
        "por_que_el_mas_cercano_no_basta": (
            "su licencia es real y su periodo esta cerrado y documentado a 2020, pero el "
            "periodo de TOXIC no esta documentado, luego la no coincidencia con el test no "
            "se puede demostrar"),
        "bloqueo": bloqueo,
    }

    # ---------- ficheros ----------
    subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    campos = ["plataforma", "candidato", "prioridad", "url", "provider", "source",
              "collection_period", "number_of_tweets", "language_filtering", "labels_existen",
              "metadata_existe", "license_terms", "download_date", "hash",
              "test_overlap_garantizable", "clasificacion", "razon_principal"]
    with CSV_REG.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(campos)
        for c in todos:
            w.writerow([a_ascii(str(c.get(k, ""))) for k in campos])

    salida = dict(previo)
    salida["registro_de_corpus"] = todos
    salida["decision"] = decision
    salida["busqueda_ampliada"] = {
        "plataformas_consultadas": ["HuggingFace", "Kaggle", "GitHub"],
        "consultas": ["6 en HuggingFace", "8 en Kaggle", "4 en GitHub"],
        "candidatos_unicos_kaggle": 42,
        "hallazgos_nuevos": len(NUEVOS),
    }
    (REPORTS / "twitter_word2vec_corpus_audit.json").write_text(
        json.dumps(salida, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # ---------- el informe ----------
    A: list[str] = []
    W = A.append
    W("# Domain-specific Word2Vec sobre Twitter/X: auditoria de corpus")
    W("")
    W(f"Generado por `experiments/p71_00_auditoria_corpus.py` y "
      f"`experiments/p71_01_auditoria_ampliada.py`. Fecha "
      f"{inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## La hipotesis, que no se ha podido poner a prueba")
    W("")
    W("> Un Word2Vec entrenado desde cero sobre texto real de Twitter/X puede capturar")
    W("> informacion distribucional especifica del dominio que los entrenados sobre Wikipedia y")
    W("> mC4 no capturaron.")
    W("")
    W("```text")
    W("la pregunta SIGUE SIN RESPUESTA.")
    W("```")
    W("")
    W("## La decision")
    W("")
    W("```text")
    W("CLOSED_AT_P0_BY_CORPUS_UNAVAILABILITY")
    W("")
    W(f"{len(todos)} candidatos auditados en 3 plataformas")
    W(f"ALLOWED {conteo[ALLOWED]} | AMBIGUOUS {conteo[AMBIGUOUS]} | "
      f"PROHIBITED {conteo[PROHIBITED]} | NOT_APPLICABLE {conteo[NA]}")
    W("experimentos gastados: 0")
    W("```")
    W("")
    W("## Los candidatos, por plataforma")
    W("")
    W("| plataforma | candidato | licencia | periodo | etiquetado | clase |")
    W("|---|---|---|---|---|---|")
    for c in todos:
        lab = "SI" if c.get("labels_existen") is True else (
            "no" if c.get("labels_existen") is False else "n/d")
        W(f"| {c['plataforma']} | `{c['candidato'][:44]}` | "
          f"{a_ascii(str(c['license_terms']))[:30]} | "
          f"{a_ascii(str(c['collection_period']))[:26]} | {lab} | **{c['clasificacion']}** |")
    W("")
    W("## El bloqueo ha cambiado de naturaleza, y eso es el resultado de la busqueda")
    W("")
    W("En el primer P0 los candidatos fallaban por motivos **del corpus**: sin licencia")
    W("declarada, o con recoleccion abierta y sin periodo acotado.")
    W("")
    W("La busqueda ampliada encontro algo distinto:")
    W("")
    W("### `mmaguero/covid19-spanishtweets-earlylateapril2020`")
    W("")
    W("```text")
    W("licencia    CC BY-SA 4.0, real y explicita")
    W("periodo     29 de marzo a 1 de mayo de 2020, CERRADO y documentado")
    W("contenido   2,27 millones de tweets espanoles, con columna `text`")
    W("etiquetas   ninguna de toxicidad")
    W("```")
    W("")
    W("**Es el candidato mas cercano a utilizable que ha aparecido.** Licencia y periodo son")
    W("ambos correctos, y es el unico.")
    W("")
    W("### Y el hueco que queda es del otro lado")
    W("")
    W("```text")
    W("el periodo de recoleccion de TOXIC NO ESTA DOCUMENTADO EN NINGUNA PARTE")
    W("```")
    W("")
    W("Se comprobo en `docs/competition_rules.md`, el texto verbatim del organizador: las reglas")
    W("hablan del formato de los ficheros y de que el dataset de prueba no se use para")
    W("entrenamiento, y **no mencionan el periodo de recoleccion de los datos ni los corpus")
    W("externos**.")
    W("")
    W("Sin ese dato, la no coincidencia con el test no se puede demostrar, y la regla del")
    W("encargo es conservadora.")
    W("")
    W("## Los tres hallazgos que importan")
    W("")
    W("### Uno: un corpus legal y bien documentado que no sirve")
    W("")
    W("`soncco/historical-twitter-trends-spanish-locations-2020-2021` tiene ODC-By, periodo")
    W("cerrado y ninguna etiqueta. Y su propio card lo dice sin ambiguedad:")
    W("")
    W("> *It does not contain tweets, user accounts, news articles, URLs, or veracity labels.*")
    W("")
    W("Son cadenas de tendencias. No hay texto de tweets sobre el que aprender distribuciones.")
    W("Es `NOT_APPLICABLE`: no lo prohibe la regla, simplemente no es el recurso.")
    W("")
    W("### Dos: un corpus con todo correcto salvo el dato que falta")
    W("")
    W("El de Kaggle, descrito arriba.")
    W("")
    W("### Tres: un corpus prohibido con licencia impecable")
    W("")
    W("`sinai-uja/COST` esta bajo CC BY-NC-SA 4.0 y contiene **34.634 tweets etiquetados de")
    W("sentimiento**, 17.317 positivos y 17.317 negativos. La regla del encargo prohibe los")
    W("corpus cuya finalidad era un dataset etiquetado de sentimiento, **con independencia de su")
    W("licencia**.")
    W("")
    W("## Lo que desbloquearia la linea")
    W("")
    W("```text")
    W("que el organizador documente el periodo de recoleccion de TOXIC")
    W("o una regla explicita sobre el uso de corpus externos anteriores")
    W("o un corpus cuya recoleccion fuese demostrablemente anterior al inicio de la")
    W("  competicion por una fuente independiente")
    W("```")
    W("")
    W("**Eso es mas util que cerrar sin saber por que**, y es lo que la busqueda ampliada ha")
    W("anadido: no solo que no hay corpus, sino exactamente que haria falta.")
    W("")
    W("## Lo que no se hace")
    W("")
    W("```text")
    W("no se descarga ningun corpus")
    W("no se entrena ningun embedding")
    W("no se toca el test, ni el holdout, ni el leaderboard")
    W("no se filtran corpus externos contra el test")
    W("no se usa el corpus de Kaggle aunque su licencia sea correcta")
    W("```")
    W("")
    W("## El estado del proyecto, que no cambia")
    W("")
    W("```text")
    W("V03                          FROZEN")
    W("A_cost_sensitive_downweight  PROMISING_UNCONFIRMED")
    W("research_status              EXPERIMENTAL_RESEARCH_CLOSED")
    W("domain-specific Word2Vec     CLOSED_AT_P0_BY_CORPUS_UNAVAILABILITY")
    W("```")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Sin push.*")
    MD_A.write_text(a_ascii("\n".join(A)), encoding="ascii")

    docs = [CSV_REG, MD_A, REPORTS / "twitter_word2vec_corpus_audit.json"]
    sucios = [d.name for d in docs if any(b > 127 for b in d.read_bytes())]
    if sucios:
        raise SystemExit(f"PARADA: documentos con bytes no ASCII: {sucios}")

    log("")
    log("=" * 78)
    log(f"P71-01 OK | {decision['clasificacion_de_la_linea']}")
    log(f"   {len(todos)} candidatos | ALLOWED {conteo[ALLOWED]} | "
        f"se mantiene el cierre")
    log("   el bloqueo ha pasado de ser del corpus a ser del periodo de TOXIC")
    log("=" * 78)


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
