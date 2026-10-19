#!/usr/bin/env python
"""
P100-00 (Fase S1): registro de fuentes nuevas, verificacion de licencias y
decisiones de autorizacion.

Solo escribe artefactos de gobernanza (no entrena, no descarga corpus):
  reports/p100_corpus_research.md
  reports/p100_corpus_registry.csv
  reports/p100_source_decisions.md

Verifica los hashes locales de lo descargado bajo autorizacion del propietario
(ChatSubs) y la evidencia de disponibilidad de S100-A. Los campos desconocidos
se marcan "no descargado"/"por verificar"; no se inventan cifras.

Uso:
    /home/carlos/venv/bin/python experiments/p100_00_registro_fuentes.py
"""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
SAL_INV = RAIZ / "reports" / "p100_corpus_research.md"
SAL_REG = RAIZ / "reports" / "p100_corpus_registry.csv"
SAL_DEC = RAIZ / "reports" / "p100_source_decisions.md"

EV_MIAMI = RAIZ / "data" / "external" / "corpora" / "p100_miami" / "evidencia"
EV_CHAT = RAIZ / "data" / "external" / "corpora" / "p100_chatsubs"

CAB = ["source_id", "source_name", "source_url", "version", "access_date",
       "license", "permission_status", "document_count", "raw_token_count",
       "clean_token_count", "unique_document_count", "language_distribution",
       "deduplication_method", "sample_seed", "sha256", "artifact_path",
       "notas"]

# Hashes oficiales del registro Zenodo 8220853 (ChatSubs), verificados.
MD5_OFICIAL = {
    "LICENSE.txt": "f5a53c7ab38ba3772e879f1407d3d412",
    "README.txt": "74bbf06a93146132a54bb7ade64654b8",
    "ChatSubs.tar.gz": "afd2d3967cf8cdf78883a7baa8d061e4",
}
TAR_BYTES = 1377313069

FILAS = [
    ["S100-A", "Digital Narratives of COVID-19 (DHCOVID)",
     ("https://covid.dh.miami.edu/ ; https://github.com/dh-miami/"
      "narratives_covid19"),
     "recoleccion 2020-04-24 a 2021-10-31; Zenodo 3824950 v1.0", "2026-10-13",
     "CC BY 4.0 (sitio y metadato Zenodo 3824950)",
     "PROHIBITED", "555 dias x 9 archivos (listas de IDs)", "no disponible",
     "no disponible", "no disponible", "es, en",
     "n/a (no descargado; solo IDs)", "no aplica",
     ("md5 zip oficial 2fc2148ac297c10b5ddebdb155f7baf1; respuesta /get0/ md5 "
      "b6bce1833ff2f01c84e01739be58bc77"),
     "data/external/corpora/p100_miami/evidencia/",
     ("PROHIBITED por indisponibilidad de TEXTO: artefactos publicos = listas "
      "de IDs de tweets (muestra verificada); servicio de texto /get0/ roto "
      "(PHP fatal error index.php:611, identico para toda consulta, "
      "verificado 2026-10-13); rehidratacion excluida por regla. Licencia OK.")],
    ["S100-B", "ChatSubs (dialogos derivados de OpenSubtitles)",
     "https://zenodo.org/records/8220853", "2023-07-27 (ficha Zenodo)",
     "2026-10-13",
     "CC BY-NC 4.0 (LICENSE.txt completo; ficha: other-nc + descripcion)",
     "ALLOWED_WITH_RESTRICTIONS",
     "206706 JSON (4 idiomas); es = open_subtitles_es (mayoritario)",
     "no disponible (objetivo P100: 9M de es; medicion al extraer)",
     "no disponible", "no disponible (dedup exacta al construir)",
     "es, ca, eu, gl", "sha1 del texto normalizado (planificado)",
     "20260925 (planificado)",
     ("md5 tar.gz afd2d3967cf8cdf78883a7baa8d061e4; LICENSE "
      "f5a53c7ab38ba3772e879f1407d3d412; README "
      "74bbf06a93146132a54bb7ade64654b8"),
     "data/external/corpora/p100_chatsubs/",
     ("Condiciones del propietario (2026-10-13): academico no comercial "
      "interno; solo es; sin anotaciones; atribucion; no redistribuir; no "
      "publicar checkpoints/pesos/derivados sin revision especifica; detener "
      "si restriccion material. Sin garantia juridica sobre el texto "
      "subyacente.")],
    ["S100-C", "esCorpiusDialog (LHF)",
     "https://huggingface.co/datasets/LHF (ficha por verificar)",
     "release 2026-02", "2026-10-13", "por verificar (no asumida)",
     "NEEDS_EXPLICIT_PERMISSION", "30518397 dialogos (todas las lenguas)",
     "2187821082 (todas las lenguas)", "no descargado", "no descargado",
     "es, ca, eu, gl", "n/a (no descargado)", "n/a", "no descargado", "",
     ("Candidata registrada; foros deshidratados (sin texto); Usenet por "
      "verificar; comparte familia OpenSubtitles con S100-B; no se usa.")],
    ["S100-D", "Spanish Reddit Dialogues (sunnweiwei)",
     "https://github.com/sunnweiwei/spanish-reddit-dialogues-corpus", "n/a",
     "2026-10-13", "sin licencia declarada en el repositorio",
     "NEEDS_EXPLICIT_PERMISSION", "no descargado", "no descargado",
     "no descargado", "no descargado", "es", "n/a (no descargado)", "n/a",
     "no descargado", "",
     "Datos via enlaces externos; sin licencia; no descargado."],
    ["S100-E", "OPUS OpenSubtitles (es)",
     "https://opus.nlpl.eu/OpenSubtitles.php", "por verificar", "2026-10-13",
     "por corpus / por verificar", "NEEDS_EXPLICIT_PERMISSION",
     "no descargado", "no descargado", "no descargado", "no descargado", "es",
     "n/a (no descargado)", "n/a", "no descargado", "",
     ("Redundante con S100-B (misma fuente subyacente OpenSubtitles); no se "
      "suma; no descargado.")],
    ["S100-F", "Wikimedia talk pages (eswiki)",
     "https://dumps.wikimedia.org/eswiki/", "dump por verificar", "2026-10-13",
     "CC BY-SA 4.0 (atribucion + share-alike)", "ALLOWED_WITH_RESTRICTIONS",
     "no descargado", "no descargado", "no descargado", "no descargado", "es",
     "n/a (no descargado)", "n/a", "no descargado", "",
     ("Reserva; instruccion del propietario: investigar sin descargar corpus "
      "completos hasta verificar licencia, volumen util y justificacion.")],
    ["S100-G", "PanaceaLab covid19_twitter",
     "https://github.com/thepanacealab/covid19_twitter", "por fecha",
     "2026-10-13", "n/a (proyecto de IDs)", "PROHIBITED", "solo IDs",
     "no disponible", "no disponible", "no disponible", "multi",
     "n/a (sin texto)", "no aplica", "no descargado", "",
     ("Solo listas de IDs; rehidratacion excluida por regla; sin texto "
      "utilizable.")],
    ["S100-H", "COVID-19-TweetIDs (echen102)",
     "https://github.com/echen102/COVID-19-TweetIDs", "2020-", "2026-10-13",
     "n/a (IDs)", "PROHIBITED", "solo IDs", "no disponible", "no disponible",
     "no disponible", "multi", "n/a (sin texto)", "no aplica", "no descargado",
     "", ("Solo listas de IDs; rehidratacion excluida por regla; sin texto "
          "utilizable.")],
    ["S100-I", "Harvard Dataverse Coronavirus Tweet IDs",
     ("https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/"
      "DVN/LW0BTB"), "por version", "2026-10-13", "n/a (IDs)", "PROHIBITED",
     "solo IDs", "no disponible", "no disponible", "no disponible", "multi",
     "n/a (sin texto)", "no aplica", "no descargado", "",
     ("Solo listas de IDs; rehidratacion excluida por regla; sin texto "
      "utilizable.")],
    ["S100-J", "TASS 2020 y corpus de sentimiento/emocion (es)",
     "varios (TASS, corpora de sentimiento)", "por fuente", "2026-10-13",
     "terminos por fuente", "NEEDS_EXPLICIT_PERMISSION", "no descargado",
     "no descargado", "no descargado", "no descargado", "es",
     "n/a (no descargado)", "n/a", "no descargado", "",
     "Sin texto publico directo ni licencia clara; no descargado."],
]

INVESTIGACION = """# P100 - S1: investigacion de fuentes nuevas (social / conversacional)

Fecha: 2026-10-13. Fase S1: verificacion de fichas, licencias, disponibilidad
real y solapamiento. Sin uso de toxic_test; sin labels; cuarentenas intactas;
descargas solo bajo autorizacion del propietario (2026-10-13).

## 1. Alcance y metodo

- Objetivo: identificar al menos 8 candidatas nuevas (texto social o
  conversacional en espanol) para la hipotesis de composicion de P100.
- Metodo: verificacion primaria en linea (fichas, licencias, repositorios,
  APIs, checksums oficiales) + sondas tecnicas de disponibilidad real +
  muestreo minimo. Toda afirmacion con evidencia reproducible archivada.
- Evidencia: data/external/corpora/p100_miami/evidencia/ y
  data/external/corpora/p100_chatsubs/.
- Estados finales: ALLOWED / ALLOWED_WITH_RESTRICTIONS /
  NEEDS_EXPLICIT_PERMISSION / PROHIBITED (combinan licencia y disponibilidad).

## 2. S100-A - Digital Narratives of COVID-19 (DHCOVID, Univ. Miami)

- Licencia: CC BY 4.0 (pie del sitio covid.dh.miami.edu; metadato Zenodo
  3824950, license.id = cc-by-4.0). Procedencia documentada: criterios de
  hashtags/terminos (criteria.md), sin retuits, recoleccion desde 2020-04-24,
  procesado de privacidad en el servicio de texto (menciones/URLs/emojis).
- Artefactos publicos:
  - GitHub dh-miami/narratives_covid19/twitter-corpus: 555 carpetas diarias
    (2020-04-24 .. 2021-10-31), 9 archivos .txt por dia (es, es_es, es_fl,
    en_fl, es_ar, es_mx, es_co, es_pe, es_ec).
  - Zenodo 3824950 (v1.0): dhcovid_Twitter_Datasets.zip (14637255 bytes,
    md5 2fc2148ac297c10b5ddebdb155f7baf1).
- CONTENIDO REAL VERIFICADO: los archivos son LISTAS DE IDs de tweets.
  Muestra: 2020-05-02/dhcovid_2020-05-02_es_fl.txt = 178 IDs numericos;
  dhcovid_2020-05-02_es.txt = 76119 IDs (1522380 bytes). Evidencia:
  evidencia/s100a_sample_ids_2020-05-02_es_fl.txt.
- Servicio de TEXTO (/get/ -> iframe /get0/): ROTO. Respuesta identica para
  toda consulta (incluidos los ejemplos documentados del blog), 393 bytes:
  "Fatal error: Uncaught ValueError: array_combine(): Argument #1 ($keys)
  and argument #2 ($values) must have the same number of elements in
  /var/www/covid.dh.miami.edu/html/get0/index.php:611". La pagina /get/ sin
  parametros devuelve HTML de 56717 bytes (md5
  feb2aa209f1f6eaaf560d6dba0ac29eb) y el iframe /get0/ el error. Evidencia:
  evidencia/s100a_get0_error.html.
- Rehidratacion de IDs: EXCLUIDA (regla vigente: no rehidratar sin
  autorizacion y procedimiento compatible; no existe procedimiento viable ni
  autorizado en 2026).
- Determinacion: licencia OK pero TEXTO NO OBTENIBLE -> no utilizable como
  fuente de texto social nuevo (ver p100_source_decisions.md).

## 3. S100-B - ChatSubs (Zenodo 8220853)

- Licencia: CC BY-NC 4.0 (LICENSE.txt = texto legal completo de
  Attribution-NonCommercial 4.0 International; ficha Zenodo: license.id =
  other-nc con descripcion "CC BY-NC 4.0").
- Artefactos: ChatSubs.tar.gz (1377313069 bytes; md5 oficial
  afd2d3967cf8cdf78883a7baa8d061e4; el espejo local se re-verifica por
  este script al completarse la descarga), README.txt (md5
  74bbf06a93146132a54bb7ade64654b8),
  LICENSE.txt (md5 f5a53c7ab38ba3772e879f1407d3d412), chatsubs_code.zip
  (md5 9544e03ec4d220c85aa2b940fbd40e40).
- Contenido: dialogos en espanol/catalan/euskera/gallego derivados de
  OpenSubtitles, procesados a dialogos y turnos; 206706 archivos JSON
  totales; mas de 20M dialogos; 96M turnos. Subconjunto es
  (open_subtitles_es) = el mayoritario; medicion exacta al extraer.
- Uso autorizado por el propietario (2026-10-13) con condiciones: academico
  no comercial; solo subconjunto es; sin anotaciones/labels; atribucion
  documentada; sin redistribuir; sin publicar checkpoints/pesos/derivados
  sin revision especifica; detener si aparece restriccion material; sin
  garantia juridica sobre el derecho del texto subyacente (OpenSubtitles).
- Determinacion: ALLOWED_WITH_RESTRICTIONS (fuente conversacional de la
  variante B de P100).

## 4. S100-C - esCorpiusDialog (LHF / HiTZ)

- Fuente: 30518397 dialogos, 2187821082 tokens, 15.8 GiB (release 2026-02);
  fuentes: subtitulos (OpenSubtitles), Usenet (newsgroups), foros (Meneame,
  Mediavida, Reddit: deshidratados, sin texto), literatura (Gutenberg).
- Licencia: por verificar en la ficha del dataset (no asumida; la familia
  esCorpius web es CC BY-NC-ND 4.0).
- Determinacion: NEEDS_EXPLICIT_PERMISSION (registrada; no se usa; comparte
  familia OpenSubtitles con S100-B).

## 5. S100-D / S100-E - Reddit dialogues y OPUS OpenSubtitles

- S100-D (sunnweiwei): sin licencia declarada en el repositorio; datos via
  enlaces externos; no descargado; NEEDS_EXPLICIT_PERMISSION.
- S100-E (OPUS): licencia por corpus/por verificar; misma fuente subyacente
  que S100-B (OpenSubtitles); redundante para esta hipotesis; no descargado;
  NEEDS_EXPLICIT_PERMISSION (no perseguida).

## 6. S100-F - Wikimedia talk pages (eswiki)

- Licencia: CC BY-SA 4.0 (texto de proyectos Wikimedia; atribucion +
  share-alike). Dumps publicos; volumen por medir si se decide su uso.
- Instruccion del propietario: investigar como reserva sin descargar corpus
  completos hasta verificar licencia, volumen util y justificacion.
- Determinacion: ALLOWED_WITH_RESTRICTIONS (reserva; no descargada).

## 7. S100-G / S100-H / S100-I - Conjuntos de IDs de Twitter COVID

- PanaceaLab covid19_twitter; COVID-19-TweetIDs (echen102); Harvard Dataverse
  (Kerchner-Wrubel). Publican LISTAS DE IDs (no texto). Rehidratacion
  excluida por regla y por no existir procedimiento viable/autorizado.
- Determinacion: PROHIBITED (incompatibles con la regla de no rehidratacion;
  sin texto).

## 8. S100-J - TASS 2020 y corpus de sentimiento/emocion (es)

- Terminos de investigacion por fuente; sin texto publico directo; no
  descargado.
- Determinacion: NEEDS_EXPLICIT_PERMISSION.

## 9. Resultado del barrido

- 10 candidatas nuevas evaluadas (>= 8 requeridas; sin forzar cuota).
- Sin fuente nueva de TEXTO social (tuits) utilizable -> variante A (social
  reforzado) no viable por material; se cierra sin relleno por duplicacion.
- Fuente conversacional disponible y autorizada: S100-B -> variante B viable.
- Ninguna candidata dudosa sustituye a otra; cuarentenas de odio intactas.

## 10. Reglas respetadas

- No se accedio a toxic_test, holdouts ni leaderboard; sin labels externas.
- Sin descargas no autorizadas: ChatSubs bajo autorizacion B; el resto fueron
  paginas de verificacion y ficheros minimos de metadata/evidencia.
- Sin modificaciones a V03 ni a resultados historicos de P98/P99.
"""

DECISIONES = """# P100 - S1: decisiones de fuentes y autorizacion (2026-10-13)

## 1. Marco

Autorizaciones del propietario (2026-10-13) que gobiernan esta fase:

- ChatSubs: evaluar para un experimento academico no comercial de
  preentrenamiento desde cero bajo condiciones: verificar ficha y artefacto;
  verificar condiciones del texto de origen; atribucion documentada; no
  redistribuir el corpus original ni publicar checkpoints, pesos o derivados
  sin revision especifica; solo subconjunto en espanol; sin usar anotaciones
  ni labels; cuarentena de corpus de odio intacta; detener la variante si la
  auditoria detecta una restriccion material; sin garantia juridica sobre el
  contenido subyacente.
- Descargas S1: A = Digital Narratives unicamente tras verificar procedencia,
  licencia y disponibilidad real; B = artefacto ChatSubs si la ficha confirma
  la licencia; C = investigar Wikimedia talk pages como reserva sin descargar
  corpus completos.
- Si solo existe COVID adicional y ChatSubs no puede utilizarse: cerrar con
  gap analysis. Si ChatSubs se autoriza: evaluar la viabilidad de ambas
  variantes pre-registradas; cerrar todo brazo sin documentos unicos
  suficientes, sin relleno por duplicacion.
- Presupuesto: maximo 2 configuraciones nuevas; maximo 10 GPU-h; no
  obligatorio agotarlo. Arquitectura, tokens totales, procedimientos
  downstream y criterios de parada congelados. No reabrir P99.
- Secuencia S1..S7; sin toxic_test/holdouts/leaderboard; sin modificar V03 ni
  los resultados historicos de P98/P99; sin submissions; sin commits/pushes
  automaticos.

## 2. Decision por fuente

| Fuente | Estado | Decision |
|---|---|---|
| S100-A DHCOVID | PROHIBITED (sin texto utilizable) | No usable: artefactos publicos = IDs; servicio /get0/ roto verificado; rehidratacion excluida. Licencia OK, disponibilidad NO. |
| S100-B ChatSubs | ALLOWED_WITH_RESTRICTIONS | Unica fuente conversacional usada (variante B) bajo condiciones del propietario. |
| S100-C esCorpiusDialog | NEEDS_EXPLICIT_PERMISSION | Registrada; no usada. |
| S100-D Reddit dialogues | NEEDS_EXPLICIT_PERMISSION | Sin licencia; no usada. |
| S100-E OPUS OpenSubtitles | NEEDS_EXPLICIT_PERMISSION | Redundante con S100-B; no usada. |
| S100-F Wikimedia talk pages | ALLOWED_WITH_RESTRICTIONS | Reserva; no descargada (verificar volumen/justificacion antes). |
| S100-G/H/I IDs Twitter | PROHIBITED | Sin texto; rehidratacion excluida. |
| S100-J TASS/sentimiento | NEEDS_EXPLICIT_PERMISSION | No usada. |

## 3. Consecuencias para las variantes P100

- Variante A (social reforzado, +4.5M social nuevo): NO VIABLE. No existe
  fuente autorizada y disponible de texto social nuevo (S100-A sin texto;
  S100-G/H/I solo IDs). Se cierra el brazo por material, sin relleno por
  duplicacion (regla del propietario).
- Variante B (social 15% + conversacional 30% + general 55%, 30M): VIABLE.
  Unica configuracion nueva a entrenar (dentro del maximo de 2).
- Sin sustitucion de fuentes dudosas; cuarentenas intactas; sin labels.
- La auditoria de S100-B no detecto restriccion material que impida el uso
  academico no comercial interno; obligaciones registradas (seccion 4).

## 4. Obligaciones operativas (ChatSubs)

- Solo subconjunto es (open_subtitles_es); sin anotaciones ni labels.
- Atribucion en reportes (ChatSubs: Kharitonova et al.; fuente original:
  OpenSubtitles).
- No redistribuir el corpus; no publicar checkpoints, pesos ni derivados sin
  revision especifica de condiciones. El proyecto no publica artefactos
  (sin commits/push).
- Uso interno academico no comercial; detener ante restriccion material.

## 5. Verificaciones tecnicas y pendientes

- Verificado: LICENSE.txt y README.txt (md5 oficiales); registro Zenodo;
  error del servicio /get0/; muestras de IDs; conteos del repositorio
  (555 carpetas); checksum oficial del tar.gz (md5 se re-verifica localmente
  al completar la descarga).
- Pendiente S1->S3: md5 local del tar.gz; extraccion; medicion del
  subconjunto es; construccion del corpus B1 y manifiesto (S3).

## 6. Lo que NO se hizo

- No se leyo, escaneo ni uso toxic_test ni holdouts; sin leaderboard.
- No se descargo ninguna fuente fuera de las autorizadas.
- No se tocaron artefactos historicos P98/P99 ni V03; sin commits.
"""


def md5(ruta: Path) -> str:
    h = hashlib.md5()
    with open(ruta, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verificar_descargas() -> list:
    notas = []
    for nombre, esperado in MD5_OFICIAL.items():
        p = EV_CHAT / nombre
        if not p.exists():
            if nombre == "ChatSubs.tar.gz":
                notas.append(f"{nombre}: AUSENTE (pendiente)")
            else:
                notas.append(f"{nombre}: AUSENTE (ERROR)")
            continue
        if nombre == "ChatSubs.tar.gz" and p.stat().st_size != TAR_BYTES:
            notas.append(
                f"ChatSubs.tar.gz: descarga incompleta ({p.stat().st_size} "
                f"de {TAR_BYTES} bytes); md5 pendiente")
            continue
        real = md5(p)
        estado = "OK" if real == esperado else "MISMATCH"
        notas.append(f"{nombre}: md5 {real} ({estado})")
    return notas


def anexo_evidencia() -> str:
    lineas = ["",
              ("## Anexo - evidencia archivada (hashes calculados por "
               "este script)"),
              ""]
    for base in (EV_MIAMI, EV_CHAT):
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file() or p.name == "ChatSubs.tar.gz":
                continue
            rel = p.relative_to(RAIZ)
            lineas.append(f"- {rel} | sha256={sha256(p)[:16]} | "
                          f"bytes={p.stat().st_size}")
    return "\n".join(lineas) + "\n"


def main() -> int:
    t0_notas = verificar_descargas()

    with open(SAL_REG, "w", encoding="ascii", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(CAB)
        for fila in FILAS:
            if len(fila) != len(CAB):
                raise ValueError(
                    f"{fila[0]}: campos {len(fila)} != {len(CAB)}")
            w.writerow(fila)

    with open(SAL_INV, "w", encoding="ascii") as fh:
        fh.write(INVESTIGACION)
        fh.write(anexo_evidencia())

    with open(SAL_DEC, "w", encoding="ascii") as fh:
        fh.write(DECISIONES)

    print("P100-00 registro S1 escrito:")
    print(f"  {SAL_REG.relative_to(RAIZ)} ({len(FILAS)} fuentes)")
    print(f"  {SAL_INV.relative_to(RAIZ)}")
    print(f"  {SAL_DEC.relative_to(RAIZ)}")
    print("verificacion de descargas (ChatSubs):")
    for nota in t0_notas:
        print(f"  {nota}")
    hay_error = any(("MISMATCH" in n) or ("AUSENTE (ERROR)" in n)
                    for n in t0_notas)
    if hay_error:
        print("ERROR: problemas de hash/presencia en artefactos descargados")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
