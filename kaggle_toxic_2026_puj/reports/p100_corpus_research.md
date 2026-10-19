# P100 - S1: investigacion de fuentes nuevas (social / conversacional)

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

## Anexo - evidencia archivada (hashes calculados por este script)

- data/external/corpora/p100_miami/evidencia/s100a_criteria.md | sha256=28d15ea923bf11a3 | bytes=2190
- data/external/corpora/p100_miami/evidencia/s100a_get0_error.html | sha256=4bf01e5d90fc15f9 | bytes=393
- data/external/corpora/p100_miami/evidencia/s100a_repo_readme.md | sha256=7a9e0880210dd7e6 | bytes=3428
- data/external/corpora/p100_miami/evidencia/s100a_sample_ids_2020-05-02_es_fl.txt | sha256=b0017e04aef2cbbb | bytes=3560
- data/external/corpora/p100_miami/evidencia/zenodo_3824950_record.json | sha256=dde68a932f6fb378 | bytes=7778
- data/external/corpora/p100_chatsubs/LICENSE.txt | sha256=41003d4a74749c02 | bytes=19347
- data/external/corpora/p100_chatsubs/README.txt | sha256=bca7da93b70716e7 | bytes=1971
- data/external/corpora/p100_chatsubs/zenodo_record.json | sha256=f06c75c4e465dbb0 | bytes=6050
