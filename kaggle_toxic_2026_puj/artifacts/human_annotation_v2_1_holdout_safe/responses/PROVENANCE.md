# Procedencia de las respuestas principales (recepcion 2026-10-13)

Dato humano; se conserva SIN CAMBIOS (las notas pueden contener caracteres
no ASCII del texto original de los anotadores; no se convierten).

- response_A.csv: transcrito verbatim del archivo entregado por el anotador A
  y reenviado por Carlos en mensaje de 2026-10-13. Formato recibido: 7
  columnas SIN la columna text (desviacion de esquema respecto al paquete;
  anotaciones y notas intactas).
- response_B.csv: idem para el anotador B; 8 columnas CON text (formato del
  paquete, con el estilo de comillas del export original).
- response_B_sin_texto.csv: segundo archivo entregado por B (7 columnas,
  sin text); se conserva verbatim; cotejo contra response_B.csv en el informe.
- Los textos de response_B.csv se cotejaron contra la fuente elegible
  (toxic_train.csv): mayoria exactos; un subconjunto con artefactos de
  export (comillas/espacios) y variantes residuales menores, listados en el
  informe de recepcion. Las ANOTACIONES no se ven afectadas; el derivado
  completado usa los textos de la fuente.
- La muestra corresponde a la MUESTRA PRINCIPAL (250 tweets P92W-0001..0250),
  no a la calibracion de 15 (P92W-C01..C15) preparada en P92-Y.
- Derivados que NO sustituyen a los originales:
  annotator_A_completado.csv / annotator_B_completado.csv (esquema del
  paquete, orden del paquete, anotaciones unidas por review_id).

Hashes (sha256 completo):

- annotator_A_completado.csv: 27328e91e41f1f9ac8ce325b9bae773cf3b0dff0edfca28b400589b6cde782bc
- annotator_B_completado.csv: c06f846426bf34e4b362db5b330ca247f1d987e283996a56272c0a4a0a8e8067
- response_A.csv: 599832fc398940078aa28fc074286db74a123fbee778eec4b2c6885605ff8a3a
- response_B.csv: fc213a143e72517f2cf04ded265cc457a1e413e067bda3466b789351653f8480
- response_B_sin_texto.csv: 770a87353eb1db9d64a4e361173b5f52997cef2905c3e4099943407f016d763f
