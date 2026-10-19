# P93 - Auditoria de procedencia (P0/P1)

Fecha: 2026-10-13. Clasificacion de guia: **GUIDE_V2_1_CONFIRMED**.

## 1. P0 - Que guia recibieron los anotadores

- Las carpetas del paquete contienen UN unico archivo de guia:
  annotation_guidelines_v2_1_holdout_safe.md (identico byte a byte en A y B;
  hash interno ff699f54a48da779, hash completo fd8d4c65ccd194fb).
- El archivo v2 (annotation_guidelines_v2_frozen.md) NO esta en las carpetas;
  permanece solo como historico en reports/.
- La columna guideline_version de los CSV dice v2: la propia guia v2.1 define
  ese valor como la version de POLITICA (linea citada en la evidencia); no es
  metadata obsoleta ni una discrepancia.
- Evidencia de uso real: el orden de filas del archivo de B coincide
  EXACTAMENTE con el orden del paquete B; el de A coincide salvo una
  transposicion adyacente (indices 203-204). Los anotadores trabajaron sobre
  los CSV de estos paquetes.
- LIMITACION: no existe un acuse de recibo firmado; la confirmacion es a nivel
  de contenido preparado y verificado del paquete (pre-P93 y post-respuestas).

## 2. P1 - Validacion de los datos humanos

- P1_A_250: OK
- P1_B2_250: OK
- P1_membresia_igual: OK
- P1_A_tokens: OK
- P1_B_tokens: OK
- P1_B_vs_B_principal_identicos: OK
- P1_ids_en_clave: OK
- P1_componentes_100_150: OK
- P1_hashes_vs_manifiesto: OK
- Fuente de labels de B: response_B_sin_texto.csv (identico a response_B.csv
  en todas las columnas de anotacion y notas; verificado).
- Originales byte a byte: hashes de las 3 respuestas coinciden con el
  manifiesto de P92-Z.
- Artefactos generados: ASCII salvo las notas humanas verbatim en CSV internos
  (excepcion documentada; el verificador oficial no marca caracteres latinos).
