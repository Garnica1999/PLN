# Cobertura de etiquetado de areas (M3, Fase I)

Fecha: 2026-09-08 07:58 -0500
Documentos: 23742 (T/C/SU)
Tiempo total: 1.1s

## Regla de etiquetado (ADR-1)
Cada articulo citado se normaliza al canon de la CP: si el codigo con letra existe ("22A") se usa tal cual; si no, se reduce a su base ("3A"->"3"); lo fuera de 1-380 o transitorios sin correspondencia se ignora con contador. La primaria es el bloque mayoritario y el desempate usa el menor numero de articulo del documento (clave base entera, sufijo con orden vacio < A < B < C). Sin citas validas, el fallback busca el patron 'derecho a X' en el texto y asigna derechos_fundamentales con el termino X conservado en fallback_termino (flag fallback_derecho_a_X); sin coincidencia se asigna sin_area_definida.

## Cobertura global por via (% docs)
- cita_cp: 98.71 (23436/23742)
- fallback_derecho: 0.92 (218/23742)
- sin_area: 0.37 (88/23742)
- asignadas (cita o fallback): 99.63 (23654/23742)

## Por tipo T/C/SU (% por via)
| tipo | cita_cp | fallback | sin_area | n |
| --- | ---: | ---: | ---: | ---: |
| C | 100.00 | 0.00 | 0.00 | 6246 |
| SU | 99.50 | 0.25 | 0.25 | 397 |
| T | 98.22 | 1.27 | 0.51 | 17099 |

## Por periodo (% por via)
| periodo | cita_cp | fallback | sin_area | n |
| --- | ---: | ---: | ---: | ---: |
| 1992-1996 | 99.42 | 0.47 | 0.11 | 2739 |
| 1997-2001 | 95.39 | 3.12 | 1.49 | 4427 |
| 2002-2006 | 98.47 | 1.18 | 0.35 | 4918 |
| 2007-2011 | 99.81 | 0.16 | 0.02 | 4861 |
| 2012-2016 | 100.00 | 0.00 | 0.00 | 4360 |
| 2017-2021 | 99.92 | 0.04 | 0.04 | 2437 |

## Distribucion de areas (top-10)
- derechos_fundamentales: 10081 docs (42.46 %)
- principios_fundamentales: 6551 docs (27.59 %)
- derechos_sociales_economicos_culturales: 2861 docs (12.05 %)
- proteccion_aplicacion_derechos: 1252 docs (5.27 %)
- rama_judicial: 1167 docs (4.92 %)
- rama_legislativa: 712 docs (3.00 %)
- regimen_economico: 419 docs (1.76 %)
- rama_ejecutiva: 209 docs (0.88 %)
- organizacion_territorial: 201 docs (0.85 %)
- organizacion_estado: 120 docs (0.51 %)

## Tabla completa de areas (n_docs)
- derechos_fundamentales: 10081 docs (42.46 %)
- principios_fundamentales: 6551 docs (27.59 %)
- derechos_sociales_economicos_culturales: 2861 docs (12.05 %)
- proteccion_aplicacion_derechos: 1252 docs (5.27 %)
- rama_judicial: 1167 docs (4.92 %)
- rama_legislativa: 712 docs (3.00 %)
- regimen_economico: 419 docs (1.76 %)
- rama_ejecutiva: 209 docs (0.88 %)
- organizacion_territorial: 201 docs (0.85 %)
- organizacion_estado: 120 docs (0.51 %)
- sin_area_definida: 88 docs (0.37 %)
- organismos_control: 29 docs (0.12 %)
- derechos_colectivos_ambiente: 22 docs (0.09 %)
- habitantes_territorio: 10 docs (0.04 %)
- reforma_constitucion: 9 docs (0.04 %)
- participacion_democratica: 8 docs (0.03 %)
- elecciones: 3 docs (0.01 %)

## Tutelas en derechos_fundamentales
- 47.52 % (8125/17099 tutelas)

## Diagnostico de normalizacion de citas
- normalizadas exactas: 253394
- normalizadas a base: 168
- ignoradas fuera de rango: 0

## Ejemplos (10)
- id 23049 | derechos_fundamentales | fallback_derecho | sin citas CP validas; patron 'derecho a la' hacia derechos_fundamentales
- id 19235 | organizacion_territorial | cita_cp | arts. 86, 241, 31, 10, 7, 96 (+13 mas) con mayoria en organizacion_territorial
- id 21761 | derechos_sociales_economicos_culturales | cita_cp | arts. 46, 86, 241, 9, 6, 48 (+15 mas) con mayoria en derechos_sociales_economicos_culturales
- id 8613 | sin_area_definida | sin_area | sin citas CP validas ni patron 'derecho a ...' en el texto
- id 2555 | regimen_economico | cita_cp | arts. 1, 150, 253, 259, 279, 352 (+13 mas) con mayoria en regimen_economico
- id 5766 | derechos_fundamentales | fallback_derecho | sin citas CP validas; patron 'derecho a la' hacia derechos_fundamentales
- id 9156 | sin_area_definida | sin_area | sin citas CP validas ni patron 'derecho a ...' en el texto
- id 19954 | proteccion_aplicacion_derechos | cita_cp | arts. 86, 241, 99 con mayoria en proteccion_aplicacion_derechos
- id 15994 | principios_fundamentales | cita_cp | arts. 243, 37, 242, 86, 2, 13 (+14 mas) con mayoria en principios_fundamentales
- id 9530 | derechos_fundamentales | cita_cp | arts. 67, 86, 241, 13, 33, 44 (+4 mas) con mayoria en derechos_fundamentales

## Verificacion de areas vacias
Areas del universo CP sin documentos asignados (3): deberes, preambulo, transitorios.
Justificacion: preambulo y transitorios quedan vacios por diseno (el NER solo extrae articulos permanentes 1-380, sin citas a esos bloques); deberes (bloque de un solo articulo, el 95) si aparece citado y en 122 top-3, pero nunca alcanza la mayoria en ningun documento, por lo que no genera primarias.

## Criterio de aceptacion (>=95 % asignadas)
- SI: 99.63 % con area asignada (23654/23742); sin area 88 docs (0.37 %), todos sin citas CP validas ni patron 'derecho a ...' en el texto.

## Esquema del parquet
Columnas: id, tipo, ano, area_primaria, top3_json, via (cita_cp/fallback_derecho/sin_area), n_citas_cp, mas fallback_termino (extension documentada que conserva X del patron para la flag fallback_derecho_a_X; nula fuera de fallback).
