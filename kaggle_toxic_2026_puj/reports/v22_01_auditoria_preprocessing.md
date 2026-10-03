# V22-01 (P0): auditoria del preprocessing del corpus frente al de la tarea

Generado por `experiments/v22_01_auditoria_preprocessing.py` (V22-01 (P0: auditoria del preprocessing del corpus frente al de la tarea)).

**Ningun modelo entrenado. Ninguna submission. El test solo se transformo.**

## 1. Las dos cadenas, tal como estan en el codigo

| paso | corpus | texto de la tarea |
|---|---|---|
| limpieza | `limpiar_marcado` (MediaWiki) | no aplica |
| normalizacion | `normalizar_emb_normal` | `normalizar_emb_normal` |
| tokenizacion | `tokenizar` | `tokenizar` dentro de `vectores_documento` |

`embedding_13_window_v5_1.py` linea 226 aplica `normalizar_emb_normal` a
`desarrollo['text']` antes de construir los vectores de documento. La normalizacion
del corpus y la del texto de la tarea **son la misma funcion**.

## 2. Las dos cadenas medidas sobre los mismos textos

| espacio | tokens | tipos | cobertura tipo | cobertura token | long. media | filas sin tokens |
|---|---:|---:|---:|---:|---:|---:|
| desarrollo_raw | 26996 | 8080 | 0.5501 | 0.7414 | 18.20 | 0 |
| desarrollo_consistent | 25539 | 5447 | 1.0000 | 1.0000 | 17.22 | 0 |
| test_raw | 20284 | 6587 | 0.2721 | 0.6403 | 17.58 | 0 |
| test_consistent | 19204 | 4503 | 0.4457 | 0.8554 | 16.64 | 0 |

## 3. El mismatch, token a token (desarrollo)

- Tipos que solo aparecen en RAW: **3635**.
- Tipos que solo aparecen en CONSISTENT: **1002**.
- Tipos comunes: **4445**.
- Delta de cobertura por token (CONSISTENT - RAW): **+0.2586**.
- Delta de cobertura por tipo (CONSISTENT - RAW): **+0.4499**.

**Veredicto de P0: EVIDENCIADO.**

## 4. Marcas de Twitter

| marca | desarrollo crudo | desarrollo normalizado | test crudo | test normalizado |
|---|---:|---:|---:|---:|
| hashtags | 255 | 140 | 235 | 120 |
| menciones | 1081 | 345 | 799 | 252 |
| URLs | 493 | 65 | 362 | 65 |
| numeros | 1063 | 187 | 857 | 168 |
| elongaciones | 105 | preservadas | 68 | preservadas |

## 5. Los 20 OOV mas frecuentes del desarrollo

| token | RAW | CONSISTENT |
|---|---:|---:|
| https | 489 | 0 |
| Y | 85 | 0 |
| PUTA | 84 | 0 |
| A | 77 | 0 |
| Callate | 72 | 0 |
| Cllate | 70 | 0 |
| LA | 66 | 0 |
| QUE | 66 | 0 |
| DE | 65 | 0 |
| No | 63 | 0 |
| El | 57 | 0 |
| La | 54 | 0 |
| Tu | 50 | 0 |
| NO | 44 | 0 |
| Que | 43 | 0 |
| TU | 35 | 0 |
| Los | 33 | 0 |
| Si | 32 | 0 |
| CALLATE | 32 | 0 |
| TE | 31 | 0 |

## 6. Reproducibilidad

- Diccionario de vectores: SIFTS del binario de fastText, 6059 tipos consultados, 6059 con vector util, 0 nulos descartados.
- Test usado solo como transformacion de texto para medir cobertura. No se leen
  etiquetas, no se entrena, no se ajusta nada.
- Completado en 0.2 s.

