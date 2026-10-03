# V29-02: similitudes mas laxas, cobertura, pureza y relacion similitud-etiqueta

Generado por `experiments/v29_02_similitudes_laxas.py` (V29-02 (similitudes mas laxas: cobertura, pureza y relacion similitud-etiqueta)).

**Estado `DESCRIPTIVE_ONLY`. Alcance: diagnostico descriptivo. No se entrena nada, no se construye predictor y no se produce submission. Las etiquetas solo se usan para MEDIR la pureza, nunca entran en el calculo de la similitud**

## 1. Los cinco metodos

| clave | metodo |
|---|---|
| `A_coseno_char4_historico` | coseno TF-IDF char_wb 4-4, normalizacion historica; es el metodo de V03 |
| `B_coseno_char4_agresivo` | coseno TF-IDF char_wb 4-4, normalizacion agresiva |
| `C_coseno_char35_agresivo` | coseno TF-IDF char_wb 3-5, normalizacion agresiva |
| `D_jaccard_char35_agresivo` | Jaccard de char 3-5 binarios, normalizacion agresiva |
| `E_jaccard_tokens_agresivo` | Jaccard de unigramas de palabra binarios, normalizacion agresiva |

Sin MinHash ni LSH: con 1731 filas hay 1 497 315 pares y la matriz densa ocupa 12 MB; un indice aproximado no aporta nada y anade una fuente de error.

Sin distancia de edicion: la directiva lo admite solo para candidatos prefiltrados. Este script es descriptivo y no lo necesita; queda como paso natural si la Fase 5 encuentra senal.

## 2. Controles

| metodo | auto | simetria | rango | no finitos |
|---|---:|---:|---|---:|
| `A_coseno_char4_historico` | 0.0e+00 | 0.0e+00 | [0.000, 1.000] | 0 |
| `B_coseno_char4_agresivo` | 1.0e+00 | 0.0e+00 | [0.000, 1.000] | 0 |
| `C_coseno_char35_agresivo` | 1.0e+00 | 0.0e+00 | [0.000, 1.000] | 0 |
| `D_jaccard_char35_agresivo` | 1.0e+00 | 0.0e+00 | [0.000, 1.000] | 0 |
| `E_jaccard_tokens_agresivo` | 1.0e+00 | 0.0e+00 | [0.000, 1.000] | 0 |

Jaccard explicito contra la matriz, error maximo: `2.89e-09`.

## 3. Cobertura por metodo y umbral

| metodo | u0.95 | u0.90 | u0.85 | u0.80 | u0.75 | u0.70 | u0.60 | u0.50 | u0.40 | u0.30 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `A_coseno_char4_historico` | 0.0006 | 0.0006 | 0.0012 | 0.0017 | 0.0029 | 0.0035 | 0.0052 | 0.0098 | 0.0300 | 0.0485 |
| `B_coseno_char4_agresivo` | 0.0058 | 0.0052 | 0.0046 | 0.0052 | 0.0069 | 0.0081 | 0.0092 | 0.0110 | 0.0225 | 0.0324 |
| `C_coseno_char35_agresivo` | 0.0058 | 0.0052 | 0.0046 | 0.0058 | 0.0069 | 0.0087 | 0.0081 | 0.0098 | 0.0231 | 0.0324 |
| `D_jaccard_char35_agresivo` | 0.0052 | 0.0052 | 0.0052 | 0.0064 | 0.0075 | 0.0087 | 0.0069 | 0.0075 | 0.0092 | 0.0191 |
| `E_jaccard_tokens_agresivo` | 0.0040 | 0.0040 | 0.0040 | 0.0058 | 0.0064 | 0.0069 | 0.0092 | 0.0179 | 0.0196 | 0.0214 |

## 4. La relacion similitud -> misma etiqueta

La linea base es la que se obtiene suponiendo las dos etiquetas independientes, es decir
`p^2 + (1-p)^2` con `p` la frecuencia del positivo. Una franja solo es interesante si la
probabilidad de compartir etiqueta supera esa linea base **y** tiene volumen de pares.

### `A_coseno_char4_historico`

| franja | HS | TR | AG |
|---|---:|---:|---:|
| 0.95_1.01 | 1.0000 (n=3) | 1.0000 (n=3) | 1.0000 (n=3) |
| 0.90_0.95 | - | - | - |
| 0.85_0.90 | 0.5000 (n=2) | 0.5000 (n=2) | 0.5000 (n=2) |
| 0.80_0.85 | 1.0000 (n=2) | 1.0000 (n=2) | 1.0000 (n=2) |
| 0.70_0.80 | 0.5217 (n=23) | 0.5652 (n=23) | 0.5217 (n=23) |
| 0.60_0.70 | 0.5882 (n=51) | 0.5882 (n=51) | 0.5882 (n=51) |
| 0.50_0.60 | 0.6116 (n=121) | 0.6116 (n=121) | 0.6198 (n=121) |
| 0.40_0.50 | 0.6716 (n=271) | 0.6863 (n=271) | 0.6716 (n=271) |
| 0.30_0.40 | 0.6189 (n=614) | 0.6645 (n=614) | 0.6433 (n=614) |
| 0.00_0.30 | 0.5076 (n=1496228) | 0.5798 (n=1496228) | 0.5273 (n=1496228) |

### `B_coseno_char4_agresivo`

| franja | HS | TR | AG |
|---|---:|---:|---:|
| 0.95_1.01 | 1.0000 (n=24) | 1.0000 (n=24) | 1.0000 (n=24) |
| 0.90_0.95 | 0.7857 (n=14) | 0.7857 (n=14) | 0.7857 (n=14) |
| 0.85_0.90 | 0.8696 (n=23) | 0.8696 (n=23) | 0.8696 (n=23) |
| 0.80_0.85 | 0.7273 (n=22) | 0.7273 (n=22) | 0.7273 (n=22) |
| 0.70_0.80 | 0.7451 (n=102) | 0.7451 (n=102) | 0.7353 (n=102) |
| 0.60_0.70 | 0.7545 (n=220) | 0.7545 (n=220) | 0.7591 (n=220) |
| 0.50_0.60 | 0.6917 (n=399) | 0.6867 (n=399) | 0.6942 (n=399) |
| 0.40_0.50 | 0.6882 (n=789) | 0.6831 (n=789) | 0.6869 (n=789) |
| 0.30_0.40 | 0.6749 (n=1821) | 0.6826 (n=1821) | 0.6568 (n=1821) |
| 0.00_0.30 | 0.5073 (n=1493901) | 0.5796 (n=1493901) | 0.5270 (n=1493901) |

### `C_coseno_char35_agresivo`

| franja | HS | TR | AG |
|---|---:|---:|---:|
| 0.95_1.01 | 1.0000 (n=24) | 1.0000 (n=24) | 1.0000 (n=24) |
| 0.90_0.95 | 0.7857 (n=14) | 0.7857 (n=14) | 0.7857 (n=14) |
| 0.85_0.90 | 0.8800 (n=25) | 0.8800 (n=25) | 0.8800 (n=25) |
| 0.80_0.85 | 0.7407 (n=27) | 0.7407 (n=27) | 0.7407 (n=27) |
| 0.70_0.80 | 0.7561 (n=123) | 0.7561 (n=123) | 0.7480 (n=123) |
| 0.60_0.70 | 0.7325 (n=228) | 0.7325 (n=228) | 0.7368 (n=228) |
| 0.50_0.60 | 0.7066 (n=467) | 0.7024 (n=467) | 0.7152 (n=467) |
| 0.40_0.50 | 0.6569 (n=819) | 0.6508 (n=819) | 0.6459 (n=819) |
| 0.30_0.40 | 0.6770 (n=2074) | 0.6837 (n=2074) | 0.6630 (n=2074) |
| 0.00_0.30 | 0.5073 (n=1493514) | 0.5796 (n=1493514) | 0.5270 (n=1493514) |

### `D_jaccard_char35_agresivo`

| franja | HS | TR | AG |
|---|---:|---:|---:|
| 0.95_1.01 | 1.0000 (n=22) | 1.0000 (n=22) | 1.0000 (n=22) |
| 0.90_0.95 | 1.0000 (n=17) | 1.0000 (n=17) | 1.0000 (n=17) |
| 0.85_0.90 | - | - | - |
| 0.80_0.85 | 0.6667 (n=3) | 0.6667 (n=3) | 0.6667 (n=3) |
| 0.70_0.80 | 0.8000 (n=20) | 0.8000 (n=20) | 0.7000 (n=20) |
| 0.60_0.70 | 0.8455 (n=123) | 0.8455 (n=123) | 0.8455 (n=123) |
| 0.50_0.60 | 0.8405 (n=257) | 0.8405 (n=257) | 0.8093 (n=257) |
| 0.40_0.50 | 0.7933 (n=653) | 0.7933 (n=653) | 0.7596 (n=653) |
| 0.30_0.40 | 0.7422 (n=2161) | 0.7427 (n=2161) | 0.7099 (n=2161) |
| 0.00_0.30 | 0.5072 (n=1494059) | 0.5795 (n=1494059) | 0.5269 (n=1494059) |

### `E_jaccard_tokens_agresivo`

| franja | HS | TR | AG |
|---|---:|---:|---:|
| 0.95_1.01 | 1.0000 (n=20) | 1.0000 (n=20) | 1.0000 (n=20) |
| 0.90_0.95 | - | - | - |
| 0.85_0.90 | - | - | - |
| 0.80_0.85 | 0.4000 (n=5) | 0.4000 (n=5) | 0.6000 (n=5) |
| 0.70_0.80 | 1.0000 (n=4) | 1.0000 (n=4) | 0.7500 (n=4) |
| 0.60_0.70 | 0.9560 (n=91) | 0.9560 (n=91) | 0.9121 (n=91) |
| 0.50_0.60 | 0.9022 (n=184) | 0.9076 (n=184) | 0.8641 (n=184) |
| 0.40_0.50 | 0.7960 (n=348) | 0.8017 (n=348) | 0.7644 (n=348) |
| 0.30_0.40 | 0.6881 (n=1651) | 0.7311 (n=1651) | 0.6651 (n=1651) |
| 0.00_0.30 | 0.5074 (n=1495012) | 0.5796 (n=1495012) | 0.5271 (n=1495012) |

## 5. La pregunta de la directiva: A, B o C

Criterio declarado por adelantado: A_RUIDO si ninguna franja mejora la linea base; B si la mejora existe pero la cobertura se queda por debajo del 10 %; C si la mejora existe con cobertura suficiente. Se exige al menos 200 pares por franja para considerar una.

| metodo | veredicto |
|---|---|
| `A_coseno_char4_historico` | **B_GRUPOS_PUROS_POCA_COBERTURA** |
| `B_coseno_char4_agresivo` | **B_GRUPOS_PUROS_POCA_COBERTURA** |
| `C_coseno_char35_agresivo` | **B_GRUPOS_PUROS_POCA_COBERTURA** |
| `D_jaccard_char35_agresivo` | **B_GRUPOS_PUROS_POCA_COBERTURA** |
| `E_jaccard_tokens_agresivo` | **B_GRUPOS_PUROS_POCA_COBERTURA** |

La mejor franja de cada metodo y label:

| metodo | label | franja | prob. misma etiqueta | linea base | mejora | pares |
|---|---|---|---:|---:|---:|---:|
| `A_coseno_char4_historico` | HS | 0.40_0.50 | 0.6716 | 0.5080 | **+0.1636** | 271 |
| `A_coseno_char4_historico` | TR | 0.40_0.50 | 0.6863 | 0.5801 | **+0.1062** | 271 |
| `A_coseno_char4_historico` | AG | 0.40_0.50 | 0.6716 | 0.5276 | **+0.1439** | 271 |
| `B_coseno_char4_agresivo` | HS | 0.60_0.70 | 0.7545 | 0.5080 | **+0.2465** | 220 |
| `B_coseno_char4_agresivo` | TR | 0.60_0.70 | 0.7545 | 0.5801 | **+0.1744** | 220 |
| `B_coseno_char4_agresivo` | AG | 0.60_0.70 | 0.7591 | 0.5276 | **+0.2314** | 220 |
| `C_coseno_char35_agresivo` | HS | 0.60_0.70 | 0.7325 | 0.5080 | **+0.2245** | 228 |
| `C_coseno_char35_agresivo` | TR | 0.60_0.70 | 0.7325 | 0.5801 | **+0.1523** | 228 |
| `C_coseno_char35_agresivo` | AG | 0.60_0.70 | 0.7368 | 0.5276 | **+0.2092** | 228 |
| `D_jaccard_char35_agresivo` | HS | 0.50_0.60 | 0.8405 | 0.5080 | **+0.3325** | 257 |
| `D_jaccard_char35_agresivo` | TR | 0.50_0.60 | 0.8405 | 0.5801 | **+0.2603** | 257 |
| `D_jaccard_char35_agresivo` | AG | 0.50_0.60 | 0.8093 | 0.5276 | **+0.2817** | 257 |
| `E_jaccard_tokens_agresivo` | HS | 0.40_0.50 | 0.7960 | 0.5080 | **+0.2880** | 348 |
| `E_jaccard_tokens_agresivo` | TR | 0.40_0.50 | 0.8017 | 0.5801 | **+0.2216** | 348 |
| `E_jaccard_tokens_agresivo` | AG | 0.40_0.50 | 0.7644 | 0.5276 | **+0.2367** | 348 |

## 6. Lo que este script no hace

- No entrena ningun modelo.
- No construye predictor ni senal para un modelo.
- No lee el test.
- No produce submission.
- Las etiquetas se usan solo para medir pureza, nunca en el calculo de similitud.

Coste: 6.6 s, memoria 373 MB, 0 modelos, 0 submissions.

