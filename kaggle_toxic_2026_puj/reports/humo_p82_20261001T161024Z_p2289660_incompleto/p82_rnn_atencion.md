# P82: RNN bidireccional con capa de atencion aditiva

Modo: humo. Marca de corrida: 20261001T161024Z_p2289660.

Inicio UTC: 2026-10-01T16:10:24.882395+00:00.

## 0. Dispositivo de calculo

Dispositivo: AMD Radeon RX 9070 XT (tipo cuda). Hilos fijados: 16. Precision mixta: NO USADA. Generador de permutaciones: cpu.

Fuente de la eleccion: 16 por medicion, no por conveniencia: con 30 hilos un ajuste tarda 3.22 s frente a 2.43 s con 16, un 32 por ciento mas, por sobrecarga de planificacion con tensores pequenos. La GPU con ROCm tarda 1.24 s por ajuste.

Coste medido por ajuste, en segundos: {"cpu_16_hilos": 2.43, "cpu_30_hilos": 3.22, "gpu_rocm": 1.24}.

Consecuencia declarada: los valores absolutos de los cinco brazos y las deltas pareadas son los de este dispositivo y no deben compararse con los de una corrida en otro. El pareamiento DENTRO de la corrida se conserva porque los cinco brazos se ejecutan en el mismo dispositivo

Parciales de esta corrida en: /home/carlos/Git/PLN-v67/kaggle_toxic_2026_puj/reports/humo_p82_20261001T161024Z_p2289660_incompleto/parciales. Los parciales del intento fallido en CPU se conservan en reports/p82_parciales y no se mezclan con estos, porque cada corrida usa su propio subdirectorio.

## 1. Formulacion

V03 remains the best verified leaderboard result.

Este informe NO afirma que Word2Vec sea mejor que V03, ni que una RNN lo supere, ni que la atencion aporte o no aporte. El limite permanente declarado por P81, el anclaje del objetivo del trainer contra una implementacion externa de SGNS, sigue ABIERTO y permanente.

NO_SE_AFIRMA_QUE_WORD2VEC_SEA_MEJOR_QUE_V03

## 2. La pregunta, y lo que este diseno NO puede responder

Si la mejora, si la hay, es menor que el limite de deteccion del diseno, este experimento NO LA VE. Un resultado NEGATIVO no se lee como que la atencion no aporta; se lee como que el efecto es inferior a lo que este diseno resuelve.

UN_RESULTADO_NEGATIVO_AQUI_NO_AFIRMA_QUE_LA_ATENCION_NO_APORTA_AFIRMA_QUE_EL_EFECTO_ESINFERIOR_AL_LIMITE_DE_DETECCION_DE_ESTE_DISENO

DIEZ_SEMILLAS_ES_EL_MAXIMO_QUE_CABE_NO_EL_OPTIMO_ESTADISTICO

Naturaleza del resultado: EXPLORATORIO.

## 3. Brazos, uno por fila y sin agregado

| brazo | nucleo | atencion | embedding | media | sd_semillas | params | params_atencion |
|---|---|---|---|---|---|---|---|
| B1 | GRU | NO | W2V_100d_congelado | 0.54448 | 0.112662 | 915 | 0 |
| B2 | GRU | SI | W2V_100d_congelado | 0.506536 | 0.111399 | 1243 | 280 |
| B3 | LSTM | NO | W2V_100d_congelado | 0.492273 | 0.091929 | 1203 | 0 |
| B4 | LSTM | SI | W2V_100d_congelado | 0.474653 | 0.064857 | 1531 | 280 |
| B5 | GRU | SI | W2V_100d_permutado | 0.50509 | 0.108486 | 1243 | 280 |

Coste parametrico de la capa de atencion. Referencia vigente, de la Enmienda 1: 16448. Valor del pre-registro original, ya corregido y NO vigente: 8256. Fuente de la referencia: docs/p82_enmienda_1.md sha256 53c8555d1a5ef0fcebb06c4a6f061acf56e568d95242e907387f0af1b03420d8.

| magnitud | valor |
|---|---|
| pesos de la atencion SIN sesgos, medidos | 264 |
| referencia de la Enmienda 1, sin sesgos | 16448 |
| total de la atencion CON sesgos, medidos | 280 |
| convencion | la referencia de la enmienda, 16448, cuenta PESOS SIN SESGOS, que es la misma convencion del 8256 pre-registrado: W_h de 128x64 mas W_s de 128x64 mas v de 64; el conteo con sesgos es 128 mas y se publica tambien |

La aritmetica original de 8256 asumia W_h de 64x64, es decir una celda UNIDIRECCIONAL de ancho 64. Los brazos de P82 son BIDIRECCIONALES con 64 por direccion, luego el estado que entra en la atencion tiene ancho 128 y W_h y W_s son de 128x64. La Enmienda 1 corrige el numero y este script publica las dos cifras, la pre-registrada y la real, y declara la discrepancia en vez de retocarla.

## 4. Comparaciones

| id | definicion | naturaleza | media | IC95 |
|---|---|---|---|---|
| P1 | B2 - B1 | PRIMARIA | -0.037944 | [-0.144342, 0.068454] |
| P2 | B4 - B3 | SECUNDARIA | -0.017621 | [-0.07347, 0.038229] |
| P3 | B2 - B5 | SECUNDARIA_DESCRIPTIVA | 0.001446 | [-0.002531, 0.005423] |
| P4 | B1 - B5 | DIAGNOSTICA_DESCRIPTIVA | 0.03939 | [-0.065762, 0.144542] |
| P5 | B1 - B3 | DIAGNOSTICA_DESCRIPTIVA | 0.052207 | [-0.058887, 0.1633] |
## 5. Limite de deteccion

Desviacion planificada: 0.021588. Desviacion MEDIDA en esta corrida: 0.14873429294696155. Gap con V03, leido del informe de P78 y no escrito a mano: 0.11745. MDE a n=10 con la desviacion medida: 0.14814.

### Tabla del pre-registro, recalculada y NO modificada

| n | MDE publicado | MDE recalculado | coincide | semiancho IC95 | potencia en el ancla |
|---|---|---|---|---|---|
| 3 | 0.070464 | 0.070464 | SI | 0.053628 | 0.2179 |
| 5 | 0.036311 | 0.036311 | SI | 0.026805 | 0.5 |
| 8 | 0.024956 | 0.024956 | SI | 0.018048 | 0.798 |
| 10 | 0.021502 | 0.021502 | SI | 0.015443 | 0.8993 |
| 12 | 0.019184 | 0.019184 | SI | 0.013716 | 0.9522 |
| 20 | 0.014258 | 0.014258 | SI | 0.010103 | 0.9983 |
| 30 | 0.011425 | 0.011425 | SI | 0.008061 | 1.0 |
| 50 | 0.008726 | 0.008726 | SI | 0.006135 | 1.0 |

### Sensibilidad con la desviacion MEDIDA

| factor | sd supuesta | MDE | potencia en el ancla | MDE/gap |
|---|---|---|---|---|
| 1.0 | 0.148734 | 0.14814 | 0.0762 | 1.26 |
| 1.5 | 0.223101 | 0.222209 | 0.0616 | 1.89 |
| 2.0 | 0.297469 | 0.296279 | 0.0565 | 2.52 |
| 3.0 | 0.446203 | 0.444419 | 0.0529 | 3.78 |

### Reproduccion del intervalo ya publicado por P78

| intervalo calculado | intervalo publicado | diferencia maxima |
|---|---|---|
| [-0.028735, 0.078521] | [-0.028736, 0.078522] | 1.000000000001e-06 |

## 6. Serie de deltas por semilla del primario

| orden | semilla | delta | signo |
|---|---|---|---|
| 1 | 20261401 | -0.222222 | NEGATIVO |
| 2 | 20261402 | -0.222222 | NEGATIVO |
| 3 | 20261403 | -0.035088 | NEGATIVO |
| 4 | 20261404 | 0.041270 | POSITIVO |
| 5 | 20261405 | 0.000000 | CERO |
| 6 | 20261406 | 0.000000 | CERO |
| 7 | 20261407 | -0.045752 | NEGATIVO |
| 8 | 20261408 | 0.222222 | POSITIVO |
| 9 | 20261409 | -0.222222 | NEGATIVO |
| 10 | 20261410 | 0.104575 | POSITIVO |

Conteo de signos positivos: 3. Media: -0.037944. Desviacion: 0.148734. Metodo del intervalo que decide: t_pareado_sobre_semillas, t de Student sobre los deltas POR SEMILLA. No es bootstrap.

## 7. Cobertura medida

| cifra | valor |
|---|---|
| cifras | {'proporcion_posiciones_oov': 0.10694444444444444, 'proporcion_posiciones_relleno': 0.6513888888888889, 'proporcion_oov_sobre_tokens_que_llegan': 0.30677290836653387, 'tipos_distintos_fuera_de_vocabulario': 10, 'longitud_mediana_tokens_por_fila': 3.0, 'longitud_media_tokens_por_fila': 5.783333333333333, 'max_len': 12, 'fraccion_filas_truncadas': 0.2, 'n_filas_sin_tokens': 12, 'convencion_de_fila_vacia': 'una fila con cero tokens tiene longitud efectiva 1 para el empaquetado y para el softmax, de modo que su unica posicion recibe el peso; se publica cuantas hay', 'reproduccion_referencia_relleno_de_p81': 'NO_REPRODUCIDA', 'reproduccion_referencia_oov_de_p81': 'NO_REPRODUCIDA', 'referencia_relleno_p81': 0.7309191672285907, 'referencia_oov_p81': 0.153295, 'tokenizador': 'tokenizar(normalizar_emb_normal(texto)) recortado a MAX_LEN; es el tokenizador de P82 y las cifras de A8 se miden con el, no se arrastran'} |
| comprobaciones | {'ocho_cifras_presentes': True, 'todas_finitas': True, 'proporciones_en_rango': True, 'linea_de_reproduccion_declarada': True, 'max_len_positivo': True} |
| faltan | [] |
| n_cifras_exigidas | 8 |
| no_finitas | [] |
| proporciones_fuera_de_rango | [] |

## 8. Interseccion y rescue damage net contra V03

| etiqueta | aciertos comunes | rescate | dano | neto |
|---|---|---|---|---|
| cifras | None | None | None | None |
| comprobaciones | None | None | None | None |
| signo_del_neto | None | None | None | None |

Estas cifras NO son comparables con corrected_cv ni con ningun macro-F1. Rescue damage net es un recuento de celdas, no una mejora de metrica.

## 9. Lecturas por fuente

| fuente | contador |
|---|---|
| artefacto_v03_json | umbrales y metodo |
| artefacto_v03_oof | umbrales |
| artefacto_w2v | solo hash y claves en humo; el entrenamiento es sintetico |
| filas_de_desarrollo_usadas | 60 |
| filas_holdout_leidas | 0 |
| filas_test_leidas | 0 |
| informe_p78_addendum | solo el gap |
| read_train_filas | 0 |

## 10. Cronometro

| clave | valor |
|---|---|
| ajustes_de_red | 310 |
| budget_segundos | 5400 |
| escrituras_fallidas | 0 |
| etapa | humo:INCOMPLETO |
| tiempo_real_segundos | 23.951219 |
## 11. Las veintitres afirmaciones

| id | bloque | estado | stop |
|---|---|---|---|
| A0 | 0 | CONFIRMADO | si |
| A1 | 0 | CONFIRMADO | si |
| A10 | 2 | CONFIRMADO | si |
| A11 | 2 | CONFIRMADO | si |
| A12 | 2 | CONFIRMADO | si |
| A13 | 3 | CONFIRMADO | si |
| A14 | 3 | CONFIRMADO | si |
| A15 | 3 | CONFIRMADO | si |
| A16 | 3 | CONFIRMADO | no |
| A17 | 4 | CONFIRMADO | no |
| A18 | 4 | CONFIRMADO | si |
| A19 | 4 | CONFIRMADO | si |
| A2 | 0 | CONFIRMADO | si |
| A20 | 4 | CONFIRMADO | si |
| A21 | 4 | CONFIRMADO | no |
| A22 | 4 | CONFIRMADO | si |
| A3 | 1 | CONFIRMADO | si |
| A4 | 1 | CONFIRMADO | si |
| A5 | 1 | CONFIRMADO | si |
| A6 | 1 | CONFIRMADO | si |
| A7 | 1 | CONFIRMADO | si |
| A8 | 1 | CONFIRMADO | no |
| A9 | 2 | CONFIRMADO | si |

ERROR no es ni CONFIRMADO ni REFUTADO ni NO_EVALUABLE: significa que la afirmacion no llego a completarse por fallo PROPIO del script, y la corrida continuo con la siguiente. Lo unico que aborta de verdad es una Parada con STOP.

## 12. Contrato entre capas y auditoria de accesos por literal

| callable | origen | estado |
|---|---|---|
| metricas_multietiqueta | src/evaluation.py | ok |
| read_train | src/data.py | ok |
| holdout_v2 | src/splits.py | ok |
| pliegues_corregidos | src/splits.py | ok |
| tokenizar | src/embedding_utils.py | ok |
| normalizar_emb_normal | src/embedding_utils.py | ok |
| pack_padded_sequence | torch.nn.utils.rnn | ok |
| pad_packed_sequence | torch.nn.utils.rnn | ok |
| clip_grad_norm_ | torch.nn.utils | ok |
| epoca_estrella | paridad con p75_01 epoca_estrella | ok |
| entrenar_fijo | paridad con p75_01 entrenar_fijo | ok |

Accesos por literal a claves de contrato en este fichero: 0.


V03 remains the best verified leaderboard result.
