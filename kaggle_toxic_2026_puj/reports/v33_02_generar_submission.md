# V33-02: submission de V32-05 generada y verificada

Generado por `experiments/v33_02_generar_submission.py` (V33-02 (generar la submission de V32-05 y verificar los checks)).

**Estado `SUBMISSION_GENERATED`. Alcance: fases 9 a 11 de la directiva V33. No se envia a Kaggle en este script**

## 1. El pipeline

- word(1,2) + char_wb(3,5) TF-IDF ajustado sobre el train completo
- 3 bins con cuantiles del train: (11.0, 19.391000000000076)
- interacciones: url_x_b0, url_x_b1, url_x_b2, url_query_x_largo, url_corta_x_corto
- 3x LogisticRegression C=2.0 balanced liblinear max_iter=1000 random_state=42
- umbrales congelados: {'HS': 0.44, 'TR': 0.51, 'AG': 0.46}

## 2. Anti-leakage

- vocabulario_ajustado_solo_con_train: si
- bins_derivados_solo_del_train: si
- interacciones_sin_labels: si
- test_no_usado_para_seleccion: si

## 3. Los checks del CSV

| check | resultado |
|---|:--:|
| exactamente_1154_filas | OK |
| ids_unicos | OK |
| sin_duplicados | OK |
| sin_nan_en_predicciones | OK |
| solo_valores_0_o_1 | OK |
| columnas_exactas | OK |
| orden_correcto | OK |

## 4. El fichero

- ruta: `submissions/toxic_submission_v32_05_url_length.csv`
- filas: 1154
- tamano: 13860 bytes
- sha256: `7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e`

## 5. Comparacion contra V03 en test

**0 filas cambiadas (0.00 % del test).**

| label | cambios | 0->1 | 1->0 |
|---|---:|---:|---:|
| HS | 0 | 0 | 0 |
| TR | 0 | 0 | 0 |
| AG | 0 | 0 | 0 |

## 6. Lo que este script no hace

- No envia a Kaggle: eso es V33-03.
- No modifica V03 ni V32-05.

Coste: 0.4 s, memoria 229 MB, 3 modelos, 0 submissions.

