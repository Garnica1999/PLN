# V18-01: conjunto de cambio y senal de aceptacion

Generado por `experiments/v18_01_cambio_y_senal.py` (V18-01 (P1 conjunto de cambio y P2 senal de aceptacion)).

**No se entrena nada, no se construye ninguna regla y no se gasta ninguna submission.**

## 1. Los conjuntos de cambio, uno por challenger (P1)

Cada challenger propone un conjunto distinto, con celdas distintas, y su veredicto tiene que
sostenerse en su propio conjunto. Agregarlos exigiria que coincidieran las celdas, que es
justo lo que no pasa.

| challenger | cambia | filas | rescata | dana | neto | P(rescate) |
| --- | --- | --- | --- | --- | --- | --- |
| oof_candidates_v4/C_meta36_downweight | 202 | 173 | 137 | 65 | +72 | 0.678 |
| oof_ens04_v04/proba_ens | 287 | 237 | 173 | 114 | +59 | 0.603 |
| oof_meta_shift_v4/meta_control | 223 | 184 | 131 | 92 | +39 | 0.587 |
| oof_mod03b/proba | 271 | 221 | 161 | 110 | +51 | 0.594 |
| oof_candidates_v4/B_meta36 | 202 | 176 | 120 | 82 | +38 | 0.594 |

## 2. Lo que se midio, y por que el objetivo no es la etiqueta

El objetivo de cada feature es si el **cambio propuesto** es correcto frente a V03: 1 es
rescate y 0 es dano. Predecir la etiqueta es otra pregunta, y es la que ya se midio en V15
sin que eso produjera nada. Aqui las dos clases son los desacuerdos entre dos modelos, y
lo que decide es cual de los dos se equivoco.

## 3. El test global de cada challenger, que es el que decide

El estadistico es el maximo de |AUC - 0.5| sobre todas las reglas, y su nula se obtiene
permutando las etiquetas 1000 veces. Es un test exacto bajo la nula global, asi
que no hace falta corregir a mano por el numero de pruebas. Un regimen necesita al menos
30 celdas para que su proporcion tenga precision suficiente.

| challenger | regimenes | maximo desvio | p global | mejor regimen | mejor AUC |
| --- | --- | --- | --- | --- | --- |
| oof_candidates_v4/C_meta36_downweight | 371 | 0.1181 | 0.0989 | diferencia_abs_de_scores>=0.1308 | 0.6181 |
| oof_ens04_v04/proba_ens | 408 | 0.1036 | 0.0380 | aux_n_plurales<=1.0000 | 0.6036 |
| oof_meta_shift_v4/meta_control | 364 | 0.1025 | 0.1608 | aux_pos_NOUN_frac>=0.1667 | 0.3975 |
| oof_mod03b/proba | 408 | 0.1107 | 0.0200 | aux_n_plurales<=1.0000 | 0.6107 |
| oof_candidates_v4/B_meta36 | 363 | 0.0924 | 0.4286 | diferencia_abs_de_scores>=0.0688 | 0.5924 |

## 4. Lo que la regla daria fuera de muestra

La regla se elige en cuatro pliegues y se evalua en el quinto, rotando los cinco. La
mejora es sobre el neto de aceptar todos los cambios, que es la referencia real: una regla
que no mejora esa cifra no aporta nada, por buena que parezca la separacion.

| challenger | caracteristica | mejora media | desviacion | pliegues que mejoran |
| --- | --- | --- | --- | --- |
| C_meta36_downweight | aux_n_oraciones | +0.00 | 0.00 | 0/5 |
| C_meta36_downweight | aux_n_menciones | +0.00 | 0.00 | 0/5 |
| C_meta36_downweight | aux_n_mencion_interna | +0.00 | 0.00 | 0/5 |
| proba_ens | aux_n_plurales | +2.20 | 2.49 | 3/5 |
| proba_ens | aux_plural_generico==0 | +1.20 | 3.27 | 4/5 |
| proba_ens | aux_dep_cc_n | +1.00 | 2.45 | 3/5 |
| meta_control | aux_pos_NOUN_frac | +1.00 | 5.20 | 3/5 |
| meta_control | aux_plural_densidad | +0.80 | 1.48 | 3/5 |
| meta_control | aux_pos_PROPN_frac | +0.40 | 5.03 | 2/5 |
| proba | aux_n_plurales | +2.40 | 3.29 | 3/5 |
| proba | aux_plural_generico==0 | +1.40 | 3.78 | 4/5 |
| proba | aux_comillas_dobles | +1.40 | 1.14 | 4/5 |
| B_meta36 | diferencia_abs_de_scores | +1.40 | 3.65 | 3/5 |
| B_meta36 | aux_col_segunda_persona==0 | +0.40 | 1.34 | 3/5 |
| B_meta36 | aux_pos_DET_frac | +0.40 | 2.41 | 3/5 |

## 5. Veredicto

**NO SELECTIVE CORRECTION SIGNAL**

- Challengers con p nominal por debajo de 0.05: **2 de 5**.
- Challengers que sobreviven a la correccion de Bonferroni entre challengers, con umbral 0.0100: **0**.
- Mejor valor p global: **0.0200**.
- Mejor mejora de neto fuera de muestra: **+2.40 celdas**, frente a las **27** que hacen falta para que una regla de este tamano pueda mover el leaderboard.
- Submissions gastadas: **0**.


Fecha: 2026-09-27T02:58:56+00:00. Duracion: 47.3 s.
