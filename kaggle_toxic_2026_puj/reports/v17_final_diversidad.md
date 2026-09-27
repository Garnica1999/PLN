# V17: busqueda de diversidad de errores en el patrimonio historico

Generado por `experiments/v17_01_informe_final.py` (V17-01 (informe final de la busqueda de diversidad de errores)).

**No se entrena nada, no se regenera nada y no se gasta ninguna submission.**

## 1. Primera pregunta: rescate y dano, modelo por modelo

Cada variante se mide en dos puntos de operacion: el umbral que declara el artefacto, que
es su version fiel, y el umbral que maximiza su rescate neto, que es su version mas
favorable. Rescates, danos y neto se reportan **siempre en el optimo**, que es el que
decide. Un modelo que solo tiene complementariedad con un umbral exotico y pierde con el suyo
no es un candidato.

- Rejilla: **4449 celdas**, de las cuales V03 falla **757**.
- Variantes medidas: **90**, sobre catorce artefactos OOF y veintidos archivos de prediccion de test ya existentes.
- Variantes que superan a V03 en el OOF con su umbral optimo: **14**.
- Mejor rescate neto individual: **+72 celdas**, con 137 rescates contra 65 danos en su umbral optimo 0.47/0.44/0.50, en `oof_candidates_v4/C_meta36_downweight`.

Agrupadas segun el P2: grupo A 14, grupo B 31, grupo C 45.

## 2. Segunda pregunta: hay alguna combinacion con rescate neto suficiente?

**NO. El rescate neto de la union de las ocho mejores variantes es +39 celdas, que es PEOR que el de cualquiera de las ocho por separado, y muy por debajo de las 107 que hacen falta para mover el leaderboard 0.003. La razon esta medida: el solape de los rescates es del 77 por ciento, luego los modelos fallan en las mismas filas y el dano se acumula mas rapido que el rescate.**

| magnitud | celdas |
| --- | ---: |
| rescates de las ocho mejores, sumados | 1126 |
| rescates de la union de las ocho | 263 |
| danios de la union | 224 |
| rescate neto de la union | +39 |
| solape de los rescates | 77 por ciento |
| necesarias para mover el LB 0.003 | 107 |

El dato relevante es que **la union es peor que el mejor componente individual**: combinar
ocho modelos no acumula rescate, porque el dano se suma mas rapido de lo que el rescate
se suma. Un ensemble de este patrimonio no es una version debil de una buena idea: es
peor que cualquiera de sus partes.

## 3. El cruce que confirma la transferencia

Tres de las variantes que ganan en el OOF ya se enviaron antes, y las tres perdieron.
Solo se afirma equivalencia donde el modelo es el mismo, no donde solo se parece en la
familia, porque una equivalencia aproximada no sostiene una conclusion.

| variante | delta OOF | LB historico | delta LB | nota |
| --- | --- | --- | --- | --- |
| oof_ens04_v04/proba_ens | +0.01386 | 0.77510 | -0.00796 | es el mismo modelo, el promedio con mediana de M2 y  |
| oof_mod03b/proba | +0.01066 | 0.77993 | -0.00313 | es la misma union TF-IDF, con umbral 0.5 en vez de l |
| oof_candidates_v4/B_meta36 | +0.00609 | 0.78163 | -0.00143 | es la misma familia de meta de 36 features, con el r |

**En los tres casos donde la equivalencia con una submission ya enviada es solida, el orden del OOF y el orden del leaderboard estan invertidos: la mayor ganancia de OOF produjo la mayor perdida de LB. Es el patron que V16 midio con cuatro sondas, aqui cuantificado sobre noventa variantes de catorce artefactos.**

## 4. La tabla competitiva de la fase

| challenger | grupo | OOF optimo | delta OOF | rescates optimos | danios optimos | neto optimo | LB | estado |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| oof_candidates_v4/C_meta36_downwei | A | 0.790001 | +0.012790 | 137 | 65 | +72 |  | NO ENVIADO |
| oof_ens04_v04/proba_ens | A | 0.791075 | +0.013864 | 173 | 114 | +59 | 0.77510 | PERDIO EN EL LEADERBOARD |
| oof_meta_shift_v4/meta_control | A | 0.782576 | +0.005364 | 133 | 77 | +56 |  | NO ENVIADO |
| oof_mod03b/proba | A | 0.787876 | +0.010664 | 162 | 107 | +55 | 0.77993 | PERDIO EN EL LEADERBOARD |
| oof_candidates_v4/B_meta36 | A | 0.783306 | +0.006094 | 126 | 73 | +53 | 0.78163 | PERDIO EN EL LEADERBOARD |
| oof_meta_shift_v4/meta_shift | A | 0.783306 | +0.006094 | 126 | 73 | +53 |  | NO ENVIADO |
| oof_meta_shift_v4/meta_shift_cuant | A | 0.783234 | +0.006023 | 124 | 72 | +52 |  | NO ENVIADO |
| oof_meta_shift_v4/meta_shift_platt | B | 0.775573 | -0.001638 | 145 | 99 | +46 |  | NO ENVIADO |
| oof_candidates_v4/D_svd300_rbf | A | 0.788564 | +0.011352 | 134 | 90 | +44 |  | NO ENVIADO |
| oof_tracks_g_v4/svd__rbf_svc | A | 0.788564 | +0.011352 | 134 | 90 | +44 |  | NO ENVIADO |
| oof_candidates_v4/A_cost_sensitive | A | 0.786199 | +0.008987 | 104 | 63 | +41 |  | NO ENVIADO |
| oof_joint_label_v4/joint__lr4_excl | B | 0.775495 | -0.001717 | 165 | 124 | +41 |  | NO ENVIADO |

**Ninguno se envio en V17.** 45 candidatos tienen
rescate neto positivo y **ninguno alcanza la escala de 107 celdas**, luego ninguno tiene
una hipotesis competitiva concreta que justifique gastar una submission. Los tres que ya
se habian enviado en fases anteriores son precisamente los de mayor ganancia de OOF, y
los tres perdieron.

## 5. Decision

**Estado B: CLOSED, sin mejora. Los modelos historicos no proporcionan diversidad de errores suficiente para construir un ensemble que mejore al champion V03.**

Ninguna de las noventa variantes medidas alcanza la escala de cambio detectable, y la union de las ocho mejores es peor que el mejor componente individual. Los tres candidatos con equivalencia solida con una submission ya enviada son precisamente los que mas ganaban en OOF, y los tres perdieron en el leaderboard.

- Submissions gastadas: **0**.
- Submissions conservadas: **5**.
- V18 no se abre.


Fecha: 2026-09-27T02:20:54+00:00. Duracion: 0.0 s.
