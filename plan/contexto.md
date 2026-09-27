# Contexto de la cadena - V21 (CERRADA)

Actualizado 2026-09-26. Worktree `/home/carlos/Git/PLN-v21`, rama `v21`, base `ad761ba`.
**Fase cerrada. Sin submission enviada. V03 = 0.78306 LB, intacto.**

## Decision de cierre

**NO CANDIDATE JUSTIFIES SUBMISSION.** Las **4 submissions restantes se reservan**.

Orden pre-registrado por si hubiera restriccion externa: **WS-3 > VS-100 > L1 C=4.0**.

## La tabla

| candidato | DeltaOOF | rescates | danos | neto | novedad | coste | submission |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| FUSION-WS-3 | +0.005747 | 74 | 63 | +11 | ya entrenada | cero | no |
| FUSION-VS-100 | +0.005046 | 77 | 67 | +10 | ya entrenada | cero | no |
| L1 C=4.0 | +0.006431 | 208 | 205 | +3 | nueva | bajo | no |
| All-but-the-top (D=1..3) | no medida | -- | -- | -- | nueva | bajo | no |
| SSWE | no medida | -- | -- | -- | nueva | alto | no |
| CBOW / GloVe / word2vec | no medida | -- | -- | -- | nueva parcial | alto | no |

## Los tres argumentos de la decision

1. El cambio de WS-3 no es distinguible de ruido: 74 contra 63 sobre 137 pares
   discordantes, **McNemar exacto p = 0.3930**, menos de un sigma. Su perfil implica
   +0.00024 de LB, ~2 % de un error estandar.
2. WS-3 y VS-100 solapan el **80.3 %** de sus cambios: son la misma perturbacion.
3. L1 tiene el mayor DeltaOOF y el peor perfil: 413 celdas para un neto de +3, 2,7 veces
   menos eficiente por celda, y desplaza la tasa de TR en +0.0357 con saldo negativo, que
   es el mecanismo exacto de V20 (LB -0.00643).

## Correcciones registradas

- **McNemar del especialista**: p = 0.3930, no 0.347. Conclusion igual.
- **Su regresion dLB sobre dOOF se descarta**: ajusto con n=7 pero solo 4 sondas tienen
  `local_f1` numerico; recalculada da `dLB = -0.00744 + 0.508 * dOOF`, n=4, dominada por V20.
  Y `local_f1` no significa lo mismo en todas las fases.
- **P0**: la afirmacion de que las cinco sondas de V16 tenian DeltaOOF positivo era
  **incorrecta**; tres de cuatro lo tienen negativo. Corregido en el informe de P0.

## Hallazgos que sobreviven a la fase

1. El **NB-SVM del proyecto estaba mal implementado** (desviacion del ratio 4.32 frente a
   0.79 del canonico, 30-46 % fuera de [-3,3]) y la version **correcta es peor** (0.7638),
   porque binariza y descarta el term frequency, que en tweets toxicos es senal.
2. Las metricas de la linea de embeddings son **sensibles a detalles no publicados**. V5.1
   consulta el **binario** de fastText, donde 2.282 tipos fuera de vocabulario reciben vector
   por subpalabra; el `.vec` solo exporta los 100.098 del modelo. Con el `.vec`,
   FUSION-B0-200 daba 0.7728 en vez de 0.7818. Toda comparacion futura debe declarar de
   donde salen los vectores de tipo.
3. **L1 recorta la brecha train-OOF de 0.1980 a 0.0665** a C=2.0: mecanismo real y medido,
   que no se convierte en celdas acertadas.
4. La **unica palanca con evidencia publicada solida en espanol** es el preprocesamiento
   del **texto de la tarea**, que V5 aplico al corpus y jamas al texto de la tarea.
   **No medida, no descartada.**

## Artefactos de la fase

- `reports/v21_p0_00_inventario_v5_v5_1.md`: inventario y auditoria de V5/V5.1 (20 filas).
- `reports/v21_p1_01_embeddings_desde_cero.md`, `reports/v21_p1_02_toxicidad_nlp_clasico.md`,
  `reports/v21_p1_03_nbsvm_y_hibridos.md`, `reports/v21_p1_03_addendum_verificacion.md`.
- `reports/v21_01_triaje_competitivo.md` + `.json`, `experiments/v21_01_triaje_competitivo.py`.
- `reports/v21_02_dictamen_competitivo.md` (kaggle-specialist).
- `reports/v21_03_cierre_de_fase.md`: la tabla y la decision.
- `docs/v21_p2_especificacion_niveles_limpieza.md`: spec de P2, **no ejecutada** (P2 queda
  sin ejecutar; el cierre se apoyo en el techo medido y en el dictamen).

Commits: `fbe6b94`, `f8b1400`, `2e095d8`, `0379602`, `37ed7a3`, `4e5c0e0`.

## Estado

- **V03 = 0.78306**, sha256 `7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e`,
  `baseline_v03_frozen` sin cambios.
- **Submissions: 21 en el log, 0 enviadas en V21, 4 disponibles.**
- V22 **no** se abre sin una hipotesis con diferencia informacional o algoritmica
  identificable, o sin restriccion externa que obligue a usar WS-3.
