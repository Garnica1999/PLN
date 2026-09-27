# V18 P0: auditoria de novedad, correccion selectiva (2026-09-26)

## Que se entrego

`kaggle_toxic_2026_puj/reports/v18_00_auditoria_novelad.md`. Sin entrenamiento, sin
artefactos, sin|leaderboard. Ninguna cifra del informe es nueva.

## Correccion de nombre importante

`experiments/v11_p7_corrector_margenes.py` **no existe** en ninguna worktree. Los dos ficheros
reales son `experiments/v8_08_corrector_margenes.py` (desde v8) y
`experiments/v11_07_correccion_prior.py` (desde v11). El prefijo `v11_p7` viene de la
submission `toxic_submission_v11_p7_corrector_margenes.csv`, no del script.

## Los dos indicios, resueltos

1. **V8-08 corrector de margenes: ejecutado y descartado.** -0.002063, tasa de acierto
   0.3611 contra minimo 0.5166, 0 de 5 pliegues. Submission escrita pero **no enviada**:
   la columna `lb` esta vacia en `resultados/v16_00_prevalencia_y_curva_lb.csv`. Se envio la
   otra rama del mismo script, el promedio de repeticiones, con **LB 0.77122**. Clave para
   V18: el corrector **sustituye** la prediccion de forma incondicional. No hay mascara, no
   hay interruptor, no hay aceptacion condicional.
2. **El eje error correction de v12 no es un hueco catalogado.** La fila de
   `v12_gap_audit.csv` tiene `experiment_ids` = V8-08 y estado PARTIALLY_TESTED: el
   mecanismo base ya se ejecuto y fallo. Lo unico que v12 deja abierto son las
   **interacciones multiplicativas entre margenes**, que no tienen script ni informe en
   v13 a v17. Es una extension del corrector, no una regla de aceptacion.

## El resultado de v15, localizado y clasificado

Script `experiments/v15_01_gap_y_ortogonalidad.py`; informes `reports/v15_01_gap_y_ortogonalidad.md`
y `reports/v15_02_informe_cierre.md`; entrada `V15-01` de `experiments.csv`, decision
"NO ADOPTADA. 0 candidatas". Cifras confirmadas: -0.002868, -0.001352, -0.003037, con
acierto 0.0000 en las celdas modificadas.

**Fue anadir features a V03, no una regla de aceptacion.** La funcion se llama
`_adicion_a_v03` y el JSON dice "se anade al final de la matriz. Todo lo demas queda
congelado". El AUC de 0.5547 mide que la senal existe; el cambio se aplica globalmente.

## Veredicto

La formulacion de V18 es genuinamente nueva. De 18 mecanismos catalogados, 16 se ejecutaron
y descartaron, 1 quedo catalogado sin ejecutar (interacciones multiplicativas) y ninguno
tiene forma de gate de aceptacion por fila. 29 terminos buscados devolvieron cero
ocurrencias; la ausencia se sostiene en el universo de los 18 mecanismos, no en las frases.

## Advertencias para la fase

- Precedente mas duro, y es de otro eje: **E2-02 y E2-18**, modelo auxiliar que gana
  +0.0102 local y pierde 0.00143 en el externo. Interseccion de errores entre bases: 12
  filas de 1483.
- Techo del proyecto: mejor ganancia anidada +0.000148 contra gate de +0.002, banda de
  ruido entre particiones equivalentes 0.017184, desacuerdo entre cinco modelos 0.92.
- La senal residual de v15 dio acierto 0.0000. Si V18 la usa como regla, tiene que medir
  el acierto entre las celdas donde la regla **acepta**, no entre las que el challenger
  **cambia**.
