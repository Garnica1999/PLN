# Contexto V20 - auditoria de interacciones de scores (V20-00)

Fecha: 2026-09-26. Metodo: lectura directa de codigo en 12 worktrees (v7, v8, v11 a v20).
Detalle completo: `kaggle_toxic_2026_puj/reports/v20_00_auditoria_interacciones.md`.

1. VEREDICTO: la formulacion de V20 es genuinamente nueva. Ningun script multiplica los scores
   OOF de V03 (HS, TR, AG) entre si para cambiar la decision de una sola etiqueta.
2. Comprobacion de cierre: los 64 scripts que leen `features/oof_v03_final.npz` se enumeraron uno
   a uno y se busco producto entre columnas en cada uno. Cero en 63; el unico positivo es
   `v8_12_seleccion_por_grupo.py:170`, que es `y[:,0]*4 + y[:,1]*2 + y[:,2]`, powerset de
   etiquetas VERDADERAS, no scores.
3. Triple producto `HS*TR*AG`: busqueda explicita en los 12 worktrees, cero coincidencias.
4. Unico ejecutado adyacente: `v6_02_label_factorization.py:188`, `proba_tr = proba_hs *
   cond_tr_hs1`. Medido -0.010716, IC95 [-0.01843, -0.003339], 1 rescate / 35 danio, 1/5 pliegues,
   Gate 1 no superado. NO es V20: el segundo factor es un modelo NUEVO entrenado solo en filas
   HS=1 (lineas 181-187) y el primero es un HS reentrenado por ese codigo (lineas 167-170).
5. V6-B si comparte con V20 lo importante: V03 como base y cambio de una sola columna
   (lineas 342-344). V20 es la version de V6-B con los factores correctos.
6. `x * (1 - y)`: presente solo como mezcla de ramas en `v6_02:207-210` con condicionales, no con
   scores. Medido B2-AG-dos-ramas -0.000832, IC95 [-0.007475, +0.005724].
7. `logit(x) * logit(y)`: presente en `v12_03_entrenamiento_por_pares.py:257-258` como
   `arriba_hs * (logitos[:,0] - logitos[:,1])`. NO es V20: es un termino de perdida, y
   `arriba_hs` es +1/-1 segun la etiqueta VERDADERA (linea 256). Medido lambda=0.25 da
   -0.002469, deteccion 0.4915 bajo el minimo 0.5166. H3 cerrada.
8. Descartado por categoria 1 (texto): `v9_05_proyecciones_supervisadas.py:100-101` tiene
   literalmente `p[:,0]*p[:,1], p[:,0]*p[:,2], p[:,1]*p[:,2]`, pero `p` son proyecciones del
   TF-IDF sobre direcciones discriminantes (lineas 98-99) que alimentan una LR nueva (linea 110).
   Medido SDR 0.706634 vs SVD 0.653145, -0.070578 frente a V03. No es V20.
9. Descartado por categoria 2 (meta-modelo): V8-08 corrector sobre margenes, V13
   `17_stacking_v2.py`, y `33_class_conditional_lm_v4.py:347-349` que calcula la entropia binaria
   elemento a elemento de las tres columnas de V03 como columnas de meta.
10. Descartado por no continuo: V19-02 `v19_02_restriccion_tr_implica_hs.py` es una regla dura
    con gatillo (docstring lineas 27-30). Medido +0.0006626 +- 0.0023678, 3/5 pliegues,
    `reproducible: false`, cambios netos -3 frente a 26.75 necesarias, REGLA DESCARTADA.
11. V16 (00-03) y V8 (00-13) revisados: prevalencia, submissions, promedio de umbrales, vecinos de
    capacidad, ensemble, log-count ratio sobre la matriz TF-IDF, C y pesos, curva de transferencia.
    Nada de interacciones entre scores.
12. V20 no esta pre-registrada: la cadena `\bV20\b` no aparece en `DECISIONS.md` ni en
    `experiments/`.
13. AVISO para la fase: `DECISIONS.md` D-095 ya diagnostico que el producto atenua la probabilidad
    en filas con score intermedio y que V03 acertaba ahi. V20 debe confrontar ese mecanismo, y la
    version en log-espacio (suma de logits) lo evita por construccion.
14. No se escribio ningun script nuevo, no se entreno nada, no se generaron artefactos de datos.
    Solo lectura, greps y los dos ficheros de informe.
