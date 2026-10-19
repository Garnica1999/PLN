## 5.4 Análisis matemáticos y estadísticos [DOC]

Cuatro análisis delimitan el desempeño del campeón: la cota superior de un selector perfecto (el techo oracle), la refutación de la prueba que declaraba inalcanzable el LB 0.70, la multiplicidad estadística de las ~151 corridas y el sesgo del inner split de Optuna (Akiba et al., «Optuna: A Next-generation Hyperparameter Optimization Framework», KDD 2019). Se presentan en lenguaje llano, con su receta y sus fuentes.

### 5.4.1 El techo oracle del ensamble: 0.693547

El cálculo parte de las 169.336 predicciones OOF de los seis miembros del ensamble (V30, V43, V42, V40, V39-ordinal y V39-HGB). Con la mezcla fija p_blend = 0.555·V30 + 0.085·V43 + 0.175·V42 + 0.080·V40 + 0.050·V39_ord + 0.055·V39_hgb y la regla plug-in congelada (Lipton, Elkan y Narayanaswamy, «Thresholding Classifiers to Maximize F1 Score», arXiv, 2014) w = [1.0000, 1.1749, 1.2721, 1.1630, 1.0010, 0.8128], el blend comete 65.271 errores (38.55%). El oráculo se construye por sustitución por fila: para cada error, si algún miembro acierta la etiqueta verdadera bajo su propia regla plug-in, la predicción se reemplaza por y_true. El macro-F1 del vector corregido es 0.693547. La descomposición ordinal muestra dónde está el error: 61.94% a distancia 1 (40.429 filas), 23.59% a distancia 2 (15.395) y 14.47% a distancia 3 o más (9.447); el 85.5% está a distancia ≤ 2. Advertencia de lectura obligatoria: es un oráculo de selección por fila sobre miembros ya entrenados, la cota superior de un selector perfecto con la misma evidencia, no un modelo. El intento de realizarlo (árbitro V17-P9) falló con 3 rescates y 22 daños; no debe presentarse como objetivo alcanzable. (D-097, `DECISIONS.md:3720-3759`; `notebooks/99_entrega_consolidado.ipynb` sección 44.1; `RUNLOG.md:2681-2701`).

### 5.4.2 La refutación del techo de LB 0.70 (K-01)

El acta D-087 declaraba inalcanzable un LB de 0.70 apoyándose en un oráculo de umbrales. La auditoría D-090 revisó la prueba y la refutó: la rejilla del script solo exploraba tau ≥ 0.30 (TAUS = (0.30, 0.35, 0.40, 0.45, 0.50), en `scripts/v35_00g_techo_analisis.py:69`) y el techo se tomó como su máximo, pero la curva del oráculo es monótona creciente al bajar tau: un umbral más permisivo corrige más filas. La reproducción sobre `artifacts/p33_scores/todos/` (base 0.6159732, 65.683 errores) lo cuantifica:

| tau | filas corregibles | ganancia macro-F1 | ¿supera +0.0626? |
|---|---|---|---|
| 0.30 (máximo publicado) | 10.384 | +0.0593677 | no |
| 0.2930 | 11.298 | +0.0646335 | sí |
| 0.28 | 13.199 | +0.0755813 | sí |
| 0.25 | 18.345 | +0.1054505 | sí |

La conclusión correcta es más fuerte y distinta: el objetivo queda exactamente en el borde de lo que un oráculo sobre la salida de V30 podría entregar, pues exigiría acertar ~16% de los errores con p(y_true) ≥ 0.293, es decir, conocimiento casi-oracular. El LB 0.70 sigue siendo inalcanzable en la práctica, pero la prueba publicada no lo demuestra. (D-090, `DECISIONS.md:3462-3478`; `experiments.csv:198-200`; `plan/auditoria/02c_protocolo_y_qa.md`; `PROJECT_STATE.md:20-27`).

### 5.4.3 La multiplicidad de las corridas (A-01)

El registro acumuló ~151 corridas y cada adopción se decidió con un gate de delta sobre el CV. Con un macro_f1_std mediano de 0.0042 (76 filas que lo declaran), el error estándar de la media de 5 folds es SE ≈ 0.00188. En esas condiciones, un gate de Delta ≥ +0.0015 equivale a z ≈ 0.80 y, sobre una familia de 151 pruebas, se esperarían del orden de 32 falsos positivos; con Delta ≥ +0.0030 (z ≈ 1.60), unos 8. Una corrección de Bonferroni de familia 151 exigiría Delta ≥ 0.0147, umbral que ninguna adopción del proyecto declara por sí sola. La auditoría lo registró como defecto de marco (A-01), no de ejecución. Lo que sostiene las conclusiones: las cuatro adopciones reales (V19-B1, V20, V30 y el plug-in por clase) tienen IC95 pareado que excluye el cero y confirmación externa en el leaderboard, y el único control de multiplicidad previo era el Bonferroni interno de V37 (18 pruebas por clase). (D-090, `DECISIONS.md:3484-3494`; `experiments.csv:198`; `plan/auditoria/05_sintesis_y_cierre.md:13-56`).

### 5.4.4 El sesgo del inner split de Optuna

Los cinco scripts del camino al campeón (`v29_04`, `v29_06`, `v30_01`, `v30_02`, `v31_05`) seleccionan hiperparámetros con un inner split estratificado sin grupos; como el outer split está agrupado por clústeres de near-duplicados, un mismo clúster puede caer a ambos lados del inner. La medición reconstruye las dos particiones interiores exactamente como las construye el código real, sobre los 5 folds canónicos (169.336 filas, 164.018 clústeres):

| Nivel | solape de clúster en la validación interior |
|---|---|
| outer agrupado (canónico) | 0.000% |
| inner ingenuo (código real) | 3.907% |
| inner agrupado (propuesto) | 0.000% |

El criterio se declaró antes de correr (diferencia > 0.02 = material); lo medido fue 0.039071, casi el doble, y consistente en los 5 folds (3.787%-4.152%). Esto corrige la medición previa registrada en D-090 ("0.963% vs 1.030%"), que medía solapamiento de texto exacto cuando los clústeres agrupan por Jaccard (Jaccard, «The Distribution of the Flora in the Alpine Zone», New Phytologist, 1912) y comparaba contra cero teórico en vez del inner agrupado. Alcance: el leakage está en la SELECCIÓN de hiperparámetros, no en la estimación externa, que usó el outer agrupado; la cifra publicada del campeón no cambia. La re-optimización quedó pendiente de decisión y no se ejecutó. Nota de citación: hay dos actas numeradas D-091 (inner split, 2026-10-01, y V38, 2026-10-16); al citar, usar el título. (D-091 inner split, `DECISIONS.md:3534-3606`; `experiments.csv:201`; `plan/auditoria/06_accion_inner_split.md`; `scripts/auditoria_medir_inner_split_optuna.py`).

Fuentes: `plan/kaggle_notebook/paquetes/06_fracasos_y_analisis.md` (B.1-B.4); `DECISIONS.md` (D-090, D-091 inner split, D-097); `experiments.csv:198-201`; `plan/auditoria/05_sintesis_y_cierre.md` y `06_accion_inner_split.md`; `reports/auditoria_experimentos_v1_v37.md`.
