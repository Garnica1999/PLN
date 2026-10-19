## 2.5 Auditoría del historial V1-V37 [DOC]

*Sección documental: no re-ejecuta ningún script; cita cifras ya verificadas contra el registro experimental, las actas y los informes. Fuentes: `DECISIONS.md` (actas D-090 y D-091), `reports/auditoria_experimentos_v1_v37.md` (citado como `informe`), `experiments.csv` y los paquetes 06 y 07 de `plan/kaggle_notebook/paquetes/`.*

El cierre del ciclo V37 dejó pendiente verificar si las corridas anteriores se habían ejecutado correctamente y si sus decisiones estaban justificadas. Se auditó el historial completo V1-V37 con criterio de evidencia, no de opinión: cada afirmación debía citar su artefacto con ruta y línea. El objeto fue el registro completo de `experiments.csv`: 195 filas que no son 195 experimentos, sino 151 ejecuciones, 33 documentos (21 documentales y 12 pre-registros), 10 enmiendas y 1 corrida abortada (`V35-E1-INTENTO-1`, por `UnboundLocalError`); el inventario actualizado suma 200 filas al añadir las 5 filas `AUD-*` (`DECISIONS.md:3445-3446`; informe §3; paquete 07 §5). El peso del registro está en V1-V12: 113 de las 195 filas, donde los parámetros heredados acumulaban más riesgo (informe §2).

### 2.5.1 El protocolo de verificación

La auditoría usó seis revisiones independientes por especialidad (nlp, arquitectura, DL, kaggle, reviewer y QA) más la verificación propia del auditor; ningún hallazgo se aceptó sin comprobación contra la fuente primaria con ruta y línea (informe §2). Las restricciones fueron estrictas: no se re-ejecutó ningún script, no se abrió test, holdout ni leaderboard, no se generó ninguna submission y el campeón quedó intacto (D-090; informe §2). El protocolo de competencia se verificó por barrido del árbol sintáctico de 308 scripts: cero `fit` sobre test (los dos únicos `fit` con receptor de test calibran sobre OOF de train), vectorizadores ajustados solo sobre train, `id` nunca usado como característica, sin fuga de autor ni temporal (Kaufman, Rosset, Perlich y Stitelman, «Leakage in Data Mining: Formulation, Detection, and Avoidance», ACM Transactions on Knowledge Discovery from Data, 2012); el 4.2554% del test cae en un clúster de train y el esquema agrupado deja 0.00% de memorización frente al 3.99% del ingenuo (informe §6). La resolución de artefactos también se comprobó: de 262 citas, 230 existen en disco y las 32 restantes son globs, carpetas o referencias textuales; los 95 "huecos" iniciales eran problemas de formato de cita, no evidencia ausente (informe §3).

### 2.5.2 Los defectos confirmados

Se confirmaron 20 defectos (4 media-alta, 9 media, 7 baja); ninguno invalidó una adopción ni exigió cambiar el campeón (`experiments.csv:198`; paquete 06 §B.3). Los más relevantes:

| Defecto | Hallazgo | Fuente |
|---|---|---|
| K-01 (alta) | El "techo" de LB 0.70 queda refutado como prueba de imposibilidad (detalle en 2.5.3) | D-090; informe §4 |
| A-01 (alta) | La multiplicidad de las ~151 corridas no estaba cuantificada (detalle en 2.5.4) | D-090; informe §5 |
| K-02 (media) | El holdout de V12 está quemado: declara su criterio (Δ ≥ +0.003) sobre sí mismo con `confirmado: true`; es puerta de selección, no confirmación. Mitigado: el CV ya daba +0.0031 en 5/5 folds | informe §7; D-090 |
| N-01 (media) | El brazo NER `N1b` filtra spans con índices desalineados en ~55.8% de los documentos; sin impacto porque los seis brazos quedan bajo el gate | informe §7 |
| N-02 (media) | `es_core_news_lg` no estaba fijado en `requirements.txt`; corregido en D-091 con el wheel 3.8.0 (sha256 `7c6c2127…`) | D-091; informe §7 |
| A-03 (media) | `max_iter=40` en el MLP miembro del campeón (peso 0.25); su JSON de reoptimización no existe | informe §7 |
| A-06 (baja) | Falta `torch.manual_seed`: V18-P3/P4/P5 no son reproducibles | informe §7 |
| K-03 (media) | Registro de envíos obsoleto: contaba 15/12 cuando hay 16 puntuados | informe §7 |
| Registro | `V37-DIAG-POS` escrita con 21 de 22 campos; SHA256 del pre-registro V37 mal transcrito (`30d5d3cf…` -> `1d6d6573…`); ambos corregidos | informe §9; D-090 |

El propio artefacto de la auditoría pasó por revisión: su script de inventario tuvo tres defectos bloqueantes, detectados por `ml-reviewer` y corregidos antes de publicarse (informe §9).

### 2.5.3 El "techo" de LB 0.70 queda refutado (K-01)

El hallazgo más grave no fue de ejecución, sino de razonamiento. D-087 afirmó que el objetivo de LB 0.70 era inalcanzable porque el oráculo rendía +0.0568 frente a los +0.0626 exigidos; pero la rejilla del script era `TAUS = (0.30, 0.35, 0.40, 0.45, 0.50)` (`scripts/v35_00g_techo_analisis.py:69`) y el "techo" se tomó como su máximo, mientras la curva del oráculo es monótona creciente al bajar tau. La reproducción sobre `artifacts/p33_scores/todos/` (base 0.6159732; 65.683 errores) da:

| tau | filas corregibles | ganancia macro-F1 | ¿supera +0.0626? |
|---|---|---|---|
| 0.30 (máximo publicado) | 10.384 | +0.0593677 | no |
| 0.2930 (el punto que D-087 usa como objetivo) | 11.298 | +0.0646335 | sí |
| 0.28 | 13.199 | +0.0755813 | sí |
| 0.25 | 18.345 | +0.1054505 | sí |

En el punto exacto que D-087 define como objetivo el oráculo da +0.0646, un 3% por encima de lo exigido: el "no llegaría" era un artefacto del redondeo de la rejilla. La conclusión correcta es más fuerte y distinta: el objetivo queda exactamente en el borde de lo que un oráculo sobre la salida de V30 podría entregar (exigiría acertar ~16% de los errores con `p(y_true) >= 0.293`, conocimiento casi-oracular). LB 0.70 sigue siendo inalcanzable en la práctica; la prueba publicada no lo demuestra (D-090; informe §4). El IC95 [0.635, 0.653] de D-087 queda etiquetado como declarado, no estimado (acción `AUD-IC95-TECHO`; D-091).

### 2.5.4 Multiplicidad de las ~151 corridas (A-01)

El único defecto de marco analítico: la multiplicidad de las ~151 corridas nunca se cuantificó (el único control previo era el Bonferroni interno de V37, 18 pruebas por clase). Con `macro_f1_std` mediano 0.0042 en las 76 filas que lo declaran, el error estándar de la media de 5 folds es ~0.00188; el gate vigente acumulaba estos falsos positivos esperados:

| Gate | z aproximado | Falsos positivos esperados en 151 corridas |
|---|---|---|
| Δ ≥ +0.0015 | ~0.80 | ~32 |
| Δ ≥ +0.0030 | ~1.60 | ~8 |
| Bonferroni de familia 151 | — | exigiría Δ ≥ 0.0147 |

El argumento "ninguno pasó el gate" no está protegido contra multiplicidad; la afirmación correcta es "ninguno pasó el gate *y* su bootstrap pareado (Efron, «Bootstrap Methods: Another Look at the Jackknife», The Annals of Statistics, 1979) excluyendo 0" (informe §5; D-090).

### 2.5.5 El sesgo del inner split de Optuna (D-091)

El inner split de Optuna (Akiba et al., «Optuna: A Next-generation Hyperparameter Optimization Framework», KDD 2019) usa `scheme="stratified"` con `groups=np.arange(...)` en cinco scripts del camino al campeón (`v29_04`, `v29_06`, `v30_01`, `v30_02`, `v31_05`). La medición reconstruyó las dos particiones interiores exactamente como las construye el código real, sobre los 5 folds canónicos (169.336 filas, 164.018 clústeres):

| Nivel | solape de clúster en la validación interior |
|---|---|
| outer agrupado (canónico) | 0.000% |
| inner ingenuo (código real) | 3.907% |
| inner agrupado (propuesto) | 0.000% |

Con el criterio declarado antes de correr (diferencia > 0.02 = material), la medida fue 0.039071, casi el doble, y consistente en los 5 folds (3.787%-4.152%). La medición corrige la cifra previa de D-090 ("sin impacto medible": 0.963% vs 1.030%), errónea por dos motivos: medía solapamiento de texto exacto cuando los clústeres agrupan por Jaccard (Jaccard, «The Distribution of the Flora in the Alpine Zone», New Phytologist, 1912), y comparaba contra cero teórico en vez del inner agrupado. El leakage está en la *selección* de hiperparámetros, no en la estimación: la cifra externa del campeón usó el outer agrupado y no cambia; lo que no es reproducible bajo un esquema honesto es el proceso de selección. Re-optimizar quedó pendiente de decisión y no se ejecutó (D-091; paquete 06 §B.4). Nota de numeración: existen dos actas D-091 (inner split, 2026-10-01, y V38, 2026-10-16); al citar, indicar línea y contexto (paquete 06 §B.4).

### 2.5.6 Consecuencias: qué no invalida y por qué

La auditoría no invalidó ninguna adopción, y la razón es verificable: las cuatro adopciones reales del proyecto (V19-B1, V20, V30 y el plug-in) tienen IC95 pareado excluyendo 0 y confirmación externa. El bootstrap pareado corrige la varianza de estimación; no corrige la selección del máximo sobre 151 corridas, pero las adopciones que importan no dependen de esa corrección (informe §5). El campeón quedó verificado bit a bit en 0.6159732487657944 (SHA256 `29cbe6cc…`; 112.892 filas; 6 clases; 0 nulos) y el historial es auditable con `scripts/auditoria_v1v37_inventario.py` (6/6 checks) (D-090; paquete 07 §5). Quedaron reservas declaradas: el IC95 del techo es un rango declarado (ya etiquetado) y el `max_iter=40` del MLP miembro del campeón sigue sin revisar porque su JSON de reoptimización no existe (informe §10). Ninguna reserva exigió cambiar el campeón.
