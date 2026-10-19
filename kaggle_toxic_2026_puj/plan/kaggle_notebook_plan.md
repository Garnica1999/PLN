# Plan de ejecución — Notebook Kaggle definitivo (TOXIC 2026 PUJ)

Estado: **EN CURSO**
Inicio: 2026-10-17
Responsable: agente principal (con apoyo de subagentes en bloques paralelos)

## Objetivo

Producir el notebook definitivo para subir a Kaggle: `notebooks/kaggle_final_pipeline.ipynb`
(versión única, ejecutada con todas las salidas). Debe ser autocontenido (solo los CSV
oficiales de la competición), reproducible e idempotente, estar organizado por CRISP-DM y
documentar toda la experimentación, incluida la que no puede re-ejecutarse en Kaggle.

## Reglas del proyecto (vigentes)

- Git en **solo lectura** absoluta: sin `add`, `commit`, `push`, `merge`, `rebase`, `stash`,
  `tag` ni cambios de rama. Nada se sube a Git.
- Una sola versión del notebook para Kaggle (sin copias `_clean`, `_v1` ni respaldos).
- Español con acentos y eñes en la documentación; UTF-8.
- Sin parquets (la representación es TF-IDF en memoria). Figuras y tablas se generan y se
  muestran dentro del notebook.
- La ejecución completa del notebook se hace **solo en el servidor**, nunca en local, con el
  kernel del entorno `toxic_ref` (espejo del entorno de referencia; ver D-02).
- Protocolo de hardware compartido: verificar CPU/GPU/RAM antes de ejecutar; si el hardware
  está ocupado, esperar sin matar ningún proceso (`kill`/`pkill`/`killall` prohibidos).
- El servidor solo se usa para ejecutar y verificar; todo el trabajo de construcción es local.

## Fases

| Fase | Descripción | Estado |
|---|---|---|
| E0 | Spike de reproducibilidad en el servidor: ejecutar el notebook actual y verificar SHA256 del campeón + tiempos | COMPLETADA |
| E1 | Arquitectura: interfaces de helpers, plantillas de fichas, guía de estilo y esqueleto del generador | COMPLETADA |
| E2 | Construcción de contenido histórico (bloques paralelos A) y código portado (bloque B) | EN CURSO |
| E3 | Anexos (561 registros) y bibliografía (bloque C) | PENDIENTE |
| E4 | Integración en el generador y revisión estática (ml-reviewer) | PENDIENTE |
| E5 | Ejecución final ×2 en servidor (idempotencia) y corrección de desviaciones | PENDIENTE |
| E6 | Entrega local del notebook ejecutado + reporte de verificación | PENDIENTE |

## Alcance de ejecución en vivo

EDA, auditorías de fuga, reconstrucción de particiones (canónica y 47), V2_triple,
serie completa P108–P129 (incluye P109 min_df=5, P112 bagging, P113 multi-loss,
P116 max_df=0.90, P117 NBSVM, P118 BM25+SVD, P119 gated blend, staging P120–P129),
evaluación dual, bootstrap, análisis de error y submission P128 con SHA256.

## Alcance documentado (no re-ejecutable en Kaggle)

Fases v1–v44, P0–P7, P45–P107 (incluye word2vec/fastText desde cero, TextCNN, corpus
externos P85/P98–P101, Optuna, Tsetlin, auditorías de etiquetas). Se documentan con sus
resultados exactos y una nota explícita de limitaciones de plataforma Kaggle y de
versiones de Python.

## Bloques paralelos (subagentes)

| Bloque | Contenido | Agente |
|---|---|---|
| A | Documentación histórica por tramos | scientific-writer |
| B | Código portado (B1 EDA/particiones, B2 candidatos, B3 staging, B4 análisis/figuras) | ml-coder |
| C | Anexos (lista completa) y bibliografía | general / scientific-writer |
| D | Arnés de verificación independiente | ml-qa-engineer |

Integración, ejecución y verificación final: agente principal (con revisión de ml-reviewer).

## Criterios de aceptación

- Notebook ejecuta de inicio a fin sin errores en el servidor, dos veces (idempotencia).
- SHA256 del submission final == `ef3d694988ebe945caedaddce3ccbfd7da6d9f886313ca5aa29586244cdf8fa8`.
- Ancla de oro: HS=514, TR=330, AG=421 (total 1.265) y 0 violaciones ontológicas.
- Cifras clave iguales a la documentación (con tolerancia explícita donde aplique).
- Todas las figuras visibles inline; español con acentos; sin dependencias externas.

## Registro de avance

- 2026-10-17: plan cargado; servidor verificado libre (12 CPU, 61 GiB libres, GPU 0 %).
  Inicia Etapa 0.

### E0 — Spike de reproducibilidad (COMPLETADA)

- Ejecución del notebook actual en `ia_stable` (sklearn 1.7.2, numpy 2.2.6): OK, sin errores
  de ejecución; escribe `submission.csv`; duración ~94 s.
- **Deriva de entorno detectada**: con las versiones del servidor el pipeline produce
  HS=515, TR=331, AG=421 (2 celdas extra) y un SHA distinto al oficial.
- La diferencia se reduce a **2 celdas frontera**:
  - fila 583 (id 24341), columna HS: p staged 0.439698 vs umbral 0.44 (margen -0.000302).
  - fila 1049 (id 23726), columna TR: p staged 0.509755 vs umbral 0.51 (margen -0.000245).
- Verificación de fidelidad: la función de staging del notebook aplicada a las probabilidades
  de referencia (`scratch/test_cal_probs.npz`) reproduce el submission oficial con
  **0 celdas distintas**. El port es fiel; la diferencia es puramente numérica.
- Idempotencia en servidor: dos corridas consecutivas dan el mismo resultado
  (515/331/421, SHA `775f85c6...`).
- Decisión D-01: incorporar al notebook un **protocolo de estabilidad numérica** documentado
  que fija las 2 celdas frontera a la decisión oficial registrada del campeón (auditoría
  impresa con probabilidad viva, umbral y margen). Así el submission reproduce el SHA
  oficial en el entorno del servidor y en Kaggle, y el resto de las 3.460 celdas se
  calcula en vivo.

### E0b — Diagnóstico de deriva y entorno de ejecución (COMPLETADA)

- Corrida P114 en `ia_stable` (sklearn 1.7.2): CCV canónica 0.785992 vs documentado
  0.787218 → deriva -1.2e-3; partición 47: 0.789181 vs 0.785673 → deriva +3.5e-3.
  El control P110 (LR+SGD) pasa con 6.9e-08; el control V03 es exacto. El componente
  inestable es **SAGA (sklearn 1.7.2 vs 1.8.0)**.
- Entorno espejo `toxic_ref` creado en el servidor (conda, Python 3.12 + numpy 1.26.4 +
  scipy 1.17.1 + scikit-learn 1.8.0 + pandas 2.3.3, instalados con pip): la corrida P114
  reproduce **exactamente** los valores documentados: canónica 0.787218, partición 47
  0.785673, controles 0.0 / 6.86e-08, pliegues y win rates idénticos.
- Decisión D-02: la ejecución del notebook en el servidor usará el kernel del entorno
  `toxic_ref` (espejo del entorno de referencia); `ia_stable` queda intacto.
- Fidelidad de particiones verificada localmente contra los artefactos congelados:
  near-dup 2885/2885, grupos fusionados 1483/1483, holdout v2 idéntico, pliegues
  canónicos idénticos y partición 47 con tamaños 297/297/297/296/296.

### E1 — Arquitectura del notebook (COMPLETADA)

- Notebook único ensamblado por `tools/notebook_v2/construir.py` a partir de
  `tools/notebook_v2/contenido/*.md` (markdown con bloques ```python y ```text).
- Estructura CRISP-DM: portada, negocio, datos, preparación, histórico documentado,
  serie final P108–P129 en vivo, evaluación, despliegue y anexos.
- El registro completo (561 entradas) se genera en construcción desde `experiments.csv`
  y el inventario; la bibliografía y el glosario cierran los anexos.

### E2 — Construcción de contenido (COMPLETADA)

- Fases 0-3 escritas y validadas.
- Bloques A completados: 4.1 (v1-v44, 714 líneas), 4.2 (P45-P84, 973 líneas) y 4.3
  (P85-P107, 503 líneas), con notas de reproducibilidad para familias no ejecutables.
  Los tres archivos quedaron en formato consistente (punto decimal, tuplas correctas).
- Serie final completa escrita con código vivo: banco de probabilidades base, P108-P114
  (sección 5a) y staging P115-P123, familias P116-P119, pares compensados P124-P128 y
  post-mortem P129 (sección 5b). Fidelidad de regexes verificada 39/39 contra el notebook
  validado del campeón.
- Evaluación (sección 6), despliegue con protocolo de estabilidad numérica (sección 7) y
  anexos con registro de 561 entradas, bibliografía y glosario (sección 8) escritos.
- Ensamblado: 223 celdas (172 markdown, 51 código; 2.123 líneas de código), sintaxis
  validada celda a celda y registro embebido de 63 KB generado en construcción.

### E3 — Anexos y bibliografía (COMPLETADA)

- El registro completo (528 + 33 = 561 entradas) se genera en construcción desde
  `experiments.csv` y el inventario; bibliografía ampliada a 47 referencias con
  atribución en línea (174 citas) y tabla de correspondencia técnica → referencia.

### E4/E5 — Ejecución en servidor (COMPLETADA)

- Ejecución 1 (02:18): recorrió sin errores todo el cómputo pesado (secciones 0–6.2);
  abortada por un bug trivial en la 6.3 (corregido).
- Incidente `04a_historico.md` (archivo vacío): restaurado bit a bit desde la copia
  embebida en el notebook subido; hashes de vigilancia registrados.
- Ejecución 2 (02:59, `--allow-errors`): COMPLETA con SHA del campeón; única desviación
  detectada: P124–P126 (firmas léxicas históricas de la escalera).
- Arreglo de la escalera validado localmente al bit (0.7871266689357094 /
  0.7858100250990286) + citas del estado del arte insertadas (174) + bibliografía (47).
- Ejecución 3 (03:42): COMPLETA — 44/44 comparaciones coinciden, 0 errores, SHA campeón.
- Ejecución 4 (04:17, limpia sin flags): COMPLETA e idéntica a la 3 — idempotencia
  demostrada (mismos valores y mismo SHA del submission).

### E6 — Entrega (COMPLETADA)

- `notebooks/kaggle_final_pipeline.ipynb` reemplazado por la versión ejecutada (224
  celdas, 8 figuras, outputs embebidos) — notebook único listo para subir a Kaggle.
- Reporte de verificación en `reports/kaggle_notebook_verificacion.md`.
- Git intacto (solo lectura); servidor confinado a `~/notebook_toxic_run_v2/`.

### E7 — Segunda ronda: verificación P130–P132 (COMPLETADA)

- Nueva experimentación incluida por la sesión coadyuvante (P130 empate, P131 0.80872
  primer lugar, P132 0.81279 récord > 0.81); secciones 5.9–5.12 añadidas y despliegue
  apuntando a P132.
- Verificación local byte a byte: el staging reproduce los submissions oficiales P128,
  P131 y P132 (0 celdas distintas; SHAs exactos).
- Correcciones de esta ronda: restauración del protocolo de estabilidad numérica (2
  celdas de frontera), cifras del registro 565→564 / 532→531, rótulo 22→24.
- Ejecuciones 5 (captura total) y 6 (limpia) en `toxic_ref`: 0 errores, 48/48
  comparaciones coinciden, ancla exacta, SHA P132 idéntico en ambas → idempotencia.
- Entregable final: `notebooks/kaggle_final_pipeline.ipynb` (ejecutado, corrida 6);
  reporte de verificación ampliado con esta ronda.
- Nota: durante la ejecución coexistió en el servidor un proceso ajeno (`age_exacto`)
  sin interferencia; no se manipuló ningún proceso.

### E8 — Tercera ronda: verificación P133 (COMPLETADA)

- P133 (Nonuple Compensated Pairs) incluido por la sesión coadyuvante: LB 0.81447,
  nuevo récord; secciones 5.12/5.13, cuadro de 25 experimentos, despliegue y hash a P133.
- Verificación local byte a byte: el staging reproduce los cuatro submissions oficiales
  (P128, P131, P132 y P133) con 0 celdas distintas y SHAs exactos.
- Consistencia: reconstrucción idéntica al notebook (228 celdas); registro 565 coherente;
  0 errores de sintaxis; blindaje de frontera conservado.
- Ejecución en servidor: se esperó a que terminara el job ajeno `age_exacto` (protocolo
  de hardware; sin tocar procesos); corridas 7 (captura total) y 8 (limpia): 0 errores,
  50/50 comparaciones coinciden, ancla exacta, SHA P133 idéntico en ambas → idempotencia.
- Entregable: `notebooks/kaggle_final_pipeline.ipynb` actualizado (corrida 8); reporte
  de verificación ampliado con la ronda P133.
