## 5.5 Calibración CV <-> leaderboard [DOC]

La relación entre la validación agrupada y el leaderboard de Kaggle se midió con los envíos oficiales del proyecto. Esta sección fija la ley lineal, la brecha media y su lectura, y deja la progresión completa de pares CV/LB como contexto.

### 5.5.1 La ley lineal entre CV y LB

Con los cuatro pares canónicos cuyos extremos son verificables en disco (v01 0.5344444/0.55235; v07 0.586/0.60372; v08 0.5866/0.60376; v12 0.6005017473/0.62182), la regresión del leaderboard sobre el CV es $\text{LB} = 0.00223 + 1.02825 \times \text{CV}$. El informe publica un R2 de 0.998; recomputado sobre esos cuatro pares, el ajuste da 0.996776 (≈0.9968, `experiments/age_cierre_auditoria.json`), de modo que la cifra 0.998 no es reconstruible de los pares documentados: la discrepancia se declara aquí (ver `07_checklist_verificacion.md`, sección 0.4). La desviación residual es 0.00147. La primera regresión, que incluía el puntaje degenerado de v08 (0.39563, subida con formato roto, incidente D-021), dio pendiente 0.231 y R2 0.0043; al excluir ese punto se obtiene la línea final. Lectura práctica: la brecha es un offset aditivo estable; un gate de +0.003 en CV vale +0.00308 en LB, luego los gates locales ni engañan ni regalan puntos. (D-050, `DECISIONS.md:1247-1298`; `reports/INFORME_EXPERIMENTOS_Y_LEADERBOARD.md:62-66`; `reports/age_lb_calibracion.csv`; `notebooks/99_entrega_consolidado.ipynb` sección 38.1).

### 5.5.2 La brecha media y su lectura

El informe cita la brecha media CV → LB como +0.0177 ± 0.0023 sobre diez mediciones. La recomputación del checklist sobre once envíos históricos da media +0.01768 y sd 0.00204 (≈0.0020), y sobre los doce modernos ≈0.0010; el ±0.0023 es, por tanto, una cifra de prosa. Sobre los cuatro pares canónicos de la regresión, la brecha media es 0.01853 con sd 0.00164 y rango 0.01716-0.02132 (`reports/age_lb_calibracion.csv`). La lectura: la validación agrupada es conservadora. El test contiene near-duplicados (Kaufman, Rosset, Perlich y Stitelman, «Leakage in Data Mining: Formulation, Detection, and Avoidance», ACM Transactions on Knowledge Discovery from Data, 2012) del train que el CV agrupado elimina por diseño (los clústeres agrupan por similitud Jaccard; Jaccard, «The Distribution of the Flora in the Alpine Zone», New Phytologist, 1912) y el leaderboard los cobra; por eso el LB supera sistemáticamente al CV con un offset estable. No es un artefacto de la métrica ni un error del protocolo: es una propiedad medida del corpus. (D-050; `reports/INFORME_EXPERIMENTOS_Y_LEADERBOARD.md:68`; `07_checklist_verificacion.md` sección 6; `notebooks/99_entrega_consolidado.ipynb:1120.1286-1287`).

### 5.5.3 Progresión de envíos (v01 → V48)

La tabla recoge los 22 envíos puntuados del arco; los CV son locales (recomputables) y los LB son puntajes oficiales de Kaggle citados como documentados, no recomputables en el notebook.

| Versión | CV local | LB Kaggle | Ganancia LB |
|---|---|---|---|
| v01 | 0.5344 | 0.55235 | - |
| v02 | 0.5557 | 0.57018 | +0.01783 |
| v03 | 0.5656 | 0.58081 | +0.01063 |
| v04 | 0.5748 | 0.59178 | +0.01097 |
| v05 | 0.5784 | 0.59518 | +0.00340 |
| v06 | 0.5851 | 0.60273 | +0.00755 |
| v07 | 0.5860 | 0.60372 | +0.00099 |
| v08 | 0.5866 | 0.60376 | +0.00004 |
| v09 | 0.5897 | 0.60807 | +0.00431 |
| v11 | 0.5966 | 0.61744 | +0.00937 |
| v12 | 0.6005 | 0.62182 | +0.00438 |
| V19 | 0.6079388 | 0.62578 | +0.00396 |
| V20 | 0.6097782 | 0.62689 | +0.00111 |
| V30 | 0.6134471 | 0.63302 | +0.00613 |
| V30 + plug-in | 0.6159732 | 0.63575 | +0.00273 |
| V38 | 0.6166579 | 0.63590 | +0.00015 |
| V39 | 0.6168017 | 0.63588 | -0.00002 |
| V40 | 0.6175912 | 0.63685 | +0.00097 |
| V41 | 0.6181119 | 0.63722 | +0.00037 |
| V42 | 0.6184162 | 0.63716 | -0.00006 |
| V43 | 0.6184783 | 0.63722 | +0.00006 |
| V48 | 0.6186451 | 0.63724 | +0.00002 |

Notas. El registro conserva 22 versiones puntuadas y 23 puntajes: v08 tiene un segundo puntaje de 0.39563 procedente de una subida con formato roto (incidente D-021). El candidato v10 se construyó y validó (CV 0.5904) pero no se envió. La ganancia acumulada del arco es +0.08489 (de 0.55235 a 0.63724). La ley de calibración se ajustó solo con los cuatro pares canónicos (v01, v07, v08, v12); el resto de la tabla ilustra la consistencia del offset. (`reports/INFORME_EXPERIMENTOS_Y_LEADERBOARD.md:34-68`; `CHAMPION.md`; `07_checklist_verificacion.md` secciones 2 y 6).

Fuentes: `reports/INFORME_EXPERIMENTOS_Y_LEADERBOARD.md`; `reports/age_lb_calibracion.csv`; `experiments/age_cierre_auditoria.json`; `DECISIONS.md` (D-050, D-021); `CHAMPION.md`; `plan/kaggle_notebook/paquetes/07_checklist_verificacion.md` (secciones 0.4-0.5 y 6).
