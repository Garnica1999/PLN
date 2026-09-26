# Fase 00 — Intención confirmada de V18

- **Fecha**: 2026-09-26
- **Rama**: `kaggle-age-2026-puj-v18` (creada desde `46b32f9`, árbol limpio)
- **Ámbito**: exclusivamente `KAGGLE_AGE_2026_PUJ/`
- **Origen**: brief de V18 escrito por Carlos (49 secciones), tan exhaustivo en objetivo, restricciones,
  orden y presupuesto que **no procede entrevista de requisitos**: este documento es su transcripción
  operativa, no una interpretación.

## 1. Objetivo

V17 cerró con una conclusión muy precisa: **la señal de frontera existe y es grande, pero el problema no
está en identificarla después de la predicción, está en cómo el objetivo global trata las fronteras**.
V18 cambia el objetivo de entrenamiento en vez del decision layer, y su pregunta es si el modelo global
puede usar mejor la información de las fronteras **sin aumentar la capacidad y sin introducir ninguna
representación nueva**.

## 2. Referencia absoluta

- **Canónica**: V12 pooled macro-F1 **0,6005017473** (redondeo 0,6005).
- **0,6004103057** es la **media de folds**, y debe seguir etiquetada como tal. **No** usarla como
  baseline canónico (V17 lo resolvió y `reports/v17_p0_reproduccion.md` lo explica).
- **Leaderboard**: 0,62188. No se usa para seleccionar nada.
- V12 no se modifica. Ninguna de las ramas anteriores se modifica.

## 3. Lo que la fase NO va a hacer

Prohibido por el brief y por D-040 a D-047: embeddings de cualquier tipo (incluidos los entrenados desde
cero), vectorizadores aprendidos nuevos, corpus o datasets externos, etiquetas externas,
pseudo-labels, etiquetas o texto de test para cualquier propósito, holdout2 para explorar, leaderboard
para seleccionar. Tampoco se reabren lexicons, decision layers, gating, pairwise generals, pruning,
aprendizaje de pesos global, meta-combinador genérico, decodificación de edad esperada ni
post-procesamiento ordinal: solo si la auditoría encuentra un defecto metodológico concreto y
nombrado.

## 4. Orden y puertas

| Fase | Contenido | Puerta |
|---|---|---|
| **P0** | Auditoría de brechas. Cada propuesta clasificada `CLOSED` / `PARTIALLY_TESTED` / `INVALIDLY_TESTED` / `GENUINELY_NEW` | Sin entrenar nada |
| **P0'** | Reproducir V12, A2 de V16 y los dos especialistas de V17, con sus fronteras y sus folds | Si no reproduce: **STOP** |
| **P1 · P1.1-P1.3** | Peso de clases en frontera, tres niveles, con control V12-equivalente | Reportar global, F1 por clase y las dos fronteras |
| **P2 · P2.1-P2.2** | Coste ordinal sensible en el **entrenamiento**, con matriz `coste(i,j) ∝ |i-j|` sin ajustar | Solo si P1 muestra señal |
| **P3 · P3.1-P3.2** | Margen de adyacencia en entrenamiento, primero solo en las dos centrales | Solo si P1-P2 muestran señal |
| **P4 · P4.1-P4.2** | Entrenamiento multi-objetivo: global + heads auxiliares de frontera, con y sin pérdida de frontera | Separar arquitectura de objetivo |
| **P5 · P5.1-P5.7** | Destilación de conocimiento de los especialistas, con **cross-fitting** obligatorio | Solo si P1-P4 demuestran que cambiar el objetivo tiene señal |
| **P6 · P7** | Destilación de miembros seleccionados y destilación de residuo | Solo si P5 funciona |
| **P9** | Análisis de error y destino de las victorias del especialista | Obligatorio para cada candidato |

Presupuesto: **14 experimentos principales y 6 de reserva**, sin gastar por obligación.

## 5. Prohibiciones metodológicas

- Partición **agrupada** histórica (`StratifiedGroupKFold` con grupos de casi duplicados más
  pseudo-autor), la misma de siempre. No se crean particiones nuevas salvo la segunda partición de
  los finalistas.
- **Ningún** ajuste de pesos, pérdidas, lambdas, temperaturas o costes puede depender de la validación
  exterior. Si depende de datos, se resuelve con `outer train`, luego `inner train`, luego `parameter selection`, luego
  `outer validation`.
- **Destilación**: la señal del teacher que entrena al student tiene que venir de **cross-fitting** —
  cada ejemplo predicho por un teacher que no lo usó en su entrenamiento. El teacher **nunca**
  reemplaza la etiqueta real: la etiqueta sigue siendo el objetivo principal y la del teacher es
  auxiliar. Los teachers son binarios y su señal es el **logit o margen de frontera**, sin
  reconvertirlos artificialmente en probabilidad de seis clases.
- **Adoptar exige ganancia global.** Una mejora de frontera sin mejora global no es candidata.
- Gate exploratorio Δ ≥ +0,0015 con ≥4/5 folds; gate de candidato Δ ≥ +0,003 con bootstrap
  consistente, segunda partición positiva, reproducibilidad y sin fuga.
- Una hipótesis se cierra con mecanismo válido y resultado estable negativo. No se responde "probar
  más hiperparámetros" sin hipótesis nueva.

## 6. La pregunta que la auditoría debe responder antes de nada

V17 dejó un dato que condiciona toda la fase, y el brief lo enuncia sin extraer la consecuencia:

- El especialista de frontera alcanza 0,6864 y 0,7218 en las filas de cada frontera.
- **V13 P3a ya entrenó un especialista binario sobre las 279 variables de V12**, con el mismo objetivo
  restringido al par, y obtuvo **0,4944 y 0,4942**, es decir **peor** que el propio V12 (0,4963 y 0,4974).
- La misma restricción de objetivo, sobre la representación dispersa TF-IDF, da 0,6864 y 0,7218.

Si esa descomposición se sostiene, entonces **la ventaja del especialista no viene del objetivo sino de
la representación**: V12 comprime a 6 números la salida de cada miembro, y esa compresión destruye la
información de frontera. En ese caso, cambiar el objetivo sobre la representación de V12 (P1, P2, P3)
no tiene por donde tomar la señal, y **la única pista con información disponible es P4**, que el propio
brief autoriza con "las heads deben utilizar la representación TF-IDF permitida".

**Esto no es una conclusión: es la primera pregunta de la auditoría de P0**, y hay que responderla con
evidencia de los artefactos, no repetirla como si fuera un hecho. Si se confirma, la fase se reenfoca y
se dice por qué; si no se confirma, la explicación es otra y también hay que publicarla.

## 7. Entregables

`reports/v18_gap_audit.{md,csv}` · `v18_master_results.csv` · `v18_hypothesis_registry.csv` ·
`v18_boundary_results.csv` · `v18_training_objectives.csv` · `v18_distillation.csv` ·
`v18_error_intersection.csv` · `v18_candidate_comparison.csv` · `v18_boundary_rescue.csv` ·
`v18_reproducibility.md` · `v18_compliance.md` · `v18_final_research.md` · figuras · integración en el
notebook consolidado · actualización de `PROJECT_STATE.md`, `DECISIONS.md`, `RUNLOG.md`,
`experiments.csv` y `FINAL_REPORT.md` **solo tras validar**.

## 8. Notas de coordinación

- Los subagentes reciben contexto dirigido, no el proyecto entero. El escaneo completo es **solo** de
  P0, por definición.
- Todo script usa `/home/carlos/venv/bin/python` y empieza con `autotest_numerico()`
  (`src/verificacion_numerica.py`).
- La correlación se calcula con `grama_exacta`, **nunca** con `A @ B.T` (D-040).
- Un experimento = un script nuevo; los scripts ya ejecutados no se editan.
- Auditoría de idioma y de caracteres antes de cada commit, con resultado 0 errores.
