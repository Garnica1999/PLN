# Fase 00 — Intención confirmada de V17

- **Fecha**: 2026-09-26
- **Rama**: `kaggle-age-2026-puj-v17` (creada desde `a818349`, árbol limpio)
- **Ámbito de trabajo**: exclusivamente `KAGGLE_AGE_2026_PUJ/`
- **Origen de esta intención**: brief de V17 escrito por Carlos (53 secciones). Es exhaustivo en
  objetivo, restricciones, orden y entregables, así que **no seeedió entrevista de requisitos**:
  hacerla habría repetido lo que el brief ya fija. Este documento es la transcripción operativa de
  ese brief, no una interpretación.

## 1. Objetivo único

V16 encontró que el combinador logístico sobre las dos cabezas de V12 mejora **solo** en las dos
fronteras centrales: `18-24|25-34` +0,0102 y `25-34|35-49` +0,0157, con un efecto global de solo
+0,0006 que no supera el gate. La pregunta de V17 es si esa información complementaria **se
convierte en ganancia global** cuando se usa exclusivamente en las regiones donde ha demostrado
utilidad.

La novedad de la fase es **entrenamiento especializado + gating local + modelos pairwise
supervisados**. Todo lo demás está cerrado.

## 2. Lo que queda prohibido (reiterado del brief, para que ningún agente lo reabra)

- **Embeddings de cualquier tipo**, incluidos los entrenados desde cero: Word2Vec, FastText, GloVe,
  autoencoders de texto, embeddings aprendidos de token, sentence embeddings, sentence-transformers,
  BERT/RoBERTa/transformers, LLM, modelos o corpus externos, etiquetas externas, pseudo-labeling.
- **No** usar test (texto ni etiquetas) para entrenamiento, selección de variables, calibración,
  hiperparámetros o estadísticas supervisadas.
- **No** usar el holdout histórico para explorar. **No** usar el leaderboard para elegir thresholds,
  especialistas, folds o fronteras. **No** enviar submissions durante la exploración.
- **No** podar V12 (V15 lo_closed con Δ −0,0040). **No** tocar V12, V15 ni V16, ni las submissions
  históricas. **No** modificar remotes, no hacer push, no reescribir historia, no merge automático.

## 3. Cerrado definitivamente salvo que la auditoría encuentre un bug

Investigación léxica · capas de decisión genéricas · decision de edad esperada · Luce ·
Bradley-Terry como capa de decisión · proyección ordinal · poda de features · aprendizaje de pesos
global · búsqueda de combinador global · ensemble V12 existente · búsqueda genérica de TF-IDF.

La auditoría de la fase debe confirmar que las formulaciones concretas que se quiera ejecutar
(P2 con LogReg/LinearSVC sobre pares de clases, P4 con modelos entrenados sobre pares adyacentes,
P5 con combinador local) **no** son sinónimos de ninguna de las anteriores.

## 4. Orden de ejecución y puertas

| Fase | Contenido | Puerta |
|---|---|---|
| **P0** | Auditoría de brechas: inventario de lo hecho, lo cerrado y lo genuinamente nuevo | Obligatoria, sin entrenar nada |
| **P0'** | Reconciliar la discrepancia 0,6004 vs 0,6005 y reproducir A2 de V16 | Si A2 no reproduce: **STOP** |
| **P1 · P1.1** | Matriz de error por miembro **en las fronteras** y complementariedad local | Si no hay complementariedad: cerrar P2/P3 sin entrenar |
| **P2 · P2.1 · P2.2** | Especialistas reales `18-24 vs 25-34` y `25-34 vs 35-49`, protocolo anidado | P2.3 (tercera frontera) solo si el enfoque funciona |
| **P3 · P3.1-P3.3** | Gating de ambigüedad: hard gate, luego soft gate | Solo si hay señal |
| **P4 · P4.1-P4.2** | Entrenamiento pairwise real (adyacente primero) | Solo si los especialistas son útiles |
| **P5 · P5.1-P5.2** | Meta-combinador local por frontera con inner OOF | Solo con señal |
| **P6 a P10** | Residual, coeficientes específicos de frontera, perfil de confianza, experto selectivo, decodificación local | Solo si P2/P3/P5 muestran señal y separación por confianza |

Presupuesto: **16 experimentos principales y 6 de reserva**, sin gastar por obligación. El orden
`P6-P10` es condicional, no automático.

## 5. Referencia absoluta y canonización pendiente

- **Leaderboard**: 0,62188, referencia externa inmutable.
- **CV**: el brief exige reconciliar 0,6004 frente a 0,6005 **antes de cualquier experimento**,
  determinando valor canónico, valor reproducido y motivo de la diferencia, sin sustituirlos en
  silencio. Esa reconciliación es parte de P0 y no es un experimento.
- **Gate de V17**: exploratorio Δ ≥ +0,0015 con ≥4/5 folds; de adopción Δ ≥ +0,003 con IC95 de
  bootstrap consistente, segunda partición positiva y mejora no concentrada en una sola fold. Una
  mejora grande de frontera con macro-F1 global ≤ baseline **no es candidata**: es evidencia
  mecánica, no adopción.

## 6. Prohibiciones metodológicas que se aplican a todo agente que trabaje en esta fase

- Ninguna decisión (selección de variables, umbral de gate, hiperparámetros, calibración) puede
  usar la validación exterior.
- Los combinadores se entrenan **solo con predicciones OOF internas**; nunca con predicciones de
  modelos que ya vieron ese ejemplo. El patrón prohibido es `entrenar completo -> generar OOF ->
  optimizar gate -> reportar ese mismo OOF`.
- El especialista se entrena **exclusivamente** con `outer_train ∩ {las dos clases de la frontera}` y
  se evalúa **exclusivamente** en `outer_validation ∩ {esas dos clases}`.
- Si el protocolo anidado de P6 resulta demasiado complejo, **se cierra P6** en lugar de
  introducir fuga.
- Una hipótesis se cierra con `mecanismo válido + resultado estable negativo`. No se responde
  "probar más hiperparámetros" sin hipótesis nueva.

## 7. Entregables

`reports/v17_gap_audit.{md,csv}` · `reports/v17_master_results.csv` ·
`reports/v17_hypothesis_registry.csv` · `reports/v17_member_boundary_matrix.csv` ·
`reports/v17_boundary_analysis.csv` · `reports/v17_error_intersection.csv` ·
`reports/v17_candidate_comparison.csv` · `reports/v17_reproducibility.md` ·
`reports/v17_compliance.md` · `reports/v17_final_research.md` · **10 figuras mínimas** ·
integración en el notebook consolidado · actualización de `PROJECT_STATE.md`, `DECISIONS.md`,
`RUNLOG.md`, `experiments.csv` y `FINAL_REPORT.md` **solo tras validar los resultados**.

## 8. Clasificación final obligatoria

Cada línea se cierra como `CLOSED` · `MARGINAL` · `COMPLEMENTARY` · `PROMISING` · `FINALIST`.
`PROMISING` **no** significa adoptada: solo un candidato que mejore la métrica global puede serlo.

## 9. Pregunta científica que la fase debe responder

¿La información adicional que V16 descubrió en las dos fronteras centrales puede explotarse mediante
modelos entrenados específicamente para esas fronteras y activados solo cuando V12 es ambiguo?
Separando: (A) V12 no tiene información · (B) un miembro de V12 sí la tiene · (C) un especialista
aprende información nueva · (D) el problema es solo de decisión · (E) se resuelve con gating ·
(F) la mejora local no tiene impacto global.

## 10. Notas de coordinación

- Los subagentes reciben contexto dirigido (rutas, esta spec, y el resumen de `plan/contexto.md`),
  no el proyecto entero. El escaneo completo del proyecto es **solo** de P0, por definición.
- Todo script se ejecuta con `/home/carlos/venv/bin/python` y empieza con `autotest_numerico()`
  (`src/verificacion_numerica.py`).
- La correlación se calcula con `grama_exacta`/`einsum`, **nunca** con `A @ B.T` (D-040).
- Un experimento = un script nuevo; los scripts ya ejecutados no se editan.
- Auditoría de idioma y de caracteres antes de cada commit, con resultado 0 errores.
