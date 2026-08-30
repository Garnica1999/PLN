# Verificación de rúbrica criterio por criterio — TALLER-2

Fuente: `docs/Taller - Ontologías - Procesamiento lenguaje natural - U Javeriana Bogota.pdf`.
Cada criterio vale 25 puntos (Excelente). Objetivo: **25/25 en los 8 criterios (200/200)**.
Estado: verificado 2026-08-30; extensión a 6 grupos + modelo semántico fuerte aplicada y re-verificada el mismo día.

## Criterio 1 — Selección y justificación de la ontología (25)
- **Nivel Excelente**: "La ontología seleccionada es altamente relevante para el sector específico y está bien documentada; la justificación demuestra un entendimiento profundo de la importancia de la elección."
- **Evidencia**: selección automática con 5 criterios cuantitativos documentados (sec02); justificación por grupo con números (tamaño, profundidad, cobertura WordNet, eje); celda nueva "Relevancia para el sector" que conecta la elección con el mandato de la CEV (memoria histórica, ciclo del hecho victimizante, utilidad SPARQL); informe §3.2.
- **Cierre aplicado**: celda markdown de relevancia sectorial añadida en sec02.

## Criterio 2 — Implementación del programa en Python (25)  
**Refuerzo (extensión)**: el modelo semántico fuerte (17 object properties con 6 pares inversos, 6 clases definidas y 7 axiomas del ciclo) eleva el nivel de la implementación sobre WordNet+OWL.
- **Nivel Excelente**: "El programa está correctamente implementado y cumple con todos los requisitos utilizando WordNet."
- **Evidencia**: función `extraer_relaciones` (MotorRelacionesWordNet, sec04) que recibe conceptos de la ontología (leídos de la OWL) y devuelve las 9 relaciones; normalización robusta (path, plurales, acentos, multiword); resultados verificados (cobertura 49,7%, 51.795 relaciones); asserts en la celda de verificación.
- **Cierre aplicado**: celda de verificación de las 9 relaciones con conceptos reales.

## Criterio 3 — Identificación de entidades y relaciones (25)  
**Refuerzo (extensión)**: 610 conceptos de 6 grupos clasificados; verificación de las 9 relaciones con conceptos reales incluido el grupo Afrontamientos; el razonador clasifica automáticamente (HechoPerpetradoPorActor, PoblacionConAfrontamiento).
- **Nivel Excelente**: "Todas las entidades y sus relaciones están correctamente identificadas y clasificadas utilizando WordNet."
- **Evidencia**: 592 conceptos de la ontología clasificados por grupo; las 9 relaciones demostradas una a una con ejemplos reales (sinonimia: verdad→realidad/evidencia; antonimia: mujer→hombre/masculino; hiponimia/hiperonimia: desplazamiento→cambio; meronimia/holonimia: ejemplos de Armas/Poblaciones; troponimia: verbos; coordinados y derivación: desplazamiento→desplazarse/trasladar); limitaciones de omw-1.4 reportadas honestamente (cobertura por grupo en figuras).

## Criterio 4 — Calidad del código (25)
- **Nivel Excelente**: "El código es limpio, funcional y sigue buenas prácticas de programación."
- **Evidencia**: núcleo POO con patrones de diseño (Repository, Singleton+Facade, Abstract Factory+Registry, Strategy, Facade); SOLID y DRY (imports y constantes únicos en sec00; sin duplicación entre secciones); ruff 0 errores; docstrings en español; sin emojis; 60 tests unitarios en `tests/`; revisión de `ml-reviewer`.
- **Cierre aplicado**: refactor completo de las 6 secciones al núcleo POO; tests añadidos.

## Criterio 5 — Gráficos y análisis (25)  
**Refuerzo (extensión)**: grafo completo actualizado a 1.158 nodos/1.286 aristas con 6 grupos y vista contraída de 7 nodos/12 aristas que muestra el ciclo del conflicto.
- **Nivel Excelente**: "Los gráficos presentados son visualmente claros... acompañados de una explicación detallada y coherente."
- **Evidencia**: 14 figuras PNG en `imagenes/` (EDA: 6; ontología: árbol + grafo completo 1.110 nodos + vista contraída; WordNet: 5 con heatmaps, grafos y scatters), cada una con celda markdown de interpretación cuantitativa; nueva sección sec03b de visualización del grafo completo como validación visual.
- **Cierre aplicado**: sec03b (grafo_ontologia_completo.png + grafo_ontologia_grupos.png).

## Criterio 6 — Calidad del informe escrito (25)
- **Nivel Excelente**: "El informe está bien estructurado, detallado y cumple con los criterios establecidos."
- **Evidencia**: `informe/informe_taller2.md` (4.437 palabras, 7 secciones, tablas de resultados, figuras, referencias), actualizado con arquitectura de código (POO/SOLID), visualización de la ontología y verificación de requisitos.
- **Cierre aplicado**: subsecciones 3.6 (arquitectura), 4.5 (visualización), 4.6 (verificación) y cifras actualizadas.

## Criterio 7 — Conclusiones y reflexión crítica (25)
- **Nivel Excelente**: "Las conclusiones son profundas y demuestran una comprensión sólida de los resultados obtenidos."
- **Evidencia**: sec06 con síntesis tabular, 4 conclusiones con datos, reflexión explícita "¿Qué ventajas aporta representar conocimiento en ontologías frente a un análisis plano de texto en PLN?" (4 argumentos + complementariedad); informe §6 con la misma reflexión desarrollada.

## Criterio 8 — Formato y entrega (25)
- **Nivel Excelente**: "El trabajo cumple completamente con las indicaciones de formato y entrega, en la fecha establecida."
- **Evidencia**: notebook único Colab-compatible con portada (integrantes editables), setup automático (pip + corpus NLTK), rutas relativas, celda final "Entrega a CampusVirtual" listando los archivos (notebook + OWL); archivo OWL entregable; estructura del repositorio documentada en README.
- **Cierre aplicado**: celda de entrega en sec06.

## Resumen
| Criterio | Nivel objetivo | Estado |
|---|---|---|
| 1. Selección/justificación | Excelente (25) | OK |
| 2. Implementación Python | Excelente (25) | OK |
| 3. Entidades y relaciones | Excelente (25) | OK |
| 4. Calidad del código | Excelente (25) | OK |
| 5. Gráficos y análisis | Excelente (25) | OK |
| 6. Informe escrito | Excelente (25) | OK |
| 7. Conclusiones/reflexión | Excelente (25) | OK |
| 8. Formato y entrega | Excelente (25) | OK |
| **Total** | **200/200** | **Verificado (tras extensión a 6 grupos + modelo semántico)** |
