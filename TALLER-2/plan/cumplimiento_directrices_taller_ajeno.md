# Matriz de cumplimiento de directrices — Taller resuelto (Taller_ontologia.rar)

Verificación del taller entregado por otros estudiantes contra las directrices de los 2 PDFs del curso.
Fuentes: `docs/Taller_Ontologias_2530.pdf` (enunciado) y `docs/Taller - Ontologías...pdf` (rúbrica).
Archivos revisados: `Taller_ontologia/Taller_PLN_Ontologias.ipynb`, `ontologia_comision_verdad.owl`,
`ontologia.png`, `tesauro.xlsx` (extraídos en /mnt/.../TALLER II/).
Revisión: 2026-08-30 (HermiT + rdflib + ejecución del notebook + ruff).

## A. Enunciado (Taller_Ontologias_2530.pdf)

| # | Requisito | Veredicto | Evidencia |
|---|---|---|---|
| 1 | Seleccionar 5 grupos de tesauros + justificar | **CUMPLE** | Actores armados (82), Afrontamientos y resistencias no violentas - cotidianas (23), Hechos victimizantes (90), Impactos (107), Poblaciones (213) = 515 términos. Justificación por ciclo conceptual: Actor → Hecho → Población → Impacto → Afrontamiento (md celdas 31-47). Sin criterios cuantitativos, pero justificación razonada. |
| 2 | Construir ontología en Protégé: entidades, relaciones, jerarquías; exportar .owl | **CUMPLE** | OWL válida: 517 clases, 19 object properties (9 pares inversos), 6 clases definidas con restricciones, 11 individuos, 508 subClassOf. **Consistente (HermiT: 0 inconsistentes)**. Carga OK en Protégé 5.6.9. Jerarquía fiel al tesauro: 94,4% de coincidencia exacta de paths (5 errores menores reales: LGBTI, DESCA intermedio, OSIGD con jerarquía rota, MAP-MUSE separado, MUSE duplicado). |
| 3 | Función Python (NLTK WordNet) que reciba conceptos de la ontología y devuelva las 9 relaciones | **CUMPLE** | `obtener_relaciones_wordnet(concepto)` + `analizar_ontologia_con_wordnet(OWL_PATH)` extrae los conceptos de la OWL y devuelve las 9 relaciones (sinonimia, antonimia, hiponimia, hiperonimia, meronimia, holonimia, troponimia, coordinadas, derivación). Verificado: las 9 con >0 filas (2.298 relaciones en total). Limitación: **no reportan cobertura** (solo ~6,4% de los conceptos se encuentran en WordNet español). |
| 4 | 15 consultas SPARQL (3×5) documentadas: qué busca + relación con tesauro/ontología | **CUMPLE** | 15 consultas ejecutadas y verificadas con rdflib contra su OWL, todas con resultados (30/81/3/23/23/2/22/16/4/8/8/6/44/9/3 filas) y documentadas con "¿Qué información busca?" y "Relación con el tesauro y la ontología". |
| 5 | Entrega: notebook + OWL + **reflexión** (ventajas ontologías vs. análisis plano de texto en PLN) | **NO CUMPLE** | Notebook y OWL presentes; **la reflexión NO existe** (0 celdas con "reflexi"; el notebook termina en la consulta 5.3). Es un ítem explícito del enunciado. |

**Resumen enunciado: 4/5 requisitos cumplidos (falta la reflexión).**

## B. Rúbrica (8 criterios × 25 pts)

| Criterio | Nivel estimado | Evidencia / riesgo |
|---|---|---|
| 1. Selección y justificación de la ontología | **Bueno (20)** | Justificación conceptual coherente con el sector (ciclo del conflicto) pero sin datos cuantitativos ni documentación del proceso de selección. |
| 2. Implementación del programa en Python | **Excelente (25)** | Las 9 relaciones correctamente implementadas con WordNet; funciones documentadas (docstrings NumPy), pipeline en capas, manejo de concepto no encontrado, limpieza de sinonimia auto-referente. |
| 3. Identificación de entidades y relaciones | **Bueno (20)** | Entidades de la ontología extraídas y clasificadas por tipo de relación, con tablas por relación; pero **sin análisis de cobertura** (solo ~6,4% de conceptos encontrados; la mayoría del análisis recae en pocos ejemplos) y sin estadísticas agregadas. |
| 4. Calidad del código | **Bueno (20)** | Código limpio, type hints, funciones pequeñas; sin patrones ni tests ni ruff (no exigidos). 19 hallazgos ruff: 17 son B018 esperables en Jupyter; I001 y UP035 menores. |
| 5. Gráficos y análisis | **Insuficiente (10)** | **0 figuras en el notebook**; `ontologia.png` (árbol OWLViz 2236×15992) existe en la carpeta pero NO está integrado ni explicado. La rúbrica exige gráficos con explicación. |
| 6. Calidad del informe escrito | **Satisfactorio (15)** | No hay informe separado; el markdown del notebook está bien estructurado (objetivos, comprensión, EDA básica, modelo, consultas) pero no constituye un informe completo con discusión y referencias. |
| 7. Conclusiones y reflexión crítica | **Insuficiente (10)** | **No hay sección de conclusiones ni reflexión** (el taller la pide explícitamente). |
| 8. Formato y entrega | **Satisfactorio (15)** | Entregado en .rar con notebook ejecutado (0 errores), OWL, PNG y tesauro procesado; **falta la reflexión**; celda de integrantes vacía; reproducibilidad frágil (rutas /content de Colab, subida manual de archivos). |
| **Total estimado** | **~135/200** | (20+25+20+20+10+15+10+15) |

## C. Notas adicionales de la revisión técnica

**Fortalezas reales del taller ajeno:**
- Modelo semántico rico: 19 object properties con dominio/rango concretos, 9 pares inversos, `relacionadoCon` simétrica y **6 clases definidas con restricciones** (p. ej. Población afectada ≡ Población AND esAfectadaPor SOME HechoVictimizante) — un aporte genuino sobre el tesauro.
- Trazabilidad total: anotación `rutaTesauro` con el path original en todas las clases (97,1%).
- Ontología consistente (HermiT) y fiel al tesauro original (94,4% de paths exactos).
- Consultas SPARQL verificadas una a una: todas válidas y con resultados correctos.

**Debilidades detectadas:**
- IRI `example.org` sin personalizar (debería ser un dominio institucional).
- Doble etiquetado (rdfs:label + skos:prefLabel sin declarar el prefijo skos).
- `relacionadoCon` con dominio/rango owl:Thing (demasiado genérica).
- 5 errores de modelado vs tesauro (LGBTI, DESCA, OSIGD, MAP-MUSE, MUSE duplicado).
- Solo 11 individuos (el resto de términos son clases, no entidades).
- Sin cobertura WordNet reportada: 93,6% de los conceptos no se encuentran en español (sin fallback).
- Sin figuras, sin reflexión, sin conclusiones, sin informe escrito.
