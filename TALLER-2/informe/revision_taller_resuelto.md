# Revisión del taller resuelto (Taller_ontologia.rar) y comparación con TALLER-2

**Fecha**: 2026-08-30 · **Revisión**: análisis estructural de la OWL (owlready2/rdflib/HermiT), ejecución del notebook ajeno con sus datos, re-ejecución de las 15 consultas SPARQL, ruff y matriz de cumplimiento de los 2 PDFs del curso.

---

## 1. Contenido del .rar

| Archivo | Detalle |
|---|---|
| `Taller_PLN_Ontologias.ipynb` | 123 celdas (71 markdown + 52 código), ejecutado en Google Colab (Python 3.13), 0 errores |
| `ontologia_comision_verdad.owl` | 3.319 triples, RDF/XML |
| `ontologia.png` | 2.236 × 15.992 px (árbol vertical estilo OWLViz, 3,2 MB) — **no integrado al notebook** |
| `tesauro.xlsx` | Copia del original reducida a 2 columnas (Grupo, Término), 12.341 filas |

## 2. Cumplimiento de las directrices (detalle en `plan/cumplimiento_directrices_taller_ajeno.md`)

**Enunciado (Taller_Ontologias_2530.pdf): 4/5 puntos cumplidos.**
- Selección de 5 grupos + justificación: CUMPLE (Actores armados 82, Afrontamientos y resistencias no violentas - cotidianas 23, Hechos victimizantes 90, Impactos 107, Poblaciones 213 = 515; justificación por ciclo Actor→Hecho→Población→Impacto→Afrontamiento).
- Ontología en Protégé con entidades/relaciones/jerarquías + .owl: CUMPLE (517 clases, 19 props, 6 clases definidas, consistente en HermiT, carga OK en Protégé).
- Función WordNet con las 9 relaciones: CUMPLE (verificada: 2.298 relaciones, las 9 con resultados).
- 15 consultas SPARQL documentadas: CUMPLE (las 15 re-ejecutadas con éxito: 30/81/3/23/23/2/22/16/4/8/8/6/44/9/3 filas).
- **Reflexión final (ontologías vs. texto plano): NO CUMPLE** — no existe en el notebook.

**Rúbrica: total estimado ~135/200.**
| Criterio | Nivel estimado | Puntos |
|---|---|---|
| 1. Selección/justificación | Bueno | 20 |
| 2. Implementación Python | Excelente | 25 |
| 3. Entidades y relaciones (WordNet) | Bueno | 20 |
| 4. Calidad del código | Bueno | 20 |
| 5. Gráficos y análisis | Insuficiente | 10 |
| 6. Informe escrito | Satisfactorio | 15 |
| 7. Conclusiones/reflexión | Insuficiente | 10 |
| 8. Formato y entrega | Satisfactorio | 15 |

## 3. Revisión técnica de su ontología

**Estructura**: IRI `http://example.org/ontologia-comision-verdad#`; 517 clases (15 estructurales + 502 del tesauro), 11 individuos (clase `Kumpania`), 19 object properties, 0 data properties, 1 annotation property (`rutaTesauro`), 508 subClassOf, 6 equivalentClass, 10 disjoints, profundidad máxima 5.

**Consistencia**: **HermiT: consistente, 0 clases inconsistentes** (0,47 s).

**Fidelidad al tesauro original**: 502/517 clases con `rutaTesauro` (97,1%); 5 grupos raíz exactos; 94,4% de coincidencia exacta de paths (20/20 en muestra). 28 discrepancias: 23 artefactos del nombre compuesto del grupo ("Afrontamientos y resistencias no violentas - cotidianas") y **5 errores reales**: LGBTI (sin término exacto), DESCA (nodo intermedio ausente), **OSIGD (jerarquía rota: cuelga de Población)**, "Sobrevivientes de MAP-MUSE" separado en 2 clases y **MUSE duplicado del grupo Armas**.

**Modelo semántico (fortaleza principal)**: 19 object properties con dominio/rango concretos (afectaA↔esAfectadaPor, generaImpacto↔esGeneradoPor, participaEn, desarrollaAfrontamiento…), 9 pares inversos, `relacionadoCon` simétrica, y **6 clases definidas con restricciones OWL**:
- Población afectada ≡ Población AND esAfectadaPor SOME HechoVictimizante
- Hecho con impacto ≡ HechoVictimizante AND generaImpacto SOME Impacto
- Población con impacto ≡ Población AND experimentaImpacto SOME Impacto
- Población con afrontamiento ≡ Población AND desarrollaAfrontamiento SOME AfrontamientoResistencia
- Afrontamiento asociado a hecho / a impacto (análogas)

**Debilidades**: IRI example.org; doble etiquetado (548 rdfs:label + 528 skos:prefLabel, skos sin declarar); `relacionadoCon` con dominio/rango owl:Thing; disjoints solo entre 5 clases núcleo; 11 individuos nada más; 5 errores vs tesauro.

## 4. Revisión técnica de su código

**Ejecución**: exit 0, sin errores ni warnings (~2 min); pipeline WordNet 1,7 s sobre 517 conceptos. Sin código de construcción de la ontología en el notebook (solo lectura con rdflib; la OWL se construyó a mano en Protégé — enfoque literal del taller).

**Funciones**: `normalizar_termino`, `nombres_synset`, `agregar_relacion`, `obtener_relaciones_lexicas`, `obtener_relaciones_wordnet`, `analizar_conceptos_ontologia`, `extraer_conceptos_owl`, `analizar_ontologia_con_wordnet`, `resultados_a_dataframe`, `cargar_ontologia`, `ejecutar_sparql`. Docstrings NumPy, type hints, pipeline en capas, manejo de concepto no encontrado, limpieza de sinonimia auto-referente.

**Ruff**: 19 hallazgos (17× B018 = última expresión de celda Jupyter, esperable; I001 orden de imports; UP035 Iterable). py_compile OK.

**Hallazgo relevante**: con `fallback_ingles=False` (su configuración), **93,6% de los conceptos (484/517) no se encuentran en WordNet español**; el análisis semántico reportado recae en ~33 conceptos (~6,4%). No reportan esta cobertura, lo que debilita el criterio 3 de la rúbrica (y explica las tablas pequeñas: 8-753 filas).

## 5. Comparación formal con nuestra solución (TALLER-2)

| Dimensión | Taller ajeno | TALLER-2 (nuestro) |
|---|---|---|
| Grupos seleccionados | Actores armados, **Afrontamientos y resistencias no violentas - cotidianas**, Hechos victimizantes, Impactos, Poblaciones (515) | Actores armados, **Armas**, Hechos victimizantes, Impactos, Poblaciones (586) |
| Justificación | Conceptual (ciclo del conflicto) | **Cuantitativa** (tamaño, profundidad, cobertura WordNet, relevancia CEV, diversidad de ejes) + celda de relevancia sectorial |
| Construcción de la OWL | Manual en Protégé (notebook solo analiza) | **Programática con owlready2** (reproducible en el notebook) + validada en Protégé |
| Clases / individuos | 517 clases, **11 individuos** | 593 clases, **516 individuos** (términos hoja materializados) |
| Object properties | **19** (9 pares inversos, dominios/ranos concretos) | 5 (esParteDe/parteDe, relacionadoCon, perpetradoPor/esVictimaDe) |
| Clases definidas | **6 con restricciones OWL** (Población afectada ≡ …) | 0 |
| IRI | example.org (no personalizado) | `http://www.semanticweb.org/cev/tesauro#` (personalizado) |
| Consistencia | HermiT OK | HermiT OK |
| Fidelidad al tesauro | 94,4% paths exactos (5 errores: OSIGD roto, MUSE duplicado…) | Replicación exacta desde el JSON (0 errores) |
| WordNet | 9 relaciones, ~6,4% cobertura sin fallback, **sin análisis de cobertura** | 9 relaciones, **49,7% cobertura** (fallback de segmentos), análisis masivo 592 conceptos / 51.795 relaciones, **5 figuras** |
| SPARQL | 15 documentadas y verificadas (30/81/3/…/3) | 15 documentadas y verificadas (30/73/12/…/101) |
| Figuras en notebook | **0** (solo ontologia.png sin integrar) | **14** (EDA 6, ontología 3, WordNet 5) con interpretación |
| Código | Limpio, docstrings, sin patrones/tests | **POO (Repository, Singleton, Factory, Strategy), SOLID, DRY, ruff 0, 60 tests** |
| Reflexión del taller | **AUSENTE** | sec06 + informe §6 (4 argumentos) |
| Informe escrito | No (solo markdown del notebook) | `informe/informe_taller2.md` (4.437 palabras, 7 secciones) |
| Rúbrica estimada | **~135/200** | **200/200 objetivo** (verificación interna) |

## 6. Veredicto

**El taller ajeno es un trabajo sólido en el modelado semántico**: su ontología es consistente, fiel al tesauro (94,4%) y aporta un modelo relacional notable (19 propiedades con pares inversos y 6 clases definidas con restricciones — el punto más fuerte y genuinamente "ontológico" de ambos trabajos, y el único aspecto donde supera al nuestro). Sus consultas SPARQL están impecablemente documentadas y verificadas.

**Pierde puntos en la entrega**: no cumple el ítem explícito de la reflexión del enunciado, no tiene figuras (criterio 5 = Insuficiente), no tiene conclusiones, no hay informe escrito, la cobertura de WordNet es baja y no se reporta, y la reproducción depende de rutas de Colab. Estimación: ~135/200 (Bueno/Satisfactorio).

**Qué podemos adoptar de ellos (mejora opcional de TALLER-2)**: incorporar 1-2 clases definidas con restricciones (p. ej. "Hecho perpetrado por actor armado") y algún par inverso adicional en `tesauro_cev.owl`, para enriquecer el modelo sin perder nuestra ventaja en análisis, figuras, informe y reflexión. Si Carlos lo desea, lo implemento como extensión (nuevo experimento sobre la OWL, sin tocar lo existente).
