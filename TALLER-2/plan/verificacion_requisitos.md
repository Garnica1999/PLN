# Verificación de requisitos del taller — TALLER-2

Fuente: `docs/Taller_Ontologias_2530.pdf` (enunciado del taller).
Estado: **8/8 requisitos cumplidos** (verificado 2026-08-30; extensión a 6 grupos aplicada el mismo día).

## 1. Selección del dominio (5 grupos de tesauros + justificación)
- **Requisito**: buscar y seleccionar 5 grupos de tesauros de la Comisión de la Verdad; justificar brevemente la selección.
- **Evidencia**: sección 02 del notebook (`sec02_seleccion.py`): criterios cuantitativos (tamaño 50-450, profundidad jerárquica, cobertura WordNet-español >= 1%, relevancia CEV, diversidad de ejes), tabla de candidatos, justificación por grupo y celda "Relevancia para el sector". Resultado: Actores armados (82), Impactos (107), Armas (94), Poblaciones (213), Hechos victimizantes (90) y Afrontamientos y resistencias no violentas - cotidianas (23) = 609 términos. Registro: `data/grupos_seleccionados.json` (ADR-001; extensión ADR-006).

## 2. Construcción de la ontología (entidades, relaciones, jerarquías)
- **Requisito**: construir la ontología en Protégé reflejando entidades, relaciones y jerarquías conceptuales; exportar en .owl o .rdf.
- **Evidencia**: sección 03 (`sec03_ontologia.py`): 623 clases (6 raíces + jerarquía completa + 6 clases definidas), 534 individuos (términos hoja), 17 object properties (5 estructurales + 12 del modelo semántico en 6 pares inversos: afectaA/esAfectadaPor, generaImpacto/esGeneradoPor, experimentaImpacto/impactaA, desarrollaAfrontamiento/esDesarrolladoPor, respondeA/esRespondidoMediante, empleaArma/esEmpleadaEn), 6 clases definidas con restricciones (equivalentClass), 7 axiomas raíz del ciclo del conflicto, 2 annotation properties. Archivo: `ontologia/tesauro_cev.owl` (RDF/XML, 9.812 triples). Validación: owlready2 + rdflib + HermiT (consistente y CLASIFICA) + apertura verificada en Protégé 5.6.9 (ADR-002).

## 3. Implementación de relaciones semánticas con WordNet (9 relaciones)
- **Requisito**: función en Colab que, con NLTK WordNet y la ontología creada, identifique y retorne los tipos de relación entre los conceptos de la ontología. Mínimas: sinonimia, antonimia, hiponimia, hiperonimia, meronimia, holonimia, troponimia, coordinadas, derivación morfológica.
- **Evidencia**: sección 04 (`sec04_wordnet.py`): `MotorRelacionesWordNet` (Facade) que recibe los conceptos de la ontología (leídos de la OWL vía `ServicioOntologia.conceptos()`) y devuelve `ResultadoRelaciones` con las 9 relaciones. Fábrica de extractores registra las 9 (verificación por tipo con conceptos reales + assertions). Resultados: 54.130 relaciones sobre 610 conceptos (cobertura 50,5%).

## 4. Consultas SPARQL (15 = 3 por grupo)
- **Requisito**: 3 consultas SPARQL diferentes por cada uno de los 5 grupos (total 15), cada una documentada (qué información busca y cómo se relaciona con los tesauros y la ontología).
- **Evidencia**: sección 05 (`sec05_sparql.py`): **22 consultas** (18 de grupos = 3 por cada uno de los 6 grupos + 4 del modelo semántico: clases definidas, pares inversos, ciclo completo Actor→Hecho→Población→Impacto→Afrontamiento, individuos por grupo) ejecutadas contra `tesauro_cev.owl` con rdflib, cada una con markdown de documentación (objetivo + relación con TG/TE/TR) y tabla de resultados.

## 5. Entrega: notebook único
- **Requisito**: un notebook en Google Colab con la implementación de las funciones y las 15 consultas SPARQL documentadas.
- **Evidencia**: `notebooks/Taller_Ontologias_WordNet.ipynb` (Colab-compatible: sec00 instala dependencias y descarga corpus; rutas relativas; celda final "Entrega a CampusVirtual"). 120 celdas, ejecutado end-to-end con 0 errores.

## 6. Archivo de la ontología OWL
- **Requisito**: el archivo de la ontología en formato OWL construida en Protégé.
- **Evidencia**: `ontologia/tesauro_cev.owl` (OWL 2, RDF/XML), generada con owlready2 y validada abriéndola en Protégé 5.6.9 (arranque + carga sin errores).

## 7. Reflexión: ontologías vs. análisis plano de texto en PLN
- **Requisito**: breve reflexión sobre las ventajas de representar conocimiento en ontologías frente a un análisis plano de texto.
- **Evidencia**: sección 06 (`sec06_conclusiones.py`) — reflexión con 4 argumentos (semántica explícita y compartida, razonamiento/consultas formales, reutilización y evolución, complementariedad con técnicas planas) + informe §6.

## 8. Calidad y trazabilidad
- **Requisito implícito de la rúbrica**: código limpio, gráficos con análisis, informe escrito, conclusiones.
- **Evidencia**: núcleo POO con patrones de diseño y SOLID (sec00_clases.py), 14 figuras con interpretación, `informe/informe_taller2.md` (4.437 palabras), tests unitarios en `tests/` (60 tests, no incluidos en la entrega), ADRs en `plan/decisiones/`.
