# %% [markdown]
"""# Sec06 — Conclusiones y reflexión crítica

## Síntesis del taller

| Componente | Resultado |
|---|---|
| Fuente | Tesauro de la Comisión de la Verdad (12.341 términos, 41 grupos) |
| Grupos seleccionados (6) | Actores armados (82), Impactos (107), Armas (94), Poblaciones (213), Hechos victimizantes (90), Afrontamientos y resistencias no violentas (23) — 609 términos |
| Ontología OWL | `tesauro_cev.owl`: 623 clases, 534 individuos, 17 object properties, 6 clases definidas, consistente (HermiT) |
| Modelo semántico | 7 axiomas raíz del ciclo del conflicto (perpetradoPor, afectaA, generaImpacto, experimentaImpacto, desarrollaAfrontamiento, respondeA, empleaArma) con sus inversas |
| Relaciones WordNet (9 tipos) | 54.130 relaciones sobre 610 conceptos; cobertura 50,5% |
| Consultas SPARQL | 22 (3 por grupo × 6 grupos + 4 del modelo semántico), documentadas y ejecutadas contra la ontología |

## Conclusiones

1. **El tesauro CEV es un recurso jerárquico denso**: el 74% de sus términos se
   ubica entre 3 y 5 niveles de profundidad, lo que lo hace un insumo natural
   para ontologías (la jerarquía ya está codificada en el término). La
   conversión a OWL preservó íntegramente esa estructura (subclassOf + instancias
   para términos hoja), ahora con 6 grupos: los 534 términos hoja quedaron
   instanciados como individuos consultables.

2. **WordNet cubre parcialmente el vocabulario del conflicto colombiano**: la
   cobertura exacta en el español de omw-1.4 es solo del 6,1% y sube al 50,5%
   con el fallback de segmentos; grupos temáticos centrales (Poblaciones, 16,7%)
   quedan por debajo de la media por su vocabulario especializado (pueblos
   étnicos, OSIGD, LGBTI), ausente de un recurso léxico general como WordNet.

3. **Las relaciones semánticas son informativas pero ruidosas**: hiponimia
   (21.806) y coordinados (23.463) concentran la mayor parte del volumen por la
   polisemia de los núcleos léxicos de fallback ("vida", "familia", "género"),
   mientras que sinonimia (2.473) y derivación (1.460) son más selectivas. Esto
   refuerza la necesidad de delimitar el dominio al consultar recursos léxicos
   genéricos.

4. **SPARQL convierte la jerarquía en conocimiento consultable**: las 22
   consultas demuestran recuperación de subárboles (TG/TE), entidades por grupo,
   relaciones asociativas (TR), rutas de 2 niveles y conteos, con tiempos de
   milisegundos una vez cargada la ontología.

5. **El modelo semántico convierte la ontología en una base inferible**: las 6
   clases definidas (`HechoPerpetradoPorActor`, `PoblacionAfectadaPorHechos`,
   `PoblacionConImpacto`, `PoblacionConAfrontamiento`, `HechoQueGeneraImpacto`,
   `AfrontamientoAsociadoAHecho`) se declaran con `owl:equivalentClass` (base AND
   restricción), de modo que HermiT clasifica automáticamente quién es víctima,
   quién está afectado o qué hecho genera impacto, sin declararlo a mano. Los 7
   axiomas raíz y sus inversas permiten consultar el ciclo completo del conflicto
   (Actor → Hecho → Población → Impacto → Afrontamiento) en una sola cadena
   SPARQL, algo imposible sobre el tesauro plano.

6. **El grupo de Afrontamientos y resistencias aporta valor temático pese a su
   tamaño**: con 23 términos es el grupo más pequeño y su vocabulario (juntas de
   acción comunal, convivio, silencio como afrontamiento, redes de apoyo) no
   quedó incluido en el CSV de cobertura WordNet, que se generó con los 5 grupos
   iniciales; por su especificidad cultural es razonable esperar una cobertura
   baja, como la de Poblaciones (1,41%). Su valor no es cuantitativo sino
   conceptual: documenta la respuesta social y cotidiana al conflicto y cierra el
   ciclo del modelo semántico (las poblaciones desarrollan afrontamientos que
   responden a hechos), completando la narrativa de la Comisión de la Verdad.

## Reflexión: ¿qué ventajas aporta representar conocimiento en ontologías frente a un análisis plano de texto en PLN?

Representar conocimiento en ontologías aporta ventajas estructurales y
semánticas que un análisis plano de texto no puede ofrecer:

1. **Semántica explícita y compartida**: en un análisis plano, el significado de
   los términos se infiere por frecuencias, contextos o embeddings; en una
   ontología, las clases y relaciones (esParteDe, relacionadoCon, perpetradoPor)
   formalizan el significado de forma inequívoca y legible por máquinas y por
   humanos. Dos sistemas pueden interoperar sin ambigüedad.

2. **Razonamiento y consulta formal**: con SPARQL y razonadores (HermiT) es
   posible derivar conocimiento que no está escrito: subclases transitivas,
   consistencia de la jerarquía, individuos que satisfacen restricciones. Con el
   modelo fuerte, la inferencia va más allá: las 6 clases definidas permiten que
   el razonador clasifique automáticamente los conceptos que cumplen una fórmula
   (p. ej. toda población con restricción `esAfectadaPor some Hechos` cae en
   `PoblacionAfectadaPorHechos`), y los 7 axiomas raíz con sus inversas permiten
   consultar ciclos causales completos (quién perpetra qué, a quién afecta, qué
   impacto genera y con qué afrontamiento responde). El análisis plano solo
   permite búsquedas léxicas o estadísticas; no puede inferir ni recorrer
   trayectorias causales.

3. **Reutilización y evolución**: una ontología es un artefacto versionable,
   reutilizable y extensible; un análisis plano es un resultado puntual atado a
   un corpus. El tesauro CEV, al convertirse en OWL, queda listo para
   enriquecerse con otros vocabularios (SKOS, otras ontologías) y servir de capa
   semántica para recuperación de información, catalogación y búsqueda. La
   extensión aprobada (grupo de afrontamientos + modelo semántico) se incorporó
   sin rehacer la construcción, demostrando esa evolutividad.

4. **Complementariedad, no sustitución**: las técnicas planas (WordNet, NER,
   embeddings) aportan cobertura léxica y escalabilidad, mientras que la
   ontología aporta precisión y estructura. En este taller se evidencia la
   sinergia: WordNet enriquece los conceptos de la ontología con relaciones
   léxicas, y la ontología delimita el dominio para que WordNet no se disperse
   en la polisemia. Juntos forman un sistema de organización del conocimiento
   como el que la propia Comisión de la Verdad construyó con su tesauro.

## Recursos

- Protégé: https://protege.stanford.edu/
- Tutorial SPARQL (W3C): https://www.w3.org/TR/sparql11-query/
- WordNet (Princeton): https://wordnet.princeton.edu
- Open Multilingual WordNet (omw-1.4): https://omwn.org/
- Tesauro de la Comisión de la Verdad (documento de sistematización, 2022)
"""

# %% [markdown]
"""## Entrega a CampusVirtual (criterio 8 de la rúbrica)

**Archivos a entregar:**

1. `notebooks/Taller_Ontologias_WordNet.ipynb` — notebook único
   (Colab-compatible) con las secciones 00 a 06 ejecutadas: EDA, selección de
   los 6 grupos, construcción de la ontología, 9 relaciones WordNet con
   verificación, 22 consultas SPARQL documentadas y conclusiones.
2. `ontologia/tesauro_cev.owl` — ontología construida (623 clases, 534
   individuos, 17 object properties, 6 clases definidas), validada con HermiT
   y Protégé.
3. `informe/informe_taller2.md` — informe escrito (opcional, según las
   indicaciones del curso).

**Cómo entregar:**

- Subir la carpeta `TALLER-2` completa a Google Drive y abrir el notebook desde
  `notebooks/` en Colab: la sección sec00 instala las dependencias y descarga
  los corpus NLTK (wordnet + omw-1.4) automáticamente.
- Ejecutar todas las celdas en orden (sec00 → sec06) y verificar que las
  figuras, el CSV de relaciones y las 22 consultas SPARQL quedan renderizados.
- Adjuntar en CampusVirtual el notebook ejecutado y la ontología
  `tesauro_cev.owl` (o la carpeta `TALLER-2` comprimida).

**Recordatorio:**

- Completar los nombres de los integrantes en la portada del notebook (primera
  celda, línea "Integrantes": [Nombre 1] — [Nombre 2] — [Nombre 3]).
- Revisar la rúbrica (8 criterios × 25 puntos) antes de entregar: selección y
  justificación (sec02), implementación WordNet con las 9 relaciones y su
  verificación (sec04), ontología y SPARQL (sec03-sec05), calidad del código,
  gráficos y análisis, informe, conclusiones y formato de entrega.
"""
