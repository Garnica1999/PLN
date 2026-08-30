# Informe Taller 2: Ontología OWL a partir del Tesauro de la Comisión de la Verdad, enriquecimiento léxico con WordNet y consultas SPARQL

---

**Curso:** Procesamiento de Lenguaje Natural

**Programa:** Maestría en Procesamiento de Lenguaje Natural

**Institución:** Pontificia Universidad Javeriana

**Taller:** Taller 2 — Construcción de una ontología OWL, extracción de relaciones semánticas con WordNet y consultas SPARQL

**Fecha de entrega:** 30 de agosto de 2026

**Entregables:** notebook único (Google Colab), archivo OWL (RDF/XML), presente informe con reflexión crítica sobre ontologías frente al análisis plano de texto en PLN

---

## 1. Introducción y objetivo

Los tesauros de la Comisión de la Verdad (CEV) sistematizan la terminología del conflicto armado colombiano: organizan miles de términos en grupos temáticos con relaciones jerárquicas y asociativas para facilitar la indexación y la recuperación de información. Sin embargo, un tesauro es un recurso descriptivo: sus relaciones no están formalizadas en una lógica computacional y no puede razonarse sobre ellas de manera automática.

Este taller propuso cerrar esa brecha mediante tres tareas articuladas: (i) seleccionar un subconjunto representativo de grupos del Tesauro CEV 2022 y convertirlo en una ontología OWL con el editor programático owlready2; (ii) implementar una función Python que, apoyada en WordNet (NLTK, versión multilingüe omw-1.4), extraiga nueve tipos de relaciones semánticas para los conceptos de la ontología; y (iii) documentar quince consultas SPARQL (tres por grupo) que explotan la ontología resultante.

El objetivo general es demostrar, sobre datos reales y verificables, el ciclo completo de representación formal de conocimiento: de la estructura semántica implícita de un tesauro a una ontología expresada en un estándar (OWL), enriquecida con relaciones léxicas externas y consultable mediante un lenguaje formal (SPARQL). Un objetivo complementario es reflexionar sobre las ventajas de esta representación frente a los análisis planos de texto típicos del PLN (sección final).

## 2. Marco conceptual

### 2.1 Tesauros: TG, TE, TR, USE y UP

Un tesauro es un vocabulario controlado que organiza los términos de un dominio mediante tres tipos de relaciones normalizadas [1]. La relación jerárquica incluye el término genérico (TG) y los términos específicos (TE), que estructuran el vocabulario de lo general a lo particular (p. ej., "Armas" como TG de "Explosivos"). La relación asociativa (TR) vincula términos relacionados pero no jerárquicamente, por proximidad temática o causal. Las relaciones de equivalencia distinguen el término preferido (USE) de sus variantes no preferidas (UP, "usado por"), que típicamente son sinónimos o variantes ortográficas. En el Tesauro CEV, la jerarquía se codifica en el propio término mediante el separador " - ", lo que hace explícita la cadena TG→TE sin requerir un modelo de datos relacional.

### 2.2 Ontologías

Una ontología es una especificación explícita y formal de una conceptualización compartida [2]. OWL (Web Ontology Language) es el estándar del W3C para expresarla: define clases (conceptos), individuos (instancias), propiedades de objeto (relaciones entre individuos) y propiedades de anotación (metadatos legibles por humanos). A diferencia de un tesauro, una ontología OWL tiene semántica lógica (p. ej., propiedades inversas o simétricas), lo que habilita razonadores automáticos como HermiT y la consulta formal con SPARQL [3]. La herramienta de referencia para la edición y validación de ontologías es Protégé [4].

### 2.3 WordNet y sus relaciones semánticas

WordNet es una base de datos léxica de la Universidad de Princeton que organiza el vocabulario del inglés en conjuntos de sinónimos (synsets) interconectados por relaciones semánticas [5, 6]. La versión multilingüe Open Multilingual WordNet (omw-1.4) alinea estos synsets con otras lenguas, incluido el español [7]. Las nueve relaciones utilizadas en este taller son: **sinonimia** (equivalencia de significado), **antonimia** (oposición), **hiponimia** ("es un tipo de"), **hiperonimia** (inversa de la anterior), **meronimia** (parte-todo), **holonimia** (todo-parte), **troponimia** (relación de manera entre verbos), **derivación** (vínculo morfológico entre palabras de la misma familia) y **coordinados** (términos que comparten un mismo hiperónimo inmediato).

## 3. Datos y metodología

### 3.1 Fuente de datos: Tesauro CEV 2022

La fuente es el Excel del Tesauro CEV 2022, con dos hojas: la hoja "Tesauro", con **12.341 términos** en **41 grupos** (núcleos, dominios y directorios), y la hoja "Listas", con **6.880 términos**. La jerarquía está codificada en el propio término con el separador " - " y alcanza entre **1 y 10 niveles de profundidad**, con el **74 % de los términos entre los niveles 3 y 5**. El control de calidad es alto: solo **3 duplicados** en todo el archivo.

### 3.2 Criterios de selección de grupos

Sobre los 41 grupos se aplicaron cinco criterios de selección: (i) tamaño entre 50 y 450 términos; (ii) profundidad jerárquica representativa; (iii) cobertura de WordNet en español ≥ 1 % (sondeo previo); (iv) relevancia temática para el mandato de la CEV; y (v) diversidad de ejes semánticos (actores, hechos, objetos, impactos y poblaciones). A los cinco grupos resultantes se sumó un sexto —**Afrontamientos y resistencias no violentas - cotidianas**— incorporado de la revisión del taller resuelto por otro equipo para completar el ciclo del conflicto **Actor → Hecho → Población → Impacto → Afrontamiento** (documenta cómo las comunidades responden a la guerra con resistencia no violenta). La Tabla 1 resume los **seis grupos**, que suman **609 términos**.

**Tabla 1.** Grupos del Tesauro CEV seleccionados para la ontología.

| Grupo | Términos | Cobertura WordNet-es |
|-------|----------|----------------------|
| Actores armados y otros participantes en las dinámicas de la guerra | 82 | 14,63 % |
| Impactos | 107 | 11,21 % |
| Armas | 94 | 7,45 % |
| Poblaciones | 213 | 1,41 % |
| Hechos victimizantes | 90 | 2,22 % |
| Afrontamientos y resistencias no violentas - cotidianas | 23 | 4,3 % (exacta) / 73,9 % (fallback) |
| **Total** | **609** | — |

Cuatro grupos fueron descartados por incumplir el criterio de tamaño (Tabla 2). Su exclusión evita que la ontología quede dominada por un único eje temático y mantiene proporciones manejables para la extracción de relaciones y las consultas.

**Tabla 2.** Grupos descartados y motivo.

| Grupo | Términos | Motivo |
|-------|----------|--------|
| Organizaciones | 6.405 | Excede el límite de 450 |
| Estructuras armadas | 1.416 | Excede el límite de 450 |
| Medios | 1.354 | Excede el límite de 450 |
| Profesiones | 942 | Excede el límite de 450 |

![Grupos con mayor número de términos](<../imagenes/barras_top_grupos.png>)

*Figura 1. Distribución de los grupos del Tesauro CEV por número de términos (figura disponible en la carpeta `imagenes/`).*

### 3.3 Construcción de la ontología OWL

La ontología se construyó con **owlready2** y se serializó en **RDF/XML**. A partir de los 609 términos se derivaron **623 clases**: los 610 conceptos del tesauro (incluidos los nodos sintéticos necesarios para completar la jerarquía) más las **6 clases definidas** y la superclase raíz `ConceptoDelTesauro`. Los **534 términos hoja** (sin descendientes) se modelaron además como **individuos**, consultables como clases (navegación jerárquica) y como instancias (consultas de población). Las **17 object properties** se organizan en dos familias:

- **Estructurales:** `esParteDe`/`parteDe` (materializan la jerarquía del tesauro), `relacionadoCon` (**simétrica**, con 50 axiomas TR) y `perpetradoPor`/`esVictimaDe` (12 axiomas hecho-actor).
- **Modelo semántico del ciclo del conflicto** (6 pares inversos): `afectaA`/`esAfectadaPor` (hecho→población), `generaImpacto`/`esGeneradoPor` (hecho→impacto), `experimentaImpacto`/`impactaA` (población↔impacto), `desarrollaAfrontamiento`/`esDesarrolladoPor` (población→afrontamiento), `respondeA`/`esRespondidoMediante` (afrontamiento→hecho) y `empleaArma`/`esEmpleadaEn` (actor↔armas), con **7 axiomas raíz** que cierran el ciclo a nivel de clase.

Además se definieron **6 clases definidas con restricciones** (`owl:equivalentClass`), que el razonador clasifica automáticamente: `HechoPerpetradoPorActor` ≡ Hechos ∩ `perpetradoPor` SOME Actores_armados; `PoblacionAfectadaPorHechos` ≡ Poblaciones ∩ `esAfectadaPor` SOME Hechos; `PoblacionConImpacto`; `PoblacionConAfrontamiento`; `HechoQueGeneraImpacto`; y `AfrontamientoAsociadoAHecho`.

Dos propiedades de anotación, `cev:grupo` y `cev:eje`, preservan la procedencia de cada término. El grafo resultante contiene **9.812 triples**. La validación fue triple: con **rdflib** se verificó la parseabilidad, con el razonador **HermiT** la consistencia lógica —que además **clasifica**: `DesplazamientoForzado` es inferido subclase de `HechoPerpetradoPorActor` y las subclases de Poblaciones de `PoblacionConAfrontamiento`—, y el archivo se abrió sin errores en **Protégé 5.6.9**.

![Árbol jerárquico de la ontología](<../imagenes/arbol_ontologia.png>)

*Figura 2. Árbol de la ontología con sus seis raíces por grupo (figura disponible en la carpeta `imagenes/`).*

### 3.4 Motor WordNet de extracción de relaciones

Se implementó un motor de extracción que, dado un concepto, consulta WordNet en español mediante **NLTK y omw-1.4** y devuelve las nueve relaciones de la sección 2.3 (diseño POO en 3.6). El proceso incluye: normalización del término (minúsculas, sin acentos), búsqueda de coincidencia **exacta** en omw-1.4 y, si falla, un **fallback por segmentos** que descompone el término del tesauro (que puede contener la cadena jerárquica con " - ") y busca cada segmento por separado, probando variantes ortográficas (singular/plural, sin acentos, guiones bajos, primer y último token de los compuestos). Se genera un **reporte de cobertura** con la variante usada por concepto.

Sobre los **610 conceptos analizados**, la cobertura global fue del **50,5 % (308/610)**. La coincidencia exacta fue baja —solo **37 conceptos (6,1 %)**—: 158 se resolvieron por el último token del compuesto, 103 por el primero y el resto por variaciones morfológicas u ortográficas (singular/plural, guiones bajos); casi toda la cobertura se obtuvo vía el fallback por segmentos.

### 3.5 Consultas SPARQL

Se diseñaron y ejecutaron **22 consultas SPARQL** contra la ontología con **rdflib**: **18** (tres por cada uno de los **seis** grupos) **+ 4 del modelo semántico** (clases definidas con su fórmula, pares inversos del ciclo, la cadena Actor→Hecho→Población→Impacto→Afrontamiento e individuos por grupo). Cada consulta quedó documentada en el notebook con su objetivo explícito y su relación con las etiquetas del tesauro (TG/TE/TR), de modo que se pueda verificar qué estructura del tesauro materializa. Los resultados se presentan en la sección 4.4.

### 3.6 Arquitectura de código

El notebook implementa un núcleo orientado a objetos —los módulos `sec00_clases.py` y `sec00_setup.py`— sobre el que se apoyan las secciones 01 a 06, con patrones clásicos:

- **Repository — `TesauroRepositorio`:** acceso a las fuentes (Excel y JSON de grupos): carga de hojas, detección de anomalías, cálculo de niveles y hojas, y términos por grupo.
- **Singleton + Facade — `ServicioOntologia`:** una única instancia por ruta de OWL (método `obtener()`), compartida por las secciones 03 a 05, que expone una fachada de construcción (clases, jerarquías, individuos, propiedades y axiomas) y de consulta.
- **Abstract Factory + Registry — `FabricaExtractores`:** registro declarativo de las **9 relaciones**; `crear_todos()` devuelve una estrategia por tipo y `tipos_registrados()` alimenta los análisis.
- **Strategy — nueve extractores con interfaz común:** la clase abstracta `ExtractorRelacion` define `extraer(synset, profundidad)` y cada relación tiene su implementación (un extractor por tipo: Sinonimia, Antonimia, Derivación, Hiponimia, Hiperonimia, Meronimia, Holonimia, Troponimia, Coordinados). La normalización es también una cadena de estrategias (`CadenaNormalizadores`) con variantes base, singular/plural y acentos.
- **Facade — `MotorRelacionesWordNet`, `SparqlServicio` y `GrafoOntologia`:** ocultan la complejidad de NLTK/omw-1.4, rdflib y networkx, respectivamente: extracción masiva, consultas SPARQL y vistas de la sección 4.5.

El diseño cumple **SOLID**: (S) una clase por responsabilidad (un extractor por relación, un servicio por recurso externo); (O) el registro de extractores permite añadir relaciones sin modificar la fábrica ni el motor; (L/I) las interfaces abstractas (`ExtractorRelacion`, `NormalizadorTermino`) hacen intercambiables las implementaciones; (D) las dependencias se inyectan por constructor (los analizadores reciben el motor, las rutas y los colores en lugar de crearlos). Además es **DRY**: constantes (tipos de relación, rutas, colores, figuras) e imports viven una sola vez en `sec00`, sin lógica duplicada entre secciones.

El código vive íntegramente en el notebook (celdas exportadas como módulos `sec00`–`sec06` y ensambladas en un único `.ipynb`). Existen además tests unitarios independientes en `tests/` —no incluidos en la entrega— que importan el mismo módulo `sec00_clases` para verificar el núcleo.

## 4. Resultados

### 4.1 Estadísticas de la ontología

La Tabla 3 resume las estadísticas finales del artefacto OWL.

**Tabla 3.** Estadísticas de la ontología OWL.

| Elemento | Valor |
|----------|-------|
| Clases | 623 (610 conceptos del tesauro + 6 clases definidas + `ConceptoDelTesauro`) |
| Individuos | 534 (términos hoja) |
| Object properties | 17 (5 estructurales + 12 del modelo semántico en 6 pares inversos) |
| Clases definidas | 6 (`owl:equivalentClass` con restricciones) |
| Annotation properties | 2 (`cev:grupo`, `cev:eje`) |
| Axiomas TR (`relacionadoCon`) | 50 |
| Axiomas `perpetradoPor`/`esVictimaDe` | 12 (heurística) + 1 (raíz del ciclo) |
| Axiomas del ciclo del conflicto | 7 (raíz) |
| Triples | 9.812 |
| Validación | rdflib (parseo), HermiT (consistencia), Protégé 5.6.9 (apertura) |

![Heatmap de términos por grupo y nivel](<../imagenes/heatmap_terminos_grupos_nivel.png>)

*Figura 3. Distribución de términos por grupo y nivel jerárquico (figura disponible en la carpeta `imagenes/`).*

### 4.2 Cobertura de WordNet por grupo

La cobertura varía fuertemente entre grupos (Tabla 4): los grupos de conceptos abstractos y actos (Hechos victimizantes, Impactos, Actores armados) superan el 69 %, mientras que **Poblaciones** —dominado por nombres propios y pueblos étnicos— apenas alcanza el 16,7 %.

**Tabla 4.** Cobertura de WordNet en español por grupo.

| Grupo | Conceptos analizados | Cobertura (%) |
|-------|----------------------|---------------|
| Hechos victimizantes | 90 | 78,9 |
| Impactos | 107 | 74,1 |
| Actores armados | 82 | 69,5 |
| Armas | 94 | 51,5 |
| Poblaciones | 213 | 16,7 |
| Afrontamientos y resistencias no violentas - cotidianas | 23 | 73,9 (17/23, vía fallback) |
| **Global** | **610** | **50,5 (308/610)** |

![Cobertura WordNet por grupo](<../imagenes/cobertura_wordnet_grupos.png>)

*Figura 4. Cobertura de WordNet en español por grupo (figura disponible en la carpeta `imagenes/`).*

### 4.3 Relaciones extraídas por tipo

La extracción produjo **54.130 relaciones** en total (Tabla 5). Tras la refactorización del motor —que deduplica por **nombre de la relación**, no por par nombre-synset—, la distribución es bimodal: **coordinados (24.920; 46,0 %)** e **hiponimia (22.290; 41,2 %)** concentran el **87,2 %**, mientras que los demás tipos son marginales. La hiponimia domina porque el motor recorre recursivamente dos niveles de hipónimos sobre todos los synsets de la consulta: los conceptos compuestos que no existen como lema se resuelven por un núcleo polisémico (p. ej., "vida", "familia") cuyos múltiples synsets disparan decenas de hipónimos cada uno.

**Tabla 5.** Relaciones semánticas extraídas por tipo.

| Tipo de relación | Cantidad | % del total |
|------------------|----------|-------------|
| Coordinados | 24.920 | 46,0 |
| Hiponimia | 22.290 | 41,2 |
| Sinonimia | 2.638 | 4,9 |
| Hiperonimia | 2.082 | 3,8 |
| Derivación | 1.566 | 2,9 |
| Meronimia | 310 | 0,6 |
| Antonimia | 163 | 0,3 |
| Holonimia | 80 | 0,1 |
| Troponimia | 81 | 0,1 |
| **Total** | **54.130** | **100,0** |

![Heatmap de relaciones por grupo](<../imagenes/heatmap_relaciones_grupos.png>)

*Figura 5. Heatmap del cruce grupos × tipos de relación (figura disponible en la carpeta `imagenes/`).*

**Grafo de relaciones compartidas.** A partir de las relaciones se construyó un grafo de conceptos (una arista por par de conceptos que comparten al menos un término relacionado): **5.765 aristas** en el CSV completo, de las cuales **758 son "fuertes"** (comparten al menos dos tipos) y se dibujan en la figura sobre los **308 nodos** con relaciones; la red se descompone en **124 componentes conexas** en esa vista. Los puentes léxicos entre grupos existen (conceptos como "vida", "familia" o "género" aparecen en varios grupos), pero la mayoría de los conceptos queda sin aristas compartidas, evidencia de dominios léxicos autónomos. La correlación entre profundidad jerárquica y número de relaciones es **Spearman ρ = −0,180** (Pearson −0,134): débil y no monótona, pues el nivel 3 concentra los núcleos polisémicos que más relaciones generan, mientras que los niveles 5–6 (conceptos muy específicos del tesauro) tienen las medianas más bajas.

**Ejemplos verificados de relaciones.** El motor encontró, entre otros: para *desplazamiento*, los sinónimos **movimiento** y **traslado**, el hiperónimo **cambio** y los derivados **desplazarse** y **trasladar**; para *mujer*, los antónimos **hombre** y **masculino** y el derivado **femenino**; para *verdad*, los sinónimos **realidad** y **evidencia** y el antónimo **falsedad**. La extracción captura tanto relaciones de equivalencia útiles para expandir consultas como oposiciones pertinentes para el dominio del conflicto.

### 4.4 Resumen de las 22 consultas SPARQL

Las consultas se organizaron en tres patrones por grupo: (a) estructura jerárquica (navegación TG/TE), (b) individuos y población, y (c) relaciones y cruces (TR materializadas, propiedades inversas, intersecciones entre grupos), más una sección final con las consultas del modelo semántico (M.1–M.4: clases definidas con su fórmula, pares inversos del ciclo, la cadena Actor→Hecho→Población→Impacto→Afrontamiento —que devuelve el ciclo completo— e individuos por grupo). La Tabla 6 resume los resultados registrados.

**Tabla 6.** Resumen de las consultas SPARQL documentadas.

| N.º | Grupo | Tipo de consulta | Filas resultantes |
|-----|-------|------------------|-------------------|
| 1 | Actores armados | Jerarquía por grupo (TG/TE) | 30 |
| 2 | Actores armados | Individuos por grupo | 73 |
| 3 | Actores armados | `perpetradoPor` / `esVictimaDe` | 12 |
| 4 | Impactos | Jerarquía por grupo | — |
| 5 | Impactos | Individuos por grupo | — |
| 6 | Impactos | `relacionadoCon` e intersección entre grupos | 4 |
| 7 | Armas | Jerarquía por grupo | — |
| 8 | Armas | `esParteDe` completo | 96 |
| 9 | Armas | Caminos de 2 niveles | 82 |
| 10 | Poblaciones | Jerarquía por grupo | — |
| 11 | Poblaciones | Individuos por grupo | — |
| 12 | Poblaciones | Keyword en labels | 8–39 |
| 13 | Hechos victimizantes | Jerarquía por grupo | — |
| 14 | Hechos victimizantes | Conteo de términos hoja por subcategoría | 77 / 93 / 199 |
| 15 | Hechos victimizantes | Individuos por grupo | — |

Nota: las filas con "—" son consultas documentadas cualitativamente en el notebook (objetivo + relación con TG/TE/TR) sin conteo registrado; las sumas de la consulta 14 fueron verificadas (77 + 93 + 199 = 369 términos hoja).

### 4.5 Visualización de la ontología

Además del árbol jerárquico de la Figura 2, la sección 03 genera dos vistas gráficas de la ontología con **matplotlib y networkx**, guardadas en disco como imágenes estáticas de alta resolución.

**Grafo completo** (`grafo_ontologia_completo.png`, Figura 6): **1.158 nodos y 1.286 aristas**. Las 623 clases se colorean por grupo raíz (seis colores, uno por grupo) y la superclase `ConceptoDelTesauro` en gris; los 534 individuos (términos hoja) aparecen como puntos pequeños en la periferia. Las aristas mezclan la jerarquía (`subclassOf` 623 y `esParteDe` 610), la afinidad terminológica (`relacionadoCon`, 50 axiomas), los vínculos hecho-actor (`perpetradoPor`, 13) y las 534 aristas de instancia. La figura valida visualmente: (1) las seis raíces están presentes y cada grupo conserva su color; (2) las jerarquías profundas de Armas (hasta 6 niveles) aparecen como trayectorias encadenadas de nodos grandes; (3) los hubs léxicos y los grupos con más términos hoja (Armas, Poblaciones) dominan la periferia; y (4) la superclase es el punto central del que cuelgan las seis raíces.

![Grafo completo de la ontología](<../imagenes/grafo_ontologia_completo.png>)

*Figura 6. Grafo completo de la ontología: 1.158 nodos y 1.286 aristas, clases e individuos coloreados por grupo raíz (figura disponible en la carpeta `imagenes/`).*

**Vista contraída por grupos** (`grafo_ontologia_grupos.png`, Figura 7): cada grupo se condensa en un nodo (6 grupos + superclase) y las **12 aristas inter-grupo** se etiquetan con los tipos que las generan (`relacionadoCon`, `perpetradoPor`, `esAfectadaPor`, `generaImpacto`, `experimentaImpacto`, `desarrollaAfrontamiento`, `respondeA`, `empleaArma`). La vista confirma la coherencia del modelo conceptual CEV: `perpetradoPor` conecta **Hechos → Actores**; **Armas** conecta con Actores y Hechos (los actores emplean armas); **Impactos** conecta con Hechos y Poblaciones (los hechos generan impactos que recaen sobre las poblaciones); y **Afrontamientos** conecta con Poblaciones y Hechos (las poblaciones desarrollan afrontamientos que responden a los hechos), cerrando el ciclo del conflicto en el grafo.

![Vista contraída de la ontología por grupos](<../imagenes/grafo_ontologia_grupos.png>)

*Figura 7. Vista contraída de la ontología: 7 nodos (6 grupos + superclase) y 12 aristas inter-grupo etiquetadas con sus tipos (figura disponible en la carpeta `imagenes/`).*

Los conteos numéricos se contrastan con las métricas del servicio (623 clases, 534 individuos, ≥610 aristas `esParteDe`, 50 `relacionadoCon`, 13 `perpetradoPor`, 534 aristas de instancia), de modo que la validación visual y la numérica coinciden; la sección valida además los 7 axiomas del ciclo y las 6 clases definidas.

### 4.6 Verificación de requisitos del taller

La Tabla 7 resume la verificación de cada requisito de la guía del taller contra las secciones del notebook.

**Tabla 7.** Verificación de requisitos del taller.

| Requisito | Evidencia |
|-----------|-----------|
| Selección de 6 grupos con justificación | Sección 02: cinco criterios explícitos + grupo adicional del ciclo del conflicto (Tablas 1 y 2) |
| Ontología OWL compatible con Protégé (entidades, relaciones, jerarquías) | Sección 03: 623 clases, 534 individuos, 17 object properties (con inversas y simetría), 6 clases definidas; validada con rdflib y HermiT (clasifica), abierta en Protégé 5.6.9 |
| Función WordNet con las 9 relaciones sobre conceptos de la ontología | Sección 04: el motor recibe los 610 conceptos y una celda de verificación recorre los nueve tipos reportando totales (Tabla 5) |
| 22 consultas SPARQL documentadas | Sección 05: tres por grupo (18) + 4 del modelo semántico, con objetivo explícito y correspondencia TG/TE/TR (Tabla 6) |
| Reflexión ontologías vs. texto plano | Sección 06: cuatro argumentos (semántica explícita, razonamiento formal, reutilización, complementariedad) |

## 5. Discusión

**Jerarquía densa del tesauro.** La codificación jerárquica con el separador " - " produjo una estructura de 1 a 10 niveles (74 % entre 3 y 5) que se trasladó casi sin fricción a la ontología: cada segmento de la cadena se convirtió en una clase y cada término hoja en un individuo, con las propiedades inversas `esParteDe`/`parteDe` garantizando la navegación en ambas direcciones. Esto confirma que el Tesauro CEV es, estructuralmente, una semilla de ontología de alta calidad: solo 3 duplicados en 12.341 términos.

**Cobertura parcial de WordNet en español.** La cobertura global del 50,5 % debe leerse a la luz de dos limitaciones del recurso: omw-1.4 solo traduce aproximadamente el **33 % de los synsets de Princeton** al español, y la coincidencia exacta con los términos del tesauro fue de apenas el 6,1 % (el resto provino del fallback por segmentos). Aun así, la extracción produjo 54.130 relaciones, lo que muestra el valor de combinar un recurso léxico general con una estrategia de descomposición. El grupo Afrontamientos (73,9 % vía fallback) ilustra cómo los términos cotidianos (alimentación, silencio, cultura) sí tienen correspondencia léxica, a diferencia del vocabulario institucional de Poblaciones.

**Dominancia de coordinados e hiponimia.** Coordinados e hiponimia representan conjuntamente el 87,2 % de las relaciones extraídas (46,0 % y 41,2 %). Ambas son las relaciones "más baratas" de calcular y también las más ruidosas: dos términos comparten hiperónimo o coordinados por el simple hecho de pertenecer al mismo eje temático, y la hiponimia recursiva de dos niveles se dispara cuando el concepto se resuelve por un núcleo polisémico —los 15 conceptos con más relaciones acumulan 19.615 (36,2 % del total), concentradas en núcleos como "vida" o "familia"—. El conteo alto no indica, por tanto, relevancia semántica sino ambigüedad del núcleo de fallback. La sinonimia (4,9 %), la hiperonimia (3,8 %) y la derivación (2,9 %) son cualitativamente más valiosas para expansión de consultas y desambiguación, pese a su menor volumen.

**Red léxica y puentes entre grupos.** La deduplicación del motor por nombre de relación mantiene una red informativa: la figura con aristas fuertes (758) sobre 308 nodos se descompone en 124 componentes conexas, con puentes léxicos reales entre grupos (conceptos compartidos como "vida", "familia" o "género"), mientras que la mayoría de los conceptos queda sin aristas compartidas, delimitando dominios léxicos autónomos. La correlación débil con la profundidad (ρ = −0,180) indica que la especificidad del dominio pesa más que la posición jerárquica a la hora de explicar la cobertura.

**Poblaciones como límite del recurso léxico.** El grupo Poblaciones (16,7 % de cobertura) evidencia la brecha entre vocabulario de dominio especializado y recursos léxicos generales: los nombres de pueblos étnicos, las categorías OSIGD y la terminología LGBTI están ausentes de omw-1.4.

**Limitaciones del taller.** Además de las anteriores: la ontología cubre 609 de 12.341 términos (4,9 % del tesauro); los nodos sintéticos son una convención de modelado; HermiT garantiza consistencia lógica, no corrección del contenido; las clases definidas dependen de los axiomas raíz (si un hecho no declara su actor, el razonador no lo clasifica); y las relaciones de WordNet no están anotadas con el contexto del conflicto (p. ej., "desplazamiento" recibe relaciones léxicas generales que no distinguen el desplazamiento forzado).

**Lecciones.** El taller mostró que (i) los tesauros bien estructurados son materia prima directa para ontologías; (ii) el fallback por segmentos es una técnica simple y eficaz para conectar vocabulario jerárquico con recursos léxicos externos; (iii) el diseño orientado a objetos con patrones (Repository, Factory, Strategy, Facade) mantiene el notebook extensible: añadir una relación no exige tocar el motor, solo registrar un extractor; y (iv) documentar cada consulta SPARQL con su correspondencia TG/TE/TR convierte la verificación en un ejercicio de trazabilidad entre el tesauro y la ontología.

## 6. Conclusiones y reflexión crítica

Se construyó una ontología OWL consistente de 623 clases y 534 individuos a partir de seis grupos del Tesauro CEV (609 términos), con un modelo semántico fuerte —17 object properties (6 pares inversos del ciclo del conflicto), 7 axiomas raíz y 6 clases definidas que el razonador HermiT clasifica automáticamente—, enriquecida con 54.130 relaciones léxicas de WordNet en español (cobertura 50,5 % sobre 610 conceptos) y validada con HermiT y Protégé; además, 22 consultas SPARQL documentadas (18 de grupos + 4 del modelo semántico) demuestran la explotación del artefacto, incluida la consulta del ciclo completo Actor→Hecho→Población→Impacto→Afrontamiento.

Frente a la pregunta central del taller —*¿qué ventajas aporta representar conocimiento en ontologías frente a un análisis plano de texto en PLN?*—, los resultados permiten sostener cuatro argumentos:

1. **Semántica explícita y compartida.** El análisis plano (p. ej., bolsas de palabras, n-gramas o vectores) trata el texto como una secuencia de símbolos cuyo significado queda implícito y depende del corpus. Una ontología declara explícitamente qué significa cada término (clases, individuos, propiedades, axiomas de inversa y simetría) y lo publica en un estándar interoperable (OWL/RDF), de modo que el conocimiento es compartible y auditable por humanos, como lo evidencian las anotaciones `cev:grupo` y `cev:eje`.

2. **Razonamiento y consultas formales.** Sobre la ontología es posible inferir conocimiento no escrito: las propiedades inversas (los 6 pares del ciclo, `esParteDe`/`parteDe`, `perpetradoPor`/`esVictimaDe`) y la simetría de `relacionadoCon` se razonan automáticamente, y las **clases definidas** permiten que el razonador clasifique conceptos que no declaran explícitamente su rol: `DesplazamientoForzado` es inferido subclase de `HechoPerpetradoPorActor` y las subclases de Poblaciones de `PoblacionConAfrontamiento`. SPARQL responde preguntas estructuradas (jerarquías, poblaciones, cruces entre grupos, caminos de profundidad, el ciclo completo del conflicto) que en texto plano exigirían costosos pipelines de extracción de información, en milisegundos y con resultado determinista, algo que ningún método estadístico plano garantiza.

3. **Reutilización y evolución.** La ontología es un activo incremental: los 41 grupos del tesauro pueden incorporarse sin rediseñar el modelo, y nuevos recursos léxicos pueden alimentar la extracción sin tocar el artefacto. El OWL es portable entre herramientas (owlready2, rdflib, Protégé, HermiT), lo que no ocurre con los resultados de un análisis ad hoc.

4. **Complementariedad, no sustitución.** Ambas aproximaciones se potencian: WordNet enriquece la ontología con sinonimia, hiponimia o antonimia que el tesauro no declara, mientras que la ontología delimita el dominio y protege al análisis plano de su principal debilidad, la polisemia: al anclar los términos a clases formalizadas (p. ej., distinguir "desplazamiento" como hecho victimizante de "desplazamiento" como movimiento físico), se reduce la ambigüedad que degrada los métodos estadísticos. El análisis plano sigue siendo insustituible a escala masiva y para conocimiento emergente no modelado; la ontología aporta precisión, trazabilidad y razonamiento donde el dominio está estructurado, como ocurre con el Tesauro CEV.

En síntesis, el taller demuestra que un tesauro de dominio —recurso tradicionalmente documental— puede transformarse en una ontología operable por razonadores y consultable formalmente, y que esta representación no compite con el PLN plano sino que lo complementa: fija el vocabulario del dominio, desambigua los términos y habilita inferencias que el texto, por sí solo, no puede responder.

## 7. Referencias

[1] Comisión de la Verdad (CEV). *Tesauro CEV 2022: Documento de sistematización*. Documento interno de trabajo, 2022.

[2] Studer, R., Benjamins, V. R., & Fensel, D. (1998). Knowledge engineering: Principles and methods. *Data & Knowledge Engineering*, 25(1–2), 161–197. https://doi.org/10.1016/S0169-023X(97)00039-6

[3] W3C. *SPARQL 1.1 Query Language*. W3C Recommendation, 2013. https://www.w3.org/TR/sparql11-query/

[4] Musen, M. A. (2015). The Protégé project: A look back and a look forward. *AI Matters*, 1(4), 4–12. https://doi.org/10.1145/2757001.2757003

[5] Miller, G. A. (1995). WordNet: A lexical database for English. *Communications of the ACM*, 38(11), 39–41. https://doi.org/10.1145/219717.219748

[6] Fellbaum, C. (Ed.). (1998). *WordNet: An Electronic Lexical Database*. MIT Press.

[7] Bond, F., & Foster, R. (2013). Linking and extending an open multilingual WordNet. En *Proceedings of ACL 2013*, 1352–1362.

**Recursos en línea consultados:**
- Protégé: https://protege.stanford.edu/
- SPARQL 1.1 (W3C): https://www.w3.org/TR/sparql11-query/
- WordNet (Princeton): https://wordnet.princeton.edu
- Open Multilingual WordNet: https://omwn.org/
