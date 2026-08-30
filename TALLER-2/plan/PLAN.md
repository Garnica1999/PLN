# PLAN TALLER-2 — Ontología y Análisis Semántico con WordNet

Curso: Procesamiento de Lenguaje Natural — Pontificia Universidad Javeriana
Objetivo: representación semántica del conocimiento con WordNet + ontología OWL
del Tesauro de la Comisión de la Verdad (5 grupos seleccionados).

## Entregables (según taller)
1. Notebook único (Colab): funciones WordNet + 15 consultas SPARQL documentadas.
2. Ontología OWL construida (Protégé-compatible) — `ontologia/tesauro_cev.owl`.
3. Reflexión: ventajas de ontologías vs. análisis plano de texto en PLN.

## Rúbrica (200 pts): selección/justificación, implementación Python, identificación
entidades-relaciones WordNet, calidad de código, gráficos y análisis, informe escrito,
conclusiones/reflexión crítica, formato y entrega (25 pts c/u).

## Fases y estado

- [x] **Fase 0 — Setup**: estructura de carpetas creada; tesauro.xlsx/.pdf copiados a
      `data/`; PDFs de instrucciones a `docs/`; instalados `owlready2` y `rdflib`;
      corpora NLTK `wordnet` + `omw-1.4` (WordNet español OK: 6 synsets para
      "desplazamiento").
- [x] **Fase 1a — EDA + selección de 5 grupos** (`kaggle-specialist`): EDA completo
      (12.341 términos, 41 grupos; duplicados: 3; 74% entre 3-5 niveles) + 6 figuras +
      `data/grupos_seleccionados.json` con los 5 grupos: **Actores armados (82,
      14.63%) · Impactos (107, 11.21%) · Armas (94, 7.45%) · Poblaciones (213, 1.41%)
      · Hechos victimizantes (90, 2.22%)** = 586 términos, 5 ejes distintos. Celdas
      sec01_eda.py + sec02_seleccion.py.
- [x] **Fase 1b — Prototipo extractor WordNet español** (`nlp-specialist`): 9
      relaciones implementadas en `notebook_cells/sec04_wordnet.py` (función
      única `extraer_relaciones(termino, depth=1)` + normalización robusta);
      validado sobre muestra de 30 términos (cobertura 83.3%); omw-1.4:
      38.512/117.659 synsets en español; limitaciones: tilde requerida,
      antonimia/derivación vía lemmas ingleses con retraducción; pendiente
      integración con la ontología (Fase 2b).
- [x] **Fase 2 — Ontología + relaciones**:
      - 2a `nlp-specialist`: `ontologia/tesauro_cev.owl` (RDF/XML, 9.284 triples):
        593 clases, 516 individuos (términos hoja), 5 object properties
        (esParteDe/parteDe, relacionadoCon TR simétrica, perpetradoPor/esVictimaDe),
        2 annotation props (grupo, eje). Validada: rdflib OK + HermiT (embebido)
        consistente. Heurística IDF documentada (relaciones_afines.txt). Celdas
        sec03 + arbol_ontologia.png.
      - 2b `nlp-specialist`: WordNet aplicado a 592 conceptos desde la OWL:
        cobertura 49,7% (294/592; por grupo: Hechos victimizantes 78,9%, Impactos
        74,1%, Actores armados 69,5%, Armas 51,5%, Poblaciones 16,7%). 65.717
        relaciones (coordinados 48.525, sinonimia 4.879, derivación 4.840...).
        Solo 6,1% coincidencia exacta omw-1.4 (resto vía fallback de tokens).
        Celdas sec04 (19 celdas) + 5 PNG + CSVs.
- [x] **Fase 3 — SPARQL + QA**:
      - 3a `nlp-specialist`: 15 consultas SPARQL (3 por grupo) documentadas y
        ejecutadas en `notebook_cells/sec05_sparql.py` (rdflib; optimizadas:
        69s -> 19.5s con inner joins transitivos). Resultados 0-101 filas.
      - 3b `ml-reviewer`: **APROBADO CON OBSERVACIONES** — 5 archivos ejecutan
        end-to-end (109s), ruff 0 errores (19 correcciones), sin emojis,
        sin PII, rutas relativas. Observaciones menores no bloqueantes.
- [x] **Fase 4a — Ensamble**: notebook único `notebooks/Taller_Ontologias_WordNet.ipynb`
      (103 celdas: 56 markdown + 47 código) con portada + sec00-sec06; ejecución
      end-to-end con nbclient: 0 errores, 5 figuras PNG embebidas (sec04) + 6
      referenciadas vía markdown (sec01) + arbol_ontologia.png. Fixes de
      integración: rutas `__file__`->`os.getcwd()` para modo notebook; backend
      Agg de sec03 condicionado a modo script (figuras inline OK).
- [x] **Fase 4b — Informe + verificación final**:
      - Protégé 5.6.9 verificado: arranca con `tesauro_cev.owl`, sin errores.
      - ADR-001/002/003 en plan/decisiones/.
      - Informe escrito: `informe/informe_taller2.md` (3.402 palabras, 7
        secciones, 4 tablas, 6 figuras, reflexión ontologías vs texto plano).
      - README.md de TALLER-2 creado.
- [x] **Fase 5 — Plan de calidad (POO/SOLID + visualización + rúbrica)**:
      - **0. Graphiti**: 10 episodios del proyecto registrados (saga `taller-2`,
        group `pln`): contexto, plan original, datos del tesauro, selección,
        ontología, WordNet, SPARQL, nuevo plan, rúbrica, entregables.
      - **A. Núcleo POO** `sec00_clases.py` (embebido en el notebook): Repository
        (TesauroRepositorio), Singleton+Facade (ServicioOntologia.obtener()),
        Abstract Factory+Registry (FabricaExtractores, 9 relaciones), Strategy
        (9 extractores + CadenaNormalizadores), Facade (MotorRelacionesWordNet,
        SparqlServicio, GrafoOntologia). Construcción de la OWL replicada
        exactamente (593/516/587/50/12, rdflib 9.284 triples, HermiT OK).
      - **B. sec00_setup**: todos los imports + todas las constantes del proyecto
        en las primeras celdas (rutas, colores, grupos, umbrales, IRI).
      - **C. Refactor DRY**: sec01 (EdaTesauro, GraficadorEda), sec02
        (SelectorGrupos; JSON byte-idéntico, 5 grupos + 586 términos), sec03
        (VisualizadorJerarquia + ServicioOntologia), **sec03b NUEVA**
        (grafo completo 1.110 nodos/1.171 aristas + vista contraída 6 nodos/10
        aristas, PNG en disco), sec04 (AnalisisRelaciones/AnalizadorRelaciones;
        Motor actualizado: 51.795 relaciones, cobertura 49,7% idéntica;
        verificación de las 9 relaciones con asserts), sec05 (15 consultas como
        datos + SparqlServicio; totales idénticos), sec06 (celda Entrega).
      - **D. Tests**: `tests/` (60 tests pytest, 20s) que importan el MISMO
        módulo sec00_clases (fuente única); no van en el entregable (ADR-005).
      - **E. Brechas rúbrica**: celda "Relevancia para el sector" en sec02;
        informe actualizado (arquitectura POO, visualización, verificación de
        requisitos; cifras reales del CSV: 51.795 relaciones, coordinados
        23.463, hiponimia 21.806, sinonimia 2.473, hiperonimia 1.978,
        derivación 1.460, meronimia 314, antonimia 154, holonimia 82,
        troponimia 65; grafo 5.227 aristas; Spearman -0.180);
        `plan/verificacion_requisitos.md` (8/8) + `plan/verificacion_rubrica.md`
        (200/200); ADR-004/005.
      - **F. QA**: ml-reviewer APROBADO CON OBSERVACIONES (criterio 4: 4.8/5;
        ruff 0, py_compile 16/16, 60 tests, standalone EXIT 0 en las 9
        secciones); corregidos bug del ensamblador (conteo de celdas) y
        eficiencia df_local; notebook reensamblado (119 celdas: 61 md + 58
        código) y ejecutado end-to-end con nbclient: **0 errores, 7 PNG
        embebidos** (grafo completo + contraído + 5 de sec04).

## Check rúbrica (8 criterios × 25 pts = 200) — detalle en plan/verificacion_rubrica.md
1. Selección/justificación -> sec02 (criterios numéricos + celda relevancia sector) + informe. OK
2. Implementación Python (WordNet) -> sec04 (9 relaciones + verificación con asserts). OK
3. Entidades y relaciones (WordNet) -> sec03 + sec04 (592 conceptos, 9 tipos demostrados). OK
4. Calidad del código -> núcleo POO (patrones, SOLID, DRY) + ruff 0 + 60 tests + revisión. OK
5. Gráficos y análisis -> 14 PNG + interpretaciones + grafo completo de la ontología. OK
6. Informe escrito -> informe_taller2.md actualizado (arquitectura + visualización + verificación). OK
7. Conclusiones y reflexión crítica -> sec06 + informe §6. OK
8. Formato y entrega -> notebook único Colab + OWL + celda de entrega. OK

## Decisiones (ADR) — plan/decisiones/
- ADR-001: selección automática de 5 grupos con justificación por datos (aprobado).
- ADR-002: ontología generada programáticamente con owlready2 (OWL estándar) y
  validada abriéndola en Protégé 5.6.9 (disponible con JRE incluido, DISPLAY=:0).
- ADR-003: WordNet en español vía NLTK + omw-1.4; cobertura parcial reportada
  honestamente como métrica del análisis.
- ADR-004: núcleo POO con patrones de diseño dentro del notebook (Repository,
  Singleton+Facade, Abstract Factory+Registry, Strategy, Facade; SOLID/DRY).
- ADR-005: tests unitarios independientes en tests/ (fuera del entregable) que
  importan el mismo módulo sec00_clases (una sola fuente de verdad).

## Datos clave
- Excel: hoja `Tesauro` = 12.341 términos, 41 grupos (núcleos/dominios/directorios),
  jerarquía en el término con separador " - " (1-10 niveles). Hoja `Listas` = 6.880.
- Grupos más grandes: Organizaciones (6.405), Estructuras armadas (1.416),
  Medios de comunicación (1.354), Profesiones y oficios (942), Mecanismos de
  participación ciudadana (223), Poblaciones (213), Hitos (150), Armas (94).
- Entorno: /home/carlos/venv (pandas, openpyxl, matplotlib, seaborn, nbformat, nltk,
  owlready2, rdflib). Protégé 5.6.9 en /mnt (no se copia, 114 MB).

## Fase 6 — Revisión del taller resuelto (Taller_ontologia.rar)
- [x] **Fase 1 — Descompresión**: `Taller_ontologia.rar` extraído en
      /mnt/.../TALLER II/Taller_ontologia/ (notebook 123 celdas, OWL 3.319
      triples, ontologia.png 2236x15992, tesauro.xlsx 2 columnas).
- [x] **Fase 1 — Descompresión**: `Taller_ontologia.rar` extraído en
      /mnt/.../TALLER II/Taller_ontologia/ (notebook 123 celdas, OWL 3.319
      triples, ontologia.png 2236x15992, tesauro.xlsx 2 columnas).
- [x] **Fase 2 — Revisión en paralelo**:
      - 2a ontología+grafo: HermiT CONSISTENTE; 517 clases, 19 props (9 pares
        inversos), 6 clases definidas con restricciones, 11 individuos; 94,4%
        de paths fieles al tesauro (5 errores: OSIGD jerarquía rota, MUSE
        duplicado...); IRI example.org; PNG = árbol OWLViz no integrado;
        carga OK en Protégé.
      - 2b código+ejecución: notebook ajeno ejecutado con sus datos (exit 0);
        15 consultas SPARQL re-ejecutadas y verificadas (30/81/3/23/23/2/22/
        16/4/8/8/6/44/9/3 filas); 9 relaciones con resultados; ruff 19 (17
        B018 esperables de Jupyter); cobertura WordNet ~6,4% (93,6% no
        encontrados) y NO reportada.
- [x] **Fase 3 — Matriz de cumplimiento**: `plan/cumplimiento_directrices_taller_ajeno.md`
      — Enunciado 4/5 (falta la REFLEXIÓN explícita); Rúbrica estimada ~135/200
      (gráficos Insuficiente, conclusiones/reflexión Insuficiente, informe
      Satisfactorio).
- [x] **Fase 4 — Informe comparativo**: `informe/revision_taller_resuelto.md`
      (secciones 1-6: contenido, cumplimiento, revisión técnica de OWL y
      código, tabla comparativa, veredicto + mejora opcional sugerida:
      clases definidas con restricciones en nuestra OWL).

## Fase 7 — Extensión: 6 grupos + modelo semántico fuerte (ADR-006)
- [x] **A1. Núcleo POO**: SLUG_RAIZ/EJE_POR_GRUPO con 6 grupos;
      `_definir_propiedades` ampliado a 17 object properties (6 pares inversos
      del ciclo); `_agregar_relaciones_entre_grupos` (7 axiomas raíz);
      `_definir_clases_definidas` (6 equivalentClass); `conceptos()` excluye
      clases definidas. Verificado: 623 clases, 534 individuos, HermiT
      consistente y CLASIFICA (DesplazamientoForzado -> HechoPerpetradoPorActor).
- [x] **A2. Datos**: `grupos_seleccionados.json` con 6 grupos / 609 términos
      (Afrontamientos y resistencias no violentas - cotidianas, 23 términos).
- [x] **A3. Constantes**: sec00_setup con 6 grupos, 6 colores, GRUPOS_ADICIONALES.
- [x] **B1. sec02/sec03**: selección con grupo adicional + justificación del
      ciclo; OWL reconstruida (623/534/17/7/6, 9.812 triples, HermiT True);
      celda "Usabilidad de la ontología"; VisualizadorJerarquia con 6 raíces.
- [x] **B2. sec03b**: grafo 1.158 nodos / 1.286 aristas; vista contraída 7
      nodos / 12 aristas; validación del modelo semántico (7 axiomas, 6
      definidas).
- [x] **C. sec04**: 610 conceptos, cobertura 50,5% (308), 54.130 relaciones
      (coordinados 24.920, hiponimia 22.290, sinonimia 2.638...); verificación
      9 relaciones con conceptos de Afrontamientos (fix de patrón por grupo).
- [x] **D. sec05/sec06**: 22 consultas (18 grupos + 4 modelo semántico:
      clases definidas, pares inversos, ciclo completo, individuos por grupo);
      sec06 con cifras nuevas y reflexión ampliada (razonamiento/inferencia).
- [x] **E. Tests**: 67 tests (23s), incluidos 7 nuevos (clases definidas,
      axiomas del ciclo, clasificación HermiT, 17 props, grupo Afrontamientos).
      ruff 0; notebook reensamblado (140 celdas) y ejecutado end-to-end:
      0 errores; Protégé carga la nueva OWL sin errores.
- [x] **F. Cierre**: informe actualizado (4.783 palabras, 0 cifras viejas);
      verificacion_requisitos/rubrica re-verificadas (200/200);
      ADR-006; respaldo de la OWL previa en /tmp/opencode/taller2_ref/.

## Fase 8 — Informe LaTeX + autores + ejecución final
- [x] **Informe LaTeX**: `informe_latex/` con plantilla del TALLER-1 (tg_usta.cls,
      autores Rivera/Carrillo/Garnica/Prieto, abstract, 6 figuras, tablas
      booktabs); compilado con TinyTeX (~/.TinyTeX, pdflatex+bibtex):
      **Taller_2_Ontologias.pdf, 9 páginas, 0 errores**, con la REFLEXIÓN como
      sección central (4 argumentos con evidencia: semántica explícita,
      razonamiento/inferencia HermiT, reutilización, complementariedad).
- [x] **Autores en el notebook**: portada actualizada con los 4 integrantes del
      equipo (ensamblar_notebook.py) y notebook reensamblado.
- [x] **Ejecución completa**: notebook ejecutado end-to-end con nbclient:
      0 errores, 67/67 celdas, 6 PNGs embebidos, cifras verificadas.
- [x] Graphiti: episodio de cierre registrado.

## Fase 9 — Carpeta entregable (ZIP autocontenido para CampusVirtual)
- [x] **entregable/ENTREGABLE_TALLER2.zip** (15,4 MB, 38 archivos):
      notebooks/ (ipynb 140 celdas ejecutado, autores), ontologia/ (tesauro_cev.owl),
      informe/ (Taller_2_Informe_Final.pdf con reflexión), data/
      (tesauro_orginal_cev.xlsx + json + csvs), imagenes/ (14 PNG),
      notebook_cells/ (fuentes, necesarias para import), LEEME.txt.
- [x] **Robustez portátil**: `_ruta_base()` en sec00 busca la raíz por
      candidatos (cwd/padre/abuelo) en vez de asumir notebooks/.
- [x] **FIX de portabilidad encontrado por prueba**: sin notebook_cells/ en el
      ZIP, sec01 creaba un módulo `sec00_clases` "fake" (solo TesauroRepositorio)
      que rompía sec03 (AttributeError). Incluir notebook_cells/ en el ZIP
      resolvió el problema.
- [x] **Pruebas antes de empaquetar**: ruff OK, 67 tests OK, notebook end-to-end
      0 errores en TALLER-2 Y desde /tmp (descomprimido fuera del proyecto):
      0 errores, 6 PNGs embebidos.
- [x] **Validación del ZIP**: unzip -t sin errores; 3 entregables del taller
      (notebook + OWL + informe con reflexión) verificados; tesauro xlsx y
      LEEME.txt presentes; copias sueltas en entregable/.

## Fase 10 — ZIP sin notebook_cells (notebook autocontenido)
- [x] **Cambio solicitado**: eliminada la carpeta notebook_cells/ del ZIP (solo
      debe existir el notebook).
- [x] **Autocontención del notebook** (para que siga ejecutándose en cualquier
      máquina SIN notebook_cells/):
      1. Nueva celda en sec00_clases.py: registra en sys.modules el módulo
         `sec00_clases` con todas las clases/constantes/dependencias del
         núcleo (las celdas ya las definen en el namespace del notebook).
      2. sec04/sec05: imports de sec00_setup hechos CONDICIONALES
         (`if "RUTA_ONTOLOGIA" not in globals()`) — en el notebook las
         constantes ya existen; en modo script standalone el archivo existe.
- [x] **Pruebas**: ruff 0; 67 tests OK; notebook end-to-end 0 errores en
      TALLER-2; PRUEBA DE PORTABILIDAD SIN notebook_cells (descomprimido en
      /tmp): 0 errores, 6 PNGs embebidos.
- [x] **ZIP final reconstruido** (10,5 MB, 28 archivos, unzip -t OK): notebooks/
      (ipynb 141 celdas), informe/ (PDF con reflexión), ontologia/ (OWL),
      data/ (tesauro xlsx + resultados), imagenes/ (14 PNG), LEEME.txt
      actualizado (notebook autocontenido). Sin notebook_cells/ (0 referencias).
- [x] Entregables requeridos verificados + copias sueltas en entregable/.
