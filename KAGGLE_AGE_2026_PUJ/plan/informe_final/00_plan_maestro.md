# Plan maestro — Informe final de la competencia AGE (`tweet-age-2026-puj`)

**Objetivo**: artículo científico en LaTeX (clase IEEEtai, IEEE Transactions on AI) como informe
final de la competencia, en `KAGGLE_AGE_2026_PUJ/informe/` (`informe.tex` + `informe.pdf` +
`figuras/`). Español, 12-15 páginas, con énfasis en el campeón V48 y las decisiones cruciales.

## Especificación aprobada por Carlos
- Ubicación: `KAGGLE_AGE_2026_PUJ/informe/` (confirmada).
- Autores: Erika V. Rivera Jiménez, Julián G. Carrillo Meneses, Carlos A. Garnica Salazar,
  Oscar A. Prieto Moreno (los 4 de los talleres de PLN; Alvaro Gómez NO es autor).
- Título (responde a la pregunta de investigación): "Predicción de rangos etarios a partir de tweets en español mediante técnicas de procesamiento del lenguaje natural y aprendizaje automático supervisado".
- IEEE Impact Statement: incluido.
- Énfasis: el ganador (V48, LB 0.63724 #1) + decisiones cruciales (D-003, D-004, D-005, D-009,
  D-087, D-101, D-106..D-110) explicadas y justificadas.
- Figuras y tablas: sí (figuras del repo/notebook + nuevas generadas).
- Compilador: TinyTeX (`~/.TinyTeX/bin/x86_64-linux/`); cadena pdflatex -> bibtex -> pdflatex x2
  (probada con la clase IEEEtai y `ieeetr.bst` + `references.bib`).

## Estado (bitácora)
- [x] Exploración: compilador, clase IEEEtai (copiada del Taller 3), autores verificados en
      TALLER-2/TALLER-4/PROYECTO de PLN, figuras inventariadas (reports/figures: 111; notebook
      ejecutado r7: 12 embebidas).
- [x] `informe/` creada: `IEEEtai.cls`, `references.bib` (52 fichas), `figuras/`.
- [x] 12 figuras extraídas del notebook ejecutado (`nb_01..nb_12`).
- [x] 3 figuras nuevas generadas (`informe/genera_figuras.py`): `fig_evolucion.png`,
      `fig_arquitectura_v48.png`, `fig_v57_analisis.png` (símplex + deciles V57).
- [x] Test de compilación con la clase + bibliografía: OK (pdflatex + bibtex + ieeetr).
- [x] Dossier de redacción: `informe/dossier_informe.md` (fuente única de verdad, cifras
      verificadas, estructura I-XI, tablas I-VII, mapa de figuras, claves de citas).
- [x] graphiti: episodio "Informe final AGE 2026 — inicio de construcción" registrado.
- [x] codebase-memory: consulta de verificación sobre el consenso/PoE (índice previo del repo).
- [x] Redacción del `informe.tex` (scientific-writer en 3 rondas): secciones I-XI + resumen +
      impacto + 13 figuras + 8 tablas + cierre bibliográfico. Compilación completa con TinyTeX:
      **14 páginas**, 0 errores fatales, 0 citas indefinidas, 0 referencias rotas, 51 citas
      resueltas (rosenthal2011 sin citar). `informe.pdf` md5 `4855eb5b...`.
- [x] Verificación visual del PDF (portada, campeón, falsaciones, frontera 2026, referencias):
      correcta.
- [x] QA de consistencia (agente ml-qa-engineer + cierre del líder): TODAS las cifras clave
      verificadas contra las fuentes (campeón, récord 2026, plug-in, hitos, EDA, V57, techo,
      reproducibilidad, envíos, falsaciones). Única discrepancia (línea 208: mezcla de particiones
      canónica/alternativa en la confianza media) CORREGIDA con valores canónicos y recompilada.
      Integridad OK: 51 citas resueltas, rosenthal2011 sin citar, 14 figuras existentes, sin
      placeholders/credenciales. Reporte: `plan/informe_final/qa_informe.md`. Veredicto: **GO**.
- [x] PDF final: 14 páginas, md5 `f8df262d4893c6cb86ba91adf953ccd1` (tras la corrección del QA).
- [x] Entrega a Carlos (ruta del PDF + resumen). Sin commits (regla).

## Estructura del artículo (aprobada)
I. Introducción · II. Trabajos relacionados · III. Datos y EDA · IV. Metodología y decisiones
fundacionales · V. Representación, features y línea base · VI. El campeón V48 (énfasis) ·
VII. Qué funcionó y qué no · VIII. Frontera 2026 + auditoría matemática V57 · IX. Validación
estadística y reproducibilidad · X. Discusión y limitaciones · XI. Conclusiones · Referencias.

## Fuentes clave
`informe/dossier_informe.md`; `CHAMPION.md`; `RUNLOG.md`; `DECISIONS.md`; `INFORME_COMPLETO.md`;
`plan/kaggle_notebook/secciones/` (fase4_5, fase4_6, 05_fase5); `experiments/v57_*.json`;
`reports/state_of_art.csv`; `references.bib`; notebook ejecutado `experimentos/ejecutado_r7.ipynb`.

## Ronda de correcciones del autor (2026-10-19)
- Actas eliminadas: "acta D-XYZ" → numeración "Decisión 1..11" (mapeo en el dossier, sección 0bis);
  columna "Acta" de la tabla de decisiones → "Decisión 1..6 y 7--11".
- Eliminado el bloque de entrega de notebook e idempotencia (subsección IX-B, tabla de tolerancias,
  contribución 5 y menciones en resumen/impacto/conclusiones); sección IX renombrada a "Validación
  estadística".
- Eliminadas las afirmaciones de primer puesto: se reporta 0.63724 como "mejor puntaje del proyecto
  en el leaderboard" (sin rango); figuras regeneradas sin "(#1)".
- Figura 6 (arquitectura) rediseñada con márgenes amplios (buses de entrada/salida, una columna de
  expertos con separación clara).
- Pasada de estilo: eliminadas construcciones retóricas ("no fue X: es Y", "blinda", "elegante",
  "demostró", "la lección de fondo") en ~20 frases; registro técnico uniforme.
- Recompilado: 13 páginas, 0 errores, 0 citas/referencias indefinidas.

## Ronda de figuras legibles (2026-10-19)
- Revision visual de las 13 figuras renderizadas desde el PDF a tamano real: 11 requerian
  regeneracion (texto ilegible sin zoom) y 2 estaban bien.
- Regeneradas a tamano de imprenta con fuentes nativas (`genera_figuras_articulo.py`): fig 1 (3
  paneles), fig 2 (ancho completo, 6 paneles), fig 3, 4, 5, 7 (ancho completo; reconstruye el
  campeon con macro-F1 0.6186451 exacta), 8, 10, 11, 13; fig 9 con etiquetas rotadas.
- PDF final: 15 paginas (limite), 0 errores, 4 overfull menores (<7pt). Todas las figuras
  verificadas visualmente como legibles sin zoom.

## Ronda de contenido tipo informe (2026-10-19)
- Agregado el marco metodologico CRISP-DM como metodologia principal (subseccion IV-A; seis fases
  mapeadas al proyecto + iteraciones; cita wirth2000 agregada al bib).
- Nueva subseccion IV-E "Motivacion de las decisiones estructurales" (por que se tomaron las
  decisiones clave: validacion agrupada, limpieza, envios, segunda particion, techo, plug-in, PoE).
- Nueva Tabla IV "Variables del conjunto de modelado" (composicion de la matriz de 279 columnas) +
  parrafo con las variables de origen y los 14 miembros con su familia de modelo.
- Ampliado VII-B con el detalle medido de las falsaciones V13-V17 y V44-V52 (numeros exactos);
  "Leccion" -> "Conclusion" en tabla/caption de falsaciones.
- PDF: 16 paginas, 0 errores, 0 referencias indefinidas (md5 e1bf90f9...).

## Ronda editorial (2026-10-19): auditoria externa aplicada
- Checklist de 7 puntos aplicado: cat'logo->cat'alogo; sit'uala->"sit'ua esa brecha";
  rind'io->rindi'o; "de m'argen"->"de margen"; matizado el IC95 de V53 (bordeando el cero, 4/5)
  frente a V54/V56 estrictamente positivos; recuento de envios actualizado (12 iniciales ->
  26 puntuados al cierre); seleccion dual integrada como portafolio estratificado de hasta 10
  envios (VIII-F y conclusiones) con las soluciones intermedias de la frontera.
- Extra: aclaracion de roles V48 (ancla publica) vs V56 (candidata regularizada) al inicio de la
  Seccion VI.
- Se mantuvo la subseccion IV-E (pedida explicitamente por Carlos como motivacion de decisiones);
  unificacion UTF-8/escapes pospuesta (el documento compila limpio con la mezcla).
- PDF: 16 paginas, 0 errores, 0 referencias indefinidas (md5 ec88d246...).

## Ronda editorial Fase 2 (2026-10-19)
- Nueva subseccion VII-E "Inventario del espacio experimental" + Tabla VII de taxonomia: las 7
  familias del catalogo (V1-V12 113, V13-V17 13, V18-V29 25, V30-V37 42, V38-V52 15, V53-V57 5,
  auditoria 7 = 220 filas; 171 ejecutadas + 26 documentales + 12 pre-registros + 10 enmiendas +
  1 bloqueada). Conteos verificados contra el Anexo A.2 del notebook y experiments.csv (la tabla
  propuesta por el revisor no cuadraba con el catalogo; se uso la taxonomia oficial).
- Párrafo de VII-B recortado: eliminada la repeticion de las cifras de V44-V52 (ya tabuladas en
  Tabla VI); se conserva el detalle de V13-V17 no tabulado.
- IV-E alineado con el recuento de 26 envios; (Tabla~I) -> (Tabla~\ref{tab:datos}) en la Seccion V.
- PDF: 16 paginas, 0 errores, 0 referencias indefinidas (md5 426856a7...).

## Ronda editorial Fase 3 (2026-10-19): validacion final
- Corregida la referencia rota de L471 ("Tabla~ef{tab:falsaciones}" -> "Tabla~\ref{...}"):
  causa raiz = el reemplazo con re.sub interpreto el "\r" de "\ref" como retorno de carro;
  escaneado el .tex completo en busca de otros artefactos del mismo tipo (ninguno).
- Ajustada la tabla de taxonomia (overfull 27.4pt -> 1.05pt) y, de paso, la tabla de variables
  (15.8pt -> 0) con tabcolsep reducido.
- PDF: 16 paginas, 0 errores, 0 referencias indefinidas; overfulls restantes solo de parrafo
  (<=6.5pt, tipicos a dos columnas). md5 9be5908a...

## Ejecucion del notebook consolidado (2026-10-19)
- Verificado: el consolidado ya estaba sincronizado hasta V57 (secciones 49-53 + bibliografia 54)
  y es artifact-driven (sin entrenamiento: 0 llamadas a .fit/.backward; arquitecturas torch solo
  definidas).
- Ejecutado completo en el servidor (venv_exacto + nbconvert instalado): **EXIT=0, 287 s (4 min 47 s)**
  para 213 celdas de codigo, 39 figuras, 0 errores. Muy por debajo del objetivo de 2 h, por lo que
  no fue necesaria la paralelizacion de los bootstraps (1-2 min del total, con semillas fijas).
- Verificacion: valores clave identicos a la ejecucion previa (0.6186451, 0.6189929, 0.6188887,
  0.6188259, 0.63724, p=0.0010/0.0050/0.0280, 0.6055514, 0.6066473) y mismas 39 figuras.
- Entregable actualizado: notebooks/99_entrega_consolidado.ipynb = ejecucion fresca validada
  (respaldo de la version anterior en /tmp/opencode/99_entrega_consolidado_previo.ipynb).

## Carpeta de entrega final (2026-10-19)
- Creada `KAGGLE_AGE_2026_PUJ/entrega/` + `KAGGLE_AGE_2026_PUJ_entrega.zip` (285 MB, 659 entradas):
  informe.pdf (solo PDF), notebooks/ (Kaggle + consolidado ejecutado), arbol espejo de artefactos
  (data, features, artifacts, reports, submissions, experiments + experiments.csv), scripts/ + src/
  + config/ (codigo de entrenamiento, 342 scripts), validacion/ (repro + payloads), README.txt.
  Sin archivos markdown (solo README.txt), por instruccion del autor.
- El cuaderno consolidado se parcheo en 5 celdas de arranque para tolerar la ausencia de
  DECISIONS.md (fallback) y se ejecuto DENTRO de la entrega descomprimida en el servidor:
  **EXIT=0, 285 s (4 min 45 s)**, 213 celdas, 39 figuras, 0 errores -> entrega autocontenida validada.
- Incidencias resueltas: dependencias ocultas al hook (Path.read_text: 35 JSON de experiments/,
  DECISIONS.md, y 71 JSON/CSV adicionales referenciados por listas); scripts/ debe estar en la
  raiz del arbol (import v19_b1_boosting).

## Notebook de Kaggle: EDA completo + construccion del dataset (2026-10-19)
- Corregida la tabla rota de la seccion 3.4 (fila E-015s con pipes sin escapar: `softmax(-|d|/5)`
  -> pipes escapados).
- Nuevo EDA completo en la Fase 2: auditoria ampliada (vacios 104/62, ruido de etiqueta 203 textos/
  643 filas 0.38%, '¿' 88 en el corpus, id-proxy 1.248 pp), subseccion 2.2b de textometria con
  Kruskal-Wallis (10 variables + epsilon^2 + boxplot), suelos de referencia (azar 1/6, clase
  mayoritaria 0.04854) y suelo de memorizacion pura (0.0605/3.99% ingenuo vs 0.0000/0.00% agrupado).
- Nueva seccion 2.5 "Construccion del conjunto de datos de modelado": pipeline en 7 pasos
  (corpus -> limpieza -> folds -> TF-IDF -> 78 miembros -> 35 densas -> 87 estructuradas ->
  73 supervisadas + 6 ADV-07 = 279 columnas) + tabla de variables.
- Valores de las celdas nuevas verificados contra el registro canonico (coinciden).
- Notebook regenerado: 182 celdas, 0 fallos de verificacion. Corrida completa lanzada en el
  servidor (notebook_exec8, ~4 h 40); al terminar: checklist + actualizar entrega/ y el ZIP.

## Notebook de Kaggle ejecutado por empalme (2026-10-19)
- Respuesta a Carlos: el consolidado ya tiene todas las ejecuciones (corre en 4:47 y carga artefactos);
  el de Kaggle tarda 4h35 porque RECALCULA la cadena completa.
- Via rapida adoptada (en vez de esperar 4h35): empalme de salidas. 51 de las 55 celdas de codigo son
  byte-identicas a la ejecucion de referencia r7 (par idempotente verificado) -> se copiaron sus
  salidas; las 4 celdas nuevas/modificadas (auditoria ampliada, Fig 2.3 con suelos, textometria
  Kruskal, memorizacion) se ejecutaron en el servidor y se inyectaron.
- Verificacion: 182 celdas, 55 de codigo, 13 figuras, 0 errores, checklist 13/13; valores nuevos
  coincidentes con el registro (643 filas 0.38%, 1.25 pp, 88; memorizacion 0.0605/3.98% vs 0.0000).
- Corrida larga notebook_exec8 detenida (kernel liberado). Entrega actualizada y ZIP regenerado
  (285 MB, 659 archivos). Notebook md5 b08e7bcd...

## Limpieza del repo: solo los dos notebooks (2026-10-19)
- Movidos fuera del repo (a /tmp/opencode/notebooks_retirados/, recuperables) los notebooks
  auxiliares: notebooks/01..09 + _smoke_arranque + experimentos/ejecutado_r1..r7.
- Quedan en el repo unicamente notebooks/99_entrega_consolidado.ipynb y
  notebooks/kaggle_final_pipeline.ipynb (mas sus copias identicas en entrega/notebooks).
