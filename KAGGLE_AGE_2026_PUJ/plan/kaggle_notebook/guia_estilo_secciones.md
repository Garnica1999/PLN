# Guía de estilo y mapa de secciones — notebook `kaggle_final_pipeline.ipynb`

Documento de trabajo para la redacción de las secciones documentales del notebook final de
TweetAge_2026_PUJ (proyecto KAGGLE_AGE_2026_PUJ). Los archivos se guardan en
`plan/kaggle_notebook/secciones/` y el generador del notebook los embebe como celdas markdown.

## 1. Estructura CRISP-ML del notebook (mapa completo)

- **Portada y resumen ejecutivo** [YO]
- **FASE 1 — Comprensión del problema**
  - 1.1 La competición, la tarea y la métrica [YO]
  - 1.2 Restricciones inviolables [YO]
  - 1.3 Gobernanza experimental (pre-registro, gates, control negativo) [YO]
- **FASE 2 — Comprensión de los datos**
  - 2.1 Carga autónoma y auditoría de integridad [YO]
  - 2.2 EDA y textometría, con figuras [YO]
  - 2.3 La estructura de los seis rangos [YO]
  - 2.4 La fuga por near-duplicados y los folds canónicos [YO]
  - 2.5 Auditoría del historial V1-V37 [DOC] -> PR1
- **FASE 3 — Preparación y representación (V1-V17 re-ejecutada)**
  - 3.1 Limpieza y normalización [YO]
  - 3.2 Representaciones comparadas (V1-V4) [RE] [YO]
  - 3.3 Las fases PH y el meta-modelo (V5-V12) [RE] [YO]
  - 3.4 Features densas, estructuradas y supervisadas [RE] [YO]
  - 3.5 La búsqueda residual V13-V17 [DOC] -> PR1
- **FASE 4 — Modelado**
  - 4.1 V18-V29: frontera, ordinal y selección [DOC] -> PR2
  - 4.2 V30: cuatro cuartos, ensemble y plug-in [RE] [YO]
  - 4.3 Fracasos seleccionados (MLP, CatBoost, FTRL, CORAL, canales) [RE/DOC] -> PR2
  - 4.4 Los componentes neuronales V38-V43 [RE] [YO]
  - 4.5 V44-V52: falsaciones documentadas [DOC] -> PR3
- **FASE 5 — Evaluación**
  - 5.1 El protocolo de validación y su reproducibilidad [YO]
  - 5.2 El campeón V48: PoE y submission verificada [YO]
  - 5.3 Análisis de error, con figuras [YO]
  - 5.4 Análisis matemáticos y estadísticos [DOC] -> PR3
  - 5.5 Calibración CV <-> leaderboard [DOC] -> PR3
- **FASE 6 — Despliegue** [YO]
- **ANEXOS**: A. Inventario completo; B. Referencias del estado del arte; C. Glosario -> PR4

## 2. Reglas de estilo (obligatorias)

1. **Español semiformal, claro y directo**, con tildes correctas. Frases cortas. Sin emojis.
2. **Cada experimento responde**: (a) qué se hizo; (b) por qué (motivación enlazada al EDA o al
   experimento anterior); (c) para qué; (d) qué resultado exacto se obtuvo; (e) decisión tomada.
   En párrafos verdaderamente redactados (las listas solo para parámetros o tablas).
3. **Cifras con fuente obligatoria**: cada número debe existir en los paquetes de porte
   (`plan/kaggle_notebook/paquetes/`), `experiments.csv`, `RUNLOG.md`, `DECISIONS.md` o los
   informes de `reports/`. Si un número no aparece, escribir `[VERIFICAR]` en vez de inventarlo.
4. **Referencias del estado del arte**: solo las verificables en el repo (docstrings, informes);
   formato `Autor et al. (Año). Titulo. Venue. DOI/URL` cuando exista.
5. **Markdown estándar**: `##`/`###`, tablas, listas, y LaTeX simple (`$...$`) para fórmulas.
6. **Longitud**: 120-300 palabras por subsección; las de fase pueden llegar a 400.
7. **Terminología consistente**: macro-F1, rangos, miembros, cuartos, plug-in, corrida,
   protocolo, folds, OOF, meta-modelo.
8. Describir solo lo hecho (sin promesas de futuro); tono honesto (los fracasos se documentan
   como evidencia, no se esconden).

## 3. Formato de entrega

- Un archivo `.md` por sección en `plan/kaggle_notebook/secciones/` con nombre
  `faseX_Y_tema.md` y encabezado `## X.Y Título`.
- Tablas markdown para resultados (columnas: experimento, configuración clave, resultado,
  decisión).
- Marcar `[RE]` (acompaña código re-ejecutado) o `[DOC]` (documental) al inicio del archivo.
- ASCII no es requisito aquí (son documentos); código citado dentro, tal cual (ASCII).
