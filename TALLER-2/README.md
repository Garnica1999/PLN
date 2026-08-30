# TALLER-2 — Construcción de Ontología y Análisis Semántico con WordNet

Taller del curso **Procesamiento de Lenguaje Natural** (Pontificia Universidad
Javeriana) sobre representación semántica del conocimiento: ontología OWL del
Tesauro de la Comisión de la Verdad + relaciones semánticas con WordNet.

## Entregables
- `notebooks/Taller_Ontologias_WordNet.ipynb` — notebook único (Colab-compatible):
  EDA → selección de 5 grupos → construcción de la ontología → 9 relaciones
  WordNet → 15 consultas SPARQL documentadas → conclusiones y reflexión.
- `ontologia/tesauro_cev.owl` — ontología OWL (593 clases, 516 individuos,
  5 object properties), consistente (HermiT) y validada en Protégé 5.6.9.
- `informe/informe_taller2.md` — informe escrito completo.

## Estructura
```
TALLER-2/
├── data/            tesauro_orginal_cev.xlsx/.pdf + JSON/CSVs generados
├── docs/            PDFs de instrucciones y rúbrica del taller
├── notebooks/       Taller_Ontologias_WordNet.ipynb (ENTREGABLE)
├── notebook_cells/  celdas fuente por sección (sec00-sec06), ensamblables
├── ontologia/       tesauro_cev.owl (ENTREGABLE) + relaciones_afines.txt
├── imagenes/        12 figuras del análisis
├── informe/         informe_taller2.md (ENTREGABLE)
├── scripts/         ensamblar_notebook.py (utilidad interna)
└── plan/            PLAN.md + ADRs (decisiones de arquitectura)
```

## Los 5 grupos seleccionados (586 términos)
Actores armados (82) · Impactos (107) · Armas (94) · Poblaciones (213) ·
Hechos victimizantes (90) — justificación basada en datos en `data/grupos_seleccionados.json`.

## Ejecución
- Local: abrir el notebook desde `TALLER-2/notebooks/` (rutas relativas).
- Colab: subir la carpeta `TALLER-2` y abrir el notebook; la sección sec00
  instala dependencias y descarga los corpus NLTK automáticamente.
