# Verificación de requisitos — TALLER-4 IR CEV (puntos 1-3)

Fecha: 2026-09-04 | Autor: ml-qa-engineer (evidencia) + lead | Método: Prove-It (comandos reproducidos)

## Actividades 1-3 del enunciado (docs/"Taller recuperacion.pdf")

| # | Requisito | Evidencia | Estado |
|---|-----------|-----------|--------|
| 1.1 | Extraer secciones/testimonios relevantes de cada libro CEV | 10/10 tomos, 917 segmentos (TOC nivel 1-2, fusión <300 palabras, techo 4.000, ADR-001); mapeo TOC→encabezados 91-100%, cobertura 100%, 0 solapes, 0 vacíos (`resultados/extraccion_libros_report.json`) | CUMPLE |
| 1.2 | Definir la morfología del documento a recuperar y explicarla | `plan/arquitectura.md` (a) + ADR-001 + informe F8 §3 | CUMPLE |
| 1.3 | Preprocesar libros y entrevistas (tokenización, stopwords, lematización) | spaCy es_core_news_md: libros 917 docs (30,5 s) y entrevistas 2.486 docs (~510 s, 4,88 docs/s); parada en consumo (spaCy STOP_WORDS + 51 muletillas) + 62 correcciones de lemas (`data/corpus/preprocesado/`, `config/`) | CUMPLE |
| 1.4 | Entregar corpus limpio y estructurado para análisis comparativo | `data/corpus/segmentos_libros.jsonl` (917), `entrevistas_normalizadas.jsonl` (2.486), `preprocesado/{libros,entrevistas}_{lemas,tokens}.txt` + `*_docs.json` | CUMPLE |
| 2.1 | Estadísticas descriptivas (longitud, diversidad léxica, frecuencia) | `resultados/eda/` (18 CSVs): TTR_m1000 libros 0,609 vs entrevistas 0,514; Yule K 31,8 vs 38,7; top-30 uni/bi/tri | CUMPLE |
| 2.2 | Nubes de palabras por corpus | `imagenes/eda_*_nube_{uni,bi,tri}*.png` (6 nubes, 2 corpus × 3 órdenes) | CUMPLE |
| 2.3 | Comparación de patrones léxicos libros vs entrevistas | Jaccard 0,221 (25.851/116.775 tipos), cobertura 95,7%, log-odds top-25 bidireccional (`resultados/eda/`, figuras) | CUMPLE |
| 3.1 | Modelo IR que relacione entrevistas con textos de libros | `experimentos/experimento_1_tfidf_base.py`: TF-IDF fit colección 917 + transform 2.486 queries, coseno; `resultados/ranking_top5.csv` (12.420 filas) + top10 | CUMPLE |
| 3.2 | Vectorización TF-IDF documentada | ADR-005 + `config/tfidf_config.json` (min_df=2, max_df=0.85, ngram (1,2), sublinear_tf, norm=l2; justificado con 43,2% hapax) | CUMPLE |
| 3.3 | Informe detallado con resultados y observaciones | `resultados/ir_tfidf_report.json`, `evaluacion_metricas.json`, `ejemplos_top1_ilustrativos.md` + informe F8 §5 | [PENDIENTE F8: informe consolidado] |
| — | Informe final integrador (punto 6) | — | FUERA DE ALCANCE (aprobado puntos 1-3) |
| — | Rocchio/BM25, comparación de métricas, heatmap (puntos 4-5) | — | FUERA DE ALCANCE (aprobado puntos 1-3) |

## Alertas documentadas (no bloqueantes)
1. `data/corpus/entrevistas_normalizadas.jsonl` (intermedio de F4) contiene PII hablada en transcripciones públicas: 1 cédula (94453964), 2 celulares, 118 runs de 6-10 dígitos (mayoría cantidades), 1.773 tokens UPPER+dígito pegados (p. ej. CONOCIDO9lo, FAMILIAR3me) que NO pasan el filtro hasta la etapa spaCy (diseño en 2 etapas, ADR-002/003). **Los corpus finales usados (`_lemas.txt`, `resultados/*`) están limpios: 0 emails/cédulas/teléfonos/UPPER+dígito.** Se documenta en el informe F8 §7 (ética de datos, Bender & Friedman 2018).
2. Discrepancia de métrica: `residuales_caps_digito=0` en F4 se refiere a tokens sueltos; las marcas pegadas (1.773) se contabilizan aparte (anotar en metricas.json fase4_normalizacion como nota).
3. `_lemas.txt` conserva residuales funcionales (de 214, para 1.111, osea 2.416...) porque la parada se aplica en consumo (documentado ADR-003): consistente entre EDA e IR.

## Verificación técnica (comandos)
- ruff check: 0 errores en scripts/01_extraer_libros.py, 02_preprocesar.py, 03_eda.py, experimentos/experimento_1_tfidf_base.py.
- pytest: 43 passed en 3,27 s (test_extraccion 8, test_preprocesado 24, test_ir_tfidf 11).
- metricas.json: JSON válido, 6 entradas (fase3, fase4_normalizacion, fase4_spacy, fase5_eda, fase6_tfidf, fase6_evaluacion), `estado=completado`.
- ranking_top5.csv: 12.420 filas, 0 nulos. segmentos_libros.jsonl: 917 filas. lemas 917/2.486 líneas. evaluacion_muestra.csv: 30 filas, md5 4ed730a5 (determinista, semilla 42).
- data/raw/ sin modificar (mtime 3-sep; PDFs muestreados byte-idénticos al zip).
