# VEREDICTO QA FINAL - Entrega E2 "Juris-PLN" (Fase N)

Fecha: 2026-09-09
Verificador: ml-qa-engineer (skill ml-shipping como guia)
Entorno: /home/carlos/venv/bin/python, CPU <= 8 hilos
Script de verificacion: `scripts/checks_qa_final.py` (evidencia ejecutable, no se borra)

## VERIFICACION - Fase N (producto final 10_evaluar_sentencia.py)

Estado: **APROBADO** (GO)

### 1. Prueba E2E muestra estratificada (10 casos)

Seleccion determinista (mediana por celda periodo x tipo, semilla implícita por
orden de `id`): 8 positivos del corpus (2 por periodo 1992-97 / 1998-03 /
2004-10 / 2011-21; mezcla T/C/SU con 1 SU incluido) + 2 negativos (discursos
presidenciales de `open_corpus_co_es.load_corpus('presidentes')`: Gaitan
1942-46 y Petro 2022-26, guardados a .txt temporal y evaluados via --archivo).

| caso | id | tipo/ano | es_sentencia | area top-1 (prob) | arts | bm25 | citas (res) | t(s) |
|---|---|---|---|---|---|---|---|---|
| pos_2337 | 2337 | T/1997 | True | derechos_fundamentales (0.9515) | 4 | 5 | 1 (1) | 8.30 |
| pos_2728 | 2728 | SU/1997 | True | regimen_economico (0.9098) | 9 | 5 | 11 (9) | 10.33 |
| pos_6251 | 6251 | T/2001 | True | derechos_fundamentales (0.6350) | 2 | 5 | 13 (13) | 8.11 |
| pos_6968 | 6968 | C/2001 | True | derechos_fundamentales (0.6462) | 14 | 5 | 31 (30) | 7.76 |
| pos_13015 | 13015 | C/2007 | True | principios_fundamentales (0.9134) | 14 | 5 | 9 (4) | 8.96 |
| pos_13494 | 13494 | T/2006 | True | principios_fundamentales (0.8728) | 4 | 5 | 1 (1) | 7.76 |
| pos_19890 | 19890 | T/2013 | True | derechos_fundamentales (0.7850) | 14 | 5 | 42 (37) | 10.73 |
| pos_20178 | 20178 | C/2013 | True | principios_fundamentales (0.9101) | 25 | 5 | 51 (44) | 11.37 |
| neg_gaitan | - | discurso | False | - | 0 | 0 | 0 (0) | 0.87 |
| neg_petro | - | discurso | False | - | 0 | 0 | 0 (0) | 0.88 |

Checks ejecutados (10/10 casos pasaron, `scripts/checks_qa_final.py` exit 0):

- [x] es_sentencia correcto en los 10 casos (8 True, 2 False; score_validacion 1.0 en positivos).
- [x] area top-1 presente con prob en (0, 1] en los 8 positivos (top3 completo).
- [x] articulos extraidos > 0 en los 8 positivos (2-25; con texto_150chars y bloque).
- [x] BM25 == 5 en los 8 positivos; vacio en negativos.
- [x] Citas resueltas coherentes: 120 citas resueltas cruzadas contra
  `data/red_citas_resumen.parquet` -> 0 errores de ano, pagerank (>0, redondeo
  6 decimales exacto) y area.
- [x] Sin campos null inesperados: ponente presente en los 8 positivos
  (magistrados reales, p.ej. Eduardo Cifuentes Muñoz en SU-559/97); en
  negativos area/ponente/arts/citas quedan None/vacios por diseño.
- [x] Tiempo < 30 s en los 10 casos (max 11.37 s; negativos < 1 s).
- [x] id_detectado == id de entrada en los 8 casos --id.
- [x] Coherencia con demos previas (T-293/97: derechos_fundamentales 0.9429;
  C-002/98: 0.934) sin drift de comportamiento.

### 2. Consistencia de metricas (15 cifras canonicas vs JSON registrados)

| cifra | esperado | registrado (JSON) | estado |
|---|---|---|---|
| corpus total | 23.742 | 23742 (metricas_formato.json) | OK |
| corpus T | 17.099 | 17099 | OK |
| corpus C | 6.246 | 6246 | OK |
| corpus SU | 397 | 397 | OK |
| etiquetado asignado | 99.63 % | 99.63 (23654/23742, etiquetado_cobertura.md) | OK |
| M4 LogReg completo | 0.2852 | 0.28517 (m4_baseline.json) | OK |
| M4 LogReg enmascarado | 0.2870 | 0.28701 (m4_baseline.json) | OK |
| BETO completo | 0.2080 | 0.20801 (m4_transformer.json) | OK |
| BETO enmascarado | 0.2042 | 0.20415 (m4_transformer.json) | OK |
| BETO gap leakage | 0.0039 | 0.00386 (m4_transformer.json) | OK |
| M7 v2 BM25 F1@5 | 0.218 | 0.21797 (vinculo_cp_benchmark_v2.json) | OK |
| M7 v2 BM25 MRR | 0.717 | 0.71720 (vinculo_cp_benchmark_v2.json) | OK |
| M0 accuracy | 1.0 | 1.0 (m0_validacion.json, 30 pos + 22 neg) | OK |
| red citas aristas | 456.321 | 456321 (red_citas.json) | OK |
| red gigante | 67.6 % | 67.6 (red_citas.json) | OK |
| MTLD 1992 -> 2021 | 64.9 -> 56.8 | 64.88 -> 56.77 (diacronia.json) | OK |

16/16 verificadas OK, sin drift entre lo reportado y lo medido. Nota: el
check inicial de "etiquetado 99.63 %" fallo por regex del verificador (tomo
el primer % del .md, 42.46 de la distribucion de areas); verificado
manualmente contra la linea "asignadas (cita o fallback): 99.63
(23654/23742)" -> OK. Corregido en el veredicto, no es drift del producto.

### 3. Checklist de entrega (GO/NO-GO)

- [x] Scripts ruff-limpios: `ruff check scripts/` -> All checks passed
  (4 avisos RUF010 eran del propio script de QA, corregidos con --fix).
- [x] Ejecutables: 10/10 invocaciones CLI exit=0 (3 modos: --id, --archivo).
- [x] Sin emojis en scripts/ (regex U+1F300-1FAFF etc. -> NINGUNO).
- [x] Docstrings completos en 10_evaluar_sentencia.py y
  componentes_producto.py (modulo + todas las clases/funciones publicas).
- [x] Splits intactos: data/splits_m4.parquet mtime sep 8 15:54 (ANTES de
  todos los experimentos M4/BETO del sep 9) -> no regenerado tras el
  entrenamiento; sin solapes train/val/test (0/0/0), sin duplicados,
  16557/3549/3548 = 23654 docs (= docs etiquetados). data/ no trackeado en
  git (por .gitignore), integridad confirmada por coherencia interna.
- [x] Modelos cargables: 4 joblib M4 cargan (TfidfVectorizer +
  LogisticRegression x completo/enmascarado); dirs BETO completos
  (config.json, tokenizer, model.safetensors x2).
- [x] Figuras: 9 PNG en figuras/ (confusion M4, SHAP, red citas, PageRank,
  MTLD, logodds, areas, deriva retrieval v1/v2).
- [x] ADRs 1-10 presentes en plan/decisiones/ (ADR-1 a ADR-10 verificados).
- [x] Reproducibilidad: semilla 42 documentada en configs y scripts
  (06: random_state/TPESampler; 07: torch.manual_seed + np.random.seed;
  11/12: semilla 42 en muestra temporal).
- [x] Rollback posible: cada artefacto de resultados/ tiene script productor
  trazable (00->12 + lib); data/ es la unica entrada no regenerable (fuente
  externa) y todos los modelos/figuras/JSON se regeneran desde scripts + data/.

### 4. Decision GO/NO-GO

**GO**

Bloqueantes: ninguno.

Correcciones recomendadas: ninguna critica. (Menor: el verificador inicial
de cobertura uso un regex fragil; corregido en este documento.)

Riesgos aceptados (residuales):

1. **M4 macro-F1 0.287 es modesto** (8 areas muy desbalanceadas,
   derechos_fundamentales 42 %): el top-1 del producto puede confundir areas
   vecinas (p.ej. pos_6251 con prob 0.635). Mitigacion: el JSON expone top-3
   con probabilidades para decision informada; ADR-10 documenta la eleccion
   LogReg vs BETO.
2. **Resolucion de citas ~84 %** (60.208 canones citados no resueltos al
   corpus): citas a sentencias fuera del corpus quedan resuelta=False con
   ano extraido del canon. Mitigacion: el campo resuelta distingue
   explicitamente; coherente con ADR-5 (weak labels).
3. **M0 con 52 casos de validacion** (accuracy 1.0): muestra pequena para
   la heuristica 2/3; un documento judicial no-Corte (p.ej. Consejo de
   Estado) podria pasar M0. Mitigacion: los 2 negativos E2E (discursos)
   rechazados correctamente; riesgo acotado al dominio declarado.

Plan de rollback: si el producto falla en produccion, revertir a los
componentes modulares (scripts 06/08/11/12 siguen intactos y ejecutables);
el indice BM25 se reconstruye desde data/constitucion_articulos.parquet
(cache borrable), y los modelos M4 se recargan desde modelos/m4_baseline/.
Objetivo de recuperacion: < 15 min (solo carga de artefactos).

### Rutas escritas (evidencia)

- resultados/qa_final/pos_{2337,2728,6251,6968,13015,13494,19890,20178}.json
- resultados/qa_final/neg_{gaitan,petro}.json
- resultados/qa_final/VEREDICTO_QA.md (este documento)
- scripts/checks_qa_final.py (script de verificacion, evidencia ejecutable)
