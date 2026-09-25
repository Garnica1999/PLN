# PROJECT_STATE - KAGGLE_TOXIC_2026_PUJ

Ultima actualizacion: 2026-09-22 (Fases 0, 1 y 2 completadas).

## Fase actual

**FASE v2 (Advanced Research) CONCLUIDA el 2026-09-25 sin mejora transferible.
BACKLOG v2 AL 100 % (24/24 lineas evaluadas y cerradas, D-021).**
Modelo final: **V03** (LB **0.78306**), congelado en
`experiments/baseline_v03_frozen/` y reconstruible byte a byte.

Resumen v2 (detalle en `reports/advanced_research_v2.md`): se evaluaron 24 lineas
de investigacion con un protocolo corregido (grupos near-duplicado UNION
pseudo-autor, holdout congelado de 248 tweets, bootstrap por tweet). El meta de
dependencia de etiquetas mejoro en desarrollo (+0.0081) y en el holdout (+0.0180
pareado) pero **no transfirio**: LB v08 = 0.78163 vs 0.78306 de v03 (dentro del
error estandar ~0.013). El paquete P8 cerro las cinco lineas restantes
(calibracion, encoder de entidades por sparsity 0.5 %, umbrales por rebanada,
especialistas y espacios por etiqueta) sin mejora. Lectura: con 9 sondas,
0.78306 esta sesgado al alza por seleccion (winner's curse); el nivel esperado de
la familia es ~0.776-0.780.

Entregables (actualizados y verificados el 2026-09-25):
- `notebooks/10_final_pipeline.ipynb`: 42 celdas, 2.4 s, 0 errores, **26 checks
  [OK] / 0 fallas**, idempotente; 16 secciones con fase 1 + fase v2.
- `FINAL_REPORT.md`: 26 secciones numeradas + resumen + referencias; seccion 26
  dedicada a la fase v2 (incluye el cierre del backlog).
- `reports/advanced_research_v2.md`: informe de cierre de la fase v2.
- `docs/advanced_research_v2_spec.md` + `reports/spec_compliance_v2.md`: spec
  maestra archivada y matriz de cumplimiento FASE 0-99 con evidencia, mas
  `reports/rule_audit_v2.md` (0 violaciones) y
  `reports/target_encoding_audit_v2.md` (sin leakage de aprendizaje).

## Entregables finales

| Entregable | Ruta | Estado |
|---|---|---|
| Notebook entregable (30 % de la nota) | `notebooks/10_final_pipeline.ipynb` | ejecutado, 0 errores |
| Informe final | `FINAL_REPORT.md` (27 secciones, 42 citas) | completo |
| Auditoria final (QA) | `reports/auditoria_final.md` | GO con condiciones cumplidas |
| Auditoria de rendimiento | informe de `ml-performance-auditor` | P1-P3 documentadas |
| Modelo final | `models/mod03b_lr_union.joblib` | reconstruye v03 exacto |
| Submission de respaldo | `submissions/toxic_submission_v03_lr_union_thr.csv` | LB 0.78306 |
| Memoria | `PROJECT_STATE.md`, `RUNLOG.md`, `DECISIONS.md` (D-001..D-015), `experiments.csv` (51 filas) | al dia |
| Evidencia | 13 scripts, 26 figuras, 53 tablas CSV, 9 submissions | en disco |

## Fases 6-8 (features densas) - resultado negativo documentado

B5-DENSE 0.6607 (46 features); hibridos sobre la union: HIB-H12 0.7836
(-0.0091), HIB-H11 0.7713, HIB-H14 0.7630, HIB-UNION 0.7610 (-0.0316).
Control con las densas a cero reproduce 0.792659 exacto -> la perdida es de las
features, no del hstack. H11/H12/H14 no se validan como palanca; quedan como
evidencia negativa para el informe. Rebanadas: solo mejoran referencia colectiva
y Q4 con HIB-H12. Artefactos: `reports/features_densas_resultados.csv/.json`,
`reports/eda/features_densas_slices.csv`, filas B5-DENSE/HIB-* en
`experiments.csv`.

## Sondas de robustez pendientes de envio (D-013, max 2)

- `submissions/toxic_submission_v06_robustez_c1.csv` (C=1.0, min_df=2):
  OOF 0.7929, gap +0.1635 -> candidato debil.
- `submissions/toxic_submission_v07_robustez_mindf5.csv` (C=1.0, min_df=5,
  10.515 features): OOF 0.7993, gap +0.1272 (la mitad que MOD-03b) -> candidato
  fuerte si la hipotesis de capacidad es correcta.
Regla: se adopta la que supere 0.78306 (v03); si ninguna, se mantiene MOD-03b.

`submissions/toxic_submission_v04_ens04_mediana.csv`: par M2+M3 con umbrales
medianos (0.44/0.47/0.44), F1 anidado 0.7985. Predice mas positivos que la
prevalencia de train (HS 46.5 % vs 43.7 %), asi que si el LB decepciona el
sospechoso es el umbral, no la representacion.

**RESULTADO (2026-09-23)**: LB 0.77510 -> el ensemble NO se traslada (peor que
v02 y v03). El sospechoso era el umbral y se confirmo. Descartada.

## Ensembles (Fase 13b) - resultado

4 bases OOF con el mismo split: M1 word LR 0.7587, M2 union LR 0.7927,
M3 union LinearSVC 0.7849, M4 char LR 0.7567. Diversidad real (desacuerdo
4-11 %, correlacion de errores 0.63-0.84) pero ningun ensemble supera al single
de forma convincente: ENS-02 0.7949 anidado (+0.0023), ENS-04 con umbrales
medianos 0.7985 (+0.0058, maximo de 12 estimaciones). Decision: NO se gasta
envio; se mantienen v02/v03.

## Analisis de error (Fase 12) - hallazgos accionables

Rebanadas mas debiles (F1, n>=40): hashtags 0.586 (-0.213 vs global),
preguntas 0.633, referencia colectiva 0.639, textos largos Q4 0.645,
sin mencion 0.697.
Patrones dominantes: (1) insulto interpersonal en registro de broma/afiliacion
(40.5 % de los textos con insulto + mencion + sin colectivo son (0,0,0));
(2) odio implicito contra grupos sin insulto del lexico (45.5 % de los FN de AG
tiene referencia colectiva vs 13.9 % de los TP); (3) regimen noticia/atribucion
a terceros con URL.
Hipotesis nuevas: H11 registro bromista/afectivo, H12 bloque de objetivo
individual vs colectivo para TR, H13 features de cita/atribucion, H14 lexico de
deshumanizacion inducido, H15 calibracion por rebanada (se espera sin aporte).
Detalle en `reports/error_analysis.md`.

## Submissions: estado

- v00 (sonda E00): LB 0.51494. v01 (B1): 0.75677. v02 (MOD-03b, 0.5): 0.77993.
  **v03 (MOD-03b + umbrales): 0.78306 = MEJOR ACTUAL.**
- v04 (ensemble): 0.77510 descartada. v05 (OPT-01 de Optuna): 0.77602 descartada.
- v06/v07 (sondas de robustez, D-013): pendientes de envio por Carlos.

## Hallazgos del EDA (Fase 2, resumen)

- 0 emojis Unicode en el corpus (normalizacion ASCII); 6 bytes no ASCII en test.
- Senales mas fuertes (V de Cramer, train): `tiene_insulto` (HS/AG 0.355,
  TR 0.339), `tiene_mencion_inicial` (TR 0.325), `tiene_insulto_dirigido`
  (TR 0.266), `tiene_referencia_colectiva` (TR 0.215, direccion negativa),
  `tiene_hashtag` (asociado a HS=0).
- Dentro de HS=1 (n=756): TR=1 tiene mencion inicial 64.9 % vs 9.7 % (TR=0) y
  segunda persona/insulto dirigido altos; TR=0 concentra referencias
  colectivas explicitas (57.0 % vs 10.0 %) y URLs.
- Desbalanceo maximo 2.34:1: no requiere remuestreo agresivo.
- Deduplicar por near-duplicado elimina solo 4 filas de train y no cambia el
  orden ni la magnitud de los efectos.
- Hipotesis con apoyo inicial: H1 (indirecta), H2 (parcial), H3 (con reservas),
  H6, H8 y H9. Sin evaluar: H4, H5, H7, H10.

## Revision de codigo (Fase C)

- `ml-reviewer`: APROBADO CON CAMBIOS MENORES (0 criticos, 1 mayor, 5 menores).
- HALLAZGO MAYOR a implementar en modelado: los lexicos inducidos con
  etiquetas de train (`n_insultos`, `tiene_insulto`, `tiene_colectivo_lexico`)
  estan sesgados in-sample; en los modelos deben inducirse DENTRO de cada fold
  de `StratifiedGroupKFold` y reportar el efecto out-of-fold.
- Cambios menores aplicados y re-verificados (ruff + idempotencia).

## Decisiones pendientes de Carlos

1. **D-005 (semantica de TR)**: revisar
   `reports/eda/tr_ejemplos_para_revision.csv` (40 TR=1 + 40 TR=0 anotados) y
   `reports/eda/fase2_tr0_con_referencia_colectiva.csv` (casos limite) para
   fijar si TR = rango del objetivo (individuo vs grupo) o dirigido vs no
   dirigido.
2. **E00 (metrica)**: enviar `submissions/toxic_submission_v00_plantilla.csv`
   y reportar el score publico para identificar la implementacion de F1.

## Estado del arte (Fase 1, resumen)

- 45 referencias verificadas (`reports/estado_arte.md`, `.csv`, `references.bib`).
- Hallazgo central: MineriaUNAM (Argota Vega et al. 2019, DOI 10.18653/v1/S19-2079)
  gano la Subtask A en espanol de HatEval SIN transformers ni embeddings
  preentrenados: SVM lineal + char n-gramas, n-gramas de palabras funcionales
  2-4, n-gramas de puntuacion 2-4, POS, patrones morfologicos y series de
  palabras agresivas (macro-F1 0.73 en espanol).
- Evidencia util: Nina-Alcocer 2019 (TFIDF+EMOTIONS EMR 0.711 vs 0.629 base;
  TFIDF+STRUCT mejor F1 0.780); contraevidencia: combinar 3 bloques de features
  degrada y el ensemble por votacion no supera al SVM lineal (SINAI 0.707 vs 0.701).
- Hipotesis H1-H10 y experimentos E01-E10 en `reports/hipotesis.md`.
- Regla de decision pre-registrada: una mejora se adopta si es >= +0.005 absoluto
  en la metrica primaria y en >= 3 de 5 folds.

## Competicion

- Kaggle `tweet-toxic-2026-puj` (id 167384). Organizacion JarvisLab_PUJ.
- Metrica oficial: F1 Score. Deadline: 2026-10-19 23:00 UTC. Max 10 envios/dia.
- Nota: 70 % ranking Kaggle (LB privado) + 30 % calidad del notebook.
- Restricciones: sin transformers/LLM; sin embeddings preentrenados; todo
  vectorizador/embedding entrenado desde cero; test nunca para entrenar.

## Datos (verificados)

- `data/raw/toxic_train.csv` (205.417 B, sha256 3d40e734...): 1.731 filas,
  sep `;`, UTF-8. HS 756 (43.67 %), TR 519 (29.98 %), AG 662 (38.24 %).
- `data/raw/toxic_test.csv` (127.006 B, sha256 e91ff825...): 1.154 filas,
  sep `;`, **cp1252** (6 bytes 0xBF). Solo `id,text`.
- `data/raw/submission.csv` (13.860 B): plantilla `id,HS,TR,AG` separada por
  coma, todo 1,1,1. Ids identicos al test.
- Combinaciones HS x TR x AG: (0,0,0)=949, (1,1,1)=491, (1,0,1)=145,
  (1,0,0)=92, (1,1,0)=28, (0,0,1)=26. **No existe (0,1,\*)**.
- Texto ASCII sin tildes/enye (normalizacion de los organizadores).
- Longitud: 6-306 chars, mediana 91.

## Hallazgos de la auditoria (Fase 0)

- Parser correcto: quoting estandar CSV (el quoting QUOTE_NONE rompe 13 filas).
  1 fila con comilla anomala resuelta por el parser estandar.
- **10 textos de test estan en grupos de near-duplicados con train**
  (umbral 0.85; 4 grupos; 7 filas de train y 10 de test afectadas).
  Guardado en `data/processed/grupos_near_duplicados.csv`.
- 5 duplicados exactos dentro de test; 0 entre train y test.
- `id` no asociado a etiquetas (Spearman p>0.49; chi2 p>0.20): sin fuga por orden.
- 17 pseudo-autores (primera mencion @usuario) con >=3 tweets, 77 tweets
  (4.4 %); 9 grupos homogeneos en HS.
- Riesgo externo: el corpus sigue el esquema HatEval (SemEval-2019 T5 ES);
  prohibido usar etiquetas/textos externos (D-002).

## Leaderboard (registro en `submissions/leaderboard_log.csv`)

| Envio | Local (OOF) | Kaggle | Gap | Que es |
|---|---|---|---|---|
| v00 plantilla | - | 0.51494 | - | sonda E00 (todo 1,1,1) |
| v01 baseline TF-IDF+LR | 0.75778 | 0.75677 | -0.0010 | B1 word(1,2) |
| v02 union word+char LR | 0.79270 | 0.77993 | -0.0128 | MOD-03b, umbral 0.5 |
| **v03 union + umbrales** | 0.79890 | **0.78306** | -0.0158 | **MEJOR / modelo final** |
| v04 ensemble M2+M3 | 0.79850 | 0.77510 | -0.0234 | descartada |
| v05 OPT-01 Optuna | 0.80390 | 0.77602 | -0.0279 | descartada |
| v06 robustez C=1.0 | 0.79290 | 0.77227 | -0.0206 | descartada |
| v07 robustez min_df=5 | 0.79930 | 0.77034 | -0.0290 | descartada |

Conclusiones: metrica = F1 macro sobre 3 columnas (D-009); los umbrales por
etiqueta aportan +0.0031 pareado (D-012); **plateau de la familia union: el
rango 0.77034-0.78306 equivale a 0.99 errores estandar (n=1154) -> las
diferencias entre variantes son ruido** (D-014). La cota local es optimista entre
-0.013 y -0.029; el gap train-OOF NO predice el gap local-LB (D-013 refutada).

## Mejor modelo

**MODELO FINAL (D-014): MOD-03b_BEST - union TF-IDF word(1,2) min_df=2
sublinear + char_wb(3,5) min_df=2 sublinear + LogisticRegression C=2.0 balanced
x3, con umbrales por etiqueta 0.44/0.51/0.46.**
Script `experiments/04_refit_submissions_v02_v03.py`; modelo
`models/mod03b_lr_union.joblib`; submission
`submissions/toxic_submission_v03_lr_union_thr.csv` (LB **0.78306**, el mejor).
OOF 0.7927 (0.5) / 0.7989 (umbrales). Dimension 27.431.

**Plateau documentado (D-014)**: 7 submissions utiles; toda la familia union cae
en la banda 0.77034-0.78306 del LB, un rango de 0.0127 = **0.99 errores
estandar** (n=1154). Las diferencias entre variantes son ruido; el unico
contraste pareado fiable (v02 vs v03, mismo modelo) confirma +0.0031 por los
umbrales OOF. Se detiene el sondeo del leaderboard.

Alternativas evaluadas y descartadas con evidencia: ensembles (v04 0.77510),
Optuna OPT-01 (v05 0.77602), control de capacidad (v06 0.77227, v07 0.77034),
post-proceso jerarquico (-0.0056 local), features densas H11/H12/H14
(-0.009 a -0.032 local), todas las variantes de preprocesamiento (parte A).

## Mejores features

Pendiente. Hipotesis de partida por validar en Fases 2-8:
n-gramas de caracter (evasion ortografica), lexico de agresividad,
2a persona/imperativos (targeting), estilometria, sentimiento/emocion.

## Problemas conocidos / riesgos

1. Metrica exacta ambigua -> sonda E00 con `submission.csv` (todo 1) que envia
   Carlos; los valores esperados difieren por definicion (0.541 vs 0.543 vs
   0.270 vs 0.284 con distribucion de train).
2. Semantica de TR ambigua entre la definicion oficial y HatEval -> Fase 2.
3. Near-duplicados train-test -> validacion con `StratifiedGroupKFold`
   (grupos = near-duplicados, estratificado por combinacion).
4. Corpus sin diacriticos -> no se deben "inventar" tildes; documentar.
5. Fuga potencial por pseudo-autor -> control de robustez adicional.

## Hipotesis abiertas (a refinar en Fase 1)

- H1: char n-grams 3-6 capturan evasion ortografica mejor que word n-grams.
- H2: lexico agresivo + 2a persona separa AG.
- H3: features de targeting separan TR.
- H4: word+char > cada representacion sola.
- H5: sentimiento/emocion aporta informacion complementaria.
- H6: estilometria ayuda en tweets cortos.
- H7: lematizacion no aporta en tweets.
- H8: post-proceso jerarquico HS=0 => TR=0, AG=0 mejora F1.

## Proximos experimentos

- E00: identificacion de la metrica oficial (sonda de submission).
- E01: char n-grams vs word n-grams (Fase 5/9).
- Fases 2-4: EDA, textometria, variantes de preprocesamiento.

## Rutas clave

- Datos: `data/raw/`; grupos near-dup: `data/processed/`.
- Reportes: `reports/` (auditoria, data dictionary, eda/).
- Memoria: `PROJECT_STATE.md`, `RUNLOG.md`, `DECISIONS.md`, `experiments.csv`.
- Reglas verbatim: `docs/competition_rules.md`.
- Codigo: `src/data.py`; auditoria: `experiments/00_auditoria_datos.py`.

## Entorno

`/home/carlos/venv` (Python 3.14.7) - sklearn 1.8, lightgbm 4.6, xgboost 3.2,
catboost 1.2.10, nltk 3.9.4, spacy 3.8.13 + es_core_news_md, torch 2.14+rocm7.2,
kaggle 2.2.4, emoji 2.16.0, tabulate. GPU RX 9070 XT (ROCm) con protocolo de
chequeo previo (D-008).
