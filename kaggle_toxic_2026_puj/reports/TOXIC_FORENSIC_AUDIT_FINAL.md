# TOXIC 2026 PUJ - Auditoria forense integral (INFORME FINAL)

Proyecto: `kaggle_toxic_2026_puj` (competencia `tweet-toxic-2026-puj`).
Snapshot auditado: worktree v8 (rama `proyecto-kaggle-toxic-2026-v8`, commit b0edf7a),
con las superficies A (directorio actual, v2-final), B (v7) y C (v8) contrastadas.
Rama de auditoria: `proyecto-kaggle-toxic-2026-audit`. Todos los datos son read-only
sobre las superficies originales; los reentrenos corren en un worktree aislado
(`PLN-toxic-rerun`) con copia local de datos y corpora.

---

## Executive summary

- Integridad experimental: PASS CON OBSERVACIONES. Todos los numeros centrales se
  recomputaron de forma independiente (50 comprobaciones, 0 fallos) y el PBE
  principal (V03 = 0.777211) se reproduce exacto desde el OOF guardado.
- Fugas: ninguna fuga de etiquetas. Dos mecanismos de informacion estructural se
  midieron: near-duplicados train+test en la construccion de grupos (impacto
  demostrado 0.0 sobre particiones) y pesos de sospecha de v6-07 con etiquetas del
  pliegue evaluado (impacto medido +0.000142 -> 0 con version fold-local).
- Reproducibilidad: ALTA. 13 grupos R3 (incluidas las submissions v01-v03 byte a
  byte y la cadena v4 completa), 2 familias R2 con tolerancia medida (DL-02
  +-5.4e-4; FastText semantico con vecinos 0.833), 2 lineas R1 por presupuesto
  (ejes v5.1; 21/23/25) y una excepcion: el notebook final NO ejecuta por una
  asercion desactualizada (B-14).
- Notebook: ejecutable y verificado (63 celdas, 0 errores) pero INCOMPLETO: cubre
  v1-v5.1; no tiene secciones de v6, v7 ni v8.
- Documentacion: 4 inconsistencias reales (rutas duplicadas, '2/5' del holdout,
  colision de numeracion de submissions, filas EBD corruptas) + 8 conclusiones
  pendientes en DECISIONS.md. Ninguna afecta resultados.
- Caracteres: puerta oficial 0 errores; 0 emoji/zero-width/contaminacion externa.
- Seguridad: 0 secretos en arbol e historia.
- Mejor submission historica y recomendada: V03 (LB 0.78306); las 15 sondas
  registradas confirman que ninguna variante posterior la supera.

---

## 1. Alcance del proyecto

- Tarea: clasificacion multi-etiqueta (HS, TR, AG) de tuits en espanol; metrica
  primaria F1 media column-wise (D-009, sonda E00); restricciones: sin
  transformers/LLM, sin recursos externos etiquetados, embeddings desde cero
  (D-002/D-003/D-007/D-030).
- Superficies: A=v2-final (128 filas de registro), B=v7 (188), C=v8 (202).
- Fases: v1, v2, v4, v5, v5.1, v6, v7, v8; 56 commits que tocan el proyecto.

## 2. Estado del repositorio

- Worktrees: principal (`kaggle-age-2026-puj-v13`), v7, v8 y los dos de auditoria.
- Ramas v5..v8 SOLO en local (no estan en origin); v2 y v4 si. Riesgo de perdida
  documentado (F-02); recomendacion: push de respaldo.
- Limpieza: A tiene `tools/` sin versionar (fastText + verificar_caracteres.py);
  B y C limpios. Los commits de la sesion AGE que tocaron `tools/` del toxic
  (accidente `git add -A`) fueron retirados del indice (F-04).

## 3. Provenance git

- 56 commits sobre el path; linea de fases: v1 (1) -> v2 (6) -> v4 (6) ->
  v5/v5.1 (16) -> v6 (10) -> v7 (5) -> v8 (9) + 3 commits AGE accidentales.
- Por ruta: `audit_git_provenance.csv` (539 rutas con primer/ultimo commit).
- Introduccion por fase del registro: v1=115, v2=13, v4=15, v5=26, v6=10, v7=4,
  v8=14 claves; 0 filas eliminadas; 4 filas modificadas, todas con commit
  explicito (FIN-08/09/10 y V6-07).
- Sin stash del toxic (el unico stash es del proyecto AGE; no se toco).

## 4. Integridad del dataset

- 1731 train / 1154 test; bytes y encodings documentados verificados (train
  utf-8; test cp1252 con 6 bytes '¿'); ids unicos y disjuntos; test = plantilla.
- Etiquetas: HS 43.67 %, TR 29.98 %, AG 38.24 %; TR=>HS 0 violaciones; 26 casos
  AG=1 con HS=0; 5 duplicados exactos en test; 0 textos compartidos train-test.
- Near-duplicados: 4 grupos no triviales; 3 de ellos mezclan train y test (7
  filas train, 4 test); 6 pares train-test >= 0.90 (H2 los explota como sonda).
- Hashes de referencia en `audit_hashes_reference.csv` (131 filas).

## 5. Inventario de experimentos

- Registro final: 202 filas / 197 claves = 196 validas + 1 clave corrupta (6
  copias de una fila de 9 campos del lote EBD-01..06, commit ca275b9). EBD-01..06
  no existen como filas; EBD-11/12 tambien faltan (B-01/B-02).
- 10 scripts sin fila directa en la columna `script` (9 citados en el texto).
- Cobertura completa: `audit_experiment_coverage.csv`.

## 6. Consistencia numerica

- `audit/out/p03_numeric_checks.csv`: 50 PASS, 0 FAIL, 0 UNKNOWN, 13 INFO.
- V03 recomputado desde el OOF: 0.7772115 (pooled) = 0.777211 publicado; F1 por
  etiqueta .7617/.8157/.7542; f1_pliegues guardados reproducen.
- Deltas verificados aritmeticamente en v4 (5/5), v6 (25/25), v7 (9/9), finals
  (15/15). OOF historico 0.792659 / 0.798933 = 0.7927 / 0.7989.
- Convenciones documentadas: `corrected_cv` es pooled (no media de pliegues,
  difiere hasta 2.2e-4); finals usan referencia redondeada 0.7772 (B-10); V03 no
  aplica post-proceso jerarquico (55 celdas).

## 7. Protocolo de validacion

- Grupos = near-duplicado UNION pseudo-autor (union-find) calculados con train;
  holdout 248 filas y 5 pliegues 296-297 reproducen EXACTAMENTE los artefactos;
  0 grupos compartidos; sin combinaciones ausentes.
- El clustering base se ajusto sobre train+test (F-05/B-07): mecanismo presente,
  impacto medido NULO (contrafactual solo-train: ARI=1.0; holdout y pliegues
  identicos).
- Calibracion anidada v7 verificada como protocolo real de 3 niveles (modelo en
  ajuste interior, umbral en calibracion interior, evaluacion unica exterior).
- Stacking v1/v2 y target encoding: fold-local verificados por codigo.

## 8. Auditoria de leakage (`audit_leakage.csv`, 10 filas)

- L1 near-dup train+test: sin impacto (F-05).
- L4 umbrales historicos: optimizar sobre el mismo OOF infla +0.003629 (reprodujo
  la cifra "+0.0036" citada); anidado por reutilizacion deja +0.000925; corregido
  por D-016 (umbrales congelados).
- L4 v6-07: pesos con etiquetas del pliegue evaluado; 5819 relaciones cruzadas,
  2139 opuestas; contrafactual fold-local 0.000142 -> 0.000000 (F-22).
- L5 test contamination: 0/2118 patrones en 32 txt + gz + dump bz2 completo.
- L7 embeddings: cumplimiento D-030; 5/5 hashes de corpus verificados; test
  excluido; pysentimiento/VADER descartado.
- L0: splits, stacking, target encoding, parquet de features, holdout (5/5 usos).

## 9. Salud de modelos (`audit_model_health.csv`, 22 filas)

- M2/V03: liblinear converge en 5-6 iteraciones, 0 warnings, norma L2 media 22.8;
  gap in-sample vs OOF +0.198 (memorizacion estable, no underfitting).
- M3 LinearSVC: 33-35 iteraciones, 0 warnings.
- DL: DL-01 MLP (0.7949 OOF historico) > DL-02 CharCNN (0.741) > DL-03 BiLSTM
  (0.736); curvas guardadas en el JSON.
- Embeddings downstream: SOLO pooling por media + modelo lineal; nunca secuencia
  (H1 nace de esta brecha, §24 del encargo).

## 10. Reproducibilidad (P6)

Matriz completa: `audit_reproducibility_matrix.csv` (20 filas). Re-ejecucion en
worktree aislado (`PLN-toxic-rerun`) con copia local de datos y corpora, sin red:

- **R3 (exacto; solo cambian fecha/duracion en los JSON)**: V03 (14), control
  pseudo-autor, los 7 controles v6, v7 completo (P9/P8/P0.1), v8 (01/03/05),
  la cadena v4 completa (14->16->17->19->20->27..36; 15/15 scripts exit=0;
  `v4_candidates.json` exacto: la tabla que sostiene D-029 se reproduce),
  las submissions v01/v02/v03 REGENERADAS BYTE A BYTE (sha256 identico) y
  DL-01/DL-03.
- **R2 (tolerancia medida)**: DL-02 CharCNN +-5.4e-4 OOF (no determinismo GPU,
  coherente con su multilla sigma=0.0011); una especificacion de stacking v2
  (max 1.1e-3) y una variante de structural_sparse (max 6.4e-4); FastText
  baseline (no byte-determinista con 16 hilos; vocabulario exacto 100098;
  solapamiento de vecinos top-10 = 0.833; su propio control de semillas daba 0.792).
- **R1 (no re-ejecutado por presupuesto; entorno completo)**: ejes v5.1 y
  scripts 21/23/25.
- **NO EJECUTA en el estado final**: el notebook (B-14: la celda 30 exige "12
  sondas" y el log tiene 15; EBD-26 era previo a las sondas v8).
- Matices documentados: cadena de dependencias v4 (stores intermedios no
  retenidos; resuelta re-ejecutando la cadena completa) y `modo_anotacion`
  'spacy-cache' vs 'spacy' (las anotaciones se regeneraron; las cifras no cambian).

## 11. Notebook (`audit_notebook.csv`, `audit_notebook_gaps.md`)

- 63 celdas (32 codigo), 0 errores, ejecucion completa registrada (EBD-26, 4.6 s,
  GO). Celdas de verificacion de cifras clave presentes (0.777211, 0.78306,
  0.7989, 0.7927, 0.8285, 0.8465, 0.78163).
- Estructura: secciones 1-18 cubren v1, v2, v4, v5/v5.1; NO hay secciones v6/v7/v8.
- Estado: documentado y verificable en cifras; no reproducible end-to-end desde el
  notebook para el modelado; incompleto para las fases finales.

## 12. Consistencia documental (`audit_documentation.csv`)

- Inconsistencias reales: B-03 (rutas `experiments/experiments/`), B-04 ('2/5' del
  holdout), B-09 (colision de numeracion v08-v12), B-01 (filas EBD).
- Informativas verificadas: conteo del registro, celdas del notebook, 15 sondas.
- 67 decisiones: 58 VALIDATED, 5 PARTIALLY (D-009/D-020/D-043/D-050/D-063),
  3 SUPERSEDED (D-004/D-014/D-019), 1 INVALIDATED (D-013; cuerpo actualizado).
  8 con conclusion pendiente (B-05).
- Claims de "techo": cotas empiricas medidas (interseccion +0.0027, curva plana,
  umbrales debilmente identificados), no teoremas (excepto cotas derivadas).

## 13. Caracteres e idioma (`audit_characters.md`)

- Puerta oficial: 0 errores. Escaneo ampliado (487 archivos): 0 emoji, 0
  zero-width, 0 selectores, 0 contaminacion extranjera; 5148 caracteres de
  tipografia espanola en documentos generados (desviacion LOW de ASCII estricto,
  no corregida para no invalidar EBD-26 ni las huellas).
- Emoticonos: 0 reales en contenido generado (falsos positivos en NOT_TESTED,
  gpu_use e IDs); el dataset conserva los suyos intactos.

## 14. Seguridad

- 0 secretos en el arbol y en la historia del path (9 familias de patrones).
- El token de Kaggle se lee de entorno en `src/data.py` y nunca se escribe.

## 15. Submissions (`audit_submissions.csv`, `audit_transfer_analysis.csv`)

- 47 archivos validados en A/B/C: ids en orden, binarias, sin nulos, hashes.
- Log de LB: 15 filas; doble semantica de `diferencia` (12 filas "LB-local",
  3 de v8 "LB-LB(V03)", B-12).
- Mejor LB: V03 0.78306. Mejor sin V03: meta v08 0.78163; candidato c 0.78150.
- Gaps local-LB: -0.0290 (v07) a +0.0018 (v11); patron: menos capacidad y metas
  restringidos transfieren mejor; la union con mayor capacidad transfiere peor.
- **Regeneracion de submissions (P6j, `audit_submission_regeneration.csv`)**: se
  re-ajustaron los modelos en el worktree aislado y se regeneraron 16/16
  submissions historicas **byte a byte identicas** (v01-v04, v06-v07,
  v08_p1_fold_ensemble, v09-v11, v11_corrector, v12-v16; 17/17 scripts exit=0).
  Excluidos con motivo: v05 (re-busqueda Optuna = experimentacion nueva) y
  v08_meta (script 18 evalua el holdout). Observacion: v12 y v13 son
  byte-identicos entre si (el ensemble de cinco modelos produce el mismo
  etiquetado que el promedio de repeticiones; coherente con D-067).

## 16. Decisiones

- Ver seccion 12. D-013 queda INVALIDADA por su propia refutacion (registrada);
  D-004/D-014/D-019 SUPERSEDED; las 5 PARTIALLY tienen razon auditada.

## 17. Conclusiones historicas

- Familias v4-v7: cierres validos con la salvedad de v6-07 (F-22); v5/v5.1: cierre
  valido con reproducibilidad limitada a nivel de representacion; v8: cierre valido
  (las sondas LB confirman el limite externo).
- "El limite esta en el externo" (D-059) es la sintesis mas consistente con toda la
  evidencia: el gap local-LB no lo explican capacidad, representacion ni calibracion.

## 18. Transfer gap

- Patron central: el OOF local premia ajuste a la prevalencia de train; el test
  tiene ~7 % menos positivos (E00) -> cuanto mas se adapta el modelo al OOF
  (capacidad alta), peor transfiere (optuna -0.0279; union -0.013/-0.029;
  word-only -0.001; metas restringidos ~0.000).
- Implicacion: el cuello de botella es de DISTRIBUCION/CALIBRACION, no de
  representacion (H4 la ataca con prior shift EM).

## 19. Hallazgos corregidos

- B-01..B-13 en `audit_corrections.md`; impactos cuantificados: +0.0036 (umbrales,
  historico), +0.000142 -> 0 (v6-07), 0.0 (near-dup), resto documental/higiene.

## 20. Nuevas hipotesis (`audit_new_hypotheses.md/.csv`)

- H1 secuencial sobre embeddings (A), H2 near-dup train->test (A), H3 PLS fold-local
  (A), H4 prior shift EM (C), H5 revalidacion de sensibilidad (A).

## 21. Fase siguiente recomendada

1) H5 (sensibilidad, minutos), 2) H4 (sonda unica pre-registrada), 3) H2 (sonda
barata con verificacion manual de pares), 4) H1 (GPU), 5) H3 (condicional). Antes:
aplicar B-01 (registro), push de ramas y actualizar notebook con v6-v8 si se desea
entregable completo.

## 22. Veredicto final

- Integridad experimental: PASS CON OBSERVACIONES
- Data leakage: NINGUNA DETECTADA (dos mecanismos potenciales medidos con impacto
  0.0 y 1.4e-4, ninguno adoptado)
- Consistencia numerica: PASS (50 checks; 13 convenciones documentadas)
- Reproducibilidad: ALTA (R3 en 13 grupos; R2 acotado en DL-02 y FastText con
  tolerancia medida; R1 en presupuesto no re-ejecutado; notebook no ejecutable
  en el estado final por B-14)
- Completitud del notebook: PARCIAL (v6-v8 ausentes)
- Consistencia documental: ISSUES (4 reales, ninguna cientifica)
- Salud de modelos: PASS
- Cumplimiento de caracteres: PASS (con desviacion LOW de estilo ASCII)
- Seguridad: PASS
- Conclusiones cientificas: SOPORTADAS (con D-013 invalidada y D-050 matizada)

## Respuestas directas a las 12 preguntas del encargo

1. V03 reproducible: SI. Re-ejecucion completa R3 (0.777211 exacto) y la
   submission v03 regenerada BYTE A BYTE (sha256 7d7478f2...).
2. 184 experimentos trazados: el registro real es 197 claves (196 validas + 1
   corrupta); 47 commits de registro mapeados commit a commit.
3. Discrepancias numericas: 0 en valores centrales; 13 convenciones/INFO.
4. Leakage: 0 confirmadas; 2 potenciales cuantificadas sin impacto en decisiones.
5. Problemas de entrenamiento: ninguno (convergencia limpia; DL con curvas).
6. Problemas de documentacion: 4 reales + 8 conclusiones pendientes.
7. Notebook completo: NO (falta v6-v8).
8. Caracteres: PASS en la puerta; desviacion LOW de ASCII estricto documentada.
9. Resultados a invalidar: ninguno; D-013 ya invalidada en su momento.
10. Decisiones a reabrir: ninguna por la auditoria; 5 parciales con razon medida.
11. Hipotesis que sobreviven: H1, H2, H3, H4 (y H5 como revalidacion).
12. Orden siguiente: H5 -> H4 -> H2 -> H1 -> H3 (seccion 21).
