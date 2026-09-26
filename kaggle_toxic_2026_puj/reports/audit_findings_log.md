# Registro de hallazgos - Auditoria forense TOXIC 2026 PUJ

Actualizado fase a fase. Cada hallazgo apunta a su evidencia en este mismo
arbol de auditoria (`/home/carlos/Git/PLN-toxic-audit/kaggle_toxic_2026_puj/`).

| # | Fase | Hallazgo | Evidencia | Severidad | Estado |
|---|---|---|---|---|---|
| F-01 | P0 | Existen 3 superficies divergentes del proyecto: A=128 filas (v2-final), B=188 (v7), C=202 (v8). El "~184 filas" del encargo corresponde a B (188 filas; 183 ids unicos). Lineaje lineal A -> B -> C sin filas eliminadas. | reports/audit_git_provenance.md; audit/p00_git_provenance.py | INFO | resuelto |
| F-02 | P0 | Las ramas v5, v6, v7 y v8 NO existen en origin (solo v2 y v4); todo el trabajo posterior a v4 vive solo en local. | reports/audit_git_provenance.md (tabla de ramas) | MEDIUM (provenance) | abierto (recomendar push) |
| F-03 | P0 | El notebook A (v2) difiere de B==C (v7/v8 identicos). `src/evaluation.py` tambien difiere A vs B==C. Submissions identicas entre superficies salvo leaderboard_log y submission_manifest (v8 anadio sondas). | reports/audit_hashes_reference.csv | INFO | resuelto |
| F-04 | P0 | 3 commits de la sesion AGE v13 tocaron `tools/` del proyecto toxic: gitlink accidental de `tools/fastText` y add/retirada de `tools/verificar_caracteres.py` (213 lineas). En A el archivo queda sin versionar. | git show f68637c 32ab2f2; git log | LOW (higiene git) | abierto |
| F-05 | P1 | Los grupos de near-duplicados se construyen con train+test: `clusters_near_duplicados(train, test, 0.85)` ajusta TfidfVectorizer char_wb(4,4) sobre los 2885 textos y guarda `data/processed/grupos_near_duplicados.csv` con filas de test. 3 grupos mixtos (7 filas de train). Contrafactual solo-train: particion identica (ARI=1.0), holdout y pliegues IDENTICOS. | audit/p01_findings.json; reports/eda/clusters_near_duplicados.csv | L1 potencial, LOW (sin impacto demostrado) | documentado |
| F-06 | P1 | Dataset intacto: 1731/1154 filas, bytes y encodings documentados verificados; ids unicos y disjuntos; TR=>HS 0 violaciones; AG sin HS 26; prevalencias HS 43.67 / TR 29.98 / AG 38.24; 5 duplicados exactos en test; 0 textos compartidos train-test. | audit/p01_findings.json | PASS | cerrado |
| F-07 | P1 | Holdout v2 (248 filas) y 5 pliegues corregidos (296-297 filas) reproducen EXACTAMENTE los artefactos guardados; cero grupos compartidos; sin combinaciones ausentes. | audit/p01_findings.json | PASS | cerrado |
| F-08 | P2 | 6 filas malformadas (9 campos) con id `KAGGLE_TOXIC_2026_PUJ` en B y C, introducidas en el commit `ca275b9` (v5 P2, lote "EBD-01..06"). EBD-01..06 NUNCA existieron como filas propias: el registro de ese lote quedo como 6 copias corruptas. | audit/p02_findings.json; reports/audit_experiment_coverage.csv | HIGH (integridad de registro) | abierto |
| F-09 | P2 | 4 filas modificadas tras su introduccion: FIN-08-C-EMO, FIN-09-C-SYN, FIN-10-C-EMO-SYN (7f7f1d6 -> f5d23cb -> abd4ec4) y V6-07 (b63519f -> 834d565). Todas con commit explicativo; ninguna edicion silenciosa. | audit/out/p02_introduction_map.csv | INFO | documentado |
| F-10 | P2 | 5 filas sin cita de su id en reports/: FIN-08-C-EMO, FIN-09-C-SYN, FIN-10-C-EMO-SYN (sondas LB v09-v11), V4-F01-F04 (track agrupado) y P8-REFIT-C. | reports/audit_experiment_coverage.csv | LOW (documentacion) | pendiente P8 |
| F-11 | P2 | 10 scripts sin fila en la columna script del registro; 9 de 10 si aparecen citados en el texto del proyecto. `embedding_22_auditoria_final_v5_1.py` no aparece en ningun texto. | audit/p02_findings.json | LOW | pendiente P8 |
| F-12 | P2 | Huecos internos de numeracion: EBD-11 y EBD-12 no existen en ningun commit. | audit/p02_findings.json | INFO | documentado |
| F-13 | P0 | Conteos reconciliados del registro: 197 claves unicas = 196 filas validas + 1 clave corrupta (6 filas). Introduccion por fase: v1=115, v2=13, v4=15, v5=26, v6=10, v7=4, v8=14. | audit/out/p02_introduction_map.csv | INFO | cerrado |

| F-14 | P0/P3 | `features/features/oof_v03_final.npz` es una copia identica (mismo sha256) de `features/oof_v03_final.npz`: directorio anidado duplicado en la superficie v8 (higiene de artefactos). | sha256sum; audit/p03_findings.json | LOW | documentado |
| F-15 | P2 | `reports/v6_reproducibility.md` cita 9 veces rutas con prefijo duplicado `experiments/experiments/v6_*.py` (los ficheros reales estan en `experiments/`). | grep -c en v6_reproducibility.md | LOW (documentacion) | pendiente P8 |
| F-16 | P3 | Convencion confirmada: `corrected_cv` = F1 del OOF agrupado (pooled, 1483 filas), NO la media de pliegues; pooled vs media de pliegues difiere hasta 2.2e-4 (FIN-04: 0.779661 vs 0.779437). | audit/out/p03_numeric_checks.csv (A7, E-fold-*) | INFO | documentado |
| F-17 | P3 | `final_candidates.csv` calcula `delta_vs_v03` contra la referencia redondeada 0.7772 en vez de 0.777211: desvios de hasta 1.1e-5 en 10 filas. | p03_numeric_checks.csv (E-ref-*) | INFO | documentado |
| F-18 | P3 | V03 (referencia congelada) NO aplica post-proceso jerarquico: el `pred` guardado es el umbral directo; aplicar el postproceso cambiaria 55 celdas. Consistente con "sin post-proceso" en leaderboard_log. | p03_numeric_checks.csv (A2/A3) | INFO | documentado |
| F-19 | P3 | Verificacion numerica independiente: 50 PASS sin FAIL. Controles v6 7/7 = 0.777211 exactos; deltas coherentes (v6 25/25, v7 9/9, finals 15/15, netos v4 5/5); OOF historico reproduce 0.7927 (0.792659) y 0.7989 (0.798933); bootstrap reproducible (misma semilla) y con IC coherente; medianas del umbral anidado v7 coherentes con la dispersion declarada. | audit/out/p03_numeric_checks.csv | PASS | cerrado |
| F-20 | P3 | Evaluaciones del holdout registradas en `holdout_evaluaciones_v2.csv` (5 valores F1 localizados ademas en p7_holdout_resultados.json y otros artefactos). El log las documenta una a una con motivo y decision. | p03_findings.json | INFO | pendiente P4/P6 (auditoria de uso) |

## Pendiente por fase
- P4: leakage completo (thresholds, TF-IDF fit por pliegue, OOF real, instance-label, embeddings, test contamination, holdout).
- P5: salud de modelos (convergencia LR, DL, FastText downstream).
- P6: reproduccion (V03=0.777211, 7 controles v6, clave v4/v6/v7, FastText, DL).
- P7: notebook. P8: documentacion y 56 decisiones. P9: caracteres. P10: seguridad. P11: submissions. P12: correcciones.
