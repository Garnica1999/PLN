# TODO v2 - KAGGLE_TOXIC_2026_PUJ

Tablero vivo del plan v2 (`tasks/plan.md`). Se actualiza al cerrar cada gate.
Estados: pendiente / en curso / hecho / cerrado (sin mejora) / descartado.

## P0 - Scaffold

- [hecho] P0.1 Congelar V03 en `experiments/baseline_v03_frozen/` (hashes verificados)
- [hecho] P0.2 Holdout v2 congelado (dev 1.483 / holdout 248, sha256 `e5b62e5f...`)
- [hecho] P0.3 OOF corregido de los 4 bases (`features/v2_oof_base_corrected.npz`)
- [hecho] P0.4 `reports/advanced_gap_analysis_v2.md` y `reports/research_backlog_v2.md`
- [hecho] P0.5 D-016 + RUNLOG + experiments.csv + tablero

**Referencia v2 establecida**: V03 con umbrales sobre el desarrollo = **0.7772**
(M2 a 0.5 = 0.7718). Por etiqueta: HS 0.762 / TR 0.816 / AG 0.754.

## P1 - Label dependency suave

- [hecho] P1.1 Cadenas HS->TR->AG, HS->AG->TR, AG->HS->TR con P(otras etiquetas) OOF como feature (3 ordenes x 2 escalados, CV interna de 4 pliegues dentro de cada pliegue exterior): ninguna supera la referencia. Mejor variante HS_AG_TR_raw 0.7744 (-0.0028 vs V03 0.7772; 1/5 pliegues); el mejor ensemble_gain es +0.0053 pero con IC bootstrap que incluye 0. Linea CERRADA sin mejora: el score continuo de la etiqueta previa no aporta sobre la representacion de V03.
- [hecho] P1.2 Meta-modelo regularizado sobre scores OOF + margenes + entropias + 8 rasgos estructurales (44 features, LR con C en {0.1, 0.3, 1.0}): `experiments/15_cadenas_etiquetas_v2.py`. Verificacion ESTRICTAMENTE anidada (los 4 bases se reentrenan en cada pliegue) = **0.7853** (C 0.1), delta +0.0081 vs V03, 4/5 pliegues, ensemble_gain +0.0152 con IC bootstrap [+0.0021, +0.0248] > 0; la version con el OOF congelado da 0.7850 (coincide). ADOPTADO como candidato P1.
- [hecho] P1.3 Gate P1

## P2 - Targeting estructural

- [hecho] P2.1 Distancias target-insulto (token/char, orden, densidad, minimos): corrected_cv 0.6484 (delta vs V03 -0.1288, pliegues 0/5; AG recall colectiva 0.478 vs 0.452 de V03)
- [hecho] P2.2 Target profiling (persona/grupo/colectivo/gentilicio/2a persona): canal simbolico + PROPN/segunda persona/imperativo fold-locales; CH-TGT corrected_cv 0.6076 (delta vs V03 -0.1696, pliegues 0/5; AG recall colectiva 0.443 vs 0.452 de V03)
- [hecho] P2.3 Colectivo + agresion (linea de los FN de AG: 45.5 % vs 13.9 %): corrected_cv 0.7356 (delta vs V03 -0.0416, pliegues 1/5; AG recall colectiva 0.478 vs 0.452 de V03)
- [hecho] P2.4 Gate P2: CERRADO sin mejora (TXT+TGT-SPARSE@C=2.0, corrected_cv 0.7487)

## P3 - Sintaxis sparse

- [hecho] P3.1 POS n-gramas sparse (1-4) como CountVectorizer/TF-IDF: SYN-POS SYN-POS@C=1.0 corrected_cv 0.5637 (delta -0.2135 vs V03, pliegues 0/5); TXT+POS TXT+POS@C=2.0 corrected_cv 0.7656 (delta -0.0116 vs V03, pliegues 0/5)
- [hecho] P3.2 Dependencias como tokens simbolicos (PRON_VERB, NEG_VERB, AMOD_HEAD): SYN-DEPS SYN-DEPS@C=1.0 corrected_cv 0.5514 (delta -0.2258 vs V03, pliegues 0/5); TXT+DEPS TXT+DEPS@C=2.0 corrected_cv 0.7653 (delta -0.0119 vs V03, pliegues 1/5)
- [hecho] P3.3 Morfologia (numero/genero/tiempo/modo/persona) sparse: SYN-MORPH SYN-MORPH@C=1.0 corrected_cv 0.5477 (delta -0.2295 vs V03, pliegues 0/5); TXT+MORPH TXT+MORPH@C=2.0 corrected_cv 0.7567 (delta -0.0205 vs V03, pliegues 0/5); SYN-3 SYN-3@C=1.0 corrected_cv 0.5865 (delta -0.1907 vs V03, pliegues 0/5)
- [hecho] P3.4 Gate P3: CERRADO sin mejora (TXT+POS@C=2.0, corrected_cv 0.7656)

## P4 - Cita / atribucion / negacion (H13)

- [hecho] P4.1 Canales autor vs citado (comillas, RT, segun/dijo/afirmo): CITA CITA@C=2.0 corrected_cv 0.7437 (delta -0.0335 vs V03, pliegues 1/5); TXT+CITA TXT+CITA@C=2.0 corrected_cv 0.7554 (delta -0.0218 vs V03, pliegues 1/5)
- [hecho] P4.2 Insulto dentro/fuera de cita; target dentro/fuera: features del canal CITA (lexico inductivo fold-local); TXT+CITA TXT+CITA@C=2.0 corrected_cv 0.7554 (delta -0.0218 vs V03, pliegues 1/5)
- [hecho] P4.3 Negacion sintactica (neg + insulto, neg + target): NEG NEG@C=2.0 corrected_cv 0.5267 (delta -0.2505 vs V03, pliegues 0/5); TXT+NEG TXT+NEG@C=2.0 corrected_cv 0.7657 (delta -0.0115 vs V03, pliegues 0/5); TXT+CITA+NEG TXT+CITA+NEG@C=2.0 corrected_cv 0.7552 (delta -0.0220 vs V03, pliegues 1/5)
- [hecho] P4.4 Gate P4: CERRADO sin mejora (TXT+NEG@C=2.0, corrected_cv 0.7657)

## P5 - Lexico, emocion, NB-SVM y ortografia

- [hecho] P5.1 Emocion H5 (sin lexicos externos: `data/external/lexicons/` vacio y la sesion prohibe descargas; se usa el lexico inducido fold-local documentado en el JSON): EMO-EST EMO-EST@C=2.0 corrected_cv 0.5897 (delta -0.1875 vs V03, pliegues 0/5); EMO-LEX EMO-LEX@C=1.0 corrected_cv 0.6205 (delta -0.1567 vs V03, pliegues 0/5); EMO-ALL EMO-ALL@C=1.0 corrected_cv 0.6230 (delta -0.1542 vs V03, pliegues 0/5); TXT+EMO TXT+EMO@C=2.0 corrected_cv 0.6330 (delta -0.1442 vs V03, pliegues 0/5)
- [hecho] P5.2 NB-SVM (log-count ratio fold-local) en word/char/union: NB-WORD NB-WORD-LR@C=1.0 corrected_cv 0.7538 (delta -0.0234 vs V03, pliegues 0/5); NB-CHAR NB-CHAR-LR@C=1.0 corrected_cv 0.7380 (delta -0.0392 vs V03, pliegues 0/5); NB-UNION NB-UNION-LR@C=1.0 corrected_cv 0.7612 (delta -0.0160 vs V03, pliegues 2/5); NB-UNION-SVC NB-UNION-SVC@C=1.0 corrected_cv 0.7429 (delta -0.0343 vs V03, pliegues 0/5)
- [hecho] P5.3 Multicanal ortografico (raw + normalizado + patrones): ORT-A ORT-A@C=1.0 corrected_cv 0.7486 (delta -0.0286 vs V03, pliegues 0/5); ORT-ABCD ORT-ABCD@C=2.0 corrected_cv 0.7534 (delta -0.0238 vs V03, pliegues 0/5); ORT-AE ORT-AE@C=2.0 corrected_cv 0.7496 (delta -0.0276 vs V03, pliegues 0/5)
- [hecho] P5.4 Gate P5: CERRADO sin mejora (E2-09 TXT+EMO@C=2.0 0.6330; E2-15 NB-UNION-LR@C=1.0 0.7612; E2-10 ORT-AB@C=2.0 0.7534)

## P6 - Late fusion y stacking v2

- [hecho] P6.1 Canales: solo habia 2 canales OOF disponibles (4 bases de texto y 2 de targeting CH-TGT+DIST y TXT+CH); el meta usa 44 features (scores, margenes, entropias y estructurales base+extra); sintaxis/lexico/emocion no existian aun como canales OOF: los canales NO aportan al meta (linea cerrada)
- [hecho] P6.2 Meta-modelo con OOF anidado + bootstrap por tweet (2000 remuestreos): mejor META-BASES-RIDGE corrected_cv 0.7874 (delta vs v2-A +0.0020, pliegues 3/5, IC bootstrap del delta [-0.0054,+0.0093])
- [hecho] P6.3 Gate P6: ADOPTADO (Adoptado: complementariedad (ensemble_gain +0.0089, IC [+0.0019,+0.0160]).)

## P7 - Cierre

- [hecho] P7.1 Finalistas evaluados en el holdout congelado (2/5 evaluaciones, P7-01 V03 y P7-02 meta): V03 0.8285 vs meta 0.8465 (delta +0.0180; regla pre-registrada meta >= V03 - 0.005). Evidencia: `experiments/18_p7_holdout_y_submission_v08.py`, `reports/p7_holdout_resultados.json`, `reports/holdout_evaluaciones_v2.csv`
- [pendiente] P7.2 Auditoria (leakage/reglas/reproducibilidad) del candidato final
- [hecho] P7.3 2-3 sondas de LB con hipotesis: submission `submissions/toxic_submission_v08_meta_v2.csv` generada y validada (sonda v2 del meta Ridge alpha=3; la envia Carlos; hipotesis: superar el LB de v03 0.78306) | CIERRE 2026-09-25: sondas v09/v10/v11 enviadas (LB 0.77895 / 0.77978 / 0.78150); ninguna supera a V03 (0.78306); V03 sigue como referencia (D-024)
- [pendiente] P7.4 Actualizar notebook e informe si el candidato supera V03
- [pendiente] P7.5 Actualizar memoria (PROJECT_STATE, RUNLOG, DECISIONS, experiments.csv)

## P8 - Cierre del backlog restante (E2-13, E2-14, E2-16, E2-20, E2-21)

- [cerrado] P8.1 E2-13 encoder de entidades: cobertura informativa 0.005 < 0.20 -> CERRADA por sparsity; mejor META-ENT-RIDGE corrected_cv 0.7885 (delta vs meta +0.0011)
- [cerrado] P8.2 E2-14 calibracion: mejor META-ISOTONIC corrected_cv 0.7857 (delta vs su par sin calibrar -0.0017); invariante salvo empates, aporte solo en Brier/log-loss -> CERRADA sin mejora
- [cerrado] P8.3 E2-16 umbrales por rebanada con shrinkage: mejor SHRINK-M100 corrected_cv 0.7750 (delta vs global anidado -0.0017) -> CERRADA sin mejora
- [cerrado] P8.4 E2-20 especialistas por rebanada: 21 pruebas, 11 celdas significativas tras BH -> evaluados, ver JSON
- [cerrado] P8.5 E2-21 espacios por etiqueta: corrected_cv 0.7519 (delta vs espacio compartido -0.0199) -> CERRADA sin mejora
- [hecho] P8.6 Gate P8: backlog v2 cerrado al 100 % (24/24 lineas); ninguna de las 5 lineas restantes mejora la referencia 0.7772

## Registro de gates

| Gate | Fecha | Resultado |
|---|---|---|
| P0 | 2026-09-25 | SUPERADO. V03 congelada (hashes OK), holdout 248/d ev 1.483 congelado, OOF corregido de 4 bases, gap analysis y backlog v2, D-016. Referencia v2 = **0.7772** (V03 con umbrales en desarrollo) |
| P1 | 2026-09-25 | SUPERADO (parcial). E2-01 cadenas de etiquetas: NO adoptadas (mejor 0.7744, -0.0028 vs referencia; ninguna variante alcanza el criterio). E2-02 meta-modelo de dependencia: **ADOPTADO** como candidato, corrected_cv anidado **0.7853** (delta +0.0081 vs V03, 4/5 pliegues) y ensemble_gain +0.0152 con IC bootstrap [+0.0021, +0.0248] > 0; cota optimista con umbrales OOF 0.7888. Evidencia: `experiments/15_cadenas_etiquetas_v2.py`, `reports/cadenas_etiquetas_v2.{csv,json}`, `reports/figuras/fig_27_cadenas_v2.png`, filas E2-01-*/E2-02-META |
| P2 | 2026-09-25 | CERRADO (sin mejora). E2-05/E2-06 targeting estructural: mejor variante TXT+TGT-SPARSE@C=2.0 con corrected_cv 0.7487 (-0.0285 vs V03 0.7772, -0.0366 vs v2-A 0.7853, 1/5 pliegues). Evidencia: `experiments/16_targeting_v2.py`, `reports/targeting_v2.json`, `reports/figuras/fig_28_targeting_v2.png` |
| P3 | 2026-09-25 | CERRADO (sin mejora). P3 sintaxis/cita/negacion: mejor variante TXT+POS@C=2.0 con corrected_cv 0.7656 (-0.0116 vs V03 0.7772, 0/5 pliegues, ensemble_gain -0.0080). Evidencia: `experiments/19_sintaxis_cita_negacion_v2.py`, `reports/sintaxis_cita_negacion_v2.json` |
| P4 | 2026-09-25 | CERRADO (sin mejora). P4 sintaxis/cita/negacion: mejor variante TXT+NEG@C=2.0 con corrected_cv 0.7657 (-0.0115 vs V03 0.7772, 0/5 pliegues, ensemble_gain -0.0028). Evidencia: `experiments/19_sintaxis_cita_negacion_v2.py`, `reports/sintaxis_cita_negacion_v2.json` |
| P5 (P5 emocion) | 2026-09-25 | CERRADO (sin mejora). Mejor variante TXT+EMO@C=2.0 con corrected_cv 0.6330 (-0.1442 vs V03 0.7772, 0/5 pliegues, ensemble_gain -0.0782). Evidencia: `experiments/20_emocion_nbsvm_ortografia_v2.py`, `reports/emocion_nbsvm_ortografia_v2.json` |
| P5 (P5 NB-SVM) | 2026-09-25 | CERRADO (sin mejora). Mejor variante NB-UNION-LR@C=1.0 con corrected_cv 0.7612 (-0.0160 vs V03 0.7772, 2/5 pliegues, ensemble_gain +0.0038). Evidencia: `experiments/20_emocion_nbsvm_ortografia_v2.py`, `reports/emocion_nbsvm_ortografia_v2.json` |
| P5 (P5 ortografia) | 2026-09-25 | CERRADO (sin mejora). Mejor variante ORT-AB@C=2.0 con corrected_cv 0.7534 (-0.0238 vs V03 0.7772, 0/5 pliegues, ensemble_gain -0.0097). Evidencia: `experiments/20_emocion_nbsvm_ortografia_v2.py`, `reports/emocion_nbsvm_ortografia_v2.json` |
| P6 | 2026-09-25 | SUPERADO. E2-18 stacking v2: mejor meta META-BASES-RIDGE con corrected_cv 0.7874 (+0.0102 vs V03 0.7772, +0.0020 vs v2-A 0.7853, 3/5 pliegues); IC bootstrap del delta vs v2-A [-0.0054,+0.0093]; complementariedad ensemble_gain +0.0089 con IC [+0.0019,+0.0160]. Canales de targeting en el meta: NO. Evidencia: `experiments/17_stacking_v2.py`, `reports/stacking_v2.{csv,json}`, `reports/figuras/fig_29_stacking_v2.png` |
| P7 | 2026-09-25 | GO en el holdout. Holdout congelado (2/5 evaluaciones): V03 0.8285 vs meta Ridge alpha=3 0.8465 (delta pareado +0.0180, IC [+0.0003,+0.0372]); submission v08 validada (positivos HS 517, TR 375, AG 445 de 1154); auditoria `ml-reviewer` APROBADA (sin fuga en el stacking, submission identica al pipeline). **Sonda de LB: 0.78163** -> no supera a v03 (0.78306) |
| **CIERRE v2** | 2026-09-25 | **SIN MEJORA TRANSFERIBLE.** 24 lineas evaluadas: cerradas cadenas (E2-01), targeting (E2-05/06), sintaxis (E2-03/04), cita/negacion (E2-07/08), emocion (E2-09), NB-SVM (E2-15), ortografia (E2-10); meta de dependencia y stacking adoptados en local (+0.0081/+0.0102) y confirmados en holdout (+0.0180), pero **no transfirieron al test** (0.78163 vs 0.78306). Modelo final: **V03**. Informe: `reports/advanced_research_v2.md` |
| P8 | 2026-09-25 | CERRADO (sin mejora). Backlog v2 al 100 % (24/24 lineas). E2-13 encoder de entidades: cobertura informativa 0.005 < 0.20 -> cerrada por sparsity (META-ENT-RIDGE 0.7885). E2-14 calibracion: META-ISOTONIC 0.7857 (delta -0.0017 vs sin calibrar). E2-16 umbrales por rebanada: SHRINK-M100 0.7750 (delta -0.0017 vs global anidado). E2-20 especialistas: 11 celdas significativas tras BH; los especialistas son peores que el global dentro de la rebanada (delta medio de la mezcla -0.0058) -> cerrada sin mejora. E2-21 espacios por etiqueta: 0.7519 (delta -0.0199 vs compartido). Regla D-016 aplicada contra el comparador directo de cada mecanismo: solo el meta calibrado alcanza 0.7857 (5/5 pliegues) por reproducir el meta ya adoptado, sin aportar sobre el meta sin calibrar; ninguna linea aporta +0.005 con 3/5 pliegues ni complementariedad con IC > 0. Evidencia: `experiments/21_backlog_restante_v2.py`, `reports/backlog_restante_v2.{csv,json}`, `reports/figuras/fig_32_backlog_restante_v2.png` |

## P80 - Auditoria del entrenador Word2Vec (Fase 1)

Seccion nueva, anadida al final. No modifica nada de lo anterior.
Pre-registro: `docs/p80_preregistro_auditoria_w2v.md`. Script previsto:
`experiments/p80_01_auditoria_w2v.py`. Cero entrenamientos del Word2Vec.

- [hecho] P80.1 Leer el plan maestro `p80_p82_auditoria_w2v_rnn_bayesiano.md` (Fases 0-8), las secciones citadas de `p79_01_auditoria_sgns.py`, `p71_04:182-216`, `p72_01:62-136` y `:152-162`, `reports/p79_auditoria_sgns.json` y `docs/p78_preregistro.md`
- [hecho] P80.2 Escribir `docs/p80_preregistro_auditoria_w2v.md` con 19 afirmaciones (A1-A19) que cubren H1 a H7, cada una con tolerancia, exito, fallo, marca de STOP y Naturaleza (CONFIRMAR / REFUTAR / NO APLICABLE POR CONSTRUCCION), mas criterio de parada y la frase literal `V03 remains the best verified leaderboard result.`
- [hecho] P80.3 Verificar ASCII puro byte a byte (0 bytes mayores que 127) y con `tools/verificar_caracteres.py`. Anadir la seccion P80 a `tasks/plan.md` y a este fichero, sin reescribir lo previo
- [pendiente] P80.4 Gate 1: commit del pre-registro con su sha256 en el mensaje y en `RUNLOG.md`, con `-F fichero` (regla de P78: las comillas internas truncan el mensaje). Sin commit NO se ejecuta
- [pendiente] P80.5 Implementar `experiments/p80_01_auditoria_w2v.py` con `auditar_semillas_por_ast`, `test1_loss`, `test2_gradientes`, `test3_scheduler`, `test4_identidad_con_signo`, `test5_artefacto`, `test6_bigru`, `test7_fase_dl` y `main` que se detiene en el primer STOP
- [pendiente] P80.6 Gate 3: revision de `ml-reviewer` con 5 ejes ML, buscando replicacion circular, comparaciones con el signo mal puesto, uso de una matriz que no existe y aserciones que no abortan. Sin dictamen no se ejecuta
- [pendiente] P80.7 Ejecutar con `/home/carlos/venv/bin/python`. sha256 del congelado `302c994d...d705a` comprobado antes de la primera lectura y despues de la ultima, junto con `st_size` y `st_mtime_ns`; STOP inmediato si cualquiera de los tres cambia
- [pendiente] P80.8 Emitir `reports/p80_auditoria_w2v.json` y `reports/p80_auditoria_w2v.md` con veredicto por hallazgo en exactamente tres estados: CONFIRMADO, REFUTADO, NO APLICABLE POR CONSTRUCCION. ASCII verificado dos veces
- [pendiente] P80.9 Corregir en documento NUEVO las tres afirmaciones erroneas de P79: el booleano `semillas_desalineadas`, el signo de `la_identidad_se_cumple`, y el bloque `punto_5_artefacto_congelado`. NO editar P79 ni su JSON
- [pendiente] P80.10 Dictamenes en paralelo de `dl-specialist`, `nlp-specialist`, `ml-reviewer` y `kaggle-specialist`. Si contradicen a los datos, ganan los datos y se documenta el por que
- [pendiente] P80.11 Gate 4: actualizar DECISIONS, RUNLOG y PROJECT_STATE con el veredicto por hallazgo
- [pendiente] P80.12 Decision de la Fase RNN con los datos de P80. Si H5 y H6 invalidan V9-04 y V10-11, la via RNN se cierra con evidencia y no se abre P81. Si son reinterpretables, se pre-registra P81 (early stopping anidado, relleno correcto, sin atencion) y lo decide Carlos

### Notas de frontera para P80

- Prohibido afirmar que Word2Vec sea mejor que V03. Prohibido modificar V03 ni sus
  umbrales 0.44 / 0.51 / 0.46. Prohibido reentrenar o sobrescribir el artefacto
  Word2Vec 100d. Prohibido leer test, holdout o leaderboard, o generar submission.
- De `config/holdout_v2.json` solo se leen `n_tweets_dev` y `n_tweets_holdout`; la
  lista de ids no se deserializa ni se cruza con ninguna tabla.
- Los numeros de P73, P74, P77 y P79 se conservan intactos. Si uno esta mal, se
  corrige la AFIRMACION con un documento nuevo, no el informe viejo.
