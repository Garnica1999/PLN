# Resumen del escaneo estático de experimentos

Fecha de referencia: 2026-10-02 (fijada en CONFIG para que la salida sea determinista).

## Alcance

- Archivos `.py` barridos: **376** (`experiments/**/*.py`, sin `__pycache__`).
- Archivos con error de sintaxis: **0**.
- Hallazgos totales: **2165**.

## Conteo por detector

| Detector | ALTA | MEDIA | BAJA | INFO | Total |
| --- | --- | --- | --- | --- | --- |
| D1_funciones_probadas_sin_llamador | 54 | 100 | 0 | 0 | 154 |
| D2_asercion_vacia | 0 | 0 | 0 | 0 | 0 |
| D3_rama_muerta | 0 | 3 | 0 | 0 | 3 |
| D4_break_en_medicion | 0 | 1 | 0 | 0 | 1 |
| D5_umbral_congelado_hardcodeado | 0 | 0 | 188 | 2 | 190 |
| D6_toca_test_o_holdout | 0 | 215 | 0 | 1448 | 1663 |
| D7_semilla_ausente | 0 | 0 | 154 | 0 | 154 |
| D8_except_desnudo | 0 | 0 | 0 | 0 | 0 |

## Top 10 de archivos por hallazgos de severidad ALTA o MEDIA

| Archivo | ALTA | MEDIA | Total |
| --- | --- | --- | --- |
| experiments/38_refit_final_v4.py | 0 | 32 | 32 |
| experiments/p81_01_cierre_huecos.py | 5 | 20 | 25 |
| experiments/p80_01_auditoria_w2v.py | 5 | 16 | 21 |
| experiments/p83_01_bayesiano.py | 5 | 16 | 21 |
| experiments/23_candidatos_finales.py | 0 | 21 | 21 |
| experiments/p82_01_rnn_atencion.py | 0 | 15 | 15 |
| experiments/p82_02_submission_rnn.py | 0 | 15 | 15 |
| experiments/p69_10_submission_candidato.py | 0 | 14 | 14 |
| experiments/p77_01_operador_selectivo.py | 1 | 6 | 7 |
| experiments/v33_01_validacion.py | 0 | 7 | 7 |

## Funciones D1 sin llamadores reales

Formato `archivo:línea:nombre`. Incluye las funciones sin ningún consumidor y las que solo son consumidas desde el modo de humo.

### Sin ningún llamador (severidad ALTA)

- `experiments/03_variantes_representaciones.py:889:grupo_features`
- `experiments/08_features_densas_h11_h12.py:1183:f1_train_medio`
- `experiments/10_sondas_robustez.py:225:nombre_modelo`
- `experiments/11_dl_desde_cero.py:1314:f1_val_media`
- `experiments/11_dl_desde_cero.py:1319:f1_val_std`
- `experiments/11_dl_desde_cero.py:1326:gap_train_val`
- `experiments/17_stacking_v2.py:222:key`
- `experiments/p59_01_voronoi_local.py:116:a_ascii`
- `experiments/p59_05_nmf_gap_analysis.py:120:a_ascii`
- `experiments/p59_06_robust_loss.py:353:f1_por_celda`
- `experiments/p61_02_inventario.py:112:a_ascii`
- `experiments/p62_01_eda_linguistica.py:168:hash_texto`
- `experiments/p62_07_presupuesto_canal.py:186:seleccionar_por_enriquecimiento`
- `experiments/p64_04_leave_one_out.py:287:escalar`
- `experiments/p66_03_complementariedad_bootstrap.py:255:residual_de`
- `experiments/p70_01_sensibilidad_tamano.py:101:a_ascii`
- `experiments/p75_01_complementariedad_textcnn.py:86:f1_filas`
- `experiments/p75_01_complementariedad_textcnn.py:214:n_alcanzables`
- `experiments/p77_01_operador_selectivo.py:167:celdas`
- `experiments/p80_01_auditoria_w2v.py:2950:nombres_de_expresion`
- `experiments/p80_01_auditoria_w2v.py:2965:atributo_exterior`
- `experiments/p80_01_auditoria_w2v.py:2984:clase_por_nombre`
- `experiments/p80_01_auditoria_w2v.py:4609:_ruta`
- `experiments/p80_01_auditoria_w2v.py:4783:keys`
- `experiments/p81_01_cierre_huecos.py:458:linea_de`
- `experiments/p81_01_cierre_huecos.py:768:txt_de_lista`
- `experiments/p81_01_cierre_huecos.py:4197:contrato_de_la_afirmacion`
- `experiments/p81_01_cierre_huecos.py:4211:_sin_valor`
- `experiments/p81_01_cierre_huecos.py:4316:normalizar_emb_normal`
- `experiments/p83_01_bayesiano.py:1203:con_extra`
- `experiments/p83_01_bayesiano.py:1332:avisar`
- `experiments/p83_01_bayesiano.py:1820:comprobar`
- `experiments/p83_01_bayesiano.py:2856:reoptimizar_umbral`
- `experiments/p83_01_bayesiano.py:2905:reproducir_liston`
- `experiments/v10_03_arnes_evaluacion.py:115:mejora`
- `experiments/v10_03_arnes_evaluacion.py:158:resumen`
- `experiments/v10_03_arnes_evaluacion.py:172:a_dict`
- `experiments/v10_03_arnes_evaluacion.py:330:evaluar`
- `experiments/v10_03_arnes_evaluacion.py:417:tabla_filas`
- `experiments/v10_09_correlacion_eliminacion.py:256:auc_de_una_caracteristica`
- `experiments/v11_ayudas.py:13:fila`
- `experiments/v11_ayudas.py:30:sello`
- `experiments/v12_02_perdida_y_estructura.py:394:subpliegues_internos`
- `experiments/v16_00_prevalencia_y_curva_lb.py:218:aplicar_prior`
- `experiments/v19_02_restriccion_tr_implica_hs.py:126:conteos`
- `experiments/v26_01_geometria_kernel.py:97:filas_de`
- `experiments/v26_02_tsetlin.py:121:log`
- `experiments/v26_02_tsetlin.py:130:memoria_pico_mb`
- `experiments/v26_02_tsetlin.py:139:filas_de`
- `experiments/v26_02_tsetlin.py:244:ajustar`
- `experiments/v26_02_tsetlin.py:326:predecir`
- `experiments/v32_00_reproducibilidad.py:344:hashlib_sha256`
- `experiments/v6_01_estructura_errores.py:111:_cargar_modulo`
- `experiments/v8_06_c_por_etiqueta.py:174:f1_media`

### Solo llamadores de humo (severidad MEDIA)

- `experiments/07_submission_v04_ens04.py:379:f1_columnas_rapido`
- `experiments/07_submission_v04_ens04.py:621:rejilla_umbrales`
- `experiments/07_submission_v04_ens04.py:660:evaluar_mediana_por_pliegue`
- `experiments/11_dl_desde_cero.py:290:muestra_gpu`
- `experiments/11_dl_desde_cero.py:402:registrar_preflight`
- `experiments/18_p7_holdout_y_submission_v08.py:230:sha256_de`
- `experiments/embedding_06_auditoria_v5.py:103:_leer_ruta`
- `experiments/embedding_06_auditoria_v5.py:325:informes_de_modelos`
- `experiments/embedding_11_auditoria_firmas_v5_1.py:298:_cargar_modulo`
- `experiments/p74_01_null_replicado.py:295:instrumentar`
- `experiments/p74_01_null_replicado.py:304:gancho`
- `experiments/p78_01_control_anidado.py:642:_dos_pasos`
- `experiments/p80_01_auditoria_w2v.py:2671:entrada_apunta_a_skipgram`
- `experiments/p80_01_auditoria_w2v.py:5126:fuente_sintetica_del_scheduler`
- `experiments/p80_01_auditoria_w2v.py:5196:fuente_sintetica_con_particion_interna`
- `experiments/p80_01_auditoria_w2v.py:5222:fuente_sintetica_con_clase_en_pascal`
- `experiments/p80_01_auditoria_w2v.py:5248:fuente_sintetica_con_call_site`
- `experiments/p80_01_auditoria_w2v.py:5268:fuente_sintetica_con_call_site_anidado`
- `experiments/p80_01_auditoria_w2v.py:5290:fuente_sintetica_del_pool_de_cv`
- `experiments/p80_01_auditoria_w2v.py:5318:fuente_sintetica_forma_de_11_dl`
- `experiments/p80_01_auditoria_w2v.py:5351:tipos_de_modulo_minimo`
- `experiments/p80_01_auditoria_w2v.py:5386:sondear_relleno_en_toy`
- `experiments/p80_01_auditoria_w2v.py:5452:npz_sintetico_en_memoria`
- `experiments/p80_01_auditoria_w2v.py:5468:npz_con_tabla_de_salida`
- `experiments/p80_01_auditoria_w2v.py:5484:npz_con_falsos_positivos`
- `experiments/p80_01_auditoria_w2v.py:5503:json_con_clave_de_forma`
- `experiments/p80_01_auditoria_w2v.py:5567:comprobar`
- `experiments/p80_01_auditoria_w2v.py:5867:lectura_rota`
- `experiments/p81_01_cierre_huecos.py:5037:_afirmacion_que_lanza`
- `experiments/p81_01_cierre_huecos.py:5049:_la_parada_no_se_traga`
- `experiments/p81_01_cierre_huecos.py:5072:_informe_declara_aislamiento`
- `experiments/p81_01_cierre_huecos.py:5095:_markdown_declara_robustez`
- `experiments/p81_01_cierre_huecos.py:5117:_detecta_modulo_cruzado`
- `experiments/p81_01_cierre_huecos.py:5133:_detecta_modulo_sin_constantes`
- `experiments/p81_01_cierre_huecos.py:5153:_detecta_origen_ausente`
- `experiments/p81_01_cierre_huecos.py:5169:_detecta_firma_desalineada`
- `experiments/p81_01_cierre_huecos.py:5187:_detecta_el_fallo_real_de_a18`
- `experiments/p81_01_cierre_huecos.py:5213:_checkpoint_cubre_las_afirmaciones_del_bloque_2`
- `experiments/p81_01_cierre_huecos.py:5255:_informe_declara_checkpoint_del_bloque_2`
- `experiments/p81_01_cierre_huecos.py:5323:_tres_parciales_consecutivos_no_se_pisan`
- `experiments/p81_01_cierre_huecos.py:5360:_el_fallo_de_un_parcial_degrada_a_aviso`
- `experiments/p81_01_cierre_huecos.py:5406:_escribir_un_final_dos_veces_aborta`
- `experiments/p81_01_cierre_huecos.py:5431:_una_parada_en_la_capa_de_escritura_aborta`
- `experiments/p81_01_cierre_huecos.py:5460:_informe_declara_el_patron_de_parciales`
- `experiments/p81_01_cierre_huecos.py:5480:_a19_con_a18_aislado_da_error`
- `experiments/p81_01_cierre_huecos.py:5499:_detecta_literal_cruzado`
- `experiments/p81_01_cierre_huecos.py:5533:comprobar`
- `experiments/p82_01_rnn_atencion.py:195:comprobar_generador_real`
- `experiments/p82_01_rnn_atencion.py:460:tabla_contratos`
- `experiments/p82_01_rnn_atencion.py:571:codigo_sin_afirmacion`
- `experiments/p82_01_rnn_atencion.py:660:auditar_booleanos_literales`
- `experiments/p82_01_rnn_atencion.py:1321:intervalo_bootstrap`
- `experiments/p82_01_rnn_atencion.py:1544:reoptimizar_umbral`
- `experiments/p82_01_rnn_atencion.py:1718:suma_hashes`
- `experiments/p82_01_rnn_atencion.py:1729:suma_de_trazas_de_filas`
- `experiments/p82_01_rnn_atencion.py:2034:construir_ajustes`
- `experiments/p82_01_rnn_atencion.py:4078:datos_sinteticos`
- `experiments/p82_01_rnn_atencion.py:4129:construir_contexto_sintetico`
- `experiments/p82_01_rnn_atencion.py:4262:stub`
- `experiments/p82_01_rnn_atencion.py:4285:stub_malo`
- `experiments/p82_01_rnn_atencion.py:4803:_cifras_de_interseccion`
- `experiments/p82_01_rnn_atencion.py:5316:verificar_mutaciones`
- `experiments/p82_02_submission_rnn.py:191:sha256_de_bytes`
- `experiments/p82_02_submission_rnn.py:225:verificar_identificadores`
- `experiments/p82_02_submission_rnn.py:261:verificar_no_degenerada`
- `experiments/p82_02_submission_rnn.py:291:construir_csv`
- `experiments/p82_02_submission_rnn.py:384:_columnas_descolocadas_fallan`
- `experiments/p82_02_submission_rnn.py:467:_manifiesto_canonico_una_fila`
- `experiments/p82_02_submission_rnn.py:489:_manifiesto_no_duplica`
- `experiments/p82_02_submission_rnn.py:501:_manifiesto_columnas_distintas_falla`
- `experiments/p82_02_submission_rnn.py:515:releidas_completas`
- `experiments/p82_02_submission_rnn.py:827:registrar`
- `experiments/p82_02_submission_rnn.py:841:codigo_salida`
- `experiments/p82_02_submission_rnn.py:1008:_ascii_rechazado`
- `experiments/p83_01_bayesiano.py:387:tabla_contratos`
- `experiments/p83_01_bayesiano.py:461:verificar_contratos`
- `experiments/p83_01_bayesiano.py:707:auditar_propio_script`
- `experiments/p83_01_bayesiano.py:729:codigo_sin_afirmacion`
- `experiments/p83_01_bayesiano.py:764:afirmar_evaluadas_en_completo`
- `experiments/p83_01_bayesiano.py:831:identificadores_de_llamadas`
- `experiments/p83_01_bayesiano.py:3926:comparar_contra_el_liston`
- `experiments/p83_01_bayesiano.py:3990:comprobar_presupuesto`
- `experiments/p83_01_bayesiano.py:4178:codigo_salida`
- `experiments/p83_01_bayesiano.py:4823:contexto_de_ejemplo`
- `experiments/p83_01_bayesiano.py:4847:_cifras_de_pregunta_1`
- `experiments/p83_01_bayesiano.py:4867:_medidas_de_pregunta_2`
- `experiments/p83_01_bayesiano.py:5002:_copia_sintetica_de_control`
- `experiments/p83_01_bayesiano.py:5045:_copia_mutada`
- `experiments/p83_01_bayesiano.py:5132:_falla_a3`
- `experiments/v25_01_factorization_machine.py:332:valor_en`
- `experiments/v6_00_gap_analysis.py:213:_codigo`
- `experiments/v6_00_gap_analysis.py:224:_artefactos`
- `experiments/v8_01_ensamble_folds.py:424:anadir`
- `experiments/v8_06_c_por_etiqueta.py:484:anadir`
- `experiments/v8_07_peso_por_etiqueta.py:470:anadir`
- `experiments/v8_08_corrector_margenes.py:541:anadir`
- `experiments/v8_10_ensemble_cinco_modelos.py:337:anadir`
- `experiments/v8_11_complejidad_idiosincrasias.py:481:anadir`
- `experiments/v8_12_seleccion_por_grupo.py:488:anadir`
- `experiments/v8_13_clasificadores_alternativos.py:457:anadir`

## Notas de interpretación

- D1 excluye `main`, `principal`, dunder, funciones decoradas con `fixture`, auxiliares de humo (`humo`, `caso`, `smoke`, `test_`) y ganchos de método de framework (`forward`, `setUp`, `tearDown`).
- D5 marca como informativo (severidad INFO) el archivo que lee los umbrales de un artefacto (`leer_umbrales` o un `np.load` de umbrales).
- D6 sube a MEDIA cuando la misma función menciona el test o el holdout y además contiene una llamada de ajuste (`fit` o `train`).
- D7 emite un solo hallazgo por archivo, en la primera línea de uso.
- El CSV `escaner_hallazgos.csv` es la fuente de este resumen y está ordenado por severidad y archivo.
