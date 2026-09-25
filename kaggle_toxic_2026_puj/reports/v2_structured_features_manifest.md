# Manifest - features estructuradas v2 (FASE 55/89)

Fecha: 2026-09-25. Artefacto: `features/v2_structured_features.parquet`.
Contenido: 2885 filas x 123 features, mas `id` y `particion`
(dev=1483, holdout=248, test=1154). Solo features deterministas y sin etiquetas.

## Bloques y fuentes

| bloque | columnas | fuente | fold-safety |
| --- | --- | --- | --- |
| longitudes_y_conteos | 15 | len(text), text.split() y regex de P6.PATRON_MENCION/PATRON_HASHTAG/PATRON_URL; n_tokens/n_oraciones/n_propn/n_targets de la anotacion spaCy | determinista, sin etiquetas |
| targeting_estructural | 13 | regla estructural de P2.es_colectivo_estructural y P2.es_segunda_persona/P2.es_imperativo replicadas sobre la anotacion serializada; inventarios fijos de P2.NEGACION/CUANTIFICADORES_UNIVERSALES y constantes del script 06 | determinista, sin etiquetas |
| cita_negacion | 13 | P3.tramos_comillas, P3.PATRON_RT, P3.MARCADORES_ATRIBUCION y P2.NEGACION sobre la anotacion serializada | determinista, sin etiquetas |
| ortografia | 12 | P5.PATRON_REPETICIONES, P5.PATRON_PUNTUACION_INTERNA, PATRON_LEET_PROXY (proxy de la tabla leet de P5), P6.PATRON_ELONGACION, P6.PATRON_RISAS y P6.PATRON_MAYUSCULAS_SOSTENIDAS | determinista, sin etiquetas |
| morfologia | 3 | rasgos Number/Gender/Tense de la anotacion spaCy serializada | determinista, sin etiquetas |
| sintaxis (POS y dependencias) | 67 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |

## Columnas

| columna | dtype | fuente | fold-safety |
| --- | --- | --- | --- |
| n_chars | int64 | len(text), text.split() y regex de P6.PATRON_MENCION/PATRON_HASHTAG/PATRON_URL; n_tokens/n_oraciones/n_propn/n_targets de la anotacion spaCy | determinista, sin etiquetas |
| n_tokens_split | int64 | len(text), text.split() y regex de P6.PATRON_MENCION/PATRON_HASHTAG/PATRON_URL; n_tokens/n_oraciones/n_propn/n_targets de la anotacion spaCy | determinista, sin etiquetas |
| n_tokens | int64 | len(text), text.split() y regex de P6.PATRON_MENCION/PATRON_HASHTAG/PATRON_URL; n_tokens/n_oraciones/n_propn/n_targets de la anotacion spaCy | determinista, sin etiquetas |
| n_oraciones | int64 | len(text), text.split() y regex de P6.PATRON_MENCION/PATRON_HASHTAG/PATRON_URL; n_tokens/n_oraciones/n_propn/n_targets de la anotacion spaCy | determinista, sin etiquetas |
| n_menciones | int64 | len(text), text.split() y regex de P6.PATRON_MENCION/PATRON_HASHTAG/PATRON_URL; n_tokens/n_oraciones/n_propn/n_targets de la anotacion spaCy | determinista, sin etiquetas |
| n_mencion_inicial | int64 | len(text), text.split() y regex de P6.PATRON_MENCION/PATRON_HASHTAG/PATRON_URL; n_tokens/n_oraciones/n_propn/n_targets de la anotacion spaCy | determinista, sin etiquetas |
| n_mencion_interna | int64 | len(text), text.split() y regex de P6.PATRON_MENCION/PATRON_HASHTAG/PATRON_URL; n_tokens/n_oraciones/n_propn/n_targets de la anotacion spaCy | determinista, sin etiquetas |
| n_hashtags | int64 | len(text), text.split() y regex de P6.PATRON_MENCION/PATRON_HASHTAG/PATRON_URL; n_tokens/n_oraciones/n_propn/n_targets de la anotacion spaCy | determinista, sin etiquetas |
| n_urls | int64 | len(text), text.split() y regex de P6.PATRON_MENCION/PATRON_HASHTAG/PATRON_URL; n_tokens/n_oraciones/n_propn/n_targets de la anotacion spaCy | determinista, sin etiquetas |
| tiene_mencion | int64 | len(text), text.split() y regex de P6.PATRON_MENCION/PATRON_HASHTAG/PATRON_URL; n_tokens/n_oraciones/n_propn/n_targets de la anotacion spaCy | determinista, sin etiquetas |
| tiene_mencion_inicial | int64 | len(text), text.split() y regex de P6.PATRON_MENCION/PATRON_HASHTAG/PATRON_URL; n_tokens/n_oraciones/n_propn/n_targets de la anotacion spaCy | determinista, sin etiquetas |
| tiene_hashtag | int64 | len(text), text.split() y regex de P6.PATRON_MENCION/PATRON_HASHTAG/PATRON_URL; n_tokens/n_oraciones/n_propn/n_targets de la anotacion spaCy | determinista, sin etiquetas |
| tiene_url | int64 | len(text), text.split() y regex de P6.PATRON_MENCION/PATRON_HASHTAG/PATRON_URL; n_tokens/n_oraciones/n_propn/n_targets de la anotacion spaCy | determinista, sin etiquetas |
| n_propn | int64 | len(text), text.split() y regex de P6.PATRON_MENCION/PATRON_HASHTAG/PATRON_URL; n_tokens/n_oraciones/n_propn/n_targets de la anotacion spaCy | determinista, sin etiquetas |
| n_targets | int64 | len(text), text.split() y regex de P6.PATRON_MENCION/PATRON_HASHTAG/PATRON_URL; n_tokens/n_oraciones/n_propn/n_targets de la anotacion spaCy | determinista, sin etiquetas |
| n_colectivos | int64 | regla estructural de P2.es_colectivo_estructural y P2.es_segunda_persona/P2.es_imperativo replicadas sobre la anotacion serializada; inventarios fijos de P2.NEGACION/CUANTIFICADORES_UNIVERSALES y constantes del script 06 | determinista, sin etiquetas |
| tiene_colectivo | int64 | regla estructural de P2.es_colectivo_estructural y P2.es_segunda_persona/P2.es_imperativo replicadas sobre la anotacion serializada; inventarios fijos de P2.NEGACION/CUANTIFICADORES_UNIVERSALES y constantes del script 06 | determinista, sin etiquetas |
| col_densidad | float64 | regla estructural de P2.es_colectivo_estructural y P2.es_segunda_persona/P2.es_imperativo replicadas sobre la anotacion serializada; inventarios fijos de P2.NEGACION/CUANTIFICADORES_UNIVERSALES y constantes del script 06 | determinista, sin etiquetas |
| n_cuant_universal | int64 | regla estructural de P2.es_colectivo_estructural y P2.es_segunda_persona/P2.es_imperativo replicadas sobre la anotacion serializada; inventarios fijos de P2.NEGACION/CUANTIFICADORES_UNIVERSALES y constantes del script 06 | determinista, sin etiquetas |
| n_plurales | int64 | regla estructural de P2.es_colectivo_estructural y P2.es_segunda_persona/P2.es_imperativo replicadas sobre la anotacion serializada; inventarios fijos de P2.NEGACION/CUANTIFICADORES_UNIVERSALES y constantes del script 06 | determinista, sin etiquetas |
| plural_densidad | float64 | regla estructural de P2.es_colectivo_estructural y P2.es_segunda_persona/P2.es_imperativo replicadas sobre la anotacion serializada; inventarios fijos de P2.NEGACION/CUANTIFICADORES_UNIVERSALES y constantes del script 06 | determinista, sin etiquetas |
| plural_generico | int64 | regla estructural de P2.es_colectivo_estructural y P2.es_segunda_persona/P2.es_imperativo replicadas sobre la anotacion serializada; inventarios fijos de P2.NEGACION/CUANTIFICADORES_UNIVERSALES y constantes del script 06 | determinista, sin etiquetas |
| n_segunda | int64 | regla estructural de P2.es_colectivo_estructural y P2.es_segunda_persona/P2.es_imperativo replicadas sobre la anotacion serializada; inventarios fijos de P2.NEGACION/CUANTIFICADORES_UNIVERSALES y constantes del script 06 | determinista, sin etiquetas |
| tiene_segunda_persona | int64 | regla estructural de P2.es_colectivo_estructural y P2.es_segunda_persona/P2.es_imperativo replicadas sobre la anotacion serializada; inventarios fijos de P2.NEGACION/CUANTIFICADORES_UNIVERSALES y constantes del script 06 | determinista, sin etiquetas |
| n_imperativos | int64 | regla estructural de P2.es_colectivo_estructural y P2.es_segunda_persona/P2.es_imperativo replicadas sobre la anotacion serializada; inventarios fijos de P2.NEGACION/CUANTIFICADORES_UNIVERSALES y constantes del script 06 | determinista, sin etiquetas |
| col_segunda_persona | int64 | regla estructural de P2.es_colectivo_estructural y P2.es_segunda_persona/P2.es_imperativo replicadas sobre la anotacion serializada; inventarios fijos de P2.NEGACION/CUANTIFICADORES_UNIVERSALES y constantes del script 06 | determinista, sin etiquetas |
| n_tramos_cita | int64 | P3.tramos_comillas, P3.PATRON_RT, P3.MARCADORES_ATRIBUCION y P2.NEGACION sobre la anotacion serializada | determinista, sin etiquetas |
| tiene_cita | int64 | P3.tramos_comillas, P3.PATRON_RT, P3.MARCADORES_ATRIBUCION y P2.NEGACION sobre la anotacion serializada | determinista, sin etiquetas |
| n_tokens_en_cita | int64 | P3.tramos_comillas, P3.PATRON_RT, P3.MARCADORES_ATRIBUCION y P2.NEGACION sobre la anotacion serializada | determinista, sin etiquetas |
| frac_tokens_cita | float64 | P3.tramos_comillas, P3.PATRON_RT, P3.MARCADORES_ATRIBUCION y P2.NEGACION sobre la anotacion serializada | determinista, sin etiquetas |
| comillas_dobles | int64 | P3.tramos_comillas, P3.PATRON_RT, P3.MARCADORES_ATRIBUCION y P2.NEGACION sobre la anotacion serializada | determinista, sin etiquetas |
| comillas_simples | int64 | P3.tramos_comillas, P3.PATRON_RT, P3.MARCADORES_ATRIBUCION y P2.NEGACION sobre la anotacion serializada | determinista, sin etiquetas |
| comillas_angulares | int64 | P3.tramos_comillas, P3.PATRON_RT, P3.MARCADORES_ATRIBUCION y P2.NEGACION sobre la anotacion serializada | determinista, sin etiquetas |
| rt_presente | int64 | P3.tramos_comillas, P3.PATRON_RT, P3.MARCADORES_ATRIBUCION y P2.NEGACION sobre la anotacion serializada | determinista, sin etiquetas |
| n_marcadores_atribucion | int64 | P3.tramos_comillas, P3.PATRON_RT, P3.MARCADORES_ATRIBUCION y P2.NEGACION sobre la anotacion serializada | determinista, sin etiquetas |
| atribucion_presente | int64 | P3.tramos_comillas, P3.PATRON_RT, P3.MARCADORES_ATRIBUCION y P2.NEGACION sobre la anotacion serializada | determinista, sin etiquetas |
| n_negaciones | int64 | P3.tramos_comillas, P3.PATRON_RT, P3.MARCADORES_ATRIBUCION y P2.NEGACION sobre la anotacion serializada | determinista, sin etiquetas |
| tiene_negacion | int64 | P3.tramos_comillas, P3.PATRON_RT, P3.MARCADORES_ATRIBUCION y P2.NEGACION sobre la anotacion serializada | determinista, sin etiquetas |
| n_neg_polaridad | int64 | P3.tramos_comillas, P3.PATRON_RT, P3.MARCADORES_ATRIBUCION y P2.NEGACION sobre la anotacion serializada | determinista, sin etiquetas |
| n_morph_plural | int64 | rasgos Number/Gender/Tense de la anotacion spaCy serializada | determinista, sin etiquetas |
| n_morph_femenino | int64 | rasgos Number/Gender/Tense de la anotacion spaCy serializada | determinista, sin etiquetas |
| n_morph_pasado | int64 | rasgos Number/Gender/Tense de la anotacion spaCy serializada | determinista, sin etiquetas |
| tiene_repeticion | int64 | P5.PATRON_REPETICIONES, P5.PATRON_PUNTUACION_INTERNA, PATRON_LEET_PROXY (proxy de la tabla leet de P5), P6.PATRON_ELONGACION, P6.PATRON_RISAS y P6.PATRON_MAYUSCULAS_SOSTENIDAS | determinista, sin etiquetas |
| n_repeticiones | int64 | P5.PATRON_REPETICIONES, P5.PATRON_PUNTUACION_INTERNA, PATRON_LEET_PROXY (proxy de la tabla leet de P5), P6.PATRON_ELONGACION, P6.PATRON_RISAS y P6.PATRON_MAYUSCULAS_SOSTENIDAS | determinista, sin etiquetas |
| tiene_puntuacion_interna | int64 | P5.PATRON_REPETICIONES, P5.PATRON_PUNTUACION_INTERNA, PATRON_LEET_PROXY (proxy de la tabla leet de P5), P6.PATRON_ELONGACION, P6.PATRON_RISAS y P6.PATRON_MAYUSCULAS_SOSTENIDAS | determinista, sin etiquetas |
| n_puntuacion_interna | int64 | P5.PATRON_REPETICIONES, P5.PATRON_PUNTUACION_INTERNA, PATRON_LEET_PROXY (proxy de la tabla leet de P5), P6.PATRON_ELONGACION, P6.PATRON_RISAS y P6.PATRON_MAYUSCULAS_SOSTENIDAS | determinista, sin etiquetas |
| tiene_leet | int64 | P5.PATRON_REPETICIONES, P5.PATRON_PUNTUACION_INTERNA, PATRON_LEET_PROXY (proxy de la tabla leet de P5), P6.PATRON_ELONGACION, P6.PATRON_RISAS y P6.PATRON_MAYUSCULAS_SOSTENIDAS | determinista, sin etiquetas |
| n_caracteres_leet | int64 | P5.PATRON_REPETICIONES, P5.PATRON_PUNTUACION_INTERNA, PATRON_LEET_PROXY (proxy de la tabla leet de P5), P6.PATRON_ELONGACION, P6.PATRON_RISAS y P6.PATRON_MAYUSCULAS_SOSTENIDAS | determinista, sin etiquetas |
| tiene_elongacion | int64 | P5.PATRON_REPETICIONES, P5.PATRON_PUNTUACION_INTERNA, PATRON_LEET_PROXY (proxy de la tabla leet de P5), P6.PATRON_ELONGACION, P6.PATRON_RISAS y P6.PATRON_MAYUSCULAS_SOSTENIDAS | determinista, sin etiquetas |
| n_elongaciones | int64 | P5.PATRON_REPETICIONES, P5.PATRON_PUNTUACION_INTERNA, PATRON_LEET_PROXY (proxy de la tabla leet de P5), P6.PATRON_ELONGACION, P6.PATRON_RISAS y P6.PATRON_MAYUSCULAS_SOSTENIDAS | determinista, sin etiquetas |
| tiene_risas | int64 | P5.PATRON_REPETICIONES, P5.PATRON_PUNTUACION_INTERNA, PATRON_LEET_PROXY (proxy de la tabla leet de P5), P6.PATRON_ELONGACION, P6.PATRON_RISAS y P6.PATRON_MAYUSCULAS_SOSTENIDAS | determinista, sin etiquetas |
| n_risas | int64 | P5.PATRON_REPETICIONES, P5.PATRON_PUNTUACION_INTERNA, PATRON_LEET_PROXY (proxy de la tabla leet de P5), P6.PATRON_ELONGACION, P6.PATRON_RISAS y P6.PATRON_MAYUSCULAS_SOSTENIDAS | determinista, sin etiquetas |
| mayusculas_sostenidas | int64 | P5.PATRON_REPETICIONES, P5.PATRON_PUNTUACION_INTERNA, PATRON_LEET_PROXY (proxy de la tabla leet de P5), P6.PATRON_ELONGACION, P6.PATRON_RISAS y P6.PATRON_MAYUSCULAS_SOSTENIDAS | determinista, sin etiquetas |
| tiene_mayusculas_sostenidas | int64 | P5.PATRON_REPETICIONES, P5.PATRON_PUNTUACION_INTERNA, PATRON_LEET_PROXY (proxy de la tabla leet de P5), P6.PATRON_ELONGACION, P6.PATRON_RISAS y P6.PATRON_MAYUSCULAS_SOSTENIDAS | determinista, sin etiquetas |
| tiene_colectivo_regex | int64 | regla estructural de P2.es_colectivo_estructural y P2.es_segunda_persona/P2.es_imperativo replicadas sobre la anotacion serializada; inventarios fijos de P2.NEGACION/CUANTIFICADORES_UNIVERSALES y constantes del script 06 | determinista, sin etiquetas |
| tiene_segunda_persona_regex | int64 | regla estructural de P2.es_colectivo_estructural y P2.es_segunda_persona/P2.es_imperativo replicadas sobre la anotacion serializada; inventarios fijos de P2.NEGACION/CUANTIFICADORES_UNIVERSALES y constantes del script 06 | determinista, sin etiquetas |
| pos_ADJ_n | int64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| pos_ADJ_frac | float64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| pos_ADP_n | int64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| pos_ADP_frac | float64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| pos_ADV_n | int64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| pos_ADV_frac | float64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| pos_AUX_n | int64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| pos_AUX_frac | float64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| pos_CCONJ_n | int64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| pos_CCONJ_frac | float64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| pos_DET_n | int64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| pos_DET_frac | float64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| pos_INTJ_n | int64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| pos_INTJ_frac | float64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| pos_NOUN_n | int64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| pos_NOUN_frac | float64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| pos_NUM_n | int64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| pos_NUM_frac | float64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| pos_PART_n | int64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| pos_PART_frac | float64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| pos_PRON_n | int64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| pos_PRON_frac | float64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| pos_PROPN_n | int64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| pos_PROPN_frac | float64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| pos_PUNCT_n | int64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| pos_PUNCT_frac | float64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| pos_SCONJ_n | int64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| pos_SCONJ_frac | float64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| pos_SPACE_n | int64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| pos_SPACE_frac | float64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| pos_SYM_n | int64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| pos_SYM_frac | float64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| pos_VERB_n | int64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| pos_VERB_frac | float64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| pos_X_n | int64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| pos_X_frac | float64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| dep_ROOT_n | int64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| dep_acl_n | int64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| dep_advcl_n | int64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| dep_advmod_n | int64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| dep_amod_n | int64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| dep_appos_n | int64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| dep_aux_n | int64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| dep_case_n | int64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| dep_cc_n | int64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| dep_ccomp_n | int64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| dep_compound_n | int64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| dep_conj_n | int64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| dep_cop_n | int64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| dep_csubj_n | int64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| dep_dep_n | int64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| dep_det_n | int64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| dep_expl:impers_n | int64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| dep_expl:pass_n | int64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| dep_expl:pv_n | int64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| dep_fixed_n | int64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| dep_flat_n | int64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| dep_iobj_n | int64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| dep_mark_n | int64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| dep_nmod_n | int64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| dep_nsubj_n | int64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| dep_nummod_n | int64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| dep_obj_n | int64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| dep_obl_n | int64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| dep_parataxis_n | int64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| dep_punct_n | int64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |
| dep_xcomp_n | int64 | anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER) | determinista, sin etiquetas |

## Verificaciones ejecutadas

- El dataset solo tiene `id` y `particion` como metadatos: no hay ninguna etiqueta HS/TR/AG.
- Las filas fuera del desarrollo (holdout y test) se anotan con spaCy con los mismos parametros
  de P3 (`es_core_news_md`, batch 64, sin NER ni vectores); la anotacion se cachea en
  `features/v2_audit_annotations_extra.json.gz`. Es determinista y no usa etiquetas.
- La regla estructural de colectivo, el detector de segunda persona y el de imperativo se
  replicaron sobre la anotacion serializada y se compararon con las funciones de P2 en los 1483
  documentos del desarrollo (resultado en la seccion de discrepancias).
- Los tamanos de los lexicos de insulto inducidos por pliegue se contrastan con
  `reports/targeting_v2.json` (P2): ver tabla siguiente.

## Excluido y por que (honestidad > cobertura)

- **Lexicos inducidos** (insulto, adjetivos negativos, afecto) y **target encoding**: dependen de
  las etiquetas del pliegue de entrenamiento (fold-local), por lo que NO son deterministas ni
  exportables a un parquet unico. Se usan solo en las auditorias sobre el desarrollo y estan en
  el catalogo.
- **Distancia target-insulto y pares colectivo-insulto**: requieren el lexico de insulto inducido;
  existen solo como features fold-locales de E2-05/E2-06, no en este dataset.
- **Canales sparse (TF-IDF word/char/POS/dep/morph/NB-SVM)**: vocabulario e IDF se ajustan por
  pliegue (fold-local); no se consolidan como parquet.
- **Emocion/afecto** (E2-09) y **espacios por etiqueta** (E2-21): fold-locales por lexicos
  inducidos; se catalogan, no se consolidan.
- **Umbrales y scores de modelos**: no son features; viven en los stores OOF.

## Catalogo de features (FASE 55/88)

`reports/feature_catalog_v2.csv`: 225 features. Cobertura por familia: cita_negacion=21, dependencia_etiquetas=71, emocion=71, entidades=12, nbsvm=3, ortografia=5, sintaxis=3, targeting=37, texto_base=2.

## Lexicos de insulto fold-local usados en las auditorias

| pliegue | n_train | n_validacion | insulto | adjetivos negativos | colectivos |
| --- | --- | --- | --- | --- | --- |
| 1 | 1186 | 297 | 37 | 16 | 43 |
| 2 | 1186 | 297 | 36 | 13 | 42 |
| 3 | 1187 | 296 | 33 | 11 | 44 |
| 4 | 1187 | 296 | 36 | 12 | 42 |
| 5 | 1186 | 297 | 36 | 13 | 42 |

Contraste con `reports/targeting_v2.json` (diagnostico_por_pliegue): los tamanos deben coincidir;
cualquier diferencia se registra abajo como discrepancia.

## Discrepancias detectadas

- Ninguna: todos los canales OOF reproducen su F1 documentado con tolerancia 0.001 y los detectores replicados coinciden con P2.
