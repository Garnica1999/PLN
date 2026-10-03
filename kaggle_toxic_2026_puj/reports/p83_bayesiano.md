# P83: modelo bayesiano disperso, doble objetivo

V03 remains the best verified leaderboard result.

- Experimento: P83
- Script: experiments/p83_01_bayesiano.py
- Modo: completo
- Marca de corrida: completo_a7eac4
- Pre-registro: docs/p83_preregistro_bayesiano.md
- sha256 del pre-registro: 7f376fa644c8ee4c696d8f47d6fe79ace7560d96af52235c5c14a20cd9d9ad36

## Formulacion y limites, escritos antes de leer los numeros

- LA_PREGUNTA_UNO_NO_COMPITE_POR_F1_Y_NO_ACE_POR_ELIMINADO_EL_ORDENAMIENTO_POR_F1
- LA_PREGUNTA_DOS_NO_BUSCA_SUPERAR_A_V03_SU_CRITERIO_ES_TENER_UN_PERFIL_DE_ERRORDISTINTO_Y_MEDIBLE
- CON_1483_FILAS_Y_CELDAS_RARISIMAS_EL_PAPEL_ESPERADO_DEL_BAYESIANO_ES_MIEMBRO_DIVERSONO_CAMPEON
- LA_POSTERIOR_DE_M1_ES_APROXIMADA_Y_NO_ES_LA_POSTERIOR_EXACTA
- EL_NUMERO_DE_PARAMETROS_EFECTIVOS_QUE_PRODUCE_ESTE_METODO_ES_UN_PARAMETRO_DEL_METODOY_ESTA_DETERMINADO_POR_LA_ESCALA_DE_LA_PRIOR_ALPHA_0_Y_POR_LA_VARIANZA_POSTERIORNO_ES_UN_NUMERO_DE_MUESTRAS_EFECTIVAS_DEL_CONJUNTO_DE_DATOS
- NO_SE_AFIRMA_QUE_WORD2VEC_SEA_MEJOR_QUE_V03
- NO_SE_AFIRMA_QUE_EL_BAYESIANO_SUPERE_A_V03
- NO_SON_INDEPENDIENTES
- LA_SELECCION_POR_VALOR_ABSOLUTO_PUEDE_FAVORECER_A_LAS_COLUMNAS_LARGASPOR_ESO_UN_BAYESIANO_QUE_DECLARA_MAS_VARIABLES_ACTIVAS_NO_HA_GANADO_POR_GANARSINO_QUE_HA_GANADO_POR_EL_CRITERIO_DE_SELECCION

## Lectura de la pregunta 1: incertidumbre y calibracion

- Lectura de la regla de 1.5: **NEGATIVO**
- Ambiguedad declarada: NO_SON_INDEPENDIENTES
- Los dos intervalos no son independientes: NO_SON_INDEPENDIENTES
- El F1 no aparece en esta regla, porque la pregunta 1 no compite por F1.
- UN_F1_POR_DEBAJO_DE_V03_NO_REFUTA_NADA_SOBRE_LA_PREGUNTA_1PORQUE_LA_PREGUNTA_1_NO_COMPITE_POR_F1

## Lectura de la pregunta 2: complementariedad de errores

- EL_RESCATE_Y_EL_DANO_SON_UN_RECUENTO_DE_CELDAS_Y_NO_SON_COMPARABLES_CON_EL_CORRECTED_CVNI_CON_NINGUN_MACRO_F1
- EL_CORRECTED_CV_DEL_ENSAMBLAJE_SOBRE_EL_OOF_NO_ES_UNA_MEJORA_DE_V03PORQUE_EL_OOF_YA_SE_USO_PARA_ELEGIR_EL_ALPHA_Y_PARA_CUALQUIER_OTRA_ELECCIONLUEGO_ES_UN_NUMERO_OPTIMISTA_Y_NO_PUBLICABLE_COMO_RESULTADO
- El F1 no aparece en esta regla, porque la pregunta 2 no busca superar a V03.
- UN_F1_POR_DEBAJO_DE_V03_NO_REFUTA_NADA_SOBRE_LA_PREGUNTA_2PORQUE_LA_PREGUNTA_2_NO_BUSCA_SUPERAR_A_V03

## F1 del miembro, cifra descriptiva

- Etiqueta: DESCRIPTIVA_Y_SIN_CUALQUIER_LECTURA_INFERENCIAL
- f1_media_columnas: 0.7155415037123107
- desviacion_entre_semillas: 0.004391366887091081
- intervalo: [0.7068499033519289, 0.7242331040726926]
- n_combinaciones: 25
- UN_F1_POR_DEBAJO_DE_V03_NO_REFUTA_NADA_SOBRE_LA_PREGUNTA_1PORQUE_LA_PREGUNTA_1_NO_COMPITE_POR_F1
- UN_F1_POR_DEBAJO_DE_V03_NO_REFUTA_NADA_SOBRE_LA_PREGUNTA_2PORQUE_LA_PREGUNTA_2_NO_BUSCA_SUPERAR_A_V03

## Cifras de la pregunta 1

| magnitud | M1 | M2 | liston v03_cal |
|---|---|---|---|
| log-loss macro recortado | 0.4432757507690348 | 0.45992445512665364 | 0.39100913635320794 |
| log-loss macro sin recortar | 0.4432757507690348 | 0.45992445512665364 | None |
| Brier macro | 0.14167180615398664 | 0.14880603089223432 | 0.12245432782248612 |
| ECE macro | 0.031658918536293115 | 0.032623222871789544 | 0.0279968411759509 |

## Cifras de la pregunta 2

| alpha | corrected_cv | delta | rescate | dano | neto |
|---|---|---|---|---|---|
| 0.05 | 0.7271627480532711 | -0.05004870446330101 | 185 | 299 | -114 |
| 0.10 | 0.7331970644707427 | -0.044014388045829356 | 181 | 278 | -97 |
| 0.15 | 0.7345389273646016 | -0.04267252515197051 | 172 | 264 | -92 |

## Veredictos de las veinte afirmaciones

| id | bloque | estado | stop | que hace |
|---|---|---|---|---|
| A0 | 0 | CONFIRMADO | si | integridad de los artefactos congelados, sha256 tamano y mtime antes y despues |
| A1 | 1 | CONFIRMADO | si | solo se leen las fuentes declaradas, cero test y cero holdout |
| A10 | 10 | CONFIRMADO | si | cobertura y anchura, y el control degenerado de V03 |
| A11 | 11 | CONFIRMADO | si | la regla de lectura de 1.5 aplicada, sin reinterpretacion |
| A12 | 12 | CONFIRMADO | no | las cuatro medidas de complementariedad contra V03 |
| A13 | 13 | CONFIRMADO | no | el rescate y el dano, sobre OOF, con los tres alphas |
| A14 | 14 | CONFIRMADO | si | el calibrador se ajusta en internal, verificado por contadores |
| A15 | 15 | CONFIRMADO | no | el limite de 5.5 verificado: anclaje externo presente o ausente |
| A16 | 16 | CONFIRMADO | no | la rejilla de alpha_0 y la sensibilidad de umbral, publicadas |
| A17 | 17 | CONFIRMADO | no | dimension seleccionada por pliegue, y la salvedad de sesgo |
| A18 | 18 | CONFIRMADO | si | las cadenas de formulacion, de las dos preguntas y del proyecto |
| A19 | 19 | CONFIRMADO | si | el F1 del miembro se publica como descriptiva, sin lectura inferencial |
| A2 | 2 | CONFIRMADO | si | umbrales de V03 intactos, V03 no reentrenado, cero reoptimizaciones |
| A3 | 3 | CONFIRMADO | si | el OOF de referencia es el canonico, coincide con el published |
| A4 | 4 | CONFIRMADO | si | la seleccion de variables se hace solo sobre el inner_train |
| A5 | 5 | CONFIRMADO | si | la posterior de M1 esta aproximada y el metodo se declara |
| A6 | 6 | CONFIRMADO | si | el numero efectivo es un parametro del metodo y se mueve con alpha_0 |
| A7 | 7 | CONFIRMADO | si | M2 es exacto y coincide con un segundo camino de calculo |
| A8 | 8 | CONFIRMADO | no | las seis magnitudes de la pregunta 1 medidas y publicadas |
| A9 | 9 | CONFIRMADO | no | la curva de calibracion completa, con bins poblados |

## Aislamiento de afirmaciones

- ERROR no es ni CONFIRMADO ni REFUTADO ni NO_EVALUABLE; la corrida continua con la siguiente y una Parada con STOP se propaga.
- Errores: 0
- Avisos: 0
- Escrituras fallidas: 0

## Resto del detalle

```json
{
  "A0": {
    "antes": {
      "oof": {
        "mtime_ns": 1790698435954799478,
        "ruta": "features/oof_v03_final.npz",
        "sha256": "559265816ce4f6312c87f008f90324f0eec70a74d1e6247364c162793338feab",
        "tamano": 143796
      },
      "v03": {
        "mtime_ns": 1790698435936799215,
        "ruta": "experiments/baseline_v03_frozen/pipeline_v03.json",
        "sha256": "89ee399ae849296f7a9a76c21d49e15e616a202b6ef8a5b1d87afbf8c528a90b",
        "tamano": 2222
      },
      "w2v": {
        "mtime_ns": 1790744455825458351,
        "ruta": "modelos/twitter_domain/w2v_twitter_100d.npz",
        "sha256": "302c994d13b75f8f5fbd440eea2c4f4113997e2581524021923b92f8244d705a",
        "tamano": 29478868
      }
    },
    "comprobaciones": {
      "el_oof_no_cambio": true,
      "el_v03_no_cambio": true,
      "el_word2vec_esta": true,
      "el_word2vec_no_cambio": true,
      "el_word2vec_tiene_el_sha256_declarado": true
    },
    "despues": {
      "oof": {
        "mtime_ns": 1790698435954799478,
        "ruta": "features/oof_v03_final.npz",
        "sha256": "559265816ce4f6312c87f008f90324f0eec70a74d1e6247364c162793338feab",
        "tamano": 143796
      },
      "v03": {
        "mtime_ns": 1790698435936799215,
        "ruta": "experiments/baseline_v03_frozen/pipeline_v03.json",
        "sha256": "89ee399ae849296f7a9a76c21d49e15e616a202b6ef8a5b1d87afbf8c528a90b",
        "tamano": 2222
      },
      "w2v": {
        "mtime_ns": 1790744455825458351,
        "ruta": "modelos/twitter_domain/w2v_twitter_100d.npz",
        "sha256": "302c994d13b75f8f5fbd440eea2c4f4113997e2581524021923b92f8244d705a",
        "tamano": 29478868
      }
    },
    "motivo_de_parada": "cualquier artefacto congelado difiere, y no se registra ningun otro numero",
    "sha256_declarado_del_word2vec": "302c994d13b75f8f5fbd440eea2c4f4113997e2581524021923b92f8244d705a"
  },
  "A1": {
    "comprobaciones": {
      "el_pool_de_desarrollo_es_el_esperado": true,
      "el_presupuesto_de_evaluaciones_no_se_ha_tocado": true,
      "las_evaluaciones_del_holdout_son_las_esperadas": true,
      "las_filas_de_holdout_son_las_esperadas": true,
      "las_filas_de_test_son_las_esperadas": true
    },
    "evaluaciones_del_holdout_esperadas": 0,
    "evaluaciones_del_holdout_gastadas": 0,
    "evaluaciones_del_holdout_gastadas_por_p83": 0,
    "filas_de_desarrollo_esperadas": 1483,
    "filas_de_desarrollo_leidas": 1483,
    "filas_de_holdout_esperadas": 0,
    "filas_de_holdout_leidas": 0,
    "filas_de_test_esperadas": 0,
    "filas_de_test_leidas": 0,
    "motivo_de_parada": "se ha leido test o el holdout, o se ha gastado una evaluacion del holdout",
    "presupuesto_de_evaluaciones_del_holdout_del_proyecto": 5
  },
  "A10": {
    "comprobaciones": {
      "M1_publica_anchura_media_normalizada": true,
      "M1_publica_cobertura": true,
      "M1_publica_cobertura_menos_pesada_por_anchura": true,
      "M2_publica_anchura_media_normalizada": true,
      "M2_publica_cobertura": true,
      "M2_publica_cobertura_menos_pesada_por_anchura": true,
      "el_control_v03_publica_anchura_media_normalizada": true,
      "el_control_v03_publica_cobertura": true,
      "el_control_v03_publica_cobertura_menos_pesada_por_anchura": true,
      "la_cobertura_del_intervalo_completo_es_1": true,
      "los_dos_brazos_estan_medidos": true
    },
    "control_intervalo_completo_0_1": 1.0,
    "control_v03_anchura_cero": {
      "anchura_media_normalizada": 0.0,
      "cobertura": 0.0,
      "cobertura_menos_pesada_por_anchura": 0.0,
      "peso_de_la_penalizacion": 0.5
    },
    "intervalo_bayesiano_por_brazo": {
      "M1": {
        "anchura_media_normalizada": 0.5873079665980738,
        "cobertura": 0.0,
        "cobertura_menos_pesada_por_anchura": -0.2936539832990369,
        "peso_de_la_penalizacion": 0.5
      },
      "M2": {
        "anchura_media_normalizada": 9.728526470710113e-05,
        "cobertura": 0.0,
        "cobertura_menos_pesada_por_anchura": -4.8642632353550565e-05,
        "peso_de_la_penalizacion": 0.5
      }
    },
    "motivo_de_parada": "la cobertura del intervalo completo no da 1.0 y el intervalo no se construye bien",
    "por_que_ambos_extremos": "un intervalo estrecho que no cubre pierde contra el control de V03 y un intervalo [0, 1] tiene cobertura 1 y no dice nada, luego los dos extremos acorralan la metrica",
    "que_no_se_autoriza": "decir que el modelo sabe mas, porque la cobertura no mide saber"
  },
  "A11": {
    "ambiguedad_declarada": "NO_SON_INDEPENDIENTES",
    "comprobaciones": {
      "el_campo_lectura_existe": true,
      "el_caso_ambiguo_no_publica_afirmacion": true,
      "el_valor_coincide_con_la_regla": true,
      "el_valor_es_uno_de_las_cuatro": true,
      "la_ambiguedad_esta_declarada": true
    },
    "lectura_de_la_regla": {
      "ambiguedad_declarada": "NO_SON_INDEPENDIENTES",
      "contiene_el_margen_por_pliegues": false,
      "contiene_el_margen_por_semilla": false,
      "extremo_desfavorable_por_pliegues": true,
      "extremo_desfavorable_por_semilla": true,
      "extremo_favorable_por_pliegues": false,
      "extremo_favorable_por_semilla": false,
      "fragilidad_declarada": "el numero de pliegues es cinco y un intervalo de cinco observaciones es fragil por construccion",
      "intervalo_por_pliegues": {
        "inferior": 0.03919132736697088,
        "n": 5,
        "superior": 0.06816708206631075
      },
      "intervalo_por_semillas": {
        "inferior": 0.05011582272924193,
        "n": 5,
        "superior": 0.05724258670403971
      },
      "lectura": "NEGATIVO",
      "margen": 0.005,
      "n_de_pliegues": 5,
      "por_que_no_son_independientes": "los cinco pliegues particionan las mismas 1483 filas y cada una se repite en cinco semillas, luego son veinticinco unidades altamente correlacionadas sobre un unico conjunto y no se puede declarar un nivel de confianza del 95 por ciento sobre ninguno de los dos intervalos"
    },
    "lectura_en_el_informe": "NEGATIVO",
    "motivo_de_parada": "el valor no coincide con la regla, o el caso ambiguo publica una afirmacion"
  },
  "A12": {
    "comprobaciones": {
      "M1_tiene_acuerdo_con_v03": true,
      "M1_tiene_auc_del_error_de_v03": true,
      "M1_tiene_diferencias_por_etiqueta": true,
      "M1_tiene_matriz_de_interseccion": true,
      "M2_tiene_acuerdo_con_v03": true,
      "M2_tiene_auc_del_error_de_v03": true,
      "M2_tiene_diferencias_por_etiqueta": true,
      "M2_tiene_matriz_de_interseccion": true,
      "los_dos_brazos_estan_medidos": true,
      "ninguna_medida_falta": true
    },
    "faltantes": [],
    "medidas": {
      "M1": {
        "acuerdo_con_v03": {
          "global": 0.8846931894807822,
          "por_etiqueta": {
            "AG": 0.8712070128118679,
            "HS": 0.8610923803101821,
            "TR": 0.9217801753202967
          }
        },
        "auc_del_error_de_v03": {
          "global": 0.6008081309726052,
          "por_etiqueta": {
            "AG": 0.5562590177871174,
            "HS": 0.548900133415769,
            "TR": 0.6706207585199431
          }
        },
        "diferencias_por_etiqueta": {
          "AG": {
            "acuerdo_con_v03": 0.8712070128118679,
            "auc_del_error_de_v03": 0.5562590177871174,
            "miembro_acierta_y_v03_falla": 78,
            "v03_acierta_y_miembro_falla": 113
          },
          "HS": {
            "acuerdo_con_v03": 0.8610923803101821,
            "auc_del_error_de_v03": 0.548900133415769,
            "miembro_acierta_y_v03_falla": 77,
            "v03_acierta_y_miembro_falla": 129
          },
          "TR": {
            "acuerdo_con_v03": 0.9217801753202967,
            "auc_del_error_de_v03": 0.6706207585199431,
            "miembro_acierta_y_v03_falla": 37,
            "v03_acierta_y_miembro_falla": 79
          }
        },
        "matriz_de_interseccion": {
          "global": {
            "ambos_aciertan": 3371,
            "ambos_fallan": 565,
            "solo_miembro": 192,
            "solo_v03": 321
          },
          "por_etiqueta": {
            "AG": {
              "miembro_acierta_y_v03_falla": 78,
              "v03_acierta_y_miembro_falla": 113
            },
            "HS": {
              "miembro_acierta_y_v03_falla": 77,
              "v03_acierta_y_miembro_falla": 129
            },
            "TR": {
              "miembro_acierta_y_v03_falla": 37,
              "v03_acierta_y_miembro_falla": 79
            }
          }
        },
        "n_unidades_tweet_etiqueta": 4449,
        "origen": "medido sobre las probabilidades recibidas, no sobre constantes"
      },
      "M2": {
        "acuerdo_con_v03": {
          "global": 0.8608676106990335,
          "por_etiqueta": {
            "AG": 0.851652056641942,
            "HS": 0.8064733648010789,
            "TR": 0.9244774106540796
          }
        },
        "auc_del_error_de_v03": {
          "global": 0.5701216239618383,
          "por_etiqueta": {
            "AG": 0.514524249989523,
            "HS": 0.526832297462931,
            "TR": 0.6448482775199246
          }
        },
        "diferencias_por_etiqueta": {
          "AG": {
            "acuerdo_con_v03": 0.851652056641942,
            "auc_del_error_de_v03": 0.514524249989523,
            "miembro_acierta_y_v03_falla": 76,
            "v03_acierta_y_miembro_falla": 144
          },
          "HS": {
            "acuerdo_con_v03": 0.8064733648010789,
            "auc_del_error_de_v03": 0.526832297462931,
            "miembro_acierta_y_v03_falla": 102,
            "v03_acierta_y_miembro_falla": 185
          },
          "TR": {
            "acuerdo_con_v03": 0.9244774106540796,
            "auc_del_error_de_v03": 0.6448482775199246,
            "miembro_acierta_y_v03_falla": 29,
            "v03_acierta_y_miembro_falla": 83
          }
        },
        "matriz_de_interseccion": {
          "global": {
            "ambos_aciertan": 3280,
            "ambos_fallan": 550,
            "solo_miembro": 207,
            "solo_v03": 412
          },
          "por_etiqueta": {
            "AG": {
              "miembro_acierta_y_v03_falla": 76,
              "v03_acierta_y_miembro_falla": 144
            },
            "HS": {
              "miembro_acierta_y_v03_falla": 102,
              "v03_acierta_y_miembro_falla": 185
            },
            "TR": {
              "miembro_acierta_y_v03_falla": 29,
              "v03_acierta_y_miembro_falla": 83
            }
          }
        },
        "n_unidades_tweet_etiqueta": 4449,
        "origen": "medido sobre las probabilidades recibidas, no sobre constantes"
      }
    },
    "naturaleza": "EXPLORATORIA, y su resultado no puede producir ninguna afirmacion sobre el leaderboard",
    "tolerancia": "seis decimales, enteros exactos para los conteos"
  },
  "A13": {
    "alpha_primario": 0.1,
    "comprobaciones": {
      "el_alpha_primario_esta": true,
      "la_declaracion_de_la_escala_esta": true,
      "la_declaracion_del_corrected_cv_esta": true,
      "la_lectura_diagnostica_esta_publicada": true,
      "los_tres_alphas_estan_completos": true,
      "ninguna_cifra_falta": true
    },
    "configuraciones": {
      "0.05": {
        "aciertos_de_v03": 3692,
        "aciertos_del_ensamblaje": 3578,
        "alpha": 0.05,
        "cambios_de_decision_para_igualar": 114,
        "cambios_de_decision_para_superar": 115,
        "corrected_cv": 0.7271627480532711,
        "corrected_cv_v03": 0.7772114525165721,
        "dano": 299,
        "delta": -0.05004870446330101,
        "lectura": "DIAGNOSTICA_Y_SIN_LECTURA_INFERENCIAL",
        "neto": -114,
        "piso": 3692,
        "pliegues_a_favor": 0,
        "que_no_se_afirma": "EL_RESCATE_Y_EL_DANO_SON_UN_RECUENTO_DE_CELDAS_Y_NO_SON_COMPARABLES_CON_EL_CORRECTED_CVNI_CON_NINGUN_MACRO_F1 | EL_CORRECTED_CV_DEL_ENSAMBLAJE_SOBRE_EL_OOF_NO_ES_UNA_MEJORA_DE_V03PORQUE_EL_OOF_YA_SE_USO_PARA_ELEGIR_EL_ALPHA_Y_PARA_CUALQUIER_OTRA_ELECCIONLUEGO_ES_UN_NUMERO_OPTIMISTA_Y_NO_PUBLICABLE_COMO_RESULTADO",
        "rescate": 185,
        "significado_del_techo_y_del_piso": "techo es el numero de celdas en las que V03 falla, que es el tope del rescate bruto; piso es el numero de celdas en las que V03 acierta, que es el suelo al que puede llevar el dano maximo",
        "techo": 757
      },
      "0.10": {
        "aciertos_de_v03": 3692,
        "aciertos_del_ensamblaje": 3595,
        "alpha": 0.1,
        "cambios_de_decision_para_igualar": 97,
        "cambios_de_decision_para_superar": 98,
        "corrected_cv": 0.7331970644707427,
        "corrected_cv_v03": 0.7772114525165721,
        "dano": 278,
        "delta": -0.044014388045829356,
        "lectura": "DIAGNOSTICA_Y_SIN_LECTURA_INFERENCIAL",
        "neto": -97,
        "piso": 3692,
        "pliegues_a_favor": 0,
        "que_no_se_afirma": "EL_RESCATE_Y_EL_DANO_SON_UN_RECUENTO_DE_CELDAS_Y_NO_SON_COMPARABLES_CON_EL_CORRECTED_CVNI_CON_NINGUN_MACRO_F1 | EL_CORRECTED_CV_DEL_ENSAMBLAJE_SOBRE_EL_OOF_NO_ES_UNA_MEJORA_DE_V03PORQUE_EL_OOF_YA_SE_USO_PARA_ELEGIR_EL_ALPHA_Y_PARA_CUALQUIER_OTRA_ELECCIONLUEGO_ES_UN_NUMERO_OPTIMISTA_Y_NO_PUBLICABLE_COMO_RESULTADO",
        "rescate": 181,
        "significado_del_techo_y_del_piso": "techo es el numero de celdas en las que V03 falla, que es el tope del rescate bruto; piso es el numero de celdas en las que V03 acierta, que es el suelo al que puede llevar el dano maximo",
        "techo": 757
      },
      "0.15": {
        "aciertos_de_v03": 3692,
        "aciertos_del_ensamblaje": 3600,
        "alpha": 0.15,
        "cambios_de_decision_para_igualar": 92,
        "cambios_de_decision_para_superar": 93,
        "corrected_cv": 0.7345389273646016,
        "corrected_cv_v03": 0.7772114525165721,
        "dano": 264,
        "delta": -0.04267252515197051,
        "lectura": "DIAGNOSTICA_Y_SIN_LECTURA_INFERENCIAL",
        "neto": -92,
        "piso": 3692,
        "pliegues_a_favor": 0,
        "que_no_se_afirma": "EL_RESCATE_Y_EL_DANO_SON_UN_RECUENTO_DE_CELDAS_Y_NO_SON_COMPARABLES_CON_EL_CORRECTED_CVNI_CON_NINGUN_MACRO_F1 | EL_CORRECTED_CV_DEL_ENSAMBLAJE_SOBRE_EL_OOF_NO_ES_UNA_MEJORA_DE_V03PORQUE_EL_OOF_YA_SE_USO_PARA_ELEGIR_EL_ALPHA_Y_PARA_CUALQUIER_OTRA_ELECCIONLUEGO_ES_UN_NUMERO_OPTIMISTA_Y_NO_PUBLICABLE_COMO_RESULTADO",
        "rescate": 172,
        "significado_del_techo_y_del_piso": "techo es el numero de celdas en las que V03 falla, que es el tope del rescate bruto; piso es el numero de celdas en las que V03 acierta, que es el suelo al que puede llevar el dano maximo",
        "techo": 757
      }
    },
    "declaraciones": {
      "corrected_cv_optimista": true,
      "escala": true,
      "lectura_diagnostica": "DIAGNOSTICA_Y_SIN_LECTURA_INFERENCIAL"
    },
    "faltantes": [],
    "naturaleza": "DESCRIPTIVA; el corrected_cv del ensamblaje sobre el OOF es optimista y no es publicable como resultado",
    "tolerancia": "seis decimales para el corrected_cv, enteros exactos para los recuentos"
  },
  "A14": {
    "ajustes_que_tocan_filas_de_evaluacion": 0,
    "ajustes_sin_declarar_filas": 0,
    "ajustes_sobre_el_oof_completo": 0,
    "comprobaciones": {
      "cero_ajustes_fuera_de_lo_permitido": true,
      "cero_ajustes_que_toquen_filas_de_evaluacion": true,
      "cero_ajustes_sobre_el_oof_completo": true,
      "el_alpha_no_usa_filas_del_calibrador": true,
      "el_alpha_y_el_calibrador_no_se_solapan": true,
      "el_calibrador_solo_uso_filas_internas": true,
      "el_numero_de_ajustes_es_el_esperado": true,
      "hubo_filas_de_calibracion": true,
      "hubo_filas_internas": true,
      "la_comprobacion_sabe_fallar_por_mutacion": true,
      "la_decision_del_alpha_solo_uso_filas_internas": true,
      "las_filas_internas_y_externas_no_se_solapan": true,
      "todo_ajuste_declaro_sus_filas": true
    },
    "de_que_corrida_es_el_recuento": "la instantanea se toma DESPUES de la corrida principal de M1 y ANTES de la rejilla de alpha_0 de A6, que vuelve a ejecutar M1 tres veces mas; por eso el recuento es cinco pliegues por cinco semillas y no el producto",
    "instrumentacion": {
      "ajustes_fuera_de_lo_permitido": 0,
      "ajustes_que_tocan_el_oof": 0,
      "alpha_fuera_de_interno": 0,
      "calibrador_fuera_de_interno": 0,
      "filas_comunes_entre_el_alpha_y_el_calibrador": 0,
      "llamadas_al_alpha_sin_declarar_filas": 0
    },
    "interseccion_interna_externa": 1480,
    "motivo_de_parada": "el calibrador se ha ajustado sobre el OOF y la pregunta 1 es circular",
    "mutacion": "se ejecuta una vez ajustando el calibrador sobre el OOF completo y A14 tiene que detectarlo",
    "mutacion_detectada": true,
    "n_ajustes_del_calibrador": 25,
    "n_ajustes_esperados": 25,
    "n_filas_del_inner_calibration": 1480,
    "n_filas_del_outer_validation": 1483,
    "n_filas_internas_de_la_decision_del_alpha": 828,
    "registros_contra_registro": {
      "interseccion_por_registro_del_alpha_con_su_calibrador": [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
      ],
      "n_registros_del_alpha": 25,
      "n_registros_del_calibrador": 25,
      "por_que": "los atributos vivos solo guardan la ultima combinacion; la unica comparacion que la regla 3.5 protege es la de cada registro consigo mismo, y por eso las dos comprobaciones de solapamiento se calculan aqui llamada por llamada",
      "registros_del_alpha_sin_filas_declaradas": 0,
      "registros_del_calibrador_sin_filas_de_evaluacion": 0,
      "solapamiento_por_registro_del_calibrador": [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
      ]
    }
  },
  "A15": {
    "comprobaciones": {
      "el_estado_del_limite_esta_declarado": true,
      "la_comprobacion_se_ejecuto_en_la_corrida": true,
      "la_lista_de_nombres_comprobados_esta_publicada": true
    },
    "consecuencia": "no hay ninguna libreria de inferencia bayesiana en el venv, luego el limite de 5.5 se mantiene y P83 autoriza afirmar sobre el COMPORTAMIENTO del modelo que este proyecto implemento y sobre la COMPARACION entre M1 y M2, y no sobre la correccion de la inferencia variacional frente a una posterior exacta",
    "encontradas": [],
    "estado_del_limite": "SIN_HACER",
    "librerias_comprobadas": [
      "aesara",
      "cmdstanpy",
      "edward",
      "numpyro",
      "pymc",
      "pymc3",
      "pyro",
      "pyro-ppl",
      "pystan",
      "stan",
      "tensorflow_probability",
      "theano"
    ],
    "lo_que_p83_autoriza": "afirmar sobre el comportamiento del modelo que este proyecto implemento y sobre la comparacion entre M1 y M2",
    "lo_que_p83_no_autoriza": "decir que la inferencia variacional de M1 es correcta, ni que su posterior aproximada se acerca a la exacta",
    "presentes": {
      "aesara": false,
      "cmdstanpy": false,
      "edward": false,
      "numpyro": false,
      "pymc": false,
      "pymc3": false,
      "pyro": false,
      "pyro-ppl": false,
      "pystan": false,
      "stan": false,
      "tensorflow_probability": false,
      "theano": false
    }
  },
  "A16": {
    "comprobaciones": {
      "el_texto_del_metodo_no_contamina_el_recuento": true,
      "la_rejilla_de_alpha_0_tiene_tres_valores": true,
      "la_rejilla_de_umbral_tiene_tres_valores": true,
      "las_dos_rejillas_estan_publicadas": true,
      "las_magnitudes_de_alpha_0_estan_medidas": true,
      "las_magnitudes_del_umbral_estan_medidas": true,
      "ningun_valor_se_publica_como_el_mejor": true
    },
    "ningun_valor_elegido": true,
    "por_que_no_se_elige": "elegir el mejor valor de la rejilla sobre el OOF seria seleccion sobre el OOF, que es la contamination que A14 prohibe",
    "rejillas": {
      "alpha_0": {
        "0.50": {
          "brier_macro": 0.1426228452614681,
          "log_loss_macro": 0.4457108871115057,
          "n_activas_total": 6.0,
          "n_efectivos_total": 456.7722281544299,
          "n_filas_medidas": 1483,
          "por_etiqueta": {
            "AG": 446.42991281314335,
            "HS": 443.99171319117136,
            "TR": 479.8950584589751
          }
        },
        "1.00": {
          "brier_macro": 0.14167180615398664,
          "log_loss_macro": 0.4432757507690348,
          "n_activas_total": 13.0,
          "n_efectivos_total": 418.89253397720495,
          "n_filas_medidas": 1483,
          "por_etiqueta": {
            "AG": 407.4401559201983,
            "HS": 402.308485641683,
            "TR": 446.92896036973366
          }
        },
        "2.00": {
          "brier_macro": 0.1399489997350212,
          "log_loss_macro": 0.4377755699139304,
          "n_activas_total": 45.0,
          "n_efectivos_total": 370.1679464134697,
          "n_filas_medidas": 1483,
          "por_etiqueta": {
            "AG": 358.79367229203217,
            "HS": 350.0706270031783,
            "TR": 401.6395399451984
          }
        }
      },
      "texto_efectivos_metodo": "EL_NUMERO_DE_PARAMETROS_EFECTIVOS_QUE_PRODUCE_ESTE_METODO_ES_UN_PARAMETRO_DEL_METODOY_ESTA_DETERMINADO_POR_LA_ESCALA_DE_LA_PRIOR_ALPHA_0_Y_POR_LA_VARIANZA_POSTERIORNO_ES_UN_NUMERO_DE_MUESTRAS_EFECTIVAS_DEL_CONJUNTO_DE_DATOS",
      "umbral": {
        "2.0": {
          "brier_macro": 0.13958415286845396,
          "dimension_maxima": 1124,
          "dimension_media": 852.2,
          "dimension_minima": 611,
          "log_loss_macro": 0.4378698408799649,
          "n_columnas_por_pliegue": {
            "umbral_2.0_pliegue_0": {
              "n_columnas": 1070,
              "n_filas_inner_train": 817,
              "pliegue": 0,
              "umbral": 2.0
            },
            "umbral_2.0_pliegue_1": {
              "n_columnas": 1069,
              "n_filas_inner_train": 838,
              "pliegue": 1,
              "umbral": 2.0
            },
            "umbral_2.0_pliegue_2": {
              "n_columnas": 1124,
              "n_filas_inner_train": 831,
              "pliegue": 2,
              "umbral": 2.0
            },
            "umbral_2.0_pliegue_3": {
              "n_columnas": 1049,
              "n_filas_inner_train": 827,
              "pliegue": 3,
              "umbral": 2.0
            },
            "umbral_2.0_pliegue_4": {
              "n_columnas": 1090,
              "n_filas_inner_train": 828,
              "pliegue": 4,
              "umbral": 2.0
            },
            "umbral_3.0_pliegue_0": {
              "n_columnas": 611,
              "n_filas_inner_train": 817,
              "pliegue": 0,
              "umbral": 3.0
            },
            "umbral_3.0_pliegue_1": {
              "n_columnas": 629,
              "n_filas_inner_train": 838,
              "pliegue": 1,
              "umbral": 3.0
            },
            "umbral_3.0_pliegue_2": {
              "n_columnas": 643,
              "n_filas_inner_train": 831,
              "pliegue": 2,
              "umbral": 3.0
            },
            "umbral_3.0_pliegue_3": {
              "n_columnas": 611,
              "n_filas_inner_train": 827,
              "pliegue": 3,
              "umbral": 3.0
            },
            "umbral_3.0_pliegue_4": {
              "n_columnas": 626,
              "n_filas_inner_train": 828,
              "pliegue": 4,
              "umbral": 3.0
            }
          }
        },
        "3.0": {
          "brier_macro": 0.14167180615398664,
          "dimension_maxima": 643,
          "dimension_media": 624.0,
          "dimension_minima": 611,
          "log_loss_macro": 0.4432757507690348,
          "n_columnas_por_pliegue": {
            "umbral_3.0_pliegue_0": {
              "n_columnas": 611,
              "n_filas_inner_train": 817,
              "pliegue": 0,
              "umbral": 3.0
            },
            "umbral_3.0_pliegue_1": {
              "n_columnas": 629,
              "n_filas_inner_train": 838,
              "pliegue": 1,
              "umbral": 3.0
            },
            "umbral_3.0_pliegue_2": {
              "n_columnas": 643,
              "n_filas_inner_train": 831,
              "pliegue": 2,
              "umbral": 3.0
            },
            "umbral_3.0_pliegue_3": {
              "n_columnas": 611,
              "n_filas_inner_train": 827,
              "pliegue": 3,
              "umbral": 3.0
            },
            "umbral_3.0_pliegue_4": {
              "n_columnas": 626,
              "n_filas_inner_train": 828,
              "pliegue": 4,
              "umbral": 3.0
            }
          }
        },
        "5.0": {
          "brier_macro": 0.15153415152521896,
          "dimension_maxima": 1124,
          "dimension_media": 667.4,
          "dimension_minima": 291,
          "log_loss_macro": 0.4669643255121672,
          "n_columnas_por_pliegue": {
            "umbral_2.0_pliegue_0": {
              "n_columnas": 1070,
              "n_filas_inner_train": 817,
              "pliegue": 0,
              "umbral": 2.0
            },
            "umbral_2.0_pliegue_1": {
              "n_columnas": 1069,
              "n_filas_inner_train": 838,
              "pliegue": 1,
              "umbral": 2.0
            },
            "umbral_2.0_pliegue_2": {
              "n_columnas": 1124,
              "n_filas_inner_train": 831,
              "pliegue": 2,
              "umbral": 2.0
            },
            "umbral_2.0_pliegue_3": {
              "n_columnas": 1049,
              "n_filas_inner_train": 827,
              "pliegue": 3,
              "umbral": 2.0
            },
            "umbral_2.0_pliegue_4": {
              "n_columnas": 1090,
              "n_filas_inner_train": 828,
              "pliegue": 4,
              "umbral": 2.0
            },
            "umbral_3.0_pliegue_0": {
              "n_columnas": 611,
              "n_filas_inner_train": 817,
              "pliegue": 0,
              "umbral": 3.0
            },
            "umbral_3.0_pliegue_1": {
              "n_columnas": 629,
              "n_filas_inner_train": 838,
              "pliegue": 1,
              "umbral": 3.0
            },
            "umbral_3.0_pliegue_2": {
              "n_columnas": 643,
              "n_filas_inner_train": 831,
              "pliegue": 2,
              "umbral": 3.0
            },
            "umbral_3.0_pliegue_3": {
              "n_columnas": 611,
              "n_filas_inner_train": 827,
              "pliegue": 3,
              "umbral": 3.0
            },
            "umbral_3.0_pliegue_4": {
              "n_columnas": 626,
              "n_filas_inner_train": 828,
              "pliegue": 4,
              "umbral": 3.0
            },
            "umbral_5.0_pliegue_0": {
              "n_columnas": 300,
              "n_filas_inner_train": 817,
              "pliegue": 0,
              "umbral": 5.0
            },
            "umbral_5.0_pliegue_1": {
              "n_columnas": 299,
              "n_filas_inner_train": 838,
              "pliegue": 1,
              "umbral": 5.0
            },
            "umbral_5.0_pliegue_2": {
              "n_columnas": 291,
              "n_filas_inner_train": 831,
              "pliegue": 2,
              "umbral": 5.0
            },
            "umbral_5.0_pliegue_3": {
              "n_columnas": 301,
              "n_filas_inner_train": 827,
              "pliegue": 3,
              "umbral": 5.0
            },
            "umbral_5.0_pliegue_4": {
              "n_columnas": 298,
              "n_filas_inner_train": 828,
              "pliegue": 4,
              "umbral": 5.0
            }
          }
        }
      },
      "valores_elegidos": []
    },
    "tolerancia": "seis decimales",
    "valores_elegidos": []
  },
  "A17": {
    "comprobaciones": {
      "hay_dimensiones_publicadas": true,
      "la_comparacion_con_la_l1_esta_publicada": true,
      "la_dimension_es_menor_que_las_filas_o_la_no_determinacion_esta_declarada": true,
      "la_salvedad_de_sesgo_esta_declarada": true
    },
    "declaracion_de_no_determinacion": "LA_DIMENSION_SELECCIONADA_NO_ES_MENOR_QUE_LAS_FILAS_DEL_INNER_TRAIN_EN_ALGUN_PLIEGUE_Luego_LA_REGRESION_NO_ESTA_DETERMINADA_Y_EL_RESULTADO_HAY_QUE_LEERLO_CON_ESA_SALVEDAD",
    "dimension_por_pliegue_y_umbral": {
      "umbral_2.0_pliegue_0": {
        "n_columnas": 1070,
        "n_filas_inner_train": 817,
        "pliegue": 0,
        "umbral": 2.0
      },
      "umbral_2.0_pliegue_1": {
        "n_columnas": 1069,
        "n_filas_inner_train": 838,
        "pliegue": 1,
        "umbral": 2.0
      },
      "umbral_2.0_pliegue_2": {
        "n_columnas": 1124,
        "n_filas_inner_train": 831,
        "pliegue": 2,
        "umbral": 2.0
      },
      "umbral_2.0_pliegue_3": {
        "n_columnas": 1049,
        "n_filas_inner_train": 827,
        "pliegue": 3,
        "umbral": 2.0
      },
      "umbral_2.0_pliegue_4": {
        "n_columnas": 1090,
        "n_filas_inner_train": 828,
        "pliegue": 4,
        "umbral": 2.0
      },
      "umbral_3.0_pliegue_0": {
        "n_columnas": 611,
        "n_filas_inner_train": 817,
        "pliegue": 0,
        "umbral": 3.0
      },
      "umbral_3.0_pliegue_1": {
        "n_columnas": 629,
        "n_filas_inner_train": 838,
        "pliegue": 1,
        "umbral": 3.0
      },
      "umbral_3.0_pliegue_2": {
        "n_columnas": 643,
        "n_filas_inner_train": 831,
        "pliegue": 2,
        "umbral": 3.0
      },
      "umbral_3.0_pliegue_3": {
        "n_columnas": 611,
        "n_filas_inner_train": 827,
        "pliegue": 3,
        "umbral": 3.0
      },
      "umbral_3.0_pliegue_4": {
        "n_columnas": 626,
        "n_filas_inner_train": 828,
        "pliegue": 4,
        "umbral": 3.0
      },
      "umbral_5.0_pliegue_0": {
        "n_columnas": 300,
        "n_filas_inner_train": 817,
        "pliegue": 0,
        "umbral": 5.0
      },
      "umbral_5.0_pliegue_1": {
        "n_columnas": 299,
        "n_filas_inner_train": 838,
        "pliegue": 1,
        "umbral": 5.0
      },
      "umbral_5.0_pliegue_2": {
        "n_columnas": 291,
        "n_filas_inner_train": 831,
        "pliegue": 2,
        "umbral": 5.0
      },
      "umbral_5.0_pliegue_3": {
        "n_columnas": 301,
        "n_filas_inner_train": 827,
        "pliegue": 3,
        "umbral": 5.0
      },
      "umbral_5.0_pliegue_4": {
        "n_columnas": 298,
        "n_filas_inner_train": 828,
        "pliegue": 4,
        "umbral": 5.0
      }
    },
    "filas_del_inner_train": 828,
    "maximo": 1124,
    "media": 667.4,
    "minimo": 291,
    "pliegues_no_determinados": [
      "umbral_2.0_pliegue_0",
      "umbral_2.0_pliegue_1",
      "umbral_2.0_pliegue_2",
      "umbral_2.0_pliegue_3",
      "umbral_2.0_pliegue_4"
    ],
    "referencia_l1_c4": {
      "f1": 0.783423,
      "ic_contiene_el_cero": true,
      "ic_pareado": [
        -0.0081,
        0.0202
      ],
      "origen": "medida en la corrida con un ajuste L1 C igual a 4.0 sobre el inner_train del pliegue 0",
      "variables_dejadas_por_las_c4": 1628
    },
    "regresion_determinada": false,
    "salvedad_de_sesgo": "LA_SELECCION_POR_VALOR_ABSOLUTO_PUEDE_FAVORECER_A_LAS_COLUMNAS_LARGASPOR_ESO_UN_BAYESIANO_QUE_DECLARA_MAS_VARIABLES_ACTIVAS_NO_HA_GANADO_POR_GANARSINO_QUE_HA_GANADO_POR_EL_CRITERIO_DE_SELECCION",
    "umbral_primario": 3.0,
    "umbrales_de_sensibilidad": [
      2.0,
      3.0,
      5.0
    ]
  },
  "A18": {
    "cadenas_presentes": {
      "CON_1483_FILAS_Y_CELDAS_RARISIMAS_EL_PAPEL_ESPERADO_DEL_BAYESIANO_ES_MIEMBRO_DIVERSONO_CAMPEON": true,
      "LA_PREGUNTA_DOS_NO_BUSCA_SUPERAR_A_V03_SU_CRITERIO_ES_TENER_UN_PERFIL_DE_ERRORDISTINTO_Y_MEDIBLE": true,
      "LA_PREGUNTA_UNO_NO_COMPITE_POR_F1_Y_NO_ACE_POR_ELIMINADO_EL_ORDENAMIENTO_POR_F1": true,
      "V03 remains the best verified leaderboard result.": true
    },
    "cadenas_requeridas": [
      "V03 remains the best verified leaderboard result.",
      "LA_PREGUNTA_UNO_NO_COMPITE_POR_F1_Y_NO_ACE_POR_ELIMINADO_EL_ORDENAMIENTO_POR_F1",
      "LA_PREGUNTA_DOS_NO_BUSCA_SUPERAR_A_V03_SU_CRITERIO_ES_TENER_UN_PERFIL_DE_ERRORDISTINTO_Y_MEDIBLE",
      "CON_1483_FILAS_Y_CELDAS_RARISIMAS_EL_PAPEL_ESPERADO_DEL_BAYESIANO_ES_MIEMBRO_DIVERSONO_CAMPEON"
    ],
    "comprobaciones": {
      "la_cadena_presente_0": true,
      "la_cadena_presente_1": true,
      "la_cadena_presente_2": true,
      "la_cadena_presente_3": true,
      "la_variante_ausente_0": true,
      "la_variante_ausente_1": true,
      "la_variante_ausente_2": true,
      "la_variante_ausente_3": true,
      "la_variante_ausente_4": true,
      "la_variante_ausente_5": true
    },
    "motivo_de_parada": "falta una cadena de formulacion o aparece una variante",
    "variantes_encontradas": {
      "V03 is not the best verified leaderboard result": false,
      "V03 remains the best verified result": false,
      "Word2Vec es mejor que V03": false,
      "Word2Vec improves sobre V03": false,
      "el bayesiano mejora a V03": false,
      "el bayesiano supera a V03": false
    },
    "variantes_prohibidas": [
      "V03 is not the best verified leaderboard result",
      "V03 remains the best verified result",
      "Word2Vec es mejor que V03",
      "Word2Vec improves sobre V03",
      "el bayesiano supera a V03",
      "el bayesiano mejora a V03"
    ]
  },
  "A19": {
    "bloque_de_la_lectura_sin_f1": true,
    "comprobaciones": {
      "el_f1_esta_publicado": true,
      "el_f1_no_aparece_en_la_regla_de_lectura": true,
      "el_intervalo_esta_publicado": true,
      "la_desviacion_entre_semillas_esta": true,
      "la_etiqueta_descriptiva_esta": true,
      "la_frase_sobre_la_pregunta_1_esta": true,
      "la_frase_sobre_la_pregunta_2_esta": true
    },
    "etiqueta": "DESCRIPTIVA_Y_SIN_CUALQUIER_LECTURA_INFERENCIAL",
    "f1_del_miembro": {
      "desviacion_entre_semillas": 0.004391366887091081,
      "etiqueta": "DESCRIPTIVA_Y_SIN_CUALQUIER_LECTURA_INFERENCIAL",
      "f1_media_columnas": 0.7155415037123107,
      "f1_medio_por_semilla": {
        "20261401": 0.7092477642752562,
        "20261402": 0.7141479489375369,
        "20261403": 0.7149469232232789,
        "20261404": 0.7189986972152285,
        "20261405": 0.7203661849102527
      },
      "f1_por_combinacion": [
        0.7177708006279436,
        0.6993131544765823,
        0.707134167656211,
        0.7114809000523286,
        0.7030895691609977,
        0.739616466340102,
        0.7131231429514692,
        0.7392880896618887,
        0.7358322699952691,
        0.7176637918201779,
        0.736895157409601,
        0.7282995047006041,
        0.7337638124367981,
        0.7238613569661004,
        0.747536244352939,
        0.6786917017570167,
        0.6943459080883189,
        0.6726092896174863,
        0.704604469847955,
        0.7009341840918687,
        0.6732646952416178,
        0.7356580344707093,
        0.7219392567440105,
        0.7192144892144893,
        0.7326071351252805
      ],
      "intervalo": [
        0.7068499033519289,
        0.7242331040726926
      ],
      "n_combinaciones": 25,
      "origen": "medido con metricas_multietiqueta sobre las predicciones binarizadas con los umbrales congelados de cada combinacion de pliegue y semilla"
    },
    "frase_sobre_la_pregunta_1": "UN_F1_POR_DEBAJO_DE_V03_NO_REFUTA_NADA_SOBRE_LA_PREGUNTA_1PORQUE_LA_PREGUNTA_1_NO_COMPITE_POR_F1",
    "frase_sobre_la_pregunta_2": "UN_F1_POR_DEBAJO_DE_V03_NO_REFUTA_NADA_SOBRE_LA_PREGUNTA_2PORQUE_LA_PREGUNTA_2_NO_BUSCA_SUPERAR_A_V03",
    "motivo_de_parada": "el F1 se ha usado como criterio"
  },
  "A2": {
    "comprobaciones": {
      "cero_reoptimizaciones_de_umbral": true,
      "el_artefacto_de_v03_no_cambio": true,
      "los_tres_umbrales_son_los_declarados": true
    },
    "huella_del_artefacto_de_v03": {
      "mtime_ns": 1790698435936799215,
      "ruta": "experiments/baseline_v03_frozen/pipeline_v03.json",
      "sha256": "89ee399ae849296f7a9a76c21d49e15e616a202b6ef8a5b1d87afbf8c528a90b",
      "tamano": 2222
    },
    "motivo_de_parada": "se ha reentrenado V03 o se ha reoptimizado un umbral",
    "reoptimizaciones_sobre_cualquier_conjunto": 0,
    "umbrales_congelados": {
      "AG": 0.46,
      "HS": 0.44,
      "TR": 0.51
    },
    "umbrales_leidos": {
      "AG": 0.46,
      "HS": 0.44,
      "TR": 0.51
    },
    "v03_reentrenado": false
  },
  "A3": {
    "comprobaciones": {
      "el_corrected_cv_recalculado_coincide_con_el_publicado": true,
      "las_filas_de_desarrollo_son_las_esperadas": true,
      "las_unidades_tweet_etiqueta_son_las_esperadas": true,
      "los_grupos_son_los_esperados": true
    },
    "corrected_cv_publicado": 0.7772114525,
    "corrected_cv_recalculado": 0.7772114525165721,
    "diferencia": 1.6572077043974787e-11,
    "motivo_de_parada": "el OOF de referencia no es el canonico y la pregunta 2 se queda sin contra",
    "n_filas": 1483,
    "n_grupos": 5,
    "nota_de_independencia": "el corrected_cv se recalcula con metricas_multietiqueta sobre las filas de prediccion y se compara con el publicado del pre-registro; la constante guardada en el npz se publica pero no se usa como ninguno de los dos lados",
    "tolerancia": 1e-09,
    "unidades_tweet_etiqueta": 4449,
    "valor_de_la_constante_guardada_en_el_npz": 0.7772114525165721
  },
  "A4": {
    "comprobaciones": {
      "el_conjunto_permitido_no_esta_vacio": true,
      "hubo_al_menos_una_llamada_al_selector": true,
      "la_comprobacion_sabe_fallar_por_mutacion": true,
      "ninguna_llamada_recibio_filas_fuera_del_inner_train": true
    },
    "llamadas": [
      {
        "filas_fuera_de_lo_permitido": 0,
        "n_columnas": 653,
        "n_filas": 836,
        "pliegue": 0,
        "umbral": 3.0
      },
      {
        "filas_fuera_de_lo_permitido": 0,
        "n_columnas": 605,
        "n_filas": 815,
        "pliegue": 0,
        "umbral": 3.0
      },
      {
        "filas_fuera_de_lo_permitido": 0,
        "n_columnas": 618,
        "n_filas": 818,
        "pliegue": 0,
        "umbral": 3.0
      },
      {
        "filas_fuera_de_lo_permitido": 0,
        "n_columnas": 615,
        "n_filas": 827,
        "pliegue": 0,
        "umbral": 3.0
      },
      {
        "filas_fuera_de_lo_permitido": 0,
        "n_columnas": 611,
        "n_filas": 817,
        "pliegue": 0,
        "umbral": 3.0
      },
      {
        "filas_fuera_de_lo_permitido": 0,
        "n_columnas": 636,
        "n_filas": 831,
        "pliegue": 1,
        "umbral": 3.0
      },
      {
        "filas_fuera_de_lo_permitido": 0,
        "n_columnas": 643,
        "n_filas": 835,
        "pliegue": 1,
        "umbral": 3.0
      },
      {
        "filas_fuera_de_lo_permitido": 0,
        "n_columnas": 628,
        "n_filas": 835,
        "pliegue": 1,
        "umbral": 3.0
      },
      {
        "filas_fuera_de_lo_permitido": 0,
        "n_columnas": 637,
        "n_filas": 835,
        "pliegue": 1,
        "umbral": 3.0
      },
      {
        "filas_fuera_de_lo_permitido": 0,
        "n_columnas": 629,
        "n_filas": 838,
        "pliegue": 1,
        "umbral": 3.0
      },
      {
        "filas_fuera_de_lo_permitido": 0,
        "n_columnas": 639,
        "n_filas": 838,
        "pliegue": 2,
        "umbral": 3.0
      },
      {
        "filas_fuera_de_lo_permitido": 0,
        "n_columnas": 638,
        "n_filas": 826,
        "pliegue": 2,
        "umbral": 3.0
      },
      {
        "filas_fuera_de_lo_permitido": 0,
        "n_columnas": 629,
        "n_filas": 822,
        "pliegue": 2,
        "umbral": 3.0
      },
      {
        "filas_fuera_de_lo_permitido": 0,
        "n_columnas": 618,
        "n_filas": 825,
        "pliegue": 2,
        "umbral": 3.0
      },
      {
        "filas_fuera_de_lo_permitido": 0,
        "n_columnas": 643,
        "n_filas": 831,
        "pliegue": 2,
        "umbral": 3.0
      },
      {
        "filas_fuera_de_lo_permitido": 0,
        "n_columnas": 633,
        "n_filas": 839,
        "pliegue": 3,
        "umbral": 3.0
      },
      {
        "filas_fuera_de_lo_permitido": 0,
        "n_columnas": 633,
        "n_filas": 823,
        "pliegue": 3,
        "umbral": 3.0
      },
      {
        "filas_fuera_de_lo_permitido": 0,
        "n_columnas": 635,
        "n_filas": 837,
        "pliegue": 3,
        "umbral": 3.0
      },
      {
        "filas_fuera_de_lo_permitido": 0,
        "n_columnas": 620,
        "n_filas": 821,
        "pliegue": 3,
        "umbral": 3.0
      },
      {
        "filas_fuera_de_lo_permitido": 0,
        "n_columnas": 611,
        "n_filas": 827,
        "pliegue": 3,
        "umbral": 3.0
      },
      {
        "filas_fuera_de_lo_permitido": 0,
        "n_columnas": 604,
        "n_filas": 836,
        "pliegue": 4,
        "umbral": 3.0
      },
      {
        "filas_fuera_de_lo_permitido": 0,
        "n_columnas": 625,
        "n_filas": 829,
        "pliegue": 4,
        "umbral": 3.0
      },
      {
        "filas_fuera_de_lo_permitido": 0,
        "n_columnas": 624,
        "n_filas": 832,
        "pliegue": 4,
        "umbral": 3.0
      },
      {
        "filas_fuera_de_lo_permitido": 0,
        "n_columnas": 603,
        "n_filas": 826,
        "pliegue": 4,
        "umbral": 3.0
      },
      {
        "filas_fuera_de_lo_permitido": 0,
        "n_columnas": 626,
        "n_filas": 828,
        "pliegue": 4,
        "umbral": 3.0
      }
    ],
    "llamadas_con_fugas": [],
    "motivo_de_parada": "la seleccion de variables se ha hecho fuera del inner_train y hay contamination en los dos bloques",
    "mutacion": "se ejecuta una vez la comprobacion con una seleccion hecha sobre el outer_train completo y tiene que detectar la fuga",
    "mutacion_detectada": true,
    "n_filas_del_inner_train": 1483,
    "n_llamadas_registradas": 25
  },
  "A5": {
    "cadena_literal": "LA_POSTERIOR_DE_M1_ES_APROXIMADA_Y_NO_ES_LA_POSTERIOR_EXACTA",
    "cadena_presente": true,
    "comprobaciones": {
      "el_campo_aproximacion_de_la_posterior_esta": true,
      "el_campo_iteraciones_esta": true,
      "el_campo_nombre_esta": true,
      "el_campo_prior_esta": true,
      "el_campo_tolerancia_esta": true,
      "la_cadena_literal_de_posterior_aproximada_esta": true
    },
    "metodo_declarado": {
      "actualizacion_de_la_dispersion": "alpha_k igual a gamma_k partido por m_k al cuadrado mas alpha_0, con Newton amortiguado sobre la verosimilitud penalizada",
      "aproximacion_de_la_posterior": "INFERENCIA VARIACIONAL DE CAMPO MEDIO, producto de gaussianas independientes",
      "iteraciones_del_punto_de_variacion": 30,
      "nombre": "logistica bayesiana dispersa con hiper-prior Gamma sobre columnas reescaladas",
      "por_que_no_la_ingenua": "la actualizacion de MacKay ingenua da un conteo de parametros efectivo invariante a alpha_0, luego A6 no tendria objeto; la forma con hiper-prior Gamma sobre columnas estandarizadas es la que hace que el conteo se mueva",
      "prior": "NORMAL de media cero y precision alpha_k, con un unico alpha_0 declarado",
      "referencias": [
        "Michael E. Tipping, Sparse Bayesian Learning and the Relevance Vector Machine, JMLR 2, 2001, 226 a 253",
        "Yuanqing Ng y Andrew Y. Ng, Adaptive Sparsity for Nonparametric Variable Selection, JMLR 13, 2012, 1833 a 1861",
        "Christopher M. Bishop, Pattern Recognition and Machine Learning, Springer, 2006, capitulo 11"
      ],
      "tolerancia_sobre_la_varianza_de_la_posterior": 1e-06
    },
    "motivo_de_parada": "el informe describe la posterior como exacta y el bloque 1 no describe lo que se ejecuto"
  },
  "A6": {
    "cadena_literal": "EL_NUMERO_DE_PARAMETROS_EFECTIVOS_QUE_PRODUCE_ESTE_METODO_ES_UN_PARAMETRO_DEL_METODOY_ESTA_DETERMINADO_POR_LA_ESCALA_DE_LA_PRIOR_ALPHA_0_Y_POR_LA_VARIANZA_POSTERIORNO_ES_UN_NUMERO_DE_MUESTRAS_EFECTIVAS_DEL_CONJUNTO_DE_DATOS",
    "comprobaciones": {
      "al_menos_una_etiqueta_supera_el_cinco_por_ciento": true,
      "el_numero_mueve_con_alpha_0": true,
      "la_cadena_literal_del_metodo_esta": true,
      "la_direccion_es_la_esperada": true,
      "los_tres_valores_de_la_rejilla_estan_publicados": true
    },
    "diferencia_relativa_entre_extremos": 0.23395942998323552,
    "direccion": "mas parametros efectivos con alpha_0 mas pequeno, que es la direccion fisica esperada: un alpha_0 mas pequeno deja la dispersion mas libre",
    "motivo_de_parada": "el numero de parametros efectivos no se mueve con alpha_0 y no es el numero que dice ser",
    "n_efectivos_por_alpha_0": {
      "0.50": {
        "brier_macro": 0.1426228452614681,
        "log_loss_macro": 0.4457108871115057,
        "n_activas_total": 6.0,
        "n_efectivos_total": 456.7722281544299,
        "n_filas_medidas": 1483,
        "por_etiqueta": {
          "AG": 446.42991281314335,
          "HS": 443.99171319117136,
          "TR": 479.8950584589751
        }
      },
      "1.00": {
        "brier_macro": 0.14167180615398664,
        "log_loss_macro": 0.4432757507690348,
        "n_activas_total": 13.0,
        "n_efectivos_total": 418.89253397720495,
        "n_filas_medidas": 1483,
        "por_etiqueta": {
          "AG": 407.4401559201983,
          "HS": 402.308485641683,
          "TR": 446.92896036973366
        }
      },
      "2.00": {
        "brier_macro": 0.1399489997350212,
        "log_loss_macro": 0.4377755699139304,
        "n_activas_total": 45.0,
        "n_efectivos_total": 370.1679464134697,
        "n_filas_medidas": 1483,
        "por_etiqueta": {
          "AG": 358.79367229203217,
          "HS": 350.0706270031783,
          "TR": 401.6395399451984
        }
      },
      "__texto__": "EL_NUMERO_DE_PARAMETROS_EFECTIVOS_QUE_PRODUCE_ESTE_METODO_ES_UN_PARAMETRO_DEL_METODOY_ESTA_DETERMINADO_POR_LA_ESCALA_DE_LA_PRIOR_ALPHA_0_Y_POR_LA_VARIANZA_POSTERIORNO_ES_UN_NUMERO_DE_MUESTRAS_EFECTIVAS_DEL_CONJUNTO_DE_DATOS"
    },
    "por_etiqueta": {
      "AG": {
        "alpha_0_0.5": 446.42991281314335,
        "alpha_0_2.0": 358.79367229203217,
        "diferencia_relativa": 0.24425246956357022,
        "supera_el_cinco_por_ciento": true
      },
      "HS": {
        "alpha_0_0.5": 443.99171319117136,
        "alpha_0_2.0": 350.0706270031783,
        "diferencia_relativa": 0.2682918215447432,
        "supera_el_cinco_por_ciento": true
      },
      "TR": {
        "alpha_0_0.5": 479.8950584589751,
        "alpha_0_2.0": 401.6395399451984,
        "diferencia_relativa": 0.1948401756571433,
        "supera_el_cinco_por_ciento": true
      }
    },
    "rejilla_de_alpha_0": [
      0.5,
      1.0,
      2.0
    ]
  },
  "A7": {
    "camino_1": "forma cerrada, log-odds de la diferencia de log-likelihoods y sigmoide",
    "camino_2": "recuento explicito con bincount sobre los no nulos y normalizacion con logaddexp",
    "comprobaciones": {
      "el_control_abarco_filas_reales": true,
      "el_segundo_camino_se_ejecuto_de_verdad": true,
      "la_diferencia_maxima_no_supera_1e_12": true
    },
    "diferencia_maxima_absoluta": 2.6867397195928788e-14,
    "motivo_de_parada": "la cuenta cerrada de M2 no se reproduce y no se puede confiar en la aritmetica del brazo de referencia",
    "n_ejecuciones_del_control": 15,
    "n_filas_comparadas": 4449,
    "por_que_una_tolerancia_tan_estrecha_es_razonable": "las dos cuentas son cerradas en aritmetica de punto flotante, luego la diferencia tiene que ser del orden del epsilon de la maquina",
    "tolerancia": 1e-12
  },
  "A8": {
    "cifras_por_brazo": {
      "M1": {
        "brier_macro": 0.14167180615398664,
        "brier_por_etiqueta": {
          "AG": 0.15654787896486652,
          "HS": 0.16737546344043328,
          "TR": 0.10109207605666011
        },
        "ece_macro": 0.031658918536293115,
        "log_loss_macro": 0.4432757507690348,
        "log_loss_sin_recortar": 0.4432757507690348,
        "n_filas_medidas": 1483,
        "pendientes": 0.9736575454733568
      },
      "M2": {
        "brier_macro": 0.14880603089223432,
        "brier_por_etiqueta": {
          "AG": 0.1653176573817833,
          "HS": 0.17581182666905174,
          "TR": 0.1052886086258679
        },
        "ece_macro": 0.032623222871789544,
        "log_loss_macro": 0.45992445512665364,
        "log_loss_sin_recortar": 0.45992445512665364,
        "n_filas_medidas": 1483,
        "pendientes": 0.9439487062939853
      }
    },
    "comparacion_contra_el_liston": {
      "M1": {
        "deltas_contra_el_liston": {
          "brier_macro": 0.019217478331500518,
          "ece_macro": 0.003662077360342214,
          "log_loss_macro": 0.05226661441582686
        },
        "dentro_de_su_margen": {
          "brier_macro": false,
          "ece_macro": true,
          "log_loss_macro": false
        },
        "direccion": "el margen de 1.3 es de NO INFERIORIDAD, luego solo el lado desfavorable se penaliza y estar mejor que el liston no es un fallo",
        "faltan": [],
        "lectura": "NEGATIVO",
        "magnitudes": [
          "log_loss_macro",
          "brier_macro",
          "ece_macro"
        ],
        "margenes": {
          "brier_macro": 0.0015,
          "ece_macro": 0.01,
          "log_loss_macro": 0.005
        },
        "que_no_se_afirma": "que el bayesiano supere a V03 en calibracion, ni que su margen sea superior; V03 remains the best verified leaderboard result."
      },
      "M2": {
        "deltas_contra_el_liston": {
          "brier_macro": 0.026351703069748197,
          "ece_macro": 0.0046263816958386425,
          "log_loss_macro": 0.0689153187734457
        },
        "dentro_de_su_margen": {
          "brier_macro": false,
          "ece_macro": true,
          "log_loss_macro": false
        },
        "direccion": "el margen de 1.3 es de NO INFERIORIDAD, luego solo el lado desfavorable se penaliza y estar mejor que el liston no es un fallo",
        "faltan": [],
        "lectura": "NEGATIVO",
        "magnitudes": [
          "log_loss_macro",
          "brier_macro",
          "ece_macro"
        ],
        "margenes": {
          "brier_macro": 0.0015,
          "ece_macro": 0.01,
          "log_loss_macro": 0.005
        },
        "que_no_se_afirma": "que el bayesiano supere a V03 en calibracion, ni que su margen sea superior; V03 remains the best verified leaderboard result."
      }
    },
    "comprobaciones": {
      "existe_M1_brier_macro": true,
      "existe_M1_brier_por_etiqueta": true,
      "existe_M1_ece_macro": true,
      "existe_M1_log_loss_macro": true,
      "existe_M1_log_loss_sin_recortar": true,
      "existe_M1_pendientes": true,
      "existe_M2_brier_macro": true,
      "existe_M2_brier_por_etiqueta": true,
      "existe_M2_ece_macro": true,
      "existe_M2_log_loss_macro": true,
      "existe_M2_log_loss_sin_recortar": true,
      "existe_M2_pendientes": true,
      "los_dos_brazos_estan_medidos": true,
      "ninguna_magnitud_es_nan": true,
      "ninguna_magnitud_falta": true
    },
    "con_nan": [],
    "etiqueta_del_f1": "DESCRIPTIVA_Y_SIN_CUALQUIER_LECTURA_INFERENCIAL",
    "faltantes": [],
    "liston_v03_cal": {
      "brier_diferencia": 6.7217751387183e-07,
      "brier_macro": 0.12245432782248612,
      "brier_publicado": 0.122455,
      "ece_macro": 0.0279968411759509,
      "log_loss_diferencia": 1.363532079468044e-07,
      "log_loss_macro": 0.39100913635320794,
      "log_loss_macro_recortado": 0.39100913635320794,
      "log_loss_publicado": 0.391009,
      "origen": "pares de Platt CONGELADOS de reports/p76_calibracion.json aplicados al OOF canonico, sin reejecutar P76",
      "pendientes": {
        "AG": 0.9741544369500666,
        "HS": 0.9965694564628117,
        "TR": 0.9989420790784712
      },
      "pendientes_maxima_diferencia": 4.5646281165989677e-07,
      "pendientes_publicadas": {
        "AG": 0.974154,
        "HS": 0.996569,
        "TR": 0.998942
      },
      "reproducido_dentro_de_tolerancia": true,
      "tolerancia": 5e-06
    },
    "magnitudes": [
      "log_loss_macro",
      "log_loss_sin_recortar",
      "brier_macro",
      "brier_por_etiqueta",
      "ece_macro",
      "pendientes"
    ],
    "tolerancia": "seis decimales; no hay cota de exito numerico, lo que falla es la medicion"
  },
  "A9": {
    "banda_muerta_bin": 1e-09,
    "comprobaciones": {
      "M1_AG_no_tiene_mas_de_la_mitad_de_bins_vacios": true,
      "M1_AG_no_tiene_nan": true,
      "M1_AG_tiene_10_bins": true,
      "M1_AG_tiene_un_bin_con_20_unidades": true,
      "M1_HS_no_tiene_mas_de_la_mitad_de_bins_vacios": true,
      "M1_HS_no_tiene_nan": true,
      "M1_HS_tiene_10_bins": true,
      "M1_HS_tiene_un_bin_con_20_unidades": true,
      "M1_TR_no_tiene_mas_de_la_mitad_de_bins_vacios": true,
      "M1_TR_no_tiene_nan": true,
      "M1_TR_tiene_10_bins": true,
      "M1_TR_tiene_un_bin_con_20_unidades": true,
      "M2_AG_no_tiene_mas_de_la_mitad_de_bins_vacios": true,
      "M2_AG_no_tiene_nan": true,
      "M2_AG_tiene_10_bins": true,
      "M2_AG_tiene_un_bin_con_20_unidades": true,
      "M2_HS_no_tiene_mas_de_la_mitad_de_bins_vacios": true,
      "M2_HS_no_tiene_nan": true,
      "M2_HS_tiene_10_bins": true,
      "M2_HS_tiene_un_bin_con_20_unidades": true,
      "M2_TR_no_tiene_mas_de_la_mitad_de_bins_vacios": true,
      "M2_TR_no_tiene_nan": true,
      "M2_TR_tiene_10_bins": true,
      "M2_TR_tiene_un_bin_con_20_unidades": true
    },
    "curvas": {
      "M1_AG": {
        "area_entre_curva_y_diagonal": 0.029292194232864294,
        "bins": [
          {
            "bin": 0,
            "frecuencia_observada": 0.06666666666666667,
            "inferior": 0.0,
            "media_predicha": 0.05350320799556235,
            "n": 195,
            "superior": 0.1
          },
          {
            "bin": 1,
            "frecuencia_observada": 0.12393162393162394,
            "inferior": 0.1,
            "media_predicha": 0.1502859825002197,
            "n": 234,
            "superior": 0.2
          },
          {
            "bin": 2,
            "frecuencia_observada": 0.23411371237458195,
            "inferior": 0.2,
            "media_predicha": 0.2513317514453222,
            "n": 299,
            "superior": 0.30000000000000004
          },
          {
            "bin": 3,
            "frecuencia_observada": 0.2923076923076923,
            "inferior": 0.30000000000000004,
            "media_predicha": 0.34243571683789686,
            "n": 195,
            "superior": 0.4
          },
          {
            "bin": 4,
            "frecuencia_observada": 0.4621212121212121,
            "inferior": 0.4,
            "media_predicha": 0.44446172690164776,
            "n": 132,
            "superior": 0.5
          },
          {
            "bin": 5,
            "frecuencia_observada": 0.6464646464646465,
            "inferior": 0.5,
            "media_predicha": 0.5508129191205039,
            "n": 99,
            "superior": 0.6000000000000001
          },
          {
            "bin": 6,
            "frecuencia_observada": 0.6857142857142857,
            "inferior": 0.6000000000000001,
            "media_predicha": 0.6465605785468697,
            "n": 70,
            "superior": 0.7000000000000001
          },
          {
            "bin": 7,
            "frecuencia_observada": 0.75,
            "inferior": 0.7000000000000001,
            "media_predicha": 0.7502800635512138,
            "n": 56,
            "superior": 0.8
          },
          {
            "bin": 8,
            "frecuencia_observada": 0.8571428571428571,
            "inferior": 0.8,
            "media_predicha": 0.8530658300150427,
            "n": 63,
            "superior": 0.9
          },
          {
            "bin": 9,
            "frecuencia_observada": 0.9214285714285714,
            "inferior": 0.9,
            "media_predicha": 0.9569250172156267,
            "n": 140,
            "superior": 1.0
          }
        ],
        "n": 1483,
        "n_bins": 10,
        "n_bins_con_al_menos_20_unidades": 10,
        "n_vacios": 0
      },
      "M1_HS": {
        "area_entre_curva_y_diagonal": 0.03397634386719374,
        "bins": [
          {
            "bin": 0,
            "frecuencia_observada": 0.05970149253731343,
            "inferior": 0.0,
            "media_predicha": 0.046441431582866,
            "n": 134,
            "superior": 0.1
          },
          {
            "bin": 1,
            "frecuencia_observada": 0.11538461538461539,
            "inferior": 0.1,
            "media_predicha": 0.15051946470787597,
            "n": 156,
            "superior": 0.2
          },
          {
            "bin": 2,
            "frecuencia_observada": 0.20192307692307693,
            "inferior": 0.2,
            "media_predicha": 0.25232498939237524,
            "n": 208,
            "superior": 0.30000000000000004
          },
          {
            "bin": 3,
            "frecuencia_observada": 0.3021276595744681,
            "inferior": 0.30000000000000004,
            "media_predicha": 0.350437506007888,
            "n": 235,
            "superior": 0.4
          },
          {
            "bin": 4,
            "frecuencia_observada": 0.4523809523809524,
            "inferior": 0.4,
            "media_predicha": 0.44753821827884865,
            "n": 210,
            "superior": 0.5
          },
          {
            "bin": 5,
            "frecuencia_observada": 0.625,
            "inferior": 0.5,
            "media_predicha": 0.5475215718506726,
            "n": 144,
            "superior": 0.6000000000000001
          },
          {
            "bin": 6,
            "frecuencia_observada": 0.6526315789473685,
            "inferior": 0.6000000000000001,
            "media_predicha": 0.6496770772509476,
            "n": 95,
            "superior": 0.7000000000000001
          },
          {
            "bin": 7,
            "frecuencia_observada": 0.7837837837837838,
            "inferior": 0.7000000000000001,
            "media_predicha": 0.7471356480450371,
            "n": 74,
            "superior": 0.8
          },
          {
            "bin": 8,
            "frecuencia_observada": 0.8059701492537313,
            "inferior": 0.8,
            "media_predicha": 0.8502406212953124,
            "n": 67,
            "superior": 0.9
          },
          {
            "bin": 9,
            "frecuencia_observada": 0.9375,
            "inferior": 0.9,
            "media_predicha": 0.9572499527481119,
            "n": 160,
            "superior": 1.0
          }
        ],
        "n": 1483,
        "n_bins": 10,
        "n_bins_con_al_menos_20_unidades": 10,
        "n_vacios": 0
      },
      "M1_TR": {
        "area_entre_curva_y_diagonal": 0.03170821750882132,
        "bins": [
          {
            "bin": 0,
            "frecuencia_observada": 0.03669724770642202,
            "inferior": 0.0,
            "media_predicha": 0.039704080936022915,
            "n": 654,
            "superior": 0.1
          },
          {
            "bin": 1,
            "frecuencia_observada": 0.1319796954314721,
            "inferior": 0.1,
            "media_predicha": 0.14431753765842345,
            "n": 197,
            "superior": 0.2
          },
          {
            "bin": 2,
            "frecuencia_observada": 0.19327731092436976,
            "inferior": 0.2,
            "media_predicha": 0.24659992220040486,
            "n": 119,
            "superior": 0.30000000000000004
          },
          {
            "bin": 3,
            "frecuencia_observada": 0.37333333333333335,
            "inferior": 0.30000000000000004,
            "media_predicha": 0.34995708617028937,
  
```

## Contexto del proyecto, citado y no recalculado

- V03 corrected_cv: 0.7772114525
- NB-SVM canonico: 0.763839
- Logistica L1 con C 4.0: 0.783423, IC pareado [-0.0081, 0.0202] que contiene el cero
- P82 RNN local 0.639659 y leaderboard 0.6815

## Sobre el margen y el desajuste de dominio

- El margen de error de la validacion cruzada depende del desajuste de dominio, luego comparar familias con capacidades distintas con la misma metrica local es enganoso. Un bayesiano disperso tiene menos capacidad que una RNN, asi que su margen sera probablemente mas estrecho.
- Si se observa, se publica como una OBSERVACION SOBRE EL MARGEN y no como una afirmacion de rendimiento.

## Limite de 5.5 y anclaje externo

- M1_SE_IMPLEMENTA_DESDE_CERO_EN_ESTE_PROYECTO_Y_NO_SE_ANCLA_CONTRA_NINGUNA_LIBRERIADE_INFERENCIA_BAYESIANA
- Librerias comprobadas: 12
- Encontradas: []
- Estado del limite: SIN_HACER

## Conclusion

V03 remains the best verified leaderboard result.

- NO_SE_AFIRMA_QUE_EL_BAYESIANO_SUPERE_A_V03
- P83_ES_UN_MODELO_LINEAL_SOBRE_UNA_MATRIZ_RALA_DE_1483_FILAS_LUEGO_EL_PRESUPUESTO_ES_DE_MINUTOSY_ESA_ASIMETRIA_CON_P82_SE_DECLARA_PARA_QUE_NO_SE_LEA_COMO_QUE_P83_ES_EL_EXPERIMENTO_IMPORTANTE
