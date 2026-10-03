# P83: modelo bayesiano disperso, doble objetivo

V03 remains the best verified leaderboard result.

- Experimento: P83
- Script: experiments/p83_01_bayesiano.py
- Modo: humo
- Marca de corrida: b27ab662
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

- Lectura de la regla de 1.5: **AFFIRMATIVO**
- Ambiguedad declarada: NO_SON_INDEPENDIENTES
- Los dos intervalos no son independientes: NO_SON_INDEPENDIENTES
- El F1 no aparece en esta regla, porque la pregunta 1 no compite por F1.
- UN_F1_POR_DEBAJO_DE_V03_NO_REFUTA_NADA_SOBRE_LA_PREGUNTA_1PORQUE_LA_PREGUNTA_1_NO_COMPITE_POR_F1

## Lectura de la pregunta 2: complementariedad de errores

- EL_RESCATE_Y_EL_DANO_SON_UN_RECUENTO_DE_CELDAS_Y_NO_SON_COMPARABLES_CON_EL_CORRECTED_CVNI_CON_NINGUN_MACRO_F1
- EL_CORRECTED_CV_DEL_ENSAMBLAJE_SOBRE_EL_OOF_NO_ES_UNA_MEJORA_DE_V03PORQUE_EL_OOF_YA_SE_USO_PARA_ELEGIR_EL_ALPHA_Y_PARA_CUALQUIER_OTRA_ELECCIONLUEGO_ES_UN_NUMERO_OPTIMISTA_Y_NO_PUBLICABLE_COMO_RESULTADO
- El F1 no aparece en esta regla, porque la pregunta 2 no busca superar a V03.
- UN_F1_POR_DEBAJO_DE_V03_NO_REFUTA_NADA_SOBRE_LA_PREGUNTA_2PORQUE_LA_PREGUNTA_2_NO_BUSCA_SUPERAR_A_V03

## Cifras de la pregunta 1

## Cifras de la pregunta 2

## Veredictos de las veinte afirmaciones

| id | bloque | estado | stop | que hace |
|---|---|---|---|---|

## Aislamiento de afirmaciones

- ERROR no es ni CONFIRMADO ni REFUTADO ni NO_EVALUABLE; la corrida continua con la siguiente y una Parada con STOP se propaga.
- Errores: 0
- Avisos: 0
- Escrituras fallidas: 0

## Resto del detalle

```json
{}
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
