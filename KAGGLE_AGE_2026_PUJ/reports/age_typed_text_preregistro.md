# Pre-registro congelado - P7: texto tipado (texto original + lemma_POS)

**Fecha**: 2026-09-30. **Estado**: CONGELADO antes de ejecutar (commit y hashes en
`experiments/age_linguistic_08_texto_tipado_preregistro.json`). Autorizado por D-076
(etiquetas y lemas si; vectores no) y por el brief del ciclo.

## La pregunta

> Puede una representacion lexica tipada conservar el contenido que hace fuerte al
> TF-IDF y, simultaneamente, incorporar la funcion linguistica de cada token de una
> forma que el campeon no capture?

A = V30 + plug-in (campeon). B = A + canal de texto tipado. **El unico cambio es el
canal**: bloques de texto original + lemma_POS.

## Materiales congelados (SHA256)

| Artefacto | SHA256 (16) |
|---|---|
| anotacion_md_parquet | `dacbc3055ccde0d9...` |
| anotacion_lg_parquet | `81287d541c082ea8...` |
| modelo_md_meta_json | `889a2fd05d217fa7...` |
| modelo_lg_meta_json | `5007184c49f3cf23...` |
| corpus_train | `39c6c8270366fe15...` |
| canonico_y_true | `b1c928f9b9f55755...` |
| canonico_fold_idx | `74d18330c40c3598...` |
| canonico_v30_prob | `3d77352486605971...` |
| manifest_plugin | `8d635e6a546b5a0e...` |

Pesos congelados del plug-in: `[1.0, 1.1749, 1.2721, 1.163, 1.001, 0.8128]`
Baseline primario recomputado: **0.6159732** (brazo A; tambien la ablacion sin canal de D-076 condicion 4). Referencia secundaria: nested 0.6154643.

## Representacion (congelada)

```text
bloque A: texto original        -> TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=3, max_df=0.5, sublinear_tf=True, lowercase=True, strip_accents=None, token_pattern=r"(?u)\b\w+\b")
bloque B: tokens lemma_POS      -> TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=5, sublinear_tf=True, lowercase=True, token_pattern=r"(?u)\S+")
escala por bloque: 1.0 fija (neutra; definida antes; sin busqueda de escala)
vectorizadores SEPARADOS (sin colision de vocabulario); cada uno se ajusta SOLO
en el train de cada fold; cada bloque L2 por fila (TF-IDF por defecto)
modelo del canal: LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs', tol=1e-3)
```

- Bloque A clonado bit a bit de V31-03 (VEC_PALABRA); bloque B: unigramas y bigramas de
  lemma_POS, sin trigramas. No se anaden NER, dependencias, POS puro, morfologia,
  longitud, URL, hashtags ni features densas (la finalidad es aislar P7).
- El parquet de anotacion se valida en el runner (sin nulos, tokenizacion coherente,
  documentos con B vacio: se cuentan y se reportan; no hay mas tratamiento).

## Integracion (unica, pre-registrada)

- **mecanismo**: combinador logistico anidado sobre dos bloques de probabilidad (identico a P6)
- **features**: [p_V30 L2 por fila (6) || p_texto_tipado L2 por fila (6)] = 12 features
- **modelo**: LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs') ajustado SOLO en el train de cada fold, con predicciones OOF del canal (nunca in-sample)
- **decision**: argmax_c(w_c * p_combinada_c) con regla y pesos del plug-in CONGELADOS
- **justificacion**: igual que P6 (procedimiento equivalente); evita la mezcla fija que diluyo en P2; V30 queda como caja negra; sin features externas ni residuos
- **ablacion_sin_canal**: el brazo A (V30 + plug-in, 0.6159732) ES la ablacion sin canal que exige la condicion 4 de D-076; se reporta explicitamente como tal

## Control (definido ANTES de ver el resultado)

- **que**: el bloque B (lemma_POS) se permuta DENTRO DE CADA FOLD (5 permutaciones, semilla 20261030; el mismo mapeo para md y lg); el bloque A (texto) queda intacto
- **que_preserva**: misma dimensionalidad, misma frecuencia de tokens, misma distribucion marginal por fold (el IDF por fold queda identico al original)
- **que_rompe**: solo la correspondencia tweet <-> estructura tipada
- **por_que_dentro_de_fold**: recomendacion del revisor ML: aisla el emparejamiento token-etiqueta sin mover la marginal de ningun split; queda registrado tambien el barajado global como alternativa no usada

## Gates (todos obligatorios, por anotador)

- delta >= **+0.0015** sobre el baseline primario;
- folds positivos >= **4/5**;
- delta real > delta barajado (misma representacion y anotador);
- IC95 bootstrap pareado (2.000, semilla 20261010) sin cero;
- sin leakage (canal OOF por fold; combinador solo con train; vectorizadores por fold).

Resultado: PROMISING si pasa; MARGINAL si es positivo por debajo del gate; CLOSED si es
negativo o el control empata o gana. Si falla: no se abre grid (brief).

## Revisiones de agentes (brief seccion 5) - puntos adoptados

| Agente | Punto | Decision |
|---|---|---|
| Kaggle Specialist | legalidad OK; lemas/etiquetas son anotacion de forma | adoptado |
| Kaggle Specialist | PII: los lemas PROPN vienen del propio tweet (ya en bloque A) | registrado |
| Kaggle Specialist | hashes de particion y anotacion; join 1:1 por fila; IDF por fold | adoptado |
| Kaggle Specialist | barajar dentro del split | adoptado (dentro de cada fold) |
| Kaggle Specialist | brazo neutral 0.5 extra | NO adoptado: presupuesto 1 experimento + 1 control |
| Kaggle Specialist | correr md y lg (D-076) | adoptado |
| Revisor ML | clonar VEC_PALABRA de V31-03 bit a bit | adoptado |
| Revisor ML | tol=1e-3, max_iter=2000, registrar n_iter_ y warnings | adoptado (smoke: 123 iter) |
| Revisor ML | escala 1.0 por bloque, no convertirla en grid | adoptado |
| Revisor ML | barajar dentro de cada split, mismo criterio md/lg | adoptado |
| Revisor ML | asserts de orden de clases; diagnostico T sola y sin regla | adoptado |
| Revisor ML | smoke test previo de 1 fold | hecho: 19,1 s/fold, ~6 min los 20 fits |

## Diagnostico adicional (no decisorio)

- macro-F1 del canal T solo; macro-F1 del combinador con argmax puro (sin regla).
- Delta descriptivo por cubos de longitud: cortos<=40, medios_41-93, largos_94-130, muy_largos>130.
- Si T sola supera a V30 en la particion canonica se registra como hallazgo, sin tocar
  al campeon congelado.

## Que NO se hara

- No hay grid de n-gramas, escala, tokenizadores, dimensiones, semillas ni combinadores.
- No se toca el test, ni el holdout, ni el leaderboard. Sin submission automatica.
- No se reabren P2, P6 ni P5 en este ciclo; P5 se evaluara SOLO tras clasificar P7.

## Arbol de decision pre-registrado

| Resultado | Estado | Siguiente paso |
|---|---|---|
| pasa los 5 gates | **PROMISING** | bootstrap + complementariedad ya calculados; segunda particion agrupada con configuracion congelada (semilla 987654); sin submission |
| positivo < +0.0015 | **MARGINAL** | cierre documentado; solo P5 si el gap la justifica |
| <=0 o control empata/gana | **CLOSED** | cierre documentado; revisar P5 una sola vez |

## Interpretacion pre-registrada (regla del brief)

- Si P7 falla: 'la representacion lexema+POS no aporta mejora al campeon bajo el
  mecanismo evaluado' (NO 'la informacion linguistica no sirve').
- Si P7 mejora: 'la representacion lexema+POS aporta informacion complementaria al
  campeon bajo el protocolo evaluado' (NO 'el POS funciona').
