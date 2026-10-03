# Pre-registro congelado - P6: canal NER estructural (tipos de entidad)

**Fecha**: 2026-09-29. **Estado**: CONGELADO antes de ejecutar (commit y hashes en
`experiments/age_linguistic_05_preregistro.json`). Autorizado por D-076 (etiquetas si,
vectores no) y por el brief del ciclo.

## La pregunta

> La estructura de entidades (TIPOS PER/ORG/LOC/MISC, nunca nombres) contiene informacion
> complementaria al campeon adoptado (V30 + plug-in), y un combinador anidado la explota
> para mejorar el macro-F1 corregido?

P2 cerro POS-secuencia (senal aislada 0.2751; integracion fija 50/50 negativa por
dilucion). P6 responde otra pregunta: entidades + integracion CONTROLADA. No se extrapola
el resultado de P2 a P6.

## Materiales congelados (SHA256)

| Artefacto | SHA256 (16) |
|---|---|
| anotacion_md_parquet | `dacbc3055ccde0d9...` |
| anotacion_lg_parquet | `81287d541c082ea8...` |
| corpus_train | `39c6c8270366fe15...` |
| canonico_y_true | `b1c928f9b9f55755...` |
| canonico_fold_idx | `74d18330c40c3598...` |
| canonico_v30_prob | `3d77352486605971...` |
| manifest_plugin | `8d635e6a546b5a0e...` |

Pesos congelados del plug-in: `[1.0, 1.1749, 1.2721, 1.163, 1.001, 0.8128]`
Baseline primario recomputado: **0.6159732** (la regla que ejecuta la submission).
Referencia secundaria de continuidad: nested 0.6154643.

## Representaciones (todas sin nombres concretos: solo tipos)

| ID | Definicion | Vectorizador | Rol |
|---|---|---|---|
| N1 | un token ENT_<TIPO> por span de entidad, en orden; vacio si no hay | TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df=5, sublinear_tf=True, lowercase=True, token_pattern=r'(?u)\S+') | primaria (brief seccion 8) |
| N1b | N1 excluyendo spans que contienen menciones (@), hashtags (#), URL o RT | identico a N1 | atribucion (revision Kaggle Specialist) |
| N2 | por span: ENT_<TIPO>, <TIPO>_L_<POSizq|START>, <TIPO>_R_<POSder|END> | TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df=5, max_features=2000, sublinear_tf=True, lowercase=True, token_pattern=r'(?u)\S+') | extension (brief seccion 9; guarda de dimensionalidad del revisor ML) |

Canal: LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs'), ajustado SOLO en el train de cada fold; filas con secuencia vacia -> probabilidad uniforme 1/6 (identico en todos los brazos).
Anotadores: md y lg (D-076 ordeno comparacion; los dos corren el mismo protocolo).

## Integracion (unica, pre-registrada)

- Mecanismo: combinador logistico anidado sobre dos bloques de probabilidad.
- Features: [p_V30 normalizadas por fila (L2, 6 dims) concatenadas con p_canal normalizadas por fila (L2, 6 dims)] = 12 features.
- Modelo: LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs') ajustado SOLO en las filas de train de cada fold (con OOF estricto del canal, no in-sample).
- Decision: argmax_c(w_c * p_combinada_c) con la regla y los pesos del plug-in CONGELADOS.
- Suposicion declarada: los pesos del plug-in no son invariantes a escala; aplicarlos a p_combinada es una suposicion pre-registrada (regla congelada, brief seccion 24).
- Por que no es la familia cerrada: V31-06 anadio features del canal al meta-modelo de V30 (refit de V30); P33/P34/G12 inyectaron un residuo en el score. Aqui V30 es caja negra (solo interfaz de probabilidad), sin features externas ni residuo, y la regla de decision queda congelada. Es un candidato nuevo.

Diagnostico no decisorio: se reportara tambien argmax(p_combinada) sin regla, para
lectura; la decision usa la regla congelada.

## Controles (definidos ANTES de ver el resultado)

- `barajado`: la secuencia de cada representacion se permuta ENTRE tweets (una
  permutacion, semilla 20261029), preservando marginal exacta. El pipeline
  COMPLETO (canal + combinador + regla) se repite con la version barajada.
- `N1b`: variante que excluye spans sobre menciones/RT/hashtags/URLs (~28% de los tokens
  de entidad; canales de V31 cerrados). Si el aporte solo sobrevive con marcas de Twitter,
  se clasifica como redundancia, no como senal nueva.
- Se registran los coeficientes del combinador por fold (peso del bloque canal): un peso
  ~0 significa recalibracion de V30, no informacion nueva de NER.

## Gates (todos obligatorios, por brazo representacion x anotador)

- delta >= **+0.0015** sobre el baseline primario;
- folds positivos >= **4/5**;
- delta real > delta barajado (misma representacion y anotador);
- IC95 bootstrap pareado (2.000, semilla 20261010) sin cero;
- sin leakage (auditado por diseno: canal OOF por fold, combinador solo con train).

## Revisor de agentes (auditoria del brief seccion 5) - puntos adoptados

| Agente | Punto | Decision |
|---|---|---|
| Kaggle Specialist | legalidad OK bajo D-076; documentar solapamiento de marcas | adoptado (N1b + doc) |
| Kaggle Specialist | correr md y lg (mandato D-076) | adoptado |
| Kaggle Specialist | segunda particion antes de cualquier submission | ya en el arbol |
| Revisor ML | N2 con (1,2) + max_features=2000 (dimensionalidad) | adoptado |
| Revisor ML | combinador distinto de la familia cerrada; documentar en ADR | adoptado |
| Revisor ML | registrar coeficientes del combinador | adoptado |
| Revisor ML | varias semillas de barajado | NO adoptado: brief seccion 26 prohibe multiples semillas; queda como limite declarado |
| Revisor ML | control barajado por pipeline completo | adoptado |

## Que NO se hara

- No hay grid de NER, ni variantes de pesos, ni varios modelos NER, ni multiples semillas,
  ni tuning del anotador (brief seccion 26).
- No se toca el test, ni el holdout, ni el leaderboard. Sin submission automatica.
- El campeon adoptado (V30 + plug-in, LB 0.63575) no se modifica.

## Arbol de decision pre-registrado

| Resultado | Estado | Siguiente paso |
|---|---|---|
| algun brazo pasa los 5 gates | **PROMISING** | segunda particion agrupada con configuracion congelada (semilla 987654, reejecucion declarada); sin submission sin autorizacion |
| ningun brazo pasa | **CIERRE** | NER se cierra documentado; no se buscan otras variantes inmediatamente (brief seccion 22) |

## Limites declarados

- La anotacion NER esta sobredetectando sobre este corpus (23,7% de tokens en entidades;
  MISC 57%; ~28% sobre marcas). El claim futuro sera 'senal empirica en anotacion
  ruidosa', no semantica.
- Una sola semilla de barajado (presupuesto); el revisor ML recomendo varias y queda
  registrado como recomendacion no ejecutada.
- El esquema de stacking usa OOF estricto del canal para las filas de train del combinador
  (practica estandar del proyecto: V12/V20); el control pasa por el mismo esquema.
