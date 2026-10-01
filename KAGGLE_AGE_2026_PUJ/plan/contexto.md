# Contexto para el siguiente agente (V17)

## Donde esta cada entregable de P0
- `reports/v17_gap_audit.md` - informe de la auditoria de brechas (8 secciones, 42 familias inventariadas).
- `reports/v17_gap_audit.csv` - 9 columnas exactas, 1 fila de notacion + 42 filas de familia. Las fronteras
  se escriben `18-24__25-34` porque el guion vertical delimita columnas.
- No se genero ningun otro artefacto. No se toco V12, V15, V16, `experiments.csv`, `PROJECT_STATE.md`,
  `DECISIONS.md`, `RUNLOG.md` ni `FINAL_REPORT.md`: P0 es documental y la memoria se actualiza al cierre.

## Reconciliacion de la referencia (leer la seccion 2 del informe antes de nada)
- **Canonico para V17: 0,6005**, la escala *pooled* de `metricas_globales`
  (`scripts/v13_p1b_fusion_tokens.py:132-142`).
- **NO es un redondeo distinto.** Son dos estimadores: pooled 0,6005017473 frente a media-por-fold
  0,6004103057, diferencia 9,14e-5, mayor que la unidad de redondeo. Los dos reproducen exactamente y
  comparten el vector por fold `0,6007/0,6050/0,6032/0,6004/0,5928`.
- Motivo: el macro-F1 de la confusion acumulada no es la media de los macro-F1 por fold (efecto Jensen).
- Usar 0,6004 daria **+0,0001 regalado** en cada delta y rompe el emparejamiento con A2 (0,6011) y con
  el IC95 de `bootstrap_diferencia`, que tambien es pooled.
- Ojo: 0,6007 es la **accuracy** de V12, no la macro-F1. Y 0,62182 / 0,62188 son dos leaderboards de
  dos artefacts distintos con 93,88 % de acuerdo (D-036, cerrado, no es un problema abierto).

## Reparto de las familias
42 familias: 30 cerradas o vigentes, 3 abiertas de las que 2 son las nuevas (39, 40, 41) y 1 es la
reconciliacion. De las diez pistas: **cerradas 3** (P6 residual, P8 perfil de confianza, P10
decodificacion local), **parcialmente probadas 5** (P1, P2, P4, P7, P9), **genuinamente nuevas 2**
(P3, P5). P0 es esta fase. P0' esta a medias: la reconciliacion esta hecha, la reproduccion de A2 no.

## Las tres conclusiones que mandan
1. **La referencia es 0,6005 pooled.** Ver arriba.
2. **El especialista de frontera no fallo: fallo la representacion.** Medido en modo inerte sobre
   `features/oof_E028a_pares.npy`: el especialista sparse por par saca **0,6864** y **0,7218** dentro de
   `18-24|25-34` y `25-34|35-49`, frente a los **0,4963** y **0,4974** de V12 y a los **0,4944** y
   **0,4942** del especialista de V13 P3a sobre las 279 variables. Gap de **+0,19 y +0,22** sin usar.
   Y los 5 margenes de E-028a **ya son entradas de V12** (`MEMBER_FILES`), que sigue fallando el par.
3. **P2 tal como esta escrita en el brief seria repetir P3a.** Sobre las 279 variables es sinonimo
   (mismo protocolo anidado, resultado medido: peor que V12). La version nueva exige la representacion
   dispersa. Orden propuesto en la seccion 8.1: P0' -> P1.1 -> P2SPARSE -> P3.1 -> P5 -> P2.3.

## Lo siguiente
- **P0'**: reproducir A2 con `scripts/v16_arquitectura.py` sin editarlo. Si no reproduce, STOP.
- **P1.1** primero: es el diagnostico barato que decide si P2 y P3 se cierran sin entrenar.
- Empezar todo script con `autotest_numerico()` y usar `PARES` de
  `scripts/v13_p1b_fusion_tokens.py:76` para las etiquetas de frontera (ver riesgos de la seccion 7).

## No verificado (marcado en el informe, no suponer)
- La parte (b) de D-019 (enrutamiento por margen del par-SVM con tau) **no es reproducible**: ni el
  script ni el JSON contienen ese codigo. Solo el resumen 0,0000 consta en dos informes.
- El brief de V17 completo (53 secciones) **no esta en el repositorio**. P6 a P10 solo se auditaron
  contra su nombre de una linea en `plan/00_intent.md` seccion 4.
- `reports/tables/e028_combinaciones.csv` se cita pero **no existe**.

---

## V37 · DIAGNOSTICO `P(POS | edad)` (spec escrita, nada ejecutado)

- **Spec**: `plan/V37_DIAGNOSTICO_POS_PLAN.md`. Intencion en `plan/00_intent_v37.md`. **Nada implementado.**
- **5 fases gateadas**: A perfil `P(tag|edad)` + cota de potencia -> G1; B estabilidad folds/md-lg/longitud/IC por grupos
  de duplicados -> G2; C contraste con D-026/V14/V31/P1a/P2 -> G3 (obligatorio, sin el no se cierra); D auditoria de
  novedad del canal de transiciones, **solo descriptiva** -> G4; E gate de decision. **Ninguna corre predictiva en V37.**
- **Ancla verificada**: `argmax(v30_prob * [1.0,1.1749,1.2721,1.163,1.001,0.8128])` = **0.6159732**; `fold_idx` se reproduce
  exacto con `StratifiedGroupKFold(5, seed 42)` sobre `dup_clusters` (164.018 grupos / 169.336 filas); 18 UPOS; 2.590.877
  tokens; 104 filas sin tokens; `SPACE` esta **dentro** de `pos_seq` (32.865 ocurrencias, 25.232 filas) y D-026 lo excluia.
- **Cota de potencia (decidido: va dentro, fase A, obligatoria)**: NB multinomial cerrado, proporciones por token, Laplace
  `alpha=1.0` fijo, priors empiricos,Leave-one-fold-out. Sin grilla, sin ajustar, sin fusion. Brazos A0 (mayoritario),
  A1 (proporciones), A2 (conteos crudos) y **A_perm** (control de fuga: debe caer en A0+-0,005). Sin CV anidada porque solo
  hay 114 estadisticos y ningun paso de seleccion.
- **Correccion de suelo**: clases casi balanceadas (ratio 1,0627), asi que el suelo **estricto** de la clase mayoritaria es
  **0,0485421**, no 0,1667. Toda cifra POS se cita con ambos suelos. **Rebalanceo NO APLICA.**
- **Auditoria de novedad del paso (d): veredicto (ii) VARIANTE DISFRAZADA.** No es (i) en sentido estricto (una matriz de
  transiciones indexada por clase es una interaccion de orden tres que un LogReg lineal sobre n-gramas 1-3 no puede
  expresar), pero tampoco (iii): P1a (D-041) ya midio el refinamiento por clase en la direccion dual con cross-fit
  (bloque 0,3997, fusion -0,0002, controles permutados **empatados**), y la integracion de un bloque de transiciones ya fallo
  dos veces (D-077 P2 -0,0086357; D-088 C1-A "el campeon ya la expresa mejor"). **El canal predictivo NO se ejecuta**; solo
  la medicion descriptiva D1 (`R` = divergencia entre clases / entre folds) y soporte de la matriz. Default: conservar el
  campeon. Habilitarlo exige autorizacion de Carlos con hipotesis nueva escrita.
- **Contraste honesto (fase C)**: `P(tag|edad)` es la **descomposicion por clase de las 18 features de D-026**; lo unico
  que V37 puede aportar sobre D-026 es desagregacion por clase, estabilidad y la cota de potencia (D-026 solo reporto el
  delta de integracion -0,0032). Que eso salga asi es un **resultado aceptable**, no un fracaso.
- **Artefactos**: `scripts/age_linguistic_13_preregistro_v37.py` + `experiments/age_linguistic_13_preregistro_v37.json` (congelado);
  `scripts/age_linguistic_14_diagnostico_pos.py` + `experiments/age_linguistic_14_diagnostico_pos.json`; 5 CSV en
  `reports/tables/`; informe `reports/v37_diagnostico_pos.md`; 5 figuras en **`reports/figures/`** (no existe `figuras/` en
  la raiz; `src/config.py` define `FIGURES = REPORTS/"figures"`). Runner con `--verificar` de **5 caminos** incluida la
  escritura en directorio temporal (leccion de D-086). Presupuesto: < 15 min.
- **Estado del arte** (10 refs verificadas, contexto no justificacion): Flekova & Gurevych 2013 (CEUR 1179, edad 0,53 EN /
  0,57 ES: *"in Spanish POS trigrams and quadrigrams are preferred"*); Cruz, Haro & Ortega 2013 (CEUR 1179, edad 0,62 ES con
  n-gramas POS 1-3); Posadas-Duran et al. 2015 (CEUR 1391/136: sintactico "not that successful for the age"); Modaresi,
  Liebeck & Conrad 2016 (CEUR 1609/970, edad 0,5179 ES: la distribucion L2 de POS **empeoro** los resultados);
  Johannsen, Hovy & Sogaard 2015 (CoNLL, `K15-1011`: para edad hay **menos soporte** que para genero); Flekova,
  Preotiuc-Pietro & Ungar 2016 (ACL, `P16-2051`: nouns/determiners/adjectives suben con edad, pronombres con signo
  **debil**); Morgan-Lopez et al. 2017 (PLoS ONE e0183537: 0,72-0,74 F1 pero **por usuario** con 200 tweets). El paso (d) NO
  cambia de veredicto por la literatura: cambia por D-041, D-077 y D-088, que son los que aplican a este corpus.

---

## V37 · RESULTADO Y CIERRE (D-089) · 2026-10-01

- **Veredicto**: `DIAGNOSTICO_CERRADO`. Rama `G2 falla -> DIAGNOSTICO_CERRADO por inestabilidad.
  Campeon conservado.` **Ninguna corrida predictiva**: el plan preveia que, si la auditoria de novedad
  daba (i) o (ii), el paso (d) no se corria; dio **(ii) variante disfrazada**.
- **Cota de potencia (el numero que decide)**: `A1(md) = 0.1720532` frente a **azar uniforme
  0.1666667** (+0.0054) y **suelo estricto 0.0485421**. Por debajo del margen de 0.01 que el propio
  diseno fijaba. `A2(md) = 0.2052270` (+0.0331738 sobre A1; **48.440 filas** cambian de prediccion:
  refuta la igualdad analitica `A2 == A1`, que solo se cumple sin priors). `A_perm(md) = 0.0799411` con
  criterio unilateral `<= 0.176667` e IC pareado `A1 - A_perm` = `[+0.0900353, +0.0942188]`, sin cero.
  Referencias: P2 `0.2750921` (D-077), P1a `0.3997` (D-041), campeon `0.6159732`.
- **El patron es real y reproducible**: transiciones POS de primer orden por edad, `D = 0.5695637`
  (IC95 `[0.37976, 0.684608]`, excluye cero), `R = 2.0645054` (IC95 `[1.5748, 2.3354]`), 2.421.645
  bigramas. **E4**: bajo permutacion **0 celdas** sobreviven los floors de V1..V4 frente a **20**
  reales -> los floors estan bien calibrados. `V1`/`V2`/`V6` cumplen; `V3` 14/20; `V4` 16/20; `V5` falla.
  Cierre formal por **G2**: 27/27 estables entre folds pero solo **20/27** de acuerdo md-lg.
- **Correciones de las 15 enmiendas que importan**: **E1** (el criterio `|A_perm - A0| <= 0,005` habria
  dado un STOP **falso** en 5 sitios: el NB permutado degenera hacia la clase mayoritaria, 159.653 de
  169.336 filas, no hacia el azar uniforme) · **E2** (`A2 != A1`) · **E3** (`A_perm_val` cubre el lado de
  la evaluacion) · **E7** (V5 degradado a sanity floor) · **E9** (sonda previa gateada en la corrida:
  `dif_A1 = 3e-08`, `dif_A2 = 0.0` frente a tolerancia `1e-06`; `A_perm` **no** se compara por
  tolerancia porque es estocastico) · **E11** (unidad del SE = tweet, no token; la inferencia la
  gobierna el bootstrap por grupos) · **E13** (B2 es sanity check de la proyeccion universal: ambos
  anotadores proyectan sobre las mismas 18 UPOS y coinciden solo el **16,3 %** de las secuencias)
  · **E14** (tres atribuciones bibliograficas corregidas).
- **Dos defectos de gate detectados en QA y corregidos**: (a) `cumple` de V2/V3/V4 era **imposible que
  fuera `True`** (las 88 celdas ajenas a `conjunta` valian `False`, luego `all()` nunca pasaba;
  el informe decia "20/20 cumplen" y a la vez `False`); (b) `dispersion` era el **maximo cruzado entre
  anotadores**, luego la columna del CSV no reconciliaba con `p_max - p_min` de su fila. Ahora es por
  anotador, con asercion que lo verifica en las 216 filas.
- **Leccion de D-086 pagada**: el autoejercicio de `--verificar` **mato 25 defectos reales antes de
  gastar presupuesto**, cuatro **silenciosos**: doble indexacion en `perfil_por_clase` (perfil por
  quintil), `pos` de clusters sin deshacer la permutacion (**fase D entera**), arrays indexados por
  bigrama con mascara por fila, y `n_filas_por_clase_y_quintil[q]` sobrescrito en el bucle de clases.
- **Verificaciones**: `--verificar` **5/5** (39 s) + **9/9** aserciones; `--humo` OK (4,4 s); corrida
  completa **89,5 s**. `ml-reviewer` APRUEBA CON RESERVAS; `ml-qa-engineer` PASA CON RESERVAS.
- **Campeon intacto**: `v30_prob` y los seis pesos sin escribir; sin submission; sin tocar test, holdout
  ni leaderboard. `artifacts/v37/a1_prob_oof.npy` persistido con SHA256 declarado (E8).
- **Estado para la siguiente sesion**: ciclo V37 **cerrado**. No hay ninguna familia legal de rasgos
  POS no medida con potencial (E15). El canal de transiciones queda **deshabilitado** salvo
  autorizacion expresa con hipotesis nueva escrita por adelantado. Nada pendiente salvo que Carlos
  pida (a) el informe de las dos lecturas para el profesor o (b) una hipotesis nueva con ruling del
  profesor sobre las reglas.

---

## Auditoria de fases lingüisticas (subagente, 2026-10-01, solo lectura)

Alcance: P0/P1, anotación, P2 POS-seq, P6 NER, P7 texto tipado, P3/P4/P5, dep_edges, V37.
Veredictos: **3 correctos con reservas, 1 defecto confirmado (media), resto correcto**.
- **Sin controles negativos degenerados** (el patrón V36-E1 NO se repite): P2 baraja filas y
  etiquetas intra-doc (no degenerado porque el vectorizador usa 1-3-gramas); P6/P7/P3-P5 barajan
  filas o el bloque B intra-fold; V37 permuta etiquetas con esquema declarado + FPR conjunta (0 vs 20).
- **Hashes reproducibles**: `tobytes()` solo en `age_linguistic_13_preregistro_v37.py:379` y sobre
  arrays float contiguos (`log_prior`/`log_theta`); el resto son sha256 de fichero. Sin el defecto V36.
- **Defecto confirmado (media)**: el brazo N1b (`age_linguistic_06_ner_estructura.py:82,168-171`)
  filtra spans con un tokenizador regex distinto del de spaCy; coincide con `n_tokens` en solo el
  56,1 % de los documentos (medido en 20.000), luego N1b NO es "N1 sin menciones/hashtags/URLs".
- **El 16,35 % de acuerdo md-lg es un artefacto de longitud, no un problema de modelo**: con
  acuerdo por token 86,269 % el esperado bajo independencia es 0,1930 > 0,1635 observado y sigue a
  la longitud (1-3 tok: 36,8 % obs / 44,3 % esp; 16-23: 9,9 % / 21,1 %). V37 lo corrige
  (`reports/v37_diagnostico_pos.md:98`); el informe original no.
- **Premisa no.extraída**: `meta.json` de md vs lg declara tag_acc 0,96118 vs 0,96147 y pos_acc
  0,98495 vs 0,98518 → "lg es mejor anotador" no se sostiene; ningún informe lo dice.
- **Otros (baja)**: `es_core_news_lg` sin pin en `requirements.txt:49`; sha256 `30d5d3cf…` citado en
  `DECISIONS.md:3405` no coincide con el prereg JSON en disco (`1d6d6573…`); `n_process: 4`
  hardcodeado en `age_linguistic_00_anotacion.py:127` pese al fallback de la línea 116; claim sin
  sustento "cota superior de lo que puede mover el canal" (comparativa:92-93); spec congelado
  conserva el claim retirado de Johannsen (`V37_DIAGNOSTICO_POS_PLAN.md:563,1087`) y `generoEncuentra`.
- Campeon intacto: 0 escrituras a `v30_prob.npy` ni a los pesos en todo el scope lingüistico.

## Auditoria de experimentos con redes neuronales (subagente, 2026-10-01, solo lectura)

Alcance: todo lo que entrena o integra una red. Localizados 4 scorpi ones torch (V32-E1, V18-P3/P4/P5),
2 sklearn MLP (E-019a/b, V19-B1) y 2(meta-)ensambladores que consumen un MLP (E-020, E-018).
Veredictos: **1 correcto con reservas (V32-E1), 3 correctos, 3 defecto confirmado (baja, V18-P3/P4/P5),
3 no son NN**. Campeon 0.6159732 intacto; ninguna escritura a `v30_prob.npy`.
- **Higiene exemplar en V32-E1**: 40 campos del pre-registro verificados contra el codigo + 7 SHA256 de
  entradas congeladas, escalador fit solo en fold-train (`:697`), split interno 90/10 (`:602`), fold-val
  jamas usado para parar (`:766-781`), combinador anidado fit en fold!=k (`:996-1003`), sondа de
  determinismo con max|delta p| = 0.0 (`experiments/v32_01_surrogate_macrof1.json:/determinismo`).
  ES estricto declarado por Carlos en el pre-registro (`v32_00_preregistro.py:21`) y consignado en
  `compliance.es_estricto: true`. Sin fuga: el meta y el arbitro usan OOF real, no in-sample.
- **Defecto confirmado (baja)**: V18-P3/P4/P5 no fijan `torch.manual_seed` en ningun punto, y
  `set_seed()` (`src/config.py:75-85`) solo siembra `random`/`numpy`. La inicializacion de `nn.Linear`
  queda fuera de control -> esos tres scripts no son reproducibles bit a bit. E-019 y V19-B1 si son
  reproducibles (`MLPClassifier(random_state=cfg.SEED)`).
- **Reserva metodologica (V32-E1)**: con lote 8192 sobre ~122k filas hay 15 pasos/epoca y el tope de 60
  epocas da <=900 pasos. Los 5 folds de A1 chocaron el tope (mejor epoca 54-59 de 60): el brazo decisional
  NUNCA convergio, mientras A2 (CE) si paro por ES (42-56 epocas). El veredicto NO_ATRIBUIBLE es correcto,
  pero atribuirlo "a la perdida" (`reports/v32_e1_surrogate_macrof1.md`, Control negativo) no se sostiene:
  A1 vs A2 esta confundido con el presupuesto de entrenamiento.
- **Diagnostico util**: el suelo mecanico del combinador anidado es ~-0.0009 (A3 con etiquetas barajadas
  da -0.00090, practicamente igual que A2 con CE -0.00093). A1 da -0.00342, o sea 0.0025 PEOR que un
  brazo sin informacion: el integrador sobre-pondera un miembro mas debil. Ninguna NN aporta senal
  no lineal: MLP solo 0.5989 (A1) / 0.6028 (A2) frente a 0.6160 del campeon y ~0.6005 de la LogReg
  de V12 sobre el mismo espacio. La no adopcion esta bien justificada; NO es una afirmacion de optimalidad.
- **E-020** no entrena la NN: consume el OOF real de E-019b y refit LogReg con leave-one-fold-out, sin
  fuga. Su decision "incluir el MLP" (+0.0066) se tomo sin IC bootstrap ni recuento de folds positivos,
  careciendo del gate que el proyecto adoptaria despues. Riesgo historico: la base era 0.5784.
- **No son NN** (confirmado): E-018 (LogReg), V17-P9-ARBITRO (LogReg binaria con protocolo anidado,
  umbral en bloque interior que el arbitro no vio), ADV-01..07 (LightGBM/NBSVM/holdout), `src/gpu.py`
  (guardian D-007, no entrena), `fase00_entorno.py` (sonda de entorno). V18-P5 si hace cross-fitting
  explicito del teacher OOF: el patron correcto de anti-fuga.

---

## AUDITORIA DEL HISTORIAL V1-V37 (D-090) · 2026-10-01 · CICLO CERRADO

- **Alcance**: las 195 filas de `experiments.csv`, V1 a V37. Auditoria de EVIDENCIA, no de opinion:
  7 revisiones independientes (nlp-specialist, ml-architect, dl-specialist, kaggle-specialist,
  ml-reviewer, ml-qa-engineer + verificacion propia del auditor). Cada hallazgo se confirmo contra la
  fuente primaria con ruta y linea antes de aceptarse. **Sin reejecutar, sin test/holdout/leaderboard,
  sin submissions, campeon intacto.**
- **195 filas NO son 195 experimentos**: **151 EJECUTADO** + 21 DOCUMENTAL + 12 PREREGISTRO +
  10 ENMIENDA + 1 BLOQUEADO. **0 artefactos ausentes.** Los 95 huecos iniciales eran rutas mal citadas
  (coma en vez de ` | `, CSV de `reports/` sin carpeta, prosa entre parentesis).
- **PROTOCOLO LIMPIO (verificado por AST sobre 308 scripts)**: solo 2 `fit()` con receptor test, ambos
  sobre OOF de train; cero calibracion de umbrales sobre test; cero uso del LB como validacion u
  objetivo; `id` nunca como feature; sin fuga de autor ni temporal; campeon bit-a-bit reproducible
  (SHA256 `29cbe6cc…`, 112.892 filas, 6 clases, 0 nulos); **cero adopciones sin pre-registro**.
- **K-01 (ALTA) - el techo de LB 0.70 de D-087 NO demuestra lo que afirma**: `v35_00g_techo_analisis.py:69`
  define `TAUS = (0.30, 0.35, 0.40, 0.45, 0.50)` -> la rejilla **empieza en 0.30 y sube** y el techo se
  tomo como su maximo. En `tau = 0.2930` (el punto que el propio D-087 usa como objetivo) el oraculo da
  **+0.0646 > +0.0626**. Base 0.6159732, 65.683 errores. **Si se sostiene (mas fuerte):** habria que
  acertar el ~16 % de los errores con `p(y_true) >= 0.293` = conocimiento casi-oracular. LB 0.70 sigue
  inalcanzable en la practica; **la prueba no**. Su IC95 [0.635, 0.653] **no esta en ningun artefacto**.
- **A-01 (ALTA) - multiplicidad de ~151 corridas sin cuantificar**: grep de multiplicidad / multiple
  comparisons / FWER / family-wise en los 4 documentos maestros = **cero**. `macro_f1_std` mediano
  0.0042 -> **SE ~ 0.00188**: gate `+0.0015` -> **~32 FP esperados**; `+0.0030` -> **~8**; Bonferroni 151
  exigiria `+0.0147`. **No invalida conclusiones** (4 adopciones con IC95 pareado excluyendo 0 +
  confirmacion externa), pero obliga a reformular: "ninguno paso el gate **y su bootstrap pareado
  excluyendo 0**".
- **Otros confirmados**: K-02 holdout de V12 **quemado** | K-03 registro de envios 15/12 vs **16
  puntuados** | N-02 **`es_core_news_lg` sin fijar** en `requirements.txt:49` (anotador de todos los
  resultados md/lg no instalable = **el mas accionable**) | N-01 `N1b` con indices desalineados (~55,8 %
  vs mi 56,11 %, sin impacto por estar bajo el gate) | N-03 16,35 % de acuerdo md-lg sin controlar
  longitud (por token **86,269 %**) | A-03 `max_iter=40` del MLP miembro del campeon con peso 0,25 |
  A-06 V18-P3/P4/P5 sin `torch.manual_seed` (`src/config.py:75-85` solo siembra random/numpy) | A-04 V32-E1
  atribuye a "la perdida" cuando la causa es el presupuesto de 60 epocas.
- **Correcciones aplicadas**: (1) `V37-DIAG-POS` escrita con **21 de 22 campos** -> reparada con el
  patron de `V36-E1`, recuperada del JSON **sin inventar texto** (mi 1a reparacion fue incorrecta y el
  QA la detecto: consumio el hueco equivocado e inserto texto nuevo). (2) SHA256 del pre-registro V37
  corregido de `30d5d3cf…` a `1d6d6573…`: **el JSON lo registra bien, luego era transcripcion y el
  pre-registro NO se muto**. (3) El clasificador afirmaba falsamente que V33/V34/V35 no se ejecutaron.
  (4) 3 checks tautologicos. (5) Fragmento en chino colado desde un subagente.
- **Notebook**: capitulo PARTE III, 326 -> 330 celdas, **las 326 originales byte-identicas**; las
  cifras del capitulo **se leen del inventario**, no escritas a mano. 5/5 checks.
- **Deuda documental declarada**: el notebook llega hasta V35; **V36 y V37 no tienen seccion propia**.
  4 ficheros de `submissions/` sin marca NO_ADOPTADO. `config/hardware.json` reescrito a mano difiere
  de `fase00_entreno.py`.
- **4 reservas abiertas y declaradas**: sesgo del inner split de Optuna sin medir **en la metrica**
  (medido sin impacto en overlap: 0,963 % ingenuo vs 1,030 % agrupado); `max_iter` del MLP sin revisar;
  IC95 del techo declarado; falta registro de autorizacion de los 4 envios posteriores.
- **Campeon INTACTO** 0.6159732487657944 (hash `3d77352486605971` + recomputo), submissions sin tocar
  (mtime maximo 2026-09-29), sin push/merge/rebase/reset. **Ninguna accion propuesta cambia el campeon.**
- **Artefactos**: `reports/auditoria_experimentos_v1_v37.md` | `reports/tables/auditoria_v1v37_inventario.csv`
  (197 x 24) | `experiments/auditoria_v1v37_inventario.json` | `scripts/auditoria_v1v37_inventario.py`
  (6/6) | `scripts/auditoria_v1v37_notebook.py` (5/5) | `plan/auditoria/0{0,1,2a,2b,2c,4,5}_*.md` |
  `plan/auditoria/notebook_capitulos.md` | acta **D-090** | filas `AUD-INV` y `AUD-REV`.
- **Reproducir**: `python scripts/auditoria_v1v37_inventario.py --verificar` (6/6, no escribe) y
  `python scripts/auditoria_v1v37_notebook.py --verificar` (5/5, no toca el notebook real).
- **Estado para la siguiente sesion**: nada pendiente. Las 6 acciones futuras propuestas (fijar `lg`,
  medir el inner split, cerrar la via MLP, etiquetar el IC95, anadir V36/V37 al notebook, registrar
  autorizaciones) requieren **autorizacion de Carlos** y no se ejecutaron.

---

## ACCIONES DE FOLLOW-UP DE LA AUDITORIA (D-091) · 2026-10-01

- **Cerradas (3 de 4)**:
  1. **`es_core_news_lg` fijado** en `requirements.txt:50` con el hash oficial de upstream
     `7c6c212715a12f31aacde3361754436945ff7376fb24cde57d0c277c9c9b050b`, **verificado descargando el
     wheel real (567.975.270 bytes)**. Cierra N-02, el hallazgo mas accionable de D-090.
  2. **IC95 del techo etiquetado como DECLARADO**, no estimado, en D-087 (`DECISIONS.md`) y en
     `PROJECT_STATE.md`, con la nota de que la prueba quedo refutada (K-01).
  3. **Sesgo del inner split: MATERIAL.** `scripts/auditoria_medir_inner_split_optuna.py` (5/5),
     5 folds, 169.336 filas, 164.018 clusters: inner ingenuo **3,907 %** de solape de cluster vs inner
     agrupado **0,000 %** (diferencia 0,039071, casi 2x el umbral declarado de 0,02). Consistente en
     los 5 folds (3,787 %-4,152 %). Afecta a 5 scripts: `v29_04`, `v29_06`, `v30_01`, `v30_02`,
     `v31_05`.
- **CORRIGE una medicion erronea propia de D-090**: alli registro "sin impacto medible (0,963 % vs
  1,030 %)". Media **texto exacto** en vez de cluster (los clusters agrupan por Jaccard) y comparaba
  contra cero teorico, imposible porque el inner train hereda clusters que el outer split permitio. La
  referencia correcta es el inner agrupado. El leakage real es ~4x mayor.
- **Alcance acotado**: el leakage estaba en la **SELECCION** de hiperparametros, no en la estimacion.
  La estimacion externa de V29/V30 uso el outer agrupado y es **valida**, luego **la cifra del campeon
  no cambia**. Lo que no es reproducible bajo esquema honesto es el proceso de seleccion.
- **PENDIENTE (accion 4)**: re-optimizar V29/V30. Coste real medido ~2 h 10 min (V29 5.152 s con 30
  trials/fold, V30 2.449 s con 15). **Re-optimizar NO es una mejora esperada: es una correccion cuyo
  signo es desconocido** y puede BAJAR, con precedente en `v30_02_ensemble.py:257` (mejor
  `corrected_cv` 0,6130051 vs 0,6134471 del promedio ponderado, D-066 lo reconoce). Requiere
  pre-registro con criterio de decision. Detalle: `plan/auditoria/07_accion_reoptimizar.md`.
- **Defecto del clasificador encontrado y fijado**: al anadir las filas `AUD-*`, su `resultado` es un
  RECUENTO que menciona "ENMIENDA" y "BLOQUEADO", y los regex las clasificaban asi. Anclados a `^\s*`
  y nuevo check **C4b**. El autocaso del inventario paso de **6 a 7 checks**.
- **Otro defecto mio**: las 3 filas nuevas se escribieron al principio del CSV en vez de al final; el
  validador de largo de campo lo cazo y se reordenaron, verificado byte a byte que las 197 previas
  siguen identicas.
- **Inventario**: 200 filas = 151 EJECUTADO + 26 DOCUMENTAL + 12 PREREGISTRO + 10 ENMIENDA +
  1 BLOQUEADO. 7/7 checks. Acta **D-091**.
- **Campeon intacto** 0.6159732487657944. Sin push/merge/rebase/reset.
