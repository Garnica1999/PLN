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
