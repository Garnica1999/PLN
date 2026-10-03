# Auditoria V1-V37 · Fase 2c - Protocolo de competencia y QA independiente

Dos revisiones sobre solo lectura: `kaggle-specialist` (cumplimiento de reglas y honestidad del
protocolo) y `ml-qa-engineer` (pasada independiente sobre las correcciones del auditor y los dictámenes
ajenos). Ningun artefacto de experimento fue editado por los agentes.

## Parte A - Protocolo de competencia (kaggle-specialist)

### Veredicto por ambito

| Ambito | veredicto | severidad |
|---|---|---|
| Fit sobre test (AST, 308 `.py`) | **correcto** | info |
| Vectorizadores ajustados solo en train | **correcto** | info |
| Uso de `age_test.csv` para elegir | **correcto** | info |
| Uso del leaderboard | correcto con reservas | media |
| Holdout usado para decidir | **defecto confirmado** | media |
| Calibracion CV -> LB con n=5 | **defecto confirmado** | alta |
| Techo 0.642 / su IC95 | correcto con reservas | media |
| Near-duplicates train/test | **correcto** | info |
| Fuga por metadatos | **correcto** | info |
| Reproducibilidad del campeon | **correcto** | info |
| Integridad de las 22 submissions | correcto con reservas | baja |
| Rotulo del generador CV en un prereg | **defecto confirmado** | baja |

### Lo que esta limpio, y conviene decirlo sin rodeos

**No hubo fit sobre test. No hubo calibracion de umbrales sobre test. No hubo uso del leaderboard como
conjunto de validacion ni como funcion objetivo. `id` nunca se uso como feature. No hay leakage de autor
ni temporal. Y el campeon congelado es bit-a-bit reproducible y valido en formato.**

Todo verificado por AST sobre el codigo real, no por las declaraciones del propio proyecto:

- Barrido AST: solo 2 `fit()` con receptor de tipo test, en `scripts/phase32_correccion_probabilistica.py:181`
  y `scripts/phase33_corrector_minimo.py:144`. Ambos calibran sobre **OOF de train**
  (`oof[train_idx, k]` con etiqueta `indices[train_idx] == k`). La afirmacion
  `n_llamadas_fit_sobre_test: 0` de `experiments/auditorias.json` queda **verificada independientemente**.
- Las ~20 instancias de `TfidfVectorizer` ajustan solo sobre train (`advanced_06_submission_v10.py:82-83`,
  `src/pipeline.py:186`, `experimento_e013_barrido_c.py:73-74`); el test solo recibe `.transform`.
- `scripts/fase23_auditorias.py:51` solo *lee* cobertura de duplicados; los scripts de techo declaran
  `test_tocado: false`.
- `scripts/fase01_auditoria.py:190` descarta `id`; `ID_COL` solo aparece en armado y validacion de
  submission.
- 4.2554 % del test cae en un cluster de train (`experiments/audit_01_resumen.json:65`), y los clusters
  se construyen sobre train+test **sin usar etiquetas** (`fase01_auditoria.py:252`).
- SHA256 `29cbe6cc...bd6c0` de `age_submission_plugin_candidate.csv` **coincide**: 112.892 filas mas
  cabecera, 2 columnas, ids 1…112.892 en orden, 6 clases validas, 0 nulos.

### Defectos confirmados

#### K-01 (ALTA) El techo de LB 0.70 (D-087) no demuestra lo que afirma

`scripts/v35_00g_techo_analisis.py:69` define `TAUS = (0.30, 0.35, 0.40, 0.45, 0.50)`. El "techo" se
tomo como el maximo de esa rejilla, que **nunca exploro por debajo de 0.30**, mientras la curva es
monotona creciente al bajar tau.

Reproduccion del auditor sobre `artifacts/p33_scores/todos/` (base 0.6159732, 65.683 errores):

| tau | filas corregibles | ganancia | supera +0.0626? |
|---|---|---|---|
| 0.30 (maximo publicado) | 10.384 | +0.0593677 | no |
| **0.2930 (el punto que el propio D-087 usa como objetivo)** | 11.298 | **+0.0646335** | **SI** |
| 0.29 | 11.725 | +0.0670929 | SI |
| 0.28 | 13.199 | +0.0755813 | SI |
| 0.25 | 18.345 | +0.1054505 | SI |

D-087 afirmaba que "un modelo perfecto con la informacion que el campeon ya tiene no llegaria". Es
falso: es un artefacto del redondeo de la rejilla. En el punto exacto que el propio D-087 define como
objetivo, el oraculo da **+0.0646**, un 3 % por encima del +0.0626 exigido.

**Lo que si se sostiene, y es un argumento mas fuerte**: un recalibrador que reorderara la salida de
V30 tendria que acertar el ~16 % de los errores con `p(y_true) >= 0.293`, es decir conocimiento
casi-oracular. La redaccion correcta es "el objetivo queda exactamente en el borde de lo que un
oraculo sobre la salida de V30 podria entregar", no "no llegaria".

**LB 0.70 sigue siendo inalcanzable en la practica; la prueba publicada no lo demuestra.**

#### K-02 (media) El holdout de V12 esta quemado

`experiments/phase19_holdout_VistaEstructura.json` declara criterio `"Δ ≥ +0,003 en el holdout
congelado"` y `"confirmado": true`. Eso es una **puerta de seleccion**, no una confirmacion. En
`DECISIONS.md:331-337` (D-017) el holdout aparece como "confirmado" en la justificacion de la adopcion
de `AGE_ADVANCED_V01`.

Agravante mitigado: el CV ya habia dado +0.0031 en 5/5 folds, luego el holdout fue un **segundo gate
aplicado sobre el mismo resultado**, no el arbitro unico.

#### K-03 (media) El registro de envios esta obsoleto y subcuenta el leaderboard

`experiments/phase12_auditoria_estado.json` (`n_registrados: 15`, `n_puntuados: 12`) es una foto del
2026-09-24 que solo llega a v12. Hay **4 envios posteriores puntuados** que no estan en ningun registro:

| Envio | LB | Fuente |
|---|---|---|
| V19 | 0.62578 | `RUNLOG.md:1200` |
| V20 | 0.62689 | `DECISIONS.md:1368` |
| V30 | 0.63302 | `experiments/v31_00_auditoria.json:62` |
| plug-in | 0.63575 | `experiments/age_plugin_cierre_final.json:21` |

Total real: **16 registros puntuados / 15 ficheros distintos subidos**, no 15/12.

#### K-04 (baja) Rotulo erroneo del generador de CV en un pre-registro

`scripts/v36_00_preregistro.py:338` anuncia `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
sobre filas de TRAIN, mientras `:340-341` aclara que los indices se **leen** de
`artifacts/p33_scores/todos/fold_idx.npy` y "NO se re-crean". Ese `fold_idx` es la particion canonica
`StratifiedGroupKFold`. **El comportamiento del codigo es correcto; solo el rotulo**, que induce a leer
una validacion naive que no ocurrio.

### Reservas

- **La calibracion no es un instrumento de extrapolacion, y su propia sigma no lo es.** Los 5 puntos no
  son i.i.d.: son 5 modelos encadenados sobre un test **fijo** de 112.892 filas, luego el LB es
  determinista y sigma = 0.001701 mide **idiosincrasia de familia de modelo dentro del rango**, no ruido
  de medicion. No hay base para suponer que se mantenga fuera del rango observado (que cubre el 77 % del
  span). D-087 ya lo admite textualmente; el defecto es que el resto del documento (el IC95 y el
  `P(LB>=0.70)~1%`) si se apoya en ella.
- **La pendiente es estable.** IC95 de la pendiente [0.941, 1.118]; el CV necesario para LB 0.70 se
  mueve entre 0.671 y 0.688, asi que la conclusion sobrevive a la incertidumbre de la pendiente. **No
  sobrevive al argumento del oraculo (K-01).**
- **Falta el registro de autorizacion de los 4 envios posteriores.** Para v01-v12 el `RUNLOG.md` dice
  explicitamente "Subida manual de Carlos a Kaggle". Para V19/V20/V30/plug-in **no existe linea de
  autorizacion** en `DECISIONS.md` ni en `reports/`. El brief exige autorizacion explicita: **no hay
  evidencia de infraccion, si ausencia de evidencia.**
- **Dos envios de otra competicion (TOXIC) aparecen en el registro de AGE** (2026-09-24 04:17, ambos
  `ERROR`). No rompieron reglas de AGE porque no puntuaron, pero el log de envios no esta aislado por
  competicion en un host compartido.
- **Cuatro ficheros sin marca de NO-ADOPTADO**: `age_submission_v10.csv`, `age_submission_final.csv`,
  `age_submission_v12_reproducible.csv` y `age_submission_v30_final_candidate.csv` conviven con el campeon
  sin marca en el nombre. Solo sus hashes los distinguen.
- **D-003 vs implementacion**: D-003 dice "5 folds x 3 semillas"; la particion canonica persistida es
  una sola repeticion con semilla 42. No es un defecto de integridad (la repeticion unica es mas
  conservadora para la varianza) pero la decision escrita y la ejecucion no coinciden.

## Parte B - QA independiente (ml-qa-engineer)

### Veredicto: PASA CON RESERVAS

Campeon V30 intacto; `scripts/`, `artifacts/p33_scores/` y `submissions/` sin modificacion;
`DECISIONS.md` y `PROJECT_STATE.md` byte-identicos (180 ocurrencias `D-xxx`, conjunto identico);
195 filas unicas.

#### Los cinco hallazgos ajenos se sostienen

| # | veredicto del QA | matiz |
|---|---|---|
| **N-01** (`TOKEN_RE` desalineado) | **CONFIRMADO** | cifra exacta 55.80 % (no 56.11 %): el auditor midi sobre 20.000 filas, el QA sobre las 169.336. Delta −0.31 pp, immaterial. |
| **N-02** (`lg` sin fijar) | **CONFIRMADO** | `grep -c es_core_news_lg requirements.txt` = **0**. |
| **N-05** (sha256 del pre-registro) | **CONFIRMADO, MATIZADO** | `experiments/age_linguistic_14_diagnostico_pos.json:8` registra el hash **correcto** `1d6d6573...` con `coincide_con_el_esperado: true`. Es error de transcripcion en los 2 documentos maestros, **no** prueba de que el pre-registro se mutara. |
| **A-01** (multiplicidad) | **CONFIRMADO** | 0 coincidencias en los documentos maestros. Los 7 hits de "FDR" en el notebook son **falso positivo** (payload base64 de PNG). |
| **A-06** (`torch.manual_seed`) | **CONFIRMADO, y material** | `v18_p3_margen.py:162`, `v18_p4_multiobjetivo.py:174,181`, `v18_p5_destilacion.py:212` construyen `nn.Linear` con el RNG global de torch sin sembrar. |

#### Defecto D-A que el QA encontro en MI correccion (CRITICO, ya corregido)

Mi primera reparacion de `V37-DIAG-POS` **consumo el hueco equivocado**: dejo la narrativa de metricas
en `accuracy_mean` (que debe ser numerica o vacia) e **inserte texto nuevo en `decision`** que no
existia en el registro.

**Correccion aplicada**: se adopto el patron correcto de `V36-E1` (columnas 13-16 vacias). La narrativa
de metricas pasa a `resultado` (que es lo que describe), `accuracy_mean` queda vacia, y `decision`
recupera **el cierre ya escrito** en `interpretacion`, sin inventar nada.

Verificacion: los 17 campos descriptivos de la fila son **identicos** al original de 21 campos; solo
cambian `accuracy_mean` (a vacio), `resultado` (recibe la narrativa), `decision` (cierre real) y
`duracion_seg` (89.5 desde el JSON). El CSV queda con 197 filas x 22 columnas, sin largos erroneos.

#### Checks del script de inventario que no pueden fallar

Probados por inyeccion de fallos en copia aislada, 7 escenarios:

- **C1**: la mitad `len(filas_out) == n_csv` es **tautologica** (mismo bucle, misma llamada). Solo la
  mitad `set(r) != set(COLS)` tiene poder.
- **C3**: `clasificar()` solo retorna literales ya dentro de `tipos_ok`, luego **no puede fallar por
  datos**. No valida que la clasificacion sea *correcta*.
- **C4**: solo mira `macro_f1_mean`. Una fila PREREG con `accuracy_mean` poblada pasaria.
- **C5**: el comentario "control independiente" era **falso**: reutiliza `resolver()` y
  `split_artefactos()`, y leia `artefactos_citados` **truncado a 400 chars**.
- Sin check del valor campeon 0.6159732 ni de la integridad de V30.

Los que **si** fallan de verdad: C2, C4, C5 y C6. Ademas `cargar_csv:178` aborta con `RuntimeError`
ante una fila de largo erroneo. El autocaso de esta auditoria lo demostro en la practica: **C5 fallo
de verdad** al anadir las filas `AUD-*` que citaban un `.md` aun no escrito.

#### Integridad confirmada

`--verificar` y `--resumen` **no escriben nada**: verificado con `sha256sum` de los 1.785 archivos del
arbol antes y despues → arbol identico. 6/6, exit 0. ASCII puro (0 bytes >127), compila, sin
`test`/`holdout`/`leaderboard`/`submissions`.

#### D-B (baja) `config/hardware.json` reescrito a mano

No coincide con lo que genera `scripts/fase00_entorno.py:134-136` (pierde `reglas_derivadas`,
`n_jobs`, `hilos_blas`; renombra `version_python`). Regenerable, sin ruptura funcional.

## Resumen de la Fase 2c

- **El protocolo de competencia esta limpio en lo esencial.** Es el resultado mas importante de toda
  la auditoria y conviene decirlo sin rodeos: **no hubo fuga hacia test ni hacia el leaderboard**, y el
  campeon congelado es reproducible bit a bit.
- **Lo que no esta limpio es la capa de razonamiento**: K-01 (el techo de LB 0.70), K-02 (holdout
  quemado), K-03 (registro de envios obsoleto) y K-04 (rotulo del generador de CV).
- **Y hay un defecto de marco analitico, no de ejecucion**: la multiplicidad de ~151 corridas sin
  cuantificar (documentado en `02b`).
- **Dos correccioneselasque el propio auditor aplico sobre si mismo**: la reparacion de `V37-DIAG-POS`
  (que primero fue incorrecta y el QA detecto) y el hash del pre-registro V37.
