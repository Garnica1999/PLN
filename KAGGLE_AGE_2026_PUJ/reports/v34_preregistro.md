# V34 - E1 - Pre-registro congelado: falsacion de la linea FTRL-Proximal

**Fecha**: 2026-09-30. **Estado**: CONGELADO antes de ejecutar (commit `3e97be7`).
**Baseline primario recomputado**: 0.6159732 (nested 0.6154643).

## Caracter del experimento

> Este NO es un intento de mejora: es una **FALSACION con la forma mas fuerte de la
> hipotesis**. La exploracion previa mostro que no existe alpha sano, y su valor es
> cerrar la familia online con evidencia **reproducible dentro del runner**, sin gastar
> mas gates de delta.

## Evidencia previa (exploracion metodologica, no es un grid)

- Script `v34_00b_exploracion_alpha.py` (sha256 `9d898ab8cf5908ef...`), **27 celdas**, presupuesto 30000 filas x 10 pasadas, semilla 7 para el subconjunto y `1000+pasada` para los ordenes.
- `H(prior)` del subconjunto = 1.7914536815044917; `ln 6` = 1.791759469228055.
- Mejor celda (alpha 0.176316, lambda2 1.0, power_t 1.0, denominador): ll_train = 1.7917637845854677 = uniforme + 4.3e-6.
- Peor celda (alpha 0.08, power_t 0.0): ll_train = 28.86405543322685, la firma de colapso de D-083.
- **0 de 27 celdas** bajan de `H(prior)`; el gemelo SGD con el mismo presupuesto llega a 0.15803953554913736: el presupuesto NO es el limite.
- Declaracion exigida: ningun umbral de G0-G4 se calibro con estas celdas; **no se repite dentro del runner** (seria tuning disfrazado).

## Mecanismo que fija el diseno

El punto fijo de FTRL es `w_j ~ alpha*sqrt(k)`: norma **no acotada**. SGD con schedule
`1/(alpha*(t0+t))` tiene pesos **acotados**. Con un solo alpha se reescala una direccion no
acotada: o el amortiguamiento proximal domina (V33-E1, razon `|sigma*w|/|g|` = 0.042,
inercia) o los pesos desbordan (D-083, divergencia).

## Regla de alpha (forma cerrada, 0 valores evaluados)

- `alpha* = 2*M/(d_norm * P_max^2)` con `M = ln(K-1) = 1.6094379124341003` (constante DECLARADA, linealidad: sensibilidad proporcional 1.0),
  `d_norm = sqrt(1 - sum_c prior_c^2)` del prior **MEDIDO del subconjunto ES-train** (NO el del fold-train completo: el ES consume el 10 %), y `P_max = 10` (ancla D-083).
- Con el prior de referencia (`d_norm = 0.912815416414756`) sale `alpha* = 0.035263162376364156`; el runner lo recalcula por fold y verifica reproducibilidad exacta (sonda g, exit 2 si no reproduce).
- `P_max` entra **a la vez** como presupuesto y como ancla: hay **un solo grado de libertad** (no es busqueda) y cambiarlo re-deriva `alpha*` (prohibido tras ejecutar).

## Presupuesto con early stopping interno (nunca fold-val)

- `P_max = 10` pasadas, `P_min = 3`, paciencia 2, mejora minima 0.0001; una permutacion distinta por pasada (`default_rng(20261010+fold+pasada)`).
- ES interno: `StratifiedShuffleSplit(test_size=0.1, random_state=20261010+fold)` sobre **filas de TRAIN**, metrica macro-F1; al parar se restaura el mejor `w` (seleccion sobre TRAIN).
- Simetria: ambos brazos reciben el mismo `P_max`, la misma particion (sha256), los mismos
  ordenes (sha256) y reportan su pasada de parada. El ES **no elige entre brazos**.

## Control y enmienda pre-ejecucion 1

- `SGDClassifier(log_loss, alpha=1e-6, learning_rate='optimal', shuffle=False, fit_intercept=True)`
  con `max_iter = 10` sobre la MISMA matriz y **sin columna de sesgo pre-anadida** (un solo camino de sesgo por brazo).
- Defecto corregido: D-083 y V33-E1 dar(on al control la matriz con sesgo pre-anadido Y
  `fit_intercept=True` (doble sesgo). Motivacion de correccion, no de conveniencia;
  verificado por sonda (max|gradiente de la coordenada de sesgo| y paridad con
  `intercept_` de A2).
- Asimetria declarada: `lambda2 = 1.0` (FTRL) frente a `alpha_reg = 1e-6` (SGD) es
  **NO-EQUIVALENCIA ESTRUCTURAL** (con `lambda2 = 1e-6` el L2 en acumulador y en
  denominador son identicos: no-op medido). El cierre se sostiene en **A1 frente a la
  DERIVACION**, no en A1 frente a A2, y se publica esa sonda.

## Puerta fail-fast G0-G6 (fold 0, antes de folds 1-4 y de integrar; ambos brazos)

- **G0 (nuevo)**: objetivo de la regla cumplido: A1 `M/2 <= D_acum <= 2*M`; A2 `ll_train <= H(prior)_ES - 0.01`.
- **G1**: banda `[0.05, H(prior)_ES - 0.01]`; el techo se ancla en el prior **medido** (el de D-084, `0.9*ln6`, quedaba por debajo del prior y fallaba por amplitud).
- **G2**: la mejor `ll_train` mejora en >= 1 de las primeras 3 pasadas.
- **G3**: macro-F1 standalone en fold-val >= 0.30 (anclas 0.0485421 y 0.4934789).
- **G4**: frecuencias por clase en `[0.5*prior_c, 1.5*prior_c]` del prior medido (la banda
  arbitraria [0.10, 0.30] de D-084 rechazo al gemelo sano: clase 5 al 42.6 %).
- **G5**: determinismo `max|delta p| = 0.0` exacto; **G6**: filas que suman 1.
- Fallo -> `NO_ATRIBUIBLE_BY_GATE`: STOP, sin fallback, sin segundo alpha, sin integrar,
  sin bootstrap; el fold 0 queda consumido-diagnostico y su fallo **no toca delta**.

## Nodo nuevo de cierre por mecanismo (condiciones)

- Se concede **solo** si (a) hay **triage formal registrado** que clasifica el fallo como
  `MECANISMO_DEFINITIVO` y no como `DEFECTO_DE_IMPLEMENTACION` (bug, sesgo duplicado,
  presupuesto truncado, ES mal configurado), decidido ANTES de elegir nodo; (b) la tabla
  de mecanismo esta **publicada dentro del runner**; (c) la lista exhaustiva de la familia
  esta registrada con el estado de cada miembro. Si un miembro no se midio queda
  `NO CUBIERTO`, nunca "cerrado". **One-shot**: reabrir exige acta nueva.

| Miembro de la familia | Estado | Evidencia |
|---|---|---|
| FTRL-Proximal propio, K=6 softmax (un solo juego de pesos) | CUBIERTO | V33-E1 (D-084, una pasada) y este ciclo (V34, multi-pasada con ES interno) |
| FTRL-Proximal OvR (6 sigmoides) | CUBIERTO | D-083 (diez pasadas, alpha 0.05, divergencia) |
| power_t (decaimiento) en {0.25, 0.5, 1.0} | CUBIERTO | exploracion previa (27 celdas): power_t con su alpha de margen da ll_train 1.8094 / 1.7929 / 1.7918, nunca bajo H(prior) |
| lambda2 en {.05, .2, 1, 1e-6} y L2 en acumulador vs denominador | CUBIERTO | exploracion previa (27 celdas); con lambda2 = 1e-6 ambas variantes L2 son identicas (no-op medido) |
| presupuesto de una pasada | CUBIERTO | V33-E1 (A1 y A2 con una pasada) y D-083 (diez pasadas) |
| FTRL-Proximal con features preentrenadas o normalizacion aprendida | EXCLUIDO | prohibido por las reglas (embeddings, preentrenados o entrenados) y sin cobertura historica |
| FTRL-Proximal con regularizacion L1 (lambda1 > 0) | NO CUBIERTO | no medido; queda EXCLUIDO del cierre de familia, no se declara cerrado |
| FTRL-Proximal multiclase con matriz de pesos por clase y L2 normalizado fuera de FTRL | NO CUBIERTO | no medido |

## Sondas obligatorias antes de entrenar (exit 4)

(a) vectorizado == lazo (1e-12); (b) raiz analitica con lambda 0 (1e-9); (c) TotalF1:average=Macro == macro-F1 (1e-6);
(d) sin NaN/inf; (e) log_loss multiclase propio == sklearn (1e-9); (f) filas que suman 1 (1e-9);
(g) **reproducibilidad de `alpha*`** (error 0 + sha256 de prior y alpha*, exit 2 si no reproduce);
(h) **estabilidad del ES** (dos corridas identicas: misma pasada de parada, sha256 de w identico, tolerancia 0);
(i) **simetria de presupuesto** (sha256 de particion ES y de ordenes iguales en ambos brazos);
(j) **tabla de mecanismo** publicada dentro del runner (`ll_train` y `D_acum` por pasada de A1 y A2, razon
  `|sigma*w|/|g|`): la falsacion se reproduce aqui, NO se importa de /tmp.

## Materiales congelados (SHA256, 16)

| Artefacto | SHA256 |
|---|---|
| corpus_archivo | `39c6c8270366fe15...` |
| corpus_textos | `53c1a348a7954f3e...` |
| vectorizador | `33311e5eddc0b0bb...` |
| manifest_canonico | `8881a9f53e8613aa...` |
| exploracion_script | `9d898ab8cf5908ef...` |
| exploracion_json | `92d24f8545c6b50b...` |
| exploracion_md | `4b11f38f82816a29...` |
| canonico_y_true | `b1c928f9b9f55755...` |
| canonico_fold_idx | `74d18330c40c3598...` |
| canonico_v30_prob | `3d77352486605971...` |
| canonico_ids | `d69ecb339f1dc57d...` |
| canonico_cand_pred | `2fd8ab0292e8b736...` |
| canonico_cand_prob | `57be29623543c261...` |

## Compliance, invariantes y coste

- 0 embeddings, 0 transformers, 0 LLM, 0 datos externos; sin test, sin holdout, sin
  leaderboard, sin submission; un script nuevo; campeon V30 + plug-in (LB 0.63575)
  intacto; NO se edita `v33_01` ni ningun script ejecutado ni se toca
  `artifacts/p33_scores/todos`.
- AST audit de imports y literales; lecturas de datos limitadas a `data/raw/age_train.csv` y las 4 particiones reservadas.
- Coste esperado ~590 s con presupuesto declarado de 1200 s; RSS < 1.5 GB con tope 20 GB medido con `ru_maxrss`; checkpoints por fold y por brazo con escritura atomica.
- Referencia de variabilidad de entrenamiento: 0.0008876.
- La exploracion previa tardo 454 s con RSS 0.56 GB.

## Enmienda pre-ejecucion 2 (2026-09-30)

- **Control A2**: `fit(max_iter=P_MAX=10, ...)` en una sola llamada pasa a `partial_fit` x 10, una por
  orden congelado de cada pasada. Motivo (medido en /tmp por el implementador): `BaseSGDClassifier._fit` hace
  `self.t_ = 1.0` en cada llamada, luego `fit(max_iter=1, warm_start=True)` repetido reinicia el schedule 10 veces y queda en
  `t_ = 401` en vez de `t_ = 4001`; el control recibiria un presupuesto de un orden de magnitud menor que A1. Con
  `partial_fit` el `t_` se acumula y se alcanza `t_ = 4001`, identico al de `fit(max_iter=10)`.
- **Desvio declarado**: `partial_fit` x 10 no es identico bit a bit a `fit(max_iter=10)` (4.5e-5 a 2 pasadas y 2.0e1 a 4 pasadas en un
  problema sintetico, por el contabilizado interno de la primera epoca). Es la unica via que permite el ES por pasada que el
  protocolo exige (`P_min = 3`, paciencia 2, restaurar el mejor `coef_`) y mantiene el presupuesto de 10 pasadas.
- No cambia gates, puerta, regla de alpha ni criterio de exito. Sin resultados: enmienda **pre-ejecucion**.

## Enmienda pre-ejecucion 3 (2026-09-30)

- **Fidelidad de la representacion**: el pre-registro de V34 congelo `token_pattern` con dobles backslashes (typo mio), de modo que el
  sha256 del dict del vectorizador era `33311e5eddc0b0bb...` en lugar del canonico `9bae157614458dbd...`. Se restaura
  la receta canonica de `v31_02_banco_representaciones.py`: `token_pattern = (?u)\b\w\b` y el sha256 vuelve al valor canonico.
- El check de no-deriva del runner (comparacion de su espejo con el JSON) fue quien lo detecto. No cambia ninguna otra
  constante, ningun gate, ni la puerta, ni la regla de alpha. Sin resultados: pre-ejecucion.
