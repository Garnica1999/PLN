# V35-E1 - Intento 1: CORRIDA BLOQUEADA por bug de ambito (no es un resultado)

**Fecha**: 2026-09-30. **Runner**: `scripts/v35_01_ftrl_lambda1.py` (commit `8afaef0`, APROVED por
`ml-reviewer` sin bloqueantes). **Pre-registro**: `experiments/v35_00_preregistro.json` congelado con
**5 enmiendas pre-ejecucion** (`651084f`, `5152573`, `c9c5859`, `3f43beb`, `c07cc7f`).
**Veredicto de este intento**: `DEFECTO_DE_IMPLEMENTACION`. **NO** es un resultado del experimento.

## Que pasó

La corrida real termino con **exit 1** en **42.82 s** de los 1200 s declarados (3.6 % del
presupuesto), con RSS pico de 0.688 GB frente al tope de 20 GB. El crash fue:

```
UnboundLocalError: cannot access local variable 'anclaje' where it is not associated with a value
  scripts/v35_01_ftrl_lambda1.py:4232  (entrenar_brazo_fold, ensamblado del dict de retorno de A2)
```

**Causa**: `anclaje` se asigna **solo** dentro de `if clave == "a1":` (linea 4136, la verificacion
del anclaje en la pasada 1 del fold 0) pero se lee **incondicionalmente** en el `return` comun
(linea 4232, `"anclaje_lambda1": anclaje`). Por tanto **A2 es inalcanzable de forma determinista**,
con independencia de los datos: no es un fallo numerico, ni de puerta, ni de presupuesto, ni de
memoria, y no depende de la semilla.

## Por que A1 si llego a producir resultados

El orden dentro de `entrenar_brazo_fold` es: entrenar -> verificar anclaje (solo A1) -> ensamblar el
retorno. El ensamblado es lo ultimo, de modo que **A1 fold 0 completo y registrado**, y el crash
ocurrio al ensamblar el retorno de A2. Por eso el estado tiene `A1_ftrl_lambda1.folds_completados
= ["0"]` y `A2_sgd_presupuesto.folds_completados = []`.

## Que quedo medido antes del crash (evidencia, NO veredicto)

| Magnitud | Valor | Lectura |
|---|---|---|
| Anclaje de `lambda1` en runtime | mediana `0.23993793862904017`, diferencia **EXACTA 0.0**, `sha256 ec1d1a7b...` identico | la **enmienda 1** (nucleo de anclaje) funciona |
| A1 fold 0, `ll_train` p1 -> p3 | 10.058343 -> 15.401682 -> 19.159444 | divergencia, igual que sin L1 (19.174421) |
| A1 fold 0, `\|\|w\|\|_2` p1 -> p3 | 28.274594 -> 46.452124 -> **60.237202** | **razon 2.1304**, el mismo numero que falso la hipotesis en la sonda pre-ejecucion |
| Escasez efectiva p1 -> p3 | 0.951073 -> 0.980667 -> 0.980667 | el L1 mata ~1.9 % de coordenadas y la escasez CRECE |
| Fraccion con `\|z_j\| <= lambda1` (p2, p3) | **0.019333** | **no es 0** -> el criterio "L1 inerte" **NO** se cumple |
| `macro_f1_es` | 0.046077 en las 3 pasadas | colapso a la clase mayoritaria |
| Frecuencias fold-val de A1 | `[0, 0, 0, 0, 0, 1]` | colapso a una sola clase |
| G7/G8 en el mini (`razon max/min`) | **1.7907335321613125 > 1.0** | G8 fallaria (umbral 1.0) |
| `max_abs_suma_menos_uno` | 2.22e-16 | |
| Triangulo de sesgo de A1 | 1 camino, `x0 = 1.0`, desplazamiento por clase `[-2.8985, -3.0005, -2.9716, -2.9715, -2.9426, +6.5998]`, `coord_sesgo sha256 8d3711b0...` | sesgo correcto en A1 |

La lectura mecanica es **fuerte y coherente con la prediccion registrada**: el L1 esta **activo pero
insuficiente** (no inerte), y a pesar de matar coordenadas la norma de pesos **se duplica**, luego
la hipotesis de acotacion de D-086 queda falsada en la corrida real y no solo en la sonda previa.
**Lo que NO puede hacerse** (y no se hizo) es derivar de aqui el nodo del arbol: la rama
pre-declarada solo la puede emitir la puerta, y **la puerta no corrio**. Por eso
`experiments/v35_01_estado.json` de este intento tiene `puerta = None`, `triage = None` y
`rama_arbol = None`, y `identidad_lambda1_fold_0 = None` (los tres casos no se ejecutaron; solo
corrieron las sondas (k)-(m) de firma sobre el mini, con `reproduce = True` como firma y
`identidad numerica = False`, no aplicable entre problemas distintos).

## Triage

- **Pasaron**: 111/111 verificaciones de no-deriva; versiones; baseline recomputado `0.6159732`
  exacto; corpus de 169336 textos con sus tres hashes; ancla trivial exacta
  `0.04854212180687087`; auditoria AST de compliance (39 literales, 30 modulos) con **0 violaciones**,
  y 0 violaciones de lectura; sondas (a)-(i) **9/9 pass**; sondas (j), (k), (l), (m) publicadas como
  no-gate; `sha256` del vectorizador coincidente con el pre-registro; **anclaje verificado con
  diferencia EXACTA 0 y `sha256` identico**; TF-IDF del fold 0 ajustado **solo** en ES-train
  (121921 filas, 107792 coordenadas); restauracion del mejor `w` de A1 con `sha256` reproducido.
- **Fallo (1)**: ensamblado del retorno de A2, `UnboundLocalError`.
- **No alcanzado**: tabla `estado["mecanismo"]`, identidad de `lambda1` en el fold 0, OOF de A2,
  folds 1-4, segunda corrida identica (G5), `puerta`, `triage`, `rama_arbol`, informe, acta y
  anotaciones.

## Integridad y prohibiciones

- `artifacts/p33_scores/todos/manifest.json` = `8881a9f53e8613aa943681e7fd9a42a3a281aa66b5d7c796d13d2758c6226076`,
  **igual antes y despues**; arbol `todos` = `377aa775be3b342a4f8792ce5634706ce5e948d416d2a32c225e1ba42092d765`,
  **igual**; `p33_scores` completo = `3cb240ae6ea30772c2ea950d5858a36e77868c2c7275192e06e2fda148026370`,
  **igual**. Campeon V30 + plug-in **intacto**.
- **Sin submission**, `submissions/` sin cambios. Sin test, sin holdout, sin leaderboard. Sin datos
  externos, sin transformers, sin LLM, sin embeddings.
- **No se integro nada ni se hizo bootstrap**: `delta`, `bootstrap` e `integracion` **no existen** en
  el estado de este intento, y `motivo_de_no_calculo` tampoco se escribio, porque el ciclo nunca
  alcanzo el paso de decision. No es una omision: es que ese paso no se ejecuto.
- **No se derivo un segundo `lambda1`**, no hubo fallback y no se toco el pre-registro.

## Decision y tratamiento

El pre-registro permite **una unica re-emision, solo por `DEFECTO_DE_IMPLEMENTACION`, nunca por
cambio de hipotesis**. Este caso es exactamente eso: un bug de ambito en Python, sin relacion con la
hipotesis mecanica, con la re-emision acotada a corregir el defecto y re-ejecutar. **D-086 no puede
escribirse todavia como cierre**: la puerta nunca emitio nodo. El ciclo queda **INVALIDADO**.

La re-emission se hara **desde cero**, sin reutilizar el checkpoint de A1 fold 0 de este intento,
aunque el runner sea reanudable: asi **una sola version del codigo produce todos los artefactos del
resultado** y no se mezclan ficheros de dos versiones. El coste son ~110 s y la evidencia de este
intento queda preservada en `experiments/v35_e1_intento_1_bloqueado/`.

## Evidencia preservada de este intento

| Fichero | sha256 (16) |
|---|---|
| `fold0_A1_ftrl_lambda1.npz` | `26ce90c6c483baae` |
| `fold0_A2_sgd_presupuesto.npz` | `f580b546af3f1b4b` |
| `a1_ftrl_lambda1.npy` (OOF parcial) | `d218bb92a1cafc59` |
| `v35_01_estado_intento_1.json` | `8978fd32209e06a6` |
| `v35_01_ejecucion.log` | `a70ccbc7aafe6dfb` |

Bug **bloqueante #4** del ciclo (los tres previos: 17 lecturas `prereg` inexistentes,
`NODOS_ARBOL` sombreado, y dos `KeyError` de la sonda de asimetria, quedaron cerrados y validados
por `ml-reviewer` en `c07cc7f` y `3843e5d`).
