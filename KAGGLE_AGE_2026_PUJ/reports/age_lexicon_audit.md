# Auditoría de la fase de lexicones (P8)

**Pregunta**: ¿aporta un pequeño número de lexicones lingüísticos generales una señal de registro, etapa vital o dominio temático que complemente a V12?

**Gate pre-registado**: delta >= +0,003 y >= 4/5 folds positivos, y superar al control de permutacion. **Holdout**: agotado, no se ha reutilizado. **Leaderboard**: no se ha consultado. **Sin** embeddings, transformers, modelos externos, vectorizadores aprendidos nuevos, corpus/labels externos ni etiquetas de test.

## 1. Qué se construyó

- **24 subcategorías** en tres familias: registro (estilo), etapas de vida (contexto vital) y dominios temáticos (contenido).
- 1588 entradas declaradas -> **1359 efectivas** tras el cribado de aplicabilidad al corpus (frecuencia mínima 3 en el train, sin usar etiquetas).
- 1081 tokens únicos; 131 tokens ambiguos descartados (DD-5).
- Cobertura de tokens por familia: registro 5.72%, vida 0.62%, temas 10.04%.
- Subcategorías retiradas por quedar sin vocabulario utilizable: ninguna (mínimo 5 entradas efectivas).

## 2. Control lexicon-only: ¿existe la señal? (§14)

Suelo de referencia: azar 0,1667 y clase prior 0,0485 (clases casi balanceadas). V12 alcanza 0,6004.

| Familia | Columnas | Filas con alguna señal | LogReg | LinearSVC | Lectura |
|---|---|---|---|---|---|
| registro | 19 | 55.76% | 0.152 ± 0.0014 | 0.1483 ± 0.0021 | por debajo del azar: señal muy débil o nula |
| vida | 16 | 6.25% | 0.0731 ± 0.0011 | 0.0729 ± 0.0009 | por debajo del azar: señal muy débil o nula |
| temas | 37 | 58.02% | 0.2033 ± 0.0022 | 0.2018 ± 0.0022 | supera el azar |

## 3. Redundancia (§12)

- 3 de 72 columnas superan |rho| >= 0.8 contra alguna variable existente (3 contra las densas, 3 contra las 279 features de V12; correlacion maxima 0.966).
- El bloque **no** es una reformulación de lo que V12 ya tiene: su contenido es nuevo para el modelo, aunque sea débil.

## 4. Fusión con V12 y control de permutación (P6, DD-6)

Referencia V12 reproducida en la misma sesión: **0.6004** ± 0.0042 (per fold [0.6007, 0.605, 0.6032, 0.6004, 0.5928]).

| Candidato | Columnas | macro-F1 | Δ | Folds+ | Control permutado (media/max) | Ventaja sobre control | Gate |
|---|---|---|---|---|---|---|---|
| V12 + Registro | 16 | 0.6005 | +0.0001 | 4/5 | 0.6005 / 0.6005 | +0.0000 | no |
| V12 + Vida | 16 | 0.6004 | +0.0000 | 2/5 | 0.5994 / 0.5994 | +0.0010 | no |
| V12 + Temas | 37 | 0.5995 | -0.0009 | 2/5 | 0.5997 / 0.5997 | -0.0002 | no |
| V12 + Registro + Vida + Temas | 69 | 0.5986 | -0.0018 | 2/5 | 0.5965 / 0.5965 | +0.0021 | no |

**El control de permutación es la pieza decisiva**: un bloque real que no supere a su propia nula (mismo número de columnas, filas permutadas) no aporta señal, aunque mejore la media.

## 5. Complementariedad y error (§19-§21)

- **__registro**: rescate 5541, daño 5489, **neto +52**, modificadas 10.274%, fronteras centrales {'18-24|25-34': -150, '25-34|35-49': -6}.
- **__registro___vida___temas**: rescate 5782, daño 6121, **neto -339**, modificadas 11.100%, fronteras centrales {'18-24|25-34': -75, '25-34|35-49': 193}.
- **__temas**: rescate 5396, daño 5548, **neto -152**, modificadas 10.344%, fronteras centrales {'18-24|25-34': -75, '25-34|35-49': 73}.
- **__vida**: rescate 5427, daño 5380, **neto +47**, modificadas 10.129%, fronteras centrales {'18-24|25-34': -132, '25-34|35-49': 31}.

## 6. Veredicto por familia (§28)

| Familia | Resultado | Decisión |
|---|---|---|
| Registro | 0.6005 (Δ +0.0001, 4/5) | A - negativa: cerrada |
| Vida | 0.6004 (Δ +0.0000, 2/5) | B/D - señal por encima de la nula pero sin llegar al gate: cerrada |
| Temas | 0.5995 (Δ -0.0009, 2/5) | A - negativa: cerrada |
| Registro | 0.5986 (Δ -0.0018, 2/5) | B/D - señal por encima de la nula pero sin llegar al gate: cerrada |

## 7. Respuesta a la pregunta científica (§35)

> ¿Los lexicones proporcionan una dimensión lingüística de edad que V12 no captura?

**No, de forma explotable.** El detalle por dimensión:

- **registro**: control lexicon-only 0.1520 (cobertura 5.72% de los tokens) -> sin señal medible por sí misma (por debajo del azar de 0,1667); fusión con V12: Δ +0.0001 con ventaja sobre el control +0.0000.
- **vida**: control lexicon-only 0.0731 (cobertura 0.62% de los tokens) -> sin señal medible por sí misma (por debajo del azar de 0,1667); fusión con V12: Δ +0.0000 con ventaja sobre el control +0.0010.
- **temas**: control lexicon-only 0.2033 (cobertura 10.04% de los tokens) -> señal débil pero real por sí misma: el bloque supera el azar sin V12; fusión con V12: Δ -0.0009 con ventaja sobre el control -0.0002.

## 8. Criterio de cierre (§29)

Se aplica **A** (ninguna familia alcanza el gate), reinforce **B/D** (el bloque no es redundante, pero su señal no se traduce en mejora) y **C** (donde el Δ nominal no sobrevive al control de permutación). **La línea se cierra.** V12 permanece congelado y no se genera ninguna submission.

## 9. Auditoría de recursos (§32)

- `transformers`, `sentence-transformers` y `torch` están instalados en el venv y listados en `requirements.txt`; **el uso en el camino de modelado es 0** y el único import del proyecto es una detección de capacidad de GPU (`src/gpu.py`).
- Embeddings, Word2Vec, FastText, GloVe, sentence-transformers, modelos de lenguaje: **0 usos**. Corpus o labels externos: **0**. Etiquetas de test: **0**.
- Los únicos recursos externos Allowed en esta fase son listas léxicas simbólicas elaboradas en el proyecto, versionadas con SHA256 en `reports/age_lexicon_catalog.csv`.
- El cribado de frecuencia mínima usa el vocabulario del train completo: es **selección no supervisada y transductiva**, sin usar etiquetas en ningún punto; se declara por transparencia.

## 10. Incidente de versionado y recuperacion (2026-09-25)

Durante la ejecucion de P6 el estado no versionado de la fase (`lexicons/**`, `src/lexicons.py`,
`src/lexicon_lists.py`, `scripts/phase36..41`, informes de P0-P5) fue apartado a un **stash de git**
(`stash@{0}`, "WIP AGE lexicones fases 36-41") mientras otra sesion del repositorio compartido
trabajaba en `KAGGLE_TOXIC_2026_PUJ`.

- **Recuperacion completa**: los 44 ficheros se han extraido del stash y verificado. Los bloques
  `registro`, `vida` y `temas` regenerados desde las listas originales son **identicos bit a bit**
  a las matrices que consumio P6 (19, 16 y 37 columnas respectivamente).
- **Sin coste de recomputacion**: P6 **no** ha necesitado reejecutarse, porque las matrices que usa
  son exactamente las originales; sus resultados y los de P7 son validos tal cual.
- **Un bug encontrado al recuperar**: `scripts/phase40_analisis_error.py` tenia un error de
  ortografia (`Informe`/`informe`) que hacia fallar la ejecucion; corregido y reejecutado, reproduce
  las mismas cifras de rescate y dano.
- **Copias de seguridad**: el material recuperado quedo ademas en `/tmp/age_stash/` y
  `/tmp/age_rescate/`, y las cifras de P6 en el commit `ed783cb`, anterior a la regeneracion.

Detalle completo en `INCIDENTE_GIT_2026-09-25.md`.
