# V35 - E1 (via B) - Pre-registro congelado: `lambda1 > 0` sobre el nucleo FTRL de V34

**Fecha**: 2026-09-30. **Estado**: CONGELADO antes de ejecutar (commit `c0c762e`).
**Baseline recomputado**: 0.6159732 (nested 0.6154643); ancla trivial 0.0485421218068709 (clase 13-17).

## Que es y que NO es

- **Acta de reinicio**: **D-086**, anexada a `DECISIONS.md` al cierre. Miembro NO CUBIERTO: `lambda1 > 0`.
- **No es un reintento de `alpha`**: `alpha*` queda CONGELADO e identico al de V34-E1 (0.035262336465737745). Lo que cambia es el espacio de soluciones.
- **Miembros cerrados e INTOCABLES**: OvR (6 sigmoides); power_t (decaimiento); multi-pasada; lambda2 en {.05, .2, 1, 1e-6}; L2 en acumulador vs denominador; alpha.

## Anclaje de lambda1 (derivado, cero grid)

- `lambda1 := mediana(|z_j| > 0)` al final de la pasada 1 sobre **TRAIN** del fold 0 = **0.23993793862904017** (medido; `alpha*` = 0.035262336465737745, luego mediana/alpha = 6.8044).
- **Flag 1 resuelto**: `lambda1 = alpha` es no-op precisamente porque la mediana supera a `alpha` por un factor 6.80.
- **Flag 2 resuelto en negativo**: `lambda1` fijo **no acota** `||w||_2` (razon max/min = 2.1304 con la mediana), luego la hipotesis de acotacion queda **refutada por medicion**.
- Prohibido re-derivar `lambda1` tras ver las gates: un segundo `lambda1` seria grid.

## Puerta

- **G0-G6**: las de V34 **sin cambios** (techo anclado en `H(prior)_ES` - 0.01; banda de frecuencias en `[0.5*prior_c, 1.5*prior_c]` del prior **medido**).
- **G7 (L1 activo)**: escasez efectiva > 0 en todas las pasadas y no degradada mas alla de una tolerancia declarada de 0.02 respecto de la pasada 1. Anclada a la magnitud medida (NO al 0.5 arbitrario, NO al piso 0 tautologico). **Solo A1**.
- **G8 (acotacion)**: razon max/min de `||w||_2` <= **1.0**, anclada a la hipotesis misma (acotar implica no crecer), no a un factor arbitrario. **Solo A1**.
- **Rama pre-declarada**: G7 o G8 fallan -> `LINEA_ONLINE_CERRADA_POR_MECANISMO` (cierre DEFINITIVO de la familia); G7 y G8 pasan y G0-G4 fallan -> `HIPOTESIS_CONJUNTA_REFUTADA` (acota pero no aprende; no es cierre por conveniencia).
- **Asimetria declarada**: A2 es control de **PRESUPUESTO**, no gemelo mecanico; ninguna conclusion puede descansar en el contraste A1 vs A2 (queda contaminado).

## Sondas nuevas (no-gate) dentro del runner

- (k) caso `lambda1 = 0`, (l) caso `lambda1 = alpha`, (m) caso `lambda1 = mediana`: se reproducen aqui los tres casos del gemelo, porque la decision de mecanismo no puede descansar en evidencia importada de `/tmp`.

## Regla terminal one-shot

- Si `lambda1 > 0` falla, el segundo miembro NO CUBIERTO (pesos por clase con L2 normalizado fuera de FTRL) queda cerrado por declaracion si lambda1 > 0 falla, con su evidencia de techo.
- Una unica re-emision, solo por `DEFECTO_DE_IMPLEMENTACION`.

## Prediccion registrada antes de ejecutar

- con la sonda pre-ejecucion ya falsada la hipotesis de acotacion (razon ||w||_2 = 2.1304 con lambda1 = mediana), el veredicto mas probable es LINEA_ONLINE_CERRADA_POR_MECANISMO por fallo de G8. El ciclo NO puede terminar en PROMISING: alpha esta congelado e identico al que ya divergio

## Materiales congelados (SHA256, 16)

| Artefacto | SHA256 |
|---|---|
| sonda_lambda1_script | `4330c00a3f517467...` |
| sonda_lambda1_json | `e6554eb25269a3a4...` |
| nucleo_v34 | `cf443fc86bf291ca...` |
| manifest_canonico | `8881a9f53e8613aa...` |
| canonico_y_true | `b1c928f9b9f55755...` |
| canonico_fold_idx | `74d18330c40c3598...` |
| canonico_v30_prob | `3d77352486605971...` |
| canonico_ids | `d69ecb339f1dc57d...` |
| canonico_cand_pred | `2fd8ab0292e8b736...` |
| canonico_cand_prob | `57be29623543c261...` |
| corpus_archivo | `39c6c8270366fe15...` |
| corpus_textos | `53c1a348a7954f3e...` |
| vectorizador | `33311e5eddc0b0bb...` |

## Compliance e invariantes

- 0 embeddings, 0 transformers, 0 LLM, 0 datos externos; sin test, sin holdout, sin leaderboard, sin submission; un script nuevo; NO se edita ningun script ya ejecutado; `artifacts/p33_scores/todos` intocable.
- AST audit de imports y literales; lecturas limitadas al corpus y a las particiones reservadas.
- Coste esperado ~51 s con presupuesto declarado de 1200 s; RSS con tope 20 GB medido con `ru_maxrss`; checkpoints por fold y por brazo con escritura atomica.
