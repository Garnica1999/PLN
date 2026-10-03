# V32-06: Fase 7 de V32, context-gated n-grams

Generado por `experiments/v32_06_context_gated.py` (V32-06 (Fase 7 de V32: context-gated n-grams)).

**Estado `OOF_EVALUATED`. Alcance: version controlada de n-grama x length_bin para los K n-gramas mas frecuentes y estables del training fold**

## 1. La medicion previa, que es la que decide si se ejecuta

| magnitud | valor |
|---|---:|
| K n-gramas seleccionados | 100 |
| min_df | 5 |
| bins de longitud | 3 |
| nuevas columnas | 300 |
| vocabulario del training | 2673 |
| porcentaje del vocabulario | 11.22 % |
| cortes de longitud | [11.0, 20.0] |

los n-gramas aparecen en al menos MIN_DF documentos y en documentos de mas de un tramo de longitud.

## 2. Resultado

| etiqueta | F1 control | F1 challenger | delta |
|---|---:|---:|---:|
| HS | 0.7617 | 0.7487 | -0.0130 |
| TR | 0.8157 | 0.7986 | -0.0171 |
| AG | 0.7542 | 0.7394 | -0.0148 |

**corrected_cv 0.7772114525 -> 0.7622554418 | delta -0.0149560107**

train F1 0.9763 | gap +0.2141

rescates 120 | danios 172 | neto -52.

## 3. Lo que este script no hace

- No aplica el gating a todo el vocabulario.
- No genera submission. No lee el test.

Coste: 15.8 s, memoria 238 MB, 3 modelos, 0 submissions.

