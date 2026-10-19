# P99 - Informe final: escalado del corpus a 60M tokens unicos

Fecha: 2026-10-13. Hipotesis H99-01 pre-registrada
(`p99_preregistration.md`, `p99_corpus_scaling_plan.md`) antes de construir y
entrenar. P98 permanece cerrado e intacto. V03 congelado.

## 1. Tokens y documentos unicos adicionales

- S99-01: **60.707.427 tokens / 340.219 documentos**; **+30.970.244 tokens
  nuevos unicos** (+174.197 documentos) sobre C98-1, que esta contenido
  integro (166.022 docs / 29.737.183 tokens, 0 faltantes).
- 0 coincidencias exactas con el train de TOXIC; 0 solapamiento con la
  validacion de lenguaje; 28.429 duplicados internos removidos de twitter
  (6 duplicados en mC4) en la dedup exacta.

## 2. Composicion del corpus

- Nominal prevista 60/25/15 (wiki/mC4/twitter) **no alcanzable con material
  local** (tope de mC4 = 10M; sin descargas, por regla). Desviacion registrada
  ANTES de construir/entrenar.
- Efectiva: **wikipedia 42.001.417 (69.2%) / mC4 9.999.717 (16.5%) / twitter
  8.706.293 (14.3%)**. El objetivo de twitter (9M) fue exacto antes de dedup.

## 3. Presupuesto equivalente

- Exposicion: ~121M tokens procesados por direccion (2 x 60.5M) vs 117M de
  C98-1 (4 x 29.3M) — equivalente por diseno.
- Coste real: **3.876 GPU-h** (tope 6) + ~0.15 GPU-h de intentos fallidos y
  correcciones, todos documentados. La referencia 30M no se reentreno.

## 4. Perplexity por dominio (validacion, misma particion de P98)

| Direccion | ep1 | ep2 | wiki | mC4 | twitter |
|---|---|---|---|---|---|
| LTR | 136.9 | 100.6 | 81.8 | 187.4 | 67.4 |
| RTL | 135.6 (corregido) | 98.3 | 81.2 | 181.4 | 63.5 |

Frente a C98-1 (94.8 / 91.7 tras 117M tokens): S99-01 queda ~6% arriba —
coherente con 2x diversidad y la mitad de repeticiones; el objetivo de
lenguaje se aprende correctamente en ambos brazos. (Errata: el gauge RTL ep1
original era invalido; corregido con el checkpoint preservado.)

## 5. Mejora 60M vs 30M

**No.** Canonica: fusion60 0.781586 vs fusion30 0.782266 (**-0.000680**).
Particion 47: 0.774784 vs 0.776783 (**-0.001999**). El doble de documentos
unicos, con exposicion constante, no mejoro la representacion downstream:
la empeoro marginalmente en ambas particiones.

## 6. Mejora frente a V03

- Canonica: +0.004375 (4/5 pliegues; IC por clusters incluye cero).
- Particion 47: **-0.000848** (3/5).
- Igual que la senal de 30M en su momento, la mejora canonica NO sobrevive.

## 7. Supervivencia a la segunda particion

**No.** Delta negativo en la particion 47 (pre-registrada de P98, hash
verificado). GATE_FAILED: incumplidas "positivo vs 30M" y "positivo en la 47"
(y neto no positivo en 47).

## 8. Complementariedad y slices

- Canonica: neto + vs V03 positivo; las ganancias se concentran donde V03
  sufre (tuits largos, hashtags, URLs); las slices negativas existen y se
  reportan en `p99_complementarity.csv`.
- Particion 47: complementariedad no persiste (neto no positivo), replicando
  el patron de D98-1b: el perfil por slices de una particion no se repite en
  la otra.

## 9. Evidencia para una prueba separada de arquitectura

**No.** El gate de escala fallo; P99 seccion 15 prohibe abrir la busqueda de
arquitectura como rescate. Sin cambios de dimensiones, capas, pooling ni
cabeza secuencial.

## 10. Ruta empirica hacia una mejora de leaderboard

Con la evidencia actual: **no respaldada**. Cuatro mecanismos genuinos
contextuales (30M canonica, 30M replica, 60M canonica, 60M replica) muestran
la misma firma: senal canonica de +0.004~+0.005 que no sobrevive a particion y
nunca supera con estabilidad a la referencia. V03 (LB 0.78306) sigue siendo el
mejor resultado verificado. 0.80 no es alcanzable "por tener mas tokens"; mas
documentos, mejor cobertura o perplexity comparable no implican mejor F1.

## Cumplimiento (seccion 16)

- transformers 0 | BERT 0 | RoBERTa 0 | LLM 0 | pesos de terceros 0 | labels
  externas 0 | pseudo-labeling 0 | texto de test en corpus 0 | reuso de
  holdouts 0 | tuning por leaderboard 0 | corpus no autorizados 0 (5 corpus de
  odio siguen en cuarentena).
- `tools/verificar_caracteres.py` OK; artefactos ASCII; P98 intacto (corpus
  9f562718..., checkpoints 472891a7.../cff68d45...).
- Registro: filas P99-00/11/12/13/14 (total 485); secciones 158-162 del
  notebook; sin commits; sin submissions.

## Decision

```text
H99-01 FALSIFICADA (GATE_FAILED)
HIPOTESIS DE ESCALA CERRADA CON EVIDENCIA
SIN RESCATES (no mas tokens, sin arquitectura, sin D98-2)
V03 = FROZEN | NO_SUBMISSIONS_AUTHORIZED
```
