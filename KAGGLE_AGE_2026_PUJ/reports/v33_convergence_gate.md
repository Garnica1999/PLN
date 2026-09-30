# V33 - E1 - Puerta de convergencia fail-fast (G1-G6)

Documento normativo de la puerta. Se congela con el pre-registro; sus umbrales estan
**DERIVADOS** (no calibrados con datos) y las sondas en `/tmp` solo sirven para
programar la regla: si una sonda fijara un umbral, seria tuning y esta prohibido.

| Gate | Criterio | Anclaje | Tolerancia |
|---|---|---|---|
| G1 | 0.25 <= log_loss multiclase de train <= 0.60 | uniforme `log(6)` = 1.791759; la cota alta caza la divergencia 11.72 -> 23.03 de D-083 | 1e-9 |
| G2 | log_loss no creciente en 25 %, 50 % y 100 % de la pasada | caza la firma de D-083 | 1e-6 |
| G3 | macro-F1 standalone en el fold-val >= 0.30 | trivial 0.0485421 (clase 0) y gemelo SGD de D-083 0.4934789 | 1e-9 |
| G4 | frecuencia predicha por clase en [0.10, 0.30] | proporciones reales del train | 1e-9 |
| G5 | max|delta p| = 0.0 entre dos corridas identicas | determinismo CPU esperado exacto | 0 |
| G6 | max|sum_c p_c - 1| <= 1e-9 | softmax de 6 logits | 1e-9 |

## Cuando corre y que pasa si falla

- Corre dentro del **fold 0**, antes de los folds 1-4 y antes de integrar.
- Se aplica **igual a A1 y a A2**: si el control falla, tambien se detiene (asi el
  diagnostico distingue 'el presupuesto no alcanza' de 'el optimizador no converge').
- G3 y G4 leen el fold-val del fold 0, que es parte de la validacion canonica: el
  fold 0 queda registrado como **consumido-diagnostico** y su fallo **no toca delta**.
- Si falla cualquiera: veredicto `NO_ATRIBUIBLE_BY_GATE`, **sin fallback**, **sin
  segundo alpha** (reintentar seria grid), sin integracion y sin bootstrap.

## Sonda de atribucion obligatoria

razon media |sigma*w| / |g| por coordenada en el fold 0. el control cambia dos variables (alpha y regularizacion del intercepto): la razon sigma/g es la que desambigua la causa (correccion inflada frente a multi-pasada). Consecuencia si no se publica: el veredicto queda capado a MARGINAL.

## Enmienda pre-ejecucion 1 (2026-09-30)

- La banda de **G1** pasa de `[0.25, 0.60]` a `[0.05, 1.6125835223052496]`.
  Motivo: era inalcanzable para un modelo sano (el gemelo sano de D-083 dio log_loss multiclase
  1.3568) y garantizaba `NO_ATRIBUIBLE_BY_GATE` sin medir nada. Derivacion del suelo y del techo
  en la tabla. Sin resultados: pre-ejecucion.
- El control **A2** fija `shuffle=False` para consumir el orden congelado tal cual.
