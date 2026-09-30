# P2 - canal POS-secuencia contra el campeon adoptado

**Fecha**: 2026-09-29 | **Pre-registro**: `reports/age_linguistic_pos_preregistro.md` (congelado antes de ejecutar) | **Baseline primario**: 0.6159732 (regla de la submission, recomputado)

## Veredicto

| Anotador | Canal solo (macro-F1) | delta integracion | folds+ | IC95 delta | delta control barajado | Veredicto |
|---|---|---|---|---|---|---|
| md | 0.2750921 | -0.0086357 | 0/5 | [-0.0098444; -0.0074067] | -0.0039528 | **NEGATIVO** |
| lg | 0.2725490 | -0.0084168 | 0/5 | [-0.0096703; -0.0072446] | -0.0041620 | **NEGATIVO** |

## Medicion 2 - complementariedad contra las decisiones del campeon

| Anotador | rescate unico | rescate compartido | danio | neto | rescates unicos de alta confianza | tasa sobre errores de alta confianza |
|---|---|---|---|---|---|---|
| md | 12,842 | 34,996 | 68,657 | -55,815 | 1,827 / 9,780 | 0.18681 |
| lg | 12,828 | 34,442 | 69,211 | -56,383 | 1,773 / 9,780 | 0.18129 |

## Medicion 3 - integracion neutral (alfa 0.5) y controles

| Anotador | brazo | delta vs baseline | folds+ | filas que cambian |
|---|---|---|---|---|
| md | real | -0.0086357 | 0/5 | 19,373 |
| md | barajado_filas | -0.0039528 | 0/5 | 13,289 |
| md | barajado_intra | -0.0080310 | 0/5 | 18,433 |
| lg | real | -0.0084168 | 0/5 | 19,167 |
| lg | barajado_filas | -0.0041620 | 0/5 | 13,513 |
| lg | barajado_intra | -0.0071584 | 0/5 | 18,241 |

## Lectura y decision (arbol pre-registrado)

### Anotador md: NEGATIVO

- delta de la integracion: -0.0086357 (ratio sobre la variabilidad 8,9e-04: 9.73);
- bootstrap IC95: [-0.0098444; -0.0074067], P(delta>0) = 0.0000;
- control barajado de filas: -0.0039528; control intra-documento: -0.0080310;
- canal solo: 0.2750921; rescate unico 12,842 vs danio 68,657.

### Anotador lg: NEGATIVO

- delta de la integracion: -0.0084168 (ratio sobre la variabilidad 8,9e-04: 9.48);
- bootstrap IC95: [-0.0096703; -0.0072446], P(delta>0) = 0.0000;
- control barajado de filas: -0.0041620; control intra-documento: -0.0071584;
- canal solo: 0.2725490; rescate unico 12,828 vs danio 69,211.

## Limites declarados

- Alfa = 0.5 es el punto neutro pre-registrado; ningun parametro se ajusta mirando el
  resultado. La conclusion vale sobre ESTA integracion y este canal.
- El test no se ha tocado; el holdout tampoco. No hay submission.

### Nota de lectura de la medicion 2 (anadida tras ejecutar; los numeros no cambian)

El rescate neto negativo (-55.815 / -56.383) **no es por si solo** la evidencia de redundancia:
un clasificador mas debil que el campeon lo produce por construccion (leccion ya documentada en
P32). Lo que decide esta medicion es el **delta de la integracion** (medicion 3): negativo, con
IC95 enteramente bajo cero, y con el control de filas rindiendo **mejor** que el canal real
(-0.0039528 frente a -0.0086357 en md; -0.0041620 frente a -0.0084168 en lg). El canal tiene
senal propia (0.2751 aislado, muy por encima del suelo 0.1667) y rescata 1.827 de los 9.780
errores de alta confianza del campeon, pero bajo la integracion neutral no aporta atribucion
positiva: su peso del 50% diluye al campeon mas de lo que sus aciertos unicos compensan. El
modo de fallo es de **integracion (dilucion)**, no de ausencia de informacion.
