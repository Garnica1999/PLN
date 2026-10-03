# P64 - Representacion del tweet largo y distribucion de la evidencia interna

Informe final de la fase. Fecha 2026-09-29T07:31:17+00:00. Cuatro experimentos, dos auditorias. 0 submissions, 0 Kaggle, 0 holdout, 0 test, 0 leaderboard.

## La pregunta de la fase, y la respuesta en dos mitades

> En los tweets largos y ambiguos donde V03 tiene el mayor margen residual, la
> informacion faltante esta en nuevas palabras o en como la evidencia textual esta
> distribuida y entra en conflicto dentro del mismo tweet?

**Las dos mitades son las dos mitades de la respuesta.**

1. **La estructura interna es REAL y medible.** El 80,88 por ciento de los documentos
   multi segmento de la region MUY_LARGA tienen segmentos que empujan la decision
   en sentidos OPUESTOS, y la entropia de la distribucion de evidencia dentro del
   tweet sube de 0,1048 en la region corta a 0,4565 en la muy larga. Eso es
   exactamente lo que una bolsa de n-gramas aplana al promediar, y es la hipotesis de
   la fase confirmada en su version descriptiva.

2. **Y no aporta, en ninguna parte.** El bloque no supera ni al control de
   capacidad ni al margen de V03, y no los supera tampoco en la region larga, que
   es donde P63 midio el margen. El margen tampoco mejora ahi, luego el problema de
   MUY_LARGA no es de representacion ni de confianza.

**Y por que no aporta, que es la parte que responde a la segunda pregunta.** V03
**ya lleva la estructura dentro de su confianza.** El margen de V03 solo explica sus
propios errores con AUC 0,770875 contra el acierto, y el mejor rasgo de evidencia
interna llega a 0,605302. La estructura no falta: esta dentro, y anadirla solo
agrega columnas que compiten por el presupuesto de un espacio de 21000
dimensiones.

## La cobertura, que es lo que abrio la fase

| criterio | filas con mas de una unidad | porcentaje |
|---|---:|---:|
| ventana fija 32 tokens, avance 16, de P59-07 | 142 | 9.6% |
| segmentacion por puntuacion fuerte, S1 | 751 | 50.6% |

**Factor 5.29x.** P59-07 explico su
fracaso por cobertura y no por concepto, y esto es la cobertura corregida. Un
factor de esa magnitud no es una variacion de parametro.

## Los cuatro experimentos

| ID | track | corrected_cv | delta vs V03 | decision |
|---|---|---:|---:|---|
| P64-00 | gap analysis | - | - | AUDITORIA |
| P64-01 | segmentacion y perfil EDA | - | - | PERFIL CERRADO. Continuar con descomposicion de evidencia |
| P64-02 | P4 evidencia segmental | 0.7520352870 | -0.0251761655 | CLOSED como representacion, y CONFIRMADA como descripcion. La estructura i |
| P64-03 | P5-P6 leave-one-out y contradiccion | 0.7656662317 | -0.0115452208 | CLOSED por redundancia con la confianza |
| P64-04 | P8 especializacion en la region larga | 0.7587037656 | -0.0185076869 | CLOSED. La evidencia no funciona ni donde P63 dijo que estaba el margen |

## El leave-one-out, que es el hallazgo mas fuerte de la fase

La contribucion de cada segmento al score de V03 es **con signo**, y eso separa un
segmento que empuja la decision de uno que la frena. P59 nunca lo computo.

| estrato | n | concentracion | conflicto | cambia de signo | implicita |
|---|---:|---:|---:|---:|---:|
| todas multi | 751 | 0.7080 | 0.0182 | 0.7168 | 0.6602 |
| V03 correcto | 480 | 0.7262 | 0.0156 | 0.6764 | 0.6880 |
| V03 incorrecto | 271 | 0.6757 | 0.0227 | 0.7884 | 0.6109 |
| error de alta confianza | 176 | 0.6881 | 0.0201 | 0.7538 | 0.6184 |
| CORTA | 50 | 0.7554 | 0.0202 | 0.6133 | 0.6900 |
| MEDIA | 247 | 0.7414 | 0.0202 | 0.6977 | 0.6639 |
| LARGA | 325 | 0.7114 | 0.0176 | 0.7108 | 0.6502 |
| MUY_LARGA | 129 | 0.6170 | 0.0148 | 0.8088 | 0.6666 |

El cambio de signo **crece con la longitud**, de 0,6133 en CORTA a 0,8088 en
MUY_LARGA, y la concentracion **cae** de 0,7554 a 0,6170. En los tweets largos la
decision no descansa en un segmento dominante sino en un reparto en el que las
piezas se contradicen.

Y ese bloque da una brecha contra el margen de apenas -0.0004650203, la mas pequena de toda la fase.

## Por estrato, y el detalle que revela mucho

| bin | n | G1 | G2 | juntos | aleatorio | margen |
|---|---:|---:|---:|---:|---:|---:|
| CORTA | 167 | -0.0082 | -0.0106 | -0.0101 | -0.0103 | -0.0101 |
| MEDIA | 602 | -0.0160 | -0.0022 | -0.0164 | -0.0096 | -0.0140 |
| LARGA | 547 | -0.0148 | -0.0138 | -0.0223 | +0.0018 | -0.0028 |
| MUY_LARGA | 167 | -0.0049 | -0.0063 | -0.0019 | -0.0312 | -0.0010 |

El **unico** brazo que gana en LARGA es el control aleatorio, +0,0018, y es
tambien el que mas pierde en MUY_LARGA, -0,0312. Eso es la firma de que en esa
region manda la varianza de la estimacion y no la informacion.

## El perfil segmental, cerrado con honestidad

| rasgo | AUC para predecir si V03 falla |
|---|---:|
| entropia_de_la_longitud | 0.589847 |
| n_tokens, REFERENCIA TRIVIAL | 0.600506 |

Ventaja: **-0.010659**. El
mejor rasgo segmental es PEOR que contar tokens. La asociacion entre muchos
segmentos y muchos errores es longitud con disfraz de estructura.

## Las siete familias cerradas sin gastar experimento

| familia | quien la cerro |
|---|---|
| ventana numerica fija | P59-07 |
| fusion de score local y global | P59-09 |
| geometria local como bloque | P59-02, V6-07 |
| max-pooling de evidencia local | P59-07 |
| ventana como canal sparse extra | V4-D01-D04, E2-09 |
| canal TF-IDF segmental | P59-07, V4-D01-D04 |
| umbral por estrato de longitud o de segmentos | E2-16, E2-20, V8-09 |

## La interseccion de errores

| bin | etiqueta | n | errores | alta confianza | en multi segmento |
|---|---|---:|---:|---:|---:|
| CORTA | HS | 167 | 13 | 9 | 5 |
| CORTA | TR | 167 | 6 | 5 | 3 |
| CORTA | AG | 167 | 14 | 11 | 5 |
| MEDIA | HS | 602 | 127 | 75 | 72 |
| MEDIA | TR | 602 | 77 | 49 | 30 |
| MEDIA | AG | 602 | 101 | 60 | 50 |
| LARGA | HS | 547 | 128 | 77 | 83 |
| LARGA | TR | 547 | 63 | 43 | 36 |
| LARGA | AG | 547 | 115 | 62 | 74 |
| MUY_LARGA | HS | 167 | 48 | 28 | 35 |
| MUY_LARGA | TR | 167 | 18 | 14 | 13 |
| MUY_LARGA | AG | 167 | 47 | 33 | 34 |

## Las seis correcciones propias de la fase

1. El subconjunto de columnas de P64-04 duplicaba el prefijo del rasgo, `seg_seg_max`,
   y el script abortaba en vez de ejecutar con columnas equivocadas.
2. La funcion de puntuar de P64-03 capturaba vectorizadores y clasificadores por
   cierre, y en la segunda vuelta de brazos habria usado los del ultimo pliegue.
3. La entropia de las contribuciones dividia por cero cuando todos los segmentos
   aportan cero, y el bloque entero llego al clasificador con NaN.
4. El AUC del margen de V03 se expresaba contra el ERROR, y como un margen alto
   significa ACIERTO, su AUC sale por debajo de 0,5 y la comparacion con un rasgo
   cuyo AUC sale alto por el motivo opuesto no significaba nada.
5. El veredicto de P64-02 se escribio con una afirmacion fija que no comprobaba los
   signos, y afirmaba que la interaccion existia con independencia de si la brecha
   de MUY_LARGA era positiva o negativa.
6. La conclusion de la EDA de P64-01 se corto a media frase.

## Estado

| | |
|---|---|
| experimentos | 4 de 22 principales, 0 de 6 reservas |
| registro | 359 filas, 359 identificadores unicos |
| correcciones propias | 6, con dos que cambiaron un resultado |
| cumplimiento | 26 comprobaciones, 0 fallos, por analisis AST |
| figuras | 13, ninguna oculta un negativo |
| submissions | 0 |
| Kaggle | 0 |
| holdout | 0 |
| test | 0 |
| reproducibilidad | R2, sin R4 |

## Cierre

```text
the internal evidence distribution of long tweets is real, measured and
reproducible, and V03 already carries it inside its own confidence

V03 remains the best verified leaderboard result.
```

No se afirma que V03 sea matematicamente optimo, ni que la competencia tenga un
techo alcanzable con los recursos permitidos. Se afirma exactamente lo que las
mediciones sostienen: la estructura interna del tweet largo es real, esta repartida
y es contradictoria cuanto mas largo es el texto, y es la que la bolsa de n-gramas
aplana. Y tambien que V03 ya la ve, luego el limite que queda en los tweets largos
no esta en su representacion ni en su confianza, sino en lo que un modelo de
bolsa de n-gramas puede o no puede hacer con un texto cuyo sentido depende de como
se reparten sus partes, y para eso harian falta transformadores, que las reglas de
la competencia prohiben.

## Artefactos

* `reports/p64_candidate_comparison.csv`
* `reports/p64_compliance.md`
* `reports/p64_contradiction_analysis.csv`
* `reports/p64_error_intersection.csv`
* `reports/p64_especializacion_larga.json`
* `reports/p64_especializacion_larga.md`
* `reports/p64_evidence_distribution.csv`
* `reports/p64_evidencia_segmental.json`
* `reports/p64_evidencia_segmental.md`
* `reports/p64_final_research.md`
* `reports/p64_final_summary.json`
* `reports/p64_hypothesis_gap_analysis.csv`
* `reports/p64_hypothesis_gap_analysis.json`
* `reports/p64_hypothesis_gap_analysis.md`
* `reports/p64_hypothesis_registry.csv`
* `reports/p64_leave_one_out.json`
* `reports/p64_leave_one_out.md`
* `reports/p64_master_results.csv`
* `reports/p64_reproducibility.md`
* `reports/p64_segment_analysis.csv`
* `reports/p64_segment_eda.json`
* `reports/p64_segment_eda.md`
* `figuras/p64/p64_01_v03_vs_candidatos.png`
* `figuras/p64/p64_02_delta_vs_v03.png`
* `figuras/p64/p64_03_rendimiento_por_longitud.png`
* `figuras/p64/p64_04_rendimiento_por_segmentos.png`
* `figuras/p64/p64_05_distribucion_longitud_segmento.png`
* `figuras/p64/p64_06_confianza_vs_conflicto.png`
* `figuras/p64/p64_07_concentracion_evidencia.png`
* `figuras/p64/p64_08_contribucion_leave_one_out.png`
* `figuras/p64/p64_09_rescate_vs_dano.png`
* `figuras/p64/p64_10_interseccion_errores.png`
* `figuras/p64/p64_11_rescate_alta_confianza.png`
* `figuras/p64/p64_12_consistencia_pliegues.png`
* `figuras/p64/p64_13_evolucion_candidatos.png`

> V03 remains the best verified leaderboard result.
