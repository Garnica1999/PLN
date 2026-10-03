# A_cost_sensitive_downweight - Estado final

Generado por `experiments/p69_03_cierre_downweight.py`. Fecha 2026-09-29T17:54:08+00:00.

## Las tres capas, que no se confunden entre si

| capa | corrected_cv | delta | pliegues | nota |
|---|---:|---:|---:|---|
| ORIGINAL | 0.7854160387 | +0.0082045862 | 5 de 5 | 2026-09-25, nunca enviado, nunca auditado |
| REPRODUCED | 0.7854160387 | +0.0082045862 | 5 de 5 | **R3 exacto**, diferencia 3,2e-11 |
| SEGUNDA PARTICION | - | +0.0105253130 | 5 de 5 | semilla 20260928, particion nueva |

Los deltas por pliegue coinciden en las dos primeras capas, valor a valor:

```text
canonica  +0.002243 +0.001592 +0.003378 +0.023607 +0.010356
segunda   +0.026302 +0.004643 +0.002975 +0.020868 +0.003209
```

## La incertidumbre, con la unidad declarada

El bootstrap historico remuestrea **filas**. Con 1415 grupos de near-duplicados sobre
1483 filas, eso trata duplicados como independientes y **subestima** la incertidumbre.
El proyecto usa remuestreo por **grupos** desde P66.

| unidad | IC95 | ancho | incluye el cero |
|---|---|---:|---|
| por filas (historico) | [-0.0002247696, +0.0116880198] | 0.011913 | True |
| por grupos (proyecto) | [-0.0006906975, +0.0118175641] | 0.012508 | True |

Las dos incluyen el cero. La de grupos, que es la correcta, lo incluye mas.

## Los gates

| gate | pasa |
|---|---|
| reproducido_exactamente_R3 | **SI** |
| cinco_de_cinco_en_la_canonica | **SI** |
| cinco_de_cinco_en_la_segunda | **SI** |
| misma_direccion_en_la_segunda | **SI** |
| bootstrap_por_filas_compatible_con_mejora | NO |
| bootstrap_por_grupos_compatible_con_mejora | NO |
| compliance_sin_fallos | **SI** |

**5 de 7.**

## El estado final

```text
MARGINAL

el candidato se reproduce EXACTAMENTE y confirma su direccion en una segunda particion con 5 de 5 pliegues, pero el intervalo de confianza incluye el cero bajo las DOS unidades de remuestreo. Con un limite inferior de -0.000691 por grupos, la mejora no es distinguible del ruido a este tamano de muestra. CASO C del encargo
```

### Lo que se puede afirmar

```text
el resultado es REAL y AUDITABLE: se reproduce exactamente, a 3,2e-11
el mecanismo esta documentado: downweight por violacion, no un meta-modelo
la mejora NO es un artefacto de la particion: se repite con 5 de 5 en otra
el tamano es pequeno y su intervalo incluye el cero bajo las DOS unidades
```

### Lo que NO se puede afirmar

```text
NO que la mejora sea distinguible del ruido a este tamano de muestra
NO que sea adoptable: no hay artefacto de prediccion y no hay veredicto externo
NO que el downweight por si solo explique el efecto: va combinado con el barrido
   anidado del multiplicador de class weight, y separarlos seria otra hipotesis
```

## La correccion documental

P58 describe el mecanismo como *'meta con downweight coste-sensible'*. El codigo no
construye ningun meta-modelo: entrena `LogisticRegression` directamente sobre la
representacion de V03 con `sample_weight`. La etiqueta 'meta' podria hacer creer que
pertenece a la familia de stacking, que ya fue rechazada por transferencia.
**No pertenece.** La fuente autoritativa es el codigo.

## Los metadatos de ejecucion NO son reproducibles, y se declara

Regenerar la cadena reescribe cuatro informes historicos de la era v4:

```text
reports/hardness_v4.json                   fecha y duracion_s
reports/label_consistency_v4.json          fecha y duracion_s
reports/label_consistency_v4_ejemplos.md   la fecha en una linea de texto
reports/supervised_weighting_v4.json       fecha y duracion_s
```

**Los numeros no cambian**: `mejor_corrected_cv` sigue siendo `0.785416` y la variante ganadora
sigue siendo `combo:downweight_violacion_x_clase`. Lo que cambia es la **fecha** y la
**duracion**.

Y la duracion merece un comentario: el original registro `704,67 s`, la regeneracion de hoy
`118,24 s`. Un factor de seis. La duracion **no es una propiedad del experimento**: depende del
hardware y de la carga de la maquina, luego no es reproducible ni debe tratarse como evidencia.

Los cuatro ficheros se han **restaurado a su version historica**. Su provenance pertenece a su
propia fase, y reescribirla haria que un informe de 2026-09-25 dijera que se ejecuto en
2026-09-29. Es la misma regla que se aplico al artefacto de P66: preservar la correspondencia
entre codigo y resultados, y no tocar el codigo ni los informes de una fase anterior.

Si alguna fase futura regenera esta cadena, el procedimiento es el mismo que el de P66:

```bash
git checkout HEAD -- reports/hardness_v4.json reports/label_consistency_v4.json
git checkout HEAD -- reports/label_consistency_v4_ejemplos.md
git checkout HEAD -- reports/supervised_weighting_v4.json
```

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p69_03_cierre_downweight.py`. Sin push.*