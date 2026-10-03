# P62 - Es la restriction la cobertura del vocabulario y no la gramatica

Generado por `experiments/p62_09_cobertura.py`. Fecha 2026-09-29T05:43:52+00:00.

## Como se llega aqui

P62-04 comparo el canal de interaccion con y sin el sufijo de POS y el resultado no
fue el que la hipotesis anticipaba: el sufijo ayudaba en 6 de 8 presupuestos, con
hasta +0,006881, luego la interaccion no es solo partir filas. Pero AMBOS pierden en
8 de 8, y el script se negn a llamarlo redundancia porque su propia condicion
pre-declarada fallo en uno de sus tres terminos.

La condicion fallo en la COBERTURA, y seguir ese hilo dio el experimento.

## La medicion que reencuadra la fase

El canal de palabras de V03 tiene **2673 terminos** en el pliegue 1. Las 21.000
entidades son la union de bigramas de palabra y gramas de caracter, y el canal de
UNIGRAMAS es pequeno porque `min_df=2` tira todo termino que aparece una vez.

```text
formas superficiales presentes en el pliegue 1   4569
  en el vocabulario de V03                       1259   27,6 por ciento
lemas distintos presentes                        4057
  en el vocabulario de V03                        957   23,6 por ciento
```

Unas tres cuartas partes de las palabras que estan de verdad en el texto de
entrenamiento no estan en el canal de palabras de V03. Los ejemplos son palabras
ordinarias: `abandonado`, `aberrante`, `abolir`, cada una aparece una vez y por eso
se va.

**Eso reencuadra todos los resultados linguisticos de la fase.** Un canal
condicionado por POS no es sobre todo una fuente de informacion gramatical: es
tambien una manera de devolver palabras que `min_df=2` habia descartado, indexadas
por lema y partidas por categoria. Y lo mismo vale para el canal de lemas solos. Si
esa lectura es correcta, el cuello de botella es la COBERTURA, y la cobertura se
puede probar sin gramatica alguna.

## Los cuatro brazos

| brazo | que ansla |
|---|---|
| control | V03 exacto, `min_df=2` en los dos canales |
| A | solo la cobertura del canal de palabras |
| B | la cobertura de los dos canales |
| C | cobertura completa mas un canal POS de K=256, para preguntar si la gramatica aporta cuando ya no queda hueco a la falta de palabras |

Una configuracion por brazo, sin rejilla y sin busqueda. El K del brazo C se fijo
antes de ejecutar para que C no se convirtiera en una busqueda sobre su propio
resultado.

## La cobertura

| pliegue | vocab md2 | vocab md1 | formas | cobertura md2 | cobertura md1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 2673 | 19723 | 4652 | 0.2799 | 0.9951 |
| 2 | 2766 | 20114 | 4737 | 0.2820 | 0.9956 |
| 3 | 2745 | 19904 | 4715 | 0.2831 | 0.9953 |
| 4 | 2776 | 20137 | 4743 | 0.2903 | 0.9952 |
| 5 | 2744 | 20118 | 4744 | 0.2820 | 0.9954 |

Cobertura media: `min_df=2` **0.2835**, `min_df=1` **0.9953**.

## Los resultados

| brazo | rasgos | corrected_cv | delta vs V03 |
|---|---:|---:|---:|
| control, V03 exacto | 20441 | 0.7772114525 | +0.0000000000 |
| A_palabras_min_df_1 | 37491 | 0.7731887186 | -0.0040227339 |
| B_palabras_y_char_min_df_1 | 75006 | 0.7655315159 | -0.0116799367 |
| C_min_df_1_mas_canal_POS | 37747 | 0.7738605589 | -0.0033508936 |

### Que se lleva cada brazo

```text
A, solo el canal de palabras con min_df=1   -0.0040227339
B, mas el canal de caracteres con min_df=1   -0.0116799367
C, cobertura completa mas canal POS          -0.0033508936
el canal POS sobre cobertura completa vale   +0.0006718403
```

```text
ampliar la cobertura NO mejora: A -0.0040227339, B -0.0116799367, C -0.0033508936. Asi que las palabras que min_df=2 descartaba no eran las que faltaban, y la pista de la cobertura queda descartada junto con la de la gramatica
```

## Gate

Mejor brazo **C_min_df_1_mas_canal_POS**: delta -0.0033508936, pliegues 2/5, gate **False**, bootstrap NO EJECUTADO

## Decision

```text
CLOSED. Ampliar la cobertura del vocabulario no aporta sobre V03
```

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p62_09_cobertura.py`. Sin push.*