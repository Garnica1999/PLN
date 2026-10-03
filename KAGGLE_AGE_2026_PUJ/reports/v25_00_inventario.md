# V25 · 00 — Inventario y procedencia de las 279 features

Este script **no entrena**. Verifica que las 73 features label-informed sean legales antes de que la seleccion supervisada las toque.

## 1. Composicion de la matriz

- 169336 filas x **279 columnas**.

| Bloque | Columnas | Que es |
|---|---|---|
| `estructurado` | 87 | features estructuradas calculadas sin supervision |
| `miembro_ensamble` | 78 | probabilidades OOF de los 14 miembros del ensamble de V12 |
| `label_informed` | 73 | agregados condicionales a la clase, construidos in-fold |
| `denso_textometrico` | 35 | estadisticos de texto y de estilo, sin supervision |
| `nbsvm_palabra` | 6 | 6 columnas NB-SVM de palabra, OOF |

## 2. Procedencia de las 73 label-informed

La pregunta es si se construyeron con las etiquetas de **una particion externa al fold**. Se responde por codigo, por comparacion de arrays y por separacion empirica.

| Verificacion | Resultado |
|---|---|
| El codigo calcula con `y_train` | True |
| El codigo escribe **solo** en `valid_idx` | True |
| `y_train` llega a la firma de la funcion | True |
| El array de grupos es identico al de V20 | True |
| Los folds de validacion son identicos | True |
| Tamano de cada fold | [33868, 33868, 33867, 33866, 33867] |

**Veredicto: LEGAL: construidas con las etiquetas del training fold y con la misma particion de folds que V20**

Las 73 columnas se pueden usar en la seleccion supervisada sin riesgo de leakage de etiquetas.

## 4. Escala y tipo, que es lo que condiciona el mutual information

| Bloque | Columnas | Magnitudes distintas (mediana) | max abs | Negativas | Constantes | Separacion por clase |
|---|---|---|---|---|---|---|
| supervisada | 73 | 2049 | 5.944e+04 | 5 | 1 | 0.2501 |
| no supervisada | 206 | 2598 | 9500 | 87 | 0 | 0.2708 |

Las supervisadas llegan a **59.439** de magnitud absoluta, mientras que las probabilidades de miembro viven en [0, 1]. Por eso el mutual information se calcula sobre columnas **estandarizadas dentro de cada fold**: sin estandarizar, un estimador que divida y busque puede quedarse sin senal en columnas con muchos valores distintos, solo por la escala.

## 5. Columnas que no aportan nada

- **1 columna(s) constante(s)**: `rt_source_unknown_ratio`.
- Su mutual information sera exactamente cero, luego la seleccion las descarta de forma automatica. Se documentan para que el conteo de 279 sea explicito.

## 6. Verificaciones

| Comprobacion | Resultado |
|---|---|
| matriz de 279 | OK |
| inventario completo | OK |
| nombres sin repetir | OK |
| nombres no vacios | OK |
| procedencia legal | OK |
| clasificacion etiquetada | OK |

Ninguna cifra de este informe esta escrita a mano: el inventario se construye leyendo las anchuras reales de cada fichero de `MEMBER_FILES`, la lista real de `DENSE_FEATURES`, las columnas reales del parquet estructurado, los nombres reales de las supervisadas y la anchura real del NB-SVM.
