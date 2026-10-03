# V28-03: ambiguedad de etiqueta en la region donde V03 falla

Generado por `experiments/v28_03_ambiguedad_etiquetas.py` (V28-03 (ambiguedad de etiqueta estratificada por la region donde V03 falla: el margen residual es reducible?)).

**Estado `DIAGNOSTIC_ONLY`. Alcance: no se entrena ningun modelo ni se construye ninguna feature. Se usa la estructura de grupos fusionados que ya existe en el proyecto para medir el desacuerdo entre anotaciones**

## 1. La pregunta

V28-01 localizo el margen residual de V03 en los textos largos y con URL. Localizar
no es lo mismo que alcanzar. Este script separa las dos explicaciones posibles:
senal reducible, que una fuente de informacion nueva podria capturar, o ambiguedad
irreducible, donde los anotadores no coinciden entre si y ninguna representacion
puede llegar mas lejos.

El experimento natural ya existe en el proyecto: `grupos_fusionados` une
near-duplicados y pseudo-autores, y los pliegues de V03 son `StratifiedGroupKFold`
sobre esos grupos. Si dos textos casi identicos llevan etiquetas distintas, la
discrepancia es de la anotacion, no del modelo.

## 2. Cuanto aporta este experimento

| magnitud | valor |
|---|---:|
| filas del desarrollo | 1483 |
| grupos distintos | 1415 |
| grupos con 2 o mas filas | 37 |
| filas dentro de grupos con 2 o mas | 105 |
| fraccion de filas que aporta informacion | 0.0708 |
| filas dentro de grupos contradictorios | 43 |

## 3. Desacuerdo intra-grupo por region

La tasa de ambiguedad se calcula **solo** sobre filas que viven en un grupo de 2 o
mas, porque una fila sola no puede estar en contradiccion consigo misma. La ultima
region es la que importa: si los textos largos tienen mas desacuerdo, la ambiguedad
explica justo la region donde V03 falla mas.

| region | filas | en grupos 2+ | fraccion | etiqueta | tasa ambiguedad | error V03 |
|---|---:|---:|---:|---|---:|---:|
| texto largo (>= 25 tokens) | 301 | 32 | 0.1063 | HS | 0.3438 | 0.2492 |
|  |  |  | 0.1063 | TR | 0.1875 | 0.0963 |
|  |  |  | 0.1063 | AG | 0.4688 | 0.2458 |
| texto no largo (< 25 tokens) | 1182 | 73 | 0.0618 | HS | 0.3562 | 0.2039 |
|  |  |  | 0.0618 | TR | 0.2740 | 0.1142 |
|  |  |  | 0.0618 | AG | 0.3562 | 0.1717 |
| con URL | 458 | 35 | 0.0764 | HS | 0.3714 | 0.2445 |
|  |  |  | 0.0764 | TR | 0.1714 | 0.0852 |
|  |  |  | 0.0764 | AG | 0.5429 | 0.2227 |
| sin URL | 1025 | 70 | 0.0683 | HS | 0.3429 | 0.1990 |
|  |  |  | 0.0683 | TR | 0.2857 | 0.1220 |
|  |  |  | 0.0683 | AG | 0.3143 | 0.1707 |

## 3bis. El limite de este experimento: poder estadistico

Solo el 7.08% de las filas del desarrollo vive en un grupo de 2 o mas, y solo
43 filas (**2.90%**) estan en un grupo con anotaciones
contradictorias. Ese es el techo del numero de filas que pueden portar informacion aqui,

**Con 43 filas informativas, este experimento no puede responder la pregunta que se le
hizo.** Los 12 tests de Fisher de la seccion siguiente se calculan sobre bloques de 32 a
73 filas, donde la mitad de las celdas no puede alcanzar significacion aunque el efecto
fuera real. El limite no es de metodo sino de datos: para medir ambiguedad hace falta una
criterio de emparejamiento **mas laxo** que el near-duplicate, que es una tarea de
ingenieria de datos y no de modelado.

Lo que si se puede leer es lo unico que sobrevive al ruido: en las celdas con
significancia, la direccion es inequivoca.

## 4. Los errores de V03 se concentran en las filas ambiguas

Si el error residual fuera ambiguedad, las filas ambiguas fallarian mas que las no
ambiguas. Fisher exacto mide si la diferencia es distinguishable del azar.

| region | etiqueta | error si ambigua | error si no ambigua | razon | p Fisher |
|---|---|---:|---:|---:|---:|
| texto largo (>= 25 tokens) | HS | 0.1818 | 0.1429 | 1.27 | 1.00e+00 |
| texto largo (>= 25 tokens) | TR | 0.0000 | 0.0000 | n/d (denominador 0) | 1.00e+00 |
| texto largo (>= 25 tokens) | AG | 0.3333 | 0.1176 | 2.83 | 2.09e-01 |
| texto no largo (< 25 tokens) | HS | 0.3077 | 0.1702 | 1.81 | 2.38e-01 |
| texto no largo (< 25 tokens) | TR | 0.4000 | 0.0755 | 5.30 | 2.23e-03 |
| texto no largo (< 25 tokens) | AG | 0.3462 | 0.0851 | 4.07 | 9.19e-03 |
| con URL | HS | 0.1538 | 0.1818 | 0.85 | 1.00e+00 |
| con URL | TR | 0.1667 | 0.0000 | n/d (denominador 0) | 1.71e-01 |
| con URL | AG | 0.3684 | 0.1250 | 2.95 | 1.35e-01 |
| sin URL | HS | 0.3333 | 0.1522 | 2.19 | 1.24e-01 |
| sin URL | TR | 0.3500 | 0.0800 | 4.38 | 9.49e-03 |
| sin URL | AG | 0.3182 | 0.0833 | 3.82 | 2.86e-02 |

## 5. Cota superior por voto mayoritario dentro del grupo

Cada fila se acierta si su etiqueta coincide con la mayoritaria de su grupo. Una fila
dentro de un grupo contradictorio nunca puede acertarse por un modelo que solo vea el
texto, asi que esta cifra acota el ruido de anotacion.

**Es una cota, no un resultado alcanzable y no un modelo.** Usa la etiqueta, de modo
que no puede convertirse en submission ni en estimador de generalizacion.

| etiqueta | cota de acierto |
|---|---:|
| HS | 0.9906 |
| TR | 0.9939 |
| AG | 0.9906 |
| macro | 0.9917 |

## 6. Lo que este script no hace

- No entrena ningun modelo.
- No construye ninguna feature a partir de los rasgos de V28-01.
- No lee el test.
- No produce submission.

Coste: 0.1 s, memoria 197 MB, 0 modelos, 0 submissions.

