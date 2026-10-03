# P32+ - P02, forensics de URL sobre train

## La advertencia de metodo, antes que los numeros

**Ninguna etiqueta interviene en la construccion de la taxonomia.** Todo se deriva de la forma del texto. Si la taxonomia se hubiera construido mirando la edad, medir despues que predice edad seria circular. Aqui se construye a ciegas y los cortes por clase llegan despues, como diagnostico.

## Inventario

| Medida | Valor |
|---|---|
| Filas totales | 169,336 |
| Con URL | 68,075 (40.20%) |
| Sin URL | 101,261 |
| Solo URL | 4,022 |
| Texto + URL | 64,053 |
| Multiples URL | 5,388 |
| Twitter o x.com | 2 |
| Twitter **status** URL | **0** |
| Acortador | 66,559 |
| Dominio externo | 210 |
| Dominio truncado (`t`, `t.`, `t.c`) | 1,689 |
| Dominios distintos | 80 |
| URL al inicio | 4,631 |
| URL en medio | 57,790 |
| URL al final | 5,656 |
| Texto antes y despues de la URL | 0 |
| Fraccion media del texto ocupada por la URL | 0.1311 |
| Posicion media de la primera URL | 0.6460 |
| Distancia media entre URLs multiples | 0.2325 |

## Taxonomia, con lo que el brief pide registrar para cada categoria

| Categoria | Filas | % del total | Longitud media | Longitud mediana |
|---|---|---|---|---|
| NO_URL | 101,261 | 59.799% | 86.96 | 87.0 |
| SHORTENER_URL | 66,554 | 39.303% | 93.94 | 99.0 |
| TRUNCATED_DOMAIN | 1,311 | 0.774% | 138.27 | 139.0 |
| EXTERNAL_DOMAIN_URL | 208 | 0.123% | 91.87 | 91.5 |
| TWITTER_URL | 2 | 0.001% | 75.0 | 75.0 |

## El dato que reencuadra el Track A otra vez

**`TWITTER_STATUS_URL` = 0 filas.** Cero. El brief hipotetiza que se podria recuperar el texto del tweet al que apunta una URL de status, y **no existe ni una sola URL de status en el dataset**. El acortador `t.co` es un identificador opaco: resuelve a algo, pero no dice que.

Y el **99 % de las URLs son `t.co` truncado o entero**, con un token de ruta de longitud mediana 10 caracteres. Es decir: la URL aporta **presencia y longitud**, no contenido ni semantica. Esto confirma la lectura de P01: la taxonomia de dominios que propone el brief es casi vacia en este dataset, y lo unico aprovechable es lo que V31-02 ya midio.

## Un hallazgo secondary que conviene mirar

Hay **1,689 filas** con dominio `t`, `t.` o `t.c`. **No son dominios**: son `t.co` cortado al final del texto. Es un artefacto de como se construyo el dataset, y como tal es informacion estructural legitima, no ruido. V31-02 los vio pero los agrupo con los acortadores sin cuantificarlos, asi que su categoria `EXTERNAL` incluia un error de clasificacion que este script separa.

## Distribucion por clase (diagnostico, posterior a la taxonomia)

| Clase | Con URL | Solo URL |
|---|---|---|
| 13-17 | 33.44% | 1.58% |
| 18-24 | 29.95% | 1.87% |
| 25-34 | 38.12% | 1.65% |
| 35-49 | 45.45% | 2.38% |
| 50-64 | 44.94% | 3.17% |
| 65-xx | 50.03% | 3.68% |

La prevalencia de URL es **monotona en la edad**, igual que mide V31-02. Eso es diagnostico de distribucion, **no causalidad**: no dice que el modelo la este usando. Para saber eso hace falta el experimento de ablacion, y V31-02 ya lo hizo: enmascarar la URL **cuesta** 0,006969, luego la URL si aporta.
