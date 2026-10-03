# P64 - La evidencia de V03 descompuesta por segmento

Generado por `experiments/p64_03_evidencia_segmental.py`. Fecha 2026-09-29T07:20:57+00:00.

## Por que esta es la linea de la que depende la fase

P64-02 cerro el perfil segmental como bloque de features, y con honestidad: el mejor
descriptor estructural llego a AUC 0,589847 para predecir los errores de V03, que es
PEOR que contar tokens, 0,600506. La asociacion entre muchos segmentos y muchos errores
es longitud con disfraz de estructura.

Eso no toca lo que este script mide, y la diferencia no es sutil. El perfil pregunta
que FORMA tiene la segmentacion. Esto pregunta **donde esta la evidencia del modelo**
dentro de ella. Un tweet puede tener un perfil corriente y llevar su evidencia toxica en
un segmento corto que el promedio del documento entierra, y solo la segunda pregunta lo
ve.

## Por que esto es descomposicion y no pseudo-labeling

Cada segmento se transforma con los vectorizadores de V03 **y se puntua con los
clasificadores de V03, ambos ajustados SOLO en las filas de entrenamiento del pliegue**.
El segmento de una fila de validacion lo puntua un modelo que no vio esa fila, ni
ninguna otra de ese pliegue. Ningun segmento recibe etiqueta, ninguno entra en una
funcion objetivo y ninguno se usa como objetivo. Los segmentos se leen a traves de un
modelo que ya existe, como un microscopio lee una lamina, y no dejan marca.

## Por que no es un P59 repetido

P59-07 agrego por **maximo**, que sesga al alza y termino con neto -4 y acierto 0,423 en
las celdas que cambiaban. Aqui se mide la **DISPERSION**: desviacion, rango, entropia,
concentracion y conflicto. Y P59 no computo nunca la contribucion de un segmento al
score completo.

El bloque tiene **48 columnas**, 16 por etiqueta: maximo, minimo,
media, mediana, top-2, top-3, fraccion de los tres primeros, desviacion, rango,
entropia, margen maximo y medio, evidencia positiva, negativa, conflicto y segmentos en
desacuerdo.

## Los tres brazos

| brazo | corrected_cv | delta vs V03 | pliegues | neto |
|---|---:|---:|---:|---:|
| V03_seg | 0.7520352870 | -0.0251761655 | 0/5 | -65 |
| V03_random | 0.7603432554 | -0.0168681972 | 1/5 | -68 |
| V03_margen | 0.7661312520 | -0.0110802005 | 1/5 | -90 |

El tercer brazo es el que mas importa. **V03_margen** es V03 mas sus propios margenes,
y P63-07 mostro que el margen solo ya explica los errores de V03 con AUC 0,757141 y que
anadir bloques lo empeoraba. Si la distribucion de evidencia no supera al margen, la
familia se cierra por una razon fuerte.

## Las dos brechas, que son las unicas preguntas

```text
contra el control de capacidad: -0.0083079683
contra el margen de V03:      -0.0140959650
```

### Por rebanada de longitud

| bin | n | F1 de V03 | F1 de V03_seg | delta |
|---|---:|---:|---:|---:|
| CORTA | 167 | 0.9480 | 0.9375 | -0.0104 |
| MEDIA | 602 | 0.7964 | 0.7759 | -0.0205 |
| LARGA | 547 | 0.6834 | 0.6483 | -0.0351 |
| MUY_LARGA | 167 | 0.5606 | 0.4671 | -0.0935 |

## La distribucion de evidencia interna por estrato

| estrato | n | `evidencia_positiva` | `evidencia_negativa` | `evidencia_conflicto` | `segmentos_en_desacuerdo` | `seg_desviacion` | `seg_entropia` | `seg_top3_fraccion` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| todas | 1483 | 0.1173 | 0.1081 | 0.0144 | 0.3922 | 0.0649 | 0.2237 | 0.9753 |
| V03 correcto | 1042 | 0.1247 | 0.1247 | 0.0111 | 0.4391 | 0.0576 | 0.1975 | 0.9805 |
| V03 incorrecto | 441 | 0.1000 | 0.0690 | 0.0221 | 0.2813 | 0.0820 | 0.2855 | 0.9629 |
| error de alta confianza | 295 | 0.1186 | 0.0740 | 0.0201 | 0.2927 | 0.0789 | 0.2775 | 0.9627 |
| CORTA | 167 | 0.2733 | 0.0729 | 0.0115 | 0.2665 | 0.0459 | 0.1048 | 1.0000 |
| MEDIA | 602 | 0.1322 | 0.1085 | 0.0124 | 0.3932 | 0.0555 | 0.1565 | 0.9940 |
| LARGA | 547 | 0.0700 | 0.1178 | 0.0146 | 0.4464 | 0.0704 | 0.2627 | 0.9749 |
| MUY_LARGA | 167 | 0.0630 | 0.1103 | 0.0233 | 0.3365 | 0.0998 | 0.4565 | 0.8845 |

## La evidencia contra la confianza

| rasgo | AUC para predecir si V03 falla |
|---|---:|
| evidencia_positiva_HS | 0.556752 |
| evidencia_conflicto_HS | 0.604785 |
| desacuerdo_HS | 0.394698 |
| desviacion_HS | 0.590605 |
| evidencia_positiva_agregada | 0.565074 |
| margen_de_V03_para_predecir_el_ACIERTO | 0.770875 |
| evidencia_positiva_HS_contra_el_ACIERTO | 0.443248 |
| evidencia_conflicto_HS_contra_el_ACIERTO | 0.395215 |
| desacuerdo_HS_contra_el_ACIERTO | 0.605302 |
| desviacion_HS_contra_el_ACIERTO | 0.409395 |
| evidencia_positiva_agregada_contra_el_ACIERTO | 0.434926 |

El mejor rasgo de evidencia es **desacuerdo_HS_contra_el_ACIERTO**, con una ventaja de **-0.165573**
sobre la referencia del margen de V03.

## Veredicto

```text
La distribucion de la evidencia interna NO supera al margen de V03, con una brecha de -0.0140959650, y tampoco supera al control de capacidad, con -0.0083079683. Lo que la evidencia interna sabe sobre los fallos de V03 ya esta en su confianza: el mejor rasgo alcanza 0.605302 de AUC contra el acierto frente a 0.770875 del margen, y la diferencia es de -0.165573 a favor del margen. En eso la descomposicion se parece al perfil, que tambien habia perdido contra la referencia trivial de longitud.

PERO LA TABLA DE DISTRIBUCION DICE UNA COSA QUE EL VEREDICTO DE INTEGRACION NO PUEDE TRAGARSE, y es el hallazgo real de la fase. Entre las filas que V03 resuelve y las que falla con alta confianza, el CONFLICTO INTERNO SE DUPLICA: 0.0111 frente a 0.0201. Y la entropia de la distribucion de evidencia dentro del tweet crece de forma monotona con la longitud, de 0.1048 en la region CORTA a 0.4565 en MUY_LARGA, y la fraccion de evidencia en los tres primeros segmentos cae de 1.0000 a 0.8845, o sea que en los tweets largos la senal deja de estar concentrada y se reparte.

La hipotesis de la fase queda CONFIRMADA COMO DESCRIPCION y REFUTADA COMO REPRESENTACION, y las dos cosas a la vez. La estructura interna es real y diferente, y es exactamente la que la bolsa de n-gramas aplana. Lo que no funciona es anadirla a V03, porque V03 ya la lleva dentro en su confianza, y eso es un resultado sobre la REPRESENTACION y no sobre la ESTRUCTURA

LECTURA: REDUNDANTE_CON_LA_CONFIANZA
```

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p64_03_evidencia_segmental.py`. Sin push.*