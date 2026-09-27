# V22: cierre de la fase

Worktree `/home/carlos/Git/PLN-v22`, rama `v22`. **Ningun modelo entrenado. Ninguna
submission enviada.**

## 1. La tabla que se pidio (seccion 22)

| configuracion | CV | delta V03 | delta control | LB | estado |
|---|---:|---:|---:|---:|---|
| HISTORICO (control, sin ABTT) | **0.781766** | +0.004555 | 0 | no enviada | referencia, ya conocida de V5 |
| ABTT D=1 | 0.779935 | +0.002724 | -0.001831 | no enviada | descartada |
| ABTT D=2 | 0.775645 | -0.001566 | -0.006121 | no enviada | descartada |
| ABTT D=5 | 0.778144 | +0.000933 | -0.003622 | no enviada | descartada |
| CRUDO (sin normalizar) | 0.768895 | -0.008316 | -0.012871 | no enviada | descartada |

V03 = 0.7772114525165721 OOF, **0.78306 LB**.

## 2. Las tres distinciones que se pidieron

- **Mejor resultado local**: el control historico, 0.781766. Y no es un hallazgo de V22: es el
  valor que V5 publico hace varias fases. La mejor configuracion *nueva* de V22 es ABTT D=1
  con 0.779935, que esta **por debajo** del control.
- **Mejor resultado competitivo**: **ninguno**. Ninguna configuracion de V22 supera al
  control, y el control mismo solo aporta +0.004555, cuyo perfil de cambios (136 celdas,
  neto +10) es practicamente indistinguible del de FUSION-WS-3 (137 celdas, neto +11) que
  V21 ya somete a McNemar con p = 0.3930.
- **Champion real**: **V03 = 0.78306**. No se modifica.

## 3. Hipotesis 1: REFUTADA

La hipotesis era que el embedding se usaba sobre un texto de tarea con preprocessing
incompatible con el del corpus. **Falsa, y por una razon que。 el propio proyecto ya la
habia resuelto**: `embedding_13_window_v5_1.py` linea 226 aplica `normalizar_emb_normal` al
texto de la tarea antes de construir los vectores de documento. La normalizacion del corpus y
la del texto de la tarea son la misma funcion.

Medido, con el mismo embedding, el mismo clasificador y el mismo diccionario de 6.059
vectores de tipo:

| brazo | corrected_cv | delta V03 | emb solo | celdas | rescates | danos | neto |
|---|---:|---:|---:|---:|---:|---:|---:|
| HISTORICO | 0.781766 | +0.004555 | 0.666085 | 136 | 73 | 63 | +10 |
| CRUDO | 0.768895 | -0.008316 | 0.640933 | 133 | 50 | 83 | -33 |

**HISTORICO - CRUDO = +0.012871.** El pipeline historico es el mejor de los dos, luego no
habia nada que corregir. La cobertura del desarrollo es **1.0000** por tipo y por token
normalizado, y cae a 0.5501 y 0.7414 sin normalizar.

Lo que si queda es un **merito medido de una decision de diseno previa**: la normalizacion
EMB-NORMAL vale 0.0129 de corrected_cv frente a no normalizar. Sin ella, el bloque denso
habria dado -0.0083, es decir, el embedding habria empeorado V03 en vez de mejorarlo.

**La rama CONSERVATIVE no es evaluable de forma limpia.** `vocabulario_toxic` construye el
diccionario de 6.059 tipos con `tokenizar(normalizar_emb_normal(texto))`, o sea, **el
diccionario vive en el espacio normalizado**. Preservar el texto de menciones y hashtags en
vez de colapsarlos a marcadores perderia los 345 tokens `usuario` y 140 `hashtag`, que si
tienen vector, a cambio de nombres propios que no lo tienen. Y cambiar el diccionario seria
cambiar dos variables a la vez.

## 4. Hipotesis 2: REFUTADA con explicacion mecanica

All-But-The-Top (Mu y Viswanath 2018, arXiv:1702.01417), sobre la matriz de los 6.059
vectores de tipo, antes del pooling, sin renormalizacion posterior:

| D | var. eliminada | corrected_cv | delta control | emb solo | celdas | rescates | danos | neto |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | -- | 0.781766 | -- | 0.666085 | 136 | 73 | 63 | +10 |
| 1 | 0.0531 | 0.779935 | -0.001831 | 0.659548 | 133 | 67 | 66 | +1 |
| 2 | 0.0863 | 0.775645 | -0.006121 | 0.661255 | 114 | 51 | 63 | -12 |
| 5 | 0.1684 | 0.778144 | -0.003622 | 0.658931 | 108 | 53 | 55 | -2 |

Los tres empeoran, y **el embedding aislado tambien** (0.666085 -> 0.659548, 0.661255,
0.658931). Ese ultimo punto es el que decide: la representacion quedo peor en si misma, no
solo su uso en la fusion.

**La explicacion es el espectro.** La primera direccion principal explica solo el **5.3 %**
de la varianza, y las seis primeras el **18.8 %**. El espectro es plano. El procedimiento
asume que existe una componente dominante, que en otros espacios es la de frecuencia global;
en este embedding no hay nada que quitar, y quitar cualquier direccion solo resta varianza
util.

## 5. Dictamen del especialista: NO en las dos preguntas

**All-But-The-Top de Arora, Liang y Ma (2017) sobre documento: NO merece ejecutarse.** Sus
dos mitades ya estan medidas y las dos perdieron: la ponderacion `a/(a+p(w))` es el pooling B
de V5, que perdio contra el mean; y la eliminacion de PC1 perdio en V22-03. El
contraargumento de que el espectro de documento puede diferir no salva la linea, porque en
mean pooling la PC1 de documentos es un modo de longitud, y la eliminacion degrada el bloque.

**SSWE: NO merece ejecutarse.** El especialista le concedes que puede producir una
representacion genuinamente distinta, y por eso lo analizo con mas exigencia. Lo cierran
cuatro hechos, y el tercero es cuantitativo: se dispone de **1.483 etiquetas binarias** frente
a **6.059 vectores de tipo por 200 dimensiones**, del orden de **1.2 millones de
parametros** que el objetivo supervisado moveria, es decir, unas 817 variables por ejemplo.
Para que SSWE no sobreajuste hace falta regularizacion fuerte; con regularizacion fuerte el
bloque se contrae hacia su inicializacion, que es exactamente la solucion no supervisionada
que ya se medico y que esta por debajo. **El unico regimen en que SSWE es seguro es el
regimen en que es un no-op.** No es un problema de hiperparametros: los conjuntos de
soluciones seguras y de soluciones distintas del baseline no se intersecan.

Linea final del dictamen: `CERRAR V22`.

## 6. Estado

- **V03 = 0.78306**, sha256
  `7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e`,
  `baseline_v03_frozen` sin cambios.
- **Submissions: 21 en el log, 0 enviadas en V22, 4 disponibles.**
- Controles de cordura: los dos exactos, desviacion **0.000000**.
- V23 **no** se abre.

## 7. Lo que V22 deja, y que V22 no

1. **La normalizacion EMB-NORMAL esta cuantificada en 0.0129 de corrected_cv.** Era una
   decision de diseno correcta y no documentada con esa magnitud. Ahora lo esta.
2. **El diccionario de 6.059 tipos vive en el espacio normalizado.** Es la razon por la que
   la rama CONSERVATIVE no es evaluable sin cambiar dos variables, y es un detalle que
   cualquier trabajo futuro con este embedding debe respetar.
3. **El espectro del embedding es plano** (PC1 = 5.3 %). Descarta de una vez la familia de
   post-procesados lineales que operan sobre las direcciones dominantes, y explica por que
   el embedding no se deja "limpiar" hacia arriba.
4. **Un fallo propio corregido antes de cerrar**: la primera version de `v22_02` leyo el
   signo de HISTORICO - CRUDO al reves y concluyo que el pipeline historico dejaba
   informacion perdida, cuando lo que ocurria es lo contrario. Corregido y re-ejecutado.
5. **Lo que V22 no midio y queda registrado como tal**: nada de la lista de familias del
   encargo. Las dos hipotesis autorizadas tienen respuesta, y SSWE no se autorizo.
