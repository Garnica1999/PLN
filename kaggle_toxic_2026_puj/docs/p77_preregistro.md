# P77: pre-registro del operador selectivo

Este documento se escribe y se hashea **ANTES** de ejecutar nada de P77. Ningun valor de este
fichero puede cambiarse despues de mirar la outer validation.

```text
sha256 de este fichero (sobre el contenido anterior a la linea de hash)  se anade al final
```

## 1. Motivacion y cadena de evidencia

```text
P72  mean pooling no demuestra ventaja aprendida frente a embedding aleatorio
P73  la TextCNN revela una diferencia frente al aleatorio
P74  la diferencia sobrevive a un null aleatorio replicado, 10 de 10
P75  la diferencia es estructuralmente complementaria pero no transferible por blend lineal
P76  la calibracion corrige la escala pero tampoco recupera la transferencia
```

P75: V03 tiene 757 errores, el CNN corrige 268 y dana 813. AUC del score del CNN para predecir el
error de V03: HS 0.4872, TR 0.6045, AG 0.4973. P76 mostro que la calibracion, al ser monotona, deja
ese AUC en 0.4910 / 0.6032 / 0.4995.

La conclusion acumulada es que **no falta informacion, falta un criterio de cuando usarla**. P77
cambia el operador de transferencia, no la representacion.

## 2. Congelado e inmutable

```text
V03 = FROZEN, corrected_cv 0.7772114525165721, umbrales HS 0.44 TR 0.51 AG 0.46
Word2Vec Twitter 100d = FROZEN      TextCNN = FROZEN (sin fine-tuning)
5 folds externas canonicas, identicas a P75 y P76
test = 0    holdout = 0    leaderboard = 0    submission = 0
```

**El holdout de 248 filas no se usara ni para un finalista.** Una revision propuso confirmarlo
alla; se rechaza porque el encargo lo prohibe explicitamente.

## 3. Hipotesis

> La informacion complementaria de Word2Vec + TextCNN no debe aplicarse globalmente. Una regla
> selectiva que modifique la decision de V03 unicamente cuando exista evidencia concordante y
> suficientemente fuerte de la CNN podria conservar los rescates y evitar parte del damage.

La pregunta NO es si la CNN tiene informacion, eso ya lo sostiene P74. Es si un operador selectivo
puede extraer los 268 rescates de P75 sin pagar los 813 danos.

## 4. Senales del operador

La **decision** vive en el espacio crudo de V03 con los umbrales congelados, porque esos umbrales
definen el punto de operacion historico del campeon y recalibrar V03 lo empeora 0.0086 (P76).

Las **magnitudes de evidencia** se calculan en escala logit, porque comparar margenes crudos entre
un modelo sobreconfiado (pendiente 1.31) y uno infraconfiado (pendiente 0.59) es un artefacto de
escala. El logit es una transformacion **monotona y sin parametros ajustados**, luego no introduce
calibracion alguna y, por P76, no puede crear informacion nueva: solo hace comparables las
magnitudes.

```text
t_j         0.44 (HS), 0.51 (TR), 0.46 (AG), congelados
s_v_j       p_V03_j  - t_j                        margen firmado, espacio crudo, decide el signo
s_c_j       p_CNN_j  - t_j                        margen firmado, espacio crudo, decide el signo
m_v_j       |logit(p_V03_j) - logit(t_j)|         magnitud de evidencia de V03
m_c_j       |logit(p_CNN_j) - logit(t_j)|         magnitud de evidencia de la CNN
d_j         m_c_j - m_v_j                         evidencia relativa de la CNN frente a V03
desacuerdo  1 si signo(s_v_j) != signo(s_c_j) en 0
```

Prohibido: features textuales, slices, id, group size, metadata, target statistics, y **la etiqueta
como feature del gate** (seria multiplicar el tuning por tres).

## 5. Mecanismo 1 y 2: operador determinista con corte por cuantil

V03 es la decision por defecto. La CNN sustituye a V03 en la etiqueta j **solo si** las tres
condiciones se cumplen:

```text
1  hay desacuerdo en j
2  m_c_j  >= C_m
3  d_j    >= C_d
```

Los cortes **no son numeros fijos**: se definen por una **regla de cuantil pre-registrada**,
calculada unicamente dentro de `outer_train` mediante un OOF interno de ambos modelos con 4 inner
folds por clusters. La base del cuantil es la distribucion de las **unidades discrepantes**, que es
donde el operador decide, y no la poblacion global.

```text
nivel 1 conservador    C_m = cuantil 80 de m_c en discrepantes internas
                       C_d = cuantil 80 de d   en discrepantes internas
nivel 2 mas permisivo   C_m = cuantil 50 de m_c en discrepantes internas
                       C_d = cuantil 50 de d   en discrepantes internas
```

Declaracion de honestidad: **un cuantil es un parametro ajustado**. Anidado deja de ser un
hyperparametro libre. No se maximiza la metrica de evaluacion con el, es un transform fijo de la
distribucion de entrenamiento. Aun asi, los dos niveles mas el gate son **tres mecanismos
evaluados sobre el mismo OOF**: se reportan los tres deltas sin ocultar los perdedores.

Sin grid. Sin parametro por tweet. Sin parametro por etiqueta. Sin recalibrar umbrales.

## 6. Mecanismo 3: gate aprendido nested

```text
features   m_v, m_c, d, desacuerdo
modelo     LogisticRegression, lineal, class_weight balanced, regularizacion fuerte
corte      0.5 fijo, no ajustado
entrenado  SOLO sobre las unidades discrepantes del OOF interno de outer_train
aplicado   una sola vez sobre outer_validation
```

El target en las discrepancias es un unico evento: **1 si la CNN acierta, 0 si falla**. Ampliarlo a
toda la muestra cambiaria el evento y destruiria el intercepto util.

Los inner folds heredan literalmente los hiperparametros congelados de V03 y del CNN, y las mismas
epocas E* que uso P75. No hay ninguna seleccion nueva.

Los dos sesgos que introduce el anidamiento, declarados y de signo conocido:

```text
modelos internos mas debiles  ->  margenes menores  ->  gate conservador     (sesgo a la baja)
cuantil interno demasiado bajo ->  dispara de mas    ->  sesgo a la alta
se anulan en parte y ninguno esta acotado
```

## 7. Metricas obligatorias

```text
corrected_cv, fold_1..fold_5, mean, std, delta_vs_V03
rescue, damage, net por etiqueta HS TR AG
rescue conservado, damage evitado, rescue perdido, damage introducido
numero de switches, total y por etiqueta
numero efectivo de TWEETS (clusters), no de filas tweet-etiqueta
bootstrap pareado por clusters, INCONDICIONAL para los tres mecanismos
p de McNemar sobre las unidades discordantes
IC del AUC del gate
```

`rescue`/`damage`/`net` son recuentos de unidades tweet-etiqueta, **no** una mejora de la metrica.

## 8. Gate de finalista

Las tres condiciones de Carlos, y **dos condiciones adicionales mas estrictas** que solo pueden
impedir una adopcion, nunca fabricarla:

```text
C1  delta_vs_V03 >= +0.003
C2  >= 4 de 5 folds positivos
C3  rescue > damage
C4  limite inferior del IC95 del delta > 0
C5  p_McNemar < 0.05 y no inferioridad por etiqueta: delta_label >= -0.005 en las tres
```

C4 y C5 se anaden porque con 5 folds y unos cientos de tweets el error tipico de `corrected_cv` es
del orden de 0.015 a 0.020, luego **+0.003 esta muy por debajo del ruido** y podria dispararse por
azar. La regla de finalista exige las cinco.

Si aparece **exactamente un** finalista: congelarlo y detenerse. No elegir entre varios. No hacer
submission. No|leaderboard.

Si no aparece: **no se ejecuta segunda particion**.

## 9. Reparto de casos

```text
CASO 1  ningun operador transfiere, corrected_cv <= V03               -> cerrar P77
CASO 2  reduce damage frente al blend global pero delta <= 0           -> sin submission
CASO 3  finalista, las cinco condiciones                             -> congelar uno y detener
CASO 4  delta > 0 pero < +0.003                                      -> MARGINAL, sin nueva grid
```

## 10. Fuera de alcance

Fine-tuning de Word2Vec o del CNN, BiGRU, attention, arquitectura nueva, dimension nueva, corpus
nuevo, pooling nuevo, stacking, kNN, augmentation, busqueda global de alpha, leaderboard,
submission.

## 11. Slices

Solo las 11 oficiales, **diagnosticas**. No se usan para seleccionar el operador ni sus parametros.
No se cita una slice que mejore sin declarar que es seleccion no declarada.

## 12. Criterio de parada antecipado

Detenerse si aparece: OOF mismatch, fuga de folds, fuga del gate, filas distintas entre V03 y CNN,
preprocesamiento inesperado, contaminacion de umbrales, contaminacion de test o holdout,
discrepancia numerica, artefacto o provenance ausente, salida incorrecta del notebook, figura mal
etiquetada.
024b0288af5746baaa1b4acfbdade3c94fb2fbcf29601f41447c893706e7b0ca
