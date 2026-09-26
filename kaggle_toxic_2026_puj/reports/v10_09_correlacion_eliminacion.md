# V10-09: correlacion entre caracteristicas, con la etiqueta y eliminacion de redundancia

Generado por `experiments/v10_09_correlacion_eliminacion.py` (V10-09 (correlacion entre caracteristicas y con la etiqueta)).

## 1. Por que esto no repite la linea de la fase v8

La fase v8 midio una seleccion por correlacion y la cerro diciendo que reducir 20.870
rasgos a 251 por magnitud y redundancia deja la metrica en 0,594, y que por tanto la
linea quedaba *descartada de raiz*.

Eso tiene dos problemas. El primero es que **colapsar el 99 por ciento de las
caracteristicas** no permite distinguir quitar una de cada par correlacionado de
quitar casi toda la senal. El segundo es que el criterio usado era la magnitud del
rasgo, que es distinto del de redundancia, como la propia conclusion reconocia.

Aqui se mide lo que faltaba: la estructura de correlacion real, la correlacion de cada
caracteristica con cada etiqueta, y el efecto de eliminar **una sola** de cada par.

## 2. Bloque A: estructura de correlacion

| magnitud | valor |
|---|---|
| caracteristicas ajustadas en el pliegue | 20441 |
| correlacion maxima por caracteristica, mediana | 0.1704 |
| percentil 90 | 0.3358 |
| maximo | 1.0000 |
| fraccion que no correlaciona con nada por encima de 0,90 | 0.9978 |

Pares de caracteristicas por encima de cada umbral:

| umbral | pares |
|---|---|
| 0.9 | 1203 |
| 0.95 | 973 |
| 0.99 | 835 |

## 3. Bloque B: cuanto aporta cada caracteristica a cada etiqueta

Medido con el AUC de la caracteristica sola, que es una medida de rango y por tanto no
se rompe con la cola larga de los conteos. Vale 0.5 exactamente cuando la caracteristica
no aporta nada.

| etiqueta | AUC mediano | mejor AUC | fraccion fuera de [0,45, 0,55] |
|---|---|---|---|
| HS | 0.5133 | 0.6717 | 0.0029 |
| TR | 0.5198 | 0.7569 | 0.0080 |
| AG | 0.5147 | 0.6844 | 0.0035 |

Las 20 caracteristicas mas asociadas a cada etiqueta estan en
`reports/v10_09_correlacion_caracteristicas.csv`, con su AUC, su fuerza y su correlacion maxima
con otra caracteristica, que es lo que permite ver si las mas utiles son ademas las mas
redundantes.

## 4. Bloque C: eliminar una de cada par y reevaluar

| umbral | caracteristicas conservadas en el pliegue 0 | media | peor | dispersion | delta frente a no eliminar |
|---|---|---|---|---|---|
| sin eliminar | 20441 | 0.775910 | 0.772021 | 0.003365 | - |
| 0.9 | 20158 de 20441 (283 eliminadas) | 0.745284 | 0.741419 | 0.002863 | -0.030626 |
| 0.95 | 20226 de 20441 (215 eliminadas) | 0.748249 | 0.742486 | 0.003627 | -0.027661 |
| 0.99 | 20249 de 20441 (192 eliminadas) | 0.748781 | 0.746166 | 0.002522 | -0.027129 |

## 5. Gate C, pre-registrado

- mejor umbral: **0.99**
- delta medio de al menos 0.002: **False**
- **gate C superado: False**

## 6. Veredicto

**GATE C NO SUPERADO, VIA CERRADA.** Ningun umbral de eliminacion mejora a V03, los
tres empeoran, y ninguno mejora en una sola particion de las cinco. No se reintenta
con mas umbrales porque los tres ya estaban pre-registrados.

## 7. Por que empeora, y donde esta el matiz

La primera explicacion seria que **L2 ya es la respuesta estandar a la colinealidad**,
porque reparte el peso entre caracteristicas redundantes en vez de dejar que una
aniquile a la otra. Quitar una de cada par no anade informacion que el modelo no
tuviera, y solo puede quitarla. Eso es coherente con el resultado.

Pero hay una segunda explicacion, y es una confucion del propio diseno, asi que hay
que decirla: **la regla de cual se conserva es discutible, y probablemente es la
causa principal de la perdida**. Se conservo de cada par la caracteristica con mayor
separacion de su AUC sobre 0.5, es decir la mas informativa **en solitario**. Para un
modelo lineal eso es exactamente lo contrario de lo que conviene: entre dos rasgos
correlacionados, el valor para el modelo esta en lo que **cada uno aporta de lo que el
otro no aporta**, y quedarse siempre con el mas fuerte en solitario elimina
sistematicamente la pieza que desambiguaba al primero. Por eso quitar menos del 6 por
ciento de las caracteristicas cuesta tanto: no se quita ruido, se quita la
complementariedad.

**Lo que este experimento cierra es el umbral, no la regla.** Se cierra que elegir
que descartar por nivel de correlacion no mejora a V03. No se cierra que exista una
regla mejor de cual conservar, porque esa regla no estaba pre-registrada y por tanto no
se ha medido. Afirmar lo contrario seria hacer exactamente lo que la fase v8 hizo con
su prueba de 251 rasgos.

## 8. Lo que si queda medido, y antes no existia

Con independencia del veredicto, la fase v8 dejo esta medicion sin hacer y es la que
hacia falta para cualquier linea futura que quiera atacar la representacion:

- las 20441 caracteristicas estan **poco correlacionadas**:
  la correlacion maxima por caracteristica tiene mediana 0.1704 y solo -98.78 por ciento llega a 0,90.
  Es decir, la redundancia alta es **rara y muy localizada**, no un problema general.
- y casi todas las caracteristicas **aportan poco por si solas**: el AUC mediano esta
  entre 0.5133 y 0.5198, y solo entre el 0.29 y el 0.80 por ciento de ellas se sale de
  la banda [0,45, 0,55]. Las mejores llegan a 0.67, 0.76 y 0.68, o sea que el modelo lo
  construye **combinando miles de rasgos debiles**, no apoyandose en unos pocos
  fuertes. Eso explica por que quitar unos pocos hurtaria mas de lo que parece.

Fecha: 2026-09-26T08:14:40+00:00. Duracion: 269.9 s.
