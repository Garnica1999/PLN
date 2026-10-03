# V15: inventario de fuentes de senal y de que llega a V03 (P0)

Generado por `experiments/v15_00_inventario_senal.py` (V15-00 (P0, inventario de fuentes de senal)).

**No se entrena nada.** Es una lectura del input real, del codigo y del registro, y es
obligatoria antes de gastar presupuesto experimental.

## 1. Las tres conclusiones, primero

**La categoria E no existe.** El fichero bruto tiene exactamente
`id;text;HS;TR;AG` y el cargador no descarta ninguna columna. No hay user ID,
ni fecha, ni perfil. La unica columna no textual es `id`, que es un contador de fila con
valores ['20002', '20003', '20005', '20007', '20008'], y que la seccion 3.E prohibe introducir.

**V03 tiene dos propiedades demostrablemente incompletas**, y de ellas se sigue una clase
de senal que V03 no puede representar. No son suposiciones, son propiedades del

**Toda clase de senal identificable ya fue implementada y medida**, y los rasgos
estructurales son **negativos en dos estimaciones independientes** del contexto mas rico
disponible.

## 2. Que ve V03, y que es lo que queda fuera por construccion

| propiedad de la representacion | que senal deja fuera | por que es demostrable |
| --- | --- | --- |
| La posicion absoluta en el documento no esta rep | donde aparece algo, no solo si aparece | TfidfVectorizer extrae n-gramas de cada palabra sin registrar en que posicion del docume |
| Las relaciones entre palabras distantes no estan | coocurrencias lejanas y relaciones de orden  | word(1,2) solo observa una ventana de dos tokens, y un modelo lineal sobre TF-IDF no tie |

Las dos propiedades tienen una comprobacion empirica comun, y es la que decide: **el AUC
estratificado por el margen de V03**. Si un rasgo discrimina dentro de los estratos donde
V03 ya ha decidido, entonces anade informacion que V03 no tiene. Si no discrimina en
ningun estrato, V03 ya lo ve y el rasgo no es nuevo. Esa medicion es P3.

## 3. El inventario completo

| categoria | senal | en el input | llega a V03 | historico | veredicto |
| --- | --- | --- | --- | --- | --- |
| E identidad o metadata | user ID, autor, perfil | NO | no aplica | ninguno | NO EXISTE. El fichero bruto tiene solo id, text, HS, TR y AG |
| E identidad o metadata | fecha o timestamp | NO | no aplica | ninguno | NO EXISTE como campo. Hay fechas dentro del texto, y ahi si  |
| E identidad o metadata | id de fila | SI | no | ninguno | PROHIBIDA por la seccion 3.E. Es un contador, con valores 20 |
| D no supervisada permitida | corpus externo mc4_es y mixto | en disco, fuera del in | no | track de embeddings v5, cerrado | NO UTILIZABLE. La seccion 3.D prohibe asumir que no este eti |
| C supervisada | margenes, probabilidades y scores OOF | SI | si | meta E2-02 y E2-18, corrector V8-08, c | CERRADA. El meta de E2-18 dio 0.7874 y se rechazo por transf |
| C supervisada | relaciones entre HS, TR y AG | SI | no | V6-B, V4-E, H2, H3, V13 | CERRADA en cinco familias distintas. V13 demostro que correg |
| A textual directa | n-gramas de palabra y de caracter | SI | si | V03, y todas las variantes de word y c | ES EL NUCLEO DE V03. No es senal residual, es la senal que y |
| A textual directa | subpalabras implicitas | SI | si | char_wb(3,5) es exactamente eso | CUBIERTA por la representacion actual |
| B estructura del texto | longitud, menciones, hashtags, URLs, c | SI | parcialmente | E2-18, rasgos estructurales del meta:  | MEDIDA Y NEGATIVA DOS VECES. META+CH+ESTR-RIDGE 0.7822 frent |
| B estructura del texto | proporciones y densidades de las anter | SI | no | E2-20, bloque de 24 columnas con densi | MEDIDA en el track de emocion, ortografia y NBSVM, que quedo |
| A textual directa | posicion de la mencion en el texto | SI | NO, y es demostr | 16_targeting_v2 linea 1280 con pos_men | IMPLEMENTADA como feature en dos canales, ambos cerrados. No |
| A textual directa | orden entre palabras distantes | SI | NO, y es demostr | ninguno que lo exprese como rasgo | NO REPRESENTADO por V03, porque la ventana de word(1,2) es d |

Conteo: **12 clases**, 3 inexistentes,
3 cubiertas directamente por V03, 4
medidas o cerradas, y 1 aislada para P3.

## 4. La comprobacion mas fuerte del inventario: los rasgos estructurales, dos veces

| comparacion | resultado |
| --- | --- |
| E2-18-META+CH+ESTR-RIDGE frente a E2-18-META-BASES-RIDGE | 0.7822 frente a 0.7874, diferencia -0.0052 |
| E2-18-META+CH+ESTR-ANID frente a E2-18-META-BASES-ANID | 0.7802 frente a 0.7853, diferencia -0.0051 |

Los rasgos estructurales se anadieron al **contexto mas rico disponible**, que es el meta de
E2-18 con 44 features, y en las dos estimaciones **empeoraron** el resultado en torno a
0.005. Eso no cierra la possibility de anadirlos a V03, que es una intervencion menor, pero
si establece que la hipotesis de que faltaban tiene la evidencia en contra.

## 5. Lo que queda aislado, y por que es exactamente una clase

Queda una sola clase que V03 no puede representar por construccion: **la posicion en el
documento**. Y hay que ser preciso sobre su estado:

- **No es nueva en el historial.** `pos_mencion_inicial` fue feature de modelo en
  `16_targeting_v2.py` linea 1280, y `mencion_inicial` en
  `20_emocion_nbsvm_ortografia_v2.py` linea 936. Ambas llegaron al meta, y ambos tracks
  quedaron cerrados.
- **No es un buen rasgo por si solo.** Su AUC aislado contra HS es de 0.5598, y su poder
  margen a margen, es decir R2 predecirlo desde los tres margenes de V03, es de 0.1258.
- **No se ha probado como columna de V03.** Eso es P4, y solo si P3 sobrevive.

La razon de aislarla no es que sea nueva, que no lo es, sino que es **la unica clase cuyo
mecanismo de fallo conocido no la cubre**: las demas se han medido en el meta y han
empeorado. La position no se ha medido en el sitio donde V03 toma su decision.

## 6. Conclusion

> De las doce clases de senal del inventario, ninguna es nueva sin matiz. La categoria E no existe. Las de texto y estructura ya fueron implementadas y medidas, y los rasgos estructurales son negativos en dos estimaciones independientes del contexto mas rico disponible. Queda una sola clase que V03 no puede representar por construccion, la posicion, y su poder aislado es de 0.5598 de AUC contra HS. Se aisa para P3 con la comprobacion que decide, que es el AUC estratificado por el margen de V03.

Y una advertencia sobre el metodo, que es la que evita la dinamica que la seccion 14
prohibe: **no se propone ninguna feature nueva**. Se propone una comprobacion, el AUC
estratificado, aplicada a una clase que el historial ya toco. Si esa comprobacion falla, la
fase se cierra sin haber gastado un solo experimento de busqueda.


Fecha: 2026-09-27T00:43:17+00:00. Duracion: 0.6 s.
