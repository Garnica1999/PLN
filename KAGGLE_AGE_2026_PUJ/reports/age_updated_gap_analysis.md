# AGE - gap analysis actualizado

**Baseline congelado:** V30 = 0.6134471, leaderboard 0.63302.
**Variabilidad de entrenamiento:** 0.0008876, que es la escala contra la que se juzga cualquier delta de esta fase.

## La medicion que fundamenta el inventario

El brief pide responder que informacion del train queda sin explotar. La tentacion es responderlo leyendo el historial, pero el historial ya dice cosas contradictorias: hay lineas que cerraron por resultado y lineas que cerraron por mecanismo. Lo que las separa es una sola medicion, y se hace antes de escribir nada.

Se toman los **65,403** errores de V30 y se reparte segun si alguna de las **26** fuentes disponibles (los tres miembros de V30, los artefactos del pool de V29 y los diez canales de V31) acierta esas filas:

| Grupo | Filas | % de los errores | % del dataset | Que exige |
|---|---|---|---|---|
| **A: nadie corrige** | 27,007 | **41.3%** | 15.9% | una representacion que ninguno de los modelos actuales posee |
| **B: alguien corrige** | 38,396 | **58.7%** | 22.7% | un mecanismo que use una evidencia que ya existe |

**La lectura que importa es la del grupo A.** Es un limite duro, y conviene decirlo con claridad: si mas de la mitad de los errores de V30 no los corrige ninguna fuente disponible, entonces ninguna reformulacion de las representaciones actuales va a tocarlos, por muy distinta que sea. El margen de esta investigacion esta en el grupo B.

### Quien rescata mas

| Fuente | Errores de V30 que corrige |
|---|---|
| `canal:todos` | 12,437 |
| `canal:forma_palabra` | 12,097 |
| `canal:posicion` | 11,879 |
| `canal:ortografia+composicion` | 11,797 |
| `canal:ortografia+composicion+discurso` | 11,761 |
| `canal:composicion` | 11,741 |
| `canal:discurso` | 11,661 |
| `canal:ortografia` | 11,657 |

## Las cuatro categorias, y cual manda

| Categoria | Que significa | Cuantas |
|---|---|---|
| **A. Disponible pero mal explotada** | la informacion existe y su integracion fallo, no la informacion | 4 |
| **B. Nueva, no probada** | el mecanismo no se ha implementado en ningun sitio | **2** |
| **C. La integracion ya fallo** | cerrar como familia, no como configuracion | 6 |
| **D. Sin evidencia suficiente** | descartada por irrelevancia o por no tener respaldo | 2 |

El brief pone la prioridad maxima en **B** y, en segundo lugar, en **A** solo si el mecanismo es genuinamente distinto. Aqui hay **1** hipotesis en B sin precedente cerrado, y es la unica que sobrevive al filtro de duplicados.

## Inventario completo

| ID | Cat | Familia | Hipotesis | Estado | Prioridad |
|---|---|---|---|---|---|
| G12 | A | integracion de score | correccion cuya intensidad dependa del margen de V30 (pendiente sobre  | **sin evidencia, mecanismo no probado** | 1 |
| G08 | A | entrenamiento | entrenar con targets suavizados en las zonas de frontera | **sin evidencia directa** | 4 |
| G10 | A | calibracion | corregir el sesgo de probabilidad por clase | **PARTIAL_PRECEDENT** | 6 |
| G09 | A | entrenamiento | reponder el entrenamiento hacia el grupo A (nadie corrige) | **sin evidencia, y con contra-indicacion** | 8 |
| G06 | B | retrieval de URL | recuperar el texto de la URL de la fila | **QUARANTINED** | 0 |
| G14 | B | fuente de datos | informacion del train que ningun modelo disponible ha leido todavia | **sin explorar, y no es un experimento** | 2 |
| G01 | C | correccion de score | corregir el score de V30 sumando el residuo linguistico | **INVALIDATED (P33 global, P34 local)** | 0 |
| G02 | C | representacion linguistica | anadir un canal linguistico al espacio de features | **CLOSED (V31-06)** | 0 |
| G03 | C | capacidad del meta-modelo | mas capacidad en el meta-arbol | **CLOSED (V29, V30)** | 0 |
| G04 | C | diversidad de ensembles historicos | combinar mas modelos historicos | **CLOSED (V29)** | 0 |
| G05 | C | correccion de umbral | mover el umbral de decision de V30 | **CLOSED (V20-F3)** | 0 |
| G13 | C | bagging | promediar el residuo sobre subconjuntos de datos | **CLOSED (V28-02)** | 7 |
| G07 | D | metrica | optimizar el ranking del margen en vez del argmax | **DESCARTADO por irrelevancia** | 9 |
| G11 | D | estructura de clases | fusionar las dos clases extremas | **sin evidencia** | 10 |

## Los cierres, y por que son de familia y no de configuracion

G01 a G05 no se reintentan con otro parametro. La razon es que **la mecanica ya fallo con un control de atribucion**, y cambiar `C`, `seed` o `beta` no cambia ese hecho:

| Familia | Por que se cierra como familia |
|---|---|
| G01 correccion de score | P33: el ruido aleatorio rindio **mejor** que la senal real. P34: cero rescates de alta confianza |
| G02 canal linguistico como feature | V31-06: +0,0002948 sobre el control barajado |
| G03 capacidad del meta-modelo | V30 refinamiento +0,000069 sobre V29: la region estaba ya encontrada |
| G04 ensembles historicos | 0 de 47 combinaciones con rescate neto positivo |
| G13 bagging del residuo | V28-02 ya probo el bagging del meta-modelo: -0,000312 |

## La unica linea que sobrevive, y por que es distinta

**G12: meta-aprendizaje de baja dimension sobre el residuo.**

No es un `beta` distinto. P33 probo `beta` compartido, que es **un** parametro, y un intercepto, y salio `alpha = 0`. Lo que no se probo es una funcion de correccion que dependa de **donde esta la fila respecto al margen de V30**: no un `beta` para todos, sino una correccion cuya intensidad varye con la confianza que V30 ya tiene.

Tres razones por las que no es un duplicado:

1. **Mecanismo**: P33 ajusto una constante; esto ajusta una pendiente sobre una variable que P33 no uso.
2. **La evidencia apunta en esa direccion**: el AUC del margen de V30 para predecir sus propios errores es **0,8005**, y el del residuo es 0,5411. La correccion deberia pesar mas donde V30 ya es confiado, no donde duda.
3. **Costo**: cabe en 2-3 parametros ajustados nested, sin grid.

Y una razon para ser cauto, que conviene decir antes de ejecutarla: **es exactamente el mecanismo que la regla de continuidad prohibe repetir**. P34 ya toco la confianza de V30 en la version local, y la epico P33 +0,0002. La version global con una pendiente en vez de una constante es genuinamente otra operacion, pero el riesgo de que vuelva a fallar por el mismo motivo es real.

**G14 (informacion nueva observable) tiene prioridad de exploracion pero no es ejecutable hoy**: exige identificar que campo del train no se ha leido todavia, y eso es trabajo de inventario, no un experimento. Es la unica hipotesis de categoria B con margen estructural, y por eso queda seleccionada para la siguiente fase.

## Conclusion de la fase de investigacion actual

El grupo A tiene 27,007 filas. Ese es el techo. Entre las categorias A, B, C y D, solo **dos** hipotesis no son duplicados funccionales, y una de las dos requiere trabajo previo de inventario que no es un experimento.
