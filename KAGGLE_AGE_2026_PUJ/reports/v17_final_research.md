# V17 — Informe final de fase: especialistas de frontera y decision local

**Rama** `kaggle-age-2026-puj-v17` · **commit** `ccdbcfa` · **fecha** 2026-09-26 · **experimentos** 8 de 16 previstos, todos por debajo de los 20 minutos de CPU

## 1. La pregunta y la respuesta

> ¿La informacion adicional descubierta por V16 en las dos fronteras centrales puede explotarse mediante modelos entrenados especificamente para esas fronteras y activados solamente cuando V12 es ambiguo?

**Respuesta: no, y el motivo esta medido, no conjeturado.** La informacion existe y es grande; lo que no existe es una forma de identificarla.

## 2. Lo que se encontro, en orden de importancia

### 2.1 La referencia del proyecto eran dos estimadores

0,6004 y 0,6005 no son el mismo numero redondeado: son el macro-F1 **pooled** (0.6005017473) y la **media de los cinco folds** (0.6004103057), que difieren en -0.00009144 por efecto Jensen. **Canonico: 0.6005 pooled**, porque el umbral de D-042 y el IC95 del bootstrap viven en esa escala. Usar 0,6004 regalaria +0,0001 de delta y romperia la comparacion con las cinco fases anteriores.

### 2.2 La complementariedad local es real y grande

| Frontera | V12 | especialista de pares | rescata | rompe | techo con enrutamiento |
|---|---|---|---|---|---|
| 18-24|25-34 | 0.4963 | 0.6864 | 14,356 | 3,519 | **0.7481** |
| 25-34|35-49 | 0.4974 | 0.7218 | 15,571 | 2,872 | **0.7726** |

El especialista de pares **ya existia** en el proyecto (E-028a: LinearSVC sobre el TF-IDF disperso, entrenado solo con las dos clases del par, con folds agrupados). Alcanza 0.6864 y 0.7218 en las filas de cada frontera, frente a 0.4963 y 0.4974 de V12. Resuelve entre 10.837 y 12.699 filas de forma neta.

**Lo que no es alcanzable**: en un 23 a un 25 % de las filas de la frontera ambos fallan, y ese techo nadie lo puede cobrar.

### 2.3 Por eso no funciona ninguna compuerta: el mecanismo

| Grupo de filas | % con el par arriba en V12 | separacion de V12 |
|---|---|---|
| lo que el especialista **rescata** | 12,7 % y 8,0 % | **0,809 y 0,759** |
| lo que el especialista **rompe** | 70,6 % y 67,8 % | 0,539 y 0,682 |

**V12 esta mas confiado cuando se equivoca en la frontera.** Sus errores ahi no son de baja confianza, son errores *confiados*. Una compuerta de ambiguedad no puede alcanzarlos, y el arbitro no tiene donde seccionar. Ademas, las victorias del especialista caen **fuera** de la region de frontera, que es la unica donde escucharle no destruye aciertos ajenos.

### 2.4 Las dos senales que si separan no estan donde se buscaban

| Senal | AUC 18-24\|25-34 | AUC 25-34\|35-49 |
|---|---|---|
| acuerdo entre miembros | 0,7070 | 0,7300 |
| margen del especialista | 0,7294 | 0,7539 |
| separacion de V12 | 0,5780 | 0,5268 |
| entropia de V12 | 0,4323 (invertida) | 0,4230 (invertida) |

El area es alta y aun asi no se traduce: hay cuatro pares senal-frontera por encima del umbral util, y ninguna regla llega al gate exploratorio.

### 2.5 Ni la seleccion ni la combinacion encuentran la senal

- **P9, seleccion** (arbitro LogReg de 47 columnas existentes, protocolo anidado): 0.6004, delta -0.0001, 0/5 folds. El arbitro elige umbrales de 0,60 a 0,90 y, cuando activa, pierde.
- **P5, combinacion** (redistribuir la masa del par conservando el total): 0.6003, delta -0.0002, 1/5 folds, y las fronteras empeoran a 0.4953 y 0.4968.

## 3. Respuesta a las seis opciones de la pregunta cientifica

| Opcion | Veredicto | Evidencia |
|---|---|---|
| A. V12 no tiene informacion | Falso | el especialista tiene +0,19 y +0,22 en esas filas |
| B. Un miembro de V12 si la tiene | Parcial | el acuerdo entre miembros la expone (0,71-0,73) pero no alcanza |
| C. Un especialista aprende informacion nueva | **Cierto y grande**, pero ya existia (E-028a) |
| D. El problema es solo de decision | Parcial | el techo es +0,0107, o sea que el decision si tiene margen |
| E. Se resuelve con gating | **Falso** | dos senales independientes y un arbitro entrenado dan +0,0000 |
| F. La mejora local no tiene impacto global | **Cierto en la practica** | alcanzable +0,0000 frente a un techo de +0,0107 |

**La formulacion mas precisa**: la senal existe, el techo la supera, y la identificacion es lo que no existe. No es un problema de datos (C es cierto), no es un problema de decision en el sentido de umbral (D y E son falsos), y no es un problema de metricica (el techo de +0,0107 superaria el gate de adopcion si se alcanzara).

## 4. Clasificacion de cada pista

| Pista | Estado | Motivo |
|---|---|---|
| P2, especialista de frontera | CLOSED | ya existia como E-028a; la forma de 279 variables la cerro V13 P3a |
| P2.3, tercera frontera | CLOSED | mecanismo identico, medido sin entrenar: mismo techo, misma falta de senal |
| P3, compuerta de ambiguedad | CLOSED | +0,0002 con V12 y +0,0000 con la regla conjunta |
| P3.3, compuerta blanda | CLOSED | es la variante continua de P3, y comparte su resultado |
| P4, entrenamiento pairwise | COMPLEMENTARY | E-028a ya es un par-SVM y aporta la medida central de la fase |
| P5, meta-combinador local | CLOSED | -0,0002 y fronteras peores |
| P6, modelo residual | CLOSED sin ejecutar | el brief autoriza cerrarlo antes que filtrar, y el mecanismo ya explica por que no habria de funcionar |
| P7, coeficientes por frontera | CLOSED | es P2 en su forma de 279 variables, que V13 P3a cerro |
| P8, perfil de confianza | CLOSED con resultado valioso | senal medida y publicada, aunque no accionable |
| P9, experto selectivo | CLOSED | -0,0001, 0/5 folds |
| P10, decodificacion local | CLOSED sin ejecutar | depende de que los especialistas centrales sean utiles, y no lo son |
| P1 / P1.1 | CLOSED con resultado central | la complementariedad local esta medida y es grande |

**Ninguna pista queda PROMISING ni FINALIST.** Y con razon: la unica que podia serlo era la via local, y las dos formulaciones posibles (seleccion y combinacion) estan medidas ycerrada.

## 5. Lo que queda abierto, y no es un experimento

La asimetria sin explicar: **el margen del par ya es una entrada de V12** (V15a lo midio al listar la matriz) y V12 no lo explota, cuando por si solo vale 0,6864 y 0,7218 en las filas de su frontera. V12 tiene la informacion en su espacio de entradas y su combinador no la usa. Eso no es un problema que se resuelva con mas experimentos de esta clase: es una limitacion de la funcion objetivo, que es macro-F1 sobre seis clases y no sobre las fronteras. Cualquier trabajo futuro que se dirija a las fronteras tiene que **cambiar la metrica que se optimiza**, no añadir una compuerta mas.

Figuras: `figuras/v17_p1_rendimiento_por_miembro.png`, `figuras/v17_p1_interseccion_de_errores.png`, `figuras/v17_p1b_curva_compuerta.png`, `figuras/v17_p8_area_por_senal.png`, `figuras/v17_p8_distribuciones.png`, `figuras/v17_p1c_reglas.png`, `figuras/v17_p9_arbitro_por_fold.png`, `figuras/v17_p5_combinador_local.png`, `figuras/v17_rendimiento_global_por_candidato.png` y `figuras/v17_evolucion_v12_v16_v17.png`.

