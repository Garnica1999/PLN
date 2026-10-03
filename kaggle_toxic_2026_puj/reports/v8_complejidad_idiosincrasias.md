# Asimetria y kurtosis, e idiosincrasias internas (V8-11)

Generado por `experiments/v8_11_complejidad_idiosincrasias.py` (V8-11 (complejidad e idiosincrasias)).

## 1. Que se mide y por que son genuinamente nuevas

El material de clase lista la asimetria y la kurtosis como medidas de complejidad de la
distribucion, y las idiosincrasias como familia de rasgos de atribucion. El proyecto ya
habia medido longitudes medias y diversidad lexica en el analisis inicial, pero **nunca
las habia usado como rasgos**, y el inventario lo confirma: no hay ninguna asignacion
ni llamada a ellas en el codigo.

La asimetria y la kurtosis de la distribucion de longitudes de palabra miden algo que la
media no mide: si los mensajes largos son la cola de la distribucion o su cuerpo. Una
cola larga significa que hay un grupo de mensajes desproporcionadamente largos, que es
justo lo que un modelo de toxicidad deberia notar.

Las idiosincrasias se operacionalizan **sin recursos externos**: un token es
idiosincratico si aparece en menos de una fraccion dada de los documentos. La frecuencia
de documento se calcula **solo con el tren de cada pliegue**, de modo que un token raro
en el pliegue no puede usar su frecuencia global.

## 2. Como se midio la consistencia

La fase v8 demostro que la estimacion es extremadamente inestable entre particiones, con
un desacuerdo medio de 0,92 a 0,97. Anadir rasgos solo vale si el cambio se mantiene a
traves de las particiones, asi que cada rama se mide en **cinco repeticiones** y se
cuenta cuantas mejoran.

## 3. Los rasgos calculados

- asimetria media: +0.2288
- kurtosis media: +0.2702
- recorrido intercuantilico medio: 1.3253

## 4. Resultados

| rama | metrica | diferencia | IC95 | repeticiones que mejoran | celdas | tasa |
|---|---|---|---|---|---|---|
| solo_complejidad | 0.780705 | +0.003493 | [-0.00573, +0.01323] | 1/5 | 201 | 0.5174 frente a 0.5478 |
| complejidad_mas_idiosincrasias | 0.777590 | +0.000378 | [-0.00993, +0.01138] | 1/5 | 281 | 0.4484 frente a 0.5342 |

Metrica de cada rama en cada una de las cinco repeticiones:

| rama | repeticion 1 | repeticion 2 | repeticion 3 | repeticion 4 | repeticion 5 |
|---|---|---|---|---|---|
| solo_complejidad | 0.777877 | 0.767585 | 0.776929 | 0.768149 | 0.776223 |
| complejidad_mas_idiosincrasias | 0.772485 | 0.768898 | 0.773874 | 0.768213 | 0.778129 |

## 5. Integridad del fichero de envio

Resultado global: **todas pasan**.

| comprobacion | resultado | detalle |
|---|---|---|
| numero de filas | correcto | 1154 frente a 1154 en la prueba |
| nombres y orden de columnas | correcto | ['id', 'HS', 'TR', 'AG'] |
| valores de etiqueta | correcto | [0, 1] |
| sin valores ausentes | correcto | 0 celdas ausentes |
| sin identificadores duplicados | correcto | 0 repetidos |
| identificadores en el orden de la prueba | correcto | coinciden con el orden original |
| filas distintas del entrenamiento | correcto | 1731 frente a 1154 |

## 6. Cuanto se aparta de V03

- celdas que cambian: 191 de 3462, en 153 filas
- por etiqueta: HS 80, TR 38, AG 73

## 7. Decision

La rama con mejor diferencia es **solo_complejidad**, con +0.003493.

**La submission se escribe pero no se recomienda enviarla.** El liston para gastar una
sonda es que la tasa de acierto entre las celdas modificadas supere el minimo y que la
mejora se mantenga en las cinco repeticiones. Con quince sondas ya pagadas y trece de
catorce indistinguibles, gastar una en una linea que se mueve dentro del ruido no
aporta informacion. **El envio es manual por D-063.**

Fecha: 2026-09-26T04:13:39+00:00. Duracion: 184.3 s.
