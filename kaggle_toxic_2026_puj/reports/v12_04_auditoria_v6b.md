# V12-04: auditoria puntual de V6-B

Generado por `experiments/v12_04_auditoria_v6b.py` (V12-04 (auditoria puntual de V6-B)).

**No se entrena nada.** Esto es una lectura de codigo y de resultados historicos.

## 1. La pregunta que obliga a esta auditoria

H2 midio dos cifras que no se pueden ignorar:

    mascara con HS VERDADERO    +0.046271
    post-proceso con HS PREVISTO -0.004187

Y la diferencia entre ambas es lo que cuesta usar la prediccion de HS como llave. De ahi la
hipotesis candidata: **una formulacion de condicionamiento suave o conjunta, donde TR y AG
aprovechen la probabilidad de HS sin convertir HS en una decision binaria previa**.

La auditoria determina si esa formulacion es nueva. Y se busca **leyendo el codigo** de
`experiments/v6_02_label_factorization.py`, no su nombre ni su resumen.

## 2. Respuestas

| pregunta | respuesta | evidencia | como_se_verifico |
| --- | --- | --- | --- |
| V6-B uso dependencia suave o llave dura? | SUAVE, y lo declara explicitamente | El encabezado dice sin routing duro, que no se impone TR=0 cuando HS=0 en la prediccion, y que la restriccion viene del producto y no de una regla. El arbol sintactico confirma que hay operaciones de producto y que ningun condicional anula una etiqueta a partir de HS. | Analisis estructural del arbol de v6_02_label_factorization.py, con recuento de productos y de condicionales cuya condicion menciona HS. |
| Restringio el dominio de entrenamiento del condicional? | SI, a HS=1 | El encabezado dice que el modelo de TR solo ve ejemplos con HS=1, que es su dominio natural. Eso es exactamente la restriccion que H2 probo en el brazo M. | Encabezado del guion, y la coincidencia con el brazo M de V12-02. |
| Que dio cada variante? | Negativo en las cuatro, con el mecanismo de fallo diagnosticado | B1 condicional de TR -0.0107 con IC95 que excluye cero, 1 de 5 pliegues, 1 fila rescatada frente a 35 rotas. B1 con AG -0.0115. B2 de AG con dos ramas -0.0008 con IC95 que incluye el cero, 2 de 5 pliegues, 0 rescatadas frente a 12 rotas. B3 -0.0115. Ninguna supera ningun gate. | Tabla de resultados de reports/v6_label_factorization.md y decision en el registro. |
| El cierre fue con evidencia o por abandono? | Con evidencia, y con el mecanismo identificado | El informe dice que el producto no solo prohibe TR=1 cuando HS=0, tambien atenua p(TR) en toda fila con p(HS) intermedia porque multiplica por un factor menor que 1, y que V03 acertaba justamente en esas filas. El saldo de B1 es 1 rescatada frente a 35 rotas. | Seccion 4 del informe de V6-02, que se cita de forma literal. |
| La hipotesis candidata sigue siendo nueva? | NO. Es V6-B | La formulacion de condicionamiento suave o conjunta, donde TR y AG aprovechan la probabilidad de HS sin convertirla en decision binaria previa, es exactamente B1, B2 y B2r. Y la restriccion de dominio es la misma que probo el brazo M de V12-02. | Contraste directo entre la formulacion candidata y las tres variantes de V6-02. |
| Coincide el fallo con el de H3? | SI, y en el mismo modo | V6-02 lo llamo efecto de escala sobre las filas que V03 acierta con p(HS) intermedia. H3 lo midio por otra via: el termino por pares mejoraba la F1 de TR donde la etiqueta esta definida, de 0,8941 a 0,9091, y empero los falsos positivos en la zona no definida, de 78 a 97. En ambos casos acoplar mult | Contraste de los dos informes, con las cifras de cada uno. |
| Queda algo por abrir? | SI, pero no con esta formulacion | Los dos numeros de V6-02 se reconcilian con el techo del oraculo de H2. V03 predice TR=1 con HS PREDICHO igual a 0 en 20 filas, que es lo que la formulacion suave puede alcanzar, y predice TR=1 en 78 filas que en realidad son HS=0, que es el techo del oraculo. Las 58 filas de diferencia son justamen | Los dos numeros del informe de V6-02 contrastados con las 78 celdas que el oraculo de H2 corrigio. |

## 3. El detalle estructural, que es lo que decide

- Operaciones de producto en el arbol de V6-02: **5**.
- Condicionales cuya condicion menciona HS: **0**.
- El encabezado declara sin routing duro: **True**.
- El encabezado declara la restriccion de dominio a HS=1: **True**.

Una llave dura tiene la forma de un condicional que pone TR a cero cuando HS es cero, es
decir una regla. **Una formulacion suave es un producto o una mezcla de probabilidades.** El
codigo de V6-02 es lo segundo, y lo dice en el encabezado antes de que nadie lo supiera.

Las lineas de producto encontradas:

- linea 188: `proba_tr[validacion] = proba_hs[validacion] * cond_tr_hs1[validacion]`
- linea 208: `proba_hs[validacion] * cond_ag_hs1[validacion]`
- linea 209: `+ (1 - proba_hs[validacion]) * cond_ag_hs0[validacion]`
- linea 616: `f"({100 * violacion_v03 / n_filas:.2f} %).",`
- linea 636: `f"{100 * violacion_v03 / n_filas:.2f} % del desarrollo) es demasiado pequena para pagar",`

## 4. La reconciliacion con el techo del oraculo, que es el dato que mas importa

| magnitud | valor | quien lo midio |
| --- | --- | --- |
| V03 predice TR=1 con HS **predicho** igual a 0 | 20 filas | V6-02, seccion 4 |
| V03 predice TR=1 en filas que en realidad son **HS=0** | 78 filas | V6-02, seccion 4, y es el techo del oraculo de H2 |
| Techo con HS **verdadero** | +0.046271 | V12-02, no adoptable porque usa la etiqueta verdadera |

**La formulacion suave alcanza las primeras y no las segundas, y la razon es la misma en los
dos casos.** p(HS) solo puede identificar las filas donde HS se predice bien, y HS se
predice mal en 143 filas del desarrollo. Las filas que la formulacion suave no puede tocar
son exactamente las que separan el oraculo de la formulacion.

Eso convierte el diagnostico de la fase en algo preciso: **el cuello de botella es la
inferencia de la condicion de HS**, y no se resuelve ni con una llave dura, que falla
enteramente, ni con una llave suave, que cubre solo la parte de la condicion que el modelo
acierta.

## 5. Decision de la auditoria

**La hipotesis candidata NO se abre, porque no es nueva: es V6-B, y V6-B esta cerrado con
evidencia y con el mecanismo de fallo identificado.** Abrirla seria repetir un
experimento cerrado, que es exactamente lo que la seccion 43 del enunciado prohibe.

Lo que **si** queda abierto, y que no es una formulacion de condicionamiento, es el
problema de fondo que las dos vias dejan al descubierto: **conseguir que la condicion de HS
se estime mejor**, porque de eso depende todo lo demas. Y esa no es una hipotesis de esta
fase.


Fecha: 2026-09-26T23:35:27+00:00. Duracion: 0.0 s.
