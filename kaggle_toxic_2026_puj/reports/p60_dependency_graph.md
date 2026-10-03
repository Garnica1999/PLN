# P60 - Grafo de dependencias

```text
BASELINE V03
   |
   +-- A ASL            [ABIERTA]
   +-- B W-ASL          [CERRADA]
   +-- C PAIR-NOISE     [ABIERTA]
   +-- D EFFNUM/DBL     [ABIERTA]
   +-- E LOGIT-ADJ/LDAM [CERRADA]
   +-- F LDL            [CERRADA]
   +-- G CONTRAST-OVL   [ABIERTA]
   +-- H OT-PROT        [ABIERTA]
   +-- I META-RW        [ABIERTA]
   +-- J POLYLOSS       [ABIERTA]
   +-- K CONFORMAL      [CERRADA]
   +-- L DRO-GROUP      [ABIERTA]
```

Ninguna pista depende de otra. Las doce son independientes y el bloque esta ejecutado de
forma continua: una pista negativa o cerrada no cancela a ninguna otra. La unica
dependencia admitida es la de las plazas de reserva, que dependen unicamente del resultado
positivo y con mecanismo activado de una primera variante.

| pista | estado inicial | motivo del veredicto |
|---|---|---|
| A | NUEVA | Sin corte duro de negativos faciles, sin asimetria de clase, sin descarte por margen |
| B | CERRADA_POR_DUPLICACION | ASL mas ponderacion por frecuencia, que es class_weight, mas coocurrencia, que es V4-E |
| C | NUEVA | La unidad de deteccion es el par instancia-etiqueta y no la instancia; V30-00 y P59-03 cerraron solo la versio |
| D | NUEVA | Numero efectivo de muestras es otra funcion de n_c, con saturacion, que el 1/(2 pi_c) de V03 |
| E | CERRADA_POR_DUPLICACION | Sobre un modelo lineal, el ajuste de logit es un desplazamiento de intercepto, y con umbral fijo un desplazami |
| F | CERRADA_POR_DUPLICACION | Tres etiquetas anidadas dan un subespacio factible de cuatro puntos, que V4-E ya enumero en doce variantes |
| G | NUEVA | El unico eje que modifica la geometria aprendida y no el objetivo, los pesos, la decision ni la entrada |
| H | NUEVA_CON_RESTAURACION | Con objeto de prueba de Dirac el transporte colapsa a distancia al centroide, que esta cerrada; solo aporta co |
| I | NUEVA | El peso viene de una alineacion de gradiente contra un objetivo meta, no de una regla de dificultad |
| J | NUEVA_PARCIAL | Familia polinomica de la perdida, ya muestreada por GCE con cuatro valores de q |
| K | CERRADA_SIN_OBJETIVO_ACCIONABLE | Convertir un conjunto conformista en decision binaria exige cortar en un punto |
| L | NUEVA | Objetivo minimax sobre grupos, sin grafo y sin tocar la funcion de perdida |
