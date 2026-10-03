# G14 - decision

```text
G14 = NOT_ACTIONABLE
```

## Por que NOT_ACTIONABLE y no CLOSED

La taxonomia del brief distingue dos cierres que se confunden facil, y la diferencia importa para lo que se hace despues:

| Estado | Significado | Que implica |
|---|---|---|
| **CLOSED** | habia un canal y ya estaba cubierto por experimentos previos | hay precedente, no hay que buscarlo |
| **NOT_ACTIONABLE** | no hay ningun canal, ni cubierto ni descubierto | no hay nada que buscar |

Aqui es **NOT_ACTIONABLE**: el esquema tiene tres campos, y ninguno esta en la situacion de "legal, disponible en prediccion y sin usar".

## Los tres campos, uno por uno

| Campo | Verdicto | Motivo |
|---|---|---|
| `text` | en uso | es la entrada de todas las representaciones; sus dimensiones no capturadas por el TF-IDF fueron cubiertas por los canales de V31-03 y cerradas por V31-06 |
| `age_range` | PROHIBITED | es el target. Usarlo como entrada seria leakage por definicion |
| `id` | NOT_ACTIONABLE | indice de fila de 1 a N por split, con los **112.892** valores del test reapareciendo en train para filas distintas, y correlacion -0,00084 con la clase |

## Lo que no se hace, y por que

El brief anticipa este caso exacto y prohibe rescatarlo. No se busca: modulo, hash, posicion, patrones de digitos, buckets ni sinonimos. La razon es que un numero de fila no contiene informacion; lo que contiene es la oportunidad de ajustar una correlacion espuria sobre 169.336 filas, que es justo lo que el protocolo corregido del proyecto lleva veinte fases evitando.

## Lo que esto cierra, y lo que no

**Cierra** la linea G14 tal como estaba formulada: no hay una fuente de entrada nueva que explotar. Y cierra de paso, con evidencia, la esperanza de que el `id` fuera un snowflake con fecha. Se midio: tiene 6 digitos, no 18.

**No cierra** que quede senal por encontrar. Cierra que quede **en la entrada**, que es una afirmacion distinta y mas fuerte, porque el esquema esta enumerado por completo.

## Estado de la investigacion tras G14

| Linea | Estado |
|---|---|
| Correccion de score, global (P33) | CLOSED |
| Correccion de score, local por pares (P34) | INVALIDATED |
| Correccion por pendiente en el margen (G12) | CLOSED sin aporte, y baja el AUC |
| Canal linguistico como feature (V31) | CLOSED |
| Capacidad del meta-modelo (V29, V30) | CLOSED |
| Diversidad de ensembles historicos (V29) | CLOSED |
| **Informacion de entrada nueva (G14)** | **NOT_ACTIONABLE** |

Con las seis lineas cerradas y la entrada enumerada por completo, la conclusion de la fase de investigacion es que **no existe actualmente informacion de entrada legal, observable en prediccion y no utilizada por los modelos, que pueda abrir una nueva linea experimental**. Eso no es lo mismo que decir que V30 sea optimo: es decir que las vias que el protocolo permite estan recorridas.

## La cifra que resume el margen que queda

De los 65.403 errores de V30, **27,007 no los corrige ninguna fuente disponible**. Y G14 demuestra que tampoco hay un campo sin explotar que pudiera hacerlo. El techo de las lineas restantes esta medido, no supuesto.
