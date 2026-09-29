# Nota de gobernanza - A_cost_sensitive_downweight

Generada por `experiments/p69_07_cierre_final.py`. Fecha 2026-09-29T19:49:10+00:00.

Esta nota responde cinco preguntas y **no decide nada por su cuenta**. En particular no
declara licita ni ilicita ninguna estrategia: se limita a registrar que el candidato no
puede evaluarse sobre el conjunto de prueba real bajo el protocolo actual.

## 1. Que evidencia existe

| prueba | resultado | pregunta que responde |
|---|---|---|
| Reproduccion | R3, identica a 3,2e-11 | el numero es real y trazable |
| Particion canonica | +0.0082045862, 5 de 5 | el efecto existe en el reparto original |
| Segunda particion | +0.0105253130, 5 de 5 | sobrevive a otro reparto |
| Tercera particion | +0.0048825807, 4 de 5 | sobrevive a un tercero |
| Permutacion, 1000 iteraciones | p = 0.000999 | la asociacion peso-ejemplo es especifica |

```text
reproducibilidad      R3
particiones positivas 3 de 3
mejora minima         +0.0048825807
permutacion           0 de 1000 asignaciones equivalentes la alcanzan
```

## 2. Que falta para adoptar

Un unico componente de los seis:

```text
bootstrap por grupos compatible   NO
IC95 = [-0.0006906975, +0.0118175641]   incluye el cero
```

Y hay que decirlo con precision: el intervalo **no** dice que el efecto sea negativo,
dice que con el tamano efectivo del train no se puede descartar que lo sea. El limite
inferior esta a seis diezmilesimas de cero.

## 3. Que no puede hacerse bajo el protocolo actual

```text
evaluar el candidato sobre toxic_test        el test esta fuera del desarrollo
estimar el efecto con el test                idem
calibrar o ajustar umbrales con el test      idem
leer las etiquetas del test                  idem
ajustar o crear una submission adaptada      idem
buscar una particion que excluya el cero     seria seleccion, no evidencia
abrir una grid de pesos o de C               la hipotesis esta congelada
```

**El candidato no puede evaluarse sobre test real bajo el protocolo actual.** Se
registra como un hecho, no como un juicio sobre lo que otras estrategias podrian hacer.

## 4. Que informacion queda fuera del alcance actual

```text
la magnitud del efecto sobre la poblacion de prueba
si la mejora se transferiria al leaderboard
si el intervalo por grupos se estrecharia con mas datos
```

Las tres cosas requieren datos que el protocolo actual no permite usar. No son
cuestiones de esfuerzo ni de tecnica: son cuestiones de acceso a datos.

## 5. Que decision seria necesaria para una eventual submission

Una decision **explicita y separada** de la investigacion, que reconozca que:

```text
1. se estaria evaluando un candidato cuyo bootstrap por grupos incluye el cero
2. la medicion sobre el test consumiria informacion que hoy esta intacta
3. el resultado, sea cual sea, no seria confirmatorio de nada nuevo
   (las tres particiones ya estan hechas, y repetir no anade evidencia)
```

Esta nota **no toma esa decision** y no sugiere que deba tomarse ahora.

## 6. Criterio para una futura reapertura

Solo se consideraria reabrir con **evidencia nueva externa al experimento actual**:

```text
autorizacion normativa explicita
nuevo dato de entrenamiento legal
nuevo conjunto etiquetado permitido
cambio de reglas
```

**No** se considera justificacion suficiente:

```text
'quiero un IC95 que no incluya 0'
mas particiones, mas semillas, mas repartos de los mismos ejemplos
```

## El estado, congelado

```text
research_status              CANDIDATE_PENDING_GOVERNANCE

V03                          FROZEN
A_cost_sensitive_downweight  PROMISING_UNCONFIRMED
SUBMISSION                   NO
TEST                         NO
LEADERBOARD                  NO
```

## Los mecanismos cerrados, que siguen cerrados

```text
P66                              residualizacion
P67                              calibracion, correccion de umbral, blend de scores
P68-H1                           enrutado textual
P68-H2                           rescue weighting por correccion ajena
```

La unica hipotesis viva es A_cost_sensitive_downweight, en estado
`PROMISING_UNCONFIRMED`.

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p69_07_cierre_final.py`. Sin push.*