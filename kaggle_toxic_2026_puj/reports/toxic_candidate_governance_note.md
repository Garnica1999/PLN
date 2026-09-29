# Nota de gobernanza - candidato A_cost_sensitive_downweight

Generada por `experiments/p69_08_paquete_candidato.py`. Fecha 2026-09-29T19:55:56+00:00.

Esta nota es **informativa**. No decide si el uso del conjunto de prueba seria licito,
no recomienda utilizarlo, y no genera ninguna submission. Su unico proposito es dejar
por escrito que esta validado, que no lo esta, y que procedimiento haria falta para
una eventual submission.

## Que esta validado

| aspecto | estado |
|---|---|
| reconstruccion del candidato desde sus artefactos | **SI**, cadena regenerada |
| reproduccion del resultado canonico | **SI**, R3, identica a 3,2e-11 |
| direccion del efecto entre particiones | **SI**, 3 de 3 positivas |
| especificidad de la correspondencia peso-ejemplo | **SI**, p = 0,000999 |
| compliance del pipeline | **SI**, 16 de 16 |
| provenance | **SI**, manifest con hashes y commits |

## Que NO esta validado

| aspecto | estado |
|---|---|
| magnitud poblacional del efecto | **NO**, IC95 por grupos incluye el cero |
| transferencia al leaderboard | **NO**, nunca se ha medido |
| comportamiento sobre datos fuera del desarrollo | **NO**, no se ha evaluado |

## La incertidumbre que permanece

```text
IC95 por grupos   [-0.0006906975, +0.0118175641]   incluye el cero
limite inferior   a seis diezmilesimas de cero
tamano            1483 filas, 1415 grupos
```

La incertidumbre es de **precision**, y la precision esta limitada por el tamano
efectivo del train. No es una incertidumbre sobre el signo: las tres particiones son
positivas y la permutacion es contundente. Es una incertidumbre sobre **cuanto**.

## Que evidencia existe, en una tabla

| prueba | resultado | pregunta que responde |
|---|---|---|
| Reproduccion | R3, 3,2e-11 | el numero es real y trazable |
| Particion canonica | +0,0082045862, 5 de 5 | el efecto existe en el reparto original |
| Segunda particion | +0,0105253130, 5 de 5 | sobrevive a otro reparto |
| Tercera particion | +0,0048825807, 4 de 5 | sobrevive a un tercero |
| Permutacion, 1000 | p = 0,000999 | la asociacion peso-ejemplo es especifica |
| Bootstrap por grupos | incluye el cero | la magnitud no queda resuelta |

## Que procedimiento necesitaria una eventual submission

Una decision **explicita y separada** de la investigacion, que reconociera por escrito:

```text
1. que se evaluaria un candidato cuyo bootstrap por grupos incluye el cero
2. que la medicion sobre el conjunto de prueba consumiria informacion hoy intacta
3. que el resultado no seria confirmatorio de nada nuevo, porque las tres
   particiones y la permutacion ya estan hechas y repetirlas no anade evidencia
4. que la decision sobre si ese uso es admisible corresponde a una regla,
   no a esta nota
```

### Lo que esta nota NO hace

```text
no decide si el uso del test es legal
no recomienda utilizar el test
no genera submission
no abre una cuarta particion
no propone una estrategia transductiva
```

## Criterio para reabrir la investigacion

Solo con evidencia **nueva y externa al experimento actual**:

```text
autorizacion normativa explicita
nuevo dato de entrenamiento legal
nuevo conjunto etiquetado permitido
cambio de reglas
```

**No** se considera justificacion suficiente:

```text
'quiero un IC95 que no incluya 0'
mas particiones sobre los mismos datos
mas semillas, mas repartos, mas grid de pesos
```

## El estado final

```text
V03                          FROZEN
A_cost_sensitive_downweight  PROMISING_UNCONFIRMED
research_status              EXPERIMENTAL_RESEARCH_CLOSED
governance_status            PENDING
submission                   0
test_development_use         0
leaderboard_tuning           0
```

La investigacion experimental termina aqui. El candidato queda congelado y
perfectamente reproducible en `reports/toxic_candidate_cost_sensitive_manifest.json`.

## El estado, sin ambiguedad

Dos documentos del expediente usan la clave `research_status` con valores distintos, y no son
contradictorios: responden a preguntas distintas. Se separan aqui.

| eje | valor | pregunta |
|---|---|---|
| `research_status` | `EXPERIMENTAL_RESEARCH_CLOSED` | se siguen buscando mecanismos nuevos? |
| `candidate_status` | `PROMISING_UNCONFIRMED` | el candidato esta adoptado? |
| `governance_status` | `PENDING` | la decision de gobernanza esta tomada? |

`CANDIDATE_PENDING_GOVERNANCE` fue una etiqueta anterior que colapsaba los dos ultimos ejes en
una palabra. Queda `SUPERSEDED`: no incorrecta, sino subespecificada. Detalle completo en
`reports/toxic_candidate_status_reconciliation.md`.

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p69_08_paquete_candidato.py`. Sin push.*