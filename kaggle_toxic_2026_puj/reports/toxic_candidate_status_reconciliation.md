# Reconciliacion de las dos etiquetas de estado

Generado por `experiments/p69_09_reconciliar_estado.py`. Fecha 2026-09-29T20:02:42+00:00.

## La ambiguedad

Dos documentos de este expediente llevan una clave llamada `research_status` y no
coinciden:

| documento | valor |
|---|---|
| `a_cost_sensitive_downweight_final_evidence.json` | `CANDIDATE_PENDING_GOVERNANCE` |
| `toxic_candidate_cost_sensitive_manifest.json` | `EXPERIMENTAL_RESEARCH_CLOSED` |

Ninguna es incorrecta. Responden a preguntas distintas, y el nombre del campo estaba
haciendo dos trabajos a la vez.

## Los tres ejes, separados

| eje | valor | pregunta | respuesta |
|---|---|---|---|
| `research_status` | `EXPERIMENTAL_RESEARCH_CLOSED` | se siguen buscando mecanismos nuevos? | no |
| `candidate_status` | `PROMISING_UNCONFIRMED` | el candidato esta adoptado? | no |
| `governance_status` | `PENDING` | la decision de gobernanza esta tomada? | no |

## Que era `CANDIDATE_PENDING_GOVERNANCE`

```text
estado      SUPERSEDED
no es       incorrecta, es subespecificada
significaba que el candidato no estaba adoptado y que la decision de
            gobernanza estaba pendiente
problema    colapsaba dos ejes en una sola palabra
```

## Que NO se ha hecho

```text
no se han modificado los dos ficheros historicos
no se ha cambiado ninguna cifra historica
no se ha reescrito ningun informe ya publicado
```

Reescribir un informe publicado para corregir su vocabulario es exactamente lo que este
proyecto no hace. Se preserva el original y se documenta la relacion.

## El estado, sin ambiguedad

```text
research_status     EXPERIMENTAL_RESEARCH_CLOSED
candidate_status    PROMISING_UNCONFIRMED
governance_status   PENDING

V03                          FROZEN
submission                   0
test_development_use         0
leaderboard_tuning           0
```

A partir de aqui, cuando un documento de este proyecto diga `research_status`, se refiere
al primer eje y solo al primero.

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p69_09_reconciliar_estado.py`. Sin push.*