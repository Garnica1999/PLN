# P59 - Grafo de dependencias de pistas

Generado por `experiments/p59_12_informe_final.py`. Fecha 2026-09-28T14:38:20+00:00.

Este documento existe para que unas instrucciones futuras no vuelvan a crear dependencias
que no existen. El error concreto que corrige: en una version anterior de las instrucciones se
declaraba que P59-E y P59-F dependian de P59-B, y no es cierto. Ninguna de las dos ha mirado
un solo artefacto de B.

## El DAG

```text
                 B1  (P59-B, ventanas locales)
                  |
        +---------+---------+
        |                   |
      P59-T               P59-U
  (calibracion local)   (fusion global+local)


  P59-A   geometria local            (independiente, cerrada)
  P59-C   ruido de etiqueta formal   (independiente, marginal)
  P59-D   influencia                 (independiente, cerrada)
  P59-G   seleccion por estabilidad  (independiente, duplicada)
  P59-H   estructura no negativa     (independiente, equivalencia funcional)
  P59-I   perdida robusta            (independiente, marginal)
  P59-E   regularizacion por grafo   (independiente, marginal)
  P59-F   n-gramas generativas       (independiente, duplicada)
```

## Dependencias reales, una por una

| pista | depende de | por que |
|---|---|---|
| P59-A | nada | solo V03, pliegues y compliance |
| P59-C | nada | idem |
| P59-D | nada | idem |
| P59-G | nada | idem |
| P59-H | nada | idem |
| P59-I | nada | idem |
| P59-B | nada | idem |
| **P59-T** | **B1** | hereda el score local crudo de B1 para calibrarlo |
| **P59-U** | **B1** | hereda el score local crudo de B1 para fusionarlo con el de V03 |
| P59-E | nada | construye su propio grafo sobre el TF-IDF de V03 |
| P59-F | nada | solo gap analysis y novedad funcional |

## Lo que NO es una dependencia

```text
P59-T  NO depende del resultado de P59-U   -> T se cerro sin mirar U
P59-U  NO depende del resultado de P59-T   -> U se ejecuto sin mirar T
P59-E  NO depende de T, U ni B1             -> E construye su propio grafo
P59-F  NO depende de T, U ni E              -> F se cerro por algebra y precedente
```

T y U son independientes entre si. Comparten el hallazgo de B1, no un resultado. Una pista
negativa no bloquea a otra, y una positiva no modifica automaticamente a otra.

## Estado final

| pista | estado | experimentos | corrected_cv | delta |
|---|---|---:|---|---|
| P59-A | CLOSED | 1 | 0.7796406441 | 0.0024291916 |
| P59-C | MARGINAL | 0 | n/a (no se entreno) | n/a (no se entreno) |
| P59-D | CLOSED | 0 | n/a (no se entreno) | n/a (no se entreno) |
| P59-G | CLOSED_BY_DUPLICATION | 0 | n/a (no se entreno) | n/a (no se entreno) |
| P59-H | CLOSED_BY_FUNCTIONAL_EQUIVALENCE | 0 | n/a (no se entreno) | n/a (no se entreno) |
| P59-I | MARGINAL | 1 | 0.7785171129 | 0.0013056604 |
| P59-B | CLOSED | 1 | 0.7765868373 | -0.0006246152 |
| P59-T | CLOSED_BY_DUPLICATION | 0 | n/a (no se entreno) | n/a (no se entreno) |
| P59-U | MARGINAL | 1 | 0.7786914600 | 0.0014800075 |
| P59-E | MARGINAL | 1 | 0.7773636094 | 0.0000988776 |
| P59-F | CLOSED_BY_DUPLICATION | 0 | n/a (no se entreno) | n/a (no se entreno) |

Experimentos consumidos: **5 de 12 principales**, con 2 de reserva intactos.

Rutas de dependencia que quedaron documentadas para que no se repitan:

1. `P59-T` y `P59-U` necesitan el score local de `B1`; el resto de pistas no.
2. `P59-E` y `P59-F` son completamente independientes de `B1` y entre si.
3. `P59-B` es la unica fuente de scores locales, y produjo True en novedad.
