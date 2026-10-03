# Candidato congelado - A_cost_sensitive_downweight

Generado por `experiments/p69_08_paquete_candidato.py`. Fecha 2026-09-29T19:55:56+00:00.

## El estado, en tres palabras que no se confunden entre si

```text
research closed       no se buscan mas mecanismos
candidate rejected    NO: nada aqui dice que el candidato este mal
candidate adopted     TAMPOCO: nada aqui dice que deba enviarse
```

```text
research_status     EXPERIMENTAL_RESEARCH_CLOSED
candidate_status    PROMISING_UNCONFIRMED
governance_status   PENDING
```

## La comparacion final

| | V03 | A_cost_sensitive_downweight |
|---|---:|---:|
| corrected_cv | 0.7772114525 | 0.7854160387 |
| particion 2 | 0.7772114525 | 0.7779658848 |
| particion 3 | 0.7772114525 | 0.7788638872 |
| delta | - | +0.0082045862 |
| pliegues positivos | referencia | 5/5, 5/5, 4/5 |
| rescue | - | 99 |
| damage | - | 74 |
| net | - | +25 |
| bootstrap por grupos | - | [-0.0006906975, +0.0118175641] incluye 0 |
| permutacion | - | p = 0.000999 |
| estado | `FROZEN_REFERENCE` | `PROMISING_UNCONFIRMED` |

Los candidatos cerrados (META-BASES-RIDGE, B_meta36, D_svd300_rbf, y todos los
mecanismos de P66 a P68) quedan como contexto historico y no entran en esta tabla:
ninguno sobrevivio a su gate.

## El mecanismo, con su nombre correcto

**downweight basado en violacion de margen del propio V03**

```text
margen_favor = (p_V03 - umbral) * (2*y - 1)
violacion_l  = clip(-margen_favor, 0, None)      por etiqueta
error_score  = violacion_l.mean(axis=1)          por ejemplo, media de 3
z            = (error_score - mediana) / IQR
peso         = clip(1 - 0.5*z, 0.5, 1.5)
```

### Las descripciones que NO aplican

| descripcion | por que no |
|---|---|
| meta-model | P58 lo etiquetaba asi, y el codigo no construye ningun meta-modelo |
| stacking | no hay modelo de segundo nivel |
| residual correction | el score no se corrige despues; la frontera cambia al entrenar |
| rescue weighting | es P68-H2, y va en direccion **opuesta** |

La diferencia con P68-H2 merece una linea, porque es facil confundirlos:

```text
P68-H2                        SUBIR el peso donde V03 falla y otro modelo acierta
A_cost_sensitive_downweight   BAJAR el peso donde V03 falla, segun la violacion
```

P68-H2 se clasifico `DUPLICATE` y ademas apuntaba en la direccion que historicamente
empeoro. Este candidato va en la contraria, y es la unica hipotesis viva.

## Los seis estados historicos

| estado | valor | nota |
|---|---|---|
| `ORIGINAL` | +0.0082045862 | 2026-09-25, nunca enviado, nunca auditado |
| `REPRODUCED` | +0.0082045862 | R3, diferencia 3,2e-11 |
| `SECOND_PARTITION` | +0.0105253130 | 5 de 5 pliegues, semilla 20260928 |
| `THIRD_PARTITION_CORRECTED` | +0.0048825807 | 4 de 5 pliegues, semilla 20261001 |
| `PERMUTATION` | p = 0.000999 | 0 de 1000 superan al observado |
| `FINAL_STATUS` | PROMISING_UNCONFIRMED | 5 de 6 componentes |

Ninguno sustituye a otro, y ninguna cifra historica se ha sobrescrito.

## Lo que la permutacion prueba, y lo que no

```text
PRUEBA    la asignacion real de pesos contiene mas informacion
          que asignaciones equivalentes permutadas

NO PRUEBA generalizacion poblacional definitiva
NO PRUEBA mejora garantizada de leaderboard
```

El bootstrap por grupos sigue siendo la limitacion principal.

## Analisis por etiqueta, descriptivo

```text
HS   positivo en las TRES particiones   +19, +7, +18
TR   inestable                          -5, -4, +3
AG   inestable                          +11, +9, -11
```

Es evidencia descriptiva de estabilidad mecanica. **No** es una invitacion a optimizar
por etiqueta, y no se ha hecho ninguna modificacion especifica por etiqueta.

## Compliance

| concepto | valor |
|---|---:|
| test_development_use | 0 |
| holdout_use | 0 |
| leaderboard_tuning | 0 |
| submissions_nuevas | 0 |

Verificado por: operacion + fichero concreto, no por busqueda textual.

## Reproducibilidad

| elemento | valor |
|---|---|
| commit del manifest | `d57a3cb` |
| commit original | `c96cb0f` |
| OOF canonico | `559265816ce4f6312c87f008f90324f0` |
| OOF del candidato | `92f2ef7ee9563eaa97ca64ece7b0a7b3` |
| submission de V03 | `7d7478f271e6edfd51afff85bad3eaa0` |

El manifest completo, con todos los hashes y parametros, esta en
`reports/toxic_candidate_cost_sensitive_manifest.json`.

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p69_08_paquete_candidato.py`. Sin push.*