# V23: cierre de la fase

Worktree `/home/carlos/Git/PLN-v23`, rama `v23`, base `ba87c68`. **1 submission enviada,
1 rechazada. V03 REMAINS CHAMPION.**

## 1. Estado actual

**V03 = 0.78306 LB.** Sin cambios, sha256 `7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e`.

**Submissions: 22 en el registro, 3 disponibles.**

## 2. Top 3 tecnicas recomendadas, y que ocurrio con cada una

| # | tecnica | veredicto de P0 | resultado |
|---|---|---|---|
| 1 | **SVC kernel RBF sobre SVD(300)** | no cerrada, mejor senal del proyecto | **REJECTED en LB** (0.77938, -0.00368) |
| 2 | Elastic Net sparse | no existe en el proyecto | no ejecutada: P1-03 ya mostro que L1 y L2 son los extremos y que el interpolador no anade nada |
| 3 | FTRL-Proximal | no existe en el proyecto | no ejecutada: es una variante del mismo estimador que liblinear ya aproxima bien, y con p/N = 18.5 no hay margen |

## 3. Experimentos realizados

| candidato | mecanismo | CV | dCV | LB | dLB | estado |
|---|---|---:|---:|---:|---:|---|
| svd:rbf_svc (sigmoid) | SVC RBF sobre SVD(300) de la union TF-IDF | **0.790410** | **+0.013198** | 0.77938 | **-0.00368** | **REJECTED** |
| svd:rbf_svc (platt) | idem con Platt scaling | 0.785958 | +0.008747 | no enviada | -- | no enviada |
| supervised weighting (15 var.) | BM25, log-odds, chi2, MI, binario | 0.662-0.785 | -0.115 a +0.008 | no enviada | -- | cerrada en contra (historico) |
| Passive-Aggressive | clasificador online | 0.762605 | -0.0146 | no enviada | -- | cerrada (historico) |
| SGD (modified_huber) | clasificador online | 0.745404 | -0.0318 | no enviada | -- | cerrada (historico) |
| combo:downweight_violacion | sample_weight por error OOF | 0.785416 | +0.008205 | no enviada | -- | **invalida por fuga** |

## 4. Decision final

# V03 REMAINS CHAMPION

## 5. Por que fallo, con el mecanismo medido

La candidata reunia el mejor conjunto de evidencia interna de toda la historia del
proyecto: `corrected_cv` 0.790410 (+0.013198 sobre V03), **5 de 5 pliegues**, perfil de
celdas con 136 rescates contra 104 danos (**neto +32**), **McNemar exacto p = 0.0452**, y
saldo positivo en las tres etiquetas. Y el leaderboard cayo 0.00368.

El gap local-LB lo explica:

| modelo | corrected_cv | LB | gap |
|---|---:|---:|---:|
| V03 | 0.777211 | 0.78306 | **+0.005849** |
| V23 rbf_sigmoid | 0.790410 | 0.77938 | **-0.011030** |

Una diferencia de **0.017 en el gap**. V03 tiene un gap local-LB positivo y V23 lo tiene
negativo, y la mejora OOF de 0.0132 no cubre ni de lejos la inversion del gap.

**El mecanismo es medible y no fue una sorpresa estadistica: los vectores de soporte.**

| etiqueta | vectores de soporte | sobre 1.731 muestras |
|---|---:|---:|
| HS | 1.321 | **76 %** |
| TR | 1.044 | 60 % |
| AG | 1.317 | **76 %** |

Un `SVC` con kernel RBF, `gamma="scale"` sobre 300 dimensiones, deja entre el 60 % y el
76 % de las muestras como vectores de soporte. Esa es la firma de un kernel que memoriza.
Cada muestra pesa directamente en la solucion.

**La leccion, y es la que V23 pretendia encontrar:** la diferencia algoritmica existia y
era real, pero en el regimen real del problema (`N = 1483` en desarrollo, `p/N = 18.5` en la
representacion) la no linealidad no se puede pagar. No es un problema de encontrar la
tecnica correcta: es un limite de densidad de informacion. Coincide con lo que V5 ya habia
diagnosticado para el bloque denso, y ahora se confirma tambien para la no linealidad.

## 6. Lo que V23 deja, mas alla del resultado

1. **La mayor senal OOF de la historia no se transfiere.** +0.0132 con p = 0.0452 dio
   -0.00368 en el leaderboard. Un p-valor por debajo de 0.05 y 5 de 5 pliegues **no** son
   garantia de mejora competitiva en este problema. Es la confirmacion mas fuerte hasta ahora
   de que la metrica interna y la competitiva estan desacopladas, y de que la unica cifra que
   las conecta, el gap local-LB, no se puede calcular sin una submission.
2. **El gap local-LB de un candidato es un indicador disponible antes de enviar.** V03 tiene
   +0.0058. El problema es que el gap de una candidata solo se conoce **despues** de enviarla.
3. **La tasa de vectores de soporte predice la transferencia sin necesidad de enviarla.** Es
   un calculo sobre el propio modelo, y aqui habria dado la alarma: 76 % de soporte es
   memorizacion. Este es un indicador que la auditoria deberia incorporate en fases futuras.
4. **Un segundo resultado historico invalido por fuga**, igual que V4-C05: el uso de
   probabilidades OOF globales como `sample_weight` o como guia de seleccion. Dos de los
   tres mejores resultados historicos del proyecto estaban mal.
5. **La variante Platt no se envio**, y el motivo es concreto: su `corrected_cv` es menor
   (+0.0087 frente a +0.0132) y su ventaja estaba en el perfil de celdas, que es justamente
   lo que este resultado demuestra que no se traduce. Enviarla seria "probar a ver cual
   funciona", que es lo que el encargo prohibe.

## 7. Un fallo propio, corregido antes de enviar

El control de integridad de `v23_02` detuvo la ejecucion en la primera linea: los ids de la
submission congelada no coincidian con los del test. La causa es que `read_csv_robusto`
deja la columna `id` como `object` porque el fichero esta en latin-1 y separado por `;`.
El contenido y el orden eran identicos, pero sin forzar el tipo a `int64` la submission se
habria generado con los identificadores como cadenas. Se corrigio antes de construir nada.

El control siguiente, mas fuerte, si paso con la huella del champion: la reconstruccion de
V03 con el recipe propio dio **sha256 identico** `7d7478f271e6edfd...` y 0 de 3.462 celdas
distintas. Eso garantiza que la candidata se construyo sobre exactamente la misma base que
se audito.

## 8. Reglas respetadas

Sin push, merge, reset ni rebase. V03 intacto. El test solo se transformo: no se entreno
con el, no se leyeron etiquetas suyas, no se ajusto ningun umbral con el, y el SVD se
ajusto exclusivamente con el train. 3 submissions disponibles.
