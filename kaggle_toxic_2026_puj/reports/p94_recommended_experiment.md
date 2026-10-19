# P94 - Recomendacion para la fase experimental futura

Fecha: 2026-10-13. Insumos: inventario P0 (24 familias), revision P1 (17 referencias
primarias), cruce P2 (17 propuestas auditadas, 0 genuinamente nuevas) y candidatas P3
(3 preseleccionadas, ninguna supera el criterio de novedad).

## Veredicto de la regla de parada (P5)

NO_NEW_MODELING_HYPOTHESIS

No existe, con la evidencia disponible, una tecnica de modelado genuinamente nueva
para el espacio permitido que justifique disenar y ejecutar otro experimento contra
V03. En consecuencia:

- No se disena ningun primer experimento (P4 no aplica: no hay hipotesis genuinamente
  nueva que recomendar).
- No se autoriza entrenamiento, ni predicciones, ni submissions.
- V03 permanece FROZEN como mejor resultado verificado (LB 0.78306; corrected_cv
  0.7772114525165721).

## Fundamento cuantificado

- Familias historicas revisadas: 24 (21 CLOSED, 1 MARGINAL, 2 BLOCKED). El inventario
  verifica 95 identificadores historicos contra las 459 filas reales del registro.
- Propuestas auditadas en P2: 17 -> 11 ALREADY_TESTED, 2 DUPLICATE, 3 NOT_FEASIBLE,
  1 INSUFFICIENT_EVIDENCE. Ninguna GENUINELY_NEW.
- Referencias primarias revisadas: 17 (12 revisadas por pares, 3 preprints marcados,
  2 internas/informativas); cada una mapeada a una familia medida o a una prohibicion.
- Las tres candidatas preseleccionadas (P3) y por que no pasan:
  1. H-P94-01 (perdida sustituta de F-beta/macro-F1): la formulacion literal ya se
     midio (V7-01: F1-suave -0.0033 vs control BCE con hueco de -0.000046, 3/5
     pliegues; focal -0.0024) y la leccion del proyecto es "aproximar mejor el
     objetivo no es mejor" (D-053). Las versiones con restricciones (TFCO) son una
     implementacion nueva del mismo mecanismo; P2 las prohibe como hipotesis nueva.
  2. H-P94-02 (marcado de alcance de negacion): la familia negacion/cita/sintaxis
     se cerro con medicion (E2-08-NEG 0.5267; E2-08-NEG-TXT -0.0115, 0/5 pliegues).
     Una variante de tokenizado no cambia el mecanismo.
  3. H-P94-03 (kNN instance-based): nunca medido como clasificador, pero sin
     evidencia a favor en la literatura revisada y con evidencia adyacente en contra
     (memorizacion del RBF V26-01; vecindad MARGINAL P59-10; prototipos cerrados
     P60-06). INSUFFICIENT_EVIDENCE; no supera el criterio de novedad.
- Regla de P1 cumplida: las referencias no cambiaron el conjunto de hipotesis viables,
  por lo que la busqueda bibliografica se cierra.

## Razon para no reentrenar aun (aunque existiera candidato)

1. El encargo lo prohibe (MODEL_TRAINING = NOT_AUTHORIZED; SUBMISSIONS = NOT_AUTHORIZED).
2. El historial demuestra que las mejoras locales no transfieren: R1-C1 (+0.0157
   local, -0.0108 LB), R2-MINDF5 (+0.0221, -0.0127), OPT-01 (+0.0240, -0.0070);
   freno de transferencia vigente desde v7/v8.
3. El unico candidato vivo (A_cost_sensitive_downweight) no es nuevo, esta congelado
   en gobernanza y su sonda externa perdio (LB 0.77522); su causa faltante
   (seleccion del baseline, D-216) no se resuelve con otro reentrenamiento generico.
4. La correccion de etiquetas (14 candidatas P93-R5) esta sin confirmar y su techo
   historico medido (~+0.009) queda por debajo de la banda de ruido de particion
   (0.017); usarla seria adelantar una gobernanza en pausa.
5. El techo de anotacion (frontera no observable; 40.49% de la region de insulto
   interpersonal etiquetada (0,0,0)) acota cualquier modelo de texto en este corpus.
6. El propio LB del proyecto forma un plateau de ~1 error estandar entre variantes
   de la familia campeona (0.77034-0.78306): el ranking a ese nivel es
   mayoritariamente ruido de muestreo.

## Presupuesto de ejecucion recomendado

- Para esta fase: 0 segundos de entrenamiento (fase de investigacion y diseno; ya
  ejecutada).
- Para una eventual reapertura (que hoy no se recomienda): sobre minimo de una
  prueba diagnostica acotada en CPU (<= 30 minutos, sin GPU), con pre-registro
  escrito y hasheado ANTES de ejecutar, y con el protocolo canonico de abajo. Ningun
  experimento de DL/GPU se recomienda mientras siga prohibido el material
  preentrenado y el DL desde cero siga por debajo de V03.

## Protocolo canonico obligatorio si algun dia se reabre

- StratifiedGroupKFold con grupos = near-duplicate UNION pseudo-autor (D-004/D-015),
  salvo auditoria que demuestre un defecto del protocolo historico (no la hay).
- Metrica primaria: la definicion canonica de corrected_cv usada para V03 (media
  de la F1 binaria por columna con los umbrales congelados 0.44/0.51/0.46, sobre el
  OOF agrupado). No se mezclan medias de pliegues con la metrica agrupada oficial.
- La particion de evaluacion no se usa para inventar, ajustar ni seleccionar la
  hipotesis; hiperparametros se controlan con diseno anidado o con una unica
  configuracion pre-registrada (sin rejillas nuevas sobre el punto medido).
- Prohibido: test, holdout, holdout2, leaderboard, pseudo-labeling, recursos
  externos, transformers/preentrenados.
- Gates vigentes: dOOF > +0.0376 con neto de celdas positivo en la misma
  configuracion (D-190/P86), mas la lectura de transferencia (una mejora local no
  es evidencia externa); un candidato de verdad exige segunda particion y
  pre-registro congelado.
- Los 14 candidatos de etiquetas P93-R5 no son justificacion automatica de features
  ni de correcciones; siguen PRESERVED_NOT_CONFIRMED.

## Condiciones que reabririan la busqueda (no presentes hoy)

1. Autorizacion del organizador para propagar etiquetas entre near-duplicados
   (hoy QUARANTINED; techo medido nulo).
2. Autorizacion explicita y gobernanza cerrada para modificar etiquetas de
   entrenamiento (hoy BLOCKED; 14 candidatas sin confirmar).
3. Evidencia primaria nueva, revisada por pares y replicable, de un mecanismo de
   modelado que no exista en ninguna de las 24 familias del inventario.
4. Cambio de las reglas de la competencia (p. ej., permitir preentrenados) o
   asignacion de datos nuevos autorizados: reabriria las familias F10/F12/M-P94-15
   con otro marco, no con el actual.
5. Acto de gobernanza especifico para el candidato PENDING (F14), con presupuesto y
   particion fijados de antemano.

## Bloque de decision final

```text
NO_NEW_MODELING_HYPOTHESIS
TRAINING_NOT_AUTHORIZED
SUBMISSIONS_NOT_AUTHORIZED
V03 = FROZEN
CORRECTED_CV = 0.7772114525165721
LEADERBOARD_REFERENCE = 0.78306
```

V03 remains the best verified leaderboard result.
