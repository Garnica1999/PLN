# V11-00: reconciliacion de referencias y auditoria del estado heredado de v10 (P0)

Generado por `experiments/v11_00_reconciliacion.py` (V11-00 (P0, reconciliacion de referencias)).

## 1. Por que esto va primero

La fase v11 se apoya en **tres cifras que vienen de fases distintas**, y hay que
comprobar que son coherentes entre si antes de construir nada encima: la referencia de
V03, congelada desde la fase v2; los factores de shift de prevalencia que dio V9-02; y
las cifras de los informes de v10, escritos en un worktree distinto. El plan dice que si
V03 no reproduce, hay que parar y resolver la infraestructura.

## 2. V03, la referencia congelada

| magnitud | valor |
|---|---|
| `corrected_cv` reconstruido | **0.7772114525165721** |
| referencia publicada | 0.7772114525165721 |
| diferencia | +0.000e+00 |
| coincide exacto | **True** |
| `sha256` de la submission | `7d7478f271e6edfd51afff85bad3eaa0...` |
| submission identica a la congelada | **True** |

F1 por etiqueta con los umbrales congelados de 0.44, 0.51 y 0.46:

| etiqueta | F1 |
|---|---|
| HS | 0.761689 |
| TR | 0.815730 |
| AG | 0.754215 |

## 3. Todas las comprobaciones

| comprobacion | valor | esperado | veredicto |
|---|---|---|---|
| V03 reproduce la referencia canonica | `0.7772114525165721` | `0.7772114525165721` | **superada** |
| numero de filas del OOF | `1483` | `1483` | **superada** |
| umbrales que se usaron | `[0.44, 0.51, 0.46]` | `0.44/0.51/0.46` | **superada** |
| sha256 de la submission v03 | `7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e` | `7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e` | **superada** |
| filas de la submission v03 | `1154` | `1154` | **superada** |
| filas de experiments.csv | `219` | `219` | **superada** |
| claves unicas del registro | `219` | `219` | **superada** |
| campos por fila | `23` | `23` | **superada** |
| filas de la fase v10 | `['V10-00', 'V10-01', 'V10-02', 'V10-03', 'V10-04', 'V10-05', 'V10-06', 'V10-07', 'V10-08', 'V10-09', 'V10-10', 'V10-11']` | `V10-00 a V10-11` | **superada** |
| los identificadores de holdout2 conservan su huella | `f6c6ed958abf728b8210d15c26ec43fbd71659a0918d36089a9c6d6cef1639dc` | `f6c6ed958abf728b...` | **superada** |
| grupos compartidos entre holdout2 y dev2 | `0` | `0` | **superada** |
| holdout2 declara que no es virginal | `NO ES VIRGIN` | `NO ES VIRGINAL` | **superada** |
| evaluaciones de holdout2 disponibles | `3` | `3` | **superada** |
| identificadores de holdout2 que estan en el desarrollo | `247` | `247` | **superada** |
| filas del fichero de identificadores frente a la configuracion | `247` | `247` | **superada** |
| gate_1_superado de V10-04 coincide entre informe y registro | `True` | `True` | **superada** |
| gate_a_superado de V10-05 coincide entre informe y registro | `False` | `False` | **superada** |
| gate_1_superado de V10-06 coincide entre informe y registro | `False` | `False` | **superada** |
| veredicto de V10-07 coincide entre informe y registro | `CERRADA` | `CERRADA` | **superada** |
| gate_b_de_mejora.superado de V10-10 coincide entre informe y registro | `False` | `False` | **superada** |
| gana_a_v03 de V10-11 coincide entre informe y registro | `False` | `False` | **superada** |
| la rama de la v11 es la correcta | `proyecto-kaggle-toxic-2026-v11` | `proyecto-kaggle-toxic-2026-v11` | **superada** |
| las ramas v10 y v11 coexisten | `2` | `2` | **superada** |
| el factor de V9-02 usa la prevalencia del desarrollo como denominador | `True` | `True` | **superada** |
| factores de V9-02 redondeados a 3 decimales | `{'HS': 0.936, 'TR': 0.955, 'AG': 0.938}` | `{'HS': 0.936, 'TR': 0.955, 'AG': 0.938}` | **superada** |

**25 de 25 comprobaciones superadas.**

## 4. Los factores de V9-02, que P1 va a reproducir

| etiqueta | factor a 3 dec. del plan | factor real | p_test EM | p implicita | p train completo | p desarrollo |
|---|---|---|---|---|---|---|
| HS | 0.936 | 0.9365 | 0.409192 | 0.436938 | 0.436742 | 0.436952 |
| TR | 0.955 | 0.9546 | 0.286452 | 0.300075 | 0.299827 | 0.300067 |
| AG | 0.938 | 0.9381 | 0.358669 | 0.382336 | 0.382438 | 0.382333 |

El denominador que explica el dato guardado es el **desarrollo**: **True**. Con el train completo el factor quedaria unos 0.0004 mas alto en HS y TR, que es exactamente la discrepancia que aparecio al contrastar de entrada.

El campo factor_vs_train de V9-02 es p_test_EM dividida por la prevalencia del DESARROLLO de 1483 filas, no por la del train completo de 1731. Con el train completo el factor queda unos 0.0004 mas alto en HS y TR. El nombre del campo induce a error y la v11 debe usar el denominador correcto.

Lo que esta comprobacion **no** hace es reproducirlos: solo lee lo que V9-02 escribio.
La reproduccion de verdad es el experimento P1, y hasta que corra no se puede afirmar que
el estimate siga siendo valido.

## 5. Una observacion sobre la trazabilidad que hay que resolver

- rama: **proyecto-kaggle-toxic-2026-v11**, commit base `23aac93`.
- el arbol tiene **48 ficheros sin commitear**.

**La fase v10 esta sin commitear.** Eso quiere decir que la rama
`proyecto-kaggle-toxic-2026-v10` tiene como unico commit el `23aac93` de la correccion
B-14, que es anterior a toda la fase v10. Las doce filas `V10-00` a `V10-11`, las
decisiones `D-068` a `D-077` y los 25 informes de v10 **existen en disco pero no en el
historial de git**.

La consecuencia practica es concreta: la columna `GitCommit` de la tabla maestra de la v11
no puede apuntar a un commit que contenga el trabajo de v10, porque no existe. Hay dos
salidas honestas, y la decision es del usuario:

1. **Commitear v10 primero**, para que la v11 partida de un commit que contenga la fase
   entera y la trazabilidad sea completa.
2. **Trabajar en v11 con v10 sin commitear**, y que la columna `GitCommit` apunte a
   `23aac93` con una nota explicita de que la fase v10 esta en el arbol de trabajo y no en
   el historial.

**No se ha commiteado nada** en esta fase, porque la instruccion vigente es no hacerlo sin
peticion expresa.

## 6. Veredicto

**P0 SUPERADO.** La referencia de V03 reproduce con coincidencia exacta, la submission
v03 conserva su huella, el registro esta integro, el holdout2 sigue congelado con su
advertencia, y los seis contrastes entre informes de v10 y sus filas del registro
coinciden. La infraestructura es valida y **la fase v11 puede continuar con P1**.

Fecha: 2026-09-26T21:31:30+00:00. Duracion: 0.5 s.
