# V11-03: auditoria de cumplimiento y condiciones de parada

Generado por `experiments/v11_03_auditoria_cumplimiento.py` (V11-03 (auditoria de cumplimiento y condiciones de parada)).

## 1. Que hace y por que es el primer experimento que puede invalidar a los anteriores

Las restricciones de ejecucion de la v11 traen una lista de comprobaciones que la
seccion 18 exige verificar **explicitamente y no solo declarar**, y una lista de
condiciones de parada en la seccion 19. Este script las ejecuta sobre el codigo y los
artefactos, y es el primero de la v11 con potestad para decir que un experimento previo
no cumple.

## 2. Seccion 17, ASCII puro, que es mas estricta que la herramienta previa

La distincion importa y produjo un hallazgo real. La herramienta que se venia usando
`tools/verificar_caracteres.py` busca **alfabetos no latinos**, y por eso da un veredicto
correcto sobre un texto con acentos. La seccion 17 pide **ASCII puro**, que es mas
estricto. La diferencia son cuatro caracteres en la v11 y tres ficheros en lo heredado.

**En la v11 se corrigen**, y se deja constancia de que el numero de comprobaciones
coincidio antes y despues, que es la evidencia de que la correccion no toco ningun
resultado.

**En lo heredado de v10 no se corrigen**, por dos razones que conviene decir juntas:
modificar un script ya ejecutado rompe la correspondencia entre codigo y resultado
historico, que prohibe la seccion 14; y v10 debe quedar intacto. Se documenta como
desviacion conocida.

Ficheros heredados de v10 con caracteres no permitidos: **3**.
- `experiments/v10_02_ruido_particion.py`: U+00F3 LATIN SMALL LETTER O WITH ACUTE
- `experiments/v10_08_confirmacion_dev2.py`: U+00BF INVERTED QUESTION MARK
- `experiments/v10_10_regla_conservacion.py`: U+00F3 LATIN SMALL LETTER O WITH ACUTE

## 3. Todas las comprobaciones

| seccion | comprobacion | resultado | evidencia |
| --- | --- | --- | --- |
| seccion 18 | seccion 18, transformers (import de 'transformers') | conforme | ningun import |
| seccion 18 | seccion 18, transformers (import de 'sentence_transformers') | conforme | ningun import |
| seccion 18 | seccion 18, transformers (import de 'roberta') | conforme | ningun import |
| seccion 18 | seccion 18, transformers (import de 'bert') | conforme | ningun import |
| seccion 18 | seccion 18, external_classifier (import de 'xgboost') | conforme | ningun import |
| seccion 18 | seccion 18, external_classifier (import de 'lightgbm') | conforme | ningun import |
| seccion 18 | seccion 18, external_classifier (import de 'catboost') | conforme | ningun import |
| seccion 18 | seccion 18, third_party_pretrained_embeddings (import de 'gensim') | conforme | ningun import |
| seccion 18 | seccion 18, third_party_pretrained_embeddings (import de 'torchtext') | conforme | ningun import |
| seccion 18 | test_training (firma de uso) | conforme | ninguna firma de ajuste con datos de prueba o holdout |
| seccion 18 | test_threshold_tuning (firma de uso) | conforme | ninguna firma de ajuste con datos de prueba o holdout |
| seccion 18 | holdout_reuse (firma de uso) | conforme | ninguna firma de ajuste con datos de prueba o holdout |
| seccion 18 | external_labels, external_labeled_dataset, new_unauthorized_corpus | conforme | no se ha introducido ningun recurso externo; el unico corpus de embeddings es el propio, e |
| seccion 18 | third_party_pretrained_embeddings | conforme | los 19 .bin de FastText son artefactos propios ya entrenados por el proyecto, reutilizable |
| seccion 18 | test_calibration, test_feature_selection, test_decision_rule_optimization | conforme | ningun calibrador, selector ni regla de decision se ajusta con datos de prueba; el uso que |
| seccion 18 | leaderboard_tuning | conforme | el leaderboard no se ha consultado en la v11, y el registro de sondas no ha crecido |
| seccion 17 | ASCII puro en todo el contenido generado de la v11 | conforme | sin caracteres no permitidos |
| seccion 17 | ASCII puro en el contenido heredado de v10 (no se corrige, se documenta) | **NO CONFORME** | 3 ficheros heredados con acentos: experiments/v10_02_ruido_particion.py, experiments/v10_0 |
| seccion 19 | unexpected baseline change | conforme | V03 reproduce 0.7772114525165721 con coincidencia exacta |
| seccion 19 | unexpected baseline change, huella de la submission | conforme | sha256 7d7478f271e6edfd... frente a 7d7478f271e6edfd... |
| seccion 19 | holdout contamination | conforme | los identificadores de holdout2 conservan su huella, ninguna evaluacion gastada |
| seccion 19 | missing experiment provenance | conforme | 3 filas V11, todas con script y decision: True |
| seccion 16 | secret exposure | conforme | sin credenciales en el contenido generado |
| seccion 13 | coherencia entre el registro y el codigo | conforme | todas las filas V11 apuntan a codigo existente |
| seccion 19 | unexpected baseline change | conforme | no detectado en esta auditoria |
| seccion 19 | control no reproducible | conforme | no detectado en esta auditoria |
| seccion 19 | validation leakage | conforme | no detectado en esta auditoria |
| seccion 19 | test contamination | conforme | no se ha contaminado el test: el uso de sus predicciones en V11-01 quedo clasificado como  |
| seccion 19 | holdout contamination | conforme | no detectado en esta auditoria |
| seccion 19 | non-convergence | conforme | no detectado en esta auditoria |
| seccion 19 | missing artifact | conforme | no detectado en esta auditoria |
| seccion 19 | missing experiment provenance | conforme | no detectado en esta auditoria |
| seccion 19 | numeric discrepancy | conforme | no detectado en esta auditoria |
| seccion 19 | notebook output mismatch | conforme | no detectado en esta auditoria |
| seccion 19 | unexpected external resource | conforme | no detectado en esta auditoria |
| seccion 19 | secret exposure | conforme | no detectado en esta auditoria |
| seccion 19 | invalid generated characters | conforme | ya corregido: los 4 caracteres acentuados de la v11 se transliteraron |

**36 de 37 comprobaciones conformes.**

## 4. Sobre el uso de la prueba, que es la condicion de parada relevante

El uso de las predicciones de la prueba para estimar prevalencia **queda en cuarentena**
y no alimenta ningun candidato. V11-01 lo reprodujo como auditoria historica, y esa
reproduccion es legitima porque el apartado 3 permite reproducir artefactos congelados
para auditoria; lo que **no** se hace es usar su salida en ninguna correccion de prior,
calibracion ni regla de decision. El experimento se conserva, y lo que se marca es que su
resultado no es utilizable como entrada de la fase.

## 5. Decision sobre lo heredado de v10

Los acentos del contenido heredado de v10 NO se corrigen en v11. Modificar un script ya ejecutado rompe la correspondencia entre codigo y resultado historico, que prohibe la seccion 14, y ademas v10 debe quedar intacto. Se corrige solo si la fase que lo produjo se reabre, y mientras tanto se documenta como desviacion conocida.

## 6. Veredicto

**AUDITORIA NO SUPERADA.** Las comprobaciones siguientes requieren resolucion antes de
ejecutar nada mas:

- ASCII puro en el contenido heredado de v10 (no se corrige, se documenta): 3 ficheros heredados con acentos: experiments/v10_02_ruido_particion.py, experiments/v10_08_confirmacion_dev2.py, experiments/v10_10_regla_conservacion.py

Fecha: 2026-09-26T21:46:46+00:00. Duracion: 0.6 s.
