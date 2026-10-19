# P101 - Nota de gobernanza (P1): objetivos alternativos de entrenamiento

Fecha: 2026-10-14. Analisis documental; NO autoriza nada por si mismo.

## 1. La pregunta precisa

Permiten las reglas de tweet-toxic-2026-puj construir una COPIA ALTERNATIVA
de los objetivos de entrenamiento (HS/TR/AG) a partir de nuevas adjudicaciones
humanas sobre ejemplos del train oficial, manteniendo INALTERADAS las
etiquetas originales y evaluando contra la referencia oficial?

## 2. Que dicen las reglas (docs/competition_rules.md, verbatim relevante)

- "El dataset de prueba no debe ser usado para entrenamiento."
- "Esta competencia es academica, privada y grupal, dirigida unicamente a
  los estudiantes de la Maestria en Inteligencia Artificial."
- "No se pueden usar transformers ni modelos LLM... deben entrenarse desde
  cero."
- No existe ninguna clausula sobre modificar, derivar o sustituir los
  OBJETIVOS de entrenamiento del train a partir de anotaciones humanas.

## 3. Analisis

- La unica restriccion de uso de datos explicita es sobre el test.
- La construccion de objetivos alternativos (manteniendo labels originales y
  evaluacion oficial) no esta ni prohibida ni autorizada por las reglas.
- La autorizacion del propietario (2026-10-14) cubre auditoria y diseno, y
  declara expresamente que NO sustituye una aclaracion academica.

VEREDICTO P1: AMBIGUOUS.

## 4. Plantilla P95 de reapertura (completada para el registro; no autoriza)

| Campo | Contenido |
|---|---|
| hypothesis_id | P101-H1 (calidad de objetivos: correccion de celdas confirmadas) |
| mechanism | Ruido de anotacion verificable en el gold del train; no equivale a ninguna familia cerrada (no es representacion, arquitectura, corpus ni composicion) |
| historical_overlap | P91-P93 (auditoria de etiquetas, sin correccion); P97 (mecanismos); no duplica |
| data_allowed | train/dev y grupos near-dup UNION pseudo-autor |
| data_prohibited | test, holdout historico, holdout2 (salvo instrumento final), leaderboard, pseudo-labeling, recursos externos |
| minimum_experiment | A (targets originales) vs B (alternativos solo en celdas confirmadas), mismo modelo e hiperparametros |
| primary_metric | corrected_cv canonico (umbrales congelados 0.44/0.51/0.46) |
| validation_protocol | StratifiedGroupKFold (near-dup UNION pseudo-autor); >=4/5 pliegues; bootstrap pareado IC95; segunda particion confirmatoria |
| stopping_rule | cierre si el delta no alcanza el gate o no hay senal en 2 iteraciones |
| resource_budget | sin GPU sin autorizacion; presupuesto a fijar antes de ejecutar |
| adoption_gate | dOOF > +0.0376 con neto positivo + reproducibilidad + revision de transferencia |

Estado: completada PARA EL REGISTRO. No ejecutable hoy: sin evidencia humana
admisible (no hay revisores) y con gobernanza pendiente de aclaracion.

## 5. Consulta propuesta al profesor (redactada; NO enviada)

Texto exacto propuesto:

"Profesor: en el marco de la competencia del curso, solicitamos una
aclaracion sobre la siguiente practica, que NO modifica el dataset ni el
leaderboard: (1) auditar la calidad de las etiquetas del train con revision
humana ciega e independiente sobre una muestra aleatoria; (2) conservar
INTACTAS las etiquetas originales del dataset y evaluar siempre contra la
referencia oficial; (3) en el caso de que exista evidencia humana suficiente
de celdas mal anotadas, entrenar un unico experimento controlado usando
objetivos alternativos SOLO para esas celdas confirmadas (mismo modelo,
mismos hiperparametros, sin tocar el test). Es compatible con las reglas de
la competencia y con la evaluacion del curso?"

Estado: PENDING (no se envia sin decision del propietario; sin revisores, la
rama experimental esta bloqueada de todos modos).

## 6. Consecuencias

- No se modifican labels; P6 no se ejecuta.
- La consulta queda a disposicion si el propietario la activa.
- La auditoria documental (P0) y este analisis quedan registrados.
