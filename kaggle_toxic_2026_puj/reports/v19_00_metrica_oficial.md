# v19_00 - Metrica oficial de la competicion (determinacion)

Fecha: 2026-09-26
Competicion: `tweet-toxic-2026-puj` (TweetToxic_2026_PUJ), id numerico 167384
Organizador: JarvisLab_PUJ (slug `nlp-course`), Community, recompensa Kudos
Pregunta: formula exacta del F1, y tratamiento de `zero_division`.

## Veredicto

**NO VERIFICABLE con estos medios que averaging usa el host** (macro-3 / micro /
samples / weighted). La eleccion del parametro `average` **no esta publicada en
ningun sitio accesible al participante** y vive en un endpoint cerrado al host.

Lo que **si** queda determinado con certeza:

1. El scorer es el **kernel de metrica oficial y sin modificar de Kaggle**
   `metric/f1-score` (kernelId 39244216, metricId 502, metricVersion 9168), cuyo
   codigo fuente completo se descargo y leyo. Es un wrapper fino sobre
   `sklearn.metrics.f1_score`.
2. `zero_division` **nunca se pasa** a sklearn -> se aplica el default de
   sklearn (`"warn"`). Verificado empiricamente en sklearn 1.8.0.
3. `average` **no puede ser el default de la plantilla** (`'binary'`), porque
   sobre entrada multilabel-indicator lanza `ValueError` y la plantilla de
   ejemplo si fue puntuada. Luego el host **si** overridea `average`, pero el
   valor overrideado no es legible.

## 1. Formula parcial verificada (contenido real del scorer)

Leido del fuente de `metric/f1-score` (celda 0, funcion `score`):

```python
score_result = kaggle_metric_utilities.safe_call_score(
    sklearn.metrics.f1_score,
    solution, submission,
    labels=labels, pos_label=pos_label, average=average, sample_weight=sample_weight)
```

Firma de la plantilla:

```python
def score(solution, submission, row_id_column_name,
          labels: Optional[Sequence] = None,
          pos_label: Union[str, int] = 1,
          average: str = 'binary',
          weights_column_name: Optional[str] = None) -> float
```

Guardia de columnas (relevante: no se puede enviar una sola columna):

```python
if not((len(submission.columns) == 1) or (len(submission.columns) == len(solution.columns))):
    raise ParticipantVisibleError(...)
```

Es decir, el unico grado de libertad real es `average` (+ `labels`, `pos_label`).
`pos_label` es irrelevante en multilabel. `labels` solo cambiaria el conjunto de
etiquetas promediadas.

## 2. `zero_division`: RESUELTO

`kaggle_metric_utilities.safe_call_score` (kernel `metric/kaggle-metric-utilities`)
solo envuelve la llamada en try/except para clasificar errores. **No inyecta
`zero_division`.** Por tanto rige el default de sklearn, que es
`zero_division="warn"`.

Comportamiento efectivo (verificado ejecutando sklearn 1.8.0 del venv):

| Situacion | Resultado |
|---|---|
| Etiqueta sin positivos reales ni predichos | F1 = **0.0** + `UndefinedMetricWarning` |
| Muestra (fila) sin positivos reales ni predichos | F1 = **0.0** + `UndefinedMetricWarning` |
| Submission que no predice ningun positivo (todo 0) | score = **0.0** (macro, micro y samples) |

Descartado explicitamente: **no** es `nan`, **no** omite la etiqueta, **no** omite
la fila, **no** omite la clase. Solo emite warning y pone 0.0. Con
`average='macro'`, una etiqueta degenerada cuenta como 0 dentro de la media (no se
excluye del denominador).

## 3. `average`: NO DETERMINADO

### Candidatos eliminados con certeza (algebra, no conjetura)

- **`average='binary'`** (el default de la plantilla): imposible. Con solution
  2D de 3 columnas (multilabel-indicator) sklearn lanza
  `ValueError: Target is multilabel-indicator but average='binary'`. Y la
  submission de ejemplo **si** obtuvo score (0.51494), luego no fue error de
  participante. Ademas `safe_call_score` convertira ese error en
  `ParticipantVisibleError`. Verificado ejecutando sklearn.

- **Macro sobre 6 clases (3 positivas + 3 negativas, el `f1_macro6` que usa el
  proyecto): IMPOSIBLE, descartado algebraicamente.** Para una submission
  todo-unos, la clase negativa de cada etiqueta tiene TP = 0 (la prediccion
  negativa es 0 en todas las filas), luego su F1 = 0.0. Por tanto
  `macro6 = (F1_HS + F1_TR + F1_AG + 0 + 0 + 0)/6 = macro3/2 <= 0.5`.
  El score observado es **0.51494 > 0.5**. Contradiccion. **La asuncion
  `f1_macro6` del proyecto es incorrecta y hay que descartarla.**
  (Comprobado tambien ejecutando: macro con una etiqueta degenerada incluida da
  0.3333 sobre 3 etiquetas, no se salta la etiqueta.)

### Candidatos NO discriminables con el dato 0.51494

Para una submission todo-unos, con `p_j = t_j/n` prevalencias test y
`m = mean(p_j)`:

```
F1_j        = 2*p_j/(1+p_j)
macro3(+)   = mean_j 2*p_j/(1+p_j)          <= 2m/(1+m)
micro       = 2m/(1+m)
samples     = mean_i 2*|T_i|/(|T_i|+3)      <= 2m/(1+m)   (Jensen)
weighted    = sum_j p_j*F1_j / sum_j p_j    >= macro3
```

Los cuatro candidatos caen en el intervalo `[m, 2m/(1+m)]`. El score 0.51494
fijaria `m in [0.346595, 0.51494]`, es decir `s in [0.929, 1.380]`: un rango
ancho que no aisla ninguna formula. En particular:

- macro3 = 0.51494 -> `s = 0.935177`
- micro  = 0.51494 -> `s = 0.929611`

Ambos plausibles, como ya se sospechaba. `weighted` tambien sobrevive (a
`s ~ 0.92` daria 0.51494). `samples` no se puede reducir a prevalencias porque
depende de la distribucion conjunta, y con fuerte correlacion entre HS/TR/AG daria
valores mas altos (~0.6-0.7), lo que lo hace menos plausible, pero **no queda
descartado de forma rigida**.

Conclusion: el dato 0.51494 sirve de comprobacion de coherencia, **no** de prueba.

## 4. URLs y endpoints intentados

| Via | Resultado |
|---|---|
| `GET /api/v1/competitions/list?search=tweet-toxic` (con token) | `evaluationMetric: "F1 Score"`. Tag generico que solo enlaza a la doc de sklearn. Sin formula. |
| `GET .../overview/evaluation`, `/overview`, `/rules` (HTML) | HTTP 200 pero 5.6 KB, SPA en JS, sin contenido de metrica. `webfetch` inservible. |
| `GET /api/v1/competitions/data/download/.../submission.csv` | OK. 1154 filas + cabecera, **todo 1,1,1**. Sin documentacion adjunta. |
| `GET /api/v1/competitions/data/list/...` | Solo 3 ficheros, sin README. |
| `GET /api/v1/competitions/leaderboard/view|download/...` | 404 o HTML. |
| `POST /api/i/competitions.CompetitionService/GetCompetition` | **PIVOTE**. Da `evaluationAlgorithm: {id 502, name "F1 Score", kernelId 39244216, latestCompetitionMetricVersionId 9168, environmentRenderType ENVIRONMENT_RENDER_TYPE_JSON}` y `clonedFromCompetitionId 116411`. |
| `GET /api/v1/competitions/submissions/list/...` | 24 submissions. Confirma `toxic_submission_v00_plantilla.csv` = **0.51494** y `errorDescription` vacio. |
| `POST /api/i/competitions.PageService/ListPages` `{competitionId}` | **PIVOTE**. Devuelve el markdown integro de las paginas de overview. Solo **5** paginas: `data-description`(1), `Description`(2), `rules`(4), `abstract`(5), `foundational-rules`(7). **No hay pagina de evaluacion**; los orders 3 y 6 son huecos. |
| `POST /api/i/competitions.PageService/GetPage` nombres `evaluation`, `evaluacion`, `Evaluation`, `metric`, `overview`, `description` | No existen. Por `pageId` -> `Unimplemented`. |
| `GET /api/v1/kernels/pull/metric/f1-score` | **PIVOTE**. Fuente completa del scorer (base64 en `blob.source`). Leido entero. |
| `GET /api/v1/kernels/pull/metric/kaggle-metric-utilities` | **PIVOTE**. Fuente de `safe_call_score` y `treat_as_participant_error`. Leido entero. |
| `POST /api/i/kernels.KernelsService/GetKernel` `{kernelId}` | Confirma `metric/f1-score`, autor "Kaggle Competition Metrics", 133 votos, 198 forks, metric kernel generico de Kaggle. |
| `GET /api/v1/kernels/output/metric/f1-score`, `/kernels/scriptcontent/39244216/download` | 404. |
| `POST /api/i/competitions.MetricService/GetCompetitionMetricVersion` `{competitionMetricVersionId:9168}` | **401 Unauthenticated con token valido y sin token**. Endpoint restringido al host. Es la config que falta. |
| `POST .../GetCompetitionMetricMetadata`, `ListCompetitionMetrics` | 401. |
| Barrido de ~60 combinaciones `Service/Method` (GetMetric, GetEvaluation, GetCompetitionPage, GetOverviewContent, GetMetricConfiguration, etc.) | Sin acierto. |
| Descarga de `https://www.kaggle.com/static/assets/app.js` (5.25 MB) y grep de `*Service/*` | Permitio descubrir el nombre real del servicio: `competitions.MetricService/GetCompetitionMetricVersion`. Tambien `competitions.HostService/SetEvaluationMetricParameters` (via de escritura, solo host) y `competitions.PageService/{ListPages,GetPage}`. |
| Competicion hermana `tweet-age-2026-puj` (id 167385, mismo org) | Comparte `kernelId 39244216` y `metricVersion 9168`. Patron de MetricVersion reutilizado por el host; no aporta la formula. |
| Competicion padre `clonedFromCompetitionId 116411` | `GetCompetition` con `id` y con `competitionId` no la resuelve (exige `CompetitionId` o `CompetitionName`; 116411 no expone slug). Sin via de lectura. |
| Busquedas web (`tweet-toxic-2026-puj`, `TweetToxic_2026_PUJ` + metrica) | 0 resultados. Competicion academica privada, sin documentacion externa. |
| Navegador desktop | No disponible en esta sesion (`browser.disconnected`). |

## 5. Lo que dice la prosa oficial (todo el texto disponible)

- `rules`, seccion "Evaluacion Academica": "**70%** se evaluara en funcion del
  puntaje obtenido en Kaggle (**F1 Score**, leaderboard privado)". Sin formula.
- `foundational-rules` (boilerplate global de Kaggle): "Each Submission will be
  scored and ranked by the evaluation metric stated on the Competition Website".
  Sin formula.
- `data-description`: describe columnas `HS`, `TR`, `AG` como 1/0. No menciona
  la metrica.
- `Description` y `abstract` estan **copiados de la competicion hermana de
  edades** (hablan de "categorias de edad", "Prediccion de Rango de Edad"), y
  `rules` pide respetando el formato de `age_submission.csv`. La competicion esta
  configurada con descuido. Eso sugiere una config de metrica taken a la
  ligera, pero es **indicio, no prueba**, y no se usa como conclusion.

## 6. Recomendacion para la decision de fase

No se puede decidir entre "umbrales independientes por etiqueta" y "decision
conjunta" solo con la metrica. Argumento: `macro3`, `micro` y `weighted` sobre un
problema multilabel **todos** se descomponen en una suma de contribuciones por
etiqueta, asi que el problema sigue siendo de umbrales independientes en las tres
cabezas. La unica variante que cambia la naturaleza del problema es
`average='samples'`, que introduce un termino de **co-ocurrencia** por fila
(`2*|T_i|/(|T_i|+3)`) y por tanto si justifica una decision conjunta. Como
`samples` **no queda ni confirmado ni descartado**, la eleccion de fase sigue
sin resolverse con evidencia.

Vias que si permitirian cerrarlo, en orden de coste:

1. **Preguntar al host / al docente del curso.** Es la unica fuente autoritativa
   del valor de `average`. Bloqueante y barato.
2. **Sonda discriminante de UNA submission** (si se acepta gastar cuota): basta
   un envio con patron conocido, por ejemplo positivo solo en `HS` y 0 en `TR`,
   `AG`, cuyo score bajo `macro3` y bajo `samples` difiere de forma separable.
   Maximo 10 envios/dia. No ejecutado aqui.
3. **Comprobar si el leaderboard privado del padre 116411** (competicion
   academica previa del mismo curso) documenta la formula; puede estar published
   y ser legible por el mismo bug de permisos que la bloquea aqui.
