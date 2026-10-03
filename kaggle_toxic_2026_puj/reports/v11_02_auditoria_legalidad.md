# V11-02: auditoria de legalidad del uso de la prueba (P1.1)

Generado por `experiments/v11_02_auditoria_legalidad.py` (V11-02 (P1.1, auditoria de legalidad del uso de la prueba)).

## 1. Que decide y por que va antes que cualquier experimento

El plan pide clasificar cada paso como permitido, ambiguo o prohibido, y fija la
consecuencia: si estimar la prevalencia con predicciones no etiquetadas resulta
ambiguo, **no se usa para generar submission** y la metodologia se estudia solo con
shift simulado.

No es un detalle administrativo. El proyecto lleva mas de diez fases dando por
supuesto que el uso transductivo de la prueba es legitimo: V9-02 estimo la
prevalencia con EM sobre predicciones de test y genero la sonda v17 con los
umbrales corregidos, y la v11 arranca tomando esos factores como ciertos. Si la
lectura correcta es que es ambiguo, varias decisiones heredadas estan en
cuarentena, y eso hay que saberlo antes de construir encima.

## 2. El texto que manda

Fuente: `docs/competition_rules.md`, capturada el 2026-09-22, texto
pegado por el organizador del curso. Las cuatro citas que sostienen la auditoria se
verificaron literalmente contra el documento: **True**.

- **El dataset de prueba no debe ser usado para entrenamiento.**
- *Esta permitido el uso de librerias comunes de NLP y ML, incluyendo: NLTK, Scikit-learn, TensorFlow, PyTorch, Pandas, Numpy.*
- *No se pueden usar transformers ni modelos LLM para la clasificacion del texto; no se pueden usar modelos como RoBERTa o BERT. Se pueden usar modelos de vectorizacion, pero deben entrenarse desde cero y hacer todo desde cero.*
- *Se espera un notebook claro y bien documentado, que describa: el procesamiento de los datos; las estrategias de preprocesamiento y representacion de texto; los modelos y experimentos realizados.*

La clave es la primera. Y el dato relevante es lo que **no** aparece: las reglas no
mencionan la transduccion ni para permitirla ni para prohibirla.

## 3. Clasificacion de cada paso

| paso | clase | motivo resumido |
| --- | --- | --- |
| Entrenar el clasificador sobre texto y etiquetas de entrenamiento | **A permitido** | Es la tarea de la competencia |
| Predecir probabilidades sobre el texto de la prueba con el modelo congelado | **A permitido** | Es literalmente lo que se pide: generar un submission para el test |
| Usar las ETIQUETAS de la prueba en cualquier forma | **C prohibido** | Las etiquetas estan ocultas en Kaggle y no estan en el dataset entregado |
| Usar el texto de la prueba para construir caracteristicas, entrenar el vectorizador o ajustar el modelo | **C prohibido** | Las reglas dicen que no se debe usar el dataset de prueba para entrenamiento, y un vectorizador ajustado sobre el texto de la prueba es entrenamiento |
| Estimar la prevalencia de la prueba con EM sobre probabilidades NO etiquetadas | **B ambiguo** | AQUI ESTA LA AMBIGUEDAD |
| Generar un submission usando la prevalencia estimada con EM sobre la prueba real | **C prohibido** | Es el paso que depende del anterior |
| Simular un desplazamiento de prevalencia sobre el OOF y estudiar alli la metodologia completa: estimador, correccion de prior, calibracion y regla de decision | **A permitido** | No toca la prueba en absoluto |
| Ajustar la regla de decision sobre el OOF con validacion anidada | **A permitido** | Usa solo datos etiquetados de desarrollo, y en regimen anidado que no evalua sobre lo que ajusta |
| Evaluar un candidato final en holdout2, que sale del desarrollo | **A permitido** | Son filas de entrenamiento con etiqueta, nunca vistas como validacion |
| Usar transformers, RoBERTa o BERT | **C prohibido** | Prohibido de forma expresa por el profesor, y vigente en toda la v11 |
| Usar recursos externos etiquetados, pseudoetiquetado o embeddings de terceros | **C prohibido** | Prohibido por la restriccion del profesor, que exige hacer todo desde cero |

Reparto: **5 permitidos, 1 ambiguos**, **5 prohibidos**.

## 4. El paso ambiguo, y por que es ambiguo de verdad

El paso en cuestion es **estimar la prevalencia de la prueba con EM sobre
probabilidades no etiquetadas**. Los dos argumentos son defendibles y ninguno es forzado:

- **A favor de permitido**: la regla prohibe entrenamiento, y aqui no hay
  entrenamiento. El modelo esta congelado, el EM no ajusta parametros del clasificador,
  y no se usa ninguna etiqueta. La correccion de prior es una tecnica estandar.
- **A favor de prohibido**: la intencion de la regla es que el test no participe del
  desarrollo del sistema, y usar sus predicciones para recalibrar la decision **es**
  participar, aunque no sea entrenar.

Cuando las dos lecturas son defendibles, la clasificacion honesta es **B**, y el
silencio de las reglas sobre la transduccion es precisamente lo que la hace ambigua.

## 5. Decision y su consecuencia

El uso de la prueba para estimar la prevalencia con EM es AMBIGUO, de modo que **no se usa para generar ningun submission**. La metodologia se estudia unicamente mediante shift simulado sobre el OOF, y las tres lineas centrales de la v11, que son la correccion de prior, la calibracion y la regla de decision, se evaluan bajo prevalencias simuladas. La pregunta de la transferencia al externo queda sin poder responderse con honestidad dentro de las reglas, y eso hay que decirlo en el informe final en vez de disimularlo.

### Lo que si se puede

- simular desplazamiento de prevalencia sobre el OOF del desarrollo
- estimar y comparar estimadores de prevalencia con shift artificial conocido
- calibrar con validacion anidada sobre el desarrollo
- ajustar reglas de decision con validacion anidada sobre el desarrollo
- evaluar candidatos finales en holdout2, que son filas de entrenamiento
- medir todo lo anterior con las tres cifras del addendum de v10

### Lo que no se puede

- usar la prevalencia estimada sobre la prueba real para generar un submission
- usar el texto de la prueba para construir caracteristicas o entrenar el vectorizador
- usar cualquier etiqueta de la prueba, que ademas no estan disponibles
- afirmar que una metodo mejora en el leaderboard sin haberlo medido ahi

## 6. Lo que esto le hace a la fase

Las tres lineas centrales de la v11, que son la correccion de prior, la calibracion y
la regla de decision, **siguen siendo enteramente estudiables**, pero bajo prevalencias
simuladas sobre el OOF en vez de sobre la prevalencia real de la prueba. Eso no es una
perdida pequena, y conviene decirlo con todas sus consecuencias:

1. Se puede medir si la metodologia **funciona** cuando el shift es real y conocido.
2. **No se puede medir si funciona en esta prueba concreta**, porque hacerlo exigiria
   usar la prueba para generar un submission, que es justo lo prohibido.
3. En consecuencia, la pregunta de la v11, si la correccion de prior y la decision
   consciente de F1 son una fuente real de mejora, **queda sin respuesta verificable**
   dentro de las reglas. Se puede responder la version metodologica, que es si el
   mecanismo recupera un shift conocido, y esa es la que se va a medir.

Esto no es un detalle que haya que ver con benevolencia. Es el limite real de la fase, y
si al final hay un candidato que mejora en lo simulado, la pregunta de si transfiere a
esta prueba concreta **no se podra responder con honestidad** sin autorizacion expresa
del profesor.

Fecha: 2026-09-26T21:36:56+00:00. Duracion: 0.0 s.
