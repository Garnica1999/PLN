
# Captura: reglas especificas de la competencia TweetAge_2026_PUJ

- Fuente: pagina Rules de la competicion, https://www.kaggle.com/competitions/tweet-age-2026-puj/rules
- Fecha de captura: 2026-09-29
- Metodo: renderizado de la pagina publica con lector externo (r.jina.ai); la seccion
  "Competition Rules" (reglas especificas del organizador) se renderizo completa. La seccion
  "Kaggle Competition Foundational Rules (Non-editable)" es el texto legal generico de Kaggle y
  no se transcribe aqui.
- Fidelidad: el bloque siguiente es el texto del organizador tal cual se renderizo, con sus
  tildes. No se modifica.
- Observacion: la seccion especifica NO menciona transformers, LLMs ni modelos preentrenados.
  Esas prohibiciones existen a nivel de curso y fueron confirmadas por Carlos el 2026-09-29:
  (1) transformers y LLMs prohibidos por indicacion del profesor; (2) embeddings prohibidos
  expresamente, preentrenados o entrenados desde cero. Registro en DECISIONS.md (D-001, D-075).

---

## Competition Rules

Esta competencia es académica, privada y grupal, dirigida únicamente a los estudiantes de la
Maestría en Inteligencia Artificial – Curso de Procesamiento del Lenguaje Natural (NLP).

### Reglas Generales

- Los equipos deberán trabajar en grupos (formados según indicaciones del curso).
- Cada equipo puede enviar múltiples predicciones, pero solo se considerará la mejor puntuación
  en el leaderboard público.
- Está permitido el uso de librerías comunes de NLP y ML, incluyendo: NLTK, Scikit-learn,
  TensorFlow, PyTorch, Pandas, Numpy.
- Se espera un notebook claro y bien documentado, que describa:
  - El procesamiento de los datos.
  - Las estrategias de preprocesamiento y representación de texto.
  - Los modelos y experimentos realizados.
- El dataset de prueba no debe ser usado para entrenamiento.
- El archivo de predicción debe respetar el formato definido en age_submission.csv.

### Evaluación Académica

- La nota de la competencia se calculará de la siguiente manera:
- 70% se evaluará en función del puntaje obtenido en Kaggle (F1 Score, leaderboard privado).
- 30% corresponderá al notebook entregado, valorando claridad, justificación de las técnicas y
  análisis de resultados.

### Escala de Calificación

- La calificación máxima (5.0) se asignará al equipo con mejor desempeño en Kaggle.
- A partir de esa referencia, las notas se irán reduciendo proporcionalmente hacia abajo según el
  ranking obtenido.
- El notebook puede subir o bajar ligeramente la nota final según su calidad.
