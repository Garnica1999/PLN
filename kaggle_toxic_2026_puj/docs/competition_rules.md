# Reglas oficiales de la competencia (texto verbatim del organizador)

Fuente: página Overview de https://www.kaggle.com/competitions/tweet-toxic-2026-puj
Fecha de captura: 2026-09-22. Texto pegado por Carlos (organizador: curso PLN,
Maestría en Inteligencia Artificial, Pontificia Universidad Javeriana).

---

## Overview

Competencia Kaggle – Predicción de Rango de Edad a partir de Tweets

En esta competencia los estudiantes deberán clasificar discursos de odio en
texto: HS (Hate Speech), TR (Targeted) y AG (Aggressiveness). El énfasis estará
en el procesamiento y representación del texto para que los modelos clásicos
puedan capturar mejor las señales de toxicidad.

## Dataset Description

El dataset contiene textos anotados con etiquetas de toxicidad. El objetivo es
predecir las columnas HS, TR y AG a partir del texto (text).

- Columna id: identificador único del texto.
- Columna text: el contenido del tuit o comentario.
- Columna HS: indica si hay discurso de odio (1 = sí, 0 = no).
- Columna TR: indica si el discurso está dirigido contra una persona o grupo
  (1 = sí, 0 = no).
- Columna AG: indica si el texto es agresivo (1 = sí, 0 = no).

### Archivos

1. Toxic_train.csv: textos de entrenamiento con sus etiquetas de toxicidad.
2. test.csv: textos de prueba; en Kaggle las columnas HS, TR, AG están ocultas.
3. submission.csv: archivo de ejemplo con el formato de los envíos
   (columnas id, HS, TR, AG).

## Competition Rules

Esta competencia es académica, privada y grupal, dirigida únicamente a los
estudiantes de la Maestría en Inteligencia Artificial – Curso de Procesamiento
del Lenguaje Natural (NLP).

Reglas generales:

- Los equipos deberán trabajar en grupos (formados según indicaciones del curso).
- Cada equipo puede enviar múltiples predicciones, pero solo se considerará la
  mejor puntuación en el leaderboard público.
- Está permitido el uso de librerías comunes de NLP y ML, incluyendo: NLTK,
  Scikit-learn, TensorFlow, PyTorch, Pandas, Numpy.
- Se espera un notebook claro y bien documentado, que describa: el
  procesamiento de los datos; las estrategias de preprocesamiento y
  representación de texto; los modelos y experimentos realizados.
- El dataset de prueba no debe ser usado para entrenamiento.
- El archivo de predicción debe respetar el formato definido en
  age_submission.csv. [nota: nombre heredado de la plantilla; el archivo real
  de la competencia es submission.csv]

## Evaluación académica

- 70 %: puntaje obtenido en Kaggle (F1 Score, leaderboard privado).
- 30 %: notebook entregado (claridad, justificación de técnicas y análisis de
  resultados).

Escala: la calificación máxima (5.0) se asigna al equipo con mejor desempeño en
Kaggle; a partir de esa referencia las notas descienden proporcionalmente según
el ranking. El notebook puede subir o bajar ligeramente la nota final.

## Restricción adicional (indicada por el profesor)

No se pueden usar transformers ni modelos LLM para la clasificación del texto;
no se pueden usar modelos como RoBERTa o BERT. Se pueden usar modelos de
vectorización, pero deben entrenarse desde cero y hacer todo desde cero.

## Aclaraciones de plantilla detectadas (no afectan la tarea)

- El título menciona "Predicción de Rango de Edad": es texto heredado de otra
  competencia; la tarea real es de toxicidad (HS, TR, AG).
- Menciona `test.csv` y `age_submission.csv`: los nombres reales son
  `toxic_test.csv` y `submission.csv` (verificado vía API de Kaggle).
- Los ids de ejemplo (1, 2, 3) no coinciden con los ids reales del test
  (20001–24999); el formato de columnas sí coincide.
