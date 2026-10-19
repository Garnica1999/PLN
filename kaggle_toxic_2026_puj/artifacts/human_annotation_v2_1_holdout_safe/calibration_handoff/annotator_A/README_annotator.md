# Calibracion de anotacion - ronda de familiarizacion (anotador/a A)

Este paquete es autosuficiente. Contiene tres archivos:
1. README_annotator.md (este archivo)
2. annotation_guidelines_v2_1.md (manual operativo v2.1 de la politica PROJECT_OPERATIONAL_ANNOTATION_POLICY_V2)
3. calibration_sample.csv (15 textos; anotaciones vacias)

## Objetivo

Familiarizarte con el manual antes de la tarea principal y ayudar a evaluar la claridad
de las instrucciones. La ronda es independiente: no comentes los casos con la otra
persona anotadora hasta que AMBAS hayan entregado sus archivos.

## Que debes hacer

1. Lee primero `annotation_guidelines_v2_1.md` completo.
2. Anota los 15 textos de forma independiente y completa.
3. Decide HS, TR y AG por separado.
4. Usa EXCLUSIVAMENTE `POSITIVE`, `NEGATIVE` o `UNABLE_TO_DETERMINE` (los valores 0/1
   NO son validos).
5. Usa `UNABLE_TO_DETERMINE` cuando las reglas no permitan una respuesta responsable y
   explica la incertidumbre en `uncertainty_notes` (motivo obligatorio).
6. `tweet_note`: solo si la dificultad afecta a TODO el tweet; no sustituye las
   respuestas por etiqueta.
7. No consultes etiquetas originales, modelos, predicciones ni puntuaciones; no busques
   los tweets en internet; no uses herramientas automaticas.
8. No cambies `review_id`, `text`, el orden de las filas ni `guideline_version`.

## Como devolver tu archivo

- Guarda el CSV completado con el nombre exacto: `calibration_response_A.csv`
- Devuelve SOLO tu archivo, por separado.
- No acuerdes respuestas con la otra persona antes de devolver tu archivo.

## Que pasa despues (protocolo de recepcion)

Tu archivo se validara (esquema, 15 filas, review_id, texto sin cambios, respuestas
permitidas, guideline_version); tus respuestas originales se almacenan sin cambios. La
ronda sirve para evaluar la claridad de las instrucciones: no es una estimacion
poblacional de errores de anotacion ni forma parte de una metrica del dataset. Los
desacuerdos y dudas se documentan en la etapa posterior.

## Version

Manual aplicable: revision v2.1 (revision documental de procedencia; las reglas no
cambian respecto a la politica v2). Es una politica operativa interna del piloto; no
una definicion oficial del organizador ni una resolucion normativa definitiva.
