# Calibracion de anotacion - ronda de familiarizacion (anotador/a B)

Este paquete es autosuficiente. Contiene tres archivos:
1. README_annotator.md (este archivo)
2. annotation_guidelines_v1.md (la guia de anotacion, version v1)
3. calibration_sample.csv (15 textos para esta ronda)

## Objetivo

Familiarizarte con la guia antes de la tarea principal. No es un examen ni una prueba de
ningun sistema automatico: sirve para detectar ambiguedades en las instrucciones. Al
terminar esta ronda habra una conversacion de aclaracion con la otra persona anotadora y,
si hiciera falta, una version nueva de la guia (con changelog) que se congelara antes de
la tarea principal.

## Que debes hacer

1. Lee primero `annotation_guidelines_v1.md` completo.
2. Revisa los 15 textos del CSV individualmente y SIN comunicarte con la otra persona
   anotadora.
3. Asigna HS, TR y AG de forma independiente entre si.
4. Usa exclusivamente `POSITIVE`, `NEGATIVE` o `UNABLE_TO_DETERMINE` en las tres columnas
   correspondientes (`HS_annotation`, `TR_annotation`, `AG_annotation`).
5. Deja `uncertainty_notes` vacio si no hay ninguna aclaracion necesaria.
6. Usa `uncertainty_notes` para explicar brevemente las ambiguedades relevantes; no
   sustituye la respuesta estructurada.
7. No consultes etiquetas originales, modelos, predicciones ni puntuaciones.
8. No cambies `review_id`, `text`, el orden de las filas ni `guideline_version`.
9. Devuelve el CSV completado sin convertirlo a Excel ni alterar el esquema.

## Ambiguedades que puedes registrar (no las resuelvas tu)

- HS: odio frente a ofensa.
- AG: agresividad frente a vulgaridad.
- TR: su relacion con HS (los datos no prueban una regla).
- Citas y discurso reproducido.
- Sarcasmo.
- Contexto insuficiente.

No inventes reglas: si no puedes decidir, usa `UNABLE_TO_DETERMINE` y anota el motivo en
`uncertainty_notes`.

## Como devolver tu archivo

- Guarda el CSV completado con el nombre exacto: `calibration_response_B.csv`
- Devuelve SOLO tu archivo, por separado.
- No acuerdes respuestas con la otra persona antes de devolver tu archivo.

## Que pasa despues (protocolo de recepcion)

Tu archivo se validara con este protocolo: `verify_schema`, `verify_15_rows`,
`verify_review_ids`, `verify_text_unchanged`, `verify_allowed_responses`,
`verify_guideline_version`; tus respuestas originales se almacenan sin cambios. Solo
despues de recibir las dos devoluciones reales se analizara el acuerdo entre anotadores.
