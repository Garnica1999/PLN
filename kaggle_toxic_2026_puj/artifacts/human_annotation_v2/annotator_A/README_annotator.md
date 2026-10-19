# Anotacion principal (piloto interno) - instrucciones para el anotador/a A

Este paquete es autosuficiente. Contiene tres archivos:
1. README_annotator.md (este archivo)
2. annotation_guidelines_v2_frozen.md (guia operativa v2 congelada:
   PROJECT_OPERATIONAL_ANNOTATION_POLICY_V2)
3. annotator_A.csv (250 textos; anotaciones vacias)

## Objetivo

Anotar, de forma independiente, las tres etiquetas (HS, TR, AG) de 250 textos: 750
anotaciones de etiqueta; 3 respuestas posibles por anotacion (POSITIVE / NEGATIVE /
UNABLE_TO_DETERMINE).

## Que debes hacer

1. Lee primero `annotation_guidelines_v2_frozen.md` completo.
2. Revisa los textos SIN comunicarte con la otra persona anotadora.
3. Asigna HS, TR y AG de forma independiente entre si.
4. Usa EXCLUSIVAMENTE `POSITIVE`, `NEGATIVE` o `UNABLE_TO_DETERMINE` en las tres
   columnas de etiquetas. Los valores 0/1 NO son validos en esta ronda.
5. `uncertainty_notes`: escribe el motivo cuando uses UNABLE_TO_DETERMINE; tambien
   puedes anadir aclaraciones breves por etiqueta.
6. `tweet_note`: solo si la dificultad afecta a TODO el tweet (duda general). No
   sustituye las respuestas por etiqueta.
7. No consultes etiquetas originales, modelos, predicciones ni puntuaciones.
8. No cambies `review_id`, `text`, el orden de las filas ni `guideline_version`.
9. Devuelve el CSV completado sin convertirlo a Excel ni alterar el esquema.

## Ambiguedades que puedes registrar (no las resuelvas tu)

- HS: odio frente a ofensa.
- AG: agresividad frente a vulgaridad.
- Citas, discurso reproducido, sarcasmo e ironia.
- Contexto insuficiente.
- Cualquier otra duda material: registrala con UNABLE_TO_DETERMINE y una nota.

## Como devolver tu archivo

- Guarda el CSV completado con el nombre exacto: `annotator_A_response.csv`
- Devuelve SOLO tu archivo, por separado.
- No acuerdes respuestas con la otra persona antes de devolver tu archivo.

## Que pasa despues (protocolo de recepcion)

Tu archivo se validara (esquema, 250 filas, review_id, texto sin cambios, respuestas
permitidas, guideline_version); tus respuestas originales se almacenan sin cambios.
Los desacuerdos no se resuelven en esta ronda: se conservan y se remiten a una
adjudicacion posterior, separada y documentada. Ninguna discrepancia con material
del proyecto se considera automaticamente un error.

## Version

Guia aplicable: PROJECT_OPERATIONAL_ANNOTATION_POLICY_V2 (congelada el 2026-10-13). Es una politica operativa
interna del proyecto; no una definicion oficial de la competencia.
