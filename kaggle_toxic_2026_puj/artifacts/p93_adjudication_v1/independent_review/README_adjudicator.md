# Revision independiente de etiquetas - instrucciones para la persona revisora

Este paquete es autosuficiente. Contiene tres archivos:
1. README_adjudicator.md (este archivo)
2. annotation_guidelines_v2_1.md (manual operativo v2.1 de la politica
   PROJECT_OPERATIONAL_ANNOTATION_POLICY_V2)
3. adjudicator_C.csv (una fila por tweet; anotaciones vacias)

## Objetivo

Realizar una revision independiente de las tres etiquetas (HS, TR, AG) de cada tweet,
como tercera persona revisora cualificada, aplicando el manual incluido.

## Que debes hacer

1. Lee primero `annotation_guidelines_v2_1.md` completo.
2. Anota TODAS las filas y las TRES etiquetas, de forma independiente entre si.
3. Usa EXCLUSIVAMENTE `POSITIVE`, `NEGATIVE` o `UNABLE_TO_DETERMINE` (los valores 0/1
   NO son validos).
4. Usa `UNABLE_TO_DETERMINE` cuando el manual no permita decidir; explica el motivo en
   `uncertainty_notes`.
5. `tweet_note`: solo si la dificultad afecta a TODO el tweet.
6. No consultes etiquetas originales, modelos, predicciones ni puntuaciones; no busques
   los tweets en internet; no uses herramientas automaticas.
7. No cambies `adjudication_review_id`, `text`, el orden de las filas ni
   `guideline_version`.

## Independencia de esta ronda

Esta primera ronda es COMPLETAMENTE independiente: no veras las respuestas de otras
personas anotadoras hasta que hayas entregado tu archivo. No se te muestra informacion
sobre por que estos tweets forman parte de la revision.

## Como devolver tu archivo

- Guarda el CSV completado con el nombre exacto: `adjudicator_C_response.csv`
- Devuelve SOLO ese archivo.

## Que pasa despues

Tu archivo se valida y se conserva sin cambios. En una segunda etapa podras comparar tus
respuestas con las de otras personas anotadoras y registrar, cuando sea necesario, una
decision final razonada. La guia aplicable es una politica operativa interna del piloto;
no una definicion oficial de la competencia.
