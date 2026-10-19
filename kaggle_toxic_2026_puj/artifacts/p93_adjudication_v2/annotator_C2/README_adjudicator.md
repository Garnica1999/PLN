# Segunda ronda de revision - instrucciones para la persona revisora (C2)

Este paquete contiene tres archivos:
1. README_adjudicator.md (este archivo)
2. annotation_guidelines_v2_1.md (manual operativo v2.1)
3. adjudication_round2_C2.csv (176 celdas; columnas finales vacias)

## Contexto

En la primera ronda revisaste 127 tweets y registraste tus respuestas y notas. En esta
segunda ronda revisaras 176 celdas (pares tweet-etiqueta). Un mismo tweet puede aparecer
en varias filas, una por etiqueta a evaluar: NO son tweets distintos.

## Que veras en cada fila

- El texto completo del tweet y la etiqueta objetivo (`target_label`).
- TU respuesta inicial (`reviewer_initial_response`) y TU nota inicial.
- Las respuestas de dos personas anotadoras de la ronda de revision previa
  (`annotator_A_response`, `annotator_B_response`) para esa etiqueta, como evidencia
  adicional. No se incluyen sus notas.

## Que debes hacer

1. Revisa cada celda y decide si CONFIRMAS tu respuesta inicial o la CAMBIAS.
2. Escribe tu decision final en `final_response`: POSITIVE / NEGATIVE /
   UNABLE_TO_DETERMINE. Si confirmas, repite tu respuesta inicial.
3. Escribe una justificacion en `final_rationale` para CADA cambio y para CADA caso
   que quede en UNABLE_TO_DETERMINE.
4. `remaining_uncertainty`: opcional, para incertidumbre que persista.
5. No estas obligado/a a cambiar una respuesta solo porque las otras dos respuestas
   coincidan: decide segun la guia v2.1.
6. No consultes ningun otro material; no busques los tweets; no uses herramientas
   automaticas.
7. Devuelve SOLO el CSV completado como `adjudication_round2_C2_response.csv`.
