# Revision humana de etiquetas - instrucciones para anotadores (P92, guia v1)

Gracias por participar. Tu tarea: leer tweets y decidir, de forma independiente, si
presentan **discurso de odio (HS)**, si el discurso esta **dirigido contra una persona o
grupo (TR)** y si el texto es **agresivo (AG)**.

## Archivos

1. `calibration_sample.csv` (15 textos): ronda de familiarizacion. Anotalo completo ANTES
   de comentarlo con la otra persona anotadora.
2. Tu archivo personal (`annotator_A.csv` o `annotator_B.csv`, 250 textos): la tarea
   principal. NO lo comentes con la otra persona hasta terminar tu primera ronda.

## Orden de la calibracion (importante)

1. Las DOS personas anotan `calibration_sample.csv` de forma independiente y completa.
2. Se conservan las respuestas originales de la calibracion tal como se entregaron.
3. Solo despues de cerrar esa primera ronda se comentan las interpretaciones.
4. Si de esa conversacion salen cambios en las instrucciones, se publica una nueva version
   de la guia (con changelog); nunca se reinterpreta en silencio lo ya anotado.
5. La version aplicable a la tarea principal se congela ANTES de empezar tu archivo
   personal (`annotator_A.csv` / `annotator_B.csv`).

## Como responder

En cada fila, escribe en `HS_annotation`, `TR_annotation` y `AG_annotation` una de estas
tres respuestas EXACTAS (mayusculas): `POSITIVE`, `NEGATIVE`, `UNABLE_TO_DETERMINE`.

- Decide cada etiqueta por separado.
- `UNABLE_TO_DETERMINE` = no puedes decidir con responsabilidad con lo que tienes (p. ej.
  falta el tweet al que responde). No es una etiqueta mas: es una declaracion de
  incertidumbre.
- En `uncertainty_notes` puedes anotar el motivo (opcional, breve).
- No cambies las columnas `review_id`, `text` ni `guideline_version`; no modifiques el
  texto ni el orden de las filas.

## Definiciones (las unicas disponibles)

- HS: "indica si hay discurso de odio".
- TR: "indica si el discurso esta dirigido contra una persona o grupo".
- AG: "indica si el texto es agresivo".

Son definiciones breves y no hay criterios operativos oficiales: no hay respuestas
"trampa". Si dudas, usa `UNABLE_TO_DETERMINE` y anota por que.

## Reglas

- NO consultes las etiquetas originales, predicciones de modelos ni ningun material del
  proyecto: tu juicio debe ser independiente.
- NO busques los tweets en internet.
- NO uses herramientas automaticas (LLM, clasificadores) para decidir.
- NO discutas casos con la otra persona anotadora antes de terminar tu ronda completa.
- Los desacuerdos no se resuelven aqui: se conservan ambas respuestas y se adjudican
  despues.

## Entrega

Devuelve el CSV completo con tus respuestas. No borres filas ni columnas. Si tienes dudas
sobre el formato, pregunta antes de empezar.

La guia completa esta en `p92_annotation_guidelines_v1.md` (repositorio del proyecto).
