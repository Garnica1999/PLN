# P92-T-R - Solicitud de confirmacion directa a los anotadores A y B (borrador)

Estado: BORRADOR; NO enviado. Lo entregan los responsables del proyecto, por
separado a cada anotador y sin comunicacion entre ellos, si deciden cerrar la
incertidumbre de procedencia. No modifica ninguna respuesta original.

## Texto propuesto (identico para A y para B)

```text
Hola. Gracias por tu ronda de calibracion. Antes de la ronda principal
necesitamos dejar documentado el significado exacto de los valores que usaste
en tu archivo, sin cambiar ninguna de tus respuestas.

En tu archivo usaste 0/1 en lugar de POSITIVE / NEGATIVE /
UNABLE_TO_DETERMINE. Para dejar constancia, confirma por favor (marca una
opcion por linea):

1. En las tres columnas de etiquetas (HS_annotation, TR_annotation,
   AG_annotation):
   - El valor 0 significaba:
     ( ) pertenece a la clase (POSITIVE)
     ( ) no pertenece a la clase (NEGATIVE)
     ( ) otro: __________
   - El valor 1 significaba:
     ( ) pertenece a la clase (POSITIVE)
     ( ) no pertenece a la clase (NEGATIVE)
     ( ) otro: __________

2. En la columna uncertainty_notes:
   - El valor 1 significaba:
     ( ) dificil de determinar para cualquier clase (duda general del tweet)
     ( ) otra cosa: __________
   - El valor 0 significaba:
     ( ) sin duda general / sin comentario
     ( ) otra cosa: __________

3. Si algo quedo ambiguo, puedes aclararlo aqui: ______________________

Tus respuestas originales se conservan exactamente como las entregaste;
esto solo documenta su significado. Basta con responder este mensaje.
```

## Uso de las respuestas (protocolo)

1. Recoger las respuestas por escrito, por separado, sin comunicacion entre
   A y B antes de responder.
2. Guardarlas verbatim como archivos nuevos bajo
   artifacts/human_annotation_v1/calibration_responses/ (sin modificar los
   CSV existentes).
3. Con ambas respuestas: emitir un addendum NUEVO que actualice el estado de la
   codificacion (este documento y el addendum P92-T-R no se reescriben).

Nota: esta solicitud no revela respuestas de otros anotadores, etiquetas
originales ni informacion de modelos; solo pregunta por el significado de los
valores del propio archivo.
