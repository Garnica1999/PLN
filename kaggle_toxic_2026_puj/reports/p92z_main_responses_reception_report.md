# P92-Z - Recepcion de las respuestas principales (250 x 2)

Fecha: 2026-10-13. Estado: **MAIN_RESPONSES_RECEIVED_AND_VALIDATED**.
Sin entrenamiento, sin predicciones, sin modificacion de etiquetas ni de la
guia; sin distribucion.

## 1. Hallazgo principal y desviacion de protocolo

- Se recibieron respuestas de AMBOS anotadores para la MUESTRA PRINCIPAL de
  250 tweets (review_id P92W-0001..0250): A entrego un archivo sin columna
  text; B entrego dos archivos (con y sin text, anotaciones identicas).
- La calibracion de 15 tweets (P92W-C01..C15) preparada en P92-Y NO tiene
  respuestas registradas en esta entrega: su estado sigue PENDING.
- El protocolo de P92-Y preveia la calibracion ANTES de la ronda principal
  (MAIN_ANNOTATION = NOT_DISTRIBUTED). Las respuestas principales recibidas
  implican que la ronda principal se ejecuto por decision del propietario;
  la desviacion se registra como evento de proceso, sin revertirlo ni
  simular la calibracion.

## 2. Validaciones ejecutadas

- A_250_filas: OK
- B_250_filas: OK
- B2_250_filas: OK
- A_ids_esperados: OK
- B_ids_esperados: OK
- B2_ids_esperados: OK
- A_sin_duplicados: OK
- B_sin_duplicados: OK
- A_tokens_validos: OK
- B_tokens_validos: OK
- B2_tokens_validos: OK
- A_version_v2: OK
- B_version_v2: OK
- A_columnas_7_sin_text: OK
- B_columnas_8_con_text: OK
- B2_columnas_7_sin_text: OK
- B_textos_sin_diferencias_materiales: OK
- sin_reservados: OK
- B_vs_B2_identicos: OK
- textos de B: 215/250 exactos; 17 con artefacto de comillas; 16 con artefacto de comillas+espacios; residuales: ['P92W-0048', 'P92W-0038'] (avisos, no bloquean)

### Notas de formato (informativas)

- Orden de A vs paquete: igual=False (divergencias en indices [203, 204]; transposicion
  adyacente sin efecto en el mapeo por review_id).
- Orden de B vs paquete: igual=True.
- Aviso: textos de B con variantes residuales (no afectan anotaciones): ['P92W-0048', 'P92W-0038']

## 3. Distribuciones de respuesta (descriptivas)

| anotador | etiqueta | POSITIVE | NEGATIVE | UNABLE_TO_DETERMINE |
|---|---|---|---|---|
| A | HS | 36 | 111 | 103 |
| A | TR | 140 | 77 | 33 |
| A | AG | 79 | 157 | 14 |
| B | HS | 35 | 108 | 107 |
| B | TR | 168 | 69 | 13 |
| B | AG | 110 | 109 | 31 |

- Notas no vacias: A=117, B=142; tweet_note: A=0, B=3.

## 4. Coincidencia cruda A-B (descriptiva, provisional)

- Celdas coincidentes: HS 199/250, TR 201/250, AG 172/250; las tres etiquetas iguales en 131/250 filas.
- Sin kappa ni inferencia en esta fase; el analisis de acuerdo formal
  corresponde a una fase posterior, si el propietario la autoriza.

## 5. Estado

```text
MAIN_RESPONSES_RECEIVED_AND_VALIDATED
CALIBRATION = PENDING_HUMAN_RESPONSES (sin respuestas en esta entrega)
GUIDELINE = v2.1 (sin cambios)
MAIN_ANNOTATION = RESPONSES_RECEIVED (250 x 2; validacion de recepcion)
V03 = FROZEN | NO NEW MODELING PHASE
```

## 6. Hashes

| archivo | sha256 |
|---|---|
| PROVENANCE.md | `7c812cfa49bd5e37787e268ca28bfae6c4a55445974160356a066249866d480c` |
| annotator_A_completado.csv | `27328e91e41f1f9ac8ce325b9bae773cf3b0dff0edfca28b400589b6cde782bc` |
| annotator_B_completado.csv | `c06f846426bf34e4b362db5b330ca247f1d987e283996a56272c0a4a0a8e8067` |
| response_A.csv | `599832fc398940078aa28fc074286db74a123fbee778eec4b2c6885605ff8a3a` |
| response_B.csv | `fc213a143e72517f2cf04ded265cc457a1e413e067bda3466b789351653f8480` |
| response_B_sin_texto.csv | `770a87353eb1db9d64a4e361173b5f52997cef2905c3e4099943407f016d763f` |

## 7. Acciones pendientes (para el propietario)

1. Confirmar si la calibracion de 15 tweets se realizara, se pospone o se
   omite oficialmente (registrar la decision).
2. Confirmar que la ronda principal se considera iniciada y con que guia
   aplicable (los usos observados en las notas referencian la v2.1).
3. Autorizar o no el analisis de acuerdo A-B y la discusion posterior.
