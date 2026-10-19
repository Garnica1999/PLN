# P93-R2 - Recepcion de las revisiones independientes C1 y C2

Fecha: 2026-10-13. Estado: **C_REVIEWS_RECEIVED_AND_VALIDATED**.
Sin entrenamiento, sin predicciones, sin cambios de etiquetas; sin
comparacion con el gold (fase posterior).

## 1. Recepcion

- Dos revisiones independientes de la ronda 1 (127 tweets): C1 y C2
  completaron copias EN EL SITIO en la carpeta del paquete; el template
  original quedo intacto. Copias byte a byte archivadas en responses/.
- Formatos: C1 sin columna text (7 columnas); C2 con text (8 columnas).
  Ambas usaron la guia v2.1 del paquete (las notas referencian su seccion 6).

## 2. Validaciones ejecutadas

- C1_127_filas: OK
- C2_127_filas: OK
- C1_ids_exactos: OK
- C2_ids_exactos: OK
- C1_orden_igual_al_template: OK
- C2_orden_igual_al_template: OK
- C1_tokens: OK
- C2_tokens: OK
- C1_version_v2: OK
- C2_version_v2: OK
- C1_columnas_7_sin_text: OK
- C2_columnas_8_con_text: OK
- C2_textos_sin_diferencias_materiales: OK
- template_intacto: OK
- paquete_p93r_intacto: OK
- copias_byte_a_byte: OK
- verificador_ascii_oficial: OK
- Aviso: C2: 21 filas con artefactos de export en el texto (comillas/espacios); residuales: ['P93R-0011'] (no afectan anotaciones)
- Aviso: notas humanas con caracteres no ASCII (C1=0, C2=31; excepcion documentada, no se convierten)

## 3. Descriptivos (sin gold; provisional)

| revisor | etiqueta | POSITIVE | NEGATIVE | UNABLE_TO_DETERMINE |
|---|---|---|---|---|
| C1 | HS | 30 | 54 | 43 |
| C1 | TR | 89 | 31 | 7 |
| C1 | AG | 42 | 80 | 5 |
| C2 | HS | 34 | 66 | 27 |
| C2 | TR | 94 | 29 | 4 |
| C2 | AG | 71 | 45 | 11 |

- Coincidencia cruda C1-C2: HS 97/127, TR 109/127, AG 86/127; triple exacto 65/127.
- Sobre las 176 celdas candidatas: C1 coincide con la respuesta A/B en 174, usa UTD en 2 y difiere en 0.
- Idem C2: coincide 159, UTD 8, difiere 9.
- Estos recuentos son descriptivos de la ronda 1; NO clasifican todavia
  resultados de adjudicacion (esa clasificacion requiere las decisiones
  finales de la ronda 2 y es una fase separada).

## 4. Material de ronda 2 preparado

- p93r2_round2_comparison_dual_reviewer.csv (interno): 176 celdas con las
  decisiones INICIALES de C1 y C2, respuestas y notas de A/B, notas de C1/C2
  y campos de decision final vacios. Con DOS revisores, la ronda 2 admite
  variantes (una decision conjunta, una decision por revisor, o un unico
  revisor designado): la eleccion corresponde al propietario y no se
  asume aqui. El gold permanece oculto a los revisores.

## 5. Hashes

| archivo | sha256 |
|---|---|
| artifacts/p93_adjudication_v1/responses/PROVENANCE.md | `251888e411155dd8065d04df8fbea6a1293356d53cd126a23c9a000b245a60d1` |
| artifacts/p93_adjudication_v1/responses/response_C1.csv | `f89f74cdabd9649d9894faef7064f9c73413d1d06129733cf829b7e3338c3fe1` |
| artifacts/p93_adjudication_v1/responses/response_C1_completado.csv | `7fd2e2a5a1cd8af1e9d5903628ef9489d6343f8b69bff26690e0370179529b70` |
| artifacts/p93_adjudication_v1/responses/response_C2.csv | `e0449a811fa8b64b6e6e1eb2cbbfbcc00bca36685f53c00eaf2de46e67830daf` |
| artifacts/p93_adjudication_v1/responses/response_C2_completado.csv | `22e33186b8efe73d587e7490d4ec1a1afa2d7ca15c542d56bf6302e3276b981b` |
| independent_review/adjudicator_C1.csv (original) | `f89f74cdabd9649d9894faef7064f9c73413d1d06129733cf829b7e3338c3fe1` |
| independent_review/adjudicator_C2.csv (original) | `e0449a811fa8b64b6e6e1eb2cbbfbcc00bca36685f53c00eaf2de46e67830daf` |
| reports/p93r2_round2_comparison_dual_reviewer.csv | `85402c207addec5242232ac8999bda2d906cf09eeffdae792092f0c58fe40be7` |

## 6. Estado

```text
C_REVIEWS_RECEIVED_AND_VALIDATED
ROUND2_MATERIAL_READY (interno; sin decisiones finales)
SOURCE_LABELS_UNCHANGED
V03 = FROZEN
```
- Verificador ASCII oficial: OK: 0 caracteres no permitidos en los ficheros propios del proyecto.
