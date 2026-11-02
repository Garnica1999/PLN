# Gold estándar

Mínimo 40 preguntas: 20 en `dev` y 20 en `test`. Ninguna pregunta de `test` se usa para ajustar prompts, umbrales o parámetros.

## Esquema

Cada línea de `questions.jsonl` es un objeto:

```json
{
  "id": "smoke-001",
  "question": "¿Qué dijo el documento sobre el acuerdo?",
  "lang": "es",
  "type": "directa",
  "split": "dev",
  "reference_doc_ids": ["<document_id>"],
  "reference_chunk_ids": ["<chunk_id>"],
  "reference_span": "fragmento exacto del documento que responde",
  "reference_answer": "respuesta de referencia en una frase",
  "criteria": "qué debe contener la respuesta para considerarse correcta"
}
```

- `type`: `directa`, `comparativa`, `ambigua` o `sin_respuesta`.
- `reference_doc_ids` y `reference_span` son obligatorios. `reference_chunk_ids` se llena después de indexar el piloto.
- `reference_span` es el texto literal que responde, copiado del documento. Se usa para medir cobertura de respuesta.
- Las preguntas `sin_respuesta` llevan `reference_doc_ids` vacío y `reference_span` vacío.
- Una pregunta es válida solo si identifica el documento de forma única. Se excluyen las que usan referencias vagas ("el país", "la medida", "la empresa"), las que se responden con varios documentos (tasas de cambio, conteos globales), las que mezclan idiomas y las generadas desde boilerplate. La exclusión se marca con `"excluded": true` y `"exclusion_reason"`, y se reporta en el informe.

## Procedimiento

1. Muestrear documentos del piloto y leer el texto.
2. Escribir la pregunta y copiar el fragmento que la responde.
3. Verificar que el documento exista en el piloto y en el corpus completo.
4. Balancear: Bluesky y prensa, español e inglés, fechas de 2024 a 2026.
5. Dos anotadores revisan una submuestra y se reporta el acuerdo.
6. `split=dev` para ajustar; `split=test` se congela.

## Gold standard consolidado

- `gold_standard.xlsx`: plantilla para compartir. Una sola columna `revisor`, una columna `validado` con desplegable Sí/No/Corregir, enlaces clicables y hoja de resumen con el avance.
- `gold_standard_<nombre>.xlsx`: copia por revisor, con su nombre ya escrito en la columna `revisor`. Se genera con `python -m src.scripts.export_gold_excel --reviewer Nombre`.
- `gold_standard.jsonl`: 43 preguntas (19 desarrollo, 24 prueba), 38 respondibles y 5 sin respuesta. Campo `validated` en falso hasta la revisión humana.
- `gold_standard.csv`: la misma información en texto plano.
- `gold_standard.md`: tabla legible por split.

Para validar: cada persona abre su copia `gold_standard_<nombre>.xlsx`, revisa cada fila contra el documento citado y marca `validado` con Sí, No o Corregir. Al final se comparan las copias por `id`: una fila es válida cuando dos revisores marcan Sí. Esa decisión se consolida en `gold_standard.jsonl` (`reviewer` y `validated`) y se regeneran el CSV y el Markdown con `python -m src.scripts.build_gold_standard`.

Las preguntas sin respuesta no entran en Recall@k: se evalúan por abstención.


## Estado de validación

Carlos revisó las 43 filas: 31 con Sí y 12 con correcciones aplicadas. La segunda revisión de Julián, Erika y Óscar está pendiente; sus copias `gold_standard_<nombre>.xlsx` tienen el campo `validado` vacío.

El `gold_standard.jsonl` consolidado es la fuente de verdad. No ejecutes `build_gold_standard` después de la revisión: regeneraría el gold desde los archivos fuente y perdería las correcciones. Para cambiar una fila, edita el JSONL y vuelve a exportar el Excel con `export_gold_excel`.
