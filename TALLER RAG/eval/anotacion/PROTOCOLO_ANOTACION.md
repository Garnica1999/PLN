# Protocolo de anotación humana del gold v2

El gold v2 actual (`eval/gold/gold_ambiguedad_v2.jsonl`) es asistido por LLM y auditado contra la evidencia. Este protocolo produce su versión humana con doble anotación independiente, que es la deuda pendiente del artículo.

## Material

Un solo archivo: `eval/anotacion/anotacion_v2.xlsx`.

| Hoja | Uso |
|---|---|
| `Instrucciones` | Definición y pasos, resumidos |
| `Anotador 1` | Hoja del primer anotador |
| `Anotador 2` | Hoja del segundo anotador |
| `Acuerdo` | Calcula coincidencias, acuerdo observado y kappa de Cohen con fórmulas |
| `Consolidado` | Etiqueta final y marca `revisar` donde las dos etiquetas no coinciden |

No hace falta ejecutar ningún script para anotar ni para ver el acuerdo. El libro se abre en Excel o LibreOffice Calc.

## Definición operativa

Una pregunta es **ambigua** si, con el corpus del proyecto, admite más de una respuesta válida o más de una interpretación razonable. No es ambigua si una persona con acceso al corpus puede responderla sin preguntar nada más.

Casos que cuentan como ambiguos:

- Referencia genérica sin entidad: «¿Cuántos votos obtuvo la reforma?» (hay varias reformas).
- Unidad o escala sin especificar: «¿Cuánto cuesta el proyecto?» (pesos, dólares, millones).
- Comparación sin criterio: «¿Qué país tiene la mayor inflación?» (año, región, fuente).
- Sujeto omitido en una conversación: «¿Y Duque?» sin saber de qué se hablaba.
- Alcance temporal o institucional: «¿Cuándo se aprobó?» (Senado, Cámara, sanción).

Casos que no cuentan:

- La pregunta nombra la entidad con un nombre propio, una fecha o un lugar que la identifica.
- La pregunta pide un dato que solo un documento del corpus responde, aunque use un artículo definido.

## Procedimiento

1. Abrir `anotacion_v2.xlsx` en la hoja propia. No mirar la hoja del otro anotador.
2. Para cada fila, leer la pregunta y, si hay historial, el turno anterior.
3. En la columna `es_ambigua`, elegir `si` o `no` en la lista desplegable.
4. Si es ambigua, escribir dos interpretaciones en `interpretacion_1` e `interpretacion_2`, cada una con su respuesta corta en `respuesta_1` y `respuesta_2`. La respuesta debe incluir la cifra o entidad exacta, no una paráfrasis.
5. Si no es ambigua, dejar las interpretaciones vacías.
6. Usar `notas` solo para dudas que el otro anotador deba conocer al conciliar.

Tiempo estimado: 40 a 60 minutos por anotador. Se puede hacer en dos sesiones.

## Acuerdo y consolidación

La hoja `Acuerdo` se actualiza sola: cuenta las preguntas etiquetadas, las coincidencias, el acuerdo observado y el kappa de Cohen. La hoja `Consolidado` marca `revisar` en naranja donde las dos etiquetas no coinciden.

Regla de consolidación:

- Coinciden en `si` o en `no`: esa es la etiqueta.
- Discrepan: decide un tercer anotador y se registra en `notas`.
- Las interpretaciones se toman del anotador 1; si el anotador 2 propone una tercera, se añade como opción extra.

Cuando las dos hojas estén completas, el gold humano se genera con:

```bash
python -m src.scripts.conciliar_anotacion --escribir
```

Ese comando lee el libro, escribe `reports/tablas/anotacion_acuerdo.csv` y `eval/gold/gold_ambiguedad_v2_humano.jsonl`. El gold asistido no se sobrescribe.

## Qué se reporta en el artículo

- Kappa de Cohen sobre la etiqueta de ambigüedad.
- Número de discrepancias y cómo se resolvieron.
- Métricas del detector y del condicional sobre el gold humano, junto a las del gold asistido.

Mientras no existan las dos hojas, el artículo mantiene el gold asistido y lo declara como límite.
