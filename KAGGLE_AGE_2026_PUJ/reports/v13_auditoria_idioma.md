# V13 · P0a — Auditoría de idioma

Comprueba que el código, la documentación y los datos derivados están en español y sin caracteres de otros alfabetos. Motivo: en la fase de lexicones se colaron ideogramas CJK, cirílicos y puntuación china en listas de palabras y docstrings; en V13 eso se detecta de forma automática.

- **Ficheros auditados**: 304 (`.py`, `.md`, `.txt`, `.csv`)
- **Errores**: 1 · **Avisos**: 0

## Errores (alfabeto no latino, mojibake o control)

| Fichero | Línea | Categoría | Detalle | Contexto |
|---|---|---|---|---|
| `plan/V13_ADVANCED_RESEARCH_PLAN.md` | 180 | **ERROR:chino (han)** | CJK UNIFIED IDEOGRAPH-6307 | Los dos指标 responden a preguntas distintas y divergen: **la redundancia lineal no explica e |

## Regla aplicada en V13

Todo el código, los comentarios, los docstrings, los informes y los nombres de variable se escriben en español sin caracteres de otros alfabetos. Los únicos signos no latinos permitidos son los matemáticos deliberados (delta, rho, sigma, epsilon, beta, mas/menos, menor-igual, mayor-igual) y la puntuación española.

