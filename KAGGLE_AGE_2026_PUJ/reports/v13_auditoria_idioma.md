# V13 · P0a — Auditoría de idioma

Comprueba que el código, la documentación y los datos derivados están en español y sin caracteres de otros alfabetos. Motivo: en la fase de lexicones se colaron ideogramas CJK, cirílicos y puntuación china en listas de palabras y docstrings; en V13 eso se detecta de forma automática.

- **Ficheros auditados**: 588 (`.py`, `.md`, `.txt`, `.csv`)
- **Errores**: 0 · **Avisos**: 2

**Sin errores**: no hay ideogramas, cirílicos, árabe, hebreo, devanagari, tailandés, coreano, puntuación CJK ni mojibake en el proyecto.

## Avisos (emoji y dingbats: no son otro idioma, pero ensucian)

| Fichero | Línea | Categoría | Contexto |
|---|---|---|---|
| `DECISIONS.md` | 1027 | AVISO:flecha | - **V15b (poda evaluada)**: un representante por cluster, el de mayor eta2 **dentro del fo |
| `PROJECT_STATE.md` | 79 | AVISO:flecha | dentro del fold → **197 variables**. Macro-F1 **0,5965** frente a 0,6005 de V12: **delta - |

## Regla aplicada en V13

Todo el código, los comentarios, los docstrings, los informes y los nombres de variable se escriben en español sin caracteres de otros alfabetos. Los únicos signos no latinos permitidos son los matemáticos deliberados (delta, rho, sigma, epsilon, beta, mas/menos, menor-igual, mayor-igual) y la puntuación española.

