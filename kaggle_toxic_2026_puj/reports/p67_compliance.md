# P67 - Cumplimiento

Fecha 2026-09-29T16:51:20+00:00.

**24 comprobaciones, 0 fallos.**

Analisis del arbol sintactico: imports, atributos, nombres y llamadas.

| comprobacion | resultado | evidencia |
|---|---|---|
| transformers = 0 | OK | AST |
| BERT / RoBERTa / LLM = 0 | OK | AST |
| embeddings preentrenados = 0 | OK | AST |
| etiquetas externas = 0 | OK | solo HS, TR y AG del train |
| corpus externo = 0 | OK | ninguno |
| pseudo-labeling = 0 | OK | AST |
| test labels = 0 | OK | ningun script lo abre |
| test tuning = 0 | OK | el test no se lee |
| holdout reuse = 0 | OK | el CSV de grupos solo se uso en P66 para el bootstrap |
| leaderboard tuning = 0 | OK | no se consulto |
| submissions = 0 | OK | ninguna en P67 |
| MI_arbol NO se reentrena ni se optimiza | OK | se reutiliza el score de P66 verificado. P67 estudia integracion, no el candidato |
| V03 intacto | OK | sha256 7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e |
| V03 no modificado | OK | ni vectorizadores, ni clasificador, ni umbrales, ni OOF |
| protocolo nested | OK | a, b0 y beta en inner_train + inner_validation |
| sin rejillas de modelos | OK | una malla de umbral y siete valores de beta, declarados antes |
| reconciliacion hecha antes de fusionar | OK | P67-00 es el primer script y fija FINAL_VALID_P66_RESULT |
| control de atribucion ejecutado | OK | barajado dentro de cada pliegue, sobre los siete candidatos |
| control de calibracion ejecutado | OK | reentreno anidado de un escalar por etiqueta |
| control de sustitucion cruzada ejecutado | OK | la misma correccion con los siete candidatos |
| tasa base de los rescates medida | OK | enriquecimiento, que es la version con denominador |
| tracks omitidos declarados | OK | cada uno con su prerrequisito incumplido |
| acceso a red = 0 | OK | AST |
| presupuesto respetado | OK | 1 de 12 principales, 0 de 4 reservas |

## La regla ASCII y por que hizo falta una segunda comprobacion

La regla del proyecto es **0 bytes > 127**. La herramienta oficial `tools/verificar_caracteres.py` no la implementa: su lista de alfabetos vigilados es hebreo, arabe, devanagari, tamil y telugu, cirilico, griego y copto, hangul, kanji y kana, y emoji. **No incluye Latin-1 Supplement ni Latin Extended.**

Comprobado con un fichero de prueba que contenia `cafe con virgulilla, ene con virgulilla`, alfa, beta y un cirilico: la herramienta oficial reporto el cirilico y **no reporto los acentos latinos del mismo fichero**.

Por eso la regla se midio dos veces: con la herramienta oficial, y byte a byte sin depender de ella. La segunda comprobacion encontro un `e` con acento en la fuente de este script, que la primera habia dado por bueno. Se corrigio.

### Hallazgo 1: la herramienta oficial tiene un punto ciego

| regla | que vigila | que se le escapa |
|---|---|---|
| ASCII puro, 0 bytes > 127 | cualquier byte fuera de ASCII | nada |
| `verificar_caracteres.py` | alfabetos no latinos | `e` acentuada, `n` con virgulilla, y todo el rango Latin-1 Supplement |

La segunda comprobacion no es un redundante. Es la unica que aplica la regla del proyecto.

### Hallazgo 2: debt ASCII historico en el registro canonico

- `experiments.csv` tiene 300 bytes no ASCII.
- Las lineas afectadas son [29, 30, 31, 36, 42, 43] y siguientes, todas anteriores a P67.
- Estan presentes ya en el commit base: **no es una regresion de P67**.
- Bytes no ASCII que caen sobre las lineas escritas por P67: **0**.
- Las dos filas que P67 anade son ASCII puras: **True**.

No se corrigen. Reescribir filas historicas del registro esta prohibido, y hacerlo cambiaria la historia de las fases anteriores para gainar una propiedad de texto que no altera ningun resultado. Se declara como debt conocido.

### Resultado de las dos comprobaciones

| ambito | bytes no ASCII | veredicto |
|---|---:|---|
| scripts y documentos de P67 | 0 | ASCII PURO |
| filas anadidas a experiments.csv por P67 | 0 | ASCII PURO |
| filas historicas de experiments.csv | 300 | DEBT PREEXISTENTE, no se reescribe |

### Hallazgo 3: un script de P66 destruye un artefacto de P66 al re-ejecutarse

P67 necesita `features/p66_scores_candidatos.npz`, que vive en `features/` y esta en `.gitignore`, asi que no viaja entre worktrees. Se regenero con `experiments/p66_01_regenerar_scores.py`. Al hacerlo aparecio un efecto que P67 provoco sin pretenderlo:

```text
p66_01_regenerar_scores.py, linea 86
  CSV_REG = REPORTS / "p66_score_fusion.csv"

ese nombre lo escribe de mas, con SU tabla de verificacion de 8 filas.
el que lo posee legitimamente es p66_02_fusion_espacio_scores.py, con 214 filas.
```

El nombre colisiona entre dos scripts de la misma fase. Re-ejecutar el regenerador borra en silencio los 214 registros de la fase de fusion. P67 no se apercibio hasta que `git show --stat` del commit mostro 222 lineas borradas.

**Restaurado** con `git checkout HEAD~1 -- reports/p66_score_fusion.csv reports/p66_score_regeneration.json`, y comprobado que vuelven a ser 214 filas.

Lo que **no** hizo falta restaurar: `reports/p66_score_fusion.json`, que es el fichero que la reconciliacion lee de verdad, salio intacto. El dano era solo del CSV.

### Procedimiento para recuperar el npz sin tocar la historia

```bash
/home/carlos/venv/bin/python experiments/p66_01_regenerar_scores.py   # genera el npz
git checkout HEAD -- reports/p66_score_fusion.csv                     # deshace el dano
git checkout HEAD -- reports/p66_score_regeneration.json               # y su fecha
```

Es la unica forma limpia hoy, porque corregir `p66_01_regenerar_scores.py` seria editar el codigo de una fase anterior, que el encargo prohibe. Si alguna fase futura tuviera presupuesto para arreglarlo en vez de esquivarlo, el cambio es una linea: que `CSV_REG` de `p66_01` apunte a `p66_score_regeneration.csv`, que es el nombre que le corresponde.

> V03 remains the best verified leaderboard result.
