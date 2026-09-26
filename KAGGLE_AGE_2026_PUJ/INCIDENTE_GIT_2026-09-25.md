# Incidente: perdida de ficheros no versionados por operacion git concurrente

**Fecha**: 2026-09-25 · **Proyecto**: KAGGLE_AGE_2026_PUJ · **Rama afectada**: `kaggle-age-2026-puj-lexicons`

## Que paso

Mientras P6 (`scripts/phase39_fusion_v12.py`) se ejecutaba en segundo plano, otra sesion del
mismo repositorio (proyecto `KAGGLE_TOXIC_2026_PUJ`) hizo commit y cambio de rama a
`proyecto-kaggle-toxic-2026-v6`. El `git reflog` lo confirma: los ultimos seis commits son todos
de la rama del proyecto toxico.

Consecuencias:

1. Los ficheros **versionados** de AGE que no existen en la rama toxico desaparecieron del arbol de
   trabajo al hacer checkout. Restaurados al volver a `kaggle-age-2026-puj-lexicons` (estaban en el
   commit `5aa2543`).
2. Los ficheros **no versionados** se perdieron por un `git clean` (los no ignorados):
   `src/lexicons.py`, `src/lexicon_lists.py`, `lexicons/**`, `scripts/phase36..phase41`,
   `reports/age_lexicon_*.md|csv`, `experiments/phase37..phase40*.json`, `plan/AGE_LEXICON_PLAN.md`.

## Que sobrevivio y por que

`features/` esta en `.gitignore`, y `git clean` sin `-x` **no** borra ficheros ignorados. Por eso se
conservaron:

- `features/lexicon_features_{registro,vida,temas}.npy` y sus `*_nombres.json` (las matrices que
  consumio P6);
- `features/oof_argmax_v12.npy` y los 4 `oof_argmax_v12__*.npy` (predicciones OOF de la referencia y
  de cada bloque, que son el sustrato del analisis de P7);
- `experiments/phase39_fusion_v12.json` y `reports/age_lexicon_results.csv`, escritos por P6 **despues**
  del clean.

Ademas se hizo copia de seguridad en `/tmp/age_rescate/`.

## Impacto cientifico

**Ninguno en el veredicto.** El resultado de P6 esta contenido por completo en
`experiments/phase39_fusion_v12.json` (referencia, 4 bloques reales, sus controles de permutacion y la
decision del gate), y P7 se puede ejecutar sin repetir P6 porque las predicciones OOF sobreviven.

Lo que si se pierde es la **reproducibilidad exacta** de P1-P5 (las listas de palabras y sus
SHA256), no el resultado. La reconstruccion de las listas se verifica por hash contra las matrices
supervivientes; cualquier diferencia se documenta explicitamente en `reports/age_lexicon_audit.md`.

## Leccion operativa

1. El repositorio es compartido con otro proyecto: todo commit de AGE debe ir en su rama y
   comprobarse con `git branch --show-current` antes y despues de ejecutarse.
2. Versionar pronto: los scripts de fase se versionan **al escribirlos**, no al terminar la fase.
3. Las copias de seguridad fuera del arbol (`/tmp/age_rescate`) han resultado decisivas.
