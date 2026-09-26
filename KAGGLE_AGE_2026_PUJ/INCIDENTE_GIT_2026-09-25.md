# Incidente: el estado de la fase de lexicones quedo en un stash de git

**Fecha**: 2026-09-25 · **Proyecto**: KAGGLE_AGE_2026_PUJ · **Rama**: `kaggle-age-2026-puj-lexicons`

## Que paso

Mientras P6 (`scripts/phase39_fusion_v12.py`) se ejecutaba en segundo plano, otra sesion del
repositorio compartido (proyecto `KAGGLE_TOXIC_2026_PUJ`) hizo commit y cambio de rama. El estado
no versionado de la fase de lexicones quedo **aparcado en `stash@{0}`** ("WIP AGE lexicones fases
36-41"), por lo que desaparecio del arbol de trabajo:

`lexicons/**` (24 ficheros .txt), `src/lexicons.py`, `src/lexicon_lists.py`,
`scripts/phase36..phase41`, `scripts/fase00_entorno.py`, `reports/age_lexicon_*` de P0-P5,
`experiments/phase36..phase38*.json` y `plan/AGE_LEXICON_PLAN.md`.

## Que sobrevivio sin ayuda

- `experiments/phase39_fusion_v12.json` y `reports/age_lexicon_results.csv`: los escribio P6 despues
  del incidente.
- Todo `features/`: esta en `.gitignore` y el stash no toca ficheros ignorados. Incluye las tres
  matrices de bloques lexicos y las cinco predicciones OOF (`oof_argmax_v12*`).

## Recuperacion

1. Se identifico el stash y se extrajeron sus **44 ficheros** (del arbol del stash y, para los no
   versionados, de su tercer padre) a `/tmp/age_stash/`.
2. Se comprobo la integridad antes de sustituir nada: las listas originales regeneran las tres
   matrices **identicas bit a bit** a las que consumio P6.
   - `registro` 19 columnas: identica.
   - `vida` 16 columnas: identica (confirma que `hogar` se retiraba por quedar con 2 tokens).
   - `temas` 37 columnas: identica.
3. Se restauraron los ficheros originales y se regeneraron P7 y P8 con el codigo autentico.
4. Al recuperar `scripts/phase40_analisis_error.py` aparecio un error de ortografia
   (`Informe`/`informe`) que nunca se habia ejecutado porque la version que corrio estaba ya
   reescrita. Corregido; al reejecutarlo reproduce las mismas cifras.

## Impacto cientifico

**Ninguno.** No hizo falta reejecutar P6: las matrices que usa son las originales, y sus deltas
(+0,0001, +0,0000, -0,0009, -0,0018) y el veredicto "adopta: ninguno" se mantienen. P7 reproduce
exactamente los mismos numeros de rescate y dano.

## Lecciones operativas

1. **El stash es el mecanismo correcto** para aparcar trabajo en un repositorio compartido: no se
   pierde nada. El aviso llego despues de que el stash ya existia.
2. Un repositorio compartido exige comprobar `git branch --show-current` y `git stash list` antes
   de dar por terminado un trabajo largo.
3. Versionar pronto: los scripts de fase se versionan al escribirlos, no al cerrar la fase.
4. Las copias de seguridad fuera del arbol (`/tmp/age_rescate`, `/tmp/age_stash`) y los commits
   intermedios (`ed783cb`) fueron la red que permitio verificar antes de sustituir.
