# P66 - Dano y restauracion de un artefacto, y marca DO_NOT_RERUN_UNGUARDED

Generado por `experiments/p67_03_cierre_documental.py`. Fecha 2026-09-29T17:10:47+00:00.

## Que paso

P67 necesita `features/p66_scores_candidatos.npz`. Ese fichero vive en `features/`, que
esta en `.gitignore`, asi que **no viaja entre worktrees**. La unica forma de tenerlo es
re-ejecutar su regenerador, `experiments/p66_01_regenerar_scores.py`.

Al hacerlo, ese script escribe `reports/p66_score_fusion.csv` con **su propia** tabla de
verificacion de 8 filas, encima de los **214 registros** de la fase de fusion que ese
fichero contiene de verdad.

```python
experiments/p66_01_regenerar_scores.py   linea 86
  CSV_REG = REPORTS / "p66_score_fusion.csv"
```

El nombre colisiona entre dos scripts de la misma fase. `p66_02_fusion_espacio_scores.py`
es el dueno legitimo de ese fichero.

## Como se detecto

No salto durante la ejecucion: el script termino con codigo 0 y su propia verificacion
dijo `7/7 verifican`.

Salto al hacer `git show --stat` del commit, que mostro `222 --------` en ese fichero.
**Un fichero de resultados historico perdio 214 registros sin que ningun script fallara.**

## Lo que se hizo

```bash
git checkout HEAD~1 -- reports/p66_score_fusion.csv
git checkout HEAD~1 -- reports/p66_score_regeneration.json
```

Comprobado despues: 214 filas, y el `mejor_real` de `p66_score_fusion.json`
(que salio intacto) sigue siendo el de siempre.

## La marca

```text
DO_NOT_RERUN_UNGUARDED
```

`experiments/p66_01_regenerar_scores.py` tiene esta marca. Ejecutarlo sin el `git
checkout` de despues destruye un artefacto de su propia fase.

### Procedimiento seguro

```bash
/home/carlos/venv/bin/python experiments/p66_01_regenerar_scores.py
git checkout HEAD -- reports/p66_score_fusion.csv
git checkout HEAD -- reports/p66_score_regeneration.json
```

## Por que no se parchea el script

El arreglo es una linea: que `CSV_REG` de `p66_01` apunte a
`p66_score_regeneration.csv`, que es el nombre que le corresponde.

No se aplica porque **editar el codigo de una fase anterior rompe la correspondencia**
entre ese codigo y los resultados que publico. Un fichero de P66 que hoy dice una cosa
empezaria a decir otra, y quien audite la fase dentro de un ano no podria distinguir
que linea produjo que numero.

Se documenta la via limpia en su lugar.

## El hermano que no se toco

`reports/p66_score_fusion.json` **salio intacto**: el regenerador escribe el CSV, no el
JSON, y la reconciliacion de P67 lee el JSON. El dano fue solo del CSV.

## Leccion general

Un script de una fase puede destruir artefactos de su propia fase sin fallar. La unica
defensa barata es comparar el numero de filas de los ficheros de resultados antes y
despues de una re-ejecucion, o mirar el `--stat` del commit.

*Generado por `experiments/p67_03_cierre_documental.py`. Sin push.*