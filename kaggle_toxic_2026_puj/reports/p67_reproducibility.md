# P67 - Reproducibilidad

Fecha 2026-09-29T16:45:27+00:00.

| elemento | valor |
|---|---|
| rama | `proyecto-kaggle-toxic-2026-p67` |
| commit base | `51b0c2b` |
| semilla | 42 |
| V03 corrected_cv | `0.7772114525165721` |
| V03 leaderboard | `0.78306` |
| sha256 de la submission de V03 | `7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e` |
| n filas | 1483 |
| scripts | `p67_00_reconciliacion.py`, `p67_01_controles_decisivos.py`, `p67_02_cerrar.py` |
| artefactos de P67 | `reports/p67_*.md`, `.json`, `.csv` |
| figuras | 6 en `figuras/p67/` |

## Como reproducir

```bash
/home/carlos/venv/bin/python experiments/p66_01_regenerar_scores.py
/home/carlos/venv/bin/python experiments/p67_00_reconciliacion.py
/home/carlos/venv/bin/python experiments/p67_01_controles_decisivos.py
/home/carlos/venv/bin/python experiments/p67_02_cerrar.py
```

El regenerador de P66 es necesario porque `features/` esta en `.gitignore` y no
viaja entre worktrees. Es determinista y verifica sus siete scores contra la fase de
origen antes de continuar.

## Idempotencia

Los tres scripts se pueden ejecutar dos veces seguidas sin duplicar contenido en
`FINAL_REPORT.md`, `PROJECT_STATE.md`, `CHAMPION.md`, `RUNLOG.md`, `DECISIONS.md` ni
en el informe acumulado: cada bloque va sellado con `<!-- P67 2026-09-29 -->`.