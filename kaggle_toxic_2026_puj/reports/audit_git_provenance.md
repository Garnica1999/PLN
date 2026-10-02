# P0 - Provenance git y estado del repositorio (borrador denso generado por script)

Generado por `audit/p00_git_provenance.py`. Todos los datos son read-only sobre
`/home/carlos/Git/PLN`. Este documento se refina manualmente en la fase P0.

## Commits que tocan el proyecto (todas las refs)
- Total de commits que tocan `kaggle_toxic_2026_puj/`: 56
- Primer commit: bb87158 2026-09-25 KAGGLE_TOXIC_2026_PUJ: solucion completa (fase 1 + investigacion v2)
- Ultimo commit: b0edf7a 2026-09-26 KAGGLE_TOXIC_2026_PUJ: las seis tecnicas del material de clase -| la mejora local era el promediado

## Distribucion por fase
- age-v13(stray): 3 commits
- v1-inicial: 1 commits
- v2: 6 commits
- v4: 6 commits
- v5: 9 commits
- v5.1: 7 commits
- v6: 10 commits
- v7: 5 commits
- v8: 9 commits

## Ramas

| Rama | Tip | Fecha | Commits que tocan el proyecto | En origin | Worktree |
|---|---|---|---|---|---|
| proyecto-kaggle-toxic-2026 | 42114db | 2026-09-21 | 0 | no | - |
| proyecto-kaggle-toxic-2026-v2 | abd4ec4 | 2026-09-25 | 7 | si | - |
| proyecto-kaggle-toxic-2026-v4 | 5d06ce7 | 2026-09-25 | 13 | si | - |
| proyecto-kaggle-toxic-2026-v5 | a719153 | 2026-09-25 | 29 | no | - |
| proyecto-kaggle-toxic-2026-v6 | d370977 | 2026-09-25 | 39 | no | - |
| proyecto-kaggle-toxic-2026-v7 | 2f8fac8 | 2026-09-25 | 44 | no | /home/carlos/Git/PLN-v7 |
| proyecto-kaggle-toxic-2026-v8 | b0edf7a | 2026-09-26 | 53 | no | /home/carlos/Git/PLN-v8 |
| proyecto-kaggle-toxic-2026-audit | b0edf7a | 2026-09-26 | 53 | no | /home/carlos/Git/PLN-toxic-audit |
| main | bae7612 | 2026-09-25 | 7 | si | - |
| kaggle-age-2026-puj-v13 | c55def7 | 2026-09-25 | 10 | no | /home/carlos/Git/PLN |

## Superficies auditadas

| Superficie | Ruta | Rama | Filas experiments.csv | Notebook | Sin commitear |
|---|---|---|---|---|---|
| A_actual | /home/carlos/Git/PLN/kaggle_toxic_2026_puj | kaggle-age-2026-puj-v13 | 128 | 2454508B sha256:3a4436b4658b2168 | ?? kaggle_toxic_2026_puj/tools/ |
| B_v7 | /home/carlos/Git/PLN-v7/kaggle_toxic_2026_puj | proyecto-kaggle-toxic-2026-v7 | 188 | 2922309B sha256:640270be711c37e0 | limpio |
| C_v8 | /home/carlos/Git/PLN-v8/kaggle_toxic_2026_puj | proyecto-kaggle-toxic-2026-v8 | 202 | 2922309B sha256:640270be711c37e0 | limpio |
| D_audit | /home/carlos/Git/PLN-toxic-audit/kaggle_toxic_2026_puj | proyecto-kaggle-toxic-2026-audit | 202 | 2922309B sha256:640270be711c37e0 | ?? kaggle_toxic_2026_puj/audit/
?? kaggle_toxic_2026_puj/reports/audit_git_provenance.csv
?? kaggle_toxic_2026_puj/reports/audit_git_provenance.md
?? kaggle_toxic_2026_puj/reports/audit_hashes_reference.csv |

## Archivos que difieren entre superficies (hashes de referencia)

- DECISIONS.md: A_actual=d73028cf1a63; B_v7=43aa0e03eeb9; C_v8=3989624dadf4
- FINAL_REPORT.md: A_actual=d4b9c2c435c8; B_v7=68ae4857706e; C_v8=68ae4857706e
- PROJECT_STATE.md: A_actual=36249c597d97; B_v7=d5b015958d5b; C_v8=2b6ef585ce5d
- RUNLOG.md: A_actual=2a1de877a1ae; B_v7=abab3e302955; C_v8=716bd6ef034a
- experiments.csv: A_actual=3fb0a4a2b79f; B_v7=e3858fe19b00; C_v8=1a236cc2b437
- notebooks/10_final_pipeline.ipynb: A_actual=3a4436b4658b; B_v7=640270be711c; C_v8=640270be711c
- src/evaluation.py: A_actual=ca8fc964185b; B_v7=3b3bc7b224de; C_v8=3b3bc7b224de
- submissions/leaderboard_log.csv: A_actual=5395884e515c; B_v7=5395884e515c; C_v8=42fbecad31c2; D_audit=42fbecad31c2
- submissions/submission_manifest.csv: A_actual=0d98d078ba34; B_v7=2e400980e789; C_v8=32d69e8e61b3; D_audit=32d69e8e61b3

## Stash
- Existe 1 stash: `stash@{0}: On kaggle-age-2026-puj-lexicons: WIP AGE lexicones fases 36-41`.
  Pertenece al proyecto AGE, no al toxic. NO se toca.

## Incidencias de commits cruzados (sesion AGE v13)
- c8e220f, f68637c, 32ab2f2 tocaron `kaggle_toxic_2026_puj/tools/` desde la sesion AGE:
  f68637c retira del indice `tools/fastText` (gitlink anadido por `git add -A`);
  32ab2f2 retira del indice `tools/verificar_caracteres.py` (213 lineas, queda en disco sin versionar).

## Entorno
- Python 3.14.7 (/home/carlos/venv/bin/python), Linux 7.2.6-1-cachyos, RAM 31.2 GB
- GPU: {'version': '2.14.0+rocm7.2', 'cuda_available': True, 'device': 'AMD Radeon RX 9070 XT'}
- Librerias: {"numpy": "1.26.4", "pandas": "2.3.3", "scipy": "1.17.1", "scikit-learn": "1.8.0", "lightgbm": "4.6.0", "xgboost": "3.2.0", "catboost": "1.2.10", "nltk": "3.9.4", "optuna": "4.8.0", "joblib": "1.5.3", "pyarrow": "24.0.0", "matplotlib": "3.10.9"}

## Hashes de referencia
- Ver `audit_hashes_reference.csv` (sha256 por superficie).
