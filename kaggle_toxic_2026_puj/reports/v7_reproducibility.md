# v7: reproducibilidad

Generado por `experiments/v7_04_entregables.py` (V7-04 (cierre, analisis)).

Rama `proyecto-kaggle-toxic-2026-v7`, commit `18c6e7d`.

## 1. Como ejecutar cada experimento

```bash
cd /home/carlos/Git/PLN-v7/kaggle_toxic_2026_puj
/home/carlos/venv/bin/python experiments/v7_00_gap_audit.py
```

| experimento | script | artefactos | entrena | determinismo |
|---|---|---|---|---|
| V7-00 | `experiments/v7_00_gap_audit.py` | v7_gap_audit.md, .csv, v7_gap_verificaciones.json | no | 15 busquedas en el codigo y lectura de cifras desde los artefactos |
| V7-01 | `experiments/v7_01_perdida_f1.py` | v7_perdida_f1.md, .csv, .json, figura 66 | si, 3 perdidas x 5 pliegues x 3 etiquetas | 600 iteraciones, tasa 0.05, decaimiento 2.0; el control BCE debe dar 0.777165 |
| V7-02 | `experiments/v7_02_prior_grupo.py` | v7_prior_grupo.md, .csv, .json, figura 67 | si, 3 variantes x 5 pliegues x 3 etiquetas | el control P8-0 debe dar 0.777211 exacto |
| V7-03 | `experiments/v7_03_umbral_anidado.py` | v7_umbral_anidado.md, .csv, .json, v7_calibration_analysis.csv, figura 68 | si, 3 brazos x 5 pliegues x 3 etiquetas | el brazo P0-0 debe dar 0.777211 exacto; semilla de particion 20260926 |

## 2. Controles de calidad de la fase

Cada track lleva un control que debe reproducir una cifra conocida. Los cuatro
dieron el numero exacto o dentro de la tolerancia declarada:

| track | control | valor obtenido | valor de referencia |
|---|---|---|---|
| P9 | BCE con el mismo codigo | 0.777165 | 0.777211, tolerancia 0.006 |
| P8 | P8-0 sin columnas de grupo | 0.777211 | 0.777211, exacto |
| P0.1 | P0-0 sobre el entrenamiento exterior completo | 0.777211 | 0.777211, exacto |

**Que el control de P9 quedase en 0.777165 y no en 0.777211 importa**: un modelo lineal
ajustado de forma iterativa no es identico a la solucion dual exacta de liblinear, y
el hueco de 0.000046 es el precio de esa diferencia. Por eso todos los deltas de P9
se miden contra ese control y no contra V03.

## 3. Semillas fijadas

- bootstrap pareado: 10.000 repeticiones, semilla **20260926**
- P0.1, particion interna agrupada: semilla **20260926**
- P9, inicializacion del peso lineal: semilla **20260926**
- `LogisticRegression`: `random_state = 42`

## 4. Artefactos de entrada compartidos

| artefacto | quien lo usa |
|---|---|
| `features/oof_v03_final.npz` | V7-01 y V7-04 |
| `src/splits.py` | todos |
| `src/evaluation.py` | todos |
| `data/` (enlace simbolico al worktree principal) | lectura de `toxic_train.csv` |

## 5. Datos y su hash

El conjunto de entrenamiento no se modifica en la fase, de modo que su hash es el
de la entrega anterior. El enlace simbolico de `data/` garantiza que ambos worktrees
leen exactamente el mismo fichero.
