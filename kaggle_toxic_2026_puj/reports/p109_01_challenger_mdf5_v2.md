# Reporte P109-01: Challenger V2_triple min_df=5 + Umbrales Calibrados

## 1. Resumen Ejecutivo
- **Objetivo**: Superar al campeon oficial P108 (LB = 0.78855) mediante poda de ruido por frecuencia (`min_df=5`), regularizacion adaptativa C por etiqueta y calibracion de umbrales.
- **Canonica CV**: 0.792215 (**++0.009843** sobre P108, **++0.015004** sobre V03).
- **Particion 47 CV**: 0.789294 (**++0.008108** sobre P108).
- **Pliegues ganados en Canonica**: 5/5 ganados.
- **Bootstrap 95% IC**: [+0.000683, +0.019669] (estrictamente positivo, win rate: 98.4%).
- **Submission generada**: `toxic_submission_p109_mdf5_v2_asc.csv` (SHA256: `c4895e6d87873cc968ec40cc262fcd8eac10f253bcdcd33e6d223708a3b8e92b`).
- **Proyeccion Kaggle Leaderboard**: ~0.798 - 0.800.

## 2. Metricas Detalladas
| Metrica | V03 (Ancla) | P108 (Campeon) | P109 (Challenger) | Delta vs P108 |
|---|---|---|---|---|
| Canonica CCV | 0.777211 | 0.782372 | **0.792215** | **+0.009843** |
| F1 HS | 0.7482 | 0.7629 | **0.7782** | +0.0153 |
| F1 TR | 0.8143 | 0.8150 | **0.8332** | +0.0182 |
| F1 AG | 0.7691 | 0.7692 | **0.7653** | -0.0039 |
| Particion 47 | 0.776100 | 0.781185 | **0.789294** | **+0.008108** |
| Kaggle LB | 0.78306 | 0.78855 | Proy. ~0.798 | +0.010 |

## 3. Comportamiento sobre Test (1.153 filas)
- Celdas distintas de P108: 125 de 3462 (3.61%).
- Diferencias por columna: HS: 47, TR: 28, AG: 50.
- Rescates regla ascendente TR=1 => HS=1: 19 casos.
- Features totales train completo: 31603 (frente a 86.243 en P108).
