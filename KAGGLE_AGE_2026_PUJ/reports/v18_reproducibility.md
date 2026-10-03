# V18 — Reproducibilidad

- **Rama**: `kaggle-age-2026-puj-v18`
- **Commit base**: `46b32f9`
- **Commit de la fase**: `8163215a2c7bd7ac50a1eb0d2c57343cc926ddfb`
- **Semilla**: `cfg.SEED`
- **Interprete**: `/home/carlos/venv/bin/python` (Python 3.14.7)
- **Hardware**: AMD Ryzen 9 5950X (32 hilos), 31,2 GB. **CPU pura**: el modelo de P3, P4 y
  P5 es una cabeza lineal de 279 a 6, y la GPU no aporta nada a un producto de ese tamano.
  Se fijo en CPU de forma explicita para no depender de la estabilidad de ROCm en un
  entrenamiento largo.

## Script de cada fase

| Fase | Script |
|---|---|
| P0 auditoria | documental, sin script de metricas |
| P0' evidencia y confound | `scripts/v18_p0_reproduccion.py` |
| P1 peso de clases | `scripts/v18_p1_peso_clases.py` |
| P3 margen de adyacencia | `scripts/v18_p3_margen.py` |
| P4 multi-objetivo | `scripts/v18_p4_multiobjetivo.py` |
| P5 destilacion | `scripts/v18_p5_destilacion.py` |
| Cierre | `scripts/v18_cierre.py` |

## Entradas, todas preexistentes

| Artefacto | Que es |
|---|---|
| `features/oof_ensemble_v12.npy` | OOF de produccion de V12, 169.336 x 6 |
| `features/oof_ADV07_nbsvm_word.npy` | 6 columnas de NB-SVM que completan las 279 |
| `features/oof_E028a_pares.npy` | margenes de pares, usados solo para reproduccion |
| `features/dup_clusters.parquet` | agrupacion de casi duplicados |
| `features/age_supervised_features.npy` | 72 columnas supervisadas, cross-fitted |

## Puntos donde la reproduccion depende de una convencion

1. **El estimador de la referencia**: pooled, `metricas_globales`. El canonico es
   0,6005017473.
2. **Las fronteras de V18 son accuracy restringida al par**, no macro-F1 de seis clases.
3. **La cabeza lineal de PyTorch no iguala a la `LogReg` de sklearn**: el control de P3, P4 y
   P5 queda 0,0022-0,0023 por debajo de V12. Es un handicap de montaje del mismo orden que
   los efectos medidos, y por eso P4 no puede resolver efectos por debajo de 0,002.
4. **El cross-fitting de P5** usa cinco inner folds agrupados por duplicados sobre el train
   exterior. Si un grupo de casi duplicados cruzara, la destilacion estaria contaminada.
