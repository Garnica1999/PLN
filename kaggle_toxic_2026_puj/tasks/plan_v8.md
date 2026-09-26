# Plan de la fase v8: Transfer Gap + Supervised Sparse Reweighting

Worktree `/home/carlos/Git/PLN-v8`, rama `proyecto-kaggle-toxic-2026-v8`, desde el cierre de v7.

## Objetivo

Superar 0.80 en leaderboard y en metrica local, con el marco clasico: sin transformers, sin
modelos preentrenados de terceros, sin datasets externos, sin pseudoetiquetas y sin
etiquetas de prueba.

## Reglas que no se negocian

- V03 congelada en 0.777211 con umbrales 0.44, 0.51 y 0.46. No se modifican ni sus artefactos.
- El holdout esta agotado: no se usa.
- Una linea negativa se cierra y no se rescata con mas hiperparametros, semillas o rasgos.
- El control 0.777211 se comprueba antes de cada bloque importante.
- Todas las estadisticas supervisadas se construyen dentro del tren del pliegue.
- Gate: `delta >= +0.003`, `>=4/5` pliegues, IC95 que no incluya 0.
- Freno de transferencia: por encima de +0.005 local, no aumentar dimension ni brecha.
- 0 envios automaticos, sin push, sin modification de remotos.

## Fases

| Fase | Track | Estado |
|---|---|---|
| 1 | P0 auditoria de brecha y del material de clase | **cerrada** (D-057) |
| 2 | P1 promedio de los cinco modelos al predecir | **ficha lista, en espera del numero externo** (D-058) |
| 3 | Bifurcacion segun el resultado de P1 | pendiente |
| 4 | P3 estabilidad de features, luego P2 log-count ratio | pendiente |
| 5 | P4 a P7 segun evidencia | pendiente |

## Bifurcacion

- **Si P1 mejora de forma material**: congelar la arquitectura y ejecutar como maximo P9
  (desacuerdo entre pliegues) y P10 (segunda semilla). Prohibido seguir buscando.
- **Si P1 no mejora**: cerrar P1 sin variantes y entrar por P3, que es la forma barata de
  reducir la dimension sin perder senal. Despues P2, con el precedente ya medido en E2-15.
