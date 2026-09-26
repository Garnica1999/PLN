# Plan de la fase v8: Transfer Gap + Supervised Sparse Reweighting

Worktree `/home/carlos/Git/PLN-v8`, rama `proyecto-kaggle-toxic-2026-v8`, desde el cierre de v7.

## Objetivo

Superar 0.80 en leaderboard y en metrica local, con el marco clasico: sin transformers, sin
modelos preentrenados de terceros, sin datasets externos, sin pseudoetiquetas y sin
etiquetas de prueba.

**Anotacion honesta del objetivo** (D-059): superar 0.80 esta a +0.01694 de la referencia, o
sea **1.1 veces el ruido del leaderboard y 21 aciertos netos de ventaja limpia**, sobre once
sondas de las que diez son indistinguibles entre si. No es imposible, pero no se alcanza
empujando la capa de decision.

## Reglas que no se negocian

- V03 congelada en 0.777211 con umbrales 0.44, 0.51 y 0.46. No se modifican ni sus artefactos.
  Sigue siendo referencia **por no tener evidencia en contra, no por tener evidencia a favor**.
- El holdout esta agotado: no se usa.
- Una linea negativa se cierra y no se rescata con mas hiperparametros, semillas o rasgos.
- El control 0.777211 se comprueba antes de cada bloque importante, y el criterio fuerte es
  la identidad contra el OOF canonico, no la tolerancia de la cifra.
- Todas las estadisticas supervisadas se construyen dentro del tren del pliegue.
- Gate de mejora: `delta >= +0.003`, `>=4/5` pliegues, IC95 que no incluya 0.
- **Filtro de detectabilidad (nuevo, D-059)**: una candidata que cambie C celdas de la
  referencia solo es resoluble por el externo si su tasa de acierto entre esas celdas supera
  `0,5 + 19,2 / C`. Por debajo del minimo de la tabla, no se envia. Se estima en el desarrollo,
  antes de gastar la sonda.
- Freno de transferencia: por encima de +0.005 local, no aumentar dimension ni brecha.
- 0 envios automaticos, sin push, sin modificar remotos.

## Fases

| Fase | Track | Estado |
|---|---|---|
| 1 | P0 auditoria de brecha y del material de clase | **cerrada** (D-057) |
| 2 | P1 promedio de los cinco modelos al predecir | **cerrada, negativa** (D-059) |
| 3 | A2 resolucion del leaderboard y filtro de detectabilidad | **cerrada** (V8-02) |
| 4 | P3 estabilidad de features | **cerrada, negativa** (D-060) |
| 5 | P2 log-count ratio, con precedente adverso ya medido | **cerrada, negativa** (D-061) |
| 6 | P4 regularizacion conjunta | **cerrada, negativa** (D-062) |
| 7 | P5, P6 y P7 | sin razon para esperar que pasen; la decision es del usuario |
