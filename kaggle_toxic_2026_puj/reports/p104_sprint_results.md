# P104 - RONDA 3 (R4 + R5 + R7-1): resultados del sprint

Fecha: 2026-10-14. Pre-registro: `reports/p104_preregistro.md` (gate A3;
control V03 exacto en ambas particiones; piso de deteccion declarado;
presupuesto 1.5 h CPU). Alcance confirmado: SOLO R4, R5 y R7-1, en ese orden;
R1/R2/R3/R6/R7-2/R7-3 quedaron fuera.

## 1. Resultados

| Item | Que probo | Canonica | Particion 47 | Veredicto |
|---|---|---|---|---|
| P104-01 (R4) | stemming Snowball solo en word (S1) y union stem+raw (S2) | S1 -0.011893 (1/5; 202 celdas; IC95 [-0.021755,-0.002301]); S2 -0.007719 (1/5; 119 celdas; IC95 [-0.014381,-0.000603]) | S1 -0.009679; S2 -0.007150 | GATE_FAILED; efecto NEGATIVO detectable: el stemming perjudica |
| P104-02 (R5) | bolsa de clusters k-means (k 100/300/1000) sobre embeddings propios | K100 -0.015756 (0/5; 302 celdas); K300 -0.014801 (1/5; 279 celdas; IC95 [-0.026471,-0.003431]); K1000 -0.017862 (1/5; 293 celdas) | K100 -0.019338; K300 -0.012558; K1000 -0.022933 | GATE_FAILED; efecto NEGATIVO detectable en las tres |
| P104-03 (R7-1) | L1/ElasticNet por etiqueta (16 configs, seleccion anidada) | +0.000917 (3/5; 257 celdas; IC95 [-0.009404,+0.011041]; piso +0.010222) | -0.000418 | GATE_FAILED; sin senal detectable |

Control en los tres experimentos: reproduce V03 exacto en la canonica
(diff proba 0.00e+00; ccv 0.7772114525165721) y en la particion 47 (0 celdas
distintas; diff proba 0.00e+00).

## 2. P104-01 (R4) - detalle

- El stemming Snowball sobre la rama de palabra PERJUDICA en sus dos formas:
  quitando la palabra sin stemmar (S1, -0.0119) o conservando ambas (S2,
  -0.0077). Los IC95 excluyen cero por abajo: el dano es detectable.
- Coherente con la lematizacion spaCy (PRE-09 = 0.7425, la segunda peor de
  las diez variantes PRE de la fase v1): la normalizacion morfologica no
  aporta en este corpus porque char_wb(3,5) ya captura la morfologia util.

## 3. P104-02 (R5) - detalle

- Los clusters de palabra (k-means sobre w2v_twitter_lrfix, 77.657 palabras)
  como bolsa adicional PERJUDICAN en los tres tamanos, con IC95 negativos que
  excluyen cero. La informacion de cluster no aporta sobre word(1,2)+char_wb
  y diluye el presupuesto L2 del modelo.
- Sin fuga por construccion (pre-registrado): el embedding y el k-means usan
  solo vocabulario externo sin etiquetas; el mapa token->cluster es una
  funcion fija; el TF-IDF se ajusta dentro del pliegue.

## 4. P104-03 (R7-1) - detalle

- L1/ElasticNet por etiqueta queda en +0.000917, por debajo del piso de
  deteccion (+0.010222) y con IC95 que incluye cero: sin senal detectable.
- La seleccion anidada es inestable (penalizaciones y C distintos en cada
  pliegue, con l1 puro y elasticnet 0.2/0.5/0.8 alternando) y los nnz por
  etiqueta oscilan fuerte (HS 2000-7214; TR 107-5068; AG 142-936; medias
  4559/1697/553). Cero problemas de convergencia (saga, tol 1e-3).

## 5. Correccion registrada

- Los scripts P104-01/02/03 se regeneraron UNA vez para corregir la etiqueta
  de conclusion cuando el IC95 es enteramente negativo (antes decia "incluye
  cero"; ahora "efecto NEGATIVO detectable"). Las metricas de los artefactos
  regenerados son identicas a las de la primera corrida (verificado
  celda a celda en los campos corregidos); solo cambio la frase de lectura.

## 6. Cierre

- Sin candidato: los tres items fallan el gate A3 completo.
- V03 permanece congelado (corrected_cv 0.7772114525165721; LB 0.78306).
- Recomendacion de slots del leaderboard privado SIN CAMBIOS: V03 +
  v08_meta_v2 (mejor no-V03 enviada, LB 0.78163).
- Presupuesto: ~14 min de CPU de 1.5 h (163.7 + 287.5 + 285.3 s, regeneracion
  incluida).
- Sin test; sin holdouts; sin leaderboard; sin submissions; artefactos
  historicos intactos.

## 7. Artefactos

- Scripts: `experiments/p104_01_stemming_snowball.py`,
  `p104_02_clusters_palabras.py`, `p104_03_l1_elastic.py`,
  `p104_90_registrar.py`.
- Resultados: `reports/p104_01_stemming_snowball.{json,md}`,
  `p104_02_clusters_palabras.{json,md}`, `p104_03_l1_elastic.{json,md}`,
  `p104_preregistro.md`.
- Registro: filas P104-01/02/03 en `experiments.csv` (501 filas).
