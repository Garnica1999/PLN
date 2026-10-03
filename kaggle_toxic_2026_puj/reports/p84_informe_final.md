# P84: informe final de la fase de exploracion (2026-10-02)

## 1. Resumen ejecutivo

Objetivo: buscar una mejora sustancial sobre V03 (LB 0.78306) con candidatos permitidos,
incluyendo la meta de 0.80 en el leaderboard. Protocolo: desarrollo de 1483 filas, holdout
agotado (6 usos frente a 5 declarados, no se usa), test intocable, sin transformers, sin
embeddings preentrenados, sin etiquetas externas y sin pseudoetiquetado.

Se ejecutaron cinco experimentos (P84-01 a P84-05) y la puerta de combinacion (P84-06).
Resultado global: **ningun candidato nuevo supera a V03**. Los cinco cierran con evidencia
verificable y la puerta de combinacion queda cerrada sin candidatos elegibles.

```text
V03 remains the best verified leaderboard result.
corrected_cv 0.7772114525165721 | LB 0.78306 | umbrales 0.44/0.51/0.46
```

No se generaron submissions nuevas. Todas las conclusiones son locales (ninguna sonda
nueva); el leaderboard no se uso para decidir.

## 2. Protocolo y controles

- Identidad V03 reproducida en cada harness con tolerancia 1e-12 (max|delta proba| = 0.0).
- Contadores de acceso por corrida: dev 1483 / holdout 0 / test 0; `toxic_test.csv` nunca
  se abre.
- Umbrales congelados de V03 y lectura a 0.5 como control de arquitectura; umbrales
  anidados solo en particiones internas.
- Comparaciones pareadas por pliegue, delta global y medio, IC95 bootstrap 10.000 con
  semilla 20260926 explicita, rescate/dano, tasas predichas y F1 por etiqueta.
- `ensemble_gain` con la convencion P6 (media de probabilidades a 0.5 menos 0.771778476256122,
  recalculado del npz).
- Gates: +0.005, 4/5 pliegues e IC95 que excluya cero; para candidatos que aumentan
  capacidad, umbral efectivo +0.010 (D-057) y brecha de ajuste no mayor que el control
  +0.05. Complementariedad: `ensemble_gain >= 0.0015` con IC > 0.
- Pre-registro por experimento en `docs/p84_preregistro.md` (Enmiendas 1 a 5 y 3bis) con
  sha256 de cada script.

## 3. Resultados por experimento

| ID | Hipotesis | Resultado clave | Gate | Decision |
|---|---|---|---|---|
| P84-01 | DL-01 corregido (pool 1483, epoca anidada) supera a V03 | F1@0.5 0.758301, delta -0.018910, IC95 [-0.033910, -0.004150], 1/5; anidada 0.756968; congelada 0.761659 | NO | Cerrar DL-01; el 0.7949 historico no es comparable |
| P84-02 | fastText supervisado aporta/complementa | principal: anidada 0.583398, delta -0.193813 (0/5); control sin subwords: anidada 0.737771, delta -0.039441 (0/5) | NO | Cerrar fastText supervisado sin rejilla |
| P84-03 | ROS aporta sobre class_weight | B1-A refutacion parcial (max|dp| 7.19e-2; |dF1| 2.66e-4); B1 +0.000266 (3/5); B2 -0.044992 (0/5); B3 +0.000210 (3/5) | NO PROMOVIBLE | ROS cerrado por equivalencia (CLOSED BY EQUIVALENCE) |
| P84-04 | SGNS corregido (suma vs 1/K) cambia el downstream | ancla lograda; S 0.626091 / M 0.622888 / R 0.651825 / V03 0.777211; S vs M +0.003204 (3/5, IC incluye 0); S vs V03 -0.151120 (0/5) | NO | Cerrar SGNS corregido con resultado verificado |
| P84-05 | EM + shrinkage mejora en simulacion OOF | delta medio -0.008394 (IC95 [-0.015763, -0.001231]); 2/5 pliegues LOFO; componente EM +0.020019; umbral -0.028414 | NO | Cerrar la familia EM/threshold |
| P84-06 | Combinacion de candidatos | Sin candidatos elegibles (ningun gate ni complementariedad) | Puerta cerrada | No se ejecuta |

### P84-01 (DL-01 corregido)

Repeticion del MLP TF-IDF union -> SVD(300) -> 256-128 con pool de 1483 filas y epoca
elegida en particion interna (3 splits, E* = mediana) y reajuste fijo. 248.33 s en GPU,
5 pliegues x 3 semillas; reproducibilidad entre corridas identica (max|delta proba| = 0.0).
Brecha del MLP 0.209140 frente a 0.204721 del control (+0.004418 <= 0.05); capacidad
110.339 parametros. `ensemble_gain` -0.001679 (IC incluye 0). Cierra la lectura del 0.7949
historico como no comparable; la medicion corregida es 0.758301 a umbral 0.5.

### P84-02 (fastText supervisado)

Tres clasificadores binarios fold-local con `train_supervised` (subcomando `supervised`),
`dim 100`, `lr 0.1`, `wordNgrams 2`, epocas {5,10,20} elegidas en particiones internas.
206.4 s CPU con `-thread 1`; determinismo 0.0. El brazo principal (subwords 3-6) colapso
con los flags preregistrados (loss ~ ln 2, probabilidades ~0.5, E* 5); el control sin
subwords (E* 20) quedo en 0.737771 anidada, delta -0.039441, 0/5. La pregunta de subwords
queda sin responder y no se cita un efecto sublexico.

### P84-03 (ROS)

Micro-verificacion de equivalencia: con liblinear L2 duplicar filas es reasignar pesos.
B1 (r=1, masa n, C=2) no reproduce a A en probabilidad (redondeo entero, desviacion de
peso hasta 0.556) pero si en F1 (|dF1| 2.66e-4). Recuentos corregidos tras la refutacion
(Enmienda 3bis): B1 61 celdas (32/29), B2 446 (207/239), B3 63 (34/29) sobre 4449.
B2 (r=0.5) -0.044992; B3 (sin normalizar) +0.000210 con C_efectivo 2.25-2.80. No promovible.

### P84-04 (SGNS corregido y anclado)

Ancla lograda antes de entrenar: identidad algebraica (1.28e-16 / 1.16e-16), gradiente por
diferencias finitas (4.42e-08) y ancla ejecutable contra fastText oficial (pureza 1.0 = 1.0;
el binario oficial dio un NaN y el reintento funciono). Corpus wikipedia_es 10M
EXTERNAL-ONLY; 3 semillas; flujo pareado S/M verificado por hash. F1 congelada promedio:
S 0.626091, M 0.622888, R 0.651825, V03 0.777211. S vs M +0.003204 (3/5, IC incluye 0);
S vs V03 -0.151120 (0/5); `ensemble_gain` -0.049004. Matiz de la refutacion final: S vs R
-0.025734 es efecto del promedio de probabilidades entre semillas; por semilla los deltas
son -0.0032/-0.0035/+0.0062 (media -0.00015, empate). El corpus externo no muestra aporte
al downstream. 15511 s en GPU.

### P84-05 (EM + shrinkage simulado)

Simulacion analitica sobre el OOF de V03, 5 pliegues LOFO, rho {0.85..1.05}. Delta medio
-0.008394 (IC95 con pesos [-0.015763, -0.001231]); 2/5 pliegues LOFO positivos. La
componente EM es positiva (+0.020019) pero el umbral reubicado aporta -0.028414; sanidad
rho=1 -0.005043 (negativa); guardarrail de tasa falla en AG (-0.014835). La componente EM
positiva no es evidencia end-to-end.

## 4. Correcciones de la refutacion adversarial

- P84-03: bug de conteo (`pred` en vez de `pred_ref`); corregido, reejecutado y registrado
  en la Enmienda 3bis. La decision no cambia.
- P84-04: la afirmacion "la matriz aleatoria supera a los embeddings" se matiza: por
  semilla es un empate; el signo negativo viene del promedio de probabilidades.
- P84-02: la degeneracion del brazo principal impide responder la pregunta de subwords;
  no se cita un efecto sublexico.
- P84-01: "artefacto de protocolo" se matiza; la descomposicion causal del desplome no se
  aislo (el protocolo limpio tambien cambia E*).
- P84-05: se anaden la sanidad rho=1 negativa y la lectura no end-to-end de la componente EM.

## 5. Limitaciones

- N = 1483 filas de desarrollo; la banda de particion del instrumento es ~0.017.
- El leaderboard publico usa ~70 % del test (~808 filas): ruido ~0.0146-0.0182
  (~23 aciertos netos); el privado ~346 filas: ~0.0223-0.0278 (~35 netos).
- Holdout agotado (6/5); no se crea otro. No se usaron sondas nuevas.
- V03 es a su vez una seleccion sobre 23 sondas registradas; su ventaja puede incluir
  winner's curse.
- fastText supervisado se probo sin rejilla (por pre-registro); su brazo principal colapso.
- P84-04 midio R solo como ensemble de probabilidades; dim 200 no es comparable con la
  linea historica 100d.

## 6. Coste de la fase

| Experimento | Tiempo | Device |
|---|---:|---|
| P84-01 | 248.33 s | GPU RX 9070 XT |
| P84-02 | 206.36 s | CPU (`-thread 1`) |
| P84-03 | 20.24 s | CPU |
| P84-04 | 15511.17 s | GPU RX 9070 XT |
| P84-05 | 1.0 s | CPU |
| Total | 15987.10 s (~4,44 h) | - |

La auditoria de rendimiento confirmo que las cifras provienen de los artefactos (sin
fabricacion); P84-04 fue mas lento que su presupuesto porque la referencia de p71_08 era
de otro corpus (twitter 10M), no por ineficiencia de la implementacion.

## 7. Matriz de cierres de P84

- Cierres por resultado: DL-01 corregido, fastText supervisado, SGNS corregido, EM/threshold.
- Cierre por equivalencia: ROS con reweighting/class_weight (CLOSED BY EQUIVALENCE).
- Puerta sin candidatos: combinacion EXP-06.
- Familias ya cerradas que no se repitieron: perdidas (focal/F1 suave/ASL/PolyLoss/Tukey),
  generativo de n-gramas, term weighting (log-odds/chi2/MI/BM25/binario), cadenas/powerset,
  calibracion/umbrales, SGD/hashing/FTRL/VW, rejillas de corpus/ventana/pooling de
  embeddings no supervisados, RNN/CNN, pesos por clase y reweighting.
- Vias prohibidas por reglas/decisiones: transformers y LLM, embeddings preentrenados,
  etiquetas externas, pseudoetiquetado y generacion de texto.

## 8. Artefactos

- Preregistro y especificaciones: `docs/p84_preregistro.md` (Enmiendas 1-5 y 3bis),
  `docs/p84_spec.md`, `tasks/todo_p84.md`.
- Informes por experimento: `reports/p84_01_dl01.{json,md}` (+ previo_fix),
  `reports/p84_02_fasttext.{json,md}`, `reports/p84_03_ros.{json,md}`,
  `reports/p84_04_sgns.{json,md}`, `reports/p84_04_ancla.json`,
  `reports/p84_05_em.{json,md}`.
- Datos derivados: `features/p84_01_oof_mlp.npz`, `features/p84_02_oof_fasttext.npz`,
  `features/p84_03_oof_ros.npz`, `features/p84_04_oof_emb.npz`, `modelos/p84/`.
- Auditoria de la fase: `reports/p84_qa_parcial.md`, `reports/p84_qa_final.md`,
  `reports/p84_refutacion_final.md`, `reports/p84_rendimiento.md`,
  `reports/p84_auditoria_premisas.md`.
- Registro y memoria: filas P84-01..P84-05 en `experiments.csv`; `RUNLOG.md`;
  `DECISIONS.md` D-281..D-287; plan en
  `/home/carlos/.opencode/plan/kaggle_toxic_exploracion_modelos_f1_080.md`.

## 9. Referencias

1. Basile et al. (2019). SemEval-2019 Task 5. ACL Anthology S19-2007.
2. Lin et al. (2023). Linear Classifier: An Often-Forgotten Baseline. ACL 2023.
3. Joulin et al. (2017). Bag of Tricks for Efficient Text Classification. EACL 2017.
4. fastText. Documentacion oficial de opciones y clasificacion supervisada.
5. Zhang, Hangya y Fraser (2024). Class Imbalance in Abusive Language Detection. WOAH 2024.
6. Saerens, Latinne y Decaestecker (2002). Adjusting the Outputs of a Classifier. DOI
   10.1162/089976602753284446.
7. Alexandari, Kundaje y Shrikumar (2020). Bias-Corrected Calibration for Label Shift.
   ICML 2020.
8. Mikolov et al. (2013). Distributed Representations of Words and Phrases. arXiv:1310.4546.
9. Levy y Goldberg (2014). Neural Word Embedding as Implicit Matrix Factorization. NIPS.
10. Costache et al. (2026). Text Classification Under Class Distribution Shift. EACL 2026.
11. Kaggle. TweetToxic 2026 PUJ, leaderboard publico/privado (70/30).
