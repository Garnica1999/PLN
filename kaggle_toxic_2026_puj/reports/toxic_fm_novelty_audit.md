# TOXIC - P86-00: auditoria de novedad de la Factorization Machine dispersa

Fecha: 2026-10-09. Script: `experiments/p86_00_auditoria_novedad_fm.py`.
Directiva auditada: FM dispersa con interacciones de bajo rango entre caracteristicas
lexicas, sobre la representacion word+char de V03, con el protocolo de V03.

## 1. Veredicto

**`DUPLICATE`.** El precedente metodologicamente equivalente existe y es
**V25-01 / V25-02 (2026-09-27)**: la Factorization Machine fue implementada desde cero
sobre la representacion exacta de V03, con el mismo protocolo y un control que
reproduce V03 bit a bit. La familia quedo cerrada entonces. **No se ejecuta nada.**

## 2. La propuesta frente al precedente, elemento por elemento

| elemento de la propuesta | V25-01 / V25-02 (ejecutado el 2026-09-27) |
|---|---|
| FM dispersa con interacciones de bajo rango | FM desde cero; interaccion de segundo orden de bajo rango; `O(nnz * k)`; 208.907.020 pares no materializados |
| representacion word (1,2) + char_wb (3,5) de V03 | `PARAMETROS_WORD` y `PARAMETROS_CHAR` identicos (min_df=2, sublinear_tf=True) |
| mismos folds agrupados y conjunto de desarrollo | StratifiedGroupKFold(42) sobre grupos near-duplicate (D-016); mismo dev; holdout intacto |
| mismos umbrales congelados y corrected_cv | `UMBRALES_V03 = [0.44, 0.51, 0.46]`; el control reproduce `0.7772114525165721` (diferencia 0.0; predicciones identicas) |
| control lineal equivalente sin interacciones | rama `interacciones=False` = la ruta exacta de V03 (liblinear, C=2.0, balanced, tol=1e-4) |
| una sola dimension latente, fijada con justificacion previa | k en {1, 4, 8, 16}: k=8 y k=16 identicos a k=4 (rango efectivo agotado en 4) |
| sin busqueda de regularizacion ni de semillas | barrido de `lambda_v` de 0.25 a 1e6: optimo exacto en 1, sin margen; 0 ajustes sin converger |

## 3. El resultado medido (no queda nada nuevo que medir)

Mejor candidato del barrido: k=4, `lambda_v`=1.

```text
corrected_cv          0.7826986101342172
delta vs V03          +0.0054871576176451
train F1              0.9801692041736572  (gap train-OOF +0.1975)
celdas cambiadas      278
rescates / danos      119 / 159
neto de celdas        -40
dLB esperado          -0.0078567438152005
ajustes sin converger 0
```

Por etiqueta (F1 y celdas): HS 0.769553 (48 rescates / 62 danos),
TR 0.817797 (25 / 33),
AG 0.760746 (46 / 64).
Los tres F1 suben y los tres netos son negativos: es el perfil que la propia directiva
advierte ("no interpretar una mayor capacidad como evidencia de transferencia"); el
entrenamiento sube (train F1 0.9802) y el neto empeora.

Barrido de k=4 (V25-02):

| k | lambda_v | corrected_cv | dOOF | train F1 | gap | celdas | rescates | danos | neto |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.25 | 0.7724400473 | -0.004771405 | 1.0000 | +0.2276 | 320 | 153 | 167 | -14 |
| 4 | 1.0 | 0.7826986101 | +0.005487158 | 0.9802 | +0.1975 | 278 | 119 | 159 | -40 |
| 4 | 4.0 | 0.7773336673 | +0.000122215 | 0.9718 | +0.1945 | 143 | 56 | 87 | -31 |
| 4 | 16.0 | 0.7772114525 | +0.000000000 | 0.9752 | +0.1980 | 0 | 0 | 0 | +0 |
| 4 | 64.0 | 0.7772114525 | +0.000000000 | 0.9752 | +0.1980 | 0 | 0 | 0 | +0 |
| 4 | 256.0 | 0.7772114525 | +0.000000000 | 0.9752 | +0.1980 | 0 | 0 | 0 | +0 |
| 4 | 1000000.0 | 0.7772114525 | +0.000000000 | 0.9752 | +0.1980 | 0 | 0 | 0 | +0 |

Con `lambda_v >= 16` el modelo colapsa a V03 exacto (dOOF +0.000000 a diez decimales).

## 4. El cierre historico (por que no se reabre)

- `reports/v25_04_cierre.md`: `NO CANDIDATE JUSTIFIES SUBMISSION - CLOSE V25`. El
  sobreajuste no se puede corregir, el rango se agota en 4 y el modo conjunto memoriza
  (train F1 1.0000, OOF -0.0284).
- `reports/v25_05_dictamen_final.md`: `CERRAR EL PROYECTO`. El gate del proyecto quedo
  corregido a `dOOF > +0.0376` **y** neto de celdas positivo en la misma configuracion;
  la FM es el caso que lo motiva (cambia 278 celdas, pasa el filtro de 200 celdas y aun
  asi tiene neto -40 con dOOF positivo).
- `DECISIONS.md` (D-190): el eje "geometria de representacion" (V25, V27, V26, V43)
  queda cerrado.
- Notebook: "FACTORIZATION MACHINE REJECTED (V25)" y el cierre de las cuatro familias
  no lineales (V23 RBF-SVC, V25 FM, V26 Tsetlin, V27 spectrum).
- `experiments/p59_05_nmf_gap_analysis.py` (auditoria posterior e independiente)
  describe V25 como "a low-rank factorisation of second-order interactions on V03's
  exact representation" y lo cita como vecino funcional ya cerrado.
- Registro: V25-01 (f1 0.782699, "no candidata, cerrar V25") y
  V25-02 (f1 0.782699, "confirmar el cierre de V25 sin submission").
- El RUNLOG no menciona V25 (la fase se documento en DECISIONS, informes y notebook); esta auditoria lo registra ahora.

## 5. Familias vecinas de interacciones y no linealidad, todas cerradas

| experimento | mecanismo | resultado | estado |
|---|---|---|---|
| V23 (RBF-SVC) | no linealidad por kernel | corrected_cv 0.790409; LB 0.77938 (-0.00368) | REJECTED |
| V25-01/V25-02 | FM (interacciones de bajo rango) | +0.005487; neto -40; dLB esperado -0.0079 | CLOSE V25 |
| V26-01..03 | Tsetlin Machine | V26-02 invalidada; V26-03 0.583397 | REJECTED |
| V27-01 | kernel de espectro de caracteres | 0.762122 (-0.015089); neto -45 | CHALLENGER, sin submission |
| V43-01/V43-02 | PLS | -0.0137423671 (PLS correcto) | cerrado |
| V4-D01-D04 | interacciones como tokens | 0.7758; TRACK D NO_SIGNAL | cerrado |
| E2-10-ORT-* | forma x token | -0.0238 a -0.0286 (0/5 pliegues) | cerrado |
| V6-05 | skip-grams con hueco 1-2 | -0.0001 (IC95 incluye 0) | cerrado |
| P65 (D-162) | pares lejanos 3-5 y 6-10 | -0.0066 / -0.0074 | cerrado |

## 6. Contra el gate de la propuesta

El delta del precedente (+0.005487) supera nominalmente el +0.003 propuesto, pero:

1. el propio proyecto corrigio su gate tras medirlo: neto de celdas negativo (-40) y
   dLB esperado negativo (-0.0079) en la misma configuracion;
2. el modelo de transferencia del proyecto (V23: mejor local 0.790409, peor LB) y el
   analisis de V25-05 ("V03 fue afortunado") apuntan en la misma direccion;
3. re-ejecutar la configuracion unica (k=4, `lambda_v`=1) reproduciria el punto ya
   medido, sin informacion nueva.

Conclusion: la hipotesis queda **cerrada sin ejecucion**. No se abre ninguna via y V03
permanece congelado.

## 7. Compliance

- Operaciones reales: solo lectura de `experiments.csv`, `DECISIONS.md`, `RUNLOG.md`,
  `PROJECT_STATE.md`, `notebooks/10_final_pipeline.ipynb`, `experiments/v25_0*.py`,
  `experiments/p59_05_nmf_gap_analysis.py` y `reports/v25_0*`. Cero entrenamientos;
  train/test/holdout no se abren para desarrollo; 0 submissions; 0 push; V03 y sus
  artefactos no se tocan.
- `PROJECT_STATE.md` no menciona V25 ni la FM (verificado).
- Escaneo byte a byte de lo generado: 0 bytes > 127.
- Hashes verificados: OOF `559265816ce4f631...`; submission `7d7478f271e6edfd...`.

## 8. Hashes de la evidencia

```text
v25_01_factorization_machine.py   b5ae368fd1a7ec6743e840ecc7a1122561e30d71ccf7f2978ab645afad48fa4f
v25_02_fm_paralela_lambda.py      78aa2d3cde9cd18909e21a3b7014f5d2ef433c79e33835cf452dd08d9bf494eb
v25_02_fm_paralela_lambda.json    06af94d1bd7ec10dec54bb4601aec108e90182f38df77b47dc5641ab4463eabb
```

> V03 remains the best verified leaderboard result.
