# TOXIC - SGNS corregido sobre corpus Twitter/X (reapertura unica)

Fecha: 2026-10-08. Script: `experiments/p85_01_sgns_twitter_corregido.py`
(sha256 `d3cf2013ec10082ceea7d77e76986efac7a7aadbfd172b6607f25ea6bbf565e1`). Preregistro congelado: `docs/p85_preregistro.md`
(sha256 `17e4974a092fed28a0f4b4851db2ced4a89496f409b96ed3c97d3bb1500777af`). Sin entrenamiento de embeddings preentrenados, sin
test, sin holdout, sin leaderboard, sin submissions, sin push.

## 0. Resultado

```text
caso      C
decision  CLOSED (no reabrir la familia de embeddings)
```

## 1. Las tres piezas, sin confundirlas

- **P78 (historico, con sus limitaciones):** control anidado limpio del embedding Word2Vec
  100d, pero la matriz procedia del entrenador `p71_04`, que promediaba el termino negativo
  por K (auditoria P79): no implementa SGNS. Senal frente al azar (+0.0249 fragil), sin
  transferencia util.
- **P84-04 (SGNS corregido sobre corpus general):** implementacion corregida (suma) y
  anclada; corpus wikipedia 10M; resultados publicados con pooling sobre texto CRUDO
  (cobertura dev 0.6585). Aqui se reutiliza su embedding y se RE-PUNTUA su downstream con
  la normalizacion documentada, sin reentrenar.
- **Nuevo (SGNS corregido sobre corpus Twitter/X):** un entrenamiento, semilla 20261016,
  misma configuracion y mismo flujo; normalizacion del dev EMB-NORMAL-TW v1 identica en
  los dos brazos.

## 2. P0: hallazgos de la auditoria previa

- P78 uso `modelos/twitter_domain/w2v_twitter_100d.npz` (entrenada por `p71_04`,
  defectuosa); P84-04 usa la corregida (brazo S, ancla lograda).
- El pooling publicado de P84-04 uso texto crudo (cobertura 0.6585, 72 filas muertas):
  la discrepancia de normalizacion de P73/P74. Se corrige normalizando el dev con
  EMB-NORMAL-TW v1 en los DOS brazos; cobertura 0.8873 (Twitter)
  y 0.8036 (general).
- Corpus: Twitter `corpus_twitter_tokens.txt` sha256 `1ef872c189d0f405bc1a3f8281cafe08e6b099958e6b8a3816db1460df91f8b9` (autorizado,
  D-246, CC BY-SA 4.0, periodo cerrado); general `tokens_10M.txt` sha256 `00b839da6fbc377ee5375acc9a236e831e46ecf92cc544188b81aef1a826e871`;
  control general `emb_20261016_S.npz` sha256 `0216f9f6a952937925d8dcdd1cd885db8f776efed9ed47c339348c840bdff853` (reutilizado).
- El binario fastText no participa en este experimento (la implementacion SGNS es
  PyTorch); se registra que el binario en disco difiere del congelado de P84-04
  (worktree v67 ya retirado) y no se usa aqui.

## 3. Diseno congelado

dim 200; ventana 5; min_count 3; negativos 10; submuestreo 1e-4; 5 epocas; lote 256;
SGD lr 6.4 -> 1e-3; pooling medio; StandardScaler + LR C=2.0 balanced liblinear rs=42;
umbrales congelados de V03 (primaria) y anidados (secundaria); pliegues D-016;
bootstrap 10000/20260926; semilla unica 20261016 (limitacion declarada: no se
estima la varianza de reentrenamiento; no se rescata con mas semillas).

## 4. Resultados

Cobertura: Twitter 0.887342 (filas sin token 1);
general 0.803623 (filas sin token 2).

| brazo | lectura | ambito | f1 | delta vs V03 |
|---|---|---|---|---|
| v03 | congelada | global  | 0.7772114525 | 0.0000000000 |
| v03 | congelada | pliegue 1 | 0.7960101143 |  |
| v03 | congelada | pliegue 2 | 0.7940339498 |  |
| v03 | congelada | pliegue 3 | 0.7788600289 |  |
| v03 | congelada | pliegue 4 | 0.7385980131 |  |
| v03 | congelada | pliegue 5 | 0.7790564997 |  |
| v03 | umbral_0.5 | global  | 0.7717784763 |  |
| general | embedding_only_congelada | global  | 0.6290441473 | -0.1481673052 |
| general | embedding_only_congelada | pliegue 1 | 0.6130053421 | -0.1830047723 |
| general | embedding_only_congelada | pliegue 2 | 0.6435516310 | -0.1504823187 |
| general | embedding_only_congelada | pliegue 3 | 0.6209239811 | -0.1579360478 |
| general | embedding_only_congelada | pliegue 4 | 0.6266881391 | -0.1119098740 |
| general | embedding_only_congelada | pliegue 5 | 0.6394305201 | -0.1396259796 |
| general | embedding_only_umbral_0.5 | global  | 0.6219129041 | -0.1498655722 |
| general | fusion_congelada | global  | 0.7095163690 | -0.0676950835 |
| general | fusion_congelada | pliegue 1 | 0.7034811694 | -0.0925289449 |
| general | fusion_congelada | pliegue 2 | 0.7218470282 | -0.0721869216 |
| general | fusion_congelada | pliegue 3 | 0.7170606270 | -0.0617994018 |
| general | fusion_congelada | pliegue 4 | 0.6970503377 | -0.0415476755 |
| general | fusion_congelada | pliegue 5 | 0.7083282417 | -0.0707282581 |
| general | fusion_umbral_0.5 | global  | 0.6987483739 | -0.0730301024 |
| twitter | embedding_only_congelada | global  | 0.6291405055 | -0.1480709470 |
| twitter | embedding_only_congelada | pliegue 1 | 0.6460724332 | -0.1499376811 |
| twitter | embedding_only_congelada | pliegue 2 | 0.6500522690 | -0.1439816808 |
| twitter | embedding_only_congelada | pliegue 3 | 0.6133951717 | -0.1654648571 |
| twitter | embedding_only_congelada | pliegue 4 | 0.5901300271 | -0.1484679860 |
| twitter | embedding_only_congelada | pliegue 5 | 0.6429177933 | -0.1361387064 |
| twitter | embedding_only_umbral_0.5 | global  | 0.6195366280 | -0.1522418483 |
| twitter | fusion_congelada | global  | 0.7080390394 | -0.0691724132 |
| twitter | fusion_congelada | pliegue 1 | 0.7138531465 | -0.0821569678 |
| twitter | fusion_congelada | pliegue 2 | 0.7270005854 | -0.0670333644 |
| twitter | fusion_congelada | pliegue 3 | 0.7131998913 | -0.0656601376 |
| twitter | fusion_congelada | pliegue 4 | 0.6727836434 | -0.0658143698 |
| twitter | fusion_congelada | pliegue 5 | 0.7112314481 | -0.0678250517 |
| twitter | fusion_umbral_0.5 | global  | 0.6959202117 | -0.0758582646 |

Deltas principales:

```text
twitter  embedding-only congelada  0.629141  delta -0.148071
twitter  fusion congelada          0.708039  delta -0.069172
twitter  fusion 0.5                0.695920  delta -0.075858
general  embedding-only congelada  0.629044  delta -0.148167
general  fusion congelada          0.709516  delta -0.067695
pliegues positivos (fusion twitter vs V03): 0/5
bootstrap delta fusion twitter: IC95 [-0.087269, -0.050543]
ensemble_gain 0.5 (twitter): -0.075858 IC95 [-0.095279, -0.056789]
```

## 5. Complementariedad

Candidato (embedding-only congelada) frente a V03:

```text
global: salvage 278 | damage 896 | net -618
HS 126/334/-208 |
TR 50/228/-178 |
AG 102/334/-232
```

Fusion frente a V03:

```text
global: salvage 204 | damage 489 | net -285
HS 91/189/-98 |
TR 34/121/-87 |
AG 79/179/-100
```

## 6. Decision preregistrada

```text
C1 delta fusion >= +0.0015        False
C2 >= 4/5 pliegues positivos      False
C3 net de la fusion > 0           False
C4 bootstrap IC95 inferior > 0    False
C5 dominio twitter > general      False
```

**Caso C: CLOSED (no reabrir la familia de embeddings).** No se declara ADOPTED; no se genera submission; no se abren
mas experimentos tras esta comparacion.

## 7. Compliance y verificacion

```text
dev 1483 | holdout 0 | test 0 | leaderboard 0 | submissions 0 | push 0
entrenamientos nuevos: 1 (Twitter) | rejillas: 0
V03 FROZEN | 0.7772114525165721
tiempo total: 3393.9 s (entrenamiento Twitter 3248.8 s)
incidencias: 0
```

> V03 remains the best verified leaderboard result.

## 8. Errata (2026-10-08, posterior a la corrida)

El apendice del script a `experiments.csv` y el CSV de este informe usaban el
terminador por defecto del modulo `csv` (CRLF). Se corrigio a LF
(`lineterminator="\n"`) y la fila P85-01 se re-anexo manualmente con LF; el registro
queda en 456 filas con diff de una sola linea. Los resultados no cambian. El hash del
script ejecutado fue `d3cf2013ec10082ceea7d77e76986efac7a7aadbfd172b6607f25ea6bbf565e1`;
el hash corregido es `14035e34c9e4c12319dc4a4f5479e9a0dbee920f845616c6c9d2ded118db71a5` (Enmienda 2 del preregistro).
