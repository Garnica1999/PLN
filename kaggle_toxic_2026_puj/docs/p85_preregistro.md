# P85-01: preregistro de la reapertura experimental unica - SGNS corregido sobre corpus Twitter/X

Fecha: 2026-10-08. Estado: **CONGELADO antes de la primera ejecucion completa**.
Autorizacion: reapertura experimental unica y estrictamente acotada (instruccion de Carlos, 2026-10-08).

## 1. Hipotesis

```text
SGNS correctamente implementado + corpus legal de Twitter/X en espanol
```

Al corregir el objetivo SGNS (suma del termino negativo, implementacion anclada de P84-04)
y mantener un corpus de dominio Twitter/X: aparece informacion distribuicional
complementaria que no se observo en los experimentos anteriores?

Esta autorizacion NO reabre Word2Vec/SGNS/FastText en general.

## 2. P0: auditoria previa (hallazgos verificados)

- **P78 uso la implementacion defectuosa.** `docs/p78_preregistro.md` (linea 63) declara
  que la matriz de embeddings es `modelos/twitter_domain/w2v_twitter_100d.npz`, la misma de
  P73, entrenada por `p71_04`; la auditoria P79 confirmo que su `SkipGram.forward`
  promediaba el termino negativo por K en vez de sumarlo: no implementa SGNS. La
  comparacion de P78 no prueba el objetivo corregido.
- **P84-04 usa la implementacion corregida.** Brazo S = agregacion "suma"; su ancla quedo
  lograda (`reports/p84_04_ancla.json`); el brazo M = "media" reproduce el defecto
  historico como control. La implementacion corregida se reutiliza tal cual.
- **Configuracion congelada recuperada del artefacto** (`reports/p84_04_sgns.json`):
  dim 200, ventana 5, min_count 3, negativos 10, submuestreo 1e-4, 5 epocas, lote 256,
  lr 0.025 por ejemplo (6.4 efectivo) con decaimiento lineal a 1e-3, SGD. Semillas
  historicas 20261016/17/18.
- **Corpus.** Twitter: `corpus_twitter_tokens.txt` (sha256 `1ef872c1...`, EMB-NORMAL-TW v1,
  361.994 documentos, 10.000.029 tokens). General: `tokens_10M.txt` (sha256 `00b839da...`,
  2.847 documentos, 10.002.400 tokens). Ambos ~10M tokens. Procedencia y autorizacion del
  corpus Twitter: `docs/p73_provenance_autorizacion_corpus.md` (D-246); no se descarga nada.
- **Hallazgo de comparabilidad.** El downstream publicado de P84-04 hizo pooling sobre texto
  CRUDO: cobertura de desarrollo 0.6585 con 72 filas muertas (su JSON), la discrepancia de
  normalizacion de P73/P74 sin corregir. El spec de EXP-04 no fijo la normalizacion del
  texto de desarrollo. Bajo la regla de P78 ("texto de entrada normalizado igual que el
  corpus"), el control general publicado no es comparable tal cual.
- **Decision de protocolo.** Normalizar el texto de desarrollo con EMB-NORMAL-TW v1 (la
  tokenizacion documentada del corpus de Twitter/X) para AMBOS brazos. Cobertura medida:
  0.8873 (vocabulario Twitter) y 0.8036 (vocabulario general), filas muertas 1 y 2. El
  control general se RE-PUNTUA con esta normalizacion reutilizando su embedding
  (`modelos/p84/emb_20261016_S.npz`, sha256 `0216f9f6...`, semilla 20261016, brazo S), SIN
  reentrenar. Los numeros publicados de P84-04 se conservan como historia y se reportan
  como referencia.

## 3. Diseno congelado

- **Brazo A (control general):** embedding S de P84-04 (corpus general, semilla 20261016),
  reutilizado sin reentrenar; solo se re-puntua su downstream.
- **Brazo B (candidato Twitter/X):** UN entrenamiento SGNS nuevo, semilla 20261016, misma
  configuracion y mismo flujo de datos (`entrenar_modelos` de P84-04, agregacion "suma").
- **Constantes:** dim 200; ~10M tokens por corpus; ventana 5; min_count 3; negativos 10;
  submuestreo 1e-4; 5 epocas; lote 256; SGD lr 6.4 -> 1e-3; normalizacion del texto de
  desarrollo EMB-NORMAL-TW v1 (identica en los dos brazos); pooling medio (OOV omitidos);
  StandardScaler + LR C=2.0 balanced liblinear rs=42; umbrales congelados de V03 como
  lectura primaria y anidados como secundaria; pliegues agrupados D-016.
- **Semilla unica 20261016** (pareada con el control). Limitacion declarada: con una semilla
  no se estima la varianza de reentrenamiento (P78 midio sd 0.0216 entre semillas); se
  registra y NO se rescata con mas semillas.

## 4. Evaluacion

Para cada brazo: (1) embedding-only; (2) fusion con V03 (media de probabilidades).
Metrica primaria: F1 de la fusion con umbrales congelados ("corrected_cv de la fusion").
Secundarias: F1 a 0.5 (convencion ensemble_gain), por pliegue, por etiqueta, cobertura,
OOV, tiempos y hashes. Complementariedad: matriz V03 x candidato y V03 x fusion con
salvage/damage/net por etiqueta (umbrales congelados). Bootstrap pareado por filas,
10.000 remuestreos, semilla 20260926, sobre el delta de la fusion.

## 5. Criterios de decision (pre-registrados)

PROMISING solo si se cumplen las cinco condiciones:

```text
C1. delta de la fusion (congelada) >= +0.0015 frente a V03 (0.7772114525165721).
C2. >= 4/5 pliegues con delta positivo (fusion vs V03, lectura congelada).
C3. complementariedad: net de la fusion vs V03 > 0 (salvage > damage).
C4. bootstrap del delta de la fusion: IC95 con limite inferior > 0.
C5. valor de dominio: la fusion del brazo Twitter supera a la fusion del brazo general
    (misma lectura primaria).
```

Si no se cumplen C1..C5: CLOSED.

```text
Caso A: C1..C5 se cumplen -> PROMISING; documentar y detenerse.
Caso B: Twitter supera al general (C5) pero la fusion no mejora V03 -> CLOSED;
        la adaptacion de dominio no se transfiere al objetivo final.
Caso C: Twitter no supera al general -> CLOSED; no reabrir la familia de embeddings.
```

No se declara ADOPTED automaticamente aunque resulte positivo. No se genera submission.

## 6. Restricciones y presupuesto

```text
V03 = FROZEN          holdout = FROZEN       test labels = 0
test para desarrollo = 0                     leaderboard tuning = 0
datasets externos etiquetados = 0            pseudo-labeling = 0
embeddings preentrenados = 0                 transformers = 0        LLM = 0
submissions nuevas = 0                       push = 0
```

El corpus externo solo se usa para aprender la representacion desde cero. Sin rejillas
(dimension, ventana, semillas, sampling). Presupuesto: UN entrenamiento nuevo (Twitter) mas
la re-puntuacion del control general. Sin mas experimentos despues de esta comparacion.

## 7. Artefactos previstos

```text
experiments/p85_01_sgns_twitter_corregido.py
reports/toxic_twitter_sgns_corrected.md / .csv
reports/p85_01_sgns_twitter.json
modelos/p85/emb_20261016_S_twitter.npz
features/p85_01_oof_emb.npz
fila P85-01 en experiments.csv; D-290; RUNLOG; seccion 131 del notebook
```

## 8. Hashes congelados

```text
corpus Twitter   1ef872c189d0f405bc1a3f8281cafe08e6b099958e6b8a3816db1460df91f8b9
corpus general   00b839da6fbc377ee5375acc9a236e831e46ecf92cc544188b81aef1a826e871
oof V03          559265816ce4f6312c87f008f90324f0eec70a74d1e6247364c162793338feab
train            3d40e7345b588860cf4744914be88ddc094e4f03591516c8176ab40b7f76d42f
grupos           afabf9ae1c295ea8f203120bf28578cb4bdb4e8f16b89178d820baeaac62e622
embedding S 1016 0216f9f6a952937925d8dcdd1cd885db8f776efed9ed47c339348c840bdff853
```

## 9. Congelado del script

```text
experiments/p85_01_sgns_twitter_corregido.py
sha256 d3cf2013ec10082ceea7d77e76986efac7a7aadbfd172b6607f25ea6bbf565e1
```

## 10. Enmienda 1 (2026-10-08): Gate 0 propio sin el binario fastText

La prueba de humo paro en el Gate 0 heredado porque el binario fastText en disco
(`tools/fastText/build/fasttext`, sha256 `0307cde9...`) difiere del congelado de P84-04
(`59c51cc5...`, del worktree v67 ya retirado). El binario fastText **no participa** en
este experimento (la implementacion SGNS es PyTorch; el ancla se hizo en P84-04), asi que
el script pasa a un Gate 0 propio que verifica los hashes relevantes (corpus general,
OOF de V03, train, grupos, corpus Twitter y embedding general) y registra la incidencia
del binario. No cambia el diseno experimental; el hash del script queda actualizado en
la seccion 9.

## 11. Enmienda 2 (2026-10-08): terminador LF en los escritores CSV

Tras la corrida, la revision del diff del registro detecto que el apendice a
`experiments.csv` (y el CSV del informe) usaban el terminador por defecto del modulo
`csv` (CRLF), lo que reescribia las 456 lineas del registro. Se corrigio a
`lineterminator="\n"` y la fila P85-01 se re-anexo manualmente con LF (diff de una
sola linea). Los resultados de la corrida no cambian; el script ejecutado fue
`d3cf2013...` y el corregido es:

```text
sha256 14035e34c9e4c12319dc4a4f5479e9a0dbee920f845616c6c9d2ded118db71a5
```
