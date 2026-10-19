# P99 - Pre-registro de la hipotesis H99-01 (escala de corpus)

Fecha: 2026-10-13. Registrado ANTES de construir el corpus y de entrenar.
P98 queda cerrado historicamente; sus artefactos no se modifican ni se
reutilizan fuera de lo aqui declarado.

## 1. Hipotesis

```text
H99-01:
Aumentar de 30M a 60M tokens UNICOS de los mismos tipos de fuente y
presupuesto de tokens procesados ~constante (30M x 4 epocas -> 60M x 2 epocas
por direccion) mejora de forma estable la representacion contextual y el
downstream frente a V03.
```

Hipotesis nula H0: el mayor corpus no produce una mejora downstream material y
reproducible (los criterios operativos estan en la seccion 5).

## 2. Variables

- **Independiente**: corpus de preentrenamiento (C98-1 = 30M referencia vs
  S99-01 = 60M candidato).
- **Constantes congeladas**: arquitectura (char-aware CNN k=3,5; LSTM 2 capas
  hidden 1024 por direccion; softmax atado; ~25M parametros por direccion),
  tokenizador, vocabulario top-50k de C98-1 (`92fecce611f164b7`), char vocab,
  embedding 200, dropout 0.2, Adam, clip 5, seq_len 96, batch 8192 tokens,
  objetivo LTR/RTL, seeds (20260925), regla de validacion (2% documentos por
  fuente, mismos 3.320 documentos de P98).
- **Exposicion**: tokens procesados por direccion ~= 117-120M en ambas ramas
  (se cuentan tokens reales; el padding no cuenta porque los lotes son
  contiguos sin padding intra-lote).

## 3. Datos y composicion (pre-registrado con cifras disponibles medidas)

Material local autorizado y disponible (verificado en P0 de P99):

| Fuente | Total archivo | Usado en C98-1 | Disponible nuevo | Objetivo nominal | **Efectivo S99-01** |
|---|---|---|---|---|---|
| Wikipedia (50M) | 50.000.028 | 18.009.963 | ~31.99M | 36M | **41M** |
| mC4 (10M) | 10.000.239 | 7.500.009 | ~2.50M | 15M | **10M (tope del archivo)** |
| Twitter (10.3M) | 10.304.387 | 4.500.010 | ~5.80M | 9M | **9M** |
| **Total** | 70.3M | 29.74M | ~40.3M | 60M | **60M** |

- Los 60M son alcanzables SOLO con material local (sin descargas): el tope de
  mC4 (10M) obliga a que Wikipedia absorba la diferencia (41M en lugar de
  36M). Proporciones efectivas ~68.3/16.7/15.0 frente a 60/25/15 nominales.
  Esta desviacion queda registrada ANTES de construir y entrenar; no se
  descargan fuentes nuevas (P99 seccion 5/6). El objetivo de twitter (9M) SI
  es exacto.
- Regla anti-insuficiencia: si tras deduplicar el corpus final queda <60.0M
  tokens, se extiende el prefijo de Wikipedia (con techo en su archivo) hasta
  alcanzar >=60.0M; la cifra final observada manda y se registra.
- S99-01 CONTIENE integramente a C98-1 (los 30M de referencia + ~30M nuevos
  documentos unicos); los documentos nuevos no repiten ninguno de C98-1.
- Se conservan los mismos 3.320 documentos de validacion de lenguaje de P98
  (misma construccion, mismo seed, verificados por hash contra el conjunto
  de S99-01 y excluidos del stream de entrenamiento).

## 4. Deduplicacion

- Exacta por documento normalizado (sha1 de minusculas-sin-tildes; el mismo
  procedimiento validado en P98) ENTRE todas las selecciones antes de fijar el
  corpus; se conserva la primera aparicion (orden wiki->mc4->twitter).
- Se registran: duplicados exactos removidos por fuente, overlap cruzado,
  overlap con train de TOXIC (auditoria exacta; 0 esperado), overlap
  train-validacion (0 por construccion), y la relacion con P98 (C98-1 es
  subconjunto; se verifica que los 166.022 documentos y 29.737.183 tokens de
  referencia estan presentes EXACTOS en S99-01).
- Politica near-duplicate: la misma de P98 (solo exacta; limitacion declarada).
- No se escanea toxic_test (regla vigente).

## 5. Entrenamiento (reglas exactas)

- Misma arquitectura y pipeline de p98_12 (LMReal; vocabulario congelado de
  C98-1; EOS entre documentos; seed 20260925).
- Exposicion: 2 epocas por direccion sobre S99-01 (60M) ~= 4 epocas de C98-1
  (30M) en tokens procesados.
- **Calendario de lr por tokens (equivalente al de C98-1)**: C98-1 uso 1e-3 en
  las epocas 1-3 (0-88.0M tokens de 117.2M = 75.1%) y 5e-4 hasta el final.
  S99-01 replicara: `lr = 1e-3` durante el primer 75.1% de los pasos totales y
  `lr = 5e-4` el resto (umbral calculado sobre los pasos reales del stream).
- Parada temprana: identica (si la val loss de la epoca 2 no mejora la de la
  epoca 1, se conserva el mejor checkpoint).
- Validacion: loss/perplexity por dominio (wikipedia/mC4/twitter) y agregada,
  sobre los mismos 3.320 documentos.
- Presupuesto: **<=6 GPU-h** para S99-01 LTR+RTL (<=8 GPU-h de fase; sin
  reentrenar la referencia de 30M).

## 6. Evaluacion downstream (pre-registrada)

- Mismos procedimientos congelados de D98-1: reps = concat(capa2 LTR, capa2
  RTL) por token, media sobre posiciones (2048d); embedding-only LR C=1;
  fusion = TF-IDF del fold (word 1-2 + char_wb 3-5, min_df 2, sublinear) +
  bloque denso, LR C=2 balanced; umbrales V03 congelados; grupos canonicos.
- Brazos por particion: A=V03 (frozen), B=fusion 30M (reps congeladas de P98
  `8f1aeeec...`; recalculada en el mismo script como control de reproduccion:
  debe dar 0.7822662599 en la canonica y 0.7767830429 en la 47), C=fusion 60M
  (reps nuevas).
- Particiones: (1) canonica (folds 1..5 de `oof_v03_final`); (2) la segunda
  particion agrupada pre-registrada en P98 (`64df5d50...`, seed 43),
  recalculada y verificada por hash de folds. No se elige particion; ambas se
  reportan separadas; sin concatenar folds entre particiones.
- Metricas por brazo: corrected_cv, delta vs V03, delta vs 30M, 5 fold scores,
  F1 por etiqueta, salvage/damage/neto, IC95 por clusters (unidad=grupo),
  tiempos y hashes.
- Slices: las predefinidas (largos/cortos/hashtag/url/mencion/elongaciones;
  se anaden coloquial/insultos/colectivos/afiliacion/interpersonal/cita si
  las definiciones existentes lo permiten); se muestran tambien las negativas.

## 7. Criterios de decision (gate de escala, copiados del encargo)

Mantener abierta la investigacion de escala SOLO si se cumple TODO:
1. delta primario vs V03 >= +0.003 en la particion canonica;
2. mejora positiva frente al modelo de 30M;
3. >=4/5 pliegues positivos en la particion primaria;
4. delta positivo vs V03 en la segunda particion (consistente, no marginal);
5. complementariedad neta positiva;
6. sin defectos de leakage/procedencia.
Si el IC incluye cero, se declara la incertidumbre (no mejora demostrada).
Cerrar la hipotesis de escala si: mejora despreciable; solo mejora la
perplexity; falla la segunda particion; desaparece la complementariedad; el
candidato dana consistentemente aciertos de V03; o el coste no se justifica
por los documentos extra. Sin rescates (no mas tokens, no otro optimizador,
no cambios de arquitectura).

## 8. Predicciones falsables

- P1 (lenguaje): S99-01 converge con val ppl del mismo orden que C98-1
  (<= 1.15x la ppl de C98-1 en exposicion equivalente).
- P2 (downstream primario): fusion 60M en canonica: delta vs V03 >= +0.003 con
  >=4/5 pliegues y > 0 vs 30M.
- P3 (robustez): delta vs V03 > 0 tambien en la particion 47.
- P4 (complementariedad): neto > 0 en ambas particiones.
Falsacion: cualquiera de P2-P4 incumplida (con las reglas de la seccion 7).

## 9. Paradas

Detener y documentar sin escalar si: no converge; el presupuesto de 6 GPU-h se
agota antes de terminar; el corpus final <60.0M sin material local para
completarlo; cualquier condicion del gate falla.

## 10. Trazabilidad

Artefactos: `p99_corpus_scaling_plan.md` (plan), `artifacts/p99_s99_01_manifest.json`,
corpus `data/external/corpora/p99_s99_01/tokens_60m.txt`, checkpoints
`models/p99_s99_01_{ltr,rtl}.pt`, `reports/p99_corpus_results.csv`,
`p99_pretraining_results.csv`, `p99_downstream_results.csv`,
`p99_complementarity.csv`, `p99_final_report.md`; filas nuevas en
experiments.csv (P99-00 registro; P99-11/12 entrenamiento; P99-13/14
downstream). Sin commit; sin submissions; ASCII.
