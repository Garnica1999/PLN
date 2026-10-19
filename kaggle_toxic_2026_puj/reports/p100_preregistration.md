# P100 - Pre-registro de la hipotesis H100-01 (composicion social+conversacional)

Fecha: 2026-10-13. Registrado ANTES de construir el corpus B1 y de entrenar.
P98/P99 quedan cerrados historicamente; V03 y sus hashes no se modifican ni
se reutilizan fuera de lo aqui declarado.

## 1. Hipotesis

```text
H100-01:
A escala fija (paridad con C98-1: 29.737.183 tokens), misma arquitectura,
misma exposicion y mismos procedimientos downstream, una composicion que
sustituye ~9.0M tokens de la masa general (wikipedia/mC4; 35.3% del general
de C98-1, 30.3% del total de B1) por dialogo conversacional en espanol
autorizado (ChatSubs es), conservando intacto el bloque social de C98-1,
mejora de forma estable la representacion contextual y el downstream frente
a V03.
```

Hipotesis nula H0: la sustitucion no produce una mejora downstream material y
reproducible (criterios operativos en la seccion 8).

## 2. Cierre pre-entrenamiento de la variante A (sin relleno)

- La variante A (social reforzado a 30%) requeria ~4.5M tokens de social
  NUEVO. La unica fuente candidata (S100-A, DHCOVID) resulto SIN TEXTO
  UTILIZABLE: artefactos publicos = listas de IDs; servicio de texto /get0/
  roto (verificado); rehidratacion excluida por regla.
- Se cierra el brazo A por material, SIN relleno por duplicacion ni
  sustitucion por fuentes dudosas (regla del propietario 2026-10-13).
- Consecuencia: P100 ejecuta UNA configuracion nueva (B1) de las <= 2
  permitidas.

## 3. Variables

- **Independiente**: composicion del corpus (C98-1 = referencia 29.737.183
  tokens vs P100-B1 = candidato ~29.73M: -9.0M masa general, +9.0M
  conversacional; 30.3% del total cada movimiento).
- **Constantes congeladas**: arquitectura LMReal (identica a C98-1;
  char-aware CNN k=3,5; LSTM 2 capas hidden 1024 por direccion; softmax
  atado; ~25M parametros por direccion), tokenizador, vocabulario top-50k de
  C98-1 (`92fecce611f164b7`), char vocab, embedding 200, dropout 0.2
  (nn.Dropout manual), Adam, clip 5, seq_len 96, batch 8192 tokens, objetivos
  LTR/RTL, seeds (20260925), regla de validacion (2% de documentos por
  fuente: mismos 3.320 documentos de P98, excluidos del stream).
- **Exposicion**: 4 epocas por direccion sobre B1 (~117-118M tokens
  procesados por direccion; equivalencia con C98-1).

## 4. Datos y composicion exacta de P100-B1 (nominal ~29.73M)

| Componente | Objetivo | Fuente | Procedencia |
|---|---|---|---|
| social | 4.227.450 | twitter_domain COVID-es | MISMA seleccion de C98-1 (148.413 docs post-dedup; receta determinista de p98_11) |
| conversacional | >= 9.000.000 | ChatSubs es (open_subtitles_es) | nueva (S100-B autorizado); solo subconjunto es |
| general wiki | 11.650.000 | wikipedia_es_50M | prefijo por documentos de la seleccion wiki de C98-1 (subconjunto exacto) |
| general mC4 | 4.850.000 | mc4_es (10M) | prefijo por documentos de la seleccion mC4 de C98-1 (subconjunto exacto) |
| **Total** | **29.727.450** | | paridad con C98-1 (29.737.183; sha256 `9f562718ef4e7d45...`) |

- Cifras de referencia de C98-1 (medidas, replicadas con p98_11): wiki
  18.009.963 / mC4 7.499.770 / twitter 4.227.450 tokens; 166.022 documentos;
  sha256 `9f562718ef4e7d45622040cd9ab5973182cc5405a99bfbc1636fa57abc6b9f0e`.
- Delta neto B1 vs C98-1: -9.009.733 general +9.000.000 conversacional;
  social identico; escala par. Comparacion controlada de un solo factor.
- Regla anti-insuficiencia: si un componente queda por debajo del objetivo,
  se extiende dentro del mismo pool (techo: la seleccion completa de C98-1
  para general; el archivo ChatSubs completo para conversacional); sin
  duplicar documentos; la cifra final observada manda y se registra.
- Verificacion de subconjunto: social + general de B1 deben estar contenidos
  EXACTOS en C98-1 (reconstruccion determinista con la receta de p98_11
  sobre el mismo pool; pool twitter hash `1ef872c189d0f405`; verificacion
  por conteo 148.413 docs social y por interseccion exacta contra el corpus
  de referencia). Discrepancia -> detener y documentar.
- Obligaciones ChatSubs (propietario 2026-10-13): atribucion (ChatSubs /
  Kharitonova et al.; OpenSubtitles); solo subconjunto es; sin anotaciones;
  no redistribuir; no publicar checkpoints/pesos/derivados sin revision
  especifica. Descarga verificada: md5 `afd2d3967cf8cdf78883a7baa8d061e4`
  (1.377.313.069 bytes; igual al oficial del registro Zenodo).
- Regla de formato ChatSubs: unidad documental = 1 archivo JSONL (un
  subtitulo); texto = turnos unidos por salto de linea; muestreo uniforme por
  documento con semilla 20260925 hasta >= 9.0M tokens; dedup exacta. Si la
  estructura real difiere, se aplica la unidad analoga mas cercana y se
  documenta ANTES de contar tokens (no silencioso).

## 5. Deduplicacion

- Exacta por documento normalizado (sha1 de minusculas-sin-tildes; mismo
  procedimiento validado en P98/P99) entre componentes, orden
  wiki->mC4->social->conversacional; se conserva la primera aparicion.
- Se registran: duplicados exactos removidos por fuente, overlap cruzado,
  overlap con train de TOXIC (auditoria exacta; 0 esperado), overlap
  train-validacion (0 por construccion).
- Politica near-duplicate: la misma de P98 (solo exacta; limitacion
  declarada).
- No se escanea toxic_test (regla vigente).

## 6. Entrenamiento (reglas exactas)

- Misma pipeline de p99_11 (LMReal; vocabulario congelado de C98-1; EOS
  entre documentos; seed 20260925; barajado de documentos del corpus con la
  misma semilla).
- Exposicion: 4 epocas por direccion sobre P100-B1.
- **Calendario de lr por tokens (equivalente al de C98-1)**: `lr = 1e-3`
  durante el primer 75.1% de los pasos totales y `lr = 5e-4` el resto.
- Parada temprana: se conserva el mejor checkpoint por val loss; el run se
  detiene en la primera epoca >= 2 cuya val loss no mejore la mejor
  historica (equivalente a C98-1).
- Validacion: medidor FIJO de los mismos 3.320 documentos de P98
  (materializados desde C98-1; sha verificado), evaluado identico en ambos
  brazos; cualquier documento de validacion que aparezca en el material de
  B1 se excluye de su stream de entrenamiento (auditoria por hash; esperado
  ~3.189 documentos presentes en el material y 0 en el stream tras la
  exclusion). Dominios de validacion:
  wikipedia/mC4/twitter; chatsubs NO entra en validacion (su perdida se
  monitorea solo como diagnostico de entrenamiento, sin efecto en la parada
  temprana).
- Presupuesto: entrenamiento esperado ~3.9 GPU-h; tope de fase 10 GPU-h
  (propietario). No se reentrena el control C98-1.

## 7. Evaluacion downstream (pre-registrada)

- Mismos procedimientos congelados de D98-1: reps = concat(capa2 LTR, capa2
  RTL) por token, media sobre posiciones (2048d); embedding-only LR C=1;
  fusion = TF-IDF del fold (word 1-2 + char_wb 3-5, min_df 2, sublinear) +
  bloque denso, LR C=2 balanced; umbrales V03 0.44/0.51/0.46; grupos
  canonicos; bootstrap por clusters (unidad=grupo, semilla 20260925).
- Brazos por particion: A=V03 (frozen); B=fusion 30M (reps congeladas de P98
  `8f1aeeec...`; recalculada en el mismo script como control de reproduccion:
  debe dar 0.7822662599 en la canonica y 0.7767830429 en la 47);
  C=fusion P100-B1 (reps nuevas).
- Particiones: (1) canonica (folds 1..5 de `oof_v03_final`); (2) la segunda
  particion agrupada pre-registrada en P98 (`64df5d50...`, seed 43),
  recalculada y verificada por hash de folds. No se elige particion; ambas se
  reportan separadas; sin concatenar folds entre particiones.
- Metricas por brazo: corrected_cv, delta vs V03, delta vs 30M, 5 fold
  scores, F1 por etiqueta, salvage/damage/neto, IC95 por clusters (unidad=
  grupo), tiempos y hashes.
- Slices: las predefinidas (largos/cortos/hashtag/url/mencion/elongaciones;
  se anaden coloquial/insultos/colectivos/afiliacion/interpersonal/cita si
  las definiciones existentes lo permiten); se muestran tambien las
  negativas.
- Nada de holdouts ni leaderboard; sin submissions.

## 8. Criterios de decision (gate + atribucion)

Mantener abierta la linea SOLO si se cumple TODO:

1. delta primario vs V03 >= +0.003 en la particion canonica;
2. >= 4/5 pliegues positivos en la particion primaria;
3. complementariedad neta positiva;
4. delta positivo vs V03 en la segunda particion, consistente y no marginal
   (definido: delta47 > 0 y >= 3/5 pliegues positivos en esa particion);
5. sin defectos de leakage/procedencia;
6. **atribucion**: delta vs fusion 30M > 0 en la particion canonica; si
   B1 <= fusion 30M, H100-01 queda FALSIFICADA (la composicion no aporta
   sobre el corpus de referencia) aunque 1-5 pasen.

- Si solo mejora la canonica (o la segunda particion resulta marginal):
  MARGINAL y detener la linea.
- Si el IC incluye cero, se declara la incertidumbre (no mejora demostrada).
- Cierre por: no convergencia; presupuesto agotado; material insuficiente;
  restriccion material de ChatSubs. Sin rescates (no mas tokens, no
  arquitectura, no iteraciones extra).

## 9. Predicciones falsables

- P1 (lenguaje): P100-B1 converge con val ppl del mismo orden que C98-1
  (<= 1.15x la ppl de C98-1 en exposicion equivalente; medidor fijo 3.320).
- P2 (downstream primario): fusion B1 en canonica: delta vs V03 >= +0.003 con
  >= 4/5 pliegues.
- P3 (robustez): delta vs V03 > 0 tambien en la particion 47.
- P4 (complementariedad): neto > 0 en ambas particiones; sin damage
  consistente a V03.
- P5 (atribucion): fusion B1 > fusion 30M en la canonica (delta > 0).

Falsacion: cualquiera de P2-P5 incumplida segun las reglas de la seccion 8.

## 10. Paradas y presupuesto

- Detener y documentar sin escalar si: B1 queda < 29.5M sin material
  disponible dentro de las reglas; no converge; el gate falla; se agotan los
  10 GPU-h; aparece una restriccion material de ChatSubs.
- Configuraciones nuevas: 1 de 2 (A cerrada). GPU-h: ~3.9 esperados
  (entrenamiento) + < 0.5 (downstream/reps); tope de fase 10.

## 11. Trazabilidad

- Artefactos planificados: `artifacts/p100_b1_manifest.json`; corpus
  `data/external/corpora/p100_b1/tokens_30m.txt`; checkpoints
  `models/p100_b1_{ltr,rtl}.pt`; `reports/p100_b1_pretraining_results.csv`,
  `p100_b1_downstream_results.csv`, `p100_b1_complementarity.csv`,
  `p100_b1_final_report.md`; filas nuevas en experiments.csv (P100-00
  registro; P100-11/12 entrenamiento; P100-13/14 downstream).
- Scripts planificados: `experiments/p100_10_construir_b1.py`,
  `p100_11_preentrenar_b1.py`, `p100_12_downstream_b1.py` (no se editan
  scripts historicos).
- Fuentes S1: `reports/p100_corpus_research.md`, `p100_corpus_registry.csv`,
  `p100_source_decisions.md` (S100-B ChatSubs: ALLOWED_WITH_RESTRICTIONS;
  obligaciones registradas).
- Sin commit; sin submissions; ASCII en artefactos generados.
