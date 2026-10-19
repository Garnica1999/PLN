# P98 - Informe final: LM recurrente contextual desde cero

Fecha del cierre documental: 2026-10-13 (completado en el P0 de P99, que lo
exige como lectura; **no altera ningun resultado de P98**). Fases cubiertas:
gobernanza y Fase B completas (C98-0, C98-1, D98-1, D98-1b).

## 1. Fuentes utilizables y licencias

- **Wikipedia es (CC BY-SA 4.0)**: ALLOWED; 3 muestras locales (5M/10M/50M).
- **mC4 es (ODC-BY 1.0, allenai/c4)**: ALLOWED_WITH_RESTRICTIONS (atribucion;
  familia Common Crawl tratada como UNA fuente).
- **twitter_domain COVID-es (CC BY-SA 4.0, Kaggle mmaguero; ventana
  cerrada 2020-03-29 a 2020-05-01)**: ALLOWED; autorizacion escrita
  (docs/p73) + confirmacion del mecanismo (docs/p98); auditoria de
  solapamiento: 0 exacto/normalizado, near-dup max Jaccard 0.142857 < 0.2.
- Detalle: `reports/p98_corpus_registry.csv`, `reports/p98_corpus_licenses.md`.

## 2. Fuentes excluidas o en cuarentena

- CUARENTENA ACTIVA (5): HateEval-ES, Spanish-MisoCorpus-2020, HaterNet,
  IberEval/AMI, HaSCoSVa. Sin uso; sus labels tampoco.
- Registradas sin descarga: OSCAR (gated), CulturaX, HPLT 2.0.
- Candidatas condicionadas: Gutenberg/Wikisource (libros), TASS y corpus de
  sentimiento/emocion (NEEDS_EXPLICIT_PERMISSION).

## 3. Tokens y documentos por fuente (C98-1)

| Fuente | Seleccion | Documentos (post-dedup) | Nota |
|---|---|---|---|
| Wikipedia | 18.009.963 tokens | 5.191 | prefijo por documentos del archivo de 50M |
| mC4 | 7.500.009 tokens | 12.418 | 4 duplicados removidos |
| Twitter COVID-es | 4.500.010 tokens (9.656 dups internos removidos) | 148.413 | muestreo uniforme por documento (seed 20260925) |
| **Total** | **29.737.183 tokens** | **166.022** | sha256 `9f562718...` |

## 4. Se aprendio una representacion contextual verificable?

- C98-0 (piloto, 5M): val ppl 651 -> 351.
- C98-1 (30M, 4+4 epocas): LTR val ppl 94.8; RTL 91.7; curvas monotonas y
  simetricas; checkpoints hasheados; reps extraibles y estables.
- Conclusion: el objetivo de lenguaje se aprende correctamente (a nivel de
  modelado del lenguaje). La perplexity NO es criterio de clasificacion.

## 5. Aporta mas que FastText estatico y TF-IDF word+char?

- Embedding-only contextual: 0.652238; referencia estatica v5.1: 0.6523
  (practicamente identicas) -> NO mejor representacion discriminativa por si
  sola.
- TF-IDF word+char (V03) sola: 0.7772114525165721 >> embedding-only.
- Fusion contextual: 0.782266 vs fusion estatica v5.1 0.7818 (+0.0005) ->
  sin ventaja material sobre la linea estatica.

## 6. Depende el resultado del corpus o de la arquitectura?

Con la evidencia de P98, no puede atribuirse a ninguno de los dos: la unica
senal observada (canonica +0.005055) desaparecio al cambiar de particion
( +0.001151, 2/5 pliegues). Es consistente con variacion de particion
(banda ~0.017, D-070) y con el freno historico de transferencia.

## 7. Complementariedad con V03

- Canonica: neto +11 celdas (103 rescatadas / 92 rotas); perfil por slices:
  q4 (largos) +0.0271, hashtag +0.0125, url +0.0118; q1 (cortos) -0.0061.
- Replica (particion 47): neto +2; el perfil no replico (q4 +0.011; url y
  hashtag negativos).
- Conclusion: complementariedad NO persistente.

## 8. Sobrevive la mejora al cambio de particion?

No. D98-1b: FALSIFICADA por los criterios pre-registrados (delta +0.001151,
2/5 pliegues, IC95 [-0.007491,+0.009919] incluye cero, neto +2).

## 9. Hay evidencia suficiente para mas escala?

Al cierre de P98: no. (La pregunta de escala quedo explicitamente delegada a
una fase nueva; esa es P99, con pre-registro propio.)

## 10. Existe una ruta empirica plausible hacia una mejora de leaderboard?

Con la evidencia de P98: **no**. V03 permanece congelado
(corrected_cv 0.7772114525165721; LB 0.78306). La ruta de representaciones
contextuales desde cero quedo cerrada con evidencia en la particion de
robustez; cualquier extension exige una hipotesis nueva y falsable.

## Presupuesto y condiciones

- GPU-h P98: piloto 0.16 + C98-1 3.775 = ~3.94 de 24 de fase; principales 5/8.
- Sin transformers, pesos preentrenados, labels externas ni pseudo-labeling;
  sin texto de test ni holdouts en el corpus; sin submissions; sin alance
  (0 usos de leaderboard).
- Artefactos clave: corpus 30M (`9f562718...`), checkpoints LTR `472891a7...`
  y RTL `cff68d45...`, vocab `92fecce6...`, reps `8f1aeeec...`, D98-1 json
  `56240d43...`, pre-registro part47 `64df5d50...`, replica `6c61095d...`.
