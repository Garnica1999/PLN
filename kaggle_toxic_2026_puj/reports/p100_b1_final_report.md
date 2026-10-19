# P100-B1 - Informe final (composicion social+conversacional a escala fija)

Fecha: 2026-10-14. Fase P100 (S3-S5). H100-01 pre-registrada en
reports/p100_preregistration.md; gate aplicado sin modificaciones posteriores.

## 1. Hipotesis y diseno

- H100-01: a escala fija (~29.73M, paridad C98-1), sustituir ~9.0M tokens de
  masa general por dialogo conversacional en espanol autorizado (ChatSubs),
  conservando 15% social, mejora la representacion contextual y el downstream
  frente a V03.
- Variante A (social reforzado) cerrada por material ANTES de construir
  (S100-A sin texto utilizable; sin relleno por duplicacion). 1 configuracion
  nueva de 2 permitidas.

## 2. Corpus B1 (S3)

- 29.727.450 tokens EXACTOS: social 4.227.450 (misma seleccion de C98-1) +
  conversacional 9.000.000 (ChatSubs es) + wikipedia 11.650.000 + mC4
  4.850.000; 162.977 documentos. sha256 5111a1832fe9567b...
- Controles superados: replica exacta de C98-1 (claves identicas contra el
  corpus en disco), subconjunto verificado (2 prefijos limite registrados),
  val fijo de 3.320 documentos (3.189 presentes; excluidos del stream),
  overlap exacto vs train = 0, conversacional vs S99-01 = 0. Sin toxic_test.
- Manifiesto: artifacts/p100_b1_manifest.json.

## 3. Entrenamiento (S4)

- 4 epocas por direccion; lr 1e-3 -> 5e-4 al 75.1% de los pasos; vocab
  congelado de C98-1; val ppl sobre el medidor fijo de 3.320.
- LTR: 207.6 / 136.8 / 111.4 / 98.7. RTL: 205.7 / 138.7 / 112.0 / 98.6.
- Checkpoints: LTR sha a4c057a762500f54... (ppl 98.7); RTL sha
  fb38bbbdac055732... (ppl 98.6).
- GPU-h reales ~3.78 (tope de fase 10). Incidente: 1 eviccion del servidor
  (22:54) reanudada automaticamente por el supervisor detached; sin perdida.
  El JSON reconstruye parcialmente el historial LTR del log (2.719 GPU-h
  reportadas); la correccion quedo anotada en experiments.csv.

## 4. Downstream (S5; recetas congeladas D98-1; umbrales V03)

Controles de reproduccion obligatorios: 30M canonica 0.7822662599 y 47
0.7767830429 reproducidos EXACTOS (ok=true). Dev = 1.483 documentos; reps B1
sha 6524eae950b242e4.

| Brazo | Canonica | Particion 47 |
|---|---|---|
| V03 (frozen / equivalente) | 0.7772114525 | 0.7756319869 |
| fusion 30M (C98-1; control) | 0.7822662599 | 0.7767830429 |
| **fusion B1** | **0.7766201251** | **0.7749864975** |

Deltas pareados (mismos pliegues):

| Comparacion | Canonica | Particion 47 |
|---|---|---|
| B1 vs V03 | -0.000591 (2/5 pliegues) | -0.000645 (2/5) |
| B1 vs 30M | -0.005646 | -0.001797 |

- F1 por etiqueta de B1: HS 0.7681 / 0.7566; TR 0.8089 / 0.8140;
  AG 0.7529 / 0.7544 (canonica / 47).
- F1 por pliegue (B1): canonica [0.7909, 0.7929, 0.7816, 0.7431, 0.7748];
  47 [0.7697, 0.7653, 0.7737, 0.8003, 0.7661]. V03: canonica [0.7960,
  0.7940, 0.7789, 0.7386, 0.7791]; 47 [0.7726, 0.7645, 0.7748, 0.7911,
  0.7745].
- Correccion (salvage/damage/net): vs V03 canonica 73/73/0; 47 76/77/-1;
  vs 30M canonica -11 neto; 47 -3 neto.
- IC95 por clusters (bootstrap, unidad=grupo): B1 vs V03 [-0.008191,
  +0.007489] canonica y [-0.008613, +0.007567] 47; B1 vs 30M [-0.013324,
  +0.002151] canonica y [-0.009432, +0.005870] 47.
- Nota informativa: embedding-only B1 0.6602 > embedding-only 30M 0.6522 en la
  canonica (+0.0079), pero la fusion (metrica del gate) no lo traduce.
- Slices: los positivos de la canonica (long-q4 +0.0197, hashtag +0.0140, url
  +0.0093) NO se replican en la particion 47 (url -0.0011, hashtag -0.0016;
  long-q4 +0.0138); perfil mixto sin patron consistente.

## 5. Gate (pre-registro seccion 8; sin modificaciones)

| Criterio | Resultado |
|---|---|
| 1. delta vs V03 >= +0.003 (canonica) | FALLA (-0.000591) |
| 2. >= 4/5 pliegues (canonica) | FALLA (2/5) |
| 3. neto positivo (canonica) | FALLA (0) |
| 4. particion 47 positiva y no marginal | FALLA (-0.000645; 2/5) |
| 5. sin defectos de leakage/procedencia | CUMPLE |
| 6. atribucion: delta vs fusion 30M > 0 | FALLA (-0.005646) |

- Veredicto: **GATE_FAILED**; atribucion_falsada = true;
  marginal_solo_canonica = false.

## 6. Conclusion

- **H100-01 FALSIFICADA**: sustituir 30% de la masa general por dialogo
  conversacional (ChatSubs es) a escala fija, con la misma arquitectura,
  exposicion y procedimientos, NO mejora el downstream; queda por debajo del
  control de 30M (C98-1) en ambas particiones y marginalmente por debajo de
  V03.
- Incertidumbre: los IC95 vs V03 incluyen cero (no hay mejora demostrada; el
  estimador puntual es negativo). Vs 30M, el IC superior apenas roza cero
  (+0.0022 en la canonica) con 92.1% de las remuestras a favor del 30M.
- La linea de composicion de P100 se detiene aqui, conforme al criterio
  pre-registrado. Sin rescates; sin nuevas variantes.
- V03 permanece como referencia. Sin submissions; sin commits; P98/P99
  intocados.

## 7. Artefactos y trazabilidad

- reports/: p100_corpus_research.md, p100_corpus_registry.csv,
  p100_source_decisions.md, p100_preregistration.md, p100_auditorias_s1.md,
  p100_b1_pretraining.{json,md}, p100_b1_pretraining_results.csv,
  p100_b1_downstream_canonica.json, p100_b1_downstream_part47.json,
  p100_b1_downstream_results.csv, p100_b1_complementarity.csv,
  p100_b1_final_report.md (este), figuras/p100_12_{canonica,part47}.png.
- artifacts/p100_b1_manifest.json; data/external/corpora/p100_b1/tokens_30m.txt;
  models/p100_b1_{ltr,rtl}.pt; features/p100_b1_reps_dev.npz.
- experiments/: p100_00_registro_fuentes.py, p100_10_construir_b1.py,
  p100_11_preentrenar_b1.py, p100_12_downstream_b1.py, p100_90_registrar.py,
  p100_supervisor.sh; logs: p100_pretraining.log,
  p100_pretraining_intento_interrumpido.log, p100_supervisor.log,
  p100_downstream.log.
- experiments.csv: filas P100-00, P100-10, P100-11, P100-12, P100-13, P100-14
  (491 filas totales).
