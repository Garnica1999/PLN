# PLAN v4 - KAGGLE_TOXIC_2026_PUJ (Advanced Research v4)

Fecha de apertura: 2026-09-25. Rama: `proyecto-kaggle-toxic-2026-v4` (desde `main`).
Spec maestra: `docs/advanced_research_v4_spec.md` (108 secciones).
Decisiones de gobierno: D-025 (apertura de la fase v4).

## Referencia y protocolo (no se modifican)

- **V03** = union TF-IDF word(1,2) + char_wb(3,5), LogisticRegression C=2.0
  `class_weight='balanced'`, umbrales 0.44/0.51/0.46. LB 0.78306.
  Congelada en `experiments/baseline_v03_frozen/`; verificada en P0 con
  `experiments/26_scaffold_v4.py` (hashes OK, corrected_cv 0.777211 exacto).
- **Protocolo v2 (D-016)**: grupos = near-duplicado UNION pseudo-autor;
  desarrollo = 1.483 tweets, holdout congelado = 248; `corrected_cv` primaria;
  bootstrap pareado por tweet; umbral de contingencia std > 0.025 no activado
  (std actual 0.0161).
- **Métrica primaria**: `f1_media_columnas` (media de la F1 de la clase positiva
  de HS, TR y AG; D-009). Se registran además macro-6, micro, exact match,
  precisión, recall, media y std por pliegue (§68).
- **Regla de adopción (D-016, sin cambios)**: `corrected_cv` >= +0.005 sobre la
  referencia y mejora en >= 3/5 pliegues, **o** complementariedad demostrada
  (`ensemble_gain` >= 0.0015 con IC bootstrap > 0).
- **Presupuesto de control**: holdout 4/5 evaluaciones usadas (la restante se
  reserva para el único mejor finalista); 12 sondas de LB registradas; el envío
  es manual y solo para candidatos que pasen todos los gates.

## Fases y gates

| Fase | Objetivo | Gate de salida |
|---|---|---|
| P0 | Scaffold, verificación de referencia y protocolo, gap analysis, backlog pre-registrado, matriz de cobertura, revisión bibliográfica | V03 verificada byte a byte; `corrected_cv` reproducida; backlog con hipótesis y criteria de cierre; ninguna idea nueva sin clasificar |
| P1 | TRACK A: consistencia de etiquetas; TRACK B: mapa de hardness y separación ruido vs difícil | `label_consistency_v4.csv`, `hardness_v4` y `hardness_vs_noise_v4.md` con cifras; cuantificación de contradicciones y de su relación con el error |
| P2 | TRACK B (experimento): reweighting por hardness; TRACK C: weighting supervisado; cost-sensitive por etiqueta | Cada línea con `corrected_cv`, pliegues, bootstrap y decisión; se aplican los criterios de cierre de línea (mejora o complementariedad) |
| P3 | TRACK E: modelo conjunto de etiquetas sobre las combinaciones válidas | Comparación contra 3 binarios independientes y contra el meta; estabilidad de las combinaciones raras |
| P4 | TRACK H: shift de calibración de las entradas del meta (hipótesis abierta de v2 §4.3) | Medición del shift con simulación de tamaño de entrenamiento; efecto en `corrected_cv` y en la calibración de las entradas |
| P5 | TRACK G: LM class-conditional y clasificadores alternativos | Features de verosimilitud por clase evaluadas por late fusion; revisita acotada de Ridge/SGD/PA y no lineales sobre SVD |
| P6 | TRACK D: subítems genuinamente nuevos (sufijos/prefijos, interacciones sparse, contexto local, HS implícito por subtipo) | Solo con hipótesis ligada a un error medido; si no, cierre por evidencia heredada |
| P7 | TRACK F + discovery: shift no supervisado, vecindad/retrieval, robustez a duplicados, correction matrix, taxonomía v4 | Artefactos de discovery con cifras y verificación de que no usan labels de test |
| P8 | Cierre: máximo 3 candidatos, auditorías, reproducibilidad, memoria, entregables, PR | `v4_candidate_comparison.md`, `rule_audit_v4.md`, `advanced_research_v4.md`, `spec_compliance_v4.md`; submissions solo si pasan gates |

## Criterios de cierre de línea (idénticos a v2)

Una línea se cierra cuando: (a) se probó la variante razonable más fuerte;
(b) no hay mejora en `corrected_cv` ni complementariedad con IC bootstrap > 0;
(c) el resultado es estable entre pliegues. El cierre se documenta con número en
`experiments.csv` y decisión en `DECISIONS.md`, incluso si es negativo (§83).

## Orden de ideas y su prioridad (pre-registro)

1. **A/B**: ruido de etiqueta y dureza de ejemplos (nunca explorados).
2. **C/E**: geometría de la representación (weighting supervisado) y estructura
   conjunta de etiquetas (modelo conjunto; las cadenas ya fallaron).
3. **H**: calibración de las entradas del meta (única línea con evidencia previa
   de dev y holdout que no se transfirió, y con mecanismo identificado).
4. **F/G**: diagnóstico de shift y modelos estadísticos alternativos.
5. **D**: solo subítems no cubiertos por v2.

## Reglas de higiene del experimento

- Un experimento = un script nuevo en `experiments/`; los scripts anteriores no se
  editan ni se borran.
- Todo lo aprendido debe ser OOF o fold-local: nunca se usa el test, ni para
  fitting, ni para vocabulario, ni para umbrales, ni para calibración.
- Los scripts usan `/home/carlos/venv/bin/python` y escriben sus artefactos en
  `reports/`, `features/` y filas en `experiments.csv`.
- Los artefactos grandes van a `features/` (`.npz`, `.parquet`); los informes
  legibles a `reports/`.
