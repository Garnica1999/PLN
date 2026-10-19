# QA de consistencia — Informe final AGE (`informe/informe.tex`)

**Fecha**: 2026-10-18. **Alcance**: `informe/informe.tex` (558 líneas, 14 páginas) contra las
fuentes primarias del proyecto. **Método**: extracción de cifras del .tex y contraste contra
`CHAMPION.md`, `RUNLOG.md`, `DECISIONS.md`, `experiments/*.json`, `experiments.csv`,
`reports/age_class_thresholds_per_class.csv` y `informe/dossier_informe.md`.

## Verificaciones OK (contra fuentes)

| Bloque | Resultado |
|---|---|
| Campeón V48: CV 0.6186451 / LB 0.63724 (#1, Ref. 57283487) / ventaja +0.00196 | OK (CHAMPION.md) |
| Pesos de los 6 expertos [0.5628, 0.0832, 0.1724, 0.0754, 0.0513, 0.0549] | OK (v48_01 JSON) |
| Récord 2026: V53 0.6188259 (p=0.0280), V54 0.6188887 (p=0.0050), V56 0.6189929 (LB 0.63717, Ref. 57350295, 5/5 folds, p=0.0010, IC95 [+0.0001485, +0.0005590]) | OK (CHAMPION/RUNLOG) |
| Pesos V56 (0.5594 ... 0.0238) y deltas por fold vs V54 | OK |
| Tabla plug-in por clase (F1 argmax→plug-in, recall, n predicho) | OK (age_class_thresholds_per_class.csv) |
| Hitos E-001, V12, V19, V20, V30, V30+plug-in, V38-V43, V41/V43, V48, V53/V54/V56 | OK (CHAMPION.md) |
| Datos/EDA: 169.336/112.892; clases 17.0-16.0 %; 1.21 %/4.26 %; 271.539/282.228; mediana 93; 643 (0.38 %); suelos 0.1667/0.04854; segunda partición 0.6144120/0.6164700 | OK (audit_01_resumen.json, DECISIONS.md) |
| V57: f1 base 0.6189929489711685; deciles 0.278/0.268 y 0.992/0.995; menciones 63.9 %/0.6802 vs 36.1 %/0.4997; Power-Mean α=0 óptimo (−1.4e-4 / −2.9e-5); símplex |Δ|≤~4e-4 | OK (v57 JSON) |
| Falsaciones V44-V52 y log-loss V56 (−0.0002932, 0/5) | OK (DECISIONS/fase4_5, RUNLOG) |
| Techo D-087: 0.642; oráculo +0.0568 < +0.0626; 38.62 %/59.94 % | OK (DECISIONS.md:3160-3235) |
| Reproducibilidad: 178 celdas, 4 h 35 min, 13/13, idempotencia 12/12, tolerancias; 220 experimentos; 110 decisiones | OK (qa_final.md, plan maestro) |
| Envíos: 15 registrados/12 puntuados, brecha +0.0177 ± 0.0023 | OK (RUNLOG/FINAL_REPORT) |

## Discrepancia encontrada y corregida

- `informe.tex` (Sección IV, regla plug-in): la frase "los 11.226 cambios ... confianza media
  0.3357, frente a 0.6136 del conjunto" mezclaba la partición canónica (11.226, confianza 0.33455,
  `age_umbrales_validacion.json`) con la alternativa (11.225, 0.3357, `age_plugin_confirmacion.json`).
  **Corregido** a valores canónicos consistentes: "confianza media es 0.3346, y las 76.033 filas de
  confianza alta conservan su etiqueta (cero cambios)". Recompilado: 14 páginas, 0 errores.

## Integridad y estructura (verificación directa)

- 51 claves `\cite` usadas, todas presentes en `references.bib`; `rosenthal2011` NO citada. OK.
- 14 archivos `figuras/*.png` referenciados, todos existentes. OK.
- Sin TODOs/placeholders; "token" aparece solo como término léxico (no credenciales). OK.
- Secciones I-XI en orden + resumen + palabras clave + declaración de impacto + cierre
  `\bibliographystyle{ieeetr}`/`\bibliography{references}`. 14 páginas (log). OK.

## Veredicto

**GO** — ninguna cifra clave presenta contradicción con las fuentes; la única discrepancia menor
detectada quedó corregida y recompilada.
