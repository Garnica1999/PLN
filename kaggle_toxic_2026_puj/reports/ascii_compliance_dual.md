# Verificacion de ASCII: las dos comprobaciones, por separado

Generado por `experiments/p67_03_cierre_documental.py`. Fecha 2026-09-29T17:10:47+00:00.

## Por que hay dos

La regla del proyecto es **0 bytes generados > 127**. El verificador oficial
`tools/verificar_caracteres.py` **no implementa esa regla**: vigila un conjunto de
alfabetos no latinos, y su lista **no incluye Latin-1 Supplement ni Latin Extended**.

La lista de categorias que si vigila:

```text
hebreo, arabe, devanagari, tamil y telugu, cirilico, griego y copto,
hangul, kanji y kana, emoji y simbolos
```

Comprobado con un fichero de prueba que contenia una `e` acentuada, una `n` con
virgulilla, alfa, beta y un caracter cirilico: el verificador oficial **reporto el
cirilico y no reporto los acentos latinos del mismo fichero**.

Por tanto el verificador oficial **no demuestra compliance** por si solo. La
comprobacion autoritativa es el escaneo byte a byte, y las dos se reportan aqui
separadas en vez de fundidas en un unico 'OK'.

## Resultado 1: verificador oficial

```text
exit code            0
linea de resultado   OK: 0 caracteres no permitidos en los ficheros propios del proyecto.
```

Veredicto: **PASA**, con la salvedad de que su
criterio es mas debil que el del proyecto.

## Resultado 2: escaneo byte a byte independiente

Ficheros revisados: `34`.
Bytes > 127 encontrados: `0`.
Ficheros con algun byte > 127: `0`.

Veredicto: **PASA**. Este es el criterio que
aplica la regla del proyecto.

### Ficheros revisados

| fichero | bytes | bytes > 127 |
|---|---:|---:|
| `CHAMPION.md` | 29375 | 0 |
| `DECISIONS.md` | 386951 | 0 |
| `FINAL_REPORT.md` | 169683 | 0 |
| `PROJECT_STATE.md` | 101718 | 0 |
| `RUNLOG.md` | 181417 | 0 |
| `docs/DO_NOT_RERUN_UNGUARDED.md` | 685 | 0 |
| `experiments/p67_00_reconciliacion.py` | 24467 | 0 |
| `experiments/p67_01_controles_decisivos.py` | 29178 | 0 |
| `experiments/p67_02_cerrar.py` | 51119 | 0 |
| `experiments/p67_03_cierre_documental.py` | 60833 | 0 |
| `reports/TOXIC_COMPLETE_RESEARCH_REPORT.md` | 34127 | 0 |
| `reports/ascii_compliance_dual.csv` | 1740 | 0 |
| `reports/ascii_compliance_dual.md` | 4005 | 0 |
| `reports/p66_artifact_damage.md` | 2861 | 0 |
| `reports/p67_attribution_controls.csv` | 2058 | 0 |
| `reports/p67_closeout.json` | 6584 | 0 |
| `reports/p67_compliance.md` | 5901 | 0 |
| `reports/p67_controles_decisivos.json` | 6120 | 0 |
| `reports/p67_controles_decisivos.md` | 4281 | 0 |
| `reports/p67_error_intersection.csv` | 323 | 0 |
| `reports/p67_final_research.md` | 6562 | 0 |
| `reports/p67_final_summary.json` | 4050 | 0 |
| `reports/p67_frozen_evidence.csv` | 2423 | 0 |
| `reports/p67_frozen_evidence.json` | 4132 | 0 |
| `reports/p67_frozen_evidence.md` | 8364 | 0 |
| `reports/p67_interpretation.md` | 6992 | 0 |
| `reports/p67_master_results.csv` | 3743 | 0 |
| `reports/p67_p66_reconciliation.csv` | 3140 | 0 |
| `reports/p67_p66_reconciliation.json` | 8844 | 0 |
| `reports/p67_p66_reconciliation.md` | 6160 | 0 |
| `reports/p67_reproducibility.md` | 4296 | 0 |
| `reports/p67_residual_analysis.csv` | 1932 | 0 |
| `reports/p67_score_corrections.csv` | 152 | 0 |
| `reports/p67_training_variability.csv` | 223 | 0 |

## El registro canonico y su deuda historica

`experiments.csv` tiene `300` bytes > 127 repartidos en `91` lineas,
de fases **anteriores** a P67 y presentes ya en el commit base.

Bytes > 127 que caen sobre las lineas escritas por P67: `0`.

**No se corrigen.** Reescribir filas historicas del registro cambiaria la historia de
las fases anteriores para ganar una propiedad de texto que no altera ningun resultado.
Se declara como deuda conocida.

## La regla aplicada a los datos originales

La regla se aplica a **artefactos generados**. El texto de los tweets del dataset es
dato original y **no se modifica** para satisfacerla: contiene acentos, emojis y
caracteres no ASCII que son la materia prima del problema, no un defecto del pipeline.

*Generado por `experiments/p67_03_cierre_documental.py`. Sin push.*