# V24 · Reconciliacion documental del notebook y del registry

**Estado: PASS** · no se entrena nada · no se genera ninguna submission.

## 1. La fuente de verdad del notebook

**No existe un generador unico.** La fuente de verdad es una **cadena de scripts**, cada uno de una fase, aplicados en orden sobre el mismo archivo.

| Orden | Script | Que hace | Fecha |
|---|---|---|---|
| 1 | `scripts/fase09_99_entrega.py` | crea el notebook con la Parte I | 2026-09-25 |
| 2 | `scripts/fase12_notebook_final.py` | regenera con la fase avanzada (V13) | 2026-09-25 |
| 3 | `scripts/phase28_notebook_final_v2.py` | regenera con la Parte II (V14 a V16) | 2026-09-25 |
| 4 | `scripts/v17_notebook.py` | anade la Parte III (V17) | 2026-09-26 |
| 5 | `scripts/v23_02_auditoria.py` | anade la seccion 31 (V23) | 2026-09-27 |
| **6** | **`scripts/v24_notebook_reconciliacion.py`** | **anade V18 a V22 (este)** | **2026-09-27** |

Regenerar el notebook entero con la cadena anterior **borraria** todo lo posterior a V17. La unica forma correcta de cerrar la deuda es anadir el eslabon que falta.

## 2. Causa raiz y por que tenia 103 celdas

Las fases **V18, V19, V20, V21, V22** nunca tuvieron eslabon en la cadena. Ese es el motivo exacto de que el notebook se quedara en V17 con el champion en V20, y no un error de edicion.

Las 103 celdas eran correctas **para lo que contenian**: V12 a V17 mas la seccion de V23 anadida al final. El numero no era erroneo, estaba incompleto.

Despues de la reconciliacion: **121 celdas**, 33 secciones, versiones presentes [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23].

## 3. Que faltaba y como se corrige

| Version | Integrada | Artefactos enlazados |
|---|---|---|
| V12 | si | `reports/v17_master_results.csv` |
| V13 | si | `experiments/advanced_05c_combinacion_final.json` |
| V14 | si | `experiments/v14b_fusion_linguistico.json` |
| V15 | si | `experiments/v15b_poda.json` |
| V16 | si | `experiments/v16_arquitectura.json` |
| V17 | si | `reports/v17_master_results.csv` |
| V18 | si | `experiments/v18_cierre.json` |
| V19 | si | `experiments/v19_b1_challenger.json` |
| V20 | si | `experiments/v20_f3c_candidato.json` |
| V21 | si | `experiments/v21_b1.json` |
| V22 | si | `experiments/v22_ordinal.json` |
| V23 | si | `experiments/v23_01_coral.json` |

## 4. Discrepancia documentada sin sobreescribir

- **V12**: `ORIGINAL = 0.6004` / `CORRECTED = 0.6005017473` en `experiments/v22_frontier_audit.json`.
  phase39_fusion_v12.json registra la referencia de V12 redondeada a cuatro decimales. El valor canonico pooled es 0,6005017, que usan v18_cierre.referencia_pooled y v22_frontier_audit.baselines.V12. La tabla maestra manda el canonico.

- **V20**: `ORIGINAL = 0.609615` / `CORRECTED = 0.6097782` en `experiments/v22_frontier_audit.json`.
  v22_frontier_audit.json registro 0,609615, que es la salida del MODELO COMPARTIDO de D-057 y no la cadena de V20. El valor canonico de V20 es 0,6097782, verificado por reproduccion exacta en cuatro artefactos. El artefacto historico NO se modifica: se documenta aqui la diferencia.

## 5. Registry

- Lineas de cabecera: **3 -> 1** (2 espurias eliminadas).
- Filas de experimento: **130 -> 128**.
- IDs duplicados antes: **['experiment_id']**; despues: **ninguno**.
- IDs vacios: **0**.
- Columnas: **22**.

**El conteo real es menor que el que se reportaba.** Las dos cabeceras espurias ocupaban el identificador `experiment_id` dos veces, asi que el registry declaraba 130 experimentos y tenia 128. Este script **no anade filas**: V24 es documentacion, no un experimento, asi que el numero solo puede bajar.

## 6. Coherencia numerica

| Comprobacion | Resultado |
|---|---|
| v20 canonico correcto | OK |
| v20 en la tabla maestra | OK |
| v23 en la tabla maestra | OK |
| v23 cerrado | OK |
| sin filas de v24 | OK |
| registry sin duplicados | OK |
| todas las fases en la tabla | OK |

## 7. Auditoria de Unicode

- **Introducidos por V24: 0** (requisito bloqueante).
- Preexistentes en el notebook, de sesiones anteriores: **38**. El proyecto excluye `notebooks/` de su auditoria de idioma historica, luego nunca se detectaron. No se tocan.

## 8. Auditoria de scripts y notebook

- `py_compile` rc = **0**
- `ruff check` rc = **1** | errores de correccion: **0** | avisos de estilo: **54**

El proyecto **no tiene configuracion de ruff**, luego se aplican sus defaults. Los avisos de estilo (`ISC004`, concatenacion implicita) son **sistematicos en el proyecto**: los scripts ya commiteados tienen 12. El gate de esta fase exige cero **errores de correccion** (reglas `F` y `E9`), que es lo que se verifica.
- Ejecucion real de las celdas de V24: **9 celdas, 0 errores**
- IDs de celda unicos: **True** | titulos unicos: **True** | artefactos faltantes: **0**
- Champion presente en el notebook: **True**
- Deuda documental declarada: **True**

## 9. Deuda documental restante

1. El notebook conserva 38 caracteres Unicode de sesiones anteriores.
2. `notebooks/09_final_pipeline.ipynb` es antiguo y obsoleto respecto a V20; se deja como esta porque borrarlo no pertenece a esta fase.
3. Las secciones 0 a 30 nombran fases y las nuevas nombran versiones; la seccion 32.2 aporta el mapa que las conecta, pero la nomenclatura historica no se reescribe.

## 10. Lo que esta fase NO hace

- No entrena ningun modelo, ni histrico ni nuevo.
- No genera submissions ni consulta Kaggle.
- No lee el test ni el holdout.
- No anade filas a `experiments.csv` ni cambia resultados historicos.
- No abre ninguna hipotesis: el proximo experimento se decide despues y no automaticamente.
