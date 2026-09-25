# Compliance de la fase v5 «Embedding from Scratch»

Generado por `experiments/embedding_00_scaffold_v5.py` el 2026-09-25. Decision de gobierno: **D-030**.

## 1. Autorizacion registrada

- **Alcance**: corpus externos **no etiquetados**, exclusivamente para aprender la representacion distribuida desde cero.
- **Prohibido**: entrenamiento supervisado, clasificacion, generacion de labels, pseudo-labeling, ajuste de thresholds y cualquier otra forma de aprendizaje sobre HS/TR/AG.
- **Evidencia documental**: no existe correo, foro ni captura archivada que documente la autorizacion; se registra unicamente lo comunicado por el usuario. No se inventa evidencia.
- **Ante ambiguedad sobre una fuente**: se detiene esa fuente y se registra la duda.

## 2. Entorno de ejecucion

| Recurso | Valor |
|---|---|
| Nucleos disponibles | 32 |
| RAM total / disponible | 31.2 GB / 18.0 GB |
| Disco libre | 450.2 GB |
| Python | 3.14.7 (/home/carlos/venv/bin/python) |
| fastText C++ | commit `1142dc4c4ecb` (MIT) |
| sha256 del binario | `0307cde9a9e9dc02f2b688e4645ffbb3` |

La build es local: la libreria implementa el algoritmo y **los pesos se entrenan desde cero** con nuestro corpus. No se descarga ningun modelo ni checkpoint. El control `WORD2VEC-CONTROL / FASTTEXT-SG-SUBWORDS-OFF` usa el mismo binario con `minn=0, maxn=0`.

## 3. Compliance de recursos (analisis estatico)

| Recurso prohibido | Usos detectados |
|---|---|
| GloVe no implementado por defecto (spec 19) | 0 |
| carga de pesos preentrenados | 0 |
| descarga de datos de la competencia | 0 |
| modelo derivado de transformers | 0 |
| sentence-transformers | 0 |
| transformers | 0 |
| vectores preentrenados de spaCy | 0 |

Scripts de la fase auditados: **0**. Veredicto: **CUMPLIDA**.

## 4. Uso de los datos de TOXIC

| Script | Lee test | Lee labels | Lee textos de train | Uso |
|---|---|---|---|---|
| embedding_00_scaffold_v5.py | si | si | si | EXTERNAL-ONLY salvo adaptacion in-fold registrada |

Regla aplicada: **texto usado para el embedding** != **labels usadas para supervision**. La variante principal es `EXTERNAL-ONLY`; `toxic_test.csv` no entra en ningun corpus; el holdout esta congelado y agotado (5/5).

## 5. Integridad de la referencia

- corrected_cv de V03 recalculada desde el OOF canonico: **0.777211** (esperada 0.777211; coincide: True).
- LB de referencia: **0.78306**; umbrales 0.44/0.51/0.46.
- Particiones: train 1731, dev 1483, holdout 248 (congelado, no se usa en v5).

| Archivo oficial | sha256 |
|---|---|
| `toxic_train.csv` | `3d40e7345b588860cf4744914be88ddc` |
| `toxic_test.csv` | `e91ff8258dadbbcd57593273ff1babd8` |
| `submission.csv` | `0dfb4d07dffc31d3d452f5e53508d259` |

## 6. Fuentes habilitadas (D-030)

| Fuente | Rol | Acceso | Licencia | Estado |
|---|---|---|---|---|
| wikipedia_es | general | publico, sin registro | CC BY-SA 4.0 (atribucion obligatoria) | verificada_disponible |
| m_c4_es | web | publico, sin gating (verificado el 2026-09-25) | ODC-BY (Common Crawl); se documenta con el corpus | verificada_disponible |
| oscar_es | web | gated manual: requiere cuenta y aceptacion de terminos | CC BY 4.0 (segun ficha del dataset) | no_utilizable_sin_acceso_autorizado |
| hplt_es | web | publico, sin gating (verificado el 2026-09-25) | a verificar antes de usar (no asumida) | candidata_requiere_evaluacion_previa |
| cereal_es | web | no localizada en la auditoria inicial (404) | a verificar | no_localizada |

OSCAR queda **descartada por ahora** (gated manual): no se retrasa la investigacion ni se aplican mecanismos para saltar el gating. HPLT y CEREAL son candidatas que requieren evaluacion previa de procedencia, licencia y solapamiento.

## 7. Escala autorizada

10M → 50M → 100M → decidir → 250M → decidir → 500M → decidir. Ninguna expansion se ejecuta sin `RATIONALE_FOR_CORPUS_EXPANSION` previo.

## Veredicto

**GO para P1 y P2**: fastText C++ compila y entrena desde cero, la referencia V03 se reproduce exactamente, no hay uso de recursos prohibidos y el manifiesto de corpus documenta procedencia, licencia y estado de cada fuente.
