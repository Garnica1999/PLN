# V22-02 (P1, P2, P3): RAW frente al pipeline historico

Generado por `experiments/v22_02_preprocessing_historic_vs_raw.py` (V22-02 (P1 control, P2 y P3: RAW frente al pipeline historico)).

**Ningun modelo entrenado. Ninguna submission.**

## 1. Control de cordura de P1

El pipeline historico da **0.781766** frente a los **0.781766** publicados (desviacion 0.000000), dentro de tolerancia: **True**.

## 2. Cobertura de cada brazo, con el MISMO diccionario

| brazo | tokens | tipos | cob. tipo | cob. token | long. media | filas nulas |
|---|---:|---:|---:|---:|---:|---:|
| HISTORICO | 25539 | 5447 | 1.0000 | 1.0000 | 17.22 | 0 |
| CRUDO | 26996 | 8080 | 0.5501 | 0.7414 | 18.20 | 68 |

## 3. La comparacion

| brazo | corrected_cv | delta V03 | emb solo | celdas | rescates | danos | neto |
|---|---:|---:|---:|---:|---:|---:|---:|
| HISTORICO | 0.781766 | +0.004555 | 0.666085 | 136 | 73 | 63 | **+10** |
| CRUDO | 0.768895 | -0.008316 | 0.640933 | 133 | 50 | 83 | **-33** |

**HISTORICO - CRUDO = +0.012871**

## 4. Veredicto de P3: HIPOTESIS_DE_PREPROCESSING_REFUTADA

El pipeline historico ya era el correcto. La Hipotesis 1 de V22 afirmaba que el embedding se usaba sobre un texto de tarea incompatible con el del corpus, pero V5.1 ya aplicaba normalizar_emb_normal al texto de la tarea, y apartarse de esa normalizacion empeora la metrica en vez de mejorarla. No habia un mismatch que corregir: habia una decision de diseno ya tomada correctamente

Completado en 5.2 s.

