# V23-01: auditoria competitiva de svd:rbf_svc

Generado por `experiments/v23_01_auditoria_rbf_svc.py` (V23-01 (auditoria competitiva de svd:rbf_svc y su calibracion Platt)).

**Ninguna submission. Test no utilizado.**

## 1. El mecanismo

SVC con kernel RBF sobre la compresion TruncatedSVD(300) de la union TF-IDF de V03. Es la unica diferencia algoritmica real no cerrada: V03 es lineal sobre matriz dispersa de 27.431 features, esto es no lineal sobre 300 componentes densas, con regularizacion implicita por kernel en vez de por norma L2.

Sin leakage: el TruncatedSVD se ajusta dentro de cada pliegue con fit_transform sobre el entrenamiento y transform sobre la validacion; el SVC se ajusta sobre el bloque reducido de entrenamiento.

## 2. Control de cordura

La variante con sigmoidal reproduce **0.790410** frente a los **0.790410** publicados (desviacion 0.000000), dentro de tolerancia: **True**.

## 3. Las dos calibraciones

| configuracion | corrected_cv | delta V03 | @0.5 | celdas | rescates | danos | **neto** | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| svd_rbf_sigmoid | 0.790410 | +0.013198 | 0.779089 | 240 | 136 | 104 | **+32** | 0.0452 |
| svd_rbf_platt | 0.785958 | +0.008747 | 0.777670 | 221 | 131 | 90 | **+41** | 0.0070 |

## 4. Desglose por etiqueta

| configuracion | etiqueta | cambios | rescates | danos | neto | tasa V03 -> candidata |
|---|---|---:|---:|---:|---:|---|
| svd_rbf_sigmoid | HS | 97 | 55 | 42 | +13 | 0.4572 -> 0.4808 (+0.0236) |
| svd_rbf_sigmoid | TR | 48 | 29 | 19 | +10 | 0.3001 -> 0.3041 (+0.0040) |
| svd_rbf_sigmoid | AG | 95 | 52 | 43 | +9 | 0.3776 -> 0.3891 (+0.0115) |
| svd_rbf_platt | HS | 92 | 52 | 40 | +12 | 0.4572 -> 0.4316 (-0.0256) |
| svd_rbf_platt | TR | 50 | 29 | 21 | +8 | 0.3001 -> 0.3082 (+0.0081) |
| svd_rbf_platt | AG | 79 | 50 | 29 | +21 | 0.3776 -> 0.3581 (-0.0196) |

## 5. Veredicto: CANDIDATA_CON_EVIDENCIA

Completado en 34.5 s.

