| Gate | Criterio | Anclaje | Tolerancia |
|---|---|---|---|
| G0 | A1: M/2 <= D_acum <= 2*M con D_acum = rango medio de logits; A2: ll_train <= H(prior)_ES - 0.01 | M = ln(5) = 1.6094379124341003; factor 2 declarado | 1e-9 |
| G1 | 0.05 <= ll_train <= H(prior)_ES - 0.01 | piso = detector de humo; techo anclado en el prior MEDIDO del ES-train (NO en ln 6 = 1.791759469228055) | 1e-6 |
| G2 | la mejor ll_train mejora en >= 1 de las primeras 3 pasadas | un modelo que no aprende tiene ll_train plana | 1e-06 |
| G3 | macro-F1 standalone en el fold-val >= 0.30 | trivial 0.0485421 (clase 13-17) y gemelo SGD de D-083 0.4934789 | 1e-9 |
| G4 | frecuencia predicha por clase en [0.5*prior_c, 1.5*prior_c] | prior MEDIDO del ES-train | 1e-9 |
| G5 | max|delta p| = 0.0 entre dos corridas identicas del fold 0 | determinismo CPU esperado exacto | 0 |
| G6 | max|sum_c p_c - 1| <= 1e-09 | softmax de 6 logits | 1e-09 |

Los seis umbrales estan **DERIVADOS**, no calibrados con datos: G1 y G4 se anclan en el prior
**medido** del ES-train (datos de train), G3 en las referencias ya congeladas de D-083 y D-084, y
G0 en el margen declarado de la propia regla. **Ninguno se calibro con las 27 celdas de la exploracion
metodologica previa.**

## Cuando corre y que pasa si falla

- Corre dentro del **fold 0**, antes de los folds 1-4 y antes de integrar, e **igual** a A1 y a A2.
- G3 y G4 leen el fold-val del fold 0, parte de la validacion canonica: el fold 0 queda registrado
  como **consumido-diagnostico** y su fallo **no toca delta**.
- Si falla cualquiera: `NO_ATRIBUIBLE_BY_GATE`, sin fallback, sin segundo alpha (reintentar seria grid),
  sin integracion y sin bootstrap, con salida 0 tras dejar JSON, estado e informe.
- Si A1 falla y A2 pasa G0-G4, el nodo candidato es `LINEA_ONLINE_CERRADA_POR_MECANISMO` (cierre de FAMILIA),
  y **solo** se concede tras triage formal registrado y con la tabla de mecanismo publicada dentro del runner.
