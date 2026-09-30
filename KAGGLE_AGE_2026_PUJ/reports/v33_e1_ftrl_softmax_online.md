# V33-E1 - Ejecucion: FTRL-Proximal softmax multiclase de una pasada online

- Fecha de ejecucion: 2026-09-30
- Pre-registro congelado: `experiments/v33_00_preregistro.json` (commit 7e59b06) y reports/v33_preregistro.md
- Puerta de convergencia (normativa): reports/v33_convergence_gate.md
- Commit de ejecucion: `57f4c4e70c435c6e93f884b384d8977cc0de98d3`
- Baseline primario recomputado: 0.6159732 (referencia nested 0.6154643)
- Ancla trivial recalculada: 0.04854212180687087 (congelada 0.04854212180687087, diferencia 0.000e+00)
- 32 hilos fijos; TF-IDF ajustada por fold; UNA pasada por fold y por brazo; alpha efectivo derivado por fold; RSS pico 0.71 GB (tope 20 GB)
- Compliance AST: 94 literales de ruta y 28 modulos revisados, 0 violaciones

## Diagnostico que motiva el diseno (D-083)

- D-083 congelo FTRL-Proximal **OvR** con alpha 0.05, lambda2 1.0 y **10 pasadas**: el log_loss medio de train fue 11.721967 en la epoca 1 y 23.025869 en la 10, las probabilidades quedaron planas (~0.408 por clase tras normalizar L2) y el macro-F1 standalone fue 0.040169, POR DEBAJO de la linea base trivial 0.048542. La implementacion se habia verificado exacta (tres implementaciones coinciden a 6.7e-15 y el punto fijo analitico da 0.0), luego el fallo era de CONFIGURACION: con alpha 0.05 el termino de correccion `sigma` queda inflado y diez pasadas acumulan `z` y `n` en un regimen para el que McMahan calibro alpha con T del orden 1e6-1e9.
- La hipotesis de que solo alpha tiene palanca queda registrada como HIPOTESIS del arquitecto, no como evidencia: por eso la sonda `|sigma*w|/|g|` es OBLIGATORIA y publica la razon por coordenada de la correccion frente al gradiente.
- En este diseno alpha pasa a DERIVADO (`3/(2*sqrt(T))`, con T = n_train_fold calculado en runtime, cero valores evaluados), la descomposicion pasa a softmax multiclase con un solo juego de pesos y las 10 pasadas pasan a UNA.

## alpha efectivo por fold (DERIVADO en runtime, no buscado)

| fold | T | alpha efectivo | ll_train 25% | ll_train 50% | ll_train 100% |
|---|---|---|---|---|---|
| 0 | 135468 | 0.0040754249463367 | 1.8051 | 1.8049 | 1.8048 |

Nota: `3/(2*sqrt(T))` con T = 135468 da alpha = 0.0040754249463367, exactamente el ancla congelada del pre-registro; el runner lo verifica en runtime y sale con codigo 2 si no reproduce.

## Sondas obligatorias (a)-(f), ejecutadas antes de cualquier ajuste

| Sonda | Valor | Tolerancia | pass |
|---|---|---|---|
| (a) vectorizado vs lazo | max delta = 7.772e-16 | 1e-12 | True |
| (b) raiz analitica lambda 0 | residuo raiz = 0.000e+00, un paso = 0.000e+00 | 1e-09 | True |
| (c) TotalF1 average=Macro | delta vs macro-F1 = 0.000e+00 | 1e-06 | True |
| (d) sin NaN y log_loss finito | log_loss multiclase = 8.9772 | finito | True |
| (e) log_loss propio vs sklearn | delta = 0.000e+00 | 1e-09 | True |
| (f) filas que suman 1 | peor max|sum-1| = 4.441e-16 | 1e-09 | True |

## Puerta de convergencia G1-G6 (fail-fast en el fold 0, ANTES de integrar)

- Momento: dentro del fold 0, ANTES de los folds 1-4 y ANTES de integrar.
- Magnitud: log_loss multiclase de train en el 25/50/100 % de la pasada, medido como perdida de train del modelo ajustado a ese prefijo del orden congelado (misma magnitud en A1 y A2); en A1 se publica ademas la media online del stream, que es la lectura directa de la firma de D-083.
- consumido-diagnostico: G3 y G4 leen el fold-val del fold 0, que es parte de la validacion canonica; su fallo NO toca delta.
- Resultado: pasa = False.

| Gate | Criterio | A1 valor | A1 pass | A2 valor | A2 pass |
|---|---|---|---|---|---|
| G1_ll_train_banda | 0.05 <= ll_train(100%) <= 1.6125835223052496 | 1.8047797299502006 | False | 1.38996114808309 | True |
| G2_monotonia | ll_train no creciente en 25/50/100 % (prefijo del orden congelado) | - | True | - | True |
| G3_macro_f1_fold_val | macro-F1 standalone en el fold-val >= 0.3 | 0.040974177687717 | False | 0.395624562468643 | True |
| G4_frecuencia_por_clase | frecuencia predicha por clase en [0.1, 0.3] | - | False | - | False |
| G5_determinismo | max|delta p| = 0.0 exacto entre dos corridas identicas | None | None | None | None |
| G6_suma_unidad | max|sum_c p_c - 1| <= 1e-09 | 4.440892098500626e-16 | True | 4.440892098500626e-16 | True |

- El fold 0 queda registrado como **consumido-diagnostico**: G3 y G4 leen su fold-val, que es parte de la validacion canonica, y su fallo **no toca delta**.
- Consecuencia si falla: veredicto NO_ATRIBUIBLE_BY_GATE: sin fallback, sin segundo alpha, sin integrar, sin bootstrap; se documenta y termina.
- Prohibido por el pre-registro: reintentar con otro alpha tras el fallo (seria grid).

## Sonda de atribucion OBLIGATORIA |sigma*w| / |g| (fold 0)

- Descripcion congelada: razon media |sigma*w| / |g| por coordenada en el fold 0.
- Motivo: el control cambia dos variables (alpha y regularizacion del intercepto): la razon sigma/g es la que desambigua la causa (correccion inflada frente a multi-pasada).
- Consecuencia si no se publica: el veredicto queda capado a MARGINAL.

- Razon media por coordenada: 0.04195162985558543 (mediana 0.03801326416271605, p90 0.06327813993473001, max 0.21537883066885719).
- Razon global: 0.03768882203436883 | coordenada del sesgo: 0.0038350400065826885 | fraccion de coordenadas con razon > 1: 0.0.
- Razon por clase: [0.04482900127182571, 0.03721424870853629, 0.02915997356655026, 0.03158886680190751, 0.03977901051509745, 0.04340509714124735].
- Coordenadas tocadas: 118410 de 118410.

## Entrenamiento por fold (el fold-val solo se uso para el OOF, G3 y G4)

| Brazo | fold | n_feat | vocab | T | alpha | ll_train 25/50/100 | media online | macro-F1 standalone | suma-1 | tiempo (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| A1_ftrl_softmax | 0 | 118410 | 118409 | 135468 | 0.0040754249463367 | 1.8051/1.8049/1.8048 | 1.7944/1.7961/1.7985 | 0.0410 | 4.441e-16 | 16.1 |
| A2_sgd_1p | 0 | 118410 | 118409 | 135468 | 1e-06 | 3.3924/1.7032/1.3900 | - | 0.3956 | 4.441e-16 | 0.5 |

- La magnitud `ll_train` es la perdida de train del modelo ajustado **a ese prefijo** del orden congelado: es la misma en A1 y en A2, de modo que la puerta se aplica con un criterio unico. El modelo que produce el OOF es SIEMPRE el de la pasada completa; las corridas de prefijo son instrumentacion, no modelos.
- `media online` solo existe en A1 (es el running mean del stream, la lectura directa de la firma 11.72 -> 23.03 de D-083). A2 no puede exponerla porque `SGDClassifier` no ofrece un callback por ejemplo: se reporta como '-' y su asimetria queda declarada.

## Vocabulario e idf por fold (sha256, primeros 16) y sonda de refit

| fold | n_vocabulario | vocab_sha256 | idf_sha256 | refit reproduce |
|---|---|---|---|---|
| 0 | 118409 | `f7d4a7efa7669d59` | `ec9b876fc50d30da` | None |

- Sonda de refit: reajustar el vectorizador en los 5 folds y comprobar que los sha256 de vocabulario e idf coinciden con los de la corrida; tolerancia: identicos bit a bit entre las dos corridas; reproducido en los 5 folds = None (exigido por kaggle-specialist).

## Sondas de determinismo (reentreno del fold 0 desde cero, config identica)

- A1_ftrl_softmax: no medido (el cierre por puerta precede a las sondas de determinismo).
- A2_sgd_1p: no medido (el cierre por puerta precede a las sondas de determinismo).

## Asimetria del control (declarada, no corregida)

- `alpha 1e-6 frente a 0.004 (cuatro ordenes de magnitud) y laregularidad del intercepto: por eso la sonda |sigma*w|/|g| es OBLIGATORIA`
- `alpha` del control: 1e-06 frente al alpha DERIVADO de A1 (~0.004075), cuatro ordenes de magnitud; y el intercepto de `SGDClassifier` no se regulariza, mientras que en A1 el sesgo es una coordenada mas de FTRL con su propio `n` y `z`.
- Las filas se le entregan al control YA EN EL ORDEN CONGELADO y con `shuffle=False` (enmenda 2 pre-ejecucion), de modo que consume la MISMA permutacion que A1; el `random_state` congelado `20261010+fold` se mantiene aunque con `shuffle=False` no altera el resultado (verificado: delta P = 0.0).
- **Enmiendas pre-ejecucion registradas** (sin resultados): (1) banda de G1 de `[0.25, 0.60]` a `[0.05, 0.9*log(6)]`, porque el techo original era inalcanzable para un modelo sano (D-083 midio log_loss multiclase 1.3568) y cerraba el experimento por construccion; (2) `shuffle=False` en el control para respetar el orden congelado.

## Integracion, bootstrap y gates de delta: NO EJECUTADOS

La puerta G1-G6 fallo en el fold 0: por el pre-registro NO se integra, NO se hace bootstrap, NO hay segundo alpha y NO hay fallback. Los OOF quedan parciales diagnosticos y no se tocan los artefactos congelados del campeon.

## Novedad registrada

- Que es: FTRL-Proximal softmax multiclase propio, semantica online de una pasada, un solo juego de pesos, sobre la TF-IDF por fold.
- Lo que no es: no es un embedding, no es un vector aprendido, no es una representacion nueva y no usa datos externos.
- Diferencia con D-083: D-083 fue OvR con 10 pasadas proximales y alpha 0.05; aqui softmax, 1 pasada y alpha derivado.
- Cruces explicitos: no es embedding; no es representacion nueva; no usa datos externos. Registro en artefactos NUEVOs (registro de novedad y fila anexada en el historial).
- Via B descartada: B1 ILEGAL (usa features de test sin etiqueta para modificar pre; B2 descartada por analogia con C04 (bagging; B3 refutada por algebra: con tronco lineal .

## Auditorias y asimetrias tecnicas

- **D-083 data_hash del manifiesto**: el data_hash del manifiesto canonico (2ada573b...) queda invalidado en D-083: es un hash de punteros de memoria, no de contenido. Se congilan y verifican el sha256 del archivo del corpus, el sha256 del array de textos con la receta determinista (longitud de 8 bytes LE + bytes UTF-8 por texto, en orden) y el sha256 del dict del vectorizador. No se reordena nada para hacer cuadrar un hash.
- **Receta de hash de textos**: sha256 con longitud (8 bytes LE) + bytes UTF-8 por texto, en orden.
- **ngram_range**: la config congelada lo declara como lista y sklearn 1.8.0 exige tupla; en el fit se convierte a tupla (mismo valor) y el sha256 congelado del dict se calcula SIEMPRE sobre la CONFIG con la lista.
- **Orden de datos**: una sola permutacion por fold con `np.random.default_rng(20261010+fold)`, congelada y reutilizada por los dos brazos; la curva online de A1 se registra con un callback que no altera la aritmetica.
- **log_loss**: A1 y A2 entregan probabilidades que suman 1 por construccion (softmax de 6 logits y `predict_proba`), que es la diferencia frente al OvR de D-083; la sonda (e) lo verifica contra sklearn y la (f) verifica la suma de las filas.
- **Escritura atomica**: temporal + `os.replace` en el OOF y en el checkpoint.
- **Familias vetadas**: la auditoria AST revisa este mismo archivo, importes Y literales de ruta, contra las diez familias vetadas y las cuatro particiones reservadas. La libreria de bandits instalada en el venv se descarta y no se importa: el FTRL-Proximal es implementacion propia verificada por las sondas (a), (b) y (e).

## Veredicto

**NO_ATRIBUIBLE_BY_GATE**: veredicto NO_ATRIBUIBLE_BY_GATE: sin fallback, sin segundo alpha, sin integrar, sin bootstrap; se documenta y termina. la puerta G1-G6 falla en A1 o A2: STOP documentado, sin fallback, sin integrar (nodo nuevo, no un gate de delta)

## Compliance

- 0 embeddings, 0 transformers, 0 LLM, 0 datos externos; sin particiones reservadas leidas ni escritas; el campeon V30 + plug-in permanece intacto.
- Un experimento = un script nuevo; no se edito ningun script ya ejecutado ni el acta de D-083.
- La segunda particion (semilla 987654) NO se ejecuta: la autoriza el usuario y el runner solo la declara en `siguiente_paso`.
