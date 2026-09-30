# V32-E3 (P3) - Ejecucion: challenger FTRL-Proximal sobre TF-IDF por fold

- Fecha de ejecucion: 2026-09-30
- Pre-registro: reports/v32_p3_preregistro.md (commit 15c2be8) y `experiments/v32_04_preregistro_ftrl.json`
- Commit de ejecucion: `ee90975df941d41a84188e44ed06ef0395581261`
- Baseline primario recomputado: 0.6159732 (referencia nested 0.6154643)
- 32 hilos fijos; TF-IDF ajustada por fold; FTRL 10 pasadas fijas; alpha efectivo 0.005; RSS pico 0.87 GB (tope 20 GB)
- Compliance AST: 52 literales de ruta y 28 modulos importados revisados, 0 violaciones

## Sondas obligatorias (a)-(d), ejecutadas antes de cualquier ajuste

| Sonda | Valor | Tolerancia | pass |
|---|---|---|---|
| (a) vectorizado vs lazo | max delta = 6.661e-15 | 1e-12 | True |
| (b) raiz analitica | residuo raiz = 0.000e+00, un paso = 0.000e+00 | 1e-09 | True |
| (c) TotalF1 average=Macro | delta vs macro-F1 = 0.000e+00 | 1e-06 | True |
| (d) sin NaN y log_loss finito | log_loss OvR = 8.1029 | finito | True |

## Entrenamiento por fold (el fold-val solo se uso para el OOF y como reporte)

| Brazo | fold | n_feat | vocab | ll train ep1 | ll train ep10 | ll val | macro-F1 crudo val | alpha | tiempo (s) |
|---|---|---|---|---|---|---|---|---|---|
| A1_ftrl | 0 | 118410 | 118409 | 1.2833 | 3.6246 | 3.6530 | 0.0397 | 0.005 | 57.6 |
| A1_ftrl | 1 | 118338 | 118337 | 1.2831 | 3.6244 | 3.6542 | 0.0417 | 0.005 | 58.3 |
| A1_ftrl | 2 | 118233 | 118232 | 1.2830 | 3.6245 | 3.6551 | 0.0399 | 0.005 | 58.2 |
| A1_ftrl | 3 | 118357 | 118356 | 1.2833 | 3.6253 | 3.6560 | 0.0409 | 0.005 | 58.7 |
| A1_ftrl | 4 | 118364 | 118363 | 1.2829 | 3.6238 | 3.6517 | 0.0383 | 0.005 | 58.5 |
| A2_sgd_twin | 0 | 118410 | 118409 | 0.1711 | 0.1711 | 0.3511 | 0.4956 | - | 1.6 |
| A2_sgd_twin | 1 | 118338 | 118337 | 0.1738 | 0.1738 | 0.3567 | 0.4940 | - | 1.5 |
| A2_sgd_twin | 2 | 118233 | 118232 | 0.1724 | 0.1724 | 0.3539 | 0.4982 | - | 1.6 |
| A2_sgd_twin | 3 | 118357 | 118356 | 0.1785 | 0.1785 | 0.3599 | 0.4892 | - | 1.6 |
| A2_sgd_twin | 4 | 118364 | 118363 | 0.1733 | 0.1733 | 0.3611 | 0.4879 | - | 1.6 |
| A3_ftrl_barajado | 0 | 118410 | 118409 | 1.2825 | 3.6229 | 3.6512 | 0.0463 | 0.005 | 57.6 |
| A3_ftrl_barajado | 1 | 118338 | 118337 | 1.2824 | 3.6227 | 3.6525 | 0.0461 | 0.005 | 57.8 |
| A3_ftrl_barajado | 2 | 118233 | 118232 | 1.2823 | 3.6227 | 3.6534 | 0.0462 | 0.005 | 58.5 |
| A3_ftrl_barajado | 3 | 118357 | 118356 | 1.2826 | 3.6236 | 3.6542 | 0.0462 | 0.005 | 58.8 |
| A3_ftrl_barajado | 4 | 118364 | 118363 | 1.2822 | 3.6221 | 3.6500 | 0.0467 | 0.005 | 58.0 |

Nota: en A2 (gemelo SGD) la tabla muestra UN solo punto de log-loss por fold (las columnas ep1 y ep10 repiten el mismo valor) porque `SGDClassifier` corre sus 10 pasadas internas sin callback por epoca; `n_iter_` = [10] en los 5 folds, el mismo presupuesto de pasadas que A1. A1 y A3 si tienen curva de 10 puntos por epoca.

## Vocabulario e idf por fold (sha256, primeros 16)

| fold | n_vocabulario | vocab_sha256 | idf_sha256 | tiempo fit (s) |
|---|---|---|---|---|
| 0 | 118409 | `f7d4a7efa7669d59` | `ec9b876fc50d30da` | 2.9 |
| 1 | 118337 | `dae18845defd8277` | `7b7dd9e81ded5836` | 2.9 |
| 2 | 118232 | `c8970e0849516420` | `52768cc065aace56` | 2.9 |
| 3 | 118356 | `58db146ee417eda2` | `1fffe5cd32d29249` | 3.0 |
| 4 | 118363 | `28c0a2ff12d328c2` | `6e89ced0089e2c74` | 3.2 |

## Regla de fallback (congelada antes de ejecutar)

- Criterio: en el fold 0 de A1, sobre las filas de fold-train: log_loss OvR medio tras 10 pasadas <= 0.8 * log_loss tras la pasada 1, y sin NaN/inf
- Fold 0 de A1: log_loss train epoca 1 = 11.7220, epoca 10 = 23.0259 (umbral 0.8 x ep1 = 9.3776)
- Cumple: False | config_fallback_usada: True | alpha efectivo: 0.005 | repasos: ['A1_ftrl/0']
- La decision se tomo con las filas de fold-train; jamas con el fold-val ni con el ranking.

## Sondas de determinismo (reentreno del fold 0 con config identica)

- A1_ftrl: max delta p = 0.000e+00 (exacto), rango = None, no concluyente = False, refit de vocabulario e idf identico = True
- A2_sgd_twin: max delta p = 0.000e+00 (exacto), rango = None, no concluyente = False, refit de vocabulario e idf identico = True

## Integracion P6/P7 (combinador congelado, regla argmax(p_comb * w))

| Brazo | macro-F1 comb | delta vs baseline | folds+ | IC95 | delta sin regla | delta vs nested | log_loss p_comb | sha256 OOF (16) |
|---|---|---|---|---|---|---|---|---|
| A1_ftrl | 0.6147759 | -0.0011973 | 1/5 | [-0.0019881, -0.0004227] | -0.0029139 | -0.0006884 | 1.0355 | `d7245bef8192b2d5` |
| A2_sgd_twin | 0.6142730 | -0.0017002 | 0/5 | [-0.0024742, -0.0008943] | -0.0033893 | -0.0011913 | 1.0332 | `98c378949763f5a0` |
| A3_ftrl_barajado | 0.6147759 | -0.0011973 | 1/5 | [-0.0019881, -0.0004227] | -0.0029139 | -0.0006884 | 1.0355 | `3db25c31bc3a21f7` |

## Complementariedad standalone (argmax del OOF normalizado L2, sin pesos)

| Brazo | macro-F1 solo | diagonal 6x6 | log_loss L2 | rescate | dano | neto | rescate alta conf. | errores alta conf. campeon |
|---|---|---|---|---|---|---|---|---|
| A1_ftrl | 0.0401692 | False | 1.7921 | 6467 | 100863 | -94396 | 900 | 9780 |
| A2_sgd_twin | 0.4934789 | True | 1.3568 | 11834 | 32468 | -20634 | 1405 | 9780 |
| A3_ftrl_barajado | 0.0463068 | False | 1.7918 | 7564 | 84085 | -76521 | 951 | 9780 |

- OOF crudo de A2 (diagnostico): 0.4934789 de macro-F1 con argmax crudo, log_loss OvR 0.3565, suma media por fila 0.9776 y 2 filas que suman 1: no suma 1 por construccion (seis sigmoides independientes).
- Artefacto: `artifacts/p33_scores/v32_p3/a2_sgd_twin_crudo.npy` (sha256 `7df44e114097e5e4`). diagnostico exclusivo: no entra en la integracion, en los gates, en el arbol ni en los OOF congelados de los brazos.
- La normalizacion L2 escala cada fila por una constante positiva: el argmax (y el macro-F1 standalone) es identico al del OOF crudo.

## Gates

| Gate | Resultado |
|---|---|
| baseline_ok | True |
| delta_A1_suficiente | False |
| folds_positivos_A1_suficiente | False |
| control_favorable_A1_sobre_A2 | False |
| ic95_A1_sin_cero | True |
| a3_coherente | True |
| sin_leakage | True |
| determinismo_concluyente | True |

## Caps y controles

- Bootstrap pareado de la diferencia (F1_combA1 - F1_combA2): delta medio +0.0004978, IC95 [-0.0000835, +0.0010866], favorable = False
- delta_int(A3) = -0.0011973 vs delta_int(A1) = -0.0011973: NO_ATRIBUIBLE
- Gate A3: macro-F1 standalone 0.0463068 vs linea base de la clase mayoritaria 0.0485421 (13-17): coherente = True
- Cap por IC95 de la diferencia: True; cap por determinismo: False
- Calibracion: t_fold0 = 57.6 s, proyeccion 9 x = 0.14 h (limite 20.0 h)

## Veredicto

**NO_ATRIBUIBLE**: si delta_int(A3) >= delta_int(A1) (precedencia)

## Asimetrias declaradas y auditorias

- **Sesgo**: en A1 el sesgo es una coordenada mas de FTRL, con su propio acumulador y su propia penalizacion lambda2; en A2 el intercepto de sklearn no se regulariza. La asimetria la fija el pre-registro y se reporta, no se corrige.
- **A2 se aparta de E-011**: E-011 usaba max_iter=30 y tol=1e-4; aqui 10/tol=None para igualar el presupuesto de pasadas con A1 (desvio declarado, no fallo).
- **Cap por IC95**: si el IC95 de la diferencia contiene 0, el veredicto queda capado a MARGINAL aunque A1 pase el resto de gates (decision del lead, registrada; puede degradar un resultado genuinamente bueno).
- **Cap por determinismo**: sonda no concluyente -> el veredicto no puede ser PROMISING.
- **Auditoria del data_hash del manifiesto**: el manifiesto lo calculo con to_numpy().tobytes() sobre un array de OBJETOS: serializa punteros de memoria, no el texto; no es reproducible entre procesos (se observaron 68b9a0f2 y 40342f7c). Decision: no se reordena nada; se congelan sha256 del archivo + sha256 del array de textos con receta determinista + la receta. Los tres hashes congelados (archivo, array de textos con receta determinista y dict del vectorizador) si se reproducen; no se reordeno nada.
- **ngram_range**: la config congelada lo declara como lista y sklearn 1.8.0 exige tupla; en el fit se convierte a tupla (mismo valor) y el hash del dict congelado usa la lista.
- **A2 sin curva por epoca**: SGDClassifier corre sus 10 pasadas internas sin callback; se reporta el log_loss final por fold (figura, panel 2).
- **log_loss de la tabla de complementariedad**: el OOF normalizado L2 no suma 1 por fila y en sklearn 1.8.0 `log_loss` solo avisa, no renormaliza; por eso la metrica se calcula sobre el OOF renormalizado con la convencion del proyecto (clip [1e-12, 1] y L1 por fila), que es la misma de `combinador_oof`. Sin ese paso la perdida se subestimaba.
- **p OvR**: FTRL entrega seis sigmoides que no suman 1; el comparador canonico es la normalizacion L2 por fila y la magnitud de la regla de fallback es el log_loss OvR medio.

## Compliance

- 0 embeddings, 0 transformers, 0 LLM, 0 datos externos; ninguna particion reservada se lee ni se escribe; el campeon no se toca.
- Auditoria AST del propio runner: 0 palabras vigiladas en literales de ruta y 0 imports de las familias vetadas (vowpal, vw, torch, transformers, gensim, fasttext), que se comprueban tanto en los literales de ruta como en los modulos importados. La alternativa declarada en el pre-registro (vowpalwabbit 9.11.8, instalada en el venv) no se importa ni se usa: el FTRL-Proximal es implementacion propia verificada por las sondas (a) y (b).
- Campeon V30 + plug-in intacto; el challenger es un candidato separado; sin tuning posterior.
