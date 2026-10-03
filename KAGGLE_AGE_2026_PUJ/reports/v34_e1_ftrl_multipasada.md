# V34-E1 - Ejecucion: FTRL-Proximal softmax multiclase multi-pasada con ES interno

- Fecha de ejecucion: 2026-09-30
- Pre-registro congelado: experiments/v34_00_preregistro.json (commit 3e97be707840e4fa95f53eaf62386f9307d1d6ab) y reports/v34_preregistro.md
- Puerta de convergencia (normativa): reports/v34_convergence_gate.md
- Commit de ejecucion: `817535621719a88d6f134d2249c6b0a3aa0cfb4a`
- Baseline primario recomputado: 0.6159732 (referencia nested 0.6154643)
- Ancla trivial recalculada: 0.04854212180687087 (clase 13-17)
- 32 hilos fijos; TF-IDF ajustada SOLO en el ES-train de cada fold; P_max = 10 pasadas con ES interno (P_min = 3, paciencia 2, mejora minima 0.0001); RSS pico 0.68 GB (tope 20 GB)
- Compliance AST: 26 literales de ruta y 28 modulos, 0 violaciones; lecturas declaradas: 26 literales en 115 funciones, 0 violaciones
- Control A2 (enmienda 2): partial_fit_por_pasada x 10, con el estimador construido con max_iter = 1 (NO 10).

## Veredicto: LINEA_ONLINE_CERRADA_POR_MECANISMO

la puerta G0-G6 se cerro en el fold 0 (pasa = False, brazos con gates fallidos: {'A1_ftrl_multipasada': ['G0_objetivo_de_la_regla', 'G1_banda_ll_train', 'G2_progreso', 'G3_macro_f1_fold_val', 'G4_frecuencias'], 'A2_sgd_multipasada': []}); el pre-registro prohibe integrar y hacer bootstrap: delta y bootstrap quedan VACIOS

## 1. Mecanismo: trayectoria por pasada de A1 y A2 (fold 0 consumido-diagnostico)

- alpha* = 2*M/(d_norm*P_max^2) con M = ln(5) = 1.6094379124341003 y P_max = 10 (0 valores evaluados, no es un grid); d_norm = 0.9128367962785974 del prior MEDIDO del ES-train del fold 0, luego alpha* = 0.035262336465737745.
- Banda objetivo de la regla (gate G0 de A1): D_acum en [0.804719, 3.218876]; banda de G1 sobre ll_train: [0.05, 1.781571].
- w_max_abs por pasada (la firma de D-083 era w_max_abs = 26.4 con 28.86 de ll_train): A1 1.250407e+01 con ll_train 19.174421; A2 2.465037e+01 con ll_train 0.404220.

| brazo | pasada | ll_train (ES-train) | D_acum | macro-F1 ES-val | w_max_abs | razon |sigma*w|/|g| |
|---|---|---|---|---|---|---|---|
| A1_ftrl_multipasada | 1 | 10.086314 | 12.106446 | 0.046077 | 6.606918e+00 | 1.511e-01 |
| A1_ftrl_multipasada | 2 | 15.420625 | 18.496806 | 0.046077 | 9.996246e+00 | 1.980e-01 |
| A1_ftrl_multipasada | 3 | 19.174421 | 23.194918 | 0.046077 | 1.250407e+01 | 2.196e-01 |
| A2_sgd_multipasada | 1 | 1.062199 | 9.009318 | 0.450724 | 3.685110e+01 | n/d (A2 no es FTRL) |
| A2_sgd_multipasada | 2 | 0.531503 | 6.504952 | 0.478322 | 2.897554e+01 | n/d (A2 no es FTRL) |
| A2_sgd_multipasada | 3 | 0.461367 | 5.946400 | 0.484608 | 2.640142e+01 | n/d (A2 no es FTRL) |
| A2_sgd_multipasada | 4 | 0.444882 | 5.686317 | 0.485945 | 2.563577e+01 | n/d (A2 no es FTRL) |
| A2_sgd_multipasada | 5 | 0.443319 | 5.572206 | 0.492124 | 2.525916e+01 | n/d (A2 no es FTRL) |
| A2_sgd_multipasada | 6 | 0.416137 | 5.499474 | 0.493740 | 2.509153e+01 | n/d (A2 no es FTRL) |
| A2_sgd_multipasada | 7 | 0.413570 | 5.395290 | 0.496981 | 2.482520e+01 | n/d (A2 no es FTRL) |
| A2_sgd_multipasada | 8 | 0.412464 | 5.374682 | 0.493135 | 2.474956e+01 | n/d (A2 no es FTRL) |
| A2_sgd_multipasada | 9 | 0.409993 | 5.321915 | 0.498733 | 2.472974e+01 | n/d (A2 no es FTRL) |
| A2_sgd_multipasada | 10 | 0.404220 | 5.255189 | 0.494276 | 2.465037e+01 | n/d (A2 no es FTRL) |

Figura: reports/figures/v34_e1_curvas.png.

## 2. Tabla de simetria de sesgo (enmienda pre-ejecucion 1)

- Camino de A1: coordenada de sesgo x0=1.0 materializada dentro de FTRL (propio z, propio n, sin termino de sesgo aparte).
- Camino de A2: fit_intercept=True sobre la MISMA matriz SIN columna de sesgo pre-anadida (enmienda 1: se corrige el doble sesgo de D-083/V33-E1).
- no hay doble sesgo en ningun brazo: A1 materializa el sesgo como coordenada unica y A2 lo aporta solo por intercept_; el defecto de D-083 y V33-E1 (matriz con sesgo pre-anadido Y fit_intercept=True) NO se reproduce aqui.

| brazo | caminos de sesgo | fuente | magnitud | valor |
|---|---|---|---|---|
| A1_ftrl_multipasada | 1 | coordenada de sesgo x0=1.0 dentro de FTRL (propio z, propio n) | max|gradiente| de la coordenada x0 (fold 0) | 1.000000 |
| A2_sgd_multipasada | 1 | fit_intercept=True sobre la matriz SIN columna de sesgo pre-anadida (un solo camino) | max|intercept_| (fold 0) | 3.451903 |

| fold | max|g_sesgo| (A1) | max|intercept_| (A2) | razon A1/A2 | mismo orden de magnitud |
|---|---|---|---|---|
| 0 | 1.000000 | 3.451903 | 2.897e-01 | True |

## 3. Asimetrias declaradas

- `lambda2` de A1 = 1.0 frente a `alpha_reg` de A2 = 1e-06 (cociente 1e+06). con lambda2 = 1e-6 el L2-en-acumulador y el L2-en-denominador dan resultado IDENTICO (es un no-op, medido en la exploracion previa): por eso no existe una equivalencia FTRL<->SGD a escala 1e-6
- El veredicto NO se apoya en esa asimetria: A1 frente a la DERIVACION (regla de margen y tabla de mecanismo publicada dentro del runner); A2 es el control sano que demuestra que el presupuesto no es el limite.
- Lectura correcta de A2: el gemelo no es un equivalente de FTRL a 1e-6: se lee como 'el presupuesto de 10 pasadas NO es el limite'.
- Enmienda 2 (control A2): partial_fit_por_pasada con 10 llamadas; el numero de pasadas (10) es el que el pre-registro congelo; lo que se corrige es como se ejecuta, porque fit(max_iter=1, warm_start=True) no acumula el schedule.
- Desviacion declarada: 4.5e-5 a 2 pasadas y 2.0e1 a 4 pasadas en un problema sintetico; motivo: es la unica via que permite evaluar el early stopping despues de cada pasada, que el pre-registro exige (P_min 3, paciencia 2, restaurar el mejor coef_), y da el mismo presupuesto que A1.
- Enmienda del pre-registro (control.a2_sgd.ajuste (presupuesto de 10 pasadas)): defecto medido de API: BaseSGDClassifier._fit hace self.t_ = 1.0 en cada llamada, luego fit(max_iter=1, warm_start=True) repetido reinicia el schedule 10 veces y queda en t_ = 401 (medido) en vez de t_ = 4001; el control recibiria un presupuesto de un orden de magnitud menor que A1 (afecta a gates: False; sin resultados; enmienda pre-ejecucion).
- Enmienda del pre-registro (representacion.vectorizador.token_pattern (fidelidad de la receta canonica)): el pre-registro de V34 se apartaba de la representacion canonica del proyecto por un typo mio; el check de no-deriva del runner lo detecto al comparar su espejo con el JSON (afecta a gates: False; sin resultados; enmienda pre-ejecucion).

## 4. Consumo diagnostico del fold 0 y puerta G0-G6 (fail-fast)

- dentro del fold 0, ANTES de los folds 1-4 y ANTES de integrar.
- G3 y G4 leen el fold-val del fold 0, que es parte de la validacion canonica; su fallo NO toca delta.
- G5 evaluada: True; puerta superada: False.

- A1 (FTRL-Proximal softmax multiclase multi-pasada, decisional): gates fallidos ['G0_objetivo_de_la_regla', 'G1_banda_ll_train', 'G2_progreso', 'G3_macro_f1_fold_val', 'G4_frecuencias']
- A2 (gemelo SGD multi-pasada, control): todos los gates en verde

## 5. Triage formal (solo si A1 falla y A2 pasa G0-G4)

- Se ejecuta si: A1 falla la puerta y A2 pasa G0-G4; condicion activada: True.
- Clasificacion: MECANISMO_DEFINITIVO; indicador de defecto: False; nodo concedido: True.
- Tabla de mecanismo publicada dentro del runner: True.
- One shot: reabrir la familia exige acta nueva con hipotesis mecanica nueva, no otro gate.

| comprobacion | valor | criterio |
|---|---|---|
| D1_sondas_abcde_en_verde | True | sondas (a)-(e) con pass = True (vectorizado vs lazo, raiz analitica, TotalF1 Macro, sin NaN, log_loss propio vs sklearn) |
| D2_sondas_fghi_en_verde | True | sondas (f)-(i) con pass = True (filas que suman 1, reproducibilidad de alpha*, estabilidad del ES, simetria de presupuesto) |
| D3_sesgo_unico_por_brazo | True | A1 con una sola coordenada de sesgo (n_features == n_vocabulario, n_coordinates_a1 == n_vocabulario + 1) y A2 con fit_intercept=True sobre la matriz sin columna de sesgo |
| D4_presupuesto_no_truncado | True | pasada_de_parada en [P_min=3, P_max=10], Trayectoria con tantas entradas como pasadas y ordenes/particion ES identicos en A1 y A2 |
| D5_es_bien_configurado | True | mejor_pasada <= pasada_de_parada (se restaura el mejor w) y particion ES identica en ambos brazos, ajustada SOLO en ES-train; la metrica del ES es el macro-F1 del 10 % de TRAIN, nunca fold-val |
| D6_determinismo_exacto | True | G5 evaluado con max|delta p| = 0.0 EXACTO en los dos brazos (sonda h) |
| D7_alpha_reproducible | True | sonda (g) con diferencia exacta 0 entre las dos rutas de alpha* y mismo sha256 de alpha* en los dos brazos |
| D8_sin_segundo_alpha_ni_fallback | True | un unico alpha por fold, identico en A1 y A2, igual a la forma cerrada 2*M/(d_norm*P_max^2) al cero binario: no hay grid ni reintento |

## 6. Sondas (a)-(j), obligatorias antes de cualquier ajuste

| sonda | pass | detalle |
|---|---|---|
| (a) a_vectorizado_vs_lazo | True | FTRL softmax multiclase multi-pasada (3 pasadas) vectorizada vs lazo ingenuo, problema aleatorio 200x50x6 |
| (b) b_raiz_analitica_lambda0 | True | lambda1=lambda2=0 con feature constante: raiz analitica alpha*z/(beta+sqrt(n)) y paso unico alpha*g/(beta+|g|) con p=1/6 |
| (c) c_totalf1_macro | True | sin descripcion |
| (d) d_sin_nan_logloss_finito | True | softmax multiclase sin NaN/inf y log_loss multiclase finito |
| (e) e_logloss_multiclase_propio | True | log_loss multiclase propio -media log p[i,y[i]] contra sklearn.metrics.log_loss con filas que suman 1 |
| (f) f_filas_suman_uno | True | max|sum_c p_c - 1| del softmax de 6 logits, sin temperatura |
| (g) g_reproducibilidad_alpha | True | sin descripcion |
| (h) h_estabilidad_es | True | dos corridas identicas del fold 0 de cada brazo: misma pasada de parada, mismo ll_train y sha256 de los pesos IDENTICOS (tolerancia 0) |
| (i) i_simetria_presupuesto | True | sha256 de la particion ES y de los ordenes por pasada identicos en ambos brazos; los ordenes se generan UNA vez por fold y se pasan a los dos brazos |
| (j) j_tabla_de_mecanismo | no es gate | tabla de mecanismo: ll_train y D_acum por pasada de A1 y A2, razon |sigma*w|/|g| y w_max_abs; la falsacion se reproduce dentro del runner, NO se importa de /tmp |

- Politica: si alguna falla -> exit 4 ANTES de entrenar; cualquier ajuste exige enmienda pre-ejecucion.

## 7. Invariantes de no leakage

| invariante | valor |
|---|---|

## 8. Cierre

- cierre de FAMILIA, NO un gate de delta: se concede SOLO si (a) triage formal registrado que clasifica el fallo como MECANISMO_DEFINITIVO y no como DEFECTO_DE_IMPLEMENTACION (bug, sesgo duplicado, presupuesto truncado, ES mal configurado) decidido ANTES de elegir nodo; si es defecto -> se corrige y se re-emite; (b) la tabla de mecanismo esta PUBLICADA dentro del runner (ll_train y D_acum por pasada de A1 y A2, razon |sigma*w|/|g|), no importada de /tmp; (c) la lista exhaustiva de la familia esta registrada con el estado de cada miembro
- Sin particiones reservadas, sin envio, sin segunda particion: el champion queda intacto y el pre-registro prohibe automatizar el envio.
- Artefactos: experiments/v34_01_ftrl_softmax_multipasada.json | experiments/v34_01_estado.json | reports/v34_e1_ftrl_multipasada.md | reports/figures/v34_e1_curvas.png | reports/v34_novelty_registry.csv | artifacts/p34_scores/v34_e1/a1_ftrl_multipasada.npy | artifacts/p34_scores/v34_e1/a2_sgd_multipasada.npy.
