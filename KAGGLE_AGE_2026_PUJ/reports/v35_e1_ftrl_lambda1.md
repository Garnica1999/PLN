# V35-E1 - Ejecucion: FTRL-Proximal softmax multiclase multi-pasada con ES interno

- Fecha de ejecucion: 2026-09-30
- Pre-registro congelado: experiments/v35_00_preregistro.json (commit c0c762e9912b92ab67764fe73b2efb60aef8e65c) y reports/v35_preregistro.md
- Puerta de convergencia (normativa): reports/v34_convergence_gate.md
- Commit de ejecucion: `f4df7115faa4430823348a9763070b5f40a6ed0b`
- Baseline primario recomputado: 0.6159732 (referencia nested 0.6154643)
- Ancla trivial recalculada: 0.04854212180687087 (clase 13-17)
- 32 hilos fijos; TF-IDF ajustada SOLO en el ES-train de cada fold; P_max = 10 pasadas con ES interno (P_min = 3, paciencia 2, mejora minima 0.0001); RSS pico 0.69 GB (tope 20 GB)
- Compliance AST: 39 literales de ruta y 31 modulos, 0 violaciones; lecturas declaradas: 39 literales en 163 funciones, 0 violaciones
- Control A2 (enmienda 2): partial_fit_por_pasada x 10, con el estimador construido con max_iter = 1 (NO 10).

## Veredicto: LINEA_ONLINE_CERRADA_POR_MECANISMO

la puerta se cerro en el fold 0 (pasa = False) por la rama pre-declarada 'LINEA_ONLINE_CERRADA_POR_MECANISMO': gates fallidas de A1 ['G0_objetivo_de_la_regla', 'G1_banda_ll_train', 'G2_progreso', 'G3_macro_f1_fold_val', 'G4_frecuencias', 'G8_acotacion_norma'] y de A2 []; falla ['G8_acotacion_norma'] en A1: la hipotesis de acotacion del L1 queda refutada por medicion sobre la trayectoria real del fold 0. El cierre es DEFINITIVO de la familia: no hay segundo lambda1 (seria grid) ni acta futura dentro de este espacio. El pre-registro prohibe integrar y hacer bootstrap, luego delta y bootstrap quedan VACIOS

## 1. Mecanismo: trayectoria por pasada de A1 y A2 (fold 0 consumido-diagnostico)

- alpha* = 2*M/(d_norm*P_max^2) con M = ln(5) = 1.6094379124341003 y P_max = 10 (0 valores evaluados, no es un grid); d_norm = 0.9128367962785974 del prior MEDIDO del ES-train del fold 0, luego alpha* = 0.035262336465737745.
- Banda objetivo de la regla (gate G0 de A1): D_acum en [0.804719, 3.218876]; banda de G1 sobre ll_train: [0.05, 1.781571].
- w_max_abs por pasada (la firma de D-083 era w_max_abs = 26.4 con 28.86 de ll_train): A1 1.250016e+01 con ll_train 19.159444; A2 2.465037e+01 con ll_train 0.404220.

| brazo | pasada | ll_train (ES-train) | D_acum | macro-F1 ES-val | w_max_abs | razon |sigma*w|/|g| |
|---|---|---|---|---|---|---|---|
| A1_ftrl_lambda1 | 1 | 10.058343 | 12.072639 | 0.046077 | 6.599752e+00 | 1.332e-01 |
| A1_ftrl_lambda1 | 2 | 15.401682 | 18.474387 | 0.046077 | 9.991393e+00 | 1.836e-01 |
| A1_ftrl_lambda1 | 3 | 19.159444 | 23.177427 | 0.046077 | 1.250016e+01 | 2.079e-01 |
| A2_sgd_presupuesto | 1 | 1.062199 | 9.009318 | 0.450724 | 3.685110e+01 | n/d (A2 no es FTRL) |
| A2_sgd_presupuesto | 2 | 0.531503 | 6.504952 | 0.478322 | 2.897554e+01 | n/d (A2 no es FTRL) |
| A2_sgd_presupuesto | 3 | 0.461367 | 5.946400 | 0.484608 | 2.640142e+01 | n/d (A2 no es FTRL) |
| A2_sgd_presupuesto | 4 | 0.444882 | 5.686317 | 0.485945 | 2.563577e+01 | n/d (A2 no es FTRL) |
| A2_sgd_presupuesto | 5 | 0.443319 | 5.572206 | 0.492124 | 2.525916e+01 | n/d (A2 no es FTRL) |
| A2_sgd_presupuesto | 6 | 0.416137 | 5.499474 | 0.493740 | 2.509153e+01 | n/d (A2 no es FTRL) |
| A2_sgd_presupuesto | 7 | 0.413570 | 5.395290 | 0.496981 | 2.482520e+01 | n/d (A2 no es FTRL) |
| A2_sgd_presupuesto | 8 | 0.412464 | 5.374682 | 0.493135 | 2.474956e+01 | n/d (A2 no es FTRL) |
| A2_sgd_presupuesto | 9 | 0.409993 | 5.321915 | 0.498733 | 2.472974e+01 | n/d (A2 no es FTRL) |
| A2_sgd_presupuesto | 10 | 0.404220 | 5.255189 | 0.494276 | 2.465037e+01 | n/d (A2 no es FTRL) |

Figura: reports/figures/v35_e1_curvas.png.

## 2. Tabla de simetria de sesgo (enmienda pre-ejecucion 1)

- Camino de A1: coordenada de sesgo x0=1.0 materializada dentro de FTRL (propio z, propio n, sin termino de sesgo aparte).
- Camino de A2: fit_intercept=True sobre la MISMA matriz SIN columna de sesgo pre-anadida (enmienda 1: se corrige el doble sesgo de D-083/V33-E1).
- no hay doble sesgo en ningun brazo: A1 materializa el sesgo como coordenada unica y A2 lo aporta solo por intercept_; el defecto de D-083 y V33-E1 (matriz con sesgo pre-anadido Y fit_intercept=True) NO se reproduce aqui.

| brazo | caminos de sesgo | fuente | magnitud | valor |
|---|---|---|---|---|
| A1_ftrl_lambda1 | 1 | coordenada de sesgo x0=1.0 dentro de FTRL (propio z, propio n) | max|gradiente| de la coordenada x0 (fold 0) | 1.000000 |
| A2_sgd_presupuesto | 1 | fit_intercept=True sobre la matriz SIN columna de sesgo pre-anadida (un solo camino) | max|intercept_| (fold 0) | 3.451903 |

| fold | max|g_sesgo| (A1) | max|intercept_| (A2) | razon A1/A2 | mismo orden de magnitud |
|---|---|---|---|---|
| 0 | 1.000000 | 3.451903 | 2.897e-01 | True |

## 3. Asimetrias declaradas

- `lambda2` de A1 = 1.0 frente a `alpha_reg` de A2 = 1e-06 (cociente 1e+06). A1 tiene un regularizador (L1) que A2 no tiene: A2 es control de PRESUPUESTO, no gemelo mecanico. Ninguna conclusion puede descansar en el contraste A1 vs A2 (delta, razones entre brazos e IC bootstrap cruzado quedan contaminados): el veredicto se apoya solo en evidencia interna de A1 contra la regla derivada (G0/G1/G4) mas el triage
- El veredicto NO se apoya en esa asimetria: evidencia interna de A1 contra la regla derivada (G0/G1/G4), mas G7/G8 sobre la trayectoria de A1, mas el triage formal y las sondas (k), (l) y (m) del gemelo de lambda1.
- Lectura correcta de A2: A2 no es un gemelo mecanico de A1: se lee unicamente como 'el presupuesto de 10 pasadas NO es el limite'. El delta entre brazos, las razones entre brazos y el IC bootstrap cruzado quedan contaminados por la asimetria del regularizador y NO se publican como evidencia.
- Enmienda 2 (control A2): partial_fit_por_pasada con 10 llamadas y estimador construido con max_iter = 1 (NO 10).
- Desviacion declarada: {'no_bit_identico_a_fit_max_iter_10': True, 'medicion': '4.5e-5 a 2 pasadas y 2.0e1 a 4 pasadas en un problema sintetico', 'motivo_de_la_eleccion': 'es la unica via que permite evaluar el early stopping despues de cada pasada, que el pre-registro exige (P_min 3, paciencia 2, restaurar el mejor coef_), y da el mismo presupuesto que A1'}.
- Enmienda del pre-registro (anclaje_lambda1.verificacion (item 2 del diseno)): circularidad: en FTRL z depende de w y w depende de lambda1, luego el z de A1 nunca puede reproducir el umbral que lo define. Medido: con lambda1 = mediana la mediana del z es 0.2333793834039217 (diferencia -0.006558555225118473), con lambda1 = 0 es 0.23993793862904017 (diferencia exacta 0.0) (afecta a gates: None; sin resultados; enmienda pre-ejecucion).
- Enmienda del pre-registro (sondas_nuevas (k)-(m): dimension del problema y alcance del contraste): el mini tiene del orden de 1350 x 51 coordenadas frente a las 121921 x 107793 del gemelo pre-ejecucion, luego la identidad numerica es imposible por dimension; ademas con lambda1 = 0.2399 el mini apenas poda (escasez efectiva 1.0 en los tres casos), de modo que alli no hay magnitud que reproducir. Cargar el corpus dentro de --verificar volveria la verificacion pesada y con efectos de carga (afecta a gates: None; sin resultados; enmienda pre-ejecucion).
- Enmienda del pre-registro (config.vectorizador.token_pattern y hashes.vectorizador): el patron congelado lleva backslashes dobles: no compila ninguna palabra (medido, re.compile del patron congelado no encuentra match en 'hola mundo'), luego el vocabulario seria vacio y la representacion inservible. El canonico es el que declaran los pre-registros de V33 y V34 y el que produjo las 107793 coordenadas que registran el nucleo de V34 y la sonda pre-ejecucion de esta via (afecta a gates: False; sin resultados; enmienda pre-ejecucion).
- Enmienda del pre-registro (secciones de contexto ausentes: control, corpus, anclas, representacion, budgeto, regla_alpha): 17 lecturas prereg[...] de secciones inexistentes: la corrida moria DESPUES de entrenar los dos brazos en los 5 folds, con checkpoints y OOF ya escritos, y verificar_constantes no lo detectaba (106/106 no era cobertura real). Detectado al ejecutar de verdad construir_resultado, escribir_informe y resumen_consola con un estado simulado, porque --verificar y --humo NO ejecutan la ruta de salida (afecta a gates: False; sin resultados; enmienda pre-ejecucion).
- Enmienda del pre-registro (control.asimetria_declarada (desambiguacion de la no-equivalencia estructural)): un documento de registro con dos declaraciones para lo mismo es la ambiguedad que el proyecto ya pago; el runner ya no imprimia el texto de V34 (la linea del informe lee la asimetria de la sonda), luego el riesgo era de lectura, pero la ambiguedad seja (afecta a gates: False; sin resultados; enmienda pre-ejecucion).

### Regla de lectura UNIDIRECCIONAL de A2

- **Regla**: A2 es un control de PRESUPUESTO, no un gemelo mecanico: ninguna conclusion puede descansar en el contraste A1 frente a A2, porque el delta, las razones entre brazos y el IC bootstrap cruzado quedan contaminados por las diferencias de mecanismo (estimador, regularizador y camino de sesgo).
- Lectura correcta de A2: A2 solo sirve para comprobar que el presupuesto NO es el limite: que el control sano aprende de verdad (G0, G1, G2 y G4 de A2). Un A2 que aprende NO demuestra que un A1 con L1 funcione, y un A1 que falla frente a un A2 sano no se atribuye a que 'el L1 no sirve'.
- Por que A2 no es gemelo mecanico: A1 lleva regularizador L1 y A2 no; A2 aporta su sesgo con fit_intercept=True sobre la matriz sin columna de sesgo y A1 con una coordenada x0 = 1.0 dentro de FTRL.
- Magnitud contaminada: el delta de A1 frente al de A2.
- Magnitud contaminada: las razones entre brazos (por ejemplo la razon max/min de ||w||_2 de A1 leida como contraste frente a A2).
- Magnitud contaminada: el IC95 bootstrap cruzado de la diferencia entre brazos.
- En que se apoya el veredicto: solo evidencia interna de A1 contra la regla derivada (G0, G1 y G4) mas el triage formal y las magnitudes L1 de su propia trayectoria (G7, G8 y el criterio 'L1 inerte').
- Prohibido: no afirmar que el L1 mejora o empeora el resultado comparando A1 con A2, ni leer el signo de esa diferencia como evidencia de mecanismo.

## 4. Consumo diagnostico del fold 0 y puerta G0-G8 (fail-fast)

- dentro del fold 0, ANTES de los folds 1-4 y ANTES de integrar.
- G3 y G4 leen el fold-val del fold 0, que es parte de la validacion canonica; su fallo NO toca delta.
- G5 evaluada: True; puerta superada: False.

- A1 (FTRL-Proximal softmax multiclase multi-pasada con L1 = mediana congelada, decisional): gates fallidos ['G0_objetivo_de_la_regla', 'G1_banda_ll_train', 'G2_progreso', 'G3_macro_f1_fold_val', 'G4_frecuencias', 'G8_acotacion_norma']
- A2 (control de PRESUPUESTO: no gemelo mecanico, no tiene L1): todos los gates en verde

### Gates L1 (G7 y G8), puerta a puerta

- Exclusivas de A1: ['G7_L1_activo', 'G8_acotacion_norma']; gates no aplicables por brazo: {'A1_ftrl_lambda1': [], 'A2_sgd_presupuesto': ['G7_L1_activo', 'G8_acotacion_norma']}.
- Criterio de conteo: una gate cuenta como fallida SOLO si pass is False; pass = None significa que no aplica a ese brazo o que aun no se ha medido.
- Nota: pass = None significa que la gate NO APLICA a ese brazo (o que aun no se ha medido) y por eso NO cuenta como fallo: el unico conteo de fallos es el de gates_fallidos_de, que exige pass is False.

| brazo | gate | pass | estado | valor | lambda1 | criterio |
|---|---|---|---|---|---|---|
| A1_ftrl_lambda1 | G7_L1_activo | True | medida | True | 0.23993793862904017 | escasez efectiva (coordenadas con al menos una clase viva, |z_j| > lambda1) mayor que cero en TODAS las pasadas Y no degradada respecto de la pasada 1 mas alla de la tolerancia declarada de 0.02 |
| A1_ftrl_lambda1 | G8_acotacion_norma | False | medida | False | 0.23993793862904017 | razon max/min de ||w||_2 entre la primera y la ultima pasada <= 1.0 (||w||_2 = norma euclidea de la matriz de pesos completa) |
| A2_sgd_presupuesto | G7_L1_activo | None = NO APLICA (no cuenta como fallo) | no_aplica |  |  | escasez efectiva (coordenadas con al menos una clase viva, |z_j| > lambda1) mayor que cero en todas las pasadas y no degradada mas alla de 0.02 respecto de la pasada 1 |
| A2_sgd_presupuesto | G8_acotacion_norma | None = NO APLICA (no cuenta como fallo) | no_aplica |  |  | razon max/min de ||w||_2 entre la primera y la ultima pasada <= 1.0 |
- A1_ftrl_lambda1 / G7_L1_activo: motivo = None; lectura = G7 mide si el L1 esta ACTIVO (descarta coordenadas de verdad). No mide si el L1 ALCANZA: acotar no es lo mismo que bastar, de eso responde G8.
- A1_ftrl_lambda1 / G8_acotacion_norma: motivo = None; lectura = acotar NO es lo mismo que bastar: una razon mayor que 1.0 con escasez efectiva positiva significa L1 ACTIVO que aun asi deja crecer la norma.
- A2_sgd_presupuesto / G7_L1_activo: motivo = A2 es el control de PRESUPUESTO y NO lleva regularizador L1: no hay umbral |z_j| > lambda1 que medir ni hipotesis de acotacion que comprobar. La gate no aplica a este brazo y NO cuenta como fallo; lectura = None.
- A2_sgd_presupuesto / G8_acotacion_norma: motivo = A2 es el control de PRESUPUESTO y NO lleva regularizador L1: no hay umbral |z_j| > lambda1 que medir ni hipotesis de acotacion que comprobar. La gate no aplica a este brazo y NO cuenta como fallo; lectura = None.

### Nodo del arbol emitido por el codigo

- Rama de cierre: True.
- Nodo emitido: LINEA_ONLINE_CERRADA_POR_MECANISMO.
- Causa: ['G8_acotacion_norma'].
- Lectura: falla ['G8_acotacion_norma'] en A1: la hipotesis de acotacion del L1 queda refutada por medicion sobre la trayectoria real del fold 0. El cierre es DEFINITIVO de la familia: no hay segundo lambda1 (seria grid) ni acta futura dentro de este espacio.
- Cierre definitivo: True; integracion: None; motivo de no calculo: la puerta se cerro en el fold 0 (pasa = False) por la rama pre-declarada 'LINEA_ONLINE_CERRADA_POR_MECANISMO': gates fallidas de A1 ['G0_objetivo_de_la_regla', 'G1_banda_ll_train', 'G2_progreso', 'G3_macro_f1_fold_val', 'G4_frecuencias', 'G8_acotacion_norma'] y de A2 []; falla ['G8_acotacion_norma'] en A1: la hipotesis de acotacion del L1 queda refutada por medicion sobre la trayectoria real del fold 0. El cierre es DEFINITIVO de la familia: no hay segundo lambda1 (seria grid) ni acta futura dentro de este espacio. El pre-registro prohibe integrar y hacer bootstrap, luego delta y bootstrap quedan VACIOS
- G7 falla en A1: False; G8 falla en A1: True; G0-G4 fallan en A1: ['G0_objetivo_de_la_regla', 'G1_banda_ll_train', 'G2_progreso', 'G3_macro_f1_fold_val', 'G4_frecuencias'].
- Mapa pre-declarado de la rama (nodo al que lleva cada caso; los nombres se contrastan contra el pre-registro):
| caso | nodo | texto del pre-registro |
|---|---|---|
| G7_o_G8_fallan_en_A1 | LINEA_ONLINE_CERRADA_POR_MECANISMO | LINEA_ONLINE_CERRADA_POR_MECANISMO: la acotacion queda refutada y la familia se cierra DEFINITIVAMENTE (no hay segundo lambda1 ni acta futura dentro de este espacio) |
| G7_y_G8_pasan_pero_G0_G4_fallan_en_A1 | HIPOTESIS_CONJUNTA_REFUTADA | HIPOTESIS_CONJUNTA_REFUTADA (acota pero no aprende); NO es cierre por conveniencia y se registra con su propia evidencia |
| cualquier_otro_fallo | NO_ATRIBUIBLE_BY_GATE | mismo criterio que en V34: el fallo no es atribuible a G7 ni a G8 de A1 |
| ninguna_gate_falla | gates de la serie (rama NO de cierre) | se evaluan los gates de la serie; PROMISING exigiria segunda particion con autorizacion del usuario, que el script no ejecuta |
- Los nombres de los nodos siguen siendo los del pre-registro: False.

### Criterio "L1 inerte" (hipotesis activa o analiticamente muerta)

- Veredicto: L1 ACTIVO (lambda1 = 0.23993793862904017, 3 pasadas).
- Fraccion de coordenadas con |z_j| <= lambda1 por pasada: [0.048927110294731624, 0.019333351887413808, 0.019333351887413808]; maxima en las pasadas >= 2: 0.019333351887413808.
- Lectura: L1 ACTIVO pero insuficiente: en las pasadas >= 2 hay coordenadas con |z_j| <= lambda1 = 0.23993793862904017 (fraccion maxima medida 0.019333351887413808 > 0), luego el umbral descarta de verdad; que G8 falle significa que ACOTAR NO ES LO MISMO QUE BASTAR, no que la puerta este mal
- Responsabilidad de la puerta: la puerta aplica un umbral activo; su fallo es de hipotesis, no de implementacion.

## 5. Triage formal (solo si A1 falla y A2 pasa G0-G4)

- Se ejecuta si: la puerta se cerro en el fold 0 (rama pre-declarada del arbol); condicion activada: None.
- Clasificacion: MECANISMO_DEFINITIVO; indicador de defecto: False; nodo concedido: True.
- Tabla de mecanismo publicada dentro del runner: True.
- One shot: reabrir la familia exige acta nueva con hipotesis mecanica nueva, no otro gate; y re-derivar lambda1 tras ver la puerta seria grid.

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
| D9_un_solo_lambda1_y_esta_activo | True | un unico lambda1 por fold, igual a la mediana congelada (0 busqueda), y el L1 esta ACTIVO: la fraccion de coordenadas con |z_j| <= lambda1 es mayor que 0 en las pasadas >= 2. Si fuese 0 exacta, la hipotesis estaria analiticamente muerta y el fallo NO se culparia a la puerta |

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
| (j) j_tabla_de_mecanismo | None | tabla de mecanismo: ll_train y D_acum por pasada de A1 y A2, razon |sigma*w|/|g| y w_max_abs; la falsacion se reproduce dentro del runner, NO se importa de /tmp |
| (k) k_caso_lambda1_cero | None | sondas (k), (l) y (m): los tres casos del gemelo de lambda1 reproducidos DENTRO del runner sobre el problema mini con las mismas reglas de update, el mismo alpha y los mismos ordenes congelados, contrastados contra el artefacto versionado con tolerancia declarada |
| (l) l_caso_lambda1_alpha | None | sondas (k), (l) y (m): los tres casos del gemelo de lambda1 reproducidos DENTRO del runner sobre el problema mini con las mismas reglas de update, el mismo alpha y los mismos ordenes congelados, contrastados contra el artefacto versionado con tolerancia declarada |
| (m) m_caso_lambda1_mediana | no es gate | sondas (k), (l) y (m): los tres casos del gemelo de lambda1 reproducidos DENTRO del runner sobre el problema mini con las mismas reglas de update, el mismo alpha y los mismos ordenes congelados, contrastados contra el artefacto versionado con tolerancia declarada |

- Politica: si alguna de las nueve sondas pass/fail falla -> exit 4 ANTES de entrenar; cualquier ajuste exige enmienda pre-ejecucion. Las sondas no-gate (j) y (k), (l), (m) se publican y no deciden el veredicto.

### Identidad de los tres casos de lambda1 (NO-GATE: no decide el veredicto)

- identidad numerica de los tres casos del gemelo de lambda1 (lambda1 = 0, lambda1 = alpha y lambda1 = mediana) ejecutada en el FOLD 0 de la corrida real, sobre el stream real de ES-train, con las mismas reglas de update, el mismo alpha y el mismo orden congelado que la sonda pre-ejecucion.
- Donde se ejecuta: fold 0 de la corrida real, UNA sola vez; NO en los folds 1-4 (el stream es otro) ni en --verificar (que no carga el corpus para esto: alli la identidad es imposible por dimension y las sondas (k), (l) y (m) comparan por firma de mecanismo).
- Criterio: diferencia relativa <= 1e-09 en ll_train y ||w||_2 de cada pasada frente al artefacto; diferencia EXACTA 0 y sha256 identico en la mediana del anclaje y en el snapshot de z.

| caso | lambda1 | pasada | ll_train | \|\|w\|\|_2 |
|---|---|---|---|---|
| lambda1 = 0.0 | 0.0 | 1 | 10.086313578600262 | 29.194850667547023 |
| lambda1 = 0.0 | 0.0 | 2 | 15.42062521362065 | 47.27989643345345 |
| lambda1 = 0.0 | 0.0 | 3 | 19.174420615478212 | 60.98360849536189 |
| lambda1 = alpha | 0.035262336465737745 | 1 | 10.083240868084 | 29.02889262656771 |
| lambda1 = alpha | 0.035262336465737745 | 2 | 15.418472712408523 | 47.14299208512644 |
| lambda1 = alpha | 0.035262336465737745 | 3 | 19.172685241471445 | 60.86346392574805 |
| lambda1 = mediana | 0.23993793862904017 | 1 | 10.058343327445655 | 28.27459446151044 |
| lambda1 = mediana | 0.23993793862904017 | 2 | 15.40168213060249 | 46.45212395628998 |
| lambda1 = mediana | 0.23993793862904017 | 3 | 19.1594443184596 | 60.23720212230849 |

- Comparacion contra experiments/v35_00b_sonda_lambda1.json (sha256 e6554eb25269a3a4...) con la tolerancia RELATIVA declarada 1e-09 sobre `ll_train` y `||w||_2` de CADA pasada (3 pasadas por caso) y tolerancia 0.0 en el anclaje; alpha del fold 0 0.035262336465737745 frente al del artefacto 0.035262336465737745 (identico = True); stream real de 121921 filas x 107793 coordenadas.

| caso | lambda1 run | lambda1 artefacto | peor dif. rel. ll_train | peor dif. rel. \|\|w\|\|_2 | reproduce |
|---|---|---|---|---|---|
| lambda1 = 0.0 | 0.0 | 0.0 | 0.000e+00 | 0.000e+00 | True |
| lambda1 = alpha | 0.035262336465737745 | 0.035262336465737745 | 0.000e+00 | 0.000e+00 | True |
| lambda1 = mediana | 0.23993793862904017 | 0.23993793862904017 | 0.000e+00 | 0.000e+00 | True |
- **VEREDICTO de la identidad: reproduce = True**; fallos publicados = ninguno (la escasez efectiva se publica como diagnostico, no como criterio).

- Anclaje del ciclo (`lambda1` del NUCLEO DE ANCLAJE = 0.0): mediana(|z_j| > 0) al final de la pasada 1 = 0.23993793862904017 frente a la congelada 0.23993793862904017; diferencia EXACTA = 0.0 (cero exacto = True).
- sha256 del snapshot de `z` del anclaje = ec1d1a7b13454f8b7309446576318487b9325df018c5b454b228a6a336630f66; congelado = ec1d1a7b13454f8b7309446576318487b9325df018c5b454b228a6a336630f66; identico = True; el hash congelado es prefijo del del artefacto = True; el del artefacto = ec1d1a7b13454f8b7309446576318487b9325df018c5b454b228a6a336630f66.
- Criterio del anclaje: diferencia EXACTA 0 con la mediana congelada y sha256 del snapshot de z identico al de la sonda pre-ejecucion versionada.

- Diagnostico SECUNDARIO del core YA REGULARIZADO (`lambda1` = 0.23993793862904017): mediana(|z_j| > 0) = 0.2333793834039217; diferencia respecto del anclaje = -0.006558555225118473 frente a la esperada -0.006558555225118473 (coincide = True). **NO es el anclaje** (`es_el_anclaje` = False): NO es el anclaje: la mediana del z del core YA REGULARIZADO no puede coincidir con la del core de anclaje porque z depende de w y w depende de lambda1. Su diferencia respecto del anclaje es la magnitud del acoplamiento z <- w(lambda1) y por eso el anclaje solo puede reproducirse con lambda1 = 0.0.

- **NO-GATE**: esto es una SONDA y NO una gate, y NO decide el veredicto del ciclo (`LINEA_ONLINE_CERRADA_POR_MECANISMO`), que se apoya solo en G0-G6 mas G7 y G8 de A1 y en el triage formal. NO-GATE: la identidad informa de que la politica del pre-registro reflejada en el codigo no deriva, pero no decide el veredicto. Si alguna comparacion falla se publica como fallo de la sonda y el ciclo continua con su veredicto intacto
- Coste medido 60.8 s frente a los 60 s declarados, con un presupuesto declarado de 1200 s.

### Supuesto pendiente del runner sobre el `token_pattern` del vectorizador

- El pre-registro congelado en el commit `c0c762e9912b92ab67764fe73b2efb60aef8e65c` trae en `config.vectorizador` la clave `nota_token_pattern` (claves de nota detectadas: ['nota_token_pattern']), que es DOCUMENTACION PURA: no es un parametro del vectorizador y por eso no entra en el hash de la receta ejecutable. El ajuste usa SIEMPRE el dict local `VECTORIZADOR_FIT`, nunca el dict del pre-registro.
- Las ocho claves de receta coinciden BIT A BIT con el canonico `9bae157614458dbdf26fb7aed4c78d2954c36e895f830341e59b23fa63e2b51b`: la receta identica bit a bit = True (claves: ['analyzer', 'lowercase', 'max_df', 'min_df', 'ngram_range', 'strip_accents', 'sublinear_tf', 'token_pattern']); sha256 del dict local = 9bae157614458dbdf26fb7aed4c78d2954c36e895f830341e59b23fa63e2b51b (es el canonico = True); sha256 del dict del pre-registro = 8698703344c934437dc94d4b78add663f97791a233661a2db51894980adb4f04; sha256 del pre-registro restringido a las ocho claves de receta = 9bae157614458dbdf26fb7aed4c78d2954c36e895f830341e59b23fa63e2b51b.
- Caso reconocido: `solo_notas_del_prereg` (coincide exacto = False; diferencia solo en `token_pattern` = False); equivalente = True, y esa comprobacion SI es gate (a diferencia de la subsection de identidad anterior).
- Nota: el pre-registro V35 arrastro en su origen el token_pattern ANTERIOR a la enmienda pre-ejecucion 3 de V34 (dobles backslashes), que deja el vocabulario vacio y no es ejecutable; la tercera enmienda pre-ejecucion ya lo restauro a la receta canonica de un backslash y fijo el sha256 9bae1576..., que es la de v31_02, la de los pre-registros V33 y V34 ya enmendados y la del nucleo de V34 verificado por sha256. La comprobacion NO se relaja: exige que las ocho claves de receta coincidan bit a bit y que el dict local tenga el sha256 canonico congelado. La clave nota_token_pattern del pre-registro es documentacion pura: no es un parametro del vectorizador, porque el ajuste usa siempre el dict local VECTORIZADOR_FIT

## 7. Invariantes de no leakage

| invariante | valor |
|---|---|

## 8. Cierre

- LINEA_ONLINE_CERRADA_POR_MECANISMO: la acotacion queda refutada y la familia se cierra DEFINITIVAMENTE (no hay segundo lambda1 ni acta futura dentro de este espacio)
- Sin particiones reservadas, sin envio, sin segunda particion: el champion queda intacto y el pre-registro prohibe automatizar el envio.
- Artefactos: experiments/v35_01_ftrl_lambda1.json | experiments/v35_01_estado.json | reports/v35_e1_ftrl_lambda1.md | reports/figures/v35_e1_curvas.png | reports/v35_novelty_registry.csv | artifacts/p35_scores/v35_e1/a1_ftrl_lambda1.npy | artifacts/p35_scores/v35_e1/a2_sgd_presupuesto.npy.
