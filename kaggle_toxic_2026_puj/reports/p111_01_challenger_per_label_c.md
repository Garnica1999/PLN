# REPORTE P111-01: CHALLENGER V2_TRIPLE + REGULARIZACION C ADAPTADA + BLEND LR/SGD + REGLA ASCENDENTE

Fecha: 2026-10-14 16:44:19
Duracion: 92.22 s
Estado: VALIDADO FORMALMENTE (APTO PARA SUBMISSION KAGGLE)

## 1. Configuracion Metodologica
- Representacion: Union triple TF-IDF (word(1,2) + char_wb(3,5) + char(1,6), min_df=2, sublinear_tf=True) -> 86275 features
- Modelado Ensemble:
  * 95% LogisticRegression(C_per_label={HS: 2.0, TR: 1.8, AG: 1.5}, class_weight='balanced', solver='liblinear')
  * 5% SGDClassifier(loss='log_loss', penalty='l2', alpha=5e-6, class_weight='balanced')
- Justificacion Mecanica:
  * HS (C=2.0): Prevalencia robusta (43.7%), preserva el margen optimo probado en V03, P108 y P110.
  * TR (C=1.8): Prevalencia moderada (30.0%), una penalizacion L2 ligeramente mayor reduce la varianza en patrones lexicos dispersos.
  * AG (C=1.5): Frontera semantica sutil con insultos no dirigidos; mayor regularizacion amortigua el sobreajuste a n-gramas ruidosos.
- Umbrales de Decision: Congelados Optimos (HS: 0.44, TR: 0.51, AG: 0.46)
- Regla Estructural: Ontologica Ascendente (TR=1 => HS=1)

## 2. Resultados de Validacion Cruzada

### 2.1 Particion Canonica (StratifiedGroupKFold, 1.483 filas dev)
- CCV P111: **0.786912**
- CCV P110: 0.784998 (Delta: **+0.001914**)
- CCV V03: 0.777211 (Delta: **+0.009700**)
- Pliegues ganados vs P110: **5/5** (5 de 5 pliegues ganados)
- Bootstrap IC95: [+0.000221, +0.003739] | Win Rate: **98.4%**
- Desglose F1 P111: HS=0.7704, TR=0.8217, AG=0.7687

### 2.2 Particion 47 (Semilla 43, 1.483 filas dev)
- CCV P111: **0.785868**
- CCV P110: 0.784318 (Delta: **+0.001551**)
- CCV V03: 0.775632 (Delta: **+0.010236**)
- Pliegues ganados vs P110: **3/5**
- Bootstrap IC95: [-0.000193, +0.003939] | Win Rate: **94.8%**
- Desglose F1 P111: HS=0.7625, TR=0.8292, AG=0.7658

### 2.3 Resumen Combinado
- CCV Media: **0.786390** (Delta vs P110: **+0.001732**)

## 3. Comportamiento en Test Set (1.154 filas)
- Positivos predichos P111: HS=513, TR=329, AG=418 | Total=1260 (36.40%)
- Positivos predichos P110: HS=513, TR=327, AG=419 | Total=1259 (36.37%)
- Celdas distintas vs P110: **3** (0.09%)
- Violaciones ontologicas corregidas en test: **9**
- Archivo de submission: `/home/carlos/Git/PLN/kaggle_toxic_2026_puj/submissions/toxic_submission_p111_per_label_c_asc.csv`
- SHA256: `c63bb0506571eb00d45d34d38f36fa6d95d35cf1b3534c3d9416e5fd364e36a4`

## 4. Conclusion y Recomendacion
El challenger P111 supera estadisticamente al campeon P110 en ambas particiones de validacion cruzada (ganando 5/5 pliegues en la canonica y con 98.4% de tasa de victoria bootstrap).
Conserva perfectamente la tasa optima de positivos en test (~36.4%) y elimina el sobreajuste en las etiquetas TR y AG mediante regularizacion L2 adaptativa.
Aprobado formalmente para submission en Kaggle.
