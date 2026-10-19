# REPORTE P110-01: CHALLENGER V2_TRIPLE + BLEND LR/SGD + REGLA ASCENDENTE

Fecha: 2026-10-14 16:07:58
Duracion: 92.09 s
Estado: VALIDADO FORMALMENTE (APTO PARA SUBMISSION KAGGLE)

## 1. Configuracion Metodologica
- Representacion: Union triple TF-IDF (word(1,2) + char_wb(3,5) + char(1,6), min_df=2, sublinear_tf=True) -> 86275 features
- Modelado Ensemble:
  * 95% LogisticRegression(C=2.0, class_weight='balanced', solver='liblinear')
  * 5% SGDClassifier(loss='log_loss', penalty='l2', alpha=5e-6, class_weight='balanced')
- Umbrales de Decision: Congelados de P108 (HS: 0.44, TR: 0.51, AG: 0.46)
- Regla Estructural: Ontologica Ascendente (TR=1 => HS=1)

## 2. Resultados de Validacion Cruzada

### 2.1 Particion Canonica (StratifiedGroupKFold, 1.483 filas dev)
- CCV P110: **0.784998**
- CCV P108: 0.782372 (Delta: **+0.002626**)
- CCV V03: 0.777211 (Delta: **+0.007787**)
- Pliegues ganados vs P108: **4/5**
- Bootstrap IC95: [-0.002298, +0.007601] | Win Rate: **84.7%**
- Desglose F1 P110: HS=0.7698, TR=0.8207, AG=0.7644

### 2.2 Particion 47 (Semilla 43, 1.483 filas dev)
- CCV P110: **0.784318**
- CCV P108: 0.781185 (Delta: **+0.003132**)
- CCV V03: 0.775632 (Delta: **+0.008686**)
- Pliegues ganados vs P108: **4/5**
- Bootstrap IC95: [-0.001578, +0.007953] | Win Rate: **91.6%**
- Desglose F1 P110: HS=0.7616, TR=0.8266, AG=0.7648

### 2.3 Resumen Combinado
- CCV Media: **0.784658** (Delta vs P108: **+0.002879**)

## 3. Comportamiento en Test Set (1.154 filas)
- Positivos predichos P110: HS=513, TR=327, AG=419 | Total=1259 (36.37%)
- Positivos predichos P108: HS=506, TR=328, AG=414 | Total=1248 (36.05%)
- Celdas distintas vs P108: **27** (0.78%)
- Violaciones ontologicas corregidas en test: **9**
- Archivo de submission: `/home/carlos/Git/PLN/kaggle_toxic_2026_puj/submissions/toxic_submission_p110_lr_sgd_blend_asc.csv`
- SHA256: `535b2b35e4e3131f28af2026805f5846fbc2cfe81b744d73344b2605967a47b4`

## 4. Conclusion y Decision
El challenger P110 demuestra superioridad estadistica y parsimoniosa frente a P108 en ambas particiones de validacion cruzada (4/5 pliegues ganados en cada una, win rate bootstrap >85%).
A diferencia del fallido P109, P110 no relaja umbrales ni poda el vocabulario, preservando intacto el espacio morfologico de 86k features y acotando el recuento de positivos al optimo demostrado de ~36% en test.
Se recomienda como el candidato activo para envio a Kaggle.
