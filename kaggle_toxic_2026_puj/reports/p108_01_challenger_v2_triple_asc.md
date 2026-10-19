# P108-01: Challenger V2_triple + Regla Ascendente

**Fecha:** 2026-10-14  
**Archivo Submission:** `toxic_submission_p108_v2_triple_asc.csv`  
**SHA256:** `d3c19d2bcaf1e0dff1ee1a8106e88348c2c6c15bf4b7d4a0f3d7e834f6ec861b`  
**Duracion ejecucion:** 99.8 s  

## 1. Resumen de Desempeno en Validacion Cruzada (10 Pliegues)

| Particion | V03 CCV | Challenger CCV | Delta | Pliegues Ganados | Bootstrap IC95 | Confianza Mejora |
|---|---|---|---|---|---|---|
| **Canonica (1.483 filas)** | 0.777211 | **0.782372** | **+0.005160** | **4/5** | [-0.001117, 0.012266] | **94.2%** |
| **Particion 47 (1.483 filas)** | 0.775632 | **0.781185** | **+0.005553** | **5/5** | [-0.000417, 0.011628] | **95.9%** |

## 2. Comparacion en el Conjunto de Test (1.154 filas)

- **Total celdas distintas vs V03:** 66 de 3.462 (1.91%)
- **Diferencias por etiqueta:** HS=34, TR=10, AG=22
- **Violaciones ontologicas TR=1 & HS=0 corregidas:** 10
- **Recuento de positivos predichos:**
  * V03: HS=506, TR=334, AG=422
  * Challenger: HS=506, TR=328, AG=414

## 3. Conformidad con Reglas
- Cero transformers / LLMs.
- Cero embeddings preentrenados.
- Entrenamiento exclusivo desde cero con las 1.731 filas de `toxic_train.csv`.
- Cero fuga hacia el conjunto de test.
- Verificacion de formato de submission 100% identica a `submission.csv`.
