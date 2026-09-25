# Investigación de la capa de decisión ordinal y corrección de fronteras (2026-09-25)

## 1. Pregunta y criterio

¿Existe una forma distinta de convertir las señales ya disponibles en una decisión ordinal de seis clases que reduzca el error en las fronteras `18-24|25-34` y `25-34|35-49`, sin introducir representaciones aprendidas nuevas?

**Gate pre-registrado**: Δ ≥ +0,003 y ≥4/5 folds positivos. **Holdout**: agotado, no se reutiliza. **Leaderboard**: solo evidencia externa, nunca selección. **Sin** embeddings, vectorizadores nuevos, corpus externos ni etiquetas de test.

## 2. Matriz de cobertura (qué estaba ya hecho)

Ver `reports/age_hypothesis_coverage.md`: los modelos ordinales (E-015, E-021a) y los márgenes por pares (E-028a) **ya existen como miembros** del ensamble, y el routing duro está cerrado; lo que no existía era una **capa de decisión** sobre probabilidades congeladas. Eso es exactamente lo que se ha evaluado aquí.

## 3. Sustrato y protocolo

- Probabilidades OOF de V12: `features/oof_ensemble_v12.npy` (macro-F1 OOF 0.6005).
- Bloques OOF de E-028a (márgenes adyacentes), vista estructural y features densas de contexto: todo ya existente.
- **corrected CV**: para cada fold, la capa se ajusta con las OOF de los otros cuatro folds y se aplica al retenido. Ninguna capa se ajusta ni evalúa in-sample.
- Métricas: macro-F1 (principal), accuracy, P/R/F1 por clase, MAE ordinal, reparto por distancia ordinal, matriz 6x6, rescate/daño, errores por frontera y por segmento.

## 4. Resultados de todas las capas

| Capa | macro-F1 | Δ | folds+ | rescate | daño | neto | modif. |
|---|---|---|---|---|---|---|---|
| G desempate E[Y] (tau 0.05) | 0.6015 | +0.0011 | 4/5 | 2705 | 2822 | -117 | 5.17% |
| F coste ordinal (d2=1.5, d3=1.6) | 0.6008 | +0.0004 | 4/5 | 3747 | 4284 | -537 | 7.87% |
| H calibración por banda de confianza (5 bandas) | 0.6006 | +0.0002 | 3/5 | 997 | 1009 | -12 | 1.94% |
| C4 corrección residual (gamma 0.25, alpha 10) | 0.6004 | -0.0000 | 3/5 | 1397 | 1397 | +0 | 2.60% |
| H calibración por banda de confianza (3 bandas) | 0.6003 | -0.0001 | 2/5 | 856 | 901 | -45 | 1.56% |
| D decodificación transición (lambda 0.1) | 0.6000 | -0.0004 | 0/5 | 171 | 198 | -27 | 0.33% |
| C2 corrección residual (gamma 0.5) | 0.5999 | -0.0005 | 1/5 | 2712 | 2793 | -81 | 5.07% |
| I3 corrector LogReg (más regularizado) | 0.6000 | -0.0005 | 2/5 | 1924 | 2173 | -249 | 4.11% |
| G desempate E[Y] (tau 0.1) | 0.5999 | -0.0005 | 2/5 | 4677 | 5288 | -611 | 9.13% |
| I1 corrector LogReg sobre P(V12) | 0.5999 | -0.0005 | 2/5 | 1924 | 2174 | -250 | 4.11% |
| C3 corrección residual (+vista estructural, gamma 0.5) | 0.5999 | -0.0006 | 2/5 | 2715 | 2805 | -90 | 5.12% |
| D decodificación transición (lambda 0.25) | 0.5997 | -0.0007 | 0/5 | 539 | 550 | -11 | 0.96% |
| I5 calibración isotónica por clase | 0.5997 | -0.0007 | 1/5 | 1110 | 1168 | -58 | 2.12% |
| I2 corrector LogReg sobre log-probabilidades | 0.5992 | -0.0012 | 1/5 | 1340 | 1465 | -125 | 2.72% |
| C1 corrección residual (solo contexto, gamma 1.0) | 0.5988 | -0.0016 | 0/5 | 5270 | 5513 | -243 | 9.83% |
| D decodificación transición (lambda 0.5, condicionada) | 0.5987 | -0.0017 | 0/5 | 1290 | 1336 | -46 | 2.32% |
| D decodificación transición (lambda 0.5) | 0.5987 | -0.0018 | 0/5 | 1271 | 1323 | -52 | 2.28% |
| A2 CDF media (peso 0.5) | 0.5974 | -0.0031 | 0/5 | 3102 | 3709 | -607 | 6.30% |
| E proyección ordinal (fuerza 0.1) | 0.5971 | -0.0033 | 0/5 | 6856 | 7377 | -521 | 12.77% |
| B2 Luce + V12 (peso 0.25) | 0.5971 | -0.0034 | 0/5 | 2104 | 2294 | -190 | 4.21% |
| G desempate E[Y] (tau 0.2) | 0.5939 | -0.0065 | 0/5 | 7001 | 8925 | -1924 | 14.07% |
| D decodificación transición (lambda 1.0) | 0.5931 | -0.0073 | 0/5 | 4096 | 4454 | -358 | 7.41% |
| E proyección ordinal con gate (fuerza 0.25, tau 0.1) | 0.5916 | -0.0088 | 0/5 | 10364 | 11775 | -1411 | 20.22% |
| E proyección ordinal con gate (fuerza 0.4, tau 0.1) | 0.5915 | -0.0089 | 0/5 | 10376 | 11793 | -1417 | 20.26% |
| F coste ordinal (d2=2.0, d3=3.0) | 0.5909 | -0.0095 | 0/5 | 7414 | 10299 | -2885 | 17.36% |
| A1 solo umbrales acumulados | 0.5887 | -0.0118 | 0/5 | 5800 | 8004 | -2204 | 12.74% |
| B2 Luce + V12 (peso 0.5) | 0.5867 | -0.0137 | 0/5 | 5507 | 6694 | -1187 | 11.86% |
| E proyección ordinal (fuerza 0.25) | 0.5840 | -0.0164 | 0/5 | 13341 | 16004 | -2663 | 25.85% |
| E proyección ordinal con gate (fuerza 0.4, tau 0.2) | 0.5727 | -0.0277 | 0/5 | 16194 | 20679 | -4485 | 32.45% |
| E proyección ordinal (fuerza 0.4) | 0.5698 | -0.0306 | 0/5 | 17053 | 22057 | -5004 | 34.00% |
| I4 corrector Ridge sobre el par top-2 | 0.5572 | -0.0432 | 0/5 | 18384 | 25456 | -7072 | 38.54% |
| F coste ordinal (d2=0.5, d3=0.4) | 0.4687 | -0.1317 | 0/5 | 3960 | 25139 | -21179 | 39.23% |
| B1 Luce solo (márgenes E-028a) | 0.2947 | -0.3057 | 0/5 | 8926 | 50207 | -41281 | 60.29% |

**Ninguna de las 33 capas alcanza el gate.** La mejor es `G desempate E[Y] (tau 0.05)` con Δ +0.0011 (4/5) y rescate 2705 frente a daño 2822: **red de seguridad nulo** (neto -117).

## 5. Interpretación por familia

- **A (umbrales acumulados)**: un modelo de umbrales por sí solo alcanza 0,5887 (comparado con 0,6005 de V12) y la fusión de supervivencias no mejora: la evidencia de fronteras ya está dentro del meta-modelo.
- **E (proyección ordinal)**: desplazar masa del top-1 a su vecino siempre destruye más de lo que arregla (el daño crece con la fuerza). El argmax de V12 ya es la mejor decisión de esa distribución.
- **C (corrección residual de fronteras)**: la versión más suave (γ=0,25) queda exactamente en neto 0: el contexto aporta información, pero no más de la que ya usa el combinador.
- **D (matriz de transición)**: los sesgos de confusión existen pero corregirlos con shrinkage empeora de forma monótona al aumentar λ; la corrección y el error son del mismo orden.
- **H (calibración por banda)**: casi inerte (+0,0002), coherente con que una transformación por fila no puede cambiar el argmax y las bandas solo añaden ruido.
- **B (Luce sobre márgenes adyacentes)**: convertir los márgenes de E-028a en una distribución de 6 clases da 0,2947: los márgenes no están calibrados como escala de Bradley-Terry.
- **G (desempate por E[Y])**: mejora el MAE ordinal (0,6552 -> 0,6454) y las fronteras extremos, pero empeora las dos fronteras centrales: **minimizar error ordinal no es mejorar macro-F1** (confirmación directa de la hipótesis del brief §15).
- **I (corrector mínimo)**: todas las parametrizaciones son negativas o nulas; el corrector reescribe aciertos de V12 a cambio de casi tantos fallos nuevos.

## 6. Dónde vive el error residual (brief §36)

- **Errores totales**: 67,616 de 169,336 filas.
- **Techo de información**: 23,485 filas (34.7% del error) donde **ningún miembro acierta**.
- **Rescatable**: 45,213 filas (66.9% del error) donde algún miembro acierta y el ensamble no. Familias que más rescatan: ordinal 20.0%, prototipos 19.9%, denso 19.3%.
- **Fronteras centrales**: `18-24|25-34` acumula 10,122 cruces de frontera (sobre 28,721 errores en las filas de la pareja) y `25-34|35-49` 7,252 cruces (28,438 errores en la pareja).
- Reparto del error por distancia ordinal (base): {'0': 60.073, '1': 23.836, '2': 9.562, '3': 4.086, '4': 1.907, '5': 0.535}.

## 7. Conclusión defendible

Bajo las restricciones de AGE, **la representación ya contiene la información disponible y el límite está en la observabilidad de la edad, no en la regla de decisión**: 33 capas de decisión ordinal, corrección de frontera, decodificación por confusión, calibración por confianza, combinación probabilística de márgenes y corrector mínimo, todas ajustadas con OOF sobre las probabilidades congeladas de V12, **ninguna mejora de forma robusta**. El error restante se reparte en un techo de información (35% de los errores, ninguna fuente disponible acierta) y un 65% que es agregación: algún miembro sabe, pero no hay ninguna regla de decisión sobre las probabilidades que convierta ese saber en aciertos sin perder tantos como gana. La única señal nueva medida en toda la fase previa (vista estructural, +0,0018) es del mismo orden que el ruido de estas capas.

## 8. Estado del modelo

- **V12 permanece congelado**: `age_submission_v12_reproducible.csv` (leaderboard 0,62188).
- No se genera ninguna submission nueva: ninguna capa alcanza el gate.
- `age_submission_v12.csv` (0,62182) y `age_submission_v13_candidato_estructura_NO_ADOPTADO.csv` se conservan intactos como artefactos históricos.

_Generado por `scripts/phase34_informe_capa_decision.py`; 33 capas evaluadas._
