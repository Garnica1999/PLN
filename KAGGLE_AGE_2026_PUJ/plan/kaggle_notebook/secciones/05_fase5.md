# FASE 5 — Evaluación

## 5.1 El protocolo de validación y su reproducibilidad

Toda la evaluación del proyecto se apoya en el mismo protocolo, definido en la Fase 2 y usado sin
excepciones: **validación cruzada de 5 folds agrupados por clústeres de near-duplicados** (folds
embebidos como constante; Kaufman, Rosset, Perlich y Stitelman, «Leakage in Data Mining: Formulation, Detection, and Avoidance», ACM Transactions on Knowledge Discovery from Data, 2012), con la métrica **macro-F1 sin ponderar** calculada de dos maneras
complementarias — *pooled* (todas las predicciones juntas) y media por folds — más pruebas de
significancia cuando una decisión estaba en juego (bootstrap (Efron, «Bootstrap Methods: Another Look at the Jackknife», The Annals of Statistics, 1979) pareado con re-muestreo de clústeres, B=1000,
intervalo de confianza al 95 % y fracción de folds positivos). Las decisiones importantes se confirmaron
además en una **segunda partición** con semilla distinta.

Este diseño tiene una propiedad que conviene explicitar: el CV agrupado es **conservador** respecto del
leaderboard (sección 5.5). Al eliminar los near-duplicados entre folds, mide la capacidad de generalizar
a textos nuevos; el leaderboard, en cambio, contiene near-duplicados del train y los «cobra», de modo que
el puntaje real queda sistemáticamente por encima del CV local. El proyecto verificó esa brecha en cada
envío y la usó como calibración, nunca como objetivo.

## 5.2 El campeón V48: consenso geométrico de seis modelos

El modelo entregado combina los seis componentes de la Fase 4 mediante un **Product-of-Experts**
(consenso bayesiano multiplicativo; Hinton, «Training Products of Experts by Minimizing Contrastive Divergence», Neural Computation, 2002): las probabilidades de cada modelo se combinan en **espacio
logarítmico** con pesos congelados (Heskes, «Selecting Weighting Factors in Logarithmic Opinion Pools», NeurIPS 1998),

$$\log P_{\text{geom}}(y = k \mid x) \propto \sum_{m=1}^{6} w_m \log P_m(y = k \mid x)$$

con $w = [0.5628, 0.0832, 0.1724, 0.0754, 0.0513, 0.0549]$ (V30, V43, V42, V40, V39-ordinal,
V39-HGB), pesos óptimos por validación cruzada anidada. La intuición del PoE: si varios modelos
independientes asignan probabilidad baja a una clase, el producto la hunde más rápido que el promedio;
el consenso es más agresivo con el desacuerdo y más estable con el acuerdo. La ablación de operadores
(sección 4.5) confirmó que el geométrico es el óptimo de su familia. El miembro V30 del consenso usa
su OOF de la segunda partición —el esquema con el que el proyecto selló el récord—; la celda reporta
además la variante estricta con el V30 canónico, para que el detalle sea verificable. Sobre el
consenso se aplica la misma **regla plug-in** (Lipton, Elkan y Narayanaswamy, «Thresholding Classifiers to Maximize F1 Score», arXiv, 2014) del campeón lineal, y el resultado se envía al
leaderboard: **CV 0.6186451** y **0.63724** en Kaggle (puesto #1).

```python
# El campeon V48: PoE geometrico sobre los seis componentes, version OOF y version test.
PESOS_GEOM = np.array([0.5628, 0.0832, 0.1724, 0.0754, 0.0513, 0.0549])
PESOS_GEOM = PESOS_GEOM / PESOS_GEOM.sum()


def poe_log_espacio(modelos):
    """Consenso Product-of-Experts: suma ponderada de log-probabilidades, normalizada."""
    eps = 1e-12
    log_geom = sum(w * np.log(np.clip(modelo, eps, 1.0))
                   for w, modelo in zip(PESOS_GEOM, modelos))
    p_geom = np.exp(log_geom - np.max(log_geom, axis=1, keepdims=True))
    return p_geom / p_geom.sum(axis=1, keepdims=True)


# Protocolo publicado del proyecto: el miembro V30 del consenso usa su OOF de la segunda
# particion, tal como lo evaluo el acta D-101 y la seccion 47 del notebook consolidado.
modelos_oof = [oof_v30_segunda, componentes_nn["v43"][0], componentes_nn["v42"][0],
               componentes_nn["v40"][0], componentes_nn["v39_ordinal"][0], hgb_oof]
p_geom_oof = poe_log_espacio(modelos_oof)
cv_v48 = macro_f1(clase, regla_plug_in(p_geom_oof))
print(f"V48 reconstruido (OOF, con plug-in): {cv_v48:.7f} (documentado 0.6186451)")

# Variante estricta con el V30 canonico (se declara en vez de esconderse).
modelos_oof_canonica = [oof_v30, componentes_nn["v43"][0], componentes_nn["v42"][0],
                        componentes_nn["v40"][0], componentes_nn["v39_ordinal"][0], hgb_oof]
cv_v48_canonica = macro_f1(clase, regla_plug_in(poe_log_espacio(modelos_oof_canonica)))
print(f"variante con el V30 canonico      : {cv_v48_canonica:.7f}")

modelos_test = [prob_v30_test, componentes_nn["v43"][1], componentes_nn["v42"][1],
                componentes_nn["v40"][1], componentes_nn["v39_ordinal"][1], hgb_test]
p_geom_test = poe_log_espacio(modelos_test)
pred_final = regla_plug_in(p_geom_test)
print("consenso test:", p_geom_test.shape,
      "| filas normalizadas:", bool(np.allclose(p_geom_test.sum(axis=1), 1.0)))
```

## 5.3 Análisis de error

El análisis de error del campeón se concentra en la estructura ordinal del problema. La matriz de
confusión sobre el OOF muestra el patrón esperado y deseable: los errores viven en las **fronteras
adyacentes** (confundir `18-24` con `25-34`), mientras que los errores lejanos son marginales. Esto no
es un accidente: es exactamente lo que la pérdida de Cramér (V43; Cramér, «On the composition of elementary errors: First paper: Mathematical deductions», Scandinavian Actuarial Journal, 1928; von Mises, «Wahrscheinlichkeitsrechnung und ihre Anwendung in der Statistik und theoretischen Physik», Franz Deuticke, 1931) y la cadena de umbrales (V20; Frank y Hall, «A Simple Approach to Ordinal Classification», ECML 2001) están
diseñadas para favorecer. La tabla por clase revela además que los extremos son los mejor clasificados
(65-xx 0.7446; 13-17 0.6845) y que los rangos centrales (25-34 0.5051; 18-24 0.5306) concentran el
error, por el solapamiento de vocabulario entre cohortes vecinas.

```python
# Matriz de confusion y metricas por clase del campeon (sobre el OOF).
pred_oof_final = regla_plug_in(p_geom_oof)
confusion = np.zeros((6, 6), dtype=int)
for real, predicho in zip(clase, pred_oof_final):
    confusion[real, predicho] += 1

fig, ejes = plt.subplots(1, 2, figsize=(12, 4.2))
imagen = ejes[0].imshow(confusion, cmap="Blues")
for fila in range(6):
    for columna in range(6):
        ejes[0].text(columna, fila, f"{confusion[fila, columna]:,}", ha="center", va="center",
                     fontsize=7, color="black" if confusion[fila, columna] < confusion.max() * 0.7
                     else "white")
ejes[0].set_xticks(range(6))
ejes[0].set_xticklabels(CLASSES, rotation=45, fontsize=8)
ejes[0].set_yticks(range(6))
ejes[0].set_yticklabels(CLASSES, fontsize=8)
ejes[0].set_xlabel("predicho")
ejes[0].set_ylabel("real")
ejes[0].set_title("Matriz de confusion (OOF del campeon)")
plt.colorbar(imagen, ax=ejes[0], fraction=0.046)

from sklearn.metrics import f1_score as f1_por_clase

f1_clases = [float(f1_por_clase(clase, pred_oof_final, average=None,
                                labels=list(range(6)))[posicion]) for posicion in range(6)]
ejes[1].bar(CLASSES, f1_clases, color="#2b6cb0")
for posicion, valor in enumerate(f1_clases):
    ejes[1].text(posicion, valor + 0.005, f"{valor:.3f}", ha="center", fontsize=8)
ejes[1].set_ylim(0, 0.75)
ejes[1].set_title(f"F1 por rango (macro-F1 = {cv_v48:.4f})")
plt.xticks(rotation=30, fontsize=8)
plt.tight_layout()
plt.show()

vecinos = sum(int(confusion[k, k + 1] + confusion[k + 1, k]) for k in range(5))
lejanos = int(confusion.sum() - confusion.trace() - vecinos)
print(f"aciertos: {int(confusion.trace()):,} | confusiones adyacentes: {vecinos:,} "
      f"| confusiones lejanas: {lejanos:,}")
print(f"proporcion de errores que son de frontera: {vecinos / (vecinos + lejanos) * 100:.1f}%")
```
