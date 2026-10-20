## 4.2 V30: los cuatro cuartos, el ensemble y la regla plug-in

Con la matriz de 279 columnas fija, la pregunta pasó a ser cómo exprimirla. La respuesta llegó en cuatro
pasos (V19->V20, V29, V30 y la regla plug-in), cada uno con su acta y su verificación:

**V19 → V20 — «cuatro cuartos».** V19 añadió un árbol de boosting (LightGBM con configuración
pre-registrada; Ke et al., «LightGBM: A Highly Efficient Gradient Boosting Decision Tree», NeurIPS 2017) como tercer miembro del meta-modelo. V20 fue más allá y definió un candidato con cuatro
componentes de peso igual (0.25 cada uno): la regresión logística, el MLP(64), el árbol y una **cadena de
cinco umbrales acumulativos** (cinco clasificadores binarios «¿el rango es ≤ k?» que se corrigen para ser
monótonos y se convierten en distribución; Frank y Hall, «A Simple Approach to Ordinal Classification», ECML 2001). La cadena aporta la estructura ordinal de forma explícita.
V20 alcanzó **CV 0.6097782** y **0.62689** en el leaderboard.

**V29 — búsqueda con Optuna.** Una exploración sistemática de LightGBM (Ke et al., 2017) con Optuna (Akiba et al., «Optuna: A Next-generation Hyperparameter Optimization Framework», KDD 2019) encontró una
configuración distinta por fold (el mejor *trial* individual logró **0.6121698** de CV). Importante: las
configuraciones quedaron **congeladas** en el artefacto de la búsqueda — nada se re-optimiza después.

**V30 — el ensemble.** El campeón lineal del proyecto mezcla tres miembros — V20 (cuartos intactos), el
LightGBM de V29 y un LightGBM refinado por Optuna (Akiba et al., 2019) — con pesos que son la **media de los vectores óptimos
por fold, renormalizada** (`0.3548 / 0.3354 / 0.3098`, congelados). El resultado: **CV 0.6134471** y
**0.63302** en el leaderboard.

**La regla plug-in por clase.** La última pieza fue una regla de decisión (Lipton, Elkan y Narayanaswamy, «Thresholding Classifiers to Maximize F1 Score», arXiv, 2014) que reemplaza el `argmax`
plano por un `argmax` con **pesos por clase** (`[1.0000, 1.1749, 1.2721, 1.1630, 1.0010, 0.8128]`): los
pesos se ajustaron por ascenso por coordenadas en espacio logit, **dentro de cada fold** (usando los
otros cuatro como validación interna), y el vector final es la media de los cinco, congelada en un
pre-registro. La intuición es que la macro-F1 no trata igual a todas las clases: mover la frontera de una
clase mayoritaria puede costar más de lo que ayuda, y el plug-in re-calibra esa decisión sin tocar el
modelo. El sanity check del proyecto exige reproducir exactamente la mejora documentada
(**0.6134471 → 0.6159732487657944** en el OOF canónico, sin ningún cambio en filas de alta confianza).

```python
# V19/V20: los tres miembros de V19 y la cadena de cinco umbrales (OOF, folds canonicos).
t0 = time.time()
p19_logreg, p19_mlp, p19_arbol = v19_oofs(matriz, y, clase, fold_idx)
seis_oof = v20_cadena_oof(matriz, clase, fold_idx)
v20_oof = 0.25 * p19_logreg + 0.25 * p19_mlp + 0.25 * p19_arbol + 0.25 * seis_oof
print(f"reconstruccion V19/V20: {time.time() - t0:.0f}s\n")
for nombre, probabilidad in [("V19 logreg", p19_logreg), ("V19 MLP", p19_mlp),
                             ("V19 arbol", p19_arbol), ("cadena 5 umbrales", seis_oof),
                             ("V20 (cuartos)", v20_oof)]:
    print(f"  {nombre:20} macro-F1 {macro_f1(clase, np.argmax(probabilidad, axis=1)):.6f}")
print("\n  V20 documentado: CV 0.6097782")
```

```python
# V29/V30: los dos LightGBM con configuracion congelada por fold, la mezcla y la regla plug-in.
t0 = time.time()
v29_oof = lgbm_refit_oof(matriz, clase, fold_idx, V29_CONFIGS,
                         base={"random_state": SEED})
v30_oof = lgbm_refit_oof(matriz, clase, fold_idx, V30_CONFIGS)
oof_v30 = mezcla(v20_oof, v29_oof, v30_oof)
print(f"reconstruccion V29/V30: {time.time() - t0:.0f}s\n")
print(f"  V29 individual   : {macro_f1(clase, np.argmax(v29_oof, axis=1)):.7f} (doc 0.6121698)")
print(f"  V30 individual   : {macro_f1(clase, np.argmax(v30_oof, axis=1)):.7f} (doc 0.6122387)")
cv_v30_oof = macro_f1(clase, np.argmax(oof_v30, axis=1))
cv_plugin_oof = macro_f1(clase, regla_plug_in(oof_v30))
print(f"  V30 mezcla       : {cv_v30_oof:.7f} (doc 0.6134471)")
print(f"  V30 + plug-in    : {cv_plugin_oof:.10f} (doc 0.6159732487657944)")
```

```python
# Verificacion del orden de clases (diagonal) y figura de la progresion hasta el plug-in.
medias_condicionales = np.zeros((6, 6))
for fila in range(6):
    medias_condicionales[fila] = oof_v30[clase == fila].mean(axis=0)
diagonal = [int(np.argmax(medias_condicionales[fila])) for fila in range(6)]
assert diagonal == list(range(6)), f"orden de clases roto: {diagonal}"
print("diagonal de orden de clases verificada:", diagonal)

fig, ejes = plt.subplots(1, 2, figsize=(11.5, 3.8))
imagen = ejes[0].imshow(medias_condicionales, cmap="Blues")
ejes[0].set_title("E[P(clase) | rango real] sobre el OOF de V30")
ejes[0].set_xticks(range(6))
ejes[0].set_xticklabels(CLASSES, rotation=45, fontsize=8)
ejes[0].set_yticks(range(6))
ejes[0].set_yticklabels(CLASSES, fontsize=8)
plt.colorbar(imagen, ax=ejes[0], fraction=0.046)
hitos = ["V12", "V20", "V30", "V30\n+plug-in"]
valores = [0.6005, 0.6097782, cv_v30_oof, cv_plugin_oof]
ejes[1].bar(hitos, valores, color=["#a0aec0", "#90cdf4", "#4299e1", "#2b6cb0"])
ejes[1].set_ylim(0.59, 0.62)
ejes[1].set_ylabel("macro-F1 (CV agrupada)")
ejes[1].set_title("Progresion hasta la regla plug-in")
for posicion, valor in enumerate(valores):
    ejes[1].text(posicion, valor + 0.0004, f"{valor:.4f}", ha="center", fontsize=8)
plt.tight_layout()
plt.show()
```

```python
# Lado TEST del campeon: la matriz de 279 de test (representaciones + 13 modelos + features + NB-SVM).
def adv05_test(textos_train, etiquetas, textos_test):
    """ADV-05 lado test: NB-SVM char300+menciones ajustado sobre todo el train."""
    textos_todo = pd.concat([textos_train, textos_test], ignore_index=True)
    etiquetas_todo = np.concatenate([etiquetas, np.array(["13-17"] * len(textos_test))])
    tr = np.arange(len(textos_train))
    va = np.arange(len(textos_train), len(textos_train) + len(textos_test))
    config = {**CHAR_300K, "mentions": {"config": MENTION, "pattern": r"@\w+"}}
    return nb_svm_margins(textos_todo, etiquetas_todo, tr, va, config)


t0 = time.time()
reps_test = representaciones_test(train, test)
bloques_72 = base_models_test(reps_test, y, densas35(dense), densas35(dense_test),
                              train["text"], test["text"])
nb_test = adv05_test(train["text"], y, test["text"])
nb_word_test_matrix = nb_word_test(train["text"], y, test["text"])
matriz_test = matriz_test_fresca(bloques_72, nb_test, densas35(dense_test),
                                 estructuradas_test.to_numpy(float), supervisadas_te, nb_word_test_matrix)
print(f"lado test completo: {time.time() - t0:.0f}s | matriz test {matriz_test.shape}")
assert matriz_test.shape == (112892, 279)
```

```python
# Probabilidades de test del candidato V30 (sin plug-in): cuartos + los dos LightGBM reajustados.
t0 = time.time()
p20_test = v20_test(matriz, matriz_test, y, clase)
p29_test = lgbm_refit_test(matriz, clase, matriz_test, V29_CONFIGS)
p30_test = lgbm_refit_test(matriz, clase, matriz_test, V30_CONFIGS)
prob_v30_test = mezcla(p20_test, p29_test, p30_test)
print(f"test V30: {time.time() - t0:.0f}s | forma {prob_v30_test.shape}")
print("decisiones V30+plug-in sobre test:")
pred_v30_test = regla_plug_in(prob_v30_test)
for posicion, etiqueta in enumerate(CLASSES):
    print(f"  {etiqueta}: {int((pred_v30_test == posicion).sum()):,}")
```


### La segunda partición del campeón (semilla 987654)

El proyecto verificó el linaje V30 también sobre una **segunda partición agrupada**: misma
agrupación de clústeres, semilla 987654, con solape parcial con la canónica — es una
re-ejecución en otra partición, no una partición independiente, y así quedó declarado. Esta
segunda evaluación cumple dos papeles: confirma el linaje fuera de la partición canónica
(0.6144120 de base y 0.6164700 con el plug-in, los valores documentados) y es, además, la que el
proyecto usa como miembro V30 en la **evaluación publicada del campeón V48 y de los consensos del
arco 2026 (secciones 5.2 y 4.6)**: su OOF es el que entró en los consensos geométricos de los
récords. La receta es idéntica (cuartos + dos LightGBM con configuraciones y pesos congelados); la
única diferencia es que los reajustes de esta partición usan el techo completo de árboles, sin early
stopping (Ke et al., 2017), tal como los ejecutó el script original de la confirmación. La partición se
genera aquí mismo con la misma función de la sección 2.4 y la semilla 987654, de modo que el código
queda legible y verificable en el punto donde se usa.

```python
# Segunda particion (semilla 987654): la misma regla sobre los mismos clusters, generada junto a su uso.
fold_idx2 = generar_folds(y, cluster_de[:n_train], semilla=987654)
print("segunda particion (semilla 987654):", np.bincount(fold_idx2).tolist())
```

```python
# La segunda particion del campeon (semilla 987654): validacion del linaje V30.
t0 = time.time()
v20_segunda = cuartos_v20_oof(matriz, y, clase, fold_idx2)
v29_segunda = lgbm_refit_oof_sin_es(matriz, clase, y, fold_idx2, V29_CONFIGS)
v30_segunda = lgbm_refit_oof_sin_es(matriz, clase, y, fold_idx2, V30_CONFIGS)
oof_v30_segunda = mezcla(v20_segunda, v29_segunda, v30_segunda)
print(f"segunda particion reconstruida: {time.time() - t0:.0f}s")
print(f"  V30 segunda        : {macro_f1(clase, np.argmax(oof_v30_segunda, axis=1)):.7f} "
      f"(doc 0.6144120)")
print(f"  V30 + plug-in segunda: {macro_f1(clase, regla_plug_in(oof_v30_segunda)):.7f} "
      f"(doc 0.6164700)")
```
