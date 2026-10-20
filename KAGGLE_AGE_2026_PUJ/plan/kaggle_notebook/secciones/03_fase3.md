# FASE 3 — Preparación y representación (fase histórica V1–V17: V1–V12 re-ejecutada, V13–V17 documentadas)

Esta fase reconstruye la primera mitad del proyecto: cómo se pasó del texto crudo a una representación
que funciona, cómo se construyeron las features de segunda generación y cómo se llegó al meta-modelo de
14 miembros (V12), la base sobre la que después se construyó todo el resto. Cada experimento re-ejecutable
se vuelve a ejecutar aquí desde cero y su resultado se contrasta con el documentado; los que exigen
artefactos congelados o procesos largos quedan documentados.

## 3.1 Limpieza y normalización

La limpieza es deliberadamente mínima, y eso es una decisión medida: el corpus ya viene normalizado a
ASCII (sin tildes ni eñes) y sin URLs problemáticas, así que las transformaciones agresivas (lematizar,
quitar stopwords, minúsculas forzadas) fueron probadas en su momento y **perdían señal**. Lo único que se
hace siempre es eliminar el relleno de espacios del ancho fijo de la exportación (`strip`), que es un
artefacto y no contenido. Los vectorizadores se configuran con `lowercase=True` y `sublinear_tf=True`
(escala logarítmica de frecuencias), que resultaron estables en todos los barridos.

## 3.2 Representaciones comparadas (V1–V4)

La primera pregunta del proyecto fue cómo representar el texto **sin embeddings**: la apuesta fue la
combinación de n-gramas de palabra (capturan léxico y jerga: «crush», «examen», «hipoteca») con n-gramas
de carácter `char_wb` (capturan ortografía, sufijos y variantes: «jajaja», «xq», emojis de texto) (Sapkota, Bethard, Montes y Solorio, «Not All Character N-grams Are Created Equal: A Study in Authorship Attribution», NAACL-HLT 2015). El
baseline documentado (E-001) unió `word(1-2)` y `char_wb(3-5)` con un `LinearSVC` (Fan et al., «LIBLINEAR: A Library for Large Linear Classification», Journal of Machine Learning Research, 2008) y alcanzó
**macro-F1 0.5344 ± 0.0050** en validación agrupada (y 0.55235 en el leaderboard — el primer envío del
proyecto). Aquí se re-ejecuta la variante documentada (la unión) con el mismo
protocolo, junto con los dos suelos de referencia: el azar
uniforme (1/6) y el **suelo estricto** de predecir siempre la clase mayoritaria (0.04854 documentado).
Las variantes por separado quedan documentadas con sus cifras medidas en su momento (solo palabra
0.4971 ± 0.0034; solo carácter 0.5408 ± 0.0050): re-ejecutarlas costaba ~5 minutos de vectorización y
no cambiaba ninguna decisión.

```python
# Utilidad: validacion cruzada agrupada (5 folds canonicos) para una representacion + modelo.
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC


def cv_representacion(vectorizadores, fabrica_modelo, textos, etiquetas, folds):
    """Macro-F1 medio por folds de una lista de vectorizadores + un modelo lineal."""
    puntajes, n_features = [], 0
    for fold in range(5):
        tr = folds != fold
        va = folds == fold
        bloques_tr, bloques_va = [], []
        for vectorizador in vectorizadores:
            bloques_tr.append(vectorizador.fit_transform(textos[tr]))
            bloques_va.append(vectorizador.transform(textos[va]))
        x_tr = hstack(bloques_tr).tocsr()
        x_va = hstack(bloques_va).tocsr()
        n_features = x_tr.shape[1]
        modelo = fabrica_modelo()
        modelo.fit(x_tr, etiquetas[tr])
        puntajes.append(float(f1_score(etiquetas[va], modelo.predict(x_va), average="macro")))
    return float(np.mean(puntajes)), float(np.std(puntajes)), n_features
```

```python
# Representaciones V1-V4: word, char_wb y la union documentada (E-001), con LinearSVC.
TEXTOS = train["text"].to_numpy()


def fabrica_word():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True,
                           max_features=300_000, lowercase=True)


def fabrica_char():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                           sublinear_tf=True, max_features=300_000, lowercase=True)


def fabrica_svc():
    return LinearSVC(C=1.0, max_iter=3000)


variantes = {
    "union word+char (E-001)": [fabrica_word(), fabrica_char()],
}
resultados_representaciones = {}
for nombre, vectorizadores in variantes.items():
    t0 = time.time()
    media, desviacion, n_features = cv_representacion(vectorizadores, fabrica_svc, TEXTOS, y, fold_idx)
    resultados_representaciones[nombre] = media
    print(f"{nombre:26} macro-F1 {media:.4f} +- {desviacion:.4f} | "
          f"{n_features:,} features | {time.time() - t0:.0f}s")
print("\nreferencia documentada: E-001 (union + LinearSVC) = 0.5344 +- 0.0050")
```

```python
# Suelos de referencia y figura comparativa.
mayoritaria = pd.Series(y).value_counts().idxmax()
f1_mayoritaria = float(f1_score(y, np.full(len(y), mayoritaria), average="macro"))

fig, eje = plt.subplots(figsize=(9, 3.4))
nombres = list(resultados_representaciones) + ["suelo estricto", "azar uniforme"]
valores = list(resultados_representaciones.values()) + [f1_mayoritaria, 1 / 6]
colores = ["#2b6cb0", "#c53030", "#a0aec0"]
eje.bar(nombres, valores, color=colores)
eje.axhline(0.5344, color="#2f855a", linestyle="--", linewidth=1,
            label="E-001 documentado (0.5344)")
for posicion, valor in enumerate(valores):
    eje.text(posicion, valor + 0.004, f"{valor:.4f}", ha="center", fontsize=9)
eje.set_ylabel("macro-F1 (CV agrupada)")
eje.set_title("Representaciones y suelos (V1-V4 re-ejecutada)")
eje.legend(fontsize=8)
plt.xticks(rotation=12, fontsize=9)
plt.tight_layout()
plt.show()
print(f"suelo estricto (clase mayoritaria '{mayoritaria}'): {f1_mayoritaria:.5f} (documentado 0.04854)")
```

## 3.3 Features densas, estructuradas y supervisadas

Sobre la representación base, el proyecto construyó tres bloques de features que después alimentan al
meta-modelo (y que en el campeón ocupan 195 de las 279 columnas). Este bloque de trabajo se adoptó como
`AGE_ADVANCED_V01` tras un gate estricto: **+0.0031 de CV (positivo en 5/5 folds), +0.0052 en el holdout
congelado y +0.0043 en el leaderboard** — una de las pocas adopciones del proyecto, y de las más sólidas.

- **Densas (35 columnas):** textometría y estilometría del texto (Schler, Koppel, Argamon y Pennebaker, «Effects of Age and Gender on Blogging», AAAI Spring Symposium 2006; Argamon, Koppel, Pennebaker y Schler, «Mining the Blogosphere: Age, gender and the varieties of self-expression», First Monday 2007) — longitudes, riqueza léxica (TTR,
  hapax, entropía), proporción de stopwords, emoticonos, mayúsculas, puntuación múltiple. Son locales al
  documento (no dependen del corpus), totalmente deterministas y sin parámetros aprendidos.
- **Estructuradas (87 columnas):** seis familias de forma — distribución de longitudes de palabra,
  distribución de caracteres, riqueza léxica avanzada (MATTR, Yule, Maas), forma de las palabras,
  posición de entidades (menciones, hashtags, URLs; Nguyen, Gravel, Trieschnigg y Meder, «How Old Do You Think I Am? A Study of Language and Age in Twitter», ICWSM 2013) y entropía de estructura (puntuación, compresión).
- **Supervisadas (73 columnas):** rasgos que **usan las etiquetas del fold de entrenamiento** y por eso
  se calculan estrictamente in-fold (nunca ven la validación): probabilidad de cada rango dada una
  mención/hashtag/fuente de retuit (con suavizado sobre el prior), rareza léxica contra el vocabulario
  del fold (con centinela para palabras fuera de vocabulario) y puntuaciones léxicas por clase con
  log-odds de Monroe (Monroe, Colaresi y Quinn, «Fightin' Words: Lexical Feature Selection and Evaluation for Identifying the Content of Political Conflict», Political Analysis, 2008).

```python
# Construccion de los tres bloques de features (densas, estructuradas, supervisadas in-fold).
t0 = time.time()
dense = extraer_textometria(train)
dense_test = extraer_textometria(test)
print(f"densas: {dense.shape} | {time.time() - t0:.0f}s")

t0 = time.time()
estructuradas = extraer_estructuradas(train)
estructuradas_test = extraer_estructuradas(test)
print(f"estructuradas: {estructuradas.shape} | {time.time() - t0:.0f}s")

t0 = time.time()
supervisadas = supervisadas_infold(train["text"], y, fold_idx)
supervisadas_te = supervisadas_test(train["text"], y, test["text"])
print(f"supervisadas (in-fold): {supervisadas.shape} | {time.time() - t0:.0f}s")

print("\nreferencia: 35 densas + 87 estructuradas + 73 supervisadas = 195 columnas del meta")
```

```python
# Figura 3.2: tamano de los bloques y varianza explicada por familia (resumen visual).
fig, ejes = plt.subplots(1, 2, figsize=(11, 3.4))
ejes[0].bar(["densas", "estructuradas", "supervisadas"], [35, 87, 73],
            color=["#2b6cb0", "#2f855a", "#ed8936"])
ejes[0].set_title("Tamano de cada bloque de features")
ejes[0].set_ylabel("columnas")
varianza = pd.DataFrame({"densas": dense.std(), "estructuradas": estructuradas.std(),
                         "supervisadas": supervisadas.std()})
nulos = float((varianza == 0).to_numpy().mean()) * 100
ejes[1].bar(["densas", "estructuradas", "supervisadas"],
            [float(dense.std().mean()), float(estructuradas.std().mean()),
             float(supervisadas.std().mean())], color=["#2b6cb0", "#2f855a", "#ed8936"])
ejes[1].set_title(f"Desviacion tipica media (columnas nulas: {nulos:.1f}%)")
plt.tight_layout()
plt.show()
```

## 3.4 Las fases PH y el meta-modelo (V5–V12)

Con las representaciones y features listas, la pregunta cambió de «¿cómo representar?» a «¿cómo
combinar?». El proyecto construyó **14 miembros diversos** — cada uno con una representación, un modelo o
una formulación distinta — y los combinó en un meta-modelo:

| # | Miembro | Representación | Modelo | CV documentada |
|---|---|---|---|---|
| 1 | E-013 | char_wb(3-6) 300k + menciones | LinearSVC C=0.2 (márgenes) | 0.5628 |
| 2 | E-016 | char_wb(3-6) 700k + menciones | LinearSVC C=0.2 (márgenes) | 0.5658 |
| 3 | E-009 | char300 + menciones | LogReg OvR C=4.0 | 0.5603 |
| 4 | E-011 | char300 + menciones | SGD log-loss | 0.5579 |
| 5 | E-010 | char300 + menciones | ComplementNB a=0.3 | 0.5370 |
| 6 | E-014 | char300 + menciones + hashtags | LinearSVC C=0.2 | 0.5644 |
| 7 | E-015o | char300 + menciones | Ridge ordinal (puntos medios) | 0.3973 |
| 8 | E-015s | ídem + softmax(-\|d\|/5) | distribución suave | — |
| 9 | E-017 | word(1-2) 300k | LogReg OvR C=4.0 | 0.5015 |
| 10 | E-019b | 35 densas + SVD(64) de char300 | MLP(256,128) | 0.4071 |
| 11 | E-021a | char300 + menciones | Frank & Hall (5 binarios) | 0.5527 |
| 12 | E-021b | char300 + menciones | Centroides Rocchio + softmax | 0.4017 |
| 13 | E-028a | char300 + menciones | 5 SVM por par adyacente | 0.7428 (fronteras) |
| 14 | ADV-05 | char300 + menciones | NB-SVM OvR | 0.5641 |

El meta-modelo final (V12) toma esas 14 salidas (78 columnas tras las transformaciones log), les añade
las 195 features de la sección 3.3 y las 6 columnas del NB-SVM de palabra (ADV-07; Wang y Manning, «Baselines and Bigrams: Simple, Good Sentiment and Topic Classification», ACL 2012; Mukherjee y Liu, «Improving Gender Classification of Blog Authors», EMNLP 2010), formando la **matriz
de 279 columnas**, y combina dos cabezas — una regresión logística y un MLP(64) — promediando sus
probabilidades. V12 alcanzó **CV 0.6005** (0.6077 en el holdout congelado) y **0.62182 en el
leaderboard**, el primer salto grande del proyecto.

```python
# Reconstruccion de los 14 miembros (OOF con los folds canonicos de la seccion 2.4).
t0 = time.time()
oofs_miembros = construir_miembros_oof(train["text"], y, fold_idx, dense)
print(f"14 miembros reconstruidos en {time.time() - t0:.0f}s")

documentados = {"e013": 0.5628, "e016": 0.5658, "e009": 0.5603, "e011": 0.5579, "e010": 0.5370,
                "e014": 0.5644, "e015_ordinal": 0.3973, "e015_softmax": None, "e017": 0.5015,
                "e019b": 0.4071, "e021a": 0.5527, "e021b": 0.4017, "e028a": None, "adv05": 0.5641}
print(f"\n{'miembro':14} {'CV reconstruida':>15} {'documentada':>12}")
for nombre, oof in oofs_miembros.items():
    if oof.ndim == 1:
        pred = np.argmax(softmax_ordinal(oof), axis=1)
    else:
        pred = np.argmax(oof, axis=1)
    cv_miembro = float(f1_score(clase, pred, average="macro"))
    referencia = documentados.get(nombre)
    texto_ref = f"{referencia:.4f}" if referencia else "(n/d)"
    print(f"{nombre:14} {cv_miembro:15.4f} {texto_ref:>12}")
```

```python
# ADV-07 (NB-SVM de palabra) y la matriz 279 del campeon.
t0 = time.time()
adv07 = np.zeros((len(y), 6), dtype=np.float32)
for fold in range(5):
    tr = np.flatnonzero(fold_idx != fold)
    va = np.flatnonzero(fold_idx == fold)
    adv07[va] = nb_svm_margins(train["text"], y, tr, va, WORD)
print(f"ADV-07 reconstruido en {time.time() - t0:.0f}s")

meta273 = build_features_273(oofs_miembros, densas35(dense),
                             estructuradas.to_numpy(float), supervisadas)
matriz = matriz_279(meta273, adv07)
print("meta 273:", meta273.shape, "| matriz 279:", matriz.shape, matriz.dtype)
assert matriz.shape == (169336, 279)
```

```python
# Mapa de aporte individual de los 14 miembros del meta (el meta-modelo V12 queda documentado en
# 0.6005; su reconstruccion costaba ~3.5 min y era una verificacion historica).
fig, eje = plt.subplots(figsize=(11, 3.6))
nombres = list(oofs_miembros)
valores = []
for nombre in nombres:
    oof = oofs_miembros[nombre]
    pred = np.argmax(softmax_ordinal(oof), axis=1) if oof.ndim == 1 else np.argmax(oof, axis=1)
    valores.append(float(f1_score(clase, pred, average="macro")))
eje.bar(nombres, valores, color="#2b6cb0", alpha=0.85, label="miembros (reconstruidos)")
eje.axhline(0.6005, color="#2f855a", linestyle="--", label="meta V12 (0.6005 documentado)")
eje.axhline(0.5344, color="#c53030", linestyle=":", label="baseline E-001 (0.5344)")
eje.set_ylabel("macro-F1 (CV agrupada)")
eje.set_title("Los 14 miembros del meta-modelo y su combinacion (V12)")
eje.legend(fontsize=8)
plt.xticks(rotation=45, fontsize=8, ha="right")
plt.tight_layout()
plt.show()
```
