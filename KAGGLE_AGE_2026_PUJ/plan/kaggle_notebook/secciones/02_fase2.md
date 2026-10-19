# FASE 2 — Comprensión de los datos

## 2.1 Auditoría de integridad del corpus

Antes de cualquier modelado, el proyecto auditó la integridad del corpus: formas, tipos, valores nulos,
duplicados exactos, el relleno de ancho fijo de la exportación (el campo de texto viene con espacios a la
derecha hasta 140 caracteres — un artefacto, no señal) y la normalización a ASCII. Esta auditoría fijó
decisiones que atraviesan todo el pipeline: limpiar el relleno con `strip` antes de vectorizar, no
depender de tildes ni eñes, y tratar los duplicados con cuidado (lo que motiva la sección 2.4).

```python
# Auditoria de integridad: formas, nulos, relleno, ASCII y duplicados exactos.
print("=== forma y columnas ===")
print("train:", train.shape, list(train.columns))
print("test :", test.shape, list(test.columns))
print("nulos en train:", int(train.isna().sum().sum()),
      "| nulos en test:", int(test.isna().sum().sum()))

anchos = train["text"].str.len()
print("\n=== longitudes (train) ===")
print("min:", int(anchos.min()), "| mediana:", int(anchos.median()), "| max:", int(anchos.max()))
print(f"ancho exacto de 140 (relleno de exportacion): {float((anchos == 140).mean()) * 100:.1f}%")

no_ascii = float(train["text"].map(lambda t: any(ord(c) > 127 for c in t)).mean())
print(f"filas con caracteres no ASCII: {no_ascii * 100:.3f}%")

print("\n=== duplicados e identificadores ===")
print("textos duplicados exactos en train:", int(train["text"].duplicated().sum()))
print("ids unicos (train/test):", bool(train["id"].is_unique), bool(test["id"].is_unique))
print("clases presentes:", sorted(train["age_range"].unique().tolist()))
```

```python
# Auditoria ampliada: vacios, ruido de etiqueta, id como proxy y signo de apertura.
vacios_train = int((train["text"].str.strip() == "").sum())
vacios_test = int((test["text"].str.strip() == "").sum())
print(f"textos vacios: train {vacios_train} | test {vacios_test}")

normalizado = (train["text"].fillna("").astype(str).str.lower()
               .str.replace(r"\s+", " ", regex=True).str.strip())
etiquetas_por_texto = train.assign(_norm=normalizado).groupby("_norm")["age_range"].nunique()
textos_contradictorios = etiquetas_por_texto[etiquetas_por_texto > 1].index
ruido = int(normalizado.isin(textos_contradictorios).sum())
print(f"ruido de etiqueta: {len(textos_contradictorios)} textos con {ruido} filas "
      f"({ruido / len(train) * 100:.2f}%) con etiquetas contradictorias")

apariciones_apertura = int(train["text"].str.count("\u00bf").sum() + test["text"].str.count("\u00bf").sum())
print(f"signo de apertura (corpus completo): {apariciones_apertura} apariciones (unico caracter no ASCII)")

deciles_id = pd.qcut(train["id"], 10, labels=False)
tabla_id = pd.crosstab(deciles_id, train["age_range"], normalize="index")
desviacion = float((tabla_id - 1 / len(CLASSES)).abs().max().max()) * 100
print(f"id como proxy de clase: desviacion maxima frente a 1/6 de {desviacion:.2f} pp")

```

## 2.2 Análisis exploratorio y textometría

El análisis exploratorio respondió tres preguntas que condicionaron la representación: ¿qué tan largos son
los textos?, ¿qué vocabulario los distingue entre rangos? y ¿qué tan informativa es la forma del texto
(emoticonos, mayúsculas, puntuación; Flekova y Gurevych, «Can We Hide in the Web? Large Scale Simultaneous Age and Gender Author Profiling in Social Media», PAN@CLEF 2013)? Las figuras muestran que la distribución de longitudes es muy
similar entre train y test (la representación puede confiar en la forma) y que cada rango de edad tiene
marcadores léxicos propios (Nguyen, Gravel, Trieschnigg y Meder, «How Old Do You Think I Am? A Study of Language and Age in Twitter», ICWSM 2013) — jerga juvenil en los extremos bajos, temas de actualidad y registro formal en
los altos — lo que justifica la representación por n-gramas de palabra y de carácter que se adopta en la
Fase 3.

```python
# Figura 2.1: distribucion de longitudes (caracteres y palabras).
fig, ejes = plt.subplots(1, 2, figsize=(11, 3.6))
largos_train = train["text"].str.len()
largos_test = test["text"].str.len()
ejes[0].hist(largos_train, bins=60, color="#2b6cb0", alpha=0.85, label="train")
ejes[0].hist(largos_test, bins=60, color="#ed8936", alpha=0.55, label="test")
ejes[0].set_title("Longitud en caracteres (train vs test)")
ejes[0].set_xlabel("caracteres")
ejes[0].set_ylabel("tuits")
ejes[0].legend()
ejes[1].hist(train["text"].str.split().str.len(), bins=40, color="#2b6cb0", alpha=0.85)
ejes[1].set_title("Longitud en palabras (train)")
ejes[1].set_xlabel("palabras")
plt.tight_layout()
plt.show()
```

```python
# Figura 2.2: terminos mas frecuentes de cada rango de edad.
from collections import Counter

fig, ejes = plt.subplots(2, 3, figsize=(13, 6))
for posicion, etiqueta in enumerate(CLASSES):
    eje = ejes[posicion // 3][posicion % 3]
    palabras = Counter()
    for texto in train.loc[train["age_range"] == etiqueta, "text"]:
        palabras.update(token for token in texto.split() if len(token) >= 4)
    top = palabras.most_common(8)[::-1]
    eje.barh([t for t, _ in top], [c for _, c in top], color="#2b6cb0", alpha=0.85)
    eje.set_title(f"rango {etiqueta}", fontsize=10)
    eje.tick_params(labelsize=8)
plt.suptitle("Tokens frecuentes por rango (longitud >= 4)", y=1.0)
plt.tight_layout()
plt.show()
```

```python
# Figura 2.3: distribucion de clases y balance.
conteos = train["age_range"].value_counts().reindex(CLASSES)
fig, eje = plt.subplots(figsize=(8, 3.2))
eje.bar(conteos.index.astype(str), conteos.values, color="#2b6cb0")
for posicion, valor in enumerate(conteos.values):
    eje.text(posicion, valor + 150, f"{valor:,}", ha="center", fontsize=9)
eje.set_title("Tuits por rango de edad (train)")
eje.set_ylabel("tuits")
plt.tight_layout()
plt.show()
print(f"razon mayoritaria/minoritaria: {conteos.max() / conteos.min():.2f}x")
from sklearn.metrics import f1_score as _f1_suelo
suelo_mayoritaria = _f1_suelo(train["age_range"], [conteos.idxmax()] * len(train),
                              average="macro")
print(f"suelos de referencia: azar uniforme {1 / 6:.4f} | clase mayoritaria {suelo_mayoritaria:.5f}")
```

## 2.2b Textometría y contraste por edad

El corpus admite una caracterización cuantitativa más allá de las longitudes: riqueza léxica, densidad de
menciones y enlaces, mayúsculas y puntuación. La tabla contrasta diez de esas variables entre los seis
rangos con la prueba de Kruskal–Wallis (Kruskal y Wallis, «Use of Ranks in One-Criterion Variance
Analysis», Journal of the American Statistical Association, 1952) y reporta el tamaño del efecto
ε² = (H − k + 1)/(n − k); el barrido completo de 38 variables quedó documentado en
`reports/tables/eda_kruskal.csv`. La lectura coincide con la del informe: la longitud y la riqueza léxica
**crecen con la edad** (ε² ≈ 0.13 para `n_chars`), mientras que las marcas de oralidad juvenil
(emoticonos, alargamientos) se concentran en el extremo bajo. La mediana de longitud pasa de 73
caracteres en `13-17` a 111 en `65-xx`.

```python
# Figura 2.5: textometria por rango y contraste de Kruskal-Wallis.
from collections import Counter
from scipy.stats import kruskal

texto_train = train["text"]
palabras_train = texto_train.str.split()
n_palabras = palabras_train.str.len()
n_unicas = palabras_train.map(lambda lista: len(set(lista)))
cttr = n_unicas / np.sqrt(n_palabras.clip(lower=1))
entropia = palabras_train.map(
    lambda lista: float(-sum((cantidad / len(lista)) * np.log2(cantidad / len(lista))
                             for cantidad in Counter(lista).values())) if lista else 0.0)
variables_texto = {
    "n_chars": texto_train.str.len(),
    "n_words": n_palabras,
    "n_unique_words": n_unicas,
    "cttr": cttr,
    "lexical_entropy": entropia,
    "n_mention": texto_train.str.count(r"@\w+"),
    "n_url": texto_train.str.count(r"https?://\S+|www\.\S+"),
    "n_hashtag": texto_train.str.count(r"#\w+"),
    "all_caps_count": texto_train.str.count(r"\b[A-Z]{2,}\b"),
    "n_excl": texto_train.str.count(r"!"),
}
filas = []
for nombre, serie in variables_texto.items():
    grupos = [serie[train["age_range"] == clase].to_numpy() for clase in CLASSES]
    estadistico, p_valor = kruskal(*grupos)
    epsilon2 = (estadistico - len(CLASSES) + 1) / (len(train) - len(CLASSES))
    filas.append((nombre, estadistico, p_valor, epsilon2))
tabla_textometria = pd.DataFrame(filas, columns=["variable", "H", "p", "epsilon2"])
print(tabla_textometria.sort_values("H", ascending=False).to_string(index=False))
print(f"\nmediana de caracteres 13-17: {texto_train[train['age_range'] == '13-17'].str.len().median():.0f}"
      f" | 65-xx: {texto_train[train['age_range'] == '65-xx'].str.len().median():.0f}")

fig, eje = plt.subplots(figsize=(8, 3.2))
datos = [texto_train[train["age_range"] == clase].str.len().to_numpy() for clase in CLASSES]
eje.boxplot(datos, showfliers=False)
eje.set_xticks(range(1, len(CLASSES) + 1))
eje.set_xticklabels([str(clase) for clase in CLASSES])
eje.set_xlabel("rango de edad")
eje.set_ylabel("caracteres (sin relleno)")
eje.set_title("Longitud por rango: mediana y dispersion")
plt.tight_layout()
plt.show()
```

## 2.3 La estructura de los seis rangos

Los seis rangos son **ordinales**: `13-17 < 18-24 < 25-34 < 35-49 < 50-64 < 65-xx`. Esa estructura tiene
dos consecuencias prácticas que el proyecto explotó (y midió): primero, los errores graves (confundir
`13-17` con `65-xx`) son raros, mientras que las confusiones entre **rangos adyacentes** concentran la
mayor parte del error; segundo, la distancia entre clases admite formulaciones ordinales (regresión sobre
puntos medios, clasificadores acumulativos, pérdidas de transporte) (Nguyen, Smith y Rosé, «Author Age Prediction from Text using Linear Regression», LaTeCH@ACL-HLT 2011; Frank y Hall, «A Simple Approach to Ordinal Classification», ECML 2001; Frogner, Zhang, Mobahi, Araya y Poggio, «Learning with a Wasserstein Loss», NeurIPS 2015) que se exploran en la Fase 4. El
balance de clases es moderado (la razón mayoritaria/minoritaria ronda 1.06x), por lo que el desbalanceo no
fue el problema central — el problema fue la **frontera** entre rangos vecinos y el ruido intrínseco de la
tarea.

## 2.4 La fuga por near-duplicados y los folds canónicos

El corpus contiene **near-duplicados** (Kaufman, Rosset, Perlich y Stitelman, «Leakage in Data Mining: Formulation, Detection, and Avoidance», ACM Transactions on Knowledge Discovery from Data, 2012): textos idénticos o casi idénticos (retuits, copias con un emoji
distinto, el mismo chiste repetido) que aparecen tanto en entrenamiento como en prueba. Si la validación
cruzada se hace con folds aleatorios, un texto de validación puede tener su gemelo en entrenamiento: el
modelo lo memoriza y la validación **sobreestima** el desempeño real. Este fenómeno se midió y quedó
documentado como la decisión de protocolo más importante del proyecto: **la validación agrupada por
clústeres de near-duplicados**, donde un clúster entero cae en un solo fold. Así, ningún fold de
validación contiene un texto cuyo near-duplicado esté en el fold de entrenamiento.

Los clústeres se construyen con un procedimiento determinista (normalización del texto, comparación por
similitud de Jaccard (Jaccard, «The Distribution of the Flora in the Alpine Zone», New Phytologist, 1912) de tokens con *blocking* por tokens raros, unión de pares similares). La partición de
5 folds agrupados es el **protocolo congelado** del proyecto: todos los experimentos de este notebook se
evalúan con exactamente esos folds (embebidos como constante para inmunizar el protocolo frente a cambios
de versión de las librerías).

```python
# Figura 2.4: reconstruccion de los clusteres de near-duplicados (protocolo congelado).
t0 = time.time()
textos_todo = pd.concat([train["text"], test["text"]], ignore_index=True)
clusters = construir_clusters(textos_todo)
print(f"clusters: {clusters['cluster'].nunique():,} en {len(clusters):,} textos "
      f"({time.time() - t0:.1f}s)")

cluster_de = clusters["cluster"].to_numpy()
n_train = len(train)
pares = pd.Series(cluster_de[:n_train])
repetidos = pares.duplicated(keep=False)
exactos_test = float(test["text"].isin(set(train["text"])).mean())
compartidos_test = float(pd.Series(cluster_de[n_train:]).isin(
    set(cluster_de[:n_train])).mean())

fig, ejes = plt.subplots(1, 2, figsize=(11, 3.4))
tamanos = pd.Series(cluster_de).value_counts().value_counts().sort_index()
ejes[0].bar(tamanos.index.astype(str), tamanos.values, color="#2b6cb0")
ejes[0].set_title("Tamano de los clusters de near-duplicados")
ejes[0].set_xlabel("textos por cluster")
ejes[0].set_ylabel("numero de clusters")
ejes[0].tick_params(labelsize=8)
ejes[1].bar(["exactos test~train", "comparten cluster", "sin gemelo"],
            [exactos_test * 100, compartidos_test * 100, (1 - compartidos_test) * 100],
            color=["#c53030", "#ed8936", "#2b6cb0"])
ejes[1].set_title("Test con gemelo en train (%)")
ejes[1].set_ylabel("porcentaje")
plt.tight_layout()
plt.show()
print(f"train en clusters de tamano > 1: {float(repetidos.mean()) * 100:.1f}%")
print(f"test con duplicado exacto en train: {exactos_test * 100:.2f}%")
print(f"test que comparte cluster con train: {compartidos_test * 100:.2f}%")
```

```python
# Verificacion del protocolo de folds: la asignacion embebida se regenera con la particion canonica.
from sklearn.model_selection import StratifiedGroupKFold

def folds_desde_clusters(y_etiquetas, grupos):
    """Reproduce el fold_idx canonico: StratifiedGroupKFold(shuffle=True, random_state=42)."""
    divisor = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    indice = np.zeros(len(y_etiquetas), dtype=np.int64)
    for numero, (_, validacion) in enumerate(
            divisor.split(np.zeros(len(y_etiquetas)), y_etiquetas, groups=grupos)):
        indice[validacion] = numero
    return indice


regenerado = folds_desde_clusters(y, cluster_de[:n_train])
coincidencia = float((regenerado == fold_idx).mean())
print(f"coincidencia folds regenerados vs embebidos: {coincidencia:.8f}")
if coincidencia == 1.0:
    print("Protocolo verificado: la particion embebida es exactamente la canonica del proyecto.")
else:
    print("Aviso: la version local de scikit-learn genera una particion distinta; "
          "se conserva la particion embebida (protocolo congelado).")
print("conteo por fold:", np.bincount(fold_idx).tolist())
```

```python
# Suelo de memorizacion pura: particion ingenua frente a la particion agrupada congelada.
from sklearn.model_selection import StratifiedKFold

SENTINELA = "__no_cubierto__"
etiquetas = train["age_range"].to_numpy()
cluster_train = cluster_de[:n_train]


def oof_memorizacion(particiones):
    """Etiqueta mayoritaria del gemelo de cluster; las filas sin gemelo reciben un centinela."""
    prediccion = np.empty(len(etiquetas), dtype=object)
    cubiertas = 0
    for entrenamiento, validacion in particiones:
        etiquetas_por_cluster = {}
        for cluster, etiqueta in zip(cluster_train[entrenamiento], etiquetas[entrenamiento]):
            etiquetas_por_cluster.setdefault(cluster, []).append(etiqueta)
        for fila in validacion:
            gemelos = etiquetas_por_cluster.get(cluster_train[fila])
            if gemelos:
                prediccion[fila] = Counter(gemelos).most_common(1)[0][0]
                cubiertas += 1
            else:
                prediccion[fila] = SENTINELA
    return prediccion, cubiertas / len(etiquetas)


def macro_f1(etiquetas_reales, prediccion):
    return f1_score(etiquetas_reales, prediccion, average="macro", labels=CLASSES, zero_division=0)


ingenuas = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(cluster_train, etiquetas))
pred_ingenua, cobertura_ingenua = oof_memorizacion(ingenuas)
agrupadas = [(np.flatnonzero(fold_idx != k), np.flatnonzero(fold_idx == k)) for k in range(5)]
pred_agrupada, cobertura_agrupada = oof_memorizacion(agrupadas)

print("SUELO DE MEMORIZACION PURA (etiqueta del gemelo; sin gemelo = centinela)")
print(f"  particion estratificada ingenua : macro-F1 {macro_f1(etiquetas, pred_ingenua):.4f}"
      f" | cobertura {cobertura_ingenua * 100:.2f}%  (documentado: 0.0608 y 3.99%)")
print(f"  particion agrupada (usada)      : macro-F1 {macro_f1(etiquetas, pred_agrupada):.4f}"
      f" | cobertura {cobertura_agrupada * 100:.2f}%  (documentado: 0.0000 y 0.00%)")
print("  lectura: la validacion agrupada elimina por completo la ventaja de memorizar near-duplicados.")

```

## 2.5 Construcción del conjunto de datos de modelado

El corpus crudo no se usa directamente: el modelado consume una **matriz de 279 columnas** construida en
siete pasos, que se resumen aquí porque son el contrato que comparten todos los experimentos del notebook.

1. **Corpus y limpieza.** Se cargan `age_train.csv` (169.336 filas) y `age_test.csv` (112.892) con
   separador `;`; se recorta el relleno de ancho fijo (`strip`), se conserva la transliteración ASCII y se
   tratan los textos vacíos como cadenas vacías. La etiqueta es `age_range`, con seis clases.
2. **Particiones congeladas.** Los cinco folds agrupados por clúster de near-duplicados (sección 2.4) se
   embeben como constante: ninguna fila de validación comparte clúster con el entrenamiento.
3. **Representación.** TF-IDF de palabras (1-2) y de carácter `char_wb` (3-6), ajustado dentro de cada
   fold con el vocabulario, `min_df` y `max_features` de la receta documentada.
4. **Miembros (78 columnas).** Catorce clasificadores individuales (tabla de la sección 3.4) generan
   probabilidades o márgenes fuera de fold; se aplica `log` a las columnas indicadas.
5. **Features densas (35).** Textometría estilométrica por documento: longitudes, emoticonos,
   alargamientos, puntuación repetida, mayúsculas, dígitos, entropías y ratios de compresión.
6. **Features estructuradas (87).** Seis familias: distribución de longitudes, distribución de caracteres,
   riqueza léxica, forma de palabra, posición y entropías de puntuación.
7. **Features supervisadas (73) y ADV-07 (6).** Perfil de clase por entidad (menciones, hashtags, fuente
   de retuit), rareza léxica y léxico log-odds construidos **dentro de cada fold**, más seis márgenes
   NB-SVM de palabra.

La tabla fija el inventario de variables; la construcción efectiva y sus aserciones de forma se ejecutan en
la Fase 3 (secciones 3.3 y 3.4), donde la matriz se ensambla una sola vez
(`assert matriz.shape == (169336, 279)`) y todos los modelos posteriores la consumen.

| Bloque | Columnas | Contenido y ejemplos |
|---|---:|---|
| Miembros | 78 | Probabilidades o márgenes fuera de fold de 14 clasificadores (log si corresponde) |
| Densas | 35 | Longitudes, emoticonos, alargamientos, puntuación repetida, mayúsculas, dígitos, entropías, ratios de compresión |
| Estructuradas | 87 | Distribución de longitudes, de caracteres, riqueza léxica, forma de palabra, posición, entropías de puntuación |
| Supervisadas | 73 | Perfil de clase por entidad, rareza léxica y log-odds por clase (in-fold) |
| ADV-07 | 6 | Márgenes NB-SVM uno-vs-rest de palabra |
| **Total** | **279** | 273 del meta + 6 del canal de palabra |

Este resultado cierra la fase de comprensión de datos con el protocolo listo: corpus auditado, EDA hecho,
estructura ordinal entendida y partición honesta de validación verificada. Todo lo que sigue se evalúa con
estos folds.
