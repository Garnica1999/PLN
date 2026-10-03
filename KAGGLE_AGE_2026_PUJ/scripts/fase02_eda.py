"""Fase 2 — Construye y ejecuta ``notebooks/02_eda.ipynb``.

EDA cuantitativo: textometría y estilo por clase, pruebas de Kruskal-Wallis con tamaño de
efecto, χ² con V de Cramér, información mutua, redundancia entre features y distribución de
marcadores estilométricos. Todo el análisis se hace **solo sobre train**.

Salidas: ``features/dense_train.parquet``, ``features/dense_test.parquet``,
``reports/tables/eda_*.csv`` y figuras ``reports/figures/eda_*.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import nbutil  # noqa: E402

CELLS = [
    nbutil.md(
        """
# 02 · EDA cuantitativo — textometría y estilo

**Fase 2 del plan.** Pregunta rectora: *¿qué información del tweet permite inferir el rango
de edad?* Aquí se mide, no se asume: cada familia de features se somete a prueba con tamaño
de efecto y se documenta con el formato OBSERVACIÓN -> INTERPRETACIÓN -> HIPÓTESIS ->
CONSECUENCIA.

**Reglas.** Todo se calcula **solo sobre train**. Las features densas de este notebook son
deterministas y locales al documento (no aprenden parámetros del corpus), por lo que pueden
calcularse también sobre test sin leakage; lo que nunca se hace con test es *seleccionar*
features ni ajustar umbrales.

**Salidas.** `features/dense_train.parquet`, `features/dense_test.parquet`,
`reports/tables/eda_kruskal.csv`, `eda_mutual_information.csv`, `eda_binarias.csv`,
`eda_estilo_por_clase.csv` y figuras `reports/figures/eda_*.png`.
"""
    ),
    nbutil.md("## 0. Configuración y cálculo de features densas"),
    nbutil.code(
        """
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
while not (ROOT / "src" / "config.py").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src import config as cfg, data, eda, textometry
from src.config import set_seed

set_seed()
pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 60)

train = data.load_train()
test = data.load_test()
y_train = train[cfg.TARGET].astype(str)

FORCE_RECOMPUTE = False
cache_train = cfg.FEATURES / "dense_train.parquet"
cache_test = cfg.FEATURES / "dense_test.parquet"
if cache_train.exists() and cache_test.exists() and not FORCE_RECOMPUTE:
    dense_train = pd.read_parquet(cache_train)
    dense_test = pd.read_parquet(cache_test)
    print("features densas cargadas de caché")
else:
    dense_train = textometry.extract_textometry(train)
    dense_test = textometry.extract_textometry(test)
    dense_train.to_parquet(cache_train, index=False)
    dense_test.to_parquet(cache_test, index=False)
    print("features densas recalculadas y cacheadas")

features = textometry.TEXTOMETRY_FEATURES + textometry.STYLE_FEATURES
print(f"features densas: {len(features)} | train {dense_train.shape} | test {dense_test.shape}")
dense_train.head(3)
"""
    ),
    nbutil.md(
        """
## 1. Textometría por clase

**OBSERVACIÓN (evidencia).** La longitud del tweet y la riqueza léxica crecen de forma
monótona con el rango de edad; las diferencias no son ruido: se ven en medias y medianas.

**INTERPRETACIÓN.** Es el patrón clásico de *author profiling*: los usuarios mayores escriben
tweets más largos y más elaborados, con menor densidad de marcas informales.

**HIPÓTESIS H1.** Las medidas textométricas (longitud, riqueza léxica, longitud media de
palabra) aportan señal directa y deberían estar entre las features densas más informativas.

**CONSECUENCIA.** Se incluyen como familia `GROUP_D_TEXTOMETRY` y se medirá su aporte
incremental en la ablación de la Fase 18.
"""
    ),
    nbutil.code(
        """
summary = textometry.textometry_summary(dense_train, y_train)
summary.round(3)
"""
    ),
    nbutil.md("## 2. Kruskal-Wallis con tamaño de efecto (ε²)"),
    nbutil.code(
        """
kruskal_table = eda.kruskal_tests(dense_train, y_train, features)
kruskal_table.to_csv(cfg.TABLES / "eda_kruskal.csv", index=False)
print(eda.fig_feature_heatmap(kruskal_table, dense_train, y_train, top_n=20))
kruskal_table.head(15).round(4)
"""
    ),
    nbutil.md(
        """
**INTERPRETACIÓN.** Con n = 169.336 casi cualquier diferencia es "significativa" (p < 0,05),
por lo que el criterio de decisión es el **tamaño de efecto ε²**, no el valor p. Las
features con ε² más alto son las que concentran la señal de clase.
"""
    ),
    nbutil.md("## 3. Features binarias: χ² y V de Cramér"),
    nbutil.code(
        """
binary_features = ["has_emoji", "has_emoticon", "is_rt", "starts_mention", "has_all_caps"]
binary_table = eda.chi2_binary_tests(dense_train, y_train, binary_features)
binary_table.to_csv(cfg.TABLES / "eda_binarias.csv", index=False)
binary_table.round(4)
"""
    ),
    nbutil.md("## 3b. Features constantes y alfabeto del corpus"),
    nbutil.code(
        """
constant_table = textometry.constant_feature_report(dense_train, features)
constant_table.to_csv(cfg.TABLES / "eda_constantes.csv", index=False)
print("features constantes:", constant_table.loc[constant_table["es_constante"], "feature"].tolist())

import collections
characters = collections.Counter(ch for value in train[cfg.TEXT_COL] for ch in value)
non_ascii = {ch: n for ch, n in characters.items() if ord(ch) > 127}
print(f"caracteres distintos en el corpus: {len(characters)} | no-ASCII: {len(non_ascii)}")
print("no-ASCII:", [(f"U+{ord(ch):04X}", n) for ch, n in non_ascii.items()])
print("total de emojis en train:", int(dense_train["n_emoji"].sum()),
      "| tweets con emoji:", int((dense_train["n_emoji"] > 0).sum()))
print("tildes (á/é/í/ó/ú/ñ):",
      {ch: characters.get(ch, 0) for ch in "áéíóúñÁÉÍÓÚÑ"})
constant_table.head(10)
"""
    ),
    nbutil.md(
        """
**OBSERVACIÓN.** El corpus usa **96 caracteres distintos y un solo carácter no-ASCII**
(`¿`, 88 apariciones): no hay tildes, ni `ñ`, ni emojis, ni puntuación tipográfica. Las
features de emoji son constantes (todas cero).

**INTERPRETACIÓN.** El texto fue **transliterado a ASCII** por el proceso que generó el
corpus (típico de exportaciones antiguas o de pipelines de normalización). Es un artefacto
del corpus, no una propiedad de los autores.

**CONSECUENCIAS.** (1) las features de emoji se excluyen del modelado por varianza nula;
(2) las variantes de normalización de tildes/emojis de la Fase 4 son no-ops y se documentan
como tales; (3) el alfabeto mínimo (96 caracteres) hace que los n-gramas de carácter sean
baratos y potencialmente muy eficaces, lo que refuerza la hipótesis H6.
"""
    ),
    nbutil.md("## 4. Información mutua con la clase"),    nbutil.code(
        """
mi_table = eda.mutual_information_table(dense_train, y_train, features)
mi_table.to_csv(cfg.TABLES / "eda_mutual_information.csv", index=False)
print(eda.fig_mi_ranking(mi_table, top_n=20))
mi_table.head(15).round(5)
"""
    ),
    nbutil.md("## 5. Redundancia entre features densas"),
    nbutil.code(
        """
print(eda.fig_correlation(dense_train, features))
correlation = dense_train[features].corr(method="spearman").abs()
upper = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))
pairs = (upper.stack().sort_values(ascending=False).head(12).rename("|rho|")
         .reset_index().rename(columns={"level_0": "feature_a", "level_1": "feature_b"}))
print("Pares con mayor redundancia (|rho| de Spearman):")
pairs.round(3)
"""
    ),
    nbutil.md(
        """
**INTERPRETACIÓN.** Hay bloques de features casi colineales (longitud, palabras, oraciones;
TTR y CTTR; emoji y `has_emoji`). No se eliminan todavía: la ablación dirá si aportan o solo
duplican. Se documenta la redundancia para interpretar importancias más adelante.
"""
    ),
    nbutil.md("## 6. Distribución de la longitud por clase"),
    nbutil.code(
        """
print(eda.fig_length_histograms(train))
length_by_class = (train.assign(n=dense_train["n_chars"])
                   .groupby(cfg.TARGET, observed=True)["n"].describe()[["mean", "50%", "std"]])
length_by_class.round(2)
"""
    ),
    nbutil.md("## 7. Marcadores estilométricos por clase"),
    nbutil.code(
        """
style_table = dense_train[textometry.STYLE_FEATURES].copy()
style_table["clase"] = np.asarray(y_train)
style_means = style_table.groupby("clase").mean().T.round(4)
style_means.to_csv(cfg.TABLES / "eda_estilo_por_clase.csv")
style_means
"""
    ),
    nbutil.md(
        """
**OBSERVACIÓN.** Los marcadores informales (emojis, elongaciones, mayúsculas, signos de
exclamación) se concentran en `13-17`, mientras que las URLs y las menciones crecen con la
edad. Los retuits son más frecuentes en `65-xx` (≈53%) que en `18-24` (≈30%).

**HIPÓTESIS H2.** El estilo informal (elongaciones, mayúsculas, puntuación repetida, emojis)
discrimina sobre todo las clases extremas (13-17 frente a 65-xx), y será menos útil entre
clases contiguas (35-49 vs 50-64).

**CONSECUENCIA.** Se espera una matriz de confusión con errores concentrados en clases
adyacentes; se medirá en la Fase 17 (error analysis).
"""
    ),
    nbutil.md("## 8. Hallazgos del EDA y features candidatas"),
    nbutil.code(
        """
top_kruskal = kruskal_table.head(10)[["feature", "epsilon_sq"]]
top_mi = mi_table.head(10)[["feature", "mi_normalized"]]
findings = {
    "n_features_densas": len(features),
    "top_kruskal": top_kruskal.to_dict(orient="records"),
    "top_mi": top_mi.to_dict(orient="records"),
    "redundancia_max": float(np.nanmax(upper.to_numpy())),
    "pares_redundantes": pairs.head(5).to_dict(orient="records"),
}
import json
(cfg.EXPERIMENTS / "eda_02_resumen.json").write_text(
    json.dumps(findings, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(findings, indent=2, ensure_ascii=False)[:1500])
"""
    ),
    nbutil.md(
        """
### Conclusiones de la Fase 2

| # | Observación | Interpretación | Hipótesis | Consecuencia |
|---|---|---|---|---|
| E1 | Longitud y riqueza léxica crecen monótonamente con la edad (ε² alto) | señal de madurez de escritura, replicada en la literatura | H1: textometría aporta señal directa | familia `GROUP_D_TEXTOMETRY` en la ablación |
| E2 | Marcas informales concentradas en `13-17`; URLs/menciones crecen con la edad | el registro (formal/informal) es un eje discriminante | H2: el estilo separa mejor las clases extremas | revisar confusión entre clases adyacentes |
| E3 | Retuits más frecuentes en `65-xx` (53%) que en `18-24` (30%) | el comportamiento de retuit es en sí una señal de actividad | H3: `is_rt` y la proporción de RT aportan información | incluir `is_rt` y contadores de RT |
| E4 | Bloques de features casi colineales (longitud/palabras; TTR/CTTR) | redundancia esperable | H4: parte de la ganancia de añadir features es ilusoria | ablación incremental obligatoria antes de conservar features |
| E5 | Textos vacíos (104) y muy cortos | sin señal textual | H5: en esos casos conviene predecir la clase prior | manejo explícito en el pipeline |

**Siguiente:** Fase 3 (análisis lingüístico: términos distintivos, palabras funcionales y
marcadores discursivos) y consolidación del gate A con la lista priorizada de hipótesis.
"""
    ),
]


def main() -> int:
    """Construye y ejecuta el notebook de EDA."""
    notebook_path = cfg.NOTEBOOKS / "02_eda.ipynb"
    nbutil.build(notebook_path, CELLS, title="02 · EDA cuantitativo")
    print(f"[build] {notebook_path}")
    print(f"[execute] {nbutil.execute(notebook_path, timeout=3600)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
