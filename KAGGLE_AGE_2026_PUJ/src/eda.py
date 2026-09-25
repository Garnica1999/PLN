"""Análisis exploratorio y pruebas estadísticas para la Fase 2-3.

Incluye: comparación de features por clase, pruebas de Kruskal-Wallis con tamaño de efecto,
χ² con V de Cramér para features binarias, información mutua, log-odds con prior Dirichlet
(Monroe et al. 2008) para términos distintivos por clase, y las figuras del EDA.

Nota metodológica: todo el análisis de esta fase se hace **solo sobre train**.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.feature_extraction.text import CountVectorizer  # noqa: E402
from sklearn.feature_selection import mutual_info_classif  # noqa: E402

from src import config as cfg  # noqa: E402
from src.textometry import SPANISH_STOPWORDS  # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "font.size": 9})


# --------------------------------------------------------------------------------------
# Estadística descriptiva e inferencial
# --------------------------------------------------------------------------------------
def describe_by_class(features: pd.DataFrame, target: pd.Series,
                      columns: list[str]) -> pd.DataFrame:
    """Media y desviación de cada feature por clase."""
    working = features[columns].copy()
    working["_class"] = np.asarray(target.astype(str))
    grouped = working.groupby("_class")
    means = grouped.mean().T
    means.columns = [f"media_{column}" for column in means.columns]
    stds = grouped.std().T
    stds.columns = [f"std_{column}" for column in stds.columns]
    return pd.concat([means, stds], axis=1)


def kruskal_tests(features: pd.DataFrame, target: pd.Series,
                  columns: list[str]) -> pd.DataFrame:
    """Kruskal-Wallis por feature con tamaño de efecto ε².

    ε² = (H − k + 1) / (n − k): proporción de la variabilidad atribuible a la clase.
    Se reporta el valor p, pero la decisión se toma por **tamaño de efecto** (p con n grande
    es significativo casi siempre y no implica relevancia práctica).
    """
    labels = np.asarray(target.astype(str))
    rows: list[dict] = []
    n = len(labels)
    k = len(cfg.CLASSES)
    for column in columns:
        values = [features[column].to_numpy()[labels == label] for label in cfg.CLASSES]
        statistic, p_value = stats.kruskal(*values)
        epsilon_sq = (statistic - k + 1) / (n - k)
        rows.append({"feature": column, "H": statistic, "p_value": p_value,
                     "epsilon_sq": epsilon_sq,
                     "mean_13_17": float(np.mean(values[0])),
                     "mean_65_xx": float(np.mean(values[-1]))})
    result = pd.DataFrame(rows).sort_values("epsilon_sq", ascending=False)
    return result.reset_index(drop=True)


def chi2_binary_tests(features: pd.DataFrame, target: pd.Series,
                      columns: list[str]) -> pd.DataFrame:
    """χ² y V de Cramér para features binarias contra la clase."""
    labels = np.asarray(target.astype(str))
    rows: list[dict] = []
    for column in columns:
        table = pd.crosstab(features[column].to_numpy(), labels)
        if table.shape[0] < 2:
            continue
        chi2, p_value, _, _ = stats.chi2_contingency(table)
        n = table.to_numpy().sum()
        cramers_v = float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))
        rows.append({"feature": column, "chi2": chi2, "p_value": p_value,
                     "cramers_v": cramers_v})
    return pd.DataFrame(rows).sort_values("cramers_v", ascending=False).reset_index(drop=True)


def mutual_information_table(features: pd.DataFrame, target: pd.Series,
                             columns: list[str], seed: int = cfg.SEED) -> pd.DataFrame:
    """Información mutua de cada feature con la clase (normalizada por H(y))."""
    labels = np.asarray(target.astype(str))
    matrix = features[columns].to_numpy(dtype=float)
    mi = mutual_info_classif(matrix, labels, discrete_features=False, random_state=seed)
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = float(-(probabilities * np.log2(probabilities)).sum())
    table = pd.DataFrame({"feature": columns, "mutual_info": mi})
    table["mi_normalized"] = table["mutual_info"] / entropy
    return table.sort_values("mutual_info", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Términos distintivos por clase (log-odds con prior Dirichlet)
# --------------------------------------------------------------------------------------
def top_terms_logodds(texts: pd.Series, target: pd.Series, classes: list[str] | None = None,
                      min_count: int = 10, alpha0: float = 500.0, top_k: int = 25,
                      remove_stopwords: bool = True,
                      token_pattern: str = r"(?u)\b\w\w+\b") -> pd.DataFrame:
    """Términos más distintivos de cada clase con log-odds y prior Dirichlet informativo.

    Implementa el estadístico z de Monroe, Colaresi & Quinn (2008) comparando cada clase
    contra el resto del corpus de entrenamiento.

    Parameters
    ----------
    texts : pandas.Series
        Textos normalizados (minúsculas, sin URLs/menciones).
    target : pandas.Series
        Etiquetas.
    min_count : int
        Frecuencia documental mínima para entrar al vocabulario.
    alpha0 : float
        Peso del prior Dirichlet informativo.
    top_k : int
        Número de términos por clase en la salida.

    Returns
    -------
    pandas.DataFrame
        ``clase, rank, termino, n_clase, n_resto, log_odds, z``.
    """
    classes = classes or cfg.CLASSES
    stop_words = sorted(SPANISH_STOPWORDS) if remove_stopwords else None
    vectorizer = CountVectorizer(lowercase=False, token_pattern=token_pattern,
                                 min_df=min_count, stop_words=stop_words, dtype=np.int32)
    matrix = vectorizer.fit_transform(texts.fillna(""))
    vocabulary = vectorizer.get_feature_names_out()
    labels = np.asarray(target.astype(str))
    total_counts = np.asarray(matrix.sum(axis=0)).ravel().astype(float)
    total = total_counts.sum()
    prior = alpha0 * total_counts / total

    rows: list[dict] = []
    for label in classes:
        mask = labels == label
        count_class = np.asarray(matrix[mask].sum(axis=0)).ravel().astype(float)
        count_rest = total_counts - count_class
        n_class, n_rest = count_class.sum(), count_rest.sum()
        log_odds = (np.log((count_class + prior) / (n_class + alpha0 - count_class - prior))
                    - np.log((count_rest + prior) / (n_rest + alpha0 - count_rest - prior)))
        variance = 1.0 / (count_class + prior) + 1.0 / (count_rest + prior)
        z_score = log_odds / np.sqrt(variance)
        order = np.argsort(-z_score)[:top_k]
        for rank, index in enumerate(order, start=1):
            rows.append({
                "clase": label, "rank": rank, "termino": vocabulary[index],
                "n_clase": int(count_class[index]), "n_resto": int(count_rest[index]),
                "log_odds": float(log_odds[index]), "z": float(z_score[index]),
            })
    return pd.DataFrame(rows)


def function_word_profile(texts: pd.Series, target: pd.Series) -> pd.DataFrame:
    """Perfil de palabras funcionales y marcadores discursivos por clase.

    Usa una lista cerrada de marcadores (pronombres, negaciones, conectores, intensificadores,
    vocativos) sin ningún modelo preentrenado.
    """
    markers = {
        "yo": r"\byo\b", "tu": r"\btu\b|\btú\b", "usted": r"\busted\b|\bud\b|\buds\b",
        "nosotros": r"\bnosotros\b|\bnosotras\b", "negacion": r"\bno\b|\bni\b|\bnunca\b|\btampoco\b",
        "pregunta": r"\?", "exclamacion": r"!",
        "conector": r"\bpero\b|\nporque\b|\nentonces\b|\nademas\b|\nsin embargo\b|\naunque\b",
        "intensificador": r"\bmuy\b|\bsuper\b|\bdemasiado\b|\bharto\b|\bre\b|\brequete\b",
        "vocativo": r"\bparce\b|\bmarica\b|\bhermano\b|\bmano\b|\bviejo\b|\bgordo\b|\bamigo\b",
        "jerga_juvenil": r"\blol\b|\bwtf\b|\bomg\b|\bxq\b|\bxq\b|\btqm\b|\bcr7\b|\bvale\b",
        "cortesia": r"\bgracias\b|\bpor favor\b|\bsaludos\b|\bbendiciones\b|\bfeliz\b",
    }
    working = pd.DataFrame({"text": texts.fillna("").str.lower(),
                            "clase": np.asarray(target.astype(str))})
    for name, pattern in markers.items():
        working[name] = working["text"].str.count(pattern)
    return working.groupby("clase")[list(markers)].mean().T


# --------------------------------------------------------------------------------------
# Figuras
# --------------------------------------------------------------------------------------
def fig_feature_heatmap(stats_table: pd.DataFrame, features: pd.DataFrame,
                        target: pd.Series, top_n: int = 20) -> str:
    """Mapa de calor con la media estandarizada por clase de las features más informativas."""
    selected = stats_table["feature"].head(top_n).tolist()
    labels = np.asarray(target.astype(str))
    means = features[selected].groupby(labels).mean()
    means = means.reindex(cfg.CLASSES)
    z_scores = (means - means.mean()) / means.std().replace(0, np.nan)
    z_scores = z_scores.fillna(0.0)
    figure, axis = plt.subplots(figsize=(8.4, 0.34 * len(selected) + 1.6))
    image = axis.imshow(z_scores.T.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    axis.set_yticks(range(len(selected)), selected, fontsize=7)
    axis.set_xticks(range(len(cfg.CLASSES)), cfg.CLASSES, rotation=0, fontsize=8)
    axis.set_title("Features más discriminativas (z-score de la media por clase)")
    figure.colorbar(image, ax=axis, shrink=0.7, label="z")
    path = cfg.FIGURES / "eda_heatmap_features.png"
    figure.savefig(path)
    plt.close(figure)
    return str(path)


def fig_mi_ranking(mi_table: pd.DataFrame, top_n: int = 20) -> str:
    """Ranking de información mutua normalizada."""
    subset = mi_table.head(top_n).iloc[::-1]
    figure, axis = plt.subplots(figsize=(6.6, 0.28 * len(subset) + 1.4))
    axis.barh(subset["feature"], subset["mi_normalized"], color="#2b6cb0")
    axis.set_xlabel("IM normalizada por H(y)")
    axis.set_title(f"Top {len(subset)} features por información mutua con el rango de edad")
    path = cfg.FIGURES / "eda_mutual_information.png"
    figure.savefig(path)
    plt.close(figure)
    return str(path)


def fig_top_terms(terms: pd.DataFrame, per_class: int = 8) -> str:
    """Barras horizontales de términos distintivos por clase."""
    classes = list(dict.fromkeys(terms["clase"]))
    figure, axes = plt.subplots(2, 3, figsize=(11.5, 6.0))
    for axis, label in zip(axes.ravel(), classes, strict=False):
        subset = terms[terms["clase"] == label].head(per_class).iloc[::-1]
        axis.barh(subset["termino"], subset["z"], color="#2f855a")
        axis.set_title(label, fontsize=9)
        axis.tick_params(axis="y", labelsize=7)
        axis.set_xlabel("z (log-odds)", fontsize=7)
    figure.suptitle("Términos más distintivos por rango de edad (log-odds con prior Dirichlet)", fontsize=10)
    figure.tight_layout()
    path = cfg.FIGURES / "eda_terminos_por_clase.png"
    figure.savefig(path)
    plt.close(figure)
    return str(path)


def fig_correlation(features: pd.DataFrame, columns: list[str]) -> str:
    """Matriz de correlación de Spearman entre features densas (redundancia)."""
    correlation = features[columns].corr(method="spearman")
    figure, axis = plt.subplots(figsize=(8.2, 7.0))
    image = axis.imshow(correlation.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(columns)), columns, rotation=90, fontsize=6)
    axis.set_yticks(range(len(columns)), columns, fontsize=6)
    axis.set_title("Correlación de Spearman entre features densas")
    figure.colorbar(image, ax=axis, shrink=0.75)
    path = cfg.FIGURES / "eda_correlacion_features.png"
    figure.savefig(path)
    plt.close(figure)
    return str(path)


def fig_length_histograms(frame: pd.DataFrame, text_col: str = "text") -> str:
    """Distribución de longitudes por clase (densidad)."""
    figure, axis = plt.subplots(figsize=(7.2, 3.6))
    for label in cfg.CLASSES:
        values = frame.loc[frame[cfg.TARGET] == label, text_col].str.len()
        axis.hist(values, bins=45, histtype="step", linewidth=1.2, density=True, label=label)
    axis.set_xlabel("Caracteres (sin padding)")
    axis.set_ylabel("Densidad")
    axis.set_title("Distribución de la longitud del tweet por rango de edad")
    axis.legend(fontsize=7, ncols=3)
    path = cfg.FIGURES / "eda_longitud_densidad.png"
    figure.savefig(path)
    plt.close(figure)
    return str(path)
