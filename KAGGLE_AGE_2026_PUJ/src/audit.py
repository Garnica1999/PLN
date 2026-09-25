"""Funciones de auditoría de datos, leakage y diccionario de datos.

Se usan desde ``scripts/fase01_auditoria.py`` (que genera y ejecuta
``notebooks/01_data_audit.ipynb``). Todas las funciones son de solo lectura sobre los
DataFrames y escriben, cuando corresponde, figuras y tablas en ``reports/``.
"""

from __future__ import annotations

from collections import Counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config as cfg  # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "font.size": 9})


def basic_profile(frame: pd.DataFrame, name: str) -> dict:
    """Perfil básico de un archivo: forma, nulos, duplicados de id y cardinalidad."""
    profile = {
        "file": name,
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "dtypes": {col: str(dtype) for col, dtype in frame.dtypes.items()},
        "missing": {col: int(value) for col, value in frame.isna().sum().items()},
        "unique": {col: int(frame[col].nunique(dropna=True)) for col in frame.columns},
    }
    if cfg.ID_COL in frame.columns:
        profile["id_min"] = int(frame[cfg.ID_COL].min())
        profile["id_max"] = int(frame[cfg.ID_COL].max())
        profile["id_duplicated"] = int(frame[cfg.ID_COL].duplicated().sum())
    return profile


def length_profile(frame: pd.DataFrame, name: str, text_col: str = cfg.TEXT_COL) -> dict:
    """Estadística de longitud de texto cruda vs. limpia (evidencia del padding D-005)."""
    raw = frame[text_col].fillna("").astype(str)
    stripped = raw.str.strip()
    return {
        "file": name,
        "raw_len_min": int(raw.str.len().min()),
        "raw_len_median": float(raw.str.len().median()),
        "raw_len_max": int(raw.str.len().max()),
        "raw_len_eq_padded_width_pct": round(100 * float((raw.str.len() == cfg.PADDED_WIDTH).mean()), 3),
        "clean_len_min": int(stripped.str.len().min()),
        "clean_len_median": float(stripped.str.len().median()),
        "clean_len_mean": round(float(stripped.str.len().mean()), 2),
        "clean_len_max": int(stripped.str.len().max()),
        "clean_len_eq_padded_width": int((stripped.str.len() == cfg.PADDED_WIDTH).sum()),
        "empty_texts": int((stripped.str.len() == 0).sum()),
    }


def id_proxy_check(frame: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Comprueba si el ``id`` actúa como proxy de la clase (riesgo de leakage).

    Returns
    -------
    pandas.DataFrame
        Probabilidad de cada clase por decil de ``id`` (filas = deciles).
    """
    working = frame[[cfg.ID_COL, cfg.TARGET]].copy()
    working["decile"] = pd.qcut(working[cfg.ID_COL], n_bins, labels=False)
    return pd.crosstab(working["decile"], working[cfg.TARGET], normalize="index").round(4)


def label_conflict_stats(frame: pd.DataFrame, text_col: str = cfg.TEXT_COL) -> dict:
    """Mide ruido de etiqueta: textos idénticos con más de una etiqueta."""
    normalized = (frame[text_col].fillna("").astype(str).str.lower()
                  .str.replace(r"\s+", " ", regex=True).str.strip())
    labels_per_text = frame.assign(_norm=normalized).groupby("_norm")[cfg.TARGET].nunique()
    conflicted_texts = labels_per_text[labels_per_text > 1].index
    affected = int(normalized.isin(conflicted_texts).sum())
    return {
        "texts_with_conflict": int(len(conflicted_texts)),
        "rows_affected": affected,
        "rows_affected_pct": round(100 * affected / len(frame), 4),
    }


def duplicate_stats(train: pd.DataFrame, test: pd.DataFrame,
                    text_col: str = cfg.TEXT_COL) -> dict:
    """Duplicados exactos intra-train, intra-test y cross-split (texto normalizado)."""

    def normalize(series: pd.Series) -> pd.Series:
        return (series.fillna("").astype(str).str.lower()
                .str.replace(r"\s+", " ", regex=True).str.strip())

    train_norm = normalize(train[text_col])
    test_norm = normalize(test[text_col])
    train_set = set(train_norm[train_norm.str.len() > 0])
    test_set = set(test_norm[test_norm.str.len() > 0])
    covered = test_norm.isin(train_set)
    return {
        "train_exact_dup_rows": int(train_norm.duplicated().sum()),
        "train_exact_dup_pct": round(100 * float(train_norm.duplicated().mean()), 4),
        "train_unique_texts": int(train_norm.nunique()),
        "test_unique_texts": int(test_norm.nunique()),
        "test_rows_seen_in_train": int(covered.sum()),
        "test_rows_seen_in_train_pct": round(100 * float(covered.mean()), 4),
        "test_unique_texts_seen_in_train": int(len(train_set & test_set)),
    }


def build_data_dictionary(train: pd.DataFrame, test: pd.DataFrame,
                          sample: pd.DataFrame) -> pd.DataFrame:
    """Construye ``reports/data_dictionary.csv`` con el esquema documentado.

    Returns
    -------
    pandas.DataFrame
        Columnas ``column``, ``type``, ``missing``, ``unique``, ``description``, ``role``.
    """
    descriptions = {
        (cfg.TRAIN_FILE.name, "id"): "Identificador único del tweet dentro del archivo de train",
        (cfg.TRAIN_FILE.name, "text"): "Contenido del tweet (ancho fijo 140 con padding de espacios)",
        (cfg.TRAIN_FILE.name, "age_range"): "Variable objetivo: rango de edad del autor del tweet",
        (cfg.TEST_FILE.name, "id"): "Identificador único del tweet dentro del archivo de test",
        (cfg.TEST_FILE.name, "text"): "Contenido del tweet (ancho fijo 140 con padding de espacios)",
        (cfg.SUBMISSION_FILE.name, "id"): "Identificador del tweet a predecir (mismo orden que test)",
        (cfg.SUBMISSION_FILE.name, "age_range"): "Etiqueta predicha; plantilla con el valor 13-17",
    }
    roles = {"id": "id", "text": "feature", "age_range": "target"}
    rows: list[dict] = []
    for name, frame in ((cfg.TRAIN_FILE.name, train), (cfg.TEST_FILE.name, test),
                        (cfg.SUBMISSION_FILE.name, sample)):
        for column in frame.columns:
            role = roles[column]
            if name == cfg.SUBMISSION_FILE.name:
                role = "id" if column == "id" else "prediction"
            rows.append({
                "column": f"{name.split('.')[0]}.{column}",
                "type": str(frame[column].dtype),
                "missing": int(frame[column].isna().sum()),
                "unique": int(frame[column].nunique(dropna=True)),
                "description": descriptions.get((name, column), ""),
                "role": role,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Figuras
# --------------------------------------------------------------------------------------
def fig_class_distribution(train: pd.DataFrame) -> str:
    """Barras con el conteo y porcentaje de cada clase."""
    counts = train[cfg.TARGET].value_counts().reindex(cfg.CLASSES)
    figure, axis = plt.subplots(figsize=(6.4, 3.2))
    bars = axis.bar(counts.index, counts.values, color="#2b6cb0")
    axis.set_title("Distribución de clases en train (n=%d)" % len(train))
    axis.set_xlabel("Rango de edad")
    axis.set_ylabel("Tweets")
    for bar, value in zip(bars, counts.values, strict=True):
        axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}\n({100*value/len(train):.1f}%)",
                  ha="center", va="bottom", fontsize=7)
    axis.set_ylim(0, counts.max() * 1.18)
    path = cfg.FIGURES / "audit_distribucion_clases.png"
    figure.savefig(path)
    plt.close(figure)
    return str(path)


def fig_length_by_class(train: pd.DataFrame) -> str:
    """Boxplot de longitud limpia del texto por clase."""
    lengths = [train.loc[train[cfg.TARGET] == label, cfg.TEXT_COL].str.len().values
               for label in cfg.CLASSES]
    figure, axis = plt.subplots(figsize=(6.4, 3.4))
    axis.boxplot(lengths, tick_labels=cfg.CLASSES, showfliers=False, patch_artist=True,
                 boxprops={"facecolor": "#bee3f8"})
    medians = [float(np.median(values)) for values in lengths]
    for index, median in enumerate(medians, start=1):
        axis.text(index, median + 1.5, f"{median:.0f}", ha="center", fontsize=7)
    axis.set_title("Longitud del texto (caracteres, sin padding) por rango de edad")
    axis.set_xlabel("Rango de edad")
    axis.set_ylabel("Caracteres")
    path = cfg.FIGURES / "audit_longitud_por_clase.png"
    figure.savefig(path)
    plt.close(figure)
    return str(path)


def fig_padding_effect(raw_lengths: pd.Series, clean_lengths: pd.Series) -> str:
    """Compara la longitud cruda (ancho fijo) con la longitud limpia."""
    figure, axes = plt.subplots(1, 2, figsize=(8.6, 3.0))
    axes[0].hist(raw_lengths, bins=40, color="#c05621")
    axes[0].set_title("Longitud cruda (train): padding a 140")
    axes[0].set_xlabel("Caracteres")
    axes[0].set_ylabel("Tweets")
    axes[1].hist(clean_lengths, bins=40, color="#2f855a")
    axes[1].set_title("Longitud real tras strip (train)")
    axes[1].set_xlabel("Caracteres")
    figure.suptitle("Efecto del ancho fijo del corpus (D-005)", fontsize=9)
    path = cfg.FIGURES / "audit_padding.png"
    figure.savefig(path)
    plt.close(figure)
    return str(path)


def fig_dup_exposure(exact_pct: float, signature_pct: float, purity_pct: float) -> str:
    """Barras de exposición a leakage por duplicados."""
    labels = ["Duplicado exacto\n(test visto en train)", "Firma 60 chars\n(near-duplicado)"]
    values = [exact_pct, signature_pct]
    figure, axis = plt.subplots(figsize=(6.0, 3.0))
    bars = axis.bar(labels, values, color=["#c53030", "#dd6b20"])
    for bar, value in zip(bars, values, strict=True):
        axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}%", ha="center",
                  va="bottom", fontsize=8)
    axis.set_ylabel("% de filas de test")
    axis.set_title(f"Exposición de test a textos de train (pureza de etiqueta: {purity_pct:.1f}%)")
    axis.set_ylim(0, max(values) * 1.3)
    path = cfg.FIGURES / "audit_leakage_duplicados.png"
    figure.savefig(path)
    plt.close(figure)
    return str(path)


def fig_id_proxy(table: pd.DataFrame) -> str:
    """Muestra que la clase es estable a lo largo de los deciles de ``id``."""
    figure, axis = plt.subplots(figsize=(6.4, 3.2))
    for label in cfg.CLASSES:
        axis.plot(table.index, table[label], marker="o", markersize=3, label=label)
    axis.axhline(1 / len(cfg.CLASSES), color="black", linestyle="--", linewidth=0.8)
    axis.set_xlabel("Decil de id")
    axis.set_ylabel("P(clase | decil)")
    axis.set_title("Verificación de leakage por id (línea base = 1/6)")
    axis.legend(fontsize=7, ncols=3)
    path = cfg.FIGURES / "audit_id_proxy.png"
    figure.savefig(path)
    plt.close(figure)
    return str(path)


def most_common_labels_per_cluster(clusters: pd.DataFrame, n_train: int,
                                   y_train: pd.Series) -> pd.DataFrame:
    """Resumen de los clústeres más grandes con su composición de etiquetas."""
    labels = np.asarray(y_train.astype(str))
    cluster_train = clusters["cluster"].to_numpy()[:n_train]
    counts = Counter(cluster_train.tolist())
    rows: list[dict] = []
    for cluster_id, size in counts.most_common(10):
        mask = cluster_train == cluster_id
        composition = Counter(labels[mask])
        rows.append({
            "cluster": int(cluster_id),
            "size_train": int(size),
            "labels": dict(composition),
            "example": clusters["norm"].to_numpy()[:n_train][mask][0][:90],
        })
    return pd.DataFrame(rows)
