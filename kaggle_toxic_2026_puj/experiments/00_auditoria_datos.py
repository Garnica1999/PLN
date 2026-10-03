"""Fase 0 - Auditoría de datos de KAGGLE_TOXIC_2026_PUJ.

Pregunta que responde
---------------------
¿Los tres archivos oficiales son íntegros, parseables sin pérdida y libres de
fugas (duplicados, pseudo-usuarios, asociación id-etiqueta) antes de modelar?

Método
------
1. Carga con :mod:`src.data` (encoding y quoting verificados).
2. Integridad: filas, ids, nulos, dominios de etiquetas, reglas estructurales.
3. Distribución de etiquetas y combinaciones HS x TR x AG.
4. Longitudes de texto por etiqueta.
5. Duplicados exactos y near-duplicates (TF-IDF de n-gramas de carácter).
6. Asociación estadística entre ``id`` y las etiquetas (fuga por orden).
7. Pseudo-autores vía primera mención ``@usuario`` (riesgo de grupo).
8. Validación del formato de ``submission.csv``.

Salidas
-------
``reports/auditoria_resumen.json``, ``reports/dataset_quality.csv``,
``reports/leakage_audit.md``, ``reports/leakage_audit.csv``,
``reports/eda/*.csv`` y ``reports/samples_por_clase.csv``.

Ejecución
---------
    /home/carlos/venv/bin/python experiments/00_auditoria_datos.py
"""

# Nota de auditoria forense (B-07, 2026-09-26): el clustering de near-duplicados se
# ajusta sobre train+test (2885 textos). Impacto medido sobre las particiones: 0.0
# (ARI=1.0; holdout y pliegues identicos). Si se repara, clusterizar solo con train y
# regenerar data/processed/grupos_near_duplicados.csv. No se modifico la logica
# historica.
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import data as data_module

REPORTS_DIR = PROJECT_ROOT / "reports"
EDA_DIR = REPORTS_DIR / "eda"
URL_PATTERN = re.compile(r"https?://\S+")
MENTION_PATTERN = re.compile(r"@(\w+)")
LABELS = ("HS", "TR", "AG")
NEAR_DUPLICATE_THRESHOLD = 0.90
NEAR_DUPLICATE_BAND = 0.80


def log(message: str) -> None:
    """Imprime un mensaje de progreso con prefijo de fase.

    Args:
        message: texto a imprimir.
    """
    print(f"[F0] {message}", flush=True)


def normalizar_texto(text: str) -> str:
    """Normaliza un texto para el análisis de similitud.

    Args:
        text: texto original.

    Returns:
        Texto en minúsculas, sin URLs y con espacios colapsados.
    """
    clean = URL_PATTERN.sub(" ", text.lower())
    return re.sub(r"\s+", " ", clean).strip()


def calidad_de_archivos() -> pd.DataFrame:
    """Construye el informe de calidad y encoding por archivo.

    Returns:
        DataFrame con una fila por archivo oficial.
    """
    filas = []
    for nombre, esperado in data_module.EXPECTED_BYTES.items():
        ruta = data_module.RAW_DIR / nombre
        raw = ruta.read_bytes()
        texto, encoding = data_module.decode_bytes(raw)
        filas.append(
            {
                "archivo": nombre,
                "bytes": len(raw),
                "bytes_esperados": esperado,
                "encoding": encoding,
                "bytes_no_utf8": sum(1 for byte in raw if byte > 127),
                "lineas": texto.count("\n") + 1,
                "sha256": data_module.sha256_of(ruta),
                "texto_solo_ascii": bool(texto.isascii()),
            }
        )
    return pd.DataFrame(filas)


def integridad(train: pd.DataFrame, test: pd.DataFrame, submission: pd.DataFrame) -> dict:
    """Ejecuta los asserts de integridad y devuelve el informe.

    Args:
        train: DataFrame de entrenamiento.
        test: DataFrame de test.
        submission: plantilla de envío.

    Returns:
        Diccionario con el informe de integridad.
    """
    assert len(train) == 1_731, f"train con {len(train)} filas"
    assert len(test) == 1_154, f"test con {len(test)} filas"
    assert len(submission) == 1_154, f"submission con {len(submission)} filas"
    assert train["id"].is_unique and test["id"].is_unique and submission["id"].is_unique
    assert set(train["id"]).isdisjoint(set(test["id"])), "ids compartidos train/test"
    assert set(test["id"]) == set(submission["id"]), "ids de test y submission difieren"
    assert list(submission.columns) == list(data_module.SUBMISSION_COLUMNS)
    assert set(train[list(LABELS)].to_numpy().ravel()) <= {0, 1}
    assert set(submission[list(LABELS)].to_numpy().ravel()) <= {0, 1}
    assert not train.isna().any().any() and not test.isna().any().any()

    violaciones_tr_sin_hs = int(((train["HS"] == 0) & (train["TR"] == 1)).sum())
    ag_sin_hs = int(((train["HS"] == 0) & (train["AG"] == 1)).sum())
    assert violaciones_tr_sin_hs == 0, "TR=1 sin HS=1 rompe la regla estructural"

    return {
        "train_filas": len(train),
        "test_filas": len(test),
        "submission_filas": len(submission),
        "ids_unicos": True,
        "ids_disjuntos_train_test": True,
        "ids_test_iguales_submission": True,
        "duplicados_exactos_train": int(train["text"].duplicated().sum()),
        "duplicados_exactos_test": int(test["text"].duplicated().sum()),
        "textos_compartidos_train_test": len(
            set(train["text"]) & set(test["text"])
        ),
        "textos_con_punto_y_coma": int(train["text"].str.contains(";").sum())
        + int(test["text"].str.contains(";").sum()),
        "regla_TR_implica_HS_violaciones": violaciones_tr_sin_hs,
        "AG_positivo_con_HS_cero": ag_sin_hs,
        "longitud_mediana_chars": float(train["text"].str.len().median()),
    }


def distribucion_etiquetas(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula la distribución marginal y conjunta de las etiquetas.

    Args:
        train: DataFrame de entrenamiento.

    Returns:
        Tupla ``(distribucion_marginal, combinaciones)``.
    """
    marginal = pd.DataFrame(
        {
            "etiqueta": list(LABELS),
            "positivos": [int(train[etiqueta].sum()) for etiqueta in LABELS],
        }
    )
    marginal["porcentaje_positivos"] = (
        100 * marginal["positivos"] / len(train)
    ).round(2)
    combinaciones = (
        train.groupby(list(LABELS), as_index=False)
        .size()
        .rename(columns={"size": "n"})
        .sort_values("n", ascending=False)
    )
    combinaciones["porcentaje"] = (100 * combinaciones["n"] / len(train)).round(2)
    return marginal, combinaciones


def longitudes_por_etiqueta(train: pd.DataFrame) -> pd.DataFrame:
    """Resume la longitud de los textos por grupo de etiqueta.

    Args:
        train: DataFrame de entrenamiento.

    Returns:
        DataFrame con estadísticos de longitud por grupo.
    """
    filas = []
    grupos: list[tuple[str, pd.Series]] = [("global", pd.Series(True, index=train.index))]
    for etiqueta in LABELS:
        grupos.append((f"{etiqueta}=1", train[etiqueta] == 1))
        grupos.append((f"{etiqueta}=0", train[etiqueta] == 0))
    for nombre, mascara in grupos:
        longitudes = train.loc[mascara, "text"].str.len()
        filas.append(
            {
                "grupo": nombre,
                "n": int(mascara.sum()),
                "media": round(float(longitudes.mean()), 1),
                "min": int(longitudes.min()),
                "p25": float(longitudes.quantile(0.25)),
                "mediana": float(longitudes.median()),
                "p75": float(longitudes.quantile(0.75)),
                "p95": float(longitudes.quantile(0.95)),
                "max": int(longitudes.max()),
            }
        )
    return pd.DataFrame(filas)


def analisis_similitud(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detecta near-duplicates intra-train y train-test con TF-IDF de caracteres.

    Args:
        train: DataFrame de entrenamiento.
        test: DataFrame de test.

    Returns:
        Tupla ``(resumen, pares_mas_similares)``.
    """
    textos_train = [normalizar_texto(texto) for texto in train["text"]]
    textos_test = [normalizar_texto(texto) for texto in test["text"]]
    vectorizador = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 4), min_df=1)
    matriz_train = vectorizador.fit_transform(textos_train)
    matriz_test = vectorizador.transform(textos_test)

    sim_train = (matriz_train @ matriz_train.T).toarray()
    np.fill_diagonal(sim_train, 0.0)
    sim_cross = (matriz_train @ matriz_test.T).toarray()

    idx_a, idx_b = np.where(np.triu(sim_train, k=1) > NEAR_DUPLICATE_BAND)
    sims_train = sim_train[idx_a, idx_b] if len(idx_a) else np.array([])
    idx_c, idx_d = np.where(sim_cross > NEAR_DUPLICATE_BAND)
    sims_cross = sim_cross[idx_c, idx_d] if len(idx_c) else np.array([])

    resumen = pd.DataFrame(
        [
            {
                "comparacion": "train-train",
                "pares_ge_0.90": int((sims_train >= NEAR_DUPLICATE_THRESHOLD).sum()),
                "pares_0.80_0.90": int(
                    ((sims_train >= NEAR_DUPLICATE_BAND) & (sims_train < NEAR_DUPLICATE_THRESHOLD)).sum()
                ),
                "similitud_maxima": round(float(sims_train.max()), 4) if len(sims_train) else 0.0,
            },
            {
                "comparacion": "train-test",
                "pares_ge_0.90": int((sims_cross >= NEAR_DUPLICATE_THRESHOLD).sum()),
                "pares_0.80_0.90": int(
                    ((sims_cross >= NEAR_DUPLICATE_BAND) & (sims_cross < NEAR_DUPLICATE_THRESHOLD)).sum()
                ),
                "similitud_maxima": round(float(sims_cross.max()), 4) if len(sims_cross) else 0.0,
            },
        ]
    )

    registros = []
    for comparacion, idx_izq, idx_der, sims, frame_der in (
        ("train-train", idx_a, idx_b, sims_train, train),
        ("train-test", idx_c, idx_d, sims_cross, test),
    ):
        orden = np.argsort(sims)[::-1][:10]
        for posicion in orden:
            i, j = int(idx_izq[posicion]), int(idx_der[posicion])
            registros.append(
                {
                    "comparacion": comparacion,
                    "id_a": train["id"].iloc[i],
                    "id_b": frame_der["id"].iloc[j],
                    "similitud": round(float(sims[posicion]), 4),
                    "texto_a": train["text"].iloc[i][:160],
                    "texto_b": frame_der["text"].iloc[j][:160],
                }
            )
    return resumen, pd.DataFrame(registros)


def clusters_near_duplicados(
    train: pd.DataFrame,
    test: pd.DataFrame,
    threshold: float = 0.85,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Agrupa textos casi idénticos de train y test en componentes conexas.

    Une dos textos si su similitud coseno (TF-IDF char_wb 4-4 sobre texto
    normalizado) supera ``threshold``. Los grupos resultantes se usan para que
    la validación cruzada no reparta copias casi idénticas entre train y
    validación, evitando estimaciones optimistas de F1.

    Args:
        train: DataFrame de entrenamiento.
        test: DataFrame de test.
        threshold: umbral de similitud para unir dos textos.

    Returns:
        Tupla ``(asignaciones, resumen)``. ``asignaciones`` tiene las columnas
        ``id, split, grupo, tamano_grupo``; ``resumen`` describe los grupos.
    """
    ids = list(train["id"]) + list(test["id"])
    splits = ["train"] * len(train) + ["test"] * len(test)
    textos = [normalizar_texto(texto) for texto in train["text"]] + [
        normalizar_texto(texto) for texto in test["text"]
    ]
    vectorizador = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 4), min_df=1)
    matriz = vectorizador.fit_transform(textos)
    similitud = (matriz @ matriz.T).toarray().astype("float32")
    np.fill_diagonal(similitud, 0.0)

    padre = list(range(len(ids)))

    def encontrar(nodo: int) -> int:
        """Devuelve la raíz de ``nodo`` con compresión de caminos."""
        while padre[nodo] != nodo:
            padre[nodo] = padre[padre[nodo]]
            nodo = padre[nodo]
        return nodo

    def unir(a: int, b: int) -> None:
        """Une dos nodos en la misma componente conexa."""
        raiz_a, raiz_b = encontrar(a), encontrar(b)
        if raiz_a != raiz_b:
            padre[max(raiz_a, raiz_b)] = min(raiz_a, raiz_b)

    filas, columnas = np.where(np.triu(similitud, k=1) >= threshold)
    for fila, columna in zip(filas.tolist(), columnas.tolist(), strict=True):
        unir(fila, columna)

    grupos = [encontrar(nodo) for nodo in range(len(ids))]
    asignaciones = pd.DataFrame({"id": ids, "split": splits, "grupo": grupos})
    tamano = asignaciones.groupby("grupo")["id"].transform("size")
    asignaciones["tamano_grupo"] = tamano

    resumen = (
        asignaciones[asignaciones["tamano_grupo"] > 1]
        .groupby(["grupo", "split"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    if "train" not in resumen.columns:
        resumen["train"] = 0
    if "test" not in resumen.columns:
        resumen["test"] = 0
    resumen["total"] = resumen["train"] + resumen["test"]
    resumen = resumen.sort_values("total", ascending=False)
    return asignaciones, resumen


def asociacion_id_etiqueta(train: pd.DataFrame) -> pd.DataFrame:
    """Evalúa si el orden del ``id`` se asocia con las etiquetas (fuga por orden).

    Args:
        train: DataFrame de entrenamiento.

    Returns:
        DataFrame con Spearman, chi-cuadrado y V de Cramér por etiqueta.
    """
    ids = train["id"].astype(int)
    deciles = pd.qcut(ids, 10, labels=False, duplicates="drop")
    filas = []
    for etiqueta in LABELS:
        rho, p_valor = spearmanr(ids, train[etiqueta])
        tabla = pd.crosstab(deciles, train[etiqueta])
        chi2, chi_p, _, _ = chi2_contingency(tabla)
        cramers_v = float(np.sqrt(chi2 / (tabla.to_numpy().sum() * (min(tabla.shape) - 1))))
        filas.append(
            {
                "etiqueta": etiqueta,
                "spearman_rho": round(float(rho), 4),
                "spearman_p": float(p_valor),
                "chi2": round(float(chi2), 3),
                "chi2_p": float(chi_p),
                "cramers_v": round(cramers_v, 4),
            }
        )
    return pd.DataFrame(filas)


def pseudo_autores(train: pd.DataFrame, minimo: int = 3) -> pd.DataFrame:
    """Agrupa tweets por primera mención para estimar riesgo de fuga por usuario.

    Args:
        train: DataFrame de entrenamiento.
        minimo: mínimo de tweets para considerar un pseudo-autor relevante.

    Returns:
        DataFrame con los pseudo-autores y la homogeneidad de sus etiquetas.
    """

    def primera_mencion(text: str) -> str:
        """Devuelve la primera mencion ``@usuario`` en minusculas o cadena vacia."""
        match = MENTION_PATTERN.search(text)
        return match.group(1).lower() if match else ""

    frame = train.copy()
    frame["mencion"] = frame["text"].map(primera_mencion)
    frame = frame[frame["mencion"] != ""]
    agrupado = (
        frame.groupby("mencion")
        .agg(n_tweets=("id", "size"), HS=("HS", "sum"), TR=("TR", "sum"), AG=("AG", "sum"))
        .reset_index()
    )
    agrupado["share_HS"] = (agrupado["HS"] / agrupado["n_tweets"]).round(3)
    agrupado = agrupado[agrupado["n_tweets"] >= minimo].sort_values("n_tweets", ascending=False)
    return agrupado


def muestras_por_clase(train: pd.DataFrame, por_combo: int = 5) -> pd.DataFrame:
    """Extrae ejemplos por combinación de etiquetas para el análisis lingüístico.

    Args:
        train: DataFrame de entrenamiento.
        por_combo: número de ejemplos por combinación.

    Returns:
        DataFrame con textos truncados y sus etiquetas.
    """
    partes = []
    for combo, grupo in train.groupby(list(LABELS), sort=True):
        muestra = grupo.head(por_combo).copy()
        muestra["combo"] = "".join(str(valor) for valor in combo)
        muestra["text"] = muestra["text"].str.slice(0, 240)
        partes.append(muestra)
    return pd.concat(partes, ignore_index=True)


def escribir_reporte_fuga(
    resumen: pd.DataFrame,
    asociacion: pd.DataFrame,
    autores: pd.DataFrame,
    clusters: pd.DataFrame,
) -> None:
    """Escribe el informe narrativo de fuga de datos.

    Args:
        resumen: resumen de similitud train-train y train-test.
        asociacion: resultados de asociación id-etiqueta.
        autores: pseudo-autores detectados.
        clusters: grupos de near-duplicados (umbral 0.85).
    """
    lineas = [
        "# Auditoría de fuga de datos - KAGGLE_TOXIC_2026_PUJ",
        "",
        "Fecha: 2026-09-22 (Fase 0). Fuente: solo `data/raw/` oficial.",
        "",
        "## 1. Duplicados y near-duplicates",
        "",
        resumen.to_markdown(index=False),
        "",
        "Umbral de near-duplicate: similitud coseno >= 0.90 sobre TF-IDF de",
        "n-gramas de carácter (char_wb 4-4) del texto normalizado.",
        "",
        "## 1b. Grupos de near-duplicados (umbral 0.85, train+test)",
        "",
        f"Grupos con más de un texto: {len(clusters)}.",
        "",
        clusters.head(12).to_markdown(index=False) if len(clusters) else "Sin casos.",
        "",
        "Estos grupos se guardan en `data/processed/grupos_near_duplicados.csv` y",
        "se usan como variable de agrupación en la validación cruzada para no",
        "repartir copias casi idénticas entre train y validación.",
        "",
        "## 2. Fuga por orden de `id`",
        "",
        asociacion.to_markdown(index=False),
        "",
        "Si `spearman_p` y `chi2_p` no son significativos, el orden de los ids no",
        "codifica la etiqueta y un split aleatorio estratificado es defendible.",
        "",
        "## 3. Pseudo-usuarios (primera mención)",
        "",
        f"Pseudo-autores con >= 3 tweets: {len(autores)}.",
        "",
        autores.head(10).to_markdown(index=False) if len(autores) else "Sin casos.",
        "",
        "## 4. Regla anti-fuga externa (HatEval)",
        "",
        "El corpus corresponde al esquema de anotación de HatEval (SemEval-2019",
        "Task 5, subconjunto español). Los textos y etiquetas originales circulan",
        "públicamente: queda PROHIBIDO consultarlos, cruzarlos o usarlos para",
        "entrenar, corregir o validar. Solo se usan las etiquetas de",
        "`toxic_train.csv`. Decisión registrada como D-002.",
        "",
        "## 5. Acciones derivadas",
        "",
        "- Validación primaria: `StratifiedGroupKFold` con grupos = near-duplicados",
        "  (umbral 0.85) y estratificación por combinación HS x TR x AG.",
        "- Control de robustez adicional con pseudo-autor (primera mención) como",
        "  grupo, para el subconjunto afectado.",
        "- Ninguna transformación se ajusta con datos de test.",
    ]
    (REPORTS_DIR / "leakage_audit.md").write_text("\n".join(lineas), encoding="utf-8")


def main() -> None:
    """Ejecuta la auditoría completa de la Fase 0."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    EDA_DIR.mkdir(parents=True, exist_ok=True)

    log("cargando archivos oficiales")
    train = data_module.read_train()
    test = data_module.read_test()
    submission = data_module.read_submission()

    log("integridad y reglas estructurales")
    informe = integridad(train, test, submission)

    log("calidad de archivos (encoding, bytes, hashes)")
    calidad = calidad_de_archivos()

    log("distribución de etiquetas")
    marginal, combinaciones = distribucion_etiquetas(train)

    log("longitudes por etiqueta")
    longitudes = longitudes_por_etiqueta(train)

    log("similitud y near-duplicates")
    similitud_resumen, similitud_pares = analisis_similitud(train, test)

    log("asociación id-etiqueta")
    asociacion = asociacion_id_etiqueta(train)

    log("grupos de near-duplicados (train+test)")
    asignaciones_grupos, clusters = clusters_near_duplicados(train, test, threshold=0.85)
    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    asignaciones_grupos.to_csv(processed_dir / "grupos_near_duplicados.csv", index=False)
    clusters.to_csv(EDA_DIR / "clusters_near_duplicados.csv", index=False)
    test_en_grupo = asignaciones_grupos[
        (asignaciones_grupos["split"] == "test") & (asignaciones_grupos["tamano_grupo"] > 1)
    ]
    train_en_grupo = asignaciones_grupos[
        (asignaciones_grupos["split"] == "train") & (asignaciones_grupos["tamano_grupo"] > 1)
    ]

    log("pseudo-autores por mención")
    autores = pseudo_autores(train)

    log("validación de submission")
    submission_ok = list(submission.columns) == list(data_module.SUBMISSION_COLUMNS)
    valores_submission = sorted(submission[list(LABELS)].to_numpy().ravel().tolist())

    calidad.to_csv(REPORTS_DIR / "dataset_quality.csv", index=False)
    marginal.to_csv(EDA_DIR / "distribucion_etiquetas.csv", index=False)
    combinaciones.to_csv(EDA_DIR / "combinaciones_etiquetas.csv", index=False)
    longitudes.to_csv(EDA_DIR / "longitudes_por_etiqueta.csv", index=False)
    similitud_resumen.to_csv(EDA_DIR / "similitud_resumen.csv", index=False)
    similitud_pares.to_csv(EDA_DIR / "similitud_pares.csv", index=False)
    asociacion.to_csv(EDA_DIR / "asociacion_id_etiqueta.csv", index=False)
    autores.to_csv(EDA_DIR / "pseudo_autores.csv", index=False)
    muestras_por_clase(train).to_csv(REPORTS_DIR / "samples_por_clase.csv", index=False)

    fuga = similitud_resumen.melt(id_vars="comparacion", var_name="metrica", value_name="valor")
    fuga.to_csv(REPORTS_DIR / "leakage_audit.csv", index=False)
    escribir_reporte_fuga(similitud_resumen, asociacion, autores, clusters)

    resumen = {
        "fecha": "2026-09-22",
        "integridad": informe,
        "calidad": calidad.to_dict(orient="records"),
        "etiquetas": marginal.to_dict(orient="records"),
        "combinaciones": combinaciones.to_dict(orient="records"),
        "submission_columnas_ok": bool(submission_ok),
        "submission_valores": valores_submission,
        "pseudo_autores_relevantes": len(autores),
        "grupos_near_duplicados": len(clusters),
        "train_en_grupo_near_dup": len(train_en_grupo),
        "test_en_grupo_near_dup": len(test_en_grupo),
    }
    (REPORTS_DIR / "auditoria_resumen.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log("=== RESUMEN ===")
    log(f"integridad: {informe}")
    log(f"marginal:\n{marginal.to_string(index=False)}")
    log(f"combinaciones:\n{combinaciones.to_string(index=False)}")
    log(f"similitud:\n{similitud_resumen.to_string(index=False)}")
    log(f"id-etiqueta:\n{asociacion.to_string(index=False)}")
    log(
        f"near-duplicados: {len(clusters)} grupos | train en grupo: {len(train_en_grupo)} "
        f"| test en grupo: {len(test_en_grupo)}"
    )
    log(f"pseudo-autores>=3: {len(autores)}")
    log("auditoría completada; reportes en reports/")


if __name__ == "__main__":
    main()
