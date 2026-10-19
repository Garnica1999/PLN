#!/usr/bin/env python
"""Calcula las cifras y pruebas estadísticas del informe final TOXIC 2026 PUJ.

Lee los artefactos congelados del proyecto (probabilidades OOF, datos crudos y
registro del leaderboard), reconstruye las predicciones del campeón P133 a partir
de P114 y guarda un JSON con todas las cifras citadas en el artículo. No usa el
test para estimar nada: las únicas probabilidades del test que existen en el
proyecto son las del envío final y aquí no se tocan.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import binomtest, chi2_contingency
from sklearn.metrics import f1_score

RAIZ = Path(__file__).resolve().parents[2]
SALIDA = RAIZ / "informe" / "datos"
SALIDA.mkdir(parents=True, exist_ok=True)

ETIQUETAS = ("HS", "TR", "AG")
UMBRALES = np.array([0.44, 0.51, 0.46])
SEMILLA_BOOTSTRAP = 20260925
N_REPLICAS = 2000


def leer_train() -> pd.DataFrame:
    """Lee el CSV de entrenamiento limpiando las comillas de los identificadores."""
    for codificacion in ("utf-8", "cp1252"):
        try:
            marco = pd.read_csv(
                RAIZ / "data" / "raw" / "toxic_train.csv",
                sep=";",
                dtype=str,
                keep_default_na=False,
                encoding=codificacion,
            )
            break
        except UnicodeDecodeError:
            continue
    marco.columns = [columna.strip() for columna in marco.columns]
    marco["id"] = marco["id"].str.strip().str.strip('"')
    for etiqueta in ETIQUETAS:
        marco[etiqueta] = marco[etiqueta].astype(int)
    return marco


def predecir(prob: np.ndarray, ascendente: bool = True) -> np.ndarray:
    """Binariza probabilidades con los umbrales congelados y la regla HatEval."""
    pred = np.column_stack(
        [(prob[:, j] >= UMBRALES[j]).astype(int) for j in range(len(ETIQUETAS))]
    )
    if ascendente:
        pred[(pred[:, 1] == 1) & (pred[:, 0] == 0), 0] = 1
    return pred


def metrica(y_true: np.ndarray, pred: np.ndarray) -> dict:
    """F1 media column-wise y F1 por etiqueta."""
    f1 = [float(f1_score(y_true[:, j], pred[:, j])) for j in range(len(ETIQUETAS))]
    return {"macro": float(np.mean(f1)), "por_etiqueta": dict(zip(ETIQUETAS, f1))}


def wilson(y: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Intervalo de confianza de Wilson para una proporción binomial."""
    p = y / n
    denominador = 1.0 + z * z / n
    centro = (p + z * z / (2.0 * n)) / denominador
    radio = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominador
    return float(centro - radio), float(centro + radio)


def phi(x: np.ndarray, y: np.ndarray) -> float:
    """Coeficiente Phi entre dos variables binarias."""
    return float(np.corrcoef(x.astype(int), y.astype(int))[0, 1])


def mcnemar(correcto_a: np.ndarray, correcto_b: np.ndarray) -> dict:
    """McNemar exacto (binomial bilateral) sobre aciertos pareados."""
    a = np.asarray(correcto_a, dtype=bool).ravel()
    b = np.asarray(correcto_b, dtype=bool).ravel()
    b01 = int(np.sum(a & ~b))
    b10 = int(np.sum(~a & b))
    n = b01 + b10
    p = 1.0 if n == 0 else float(binomtest(min(b01, b10), n, 0.5).pvalue)
    return {"a_cierto_b_error": b01, "a_error_b_cierto": b10, "p": p}


def bootstrap_diferencia(
    y_true: np.ndarray,
    pred_referencia: np.ndarray,
    pred_candidato: np.ndarray,
    grupos: np.ndarray,
    n_replicas: int = N_REPLICAS,
    semilla: int = SEMILLA_BOOTSTRAP,
) -> dict:
    """IC95 de la diferencia de F1 macro con remuestreo por grupos completos."""
    generador = np.random.default_rng(semilla)
    unicos = np.unique(grupos)
    indices = {grupo: np.flatnonzero(grupos == grupo) for grupo in unicos}
    diferencias = np.empty(n_replicas)
    for repeticion in range(n_replicas):
        muestra = generador.choice(unicos, size=len(unicos), replace=True)
        elegidos = np.concatenate([indices[grupo] for grupo in muestra])
        f_ref = np.mean(
            [f1_score(y_true[elegidos, j], pred_referencia[elegidos, j]) for j in range(3)]
        )
        f_cand = np.mean(
            [f1_score(y_true[elegidos, j], pred_candidato[elegidos, j]) for j in range(3)]
        )
        diferencias[repeticion] = f_cand - f_ref
    return {
        "diferencia_observada": float(
            metrica(y_true, pred_candidato)["macro"] - metrica(y_true, pred_referencia)["macro"]
        ),
        "media_bootstrap": float(np.mean(diferencias)),
        "ic95_inferior": float(np.percentile(diferencias, 2.5)),
        "ic95_superior": float(np.percentile(diferencias, 97.5)),
        "probabilidad_mejora": float(np.mean(diferencias > 0)),
    }


def main() -> None:
    """Calcula y guarda todas las cifras del informe."""
    train = leer_train()
    oof = np.load(RAIZ / "features" / "oof_v03_final.npz", allow_pickle=True)
    y_true = oof["y_true"]
    ids_dev = oof["ids"]
    prob_v03 = oof["proba"]
    prob_p114 = np.load(RAIZ / "scratch" / "p114_oofs.npz")["can"]
    grupos = np.load(RAIZ / "features" / "v2_oof_base_corrected.npz", allow_pickle=True)["grupos"]

    # P133: base P121/P122 (sin reglas léxicas, que no disparan en train).
    eps = 1e-6
    z_hs = logit(np.clip(prob_p114[:, 0], eps, 1.0 - eps))
    z_tr = logit(np.clip(prob_p114[:, 1], eps, 1.0 - eps))
    z_ag = logit(np.clip(prob_p114[:, 2], eps, 1.0 - eps))
    aislado = (prob_p114[:, 1] < 0.15) & (prob_p114[:, 2] < 0.35)
    z_hs[aislado] -= 0.02 * (0.35 - prob_p114[aislado, 2])
    dirigido = prob_p114[:, 1] >= 0.58
    z_ag[dirigido] += 0.60 * (prob_p114[dirigido, 1] - 0.58)
    prob_p133 = np.column_stack([expit(z_hs), expit(z_tr), expit(z_ag)])

    pred_v03 = predecir(prob_v03, ascendente=False)
    pred_p114 = predecir(prob_p114, ascendente=True)
    pred_p133 = predecir(prob_p133, ascendente=True)

    cifras: dict = {
        "corpus": {
            "n_train": int(len(train)),
            "n_test": 1154,
            "n_dev": int(len(y_true)),
            "n_holdout": int(len(train) - len(y_true)),
        },
        "etiquetas": {},
        "perfiles": {
            clave: int(valor)
            for clave, valor in (
                train["HS"].astype(str) + train["TR"].astype(str) + train["AG"].astype(str)
            )
            .value_counts()
            .reindex(["000", "001", "100", "101", "110", "111"])
            .fillna(0)
            .astype(int)
            .items()
        },
        "longitudes": {
            "caracteres_media": float(train["text"].str.len().mean()),
            "caracteres_mediana": float(train["text"].str.len().median()),
            "caracteres_max": int(train["text"].str.len().max()),
            "palabras_media": float(train["text"].str.split().str.len().mean()),
            "palabras_mediana": float(train["text"].str.split().str.len().median()),
        },
        "near_duplicados": {
            "textos_train_test": 2885,
            "grupos_multiples": 4,
            "grupo_mayor": 6,
            "textos_en_grupos": 17,
            "grupos_fusionados_train": 1645,
            "filas_agrupadas": 86,
        },
        "modelos": {},
        "phi": {},
        "bootstrap": {},
        "mcnemar": {},
        "umbrales": {},
        "longitud_franjas": {},
        "serie": [],
        "leaderboard": [],
    }

    for etiqueta in ETIQUETAS:
        positivos = int(train[etiqueta].sum())
        inferior, superior = wilson(positivos, len(train))
        cifras["etiquetas"][etiqueta] = {
            "positivos": positivos,
            "negativos": int(len(train) - positivos),
            "prevalencia": positivos / len(train),
            "wilson_inferior": inferior,
            "wilson_superior": superior,
        }

    for nombre, prob, ascendente in (
        ("V03", prob_v03, False),
        ("P114", prob_p114, True),
        ("P133", prob_p133, True),
    ):
        pred = predecir(prob, ascendente=ascendente)
        detalle = metrica(y_true, pred)
        confusion = {}
        for j, etiqueta in enumerate(ETIQUETAS):
            tp = int(((y_true[:, j] == 1) & (pred[:, j] == 1)).sum())
            fp = int(((y_true[:, j] == 0) & (pred[:, j] == 1)).sum())
            fn = int(((y_true[:, j] == 1) & (pred[:, j] == 0)).sum())
            tn = int(((y_true[:, j] == 0) & (pred[:, j] == 0)).sum())
            confusion[etiqueta] = {"TP": tp, "FP": fp, "FN": fn, "TN": tn}
        cifras["modelos"][nombre] = {
            "ccv": detalle["macro"],
            "f1_por_etiqueta": detalle["por_etiqueta"],
            "confusion": confusion,
            "exact_match": float((pred == y_true).all(axis=1).mean()),
        }

    for a, b in (("HS", "TR"), ("HS", "AG"), ("TR", "AG")):
        ph = phi(train[a].to_numpy(), train[b].to_numpy())
        tabla = pd.crosstab(train[a], train[b])
        chi2, valor_p, _, _ = chi2_contingency(tabla, correction=False)
        cifras["phi"][f"{a}-{b}"] = {"phi": ph, "chi2": float(chi2), "p": float(valor_p)}

    cifras["bootstrap"]["P133-V03"] = bootstrap_diferencia(
        y_true, pred_v03, pred_p133, grupos
    )
    cifras["bootstrap"]["P133-P114"] = bootstrap_diferencia(
        y_true, pred_p114, pred_p133, grupos
    )
    cifras["bootstrap"]["P133-P110"] = {
        "diferencia_observada": 0.002584,
        "ic95_inferior": 0.000509,
        "ic95_superior": 0.004783,
        "probabilidad_mejora": 0.992,
        "fuente": "cuaderno ejecutado, sección 6.2",
    }

    # Error estandar bootstrap de la F1 del campeon (remuestreo por grupos).
    generador = np.random.default_rng(SEMILLA_BOOTSTRAP)
    unicos = np.unique(grupos)
    indices_por_grupo = {grupo: np.flatnonzero(grupos == grupo) for grupo in unicos}
    f1_bootstrap = np.empty(N_REPLICAS)
    for repeticion in range(N_REPLICAS):
        muestra = generador.choice(unicos, size=len(unicos), replace=True)
        elegidos = np.concatenate([indices_por_grupo[grupo] for grupo in muestra])
        f1_bootstrap[repeticion] = metrica(y_true[elegidos], pred_p133[elegidos])["macro"]
    error_estandar = float(f1_bootstrap.std(ddof=1))
    cifras["bootstrap"]["P133_F1_error_estandar"] = error_estandar
    cifras["bootstrap"]["P133_F1_error_estandar_escala_test"] = float(
        error_estandar * np.sqrt(len(y_true) / 1154)
    )

    cifras["mcnemar"]["P133-V03_celdas"] = mcnemar(pred_p133 == y_true, pred_v03 == y_true)
    cifras["mcnemar"]["P133-P114_celdas"] = mcnemar(pred_p133 == y_true, pred_p114 == y_true)
    cifras["mcnemar"]["P133-V03_filas"] = mcnemar(
        (pred_p133 == y_true).all(axis=1), (pred_v03 == y_true).all(axis=1)
    )
    cifras["mcnemar"]["P133-P114_filas"] = mcnemar(
        (pred_p133 == y_true).all(axis=1), (pred_p114 == y_true).all(axis=1)
    )

    rejilla = np.round(np.linspace(0.20, 0.80, 121), 3)
    curvas = {etiqueta: [] for etiqueta in ETIQUETAS}
    for umbral in rejilla:
        # HS con la regla ascendente activa; TR y AG no se ven afectados por ella.
        pred_hs = (prob_p133[:, 0] >= umbral).astype(int)
        pred_tr = (prob_p133[:, 1] >= 0.51).astype(int)
        pred_ag = (prob_p133[:, 2] >= 0.46).astype(int)
        pred_conjunta = np.column_stack([pred_hs, pred_tr, pred_ag])
        pred_conjunta[(pred_conjunta[:, 1] == 1) & (pred_conjunta[:, 0] == 0), 0] = 1
        curvas["HS"].append(float(f1_score(y_true[:, 0], pred_conjunta[:, 0])))
        for j, etiqueta in ((1, "TR"), (2, "AG")):
            curvas[etiqueta].append(
                float(f1_score(y_true[:, j], (prob_p133[:, j] >= umbral).astype(int)))
            )
    cifras["umbrales"] = {
        "rejilla": rejilla.tolist(),
        "curvas": curvas,
        "congelados": {"HS": 0.44, "TR": 0.51, "AG": 0.46},
        "optimos_oof": {
            etiqueta: {
                "umbral": float(rejilla[int(np.argmax(curvas[etiqueta]))]),
                "f1": float(np.max(curvas[etiqueta])),
            }
            for etiqueta in ETIQUETAS
        },
    }

    dev = train.copy()
    dev["id"] = dev["id"].astype(int)
    dev = dev.set_index("id").loc[ids_dev].reset_index()
    longitudes = dev["text"].str.len().to_numpy()
    corte_1, corte_2 = np.quantile(longitudes, [1 / 3, 2 / 3])
    for nombre, mascara in (
        ("corta", longitudes <= corte_1),
        ("media", (longitudes > corte_1) & (longitudes <= corte_2)),
        ("larga", longitudes > corte_2),
    ):
        pred = pred_p133[mascara]
        cifras["longitud_franjas"][nombre] = {
            "n": int(mascara.sum()),
            "macro": metrica(y_true[mascara], pred)["macro"],
            "por_etiqueta": metrica(y_true[mascara], pred)["por_etiqueta"],
        }
    cifras["longitud_franjas"]["cortes"] = [float(corte_1), float(corte_2)]

    serie = [
        ("P108", 0.782372, 0.781185, 0.78855, "Adoptado"),
        ("P109", 0.792215, 0.789294, 0.77447, "Rechazado (min_df=5)"),
        ("P110", 0.784998, 0.784318, 0.78967, "Campeón"),
        ("P111", 0.786912, 0.785868, 0.78951, "Rechazado"),
        ("P112", 0.780834, 0.781824, 0.78652, "Rechazado"),
        ("P113", 0.787241, 0.785049, 0.78664, "Rechazado"),
        ("P114", 0.787218, 0.785673, 0.79057, "Campeón"),
        ("P115", 0.788431, 0.787190, 0.78825, "Rechazado"),
        ("P116", 0.786348, 0.788892, 0.78975, "Rechazado"),
        ("P117", 0.787967, 0.787259, 0.78885, "Rechazado"),
        ("P118", 0.787881, 0.786026, 0.78930, "Rechazado"),
        ("P119", 0.785897, 0.784121, 0.78860, "Rechazado"),
        ("P120", 0.787582, 0.785673, 0.79035, "Rechazado"),
        ("P121", 0.787582, 0.785673, 0.79095, "Campeón"),
        ("P122", 0.787582, 0.786034, 0.79095, "Campeón (empate)"),
        ("P124", 0.787127, 0.785810, 0.79095, "Neutral"),
        ("P125", 0.787127, 0.785810, 0.79337, "Campeón"),
        ("P126", 0.787127, 0.785810, 0.79375, "Campeón"),
        ("P127", 0.787582, 0.786034, 0.79517, "Campeón"),
        ("P128", 0.787582, 0.786034, 0.80516, "Campeón histórico"),
        ("P129", 0.787582, 0.786034, 0.80078, "Rechazado (offset 4.0)"),
        ("P130", 0.787582, 0.786034, 0.80516, "Rechazado (empate)"),
        ("P131", 0.787582, 0.786034, 0.80872, "Campeón (1.er lugar)"),
        ("P132", 0.787582, 0.786034, 0.81279, "Campeón (>0.81)"),
        ("P133", 0.787582, 0.786034, 0.81447, "Campeón supremo"),
    ]
    cifras["serie"] = [
        {
            "experimento": experimento,
            "ccv_canonica": canonica,
            "ccv_47": particion,
            "lb": lb,
            "decision": decision,
        }
        for experimento, canonica, particion, lb, decision in serie
    ]

    hitos = [
        ("Plantilla", 0.51494),
        ("V01", 0.75677),
        ("V02", 0.77993),
        ("V03", 0.78306),
        ("P108", 0.78855),
        ("P110", 0.78967),
        ("P114", 0.79057),
        ("P121", 0.79095),
        ("P125", 0.79337),
        ("P126", 0.79375),
        ("P127", 0.79517),
        ("P128", 0.80516),
        ("P129", 0.80078),
        ("P130", 0.80516),
        ("P131", 0.80872),
        ("P132", 0.81279),
        ("P133", 0.81447),
    ]
    cifras["leaderboard"] = [{"hito": hito, "lb": lb} for hito, lb in hitos]

    with open(SALIDA / "estadisticas.json", "w", encoding="utf-8") as archivo:
        json.dump(cifras, archivo, ensure_ascii=False, indent=2)

    print("Cifras calculadas y guardadas en", SALIDA / "estadisticas.json")
    for nombre in ("V03", "P114", "P133"):
        print(f"  {nombre}: CCV = {cifras['modelos'][nombre]['ccv']:.6f}")
    print("  Bootstrap P133-V03:", cifras["bootstrap"]["P133-V03"])
    print("  Bootstrap P133-P114:", cifras["bootstrap"]["P133-P114"])
    print("  McNemar P133-V03:", cifras["mcnemar"]["P133-V03_celdas"])
    print("  McNemar P133-P114:", cifras["mcnemar"]["P133-P114_celdas"])


if __name__ == "__main__":
    main()
