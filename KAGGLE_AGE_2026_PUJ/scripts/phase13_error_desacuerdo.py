"""FASE 13 — Análisis de errores y estructura de desacuerdo del modelo final v12.

Observación
    El diagnóstico L1 midió que en el **25,94%** de las filas el ensamble falla mientras algún miembro
    acierta, y que los rescatadores son los miembros débiles y diversos. Esa medición usaba el
    ensamble de la fase L1 (meta LogReg sobre 273 features). Esta fase la repite sobre el **modelo
    final v12** (promedio de combinadores sobre 279), añade el análisis por **familias** de modelos,
    formaliza la estructura de votos (consenso, entropía, empates, margen) y la cruza con los
    segmentos de error. Es el insumo de hipótesis de las fases C (consenso), E (estructural) y G.

Protocolo
    1. OOF leave-one-fold-out del v12 con las mismas particiones agrupadas (se cachea).
    2. Un voto por miembro: argmax de su bloque de 6 columnas; el bloque acumulativo de 5 márgenes se
       convierte por conteo de signos; la fuente ordinal continua se excluye del voto y se analiza
       aparte (correlación con el índice ordinal real).
    3. Análisis: matriz de confusión, distancia ordinal del error, error por segmentos
       (longitud, tokens, menciones, hashtags, URL, RT, tipo de tweet, densidad de entidades),
       fronteras adyacentes, estructura de votos y rescate por familia.
    4. Todo el análisis es sobre OOF: no se toca test ni leaderboard.

Salida: ``experiments/phase13_error_desacuerdo.json``, tablas en ``reports/tables/phase13_*.csv``,
figura ``reports/figures/phase13_matriz_confusion_v12.png`` e informe
``reports/phase13_error_desacuerdo.md``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg
from src import cv
from src.config import set_seed

sys.path.insert(0, str(ROOT / "scripts"))
from phase2_ponderacion import MEMBER_FILES, EncodedMLP, build_features

RESULTS_PATH = cfg.EXPERIMENTS / "phase13_error_desacuerdo.json"
REPORT_PATH = cfg.REPORTS / "phase13_error_desacuerdo.md"
FIG_PATH = cfg.FIGURES / "phase13_matriz_confusion_v12.png"
OOF_CACHE = cfg.FEATURES / "oof_ensemble_v12.npy"

#: Familias de miembros (los bloques con 6 columnas; E015 ordinal y E028a se tratan aparte).
FAMILIAS = {
    "svm_lineal": ["oof_E013_margenes", "oof_E016_margenes", "oof_E014_margenes"],
    "prob_lineal": ["oof_E-009", "oof_E-011"],
    "bayes": ["oof_E-010"],
    "ordinal": ["oof_E015_softmax", "oof_E021a_ordinal", "oof_E028a_pares"],
    "prototipos": ["oof_E021b_rocchio"],
    "denso": ["oof_E019b_densas_svd"],
    "nbsvm": ["oof_ADV05_nbsvm"],
}


def submuestreo(matriz: np.ndarray, max_filas: int, semilla: int = cfg.SEED) -> np.ndarray:
    """Devuelve una vista con a lo sumo ``max_filas`` filas (para figuras y tablas de apoyo)."""
    if len(matriz) <= max_filas:
        return matriz
    generador = np.random.default_rng(semilla)
    return matriz[generador.choice(len(matriz), size=max_filas, replace=False)]


def oof_v12(features: np.ndarray, y: np.ndarray, folds: list[dict]) -> np.ndarray:
    """Calcula (o carga de caché) las probabilidades OOF del promedio de combinadores v12."""
    if OOF_CACHE.exists():
        cacheado = np.load(OOF_CACHE)
        if cacheado.shape == (len(y), len(cfg.CLASSES)):
            print(f"[oof] cargado de caché: {OOF_CACHE.name}", flush=True)
            return cacheado
    probabilidades = np.zeros((len(y), len(cfg.CLASSES)), dtype=np.float32)
    for fold in folds:
        train_idx = np.concatenate([f["val_idx"] for f in folds if f["fold"] != fold["fold"]])
        valid_idx = fold["val_idx"]
        logreg = Pipeline([("scaler", StandardScaler()),
                           ("clf", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs",
                                                      random_state=cfg.SEED))])
        logreg.fit(features[train_idx], y[train_idx])
        mlp = EncodedMLP()
        mlp.fit(features[train_idx], y[train_idx])
        probabilidades[valid_idx] = 0.5 * (logreg.predict_proba(features[valid_idx])
                                           + mlp.model.predict_proba(
                                               mlp.scaler.transform(features[valid_idx])))
        print(f"[oof] fold {fold['fold']} listo", flush=True)
    np.save(OOF_CACHE, probabilidades)
    return probabilidades


def votos_de_miembros() -> tuple[np.ndarray, list[str], np.ndarray | None]:
    """Construye la matriz de votos por miembro (clase predicha) y la fuente ordinal continua."""
    votos: list[np.ndarray] = []
    nombres: list[str] = []
    ordinal: np.ndarray | None = None
    for nombre, _ in MEMBER_FILES:
        matriz = np.load(cfg.FEATURES / nombre)
        if matriz.ndim == 1 or matriz.shape[1] == 1:
            ordinal = np.asarray(matriz).reshape(-1)
            continue
        if matriz.shape[1] == 5:
            votos.append(np.asarray((matriz > 0).sum(axis=1)).reshape(-1))
        else:
            votos.append(np.asarray(np.argmax(matriz, axis=1)).reshape(-1))
        nombres.append(Path(nombre).stem)
    return np.column_stack(votos), nombres, ordinal


def voto_por_familia(votos: np.ndarray, nombres: list[str]) -> dict[str, np.ndarray]:
    """Voto mayoritario de cada familia (moda de los votos de sus miembros)."""
    indice = {nombre: posicion for posicion, nombre in enumerate(nombres)}
    resultado = {}
    for familia, miembros in FAMILIAS.items():
        columnas = [indice[miembro] for miembro in miembros if miembro in indice]
        if not columnas:
            continue
        seleccion = votos[:, columnas]
        resultado[familia] = np.array([np.bincount(fila, minlength=len(cfg.CLASSES)).argmax()
                                       for fila in seleccion])
    return resultado


def tabla_fronteras(indice_real: np.ndarray, indice_pred: np.ndarray) -> pd.DataFrame:
    """Exactitud dentro de cada frontera adyacente y contribución al error total."""
    total_errores = int((indice_real != indice_pred).sum())
    filas = []
    for borde in range(len(cfg.CLASSES) - 1):
        pareja = (indice_real == borde) | (indice_real == borde + 1)
        aciertos = int((indice_pred[pareja] == indice_real[pareja]).sum())
        errores_borde = int(((indice_pred != indice_real) & pareja).sum())
        filas.append({
            "frontera": f"{cfg.CLASSES[borde]}|{cfg.CLASSES[borde + 1]}",
            "filas": int(pareja.sum()),
            "exactitud_dentro": round(aciertos / max(1, int(pareja.sum())), 4),
            "errores": errores_borde,
            "pct_del_error_total": round(100 * errores_borde / max(1, total_errores), 2),
            "pct_de_cruces_entre_la_pareja": round(
                100 * int(((indice_pred == borde) & (indice_real == borde + 1)).sum()
                          + ((indice_pred == borde + 1) & (indice_real == borde)).sum())
                / max(1, int(pareja.sum())), 2),
        })
    return pd.DataFrame(filas)


def segmentos(dense: pd.DataFrame, tipo: np.ndarray, correcto: np.ndarray,
              rescatable: np.ndarray) -> pd.DataFrame:
    """Error y potencial de rescate por segmento estructural."""
    marcos = {
        "longitud": pd.qcut(dense["n_chars"], 4, labels=["L1", "L2", "L3", "L4"]).astype(str),
        "tokens": pd.qcut(dense["n_words"], 4, labels=["T1", "T2", "T3", "T4"]).astype(str),
        "menciones": pd.cut(dense["n_mention"], [-1, 0, 1, 100], labels=["0", "1", "2+"]).astype(str),
        "hashtags": pd.cut(dense["n_hashtag"], [-1, 0, 10], labels=["0", "1+"]).astype(str),
        "url": dense["n_url"].gt(0).map({True: "con_url", False: "sin_url"}),
        "retuit": dense["is_rt"].map({1: "rt", 0: "no_rt"}),
        "tipo": pd.Series(tipo),
    }
    filas = []
    for nombre, serie in marcos.items():
        for valor, posiciones in pd.Series(np.arange(len(serie))).groupby(serie.to_numpy(),
                                                                          observed=True):
            posiciones = np.asarray(posiciones)
            filas.append({
                "segmento": nombre, "valor": str(valor), "filas": len(posiciones),
                "error_pct": round(100 * float((~correcto[posiciones]).mean()), 2),
                "rescatable_pct": round(100 * float(rescatable[posiciones].mean()), 2),
            })
    return pd.DataFrame(filas)


def main() -> int:
    """Ejecuta el análisis de errores y desacuerdo del modelo v12."""
    set_seed()
    started = time.time()
    payload: dict = {"fase": "B — errores y desacuerdo", "fecha": "2026-09-25"}

    features, y, dense, tipo, _ = build_features()
    features = np.concatenate([features, np.load(cfg.FEATURES / "oof_ADV07_nbsvm_word.npy")], axis=1)
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")
    folds = cv.make_folds(y, groups=clusters["cluster"].to_numpy()[:len(y)], n_splits=5,
                          n_repeats=1, scheme="group")

    probabilidades = oof_v12(features, y, folds)
    indice_real = np.array([cfg.CLASSES.index(valor) for valor in y])
    indice_pred = np.argmax(probabilidades, axis=1)
    correcto = indice_pred == indice_real

    votos, nombres, ordinal = votos_de_miembros()
    aciertos_por_miembro = votos == indice_real[:, None]
    n_aciertos = aciertos_por_miembro.sum(axis=1)
    rescatable = (~correcto) & (n_aciertos >= 1)
    familias = voto_por_familia(votos, nombres)
    aciertos_familia = {familia: (voto == indice_real) for familia, voto in familias.items()}

    # ------------------------------------------------------------------ 1. Métricas globales
    confusion = confusion_matrix(indice_real, indice_pred, labels=np.arange(len(cfg.CLASSES)))
    distancia = np.abs(indice_pred - indice_real)
    payload["global"] = {
        "macro_f1": round(float(f1_score(y, np.array(cfg.CLASSES)[indice_pred], average="macro",
                                         labels=cfg.CLASSES)), 4),
        "accuracy": round(float(correcto.mean()), 4),
        "f1_por_clase": {clase: round(float(valor), 4) for clase, valor in
                         zip(cfg.CLASSES, f1_score(indice_real, indice_pred, average=None,
                                                   labels=np.arange(len(cfg.CLASSES))))},
        "error_por_distancia_ordinal": {str(grados): round(100 * float((distancia == grados).mean()), 3)
                                        for grados in range(len(cfg.CLASSES))},
        "matriz_confusion": confusion.tolist(),
        "matriz_confusion_normalizada": np.round(
            confusion / np.maximum(1, confusion.sum(axis=1, keepdims=True)), 4).tolist(),
    }
    print(f"[global] macro-F1 {payload['global']['macro_f1']} · accuracy "
          f"{payload['global']['accuracy']}", flush=True)

    # ------------------------------------------------------- 2. Estructura de votos y consenso
    votos_por_clase = np.zeros((len(y), len(cfg.CLASSES)))
    for posicion in range(votos.shape[1]):
        votos_por_clase[np.arange(len(y)), votos[:, posicion]] += 1
    reparto = votos_por_clase / votos.shape[1]
    entropia = -(np.where(reparto > 0, reparto * np.log(np.maximum(reparto, 1e-12)), 0.0)).sum(axis=1)
    ordenado = np.sort(reparto, axis=1)
    consenso = ordenado[:, -1]
    margen_consenso = ordenado[:, -1] - ordenado[:, -2]
    empate_top2 = ordenado[:, -1] == ordenado[:, -2]
    share_ensamble = reparto[np.arange(len(y)), indice_pred]
    payload["votos"] = {
        "miembros_que_votan": int(votos.shape[1]),
        "n_aciertos_por_fila": {str(k): int((n_aciertos == k).sum())
                                for k in range(votos.shape[1] + 1)},
        "pct_todos_fallan": round(100 * float((n_aciertos == 0).mean()), 3),
        "pct_exactamente_uno_acierta": round(100 * float((n_aciertos == 1).mean()), 3),
        "pct_rescatable": round(100 * float(rescatable.mean()), 3),
        "pct_empate_top2": round(100 * float(empate_top2.mean()), 3),
        "consenso_medio_global": round(float(consenso.mean()), 4),
        "consenso_medio_correctas": round(float(consenso[correcto].mean()), 4),
        "consenso_medio_incorrectas": round(float(consenso[~correcto].mean()), 4),
        "entropia_media_correctas": round(float(entropia[correcto].mean()), 4),
        "entropia_media_incorrectas": round(float(entropia[~correcto].mean()), 4),
        "share_ensamble_medio_correctas": round(float(share_ensamble[correcto].mean()), 4),
        "share_ensamble_medio_incorrectas": round(float(share_ensamble[~correcto].mean()), 4),
    }
    aristas = np.quantile(consenso, [0.2, 0.4, 0.6, 0.8])
    deciles = np.digitize(consenso, aristas)
    tabla_consenso = pd.DataFrame([
        {"quintil_consenso": int(q), "filas": int((deciles == q).sum()),
         "error_pct": round(100 * float((~correcto[deciles == q]).mean()), 2),
         "rescatable_pct": round(100 * float(rescatable[deciles == q].mean()), 2)}
        for q in range(5)])
    tabla_consenso.to_csv(cfg.TABLES / "phase13_consenso.csv", index=False)
    print(f"[votos] todos fallan {payload['votos']['pct_todos_fallan']}% · rescatable "
          f"{payload['votos']['pct_rescatable']}%", flush=True)

    # --------------------------------------------------------------------- 3. Rescate por familia
    filas_familia = []
    for familia, acierto in aciertos_familia.items():
        filas_familia.append({
            "familia": familia, "miembros": len(FAMILIAS[familia]),
            "exactitud_global": round(float(acierto.mean()), 4),
            "pct_acierta_cuando_ensamble_falla": round(100 * float(acierto[~correcto].mean()), 2),
            "unica_familia_que_acierta_pct": round(
                100 * float((acierto & (np.sum(np.column_stack(list(aciertos_familia.values())),
                                               axis=1) == 1) & (~correcto)).mean()), 3),
        })
    tabla_familias = pd.DataFrame(filas_familia).sort_values(
        "pct_acierta_cuando_ensamble_falla", ascending=False)
    tabla_familias.to_csv(cfg.TABLES / "phase13_familias.csv", index=False)
    payload["familias"] = tabla_familias.to_dict(orient="records")
    print(f"[familias] {tabla_familias.to_string(index=False)}", flush=True)

    # ------------------------------------------------------------------------ 4. Segmentos
    tabla_segmentos = segmentos(dense, tipo, correcto, rescatable)
    tabla_segmentos.to_csv(cfg.TABLES / "phase13_segmentos.csv", index=False)
    payload["segmentos"] = tabla_segmentos.to_dict(orient="records")

    # ------------------------------------------------------------------------ 5. Fronteras
    tabla_frontera = tabla_fronteras(indice_real, indice_pred)
    tabla_frontera.to_csv(cfg.TABLES / "phase13_fronteras.csv", index=False)
    payload["fronteras"] = tabla_frontera.to_dict(orient="records")
    print(f"[fronteras]\n{tabla_frontera.to_string(index=False)}", flush=True)

    # ----------------------------------------------------- 6. Matriz de confusión (figura)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figura, eje = plt.subplots(figsize=(7, 6))
        imagen = eje.imshow(confusion / np.maximum(1, confusion.sum(axis=1, keepdims=True)),
                            cmap="Blues", vmin=0, vmax=1)
        eje.set_xticks(range(len(cfg.CLASSES)), cfg.CLASSES, rotation=45, ha="right")
        eje.set_yticks(range(len(cfg.CLASSES)), cfg.CLASSES)
        eje.set_xlabel("predicho")
        eje.set_ylabel("real")
        eje.set_title(f"Matriz de confusión normalizada — OOF v12 (macro-F1 "
                      f"{payload['global']['macro_f1']:.4f})")
        for fila in range(len(cfg.CLASSES)):
            for columna in range(len(cfg.CLASSES)):
                valor = confusion[fila, columna] / max(1, confusion[fila].sum())
                eje.text(columna, fila, f"{valor:.2f}", ha="center", va="center",
                         fontsize=8, color="white" if valor > 0.5 else "black")
        figura.colorbar(imagen, ax=eje, fraction=0.046)
        figura.tight_layout()
        figura.savefig(FIG_PATH, dpi=150)
        plt.close(figura)
        payload["figura"] = str(FIG_PATH)
    except Exception as error:  # noqa: BLE001 - la figura no debe tumbar el análisis
        payload["figura"] = f"error: {type(error).__name__}: {error}"

    # --------------------------------------------------- 7. Fuente ordinal continua (aparte)
    if ordinal is not None:
        correlacion = float(np.corrcoef(ordinal, indice_real)[0, 1])
        payload["ordinal_continuo"] = {
            "correlacion_con_indice_real": round(correlacion, 4),
            "orientacion": "creciente" if correlacion >= 0 else "decreciente (se invierte)",
            "desviacion_media_absoluta_en_clases": round(
                float(np.mean(np.abs(ordinal * np.sign(correlacion)
                                     - (indice_real - indice_real.mean()) * np.sign(correlacion)))), 4),
        }

    # ------------------------------------------------------------------------ 8. Informe
    payload["segundos"] = round(time.time() - started, 1)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                            encoding="utf-8")

    lineas = [
        "# Fase B — Análisis de errores y desacuerdo del modelo v12", "",
        "Generado por `scripts/phase13_error_desacuerdo.py` sobre predicciones **OOF** "
        "(leave-one-fold-out, particiones agrupadas por clúster de near-duplicados).", "",
        "## 1. Global", "",
        f"- macro-F1 OOF **{payload['global']['macro_f1']}** · accuracy "
        f"{payload['global']['accuracy']}",
        f"- F1 por clase: {payload['global']['f1_por_clase']}",
        f"- Error por distancia ordinal: {payload['global']['error_por_distancia_ordinal']}", "",
        "## 2. Estructura de votos", "",
        f"- Todos los miembros fallan: **{payload['votos']['pct_todos_fallan']}%** · exactamente uno "
        f"acierta: {payload['votos']['pct_exactamente_uno_acierta']}% · **rescatable (el ensamble "
        f"falla y algún miembro acierta): {payload['votos']['pct_rescatable']}%**",
        f"- Consenso medio: correctas {payload['votos']['consenso_medio_correctas']} vs incorrectas "
        f"{payload['votos']['consenso_medio_incorrectas']}",
        f"- Entropía media: correctas {payload['votos']['entropia_media_correctas']} vs incorrectas "
        f"{payload['votos']['entropia_media_incorrectas']}",
        f"- Empate top-2: {payload['votos']['pct_empate_top2']}%", "",
        "## 3. Rescate por familia", "", tabla_familias.to_markdown(index=False), "",
        "## 4. Fronteras adyacentes", "", tabla_frontera.to_markdown(index=False), "",
        "## 5. Consenso y error", "", tabla_consenso.to_markdown(index=False), "",
        "## 6. Segmentos (extracto)", "",
        tabla_segmentos.to_markdown(index=False), "",
        "## 7. Fuente ordinal continua", "",
        f"{payload.get('ordinal_continuo', 'no disponible')}", "",
        f"_Duración: {payload['segundos']} s._",
    ]
    REPORT_PATH.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"\ninforme: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
