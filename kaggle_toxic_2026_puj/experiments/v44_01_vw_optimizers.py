"""V44-01: Vowpal Wabbit con FTRL, Coin Betting y PiSTOL sobre el TF-IDF de V03.

Que responde
------------
La pregunta de la fase, dicha con las palabras de la directiva:

> Puede alguno de los optimizadores online/proximal de Vowpal Wabbit encontrar una
> solucion lineal que generalice mejor o complemente a V03?

La correccion respecto a V43
----------------------------
V43 uso el wrapper low-level `pyvw` y no pudo construir correctamente el protocolo
train/predict OOF. Este script usa la **API scikit-learn oficial**:
`vowpalwabbit.sklearn.VWClassifier`, con interfaz `fit/predict/predict_proba`.

Los tres optimizadores
----------------------
La documentacion oficial de Vowpal Wabbit expone:

- **FTRL** (`ftrl=True`): Follow the Proximal Regularized Leader. Regularizacion
  L1/L2 por coordenada para regresion logistica.
- **Coin Betting** (`coin_betting=True`): Optimizador propio de VW basado en
  coin betting, con garantias de arrepentimiento logaritmicas.
- **PiSTOL** (`pistol=True`): Parameter-free STOchastic Learning.

Configuracion base (identica para los tres)
-------------------------------------------
- `loss_function="logistic"`, `link="logistic"`
- `passes=10` (numero de pasadas sobre el training fold)
- `random_seed=42`
- `learning_rate=0.5` (por defecto de VW)
- `power_t=0.5` (por defecto de VW)
- Sin tuning de hiperparametros

Estrategia multilabel
---------------------
VWClassifier es binario. Se entrenan tres modelos independientes (uno por etiqueta:
HS, TR, AG), manteniendo el paradigma de V03 de tres heads binarias.

Representacion identica
-----------------------
Se usa exactamente el espacio TF-IDF de V03: word (1,2) + char_wb (3,5), min_df=2,
sublinear_tf=True, ajustado dentro de cada training fold. No se introduce ninguna
feature nueva.

Uso
---
    /home/carlos/venv/bin/python experiments/v44_01_vw_optimizers.py
"""

from __future__ import annotations

import json
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train
from src.evaluation import metricas_multietiqueta
from src.splits import grupos_fusionados, pliegues_corregidos

EXPERIMENTO = "V44-01 (Vowpal Wabbit: FTRL, Coin Betting y PiSTOL sobre el TF-IDF de V03)"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v44_01_vw_optimizers.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "v44_01_vw_optimizers.md"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
PASSES = 10
SEED = 42
UMBRALES = {"HS": 0.44, "TR": 0.51, "AG": 0.46}
ETIQUETAS = ("HS", "TR", "AG")


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V44-01] {mensaje}", flush=True)


def memoria_pico_mb() -> float:
    """Resident set maximum del proceso, en megabytes.

    Returns:
        La memoria maxima residente en MB.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def matriz_v03(textos_train: list[str], textos_val: list[str]) -> tuple[Any, Any, dict[str, Any]]:
    """Representacion exacta de V03: word (1,2) + char_wb (3,5) TF-IDF.

    Args:
        textos_train: textos del training fold.
        textos_val: textos del validation fold.

    Returns:
        Tupla con las matrices de training, validation y las estadisticas.
    """
    from scipy import sparse
    from sklearn.feature_extraction.text import TfidfVectorizer

    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
    train = sparse.hstack([
        word.fit_transform(textos_train),
        char.fit_transform(textos_train),
    ]).tocsr()
    val = sparse.hstack([
        word.transform(textos_val),
        char.transform(textos_val),
    ]).tocsr()
    stats = {
        "word_features": word.transform(textos_val[:1]).shape[1],
        "char_features": char.transform(textos_val[:1]).shape[1],
        "total_features": train.shape[1],
        "train_nnz": int(train.nnz),
        "val_nnz": int(val.nnz),
    }
    return train, val, stats


def evaluar_optimizador(
    train_dev: pd.DataFrame,
    pliegues: list[tuple[np.ndarray, np.ndarray]],
    optimizer: str,
    passes: int = PASSES,
    seed: int = SEED,
) -> dict[str, Any]:
    """Evalua un optimizador de VW sobre los pliegues canonicos.

    Args:
        train_dev: DataFrame del desarrollo.
        pliegues: pliegues canonicos.
        optimizer: nombre del optimizador ("ftrl", "coin_betting" o "pistol").
        passes: numero de pasadas sobre el training fold.
        seed: semilla.

    Returns:
        El resultado de la evaluacion.
    """
    from vowpalwabbit.sklearn import VWClassifier

    y_true = train_dev[list(ETIQUETAS)].to_numpy(dtype=np.int64)
    n = len(train_dev)
    proba = np.zeros((n, 3), dtype=float)
    metricas = []
    train_f1s = []
    tiempos = []
    for numero, (idx_tr, idx_va) in enumerate(pliegues, start=1):
        x_tr, x_va, _ = matriz_v03(
            train_dev["text"].iloc[idx_tr].astype(str).tolist(),
            train_dev["text"].iloc[idx_va].astype(str).tolist(),
        )
        y_tr = y_true[idx_tr]
        pred_train = np.zeros((len(idx_tr), 3), dtype=np.int64)
        t0 = time.time()
        for j, etiqueta in enumerate(ETIQUETAS):
            kwargs = {
                "loss_function": "logistic",
                "link": "logistic",
                "passes": passes,
                "random_seed": seed,
                "quiet": True,
            }
            if optimizer == "ftrl":
                kwargs["ftrl"] = True
            elif optimizer == "coin_betting":
                kwargs["coin"] = True
            elif optimizer == "pistol":
                kwargs["pistol"] = True
            modelo = VWClassifier(**kwargs)
            modelo.fit(x_tr, y_tr[:, j])
            proba[idx_va, j] = modelo.predict_proba(x_va)[:, 1]
            pred_train[:, j] = (
                modelo.predict_proba(x_tr)[:, 1] >= UMBRALES[etiqueta]
            ).astype(int)
        tiempos.append(time.time() - t0)
        pred_pliegue = np.column_stack(
            [(proba[idx_va, j] >= UMBRALES[et]).astype(int) for j, et in enumerate(ETIQUETAS)]
        )
        metrica = metricas_multietiqueta(y_true[idx_va], pred_pliegue)
        metrica["pliegue"] = numero
        metricas.append(metrica)
        train_f1s.append(metricas_multietiqueta(y_true[idx_tr], pred_train)["f1_media_columnas"])
        log(f"   pliegue {numero}: F1 {metrica['f1_media_columnas']:.4f} | tiempo {tiempos[-1]:.2f} s")
    pred = np.column_stack(
        [(proba[:, j] >= UMBRALES[et]).astype(int) for j, et in enumerate(ETIQUETAS)]
    )
    return {
        "proba": proba, "pred": pred, "metricas_pliegues": metricas,
        "train_f1": float(np.mean(train_f1s)),
        "tiempo_total_s": float(sum(tiempos)),
        "tiempo_por_pliegue_s": float(np.mean(tiempos)),
    }


def main() -> None:
    """Ejecuta la bateria V44.

    Returns:
        None.
    """
    inicio = time.time()
    log(f"{EXPERIMENTO}. NINGUNA submission en esta fase")

    train = read_train()
    particion = pd.read_csv(PARTICION)
    ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
    desarrollo = train[train["id"].astype(int).isin(ids_dev)].reset_index(drop=True)
    pliegues = pliegues_corregidos(desarrollo, grupos_fusionados(desarrollo))
    y = desarrollo[list(ETIQUETAS)].to_numpy(dtype=np.int64)
    log(f"   desarrollo {len(desarrollo)} filas | {len(pliegues)} pliegues | passes {PASSES}")

    log("1/4 control TF-IDF sin proyeccion")
    from src.structural_eval import evaluar_bloque, resumen, verificar_control
    control = resumen(evaluar_bloque(desarrollo, pliegues, bloque=None))
    verificar_control(control["corrected_cv"])
    log(f"   control exacto: {control['corrected_cv']:.10f}")

    log("2/4 FTRL")
    ftrl = evaluar_optimizador(desarrollo, pliegues, "ftrl")
    m_ftrl = metricas_multietiqueta(y, ftrl["pred"])
    log(f"   FTRL: corrected_cv {m_ftrl['f1_media_columnas']:.10f} | "
        f"delta {m_ftrl['f1_media_columnas'] - control['corrected_cv']:+.10f}")

    log("3/4 Coin Betting")
    coin = evaluar_optimizador(desarrollo, pliegues, "coin_betting")
    m_coin = metricas_multietiqueta(y, coin["pred"])
    log(f"   Coin: corrected_cv {m_coin['f1_media_columnas']:.10f} | "
        f"delta {m_coin['f1_media_columnas'] - control['corrected_cv']:+.10f}")

    log("4/4 PiSTOL")
    pistol = evaluar_optimizador(desarrollo, pliegues, "pistol")
    m_pistol = metricas_multietiqueta(y, pistol["pred"])
    log(f"   PiSTOL: corrected_cv {m_pistol['f1_media_columnas']:.10f} | "
        f"delta {m_pistol['f1_media_columnas'] - control['corrected_cv']:+.10f}")

    log("5/4 complementariedad y decision")
    pred_v03 = evaluar_bloque(desarrollo, pliegues, bloque=None)["pred"]
    comparaciones = {}
    for nombre, r in (("FTRL", ftrl), ("Coin Betting", coin), ("PiSTOL", pistol)):
        cambios = pred_v03 != r["pred"]
        comp = {
            "corrected_cv": float(metricas_multietiqueta(y, r["pred"])["f1_media_columnas"]),
            "delta_vs_v03": float(metricas_multietiqueta(y, r["pred"])["f1_media_columnas"] - control["corrected_cv"]),
            "rescates": int(np.count_nonzero((pred_v03 != y) & (r["pred"] == y))),
            "danios": int(np.count_nonzero((pred_v03 == y) & (r["pred"] != y))),
            "celdas_modificadas": int(cambios.sum()),
            "correlacion_de_decisiones": float(np.mean(pred_v03.ravel() == r["pred"].ravel())),
            "train_f1": r["train_f1"],
            "gap_train_oof": r["train_f1"] - float(metricas_multietiqueta(y, r["pred"])["f1_media_columnas"]),
            "tiempo_total_s": r["tiempo_total_s"],
        }
        comp["neto"] = comp["rescates"] - comp["danios"]
        comparaciones[nombre] = comp
        log(
            f"   {nombre}: cv {comp['corrected_cv']:.6f} delta {comp['delta_vs_v03']:+.6f} | "
            f"rescates {comp['rescates']} danios {comp['danios']} neto {comp['neto']:+d} | "
            f"correlacion {comp['correlacion_de_decisiones']:.4f}"
        )

    mejor = max(comparaciones.items(), key=lambda kv: kv[1]["corrected_cv"])
    decision = _decision(control, comparaciones)
    log(f"   MEJOR: {mejor[0]} con {mejor[1]['corrected_cv']:.6f}")
    log(f"   DECISION: {decision}")

    informe = {
        "familia": "SPARSE_ONLINE_OPTIMIZATION",
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estado": "OOF_EVALUATED",
        "pregunta": (
            "puede alguno de los optimizadores online/proximal de Vowpal Wabbit encontrar una "
            "solucion lineal que generaliza mejor o complementa a V03 sobre el mismo espacio TF-IDF"
        ),
        "api_utilizada": "vowpalwabbit.sklearn.VWClassifier (API scikit-learn oficial)",
        "version_vowpalwabbit": "9.11.8",
        "configuracion_comun": {
            "loss_function": "logistic",
            "link": "logistic",
            "passes": PASSES,
            "random_seed": SEED,
            "power_t": 0.5,
            "learning_rate": 0.5,
        },
        "control": control,
        "resultados": comparaciones,
        "mejor": {"estrategia": mejor[0], "corrected_cv": mejor[1]["corrected_cv"]},
        "decision": decision,
        "gastos": {
            "submissions": 0,
            "modelos_entrenados": 3,
            "memoria_pico_mb": round(memoria_pico_mb(), 1),
            "segundos": round(time.time() - inicio, 1),
        },
    }
    REPORTE_JSON.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _informe_md(informe)
    log("   informe escrito en reports/v44_01_vw_optimizers.{json,md}")
    log(f"   memoria pico {memoria_pico_mb():.0f} MB | tiempo {round(time.time() - inicio, 1)} s")
    log("VEREDICTO: bateria completada, 0 submissions")


def _decision(control: dict[str, Any], comps: dict[str, Any]) -> str:
    """Decision segun los criterios de la directiva.

    Args:
        control: metricas del control.
        comps: metricas de cada optimizador.

    Returns:
        La decision como cadena.
    """
    base = control["corrected_cv"]
    for nombre, c in comps.items():
        if c["corrected_cv"] > base and c["neto"] > 0 and c["gap_train_oof"] < 0.15:
            return f"SECOND_PHASE_CANDIDATE: {nombre}"
    mejor = max(comps.items(), key=lambda kv: kv[1]["corrected_cv"])
    if mejor[1]["corrected_cv"] < base and mejor[1]["neto"] < 0:
        return "SPARSE_ONLINE_OPTIMIZATION = CLOSED"
    return "WEAK_SIGNAL"


def _informe_md(informe: dict[str, Any]) -> None:
    """Escribe el informe legible.

    Args:
        informe: informe con los datos.

    Returns:
        None.
    """
    informe["control"]
    g = informe["gastos"]
    lineas = [
        "# V44-01: Vowpal Wabbit con FTRL, Coin Betting y PiSTOL",
        "",
        f"Generado por `experiments/v44_01_vw_optimizers.py` ({EXPERIMENTO}).",
        "",
        (f"**Estado `{informe['estado']}`. API: {informe['api_utilizada']}. "
        f"VW {informe['version_vowpalwabbit']}.**"),
        "",
        "## 1. Configuracion",
        "",
        f"- {informe['configuracion_comun']}",
        ("- Tres optimizadores: FTRL (`ftrl=True`), Coin Betting (`coin_betting=True`),"
        " PiSTOL (`pistol=True`)."),
        "- Misma representacion TF-IDF de V03, ajustada dentro de cada training fold.",
        "",
        "## 2. Resultados",
        "",
        "| estrategia | corrected_cv | Δ vs V03 | train | gap | rescates | danios | net | correlacion |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for nombre, r in informe["resultados"].items():
        lineas.append(
            f"| {nombre} | {r['corrected_cv']:.6f} | {r['delta_vs_v03']:+.6f} | "
            f"{r['train_f1']:.4f} | {r['gap_train_oof']:+.4f} | {r['rescates']} | {r['danios']} | "
            f"{r['neto']:+d} | {r['correlacion_de_decisiones']:.4f} |"
        )
    lineas += [
        "",
        f"**Mejor: {informe['mejor']['estrategia']} con {informe['mejor']['corrected_cv']:.6f}.**",
        "",
        f"**DECISION: {informe['decision']}**",
        "",
        "## 3. Lo que este script no hace",
        "",
        "- No envia submission.",
        "- No hace grid de hiperparametros.",
        "- No modifica V03.",
        "",
        (f"Coste: {g['segundos']} s, memoria {g['memoria_pico_mb']:.0f} MB, "
        f"{g['modelos_entrenados']} modelos, {g['submissions']} submissions."),
        "",
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
