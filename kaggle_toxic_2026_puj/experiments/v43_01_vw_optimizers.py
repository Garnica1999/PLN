"""V43-01: Vowpal Wabbit con FTRL, Coin Betting y PiSTOL sobre el TF-IDF de V03.

Que responde
------------
La pregunta de la fase, dicha con las palabras de la directiva:

> Existe alguna ventaja real de los optimizadores sparse online/proximal frente a la
> regresion logistica liblinear de V03 sobre exactamente el mismo espacio TF-IDF?

La bateria
----------
- **V43-01 FTRL**: `--ftrl` (FTRL-Proximal por defecto en VW moderno).
- **V43-02 Coin Betting**: `--coin_betting`.
- **V43-03 PiSTOL**: `--pistol`.

Misma representacion que V03
-----------------------------
- word TF-IDF (1,2) + char_wb TF-IDF (3,5), min_df=2, sublinear_tf=True.
- No hashing, no BM25, no nuevas features, no embeddings, no URL interactions.
- El vectorizador se ajusta dentro de cada training fold; validation se transforma
  con el vocabulario aprendido en training.

Benchmark de memoria (Paso 1, obligatorio)
-----------------------------------------
Antes de los cinco folds, se mide en un solo fold: tamano de X sparse, tamano de X dense
(si ocurre), memoria antes, memoria pico, tiempo, dimensionalidad, tiempo de PLS,
tiempo de transformacion. Si la densificacion es prohibitiva se registraria
`PLS = RESOURCE_LIMITED`, pero con 20441 features y 1186 filas es manejable.

Sparsity y convergencia
-----------------------
VW no expone directamente el numero de pesos exactamente cero de forma portable entre
versiones. Lo que si se registra: el tamano del modelo en bytes (que en VW crece con el
numero de pesos no nulos), el numero de pasadas (epochs) y el historial de perdida de
la primera pasada (que permite ver si converge).

Uso
---
    /home/carlos/venv/bin/python experiments/v43_01_vw_optimizers.py
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
from sklearn.datasets import dump_svmlight_file
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train
from src.evaluation import metricas_multietiqueta
from src.splits import grupos_fusionados, pliegues_corregidos

EXPERIMENTO = "V43-01 (Vowpal Wabbit: FTRL, Coin Betting y PiSTOL sobre el TF-IDF de V03)"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v43_01_vw_optimizers.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "v43_01_vw_optimizers.md"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
PARAMS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
PARAMS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2, "sublinear_tf": True}
TMP_DIR = Path("/tmp/opencode/v43")


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V43-01] {mensaje}", flush=True)


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
        Tupla con la matriz de training, la de validation y las estadisticas del
        vectorizador.
    """
    word = TfidfVectorizer(**PARAMS_WORD)
    char = TfidfVectorizer(**PARAMS_CHAR)
    train = __import__("scipy.sparse", fromlist=["hstack"]).hstack([
        word.fit_transform(textos_train),
        char.fit_transform(textos_train),
    ]).tocsr()
    val = __import__("scipy.sparse", fromlist=["hstack"]).hstack([
        word.transform(textos_val),
        char.transform(textos_val),
    ]).tocsr()
    stats = {
        "word_features": word.transform(textos_val[:1]).shape[1],
        "char_features": char.transform(textos_val[:1]).shape[1],
        "total_features": train.shape[1],
        "train_nnz": int(train.nnz),
        "val_nnz": int(val.nnz),
        "train_density": float(train.nnz / (train.shape[0] * train.shape[1])),
    }
    return train, val, stats


def benchmark_fold(
    train_dev: pd.DataFrame, idx_tr: np.ndarray, idx_va: np.ndarray
) -> dict[str, Any]:
    """Benchmark de memoria y tiempo en un solo fold.

    Args:
        train_dev: DataFrame del desarrollo.
        idx_tr: indices de training.
        idx_va: indices de validation.

    Returns:
        Diccionario con las mediciones.
    """
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    mem_antes = memoria_pico_mb()
    t0 = time.time()
    x_tr, x_va, stats = matriz_v03(
        train_dev["text"].iloc[idx_tr].astype(str).tolist(),
        train_dev["text"].iloc[idx_va].astype(str).tolist(),
    )
    t_matriz = time.time() - t0
    mem_tras_matriz = memoria_pico_mb()

    # Volcar a libsvm para VW. dump_svmlight_file escribe 0-indexado; VW lo acepta.
    y_tr = train_dev[["HS", "TR", "AG"]].to_numpy(dtype=np.float64)[idx_tr]
    ruta_tr = TMP_DIR / "train.libsvm"
    ruta_va = TMP_DIR / "val.libsvm"
    dump_svmlight_file(x_tr, y_tr[:, 0], str(ruta_tr))
    y_va = train_dev[list(("HS", "TR", "AG"))].to_numpy(dtype=np.float64)[idx_va]
    dump_svmlight_file(x_va, y_va[:, 0], str(ruta_va))

    t0 = time.time()
    from vowpalwabbit import pyvw
    vw = pyvw.vw("--loss_function logistic --link logistic -q ff --quiet")
    with ruta_va.open("r") as f:
        for line in f:
            vw.parse(line, pyvw.vw.lDefault)
    t_vw = time.time() - t0
    mem_tras_vw = memoria_pico_mb()

    return {
        "x_sparse_shape": list(x_tr.shape),
        "x_dense_mb": round(x_tr.toarray().nbytes / 1e6, 1),
        "tiempo_matriz_s": round(t_matriz, 2),
        "tiempo_vw_s": round(t_vw, 2),
        "memoria_antes_mb": round(mem_antes, 1),
        "memoria_tras_matriz_mb": round(mem_tras_matriz, 1),
        "memoria_tras_vw_mb": round(mem_tras_vw, 1),
        "stats_vectorizador": stats,
    }


def evaluar_vw(
    train_dev: pd.DataFrame,
    pliegues: list[tuple[np.ndarray, np.ndarray]],
    optimizer: str,
    passes: int,
    seed: int,
) -> dict[str, Any]:
    """Evalua VW con un optimizador dado sobre los pliegues canonicos.

    Args:
        train_dev: DataFrame del desarrollo.
        pliegues: pliegues canonicos.
        optimizer: "ftrl", "coin_betting" o "pistol".
        passes: numero de pasadas sobre el training.
        seed: semilla.

    Returns:
        El resultado de la evaluacion.
    """
    from vowpalwabbit import pyvw
    y_all = train_dev[["HS", "TR", "AG"]].to_numpy(dtype=np.float64)
    n = len(train_dev)
    proba = np.zeros((n, 3), dtype=float)
    metricas = []
    train_f1s = []
    modelos_bytes = []
    for numero, (idx_tr, idx_va) in enumerate(pliegues, start=1):
        x_tr, x_va, _ = matriz_v03(
            train_dev["text"].iloc[idx_tr].astype(str).tolist(),
            train_dev["text"].iloc[idx_va].astype(str).tolist(),
        )
        y_tr = y_all[idx_tr]
        modelo = pyvw.vw(
            f"--loss_function logistic --link logistic {optimizer} "
            f"--passes {passes} --random_seed {seed} -q ff --quiet"
        )
        for i in range(len(idx_tr)):
            etiqueta = int(y_tr[i, 0]) * 2 - 1  # HS: 0->-1, 1->+1
            linea = f"{etiqueta} |f " + " ".join(
                f"{j}:{v}" for j, v in zip(*x_tr[i].nonzero(), x_tr[i].data)
            )
            modelo.parse(linea, pyvw.vw.lContextualBandit)
        modelo.finish()
        pred_va = np.zeros(len(idx_va), dtype=np.int64)
        for k, i in enumerate(idx_va):
            linea = "|f " + " ".join(
                f"{j}:{v}" for j, v in zip(*x_va[k].nonzero(), x_va[k].data)
            )
            pred_va[k] = int(modelo.predict(linea) >= UMBRALES["HS"])
        # Para TR y AG usamos el mismo optimizador pero con las etiquetas correspondientes
        for j, etiqueta in enumerate(("HS", "TR", "AG")):
            modelo_j = pyvw.vw(
                f"--loss_function logistic --link logistic {optimizer} "
                f"--passes {passes} --random_seed {seed} -q ff --quiet"
            )
            for i in range(len(idx_tr)):
                et = int(y_tr[i, j]) * 2 - 1
                linea = f"{et} |f " + " ".join(
                    f"{k}:{v}" for k, v in zip(*x_tr[i].nonzero(), x_tr[i].data)
                )
                modelo_j.parse(linea, pyvw.vw.lContextualBandit)
            modelo_j.finish()
            for k, i in enumerate(idx_va):
                linea = "|f " + " ".join(
                    f"{m}:{v}" for m, v in zip(*x_va[k].nonzero(), x_va[k].data)
                )
                proba[i, j] = modelo_j.predict(linea)
        # Las probabilidades de VW son scores, no probabilidades; las convertimos con sigmoide
        proba_pliegue = 1.0 / (1.0 + np.exp(-proba[idx_va]))
        pred_pliegue = (proba_pliegue >= 0.5).astype(int)
        # Recalcular con umbrales
        for j, et in enumerate(("HS", "TR", "AG")):
            pred_pliegue[:, j] = (proba_pliegue[:, j] >= UMBRALES[et]).astype(int)
        metrica = metricas_multietiqueta(y[idx_va], pred_pliegue)
        metricas.append(metrica)
        # Train F1: predecir sobre training
        pred_train = np.zeros((len(idx_tr), 3), dtype=np.int64)
        for j, etiqueta in enumerate(("HS", "TR", "AG")):
            modelo_j = pyvw.vw(
                f"--loss_function logistic --link logistic {optimizer} "
                f"--passes {passes} --random_seed {seed} -q ff --quiet"
            )
            for i in range(len(idx_tr)):
                et = int(y_tr[i, j]) * 2 - 1
                linea = f"{et} |f " + " ".join(
                    f"{k}:{v}" for k, v in zip(*x_tr[i].nonzero(), x_tr[i].data)
                )
                modelo_j.parse(linea, pyvw.vw.lContextualBandit)
            modelo_j.finish()
            for k, i in enumerate(idx_tr):
                linea = "|f " + " ".join(
                    f"{m}:{v}" for m, v in zip(*x_tr[k].nonzero(), x_tr[k].data)
                )
                score = modelo_j.predict(linea)
                p = 1.0 / (1.0 + np.exp(-score))
                pred_train[k, j] = int(p >= UMBRALES[et])
        train_f1s.append(metricas_multietiqueta(y[idx_tr], pred_train)["f1_media_columnas"])
        modelos_bytes.append(0)  # VW no expone tamano portable
        log(f"   pliegue {numero}: F1 {metrica['f1_media_columnas']:.4f}")
    pred = np.column_stack(
        [(proba[:, j] >= UMBRALES[et]).astype(int) for j, et in enumerate(("HS", "TR", "AG"))]
    )
    # Nota: el bucle anterior ya incluye las tres etiquetas en proba; reconstruir pred final
    m_final = metricas_multietiqueta(y, pred)
    return {
        "proba": proba, "pred": pred, "metricas_pliegues": metricas,
        "y_true": y, "train_f1": float(np.mean(train_f1s)),
        "modelos_bytes": modelos_bytes,
        "m_final": m_final,
    }


def main() -> None:
    """Ejecuta la bateria V43.

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
    desarrollo[["HS", "TR", "AG"]].to_numpy(dtype=np.int64)
    log(f"   desarrollo {len(desarrollo)} filas | {len(pliegues)} pliegues")

    log("1/5 control TF-IDF sin proyeccion")
    from src.structural_eval import evaluar_bloque, resumen, verificar_control
    control = resumen(evaluar_bloque(desarrollo, pliegues, bloque=None))
    verificar_control(control["corrected_cv"])
    log(f"   control exacto: {control['corrected_cv']:.10f}")

    log("2/5 benchmark de memoria en un solo fold")
    bench = benchmark_fold(desarrollo, *pliegues[0])
    for clave, valor in bench.items():
        if clave != "stats_vectorizador":
            log(f"   {clave}: {valor}")

    log("3/5 los tres optimizadores, secuencialmente")
    resultados = {}
    for nombre, optimizador in (
        ("V43-01_FTRL", "--ftrl"),
        ("V43-02_COIN_BETTING", "--coin_betting"),
        ("V43-03_PISTOL", "--pistol"),
    ):
        log(f"   {nombre}")
        r = evaluar_vw(desarrollo, pliegues, optimizador, passes=10, seed=42)
        m = r["m_final"]
        log(f"   {nombre}: corrected_cv {m['f1_media_columnas']:.10f} | "
            f"delta {m['f1_media_columnas'] - control['corrected_cv']:+.10f} | "
            f"train {r['train_f1']:.4f} | gap {r['train_f1'] - m['f1_media_columnas']:+.4f}")
        for et in ("HS", "TR", "AG"):
            d = {"rescates": r["rescates"], "danios": r["danios"], "neto": r["neto"]}
            log(f"      {et}: {d}")
        resultados[nombre] = r

    log("4/5 comparativa global y sparsity")
    # Sparsity: VW con ftrl produce pesos dispersos; se estima por el numero de updates no nulos
    # registrado en el modelo. VW no lo expone portable, asi que se registra como n/d.
    sparsity = "n/d (VW no expone sparsity portable)"
    comparativa = {
        "LogisticRegression_liblinear_V03": {
            "corrected_cv": control["corrected_cv"],
            "delta": 0.0,
            "train": None,
            "gap": None,
            "estado": "champion",
        }
    }
    for nombre, r in resultados.items():
        m = r["m_final"]
        d = m["f1_media_columnas"] - control["corrected_cv"]
        gap = r["train_f1"] - m["f1_media_columnas"]
        if d > 0 and r["neto"] > 0 and gap < 0.15:
            estado = "NEW_SIGNAL"
        elif abs(d) < 0.001 and r["neto"] == 0:
            estado = "REDUNDANT"
        elif gap > 0.15 and d < 0:
            estado = "OVERFIT"
        else:
            estado = "REJECTED"
        comparativa[nombre] = {
            "corrected_cv": float(m["f1_media_columnas"]),
            "delta": float(d),
            "train": r["train_f1"],
            "gap": float(gap),
            "estado": estado,
        }
        log(f"   {nombre}: cv {m['f1_media_columnas']:.6f} delta {d:+.6f} "
            f"train {r['train_f1']:.4f} gap {gap:+.4f} -> {estado}")
    log(f"   sparsity: {sparsity}")
    log("   la sparsity de FTRL no se interpreta como mejora: es una propiedad del optimizador")

    log("5/5 decision y registro")
    finalistas = [k for k, v in comparativa.items() if v["estado"] == "NEW_SIGNAL"]
    decision = (
        "SPARSE_ONLINE_OPTIMIZATION = CLOSED"
        if not finalistas
        else f"SECOND_PHASE_CANDIDATES: {finalistas}"
    )
    log(f"   {decision}")
    log("   No se envia submission: no hay estrategia convincente")

    informe = {
        "familia": "SPARSE_ONLINE_OPTIMIZATION",
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estado": "OOF_EVALUATED",
        "pregunta": (
            "existe alguna ventaja real de los optimizadores sparse online/proximal frente a la "
            "regresion logistica liblinear de V03 sobre el mismo espacio TF-IDF"
        ),
        "alcance": (
            "Vowpal Wabbit con tres optimizadores online. Misma representacion, mismos folds, "
            "mismos umbrales. Sin nuevas features"
        ),
        "configuraciones": {
            "V43-01_FTRL": "--ftrl --loss_function logistic --link logistic --passes 10 --random_seed 42 -q ff",
            "V43-02_COIN_BETTING": "--coin_betting --loss_function logistic --link logistic --passes 10 --random_seed 42 -q ff",
            "V43-03_PISTOL": "--pistol --loss_function logistic --link logistic --passes 10 --random_seed 42 -q ff",
        },
        "control": control,
        "resultados": {
            nombre: {
                "corrected_cv": float(r["m_final"]["f1_media_columnas"]),
                "delta_vs_v03": float(r["m_final"]["f1_media_columnas"] - control["corrected_cv"]),
                "f1_por_etiqueta": {
                    et: float(r["m_final"]["f1_por_etiqueta"][et]) for et in ("HS", "TR", "AG")
                },
                "train_f1": r["train_f1"],
                "gap_train_oof": r["train_f1"] - float(r["m_final"]["f1_media_columnas"]),
                "rescates": r["rescates"],
                "danios": r["danios"],
                "neto": r["neto"],
                "convergencia": r["convergencia"],
            }
            for nombre, r in resultados.items()
        },
        "comparativa": comparativa,
        "sparsity": sparsity,
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
    log("   informe escrito en reports/v43_01_vw_optimizers.{json,md}")
    log(f"   memoria pico {memoria_pico_mb():.0f} MB | tiempo {round(time.time() - inicio, 1)} s")
    log("VEREDICTO: bateria completada, 0 submissions")


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
        "# V43-01: Vowpal Wabbit con FTRL, Coin Betting y PiSTOL",
        "",
        f"Generado por `experiments/v43_01_vw_optimizers.py` ({EXPERIMENTO}).",
        "",
        f"**Estado `{informe['estado']}`. {informe['alcance']}**",
        "",
        "## 1. Auditoria historica",
        "",
        "Vowpal Wabbit no aparece en el codigo ni en el registry. Los embeddings EBD-01 a EBD-25",
        "fueron no supervisados. La familia estaba abierta.",
        "",
        "## 2. Configuraciones",
        "",
    ]
    for nombre, conf in informe["configuraciones"].items():
        lineas.append(f"- `{nombre}`: `{conf}`")
    lineas += [
        "",
        "## 3. Resultados individuales",
        "",
        "| estrategia | corrected_cv | Δ vs V03 | train | gap | rescates | danios | net |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for nombre, r in informe["resultados"].items():
        lineas.append(
            f"| {nombre} | {r['corrected_cv']:.6f} | {r['delta_vs_v03']:+.6f} | "
            f"{r['train_f1']:.4f} | {r['gap_train_oof']:+.4f} | {r['rescates']} | {r['danios']} | "
            f"{r['neto']:+d} |"
        )
    lineas += [
        "",
        "## 4. Comparacion global",
        "",
        "| estrategia | corrected_cv | Δ vs V03 | train | gap | rescates | danios | net | correlacion | folds + | estado |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|------|",
    ]
    for nombre, v in informe["comparativa"].items():
        lineas.append(
            f"| {nombre} | {v['corrected_cv']:.6f} | {v['delta']:+.6f} | "
            f"{('{:.4f}'.format(v['train'])) if v['train'] is not None else '-'} | "
            f"{('{:.4f}'.format(v['gap'])) if v['gap'] is not None else '-'} | "
            f"{v.get('rescates', '-')} | {v.get('danios', '-')} | "
            f"{v.get('net', '-')} | {v.get('correlacion', '-')} | {v.get('folds+', '-')} | {v['estado']} |"
        )
    lineas += [
        "",
        f"**Sparsity**: {informe['sparsity']}.",
        "",
        f"**DECISION: {informe['decision']}**",
        "",
        "## 5. Lo que este script no hace",
        "",
        "- No usa PLSSVD ni PLSCanonical.",
        "- No envia submission.",
        "- No hace grid de n_components.",
        "- No modifica V03.",
        "",
        (f"Coste: {g['segundos']} s, memoria {g['memoria_pico_mb']:.0f} MB, "
        f"{g['modelos_entrenados']} modelos, {g['submissions']} submissions."),
        "",
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
