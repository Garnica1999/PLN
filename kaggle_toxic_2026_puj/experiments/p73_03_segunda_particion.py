"""P73-03: la segunda particion agrupada. Robustez del hallazgo positivo de P73-02.

Por que existe
--------------
P73-02 encontro `A_SUPERA_A_B_CLARO`: delta +0.0316259472 con IC 95 por clusters
[+0.010066, +0.053484], enteramente por encima de cero, y 4 de 5 pliegues a favor de A. Un
resultado POSITIVO es exactamente donde este proyecto exige mas escepticismo, porque su historia
tiene tres falsos positivos producidos por fallos silenciosos.

La amenaza principal: los pliegues externos son UNO de los repartos posibles de la misma
agrupacion de near-duplicados. Si el delta dependiera del reparto concreto, el hallazgo seria un
artefacto de la particion.

Que se hace: se repite el contraste completo con DOS particiones adicionales. La agrupacion por
near-duplicados NO cambia; solo cambia el reparto en pliegues. Todas las particiones se reportan,
ninguna se selecciona, y ninguna sustituye a la congelada.

Lo que NO se hace, por mandato de la fase: no se cambia la semilla del modelo, ni la arquitectura,
ni la dimension, ni el corpus, ni se prueba otro kernel, ni otro dropout, ni otro learning rate.
La unica palanca que se mueve es el reparto en pliegues.
"""

from __future__ import annotations

import ast
import csv as _csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p73_02_textcnn_vs_random import (
    MODELOS,
    N_BOOTSTRAP,
    REPORTS,
    SEMILLA,
    SEMILLA_BOOT,
    UMBRALES,
    TextCNN,
    asignar_ids,
    bootstrap_clusters,
    entrenar_pliegue,
    f1m,
    log,
    log_odds,
    partir_interna,
    sha_matriz,
    tokenizar,
)

from src.data import read_train
from src.splits import combinaciones, grupos_fusionados, holdout_v2

SEMILLAS_PARTICION = (20261103, 20261104)


def pliegues_alternativos(des, grupos, semilla: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Reparto alternativo en 5 pliegues, con la MISMA agrupacion por near-duplicados.

    Args:
        des: DataFrame de desarrollo.
        grupos: etiqueta de cluster por fila.
        semilla: semilla del reparto.

    Returns:
        Lista de tuplas (entrenamiento, validacion).
    """
    from sklearn.model_selection import StratifiedGroupKFold
    div = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=semilla)
    out = []
    for a, b in div.split(des, combinaciones(des), groups=grupos):
        out.append((np.asarray(a), np.asarray(b)))
    return out


def correr(ids, largos, y, grupos, pl, matriz_a, matriz_b, init_comun, etiqueta, tx_check):
    """Corre las dos ramas sobre una particion y devuelve el contraste.

    Args:
        ids: ids de todas las filas.
        largos: longitudes.
        y: etiquetas.
        grupos: clusters.
        pl: pliegues de esta particion.
        matriz_a: matriz entrenada.
        matriz_b: matriz aleatoria.
        init_comun: state_dict inicial comun.
        etiqueta: nombre de la particion.
        tx_check: comprobacion de que las filas de validacion suman el total.

    Returns:
        Diccionario con el resultado de la particion.
    """
    assert tx_check, "los pliegues no cubren todas las filas"
    res = {}
    for nombre, matriz in (("WORD2VEC", matriz_a), ("ALEATORIO", matriz_b)):
        pr = np.zeros((len(y), 3))
        for k, (ia, ib) in enumerate(pl, start=1):
            m_in, _ = partir_interna(grupos[ia], y[ia], SEMILLA + 100 * k + 10_000)
            proba, _reg = entrenar_pliegue(
                matriz, init_comun,
                ids[ia][m_in], largos[ia][m_in], y[ia][m_in],
                ids[ib], largos[ib], y[ib],
                y[ia][m_in].mean(axis=0))
            pr[ib] = proba
        res[nombre] = pr
    pred_a = (res["WORD2VEC"] >= UMBRALES[None, :]).astype(int)
    pred_b = (res["ALEATORIO"] >= UMBRALES[None, :]).astype(int)
    cv_a, cv_b = f1m(y, pred_a), f1m(y, pred_b)
    d_pl = [round(f1m(y[np.asarray(b)], pred_a[np.asarray(b)])
                  - f1m(y[np.asarray(b)], pred_b[np.asarray(b)]), 6) for _, b in pl]
    media, lo, hi = bootstrap_clusters(y, pred_a, pred_b, grupos, SEMILLA_BOOT, N_BOOTSTRAP)
    fuera = bool(np.isfinite(res["WORD2VEC"]).all() and np.isfinite(res["ALEATORIO"]).all())
    log(f"   particion {etiqueta}: A {cv_a:.10f} | B {cv_b:.10f} | delta {cv_a - cv_b:+.10f} | "
        f"gana A en {sum(1 for d in d_pl if d > 0)}/5 | IC95 [{lo:+.6f}, {hi:+.6f}] | "
        f"todo finito {fuera}")
    log(f"      delta por pliegue {d_pl}")
    return {"particion": etiqueta, "a": round(cv_a, 10), "b": round(cv_b, 10),
            "delta": round(cv_a - cv_b, 10), "delta_por_pliegue": d_pl,
            "pliegues_que_gana_a": sum(1 for d in d_pl if d > 0),
            "bootstrap_media": media, "ic95_bajo": lo, "ic95_alto": hi,
            "ic95_excluye_cero": bool(lo > 0 or hi < 0), "todo_finito": fuera,
            "sha_matriz_a": sha_matriz(matriz_a), "sha_matriz_b": sha_matriz(matriz_b)}


def main() -> None:
    """Ejecuta el contraste sobre las particiones alternativas.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    torch.set_num_threads(32)
    log("P73-03 segunda particion agrupada: robustez del hallazgo de P73-02")

    tr = read_train()
    g = grupos_fusionados(tr)
    idev, _ = holdout_v2(tr, g)
    des = tr.iloc[idev].reset_index(drop=True)
    grupos = g[idev]
    y = np.load(PROJECT_ROOT / "features" / "oof_v03_final.npz", allow_pickle=True)["y_true"]
    y = y.astype(int)
    docs = tokenizar(des["text"])
    d100 = np.load(MODELOS / "w2v_twitter_100d.npz", allow_pickle=True)
    vocab = {w: i + 1 for i, w in enumerate(d100["palabras"])}
    W = d100["W"].astype(np.float32)
    normas = np.linalg.norm(W, axis=1)
    ids, largos = asignar_ids(docs, vocab)
    matriz_a = np.zeros((len(vocab) + 1, W.shape[1]), dtype=np.float32)
    matriz_a[1:] = W
    rng = np.random.default_rng(SEMILLA + 1)
    U = rng.standard_normal(W.shape)
    U /= np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)
    matriz_b = np.zeros_like(matriz_a)
    matriz_b[1:] = (normas[:, None] * U).astype(np.float32)
    torch.manual_seed(SEMILLA)
    _t = TextCNN(matriz_a)
    with torch.no_grad():
        _t.fc2.bias.copy_(torch.as_tensor(log_odds(y.mean(axis=0)), dtype=torch.float32))
    init_comun = {k: v.detach().clone() for k, v in _t.state_dict().items()
                  if not k.startswith("emb.") and k != "fc2.bias"}
    log(f"   matrices fijas: A {sha_matriz(matriz_a)[:12]} | B {sha_matriz(matriz_b)[:12]}")
    log("   nota: las matrices son EXACTAMENTE las de P73-02, luego la unica palanca es el reparto")

    resultados = []
    t0 = time.time()
    for sem in SEMILLAS_PARTICION:
        pl = pliegues_alternativos(des, grupos, sem)
        cubre = sum(len(b) for _, b in pl) == len(y)
        resultados.append(correr(ids, largos, y, grupos, pl, matriz_a, matriz_b, init_comun,
                                 f"particion_{sem}", cubre))
    total = time.time() - t0

    deltas = [r["delta"] for r in resultados]
    fuera = [r["ic95_excluye_cero"] for r in resultados]
    positivos = [r["delta"] > 0 for r in resultados]
    estables = all(positivos) and all(fuera)
    log("")
    log("   RESUMEN DE LA ROBUSTEZ")
    for r in resultados:
        log(f"      {r['particion']}: delta {r['delta']:+.10f} | A gana en "
            f"{r['pliegues_que_gana_a']}/5 | IC95 [{r['ic95_bajo']:+.6f}, {r['ic95_alto']:+.6f}]")
    log(f"      delta medio de las particiones alternativas {np.mean(deltas):+.10f} | "
        f"rango [{min(deltas):+.6f}, {max(deltas):+.6f}]")
    veredicto = "ROBUSTO" if estables else ("NO_ROBUSTO" if not all(positivos) else "PARCIAL")
    log(f"      VEREDICTO DE ROBUSTEZ: {veredicto}")

    salida = {
        "familia": "P73_TEXTCNN_WORD2VEC_RANDOM", "experiment_id": "P73-03",
        "fase": "segunda particion agrupada, robustez del hallazgo positivo",
        "fecha": inicio.isoformat(timespec="seconds"),
        "naturaleza": "ROBUSTEZ. Dos particiones adicionales sobre las MISMAS matrices y el mismo "
                      "protocolo. Ninguna sustituye a la particion congelada",
        "que_cambia": "solo el reparto en pliegues: la agrupacion por near-duplicados no cambia",
        "que_NO_cambia": ["semilla del modelo", "arquitectura", "dimension", "corpus",
                          "matriz de embeddings entrenada", "matriz aleatoria",
                          "init de la red", "umbrales", "regla de evaluacion"],
        "semillas_de_particion": list(SEMILLAS_PARTICION),
        "resultados": resultados,
        "delta_medio": round(float(np.mean(deltas)), 10),
        "delta_minimo": round(float(min(deltas)), 10),
        "delta_maximo": round(float(max(deltas)), 10),
        "todas_positivas": bool(all(positivos)),
        "todos_los_intervalos_excluyen_cero": bool(all(fuera)),
        "segundos": round(total, 1),
        "veredicto": veredicto,
        "referencia_p73_02": {"delta": 0.0316259472, "ic95": [0.0100657802, 0.0534837793],
                              "particion": "la congelada"},
        "limites": [("el hallazgo positivo de P73-02 viene de UNA sola tirada del control "
                    "aleatorio. Dos particiones adicionales no miden la varianza de esa tirada, "
                    "que es la amenaza principal y no se puede medir con el mandato vigente"),
                    ("el mandato de la fase prohibe otra semilla del modelo, luego esto mide "
                    "varianza de PARTICION, no varianza de inicializacion")],
        "test_tocado": False, "holdout_tocado": False,
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    (REPORTS / "p73_segunda_particion.json").write_text(
        json.dumps(salida, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (REPORTS / "p73_segunda_particion.csv").open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["particion", "a_word2vec", "b_aleatorio", "delta", "pliegues_que_gana_a",
                    "ic95_bajo", "ic95_alto", "ic95_excluye_cero",
                    "delta_pl1", "delta_pl2", "delta_pl3", "delta_pl4", "delta_pl5"])
        w.writerow(["particion_congelada_p73_02", 0.6090700099, 0.5774440627, 0.0316259472, 4,
                    0.0100657802, 0.0534837793, True, 0.061471, -0.053772, 0.064504, 0.060351,
                    0.022266])
        for r in resultados:
            w.writerow([r["particion"], r["a"], r["b"], r["delta"], r["pliegues_que_gana_a"],
                        r["ic95_bajo"], r["ic95_alto"], r["ic95_excluye_cero"],
                        *r["delta_por_pliegue"]])
        w.writerow([])
        w.writerow(["delta_medio", round(float(np.mean(deltas)), 10), "veredicto", veredicto])
    sucios = [n for n in ("p73_segunda_particion.json", "p73_segunda_particion.csv")
              if any(b > 127 for b in (REPORTS / n).read_bytes())]
    if sucios:
        raise SystemExit(f"PARADA: documentos con bytes no ASCII: {sucios}")
    log("")
    log("=" * 78)
    log(f"P73-03 OK | ROBUSTEZ {veredicto} | deltas {deltas}")
    log("=" * 78)


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
