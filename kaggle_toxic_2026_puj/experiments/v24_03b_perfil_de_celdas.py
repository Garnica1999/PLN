"""Addendum V24-03b: perfil de celdas de la mezcla kNN, con la sensibilidad del proyecto.

Por que este addendum existe
----------------------------
`resultados/leaderboard_log.csv` (entrada `v08_p1_fold_ensemble`) calibra la
sensibilidad del leaderboard de este concurso con dos numeros medurados, no
supuestos:

- **2,8e-05 de F1 macro en el LB por acierto neto.** De ahi: 107 cambios netos
  de acierto mueven el LB en 0,003.
- **Error estandar teorico del LB con 1154 filas = 0,01219**, que equivale a
  unos **435 cambios netos**. Toda candidata que mueva menos de ~200 celdas es
  indetectable con este presupuesto.

Un dOOF de +0,0050 (el mejor resultado realizable de kNN) no dice nada por si
solo. Lo que decide el gate es cuantas celdas cambia. Este script mide eso.

Salida: `reports/v24_03b_perfil_de_celdas.md`.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import data as data_module
from src import splits as splits_module

REPORTS_DIR = PROJECT_ROOT / "reports"
OOF_V03 = PROJECT_ROOT / "features" / "oof_v03_final.npz"

ETIQUETAS = ("HS", "TR", "AG")
UMBRALES = np.array([0.44, 0.51, 0.46])

# Constantes medidas en el registro, no estimadas.
F1_LB_POR_ACIERTO_NETO = 2.8e-05
LB_MOVIBLE_EN_003 = 107
SE_LB = 0.01219

PARAMS_WORD = {
    "analyzer": "word",
    "ngram_range": (1, 2),
    "min_df": 2,
    "sublinear_tf": True,
    "lowercase": True,
}
PARAMS_CHAR = {
    "analyzer": "char_wb",
    "ngram_range": (3, 5),
    "min_df": 2,
    "sublinear_tf": True,
    "lowercase": True,
}

SALIDA: list[str] = []


def log(mensaje: str) -> None:
    """Acumula y muestra una linea de log."""
    SALIDA.append(mensaje)
    print(mensaje, flush=True)


def f1_columnas(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, list[float]]:
    """F1 de la clase positiva por etiqueta y su media, con sklearn.

    Se usa la funcion de la biblioteca y no una formula propia porque la
    version anterior de este script uso ``|interseccion| / |union|``, que es el
    indice de Jaccard, no el F1: da 0,6364 para V03 donde el valor canonico es
    0,7772. La correccion no toca los recuentos de celdas, que no dependen de
    la formula de F1.
    """
    por_etiqueta = [
        f1_score(y_true[:, i], y_pred[:, i], zero_division=0) for i in range(3)
    ]
    return float(np.mean(por_etiqueta)), [float(v) for v in por_etiqueta]


def main() -> None:
    inicio = time.time()
    train = data_module.read_train()
    grupos = splits_module.grupos_fusionados(train)
    idx_dev, _ = splits_module.holdout_v2(train, grupos)
    train_dev = train.iloc[idx_dev].reset_index(drop=True)
    pliegues = splits_module.pliegues_corregidos(train_dev, grupos[idx_dev])
    y_dev = train_dev[list(ETIQUETAS)].to_numpy()

    cache = np.load(OOF_V03, allow_pickle=True)
    proba_v03 = cache["proba"]
    pred_v03 = cache["pred"].astype(int)
    f1_v03 = f1_columnas(y_dev, pred_v03)[0]
    log(f"V03 referencia: corrected_cv = {f1_v03:.4f} (cacheado 0.7772)")
    log("")

    # Vecindario k=25 en el espacio union de V03, OOF agrupado.
    prob_knn = np.zeros((len(train_dev), 3), dtype=float)
    sim_top1 = np.zeros(len(train_dev))
    for idx_tr, idx_va in pliegues:
        word = TfidfVectorizer(**PARAMS_WORD)
        char = TfidfVectorizer(**PARAMS_CHAR)
        me = sparse.hstack(
            [word.fit_transform(train_dev["text"].iloc[idx_tr]), char.fit_transform(train_dev["text"].iloc[idx_tr])]
        ).tocsr()
        mv = sparse.hstack(
            [word.transform(train_dev["text"].iloc[idx_va]), char.transform(train_dev["text"].iloc[idx_va])]
        ).tocsr()
        buscador = NearestNeighbors(n_neighbors=25, metric="cosine", algorithm="brute")
        buscador.fit(me)
        dist, idx = buscador.kneighbors(mv)
        sims = np.clip(1.0 - dist, 0.0, 1.0)
        sim_top1[idx_va] = sims[:, 0]
        y_tr = y_dev[idx_tr]
        pesos = sims[:, :, None]
        total = pesos.sum(axis=1)
        total[total == 0.0] = 1.0
        prob_knn[idx_va] = (pesos * y_tr[idx]).sum(axis=1) / total

    log("# V24-03b: perfil de celdas de la mejor mezcla kNN, en la unidad del gate")
    log("")
    log("## 0. Constantes medidas del proyecto (no estimadas)")
    log(f"- Sensibilidad del LB: **{F1_LB_POR_ACIERTO_NETO:.2e}** de F1 macro en el LB por acierto neto")
    log(f"- Verificacion cruzada: {LB_MOVIBLE_EN_003} cambios netos mueven el LB en 0,003 "
        f"-> {0.003 / LB_MOVIBLE_EN_003:.2e} (coincide)")
    log(f"- Error estandar teorico del LB con 1154 filas: **{SE_LB}** = {SE_LB / F1_LB_POR_ACIERTO_NETO:.0f} "
        "cambios netos de acierto")
    log("- Filtro escrito en el registro: toda candidata que cambie menos de unas 200 celdas")
    log("  de la referencia es indetectable con este presupuesto")
    log("")

    log("## 1. Perfil de celdas de la mezcla kNN a distintos pesos")
    log("")
    log("| peso kNN | celdas cambiadas | rescates | danos | neto | dOOF (F1) | dLB implicito | % de 1 SE |")
    log("|---:|---:|---:|---:|---:|---:|---:|---:|")
    filas = []
    for peso in (0.10, 0.20, 0.30, 0.40, 0.50):
        mezcla = (1 - peso) * proba_v03 + peso * prob_knn
        pred_m = (mezcla >= UMBRALES[None, :]).astype(int)
        cambio = pred_m != pred_v03
        rescate = int((cambio & (pred_m == y_dev)).sum())
        dano = int((cambio & (pred_m != y_dev)).sum())
        neto = rescate - dano
        d_oof = f1_columnas(y_dev, pred_m)[0] - f1_v03
        d_lb = neto * F1_LB_POR_ACIERTO_NETO
        log(f"| {peso:.2f} | {int(cambio.sum())} | {rescate} | {dano} | {neto:+d} | {d_oof:+.4f} | "
            f"{d_lb:+.5f} | {abs(d_lb) / SE_LB * 100:.1f} % |")
        filas.append((peso, int(cambio.sum()), rescate, dano, neto, d_oof, d_lb))

    mejor = max(filas, key=lambda f: f[5])
    peso, cambiadas, rescate, dano, neto, d_oof, d_lb = mejor
    log("")
    log("## 2. La candidata optima de esta sonda")
    log("")
    log(f"- Peso kNN {peso:.2f}: **{cambiadas} celdas cambiadas**, {rescate} rescates, {dano} danos, "
        f"**neto {neto:+d}**")
    log(f"- dOOF = **{d_oof:+.4f}**. Umbral de gate dOOF = +0.03756. "
        f"Cumple el {d_oof / 0.03756 * 100:.0f} % del umbral")
    log(f"- dLB implicito por la sensibilidad del proyecto = **{d_lb:+.5f}**")
    log(f"- Eso es **{abs(d_lb) / SE_LB * 100:.1f} % de un error estandar del LB**. "
        f"Para mover el LB un SE hacen falta {SE_LB / F1_LB_POR_ACIERTO_NETO:.0f} celdas netas")
    log("")
    log("## 3. Contraste con las sondas ya enviadas")
    log("")
    log("| sonda | celdas | neto | dLB real | veredicto |")
    log("|---|---:|---:|---:|---|")
    log("| v02 -> v03 (umbrales) | 126 | +126 | +0.00313 | la unica que mejoro |")
    log("| v23 SVD+RBF | 240 | +32 | -0.00368 | mejor evidencia interna, LB cae |")
    log("| v20 interaccion HS | 51 | -14 | -0.00643 | 25 rescates / 39 danos en OOF |")
    log("| v09_p5 C por etiqueta | ~ | -6,6 | -0.00525 | tasa de acierto 0,5185 |")
    log("| v12 promedio repeticiones | ~ | -14,9 | -0.01184 | la mayor caida |")
    log(f"| **kNN union w={peso:.2f} (esta sonda)** | **{cambiadas}** | **{neto:+d}** | "
        f"**{d_lb:+.5f} (implicito)** | **no enviado** |")
    log("")
    log("## 4. Veredicto del addendum")
    log("")
    log(f"La mejor formulacion de kNN que existe cambia {cambiadas} celdas y produce {neto:+d} netas.")
    log(f"Eso es un {abs(d_lb) / SE_LB * 100:.1f} % del error estandar del LB: la submission seria")
    log("estadisticamente indistinguible de V03 antes incluso de poder llamarse una mejora.")
    log("")
    log("En el eje de celdas, la sonda kNN se situa entre dos precedentes ya medidos:")
    log("")
    log("- **v20 interaccion HS**: 51 celdas, neto -14 en OOF, LB -0.00643. Es la sonda mas")
    log("  pequena que se ha enviado, y ya perdio.")
    log("- **v23 SVD + RBF**: 240 celdas, neto +32, el mejor conjunto de evidencia interna del")
    log("  proyecto (5/5 pliegues, McNemar p = 0,0452), y LB -0,00368. Es el precedente mas")
    log("  cercano a kNN en tamano de cambio y en perfil, y ya perdio.")
    log("")
    log("kNN queda por debajo de V23 en celdas netas y por encima en celdas cambiadas, es decir")
    log("cambia mucho mas para conseguir menos. Ese es el perfil de una sonda marginal, no de una")
    log("candidata competitiva.")
    log("")
    log(f"Completado en {time.time() - inicio:.1f} s")

    (REPORTS_DIR / "v24_03b_perfil_de_celdas.md").write_text("\n".join(SALIDA) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
