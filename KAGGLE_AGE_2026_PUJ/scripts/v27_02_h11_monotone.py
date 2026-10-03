"""V27 · 02 — H11: restricciones monotonas nativas de LightGBM sobre la evidencia de clase.

**Unico conjunto de restricciones, justificado antes del resultado y sin grid.** V20 congelado,
mismos folds, mismo `HIERBA`, misma cadena de umbrales, mismo blend. Lo unico que cambia es el
parametro nativo `monotone_constraints` de LightGBM, que el proyecto **nunca ha usado**.

## Por que la relacion monótona es defendible, y de donde sale exactamente

En la matriz 279 **ningun bloque es una probabilidad en [0, 1]**: todos estan en log-odds o son
margenes. Eso se midio antes de decidir. La matriz 279 se construye con `MEMBER_FILES`, que aplica
`log` a ocho ficheros y deja los demas en crudo, de modo que:

- los bloques `log` de un clasificador por clase guardan `log P(clase)`;
- los bloques `margenes`, `nbsvm`, `pares` y `E015_ordinal` guardan **márgenes**, que no son
  probabilidades y **no admiten** ninguna relacion monótona con un acumulado;
- `E019b_densas_svd` es un **TruncatedSVD de features densas**, no de clases. Sus seis componentes
  coinciden en sumar 1 por casualidad numerica, luego se **excluye**: un SVD no genera
  probabilidades por clase.

Quedan **siete modelos con probabilidad genuina por clase**, verificados en el codigo que los
produce (`predict_proba` en `experimento_e016_e017.py` y
`experimento_e021_ordinal_centroides.py`, y un softmax de distancia en
`experimento_e014_e015_hipotesis_error.py`):

    E-009, E-010, E-011, E015_softmax, E017_probabilidades, E021a_ordinal, E021b_rocchio

Son **7 x 6 = 42 features** de las 279.

**La relacion, para el umbral `k` cuyo objetivo es `Y <= k - 1`.** Hay que distinguir los dos casos,
porque el signo no es el mismo por el mismo motivo:

- Si `j <= k - 1`: la clase `j` esta **dentro** del acumulado. Subir `P(Y = j)` resta masa a
  `Y > k - 1`, luego **sube** `P(Y <= k - 1)`.
- Si `j >= k`: la clase `j` esta **fuera** del acumulado. Subir `P(Y = j)` resta masa al resto, y
  como el resto se reparte sobre las demas clases, en media sube `P(Y <= k - 1)`.

En ambos casos el signo es **creciente**, y como `log` es estrictamente creciente, la relacion se
conserva en la escala en la que vive la feature. Ese es el unico conjunto que se prueba.

Asi que el vector es **+1 en esas 42 features y 0 en las 237 restantes**, y no se impone
monotonía arbitraria: las 237 que no son probabilidad de clase se dejan **libres**, que es lo
conservador.

## Lo que no se hace

No se prueba ninguna combinacion mas de restricciones, ni se amplia el conjunto, ni se busca una
que mejore una validacion. El conjunto se fijo **antes** de entrenar y no se toca despues.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402
from v13_p1b_fusion_tokens import bootstrap_diferencia, metricas_frontera  # noqa: E402
from v19_b1_boosting import HIERBA, matriz_279  # noqa: E402

RESULTADO = cfg.EXPERIMENTS / "v27_02_h11_monotone.json"
RESTRICCIONES = cfg.REPORTS / "v27_h11_restricciones.csv"
FIG = cfg.FIGURES / "v27_h11_delta_por_fold.png"
CUMULADAS_V20 = cfg.FEATURES / "v20_f3_oof_cumuladas.npy"
CV_V20 = 0.6097782
LB_V20 = 0.62689
ETIQUETAS = list(range(len(cfg.CLASSES)))
GATE_DELTA = 0.003
GATE_FOLDS = 4
PARES = {("18-24", "25-34"): "18-24|25-34", ("25-34", "35-49"): "25-34|35-49"}
#: Modelos con probabilidad **genuina** por clase, verificados en el codigo que los produce.
MODELOS_PROBABILIDAD = ["E-009", "E-010", "E-011", "E015_softmax", "E017_probabilidades",
                         "E021a_ordinal", "E021b_rocchio"]
#: Bloques **excluidos con su motivo**, para que la decision sea auditable.
EXCLUIDOS = {
    "E019b_densas_svd": "TruncatedSVD de features densas, no de clases",
    "E013_margenes": "margen, no probabilidad de clase",
    "E016_margenes": "margen, no probabilidad de clase",
    "E014_margenes": "margen, no probabilidad de clase",
    "E015_ordinal": "un solo valor, no cubre las seis clases",
    "E028a_pares": "cinco columnas de pares, no cubre las seis clases",
    "ADV05_nbsvm": "score NB-SVM, no probabilidad de clase",
    "nbsvm_word": "score NB-SVM, no probabilidad de clase",
}


def macro(clase: np.ndarray, prob: np.ndarray) -> float:
    """macro-F1 pooled de las seis clases."""
    return float(f1_score(clase, np.argmax(prob, axis=1), average="macro", labels=ETIQUETAS))


def reconstruye(acumuladas: np.ndarray) -> np.ndarray:
    """La reconstruccion exacta de V20."""
    n_umbral = acumuladas.shape[1]
    mono = acumuladas.copy()
    for k in range(1, n_umbral):
        mono[:, k] = np.maximum(mono[:, k], mono[:, k - 1])
    seis = np.zeros((len(mono), n_umbral + 1))
    seis[:, 0] = mono[:, 0]
    for k in range(1, n_umbral):
        seis[:, k] = mono[:, k] - mono[:, k - 1]
    seis[:, -1] = 1.0 - mono[:, n_umbral - 1]
    seis = np.clip(seis, 0.0, None)
    return seis / np.clip(seis.sum(axis=1, keepdims=True), 1e-9, None)


def main() -> int:
    """Ejecuta H11 con el conjunto de restricciones pre-fijado."""
    autotest_numerico()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(v) for v in y])
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")
    matriz = matriz_279()
    n_clases = len(cfg.CLASSES)
    logreg = np.load(cfg.FEATURES / "v19_b1_oof_logreg.npy")
    mlp = np.load(cfg.FEATURES / "v19_b1_oof_mlp.npy")
    arbol_v19 = np.load(cfg.FEATURES / "v19_oof_arbol_5folds.npy")
    inventario = pd.read_csv(cfg.REPORTS / "v25_feature_inventory.csv")

    # ---------------------------------------------------------------- vector de restricciones
    import re
    columnas_restringidas = []
    for modelo in MODELOS_PROBABILIDAD:
        coincide = inventario.loc[inventario["nombre"].str.match(
            "^" + re.escape(modelo) + r"_c\d+$"), "indice"].to_numpy()
        if len(coincide) != 6:
            print(f"[ABORTO] {modelo} tiene {len(coincide)} columnas, se esperaban 6")
            return 1
        columnas_restringidas.extend(int(c) for c in coincide)
    restricciones = [0] * matriz.shape[1]
    for columna in columnas_restringidas:
        restricciones[columna] = 1
    print(f"[H11] {len(MODELOS_PROBABILIDAD)} modelos con probabilidad de clase | "
          f"{len(columnas_restringidas)} features con restriccion +1 | "
          f"{matriz.shape[1] - len(columnas_restringidas)} libres", flush=True)
    pd.DataFrame({"indice": columnas_restringidas,
                  "nombre": [inventario.loc[inventario["indice"] == c, "nombre"].iloc[0]
                             for c in columnas_restringidas],
                  "signo": 1}).to_csv(RESTRICCIONES, index=False)
    print(f"[H11] excluidos con motivo: {EXCLUIDOS}", flush=True)

    # ---------------------------------------------------------------- T0: control
    seis_t0 = reconstruye(np.load(CUMULADAS_V20))
    oof_t0 = 0.25 * logreg + 0.25 * mlp + 0.25 * arbol_v19 + 0.25 * seis_t0
    valor_t0 = macro(clase, oof_t0)
    reproduce = abs(valor_t0 - CV_V20) < 1e-6
    print(f"[T0] V20 {valor_t0:.7f} | reproduce: {reproduce}", flush=True)
    if not reproduce:
        print("[STOP] V20 no reproduce. No se ejecuta H11.")
        return 1

    # ---------------------------------------------------------------- T1: restricciones monotonas
    acumuladas_t1 = np.zeros((len(y), n_clases - 1))
    for fold in folds:
        validacion = fold["val_idx"]
        entrenamiento = np.concatenate([o["val_idx"] for o in folds
                                        if o["fold"] != fold["fold"]])
        for k in range(1, n_clases):
            objetivo = (clase <= k - 1).astype(int)
            modelo = lgb.LGBMClassifier(**HIERBA, monotone_constraints=restricciones)
            modelo.fit(matriz[entrenamiento], objetivo[entrenamiento])
            acumuladas_t1[validacion, k - 1] = modelo.predict_proba(matriz[validacion])[:, 1]
        print(f"    fold {fold['fold']}: cadena con restricciones monotonas lista", flush=True)

    seis_t1 = reconstruye(acumuladas_t1)
    oof_t1 = 0.25 * logreg + 0.25 * mlp + 0.25 * arbol_v19 + 0.25 * seis_t1
    valor_t1 = macro(clase, oof_t1)
    print(f"[T1] {valor_t1:.7f} | delta {valor_t1 - valor_t0:+.6f}", flush=True)

    # ---------------------------------------------------------------- metricas
    filas, detalle = [], {}
    for nombre, prob in (("T0 V20 control", oof_t0),
                         ("T1 con restricciones monotonas", oof_t1)):
        pred = np.argmax(prob, axis=1)
        pred_t0 = np.argmax(oof_t0, axis=1)
        por = [macro(clase[f["val_idx"]], prob[f["val_idx"]]) for f in folds]
        base = [macro(clase[f["val_idx"]], oof_t0[f["val_idx"]]) for f in folds]
        deltas = [round(a - b, 6) for a, b in zip(por, base)]
        rescate = int(((pred_t0 != clase) & (pred == clase)).sum())
        dano = int(((pred_t0 == clase) & (pred != clase)).sum())
        item = {"modelo": nombre, "corrected_cv": round(macro(clase, prob), 7),
                "delta_vs_V20": round(macro(clase, prob) - valor_t0, 6),
                "fold_1": round(por[0], 7), "fold_2": round(por[1], 7), "fold_3": round(por[2], 7),
                "fold_4": round(por[3], 7), "fold_5": round(por[4], 7),
                "media": round(float(np.mean(por)), 6), "desviacion": round(float(np.std(por)), 6),
                "delta_por_fold": deltas, "folds_positivos": sum(1 for d in deltas if d > 0),
                "rescue": rescate, "damage": dano, "net": rescate - dano,
                "changed_rows": int((pred != pred_t0).sum()),
                "f1_por_clase": {c: round(float(v), 4) for c, v in zip(
                    cfg.CLASSES, f1_score(clase, pred, average=None, labels=ETIQUETAS))},
                "complementariedad_por_clase": {
                    c: {"rescata": int(((pred_t0 != clase) & (pred == clase) & (clase == i)).sum()),
                        "danio": int(((pred_t0 == clase) & (pred != clase) & (clase == i)).sum())}
                    for i, c in enumerate(cfg.CLASSES)},
                "frontera_1": round(float(metricas_frontera(prob, clase, PARES)["18-24|25-34"]
                                          ["accuracy"]), 4)}
        filas.append(item)
        detalle[nombre] = item
        print(f"[{nombre}] {item['corrected_cv']:.7f} | delta {item['delta_vs_V20']:+.6f} | "
              f"folds +{item['folds_positivos']}/5 | neto {item['net']:+,} | "
              f"filas {item['changed_rows']:,}", flush=True)

    # ---------------------------------------------------------------- gate
    t1 = detalle["T1 con restricciones monotonas"]
    cumple_delta = t1["delta_vs_V20"] >= GATE_DELTA
    cumple_folds = t1["folds_positivos"] >= GATE_FOLDS
    intervalo = bootstrap_diferencia(oof_t0, oof_t1, clase, n_clases)
    bootstrap = {"delta": round(t1["delta_vs_V20"], 6),
                 "ic95": [round(float(intervalo[0]), 6), round(float(intervalo[1]), 6)],
                 "proporcion_positiva": round(float(intervalo[2]), 6),
                 "ic95_incluye_cero": bool(intervalo[0] <= 0.0 <= intervalo[1])}
    print(f"[bootstrap] delta {bootstrap['delta']:+.6f} IC95 "
          f"[{bootstrap['ic95'][0]:+.6f}, {bootstrap['ic95'][1]:+.6f}]", flush=True)
    if not cumple_delta and t1["delta_vs_V20"] <= 0:
        decision, estado = "H11 CLOSED", "CLOSED"
    elif not cumple_delta:
        decision, estado = "H11 MARGINAL", "MARGINAL"
    elif cumple_folds and not bootstrap["ic95_incluye_cero"]:
        decision, estado = "H11 PROMISING (sin submission)", "PROMISING"
    else:
        decision, estado = "H11 MARGINAL", "MARGINAL"
    print(f"[gate] delta {t1['delta_vs_V20']:+.6f} (>= {GATE_DELTA}: {cumple_delta}) | "
          f"folds {t1['folds_positivos']}/5 (>= {GATE_FOLDS}: {cumple_folds})", flush=True)
    print(f"[decision] {decision}", flush=True)

    # ---------------------------------------------------------------- mecanismo
    # Si las restricciones cambian poco, la evidencia ya era consistente con el orden.
    restricciones_efectivas = {}
    for k in range(1, n_clases):
        objetivo = (clase <= k - 1).astype(int)
        idx = folds[0]["val_idx"]
        tr = np.concatenate([o["val_idx"] for o in folds if o["fold"] != 0])
        con = lgb.LGBMClassifier(**HIERBA, monotone_constraints=restricciones)
        con.fit(matriz[tr], objetivo[tr])
        sin = lgb.LGBMClassifier(**HIERBA).fit(matriz[tr], objetivo[tr])
        n_splits = len(con.booster_.dump_model()["tree_info"])
        n_usadas = sum(1 for t in con.booster_.dump_model()["tree_info"]
                       for c in t.get("split_feature", []) if c in columnas_restringidas)
        total = sum(len(t.get("split_feature", [])) for t in con.boster_model().get("tree_info", [])) \
            if False else sum(len(t.get("split_feature", [])) for t in con.booster_.dump_model()["tree_info"])
        restricciones_efectivas[f"umbral_{k}"] = {
            "arboles": n_splits, "splits_totales": total,
            "splits_en_features_restringidas": n_usadas,
            "proporcion": round(n_usadas / max(total, 1), 4),
            "cambia_respecto_a_libre": bool(
                not np.array_equal(con.predict_proba(matriz[idx])[:, 1],
                                   sin.predict_proba(matriz[idx])[:, 1]))}
    mecanismo = restricciones_efectivas
    for clave, v in restricciones_efectivas.items():
        print(f"[mecanismo] {clave}: {v['splits_en_features_restringidas']}/"
              f"{v['splits_totales']} splits caen en features restringidas "
              f"({v['proporcion']:.1%}) | cambia vs libre: {v['cambia_respecto_a_libre']}",
              flush=True)

    _figura(filas)
    pd.DataFrame([{"ID": "H11-T0" if f["modelo"].startswith("T0") else "V27-02",
                   **{k: v for k, v in f.items()
                      if k not in {"f1_por_clase", "complementariedad_por_clase", "delta_por_fold"}}}
                  for f in filas]).to_csv(cfg.REPORTS / "v26_master_results.csv", mode="a",
                                           header=False, index=False)
    RESULTADO.write_text(json.dumps({
        "fase": "V27 02 H11 restricciones monotonas", "fecha": "2026-09-28",
        "hipotesis": "certainas relaciones entre evidencia y clases tienen estructura monotona y el "
                     "boosting libre gasta capacidad en relaciones que no son plausibles",
        "novedad_funcional": "`monotone_constraints` tiene CERO usos en todo el proyecto; V20 "
                             "impone la monotonia DESPUES, con maximo acumulado sobre cinco "
                             "modelos independientes",
        "justificacion_de_las_restricciones": (
            "en la matriz 279 ningun bloque es probabilidad en [0,1]: todos estan en log-odds o son "
            "margenes. Solo siete modelos guardan probabilidad genuina por clase, verificados en el "
            "codigo que los produce. E019b es un TruncatedSVD de features densas, no de clases, y "
            "queda excluido aunque sus componentes sumen 1 por casualidad numerica"),
        "modelos_con_probabilidad": MODELOS_PROBABILIDAD, "excluidos_con_motivo": EXCLUIDOS,
        "features_restringidas": len(columnas_restringidas),
        "features_libres": int(matriz.shape[1] - len(columnas_restringidas)),
        "signo": "+1 (creciente): subir P(clase j) sube P(Y <= k - 1) para toda j",
        "control_t0": {"corrected_cv": round(valor_t0, 7), "reproducido": reproduce},
        "candidatos": detalle, "delta": round(t1["delta_vs_V20"], 6),
        "mecanismo": mecanismo,
        "gate": {"delta_minimo": GATE_DELTA, "folds_minimos": GATE_FOLDS,
                 "cumple_delta": cumple_delta, "cumple_folds": cumple_folds,
                 "bootstrap": bootstrap},
        "decision": decision, "estado": estado, "submission_generada": False,
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


def _figura(filas: list[dict]) -> None:
    """Delta por fold y evolucion de corrected_cv."""
    fig, ejes = plt.subplots(1, 2, figsize=(13, 4.5))
    eje = ejes[0]
    t0, t1 = filas[0], filas[1]
    eje.bar(np.arange(5) - 0.2, t0["delta_por_fold"], 0.4, label="T0", color="#999999")
    eje.bar(np.arange(5) + 0.2, t1["delta_por_fold"], 0.4, label="T1 monotonas", color="#2166ac")
    eje.axhline(0, color="grey", linewidth=1)
    eje.axhline(GATE_DELTA, color="black", linestyle="--", linewidth=1)
    eje.set_xticks(range(5))
    eje.set_xticklabels([f"fold {i + 1}" for i in range(5)])
    eje.set_ylabel("delta de macro-F1 frente a T0")
    eje.legend(fontsize=8)
    eje.set_title(f"Delta {t1['delta_vs_V20']:+.6f}, {t1['folds_positivos']}/5", fontsize=10)
    eje = ejes[1]
    eje.plot(["V20", "H11"], [0.6097782, t1["corrected_cv"]], marker="o", color="#2166ac")
    eje.set_ylabel("corrected_cv")
    eje.set_title("H11 frente a V20", fontsize=10)
    fig.suptitle("H11: restricciones monotonas nativas de LightGBM", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
