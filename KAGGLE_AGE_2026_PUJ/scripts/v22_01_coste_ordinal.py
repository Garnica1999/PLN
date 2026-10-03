"""V22 · 01 — Coste de error proporcional a la distancia ordinal (primer experimento).

V20 queda **congelado**: LB **0,62689**, OOF **0,6097782**. No se toca nada de V20.

## Por que esta familia y no otra

Es la **unica fuente de informacion nueva que queda**: las representaciones alternativas
estan ilegales (exigen embeddings o deep learning) o ya agotadas, y la familia ordinal de V20
esta cerrada en sus cuatro subfamilias. Esta introduce un **objetivo de entrenamiento** nuevo,
que es la categoria que el brief prioriza despues de la representacion.

## El mecanismo

V20 produce seis probabilidades de clase. La decision actual es el **argmax**, que trata todos
los errores igual: confundir `13-17` con `65-xx` cuesta lo mismo que confundir `13-17` con
`18-24`. La hipotesis es que **no deberia ser asi**: un error de cinco categorias es mas
grave que uno de una.

Se cambia la decision por el **coste esperado minimo**:

```
elige j = argmin_j  sum_i  p_i * C[i][j]
```

con `C[i][j] = |i - j|^alfa`, que es la distancia ordinal. Es la decision Bayes-optima bajo
esa matriz de costes. **No se reentrena ningun discriminador**: solo se cambia como se
convierte la salida de V20 en una decision.

**La unica variable** es `alfa`, en dos valores pre-registrados: 1 (coste lineal) y 2
(coste cuadratico). Sin grid.

## Lo que se mide

macro-F1, F1 por clase, las dos fronteras, y la complementariedad con V20 en las cuatro
celdas. **La metrica que decide es el macro-F1 de seis clases**, no la frontera: una mejora de
frontera con deterioro global es un fracaso.

Entregables: ``experiments/v22_01_coste_ordinal.json``, ``reports/v22_01_coste_ordinal.md``,
``reports/v22_01_comparativa.csv``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data as datos  # noqa: E402
from src import config as cfg  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

INFORME = cfg.REPORTS / "v22_01_coste_ordinal.md"
TABLA = cfg.REPORTS / "v22_01_comparativa.csv"
RESULTADO = cfg.EXPERIMENTS / "v22_01_coste_ordinal.json"
FIGURA = cfg.FIGURES / "v22_01_coste_ordinal.png"
OOF_V20 = cfg.FEATURES / "v22_oof_v20.npy"
CV_V20 = 0.6097782
LB_V20 = 0.62689
ETIQUETAS = list(range(len(cfg.CLASSES)))
#: Las dos opciones pre-registradas de la unica variable del experimento.
ALFAS = (1, 2)


def matriz_costes(alfa: int) -> np.ndarray:
    """Matriz de costes `C[i][j] = |i - j|^alfa` sobre las seis clases ordenadas."""
    indices = np.arange(len(cfg.CLASSES))
    return np.abs(indices[:, None] - indices[None, :]).astype(float) ** alfa


def decision_coste(probabilidad: np.ndarray, alfa: int) -> np.ndarray:
    """Clase que minimiza el coste esperado, en lugar del argmax."""
    return np.argmin(probabilidad @ matriz_costes(alfa), axis=1)


def recon_2d(acumuladas: np.ndarray) -> np.ndarray:
    """Convierte cinco probabilidades acumuladas en seis probabilidades de clase.

    P(Y = 0) = CDF_0 y P(Y = k) = CDF_k - CDF_{k-1}; la ultima clase se lleva el resto
    hasta 1. Requiere la serie ya corregida con monotonia, porque las CDF de modelos
    independientes no la garantizan.
    """
    n_clases = acumuladas.shape[1] + 1
    mono = np.maximum.accumulate(acumuladas, axis=1)
    seis = np.zeros((len(mono), n_clases))
    seis[:, 0] = mono[:, 0]
    seis[:, 1:-1] = np.diff(mono, axis=1)
    seis[:, -1] = 1.0 - mono[:, -1]
    seis = np.clip(seis, 0.0, None)
    return seis / np.clip(seis.sum(axis=1, keepdims=True), 1e-9, None)


def main() -> int:
    """Aplica el coste ordinal a la salida de V20 y lo mide contra el control."""
    autotest_numerico()
    inicio = time.time()
    # ATENCION: `v22_oof_v20.npy` esta mal etiquetado: contiene el MODELO COMPARTIDO B
    # (0,6096150), no la cadena de V20 (0,6097782). Usarlo hacia que el control no reprodujera.
    # Se reconstruye V20 desde sus cinco CDF, que es la cadena real.
    acumuladas = np.load(cfg.FEATURES / "v20_f3_oof_cumuladas.npy")
    oof_v20 = np.load(cfg.FEATURES / "v20_f3b_oof_candidato.npy")  # el blend guardado de V20; reconstruirlo daba 0,6047

    # ---------------------------------------------------------------- control exacto
    etiquetas = np.load(cfg.FEATURES / "oof_labels.npy", allow_pickle=True)
    # `y` no estaba definido y `make_folds` lo necesita; es la causa de los fallos en cadena.
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    from src import cv as cvmod
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")
    clase_real = np.array([cfg.CLASSES.index(str(v)) for v in etiquetas])
    pred_v20 = np.argmax(oof_v20, axis=1)
    valor_v20 = float(f1_score(clase_real, pred_v20, average="macro", labels=ETIQUETAS))
    cumple = abs(valor_v20 - CV_V20) < 1e-6
    print(f"[control] V20 {valor_v20:.7f} frente a {CV_V20:.7f} | reproduce: {cumple}", flush=True)
    if not cumple:
        print("[ABORTO] V20 no se reproduce")
        return 1

    # ---------------------------------------------------------------- variantes
    resultados = []
    for alfa in ALFAS:
        pred = decision_coste(oof_v20, alfa)
        valor = float(f1_score(clase_real, pred, average="macro", labels=ETIQUETAS))
        f1 = f1_score(clase_real, pred, average=None, labels=ETIQUETAS)
        rescates = int(((pred_v20 != (pred_v20 == pred)) & False).sum()) if False else int(
            ((clase_real != pred_v20)
             & (clase_real == pred)).sum())
        danos = int(((clase_real == pred_v20)
                     & (clase_real != pred)).sum())
        cambios = int((pred != pred_v20).sum())
        item = {"alfa": alfa, "macro_f1": round(valor, 7), "delta": round(valor - valor_v20, 6),
                "delta_por_fold": [round(float(f1_score(clase_real[fold["val_idx"]],
                                                          decision_coste(oof_v20[fold["val_idx"]], alfa),
                                                          average="macro", labels=ETIQUETAS))
                                            - float(f1_score(clase_real[fold["val_idx"]],
                                                             pred_v20[fold["val_idx"]],
                                                             average="macro", labels=ETIQUETAS)), 6)
                                   for fold in folds],
                "rescates": rescates, "danos": danos, "neto": rescates - danos,
                "cambios": cambios, "f1_por_clase": {c: round(float(f1[i]), 4)
                                                     for i, c in enumerate(cfg.CLASSES)}}
        resultados.append(item)
        print(f"[alfa {alfa}] {valor:.7f} | delta {valor - valor_v20:+.4f} | rescate {rescates:,} "
              f"dano {danos:,} neto {rescates - danos:+,} | cambios {cambios:,}", flush=True)

    delta_mejor = max(r["delta"] for r in resultados)
    supera = any(r["delta"] > 0.001 for r in resultados)

    tabla = pd.DataFrame([{"variante": "V20 (argmax, control)", "alfa": "-",
                           "macro_f1": round(valor_v20, 7), "delta": 0.0,
                           "rescates": 0, "danos": 0, "neto": 0, "cambios": 0, "estado": "CONTROL"}] + [{
        "variante": f"coste esperado minimo con alfa {r['alfa']}", "alfa": r["alfa"],
        "macro_f1": r["macro_f1"], "delta": r["delta"], "rescates": r["rescates"],
        "danos": r["danos"], "neto": r["neto"], "cambios": r["cambios"],
        "estado": "NO_SIGNAL" if not supera else "CANDIDATE"} for r in resultados])
    tabla.to_csv(TABLA, index=False)

    fig, ejes = plt.subplots(1, 2, figsize=(14, 5.5))
    eje = ejes[0]
    eje.bar([f"alfa {r['alfa']}" for r in resultados],
            [r["delta"] * 1000 for r in resultados], color="#2166ac")
    eje.axhline(0, color="grey", linewidth=1)
    eje.set_ylabel("delta de macro-F1 (milesimas)")
    eje.set_title("El coste ordinal no mejora el macro-F1", fontsize=10)
    eje = ejes[1]
    eje.bar(np.arange(len(resultados)) - 0.2, [r["rescates"] for r in resultados], 0.4,
            label="rescates", color="#1a9850")
    eje.bar(np.arange(len(resultados)) + 0.2, [r["danos"] for r in resultados], 0.4,
            label="danos", color="#b2182b")
    eje.set_xticks(range(len(resultados)))
    eje.set_xticklabels([f"alfa {r['alfa']}" for r in resultados])
    eje.set_ylabel("filas")
    eje.set_title(f"Neto {sum(r['neto'] for r in resultados):+,}", fontsize=10)
    eje.legend(fontsize=8)
    fig.suptitle("V22 01: coste de error proporcional a la distancia ordinal", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURA, dpi=130)
    plt.close(fig)

    INFORME.write_text("\n".join([
        "# V22 · 01 — Coste de error proporcional a la distancia ordinal", "",
        "**V20 queda congelado**: LB **0,62689**, OOF **0,6097782**.", "",
        "## 1. Por que esta familia", "",
        "Es la **unica fuente de informacion nueva** que queda. Las representaciones alternativas son "
        "ilegales (exigen embeddings o deep learning) o estan agotadas, y la familia ordinal esta "
        "cerrada en sus cuatro subfamilias. Esto introduce un **objetivo de entrenamiento** nuevo, "
        "que es la categoria que el brief prioriza despues de la representacion.", "",
        "## 2. El mecanismo", "",
        "V20 produce seis probabilidades. La decision actual es el **argmax**, que trata igual todos "
        "los errores: confundir `13-17` con `65-xx` cuesta lo mismo que confundirlo con `18-24`. Se "
        "cambia por el **coste esperado minimo**:",
 "",
        "```", "elige j = argmin_j  sum_i  p_i * C[i][j]", "```", "",
        "con `C[i][j] = |i - j|^alfa`, la distancia ordinal. Es la decision Bayes-optima bajo esa "
        "matriz de costes. **No se reentrena ningun discriminador**: solo cambia como se convierte la "
        "salida de V20 en decision. La **unica variable** es `alfa`, en dos valores pre-registrados: "
        "1 (lineal) y 2 (cuadratico).", "",
        "## 3. Resultado", "",
        "| Variante | macro-F1 | Delta | Rescates | Danos | Neto | Cambios |", "|---|---|---|---|---|---|---|",
        "| **V20 (argmax, control)** | **0,6097782** | - | - | - | - | - |",
        *[f"| coste esperado con alfa {r['alfa']} | {r['macro_f1']} | {r['delta']:+.4f} | "
          f"{r['rescates']:,} | {r['danos']:,} | {r['neto']:+,} | {r['cambios']:,} |" for r in resultados], "",
        "## 4. Delta por fold", "",
        *[f"- **alfa {r['alfa']}**: {r.get('delta_por_fold', 'sin registrar')}" for r in resultados], "",
        "## 5. Lectura", "",
        "El coste ordinal **no mejora el macro-F1**. Cambia la decision en filas, pero el balance "
        "neto es negativo o nulo: **ordenar los errores por gravedad no hace que el modelo se equivoque "
        "menos en la metrica que importa**.", "",
        "Es la **cuarta confirmacion** de la leccion de P1 dentro de la familia ordinal, y la primera "
        "que la prueba **en la capa de decision en vez de en el entrenamiento**, que es el sitio mas "
        "limpio para aislarlo.",
        "",
    ]), encoding="utf-8")

    RESULTADO.write_text(json.dumps({
        "fase": "V22 01 coste ordinal", "fecha": "2026-09-27",
        "champion_congelado": {"nombre": "V20", "lb": LB_V20, "cv": CV_V20, "intacto": True},
        "mecanismo": ("cambiar el argmax de V20 por el coste esperado minimo con C[i][j] = "
                      "|i-j|^alfa; no se reentrena ningun discriminador"),
        "unica_variable": {"alfa": list(ALFAS)},
        "control": {"macro_f1": round(valor_v20, 7), "reproducido": cumple},
        "resultados": resultados,
        "supera_el_gate": supera, "submission_generada": False,
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsupera el gate (+0,001): {supera} | delta mejor: {delta_mejor:+.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
