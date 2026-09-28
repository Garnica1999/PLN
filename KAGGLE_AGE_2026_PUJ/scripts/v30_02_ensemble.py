"""V30 - 02 - P3 y P4: comparar candidatos, medir su diversidad y construir el ensemble final.

V29 ya enseno cual es el reparto util: **V20 mas el LightGBM optimizado**, con el modelo nuevo
llevando la mayor parte del peso. V30 no repite la busqueda sobre el pool historico de 16 miembros,
que ya dio su respuesta: **0 de 47** tenian rescate neto positivo y los tres metodos eran
negativos sin el modelo nuevo. Aqui el pool es **pequeno y justificado**, como pide el punto 17.

## Los tres candidatos y que se decide con cada uno

| Candidato | Para que sirve |
|---|---|
| V20 | referencia congelada |
| V29 LightGBM | el punto de partida que V30 intenta consolidar |
| V30 LightGBM | el refinamiento local |

Antes de combinar se mide si V30 **aprende mejor y corrige errores distintos**, o si solo reproduce
a V29. Si sus errores coinciden con los de V29, no hay nada que combinar y el ensemble se queda
con V20 mas uno solo.

## La combinacion

Un unico metodo: **Optuna sobre el holdout interior**, con `w = softmax(theta)` para imponer
`w_i >= 0` y `sum w_i = 1` sin restricciones. Se evaluan el objetivo de macro-F1 puro y el de
macro-F1 menos una penalizacion de entropia, porque en V29 el segundo gano al primero y eso mismo
tiene que comprobarse aqui.

El numero de miembros se limita a **cuatro** por construccion, no por penalizacion posterior.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402
from v13_p1b_fusion_tokens import bootstrap_diferencia  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)

RESULTADO = cfg.EXPERIMENTS / "v30_02_ensemble.json"
COMPARATIVA = cfg.REPORTS / "v30_candidate_comparison.csv"
TABLA = cfg.REPORTS / "v30_ensemble_results.csv"
FIG = cfg.FIGURES / "v30_ensemble.png"
CV_V20 = 0.6097782
CV_LGBM_V29 = 0.6121698
GATE_DELTA = 0.003
GATE_FOLDS = 4
#: Bandera que el punto 20 del brief pide registrar por separado del gate historico.
MARGINAL_MIN = 0.002
TRIALS_PESOS = 300
LAMBDA = 0.002
SEMILLA_OPTUNA = 20300930
N_CLASES = 6
MAXIMO_MIEMBROS = 4


def _macro(clase: np.ndarray, prob: np.ndarray) -> float:
    return float(f1_score(clase, np.argmax(prob, axis=1), average="macro",
                          labels=list(range(N_CLASES))))


def _softmax(theta: np.ndarray) -> np.ndarray:
    e = np.exp(theta - theta.max())
    return e / e.sum()


def _combina(matrices: np.ndarray, pesos: np.ndarray) -> np.ndarray:
    return np.tensordot(pesos, matrices, axes=(0, 0))


def _objetivo_2(macro: float, pesos: np.ndarray, lam: float) -> float:
    """`macro_F1 - lambda * H(w)/log(k)`. Penaliza el **reparto** del peso, no la diversidad."""
    if len(pesos) <= 1:
        return macro
    entropia = float(-(pesos * np.log(np.clip(pesos, 1e-12, None))).sum())
    return macro - lam * entropia / np.log(len(pesos))


def _interior(clase: np.ndarray, entrenamiento: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    partes = cvmod.make_folds(clase[entrenamiento], groups=np.arange(len(entrenamiento)),
                              n_splits=5, n_repeats=1, scheme="stratified")
    elegido = min(partes, key=lambda f: abs(len(f["val_idx"]) - int(0.2 * len(entrenamiento))))
    val = entrenamiento[np.asarray(elegido["val_idx"])]
    tr = entrenamiento[np.concatenate([np.asarray(o["val_idx"]) for o in partes
                                       if o["fold"] != elegido["fold"]])]
    return tr, val


def _diversidad(nombres: list[str], argmax: dict, clase: np.ndarray) -> list[dict]:
    """Las tres medidas de V30, entre todos los pares y de cada uno frente a V20."""
    bien = {n: argmax[n] == clase for n in nombres}
    filas = []
    for n in nombres:
        otras = bien[n] if n == "v20_f3b_oof_candidato" else bien["v20_f3b_oof_candidato"]
        rescate = int((bien[n] & ~otras).sum())
        dano = int((~bien[n] & otras).sum())
        filas.append({
            "modelo_a": n, "modelo_b": "v20_f3b_oof_candidato", "alcance": "contra V20",
            "individual_cv": round(_macro(clase, np.eye(N_CLASES)[argmax[n]]), 7),
            "corr_errores": round(float(np.corrcoef((~bien[n]).astype(float),
                                                     (~otras).astype(float))[0, 1]), 6),
            "tasa_discrepancia": round(float((argmax[n] !=
                                               argmax["v20_f3b_oof_candidato"]).mean()), 6),
            "rescate_unico": rescate, "dano_unico": dano, "neto": rescate - dano,
        })
    for i, a in enumerate(nombres):
        for b in nombres[i + 1:]:
            filas.append({
                "modelo_a": a, "modelo_b": b, "alcance": "par a par",
                "individual_cv": "",
                "corr_errores": round(float(np.corrcoef((~bien[a]).astype(float),
                                                         (~bien[b]).astype(float))[0, 1]), 6),
                "tasa_discrepancia": round(float((argmax[a] != argmax[b]).mean()), 6),
                "rescate_unico": int((bien[a] & ~bien[b]).sum()),
                "dano_unico": int((~bien[a] & bien[b]).sum()),
                "neto": int((bien[a] & ~bien[b]).sum() - (~bien[a] & bien[b]).sum()),
            })
    return filas


def main() -> int:
    """Compara candidatos, mide diversidad y optimiza el peso de un ensemble de tres o cuatro."""
    autotest_numerico()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(v) for v in y])
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")

    # ---------------------------------------------------------------- cargar candidatos
    # V30 puede no existir todavia si el refinamiento no ha terminado: en ese caso el ensemble se
    # construye con lo que haya y **se declara el hueco**, en vez de inventar un numero.
    oof_v20 = np.load(cfg.FEATURES / "v20_f3b_oof_candidato.npy")
    oof_v29 = np.load(cfg.FEATURES / "v29_lgbm_optuna_oof.npy")
    ref = {"v20_f3b_oof_candidato": oof_v20, "v29_lgbm_optuna": oof_v29}
    hay_v30 = (cfg.FEATURES / "v30_lgbm_refinado_oof.npy").exists()
    if hay_v30:
        ref["v30_lgbm_refinado"] = np.load(cfg.FEATURES / "v30_lgbm_refinado_oof.npy")
    nombres = list(ref)
    print(f"[V30] candidatos en el pool: {nombres}", flush=True)
    argmax = {n: np.argmax(m, axis=1) for n, m in ref.items()}

    # ---------------------------------------------------------------- comparativa y diversidad
    filas_div = _diversidad(nombres, argmax, clase)
    pd.DataFrame(filas_div).to_csv(COMPARATIVA, index=False)
    print("\n=== los tres candidatos frente a V20 ===", flush=True)
    for f in filas_div:
        if f["alcance"] != "contra V20":
            continue
        print(f"  {f['modelo_a']:<24} CV {f['individual_cv']:.7f} | corr_err "
              f"{f['corr_errores']:.4f} | rescate {f['rescate_unico']:>6,} | dano "
              f"{f['dano_unico']:>6,} | neto {f['neto']:>+7,}", flush=True)
    print("\n=== los pares entre candidatos, y si V30 aporta algo sobre V29 ===", flush=True)
    for f in filas_div:
        if f["alcance"] != "par a par":
            continue
        marca = ""
        if {f["modelo_a"], f["modelo_b"]} == {"v29_lgbm_optuna", "v30_lgbm_refinado"}:
            marca = "   <-- los dos LightGBM"
        print(f"  {f['modelo_a']:<24} vs {f['modelo_b']:<24} corr_err {f['corr_errores']:.4f} | "
              f"discrep {f['tasa_discrepancia']:.4f} | uno arregla al otro "
              f"{f['rescate_unico']:>6,}{marca}", flush=True)

    # ---------------------------------------------------------------- ensemble
    matrices = np.stack([ref[n].astype(np.float64) for n in nombres])
    metodos = {}
    for etiqueta, objetivo, lam in (("E1 Optuna macro-F1", 1, 0.0),
                                    ("E2 Optuna macro-F1 menos lambda por entropia", 2, LAMBDA)):
        oof = np.zeros((len(clase), N_CLASES))
        detalle = {}
        for fold in folds:
            val = np.asarray(fold["val_idx"])
            tr = np.concatenate([np.asarray(o["val_idx"]) for o in folds
                                 if o["fold"] != fold["fold"]])
            _, val_in = _interior(clase, tr)
            assert not np.intersect1d(val_in, val).size, "el interior toca la exterior"
            mejor = {"theta": None, "valor": -np.inf, "macro": -np.inf}

            def f(trial: optuna.Trial) -> float:
                theta = np.array([trial.suggest_float(f"w{i}", -4.0, 4.0)
                                  for i in range(len(nombres))])
                pesos = _softmax(theta)
                macro = _macro(clase[val_in], _combina(matrices[:, val_in, :], pesos))
                valor = macro if objetivo == 1 else _objetivo_2(macro, pesos, lam)
                if valor > mejor["valor"]:
                    mejor.update({"theta": theta, "valor": valor, "macro": macro})
                return valor

            estudio = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=SEMILLA_OPTUNA + int(fold["fold"])))
            estudio.optimize(f, n_trials=TRIALS_PESOS, show_progress_bar=False)
            pesos = _softmax(mejor["theta"])
            oof[val] = _combina(matrices[:, val, :], pesos)
            detalle[int(fold["fold"])] = {
                "n_inner_val": int(len(val_in)), "n_trials": TRIALS_PESOS,
                "mejor_trial": int(estudio.best_trial.number),
                "mejor_macro_f1_inner": round(float(mejor["macro"]), 6),
                "mediana_trial": round(float(np.median([t.value for t in estudio.trials
                                                        if t.value is not None])), 6),
                "pesos": {n: round(float(w), 4) for n, w in zip(nombres, pesos)},
                "macro_f1_outer_val": round(_macro(clase[val], oof[val]), 6)}
        medios = np.mean([[detalle[k]["pesos"].get(n, 0.0) for n in nombres]
                          for k in detalle], axis=0)
        medios = medios / medios.sum()
        metodos[etiqueta] = {"oof": oof, "pesos": medios, "detalle": detalle}

    # ---------------------------------------------------------------- metricas y gate
    filas = []
    for etiqueta in ("E1 Optuna macro-F1", "E2 Optuna macro-F1 menos lambda por entropia"):
        oof = metodos[etiqueta]["oof"]
        pred = np.argmax(oof, axis=1)
        pred20 = argmax["v20_f3b_oof_candidato"]
        valor = _macro(clase, oof)
        por_fold = [_macro(clase[f["val_idx"]], oof[f["val_idx"]]) for f in folds]
        base_fold = [_macro(clase[f["val_idx"]], oof_v20[f["val_idx"]]) for f in folds]
        deltas = [round(a - b, 6) for a, b in zip(por_fold, base_fold)]
        rescate = int(((pred20 != clase) & (pred == clase)).sum())
        dano = int(((pred20 == clase) & (pred != clase)).sum())
        cumple = valor - CV_V20 >= GATE_DELTA and sum(1 for d in deltas if d > 0) >= GATE_FOLDS
        marginal = (sum(1 for d in deltas if d > 0) == 5
                    and MARGINAL_MIN <= valor - CV_V20 < GATE_DELTA)
        filas.append({
            "metodo": etiqueta, "corrected_cv": round(valor, 7),
            "delta_vs_V20": round(valor - CV_V20, 6),
            "delta_vs_lgbm_v29": round(valor - CV_LGBM_V29, 6),
            "folds": [round(v, 7) for v in por_fold], "delta_por_fold": deltas,
            "folds_positivos": sum(1 for d in deltas if d > 0),
            "rescue": rescate, "damage": dano, "net": rescate - dano,
            "filas_cambiadas_vs_v20": int((pred != pred20).sum()),
            "cumple_gate_historico": bool(cumple),
            "es_strong_marginal": bool(marginal),
            "pesos": {n: round(float(w), 4) for n, w in zip(nombres,
                                                             metodos[etiqueta]["pesos"])},
        })
        print(f"\n[{etiqueta}] CV {valor:.7f} | delta V20 {valor - CV_V20:+.6f} | delta V29-LGBM "
              f"{valor - CV_LGBM_V29:+.6f} | folds +{filas[-1]['folds_positivos']}/5 | neto "
              f"{rescate - dano:+,}", flush=True)
        print(f"  pesos: {filas[-1]['pesos']}", flush=True)
        print(f"  gate historico: {cumple} | STRONG_MARGINAL: {marginal}", flush=True)

    principal = max(filas, key=lambda f: f["corrected_cv"])
    print(f"\n[V30] candidato principal: {principal['metodo']} con "
          f"{principal['corrected_cv']}", flush=True)

    # ---------------------------------------------------------------- bootstrap
    # Autorizado en V30 para el candidato principal aunque se quede por debajo del gate, pero
    # **separado** del gate: son dos preguntas distintas.
    intervalo = bootstrap_diferencia(oof_v20, metodos[principal["metodo"]]["oof"], clase, N_CLASES)
    bootstrap = {
        "delta": principal["delta_vs_V20"],
        "ic95": [round(float(intervalo[0]), 6), round(float(intervalo[1]), 6)],
        "proporcion_positiva": round(float(intervalo[2]), 6),
        "ic95_incluye_cero": bool(intervalo[0] <= 0.0 <= intervalo[1]),
        "nota": "calculado **solo** para el candidato principal, y no se usa para elegir "
                "hiperparametros ni pesos: es evidencia de decision de la submission, no senal de "
                "entrenamiento",
    }
    print(f"[bootstrap] delta {bootstrap['delta']:+.6f} IC95 {bootstrap['ic95']} | incluye cero: "
          f"{bootstrap['ic95_incluye_cero']}", flush=True)

    pd.DataFrame([{k: str(v) if isinstance(v, (dict, list)) else v for k, v in f.items()}
                  for f in filas]).to_csv(TABLA, index=False)
    _figura(filas, argmax, clase, nombres, ref)

    # ---------------------------------------------------------------- manifest del candidato
    artefacto_30 = (cfg.EXPERIMENTS / "v30_01_optuna_refinamiento.json")
    refin = (json.loads(artefacto_30.read_text(encoding="utf-8"))
             if artefacto_30.exists() else {})
    manifest = {
        "miembros": nombres,
        "pesos": principal["pesos"],
        "composicion": {
            "v20_f3b_oof_candidato": "los cuatro cuartos de V20, intactos, con sus pesos de 0,25",
            "v29_lgbm_optuna": "LightGBM de V29, con la configuracion de su mejor trial por fold",
            "v30_lgbm_refinado": "LightGBM de V30, con la configuracion de su mejor trial por fold",
        } if hay_v30 else {"nota": "V30 no disponible: el pool es el de V29"},
        "params_por_fold_v29": refin.get("detalle_por_fold", {}),
        "hiperparametros_congelados": True,
    }
    RESULTADO.write_text(json.dumps({
        "fase": "V30 02 comparacion de candidatos y ensemble", "fecha": "2026-09-30",
        "experimenta": False,
        "hay_candidato_v30": hay_v30,
        "candidatos": {n: {"corrected_cv": round(_macro(clase, ref[n]), 7),
                          "delta_vs_V20": round(_macro(clase, ref[n]) - CV_V20, 6)}
                      for n in nombres},
        "diversidad": filas_div,
        "ensemble": {k: {"corrected_cv": v["corrected_cv"],
                         "delta_vs_V20": v["delta_vs_V20"], "folds_positivos": v["folds_positivos"],
                         "net": v["net"]} for k, v in zip([f["metodo"] for f in filas], filas)},
        "tabla": filas, "candidato_principal": principal["metodo"],
        "bootstrap": bootstrap, "manifest": manifest,
        "maximo_miembros": MAXIMO_MIEMBROS, "n_miembros": len(nombres),
        "cv_v20": CV_V20, "cv_lgbm_v29": CV_LGBM_V29,
        "finalistas": 0,
        "gate_historico_superado": any(f["cumple_gate_historico"] for f in filas),
        "strong_marginal": any(f["es_strong_marginal"] for f in filas),
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[V30] {time.time() - inicio:.0f} s")
    return 0


def _figura(filas: list[dict], argmax: dict, clase: np.ndarray, nombres: list[str],
            ref: dict) -> None:
    """CV por candidato, peso del ensemble, delta por fold y rescate frente a dano."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ejes = plt.subplots(1, 4, figsize=(21, 4.8))

    eje = ejes[0]
    cvs = [round(_macro(clase, ref[n]), 7) for n in nombres]
    eje.bar(range(len(nombres)), cvs, color=["#333333", "#08519c", "#238b45"][:len(nombres)])
    eje.axhline(CV_V20, color="red", linestyle="--", linewidth=1, label=f"V20 = {CV_V20}")
    eje.set_xticks(range(len(nombres)))
    eje.set_xticklabels([n[:16] for n in nombres], rotation=20, ha="right", fontsize=7)
    eje.set_ylabel("corrected_cv individual")
    eje.set_title("Los candidatos por si solos", fontsize=9)
    eje.legend(fontsize=7)

    eje = ejes[1]
    principal = filas[0]
    pesos = principal["pesos"]
    eje.bar(range(len(nombres)), [pesos.get(n, 0.0) for n in nombres],
            color=["#333333", "#08519c", "#238b45"][:len(nombres)])
    eje.set_xticks(range(len(nombres)))
    eje.set_xticklabels([n[:16] for n in nombres], rotation=20, ha="right", fontsize=7)
    eje.set_ylabel("peso medio entre folds")
    eje.set_title(f"Peso del ensemble\n{principal['metodo'][:28]}", fontsize=9)

    eje = ejes[2]
    for k, f in enumerate(filas):
        eje.plot(range(5), f["delta_por_fold"], marker="o", linewidth=1.3,
                 label=f["metodo"][:18])
    eje.axhline(0, color="grey", linewidth=1)
    eje.axhline(GATE_DELTA, color="black", linestyle="--", linewidth=1, label="gate +0,003")
    eje.set_xticks(range(5))
    eje.set_xticklabels([f"fold {i + 1}" for i in range(5)])
    eje.set_ylabel("delta de macro-F1 frente a V20")
    eje.set_title("Consistencia por fold", fontsize=9)
    eje.legend(fontsize=7)

    eje = ejes[3]
    x = np.arange(len(filas))
    eje.bar(x - 0.2, [f["rescue"] for f in filas], 0.4, label="rescate", color="#2171b5")
    eje.bar(x + 0.2, [-f["damage"] for f in filas], 0.4, label="dano", color="#cb181d")
    eje.axhline(0, color="k", linewidth=1)
    eje.set_xticks(x)
    eje.set_xticklabels([f["metodo"][:18] for f in filas], fontsize=7)
    eje.set_ylabel("filas")
    eje.set_title("Rescate frente a dano frente a V20", fontsize=9)
    eje.legend(fontsize=7)
    fig.suptitle("V30: comparacion de candidatos y ensemble de tres miembros", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
