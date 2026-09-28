"""V30 - 05: segunda particion agrupada para comprobar que el candidato se sostiene.

## Que es y que no es

**No es una busqueda.** Los hiperparametros y los pesos estan **congelados** antes de entrar aqui.
Lo unico que cambia es la **particion**: se generan folds nuevos con otra semilla del mismo
`StratifiedGroupKFold`, conservando los mismos grupos, y se evalua el candidato sobre ellos.

Eso la convierte en una prueba de robustez real. Si el resultado dependiera de haber encontrado
esos hiperparametros en esos folds concretos, al cambiar la particion se caeria. Si aguanta, la
evidencia es que la configuracion **conduce a algo general** y no a un artefacto de la particion.

## Que preguntas responde

1. ¿El candidato sigue por encima de V20 en una particion que no vio?
2. ¿Cuantos folds salen positivos? El punto 22 del brief exige 5/5 en la particion original.
3. ¿La mejora se mantiene en magnitud, o era ruido de la particion?

## Por que se ejecuta con la configuracion congelada y no con una nueva busqueda

Porque el punto 42 lo prohibe explicitamente. Una segunda particion con re-optimizacion seria una
segunda oportunidad de encontrar una buena solucion, y ahi un resultado bueno **no significaria
nada**: seria el maximo de dos tiradas, no una medida. aqui solo hay una medida y por eso vale.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from phase2_ponderacion import EncodedMLP  # noqa: E402
from v13_p1b_fusion_tokens import logreg  # noqa: E402
from v19_b1_boosting import HIERBA, matriz_279  # noqa: E402

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.config import set_seed  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

RESULTADO = cfg.EXPERIMENTS / "v30_05_segunda_particion.json"
TABLA = cfg.REPORTS / "v30_segunda_particion.csv"
FIG = cfg.FIGURES / "v30_segunda_particion.png"
CV_V20 = 0.6097782
N_CLASES = 6
HILOS = 28
#: Semilla distinta de la de la particion original. Es lo unico que cambia.
SEMILLA_SEGUNDA = 987654
GATE_DELTA = 0.003
MARGINAL_MIN = 0.002


def _macro(clase: np.ndarray, prob: np.ndarray) -> float:
    return float(f1_score(clase, np.argmax(prob, axis=1), average="macro",
                          labels=list(range(N_CLASES))))


def a_clases(probabilidad: np.ndarray, clases: np.ndarray, filas: int) -> np.ndarray:
    salida = np.zeros((filas, N_CLASES))
    for destino, etiqueta in enumerate(cfg.CLASSES):
        salida[:, destino] = probabilidad[:, list(clases).index(etiqueta)]
    return salida


def _cuartos_v20(matriz: np.ndarray, clase: np.ndarray, y: np.ndarray, tren: np.ndarray,
                 val: np.ndarray) -> np.ndarray:
    """Los cuatro cuartos de V20 sobre `tren`, medidos en `val`. Receta intacta."""
    prob_lr = a_clases(logreg().fit(matriz[tren], y[tren]).predict_proba(matriz[val]),
                       np.array(cfg.CLASSES), len(val))
    red = EncodedMLP().fit(matriz[tren], y[tren])
    prob_mlp = a_clases(red.model.predict_proba(red.scaler.transform(matriz[val])),
                        red.encoder.classes_, len(val))
    prob_arbol = a_clases(lgb.LGBMClassifier(**HIERBA).fit(matriz[tren], y[tren])
                          .predict_proba(matriz[val]), np.array(cfg.CLASSES), len(val))
    acumuladas = np.zeros((len(val), N_CLASES - 1))
    for k in range(1, N_CLASES):
        objetivo = (clase <= k - 1).astype(int)
        modelo = lgb.LGBMClassifier(**HIERBA).fit(matriz[tren], objetivo[tren])
        acumuladas[:, k - 1] = modelo.predict_proba(matriz[val])[:, 1]
    for k in range(1, N_CLASES - 1):
        acumuladas[:, k] = np.maximum(acumuladas[:, k], acumuladas[:, k - 1])
    seis = np.zeros((len(val), N_CLASES))
    seis[:, 0] = acumuladas[:, 0]
    for k in range(1, N_CLASES - 1):
        seis[:, k] = acumuladas[:, k] - acumuladas[:, k - 1]
    seis[:, N_CLASES - 1] = 1.0 - acumuladas[:, N_CLASES - 2]
    seis = np.clip(seis, 0.0, None)
    seis = seis / np.clip(seis.sum(axis=1, keepdims=True), 1e-9, None)
    return 0.25 * prob_lr + 0.25 * prob_mlp + 0.25 * prob_arbol + 0.25 * seis


#: Parametros de LightGBM que la libreria exige como enteros. Al pasar por un CSV, pandas los
#: devuelve como float y LightGBM aborta con `Parameter num_leaves should be of type int`. Se
#: normalizan aqui, en el punto de uso, en vez de confiar en que el origen los traiga bien.
PARAMETROS_ENTEROS = ("num_leaves", "max_depth", "min_child_samples", "subsample_freq",
                      "n_estimators")


def _normaliza(params: dict) -> dict:
    """Devuelve la configuracion con los parametros enteros convertidos a `int` de verdad."""
    salida = dict(params)
    for clave in PARAMETROS_ENTEROS:
        if clave in salida and salida[clave] is not None:
            salida[clave] = int(round(float(salida[clave])))
    return salida


def _configs_del_miembro(nombre: str) -> list[dict]:
    """Los parametros del mejor trial de cada fold, en orden de fold.

    **V29 no los persistio en su artefacto**: su JSON guarda la lista `folds` con estadisticos pero
    no los `params_mejores`, asi que hay que recuperarlos de `reports/v29-01_trials.csv`. V30 si los
    guarda en su artefacto. Es un hueco de trazabilidad de V29, menor pero real, y se declara en
    lugar de disimularlo: sin ese CSV, la configuracion de V29 seria irrecuperable.
    """
    artefacto = ("v29_04_optuna_lightgbm.json" if "v29" in nombre
                 else "v30_01_optuna_refinamiento.json")
    datos = json.loads((cfg.EXPERIMENTS / artefacto).read_text(encoding="utf-8"))
    por_fold = datos.get("detalle_por_fold") or {}
    if por_fold and "params_mejores" in next(iter(por_fold.values())):
        return [f["params_mejores"] for _, f in sorted(por_fold.items(),
                                                       key=lambda kv: int(kv[0]))]
    csv = cfg.REPORTS / ("v29-01_trials.csv" if "v29" in nombre else "v30_optuna_trials.csv")
    tabla = pd.read_csv(csv)
    salida = []
    for _, g in tabla.groupby("fold"):
        mejor = g[g["es_el_mejor"]]
        fila = (mejor if not mejor.empty else g.nsmallest(1, "macro_f1_inner")).iloc[0]
        salida.append({k.replace("p_", ""): (int(fila[k]) if float(fila[k]).is_integer()
                                              else float(fila[k]))
                        for k in g.columns if k.startswith("p_")})
    return salida


def main() -> int:
    """Evalua el candidato congelado sobre una particion agrupada distinta."""
    autotest_numerico()
    set_seed()
    inicio = time.time()
    ens = json.loads((cfg.EXPERIMENTS / "v30_02_ensemble.json").read_text(encoding="utf-8"))
    manifest = ens["manifest"]
    pesos = manifest["pesos"]
    principal = next(f for f in ens["tabla"] if f["metodo"] == ens["candidato_principal"])
    print(f"[V30] candidato congelado: {ens['candidato_principal']}", flush=True)
    print(f"[V30] pesos congelados: {pesos}", flush=True)
    print(f"[V30] su delta en la particion ORIGINAL era {principal['delta_vs_V20']:+.6f} con "
          f"{principal['folds_positivos']}/5 folds. Ahora se mide sobre otra particion, con la "
          f"configuracion **congelada** y sin re-optimizar nada.", flush=True)

    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(v) for v in y])
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    matriz = matriz_279()

    # La particion nueva: mismos grupos, misma familia de CV, otra semilla.
    folds_segundos = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1,
                                      scheme="group", seed=SEMILLA_SEGUNDA)
    originales = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")
    solapes = [len(set(np.asarray(o["val_idx"])) & set(np.asarray(n["val_idx"])))
               for o, n in zip(originales, folds_segundos)]
    print(f"[V30] particiones realmente distintas: solapes por fold {solapes} "
          f"(0 significaria que no cambio nada)", flush=True)

    # Configuraciones congeladas de cada LightGBM, indexadas por fold.
    configs = {n: _configs_del_miembro(n) for n in manifest["miembros"]
               if n != "v20_f3b_oof_candidato"}

    oof_candidato = np.zeros((len(y), N_CLASES))
    oof_v20 = np.zeros((len(y), N_CLASES))
    filas = []
    for fold in folds_segundos:
        idx = int(fold["fold"])
        val = np.asarray(fold["val_idx"])
        tren = np.concatenate([np.asarray(o["val_idx"]) for o in folds_segundos
                               if o["fold"] != idx])
        prob_v20 = _cuartos_v20(matriz, clase, y, tren, val)
        oof_v20[val] = prob_v20
        miembros = {"v20_f3b_oof_candidato": prob_v20}
        for nombre, lista in configs.items():
            p = {"objective": "multiclass", "num_class": N_CLASES, "verbose": -1,
                 "n_jobs": HILOS, "subsample_freq": 1, **_normaliza(lista[idx])}
            techo = int(p.pop("n_estimators"))
            modelo = lgb.LGBMClassifier(**p, n_estimators=techo).fit(matriz[tren], y[tren])
            miembros[nombre] = a_clases(modelo.predict_proba(matriz[val]),
                                        modelo.classes_, len(val))
        nombres = list(miembros)
        vector = np.array([pesos.get(n, 0.0) for n in nombres], dtype=float)
        vector = vector / vector.sum()
        prob = np.tensordot(vector, np.stack([miembros[n] for n in nombres]), axes=(0, 0))
        oof_candidato[val] = prob
        cv_c = _macro(clase[val], prob)
        cv_v = _macro(clase[val], prob_v20)
        filas.append({"fold": idx, "n_train": int(len(tren)), "n_val": int(len(val)),
                      "corrected_cv_V20": round(cv_v, 7),
                      "corrected_cv_candidato": round(cv_c, 7),
                      "delta": round(cv_c - cv_v, 6)})
        print(f"    fold {idx}: V20 {cv_v:.6f} | candidato {cv_c:.6f} | delta "
              f"{cv_c - cv_v:+.6f}", flush=True)

    cv_c = _macro(clase, oof_candidato)
    cv_v = _macro(clase, oof_v20)
    deltas = [f["delta"] for f in filas]
    positivos = sum(1 for d in deltas if d > 0)
    # El gate se evalua contra el V20 **recomputado en esta misma particion**, porque los folds
    # son distintos: la referencia correcta es la de al lado, no la constante del champion.
    delta_contra_recomputado = cv_c - cv_v
    cumple = delta_contra_recomputado >= GATE_DELTA and positivos >= 4
    marginal = positivos == 5 and MARGINAL_MIN <= delta_contra_recomputado < GATE_DELTA
    print(f"\n[V30] segunda particion: V20 recomputado {cv_v:.7f} | candidato {cv_c:.7f} | "
          f"delta {delta_contra_recomputado:+.6f} | folds +{positivos}/5", flush=True)
    print(f"[V30] el mismo delta frente al canonico {CV_V20} seria {cv_c - CV_V20:+.6f}; se "
          f"informa tambien, pero **el gate usa el recomputado**", flush=True)
    print(f"[V30] gate historico: {cumple} | STRONG_MARGINAL: {marginal}", flush=True)

    pd.DataFrame(filas).to_csv(TABLA, index=False)
    _figura(filas, principal["delta_por_fold"])

    RESULTADO.write_text(json.dumps({
        "fase": "V30 05 segunda particion", "fecha": "2026-09-30", "experimenta": False,
        "candidato": ens["candidato_principal"], "pesos_congelados": pesos,
        "semilla_nueva": SEMILLA_SEGUNDA,
        "grupos": "los mismos grupos de dup_clusters, sin cambiar",
        "reoptimizado": False,
        "solapes_con_la_particion_original": solapes,
        "particion_realmente_distinta": all(s < 1000 for s in solapes),
        "corrected_cv_V20_segunda": round(cv_v, 7),
        "corrected_cv_candidato_segunda": round(cv_c, 7),
        "delta_vs_V20_recomputado": round(delta_contra_recomputado, 6),
        "delta_vs_canonico_V20": round(cv_c - CV_V20, 6),
        "criterio_del_delta": "el gate se evalua contra el V20 **recomputado en esta misma "
                              "particion**, porque los folds son distintos; contra el canonico se "
                              "informa tambien, pero no decide",
        "delta_vs_V20_particion_original": principal["delta_vs_V20"],
        "diferencia_de_delta_entre_particiones": round(
            delta_contra_recomputado - principal["delta_vs_V20"], 6),
        "folds_positivos": positivos, "folds": filas,
        "cumple_gate_historico": bool(cumple), "es_strong_marginal": bool(marginal),
        "criterio_del_brief": "se ejecuta si el candidato tiene 5/5 folds y delta >= +0,002 en la "
                              "particion original. La configuracion se aplica **congelada**.",
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[V30] {time.time() - inicio:.0f} s")
    return 0


def _figura(filas: list[dict], deltas_originales: list) -> None:
    """Delta por fold en las dos particiones, lado a lado."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ejes = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(5)
    eje = ejes[0]
    eje.bar(x - 0.2, deltas_originales, 0.4, label="particion original", color="#08519c")
    eje.bar(x + 0.2, [f["delta"] for f in filas], 0.4, label="segunda particion", color="#cb181d")
    eje.axhline(0, color="grey", linewidth=1)
    eje.axhline(GATE_DELTA, color="black", linestyle="--", linewidth=1, label="gate +0,003")
    eje.set_xticks(x)
    eje.set_xticklabels([f"fold {i + 1}" for i in x])
    eje.set_ylabel("delta de macro-F1 frente a V20")
    eje.set_title("El candidato en las dos particiones, con configuracion congelada", fontsize=9)
    eje.legend(fontsize=7)

    eje = ejes[1]
    eje.scatter([0] + [1] * 5, [0] + [f["delta"] for f in filas], s=60, color="#cb181d",
                label="segunda particion")
    eje.scatter([2] + [3] * 5, [0] + list(deltas_originales), s=60, color="#08519c",
                label="particion original")
    eje.axhline(GATE_DELTA, color="black", linestyle="--", linewidth=1, label="gate +0,003")
    eje.axhline(MARGINAL_MIN, color="grey", linestyle=":", linewidth=1, label="borde STRONG_MARGINAL")
    eje.set_xticks([0.5, 2.5])
    eje.set_xticklabels(["segunda", "original"])
    eje.set_ylabel("delta frente a V20")
    eje.set_title("Delta agregado y por fold\n(el 0 a la izquierda es el agregado)", fontsize=9)
    eje.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
