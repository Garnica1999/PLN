"""V30 - 01 - P1: refinamiento local de capacidad del LightGBM de V29.

V29 demostro que hay capacidad nueva real: un LightGBM sobre la **misma** matriz de 279 features
supero a V20. V30 se pregunta si esa ganancia **se consolida** con una busqueda mas estrecha y
centrada. No es una busqueda nueva desde cero, y por eso el espacio y la siembra son explicitos.

## Por que es local y como se hace local de verdad

Los cinco mejores trials de V29 coinciden en una region muy estrecha:

| Parametro | Rango de los 5 mejores de V29 | `HIERBA` de V20 |
|---|---|---|
| `num_leaves` | 43 - 87 | **8** |
| `max_depth` | 5 - 9 | **3** |
| `learning_rate` | 0,021 - 0,043 | 0,05 |
| `min_child_samples` | 22 - 94 | **200** |
| `colsample_bytree` | 0,47 - 0,71 | 0,5 |
| `reg_lambda` | 1e-7 - 0,018 | **1,0** |

`HIERBA` esta **sub-parametrizada en casi todo**: 8 hojas frente a 43-87, y `min_child_samples` 200
frente a 22-94. Ese es el mecanismo de la mejora de V29, y por eso el espacio de V30 se centra ahi
en vez de recorrer la region entera.

La localidad no es solo un rango estrecho: **el trial 0 de cada fold se encola con los parametros
exactos del mejor trial de V29 para ese fold**. La busqueda parte de ahi, no de un punto al azar.

## Estructura anidada, sin excepciones

    outer_train
      +-- inner_train / inner_val    <-- Optuna elige aqui, y el trial 0 es el de V29
      +-- refit sobre TODO el outer_train con la mejor configuracion
            +--> outer_val           <-- aqui SOLO se mide

Hay una **asercion** que aborta si el holdout interior toca la validacion exterior. El objetivo es
`macro-F1` del interior y nada mas: no hay ninguna funcion que mire el leaderboard ni el OOF
externo de V29.

## Presupuesto

40 trials por fold, **200 en total**, que es el maximo que autoriza el brief. Con early stopping y
techo de 2.500 estimadores; el corte real lo pone el criterio interior, que en V29 fue de 499 a
731. **early stopping en 350 o 500 iteraciones es el comportamiento correcto**, no un fallo: es lo
que dice el punto 38 del brief.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
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
from v19_b1_boosting import matriz_279  # noqa: E402

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

RESULTADO = cfg.EXPERIMENTS / "v30_01_optuna_refinamiento.json"
OOF_V30 = cfg.FEATURES / "v30_lgbm_refinado_oof.npy"
FIG = cfg.FIGURES / "v30_optuna_refinamiento.png"
CV_V20 = 0.6097782
CV_LGBM_V29 = 0.6121698
N_CLASES = 6
#: Trials por fold. El brief autoriza "40 por outer fold como maximo" y "no superar 200 en
#: total". Se usan **15 por fold, 75 en total**, dentro del tope, y por una razon que se declara
#: en vez de disimular: la busqueda **arranca sembrada en el mejor trial de V29**, luego no
#: necesita 40 tiradas para saber si hay algo mejor a su alrededor; le bastan unas pocas para que
#: TPE haya explorado el vecindario.
#:
#: El techo de iteraciones baja de 2.500 a 1.500 por la misma clase de razon, y con un dato a su
#: favor: en V29 el early stopping corto en **499 a 731** estimadores con un techo de 1.500. Un
#: techo de 2.500 no daba margen adicional, solo coste. Las 1.500 coinciden con el techo de V29 y
#: hacen comparables las dos busquedas.
#:
#: Nota de trazabilidad: se lanzo primero una version con 40 trials y techo 2.500, y tras **1 h 46
#: min no habia terminado ni el fold 0**, con contencion de CPU externa: otro proyecto del equipo
#: consumia siete nucleos y la maquina iba a load 45 sobre 32. Se detuvo y se relanzo con el
#: presupuesto de arriba, que es el que produjo el artefacto. Ningun numero de este script viene
#: de la version descartada.
TRIALS_POR_FOLD = 15
HILOS = 28
SEMILLA_OPTUNA = 20300930
MAX_ESTIMADORES = 1500
EARLY_STOPPING = 60
FRACCION_INTERIOR = 0.2
#: Centro del espacio: la mediana de los cinco mejores trials de V29. Se usa como punto de partida
#: declarado, no como un parametro oculto.
CENTRO_V29 = {
    "num_leaves": 56, "max_depth": 7, "learning_rate": 0.030,
    "min_child_samples": 54, "subsample": 0.78, "colsample_bytree": 0.60,
    "reg_alpha": 0.015, "reg_lambda": 0.0005,
}


def _jsonable(valor):
    """Convierte tipos de numpy a tipos nativos de Python, recursivamente.

    Sin esto, `json.dumps` aborta con `Object of type int64 is not JSON serializable` en cuanto
    un valor viene de un DataFrame de pandas. Es exactamente lo que paso con `siembra_por_fold`.
    """
    if isinstance(valor, dict):
        return {str(k): _jsonable(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_jsonable(v) for v in valor]
    if isinstance(valor, np.integer):
        return int(valor)
    if isinstance(valor, np.floating):
        return float(valor)
    if isinstance(valor, np.bool_):
        return bool(valor)
    if isinstance(valor, np.ndarray):
        return _jsonable(valor.tolist())
    return valor


def _macro(clase: np.ndarray, pred: np.ndarray) -> float:
    return float(f1_score(clase, pred, average="macro", labels=list(range(N_CLASES))))


def _espacio(trial: optuna.Trial) -> dict:
    """Espacio local: ceñido a la region donde V29 encontro lo mejor.

    Cada rango se centra en la mediana de los cinco mejores trials de V29 y se abre lo justo para
    explorar alrededor. No se recorre la region entera de hiperparametros, que es lo que el brief
    prohibe.
    """
    c = CENTRO_V29
    return {
        "num_leaves": trial.suggest_int("num_leaves", max(8, int(c["num_leaves"] * 0.55)),
                                        int(c["num_leaves"] * 1.70), log=True),
        "max_depth": trial.suggest_int("max_depth", max(3, c["max_depth"] - 3),
                                       c["max_depth"] + 4),
        "learning_rate": trial.suggest_float("learning_rate", c["learning_rate"] * 0.5,
                                             c["learning_rate"] * 1.8, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples",
                                               max(8, int(c["min_child_samples"] * 0.35)),
                                               int(c["min_child_samples"] * 2.6), log=True),
        "subsample": trial.suggest_float("subsample", max(0.4, c["subsample"] - 0.25),
                                         min(1.0, c["subsample"] + 0.22)),
        "colsample_bytree": trial.suggest_float("colsample_bytree",
                                                max(0.25, c["colsample_bytree"] - 0.25),
                                                c["colsample_bytree"] + 0.25),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 3.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 0.05, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 800, MAX_ESTIMADORES, log=True),
    }


def _entrena(params: dict, matriz: np.ndarray, clase: np.ndarray, entrenamiento: np.ndarray,
             validacion: np.ndarray) -> tuple[np.ndarray, dict]:
    p = {**params, "objective": "multiclass", "num_class": N_CLASES, "verbose": -1,
         "n_jobs": HILOS, "subsample_freq": 1}
    techo = p.pop("n_estimators")
    modelo = lgb.LGBMClassifier(**p, n_estimators=techo)
    modelo.fit(matriz[entrenamiento], clase[entrenamiento],
               eval_set=[(matriz[validacion], clase[validacion])],
               eval_metric="multi_logloss",
               callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)])
    usadas = int(modelo.best_iteration_ or techo)
    return modelo.predict_proba(matriz[validacion]), {
        "iteraciones": usadas, "max_iteraciones": techo, "early_stopping": usadas < techo}


def _mejores_v29() -> dict[int, dict]:
    """Los parametros del mejor trial de V29 para cada fold, desde su CSV de trials."""
    ruta = cfg.REPORTS / "v29-01_trials.csv"
    if not ruta.exists():
        return {}
    tabla = pd.read_csv(ruta)
    salida = {}
    for fold, g in tabla.groupby("fold"):
        mejor = g[g["es_el_mejor"]]
        if mejor.empty:
            continue
        fila = mejor.iloc[0]
        salida[int(fold)] = {
            k.replace("p_", ""): (int(fila[k]) if float(fila[k]).is_integer()
                                  else float(fila[k]))
            for k in g.columns if k.startswith("p_")}
    return salida


def main() -> int:
    """Ejecuta el refinamiento local y decide si consolida la ganancia de V29."""
    autotest_numerico()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(v) for v in y])
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")
    matriz = matriz_279()
    semilla_v29 = _mejores_v29()
    print(f"[V30] {matriz.shape[1]} features | {TRIALS_POR_FOLD} trials por fold "
          f"({TRIALS_POR_FOLD * len(folds)} en total) | {HILOS} hilos", flush=True)
    print(f"[V30] espacio local centrado en la mediana de los mejores trials de V29: "
          f"{CENTRO_V29}", flush=True)
    print(f"[V30] siembras disponibles para {len(semilla_v29)} de {len(folds)} folds", flush=True)

    oof = np.zeros((len(clase), N_CLASES))
    registro, resumen = [], {}
    for fold in folds:
        idx_fold = int(fold["fold"])
        validacion = np.asarray(fold["val_idx"])
        entrenamiento = np.concatenate([np.asarray(o["val_idx"]) for o in folds
                                        if o["fold"] != idx_fold])
        interior = cvmod.make_folds(clase[entrenamiento], groups=np.arange(len(entrenamiento)),
                                    n_splits=5, n_repeats=1, scheme="stratified")
        objetivo = int(len(entrenamiento) * FRACCION_INTERIOR)
        elegido = min(interior, key=lambda f: abs(len(f["val_idx"]) - objetivo))
        idx_rel = np.asarray(elegido["val_idx"])
        idx_ent = np.concatenate([np.asarray(o["val_idx"]) for o in interior
                                  if o["fold"] != elegido["fold"]])
        inner_train, inner_val = entrenamiento[idx_ent], entrenamiento[idx_rel]
        assert not np.intersect1d(inner_val, validacion).size, \
            "la validacion interior se solapa con la exterior: estructura anidada rota"

        valores: dict[int, dict] = {}

        def f(trial: optuna.Trial) -> float:
            params = _espacio(trial)
            prob, diag = _entrena(params, matriz, clase, inner_train, inner_val)
            valores[trial.number] = {"valor": _macro(clase[inner_val], np.argmax(prob, axis=1)),
                                     "params": params, "diag": diag}
            trial.set_user_attr("iteraciones", diag["iteraciones"])
            return valores[trial.number]["valor"]

        estudio = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SEMILLA_OPTUNA + idx_fold))
        # **Siembra local**: el trial 0 es exactamente el mejor de V29 para este fold, dentro del
        # espacio. La busqueda arranca ahi y no en un punto al azar.
        if idx_fold in semilla_v29:
            try:
                estudio.enqueue_trial(semilla_v29[idx_fold])
                print(f"    fold {idx_fold}: trial 0 encolado con los parametros de V29", flush=True)
            except Exception as error:  # noqa: BLE001
                print(f"    fold {idx_fold}: no se pudo sembrar ({error}); la busqueda arranca "
                      f"sin la semilla de V29", flush=True)
        t_fold = time.time()
        estudio.optimize(f, n_trials=TRIALS_POR_FOLD, show_progress_bar=False)
        mejor = estudio.best_trial.number
        prob_ext, diag_ext = _entrena(valores[mejor]["params"], matriz, clase,
                                      entrenamiento, validacion)
        oof[validacion] = prob_ext

        lista = np.array([valores[n]["valor"] for n in sorted(valores)])
        for n in sorted(valores):
            registro.append({"fold": idx_fold, "trial": n,
                             "macro_f1_inner": round(valores[n]["valor"], 6),
                             "iteraciones": valores[n]["diag"]["iteraciones"],
                             "max_iteraciones": valores[n]["diag"]["max_iteraciones"],
                             "early_stopping": valores[n]["diag"]["early_stopping"],
                             "es_el_mejor": n == mejor,
                             "es_la_siembra_de_v29": n == 0,
                             **{f"p_{k}": v for k, v in valores[n]["params"].items()}})
        mediana, desv = float(np.median(lista)), float(lista.std())
        # Coherencia entre folds: el desvio de la mejora sobre la mediana se mide en desviaciones
        # tipicas de los trials de ese fold. Un ganador outlier no se puede generalizar.
        sigmas = (float(lista.max()) - mediana) / max(desv, 1e-9)
        resumen[idx_fold] = {
            "mejor_trial": int(mejor), "valor_mejor_inner": round(float(lista.max()), 6),
            "mediana_trial": round(mediana, 6), "min_trial": round(float(lista.min()), 6),
            "desviacion_trial": round(desv, 6),
            "mejor_sobre_mediana_en_sigmas": round(sigmas, 4),
            "mejor_es_outlier": bool(sigmas > 3.0),
            "macro_f1_outer_val": round(_macro(clase[validacion], np.argmax(prob_ext, axis=1)), 6),
            "diagnostico_refit": diag_ext, "segundos": round(time.time() - t_fold, 1),
            "params_mejores": {k: (float(v) if isinstance(v, (int, float)) else v)
                               for k, v in valores[mejor]["params"].items()},
        }
        print(f"    fold {idx_fold}: mejor {mejor} inner {lista.max():.6f} | mediana {mediana:.6f} "
              f"| sigma {sigmas:.2f} | iter {diag_ext['iteraciones']} | outer "
              f"{resumen[idx_fold]['macro_f1_outer_val']:.6f} | {time.time() - t_fold:.0f} s",
              flush=True)

    np.save(OOF_V30, oof)
    cv = _macro(clase, np.argmax(oof, axis=1))
    por_fold = [f["macro_f1_outer_val"] for f in resumen.values()]
    detalle_tabla = pd.DataFrame(registro)
    detalle_tabla.to_csv(cfg.REPORTS / "v30_optuna_trials.csv", index=False)
    _figura(detalle_tabla, resumen)

    # ---------------------------------------------------------------- decision
    # La pregunta de V30 es si CONSOLIDA la ganancia de V29. El criterio no es "supera a V20", que
    # ya lo hace, sino "supera al LightGBM de V29" o "lo iguala con mejor complementariedad".
    delta_v20 = cv - CV_V20
    delta_v29 = cv - CV_LGBM_V29
    folds_base = [f["macro_f1_outer_val"] for f in
                  json.loads((cfg.EXPERIMENTS / "v29_04_optuna_lightgbm.json")
                             .read_text(encoding="utf-8"))["folds"]]
    contra_v29_fold = [round(a - b, 6) for a, b in zip(por_fold, folds_base)]
    # El delta de V29 en el mismo convenio, para que la comparacion sea de peras a peras.
    print(f"\n[V30] corrected_cv {cv:.7f} | delta V20 {delta_v20:+.6f} | delta V29-LGBM "
          f"{delta_v29:+.6f}", flush=True)
    print(f"[V30] fold a fold contra el LightGBM de V29: {contra_v29_fold}", flush=True)
    if delta_v29 > 0.0005:
        estado = "MEJORA_SOBRE_V29"
    elif delta_v29 > -0.0005:
        estado = "EQUIVALE_A_V29"
    else:
        estado = "PEOR_QUE_V29"
    print(f"[V30] veredicto: {estado}", flush=True)

    RESULTADO.write_text(json.dumps(_jsonable({
        "fase": "V30 01 refinamiento local del LightGBM", "fecha": "2026-09-30", "experimenta": True,
        "experimento_id": "V30-01",
        "espacio_local": CENTRO_V29,
        "justificacion_del_espacio": "centrado en la mediana de los cinco mejores trials de V29, "
                                     "no en la region entera; `HIERBA` de V20 esta "
                                     "sub-parametrizada (8 hojas frente a 43-87 de V29, "
                                     "min_child_samples 200 frente a 22-94)",
        "siembra_por_fold": {str(k): v for k, v in semilla_v29.items()},
        "trials_por_fold": TRIALS_POR_FOLD, "trials_totales": TRIALS_POR_FOLD * len(folds),
        "sampler": f"TPESampler(seed={SEMILLA_OPTUNA} + fold) con trial 0 encolado",
        "max_estimadores": MAX_ESTIMADORES, "early_stopping_rondas": EARLY_STOPPING,
        "corrected_cv": round(cv, 7), "delta_vs_V20": round(delta_v20, 6),
        "delta_vs_lgbm_v29": round(delta_v29, 6),
        "media_por_fold": round(float(np.mean(por_fold)), 6),
        "desviacion_por_fold": round(float(np.std(por_fold)), 6),
        "folds_outer_val": por_fold,
        "folds_lgbm_v29": folds_base,
        "delta_por_fold_contra_v29": contra_v29_fold,
        "folds_mejor_que_v29": sum(1 for d in contra_v29_fold if d > 0),
        "mejores_trial_outlier": sum(1 for f in resumen.values() if f["mejor_es_outlier"]),
        "detalle_por_fold": resumen,
        "oof_guardado": OOF_V30.name,
        "estado_vs_v29": estado,
        "criterio_de_consolidacion": "delta frente al LightGBM de V29 superior a +0,0005; entre "
                                     "-0,0005 y +0,0005 se considera equivalente y se decide por "
                                     "complementariedad en la fase de ensemble",
        "cv_v20": CV_V20, "cv_lgbm_v29": CV_LGBM_V29,
        "segundos": round(time.time() - inicio, 1),
    }), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[V30] {time.time() - inicio:.0f} s totales")
    return 0


def _figura(tabla: pd.DataFrame, resumen: dict) -> None:
    """Historia de optimizacion por fold, distribucion de trials y outer_val frente a V29."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ejes = plt.subplots(1, 3, figsize=(16, 4.6))
    eje = ejes[0]
    for fold, g in tabla.groupby("fold"):
        eje.plot(g["trial"], g["macro_f1_inner"], marker="o", markersize=3, linewidth=1,
                 label=f"fold {fold}")
    eje.set_xlabel("trial (el 0 es la siembra de V29)")
    eje.set_ylabel("macro-F1 en validacion interior")
    eje.set_title("Historia del refinamiento local\n(el primer punto de cada curva es V29)",
                  fontsize=9)
    eje.legend(fontsize=7)

    eje = ejes[1]
    for fold, g in tabla.groupby("fold"):
        eje.hist(g["macro_f1_inner"], bins=12, alpha=0.5, label=f"fold {fold}")
    for fold, s in resumen.items():
        eje.axvline(s["valor_mejor_inner"], linewidth=1, linestyle="--")
    eje.set_xlabel("macro-F1 interior")
    eje.set_ylabel("trials")
    eje.set_title("Distribucion de trials; las lineas son el mejor de cada fold", fontsize=9)
    eje.legend(fontsize=7)

    eje = ejes[2]
    etiquetas = [f"f{f}" for f in sorted(resumen)]
    v30 = [resumen[f]["macro_f1_outer_val"] for f in sorted(resumen)]
    v29 = json.loads((cfg.EXPERIMENTS / "v29_04_optuna_lightgbm.json")
                     .read_text(encoding="utf-8"))["folds"]
    v29 = [f["macro_f1_outer_val"] for f in v29]
    x = np.arange(len(etiquetas))
    eje.bar(x - 0.2, v29, 0.4, label="V29 LightGBM", color="#999999")
    eje.bar(x + 0.2, v30, 0.4, label="V30 refinado", color="#08519c")
    eje.axhline(CV_V20, color="red", linestyle=":", linewidth=1, label=f"V20 = {CV_V20}")
    eje.set_xticks(x)
    eje.set_xticklabels(etiquetas)
    eje.set_ylabel("macro-F1 en validacion EXTERIOR")
    eje.set_title("Outer validation: el refinamiento frente a V29", fontsize=9)
    eje.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
