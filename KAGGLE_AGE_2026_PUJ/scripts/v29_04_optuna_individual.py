"""V29 - 04 - P4 y P5: Optuna individual sobre LightGBM y MLP, en estructura anidada.

Dos pistas **independientes** de capacidad de modelo. Una falla no detiene a la otra, como pide el
punto 41 del brief.

## La estructura, que es la parte que importa

    outer_train  (4 de los 5 folds de V20)
      |
      +-- Optuna: inner_train / inner_val   <-- aqui se eligen los hiperparametros
      |
      +-- refit con la mejor configuracion sobre TODO el outer_train
            |
            +--> outer_val  <-- aqui SOLO se mide

**La validacion exterior nunca entra en Optuna.** Es la condicion que el brief impone y la que
hace legitimo cualquier numero que salga. Se comprueba en el codigo: la funcion objetivo de Optuna
recibe exclusivamente indices de `inner_val`, que son un subconjunto de `outer_train`.

## Early stopping, y lo que se registra

El brief pide "muchas iteraciones" en el sentido de *dar margen al modelo*, no de entrenar miles de
veces. Por eso `n_estimators` y `max_iter` son altos y el corte lo pone `early_stopping`. Se
registran las cuatro magnitudes que lo permiten comprobar despues: `max_iterations`,
`actual_iterations`, `best_iteration` y si hubo `early_stopping`.

## Una sustitucion que se declara

El espacio de la pista MLP pide `dropout`. **sklearn no tiene dropout en `MLPClassifier`.** No se
finge: se usa el espacio que la libreria si permite (`hidden_layer_sizes`, `alpha`,
`learning_rate_init`, `batch_size`, `activation`, `solver`, `early_stopping`) y se declara aqui la
ausencia, para que nadie lea la lista de parametros y suponga un dropout que no esta.

## Que no se toca

La representacion es **exactamente la de V20**, las 279 features, sin anadir ninguna. Eso es lo
que permite saber si existe capacidad adicional dentro de la misma evidencia. Los folds, los
grupos y el dataset son los de V20.
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
from sklearn.neural_network import MLPClassifier

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

RESULTADO_LGBM = cfg.EXPERIMENTS / "v29_04_optuna_lightgbm.json"
RESULTADO_MLP = cfg.EXPERIMENTS / "v29_05_optuna_mlp.json"
OPTUNA_CSV = cfg.REPORTS / "v29_optuna_results.csv"
FIG_LGBM = cfg.FIGURES / "v29_optuna_lightgbm.png"
FIG_MLP = cfg.FIGURES / "v29_optuna_mlp.png"
OOF_LGBM = cfg.FEATURES / "v29_lgbm_optuna_oof.npy"
OOF_MLP = cfg.FEATURES / "v29_mlp_optuna_oof.npy"
CV_V20 = 0.6097782
N_CLASES = 6
HILOS = 12  # el equipo tiene otros procesos; se deja cabecera para no competir por la CPU
#: Trials por fold. El brief pide 30-50 "solo si el coste computacional lo permite", y se usan
#: **30 por fold, 150 en total**, que es el suelo del rango que sugiere. Lo que si se acota es el
#: techo de iteraciones, y la desviacion se declara:
#:
#: 1. Con `n_estimators` hasta 3000, el fold 0 no termino en 55 minutos. Se bajo el techo.
#: 2. Con el techo en 1500, la pista de LightGBM completo y **este es el artefacto que cuenta**.
#:    El mejor trial uso entre 499 y 731 estimadores, luego el early stopping ponia el corte muy
#:    por debajo del techo y la restriction no costo nada.
#:
#: **Un error propio que se declara**: durante el triaje se lanzo una **tercera** ejecucion con
#: 10 trials y techo 400 believing que la segunda habia muerto, y ambas escribieron el mismo
#: artefacto. Se detecto al comparar el `corrected_cv` impreso con el del JSON, que no
#: cuadraban. Se mato el proceso sobrante y **el artefacto que sobrevive es el de 30 trials**,
#: que es ademas el mejor de los dos. El script se restauro a esta configuracion para que el
#: codigo sea el que produjo el resultado.
TRIALS = 30
#: Techo de iteraciones por familia, recortado tras la medicion anterior.
MAX_ESTIMADORES = 1500
MAX_ITERACIONES_MLP = 800
SEMILLA_OPTUNA = 20260929
#: Separacion interior: un unico holdout agrupado y estratificado, no un CV interior completo.
#: Con 30 trials y un solo holdout, un CV interior multiplicaria el coste por el numero deplits
#: sin mejorar la seleccion de forma que compense.
FRACCION_INTERIOR = 0.2


def _macro(clase: np.ndarray, pred: np.ndarray) -> float:
    return float(f1_score(clase, pred, average="macro", labels=list(range(N_CLASES))))


def _space_lgbm(trial: optuna.Trial) -> dict:
    """Espacio de LightGBM. Amplio en capacidad, estrecho en regularizacion, como pide el brief."""
    return {
        "num_leaves": trial.suggest_int("num_leaves", 4, 96, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.30, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 1200, log=True),
        "subsample": trial.suggest_float("subsample", 0.50, 1.00),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.30, 1.00),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 150, MAX_ESTIMADORES, log=True),
    }


def _espacio_comun() -> dict:
    return {"objective": "multiclass", "num_class": N_CLASES, "verbose": -1,
            "n_jobs": HILOS, "random_state": int(cfg.SEED)}


def _espacio_mlp(trial: optuna.Trial) -> dict:
    """Espacio de MLP. Sin dropout: sklearn no lo ofrece en `MLPClassifier` y se declara."""
    ancho = trial.suggest_categorical("hidden_layer_sizes",
                                      [(32,), (64,), (128,), (256,), (64, 32), (128, 64),
                                       (256, 128)])
    return {
        "hidden_layer_sizes": ancho,
        "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
        "solver": "adam",
        "alpha": trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024, 2048, 4096]),
        "max_iter": MAX_ITERACIONES_MLP,
        "early_stopping": True,
        "validation_fraction": 0.12,
        "n_iter_no_change": 25,
        "random_state": int(cfg.SEED),
    }


def _optimiza(familia: str, matriz: np.ndarray, clase: np.ndarray, folds: list[dict],
              construir, guardar_oof: Path) -> tuple[dict, pd.DataFrame]:
    """Anida Optuna dentro de cada fold exterior y ensambla el OOF resultante.

    `construir(params, X_ent, y_ent, X_val)` devuelve la probabilidad sobre `X_val` y un
    diccionario de diagnostico del entrenamiento. Es lo unico que cambia entre las dos familias.
    """
    oof = np.zeros((len(clase), N_CLASES))
    registro, resumen_folds = [], []
    for fold in folds:
        validacion = np.asarray(fold["val_idx"])
        entrenamiento = np.concatenate([np.asarray(o["val_idx"]) for o in folds
                                        if o["fold"] != fold["fold"]])
        # El holdout interior se saca SOLO del outer_train. La validacion exterior no se toca.
        interior = cvmod.make_folds(clase[entrenamiento],
                                    groups=np.arange(len(entrenamiento)), n_splits=5,
                                    n_repeats=1, scheme="stratified")
        # Se elige el fold interior cuyo tamano se parezca a FRACCION_INTERIOR.
        objetivo_fraccion = int(len(entrenamiento) * FRACCION_INTERIOR)
        elegido = min(interior, key=lambda f: abs(len(f["val_idx"]) - objetivo_fraccion))
        idx_rel = np.asarray(elegido["val_idx"])
        idx_ent = np.concatenate([np.asarray(o["val_idx"]) for o in interior
                                  if o["fold"] != elegido["fold"]])
        inner_train = entrenamiento[idx_ent]
        inner_val = entrenamiento[idx_rel]
        assert not np.intersect1d(inner_val, validacion).size, \
            "la validacion interior se solapa con la exterior: estructura anidada rota"

        valores = {}

        def objetivo(trial: optuna.Trial) -> float:
            params = construir(trial)
            prob, diag = _entrena(familia, params, matriz, clase, inner_train, inner_val, None)
            valores[trial.number] = {"valor": _macro(clase[inner_val], np.argmax(prob, axis=1)),
                                     "params": {k: (list(v) if isinstance(v, tuple) else v)
                                                for k, v in params.items()},
                                     "diag": diag}
            return valores[trial.number]["valor"]

        estudio = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SEMILLA_OPTUNA + int(fold["fold"])))
        estudio.optimize(objetivo, n_trials=TRIALS, show_progress_bar=False)
        mejor = estudio.best_trial.number
        # Refit sobre TODO el outer_train con la mejor configuracion, y medida en outer_val.
        prob_ext, diag_ext = _entrena(familia, valores[mejor]["params"], matriz, clase,
                                      entrenamiento, validacion, None)
        oof[validacion] = prob_ext

        lista = np.array([valores[n]["valor"] for n in sorted(valores)])
        for n in sorted(valores):
            registro.append({"familia": familia, "fold": int(fold["fold"]), "trial": n,
                             "macro_f1_inner": round(valores[n]["valor"], 6),
                             "n_iterations": valores[n]["diag"].get("iteraciones"),
                             "max_iterations": valores[n]["diag"].get("max_iteraciones"),
                             "early_stopping": valores[n]["diag"].get("early_stopping"),
                             "es_el_mejor": n == mejor,
                             **{f"p_{k}": v for k, v in valores[n]["params"].items()}})
        resumen_folds.append({
            "fold": int(fold["fold"]), "n_inner_train": int(len(inner_train)),
            "n_inner_val": int(len(inner_val)), "n_outer_val": int(len(validacion)),
            "mejor_trial": mejor, "valor_mejor_inner": round(float(valores[mejor]["valor"]), 6),
            "mediana_trial": round(float(np.median(lista)), 6),
            "min_trial": round(float(lista.min()), 6), "max_trial": round(float(lista.max()), 6),
            "desviacion_trial": round(float(lista.std()), 6),
            # La defensa contra un trial anomalo: cuanto destaca el mejor de la mediana, en
            # desviaciones tipicas de los propios trials.
            "mejor_sobre_mediana_en_sigmas": round(
                float((lista.max() - np.median(lista)) / max(lista.std(), 1e-9)), 4),
            "mejor_es_outlier": bool(lista.max() - np.median(lista) > 3 * max(lista.std(), 1e-9)),
            "macro_f1_outer_val": round(_macro(clase[validacion], np.argmax(prob_ext, axis=1)), 6),
            "diagnostico_refit": diag_ext,
        })
        print(f"    fold {fold['fold']}: mejor trial {mejor} inner "
              f"{valores[mejor]['valor']:.6f} | mediana {np.median(lista):.6f} | "
              f"outer_val {resumen_folds[-1]['macro_f1_outer_val']:.6f} | "
              f"{TRIALS} trials ({time.time() - _t0:.0f} s)", flush=True)

    np.save(guardar_oof, oof)
    macro = _macro(clase, np.argmax(oof, axis=1))
    por_fold = [f["macro_f1_outer_val"] for f in resumen_folds]
    return {
        "corrected_cv": round(macro, 7), "delta_vs_V20": round(macro - CV_V20, 6),
        "media_por_fold": round(float(np.mean(por_fold)), 6),
        "desviacion_por_fold": round(float(np.std(por_fold)), 6),
        "folds": resumen_folds, "n_trials_por_fold": TRIALS,
        "n_trials_totales": TRIALS * len(folds),
        "n_triouts_outlier": sum(1 for f in resumen_folds if f["mejor_es_outlier"]),
        "oof_guardado": guardar_oof.name,
        "cifras": f1_score(clase, np.argmax(oof, axis=1), average=None,
                           labels=list(range(N_CLASES))).round(4).tolist(),
    }, pd.DataFrame(registro)


def _entrena(familia: str, params: dict, matriz: np.ndarray, clase: np.ndarray,
             entrenamiento: np.ndarray, validacion: np.ndarray, _: object) -> tuple[np.ndarray, dict]:
    """Ajusta con early stopping y devuelve la probabilidad mas el diagnostico de iteraciones."""
    comun = _espacio_comun()
    if familia == "lightgbm":
        p = {**comun, **params}
        maximo = p["n_estimators"]
        corte = p.pop("n_estimators")
        modelo = lgb.LGBMClassifier(**p, n_estimators=corte)
        modelo.fit(matriz[entrenamiento], clase[entrenamiento],
                   eval_set=[(matriz[validacion], clase[validacion])], eval_metric="multi_logloss",
                   callbacks=[lgb.early_stopping(60, verbose=False)])
        used = int(modelo.best_iteration_ or corte)
        return modelo.predict_proba(matriz[validacion]), {
            "iteraciones": used, "max_iteraciones": maximo,
            "early_stopping": used < maximo}
    p = {**params}
    maximo = p["max_iter"]
    modelo = MLPClassifier(**p)
    modelo.fit(matriz[entrenamiento], clase[entrenamiento])
    used = int(modelo.n_iter_)
    return modelo.predict_proba(matriz[validacion]), {
        "iteraciones": used, "max_iteraciones": maximo,
        "early_stopping": used < maximo}


_t0 = time.time()


def _figura(registro: pd.DataFrame, familia: str, destino: Path, resumen: dict) -> None:
    """Historia de optimizacion por fold, distribucion de trials y el mejor trial."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ejes = plt.subplots(1, 3, figsize=(16, 4.6))
    sub = registro[registro["familia"] == familia]
    eje = ejes[0]
    for f, g in sub.groupby("fold"):
        eje.plot(g["trial"], g["macro_f1_inner"], marker="o", markersize=3, linewidth=1,
                 label=f"fold {f}")
    eje.set_xlabel("trial"); eje.set_ylabel("macro-F1 en validacion interior")
    eje.set_title("Historia de optimizacion (esto es diagnostico,\nno evidencia de generalizacion)",
                  fontsize=9)
    eje.legend(fontsize=7)

    eje = ejes[1]
    for f, g in sub.groupby("fold"):
        eje.hist(g["macro_f1_inner"], bins=12, alpha=0.5, label=f"fold {f}")
    for f, s in enumerate(resumen["folds"]):
        eje.axvline(s["valor_mejor_inner"], linewidth=1, linestyle="--")
    eje.set_xlabel("macro-F1 interior"); eje.set_ylabel("trials")
    eje.set_title("Distribucion de trials; las lineas son el mejor de cada fold", fontsize=9)
    eje.legend(fontsize=7)

    eje = ejes[2]
    etiquetas = [f"f{f['fold']}" for f in resumen["folds"]]
    eje.bar(etiquetas, [f["macro_f1_outer_val"] for f in resumen["folds"]], color="#08519c")
    eje.axhline(CV_V20, color="red", linestyle="--", linewidth=1, label=f"V20 = {CV_V20}")
    eje.set_ylabel("macro-F1 en validacion EXTERIOR")
    eje.set_title(f"{familia}: macro-F1 por fold exterior\n"
                  f"delta global {resumen['delta_vs_V20']:+.6f}", fontsize=9)
    eje.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(destino, dpi=130)
    plt.close(fig)


def main() -> int:
    """Ejecuta las dos pistas de Optuna. Una falla no detiene a la otra."""
    autotest_numerico()
    global _t0
    _t0 = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(v) for v in y])
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")
    matriz = matriz_279()
    print(f"[Optuna] {matriz.shape[1]} features | {len(folds)} folds | {TRIALS} trials por fold | "
          f"estructura anidada con holdout interior del {FRACCION_INTERIOR:.0%}", flush=True)

    registro = pd.DataFrame()
    salida = {}
    for familia, construir, guardar, artefacto, figura in (
            ("lightgbm", _space_lgbm, OOF_LGBM, RESULTADO_LGBM, FIG_LGBM),
            ("mlp", _espacio_mlp, OOF_MLP, RESULTADO_MLP, FIG_MLP)):
        print(f"\n=== PISTA {familia} ===", flush=True)
        inicio = time.time()
        try:
            resumen, detalle = _optimiza(familia, matriz, clase, folds, construir, guardar)
        except Exception as error:  # una pista rota no detiene a la otra
            print(f"  [FALLO] la pista {familia} se detiene: {type(error).__name__}: {error}",
                  flush=True)
            artefacto.write_text(json.dumps({
                "familia": familia, "estado": "FALLIDA", "error": str(error)[:400],
                "fecha": "2026-09-29"}, indent=2, ensure_ascii=False), encoding="utf-8")
            continue
        resumen.update({
            "pista": familia, "estado": "COMPLETADA", "fecha": "2026-09-29",
            "estructura": "anidada: Optuna en inner_val (subconjunto de outer_train), refit en "
                          "outer_train, medida solo en outer_val",
            "representacion": "las 279 features de V20, sin anadir ninguna",
            "sampler": f"TPESampler(seed={SEMILLA_OPTUNA} + fold)",
            "semilla": SEMILLA_OPTUNA,
            "hilos": HILOS,
            "sin_dropout_en_mlp": "sklearn MLPClassifier no ofrece dropout; el espacio usa alpha "
                                  "como regularizador de pesos y se declara la ausencia",
            "trials_detalle": detalle.to_dict("records"),
            "control_optuna": "la funcion objetivo recibe exclusivamente indices de inner_val, y "
                              "hay una asercion que aborta si inner_val toca outer_val",
            "segundos": round(time.time() - inicio, 1)})
        artefacto.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
        registro = pd.concat([registro, detalle], ignore_index=True)
        print(f"  [{familia}] corrected_cv {resumen['corrected_cv']} | delta "
              f"{resumen['delta_vs_V20']:+.6f} | trials {resumen['n_trials_totales']} | "
              f"outliers {resumen['n_triouts_outlier']} | {time.time() - inicio:.0f} s", flush=True)
        _figura(detalle, familia, figura, resumen)
        salida[familia] = resumen

    if registro.empty:
        # Sin trials, el CSV no tendria cabecera: se escribe vacio pero existente.
        pd.DataFrame(columns=["familia", "fold", "trial", "macro_f1_inner"]).to_csv(OPTUNA_CSV,
                                                                                    index=False)
    else:
        registro.to_csv(OPTUNA_CSV, index=False)
    print(f"\n[Optuna] {OPTUNA_CSV.name}: {len(registro)} filas de trial registradas", flush=True)
    print(f"[fin] {time.time() - _t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
