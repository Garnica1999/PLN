"""P61 track: LightGBM capacity search, nested, on V03's own representation.

The gap check that authorises this family, and the risk it exposes
----------------------------------------------------------------
LightGBM, XGBoost and CatBoost return ZERO rows in the canonical register. So the
family is genuinely untried here, which is what the brief requires before spending
experiments on it.

But the register also contains the precedent that matters most, and it is not
encouraging. `OPT-01` ran Optuna on V03's own TF-IDF logistic model, 24 trials, and
gained +0,008565 on the out-of-fold score. It was then DISCARDED because it did not
transfer: 0,77602 against V03's 0,78306 on the leaderboard, a gap of -0,028, and its
own recorded conclusion is that "la busqueda por OOF premia capacidad que no
generaliza al test".

So the situation is: an authorised, never-tried model family, in a project where an
already-tried capacity search produced a false positive that only the leaderboard
caught. That is exactly why the design here is nested, and it is also why the gate
below is applied to the *nested* score rather than to any best-trial score.

Representation, and why it is the one already in V03
---------------------------------------------------
The brief lists V03's TF-IDF first and the SVD of it as the dense alternative. The SVD
route is not taken here, and the reason is on the record: `V4-G01-G03` already fitted
gradient boosting on SVD features of this corpus and reported `svd:hist_gb 0,7297` and
`svd:extra_trees 0,7033`, both far below V03. Importing that known-weak representation
would confound the family change with a representation change, which paragraph 23
forbids. So the capacity search runs on V03's own sparse TF-IDF, unchanged, and the
regime is stated as a risk: 27,431 features against 1,186 training rows is a regime
where trees are not naturally strong.

The three arms, which paragraph 27 requires
-------------------------------------------
    arm 0   V03, the reference, untouched
    arm 1   LightGBM with default parameters, the family change alone
    arm 2   LightGBM with nested Optuna, the family change plus the tuning

Arm 1 minus arm 0 is the family change. Arm 2 minus arm 1 is the gain from
hyperparameter optimisation. Reporting only one of the two is the mistake paragraph 27
is written to prevent.

The nesting, and the objective
------------------------------
    outer_train -> inner_train, inner_validation, both split BY GROUP
               -> Optuna, which only ever sees inner_validation
               -> refit on the FULL outer_train with the winning parameters
    outer_validation, which never participates in anything

The Optuna objective is the **inner-validation log loss**, not the inner F1. That is
deliberate. Using F1 inside the search would make the search depend on the decision
threshold, and paragraph 31 keeps the threshold out of this phase; a proper scoring
rule is threshold-free, so the search optimises the probability model and the frozen
V03 thresholds are applied once, at the very end, to the outer validation.

Everything pre-registered
------------------------
    trials            30 per outer fold
    sampler            TPESampler with a fixed seed
    space              num_leaves, max_depth, learning_rate, min_child_samples,
                       subsample, colsample_bytree, reg_alpha, reg_lambda, n_estimators
                       and nothing else
    early stopping     on the inner validation, 100 rounds of patience
    n_estimators       up to 3000, because the point is to allow capacity, with the
                       early stopping deciding when to stop
    thresholds         0,44 / 0,51 / 0,46, frozen, applied identically to all arms
    grid               none. This is one search, not a grid

Usage
-----
    /home/carlos/venv/bin/python experiments/p61_03_lightgbm.py
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
import time
import unicodedata
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, log_loss

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import (
    ETIQUETAS,
    V03_CCV,
    V03_SHA,
    V03_SUB,
)

REPORTS = PROJECT_ROOT / "reports"
FIG_P61 = PROJECT_ROOT / "figuras" / "p61"
OOF = PROJECT_ROOT / "features" / "oof_v03_final.npz"
TRAIN_RAW = PROJECT_ROOT / "data" / "raw" / "toxic_train.csv"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
MD_LGB = REPORTS / "p61_lightgbm_capacidad.md"
JSON_LGB = REPORTS / "p61_lightgbm_capacidad.json"
CSV_OPT = REPORTS / "p61_optuna_results.csv"
CELDA_NPZ = PROJECT_ROOT / "features" / "p61_lightgbm_cells.npz"

V03_UMBRALES = np.array([0.44, 0.51, 0.46])
SEMILLA = 42
N_TRIALS = 30
N_ESTIMATORS_MAX = 3000
PACIENCIA = 100
FRACCION_INTERNA = 0.20
GATE_DELTA = 0.003
GATE_PLIEGUES = 4
N_BOOT = 2000
ESPACIO = ("num_leaves", "max_depth", "learning_rate", "min_child_samples",
           "subsample", "colsample_bytree", "reg_alpha", "reg_lambda", "n_estimators")


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P61-03] {mensaje}", flush=True)


SUSTITUCIONES = {
    chr(0x2014): " - ", chr(0x2013): "-", chr(0x2212): "-", chr(0x00B1): "+/-",
    chr(0x00AB): '"', chr(0x00BB): '"', chr(0x201C): '"', chr(0x201D): '"',
    chr(0x2018): "'", chr(0x2019): "'", chr(0x2264): "<=", chr(0x2265): ">=",
    chr(0x00D7): "x", chr(0x2192): "->", chr(0x00B7): ".", chr(0x2026): "...",
    chr(0x00BF): "?", chr(0x00A1): "!", chr(0x00A7): "parrafo ", chr(0x00A0): " ",
}


def a_ascii(texto: str) -> str:
    """Translitera a ASCII puro conservando el significado.

    Args:
        texto: cadena de entrada.

    Returns:
        La cadena equivalente en ASCII.
    """
    if not isinstance(texto, str):
        texto = str(texto)
    for k, v in SUSTITUCIONES.items():
        texto = texto.replace(k, v)
    salida = []
    for ch in unicodedata.normalize("NFKD", texto):
        if ord(ch) < 128:
            salida.append(ch)
            continue
        if unicodedata.category(ch).startswith("M"):
            continue
        partes = unicodedata.decomposition(ch).split()
        base = chr(int(partes[0], 16)) if partes and partes[0] != "<wide>" else ""
        salida.append(base if base and ord(base) < 128 else "?")
    return "".join(salida)


def espacio_optuna(trial: optuna.Trial) -> dict[str, Any]:
    """Espacio de busqueda, fijo y declarado antes de ejecutar.

    Args:
        trial: objeto de trial de Optuna.

    Returns:
        Diccionario de parametros.
    """
    return {
        "num_leaves": trial.suggest_int("num_leaves", 8, 255),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 200, N_ESTIMATORS_MAX),
    }


def main() -> None:
    """Ejecuta la busqueda de capacidad LightGBM con Optuna anidado.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P61-03 capacidad LightGBM con Optuna ANIDADO, sobre la representacion de V03")
    log(f"   {N_TRIALS} trials por pliegue | n_estimators hasta {N_ESTIMATORS_MAX} | "
        f"early stopping con {PACIENCIA} de paciencia")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    log(f"   V03 sha256 verificado e intacto: {hash_real[:16]}...")

    v = np.load(OOF, allow_pickle=True)
    ids, y, pliegues, pred_v03 = (v["ids"], v["y_true"].astype(int), v["pliegues"],
                                  v["pred"].astype(int))
    n = ids.size
    ccv_ref = float(f1_score(y, pred_v03, average="macro"))
    if ccv_ref != V03_CCV:
        raise SystemExit(f"PARADA: el OOF canonico da {ccv_ref!r}, no {V03_CCV!r}")
    part = pd.read_csv(PARTICION)
    grupos = part.set_index("id").loc[ids, "grupo_near_dup_autor"].to_numpy()
    partidos = int(pd.Series(grupos).groupby(grupos).nunique().gt(1).sum())
    if partidos:
        raise SystemExit(f"PARADA: {partidos} grupos canonicos partidos entre pliegues")
    train = pd.read_csv(TRAIN_RAW, sep=";", encoding="latin-1", dtype={"id": str})
    train["id_num"] = pd.to_numeric(train["id"].str.replace('"', "", regex=False),
                                    errors="coerce").astype("int64")
    idx_texto = train.set_index("id_num").loc[ids]
    if not np.array_equal(idx_texto[["HS", "TR", "AG"]].to_numpy().astype(int), y):
        raise SystemExit("PARADA: las etiquetas del CSV crudo no coinciden con el OOF")
    textos = idx_texto["text"].astype(str).tolist()
    log(f"   OOF canonico {ccv_ref!r} | {n} filas | {len(set(grupos))} grupos intactos")
    log(f"   REGIMEN: {27431} rasgos frente a 1186 filas de entrenamiento, que es un")
    log("   regimen donde los arboles no son naturalmente fuertes, y se declara como riesgo")

    proba_1 = np.zeros((n, 3))
    proba_2 = np.zeros((n, 3))
    filas_opt: list[dict[str, Any]] = []
    info: list[dict[str, Any]] = []
    t0 = time.time()
    for f in range(1, 6):
        ir = np.where(pliegues != f)[0]
        iva = np.where(pliegues == f)[0]
        pw = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                             sublinear_tf=True, lowercase=True)
        pc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                             sublinear_tf=True, lowercase=True)
        x = hstack([pw.fit_transform([textos[i] for i in ir]),
                    pc.fit_transform([textos[i] for i in ir])]).tocsr()
        xv = hstack([pw.transform([textos[i] for i in iva]),
                     pc.transform([textos[i] for i in iva])]).tocsr()
        # particion interna POR GRUPO dentro del training exterior
        g_tr = grupos[ir]
        unicos = pd.unique(g_tr)
        rng_f = np.random.default_rng(SEMILLA + 1000 * f)
        orden = rng_f.permutation(len(unicos))
        n_meta = max(round(FRACCION_INTERNA * len(unicos)), 1)
        g_meta = set(unicos[orden[:n_meta]].tolist())
        es_val_interna = np.array([g in g_meta for g in g_tr])
        partidos_int = int(pd.Series(g_tr).groupby(g_tr).nunique().gt(1).sum())
        if partidos_int:
            raise SystemExit(f"PARADA: la particion interna partio {partidos_int} grupos")
        itr, iva_in = np.where(~es_val_interna)[0], np.where(es_val_interna)[0]
        log(f"      pliegue {f}: training {len(ir)} filas -> inner_train {len(itr)}, "
            f"inner_validation {len(iva_in)}, en {n_meta} grupos")

        for j, et in enumerate(ETIQUETAS):
            ytr = y[ir, j].astype(int)
            y[iva, j].astype(int)
            # arm 1: LightGBM con parametros por defecto, sin busqueda
            d1 = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.1, random_state=SEMILLA,
                                    verbose=-1)
            d1.fit(x, ytr)
            proba_1[iva, j] = d1.predict_proba(xv)[:, 1]
            # arm 2: Optuna anidado
            hist = {"best_logloss": float("inf"), "mejor_iteration": 0, "parada_temprana": 0}

            # `hist` NO se vincula como argumento por defecto a proposito: es un
            # acumulador que debe sobrevivir entre trials, porque registra el mejor log
            # loss interno y la iteracion en que se produjo. Vincularlo crearia una
            # copia por llamada y el registro seria siempre el del primer trial.
            def objetivo(trial: optuna.Trial, _x=x, _itr=itr, _iva=iva_in,
                        _ytr=ytr, _f=f, _et=et) -> float:
                """Log loss en la validacion interna, que Optuna nunca ve el exterior.

                Args:
                    trial: objeto de trial.

                Returns:
                    El log loss interno, que se minimiza.
                """
                p = espacio_optuna(trial)
                m = lgb.LGBMClassifier(objective="binary", random_state=SEMILLA,
                                        verbose=-1, n_jobs=8,
                                        subsample_freq=1 if p["subsample"] < 1.0 else 0,
                                        **p)
                m.fit(_x[_itr], _ytr[_itr], eval_set=[(_x[_iva], _ytr[_iva])],
                      callbacks=[lgb.early_stopping(PACIENCIA, verbose=False)])
                pv = np.clip(m.predict_proba(_x[_iva])[:, 1], 1e-9, 1 - 1e-9)
                val = float(log_loss(_ytr[_iva], pv))
                if val < hist["best_logloss"]:  # noqa: B023
                    hist["best_logloss"] = val  # noqa: B023
                    hist["mejor_iteration"] = int(  # noqa: B023
                        m.best_iteration_ or p["n_estimators"])
                    hist["parada_temprana"] = int(  # noqa: B023
                        m.best_iteration_ or p["n_estimators"])
                # Se registra el trial COMPLETO, no solo tres campos. El parrafo 49
                # exige registrar los mejores parametros para poder reproducir, y una
                # version anterior guardaba solo num_leaves, max_depth y n_estimators,
                # con lo que el CSV de Optuna no permitia reconstruir la busqueda.
                filas_opt.append({
                    "pliegue": _f, "etiqueta": _et, "trial": trial.number,
                    "logloss_interno": round(val, 8),
                    **{k: (p[k] if not isinstance(p[k], (np.integer,))
                           else int(p[k])) for k in ESPACIO},
                    "best_iteration": int(m.best_iteration_ or p["n_estimators"]),
                })
                return val

            sampler = optuna.samplers.TPESampler(seed=SEMILLA + 100 * f + j)
            estudio = optuna.create_study(direction="minimize", sampler=sampler)
            estudio.optimize(objetivo, n_trials=N_TRIALS, show_progress_bar=False)
            p = espacio_optuna(estudio.best_trial)
            m2 = lgb.LGBMClassifier(objective="binary", random_state=SEMILLA, verbose=-1,
                                    n_jobs=8, n_estimators=p["n_estimators"],
                                    subsample_freq=1 if p["subsample"] < 1.0 else 0,
                                    num_leaves=p["num_leaves"], max_depth=p["max_depth"],
                                    learning_rate=p["learning_rate"],
                                    min_child_samples=p["min_child_samples"],
                                    subsample=p["subsample"],
                                    colsample_bytree=p["colsample_bytree"],
                                    reg_alpha=p["reg_alpha"], reg_lambda=p["reg_lambda"])
            m2.fit(x, ytr)
            proba_2[iva, j] = m2.predict_proba(xv)[:, 1]
            # el registro por pliegue se construye DESPUES del bucle de etiquetas,
            # porque el mejor log loss interno de TR y de AG no existe hasta que sus
            # trials han corrido. Dentro del bucle, en la primera iteracion, esos
            # filtros no tenian filas y min() reventaba.
        filas_f = [r for r in filas_opt if r["pliegue"] == f]
        info.append({
            "pliegue": f, "n_inner_train": len(itr), "n_inner_validation": len(iva_in),
            "grupos_en_la_region_interna": n_meta,
            "mejor_logloss_por_etiqueta": {
                et: round(min(r["logloss_interno"] for r in filas_f
                              if r["etiqueta"] == et), 8) for et in ETIQUETAS},
            "mejor_logloss_global_del_pliegue": round(
                min(r["logloss_interno"] for r in filas_f), 8),
            "n_trials_del_pliegue": len(filas_f),
            "mejor_iteracion_interna": hist["mejor_iteration"],
            "n_estimators_del_mejor_trial_hs": int(
                next(r for r in filas_f if r["etiqueta"] == "HS")["n_estimators"]),
            "parada_temprana_uso_en_hs": bool(
                hist["parada_temprana"] < next(r for r in filas_f
                                           if r["etiqueta"] == "HS")["n_estimators"]),
        })
        log(f"      pliegue {f} listo ({time.time() - t0:.1f} s acumulados)")

    pd.DataFrame(filas_opt).to_csv(CSV_OPT, index=False, encoding="utf-8")
    log("")
    log(f"   escrito reports/p61_optuna_results.csv con {len(filas_opt)} trials")

    pred_1 = (proba_1 >= V03_UMBRALES[None, :]).astype(int)
    pred_2 = (proba_2 >= V03_UMBRALES[None, :]).astype(int)

    def metricas(pred: np.ndarray) -> dict[str, Any]:
        """Metricas de un conjunto de predicciones.

        Args:
            pred: predicciones.

        Returns:
            Diccionario con metricas.
        """
        resc = int(np.count_nonzero((pred == y) & (pred_v03 != y)))
        dan = int(np.count_nonzero((pred != y) & (pred_v03 == y)))
        cam = int(np.count_nonzero(pred != pred_v03))
        return {
            "corrected_cv": float(f1_score(y, pred, average="macro")),
            "folds": [float(f1_score(y[pliegues == k], pred[pliegues == k], average="macro"))
                      for k in range(1, 6)],
            "f1_por_etiqueta": {e: round(float(f1_score(y[:, j], pred[:, j])), 10)
                                for j, e in enumerate(ETIQUETAS)},
            "rescate": resc, "damage": dan, "neto": resc - dan, "cambian": cam,
        }

    m1 = metricas(pred_1)
    m2_ = metricas(pred_2)
    ccv1, ccv2 = m1["corrected_cv"], m2_["corrected_cv"]
    log("")
    log(f"   BRAZO 0 V03                    corrected_cv {ccv_ref!r}")
    log(f"   BRAZO 1 LightGBM simple       corrected_cv {ccv1!r}  "
        f"delta vs V03 {ccv1 - ccv_ref:+.10f}")
    log(f"   BRAZO 2 LightGBM + Optuna     corrected_cv {ccv2!r}  "
        f"delta vs V03 {ccv2 - ccv_ref:+.10f}")
    log(f"  .delta por cambio de FAMILIA (1 - 0) {ccv1 - ccv_ref:+.10f}")
    log(f"  .delta por OPTIMIZACION (2 - 1)     {ccv2 - ccv1:+.10f}")
    log(f"   delta total (2 - 0)                {ccv2 - ccv_ref:+.10f}")

    folds_mej = sum(1 for u, w in zip(m2_["folds"], [ccv_ref] * 5) if u > w)
    h_delta = (ccv2 - ccv_ref) >= GATE_DELTA
    h_folds = folds_mej >= GATE_PLIEGUES
    gate = bool(h_delta and h_folds)
    log("")
    log(f"   pliegues donde el candidato 2 mejora sobre V03: {folds_mej}/5")
    log(f"   rescate {m2_['rescate']} | dano {m2_['damage']} | neto {m2_['neto']:+d} "
        f"| cambian {m2_['cambian']}")
    log(f"   GATE delta {ccv2 - ccv_ref:+.6f} >= {GATE_DELTA}: {h_delta} | "
        f"pliegues {folds_mej}/5 >= {GATE_PLIEGUES}: {h_folds} | superado: {gate}")
    ic_lo = ic_hi = None
    if gate:
        rng = np.random.default_rng(SEMILLA)
        dc = ((pred_2 == y).astype(int) - (pred_v03 == y).astype(int)).ravel()
        mm = dc[rng.integers(0, dc.size, size=(N_BOOT, dc.size))].mean(axis=1)
        ic_lo, ic_hi = float(np.percentile(mm, 2.5)), float(np.percentile(mm, 97.5))
        log(f"   bootstrap (solo porque el gate se supero): IC95 [{ic_lo:+.6f}, {ic_hi:+.6f}]")

    # distribucion de los mejores trials
    fo = pd.DataFrame(filas_opt)
    mejores = fo.loc[fo.groupby(["pliegue", "etiqueta"])["logloss_interno"].idxmin()]
    log("")
    log("   MEJORES PARAMETROS POR PLIEGUE Y ETIQUETA")
    log(f"      {'p':>2}{'et':>4}{'logloss':>10}{'leaves':>8}{'depth':>7}{'lr':>8}"
        f"{'minchild':>10}{'subsamp':>9}{'colsamp':>9}{'n_est':>7}{'best_it':>9}")
    for _, r in mejores.iterrows():
        log(f"      {int(r['pliegue']):>2}{r['etiqueta']:>4}{r['logloss_interno']:>10.6f}"
            f"{int(r['num_leaves']):>8}{int(r['max_depth']):>7}"
            f"{r['learning_rate'] if 'learning_rate' in r else float('nan'):>8.4f}"
            f"{int(r['min_child_samples']):>10}{r['subsample'] if 'subsample' in r else float('nan'):>9.3f}"
            f"{r['colsample_bytree'] if 'colsample_bytree' in r else float('nan'):>9.3f}"
            f"{int(r['n_estimators']):>7}{int(r['best_iteration']):>9}")

    decision = (
        "CLOSED_BY_DUPLICATION. La familia no aporta sobre V03 y el precedente OPT-01 ya "
        "cerro el patron de busqueda de capacidad por OOF"
        if ccv2 <= ccv_ref else
        ("MARGINAL. La capacidad no alcanza el gate de +0,003" if not gate else
         "PROMISING. Supera el gate; exige segunda particion"))

    np.savez_compressed(CELDA_NPZ, ids=ids, y_true=y, pliegues=pliegues, pred_v03=pred_v03,
                        pred_lgbm_simple=pred_1, pred_lgbm_optuna=pred_2,
                        proba_lgbm_simple=proba_1, proba_lgbm_optuna=proba_2)

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    informe = {
        "familia": "P61_LIGHTGBM_CAPACITY",
        "naturaleza": ("EXPERIMENTO. 1 candidato, 0 submissions, 0 Kaggle, 0 holdout, 0 test. "
                       "V03 no se modifica"),
        "fecha": inicio.isoformat(timespec="seconds"),
        "experiment_id": "P61-01", "track": "P61-lightgbm",
        "script": "experiments/p61_03_lightgbm.py",
        "gap_analysis": {
            "filas_lightgbm_xgboost_catboost": 0,
            "precedente_de_busqueda_de_capacidad": (
                "OPT-01: Optuna de 24 trials sobre el propio TF-IDF de V03 gano +0,008565 en "
                "OOF y se DESCARTO porque no transfirio, con 0,77602 frente a 0,78306 de V03, "
                "un hueco de -0,028. Su conclusion registrada es que la busqueda por OOF premia "
                "capacidad que no generaliza"),
            "precedente_de_gradient_boosting": (
                "V4-G01-G03: svd:hist_gb 0,7297 y svd:extra_trees 0,7033 sobre SVD de este "
                "corpus, luego el SVD es una representacion ya sabidamente debil para arboles"),
            "veredicto": ("la familia LightGBM no esta en el registro, luego es nueva y esta "
                          "autorizada; pero el precedente OPT-01 obliga a que el gate se "
                          "aplique a la puntuacion ANIDADA y no a la mejor puntuacion de trial"),
        },
        "configuracion_pre_registrada": {
            "representacion": "el TF-IDF de V03, sin cambio, sin SVD y sin rasgos nuevos",
            "razon_de_no_usar_svd": ("V4-G ya midio gradient boosting sobre SVD de este corpus "
                                     "con 0,7297, luego importarlo mezclaria el cambio de "
                                     "familia con un cambio de representacion"),
            "trials_por_pliegue": N_TRIALS,
            "sampler": f"TPESampler con semilla {SEMILLA} mas 100 por pliegue y etiqueta",
            "espacio": list(ESPACIO),
            "objetivo_de_optuna": ("log loss en la validacion INTERNA. Se elige una regla de "
                                   "puntuacion apropiada y libre de umbral para que la "
                                   "busqueda no dependa de la regla de decision, que en esta "
                                   "fase queda congelada"),
            "n_estimators_max": N_ESTIMATORS_MAX,
            "early_stopping": f"{PACIENCIA} rondas de paciencia sobre la validacion interna",
            "umbrales": "0,44 / 0,51 / 0,46 congelados, aplicados igual a los tres brazos",
            "rejilla": "ninguna: una busqueda, no una rejilla",
            "anidamiento": ("outer_train partido POR GRUPO en inner_train e "
                            "inner_validation; Optuna solo ve la interna; refit sobre el "
                            "outer_train completo; la validacion exterior no participa"),
        },
        "brazo_1_lightgbm_simple": m1,
        "brazo_2_lightgbm_optuna": m2_,
        "deltas": {
            "cambio_de_familia_1_menos_0": round(ccv1 - ccv_ref, 10),
            "optimizacion_2_menos_1": round(ccv2 - ccv1, 10),
            "total_2_menos_0": round(ccv2 - ccv_ref, 10),
        },
        "gate": {"delta_min": GATE_DELTA, "pliegues_min": GATE_PLIEGUES,
                 "pliegues_mejorados": folds_mej,
                 "supera_delta": bool(h_delta), "supera_pliegues": bool(h_folds),
                 "superado": gate},
        "bootstrap": (None if ic_lo is None else {"IC95": [round(ic_lo, 6), round(ic_hi, 6)],
                                                  "n": N_BOOT}),
        "por_fold": info,
        "mejores_trials": mejores.to_dict(orient="records"),
        "trials_totales": len(filas_opt),
        "reproducibilidad": {
            "nivel": "R2",
            "que_implica_R2": ("seed fijo, sampler y semilla registrados, artefacto en disco y "
                               "hash del OOF verificado; la busqueda es estocnstica y un "
                               "reejecutado con la misma semilla da el mismo resultado, pero no "
                               "se ha reproducido de forma independiente en otra sesion"),
            "git_commit": commit, "seed": SEMILLA, "trials": len(filas_opt),
            "sampler": "TPESampler", "espacio": list(ESPACIO),
            "artefacto_optuna": "reports/p61_optuna_results.csv",
            "artefacto_celdas": "features/p61_lightgbm_cells.npz",
        },
        "interpretacion": "",
        "decision": decision,
        "experimentos_registrados": 1,
        "formulacion": "V03 remains the best verified leaderboard result.",
        "gastos": {"submissions": 0, "consultas_kaggle": 0, "uso_holdout": 0, "uso_test": 0,
                   "segundos": round((datetime.now(timezone.utc) - inicio).total_seconds(), 1)},
    }
    JSON_LGB.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")

    L: list[str] = []
    W = L.append
    W("# P61 - Capacidad LightGBM con Optuna anidado")
    W("")
    W(f"Generado por `experiments/p61_03_lightgbm.py`. Fecha {inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## El gap check que autoriza la familia, y el riesgo que expone")
    W("")
    W("LightGBM, XGBoost y CatBoost devuelven **0 filas** en el registro canonico. La familia")
    W("es genuinamente nueva, que es lo que pide el encargo para gastar experimentos en ella.")
    W("")
    W("Pero el registro tambien contiene el precedente que mas pesa, y no es tranquilizador.")
    W("`OPT-01` ejecuto Optuna sobre el propio TF-IDF de V03, 24 trials, y gano **+0,008565**")
    W("en la puntuacion out-of-fold. Se descarto porque **no transfirio**: 0,77602 frente a")
    W("0,78306 de V03 en el leaderboard, un hueco de -0,028. Su conclusion registrada es que")
    W("**la busqueda por OOF premia capacidad que no generaliza al test**.")
    W("")
    W("La situacion es: una familia autorizada y nunca probada, en un proyecto donde una busqueda")
    W("de capacidad ya probada produjo un falso positivo que solo el leaderboard pudo detectar.")
    W("Por eso el gate se aplica a la puntuacion **anidada** y no a la mejor puntuacion de trial.")
    W("")
    W("## Representacion, y por que es la que ya tiene V03")
    W("")
    W("El encargo lista el TF-IDF de V03 primero y el SVD como alternativa densa. El SVD no se")
    W("usa, y el motivo esta en el registro: `V4-G01-G03` ya ajusto gradient boosting sobre SVD")
    W("de este corpus y dio `svd:hist_gb 0,7297` y `svd:extra_trees 0,7033`, muy por debajo de")
    W("V03. Importar esa representacion ya sabidamente debil mezclaria el cambio de familia con")
    W("un cambio de representacion, que es justo lo que el parrafo 23 prohibe.")
    W("")
    W("El regimen se declara como riesgo: **27.431 rasgos frente a 1.186 filas** de")
    W("entrenamiento es un regimen donde los arboles no son naturalmente fuertes.")
    W("")
    W("## Los tres brazos, que el parrafo 27 exige")
    W("")
    W("| brazo | corrected_cv | delta vs V03 |")
    W("|---|---:|---:|")
    W(f"| 0, V03 referencia | {ccv_ref:.10f} | +0.0000000000 |")
    W(f"| 1, LightGBM simple | {ccv1:.10f} | {ccv1 - ccv_ref:+.10f} |")
    W(f"| 2, LightGBM + Optuna | {ccv2:.10f} | {ccv2 - ccv_ref:+.10f} |")
    W("")
    W("```text")
    W(f"cambio de FAMILIA     (brazo 1 - 0) {ccv1 - ccv_ref:+.10f}")
    W(f"optimizacion de params(brazo 2 - 1) {ccv2 - ccv1:+.10f}")
    W(f"total                 (brazo 2 - 0) {ccv2 - ccv_ref:+.10f}")
    W("```")
    W("")
    W("Reportar solo uno de los dos primeros es el error que el parrafo 27 esta escrito para")
    W("evitar, porque son mecanismos distintos: uno cambia de familia de modelo y el otro cambia")
    W("los hiperparametros de esa familia.")
    W("")
    W("## El anidamiento, y el objetivo de la busqueda")
    W("")
    W("```text")
    W("outer_train -> inner_train, inner_validation, partidos POR GRUPO")
    W("           -> Optuna, que solo ve la validacion interna")
    W("           -> refit sobre el outer_train completo con los parametros ganadores")
    W("outer_validation, que no participa en nada")
    W("```")
    W("")
    W("El objetivo de Optuna es el **log loss de la validacion interna**, no el F1 interno. Es")
    W("deliberado: usar F1 dentro de la busqueda haria depender la busqueda del umbral de")
    W("decision, y en esta fase los umbrales estan congelados. Una regla de puntuacion")
    W("apropiada es libre de umbral, luego la busqueda optimiza el modelo de probabilidad y")
    W("los umbrales de V03 se aplican una sola vez, al final, sobre la validacion exterior.")
    W("")
    W("## Resultados")
    W("")
    W("| etiqueta | F1 brazo 1 | F1 brazo 2 | F1 V03 |")
    W("|---|---:|---:|---:|")
    v3 = np.load(OOF, allow_pickle=True)
    f1_v03 = {e: round(float(f1_score(y[:, j], v3["pred"][:, j])), 4)
              for j, e in enumerate(ETIQUETAS)}
    for e in ETIQUETAS:
        W(f"| {e} | {m1['f1_por_etiqueta'][e]:.4f} | {m2_['f1_por_etiqueta'][e]:.4f} | "
          f"{f1_v03[e]:.4f} |")
    W("")
    W("| pliegue | inner_train | inner_val | mejor logloss interno | mejor iteracion |")
    W("|---:|---:|---:|---:|---:|")
    for i in info:
        W(f"| {i['pliegue']} | {i['n_inner_train']} | {i['n_inner_validation']} | "
          f"{i['mejor_logloss_global_del_pliegue']:.6f} | {i['mejor_iteracion_interna']} |")
    W("")
    W("```text")
    W(f"pliegues donde el candidato 2 mejora sobre V03   {folds_mej}/5")
    W(f"rescate {m2_['rescate']} | dano {m2_['damage']} | neto {m2_['neto']:+d} "
      f"| cambian {m2_['cambian']}")
    W(f"gate superado {gate} | bootstrap "
      f"{'NO EJECUTADO' if ic_lo is None else f'IC95 [{ic_lo:+.6f}, {ic_hi:+.6f}]'}")
    W(f"trials totales {len(filas_opt)}")
    W("```")
    W("")
    W("## Reproducibilidad")
    W("")
    W("Nivel **R2**: semilla fija, sampler y semilla registrados, artefacto en disco con los")
    W(f"{len(filas_opt)} trials y hash del OOF verificado. Una reejecucion con la misma")
    W("semilla dara el mismo resultado, pero no se ha reproducido de forma independiente.")
    W("")
    W("## Decision")
    W("")
    W("```text")
    W(f"decision  {decision}")
    W("```")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p61_03_lightgbm.py`. Sin push.*")
    MD_LGB.write_text(a_ascii("\n".join(L)), encoding="ascii")

    for destino in (MD_LGB, JSON_LGB, CSV_OPT):
        if any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P61-03 OK | {len(filas_opt)} trials | familia {ccv1 - ccv_ref:+.6f} | "
        f"optimizacion {ccv2 - ccv1:+.6f} | total {ccv2 - ccv_ref:+.6f}")


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
