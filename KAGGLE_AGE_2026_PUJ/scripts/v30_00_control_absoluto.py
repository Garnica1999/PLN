"""V30 - 00 - P0: control absoluto. Reproducir V20 y los dos resultados de V29.

V30 empieza exigiendose a si mismo lo mismo que exigio a las fases anteriores: que el punto de
partida sea exactamente el que dice ser. Si V20 no reproduce bit a bit, o si el LightGBM de V29 o
el ensemble de V29 no reproducen, V30 no empieza.

## Los tres controles y por que son distintos en fuerza

| Control | Que comprueba | Fuerza |
|---|---|---|
| V20 | se reconstruye desde los artefactos y se exige el OOF **identico bit a bit** | la mas fuerte: dos reconstrucciones pueden dar el mismo score con probabilidades distintas |
| LightGBM de V29 | el OOF almacenado reproduce el `corrected_cv` del artefacto, exactamente | fuerte: es un fichero, no una reentrenamiento |
| Ensemble de V29 | se **re-optimizan** los pesos con la misma semilla y se exige Cain dentro de tolerancia | mas debil, y se declara: Optuna con `TPESampler` y semilla fija es reproducible **en la misma version de las librerias**, no se declara exactitud bit a bit entre reproducciones |

## Lo que este script NO hace

No entrena nada nuevo, no toca el leaderboard, no reescribe ningun artefacto de V20 ni de V29.
Solo comprueba y deja por escrito si el punto de partida es el que se-annunciaa.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
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

RESULTADO = cfg.EXPERIMENTS / "v30_00_control_absoluto.json"
CV_V20 = 0.6097782
CV_LGBM_V29 = 0.6121698
CV_ENS_V29 = 0.6125960
HASH_V20 = "2c964af78c90bf0884b4107665b5100f0c1aa2d9059c47692e613578b6cf660d"
DATA_HASH = "32393abca28db4c9"
#: Tolerancia del ensemble: Optuna es un metodo de busqueda estocastica y el propio brief dice que
#: no se declare exactitud bit a bit entre reproducciones. Se admite una desviacion pequena, y
#: **el valor exacto de la desviacion se registra**: si fuera grande, seria una senal de alarma.
TOLERANCIA_ENSEMBLE = 0.002


def _macro(clase: np.ndarray, prob: np.ndarray) -> float:
    return float(f1_score(clase, np.argmax(prob, axis=1), average="macro",
                          labels=list(range(len(cfg.CLASSES)))))


def main() -> int:
    """Ejecuta los tres controles y detiene V30 si alguno falla."""
    autotest_numerico()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(v) for v in y])
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")

    # ---------------------------------------------------------------- control 1: V20
    matriz = matriz_279()
    logreg = np.load(cfg.FEATURES / "v19_b1_oof_logreg.npy")
    mlp = np.load(cfg.FEATURES / "v19_b1_oof_mlp.npy")
    arbol = np.load(cfg.FEATURES / "v19_oof_arbol_5folds.npy")
    acumuladas = np.load(cfg.FEATURES / "v20_f3_oof_cumuladas.npy")
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
    seis /= np.clip(seis.sum(axis=1, keepdims=True), 1e-9, None)
    oof_v20 = 0.25 * logreg + 0.25 * mlp + 0.25 * arbol + 0.25 * seis
    almacenado = np.load(cfg.FEATURES / "v20_f3b_oof_candidato.npy")
    dif = float(np.abs(almacenado - oof_v20).max())
    control_v20 = {
        "corrected_cv_recomputado": round(_macro(clase, oof_v20), 7),
        "canonico": CV_V20,
        "reproduce_macro_f1": abs(_macro(clase, oof_v20) - CV_V20) < 1e-6,
        "max_abs_diferencia_vs_artefacto": dif,
        "reproduce_bit_a_bit": dif == 0.0,
        "folds": [int(len(f["val_idx"])) for f in folds],
        "folds_esperados": [33868, 33868, 33867, 33866, 33867],
        "n_features": int(matriz.shape[1]),
    }
    control_v20["folds_identicos"] = control_v20["folds"] == control_v20["folds_esperados"]
    print("=== control 1: V20 ===", flush=True)
    print(f"  corrected_cv {control_v20['corrected_cv_recomputado']:.7f} | canonico {CV_V20} | "
          f"reproduce: {control_v20['reproduce_macro_f1']}", flush=True)
    print(f"  OOF bit a bit identico: {control_v20['reproduce_bit_a_bit']} "
          f"(max dif {dif:.2e})", flush=True)
    print(f"  folds identicos: {control_v20['folds_identicos']} {control_v20['folds']}", flush=True)

    # ---------------------------------------------------------------- control 2: LightGBM de V29
    ruta_lgbm = cfg.FEATURES / "v29_lgbm_optuna_oof.npy"
    lgbm_arte = json.loads((cfg.EXPERIMENTS / "v29_04_optuna_lightgbm.json")
                           .read_text(encoding="utf-8"))
    oof_lgbm = np.load(ruta_lgbm)
    cv_lgbm = _macro(clase, oof_lgbm)
    control_lgbm = {
        "corrected_cv_del_oof_en_disco": round(cv_lgbm, 7),
        "corrected_cv_del_artefacto": lgbm_arte["corrected_cv"],
        "esperado": CV_LGBM_V29,
        "coincide_con_artefacto": abs(cv_lgbm - lgbm_arte["corrected_cv"]) < 1e-6,
        "coincide_con_esperado": abs(cv_lgbm - CV_LGBM_V29) < 1e-6,
        "delta_vs_V20": round(cv_lgbm - CV_V20, 7),
        "n_trials_por_fold": lgbm_arte["n_trials_por_fold"],
        "n_trials_totales": lgbm_arte["n_trials_totales"],
        "iteraciones_por_fold": [f["diagnostico_refit"]["iteraciones"] for f in lgbm_arte["folds"]],
        "best_trial_por_fold": [f["mejor_trial"] for f in lgbm_arte["folds"]],
    }
    print("\n=== control 2: LightGBM Optuna de V29 ===", flush=True)
    print(f"  corrected_cv {control_lgbm['corrected_cv_del_oof_en_disco']:.7f} | esperado "
          f"{CV_LGBM_V29} | coincide: {control_lgbm['coincide_con_esperado']}", flush=True)
    print(f"  delta frente a V20: {control_lgbm['delta_vs_V20']:+.7f}", flush=True)
    print(f"  mejor trial por fold: {control_lgbm['best_trial_por_fold']} | iteraciones por el "
          f"early stopping: {control_lgbm['iteraciones_por_fold']}", flush=True)

    # ---------------------------------------------------------------- control 3: ensemble de V29
    # Re-optimizar los pesos es lo mas cerca de "reproducir" que se puede estar de una busqueda de
    # Optuna. Se importa la funcion tal cual se ejecuto, con la misma semilla, y se mide.
    from v29_06_ensemble import LAMBDA, _optuna_pesos  # noqa: PLC0415
    datos_pool = np.load(cfg.EXPERIMENTS / "v29_pool_oof.npz")
    miembros = list(datos_pool.files)
    matrices = np.stack([datos_pool[m].astype(np.float64) for m in miembros])
    print("\n=== control 3: re-optimizando los pesos del ensemble de V29 ===", flush=True)
    print(f"  pool de {len(miembros)} miembros, objetivo macro-F1 menos lambda por entropia "
          f"(lambda={LAMBDA})", flush=True)
    reopt = _optuna_pesos(matrices, clase, folds, miembros, objetivo=2, lam=LAMBDA)
    cv_reopt = _macro(clase, reopt["oof"])
    desviacion = abs(cv_reopt - CV_ENS_V29)
    control_ens = {
        "corrected_cv_reoptimizado": round(cv_reopt, 7),
        "corrected_cv_de_V29": CV_ENS_V29,
        "desviacion": round(desviacion, 6),
        "tolerancia": TOLERANCIA_ENSEMBLE,
        "reproduce": desviacion <= TOLERANCIA_ENSEMBLE,
        "delta_vs_V20": round(cv_reopt - CV_V20, 7),
        "folds_por_fold": [f["macro_f1_outer_val"] for f in reopt["detalle"].values()],
        "folds_positivos": sum(1 for f in reopt["detalle"].values()
                               if f["macro_f1_outer_val"] > _macro(
                                   clase[np.asarray(next(o for o in folds if o['fold'] == int(k))
                                                    ['val_idx'])],
                                   oof_v20[np.asarray(next(o for o in folds
                                                            if o['fold'] == int(k))['val_idx'])])),
        "n_miembros_pool": len(miembros),
        "declaracion": "Optuna es un metodo de busqueda estocastica. Con TPESampler y semilla fija "
                       "el resultado es reproducible en la misma version de las librerias, pero no "
                       "se declara exactitud bit a bit entre reproducciones. La desviacion medida "
                       "se registra al lado del resultado.",
    }
    print(f"  re-optimizado {cv_reopt:.7f} | V29 {CV_ENS_V29} | desviacion "
          f"{desviacion:.6f} | tolerancia {TOLERANCIA_ENSEMBLE} | reproduce: "
          f"{control_ens['reproduce']}", flush=True)

    # ---------------------------------------------------------------- veredicto
    todo = (control_v20["reproduce_macro_f1"] and control_v20["reproduce_bit_a_bit"]
            and control_v20["folds_identicos"] and control_lgbm["coincide_con_esperado"]
            and control_ens["reproduce"])
    RESULTADO.write_text(json.dumps({
        "fase": "V30 00 P0 control absoluto", "fecha": "2026-09-30", "experimenta": False,
        "control_v20": control_v20, "control_lightgbm_v29": control_lgbm,
        "control_ensemble_v29": control_ens,
        "v30_autorizado": bool(todo),
        "cv_v20": CV_V20, "hash_submission_v20": HASH_V20, "data_hash": DATA_HASH,
        "artefactos_de_v20_tocados": [],
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    if not todo:
        print("\n[STOP] Algun control falla. V30 no debe empezar.", flush=True)
        return 1
    print("\n[P0] APROBADO. V30 puede empezar.", flush=True)
    print(f"[P0] {time.time() - inicio:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
