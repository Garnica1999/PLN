"""V19/V20 (cuatro cuartos), ensemble V30 y regla plug-in (porte fiel de v19_b1/v20_f3/v30_03/age_plugin).

Configuraciones congeladas embebidas como constantes del protocolo (no se leen de ficheros):
- Pesos del ensemble V30: media de los vectores Optuna por fold, renormalizada (v30_02_ensemble.json).
- Configs V29 (reports/v29-01_trials.csv) y V30 (v30_01_optuna_refinamiento.json), en orden de fold.
- Pesos plug-in congelados (age_umbrales_validacion.json, commit bd08aba).
ASCII puro. Sin efectos en import.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np

from cadena import comun
from cadena.meta import EncodedMLP, logreg

#: Arbol pre-registrado del proyecto (scripts/v19_b1_boosting.py:83-86).
HIERBA = {"objective": "multiclass", "num_class": len(comun.CLASSES), "n_estimators": 300,
          "learning_rate": 0.05, "num_leaves": 8, "max_depth": 3, "min_child_samples": 200,
          "colsample_bytree": 0.5, "subsample": 0.8, "subsample_freq": 1, "reg_lambda": 1.0,
          "n_jobs": 32, "random_state": comun.SEED, "verbose": -1}

#: Pesos del ensemble V30 (v30_02_ensemble.json -> manifest): v20, v29, v30.
PESOS_ENSEMBLE = (0.3548, 0.3354, 0.3098)

#: Pesos plug-in congelados (age_plugin_submission.py:79).
PESOS_PLUG_IN = np.array([1.0000, 1.1749, 1.2721, 1.1630, 1.0010, 0.8128])

#: Mejor trial por fold de V29 (reports/v29-01_trials.csv), en orden de fold.
V29_CONFIGS = [
    {"num_leaves": 87, "max_depth": 5, "learning_rate": 0.04301669500288421,
     "min_child_samples": 76, "subsample": 0.6581570543882859,
     "colsample_bytree": 0.70766450002428, "reg_alpha": 0.6147385859030499,
     "reg_lambda": 9.792145504680492e-05, "n_estimators": 816},
    {"num_leaves": 55, "max_depth": 6, "learning_rate": 0.03352698655336303,
     "min_child_samples": 54, "subsample": 0.8941120537508466,
     "colsample_bytree": 0.6299160105578635, "reg_alpha": 4.026902217415037e-07,
     "reg_lambda": 6.6850377528309e-07, "n_estimators": 1372},
    {"num_leaves": 57, "max_depth": 8, "learning_rate": 0.022124342973521553,
     "min_child_samples": 94, "subsample": 0.9225205673898446,
     "colsample_bytree": 0.52058407816027, "reg_alpha": 1.8772169842126896,
     "reg_lambda": 0.018194045769254683, "n_estimators": 912},
    {"num_leaves": 43, "max_depth": 9, "learning_rate": 0.03000286836410436,
     "min_child_samples": 22, "subsample": 0.6998254389554196,
     "colsample_bytree": 0.5250455092128341, "reg_alpha": 1.0090819656249398e-05,
     "reg_lambda": 7.87982213180386e-06, "n_estimators": 863},
    {"num_leaves": 56, "max_depth": 7, "learning_rate": 0.020811077189914432,
     "min_child_samples": 29, "subsample": 0.7762118181212365,
     "colsample_bytree": 0.6242271327287489, "reg_alpha": 1.3833851148235308e-05,
     "reg_lambda": 0.0004885693686069363, "n_estimators": 1084},
]

#: Mejor trial por fold de V30 (v30_01_optuna_refinamiento.json), en orden de fold.
V30_CONFIGS = [
    {"num_leaves": 41, "max_depth": 10, "learning_rate": 0.015612823179143214,
     "min_child_samples": 79, "subsample": 0.775913545060191,
     "colsample_bytree": 0.5969526009433974, "reg_alpha": 1.0941449983677585e-05,
     "reg_lambda": 0.00036896192791107643, "n_estimators": 1011},
    {"num_leaves": 32, "max_depth": 7, "learning_rate": 0.017387326275059966,
     "min_child_samples": 23, "subsample": 0.8215355497227395,
     "colsample_bytree": 0.8291548707315466, "reg_alpha": 0.8747434883808647,
     "reg_lambda": 2.053077365511584e-05, "n_estimators": 1486},
    {"num_leaves": 63, "max_depth": 7, "learning_rate": 0.02867292541012883,
     "min_child_samples": 110, "subsample": 0.7807106452264457,
     "colsample_bytree": 0.4021908926823267, "reg_alpha": 1.1486360263326448e-05,
     "reg_lambda": 0.00031823134119350724, "n_estimators": 1023},
    {"num_leaves": 42, "max_depth": 7, "learning_rate": 0.015729268043126833,
     "min_child_samples": 41, "subsample": 0.999685646683389,
     "colsample_bytree": 0.596195315949217, "reg_alpha": 0.004978399116554409,
     "reg_lambda": 1.8436108646384413e-08, "n_estimators": 1477},
    {"num_leaves": 64, "max_depth": 6, "learning_rate": 0.03818854692388361,
     "min_child_samples": 52, "subsample": 0.8731483359860639,
     "colsample_bytree": 0.46934472778342573, "reg_alpha": 1.927060050589197e-08,
     "reg_lambda": 0.004339321331656464, "n_estimators": 1000},
]

PARAMETROS_ENTEROS = ("num_leaves", "max_depth", "min_child_samples", "subsample_freq",
                      "n_estimators")


def a_clases(probabilidad: np.ndarray, clases, filas: int) -> np.ndarray:
    """Reordena columnas al orden canonico (v30_03_submission.py:72-90)."""
    salida = np.zeros((filas, len(comun.CLASSES)))
    existentes = list(clases)
    son_enteros = all(isinstance(v, (int, np.integer)) for v in existentes)
    for destino, etiqueta in enumerate(comun.CLASSES):
        buscar = destino if son_enteros else etiqueta
        salida[:, destino] = probabilidad[:, existentes.index(buscar)]
    return salida


def _normaliza(params: dict) -> dict:
    """Convierte a int los parametros enteros de LightGBM (v30_03_submission.py:139-146)."""
    salida = dict(params)
    for clave in PARAMETROS_ENTEROS:
        if clave in salida and salida[clave] is not None:
            salida[clave] = int(round(float(salida[clave])))
    return salida


def _arbol(semilla: int = comun.SEED) -> lgb.LGBMClassifier:
    """Arbol HIERBA con semilla (scripts/v19_b1_boosting.py:110-115)."""
    return lgb.LGBMClassifier(**{**HIERBA, "random_state": semilla})


def v19_oofs(matriz: np.ndarray, y: np.ndarray, clase: np.ndarray, fold_idx: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OOF de los tres miembros de V19 (logreg, mlp, arbol) sobre la matriz 279."""
    n = len(y)
    p_lr = np.zeros((n, 6))
    p_mlp = np.zeros((n, 6))
    p_arbol = np.zeros((n, 6))
    for fold in range(5):
        tr = fold_idx != fold
        va = fold_idx == fold
        lr = logreg().fit(matriz[tr], y[tr])
        completa = np.zeros((int(va.sum()), 6))
        for posicion, nombre in enumerate(lr.classes_):
            completa[:, comun.CLASSES.index(str(nombre))] = lr.predict_proba(matriz[va])[:, posicion]
        p_lr[va] = completa
        red = EncodedMLP().fit(matriz[tr], y[tr])
        p_mlp[va] = red.predict_proba(matriz[va])
        modelo = _arbol(comun.SEED).fit(matriz[tr], y[tr])
        p_arbol[va] = a_clases(modelo.predict_proba(matriz[va]), modelo.classes_, int(va.sum()))
    return p_lr, p_mlp, p_arbol


def v20_cadena_oof(matriz: np.ndarray, clase: np.ndarray, fold_idx: np.ndarray) -> np.ndarray:
    """Cadena de 5 umbrales anidados, OOF (scripts/v20_f3_umbrales.py:145-172 + 196-206)."""
    n = len(clase)
    n_clases = len(comun.CLASSES)
    acumuladas = np.zeros((n, n_clases - 1))
    for fold in range(5):
        tr = fold_idx != fold
        va = fold_idx == fold
        for k in range(1, n_clases):
            objetivo = (clase <= k - 1).astype(int)
            modelo = lgb.LGBMClassifier(**HIERBA)
            modelo.fit(matriz[tr], objetivo[tr])
            acumuladas[va, k - 1] = modelo.predict_proba(matriz[va])[:, 1]
    for k in range(1, n_clases - 1):
        acumuladas[:, k] = np.maximum(acumuladas[:, k], acumuladas[:, k - 1])
    seis = np.zeros((n, n_clases))
    seis[:, 0] = acumuladas[:, 0]
    for k in range(1, n_clases - 1):
        seis[:, k] = acumuladas[:, k] - acumuladas[:, k - 1]
    seis[:, n_clases - 1] = 1.0 - acumuladas[:, n_clases - 2]
    seis = np.clip(seis, 0.0, None)
    return seis / np.clip(seis.sum(axis=1, keepdims=True), 1e-9, None)


def v20_oof(matriz: np.ndarray, y: np.ndarray, clase: np.ndarray, fold_idx: np.ndarray
            ) -> np.ndarray:
    """OOF del candidato V20: 0.25 de cada cuarto (v20_f3_umbrales.py:207-213)."""
    p_lr, p_mlp, p_arbol = v19_oofs(matriz, y, clase, fold_idx)
    seis = v20_cadena_oof(matriz, clase, fold_idx)
    return 0.25 * p_lr + 0.25 * p_mlp + 0.25 * p_arbol + 0.25 * seis


def v20_test(matriz: np.ndarray, matriz_test: np.ndarray, y: np.ndarray, clase: np.ndarray
             ) -> np.ndarray:
    """Receta exacta de V20 sobre el train completo (v30_03_submission.py:93-117)."""
    n_test = matriz_test.shape[0]
    modelo_lr = logreg().fit(matriz, y)
    prob_lr = a_clases(modelo_lr.predict_proba(matriz_test), modelo_lr.classes_, n_test)
    red = EncodedMLP().fit(matriz, y)
    prob_mlp = a_clases(red.model.predict_proba(red.scaler.transform(matriz_test)),
                        red.encoder.classes_, n_test)
    modelo_arbol = lgb.LGBMClassifier(**HIERBA).fit(matriz, y)
    prob_arbol = a_clases(modelo_arbol.predict_proba(matriz_test), modelo_arbol.classes_, n_test)
    acumuladas = np.zeros((n_test, len(comun.CLASSES) - 1))
    for k in range(1, len(comun.CLASSES)):
        objetivo = (clase <= k - 1).astype(int)
        modelo = lgb.LGBMClassifier(**HIERBA).fit(matriz, objetivo)
        acumuladas[:, k - 1] = modelo.predict_proba(matriz_test)[:, 1]
    for k in range(1, len(comun.CLASSES) - 1):
        acumuladas[:, k] = np.maximum(acumuladas[:, k], acumuladas[:, k - 1])
    seis = np.zeros((n_test, len(comun.CLASSES)))
    seis[:, 0] = acumuladas[:, 0]
    for k in range(1, len(comun.CLASSES) - 1):
        seis[:, k] = acumuladas[:, k] - acumuladas[:, k - 1]
    seis[:, len(comun.CLASSES) - 1] = 1.0 - acumuladas[:, len(comun.CLASSES) - 2]
    seis = np.clip(seis, 0.0, None)
    seis = seis / np.clip(seis.sum(axis=1, keepdims=True), 1e-9, None)
    return 0.25 * prob_lr + 0.25 * prob_mlp + 0.25 * prob_arbol + 0.25 * seis


def lgbm_refit_oof(matriz: np.ndarray, clase: np.ndarray, fold_idx: np.ndarray,
                   configs: list[dict], base: dict | None = None) -> np.ndarray:
    """OOF de un LightGBM con la configuracion congelada por fold y early stopping del original.

    Porte de ``scripts/v29_04_optuna_individual.py:243-266`` (``_entrena``, rama lightgbm; la
    base incluye ``random_state=SEED``) y ``scripts/v30_01_optuna_refinamiento.py:164-176``
    (``_entrena``; la base NO incluye ``random_state``). El early stopping (60 rondas de
    ``multi_logloss`` sobre el fold retenido) es parte del protocolo original y el corte
    ocurrio de hecho en los reajustes documentados.

    Nota de idempotencia: se anaden ``deterministic=True`` y ``force_col_wise=True`` porque la
    construccion multihilo de histogramas de LightGBM no es estable entre procesos con la
    configuracion original (medido con un probe de dos procesos: el OOF reconstruido cambiaba);
    con ambos parametros dos corridas completas reproducen el mismo resultado bit a bit.
    """
    n = len(clase)
    oof = np.zeros((n, len(comun.CLASSES)))
    for fold in range(5):
        tr = fold_idx != fold
        va = fold_idx == fold
        params = {"objective": "multiclass", "num_class": len(comun.CLASSES), "verbose": -1,
                  "n_jobs": 32, "subsample_freq": 1, "deterministic": True,
                  "force_col_wise": True, **(base or {}),
                  **_normaliza(configs[fold])}
        techo = int(params.pop("n_estimators"))
        modelo = lgb.LGBMClassifier(**params, n_estimators=techo)
        modelo.fit(matriz[tr], clase[tr],
                   eval_set=[(matriz[va], clase[va])], eval_metric="multi_logloss",
                   callbacks=[lgb.early_stopping(60, verbose=False)])
        oof[va] = a_clases(modelo.predict_proba(matriz[va]), modelo.classes_, int(va.sum()))
    return oof


def lgbm_refit_test(matriz: np.ndarray, clase: np.ndarray, matriz_test: np.ndarray,
                    configs: list[dict]) -> np.ndarray:
    """Cinco reajustes (uno por configuracion congelada) sobre todo el train, promediados.

    Incluye ``deterministic=True`` y ``force_col_wise=True`` por la misma razon que
    ``lgbm_refit_oof`` (estabilidad entre procesos de la construccion de histogramas).
    """
    n_test = matriz_test.shape[0]
    predicciones = []
    for params in configs:
        p = {"objective": "multiclass", "num_class": len(comun.CLASSES), "verbose": -1,
             "n_jobs": 28, "subsample_freq": 1, "deterministic": True,
             "force_col_wise": True, **_normaliza(params)}
        techo = int(p.pop("n_estimators"))
        modelo = lgb.LGBMClassifier(**p, n_estimators=techo)
        modelo.fit(matriz, clase)
        predicciones.append(a_clases(modelo.predict_proba(matriz_test), modelo.classes_, n_test))
    return np.mean(predicciones, axis=0)


def mezcla(p20: np.ndarray, p29: np.ndarray, p30: np.ndarray) -> np.ndarray:
    """Mezcla del ensemble V30 con los pesos congelados (v30_03_submission.py:232-240)."""
    vector = np.array(PESOS_ENSEMBLE)
    vector = vector / vector.sum()
    return np.tensordot(vector, np.stack([p20, p29, p30]), axes=(0, 0))


def regla_plug_in(prob: np.ndarray) -> np.ndarray:
    """Decision plug-in por clase: argmax(prob * PESOS_PLUG_IN) (age_plugin_submission.py:258-261)."""
    return np.argmax(prob * PESOS_PLUG_IN, axis=1)


def macro_f1(clase: np.ndarray, pred: np.ndarray) -> float:
    """Macro-F1 sobre indices 0-5 (sin dependencia de sklearn para el chequeo rapido)."""
    from sklearn.metrics import f1_score
    return float(f1_score(clase, pred, average="macro", labels=list(range(len(comun.CLASSES)))))


def _cuartos_v20_fold(matriz: np.ndarray, clase: np.ndarray, y: np.ndarray,
                      tren: np.ndarray, val: np.ndarray) -> np.ndarray:
    """Los cuatro cuartos de V20 sobre `tren`, medidos en `val` (v30_05:146-175)."""
    prob_lr = a_clases(logreg().fit(matriz[tren], y[tren]).predict_proba(matriz[val]),
                       np.array(comun.CLASSES), len(val))
    red = EncodedMLP().fit(matriz[tren], y[tren])
    prob_mlp = a_clases(red.model.predict_proba(red.scaler.transform(matriz[val])),
                        red.encoder.classes_, len(val))
    prob_arbol = a_clases(lgb.LGBMClassifier(**HIERBA).fit(matriz[tren], y[tren])
                          .predict_proba(matriz[val]), np.array(comun.CLASSES), len(val))
    acumuladas = np.zeros((len(val), len(comun.CLASSES) - 1))
    for k in range(1, len(comun.CLASSES)):
        objetivo = (clase <= k - 1).astype(int)
        modelo = lgb.LGBMClassifier(**HIERBA).fit(matriz[tren], objetivo[tren])
        acumuladas[:, k - 1] = modelo.predict_proba(matriz[val])[:, 1]
    for k in range(1, len(comun.CLASSES) - 1):
        acumuladas[:, k] = np.maximum(acumuladas[:, k], acumuladas[:, k - 1])
    seis = np.zeros((len(val), len(comun.CLASSES)))
    seis[:, 0] = acumuladas[:, 0]
    for k in range(1, len(comun.CLASSES) - 1):
        seis[:, k] = acumuladas[:, k] - acumuladas[:, k - 1]
    seis[:, len(comun.CLASSES) - 1] = 1.0 - acumuladas[:, len(comun.CLASSES) - 2]
    seis = np.clip(seis, 0.0, None)
    seis = seis / np.clip(seis.sum(axis=1, keepdims=True), 1e-9, None)
    return 0.25 * prob_lr + 0.25 * prob_mlp + 0.25 * prob_arbol + 0.25 * seis


def cuartos_v20_oof(matriz: np.ndarray, y: np.ndarray, clase: np.ndarray,
                    fold_idx: np.ndarray) -> np.ndarray:
    """OOF de los cuatro cuartos por fold (protocolo de la segunda particion, v30_05:146-175)."""
    n = len(clase)
    oof = np.zeros((n, len(comun.CLASSES)))
    for fold in range(5):
        tren = np.flatnonzero(fold_idx != fold)
        val = np.flatnonzero(fold_idx == fold)
        oof[val] = _cuartos_v20_fold(matriz, clase, y, tren, val)
    return oof


def lgbm_refit_oof_sin_es(matriz: np.ndarray, clase: np.ndarray, y: np.ndarray,
                          fold_idx: np.ndarray, configs: list[dict]) -> np.ndarray:
    """Refit del techo congelado SIN early stopping (age_plugin_confirmacion.py:141-147).

    Incluye ``deterministic=True`` y ``force_col_wise=True``: sin ellos este reajuste NO era
    estable entre procesos (medido con el probe de determinismo), y es el miembro V30 de la
    segunda particion y de la evaluacion publicada del campeon.
    """
    n = len(clase)
    oof = np.zeros((n, len(comun.CLASSES)))
    for fold in range(5):
        tr = fold_idx != fold
        va = fold_idx == fold
        params = {"objective": "multiclass", "num_class": len(comun.CLASSES), "verbose": -1,
                  "n_jobs": 28, "subsample_freq": 1, "deterministic": True,
                  "force_col_wise": True, **_normaliza(configs[fold])}
        techo = int(params.pop("n_estimators"))
        modelo = lgb.LGBMClassifier(**params, n_estimators=techo).fit(matriz[tr], y[tr])
        oof[va] = a_clases(modelo.predict_proba(matriz[va]), modelo.classes_, int(va.sum()))
    return oof


def v30_oof_segunda(matriz: np.ndarray, y: np.ndarray, clase: np.ndarray,
                    fold_idx2: np.ndarray) -> np.ndarray:
    """V30 OOF en la particion alternativa (age_plugin_confirmacion.py:126-156).

    Misma receta y pesos congelados, con los reajustes del techo SIN early stopping (protocolo
    propio de la segunda particion) y los cuatro cuartos por fold.
    """
    p20 = cuartos_v20_oof(matriz, y, clase, fold_idx2)
    p29 = lgbm_refit_oof_sin_es(matriz, clase, y, fold_idx2, V29_CONFIGS)
    p30 = lgbm_refit_oof_sin_es(matriz, clase, y, fold_idx2, V30_CONFIGS)
    return mezcla(p20, p29, p30)
