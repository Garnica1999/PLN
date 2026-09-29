"""V31+ - 06: la otra via de integracion, y el diagnostico que la justifica.

## Dos cosas distintas en un script

**N2: el canal como FEATURES, no como miembro.**

V31-05 integro el canal como cuarto miembro de la mezcla y el resultado fue casi nulo: +0,0000642
sobre el control reoptimizado. Pero eso solo descarta **una** via. La via distinta es meter las
salidas del canal en el meta-espacio de 279 features y dejar que el LightGBM las use como
cortes. El mecanismo no es el mismo: como miembro, el canal solo aporta una_probabilidad_ media
que promedian los pesos; como feature, el arbol puede condicionar de forma **no lineal e interactiva** el uso de esa probabilidad junto con las 279 features. Un canal aporta mas por la
segunda via que por la primera, y V30 ya demostro que el meta-espacio es donde vive el margen.

El modelo y sus hiperparametros son **los de V30, fold a fold**, sin tocar. La unica variable que
cambia es el bloque de features anadido. Por eso el resultado es interpretable.

**Track J: donde falla V30 con alta confianza.**

El brief lo marca como prioridad especial, con un motivo concreto: un error en el que el modelo
esta seguro indica un fallo de representacion, no de umbral. Y da un punto de corte para N2: si
los errores de alta confianza se concentran en un patron linguistico concreto, hay algo que una
representacion mejor pudiera expresar. Si se reparten uniformemente, no.

Este diagnostico no produce modelo. Produce la respuesta a la pregunta cientifica del brief, y
esa respuesta vale aunque la linea del modelo se cierre.

## Los controles, otra vez

Igual que en V31-05, y por el mismo motivo: anadir 24 columnas siempre da algo, y ese algo
puede venir de la capacidad extra y no de la senal. Se comparan tres brazos con el mismo modelo,
los mismos hiperparametros y los mismos folds:

```text
A  279 features                       (control de modelo)
B  279 + canal                        (la integracion)
C  279 + canal barajado               (control de atribucion)
```

Si B no supera a C, las columnas no aportan senal: solo capacidad.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import lightgbm as lgb  # noqa: E402

from v19_b1_boosting import matriz_279  # noqa: E402
from v31_03_canales_linguisticos import (  # noqa: E402
    canal_composicion, canal_discurso, canal_ortografia,
)
from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

RESULTADO = cfg.EXPERIMENTS / "v31_06_features_y_errores.json"
ERRORES_CSV = cfg.REPORTS / "v31_high_confidence_errors.csv"
CONF_CSV = cfg.REPORTS / "v31_confusion_matrix.csv"
FECHA = "2026-09-30"
CV_V30 = 0.6134471
CV_V30_PUBLICADO = 0.6130051
N_CLASES = 6
CANALES = ("ortografia", "composicion", "discurso")
CANAL_OOF = cfg.FEATURES / "v31_canal_oof.npy"
SEMILLA = 20260930
#: Fraccion de confianza a partir de la cual un error se considera "de alta confianza".
UMBRAL_ALTA_CONFIANZA = 0.60


def _macro(clase: np.ndarray, prob: np.ndarray) -> float:
    return float(f1_score(clase, np.argmax(prob, axis=1), average="macro",
                          labels=list(range(N_CLASES))))


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def parametros_v30() -> dict:
    """Los hiperparametros que V30 eligio, fold a fold. No se reoptimizan."""
    d = json.loads((cfg.EXPERIMENTS / "v30_01_optuna_refinamiento.json").read_text(encoding="utf-8"))
    return {k: v["params_mejores"] for k, v in d["detalle_por_fold"].items()}


def features_canal(oof_canal: np.ndarray, textos: np.ndarray) -> np.ndarray:
    """Convierte el OOF del canal en un bloque compacto de features numericas.

    Todas son funciones deterministas de probabilidades **estrictamente out-of-fold** y de
    regex sobre el texto de la propia fila. No hay ninguna etiqueta en su construccion, asi que
    no hay forma de queendale informacion del fold de validacion: el canal ya se calculo sin
    mirar esos labels.
    """
    p = np.clip(oof_canal, 1e-9, 1.0)
    orden = np.argsort(-oof_canal, axis=1)
    columnas = [oof_canal, np.log(p / (1.0 - p))]
    maximo = oof_canal[np.arange(len(oof_canal)), orden[:, 0]][:, None]
    columnas.append(maximo)
    segunda = oof_canal[np.arange(len(oof_canal)), orden[:, 1]][:, None]
    columnas.append(maximo - segunda)
    entropia = -(p * np.log(p)).sum(axis=1)[:, None]
    columnas.append(entropia)
    onehot = np.zeros_like(oof_canal)
    onehot[np.arange(len(oof_canal)), orden[:, 0]] = 1.0
    columnas.append(onehot)
    for fn in (canal_ortografia, canal_composicion, canal_discurso):
        etiquetas = [fn(t).split() for t in textos]
        diccionarios = [set(e) for e in etiquetas]
        for token in ("ORTOGRAFIAPLANA", "CITAPRESENTE", "RISA", "PREGUNTA", "URLAUSENTE",
                      "REPETICION", "MENCION", "EXCLAMACIONMULTIPLE",
                      "CITAPRESENTE"):
            columnas.append(np.array([1.0 if token in d else 0.0 for d in diccionarios])[:, None])
    return np.hstack(columnas).astype(np.float64)


def entrena(matriz: np.ndarray, clase: np.ndarray, folds: list[dict],
            params_por_fold: dict, etiqueta: str) -> dict:
    """OOF de un LightGBM con los parametros de V30. Sin Optuna: la variable son las features."""
    inicio = time.time()
    oof = np.zeros((len(clase), N_CLASES), dtype=np.float64)
    por_fold = []
    for i, f in enumerate(folds):
        ent, val = f["train_idx"], f["val_idx"]
        params = dict(params_por_fold[str(i)])
        for clave in ("num_leaves", "max_depth", "min_child_samples", "n_estimators"):
            if clave in params and params[clave] is not None:
                params[clave] = int(params[clave])
        modelo = lgb.LGBMClassifier(objective="multiclass", n_jobs=8, random_state=SEMILLA,
                                    verbose=-1, **params)
        modelo.fit(matriz[ent], clase[ent])
        prob = modelo.predict_proba(matriz[val])
        oof[val] = prob
        por_fold.append(_macro(clase[val], prob))
    cv = _macro(clase, oof)
    np.save(cfg.FEATURES / f"v31_06_oof_{etiqueta}.npy", oof)
    return {"etiqueta": etiqueta, "n_features": int(matriz.shape[1]), "macro_f1": round(cv, 7),
            "folds": [round(v, 7) for v in por_fold], "segundos": round(time.time() - inicio, 1)}


def analisis_errores(clase: np.ndarray, oof_v30: np.ndarray, textos: np.ndarray) -> tuple:
    """Track J: donde falla V30, separando el error de frontera del error de representacion.

    La distincion que manda: un error con confianza baja es un error de **umbral**, y cualquier
    calibracion lo arregla. Un error con confianza alta dice que el modelo estaba seguro de algo
    falso, y eso solo lo arregla una representacion distinta. Solo la segunda categoria informa
    sobre que falta.
    """
    pred = np.argmax(oof_v30, axis=1)
    conf = oof_v30[np.arange(len(clase)), pred]
    error = pred != clase
    alta = error & (conf >= UMBRAL_ALTA_CONFIANZA)
    baja = error & (conf < UMBRAL_ALTA_CONFIANZA)
    filas = []
    for nombre_clase, c in zip(cfg.CLASSES, range(N_CLASES)):
        m = clase == c
        filas.append({
            "clase": nombre_clase, "soporte": int(m.sum()),
            "errores": int((error & m).sum()),
            "tasa_error": round(float((error & m).sum() / max(m.sum(), 1)), 5),
            "errores_alta_confianza": int((alta & m).sum()),
            "de_que_fraccion": round(float((alta & m).sum() / max((error & m).sum(), 1)), 4),
            "confianza_media_en_sus_errores": round(float(conf[error & m].mean()), 4),
        })
    tabla = pd.DataFrame(filas)
    # Confusiones adyacentes y no adyacentes.
    cm = confusion_matrix(clase, pred, labels=list(range(N_CLASES))
                          ) / confusion_matrix(clase, pred,
                                               labels=list(range(N_CLASES))).sum(axis=1, keepdims=True)
    confusiones = []
    for a in range(N_CLASES):
        for b in range(N_CLASES):
            if a == b or cm[a, b] < 0.02:
                continue
            confusiones.append({"real": cfg.CLASSES[a], "predicho": cfg.CLASSES[b],
                                "tasa_de_la_fila": round(float(cm[a, b]), 4),
                                "distancia": abs(a - b),
                                "adyacente": abs(a - b) == 1})
    confusiones.sort(key=lambda r: -r["tasa_de_la_fila"])
    tabla.to_csv(ERRORES_CSV, index=False)
    pd.DataFrame(confusiones).to_csv(CONF_CSV, index=False)

    # Perfil linguistico de los errores de alta confianza frente a los aciertos.
    texto = np.array(textos, dtype=object)
    long = np.array([len(t) for t in texto])
    digitos = np.array([sum(c.isdigit() for c in t) / max(len(t), 1) for t in texto])
    mayus = np.array([sum(1 for p in t.split() if p.isupper() and len(p) > 1) /
                      max(len(t.split()), 1) for t in texto])
    exclam = np.array([t.count("!") for t in texto])
    perfil = pd.DataFrame([{
        "grupo": "aciertos", "n": int((~error).sum()),
        "longitud_media": round(float(long[~error].mean()), 2),
        "densidad_digitos": round(float(digitos[~error].mean()), 5),
        "densidad_mayus": round(float(mayus[~error].mean()), 5),
        "exclamaciones": round(float(exclam[~error].mean()), 3),
        "confianza_media": round(float(conf[~error].mean()), 4),
    }, {
        "grupo": "errores de baja confianza (umbral)", "n": int(baja.sum()),
        "longitud_media": round(float(long[baja].mean()), 2),
        "densidad_digitos": round(float(digitos[baja].mean()), 5),
        "densidad_mayus": round(float(mayus[baja].mean()), 5),
        "exclamaciones": round(float(exclam[baja].mean()), 3),
        "confianza_media": round(float(conf[baja].mean()), 4),
    }, {
        "grupo": "errores de ALTA confianza (representacion)", "n": int(alta.sum()),
        "longitud_media": round(float(long[alta].mean()), 2),
        "densidad_digitos": round(float(digitos[alta].mean()), 5),
        "densidad_mayus": round(float(mayus[alta].mean()), 5),
        "exclamaciones": round(float(exclam[alta].mean()), 3),
        "confianza_media": round(float(conf[alta].mean()), 4),
    }])
    return {
        "n_errores": int(error.sum()), "tasa_error": round(float(error.mean()), 5),
        "errores_alta_confianza": int(alta.sum()),
        "fraccion_de_errores_que_son_de_alta_confianza": round(float(alta.sum() / max(error.sum(), 1)), 5),
        "errores_baja_confianza": int(baja.sum()),
        "umbral": UMBRAL_ALTA_CONFIANZA,
        "por_clase": tabla.to_dict("records"),
        "confusiones_principales": confusiones[:10],
        "perfil_linguistico": perfil.to_dict("records"),
        "fraccion_no_adyacente": round(float(np.mean([c["adyacente"] is False
                                                      for c in confusiones[:10]])), 4),
    }


def main() -> int:
    autotest_numerico()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(v) for v in y])
    textos = train["text"].fillna("").astype(str).to_numpy()
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")
    params = parametros_v30()

    print("=== cargando el meta-espacio de 279 features ===", flush=True)
    base = matriz_279()
    print(f"  {base.shape}", flush=True)
    if not CANAL_OOF.exists():
        print("[STOP] falta el OOF del canal. Ejecutar antes v31_05_integracion.py", flush=True)
        return 1
    oof_canal = np.load(CANAL_OOF)
    extra = features_canal(oof_canal, textos)
    print(f"  bloque del canal: {extra.shape}", flush=True)
    completa = np.hstack([base, extra])
    rng = np.random.default_rng(SEMILLA)
    barajada = extra[rng.permutation(len(extra))]

    print("\n=== A: 279 features (control de modelo) ===", flush=True)
    a = entrena(base, clase, folds, params, "solo279")
    print(f"  {a['macro_f1']:.7f} | {a['segundos']:.0f} s", flush=True)

    print("\n=== B: 279 + canal (N2) ===", flush=True)
    b = entrena(completa, clase, folds, params, "mas_canal")
    b["delta_vs_A"] = round(b["macro_f1"] - a["macro_f1"], 7)
    print(f"  {b['macro_f1']:.7f} | vs A {b['delta_vs_A']:+.7f} | {b['segundos']:.0f} s", flush=True)

    print("\n=== C: 279 + canal BARAJADO (control de atribucion) ===", flush=True)
    c = entrena(np.hstack([base, barajada]), clase, folds, params, "mas_canal_barajado")
    c["delta_vs_A"] = round(c["macro_f1"] - a["macro_f1"], 7)
    print(f"  {c['macro_f1']:.7f} | vs A {c['delta_vs_A']:+.7f} | {c['segundos']:.0f} s", flush=True)

    print("\n=== Track J: errores de alta confianza de V30 ===", flush=True)
    ens = json.loads((cfg.EXPERIMENTS / "v30_02_ensemble.json").read_text(encoding="utf-8"))
    pool = np.load(cfg.EXPERIMENTS / "v29_pool_oof.npz")
    mats, nombres = [], []
    for n in ens["manifest"]["miembros"]:
        if n == "v30_lgbm_refinado":
            mats.append(np.load(cfg.FEATURES / "v30_lgbm_refinado_oof.npy").astype(float))
        elif n in pool.files:
            mats.append(pool[n].astype(float))
        else:
            mats.append(np.load(cfg.FEATURES / f"{n}.npy").astype(float))
        nombres.append(n)
    oof_v30 = np.tensordot(np.array([ens["manifest"]["pesos"][n] for n in nombres]),
                           np.stack(mats), axes=(0, 0))
    cv_v30 = _macro(clase, oof_v30)
    print(f"  V30 {cv_v30:.7f} | canonico {CV_V30}", flush=True)
    errores = analisis_errores(clase, oof_v30, textos)
    print(f"  errores: {errores['n_errores']:,} ({errores['tasa_error']:.2%})", flush=True)
    print(f"  de alta confianza: {errores['errores_alta_confianza']:,} "
          f"({errores['fraccion_de_errores_que_son_de_alta_confianza']:.1%} de los errores)",
          flush=True)
    print(f"  confusiones no adyacentes entre las 10 principales: "
          f"{errores['fraccion_no_adyacente']:.0%}", flush=True)
    print("\n  perfil de los tres grupos:", flush=True)
    for fila in errores["perfil_linguistico"]:
        print(f"    {fila['grupo']:<42} n={fila['n']:>7,} conf={fila['confianza_media']:.3f} "
              f"long={fila['longitud_media']:.1f} mayus={fila['densidad_mayus']:.3f}", flush=True)

    aporte = b["macro_f1"] - c["macro_f1"]
    if aporte <= 0:
        veredicto = (f"SIN APORTE: las {extra.shape[1]} columnas del canal rindieron "
                     f"{aporte:+.7f} frente a las mismas columnas barajadas")
    elif aporte < 0.0005:
        veredicto = f"APORTE MARGINAL: {aporte:+.7f} sobre el control barajado, bajo el gate"
    else:
        veredicto = f"APORTE REAL: {aporte:+.7f} sobre el control barajado"

    RESULTADO.write_text(json.dumps(_jsonable({
        "fase": "V31+ 06 canal como features y analisis de errores", "fecha": FECHA,
        "experimenta": True, "baseline_v30": CV_V30, "cv_v30_publicado": CV_V30_PUBLICADO,
        "n2_brazos": [a, b, c],
        "n2_veredicto": veredicto,
        "n2_aporte_sobre_control_barajado": round(aporte, 7),
        "n2_delta_vs_control_de_modelo": b["delta_vs_A"],
        "track_j_errores": errores,
        "restricciones": {"usa_test": False, "usa_red": False, "usa_embeddings": False,
                          "reoptimiza_hiperparametros": False,
                          "nota_hiperparametros": "son los de V30, fold a fold, sin tocar"},
        "segundos": round(time.time() - inicio, 1),
    }), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[fin] {time.time() - inicio:.0f} s | {veredicto}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
