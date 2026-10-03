"""V30 - 03: construir y validar la submission del candidato congelado.

**Este script NO sube nada a Kaggle.** Construye el fichero, lo valida y lo sella con su hash. La
subida vive en `v30_04_upload.py`, aparte, para que se pueda inspeccionar el resultado antes de
que exista una unica e irreversible sonda.

## La utilizacion del test, y por que es solo esta

Este es el **unico** punto de toda la investigacion donde el test se toca, y se toca de la unica
forma que el brief autoriza:

    test text -> transform -> predict -> submission

No se lee ninguna etiqueta del test, ni se calcula ninguna estadistica sobre el test mas alla de
la matriz de features que ya existia, ni se ajusta ningun umbral con el test. Los pesos y los
hiperparametros estan **congelados antes** de que este script se ejecute.

## Los tres miembros y como se ajustan sobre todo el train

| Miembro | Como se ajusta |
|---|---|
| V20 | sus cuatro cuartos, con la receta exacta de `v20_f3c_candidato.py` y sus pesos de 0,25 |
| LightGBM de V29 | una configuracion **por fold** (la mejor de V29 para ese fold), las cinco ajustadas sobre todo el train y promediadas |
| LightGBM de V30 | igual, con las configuraciones de V30 |

**Por que cinco modelos y no uno.** Cada fold produjo su mejor configuracion, y son distintas
(num_leaves de 43 a 87 en V29). Elegir una sola seria un arbitrio sin base. Ajustar las cinco
sobre todo el train y promediar usa toda la evidencia de la busqueda y anade una reduccion de
varianza gratuita. Se declara como decision, no como detalle.

## Los checks, que son los que deciden si el fichero se sube

Numero de filas, identificadores iguales al test, orden correcto, columnas correctas, sin NaN,
sin duplicados, etiquetas validas, y el ancho de las etiquetas. Ademas se comprueba **cuantas
decisiones difieren del envio de V20**, que es la magnitud que interesa antes de gastar la sonda.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from phase2_ponderacion import EncodedMLP  # noqa: E402
from v13_p1b_fusion_tokens import logreg  # noqa: E402
from v19_b1_boosting import HIERBA, matriz_279  # noqa: E402

from src import config as cfg  # noqa: E402
from src import data as datos  # noqa: E402
from src.config import set_seed  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

RESULTADO = cfg.EXPERIMENTS / "v30_03_submission_construccion.json"
MATRIZ_TEST = cfg.FEATURES / "meta_features_test_v12_fresca.npy"
ARCHIVO = cfg.ROOT / "submissions" / "age_submission_v30_final_candidate.csv"
CV_V20 = 0.6097782
N_CLASES = 6
HILOS = 28


def a_clases(probabilidad: np.ndarray, clases: np.ndarray, filas: int) -> np.ndarray:
    """Reordena las columnas de probabilidad al orden canonico de `cfg.CLASSES`.

    Acepta **las dos** representaciones de `clases_` que aparecen en el proyecto: las etiquetas
    string (cuando el modelo se ajusta sobre `y`) y los indices enteros 0-5 (cuando se ajusta
    sobre `clase`). Sin esa tolerancia, ajustar el LightGBM sobre `clase` y pasar su `classes_`
    aqui reventaba con `list.index(x): x not in list`, porque se buscaban strings en un array de
    enteros.
    """
    salida = np.zeros((filas, N_CLASES))
    existentes = list(clases)
    son_enteros = all(isinstance(v, (int, np.integer)) for v in existentes)
    for destino, etiqueta in enumerate(cfg.CLASSES):
        buscar = destino if son_enteros else etiqueta
        if buscar not in existentes:
            raise ValueError(
                f"el modelo no conoce la clase {buscar!r}; sus classes_ son {existentes}")
        salida[:, destino] = probabilidad[:, existentes.index(buscar)]
    return salida


def _v20_en_test(matriz: np.ndarray, test_matriz: np.ndarray, y: np.ndarray, clase: np.ndarray,
                 n_test: int) -> np.ndarray:
    """La receta exacta de V20, sin tocar ni un parametro."""
    modelo_lr = logreg().fit(matriz, y)
    prob_lr = a_clases(modelo_lr.predict_proba(test_matriz), modelo_lr.classes_, n_test)
    red = EncodedMLP().fit(matriz, y)
    prob_mlp = a_clases(red.model.predict_proba(red.scaler.transform(test_matriz)),
                        red.encoder.classes_, n_test)
    modelo_arbol = lgb.LGBMClassifier(**HIERBA).fit(matriz, y)
    prob_arbol = a_clases(modelo_arbol.predict_proba(test_matriz), modelo_arbol.classes_, n_test)
    acumuladas = np.zeros((n_test, N_CLASES - 1))
    for k in range(1, N_CLASES):
        objetivo = (clase <= k - 1).astype(int)
        modelo = lgb.LGBMClassifier(**HIERBA).fit(matriz, objetivo)
        acumuladas[:, k - 1] = modelo.predict_proba(test_matriz)[:, 1]
    for k in range(1, N_CLASES - 1):
        acumuladas[:, k] = np.maximum(acumuladas[:, k], acumuladas[:, k - 1])
    seis = np.zeros((n_test, N_CLASES))
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
    """Parametros del mejor trial de cada fold, en orden de fold.

    V29 no los persistio en su JSON, asi que se recuperan de `reports/v29-01_trials.csv`; V30 si
    los guarda en su artefacto. Se usa **la misma funcion** que en la segunda particion, para que
    no haya dos lecturas distintas de la misma configuracion.
    """
    from v30_05_segunda_particion import _configs_del_miembro as _comun  # noqa: PLC0415
    return _comun(nombre)


def _lgbm_en_test(matriz: np.ndarray, test_matriz: np.ndarray, clase: np.ndarray,
                  configs: list[dict], n_test: int) -> tuple[np.ndarray, list[dict]]:
    """Una configuracion por fold, las cinco sobre todo el train, y el promedio de sus prediciones.

    Las configuraciones llegan congeladas desde `v30_02_ensemble.json`. Este script **no elige
    ninguna**.
    """
    predicciones, diagnostico = [], []
    for indice, params in enumerate(configs):
        p = {"objective": "multiclass", "num_class": N_CLASES, "verbose": -1, "n_jobs": HILOS,
             "subsample_freq": 1, **_normaliza(params)}
        techo = int(p.pop("n_estimators"))
        modelo = lgb.LGBMClassifier(**p, n_estimators=techo)
        # Sin early stopping aqui: el criterio interior **ya** decidio cuantas iteraciones hacen
        # falta, y esas son las que se usan. Entrenar mas alla seria_artificial.
        modelo.fit(matriz, clase)
        predicciones.append(a_clases(modelo.predict_proba(test_matriz), modelo.classes_, n_test))
        diagnostico.append({"config": indice, "n_estimators": techo,
                            "iteraciones_efectivas": techo, "early_stopping": False,
                            "nota": "el numero de iteraciones lo fijo el early stopping del "
                                    "interior de Optuna; aqui se entrena ese numero exacto"})
    return np.mean(predicciones, axis=0), diagnostico


def main() -> int:
    """Construye el fichero de submission del candidato congelado y lo valida."""
    autotest_numerico()
    set_seed()
    inicio = time.time()
    marca = datetime.now(timezone.utc).isoformat(timespec="seconds")

    ens = json.loads((cfg.EXPERIMENTS / "v30_02_ensemble.json").read_text(encoding="utf-8"))
    manifest = ens["manifest"]
    pesos = manifest["pesos"]
    print(f"[V30] candidato: {ens['candidato_principal']}", flush=True)
    print(f"[V30] pesos congelados: {pesos}", flush=True)

    train = datos.load_train()
    test = datos.load_test()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(v) for v in y])
    matriz = matriz_279()
    test_matriz = np.asarray(np.load(MATRIZ_TEST, mmap_mode="r"))
    id_test = test[cfg.ID_COL].astype(str).to_numpy()
    n_test = len(test)
    print(f"[V30] train {len(train):,} filas x {matriz.shape[1]} features | test {n_test:,} filas",
          flush=True)

    # ---------------------------------------------------------------- train/test schema
    # Comparacion de esquema **antes** de nada: si el train y el test no tuvieran las mismas
    # columnas, todo lo demas seria invalido y hay que saberlo ahora.
    esquema = {
        "train_columnas": list(train.columns),
        "test_columnas": list(test.columns),
        "mismas_columnas_sin_el_target": (
            [c for c in train.columns if c != cfg.TARGET] == list(test.columns)),
        "target_ausente_en_test": cfg.TARGET not in test.columns,
        "matriz_test_shape": list(test_matriz.shape),
        "matriz_test_coincide_con_ancho": int(test_matriz.shape[1]) == matriz.shape[1],
    }
    if not (esquema["mismas_columnas_sin_el_target"] and esquema["target_ausente_en_test"]
            and esquema["matriz_test_coincide_con_ancho"]):
        print(f"[ABORTO] el esquema train/test no es el esperado: {esquema}")
        return 1
    print("[1/5] esquema train/test verificado", flush=True)

    # ---------------------------------------------------------------- miembros
    print("[2/5] V20 sobre todo el train, con su receta exacta e intacta", flush=True)
    prob_v20 = _v20_en_test(matriz, test_matriz, y, clase, n_test)

    miembros, detalle_miembros = {}, {}
    for nombre in manifest["miembros"]:
        if nombre == "v20_f3b_oof_candidato":
            miembros[nombre] = prob_v20
            detalle_miembros[nombre] = {"origen": "receta de V20, sin modificar"}
            print(f"      {nombre}: listo", flush=True)
            continue
        configs = _configs_del_miembro(nombre)
        print(f"      {nombre}: {len(configs)} configuraciones congeladas, ajustando sobre todo "
              f"el train", flush=True)
        prob, diag = _lgbm_en_test(matriz, test_matriz, clase, configs, n_test)
        miembros[nombre] = prob
        detalle_miembros[nombre] = {"artefacto": ("v29_04_optuna_lightgbm.json" if "v29" in nombre else "v30_01_optuna_refinamiento.json"), "n_configuraciones": len(configs),
                                    "configuraciones": configs, "diagnostico": diag}

    print("[3/5] combinando con los pesos congelados", flush=True)
    nombres = list(miembros)
    vector = np.array([pesos.get(n, 0.0) for n in nombres], dtype=float)
    if vector.sum() <= 0:
        print(f"[ABORTO] los pesos congelados suman cero: {pesos}")
        return 1
    vector = vector / vector.sum()
    prob_candidata = np.tensordot(vector, np.stack([miembros[n] for n in nombres]), axes=(0, 0))
    pred = np.array(cfg.CLASSES)[np.argmax(prob_candidata, axis=1)]

    # ---------------------------------------------------------------- checks
    print("[4/5] validando el fichero", flush=True)
    marco = pd.DataFrame({cfg.ID_COL: id_test, "age_range": pred})
    checks = {
        "rows_ok": len(marco) == n_test,
        "columns_ok": list(marco.columns) == [cfg.ID_COL, "age_range"],
        "ids_equal_test": bool(np.array_equal(marco[cfg.ID_COL].astype(str).to_numpy(), id_test)),
        "ids_same_order": bool(np.array_equal(marco[cfg.ID_COL].astype(str).to_numpy(), id_test)),
        "ids_unique": bool(marco[cfg.ID_COL].is_unique),
        "no_nan": bool(not marco.isna().any().any()),
        "labels_valid": bool(set(marco["age_range"]).issubset(set(cfg.CLASSES))),
        "labels_width_ok": bool(all(len(v) <= cfg.PADDED_WIDTH for v in marco["age_range"])),
        "sin_filas_vacias": int(marco[cfg.ID_COL].isna().sum()) == 0,
        "schema_verificado": True,
    }
    fallidos = [k for k, v in checks.items() if not v]
    if fallidos:
        print(f"[ABORTO] checks fallidos: {fallidos}")
        RESULTADO.write_text(json.dumps({"fase": "V30 03 construccion", "checks": checks,
                                         "fallidos": fallidos}, indent=2, ensure_ascii=False),
                             encoding="utf-8")
        return 1

    # Cuantas decisiones difieren del envio de V20: es la magnitud que conviene mirar antes de
    # gastar la unica sonda.
    ruta_v20 = cfg.ROOT / "submissions" / "age_submission_v20_candidato_umbrales.csv"
    comparacion = {}
    if ruta_v20.exists():
        previa = pd.read_csv(ruta_v20)
        if np.array_equal(previa[cfg.ID_COL].astype(str).to_numpy(), id_test):
            antes = previa["age_range"].astype(str).to_numpy()
            distintas = int((pred != antes).sum())
            comparacion = {
                "envio_de_referencia": ruta_v20.name,
                "decisiones_distintas": distintas,
                "proporcion": round(distintas / len(antes), 6),
                "distribucion_envio_v20": {c: int((antes == c).sum()) for c in cfg.CLASSES},
            }
            print(f"      difiere del envio de V20 en {distintas:,} de {len(antes):,} filas "
                  f"({distintas / len(antes):.2%})", flush=True)
    else:
        comparacion = {"envio_de_referencia": "no encontrado",
                       "nota": "no se encontro el envio de V20 para comparar; los checks siguen "
                               "siendo validos porque el test es la referencia"}

    ARCHIVO.parent.mkdir(parents=True, exist_ok=True)
    marco.to_csv(ARCHIVO, index=False)
    sha = hashlib.sha256(ARCHIVO.read_bytes()).hexdigest()
    print(f"[5/5] {sum(checks.values())}/{len(checks)} checks | {ARCHIVO.name} | "
          f"SHA256 {sha}", flush=True)
    print(f"      filas {len(marco):,} | distribucion "
          f"{ {c: int((pred == c).sum()) for c in cfg.CLASSES} }", flush=True)

    RESULTADO.write_text(json.dumps({
        "fase": "V30 03 construccion de la submission", "fecha": "2026-09-30",
        "momento_utc": marca, "experimenta": False,
        "candidato": ens["candidato_principal"],
        "pesos_congelados": pesos,
        "vector_pesos_usado": {n: round(float(w), 6) for n, w in zip(nombres, vector)},
        "miembros": detalle_miembros,
        "schema_train_test": esquema,
        "checks": checks, "checks_superados": sum(checks.values()), "checks_total": len(checks),
        "distribucion_predicciones": {c: int((pred == c).sum()) for c in cfg.CLASSES},
        "comparacion_con_envio_v20": comparacion,
        "archivo": ARCHIVO.name, "sha256": sha, "bytes": int(ARCHIVO.stat().st_size),
        "filas": int(len(marco)),
        "uso_del_test": "solo transform y predict. Ninguna etiqueta, ninguna estadistica, ningun "
                        "ajuste de umbral. Los pesos y los hiperparametros estaban congelados "
                        "antes de entrar aqui.",
        "subido_a_kaggle": False,
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[V30] {time.time() - inicio:.0f} s. La subida es un script aparte.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
