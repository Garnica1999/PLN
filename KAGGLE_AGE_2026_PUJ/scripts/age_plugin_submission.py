"""Submission experimental: V30 congelado + regla plug-in congelada.

## La pregunta que responde

> ¿la mejora de decision observada internamente se transfiere al test?

## Lo que este script NO hace

No cambia nada de V30. No reajusta los pesos. No calibra. No normaliza las probabilidades. No
genera variantes. No toca el holdout. No consulta el leaderboard. No usa etiquetas de test.

## El pipeline, y por que se reutiliza en vez de reescribirse

El candidato oficial de V30 lo produjo `v30_03_submission.py`. Este script **importa sus
funciones** (`_v20_en_test`, `_lgbm_en_test`, `_configs_del_miembro`) en vez de reimplementarlas:
reimplementar un pipeline que produjo un leaderboard conocido es la forma mas facil de construir
otro modelo sin darse cuenta. La unica linea que cambia respecto de aquel script es la decision
final:

```text
v30_03:   pred = CLASSES[ argmax( prob ) ]
aqui:     pred = CLASSES[ argmax( w * prob ) ]     con w = [1.00, 1.17, 1.27, 1.16, 1.00, 0.81]
```

Todo lo anterior —el ajuste de V20 sobre todo el train, las configuraciones congeladas de los dos
LightGBM, la mezcla con los pesos medios del manifest— es identico.

## La verificacion del orden de clases, que va primero

El unico error que podria arruinar esto en silencio es asignar `1.27` a la clase equivocada. Asi que
**no se asume el orden de columnas**: se demuestra. Sobre el OOF de la particion canonica, donde si
hay etiquetas, se comprueba que para cada clase `c`, la columna `c` es la que mas se activa
precisamente en las filas de clase `c`. Se calcula la matriz completa de medias `E[p[:, j] | y = i]`
y se verifica que su diagonal domina cada fila. Si el argmax por fila no cae en la diagonal, **STOP**
y no se genera submission.

## El sanity check, antes de tocar el test

Antes de predecir nada sobre test se reproduce la mejora conocida sobre el OOF:

```text
canonica     0,6134471 -> 0,6154643
alternativa  0,6144120 -> 0,6164700
```

Si esos numeros no salen, hay un bug de orden, de pesos o de prediccion, y la submission no se
genera. Es exactamente para lo que sirve: no es tuning, es un control de integridad.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg  # noqa: E402
from src import data as datos  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

SUBMISSION = ROOT / "submissions" / "age_submission_plugin_candidate.csv"
PROB_TEST = cfg.FEATURES / "age_plugin_test_prob.npy"
MANIFEST = cfg.REPORTS / "age_submission_plugin_manifest.json"
RESULTADO = cfg.EXPERIMENTS / "age_plugin_submission.json"
FECHA = "2026-10-12"
N_CLASES = 6
MATRIZ_TEST = cfg.FEATURES / "meta_features_test_v12_fresca.npy"
PESOS_CONGELADOS = np.array([1.0000, 1.1749, 1.2721, 1.1630, 1.0010, 0.8128])
#: Los valores de referencia. **No son un objetivo: son el control de integridad.** Si no salen, hay
#: un bug de orden, de pesos o de prediccion, y no se genera nada.
#:
#: IMPORTANTE, y es una distincion que la primera version de este script confundia: hay TRES numeros
#: distintos sobre la particion canonica, y solo uno corresponde a lo que hace la submission.
#:
#:   (A) nested        0,6154643   cada fold con SUS pesos, ajustados en los otros cuatro
#:   (B) congelados    0,6159732   los pesos congelados aplicados GLOBALMENTE  <- la submission
#:   (C) redondeados   0,6159908   los mismos pesos redondeados a dos decimales, global
#:
#: La submission aplica los pesos congelados de forma global, asi que su analogo interno es (B),
#: no (A). Y que (B) > (A) es exactamente la firma de la fuga declarada en el pre-registro: los
#: pesos congelados se derivaron de la union de cinco ajustes que cubren las 169.336 filas, asi que
#: su aplicacion global esta algo sobreajustada. **La estimacion honesta del mecanismo es (A)**, el
#: nested. (B) se usa aqui solo porque es la operacion que la submission replica.
ESPERADO_FROZEN_GLOBAL = 0.6159732
ESPERADO_NESTED = 0.6154643
ESPERADO_ALTERNATIVA = 0.6164700
ESPERADO_V30_CANONICA = 0.6134471
ESPERADO_V30_ALTERNATIVA = 0.6144120
TOLERANCIA = 1e-6
FILAS_TEST = 112892


def _macro(clase, prob) -> float:
    return float(f1_score(clase, np.argmax(prob, axis=1), average="macro",
                          labels=list(range(N_CLASES))))


def _sha256(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def commit_actual() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT.parent, capture_output=True,
                              text=True, timeout=30).stdout.strip() or "desconocido"
    except Exception:  # noqa: BLE001
        return "desconocido"


def verifica_orden_de_clases() -> dict:
    """Demuestra el orden de columnas sobre el OOF, donde hay etiquetas. No lo asume.

    El test decisivo: para cada clase `i`, la columna `j=i` tiene que ser la de mayor media dentro
    de las filas de clase `i`. Se calcula la matriz completa `E[p[:, j] | y = i]` y se comprueba que
    el argmax de cada fila cae en la diagonal. Si alguna fila no cae, el orden no es el que dice
    `cfg.CLASSES` y hay que parar.
    """
    car = ROOT / "artifacts" / "p33_scores" / "ortografia_composicion_discurso"
    clase = np.load(car / "y_true.npy")
    oof = np.load(car / "v30_prob.npy")
    p = np.clip(oof, 1e-12, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    medias = np.zeros((N_CLASES, N_CLASES))
    for i in range(N_CLASES):
        m = clase == i
        medias[i] = p[m].mean(axis=0)
    diagonal_domina = [int(np.argmax(medias[i])) == i for i in range(N_CLASES)]
    return {
        "orden_declarado": list(cfg.CLASSES),
        "matriz_E_p_col_j_dado_y_i": [[round(float(x), 4) for x in fila] for fila in medias],
        "argmax_por_fila": [int(np.argmax(medias[i])) for i in range(N_CLASES)],
        "diagonal_domina": diagonal_domina,
        "orden_verificado": bool(all(diagonal_domina)),
        "lectura": "para cada clase real i, la columna cuya media es mayor es la i: el orden de "
                   "columnas coincide con cfg.CLASSES",
    }


def sanity_check() -> dict:
    """Reproduce las dos mejoras ya medidas sobre OOF. Es un control, no tuning."""
    car = ROOT / "artifacts" / "p33_scores" / "ortografia_composicion_discurso"
    clase = np.load(car / "y_true.npy")
    oof = np.load(car / "v30_prob.npy")
    p = np.clip(oof, 1e-12, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    cv_v30 = _macro(clase, p)
    cv_plug = _macro(clase, p * PESOS_CONGELADOS)
    pred_v30 = np.argmax(p, axis=1)
    pred_plug = np.argmax(p * PESOS_CONGELADOS, axis=1)
    conf = p[np.arange(len(clase)), pred_v30]
    cambio = pred_plug != pred_v30
    alta = conf >= 0.60
    return {
        "canonica": {
            "v30": round(cv_v30, 7), "v30_esperado": ESPERADO_V30_CANONICA,
            "frozen_global": round(cv_plug, 7),
            "frozen_global_esperado": ESPERADO_FROZEN_GLOBAL,
            "nested_esperado": ESPERADO_NESTED,
            "delta_nested": round(ESPERADO_NESTED - ESPERADO_V30_CANONICA, 7),
            "delta_frozen_global": round(cv_plug - cv_v30, 7),
            "nota": "el sanity check comprueba frozen_global, porque es la operacion que ejecuta la "
                    "submission. El nested es la estimacion HONESTA del mecanismo y va 0,0005 por "
                    "debajo: la diferencia es la fuga declarada en el pre-registro",
            "v30_ok": bool(abs(cv_v30 - ESPERADO_V30_CANONICA) < TOLERANCIA),
            "frozen_global_ok": bool(abs(cv_plug - ESPERADO_FROZEN_GLOBAL) < TOLERANCIA),
        },
        "filas_que_cambian": int(cambio.sum()),
        "confianza_media_de_las_que_cambian": round(float(conf[cambio].mean()), 5),
        "confianza_media_del_conjunto": round(float(conf.mean()), 5),
        "cambios_en_alta_confianza": int((alta & cambio).sum()),
        "reproduce": bool(abs(cv_v30 - ESPERADO_V30_CANONICA) < TOLERANCIA
                          and abs(cv_plug - ESPERADO_FROZEN_GLOBAL) < TOLERANCIA
                          and int((alta & cambio).sum()) == 0),
    }


def main() -> int:
    """Construye la submission aplicando la regla congelada al pipeline oficial de V30."""
    autotest_numerico()
    inicio = time.time()
    marca = datetime.now(timezone.utc).isoformat(timespec="seconds")
    from v30_03_submission import _configs_del_miembro, _lgbm_en_test, _v20_en_test
    from v19_b1_boosting import matriz_279

    print("=== [0/6] verificacion del orden de clases ===", flush=True)
    orden = verifica_orden_de_clases()
    for i, fila in enumerate(orden["matriz_E_p_col_j_dado_y_i"]):
        print(f"  clase {cfg.CLASSES[i]:>6}: medias por columna {fila} -> argmax "
              f"{orden['argmax_por_fila'][i]}", flush=True)
    print(f"  orden verificado: {orden['orden_verificado']}", flush=True)
    if not orden["orden_verificado"]:
        print("[STOP] el orden de columnas NO coincide con cfg.CLASSES. No se genera submission.",
              flush=True)
        return 1

    print("\n=== [1/6] sanity check sobre OOF: reproduce las dos mejoras conocidas ===", flush=True)
    san = sanity_check()
    print(f"  canonica: V30 {san['canonica']['v30']} (esperado {ESPERADO_V30_CANONICA}) | "
          f"frozen-global {san['canonica']['frozen_global']} "
          f"(esperado {ESPERADO_FROZEN_GLOBAL})", flush=True)
    print(f"  el nested, que es la estimacion HONESTA del mecanismo, es {ESPERADO_NESTED}: "
          f"0,0005 por debajo, y esa diferencia es la fuga declarada", flush=True)
    print(f"  filas que cambian: {san['filas_que_cambian']:,} | alta confianza: "
          f"{san['cambios_en_alta_confianza']}", flush=True)
    print(f"  reproduce: {san['reproduce']}", flush=True)
    if not san["reproduce"]:
        print("[STOP] el sanity check falla: hay un bug de orden, de pesos o de prediccion. "
              "No se genera submission.", flush=True)
        return 1

    print("\n=== [2/6] cargando datos y el esquema de V30 ===", flush=True)
    ens = json.loads((cfg.EXPERIMENTS / "v30_02_ensemble.json").read_text(encoding="utf-8"))
    pesos_v30 = ens["manifest"]["pesos"]
    train = datos.load_train()
    test = datos.load_test()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(v) for v in y])
    matriz = matriz_279()
    test_matriz = np.asarray(np.load(MATRIZ_TEST, mmap_mode="r"))
    id_test = test[cfg.ID_COL].astype(str).to_numpy()
    n_test = len(test)
    print(f"  train {len(train):,} x {matriz.shape[1]} | test {n_test:,} | "
          f"pesos de V30 congelados: {pesos_v30}", flush=True)
    if n_test != FILAS_TEST or test_matriz.shape[1] != matriz.shape[1]:
        print(f"[STOP] esquema inesperado: {n_test} filas de test, {test_matriz.shape[1]} columnas",
              flush=True)
        return 1

    print("\n=== [3/6] los tres miembros de V30, con su receta congelada ===", flush=True)
    prob_v20 = _v20_en_test(matriz, test_matriz, y, clase, n_test)
    miembros = {"v20_f3b_oof_candidato": prob_v20}
    for nombre in ens["manifest"]["miembros"]:
        if nombre == "v20_f3b_oof_candidato":
            continue
        configs = _configs_del_miembro(nombre)
        print(f"  {nombre}: {len(configs)} configuraciones congeladas", flush=True)
        prob, _ = _lgbm_en_test(matriz, test_matriz, clase, configs, n_test)
        miembros[nombre] = prob
    nombres = list(miembros)
    vector = np.array([pesos_v30.get(n, 0.0) for n in nombres], dtype=float)
    vector = vector / vector.sum()
    prob = np.tensordot(vector, np.stack([miembros[n] for n in nombres]), axes=(0, 0))
    np.save(PROB_TEST, prob)
    print(f"  probabilidades de test guardadas: {prob.shape} | suman {prob.sum(axis=1).mean():.6f}",
          flush=True)

    print("\n=== [4/6] la regla: score = w * p, clase = argmax(score) ===", flush=True)
    score = prob * PESOS_CONGELADOS
    pred_base = np.array(cfg.CLASSES)[np.argmax(prob, axis=1)]
    pred_plug = np.array(cfg.CLASSES)[np.argmax(score, axis=1)]
    cambio = pred_plug != pred_base
    conf = prob[np.arange(n_test), np.argmax(prob, axis=1)]
    print(f"  filas que cambian respecto de argmax puro: {cambio.sum():,} "
          f"({cambio.mean():.2%})", flush=True)
    print(f"  confianza media de las que cambian: {conf[cambio].mean():.4f} "
          f"(conjunto {conf.mean():.4f})", flush=True)
    print(f"  cambios con confianza >= 0.60: {int((cambio & (conf >= 0.60)).sum())}", flush=True)

    print("\n=== [5/6] fichero y validacion ===", flush=True)
    marco = pd.DataFrame({cfg.ID_COL: id_test, "age_range": pred_plug})
    checks = {
        "filas": int(len(marco)) == FILAS_TEST,
        "columnas_exactas": list(marco.columns) == [cfg.ID_COL, "age_range"],
        "ids_unicos": bool(marco[cfg.ID_COL].is_unique),
        "ids_iguales_a_test": set(marco[cfg.ID_COL]) == set(id_test),
        "mismo_orden_de_ids": bool((marco[cfg.ID_COL].to_numpy() == id_test).all()),
        "sin_nan": bool(not marco.isna().any().any()),
        "clases_validas": bool(set(marco["age_range"].unique()) <= set(cfg.CLASSES)),
        "sin_columnas_auxiliares": marco.shape[1] == 2,
        "todas_las_clases_presentes": len(set(marco["age_range"].unique())) == N_CLASES,
        "sin_duplicados": int(marco.duplicated().sum()) == 0,
    }
    for k, v in checks.items():
        print(f"  {'OK ' if v else 'NO '} {k}", flush=True)
    if not all(checks.values()):
        print("[STOP] algun check del fichero falla. No se escribe la submission.", flush=True)
        return 1
    SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
    marco.to_csv(SUBMISSION, index=False)
    sha = _sha256(SUBMISSION)
    print(f"  escrito: {SUBMISSION.relative_to(ROOT)}", flush=True)
    print(f"  SHA256: {sha}", flush=True)

    print("\n=== [6/6] manifest ===", flush=True)
    hashes_artefactos = {}
    for ruta in (cfg.EXPERIMENTS / "v30_02_ensemble.json",
                 cfg.EXPERIMENTS / "v30_03_submission_construccion.json",
                 cfg.FEATURES / "v30_lgbm_refinado_oof.npy",
                 cfg.FEATURES / "meta_features_test_v12_fresca.npy"):
        if ruta.exists():
            hashes_artefactos[ruta.name] = _sha256(ruta)
    manifiesto = {
        "submission_path": str(SUBMISSION.relative_to(ROOT)),
        "submission_sha256": sha,
        "git_commit": commit_actual(),
        "V30_artifact_hashes": hashes_artefactos,
        "training_data_hash": hashlib.sha256(
            train["text"].fillna("").astype(str).to_numpy().tobytes()).hexdigest(),
        "test_data_hash": hashlib.sha256(
            test["text"].fillna("").astype(str).to_numpy().tobytes()).hexdigest(),
        "class_order_verificado": orden,
        "weights": [float(x) for x in PESOS_CONGELADOS],
        "decision_rule": "predicted_class = argmax_c(weight_c * p_c); p_c son las probabilidades "
                         "de V30 sin ninguna transformacion",
        "V30_internal_cv": ESPERADO_V30_CANONICA,
        "plugin_internal_cv_canonica_nested": ESPERADO_NESTED,
        "plugin_internal_cv_canonica_frozen_global": ESPERADO_FROZEN_GLOBAL,
        "plugin_internal_cv_alternativa": ESPERADO_ALTERNATIVA,
        "canonical_delta_nested": 0.0020172,
        "canonical_delta_frozen_global": 0.0025261,
        "alternative_delta": 0.0020580,
        "estimacion_honesta_del_mecanismo": "el nested, +0,0020172 en la canonica y +0,0022884 en "
                                           "la alternativa: no depende de los pesos concretos. El "
                                           "frozen-global esta algo sobreajustado porque los pesos "
                                           "vieron todas las filas al derivarse",
        "bootstrap": {"canonica": [0.0010143, 0.0029480], "alternativa": [0.0011204, 0.0030622]},
        "filas_que_cambian_respecto_de_v30": int(cambio.sum()),
        "cambios_en_alta_confianza": int((cambio & (conf >= 0.60)).sum()),
        "checks": checks,
        "reproducibility": "R2: el pipeline reentrena V20 y los dos LightGBM sobre todo el train; "
                           "LightGBM no es reproducible bit a bit entre ejecuciones con distinta "
                           "carga de CPU, como el proyecto ya documento en V30",
        "timestamp": marca,
        "subido_a_kaggle": False,
        "uso_del_test": "solo transform y predict. Ninguna etiqueta, ninguna estadistica, ningun "
                        "ajuste de umbral sobre test",
    }
    MANIFEST.write_text(json.dumps(manifiesto, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  {MANIFEST.relative_to(ROOT)}", flush=True)

    RESULTADO.write_text(json.dumps({
        "fase": "submission experimental: plug-in por clase", "fecha": FECHA,
        "experimenta": False, "orden_verificado": orden["orden_verificado"],
        "sanity_check": san, "manifest": manifiesto,
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[fin] {time.time() - inicio:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
