"""Validacion confirmatoria: el plug-in por clase bajo la particion alternativa CONGELADA.

## Lo que hace, y lo que no

Responde una sola pregunta:

> ¿la mejora de +0,0020172 observada en la particion canonica se mantiene cuando cambia el reparto
> de entrenamiento y validacion?

**No** reajusta los pesos. **No** busca otra rejilla. **No** elige otra semilla. **No** ejecuta una
tercera particion. **No** toca el test — ni para contar filas. **No** consulta el leaderboard.

Los pesos estan congelados en `PESOS_CONGELADOS`, copiados literalmente del pre-registro
commit `bd08aba`, y este script **no contiene ningun optimizador**: solo aplica.

## Los tres numeros que se miden

```text
congelado   los pesos congelados sobre la particion alternativa   <- decide
nested      pesos re-derivados dentro de la particion alternativa <- test limpio del MECANISMO
            (cada fold los ajusta en los otros cuatro; el optimizador vive SOLO aqui)
```

El `nested` existe para poder interpretar el `congelado`. Como los pesos congelados vieron las
169.336 filas al derivarse (su union cubre el train completo), el `congelado` no es una evaluacion
fuera de muestra para ellos. Si el `nested` tambien sale positivo, el **mecanismo** aparece solo en
la particion nueva, y eso es lo que le da valor a la confirmacion. Si el `nested` sale negativo, la
diferencia es de pesos y hay que decirlo.

## La particion, y por que es esa semilla

Se usa la **semilla 987654**, que es la que uso V30-05 para su segunda particion. No se elige: se
hereda, para que nadie pueda acusar al resultado de venir de una semilla escogida. V20 y los dos
LightGBM se **reentrenan desde cero** sobre los folds nuevos, con las configuraciones **congeladas**
de V30 (sin reoptimizar), y se mezclan con los pesos medios del manifest de V30.

**Consecuencia que se declara**: igual que en V30-05, esta "segunda particion" usa los **mismos
grupos** `dup_clusters` repartidos de otra forma, asi que solapa parcialmente con la canonica. Es
una reejecucion, no una particion independiente. Es la particion que el proyecto tiene definida y
la que el brief pide; el limite se documenta y no se disimula.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

RESULTADO = cfg.EXPERIMENTS / "age_plugin_confirmacion.json"
PARTICIONES_CSV = cfg.REPORTS / "age_plugin_macro_f1_partitions.csv"
OOF_GUARDADO = cfg.FEATURES / "age_plugin_segunda_particion_oof.npy"
FECHA = "2026-10-11"
CV_V30 = 0.6134471
N_CLASES = 6
SEMILLA_PARTICION = 987654
SEMILLA_BOOTSTRAP = 20261010
UMBRAL_ALTA = 0.60
VARIABILIDAD = 0.0008876
N_BOOT = 2000
HILOS = 8
#: CONGELADOS del pre-registro `bd08aba`. No se recalculan en ningun punto de este script.
PESOS_CONGELADOS = np.array([1.0000, 1.1749, 1.2721, 1.1630, 1.0010, 0.8128])
PESOS_V30 = {"v20_f3b_oof_candidato": 0.3548, "v29_lgbm_optuna": 0.3354,
             "v30_lgbm_refinado": 0.3098}
REJILLA = np.exp(np.linspace(-0.7, 0.7, 29))


def _macro(clase, prob) -> float:
    return float(f1_score(clase, np.argmax(prob, axis=1), average="macro",
                          labels=list(range(N_CLASES))))


def optimiza_pesos(clase, prob, idx) -> np.ndarray:
    """Unico lugar del script donde se optimiza algo. Se usa SOLO para el diagnostico `nested`."""
    pesos = np.ones(N_CLASES)
    mejor = _macro(clase[idx], prob[idx] * pesos)
    for _ in range(8):
        mejoro = False
        for c in range(1, N_CLASES):
            base = pesos[c]
            for f in REJILLA:
                pesos[c] = base * f
                valor = _macro(clase[idx], prob[idx] * pesos)
                if valor > mejor + 1e-9:
                    mejor, base, mejoro = valor, pesos[c], True
            pesos[c] = base
        if not mejoro:
            break
    return pesos


def main() -> int:
    """Reentrena V30 en la particion alternativa y aplica la regla congelada."""
    autotest_numerico()
    inicio = time.time()
    from v30_05_segunda_particion import (_configs_del_miembro, _cuartos_v20,
                                          _normaliza, a_clases)
    from v19_b1_boosting import matriz_279

    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(v) for v in y])
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    matriz = matriz_279()
    print(f"=== particion alternativa: semilla {SEMILLA_PARTICION} (heredada de V30-05) ===",
          flush=True)

    folds = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group",
                             seed=SEMILLA_PARTICION)
    canonica = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")
    solapes = [len(set(np.asarray(a["val_idx"])) & set(np.asarray(b["val_idx"])))
               for a, b in zip(canonica, folds)]
    print(f"  solapes con la canonica por fold: {solapes}", flush=True)
    print(f"  particion realmente distinta: {any(s > 0 for s in solapes)} "
          f"(solapa, luego es reejecucion y no particion independiente)", flush=True)

    configs = {n: _configs_del_miembro(n) for n in PESOS_V30
               if n != "v20_f3b_oof_candidato"}
    import lightgbm as lgb
    oof = np.zeros((len(clase), N_CLASES))
    for fold in folds:
        idx = int(fold["fold"])
        val = np.asarray(fold["val_idx"])
        tren = np.concatenate([np.asarray(o["val_idx"]) for o in folds
                               if int(o["fold"]) != idx])
        prob_v20 = _cuartos_v20(matriz, clase, y, tren, val)
        miembros = {"v20_f3b_oof_candidato": prob_v20}
        for nombre, lista in configs.items():
            p = {"objective": "multiclass", "num_class": N_CLASES, "verbose": -1,
                 "n_jobs": HILOS, "subsample_freq": 1, **_normaliza(lista[idx])}
            techo = int(p.pop("n_estimators"))
            modelo = lgb.LGBMClassifier(**p, n_estimators=techo).fit(matriz[tren], y[tren])
            miembros[nombre] = a_clases(modelo.predict_proba(matriz[val]), modelo.classes_,
                                        len(val))
        nombres = list(miembros)
        vector = np.array([PESOS_V30.get(n, 0.0) for n in nombres], dtype=float)
        vector = vector / vector.sum()
        oof[val] = np.tensordot(vector, np.stack([miembros[n] for n in nombres]), axes=(0, 0))
        print(f"  fold {idx} hecho ({time.time() - inicio:.0f} s)", flush=True)

    np.save(OOF_GUARDADO, oof)
    print(f"\n  OOF de la particion alternativa guardado en {OOF_GUARDADO.name}", flush=True)
    prob = np.clip(oof, 1e-12, 1.0)
    prob = prob / prob.sum(axis=1, keepdims=True)
    cv_v30 = _macro(clase, prob)
    print(f"  V30 en la particion alternativa: {cv_v30:.7f}", flush=True)

    # Una mascara booleana por fold, construida una sola vez.
    mascaras = []
    for k in range(5):
        m = np.zeros(len(clase), dtype=bool)
        for f in folds:
            if int(f["fold"]) == k:
                m[np.asarray(f["val_idx"])] = True
        mascaras.append(m)

    rng = np.random.default_rng(SEMILLA_BOOTSTRAP)
    base_pred = np.argmax(prob, axis=1)
    filas = []
    brazos = {}
    # CONGELADO: los pesos del pre-registro, sin tocar.
    brazos["congelado"] = prob * PESOS_CONGELADOS
    # CONTROLES, con el mismo protocolo que la canonica.
    brazos["uniforme"] = prob.copy()
    brazos["permutado"] = prob * PESOS_CONGELADOS[rng.permutation(N_CLASES)]
    brazos["aleatorio"] = prob * np.exp(rng.normal(0, 0.19, size=N_CLASES))
    # DIAGNOSTICO nested: el optimizador solo vive aqui.
    nested = np.zeros_like(prob)
    pesos_nested = []
    for k in range(5):
        # `tr` y `va` son indices de FILA del dataset, no posiciones dentro del fold.
        tr = np.concatenate([np.asarray(f["val_idx"]) for f in folds if int(f["fold"]) != k])
        va = np.concatenate([np.asarray(f["val_idx"]) for f in folds if int(f["fold"]) == k])
        w = optimiza_pesos(clase, prob, tr)
        pesos_nested.append([round(float(x), 4) for x in w])
        nested[va] = prob[va] * w
    brazos["nested"] = nested

    for nombre, b in brazos.items():
        cv = _macro(clase, b)
        por_fold, base_fold = [], []
        for k in range(5):
            m = mascaras[k]
            por_fold.append(_macro(clase[m], b[m]))
            base_fold.append(_macro(clase[m], prob[m]))
        pred = np.argmax(b, axis=1)
        ok_c, ok_v = pred == clase, base_pred == clase
        filas.append({
            "brazo": nombre, "corrected_cv": round(cv, 7), "delta_vs_v30": round(cv - cv_v30, 7),
            "folds": [round(v, 6) for v in por_fold],
            "folds_positivos": int(sum(1 for a, bb in zip(por_fold, base_fold) if a > bb)),
            "rescate": int((~ok_v & ok_c).sum()), "danio": int((ok_v & ~ok_c).sum()),
            "neto": int((~ok_v & ok_c).sum()) - int((ok_v & ~ok_c).sum()),
            "filas_que_cambian": int((pred != base_pred).sum()),
            "accuracy": round(float(accuracy_score(clase, pred)), 5),
            "ratio_sobre_variabilidad": round(abs(cv - cv_v30) / VARIABILIDAD, 3),
        })
    tabla = pd.DataFrame(filas)
    tabla.to_csv(PARTICIONES_CSV, index=False)
    print("\n=== resultados en la particion alternativa ===", flush=True)
    for _, f in tabla.iterrows():
        print(f"  {f['brazo']:10s} {f['corrected_cv']:.7f}  {f['delta_vs_v30']:+.7f}  "
              f"folds+ {f['folds_positivos']}/5  cambia {f['filas_que_cambian']:>6,d}  "
              f"acc {f['accuracy']:.5f}", flush=True)
    print("\n  pesos re-derivados en la particion alternativa (diagnostico):", flush=True)
    for k, w in enumerate(pesos_nested):
        print(f"    fold {k}: {w}", flush=True)

    d = {f["brazo"]: f["delta_vs_v30"] for f in filas}
    print(f"\n  congelado - permutado = {d['congelado'] - d['permutado']:+.7f}", flush=True)
    print(f"  congelado - aleatorio = {d['congelado'] - d['aleatorio']:+.7f}", flush=True)
    print(f"  congelado - nested    = {d['congelado'] - d['nested']:+.7f}", flush=True)

    # Alta confianza, el diagnostico que se repite.
    conf = prob[np.arange(len(clase)), base_pred]
    alta = conf >= UMBRAL_ALTA
    cambia = np.argmax(brazos["congelado"], axis=1) != base_pred
    n_alta = int((alta & cambia).sum())
    print("\n=== alta confianza ===", flush=True)
    print(f"  filas que cambian: {cambia.sum():,} | de alta confianza: {n_alta:,}", flush=True)
    print(f"  confianza media de las que cambian: {conf[cambia].mean():.4f} "
          f"(conjunto {conf.mean():.4f})", flush=True)

    fila_c = tabla[tabla["brazo"] == "congelado"].iloc[0]
    criterio = bool(fila_c["delta_vs_v30"] >= 0.0015 and fila_c["folds_positivos"] >= 4
                    and d["congelado"] > d["permutado"] and d["congelado"] > d["aleatorio"])
    if criterio:
        veredicto = "PROMISING_CONFIRMED"
    elif fila_c["delta_vs_v30"] > 0 and d["nested"] > 0:
        veredicto = "MARGINAL"
    elif fila_c["delta_vs_v30"] > 0 and d["nested"] <= 0:
        veredicto = "UNSTABLE"
    elif d["congelado"] > d["permutado"] and d["congelado"] > d["aleatorio"]:
        veredicto = "MARGINAL"
    else:
        veredicto = "CLOSED"
    print("\n=== criterio pre-registrado ===", flush=True)
    print(f"  delta >= +0,0015: {fila_c['delta_vs_v30'] >= 0.0015} "
          f"({fila_c['delta_vs_v30']:+.7f})", flush=True)
    print(f"  folds >= 4/5: {fila_c['folds_positivos'] >= 4} ({fila_c['folds_positivos']}/5)",
          flush=True)
    print(f"  real > permutado: {d['congelado'] > d['permutado']}", flush=True)
    print(f"  real > aleatorio: {d['congelado'] > d['aleatorio']}", flush=True)
    print(f"\n=== {veredicto} ===", flush=True)

    RESULTADO.write_text(json.dumps({
        "fase": "validacion confirmatoria del plug-in por clase", "fecha": FECHA,
        "experimenta": True, "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT.parent, capture_output=True, text=True,
            timeout=30).stdout.strip(),
        "pre_registro": "commit bd08aba",
        "particion_alternativa": {"semilla": SEMILLA_PARTICION, "solapes_con_la_canonica": solapes,
                                  "es_reejecucion_no_independiente": True},
        "pesos_congelados": [float(x) for x in PESOS_CONGELADOS],
        "v30_en_la_particion_alternativa": round(cv_v30, 7),
        "resultados": tabla.to_dict("records"),
        "pesos_nested_diagnostico": pesos_nested,
        "alta_confianza": {"n_cambian": int(cambia.sum()), "n_alta_que_cambian": n_alta,
                           "confianza_media_de_las_que_cambian": round(float(conf[cambia].mean()), 5)
                           if cambia.any() else None},
        "criterio_cumplido": criterio, "veredicto": veredicto,
        "restricciones": {"usa_test": False, "usa_holdout": False, "usa_leaderboard": False,
                          "reajusta_pesos_congelados": False, "genera_submission": False,
                          "proyecta_score": False, "rejilla_adicional": False},
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[fin] {time.time() - inicio:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
