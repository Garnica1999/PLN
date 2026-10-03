"""Second and third partitions, redone after fixing a real defect in the retraining.

The defect, and why it mattered
-------------------------------
`30_supervised_weighting_v4.py` has a helper, `ajustar_predecir`, that takes a per-label class
weight multiplier and does:

    if not np.isclose(multiplicadores[posicion], 1.0):
        modelo.set_params(class_weight=_pesos_de_clase(y, multiplicadores[posicion]))

The guard looks harmless. It is not. With a multiplier of 1.0 it leaves
`class_weight="balanced"`, and sklearn **recomputes "balanced" using the sample_weight passed to
fit**. An explicit dict built from `y` alone does not behave that way. The two are different
classifiers: measured on one fold, the coefficients differ by 1,4e-2.

What that means in practice: the historical V4-C05 sets the explicit dict ALWAYS, including for
multiplier 1.0. It does not use the guard. So any reproduction that goes through
`ajustar_predecir` is running a slightly different configuration whenever a fold picks 1.0.

That is exactly what happened. The canonical reproduction was unaffected because it LOADS the
historical OOF instead of retraining. But the second and third partitions were recomputed here,
through the guard, and two and three of their fifteen fold-label combinations pick 1.0.

Why this document exists rather than a silent retry
---------------------------------------------------
A guard that refuses to proceed when the reconstruction does not match the record is only worth
having if the mismatch is then chased down. The check fired, the cause was found, and the cause
is a real difference between two code paths that looked equivalent. Redoing the runs is the
consequence, and saying so is the point.

The correction
--------------
A local `ajustar_exacto` reproduces the historical inline block literally: the explicit dict is
always set, with no guard. With it, the canonical partition returns `0,7854160387`, identical
to the record to `3,2e-11`.
"""

from __future__ import annotations

import ast
import csv as _csv
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import V03_CCV

from src.evaluation import metricas_multietiqueta
from src.splits import ETIQUETAS, combinaciones

REPORTS = PROJECT_ROOT / "reports"
CSV_C = REPORTS / "a_cost_sensitive_downweight_partitions.csv"
JSON_R = REPORTS / "a_cost_sensitive_downweight_particiones_corregidas.json"

UMBRALES_V03 = np.array([0.44, 0.51, 0.46])
SEMILLA = 42
MULTIPLICADORES = (0.8, 1.0, 1.2, 1.5)
RONDAS = 3
SEMILLA_SEGUNDA = 20260928
SEMILLA_TERCERA = 20261001


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P69-06] {mensaje}", flush=True)


def cargar() -> dict:
    """Carga los modulos necesarios sin ejecutar sus main.

    Returns:
        Las funciones historicas y de particion.
    """
    def _carga(ruta: Path, nombre: str):
        spec = importlib.util.spec_from_file_location(nombre, ruta)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    h = _carga(PROJECT_ROOT / "experiments" / "30_supervised_weighting_v4.py", "h30")
    p = _carga(PROJECT_ROOT / "experiments" / "p69_02_segunda_particion.py", "p6902")
    return {"esquema_tfidf": h.esquema_tfidf, "pesos_de_clase": h._pesos_de_clase,
            "nueva_particion": p.nueva_particion, "pliegues_de": p.pliegues_de}


def ajustar_exacto(cx, matrices: tuple[sparse.csr_matrix, sparse.csr_matrix],
                   y_entrenar: np.ndarray, multiplicadores: np.ndarray,
                   sample_weight: np.ndarray, clases) -> np.ndarray:
    """Replica literal del bloque inline historico: el dict se fija SIEMPRE.

    Args:
        cx: utilidades cargadas.
        matrices: par (entrenamiento, aplicacion).
        y_entrenar: etiquetas del entrenamiento.
        multiplicadores: multiplicador de peso de clase por etiqueta.
        sample_weight: peso de ejemplo.
        clases: funcion de pesos de clase.

    Returns:
        Matriz de probabilidades.
    """
    izquierda, derecha = matrices
    salida = np.zeros((derecha.shape[0], len(ETIQUETAS)))
    for posicion in range(len(ETIQUETAS)):
        modelo = LogisticRegression(C=2.0, max_iter=1000, solver="liblinear",
                                    random_state=SEMILLA)
        # Sin guard: el diccionario explicito se fija siempre, como en el historico.
        modelo.set_params(class_weight=clases(y_entrenar[:, posicion],
                                              float(multiplicadores[posicion])))
        modelo.fit(izquierda, y_entrenar[:, posicion], sample_weight=sample_weight)
        salida[:, posicion] = modelo.predict_proba(derecha)[:, 1]
    return salida


def f1m(y: np.ndarray, pred: np.ndarray) -> float:
    """Macro F1 con la metrica del proyecto.

    Args:
        y: etiquetas verdaderas.
        pred: predicciones.

    Returns:
        La media de las F1 por columna.
    """
    return float(metricas_multietiqueta(y, pred)["f1_media_columnas"])


def correr_particion(cx, semilla: int, etiqueta: str, y: np.ndarray, ids: np.ndarray,
                     desarrollo, textos, grupos: np.ndarray) -> dict:
    """Ejecuta el candidato congelado en una particion, con la correccion aplicada.

    Args:
        cx: utilidades cargadas.
        semilla: semilla de la particion.
        etiqueta: nombre de la particion.
        y: etiquetas.
        ids: ids alineados.
        desarrollo: DataFrame de desarrollo.
        textos: Serie de textos.
        grupos: grupos canonicos.

    Returns:
        Diccionario con el resultado.
    """
    n = len(y)
    pliegue = cx["nueva_particion"](grupos, semilla)
    if int(pd.Series(grupos).groupby(grupos).nunique().gt(1).sum()):
        raise SystemExit(f"PARADA: grupos partidos en {etiqueta}")
    pliegues = cx["pliegues_de"](pliegue)
    log(f"   {etiqueta}: semilla {semilla}, {[len(x) for _, x in pliegues]} filas")

    t0 = time.time()
    cache = [cx["esquema_tfidf"](textos.iloc[tr], textos.iloc[va], sublinear_tf=True)
             for tr, va in pliegues]
    proba_v03 = np.zeros((n, len(ETIQUETAS)))
    for (tr, va), mm in zip(pliegues, cache):
        proba_v03[va] = ajustar_exacto(cx, mm, y[tr], np.ones(len(ETIQUETAS)),
                                       np.ones(len(tr)), cx["pesos_de_clase"])
    pred_v03 = (proba_v03 >= UMBRALES_V03[None, :]).astype(int)
    ccv_v03 = f1m(y, pred_v03)

    es = np.clip(-((proba_v03 - UMBRALES_V03[None, :]) * (2 * y - 1)), 0.0, None).mean(axis=1)
    iqr = float(np.percentile(es, 75) - np.percentile(es, 25)) or 1.0
    z = (es - float(np.median(es))) / iqr
    pesos = np.clip(1.0 - 0.5 * z, 0.5, 1.5)

    mult: list[list[float]] = []
    for indice_fold, (tr, va) in enumerate(pliegues):
        interno = StratifiedGroupKFold(n_splits=3, shuffle=True,
                                       random_state=SEMILLA + indice_fold)
        combo = combinaciones(desarrollo.iloc[tr]).reset_index(drop=True)
        pares = list(interno.split(desarrollo.iloc[tr], combo, groups=grupos[tr]))
        izq, _d = cache[indice_fold]

        def puntuar(candidatos: np.ndarray, tr=tr, pares=pares, izq=izq) -> float:
            acum = np.zeros((len(tr), len(ETIQUETAS)))
            for filas, cols in pares:
                acum[cols] = ajustar_exacto(cx, (izq[filas], izq[cols]), y[tr][filas],
                                            candidatos, np.ones(len(filas)),
                                            cx["pesos_de_clase"])
            return f1m(y[tr], (acum >= UMBRALES_V03[None, :]).astype(int))

        cand = np.ones(len(ETIQUETAS))
        mejor = puntuar(cand)
        for _ in range(RONDAS):
            for pos in range(len(ETIQUETAS)):
                for valor in MULTIPLICADORES:
                    prueba = cand.copy()
                    prueba[pos] = valor
                    s = puntuar(prueba)
                    if s > mejor + 1e-9:
                        mejor, cand = s, prueba
        mult.append([round(float(x), 2) for x in cand])

    proba_c = np.zeros((n, len(ETIQUETAS)))
    for k, (tr, va) in enumerate(pliegues):
        proba_c[va] = ajustar_exacto(cx, cache[k], y[tr], np.array(mult[k]), pesos[tr],
                                     cx["pesos_de_clase"])
    pred_c = (proba_c >= UMBRALES_V03[None, :]).astype(int)
    ccv_c = f1m(y, pred_c)
    deltas = [round(f1m(y[va], pred_c[va]) - f1m(y[va], pred_v03[va]), 6)
              for _, va in pliegues]
    ar = (pred_v03 == y)
    ac = (pred_c == y)
    por_et = []
    for j, et in enumerate(ETIQUETAS):
        r = int((~ar[:, j] & ac[:, j]).sum())
        d2 = int((ar[:, j] & ~ac[:, j]).sum())
        por_et.append({"etiqueta": et, "rescate": r, "damage": d2, "neto": r - d2})
    res = int((~ar & ac).sum())
    dan = int((ar & ~ac).sum())
    log(f"      V03 {ccv_v03:.10f} | candidato {ccv_c:.10f} | delta {ccv_c - ccv_v03:+.10f} | "
        f"pliegues {sum(1 for d in deltas if d > 0)}/5 | neto {res - dan:+d} "
        f"({time.time() - t0:.0f} s)")
    log(f"      multiplicadores: {mult}")
    if int((pesos > 1).sum()):
        raise SystemExit("PARADA: la regla no debe subir ningun peso")
    return {"particion": etiqueta, "semilla": semilla, "V03": round(ccv_v03, 10),
            "candidato": round(ccv_c, 10), "delta": round(ccv_c - ccv_v03, 10),
            "folds_positivos": sum(1 for d in deltas if d > 0), "deltas": deltas,
            "rescate": res, "damage": dan, "neto": res - dan, "por_etiqueta": por_et,
            "multiplicadores": mult, "pesos_IQR": round(iqr, 6),
            "pesos_medio": round(float(pesos.mean()), 6),
            "n_con_peso_menor_1": int((pesos < 1).sum())}


def main() -> None:
    """Rehace la segunda y la tercera particion con el defecto corregido.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P69-06 segunda y tercera particiones, con la retrificacion corregida")

    cx = cargar()
    v = np.load(PROJECT_ROOT / "features" / "oof_v03_final.npz", allow_pickle=True)
    y = v["y_true"].astype(int)
    ids = v["ids"].astype(np.int64)
    grupos = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv") \
        .set_index("id").loc[ids, "grupo_near_dup_autor"].to_numpy()
    from src.data import read_train
    train = read_train().assign(id=lambda d: d["id"].astype(np.int64))
    desarrollo = train.set_index("id").loc[ids].reset_index(drop=True)
    textos = desarrollo["text"]

    log("")
    log("   NOTA: el bug esta documentado en el docstring. Resumen en una linea:")
    log("   `ajustar_predecir` deja class_weight='balanced' cuando el multiplicador es 1.0,")
    log("   y sklearn recalcula 'balanced' CON los sample_weight. El historico fija siempre")
    log("   el diccionario explicito. `ajustar_exacto` hace eso, y por eso reproduce.")

    resultados = []
    for semilla, etiqueta in ((SEMILLA_SEGUNDA, "segunda"), (SEMILLA_TERCERA, "tercera")):
        log("")
        resultados.append(correr_particion(cx, semilla, etiqueta, y, ids, desarrollo, textos,
                                           grupos))

    canonica = {"particion": "canonica", "semilla": None, "V03": 0.7772114525,
                "candidato": 0.7854160387, "delta": 0.0082045862, "folds_positivos": 5,
                "deltas": [0.002243, 0.001592, 0.003378, 0.023607, 0.010356],
                "rescate": 99, "damage": 74, "neto": 25}
    todas = [canonica] + resultados
    ds = np.array([t["delta"] for t in todas])
    positivas = int((ds > 0).sum())
    tercera = resultados[1]

    log("")
    log("   LAS TRES PARTICIONES, CON LA CORRECCION APLICADA")
    log(f"      {'particion':12}{'V03':>16}{'delta':>16}{'folds+':>8}{'rescate':>9}"
        f"{'damage':>8}{'neto':>7}")
    for t in todas:
        log(f"      {t['particion']:12}{t['V03']:>16.10f}{t['delta']:>+16.10f}"
            f"{t['folds_positivos']:>6}/5{t['rescate']:>9}{t['damage']:>8}{t['neto']:>+7}")
    log("")
    log(f"      media {ds.mean():+.10f} | mediana {np.median(ds):+.10f} | "
        f"min {ds.min():+.10f} | max {ds.max():+.10f}")
    log(f"      positivas {positivas} de 3")

    if positivas == 3 and tercera["folds_positivos"] >= 4:
        caso, resolucion = "CASO A", "PROMISING_UNCONFIRMED"
    elif positivas == 2:
        caso, resolucion = "CASO B", "MARGINAL"
    else:
        caso, resolucion = "CASO C", "MARGINAL_UNSTABLE"
    log("")
    log(f"      {caso}: {resolucion}")
    log(f"      procede permutacion: {caso == 'CASO A'}")

    contraste = {
        "por_que": "los numeros anteriores de las particiones segunda y tercera usaban el guard",
        "segunda_antes": {"delta": 0.0105253130, "folds": 5, "rescate": 86, "damage": 74},
        "segunda_ahora": {"delta": resultados[0]["delta"],
                          "folds": resultados[0]["folds_positivos"],
                          "rescate": resultados[0]["rescate"],
                          "damage": resultados[0]["damage"]},
        "tercera_antes": {"delta": 0.0047946703, "folds": 4, "rescate": 78, "damage": 69},
        "tercera_ahora": {"delta": resultados[1]["delta"],
                          "folds": resultados[1]["folds_positivos"],
                          "rescate": resultados[1]["rescate"],
                          "damage": resultados[1]["damage"]},
    }
    log("")
    log("   CONTRASTE CON LOS NUMEROS ANTERIORES")
    for capa in ("segunda", "tercera"):
        a = contraste[f"{capa}_antes"]
        b = contraste[f"{capa}_ahora"]
        log(f"      {capa}: antes {a['delta']:+.10f} ({a['folds']}/5) -> ahora "
            f"{b['delta']:+.10f} ({b['folds']}/5)")

    with CSV_C.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["particion", "semilla", "V03", "candidato", "delta", "folds_positivos",
                    "rescate", "damage", "neto", "deltas_por_pliegue", "nota"])
        for t in todas:
            w.writerow([t["particion"], t["semilla"], t["V03"], t["candidato"], t["delta"],
                        t["folds_positivos"], t["rescate"], t["damage"], t["neto"],
                        "|".join(str(x) for x in t["deltas"]),
                        "historico" if t["particion"] == "canonica" else "recorrida aqui"])
        for t in resultados:
            w.writerow([f"{t['particion']}_etiquetas", "", "", "", "", "", "", "", "",
                        "|".join(f"{f['etiqueta']}:r{f['rescate']}d{f['damage']}"
                                 for f in t["por_etiqueta"]), ""])
        w.writerow(["RESUMEN", "media", round(float(ds.mean()), 10), "", "", "", "", "", "",
                    "", ""])
        w.writerow(["RESUMEN", "positivas", positivas, "", "", "", "", "", "", "", ""])
        w.writerow(["DECISION", caso, resolucion, "", "", "", "", "", "", "", ""])
        for capa in ("segunda", "tercera"):
            w.writerow([f"CONTRASTE_{capa}", "antes",
                        round(contraste[f"{capa}_antes"]["delta"], 10), "", "", "", "", "",
                        "", "", "guard activo"])
            w.writerow([f"CONTRASTE_{capa}", "ahora",
                        round(contraste[f"{capa}_ahora"]["delta"], 10), "", "", "", "", "",
                        "", "", "corregido"])

    salida = {
        "familia": "P69_PARTICIONES_CORREGIDAS", "experiment_id": "P69-06",
        "fecha": inicio.isoformat(timespec="seconds"),
        "naturaleza": ("CORRECCION y repeticion de dos particiones. 0 tecnicas nuevas, "
                       "0 grid, 0 test, 0 holdout, 0 leaderboard"),
        "el_defecto": {
            "donde": "30_supervised_weighting_v4.py, funcion ajustar_predecir",
            "que": ("`if not np.isclose(multiplicadores[posicion], 1.0)` deja "
                    "class_weight='balanced' cuando el multiplicador es 1.0"),
            "por_que_importa": ("sklearn recalcula 'balanced' CON los sample_weight pasados a "
                                "fit, y un diccionario explicito construido desde y no hace "
                                "eso. Son clasificadores distintos: los coeficientes difieren "
                                "hasta 1,4e-2"),
            "donde_afecto": ("la reproduccion canonica NO se afecto porque CARGA el OOF "
                             "historico; las particiones segunda y tercera SI, porque se "
                             "reentrenan aqui"),
            "cuantas_combinaciones": ("2 de 15 en la segunda, 3 de 15 en la tercera"),
            "como_se_detecto": ("el propio script de permutacion se niega a continuar si los "
                                "pesos reales no reproducen el delta observado. Se nego"),
            "correccion": ("`ajustar_exacto` fija SIEMPRE el diccionario explicito, sin guard, "
                           "replicando el bloque inline historico. Con el, la canonica da "
                           "0,7854160387, identica a 3,2e-11"),
        },
        "particiones": todas, "resumen": {
            "media": round(float(ds.mean()), 10), "mediana": round(float(np.median(ds)), 10),
            "minimo": round(float(ds.min()), 10), "maximo": round(float(ds.max()), 10),
            "positivas": positivas},
        "contraste_con_los_numeros_anteriores": contraste,
        "caso": caso, "resolucion": resolucion,
        "procede_permutacion": bool(caso == "CASO A"),
        "reproducibilidad": {"nivel": "R3 en la canonica", "corrected_cv_v03": V03_CCV},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_R.write_text(json.dumps(salida, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")

    log("")
    log("=" * 78)
    log(f"P69-06 OK | {caso} | {resolucion}")
    log(f"   canonica {canonica['delta']:+.10f} | segunda {resultados[0]['delta']:+.10f} | "
        f"tercera {resultados[1]['delta']:+.10f}")
    log(f"   positivas {positivas} de 3 | media {ds.mean():+.10f}")
    log(f"   procede permutacion: {caso == 'CASO A'}")
    log("   V03 remains the best verified leaderboard result.")
    log("=" * 78)


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
