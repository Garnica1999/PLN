"""V56-00 (P56): transferencia de umbrales con anidamiento completo.

Que resuelve
------------
P55 encontro un defecto procedimental: los umbrales de V03 (0,44 / 0,51 / 0,46) se
adoptaron sobre un OOF generado con el protocolo D-004, que deja que un pseudo-autor
cruce pliegues, mientras que el ``corrected_cv`` se reporta con el protocolo D-016.
Medido con umbrales anidados, V03 queda **0,03 por encima** de lo que pide D-016 en HS
y TR, en los cinco pliegues.

La mejora medida en P55 fue de **+0,000925**, por debajo del margen de adopcion. Pero
eso no basta para cerrar el mecanismo, porque el procedimiento de P55 **no era un
anidamiento completo**: tomaba el umbral de las probabilidades out-of-fold de los
cuatro pliegues de entrenamiento, que proceden de modelos ya entrenados. Aqui se hace
el anidamiento que la directiva pide:

    outer_train
      -> inner split AGRUPADO dentro de outer_train
      -> OOF interior, con vectorizador y modelo re-ajustados
      -> ajuste de umbral sobre ese OOF interior
      -> umbral fijo
    outer_validation, que no participa en nada del ajuste

Ademas se evalua la regla de agregacion que la directiva fija, la **mediana** de las
cinco estimaciones, y solo esa.

Lo que NO se hace
-----------------
- No se genera ninguna submission.
- No se consulta Kaggle.
- No se cambia representacion, clasificador, C, ngrams ni class weights.
- No se usa holdout ni test.
- La unica variable es el procedimiento de estimacion de umbrales.

Uso
---
    /home/carlos/venv/bin/python experiments/v56_00_transferencia_umbrales.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train
from src.splits import grupos_fusionados, pliegues_corregidos

FEATURES_EXTERNOS = Path("/home/carlos/Git/PLN/kaggle_toxic_2026_puj/features")
OOF_D016 = FEATURES_EXTERNOS / "v2_oof_base_corrected.npz"

ETIQUETAS = ("HS", "TR", "AG")
N_FOLD = 5
N_INNER = 4
REJILLA = np.round(np.arange(0.20, 0.801, 0.01), 2)
V03_UMBRALES = {"HS": 0.44, "TR": 0.51, "AG": 0.46}
PARAMS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2,
               "sublinear_tf": True, "dtype": np.float64}
PARAMS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2,
               "sublinear_tf": True, "dtype": np.float64}
LR_PARAMS = {"C": 2.0, "class_weight": "balanced", "max_iter": 1000,
             "solver": "liblinear", "random_state": 42}
V03_CORRECTED_CV = 0.7772114525165721
P55_CORRECTED = 0.7781367700
TOLERANCIA = 1e-9

REPORTE_JSON = PROJECT_ROOT / "reports" / "p56_threshold_transfer.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "p56_threshold_transfer.md"
CSV_TRANSFER = PROJECT_ROOT / "reports" / "p56_threshold_transfer.csv"
CSV_ESTABILIDAD = PROJECT_ROOT / "reports" / "p56_threshold_stability.csv"


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V56-00] {mensaje}", flush=True)


def umbral_optimo(y: np.ndarray, p: np.ndarray) -> float:
    """Umbral que maximiza la F1 binaria, en la rejilla historica.

    Args:
        y: etiqueta verdadera.
        p: probabilidad de la clase positiva.

    Returns:
        El umbral de la rejilla que da la mayor F1.
    """
    mejor, mejor_f1 = float(REJILLA[0]), -1.0
    for u in REJILLA:
        f = float(f1_score(y, (p >= u).astype(int), zero_division=0))
        if f > mejor_f1:
            mejor, mejor_f1 = float(u), f
    return mejor


def matriz(texto_ent: pd.Series, texto_val: pd.Series) -> tuple[Any, Any]:
    """Construye las matrices TF-IDF ajustando SOLO con el texto de entrenamiento.

    Args:
        texto_ent: textos de entrenamiento.
        texto_val: textos de validacion.

    Returns:
        Tupla con la matriz de entrenamiento y la de validacion, ya unidas.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    vw = TfidfVectorizer(**PARAMS_WORD)
    Xw_e = vw.fit_transform(texto_ent)
    Xw_v = vw.transform(texto_val)
    vc = TfidfVectorizer(**PARAMS_CHAR)
    Xc_e = vc.fit_transform(texto_ent)
    Xc_v = vc.transform(texto_val)
    return (sparse.hstack([Xw_e, Xc_e]).tocsr(),
            sparse.hstack([Xw_v, Xc_v]).tocsr())


def main() -> None:
    """Ejecuta la evaluacion de la transferencia de umbrales.

    Returns:
        None.
    """
    inicio = time.time()
    log("P56-00 transferencia de umbrales con anidamiento completo. 0 submissions")

    d = np.load(OOF_D016, allow_pickle=True)
    ids = d["ids"].astype(int)
    y = np.asarray(d["y_true"], dtype=np.int64)
    modelos = [str(x) for x in d["modelos"]]
    proba_oof = d["proba"][:, modelos.index("M2_union_lr_c2"), :].astype(np.float64)

    tr = read_train()
    tr["id"] = tr["id"].astype("int64")
    part = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv")
    dev_ids = set(part.loc[part["particion"] == "dev", "id"].astype("int64"))
    dev = tr[tr["id"].isin(dev_ids)].reset_index(drop=True)
    if not np.array_equal(dev["id"].to_numpy(), ids):
        raise SystemExit("PARADA: el orden del dev no coincide con el OOF canonico")
    texto = dev["text"].reset_index(drop=True)
    grupos = grupos_fusionados(dev)
    asignacion = np.full(len(ids), -1, dtype=np.int64)
    for k, (_, val) in enumerate(pliegues_corregidos(dev, grupos)):
        asignacion[val] = k
    texto = texto.astype(str)
    log(f"   dev {len(dev)} filas | grupos {len(np.unique(grupos))} | "
        f"pliegues {[int((asignacion == v).sum()) for v in range(N_FOLD)]}")

    # ================================================ 1. REPRODUCIR P55
    log("1/5 reproduccion del resultado de P55")

    def anidado_simple(p: np.ndarray) -> np.ndarray:
        """Aplica el procedimiento de P55: umbral de los 4 pliegues, al quinto.

        Args:
            p: matriz de probabilidades out-of-fold.

        Returns:
            Matriz de predicciones.
        """
        pred = np.zeros_like(y)
        for v in range(N_FOLD):
            ent = asignacion != v
            val = asignacion == v
            u = [umbral_optimo(y[ent, j], p[ent, j]) for j in range(3)]
            pred[val] = (p[val] >= np.array(u)[None, :]).astype(int)
        return pred

    pred_p55 = anidado_simple(proba_oof)
    f1_p55 = float(np.mean([f1_score(y[:, j], pred_p55[:, j], zero_division=0)
                            for j in range(3)]))
    reproduce = abs(f1_p55 - P55_CORRECTED) < TOLERANCIA
    log(f"   P55 anidado simple: {f1_p55:.10f} | P55 registro {P55_CORRECTED:.10f}")
    log(f"   P55_CORRECTED_RESULT_REPRODUCED = {bool(reproduce)}")
    if not reproduce:
        log("   PARADA por tolerancia de reproduccion")
        raise SystemExit("P55 no se reproduce, se detiene la fase")

    # ================================================ 2. ANIDAMIENTO COMPLETO
    log("2/5 anidamiento completo: inner split agrupado con refit de vectorizador y modelo")
    filas_fold = []
    umbrales_por_fold: list[dict[str, float]] = []
    for v in range(N_FOLD):
        outer_ent = np.where(asignacion != v)[0]
        outer_val = np.where(asignacion == v)[0]
        g_outer = grupos[outer_ent]
        y_outer = y[outer_ent]
        # Estratificacion multietiqueta por la combinacion de las tres etiquetas.
        estrato = np.array(["".join(map(str, fila)) for fila in y_outer])
        inner = StratifiedGroupKFold(n_splits=N_INNER, shuffle=True, random_state=42)
        proba_inner = np.zeros((len(outer_ent), 3), dtype=np.float64)
        for k, (idx_e, idx_v) in enumerate(inner.split(outer_ent, estrato, g_outer), 1):
            Xe, Xv = matriz(texto.iloc[outer_ent[idx_e]], texto.iloc[outer_ent[idx_v]])
            for j in range(3):
                modelo = LogisticRegression(**LR_PARAMS)
                modelo.fit(Xe, y_outer[idx_e, j])
                proba_inner[idx_v, j] = modelo.predict_proba(Xv)[:, 1]
        u = {et: umbral_optimo(y_outer[:, j], proba_inner[:, j])
             for j, et in enumerate(ETIQUETAS)}
        umbrales_por_fold.append(u)
        filas_fold.append({
            "outer_fold": v, "n_outer_train": len(outer_ent),
            "n_outer_val": len(outer_val),
            "n_inner": N_INNER,
            **{f"inner_threshold_{et}": u[et] for et in ETIQUETAS},
            **{f"V03_{et}": V03_UMBRALES[et] for et in ETIQUETAS},
        })
        log(f"   outer {v}: train {len(outer_ent):4d} val {len(outer_val):4d} | "
            f"umbrales inner {u['HS']:.2f}/{u['TR']:.2f}/{u['AG']:.2f}")

    # Aplicacion por pliegue exterior, que es la estimacion plenamente honesta.
    pred_nested = np.zeros_like(y)
    for v in range(N_FOLD):
        val = asignacion == v
        u = umbrales_por_fold[v]
        pred_nested[val] = (proba_oof[val]
                            >= np.array([u[et] for et in ETIQUETAS])[None, :]).astype(int)
    f1_nested = float(np.mean([f1_score(y[:, j], pred_nested[:, j], zero_division=0)
                               for j in range(3)]))
    log(f"   macro-F1 con umbral POR PLIEGUE EXTERIOR: {f1_nested:.10f} | "
        f"delta vs V03 {f1_nested - V03_CORRECTED_CV:+.10f}")

    # ================================================ 3. DISTRIBUCION
    log("3/5 distribucion de umbrales y regla de agregacion")
    filas_est = []
    for et in ETIQUETAS:
        vals = np.array([u[et] for u in umbrales_por_fold], dtype=float)
        filas_est.append({
            "label": et, "n_estimaciones": int(vals.size),
            "media": round(float(vals.mean()), 6), "mediana": round(float(np.median(vals)), 6),
            "std": round(float(vals.std(ddof=1)), 6),
            "min": round(float(vals.min()), 6), "max": round(float(vals.max()), 6),
            "V03": V03_UMBRALES[et],
            "V03_dentro_del_rango": bool(vals.min() <= V03_UMBRALES[et] <= vals.max()),
            "desviacion_de_la_mediana": round(float(np.median(vals) - V03_UMBRALES[et]), 6),
            "estimaciones": " ".join(f"{v:.2f}" for v in vals),
        })
        log(f"   {et}: {filas_est[-1]['estimaciones']} | mediana "
            f"{np.median(vals):.2f} | std {vals.std(ddof=1):.4f} | V03 "
            f"{V03_UMBRALES[et]:.2f} | dentro del rango: "
            f"{filas_est[-1]['V03_dentro_del_rango']}")
    estdf = pd.DataFrame(filas_est)

    # T1: la MEDIANA de las cinco estimaciones. Una sola regla, segun la directiva.
    T1 = {x["label"]: x["mediana"] for _, x in estdf.iterrows()}
    log(f"   T1 (mediana de las cinco estimaciones) = "
        f"{T1['HS']:.2f}/{T1['TR']:.2f}/{T1['AG']:.2f}")
    log(f"   T0 (historico de V03)                   = "
        f"{V03_UMBRALES['HS']:.2f}/{V03_UMBRALES['TR']:.2f}/{V03_UMBRALES['AG']:.2f}")

    # ================================================ 4. T0 CONTRA T1
    log("4/5 T0 contra T1, por pliegue, por etiqueta y por celda")
    def evaluar(umb: dict[str, float]) -> dict[str, Any]:
        """Evalua un juego de umbrales sobre el OOF canonico.

        Args:
            umb: umbral por etiqueta.

        Returns:
            Diccionario con la metrica global, por pliegue y por etiqueta.
        """
        pred = (proba_oof >= np.array([umb[et] for et in ETIQUETAS])[None, :]).astype(int)
        f1 = float(np.mean([f1_score(y[:, j], pred[:, j], zero_division=0)
                            for j in range(3)]))
        por_pliegue = []
        for v in range(N_FOLD):
            sel = asignacion == v
            fv = float(np.mean([f1_score(y[sel, j], pred[sel, j], zero_division=0)
                                for j in range(3)]))
            por_pliegue.append({
                "pliegue": v,
                "macro_f1": round(fv, 10),
                "delta_vs_V03_global": round(fv - V03_CORRECTED_CV, 10),
            })
        return {
            "umbrales": umb,
            "macro_f1": round(f1, 10),
            "delta_vs_V03": round(f1 - V03_CORRECTED_CV, 10),
            "por_etiqueta": {et: round(float(f1_score(y[:, j], pred[:, j], zero_division=0)),
                                    10) for j, et in enumerate(ETIQUETAS)},
            "por_pliegue": por_pliegue,
            "tasa_positiva": {et: round(float(pred[:, j].mean()), 6)
                              for j, et in enumerate(ETIQUETAS)},
        }

    t0 = evaluar(V03_UMBRALES)
    t1 = evaluar(T1)
    t_nested = {
        "macro_f1": round(f1_nested, 10),
        "delta_vs_V03": round(f1_nested - V03_CORRECTED_CV, 10),
    }
    log(f"   T0 macro-F1 {t0['macro_f1']:.10f} (delta {t0['delta_vs_V03']:+.10f})")
    log(f"   T1 macro-F1 {t1['macro_f1']:.10f} (delta {t1['delta_vs_V03']:+.10f})")
    log("   por etiqueta T0 -> T1:")
    for et in ETIQUETAS:
        log(f"      {et}: {t0['por_etiqueta'][et]:.10f} -> {t1['por_etiqueta'][et]:.10f} "
            f"({t1['por_etiqueta'][et] - t0['por_etiqueta'][et]:+.10f})")
    log(f"   tasa positiva T0: {t0['tasa_positiva']}")
    log(f"   tasa positiva T1: {t1['tasa_positiva']}")

    pliegues_positivos = sum(1 for x in t1["por_pliegue"] if x["delta_vs_V03_global"] > 0)
    log(f"   pliegues con delta positivo: {pliegues_positivos} de {N_FOLD}")
    for x in t1["por_pliegue"]:
        log(f"      pliegue {x['pliegue']}: macro-F1 {x['macro_f1']:.10f} | "
            f"delta {x['delta_vs_V03_global']:+.10f}")

    # Complementariedad: T1 frente a T0 celda a celda.
    pred_t0 = (proba_oof >= np.array([V03_UMBRALES[et]
                                      for et in ETIQUETAS])[None, :]).astype(int)
    pred_t1 = (proba_oof >= np.array([T1[et] for et in ETIQUETAS])[None, :]).astype(int)
    inter = {
        "ambos_correctos": int(np.count_nonzero((pred_t0 == y) & (pred_t1 == y))),
        "ambos_incorrectos": int(np.count_nonzero((pred_t0 != y) & (pred_t1 != y))),
        "T0_mal_T1_bien": int(np.count_nonzero((pred_t0 != y) & (pred_t1 == y))),
        "T0_bien_T1_mal": int(np.count_nonzero((pred_t0 == y) & (pred_t1 != y))),
    }
    net = inter["T0_mal_T1_bien"] - inter["T0_bien_T1_mal"]
    log(f"   interseccion T0 contra T1: {inter} | net {net:+d}")
    inter_por_label = {}
    for j, et in enumerate(ETIQUETAS):
        r = int(np.count_nonzero((pred_t0[:, j] != y[:, j]) & (pred_t1[:, j] == y[:, j])))
        dm = int(np.count_nonzero((pred_t0[:, j] == y[:, j]) & (pred_t1[:, j] != y[:, j])))
        inter_por_label[et] = {"rescate": r, "damage": dm, "net": r - dm}
        log(f"      {et}: rescate {r:3d} damage {dm:3d} net {r - dm:+4d}")

    # Bootstrap opcional, solo si el delta alcanza +0,001.
    delta_t1 = t1["delta_vs_V03"]
    bootstrap = None
    if delta_t1 >= 0.001:
        rng = np.random.default_rng(42)
        n = len(ids)
        muestras = []
        p0 = pred_t0.ravel()
        p1 = pred_t1.ravel()
        y_2 = y
        for _ in range(2000):
            idx = rng.choice(n, size=n, replace=True)
            f0 = float(np.mean([f1_score(y_2[idx, j], p0.reshape(n, 3)[idx, j],
                                         zero_division=0) for j in range(3)]))
            f1b = float(np.mean([f1_score(y_2[idx, j], p1.reshape(n, 3)[idx, j],
                                          zero_division=0) for j in range(3)]))
            muestras.append(f1b - f0)
        arr = np.array(muestras)
        ic_low, ic_high = np.percentile(arr, [2.5, 97.5])
        bootstrap = {"mediana": round(float(np.median(arr)), 8),
                     "ic95_bajo": round(float(ic_low), 8),
                     "ic95_alto": round(float(ic_high), 8),
                     "incluye_cero": bool(ic_low <= 0 <= ic_high),
                     "n_replicas": 2000}
        log(f"   bootstrap del delta T1: mediana {np.median(arr):+.6f} | "
            f"IC95 [{ic_low:+.6f}, {ic_high:+.6f}] | incluye 0: {bootstrap['incluye_cero']}")
    else:
        log(f"   bootstrap OMITIDO: delta {delta_t1:+.6f} < +0,001")

    # ================================================ 5. DECISION
    log("5/5 decision")
    # Lo que decide es la cifra PLENAMENTE HONESTA: el umbral de cada pliegue
    # exterior aplicado a ese pliegue, sin que el umbral haya visto sus labels.
    delta_honesto = t_nested["delta_vs_V03"]
    v03_fuera_de_rango = [x["label"] for x in filas_est
                         if not x["V03_dentro_del_rango"]]
    # Y hay que separar dos cosas que la cifra global de T1 mezcla: aplicar la
    # mediana a todos los pliegues usa datos de todos ellos para definirla.
    log(f"   delta HONESTO, umbral por pliegue exterior: {delta_honesto:+.10f}")
    log(f"   delta de T1, mediana aplicada globalmente : {delta_t1:+.10f}")
    log(f"   la diferencia entre ambos es {delta_t1 - delta_honesto:+.10f}, y es el")
    log("   optimismo de definir el umbral con datos de todos los pliegues")
    log(f"   etiquetas de V03 fuera del rango interior: {v03_fuera_de_rango or 'ninguna'}")
    log(f"   bootstrap del delta de T1 incluye 0: "
        f"{bootstrap['incluye_cero'] if bootstrap else 'n/d'}")
    log(f"   net a nivel de celda T1 contra T0: {net:+d}")

    if delta_honesto < 0:
        caso = "B"
    elif delta_t1 >= 0.003 and pliegues_positivos >= 4:
        caso = "C"
    else:
        caso = "A"
    log(f"   CASE = {caso}")
    if caso == "A":
        log("   THRESHOLD_TRANSFER_BIAS = CONFIRMED, STATUS = MARGINAL")
    elif caso == "B":
        log("   THRESHOLD_TRANSFER_BIAS = NOT_CONFIRMED: el procedimiento anidado")
        log("   completo NO reproduce mejora, luego se cierra el mecanismo")
    else:
        log("   CANDIDATE, sin submission automatica")

    estdf.to_csv(CSV_ESTABILIDAD, index=False, encoding="utf-8")
    pd.DataFrame(filas_fold).to_csv(CSV_TRANSFER, index=False, encoding="utf-8")

    informe = {
        "familia": "P56_CORRECTED_THRESHOLD_TRANSFER",
        "experimento": "V56-00 (transferencia de umbrales con anidamiento completo)",
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pregunta": ("el sesgo de transferencia de umbrales de P55 es un mecanismo real "
                     "y reproducible que merece formar parte del pipeline final?"),
        "P55_CORRECTED_RESULT_REPRODUCED": bool(reproduce),
        "P55_valor_reproducido": round(f1_p55, 10),
        "P55_valor_registrado": P55_CORRECTED,
        "procedimiento": {
            "outer": f"{N_FOLD} pliegues canonicos de TOXIC, grupos near-dup UNION pseudo-autor",
            "inner": (f"{N_INNER} pliegues agrupados dentro de cada outer train, con "
                      "vectorizador y LogisticRegression re-ajustados"),
            "rejilla": "0,20 a 0,80 paso 0,01, la misma rejilla historica",
            "agregacion": "mediana de las cinco estimaciones. Una sola regla",
            "outer_validation_participa": "en nada del ajuste",
        },
        "umbrales_por_outer_fold": filas_fold,
        "distribucion": filas_est,
        "T0_historico": t0,
        "T1_mediana": t1,
        "nested_por_pliegue_exterior": t_nested,
        "P55_simple": {"macro_f1": round(f1_p55, 10),
                       "delta_vs_V03": round(f1_p55 - V03_CORRECTED_CV, 10)},
        "pliegues_positivos": int(pliegues_positivos),
        "interseccion_T0_T1": inter,
        "interseccion_por_label": inter_por_label,
        "net": int(net),
        "bootstrap": bootstrap,
        "umbrales_V03_fuera_del_rango_interior": v03_fuera_de_rango,
        "delta_honesto_por_pliegue_exterior": round(delta_honesto, 10),
        "optimismo_de_la_cifra_global_de_T1": round(delta_t1 - delta_honesto, 10),
        "nota_decision": (
            "la cifra que decide es la del umbral por pliegue exterior, no la "
            "de la mediana aplicada globalmente. La segunda usa datos de "
            "todos los pliegues para definir el umbral y por eso esta "
            "optimista en mas de 0,005"),
        "THRESHOLD_TRANSFER_BIAS": "CONFIRMED" if caso != "B" else "NOT_CONFIRMED",
        "STATUS": {"A": "MARGINAL", "B": "CLOSED", "C": "CANDIDATE"}[caso],
        "caso": caso,
        "gastos": {"submissions": 0, "consultas_kaggle": 0, "uso_holdout": 0,
                   "uso_test": 0, "segundos": round(time.time() - inicio, 1)},
    }
    REPORTE_JSON.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    _md(informe)
    log("   escrito reports/p56_*")
    log(f"VEREDICTO: caso={caso} | T1_delta={delta_t1:+.6f} | "
        f"pliegues={pliegues_positivos}/{N_FOLD} | 0 submissions, "
        f"{round(time.time() - inicio, 1)} s")


def _md(informe: dict[str, Any]) -> None:
    """Escribe el informe de la transferencia de umbrales.

    Args:
        informe: diccionario con los resultados.

    Returns:
        None.
    """
    p = informe["procedimiento"]
    d = informe["distribucion"]
    t0 = informe["T0_historico"]
    t1 = informe["T1_mediana"]
    inter = informe["interseccion_T0_T1"]
    b = informe["bootstrap"]
    lineas = [
        "# P56 - Transferencia de umbrales corregida",
        "",
        (f"Generado por `experiments/v56_00_transferencia_umbrales.py` "
        f"({informe['experimento']})."),
        "",
        (f"Fecha {informe['fecha']}. **0 submissions, 0 consultas a Kaggle, 0 uso de "
        "holdout o test. La unica variable es el procedimiento de estimacion de "
        "umbrales.**"),
        "",
        "## 1. Reproduccion de P55",
        "",
        f"- valor registrado: **{informe['P55_valor_registrado']:.10f}**",
        f"- valor reproducido: **{informe['P55_valor_reproducido']:.10f}**",
        f"- `P55_CORRECTED_RESULT_REPRODUCED = {informe['P55_CORRECTED_RESULT_REPRODUCED']}`",
        "",
        "## 2. El procedimiento, que aqui SI es un anidamiento completo",
        "",
        f"- outer: {p['outer']}",
        f"- inner: {p['inner']}",
        f"- rejilla: {p['rejilla']}",
        f"- agregacion: {p['agregacion']}",
        f"- outer_validation_participa: **{p['outer_validation_participa']}**",
        "",
        ("La diferencia con P55 es que aqui los umbrales se ajustan sobre un OOF "
        "**interior**, con vectorizador y modelo re-ajustados, en vez de sobre el OOF de "
        "los cuatro pliegues de entrenamiento."),
        "",
        "## 3. Umbrales por pliegue exterior y distribucion",
        "",
        "| outer | " + " | ".join(f"inner {et}" for et in ETIQUETAS)
        + " | " + " | ".join(f"V03 {et}" for et in ETIQUETAS) + " |",
        "|---:|" + "---:|" * 6,
    ]
    for f in informe["umbrales_por_outer_fold"]:
        lineas.append(f"| {f['outer_fold']} | "
                      + " | ".join(f"{f[f'inner_threshold_{et}']:.2f}" for et in ETIQUETAS)
                      + " | "
                      + " | ".join(f"{f[f'V03_{et}']:.2f}" for et in ETIQUETAS) + " |")
    lineas += [
        "",
        ("| etiqueta | estimaciones | media | mediana | std | min | max | V03 | "
        "V03 en el rango | V03 - mediana |"),
        "|---|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for x in d:
        lineas.append(f"| {x['label']} | {x['estimaciones']} | {x['media']:.4f} | "
                      f"**{x['mediana']:.2f}** | {x['std']:.4f} | {x['min']:.2f} | "
                      f"{x['max']:.2f} | {x['V03']:.2f} | {x['V03_dentro_del_rango']} | "
                      f"{x['desviacion_de_la_mediana']:+.2f} |")
    lineas += [
        "",
        "## 4. T0 contra T1",
        "",
        "| politica | macro-F1 | delta vs V03 |",
        "|---|---:|---:|",
        (f"| T0 historico (0,44/0,51/0,46) | {t0['macro_f1']:.10f} | "
        f"{t0['delta_vs_V03']:+.10f} |"),
        (f"| **T1 mediana de las cinco** | **{t1['macro_f1']:.10f}** | "
        f"**{t1['delta_vs_V03']:+.10f}** |"),
        (f"| umbral por pliegue exterior (plenamente honesto) | "
        f"{informe['nested_por_pliegue_exterior']['macro_f1']:.10f} | "
        f"{informe['nested_por_pliegue_exterior']['delta_vs_V03']:+.10f} |"),
        (f"| P55 simple, para referencia | {informe['P55_simple']['macro_f1']:.10f} | "
        f"{informe['P55_simple']['delta_vs_V03']:+.10f} |"),
        "",
        "### F1 por etiqueta",
        "",
        "| etiqueta | T0 | T1 | delta |",
        "|---|---:|---:|---:|",
    ]
    for et in ETIQUETAS:
        lineas.append(f"| {et} | {t0['por_etiqueta'][et]:.10f} | "
                      f"{t1['por_etiqueta'][et]:.10f} | "
                      f"{t1['por_etiqueta'][et] - t0['por_etiqueta'][et]:+.10f} |")
    lineas += [
        "",
        "### Tasa positiva",
        "",
        "| etiqueta | T0 | T1 | diferencia |",
        "|---|---:|---:|---:|",
    ]
    for et in ETIQUETAS:
        a, bb = t0["tasa_positiva"][et], t1["tasa_positiva"][et]
        lineas.append(f"| {et} | {a:.4f} | {bb:.4f} | {bb - a:+.4f} |")
    lineas += [
        "",
        "### Por pliegue",
        "",
        "| pliegue | macro-F1 de T1 | delta vs V03 global |",
        "|---:|---:|---:|",
    ]
    for x in t1["por_pliegue"]:
        lineas.append(f"| {x['pliegue']} | {x['macro_f1']:.10f} | "
                      f"{x['delta_vs_V03_global']:+.10f} |")
    lineas += [
        "",
        f"Pliegues con delta positivo: **{informe['pliegues_positivos']} de 5**.",
        "",
        "## 5. Complementariedad, T1 frente a T0",
        "",
        "| categoria | celdas |",
        "|---|---:|",
        f"| ambos correctos | {inter['ambos_correctos']} |",
        f"| ambos incorrectos | {inter['ambos_incorrectos']} |",
        f"| T0 mal, T1 bien (rescate) | {inter['T0_mal_T1_bien']} |",
        f"| T0 bien, T1 mal (damage) | {inter['T0_bien_T1_mal']} |",
        f"| **net** | **{informe['net']:+d}** |",
        "",
        "| etiqueta | rescate | damage | net |",
        "|---|---:|---:|---:|",
    ]
    for et, v in informe["interseccion_por_label"].items():
        lineas.append(f"| {et} | {v['rescate']} | {v['damage']} | {v['net']:+d} |")
    lineas += [
        "",
        "## 6. Bootstrap",
        "",
    ]
    if b is None:
        lineas.append("Omitido: el delta de T1 no alcanza +0,001, luego la directiva lo "
                      "hace opcional.")
    else:
        lineas += [
            "| magnitud | valor |",
            "|---|---:|",
            f"| mediana del delta | {b['mediana']:+.6f} |",
            f"| IC95 | [{b['ic95_bajo']:+.6f}, {b['ic95_alto']:+.6f}] |",
            f"| incluye 0 | {b['incluye_cero']} |",
            f"| replicas | {b['n_replicas']} |",
        ]
    lineas += [
        "",
        "## 7. Decision",
        "",
        (f"- umbrales de V03 fuera del rango interior en las cinco estimaciones: "
        f"**{informe['umbrales_V03_fuera_del_rango_interior']}**"),
        (f"- delta de T1: **{t1['delta_vs_V03']:+.6f}**, con "
        f"**{informe['pliegues_positivos']} de 5** pliegues positivos"),
        f"- `THRESHOLD_TRANSFER_BIAS = {informe['THRESHOLD_TRANSFER_BIAS']}`",
        f"- `STATUS = {informe['STATUS']}`",
        f"- `CASE = {informe['caso']}`",
        "",
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
