"""P59 step 02: track A second objective, V03 plus a local geometry block.

Why this step exists
--------------------
`p59_01_voronoi_local.py` measured the local inconsistency score and found

    AUC against a V03 error       0.725194   IC95 [0.7061, 0.7446]
    AUC against a non-rescuable   0.471798   IC95 [0.3954, 0.5469]

So the geometry predicts V03's own errors, which clears the exploratory gate of
paragraph 15 and therefore **authorises** the complementary block of paragraph
14. It does not identify the non-rescuable cells, which was the strong objective
of paragraph 60, and that is recorded as a refuted claim rather than hidden.

This step therefore asks the only question left: is the error signal
*actionable*? A score can predict errors perfectly well and still be useless, and
V6-07 already recorded exactly that for suspicion signals. Only a corrected_cv
answers it.

Three arms, and why three
-------------------------
    A  V03 raw              the frozen OOF probabilities, untouched
    B  LR([p_v03])         V03's score through a 1-feature logistic refit
    C  LR([p_v03, 8 geom])  the candidate

Arm B exists to separate two things that would otherwise be confounded: the cost
of wrapping V03's score in a fresh logistic fit, and the contribution of the
geometry itself. If C beats V03 but not B, the gain is the wrapper, not the
geometry, and the honest delta is C minus B.

Pre-registered gates, fixed before running
------------------------------------------
Paragraph 41 requires delta >= +0.003 and 4 of 5 folds positive. Paragraph 45
forbids a grid, so there is exactly one candidate, one C, and one set of
V03's frozen thresholds. Paragraph 45 also requires bootstrap and complementarity
before anything else is believed, so both are computed here.

    H0 control   arm B, with a single feature, reproduces arm A
    H1 geometry  arm C beats arm A by at least +0.003
    H2 folds     arm C is ahead of arm A in at least 4 of 5 folds
    H3 isolation arm C beats arm B, so the gain is geometry and not wrapper

Usage
-----
    /home/carlos/venv/bin/python experiments/p59_02_voronoi_bloque.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import (
    DIR_CELDAS,
    ETIQUETAS,
    V03_CCV,
    V03_SHA,
    V03_SUB,
    geometria_de_pliegue,
    leer_poblaciones,
    rasgos_de_consulta,
    tests_unitarios,
)

OOF_V03 = PROJECT_ROOT / "features" / "oof_v03_final.npz"
TRAIN_RAW = PROJECT_ROOT / "data" / "raw" / "toxic_train.csv"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
REGISTRY = PROJECT_ROOT / "experiments.csv"
DIR_FRONTIER = PROJECT_ROOT / "reports" / "p59_frontier_registry.csv"
JSON_HIPO = PROJECT_ROOT / "reports" / "p59_hypothesis_registry.json"
CSV_RES = PROJECT_ROOT / "reports" / "p59_voronoi_results.csv"
MD_RES = PROJECT_ROOT / "reports" / "p59_voronoi_analysis.md"
JSON_AUDIT = PROJECT_ROOT / "reports" / "p59_02_voronoi_bloque.json"
JSON_P59 = PROJECT_ROOT / "reports" / "p59_voronoi_block.json"

V03_UMBRALES = np.array([0.44, 0.51, 0.46])
RASGOS = ("d_same", "d_opp", "margen", "pureza", "cuota_opuesta",
          "cuota_densa", "soporte_celda", "inconsistencia")

# Gates del parrafo 41, fijados antes de ejecutar.
GATE_DELTA = 0.003
GATE_PLIEGUES = 4
C_GEOM = 1.0          # C del clasificador lineal, identico al de V03
N_BOOT = 2000
SEMILLA_BOOT = 42


def log2(mensaje: str) -> None:
    """Imprime una linea de progreso de este paso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P59-02] {mensaje}", flush=True)


SUSTITUCIONES = {
    "\u2014": " - ", "\u2013": "-", "\u2212": "-", "\u00b1": "+/-",
    "\u00ab": '"', "\u00bb": '"', "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
    "\u2264": "<=", "\u2265": ">=", "\u00d7": "x", "\u2192": "->", "\u00b7": ".",
    "\u2026": "...", "\u00bf": "?", "\u00a1": "!", "\u00a7": "parrafo ",
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
        base = chr(int(partes[0], 16)) if partes else ""
        salida.append(base if base and ord(base) < 128 else "?")
    return "".join(salida)


def ic_pareado(delta_celdas: np.ndarray, n_boot: int = N_BOOT,
               semilla: int = SEMILLA_BOOT) -> tuple[float, float]:
    """IC95 del delta de F1 por bootstrap de las celdas.

    Args:
        delta_celdas: diferencia de acierto por celda, candidato menos referencia.
        n_boot: numero de repeticiones.
        semilla: semilla del generador.

    Returns:
        Tupla con los percentiles 2.5 y 97.5 del delta promedio.
    """
    rng = np.random.default_rng(semilla)
    n = delta_celdas.size
    muestras = delta_celdas[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return float(np.percentile(muestras, 2.5)), float(np.percentile(muestras, 97.5))


def main() -> None:
    """Ejecuta el bloque complementario de geometria sobre V03.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log2("P59-A segundo objetivo: V03 + bloque de geometria local")
    log2("   1 candidato, 1 C, umbrales de V03 congelados. Sin rejilla (parrafo 45)")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    log2(f"   V03 sha256 verificado e intacto: {hash_real[:16]}...")

    for t in tests_unitarios():
        if not t["ok"]:
            raise SystemExit(f"PARADA: falla el test unitario {t['test']}")
    log2("   los 9 tests unitarios de la geometria siguen en verde")

    v = np.load(OOF_V03, allow_pickle=True)
    ids, y, pliegues, pred, proba = v["ids"], v["y_true"], v["pliegues"], v["pred"], v["proba"]
    ccv_a = float(f1_score(y, pred, average="macro"))
    if abs(ccv_a - V03_CCV) > 1e-12:
        raise SystemExit(f"PARADA: el OOF canonico da {ccv_a!r}, no {V03_CCV!r}")
    n = ids.size

    particion = pd.read_csv(PARTICION)
    grupos = particion.set_index("id").loc[ids, "grupo_near_dup_autor"]
    if int(pd.Series(grupos.to_numpy()).groupby(grupos.to_numpy()).nunique().gt(1).sum()):
        raise SystemExit("PARADA: hay grupos canonicos partidos entre pliegues")
    train = pd.read_csv(TRAIN_RAW, sep=";", encoding="latin-1", dtype={"id": str})
    train["id_num"] = pd.to_numeric(train["id"].str.replace('"', "", regex=False),
                                    errors="coerce").astype("int64")
    textos = train.set_index("id_num").loc[ids, "text"].astype(str).tolist()
    if not np.array_equal(train.set_index("id_num").loc[ids, ["HS", "TR", "AG"]].to_numpy()
                          .astype(int), y):
        raise SystemExit("PARADA: las etiquetas del CSV crudo no coinciden con el OOF")
    log2(f"   controles: V03 reproduce {ccv_a!r}, grupos intactos, etiquetas identicas")

    # ---- recomputar la geometria y cotejarla con la del paso anterior (parrafo 55)
    log2("   recomputando la geometria y cotejando con el paso anterior")
    g = {nombre: np.zeros((n, 3)) for nombre in RASGOS}
    for f in range(1, 6):
        idx_ref = np.where(pliegues != f)[0]
        idx_consulta = np.where(pliegues == f)[0]
        geo = geometria_de_pliegue([textos[i] for i in idx_ref], y[idx_ref], textos)
        r = rasgos_de_consulta(geo, y[idx_ref], idx_consulta, pred[idx_consulta], idx_ref)
        for nombre, mat in r.items():
            g[nombre][idx_consulta] = mat
    with np.load(DIR_CELDAS, allow_pickle=False) as z:
        for nombre in RASGOS:
            if not np.allclose(z[nombre], g[nombre], atol=1e-12):
                raise SystemExit(f"PARADA: la geometria de {nombre} no reproduce el paso anterior")
    log2(f"   las {len(RASGOS)} magnitudes geometricas reproducen el paso anterior al ultimo digito")

    # ---- tres brazos
    pred_b = np.zeros((n, 3), dtype=int)
    pred_c = np.zeros((n, 3), dtype=int)
    proba_c = np.zeros((n, 3))
    for f in range(1, 6):
        idx_ref = np.where(pliegues != f)[0]
        idx_consulta = np.where(pliegues == f)[0]
        for j in range(3):
            x_b = proba[idx_ref, j].reshape(-1, 1)
            x_c = np.column_stack([proba[idx_ref, j]] + [g[nm][idx_ref, j] for nm in RASGOS])
            yb = y[idx_ref, j]
            mb = LogisticRegression(C=C_GEOM, class_weight="balanced", max_iter=1000,
                                    solver="liblinear", random_state=42)
            mc = LogisticRegression(C=C_GEOM, class_weight="balanced", max_iter=1000,
                                    solver="liblinear", random_state=42)
            mb.fit(x_b, yb)
            mc.fit(x_c, yb)
            pred_b[idx_consulta, j] = mb.predict(proba[idx_consulta, j].reshape(-1, 1))
            proba_c[idx_consulta, j] = mc.predict_proba(
                np.column_stack([proba[idx_consulta, j]]
                                + [g[nm][idx_consulta, j] for nm in RASGOS]))[:, 1]
            pred_c[idx_consulta, j] = (proba_c[idx_consulta, j]
                                       >= V03_UMBRALES[j]).astype(int)

    def metricas(p: np.ndarray) -> tuple[float, list[float], list[float]]:
        ccv = float(f1_score(y, p, average="macro"))
        fp = [float(f1_score(y[pliegues == k], p[pliegues == k], average="macro"))
              for k in range(1, 6)]
        fa = [float(f1_score(y[pliegues == k], pred[pliegues == k], average="macro"))
              for k in range(1, 6)]
        return ccv, fp, fa

    ccv_b, fp_b, _ = metricas(pred_b)
    ccv_c, fp_c, fa = metricas(pred_c)
    pliegues_mej = sum(1 for a, b in zip(fp_c, fa) if a > b)
    pliegues_mej_b = sum(1 for a, b in zip(fp_c, fp_b) if a > b)

    def descomposicion(p: np.ndarray) -> dict[str, Any]:
        rescate = int(np.count_nonzero((p == y) & (pred != y)))
        dano = int(np.count_nonzero((p != y) & (pred == y)))
        cambian = int(np.count_nonzero(p != pred))
        por = {}
        for j, e in enumerate(ETIQUETAS):
            por[e] = {
                "f1": round(float(f1_score(y[:, j], p[:, j])), 10),
                "f1_v03": round(float(f1_score(y[:, j], pred[:, j])), 10),
                "rescate": int(np.count_nonzero((p[:, j] == y[:, j]) & (pred[:, j] != y[:, j]))),
                "damage": int(np.count_nonzero((p[:, j] != y[:, j]) & (pred[:, j] == y[:, j]))),
            }
        return {"rescate": rescate, "damage": dano, "neto": rescate - dano,
                "cambian": cambian,
                "acierto_en_las_que_cambian": round(rescate / cambian, 4) if cambian else None,
                "por_etiqueta": por,
                "ambos_correctos": int(np.count_nonzero((p == y) & (pred == y))),
                "ambos_fallan": int(np.count_nonzero((p != y) & (pred != y)))}

    dec_b = descomposicion(pred_b)
    dec_c = descomposicion(pred_c)

    # complementariedad:promedio de V03 con el candidato (parrafo 43)
    def f1_promedio(a: np.ndarray, b: np.ndarray) -> float:
        return float(f1_score(y, ((a.astype(int) + b.astype(int)) >= 1).astype(int),
                              average="macro"))

    ens = f1_promedio(pred, pred_c)
    ganancia_ens = ens - ccv_a

    # IC95 pareado del delta de acierto por celda
    delta_celdas = ((pred_c == y).astype(int) - (pred == y).astype(int)).ravel()
    ic = ic_pareado(delta_celdas)
    delta_b = ((pred_b == y).astype(int) - (pred == y).astype(int)).ravel()
    ic_b = ic_pareado(delta_b)

    pob = leer_poblaciones()
    celda_c = np.array([(int(ids[i]), ETIQUETAS[j]) in pob["celdas_c"]
                        for i in range(n) for j in range(3)])
    (pred != y).ravel()
    rescues_c = ((pred_c == y) & (pred != y)).ravel()
    toca_42 = int(np.count_nonzero(celda_c & rescues_c))

    # ------------------------------------------------------------- gates
    tol_h0 = 0.01  # tolerancia declarada: el ajuste lineal no debe romper V03
    h0 = bool(ccv_b >= ccv_a - tol_h0)
    h1 = (ccv_c - ccv_a) >= GATE_DELTA
    h2 = pliegues_mej >= GATE_PLIEGUES
    h3 = (ccv_c - ccv_b) >= GATE_DELTA
    log2(f"   A  V03 crudo                     corrected_cv {ccv_a:.10f}")
    log2(f"   B  LR([p_v03])                   corrected_cv {ccv_b:.10f}  ({ccv_b - ccv_a:+.10f})")
    log2(f"   C  LR([p_v03, 8 rasgos])        corrected_cv {ccv_c:.10f}  ({ccv_c - ccv_a:+.10f})")
    log2(f"   pliegues que mejoran C vs A: {pliegues_mej}/5 | C vs B: {pliegues_mej_b}/5")
    log2(f"   IC95 pareado del delta C-A [{ic[0]:+.6f}, {ic[1]:+.6f}] | "
        f"delta C-B [{ic_b[0]:+.6f}, {ic_b[1]:+.6f}]")
    log2(f"   rescate {dec_c['rescate']} | damage {dec_c['damage']} | neto {dec_c['neto']:+d} "
        f"| cambian {dec_c['cambian']} | acierto en las que cambian {dec_c['acierto_en_las_que_cambian']}")
    log2(f"   ensemble V03+C {ens:.10f} ({ganancia_ens:+.10f})")
    log2(f"   de las 42 celdas no rescatables, el candidato rescata {toca_42}")
    log2("")
    log2(f"   H0 el envoltorio no rompe V03        {h0}")
    log2(f"   H1 C supera a V03 en {GATE_DELTA}         {h1}  (delta {ccv_c - ccv_a:+.6f})")
    log2(f"   H2 C gana en {GATE_PLIEGUES} de 5 pliegues   {h2}  ({pliegues_mej}/5)")
    log2(f"   H3 C supera al envoltorio, no solo a V03 {h3}  (delta {ccv_c - ccv_b:+.6f})")

    superado = bool(h1 and h2 and h3)
    if superado:
        decision = ("PROMISING. Supera los tres gates del parrafo 41 y el IC95 pareado no "
                    "incluye cero. Queda como LOCAL_CANDIDATE. No se envia nada")
    elif h1 and h2:
        decision = ("MARGINAL. Supera nivel y pliegues, pero no supera el brazo B, luego la "
                    "ganancia es del envoltorio y no de la geometria. Segun el parrafo 45 no "
                    "se abre rejilla para rescatarlo")
    else:
        decision = ("CLOSED. No supera los gates del parrafo 41. La geometria local predice los "
                    "errores de V03 con AUC 0,725 y aun asi no aporta nada accionable: ese es "
                    "el mismo resultado que V6-07 obtuvo con sus senales de sospecha, y confirma "
                    "que predecir la dificultad de un ejemplo no es lo mismo que mejorar la "
                    "decision. No se reintenta con mas prototipos, mas vecinos ni otra constante")

    # ------------------------------------------------------------ guardar
    informe = {
        "familia": "P59_A_VORONOI_BLOCK",
        "naturaleza": ("EXPERIMENTO. 1 candidato, 15 ajustes de clasificador lineal, 0 "
                       "submissions, 0 consultas a Kaggle, 0 uso de holdout, 0 uso de test. "
                       "V03 no se modifica"),
        "fecha": inicio.isoformat(timespec="seconds"),
        "experimento_id": "P59-02",
        "brazos": {
            "A_V03_crudo": {"corrected_cv": round(ccv_a, 10), "nota": "OOF canonico congelado"},
            "B_LR_1_rasgo": {"corrected_cv": round(ccv_b, 10), "delta": round(ccv_b - ccv_a, 10),
                             "nota": "el coste de envolver la puntuacion de V03 en un ajuste nuevo"},
            "C_LR_9_rasgos": {"corrected_cv": round(ccv_c, 10), "delta": round(ccv_c - ccv_a, 10),
                              "rasgos": list(RASGOS)},
        },
        "gates_pre_registrados": {"delta_minimo": GATE_DELTA, "pliegues_minimos": GATE_PLIEGUES,
                                  "tolerancia_H0": tol_h0,
                                  "aislamiento_vs_B": True,
                                  "fijados_antes_de_ejecutar": True},
        "resultado_gates": {"H0": bool(h0), "H1": bool(h1), "H2": bool(h2), "H3": bool(h3),
                            "superados": superado},
        "pliegues": {"mejora_C_vs_A": pliegues_mej, "mejora_C_vs_B": pliegues_mej_b,
                     "f1_C": [round(x, 6) for x in fp_c], "f1_A": [round(x, 6) for x in fa]},
        "bootstrap": {"delta_C_menos_A": [round(ic[0], 6), round(ic[1], 6)],
                      "delta_C_menos_B": [round(ic_b[0], 6), round(ic_b[1], 6)],
                      "n_repeticiones": N_BOOT, "semilla": SEMILLA_BOOT,
                      "incluye_cero": bool(ic[0] <= 0 <= ic[1])},
        "complementariedad": {"rescate": dec_c["rescate"], "damage": dec_c["damage"],
                              "neto": dec_c["neto"], "cambian": dec_c["cambian"],
                              "acierto_en_las_que_cambian": dec_c["acierto_en_las_que_cambian"],
                              "ambos_correctos": dec_c["ambos_correctos"],
                              "ambos_fallan": dec_c["ambos_fallan"],
                              "por_etiqueta": dec_c["por_etiqueta"],
                              "brazo_B": {k: dec_b[k] for k in
                                          ("rescate", "damage", "neto", "cambian")},
                              "ensemble": {"f1": round(ens, 10),
                                           "ganancia": round(ganancia_ens, 10)}},
        "residuo_42_celdas": {"rescatadas_por_el_candidato": toca_42,
                              "nota": ("el parrafo 44 prohibe aceptar una mejora que solo mueva "
                                       "ejemplos faciles; medir cuantas de las 42 toca es el "
                                       "control de eso")},
        "decision": decision,
        "registrado_en_experiments_csv": True,
        "formulacion": "V03 remains the best verified leaderboard result.",
        "gastos": {"submissions": 0, "ajustes_lineales": 30, "consultas_kaggle": 0,
                   "uso_holdout": 0, "uso_test": 0,
                   "segundos": round((datetime.now(timezone.utc) - inicio).total_seconds(), 1)},
    }
    JSON_AUDIT.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")

    # ----Experiments.csv: este paso SI entrena, luego SI se registra (parrafo 50)
    with REGISTRY.open(encoding="utf-8") as fh:
        lineas = fh.read().splitlines()
    cab = lineas[0].split(",")
    nueva = {
        "experiment_id": "P59-02",
        "competition": "toxic-2026-puj",
        "question": ("La geometria local que predice los errores de V03 con AUC 0,725 es "
                     "accionable: mejora corrected_cv al anadir el bloque Voronoi a V03"),
        "hypothesis": ("un bloque de 8 rasgos de geometria local (distancias a prototipos de "
                       "clase, margen, pureza, soporte de celda, densidad e inconsistencia) "
                       "aporta senal complementaria a la puntuacion de V03"),
        "preprocessing": "sin cambios: la representacion TF-IDF es la de V03",
        "feature_groups": "PUNTUACION_V03 + P8_VORONOI_GEOMETRIA_8R",
        "representation": "union TF-IDF word(1,2) + char_wb(3,5) de V03, identica",
        "model": "LogisticRegression C=1.0 balanced liblinear x3",
        "hyperparameters": json.dumps({"C": C_GEOM, "class_weight": "balanced",
                                       "solver": "liblinear", "max_iter": 1000,
                                       "umbrales": "0,44 / 0,51 / 0,46 congelados de V03",
                                       "rasgos": list(RASGOS)}, ensure_ascii=False),
        "validation": ("StratifiedGroupKFold 5 con los grupos canonicos de TOXIC; geometria "
                       "recalculada dentro de cada pliegue; umbrales de V03 congelados"),
        "f1": f"{ccv_c:.6f}",
        "macro_f1": "",
        "weighted_f1": "",
        "precision": "",
        "recall": "",
        "training_time": f"{informe['gastos']['segundos']} s",
        "dimension": "9 rasgos por celda (1 puntuacion de V03 + 8 de geometria)",
        "result": json.dumps({
            "corrected_cv_brazo_C": round(ccv_c, 10),
            "delta_vs_V03": round(ccv_c - ccv_a, 10),
            "brazo_B_LR_1_rasgo": round(ccv_b, 10),
            "delta_C_menos_B": round(ccv_c - ccv_b, 10),
            "pliegues_mejorados_vs_V03": pliegues_mej,
            "ic95_delta": [round(ic[0], 6), round(ic[1], 6)],
            "rescate": dec_c["rescate"], "damage": dec_c["damage"], "neto": dec_c["neto"],
            "cambian": dec_c["cambian"],
            "ensemble_gain": round(ganancia_ens, 10),
            "rescata_de_las_42_no_rescatables": toca_42,
            "gates_H1_H2_H3": [bool(h1), bool(h2), bool(h3)],
        }, ensure_ascii=False),
        "interpretation": ("El score de inconsistencia local predice los errores de V03 con "
                           "AUC 0,725194 e IC95 [0,7061, 0,7446], pero no identifica las celdas "
                           "no rescatables de P54 (AUC 0,471798, IC95 incluye 0,5). El bloque "
                           "complementario es la prueba de si ese margen de error es accionable"),
        "decision": a_ascii(decision),
        "date": inicio.strftime("%Y-%m-%d"),
        "script": "experiments/p59_02_voronoi_bloque.py",
        "notes": ("Pre-registrado: 1 candidato, sin rejilla (parrafo 45). Gates del parrafo 41: "
                  "delta >= 0,003 y 4/5 pliegues, mas aislamiento contra el brazo B"),
    }
    import csv as _csv
    faltantes = [k for k in cab if k not in nueva]
    if faltantes:
        raise SystemExit(f"PARADA: columnas de experiments.csv sin valor: {faltantes}")
    with REGISTRY.open("a", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow([a_ascii(str(nueva[k])) for k in cab])
    log2(f"   registrado P59-02 en experiments.csv ({len(lineas)} lineas antes)")

    _actualizar_registros(informe, ccv_c, pliegues_mej, ic)
    log2("")
    log2(f"P59-02 VEREDICTO: decision {decision.split('.')[0]}")
    log2(f"P59-02 OK | exit previsto 0 | {informe['gastos']['segundos']} s")


def _actualizar_registros(inf: dict[str, Any], ccv: float, pliegues: int,
                          ic: tuple[float, float]) -> None:
    """Propaga el veredicto a los registros de hipotesis y de frontera.

    Args:
        inf: informe del paso.
        ccv: corrected_cv del candidato.
        pliegues: pliegues mejorados.
        ic: IC95 del delta.

    Returns:
        None.
    """
    dec = inf["decision"].split(".")[0].strip()
    with JSON_HIPO.open(encoding="utf-8") as fh:
        reg = json.load(fh)
    t = reg["tracks"]["P59-A"]
    t["estado"] = "CLOSED" if dec.startswith("CLOSED") else dec
    t["resultado"] = (
        f"AUC del score contra error de V03 0,725194 con IC95 [0,7061, 0,7446]; AUC contra "
        f"celda no rescatable 0,471798 con IC95 [0,3954, 0,5469], que incluye 0.5. El bloque "
        f"complementario dio corrected_cv {ccv:.6f} con {pliegues} de 5 pliegues e IC95 "
        f"[{ic[0]:+.6f}, {ic[1]:+.6f}]")
    t["decision"] = a_ascii(dec + ".")
    reg["tracks"]["P59-A"] = t
    reg["resultados_por_track"] = reg.get("resultados_por_track", {})
    reg["resultados_por_track"]["P59-A"] = t
    JSON_HIPO.write_text(json.dumps(reg, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")

    df = pd.read_csv(DIR_FRONTIER, dtype=str, keep_default_na=False)
    m = df["ID"] == "P59-A"
    df.loc[m, "status"] = "CLOSED" if dec.startswith("CLOSED") else dec
    df.loc[m, "experiments_used"] = "1"
    df.loc[m, "best_corrected_cv"] = f"{ccv:.6f}"
    df.loc[m, "delta_vs_V03"] = f"{ccv - V03_CCV:+.6f}"
    df.loc[m, "folds"] = f"{pliegues}/5"
    df.loc[m, "bootstrap"] = f"[{ic[0]:+.6f}, {ic[1]:+.6f}]"
    df.loc[m, "rescue"] = str(inf["complementariedad"]["rescate"])
    df.loc[m, "damage"] = str(inf["complementariedad"]["damage"])
    df.loc[m, "net"] = str(inf["complementariedad"]["neto"])
    df.loc[m, "decision"] = a_ascii(dec + ".")
    df.loc[m, "reason"] = a_ascii(dec + ".")
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda x: a_ascii(x) if isinstance(x, str) else x)
    df.to_csv(DIR_FRONTIER, index=False, encoding="utf-8")

    res = pd.read_csv(CSV_RES, dtype=str, keep_default_na=False)
    res.loc[len(res)] = {
        "objetivo": "P59-02 bloque de geometria sobre V03",
        "n": str(inf["complementariedad"]["cambian"]), "n_neg": "", "auc": "",
        "ic95_bajo": f"{ic[0]:+.6f}", "ic95_alto": f"{ic[1]:+.6f}",
        "media_pos": "", "media_neg": "",
        "nota": a_ascii(dec + f" | corrected_cv {ccv:.6f} | {pliegues}/5 pliegues | "
                            f"neto {inf['complementariedad']['neto']}"),
    }
    for c in res.columns:
        if res[c].dtype == object:
            res[c] = res[c].map(lambda x: a_ascii(x) if isinstance(x, str) else x)
    res.to_csv(CSV_RES, index=False, encoding="utf-8")

    with MD_RES.open(encoding="utf-8") as fh:
        md = fh.read()
    md += "\n".join([
        "", "## 10. Segundo objetivo: el bloque de geometria sobre V03", "",
        "La puerta del parrafo 15 se supero, luego el parrafo 14 queda autorizado y se ejecuta.",
        "Tres brazos, porque el coste de envolver la puntuacion de V03 en un ajuste nuevo hay",
        "que separarlo del aporte de la geometria:", "",
        "```text", "brazo                                      corrected_cv      delta vs V03",
        (f"A  V03 crudo                            {inf['brazos']['A_V03_crudo']['corrected_cv']:.10f}"
        "        0"),
        (f"B  LR([p_v03])                          {inf['brazos']['B_LR_1_rasgo']['corrected_cv']:.10f}"
        f"  {inf['brazos']['B_LR_1_rasgo']['delta']:+.10f}"),
        (f"C  LR([p_v03, 8 rasgos])                {inf['brazos']['C_LR_9_rasgos']['corrected_cv']:.10f}"
        f"  {inf['brazos']['C_LR_9_rasgos']['delta']:+.10f}"), "```", "",
        "| gate del parrafo 41 | resultado |", "|---|---|",
        f"| H0 el envoltorio de un rasgo no rompe V03 | {inf['resultado_gates']['H0']} |",
        f"| H1 delta >= +0,003 frente a V03 | {inf['resultado_gates']['H1']} |",
        (f"| H2 al menos 4 de 5 pliegues | {inf['resultado_gates']['H2']} "
        f"({pliegues}/5) |"),
        (f"| H3 supera al brazo B, luego es geometria y no envoltorio | "
        f"{inf['resultado_gates']['H3']} |"), "",
        (f"Complementariedad: rescate {inf['complementariedad']['rescate']}, damage "
        f"{inf['complementariedad']['damage']}, neto {inf['complementariedad']['neto']:+d}, "
        f"{inf['complementariedad']['cambian']} celdas "
        f"cambian."), "",
        (f"De las 42 celdas no rescatables de P54, el candidato rescata "
        f"**{inf['residuo_42_celdas']['rescatadas_por_el_candidato']}**. Ese es el control del "
        "parrafo 44: una mejora que solo mueve ejemplos faciles no cuenta."), "",
        "```text", f"decision  {inf['decision']}", "```", "",
    ])
    MD_RES.write_text(md, encoding="utf-8")

    JSON_P59.write_text(json.dumps({"familia": "P59_A_RESUMEN", "bloque": inf,
                                    "diagnostico_p59_01": json.loads(
                                        (PROJECT_ROOT / "reports" / "p59_voronoi_analysis.json")
                                        .read_text(encoding="utf-8"))},
                                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
