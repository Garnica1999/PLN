"""Training-size sensitivity: how much does the CV move when the training set shrinks.

Why this measurement, and what it cannot do
-------------------------------------------
The submitted candidate was fit on 1483 rows and V03's submission on 1731. That difference is
real and it is a confound: the external comparison is not a clean test of the weighting, because
two things changed at once.

What cannot be done is the direct experiment. Reproducing 1731 versus 1483 would need the
holdout, and the holdout stays frozen. So the question is asked the only way it can be asked
without it:

    take the development set, subsample it by GROUPS at 100, 90, 80 and 70 per cent,
    run the SAME frozen V03 pipeline on each subsample, and measure how far the CV moves.

That measures the magnitude of the variability that comes purely from having fewer examples. It
does not measure what would have happened on 1731, and it does not explain the leaderboard. It
answers a narrower question: is a 14 per cent reduction in training data, on its own, the kind
of change that moves this metric by the size of the effect we are trying to explain.

What is frozen
--------------
    V03              untouched
    the weighting    untouched, and its per-fold multipliers stay at the canonical medians so
                     that nothing is re-searched at each size
    the seeds        fixed here, in writing, before the first run
    the folds        StratifiedGroupKFold over the same fused groups

The subsampling is by GROUP: a near-duplicate group is kept or dropped whole, never split. A
subsample that splits a duplicate group would understate the variability it is trying to measure.
"""

from __future__ import annotations

import ast
import csv as _csv
import hashlib
import importlib.util
import json
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import V03_CCV, V03_SHA, V03_SUB

from src.evaluation import metricas_multietiqueta
from src.splits import ETIQUETAS, combinaciones, grupos_fusionados, holdout_v2

REPORTS = PROJECT_ROOT / "reports"
FIGURAS = PROJECT_ROOT / "figuras" / "p70"
CSV_T = REPORTS / "toxic_training_size_sensitivity.csv"
CSV_W = REPORTS / "toxic_weight_stability.csv"
JSON_S = REPORTS / "toxic_cv_lb_transfer_postmortem.json"

UMBRALES = np.array([0.44, 0.51, 0.46])
SEMILLA = 42
# Fijadas ANTES de ejecutar, una por tamano y repeticion.
SEMILLAS_SUBMUESTRA = {100: [20261003, 20261004, 20261005],
                       90: [20261006, 20261007, 20261008],
                       80: [20261009, 20261010, 20261011],
                       70: [20261012, 20261013, 20261014]}
TAMANOS = (100, 90, 80, 70)
# Mediana canonica de los multiplicadores por pliegue, congelada: no se re-busca por tamano.
MULT_CONGELADOS = [1.2, 1.5, 1.2]
# Evidencia externa, congelada.
LB_V03 = 0.78306
LB_CAND = 0.77522
DELTA_LB = -0.00784
DELTA_CV = 0.0082045862


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P70-01] {mensaje}", flush=True)


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


def f1m(y: np.ndarray, pred: np.ndarray) -> float:
    """Macro F1 con la metrica del proyecto.

    Args:
        y: etiquetas verdaderas.
        pred: predicciones.

    Returns:
        La media de las F1 por columna.
    """
    return float(metricas_multietiqueta(y, pred)["f1_media_columnas"])


def pesos_de_clase(y: np.ndarray, mult: float) -> dict[int, float]:
    """Pesos de clase explicitos con multiplicador.

    Args:
        y: etiquetas binarias.
        mult: multiplicador de la clase positiva.

    Returns:
        Diccionario clase -> peso.
    """
    n = len(y)
    npos = int((y == 1).sum())
    return {0: n / (2.0 * max(n - npos, 1)), 1: (n / (2.0 * max(npos, 1))) * mult}


def cargar_tfidf():
    """Carga `esquema_tfidf` del modulo historico sin ejecutar su main.

    Returns:
        La funcion.
    """
    ruta = PROJECT_ROOT / "experiments" / "30_supervised_weighting_v4.py"
    spec = importlib.util.spec_from_file_location("h30s", ruta)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.esquema_tfidf


def ajustar_exacto(esquema, matrices, y_entrenar, mult, sample_weight):
    """Replica del bloque inline historico: class_weight explicito SIEMPRE.

    Args:
        esquema: no se usa, se mantiene por claridad de la firma.
        matrices: par (entrenamiento, aplicacion).
        y_entrenar: etiquetas del entrenamiento.
        mult: multiplicadores por etiqueta.
        sample_weight: peso de ejemplo.

    Returns:
        Matriz de probabilidades.
    """
    del esquema
    izquierda, derecha = matrices
    salida = np.zeros((derecha.shape[0], len(ETIQUETAS)))
    for pos in range(len(ETIQUETAS)):
        modelo = LogisticRegression(C=2.0, max_iter=1000, solver="liblinear",
                                    random_state=SEMILLA)
        modelo.set_params(class_weight=pesos_de_clase(y_entrenar[:, pos], float(mult[pos])))
        modelo.fit(izquierda, y_entrenar[:, pos], sample_weight=sample_weight)
        salida[:, pos] = modelo.predict_proba(derecha)[:, 1]
    return salida


def pesos_de_margen(proba: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict]:
    """Aplica la regla de pesos CONGELADA y devuelve sus descriptores.

    Args:
        proba: probabilidades OOF.
        y: etiquetas.

    Returns:
        Los pesos y un diccionario con su distribucion.
    """
    mf = (proba - UMBRALES[None, :]) * (2 * y - 1)
    error_score = np.clip(-mf, 0.0, None).mean(axis=1)
    iqr = float(np.percentile(error_score, 75) - np.percentile(error_score, 25)) or 1.0
    z = (error_score - float(np.median(error_score))) / iqr
    p = np.clip(1.0 - 0.5 * z, 0.5, 1.5)
    return p, {"mediana_error_score": round(float(np.median(error_score)), 6),
               "IQR": round(iqr, 6),
               "n_peso_menor_1": int((p < 1).sum()),
               "n_peso_igual_1": int((p == 1).sum()),
               "peso_medio": round(float(p.mean()), 6),
               "peso_min": round(float(p.min()), 6),
               "peso_max": round(float(p.max()), 6),
               "peso_total_retirado": round(1.0 - float(p.sum()) / len(p), 6)}


def main() -> None:
    """Mide la sensibilidad del CV al tamano de entrenamiento y la estabilidad de los pesos.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P70-01 sensibilidad al tamano de entrenamiento y estabilidad de los pesos")
    log("   sin test, sin holdout, sin leaderboard")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    esquema = cargar_tfidf()

    from src.data import read_train
    train = read_train()
    grupos_all = grupos_fusionados(train)
    idev, ihold = holdout_v2(train, grupos_all)
    ids_dev = train.iloc[idev]["id"].to_numpy(dtype=np.int64)
    if not np.array_equal(ids_dev,
                          np.load(PROJECT_ROOT / "features" / "oof_v03_final.npz",
                                  allow_pickle=True)["ids"].astype(np.int64)):
        raise SystemExit("PARADA: los ids del desarrollo no coinciden con el OOF canonico")
    desarrollo = train.iloc[idev].reset_index(drop=True)
    grupos = grupos_all[idev]
    desarrollo["text"]
    y = desarrollo[["HS", "TR", "AG"]].to_numpy(dtype=int)

    # ITEM 4: el confound, con sus hashes.
    confound = {
        "V03_enviado_filas": len(train),
        "candidato_enviado_filas": len(desarrollo),
        "holdout_excluido_filas": len(ihold),
        "confirmado_por": ["experiments/p69_10_submission_candidato.py",
                           "reports/toxic_submission_cost_sensitive_manifest.json",
                           "experiments/baseline_v03_frozen/pipeline_v03.json"],
        "training_ids_hash": hashlib.sha256(ids_dev.tobytes()).hexdigest()[:32],
        "excluded_holdout_ids_hash": hashlib.sha256(
            train.iloc[ihold]["id"].to_numpy(dtype=np.int64).tobytes()).hexdigest()[:32],
        "holdout_labels_leidas": False,
    }
    log("")
    log("   ITEM 4, EL CONFOUND DE TAMANO, CONFIRMADO")
    for k in ("V03_enviado_filas", "candidato_enviado_filas", "holdout_excluido_filas"):
        log(f"      {k:26} {confound[k]}")
    log(f"      training_ids_hash          {confound['training_ids_hash']}")
    log(f"      holdout_ids_hash           {confound['excluded_holdout_ids_hash']}")
    log("      las labels del holdout NO se leen")

    log("")
    log("   ITEM 7, SENSIBILIDAD DEL CV AL TAMANO DE ENTRENAMIENTO")
    log(f"      semillas fijadas antes de ejecutar: {SEMILLAS_SUBMUESTRA}")
    log("      submuestreo POR GRUPO: un grupo near-duplicate entero entra o sale")
    log("")

    filas: list[dict] = []
    pesos_por_corrida: dict[str, np.ndarray] = {}
    ids_por_corrida: dict[str, np.ndarray] = {}
    for tam in TAMANOS:
        for rep, semilla in enumerate(SEMILLAS_SUBMUESTRA[tam], start=1):
            t0 = time.time()
            rng = np.random.default_rng(semilla)
            unicos = pd.unique(grupos)
            n_grupos = max(round(len(unicos) * tam / 100.0), 5)
            elegidos = rng.choice(unicos, size=n_grupos, replace=False)
            sel = np.isin(grupos, elegidos)
            idx = np.where(sel)[0]
            sub = desarrollo.iloc[idx].reset_index(drop=True)
            ysub = y[idx]
            gsub = grupos[idx]

            pliegues = []
            divisor = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
            for tr, va in divisor.split(sub, combinaciones(sub), groups=gsub):
                pliegues.append((tr, va))

            cache = [esquema(sub["text"].iloc[tr], sub["text"].iloc[va], sublinear_tf=True)
                     for tr, va in pliegues]
            # V03 congelado: sin pesos, class_weight explicito a 1.0
            p03 = np.zeros((len(sub), 3))
            for k, (tr, va) in enumerate(pliegues):
                p03[va] = ajustar_exacto(esquema, cache[k], ysub[tr], np.ones(3), np.ones(len(tr)))
            cv03 = f1m(ysub, (p03 >= UMBRALES[None, :]).astype(int))

            # La regla de pesos CONGELADA, y el candidato con los multiplicadores de la mediana
            pesos, desc = pesos_de_margen(p03, ysub)
            pc = np.zeros((len(sub), 3))
            for k, (tr, va) in enumerate(pliegues):
                pc[va] = ajustar_exacto(esquema, cache[k], ysub[tr],
                                        np.array(MULT_CONGELADOS), pesos[tr])
            cvc = f1m(ysub, (pc >= UMBRALES[None, :]).astype(int))

            clave = f"{tam}_{rep}"
            pesos_por_corrida[clave] = pesos
            ids_por_corrida[clave] = sub["id"].to_numpy(dtype=np.int64)
            filas.append({"tamano_pct": tam, "repeticion": rep, "semilla": semilla,
                          "n_filas": len(sub), "n_grupos": int(n_grupos),
                          "cv_v03": round(cv03, 10), "cv_candidato": round(cvc, 10),
                          "delta": round(cvc - cv03, 10), **desc})
            log(f"      {tam:3d}% rep {rep} | semilla {semilla} | {len(sub):4d} filas | "
                f"V03 {cv03:.6f} | cand {cvc:.6f} | delta {cvc - cv03:+.6f} | "
                f"peso<1 {desc['n_peso_menor_1']:4d} | ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(filas)
    df.to_csv(CSV_T, index=False, encoding="ascii", lineterminator="\n")

    log("")
    log("   ITEM 8, VARIABILIDAD POR TAMANO FRENTE AL EFECTO DEL WEIGHTING")
    log(f"      del efecto que hay que explicar: {DELTA_CV:+.10f}")
    log("")
    log(f"      {'tam':>4}{'cv_v03 medio':>15}{'desv':>10}{'rango':>10}"
        f"{'delta medio':>14}{'desv delta':>12}")
    resumen = []
    for tam in TAMANOS:
        sub = df[df["tamano_pct"] == tam]
        r = {"tamano_pct": tam, "cv_v03_medio": round(float(sub["cv_v03"].mean()), 10),
             "cv_v03_desv": round(float(sub["cv_v03"].std(ddof=1)), 10),
             "cv_v03_rango": round(float(sub["cv_v03"].max() - sub["cv_v03"].min()), 10),
             "delta_medio": round(float(sub["delta"].mean()), 10),
             "delta_desv": round(float(sub["delta"].std(ddof=1)), 10),
             "n_filas_medio": int(sub["n_filas"].mean())}
        resumen.append(r)
        log(f"      {tam:>3}%{r['cv_v03_medio']:>15.6f}{r['cv_v03_desv']:>10.6f}"
            f"{r['cv_v03_rango']:>10.6f}{r['delta_medio']:>+14.6f}{r['delta_desv']:>12.6f}")

    desv_peor = max(r["cv_v03_desv"] for r in resumen)
    rango_peor = max(r["cv_v03_rango"] for r in resumen)
    caida_70 = resumen[0]["cv_v03_medio"] - resumen[-1]["cv_v03_medio"]
    log("")
    log(f"      desviacion maxima entre repeticiones del mismo tamano: {desv_peor:.6f}")
    log(f"      rango maximo dentro de un tamano:                      {rango_peor:.6f}")
    log(f"      caida de V03 del 100% al 70%:                          {caida_70:+.6f}")
    log("")
    log("      LECTURA, sin afirmar causalidad: si la desviacion por reducir datos es del orden")
    log("      del efecto, entonces el tamano de entrenamiento es un confounder plausible.")
    log(f"      desviacion por tamano {desv_peor:.6f} frente a efecto {DELTA_CV:.6f} "
        f"-> ratio {desv_peor / DELTA_CV:.2f}")

    # ---------- ITEMS 9 y 10: distribucion y estabilidad de los pesos ----------
    log("")
    log("   ITEM 9, LA MISMA FORMULA GENERA POBLACIONES DE PESOS DISTINTAS?")
    log(f"      {'tam':>4}{'mediana':>10}{'IQR':>10}{'peso<1':>9}{'peso=1':>9}"
        f"{'p.medio':>10}{'min':>8}{'max':>8}{'retirado':>11}")
    for tam in TAMANOS:
        s = df[df["tamano_pct"] == tam]
        log(f"      {tam:>3}%{s['mediana_error_score'].mean():>10.4f}"
            f"{s['IQR'].mean():>10.6f}{s['n_peso_menor_1'].mean():>9.1f}"
            f"{s['n_peso_igual_1'].mean():>9.1f}{s['peso_medio'].mean():>10.4f}"
            f"{s['peso_min'].mean():>8.3f}{s['peso_max'].mean():>8.3f}"
            f"{s['peso_total_retirado'].mean():>11.6f}")

    log("")
    log("   ITEM 10, ESTABILIDAD DE LOS PESOS ENTRE SUBMUESTRAS")
    pares = []
    claves = list(pesos_por_corrida)
    for i, a in enumerate(claves):
        for b in claves[i + 1:]:
            if not np.array_equal(ids_por_corrida[a], ids_por_corrida[b]):
                continue
            wa, wb = pesos_por_corrida[a], pesos_por_corrida[b]
            jac = float(((wa < 1) & (wb < 1)).sum() / max(((wa < 1) | (wb < 1)).sum(), 1))
            try:
                from scipy.stats import spearmanr
                rho = float(spearmanr(wa, wb).statistic)
            except Exception:
                rho = float(np.corrcoef(wa, wb)[0, 1])
            pares.append({"a": a, "b": b, "spearman": round(rho, 6),
                          "pearson": round(float(np.corrcoef(wa, wb)[0, 1]), 6),
                          "jaccard_peso_menor_1": round(jac, 6)})
    log(f"      pares comparables (mismos ids): {len(pares)} de "
        f"{len(claves) * (len(claves) - 1) // 2} posibles")
    log("      la mayoria NO son comparables: cada submuestra tiene ids distintos")
    if pares:
        for p in pares[:6]:
            log(f"      {p['a']:>7} vs {p['b']:>7} | spearman {p['spearman']:+.4f} | "
                f"pearson {p['pearson']:+.4f} | jaccard {p['jaccard_peso_menor_1']:.4f}")
    else:
        log("      ninguno: el submuestreo por grupo cambia los ids en cada corrida")
        log("      -> la comparacion directa de pesos entre tamanos NO es identificable asi")
    with CSV_W.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["a", "b", "spearman", "pearson", "jaccard_peso_menor_1"])
        for p in pares:
            w.writerow([p["a"], p["b"], p["spearman"], p["pearson"],
                        p["jaccard_peso_menor_1"]])
        if not pares:
            w.writerow(["SIN_PARES_COMPARABLES", "", "", "", ""])

    # ---------- ITEM 22: figuras 1, 2, 3 ----------
    log("")
    log("   ITEM 22, FIGURAS 1 a 3")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGURAS.mkdir(parents=True, exist_ok=True)
    hechos: list[str] = []

    def guardar(n: str) -> None:
        """Guarda la figura actual.

        Args:
            n: nombre del fichero.
        """
        plt.tight_layout()
        plt.savefig(FIGURAS / n, dpi=110)
        plt.close()
        hechos.append(n)
        log(f"      {n}")

    x = np.array([r["n_filas_medio"] for r in resumen])
    plt.figure(figsize=(10, 5))
    plt.errorbar(x, [r["cv_v03_medio"] for r in resumen],
                 yerr=[r["cv_v03_desv"] for r in resumen], marker="o", capsize=5,
                 label="V03, CV a ese tamano")
    plt.errorbar(x, [r["cv_v03_medio"] + r["delta_medio"] for r in resumen],
                 yerr=[r["delta_desv"] for r in resumen], marker="s", capsize=5,
                 label="candidato, con los multiplicadores congelados")
    plt.axhline(V03_CCV, color="k", ls="--", lw=1, label="V03 canonico en 1483")
    plt.xlabel("filas de entrenamiento (submuestreo por grupo)")
    plt.ylabel("corrected_cv")
    plt.title("Sensibilidad al tamano de entrenamiento: 3 repeticiones por tamano")
    plt.legend(fontsize=8)
    guardar("p70_01_cv_vs_tamano.png")

    plt.figure(figsize=(10, 5))
    plt.bar([f"{t}%" for t in TAMANOS],
            [r["cv_v03_desv"] for r in resumen], color="#4a7", label="desviacion de V03")
    plt.bar([f"{t}%" for t in TAMANOS],
            [r["delta_desv"] for r in resumen], color="#84c", alpha=0.7,
            label="desviacion del delta")
    plt.axhline(DELTA_CV, color="r", ls="--", lw=1,
                label=f"efecto del weighting {DELTA_CV:+.6f}")
    plt.ylabel("desviacion estandar entre repeticiones")
    plt.title("Variabilidad por tamano, frente al efecto que hay que explicar")
    plt.legend(fontsize=8)
    guardar("p70_02_variabilidad_vs_tamano.png")

    plt.figure(figsize=(10, 5))
    ancho = 0.35
    pos = np.arange(len(TAMANOS))
    plt.bar(pos - ancho / 2, [df[df["tamano_pct"] == t]["peso_total_retirado"].mean()
                              for t in TAMANOS], width=ancho, label="peso total retirado")
    plt.bar(pos + ancho / 2, [df[df["tamano_pct"] == t]["n_peso_menor_1"].mean()
                              / df[df["tamano_pct"] == t]["n_filas"].mean()
                              for t in TAMANOS], width=ancho,
            label="fraccion de ejemplos con peso < 1")
    plt.xticks(pos, [f"{t}%" for t in TAMANOS])
    plt.ylabel("proporcion")
    plt.title("La misma formula, aplicada a conjuntos de entrenamiento distintos")
    plt.legend(fontsize=8)
    guardar("p70_03_pesos_vs_tamano.png")

    salida = {
        "familia": "P70_POSTMORTEM_TRANSFERENCIA", "experiment_id": "P70-01",
        "fecha": inicio.isoformat(timespec="seconds"),
        "naturaleza": ("POST-MORTEM. 0 submissions nuevas, 0 test, 0 holdout, "
                       "0 leaderboard"),
        "resultado_externo": {"clase": "EXTERNAL_TRANSFER_RESULT",
                              "LB_V03": LB_V03, "LB_candidato": LB_CAND,
                              "delta_LB": DELTA_LB, "transferencia": "NEGATIVA"},
        "confound_de_tamano": confound,
        "sensibilidad": {"semillas": SEMILLAS_SUBMUESTRA, "tamanos": list(TAMANOS),
                         "resumen_por_tamano": resumen,
                         "desviacion_maxima": round(desv_peor, 10),
                         "rango_maximo": round(rango_peor, 10),
                         "caida_100_a_70": round(caida_70, 10),
                         "efecto_del_weighting": DELTA_CV,
                         "ratio_desviacion_sobre_efecto": round(desv_peor / DELTA_CV, 4)},
        "estabilidad_de_pesos": {"pares_comparables": len(pares),
                                 "nota": ("el submuestreo por grupo cambia los ids en cada "
                                          "corrida, luego la comparacion directa de pesos entre "
                                          "tamanos no es identificable por esa via")},
        "reproducibilidad": {"hash_v03": hash_real, "corrected_cv_v03": V03_CCV,
                             "seed_base": SEMILLA},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_S.write_text(json.dumps(salida, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")

    documentos = [CSV_T, CSV_W, JSON_S] + [FIGURAS / f for f in hechos]
    sucios = [d.name for d in documentos
              if d.suffix != ".png" and any(b > 127 for b in d.read_bytes())]
    if sucios:
        raise SystemExit(f"PARADA: documentos con bytes no ASCII: {sucios}")

    log("")
    log("=" * 78)
    log(f"P70-01 OK | {len(filas)} corridas | {len(hechos)} figuras")
    log(f"   desviacion maxima por tamano {desv_peor:.6f} | "
        f"ratio sobre el efecto {desv_peor / DELTA_CV:.2f}")
    log(f"   caida de V03 del 100% al 70%: {caida_70:+.6f}")
    log("   sin test, sin holdout, sin leaderboard")
    log("=" * 78)


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
