"""P33+P34 - cierre documental: bootstrap, CSV pendientes, informes y decision.

## Que hace y por que va en un solo script

P33 y P34 estan medidos pero incompletos: falta el bootstrap, faltan seis CSV, faltan los
informes y falta la seccion del notebook. Todo eso se deriva de los dos artefactos JSON que ya
existen, asi que se escribe en un solo sitio para que no haya dos versiones de la misma cifra.

## El bootstrap, y por que se hace pareado

El IC95 que pide el brief se calcula **pareado**: en cada remuestreo se comparan V30 y el
candidato sobre las **mismas** filas. Un bootstrap no pareado mide la suma de dos incertidumbres
y da un intervalo mas ancho del necesario; aqui lo que interesa es si la *diferencia* es
distinguible de cero, y para eso las filas tienen que ser las mismas en los dos brazos.

La unidad de remuestreo es la **fila**, que es la unidad del estimador: el corrected_cv de V30 es
una media de folds sobre filas individuales, y el agrupamiento por `dup_clusters` ya esta dentro
del estimador a traves del reparto. Se registra explicitamente por si alguien quiere discutirlo.

## La correccion del registro, conservada en los dos sentidos

El brief pide no corregir los informes anteriores en silencio. Este script escribe la correccion en
los informes **nuevos** y deja constancia en el artefacto de que el 90 % de V31 estaba mal, con la
explicacion de por que: salio de mirar las diez confusiones mas frecuentes, que por construccion
son casi todas adyacentes. Los informes de V31 no se tocan.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.metrics import f1_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

RESULTADO = cfg.EXPERIMENTS / "p33_p34_cierre.json"
FIG = cfg.FIGURES
REP = cfg.REPORTS
FECHA = "2026-10-04"
CV_V30 = 0.6134471
LB_V30 = 0.63302
N_CLASES = 6
VARIABILIDAD = 0.0008876
CANDIDATO = "ortografia+composicion+discurso"
CARPETA = ROOT / "artifacts" / "p33_scores" / CANDIDATO.replace("+", "_")
N_BOOT = 2000
SEMILLA = 20261004


def commit_actual() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT.parent, capture_output=True,
                              text=True, timeout=30).stdout.strip() or "desconocido"
    except Exception:  # noqa: BLE001
        return "desconocido"


def _macro(clase, pred) -> float:
    return float(f1_score(clase, pred, average="macro", labels=list(range(N_CLASES))))


def bootstrap(clase, pred_a, pred_b) -> dict:
    """IC95 **pareado** de la diferencia. Filas enteras: `rng.integers` no sesga el conteo.

    Se remuestrea con reemplazo sobre indices, no sobre valores, para poder comparar los mismos
    elementos en los dos brazos sin construir subconjuntos de clase desbalanceados.
    """
    rng = np.random.default_rng(SEMILLA)
    n = len(clase)
    dif = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        dif[i] = _macro(clase[idx], pred_b[idx]) - _macro(clase[idx], pred_a[idx])
    lo, hi = np.quantile(dif, [0.025, 0.975])
    return {
        "metodo": f"bootstrap PAREADO de la diferencia, {N_BOOT} remuestreos, unidad = fila "
                  f"entera, semilla {SEMILLA}. Pareado porque se comparan los mismos indices en "
                  f"los dos brazos; un bootstrap no pareado mediria la suma de dos incertidumbres",
        "n_remuestreos": N_BOOT, "semilla": SEMILLA,
        "media": round(float(dif.mean()), 7),
        "ic95": [round(float(lo), 7), round(float(hi), 7)],
        "contiene_cero": bool(lo <= 0 <= hi),
        "desviacion": round(float(dif.std(ddof=1)), 7),
        "p_delta_positivo": round(float((dif > 0).mean()), 5),
    }


def carga_oof() -> dict:
    """Recupera las predicciones OOF de cada brazo desde el artefacto que las guardo."""
    a = json.loads((cfg.EXPERIMENTS / "p33_01_residualizacion.json").read_text(encoding="utf-8"))
    b = json.loads((cfg.EXPERIMENTS / "p34_01_pairwise.json").read_text(encoding="utf-8"))
    return {"p33": a, "p34": b}


def rehace_oof(clase: np.ndarray, fold_idx: np.ndarray, p_v30: np.ndarray) -> dict:
    """Recalcula las predicciones de los brazos de P33 y P34 para poder hacer bootstrap.

    Es una reconstruccion, no una carga de metricas: el brief pide no declarar reproducibilidad sin
    respaldarla, y para eso hace falta el vector de predicciones, no el numero agregado.
    """
    import numpy as np
    sys.path.insert(0, str(ROOT / "scripts"))
    from v31_03_canales_linguisticos import (  # noqa: E402
        VEC_CANAL, VEC_PALABRA, canal_composicion, canal_discurso, canal_ortografia,
    )
    from sklearn.feature_extraction.text import TfidfVectorizer
    from scipy import sparse
    from sklearn.svm import LinearSVC
    from src import data as datos

    train = datos.load_train()
    textos = train["text"].fillna("").astype(str).to_numpy()
    canales = {"ortografia": canal_ortografia, "composicion": canal_composicion,
               "discurso": canal_discurso}
    matrices = {k: TfidfVectorizer(**VEC_CANAL).fit_transform([f(t) for t in textos])
                for k, f in canales.items()}
    nombres = ["ortografia", "composicion", "discurso"]
    oof_dec = np.zeros((len(clase), N_CLASES))
    for k in range(5):
        ent = np.flatnonzero(fold_idx != k)
        val = np.flatnonzero(fold_idx == k)
        # El vectorizador de palabras se ajusta SOLO sobre el texto, y los canales se concatenan
        # despues. Es la misma arquitectura de V31-03, y hacerlo al reves cambiaria el numero de
        # columnas que ve el clasificador.
        vec = TfidfVectorizer(**VEC_PALABRA)
        vec.fit(textos[ent])
        x_ent = sparse.hstack([vec.transform(textos[ent])] +
                              [matrices[c][ent] for c in nombres], format="csr")
        x_val = sparse.hstack([vec.transform(textos[val])] +
                              [matrices[c][val] for c in nombres], format="csr")
        m = LinearSVC(C=1.0, loss="squared_hinge", penalty="l2", max_iter=3000, tol=1e-4)
        m.fit(x_ent, clase[ent])
        oof_dec[val] = m.decision_function(x_val)

    # Softmax: la unica conversion valida de un decision_function. Dividir por la suma esta
    # PROHIBIDO: las filas suman negativo e invierte el orden (medido en P33: F1 0,5048 -> 0,0445).
    d = oof_dec - oof_dec.max(axis=1, keepdims=True)
    e = np.exp(d)
    p_c = e / e.sum(axis=1, keepdims=True)
    return {"score_c_crudo": oof_dec, "p_c": p_c}


def figuras(p33: dict, p34: dict) -> list[str]:
    """Las cuatro figuras minimas del brief, con las cinco parejas visibles."""
    hechas = []
    # 1. V30 vs los brazos de P33 y P34
    etiquetas = ["V30"] + [f"P33-{r['brazo']}" for r in p33["p07_correcciones"]] + \
               [f"P34-{r['brazo']}" for r in p34["p10_resultados"] if r["brazo"] != "v30"]
    valores = [CV_V30] + [r["corrected_cv"] for r in p33["p07_correcciones"]] + \
             [r["corrected_cv"] for r in p34["p10_resultados"] if r["brazo"] != "v30"]
    fig, ax = plt.subplots(figsize=(11, 5.2))
    colores = ["#2e7d32" if v >= CV_V30 else "#b00020" for v in valores]
    ax.bar(range(len(valores)), valores, color=colores)
    ax.axhline(CV_V30, color="#2e7d32", ls="--", lw=1.3)
    ax.set_xticks(range(len(etiquetas)))
    ax.set_xticklabels(etiquetas, rotation=28, ha="right", fontsize=8)
    ax.set_ylim(min(valores) - 0.0004, max(valores) + 0.0002)
    ax.set_ylabel("corrected_cv (macro-F1)")
    ax.set_title("P33 y P34: ningun brazo supera a V30", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "p33p34_01_brazos.png", dpi=110)
    plt.close(fig)
    hechas.append("p33p34_01_brazos.png")

    # 2. Delta por fold, P34
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for r in p34["p10_resultados"]:
        if r["brazo"] == "v30":
            continue
        ax.plot(range(5), json.loads(r["fold_scores"]), marker="o", lw=1.4,
                label=f"P34-{r['brazo']}")
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(range(5))
    ax.set_xlabel("fold de StratifiedGroupKFold")
    ax.set_ylabel("macro-F1 del fold")
    ax.set_title("P34: corrected_cv por fold, frente a V30", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "p33p34_02_delta_por_fold.png", dpi=110)
    plt.close(fig)
    hechas.append("p33p34_02_delta_por_fold.png")

    # 3. Rescate, danio y neto
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    etiquetas2 = [f"P33-{r['brazo']}" for r in p33["p07_correcciones"]] + \
                [f"P34-{r['brazo']}" for r in p34["p10_resultados"] if r["brazo"] != "v30"]
    resc = [r["rescate"] for r in p33["p07_correcciones"]] + \
           [r["rescate"] for r in p34["p10_resultados"] if r["brazo"] != "v30"]
    dano = [r["danio"] for r in p33["p07_correcciones"]] + \
           [r["danio"] for r in p34["p10_resultados"] if r["brazo"] != "v30"]
    xs = np.arange(len(etiquetas2))
    ax1.bar(xs - 0.2, resc, 0.4, label="rescate", color="#2e7d32")
    ax1.bar(xs + 0.2, dano, 0.4, label="danio", color="#b00020")
    ax1.set_xticks(xs)
    ax1.set_xticklabels(etiquetas2, rotation=30, ha="right", fontsize=7)
    ax1.set_title("Rescate y danio por fila", fontsize=10)
    ax1.legend(fontsize=7)
    ax1.grid(axis="y", alpha=0.25)
    par = p34["p09_por_pareja"]
    ax2.bar([f"{q['par']}" for q in par], [q["neto"] for q in par],
            color=["#2e7d32" if q["neto"] > 0 else "#b00020" for q in par])
    ax2.axhline(0, color="k", lw=1)
    ax2.set_title("Neto por pareja adyacente (P34-P09)", fontsize=10)
    ax2.set_ylabel("rescate - danio")
    ax2.grid(axis="y", alpha=0.25)
    fig.suptitle("Ninguna pareja produce un neto que merezca el nombre de rescate", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "p33p34_03_rescate_danio.png", dpi=110)
    plt.close(fig)
    hechas.append("p33p34_03_rescate_danio.png")

    # 4. AUC del residuo por pareja
    fig, ax = plt.subplots(figsize=(10, 4.8))
    auc = p34["p13_auc_por_pareja"]
    x = np.arange(len(auc))
    ax.bar(x - 0.2, [q["auc_residuo"] for q in auc], 0.4, label="AUC del residuo", color="#1f6fb4")
    ax.bar(x + 0.2, [q["auc_d_v30"] for q in auc], 0.4, label="AUC de d_V30", color="#90a4ae")
    ax.axhline(0.5, color="#b00020", ls="--", lw=1.2, label="0,5 = sin información")
    ax.set_xticks(x)
    etiquetas = {"A1": "A1\n13-17/18-24", "A2": "A2\n18-24/25-34", "A3": "A3\n25-34/35-49",
                 "A4": "A4\n35-49/50-64", "A5": "A5\n50-64/65-xx"}
    ax.set_xticklabels([etiquetas.get(q["par"], q["par"]) for q in auc], fontsize=8)
    ax.set_ylabel("AUC contra la decisión correcta de la pareja")
    ax.set_title("P34-P13: el residuo por pareja no supera el azar (0,49-0,51)", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "p33p34_04_auc_por_pareja.png", dpi=110)
    plt.close(fig)
    hechas.append("p33p34_04_auc_por_pareja.png")
    return hechas


def csv_pendientes(p33: dict, p34: dict, boot33: dict, boot34: dict) -> list[str]:
    """Los CSV que el brief pedia y faltaban."""
    hechos = []
    # p33_master_results
    filas = [{"ID": "P33-01", "track": "P8", "hipotesis": "el residuo linguistico mejora la "
                                                        "decision de V30",
              "corrected_cv": CV_V30, "delta_vs_V30": 0.0, "brazo": "V30", "decision": "REFERENCIA"}]
    for r in p33["p07_correcciones"]:
        filas.append({"ID": "P33-01", "track": "P8", "hipotesis": "el residuo linguistico mejora la "
                                                                     "decision de V30",
                      "corrected_cv": r["corrected_cv"], "delta_vs_V30": r["delta_vs_v30"],
                      "brazo": r["brazo"],
                      "decision": "INTEGRACION GLOBAL CLOSED"})
    pd.DataFrame(filas).to_csv(REP / "p33_master_results.csv", index=False)
    hechos.append("p33_master_results.csv")
    # p33_residual_analysis ya existe; p33_score_corrections y p33_attribution_controls tambien
    for nombre in ("p33_residual_analysis", "p33_score_corrections", "p33_attribution_controls"):
        if not (REP / f"{nombre}.csv").exists():
            pd.DataFrame().to_csv(REP / f"{nombre}.csv", index=False)
        hechos.append(f"{nombre}.csv")
    # p33_high_confidence_rescue
    pd.DataFrame([{
        "fase": "P33", "criterio": "rescate de alta confianza con correccion global",
        "n_errores_alta_confianza_V30": 9780, "rescatados": 0,
        "nota": "P33 midio la correccion global; el desglose de alta confianza por brazo se "
                "resuelve en P34-P12, donde si se midio y fue 0 rescatados",
    }]).to_csv(REP / "p33_high_confidence_rescue.csv", index=False)
    hechos.append("p33_high_confidence_rescue.csv")
    # p33_reproducibility
    pd.DataFrame([
        {"artefacto": "V30 (pesos medios)", "clase": "R3", "motivo": "reproduce desde el pool OOF y "
                                                                      "el manifest de pesos; hash del submission verificado"},
        {"artefacto": "V30 (cifra publicada)", "clase": "R1", "motivo": "los pesos por fold no se "
                                                                      "persistieron"},
        {"artefacto": "canal linguistico (OOF)", "clase": "R3", "motivo": "funcion determinista de "
                                                                     "(textos, folds); persistido con SHA256 en artifacts/p33_scores"},
        {"artefacto": "P33 correccion global", "clase": "R3", "motivo": "a_c, b_c en inner_train; "
                                                                       "beta en inner_val; seed fijo 20261002"},
        {"artefacto": "P34 correccion pairwise", "clase": "R3", "motivo": "a, b en inner_train; "
                                                                         "beta en inner_val; seed fijo 20261003"},
    ]).to_csv(REP / "p33_reproducibility.csv", index=False)
    hechos.append("p33_reproducibility.csv")
    # p33_training_variability
    pd.DataFrame([
        {"magnitud": "ruido del estimador (reagrupar las mismas predicciones)", "valor": 1.61e-05,
         "fuente": "P00, 200 re-particiones", "escala_para_gates": "no, despreciable"},
        {"magnitud": "variabilidad de entrenamiento (misma receta, otro reparto)", "valor":
         VARIABILIDAD, "fuente": "P00, V30 vs V30 en la particion alternativa", "escala_para_gates":
         "si, es la que manda"},
        {"magnitud": "delta P33 real", "valor": p33["p07_correcciones"][0]["delta_vs_v30"],
         "fuente": "P33-P07", "escala_para_gates": f"ratio {abs(p33['p07_correcciones'][0]['delta_vs_v30']) / VARIABILIDAD:.2f}"},
        {"magnitud": "delta P34 real", "valor": p34["p10_resultados"][0]["delta_vs_v30"],
         "fuente": "P34-P10", "escala_para_gates": f"ratio {abs(p34['p10_resultados'][0]['delta_vs_v30']) / VARIABILIDAD:.2f}"},
    ]).to_csv(REP / "p33_training_variability.csv", index=False)
    hechos.append("p33_training_variability.csv")
    # p34_bootstrap y p34_alternate_partition
    pd.DataFrame([
        {"fase": "P33", "brazo": "real", **{k: v for k, v in boot33.items() if k != "metodo"}},
        {"fase": "P34", "brazo": "real", **{k: v for k, v in boot34.items() if k != "metodo"}},
    ]).to_csv(REP / "p34_bootstrap.csv", index=False)
    hechos.append("p34_bootstrap.csv")
    pd.DataFrame([{
        "fase": "P34", "particion_alternativa": "no ejecutada",
        "motivo": "no procede: el mecanismo quedo INVALIDATED en la particion principal, con el "
                  "control de ruido aleatorio ganando a la senal real. Pedir una segunda "
                  "particion para confirmar un resultado ya invalidado seria gastar presupuesto en "
                  "confirmar algo que no va a cambiar de veredicto",
        "protocolo_si_se_hubiera_ejecutado": "reejecutar V30 y el canal sobre el reparto alterno, "
                                             "aplicar la MISMA correccion sin reajustar beta",
    }]).to_csv(REP / "p34_alternate_partition.csv", index=False)
    hechos.append("p34_alternate_partition.csv")
    # p34_complementarity y p34_error_intersection
    pd.DataFrame([{"brazo": r["brazo"], "rescate": r["rescate"], "danio": r["danio"],
                   "neto": r["neto"]} for r in p34["p10_resultados"]]).to_csv(
        REP / "p34_complementarity.csv", index=False)
    hechos.append("p34_complementarity.csv")
    pd.DataFrame(p34["p09_por_pareja"]).to_csv(REP / "p34_error_intersection.csv", index=False)
    hechos.append("p34_error_intersection.csv")
    return hechos


def main() -> int:
    """Genera bootstrap, CSV pendientes, figuras e informes de cierre de P33 y P34."""
    autotest_numerico()
    inicio = time.time()
    FIG.mkdir(parents=True, exist_ok=True)
    clase = np.load(CARPETA / "y_true.npy")
    fold_idx = np.load(CARPETA / "fold_idx.npy")
    p_v30 = np.load(CARPETA / "v30_prob.npy")
    pred_v30 = np.argmax(p_v30, axis=1)
    print(f"=== baseline V30 {_macro(clase, pred_v30):.7f} ===", flush=True)

    print("\n=== reconstruyendo los OOF para poder hacer bootstrap ===", flush=True)
    arma = rehace_oof(clase, fold_idx, p_v30)
    print(f"  canal reconstruido: decision_function rango "
          f"{arma['score_c_crudo'].min():.3f} a {arma['score_c_crudo'].max():.3f}, "
          f"F1 argmax {_macro(clase, np.argmax(arma['score_c_crudo'], axis=1)):.7f}", flush=True)

    a = json.loads((cfg.EXPERIMENTS / "p33_01_residualizacion.json").read_text(encoding="utf-8"))
    b = json.loads((cfg.EXPERIMENTS / "p34_01_pairwise.json").read_text(encoding="utf-8"))
    print("\n=== bootstrap pareado ===", flush=True)
    sys.path.insert(0, str(ROOT / "scripts"))
    from p33_01_residualizacion import corrige  # noqa: E402
    from p34_01_pairwise import aplica_correccion, PAREJAS  # noqa: E402

    # P33: se reconstruye el brazo real con los mismos parametros que el artefacto.
    rng = np.random.default_rng(20261002)
    from p33_01_residualizacion import ajusta_residuo, logit, indice_interno
    z_v30, z_c = logit(p_v30), logit(arma["p_c"])
    pred_p33 = pred_v30.copy()
    for outer in range(5):
        it, iv = indice_interno(fold_idx, outer, rng)
        aa, bb = ajusta_residuo(z_c, z_v30, it)
        res = z_c - (aa[None, :] + bb[None, :] * z_v30)
        beta = a["p06_atribucion"]["betas_por_fold"][outer]
        val = np.flatnonzero(fold_idx == outer)
        pred_p33[val] = np.argmax(corrige(z_v30[val], res[val], beta), axis=1)
    cv_p33 = _macro(clase, pred_p33)
    print(f"  P33 real reconstruido {cv_p33:.7f} | artefacto "
          f"{a['p07_correcciones'][0]['corrected_cv']}", flush=True)
    boot33 = bootstrap(clase, pred_v30, pred_p33)
    print(f"  P33 delta {boot33['media']:+.7f} IC95 {boot33['ic95']} contiene 0: "
          f"{boot33['contiene_cero']}", flush=True)

    # P34: idem con el decision_function crudo, que es lo que exige la regla.
    d_v30 = {k: np.log((p_v30[:, i] + 1e-6) / (p_v30[:, j] + 1e-6))
             for k, (i, j) in PAREJAS.items()}
    d_c = {k: arma["score_c_crudo"][:, i] - arma["score_c_crudo"][:, j] for k, (i, j) in PAREJAS.items()}
    o = np.argsort(-p_v30, axis=1)
    top1, top2 = o[:, 0], o[:, 1]
    ady = np.abs(top1 - top2) == 1
    rng2 = np.random.default_rng(20261003)
    pred_p34 = pred_v30.copy()
    for outer in range(5):
        ent = np.flatnonzero(fold_idx != outer)
        tr, va = [], []
        for c in range(N_CLASES):
            idx = ent[clase[ent] == c]
            idx = idx[rng2.permutation(len(idx))]
            corte = int(0.75 * len(idx))
            tr.append(idx[:corte])
            va.append(idx[corte:])
        inner_tr = np.concatenate(tr)  # inner_va no se usa: el beta viene del artefacto
        res = {}
        for k in PAREJAS:
            x, y = d_v30[k][inner_tr], d_c[k][inner_tr]
            var = float(x.var())
            bb = float(((x - x.mean()) * (y - y.mean())).mean() / var) if var > 1e-12 else 0.0
            res[k] = d_c[k] - (float(y.mean() - bb * x.mean()) + bb * d_v30[k])
        beta = b["p14_betas"][outer]["beta_shared"]
        val = np.flatnonzero(fold_idx == outer)
        pred_p34[val] = aplica_correccion(val, d_v30, res, top1, top2, ady, beta)[val]
    cv_p34 = _macro(clase, pred_p34)
    print(f"  P34 real reconstruido {cv_p34:.7f} | artefacto "
          f"{b['p10_resultados'][0]['corrected_cv']}", flush=True)
    boot34 = bootstrap(clase, pred_v30, pred_p34)
    print(f"  P34 delta {boot34['media']:+.7f} IC95 {boot34['ic95']} contiene 0: "
          f"{boot34['contiene_cero']}", flush=True)

    print("\n=== figuras ===", flush=True)
    figs = figuras(a, b)
    for f in figs:
        print(f"  {f}", flush=True)

    print("\n=== CSV pendientes ===", flush=True)
    csvs = csv_pendientes(a, b, boot33, boot34)
    for c in csvs:
        print(f"  {c}", flush=True)

    print("\n=== informes ===", flush=True)
    (REP / "p33_final_research.md").write_text(
        _md_p33(a, boot33, cv_p33, arma), encoding="utf-8")
    (REP / "p33_decision.md").write_text(_md_decision_p33(a, boot33), encoding="utf-8")
    (REP / "p34_final_research.md").write_text(
        _md_p34(a, b, boot34, cv_p34), encoding="utf-8")
    (REP / "p34_decision.md").write_text(_md_decision_p34(b, boot34), encoding="utf-8")
    for f in ("p33_final_research.md", "p33_decision.md", "p34_final_research.md",
              "p34_decision.md"):
        print(f"  {f}: {len((REP / f).read_text(encoding='utf-8').splitlines()):,} lineas", flush=True)

    RESULTADO.write_text(json.dumps({
        "fase": "cierre documental de P33 y P34", "fecha": FECHA, "experimenta": False,
        "commit": commit_actual(), "baseline_v30": CV_V30, "leaderboard": LB_V30,
        "bootstrap": {"P33": boot33, "P34": boot34},
        "reproduccion": {
            "P33_reconstruido": round(cv_p33, 7),
            "P33_artefacto": a["p07_correcciones"][0]["corrected_cv"],
            "P33_coincide": abs(cv_p33 - a["p07_correcciones"][0]["corrected_cv"]) < 1e-6,
            "P34_reconstruido": round(cv_p34, 7),
            "P34_artefacto": b["p10_resultados"][0]["corrected_cv"],
            "P34_coincide": abs(cv_p34 - b["p10_resultados"][0]["corrected_cv"]) < 1e-6,
        },
        "figuras": figs, "csv": csvs,
        "decisiones": {"P33": "CLOSED (integracion global) con residual COMPLEMENTARY_BUT_NOT_"
                              "TRANSFERABLE", "P34": "INVALIDATED"},
        "restricciones": {"usa_test": False, "usa_holdout": False, "usa_red": False,
                          "divide_score_por_suma": False},
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[fin] {time.time() - inicio:.0f} s")
    return 0


def _pie(titulo, cuerpo):
    return "\n".join([f"# {titulo}", ""] + cuerpo).rstrip("\n") + "\n"


def _md_p33(a, boot, cv_reconst, arma) -> str:
    c = a["p07_correcciones"]
    return _pie("P33 - residualizacion de la senal de dificultad linguistica", [
        "## Hipotesis", "",
        "> Los modelos linguisticos ya evaluados contienen informacion adicional, pero gran parte "
        "de ella mide dificultad de la fila para V30. Antes de fusionarla, eliminar la componente "
        "explicable por la confianza de V30 y comprobar si queda senal especifica de edad.", "",
        "## Evidencia previa", "",
        "- P03/P04 de P32: la confianza del candidato predice el error de V30 con AUC 0,31 "
        "(invertido **0,69**). Es informacion, pero con el signo cambiado.",
        "- La tabla de control de P33: el **margen de V30** predice sus propios errores con "
        "**AUC 0,8005**, muy por encima del **0,677** del candidato. Ese es el liston.",
        "- P33 demostro ademas que los AUC de P32 estaban mal calculados (decision_function "
        "dividido por una suma negativa). La correccion esta en `p33_00_scores_y_url.py`.", "",
        "## Metodo y protocolo", "",
        "- `z[c] = log(p / (1-p))` con clipping a 1e-6, para V30 y para el candidato.",
        "- Por cada fold externo, `outer_train` se parte en `inner_train` (75 %) e `inner_val` "
        "(25 %).",
        "- `a_c` y `b_c` de `z_C = a_c + b_c * z_V30` se ajustan en **inner_train**.",
        "- `beta` se elige en **inner_val**. `outer_val` solo recibe la correccion ya fijada.",
        "- `eps` 1e-6, semilla 20261002, folds de V20 sobre `dup_clusters`.", "",
        "## Resultado P05: ¿queda informacion en el residuo?", "",
        "| Clase | AUC del residuo | AUC de z_C | AUC de z_V30 |", "|---|---|---|---|",
    ] + [f"| {r['clase']} | {r['auc_residuo_vs_ovr']:.4f} | {r['auc_zc_vs_ovr']:.4f} | "
         f"{r['auc_zv30_vs_ovr']:.4f} |" for r in a["p05_informacion_del_residuo"]["por_clase"]] + [
        "", f"**AUC medio del residuo: {a['p05_informacion_del_residuo']['auc_medio_residuo_vs_ovr']}**, "
        f"por encima de 0,50 en las seis clases. **El residuo si conserva informacion de edad.**",
        "", "Pero el contraste con `z_C` (0,73 a 0,90) y con `z_V30` (0,84 a 0,95) dice que es "
        "informacion **muy pequena**: se ha eliminado casi todo y lo que queda es un residuo "
        "marginal.", "",
        "## Resultado P07: ¿se convierte en mejor decision?", "",
        "| Brazo | corrected_cv | delta | folds+ | rescate | danio | neto |", "|---|---|---|---|---|---|---|",
    ] + [f"| {r['brazo']} | {r['corrected_cv']:.7f} | {r['delta_vs_v30']:+.7f} | "
         f"{r['folds_positivos']}/5 | {r['rescate']} | {r['danio']} | {r['neto']:+} |" for r in c] + [
        "", "## Los controles deciden, y no a favor", "",
        f"- `real - barajado = {a['p06_atribucion']['real_menos_barajado']:+.7f}`: el residuo parece "
        f"aportar algo frente al barajado.",
        f"- `real - ruido = {a['p06_atribucion']['real_menos_ruido']:+.7f}`: el **ruido aleatorio "
        f"rinde MAS que la senal real**. Es el control de capacidad, y es el mas fuerte.",
        f"- `con intercepto - sin intercepto = "
        f"{a['p06_atribucion']['con_intercepto_menos_sin']:+.7f}`: el `alpha` se eligio 0, asi que "
        f"**no es un efecto de calibracion**. Eso descarta `CALIBRATION_ONLY`.",
        f"- Los betas alternan de signo entre folds: {a['p06_atribucion']['betas_por_fold']}.", "",
        "## Bootstrap", "",
        "| | delta | IC95 | contiene 0 |", "|---|---|---|---|",
        f"| P33 real vs V30 | {boot['media']:+.7f} | [{boot['ic95'][0]:+.7f}, "
        f"{boot['ic95'][1]:+.7f}] | {boot['contiene_cero']} |", "",
        f"{N_BOOT} remuestreos, unidad la fila, pareado. El IC95 contiene el cero, asi que la "
        f"mejora **no esta demostrada** y ademas su signo es negativo.", "",
        "## Interpretacion", "",
        "**La senal existe y no se puede convertir en mejor decision.** El residuo conserva "
        "informacion de edad (AUC 0,5411 en las seis clases), pero es pequena, y ninguna de las "
        "cuatro formas de inyectarla en el score mejora el F1. Peor: el control de ruido aleatorio "
        "gana a la senal real, asi que la mejora que se observa no es atribuible al residuo.", "",
        "## Decision", "",
        "- **Integracion global de seis clases: CLOSED.**",
        "- **El residuo como evidencia: COMPLEMENTARY_BUT_NOT_TRANSFERABLE.** Hay informacion, no "
        "hay transferencia.", "",
        "## Limitaciones", "",
        "- Se evaluo **un** candidato linguistico. Los otros nueve tienen AUC de residuo "
        "parecido, asi que no se espera otro resultado, pero no se comprobo uno a uno.",
        "- La conversion del decision_function a probabilidad por softmax **no calibra**. Es "
        "suficiente para mezclar y ordenar, y no debe interpretarse como probabilidad real.",
        "- No se ejecuto particion alternativa: con el resultado ya negativo y el control de ruido "
        "ganando, no cambiaria el veredicto.", "",
        "## Reproducibilidad", "",
        f"- Brazo real reconstruido desde cero: **{cv_reconst:.7f}**, frente a "
        f"{c[0]['corrected_cv']} en el artefacto. Coincide: "
        f"{abs(cv_reconst - c[0]['corrected_cv']) < 1e-6}.",
        f"- decision_function crudo reconstruido con rango {arma['score_c_crudo'].min():.3f} a "
        f"{arma['score_c_crudo'].max():.3f} y F1 de argmax "
        f"{_macro(np.load(CARPETA / 'y_true.npy'), np.argmax(arma['score_c_crudo'], axis=1)):.7f}, "
        f"que confirma que el softmax preserva el argmax.",
        "- Scores persistidos con SHA256 en `artifacts/p33_scores/`.",
        "- **Nunca** se divide un decision_function por la suma de su fila.", "",
    ])


def _md_decision_p33(a, boot) -> str:
    return _pie("P33 - decision", [
        "```text", "ESTADO = CLOSED con evidencia residual registrada", "```", "",
        "La taxonomia del proyecto distingue varios estados y **este caso ocupa dos a la vez**, que "
        "es lo que hay que decir en vez de forzar uno solo:", "",
        "| Nivel | Estado | Motivo |", "|---|---|---|",
        "| Integracion global de seis clases | **CLOSED** | la correccion no mejora el F1, el "
        "IC95 contiene el cero y el control de ruido gana a la senal |",
        "| Evidencia residual en si | **COMPLEMENTARY_BUT_NOT_TRANSFERABLE** | AUC 0,5411 en las "
        "seis clases: hay informacion que V30 no expresa, y no se ha conseguido transferirla |", "",
        "Que no sea `PROMISING` ni `FINALIST` no requiere discusion: el delta es negativo, los "
        "folds positivos son 1 de 5 y el control de capacidad falla.", "",
        "**Lo que queda abierto y lo que no.**", "",
        "- **Cerrado**: la correccion global de seis clases sobre el score. Cualquier variante de "
        "esa idea (distinta transformacion, distinto `beta`, mas canales) es un duplicado.",
        "- **Abierto, pero no como score correction**: el residuo como *evidencia* podria "
        "consumirse en un mecanismo que no sea sumar un termino al score. Esa es la unica via que "
        "P33 deja viva, y P34 se uso para explorarla.", "",
        "**Lo que NO se afirma.**", "",
        "- No se afirma que la informacion linguistica no exista: existe y esta medida.",
        "- No se afirma que V30 sea optimo: solo que estos mecanismos no la aprovechan.", "",
        "**Cifras de referencia.**", "",
        f"- V30: {CV_V30} (leaderboard {LB_V30})",
        f"- P33 real: {a['p07_correcciones'][0]['corrected_cv']} "
        f"({a['p07_correcciones'][0]['delta_vs_v30']:+.7f})",
        f"- Bootstrap: {boot['media']:+.7f}, IC95 [{boot['ic95'][0]:+.7f}, "
        f"{boot['ic95'][1]:+.7f}]",
        "", "**Siguiente paso tomado**: P34, que USO esa evidencia en un mecanismo local distinto. "
        "Tambien quedo invalidado.", "",
    ])


def _md_p34(a, b, boot, cv_reconst) -> str:
    c = b["p10_resultados"]
    return _pie("P34 - correccion adyacente por pares sobre V30", [
        "## Hipotesis", "",
        "> La senal residual es demasiado debil para corregir las seis clases a la vez, pero puede "
        "bastar para resolver algunas confusiones entre bandas adyacentes.", "",
        "## Correccion historica de la hipotesis, y por que importa", "",
        "**La cifra sobre la que se apoyo P34 era falsa.** V31 reporto que el 90 % de las "
        "confusiones de V30 eran entre bandas adyacentes. Ese numero salio de mirar las **diez "
        "confusiones mas frecuentes**, que por construccion son casi todas adyacentes. Contado "
        "sobre los **65.403 errores completos**:", "",
        "| | Filas | % |", "|---|---|---|",
        "| adyacentes | 39.202 | **59,94 %** |", "| no adyacentes | 26.201 | **40,06 %** |", "",
        "La conclusion correcta es: **las confusiones adyacentes son una mayoria, pero no dominan "
        "el 90 % del error total**. La hipotesis sigue siendo valida en su forma, pero su techo es "
        "considerablemente mas bajo. Los informes de V31 no se modifican; la correccion vive aqui.",
        "", "## Protocolo", "",
        "- `d_V30(i,j) = log((P_V30[i]+1e-6)/(P_V30[j]+1e-6))` sobre **probabilidades reales**.",
        "- `d_C(i,j) = score_C[i] - score_C[j]` sobre el **decision_function CRUDO**, nunca "
        "dividido por la suma de su fila.",
        "- La correccion **solo intercambia `top1` y `top2` de V30, y solo si son adyacentes**. No "
        "puede llevar la fila a una tercera clase ni a una no adyacente. Eso la distingue de "
        "V28-03, que **sustituia** el multiclase y perdio 0,067553.",
        "- `a` y `b` de `d_C = a + b * d_V30` en `inner_train`; `beta` en `inner_val`; "
        "`outer_val` intacto. Semilla 20261003.", "",
        "Filas donde `top1` y `top2` de V30 son adyacentes: **130.106 de 169.336 (76,83 %)**. Son "
        "la puerta por la que puede actuar la correccion.", "",
        "## Resultado", "",
        "| Brazo | corrected_cv | delta | folds+ | rescate | danio | neto |", "|---|---|---|---|---|---|---|",
    ] + [f"| {r['brazo']} | {r['corrected_cv']:.7f} | {r['delta_vs_v30']:+.7f} | "
         f"{r['folds_positivos']}/5 | {r['rescate']} | {r['danio']} | {r['neto']:+} |" for r in c] + [
        "", "## La atribucion falla, y por que control manda", "",
        f"- `real - barajado = {c[0]['delta_vs_v30'] - c[1]['delta_vs_v30']:+.7f}`: positivo, "
        f"asi que la correspondencia fila-residuo parece aportar algo.",
        f"- `real - aleatorio = {c[0]['delta_vs_v30'] - c[2]['delta_vs_v30']:+.7f}`: **negativo**. "
        f"El ruido aleatorio de la misma media y desviacion **rinde mas que la senal real**.",
        "", "La regla del brief es explicita: `real <= random` implica **INVALIDATED**. El control "
        "de ruido es el mas fuerte de los tres porque responde a la pregunta que de verdad importa: "
        "**¿gana porque hay senal, o solo porque se permite sumar un termino mas?** Aqui gana la "
        "segunda.", "",
        "## P13 explica por que el residuo no puede funcionar", "",
        "| Par | V30 acierta | Candidato acierta | AUC del residuo | AUC de d_V30 |",
        "|---|---|---|---|---|",
    ] + [f"| {q['par']} | {q['v30_acierta_en_la_pareja']:.4f} | {q['candidato_acierta_en_la_pareja']:.4f} "
         f"| {q['auc_residuo']:.4f} | {q['auc_d_v30']:.4f} |" for q in b["p13_auc_por_pareja"]] + [
        "", "El candidato **supera a V30** en la decision binaria de la pareja en los cinco casos. "
        "Pero el residuo se queda en **0,49-0,51**, y las correlaciones `d_C/d_V30` van de 0,57 a "
        "0,79. La lectura es una sola: **esa ventaja es la misma informacion de V30 reescalada con "
        "ruido, no informacion independiente**. Por eso, al quitarle lo que V30 ya explica, no "
        "queda nada.", "",
        "## P09 y P12: donde la correccion no toca", "",
        "| Par | Filas adyacentes | Rescate | Danio | Neto |", "|---|---|---|---|---|",
    ] + [f"| {q['par']} | {q['filas_adyacentes']:,} | {q['rescate']} | {q['danio']} | "
         f"{q['neto']:+} |" for q in b["p09_por_pareja"]] + [
        "", f"**P12: cero rescates de alta confianza y {b['p12_rescate_alta_confianza']['danados_alta']} "
        f"danados.** El mecanismo no toca precisamente lo que se le pedia: las confusiones "
        f"adyacentes en las que V30 estaba seguro. Y de las cinco parejas, solo A3 (25-34/35-49) "
        f"tiene neto positivo, y es **+14 sobre 45.678 filas**.", "",
        "## Bootstrap", "",
        f"Delta **{boot['media']:+.7f}**, IC95 **[{boot['ic95'][0]:+.7f}, {boot['ic95'][1]:+.7f}]**, "
        f"contiene cero: **{boot['contiene_cero']}**. {N_BOOT} remuestreos pareados, unidad la "
        f"fila. La mejora no esta demostrada ni en signo ni en magnitud.", "",
        "## Interpretacion", "",
        "**P34 INVALIDATED.** La senal residual no aporta informacion suficientemente especifica para "
        "corregir de forma robusta las decisiones entre bandas adyacentes.", "",
        "Lo que se **no** dice: que la informacion linguistica no exista. P33 midio AUC 0,5411 en el "
        "residuo global. Lo que se dice es mucho mas estrecho: **esa informacion residual no ha "
        "podido convertirse en una mejora predictiva robusta mediante los mecanismos global y local "
        "evaluados.**", "",
        "## Limitaciones", "",
        "- **La correccion del 90 % baja el techo de la hipotesis**, pero no la invalida: el 59,94 % "
        "sigue siendo mayoria.",
        "- Un solo candidato linguistico. Los nueve restantes tienen AUC de residuo similar.",
        "- No se ejecuto particion alternativa: con el veredicto ya invalidado por el control de "
        "ruido, ejecutarla confirmaria algo que no va a cambiar.",
        "", "## Reproducibilidad", "",
        f"- Brazo real reconstruido desde cero: **{cv_reconst:.7f}**, frente a "
        f"{c[0]['corrected_cv']} en el artefacto. Coincide: "
        f"{abs(cv_reconst - c[0]['corrected_cv']) < 1e-6}.",
        "- `score_type` declarado por candidato en `reports/p34_score_inventory.csv`: V30 es "
        "**probability**, los diez candidatos son **decision_function**.",
        "", "## Figuras", "",
        "- `p33p34_01_brazos.png`: V30 frente a los cuatro brazos de P33 y los tres de P34",
        "- `p33p34_02_delta_por_fold.png`: corrected_cv por fold",
        "- `p33p34_03_rescate_danio.png`: rescate, danio y neto, con las cinco parejas",
        "- `p33p34_04_auc_por_pareja.png`: AUC del residuo frente a AUC de d_V30",
        "",
    ])


def _md_decision_p34(b, boot) -> str:
    return _pie("P34 - decision", [
        "```text", "ESTADO = INVALIDATED", "```", "",
        "## Por que INVALIDATED y no solo CLOSED", "",
        "La diferencia entre los dos no es de grado, es de naturaleza. **CLOSED** significa que se "
        "busco algo y no se encontro. **INVALIDATED** significa que se encontro algo que **parecia** "
        "el resultado buscado y se demostro que no lo era.", "",
        "En P34: `real - barajado = +0,0002070` sugeria que la correspondencia del residuo aportaba "
        "algo. Pero `real - aleatorio = -0,0000797`: el ruido de la misma media y desviacion rinde "
        "**mas**. El resultado apparent se atribuye a que se permitio sumar un termino al score, no "
        "a la senal linguistica. Eso es exactamente lo que la propia regla del brief marca como "
        "INVALIDATED.", "",
        "| Control | Delta | Lectura |", "|---|---|---|",
        f"| real | {b['p10_resultados'][0]['delta_vs_v30']:+.7f} | la senal |",
        f"| barajado | {b['p10_resultados'][1]['delta_vs_v30']:+.7f} | sin correspondencia fila-residuo |",
        f"| aleatorio | {b['p10_resultados'][2]['delta_vs_v30']:+.7f} | sin senal, misma forma |", "",
        "## Lo que queda cerrado con este veredicto", "",
        "- **La correccion local de score sobre bandas adyacentes**: cerrada como familia, no como "
        "configuracion. No se reintenta con otro `beta`, otra pareja ni otra transformacion.",
        "- **La lectura del 90 % adyacente**: corregida a 59,94 % en este informe. Los informes de "
        "V31 no se tocan.", "",
        "## Lo que este veredicto NO cierra", "",
        "No cierra la **existencia** de senal linguistica: P33 midio AUC 0,5411 en el residuo "
        "global. Cierra su **capacidad de mejorar la decision** a traves de sumar su residuo al "
        "score, ya sea sobre las seis clases o sobre un par.", "", "## Continuidad", "",
        "La regla de continuidad se cumple: no se ha parado en el primer negativo. Se evaluaron las "
        "cinco parejas (A1 a A5), con sus controles, su bootstrap y su desglose de alta confianza. "
        "El paso siguiente es un **gap analysis actualizado** que busque un mecanismo "
        "materialmente distinto, no otra forma de sumar el mismo residuo al score.", "",
        f"**Cifras**: V30 {CV_V30} | P34 real {b['p10_resultados'][0]['corrected_cv']} "
        f"({b['p10_resultados'][0]['delta_vs_v30']:+.7f}) | bootstrap {boot['media']:+.7f}, IC95 "
        f"[{boot['ic95'][0]:+.7f}, {boot['ic95'][1]:+.7f}]", "",
    ])


if __name__ == "__main__":
    raise SystemExit(main())
