#!/usr/bin/env python
"""
P98-14 (D98-1b): replica de robustez de la fusion contextual en una SEGUNDA
particion agrupada.

Autorizado por el propietario el 2026-10-13 con condiciones estrictas.
PRE-REGISTRO (fijado ANTES de evaluar; se escribe primero como artefacto):

  - Particion: StratifiedGroupKFold(5, shuffle=True, random_state=43) sobre
    las mismas 1483 filas de desarrollo; grupos = near-duplicate UNION
    pseudo-autor (identicos a los del protocolo canonico); estratificacion por
    la combinacion HS x TR x AG. Semilla de bootstrap: 20260925.
  - Brazos sobre EXACTAMENTE los mismos folds nuevos:
      A) V03-equivalente: TF-IDF word(1,2)+char_wb(3,5) min_df=2 sublinear
         + LR C=2.0 balanced liblinear por etiqueta, ajustado SOLO en el train
         de cada fold; umbrales V03 congelados [0.44, 0.51, 0.46].
      B) Fusion contextual: el MISMO clasificador de D98-1 (TF-IDF del fold
         + bloque denso congelado de reps C98-1 2048d; LR C=2 balanced);
         sin reajustar pooling, thresholds ni pesos.
  - Congelados: checkpoints LTR/RTL, reps (features/p98_13_reps_dev.npz),
    configuracion de fusion y umbrales. Sin test, holdouts ni leaderboard.
  - Criterios de decision: ROBUSTA si delta >= +0.003, >=4/5 pliegues
    positivos, IC95 sin cero y neto > 0; MARGINAL si positivo pero sin todos
    los criterios; FALSIFICADA si delta <= 0 o < 4/5 pliegues o neto <= 0.

Salidas: reports/p98_14_preregistro.json, reports/p98_14_replica_d98_1b.json
y .md, features/p98_14_oof_b.npz, reports/p98_downstream_results_b.csv,
reports/p98_complementarity_b.csv, figura reports/figuras/p98_14_replica.png.
No sobrescribe ningun artefacto de D98-1 ni el OOF canonico.

Uso:
    /home/carlos/venv/bin/python experiments/p98_14_replica_d98_1b.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import data as data_module  # noqa: E402
from src.embedding_utils import normalizar_emb_normal  # noqa: E402
from src.evaluation import metrica_macro, metricas_multietiqueta  # noqa: E402

SAL_PRE = RAIZ / "reports" / "p98_14_preregistro.json"
SAL_JSON = RAIZ / "reports" / "p98_14_replica_d98_1b.json"
SAL_MD = RAIZ / "reports" / "p98_14_replica_d98_1b.md"
SAL_NPZ = RAIZ / "features" / "p98_14_oof_b.npz"
SAL_RES = RAIZ / "reports" / "p98_downstream_results_b.csv"
SAL_COMP = RAIZ / "reports" / "p98_complementarity_b.csv"
SAL_FIG = RAIZ / "reports" / "figuras" / "p98_14_replica.png"

UMBRALES_V03 = np.array([0.44, 0.51, 0.46])
SEMILLA_PARTICION = 43
SEMILLA_BOOTSTRAP = 20260925
RUTA_REPS = RAIZ / "features" / "p98_13_reps_dev.npz"


def sha256(ruta: Path) -> str:
    """sha256 completo."""
    x = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            x.update(bloque)
    return x.hexdigest()


def lr_v03() -> LogisticRegression:
    """Especificacion exacta del clasificador de V03."""
    return LogisticRegression(class_weight="balanced", C=2.0, max_iter=1000,
                              solver="liblinear", random_state=42)


def correccion(pred_a, pred_b, y) -> dict:
    """Salvage/damage/neto por celda de pred_a frente a pred_b."""
    ok_a = pred_a == y
    ok_b = pred_b == y
    rescatadas = int((~ok_b & ok_a).sum())
    rotas = int((ok_b & ~ok_a).sum())
    return {"rescatadas": rescatadas, "rotas": rotas, "neto": rescatadas - rotas}


def bootstrap_clusters_delta(y, pred_a, pred_b, grupos, repeticiones=2000):
    """IC95 del delta pooled (a-b) por clusters (unidad = grupo)."""
    generador = np.random.default_rng(SEMILLA_BOOTSTRAP)
    indices = {int(g): np.flatnonzero(grupos == g) for g in np.unique(grupos)}
    claves = np.array(sorted(indices))
    dif = np.empty(repeticiones)
    for r in range(repeticiones):
        muestra = generador.integers(0, len(claves), len(claves))
        idx = np.concatenate([indices[int(claves[i])] for i in muestra])
        dif[r] = metrica_macro(y[idx], pred_a[idx]) - metrica_macro(y[idx], pred_b[idx])
    c = np.quantile(dif, [0.025, 0.975])
    return {"ic95": [round(float(c[0]), 6), round(float(c[1]), 6)],
            "pct_mejora": round(float((dif > 0).mean()), 6)}


def oof_arm(arm: str, textos: list[str], reps: np.ndarray, y: np.ndarray,
            pliegues: np.ndarray) -> np.ndarray:
    """OOF de un brazo ('v03' o 'fusion') sobre la particion dada."""
    proba = np.zeros((len(y), 3))
    for k in sorted(np.unique(pliegues).tolist()):
        idx_tr = np.flatnonzero(pliegues != k)
        idx_va = np.flatnonzero(pliegues == k)
        palabra = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                                  sublinear_tf=True)
        caracter = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                   min_df=2, sublinear_tf=True)
        textos_tr = [textos[i] for i in idx_tr]
        textos_va = [textos[i] for i in idx_va]
        bloques_tr = [palabra.fit_transform(textos_tr),
                      caracter.fit_transform(textos_tr)]
        bloques_va = [palabra.transform(textos_va), caracter.transform(textos_va)]
        if arm == "fusion":
            bloques_tr.append(sparse.csr_matrix(reps[idx_tr]))
            bloques_va.append(sparse.csr_matrix(reps[idx_va]))
        m_tr = sparse.hstack(bloques_tr).tocsr()
        m_va = sparse.hstack(bloques_va).tocsr()
        for etiqueta in range(3):
            modelo = lr_v03()
            modelo.fit(m_tr, y[idx_tr, etiqueta])  # SOLO train del fold
            proba[idx_va, etiqueta] = modelo.predict_proba(m_va)[:, 1]
    return proba


def main() -> None:
    """Pre-registra la particion y ejecuta la replica."""
    for ruta in (SAL_JSON, SAL_MD, SAL_NPZ, SAL_RES, SAL_COMP, SAL_FIG):
        if ruta.exists():
            raise SystemExit(f"Ya existe {ruta}; no se sobrescribe.")
    inicio = time.perf_counter()

    z = np.load(RAIZ / "features" / "oof_v03_final.npz", allow_pickle=True)
    zb = np.load(RAIZ / "features" / "v2_oof_base_corrected.npz", allow_pickle=True)
    ids = z["ids"].astype(np.int64)
    y = z["y_true"].astype(int)
    grupos = zb["grupos"].astype(np.int64)
    assert np.array_equal(ids, zb["ids"].astype(np.int64))
    train = data_module.read_train()
    mapa = train.assign(_id=train["id"].astype(int)).set_index("_id")
    textos = mapa.loc[ids, "text"].tolist()

    # --- FASE A: pre-registro de la particion (antes de evaluar) ---
    combinacion = np.array([f"{a}{b}{c}" for a, b, c in y])
    divisor = StratifiedGroupKFold(n_splits=5, shuffle=True,
                                   random_state=SEMILLA_PARTICION)
    pliegues = np.full(len(ids), -1, dtype=int)
    for numero, (_, val) in enumerate(divisor.split(
            np.arange(len(ids)).reshape(-1, 1), combinacion, groups=grupos)):
        pliegues[val] = numero + 1
    assert (pliegues >= 1).all()
    tamanos = [int((pliegues == k).sum()) for k in range(1, 6)]
    sin_fuga = True
    for k in range(1, 6):
        g_tr = set(grupos[pliegues != k].tolist())
        g_va = set(grupos[pliegues == k].tolist())
        sin_fuga &= len(g_tr & g_va) == 0
    sha_folds = hashlib.sha256(pliegues.astype("int64").tobytes()).hexdigest()
    if SAL_PRE.exists():
        previo = json.loads(SAL_PRE.read_text(encoding="utf-8"))
        assert previo["particion"]["sha256_pliegues"] == sha_folds, \
            "la particion pre-registrada no coincide"
        print("pre-registro existente verificado (hash de folds identico)",
              flush=True)
    else:
        preregistro = {
            "experiment_id": "P98-14",
            "fase": "D98-1b (replica de robustez, pre-registro)",
            "fecha": "2026-10-13",
            "particion": {"tipo": "StratifiedGroupKFold(5, shuffle, seed 43)",
                          "semilla": SEMILLA_PARTICION,
                          "grupos": "near-duplicate UNION pseudo-autor "
                                    "(mismos que el protocolo canonico)",
                          "estratificacion": "HS x TR x AG",
                          "n": int(len(ids)), "tamanos": tamanos,
                          "sha256_pliegues": sha_folds,
                          "sin_fuga_de_grupos": bool(sin_fuga)},
            "congelado": {
                "reps": {"ruta": str(RUTA_REPS.relative_to(RAIZ)),
                         "sha256": sha256(RUTA_REPS)},
                "umbrales_v03": UMBRALES_V03.tolist(),
                "fusion_LR": {"C": 2.0, "balanced": True, "solver": "liblinear"},
                "checkpoints_ltr_rtl": "models/p98_c98_1_ltr.pt / "
                                       "models/p98_c98_1_rtl.pt (no se recargan; "
                                       "las reps ya estan congeladas)",
            },
            "brazos": ["v03_equivalente (TF-IDF+LR C=2)", "fusion_contextual "
                       "(TF-IDF fold + denso reps C98-1)"],
            "criterios": {"robusta": "delta >= +0.003, >=4/5 pliegues, IC95 sin "
                                     "cero, neto > 0",
                          "marginal": "positivo sin todos los criterios",
                          "falsificada": "delta <= 0 o < 4/5 pliegues o neto <= 0"},
            "sin_leakage": "cada clasificador se ajusta SOLO con el train de su "
                           "fold; ninguna etiqueta de validacion participa",
        }
        SAL_PRE.write_text(json.dumps(preregistro, indent=2, ensure_ascii=True)
                           + "\n", encoding="utf-8")
        print(f"pre-registro escrito | tamanos {tamanos} | sin fuga {sin_fuga}",
              flush=True)
    preregistro = json.loads(SAL_PRE.read_text(encoding="utf-8"))

    # --- FASE B: evaluacion sobre los folds pre-registrados ---
    reps = np.load(RUTA_REPS, allow_pickle=True)["reps"]
    proba_v03 = oof_arm("v03", textos, reps, y, pliegues)
    proba_fus = oof_arm("fusion", textos, reps, y, pliegues)
    pred_v03 = (proba_v03 >= UMBRALES_V03).astype(int)
    pred_fus = (proba_fus >= UMBRALES_V03).astype(int)

    def resumen_regla(pred, nombre):
        met = metricas_multietiqueta(y, pred)
        por_pliegue = [float(metricas_multietiqueta(
            y[pliegues == k], pred[pliegues == k])["f1_media_columnas"])
            for k in range(1, 6)]
        return {"regla": nombre, "corrected_cv": met["f1_media_columnas"],
                "f1_por_etiqueta": met["f1_por_etiqueta"],
                "f1_pliegues": por_pliegue,
                "std_pliegues": float(np.std(por_pliegue, ddof=1)),
                "correccion_vs_v03": correccion(pred, pred_v03, y)}

    r_v03 = resumen_regla(pred_v03, "v03_equivalente_part47")
    r_fus = resumen_regla(pred_fus, "fusion_contextual_part47")
    delta = r_fus["corrected_cv"] - r_v03["corrected_cv"]
    pliegues_pos = int(np.sum(np.array(r_fus["f1_pliegues"])
                              > np.array(r_v03["f1_pliegues"]) + 1e-12))
    boot = bootstrap_clusters_delta(y, pred_fus, pred_v03, grupos)
    neto = r_fus["correccion_vs_v03"]["neto"]

    if delta >= 0.003 and pliegues_pos >= 4 and boot["ic95"][0] > 0 and neto > 0:
        veredicto = "ROBUSTA"
    elif delta > 0 and pliegues_pos >= 4 and neto > 0:
        veredicto = "MARGINAL"
    else:
        veredicto = "FALSIFICADA"

    resultado = {
        "experiment_id": "P98-14", "fase": "D98-1b (replica de robustez)",
        "fecha": "2026-10-13",
        "particion": preregistro["particion"],
        "reglas": {"v03_equivalente_part47": r_v03,
                   "fusion_contextual_part47": r_fus},
        "comparacion": {"delta_pooled": round(delta, 6),
                        "pliegues_positivos": pliegues_pos,
                        "bootstrap_clusters": boot,
                        "neto_celdas": neto,
                        "referencia_canonica_D98_1": {
                            "fusion": 0.7822662599,
                            "delta": 0.005055, "pliegues": 4,
                            "ic95": [-0.004058, 0.014721]}},
        "veredicto": veredicto,
        "tiempo_total_s": round(time.perf_counter() - inicio, 1),
    }
    np.savez_compressed(SAL_NPZ, ids=ids, y_true=y, grupos=grupos,
                        pliegues_b=pliegues, proba_v03=proba_v03,
                        proba_fus=proba_fus, pred_v03=pred_v03, pred_fus=pred_fus)
    SAL_JSON.write_text(json.dumps(resultado, indent=2, ensure_ascii=True) + "\n",
                        encoding="utf-8")
    filas = ["regla,corrected_cv,delta_pareado,f1_HS,f1_TR,f1_AG,std_pliegues,"
             "pliegues_positivos,neto"]
    for r, nombre, pl, nt in ((r_v03, "v03_equivalente_part47", "-", "-"),
                              (r_fus, "fusion_contextual_part47",
                               pliegues_pos, neto)):
        filas.append(f"{nombre},{r['corrected_cv']:.10f},"
                     f"{r['corrected_cv'] - r_v03['corrected_cv']:+.6f},"
                     f"{r['f1_por_etiqueta']['HS']:.6f},"
                     f"{r['f1_por_etiqueta']['TR']:.6f},"
                     f"{r['f1_por_etiqueta']['AG']:.6f},"
                     f"{r['std_pliegues']:.5f},{pl},{nt}")
    SAL_RES.write_text("\n".join(filas) + "\n", encoding="utf-8")
    lineas_c = ["regla,n,f1_v03eq,f1_fusion,delta,rescatadas,rotas,neto"]
    textos_norm = [normalizar_emb_normal(t) for t in textos]
    largo_tokens = np.array([len(t.split()) for t in textos_norm])
    slices = {"longitud_q4": largo_tokens >= np.quantile(largo_tokens, 0.75),
              "url": np.array([" url " in " " + t + " " for t in textos_norm]),
              "hashtag": np.array([" hashtag " in " " + t + " "
                                   for t in textos_norm])}
    for nombre, mascara in slices.items():
        corr = correccion(pred_fus[mascara], pred_v03[mascara], y[mascara])
        f1a = float(metricas_multietiqueta(y[mascara], pred_v03[mascara])
                    ["f1_media_columnas"])
        f1b = float(metricas_multietiqueta(y[mascara], pred_fus[mascara])
                    ["f1_media_columnas"])
        lineas_c.append(f"{nombre},{int(mascara.sum())},{f1a:.6f},{f1b:.6f},"
                        f"{f1b-f1a:+.6f},{corr['rescatadas']},{corr['rotas']},"
                        f"{corr['neto']}")
    SAL_COMP.write_text("\n".join(lineas_c) + "\n", encoding="utf-8")

    md = [
        "# P98-14 (D98-1b): replica de robustez en segunda particion agrupada",
        "",
        f"Pre-registro: `p98_14_preregistro.json` (semilla 43; tamanos "
        f"{tamanos}; sin fuga de grupos: {sin_fuga}). Todo congelado de D98-1: "
        "reps, configuracion de fusion y umbrales V03; sin blendings.",
        "",
        "| Regla | corrected_cv | delta | F1 HS/TR/AG |",
        "|---|---|---|---|",
        f"| V03-equivalente (part47) | {r_v03['corrected_cv']:.10f} | - | "
        f"{r_v03['f1_por_etiqueta']['HS']:.4f}/{r_v03['f1_por_etiqueta']['TR']:.4f}/"
        f"{r_v03['f1_por_etiqueta']['AG']:.4f} |",
        f"| Fusion contextual (part47) | {r_fus['corrected_cv']:.10f} | "
        f"{delta:+.6f} | {r_fus['f1_por_etiqueta']['HS']:.4f}/"
        f"{r_fus['f1_por_etiqueta']['TR']:.4f}/{r_fus['f1_por_etiqueta']['AG']:.4f} |",
        "",
        f"- Pliegues positivos: {pliegues_pos}/5; neto de celdas: {neto:+d} "
        f"({r_fus['correccion_vs_v03']['rescatadas']} rescatadas / "
        f"{r_fus['correccion_vs_v03']['rotas']} rotas); IC95 clusters "
        f"{boot['ic95']} (pct_mejora {boot['pct_mejora']}).",
        f"- Referencia canonica D98-1: fusion 0.782266, delta +0.005055, 4/5, "
        "IC95 [-0.004058, +0.014721].",
        "",
        f"**Veredicto: {veredicto}** (criterios pre-registrados: robusta = "
        "delta >= +0.003, >=4/5, IC sin cero, neto > 0; marginal = positivo "
        "sin todos; falsificada = delta <= 0 o <4/5 o neto <= 0).",
        "",
        "Sin test, sin holdouts, sin leaderboard; ningun artefacto de D98-1 "
        "fue modificado.",
    ]
    SAL_MD.write_text("\n".join(md) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4.2))
    xs = np.arange(5)
    ejes[0].bar(xs - 0.18, r_v03["f1_pliegues"], 0.36, label="V03-equiv (47)")
    ejes[0].bar(xs + 0.18, r_fus["f1_pliegues"], 0.36, label="fusion (47)")
    ejes[0].set_xticks(xs, [f"p{k}" for k in range(1, 6)])
    ejes[0].set_title("F1 por pliegue (particion 47)")
    ejes[0].legend(fontsize=8)
    ejes[1].bar(["canonica 42", "replica 47"],
                [0.005055, delta])
    ejes[1].axhline(0, color="black", lw=0.8)
    ejes[1].axhline(0.003, color="red", lw=0.8, ls="--")
    ejes[1].set_title("Delta fusion vs V03 (linea roja: +0.003)")
    fig.tight_layout()
    SAL_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAL_FIG, dpi=130)
    plt.close(fig)

    print(f"D98-1b: V03eq {r_v03['corrected_cv']:.6f} | fusion "
          f"{r_fus['corrected_cv']:.6f} | delta {delta:+.6f} | pliegues "
          f"{pliegues_pos}/5 | neto {neto:+d} | IC {boot['ic95']}", flush=True)
    print(f"VEREDICTO: {veredicto} | tiempo {resultado['tiempo_total_s']} s",
          flush=True)


if __name__ == "__main__":
    main()
