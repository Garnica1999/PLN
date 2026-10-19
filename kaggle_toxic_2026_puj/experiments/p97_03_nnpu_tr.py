#!/usr/bin/env python
"""
P97-03 (H97-C): nnPU/PNU para la cabeza TR (HS=0 como mezcla no supervisada).

Pre-registro: reports/p97_independent_research.md, seccion 5 (H97-C).
Mecanismo (Kiryo 2017; Sakai 2017), forma no negativa con P, N y U:

  R = pi_p * w_pos * meanP[softplus(-s)]
    + pi_n * w_neg * meanN[softplus(s)]
    + max(0, meanU[softplus(s)] - pi_p*meanP[softplus(s)] - pi_n*meanN[softplus(s)])

con P = TR=1, N = TR=0 & HS=1, U = HS=0; pi_p = tasa observada de TR en el
train del pliegue; penalizacion L2 (1/(2C))||w||^2 con C=2 (Adam wd=0.5).
HS y AG quedan estandar (BCE balanceada) en AMBOS brazos; solo TR cambia.
Control: torch BCE identico (mismo arnes, mismos datos, misma optimizacion).

Regla de decision propia (pre-registrada): umbrales por etiqueta elegidos en
un split agrupado INTERNO del train del pliegue (4 pliegues internos) y
aplicados al pliegue externo; secundario: umbrales V03 congelados.

Salidas: features/p97_03_oof_nnpu.npz, reports/p97_03_nnpu_tr.json/.md,
figura reports/figuras/p97_03_nnpu_tr.png.

Uso:
    /home/carlos/venv/bin/python experiments/p97_03_nnpu_tr.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import data as data_module  # noqa: E402
from src.evaluation import metrica_macro, metricas_multietiqueta  # noqa: E402

SAL_NPZ = RAIZ / "features" / "p97_03_oof_nnpu.npz"
SAL_JSON = RAIZ / "reports" / "p97_03_nnpu_tr.json"
SAL_MD = RAIZ / "reports" / "p97_03_nnpu_tr.md"
SAL_FIG = RAIZ / "reports" / "figuras" / "p97_03_nnpu_tr.png"

UMBRALES_V03 = np.array([0.44, 0.51, 0.46])
REJILLA = np.round(np.arange(0.20, 0.8001, 0.01), 2)
SEMILLA = 20260925
PASOS = 400
LR = 0.05
C_REG = 2.0
# Equivalencia liblinear exacta (correccion documentada): liblinear minimiza
#   0.5*||w||^2 + C * SUMA(l_i); dividiendo por (C*n) queda
#   (0.5/(C*n))*||w||^2 + media(l_i). Adam con weight_decay=wd anade la
#   penalizacion (wd/2)*||w||^2, luego wd = 1/(C*n). La primera corrida uso
#   wd=0.5 (2400x el valor correcto) y quedo como *_l2_escala_invalida.*


def construir_matriz(textos_train: pd.Series, textos_aplicar: pd.Series):
    """TF-IDF word(1,2) + char_wb(3,5) de V03, ajustado solo en el train."""
    palabra = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                              sublinear_tf=True)
    caracter = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                               sublinear_tf=True)
    m_pal = palabra.fit_transform(textos_train)
    m_car = caracter.fit_transform(textos_train)
    return (sparse.hstack([m_pal, m_car]).tocsr(),
            sparse.hstack([palabra.transform(textos_aplicar),
                           caracter.transform(textos_aplicar)]).tocsr())


def a_torch(matriz: sparse.csr_matrix) -> torch.Tensor:
    """Convierte CSR de scipy a tensor CSR de torch (float32)."""
    return torch.sparse_csr_tensor(
        torch.from_numpy(matriz.indptr.astype(np.int64)),
        torch.from_numpy(matriz.indices.astype(np.int64)),
        torch.from_numpy(matriz.data.astype(np.float32)),
        size=matriz.shape)


def entrenar_cabeza(matriz_t: torch.Tensor, y: np.ndarray, modo: str,
                    pi_p: float, pi_n: float, idx_u: np.ndarray | None = None,
                    semilla: int = 42) -> tuple:
    """Entrena una cabeza lineal (BCE balanceada o nnPU/PNU).

    Args:
        y: etiqueta binaria (P=1; para nnPU, 0 marca N (y U via idx_u)).
        idx_u: indices de las filas sin supervision (solo modo nnpu).
    """
    torch.manual_seed(semilla)
    n = y.shape[0]
    wd = 1.0 / (C_REG * n)
    n_pos = max(int((y == 1).sum()), 1)
    n_neg = max(int((y == 0).sum()), 1)
    w_pos = n / (2.0 * n_pos)
    w_neg = n / (2.0 * n_neg)
    y_t = torch.from_numpy(y).float()
    idx_p = np.flatnonzero(y == 1)
    if modo == "nnpu":
        mascara_u = np.zeros(n, dtype=bool)
        mascara_u[idx_u] = True
        idx_n = np.flatnonzero((y == 0) & ~mascara_u)
        if idx_n.size == 0 or idx_p.size == 0:
            raise RuntimeError("nnPU sin P o sin N: codificacion invalida")
    else:
        idx_n = np.flatnonzero(y == 0)
    dim = matriz_t.shape[1]
    w = torch.zeros(dim, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=LR, weight_decay=wd)
    pesos_fila = torch.where(y_t.bool(), torch.tensor(w_pos, dtype=torch.float32),
                             torch.tensor(w_neg, dtype=torch.float32))
    t_p = torch.from_numpy(idx_p)
    t_n = torch.from_numpy(idx_n)
    t_u = torch.from_numpy(idx_u) if modo == "nnpu" else None
    for _ in range(PASOS):
        s = torch.sparse.mm(matriz_t, w.unsqueeze(1)).squeeze(1) + b
        if modo == "bce":
            perdida = (pesos_fila * torch.nn.functional.softplus(
                -(2.0 * y_t - 1.0) * s)).mean()
        else:
            sp = torch.nn.functional.softplus(-s)
            sn = torch.nn.functional.softplus(s)
            rp = w_pos * sp[t_p].mean()
            rn = w_neg * sn[t_n].mean()
            corr = torch.clamp(sn[t_u].mean() - pi_p * sp[t_p].mean()
                               - pi_n * sn[t_n].mean(), min=0.0)
            perdida = pi_p * rp + pi_n * rn + corr
        opt.zero_grad()
        perdida.backward()
        opt.step()
    return w.detach(), b.detach()


def predecir(matriz_t: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    """Sigmoid de la puntuacion lineal."""
    with torch.no_grad():
        s = torch.sparse.mm(matriz_t, w.unsqueeze(1)).squeeze(1) + b
        return torch.sigmoid(s).numpy()


def entrenar_brazo(matriz_train, y_train, matriz_aplicar, modo: str,
                   semilla: int = 42) -> np.ndarray:
    """Entrena 3 cabezas; TR (indice 1) usa nnPU cuando modo == 'nnpu'."""
    probas = np.zeros((matriz_aplicar.shape[0], 3))
    mt = a_torch(matriz_train)
    ma = a_torch(matriz_aplicar)
    tasa = float(y_train[:, 1].mean())
    for posicion in range(3):
        if posicion == 1 and modo == "nnpu":
            hs = y_train[:, 0]
            tr = y_train[:, 1]
            # Codificacion P/N/U: P = TR=1; N = TR=0 & HS=1; U = HS=0.
            # (La primera corrida codifico N como 1 y dejo idx_n vacio; quedo
            # como *_pn_idx_invalido.*)
            y_efectiva = ((hs == 1) & (tr == 1)).astype(int)
            idx_u = np.flatnonzero(hs == 0)
            w, b = entrenar_cabeza(mt, y_efectiva, "nnpu", tasa, 1.0 - tasa,
                                   idx_u, semilla)
        else:
            w, b = entrenar_cabeza(mt, y_train[:, posicion], "bce", 0.0, 0.0,
                                   None, semilla)
        probas[:, posicion] = predecir(ma, w, b)
    return probas


def seleccionar_umbrales(y: np.ndarray, proba: np.ndarray,
                         filas: np.ndarray) -> list[float]:
    """Umbral por etiqueta que maximiza su F1 en las filas dadas."""
    umbrales = []
    for etiqueta in range(3):
        mejor = (0.5, -1.0)
        for t in REJILLA:
            f1 = f1_score(y[filas, etiqueta], (proba[filas, etiqueta] >= t).astype(int),
                          zero_division=0)
            if f1 > mejor[1]:
                mejor = (float(t), float(f1))
        umbrales.append(mejor[0])
    return umbrales


def bootstrap_clusters_delta(y, pred_a, pred_b, grupos, repeticiones=2000):
    """IC95 del delta pooled (a - b) por clusters."""
    generador = np.random.default_rng(SEMILLA)
    indices = {int(g): np.flatnonzero(grupos == g) for g in np.unique(grupos)}
    claves = np.array(sorted(indices))
    diferencias = np.empty(repeticiones)
    for r in range(repeticiones):
        muestra = generador.integers(0, len(claves), len(claves))
        idx = np.concatenate([indices[int(claves[i])] for i in muestra])
        diferencias[r] = metrica_macro(y[idx], pred_a[idx]) - metrica_macro(y[idx], pred_b[idx])
    c = np.quantile(diferencias, [0.025, 0.975])
    return {"ic95": [round(float(c[0]), 6), round(float(c[1]), 6)],
            "pct_mejora": round(float((diferencias > 0).mean()), 6)}


def main() -> None:
    """Ejecuta el pre-registro H97-C."""
    for ruta in (SAL_NPZ, SAL_JSON, SAL_MD, SAL_FIG):
        if ruta.exists():
            raise SystemExit(f"Ya existe {ruta}; no se sobrescribe.")
    inicio = time.perf_counter()
    z = np.load(RAIZ / "features" / "oof_v03_final.npz", allow_pickle=True)
    zb = np.load(RAIZ / "features" / "v2_oof_base_corrected.npz", allow_pickle=True)
    ids = z["ids"].astype(np.int64)
    y = z["y_true"].astype(int)
    pliegues = z["pliegues"].astype(int)
    grupos = zb["grupos"].astype(np.int64)
    assert np.array_equal(ids, zb["ids"].astype(np.int64))
    train = data_module.read_train()
    textos = (train.assign(_id=train["id"].astype(int))
              .set_index("_id").loc[ids, "text"].reset_index(drop=True))

    folds = sorted(np.unique(pliegues).tolist())
    proba_control = np.zeros((len(ids), 3))
    proba_nnpu = np.zeros((len(ids), 3))
    umbrales_propios = {"control": {}, "nnpu": {}}
    combinacion = np.array([f"{a}{b}{c}" for a, b, c in y])
    for k in folds:
        idx_outer_train = np.flatnonzero(pliegues != k)
        idx_outer_val = np.flatnonzero(pliegues == k)
        divisor = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
        inner = list(divisor.split(idx_outer_train.reshape(-1, 1),
                                   combinacion[idx_outer_train],
                                   groups=grupos[idx_outer_train]))
        for brazo in ("control", "nnpu"):
            scores_inner = np.zeros((len(idx_outer_train), 3))
            for sub_entrena, sub_val in inner:
                filas_entrena = idx_outer_train[sub_entrena]
                filas_val = idx_outer_train[sub_val]
                m_tr, m_val = construir_matriz(textos.iloc[filas_entrena],
                                               textos.iloc[filas_val])
                scores_inner[sub_val] = entrenar_brazo(m_tr, y[filas_entrena],
                                                       m_val, brazo)
            umbrales_propios[brazo][k] = seleccionar_umbrales(
                y[idx_outer_train], scores_inner, np.arange(len(idx_outer_train)))
            m_tr, m_val = construir_matriz(textos.iloc[idx_outer_train],
                                           textos.iloc[idx_outer_val])
            proba = entrenar_brazo(m_tr, y[idx_outer_train], m_val, brazo)
            if brazo == "control":
                proba_control[idx_outer_val] = proba
            else:
                proba_nnpu[idx_outer_val] = proba
        print(f"  pliegue {k}: brazos listos "
              f"({round(time.perf_counter() - inicio, 1)} s acumulados)")

    def pred_propia(proba, brazo):
        pred = np.zeros((len(ids), 3), dtype=int)
        for k in folds:
            filas = np.flatnonzero(pliegues == k)
            umbrales = umbrales_propios[brazo][k]
            for etiqueta in range(3):
                pred[filas, etiqueta] = (proba[filas, etiqueta]
                                         >= umbrales[etiqueta]).astype(int)
        return pred

    pred_control_propia = pred_propia(proba_control, "control")
    pred_nnpu_propia = pred_propia(proba_nnpu, "nnpu")
    pred_control_frozen = (proba_control >= UMBRALES_V03).astype(int)
    pred_nnpu_frozen = (proba_nnpu >= UMBRALES_V03).astype(int)

    def resumen_regla(pred, nombre):
        met = metricas_multietiqueta(y, pred)
        por_pliegue = [float(metricas_multietiqueta(
            y[pliegues == k], pred[pliegues == k])["f1_media_columnas"])
            for k in folds]
        return {"regla": nombre, "corrected_cv": met["f1_media_columnas"],
                "f1_por_etiqueta": met["f1_por_etiqueta"],
                "f1_pliegues": por_pliegue,
                "std_pliegues": float(np.std(por_pliegue, ddof=1))}

    r_propia_control = resumen_regla(pred_control_propia, "propios_control")
    r_propia_nnpu = resumen_regla(pred_nnpu_propia, "propios_nnpu")
    r_frozen_control = resumen_regla(pred_control_frozen, "frozen_control")
    r_frozen_nnpu = resumen_regla(pred_nnpu_frozen, "frozen_nnpu")

    delta_propia = (r_propia_nnpu["corrected_cv"] - r_propia_control["corrected_cv"])
    pliegues_mejorados = int(np.sum(
        np.array(r_propia_nnpu["f1_pliegues"])
        > np.array(r_propia_control["f1_pliegues"]) + 1e-12))
    delta_tr = (r_propia_nnpu["f1_por_etiqueta"]["TR"]
                - r_propia_control["f1_por_etiqueta"]["TR"])
    tr_por_pliegue_nnpu = [metricas_multietiqueta(
        y[pliegues == k], pred_nnpu_propia[pliegues == k])["f1_por_etiqueta"]["TR"]
        for k in folds]
    tr_por_pliegue_control = [metricas_multietiqueta(
        y[pliegues == k], pred_control_propia[pliegues == k])["f1_por_etiqueta"]["TR"]
        for k in folds]
    tr_mejorado = int(np.sum(np.array(tr_por_pliegue_nnpu)
                             > np.array(tr_por_pliegue_control) + 1e-12))
    boot = bootstrap_clusters_delta(y, pred_nnpu_propia, pred_control_propia, grupos)

    def correccion_vs(pred_a, pred_b):
        """Salvage/damage/neto por celda de pred_a frente a pred_b."""
        ok_a = pred_a == y
        ok_b = pred_b == y
        rescatadas = int((~ok_b & ok_a).sum())
        rotas = int((ok_b & ~ok_a).sum())
        return {"rescatadas": rescatadas, "rotas": rotas,
                "neto": rescatadas - rotas}

    correccion = {
        "vs_control": correccion_vs(pred_nnpu_propia, pred_control_propia),
        "vs_v03_frozen": correccion_vs(pred_nnpu_propia, z["pred"].astype(int)),
    }

    hs0 = y[:, 0] == 0
    tasa_tr_nnpu_hs0 = float(pred_nnpu_propia[hs0, 1].mean())
    tasa_tr_control_hs0 = float(pred_control_propia[hs0, 1].mean())
    tasa_v03_hs0 = float(((z["pred"].astype(int))[hs0, 1]).mean())

    if (delta_propia > 0 and (pliegues_mejorados >= 3 or boot["pct_mejora"] > 0.95)):
        veredicto = "SENAL"
    else:
        veredicto = "FALSIFICADA"

    resumen = {
        "experiment_id": "P97-03",
        "hipotesis": "H97-C",
        "fecha": "2026-10-13",
        "pre_registro": {"modo": "nnPU/PNU no negativo con P, N y U",
                         "P": "TR=1", "N": "TR=0 y HS=1", "U": "HS=0",
                         "pi_p": "tasa observada de TR en el train del pliegue",
                         "optimizacion": {"pasos": PASOS, "adam_lr": LR,
                                          "l2": "wd = 1/(C*n) con C=2 "
                                                "(equivalencia liblinear)"},
                         "nota_correcciones": "dos instanciaciones previas "
                         "invalidas preservadas (*_l2_escala_invalida, "
                         "*_pn_idx_invalido); esta es la ejecucion valida del "
                         "mecanismo tal como se pre-registro",
                         "umbrales_propios": "split interno agrupado de 4 pliegues"},
        "reglas": {"propia_control": r_propia_control, "propia_nnpu": r_propia_nnpu,
                   "frozen_control": r_frozen_control, "frozen_nnpu": r_frozen_nnpu},
        "umbrales_propios_por_pliegue": umbrales_propios,
        "comparacion": {"delta_pooled": delta_propia,
                        "pliegues_mejorados": pliegues_mejorados,
                        "delta_F1_TR": delta_tr,
                        "pliegues_TR_mejorados": tr_mejorado,
                        "tr_por_pliegue_nnpu": tr_por_pliegue_nnpu,
                        "tr_por_pliegue_control": tr_por_pliegue_control,
                        "correccion": correccion,
                        "bootstrap_clusters": boot},
        "zona_no_definida_HS0": {"tasa_TR1_v03": tasa_v03_hs0,
                                 "tasa_TR1_control": tasa_tr_control_hs0,
                                 "tasa_TR1_nnpu": tasa_tr_nnpu_hs0},
        "veredicto_pre_registrado": veredicto,
        "tiempo_total_s": round(time.perf_counter() - inicio, 1),
    }
    np.savez_compressed(SAL_NPZ, ids=ids, y_true=y, grupos=grupos, pliegues=pliegues,
                        proba_control=proba_control, proba_nnpu=proba_nnpu,
                        pred_propia_control=pred_control_propia,
                        pred_propia_nnpu=pred_nnpu_propia)
    SAL_JSON.write_text(json.dumps(resumen, indent=2, ensure_ascii=True) + "\n",
                        encoding="utf-8")

    lineas = [
        "# P97-03 (H97-C): nnPU/PNU para la cabeza TR",
        "",
        "Pre-registro en `p97_independent_research.md` seccion 5. Forma no "
        "negativa con P, N y U; HS y AG estandar en ambos brazos.",
        "",
        "| Regla | corrected_cv | F1 TR | std pliegues |",
        "|---|---|---|---|",
        f"| propios control | {r_propia_control['corrected_cv']:.10f} | "
        f"{r_propia_control['f1_por_etiqueta']['TR']:.6f} | "
        f"{r_propia_control['std_pliegues']:.5f} |",
        f"| propios nnpu | {r_propia_nnpu['corrected_cv']:.10f} | "
        f"{r_propia_nnpu['f1_por_etiqueta']['TR']:.6f} | "
        f"{r_propia_nnpu['std_pliegues']:.5f} |",
        f"| frozen control | {r_frozen_control['corrected_cv']:.10f} | "
        f"{r_frozen_control['f1_por_etiqueta']['TR']:.6f} | "
        f"{r_frozen_control['std_pliegues']:.5f} |",
        f"| frozen nnpu | {r_frozen_nnpu['corrected_cv']:.10f} | "
        f"{r_frozen_nnpu['f1_por_etiqueta']['TR']:.6f} | "
        f"{r_frozen_nnpu['std_pliegues']:.5f} |",
        "",
        f"Comparacion principal (nnpu - control, regla propia): delta "
        f"{delta_propia:+.6f}, pliegues {pliegues_mejorados}/5, delta F1 TR "
        f"{delta_tr:+.6f}, pliegues TR {tr_mejorado}/5, IC95 clusters "
        f"{boot['ic95']}.",
        "",
        f"Zona no definida (HS=0): tasa de TR=1 predicha - V03 "
        f"{tasa_v03_hs0:.4f} | control {tasa_tr_control_hs0:.4f} | nnpu "
        f"{tasa_tr_nnpu_hs0:.4f}.",
        "",
        f"**Veredicto pre-registrado: {veredicto}**",
    ]
    SAL_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4.2))
    xs = np.arange(5)
    ejes[0].bar(xs - 0.18, r_propia_control["f1_pliegues"], 0.36,
                label="control (BCE)")
    ejes[0].bar(xs + 0.18, r_propia_nnpu["f1_pliegues"], 0.36, label="nnPU/PNU")
    ejes[0].set_xticks(xs, [f"p{k}" for k in folds])
    ejes[0].set_title("F1 por pliegue (umbrales propios)")
    ejes[0].legend(fontsize=8)
    etiquetas = ["HS", "TR", "AG"]
    control_etq = [r_propia_control["f1_por_etiqueta"][e] for e in etiquetas]
    nnpu_etq = [r_propia_nnpu["f1_por_etiqueta"][e] for e in etiquetas]
    ejes[1].bar(np.arange(3) - 0.18, control_etq, 0.36, label="control")
    ejes[1].bar(np.arange(3) + 0.18, nnpu_etq, 0.36, label="nnpu")
    ejes[1].set_xticks(np.arange(3), etiquetas)
    ejes[1].set_title("F1 por etiqueta")
    ejes[1].legend(fontsize=8)
    fig.tight_layout()
    SAL_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAL_FIG, dpi=130)
    plt.close(fig)

    print("P97-03 H97-C: ejecutado")
    print(f"  propia control: {r_propia_control['corrected_cv']:.10f} | "
          f"propia nnpu: {r_propia_nnpu['corrected_cv']:.10f}")
    print(f"  delta={delta_propia:+.6f} | pliegues {pliegues_mejorados}/5 | "
          f"delta TR {delta_tr:+.6f} | IC {boot['ic95']}")
    print(f"  veredicto: {veredicto} | tiempo {resumen['tiempo_total_s']} s")


if __name__ == "__main__":
    main()
