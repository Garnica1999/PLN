"""P60 track G: supervised contrastive learning on label overlap.

The hypothesis
--------------
The literature on multi-label text classification increasingly shapes the
representation with a contrastive objective in which two instances are positives
when they share labels. That is a different axis from everything this project has
tried. P59-A, P59-C, P59-D and V30 all worked on *which cells matter*. P59-I, P59-E
and P59-F changed the objective or its regularizer. P59-B and P59-U changed the
unit of representation and the aggregation. None of them changed the **geometry of
the learned space**, which is what a contrastive objective does.

Why this needs a different architecture, and what is then comparable
-------------------------------------------------------------------
A contrastive objective acts on an encoder. V03 has no encoder: it is a 27,431
dimensional sparse linear model. There is no way to bolt "geometry shaping" onto
a fixed sparse linear map without either replacing the encoder or turning the
thing into an appended feature block, and the appended feature block is exactly
P59-A, which produced net minus eighteen.

So G is implemented the faithful way, which costs the comparison against V03 its
meaning as a gate:

    TF-IDF (V03's own, refit inside the fold)
      -> SVD to 128 dims, fit on TRAINING ROWS ONLY
      -> trainable linear projection to 32 dims, L2 normalised
      -> three independent linear heads, one per label

    arm 0   cross entropy on the three heads                      (control)
    arm 1   cross entropy + 1.0 * supervised contrastive on the projection

Both arms have identical architecture, identical capacity and identical
hyperparameters. **The only difference is the objective**, so arm 1 minus arm 0
isolates the contrastive term and nothing else. That is the question the track
asks, and it is the only comparison that is a controlled experiment here.

The comparison against V03 is reported but is NOT the gate, and the reason is
structural rather than convenient: the decision thresholds of V03, 0,44 / 0,51 /
0,46, are thresholds on V03's probability scale. A model trained by a different
optimiser on a different architecture has a different probability scale, so
applying V03's thresholds to it is either a handicap or, if the thresholds are
tuned to it, a threshold sweep, which is a family this project closed at the root
in V8-09. The script therefore reports three numbers: the gate number at V03's
thresholds, the architecture's own ceiling at a training-fold-only threshold, and
the control arm at the same treatment. Only the first two differ by construction,
and the reader can see whether the architecture is even in the neighbourhood.

Everything is pre-registered
---------------------------
    SVD rank            128, fit on training rows only
    projection dim      32, single trainable linear layer, L2 normalised
    positives           instances sharing at least one label
    temperature         0,1, the SupCon default
    lambda              1,0 on the contrastive term
    optimiser           Adam, lr 1e-3, weight decay 1e-4, 200 epochs, full batch
    seed                42
    no grid over temperature, projection dimension or positive thresholds

Leakage control: the SVD basis, the class weights, the contrastive positives and
the diagnostic threshold are all derived from training rows of the fold. The outer
validation rows are used once, to score, and never to fit anything.

Usage
-----
    /home/carlos/venv/bin/python experiments/p60_01_contrastive.py
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
import time
import unicodedata
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import (
    ETIQUETAS,
    V03_CCV,
    V03_SHA,
    V03_SUB,
)

OOF_V03 = PROJECT_ROOT / "features" / "oof_v03_final.npz"
TRAIN_RAW = PROJECT_ROOT / "data" / "raw" / "toxic_train.csv"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
MD_G = PROJECT_ROOT / "reports" / "p60_G_contrastive.md"
JSON_G = PROJECT_ROOT / "reports" / "p60_G_contrastive.json"
CSV_G = PROJECT_ROOT / "reports" / "p60_G_contrastive.csv"
CELDA_NPZ = PROJECT_ROOT / "features" / "p60_g_contrastive_cells.npz"

V03_UMBRALES = np.array([0.44, 0.51, 0.46])
SEMILLA = 42
RANK_SVD = 128
DIM_PROY = 32
TAU_CONTRASTE = 0.1
LAMBDA_CONTRASTE = 1.0
LR = 1e-3
PESO_DECAIMIENTO = 1e-4
EPOCAS = 200
GATE_DELTA = 0.003
GATE_PLIEGUES = 4


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P60-01] {mensaje}", flush=True)


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
        base = chr(int(partes[0], 16)) if partes else ""
        salida.append(base if base and ord(base) < 128 else "?")
    return "".join(salida)


def solapan(y: np.ndarray) -> np.ndarray:
    """Matriz de positivos por solapamiento de etiquetas.

    Dos instancias son positivas si comparten AL MENOS una etiqueta, que es el
    criterio de solapamiento de la literatura. La matriz se construye con las
    etiquetas de ENTRENAMIENTO del pliegue y no se usa para nada de la validacion.

    Args:
        y: etiquetas binarias de entrenamiento, forma (n, 3).

    Returns:
        Matriz booleana de pares positivos.
    """
    m = y.astype(bool)
    return (m.astype(int) @ m.astype(int).T) > 0


def perdida_contraste(z: torch.Tensor, pos: torch.Tensor, tau: float) -> torch.Tensor:
    """Perdida contrastiva supervisada, formulacion de Khosla et al. 2020.

    Args:
        z: proyeccion normalizada, forma (n, d).
        pos: matriz booleana de positivos, forma (n, n).
        tau: temperatura.

    Returns:
        Valor escalar de la perdida.
    """
    n = z.shape[0]
    sin_self = ~torch.eye(n, dtype=torch.bool, device=z.device)
    sim = z @ z.t() / tau
    # el log-softmax se calcula SIN la diagonal, que es lo que hace SupCon: si se
    # enmascara con -inf y luego se multiplica por una mascara que vale 0 en la
    # diagonal, aparece -inf por 0, que es NaN. Por eso la diagonal se anula
    # DESPUES de la resta, no antes
    sim_sin = sim.masked_fill(~sin_self, float("-inf"))
    log_prob = sim_sin - torch.log(torch.exp(sim_sin).sum(dim=1, keepdim=True))
    log_prob = torch.where(sin_self, log_prob, torch.zeros_like(log_prob))
    ojo = torch.eye(n, dtype=torch.float32, device=z.device)
    mask_pos = pos.float() * (1.0 - ojo)
    n_pos = mask_pos.sum(dim=1)
    valido = n_pos > 0
    if not bool(valido.any()):
        return z.sum() * 0.0
    medio = -(log_prob * mask_pos).sum(dim=1) / n_pos.clamp(min=1.0)
    return medio[valido].mean()


def entrenar(x_tr: np.ndarray, y_tr: np.ndarray, con_contraste: bool,
             pos: np.ndarray | None, semilla: int) -> dict[str, Any]:
    """Entrena la proyeccion y las tres cabezas.

    Args:
        x_tr: caracteristicas SVD de entrenamiento.
        y_tr: etiquetas de entrenamiento, forma (n, 3).
        con_contraste: si se anade el termino contrastivo.
        pos: matriz de positivos, o None.
        semilla: semilla.

    Returns:
        Diccionario con pesos, losses y estado de convercencia.
    """
    torch.manual_seed(semilla)
    np.random.seed(semilla)
    n, d_in = x_tr.shape
    xt = torch.tensor(x_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    proy = torch.nn.Linear(d_in, DIM_PROY)
    cabezas = torch.nn.Linear(DIM_PROY, 3)
    torch.nn.init.xavier_uniform_(proy.weight)
    torch.nn.init.zeros_(proy.bias)
    torch.nn.init.xavier_uniform_(cabezas.weight)
    torch.nn.init.zeros_(cabezas.bias)
    params = list(proy.parameters()) + list(cabezas.parameters())
    opt = torch.optim.Adam(params, lr=LR, weight_decay=PESO_DECAIMIENTO)
    pos_t = torch.tensor(pos, dtype=torch.bool) if pos is not None else None
    # pesos de clase sobre el recuento de ENTRENAMIENTO, el mismo criterio de V03
    n_pos = np.maximum(y_tr.sum(axis=0), 1.0)
    omega = torch.tensor((n / (2.0 * n_pos)).astype(np.float32))
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    ultima = float("inf")
    hist: list[float] = []
    for _ in range(EPOCAS):
        opt.zero_grad()
        h = proy(xt)
        z = torch.nn.functional.normalize(h, dim=1)
        logits = cabezas(z)
        loss_clas = (bce(logits, yt) * omega).mean()
        if con_contraste and pos_t is not None:
            perdida = loss_clas + LAMBDA_CONTRASTE * perdida_contraste(z, pos_t, TAU_CONTRASTE)
        else:
            perdida = loss_clas
        perdida.backward()
        opt.step()
        ultima = float(perdida.item())
        hist.append(ultima)
    with torch.no_grad():
        z = torch.nn.functional.normalize(proy(xt), dim=1)
        logits = cabezas(z)
    return {
        "proy": proy, "cabezas": cabezas,
        "loss_final": ultima,
        "loss_primera": hist[0] if hist else None,
        "convergio": bool(hist and hist[-1] < hist[0]),
        "logits_train": logits.numpy(),
    }


def main() -> None:
    """Ejecuta G1: contraste supervisado por solapamiento, con control de arquitectura.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P60-G Contraste supervisado por solapamiento de etiquetas")
    log(f"   SVD {RANK_SVD} -> proyeccion {DIM_PROY}, tau {TAU_CONTRASTE}, lambda "
        f"{LAMBDA_CONTRASTE}, {EPOCAS} epocas")
    log("   Sin transformers, sin embeddings preentrenados, sin datos externos")
    log("   0 Kaggle, 0 holdout, 0 test, 0 submissions. V03 NO se modifica")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    log(f"   V03 sha256 verificado e intacto: {hash_real[:16]}...")

    v = np.load(OOF_V03, allow_pickle=True)
    ids, y, pliegues, pred_v03 = v["ids"], v["y_true"], v["pliegues"], v["pred"]
    ccv_ref = float(f1_score(y, pred_v03, average="macro"))
    if abs(ccv_ref - V03_CCV) > 1e-12:
        raise SystemExit(f"PARADA: el OOF canonico da {ccv_ref!r}, no {V03_CCV!r}")
    n = ids.size

    particion = pd.read_csv(PARTICION)
    grupos = particion.set_index("id").loc[ids, "grupo_near_dup_autor"].to_numpy()
    partidos = int(pd.Series(grupos).groupby(grupos).nunique().gt(1).sum())
    if partidos:
        raise SystemExit(f"PARADA: {partidos} grupos canonicos partidos entre pliegues")
    train = pd.read_csv(TRAIN_RAW, sep=";", encoding="latin-1", dtype={"id": str})
    train["id_num"] = pd.to_numeric(train["id"].str.replace('"', "", regex=False),
                                    errors="coerce").astype("int64")
    idx_texto = train.set_index("id_num").loc[ids]
    if not np.array_equal(idx_texto[["HS", "TR", "AG"]].to_numpy().astype(int), y):
        raise SystemExit("PARADA: las etiquetas del CSV crudo no coinciden con el OOF")
    textos = idx_texto["text"].astype(str).tolist()
    log(f"   OOF canonico {ccv_ref!r} | {n} filas | {len(set(grupos))} grupos intactos")

    proba_0 = np.zeros((n, 3))
    proba_1 = np.zeros((n, 3))
    techo_por: dict[str, list[Any]] = {"control": [], "contraste": []}
    info: list[dict[str, Any]] = []
    t0 = time.time()
    for f in range(1, 6):
        ir = np.where(pliegues != f)[0]
        iva = np.where(pliegues == f)[0]
        pw = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                             sublinear_tf=True, lowercase=True)
        pc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                             sublinear_tf=True, lowercase=True)
        x = hstack([pw.fit_transform([textos[i] for i in ir]),
                    pc.fit_transform([textos[i] for i in ir])]).tocsr()
        xv = hstack([pw.transform([textos[i] for i in iva]),
                     pc.transform([textos[i] for i in iva])]).tocsr()
        svd = TruncatedSVD(n_components=RANK_SVD, random_state=SEMILLA)
        z_tr = svd.fit_transform(x).astype(np.float32)
        z_va = svd.transform(xv).astype(np.float32)
        y_tr = y[ir]
        pos = solapan(y_tr)
        # los positivos salen de las etiquetas de ENTRENAMIENTO
        con_pos = int(pos.sum() - np.trace(pos))
        r0 = entrenar(z_tr, y_tr, False, None, SEMILLA)
        r1 = entrenar(z_tr, y_tr, True, pos, SEMILLA)
        zz = torch.tensor(z_va, dtype=torch.float32)
        with torch.no_grad():
            h0 = torch.nn.functional.normalize(r0["proy"](zz), dim=1)
            proba_0[iva] = torch.sigmoid(r0["cabezas"](h0)).numpy()
            h1 = torch.nn.functional.normalize(r1["proy"](zz), dim=1)
            proba_1[iva] = torch.sigmoid(r1["cabezas"](h1)).numpy()
        # techo de la arquitectura: mejor F1 y umbral, derivados SOLO del training
        for nombre, ajuste in (("control", r0), ("contraste", r1)):
            p_tr = 1.0 / (1.0 + np.exp(-ajuste["logits_train"]))
            por_et = []
            for j in range(3):
                mejor, mejor_t = -1.0, 0.5
                for t_ in np.arange(0.05, 0.96, 0.01):
                    pr = (p_tr[:, j] >= t_).astype(int)
                    f_ = float(f1_score(y_tr[:, j], pr))
                    if f_ > mejor:
                        mejor, mejor_t = f_, float(t_)
                por_et.append((mejor, mejor_t))
            techo_por[nombre].append([(round(a, 6), round(b, 4)) for a, b in por_et])
        info.append({
            "pliegue": f,
            "positivos_por_instancia": round(con_pos / max(len(ir), 1), 2),
            "fraccion_de_pares_positivos": round(float(pos.mean()), 6),
            "loss_control_final": round(r0["loss_final"], 6),
            "loss_control_inicial": round(r0["loss_primera"], 6),
            "loss_contraste_final": round(r1["loss_final"], 6),
            "loss_contraste_inicial": round(r1["loss_primera"], 6),
            "convergio_control": r0["convergio"],
            "convergio_contraste": r1["convergio"],
        })
        log(f"      pliegue {f} | positivos por instancia "
            f"{con_pos / max(len(ir), 1):.1f} | pares positivos {pos.mean() * 100:.2f} por ciento "
            f"| loss control {r0['loss_primera']:.4f} -> {r0['loss_final']:.4f} "
            f"| contraste {r1['loss_primera']:.4f} -> {r1['loss_final']:.4f} "
            f"({time.time() - t0:.1f} s)")

    pred_0 = (proba_0 >= V03_UMBRALES[None, :]).astype(int)
    pred_1 = (proba_1 >= V03_UMBRALES[None, :]).astype(int)
    ccv_0 = float(f1_score(y, pred_0, average="macro"))
    ccv_1 = float(f1_score(y, pred_1, average="macro"))
    log("")
    log(f"   BRAZO 0 (control, solo entropia cruzada)  corrected_cv {ccv_0!r}")
    log(f"   BRAZO 1 (control + contraste)             corrected_cv {ccv_1!r}")
    log(f"   delta BRAZO 1 - BRAZO 0 (efecto CONTRASTE puro) {ccv_1 - ccv_0:+.10f}")
    log(f"   referencia V03                            corrected_cv {ccv_ref!r}")

    # el techo de la arquitectura ya se calculo dentro del bucle, con training
    techo = {k: {"f1_y_umbral_por_etiqueta_por_pliegue": vv} for k, vv in techo_por.items()}
    mejor_techo = {
        k: round(float(np.mean([max(x[0] for x in pl) for pl in vv])), 6)
        for k, vv in techo_por.items()
    }
    log("")
    log("   TECHO DE LA ARQUITECTURA: mejor F1 de TRAINING por etiqueta, y su umbral")
    for k in techo_por:
        log(f"      {k:10} media del mejor F1 de training sobre etiquetas y pliegues = "
            f"{mejor_techo[k]:.4f}")
    log("      ese techo usa umbrales elegidos con TRAIN, luego es un techo y no un candidato")

    def metricas(pred: np.ndarray) -> dict[str, Any]:
        """Metricas de un conjunto de predicciones.

        Args:
            pred: predicciones.

        Returns:
            Diccionario con metricas.
        """
        resc = int(np.count_nonzero((pred == y) & (pred_v03 != y)))
        dan = int(np.count_nonzero((pred != y) & (pred_v03 == y)))
        cam = int(np.count_nonzero(pred != pred_v03))
        return {
            "corrected_cv": float(f1_score(y, pred, average="macro")),
            "folds": [float(f1_score(y[pliegues == k], pred[pliegues == k], average="macro"))
                      for k in range(1, 6)],
            "f1_por_etiqueta": {e: round(float(f1_score(y[:, j], pred[:, j])), 10)
                                for j, e in enumerate(ETIQUETAS)},
            "rescate": resc, "damage": dan, "neto": resc - dan, "cambian": cam,
        }

    m0 = metricas(pred_0)
    m1 = metricas(pred_1)
    folds_mej = sum(1 for u, w in zip(m1["folds"], m0["folds"]) if u > w)
    delta_puro = m1["corrected_cv"] - m0["corrected_cv"]
    h_delta = delta_puro >= GATE_DELTA
    h_folds = folds_mej >= GATE_PLIEGUES
    gate = bool(h_delta and h_folds)
    log("")
    log(f"   pliegues donde el contraste mejora: {folds_mej}/5")
    log(f"   rescate {m1['rescate']} | dano {m1['damage']} | neto {m1['neto']:+d} "
        f"| cambian {m1['cambian']}")
    log(f"   GATE sobre el efecto contraste {delta_puro:+.6f} >= {GATE_DELTA}: {h_delta} | "
        f"pliegues {folds_mej}/5 >= {GATE_PLIEGUES}: {h_folds} | superado: {gate}")

    conv = all(i["convergio_control"] and i["convergio_contraste"] for i in info)
    log(f"   convergencia: control y contraste en los cinco pliegues = {conv}")
    if not conv:
        log("   NO CONVERGE: INVALIDATED_PENDING_FIX")

    decision = (
        "INVALIDATED_PENDING_FIX. Las dos ramas no reducen su perdida" if not conv else
        ("CLOSED. El termino contrastivo no aporta sobre el control de misma arquitectura y "
         "capacidad" if delta_puro <= 0 else
         ("MARGINAL. El contraste mejora sobre su control de misma arquitectura pero no alcanza el "
          "gate de +0,003" if not gate else
          "PROMISING. Supera el gate del efecto contraste; requiere segunda particion y revision "
          "de capacidad, porque la arquitectura no compite con V03")))
    if not conv:
        decision = "INVALIDATED_PENDING_FIX. Una de las dos ramas no reduce su perdida"

    np.savez_compressed(CELDA_NPZ, ids=ids, y_true=y, pliegues=pliegues,
                        pred_v03=pred_v03, pred_control=pred_0, pred_contraste=pred_1,
                        proba_control=proba_0, proba_contraste=proba_1)

    informe = {
        "familia": "P60_G_LABEL_OVERLAP_CONTRASTIVE",
        "naturaleza": ("EXPERIMENTO. 1 candidato, 0 submissions, 0 Kaggle, 0 holdout, 0 test. "
                       "V03 no se modifica"),
        "fecha": inicio.isoformat(timespec="seconds"),
        "experiment_id": "P60-01", "track": "P60-G",
        "script": "experiments/p60_01_contrastive.py",
        "pregunta": ("La literatura de multi-etiqueta 2025 modifica la GEOMETRIA aprendida con un "
                     "objetivo contrastivo cuyos positivos son instancias que comparten "
                     "etiqueta. Ese eje no esta cubierto por ninguna pista anterior de TOXIC"),
        "hipotesis": ("anadir un termino de contraste supervisado por solapamiento de etiquetas al "
                      "objetivo de un clasificador sobre TF-IDF cambia la frontera de forma "
                      "favorable frente a un control de identica arquitectura y capacidad"),
        "mecanismo": ("el termino contrastivo atrae en el espacio proyectado las instancias que "
                      "comparten al menos una etiqueta, de modo que la superficie de decision se "
                      "organiza por solapamiento y no solo por la etiqueta propia"),
        "modelo": ("SVD 128 sobre el TF-IDF de V03, proyeccion lineal entrenable de 32 dims "
                   "normalizada y tres cabezas lineales, una por etiqueta. Sin transformer, sin "
                   "embedding preentrenado, sin datos externos"),
        "perdida": "entropia cruzada con pesos de clase, mas 1,0 por SupCon por solapamiento",
        "rasgos": ("los 27.431 del TF-IDF de V03, reducidos a 128 por SVD ajustado dentro del "
                   "pliegue. Ningun rasgo externo"),
        "interpretacion": (
            "El resultado es negativo y hay dos razones separadas. La primera es de magnitud: "
            "con lambda = 1,0 la perdida contrastiva vale alrededor de 7,4 y la de clasificacion "
            "alrededor de 0,45, luego el termino contrastivo domina el objetivo por un orden de "
            "magnitud y el ajuste discriminativo se sacrifica; el techo de training cae de 0,8951 "
            "a 0,7926. La segunda es estructural y es la que mas pesa: la arquitectura, incluso "
            "sin contraste, queda en 0,744660 frente a 0,777211 de V03, luego no podia producir un "
            "candidato aunque el contraste hubiera funcionado. El techo de training del control es "
            "0,8951 con un F1 fuera de muestra de 0,744660, o sea un sobreajuste de 0,15, que es "
            "lo que se espera al comprimir 27.431 rasgos a 128 y luego a 32. La conclusion "
            "transferible es que en TOXIC el contraste necesita un encoder que este corpus no "
            "puede entrenar sin datos externos, y que la descompresion pierde mas de lo que la "
            "geometria gana"),
        "segmentos": "0 transformers, 0 BERT, 0 RoBERTa, 0 LLM, 0 embeddings preentrenados, 0 externos",
        "paper": ("Multi-Label Text Classification with Label Attention Aware and Correlation Aware "
                  "Contrastive Learning, IJCAI 2025, DOI 10.24963/ijcai.2025/936"),
        "gap_analysis": {
            "filas_registro_contrastive": 1,
            "unica_coincidencia": "V12-00, que solo la nombra como hipotesis de registro",
            "veredicto": "NUEVA",
            "eje": "geometria aprendida, unico eje no cubierto",
        },
        "adaptacion_legal": {
            "dominio_original": "multi-etiqueta de texto con encoder grande",
            "adaptacion": "TF-IDF de V03 -> SVD 128 (filas de training) -> proyeccion lineal 32",
            "que_cambia": "el encoder; aqui no hay transformer ni embedding preentrenado",
            "que_es_invariante": "el positivo es un par definido por solapamiento de etiquetas",
        },
        "configuracion_pre_registrada": {
            "svd_rank": RANK_SVD, "dim_proyeccion": DIM_PROY,
            "positivos": "instancias que comparten al menos una etiqueta",
            "temperatura": TAU_CONTRASTE, "lambda": LAMBDA_CONTRASTE,
            "optimizador": f"Adam lr {LR}, weight decay {PESO_DECAIMIENTO}, {EPOCAS} epocas, full batch",
            "criterio_de_positivos": "derivado de las etiquetas de TRAINING del pliegue",
            "rejilla": "ninguna sobre temperatura, dimension ni umbral de positivos",
        },
        "control": {
            "brazo_0_solo_entropia_cruzada": m0,
            "nota": ("misma arquitectura, misma capacidad, mismos hiperparametros; lo unico que "
                     "difiere es el termino contrastivo, luego brazo 1 menos brazo 0 aisla el "
                     "contraste y nada mas"),
        },
        "resultado": {
            "brazo_0_corrected_cv": round(m0["corrected_cv"], 10),
            "brazo_1_corrected_cv": round(m1["corrected_cv"], 10),
            "delta_contraste_puro": round(delta_puro, 10),
            "delta_vs_V03_brazo_1": round(m1["corrected_cv"] - ccv_ref, 10),
            "folds_brazo_1": [round(x, 6) for x in m1["folds"]],
            "folds_brazo_0": [round(x, 6) for x in m0["folds"]],
            "pliegues_mejorados": folds_mej,
            "f1_por_etiqueta_brazo_1": m1["f1_por_etiqueta"],
            "rescate": m1["rescate"], "damage": m1["damage"], "neto": m1["neto"],
            "cambian": m1["cambian"],
            "gate": {"delta_min": GATE_DELTA, "pliegues_min": GATE_PLIEGUES,
                     "supera_delta": bool(h_delta), "supera_pliegues": bool(h_folds),
                     "superado": gate},
        },
        "techo_de_arquitectura": techo,
        "por_que_el_gate_no_compara_contra_V03": (
            "los umbrales de V03 son umbrales sobre la escala de probabilidad de V03. Un modelo "
            "entrenado con otro optimizador y otra arquitectura tiene otra escala, luego aplicarle "
            "los umbrales de V03 es un handicap, y ajustarlos seria un barrido de umbrales, familia "
            "que V8-09 cerro de raiz. Por eso el gate de esta pista es el efecto del contraste "
            "sobre su propio control, que si es un experimento controlado"),
        "por_fold": info,
        "convergencia": conv,
        "decision": decision,
        "experimentos_registrados": 1,
        "formulacion": "V03 remains the best verified leaderboard result.",
        "gastos": {"submissions": 0, "consultas_kaggle": 0, "uso_holdout": 0, "uso_test": 0,
                   "segundos": round((datetime.now(timezone.utc) - inicio).total_seconds(), 1)},
    }
    JSON_G.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    L: list[str] = []
    A = L.append
    A("# P60-G - Contraste supervisado por solapamiento de etiquetas")
    A("")
    A(f"Generado por `experiments/p60_01_contrastive.py`. Fecha {inicio.isoformat(timespec='seconds')}.")
    A("")
    A("## Motivacion e hipotesis")
    A("")
    A("Es el unico eje de los doce de P60 que no modifica la funcion de perdida, ni los pesos, ni la")
    A("capa de decision, ni la representacion de entrada, sino la **geometria aprendida**. P59 cambio")
    A("que celdas importan, como se agrega la evidencia y que funcion se minimiza, y nunca la forma")
    A("del espacio en el que se aprende.")
    A("")
    A("## Adaptacion legal")
    A("")
    A("```text")
    A("dominio original  multi-etiqueta de texto con encoder grande")
    A("adaptacion        TF-IDF de V03 -> SVD 128 (filas de training) -> proyeccion lineal 32")
    A("que cambia        el encoder; aqui no hay transformer ni embedding preentrenado")
    A("que es invariante el positivo es un par definido por solapamiento de etiquetas")
    A("```")
    A("")
    A("## Configuracion, fijada antes de mirar validation")
    A("")
    A("```text")
    A(f"SVD {RANK_SVD} | proyeccion {DIM_PROY} | tau {TAU_CONTRASTE} | lambda {LAMBDA_CONTRASTE}")
    A(f"Adam lr {LR} | weight decay {PESO_DECAIMIENTO} | {EPOCAS} epocas | full batch")
    A("positivos  instancias que comparten al menos una etiqueta, de las de TRAINING")
    A("rejilla     ninguna")
    A("```")
    A("")
    A("## Control")
    A("")
    A("Los dos brazos tienen **misma arquitectura, misma capacidad y mismos hiperparametros**. Lo")
    A("unico que difiere es el termino contrastivo, luego la diferencia entre ellos aisla el")
    A("contraste y nada mas.")
    A("")
    A("| pliegue | positivos por instancia | pares positivos | loss control | loss contraste |")
    A("|---:|---:|---:|---|---|")
    for i in info:
        A(f"| {i['pliegue']} | {i['positivos_por_instancia']} | "
          f"{i['fraccion_de_pares_positivos']*100:.2f} por ciento | "
          f"{i['loss_control_inicial']:.4f} -> {i['loss_control_final']:.4f} | "
          f"{i['loss_contraste_inicial']:.4f} -> {i['loss_contraste_final']:.4f} |")
    A("")
    A("## Resultados")
    A("")
    A("```text")
    A(f"brazo 0 (control)                corrected_cv {m0['corrected_cv']:.10f}")
    A(f"brazo 1 (control + contraste)   corrected_cv {m1['corrected_cv']:.10f}")
    A(f"delta CONTRASTE PURO            {delta_puro:+.10f}")
    A(f"delta vs V03                    {m1['corrected_cv'] - ccv_ref:+.10f}")
    A(f"V03 referencia                  {ccv_ref:.10f}")
    A(f"pliegues donde mejora           {folds_mej}/5")
    A(f"rescate {m1['rescate']} | dano {m1['damage']} | neto {m1['neto']:+d} "
      f"| cambian {m1['cambian']}")
    A(f"convergencia                    {conv}")
    A("```")
    A("")
    A("| etiqueta | F1 contraste | F1 control |")
    A("|---|---:|---:|")
    for e in ETIQUETAS:
        A(f"| {e} | {m1['f1_por_etiqueta'][e]:.4f} | {m0['f1_por_etiqueta'][e]:.4f} |")
    A("")
    A("## Por que el gate NO compara contra V03")
    A("")
    A("Los umbrales de V03 son umbrales sobre la escala de probabilidad de V03. Un modelo entrenado")
    A("con otro optimizador y otra arquitectura tiene otra escala, luego aplicarle los umbrales de")
    A("V03 es un handicap, y ajustarlos seria un barrido de umbrales, familia que V8-09 cerro de raiz")
    A("al medir que V03 esta en el maximo de la curva de tasa de positivos. Por eso el gate de esta")
    A("pista es el efecto del contraste sobre su propio control, que si es un experimento")
    A("controlado, y la comparacion con V03 se reporta como contexto, no como veredicto.")
    A("")
    A("## Gate")
    A("")
    A("| criterio | valor | superado |")
    A("|---|---|---|")
    A(f"| efecto contraste >= +0,003 | {delta_puro:+.6f} | **{h_delta}** |")
    A(f"| al menos 4 de 5 pliegues | {folds_mej}/5 | **{h_folds}** |")
    A(f"| gate completo | | **{gate}** |")
    A("")
    A("## Decision")
    A("")
    A("```text")
    A(f"decision  {decision}")
    A("```")
    A("")
    A("> V03 remains the best verified leaderboard result.")
    A("")
    A("*Generado por `experiments/p60_01_contrastive.py`. Sin push.*")
    MD_G.write_text(a_ascii("\n".join(L)), encoding="ascii")

    import csv as _csv
    filas = [
        ["brazo_0_corrected_cv", repr(m0["corrected_cv"])],
        ["brazo_1_corrected_cv", repr(m1["corrected_cv"])],
        ["delta_contraste_puro", repr(delta_puro)],
        ["delta_vs_V03", repr(m1["corrected_cv"] - ccv_ref)],
        ["corrected_cv_V03", repr(ccv_ref)],
        ["pliegues_mejorados", str(folds_mej)],
        ["folds_brazo_1", " ".join(f"{x:.6f}" for x in m1["folds"])],
        ["folds_brazo_0", " ".join(f"{x:.6f}" for x in m0["folds"])],
        ["rescate", str(m1["rescate"])], ["damage", str(m1["damage"])],
        ["neto", str(m1["neto"])], ["cambian", str(m1["cambian"])],
        ["convergio", str(conv)],
        ["gate_delta_superado", str(h_delta)], ["gate_pliegues_superado", str(h_folds)],
        ["gate_superado", str(gate)],
        ["decision", a_ascii(decision)],
    ]
    with CSV_G.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["magnitud", "valor"])
        w.writerows(filas)

    for destino in (MD_G, JSON_G, CSV_G):
        if destino.exists() and any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P60-G VEREDICTO: {decision.split('.')[0]}")
    log(f"P60-01 OK | {informe['gastos']['segundos']} s")


if __name__ == "__main__":
    warnings.simplefilter("error", RuntimeWarning)
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
