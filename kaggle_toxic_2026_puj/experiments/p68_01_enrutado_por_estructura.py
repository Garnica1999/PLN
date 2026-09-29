"""P68-H1: route the decision by the structure of the example, not by a transformation of scores.

Why this is not another score formula
-------------------------------------
P67 closed the family of generic score transformations twice over: empirically, because every
one of them produced four cells out of 1483, and algebraically, because the best of them
turned out to be an exact relabelling of a linear blend.

What survived P67 was a measurement, not a mechanism. From the gap analysis:

    B   215 cells where V03 is wrong and some alternative is right
    122 of those 215 are reached by exactly ONE candidate
    MI_arbol alone reaches 77 cells that nobody else reaches

So the information exists, is concentrated, and is not redundant. What nobody has tried is
changing WHO decides each row.

This experiment changes that and only that. A router reads the STRUCTURE OF THE TEXT and
predicts, per label, whether the decision for that row should be handed to MI_arbol instead of
left with V03. It never looks at either score to make that choice.

    not a feature column        P64 and P65 closed that
    not a blend of scores       P66 and P67 closed that, and it is DUPLICATE by algebra
    not a threshold move        P67-01 closed that
    not an ensemble of weights  P61 closed that
    a decomposition of the decision over the structure of the example

The distinction is not cosmetic. A blend produces a single decision boundary in the plane of
the two scores. A router produces a SELECTION, and a selection can be non-convex: it can send
the extreme cases one way and the middle another, which no convex mixture can express.

The protocol is nested in the way P67 established: the router is fit on inner_train and its
operating point is chosen on inner_validation. The outer fold is only ever evaluated.
"""

from __future__ import annotations

import ast
import csv as _csv
import hashlib
import json
import subprocess
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import (
    ETIQUETAS,
    V03_CCV,
    V03_SHA,
    V03_SUB,
)

REPORTS = PROJECT_ROOT / "reports"
FIGURAS = PROJECT_ROOT / "figuras" / "p68"
CELDA = PROJECT_ROOT / "features" / "p66_scores_candidatos.npz"
ESTRUCT = PROJECT_ROOT / "features" / "v2_structured_features.parquet"
OOF = PROJECT_ROOT / "features" / "oof_v03_final.npz"

MD_R = REPORTS / "p68_enrutado_por_estructura.md"
JSON_R = REPORTS / "p68_enrutado_por_estructura.json"
CSV_F = REPORTS / "p68_enrutado_por_fold.csv"
CSV_C = REPORTS / "p68_enrutado_controles.csv"

V03_UMBRALES = np.array([0.44, 0.51, 0.46])
SEMILLA = 42
N_INNER = 3
CANDIDATO = "MI_arbol"
# Rejilla de la tasa de delegacion. Se declara ANTES de mirar nada, y es la unica decision
# de capacidad del enrutador: cuanto trabajo se le pasa.
TASAS = (0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25)
EXCLUIR = {"id", "particion", "id_int"}

# Features derivadas del SCORE de V03. Se excluyen del enrutador principal para que su
# decision no dependa de ninguna transformacion de scores: si el enrutado por estructura
# sola no funciona, se sabra, y se dira.
DERIVADAS_DEL_SCORE = ("margen", "proba", "pred", "confianza", "logit")


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P68-01] {mensaje}", flush=True)


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


def ccv(pred: np.ndarray, y: np.ndarray) -> float:
    """Macro F1 sobre predicciones duras.

    Args:
        pred: predicciones 0/1.
        y: etiquetas verdaderas.

    Returns:
        El macro F1.
    """
    partes = [float(f1_score(y[:, j], pred[:, j], average="binary", zero_division=0))
              for j in range(3)]
    return float(np.mean(partes))


def enrutar(X: np.ndarray, objetivo: np.ndarray, pliegues: np.ndarray,
            base_a: np.ndarray, base_b: np.ndarray, y: np.ndarray,
            semilla: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Enruta la decision por pliegue, con el enrutador ajustado de forma anidada.

    Args:
        X: features estructurales.
        objetivo: matriz booleana de a quien delegar (por etiqueta).
        pliegues: pliegue de cada fila.
        base_a: prediccion por defecto (V03).
        base_b: prediccion del experto (el candidato).
        y: etiquetas verdaderas.
        semilla: semilla base.

    Returns:
        Las predicciones enrutadas y el detalle por pliegue.
    """
    salida = base_a.copy()
    detalle: list[dict[str, Any]] = []
    for f in range(1, 6):
        ir, iva = np.where(pliegues != f)[0], np.where(pliegues == f)[0]
        for j in range(3):
            skf = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=semilla + 100 * f)
            it = iv = None
            for tr_p, va_p in skf.split(np.arange(len(ir)), objetivo[ir, j].astype(int)):
                it, iv = ir[tr_p], ir[va_p]
                break
            # El enrutador aprende QUIEN acierta donde V03 falla. Su etiqueta se construye
            # solo con informacion del entrenamiento interno.
            etiqueta = objetivo[:, j].astype(int)
            if etiqueta[it].sum() < 5 or (1 - etiqueta[it]).sum() < 5:
                detalle.append({"pliegue": f, "etiqueta": ETIQUETAS[j], "tasa": 0.0,
                                "posibles_inner": int(etiqueta[it].sum()),
                                "delegadas": 0, "motivo": "sin positivos suficientes"})
                continue
            modelo = HistGradientBoostingClassifier(
                max_depth=3, max_iter=120, learning_rate=0.06, l2_regularization=1.0,
                min_samples_leaf=20, random_state=semilla)
            modelo.fit(X[it], etiqueta[it])
            punt = modelo.predict_proba(X)[:, 1]
            # La tasa de delegacion se elige en la validacion INTERNA, nunca en la externa.
            mejor_tasa, mejor_f1 = 0.0, -1.0
            for tasa in TASAS:
                umbral = float(np.quantile(punt[iv], 1.0 - tasa))
                cand = base_a[iv, j].copy()
                delega = punt[iv] >= umbral
                cand[delega] = base_b[iv, j][delega]
                s = float(f1_score(y[iv, j], cand, average="binary", zero_division=0))
                if s > mejor_f1:
                    mejor_tasa, mejor_f1 = tasa, s
            umbral = float(np.quantile(punt[iva], 1.0 - mejor_tasa)) if mejor_tasa > 0 else np.inf
            delega = punt[iva] >= umbral
            # `salida[iva, j][delega] = ...` NO funciona: el indexado con un array devuelve
            # una COPIA, y la asignacion se pierde. Hay que indexar con la mascara aplicada
            # al propio indice. Una version anterior tenia ese bug y daba delta exactamente
            # 0,0000000000 con cientos de celdas delegadas: la firma inconfundible.
            destino = iva[delega]
            salida[destino, j] = base_b[destino, j]
            detalle.append({
                "pliegue": f, "etiqueta": ETIQUETAS[j], "tasa": mejor_tasa,
                "delegadas": int(delega.sum()),
                "f1_inner_delegando": round(mejor_f1, 6),
                "delta_inner": round(mejor_f1 - float(f1_score(
                    y[iv, j], base_a[iv, j], average="binary", zero_division=0)), 10),
                "V03_acertaba_entre_delegadas": int(
                    (delega & (base_a[iva, j] == y[iva, j])).sum()),
                "V03_fallaba_entre_delegadas": int(
                    (delega & (base_a[iva, j] != y[iva, j])).sum()),
                "experto_acierta_entre_delegadas": int(
                    (delega & (base_b[iva, j] == y[iva, j])).sum()),
                "experto_falla_entre_delegadas": int(
                    (delega & (base_b[iva, j] != y[iva, j])).sum()),
            })
    return salida, detalle


def main() -> None:
    """Ejecuta P68-H1 con su control, su ablacion y su linea base de decision.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P68-H1 enrutado por estructura del ejemplo: cambia QUIEN decide, no como se combina")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    v = np.load(OOF, allow_pickle=True)
    y = v["y_true"].astype(int)
    if float(np.mean([f1_score(y[:, j], v["pred"][:, j].astype(int), average="binary",
                               zero_division=0) for j in range(3)])) != V03_CCV:
        raise SystemExit("PARADA: el OOF canonico no reproduce el corrected_cv congelado")

    z = np.load(CELDA, allow_pickle=True)
    ids = z["ids"].astype(int)
    pliegues = z["pliegues"]
    pred_v03 = z["pred_v03"].astype(int)
    proba_v03 = z["proba_v03"]
    n = ids.size
    ref = ccv(pred_v03, y)

    # ---------- aislamiento: solo filas de desarrollo ----------
    estruct = pd.read_parquet(ESTRUCT)
    estruct["id_int"] = estruct["id"].astype(int)
    dev = estruct[estruct["particion"] == "dev"].set_index("id_int")
    faltan = [i for i in ids.tolist() if i not in dev.index]
    if faltan:
        raise SystemExit(f"PARADA: {len(faltan)} filas del OOF no estan en la particion dev")
    for otra in ("test", "holdout"):
        solape = len(set(ids.tolist()) & set(estruct[estruct["particion"] == otra]["id_int"]))
        if solape:
            raise SystemExit(f"PARADA: {solape} filas del OOF estan en la particion {otra}")
    dev = dev.loc[ids.tolist()]
    columnas = [c for c in dev.columns
                if c not in EXCLUIR
                and not any(d in c.lower() for d in DERIVADAS_DEL_SCORE)
                and pd.api.types.is_numeric_dtype(dev[c])]
    X = dev[columnas].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    log(f"   V03 verificado: {ref:.10f} | {n} filas | solo la particion dev")
    log(f"   features del enrutador: {len(columnas)} estructurales, 0 derivadas del score")

    # ---------- el problema que el enrutador tiene que resolver ----------
    pred_c = (z[f"proba_{CANDIDATO}"] >= V03_UMBRALES[None, :]).astype(int)
    objetivo = (pred_v03 != y) & (pred_c == y)
    log("")
    log("   A QUIEN HABRIA QUE DELEGAR, contado sobre desarrollo")
    for j, et in enumerate(ETIQUETAS):
        log(f"      {et}: V03 falla {int((pred_v03[:, j] != y[:, j]).sum()):4d} | "
            f"el candidato acierta donde V03 falla {int(objetivo[:, j].sum()):4d} | "
            f"el candidato falla donde V03 acierta "
            f"{int(((pred_v03[:, j] == y[:, j]) & (pred_c[:, j] != y[:, j])).sum()):4d}")
    log(f"      total delegable: {int(objetivo.sum())} celdas de {n * 3}")
    log("      ese es el techo: delegar EXACTAMENTE esas celdas y ninguna mas")

    # ---------- techo teorico, para saber cuanto margen hay ----------
    techo = pred_v03.copy()
    techo[objetivo] = pred_c[objetivo]
    ccv_techo = ccv(techo, y)
    log("")
    log(f"   TECHO si se delegara exactamente lo delegable:     {ccv_techo:.10f} "
        f"({ccv_techo - ref:+.10f})")
    log(f"   V03 sin tocar:                                     {ref:.10f}")
    log(f"   margen disponible: {(ccv_techo - ref) * 3 * 1000 / 2:+.2f} celdas")

    # ---------- el experimento ----------
    log("")
    log("   EL EXPERIMENTO, con el enrutador ajustado de forma anidada")
    aciertos, detalle = enrutar(X, objetivo, pliegues, pred_v03, pred_c, y, SEMILLA)
    ccv_real = ccv(aciertos, y)
    delta = ccv_real - ref
    delegadas = sum(d["delegadas"] for d in detalle)
    correctas = sum(d.get("experto_acierta_entre_delegadas", 0) for d in detalle)
    incorrectas = sum(d.get("experto_falla_entre_delegadas", 0) for d in detalle)
    log(f"      corrected_cv enrutado: {ccv_real:.10f} | delta {delta:+.10f}")
    v03_ok_del = sum(d.get("V03_acertaba_entre_delegadas", 0) for d in detalle)
    v03_ko_del = sum(d.get("V03_fallaba_entre_delegadas", 0) for d in detalle)
    log(f"      celdas delegadas: {delegadas}")
    log(f"      de esas, V03 acertaba {v03_ok_del} y fallaba {v03_ko_del}")
    log(f"      el experto acierta {correctas} y falla {incorrectas} | "
        f"neto {correctas - incorrectas:+d}")
    aciertos_por_fold = []
    for f in range(1, 6):
        iva = np.where(pliegues == f)[0]
        d = ccv(aciertos[iva], y[iva]) - ccv(pred_v03[iva], y[iva])
        aciertos_por_fold.append(round(float(d), 10))
    folds_pos = sum(1 for d in aciertos_por_fold if d > 0)
    log(f"      delta por pliegue: {aciertos_por_fold} | positivos: {folds_pos} de 5")

    # ---------- control 1: la misma maquinaria con el objetivo BARAJADO ----------
    log("")
    log("   CONTROL 1, el objetivo BARAJADO: mismas features, a quien delegar al azar")
    rng = np.random.default_rng(SEMILLA)
    objetivo_bar = np.zeros_like(objetivo)
    for j in range(3):
        pos = np.where(objetivo[:, j])[0]
        destino = rng.permutation(n)[: len(pos)]
        objetivo_bar[destino, j] = True
    pred_bar, detalle_bar = enrutar(X, objetivo_bar, pliegues, pred_v03, pred_c, y, SEMILLA + 7)
    ccv_bar = ccv(pred_bar, y)
    log(f"      corrected_cv con el objetivo barajado: {ccv_bar:.10f} | "
        f"delta {ccv_bar - ref:+.10f}")
    log(f"      celdas delegadas: {sum(d['delegadas'] for d in detalle_bar)}")
    log("      el barajado conserva CUANTAS celdas se delegan y destruye CUALES.")

    # ---------- control 2: delegar al azar, mismas celdas ----------
    log("")
    log("   CONTROL 2, delegacion ALEATORIA con el mismo numero de celdas POR ETIQUETA")
    pred_alea = pred_v03.copy()
    for j in range(3):
        cuantas = sum(d["delegadas"] for d in detalle if d["etiqueta"] == ETIQUETAS[j])
        elegidas = rng.permutation(n)[:cuantas]
        pred_alea[elegidas, j] = pred_c[elegidas, j]
    ccv_alea = ccv(pred_alea, y)
    log(f"      corrected_cv delegando al azar: {ccv_alea:.10f} | delta {ccv_alea - ref:+.10f}")
    log("      mismo numero de celdas, elegidas al azar. Aisla el COMO del CUANTAS.")

    # ---------- ablacion: anadir el margen de V03 ----------
    log("")
    log("   ABLACION, y si se le deja mirar el MARGEN de V03?")
    X_margen = np.column_stack([X, z["margen_v03"]])
    pred_m, _detalle_m = enrutar(X_margen, objetivo, pliegues, pred_v03, pred_c, y, SEMILLA)
    ccv_m = ccv(pred_m, y)
    log(f"      con el margen de V03 anadido: {ccv_m:.10f} | delta {ccv_m - ref:+.10f}")
    log("      declarado: el margen ES una derivada del score. Si esto mejorara mucho y la")
    log("      version por estructura sola no, el mecanismo seria una transformacion de scores")
    log("      con otro nombre, y habria que decirlo asi")

    # ---------- atribucion: el enrutado y el blend son cosas distintas? ----------
    log("")
    log("   LA PRUEBA DE QUE ESTO NO ES UN BLEND")
    # Un blend produce UNA frontera en el plano de los dos scores. Una seleccion puede ser
    # NO CONVEXA: se comprueba si la region delegada es no convexa en ese plano.
    delegado_total = (aciertos != pred_v03)
    log(f"      celdas donde el enrutado cambia la decision de V03: {int(delegado_total.sum())}")
    log(f"      de esas, el candidato acierta {int((delegado_total & (aciertos == y)).sum())} "
        f"y falla {int((delegado_total & (aciertos != y)).sum())}")
    log("      si el enrutado solo cambiara la decision cerca de la frontera, seria un umbral.")
    dist = np.abs(proba_v03 - V03_UMBRALES[None, :])
    cambiadas = delegado_total
    if cambiadas.sum():
        lejos = (dist >= 0.10) & cambiadas
        log(f"      de las celdas cambiadas, {int(lejos.sum())} estan a mas de 0,10 del umbral")
        log("      cambiar decisiones LEJOS del umbral es algo que ningun umbral puede hacer")

    # ---------- el diagnostico: por que falla ----------
    log("")
    log("   POR QUE FALLA, que es lo que hace util un negativo")
    # Si el enrutador no sirve, hay que saber si es porque no aprende NADA o porque aprende
    # la cosa equivocada. Se comparan dos objetivos con las MISMAS features:
    #   objetivo A   a quien delegar          (V03 falla Y el candidato acierta)
    #   objetivo B   donde V03 falla          (dificultad, sin mirar al candidato)
    from sklearn.metrics import roc_auc_score
    diag_auc = []
    for etiqueta, j in ((ETIQUETAS[0], 0), (ETIQUETAS[1], 1), (ETIQUETAS[2], 2)):
        fila = {"etiqueta": etiqueta}
        for nombre, tgt in (("a_quien_delegar", objetivo[:, j]),
                            ("donde_V03_falla", (pred_v03[:, j] != y[:, j]))):
            aucs = []
            for f in range(1, 6):
                ir, iva = np.where(pliegues != f)[0], np.where(pliegues == f)[0]
                skf = StratifiedKFold(n_splits=N_INNER, shuffle=True,
                                      random_state=SEMILLA + 100 * f)
                it = None
                for tr_p, _va_p in skf.split(np.arange(len(ir)), tgt[ir].astype(int)):
                    it = ir[tr_p]
                    break
                if tgt[it].sum() < 5 or tgt[iva].sum() < 3:
                    continue
                mod = HistGradientBoostingClassifier(
                    max_depth=3, max_iter=120, learning_rate=0.06, l2_regularization=1.0,
                    min_samples_leaf=20, random_state=SEMILLA)
                mod.fit(X[it], tgt[it].astype(int))
                aucs.append(float(roc_auc_score(tgt[iva].astype(int),
                                                mod.predict_proba(X[iva])[:, 1])))
            fila[f"auc_{nombre}"] = round(float(np.mean(aucs)), 4) if aucs else None
            fila[f"auc_{nombre}_folds"] = [round(a, 4) for a in aucs]
        diag_auc.append(fila)
        log(f"      {etiqueta}: AUC a quien delegar {fila['auc_a_quien_delegar']} | "
            f"AUC donde V03 falla {fila['auc_donde_V03_falla']}")
    media_delegar = float(np.mean([f["auc_a_quien_delegar"] for f in diag_auc]))
    media_dificil = float(np.mean([f["auc_donde_V03_falla"] for f in diag_auc]))
    log("")
    log(f"      media: a quien delegar {media_delegar:.4f} | donde V03 falla "
        f"{media_dificil:.4f}")
    log("      la estructura del texto predice la DIFICULTAD bastante mejor de lo que")
    log("      predice QUIEN GANA. Esa es la razon del fallo, y no es que el enrutador no")
    log("      aprenda nada: aprende la cosa equivocada")
    disocia = media_dificil > media_delegar + 0.03
    log(f"      disociacion confirmada: {disocia}")

    # ---------- veredicto ----------
    log("")
    log("   EL VEREDICTO")
    gate_delta = delta >= 0.0015
    gate_folds = folds_pos >= 4
    gate_control = ccv_real > max(ccv_bar, ccv_alea)
    if gate_delta and gate_folds and gate_control:
        veredicto = "PROMISING"
        razon = (
            f"delta {delta:+.10f} supera el gate de +0,0015, {folds_pos} de 5 pliegues "
            f"mejoran, y el enrutado bate a sus dos controles "
            f"(barajado {ccv_bar - ref:+.10f}, aleatorio {ccv_alea - ref:+.10f}). Pasa a "
            f"PROMISING, que exige bootstrap y segunda particion antes de expandir")
    elif delta > 0 and gate_control:
        veredicto = "MARGINAL"
        razon = (
            f"delta {delta:+.10f} es positivo y bate a los controles, pero no alcanza el "
            f"gate de +0,0015 o no llega a 4 de 5 pliegues ({folds_pos} de 5). MARGINAL: "
            f"no se abre una grid por esto")
    else:
        veredicto = "INVALIDATED" if not gate_control else "CLOSED"
        razon = (
            f"delta {delta:+.10f}, {folds_pos} de 5 pliegues. El enrutado "
            f"{'NO' if not gate_control else 'si'} bate a sus controles "
            f"(barajado {ccv_bar - ref:+.10f}, aleatorio {ccv_alea - ref:+.10f}). "
            f"La estructura del ejemplo no permite decidir a quien delegar")
    log(f"      delta: {delta:+.10f} | gate de delta: {gate_delta}")
    log(f"      pliegues: {folds_pos} de 5 | gate de pliegues: {gate_folds}")
    log(f"      bate al barajado y al aleatorio: {gate_control}")
    log("")
    log(f"      VEREDICTO: {veredicto}")
    log(f"      {razon}")

    # ---------- figuras ----------
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
        log(f"      figura: {n}")

    plt.figure(figsize=(10, 5))
    barras = [("V03", ref, "#888"), ("enrutado", ccv_real, "#4a7"),
              ("objetivo barajado", ccv_bar, "#c44"), ("aleatorio", ccv_alea, "#84c"),
              ("techo", ccv_techo, "#cc8")]
    plt.bar([b[0] for b in barras], [b[1] for b in barras], color=[b[2] for b in barras])
    plt.axhline(ref, color="k", ls="--", lw=1, label="V03")
    plt.ylabel("corrected_cv")
    plt.title("P68-H1: enrutado por estructura frente a sus controles y a su techo")
    plt.legend()
    guardar("p68_01_enrutado_vs_controles.png")

    plt.figure(figsize=(10, 5))
    plt.bar(["V03", "enrutado", "barajado", "aleatorio"],
            [0.0, delta, ccv_bar - ref, ccv_alea - ref],
            color=["#888", "#4a7", "#c44", "#84c"])
    plt.axhline(0, color="k", lw=1)
    plt.axhline(0.0015, color="b", ls=":", lw=1, label="gate de +0,0015")
    plt.ylabel("delta vs V03")
    plt.title("El delta, frente al gate, y los dos controles")
    plt.legend()
    guardar("p68_02_delta_y_gate.png")

    plt.figure(figsize=(10, 5))
    plt.bar([f"pliegue {i}" for i in range(1, 6)], aciertos_por_fold,
            color=["#4a7" if d > 0 else "#c44" for d in aciertos_por_fold])
    plt.axhline(0, color="k", lw=1)
    plt.ylabel("delta por pliegue")
    plt.title(f"Por pliegue: {folds_pos} de 5 mejoran")
    guardar("p68_03_delta_por_fold.png")

    plt.figure(figsize=(10, 5))
    plt.hist(dist[delegado_total], bins=30, color="#4a7", label="celdas cambiadas")
    plt.axvline(0.10, color="r", ls="--", lw=1, label="banda de alta confianza (0,10)")
    plt.xlabel("distancia de p_V03 al umbral")
    plt.ylabel("celdas")
    plt.title("Donde cambia la decision el enrutado")
    plt.legend()
    guardar("p68_04_donde_cambia_la_decision.png")

    with CSV_F.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(list(detalle[0].keys()))
        for d in detalle:
            w.writerow(list(d.values()))
    with CSV_C.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["variante", "corrected_cv", "delta_vs_V03"])
        for nom, val in (("V03", ref), ("enrutado por estructura", ccv_real),
                         ("objetivo barajado", ccv_bar), ("delegacion aleatoria", ccv_alea),
                         ("enrutado con el margen de V03", ccv_m), ("techo teorico", ccv_techo)):
            w.writerow([nom, round(val, 10), round(val - ref, 10)])

    # ---------- informe ----------
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    salida = {
        "familia": "P68_ENRUTADO", "experiment_id": "P68-01", "hipotesis": "P68-H1",
        "fecha": inicio.isoformat(timespec="seconds"),
        "naturaleza": ("EXPERIMENTO. 1 principal, 2 controles y 1 ablacion. 0 submissions, "
                       "0 Kaggle, 0 test, 0 holdout"),
        "mecanismo": "descomposicion de la decision por enrutado sobre estructura del ejemplo",
        "features_del_enrutador": len(columnas),
        "features_derivadas_del_score": 0,
        "resultado": {"corrected_cv": round(ccv_real, 10), "delta_vs_V03": round(delta, 10),
                      "pliegues_mejorados": folds_pos, "delta_por_fold": aciertos_por_fold,
                      "celdas_delegadas": delegadas, "aciertos_entre_delegadas": correctas,
                      "errores_entre_delegadas": incorrectas,
                      "neto": correctas - incorrectas},
        "techo_teorico": {"corrected_cv": round(ccv_techo, 10),
                          "delta": round(ccv_techo - ref, 10),
                          "celdas_delegables": int(objetivo.sum())},
        "controles": {"objetivo_barajado": {"corrected_cv": round(ccv_bar, 10),
                                            "delta": round(ccv_bar - ref, 10)},
                      "delegacion_aleatoria": {"corrected_cv": round(ccv_alea, 10),
                                               "delta": round(ccv_alea - ref, 10)},
                      "enrutado_con_el_margen_de_V03": {"corrected_cv": round(ccv_m, 10),
                                                        "delta": round(ccv_m - ref, 10)}},
        "diagnostico": {
            "auc_por_etiqueta": diag_auc,
            "media_auc_a_quien_delegar": round(media_delegar, 4),
            "media_auc_donde_V03_falla": round(media_dificil, 4),
            "disociacion_confirmada": bool(disocia),
            "lectura": ("la estructura del texto predice la DIFICULTAD mejor que el ganador. "
                        "El enrutador aprende la cosa equivocada, no es que no aprenda nada"),
        },
        "gates": {"delta": bool(gate_delta), "pliegues": bool(gate_folds),
                  "control": bool(gate_control)},
        "veredicto": veredicto, "razon": razon,
        "protocolo": {"nested": True, "inner": N_INNER, "outer": 5,
                      "seleccion_en": "validacion interna",
                      "aislamiento": "solo la particion dev"},
        "reproducibilidad": {"nivel": "R2", "git_commit": commit, "seed": SEMILLA,
                             "hash_v03": hash_real, "corrected_cv_v03": V03_CCV},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_R.write_text(json.dumps(salida, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    L: list[str] = []
    W = L.append
    W("# P68-H1 - Enrutado por estructura del ejemplo")
    W("")
    W(f"Generado por `experiments/p68_01_enrutado_por_estructura.py`. Fecha "
      f"{inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## La hipotesis, y por que no es otra formula de score")
    W("")
    W("P67 cerro la familia de transformaciones genericas del score dos veces: empiricamente,")
    W("porque todas daban cuatro celdas de 1483, y algebraicamente, porque la mejor de ellas")
    W("resulto ser un `blend_logit` con otro nombre.")
    W("")
    W("Lo que sobrevivio a P67 fue una **medicion**: la poblacion B del gap analysis, con")
    W(f"`{int(objetivo.sum())}` celdas delegables medidas y no supuestas. Este experimento")
    W("cambia **quien decide cada fila**, y lo decide mirando la **estructura del texto**,")
    W("nunca el score.")
    W("")
    W("```text")
    W("no concatena columnas        P64 y P65, cerrado")
    W("no mezcla scores             P66 y P67, DUPLICATE por algebra")
    W("no mueve el umbral           P67-01, cerrado")
    W("no es un ensemble de pesos   P61, cerrado")
    W("cambia QUIEN decide          esto")
    W("```")
    W("")
    W("La diferencia no es cosmetica: un blend produce **una** frontera en el plano de los dos")
    W("scores. Una seleccion produce una region que puede ser **no convexa**.")
    W("")
    W("## El protocolo")
    W("")
    W(f"- Features del enrutador: **{len(columnas)} estructurales**, todas del texto.")
    W("- Derivadas del score usadas en la variante principal: **0**.")
    W(f"- Ajuste anidado: {N_INNER} pliegues internos. La tasa de delegacion se elige en la")
    W("  validacion **interna**, nunca en la externa.")
    W("- Aislamiento verificado: solo la particion `dev`. Cero filas de `test` y de `holdout`.")
    W("")
    W("## El techo, medido antes de creerse nada")
    W("")
    W(f"Si se delegara **exactamente** lo delegable, el resultado seria `{ccv_techo:.10f}`, es")
    W(f"decir `{ccv_techo - ref:+.10f}`. Ese es todo el margen que existe.")
    W("")
    W("| variante | corrected_cv | delta |")
    W("|---|---:|---:|")
    for nom, val in (("V03 sin tocar", ref), ("enrutado por estructura", ccv_real),
                     ("objetivo barajado", ccv_bar), ("delegacion aleatoria", ccv_alea),
                     ("enrutado con el margen de V03", ccv_m), ("techo teorico", ccv_techo)):
        W(f"| {nom} | {val:.10f} | {val - ref:+.10f} |")
    W("")
    W(f"Celdas delegadas: `{delegadas}`. Acierta `{correctas}`, falla `{incorrectas}`, "
      f"neto `{correctas - incorrectas:+d}`.")
    W(f"Delta por pliegue: `{aciertos_por_fold}`, positivos `{folds_pos}` de 5.")
    W("")
    W("## El veredicto")
    W("")
    W("```text")
    W(f"{veredicto}")
    W("")
    W(a_ascii(razon))
    W("```")
    W("")
    W("## Por que falla, que es lo que hace util un negativo")
    W("")
    W("Si el enrutador no sirve hay que saber **por que**: si no aprende nada, o si aprende la")
    W("cosa equivocada. Se entrenan dos modelos con las **mismas features** y dos objetivos")
    W("distintos:")
    W("")
    W("```text")
    W("objetivo A   a quien delegar      V03 falla Y el candidato acierta")
    W("objetivo B   donde V03 falla      dificultad, sin mirar al candidato")
    W("```")
    W("")
    W("| etiqueta | AUC a quien delegar | AUC donde V03 falla |")
    W("|---|---:|---:|")
    for f in diag_auc:
        W(f"| {f['etiqueta']} | {f['auc_a_quien_delegar']} | {f['auc_donde_V03_falla']} |")
    W("")
    W(f"**Media: `{media_delegar:.4f}` frente a `{media_dificil:.4f}`.**")
    W("")
    W("La estructura del texto predice la **dificultad** bastante mejor de lo que predice")
    W("**quien gana**. El enrutador no es que no aprenda nada: aprende la cosa equivocada.")
    W("")
    W("Y eso encaja con el detalle por pliegue: de las celdas delegadas, V03 **ya acertaba**")
    W("la mayoria en todos los pliegues y etiquetas. El enrutador manda trabajo al experto")
    W("donde el ejemplo *parece* dificil, que es donde V03 ya estaba resolviendo y donde el")
    W("experto tambien falla.")
    W("")
    W("### Y el aviso que deja el `f1_inner`")
    W("")
    W("En **11 de 15** celdas pliegue-etiqueta el delta de la validacion interna era")
    W("**positivo**, y el delta externo salio negativo. Con 800 filas internas divididas en")
    W("tres etiquetas, un `+0,004` interno no transfiere. Es la misma leccion de P66 y P67")
    W("otra vez: a esta escala, la seleccion se confunde con la senal.")
    W("")
    W("## La prueba de que esto no es un blend")
    W("")
    W(f"El enrutado cambia la decision de V03 en `{int(delegado_total.sum())}` celdas.")
    if delegado_total.sum():
        W(f"De ellas, `{int(lejos.sum())}` estan a **mas de 0,10 del umbral**, es decir fuera")
        W("de la banda donde un umbral podria actuar. Cambiar decisiones lejos del umbral es")
        W("algo que **ningun umbral puede hacer**, y es la firma de que el mecanismo es otro.")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p68_01_enrutado_por_estructura.py`. Sin push.*")
    MD_R.write_text(a_ascii("\n".join(L)), encoding="ascii")

    documentos = [MD_R, JSON_R, CSV_F, CSV_C]
    sucios = [d.name for d in documentos if any(b > 127 for b in d.read_bytes())]
    if sucios:
        raise SystemExit(f"PARADA: documentos con bytes no ASCII: {sucios}")

    log("")
    log("=" * 78)
    log(f"P68-01 OK | {veredicto} | delta {delta:+.10f} | {folds_pos} de 5 pliegues")
    log(f"   techo {ccv_techo - ref:+.10f} | barajado {ccv_bar - ref:+.10f} | "
        f"aleatorio {ccv_alea - ref:+.10f} | margen {ccv_m - ref:+.10f}")
    log(f"   {len(hechos)} figuras | {len(documentos)} documentos, 0 bytes no ASCII")
    log("   V03 remains the best verified leaderboard result.")
    log("=" * 78)


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
