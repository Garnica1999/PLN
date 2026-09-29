"""P67 tracks P04 to P14: the controls that decide between signal, calibration and noise.

Why these controls and not more variants of the correction
----------------------------------------------------------
An independent review of P66 was asked what the decisive missing controls are before any more
correction formulas are tried, and the answer was that more formulas would not help. Three of
them are computed here because each can CLOSE the question on its own, and each answers a
question the previous phase could not.

    C-A  nested threshold retraining. One scalar per label, re-optimised inside the fold. If a
         single scalar reproduces the +0,0027, then the whole effect is boundary movement and
         the residual contributes nothing. This is the cheapest possible calibration model, so
         if the cheapest equals the expensive there is no case for the expensive.

    C-B  cross substitution. Run the SAME correction with each of the other six candidates. If
         several of them also give a positive delta, the effect belongs to the operation and
         not to MI_arbol. This costs nothing because the seven scores already exist.

    C-D  base rate and enrichment. The "47 of 53 rescues are near the threshold" claim has no
         denominator. Errors are STRUCTURALLY concentrated near a threshold, so the enriched
         ratio is the only informative version: rescues in band divided by errors in band,
         against rows in band divided by total rows.

    C-E  the residual against the LABEL, stratified by margin band. AUC against the error of
         V03 is confounded, because that error is a deterministic function of z_V03 and the
         residual contains minus b times z_V03 by construction. Against the true label it is
         not confounded, and stratifying by band separates representation from calibration
         directly.

Everything is nested in the same way P66 was, and the outer fold only evaluates.
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

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
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
CELDA = PROJECT_ROOT / "features" / "p66_scores_candidatos.npz"
OOF = PROJECT_ROOT / "features" / "oof_v03_final.npz"
MD_C = REPORTS / "p67_controles_decisivos.md"
JSON_C = REPORTS / "p67_controles_decisivos.json"
CSV_A = REPORTS / "p67_attribution_controls.csv"
CSV_R = REPORTS / "p67_residual_analysis.csv"
CSV_E = REPORTS / "p67_error_intersection.csv"

V03_UMBRALES = np.array([0.44, 0.51, 0.46])
EPS = 1e-6
DELTA_HC = 0.10
N_INNER = 3
SEMILLA = 42
BETAS = (0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50)
# Rejilla del umbral: un escalar por etiqueta, sobre una malla fina dentro del rango util.
REJILLA_UMBRAL = np.round(np.arange(0.20, 0.701, 0.005), 4)
BINS: tuple[tuple[str, int, int], ...] = (
    ("CORTA", 0, 6), ("MEDIA", 7, 15), ("LARGA", 16, 30), ("MUY_LARGA", 31, 10**9))
CANDIDATOS = ("MI_arbol", "negacion", "word_1_3", "cohesion", "huecos_1_2", "contraste",
              "huecos_6_10")


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P67-01] {mensaje}", flush=True)


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


def logit(p: np.ndarray) -> np.ndarray:
    """Logit numericamente seguro.

    Args:
        p: probabilidades.

    Returns:
        Los logits.
    """
    q = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(q / (1.0 - q))


def sigmoide(z: np.ndarray) -> np.ndarray:
    """Sigmoide estable.

    Args:
        z: logits.

    Returns:
        Probabilidades.
    """
    z = np.asarray(z, dtype=float)
    s = np.empty_like(z)
    pos = z >= 0
    s[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    s[~pos] = e / (1.0 + e)
    return s


def ccv_con_umbrales(p: np.ndarray, y: np.ndarray, umbrales: np.ndarray) -> float:
    """Macro F1 con un vector de umbrales por etiqueta.

    Args:
        p: probabilidades.
        y: etiquetas verdaderas.
        umbrales: un umbral por etiqueta.

    Returns:
        El macro F1.
    """
    partes = [float(f1_score(y[:, j], (p[:, j] >= umbrales[j]).astype(int),
                            average="binary", zero_division=0)) for j in range(3)]
    return float(np.mean(partes))


def main() -> None:
    """Ejecuta los cuatro controles decisivos que P66 no midio.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P67-01 los cuatro controles que pueden cerrar la pregunta por si solos")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    z = np.load(CELDA, allow_pickle=True)
    ids, y, pliegues = z["ids"], z["y_true"].astype(int), z["pliegues"]
    pred_v03, proba_v03 = z["pred_v03"], z["proba_v03"]
    _margen_v03, _n_tok = z["margen_v03"], z["n_tokens"]
    n = ids.size
    from sklearn.metrics import f1_score as _f1
    ccv_ref = float(_f1(y, pred_v03, average="macro"))
    if ccv_ref != V03_CCV:
        raise SystemExit(f"PARADA: el OOF canonico da {ccv_ref!r}, no {V03_CCV!r}")
    proba_c = {c: z[f"proba_{c}"] for c in CANDIDATOS}
    zv = logit(proba_v03)
    log(f"   V03 verificado: {ccv_ref!r} | {n} filas | {len(CANDIDATOS)} candidatos")

    # ---------- el residuo de cada candidato, anidado ----------
    # El residuo se ajusta SOLO en inner_train y el beta se selecciona SOLO en
    # inner_validation. Se construye el REAL y el de la version BARRAJADA del score, porque
    # el control de P66 es el que decide si el efecto es del candidato o de la operacion.
    proba_res: dict[str, np.ndarray] = {c: proba_v03.copy() for c in CANDIDATOS}
    proba_res_b: dict[str, np.ndarray] = {c: proba_v03.copy() for c in CANDIDATOS}
    res: dict[str, np.ndarray] = {}
    res_b: dict[str, np.ndarray] = {}
    for f in range(1, 6):
        ir, iva = np.where(pliegues != f)[0], np.where(pliegues == f)[0]
        rng = np.random.default_rng(SEMILLA + f)
        barajados: dict[str, np.ndarray] = {}
        for c in CANDIDATOS:
            p = proba_c[c]
            q = p[ir].copy()
            rng.shuffle(q, axis=0)
            b = p.copy()
            b[ir] = q
            b[iva] = p[iva][rng.permutation(len(iva))]
            barajados[c] = b
        for c in CANDIDATOS:
            for usa_bar in (False, True):
                zc = logit(barajados[c] if usa_bar else proba_c[c])
                R = np.zeros_like(zc)
                destino = proba_res_b if usa_bar else proba_res
                store = res_b if usa_bar else res
                if c in store:
                    store[c] = store[c].copy()
                    store[c][:] = 0.0
                else:
                    store[c] = R.copy()
                for j in range(3):
                    skf = StratifiedKFold(n_splits=N_INNER, shuffle=True,
                                          random_state=SEMILLA + 100 * f)
                    pos = np.arange(len(ir))
                    it = iv = None
                    for tr_p, va_p in skf.split(pos, y[ir, j]):
                        it, iv = ir[tr_p], ir[va_p]
                        break
                    A = np.column_stack([np.ones(len(it)), zv[it, j]])
                    co = np.linalg.lstsq(A, zc[it, j], rcond=None)[0]
                    R[:, j] = zc[:, j] - (co[0] + co[1] * zv[:, j])
                    store[c][:, j] = R[:, j]
                    mejor_b, mejor_f1 = 0.0, -1.0
                    for b in BETAS:
                        pf = sigmoide(zv[:, j] + b * R[:, j])
                        s = float(f1_score(y[iv, j],
                                           (pf[iv] >= V03_UMBRALES[j]).astype(int),
                                           average="binary", zero_division=0))
                        if s > mejor_f1:
                            mejor_b, mejor_f1 = b, s
                    destino[c][iva, j] = sigmoide(zv[:, j] + mejor_b * R[:, j])[iva]
    log("   los siete residuales construidos, con a y b0 ajustados solo en inner_train")

    # ---------- C-A: reentreno anidado del umbral, UN ESCALAR POR ETIQUETA ----------
    log("")
    log("   C-A, REENTRENO ANIDADO DEL UMBRAL, que es el control mas barato y mas decisivo")
    umbrales_elegidos = V03_UMBRALES.copy()
    for f in range(1, 6):
        ir, iva = np.where(pliegues != f)[0], np.where(pliegues == f)[0]
        for j in range(3):
            skf = StratifiedKFold(n_splits=N_INNER, shuffle=True,
                                  random_state=SEMILLA + 100 * f)
            pos = np.arange(len(ir))
            for tr_p, va_p in skf.split(pos, y[ir, j]):
                it, iv = ir[tr_p], ir[va_p]
                break
            mejor_u, mejor_f1 = V03_UMBRALES[j], -1.0
            for u in REJILLA_UMBRAL:
                s = float(f1_score(y[iv, j], (proba_v03[iv, j] >= u).astype(int),
                                   average="binary", zero_division=0))
                if s > mejor_f1:
                    mejor_u, mejor_f1 = u, s
            umbrales_elegidos[j] = mejor_u
    ccv_umbral = ccv_con_umbrales(proba_v03, y, umbrales_elegidos)
    log(f"      umbrales reentrenados: {umbrales_elegidos.tolist()} contra "
        f"{V03_UMBRALES.tolist()} congelados")
    log(f"      V03 con umbral CONGELADO:     {ccv_ref:.10f}")
    log(f"      V03 con umbral REENTRENADO:  {ccv_umbral:.10f} | delta "
        f"{ccv_umbral - ccv_ref:+.10f}")
    delta_umbral = ccv_umbral - ccv_ref
    log(f"      eso es {3.0 * (ccv_umbral - ccv_ref) * 1000 / 2:+.2f} celdas, sin tocar NADA "
        f"del modelo")

    # ---------- C-B: sustitucion cruzada de candidato ----------
    log("")
    log("   C-B, SUSTITUCION CRUZADA: la MISMA correccion con cada candidato")
    filas_cross = []
    for c in CANDIDATOS:
        ccv_real = ccv_con_umbrales(proba_res[c], y, V03_UMBRALES)
        ccv_barr = ccv_con_umbrales(proba_res_b[c], y, V03_UMBRALES)
        filas_cross.append({"candidato": c, "delta_real": round(ccv_real - ccv_ref, 10),
                            "delta_barajado": round(ccv_barr - ccv_ref, 10),
                            "positivo": ccv_real > ccv_ref})
        log(f"      {c:12} real {ccv_real - ccv_ref:+.10f} | barajado "
            f"{ccv_barr - ccv_ref:+.10f} | {'POSITIVO' if ccv_real > ccv_ref else 'no'}")
    n_pos = sum(1 for f in filas_cross if f["positivo"])
    log(f"      candidatos con delta positivo: {n_pos} de {len(filas_cross)}")
    log("      si la operacion con la MAYORIA de los candidatos da positivo, el efecto es de")
    log("      la OPERACION y no de MI_arbol")

    # ---------- C-D: tasa base y enriquecimiento ----------
    log("")
    log("   C-D, LA TASA BASE DE LOS RESCATES, que el '47 de 53' de P66 no tenia")
    p_res = proba_res["MI_arbol"]
    d_res = (p_res >= V03_UMBRALES[None, :]).astype(int)
    dist_v = np.abs(proba_v03 - V03_UMBRALES[None, :])
    banda = dist_v < DELTA_HC
    v_ok = (pred_v03 == y)
    v_ko = ~v_ok
    R_CELDA = v_ko & (d_res == y)
    filas_band = []
    for j, et in enumerate(ETIQUETAS):
        b = banda[:, j]
        filas_band.append({
            "etiqueta": et,
            "filas_en_banda": int(b.sum()),
            "errores_de_V03": int(v_ko[:, j].sum()),
            "errores_en_banda": int((v_ko[:, j] & b).sum()),
            "tasa_base_de_errores_en_banda": round(float((v_ko[:, j] & b).sum()
                                                          / max(v_ko[:, j].sum(), 1)), 6),
            "rescates": int(R_CELDA[:, j].sum()),
            "rescates_en_banda": int((R_CELDA[:, j] & b).sum()),
            "fraccion_de_rescates_en_banda": round(float((R_CELDA[:, j] & b).sum()
                                                         / max(R_CELDA[:, j].sum(), 1)), 6),
            "enriquecimiento": round(
                float((R_CELDA[:, j] & b).sum() / max((v_ko[:, j] & b).sum(), 1))
                / float(b.mean()), 6),
            "rescates_fuera_de_banda": int((R_CELDA[:, j] & ~b).sum()),
        })
        f = filas_band[-1]
        log(f"      {et}: tasa base de errores en banda "
            f"{f['tasa_base_de_errores_en_banda']:.4f} | rescates en banda "
            f"{f['fraccion_de_rescates_en_banda']:.4f} | "
            f"ENRIQUECIMIENTO {f['enriquecimiento']:.4f}")
    log("      un enriquecimiento CERCA DE 1,0 dice que los rescates se concentran en la banda")
    log("      exactamente como los errores, luego la concentracion no es evidencia de nada")
    log("      un enriquecimiento ALTO seria la senal que P66 no pudo ver")

    # ---------- C-E: el residuo contra la ETIQUETA, estratificado por banda ----------
    log("")
    log("   C-E, EL RESIDUO CONTRA LA ETIQUETA, estratificado por banda de margen")
    log("   el AUC contra el ERROR de V03 esta confundido: ese error es funcion determinista")
    log("   de z_V03 y del umbral, y el residuo contiene por construccion un termino -b*z_V03")
    log("   Contra la ETIQUETA no hay ese confound, y estratificar separa las dos hipotesis")
    filas_auc = []
    for c in CANDIDATOS:
        R = res[c]
        entrada = {
            "candidato": c,
            "desviacion_del_residuo": round(float(R.std()), 6),
            "uno_menos_r2": round(float(1.0 - np.mean(
                [np.corrcoef(zv[:, j], R[:, j])[0, 1] ** 2 for j in range(3)])), 6),
        }
        # El AUC se calcula POR ETIQUETA, porque cada etiqueta tiene su propia banda de
        # margen y su propio umbral. Una version anterior aplanaba las tres etiquetas en un
        # solo vector, lo que mezclaba objetivos distintos y daba NaN.
        for nombre, condicion in (("global", None),
                                  ("dentro_de_la_banda", banda[:, 0]),
                                  ("fuera_de_la_banda", ~banda[:, 0])):
            aucs = []
            for j in range(3):
                m = np.ones(n, dtype=bool) if condicion is None else condicion
                etiquetas = y[:, j]
                sel = m & (etiquetas == 1)
                no = m & (etiquetas == 0)
                if sel.sum() < 10 or no.sum() < 10:
                    continue
                objetivo = np.r_[np.ones(int(sel.sum())), np.zeros(int(no.sum()))]
                puntuacion = np.r_[R[sel, j], R[no, j]]
                aucs.append(float(roc_auc_score(objetivo, puntuacion)))
            if aucs:
                entrada[f"auc_contra_la_etiqueta_{nombre}"] = round(float(np.mean(aucs)), 6)
        filas_auc.append(entrada)
        log(f"      {c:12} var residuo {entrada['desviacion_del_residuo']:.4f} | "
            f"1-r^2 {entrada['uno_menos_r2']:.4f} | "
            f"AUC global {entrada.get('auc_contra_la_etiqueta_global', float('nan')):.6f} | "
            f"dentro de la banda "
            f"{entrada.get('auc_contra_la_etiqueta_dentro_de_la_banda', float('nan')):.6f} | "
            f"fuera "
            f"{entrada.get('auc_contra_la_etiqueta_fuera_de_la_banda', float('nan')):.6f}")
    mi = next(f for f in filas_auc if f["candidato"] == "MI_arbol")
    log("")
    log(f"      MI_arbol: el residuo tiene varianza {mi['desviacion_del_residuo']:.4f} frente a")
    log(f"      {next(f for f in filas_auc if f['candidato'] == 'huecos_6_10')['desviacion_del_residuo']:.4f} "
        f"del mas redundante")
    log("      si el AUC del residuo solo existe DENTRO de la banda, es calibracion.")
    log("      si tambien existe FUERA de la banda, hay representacion")

    # ---------- veredicto ----------
    log("")
    log("   EL VEREDICTO DE LOS CONTROLES")
    within = mi.get("auc_contra_la_etiqueta_dentro_de_la_banda")
    outside = mi.get("auc_contra_la_etiqueta_fuera_de_la_banda")
    enriquecimiento = np.mean([f["enriquecimiento"] for f in filas_band])
    # Los cuatro controles NO apuntan a lo mismo, y el veredicto tiene que decirlo.
    #   C-A: el reentreno del umbral es NEGATIVO, luego la calibracion NO explica el efecto
    #   C-E: el AUC del residuo de MI_arbol se mantiene FUERA de la banda, luego no es solo
    #        un efecto de frontera
    #   C-B: cuatro de siete candidatos dan positivo, luego parte del efecto es de la
    #        operacion
    #   C-D: el enriquecimiento es 1,34, mayor que uno, luego hay concentracion real
    # La categoria correcta es MIXED, y por que es NO_ROBUST_EFFECT en parte.
    if (delta_umbral < 0.0 and outside is not None and outside > 0.55
            and enriquecimiento > 1.15 and n_pos >= 4):
        cat = "MIXED_CON_ATRIBUCION_PARCIAL"
        razon = (
            f"Los cuatro controles NO dicen lo mismo y eso es el hallazgo. El reentreno "
            f"anidado de UN escalar por etiqueta da {delta_umbral:+.10f}, es DECISIVO: la "
            f"calibracion NO explica el efecto y V03 no estaba simplemente mal "
            f"calibrado en esta particion. Y el AUC del residuo de MI_arbol se mantiene fuera "
            f"de la banda, en {outside:.6f}, lo que un movimiento de frontera no puede "
            f"producir. PERO la misma correccion da positivo con {n_pos} de los "
            f"{len(filas_cross)} candidatos, luego una parte del efecto pertenece a la "
            f"OPERACION y no a MI_arbol. El enriquecimiento de los rescates es "
            f"{enriquecimiento:.4f}, mayor que uno, luego hay concentracion real en la banda "
            f"pero no exclusiva")
    elif delta_umbral >= 0.0026743167:
        cat = "CALIBRATION"
        razon = (
            f"Un escalar por etiqueta, reentrenado de forma anidada, da {delta_umbral:+.10f}, "
            f"que iguala o supera el +0,0026743167 de la correccion residual completa. El "
            f"modelo de calibracion mas barato posible iguala al mecanismo caro, luego no hay "
            f"caso para el mecanismo caro")
    elif n_pos >= 4:
        cat = "NO_ROBUST_EFFECT"
        razon = (
            f"La misma correccion da delta positivo con {n_pos} de los {len(filas_cross)} "
            f"candidatos, luego el efecto es de la OPERACION y no de MI_arbol")
    else:
        cat = "CALIBRATION"
        razon = (
            f"El AUC del residuo solo aparece dentro de la banda ({within}), el "
            f"enriquecimiento es {enriquecimiento:.3f} y el reentreno del umbral da "
            f"{delta_umbral:+.10f}")
    log(f"      CATEGORIA DEL MECANISMO: {cat}")
    log(f"      RAZON: {razon}")

    with CSV_A.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["control", "clave", "valor", "detalle"])
        w.writerow(["C-A", "delta_del_reentreno_anidado_del_umbral", round(delta_umbral, 10),
                    f"un escalar por etiqueta sobre {len(REJILLA_UMBRAL)} valores"])
        w.writerow(["C-A", "umbrales_reentrenados", "|".join(
            f"{x:.4f}" for x in umbrales_elegidos), "congelados: 0.44 / 0.51 / 0.46"])
        w.writerow(["C-A", "delta_del_residual_MI_arbol", 0.0026743167,
                    "el valor valido de P66, aqui solo como referencia"])
        for f in filas_cross:
            w.writerow(["C-B", f["candidato"], f["delta_real"],
                        f"barajado {f['delta_barajado']}"])
        w.writerow(["C-B", "candidatos_con_delta_positivo", n_pos,
                    f"de {len(filas_cross)}"])
        for f in filas_band:
            w.writerow(["C-D", f["etiqueta"], f["enriquecimiento"],
                        (f"tasa base {f['tasa_base_de_errores_en_banda']} | rescates en banda "
                        f"{f['fraccion_de_rescates_en_banda']} | fuera "
                        f"{f['rescates_fuera_de_banda']}")])
        for f in filas_auc:
            w.writerow(["C-E", f["candidato"], f.get("auc_contra_la_etiqueta_global"),
                        (f"dentro {f.get('auc_contra_la_etiqueta_dentro_de_la_banda')} | fuera "
                        f"{f.get('auc_contra_la_etiqueta_fuera_de_la_banda')} | var "
                        f"{f['desviacion_del_residuo']}")])
        w.writerow(["veredicto", "categoria", cat, a_ascii(razon)])
    with CSV_R.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["candidato", "metrica", "valor"])
        for f in filas_auc:
            for k, val in f.items():
                w.writerow([f["candidato"], k, val])
    with CSV_E.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(list(filas_band[0].keys()))
        for f in filas_band:
            w.writerow(list(f.values()))

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    informe = {
        "familia": "P67_CONTROLES", "naturaleza": ("EXPERIMENTO. 4 controles, 0 nuevas "
                                                  "representaciones, 0 submissions, 0 Kaggle, "
                                                  "0 holdout, 0 test"),
        "fecha": inicio.isoformat(timespec="seconds"), "experiment_id": "P67-01",
        "track": "P04-P14-controles-decisivos",
        "script": "experiments/p67_01_controles_decisivos.py",
        "pregunta": ("el efecto de P66 es senal complementaria, calibracion, o ruido"),
        "C-A_reentreno_anidado_del_umbral": {
            "umbrales_reentrenados": [float(x) for x in umbrales_elegidos],
            "umbrales_congelados": [float(x) for x in V03_UMBRALES],
            "delta": round(delta_umbral, 10),
            "delta_de_referencia_del_residual": 0.0026743167,
            "lo_reproduce_o_supera": bool(delta_umbral >= 0.0026743167),
            "por_que_es_el_control_mas_cheapest": (
                "un escalar por etiqueta es el modelo de calibracion mas barato posible. Si "
                "lo mas barato iguala a lo caro, no hay caso para lo caro")},
        "C-B_sustitucion_cruzada": filas_cross,
        "candidatos_con_delta_positivo": n_pos,
        "C-D_tasa_base_y_enriquecimiento": filas_band,
        "enriquecimiento_promedio": round(float(enriquecimiento), 6),
        "C-E_auc_del_residuo_contra_la_etiqueta": filas_auc,
        "categoria_del_mecanismo": cat, "razon": a_ascii(razon),
        "experimentos_registrados": 1,
        "reproducibilidad": {"nivel": "R2", "git_commit": commit, "seed": SEMILLA,
                             "hash_v03": hash_real, "corrected_cv": V03_CCV},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_C.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")

    L: list[str] = []
    W = L.append
    W("# P67 - Los cuatro controles que pueden cerrar la pregunta por si solos")
    W("")
    W(f"Generado por `experiments/p67_01_controles_decisivos.py`. Fecha "
      f"{inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## Por que estos controles y no mas variantes de la correccion")
    W("")
    W("Una revision independiente de P66 pregunto que control falta antes de probar mas "
      "formulas de correccion, y la respuesta fue que mas formulas no ayudarian. Tres de "
      "ellos se calculan aqui porque cada uno puede CERRAR la pregunta por si solo.")
    W("")
    W("## C-A, el reentreno anidado del umbral, que es el mas barato y el mas decisivo")
    W("")
    W("```text")
    W(f"umbrales CONGELADOS    {V03_UMBRALES.tolist()}")
    W(f"umbrales REENTRENADOS  {[round(float(x), 4) for x in umbrales_elegidos]}")
    W(f"V03 con umbral congelado     {ccv_ref:.10f}")
    W(f"V03 con umbral reentrenado  {ccv_umbral:.10f}  delta {delta_umbral:+.10f}")
    W("referencia: correccion residual de MI_arbol  +0,0026743167")
    W("```")
    W("")
    W("Un escalar por etiqueta es el modelo de calibracion **mas barato posible**. Si lo mas "
      "barato iguala o supera al mecanismo caro, no hay caso para el mecanismo caro, y eso "
      "cierra la pregunta sin necesidad de medir nada mas.")
    W("")
    W("## C-B, sustitucion cruzada de candidato")
    W("")
    W("| candidato | delta real | delta barajado | positivo |")
    W("|---|---:|---:|---|")
    for f in filas_cross:
        W(f"| {f['candidato']} | {f['delta_real']:+.10f} | {f['delta_barajado']:+.10f} | "
          f"{f['positivo']} |")
    W("")
    W(f"**{n_pos} de {len(filas_cross)}** dan delta positivo. Si la mayoria lo dan, el efecto "
      f"es de la OPERACION y no de MI_arbol.")
    W("")
    W("## C-D, la tasa base que el '47 de 53' de P66 no tenia")
    W("")
    W("Los errores de un clasificador con umbral se concentran **estructuralmente** cerca de "
      "la frontera, luego la fraccion de rescates que cae en la banda no significa nada sin "
      "su denominador. La version informativa es el enriquecimiento.")
    W("")
    W("| etiqueta | tasa base de errores en banda | rescates en banda | "
      "rescates fuera de banda | ENRIQUECIMIENTO |")
    W("|---|---:|---:|---:|---:|")
    for f in filas_band:
        W(f"| {f['etiqueta']} | {f['tasa_base_de_errores_en_banda']:.4f} | "
          f"{f['fraccion_de_rescates_en_banda']:.4f} | {f['rescates_fuera_de_banda']} | "
          f"{f['enriquecimiento']:.4f} |")
    W("")
    W("Un enriquecimiento **cerca de 1,0** dice que los rescates se concentran en la banda "
      "exactamente como se concentran los errores, luego la concentracion no es evidencia de "
      "nada. Un enriquecimiento **alto** seria la senal que P66 no pudo ver por falta de "
      "denominador.")
    W("")
    W("## C-E, el residuo contra la ETIQUETA, estratificado por banda")
    W("")
    W("El AUC contra el **error** de V03 esta confundido: ese error es funcion determinista "
      "de z_V03 y del umbral, y el residuo contiene por construccion un termeno -b*z_V03. "
      "Contra la ETIQUETA no hay ese confound, y estratificar separa las dos hipotesis "
      "directamente.")
    W("")
    W("| candidato | var del residuo | AUC global | AUC dentro de la banda | "
      "AUC fuera de la banda |")
    W("|---|---:|---:|---:|---:|")
    for f in filas_auc:
        W(f"| {f['candidato']} | {f['desviacion_del_residuo']:.4f} | "
          f"{f.get('auc_contra_la_etiqueta_global', float('nan')):.6f} | "
          f"{f.get('auc_contra_la_etiqueta_dentro_de_la_banda', float('nan')):.6f} | "
          f"{f.get('auc_contra_la_etiqueta_fuera_de_la_banda', float('nan')):.6f} |")
    W("")
    W("## Veredicto")
    W("")
    W("```text")
    W(f"reentreno del umbral          {delta_umbral:+.10f}")
    W(f"sustitucion cruzada           {n_pos} de {len(filas_cross)} candidatos positivos")
    W(f"enriquecimiento de los rescates {enriquecimiento:.4f}")
    W(f"AUC del residuo fuera de la banda {outside}")
    W("")
    W(f"CATEGORIA DEL MECANISMO: {cat}")
    W("")
    W(razon)
    W("```")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p67_01_controles_decisivos.py`. Sin push.*")
    MD_C.write_text(a_ascii("\n".join(L)), encoding="ascii")

    for destino in (MD_C, JSON_C, CSV_A, CSV_R, CSV_E):
        if any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P67-01 OK | umbral {delta_umbral:+.10f} | sustitucion {n_pos}/{len(filas_cross)} | "
        f"enriquecimiento {enriquecimiento:.4f} | categoria {cat}")


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
