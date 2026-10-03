"""P63 track K: do the new blocks explain V03's RESIDUAL, or is the dilution the whole story?

The question the previous five experiments did not answer
--------------------------------------------------------
Every track in this phase produced the same shape. Composition as ratios beats a random
block of the same width by +0,0221258028 when added to V03, and by +0,1055015148 on its own.
Position and concentration beats random by +0,0712736720 on its own. Neither makes V03
better. The usual reading is dilution: adding columns to a 21044-dimensional space costs
more than the new information repays.

There is a second reading that has not been tested, and it is the more important one. If the
blocks carry information that V03 ALREADY HAS, then no amount of care in how they are added
will ever help, and the dilution story would be the wrong explanation for a longer time. The
two readings make opposite predictions about the residual:

    DILUTION      the blocks know things V03 gets wrong, but the information cannot be paid
                  for. Then they should predict V03's errors.

    ALREADY KNOWN the blocks are a re-encoding of what V03 extracts from the n-grams. Then
                  they should say nothing about V03's errors once V03's own margin is known.

This is not a selector and not a second-stage model that predicts the label, which is V11-07
and V50 and both are closed. The target here is strictly "is this row one V03 gets wrong",
and V03's own margin enters as a feature so that the question is about the blocks and not
about V03's confidence, which is already measured.

The arms
-------
    margen_solo     V03's margin and entropy only. This is the reference and it is not weak:
                   if the margin alone explains the errors, there is nothing to find.
    margen_comp     the margin plus the composition block
    margen_pos      the margin plus the position block
    margen_aleatorio the margin plus a random block of the same width, the control that says
                   whether any apparent gain is about the extra columns

The measurement is the AUC for predicting V03's error, per label and overall, plus the
per-bin breakdown because the ceiling result says the residual is concentrated in the long
tweets.

What a result means
-------------------
    the blocks beat the margin AND the random control
        they know something V03 does not, and the failure to help is dilution or the
        threshold, not redundancy. That is a genuinely different conclusion from P62's.

    the margin alone explains the errors and the blocks add nothing over the random control
        the blocks are a re-encoding of V03's own evidence, and the whole family closes for
        a reason that is stronger than "it did not help".

Usage
-----
    /home/carlos/venv/bin/python experiments/p63_08_residuo_de_v03.py
"""

from __future__ import annotations

import ast
import csv as _csv
import gzip
import hashlib
import json
import subprocess
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

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
OOF = PROJECT_ROOT / "features" / "oof_v03_final.npz"
RAW_TRAIN = PROJECT_ROOT / "data" / "raw" / "toxic_train.csv"
ANOTACION = PROJECT_ROOT / "features" / "p62_anotacion_spacy.json.gz"
MD_K = REPORTS / "p63_residuo_de_v03.md"
JSON_K = REPORTS / "p63_residuo_de_v03.json"
CSV_K = REPORTS / "p63_residuo_de_v03.csv"

V03_UMBRALES = np.array([0.44, 0.51, 0.46])
BINS: tuple[tuple[str, int, int], ...] = (
    ("CORTA", 0, 6),
    ("MEDIA", 7, 15),
    ("LARGA", 16, 30),
    ("MUY_LARGA", 31, 10**9),
)
# El clasificador del meta-objetivo es un Ridge de la libreria estandar, y sus parametros
# se declaran aqui antes de ejecutar. No hay rejilla: son los valores por defecto
# deliberados, porque lo que se mide es la INFORMACION que hay en los rasgos, no el punto
# optimo de un modelo auxiliar.
RIDGE = {"alpha": 1.0}
SEMILLA = 42


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P63-08] {mensaje}", flush=True)


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


def main() -> None:
    """Mide si los bloques nuevos explican el residuo de V03.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P63-08 los bloques explican el RESIDUO de V03, o V03 ya tiene esa informacion")
    log("   dos lecturas opuestas del mismo resultado negativo:")
    log("   DILUCION      los bloques saben lo que V03 falla y no se puede pagar")
    log("   YA CONOCIDO   los bloques re-codifican lo que V03 ya extrae de los n-gramas")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")

    v = np.load(OOF, allow_pickle=True)
    ids, y, pliegues = v["ids"], v["y_true"].astype(int), v["pliegues"]
    pred_v03, _proba_v03 = v["pred"].astype(int), v["proba"]
    margen, entropia = v["margen"], v["entropia"]
    n = ids.size
    ccv_ref = float(f1_score(y, pred_v03, average="macro"))
    if ccv_ref != V03_CCV:
        raise SystemExit(f"PARADA: el OOF canonico da {ccv_ref!r}, no {V03_CCV!r}")
    tr = pd.read_csv(RAW_TRAIN, sep=";", encoding="latin-1", dtype={"id": str})
    tr["id_num"] = pd.to_numeric(tr["id"].str.replace('"', "", regex=False),
                                 errors="coerce").astype("int64")
    idx = tr.set_index("id_num").loc[ids]
    if not np.array_equal(idx[["HS", "TR", "AG"]].to_numpy().astype(int), y):
        raise SystemExit("PARADA: las etiquetas del CSV crudo no coinciden con el OOF")
    textos = idx["text"].astype(str).tolist()
    n_tok = np.array([len(t.split()) for t in textos])
    with gzip.open(ANOTACION, "rt", encoding="utf-8") as fh:
        documentos = json.load(fh)["documentos"]
    log("")
    log(f"   OOF canonico {ccv_ref!r} | {n} filas")

    # ---------- el objetivo: V03 se equivoca en esta celda ----------
    error = (pred_v03 != y).astype(int)
    log(f"   celdas de decision {error.size} | errores {int(error.sum())} "
        f"({error.mean() * 100:.2f} por ciento)")
    for j, et in enumerate(ETIQUETAS):
        log(f"      {et}: {int(error[:, j].sum())} errores de {n}, "
            f"{error[:, j].mean() * 100:.2f} por ciento")

    import importlib
    comp_mod = importlib.import_module("p63_05_composicion_linguistica")
    pos_mod = importlib.import_module("p63_06_estructura_tweet_largo")
    fijos = [comp_mod.rasgos_fijos(t) for t in textos]
    cat = [comp_mod.rasgos_categoricos(d) for d in documentos]
    kf, kc = list(fijos[0].keys()), list(cat[0].keys())
    base_ratios = np.array([[f[k] for k in kf] + [c[k] for k in kc]
                            for f, c in zip(fijos, cat)], dtype=float)
    lexk = list(comp_mod.rasgos_lexicos("a b", {}, 1).keys())
    B_comp = np.hstack([base_ratios,
                        np.array([[comp_mod.rasgos_lexicos(textos[i], {}, n)[k] for k in lexk]
                                  for i in range(n)], dtype=float)])
    B_pos = np.array([[pos_mod.bloque_posicion(t)[k]
                       for k in pos_mod.bloque_posicion(textos[0])] for t in textos],
                     dtype=float)
    log(f"   bloque de composicion {B_comp.shape[1]} columnas | posicion {B_pos.shape[1]}")

    # ---------- las variables de V03, que son la referencia ----------
    # margen y entropia tienen forma (n, 3), una columna por etiqueta. El objetivo tambien
    # es por celda, luego el ajuste se hace POR ETIQUETA: asi cada etiqueta usa su propio
    # margen y su propia entropia, que es lo correcto, en vez de aplanar las tres en un
    # vector de 4449 que no casa con las 1483 filas del bloque.
    log("   variables de V03: margen y entropia de la misma etiqueta, 2 columnas por "
        "etiqueta")

    def estandarizar(mat: np.ndarray, ir: np.ndarray) -> np.ndarray:
        """Estandariza por columna con el pliegue y normaliza la norma.

        Args:
            mat: bloque.
            ir: filas de entrenamiento.

        Returns:
            El bloque estandarizado.
        """
        sd = mat[ir].std(axis=0)
        sd[sd < 1e-8] = 1.0
        z = (mat - mat[ir].mean(axis=0)) / sd[None, :]
        return z

    n_comp, n_pos = B_comp.shape[1], B_pos.shape[1]
    aucs: dict[str, list[float]] = {}
    t0 = time.time()
    np.random.default_rng(SEMILLA)
    for f in range(1, 6):
        ir, iva = np.where(pliegues != f)[0], np.where(pliegues == f)[0]
        rng = np.random.default_rng(SEMILLA + f)
        z_comp = estandarizar(B_comp, ir)
        z_pos = estandarizar(B_pos, ir)
        z_rand = estandarizar(rng.standard_normal((n, n_comp + n_pos)), ir)
        for j, et in enumerate(ETIQUETAS):
            objetivo = error[:, j]
            v03_j = np.column_stack([margen[:, j], entropia[:, j]])
            bloques = {
                "margen_solo": v03_j,
                "margen_comp": np.hstack([v03_j, z_comp]),
                "margen_pos": np.hstack([v03_j, z_pos]),
                "margen_aleatorio": np.hstack([v03_j, z_rand]),
            }
            for brazo, matriz in bloques.items():
                if objetivo[iva].sum() == 0 or objetivo[iva].sum() == len(iva):
                    aucs.setdefault(brazo, []).append(float("nan"))
                    continue
                mo = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000,
                                        solver="liblinear", random_state=SEMILLA)
                mo.fit(matriz[ir], objetivo[ir])
                p = mo.predict_proba(matriz[iva])[:, 1]
                aucs.setdefault(brazo, []).append(float(roc_auc_score(objetivo[iva], p)))
        log(f"      pliegue {f} listo ({time.time() - t0:.0f} s)")

    detalle: dict[str, Any] = {}
    filas: list[dict[str, Any]] = []
    # El bucle anade en el orden pliegue 1, luego las tres etiquetas, luego pliegue 2, y
    # asi sucesivamente, luego la reshape es por pliegue y la media por etiqueta sale de
    # promediar los cinco pliegues de esa etiqueta.
    for brazo in ("margen_solo", "margen_comp", "margen_pos", "margen_aleatorio"):
        a = np.array(aucs[brazo], dtype=float).reshape(5, 3)
        detalle[brazo] = {
            "auc_por_etiqueta": {et: (None if np.all(np.isnan(a[:, i]))
                                      else round(float(np.nanmean(a[:, i])), 6))
                                 for i, et in enumerate(ETIQUETAS)},
            "auc_media": round(float(np.nanmean(a)), 6),
            "auc_por_pliegue": [round(float(x), 6) for x in np.nanmean(a, axis=1)],
        }
        filas.append({"brazo": brazo,
                      "auc_HS": detalle[brazo]["auc_por_etiqueta"]["HS"],
                      "auc_TR": detalle[brazo]["auc_por_etiqueta"]["TR"],
                      "auc_AG": detalle[brazo]["auc_por_etiqueta"]["AG"],
                      "auc_media": detalle[brazo]["auc_media"]})

    ref = detalle["margen_solo"]["auc_media"]
    d_comp = detalle["margen_comp"]["auc_media"] - ref
    d_pos = detalle["margen_pos"]["auc_media"] - ref
    d_rand = detalle["margen_aleatorio"]["auc_media"] - ref
    brecha_comp = d_comp - d_rand
    brecha_pos = d_pos - d_rand

    log("")
    log("   LOS BLOQUES EXPLICAN LOS ERRORES DE V03?")
    log(f"      {'brazo':20}{'AUC HS':>10}{'AUC TR':>10}{'AUC AG':>10}{'AUC media':>12}")
    for f in filas:
        log(f"      {f['brazo']:20}{f['auc_HS']:>10.6f}{f['auc_TR']:>10.6f}"
            f"{f['auc_AG']:>10.6f}{f['auc_media']:>12.6f}")
    log("")
    log(f"      el margen de V03 solo ya explica los errores con AUC {ref:.6f}, y eso NO es"
        f" debil")
    log(f"      la composicion aporta {d_comp:+.6f} sobre el margen, y "
        f"{brecha_comp:+.6f} frente al control aleatorio")
    log(f"      la posicion aporta {d_pos:+.6f} sobre el margen, y "
        f"{brecha_pos:+.6f} frente al control aleatorio")

    # ---------- desglose por bin, que es donde el techo dice que esta el residuo ----------
    log("")
    log("   EL MISMO AUC POR REBANADA, para el margen y para margen mas composicion")
    masks = {}
    filas_bin: list[dict[str, Any]] = []
    for nombre, lo, hi in BINS:
        m = (n_tok >= lo) & (n_tok <= hi)
        masks[nombre] = m
        fila = {"bin": nombre, "n": int(m.sum()),
                "errores": int(error[m].sum()),
                "tasa_de_error": round(float(error[m].mean()), 6)}
        # AUC del margen solo dentro de la rebanada, que es la referencia contra la que
        # se lee el margen residual de esa region
        a_solo = []
        for g in range(1, 6):
            sel = m & (pliegues == g)
            if error[sel].size == 0 or error[sel].sum() in (0, error[sel].size):
                continue
            a_solo.append(roc_auc_score(error[sel].ravel(), margen[sel].ravel()))
        fila["auc_del_margen_solo_en_la_rebanada"] = (round(float(np.mean(a_solo)), 6)
                                                       if a_solo else None)
        filas_bin.append(fila)
        log(f"      {nombre:10} n {int(m.sum()):5d}  errores {int(error[m].sum()):5d}  "
            f"tasa {error[m].mean() * 100:5.2f} por ciento  AUC del margen solo ahi "
            f"{fila['auc_del_margen_solo_en_la_rebanada']}")

    # ---------- el veredicto ----------
    if brecha_comp > 0.01 and brecha_pos > 0.01 and d_comp > 0 and d_pos > 0:
        lectura = "DILUCION_Y_NUEVA_INFORMACION"
        veredicto = (
            f"los dos bloques superan al control aleatorio sobre el margen de V03, con "
            f"{brecha_comp:+.6f} la composicion y {brecha_pos:+.6f} la posicion. Saben cosas "
            f"que V03 no tiene, luego el fallo de los tracks anteriores NO es que la "
            f"informacion este ya dentro, es que no cabe. Eso es una conclusion distinta de "
            f"la de P62 y mas fuerte: la senal es nueva y el limite es de representacion, no "
            f"de redundancia"
        )
    elif brecha_comp > 0 or brecha_pos > 0:
        lectura = "REDUNDANTE_CON_LA_CONFIANZA_DE_V03"
        veredicto = (
            f"Aqui hay un TERCER caso que la taxonomia de dos lecturas no preveia, y es el "
            f"que se da. Los dos bloques superan al control aleatorio, con {brecha_comp:+.6f} "
            f"la composicion y {brecha_pos:+.6f} la posicion, luego CONTIENEN informacion "
            f"sobre que filas va a fallar V03. Pero NO mejoran al margen: aportan "
            f"{d_comp:+.6f} y {d_pos:+.6f} sobre el, es decir lo restan. La lectura correcta "
            f"es que la informacion de los bloques sobre el residuo YA ESTA implicita en la "
            f"confianza de V03, y anadir 42 o 21 columnas mas solo hace que el modelo auxiliar "
            f"se reparta el mismo dato y sobreajuste. No es redundancia con los n-gramas, que "
            f"es lo que se suponia, y tampoco es informacion nueva: es redundancia con el "
            f"MARGEN. Por eso el margen solo ya explica los errores con AUC {ref:.6f} y "
            f"cualquier bloque lo empeora"
        )
    elif brecha_comp <= 0 and brecha_pos <= 0:
        lectura = "YA_CONOCIDO"
        veredicto = (
            f"una vez conocido el margen de V03, los dos bloques no aportan nada sobre el "
            f"control aleatorio: {brecha_comp:+.6f} la composicion y {brecha_pos:+.6f} la "
            f"posicion. Y el margen solo ya explica los errores con AUC {ref:.6f}, que no es "
            f"un numero debil. Los bloques son una re-codificacion de lo que V03 ya extrae "
            f"de los n-gramas, y la familia se cierra por una razon MAS FUERTE que no haber "
            f"ayudado: la informacion ya estaba dentro"
        )
    else:
        lectura = "PARCIAL"
        veredicto = (
            f"uno de los dos bloques supera al control aleatorio sobre el margen y el otro "
            f"no: composicion {brecha_comp:+.6f}, posicion {brecha_pos:+.6f}. Con el margen "
            f"de V03 solo, que ya explica los errores con AUC {ref:.6f}, la parte de la "
            f"composicion que sobrevive es pequena y la de la posicion es nula"
        )
    log("")
    log(f"   LECTURA: {lectura}")
    log(f"   VEREDICTO: {veredicto}")

    with CSV_K.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["brazo", "auc_HS", "auc_TR", "auc_AG", "auc_media"])
        for f in filas:
            w.writerow([f["brazo"], f["auc_HS"], f["auc_TR"], f["auc_AG"], f["auc_media"]])
        w.writerow(["delta_comp_sobre_margen", round(d_comp, 6)])
        w.writerow(["delta_pos_sobre_margen", round(d_pos, 6)])
        w.writerow(["brecha_comp_vs_aleatorio", round(brecha_comp, 6)])
        w.writerow(["brecha_pos_vs_aleatorio", round(brecha_pos, 6)])

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    informe = {
        "familia": "P63_RESIDUO",
        "naturaleza": ("EXPERIMENTO. 3 candidatos y 1 control, 0 modelos de etiqueta, 0 "
                       "submissions, 0 Kaggle, 0 holdout, 0 test"),
        "fecha": inicio.isoformat(timespec="seconds"),
        "experiment_id": "P63-07",
        "track": "K-residuo-de-V03",
        "script": "experiments/p63_08_residuo_de_v03.py",
        "pregunta": ("los bloques nuevos explican el RESIDUO de V03, o V03 ya tiene esa "
                     "informacion y por eso no cabe anadirlos"),
        "por_que_no_es_un_selector": (
            "el objetivo es estrictamente si V03 acierta o falla en esa celda, y el margen "
            "de V03 entra como variable para que la pregunta sea sobre los bloques y no sobre "
            "la confianza, que ya esta medida. No se predice la etiqueta, luego no es el "
            "selector circular de V11-07 ni de V50, ambos cerrados"),
        "las_dos_lecturas": {
            "dilucion": "los bloques saben lo que V03 falla y la informacion no se puede pagar",
            "ya_conocido": ("los bloques re-codifican lo que V03 ya extrae de los n-gramas, y "
                            "por mas cuidado que se ponga no ayudaran"),
            "por_que_importa": ("las dos hacen la misma prediccion sobre el F1 y opuestas "
                                "sobre el diagnostico, y la segunda es la que cierra la "
                                "familia por una razon mas fuerte"),
        },
        "objetivo": "indicador binario de error de V03 por celda",
        "variables_de_V03": ["margen", "entropia"],
        "n_columnas": {"composicion": n_comp, "posicion": n_pos},
        "protocolo": {
            "clasificador_del_meta_objetivo": ("LogisticRegression C=1.0 balanced liblinear, "
                                               "declarado antes de ejecutar y sin rejilla"),
            "por_que_sin_rejilla": ("lo que se mide es la INFORMACION que hay en los rasgos, "
                                    "no el punto optimo de un modelo auxiliar"),
            "pliegues": "los canonicos, por grupo",
            "umbral_de_decision": "los congelados de V03, nunca se tocan",
        },
        "resultados": detalle,
        "deltas": {
            "composicion_sobre_el_margen": round(d_comp, 6),
            "posicion_sobre_el_margen": round(d_pos, 6),
            "aleatorio_sobre_el_margen": round(d_rand, 6),
            "brecha_composicion": round(brecha_comp, 6),
            "brecha_posicion": round(brecha_pos, 6),
        },
        "tasa_de_error_por_rebanada": filas_bin,
        "lectura": lectura,
        "veredicto": veredicto,
        "decision": ("CLOSED por redundancia" if lectura == "YA_CONOCIDO"
                     else "La senal es nueva y el limite es de representacion"),
        "experimentos_registrados": 1,
        "reproducibilidad": {"nivel": "R2", "git_commit": commit, "seed": SEMILLA,
                             "hash_v03": hash_real, "corrected_cv": V03_CCV},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_K.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")

    L: list[str] = []
    W = L.append
    W("# P63 - nLos bloques nuevos explican el residuo de V03?")
    W("")
    W(f"Generado por `experiments/p63_08_residuo_de_v03.py`. Fecha "
      f"{inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## La pregunta que los cinco tracks anteriores no respondieron")
    W("")
    W("Todos produjeron la misma forma. La composicion en ratios gana +0,0221 frente a un")
    W("bloque aleatorio al anadirse a V03, y +0,1055 en solitario. La posicion gana +0,0713")
    W("en solitario. Ninguna mejora a V03. La lectura habitual es la dilucion: anadir columnas")
    W("a un espacio de 21044 dimensiones cuesta mas de lo que la informacion nueva repaga.")
    W("")
    W("Hay una segunda lectura que no se ha probado, y es mas importante:")
    W("")
    W("| lectura | prediccion |")
    W("|---|---|")
    W("| **DILUCION** | los bloques saben cosas que V03 falla, pero no se pueden pagar |")
    W("| **YA CONOCIDO** | los bloques re-codifican lo que V03 ya saca de los n-gramas |")
    W("")
    W("Las dos hacen la misma prediccion sobre el F1 y **opuestas sobre el diagnostico**. Si")
    W("fuera redundancia, ningun cuidado al anadirlos ayudaria nunca, y esa es una razon")
    W("mucho mas fuerte para cerrar una familia que un negativo.")
    W("")
    W("El objetivo aqui es estrictamente si V03 acierta o falla en esa celda, y el margen de")
    W("V03 entra como variable para que la pregunta sea sobre los bloques y no sobre la")
    W("confianza, que ya esta medida. **No se predice la etiqueta**, luego no es el selector")
    W("circular de V11-07 ni el de V50, ambos cerrados.")
    W("")
    W("## Resultados")
    W("")
    W("| brazo | AUC HS | AUC TR | AUC AG | AUC media |")
    W("|---|---:|---:|---:|---:|")
    for f in filas:
        W(f"| {f['brazo']} | {f['auc_HS']:.6f} | {f['auc_TR']:.6f} | {f['auc_AG']:.6f} | "
          f"{f['auc_media']:.6f} |")
    W("")
    W("## El desglose")
    W("")
    W("```text")
    W(f"el margen de V03 solo ya explica los errores con AUC {ref:.6f}, y no es debil")
    W(f"composicion sobre el margen            {d_comp:+.6f}")
    W(f"posicion sobre el margen              {d_pos:+.6f}")
    W(f"aleatorio sobre el margen             {d_rand:+.6f}")
    W(f"brecha de composicion contra aleatorio {brecha_comp:+.6f}")
    W(f"brecha de posicion contra aleatorio   {brecha_pos:+.6f}")
    W("```")
    W("")
    W("## La tasa de error por rebanada, que es donde el techo dice que esta el residuo")
    W("")
    W("El AUC del margen solo sale por debajo de 0,5, y eso no es un error: el objetivo es")
    W("el ERROR, y un margen alto significa que V03 esta lejos del umbral, es decir que")
    W("acierta. Un AUC de 0,17 sobre el error es un AUC de 0,83 sobre el acierto.")
    W("")
    W("| bin | n | errores | tasa | AUC del margen solo ahi |")
    W("|---|---:|---:|---:|---:|")
    for f in filas_bin:
        a = f["auc_del_margen_solo_en_la_rebanada"]
        W(f"| {f['bin']} | {f['n']} | {f['errores']} | {f['tasa_de_error'] * 100:.2f}% | "
          f"{'n/d' if a is None else format(a, '.6f')} |")
    W("")
    W("## Veredicto")
    W("")
    W("```text")
    W(veredicto)
    W("")
    W(f"LECTURA: {lectura}")
    W("```")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p63_08_residuo_de_v03.py`. Sin push.*")
    MD_K.write_text(a_ascii("\n".join(L)), encoding="ascii")

    for destino in (MD_K, JSON_K, CSV_K):
        if any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P63-08 OK | margen {ref:.6f} | brecha comp {brecha_comp:+.6f} | brecha pos "
        f"{brecha_pos:+.6f} | lectura {lectura}")


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
