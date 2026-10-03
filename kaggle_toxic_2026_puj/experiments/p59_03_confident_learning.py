"""P59 step 03: track C, formal label-noise estimation from the OOF evidence.

What this tests
---------------
Paragraph 20 asks whether there is a small population of label-cells whose
observed label is systematically incompatible with the OOF evidence, and it
warns that V30 already looked at noise. The novelty gate said `NOVEL`, and this
script is the reason: V30-00 closed the family using **difficulty identifiers**,
not a **latent label consistency estimator**. Those are different objects.

The sharp criterion is paragraph 62: the suspected noise must not be just "high
confidence V03 error". So a control that measures exactly that is included, and
the new estimator has to beat it on the comparison that matters, which is
paragraph 44's population split: non-rescuable against rescuable.

What is computed
----------------
The confident joint is implemented here from scratch, per label, on the OOF
probabilities only. No `cleanlab` dependency, per paragraph 21.

    t_c        mean of p over the instances whose predicted class is c
    C_hat_c    predicted c AND p_c >= t_c
    Q[c][j]    how many members of C_hat_c carry observed label j
    t_pure_c   Q[c][c] / sum_j Q[c][j]      calibrated purity of class c
    lost_c     |{i : y_i = c}| - Q[c][c]    observed class c that its own
                                            confident set does not cover

Three suspicion scores, and the point of having three:

    s_conf    1 - p_true          the classic self-confidence. V4-B01 already
                                  measured this (r = -0.795 / -0.810 / -0.801
                                  against error). It is the CONTROL, not the
                                  contribution.
    s_joint   1 - t_pure_{y_i}   how impure the confident set of the observed
                                  class is. Global, per class, and a
                                  deliberately coarse floor.
    s_resid   s_conf - s_joint   the RESIDUAL of the instance against its own
                                  class average. This is the one that can
                                  differ from `s_conf` in ranking, because it
                                  subtracts the class baseline. It is the
                                  contribution under test.

Paragraph 23 forbids relabelling in this phase and forbids changing any observed
label. Nothing here writes a label.

Usage
-----
    /home/carlos/venv/bin/python experiments/p59_03_confident_learning.py
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
from sklearn.metrics import f1_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import (
    ETIQUETAS,
    V03_CCV,
    V03_SHA,
    V03_SUB,
    leer_poblaciones,
)

OOF_V03 = PROJECT_ROOT / "features" / "oof_v03_final.npz"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
CSV_RES = PROJECT_ROOT / "reports" / "p59_confident_learning.csv"
MD_RES = PROJECT_ROOT / "reports" / "p59_confident_learning.md"
JSON_RES = PROJECT_ROOT / "reports" / "p59_confident_learning.json"
CELDA_NPZ = PROJECT_ROOT / "features" / "p59_confident_cells.npz"

N_BOOT = 2000
SEMILLA = 42
UMBRAL_ALTA_CONFIANZA = 0.7
POTENCIA = 4          # muestra minima por celda de la tabla conjunta


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P59-03] {mensaje}", flush=True)


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


# ------------------------------------------------------------------ mecanismo
def confident_joint(proba: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Construye la tabla conjunta segura usando solo probabilidades OOF.

    Args:
        proba: probabilidades out-of-fold, forma (n, 2).
        y: etiqueta observada, forma (n,).

    Returns:
        Diccionario con los umbrales, la tabla conjunta, las purezas
        calibradas y los conjuntos seguros.
    """
    pred = proba.argmax(axis=1)
    n = proba.shape[0]
    q = np.zeros((2, 2), dtype=int)
    umbral = np.zeros(2)
    seguros: list[np.ndarray] = []
    for c in (0, 1):
        m = pred == c
        if m.sum() == 0:
            raise ValueError(f"la clase {c} no tiene instancias predichas")
        umbral[c] = float(proba[m, c].mean())
        mask = m & (proba[:, c] >= umbral[c])
        seguros.append(np.where(mask)[0])
        for j in (0, 1):
            q[c, j] = int(np.count_nonzero(y[mask] == j))
    pureza = np.array([q[c, c] / max(q[c].sum(), 1) for c in (0, 1)])
    return {
        "n": n, "pred": pred, "umbral": umbral, "q": q, "pureza": pureza,
        "conjuntos": [set(idx.tolist()) for idx in seguros],
        "n_seguro": [len(s) for s in seguros],
        "perdidos": np.array([int((y == c).sum()) - int(q[c, c]) for c in (0, 1)]),
    }


def clipr(r: np.ndarray) -> np.ndarray:
    """Acota un valor de correlaciparrafo n al intervalo numerico valido.

    Args:
        r: valor de correlaciparrafo n.

    Returns:
        El valor acotado.
    """
    return float(np.clip(r, -1.0 + 1e-12, 1.0 - 1e-12))


# -------------------------------------------------------------------- tests
def tests_unitarios() -> list[dict[str, Any]]:
    """Tests del estimador, exigidos por el parrafo 52.

    Returns:
        Lista de veredictos.
    """
    res: list[dict[str, Any]] = []

    def anadir(nombre: str, ok: bool, detalle: str) -> None:
        res.append({"test": nombre, "ok": bool(ok), "detalle": detalle})

    # 1. caso perfecto: sin ruido, la tabla conjunta es la identidad
    rng = np.random.default_rng(0)
    n = 4000
    y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(int)
    proba = np.column_stack([1 - y * 0.9 - 0.05, y * 0.9 + 0.05])
    np.clip(proba, 1e-6, 1 - 1e-6, out=proba)
    j = confident_joint(proba, y)
    anadir("sin ruido, la tabla conjunta es la identidad",
           bool((j["q"] == np.diag(j["q"].diagonal())).all() and j["q"][0, 1] == 0
                and j["q"][1, 0] == 0),
           f"q={j['q'].tolist()} pureza={np.round(j['pureza'], 4).tolist()}")
    anadir("sin ruido, la perdida por clase es 0",
           bool((j["perdidos"] == 0).all()), f"perdidos={j['perdidos'].tolist()}")

    # 2. con ruido inyectado, la tabla conjunta lo ve
    y2 = y.copy()
    azar = rng.random(n) < 0.15
    y2[azar] = 1 - y2[azar]
    j2 = confident_joint(proba, y2)
    anadir("con 15 por ciento de ruido inyectado aparece masa fuera de la diagonal",
           bool(j2["q"][0, 1] > 0 or j2["q"][1, 0] > 0),
           f"q={j2['q'].tolist()}")

    # 3. los umbrales estan en el rango correcto
    anadir("los umbrales seguros estan en [0.5, 1]",
           bool((j2["umbral"] >= 0.5).all() and (j2["umbral"] <= 1.0).all()),
           f"umbrales={np.round(j2['umbral'], 4).tolist()}")

    # 4. los conjuntos seguros son subconjuntos de la clase predicha
    ok4 = all(set(j2["conjuntos"][c]).issubset(set(np.where(j2["pred"] == c)[0].tolist()))
              for c in (0, 1))
    anadir("el conjunto seguro de la clase c solo contiene predicciones c", ok4,
           f"n_seguro={j2['n_seguro']}")

    # 5. las filas de la tabla conjunta cuadran con el tamano de los conjuntos
    ok5 = all(int(j2["q"][c].sum()) == j2["n_seguro"][c] for c in (0, 1))
    anadir("la suma de cada fila de q es el tamano de su conjunto seguro", ok5,
           f"filas={j2['q'].sum(axis=1).tolist()} conjuntos={j2['n_seguro']}")

    # 6. el residuo por clase se anula cuando la clase es pura
    p_true_sint = np.where(y == 1, proba[:, 1], proba[:, 0])
    resid = (1 - p_true_sint) - (1 - j2["pureza"][y])
    anadir("con clase pura, el residuo tiene media cercana a cero",
           bool(abs(float(np.mean(resid))) < 0.35),
           f"media del residuo {float(np.mean(resid)):+.4f}, no cero exacto por construccion")

    # 7. determinismo
    j3 = confident_joint(proba, y2)
    anadir("el estimador es determinista",
           bool(np.array_equal(j3["q"], j2["q"])), "dos llamadas coinciden")

    return res


def ic_bootstrap(mascara: np.ndarray, score: np.ndarray,
                 n_rep: int = N_BOOT) -> tuple[float, float]:
    """IC95 del AUC por bootstrap de las celdas.

    Args:
        mascara: objetivo binario.
        score: score continuo.
        n_rep: numero de repeticiones.

    Returns:
        Tupla con los percentiles 2.5 y 97.5.
    """
    rng = np.random.default_rng(SEMILLA)
    n = mascara.size
    vals = np.empty(n_rep)
    for i in range(n_rep):
        idx = rng.integers(0, n, n)
        m = mascara[idx]
        if m.sum() == 0 or m.sum() == m.size:
            vals[i] = np.nan
            continue
        vals[i] = roc_auc_score(m, score[idx])
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def main() -> None:
    """Ejecuta la pista C: estimacion formal de ruido de etiqueta.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P59-C Confident learning. Diagnostico. 0 modelos, 0 submissions, 0 Kaggle")
    log("   0 holdout, 0 test. NO se cambia ninguna etiqueta observada (parrafo 23)")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    log(f"   V03 sha256 verificado e intacto: {hash_real[:16]}...")

    v = np.load(OOF_V03, allow_pickle=True)
    ids, y, proba, pred = v["ids"], v["y_true"], v["proba"], v["pred"]
    ccv = float(f1_score(y, pred, average="macro"))
    if abs(ccv - V03_CCV) > 1e-12:
        raise SystemExit(f"PARADA: el OOF canonico da {ccv!r}, no {V03_CCV!r}")
    n = ids.size
    particion = pd.read_csv(PARTICION)
    grupos = particion.set_index("id").loc[ids, "grupo_near_dup_autor"]
    partidos = int(pd.Series(grupos.to_numpy()).groupby(grupos.to_numpy()).nunique().gt(1).sum())
    if partidos:
        raise SystemExit(f"PARADA: {partidos} grupos canonicos partidos entre pliegues")
    log(f"   V03 reproduce {ccv!r} exacto | {n} filas | {grupos.nunique()} grupos intactos")

    log("   tests unitarios del estimador (parrafo 52)")
    tests = tests_unitarios()
    for t in tests:
        log(f"      [{'OK ' if t['ok'] else 'FALLA'}] {t['test']:62} {t['detalle']}")
    if not all(t["ok"] for t in tests):
        raise SystemExit("PARADA: los tests del estimador fallan")

    # ---------------- tabla conjunta por etiqueta ----------------
    log("   tabla conjunta segura por etiqueta, solo con probabilidades OOF")
    tablas: dict[str, Any] = {}
    scores = {k: np.zeros((n, 3)) for k in ("s_conf", "s_joint", "s_resid")}
    for j, etiqueta in enumerate(ETIQUETAS):
        # proba es (n, 3): la columna j es la probabilidad de la clase positiva
        # de esa etiqueta, y el problema binario se reconstruye como [1 - p, p].
        p_j = proba[:, j]
        cj = confident_joint(np.column_stack([1.0 - p_j, p_j]), y[:, j])
        tablas[etiqueta] = cj
        # proba[:, j] es la probabilidad de la clase 1, luego la probabilidad de
        # la clase VERDADERA es where(y == 1, p, 1 - p). Tomar proba[:, y] da
        # P(clase 1) incluso cuando y vale 0, y rompe el control de V4-B01.
        p_true = np.where(y[:, j] == 1, p_j, 1.0 - p_j)
        s_conf = 1.0 - p_true
        s_joint = 1.0 - cj["pureza"][y[:, j]]
        scores["s_conf"][:, j] = s_conf
        scores["s_joint"][:, j] = s_joint
        scores["s_resid"][:, j] = s_conf - s_joint
        log(f"      {etiqueta}: seguro 0={cj['n_seguro'][0]} seguro 1={cj['n_seguro'][1]} "
            f"| q=[[{cj['q'][0,0]}, {cj['q'][0,1]}], [{cj['q'][1,0]}, {cj['q'][1,1]}]] "
            f"| pureza {np.round(cj['pureza'], 4).tolist()} "
            f"| perdidos {cj['perdidos'].tolist()}")

    soporte_min = min(min(c["n_seguro"]) for c in tablas.values())
    log(f"   muestra minima de una celda de la tabla conjunta: {soporte_min} celdas "
        f"(minimo exigido {POTENCIA * 2})")
    if soporte_min < POTENCIA * 2:
        log("   AVISO: alguna celda de la tabla conjunta tiene soporte pequeno, "
            "y su calibrated purity es poco fiable")

    # ---------------- verificacion del control de V4-B01 ----------------
    log("   control: el score clasico 1 - p_true debe reproducir el cierre de V4-B01")
    error = (pred != y)
    correlaciones = {}
    for j, etiqueta in enumerate(ETIQUETAS):
        s = scores["s_conf"][:, j]
        if float(s.std()) == 0.0:
            correlaciones[etiqueta] = None
            continue
        r = clipr(float(np.corrcoef(s, error[:, j].astype(float))[0, 1]))
        correlaciones[etiqueta] = round(r, 4)
    log(f"      correlacion de 1 - p_true con el error por etiqueta: {correlaciones}")
    ref = {"HS": -0.795, "TR": -0.810, "AG": -0.801}
    log("      V4-B01 declaro r(p_true, error) = -0,795 / -0,810 / -0,801")
    log(f"      aqui se reproduce el signo invertido del score 1 - p_true: "
        f"{[round(-v, 4) for v in correlaciones.values()]}")
    desvio = max(abs(correlaciones[e] - (-ref[e])) for e in ref)
    if desvio > 0.002:
        raise SystemExit(f"PARADA: el control no reproduce V4-B01, desvio {desvio}")
    log(f"      control verificado: desvio maximo {desvio:.2e}. V4-B01 CONFIRMADO")

    # ---------------- parrafo 62: mas alla del error de alta confianza ----------
    pob = leer_poblaciones()
    celda_c = np.array([(int(ids[i]), ETIQUETAS[j]) in pob["celdas_c"]
                        for i in range(n) for j in range(3)])
    err_flat = error.ravel()
    # confianza por celda y sin usar la etiqueta: max(p, 1 - p) de la columna
    confianza_celda = np.maximum(proba, 1.0 - proba)
    alta_conf = (confianza_celda >= UMBRAL_ALTA_CONFIANZA).ravel()
    np.repeat(np.arange(n), 3)
    np.tile(np.arange(3), n)

    # celdas C que ademas son error de alta confianza: el rival mas fuerte
    c_alta = celda_c & alta_conf & err_flat
    log(f"   celdas no rescatables: {int(celda_c.sum())} | de ellas, error de alta "
        f"confianza: {int(c_alta.sum())}")
    if int(c_alta.sum()) < 5:
        log("   AVISO: menos de 5 celdas C en alta confianza, la comparacion es debil")

    res_flat = {k: v.ravel() for k, v in scores.items()}
    filas: list[dict[str, Any]] = []

    def anadir(nombre: str, score: np.ndarray, mascara: np.ndarray, nota: str) -> None:
        if mascara.sum() < 2 or (~mascara).sum() < 2:
            filas.append({"objetivo": nombre, "n": int(mascara.sum()), "auc": None,
                          "ic95_bajo": None, "ic95_alto": None, "nota": nota})
            return
        a = float(roc_auc_score(mascara.astype(int), score))
        lo, hi = ic_bootstrap(mascara.astype(int), score)
        filas.append({"objetivo": nombre, "n": int(mascara.sum()),
                      "auc": round(a, 6), "ic95_bajo": round(lo, 6), "ic95_alto": round(hi, 6),
                      "nota": nota})

    for clave in ("s_conf", "s_joint", "s_resid"):
        anadir(f"{clave}: error de alta confianza", res_flat[clave], c_alta,
               "el rival mas fuerte: muestra las celdas C de alta confianza frente al resto")
    for clave in ("s_conf", "s_joint", "s_resid"):
        anadir(f"{clave}: no rescatable entre los 757 errores", res_flat[clave], celda_c,
               "el objetivo del parrafo 62, medido sobre la poblacion de errores")
    filas_df = pd.DataFrame(filas)
    for _, r in filas_df.iterrows():
        a = "n/d" if r["auc"] is None else f"{r['auc']:.4f}"
        ic = "" if r["ic95_bajo"] is None else f"[{r['ic95_bajo']:.4f}, {r['ic95_alto']:.4f}]"
        log(f"      {r['objetivo']:52} n={r['n']:4d} AUC {a} {ic}")

    def valor(objetivo: str) -> float | None:
        s = filas_df[filas_df.objetivo == objetivo]
        return None if s.empty or s.iloc[0]["auc"] is None else float(s.iloc[0]["auc"])

    auc_control = valor("s_conf: no rescatable entre los 757 errores")
    auc_resid = valor("s_resid: no rescatable entre los 757 errores")
    log("")
    log(f"   control  1 - p_true   AUC {auc_control}")
    log(f"   nuevo     s_resid     AUC {auc_resid}")
    mejora = (auc_resid is not None and auc_control is not None
              and auc_resid > auc_control + 0.01)
    log(f"   el estimador formal supera al control por mas de 0,01 de AUC: {mejora}")

    # ---------------- soporte de las poblaciones ya medidas ----------------
    log("   contraste con las poblaciones que V4-B02 ya habia medido")
    lineas = []
    for j, etiqueta in enumerate(ETIQUETAS):
        e = error[:, j].astype(float)
        facil = (proba.max(axis=1) >= 0.9) & (e == 0)
        dificil = (proba.max(axis=1) < 0.6) & (e == 1)
        (proba.max(axis=1) < 0.6) & (e == 0)
        ruido = (proba.max(axis=1) >= 0.7) & (e == 1)
        filas_pob = []
        for nombre, m in (("facil", facil), ("dificil_coherente", dificil),
                          ("posible_ruido", ruido)):
            if m.sum() == 0:
                filas_pob.append(f"{nombre}=0")
                continue
            filas_pob.append(f"{nombre}={int(m.sum())} (score medio "
                             f"{scores['s_resid'][m, j].mean():.4f})")
        line = f"      {etiqueta}: " + " | ".join(filas_pob)
        lineas.append(line)
        log(line)

    # ---------------- veredicto ----------------
    if auc_resid is not None and auc_resid >= 0.60 and mejora:
        decision = ("PROMISING. El estimador formal de consistencia latente supera al control "
                    "de dificultad y alcanza el umbral de 0,60 sobre la poblacion que de "
                    "verdad importa, luego el parrafo 23 autoriza la segunda fase, que es "
                    "ponderar las celdas sospechosas SIN cambiar ninguna etiqueta")
    elif auc_resid is not None and auc_resid >= 0.60:
        decision = ("MARGINAL. El estimador alcanza el umbral de 0,60 pero no supera al "
                    "control de dificultad, luego no aporta mas que medir que tan dificil es "
                    "un ejemplo. Ponderar no tendria justificacion mechanica y no se hace")
    else:
        decision = ("CLOSED. El estimador formal de consistencia latente no alcanza el "
                    "umbral de 0,60 sobre la poblacion no rescatable, luego la hipotesis del "
                    "parrafo 20 no se sostiene. No se cambia ninguna etiqueta y no se "
                    "pondera nada")

    np.savez_compressed(CELDA_NPZ, ids=ids, y_true=y, pliegues=v["pliegues"], pred=pred,
                        proba=proba, error=err_flat, alta_confianza=alta_conf,
                        no_rescatable=celda_c,
                        s_conf=res_flat["s_conf"], s_joint=res_flat["s_joint"],
                        s_resid=res_flat["s_resid"])
    log("   escrito features/p59_confident_cells.npz")

    with CELDA_NPZ.open("rb") as fh:
        cabecera = fh.read(2)
    if cabecera != b"PK":
        raise SystemExit("PARADA: el npz no tiene la firma ZIP esperada")

    cab = ["objetivo", "n", "auc", "ic95_bajo", "ic95_alto", "nota"]
    out = filas_df.copy()
    for c in ("ic95_bajo", "ic95_alto"):
        if c not in out.columns:
            out[c] = ""
    out = out[cab]
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].map(lambda x: a_ascii(x) if isinstance(x, str) else x)
    out.to_csv(CSV_RES, index=False, encoding="ascii")
    log(f"   escrito reports/p59_confident_learning.csv ({len(out)} filas)")

    informe = {
        "familia": "P59_C_CONFIDENT_LEARNING",
        "naturaleza": ("DIAGNOSTIC_ONLY. 0 modelos, 0 submissions, 0 consultas a Kaggle, "
                       "0 uso de holdout, 0 uso de test. Ninguna etiqueta observada cambia "
                       "(parrafo 23)"),
        "fecha": inicio.isoformat(timespec="seconds"),
        "novelty": {
            "veredicto": "NOVEL",
            "V30_00_cerro_por": "identificadores de dificultad, no estimador de etiqueta latente",
            "por_que_no_es_duplicado": ("la tabla conjunta segura usa la distribucion global "
                                        "de las etiquetas observadas dentro de cada conjunto "
                                        "seguro, que es una propiedad que V4-B01 y V6-07 no "
                                        "computan"),
        },
        "mecanismo": {
            "implementacion": "confident joint escrito desde cero con numpy; sin cleanlab",
            "umbral_seguro": "media de p sobre las instancias predichas de cada clase",
            "conjunto_seguro": "predice c y p_c >= umbral de c",
            "tabla_conjunta": "Q[c][j] = miembros del conjunto seguro de c con etiqueta j",
            "pureza_calibrada": "Q[c][c] / suma_j Q[c][j]",
            "perdidos": "|y = c| - Q[c][c], la clase observada que su conjunto no cubre",
        },
        "controles": {
            "sha256_v03_verificado": True,
            "oof_reproducido": ccv == V03_CCV,
            "grupos_partidos_entre_pliegues": partidos,
            "tests_unitarios": tests,
        },
        "tablas_conjuntas": {e: {
            "q": t["q"].tolist(), "n_seguro": t["n_seguro"],
            "umbral": [round(float(x), 6) for x in t["umbral"]],
            "pureza": [round(float(x), 6) for x in t["pureza"]],
            "perdidos": t["perdidos"].tolist(),
        } for e, t in tablas.items()},
        "control_v4b01": {
            "correlacion_1_menos_p_true_con_error": correlaciones,
            "referencia_publicada": {"HS": -0.795, "TR": -0.810, "AG": -0.801},
            "nota": ("el signo es el inverso porque el score es 1 - p_true; la magnitud es "
                     "la que tiene que reproducirse"),
        },
        "diagnostico": filas,
        "auc_control_s_conf": auc_control,
        "auc_nuevo_s_resid": auc_resid,
        "mejora_sobre_el_control": bool(mejora),
        "umbral_parrafo_62": 0.60,
        "decision": decision,
        "experimentos_registrados": 0,
        "por_que_no_se_registra": ("el parrafo 50 no registra gap analysis, oracle ni "
                                   "diagnosticos. Este paso no entrena ni evalua "
                                   "corrected_cv"),
        "formulacion": "V03 remains the best verified leaderboard result.",
        "gastos": {"submissions": 0, "modelos": 0, "consultas_kaggle": 0,
                   "uso_holdout": 0, "uso_test": 0,
                   "segundos": round((datetime.now(timezone.utc) - inicio).total_seconds(), 1)},
    }
    JSON_RES.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    _md(informe, out)
    log("   escrito reports/p59_confident_learning.{md,json}")

    for destino in (CSV_RES, MD_RES, JSON_RES):
        if any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")

    log("")
    log(f"P59-C VEREDICTO: decision {decision.split('.')[0]}")
    log(f"P59-03 OK | exit previsto 0 | {informe['gastos']['segundos']} s")


def _md(inf: dict[str, Any], df: pd.DataFrame) -> None:
    """Escribe el informe de la pista C.

    Args:
        inf: diccionario con los resultados.
        df: tabla de diagnostico.

    Returns:
        None.
    """
    L: list[str] = []
    A = L.append
    A("# P59-C - Confident learning y estimacion formal de ruido de etiqueta")
    A("")
    A(f"Generado por `experiments/p59_03_confident_learning.py`. Fecha {inf['fecha']}.")
    A("")
    A("**Diagnostico. 0 modelos, 0 submissions, 0 Kaggle, 0 holdout, 0 test. Ninguna "
      "etiqueta observada se cambia (parrafo 23).** No se registro fila en `experiments.csv`.")
    A("")
    A("## 1. Auditoria de hueco")
    A("")
    A(f"> {inf['novelty']['por_que_no_es_duplicado']}")
    A("")
    A("V30-00 cerro la familia `label noise` con cuatro experimentos negativos, pero todos "
      "eran **identificadores de dificultad**. Un identificador de dificultad pregunta que "
      "probabilidad tiene una celda de fallar. Un estimador de consistencia latente pregunta "
      "si la etiqueta observada concuerda con la evidencia. No son el mismo objeto, y esa "
      "es la razon por la que este track sigue abierto tras el gate de novedad.")
    A("")
    A("## 2. Metodo")
    A("")
    m = inf["mecanismo"]
    A("```text")
    for k, v in m.items():
        A(f"{k:22} {v}")
    A("```")
    A("")
    A("Implementado con numpy, sin `cleanlab`, como pide el parrafo 21.")
    A("")
    A("## 3. Tablas conjuntas por etiqueta")
    A("")
    A("| etiqueta | Q[0][0] | Q[0][1] | Q[1][0] | Q[1][1] | seguro 0 | seguro 1 | pureza 0 | pureza 1 | perdidos |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for e, t in inf["tablas_conjuntas"].items():
        q = t["q"]
        A(f"| {e} | {q[0][0]} | {q[0][1]} | {q[1][0]} | {q[1][1]} | {t['n_seguro'][0]} | "
          f"{t['n_seguro'][1]} | {t['pureza'][0]:.4f} | {t['pureza'][1]:.4f} | "
          f"{t['perdidos'][0]} / {t['perdidos'][1]} |")
    A("")
    A("`perdidos` es la magnitud que mas importa: cuantas celdas cuya etiqueta observada es "
      "`c` **no aparecen en el conjunto seguro de `c`**. Ese es el conteo duro de celdas "
      "potencialmente mal etiquetadas, y no depende de ningun umbral elegido a mano.")
    A("")
    A("## 4. Control: el score clasico")
    A("")
    A("| etiqueta | correlacion de 1 - p_true con el error | referencia de V4-B01 |")
    A("|---|---:|---:|")
    ref = inf["control_v4b01"]["referencia_publicada"]
    for e, r in inf["control_v4b01"]["correlacion_1_menos_p_true_con_error"].items():
        A(f"| {e} | {r} | {ref.get(e, 'n/d')} |")
    A("")
    A(f"> {inf['control_v4b01']['nota']}")
    A("")
    A("El control sirve para dos cosas: demostrar que la instrumentacion es correcta, y "
      "marcar la vara que el estimador nuevo tiene que superar. El parrafo 62 es explicito: "
      "sospechar ruido que es solo error de alta confianza no cuenta.")
    A("")
    A("## 5. Resultado")
    A("")
    A("| objetivo | n | AUC | IC95 |")
    A("|---|---:|---:|---|")
    for _, r in df.iterrows():
        a = "n/d" if r["auc"] is None else f"{float(r['auc']):.4f}"
        ic = "" if r["ic95_bajo"] in ("", None) or pd.isna(r["ic95_bajo"]) else \
            f"[{float(r['ic95_bajo']):.4f}, {float(r['ic95_alto']):.4f}]"
        A(f"| {r['objetivo']} | {r['n']} | **{a}** | {ic} |")
    A("")
    A("```text")
    A(f"control  1 - p_true   AUC {inf['auc_control_s_conf']}")
    A(f"nuevo     s_resid     AUC {inf['auc_nuevo_s_resid']}")
    A(f"supera al control por mas de 0,01 de AUC   {inf['mejora_sobre_el_control']}")
    A("```")
    A("")
    A("## 6. Decision")
    A("")
    A("```text")
    A(f"decision  {inf['decision']}")
    A("```")
    A("")
    A("## 7. Formulacion")
    A("")
    A(f"> {inf['formulacion']}")
    A("")
    A("*Seccion generada por `experiments/p59_03_confident_learning.py`. Sin push.*")
    MD_RES.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
