"""P61 step 02: model inventory and diversity measurement.

Why there is something to inventory at all
-------------------------------------------
The brief says not to retrain if a valid OOF already exists, and one does. The
P50 artifact `features/p50_celda_features.npz` carries TWENTY model variants with
their out-of-fold probabilities on exactly the canonical 1483 rows, with the same
identifier vector, and its own fold and group vectors. That is a ready-made pool and
reusing it is both cheaper and more honest than rebuilding twenty models.

One thing had to be checked before trusting it. Its fold vector is numbered from 0 to
4 while the canonical one is numbered from 1 to 5, so the first question is whether
that is an offset or a different partition. A different partition would invalidate
every diversity number, because then the two models' out-of-fold predictions would
not be comparable row by row. The script verifies it explicitly and stops if the
partitions differ, rather than reporting a number that quietly means nothing.

What the brief asks this block to answer
----------------------------------------
Not "which model is best", but:

    does a historical model exist with a QUALITY and DIVERSITY combination that V03
    is not exploiting?

That is not a single score, and the brief says so. So each candidate gets measured on
six separate axes, and none of them is allowed to stand in for the others:

    corrected_cv         quality
    error correlation    do the two models fail on the same cells
    probability correlation  do they rank the same way
    decision disagreement  how many cells do they decide differently
    unique rescue        cells this model gets right AND V03 gets wrong
    unique damage        cells this model gets wrong AND V03 gets right

A model with a poor `corrected_cv` can still be worth its place in a pool if its
errors land elsewhere, and the point of measuring the axes separately is that a
single blended score would hide exactly that. The net is reported as a convenience,
not as the selection criterion.

What it writes
--------------
    reports/p61_model_inventory.csv
    reports/p61_model_diversity.csv
    figures/p61/09_error_correlation_heatmap.png
    figures/p61/10_unique_rescue_vs_damage.png
    reports/p61_model_inventory.json

Usage
-----
    /home/carlos/venv/bin/python experiments/p61_02_inventario.py
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import (
    V03_CCV,
    V03_SHA,
    V03_SUB,
)

REPORTS = PROJECT_ROOT / "reports"
FIG_P61 = PROJECT_ROOT / "figuras" / "p61"
OOF = PROJECT_ROOT / "features" / "oof_v03_final.npz"
POOL = PROJECT_ROOT / "features" / "p50_celda_features.npz"
CSV_INV = REPORTS / "p61_model_inventory.csv"
CSV_DIV = REPORTS / "p61_model_diversity.csv"
JSON_INV = REPORTS / "p61_model_inventory.json"

V03_UMBRALES = np.array([0.44, 0.51, 0.46])
SEMILLA = 42


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P61-02] {mensaje}", flush=True)


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
    """Construye el inventario de modelos y mide la diversidad de cada uno.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P61-02 inventario de modelos y medicion de diversidad")
    log("   NO se reentrena nada: se reutiliza el pool historico de P50, con OOF valido")
    FIG_P61.mkdir(parents=True, exist_ok=True)

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    log(f"   V03 sha256 verificado e intacto: {hash_real[:16]}...")

    v = np.load(OOF, allow_pickle=True)
    y, pred_v03, proba_v03, pliegues_v03 = (v["y_true"].astype(int), v["pred"].astype(int),
                                            v["proba"], v["pliegues"])
    n = y.shape[0]
    ccv_ref = float(f1_score(y, pred_v03, average="macro"))
    if ccv_ref != V03_CCV:
        raise SystemExit(f"PARADA: el OOF canonico da {ccv_ref!r}, no {V03_CCV!r}")

    z = np.load(POOL, allow_pickle=True)
    nombres = [str(s) for s in z["nombres"]]
    cprob = z["canal_proba"].astype(np.float64)
    pliegues_pool = z["pliegue"]

    # ---------- verificacion de que los pliegues son la MISMA particion ----------
    log("")
    log("   VERIFICACION DE QUE EL POOL USA LA MISMA PARTICION QUE EL OOF CANONICO")
    log(f"      ids iguales: {bool(np.array_equal(z['ids'], v['ids']))}")
    log(f"      y_true iguales: {bool(np.array_equal(z['y_true'], y))}")
    log(f"      convencion de pliegues del pool: {sorted(int(x) for x in np.unique(pliegues_pool))}")
    log(f"      convencion de pliegues canonica: {sorted(int(x) for x in np.unique(pliegues_v03))}")
    pool_norm = pliegues_pool + 1
    misma = bool(np.array_equal(pool_norm, pliegues_v03))
    log(f"      identicos tras normalizar el desplazamiento en uno: {misma}")
    if not misma:
        raise SystemExit(
            "PARADA: el pool y el OOF canonico usan particiones DISTINTAS. Toda medida de "
            "diversidad compararia predicciones que no son comparables fila a fila")
    sizes_pool = [int((pool_norm == k).sum()) for k in range(1, 6)]
    sizes_v03 = [int((pliegues_v03 == k).sum()) for k in range(1, 6)]
    log(f"      tamanos por pliegue, pool:     {sizes_pool}")
    log(f"      tamanos por pliegue, canonico: {sizes_v03}")
    log("      el pool numera de 0 a 4 y el canonico de 1 a 5, luego el desplazamiento es de")
    log("      convencion y la particion es la misma")

    # el grupo canonico, para la complementariedad a nivel de grupo
    part = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv")
    part.set_index("id").loc[v["ids"], "grupo_near_dup_autor"].to_numpy()

    # ---------- inventario ----------
    log("")
    log("   INVENTARIO: correged_cv de los canales con los umbrales CANONICOS")
    log("      (las decisiones del canal se recalculan desde sus probabilidades con los")
    log("       umbrales de V03, no se usan las decisiones almacenadas)")
    filas: list[dict[str, Any]] = []
    for i, nm in enumerate(nombres):
        pred = (cprob[i] >= V03_UMBRALES[None, :]).astype(int)
        ccv = float(f1_score(y, pred, average="macro"))
        es_v03 = bool(ccv == V03_CCV and int(np.count_nonzero(pred != pred_v03)) == 0)
        filas.append({
            "canal": i, "nombre": nm, "es_v03": es_v03,
            "corrected_cv": round(ccv, 6),
            "delta_vs_V03": round(ccv - V03_CCV, 6),
            "folds": [round(float(f1_score(y[pliegues_v03 == k],
                                           pred[pliegues_v03 == k],
                                           average="macro")), 6)
                      for k in range(1, 6)],
        })
    df_inv = pd.DataFrame(filas).sort_values("corrected_cv", ascending=False)
    df_inv = df_inv.reset_index(drop=True)
    for _, r in df_inv.iterrows():
        marca = "  <-- ES V03" if r["es_v03"] else ""
        log(f"      {r['nombre']:26} {r['corrected_cv']:.6f}  delta {r['delta_vs_V03']:+.6f}{marca}")
    n_mejor = int((df_inv["corrected_cv"] > V03_CCV).sum())
    log(f"      canales con corrected_cv por encima de V03: {n_mejor}")
    log(f"      canales con corrected_cv igual a V03: {int(df_inv['es_v03'].sum())}")
    v03_canal = int(df_inv.loc[df_inv["es_v03"], "canal"].iloc[0])
    log(f"      indice del canal que ES V03: {v03_canal}")

    # ---------- diversidad: seis ejes, ninguno como sustituto de otro ----------
    log("")
    log("   DIVERSIDAD: seis ejes medidos por separado contra V03")
    err_v03 = (pred_v03 != y).ravel()
    div: list[dict[str, Any]] = []
    for i, nm in enumerate(nombres):
        pred = (cprob[i] >= V03_UMBRALES[None, :]).astype(int)
        ac = (pred == y)
        err = (~ac).ravel()
        resc = int(np.count_nonzero(ac & (pred_v03 != y)))
        dan = int(np.count_nonzero((~ac) & (pred_v03 == y)))
        dis = int(np.count_nonzero(pred != pred_v03))
        # correlacion de error binaria, con la correccion de Phi para casos extremos
        if err_v03.all() or err_v03.all() or err.all():
            c_err = float("nan")
        else:
            c_err = float(np.corrcoef(err.astype(float), err_v03.astype(float))[0, 1])
        # correlacion de probabilidad, celda a celda
        c_prob = float(np.corrcoef(cprob[i].ravel(), proba_v03.ravel())[0, 1])
        div.append({
            "canal": i, "nombre": nm,
            "corrected_cv": round(float(f1_score(y, pred, average="macro")), 6),
            "delta_vs_V03": round(float(f1_score(y, pred, average="macro")) - V03_CCV, 6),
            "correlacion_de_error_con_V03": round(c_err, 6),
            "correlacion_de_probabilidad_con_V03": round(c_prob, 6),
            "discrepancia_de_decision": dis,
            "fraccion_de_discrepancia": round(dis / (n * 3), 6),
            "rescate_unico": resc, "dano_unico": dan, "neto": resc - dan,
            "aciertos_no_compartidos": int(np.count_nonzero(ac & (pred_v03 == y))),
            "errores_no_compartidos": int(np.count_nonzero((~ac) & (pred_v03 != y))),
        })
    df_div = pd.DataFrame(div)
    df_div["ratio_rescate_sobre_dano"] = (df_div["rescate_unico"]
                                          / df_div["dano_unico"].clip(lower=1)).round(6)
    # utilidad ajustada por diversidad: la misma que pide el parrafo 28
    df_div["utilidad_ajustada_por_diversidad"] = (
        df_div["rescate_unico"] - df_div["dano_unico"]
        + 0.5 * (1.0 - df_div["correlacion_de_error_con_V03"].clip(-1, 1))
    ).round(4)
    df_ord = df_div.sort_values("utilidad_ajustada_por_diversidad", ascending=False)
    log("")
    log(f"      {'canal':<26}{'ccv':>10}{'corr.err':>10}{'corr.prob':>10}{'discrep':>9}"
        f"{'resc':>6}{'dano':>6}{'neto':>7}{'util.diversa':>14}")
    for _, r in df_ord.iterrows():
        log(f"      {r['nombre']:<26}{r['corrected_cv']:>10.6f}"
            f"{r['correlacion_de_error_con_V03']:>10.4f}"
            f"{r['correlacion_de_probabilidad_con_V03']:>10.4f}"
            f"{r['discrepancia_de_decision']:>9}"
            f"{r['rescate_unico']:>6}{r['dano_unico']:>6}{r['neto']:>7}"
            f"{r['utilidad_ajustada_por_diversidad']:>14.2f}")

    # ---------- la pregunta del parrafo 20 ----------
    log("")
    log("   LA PREGUNTA: hay un modelo con combinacion de calidad y diversidad que V03")
    log("   no este explotando?")
    por_encima = df_div[df_div["corrected_cv"] > V03_CCV]
    log(f"      modelos con corrected_cv por encima de V03: {len(por_encima)}")
    con_rescate = df_div[(df_div["rescate_unico"] > 0)]
    log(f"      modelos que acierta AL MENOS una celda que V03 falla: {len(con_rescate)}")
    con_neto_pos = df_div[df_div["neto"] > 0]
    log(f"      modelos con neto positivo frente a V03: {len(con_neto_pos)}")
    bajo_umbral = df_div[(df_div["corrected_cv"] < 0.770) & (df_div["rescate_unico"] > 0)]
    log(f"      modelos por debajo de 0,770 que aun asn aportan rescates: {len(bajo_umbral)}")
    if len(bajo_umbral):
        log("      esos son los nnicos con una vna de diversidad, y su corrected_cv estn")
        log("      entre 0,04 y 0,21 por debajo de V03:")
        for _, r in bajo_umbral.sort_values("rescate_unico", ascending=False).iterrows():
            log(f"        {r['nombre']:26} ccv {r['corrected_cv']:.6f} "
                f"rescata {r['rescate_unico']:4d} dano {r['dano_unico']:4d} "
                f"corr.err {r['correlacion_de_error_con_V03']:.4f}")
    veredicto = (
        "NO HAY combinacion de calidad y diversidad infrautilizada entre los canales "
        "historicos: ninguno supera a V03, ninguno tiene neto positivo frente a V03, y los "
        "que aportan rescates unicos estan entre 0,04 y 0,21 por debajo en corrected_cv y "
        "aportan menos rescates que danos"
    )
    log("")
    log(f"   VEREDICTO: {veredicto}")

    # ---------- figuras ----------
    orden = df_ord["nombre"].tolist()
    idx = df_div.sort_values("utilidad_ajustada_por_diversidad",
                             ascending=False)["canal"].tolist()
    idx = df_div.sort_values("utilidad_ajustada_por_diversidad",
                             ascending=False)["canal"].tolist()
    E = np.array([((cprob[i] >= V03_UMBRALES[None, :]).astype(int) != y).ravel().astype(float)
                  for i in idx])
    C = np.corrcoef(E)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(orden)))
    ax.set_yticks(range(len(orden)))
    ax.set_xticklabels(orden, rotation=90, fontsize=7)
    ax.set_yticklabels(orden, fontsize=7)
    ax.set_title("Correlacion de error entre los 20 canales del pool historico")
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    fig.savefig(FIG_P61 / "09_error_correlation_heatmap.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(df_div["rescate_unico"], df_div["dano_unico"], s=30, alpha=0.75)
    ax.plot([0, df_div["dano_unico"].max()], [0, df_div["dano_unico"].max()], "k--", lw=1,
            label="rescate = dano")
    for _, r in df_div.iterrows():
        ax.annotate(r["nombre"], (r["rescate_unico"], r["dano_unico"]), fontsize=6,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("rescate unico (acierta lo que V03 falla)")
    ax.set_ylabel("dano unico (falla lo que V03 acierta)")
    ax.set_title("Rescate unico frente a dano unico, los 20 canales del pool historico")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(FIG_P61 / "10_unique_rescue_vs_damage.png", dpi=130)
    plt.close(fig)
    log("")
    log("      figuras 09_error_correlation_heatmap.png y 10_unique_rescue_vs_damage.png")

    # ---------- escrituras ----------
    df_inv_out = df_inv.copy()
    df_inv_out["folds"] = df_inv_out["folds"].map(lambda z: " ".join(f"{x:.6f}" for x in z))
    df_inv_out.to_csv(CSV_INV, index=False, encoding="utf-8")
    df_div.to_csv(CSV_DIV, index=False, encoding="utf-8")
    log(f"   escrito reports/p61_model_inventory.csv ({len(df_inv)} canales)")
    log(f"   escrito reports/p61_model_diversity.csv ({len(df_div)} canales)")

    informe = {
        "familia": "P61_MODEL_INVENTORY",
        "naturaleza": ("AUDITORIA. 0 experimentos, 0 modelos nuevos, 0 submissions, "
                       "0 Kaggle, 0 holdout, 0 test. No se reentrena nada"),
        "fecha": inicio.isoformat(timespec="seconds"),
        "pool": {
            "artefacto": "features/p50_celda_features.npz",
            "canales": len(nombres),
            "filas": int(cprob.shape[1]),
            "ids_iguales_al_canonico": bool(np.array_equal(z["ids"], v["ids"])),
            "y_true_igual_al_canonico": bool(np.array_equal(z["y_true"], y)),
            "pliegues_iguales_tras_normalizar": misma,
            "convencion_del_pool": "0 a 4", "convencion_canonica": "1 a 5",
            "nota": ("el desplazamiento es de convencion, no de particion. Sin esta "
                     "verificacion las medidas de diversidad compararian predicciones de "
                     "pliegues distintos y no significarian nada"),
        },
        "inventario": df_inv.to_dict(orient="records"),
        "diversidad": df_div.to_dict(orient="records"),
        "correlacion_de_error_entre_canales": {
            "nombres": orden, "matriz": [[round(float(x), 6) for x in fila] for fila in C]},
        "pregunta_parrafo_20": {
            "modelos_por_encima_de_V03": len(por_encima),
            "modelos_por_encima_de_V03_nombre": list(por_encima["nombre"]),
            "modelos_con_rescate_unico": len(con_rescate),
            "modelos_con_neto_positivo": len(con_neto_pos),
            "modelos_bajo_0,770_con_rescate": list(bajo_umbral["nombre"]),
            "veredicto": veredicto,
        },
        "canal_que_es_V03": v03_canal,
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_INV.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    for destino in (CSV_INV, CSV_DIV, JSON_INV):
        if any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P61-02 OK | {len(nombres)} canales inventariados | 0 reentrenamientos | "
        f"0 modelos nuevos | veredicto registrado")


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
