"""P75-02: recupera el analisis por slices que P75-01 perdio por un fallo de tipos en el join.

Que paso
--------
P75-01 unio la tabla de anotaciones por indice sin comprobar el tipo del identificador. El parquet
guarda `id` como TEXTO ('20430') y el OOF canonico lo guarda como entero (20430). La interseccion
fue cero, todos los reindex dieron NaN, y nueve de las once slices quedaron vacias y se saltaron
por el umbral de filas. Sobrevivieron solo las dos que no dependen del parquet.

No se repite ningun entrenamiento: el OOF pareado de los dos modelos esta guardado en
reports/p75_paired_oof.csv, que es el artefacto que manda. Esto solo recalcula las slices.

Compliance: del parquet se usa EXCLUSIVAMENTE la particion 'dev'. Las filas de 'test' y de
'holdout' no se leen ni se consultan.
"""

from __future__ import annotations

import ast
import csv as _csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p73_02_textcnn_vs_random import REPORTS, log
from p75_01_complementariedad_textcnn import (
    LISTA_INSULTOS,
    matriz_errores,
)

FIGURAS = PROJECT_ROOT / "figuras" / "p75"
FEATURES = PROJECT_ROOT / "features"


def main() -> None:
    """Recalcula las slices desde el OOF pareado guardado.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P75-02 slices recuperadas desde el OOF pareado, sin reentrenar")
    tabla = pd.read_csv(REPORTS / "p75_paired_oof.csv")
    n = int(tabla["fila"].max()) + 1
    y = tabla["y_true"].to_numpy().reshape(n, 3)
    pv = tabla["v03_pred"].to_numpy().reshape(n, 3)
    pc = tabla["cnn_pred"].to_numpy().reshape(n, 3)
    ids = tabla.drop_duplicates("fila").sort_values("fila")["row_id"].to_numpy()
    log(f"   {n} filas | {len(tabla)} unidades | ids como {ids.dtype}")

    pq = pd.read_parquet(FEATURES / "v2_structured_features.parquet")
    dev = pq[pq["particion"] == "dev"].copy()
    dev["id_int"] = dev["id"].astype(int)
    log(f"   parquet: {len(pq)} filas | particion dev {len(dev)} | test y holdout NO se leen")
    dev = dev.set_index("id_int")
    encontrados = int(pd.Series(ids).isin(dev.index).sum())
    log(f"   ids del OOF presentes en dev: {encontrados} de {n}")
    if encontrados != n:
        raise SystemExit("PARADA: el join por id no cruza todas las filas")
    f = dev.reindex(ids)
    if f["tiene_url"].isna().any():
        raise SystemExit("PARADA: quedan NaN tras el join")

    from src.data import read_train
    from src.splits import grupos_fusionados, holdout_v2
    tr = read_train()
    g = grupos_fusionados(tr)
    idev, _ = holdout_v2(tr, g)
    des = tr.iloc[idev].reset_index(drop=True)
    bajo = des["text"].astype(str).str.lower()

    slic = {
        "MENCION": f["tiene_mencion"].to_numpy() == 1,
        "COLECTIVO": f["tiene_colectivo"].to_numpy() == 1,
        "AFILIACION_proxy_densidad": f["col_densidad"].to_numpy() > 0,
        "INTERPERSONAL": f["tiene_segunda_persona"].to_numpy() == 1,
        "URL": f["tiene_url"].to_numpy() == 1,
        "HASHTAG": f["tiene_hashtag"].to_numpy() == 1,
        "CITA": f["tiene_cita"].to_numpy() == 1,
        "ELONGACION": f["tiene_elongacion"].to_numpy() == 1,
        "REPETICION": f["tiene_repeticion"].to_numpy() == 1,
        "COLOQUIAL_proxy_risas_leet": ((f["tiene_risas"].to_numpy() == 1)
                                       | (f["tiene_leet"].to_numpy() == 1)),
        "INSULTOS_lista_declarada": bajo.apply(
            lambda s: any(w in s for w in LISTA_INSULTOS)).to_numpy(),
        "SIN_ODIO_EXPLICITO": y[:, 0] == 0,
    }
    res = {}
    for nombre, m in slic.items():
        m = np.asarray(m, dtype=bool)
        if m.sum() < 20:
            log(f"   {nombre:28} filas {int(m.sum()):4d} -> por debajo del umbral, se omite")
            continue
        d = matriz_errores(y, pv, pc, m)
        res[nombre] = dict(d, filas=int(m.sum()))
        log(f"   {nombre:28} filas {int(m.sum()):4d} | rescue {d['rescue']:4d} | damage "
            f"{d['damage']:4d} | net {d['net']:+5d} | rescue_rate {d['rescue_rate']:.4f} | "
            f"damage_rate {d['damage_rate']:.4f}")

    comp = json.loads((REPORTS / "p75_fusion.json").read_text(encoding="utf-8"))
    comp["pass2_complementariedad"]["por_slice"] = res
    comp["pass2_complementariedad"]["nota_de_recuperacion"] = (
        "P75-01 perdio nueve de once slices por un fallo de tipos en el join: el parquet guarda id "
        "como texto y el OOF como entero, la interseccion fue cero y los reindex dieron NaN. Este "
        "script las recalcula desde el OOF pareado guardado, sin reentrenar nada. Solo se leyo la "
        "particion 'dev' del parquet")
    comp["pass2_complementariedad"]["proxies_declarados"]["REPETICION"] = "tiene_repeticion"
    (REPORTS / "p75_fusion.json").write_text(json.dumps(comp, ensure_ascii=False, indent=2) + "\n",
                                             encoding="utf-8")

    filas_csv = []
    with (REPORTS / "p75_textcnn_complementarity.csv").open(encoding="ascii") as fh:
        for r in _csv.reader(fh):
            filas_csv.append(r)
    nuevas = [r for r in filas_csv if not (r and r[0].startswith("SLICE_"))]
    for nombre, d in res.items():
        nuevas.append([f"SLICE_{nombre}", d["unidades"], d["rescue"], d["damage"], d["ambos_ok"],
                       d["ambos_ko"], d["net"], d["rescue_rate"], d["damage_rate"], d["net_rate"],
                       d["v03_error"], d["cnn_error"], d["filas"]])
    with (REPORTS / "p75_textcnn_complementarity.csv").open("w", encoding="ascii",
                                                            newline="") as fh:
        _csv.writer(fh, lineterminator="\n").writerows(nuevas)
    log(f"   complementarity.csv actualizado con {len(res)} slices")

    nombres = list(res)
    _fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.bar(nombres, [res[x]["net"] for x in nombres], color="#2e75b6")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("net (rescue - damage)"); ax.set_title("P75: net por slice, unidades tweet-etiqueta")
    plt.xticks(rotation=32, ha="right")
    plt.tight_layout(); plt.savefig(FIGURAS / "p75_06_slice_net.png", dpi=110); plt.close()

    salida = {"experiment_id": "P75-02", "fecha": inicio.isoformat(timespec="seconds"),
              "naturaleza": "RECUPERACION DE DIAGNOSTICO. Sin reentrenamiento",
              "causa_del_fallo": "id del parquet como texto frente a entero en el OOF",
              "slices_recuperadas": len(res), "por_slice": res,
              "compliance": "solo se leyo la particion dev del parquet"}
    (REPORTS / "p75_slices_recuperadas.json").write_text(
        json.dumps(salida, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sucios = [n for n in ("p75_textcnn_complementarity.csv", "p75_slices_recuperadas.json",
                          "p75_fusion.json")
              if any(b > 127 for b in (REPORTS / n).read_bytes())]
    if sucios:
        raise SystemExit(f"PARADA: bytes no ASCII en {sucios}")
    log("")
    log(f"   P75-02 OK | {len(res)} slices recuperadas de 11")


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
