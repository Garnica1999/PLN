"""P11: auditoria de submissions y analisis de transferencia local-LB."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

PROJ = Path("/home/carlos/Git/PLN-toxic-audit/kaggle_toxic_2026_puj")
SURFACES = {
    "A_actual": Path("/home/carlos/Git/PLN/kaggle_toxic_2026_puj"),
    "B_v7": Path("/home/carlos/Git/PLN-v7/kaggle_toxic_2026_puj"),
    "C_v8": Path("/home/carlos/Git/PLN-v8/kaggle_toxic_2026_puj"),
}


def fold(text: str) -> str:
    repl = {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
            "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u00a0": " "}
    for k, v in repl.items():
        text = text.replace(k, v)
    norm = unicodedata.normalize("NFKD", text)
    return "".join(ch if ord(ch) < 128 else "" for ch in norm if not unicodedata.combining(ch))


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:  # noqa: PLR0915
    plantilla = pd.read_csv(SURFACES["C_v8"] / "data" / "raw" / "submission.csv")
    ids_esperados = plantilla["id"].astype(int).tolist()
    manifest = pd.read_csv(SURFACES["C_v8"] / "submissions" / "submission_manifest.csv")
    log = pd.read_csv(SURFACES["C_v8"] / "submissions" / "leaderboard_log.csv")
    filas = []
    for nombre, raiz in SURFACES.items():
        for archivo in sorted((raiz / "submissions").glob("toxic_submission_*.csv")):
            frame = pd.read_csv(archivo)
            ids = frame["id"].astype(int).tolist() if "id" in frame.columns else []
            cols_ok = list(frame.columns) == ["id", "HS", "TR", "AG"]
            ids_ok = ids == ids_esperados
            vals = frame[["HS", "TR", "AG"]].to_numpy() if cols_ok else None
            binario_ok = bool(vals is not None and set(vals.ravel().tolist()) <= {0, 1})
            nan = int(frame.isna().sum().sum())
            filas.append({
                "surface": nombre, "archivo": archivo.name,
                "filas": len(frame), "columnas_ok": int(cols_ok),
                "ids_orden_ok": int(ids_ok), "binario_ok": int(binario_ok),
                "nulos": nan, "sha256_16": sha256_of(archivo)[:16],
                "filas_unicas_id": int(frame["id"].is_unique) if "id" in frame.columns else 0,
            })
    # manifest: enlazar archivo -> experimento (busca el nombre de archivo en cada fila)
    enlaces = {}
    for _, fila in manifest.iterrows():
        texto = " ".join(str(v) for v in fila.tolist())
        for nombre in re.findall(r"toxic_submission_[\w\-.]+\.csv", texto):
            enlaces[nombre] = texto[:180]
    for f in filas:
        f["enlace_manifiesto"] = fold(enlaces.get(f["archivo"], ""))[:120]
    with (PROJ / "reports" / "audit_submissions.csv").open("w", newline="", encoding="ascii") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        writer.writeheader()
        writer.writerows([{k: fold(str(v)) for k, v in f.items()} for f in filas])

    # ---------- transferencia ----------
    REF_LB = 0.78306
    filas_tr = []
    for _, r in log.iterrows():
        local = r.get("local_f1")
        kaggle = r.get("kaggle_f1")
        dif = r.get("diferencia")
        gap_local = None
        gap_v03 = None
        semantica = ""
        if pd.notna(kaggle):
            gap_v03 = round(float(kaggle) - REF_LB, 5)
            if pd.notna(local):
                gap_local = round(float(kaggle) - float(local), 5)
        if pd.isna(dif):
            semantica = "sin_diferencia_declarada"
        else:
            delta = abs(float(dif))
            if gap_local is not None and abs(delta - abs(gap_local)) <= 1e-4:
                semantica = "LB - local"
            elif gap_v03 is not None and abs(delta - abs(gap_v03)) <= 1e-4:
                semantica = "LB - LB(V03)"
            else:
                semantica = "otra/NO_CUADRA"
        filas_tr.append({
            "submission_id": fold(str(r["submission_id"])),
            "archivo": fold(str(r["archivo"])),
            "local_f1": "" if pd.isna(local) else round(float(local), 5),
            "kaggle_f1": "" if pd.isna(kaggle) else round(float(kaggle), 5),
            "diferencia_declarada": "" if pd.isna(dif) else float(dif),
            "gap_recomputado_lb_local": gap_local if gap_local is not None else "",
            "gap_recomputado_lb_vs_v03": gap_v03 if gap_v03 is not None else "",
            "semantica_declarada": semantica,
            "cuadra": int(semantica in {"LB - local", "LB - LB(V03)", "sin_diferencia_declarada"}),
            "features": fold(str(r.get("features", "")))[:90],
            "modelo": fold(str(r.get("modelo", "")))[:90],
            "fecha": fold(str(r.get("fecha", ""))),
        })
    with (PROJ / "reports" / "audit_transfer_analysis.csv").open("w", newline="", encoding="ascii") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(filas_tr[0].keys()))
        writer.writeheader()
        writer.writerows(filas_tr)

    resumen = {
        "submissions_validadas": len(filas),
        "todas_ids_ok": all(f["ids_orden_ok"] for f in filas),
        "todas_binarias": all(f["binario_ok"] for f in filas),
        "con_nulos": sum(1 for f in filas if f["nulos"]),
        "conteo_por_superficie": {s: sum(1 for f in filas if f["surface"] == s) for s in SURFACES},
        "filas_log": len(filas_tr),
        "log_cuadra": sum(f["cuadra"] for f in filas_tr),
        "semantica_por_fila": {f["submission_id"]: f["semantica_declarada"] for f in filas_tr},
        "gap_local_min": min((f["gap_recomputado_lb_local"] for f in filas_tr
                              if f["gap_recomputado_lb_local"] != ""), default=None),
        "gap_local_max": max((f["gap_recomputado_lb_local"] for f in filas_tr
                              if f["gap_recomputado_lb_local"] != ""), default=None),
    }
    (PROJ / "audit" / "out" / "p11_submissions.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=True), encoding="ascii")
    print(json.dumps(resumen, indent=2, ensure_ascii=True))
    print("transfer (todas las filas del log):")
    for f in filas_tr:
        print(f"  {f['submission_id'][:28]:28s} local={f['local_f1']:>7} lb={f['kaggle_f1']:>7} "
              f"gap_local={f['gap_recomputado_lb_local']} vs_v03={f['gap_recomputado_lb_vs_v03']} "
              f"semantica={f['semantica_declarada']}")


if __name__ == "__main__":
    main()
