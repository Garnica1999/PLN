"""P15: reporte de regeneracion de submissions (audit_submission_regeneration.csv).

Lee el resultado de la sesion P6j (worktree de re-ejecucion) y produce el CSV
deliverable con el mapeo script -> submission -> sha regenerado vs congelado.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

PROJ = Path("/home/carlos/Git/PLN-toxic-audit/kaggle_toxic_2026_puj")
RERUN = Path("/home/carlos/Git/PLN-toxic-rerun/kaggle_toxic_2026_puj")
FUENTE = RERUN / "audit" / "out_p06" / "regen" / "p06j_regen.json"

MODELO_POR_ARCHIVO = {
    "toxic_submission_v01_baseline_tfidf_lr.csv": "B1 word(1,2) + LR C=1 balanced (script 02)",
    "toxic_submission_v02_lr_union.csv": "M2 union word+char + LR C=2 (script 04, umbral 0.5)",
    "toxic_submission_v03_lr_union_thr.csv": "M2 union + umbrales 0.44/0.51/0.46 (script 04)",
    "toxic_submission_v04_ens04_mediana.csv": "ENS-04 promedio M2/M3, umbrales medianos (05/07)",
    "toxic_submission_v06_robustez_c1.csv": "union C=1.0 umbral optimo (sonda script 10)",
    "toxic_submission_v07_robustez_mindf5.csv": "union min_df=5 C=1.0 umbral optimo (sonda script 10)",
    "toxic_submission_v08_p1_fold_ensemble.csv": "promedio de modelos por pliegue (v8_01)",
    "toxic_submission_v09_candidate_a.csv": "meta restringido V03+EMO (script 25)",
    "toxic_submission_v10_candidate_b.csv": "meta restringido V03+SYN (script 25)",
    "toxic_submission_v11_candidate_c.csv": "meta restringido V03+EMO+SYN (script 25)",
    "toxic_submission_v11_p7_corrector_margenes.csv": "corrector de margenes Ridge (v8_08)",
    "toxic_submission_v12_p7_promedio_repeticiones.csv": "promedio de repeticiones (v8_08)",
    "toxic_submission_v13_p10_ensemble_cinco_modelos.csv": "ensemble de cinco modelos (v8_10)",
    "toxic_submission_v14_complejidad_idiosincrasias.csv": "complejidad e idiosincrasias (v8_11)",
    "toxic_submission_v15_seleccion_por_grupo.csv": "seleccion por grupo (v8_12)",
    "toxic_submission_v16_clasificadores_alternativos.csv": "arboles alternativos (v8_13)",
}


def main() -> None:
    datos = json.loads(FUENTE.read_text())
    exits = datos.get("exits", {})
    filas = []
    for fila in datos["submissions"]:
        archivo = fila["archivo"]
        filas.append({
            "Submission": archivo,
            "Modelo": MODELO_POR_ARCHIVO.get(archivo, ""),
            "Regenerada": "si" if fila.get("regenerada") else "no",
            "Sha_regen": fila.get("sha_regen", ""),
            "Sha_congelada": fila.get("sha_congelada", ""),
            "Byte_igual": ("si" if fila.get("byte_igual") == 1 else
                           ("no" if fila.get("byte_igual") == 0 else "n/a")),
            "Notas": "",
        })
    for f in filas:
        if f["Regenerada"] == "no":
            f["Notas"] = "no regenerada (ver exits/limitaciones)"
        elif f["Byte_igual"] == "si":
            f["Notas"] = "reproduccion byte a byte"
    salida = PROJ / "reports" / "audit_submission_regeneration.csv"
    with salida.open("w", newline="", encoding="ascii") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    print("submissions en el reporte:", len(filas))
    for f in filas:
        print(f"  {f['Byte_igual']:4s} {f['Submission']:55s} regen={f['Regenerada']}")
    print("exits:", {k: v["exit"] for k, v in exits.items()})


if __name__ == "__main__":
    main()
