"""P6j (REGEN): regenera las submissions historicas con los modelos re-ajustados.

Una sola sesion (los stores intermedios persisten), sin holdout y sin re-busqueda
de Optuna. Al final: comparacion byte a byte contra las submissions congeladas,
copia de las regeneradas a out_p06/regen/subs/ y restauracion del arbol.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path("/home/carlos/Git/PLN-toxic-rerun/kaggle_toxic_2026_puj")
REPO = ROOT.parent
OUT = ROOT / "audit" / "out_p06" / "regen"
REF = Path("/home/carlos/Git/PLN-v8/kaggle_toxic_2026_puj")
VENV = "/home/carlos/venv/bin/python"
PROJ = "kaggle_toxic_2026_puj"
WATCH = ROOT / "features"

SCRIPTS = [
    ("02_baseline", "experiments/02_baseline_tfidf_lr.py"),
    ("04_refit_v02_v03", "experiments/04_refit_submissions_v02_v03.py"),
    ("05_ensembles", "experiments/05_ensembles.py"),
    ("07_submission_v04", "experiments/07_submission_v04_ens04.py"),
    ("10_sondas_robustez", "experiments/10_sondas_robustez.py"),
    ("14_oof_corregido", "experiments/14_oof_corregido_v2.py"),
    ("16_targeting", "experiments/16_targeting_v2.py"),
    ("17_stacking", "experiments/17_stacking_v2.py"),
    ("19_sintaxis", "experiments/19_sintaxis_cita_negacion_v2.py"),
    ("20_emocion", "experiments/20_emocion_nbsvm_ortografia_v2.py"),
    ("v8_01_ensamble_folds", "experiments/v8_01_ensamble_folds.py"),
    ("25_candidatos_submission", "experiments/25_submissions_candidatos.py"),
    ("v8_08_corrector", "experiments/v8_08_corrector_margenes.py"),
    ("v8_10_ensemble5", "experiments/v8_10_ensemble_cinco_modelos.py"),
    ("v8_11_idiosincrasias", "experiments/v8_11_complejidad_idiosincrasias.py"),
    ("v8_12_seleccion_grupo", "experiments/v8_12_seleccion_por_grupo.py"),
    ("v8_13_clasificadores", "experiments/v8_13_clasificadores_alternativos.py"),
]

ESPERADAS = [
    "toxic_submission_v01_baseline_tfidf_lr.csv",
    "toxic_submission_v02_lr_union.csv",
    "toxic_submission_v03_lr_union_thr.csv",
    "toxic_submission_v04_ens04_mediana.csv",
    "toxic_submission_v06_robustez_c1.csv",
    "toxic_submission_v07_robustez_mindf5.csv",
    "toxic_submission_v08_p1_fold_ensemble.csv",
    "toxic_submission_v09_candidate_a.csv",
    "toxic_submission_v10_candidate_b.csv",
    "toxic_submission_v11_candidate_c.csv",
    "toxic_submission_v11_p7_corrector_margenes.csv",
    "toxic_submission_v12_p7_promedio_repeticiones.csv",
    "toxic_submission_v13_p10_ensemble_cinco_modelos.csv",
    "toxic_submission_v14_complejidad_idiosincrasias.csv",
    "toxic_submission_v15_seleccion_por_grupo.csv",
    "toxic_submission_v16_clasificadores_alternativos.csv",
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_status() -> dict[str, str]:
    out = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                         capture_output=True, text=True, check=True).stdout
    return {line[3:].strip(): line[:2] for line in out.splitlines() if line.strip()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    backup = OUT / "features_before"
    shutil.rmtree(backup, ignore_errors=True)
    shutil.copytree(WATCH, backup, symlinks=True)
    before = git_status()
    exits = {}
    log = OUT / "regen.log"
    with log.open("wb") as fh:
        for etiqueta, script in SCRIPTS:
            inicio = time.time()
            fh.write(f"\n===== {etiqueta}: {script}\n".encode())
            fh.flush()
            proc = subprocess.run([VENV, script], cwd=str(ROOT), stdout=fh,
                                  stderr=subprocess.STDOUT)
            exits[etiqueta] = {"exit": proc.returncode,
                               "duracion_s": round(time.time() - inicio, 1)}
    # Copia de las submissions regeneradas
    subs_dir = OUT / "subs"
    subs_dir.mkdir(parents=True, exist_ok=True)
    filas = []
    for nombre in ESPERADAS:
        regen = ROOT / "submissions" / nombre
        congelada = REF / "submissions" / nombre
        fila = {"archivo": nombre, "regenerada": regen.exists(),
                "congelada": congelada.exists()}
        if regen.exists():
            shutil.copy2(regen, subs_dir / nombre)
            fila["sha_regen"] = sha256_of(regen)[:16]
        if congelada.exists():
            fila["sha_congelada"] = sha256_of(congelada)[:16]
        if regen.exists() and congelada.exists():
            fila["byte_igual"] = int(regen.read_bytes() == congelada.read_bytes())
        filas.append(fila)
    # Restauracion
    after = git_status()
    changed = [(p, c) for p, c in after.items()
               if p.startswith(PROJ + "/") and not p[len(PROJ) + 1:].startswith("audit")
               and before.get(p) != c]
    tracked = [p for p, c in changed if c.strip() not in {"??", "!!"}]
    if tracked:
        subprocess.run(["git", "-C", str(REPO), "checkout", "--", *tracked], check=False)
    for path, code in changed:
        if code.strip() == "??":
            full = REPO / path
            if full.is_dir():
                shutil.rmtree(full, ignore_errors=True)
            elif full.exists():
                full.unlink()
    shutil.rmtree(WATCH, ignore_errors=True)
    shutil.copytree(backup, WATCH, symlinks=True)
    with (OUT / "submissions_regen.csv").open("w", newline="", encoding="ascii") as fh:
        w = csv.DictWriter(fh, fieldnames=["archivo", "regenerada", "congelada",
                                           "sha_regen", "sha_congelada", "byte_igual"])
        w.writeheader()
        for fila in filas:
            w.writerow(fila)
    resumen = {"exits": exits, "submissions": filas}
    (OUT / "p06j_regen.json").write_text(json.dumps(resumen, indent=2), encoding="ascii")
    print(json.dumps({"exits": {k: v["exit"] for k, v in exits.items()}}, ensure_ascii=True))
    for fila in filas:
        print(f"  {fila['archivo']:55s} regen={fila.get('sha_regen','-'):16s} "
              f"friza={fila.get('sha_congelada','-'):16s} igual={fila.get('byte_igual','-')}")


if __name__ == "__main__":
    main()
