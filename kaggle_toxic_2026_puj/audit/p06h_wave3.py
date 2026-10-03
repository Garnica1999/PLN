"""P6h (ola 3): cadena v4 completa 27 -> 28 -> 29 -> 36 sin restaurar features entre pasos.

Los scripts 29 y 36 dependen de features/label_consistency_v4.npz y
features/hardness_v4.npz (generados por 27 y 28), que no se conservan en las
superficies. Se ejecutan en una sola sesion con snapshot/restore al final.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path("/home/carlos/Git/PLN-toxic-rerun/kaggle_toxic_2026_puj")
REPO = ROOT.parent
OUT = ROOT / "audit" / "out_p06"
VENV = "/home/carlos/venv/bin/python"
PROJ = "kaggle_toxic_2026_puj"
WATCH = ROOT / "features"
SCRIPTS = [
    ("14_oof_corregido", "experiments/14_oof_corregido_v2.py"),
    ("16_targeting", "experiments/16_targeting_v2.py"),
    ("17_stacking", "experiments/17_stacking_v2.py"),
    ("19_sintaxis", "experiments/19_sintaxis_cita_negacion_v2.py"),
    ("20_emocion", "experiments/20_emocion_nbsvm_ortografia_v2.py"),
    ("27_label_consistency", "experiments/27_label_consistency_v4.py"),
    ("28_hardness", "experiments/28_hardness_v4.py"),
    ("29_reweighting", "experiments/29_reweighting_v4.py"),
    ("30_supervised_weighting", "experiments/30_supervised_weighting_v4.py"),
    ("31_joint_label", "experiments/31_joint_label_v4.py"),
    ("32_meta_shift", "experiments/32_meta_shift_v4.py"),
    ("33_class_conditional", "experiments/33_class_conditional_lm_v4.py"),
    ("34_structural_sparse", "experiments/34_structural_sparse_v4.py"),
    ("35_discovery", "experiments/35_discovery_v4.py"),
    ("36_candidatos", "experiments/36_candidatos_v4.py"),
]


def load_driver():
    path = ROOT / "audit" / "p06_rerun_driver.py"
    spec = importlib.util.spec_from_file_location("p06driver2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git_status() -> dict[str, str]:
    out = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                         capture_output=True, text=True, check=True).stdout
    return {line[3:].strip(): line[:2] for line in out.splitlines() if line.strip()}


def main() -> None:
    driver = load_driver()
    name = "v4_pipeline"
    run_dir = OUT / name
    (run_dir / "files").mkdir(parents=True, exist_ok=True)
    backup = run_dir / "features_before"
    shutil.rmtree(backup, ignore_errors=True)
    shutil.copytree(WATCH, backup, symlinks=True)
    before = git_status()
    exits = {}
    log = run_dir / "run.log"
    with log.open("wb") as fh:
        for etiqueta, script in SCRIPTS:
            inicio = time.time()
            fh.write(f"\n===== {etiqueta}: {script}\n".encode())
            fh.flush()
            proc = subprocess.run([VENV, script], cwd=str(ROOT), stdout=fh,
                                  stderr=subprocess.STDOUT)
            exits[etiqueta] = {"exit": proc.returncode,
                               "duracion_s": round(time.time() - inicio, 1)}
    after = git_status()
    changed = [(p, c) for p, c in after.items()
               if p.startswith(PROJ + "/") and not p[len(PROJ) + 1:].startswith("audit")
               and before.get(p) != c]
    for path, _code in changed:
        full = REPO / path
        dest = run_dir / "files" / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        if full.is_file():
            shutil.copy2(full, dest)
        elif full.is_dir():
            shutil.copytree(full, dest, dirs_exist_ok=True)
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
    features_changed = []
    if WATCH.exists() and backup.exists():
        after_files = {p.relative_to(WATCH) for p in WATCH.rglob("*") if p.is_file()}
        before_files = {p.relative_to(backup) for p in backup.rglob("*") if p.is_file()}
        for rel in sorted(after_files | before_files):
            a, b = WATCH / rel, backup / rel
            if not a.exists() or not b.exists() or (
                    hashlib.sha256(a.read_bytes()).digest()
                    != hashlib.sha256(b.read_bytes()).digest()):
                features_changed.append(str(rel))
                dest = run_dir / "files" / PROJ / "features" / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                if a.exists():
                    shutil.copy2(a, dest)
        shutil.rmtree(WATCH, ignore_errors=True)
        shutil.copytree(backup, WATCH, symlinks=True)
    resultado = {"name": name, "scripts": SCRIPTS, "exits": exits,
                 "tracked_changed": [p for p, _ in changed],
                 "features_changed": features_changed}
    resultado["comparaciones"] = driver.comparar(name)
    resultado["controles"] = driver.controles(name)
    (run_dir / "result.json").write_text(json.dumps(resultado, indent=2), encoding="ascii")
    print(json.dumps({"exits": exits, "cambiados": len(changed),
                      "features": features_changed[:6],
                      "diffs": [(c["rel"], c.get("n_diffs"), c.get("max_abs_float"))
                                for c in resultado["comparaciones"]
                                if isinstance(c.get("n_diffs"), int) and c["n_diffs"]]},
                     indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
