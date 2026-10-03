"""P6 driver: re-ejecuta experimentos clave en el worktree de re-ejecucion.

Para cada script: snapshot (git status + features/), ejecucion con el venv,
captura de ficheros cambiados y del log, restauracion del estado y comparacion
contra los artefactos de referencia (worktree v8). Sin red, sin submissions.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path("/home/carlos/Git/PLN-toxic-rerun/kaggle_toxic_2026_puj")
REPO = ROOT.parent
OUT = ROOT / "audit" / "out_p06"
REF = Path("/home/carlos/Git/PLN-v8/kaggle_toxic_2026_puj")
VENV = "/home/carlos/venv/bin/python"
WATCH = ROOT / "features"
PROJ = "kaggle_toxic_2026_puj"

RUNS = [
    ("v03_oof", "experiments/14_oof_corregido_v2.py"),
    ("pa01", "experiments/12_control_pseudo_autores.py"),
    ("v6_02", "experiments/v6_02_label_factorization.py"),
    ("v6_03", "experiments/v6_03_ponderacion_grupo.py"),
    ("v6_04", "experiments/v6_04_char_weight.py"),
    ("v6_05", "experiments/v6_05_skipgrams.py"),
    ("v6_06", "experiments/v6_06_bagging_grupos.py"),
    ("v6_07", "experiments/v6_07_ruido_par.py"),
    ("v6_08", "experiments/v6_08_aumento.py"),
    ("v7_01", "experiments/v7_01_perdida_f1.py"),
    ("v7_02", "experiments/v7_02_prior_grupo.py"),
    ("v7_03", "experiments/v7_03_umbral_anidado.py"),
    ("v4_reweighting", "experiments/29_reweighting_v4.py"),
    ("v4_meta_shift", "experiments/32_meta_shift_v4.py"),
    ("v4_candidatos", "experiments/36_candidatos_v4.py"),
    ("v8_01", "experiments/v8_01_ensamble_folds.py"),
    ("v8_03", "experiments/v8_03_estabilidad_features.py"),
    ("v8_05", "experiments/v8_05_regularizacion_conjunta.py"),
    ("dl11", "experiments/11_dl_desde_cero.py"),
]

CONTROLES = (0.777211, 0.777165, 0.7772)


def git_status() -> dict[str, str]:
    out = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                         capture_output=True, text=True, check=True).stdout
    result = {}
    for line in out.splitlines():
        if not line.strip():
            continue
        code, path = line[:2], line[3:].strip()
        result[path] = code
    return result


def run_one(name: str, script: str) -> dict:
    run_dir = OUT / name
    (run_dir / "files").mkdir(parents=True, exist_ok=True)
    before = git_status()
    backup = run_dir / "features_before"
    shutil.rmtree(backup, ignore_errors=True)
    if WATCH.exists():
        shutil.copytree(WATCH, backup, symlinks=True)
    log = run_dir / "run.log"
    inicio = time.time()
    with log.open("wb") as fh:
        proc = subprocess.run([VENV, script], cwd=str(ROOT), stdout=fh,
                              stderr=subprocess.STDOUT)
    duracion = time.time() - inicio
    after = git_status()
    changed = []
    for path, code in after.items():
        if not path.startswith(PROJ + "/"):
            continue
        rel = path[len(PROJ) + 1:]
        if rel.startswith("audit"):
            continue
        if before.get(path) != code:
            changed.append((path, code))
    for path, code in changed:
        full = REPO / path
        dest = run_dir / "files" / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        if full.is_dir():
            shutil.copytree(full, dest, dirs_exist_ok=True)
        elif full.exists():
            shutil.copy2(full, dest)
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
    resultado = {
        "name": name, "script": script, "exit": proc.returncode,
        "duracion_s": round(duracion, 1),
        "tracked_changed": [p for p, _ in changed],
        "features_changed": features_changed,
    }
    (run_dir / "result.json").write_text(json.dumps(resultado, indent=2), encoding="ascii")
    return resultado


def walk_compare(x, y, path, diffs, estado):
    if isinstance(x, dict) and isinstance(y, dict):
        for key in sorted(set(x) | set(y)):
            if key not in x or key not in y:
                diffs.append({"path": f"{path}.{key}", "tipo": "clave"})
            else:
                walk_compare(x[key], y[key], f"{path}.{key}", diffs, estado)
    elif isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            diffs.append({"path": path, "tipo": "longitud", "orig": len(x), "nuevo": len(y)})
        for i, (xi, yi) in enumerate(zip(x, y)):
            walk_compare(xi, yi, f"{path}[{i}]", diffs, estado)
    else:
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) \
                and not isinstance(x, bool) and not isinstance(y, bool):
            d = abs(float(x) - float(y))
            estado["n_floats"] += 1
            estado["max_abs"] = max(estado["max_abs"], d)
            if d > 1e-9 and len(diffs) < 40:
                diffs.append({"path": path, "tipo": "float", "orig": x, "nuevo": y, "diff": d})
        elif x != y and len(diffs) < 40:
            diffs.append({"path": path, "tipo": "valor",
                          "orig": str(x)[:80], "nuevo": str(y)[:80]})


def comparar(name: str) -> list[dict]:
    run_dir = OUT / name / "files" / PROJ
    resultados = []
    for new_path in sorted(run_dir.glob("reports/*.json")):
        rel = new_path.relative_to(run_dir)
        orig = REF / rel
        estado = {"n_floats": 0, "max_abs": 0.0}
        diffs: list[dict] = []
        if not orig.exists():
            resultados.append({"rel": str(rel), "estado": "ORIGINAL_AUSENTE"})
            continue
        try:
            a = json.loads(orig.read_text(encoding="utf-8"))
            b = json.loads(new_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            resultados.append({"rel": str(rel), "estado": f"ERROR_JSON:{exc}"})
            continue
        walk_compare(a, b, "", diffs, estado)
        resultados.append({
            "rel": str(rel), "n_floats": estado["n_floats"],
            "max_abs_float": round(estado["max_abs"], 12),
            "n_diffs": len(diffs), "diffs": diffs[:10],
        })
    for new_path in sorted(run_dir.glob("reports/*.csv")):
        rel = new_path.relative_to(run_dir)
        orig = REF / rel
        igual = orig.exists() and orig.read_bytes() == new_path.read_bytes()
        resultados.append({"rel": str(rel), "bytes_iguales": bool(igual)})
    return resultados


def controles(name: str) -> list[dict]:
    """Busca en los JSON de referencia valores ~ control y verifica su re-ejecucion."""
    run_dir = OUT / name / "files" / PROJ
    salida = []
    for new_path in sorted(run_dir.glob("reports/*.json")):
        rel = new_path.relative_to(run_dir)
        orig = REF / rel
        if not orig.exists():
            continue
        try:
            a = json.loads(orig.read_text(encoding="utf-8"))
            b = json.loads(new_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        hits = []

        def scan(x, y, path):
            if isinstance(x, dict) and isinstance(y, dict):
                for key in x:
                    if key in y:
                        scan(x[key], y[key], f"{path}.{key}")
            elif isinstance(x, list) and isinstance(y, list):
                for i, (xi, yi) in enumerate(zip(x, y)):
                    scan(xi, yi, f"{path}[{i}]")
            elif isinstance(x, float):
                for ref in CONTROLES:
                    if abs(x - ref) < 1e-6:
                        ok = isinstance(y, float) and abs(y - x) < 1e-9
                        hits.append({"path": path, "orig": x, "nuevo": y, "ok": ok})

        scan(a, b, "")
        if hits:
            salida.append({"rel": str(rel), "hits": hits[:8], "n": len(hits)})
    return salida


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    resumen = []
    for name, script in RUNS:
        print(f"== {name}: {script}", flush=True)
        resultado = run_one(name, script)
        resultado["comparaciones"] = comparar(name)
        resultado["controles"] = controles(name)
        resumen.append(resultado)
        fallos = [c for c in resultado["comparaciones"]
                  if isinstance(c.get("n_diffs"), int) and c["n_diffs"] > 0]
        print(f"   exit={resultado['exit']} dur={resultado['duracion_s']}s "
              f"cambiados={len(resultado['tracked_changed'])} "
              f"json_con_diffs={len(fallos)}", flush=True)
    (OUT / "p06_rerun_summary.json").write_text(
        json.dumps(resumen, indent=2), encoding="ascii")
    print("RESUMEN ESCRITO", flush=True)


if __name__ == "__main__":
    main()
