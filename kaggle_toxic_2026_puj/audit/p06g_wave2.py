"""P6g: segunda ola de reproducibilidad (correr DESPUES del driver P6).

- Re-ejecuta 02 (submission v01) y 04 (submissions v02/v03) con snapshot/restore.
- Compara byte a byte las submissions reproducidas contra las congeladas.
- Ejecuta el notebook completo con nbconvert en el worktree aislado y registra
  errores/duracion (verificacion en vivo de EBD-26).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/carlos/Git/PLN-toxic-rerun/kaggle_toxic_2026_puj")
REF = Path("/home/carlos/Git/PLN-v8/kaggle_toxic_2026_puj")
OUT = ROOT / "audit" / "out_p06"


def load_driver():
    path = ROOT / "audit" / "p06_rerun_driver.py"
    spec = importlib.util.spec_from_file_location("p06driver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    driver = load_driver()
    resultado = {"submissions": [], "notebook": {}}
    for name, script in (("v01_submission", "experiments/02_baseline_tfidf_lr.py"),
                         ("v03_submission", "experiments/04_refit_submissions_v02_v03.py")):
        info = driver.run_one(name, script)
        resultado[name] = info
        for archivo in ("toxic_submission_v01_baseline_tfidf_lr.csv",
                        "toxic_submission_v02_lr_union.csv",
                        "toxic_submission_v03_lr_union_thr.csv"):
            nuevo = OUT / name / "files" / "kaggle_toxic_2026_puj" / "submissions" / archivo
            congelado = REF / "submissions" / archivo
            if nuevo.exists() and congelado.exists():
                igual = nuevo.read_bytes() == congelado.read_bytes()
                resultado["submissions"].append({
                    "run": name, "archivo": archivo, "byte_igual": bool(igual),
                    "sha_repro": sha256_of(nuevo)[:16], "sha_congelado": sha256_of(congelado)[:16],
                })

    # Notebook en vivo
    destino = OUT / "notebook_exec"
    destino.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [str(Path("/home/carlos/venv/bin/jupyter")), "nbconvert", "--to", "notebook",
         "--execute", "notebooks/10_final_pipeline.ipynb",
         "--output", "10_final_pipeline_executed.ipynb", "--output-dir", str(destino),
         "--ExecutePreprocessor.timeout=1800"],
        cwd=str(ROOT), capture_output=True, text=True)
    ejecutado = destino / "10_final_pipeline_executed.ipynb"
    errores = []
    celdas = 0
    if ejecutado.exists():
        nb = json.loads(ejecutado.read_text(encoding="utf-8"))
        celdas = len(nb["cells"])
        for i, c in enumerate(nb["cells"]):
            if c["cell_type"] == "code":
                for out in c.get("outputs", []):
                    if out.get("output_type") == "error":
                        errores.append({"celda": i, "ename": out.get("ename")})
    resultado["notebook"] = {
        "returncode": proc.returncode, "celdas": celdas, "errores": errores[:10],
        "stderr_tail": proc.stderr[-400:],
    }
    (OUT / "p06g_wave2.json").write_text(json.dumps(resultado, indent=2), encoding="ascii")
    print(json.dumps({"submissions": resultado["submissions"],
                      "notebook": {k: v for k, v in resultado["notebook"].items()
                                   if k != "stderr_tail"}}, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
