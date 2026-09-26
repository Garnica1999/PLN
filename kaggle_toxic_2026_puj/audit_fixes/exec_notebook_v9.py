"""Ejecuta el notebook ya parcheado y registra la validacion (B-14, paso 2)."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

PROJ = Path("/home/carlos/Git/PLN-v8/kaggle_toxic_2026_puj")
NB_PATH = PROJ / "notebooks" / "10_final_pipeline.ipynb"
REGISTRO = PROJ / "reports" / "notebook_ejecucion_v9.json"


def main() -> None:
    inicio = time.perf_counter()
    proc = subprocess.run(
        ["/home/carlos/venv/bin/jupyter", "nbconvert", "--to", "notebook", "--execute",
         "--inplace", "notebooks/10_final_pipeline.ipynb",
         "--ExecutePreprocessor.timeout=1800"],
        cwd=str(PROJ), capture_output=True, text=True)
    duracion = round(time.perf_counter() - inicio, 1)
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    errores = []
    celdas_codigo = 0
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        celdas_codigo += 1
        for salida in c.get("outputs", []):
            if salida.get("output_type") == "error":
                errores.append({"celda": i, "ename": salida.get("ename"),
                                "evalue": str(salida.get("evalue", ""))[:200]})
    registro = {
        "fecha": "2026-09-26",
        "notebook": "notebooks/10_final_pipeline.ipynb",
        "celdas_totales": len(nb["cells"]),
        "celdas_codigo": celdas_codigo,
        "returncode": proc.returncode,
        "errores": errores,
        "duracion_s": duracion,
        "stderr_tail": proc.stderr[-800:],
    }
    REGISTRO.write_text(json.dumps(registro, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in registro.items() if k != "stderr_tail"},
                     indent=2, ensure_ascii=False))
    if registro["returncode"] != 0 or errores:
        print("STDERR:", registro["stderr_tail"][-600:])
        raise SystemExit(1)
    print("NOTEBOOK OK")


if __name__ == "__main__":
    main()
