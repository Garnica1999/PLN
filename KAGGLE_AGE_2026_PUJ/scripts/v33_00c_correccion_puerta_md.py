"""V33 - E1 - Correccion del documento NORMATIVO de la puerta (filas de la tabla).

## Que corrige

`scripts/v33_00b_enmienda_preregistro.py` intento sustituir la fila G1 de
`reports/v33_convergence_gate.md` buscando una cadena literal que no coincidia con la fila real
(el script buscaba `1.791759469228055` y la fila tiene `1.791759`). El `.replace` no ocurrio, asi
que el documento normativo seguia anunciando la banda congelada `[0.25, 0.60]` mientras el JSON
del pre-registro y el codigo del runner usan `[0.05, 1.6125835223052496]` (0.9*log(6)). Un
documento normativo que contradice al protocolo ejecutado es un defecto de trazabilidad: este
script lo arregla tomando la fila **del JSON del pre-registro como unica fuente de verdad**.

NO se edita `scripts/v33_00b_enmienda_preregistro.py` (ya ejecutado); la correccion queda
registrada aqui, con su propio rastro en RUNLOG.md.

Tambien corrige el ancla de G3, que anuncia `(clase 0)` (el indice entero de la clase) en lugar
del nombre de clase que el JSON ya congelaba (`13-17`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg

PREREG_JSON = ROOT / "experiments" / "v33_00_preregistro.json"
PUERTA_MD = ROOT / "reports" / "v33_convergence_gate.md"
RUNLOG = ROOT / "RUNLOG.md"
FECHA = "2026-09-30"


def main() -> int:
    prereg = json.loads(PREREG_JSON.read_text(encoding="utf-8"))
    banda = prereg["puerta_convergencia"]["umbrales_derivados"]["G1_ll_train_banda"]["rango"]
    anclaje = prereg["puerta_convergencia"]["umbrales_derivados"]["G1_ll_train_banda"]["anclaje"]
    clase_indice = int(prereg["anclas"]["clase_mayoritaria"])
    clase = cfg.CLASSES[clase_indice]
    log6 = prereg["anclas"]["log_uniforme_k6"]
    print(f"  fuente de verdad (JSON): G1 {banda} | log(6) = {log6!r} | clase = {clase}", flush=True)

    filas = PUERTA_MD.read_text(encoding="utf-8").splitlines()
    nueva_fila = (f"| G1 | {banda[0]} <= log_loss multiclase de train <= {banda[1]!r} | suelo "
                  f"{banda[0]} = detector de humo (una pasada no memoriza); techo 0.9*log(6) = "
                  f"{banda[1]!r} (mejorar el uniforme >= 10 % relativo) | 1e-9 |")
    cambios = 0
    for i, linea in enumerate(filas):
        if linea.startswith("| G1 |"):
            if linea != nueva_fila:
                filas[i] = nueva_fila
                cambios += 1
            break
    else:
        print("[STOP] no se encontro la fila G1 en el documento normativo", flush=True)
        return 1

    texto = "\n".join(filas) + "\n"
    if f"(clase {clase_indice})" in texto:
        texto = texto.replace(f"(clase {clase_indice})", f"(clase {clase})")
        cambios += 1
    elif "(clase 0)" in texto:
        texto = texto.replace("(clase 0)", f"(clase {clase})")
        cambios += 1
    PUERTA_MD.write_text(texto, encoding="utf-8")

    if "| G1 | 0.25 <=" in texto or "[0.25, 0.60] |" in texto:
        print("[STOP] la fila G1 obsoleta sigue en el documento normativo", flush=True)
        return 1
    if f"| G1 | {banda[0]} <=" not in texto:
        print("[STOP] la fila G1 nueva no quedó en el documento", flush=True)
        return 1
    print(f"  {cambios} fila(s) corregida(s) en {PUERTA_MD.name}", flush=True)
    print(f"  anclaje en el documento (resumen): {anclaje[:72]}...", flush=True)

    texto_runlog = RUNLOG.read_text(encoding="utf-8")
    if "Correccion de la puerta de V33-E1" in texto_runlog:
        texto_runlog = texto_runlog.replace("`(clase 0)`", f"`(clase {clase})`")
        RUNLOG.write_text(texto_runlog, encoding="utf-8")
        print("  RUNLOG.md: nota del ancla de clase corregida", flush=True)
    else:
        with RUNLOG.open("a", encoding="utf-8") as f:
            f.write("\n---\n\n## Correccion de la puerta de V33-E1 · " + FECHA + "\n\n"
                    "- El documento normativo `reports/v33_convergence_gate.md` seguia "
                    "anunciando la banda G1 congelada `[0.25, 0.60]`\n  porque el "
                    "reemplazo literal de la enmienda 1 no coincidia con la fila real. Esta "
                    "correccion reconstruye la fila\n  desde el JSON del pre-registro "
                    f"(fuente unica de verdad): G1 `{banda[0]} <= ll_train <= "
                    f"{banda[1]!r}` (0.9*log(6)), y el ancla de G3 pasa de `(clase 0)` a "
                    f"`(clase {clase})`.\n- Sin cambios en el protocolo ni en el codigo del "
                    "runner: el JSON y el runner ya usaban la banda enmendada.\n")
        print("  RUNLOG.md: correccion anotada", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
