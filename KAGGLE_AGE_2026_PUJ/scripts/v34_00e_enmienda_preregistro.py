"""V34 - E1 - Enmienda pre-ejecucion 3: fidelidad de la receta del vectorizador.

## Defecto (typo del pre-registro)

`scripts/v34_00_preregistro.py` congelo `token_pattern` con **dobles backslashes**
(`(?u)\\b\\w\\b` en el fichero, es decir backslash literal seguido de `b`), cuando la receta
CANONICA del proyecto -la de `VECTORIZADOR` en
`scripts/v31_02_banco_representaciones.py`, usada tambien por `v32_04` y `v33_00` y por el propio
runner y por la exploracion previa de V34- declara `(?u)\b\w+\b`. La consecuencia es que el sha256
del dict del vectorizador congelado en V34 era `33311e5eddc0b0bb...` en lugar del canonico
`9bae157614458dbd...`: la representacion de V34 no era la del resto de la serie.

Este error es mio (del pre-registro) y lo detecta el propio check de no-deriva del runner al
comparar su espejo con el JSON: en cuanto se corrija el JSON, el runner debe usar la receta
canonica (que es lo que ya hace).

## Enmienda

Se restaura `token_pattern = r"(?u)\b\w+\b"` (la receta canonica) y se recalcula el sha256 del dict
del vectorizador, que vuelve a ser `9bae157614458dbd...`. No cambia ninguna otra constante, ningun
gate, ni la puerta G0-G6, ni la regla de alpha, ni el criterio de exito.

## Estado

Enmienda **PRE-EJECUCION**: ningun fold de V34-E1 ha sido entrenado; `--verificar` todavia no pasa
(por este y por dos desajustes de espejo del runner que se corrigen a la vez).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREREG_JSON = ROOT / "experiments" / "v34_00_preregistro.json"
PREREG_MD = ROOT / "reports" / "v34_preregistro.md"
RUNLOG = ROOT / "RUNLOG.md"
EXPERIMENTS = ROOT / "experiments.csv"
FECHA = "2026-09-30"

TOKEN_PATTERN_CANONICO = r"(?u)\b\w+\b"
HASH_CANONICO_ESPERADO = "9bae157614458dbd"

ENMIENDA = {
    "fecha": FECHA,
    "objeto": "representacion.vectorizador.token_pattern (fidelidad de la receta canonica)",
    "de": "(?u)\\\\b\\\\w\\\\b (dobles backslashes: typo del pre-registro; sha256 del "
          "dict 33311e5eddc0b0bb)",
    "a": TOKEN_PATTERN_CANONICO + " (receta canonica de v31_02_banco_representaciones.py, "
          "usada por v32_04, v33_00, el runner v34_01 y la exploracion previa; sha256 del dict "
          + HASH_CANONICO_ESPERADO + ")",
    "motivo": "el pre-registro de V34 se apartaba de la representacion canonica del proyecto por "
              "un typo mio; el check de no-deriva del runner lo detecto al comparar su espejo "
              "con el JSON",
    "afecta_a_gates": False,
    "estado_datos": "sin resultados; enmienda pre-ejecucion",
}


def main() -> int:
    datos = json.loads(PREREG_JSON.read_text(encoding="utf-8"))
    vectorizador = datos["representacion"]["vectorizador"]
    antes = vectorizador["token_pattern"]
    vectorizador["token_pattern"] = TOKEN_PATTERN_CANONICO
    sha = hashlib.sha256(json.dumps(vectorizador, sort_keys=True).encode("utf-8")).hexdigest()
    if not sha.startswith(HASH_CANONICO_ESPERADO):
        print(f"[STOP] el sha256 recalculado {sha[:16]} no es el canonico esperado", flush=True)
        return 1
    datos["hashes"]["vectorizador"] = sha
    enmiendas = datos.setdefault("enmiendas", [])
    if not any(x.get("objeto") == ENMIENDA["objeto"] for x in enmiendas):
        enmiendas.append(ENMIENDA)
    PREREG_JSON.write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  token_pattern: {antes!r} -> {TOKEN_PATTERN_CANONICO!r}", flush=True)
    print(f"  sha256 del vectorizador: {sha[:16]} (canonico) | enmiendas={len(enmiendas)}", flush=True)

    if "## Enmienda pre-ejecucion 3" not in PREREG_MD.read_text(encoding="utf-8"):
        bloque = (
            "\n## Enmienda pre-ejecucion 3 (2026-09-30)\n\n"
            "- **Fidelidad de la representacion**: el pre-registro de V34 congelo "
            "`token_pattern` con dobles backslashes (typo mio), de modo que el\n  sha256 del dict "
            "del vectorizador era `33311e5eddc0b0bb...` en lugar del canonico "
            "`9bae157614458dbd...`. Se restaura\n  la receta canonica de "
            "`v31_02_banco_representaciones.py`: `token_pattern = (?u)\\b\\w\\b` y el sha256 vuelve "
            "al valor canonico.\n- El check de no-deriva del runner (comparacion de su espejo "
            "con el JSON) fue quien lo detecto. No cambia ninguna otra\n  constante, ningun gate, "
            "ni la puerta, ni la regla de alpha. Sin resultados: pre-ejecucion.\n"
        )
        PREREG_MD.write_text(PREREG_MD.read_text(encoding="utf-8").rstrip("\n") + "\n" + bloque,
                             encoding="utf-8")
        print("  MD del pre-registro: seccion de enmienda 3 anadida", flush=True)

    if "Enmienda pre-ejecucion 3 de V34-E1" not in RUNLOG.read_text(encoding="utf-8"):
        with RUNLOG.open("a", encoding="utf-8") as f:
            f.write("\n---\n\n## Enmienda pre-ejecucion 3 de V34-E1 · " + FECHA + "\n\n"
                    "- `token_pattern` del vectorizador restaurado a la receta canonica de v31_02 "
                    "(el pre-registro de V34 lo\n  congelo con un typo de dobles backslashes); "
                    "sha256 del dict: 33311e5e -> 9bae1576 (canonico). Detectado por el check\n  "
                    "de no-deriva del runner. Sin resultados.\n")
        print("  RUNLOG.md: enmienda 3 anotada", flush=True)

    lineas = EXPERIMENTS.read_text(encoding="utf-8").splitlines()
    cols = [c.strip() for c in lineas[0].split(",")]
    if "V34-E1-PREREG-ENMIENDA-3" not in {lin.split(",", 1)[0] for lin in lineas[1:] if lin}:
        fila = {"experiment_id": "V34-E1-PREREG-ENMIENDA-3", "fecha": FECHA, "bloque": "34.0",
                "etapa": "Enmienda pre-ejecucion 3: token_pattern del vectorizador a la receta "
                        "canonica",
                "pregunta": "el vectorizador congelado de V34 es el canonico del proyecto?",
                "hipotesis": "no: el pre-registro lo congelo con un typo de dobles backslashes",
                "features": "n/a", "preprocessing": "n/a", "representacion": "n/a", "modelo": "n/a",
                "hiperparametros": "token_pattern = (?u)\\b\\w\\b; sha256 del dict 9bae1576",
                "validacion": "sha256 recalculado y contrastado con el canonico de v31_02",
                "macro_f1_mean": "", "macro_f1_std": "", "weighted_f1_mean": "",
                "micro_f1_mean": "", "accuracy_mean": "",
                "resultado": "enmienda registrada; sin resultados del experimento",
                "interpretacion": "correccion de fidelidad de la representacion, no de conveniencia",
                "decision": "ejecutar V34-E1 con la enmienda",
                "duracion_seg": "0.0",
                "artefactos": "experiments/v34_00_preregistro.json (campo enmiendas) | "
                              "reports/v34_preregistro.md"}
        buf = io.StringIO()
        csv.writer(buf, lineterminator="\n").writerow([fila[c] for c in cols])
        with EXPERIMENTS.open("a", encoding="utf-8") as f:
            f.write(buf.getvalue())
        print("  experiments.csv: fila V34-E1-PREREG-ENMIENDA-3", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
