"""V33 - E1 - Enmienda pre-ejecucion 1 del pre-registro congelado.

## Enmienda 1: banda de G1 (log_loss multiclase de train)

La banda congelada era `[0.25, 0.60]`. Es **inalcanzable para cualquier modelo sano** de este
problema: en D-083 el gemelo lineal sano (SGD sobre la misma TF-IDF) ya dio `log_loss` multiclase
**1.3568**, muy por encima del techo 0.60; el techo solo es alcanzable con predicciones casi
perfectas, que un modelo de 6 clases con macro-F1 ~0.5 no produce. Mantenerla cerraria el
experimento por construccion (veredicto `NO_ATRIBUIBLE_BY_GATE` garantizado) sin medir nada.

Nueva banda, **DERIVADA** (no calibrada con datos):
- suelo `0.05`: detector de humo. Con UNA sola pasada y una actualizacion por ejemplo no hay
  memorizacion posible, asi que un `ll_train` medio por debajo de 0.05 solo puede venir de una
  fuga o de un error, no de un modelo legitimo.
- techo `0.9 * log(6) = 1.6125835223052495`: el `ll_train` debe mejorar el uniforme en al menos
  un 10 % relativo. Caza con holgura la firma de divergencia de D-083 (11.72 -> 23.03) y el caso
  "no aprende nada" (1.7918 = uniforme queda fuera).

## Enmienda 2: `shuffle=False` explicito en el control A2

El pre-registro congelaba los kwargs de `SGDClassifier` sin fijar `shuffle`; el default de sklearn
es `True`, que aplica una permutacion propia `random_state=20261010+fold` **encima** del orden
congelado, rompiendo la condicion "mismo barajado" que el propio protocolo exige. Se fija
`shuffle=False` para que A2 consuma el orden congelado tal cual. No altera la sustancia del
control (misma matriz, misma pasada, misma semilla, mismo presupuesto).

## Estado

Ambas enmiendas son **PRE-EJECUCION**: no existe ningun resultado de V33-E1 (ningun fold
entrenado; el runner solo ha pasado `--verificar`). El resto del protocolo queda intacto.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREREG_JSON = ROOT / "experiments" / "v33_00_preregistro.json"
PREREG_MD = ROOT / "reports" / "v33_preregistro.md"
PUERTA_MD = ROOT / "reports" / "v33_convergence_gate.md"
RUNLOG = ROOT / "RUNLOG.md"
EXPERIMENTS = ROOT / "experiments.csv"
FECHA = "2026-09-30"

G1_NUEVO = [0.05, 1.6125835223052495]
ANCLAJE_NUEVO = (
    "suelo 0.05 = detector de humo (con una sola pasada y una actualizacion por ejemplo no hay "
    "memorizacion posible); techo 0.9*log(6) = 1.6125835223052495: el ll_train debe mejorar el "
    "uniforme en al menos un 10 % relativo. Enmenda 1 pre-ejecucion: la banda congelada "
    "[0.25, 0.60] era inalcanzable para cualquier modelo sano (en D-083 el gemelo sano dio "
    "log_loss multiclase 1.3568 > 0.60) y cerraba el experimento por construccion"
)
ENMIENDAS = [
    {
        "fecha": FECHA,
        "objeto": "puerta G1 banda de log_loss multiclase de train",
        "de": [0.25, 0.6],
        "a": G1_NUEVO,
        "motivo": "la banda congelada era inalcanzable para un modelo sano (techo 0.60 muy por "
                  "deajo del log_loss multiclase 1.3568 que ya midio el gemelo en D-083)",
        "derivacion": "suelo 0.05 = detector de humo (una pasada no memoriza); techo "
                      "0.9*log(6) = 1.6125835223052495 = mejorar el uniforme >= 10 % relativo",
        "estado_datos": "sin resultados; enmienda pre-ejecucion",
    },
    {
        "fecha": FECHA,
        "objeto": "a2_sgd_control.shuffle",
        "de": "(no congelado; default de sklearn True)",
        "a": False,
        "motivo": "con shuffle=True el control aplicaba su propia permutacion encima del orden "
                  "congelado, rompiendo la condicion 'mismo barajado' del propio protocolo",
        "estado_datos": "sin resultados; enmienda pre-ejecucion",
    },
]


def main() -> int:
    datos = json.loads(PREREG_JSON.read_text(encoding="utf-8"))

    g1 = datos["puerta_convergencia"]["umbrales_derivados"]["G1_ll_train_banda"]
    antes = list(g1["rango"])
    g1["rango"] = G1_NUEVO
    g1["anclaje"] = ANCLAJE_NUEVO
    print(f"  G1 banda: {antes} -> {G1_NUEVO}", flush=True)

    a2 = datos["a2_sgd_control"]
    antes_shuffle = a2.get("shuffle", "(no congelado; default True)")
    a2["shuffle"] = False
    print(f"  A2 shuffle: {antes_shuffle} -> False", flush=True)

    enmiendas = datos.setdefault("enmiendas", [])
    for e in ENMIENDAS:
        if not any(x.get("objeto") == e["objeto"] for x in enmiendas):
            enmiendas.append(e)
    PREREG_JSON.write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  JSON: enmiendas={len(enmiendas)}", flush=True)

    if "## Enmienda pre-ejecucion 1" not in PREREG_MD.read_text(encoding="utf-8"):
        bloque = (
            "\n## Enmienda pre-ejecucion 1 (2026-09-30)\n\n"
            "- **G1**: banda de log_loss multiclase de train `[0.25, 0.60]` -> "
            f"`[{G1_NUEVO[0]}, {G1_NUEVO[1]}]`. Motivo: la banda congelada era inalcanzable "
            "para un\n  modelo sano (en D-083 el gemelo sano dio log_loss multiclase 1.3568 > "
            "0.60) y cerraba el\n  experimento por construccion. Derivacion: suelo 0.05 = "
            "detector de humo; techo\n  0.9*log(6) = 1.6125835223052495 (mejorar el uniforme "
            ">= 10 % relativo).\n"
            "- **A2**: `shuffle` no estaba congelado (default de sklearn True) -> `False`, para "
            "que el control\n  consuma el orden congelado tal cual (el default aplicaba su "
            "propia permutacion encima).\n"
            "- Sin resultados: enmienda **pre-ejecucion** (ningun fold entrenado). Resto del "
            "protocolo intacto.\n"
        )
        PREREG_MD.write_text(PREREG_MD.read_text(encoding="utf-8").rstrip("\n") + "\n" + bloque,
                             encoding="utf-8")
        print("  MD del pre-registro: seccion de enmienda anadida", flush=True)

    texto_puerta = PUERTA_MD.read_text(encoding="utf-8")
    fila_vieja = ("| G1 | 0.25 <= log_loss multiclase de train <= 0.60 | uniforme `log(6)` = "
                  "1.791759469228055; la cota alta caza la divergencia 11.72 -> 23.03 de D-083 | "
                  "1e-9 |")
    fila_nueva = ("| G1 | 0.05 <= log_loss multiclase de train <= 1.6125835223052495 | suelo 0.05 "
                  "= detector de humo; techo 0.9*log(6) = 1.6125835223052495 (mejorar el uniforme "
                  ">= 10 % relativo) | 1e-9 |")
    if fila_vieja in texto_puerta:
        texto_puerta = texto_puerta.replace(fila_vieja, fila_nueva)
        print("  puerta: fila G1 actualizada", flush=True)
    if "## Enmienda pre-ejecucion 1" not in texto_puerta:
        texto_puerta += (
            "\n## Enmienda pre-ejecucion 1 (2026-09-30)\n\n"
            f"- La banda de **G1** pasa de `[0.25, 0.60]` a `[{G1_NUEVO[0]}, {G1_NUEVO[1]}]`.\n"
            "  Motivo: era inalcanzable para un modelo sano (el gemelo sano de D-083 dio "
            "log_loss multiclase\n  1.3568) y garantizaba `NO_ATRIBUIBLE_BY_GATE` sin medir "
            "nada. Derivacion del suelo y del techo\n  en la tabla. Sin resultados: pre-ejecucion.\n"
            "- El control **A2** fija `shuffle=False` para consumir el orden congelado tal cual.\n"
        )
    PUERTA_MD.write_text(texto_puerta, encoding="utf-8")
    print("  puerta: seccion de enmienda anadida", flush=True)

    if "Enmienda pre-ejecucion de V33-E1" not in RUNLOG.read_text(encoding="utf-8"):
        with RUNLOG.open("a", encoding="utf-8") as f:
            f.write("\n---\n\n## Enmienda pre-ejecucion de V33-E1 · " + FECHA + "\n\n"
                    "- G1: banda de log_loss [0.25, 0.60] -> [0.05, 1.6125835223052495] "
                    "(inalcanzable para un modelo sano; D-083 midio\n  1.3568 en el gemelo). "
                    "A2: shuffle=False para respetar el orden congelado. Sin resultados.\n")
        print("  RUNLOG.md: enmienda anotada", flush=True)

    lineas = EXPERIMENTS.read_text(encoding="utf-8").splitlines()
    cols = [c.strip() for c in lineas[0].split(",")]
    if "V33-E1-PREREG-ENMIENDA" not in {lin.split(",", 1)[0] for lin in lineas[1:] if lin}:
        fila = {
            "experiment_id": "V33-E1-PREREG-ENMIENDA", "fecha": FECHA, "bloque": "33.0",
            "etapa": "Enmienda pre-ejecucion del prereg V33-E1: banda G1 y shuffle del control",
            "pregunta": "la banda G1 congelada es alcanzable por un modelo sano?",
            "hipotesis": "no: el techo 0.60 esta muy por debajo del log_loss sano medido (1.3568)",
            "features": "n/a", "preprocessing": "n/a", "representacion": "n/a", "modelo": "n/a",
            "hiperparametros": "G1 banda -> [0.05, 1.6125835223052495]; A2 shuffle=False",
            "validacion": "derivacion de suelo y techo; sin resultados",
            "macro_f1_mean": "", "macro_f1_std": "", "weighted_f1_mean": "",
            "micro_f1_mean": "", "accuracy_mean": "",
            "resultado": "enmienda registrada; sin resultados del experimento",
            "interpretacion": "correccion de umbral inalcanzable y de ejecucion del control",
            "decision": "ejecutar V33-E1 con la enmienda",
            "duracion_seg": "0.0",
            "artefactos": "experiments/v33_00_preregistro.json (campo enmiendas) | "
                          "reports/v33_convergence_gate.md | reports/v33_preregistro.md",
        }
        buf = io.StringIO()
        csv.writer(buf, lineterminator="\n").writerow([fila[c] for c in cols])
        with EXPERIMENTS.open("a", encoding="utf-8") as f:
            f.write(buf.getvalue())
        print("  experiments.csv: fila V33-E1-PREREG-ENMIENDA", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
