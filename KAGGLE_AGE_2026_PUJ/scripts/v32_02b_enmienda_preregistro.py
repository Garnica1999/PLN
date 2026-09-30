"""V32 - E2 (P2) - Enmienda pre-ejecucion 1 del pre-registro congelado.

## Motivo

En `catboost 1.2.10` la metrica `TotalF1` promedia por defecto **Weighted** (ponderada por
soporte), no la media no ponderada de F1 por clase. La intencion congelada del protocolo
(ml-architect: "media no ponderada de F1 por clase") es la **macro-F1**. Antes de ejecutar
el experimento se verifico con sondas (mini-modelos en /tmp) que:

- `eval_metric(..., "TotalF1")` == F1 *weighted* exacto;
- `eval_metric(..., "TotalF1:average=Macro")` == macro-F1 de sklearn exacto.

## Alcance

- `brazos.A1_catboost.config.eval_metric`: `"TotalF1"` -> `"TotalF1:average=Macro"`.
- Se registra la enmienda dentro del propio JSON (`enmiendas`) y se anade una seccion de
  enmienda al informe del pre-registro (el texto congelado original no se edita).
- NO hay ningun resultado del experimento (folds sin entrenar): la enmienda es
  pre-ejecucion. El resto del protocolo queda intacto.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREREG_JSON = ROOT / "experiments" / "v32_02_preregistro_catboost.json"
PREREG_MD = ROOT / "reports" / "v32_p2_preregistro.md"
RUNLOG = ROOT / "RUNLOG.md"
EXPERIMENTS = ROOT / "experiments.csv"
FECHA = "2026-09-30"
OBJETO = "eval_metric A1/A3 (TotalF1)"

ENMIENDA = {
    "fecha": FECHA,
    "objeto": OBJETO,
    "de": "TotalF1",
    "a": "TotalF1:average=Macro",
    "motivo": "En catboost 1.2.10 TotalF1 promedia Weighted por defecto; la intencion "
              "congelada es la media no ponderada de F1 por clase (macro-F1).",
    "evidencia": "sondas pre-ejecucion con mini-modelos: default == weighted exacto; "
                 "TotalF1:average=Macro == macro-F1 de sklearn exacto; el runner re-verifica "
                 "la equivalencia en el sanity del fold 0 (<=1e-6).",
    "estado_datos": "sin resultados; enmienda pre-ejecucion",
}


def main() -> int:
    datos = json.loads(PREREG_JSON.read_text(encoding="utf-8"))
    config = datos["brazos"]["A1_catboost"]["config"]
    antes = config["eval_metric"]
    config["eval_metric"] = "TotalF1:average=Macro"

    enmiendas = datos.setdefault("enmiendas", [])
    if not any(e.get("objeto") == OBJETO for e in enmiendas):
        enmiendas.append(ENMIENDA)
    PREREG_JSON.write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  JSON: eval_metric {antes!r} -> {config['eval_metric']!r}; "
          f"enmiendas={len(enmiendas)}", flush=True)

    texto = PREREG_MD.read_text(encoding="utf-8")
    if "## Enmienda pre-ejecucion 1" not in texto:
        bloque = (
            "\n## Enmienda pre-ejecucion 1 (2026-09-30)\n\n"
            "- **Objeto**: `eval_metric` de A1 (y A3, que sigue la config de A1).\n"
            "- **De**: `TotalF1`  ->  **A**: `TotalF1:average=Macro`.\n"
            "- **Motivo**: en catboost 1.2.10 `TotalF1` promedia *Weighted* por defecto; la\n"
            "  intencion congelada es la media no ponderada de F1 por clase (macro-F1).\n"
            "- **Evidencia**: sondas pre-ejecucion (mini-modelos): el default equivale a\n"
            "  weighted; `TotalF1:average=Macro` equivale exactamente a la macro-F1 de sklearn.\n"
            "  El runner re-verifica la equivalencia en el sanity del fold 0 de A1 (<=1e-6).\n"
            "- **Estado**: sin resultados del experimento; enmienda pre-ejecucion. El resto\n"
            "  del protocolo congelado queda intacto.\n"
        )
        PREREG_MD.write_text(texto.rstrip("\n") + "\n" + bloque, encoding="utf-8")
        print(f"  MD: seccion de enmienda anadida ({len(texto.splitlines())} -> "
              f"{len(PREREG_MD.read_text(encoding='utf-8').splitlines())} lineas)", flush=True)

    if "Enmienda pre-ejecucion de V32-E2" not in RUNLOG.read_text(encoding="utf-8"):
        with RUNLOG.open("a", encoding="utf-8") as f:
            f.write("\n---\n\n## Enmienda pre-ejecucion de V32-E2 (P2) · " + FECHA + "\n\n"
                    "- eval_metric de A1/A3: TotalF1 -> TotalF1:average=Macro (el default de "
                    "catboost 1.2.10\n  promedia Weighted; la intencion congelada es macro-F1). "
                    "Sin resultados; antes de ejecutar.\n")
        print("  RUNLOG.md: enmienda anotada", flush=True)

    lineas = EXPERIMENTS.read_text(encoding="utf-8").splitlines()
    cols = [c.strip() for c in lineas[0].split(",")]
    if "V32-E2-PREREG-ENMIENDA" not in {ln.split(",", 1)[0] for ln in lineas[1:] if ln}:
        fila = {
            "experiment_id": "V32-E2-PREREG-ENMIENDA", "fecha": FECHA, "bloque": "32.2",
            "etapa": "Enmienda pre-ejecucion del prereg V32-E2: eval_metric TotalF1 -> "
                     "TotalF1:average=Macro",
            "pregunta": "corregir la metrica de parada a la media no ponderada (macro-F1) "
                        "antes de ejecutar",
            "hipotesis": "el default Weighted de catboost 1.2.10 diffiere del objetivo "
                         "macro-F1",
            "features": "n/a", "preprocessing": "n/a", "representacion": "n/a", "modelo": "n/a",
            "hiperparametros": "eval_metric de A1/A3 corregido; resto intacto",
            "validacion": "sondas pre-ejecucion; sanity <=1e-6 en el runner",
            "macro_f1_mean": "", "macro_f1_std": "", "weighted_f1_mean": "",
            "micro_f1_mean": "", "accuracy_mean": "",
            "resultado": "enmienda registrada; sin resultados del experimento",
            "interpretacion": "correccion de nomenclatura de metrica, sin cambio de diseno",
            "decision": "ejecutar V32-E2 con la enmienda",
            "duracion_seg": "0.0",
            "artefactos": "experiments/v32_02_preregistro_catboost.json (campo enmiendas) | "
                          "reports/v32_p2_preregistro.md (seccion de enmienda)",
        }
        buf = io.StringIO()
        escritor = csv.writer(buf, lineterminator="\n")
        escritor.writerow([fila[c] for c in cols])
        with EXPERIMENTS.open("a", encoding="utf-8") as f:
            f.write(buf.getvalue())
        print("  experiments.csv: fila V32-E2-PREREG-ENMIENDA", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
