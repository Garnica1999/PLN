"""V31-02: Fase 2 de STRUCTURAL_TEXT_FEATURES. Bloque mencion / hashtag.

Control A: solo las features estructurales aditivas. Las interacciones son el Control B y solo
se prueban si este bloque muestra senal.

Uso
---
    /home/carlos/venv/bin/python experiments/v31_02_mencion_hashtag.py
"""

from __future__ import annotations

import json
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train
from src.splits import grupos_fusionados, pliegues_corregidos
from src.structural_eval import (
    ETIQUETAS,
    comparar,
    evaluar_bloque,
    resumen,
    verificar_control,
)
from src.structural_features import columnas

EXPERIMENTO = "V31-02 (Fase 2 de STRUCTURAL_TEXT_FEATURES: bloque mencion / hashtag)"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v31_02_mencion_hashtag.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "v31_02_mencion_hashtag.md"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
BLOQUE = "mencion"


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V31-02] {mensaje}", flush=True)


def memoria_pico_mb() -> float:
    """Resident set maximum del proceso, en megabytes.

    Returns:
        La memoria maxima residente en MB.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main() -> None:
    """Ejecuta la Fase 2 de STRUCTURAL_TEXT_FEATURES.

    Returns:
        None.
    """
    inicio = time.time()
    log(f"{EXPERIMENTO}. NINGUNA submission")

    train = read_train()
    particion = __import__("pandas").read_csv(PARTICION)
    ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
    desarrollo = train[train["id"].astype(int).isin(ids_dev)].reset_index(drop=True)
    pliegues = pliegues_corregidos(desarrollo, grupos_fusionados(desarrollo))
    y = desarrollo[list(ETIQUETAS)].to_numpy(dtype=np.int64)
    log(f"   desarrollo {len(desarrollo)} filas | {len(pliegues)} pliegues canonicos")

    log("1/3 control: reproducir V03 exactamente")
    control = resumen(evaluar_bloque(desarrollo, pliegues, bloque=None))
    verificar_control(control["corrected_cv"])
    log(f"   control exacto: {control['corrected_cv']:.10f}")

    log("2/3 challenger: V03 + bloque mencion/hashtag")
    ch = resumen(evaluar_bloque(desarrollo, pliegues, bloque=BLOQUE, con_interacciones=False))
    log(f"   corrected_cv {ch['corrected_cv']:.10f} | delta {ch['corrected_cv'] - control['corrected_cv']:+.10f}")

    log("3/3 rescates y danos")
    comp = comparar(y, evaluar_bloque(desarrollo, pliegues, bloque=None)["pred"],
                    evaluar_bloque(desarrollo, pliegues, bloque=BLOQUE)["pred"])
    log(f"   rescates {comp['rescates']} | danios {comp['danios']} | neto {comp['neto']:+d}")

    informe = {
        "familia": "STRUCTURAL_TEXT_FEATURES",
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estado": "OOF_EVALUATED",
        "bloque": BLOQUE,
        "fase": "2_MENCION_HASHTAG",
        "features": columnas(BLOQUE),
        "n_features": len(columnas(BLOQUE)),
        "control": control,
        "challenger": ch,
        "delta_oof": ch["corrected_cv"] - control["corrected_cv"],
        "comparacion": comp,
        "gastos": {"submissions": 0, "modelos_entrenados": 2,
                   "memoria_pico_mb": round(memoria_pico_mb(), 1),
                   "segundos": round(time.time() - inicio, 1)},
    }
    REPORTE_JSON.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _informe_md(informe)
    log("   informe escrito en reports/v31_02_mencion_hashtag.{json,md}")
    log(f"   memoria pico {memoria_pico_mb():.0f} MB | tiempo {round(time.time() - inicio, 1)} s")
    log("VEREDICTO: Fase 2 evaluada, 0 submissions")


def _informe_md(informe: dict[str, Any]) -> None:
    """Escribe el informe legible.

    Args:
        informe: informe con los datos.

    Returns:
        None.
    """
    c, ch, comp, g = informe["control"], informe["challenger"], informe["comparacion"], informe["gastos"]
    lineas = [
        "# V31-02: Fase 2 de STRUCTURAL_TEXT_FEATURES, bloque mencion / hashtag",
        "",
        f"Generado por `experiments/v31_02_mencion_hashtag.py` ({EXPERIMENTO}).",
        "",
        f"**Estado `{informe['estado']}`. Alcance: solo el bloque mencion/hashtag, sin interacciones.**",
        "",
        "## 1. Features",
        "",
        f"{informe['n_features']} features, todas de posicion y contexto estructural. No se usa el",
        "nombre del usuario ni el contenido semantico del hashtag.",
        "",
    ]
    for f in informe["features"]:
        lineas.append(f"- `{f}`")
    lineas += [
        "",
        "## 2. Resultado",
        "",
        "| etiqueta | F1 control | F1 challenger | delta |",
        "|---|---:|---:|---:|",
    ]
    for et in ETIQUETAS:
        lineas.append(
            f"| {et} | {c['f1_por_etiqueta'][et]:.4f} | {ch['f1_por_etiqueta'][et]:.4f} | "
            f"{ch['f1_por_etiqueta'][et] - c['f1_por_etiqueta'][et]:+.4f} |"
        )
    lineas += [
        "",
        (f"**corrected_cv {c['corrected_cv']:.10f} -> {ch['corrected_cv']:.10f} | "
        f"delta {informe['delta_oof']:+.10f}**"),
        "",
        f"train F1 {ch['train_f1']:.4f} | gap {ch['gap_train_oof']:+.4f}",
        "",
        "## 3. Rescates y danos",
        "",
        "| label | rescates | danios | neto | celdas |",
        "|---|---:|---:|---:|---:|",
    ]
    for et in ETIQUETAS:
        d = comp["por_label"][et]
        lineas.append(f"| {et} | {d['rescates']} | {d['danios']} | {d['neto']:+d} | {d['celdas_modificadas']} |")
    lineas += [
        "",
        f"**rescates {comp['rescates']} | danios {comp['danios']} | neto {comp['neto']:+d}**",
        "",
        "## 4. Lo que este script no hace",
        "",
        "- No prueba interacciones: eso es el Control B.",
        "- No combina bloques.",
        "- No genera submission. No lee el test.",
        "",
        (f"Coste: {g['segundos']} s, memoria {g['memoria_pico_mb']:.0f} MB, "
        f"{g['modelos_entrenados']} modelos, {g['submissions']} submissions."),
        "",
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
