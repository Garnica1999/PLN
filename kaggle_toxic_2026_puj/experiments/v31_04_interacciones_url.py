"""V31-04: Control B de STRUCTURAL_TEXT_FEATURES. Interacciones del bloque URL.

Que responde
------------
El Control A del bloque URL (V31-01) dio delta **+0.0002085130** con neto **+6**: la unica
senal positiva de los tres bloques, pero muy debil. La directiva manda probar el Control B,
interacciones explicitas, solo si el bloque muestra senal, y este es el unico que la muestra.

Las interacciones
-----------------
Se codifican como **cruces discretos**, no como productos de variables continuas de alta
cardinalidad. La longitud se agrupa en tres tramos (corto <= 12 tokens, medio <= 22, largo > 22)
y cada interaccion es una flag binaria con volumen controlado:

- `url_x_corto`, `url_x_medio`, `url_x_largo`
- `url_query_x_largo`
- `url_corta_x_corto`

Esto permite que el modelo lineal represente una condicion que V03 puro no expresa: el efecto
de una URL segun la longitud del texto, que es exactamente lo que encontro V30-01 (corto con URL
duplica el error de HS, 0,2542 frente a 0,1338).

Uso
---
    /home/carlos/venv/bin/python experiments/v31_04_interacciones_url.py
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
import pandas as pd

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
from src.structural_features import columnas, columnas_interaccion

EXPERIMENTO = "V31-04 (Control B de STRUCTURAL_TEXT_FEATURES: interacciones del bloque URL)"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v31_04_interacciones_url.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "v31_04_interacciones_url.md"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
BLOQUE = "url"


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V31-04] {mensaje}", flush=True)


def memoria_pico_mb() -> float:
    """Resident set maximum del proceso, en megabytes.

    Returns:
        La memoria maxima residente en MB.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main() -> None:
    """Ejecuta el Control B sobre el bloque URL.

    Returns:
        None.
    """
    inicio = time.time()
    log(f"{EXPERIMENTO}. NINGUNA submission")

    train = read_train()
    particion = pd.read_csv(PARTICION)
    ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
    desarrollo = train[train["id"].astype(int).isin(ids_dev)].reset_index(drop=True)
    pliegues = pliegues_corregidos(desarrollo, grupos_fusionados(desarrollo))
    y = desarrollo[list(ETIQUETAS)].to_numpy(dtype=np.int64)
    log(f"   desarrollo {len(desarrollo)} filas | {len(pliegues)} pliegues canonicos")

    log("1/4 control: reproducir V03 exactamente")
    control = resumen(evaluar_bloque(desarrollo, pliegues, bloque=None))
    verificar_control(control["corrected_cv"])
    log(f"   control exacto: {control['corrected_cv']:.10f}")

    log("2/4 Control A: solo las features del bloque (referencia de V31-01)")
    solo = resumen(evaluar_bloque(desarrollo, pliegues, bloque=BLOQUE, con_interacciones=False))
    log(f"   Control A: {solo['corrected_cv']:.10f} | delta {solo['corrected_cv'] - control['corrected_cv']:+.10f}")

    log("3/4 Control B: features + interacciones con la longitud")
    inter = resumen(evaluar_bloque(desarrollo, pliegues, bloque=BLOQUE, con_interacciones=True))
    log(f"   Control B: {inter['corrected_cv']:.10f} | delta {inter['corrected_cv'] - control['corrected_cv']:+.10f}")
    log(f"   delta del Control B sobre el Control A: {inter['corrected_cv'] - solo['corrected_cv']:+.10f}")

    log("4/4 rescates y danos del Control B")
    comp = comparar(y, evaluar_bloque(desarrollo, pliegues, bloque=None)["pred"],
                    evaluar_bloque(desarrollo, pliegues, bloque=BLOQUE, con_interacciones=True)["pred"])
    log(f"   rescates {comp['rescates']} | danios {comp['danios']} | neto {comp['neto']:+d}")

    informe = {
        "familia": "STRUCTURAL_TEXT_FEATURES",
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estado": "OOF_EVALUATED",
        "bloque": BLOQUE,
        "fase": "4_INTERACCIONES_URL",
        "features_base": columnas(BLOQUE),
        "features_interaccion": columnas_interaccion(BLOQUE),
        "n_features_total": len(columnas(BLOQUE)) + len(columnas_interaccion(BLOQUE)),
        "control": control,
        "control_A_solo_features": solo,
        "control_B_con_interacciones": inter,
        "delta_B_menos_A": inter["corrected_cv"] - solo["corrected_cv"],
        "delta_oof": inter["corrected_cv"] - control["corrected_cv"],
        "comparacion": comp,
        "gastos": {"submissions": 0, "modelos_entrenados": 3,
                   "memoria_pico_mb": round(memoria_pico_mb(), 1),
                   "segundos": round(time.time() - inicio, 1)},
    }
    REPORTE_JSON.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _informe_md(informe)
    log("   informe escrito en reports/v31_04_interacciones_url.{json,md}")
    log(f"   memoria pico {memoria_pico_mb():.0f} MB | tiempo {round(time.time() - inicio, 1)} s")
    log("VEREDICTO: Control B evaluado, 0 submissions")


def _informe_md(informe: dict[str, Any]) -> None:
    """Escribe el informe legible.

    Args:
        informe: informe con los datos.

    Returns:
        None.
    """
    c, a, b, comp, g = (informe["control"], informe["control_A_solo_features"],
                        informe["control_B_con_interacciones"], informe["comparacion"],
                        informe["gastos"])
    lineas = [
        "# V31-04: Control B de STRUCTURAL_TEXT_FEATURES, interacciones del bloque URL",
        "",
        f"Generado por `experiments/v31_04_interacciones_url.py` ({EXPERIMENTO}).",
        "",
        f"**Estado `{informe['estado']}`. Alcance: interacciones del bloque URL, que es el unico de",
        "los tres que mostro senal positiva en el Control A.**",
        "",
        "## 1. Las interacciones",
        "",
        f"{informe['n_features_total']} features en total: {len(informe['features_base'])} del bloque y",
        f"{len(informe['features_interaccion'])} de interaccion. Se codifican como cruces discretos,",
        "no como productos de variables continuas de alta cardinalidad. La longitud se agrupa en",
        "tres tramos (corto <= 12 tokens, medio <= 22, largo > 22).",
        "",
    ]
    for f in informe["features_interaccion"]:
        lineas.append(f"- `{f}`")
    lineas += [
        "",
        "## 2. Resultado",
        "",
        "| variante | corrected_cv | delta vs V03 |",
        "|---|---:|---:|",
        f"| V03 (control) | {c['corrected_cv']:.10f} | 0 |",
        f"| Control A: solo features | {a['corrected_cv']:.10f} | {a['corrected_cv'] - c['corrected_cv']:+.10f} |",
        f"| Control B: + interacciones | {b['corrected_cv']:.10f} | {b['corrected_cv'] - c['corrected_cv']:+.10f} |",
        "",
        f"**El Control B aporta {informe['delta_B_menos_A']:+.10f} sobre el Control A.**",
        "",
        "| etiqueta | F1 control | F1 Control B | delta |",
        "|---|---:|---:|---:|",
    ]
    for et in ETIQUETAS:
        lineas.append(
            f"| {et} | {c['f1_por_etiqueta'][et]:.4f} | {b['f1_por_etiqueta'][et]:.4f} | "
            f"{b['f1_por_etiqueta'][et] - c['f1_por_etiqueta'][et]:+.4f} |"
        )
    lineas += [
        "",
        f"train F1 {b['train_f1']:.4f} | gap {b['gap_train_oof']:+.4f}",
        "",
        "## 3. Rescates y danos del Control B",
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
        "- No prueba interacciones de mencion ni de enfasis: sus Controles A fueron negativos.",
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
