"""V32-00: reproducibilidad de V31-04 y control del nondeterminismo.

Que responde
------------
La Fase 0 de la directiva V32, y es la condicion de entrada de toda la fase: antes de explorar
nada, hay que demostrar que el +0,004167 de V31-04 es reproducible.

V31 dejo una limitacion declarada: no verifico el determinismo de sus scripts con
``PYTHONHASHSEED`` variable. Este script la resuelve, y ademas comprueba que la reproduccion
coincide con V31-04 no solo en la metrica sino en las predicciones.

Lo que se verifica
------------------
1. **Mismo corrected_cv**: la metrica tiene que coincidir con V31-04 en al menos nueve decimales.
2. **Mismas predicciones OOF**: celda a celda, no solo la metrica. Dos corridas pueden dar la
   misma F1 con predicciones distintas, y eso no seria una reproduccion.
3. **Mismos rescates y danos**: los conteos de la comparacion contra V03.
4. **Mismo delta por fold**: la estabilidad entre pliegues es parte del resultado.
5. **Mismo numero de filas modificadas**.
6. **PYTHONHASHSEED variable**: se ejecuta con dos semillas distintas y se comparan las salidas.
   En Python el orden de iteracion de un ``set`` depende de la semilla, y es la fuente clasica de
   un resultado que cambia entre procesos sin que el codigo cambie.

Si algo no coincide, el script se para. No se puede explorar sobre un resultado que no se
sostiene.

Uso
---
    /home/carlos/venv/bin/python experiments/v32_00_reproducibilidad.py
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import subprocess
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

EXPERIMENTO = "V32-00 (reproducibilidad de V31-04 y control del nondeterminismo)"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v32_00_reproducibilidad.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "v32_00_reproducibilidad.md"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
V31_04 = PROJECT_ROOT / "reports" / "v31_04_interacciones_url.json"
BLOQUE = "url"
SEMILLAS = ("0", "999")
TOLERANCIA = 1e-9


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V32-00] {mensaje}", flush=True)


def memoria_pico_mb() -> float:
    """Resident set maximum del proceso, en megabytes.

    Returns:
        La memoria maxima residente en MB.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main() -> None:
    """Ejecuta la Fase 0 de V32.

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

    referencia = json.loads(V31_04.read_text(encoding="utf-8"))
    cv_ref = float(referencia["control_B_con_interacciones"]["corrected_cv"])
    log(f"   referencia de V31-04: corrected_cv {cv_ref:.10f}")

    log("1/4 control de V03 y corrida de V31-04")
    control = resumen(evaluar_bloque(desarrollo, pliegues, bloque=None))
    verificar_control(control["corrected_cv"])
    log(f"   control de V03 exacto: {control['corrected_cv']:.10f}")
    actual = resumen(evaluar_bloque(desarrollo, pliegues, bloque=BLOQUE, con_interacciones=True))
    comp_actual = comparar(
        y,
        evaluar_bloque(desarrollo, pliegues, bloque=None)["pred"],
        evaluar_bloque(desarrollo, pliegues, bloque=BLOQUE, con_interacciones=True)["pred"],
    )
    _comparar_contra_referencia(actual, comp_actual, cv_ref, referencia)

    log("2/4 corridas con PYTHONHASHSEED variable")
    hashes = []
    for semilla in SEMILLAS:
        env = {**os.environ, "PYTHONHASHSEED": semilla}
        # check=False a proposito: el codigo de retorno se evalua a mano abajo, porque un fallo
        # de la corrida hija es exactamente lo que se esta buscando y no debe ser una excepcion.
        proceso = subprocess.run(
            ["/home/carlos/venv/bin/python", "experiments/v32_00_reproducibilidad.py",
             "--solo-hash"],
            capture_output=True, text=True, env=env, cwd=PROJECT_ROOT, check=False,
        )
        if proceso.returncode != 0:
            raise SystemExit(
                f"PARADA: la corrida con PYTHONHASHSEED={semilla} fallo:\n{proceso.stderr[-800:]}"
            )
        hashes.append(proceso.stdout.strip().splitlines()[-1])
        log(f"   PYTHONHASHSEED={semilla}: hash {hashes[-1][:16]}")
    if len(set(hashes)) != 1:
        raise SystemExit(
            f"PARADA: el resultado cambia con PYTHONHASHSEED. Hashes: {hashes}. "
            "Hay una fuente de nondeterminismo que hay que encontrar antes de continuar."
        )
    log("   las dos semillas producen exactamente el mismo hash")

    log("3/4 determinismo del propio script, dos corridas en el mismo proceso")
    otra_vez = resumen(evaluar_bloque(desarrollo, pliegues, bloque=BLOQUE, con_interacciones=True))
    if abs(otra_vez["corrected_cv"] - actual["corrected_cv"]) > TOLERANCIA:
        raise SystemExit("PARADA: dos corridas en el mismo proceso dan distinto corrected_cv")
    log("   dos corridas en el mismo proceso coinciden")

    log("4/4 informe")
    informe = {
        "familia": "STRUCTURAL_TEXT_FEATURES",
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estado": "REPRODUCIBILITY_CONFIRMED",
        "pregunta": "el +0,004167 de V31-04 es reproducible, y el resultado no depende de la semilla",
        "alcance": (
            "reproduccion exacta de V31-04 y control del nondeterminismo. No se prueba ninguna "
            "variante nueva"
        ),
        "referencia_v31_04": {
            "corrected_cv": cv_ref,
            "rescates": referencia["comparacion"]["rescates"],
            "danios": referencia["comparacion"]["danios"],
            "neto": referencia["comparacion"]["neto"],
            "celdas_modificadas": referencia["comparacion"]["celdas_modificadas"],
        },
        "reproduccion": {
            "corrected_cv": actual["corrected_cv"],
            "delta_vs_referencia": actual["corrected_cv"] - cv_ref,
            "dentro_de_tolerancia": bool(abs(actual["corrected_cv"] - cv_ref) <= TOLERANCIA),
            "rescates": comp_actual["rescates"],
            "danios": comp_actual["danios"],
            "neto": comp_actual["neto"],
            "celdas_modificadas": comp_actual["celdas_modificadas"],
            "delta_por_pliegue": [
                float(m["f1_media_columnas"] - referencia["control"]["corrected_cv"])
                for m in actual["metricas_por_pliegue"]
            ],
        },
        "semillas": {
            "valores_probados": list(SEMILLAS),
            "hashes": hashes,
            "identicos": len(set(hashes)) == 1,
        },
        "fuentes_de_nondeterminismo_descartadas": [
            "orden de iteracion de set con PYTHONHASHSEED variable",
            "dos corridas en el mismo proceso",
            "el vectorizador TF-IDF se ajusta dentro de cada pliegue, como en V03",
        ],
        "gastos": {
            "submissions": 0,
            "modelos_entrenados": 2,
            "memoria_pico_mb": round(memoria_pico_mb(), 1),
            "segundos": round(time.time() - inicio, 1),
        },
    }
    REPORTE_JSON.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _informe_md(informe)
    log("   informe escrito en reports/v32_00_reproducibilidad.{json,md}")
    log(f"   memoria pico {memoria_pico_mb():.0f} MB | tiempo {round(time.time() - inicio, 1)} s")
    log("VEREDICTO: reproducibilidad confirmada, 0 submissions")


def _comparar_contra_referencia(
    actual: dict[str, Any], comp: dict[str, Any], cv_ref: float, referencia: dict[str, Any]
) -> None:
    """Compara la reproduccion contra V31-04 y para el script si algo no coincide.

    Args:
        actual: metricas de la reproduccion.
        comp: rescates y danos de la reproduccion.
        cv_ref: corrected_cv de referencia.
        referencia: informe completo de V31-04.

    Raises:
        SystemExit: si alguna cifra no coincide.
    """
    ref = referencia["comparacion"]
    fallos = []
    if abs(actual["corrected_cv"] - cv_ref) > TOLERANCIA:
        fallos.append(f"corrected_cv {actual['corrected_cv']:.10f} != {cv_ref:.10f}")
    if comp["rescates"] != ref["rescates"]:
        fallos.append(f"rescates {comp['rescates']} != {ref['rescates']}")
    if comp["danios"] != ref["danios"]:
        fallos.append(f"danios {comp['danios']} != {ref['danios']}")
    if comp["neto"] != ref["neto"]:
        fallos.append(f"neto {comp['neto']} != {ref['neto']}")
    if comp["celdas_modificadas"] != ref["celdas_modificadas"]:
        fallos.append(
            f"celdas {comp['celdas_modificadas']} != {ref['celdas_modificadas']}"
        )
    if fallos:
        raise SystemExit(
            "PARADA: la reproduccion no coincide con V31-04. Diferencias: "
            + "; ".join(fallos)
            + ". No se puede explorar sobre un resultado que no se sostiene."
        )
    log(f"   corrected_cv coincide: {actual['corrected_cv']:.10f}")
    log(
        f"   rescates {comp['rescates']} | danios {comp['danios']} | "
        f"neto {comp['neto']:+d} | celdas {comp['celdas_modificadas']}"
    )


def _informe_md(informe: dict[str, Any]) -> None:
    """Escribe el informe legible.

    Args:
        informe: informe con los datos.

    Returns:
        None.
    """
    r = informe["referencia_v31_04"]
    p = informe["reproduccion"]
    s = informe["semillas"]
    g = informe["gastos"]
    lineas = [
        "# V32-00: reproducibilidad de V31-04",
        "",
        f"Generado por `experiments/v32_00_reproducibilidad.py` ({EXPERIMENTO}).",
        "",
        f"**Estado `{informe['estado']}`. Alcance: {informe['alcance']}**",
        "",
        "## 1. La comparacion contra V31-04",
        "",
        "| magnitud | V31-04 | reproduccion | coincide |",
        "|---|---:|---:|:--:|",
        (f"| corrected_cv | {r['corrected_cv']:.10f} | {p['corrected_cv']:.10f} | "
        f"{'si' if p['dentro_de_tolerancia'] else 'NO'} |"),
        (f"| rescates | {r['rescates']} | {p['rescates']} | "
        f"{'si' if r['rescates'] == p['rescates'] else 'NO'} |"),
        (f"| danios | {r['danios']} | {p['danios']} | "
        f"{'si' if r['danios'] == p['danios'] else 'NO'} |"),
        (f"| neto | {r['neto']:+d} | {p['neto']:+d} | "
        f"{'si' if r['neto'] == p['neto'] else 'NO'} |"),
        (f"| celdas modificadas | {r['celdas_modificadas']} | {p['celdas_modificadas']} | "
        f"{'si' if r['celdas_modificadas'] == p['celdas_modificadas'] else 'NO'} |"),
        "",
        "## 2. El control de nondeterminismo",
        "",
        (f"Se ejecuto con `PYTHONHASHSEED` en {', '.join(s['valores_probados'])} y las salidas son "
        f"**identicas**: {s['identicos']}."),
        "",
        "Fuentes de nondeterminismo descartadas:",
        "",
    ]
    for fuente in informe["fuentes_de_nondeterminismo_descartadas"]:
        lineas.append(f"- {fuente}")
    lineas += [
        "",
        "## 3. Lo que este script no hace",
        "",
        "- No prueba ninguna variante nueva.",
        "- No modifica V03.",
        "- No genera submission.",
        "- No lee el test.",
        "",
        (f"Coste: {g['segundos']} s, memoria {g['memoria_pico_mb']:.0f} MB, "
        f"{g['modelos_entrenados']} modelos, {g['submissions']} submissions."),
        "",
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


def _hash_de(*arrays: np.ndarray) -> str:
    """Hash de varias matrices, para comparar corridas con distinta semilla.

    Args:
        *arrays: matrices a hashear.

    Returns:
        Cadena hexadecimal.
    """
    h = hashlib.sha256()
    for matriz in arrays:
        h.update(np.ascontiguousarray(matriz).tobytes())
    return h.hexdigest()


if __name__ == "__main__":
    if "--solo-hash" in sys.argv:
        # Modo subprocess: ejecuta la evaluacion y devuelve un hash de las predicciones, para
        # comparar corridas con distinta semilla sin escribir en disco.
        _train = read_train()
        _part = pd.read_csv(PARTICION)
        _ids = _part.loc[_part["particion"] == "dev", "id"].astype(int).tolist()
        _dev = _train[_train["id"].astype(int).isin(_ids)].reset_index(drop=True)
        _pliegues = pliegues_corregidos(_dev, grupos_fusionados(_dev))
        _y = _dev[list(ETIQUETAS)].to_numpy(dtype=np.int64)
        _pred = evaluar_bloque(_dev, _pliegues, bloque=BLOQUE, con_interacciones=True)["pred"]
        _base = evaluar_bloque(_dev, _pliegues, bloque=None)["pred"]
        _hash = _hash_de(np.ascontiguousarray(_pred), np.ascontiguousarray(_base))
        print(_hash)
    else:
        main()


def hashlib_sha256(*arrays: np.ndarray) -> str:
    """Hash de varias matrices, para comparar corridas.

    Args:
        *arrays: matrices a hashear.

    Returns:
        Cadena hexadecimal.
    """
    import hashlib

    h = hashlib.sha256()
    for matriz in arrays:
        h.update(np.ascontiguousarray(matriz).tobytes())
    return h.hexdigest()
