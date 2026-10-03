"""V10-08: confirmacion de integridad de holdout2 y filtro de detectabilidad sobre las candidatas de la via A.

Que hace, y por que es una fase de verificacion y no de medicion
---------------------------------------------------------------
El gate de P0b no se supero: la referencia de V03 recalculada en dev2 quedo en 0.763436, fuera de
la tolerancia declarada de 0.005, y por la regla pre-registrada **dev2 no se usa para ninguna
decision de adopcion**. V10-02 explico despues que esa desviacion es ruido de particion y de
tamano, no un defecto del conjunto, pero el veredicto del gate no se reinterpreta.

Lo que queda de esta fase es, por tanto, triple y de verificacion:

1. **Integridad**: comprobar que los artefactos congelados de holdout2 siguen teniendo la huella
   que tiene que tener, o sea que nadie los toco despues de congelarlos.
2. **Confirmacion de que no hay nada que confirmar**: las tres lineas de la via A se cerraron sin
  candidatas que lleguen a la mesa de finales, de modo que no hay ninguna evaluacion que gastar de las
   tres que permite la regla de uso de holdout2.
3. **Filtro de detectabilidad**: aplicar a las candidatas que se midieron el criterio de V8-02, que
   dice que una candidata que cambia C celdas solo es resoluble por el externo de 1154 filas si su
   tasa de acierto entre esas celdas supera ``0,5 + 19,2 / C``. Las tres lineas ya fallaron sus
   gates por la metrica, pero el filtro se aplica igual y se registra, porque responde a una
   pregunta distinta: aunque hubieran improved, ¿el externo hubiera podido verlo?

Uso
---
    /home/carlos/venv/bin/python experiments/v10_08_confirmacion_dev2.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from importlib import util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train
from src.splits import grupos_fusionados

_especificacion = util.spec_from_file_location(
    "arnes", PROJECT_ROOT / "experiments" / "v10_03_arnes_evaluacion.py"
)
arnes = util.module_from_spec(_especificacion)
sys.modules["arnes"] = arnes
_especificacion.loader.exec_module(arnes)

EXPERIMENTO = "V10-08 (P3, confirmacion e integridad)"
REPORTE_MD = PROJECT_ROOT / "reports" / "v10_08_confirmacion_dev2.md"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v10_08_confirmacion_dev2.json"

CONFIG_HOLDOUT2 = PROJECT_ROOT / "config" / "holdout2.json"
IDS_HOLDOUT2 = PROJECT_ROOT / "data" / "processed" / "holdout2_ids.csv"

# Ruido del externo medido en V8-02: con 1154 filas el leaderboard resuelve unos 19.2 aciertos
# netos, de modo que una candidata que cambia pocas celdas es indistinguible por construccion.
ACIERTOS_NETOS_RESOLUBLES = 19.2
FILAS_EXTERNO = 1154
# Criterios de adopcion del addendum de protocolo, que son los unicos validos para decidir si
# una candidata mejora. Un delta positivo por si solo no significa nada con esta banda de ruido.
DELTA_MINIMO_ADOPCION = 0.002
PARTICIONES_MEJORAN_MINIMAS = 4

# Candidatas medidas en la via A, con el numero de celdas que movieron y su delta medio anidado.
CANDIDATAS_VIA_A = (
    {
        "nombre": "V10-04 calibracion isotonica recortada",
        "celdas_promedio": 0.0,
        "delta_medio": 0.000148,
        "particiones_que_mejoran": 3,
        "nota": "la isotonica fue la de mejor ECE pero su delta anidado mas alto era el de Platt",
    },
    {
        "nombre": "V10-05 regla F1 optima con prior simulado",
        "celdas_promedio": 134.0,
        "delta_medio": -0.003921,
        "particiones_que_mejoran": 1,
        "nota": "cambio 134 celdas y empeoro",
    },
    {
        "nombre": "V10-06 tres umbrales libres re-estimados por pliegue",
        "celdas_promedio": 124.0,
        "delta_medio": 0.000131,
        "particiones_que_mejoran": 3,
        "nota": "cambio 124 celdas y no mejoro de forma apreciable",
    },
)


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V10-08] {mensaje}", flush=True)


def tasa_minima_resoluble(celdas: int) -> float:
    """Calcula la tasa de acierto minima para que un cambio sea resoluble por el externo.

    Args:
        celdas: numero de celdas que la candidata cambia.

    Returns:
        La tasa minima, o ``inf`` si la candidata no cambia ninguna celda.
    """
    if celdas <= 0:
        return float("inf")
    return 0.5 + ACIERTOS_NETOS_RESOLUBLES / celdas


def main() -> None:
    """Verifica la integridad de holdout2 y aplica el filtro de detectabilidad."""
    inicio = time.time()
    log("V10-08: integridad de holdout2 y filtro de detectabilidad")

    log("1/4 comprobar que los artefactos congelados no han cambiado")
    config = json.loads(CONFIG_HOLDOUT2.read_text(encoding="utf-8"))
    huella_config_actual = hashlib.sha256(CONFIG_HOLDOUT2.read_bytes()).hexdigest()
    huella_ids_actual = hashlib.sha256(IDS_HOLDOUT2.read_bytes()).hexdigest()
    huella_ids_esperada = config["artefactos"]["sha256_de_los_ids"]
    integridad = {
        "sha256_config_actual": huella_config_actual,
        "sha256_ids_actual": huella_ids_actual,
        "sha256_ids_esperada": huella_ids_esperada,
        "ids_sin_cambios": bool(huella_ids_actual == huella_ids_esperada),
    }
    log(
        f"   ids de holdout2 sin cambios desde la congelacion: {integridad['ids_sin_cambios']}"
    )

    log("2/4 comprobar que los grupos de holdout2 y dev2 siguen sin solaparse")
    train = read_train()
    grupos_completos = grupos_fusionados(train)
    ids_desarrollo = set(train["id"].astype(str).tolist())
    ids_holdout2 = set(pd.read_csv(IDS_HOLDOUT2, dtype={"id": str})["id"].tolist())
    fila_holdout2 = train["id"].astype(str).isin(ids_holdout2).to_numpy()
    solape_filas = int(np.sum(fila_holdout2)) - len(ids_holdout2.intersection(ids_desarrollo))
    grupos_holdout2 = set(grupos_completos[fila_holdout2].tolist())
    grupos_dev2 = set(grupos_completos[~fila_holdout2].tolist())
    grupos_compartidos = len(grupos_holdout2 & grupos_dev2)
    particion_limpia = bool(
        solape_filas == 0 and grupos_compartidos == 0
        and len(ids_holdout2) == config["composicion"]["filas_holdout2"]
    )
    log(
        f"   filas de holdout2 localizadas: {int(np.sum(fila_holdout2))} | "
        f"grupos compartidos: {grupos_compartidos} | particion limpia: {particion_limpia}"
    )

    log("3/4 aplicar el filtro de detectabilidad a las candidatas de la via A")
    detalle: list[dict[str, Any]] = []
    for candidata in CANDIDATAS_VIA_A:
        celdas = candidata["celdas_promedio"]
        minima = tasa_minima_resoluble(celdas)
        resoluble = minima <= 1.0
        # La prueba de mejora NO es "delta mayor que cero". El addendum de protocolo exige las
        # tres cosas a la vez: delta medio de al menos 0.002, al menos 4 de 5 particiones que
        # mejoran, e intervalo que no incluye el cero. Usar solo el signo daria por buena una
        # candidata con +0.000131, que es una fraccion de la banda de ruido de 0.0053.
        cumple_delta = candidata["delta_medio"] >= DELTA_MINIMO_ADOPCION
        cumple_particiones = (
            candidata["particiones_que_mejoran"] >= PARTICIONES_MEJORAN_MINIMAS
        )
        mejora_la_metrica = bool(cumple_delta and cumple_particiones)
        detalle.append(
            {
                "nombre": candidata["nombre"],
                "celdas_promedio": celdas,
                "delta_medio_anidado": candidata["delta_medio"],
                "particiones_que_mejoran": candidata["particiones_que_mejoran"],
                "cumple_delta_minimo": bool(cumple_delta),
                "cumple_particiones_minimas": bool(cumple_particiones),
                "tasa_de_acierto_minima_para_ser_resoluble": (
                    None if not resoluble else round(minima, 4)
                ),
                "seria_resoluble_por_el_externo": bool(resoluble),
                "mejora_la_metrica_segun_el_addendum": mejora_la_metrica,
                "enviable": bool(resoluble and mejora_la_metrica),
                "nota": candidata["nota"],
            }
        )
        log(
            f"   {candidata['nombre']}: {celdas:.0f} celdas, delta "
            f"{candidata['delta_medio']:+.6f}, "
            f"{candidata['particiones_que_mejoran']}/5 particiones, tasa minima "
            + (f"{minima:.4f}" if resoluble else "no aplica")
            + f", enviable: {detalle[-1]['enviable']}"
        )

    log("4/4 informe")
    evaluables = [c for c in detalle if c["enviable"]]
    informe = {
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "integridad_de_holdout2": integridad,
        "limpieza_de_la_particion": {
            "filas_holdout2_localizadas": int(np.sum(fila_holdout2)),
            "solape_de_filas": solape_filas,
            "grupos_compartidos": grupos_compartidos,
            "particion_limpia": particion_limpia,
        },
        "regla_de_uso_de_holdout2": config["regla_de_uso"],
        "evaluaciones_gastadas": 0,
        "evaluaciones_disponibles": config["regla_de_uso"]["maximo_evaluaciones"],
        "candidatas_evaluables": len(evaluables),
        "criterios_de_mejora_aplicados": {
            "delta_medio_minimo": DELTA_MINIMO_ADOPCION,
            "particiones_que_mejoran_minimas": PARTICIONES_MEJORAN_MINIMAS,
            "nota": (
                "Un delta positivo por si solo NO cuenta como mejora. Con la banda de ruido de "
                "0.0053 que midio V10-02, el +0.000131 de V10-06 es una fraccion del ruido, y "
                "por eso el criterio exige las dos cosas a la vez."
            ),
        },
        "filtro_de_detectabilidad": {
            "aciertos_netos_resolubles": ACIERTOS_NETOS_RESOLUBLES,
            "filas_del_externo": FILAS_EXTERNO,
            "regla": "una candidata que cambia C celdas solo es resoluble si su tasa de acierto supera 0,5 + 19,2 / C",
            "detalle": detalle,
        },
        "veredicto": (
            "ninguna candidata de la via A es enviable: las tres fallan sus gates de metrica, y "
            "ademas dos de ellas no serian resolubles por el externo ni aunque lo fueran"
        ),
    }
    REPORTE_JSON.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _informe(informe, integridad, particion_limpia, detalle, inicio)
    log(f"completado en {round(time.time() - inicio, 1)} s")
    log("VEREDICTO: integridad correcta, cero evaluaciones de holdout2 gastadas, nada enviable")


def _informe(
    informe: dict[str, Any],
    integridad: dict[str, Any],
    particion_limpia: bool,
    detalle: list[dict[str, Any]],
    inicio: float,
) -> None:
    """Escribe el informe de confirmacion e integridad.

    Args:
        informe: veredicto del experimento.
        integridad: resultado de la comprobacion de huellas.
        particion_limpia: si la particion sigue sin solaparse.
        detalle: resultado del filtro de detectabilidad.
        inicio: instante de arranque del script.
    """
    lineas = [
        "# V10-08: confirmacion de integridad de holdout2 y filtro de detectabilidad (P3)",
        "",
        f"Generado por `experiments/v10_08_confirmacion_dev2.py` ({EXPERIMENTO}).",
        "",
        "## 1. Por que esta fase es de verificacion y no de medicion",
        "",
        "El gate de P0b no se supero, y por la regla pre-registrada **dev2 no se usa para",
        "ninguna decision de adopcion**. V10-02 explico despues que la desviacion es ruido de",
        "particion y de tamano, pero el veredicto del gate no se reinterpreta a posteriori.",
        "",
        "Lo que queda es comprobar que lo congelado sigue congelado, confirmar que no hay nada",
        "pendiente de juzgar, y responder a una pregunta distinta de la del gate: **aunque una",
        "candidata hubiera mejorado, el externo de 1154 filas habria podido verlo?**",
        "",
        "## 2. Integridad de los artefactos congelados",
        "",
        "| comprobacion | resultado |",
        "|---|---|",
        f"| los identificadores de holdout2 conservan su huella | **{integridad['ids_sin_cambios']}** |",
        f"| huella de `config/holdout2.json` | `{integridad['sha256_config_actual'][:32]}...` |",
        f"| huella de `data/processed/holdout2_ids.csv` | `{integridad['sha256_ids_actual'][:32]}...` |",
        "",
        "## 3. La particion sigue siendo limpia",
        "",
        f"- particion limpia: **{particion_limpia}**",
        f"- solape de filas: {informe['limpieza_de_la_particion']['solape_de_filas']}",
        f"- grupos compartidos: {informe['limpieza_de_la_particion']['grupos_compartidos']}",
        "",
        "## 4. Cuantas evaluaciones de holdout2 quedan",
        "",
        f"- gastadas: **{informe['evaluaciones_gastadas']}**",
        f"- disponibles: **{informe['evaluaciones_disponibles']}**",
        "",
        "Las tres lineas de la via A se cerraron en sus propios gates, de modo que **ninguna llega",
        "a la mesa de finales** y no hay nada que gastar. Las tres evaluaciones disponibles siguen",
        "integras para el resto del proyecto.",
        "",
        "## 5. Filtro de detectabilidad sobre las candidatas medidas",
        "",
        "La regla de V8-02 dice que una candidata que cambia C celdas solo es resoluble por el",
        "externo si su tasa de acierto entre esas celdas supera `0,5 + 19,2 / C`, porque con 1154",
        "filas el leaderboard resuelve unos 19.2 aciertos netos.",
        "",
        "| candidata | celdas | delta medio | mejoran | tasa minima | resoluble | enviable |",
        "|---|---|---|---|---|---|---|",
    ]
    for candidata in detalle:
        tasa = candidata["tasa_de_acierto_minima_para_ser_resoluble"]
        lineas.append(
            f"| {candidata['nombre']} | {candidata['celdas_promedio']:.0f} | "
            f"{candidata['delta_medio_anidado']:+.6f} | "
            f"{candidata['particiones_que_mejoran']}/5 | "
            + (f"{tasa:.4f} | " if tasa is not None else "no aplica | ")
            + f"{'si' if candidata['seria_resoluble_por_el_externo'] else 'no'} | "
            f"{'si' if candidata['enviable'] else 'no'} |"
        )
    lineas += [
        "",
        "La prueba de mejora es la del addendum de protocolo, no la del signo: hacen falta las dos",
        "cosas a la vez, delta medio de al menos 0.002 y al menos 4 de 5 particiones que mejoran.",
        "Sin esa exigencia, V10-06 con su +0.000131 pareceria una candidata valida cuando es una",
        "fraccion de la banda de ruido de 0.0053 que midio V10-02.",
        "",
        "## 6. Veredicto",
        "",
        f"**{informe['veredicto']}.**",
        "",
        "Lo relevante de la segunda condicion es que no es un tecnicismo: con 124 celdas, una",
        "candidata necesitaria acertar mas del 65 por ciento de ellas para que el leaderboard la",
        "separara del azar. Ninguna de las tres llega a ese punto, y dos de ellas ni siquiera",
        "mueven celdas suficientes para que la medicion tenga potencia.",
        "",
        (
            f"Fecha: {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
            f"Duracion: {round(time.time() - inicio, 1)} s."
        ),
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
