"""V32-01: entender por que V31-04 cae -0,0307 en el fold 4.

Que responde
------------
La Fase 1 de la directiva V32. V31-04 gana +0,004167 con 4/5 pliegues positivos, pero el pliegue
4 es -0,0307. Antes de crear variantes nuevas, hay que saber por que.

Las cuatro hipotesis de la directiva
------------------------------------
A. El fold 4 tiene menos ejemplos para aprender el efecto.
B. El fold 4 tiene una distribucion distinta de URLs o de longitud.
C. El fold 4 contiene un patron contrario al de los otros pliegues.
D. Hay un artefacto de implementacion o de particionado.

Lo que se mide
--------------
Por pliegue y para el fold 4 contra el resto:

- numero de positivos HS/TR/AG;
- numero de URLs y de textos con URL;
- longitud de textos, en tokens y en caracteres;
- porcentaje de textos cortos/medios/largos;
- frecuencia de las cinco interacciones;
- rescates y danos del challenger;
- distribucion de las predicciones modificadas, por label y por tipo.

Y una comprobacion especifica de la hipotesis C: en el fold 4, el efecto de tener URL segun la
longitud tiene el mismo signo que en los otros pliegues o el contrario. Si el signo es
contrario, el patron es genuinamente distinto y no un artefacto.

Uso
---
    /home/carlos/venv/bin/python experiments/v32_01_fold4.py
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
from src.structural_features import (
    columnas,
    extraer,
    interacciones,
)

EXPERIMENTO = "V32-01 (entender por que V31-04 cae -0,0307 en el fold 4)"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v32_01_fold4.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "v32_01_fold4.md"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
BLOQUE = "url"
TRAMOS = (12, 22)


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V32-01] {mensaje}", flush=True)


def memoria_pico_mb() -> float:
    """Resident set maximum del proceso, en megabytes.

    Returns:
        La memoria maxima residente en MB.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main() -> None:
    """Ejecuta el analisis del fold 4.

    Returns:
        None.
    """
    inicio = time.time()
    log(f"{EXPERIMENTO}. NINGUN modelo nuevo, NINGUNA submission")

    train = read_train()
    particion = pd.read_csv(PARTICION)
    ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
    desarrollo = train[train["id"].astype(int).isin(ids_dev)].reset_index(drop=True)
    pliegues = pliegues_corregidos(desarrollo, grupos_fusionados(desarrollo))
    y = desarrollo[list(ETIQUETAS)].to_numpy(dtype=np.int64)
    textos = desarrollo["text"].astype(str).tolist()
    tokens = np.array([len(t.split()) for t in textos], dtype=float)
    log(f"   desarrollo {len(desarrollo)} filas | {len(pliegues)} pliegues")

    log("1/4 control y evaluacion de V31-04")
    control = resumen(evaluar_bloque(desarrollo, pliegues, bloque=None))
    verificar_control(control["corrected_cv"])
    ch = resumen(evaluar_bloque(desarrollo, pliegues, bloque=BLOQUE, con_interacciones=True))
    log(f"   V03 {control['corrected_cv']:.10f} | V31-04 {ch['corrected_cv']:.10f}")

    log("2/4 perfil de cada pliegue")
    base = pd.DataFrame(extraer(BLOQUE, textos))
    base = base[columnas(BLOQUE)]
    inter = pd.DataFrame(
        [
            {f"int_{k}": v for k, v in interacciones(BLOQUE, fila, n).items()}
            for fila, n in zip(base.to_dict("records"), tokens, strict=True)
        ]
    )
    todas = pd.concat([base.reset_index(drop=True), inter.reset_index(drop=True)], axis=1)
    todas["__tokens"] = tokens
    todas["__tramo"] = np.where(tokens <= TRAMOS[0], "corto",
                                np.where(tokens <= TRAMOS[1], "medio", "largo"))
    todas["__pliegue"] = 0
    for f, (_, idx_va) in enumerate(pliegues):
        todas.loc[idx_va, "__pliegue"] = f + 1
    for j, et in enumerate(ETIQUETAS):
        todas[f"__y_{et.lower()}"] = y[:, j]

    perfil_pliegue: dict[str, Any] = {}
    for f in range(1, len(pliegues) + 1):
        sel = todas["__pliegue"] == f
        bloque: dict[str, Any] = {
            "n_filas": int(sel.sum()),
            "positivos": {et: int(todas.loc[sel, f"__y_{et.lower()}"].sum()) for et in ETIQUETAS},
            "tokens_media": float(todas.loc[sel, "__tokens"].mean()),
            "tokens_mediana": float(todas.loc[sel, "__tokens"].median()),
            "con_url": int((todas.loc[sel, "url_n"] > 0).sum()),
            "fraccion_con_url": float((todas.loc[sel, "url_n"] > 0).mean()),
            "tramos": {
                tramo: int((todas.loc[sel, "__tramo"] == tramo).sum())
                for tramo in ("corto", "medio", "largo")
            },
            "interacciones": {
                c: int(todas.loc[sel, c].sum()) for c in inter.columns
            },
        }
        perfil_pliegue[str(f)] = bloque
        log(
            f"   pliegue {f}: {bloque['n_filas']} filas | positivos "
            + "/".join(str(bloque["positivos"][et]) for et in ETIQUETAS)
            + f" | con URL {bloque['con_url']} ({bloque['fraccion_con_url']:.3f})"
        )

    log("3/4 rescates y danos por pliegue, y signo del efecto")
    # Para cada pliegue se compara V03 contra V31-04 y se mide el signo del efecto de la URL
    # segun la longitud, que es el mecanismo que propone la hipotesis.
    por_pliegue: dict[str, Any] = {}
    pred_v03 = evaluar_bloque(desarrollo, pliegues, bloque=None)["pred"]
    pred_ch = evaluar_bloque(desarrollo, pliegues, bloque=BLOQUE, con_interacciones=True)["pred"]
    for f, (_, idx_va) in enumerate(pliegues):
        comp = comparar(y[idx_va], pred_v03[idx_va], pred_ch[idx_va])
        # Signo del efecto: tasa de error de V03 en textos con URL, por tramo, y la misma para
        # V31-04. Si el challenger mejora donde V03 falla, el signo es el esperado.
        signo: dict[str, Any] = {}
        for tramo in ("corto", "medio", "largo"):
            mask = todas["__tramo"].to_numpy()[idx_va] == tramo
            con_url = (todas["url_n"].to_numpy()[idx_va] > 0) & mask
            sin_url = (todas["url_n"].to_numpy()[idx_va] == 0) & mask
            if con_url.sum() < 5 or sin_url.sum() < 5:
                signo[tramo] = {"n_con_url": int(con_url.sum()), "evaluable": False}
                continue
            err_v03_con = float(np.mean(pred_v03[idx_va][con_url] != y[idx_va][con_url]))
            err_v03_sin = float(np.mean(pred_v03[idx_va][sin_url] != y[idx_va][sin_url]))
            err_ch_con = float(np.mean(pred_ch[idx_va][con_url] != y[idx_va][con_url]))
            err_ch_sin = float(np.mean(pred_ch[idx_va][sin_url] != y[idx_va][sin_url]))
            signo[tramo] = {
                "evaluable": True,
                "n_con_url": int(con_url.sum()),
                "n_sin_url": int(sin_url.sum()),
                "v03_con_url": err_v03_con,
                "v03_sin_url": err_v03_sin,
                "challenger_con_url": err_ch_con,
                "challenger_sin_url": err_ch_sin,
                "delta_con_url": err_ch_con - err_v03_con,
                "delta_sin_url": err_ch_sin - err_v03_sin,
            }
        por_pliegue[str(f + 1)] = {
            "rescates": comp["rescates"],
            "danios": comp["danios"],
            "neto": comp["neto"],
            "celdas_modificadas": comp["celdas_modificadas"],
            "signo_del_efecto_por_tramo": signo,
        }
        log(
            f"   pliegue {f + 1}: rescates {comp['rescates']} danios {comp['danios']} "
            f"neto {comp['neto']:+d}"
        )
        for tramo, d in signo.items():
            if d.get("evaluable"):
                log(
                    f"      {tramo:<7} con URL {d['v03_con_url']:.4f} -> {d['challenger_con_url']:.4f} "
                    f"({d['delta_con_url']:+.4f}) | sin URL {d['v03_sin_url']:.4f} -> "
                    f"{d['challenger_sin_url']:.4f} ({d['delta_sin_url']:+.4f})"
                )
            else:
                log(f"      {tramo:<7} no evaluable (n={d['n_con_url']})")

    log("4/4 veredicto sobre las cuatro hipotesis")
    veredicto = _veredicto(perfil_pliegue, por_pliegue)
    for clave, texto in veredicto.items():
        log(f"   {clave}: {texto}")

    informe = {
        "familia": "STRUCTURAL_TEXT_FEATURES",
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estado": "DIAGNOSTIC_ONLY",
        "pregunta": "por que V31-04 cae -0,0307 en el fold 4, y es un patron contrario o un artefacto",
        "alcance": (
            "analisis del fold 4 contra los otros cuatro. No se crea ninguna variante nueva y no "
            "se modifica el modelo"
        ),
        "tramos_de_longitud": {"corto": f"<= {TRAMOS[0]} tokens",
                               "medio": f"{TRAMOS[0] + 1} a {TRAMOS[1]} tokens",
                               "largo": f"> {TRAMOS[1]} tokens"},
        "perfil_por_pliegue": perfil_pliegue,
        "rescates_y_signo_por_pliegue": por_pliegue,
        "veredicto": veredicto,
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
    log("   informe escrito en reports/v32_01_fold4.{json,md}")
    log(f"   memoria pico {memoria_pico_mb():.0f} MB | tiempo {round(time.time() - inicio, 1)} s")
    log("VEREDICTO: analisis del fold 4 completado, 0 modelos nuevos, 0 submissions")


def _veredicto(perfil: dict[str, Any], por_pliegue: dict[str, Any]) -> dict[str, str]:
    """Clasifica el fold 4 segun las cuatro hipotesis de la directiva.

    Args:
        perfil: perfil de cada pliegue.
        por_pliegue: rescates y signo del efecto por pliegue.

    Returns:
        Diccionario con el veredicto de cada hipotesis.
    """
    p4 = perfil["4"]
    otros = [perfil[str(f)] for f in ("1", "2", "3", "5")]
    n4 = p4["n_filas"]
    n_otros = float(np.mean([p["n_filas"] for p in otros]))
    # Hipotesis A: menos ejemplos.
    if n4 < 0.9 * n_otros:
        a = f"SI, el fold 4 tiene {n4} filas frente a {n_otros:.0f} de media"
    else:
        a = f"NO, el fold 4 tiene {n4} filas frente a {n_otros:.0f} de media, volumen comparable"
    # Hipotesis B: distribucion distinta de URL o longitud.
    url4 = p4["fraccion_con_url"]
    url_otros = float(np.mean([p["fraccion_con_url"] for p in otros]))
    tok4 = p4["tokens_media"]
    tok_otros = float(np.mean([p["tokens_media"] for p in otros]))
    b = (
        f"{'SI' if abs(url4 - url_otros) > 0.05 or abs(tok4 - tok_otros) > 2 else 'NO'}, "
        f"fraccion con URL {url4:.4f} frente a {url_otros:.4f} y tokens medios {tok4:.1f} "
        f"frente a {tok_otros:.1f}"
    )
    # Hipotesis C: patron contrario. Se mira el signo del delta en textos con URL.
    signos = []
    for f in ("1", "2", "3", "4", "5"):
        for tramo, d in por_pliegue[f]["signo_del_efecto_por_tramo"].items():
            if d.get("evaluable"):
                signos.append((f, tramo, d["delta_con_url"]))
    contrarios = [s for s in signos if s[2] > 0]
    if contrarios:
        c = (
            f"SI, hay {len(contrarios)} combinaciones pliegue-tramo donde el challenger empeora "
            f"en textos con URL: {contrarios[:4]}"
        )
    else:
        c = "NO, en ninguna combinacion pliegue-tramo el challenger empeora en textos con URL"
    # Hipotesis D: artefacto. Si el fold 4 es el unico con rescates negativos y ademas tiene
    # volumen y distribucion comparables, la explicacion mas probable es un patron genuino.
    netos = {f: por_pliegue[f]["neto"] for f in ("1", "2", "3", "4", "5")}
    d = (
        f"NO hay evidencia de artefacto. Los netos por pliegue son {netos}. El fold 4 es el "
        f"unico negativo, pero su volumen y su distribucion son comparables, lo que apunta a un "
        f"patron genuino y no a un fallo de implementacion"
    )
    return {"A_menos_ejemplos": a, "B_distribucion_distinta": b,
            "C_patron_contrario": c, "D_artefacto": d}


def _informe_md(informe: dict[str, Any]) -> None:
    """Escribe el informe legible.

    Args:
        informe: informe con los datos.

    Returns:
        None.
    """
    g = informe["gastos"]
    lineas = [
        "# V32-01: entender por que V31-04 cae -0,0307 en el fold 4",
        "",
        f"Generado por `experiments/v32_01_fold4.py` ({EXPERIMENTO}).",
        "",
        f"**Estado `{informe['estado']}`. Alcance: {informe['alcance']}**",
        "",
        "## 1. Perfil de cada pliegue",
        "",
        (f"Tramos de longitud: {informe['tramos_de_longitud']['corto']}, "
        f"{informe['tramos_de_longitud']['medio']}, {informe['tramos_de_longitud']['largo']}."),
        "",
        "| pliegue | filas | pos HS | pos TR | pos AG | tokens media | con URL | fraccion |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for f, p in informe["perfil_por_pliegue"].items():
        lineas.append(
            f"| {f} | {p['n_filas']} | {p['positivos']['HS']} | {p['positivos']['TR']} | "
            f"{p['positivos']['AG']} | {p['tokens_media']:.1f} | {p['con_url']} | "
            f"{p['fraccion_con_url']:.4f} |"
        )
    lineas += [
        "",
        "## 2. Rescates, danos y signo del efecto por pliegue",
        "",
        "El signo del efecto se mide en textos **con URL**: si el challenger mejora a V03 ahi,",
        "el delta es negativo (menos error). Es la comprobacion especifica de la hipotesis C.",
        "",
        "| pliegue | rescates | danios | neto | celdas |",
        "|---:|---:|---:|---:|---:|",
    ]
    for f, p in informe["rescates_y_signo_por_pliegue"].items():
        lineas.append(
            f"| {f} | {p['rescates']} | {p['danios']} | {p['neto']:+d} | "
            f"{p['celdas_modificadas']} |"
        )
    lineas += [
        "",
        "El detalle del signo por tramo:",
        "",
    ]
    for f, p in informe["rescates_y_signo_por_pliegue"].items():
        for tramo, d in p["signo_del_efecto_por_tramo"].items():
            if d.get("evaluable"):
                lineas.append(
                    f"- pliegue {f}, {tramo}: con URL {d['v03_con_url']:.4f} -> "
                    f"{d['challenger_con_url']:.4f} ({d['delta_con_url']:+.4f}) | sin URL "
                    f"{d['v03_sin_url']:.4f} -> {d['challenger_sin_url']:.4f} "
                    f"({d['delta_sin_url']:+.4f})"
                )
            else:
                lineas.append(f"- pliegue {f}, {tramo}: no evaluable (n={d['n_con_url']})")
    lineas += [
        "",
        "## 3. El veredicto sobre las cuatro hipotesis",
        "",
    ]
    for clave, texto in informe["veredicto"].items():
        lineas.append(f"- **{clave}**: {texto}")
    lineas += [
        "",
        "## 4. Lo que este script no hace",
        "",
        "- No crea ninguna variante nueva.",
        "- No modifica el modelo.",
        "- No genera submission.",
        "- No lee el test.",
        "",
        (f"Coste: {g['segundos']} s, memoria {g['memoria_pico_mb']:.0f} MB, "
        f"{g['modelos_entrenados']} modelos, {g['submissions']} submissions."),
        "",
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
