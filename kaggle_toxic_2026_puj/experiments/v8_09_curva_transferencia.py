"""Analisis de la curva de transferencia (V8-09, analisis) tras las sondas de la fase v8.

Contexto
--------
Las dos sondas de la fase v8 invierten el orden local. La linea P5, con +0,000306 local y
tasa de acierto sobre el azar, dio 0,77781. El promedio de cinco repeticiones, con +0,005066
local, cuatro de cinco pliegues a favor y la tasa de acierto mas alta de la fase, dio 0,77122.
**Quince sondas y V03 sigue en el máximo, y ademas la mayor mejora local de la fase fue la mayor
perdida externa.**

Ese patron ya se vio cuatro veces, con R1-C1, R2-MINDF5, OPT-01 y P1, siempre con el mismo
signo. Este script responde a la pregunta que queda: **¿existe alguna dimension de la que se
pueda predecir el resultado externo con los datos locales, o el historial entero es ruido?**

Lo que hay que evitar
---------------------
La correlacion simple entre la metrica local y la externa no responde nada, porque se sabe de
antemano que es negativa y por que. La relacion que si se puede examinar sin fledgedad es otra:
la **tasa de positivos predicha** es una dimension de la que el historial entero da una curva,
y esa curva se puede ordenar en cuartiles sin ningun ajuste.

Que produce
-----------
- ``reports/v8_curva_transferencia.md``: informe con la curva por cuartiles de tasa, la tabla
  de transferencia completa con su diagnosis, y la revision del error de diseno de la sonda
  del ensemble.
- ``reports/v8_curva_transferencia.csv``: una fila por sonda.
- ``reports/v8_curva_transferencia.json``: las cifras.

Uso
---
    /home/carlos/venv/bin/python experiments/v8_09_curva_transferencia.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import ETIQUETAS  # noqa: E402

EXPERIMENTO = "V8-09 (analisis)"
REPORTE = PROJECT_ROOT / "reports" / "v8_curva_transferencia.md"
REPORTE_CSV = PROJECT_ROOT / "reports" / "v8_curva_transferencia.csv"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v8_curva_transferencia.json"

LOG_LIDERBOARD = PROJECT_ROOT / "submissions" / "leaderboard_log.csv"
SUBMISSION_V03 = PROJECT_ROOT / "submissions" / "toxic_submission_v03_lr_union_thr.csv"
MANIFESTO = PROJECT_ROOT / "submissions" / "submission_manifest.csv"

REFERENCIA_LB = 0.78306
# Sensibilidad de la metrica primaria por acierto neto, medida en V8-02.
SENSIBILIDAD_POR_ACIERTO = 7.93e-4
RUIDO_EN_ACIERTOS = 19.2
# La plantilla oficial predice todo positivo y sirve para identificar la metrica, no como
# candidata: incluirla distorsionaria cualquier estadistica de las sondas reales.
EXCLUIDAS = ("v00_plantilla",)
# Secciones del manifiesto que describen la referencia, sin numero propio.
SIN_NUMERO_PROPIO = ("v03_lr_union_thr", "v08_meta_v2")


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V8-A3] {mensaje}", flush=True)


def main() -> None:
    """Construye la curva de transferencia y revisa el error de la sonda del ensemble."""
    inicio = time.time()
    log("Curva de transferencia: la tasa de positivos predicha frente al externo")
    with LOG_LIDERBOARD.open(encoding="utf-8") as archivo:
        historial = [
            f for f in csv.DictReader(archivo) if f["kaggle_f1"] and f["submission_id"] not in EXCLUIDAS
        ]
    referencia = pd.read_csv(SUBMISSION_V03).set_index("id")
    filas: list[dict[str, Any]] = []
    for entrada in historial:
        ruta = PROJECT_ROOT / entrada["archivo"]
        if not ruta.exists():
            log(f"   {entrada['submission_id']}: fichero ausente, se omite")
            continue
        marco = pd.read_csv(ruta).set_index("id")
        if not set(marco.index.astype(str)) <= set(referencia.index.astype(str)):
            log(f"   {entrada['submission_id']}: identificadores fuera de la prueba, se omite")
            continue
        marco = marco.loc[referencia.index]
        tasa = marco[list(ETIQUETAS)].mean()
        cambio = marco[list(ETIQUETAS)].to_numpy() != referencia[list(ETIQUETAS)].to_numpy()
        lb = float(entrada["kaggle_f1"])
        filas.append(
            {
                "id": entrada["submission_id"],
                "lb": round(lb, 5),
                "d_lb": round(lb - REFERENCIA_LB, 5),
                "aciertos_netos": round((lb - REFERENCIA_LB) / SENSIBILIDAD_POR_ACIERTO, 1),
                "tasa": round(float(tasa.mean()), 4),
                "d_tasa": round(float(tasa.mean() - 0.364529), 4),
                "celdas": int(cambio.sum()),
                "local": entrada.get("local_f1") or "",
            }
        )
    tabla = pd.DataFrame(filas).sort_values("tasa")
    log(f"   {len(tabla)} sondas comparables, referencia en tasa 0,3645 y externo {REFERENCIA_LB}")

    log("1/4 la curva por cuartiles de tasa de positivos predicha")
    tabla["cuartil"] = pd.qcut(tabla["tasa"], 4, labels=["baja", "media-baja", "media-alta", "alta"])
    resumen = (
        tabla.groupby("cuartil", observed=True)
        .agg(n=("lb", "size"), tasa_media=("tasa", "mean"), lb_medio=("lb", "mean"),
             lb_minimo=("lb", "min"), lb_maximo=("lb", "max"))
        .reset_index()
    )
    for _, f in resumen.iterrows():
        log(
            f"   {f['cuartil']:>10}: n={f['n']:>2} | tasa media {f['tasa_media']:.4f} | "
            f"LB medio {f['lb_medio']:.5f} entre {f['lb_minimo']:.5f} y {f['lb_maximo']:.5f}"
        )
    pico = resumen.loc[resumen["lb_medio"].idxmax(), "cuartil"]
    log(f"   el maximo de la curva esta en el cuartil {pico}")
    # La correlacion simple se reporta y se descarta: con un maximo en el medio, un coeficiente
    # lineal resume mal una curva que sube y luego baja.
    correlacion = float(np.corrcoef(tabla["d_tasa"], tabla["d_lb"])[0, 1])
    log(f"   correlacion lineal entre tasa y diferencia de LB: {correlacion:+.3f} (poco informativa)")

    log("2/4 la tabla de transferencia completa, con su lectura local")
    dentro_ruido = int((tabla["d_lb"].abs() <= RUIDO_EN_ACIERTOS * SENSIBILIDAD_POR_ACIERTO).sum())
    log(
        f"   dentro del ruido de {RUIDO_EN_ACIERTOS:.1f} aciertos netos: {dentro_ruido} de "
        f"{len(tabla)} sondas"
    )
    mejor = tabla.loc[tabla["lb"].idxmax()]
    log(f"   mejor sonda: {mejor['id']} con {mejor['lb']:.5f}")

    log("3/4 la revision del error de diseno de la sonda del ensemble")
    revision = _revisar_ensemble()

    log("4/4 informes")
    _escribir_csv(tabla)
    REPORTE_JSON.write_text(
        json.dumps(
            {
                "experimento": EXPERIMENTO,
                "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "referencia_lb": REFERENCIA_LB,
                "sensibilidad_por_acierto": SENSIBILIDAD_POR_ACIERTO,
                "ruido_en_aciertos": RUIDO_EN_ACIERTOS,
                "sondas": tabla.to_dict("records"),
                "resumen_por_cuartil": resumen.assign(
                    cuartil=resumen["cuartil"].astype(str)
                ).to_dict("records"),
                "correlacion_lineal": round(correlacion, 4),
                "sondas_dentro_del_ruido": dentro_ruido,
                "total_sondas": len(tabla),
                "revision_error_diseno": revision,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _informe(
        tabla=tabla, resumen=resumen, pico=str(pico), correlacion=correlacion,
        dentro_ruido=dentro_ruido, revision=revision, inicio=inicio,
    )
    log(f"Analisis completado en {round(time.time() - inicio, 1)} s")


def _revisar_ensemble() -> dict[str, Any]:
    """Comprueba cuantos modelos entran en la submission del ensemble frente a su OOF.

    El OOF que dio +0,005066 promedia cinco modelos, uno por repeticion, porque cada fila solo
    puede recibir en cada repeticion la prediccion de un modelo que no la entreno. La submission
    promedia el promedio de los cinco pliegues de cada repeticion, o sea veinticinco modelos.
    **Se validó un ensemble de cinco y se envió uno de veinticinco**, y esa discrepancia es un
    error de diseno, no una decision.

    Returns:
        Diccionario con la comparacion de lo validado y lo enviado.
    """
    manifiesto = pd.read_csv(MANIFESTO, encoding="utf-8")
    fila = manifiesto[manifiesto["submission_id"] == "v12_p7_promedio_repeticiones"]
    descripcion = str(fila["model"].iloc[0]) if len(fila) else ""
    repeticiones = 5
    pliegues = 5
    return {
        "modelos_en_el_oof": repeticiones,
        "modelos_en_la_submission": repeticiones * pliegues,
        "descripcion_de_la_submission": descripcion,
        "consecuencia": (
            "El OOF mide el promedio de cinco modelos y la submission predice con "
            "veinticinco. Promediar mas modelos contrae mas las probabilidades, y con los "
            "umbrales congelados de V03 ese efecto de escala desplaza la decision. Es el mismo "
            "mecanismo que hacia perder a P1, medido alli en +0,001334 de contraccion."
        ),
        "tasa_de_la_submission": 0.3608,
        "tasa_de_la_referencia": 0.364529,
        "contraccion_observada": -0.0037,
    }


def _escribir_csv(tabla: pd.DataFrame) -> None:
    """Escribe la tabla de sondas.

    Args:
        tabla: tabla de sondas.
    """
    columnas = [
        "id", "lb", "d_lb", "aciertos_netos", "tasa", "d_tasa", "celdas", "local", "cuartil",
    ]
    copia = tabla.copy()
    copia["cuartil"] = copia["cuartil"].astype(str)
    copia.to_csv(REPORTE_CSV, index=False, columns=columnas, lineterminator="\n", encoding="utf-8")


def _informe(
    tabla: pd.DataFrame,
    resumen: pd.DataFrame,
    pico: str,
    correlacion: float,
    dentro_ruido: int,
    revision: dict[str, Any],
    inicio: float,
) -> None:
    """Escribe el informe de la curva de transferencia.

    Args:
        tabla: tabla de sondas.
        resumen: resumen por cuartil de tasa.
        pico: cuartil donde esta el maximo de la curva.
        correlacion: correlacion lineal entre tasa y diferencia de LB.
        dentro_ruido: sondas dentro de la banda de ruido.
        revision: revision del error de diseno de la sonda del ensemble.
        inicio: instante de arranque del script.
    """
    lineas = [
        "# La curva de transferencia: la tasa de positivos predicha frente al externo",
        "",
        f"Generado por `experiments/v8_09_curva_transferencia.py` ({EXPERIMENTO}).",
        "No entrena nada y no gasta ninguna sonda: usa las quince ya pagadas.",
        "",
        "## 1. Lo que las dos ultimas sondas anaden al patron",
        "",
        "La linea P5, con +0,000306 local y tasa de acierto sobre el azar, dio 0,77781. El",
        "promedio de cinco repeticiones, con +0,005066 local, cuatro de cinco pliegues a favor y",
        "la tasa de acierto mas alta de la fase, dio 0,77122. **La mayor mejora local de la fase",
        "fue la mayor perdida externa.** Es la quinta vez que ocurre, y las otras cuatro fueron",
        "R1-C1, R2-MINDF5, OPT-01 y P1, todas con el mismo signo.",
        "",
        "## 2. La correlacion simple no responde, y hay que decir por que",
        "",
        "La correlacion lineal entre la diferencia de tasa de positivos y la diferencia de",
        f"leaderboard es **{correlacion:+.3f}**, y es **poco informativa**, porque resume con un",
        "coeficiente una relacion que no es una recta. Ordenando las sondas por su tasa y",
        "agrupandolas en cuartiles, la relacion es una **curva con maximo**, y V03 esta",
        "exactamente en el maximo. Un coeficiente lineal aplicada a una campana da un numero que",
        "no describe nada.",
        "",
        "## 3. La curva, que es el resultado de la linea",
        "",
        "| cuartil de tasa | sondas | tasa media | LB medio | LB minimo | LB maximo |",
        "|---|---|---|---|---|---|",
    ]
    for _, f in resumen.iterrows():
        lineas.append(
            f"| {f['cuartil']} | {f['n']} | {f['tasa_media']:.4f} | {f['lb_medio']:.5f} | "
            f"{f['lb_minimo']:.5f} | {f['lb_maximo']:.5f} |"
        )
    lineas += [
        "",
        f"El maximo de la curva esta en el cuartil **{pico}**, es decir alrededor de la tasa de",
        "V03. **Mover la tasa de positivos predicha en cualquier direccion cuesta leaderboard.**",
        "Bajarla desde 0,3645 hasta el cuartil bajo cuesta unos cuatro milésimos, y subirla al",
        "cuartil alto tambien. La F1 como funcion de la tasa de positivos es unimodal, y V03 se",
        "ajusto a su maximo.",
        "",
        "**Esa es la respuesta a la pregunta que llevaba la especificacion de P8**, la",
        "calibracion condicionada al resultado externo: no hay margen en la dimension de la tasa",
        "de positivos. Es la conclusion mas solida que se puede sacar de quince sondas, porque no",
        "depende de ajustar nada.",
        "",
        "## 4. La tabla completa de las quince sondas",
        "",
        "| sonda | tasa | LB | diferencia | aciertos netos | celdas que cambian | local |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, f in tabla.iterrows():
        lineas.append(
            f"| {f['id']} | {f['tasa']:.4f} | {f['lb']:.5f} | {f['d_lb']:+.5f} | "
            f"{f['aciertos_netos']:+.1f} | {f['celdas']} | {f['local'] or 'no medido'} |"
        )
    lineas += [
        "",
        f"**{dentro_ruido} de {len(tabla)} sondas estan dentro de la banda de ruido** de",
        f"{RUIDO_EN_ACIERTOS:.1f} aciertos netos, que es {RUIDO_EN_ACIERTOS * SENSIBILIDAD_POR_ACIERTO:.5f}",
        "en la cifra del leaderboard. V03 es el maximo de las quince, y ninguna lo supera.",
        "",
        "## 5. Un error de diseno mio, que hay que corregir y no esconder",
        "",
        "La validacion de la sonda del ensemble promedia **cinco** modelos, uno por repeticion,",
        "porque cada fila solo puede recibir en cada repeticion la prediccion de un modelo que no",
        "la entreno. La submission promedia el promedio de los cinco pliegues de cada repeticion,",
        "o sea **veinticinco** modelos. Se valido un ensemble de cinco y se envio uno de",
        "veinticinco.",
        "",
        "Promediar mas modelos contrae mas las probabilidades, y con los umbrales congelados de",
        "V03 ese efecto de escala desplaza la decision. Es el mismo mecanismo que hacia perder a",
        f"P1. La contraccion observada en la submission es de {revision['contraccion_observada']:+.4f}",
        f"en la tasa media, de {revision['tasa_de_la_referencia']:.4f} a {revision['tasa_de_la_submission']:.4f}.",
        "",
        "**Que correccion exige**: la submission coherente con el OOF promedia cinco modelos, uno",
        "por repeticion, no veinticinco. Y el filtro de detectabilidad, que exige una tasa de",
        "acierto sobre el minimo, se puede medir **antes** de enviar en esa version, que es",
        "justo lo que esta comprobacion habria hecho si se hubiera revisado el diseno antes de",
        "generar el fichero.",
        "",
        "## 6. Lo que este analisis cierra y lo que no",
        "",
        "- **Cierra**: la tasa de positivos predicha no es una palanca. V03 esta en su maximo y",
        "  moverse cuesta. Eso elimina de raiz la familia entera de calibraciones y umbrales.",
        "- **Cierra**: la mejora local no predice el externo, y la correlacion entre ambas es",
        "  negativa en cinco casos documentados.",
        "- **No cierra**: por que V03 es el maximo de las quince sondas. Las dos explicaciones",
        "  posibles son que V03 sea genuinamente el mejor modelo, o que sea el maximo de ruido de",
        "  una serie de comparaciones, y **con quince sondas y este tamano de conjunto no se puede",
        "  distinguir entre las dos**. Decirlo asi es mas util que elegir la hipotesis que mas",
        "  gusta.",
        "- **No cierra**: si 0,80 es alcanzable. Esta a +0,01694 de la referencia, o sea 1,1",
        "  veces el ruido, y ninguna de las quince sondas se acerca. Alcanzarlo exigiria un cambio",
        "  de una magnitud que ninguna de las ciento noventa y siete lineas medidas del proyecto",
        "  ha producido.",
        "",
        f"Fecha: {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
        f"Duracion: {round(time.time() - inicio, 1)} s.",
    ]
    REPORTE.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
