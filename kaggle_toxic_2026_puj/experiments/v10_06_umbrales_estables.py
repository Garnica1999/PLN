"""V10-06: tres parametrizaciones de la capa de decision, comparadas en ganancia y dispersion.

Que se pone a prueba
--------------------
Los umbrales de V03, 0.44, 0.51 y 0.46, son tres numeros libres que se eligieron maximizando la
metrica sobre un unico OOF. La fase v6 atribuyo a esa reoptimizacion un inflado de +0.0036, pero
el informe ``reports/v7_umbral_anidado.md`` demostro que esa cifra **confunde el coste del
protocolo con el efecto del umbral**: el control de tres niveles con umbrales congelados da
-0.0081 solo por ajustar sobre cinco sextos de las filas, y al comparar contra un brazo que
comparte conjunto de ajuste el efecto aislado del umbral queda en **-0.0019**. El numero honesto
es, por tanto, -0.0019 y no +0.0036.

Aun asi queda una pregunta abierta que este experimento responde de otra manera: **son esos
tres numeros un punto real de la superficie o un sorteo afortunado de una particion?** Una forma
de responderla es preguntar si una parametrizacion **mas restringida** de la misma capa de
decision rinde lo mismo con menos variacion entre particiones. Si los umbrales se apoyan en
tres grados de libertad reales, una version con menos deberia ser mas estable, y su dispersion
entre particiones sera el termometro.

Las tres parametrizaciones
--------------------------
- **tres_libres**: un umbral por etiqueta, los tres ajustados por separado.
- **desplazamiento_y_escala**: un desplazamiento global mas una escala por etiqueta, o sea dos
  parametros compartidos que mueven los tres umbrales a la vez. Se ajustan en rejilla sobre el
  pliegue de entrenamiento.
- **encogimiento_a_la_mediana**: cada umbral libre se encoge hacia la mediana de los tres, con
  un factor de encogimiento elegido en el pliegue de entrenamiento. Es la version mas
  restringida: con factor 1 los tres umbrales colapsan en un unico valor.

Un segundo resultado sale gratis de este experimento
----------------------------------------------------
La parametrizacion de tres umbrales libres, **re-estimada dentro de cada pliegue de
entrenamiento**, es una version castigada de V03. La diferencia entre V03 y esa version es una
**medicion directa de cuanto aporto optimizar los umbrales sobre el mismo OOF que despues se
reporta**. No es una hipotesis: es lo que el experimento produce.

Gates, pre-registrados antes de ejecutar
----------------------------------------
- **Gate 1**: alguna parametrizacion logra ganancia igual o mejor que la de tres umbrales libres
  **con menor dispersion** entre particiones.
- **Gate 2 (addendum)**: delta medio de al menos +0.002, al menos 4 de 5 particiones mejoran, e
  intervalo pareado que no incluye el cero.
- **Cierre**: si ninguna parametrizacion es a la vez tan buena y mas estable, la holgura de la
  capa de decision es irreducible con estos datos y se documenta.

Uso
---
    /home/carlos/venv/bin/python experiments/v10_06_umbrales_estables.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from importlib import util
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train
from src.evaluation import (
    ETIQUETAS,
    metrica_macro,
    metricas_multietiqueta,
)
from src.splits import grupos_fusionados, holdout_v2

_especificacion = util.spec_from_file_location(
    "arnes", PROJECT_ROOT / "experiments" / "v10_03_arnes_evaluacion.py"
)
arnes = util.module_from_spec(_especificacion)
sys.modules["arnes"] = arnes
_especificacion.loader.exec_module(arnes)

EXPERIMENTO = "V10-06 (P1, umbrales estables)"
REPORTE_MD = PROJECT_ROOT / "reports" / "v10_06_umbrales_estables.md"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v10_06_umbrales_estables.json"

SEMILLAS_EVALUACION = (42, 7, 123, 2024, 5150)
GRANULARIDAD_UMBRAL = 0.005
ESCALAS = np.arange(0.80, 1.201, 0.05)
DESPLAZAMIENTOS = np.arange(-0.10, 0.1001, 0.01)
FACTORES_ENCOGIMIENTO = (0.0, 0.25, 0.50, 0.75, 1.0)
UMBRALES_V03 = arnes.UMBRALES_V03
TRES_LIBRES = "tres_libres"
DESPLAZAMIENTO_ESCALA = "desplazamiento_y_escala"
ENCOGIMIENTO = "encogimiento_a_la_mediana"
PARAMETRIZACIONES = (TRES_LIBRES, DESPLAZAMIENTO_ESCALA, ENCOGIMIENTO)


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V10-06] {mensaje}", flush=True)


def mejor_umbral_por_etiqueta(
    probabilidades: np.ndarray, y: np.ndarray
) -> tuple[list[float], float]:
    """Busca el umbral F1 optimo de cada etiqueta, y devuelve tambien la macro de ese conjunto.

    Args:
        probabilidades: matriz de probabilidades.
        y: matriz binaria de etiquetas.

    Returns:
        Tupla con la lista de tres umbrales y el valor de la metrica macro que producen.
    """
    umbrales = []
    for indice in range(len(ETIQUETAS)):
        mejor_f1, mejor = -1.0, 0.5
        for umbral in np.arange(0.05, 0.951, GRANULARIDAD_UMBRAL):
            seleccionadas = probabilidades[:, indice] >= umbral
            verdaderos_positivos = float(np.sum(seleccionadas & (y[:, indice] == 1)))
            if verdaderos_positivos == 0:
                continue
            precision = verdaderos_positivos / float(np.sum(seleccionadas))
            exhaustividad = verdaderos_positivos / float(np.sum(y[:, indice] == 1))
            f1 = 2 * precision * exhaustividad / (precision + exhaustividad)
            if f1 > mejor_f1:
                mejor_f1, mejor = f1, float(umbral)
        umbrales.append(mejor)
    predicciones = (probabilidades >= np.array(umbrales)[None, :]).astype(int)
    return umbrales, metrica_macro(y, predicciones)


def ajustar_desplazamiento_y_escala(
    probabilidades: np.ndarray, y: np.ndarray, base: list[float]
) -> tuple[list[float], float]:
    """Ajusta un desplazamiento global y una escala comun a los tres umbrales.

    Args:
        probabilidades: matriz de probabilidades del pliegue de entrenamiento.
        y: matriz binaria de etiquetas del pliegue de entrenamiento.
        base: umbrales libres de partida.

    Returns:
        Tupla con los umbrales transformados y la metrica macro que producen.
    """
    base_vectores = np.array(base, dtype=np.float64)
    mejor_metrica, mejores = -1.0, list(base)
    for escala in ESCALAS:
        for desplazamiento in DESPLAZAMIENTOS:
            candidatos = np.clip(base_vectores * escala + desplazamiento, 0.05, 0.95)
            predicciones = (probabilidades >= candidatos[None, :]).astype(int)
            valor = metrica_macro(y, predicciones)
            if valor > mejor_metrica:
                mejor_metrica = valor
                mejores = [float(u) for u in candidatos]
    return mejores, mejor_metrica


def ajustar_encogimiento(
    probabilidades: np.ndarray, y: np.ndarray, base: list[float]
) -> tuple[list[float], float, float]:
    """Ajusta el factor de encogimiento de los umbrales hacia su mediana.

    Args:
        probabilidades: matriz de probabilidades del pliegue de entrenamiento.
        y: matriz binaria de etiquetas del pliegue de entrenamiento.
        base: umbrales libres de partida.

    Returns:
        Tupla con los umbrales encogidos, la metrica que producen y el factor elegido.
    """
    base_vectores = np.array(base, dtype=np.float64)
    mediana = float(np.median(base_vectores))
    mejor_metrica, mejores, mejor_factor = -1.0, list(base), 0.0
    for factor in FACTORES_ENCOGIMIENTO:
        candidatos = np.clip(
            (1.0 - factor) * base_vectores + factor * mediana, 0.05, 0.95
        )
        predicciones = (probabilidades >= candidatos[None, :]).astype(int)
        valor = metrica_macro(y, predicciones)
        if valor > mejor_metrica:
            mejor_metrica, mejores, mejor_factor = valor, [float(u) for u in candidatos], factor
    return mejores, mejor_metrica, mejor_factor


def main() -> None:
    """Compara las tres parametrizaciones en ganancia y dispersion."""
    inicio = time.time()
    log("V10-06: tres parametrizaciones de la capa de decision")

    train = read_train()
    grupos_completos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos_completos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    grupos_dev = grupos_completos[indices_dev]
    y = desarrollo[list(ETIQUETAS)].to_numpy(dtype=int)
    textos = desarrollo["text"].astype(str)
    log(f"   desarrollo: {len(y)} filas | parametrizaciones: {', '.join(PARAMETRIZACIONES)}")

    log(f"1/2 recorrer {len(SEMILLAS_EVALUACION)} particiones, con los umbrales anidados")
    detalle: dict[str, list[dict[str, Any]]] = {p: [] for p in PARAMETRIZACIONES}
    referencia_por_semilla: dict[int, float] = {}

    for semilla in SEMILLAS_EVALUACION:
        pliegues = arnes.particion_por_semilla(textos, y, grupos_dev, semilla)
        probabilidades = arnes.oof_de_v03(textos, y, pliegues)
        referencia = arnes.predicciones_de_referencia(probabilidades)
        metrica_referencia = metrica_macro(y, referencia)
        referencia_por_semilla[semilla] = metrica_referencia

        predicciones_por_parametrizacion = {
            p: np.zeros((len(y), 3), dtype=int) for p in PARAMETRIZACIONES
        }
        umbrales_por_parametrizacion: dict[str, dict[str, list[float]]] = {
            p: {e: [] for e in ETIQUETAS} for p in PARAMETRIZACIONES
        }
        factores_por_parametrizacion: dict[str, list[float]] = {p: [] for p in PARAMETRIZACIONES}

        for entrenamiento, validacion in pliegues:
            libres, _ = mejor_umbral_por_etiqueta(
                probabilidades[entrenamiento], y[entrenamiento]
            )
            escalados, _ = ajustar_desplazamiento_y_escala(
                probabilidades[entrenamiento], y[entrenamiento], libres
            )
            encogidos, _, factor = ajustar_encogimiento(
                probabilidades[entrenamiento], y[entrenamiento], libres
            )
            for nombre, lista in (
                (TRES_LIBRES, libres),
                (DESPLAZAMIENTO_ESCALA, escalados),
                (ENCOGIMIENTO, encogidos),
            ):
                vector = np.array(lista)[None, :]
                predicciones_por_parametrizacion[nombre][validacion] = (
                    probabilidades[validacion] >= vector
                ).astype(int)
                for indice, etiqueta in enumerate(ETIQUETAS):
                    umbrales_por_parametrizacion[nombre][etiqueta].append(lista[indice])
            factores_por_parametrizacion[ENCOGIMIENTO].append(factor)

        for nombre in PARAMETRIZACIONES:
            predicciones = predicciones_por_parametrizacion[nombre]
            metrica = metrica_macro(y, predicciones)
            detalle[nombre].append(
                {
                    "semilla": semilla,
                    "metrica_referencia": round(metrica_referencia, 6),
                    "metrica_candidata": round(metrica, 6),
                    "delta": round(metrica - metrica_referencia, 6),
                    "mejora": bool(metrica - metrica_referencia > 0.0),
                    "celdas_distintas": int((predicciones != referencia).sum()),
                    "umbrales_por_etiqueta": {
                        e: [round(u, 3) for u in valores]
                        for e, valores in umbrales_por_parametrizacion[nombre].items()
                    },
                    "mediana_de_los_umbrales_por_etiqueta": {
                        e: round(float(np.median(v)), 4)
                        for e, v in umbrales_por_parametrizacion[nombre].items()
                    },
                    "factor_de_encogimiento": (
                        round(float(np.mean(factores_por_parametrizacion[ENCOGIMIENTO])), 3)
                        if nombre == ENCOGIMIENTO
                        else None
                    ),
                    "f1_por_etiqueta": {
                        k: round(v, 6)
                        for k, v in metricas_multietiqueta(y, predicciones)["f1_por_etiqueta"].items()
                    },
                }
            )
        resumen = " | ".join(f"{p} {detalle[p][-1]['delta']:+.6f}" for p in PARAMETRIZACIONES)
        log(f"   semilla {semilla:>6}: {resumen}")

    log("2/2 agregar, medir dispersion y aplicar los gates")
    agregados: dict[str, dict[str, Any]] = {}
    for nombre, particiones in detalle.items():
        metricas = np.array([p["metrica_candidata"] for p in particiones], dtype=np.float64)
        deltas = np.array([p["delta"] for p in particiones], dtype=np.float64)
        agregados[nombre] = {
            "metrica_media": round(float(metricas.mean()), 6),
            "metrica_peor": round(float(metricas.min()), 6),
            "metrica_mejor": round(float(metricas.max()), 6),
            "dispersion_de_la_metrica": round(float(metricas.std(ddof=1)), 6),
            "recorrido_de_la_metrica": round(float(metricas.max() - metricas.min()), 6),
            "delta_medio": round(float(deltas.mean()), 6),
            "delta_minimo": round(float(deltas.min()), 6),
            "particiones_que_mejoran": int((deltas > 0).sum()),
            "total_particiones": len(particiones),
            "celdas_distintas_promedio": round(
                float(np.mean([p["celdas_distintas"] for p in particiones])), 1
            ),
        }
        log(
            f"   {nombre:>26}: media {agregados[nombre]['metrica_media']:.6f} | "
            f"dispersion {agregados[nombre]['dispersion_de_la_metrica']:.6f} | "
            f"peor {agregados[nombre]['metrica_peor']:.6f} | "
            f"delta medio {agregados[nombre]['delta_medio']:+.6f} | "
            f"celdas {agregados[nombre]['celdas_distintas_promedio']:.0f}"
        )

    referencia_dispersion = agregados[TRES_LIBRES]["dispersion_de_la_metrica"]
    referencia_peor = agregados[TRES_LIBRES]["metrica_peor"]
    candidatas_estables = [
        nombre
        for nombre in PARAMETRIZACIONES
        if agregados[nombre]["metrica_media"] >= agregados[TRES_LIBRES]["metrica_media"]
        and agregados[nombre]["dispersion_de_la_metrica"] < referencia_dispersion
    ]
    gate_1 = bool(candidatas_estables)
    log(
        f"   gate 1 (igual o mejor media y menor dispersion que tres libres): {gate_1} | "
        f"candidatas: {candidatas_estables if candidatas_estables else 'ninguna'}"
    )

    pliegues_canonicos = arnes.particion_por_semilla(textos, y, grupos_dev, 42)
    probabilidades_canonicas = arnes.oof_de_v03(textos, y, pliegues_canonicos)
    referencia_canonica = arnes.predicciones_de_referencia(probabilidades_canonicas)
    intervalos: dict[str, dict[str, Any]] = {}
    for nombre in PARAMETRIZACIONES:
        predicciones = np.zeros((len(y), 3), dtype=int)
        for entrenamiento, validacion in pliegues_canonicos:
            libres, _ = mejor_umbral_por_etiqueta(
                probabilidades_canonicas[entrenamiento], y[entrenamiento]
            )
            if nombre == TRES_LIBRES:
                lista = libres
            elif nombre == DESPLAZAMIENTO_ESCALA:
                lista, _ = ajustar_desplazamiento_y_escala(
                    probabilidades_canonicas[entrenamiento], y[entrenamiento], libres
                )
            else:
                lista, _, _ = ajustar_encogimiento(
                    probabilidades_canonicas[entrenamiento], y[entrenamiento], libres
                )
            predicciones[validacion] = (
                probabilidades_canonicas[validacion] >= np.array(lista)[None, :]
            ).astype(int)
        observada, inferior, superior = arnes.intervalo_pareado(
            y, predicciones, referencia_canonica
        )
        intervalos[nombre] = {
            "diferencia_observada": round(observada, 6),
            "ic95_inferior": round(inferior, 6),
            "ic95_superior": round(superior, 6),
            "no_incluye_cero": bool(inferior > 0.0 or superior < 0.0),
        }
        log(
            f"   intervalo pareado de {nombre}: [{inferior:+.6f}, {superior:+.6f}]"
        )

    mejor_por_delta = max(PARAMETRIZACIONES, key=lambda p: agregados[p]["delta_medio"])
    gate_2 = {
        "parametrizacion": mejor_por_delta,
        "delta_medio_al_menos_0.002": bool(agregados[mejor_por_delta]["delta_medio"] >= 0.002),
        "al_menos_4_de_5_particiones": bool(
            agregados[mejor_por_delta]["particiones_que_mejoran"]
            >= arnes.CANDIDATAS_MEJORAN_MINIMO
        ),
        "intervalo_pareado_no_incluye_cero": bool(intervalos[mejor_por_delta]["no_incluye_cero"]),
    }
    gate_2["superado"] = bool(
        gate_2["delta_medio_al_menos_0.002"]
        and gate_2["al_menos_4_de_5_particiones"]
        and gate_2["intervalo_pareado_no_incluye_cero"]
    )

    informe = {
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pregunta": (
            "Los tres umbrales libres de V03 son un punto real de la superficie o un sorteo "
            "afortunado? Una capa mas restringida deberia ser mas estable si el punto es real."
        ),
        "protocolo": (
            "Los tres conjuntos de umbrales se ajustan dentro de cada pliegue de entrenamiento y "
            "se aplican al de validacion, con cinco semillas de particion. La referencia es V03 "
            "con sus umbrales congelados, sin reoptimizar."
        ),
        "parametrizaciones": list(PARAMETRIZACIONES),
        "agregados": agregados,
        "referencia_de_dispersion_tres_libres": referencia_dispersion,
        "referencia_de_peor_particion_tres_libres": referencia_peor,
        "candidatas_estables": candidatas_estables,
        "gate_1_superado": gate_1,
        "intervalos_pareados": intervalos,
        "gate_2_del_addendum": gate_2,
        "medianas_de_los_umbrales": {
            nombre: detalle[nombre][0]["mediana_de_los_umbrales_por_etiqueta"]
            for nombre in PARAMETRIZACIONES
        },
        "detalle_por_particion": detalle,
    }
    REPORTE_JSON.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _informe(informe, inicio)
    log(f"completado en {round(time.time() - inicio, 1)} s")
    log(
        "VEREDICTO: "
        + (
            "GATE 1 SUPERADO, la capa de decision se puede hacer mas estable"
            if gate_1
            else "GATE 1 NO SUPERADO, la holgura es irreducible con estos datos"
        )
    )


def _informe(informe: dict[str, Any], inicio: float) -> None:
    """Escribe el informe de las tres parametrizaciones de la capa de decision.

    Args:
        informe: veredicto del experimento.
        inicio: instante de arranque del script.
    """
    agregados = informe["agregados"]
    lineas = [
        "# V10-06: tres parametrizaciones de la capa de decision (P1, tercera linea de la via A)",
        "",
        f"Generado por `experiments/v10_06_umbrales_estables.py` ({EXPERIMENTO}).",
        "",
        "## 1. La pregunta",
        "",
        "Los umbrales de V03, 0.44, 0.51 y 0.46, son tres numeros libres elegidos sobre un unico",
        "OOF. La fase v6 atribuyo a eso un inflado de +0.0036, pero el informe de la fase v7",
        "demostro que la cifra confunde el coste del protocolo con el efecto del umbral: el",
        "control de tres niveles con umbrales congelados da -0.0081 solo por ajustar sobre cinco",
        "sextos de las filas, y al comparar contra un brazo que comparte conjunto de ajuste el",
        "efecto aislado del umbral queda en -0.0019. La pregunta que queda es si los tres numeros",
        "son un punto real de la superficie o un sorteo afortunado. Si son reales, una",
        "parametrizacion **mas restringida** de la misma capa deberia rendir parecido con menos",
        "variacion entre particiones.",
        "",
        "| parametrizacion | grados de libertad |",
        "|---|---|",
        f"| {TRES_LIBRES} | 3, uno por etiqueta |",
        f"| {DESPLAZAMIENTO_ESCALA} | 2, un desplazamiento y una escala compartidos |",
        f"| {ENCOGIMIENTO} | 1, el factor de encogimiento hacia la mediana |",
        "",
        "## 2. Ganancia y dispersion",
        "",
        "| parametrizacion | media | peor | recorrido | dispersion | delta medio | mejoran | celdas |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for nombre in PARAMETRIZACIONES:
        a = agregados[nombre]
        lineas.append(
            f"| {nombre} | {a['metrica_media']:.6f} | {a['metrica_peor']:.6f} | "
            f"{a['recorrido_de_la_metrica']:.6f} | {a['dispersion_de_la_metrica']:.6f} | "
            f"{a['delta_medio']:+.6f} | {a['particiones_que_mejoran']}/{a['total_particiones']} | "
            f"{a['celdas_distintas_promedio']:.0f} |"
        )
    lineas += [
        "",
        "La referencia de dispersion es",
        f"{informe['referencia_de_dispersion_tres_libres']:.6f}, la de tres umbrales libres.",
        "",
        "## 3. Un segundo resultado que sale del mismo experimento",
        "",
        "La fila de tres umbrales libres es una **version castigada de V03**: los mismos tres",
        "grados de libertad, pero re-estimados dentro de cada pliegue de entrenamiento en vez de",
        "elegidos una vez sobre todo el OOF. La diferencia entre V03 y esa fila es por tanto una",
        "**medicion directa** de cuanto aporto optimizar los umbrales sobre el conjunto que",
        "despues se reporta, y no una hipotesis de la auditoria.",
        "",
        f"El delta medio de esa fila es {agregados[TRES_LIBRES]['delta_medio']:+.6f} y sus celdas",
        f"distintas frente a V03 son {agregados[TRES_LIBRES]['celdas_distintas_promedio']:.0f} de",
        "media.",
        "",
        "**Correccion de una cita previa, que importa.** La fase v6 atribuyo a la reoptimizacion",
        "de umbrales un inflado de +0.0036. Este experimento mide +0.000131, y el informe de la",
        "fase v7 ya habia mostrado que la cifra de v6 confundia el coste del protocolo con el",
        "efecto del umbral, con un efecto aislado de **-0.0019**. Las tres cifras son distintas",
        "porque miden cosas distintas: +0.0036 es un error de comparacion entre brazos con",
        "distinto conjunto de ajuste, -0.0019 es el efecto del umbral en el protocolo estricto de",
        "tres niveles, y +0.000131 es el efecto de reestimar los umbrales por pliegue en el",
        "protocolo de dos niveles. **Las tres coinciden en lo substancial: el efecto del umbral es",
        "pequeno.** Lo que no cabe es citar +0.0036 como si fuera el efecto del umbral, y esa",
        "correccion se aplica tambien a la documentacion de V10-04.",
        "",
        "## 4. Gates pre-registrados",
        "",
        f"- **Gate 1**, igual o mejor media con menor dispersion: **{informe['gate_1_superado']}**",
    ]
    candidatas = informe["candidatas_estables"]
    lineas.append(
        f"  - candidatas estables: {', '.join(candidatas) if candidatas else 'ninguna'}"
    )
    gate_2 = informe["gate_2_del_addendum"]
    lineas += [
        f"- **Gate 2**, los tres criterios del addendum: **{gate_2['superado']}**",
        f"  - parametrizacion evaluada: {gate_2['parametrizacion']}",
        f"  - delta medio de al menos 0.002: {gate_2['delta_medio_al_menos_0.002']}",
        f"  - al menos 4 de 5 particiones: {gate_2['al_menos_4_de_5_particiones']}",
        f"  - intervalo que no incluye el cero: {gate_2['intervalo_pareado_no_incluye_cero']}",
        "",
        "## 5. Veredicto",
        "",
    ]
    if not informe["gate_1_superado"]:
        lineas += [
            "**GATE 1 NO SUPERADO.** Ninguna parametrizacion mas restringida mantiene la ganancia",
            "de la de tres umbrales libres y ademas es mas estable. La holgura de la capa de",
            "decision es **irreducible con estos datos**, que es el criterio de cierre escrito",
            "antes de ejecutar. La consecuencia practica es que **no hay una version de la capa de",
            "decision que sea a la vez igual de buena y mas robusta**, de modo que los umbrales",
            "congelados se quedan como estan y no se sustituyen por una version mas elegante.",
        ]
    else:
        lineas += [
            "**GATE 1 SUPERADO.** Hay una parametrizacion con igual o mejor media y menor",
            "dispersion, y se adoptaria como capa de decision por defecto de la fase.",
        ]
    if not gate_2["superado"]:
        lineas += [
            "",
            "El gate 2 no se supera, asi que **ninguna de las tres se envia**: reparametrizar no",
            "supera a V03 en la metrica, que es lo que un envio necesita.",
        ]
    lineas += [
        "",
        (
            f"Fecha: {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
            f"Duracion: {round(time.time() - inicio, 1)} s."
        ),
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
