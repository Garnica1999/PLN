"""V10-10: que regla de conservacion usar cuando se descarta una caracteristica correlacionada.

Por que nace este experimento
----------------------------
V10-09 elimino una de cada par correlacionado y **empeoro mucho**: quitar solo el 1,38 por
ciento de las caracteristicas, 283 de 20.441, costo -0.030626 de F1, y los tres umbrales
empeoraron con 0 de 5 particiones a favor.

Ese numero es demasiado grande para atribuirlo al nivel de correlacion. Si al quitar 283
caracteristicas de 20.441 se pierde 0.03 de metrica, o esas 283 valen mucho, o **la regla que
decide cual se conserva esta descartando siempre la equivocada**.

Y hay una razon concreta para sospechar de la regla. V10-09 conservaba de cada par la
caracteristica con mayor separacion de su AUC sobre 0.5, es decir **la mas informativa en
solitario**. Para un modelo lineal eso es justo lo contrario de lo que conviene: entre dos rasgos
correlacionados, lo que vale es lo que **cada uno aporta de lo que el otro no aporta**, y
quedarse siempre con el mas fuerte en solitario elimina sistematicamente la pieza que
desambiguaba al primero.

Este experimento separa las dos explicaciones
---------------------------------------------
Se calculan **una sola vez por pliegue** la estructura de correlacion y los AUC, que es la parte
cara, y despues se aplican cuatro reglas sobre esos mismos numeros. Ademas se calculan las cuatro
metricas globales de V03 con la referencia sin eliminar, para no re-evaluar nada que ya se sabe.

Las cuatro reglas
-----------------
- **fuerte_en_solitario**: conservar la de mayor |AUC - 0.5|. Es la de V10-09, y se repite para
  que el numero sea comparable.
- **debil_en_solitario**: conservar la de **menor** |AUC - 0.5|. Es la contraprueba directa: si la
  explicacion de la complementariedad es correcta, esta debe dano mejor.
- **menos_redundante**: conservar la que correlaciona menos con el resto. Se usa la correlacion
  media con las caracteristicas de mayor correlación, que es la definicion natural de redundancia
  global.
- **aleatoria**: conservar una al azar. Es el **control decisivo**: si quitar 283 caracteristicas
  al azar cuesta lo mismo que quitarlas por la regla fuerte, el efecto es de **cuantas** se quitan
  y la regla es irrelevante. Si cuesta mucho menos, la regla importa de verdad.

Gates, pre-registrados antes de ejecutar
----------------------------------------
- **Gate A, el control decide la lectura**: si la regla aleatoria se parece a la fuerte, la via se
  cierra y la eliminacion por correlacion queda descartada de verdad. Si la aleatoria es mucho
  mejor, la regla importa y se evalua el gate B.
- **Gate B, mejora**: la mejor regla debe dar delta medio de al menos +0,002 frente a la
  referencia, y mejorar en al menos 4 de 5 particiones.
- **Cierre**: si ningun umbral ni ninguna regla supera el gate B, la via se cierra documentada.

Uso
---
    /home/carlos/venv/bin/python experiments/v10_10_regla_conservacion.py
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
from scipy import sparse
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train
from src.evaluation import (
    ETIQUETAS,
    metrica_macro,
)
from src.splits import RANDOM_STATE, grupos_fusionados, holdout_v2

_especificacion = util.spec_from_file_location(
    "arnes", PROJECT_ROOT / "experiments" / "v10_03_arnes_evaluacion.py"
)
arnes = util.module_from_spec(_especificacion)
sys.modules["arnes"] = arnes
_especificacion.loader.exec_module(arnes)

_especificacion_09 = util.spec_from_file_location(
    "correlacion_09", PROJECT_ROOT / "experiments" / "v10_09_correlacion_eliminacion.py"
)
modulo_09 = util.module_from_spec(_especificacion_09)
sys.modules["correlacion_09"] = modulo_09
_especificacion_09.loader.exec_module(modulo_09)

EXPERIMENTO = "V10-10 (regla de conservacion de caracteristicas correlacionadas)"
REPORTE_MD = PROJECT_ROOT / "reports" / "v10_10_regla_conservacion.md"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v10_10_regla_conservacion.json"

SEMILLAS_EVALUACION = (42, 7, 123, 2024, 5150)
UMBRAL_CORRELACION = 0.90
SEMILLA_ALEATORIA = 20260926
DELTA_MINIMO = 0.002
C_V03 = 2.0
MAX_ITERACIONES = 2000
UMBRALES_V03 = arnes.UMBRALES_V03
ETIQUETAS_LISTA = list(ETIQUETAS)
BLOQUE_CORRELACION = 1024

FUERTE = "fuerte_en_solitario"
DEBIL = "debil_en_solitario"
MENOS_REDUNDANTE = "menos_redundante"
ALEATORIA = "aleatoria"
REGLAS = (FUERTE, DEBIL, MENOS_REDUNDANTE, ALEATORIA)
SIN_ELIMINAR = "sin_eliminar"


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V10-10] {mensaje}", flush=True)


def redundancia_por_caracteristica(
    filas: np.ndarray, columnas: np.ndarray, valores: np.ndarray, n_caracteristicas: int
) -> np.ndarray:
    """Calcula la correlacion media de cada caracteristica con las que se le parecen.

    Args:
        filas: primera columna de cada par.
        columnas: segunda columna de cada par.
        valores: correlacion de cada par.
        n_caracteristicas: numero total de caracteristicas.

    Returns:
        Vector con la media ponderada de correlacion de cada caracteristica.
    """
    suma = np.zeros(n_caracteristicas, dtype=np.float64)
    cuenta = np.zeros(n_caracteristicas, dtype=np.float64)
    np.add.at(suma, filas, valores)
    np.add.at(cuenta, filas, 1.0)
    np.add.at(suma, columnas, valores)
    np.add.at(cuenta, columnas, 1.0)
    return np.divide(suma, np.maximum(cuenta, 1.0))


def decidir_por_regla(
    filas: np.ndarray,
    columnas: np.ndarray,
    valores: np.ndarray,
    fuerza: np.ndarray,
    redundancia: np.ndarray,
    n_caracteristicas: int,
    regla: str,
    generador: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Aplica una regla de conservacion sobre los pares correlacionados.

    Args:
        filas: primera columna de cada par.
        columnas: segunda columna de cada par.
        valores: correlacion de cada par.
        fuerza: fuerza de cada caracteristica, la mayor separacion de su AUC sobre 0.5.
        redundancia: correlacion media de cada caracteristica con las que se le parecen.
        n_caracteristicas: numero total de caracteristicas.
        regla: nombre de la regla.
        generador: generador con semilla fija, para que la regla aleatoria sea reproducible.

    Returns:
        Tupla con la mascara de conservacion y el numero de caracteristicas eliminadas.

    Raises:
        ValueError: si la regla no es una de las conocidas.
    """
    conservar = np.ones(n_caracteristicas, dtype=bool)
    if len(filas) == 0:
        return conservar, 0
    if regla == FUERTE:
        criterio = fuerza
        descendente = True
    elif regla == DEBIL:
        criterio = fuerza
        descendente = False
    elif regla == MENOS_REDUNDANTE:
        criterio = redundancia
        descendente = True
    elif regla == ALEATORIA:
        criterio = generador.random(n_caracteristicas)
        descendente = True
    else:
        raise ValueError(f"regla de conservacion desconocida: {regla}")
    for i, j in zip(filas, columnas, strict=True):
        if not (conservar[i] and conservar[j]):
            continue
        if descendente:
            conservar[j if criterio[j] < criterio[i] else i] = False
        else:
            conservar[j if criterio[j] > criterio[i] else i] = False
    return conservar, int(np.sum(~conservar))


def oof_de_pliegue(
    ajuste: sparse.csr_matrix,
    consulta: sparse.csr_matrix,
    y: np.ndarray,
    entrenamiento: np.ndarray,
    validacion: np.ndarray,
    mascaras: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Ajusta V03 sobre las columnas de cada mascara y devuelve el OOF de cada regla.

    Args:
        ajuste: matriz de caracteristicas de entrenamiento.
        consulta: matriz de caracteristicas de validacion.
        y: matriz binaria de etiquetas.
        entrenamiento: indices de entrenamiento.
        validacion: indices de validacion.
        mascaras: mascara de conservacion por regla.

    Returns:
        Diccionario con la matriz de probabilidades de cada regla, mas la de la referencia.
    """
    resultado: dict[str, np.ndarray] = {}
    for nombre, mascara in mascaras.items():
        columnas = np.flatnonzero(mascara)
        ajuste_recortado = ajuste[:, columnas]
        consulta_recortada = consulta[:, columnas]
        probabilidades = np.zeros((len(validacion), 3), dtype=np.float64)
        for indice_etiqueta in range(3):
            estimador = LogisticRegression(
                C=C_V03,
                class_weight="balanced",
                max_iter=MAX_ITERACIONES,
                solver="liblinear",
                random_state=RANDOM_STATE,
            )
            estimador.fit(ajuste_recortado, y[entrenamiento, indice_etiqueta])
            probabilidades[:, indice_etiqueta] = estimador.predict_proba(
                consulta_recortada
            )[:, 1]
        resultado[nombre] = probabilidades
    return resultado


def main() -> None:
    """Compara las cuatro reglas de conservacion con la referencia sin eliminar."""
    inicio = time.time()
    log("V10-10: regla de conservacion de caracteristicas correlacionadas")

    train = read_train()
    grupos_completos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos_completos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    grupos_dev = grupos_completos[indices_dev]
    y = desarrollo[ETIQUETAS_LISTA].to_numpy(dtype=int)
    textos = desarrollo["text"].astype(str)
    log(f"   desarrollo: {len(y)} filas | reglas: {', '.join(REGLAS)}")
    log(f"   umbral de correlacion: {UMBRAL_CORRELACION}")

    generador = np.random.Generator(np.random.PCG64(SEMILLA_ALEATORIA))
    metricas: dict[str, dict[int, float]] = {nombre: {} for nombre in (SIN_ELIMINAR,) + REGLAS}
    eliminadas_por_regla: dict[str, list[int]] = {nombre: [] for nombre in REGLAS}

    for semilla in SEMILLAS_EVALUACION:
        pliegues = arnes.particion_por_semilla(textos, y, grupos_dev, semilla)
        probabilidades_por_regla: dict[str, np.ndarray] = {
            nombre: np.zeros((len(y), 3), dtype=np.float64) for nombre in (SIN_ELIMINAR,) + REGLAS
        }
        for entrenamiento, validacion in pliegues:
            ajuste, consulta, _ = modulo_09.construir_espacio(
                textos.iloc[entrenamiento].reset_index(drop=True),
                textos.iloc[validacion].reset_index(drop=True),
            )
            filas, columnas, valores, _ = pares = modulo_09.pares_correlacionados(
                ajuste, BLOQUE_CORRELACION
            )
            seleccion = valores >= UMBRAL_CORRELACION
            filas, columnas, valores = filas[seleccion], columnas[seleccion], valores[seleccion]
            aucs = modulo_09.auc_de_todas(ajuste, y[entrenamiento])
            fuerza = np.zeros(ajuste.shape[1], dtype=np.float64)
            for etiqueta in ETIQUETAS_LISTA:
                fuerza = np.maximum(fuerza, np.abs(aucs[etiqueta] - 0.5))
            del aucs
            redundancia = redundancia_por_caracteristica(
                filas, columnas, valores, ajuste.shape[1]
            )
            del pares
            mascaras = {SIN_ELIMINAR: np.ones(ajuste.shape[1], dtype=bool)}
            for regla in REGLAS:
                mascara, eliminadas = decidir_por_regla(
                    filas,
                    columnas,
                    valores,
                    fuerza,
                    redundancia,
                    ajuste.shape[1],
                    regla,
                    generador,
                )
                mascaras[regla] = mascara
                if validacion[0] == pliegues[0][1][0]:
                    eliminadas_por_regla[regla].append(eliminadas)
            parcial = oof_de_pliegue(
                ajuste, consulta, y, entrenamiento, validacion, mascaras
            )
            for nombre, probabilidades in parcial.items():
                probabilidades_por_regla[nombre][validacion] = probabilidades
        for nombre, probabilidades in probabilidades_por_regla.items():
            predicciones = (probabilidades >= UMBRALES_V03[None, :]).astype(int)
            metricas[nombre][semilla] = metrica_macro(y, predicciones)
        resumen = " | ".join(
            f"{nombre} {metricas[nombre][semilla]:.6f}" for nombre in (SIN_ELIMINAR,) + REGLAS
        )
        log(f"   semilla {semilla:>6}: {resumen}")

    referencia = metricas[SIN_ELIMINAR]
    agregados: dict[str, Any] = {}
    for nombre in (SIN_ELIMINAR,) + REGLAS:
        valores = np.array([referencia[semilla] for semilla in SEMILLAS_EVALUACION])
        if nombre == SIN_ELIMINAR:
            valores = np.array(
                [referencia[semilla] for semilla in SEMILLAS_EVALUACION], dtype=np.float64
            )
        else:
            valores = np.array(
                [metricas[nombre][semilla] for semilla in SEMILLAS_EVALUACION],
                dtype=np.float64,
            )
        deltas = np.array(
            [metricas[nombre][semilla] - referencia[semilla] for semilla in SEMILLAS_EVALUACION],
            dtype=np.float64,
        )
        agregados[nombre] = {
            "metrica_media": round(float(valores.mean()), 6),
            "dispersion": round(float(valores.std(ddof=1)), 6),
            "delta_medio": round(float(deltas.mean()), 6),
            "particiones_que_mejoran": int((deltas > 0).sum()),
            "total_particiones": len(SEMILLAS_EVALUACION),
            "caracteristicas_eliminadas_en_el_primer_pliegue": (
                eliminadas_por_regla[nombre][0] if nombre in eliminadas_por_regla else 0
            ),
        }
        log(
            f"   {nombre:>22}: media {valores.mean():.6f} | delta {deltas.mean():+.6f} | "
            f"mejoran {agregados[nombre]['particiones_que_mejoran']}/"
            f"{len(SEMILLAS_EVALUACION)} | elimina "
            f"{agregados[nombre]['caracteristicas_eliminadas_en_el_primer_pliegue']}"
        )

    delta_fuerte = agregados[FUERTE]["delta_medio"]
    delta_aleatoria = agregados[ALEATORIA]["delta_medio"]
    brecha = abs(delta_fuerte - delta_aleatoria)
    gate_a = {
        "delta_de_la_regla_fuerte": delta_fuerte,
        "delta_de_la_regla_aleatoria": delta_aleatoria,
        "brecha_entre_ellas": round(brecha, 6),
        "la_regla_importa": bool(brecha > 0.005),
        "criterio": (
            "Si la regla aleatoria se parece a la regla fuerte, el efecto es de cuantas "
            "caracteristicas se quitan y no de cuales, y la via se cierra. Si la brecha supera "
            "0.005, la regla importa de verdad y se evalua el gate B."
        ),
    }
    mejor = max(REGLAS, key=lambda nombre: agregados[nombre]["delta_medio"])
    gate_b = {
        "mejor_regla": mejor,
        "delta_medio_al_menos_0.002": bool(agregados[mejor]["delta_medio"] >= DELTA_MINIMO),
        "al_menos_4_de_5_particiones": bool(
            agregados[mejor]["particiones_que_mejoran"] >= arnes.CANDIDATAS_MEJORAN_MINIMO
        ),
    }
    gate_b["superado"] = bool(
        gate_b["delta_medio_al_menos_0.002"] and gate_b["al_menos_4_de_5_particiones"]
    )
    log(f"   gate A, la regla importa: {gate_a['la_regla_importa']} (brecha {brecha:+.6f})")
    log(f"   gate B, mejor regla {mejor}: {gate_b['superado']}")

    informe = {
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "motivo": (
            "V10-09 elimino una de cada par correlacionado y empeoro -0.030626 con solo el 1,38 "
            "por ciento de las caracteristicas eliminadas. Ese numero es demasiado grande para "
            "atribuirlo al nivel de correlacion, y la sospecha es sobre la regla de conservacion."
        ),
        "protocolo": (
            "La estructura de correlacion y los AUC se calculan una sola vez por pliegue, que es "
            "la parte cara, y despues se aplican las cuatro reglas sobre esos mismos numeros. "
            "Todo fold-local."
        ),
        "umbral_de_correlacion": UMBRAL_CORRELACION,
        "reglas": list(REGLAS),
        "agregados": agregados,
        "gate_a_el_control_decide_la_lectura": gate_a,
        "gate_b_de_mejora": gate_b,
        "detalle_por_semilla": {
            nombre: {str(s): round(v, 6) for s, v in valores.items()}
            for nombre, valores in metricas.items()
        },
    }
    REPORTE_JSON.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _informe(informe, inicio)
    log(f"completado en {round(time.time() - inicio, 1)} s")
    log("VEREDICTO: " + ("GATE B SUPERADO" if gate_b["superado"] else "GATE B NO SUPERADO"))


def _informe(informe: dict[str, Any], inicio: float) -> None:
    """Escribe el informe de la comparacion de reglas de conservacion.

    Args:
        informe: veredicto del experimento.
        inicio: instante de arranque del script.
    """
    agregados = informe["agregados"]
    gate_a = informe["gate_a_el_control_decide_la_lectura"]
    gate_b = informe["gate_b_de_mejora"]
    lineas = [
        "# V10-10: regla de conservacion de caracteristicas correlacionadas",
        "",
        f"Generado por `experiments/v10_10_regla_conservacion.py` ({EXPERIMENTO}).",
        "",
        "## 1. Por que hace falta este experimento",
        "",
        "V10-09 elimino una de cada par correlacionado y empeoro mucho: quitar solo el 1,38 por",
        "ciento de las caracteristicas, 283 de 20.441, costo -0.030626 de F1.",
        "",
        "Ese numero es demasiado grande para atribuirlo al nivel de correlacion. Si al quitar 283",
        "caracteristicas de 20.441 se pierde 0.03 de metrica, o esas 283 valen mucho, o **la regla",
        "que decide cual se conserva esta descartando siempre la equivocada**.",
        "",
        "Y hay una razon para sospechar de la regla. V10-09 conservaba de cada par la",
        "caracteristica mas informativa **en solitario**. Para un modelo lineal eso es lo",
        "contrario de lo que conviene: entre dos rasgos correlacionados, lo que vale es lo que",
        "**cada uno aporta de lo que el otro no aporta**, y quedarse siempre con el mas fuerte",
        "elimina sistematicamente la pieza que desambiguaba al primero.",
        "",
        "## 2. Las cuatro reglas y el control que decide la lectura",
        "",
        "| regla | que conserva | caracteristicas eliminadas | media | delta | mejoran |",
        "|---|---|---|---|---|---|",
    ]
    descripciones = {
        SIN_ELIMINAR: "nada, es la referencia",
        FUERTE: "la de mayor |AUC - 0.5|, la de V10-09",
        DEBIL: "la de **menor** |AUC - 0.5|, la contraprueba",
        MENOS_REDUNDANTE: "la que correlaciona menos con el resto",
        ALEATORIA: "una al azar, el control decisivo",
    }
    for nombre in (SIN_ELIMINAR,) + tuple(REGLAS):
        a = agregados[nombre]
        lineas.append(
            f"| {nombre} | {descripciones[nombre]} | "
            f"{a['caracteristicas_eliminadas_en_el_primer_pliegue']} | "
            f"{a['metrica_media']:.6f} | {a['delta_medio']:+.6f} | "
            f"{a['particiones_que_mejoran']}/{a['total_particiones']} |"
        )
    lineas += [
        "",
        "## 3. Gate A, el control decide como leer todo lo demas",
        "",
        f"- delta de la regla fuerte: **{gate_a['delta_de_la_regla_fuerte']:+.6f}**",
        f"- delta de la regla aleatoria: **{gate_a['delta_de_la_regla_aleatoria']:+.6f}**",
        f"- brecha entre las dos: **{gate_a['brecha_entre_ellas']:+.6f}**",
        f"- **la regla importa de verdad: {gate_a['la_regla_importa']}**",
        "",
        gate_a["criterio"],
        "",
        "## 4. Gate B, mejora sobre V03",
        "",
        f"- mejor regla: **{gate_b['mejor_regla']}**",
        f"- delta medio de al menos {DELTA_MINIMO}: **{gate_b['delta_medio_al_menos_0.002']}**",
        f"- al menos 4 de 5 particiones: **{gate_b['al_menos_4_de_5_particiones']}**",
        f"- **gate B superado: {gate_b['superado']}**",
        "",
        "## 5. Veredicto",
        "",
    ]
    if gate_b["superado"]:
        lineas += [
            "**GATE B SUPERADO.** La regla de conservacion es la palanca, no el nivel de",
            "correlacion. Pasa al filtro de detectabilidad de V8-02 antes de considerar sonda.",
        ]
    else:
        lineas += [
            "**GATE B NO SUPERADO.** Ninguna regla de conservacion mejora a V03.",
            "",
            "La conclusion honesta tiene dos partes y conviene no confundirlas. **El umbral esta",
            "cerrado**: elegir que descartar por nivel de correlacion no mejora a V03, con ningun",
            "umbral y con ninguna regla. Eso responde a la pregunta que se hizo.",
            "",
            "Y hay una lectura adicional que V10-09 no podia dar: si la brecha entre la regla",
            "fuerte y la aleatoria es grande, entonces **el mecanismo importa mas que el numero**,",
            "y eso es informacion sobre como funciona el modelo, aunque no sirva para enviar nada.",
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
