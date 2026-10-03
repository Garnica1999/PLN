"""V11-01: reproduccion exacta de V9-02, el estimador de shift de prevalencia por EM.

Por que este experimento es una puerta y no un detalle
------------------------------------------------------
Toda la fase v11 se apoya en una cifra: **la prevalencia de la prueba estimada por EM**, con
factores de 0.936, 0.955 y 0.938. Si esa cifra no se reproduce, cualquier correccion de prior,
calibracion o decision que se construya encima hereda un numero sin verificar, y el plan ordena
expresamente no crear variantes nuevas hasta resolver la discrepancia.

Ademas hay un motivo tecnico y no solo de rigor. El campo ``factor_vs_train`` de V9-02 resulto, en
la auditoria de P0, que **no** es la prevalencia de test dividida por la del train completo, sino
por la del **desarrollo**: el script define ``prevalencia_train = y_dev.mean(axis=0)`` con las
1483 filas del desarrollo. La diferencia es de unos 0.0002 en HS y TR, que es pequena pero no nula,
y un nombre de campo que no significa lo que dice es exactamente el tipo de cosa que hace fallar
una cadena de resultados. P0 ya lo documento; aqui se comprueba sobre el pipeline completo.

Que se reproduce, paso a paso, sin reimplementar nada
-----------------------------------------------------
Se **importa** el modulo de V9-02 y se vuelven a llamar sus funciones, en vez de reescribirlas. Es
deliberado: si se reimplementara el EM, la reproduccion no probaria nada, porque dos
implementaciones parecidas pueden diferir justo donde importa. Al importar el modulo, la unica
posibilidad de discrepancia esta en los datos o en el estado de los artefactos, que es justo lo
que se quiere medir.

1. Reajuste de V03 con todo el train y prediccion sobre la prueba.
2. Calibracion Platt ajustada sobre el OOF canonico del desarrollo.
3. EM de Saerens sobre las probabilidades calibradas de la prueba.
4. Contraste de las prevalencias obtenidas contra las que V9-02 guardo.

Sobre la legalidad del uso de la prueba
---------------------------------------
Este experimento **predice** sobre la prueba y **estima una prevalencia** a partir de
probabilidades no etiquetadas. No usa ninguna etiqueta de la prueba, no entrena con el texto de la
prueba mas alla de predecir, y no ajusta ningun umbral con el. La auditoria formal de legalidad
que el plan pide en P1.1 es un experimento aparte, V11-02, y este no presupone que la resuelva: solo
reproduce una cifra y deja constancia de en que condiciones se obtuvo.

Criterio
--------
**Reproduccion exacta**: las prevalencias EM deben coincidir con las de V9-02 con tolerancia
1e-6, que es la tolerancia de reproducibilidad del proyecto. Una diferencia mayor significa que
algo cambio en los datos, en los artefactos o en las versiones, y hay que investigar antes de
seguir. Las **prevalencias absolutas** son la cifra que la v11 va a usar; los **factores** se
reportan bajo los dos denominadores posibles para que la ambiguedad de P0 quede cerrada.

Uso
---
    /home/carlos/venv/bin/python experiments/v11_01_reproduccion_v9_02.py
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
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import data as data_module
from src.evaluation import ETIQUETAS
from src.splits import grupos_fusionados, holdout_v2

_especificacion_v9 = util.spec_from_file_location(
    "v9_02", PROJECT_ROOT / "experiments" / "v9_02_prior_shift_em.py"
)
v9_02 = util.module_from_spec(_especificacion_v9)
sys.modules["v9_02"] = v9_02
_especificacion_v9.loader.exec_module(v9_02)

EXPERIMENTO = "V11-01 (P1, reproduccion exacta de V9-02)"
REPORTE_MD = PROJECT_ROOT / "reports" / "v11_01_reproduccion_v9_02.md"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v11_01_reproduccion_v9_02.json"
CSV_ANALISIS = PROJECT_ROOT / "reports" / "v11_shift_analysis.csv"

TOLERANCIA_REPRODUCCION = 1e-6
REFERENCIA_V03 = 0.7772114525165721

CAMPOS_CSV_SHIFT = [
    "Label",
    "Train_Prevalence",
    "Estimated_Test_Prevalence",
    "Shift_Factor",
    "Estimator",
    "Uncertainty",
    "Simulation_Error",
    "Decision_Impact",
]


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V11-01] {mensaje}", flush=True)


def main() -> None:
    """Reproduce V9-02 y decide si la cifra de shift es utilizable."""
    inicio = time.time()
    log("V11-01: reproduccion exacta de V9-02")

    train = data_module.read_train()
    test = data_module.read_test()
    grupos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    # `pliegues` no se usa: V9-02 solo necesita el OOF canonico, no volver a particionar.
    y_desarrollo = desarrollo[list(ETIQUETAS)].to_numpy(dtype=int)
    oof = np.load(PROJECT_ROOT / "features" / "oof_v03_final.npz", allow_pickle=True)
    proba_desarrollo = oof["proba"]
    log(
        f"   desarrollo {len(y_desarrollo)} filas | prueba {len(test)} filas | "
        f"probabilidades OOF {proba_desarrollo.shape}"
    )

    prevalencia_desarrollo = y_desarrollo.mean(axis=0)
    prevalencia_train_completo = train[list(ETIQUETAS)].to_numpy(dtype=int).mean(axis=0)
    log(
        "   prevalencias del desarrollo: "
        + ", ".join(f"{e} {prevalencia_desarrollo[i]:.6f}" for i, e in enumerate(ETIQUETAS))
    )
    log(
        "   prevalencias del train completo: "
        + ", ".join(
            f"{e} {prevalencia_train_completo[i]:.6f}" for i, e in enumerate(ETIQUETAS)
        )
    )

    log("1/4 reajustar V03 con todo el train y predecir la prueba")
    modulo14 = v9_02.cargar_modulo14()
    matriz_train, matriz_test = modulo14.construir_matrices(
        train["text"], test["text"], "union"
    )
    proba_prueba = np.zeros((len(test), 3), dtype=np.float64)
    for indice in range(3):
        modelo = LogisticRegression(**modulo14.MODELOS["M2_union_lr_c2"]["params"])
        modelo.fit(matriz_train, train[ETIQUETAS[indice]])
        proba_prueba[:, indice] = modelo.predict_proba(matriz_test)[:, 1]
    log(f"   probabilidades de prueba: {proba_prueba.shape}")

    log("2/4 calibracion Platt sobre el OOF canonico, misma receta que V9-02")
    calibradores = [v9_02.calibrar(proba_desarrollo[:, i], y_desarrollo[:, i]) for i in range(3)]
    calibrada_prueba = np.column_stack(
        [
            calibradores[i].predict_proba(
                v9_02.logit(proba_prueba[:, i]).reshape(-1, 1)
            )[:, 1]
            for i in range(3)
        ]
    )
    log(
        "   media de la probabilidad calibrada: "
        + ", ".join(
            f"{e} {calibrada_prueba[:, i].mean():.6f}" for i, e in enumerate(ETIQUETAS)
        )
    )

    log("3/4 EM de Saerens sobre las probabilidades calibradas de la prueba")
    informe_v9 = json.loads(
        (PROJECT_ROOT / "reports" / "v9_02_prior_shift_em.json").read_text(encoding="utf-8")
    )
    filas_csv: list[dict[str, Any]] = []
    detalle: list[dict[str, Any]] = []
    todo_reproducido = True
    for indice, etiqueta in enumerate(ETIQUETAS):
        pi_em, iteraciones = v9_02.em_saerens(
            calibrada_prueba[:, indice], float(prevalencia_desarrollo[indice])
        )
        esperado = float(informe_v9["em_por_etiqueta"][etiqueta]["prevalencia_em_test"])
        diferencia = pi_em - esperado
        reproducido = abs(diferencia) < TOLERANCIA_REPRODUCCION
        todo_reproducido = todo_reproducido and reproducido
        factor_dev = pi_em / float(prevalencia_desarrollo[indice])
        factor_completo = pi_em / float(prevalencia_train_completo[indice])
        filas_csv.append(
            {
                "Label": etiqueta,
                "Train_Prevalence": round(float(prevalencia_desarrollo[indice]), 6),
                "Estimated_Test_Prevalence": round(pi_em, 6),
                "Shift_Factor": round(factor_dev, 6),
                "Estimator": "EM de Saerens sobre Platt",
                "Uncertainty": "PENDIENTE_V11_05",
                "Simulation_Error": "PENDIENTE_V11_04",
                "Decision_Impact": "PENDIENTE_V11_06",
            }
        )
        detalle.append(
            {
                "etiqueta": etiqueta,
                "prevalencia_desarrollo": float(prevalencia_desarrollo[indice]),
                "prevalencia_train_completo": float(prevalencia_train_completo[indice]),
                "prevalencia_em_esperada": esperado,
                "prevalencia_em_reproducida": pi_em,
                "diferencia": diferencia,
                "reproducida": bool(reproducido),
                "iteraciones": iteraciones,
                "iteraciones_v9_02": informe_v9["em_por_etiqueta"][etiqueta]["iteraciones"],
                "factor_sobre_desarrollo": round(factor_dev, 6),
                "factor_sobre_train_completo": round(factor_completo, 6),
                "factor_guardado_en_v9_02": float(
                    informe_v9["em_por_etiqueta"][etiqueta]["factor_vs_train"]
                ),
            }
        )
        marca = "OK  " if reproducido else "FALLA"
        log(
            f"   [{marca}] {etiqueta}: EM {pi_em:.8f} frente a {esperado:.8f} de V9-02 "
            f"(diferencia {diferencia:+.2e}, {iteraciones} iteraciones)"
        )

    log("4/4 cerrar la ambiguedad del denominador que P0 detecto")
    diferencias_denominador = {
        fila["etiqueta"]: round(
            fila["factor_sobre_desarrollo"] - fila["factor_guardado_en_v9_02"], 6
        )
        for fila in detalle
    }
    denominador_resuelto = all(abs(d) < 1e-4 for d in diferencias_denominador.values())
    log(
        f"   el factor sobre el desarrollo reproduce el guardado en V9-02: {denominador_resuelto} "
        f"(diferencias {diferencias_denominador})"
    )

    informe = {
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "referencia_v03": REFERENCIA_V03,
        "metodo": (
            "Se importa el modulo de V9-02 y se llaman sus funciones, sin reimplementar el EM. Si "
            "se reimplementara, la reproduccion no probaria nada, porque dos implementaciones "
            "parecidas pueden diferir justo donde importa."
        ),
        "uso_de_la_prueba": (
            "Se predice sobre la prueba y se estima la prevalencia a partir de probabilidades NO "
            "etiquetadas. No se usa ninguna etiqueta de la prueba, no se entrena con su texto mas "
            "alla de predecir, y no se ajusta ningun umbral con el. La auditoria formal de "
            "legalidad es V11-02 y no se da por resuelta aqui."
        ),
        "detalle_por_etiqueta": detalle,
        "diferencias_por_etiqueta": {
            fila["etiqueta"]: fila["diferencia"] for fila in detalle
        },
        "tolerancia_de_reproduccion": TOLERANCIA_REPRODUCCION,
        "reproduccion_exacta": bool(todo_reproducido),
        "denominador_resuelto_es_el_desarrollo": bool(denominador_resuelto),
        "diferencias_del_denominador": diferencias_denominador,
        "prevalencias_que_usa_la_v11": {
            fila["etiqueta"]: fila["prevalencia_em_reproducida"] for fila in detalle
        },
    }
    REPORTE_JSON.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _informe(informe, detalle, inicio)
    log(f"completado en {round(time.time() - inicio, 1)} s")
    log(
        f"VEREDICTO: reproduccion exacta: {informe['reproduccion_exacta']} | "
        f"denominador resuelto: {informe['denominador_resuelto_es_el_desarrollo']}"
    )


def _informe(
    informe: dict[str, Any], detalle: list[dict[str, Any]], inicio: float
) -> None:
    """Escribe el informe de la reproduccion.

    Args:
        informe: veredicto del experimento.
        detalle: detalle por etiqueta.
        inicio: instante de arranque del script.
    """
    lineas = [
        "# V11-01: reproduccion exacta de V9-02 (P1)",
        "",
        f"Generado por `experiments/v11_01_reproduccion_v9_02.py` ({EXPERIMENTO}).",
        "",
        "## 1. Por que es una puerta",
        "",
        "Toda la fase v11 se apoya en una cifra: **la prevalencia de la prueba estimada por EM**,",
        "con factores de 0.936, 0.955 y 0.938. Si esa cifra no se reproduce, cualquier correccion",
        "de prior, calibracion o decision que se construya encima hereda un numero sin verificar.",
        "",
        "Y hay un motivo tecnico. En la auditoria de P0 se vio que el campo `factor_vs_train` de",
        "V9-02 **no** es la prevalencia de test dividida por la del train completo, sino por la del",
        "desarrollo: el script define `prevalencia_train = y_dev.mean(axis=0)` con las 1483 filas",
        "del desarrollo. Un nombre de campo que no significa lo que dice es exactamente el tipo de",
        "cosa que hace fallar una cadena de resultados.",
        "",
        "## 2. Como se reproduce sin reimplementar nada",
        "",
        informe["metodo"],
        "",
        "## 3. Resultado",
        "",
        "| etiqueta | p desarrollo | p train completo | EM esperado | EM reproducido | diferencia | iteraciones |",
        "|---|---|---|---|---|---|---|",
    ]
    for fila in detalle:
        marca = "**OK**" if fila["reproducida"] else "**FALLA**"
        lineas.append(
            f"| {fila['etiqueta']} | {fila['prevalencia_desarrollo']:.6f} | "
            f"{fila['prevalencia_train_completo']:.6f} | {fila['prevalencia_em_esperada']:.8f} | "
            f"{fila['prevalencia_em_reproducida']:.8f} | {fila['diferencia']:+.2e} {marca} | "
            f"{fila['iteraciones']} frente a {fila['iteraciones_v9_02']} |"
        )
    lineas += [
        "",
        f"Tolerancia de reproduccion: **{informe['tolerancia_de_reproduccion']}**.",
        "",
        f"**Reproduccion exacta: {informe['reproduccion_exacta']}.**",
        "",
        "## 4. La ambiguedad del denominador, cerrada",
        "",
        "| etiqueta | factor sobre desarrollo | factor guardado en V9-02 | diferencia | factor sobre train completo |",
        "|---|---|---|---|---|",
    ]
    for fila in detalle:
        lineas.append(
            f"| {fila['etiqueta']} | {fila['factor_sobre_desarrollo']:.6f} | "
            f"{fila['factor_guardado_en_v9_02']:.4f} | "
            f"{fila['factor_sobre_desarrollo'] - fila['factor_guardado_en_v9_02']:+.6f} | "
            f"{fila['factor_sobre_train_completo']:.6f} |"
        )
    lineas += [
        "",
        f"El factor sobre el desarrollo reproduce el guardado: **{informe['denominador_resuelto_es_el_desarrollo']}**.",
        "",
        "## 5. Sobre la legalidad del uso de la prueba",
        "",
        informe["uso_de_la_prueba"],
        "",
        "## 6. La cifra que la v11 va a usar",
        "",
        "**Las prevalencias absolutas**, no los factores. Son la magnitud sin ambiguedad, y las dos",
        "lecturas del factor dan la misma conclusion cualitativa: la prueba tiene entre un 4.5 y un",
        "6.4 por ciento menos positivos que el desarrollo.",
        "",
        "| etiqueta | prevalencia de la prueba |",
        "|---|---|",
    ]
    for etiqueta, valor in informe["prevalencias_que_usa_la_v11"].items():
        lineas.append(f"| {etiqueta} | **{valor:.6f}** |")
    lineas += [
        "",
        "## 7. Veredicto",
        "",
    ]
    if informe["reproduccion_exacta"]:
        lineas += [
            "**REPRODUCCION EXACTA.** La cifra de shift sobre la que descansa la fase v11 esta",
            "verificada, y la ambiguedad del denominador queda cerrada a favor del desarrollo. La",
            "fase puede continuar con el audit de legalidad, V11-02, y despues con la simulacion de",
            "shift, V11-04.",
        ]
    else:
        lineas += [
            "**REPRODUCCION FALLIDA.** No se crean variantes nuevas hasta resolver la",
            "discrepancia, tal como ordena el plan.",
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
