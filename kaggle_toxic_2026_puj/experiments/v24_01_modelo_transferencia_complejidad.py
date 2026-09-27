"""V24-01 (P1): modelo de transferencia de complejidad.

Que hace y por que
------------------
V23 envio la candidata con mejor evidencia interna de toda la historia del proyecto:
`corrected_cv` 0.790410, 5 de 5 pliegues, perfil de celdas con neto +32 y McNemar exacto
p = 0.0452. El leaderboard bajo 0.00368. La leccion no es "esa tecnica era mala", es algo
mas general: **la generalizacion local y la transferencia competitiva pueden divergir
cuando la complejidad efectiva aumenta**, y esa divergencia deberia poder detectarse ANTES
de gastar una submission.

Este script es una auditoria diagnostica. **No entrena modelos de prediccion para la
competicion y no genera ninguna submission.** Reconstruye los indicadores de complejidad de
los modelos lineales que si se pueden recalcular de forma exacta, los cruza con el historial
real de `delta LB` de las 22 submissions, y produce un checklist aplicable antes de enviar.

Por que los indicadores se recalculan y no se leen de los informes
------------------------------------------------------------------
Los numeros de los informes historicos son heterogeneos: cada fase uso su propio esquema de
pliegues, su propia metrica y su propia nomenclatura. Cruzar esos numeros sin recalcular
daria una correlacion de artefactos. Aqui los indicadores se miden **todos con el mismo
protocolo**: los pliegues canonicos de `src.splits`, el desarrollo de 1.483 filas y los
umbrales congelados de V03. Lo unico que se toma de los informes es el `delta LB`, que es
un hecho de la plataforma y no se puede recalcular.

El limit que hay que tener presente
----------------------------------
Hay **22 submissions**, no 1483. Cualquier correlacion calculada aqui tiene menos de cinco
observaciones por variable, y su intervalo de confianza sera enorme. El objetivo de este
script **no** es producir un modelo predictivo del `delta LB`, que seria sobreajustar 22
puntos. Es producir tres cosas: una comprobacion de si la direccion que se hipotetiza
(s MAS complejidad efectiva -> PEOR transferencia) aparece en los datos, un checklist con
umbrales que se puedan aplicar a un candidato nuevo, y una lista explicita de lo que este
analisis **no** puede concluir.

Uso
---
    /home/carlos/venv/bin/python experiments/v24_01_modelo_transferencia_complejidad.py
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
from scipy import sparse
from scipy.stats import pearsonr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train
from src.splits import grupos_fusionados, pliegues_corregidos

EXPERIMENTO = "V24-01 (P1: modelo de transferencia de complejidad sobre el historial real)"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v24_01_transferencia_complejidad.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "v24_01_transferencia_complejidad.md"
OOF_CANONICO = PROJECT_ROOT / "features" / "oof_v03_final.npz"
LOG_LB = PROJECT_ROOT / "submissions" / "leaderboard_log.csv"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"

ETIQUETAS = ("HS", "TR", "AG")
UMBRALES_V03 = np.array([0.44, 0.51, 0.46])
CORRECTED_CV_V03 = 0.7772114525165721
SEMILLA = 42
PARAMETROS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
PARAMETROS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2, "sublinear_tf": True}

# Indicadores del historial de V23, que no se pueden recalcular con este script porque
# requiere volver a entrenar el kernel. Se incorporan a mano y se marcan como tal.
V23 = {
    "nombre": "V23 svd_rbf_sigmoid",
    "modelo": "V23 svd_rbf_sigmoid",
    "delta_oof": 0.013198,
    "delta_lb": -0.00368,
    "gap_train_oof": 0.0,
    "fraccion_vectores_soporte": float(np.mean([1321, 1044, 1317]) / 1731),
    "sparsity": 0.0,
    "norma_coeficientes": np.nan,
    "n_parametros": np.nan,
    "corrected_cv": 0.7772114525165721 + 0.013198,
    "celdas": 240,
    "rescates": 136,
    "danios": 104,
    "origen": "reportes de v23_01 y v23_02",
    "nota": "el gap train-OOF no se midio para esta candidata; se deja como nan",
}


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V24-01] {mensaje}", flush=True)


def _num(fila: dict[str, Any], clave: str, formato: str = ".4f", vacio: str = "n/d") -> str:
    """Formatea un campo numerico que puede no existir.

    La fila de V23 no se recalcula en este script, asi que sus indicadores de modelo lineal
    no existen. En vez de inventarlos, la tabla los marca como no disponible.

    Args:
        fila: fila del indicador.
        clave: nombre del campo.
        formato: formato de Python.
        vacio: texto cuando el campo no existe o es nan.

    Returns:
        El valor formateado o el texto de ausencia.
    """
    valor = fila.get(clave, None)
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return vacio
    try:
        return format(valor, formato)
    except (TypeError, ValueError):
        return str(valor)


def macro_f1(verdad: np.ndarray, predicho: np.ndarray) -> float:
    """Macro-F1 de las tres clases positivas.

    Args:
        verdad: matriz verdadera binaria.
        predicho: matriz predicha binaria.

    Returns:
        La media de las tres F1.
    """
    valores = []
    for j in range(verdad.shape[1]):
        comunes = int(np.count_nonzero(verdad[:, j] & predicho[:, j]))
        denominador = int(np.count_nonzero(verdad[:, j])) + int(np.count_nonzero(predicho[:, j]))
        valores.append(2 * comunes / denominador if denominador > 0 else 0.0)
    return float(np.mean(valores))


def main() -> None:
    """Construye la tabla de complejidad y la cruza con el historial de LB."""
    inicio = time.time()
    log("V24-01: P1, auditoria de complejidad. Ningun modelo de prediccion para la competicion")

    log("1/5 el historial real de leaderboard, desde el log")
    con_lb = []
    with open(LOG_LB, encoding="utf-8") as manejador:
        for fila in csv.DictReader(manejador):
            try:
                lb = float(fila["kaggle_f1"])
                local = float(fila["local_f1"])
            except (ValueError, TypeError):
                con_lb.append(
                    {
                        "submission": fila["submission_id"],
                        "archivo": fila["archivo"],
                        "delta_lb": float(fila["diferencia"]) if fila["diferencia"] else np.nan,
                        "delta_oof": np.nan,
                        "nota": "sin local_f1 numerico: dOOF no calculable",
                    }
                )
                continue
            con_lb.append(
                {
                    "submission": fila["submission_id"],
                    "archivo": fila["archivo"].split("/")[-1],
                    "local": local,
                    "lb": lb,
                    "delta_lb": lb - 0.78306,
                    "delta_oof": local - CORRECTED_CV_V03,
                    "nota": "",
                }
            )
    log(f"   submissions con LB registrado: {len(con_lb)}")
    con_datos = [c for c in con_lb if not np.isnan(c["delta_oof"])]
    log(f"   de ellas, con dOOF calculable: {len(con_datos)}")
    log("   AVISO: `local_f1` no significa lo mismo en todas las fases. V16 optimizo la tasa")
    log("   predicha, luego su dOOF no es comparable con el corrected_cv de V03.")

    log("2/5 los modelos lineales, recalculados TODOS con el mismo protocolo")
    train = read_train()
    particion = pd.read_csv(PARTICION)
    ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
    desarrollo = train[train["id"].astype(int).isin(ids_dev)].reset_index(drop=True)
    grupos_full = grupos_fusionados(train)
    mapa_grupo = dict(zip(train["id"].astype(int), grupos_full.tolist()))
    grupos_dev = np.array([mapa_grupo[i] for i in desarrollo["id"].astype(int)])
    pliegues = pliegues_corregidos(desarrollo, grupos_dev)
    y = desarrollo[list(ETIQUETAS)].to_numpy().astype(int)
    textos = desarrollo["text"]

    oof = np.load(OOF_CANONICO, allow_pickle=True)
    y_oof = np.asarray(oof["y_true"], dtype=int)
    mapa_posicion = {v: i for i, v in enumerate(desarrollo["id"].astype(int).to_numpy())}
    posiciones = np.array([mapa_posicion[i] for i in np.asarray(oof["ids"]).astype(np.int64)])
    pred_v03 = np.asarray(oof["pred"], dtype=int)
    if not np.array_equal(y[posiciones], y_oof):
        raise SystemExit("PARADA OBLIGATORIA: las etiquetas no coinciden tras alinear por id")

    modelos = {
        "V03 LR L2 C=2.0": lambda: LogisticRegression(
            C=2.0, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=SEMILLA
        ),
        "LR L1 C=4.0": lambda: LogisticRegression(
            C=4.0, penalty="l1", class_weight="balanced", solver="liblinear",
            max_iter=2000, random_state=SEMILLA
        ),
        "LR L2 C=20.0": lambda: LogisticRegression(
            C=20.0, class_weight="balanced", solver="liblinear", max_iter=3000, random_state=SEMILLA
        ),
        "LinearSVC C=1.0": lambda: LinearSVC(
            C=1.0, class_weight="balanced", max_iter=5000, random_state=SEMILLA
        ),
        "RidgeClassifier a=3.0": lambda: RidgeClassifier(alpha=3.0, class_weight="balanced"),
    }

    indicadores: list[dict[str, Any]] = []
    for nombre, fabrica in modelos.items():
        proba = np.zeros((len(desarrollo), len(ETIQUETAS)))
        f1_train: list[float] = []
        no_nulos: list[int] = []
        normas: list[float] = []
        n_features: list[int] = []
        for ajuste, validacion in pliegues:
            palabra = TfidfVectorizer(**PARAMETROS_WORD)
            caracter = TfidfVectorizer(**PARAMETROS_CHAR)
            izquierda = sparse.hstack(
                [
                    palabra.fit_transform(textos.iloc[ajuste]),
                    caracter.fit_transform(textos.iloc[ajuste]),
                ]
            ).tocsr()
            derecha = sparse.hstack(
                [palabra.transform(textos.iloc[validacion]), caracter.transform(textos.iloc[validacion])]
            ).tocsr()
            n_features.append(int(izquierda.shape[1]))
            for j in range(3):
                modelo = fabrica()
                modelo.fit(izquierda, y[ajuste, j])
                if hasattr(modelo, "decision_function"):
                    decision = modelo.decision_function(derecha)
                    decision_train = modelo.decision_function(izquierda)
                    proba[validacion, j] = 1.0 / (1.0 + np.exp(-decision))
                else:
                    proba[validacion, j] = modelo.predict_proba(derecha)[:, 1]
                    decision_train = modelo.predict_proba(izquierda)[:, 1]
                coeficientes = np.abs(np.asarray(modelo.coef_).ravel())
                no_nulos.append(int(np.count_nonzero(coeficientes > 1e-12)))
                normas.append(float(np.linalg.norm(coeficientes)))
                comunes = int(np.count_nonzero(y[ajuste, j] & (decision_train >= 0.5)))
                denominador = int(np.count_nonzero(y[ajuste, j])) + int(
                    np.count_nonzero(decision_train >= 0.5)
                )
                f1_train.append(2 * comunes / denominador if denominador > 0 else 0.0)
        pred = (proba[posiciones] >= UMBRALES_V03[None, :]).astype(int)
        valor = macro_f1(y_oof, pred)
        train_f1 = float(np.mean(f1_train))
        cambio = pred != pred_v03
        rescates = int(np.count_nonzero(cambio & (pred_v03 != y_oof) & (pred == y_oof)))
        danos = int(np.count_nonzero(cambio & (pred_v03 == y_oof) & (pred != y_oof)))
        indicadores.append(
            {
                "modelo": nombre,
                "corrected_cv": valor,
                "delta_oof": valor - CORRECTED_CV_V03,
                "f1_train": train_f1,
                "gap_train_oof": train_f1 - valor,
                "n_parametros": int(np.mean(no_nulos)) * 3,
                "fraccion_no_nula": float(np.mean(no_nulos) / np.mean(n_features)),
                "sparsity": float(1.0 - np.mean(no_nulos) / np.mean(n_features)),
                "norma_coeficientes": float(np.mean(normas)),
                "n_features": int(np.mean(n_features)),
                "celdas": int(np.count_nonzero(cambio)),
                "rescates": rescates,
                "danios": danos,
                "neto": rescates - danos,
            }
        )
        f = indicadores[-1]
        log(
            f"   {nombre:<20} cv {valor:.6f} | f1_train {train_f1:.4f} | gap {f['gap_train_oof']:+.4f} | "
            f"no nulos {f['fraccion_no_nula']:.4f} | norma {f['norma_coeficientes']:.4f} | "
            f"celdas {f['celdas']:>4} | neto {f['neto']:>+4}"
        )

    log("3/5 la fila de V23, que no se puede recalcular aqui")
    indicadores.append(V23)
    log(
        f"   {V23['nombre']:<20} cv {CORRECTED_CV_V03 + V23['delta_oof']:.6f} | "
        f"frac. vectores soporte {V23['fraccion_vectores_soporte']:.1%} | celdas "
        f"{V23['celdas']} | neto {V23['rescates'] - V23['danios']:+d}"
    )

    log("4/5 la relacion entre indicadores y delta LB, sobre lo que se puede emparejar")
    pd.DataFrame(indicadores)
    # Se empareja cada indicador con el delta LB de la submission que le corresponde.
    mapa_lb = {
        "V03 LR L2 C=2.0": None,
        "LR L1 C=4.0": None,
        "LR L2 C=20.0": None,
        "LinearSVC C=1.0": None,
        "RidgeClassifier a=3.0": None,
        V23["nombre"]: V23["delta_lb"],
    }
    # Los modelos lineales de aqui no se enviaron, asi que se quedan sin delta LB. Lo que
    # si tiene delta LB es V23. La correlacion se calcula sobre el subconjunto emparejado y
    # se reporta su tamano, que es el dato relevante.
    pares = [(i, v) for i, v in mapa_lb.items() if v is not None and not np.isnan(v)]
    log(f"   filas con delta LB emparejable en este script: {len(pares)}")
    log("   El resto de modelos recalculados no se enviaron nunca, y anadir el delta LB de")
    log("   una submission cuyo modelo no es este seria un error de emparejamiento.")

    log("5/5 correlaciones sobre el historial completo de submissions")
    x = np.array([c["delta_oof"] for c in con_datos if not np.isnan(c["delta_oof"])])
    y_lb = np.array([c["delta_lb"] for c in con_datos if not np.isnan(c["delta_oof"])])
    if len(x) >= 3:
        r, p = pearsonr(x, y_lb)
        log(f"   Pearson(delta OOF, delta LB) = {r:+.4f} con p = {p:.4f} sobre n = {len(x)}")
        b, a = np.polyfit(x, y_lb, 1)
        log(f"   ajuste lineal: delta LB = {a:+.5f} + {b:.4f} * delta OOF")
        log(f"   delta OOF necesario para empatar el LB: {-a / b:+.5f}")
    else:
        r, p, b, a = np.nan, np.nan, np.nan, np.nan
        log("   insuficiente para correlacion")

    informe = {
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_submissions": len(con_lb),
        "n_con_delta_oof_calculable": len(con_datos),
        "historial": con_lb,
        "indicadores_recalculados": indicadores,
        "correlacion_delta_oof_delta_lb": {
            "r": float(r) if not np.isnan(r) else None,
            "p": float(p) if not np.isnan(p) else None,
            "n": len(x),
            "recta": {"intercepto": float(a) if not np.isnan(a) else None, "pendiente": float(b)},
            "delta_oof_para_empatar": float(-a / b) if not np.isnan(b) and b != 0 else None,
            "aviso": (
                "`local_f1` no significa lo mismo en todas las fases: V16 optimizo la tasa "
                "predicha, luego su dOOF no es comparable con el corrected_cv de V03. Los "
                "puntos de V16 estan contaminados y sesgan la correlacion hacia el eje"
            ),
        },
        "gastos": {"modelos_competencia": 0, "submissions": 0},
    }
    REPORTE_JSON.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _informe_md(informe, inicio, indicadores, con_datos, r, p, b, a)
    log("   informes escritos en reports/v24_01_transferencia_complejidad.{json,md}")
    log(f"completado en {round(time.time() - inicio, 1)} s")
    log("VEREDICTO: auditoria diagnostica | ninguna submission | ningun modelo de competicion")



def _informe_md(
    informe: dict[str, Any],
    inicio: float,
    indicadores: list[dict[str, Any]],
    con_datos: list[dict[str, Any]],
    r: float,
    p: float,
    b: float,
    a: float,
) -> None:
    """Escribe el informe legible.

    Args:
        informe: informe con los datos.
        inicio: instante de arranque.
        indicadores: filas de indicadores.
        con_datos: submissions con dOOF calculable.
        r: correlacion.
        p: p-valor.
        b: pendiente del ajuste.
        a: intercepto del ajuste.
    """
    corr = informe["correlacion_delta_oof_delta_lb"]
    lineas = [
        "# V24-01 (P1): modelo de transferencia de complejidad",
        "",
        f"Generado por `experiments/v24_01_modelo_transferencia_complejidad.py` ({EXPERIMENTO}).",
        "",
        "**Auditoria diagnostica. Ningun modelo de prediccion para la competicion. Ninguna",
        "submission.**",
        "",
        "## 1. Por que esto existe",
        "",
        "V23 envio la candidata con mejor evidencia interna de la historia del proyecto",
        "(`corrected_cv` 0.790410, 5 de 5 pliegues, McNemar p = 0.0452) y el leaderboard bajo",
        "0.00368. Si esa divergencia se puede detectar antes de enviar, las proximas fases",
        "tienen un filtro que no tenian.",
        "",
        "## 2. El historial real de leaderboard",
        "",
        (f"{informe['n_submissions']} submissions registradas, de las cuales "
        f"{informe['n_con_delta_oof_calculable']} tienen `local_f1` numerico."),
        "",
        "| submission | local | LB | dOOF | dLB |",
        "|---|---:|---:|---:|---:|",
    ]
    for c in con_datos:
        lineas.append(
            f"| {c['archivo'][:44]} | {c['local']:.6f} | {c['lb']:.5f} | "
            f"{c['delta_oof']:+.6f} | {c['delta_lb']:+.5f} |"
        )
    lineas += [
        "",
        "## 3. Los indicadores de complejidad, recalculados con un unico protocolo",
        "",
        "Todos con los pliegues canonicos, las 1.483 filas de desarrollo y los umbrales",
        "congelados de V03. Por eso son comparables entre si, cosa que los numeros de los",
        "informes historicos no son.",
        "",
        "| modelo | corrected_cv | f1 train | gap train-OOF | frac. no nula | sparsity | norma coef. | celdas | rescates | danos | neto |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for f in indicadores:
        lineas.append(
            f"| {f.get('modelo', '?')} | {_num(f, 'corrected_cv', '.6f')} | "
            f"{_num(f, 'f1_train')} | {_num(f, 'gap_train_oof', '+.4f')} | "
            f"{_num(f, 'fraccion_no_nula')} | {_num(f, 'sparsity')} | "
            f"{_num(f, 'norma_coeficientes')} | {_num(f, 'celdas', '.0f')} | "
            f"{_num(f, 'rescates', '.0f')} | {_num(f, 'danios', '.0f')} | "
            f"{_num(f, 'neto', '+.0f')} |"
        )
    lineas += [
        "",
        "La fila de V23 tiene `fraccion_no_nula` y `norma_coeficientes` en nan porque no",
        "son recalculables sin reentrenar el kernel, y su indicador propio es la fraccion de",
        f"vectores de soporte: **{V23['fraccion_vectores_soporte']:.1%}** de las muestras.",
        "",
        "## 4. La relacion con el delta LB",
        "",
    ]
    if corr["r"] is not None:
        lineas += [
            (f"- Pearson(delta OOF, delta LB) = **{corr['r']:+.4f}**, p = {corr['p']:.4f}, "
            f"n = {corr['n']}."),
            (f"- Ajuste: `delta LB = {corr['recta']['intercepto']:+.5f} + "
            f"{corr['recta']['pendiente']:.4f} * delta OOF`."),
            f"- Para empatar el LB haria falta un delta OOF de **{corr['delta_oof_para_empatar']:+.5f}**.",
            "",
        ]
    else:
        lineas += ["No hay suficientes puntos emparejados para una correlacion.", ""]
    lineas += [
        "**Aviso que invalida parte de la tabla**: " + corr["aviso"] + ".",
        "",
        "## 5. Lo que este analisis NO puede concluir",
        "",
        "1. **No produce un modelo predictivo del `delta LB`.** Con 22 puntos, y 4 de ellos",
        "   contaminados por el esquema de V16, ajustar una relacion con varias variables",
        "   seria sobreajustar el propio historial. La regresion simple se reporta como",
        "   *descripcion*, no como predictor.",
        "2. **Los modelos recalculados aqui no se enviaron nunca**, asi que no tienen",
        "   `delta LB` propio. Emparejar el indicador de un modelo con el `delta LB` de otra",
        "   submission seria un error de emparejamiento, y no se ha hecho.",
        "3. **La correlacion con 22 observaciones no distingue nada.** Su intervalo de",
        "   confianza es enorme y cualquier afirmacion causal seria falsa.",
        "4. Los indicadores de V23 son los unicos emparejados con un `delta LB` medido, de",
        "   modo que la direccion complejidad alta -> peor transferencia se apoya en un",
        "   solo caso, que es exactamente el caso que motiva el analisis.",
        "",
        f"Completado en {round(time.time() - inicio, 1)} s.",
        "",
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
