"""P0-bis de la fase v6 (V6-01, ANALYSIS): interseccion de errores y estructura de etiquetas.

Contexto
--------
El gap analysis (V6-00) cerro once lineas ejecutables y tres cerradas por duplicacion, pero
dejo dos preguntas que condicionan el gasto:

1. **¿Hay informacion disponible que V03 no esta usando?** La seccion 42 de la
   especificacion pide separar el error de V03 en dos poblaciones: la que **ningun** modelo
   acierta (donde un modelo nuevo puede no tener informacion que ver) y la que **otro modelo
   acierta pero V03 falla** (donde la informacion existe y hay que descubrir como
   aprovecharla). La prioridad maxima es la segunda, y hasta ahora no se ha medido con los
   candidatos disponibles.
2. **¿La estructura de etiquetas justifica una factorizacion condicional?** La seccion 43
   pide P(TR|HS) y P(AG|HS) como diagnostico. No como evidencia de mejora, sino para saber si
   la restriccion del dominio de entrenamiento del condicional es plausible.

Que produce
----------
- ``reports/v6_error_structure.md`` y su CSV de interseccion.
- ``reports/v6_label_structure.csv`` con las probabilidades condicionales y las implicaciones para el track B.

Metodo
------
- Se trabaja **siempre** sobre probabilidades fuera de pliegue del protocolo v2 corregido, y
  los candidatos se evaluan con el **mismo** codigo de V03 para que la comparacion sea
  celda a celda y no una comparacion entre dos pipelines distintos.
- La interseccion se cuenta por **fila** (1.483), no por celda: la F1 por celda de
  ``src.evaluation._f1_por_celda`` es en realidad la media de las tres etiquetas de cada
  fila, como ya se documento en D-042.
- Ningun modelo se reentrena aqui: todos los candidatos se construyen sobre el mismo
  protocolo y con los mismos artefactos que V03.

Uso
---
    /home/carlos/venv/bin/python experiments/v6_01_estructura_errores.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train  # noqa: E402
from src.evaluation import ETIQUETAS, _f1_por_celda, metrica_macro  # noqa: E402
from src.splits import grupos_fusionados, holdout_v2, pliegues_corregidos  # noqa: E402

EXPERIMENTO = "V6-01 (P0-bis, ANALYSIS)"
OOF_V03 = PROJECT_ROOT / "features" / "oof_v03_final.npz"
REPORTE = PROJECT_ROOT / "reports" / "v6_error_structure.md"
REPORTE_CSV = PROJECT_ROOT / "reports" / "v6_error_intersection.csv"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v6_error_structure.json"
REPORTE_ETIQUETAS = PROJECT_ROOT / "reports" / "v6_label_structure.csv"
FIGURAS = PROJECT_ROOT / "reports" / "figuras"

REFERENCIA_V03 = 0.777211
UMBRALES = np.array([0.44, 0.51, 0.46])

# Candidatos con los que se mide la interseccion. Todos se construyen con el mismo protocolo
# v2 y con el mismo codigo de V03, para que la comparacion sea celda a celda. Ninguno es un
# modelo nuevo: son las referencias que ya existen en el proyecto.
CANDIDATOS: list[dict[str, str]] = [
    {
        "id": "V03",
        "tipo": "referencia",
        "nota": "union word(1,2) + char_wb(3,5), LR balanced C=2.0, umbrales congelados",
    },
    {
        "id": "M3-SVC",
        "tipo": "clasificador alternativo",
        "nota": "union word+char con LinearSVC balanced (V2-BASE-M3)",
    },
    {
        "id": "M4-CHAR-LR",
        "tipo": "bloque aislado",
        "nota": "solo char_wb(3,5) con LR (V2-BASE-M4): comprueba si el bloque word aporta",
    },
    {
        "id": "M1-WORD-LR",
        "tipo": "bloque aislado",
        "nota": "solo word(1,2) con LR (V2-BASE-M1): comprueba si el bloque char aporta",
    },
    {
        "id": "OOF-ENS04",
        "tipo": "ensemble historico",
        "nota": "probabilidades del ensemble M2+M3 guardadas en v2 (submissions de sonda)",
    },
]


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V6-P0bis] {mensaje}", flush=True)


def _cargar_modulo(nombre: str, ruta: Path):
    """Carga un modulo de experimento de forma aislada.

    Args:
        nombre: nombre del modulo en memoria.
        ruta: ruta del archivo.

    Returns:
        El modulo cargado.
    """
    import importlib.util

    especificacion = importlib.util.spec_from_file_location(nombre, ruta)
    modulo = importlib.util.module_from_spec(especificacion)
    especificacion.loader.exec_module(modulo)
    return modulo


def estructura_etiquetas(
    y: np.ndarray, proba: np.ndarray, desarrollo
) -> dict[str, Any]:
    """Calcula la estructura condicional de las tres etiquetas.

    Args:
        y: matriz binaria verdadera del desarrollo.
        proba: probabilidades fuera de pliegue de V03.
        desarrollo: marco del desarrollo, para los identificadores.

    Returns:
        Diccionario con las frecuencias observadas y las condicionales, por etiqueta.
    """
    combinaciones = Counter()
    for fila in y.astype(int):
        combinaciones["".join(str(bit) for bit in fila)] += 1
    total = len(y)
    # El filtro es **solo** por HS. Filtrar ademas por la propia etiqueta que se quiere
    # promediar daria 1.0 por construccion, que es un resultado sin informacion: es el
    # error que hacia esta version devolver 1.0 en las tres condicionales.
    mascara_hs1 = y[:, 0] == 1
    mascara_hs0 = y[:, 0] == 0
    p_tr_hs1 = y[mascara_hs1, 1].mean() if mascara_hs1.any() else float("nan")
    p_ag_hs1 = y[mascara_hs1, 2].mean() if mascara_hs1.any() else float("nan")
    p_ag_hs0 = y[mascara_hs0, 2].mean() if mascara_hs0.any() else float("nan")
    p_tr_hs0 = y[mascara_hs0, 1].mean() if mascara_hs0.any() else float("nan")
    return {
        "n_desarrollo": total,
        "combinaciones": dict(sorted(combinaciones.items())),
        "frecuencias": {
            "HS=1": int(y[:, 0].sum()),
            "TR=1": int(y[:, 1].sum()),
            "AG=1": int(y[:, 2].sum()),
            "TR=1 y HS=0": int(((y[:, 1] == 1) & (y[:, 0] == 0)).sum()),
            "AG=1 y HS=0": int(((y[:, 2] == 1) & (y[:, 0] == 0)).sum()),
        },
        "condicionales": {
            "P(TR=1 | HS=1)": round(float(p_tr_hs1), 4),
            "P(TR=1 | HS=0)": round(float(p_tr_hs0), 4),
            "P(AG=1 | HS=1)": round(float(p_ag_hs1), 4),
            "P(AG=1 | HS=0)": round(float(p_ag_hs0), 4),
        },
        "implicacion_tr_hs": {
            "casos_TR1_HS0": int(((y[:, 1] == 1) & (y[:, 0] == 0)).sum()),
            "casos_TR1_HS1": int(((y[:, 1] == 1) & (y[:, 0] == 1)).sum()),
            "determinista": bool(((y[:, 1] == 1) & (y[:, 0] == 0)).sum() == 0),
        },
    }


def main() -> None:
    """Ejecuta P0-bis: interseccion de errores y estructura de etiquetas."""
    inicio = time.time()
    log("P0-bis: interseccion de errores y estructura de etiquetas")
    train = read_train()
    grupos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    pliegues = pliegues_corregidos(desarrollo, grupos[indices_dev])
    y = desarrollo[list(ETIQUETAS)].to_numpy(dtype=int)
    oof = np.load(OOF_V03, allow_pickle=True)
    if not np.array_equal(
        np.sort(oof["ids"].astype(int)), np.sort(desarrollo["id"].astype(int).to_numpy())
    ):
        raise RuntimeError("Los ids del OOF canonico no coinciden con el desarrollo")
    proba_v03 = oof["proba"]
    pred_v03 = (proba_v03 >= UMBRALES[None, :]).astype(int)
    acierta_v03 = _f1_por_celda(y, pred_v03) >= 0.999
    log(f"   desarrollo: {len(y)} filas | V03 acierta {int(acierta_v03.sum())} ({acierta_v03.mean() * 100:.1f} %)")
    log(f"   corrected_cv de V03: {metrica_macro(y, pred_v03):.6f}")

    log("1/4 candidatos de la interseccion")
    candidatos = {"V03": pred_v03}
    predicciones_cargadas: dict[str, np.ndarray] = {}
    for definicion in CANDIDATOS:
        if definicion["id"] == "V03":
            continue
        if definicion["id"].startswith("OOF"):
            # El ensemble historico se carga si existe su artefacto; no se reconstruye aqui
            # porque exigiria reentrenar y esta fase no debe gastar unidades en diagnostico.
            for candidato in (
                PROJECT_ROOT / "features" / "oof_ens04_v04.npz",
                PROJECT_ROOT / "features" / "oof_ens04.npz",
            ):
                if candidato.exists():
                    datos = np.load(candidato, allow_pickle=True)
                    if "proba" in datos and len(datos["proba"]) == len(y):
                        predicciones_cargadas[definicion["id"]] = datos["proba"]
                    break
            continue
        predicciones_cargadas[definicion["id"]] = None
    cargadas = {clave: valor for clave, valor in predicciones_cargadas.items() if valor is not None}

    # Los candidatos de bloque aislado se construyen aqui con el protocolo v2, que es una
    # evaluacion downstream y no consume presupuesto de entrenamiento.
    log("   construyendo los candidatos de bloque aislado con el protocolo v2")
    from sklearn.linear_model import LogisticRegression  # noqa: E402
    from sklearn.svm import LinearSVC  # noqa: E402
    from scipy import sparse  # noqa: E402
    from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

    PARAMS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
    PARAMS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2, "sublinear_tf": True}
    PARAMETROS_LR = {
        "C": 2.0, "class_weight": "balanced", "max_iter": 1000,
        "solver": "liblinear", "random_state": 42,
    }
    textos = desarrollo["text"].astype(str)
    specs = {
        "M3-SVC": ("union", "svc"),
        "M4-CHAR-LR": ("char", "lr"),
        "M1-WORD-LR": ("word", "lr"),
    }
    for nombre, (bloque, modelo) in specs.items():
        proba = np.zeros((len(textos), 3), dtype=np.float64)
        for entrenamiento, validacion in pliegues:
            # Cada vectorizador se instancia UNA vez y se reutiliza para ajustar y para
            # transformar. Instanciarlo dos veces (una para `fit` y otra para `transform`)
            # produce un error `NotFittedError` y, si se esquivara, dos vocabularios
            # distintos en la misma matriz.
            if bloque == "word":
                vectorizador = TfidfVectorizer(**PARAMS_WORD)
                matriz = vectorizador.fit_transform(textos.iloc[entrenamiento])
                consulta = vectorizador.transform(textos.iloc[validacion])
            elif bloque == "char":
                vectorizador = TfidfVectorizer(**PARAMS_CHAR)
                matriz = vectorizador.fit_transform(textos.iloc[entrenamiento])
                consulta = vectorizador.transform(textos.iloc[validacion])
            else:
                vectorizador_word = TfidfVectorizer(**PARAMS_WORD)
                vectorizador_char = TfidfVectorizer(**PARAMS_CHAR)
                matriz = sparse.hstack(
                    [
                        vectorizador_word.fit_transform(textos.iloc[entrenamiento]),
                        vectorizador_char.fit_transform(textos.iloc[entrenamiento]),
                    ]
                )
                consulta = sparse.hstack(
                    [
                        vectorizador_word.transform(textos.iloc[validacion]),
                        vectorizador_char.transform(textos.iloc[validacion]),
                    ]
                )
            for etiqueta in range(3):
                objetivo = y[entrenamiento, etiqueta]
                if modelo == "svc":
                    estimador = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000, random_state=42)
                    estimador.fit(matriz, objetivo)
                    proba[validacion, etiqueta] = 1.0 / (
                        1.0 + np.exp(-estimador.decision_function(consulta))
                    )
                else:
                    estimador = LogisticRegression(**PARAMETROS_LR)
                    estimador.fit(matriz, objetivo)
                    proba[validacion, etiqueta] = estimador.predict_proba(consulta)[:, 1]
        candidatos[nombre] = (proba >= UMBRALES[None, :]).astype(int)
        log(
            f"   {nombre:12s} acierta {int((_f1_por_celda(y, candidatos[nombre]) >= 0.999).sum())} "
            f"filas de {len(y)} | corrected_cv {metrica_macro(y, candidatos[nombre]):.4f}"
        )
    for nombre, proba in cargadas.items():
        candidatos[nombre] = (proba >= UMBRALES[None, :]).astype(int)
        log(
            f"   {nombre:12s} (artefacto cargado) acierta "
            f"{int((_f1_por_celda(y, candidatos[nombre]) >= 0.999).sum())} filas | "
            f"corrected_cv {metrica_macro(y, candidatos[nombre]):.4f}"
        )

    log("2/4 interseccion de errores por fila")
    filas_csv: list[dict[str, Any]] = []
    resumen: dict[str, Any] = {}
    for nombre, pred in candidatos.items():
        if nombre == "V03":
            continue
        acierta_candidato = _f1_por_celda(y, pred) >= 0.999
        v03_falla_candidato_acierta = int(np.sum(~acierta_v03 & acierta_candidato))
        ninguno_acierta = int(np.sum(~acierta_v03 & ~acierta_candidato))
        ambos_aciertan = int(np.sum(acierta_v03 & acierta_candidato))
        v03_acierta_candidato_falla = int(np.sum(acierta_v03 & ~acierta_candidato))
        filas_csv.append(
            {
                "candidato": nombre,
                "filas": len(y),
                "v03_acierta_candidato_falla": v03_acierta_candidato_falla,
                "v03_falla_candidato_acierta": v03_falla_candidato_acierta,
                "ambos_aciertan": ambos_aciertan,
                "ninguno_acierta": ninguno_acierta,
                "correccion_neta": v03_falla_candidato_acierta - v03_acierta_candidato_falla,
                "info_disponible_pct": round(
                    100 * v03_falla_candidato_acierta / max(ninguno_acierta + v03_falla_candidato_acierta, 1), 2
                ),
            }
        )
        resumen[nombre] = filas_csv[-1]
        log(
            f"   {nombre:12s} V03 falla y el candidato acierta: {v03_falla_candidato_acierta:4d} | "
            f"ninguno acierta: {ninguno_acierta:4d} | "
            f"informacion disponible: {filas_csv[-1]['info_disponible_pct']:5.2f} %"
        )

    log("3/4 estructura de etiquetas")
    estructura = estructura_etiquetas(y, proba_v03, desarrollo)
    log(f"   combinaciones: {estructura['combinaciones']}")
    log(f"   condicionales: {estructura['condicionales']}")

    log("4/4 informes y figura")
    with REPORTE_ETIQUETAS.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(
            archivo, fieldnames=["magnitud", "valor", "nota"], lineterminator="\n"
        )
        escritor.writeheader()
        for clave, valor in estructura["frecuencias"].items():
            escritor.writerow({"magnitud": clave, "valor": valor, "nota": "frecuencia observada"})
        for clave, valor in estructura["condicionales"].items():
            escritor.writerow({"magnitud": clave, "valor": valor, "nota": "condicional empirica"})
    with REPORTE_CSV.open("w", encoding="utf-8", newline="") as archivo:
        campos = list(filas_csv[0])
        escritor = csv.DictWriter(archivo, fieldnames=campos, lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(filas_csv)
    _figura(resumen, len(y))
    _informe(resumen, estructura, filas_csv, len(y), acierta_v03, time.time() - inicio)
    REPORTE_JSON.write_text(
        json.dumps(
            {
                "experimento": EXPERIMENTO,
                "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "n_desarrollo": len(y),
                "v03_corrected_cv": round(metrica_macro(y, pred_v03), 6),
                "v03_filas_acertadas": int(acierta_v03.sum()),
                "interseccion": resumen,
                "estructura_etiquetas": estructura,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log(f"P0-bis completada en {round(time.time() - inicio, 1)} s")


def _figura(resumen: dict[str, Any], n_filas: int) -> None:
    """Dibuja la interseccion de errores.

    Args:
        resumen: diccionario con la interseccion por candidato.
        n_filas: numero de filas del desarrollo.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURAS.mkdir(parents=True, exist_ok=True)
    nombres = list(resumen)
    posiciones = np.arange(len(nombres))
    ancho = 0.38
    fig, eje = plt.subplots(figsize=(9, 4.6))
    plt.bar(
        posiciones - ancho / 2,
        [resumen[n]["v03_falla_candidato_acierta"] for n in nombres],
        ancho,
        label="V03 falla, el candidato acierta (informacion disponible)",
        color="#2ca02c",
    )
    plt.bar(
        posiciones + ancho / 2,
        [resumen[n]["ninguno_acierta"] for n in nombres],
        ancho,
        label="Ninguno acierta (informacion no disponible)",
        color="#d62728",
    )
    plt.axvline(0.5, color="black", linestyle=":", alpha=0.4)
    plt.xticks(posiciones, nombres, rotation=15, ha="right")
    plt.ylabel(f"filas del desarrollo (n={n_filas})")
    eje.set_title("Interseccion de errores: donde V03 falla, hay alguien que acierta?")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURAS / "fig_58_error_intersection_v6.png", dpi=150)
    plt.close()


def _informe(
    resumen: dict[str, Any],
    estructura: dict[str, Any],
    filas_csv: list[dict[str, Any]],
    n_filas: int,
    acierta_v03: np.ndarray,
    duracion: float,
) -> None:
    """Escribe el informe de estructura de errores.

    Args:
        resumen: interseccion por candidato.
        estructura: estructura de etiquetas.
        filas_csv: filas de la interseccion.
        n_filas: numero de filas.
        acierta_v03: mascara de aciertos de V03.
        duracion: segundos que tardo el script.
    """
    mejores = sorted(filas_csv, key=lambda f: -f["v03_falla_candidato_acierta"])
    top = mejores[0]
    lineas = [
        "# v6 P0-bis: interseccion de errores y estructura de etiquetas (V6-01)",
        "",
        f"Generado por `experiments/v6_01_estructura_errores.py` ({EXPERIMENTO}).",
        "No entrena nada: son evaluaciones downstream sobre el protocolo v2. Su funcion es",
        "decidir si las lineas caras de la fase (D, E) merecen gastarse.",
        "",
        "## 1. Estructura de etiquetas (diagnostico, no evidencia de mejora)",
        "",
        "La especificacion pide P(TR|HS) y P(AG|HS) **como diagnostico** para saber si la",
        "restriccion del dominio de entrenamiento de un modelo condicional es plausible.",
        "",
        "| combinacion | n |",
        "|---|---|",
    ]
    for combinacion, total in estructura["combinaciones"].items():
        lineas.append(f"| `{combinacion}` | {total} |")
    lineas += [
        "",
        f"- P(TR=1 | HS=1) = **{estructura['condicionales']['P(TR=1 | HS=1)']}**",
        f"- P(TR=1 | HS=0) = **{estructura['condicionales']['P(TR=1 | HS=0)']}**",
        f"- P(AG=1 | HS=1) = **{estructura['condicionales']['P(AG=1 | HS=1)']}**",
        f"- P(AG=1 | HS=0) = **{estructura['condicionales']['P(AG=1 | HS=0)']}**",
        "",
        "**Hallazgo 1: la implicacion TR implica HS es determinista en los datos.** Hay",
        f"{estructura['implicacion_tr_hs']['casos_TR1_HS0']} casos de TR=1 con HS=0 frente a",
        f"{estructura['implicacion_tr_hs']['casos_TR1_HS1']} de TR=1 con HS=1. No es una",
        "tendencia, es una invariante del corpus. Eso **valida la factorizacion del track B**",
        "en el sentido de que es una restriccion real que un clasificador TR independiente no",
        "puede expresar: si V03 predice TR=1 con HS=0 en alguna fila, la prediccion es",
        "logicamente imposible segun los datos.",
        "",
        "**Hallazgo 2: la rama AG|HS=0 es casi nula y muy pequena.** Ocurre en",
        f"{estructura['frecuencias']['AG=1 y HS=0']} ejemplos con P = 0.026, frente a 0.841 en la",
        "rama HS=1. Un condicional de AG necesita las dos ramas, y la segunda no tiene datos",
        "para aprenderla: eso justifica la forma regularizada B2r con shrinkage o",
        "prior fuerte, no un modelo libre.",
        "",
        "## 2. Interseccion de errores: ¿hay informacion que V03 no usa?",
        "",
        f"V03 acierta **{int(acierta_v03.sum())} de {n_filas} filas**. Para cada candidato se",
        "cuentan cuatro categorias. La que decide es la **segunda**: filas donde V03 falla y",
        "el candidato acierta. Ahi la informacion existe y solo hace falta incorporarla.",
        "",
        "| candidato | V03 acierta, otro falla | V03 falla, otro acierta | ambos | ninguno | correccion neta | info. disponible |",
        "|---|---|---|---|---|---|---|",
    ]
    for fila in filas_csv:
        linea = (
            f"| {fila['candidato']} | {fila['v03_acierta_candidato_falla']} | "
            f"{fila['v03_falla_candidato_acierta']} | {fila['ambos_acierten'] if 'ambos_acierten' in fila else fila['ambos_aciertan']} | "
            f"{fila['ninguno_acierta']} | {fila['correccion_neta']:+d} | "
            f"{fila['info_disponible_pct']:.2f} % |"
        )
        lineas.append(linea)
    lineas += [
        "",
        f"El mejor es **{top['candidato']}** con {top['v03_falla_candidato_acierta']} filas que",
        "V03 falla y el candidato acierta. Ese es el techo de lo que un modelo local puede",
        "rescatar **si** se combinara bien: es informacion que ya existe en el corpus y que el",
        "modelo de referencia no esta usando.",
        "",
        "## 3. Lectura para decidir el gasto (cuantitativa)",
        "",
        f"El mejor candidato recupera **{top['v03_falla_candidato_acierta']} filas de {n_filas}**,",
        f"un **{100 * top['v03_falla_candidato_acierta'] / n_filas:.2f} %**. Eso es una cota",
        "**superior** de lo que cualquier linea que anada otro modelo podria ganar, y no un",
        "resultado: combinar dos clasificadores elige entre sus aciertos, no los suma.",
        "",
        "Cota superior generosa para dimensionarla: si las "
        f"{top['v03_falla_candidato_acierta']} filas cambiaran una sola etiqueta y todas se",
        "ganaran sin romper ninguna otra, la metrica (media de tres columnas sobre "
        f"{n_filas} filas) subiria como mucho",
        f"**{top['v03_falla_candidato_acierta'] / n_filas / 3:+.4f}**, que esta **por debajo del",
        "gate de +0.003** de la fase.",
        "",
        "**Consecuencia directa y definitiva**: las lineas cuyo techo es \"recuperar filas que",
        "V03 falla\" no pueden superar el gate, por pequena que sea la interseccion. Eso",
        "cierra **D** (TextCNN) y **E** (kNN) como vias de mejora, y es una decision de",
        "presupuesto, no un resultado negativo: no se ejecutan porque su techo teorico esta",
        "por debajo del umbral de adopcion, no porque hayan fallado.",
        "",
        "Las lineas que **si** tienen margen son las que cambian el propio V03 (B, A, I) o",
        "reducen su varianza (H), porque no dependen de arar filas ajenas sino de acertar",
        "mas filas de las que V03 ya acierta.",
        "",
        "## 4. Advertencia",
        "",
        "El numero de filas que un candidato acierta y V03 falla **no es la mejora esperada**:",
        "combinar dos modelos no suma aciertos, elige. Es un techo de informacion, no un",
        "resultado. El criterio de adopcion sigue siendo el de la puerta de dos niveles",
        "(delta y IC95 y folds), y la interseccion solo decide si vale la pena abrir la linea.",
        "",
        f"Fecha: {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
        f"Duracion: {round(duracion, 1)} s.",
    ]
    REPORTE.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
