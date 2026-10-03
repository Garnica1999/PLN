"""P2 de la fase v5.1 (EBD-19): familia window (3 y 7).

Contexto
--------
P1 cerro ``vector_size`` como inerte (D-036): 100d, 200d y 300d dan una dispersion interna
de 0.0017 y ninguna variante es adoptable. La dimension no es la palanca. La siguiente
pregunta arquitectonica es el **tamaño de ventana**, que controla cuanta distribucion se
contexto alrededor de cada palabra: con ventana 3 el modelo ve el vecindario inmediato y
con ventana 7 un contexto mas amplio.

Hipotesis de la familia
-----------------------
Con 10M tokens y 5 epocas, una ventana mayor **no** aporta: cada palabra ya acumula
decenas de ocurrencias y el corpus se queda pequeno antes de que la ventana sea el
cuello de botella. Si la ventana no cambia nada (100d, 200d y 300d tampoco), la evidencia
acumulada dira que el embedding esta limitado por los datos etiquetados y no por la
funcion objetivo, y eso cerraria la linea de la arquitectura casi entera.

Protocolo
---------
Igual que en P1: corpus E-CORP-01 (Wikipedia 10M), vector_size 200, min_count 3,
subpalabras 3-6, negative 10, 5 epocas, sg skipgram, seed 42, bucket 200.000. Cambia
**solo** ``window``. El 200d/ventana 5 es el baseline B0 y se reevalua en esta misma
corrida como control; debe reproducir 0.7818 exacto, y si no, la comparacion no vale.

Maquinaria reutilizable
----------------------
Este modulo define ``ejecutar_familia``, la maquinaria comun de las familias
arquitectonicas: entrenar, comprobar la firma, reevaluar el control, medir las tres
evaluaciones downstream por variante, calcular el gate de dos niveles y escribir informe,
figura y CSV. P3, P4 y P5 la invocan con sus propios parametros en vez de duplicarla,
que es lo que garantiza que las cuatro familias se midan con el mismo punto de
operacion, la misma referencia y el mismo criterio de cierre.

Uso
---
    /home/carlos/venv/bin/python experiments/embedding_13_window_v5_1.py
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train  # noqa: E402
from src.evaluation import ETIQUETAS, bootstrap_diferencia_pareada, metrica_macro  # noqa: E402
from src.splits import grupos_fusionados, holdout_v2, pliegues_corregidos  # noqa: E402

EXPERIMENTO = "EBD-19 (P2)"
PRESUPUESTO_JSON = PROJECT_ROOT / "reports" / "embedding_budget_v5_1.json"
TABLA_A = PROJECT_ROOT / "reports" / "embedding_architecture_results_v5_1.csv"
TABLA_B = PROJECT_ROOT / "reports" / "embedding_downstream_evaluations_v5_1.csv"
OOF_V03 = PROJECT_ROOT / "features" / "oof_v03_final.npz"
FIGURAS = PROJECT_ROOT / "reports" / "figuras"

REFERENCIA_V03 = 0.777211
UMBRALES_V03_FROZEN = np.array([0.44, 0.51, 0.46])
MEJORA_MINIMA = 0.005
REPETICIONES_BOOTSTRAP = 10_000
SEMILLA_BOOTSTRAP = 20260925
DISPERSION_MAXIMA_FAMILIA = 0.005

FAMILIA = "window"
TITULO = "Familia window (tamano de ventana de contexto)"
UNIDADES = 2
EJE_X = "window"
FIGURA = "fig_51_window_v5_1.png"
REPORTE = PROJECT_ROOT / "reports" / "embedding_window_v5_1.md"
REPORTE_CSV = PROJECT_ROOT / "reports" / "embedding_window_results_v5_1.csv"
REPORTE_REBANADAS = PROJECT_ROOT / "reports" / "embedding_window_slices_v5_1.csv"
PROPUESTAS = [
    {"codigo": "WS-3", "cambios": {"ventana": 3}, "etiqueta": "3"},
    {"codigo": "WS-7", "cambios": {"ventana": 7}, "etiqueta": "7"},
]
PUNTO_EXISTENTE = {
    "codigo": "B0-200",
    "training_id": "FASTTEXT-WIKI-10M",
    "etiqueta": "5",
}


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V5.1-{FAMILIA}] {mensaje}", flush=True)


def ejecutar_familia(
    *,
    familia: str,
    titulo: str,
    unidades: int,
    eje_x: str,
    figura: str,
    reporte: Path,
    reporte_csv: Path,
    reporte_rebanadas: Path,
    propuestas: list[dict[str, Any]],
    punto_existente: dict[str, str],
    etiqueta_de: Any,
    lectura: str,
    experimento: str,
) -> dict[str, Any]:
    """Ejecuta una familia arquitectonica completa y devuelve su veredicto.

    Args:
        familia: nombre de la familia (vector_size, window, min_count, subwords, negative).
        titulo: titulo legible del informe.
        unidades: numero de entrenamientos exploratorios que consume la familia.
        eje_x: nombre del eje horizontal de la figura.
        figura: nombre del archivo de figura.
        reporte: ruta del informe en Markdown.
        reporte_csv: ruta del CSV de resultados.
        reporte_rebanadas: ruta del CSV de rebanadas.
        propuestas: lista de ``{"codigo", "cambios", "etiqueta"}`` a entrenar.
        punto_existente: identificacion del baseline B0, que se reevalua como control.
        etiqueta_de: funcion que extrae la etiqueta del eje de una ficha.
        lectura: texto con la lectura esperada de la familia.
        experimento: identificador del experimento para los informes.

    Returns:
        Diccionario con el veredicto de la familia.
    """
    inicio = time.time()
    log(f"{familia}: {titulo}")
    modulo_p0 = _cargar_modulo(
        "embedding_11_p0", PROJECT_ROOT / "experiments" / "embedding_11_auditoria_firmas_v5_1.py"
    )
    modulo_p1 = _cargar_modulo(
        "embedding_12_p1", PROJECT_ROOT / "experiments" / "embedding_12_vector_size_v5_1.py"
    )
    modulo_p1corp = _cargar_modulo(
        "embedding_01_p1", PROJECT_ROOT / "experiments" / "embedding_01_corpus_wikipedia_v5.py"
    )
    modulo_p2 = _cargar_modulo(
        "embedding_02_p2", PROJECT_ROOT / "experiments" / "embedding_02_fasttext_baseline_v5.py"
    )
    modulo_p3 = _cargar_modulo(
        "embedding_03_p3", PROJECT_ROOT / "experiments" / "embedding_03_fusion_complementariedad_v5.py"
    )

    presupuesto = json.loads(PRESUPUESTO_JSON.read_text(encoding="utf-8"))
    firmas_existentes = {
        fila["training_signature"]: fila["training_id"] for fila in presupuesto["tabla_a"]
    }
    for item in presupuesto["propuestas"]:
        if item["status"] != "PENDING":
            firmas_existentes[item["training_signature"]] = item["training_id"]
    corpus_meta = {
        "processed_sha256": presupuesto["corpus"]["E-CORP-01"]["processed_sha256"],
        "corpus_version": presupuesto["corpus"]["E-CORP-01"]["corpus_version"],
    }
    log(f"   control: {punto_existente['training_id']} (ya entrenado, se reevalua)")

    log("1/5 entrenamiento")
    tipos, cobertura = modulo_p1.tipos_toxic_presentes(3, modulo_p1corp)
    log(
        f"   cobertura TOXIC (train+test, como v5): por tipo "
        f"{cobertura['type_coverage']:.4f}, por token {cobertura['token_coverage']:.4f} | "
        f"tipos del train a consultar: {len(tipos)}"
    )
    fichas = []
    for propuesta in propuestas:
        fichas.append(
            entrenar_variante(
                propuesta, modulo_p0, firmas_existentes, cobertura, corpus_meta, tipos, familia
            )
        )
    fila_base = next(
        fila for fila in presupuesto["tabla_a"] if fila["training_id"] == punto_existente["training_id"]
    )
    fichas = [
        {
            "codigo": punto_existente["codigo"],
            "training_id": punto_existente["training_id"],
            "familia": familia,
            "etiqueta": punto_existente["etiqueta"],
            "vector_size": fila_base["vector_size"],
            "window": fila_base["window"],
            "min_count": fila_base["min_count"],
            "min_n": fila_base["min_n"],
            "max_n": fila_base["max_n"],
            "negative": fila_base["negative"],
            "corpus": fila_base["corpus"],
            "tokens": fila_base["tokens"],
            "vocabulary_size": fila_base["vocabulary_size"],
            "n_tipos_consultados": len(tipos),
            "n_tipos_con_vector": 6059,
            "training_time_s": fila_base["training_time_s"],
            "model_sha256": fila_base["model_sha256"],
            "training_signature": fila_base["training_signature"],
            "status": "EXISTING",
            "rol": "punto medio de la familia, ya entrenado en v5",
            "cobertura": cobertura,
            "sifts": "data/external/embeddings/fasttext-baseline_sifts.txt",
        }
    ] + fichas

    log("2/5 evaluacion downstream")
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
    pred_v03 = oof["pred"]
    from src.embedding_utils import normalizar_emb_normal

    textos_normalizados = [normalizar_emb_normal(texto) for texto in desarrollo["text"].astype(str)]

    resultados: list[dict[str, Any]] = []
    predicciones: dict[str, np.ndarray] = {"V03-CONGELADO": pred_v03}
    for ficha in fichas:
        vectores = modulo_p3.cargar_sifts(PROJECT_ROOT / ficha["sifts"])
        embeddings = modulo_p3.vectores_documento(textos_normalizados, vectores)
        codigo = ficha["codigo"]
        for clasificador in ("LR", "SVC"):
            downstream = modulo_p2.evaluar_downstream(
                f"EMB-{codigo}-{clasificador}", embeddings, y, pliegues, escalador=True
            )
            pred_frozen = (downstream["proba"] >= UMBRALES_V03_FROZEN[None, :]).astype(int)
            metricas_frozen = metrica_macro(y, pred_frozen)
            resultados.append(
                {
                    "codigo": codigo,
                    "eje": etiqueta_de(ficha),
                    "evaluacion": f"EMB-{codigo}-{clasificador}",
                    "pooling": "media",
                    "clasificador": clasificador,
                    "corrected_cv": round(metricas_frozen, 6),
                    "corrected_cv_ajustados": downstream["corrected_cv_umbrales_ajustados"],
                    "delta_vs_V03": round(metricas_frozen - REFERENCIA_V03, 6),
                    "std_pliegues": downstream["std_pliegues_0.5"],
                    "ic95_0": "",
                    "ic95_1": "",
                    "f1_HS": round(downstream["f1_por_etiqueta_ajustados"]["HS"], 4),
                    "f1_TR": round(downstream["f1_por_etiqueta_ajustados"]["TR"], 4),
                    "f1_AG": round(downstream["f1_por_etiqueta_ajustados"]["AG"], 4),
                }
            )
            predicciones[resultados[-1]["evaluacion"]] = pred_frozen
            log(
                f"   {resultados[-1]['evaluacion']:26s} frozen={metricas_frozen:.4f} "
                f"ajustado={downstream['corrected_cv_umbrales_ajustados']:.4f}"
            )
        fusion = modulo_p3.evaluar_variante(
            f"FUSION-{codigo}", desarrollo["text"], y, pliegues, embeddings, modo_embedding="l2"
        )
        comparacion = bootstrap_diferencia_pareada(
            y, fusion["pred_frozen"], pred_v03, REPETICIONES_BOOTSTRAP, SEMILLA_BOOTSTRAP
        )
        matriz = modulo_p3.matriz_complementariedad(y, fusion["pred_frozen"], pred_v03)
        resultados.append(
            {
                "codigo": codigo,
                "eje": etiqueta_de(ficha),
                "evaluacion": f"FUSION-{codigo}",
                "pooling": "media",
                "clasificador": "LR + TF-IDF V03",
                "corrected_cv": fusion["corrected_cv_umbrales_frozen_v03"],
                "corrected_cv_ajustados": fusion["corrected_cv_umbrales_ajustados"],
                "delta_vs_V03": round(
                    fusion["corrected_cv_umbrales_frozen_v03"] - REFERENCIA_V03, 6
                ),
                "std_pliegues": fusion["std_pliegues_0.5"],
                "ic95_0": comparacion["ic95"][0],
                "ic95_1": comparacion["ic95"][1],
                "celdas_mejoradas": comparacion["celdas_mejoradas"],
                "celdas_empeoradas": comparacion["celdas_empeoradas"],
                "solo_candidato": matriz["solo_candidato"],
                "solo_v03": matriz["solo_referencia"],
                "f1_HS": round(fusion["f1_por_etiqueta_ajustados"]["HS"], 4),
                "f1_TR": round(fusion["f1_por_etiqueta_ajustados"]["TR"], 4),
                "f1_AG": round(fusion["f1_por_etiqueta_ajustados"]["AG"], 4),
            }
        )
        predicciones[resultados[-1]["evaluacion"]] = fusion["pred_frozen"]
        log(
            f"   {resultados[-1]['evaluacion']:26s} frozen={fusion['corrected_cv_umbrales_frozen_v03']:.4f} "
            f"delta={resultados[-1]['delta_vs_V03']:+.4f} IC95 "
            f"[{comparacion['ic95'][0]:+.4f}, {comparacion['ic95'][1]:+.4f}]"
        )

    log("3/5 slices")
    mascaras = modulo_p3.rebanadas(desarrollo, y)
    metricas_rebanada = modulo_p3.metricas_por_rebanada(y, predicciones, mascaras)

    log("4/5 gate de dos niveles y actualizacion del presupuesto")
    fusiones = {f["eje"]: f for f in resultados if f["evaluacion"].startswith("FUSION-")}
    emb_lr = {f["eje"]: f for f in resultados if f["evaluacion"].endswith("-LR")}
    valores = [f["corrected_cv"] for f in fusiones.values()]
    dispersion = round(max(valores) - min(valores), 6)
    mejor = max(fusiones.values(), key=lambda f: f["corrected_cv"])
    supera_gate = mejor["delta_vs_V03"] >= MEJORA_MINIMA
    ic95_excluye_cero = mejor["ic95_0"] > 0
    adoptable = supera_gate and ic95_excluye_cero
    margen = mejor["delta_vs_V03"] - MEJORA_MINIMA
    cerrada = (not adoptable) and dispersion <= DISPERSION_MAXIMA_FAMILIA
    if adoptable:
        decision = "ABIERTA: una variante supera el gate con IC95 que aparta el 0"
    elif cerrada:
        decision = (
            f"CERRADA: inerte. Ninguna variante es adoptable y la dispersion interna es "
            f"{dispersion:.4f} <= {DISPERSION_MAXIMA_FAMILIA}"
        )
    else:
        decision = "ABIERTA: dispersion interna por encima de 0.005 sin variante adoptable"
    log(
        f"   mejor fusion: {mejor['evaluacion']} {mejor['corrected_cv']:.4f} "
        f"(delta {mejor['delta_vs_V03']:+.4f}, IC95 [{mejor['ic95_0']:+.4f}, "
        f"{mejor['ic95_1']:+.4f}]) | dispersion {dispersion:.4f} | {decision}"
    )

    gastadas_previas = {
        item["codigo"] for item in presupuesto["propuestas"] if item["status"] != "PENDING"
    }
    nuevas = {f["codigo"] for f in fichas[1:] if f["codigo"] not in gastadas_previas}
    gasto = len(gastadas_previas | nuevas)
    presupuesto["presupuesto"]["new_v5_1_trainings"] = gasto
    presupuesto["presupuesto"]["total_new_budget_used"] = gasto + presupuesto["presupuesto"][
        "reserved_training_units"
    ]
    presupuesto["presupuesto"]["budget_remaining"] = (
        presupuesto["presupuesto"]["limite_exploratorio"]
        + presupuesto["presupuesto"]["limite_reserva"]
        - presupuesto["presupuesto"]["total_new_budget_used"]
    )
    presupuesto["aserciones"]["presupuesto_exploratorio_no_superado"] = (
        gasto <= presupuesto["presupuesto"]["limite_exploratorio"]
    )
    presupuesto["aserciones"]["presupuesto_total_no_superado"] = (
        presupuesto["presupuesto"]["total_new_budget_used"]
        <= presupuesto["presupuesto"]["limite_exploratorio"]
        + presupuesto["presupuesto"]["limite_reserva"]
    )
    ya_registrados = {fila["training_id"] for fila in presupuesto["tabla_a"]}
    presupuesto["tabla_a"].extend(
        {campo: ficha.get(campo, "") for campo in presupuesto["tabla_a"][0]}
        for ficha in fichas[1:]
        if ficha["training_id"] not in ya_registrados
    )
    codigos_gastados = {f["codigo"] for f in fichas[1:]}
    for item in presupuesto["propuestas"]:
        if item["codigo"] in codigos_gastados:
            item["status"] = "EXISTING"
            item["presupuesto"] = "gastada"
    presupuesto["familias_evaluadas"] = [
        entrada
        for entrada in presupuesto.get("familias_evaluadas", [])
        if entrada["familia"] != familia
    ]
    presupuesto["familias_evaluadas"].append(
        {
            "familia": familia,
            "unidades_gastadas": unidades,
            "dispersion_interna": dispersion,
            "mejor_fusion": mejor["evaluacion"],
            "mejor_fusion_corrected_cv": mejor["corrected_cv"],
            "mejor_delta": mejor["delta_vs_V03"],
            "supera_gate_numerico": supera_gate,
            "ic95_excluye_cero": ic95_excluye_cero,
            "adoptable": adoptable,
            "margen_sobre_el_umbral": round(margen, 6),
            "decision": decision,
        }
    )
    PRESUPUESTO_JSON.write_text(
        json.dumps(presupuesto, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with TABLA_A.open(encoding="utf-8") as archivo:
        campos = list(csv.DictReader(archivo).fieldnames)
    with TABLA_A.open("a", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=campos, lineterminator="\n")
        for ficha in fichas[1:]:
            escritor.writerow({campo: ficha.get(campo, "") for campo in campos})
    with TABLA_B.open(encoding="utf-8") as archivo:
        campos_b = list(csv.DictReader(archivo).fieldnames)
        siguiente = len(list(csv.DictReader(TABLA_B.open(encoding="utf-8")))) + 1
    with TABLA_B.open("a", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=campos_b, lineterminator="\n")
        for fila in resultados:
            escritor.writerow(
                {
                    "evaluation_id": f"EVAL-V51-{siguiente:03d}",
                    "script": f"experiments/{experimento.lower().replace(' ', '_')}",
                    "variante": fila["evaluacion"],
                    "training_id": next(
                        f["training_id"] for f in fichas if f["codigo"] == fila["codigo"]
                    ),
                    "pooling": fila["pooling"],
                    "classifier": fila["clasificador"],
                    "modo_embedding": "l2"
                    if fila["evaluacion"].startswith("FUSION")
                    else "estandar",
                    "corrected_cv": fila["corrected_cv"],
                    "delta_vs_V03": fila["delta_vs_V03"],
                    "ic95": f"[{fila['ic95_0']}, {fila['ic95_1']}]" if fila["ic95_0"] != "" else "",
                    "decision": f"familia {familia}",
                }
            )
            siguiente += 1
    columnas = [
        "codigo", "eje", "evaluacion", "pooling", "clasificador", "corrected_cv",
        "corrected_cv_ajustados", "delta_vs_V03", "std_pliegues", "ic95_0", "ic95_1",
        "celdas_mejoradas", "celdas_empeoradas", "solo_candidato", "solo_v03",
        "f1_HS", "f1_TR", "f1_AG",
    ]
    with reporte_csv.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(
            archivo, fieldnames=columnas, lineterminator="\n", extrasaction="ignore"
        )
        escritor.writeheader()
        escritor.writerows(resultados)
    with reporte_rebanadas.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(
            archivo, fieldnames=["rebanada", "variante", "n_filas", "f1_por_celda"],
            lineterminator="\n",
        )
        escritor.writeheader()
        escritor.writerows(metricas_rebanada)

    log("5/5 figura e informe")
    _figura(fusiones, emb_lr, eje_x, figura, titulo)
    _informe(
        familia,
        titulo,
        unidades,
        fichas,
        resultados,
        metricas_rebanada,
        mascaras,
        cobertura,
        lectura,
        decision,
        dispersion,
        mejor,
        {
            "supera_gate": supera_gate,
            "ic95_excluye_cero": ic95_excluye_cero,
            "adoptable": adoptable,
            "margen": margen,
        },
        reporte,
        experimento,
        time.time() - inicio,
    )
    log(
        f"{familia} completada en {round(time.time() - inicio, 1)} s | presupuesto usado "
        f"{gasto}/9 | {decision}"
    )
    return {
        "familia": familia,
        "decision": decision,
        "dispersion": dispersion,
        "mejor": mejor,
        "gasto": gasto,
    }


def _cargar_modulo(nombre: str, ruta: Path):
    """Carga un modulo de experimento de forma aislada.

    Args:
        nombre: nombre del modulo en memoria.
        ruta: ruta del archivo.

    Returns:
        El modulo cargado.
    """
    especificacion = importlib.util.spec_from_file_location(nombre, ruta)
    modulo = importlib.util.module_from_spec(especificacion)
    especificacion.loader.exec_module(modulo)
    return modulo


def _figura(
    fusiones: dict[str, dict[str, Any]],
    emb_lr: dict[str, dict[str, Any]],
    eje_x: str,
    figura: str,
    titulo: str,
) -> None:
    """Dibuja el efecto de la variable sobre la fusion y el embedding aislado.

    Args:
        fusiones: resultados de fusion por valor del eje.
        emb_lr: resultados del embedding aislado con LR por valor del eje.
        eje_x: nombre de la variable horizontal.
        figura: nombre del archivo de salida.
        titulo: titulo del grafico.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURAS.mkdir(parents=True, exist_ok=True)
    etiquetas = sorted(fusiones, key=lambda valor: float(valor))
    posiciones = range(len(etiquetas))
    plt.figure(figsize=(8, 4.6))
    plt.plot(
        posiciones,
        [fusiones[e]["corrected_cv"] for e in etiquetas],
        "o-",
        label="fusion con V03",
        color="#1f77b4",
    )
    plt.plot(
        posiciones,
        [emb_lr[e]["corrected_cv"] for e in etiquetas],
        "s-",
        label="embedding solo (mean+LR)",
        color="#7f7f7f",
    )
    plt.axhline(REFERENCIA_V03, color="black", linestyle="--", label=f"V03 {REFERENCIA_V03:.4f}")
    plt.axhline(REFERENCIA_V03 + MEJORA_MINIMA, color="red", linestyle=":", label="gate +0.005")
    plt.xticks(list(posiciones), [str(e) for e in etiquetas])
    plt.xlabel(eje_x)
    plt.ylabel("corrected_cv (umbrales congelados de V03)")
    plt.title(titulo)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURAS / figura, dpi=150)
    plt.close()


def _informe(
    familia: str,
    titulo: str,
    unidades: int,
    fichas: list[dict[str, Any]],
    resultados: list[dict[str, Any]],
    metricas_rebanada: list[dict[str, Any]],
    mascaras: dict[str, np.ndarray],
    cobertura: dict[str, Any],
    lectura: str,
    decision: str,
    dispersion: float,
    mejor: dict[str, Any],
    criterio: dict[str, Any],
    reporte: Path,
    experimento: str,
    duracion: float,
) -> None:
    """Escribe el informe de la familia.

    Args:
        familia: nombre de la familia.
        titulo: titulo legible.
        unidades: unidades de entrenamiento consumidas.
        fichas: fichas de los puntos de la familia.
        resultados: filas de resultados downstream.
        metricas_rebanada: metricas por rebanada.
        mascaras: mascaras de las rebanadas.
        cobertura: cobertura TOXIC del corpus.
        lectura: lectura esperada de la familia.
        decision: veredicto de la familia.
        dispersion: dispersion interna de las fusiones.
        mejor: mejor fusion de la familia.
        criterio: veredicto del gate de dos niveles.
        reporte: ruta del informe.
        experimento: identificador del experimento.
        duracion: segundos que tardo la familia.
    """
    por_rebanada: dict[str, dict[str, float]] = {}
    for fila in metricas_rebanada:
        por_rebanada.setdefault(fila["rebanada"], {})[fila["variante"]] = fila["f1_por_celda"]
    margen = criterio["margen"]
    lineas = [
        f"# {familia} de v5.1: {titulo} ({experimento})",
        "",
        "Generado por el script de la familia. Todo lo demas fijo: corpus E-CORP-01",
        "(Wikipedia 10M), vector_size 200, min_count 3, subpalabras 3-6, negative 10,",
        "5 epocas, sg skipgram, seed 42, bucket 200.000. Cambia **solo** la variable de la",
        "familia. El punto central es el baseline B0, ya entrenado en v5, que se reevalua",
        "en esta corrida como control y debe reproducir 0.7818.",
        "",
        "## 1. Entrenamientos",
        "",
        "| codigo | vocabulario | s | tipos con vector | sha256 |",
        "|---|---|---|---|---|",
    ]
    for ficha in fichas:
        lineas.append(
            f"| {ficha['codigo']} | {ficha['vocabulary_size']} | {ficha['training_time_s']} | "
            f"{ficha['n_tipos_con_vector']}/{ficha['n_tipos_consultados']} | "
            f"`{ficha['model_sha256'][:12]}` |"
        )
    lineas += [
        "",
        f"Cobertura TOXIC (train+test, como v5): por tipo {cobertura['type_coverage']:.4f}, por",
        f"token {cobertura['token_coverage']:.4f}, word-OOV {cobertura['n_word_oov']}. La cobertura",
        "es del **corpus**, y el corpus no cambia entre familias: lo unico que la familia",
        "puede mover es cuantos de esos tipos reciben vector del modelo.",
        "",
        "## 2. Resultados downstream",
        "",
        "| evaluacion | corrected_cv (frozen) | ajustada | delta vs V03 | std pliegues | IC95 |",
        "|---|---|---|---|---|---|",
    ]
    for fila in resultados:
        ic95 = f"[{fila['ic95_0']:+.4f}, {fila['ic95_1']:+.4f}]" if fila["ic95_0"] != "" else "-"
        lineas.append(
            f"| {fila['evaluacion']} | {fila['corrected_cv']:.4f} | "
            f"{fila['corrected_cv_ajustados']:.4f} | {fila['delta_vs_V03']:+.4f} | "
            f"{fila['std_pliegues']} | {ic95} |"
        )
    lineas += [
        "",
        "## 3. Lectura",
        "",
        lectura,
        "",
        "## 4. Slices de la mejor variante",
        "",
        "| rebanada | filas | V03 | mejor fusion | delta |",
        "|---|---|---|---|---|",
    ]
    for nombre, mascara in mascaras.items():
        if mascara.sum() == 0:
            continue
        valores = por_rebanada.get(nombre, {})
        base = valores.get("V03-CONGELADO", float("nan"))
        actual = valores.get(mejor["evaluacion"], float("nan"))
        lineas.append(
            f"| {nombre} | {int(mascara.sum())} | {base:.4f} | {actual:.4f} | {actual - base:+.4f} |"
        )
    lineas += [
        "",
        "## 5. Decision de familia",
        "",
        "Gate de dos niveles: una variante es **adoptable** si su delta alcanza +0.005 **y** su",
        "intervalo de confianza al 95 % no incluye el 0. La familia se **cierra** si ninguna",
        "variante es adoptable y la dispersion interna es <= 0.005.",
        "",
        f"- Dispersion interna: **{dispersion:.4f}** (umbral 0.005).",
        f"- Mejor delta: **{mejor['delta_vs_V03']:+.4f}**, margen {margen:+.5f} sobre el umbral.",
        f"- IC95 de la mejor: **[{mejor['ic95_0']:+.4f}, {mejor['ic95_1']:+.4f}]**, que "
        f"{'excluye' if criterio['ic95_excluye_cero'] else 'incluye'} el 0.",
        "",
        f"**{decision}**",
        "",
        f"Presupuesto: {unidades} unidades exploratorias de esta familia, acumuladas en",
        "`reports/embedding_budget_v5_1.json`. 0 unidades de reserva.",
        "",
        f"Fecha: {datetime.now(timezone.utc).isoformat(timespec='seconds')}. Duracion: "
        f"{round(duracion, 1)} s.",
    ]
    reporte.write_text("\n".join(lineas) + "\n", encoding="utf-8")


def etiqueta_de_window(ficha: dict[str, Any]) -> str:
    """Extrae la etiqueta del eje window de una ficha.

    Args:
        ficha: ficha de un entrenamiento.

    Returns:
        El valor de la ventana como texto.
    """
    return str(ficha.get("etiqueta", ficha["window"]))


def entrenar_variante(
    propuesta: dict[str, Any],
    modulo_p0,
    firmas_existentes: dict[str, str],
    cobertura: dict[str, Any],
    corpus_meta: dict[str, Any],
    tipos: list[str],
    familia: str,
) -> dict[str, Any]:
    """Entrena una variante de la familia y devuelve su ficha de identidad.

    Es la generalizacion de la funcion homonima de P1, que queda acoplada a la familia
    ``vector_size`` porque solo sabe aplicar un cambio de dimension. Aqui los cambios se
    aplican como un diccionario sobre la arquitectura base, de modo que la misma funcion
    sirve para window, min_count, subwords y negative. P1 no se modifica: es evidencia
    historica de su propia familia.

    Antes de entrenar se comprueba la firma contra las ya conocidas. Si coincide, el modelo
    **se carga** en vez de reentrenarse y no se gasta presupuesto; si coincide y el
    artefacto no esta en disco, es un error, porque reentrenar seria una unidad nueva con
    una firma que ya existe.

    Args:
        propuesta: ``{"codigo", "cambios"}`` de la variante.
        modulo_p0: modulo de P0, con la arquitectura base y la firma canonica.
        firmas_existentes: mapa firma -> training_id.
        cobertura: cobertura TOXIC del corpus.
        corpus_meta: processed_sha256 y corpus_version del corpus.
        tipos: tipos del train a consultar al modelo.
        familia: nombre de la familia, para la ficha.

    Returns:
        Ficha del entrenamiento.
    """
    from src.embedding_utils import (
        entrenar_fasttext,
        sha256_de,
        vectores_de_tipos,
    )

    arquitectura = dict(modulo_p0.ARQUITECTURA_BASE)
    arquitectura.update(propuesta["cambios"])
    firma = modulo_p0.firma_entrenamiento(
        processed_sha256=corpus_meta["processed_sha256"],
        corpus_version=corpus_meta["corpus_version"],
        arquitectura=arquitectura,
    )
    prefijo = PROJECT_ROOT / "data" / "external" / "embeddings" / (
        f"fasttext_wiki_10m_{propuesta['codigo'].lower().replace('-', '_')}"
    )
    segundos = 0.0
    reejecucion = firma in firmas_existentes
    if reejecucion and not Path(f"{prefijo}.vec").exists():
        raise RuntimeError(
            f"{propuesta['codigo']}: la firma ya existe como {firmas_existentes[firma]} pero su "
            "artefacto no esta en disco"
        )
    if not Path(f"{prefijo}.vec").exists():
        inicio = time.time()
        entrenar_fasttext(
            PROJECT_ROOT / "tools" / "fastText" / "build" / "fasttext",
            PROJECT_ROOT / "data" / "external" / "corpora" / "wikipedia_es" / "emb_normal"
            / "tokens_10M.txt",
            prefijo,
            dim=arquitectura["dim"],
            ventana=arquitectura["ventana"],
            min_count=arquitectura["min_count"],
            negativos=arquitectura["negativos"],
            epocas=arquitectura["epocas"],
            hilos=arquitectura["hilos"],
            semilla=arquitectura["semilla"],
            minn=arquitectura["minn"],
            maxn=arquitectura["maxn"],
            bucket=arquitectura["bucket"],
        )
        segundos = time.time() - inicio
    elif reejecucion:
        log(
            f"   {propuesta['codigo']}: EXISTING - LOAD ARTIFACT (firma ya registrada como "
            f"{firmas_existentes[firma]}); no se gasta presupuesto"
        )
    else:
        log(f"   {propuesta['codigo']}: modelo ya presente, se reutiliza")
    carpeta_sifts = PROJECT_ROOT / "data" / "external" / "embeddings"
    ruta_sifts = carpeta_sifts / f"fasttext-wiki-10m-{propuesta['codigo'].lower()}_sifts.txt"
    vectores = vectores_de_tipos(
        PROJECT_ROOT / "tools" / "fastText" / "build" / "fasttext",
        Path(f"{prefijo}.bin"),
        tipos,
        ruta_sifts,
    )
    with Path(f"{prefijo}.vec").open(encoding="utf-8") as archivo:
        vocabulario, dimension = archivo.readline().split()
    log(
        f"   {propuesta['codigo']:8s} {segundos:5.0f} s | vocab {int(vocabulario):>7} | "
        f"dim leida {int(dimension)} | tipos con vector {len(vectores)}/{len(tipos)} | "
        f"{sha256_de(Path(f'{prefijo}.vec'))[:12]}"
    )
    return {
        "codigo": propuesta["codigo"],
        "etiqueta": propuesta.get("etiqueta", propuesta["codigo"]),
        "training_id": f"FASTTEXT-WIKI-10M-{propuesta['codigo']}",
        "familia": familia,
        "vector_size": arquitectura["dim"],
        "window": arquitectura["ventana"],
        "min_count": arquitectura["min_count"],
        "min_n": arquitectura["minn"],
        "max_n": arquitectura["maxn"],
        "negative": arquitectura["negativos"],
        "sample": arquitectura["muestreo"],
        "epochs": arquitectura["epocas"],
        "sg": arquitectura["sg"],
        "seed": arquitectura["semilla"],
        "bucket": arquitectura["bucket"],
        "corpus": "E-CORP-01",
        "tokens": 10_002_400,
        "vocabulary_size": int(vocabulario),
        "dim_leida": int(dimension),
        "n_tipos_consultados": len(tipos),
        "n_tipos_con_vector": len(vectores),
        "training_time_s": round(segundos, 1),
        "model_sha256": sha256_de(Path(f"{prefijo}.vec")),
        "training_signature": firma,
        "status": "EXISTING",
        "rol": f"familia {familia} de v5.1",
        "config_verificada_contra_artefacto": "si",
        "cobertura": cobertura,
        "sifts": str(ruta_sifts.relative_to(PROJECT_ROOT)),
    }


def main() -> None:
    """Ejecuta P2: familia window con ventana 3 y 7."""
    ejecutar_familia(
        familia=FAMILIA,
        titulo=TITULO,
        unidades=UNIDADES,
        eje_x=EJE_X,
        figura=FIGURA,
        reporte=REPORTE,
        reporte_csv=REPORTE_CSV,
        reporte_rebanadas=REPORTE_REBANADAS,
        propuestas=PROPUESTAS,
        punto_existente=PUNTO_EXISTENTE,
        etiqueta_de=etiqueta_de_window,
        lectura=(
            "La ventana controla cuanta distribucion se reune alrededor de cada palabra. Con\n"
            "10M tokens, cada palabra del vocabulario ya aparece decenas de veces, asi que una\n"
            "ventana mayor **no deberia** aportar senal nueva: el limite esta en cuantos datos\n"
            "hay, no en cuanta distribucion se ve. Si la ventana tampoco cambia nada, se acumula\n"
            "evidencia de que el embedding esta limitado por los 1.483 ejemplos etiquetados."
        ),
        experimento=EXPERIMENTO,
    )


if __name__ == "__main__":
    main()
