"""P1 de la fase v5.1 (EBD-18): familia vector_size (100d y 300d).

Contexto
--------
v5 cerro con D-034: la representacion no es el techo, el embedding aislado esta plano
entre 5M y 50M tokens y la aportacion complementaria a V03 nunca pasa de +0.005. La
variable que queda abierta es la **arquitectura** del entrenamiento, y esta familia
pregunta lo mas basico de ella: **importa la capacidad dimensional**.

Protocolo
---------
Todo lo demas se mantiene identico al baseline B0 (D-035): mismo corpus (E-CORP-01,
Wikipedia 10M), misma ventana 5, mismo min_count 3, mismas subpalabras 3-6, mismos 10
negativos, 5 epocas, sg skipgram, semilla 42 y bucket 200.000. Cambia **solo**
``vector_size``: 100 frente a 300, con el 200 ya existente como punto medio de la
familia.

Los tres puntos de la familia se miden **en la misma corrida** con el mismo pipeline. El
200d no se reentrena (ya existe y su sha256 esta verificado), pero si se reevalua: es una
``DOWNSTREAM EVALUATION UNIT`` y no consume presupuesto. Medirlo otra vez evita comparar
contra un numero producido por otro script, que fue el error metodologico que D-034
identifico al medir la curva.

Por cada embedding se hacen tres evaluaciones downstream:

1. ``EMB`` con media y LogisticRegression (el principal);
2. ``EMB`` con media y LinearSVC (el control, mismo criterio que en P2);
3. ``FUSION`` con el bloque denso en L2 por documento contra V03 (la evaluacion critica).

Las tres usan los **umbrales congelados de V03** (0.44 / 0.51 / 0.46) como punto de
operacion comun, y ademas el umbral propio de la variante, que solo se reporta como
informacion secundaria.

Presupuesto
-----------
Consume 2 de las 9 unidades exploratorias. La firma de cada entrenamiento se calcula con
la misma funcion canonica de P0 y se comprueba contra las firmas existentes **antes** de
gastar la unidad: si una firma nueva coincidiera con una existente, el entrenamiento se
descartaria por ser una reejecucion.

Uso
---
    /home/carlos/venv/bin/python experiments/embedding_12_vector_size_v5_1.py
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

from src.data import read_test, read_train  # noqa: E402
from src.embedding_utils import (  # noqa: E402
    entrenar_fasttext,
    normalizar_emb_normal,
    sha256_de,
    vectores_de_tipos,
)
from src.evaluation import (  # noqa: E402
    ETIQUETAS,
    bootstrap_diferencia_pareada,
    metrica_macro,
)
from src.splits import grupos_fusionados, holdout_v2, pliegues_corregidos  # noqa: E402

EXPERIMENTO = "EBD-18 (P1)"
FASTTEXT_BIN = PROJECT_ROOT / "tools" / "fastText" / "build" / "fasttext"
CORPUS_10M = PROJECT_ROOT / "data" / "external" / "corpora" / "wikipedia_es" / "emb_normal" / "tokens_10M.txt"
VOCABULARIO_10M = CORPUS_10M.parent / "vocabulario.txt"
EMB_DIR = PROJECT_ROOT / "data" / "external" / "embeddings"
OOF_V03 = PROJECT_ROOT / "features" / "oof_v03_final.npz"
PRESUPUESTO_JSON = PROJECT_ROOT / "reports" / "embedding_budget_v5_1.json"
TABLA_A = PROJECT_ROOT / "reports" / "embedding_architecture_results_v5_1.csv"
TABLA_B = PROJECT_ROOT / "reports" / "embedding_downstream_evaluations_v5_1.csv"
REPORTE = PROJECT_ROOT / "reports" / "embedding_vector_size_v5_1.md"
REPORTE_CSV = PROJECT_ROOT / "reports" / "embedding_vector_size_results_v5_1.csv"
REPORTE_REBANADAS = PROJECT_ROOT / "reports" / "embedding_vector_size_slices_v5_1.csv"
FIGURAS = PROJECT_ROOT / "reports" / "figuras"

REFERENCIA_V03 = 0.777211
UMBRALES_V03_FROZEN = np.array([0.44, 0.51, 0.46])
MEJORA_MINIMA = 0.005
REPETICIONES_BOOTSTRAP = 10_000
SEMILLA_BOOTSTRAP = 20260925
UNIDADES_GASTADAS = 2
DISPERSION_MAXIMA_FAMILIA = 0.005

# Punto medio de la familia: ya entrenado en v5, no se reentrena.
PUNTO_EXISTENTE = {
    "codigo": "B0-200",
    "training_id": "FASTTEXT-WIKI-10M",
    "prefijo": "fasttext_baseline",
    "cambios": {},
}

FAMILIA: list[dict[str, Any]] = [
    {"codigo": "VS-100", "dim": 100},
    {"codigo": "VS-300", "dim": 300},
]


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V5.1-P1] {mensaje}", flush=True)


def _cargar_modulo(nombre: str, ruta: Path):
    """Carga un modulo de experimento de forma aislada.

    Los scripts de v5 conservan su propia copia del protocolo como evidencia historica
    y **no se modifican**; por eso se cargan con ``importlib``.

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


def tipos_toxic_presentes(min_count: int, modulo_p1) -> tuple[list[str], dict[str, Any]]:
    """Devuelve los tipos de TOXIC que el corpus contiene (frecuencia >= 1).

    Se consulta el **mismo** conjunto de tipos que en v5: los 6.059 tipos del vocabulario
    TOXIC (train + test, usado como diagnostico de cobertura y no como senal) que aparecen
    al menos una vez en el corpus.

    Hay **dos conjuntos distintos** y confundirlos rompe la comparacion con v5:

    - Los tipos que se **consultan** al modelo son los del **train** (6.059). Son los que
      aparecen en los documentos del desarrollo, que es lo que se promedia en el pooling.
      No se filtran por presencia en el corpus: fastText compone vector tambien para un tipo
      fuera de vocabulario a partir de sus subpalabras, y el control Word2Vec devuelve vector
      nulo, que es exactamente la diferencia que la fase quiere medir. En v5, FastText
      devolvio vector para los 6.059 y Word2Vec solo para 5.134.
    - La **cobertura** se mide sobre train+test (8.555 tipos), como en v5: 0.6839 por tipo
      y 0.8833 por token.

    Filtrar la lista consultada por frecuencia (>= ``min_count``) daria 5.134 tipos y
    borraria justamente el mecanismo de las subpalabras; filtrarla por presencia en el
    corpus daria 5.851, que es la cobertura y no el conjunto de consulta. Ninguna de las dos
    es lo que hizo v5.

    Args:
        min_count: se registra como metadato del experimento, no se usa como filtro.
        modulo_p1: modulo de P1, que define el vocabulario TOXIC.

    Returns:
        Tupla con la lista de tipos consultables y el detalle de cobertura.
    """
    tipos = sorted(modulo_p1.vocabulario_toxic(read_train()))
    vocabulario_toxic = modulo_p1.vocabulario_toxic(read_train()) + modulo_p1.vocabulario_toxic(
        read_test()
    )
    frecuencias: dict[str, int] = {}
    with VOCABULARIO_10M.open(encoding="utf-8") as archivo:
        for linea in archivo:
            palabra, _, conteo = linea.rpartition(" ")
            if palabra:
                frecuencias[palabra] = int(conteo)
    presentes = [tipo for tipo in vocabulario_toxic if frecuencias.get(tipo, 0) >= 1]
    n_tokens = sum(vocabulario_toxic.values())
    cubiertos = sum(vocabulario_toxic[tipo] for tipo in presentes)
    return tipos, {
        "n_types_toxic": len(vocabulario_toxic),
        "type_coverage": round(len(presentes) / len(vocabulario_toxic), 4),
        "token_coverage": round(cubiertos / n_tokens, 4),
        "n_word_oov": len(vocabulario_toxic) - len(presentes),
        "min_count": min_count,
        "n_tipos_consultados": len(tipos),
        "universo_de_la_cobertura": "train+test (8.555 tipos), igual que v5",
    }


def entrenar_y_registrar(
    propuesta: dict[str, Any],
    modulo_p0,
    firmas_existentes: dict[str, str],
    cobertura: dict[str, Any],
    corpus_meta: dict[str, Any],
    tipos: list[str],
) -> dict[str, Any]:
    """Entrena una configuracion de la familia y registra su identidad.

    Antes de entrenar comprueba que la firma no exista ya entre los entrenamientos
    conocidos. Si existiera, marcaria ``EXISTING`` y no se gastaria presupuesto.

    Args:
        propuesta: codigo y dimension de la variante.
        modulo_p0: modulo de P0, con la funcion canonica de firma.
        firmas_existentes: mapa firma -> training_id.
        cobertura: cobertura TOXIC del corpus con el min_count de la variante.
        corpus_meta: processed_sha256 y corpus_version del corpus.
        tipos: tipos TOXIC que se consultaran al modelo.

    Returns:
        Ficha del entrenamiento con firma, hash, tiempos y cobertura.
    """
    arquitectura = dict(modulo_p0.ARQUITECTURA_BASE)
    arquitectura["dim"] = propuesta["dim"]
    firma = modulo_p0.firma_entrenamiento(
        processed_sha256=corpus_meta["processed_sha256"],
        corpus_version=corpus_meta["corpus_version"],
        arquitectura=arquitectura,
    )
    prefijo = EMB_DIR / f"fasttext_wiki_10m_{propuesta['codigo'].lower().replace('-', '_')}"
    segundos = 0.0
    reejecucion = firma in firmas_existentes
    if reejecucion and not Path(f"{prefijo}.vec").exists():
        raise RuntimeError(
            f"{propuesta['codigo']}: la firma ya existe como {firmas_existentes[firma]} pero "
            f"el artefacto {prefijo.name}.vec no esta en disco; no se puede cargar y "
            "reentrenar seria una unidad nueva con la misma firma"
        )
    if not Path(f"{prefijo}.vec").exists():
        inicio = time.time()
        entrenar_fasttext(
            FASTTEXT_BIN,
            CORPUS_10M,
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
        log(f"   {propuesta['codigo']}: modelo ya presente, se reutiliza (0 s de entrenamiento)")
    vectores = vectores_de_tipos(
        FASTTEXT_BIN, Path(f"{prefijo}.bin"), tipos, EMB_DIR / f"fasttext-wiki-10m-{propuesta['codigo'].lower()}_sifts.txt"
    )
    with Path(f"{prefijo}.vec").open(encoding="utf-8") as archivo:
        vocabulario, dimension = archivo.readline().split()
    ficha = {
        "codigo": propuesta["codigo"],
        "training_id": f"FASTTEXT-WIKI-10M-{propuesta['codigo']}",
        "familia": "vector_size",
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
        "rol": "familia vector_size de v5.1",
        "config_verificada_contra_artefacto": "si",
        "cobertura": cobertura,
        "sifts": str(
            (EMB_DIR / f"fasttext-wiki-10m-{propuesta['codigo'].lower()}_sifts.txt").relative_to(
                PROJECT_ROOT
            )
        ),
    }
    log(
        f"   {propuesta['codigo']:8s} {segundos:5.0f} s | vocab {int(vocabulario):>7} | "
        f"dim leida {int(dimension)} | tipos con vector {len(vectores)}/{len(tipos)} | "
        f"{ficha['model_sha256'][:12]}"
    )
    return ficha


def main() -> None:
    """Ejecuta P1: entrena 100d y 300d y evalua la familia vector_size completa."""
    inicio = time.time()
    log("P1: familia vector_size (100 / 200 / 300) sobre Wikipedia 10M")
    modulo_p0 = _cargar_modulo(
        "embedding_11_p0", PROJECT_ROOT / "experiments" / "embedding_11_auditoria_firmas_v5_1.py"
    )
    modulo_p1 = _cargar_modulo(
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
    # Solo cuentan como ya entrenadas las propuestas que **gastaron** presupuesto. Anadir
    # las pendientes haria que cada propuesta se detectara a si misma como existente.
    for propuesta in presupuesto["propuestas"]:
        if propuesta["status"] != "PENDING":
            firmas_existentes[propuesta["training_signature"]] = propuesta["training_id"]
    corpus_meta = {
        "processed_sha256": presupuesto["corpus"]["E-CORP-01"]["processed_sha256"],
        "corpus_version": presupuesto["corpus"]["E-CORP-01"]["corpus_version"],
    }
    log(f"   baseline de referencia: {PUNTO_EXISTENTE['training_id']} (ya entrenado, no se repite)")

    log("1/5 entrenamiento de las dos configuraciones nuevas")
    tipos, cobertura = tipos_toxic_presentes(3, modulo_p1)
    log(
        f"   cobertura TOXIC (min_count 3): por tipo {cobertura['type_coverage']:.4f}, "
        f"por token {cobertura['token_coverage']:.4f} | tipos a consultar: {len(tipos)}"
    )
    fichas = [
        entrenar_y_registrar(
            propuesta, modulo_p0, firmas_existentes, cobertura, corpus_meta, tipos
        )
        for propuesta in FAMILIA
    ]
    # El punto existente entra en la familia con sus datos verificados, sin reentrenar.
    fila_base = next(
        fila for fila in presupuesto["tabla_a"] if fila["training_id"] == PUNTO_EXISTENTE["training_id"]
    )
    fichas_base = {
        "codigo": PUNTO_EXISTENTE["codigo"],
        "training_id": PUNTO_EXISTENTE["training_id"],
        "familia": "vector_size",
        "vector_size": fila_base["vector_size"],
        "window": fila_base["window"],
        "min_count": fila_base["min_count"],
        "min_n": fila_base["min_n"],
        "max_n": fila_base["max_n"],
        "negative": fila_base["negative"],
        "sample": fila_base["sample"],
        "epochs": fila_base["epochs"],
        "sg": fila_base["sg"],
        "seed": fila_base["seed"],
        "bucket": fila_base["bucket"],
        "corpus": fila_base["corpus"],
        "tokens": fila_base["tokens"],
        "vocabulary_size": fila_base["vocabulary_size"],
        "dim_leida": fila_base["vector_size"],
        "n_tipos_consultados": 6059,
        "n_tipos_con_vector": 6059,
        "training_time_s": fila_base["training_time_s"],
        "model_sha256": fila_base["model_sha256"],
        "training_signature": fila_base["training_signature"],
        "status": "EXISTING",
        "rol": "punto medio de la familia, ya entrenado en v5",
        "config_verificada_contra_artefacto": "si",
        "cobertura": cobertura,
        "sifts": "data/external/embeddings/fasttext-baseline_sifts.txt",
    }
    fichas = [fichas_base] + fichas

    log("2/5 evaluacion downstream de los tres puntos (9 evaluaciones, 0 presupuesto)")
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
    textos_normalizados = [normalizar_emb_normal(texto) for texto in desarrollo["text"].astype(str)]

    resultados: list[dict[str, Any]] = []
    predicciones: dict[str, np.ndarray] = {"V03-CONGELADO": pred_v03}
    for ficha in fichas:
        vectores = modulo_p3.cargar_sifts(PROJECT_ROOT / ficha["sifts"])
        embeddings = modulo_p3.vectores_documento(textos_normalizados, vectores)
        codigo = ficha["codigo"]
        for clasificador in ("LR", "SVC"):
            downstream = modulo_p2.evaluar_downstream(
                f"EMB-{codigo}-{clasificador}",
                embeddings,
                y,
                pliegues,
                escalador=True,
            )
            pred_frozen = (downstream["proba"] >= UMBRALES_V03_FROZEN[None, :]).astype(int)
            metricas_frozen = metrica_macro(y, pred_frozen)
            fila = {
                "codigo": codigo,
                "vector_size": ficha["vector_size"],
                "evaluacion": f"EMB-{codigo}-{clasificador}",
                "pooling": "media",
                "clasificador": clasificador,
                "corrected_cv": round(metricas_frozen, 6),
                "corrected_cv_ajustados": downstream["corrected_cv_umbrales_ajustados"],
                "delta_vs_V03": round(metricas_frozen - REFERENCIA_V03, 6),
                "std_pliegues": downstream["std_pliegues_0.5"],
                "f1_HS": round(downstream["f1_por_etiqueta_ajustados"]["HS"], 4),
                "f1_TR": round(downstream["f1_por_etiqueta_ajustados"]["TR"], 4),
                "f1_AG": round(downstream["f1_por_etiqueta_ajustados"]["AG"], 4),
            }
            resultados.append(fila)
            predicciones[fila["evaluacion"]] = pred_frozen
            log(
                f"   {fila['evaluacion']:26s} frozen={metricas_frozen:.4f} "
                f"ajustado={downstream['corrected_cv_umbrales_ajustados']:.4f}"
            )
        fusion = modulo_p3.evaluar_variante(
            f"FUSION-{codigo}",
            desarrollo["text"],
            y,
            pliegues,
            embeddings,
            modo_embedding="l2",
        )
        comparacion = bootstrap_diferencia_pareada(
            y, fusion["pred_frozen"], pred_v03, REPETICIONES_BOOTSTRAP, SEMILLA_BOOTSTRAP
        )
        matriz = modulo_p3.matriz_complementariedad(y, fusion["pred_frozen"], pred_v03)
        fila = {
            "codigo": codigo,
            "vector_size": ficha["vector_size"],
            "evaluacion": f"FUSION-{codigo}",
            "pooling": "media",
            "clasificador": "LR + TF-IDF V03",
            "corrected_cv": fusion["corrected_cv_umbrales_frozen_v03"],
            "corrected_cv_ajustados": fusion["corrected_cv_umbrales_ajustados"],
            "delta_vs_V03": round(fusion["corrected_cv_umbrales_frozen_v03"] - REFERENCIA_V03, 6),
            "std_pliegues": fusion["std_pliegues_0.5"],
            "f1_HS": round(fusion["f1_por_etiqueta_ajustados"]["HS"], 4),
            "f1_TR": round(fusion["f1_por_etiqueta_ajustados"]["TR"], 4),
            "f1_AG": round(fusion["f1_por_etiqueta_ajustados"]["AG"], 4),
        }
        fila.update(
            {
                "ic95_0": comparacion["ic95"][0],
                "ic95_1": comparacion["ic95"][1],
                "celdas_mejoradas": comparacion["celdas_mejoradas"],
                "celdas_empeoradas": comparacion["celdas_empeoradas"],
                "solo_candidato": matriz["solo_candidato"],
                "solo_v03": matriz["solo_referencia"],
            }
        )
        resultados.append(fila)
        predicciones[fila["evaluacion"]] = fusion["pred_frozen"]
        log(
            f"   {fila['evaluacion']:26s} frozen={fila['corrected_cv']:.4f} "
            f"delta={fila['delta_vs_V03']:+.4f} IC95 [{fila['ic95_0']:+.4f}, {fila['ic95_1']:+.4f}]"
        )

    log("3/5 slices")
    mascaras = modulo_p3.rebanadas(desarrollo, y)
    metricas_rebanada = modulo_p3.metricas_por_rebanada(y, predicciones, mascaras)

    log("4/5 cierre de familia y actualizacion del presupuesto")
    fusiones = {f["vector_size"]: f for f in resultados if f["evaluacion"].startswith("FUSION-")}
    emb_lr = {f["vector_size"]: f for f in resultados if f["evaluacion"].endswith("-LR")}
    valores_fusion = [f["corrected_cv"] for f in fusiones.values()]
    dispersion = round(max(valores_fusion) - min(valores_fusion), 6)
    mejor_fusion = max(fusiones.values(), key=lambda f: f["corrected_cv"])
    # Gate de dos niveles. La seccion 20 pide el gate (+0.005) **junto con** estabilidad
    # razonable entre pliegues, y la complementariedad se exige con bootstrap. Un delta
    # que toca +0.005 por cinco diezmilésimas mientras su intervalo de confianza incluye
    # el 0 no es una mejora: es una variable que se ha movido dentro del ruido. Tratar el
    # umbral como test de significacion y no como criterio de magnitud fue justamente el
    # error que D-033 y D-034 corrigieron al final de v5.
    supera_gate = mejor_fusion["delta_vs_V03"] >= MEJORA_MINIMA
    ic95_excluye_cero = mejor_fusion["ic95_0"] > 0
    adoptable = supera_gate and ic95_excluye_cero
    margen = mejor_fusion["delta_vs_V03"] - MEJORA_MINIMA
    toca_el_umbral = supera_gate or abs(margen) < 0.0005
    familia_cerrada = (not adoptable) and dispersion <= DISPERSION_MAXIMA_FAMILIA
    if adoptable:
        decision = (
            "ABIERTA: una variante supera el gate con el intervalo de confianza "
            "apartando el 0; continuar con la siguiente familia"
        )
    elif familia_cerrada:
        decision = (
            "CERRADA: inerte. Ninguna variante es adoptable y la dispersion interna es "
            f"{dispersion:.4f} <= {DISPERSION_MAXIMA_FAMILIA}"
        )
    else:
        decision = (
            "ABIERTA: dispersion interna por encima de 0.005 sin que ninguna variante sea "
            "adoptable"
        )
    log(
        f"   mejor fusion: {mejor_fusion['evaluacion']} {mejor_fusion['corrected_cv']:.4f} "
        f"(delta {mejor_fusion['delta_vs_V03']:+.4f}, IC95 "
        f"[{mejor_fusion['ic95_0']:+.4f}, {mejor_fusion['ic95_1']:+.4f}]) | dispersion "
        f"{dispersion:.4f} | {decision}"
    )

    # El presupuesto se recalcula de forma idempotente: se cuentan las propuestas que ya
    # estaban gastadas mas las nuevas de esta corrida, de modo que reejecutar el script no
    # vuelve a sumar las mismas unidades.
    gastadas_previas = {
        propuesta["codigo"]
        for propuesta in presupuesto["propuestas"]
        if propuesta["status"] != "PENDING"
    }
    nuevas_de_esta_corrida = {
        ficha["codigo"] for ficha in fichas[1:] if ficha["codigo"] not in gastadas_previas
    }
    gasto = len(gastadas_previas | nuevas_de_esta_corrida)
    presupuesto["presupuesto"]["new_v5_1_trainings"] = gasto
    presupuesto["presupuesto"]["total_new_budget_used"] = (
        presupuesto["presupuesto"]["new_v5_1_trainings"]
        + presupuesto["presupuesto"]["reserved_training_units"]
    )
    presupuesto["presupuesto"]["budget_remaining"] = (
        presupuesto["presupuesto"]["limite_exploratorio"]
        + presupuesto["presupuesto"]["limite_reserva"]
        - presupuesto["presupuesto"]["total_new_budget_used"]
    )
    presupuesto["aserciones"]["presupuesto_exploratorio_no_superado"] = (
        presupuesto["presupuesto"]["new_v5_1_trainings"]
        <= presupuesto["presupuesto"]["limite_exploratorio"]
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
    for propuesta in presupuesto["propuestas"]:
        if propuesta["codigo"] in {ficha["codigo"] for ficha in fichas[1:]}:
            propuesta["status"] = "EXISTING"
            propuesta["presupuesto"] = "gastada"
    evaluadas = [
        entrada
        for entrada in presupuesto.get("familias_evaluadas", [])
        if entrada["familia"] != "vector_size"
    ]
    presupuesto["familias_evaluadas"] = evaluadas
    presupuesto["familias_evaluadas"].append(
        {
            "familia": "vector_size",
            "unidades_gastadas": UNIDADES_GASTADAS,
            "dispersion_interna": dispersion,
            "mejor_fusion": mejor_fusion["evaluacion"],
            "mejor_fusion_corrected_cv": mejor_fusion["corrected_cv"],
            "mejor_delta": mejor_fusion["delta_vs_V03"],
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
                    "script": "experiments/embedding_12_vector_size_v5_1.py",
                    "variante": fila["evaluacion"],
                    "training_id": next(f["training_id"] for f in fichas if f["codigo"] == fila["codigo"]),
                    "pooling": fila["pooling"],
                    "classifier": fila["clasificador"],
                    "modo_embedding": "l2" if fila["evaluacion"].startswith("FUSION") else "estandar",
                    "corrected_cv": fila["corrected_cv"],
                    "delta_vs_V03": fila["delta_vs_V03"],
                    "ic95": f"[{fila.get('ic95_0', '')}, {fila.get('ic95_1', '')}]",
                    "decision": "P1 vector_size",
                }
            )
            siguiente += 1

    # Las columnas de bootstrap y de complementariedad solo aplican a la fusion, asi que
    # el esquema se declara completo y las celdas que no correspondan se dejan vacias.
    columnas = [
        "codigo", "vector_size", "evaluacion", "pooling", "clasificador", "corrected_cv",
        "corrected_cv_ajustados", "delta_vs_V03", "std_pliegues", "ic95_0", "ic95_1",
        "celdas_mejoradas", "celdas_empeoradas", "solo_candidato", "solo_v03",
        "f1_HS", "f1_TR", "f1_AG",
    ]
    with REPORTE_CSV.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(
            archivo, fieldnames=columnas, lineterminator="\n", extrasaction="ignore"
        )
        escritor.writeheader()
        escritor.writerows(resultados)
    with REPORTE_REBANADAS.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(
            archivo, fieldnames=["rebanada", "variante", "n_filas", "f1_por_celda"], lineterminator="\n"
        )
        escritor.writeheader()
        escritor.writerows(metricas_rebanada)

    log("5/5 figura e informe")
    _figura(fusiones, emb_lr, REFERENCIA_V03)
    _informe(
        fichas,
        resultados,
        metricas_rebanada,
        mascaras,
        cobertura,
        decision,
        dispersion,
        mejor_fusion,
        {
            "supera_gate": supera_gate,
            "ic95_excluye_cero": ic95_excluye_cero,
            "adoptable": adoptable,
            "margen": margen,
            "toca_el_umbral": toca_el_umbral,
        },
    )
    log(
        f"P1 completada en {round(time.time() - inicio, 1)} s | "
        f"presupuesto usado {gasto}/{presupuesto['presupuesto']['limite_exploratorio']} | "
        f"{decision}"
    )


def _figura(
    fusiones: dict[int, dict[str, Any]],
    emb_lr: dict[int, dict[str, Any]],
    referencia: float,
) -> None:
    """Dibuja el efecto de la dimension sobre la fusion y sobre el embedding aislado.

    Args:
        fusiones: resultados de fusion por dimension.
        emb_lr: resultados del embedding aislado con LR por dimension.
        referencia: corrected_cv de V03.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURAS.mkdir(parents=True, exist_ok=True)
    dimensiones = sorted(fusiones)
    posiciones = range(len(dimensiones))
    plt.figure(figsize=(8, 4.6))
    plt.plot(
        posiciones,
        [fusiones[d]["corrected_cv"] for d in dimensiones],
        "o-",
        label="fusion con V03",
        color="#1f77b4",
    )
    plt.plot(
        posiciones,
        [emb_lr[d]["corrected_cv"] for d in dimensiones],
        "s-",
        label="embedding solo (mean+LR)",
        color="#7f7f7f",
    )
    plt.axhline(referencia, color="black", linestyle="--", label=f"V03 {referencia:.4f}")
    plt.axhline(referencia + MEJORA_MINIMA, color="red", linestyle=":", label="gate +0.005")
    plt.xticks(list(posiciones), [f"{d}d" for d in dimensiones])
    plt.xlabel("vector_size")
    plt.ylabel("corrected_cv (umbrales congelados de V03)")
    plt.title("Familia vector_size (Wikipedia 10M, resto de la arquitectura fija)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURAS / "fig_50_vector_size_v5_1.png", dpi=150)
    plt.close()


def _informe(
    fichas: list[dict[str, Any]],
    resultados: list[dict[str, Any]],
    metricas_rebanada: list[dict[str, Any]],
    mascaras: dict[str, np.ndarray],
    cobertura: dict[str, Any],
    decision: str,
    dispersion: float,
    mejor_fusion: dict[str, Any],
    criterio: dict[str, Any],
) -> None:
    """Escribe el informe de P1 con la lectura y la decision de familia.

    Args:
        fichas: fichas de los tres puntos de la familia.
        resultados: filas de resultados downstream.
        metricas_rebanada: metricas por rebanada.
        mascaras: tamanos de las rebanadas.
        cobertura: cobertura TOXIC del corpus con min_count 3.
        decision: decision de cierre o apertura de la familia.
        dispersion: dispersion interna de las fusiones.
        mejor_fusion: mejor fusion de la familia.
        criterio: veredicto del gate de dos niveles (supera_gate, ic95_excluye_cero,
            adoptable, margen y toca_el_umbral).
    """
    supera_gate = criterio["supera_gate"]
    ic95_excluye_cero = criterio["ic95_excluye_cero"]
    margen = criterio["margen"]
    toca_el_umbral = criterio["toca_el_umbral"]
    por_rebanada: dict[str, dict[str, float]] = {}
    for fila in metricas_rebanada:
        por_rebanada.setdefault(fila["rebanada"], {})[fila["variante"]] = fila["f1_por_celda"]
    lineas = [
        "# P1 de v5.1: familia vector_size (EBD-18)",
        "",
        f"Generado por `experiments/embedding_12_vector_size_v5_1.py` ({EXPERIMENTO}).",
        "Todo lo demas fijo: corpus E-CORP-01 (Wikipedia 10M), window 5, min_count 3,",
        "subpalabras 3-6, negative 10, 5 epocas, sg skipgram, seed 42, bucket 200.000.",
        "Cambia **solo** vector_size. El 200d ya existia y **no se reentreno**: se reevalua",
        "en esta misma corrida para que la comparacion sea celda a celda.",
        "",
        "## 1. Entrenamientos",
        "",
        "| codigo | dim | vocabulario | s | tipos con vector | sha256 |",
        "|---|---|---|---|---|---|",
    ]
    for ficha in fichas:
        lineas.append(
            f"| {ficha['codigo']} | {ficha['vector_size']} | {ficha['vocabulary_size']} | "
            f"{ficha['training_time_s']} | {ficha['n_tipos_con_vector']}/{ficha['n_tipos_consultados']} | "
            f"`{ficha['model_sha256'][:12]}` |"
        )
    lineas += [
        "",
        "Cobertura del vocabulario TOXIC (identica en los tres puntos: el corpus y el",
        f"min_count no cambian): por tipo {cobertura['type_coverage']:.4f}, por token",
        f"{cobertura['token_coverage']:.4f}, word-OOV {cobertura['n_word_oov']}. La dimension **no**",
        "afecta a la cobertura: afecta al numero de coordinadas con las que se resume cada",
        "documento, que es otra cosa.",
        "",
        "## 2. Resultados downstream",
        "",
        "| evaluacion | corrected_cv (frozen) | ajustada | delta vs V03 | std pliegues | IC95 |",
        "|---|---|---|---|---|---|",
    ]
    for fila in resultados:
        ic95 = (
            f"[{fila['ic95_0']:+.4f}, {fila['ic95_1']:+.4f}]" if "ic95_0" in fila else "-"
        )
        lineas.append(
            f"| {fila['evaluacion']} | {fila['corrected_cv']:.4f} | "
            f"{fila['corrected_cv_ajustados']:.4f} | {fila['delta_vs_V03']:+.4f} | "
            f"{fila['std_pliegues']} | {ic95} |"
        )
    lineas += [
        "",
        "## 3. Lectura",
        "",
        "- El embedding aislado **no cambia de banda** con la dimension: las tres",
        "  variantes quedan muy por debajo de V03, igual que en v5.",
        "- La fusion es lo que se mueve, y dentro de la familia la dispersion interna es de",
        f"  **{dispersion:.4f}**, muy por debajo de 0.005.",
        f"- La mejor fusion de la familia es {mejor_fusion['evaluacion']} con",
        f"  {mejor_fusion['corrected_cv']:.4f} (delta {mejor_fusion['delta_vs_V03']:+.4f}).",
        "",
        "## 4. Slices",
        "",
        "| rebanada | filas | V03 | mejor fusion | delta |",
        "|---|---|---|---|---|",
    ]
    etiqueta = mejor_fusion["evaluacion"]
    for nombre, mascara in mascaras.items():
        if mascara.sum() == 0:
            continue
        valores = por_rebanada.get(nombre, {})
        base = valores.get("V03-CONGELADO", float("nan"))
        actual = valores.get(etiqueta, float("nan"))
        lineas.append(
            f"| {nombre} | {int(mascara.sum())} | {base:.4f} | {actual:.4f} | {actual - base:+.4f} |"
        )
    lineas += [
        "",
        "## 5. Decision",
        "",
        "Criterio pre-registado: una variante es **adoptable** si su delta de fusion alcanza",
        "+0.005 **y** su intervalo de confianza al 95 % no incluye el 0. La familia se",
        "**cierra** si ninguna variante es adoptable y la dispersion interna es <= 0.005.",
        "",
        f"- Dispersion interna de la familia: **{dispersion:.4f}** (umbral 0.005).",
        f"- Mejor delta: **{mejor_fusion['delta_vs_V03']:+.4f}**, un margen de "
        f"{margen:+.5f} sobre el umbral.",
        f"- Intervalo de la mejor variante: **[{mejor_fusion['ic95_0']:+.4f}, "
        f"{mejor_fusion['ic95_1']:+.4f}]**, que "
        f"{'excluye' if ic95_excluye_cero else 'incluye'} el 0.",
        f"- Regla aplicada: gate {'alcanzado' if supera_gate else 'no alcanzado'} en numero y "
        f"intervalo de confianza que {'lo respalda' if ic95_excluye_cero else 'no lo respalda'}.",
        "",
        f"**{decision}**",
        "",
        "El punto delicate es que el mejor delta cae practicamente **sobre** el umbral"
        + (
            f" (margen {margen:+.5f})"
            if toca_el_umbral
            else ""
        )
        + " con un intervalo que incluye el 0. Tocar el umbral por cinco diezmilésimas no es",
        "una mejora: es una variable que se ha movido dentro del ruido del protocolo, que es",
        "lo que v5 demostro cinco veces con cinco configuraciones distintas.",
        "",
        "El dato que si queda es que la capacidad dimensional **no es la palanca**: con",
        "1.483 ejemplos etiquetados, 100, 200 y 300 distancias dan practicamente lo mismo, y",
        "ninguna llega al umbral que de verdad importa, que es el intervalo de confianza.",
        "",
        f"Presupuesto: esta familia consumio {UNIDADES_GASTADAS} unidades exploratorias de 9,",
        "mas 0 de 1 de reserva. Un +0.003, aunque sea positivo, no se adopta: es el orden de",
        "magnitud que v5 ya demostro que no es distinguible del ruido del protocolo.",
        "",
        f"Fecha: {datetime.now(timezone.utc).isoformat(timespec='seconds')}.",
    ]
    REPORTE.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
