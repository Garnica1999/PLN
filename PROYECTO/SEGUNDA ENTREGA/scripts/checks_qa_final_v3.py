"""checks_qa_final_v3.py
Fecha: 2026-10-04
Descripcion: Verificacion ejecutada del QA v2.2 del producto Juris-PLN
    (ciclo v2.2, plan/PLAN_v2.2_adopcion.md; versionado ADR-16). Valida
    los 10 JSON de resultados/qa_final_v2_2/ contra los criterios de
    aceptacion v2.2: es_sentencia correcto, area top-3 con
    probabilidades, articulos extraidos > 0, retrieval top-10 (k=10, sin
    transitorios ni preambulo, scores finitos y ordenados), prior de
    hubs aplicado con alpha correcto por modo (0.75 default / 1.5
    rerank), flag reranker_usado coherente con el modo del caso, citas
    resueltas coherentes (ano/pagerank/area), tiempos y coherencia del
    resumen con los campos del JSON.
    Verifica ademas la integracion v2.2: md5 de los artefactos v2
    congelados (m4_*, v8, prior de hubs) y de los componentes nuevos del
    Sprint B en modelos/ (B2 retriever denso y B1 cross-encoder),
    config de VinculoCP (top-50, 3 chunks, pesos 0.5/0.5, alphas 0.75/
    1.5, flags B2/B1 y carga lazy), caches E5 (fallback) y B2, fallback
    E5/bge documentado, metricas v2 sin drift y contraste condicional
    contra el benchmark v13 (Sprint B2). Emite tabla compacta y exit !=
    0 si cualquier check critico falla.

    v3 (QA v2.2) - cambios frente al QA v2 (checks_qa_final.py, que se
    conserva INTACTO como evidencia historica):
    - Directorio por defecto resultados/qa_final_v2_2/ (JSON + logs +
      entradas regenerados con el producto v2.2).
    - md5 nuevos de modelos/b2_retriever y modelos/b1_reranker
      (ARTEFACTOS_MD5_V22) ademas de los md5 congelados de v2.
    - Strings de logs v2.2: carga "denso B2 (modelos/b2_retriever)" y
      "reranker .../modelos/b1_reranker" en los 2 casos rerank (critico,
      es la unica evidencia de que B2/B1 se usaron: el esquema JSON no
      cambio); ausencia de carga de modelos en los casos default y
      prohibicion de la ruta legacy E5/bge y del fallback.
    - Resumen del modo rerank nuevo: "recuperados por hibrido
      BM25+B2+reranker (B1)" (el texto default queda intacto).
    - Expectativas de ranking de los 10 casos E2E (TOP10_ESPERADO)
      regeneradas ejecutando el producto v2.2 el 2026-10-04: default
      para 8 casos y rerank para pos_6251 y pos_20178 (los 2 que ya
      usaban rerank en el QA v2). Los 8 JSON default/negativos son
      identicos al QA v2 salvo tiempo_segundos (ruta default intacta);
      los 2 rerank cambian por diseno (B2/B1 y ranking). Si el id de un
      caso E2E esta en la muestra del benchmark v13 se contrasta su
      top-10 contra la estrategia B2+B1 del benchmark; en esta version
      la interseccion es vacia (se reporta como observacion).
    - Control negativo contra los JSON v2 (resultados/qa_final_v2) y v1
      (resultados/qa_final): el script debe FALLAR (exit 1) con ellos,
      porque el resumen rerank, el ranking y los logs no coinciden.
    El script v2 se conserva como scripts/checks_qa_final.py y el v1
    como scripts/checks_qa_final_v1.py (evidencia historica).

Regeneracion de expectativas (una vez por version de producto):
    /home/carlos/venv/bin/python scripts/10_evaluar_sentencia.py \
        --id <ID> --salida resultados/qa_final_v2_2/pos_<ID>.json
    /home/carlos/venv/bin/python scripts/10_evaluar_sentencia.py \
        --id 6251 --rerank --salida resultados/qa_final_v2_2/pos_6251.json
    (stdout+stderr redirigidos a resultados/qa_final_v2_2/logs/<archivo>.log)

Uso:
    /home/carlos/venv/bin/python scripts/checks_qa_final_v3.py
    /home/carlos/venv/bin/python scripts/checks_qa_final_v3.py \
        --qa-dir resultados/qa_final_v2   # control negativo v2
    /home/carlos/venv/bin/python scripts/checks_qa_final_v3.py \
        --qa-dir resultados/qa_final      # control negativo v1
"""

import argparse
import hashlib
import inspect
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QA_V22 = ROOT / "resultados" / "qa_final_v2_2"

# Seccion 4 de lib/componentes_producto.py; se importa para verificar los
# defaults ejecutables del producto v2.2 (no instancia modelos por si sola).
sys.path.insert(0, str(ROOT / "scripts"))
from lib.componentes_producto import CONFIG, VinculoCP

K_RETRIEVAL = 10
ALFA_POR_MODO = {"default": 0.75, "rerank": 1.5}
TIEMPO_MAX = {"default": 30.0, "rerank": 60.0}

# Casos esperados: (archivo, id, tipo, ano, es_sentencia, modo). El modo
# "rerank" usa la ruta v2.2 (hibrido BM25/B2 top-50 + cross-encoder B1 +
# prior alpha=1.5); "default" es BM25 + prior alpha=0.75 (intacta desde
# v2). Los 8 positivos son los ids del QA v2 (comparabilidad); 1992-2001
# aporta 4 casos y 2002-2011 / 2012-2021 dos cada uno.
CASOS = [
    ("pos_2337.json", "2337", "T", 1997, True, "default"),
    ("pos_2728.json", "2728", "SU", 1997, True, "default"),
    ("pos_6251.json", "6251", "T", 2001, True, "rerank"),
    ("pos_6968.json", "6968", "C", 2001, True, "default"),
    ("pos_13015.json", "13015", "C", 2007, True, "default"),
    ("pos_13494.json", "13494", "T", 2006, True, "default"),
    ("pos_19890.json", "19890", "T", 2013, True, "default"),
    ("pos_20178.json", "20178", "C", 2013, True, "rerank"),
    ("neg_gaitan.json", None, None, None, False, "default"),
    ("neg_petro.json", None, None, None, False, "default"),
]

# Expectativas de ranking v2.2 (top-10 de numeros capturado ejecutando el
# producto v2.2 el 2026-10-04; ver docstring). Cualquier regeneracion de
# los JSON del QA exige actualizar esta tabla (ADR-16).
TOP10_ESPERADO = {
    "pos_2337.json": [
        "241", "86", "2", "13", "9", "1", "250", "29", "48", "53",
    ],
    "pos_2728.json": [
        "241", "86", "48", "2", "13", "9", "29", "67", "1", "356",
    ],
    "pos_6251.json": [
        "241", "86", "48", "33", "53", "23", "35", "150", "90", "62",
    ],
    "pos_6968.json": [
        "241", "86", "2", "13", "9", "48", "1", "242", "4", "243",
    ],
    "pos_13015.json": [
        "241", "29", "86", "1", "13", "9", "2", "48", "4", "150",
    ],
    "pos_13494.json": [
        "241", "86", "2", "9", "48", "13", "312", "29", "1", "313",
    ],
    "pos_19890.json": [
        "86", "241", "39", "13", "361", "9", "2", "1", "103", "29",
    ],
    "pos_20178.json": [
        "241", "150", "53", "58", "2", "125", "253", "209", "279", "217",
    ],
}

CAMPOS_RAIZ = [
    ("es_sentencia", "bool"),
    ("score_validacion", "num"),
    ("id_detectado", "str|null"),
    ("ponente", "str|null"),
    ("area", "dict|null"),
    ("articulos_cp", "dict"),
    ("sentencias_citadas", "list"),
    ("resumen", "str"),
    ("tiempo_segundos", "num"),
]

SUBCAMPOS_ARTICULOS = [
    ("extraidos", "list"),
    ("recuperados_bm25", "list"),
    ("reranker_usado", "bool"),
    ("hubs_usado", "bool"),
    ("alpha_hubs", "num"),
]

RESUMEN_NEGATIVO = (
    "El texto no cumple la heuristica de providencia de la Corte "
    "Constitucional (menos de 2/3 reglas); no se ejecutaron los "
    "componentes de analisis."
)

# Texto nuevo del resumen en modo rerank (ciclo v2.2): nombra los
# componentes efectivos de la ruta (denso B2 + cross-encoder B1). El
# modo default conserva "recuperados por BM25" de v2 sin cambios.
VIA_RERANK_V22 = "recuperados por hibrido BM25+B2+reranker (B1)"

# Artefactos canonicos v2 congelados (md5 intactos, ADR-11/16).
ARTEFACTOS_MD5_V2 = {
    "data/prior_hubs_cp.json": "80ef66a2fa88980bb43e8e6ed7fcde44",
    "resultados/m4_baseline.json": "60f8fd1ce4c2edef086d8ee01aaad654",
    "resultados/m4_jerarquico.json": "94425f6eab8ba03dd7b74d0f3962ca40",
    "resultados/m4_eurovoc.json": "64496c31d330c6bca148561eaaa50cc7",
    "resultados/vinculo_cp_benchmark_v8.json":
        "746621a797fe109335e2a43367717a77",
}

# Componentes nuevos del Sprint B adoptados en v2.2 (pesos, config y
# tokenizers de los repositorios locales); md5 capturados el 2026-10-04.
ARTEFACTOS_MD5_V22 = {
    "modelos/b2_retriever/model.safetensors":
        "bb40f530d1d67166f1e839c1f5a68cd0",
    "modelos/b2_retriever/config.json":
        "7154c00078bdb76e0dc637465fd904a1",
    "modelos/b2_retriever/modules.json":
        "1f17497e0cd4d0bee2b7dd653a139225",
    "modelos/b2_retriever/sentence_bert_config.json":
        "24fb91122bcc7a3e063ae455a36d6e57",
    "modelos/b2_retriever/tokenizer.json":
        "2380b47c4e9b70366a481162a48f6aff",
    "modelos/b1_reranker/model.safetensors":
        "04fab45c5798cf7e8d199be1a5306121",
    "modelos/b1_reranker/config.json":
        "b294a9dc5aab254f34c99139cb5a5cd4",
    "modelos/b1_reranker/tokenizer.json":
        "1e27a6ba9dd82d5235b7846e13630c82",
}

# Evidencia de logs v2.2 (el esquema JSON no cambio, ADR-16).
LOG_DENSO_B2 = "denso B2 (modelos/b2_retriever)"
LOG_RERANKER_B1_SUFIJO = "modelos/b1_reranker"
LOGS_LEGACY_E5 = (
    "[info] E5 ",
    "[info] reranker BAAI/bge-reranker-v2-m3",
)
LOGS_FALLBACK_V22 = (
    "[aviso] modo rerank v2.2: falta",
    "[aviso] denso B2 no disponible",
)

# Benchmark v13 (Sprint B2) para el contraste condicional de ranking.
BENCHMARK_V13 = "resultados/vinculo_cp_benchmark_v13.json"
VERSION_BENCHMARK_V13 = "v13-ciclo3-sprintB2"
ESTRATEGIA_V13_B2 = "hibrido_b2_rerank_chunks_top50_b1"


# ===== SECCION 1: utilidades de verificacion =====
def _tipo_ok(valor: object, tipo: str) -> bool:
    """Indica si un valor JSON corresponde al tipo esperado.

    Args:
        valor: Valor decodificado de JSON.
        tipo: Uno o varios tipos separados por "|": "bool", "num", "str",
            "list", "dict" o "null". "num" excluye bool (en JSON true/false
            no es un numero valido).

    Returns:
        True si el valor cumple alguno de los tipos declarados.
    """
    for opcion in tipo.split("|"):
        if opcion == "null" and valor is None:
            return True
        if opcion == "bool" and isinstance(valor, bool):
            return True
        if (
            opcion == "num"
            and isinstance(valor, (int, float))
            and not isinstance(valor, bool)
        ):
            return True
        if opcion == "str" and isinstance(valor, str):
            return True
        if opcion == "list" and isinstance(valor, list):
            return True
        if opcion == "dict" and isinstance(valor, dict):
            return True
    return False


def _verificar_campo(
    archivo: str, nombre: str, valor: object, tipo: str,
    fallos: list[str],
) -> bool:
    """Registra un fallo si el valor no cumple el tipo declarado."""
    if not _tipo_ok(valor, tipo):
        fallos.append(
            f"{archivo}: {nombre} tipo {type(valor).__name__} != {tipo}"
        )
        return False
    return True


def _resumen_esperado(
    top1: dict, extraidos: list, recuperados: list, n_citas: int,
    n_resueltas: int, modo: str, alpha: float,
) -> str:
    """Reconstruye el resumen del producto para contrastarlo con el JSON.

    Replica el formato exacto de construir_resumen() de
    10_evaluar_sentencia.py (v2.2): el modo rerank nombra el hibrido
    BM25+B2+reranker (B1); el default conserva el texto de v2. La
    igualdad estricta prueba que el resumen es coherente con los campos
    estructurados del JSON.
    """
    parte_area = (
        f"area tematica {top1['area']} (prob {top1['prob']})"
        if top1
        else "sin area"
    )
    via = VIA_RERANK_V22 if modo == "rerank" else "recuperados por BM25"
    via += f" con prior de hubs (alpha={alpha:g})"
    return (
        f"Providencia valida de la Corte Constitucional; {parte_area}; "
        f"{len(extraidos)} articulos CP extraidos y {len(recuperados)} "
        f"{via}; {n_resueltas} de {n_citas} sentencias citadas resueltas "
        "al corpus."
    )


# ===== SECCION 2: checks de los casos E2E =====
def verificar_caso(
    archivo: str, id_doc: str | None, esperado: bool, modo: str,
    qa_dir: Path,
) -> tuple[str | None, list[str]]:
    """Verifica un JSON del QA v2.2 contra los criterios de aceptacion.

    Args:
        archivo: Nombre del JSON dentro de qa_dir.
        id_doc: Id del corpus esperado en positivos (None en negativos).
        esperado: es_sentencia esperado (True positivo, False negativo).
        modo: "default" (BM25 + prior 0.75) o "rerank" (ruta v2.2 B2/B1).
        qa_dir: Directorio con los JSON del QA.

    Returns:
        Tupla (fila, fallos): fila de la tabla (None si no existe el
        archivo) y lista de fallos criticos encontrados.
    """
    fallos: list[str] = []
    ruta = qa_dir / archivo
    if not ruta.is_file():
        return None, [f"{archivo}: no existe en {qa_dir}"]
    with open(ruta, encoding="utf-8") as manejador:
        salida = json.load(manejador)

    for campo, tipo in CAMPOS_RAIZ:
        if campo not in salida:
            fallos.append(f"{archivo}: falta campo {campo}")
        else:
            _verificar_campo(archivo, campo, salida[campo], tipo, fallos)

    if salida.get("es_sentencia") != esperado:
        fallos.append(
            f"{archivo}: es_sentencia={salida.get('es_sentencia')} "
            f"esperado {esperado}"
        )

    tiempo = salida.get("tiempo_segundos")
    tope = TIEMPO_MAX[modo if esperado else "default"]
    if _tipo_ok(tiempo, "num") and not 0 < tiempo < tope:
        fallos.append(
            f"{archivo}: tiempo {tiempo} fuera de (0, {tope}) s"
        )

    articulos = salida.get("articulos_cp")
    if not isinstance(articulos, dict):
        fallos.append(f"{archivo}: articulos_cp no es dict")
        articulos = {}
    for sub, tipo in SUBCAMPOS_ARTICULOS:
        if sub not in articulos:
            fallos.append(f"{archivo}: falta articulos_cp.{sub}")
        else:
            _verificar_campo(
                archivo, f"articulos_cp.{sub}", articulos[sub], tipo, fallos
            )

    if esperado:
        fila, fallos_caso = _verificar_positivo(
            archivo, id_doc, modo, salida, articulos
        )
    else:
        fila, fallos_caso = _verificar_negativo(archivo, salida, articulos)
    fallos.extend(fallos_caso)
    return fila, fallos


def _verificar_positivo(
    archivo: str, id_doc: str | None, modo: str, salida: dict,
    articulos: dict,
) -> tuple[str, list[str]]:
    """Aplica los checks especificos de una providencia positiva."""
    fallos: list[str] = []

    # 1. Validacion y area top-3.
    score = salida.get("score_validacion")
    if _tipo_ok(score, "num") and not 0 < score <= 1:
        fallos.append(f"{archivo}: score_validacion {score} fuera de (0, 1]")
    area = salida.get("area")
    top1: dict = {}
    if not isinstance(area, dict):
        fallos.append(f"{archivo}: area no es dict")
    else:
        _verificar_campo(archivo, "area.via", area.get("via"), "str", fallos)
        top3 = area.get("top3")
        if not isinstance(top3, list) or len(top3) != 3:
            fallos.append(f"{archivo}: area.top3 != 3 entradas")
            top3 = []
        probs: list[float] = []
        for indice, item in enumerate(top3):
            if (
                not isinstance(item, dict)
                or not _tipo_ok(item.get("area"), "str")
                or not item.get("area")
                or not _tipo_ok(item.get("prob"), "num")
            ):
                fallos.append(
                    f"{archivo}: area.top3[{indice}] sin area/prob validos"
                )
                continue
            if not 0 < item["prob"] <= 1:
                fallos.append(
                    f"{archivo}: area.top3[{indice}] prob fuera de (0, 1]: "
                    f"{item['prob']}"
                )
            probs.append(item["prob"])
        if len(probs) == 3 and probs != sorted(probs, reverse=True):
            fallos.append(f"{archivo}: area.top3 no ordenado por prob")
        if top3:
            top1 = top3[0]

    # 2. Articulos extraidos (M2).
    extraidos = articulos.get("extraidos") or []
    if not extraidos:
        fallos.append(f"{archivo}: 0 articulos extraidos")
    for indice, item in enumerate(extraidos):
        if not isinstance(item, dict):
            fallos.append(f"{archivo}: extraidos[{indice}] no es dict")
            continue
        for clave in ("numero", "texto_150chars", "bloque"):
            if not _tipo_ok(item.get(clave), "str") or not item.get(clave):
                fallos.append(
                    f"{archivo}: extraidos[{indice}].{clave} invalido"
                )

    # 3. Retrieval top-10 (M7 v2.2) y expectativa de ranking capturada.
    recuperados = articulos.get("recuperados_bm25") or []
    numeros: list[str] = []
    scores: list[float] = []
    for indice, item in enumerate(recuperados):
        if not isinstance(item, dict):
            fallos.append(f"{archivo}: recuperados[{indice}] no es dict")
            continue
        numero = item.get("numero")
        score_rec = item.get("score")
        if not _tipo_ok(numero, "str") or not numero:
            fallos.append(f"{archivo}: recuperados[{indice}].numero invalido")
        else:
            numeros.append(numero)
        if not _tipo_ok(score_rec, "num") or not math.isfinite(score_rec):
            fallos.append(f"{archivo}: recuperados[{indice}].score invalido")
        else:
            scores.append(score_rec)
    if len(recuperados) != K_RETRIEVAL:
        fallos.append(
            f"{archivo}: retrieval {len(recuperados)} != {K_RETRIEVAL}"
        )
    if len(set(numeros)) != len(numeros):
        fallos.append(f"{archivo}: retrieval con numeros duplicados")
    malos = [
        n for n in numeros
        if n.startswith("Transitorio") or n == "Preambulo"
    ]
    if malos:
        fallos.append(
            f"{archivo}: retrieval con transitorios/preambulo: {malos}"
        )
    if scores != sorted(scores, reverse=True):
        fallos.append(f"{archivo}: retrieval no ordenado por score")
    esperado_top10 = TOP10_ESPERADO.get(archivo)
    if esperado_top10 is not None and numeros != esperado_top10:
        fallos.append(
            f"{archivo}: top-10 {numeros} != expectativa v2.2 "
            f"{esperado_top10}"
        )

    # 4. Prior de hubs y modo (Fase X / v2.2).
    alpha = articulos.get("alpha_hubs")
    alpha_esperado = ALFA_POR_MODO[modo]
    if not _tipo_ok(alpha, "num") or abs(alpha - alpha_esperado) > 1e-9:
        fallos.append(
            f"{archivo}: alpha_hubs={alpha} != {alpha_esperado} "
            f"(modo {modo})"
        )
    if articulos.get("hubs_usado") is not True:
        fallos.append(
            f"{archivo}: hubs_usado={articulos.get('hubs_usado')} != True"
        )
    reranker_esperado = modo == "rerank"
    if articulos.get("reranker_usado") is not reranker_esperado:
        fallos.append(
            f"{archivo}: reranker_usado={articulos.get('reranker_usado')} "
            f"incoherente con modo {modo}"
        )

    # 5. Identidad y citas resueltas (M5).
    if salida.get("id_detectado") != id_doc:
        fallos.append(
            f"{archivo}: id_detectado={salida.get('id_detectado')} "
            f"!= {id_doc}"
        )
    citas = salida.get("sentencias_citadas") or []
    n_resueltas = 0
    for indice, cita in enumerate(citas):
        if not isinstance(cita, dict):
            fallos.append(f"{archivo}: citas[{indice}] no es dict")
            continue
        _verificar_campo(
            archivo, f"citas[{indice}].id", cita.get("id"), "str", fallos
        )
        if cita.get("resuelta") is True:
            n_resueltas += 1
            if not _tipo_ok(cita.get("ano"), "num"):
                fallos.append(
                    f"{archivo}: cita resuelta {cita.get('id')} sin ano"
                )
            pagerank = cita.get("pagerank")
            if not _tipo_ok(pagerank, "num") or pagerank <= 0:
                fallos.append(
                    f"{archivo}: cita resuelta {cita.get('id')} con "
                    f"pagerank invalido {pagerank}"
                )
            if not _tipo_ok(cita.get("area"), "str") or not cita.get("area"):
                fallos.append(
                    f"{archivo}: cita resuelta {cita.get('id')} sin area"
                )
        elif cita.get("resuelta") is False:
            if cita.get("pagerank") is not None:
                fallos.append(
                    f"{archivo}: cita no resuelta con pagerank no nulo"
                )
            if cita.get("area") is not None:
                fallos.append(
                    f"{archivo}: cita no resuelta con area no nula"
                )
        else:
            fallos.append(
                f"{archivo}: citas[{indice}].resuelta no es bool"
            )

    # 6. Coherencia del resumen con los campos estructurados (v2.2).
    if top1 and _tipo_ok(top1.get("area"), "str") and _tipo_ok(
        top1.get("prob"), "num"
    ):
        resumen = _resumen_esperado(
            top1, extraidos, recuperados, len(citas), n_resueltas, modo,
            alpha if _tipo_ok(alpha, "num") else 0.0,
        )
        if salida.get("resumen") != resumen:
            fallos.append(
                f"{archivo}: resumen no coincide con los campos del JSON"
            )
    else:
        fallos.append(f"{archivo}: top1 invalido, resumen no verificable")

    fila = (
        f"{archivo:<18} {modo:<8} {salida.get('es_sentencia')!s:<8} "
        f"{top1.get('area', '-')!s:<28} {top1.get('prob', '-')!s:<7} "
        f"{len(extraidos):<5} {len(recuperados):<6} {len(citas):<6} "
        f"{n_resueltas:<5} {alpha!s:<7} {salida.get('tiempo_segundos')!s:<6}"
    )
    return fila, fallos


def _verificar_negativo(
    archivo: str, salida: dict, articulos: dict
) -> tuple[str, list[str]]:
    """Aplica los checks especificos de un texto no-providencia."""
    fallos: list[str] = []
    if salida.get("score_validacion") != 0.0:
        fallos.append(
            f"{archivo}: score_validacion={salida.get('score_validacion')} "
            "!= 0.0"
        )
    for campo in ("area", "ponente", "id_detectado"):
        if salida.get(campo) is not None:
            fallos.append(f"{archivo}: negativo con {campo} != None")
    if articulos.get("extraidos"):
        fallos.append(f"{archivo}: negativo con articulos extraidos")
    if articulos.get("recuperados_bm25"):
        fallos.append(f"{archivo}: negativo con retrieval no vacio")
    if salida.get("sentencias_citadas"):
        fallos.append(f"{archivo}: negativo con citas no vacias")
    if articulos.get("hubs_usado") is not False:
        fallos.append(
            f"{archivo}: negativo con hubs_usado={articulos.get('hubs_usado')}"
        )
    if articulos.get("reranker_usado") is not False:
        fallos.append(
            f"{archivo}: negativo con reranker_usado="
            f"{articulos.get('reranker_usado')}"
        )
    alpha = articulos.get("alpha_hubs")
    if not _tipo_ok(alpha, "num") or alpha != 0.0:
        fallos.append(f"{archivo}: negativo con alpha_hubs={alpha} != 0.0")
    if salida.get("resumen") != RESUMEN_NEGATIVO:
        fallos.append(f"{archivo}: resumen de rechazo no coincide")
    fila = (
        f"{archivo:<18} {'-':<8} {salida.get('es_sentencia')!s:<8} "
        f"{'-':<28} {'-':<7} {'-':<5} {'-':<6} {'-':<6} {'-':<5} "
        f"{'-':<7} {salida.get('tiempo_segundos')!s:<6}"
    )
    return fila, fallos


def verificar_logs(qa_dir: Path) -> tuple[list[str], list[str]]:
    """Verifica la evidencia de carga B2/B1 en los logs del QA v2.2.

    Los logs se generan en qa_dir/logs/<archivo>.log por el runner del QA
    (stdout+stderr combinados; el nombre real es "pos_XXXX.json.log").
    Para los 2 casos rerank es CRITICO: el esquema JSON no cambio (ADR-16)
    y el log es la unica evidencia de que la ruta B2/B1 (y no el fallback
    E5/bge) se uso. En los casos default se exige que NO se cargue ningun
    modelo (el modo default no debe pagar ese costo).

    Returns:
        Tupla (observaciones, fallos): observaciones informativas y
        fallos criticos encontrados.
    """
    observaciones: list[str] = []
    fallos: list[str] = []
    dir_logs = qa_dir / "logs"
    for archivo, _id, _tipo, _ano, esperado, modo in CASOS:
        ruta = dir_logs / f"{archivo}.log"
        if not ruta.is_file():
            if esperado and modo == "rerank":
                fallos.append(
                    f"{archivo}: sin log; evidencia de carga B2/B1 no "
                    "verificable"
                )
            else:
                observaciones.append(f"{archivo}: sin log (informativo)")
            continue
        texto = ruta.read_text(encoding="utf-8", errors="replace")
        if esperado and modo == "rerank":
            ok_b2 = LOG_DENSO_B2 in texto
            ok_b1 = (
                "reranker " in texto and LOG_RERANKER_B1_SUFIJO in texto
            )
            legado = [s for s in LOGS_LEGACY_E5 if s in texto]
            fallback = [s for s in LOGS_FALLBACK_V22 if s in texto]
            if not (ok_b2 and ok_b1):
                fallos.append(
                    f"{archivo}: log rerank sin carga B2/B1 "
                    f"(denso B2={ok_b2}, reranker B1={ok_b1})"
                )
            if legado or fallback:
                fallos.append(
                    f"{archivo}: log rerank con ruta legacy/fallback: "
                    f"{legado + fallback}"
                )
            observaciones.append(
                f"{archivo}: log rerank B2={ok_b2} B1={ok_b1} "
                f"fallback={bool(fallback)}"
            )
        else:
            carga = [
                s for s in (LOG_DENSO_B2, "[info] E5 ", "[info] reranker ")
                if s in texto
            ]
            if carga:
                fallos.append(
                    f"{archivo}: log default con carga de modelos: {carga}"
                )
            else:
                observaciones.append(
                    f"{archivo}: log default sin carga de modelos [OK]"
                )
    return observaciones, fallos


# ===== SECCION 3: integridad de artefactos y metricas =====
def verificar_md5() -> list[str]:
    """Verifica los md5 de artefactos v2 congelados y de B2/B1 (v2.2)."""
    fallos: list[str] = []
    for etiqueta, tabla in (
        ("v2", ARTEFACTOS_MD5_V2),
        ("v2.2", ARTEFACTOS_MD5_V22),
    ):
        for relativo, esperado in sorted(tabla.items()):
            ruta = ROOT / relativo
            if not ruta.is_file():
                fallos.append(f"md5 {etiqueta} {relativo}: archivo no existe")
                continue
            obtenido = hashlib.md5(ruta.read_bytes()).hexdigest()
            if obtenido != esperado:
                fallos.append(
                    f"md5 {etiqueta} {relativo}: {obtenido} != {esperado}"
                )
    return fallos


def verificar_config_v22() -> tuple[list[str], list[str]]:
    """Verifica la integracion v2.2 en lib/componentes_producto.py.

    Comprueba los defaults congelados por el benchmark v8 (k=10, top-50,
    3 chunks, pesos 0.5/0.5, alphas 0.75/1.5), el prior de hubs, las
    caches E5 (fallback) y B2, los flags y rutas B2/B1 del Sprint B, el
    fallback E5/bge documentado y que el denso/reranker se cargan lazy.

    Returns:
        Tupla (observaciones, fallos).
    """
    observaciones: list[str] = []
    fallos: list[str] = []
    esperados = {
        "k_bm25_default": 10,
        "usar_hubs_default": True,
        "alpha_hubs_base": 0.75,
        "alpha_hubs_rerank": 1.5,
        "k_candidatos_rerank_top50": 50,
        "k_chunks_rerank": 3,
        "version_cache": 3,
        "ruta_embedding_b2": "modelos/b2_retriever",
        "ruta_reranker_b1": "modelos/b1_reranker",
        "usar_b2_b1_en_rerank": True,
        "estricto_b2_b1": False,
    }
    for clave, esperado in esperados.items():
        obtenido = CONFIG.get(clave)
        if obtenido != esperado:
            fallos.append(
                f"CONFIG['{clave}']={obtenido!r} != {esperado!r}"
            )
    # La clave "e5" del dict de pesos es el nombre historico del denso del
    # hibrido (v2); en v2.2 el denso efectivo es B2 con el mismo peso.
    if CONFIG.get("pesos_hibrido") != {"bm25": 0.5, "e5": 0.5}:
        fallos.append(
            f"CONFIG['pesos_hibrido']={CONFIG.get('pesos_hibrido')!r} "
            "!= 0.5/0.5"
        )
    prior = ROOT / CONFIG["ruta_prior_hubs"]
    if not prior.is_file():
        fallos.append(f"prior de hubs no existe: {CONFIG['ruta_prior_hubs']}")
    else:
        with open(prior, encoding="utf-8") as manejador:
            hubs = json.load(manejador)
        if len(hubs) != 497:
            fallos.append(f"prior de hubs con {len(hubs)} canones != 497")
        if hubs.get("241") != 1.0:
            fallos.append("prior de hubs: canon 241 != 1.0")
    cache_e5 = ROOT / CONFIG["dir_cache_e5"] / CONFIG["archivo_cache_e5"]
    if not cache_e5.is_file():
        fallos.append(f"cache E5 de pasajes no existe: {cache_e5}")
    cache_b2 = ROOT / CONFIG["dir_cache_b2"] / CONFIG["archivo_cache_b2"]
    if not cache_b2.is_file():
        fallos.append(f"cache B2 de pasajes no existe: {cache_b2}")
    for ruta_b2_b1 in (CONFIG["ruta_embedding_b2"], CONFIG["ruta_reranker_b1"]):
        if not (ROOT / ruta_b2_b1).is_dir():
            fallos.append(f"componente v2.2 no existe: {ruta_b2_b1}")

    firma = inspect.signature(VinculoCP.__init__).parameters
    for nombre, esperado in (
        ("usar_hubs", True),
        ("alpha_hubs_base", 0.75),
        ("alpha_hubs_rerank", 1.5),
        ("incluir_transitorios", False),
        ("usar_b2_b1_en_rerank", True),
        ("estricto_b2_b1", False),
    ):
        obtenido = firma[nombre].default if nombre in firma else "AUSENTE"
        if obtenido != esperado:
            fallos.append(
                f"VinculoCP({nombre}={obtenido!r}) != default {esperado!r}"
            )
    fuente = inspect.getsource(VinculoCP._scores_candidatos_rerank).lower()
    if "fallback" not in fuente or "e5" not in fuente or "bm25" not in fuente:
        fallos.append(
            "fallback E5->BM25 no documentado en "
            "_scores_candidatos_rerank"
        )
    fuente_ruta = inspect.getsource(VinculoCP._resolver_ruta_rerank).lower()
    if (
        "fallback" not in fuente_ruta
        or "e5" not in fuente_ruta
        or "bge" not in fuente_ruta
    ):
        fallos.append(
            "fallback E5/bge de v2.2 no documentado en "
            "_resolver_ruta_rerank"
        )

    # Instancias reales del componente (denso/reranker deben seguir sin
    # cargar en init y la ruta B2/B1 debe resolverse solo en rerank).
    for modo, activar, alpha in (
        ("default", False, 0.75), ("rerank", True, 1.5)
    ):
        vinculo = VinculoCP(usar_reranker=activar)
        if not vinculo.usar_hubs:
            fallos.append(f"VinculoCP({modo}): usar_hubs=False")
        if abs(vinculo.alpha_hubs - alpha) > 1e-9:
            fallos.append(
                f"VinculoCP({modo}): alpha={vinculo.alpha_hubs} != {alpha}"
            )
        if vinculo._denso is not None or vinculo._reranker is not None:
            fallos.append(
                f"VinculoCP({modo}): modelos cargados en init (no lazy)"
            )
        if activar and not vinculo._b2_b1_efectivo:
            fallos.append(
                "VinculoCP(rerank): B2/B1 no efectivos (fallback activo)"
            )
        if not activar and vinculo._b2_b1_efectivo:
            fallos.append(
                "VinculoCP(default): ruta B2/B1 efectiva sin usar_reranker"
            )
        etiqueta_denso = "B2" if activar else "E5"
        if vinculo._etiqueta_denso != etiqueta_denso:
            fallos.append(
                f"VinculoCP({modo}): etiqueta denso "
                f"{vinculo._etiqueta_denso!r} != {etiqueta_denso!r}"
            )
    observaciones.append(
        "VinculoCP default/rerank: hubs activo, alpha 0.75/1.5, B2/B1 "
        "efectivo, lazy OK"
    )
    return observaciones, fallos


def _comparar_cifra(
    etiqueta: str, obtenido: object, esperado: float, fallos: list[str],
    tolerancia: float = 5e-4,
) -> None:
    """Registra un fallo si una cifra registrada difiere de la esperada."""
    if (
        not isinstance(obtenido, (int, float))
        or isinstance(obtenido, bool)
        or abs(obtenido - esperado) > tolerancia
    ):
        fallos.append(f"{etiqueta}={obtenido!r} != {esperado}")


def _cargar_json(relativo: str) -> dict:
    """Carga un JSON del proyecto o devuelve {} si falta (lo reporta)."""
    ruta = ROOT / relativo
    if not ruta.is_file():
        return {}
    with open(ruta, encoding="utf-8") as manejador:
        return json.load(manejador)


def verificar_metricas_v2() -> list[str]:
    """Verifica que las metricas v2 registradas no tengan drift.

    Contrasta las cifras del benchmark v8, los JSON de modelos v2
    (chunk-mean BETO, jerarquico C) y el clasificador v1 (sin cambios)
    contra lo documentado en el QA v1/v2. La integridad de v2 sigue
    vigente en v2.2 (los artefactos estan congelados, ADR-16).

    Returns:
        Lista de fallos criticos (vacia si todo coincide).
    """
    fallos: list[str] = []

    metricas = _cargar_json("resultados/metricas.json")
    iter2 = metricas.get("ciclo_v2", {}).get("w_faseW_a_iter2", {})
    if not iter2:
        fallos.append("metricas.json: falta ciclo_v2.w_faseW_a_iter2")
    else:
        estrategias = iter2.get("recomendacion_por_estrategia", {})
        for nombre in (
            "bm25", "hibrido", "bm25_rerank_chunks_top50",
            "hibrido_rerank_chunks_top50",
        ):
            if estrategias.get(nombre) != "adoptar":
                fallos.append(
                    f"metricas.json: {nombre} recomendacion != adoptar"
                )
        focales = iter2.get("full_90_alpha0_vs_alpha_elegido", {})
        calibracion = iter2.get("calibracion", {})
        if (
            calibracion.get("mejor_estrategia_figura")
            != "hibrido_rerank_chunks_top50"
        ):
            fallos.append(
                "metricas.json: mejor_estrategia_figura != ganadora"
            )
        ganadora = focales.get("hibrido_rerank_chunks_top50", {})
        _comparar_cifra(
            "v8 ganadora alpha full",
            ganadora.get("alpha_elegido", {}).get("full"), 1.5, fallos,
        )
        elegido = ganadora.get("full_alpha_elegido", {})
        _comparar_cifra(
            "v8 ganadora F1@5", elegido.get("f1_at_5"), 0.4246, fallos
        )
        _comparar_cifra(
            "v8 ganadora MRR", elegido.get("mrr"), 0.9505, fallos
        )
        _comparar_cifra(
            "v8 ganadora nDCG@10", elegido.get("ndcg_at_10"), 0.6198,
            fallos,
        )
        _comparar_cifra(
            "v8 ganadora F1@10", elegido.get("f1_at_10"), 0.3793, fallos
        )
        base = focales.get("bm25", {})
        _comparar_cifra(
            "v8 bm25 alpha full",
            base.get("alpha_elegido", {}).get("full"), 0.75, fallos,
        )
        _comparar_cifra(
            "v8 bm25 F1@5",
            base.get("full_alpha_elegido", {}).get("f1_at_5"), 0.3955,
            fallos,
        )
        _comparar_cifra(
            "v8 bm25 MRR",
            base.get("full_alpha_elegido", {}).get("mrr"), 0.9015, fallos,
        )

    v8 = _cargar_json("resultados/vinculo_cp_benchmark_v8.json")
    if not v8:
        fallos.append("falta resultados/vinculo_cp_benchmark_v8.json")
    else:
        if v8.get("version") != "v8-ciclo2-faseW-a-iter2":
            fallos.append(f"v8 version={v8.get('version')!r} inesperada")
        _comparar_cifra(
            "v8 hibrido_rerank_chunks_top50 F1@5 (alpha 0)",
            v8.get("global", {})
            .get("hibrido_rerank_chunks_top50", {})
            .get("f1_at_k"),
            0.3399,
            fallos,
        )
        if len(v8.get("top10_por_caso", [])) != 90:
            fallos.append("v8 top10_por_caso != 90 casos")

    chunks = _cargar_json("resultados/m4_transformer_chunks.json")
    if not chunks:
        fallos.append("falta resultados/m4_transformer_chunks.json")
    else:
        _comparar_cifra(
            "BETO chunk-mean completo test",
            chunks.get("completo", {}).get("macro_f1"), 0.2433, fallos,
        )
        _comparar_cifra(
            "BETO chunk-mean enmascarado test",
            chunks.get("enmascarado", {}).get("macro_f1"), 0.2363, fallos,
        )

    jerarquico = _cargar_json("resultados/m4_jerarquico.json")
    if not jerarquico:
        fallos.append("falta resultados/m4_jerarquico.json")
    else:
        if jerarquico.get("seleccion", {}).get("ganadora") != "C_calibracion":
            fallos.append("m4_jerarquico: ganadora != C_calibracion")
        variante_c = jerarquico.get("variante_c_calibracion", {})
        _comparar_cifra(
            "Fase U C test macro-F1",
            variante_c.get("test_con_sesgos", {}).get("macro_f1"), 0.2980,
            fallos,
        )
        _comparar_cifra(
            "Fase U C val macro-F1",
            variante_c.get("val_con_sesgos", {}).get("macro_f1"), 0.3792,
            fallos,
        )

    baseline = _cargar_json("resultados/m4_baseline.json")
    if not baseline:
        fallos.append("falta resultados/m4_baseline.json")
    else:
        _comparar_cifra(
            "M4 LogReg completo test",
            baseline.get("completo", {}).get("logreg", {}).get("macro_f1"),
            0.2852,
            fallos,
        )
        _comparar_cifra(
            "M4 LogReg enmascarado test (producto)",
            baseline.get("enmascarado", {}).get("logreg", {}).get("macro_f1"),
            0.2870,
            fallos,
        )
    return fallos


def verificar_benchmark_v13(qa_dir: Path) -> tuple[list[str], list[str]]:
    """Contrasta los casos E2E presentes en el benchmark v13 (Sprint B2).

    La muestra del benchmark v13 son 90 casos (3 por anio, semilla 42)
    con ids que no tienen por que coincidir con los del QA. Si el id de
    un caso E2E positivo esta en top10_por_caso:
    - modo rerank: se contrasta el orden del top-10 contra la estrategia
      B2+B1 del benchmark (fallo critico si difiere);
    - modo default: solo observacion informativa (el v13 no re-evalua la
      ruta default del producto).
    Si ningun id coincide se reporta como observacion (contraste directo
    no aplicable; la equivalencia producto-benchmark queda cubierta por
    la reproducibilidad de la ruta B2/B1 en los smokes del Sprint B2).

    Returns:
        Tupla (observaciones, fallos).
    """
    observaciones: list[str] = []
    fallos: list[str] = []
    ruta = ROOT / BENCHMARK_V13
    if not ruta.is_file():
        fallos.append(f"benchmark v13 no existe: {BENCHMARK_V13}")
        return observaciones, fallos
    with open(ruta, encoding="utf-8") as manejador:
        bench = json.load(manejador)
    if bench.get("version") != VERSION_BENCHMARK_V13:
        fallos.append(
            f"benchmark v13 version={bench.get('version')!r} != "
            f"{VERSION_BENCHMARK_V13!r}"
        )
    casos_bench = bench.get("top10_por_caso") or {}
    if len(casos_bench) != 90:
        fallos.append(
            f"benchmark v13 top10_por_caso={len(casos_bench)} != 90"
        )
    en_bench = 0
    for archivo, id_doc, _tipo, _ano, esperado, modo in CASOS:
        if not esperado or id_doc is None or str(id_doc) not in casos_bench:
            continue
        en_bench += 1
        ruta_json = qa_dir / archivo
        if not ruta_json.is_file():
            fallos.append(f"{archivo}: no existe para contraste v13")
            continue
        with open(ruta_json, encoding="utf-8") as manejador:
            salida = json.load(manejador)
        numeros = [
            item["numero"]
            for item in salida["articulos_cp"]["recuperados_bm25"]
        ]
        if modo == "rerank":
            entrada = casos_bench[str(id_doc)].get(ESTRATEGIA_V13_B2) or []
            numeros_bench = [str(item["id"]) for item in entrada]
            if numeros != numeros_bench:
                fallos.append(
                    f"{archivo}: top-10 {numeros} != v13 "
                    f"{ESTRATEGIA_V13_B2} {numeros_bench}"
                )
            else:
                observaciones.append(
                    f"{archivo}: top-10 == v13 {ESTRATEGIA_V13_B2}"
                )
        else:
            observaciones.append(
                f"{archivo}: id en v13 pero modo default; sin contraste "
                "directo"
            )
    if en_bench == 0:
        observaciones.append(
            "0/8 ids E2E positivos en la muestra v13 (90 casos, 3/anio, "
            "semilla 42); contraste directo no aplicable"
        )
    return observaciones, fallos


# ===== SECCION 4: reporte y CLI =====
def parsear_args() -> argparse.Namespace:
    """Define y parsea los argumentos del verificador.

    Returns:
        Namespace con qa_dir y omitir_integridad.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Verificacion ejecutada del QA v2.2 de Juris-PLN: 10 casos "
            "E2E (ruta rerank B2/B1), integridad de artefactos v2/B2/B1, "
            "metricas v2 sin drift y contraste con el benchmark v13."
        )
    )
    parser.add_argument(
        "--qa-dir",
        default=str(Path("resultados") / "qa_final_v2_2"),
        help="Directorio con los 10 JSON del QA (default: qa_final_v2_2).",
    )
    parser.add_argument(
        "--omitir-integridad",
        action="store_true",
        help="Omite md5/config/metricas/benchmark (solo valida los casos).",
    )
    return parser.parse_args()


def main() -> None:
    """Ejecuta los checks, imprime el reporte y fija el exit code."""
    args = parsear_args()
    qa_dir = Path(args.qa_dir)
    if not qa_dir.is_absolute():
        qa_dir = ROOT / qa_dir
    if not qa_dir.is_dir():
        print(f"[error] no existe el directorio QA: {qa_dir}")
        sys.exit(1)

    print(f"QA dir: {qa_dir}")
    print(
        f"{'caso':<18} {'modo':<8} {'es_sent':<8} {'area top-1':<28} "
        f"{'p_top1':<7} {'arts':<5} {'top10':<6} {'citas':<6} {'res':<5} "
        f"{'alpha':<7} {'t(s)':<6}"
    )
    fallos: list[str] = []
    for archivo, id_doc, _tipo, _ano, esperado, modo in CASOS:
        fila, fallos_caso = verificar_caso(
            archivo, id_doc, esperado, modo, qa_dir
        )
        if fila is not None:
            print(fila)
        fallos.extend(fallos_caso)

    observaciones, fallos_logs = verificar_logs(qa_dir)
    fallos.extend(fallos_logs)
    if observaciones:
        print("\nEvidencia de logs v2.2 (B2/B1):")
        for linea in observaciones:
            print(f"  - {linea}")

    if not args.omitir_integridad:
        print("\nIntegridad de artefactos y configuracion v2.2:")
        fallos_md5 = verificar_md5()
        obs_config, fallos_config = verificar_config_v22()
        fallos_metricas = verificar_metricas_v2()
        obs_bench, fallos_bench = verificar_benchmark_v13(qa_dir)
        fallos.extend(fallos_md5 + fallos_config + fallos_metricas)
        fallos.extend(fallos_bench)
        for linea in obs_config:
            print(f"  [info] {linea}")
        for linea in obs_bench:
            print(f"  [info] {linea}")
        print(
            f"  [{'ok' if not fallos_md5 else 'FALLO'}] md5 canonicos v2 "
            f"({len(ARTEFACTOS_MD5_V2)}) + componentes B2/B1 "
            f"({len(ARTEFACTOS_MD5_V22)})"
        )
        print(
            f"  [{'ok' if not fallos_config else 'FALLO'}] config "
            "VinculoCP v2.2 / fallback E5/bge / caches E5+B2"
        )
        print(
            f"  [{'ok' if not fallos_metricas else 'FALLO'}] metricas v2 "
            "sin drift (metricas.json + JSONs de referencia)"
        )
        print(
            f"  [{'ok' if not fallos_bench else 'FALLO'}] benchmark v13 "
            "(version + muestra; contraste E2E condicional)"
        )

    print()
    if fallos:
        print(f"FALLOS ({len(fallos)}):")
        for fallo in fallos:
            print(f"  - {fallo}")
        sys.exit(1)
    print(
        "TODOS LOS CHECKS PASARON (10/10 casos v2.2 + integridad v2/B2/B1)"
    )


if __name__ == "__main__":
    main()
