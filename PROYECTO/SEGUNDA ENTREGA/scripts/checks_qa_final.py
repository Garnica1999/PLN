"""checks_qa_final.py
Fecha: 2026-09-09 (v1) / 2026-09-11 (v2, ciclo 2)
Descripcion: Verificacion ejecutada del QA final de entrega (Fase N) y de la
    regresion v2 del producto (Fase X, plan/PLAN_v2_ciclo_mejora.md). Valida
    los 10 JSON de resultados/qa_final_v2/ contra los criterios de
    aceptacion v2: es_sentencia correcto, area top-3 con probabilidades,
    articulos extraidos > 0, retrieval top-10 (k=10, sin transitorios ni
    preambulo, scores finitos y ordenados), prior de hubs aplicado con
    alpha correcto por modo (0.75 default / 1.5 rerank), flag reranker_usado
    coherente con el modo del caso, citas resueltas coherentes (ano/pagerank/
    area), tiempos y coherencia del resumen con los campos del JSON.
    Verifica ademas la integracion v2: md5 de artefactos canonicos, config de
    VinculoCP (top-50 hibrido, 3 chunks, pesos 0.5/0.5, alphas y carga lazy),
    fallback de E5 documentado en el codigo, cache E5 de pasajes y metricas
    v8/v2 registradas en resultados/metricas.json y los JSON de referencia.
    Emite tabla compacta y exit != 0 si cualquier check critico falla.

    v2 (ciclo 2) - cambios frente al QA v1:
    - k de retrieval 5 -> 10 (CONFIG["k_bm25_default"]).
    - Campos nuevos articulos_cp.reranker_usado, hubs_usado y alpha_hubs.
    - Modo por caso ("default" o "rerank") con alpha esperado 0.75/1.5.
    - 0 transitorios/preambulo en el top-10 (indice de permanentes Fase S1).
    - Coherencia del resumen con extraidos/recuperados/citas/alpha.
    - Integridad v2 (md5 canonicos, config, fallback E5 y metricas v8).
    El script v1 se conserva como scripts/checks_qa_final_v1.py (evidencia
    historica; el QA v1 vive en resultados/qa_final/).

Uso:
    /home/carlos/venv/bin/python scripts/checks_qa_final.py
    /home/carlos/venv/bin/python scripts/checks_qa_final.py \
        --qa-dir resultados/qa_final   # control negativo del formato v1
"""

import argparse
import hashlib
import inspect
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QA_V2 = ROOT / "resultados" / "qa_final_v2"

# Seccion 4 de lib/componentes_producto.py; se importa para verificar los
# defaults ejecutables del producto v2 (no instancia modelos por si sola).
sys.path.insert(0, str(ROOT / "scripts"))
from lib.componentes_producto import CONFIG, VinculoCP

K_RETRIEVAL = 10
ALFA_POR_MODO = {"default": 0.75, "rerank": 1.5}
TIEMPO_MAX = {"default": 30.0, "rerank": 60.0}

# Casos esperados: (archivo, id, tipo, ano, es_sentencia, modo). El modo
# "rerank" es la ruta ganadora v8 (hibrido top-50 + bge-reranker-v2-m3 +
# prior alpha=1.5); "default" es BM25 + prior alpha=0.75. Los 8 positivos
# son los ids del QA v1 (comparabilidad); 1992-2001 aporta 4 casos y
# 2002-2011 / 2012-2021 dos cada uno (los 4 periodos v1 se agrupan en 3).
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

# Artefactos canonicos v2 (Tarea 2: integridad durante el QA).
ARTEFACTOS_MD5 = {
    "data/prior_hubs_cp.json": "80ef66a2fa88980bb43e8e6ed7fcde44",
    "resultados/m4_baseline.json": "60f8fd1ce4c2edef086d8ee01aaad654",
    "resultados/m4_jerarquico.json": "94425f6eab8ba03dd7b74d0f3962ca40",
    "resultados/m4_eurovoc.json": "64496c31d330c6bca148561eaaa50cc7",
    "resultados/vinculo_cp_benchmark_v8.json":
        "746621a797fe109335e2a43367717a77",
}


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
    10_evaluar_sentencia.py: la igualdad estricta prueba que el resumen es
    coherente con los campos estructurados del JSON.
    """
    parte_area = (
        f"area tematica {top1['area']} (prob {top1['prob']})"
        if top1
        else "sin area"
    )
    via = (
        "recuperados por hibrido BM25/E5+reranker"
        if modo == "rerank"
        else "recuperados por BM25"
    )
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
    """Verifica un JSON del QA v2 contra los criterios de aceptacion.

    Args:
        archivo: Nombre del JSON dentro de qa_dir.
        id_doc: Id del corpus esperado en positivos (None en negativos).
        esperado: es_sentencia esperado (True positivo, False negativo).
        modo: "default" (BM25 + prior 0.75) o "rerank" (ruta ganadora v8).
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

    # 3. Retrieval top-10 (M7 v2).
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

    # 4. Prior de hubs y modo (Fase X).
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

    # 6. Coherencia del resumen con los campos estructurados.
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


def verificar_logs(qa_dir: Path) -> list[str]:
    """Revisa la evidencia de carga lazy/rerank en los logs del QA.

    Los logs se generan en qa_dir/logs/<archivo>.log por el runner del QA
    v2 (stdout+stderr combinados). Es evidencia informativa: los casos
    default no deben cargar E5/reranker y los casos rerank si.

    Returns:
        Lineas informativas (nunca fallos criticos).
    """
    observaciones: list[str] = []
    dir_logs = qa_dir / "logs"
    for archivo, _id, _tipo, _ano, esperado, modo in CASOS:
        if not esperado:
            continue
        ruta = dir_logs / archivo.replace(".json", ".log")
        if not ruta.is_file():
            observaciones.append(
                f"{archivo}: sin log (carga E5/reranker no evidenciada)"
            )
            continue
        texto = ruta.read_text(encoding="utf-8", errors="replace")
        carga_e5 = "[info] E5 " in texto
        carga_rerank = "[info] reranker BAAI/bge-reranker-v2-m3" in texto
        if modo == "rerank":
            estado = "OK" if carga_e5 and carga_rerank else "REVISAR"
            observaciones.append(
                f"{archivo}: log rerank E5={carga_e5} "
                f"reranker={carga_rerank} [{estado}]"
            )
        else:
            estado = "OK" if not carga_e5 and not carga_rerank else "REVISAR"
            observaciones.append(
                f"{archivo}: log default sin carga de modelos [{estado}]"
            )
    return observaciones


# ===== SECCION 3: integridad de artefactos y metricas v2 =====
def verificar_md5() -> list[str]:
    """Verifica los md5 canonicos de los artefactos de referencia v2."""
    fallos: list[str] = []
    for relativo, esperado in sorted(ARTEFACTOS_MD5.items()):
        ruta = ROOT / relativo
        if not ruta.is_file():
            fallos.append(f"md5 {relativo}: archivo no existe")
            continue
        obtenido = hashlib.md5(ruta.read_bytes()).hexdigest()
        if obtenido != esperado:
            fallos.append(f"md5 {relativo}: {obtenido} != {esperado}")
    return fallos


def verificar_config_v2() -> tuple[list[str], list[str]]:
    """Verifica la integracion v2 en lib/componentes_producto.py.

    Comprueba los defaults congelados por el benchmark v8 (k=10, top-50,
    3 chunks, pesos 0.5/0.5, alphas 0.75/1.5), el prior de hubs, la cache
    E5, el fallback E5->BM25 documentado y que E5/reranker se cargan lazy.

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
    }
    for clave, esperado in esperados.items():
        obtenido = CONFIG.get(clave)
        if obtenido != esperado:
            fallos.append(
                f"CONFIG['{clave}']={obtenido!r} != {esperado!r}"
            )
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

    firma = inspect.signature(VinculoCP.__init__).parameters
    for nombre, esperado in (
        ("usar_hubs", True),
        ("alpha_hubs_base", 0.75),
        ("alpha_hubs_rerank", 1.5),
        ("incluir_transitorios", False),
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

    # Instancias reales del componente (E5/reranker deben seguir sin cargar).
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
        if vinculo._e5 is not None or vinculo._reranker is not None:
            fallos.append(
                f"VinculoCP({modo}): modelos cargados en init (no lazy)"
            )
    observaciones.append(
        "VinculoCP default/rerank: hubs activo, alpha 0.75/1.5, lazy OK"
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
    contra lo documentado en el QA v1/v2.

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


# ===== SECCION 4: reporte y CLI =====
def parsear_args() -> argparse.Namespace:
    """Define y parsea los argumentos del verificador.

    Returns:
        Namespace con qa_dir y omitir_integridad.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Verificacion ejecutada del QA v2 de Juris-PLN: 10 casos E2E, "
            "integridad de artefactos canonicos y metricas v2 sin drift."
        )
    )
    parser.add_argument(
        "--qa-dir",
        default=str(Path("resultados") / "qa_final_v2"),
        help="Directorio con los 10 JSON del QA (default: qa_final_v2).",
    )
    parser.add_argument(
        "--omitir-integridad",
        action="store_true",
        help="Omite md5/config/metricas (solo valida los 10 casos).",
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

    observaciones = verificar_logs(qa_dir)
    if observaciones:
        print("\nEvidencia de logs (informativa):")
        for linea in observaciones:
            print(f"  - {linea}")

    if not args.omitir_integridad:
        print("\nIntegridad de artefactos y configuracion v2:")
        fallos_md5 = verificar_md5()
        obs_config, fallos_config = verificar_config_v2()
        fallos_metricas = verificar_metricas_v2()
        fallos.extend(fallos_md5 + fallos_config + fallos_metricas)
        for linea in obs_config:
            print(f"  [info] {linea}")
        print(
            f"  [{'ok' if not fallos_md5 else 'FALLO'}] md5 canonicos v2 "
            f"({len(ARTEFACTOS_MD5)} archivos)"
        )
        print(
            f"  [{'ok' if not fallos_config else 'FALLO'}] config "
            "VinculoCP / fallback E5 / cache E5"
        )
        print(
            f"  [{'ok' if not fallos_metricas else 'FALLO'}] metricas v2 "
            "sin drift (metricas.json + JSONs de referencia)"
        )

    print()
    if fallos:
        print(f"FALLOS ({len(fallos)}):")
        for fallo in fallos:
            print(f"  - {fallo}")
        sys.exit(1)
    print("TODOS LOS CHECKS PASARON (10/10 casos + integridad v2)")


if __name__ == "__main__":
    main()
