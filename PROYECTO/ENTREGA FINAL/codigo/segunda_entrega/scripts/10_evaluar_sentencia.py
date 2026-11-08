"""10_evaluar_sentencia.py
Fecha: 2026-09-09
Descripcion: CLI del producto final Juris-PLN (Fase N). Dada una sentencia
    (archivo de texto, id del corpus o texto literal) produce un JSON con:
    (1) validacion M0 de que es providencia de la Corte Constitucional,
    (2) area tematica M4 (clasificador LogReg sobre lemas enmascarados),
    (3) articulos de la Constitucion Politica vinculados (extraccion M2 +
    expansion M7 v2 BM25 por pasajes), (4) contexto jurisprudencial M5
    (sentencias citadas resueltas al corpus con PageRank y area) y un
    resumen breve. Salida JSON a stdout y opcionalmente a --salida RUTA.
Estado del arte aplicado: heuristica M0 (accuracy 1.0), NER regex M2,
    TF-IDF de lemas enmascarados + LogReg multinomial (macro-F1 0.287),
    BM25 por pasajes con max-pooling de chunks y bigramas LLR (F1@5 0.218),
    PageRank de la red de citas (M5).

ciclo v2 - Fase S1 (2026-09-10, plan/PLAN_v2_ciclo_mejora.md): el campo
    articulos_cp.recuperados_bm25 pasa de 5 a 10 resultados con sus scores
    (CONFIG['k_bm25']); el indice BM25 subyacente ya viene limpio de
    permanentes (VinculoCP de lib/componentes_producto.py, misma fase).
    El resumen usa len(recuperados) y no asume el largo 5.

ciclo v2 - Fase S2 (2026-09-10, plan/PLAN_v2_ciclo_mejora.md): lectura
    selectiva del corpus en texto_por_id con pyarrow.dataset (filter por
    id + column pruning a texto/texto_limpio) en vez de materializar las
    23.742 filas completas con pandas (cuello de botella del 75% del
    tiempo segun resultados/qa_perf/AUDITORIA_RENDIMIENTO.md: -3.7 s/eval,
    RAM 8 -> 2 GB). La salida JSON es identica; solo cambia la eficiencia.

ciclo v2 - Fase T (2026-09-10, plan/PLAN_v2_ciclo_mejora.md): flag CLI
    --rerank que activa el modo reranker de VinculoCP (BM25 top-20 ->
    cross-encoder BAAI/bge-reranker-v2-m3 -> top-10). Sin el flag el
    comportamiento es EXACTAMENTE el actual (default OFF). El JSON anade
    articulos_cp.reranker_usado (true/false) y el resumen menciona el
    reranker cuando se uso.

ciclo v2 - Fase X (producto v2, 2026-09-11, plan/PLAN_v2_ciclo_mejora.md):
    el producto adopta las mejoras ganadoras del ciclo v2 con evidencia en
    el benchmark v8. VinculoCP (lib/componentes_producto.py) aplica por
    defecto el prior de hubs CP (alpha=0.75 en la ruta BM25) y el flag
    --rerank pasa a la estrategia ganadora del v8
    (hibrido_rerank_chunks_top50: top-50 del hibrido BM25/E5 -> rerank por
    chunks -> prior alpha=1.5 -> top-10). El JSON anade
    articulos_cp.hubs_usado (bool) y articulos_cp.alpha_hubs (float
    efectivo), conserva reranker_usado y el resto de campos, y el resumen
    menciona el prior cuando se aplico.

ciclo v2.2 (producto v2.2, 2026-10-04, plan/PLAN_v2.2_adopcion.md): el
    flag --rerank usa los componentes del Sprint B (adopcion preparada,
    sujeta al gate B3 y a ADR de adopcion): el denso del hibrido pasa de
    E5 a B2 (modelos/b2_retriever) y el cross-encoder pasa de
    BAAI/bge-reranker-v2-m3 a B1 (modelos/b1_reranker); el hibrido, los
    top-50 candidatos, los chunks por candidato, el prior alpha=1.5 y
    k=10 no cambian. FALLBACK documentado a E5/bge con WARNING si faltan
    los modelos B1/B2 (error claro con estricto_b2_b1=True). Sin --rerank
    el comportamiento es EXACTAMENTE el de v2 (BM25 + hubs alpha=0.75) y
    el JSON conserva la estructura identica (sin campos nuevos). El
    resumen del modo rerank nombra los componentes efectivos de v2.2
    ("recuperados por hibrido BM25+B2+reranker (B1)"); el modo default
    conserva su texto de v2 intacto ("recuperados por BM25").
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import pyarrow.dataset as ds
from lib.componentes_producto import (
    ClasificadorArea,
    ContextoJurisprudencial,
    ExtractorEntidades,
    ValidadorEntrada,
    VinculoCP,
)

CONFIG = {
    "ruta_corpus": "data/corpus_sentencias.parquet",
    "k_bm25": 10,
    "max_citas_resumen": 5,
    "max_arts_resumen": 5,
}

ROOT = Path(__file__).resolve().parent.parent


# ===== SECCION 1: entrada de datos =====
def leer_archivo(ruta: str) -> str:
    """Lee un archivo de texto UTF-8 (con fallback latin-1).

    Args:
        ruta: Ruta al archivo .txt de la sentencia.

    Returns:
        Contenido del archivo como cadena.

    Raises:
        SystemExit: Si el archivo no existe o esta vacio (mensaje accionable).
    """
    archivo = Path(ruta)
    if not archivo.is_file():
        raise SystemExit(
            f"Error: el archivo no existe: {ruta}. Verifique la ruta o use "
            "--id/--texto con el contenido de la sentencia."
        )
    try:
        texto = archivo.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        texto = archivo.read_text(encoding="latin-1")
    if not texto.strip():
        raise SystemExit(
            f"Error: el archivo esta vacio: {ruta}. Proporcione una sentencia "
            "con contenido."
        )
    return texto


def texto_por_id(id_corpus: str) -> tuple[str, str]:
    """Resuelve un id del corpus a su texto crudo y limpio.

    ciclo v2 - Fase S2: lectura selectiva con pyarrow.dataset en vez de
    materializar el corpus completo con pandas (23.742 filas de texto
    completo, ~8 GB de RAM). El dataset es particionado hive por
    tipo/ano, pero el id no es clave de particion, por lo que el filtro
    ds.field('id') == id escanea solo las columnas id/texto/texto_limpio
    (column pruning) y materializa UNA fila.

    Args:
        id_corpus: Id numerico del documento en data/corpus_sentencias.parquet.

    Returns:
        Tupla (texto_crudo, texto_limpio) del documento.

    Raises:
        SystemExit: Si el id no existe en el corpus (mensaje accionable).
    """
    dataset = ds.dataset(
        ROOT / CONFIG["ruta_corpus"], format="parquet", partitioning="hive"
    )
    tabla = dataset.to_table(
        columns=["id", "texto", "texto_limpio"],
        filter=ds.field("id") == str(id_corpus),
    )
    if tabla.num_rows == 0:
        total = dataset.count_rows()
        raise SystemExit(
            f"Error: el id {id_corpus} no existe en el corpus "
            f"({total} sentencias). Ejecute el producto con un id "
            "valido o use --archivo/--texto."
        )
    return (
        str(tabla["texto"][0].as_py()),
        str(tabla["texto_limpio"][0].as_py()),
    )


# ===== SECCION 2: orquestacion del producto =====
def construir_salida(
    texto_crudo: str,
    texto_limpio: str,
    id_entrada: str | None,
    componentes: dict,
) -> dict:
    """Ejecuta los cinco componentes del producto sobre una sentencia.

    Args:
        texto_crudo: Texto crudo de la sentencia (M0/M2/M4).
        texto_limpio: Texto limpio del corpus (M7) o copia del crudo.
        id_entrada: Id del corpus si la entrada fue --id; None si no.
        componentes: Dict con validador, extractor, clasificador, vinculo y
            contexto ya inicializados.

    Returns:
        Dict JSON-serializable con la estructura del producto:
        es_sentencia, score_validacion, id_detectado, ponente, area,
        articulos_cp (con reranker_usado, hubs_usado y alpha_hubs,
        Fases T/X), sentencias_citadas y resumen.
    """
    validador: ValidadorEntrada = componentes["validador"]
    extractor: ExtractorEntidades = componentes["extractor"]
    clasificador: ClasificadorArea = componentes["clasificador"]
    vinculo: VinculoCP = componentes["vinculo"]
    contexto: ContextoJurisprudencial = componentes["contexto"]

    validacion = validador.validar(texto_crudo)
    es_sentencia = validacion["es_sentencia"]

    # Componentes que solo aplican a providencias validas.
    if es_sentencia:
        entidades = extractor.extraer(texto_crudo)
        canon_detectado = extractor.detectar_canon_encabezado(texto_crudo)
        id_detectado = (
            contexto.id_de_canon(canon_detectado) if canon_detectado else None
        )
        area = clasificador.clasificar(texto_crudo)
        extraidos = []
        for canon in entidades["arts_cp"]:
            detalle = vinculo.detalle_articulo(canon)
            if detalle is None:
                continue
            extraidos.append({"numero": canon, **detalle})
        recuperados = vinculo.recuperar(texto_limpio, CONFIG["k_bm25"])
        # M5 excluye self-loops: la autocita del encabezado (canon propio)
        # no es una sentencia citada por el documento.
        citas = [
            c for c in entidades["sents_citadas"] if c != canon_detectado
        ]
        citadas = contexto.resolver(citas)
        ponente = entidades["ponente"]
    else:
        area = None
        extraidos = []
        recuperados = []
        citadas = []
        ponente = None
        id_detectado = None

    # ciclo v2 - Fase X: el prior de hubs se aplico de verdad solo si la
    # sentencia es valida y su carga fue exitosa (VinculoCP.usar_hubs).
    hubs_usado = bool(es_sentencia and vinculo.usar_hubs)
    alpha_hubs = float(vinculo.alpha_hubs) if hubs_usado else 0.0

    salida = {
        "es_sentencia": es_sentencia,
        "score_validacion": validacion["score_validacion"],
        "id_detectado": id_detectado if id_entrada is None else str(id_entrada),
        "ponente": ponente,
        "area": area,
        "articulos_cp": {
            "extraidos": extraidos,
            "recuperados_bm25": recuperados,
            # ciclo v2 - Fase T: true solo si el modo reranker estaba
            # activo (y la sentencia es valida; en invalida no se recupera).
            "reranker_usado": bool(es_sentencia and vinculo.usar_reranker),
            # ciclo v2 - Fase X: prior de hubs efectivo (bool y peso).
            "hubs_usado": hubs_usado,
            "alpha_hubs": alpha_hubs,
        },
        "sentencias_citadas": citadas,
        "resumen": construir_resumen(
            es_sentencia, area, extraidos, recuperados, citadas,
            usar_reranker=vinculo.usar_reranker,
            usar_hubs=hubs_usado,
            alpha_hubs=alpha_hubs,
        ),
    }
    return salida


def construir_resumen(
    es_sentencia: bool,
    area: dict | None,
    extraidos: list[dict],
    recuperados: list[dict],
    citadas: list[dict],
    usar_reranker: bool = False,
    usar_hubs: bool = False,
    alpha_hubs: float = 0.0,
) -> str:
    """Redacta el resumen breve de una linea del resultado.

    Args:
        es_sentencia: Decision M0.
        area: Dict del clasificador M4 (top3/via) o None.
        extraidos: Articulos CP extraidos por M2 con detalle.
        recuperados: Articulos CP recuperados (M7; rerankeados y/o
            priorizados segun los flags de las Fases T/X).
        citadas: Sentencias citadas resueltas por M5.
        usar_reranker: Si True el resumen nombra la ruta rerank de v2.2
            (hibrido BM25+B2+reranker B1); False conserva el texto de v2.
        usar_hubs: Si True el resumen menciona el prior de hubs (Fase X).
        alpha_hubs: Peso efectivo del prior de hubs (Fase X).

    Returns:
        Cadena con el resumen legible del analisis.
    """
    if not es_sentencia:
        return (
            "El texto no cumple la heuristica de providencia de la Corte "
            "Constitucional (menos de 2/3 reglas); no se ejecutaron los "
            "componentes de analisis."
        )
    top1 = area["top3"][0] if area and area["top3"] else None
    parte_area = (
        f"area tematica {top1['area']} (prob {top1['prob']})"
        if top1
        else "sin area"
    )
    n_resueltas = sum(1 for c in citadas if c["resuelta"])
    # ciclo v2.2: el modo rerank nombra los componentes efectivos de la
    # ruta adoptada (denso B2 + cross-encoder B1); el default conserva el
    # texto de v2 sin cambios. El fallback E5/bge queda documentado en el
    # WARNING del log de VinculoCP (la estructura del JSON no cambia).
    via_recuperacion = (
        "recuperados por hibrido BM25+B2+reranker (B1)"
        if usar_reranker
        else "recuperados por BM25"
    )
    if usar_hubs:
        via_recuperacion += (
            f" con prior de hubs (alpha={alpha_hubs:g})"
        )
    return (
        f"Providencia valida de la Corte Constitucional; {parte_area}; "
        f"{len(extraidos)} articulos CP extraidos y {len(recuperados)} "
        f"{via_recuperacion}; {n_resueltas} de {len(citadas)} sentencias "
        "citadas resueltas al corpus."
    )


def inicializar_componentes(usar_reranker: bool = False) -> dict:
    """Inicializa los cinco componentes del producto (una vez por proceso).

    ciclo v2 - Fase X: VinculoCP se crea con los defaults ganadores del
    benchmark v8 (usar_hubs=True, alpha_hubs_base=0.75,
    alpha_hubs_rerank=1.5).

    Args:
        usar_reranker: Si True el VinculoCP se crea en modo reranker
            (ciclo v2.2: top-50 del hibrido BM25/B2 -> rerank por chunks
            con el cross-encoder de dominio B1 -> prior alpha=1.5 ->
            top-10; fallback E5/bge con aviso si faltan B2/B1); la carga
            del denso y del cross-encoder es lazy dentro de VinculoCP.

    Returns:
        Dict con validador, extractor, clasificador, vinculo y contexto.
    """
    return {
        "validador": ValidadorEntrada(),
        "extractor": ExtractorEntidades(),
        "clasificador": ClasificadorArea(),
        "vinculo": VinculoCP(usar_reranker=usar_reranker),
        "contexto": ContextoJurisprudencial(),
    }


# ===== SECCION 3: CLI =====
def parsear_args() -> argparse.Namespace:
    """Define y parsea los argumentos de linea de comandos.

    Returns:
        Namespace con archivo, id, texto y salida.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Evalua una sentencia de la Corte Constitucional: validacion M0, "
            "area tematica M4, articulos CP (M2+M7) y contexto M5."
        )
    )
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument(
        "--archivo", help="Ruta a un archivo .txt con la sentencia"
    )
    grupo.add_argument(
        "--id", dest="id_corpus", help="Id del documento en el corpus"
    )
    grupo.add_argument(
        "--texto", help="Texto literal de la sentencia entre comillas"
    )
    parser.add_argument(
        "--salida",
        help="Ruta opcional donde guardar el JSON de salida",
    )
    # ciclo v2 - Fase X: modo ganador del v8 (default OFF).
    parser.add_argument(
        "--rerank",
        action="store_true",
        default=False,
        help=(
            "Activa la ruta reranker del producto v2.2 (ciclo v2.2): "
            "candidatos top-50 del hibrido BM25/B2 -> rerank por chunks "
            "con el cross-encoder de dominio B1 (modelos/b1_reranker) -> "
            "prior de hubs alpha=1.5 -> top-10. Si faltan modelos/"
            "b2_retriever o modelos/b1_reranker se usa el fallback E5/bge "
            "con aviso. Sin el flag se usa la ruta BM25 con prior de hubs "
            "alpha=0.75 (intacta)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Orquesta el CLI: resuelve la entrada, evalua y emite el JSON."""
    inicio = time.time()
    args = parsear_args()
    try:
        if args.archivo:
            texto_crudo = leer_archivo(args.archivo)
            texto_limpio = texto_crudo
            id_entrada = None
        elif args.id_corpus:
            texto_crudo, texto_limpio = texto_por_id(args.id_corpus)
            id_entrada = str(args.id_corpus)
        else:
            texto_crudo = args.texto
            texto_limpio = texto_crudo
            id_entrada = None
        if not texto_crudo.strip():
            raise SystemExit(
                "Error: el texto de entrada esta vacio. Proporcione una "
                "sentencia con contenido."
            )

        componentes = inicializar_componentes(usar_reranker=args.rerank)
        salida = construir_salida(
            texto_crudo, texto_limpio, id_entrada, componentes
        )
        salida["tiempo_segundos"] = round(time.time() - inicio, 2)

        json_texto = json.dumps(salida, ensure_ascii=False, indent=2)
        print(json_texto)
        if args.salida:
            ruta = Path(args.salida)
            ruta.parent.mkdir(parents=True, exist_ok=True)
            ruta.write_text(json_texto + "\n", encoding="utf-8")
            print(
                f"[info] JSON guardado en {ruta}",
                file=sys.stderr,
            )
    except SystemExit as exc:
        # SystemExit llega desde leer_archivo/texto_por_id con el mensaje
        # accionable; se imprime una sola vez y se sale con codigo 1.
        if exc.code and isinstance(exc.code, str):
            print(f"[error] {exc.code}", file=sys.stderr)
        sys.exit(1)
    except Exception:  # noqa: BLE001 - main captura todo para traceback + exit 1
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
