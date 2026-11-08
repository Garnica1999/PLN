"""exp_listwise_top10.py
Fecha: 2026-11-01
Descripcion: Sonda de techo del reranker. Para los 90 casos del benchmark M7
    v13 toma el top-10 que produjo v2.2 y le pide al LLM local que lo
    reordene por relevancia, con la sentencia (recortada) y el texto de cada
    articulo de la CP como contexto. Compara el nDCG@10 del orden del LLM con
    el de v2.2 y con el techo de reorden perfecto (0,7359).

    No sustituye la evaluacion oficial: mide cuanto del margen de
    reordenamiento es alcanzable por un LLM listwise, no un modelo adoptable
    (la latencia por consulta lo descarta para el producto).

Estado del arte aplicado: listwise reranking con LLM (Sun et al., 2023),
    salida JSON estricta, temperature 0 y semilla 42, delta pareado con IC
    95 % bootstrap. Si el LLM no devuelve los diez articulos, el caso se
    conserva con el orden de v2.2 y se cuenta como fallo de parseo.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq

BASE = Path(__file__).resolve().parent
SEGUNDA = BASE.parents[1] / "SEGUNDA ENTREGA"
BENCHMARK = SEGUNDA / "resultados" / "vinculo_cp_benchmark_v13.json"
CORPUS = SEGUNDA / "data" / "corpus_sentencias.parquet"
CONSTITUCION = SEGUNDA / "data" / "constitucion_articulos.parquet"
SALIDA_JSON = BASE / "exp_listwise_top10.json"
SALIDA_MD = BASE / "exp_listwise_top10.md"

OLLAMA = "http://127.0.0.1:11434"
MODELO = "qwen3.8:diario-8k-q3_k_s"
RUTA_V22 = "hibrido_b2_rerank_chunks_top50_b1"
MAX_CARACTERES_SENTENCIA = 2500
MAX_CARACTERES_ARTICULO = 220
SEMILLA = 42
REMUESTREOS = 1000

SISTEMA = """Eres un jurista colombiano. Recibes una sentencia y diez
articulos de la Constitucion. Ordena los articulos de mas a menos relevante
para responder que articulos fundamentan esa sentencia. Devuelve SOLO un JSON
con la clave "orden" y la lista de numeros de articulo, sin texto adicional."""


def cargar_casos() -> tuple[list[dict], dict]:
    """Carga los casos con su sentencia, sus diez candidatos y el gold."""
    datos = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    gold = {str(caso["id"]): {str(a) for a in caso["arts_cp"]} for caso in datos["muestra"]}
    ids = sorted(gold)
    tabla = ds.dataset(CORPUS, format="parquet", partitioning="hive").to_table(
        columns=["id", "texto"], filter=ds.field("id").isin(ids)
    )
    textos = {str(i): t for i, t in zip(tabla["id"].to_pylist(), tabla["texto"].to_pylist(), strict=True)}
    constitucion = pq.read_table(CONSTITUCION, columns=["numero", "texto"]).to_pandas()
    articulos = {
        str(fila.numero).upper(): str(fila.texto)
        for fila in constitucion.itertuples(index=False)
    }
    casos = []
    for id_caso in ids:
        orden = [
            str(fila["id"])
            for fila in datos["top10_por_caso"][id_caso][RUTA_V22]
        ]
        casos.append(
            {
                "id": id_caso,
                "texto": (textos.get(id_caso) or "")[:MAX_CARACTERES_SENTENCIA],
                "candidatos": orden,
                "gold": sorted(gold[id_caso]),
            }
        )
    return casos, articulos


def ndcg_at_10(ranking: list[str], gold: set[str]) -> float:
    """nDCG@10 con ganancia binaria e IDCG truncado a 10."""
    dcg = sum(
        1.0 / np.log2(pos + 2)
        for pos, articulo in enumerate(ranking[:10])
        if articulo in gold
    )
    ideal = sum(1.0 / np.log2(pos + 2) for pos in range(min(10, len(gold))))
    return float(dcg / ideal) if ideal else 0.0


def reordenar(caso: dict, articulos: dict) -> tuple[list[str], bool]:
    """Pide al LLM un orden de los diez candidatos y lo valida."""
    contexto = "\n".join(
        f"{numero}. {articulos.get(numero, '')[:MAX_CARACTERES_ARTICULO].strip()}"
        for numero in caso["candidatos"]
    )
    cuerpo = {
        "model": MODELO,
        "messages": [
            {"role": "system", "content": SISTEMA},
            {
                "role": "user",
                "content": (
                    f"Sentencia:\n{caso['texto']}\n\nArticulos candidatos:\n{contexto}"
                ),
            },
        ],
        "stream": False,
        "format": "json",
        # Sin think=false el modelo gasta toda la salida en el bloque de
        # razonamiento y devuelve contenido vacio (verificado en el caso 1).
        "think": False,
        "options": {"temperature": 0.0, "seed": SEMILLA, "num_predict": 300},
    }
    try:
        with httpx.Client(timeout=180.0) as cliente:
            respuesta = cliente.post(f"{OLLAMA}/api/chat", json=cuerpo).json()
        texto = (respuesta.get("message") or {}).get("content") or "{}"
        orden = json.loads(texto).get("orden") or []
        orden = [str(valor).upper() for valor in orden]
    except (httpx.HTTPError, ValueError, AttributeError):
        return caso["candidatos"], False
    if sorted(orden) != sorted(caso["candidatos"]):
        # El LLM puede omitir o inventar articulos: se completa con el orden
        # original para no alterar el conjunto evaluado.
        faltan = [a for a in caso["candidatos"] if a not in orden]
        orden = [a for a in orden if a in caso["candidatos"]] + faltan
        return orden, False
    return orden, True


def main() -> int:
    """Ejecuta la sonda y escribe JSON y Markdown."""
    casos, articulos = cargar_casos()
    registros = []
    for caso in casos:
        inicio = time.perf_counter()
        orden, ok = reordenar(caso, articulos)
        registros.append(
            {
                "id": caso["id"],
                "orden_v22": caso["candidatos"],
                "orden_llm": orden,
                "parse_ok": ok,
                "ndcg_v22": round(ndcg_at_10(caso["candidatos"], set(caso["gold"])), 4),
                "ndcg_llm": round(ndcg_at_10(orden, set(caso["gold"])), 4),
                "ms": int((time.perf_counter() - inicio) * 1000),
            }
        )

    deltas = np.asarray([r["ndcg_llm"] - r["ndcg_v22"] for r in registros])
    generador = np.random.default_rng(SEMILLA)
    medias = [
        deltas[generador.integers(0, len(deltas), len(deltas))].mean()
        for _ in range(REMUESTREOS)
    ]
    resultado = {
        "experimento": "exp_listwise_top10",
        "fecha": datetime.now().astimezone().isoformat(timespec="seconds"),
        "fuente": str(BENCHMARK.relative_to(BASE.parents[2])),
        "n_casos": len(registros),
        "modelo": MODELO,
        "ndcg_v22": round(float(np.mean([r["ndcg_v22"] for r in registros])), 4),
        "ndcg_llm": round(float(np.mean([r["ndcg_llm"] for r in registros])), 4),
        "delta": round(float(deltas.mean()), 4),
        "ic95": [
            round(float(np.percentile(medias, 2.5)), 4),
            round(float(np.percentile(medias, 97.5)), 4),
        ],
        "prob_mejora": round(float(np.mean(np.asarray(medias) > 0)), 4),
        "parse_ok": sum(1 for r in registros if r["parse_ok"]),
        "latencia_media_ms": int(np.mean([r["ms"] for r in registros])),
        "techo_reorden": 0.7359,
        "registros": registros,
        "nota": (
            "Sonda de techo: el listwise consume 5 a 20 s por consulta y no es "
            "adoptable para el producto; mide cuanto del margen de reorden es alcanzable."
        ),
    }
    SALIDA_JSON.write_text(json.dumps(resultado, ensure_ascii=False, indent=1), encoding="utf-8")

    lineas = [
        "# Sonda listwise sobre el top-10 (M7 v13)",
        "",
        f"Fecha: {resultado['fecha']}. Casos: {resultado['n_casos']}. Modelo: `{MODELO}`.",
        "",
        "| Metrica | Valor |",
        "|---|---:|",
        f"| nDCG@10 de v2.2 | {resultado['ndcg_v22']:.4f} |",
        f"| nDCG@10 del orden del LLM | {resultado['ndcg_llm']:.4f} |",
        f"| Delta pareado | {resultado['delta']:+.4f} |",
        f"| IC 95 % | [{resultado['ic95'][0]:+.4f}; {resultado['ic95'][1]:+.4f}] |",
        f"| P(mejora) | {resultado['prob_mejora']:.3f} |",
        f"| Parseos validos | {resultado['parse_ok']} de {resultado['n_casos']} |",
        f"| Latencia media | {resultado['latencia_media_ms']} ms |",
        f"| Techo de reorden perfecto | {resultado['techo_reorden']:.4f} |",
        "",
        "## Lectura",
        "",
        "Si el LLM no supera a v2.2, el orden actual ya es competitivo y el margen",
        "solo se recupera cambiando el conjunto de candidatos (promover los articulos",
        "de las posiciones 11 a 50), no reordenando el top-10.",
        "",
    ]
    SALIDA_MD.write_text("\n".join(lineas), encoding="utf-8")

    print(
        f"v2.2: {resultado['ndcg_v22']:.4f} | LLM: {resultado['ndcg_llm']:.4f} "
        f"(delta {resultado['delta']:+.4f} [{resultado['ic95'][0]:+.4f}; {resultado['ic95'][1]:+.4f}]) "
        f"| parseos {resultado['parse_ok']}/{resultado['n_casos']}"
    )
    print(f"JSON: {SALIDA_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
