"""exp_fusion_rutas.py
Fecha: 2026-11-01
Descripcion: Experimento de fusion tardia sobre las rutas del benchmark M7
    v13 (90 casos). No reentrena nada ni usa GPU: parte de los rankings
    top-10 por caso que dejo la evaluacion congelada y mide tres cosas.

    (1) El techo de seleccion de ruta: cuanto se ganaria si cada caso usara
    su mejor ruta (oraculo) y cual es la mejor ruta fija.
    (2) La fusion por rango reciproco (RRF) de v2.2 con rutas
    complementarias, con pesos ajustados por validacion cruzada de 5
    pliegues para no elegir sobre el mismo dato que se reporta.
    (3) El delta pareado con IC 95 % bootstrap frente a v2.2, que es el gate
    de adopcion de ADR-16.

    Nunca usa cita_exacta como entrada de la fusion: esa ruta devuelve los
    articulos que M2 extrajo del texto y el gold del benchmark son esos
    mismos articulos, asi que fusionarla seria circular.

Estado del arte aplicado: nDCG@10 con ganancia binaria e IDCG truncado a 10
    (Jarvelin y Kekalainen, 2002), reciprocal rank fusion (Cormack, Clarke y
    Buettcher, 2009), validacion cruzada por caso y bootstrap pareado con
    semilla 42.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
SEGUNDA = BASE.parents[1] / "SEGUNDA ENTREGA"
BENCHMARK = SEGUNDA / "resultados" / "vinculo_cp_benchmark_v13.json"
SALIDA_JSON = BASE / "exp_fusion_rutas.json"
SALIDA_MD = BASE / "exp_fusion_rutas.md"

RUTA_V22 = "hibrido_b2_rerank_chunks_top50_b1"
RUTA_V12 = "hibrido_rerank_chunks_top50"
RUTA_EXCLUIDA = "cita_exacta"
K_RRF = 60
SEMILLA = 42
REMUESTREOS = 1000
PLIEGUES = 5


def cargar() -> tuple[list[str], dict[str, dict[str, float]], dict[str, dict[str, list[str]]]]:
    """Carga ids, gold por caso y rankings top-10 por ruta.

    Returns:
        Tupla (ids, gold, rankings): ids de los 90 casos, arts_cp relevantes
        por caso y ranking (ruta -> ids ordenados) por caso.
    """
    datos = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    gold = {str(caso["id"]): [str(a) for a in caso["arts_cp"]] for caso in datos["muestra"]}
    rankings: dict[str, dict[str, list[str]]] = {}
    for id_caso, rutas in datos["top10_por_caso"].items():
        rankings[str(id_caso)] = {
            ruta: [str(fila["id"]) for fila in filas]
            for ruta, filas in rutas.items()
            if ruta != RUTA_EXCLUIDA
        }
    ids = sorted(gold)
    return ids, gold, rankings


def ndcg_at_10(ranking: list[str], gold: list[str]) -> float:
    """nDCG@10 con ganancia binaria e IDCG truncado a 10."""
    relevantes = set(gold)
    dcg = sum(
        1.0 / np.log2(pos + 2)
        for pos, articulo in enumerate(ranking[:10])
        if articulo in relevantes
    )
    ideal = sum(1.0 / np.log2(pos + 2) for pos in range(min(10, len(relevantes))))
    return float(dcg / ideal) if ideal else 0.0


def rrf(
    rankings: dict[str, list[str]], pesos: dict[str, float], k: int = K_RRF
) -> list[str]:
    """Fusiona rankings por rango reciproco con pesos por ruta."""
    puntajes: dict[str, float] = {}
    for ruta, peso in pesos.items():
        for posicion, articulo in enumerate(rankings.get(ruta) or [], start=1):
            puntajes[articulo] = puntajes.get(articulo, 0.0) + peso / (k + posicion)
    return [articulo for articulo, _ in sorted(puntajes.items(), key=lambda par: -par[1])]


def evaluar(ids: list[str], gold: dict, rankings: dict, pesos: dict[str, float]) -> float:
    """nDCG@10 medio de una fusion sobre todos los casos."""
    valores = []
    for id_caso in ids:
        orden = rrf(rankings[id_caso], pesos)
        valores.append(ndcg_at_10(orden, gold[id_caso]))
    return float(np.mean(valores))


def delta_pareado(
    ids: list[str], gold: dict, rankings: dict, pesos: dict[str, float], base: str
) -> dict:
    """Delta de nDCG@10 (fusion menos base) con IC 95 % bootstrap pareado."""
    deltas = []
    for id_caso in ids:
        fusion = ndcg_at_10(rrf(rankings[id_caso], pesos), gold[id_caso])
        referencia = ndcg_at_10(rankings[id_caso][base], gold[id_caso])
        deltas.append(fusion - referencia)
    deltas = np.asarray(deltas)
    generador = np.random.default_rng(SEMILLA)
    medias = [
        deltas[generador.integers(0, len(deltas), len(deltas))].mean()
        for _ in range(REMUESTREOS)
    ]
    return {
        "delta": round(float(deltas.mean()), 4),
        "ic95": [round(float(np.percentile(medias, 2.5)), 4), round(float(np.percentile(medias, 97.5)), 4)],
        "prob_mejora": round(float(np.mean(np.asarray(medias) > 0)), 4),
    }


def pliegues_por_caso(ids: list[str]) -> list[list[str]]:
    """Reparte los casos en 5 pliegues deterministas (semilla 42)."""
    generador = np.random.default_rng(SEMILLA)
    orden = list(generador.permutation(len(ids)))
    tamano = len(ids) // PLIEGUES
    pliegues = []
    for indice in range(PLIEGUES):
        desde = indice * tamano
        hasta = len(ids) if indice == PLIEGUES - 1 else (indice + 1) * tamano
        pliegues.append([ids[pos] for pos in orden[desde:hasta]])
    return pliegues


def mejor_peso(ids: list[str], gold: dict, rankings: dict, ruta: str) -> float:
    """Elige el peso de una ruta complementaria por validacion cruzada."""
    rejilla = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]
    pliegues = pliegues_por_caso(ids)
    elegidos = []
    for pliegue in range(PLIEGUES):
        entrenamiento = [c for i, p in enumerate(pliegues) if i != pliegue for c in p]
        mejores = sorted(
            rejilla,
            key=lambda peso: -evaluar(
                entrenamiento, gold, rankings, {RUTA_V22: 1.0, ruta: peso}
            ),
        )
        elegidos.append(mejores[0])
    return float(np.median(elegidos))


def main() -> int:
    """Ejecuta el analisis y escribe el JSON y el resumen en Markdown."""
    ids, gold, rankings = cargar()
    rutas = sorted(set.intersection(*(set(r) for r in rankings.values())))

    base_fija = {ruta: round(evaluar(ids, gold, rankings, {ruta: 1.0}), 4) for ruta in rutas}
    oraculo_por_caso = float(
        np.mean(
            [
                max(
                    ndcg_at_10(rankings[id_caso][ruta], gold[id_caso])
                    for ruta in rutas
                )
                for id_caso in ids
            ]
        )
    )
    cobertura_union = float(
        np.mean(
            [
                len(set(gold[id_caso]) & {a for ruta in rutas for a in rankings[id_caso][ruta]})
                / len(gold[id_caso])
                for id_caso in ids
            ]
        )
    )

    candidatas = [ruta for ruta in rutas if ruta != RUTA_V22]
    pesos_elegidos = {
        ruta: mejor_peso(ids, gold, rankings, ruta) for ruta in candidatas
    }
    fusiones = {}
    for ruta, peso in pesos_elegidos.items():
        if peso <= 0.0:
            continue
        pesos = {RUTA_V22: 1.0, ruta: peso}
        fusiones[f"v2.2 + {ruta} (peso {peso})"] = {
            "pesos": pesos,
            "ndcg_at_10": round(evaluar(ids, gold, rankings, pesos), 4),
            "vs_v2_2": delta_pareado(ids, gold, rankings, pesos, RUTA_V22),
        }

    pesos_todas = {RUTA_V22: 1.0, **{ruta: 0.25 for ruta in candidatas}}
    fusiones["v2.2 + todas las rutas (0.25)"] = {
        "pesos": pesos_todas,
        "ndcg_at_10": round(evaluar(ids, gold, rankings, pesos_todas), 4),
        "vs_v2_2": delta_pareado(ids, gold, rankings, pesos_todas, RUTA_V22),
    }
    pesos_mejores = {
        RUTA_V22: 1.0,
        **{ruta: peso for ruta, peso in pesos_elegidos.items() if peso > 0.0},
    }
    if len(pesos_mejores) > 1:
        fusiones["v2.2 + rutas con peso elegido"] = {
            "pesos": pesos_mejores,
            "ndcg_at_10": round(evaluar(ids, gold, rankings, pesos_mejores), 4),
            "vs_v2_2": delta_pareado(ids, gold, rankings, pesos_mejores, RUTA_V22),
        }

    resultado = {
        "experimento": "exp_fusion_rutas",
        "fecha": datetime.now().astimezone().isoformat(timespec="seconds"),
        "fuente": str(BENCHMARK.relative_to(BASE.parents[2])),
        "n_casos": len(ids),
        "rutas": rutas,
        "ndcg_at_10_por_ruta_fija": base_fija,
        "oraculo_seleccion_por_caso": round(oraculo_por_caso, 4),
        "cobertura_union_top10": round(cobertura_union, 4),
        "pesos_elegidos": pesos_elegidos,
        "fusiones": fusiones,
        "nota": (
            "cita_exacta se excluye de la fusion: devuelve los articulos que M2 extrajo "
            "y el gold del benchmark son esos mismos articulos."
        ),
    }
    SALIDA_JSON.write_text(json.dumps(resultado, ensure_ascii=False, indent=1), encoding="utf-8")

    lineas = [
        "# Fusion tardia de rutas (M7 v13)",
        "",
        f"Fecha: {resultado['fecha']}. Casos: {len(ids)}. Fuente: `{resultado['fuente']}`.",
        "",
        "## nDCG@10 por ruta fija",
        "",
        "| Ruta | nDCG@10 |",
        "|---|---:|",
    ]
    for ruta, valor in sorted(base_fija.items(), key=lambda par: -par[1]):
        marcador = " (v2.2)" if ruta == RUTA_V22 else ""
        lineas.append(f"| {ruta}{marcador} | {valor:.4f} |")
    lineas += [
        "",
        f"Oraculo de seleccion por caso: **{oraculo_por_caso:.4f}**.",
        f"Cobertura de la union de los top-10 de todas las rutas: **{cobertura_union:.4f}**.",
        "",
        "## Fusiones",
        "",
        "| Fusion | nDCG@10 | Delta vs v2.2 | IC 95 % | P(mejora) |",
        "|---|---:|---:|---|---:|",
    ]
    for nombre, datos_fusion in fusiones.items():
        delta = datos_fusion["vs_v2_2"]
        lineas.append(
            f"| {nombre} | {datos_fusion['ndcg_at_10']:.4f} | {delta['delta']:+.4f} | "
            f"[{delta['ic95'][0]:+.4f}; {delta['ic95'][1]:+.4f}] | {delta['prob_mejora']:.3f} |"
        )
    lineas += [
        "",
        "## Lectura",
        "",
        "Una fusion se adopta solo si el delta pareado tiene el IC 95 % por encima de cero",
        "y la mejora es consistente. El oraculo marca el techo de elegir ruta por caso.",
        "",
    ]
    SALIDA_MD.write_text("\n".join(lineas), encoding="utf-8")

    print(f"v2.2: {base_fija[RUTA_V22]:.4f} | oraculo: {oraculo_por_caso:.4f} | union: {cobertura_union:.4f}")
    for nombre, datos_fusion in fusiones.items():
        delta = datos_fusion["vs_v2_2"]
        print(
            f"  {nombre}: {datos_fusion['ndcg_at_10']:.4f} "
            f"(delta {delta['delta']:+.4f} [{delta['ic95'][0]:+.4f}; {delta['ic95'][1]:+.4f}])"
        )
    print(f"JSON: {SALIDA_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
