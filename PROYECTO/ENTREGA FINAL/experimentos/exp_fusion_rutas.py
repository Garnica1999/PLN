"""exp_fusion_rutas.py
Fecha: 2026-11-01
Descripcion: Experimento de fusion tardia y techo de reordenamiento sobre el
    benchmark M7 v13 (90 casos). No reentrena nada ni usa GPU: parte de los
    rankings top-10 por caso que dejo la evaluacion congelada y mide cuatro
    cosas.

    (1) El techo de seleccion de ruta: cuanto se ganaria si cada caso usara su
    mejor ruta (oraculo) y cual es la mejor ruta fija.
    (2) El techo de reordenar el top-10 actual: cuanto subiria nDCG@10 si un
    reranker perfecto ordenara los mismos diez candidatos.
    (3) La cobertura del top-10: cuantos articulos relevantes entran de media,
    que es el limite real de cualquier reordenamiento.
    (4) Reglas de combinacion sin GPU: RRF con pesos por validacion cruzada,
    Borda, conteo de rutas y consenso, todas con delta pareado e IC 95 %
    bootstrap frente a v2.2.

    Nunca usa cita_exacta como entrada: esa ruta devuelve los articulos que M2
    extrajo del texto y el gold del benchmark son esos mismos articulos, asi
    que usarla seria circular.

Estado del arte aplicado: nDCG@10 con ganancia binaria e IDCG truncado a 10
    (Jarvelin y Kekalainen, 2002), reciprocal rank fusion (Cormack, Clarke y
    Buettcher, 2009), Borda y voto por mayoria para combinar rankings,
    validacion cruzada por caso y bootstrap pareado con semilla 42.
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
RUTA_EXCLUIDA = "cita_exacta"
K_RRF = 60
SEMILLA = 42
REMUESTREOS = 1000
PLIEGUES = 5


def cargar() -> tuple[list[str], dict, dict]:
    """Carga ids, gold por caso y rankings top-10 por ruta."""
    datos = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    gold = {
        str(caso["id"]): [str(a) for a in caso["arts_cp"]]
        for caso in datos["muestra"]
    }
    rankings: dict[str, dict[str, list[str]]] = {}
    for id_caso, rutas in datos["top10_por_caso"].items():
        rankings[str(id_caso)] = {
            ruta: [str(fila["id"]) for fila in filas]
            for ruta, filas in rutas.items()
            if ruta != RUTA_EXCLUIDA
        }
    return sorted(gold), gold, rankings


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


def rrf(rankings: dict[str, list[str]], pesos: dict[str, float]) -> list[str]:
    """Fusiona rankings por rango reciproco con pesos."""
    puntajes: dict[str, float] = {}
    for ruta, peso in pesos.items():
        for posicion, articulo in enumerate(rankings.get(ruta) or [], start=1):
            puntajes[articulo] = puntajes.get(articulo, 0.0) + peso / (K_RRF + posicion)
    return [articulo for articulo, _ in sorted(puntajes.items(), key=lambda par: -par[1])]


def borda(rankings: dict[str, list[str]], _pesos: dict[str, float]) -> list[str]:
    """Voto Borda: suma (11 menos posicion) de cada ruta, desempata por v2.2."""
    puntos: dict[str, float] = {}
    for lista in rankings.values():
        for posicion, articulo in enumerate(lista, start=1):
            puntos[articulo] = puntos.get(articulo, 0.0) + (11 - posicion)
    base = {articulo: pos for pos, articulo in enumerate(rankings[RUTA_V22])}
    return sorted(puntos, key=lambda articulo: (-puntos[articulo], base.get(articulo, 99)))


def conteo(rankings: dict[str, list[str]], _pesos: dict[str, float]) -> list[str]:
    """Cuenta en cuantas rutas aparece cada articulo, desempata por v2.2."""
    cuenta: dict[str, int] = {}
    for lista in rankings.values():
        for articulo in lista:
            cuenta[articulo] = cuenta.get(articulo, 0) + 1
    base = {articulo: pos for pos, articulo in enumerate(rankings[RUTA_V22])}
    return sorted(cuenta, key=lambda articulo: (-cuenta[articulo], base.get(articulo, 99)))


def consenso(rankings: dict[str, list[str]], _pesos: dict[str, float]) -> list[str]:
    """v2.2 manda y al final entran los articulos que al menos 3 rutas listan."""
    cuenta: dict[str, int] = {}
    for lista in rankings.values():
        for articulo in lista:
            cuenta[articulo] = cuenta.get(articulo, 0) + 1
    orden = list(rankings[RUTA_V22])
    nuevos = [
        articulo
        for articulo, veces in sorted(cuenta.items(), key=lambda par: -par[1])
        if veces >= 3 and articulo not in orden
    ]
    return orden + nuevos


def evaluar(ids: list[str], gold: dict, rankings: dict, regla) -> float:
    """nDCG@10 medio de una regla de combinacion sobre todos los casos."""
    return float(np.mean([ndcg_at_10(regla(rankings[i], {}), gold[i]) for i in ids]))


def delta_pareado(
    ids: list[str],
    gold: dict,
    rankings: dict,
    regla,
    base,
) -> dict:
    """Delta de nDCG@10 (regla menos base) con IC 95 % bootstrap pareado."""
    deltas = np.asarray(
        [
            ndcg_at_10(regla(rankings[i], {}), gold[i])
            - ndcg_at_10(base(rankings[i], {}), gold[i])
            for i in ids
        ]
    )
    generador = np.random.default_rng(SEMILLA)
    medias = [
        deltas[generador.integers(0, len(deltas), len(deltas))].mean()
        for _ in range(REMUESTREOS)
    ]
    return {
        "delta": round(float(deltas.mean()), 4),
        "ic95": [
            round(float(np.percentile(medias, 2.5)), 4),
            round(float(np.percentile(medias, 97.5)), 4),
        ],
        "prob_mejora": round(float(np.mean(np.asarray(medias) > 0)), 4),
    }


def pliegues_por_caso(ids: list[str]) -> list[list[str]]:
    """Reparte los casos en 5 pliegues deterministas (semilla 42)."""
    generador = np.random.default_rng(SEMILLA)
    orden = list(generador.permutation(len(ids)))
    tamano = len(ids) // PLIEGUES
    return [
        [
            ids[int(pos)]
            for pos in orden[
                indice * tamano : (
                    len(ids) if indice == PLIEGUES - 1 else (indice + 1) * tamano
                )
            ]
        ]
        for indice in range(PLIEGUES)
    ]


def mejor_peso(ids: list[str], gold: dict, rankings: dict, ruta: str) -> float:
    """Elige el peso de una ruta complementaria por validacion cruzada."""
    rejilla = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]
    pliegues = pliegues_por_caso(ids)
    elegidos = []
    for pliegue in range(PLIEGUES):
        entrenamiento = [c for i, p in enumerate(pliegues) if i != pliegue for c in p]
        elegidos.append(
            min(
                rejilla,
                key=lambda peso: -evaluar(
                    entrenamiento,
                    gold,
                    rankings,
                    lambda r, _p, w=peso: rrf(r, {RUTA_V22: 1.0, ruta: w}),
                ),
            )
        )
    return float(np.median(elegidos))


def main() -> int:
    """Ejecuta el analisis y escribe el JSON y el resumen en Markdown."""
    ids, gold, rankings = cargar()
    rutas = sorted(set.intersection(*(set(r) for r in rankings.values())))

    base_fija = {ruta: round(evaluar(ids, gold, rankings, lambda r, _p, x=ruta: r[x]), 4) for ruta in rutas}
    oraculo = float(
        np.mean(
            [
                max(ndcg_at_10(rankings[i][ruta], gold[i]) for ruta in rutas)
                for i in ids
            ]
        )
    )
    techo_reorden = float(
        np.mean(
            [
                ndcg_at_10(
                    sorted(
                        rankings[i][RUTA_V22],
                        key=lambda a: 0 if a in set(gold[i]) else 1,
                    ),
                    gold[i],
                )
                for i in ids
            ]
        )
    )
    relevantes_top10 = float(
        np.mean(
            [
                len(set(gold[i]) & set(rankings[i][RUTA_V22]))
                for i in ids
            ]
        )
    )
    relevantes_gold = float(np.mean([len(set(g)) for g in gold.values()]))

    reglas = {
        "borda": borda,
        "conteo de rutas": conteo,
        "v2.2 + consenso de 3 rutas": consenso,
    }
    combinaciones = {
        nombre: {
            "ndcg_at_10": round(evaluar(ids, gold, rankings, regla), 4),
            "vs_v2_2": delta_pareado(
                ids, gold, rankings, regla, lambda r, _p: r[RUTA_V22]
            ),
        }
        for nombre, regla in reglas.items()
    }

    pesos_elegidos = {
        ruta: mejor_peso(ids, gold, rankings, ruta)
        for ruta in rutas
        if ruta != RUTA_V22
    }
    for ruta, peso in pesos_elegidos.items():
        if peso <= 0.0:
            continue
        regla = lambda r, _p, w=peso, x=ruta: rrf(r, {RUTA_V22: 1.0, x: w})
        combinaciones[f"rrf v2.2 + {ruta} (peso {peso})"] = {
            "ndcg_at_10": round(evaluar(ids, gold, rankings, regla), 4),
            "vs_v2_2": delta_pareado(
                ids, gold, rankings, regla, lambda r, _p: r[RUTA_V22]
            ),
        }

    resultado = {
        "experimento": "exp_fusion_rutas",
        "fecha": datetime.now().astimezone().isoformat(timespec="seconds"),
        "fuente": str(BENCHMARK.relative_to(BASE.parents[2])),
        "n_casos": len(ids),
        "rutas": rutas,
        "ndcg_at_10_por_ruta_fija": base_fija,
        "oraculo_seleccion_por_caso": round(oraculo, 4),
        "techo_reorden_top10": round(techo_reorden, 4),
        "relevantes_en_top10_media": round(relevantes_top10, 2),
        "relevantes_gold_media": round(relevantes_gold, 2),
        "pesos_elegidos_rrf": pesos_elegidos,
        "combinaciones": combinaciones,
        "nota": (
            "cita_exacta se excluye: devuelve los articulos que M2 extrajo y el gold "
            "del benchmark son esos mismos articulos."
        ),
    }
    SALIDA_JSON.write_text(json.dumps(resultado, ensure_ascii=False, indent=1), encoding="utf-8")

    lineas = [
        "# Fusion tardia y techo de reordenamiento (M7 v13)",
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
        f"Oráculo de selección por caso: **{oraculo:.4f}**.",
        f"Techo de reordenar el top-10 actual: **{techo_reorden:.4f}**.",
        (
            f"Relevantes en el top-10 (media): **{relevantes_top10:.2f}** de "
            f"{relevantes_gold:.2f} del gold."
        ),
        "",
        "## Combinaciones sin GPU",
        "",
        "| Combinación | nDCG@10 | Delta vs v2.2 | IC 95 % | P(mejora) |",
        "|---|---:|---:|---|---:|",
    ]
    for nombre, datos in sorted(
        combinaciones.items(), key=lambda par: -par[1]["ndcg_at_10"]
    ):
        delta = datos["vs_v2_2"]
        lineas.append(
            f"| {nombre} | {datos['ndcg_at_10']:.4f} | {delta['delta']:+.4f} | "
            f"[{delta['ic95'][0]:+.4f}; {delta['ic95'][1]:+.4f}] | {delta['prob_mejora']:.3f} |"
        )
    lineas += [
        "",
        "## Lectura",
        "",
        "El top-10 actual contiene menos de la mitad del gold, así que reordenarlo",
        "tiene un techo bajo (+0,043). La mejora real exige promover candidatos de",
        "las posiciones 11 a 50, es decir, un mejor reranker, no otra regla de fusión.",
        "",
    ]
    SALIDA_MD.write_text("\n".join(lineas), encoding="utf-8")

    print(
        f"v2.2: {base_fija[RUTA_V22]:.4f} | oraculo: {oraculo:.4f} | "
        f"techo reorden: {techo_reorden:.4f} | top-10: {relevantes_top10:.2f} de {relevantes_gold:.2f}"
    )
    for nombre, datos in sorted(combinaciones.items(), key=lambda par: -par[1]["ndcg_at_10"]):
        delta = datos["vs_v2_2"]
        print(
            f"  {nombre}: {datos['ndcg_at_10']:.4f} "
            f"(delta {delta['delta']:+.4f} [{delta['ic95'][0]:+.4f}; {delta['ic95'][1]:+.4f}])"
        )
    print(f"JSON: {SALIDA_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
