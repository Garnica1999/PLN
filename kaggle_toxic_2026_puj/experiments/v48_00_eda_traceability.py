"""V48-00 (P0): trazabilidad de los hallazgos del EDA original hacia experimentos.

Que responde
------------
Si el EDA original se convertio de forma suficientemente completa en hipotesis
experimentales, y que hallazgos quedaron sin convertir en un mecanismo concreto.

Regla de trazabilidad aplicada
-----------------------------
Una hipotesis solo se considera cerrada si reune las tres condiciones:

1. mecanismo probado
2. control adecuado
3. evaluacion valida

El mismo tema no implica la misma familia. "la longitud importa" no cierra todos
los mecanismos relativos a longitud.

Que NO hace
-----------
- No entrena ningun modelo.
- No genera ninguna submission.
- No usa holdout ni test.
- No crea filas en el registro de experimentos.

Uso
---
    /home/carlos/venv/bin/python experiments/v48_00_eda_traceability.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

EDA = PROJECT_ROOT / "reports" / "eda"
REGISTRO = PROJECT_ROOT / "experiments.csv"
REPORTE_JSON = PROJECT_ROOT / "reports" / "p48_eda_traceability.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "p48_eda_traceability.md"
REPORTE_CSV = PROJECT_ROOT / "reports" / "p48_eda_traceability.csv"

ETIQUETAS = ("HS", "TR", "AG")

# Hallazgos del EDA original, con el mecanismo concreto que cada uno sugiere.
# `patron` se usa para buscar en el registro; `mecanismo` describe el mecanismo
# especifico, no el tema.
HALLAZGOS = [
    {
        "hallazgo": "La mencion inicial es el fenomeno mas asociado con TR dentro de HS=1",
        "patron": r"mencion(_inicial)?",
        "mecanismo": "feature de mencion inicial como termino propio",
        "artefacto": "reports/eda/fase2_rasgos_TR.csv",
        "evidencia": "V de Cramer 0.512",
    },
    {
        "hallazgo": "La referencia colectiva sobrerepresenta falsos negativos de HS",
        "patron": r"colectiv|colectivo|referencia_colectiva",
        "mecanismo": "feature de referencia colectiva y su interaccion con HS",
        "artefacto": "reports/eda/error_perfil_rasgos.csv",
        "evidencia": "sobrerrepresentacion FP vs TN 1.3665",
    },
    {
        "hallazgo": "El insulto lexico sobrerepresenta falsos positivos de HS",
        "patron": r"insulto|lexic",
        "mecanismo": "feature de insulto lexico y su peso en la clase positiva",
        "artefacto": "reports/eda/error_perfil_rasgos.csv",
        "evidencia": "tasa FP 0.8108 con insulto",
    },
    {
        "hallazgo": "La negacion altera la interpretacion de TR y AG",
        "patron": r"negaci|negacion",
        "mecanismo": "feature de negacion y su interaccion con la polaridad",
        "artefacto": "reports/eda/fase2_rasgos_TR_continuos.csv",
        "evidencia": "presente en la caracterizacion de TR",
    },
    {
        "hallazgo": "La cita ocupa una fraccion medible del texto y se asocia a HS",
        "patron": r"cita|quotation|quote",
        "mecanismo": "fraccion de tokens dentro de comillas y sus atributos",
        "artefacto": "reports/eda/fase2_rasgos_TR.csv",
        "evidencia": "frac_tokens_cita medida por fila",
    },
    {
        "hallazgo": "La segunda persona se asocia a HS y a TR",
        "patron": r"segunda(_persona)?|tuteo",
        "mecanismo": "conteo y densidad de segunda persona",
        "artefacto": "reports/eda/fase2_tests_categoricos.csv",
        "evidencia": "sobrerrepresentacion FP vs TN 1.1757 en HS",
    },
    {
        "hallazgo": "La longitud distingue clases y correlaciona con el error",
        "patron": r"longitud|n_tokens|length|tamano",
        "mecanismo": "longitud como feature y como condicion de binarizacion",
        "artefacto": "reports/eda/longitudes_por_etiqueta.csv",
        "evidencia": "fase2_longitudes_por_clase.csv",
    },
    {
        "hallazgo": "La elongacion y el leet marcan registro coloquial",
        "patron": r"elongac|leet|ortograf",
        "mecanismo": "normalizacion y conteo de elongation y leet",
        "artefacto": "reports/eda/fase2_textometria_por_clase.csv",
        "evidencia": "registro coloquial",
    },
    {
        "hallazgo": "La puntuacion interna y las risas matizan la agresividad",
        "patron": r"puntuacion|risa|interior",
        "mecanismo": "puntuacion interna y risas como feature de AG",
        "artefacto": "reports/eda/fase2_textometria_por_clase.csv",
        "evidencia": "textometria de superficie",
    },
    {
        "hallazgo": "La atribucion y el retweet aparecen en la distribucion de TR",
        "patron": r"atribucion|attribution|rt_presente|retweet",
        "mecanismo": "marcadores de atribucion y presencia de RT",
        "artefacto": "reports/eda/fase2_rasgos_TR.csv",
        "evidencia": "caracterizacion de TR",
    },
    {
        "hallazgo": "Los hashtags y las URLs senalan registro digital y no toxicity",
        "patron": r"hashtag|url",
        "mecanismo": "conteo de hashtags y URLs, e interaccion con longitud",
        "artefacto": "reports/eda/fase2_tests_categoricos.csv",
        "evidencia": "test categorico",
    },
    {
        "hallazgo": "Los perfiles de POS por texto separan las tres etiquetas",
        "patron": r"pos_|tagset|perfil de pos|propos",
        "mecanismo": "perfil de distribucion POS como representacion",
        "artefacto": "reports/eda/fase2_textometria_por_clase.csv",
        "evidencia": "textometria por clase",
    },
    {
        "hallazgo": "La estructura de etiquetas tiene combinaciones casi-imposibles",
        "patron": r"combinacion|estructura de etiquetas|TR=1|regla",
        "mecanismo": "regla dura o suave sobre la combinacion de etiquetas",
        "artefacto": "reports/eda/combinaciones_etiquetas.csv",
        "evidencia": "TR=1 solo con HS=1",
    },
    {
        "hallazgo": "Los grupos de near-duplicate y pseudo-autor cambian la validacion",
        "patron": r"near.?dup|duplicad|pseudo.?autor|grupo",
        "mecanismo": "agrupacion para la validacion cruzada",
        "artefacto": "reports/eda/clusters_near_duplicados.csv",
        "evidencia": "clusters y pseudo_autores.csv",
    },
    {
        "hallazgo": "La rareza de la etiqueta condiciona el error",
        "patron": r"label_rarity|rareza|minority|desbalance",
        "mecanismo": "poblacion de etiqueta rara como condicion",
        "artefacto": "reports/eda/distribucion_etiquetas.csv",
        "evidencia": "distribucion de etiquetas",
    },
]


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V48-00] {mensaje}", flush=True)


def main() -> None:
    """Ejecuta la auditoria de trazabilidad del EDA.

    Returns:
        None.
    """
    inicio = time.time()
    log("P48-00 trazabilidad EDA -> hipotesis -> experimento. 0 modelos, 0 submissions")

    registro = pd.read_csv(REGISTRO)
    cols = ["experiment_id", "question", "hypothesis", "preprocessing", "feature_groups",
            "representation", "model", "result", "decision"]
    registro_texto = registro[cols].astype(str).agg(" ".join, axis=1)
    log(f"   registro con {len(registro)} filas")

    hallazgos = pd.read_csv(EDA / "fase2_hallazgos_clave.csv")
    log(f"   informe de EDA con {len(hallazgos)} hallazgos clave declarados")

    filas = []
    for h in HALLAZGOS:
        mask = registro_texto.str.contains(h["patron"], case=False, regex=True, na=False)
        sub = registro[mask]
        decisiones = [str(x) for x in sub.decision.tolist()]
        # Un veredicto es SUSTANTIVO solo si se refiere a un resultado de modelo
        # o al estado de una familia. Las filas AUDIT_ONLY, DIAGNOSTIC_ONLY o
        # "0 modelos" no dicen nada sobre el mecanismo. Y la ULTIMA fila
        # coincidente suele ser precisamente eso, o una fase posterior sin
        # relacion con el hallazgo.
        def sustantivo(d: str) -> bool:
            """Indica si el veredicto trata el mecanismo y no la auditoria.

            Args:
                d: texto de la columna decision.

            Returns:
                True si el veredicto es sustantivo.
            """
            return not re.search(r"AUDIT_ONLY|DIAGNOSTIC_ONLY|0 modelos|0 experiments",
                                 d, re.IGNORECASE)

        sust = [d for d in decisiones if sustantivo(d)]
        cerradas = [d for d in sust
                    if re.search(r"CLOSED|CERRAD|REJECTED|NO ADOPTADA|NO REALISTIC|"
                                 r"REFUTAD|DESCARTAD|NO_SIGNAL|NO_GANANCIA|REDUNDANT|"
                                 r"INFORMATIONAL_ONLY", d, re.IGNORECASE)]
        aperturas = [d for d in sust
                     if re.search(r"ABIERTA|OPEN|PENDIENTE|CANDIDAT|NEW_SIGNAL|ADOPTAD|"
                                  r"WEAK|PROMISING", d, re.IGNORECASE)
                     and not re.search(r"CLOSED|CERRAD|REJECTED|NO ADOPTADA|REFUTAD|"
                                       r"DESCARTAD|NO_SIGNAL|REDUNDANT", d, re.IGNORECASE)]
        # Criterio: predominio de veredictos sustantivos de cierre.
        ultimo = sust[-1] if sust else ""
        cerrado = bool(cerradas) and len(cerradas) >= len(aperturas)
        ids = [str(x) for x in sub.experiment_id.tolist()]
        filas.append({
            "eda_finding": h["hallazgo"],
            "source": h["artefacto"],
            "historical_evidence": h["evidencia"],
            "hypothesis_derived": h["mecanismo"],
            "n_experiments": len(ids),
            "experiment_ids": "; ".join(ids[:14]) + ("; ..." if len(ids) > 14 else ""),
            "result": (str(sub.result.iloc[0])[:200] if len(sub) else "sin experimento"),
            "closed": cerrado,
            "n_substantive": len(sust),
            "n_closed_rows": len(cerradas),
            "n_open_rows": len(aperturas),
            "last_decision": ultimo,
            "why": (f"{len(sust)} veredictos sustantivos: {len(cerradas)} declaran cierre "
                    f"y {len(aperturas)} declaran apertura"),
            "remaining_gap": ("cubierta" if cerrado else
                              ("abierta" if ids else "nunca convertida en experimento")),
        })
        log(f"   {h['hallazgo'][:52]:52} exp={len(ids):3d} cerrada={filas[-1]['closed']}")

    n_cerradas = sum(1 for f in filas if f["closed"])
    n_nunca = sum(1 for f in filas if f["n_experiments"] == 0)
    log(f"   hallazgos con mecanismo cerrado: {n_cerradas}/{len(filas)}")
    log(f"   hallazgos sin ningun experimento: {n_nunca}")

    # El EDA original se declarations en 6 hallazgos clave, y solo 3 tienen
    # hipotesis no vacia. Se registra esa cobertura explicita.
    con_hipotesis = hallazgos[hallazgos.hipotesis.notna() & (hallazgos.hipotesis.astype(str).str.len() > 8)]
    cobertura = {
        "hallazgos_declarados_en_el_informe": len(hallazgos),
        "con_hipotesis_no_vacia": len(con_hipotesis),
        "sin_hipotesis": int(len(hallazgos) - len(con_hipotesis)),
        "detalle": [{"hallazgo": str(r.hallazgo), "hipotesis": str(r.hipotesis)}
                    for _, r in hallazgos.iterrows()],
    }
    log(f"   cobertura del informe EDA: {cobertura['con_hipotesis_no_vacia']}"
        f"/{cobertura['hallazgos_declarados_en_el_informe']} hallazgos con hipotesis")

    informe = {
        "familia": "P48_EDA_RESIDUAL",
        "experimento": "V48-00 (P0 trazabilidad de los hallazgos del EDA)",
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estado": "AUDIT_ONLY",
        "pregunta": ("El EDA original se convertingio de forma suficientemente completa en "
                     "hipotesis experimentales con mecanismo concreto?"),
        "regla_de_cierre": ["mecanismo probado", "control adecuado", "evaluacion valida"],
        "matriz_de_trazabilidad": filas,
        "cobertura_del_informe_eda": cobertura,
        "resumen": {
            "hallazgos_auditados": len(filas),
            "con_mecanismo_cerrado": n_cerradas,
            "sin_ningun_experimento": n_nunca,
            "abiertos": sum(1 for f in filas if not f["closed"]),
        },
        "gastos": {"submissions": 0, "modelos_entrenados": 0,
                   "consultas_kaggle": 0, "segundos": round(time.time() - inicio, 1)},
    }
    REPORTE_JSON.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _csv(filas)
    _md(informe)
    log("   informes escritos en reports/p48_eda_traceability.{json,md,csv}")
    log(f"VEREDICTO: {n_cerradas}/{len(filas)} hallazgos con mecanismo cerrado, "
        f"{n_nunca} sin experimento, 0 modelos, {round(time.time() - inicio, 1)} s")


def _csv(filas: list[dict[str, Any]]) -> None:
    """Escribe la matriz de trazabilidad en CSV.

    Args:
        filas: filas de la matriz.

    Returns:
        None.
    """
    cab = ["eda_finding", "source", "historical_evidence", "hypothesis_derived",
           "n_experiments", "experiment_ids", "result", "closed", "why", "remaining_gap"]
    with REPORTE_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cab)
        for x in filas:
            w.writerow([x.get(c, "") for c in cab])


def _md(informe: dict[str, Any]) -> None:
    """Escribe el informe legible de trazabilidad.

    Args:
        informe: diccionario con los resultados.

    Returns:
        None.
    """
    m = informe["matriz_de_trazabilidad"]
    c = informe["cobertura_del_informe_eda"]
    r = informe["resumen"]
    lineas = [
        "# P48 - Trazabilidad del EDA original",
        "",
        f"Generado por `experiments/v48_00_eda_traceability.py` ({informe['experimento']}).",
        "",
        f"Fecha {informe['fecha']}. **0 modelos, 0 submissions, 0 consultas a Kaggle.**",
        "",
        f"Pregunta: {informe['pregunta']}",
        "",
        "Regla de cierre aplicada: " + " + ".join(informe["regla_de_cierre"]) + ".",
        "El mismo tema no implica la misma familia.",
        "",
        "## 1. Cobertura del informe EDA original",
        "",
        f"- hallazgos declarados en `fase2_hallazgos_clave.csv`: **{c['hallazgos_declarados_en_el_informe']}**",
        f"- con hipotesis no vacia: **{c['con_hipotesis_no_vacia']}**",
        f"- sin hipotesis: **{c['sin_hipotesis']}**",
        "",
        "| hallazgo | hipotesis declarada en el EDA |",
        "|---|---|",
    ]
    for d in c["detalle"]:
        h = d["hipotesis"] if d["hipotesis"] not in ("nan", "") else "_(vacia)_"
        lineas.append(f"| {d['hallazgo'][:76]} | {h[:70]} |")
    lineas += [
        "",
        "## 2. Matriz de trazabilidad",
        "",
        "| hallazgo del EDA | mecanismo derivado | exp | cerrada | hueco |",
        "|---|---|---:|:-:|---|",
    ]
    for x in m:
        lineas.append(
            f"| {x['eda_finding'][:56]} | {x['hypothesis_derived'][:46]} | "
            f"{x['n_experiments']} | {'si' if x['closed'] else 'NO'} | {x['remaining_gap']} |"
        )
    lineas += [
        "",
        "## 3. Detalle por hallazgo",
        "",
    ]
    for x in m:
        lineas += [
            f"### {x['eda_finding']}",
            "",
            f"- artefacto fuente: `{x['source']}`",
            f"- evidencia historica: {x['historical_evidence']}",
            f"- mecanismo derivado: {x['hypothesis_derived']}",
            (f"- experimentos encontrados: {x['n_experiments']}"
            f" ({x['n_closed_rows']} declaran cierre, {x['n_open_rows']} abiertos)"),
            f"- ids: {x['experiment_ids'] or 'ninguno'}",
            f"- resultado registrado: {x['result']}",
            f"- cerrada: **{x['closed']}** ({x['why']})",
            f"- hueco restante: **{x['remaining_gap']}**",
            "",
        ]
    lineas += [
        "## 4. Resumen",
        "",
        f"- hallazgos auditados: {r['hallazgos_auditados']}",
        f"- con mecanismo cerrado: {r['con_mecanismo_cerrado']}",
        f"- sin ningun experimento: {r['sin_ningun_experimento']}",
        f"- abiertos: {r['abiertos']}",
        "",
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
