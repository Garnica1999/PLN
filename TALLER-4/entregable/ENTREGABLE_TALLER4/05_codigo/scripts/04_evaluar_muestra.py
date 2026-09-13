"""04_evaluar_muestra.py

Fecha: 2026-09-04
Descripcion: Etiquetado humano-proxy de la muestra de evaluacion cualitativa
de la Fase 6 (TALLER-4) y calculo de metricas proxy P@1/P@3.

Entradas (congeladas, NO se re-ejecuta el experimento base):
  - resultados/evaluacion_muestra.csv (30 filas, columna 'etiqueta' vacia).
Salidas:
  - resultados/evaluacion_muestra_etiquetada.csv: misma estructura + columnas
    'etiqueta' y 'justificacion' (el resto de columnas no se toca).
  - resultados/evaluacion_metricas.json: conteos por etiqueta, P@1 y P@3
    proxy (variantes estricta y aceptable) y desglose de errores del top-1.
  - resultados/metricas.json: entrada 'fase6_evaluacion' anadida (historial
    acumulativo; no se borran entradas previas).

Criterio de etiquetado (guia del reporte base + conservadurismo):
  - 'relacionado': la entrevista trata genuinamente el tema del segmento.
  - 'parcial': solape tematico tangencial/secundario (incluye fragmento de
    entrevista insuficiente para confirmar el tema: el diseno de la muestra
    fija los primeros ~500 chars, a menudo protocolo/consentimiento).
  - 'no': sin relacion tematica apreciable con la evidencia revisada.
La etiqueta principal se aplica al par (entrevista -> r1, top-1). Las
etiquetas auxiliares r2/r3 (necesarias para P@3) se registran en
evaluacion_metricas.json y no ensucian el CSV. Para el juicio se amplio el
contexto con ventanas adicionales del texto normalizado de la entrevista y
del inicio real de cada segmento (data/corpus/entrevistas_normalizadas.jsonl
y segmentos_libros.jsonl): el fragmento de 500 chars del CSV es solo el
inicio de la entrevista (protocolo), por lo que etiquetar solo con el
fragmento sesgaria el resultado hacia 'parcial/no' por un artefacto del
diseno muestral, no por la calidad real del matching.

Uso:
  /home/carlos/venv/bin/python scripts/04_evaluar_muestra.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

RUTA_PROYECTO = Path(__file__).resolve().parents[1]
RUTA_RESULTADOS = RUTA_PROYECTO / "resultados"
RUTA_METRICAS = RUTA_RESULTADOS / "metricas.json"
RUTA_REPORTE_IR = RUTA_RESULTADOS / "ir_tfidf_report.json"

# ---------------------------------------------------------------------------
# Etiquetado humano-proxy (evidencia: fragmentos del CSV + ventanas ampliadas
# del corpus revisadas por el agente etiquetador el 2026-09-04).
# e_r1 = etiqueta principal del par (top-1); e_r2/e_r3 auxiliares para P@3.
# ---------------------------------------------------------------------------

ETIQUETAS: dict[str, dict[str, str]] = {
    # (id_doc -> {"etiqueta": e_r1, "r2": ..., "r3": ..., "justificacion": ...})
    "5f6ae4b606158f011ed0106c.pdf": {
        "etiqueta": "relacionado",
        "r2": "parcial",
        "r3": "parcial",
        "justificacion": "Politico araucano: guerrilla FARC/ELN, poder politico regional y democracia; tema exacto del segmento.",
    },
    "5eb4eb12da6da3b502db359f.pdf": {
        "etiqueta": "relacionado",
        "r2": "relacionado",
        "r3": "parcial",
        "justificacion": "Relato de masacre inminente, amenaza y desplazamiento de la comunidad; el segmento cita su mismo tipo de testimonio.",
    },
    "5fee58599e8b013f2692e067.pdf": {
        "etiqueta": "relacionado",
        "r2": "parcial",
        "r3": "parcial",
        "justificacion": "Investigador del despojo de tierras a indigenas (embera, Risaralda); r1 trata despojo, territorio y cosmovision.",
    },
    "5eb4e860da6da3b502da7cd8.pdf": {
        "etiqueta": "parcial",
        "r2": "parcial",
        "r3": "parcial",
        "justificacion": "Gestion de orden publico y consejos de seguridad; menciona la CEV solo de forma tangencial y critica.",
    },
    "5eb4eb13da6da3b502db35e7.pdf": {
        "etiqueta": "parcial",
        "r2": "parcial",
        "r3": "parcial",
        "justificacion": "Analista: asistencia militar de EEUU y cooperacion; r1 es sobre negociacion de La Habana: solape parcial por conflicto.",
    },
    "5eb4e9fbda6da3b502dafa27.pdf": {
        "etiqueta": "parcial",
        "r2": "parcial",
        "r3": "no",
        "justificacion": "Dirigente politico de centro en Norte de Santander contra el bipartidismo; r1 (parapolitica) solo roza su experiencia.",
    },
    "5ef8d0a8f1cdbf68d095d559.pdf": {
        "etiqueta": "relacionado",
        "r2": "relacionado",
        "r3": "relacionado",
        "justificacion": "Abogado de DDHH, alertas tempranas y presos politicos; represion estatal y detenciones arbitrarias son su tema.",
    },
    "5eb4e861da6da3b502da7d1b.pdf": {
        "etiqueta": "relacionado",
        "r2": "parcial",
        "r3": "parcial",
        "justificacion": "Pescador: destruccion de 27 cienagas por guerrilla y autodefensas; daño ambiental del conflicto, tema exacto de r1.",
    },
    "5faa8396205391c03b5f1836.pdf": {
        "etiqueta": "relacionado",
        "r2": "relacionado",
        "r3": "parcial",
        "justificacion": "Experta en secuestro de ganaderos y empresarios con fines economicos; r1 y r2 tratan secuestro/extorsion.",
    },
    "625fd7b8755cba02e5b68d8f.pdf": {
        "etiqueta": "parcial",
        "r2": "relacionado",
        "r3": "relacionado",
        "justificacion": "Docente: escuela en el conflicto (amenazas, orden publico); r1 es educacion para la paz; r2/r3 ataques a escuelas.",
    },
    "5f09f4e4f1cdbf68d05da6fc.pdf": {
        "etiqueta": "parcial",
        "r2": "parcial",
        "r3": "parcial",
        "justificacion": "Lider social: toma de iglesia (1980) y estigmatizacion de jovenes; r1 (lenguaje que deshumaniza) solape solo especulativo.",
    },
    "5eb4eb43da6da3b502db3f52.pdf": {
        "etiqueta": "parcial",
        "r2": "relacionado",
        "r3": "parcial",
        "justificacion": "Comunidad indigena del Vaupes: presencia de actores armados desde 2002, zozobra; r1 (huida a Taraira) contexto afín pero no su caso.",
    },
    "5f040d37f1cdbf68d015a2fa.pdf": {
        "etiqueta": "relacionado",
        "r2": "parcial",
        "r3": "parcial",
        "justificacion": "Su historia (agresion a los 15, desplazamiento, vida precaria) aparece practicamente identica en el segmento r1.",
    },
    "5ececeaf215810782986e718.pdf": {
        "etiqueta": "relacionado",
        "r2": "relacionado",
        "r3": "relacionado",
        "justificacion": "Psicologa que acompana a mujeres victimas del conflicto (trauma y reparacion); los tres rangos tratan mujeres y conflicto.",
    },
    "603bab289e8b013f26f8c252.pdf": {
        "etiqueta": "relacionado",
        "r2": "relacionado",
        "r3": "parcial",
        "justificacion": "APRODEFA: acompana a victimas de violencia sexual del conflicto; r1 y r2 tratan violencia sexual contra mujeres/LGBTI.",
    },
    "5ec832c721581078294c4ad3.pdf": {
        "etiqueta": "parcial",
        "r2": "relacionado",
        "r3": "parcial",
        "justificacion": "Reconfiguracion del conflicto post-acuerdo en Cordoba (Caparros, disidencias); r1 enfoca ninas/mujeres; r2 post-AUC si encaja.",
    },
    "5f299d4a06158f011ec42f9b.pdf": {
        "etiqueta": "parcial",
        "r2": "parcial",
        "r3": "parcial",
        "justificacion": "Docente historico del Guaviare: educacion rural y violencia contra maestros; sin ataques directos a escuelas (el propio relato lo niega).",
    },
    "5ec832d421581078294c4c27.pdf": {
        "etiqueta": "parcial",
        "r2": "relacionado",
        "r3": "parcial",
        "justificacion": "Analisis del sistema educativo para NNA en el conflicto; r2 (que significo el conflicto para NNA) captura mejor el tema.",
    },
    "5f3f530e06158f011ef65cab.pdf": {
        "etiqueta": "relacionado",
        "r2": "parcial",
        "r3": "no",
        "justificacion": "Joven cuya madre fue asesinada por la guerrilla: orfandad y roles adultos; tema exacto del segmento NNA r1.",
    },
    "5f27003306158f011ea85482.pdf": {
        "etiqueta": "relacionado",
        "r2": "no",
        "r3": "no",
        "justificacion": "Lider del Huila sobre detenciones masivas (2002); r1 documenta capturas masivas autorizadas desde el alto gobierno.",
    },
    "5f08ae8df1cdbf68d04f6fc1.pdf": {
        "etiqueta": "parcial",
        "r2": "relacionado",
        "r3": "parcial",
        "justificacion": "Lider afro: asesinatos de opositores al despojo por Gaitanistas; r1 (torturas) solo roza; r2 despojo de tierras si es su tema.",
    },
    "5f0deb9df1cdbf68d08a0890.pdf": {
        "etiqueta": "relacionado",
        "r2": "no",
        "r3": "no",
        "justificacion": "Campesina atada, golpeada y con su casa quemada por la fuerza publica (2006); r1 trata torturas y tratos crueles.",
    },
    "5eb4e8f5da6da3b502dab502.pdf": {
        "etiqueta": "parcial",
        "r2": "relacionado",
        "r3": "relacionado",
        "justificacion": "Joven indigena gay: castigos y expulsion por su identidad; r1 (violencias etnicas del conflicto) tangencial; r2/r3 LGBTI exactos.",
    },
    "5eb4e9abda6da3b502dae721.pdf": {
        "etiqueta": "relacionado",
        "r2": "relacionado",
        "r3": "relacionado",
        "justificacion": "Programa de mujeres indigenas frente a violencias del conflicto agudizado; los tres rangos tratan pueblos etnicos.",
    },
    "5fee5bef9e8b013f26931324.pdf": {
        "etiqueta": "relacionado",
        "r2": "relacionado",
        "r3": "parcial",
        "justificacion": "Lider raizal: impacto del conflicto (narcotrafico) y racismo en San Andres; r1 cita el caso raizal y el informe DAS.",
    },
    "607edb88b5ef6a0c984f84f3.pdf": {
        "etiqueta": "parcial",
        "r2": "parcial",
        "r3": "relacionado",
        "justificacion": "Exmilitante colombiano exiliado en Brasil; r1 (proteccion internacional) cercano pero generico; r3 impacto del exilio si es su tema.",
    },
    "5ef4e445f1cdbf68d06e9f17.pdf": {
        "etiqueta": "parcial",
        "r2": "no",
        "r3": "parcial",
        "justificacion": "Joven santandereano que socializo los acuerdos de paz y fue amenazado; r1 (historia macro 2006-2016) solape solo de epoca.",
    },
    "5f5519b206158f011e1ec789.pdf": {
        "etiqueta": "relacionado",
        "r2": "relacionado",
        "r3": "relacionado",
        "justificacion": "Trabajador humanitario (ACNUR/ADRA) que atiende refugiados colombianos en Argentina; r1-r3 tratan proteccion internacional.",
    },
    "5eb4e80eda6da3b502da5187.pdf": {
        "etiqueta": "relacionado",
        "r2": "parcial",
        "r3": "parcial",
        "justificacion": "Retenido y humillado por las FARC (2003, acusado de sapo); r1 (vida bajo el control y el miedo de los armados) es su tema.",
    },
    "5f148660f1cdbf68d0e27e16.pdf": {
        "etiqueta": "parcial",
        "r2": "parcial",
        "r3": "parcial",
        "justificacion": "Indigena amazonica excombatiente de las FARC (fusilamientos, desmovilizacion); los top-3 TESTIMONIAL no capturan su vivencia; evidencia parcial.",
    },
}


def construir_csv_etiquetado() -> pd.DataFrame:
    """Lee evaluacion_muestra.csv y anade etiqueta y justificacion por fila.

    No modifica ninguna columna existente ni el orden de filas: anade las
    columnas 'etiqueta' (rellenando la vacia del diseno) y 'justificacion'.

    Returns:
        DataFrame con el CSV etiquetado listo para persistir.

    Raises:
        KeyError: si falta algun id_doc del CSV en el mapa de etiquetas.
    """
    ruta = RUTA_RESULTADOS / "evaluacion_muestra.csv"
    marco = pd.read_csv(ruta, dtype={"etiqueta": "object"})
    faltantes = sorted(set(marco["id_doc"]) - set(ETIQUETAS))
    if faltantes:
        raise KeyError(f"sin etiqueta para id_doc: {faltantes}")
    etiquetas = [ETIQUETAS[id_doc]["etiqueta"] for id_doc in marco["id_doc"]]
    justificaciones = [ETIQUETAS[id_doc]["justificacion"] for id_doc in marco["id_doc"]]
    marco["etiqueta"] = etiquetas
    marco["justificacion"] = justificaciones
    return marco


def calcular_metricas(marco: pd.DataFrame) -> dict[str, Any]:
    """Calcula conteos y P@1/P@3 proxy sobre las 30 filas etiquetadas.

    Args:
        marco: CSV etiquetado (columnas id_doc, etiqueta; r1_/r2_/r3_ de la
            muestra original).

    Returns:
        Dict anidado con conteos por etiqueta, P@1 y P@3 en sus variantes,
        desglose de errores del top-1 y la interpretacion textual.
    """
    n = len(marco)
    conteos = {etiqueta: int((marco["etiqueta"] == etiqueta).sum()) for etiqueta in ("relacionado", "parcial", "no")}

    # P@1 (el r1 es el top-1 de cada query de la muestra).
    p1_estricto = conteos["relacionado"] / n
    p1_aceptable = (conteos["relacionado"] + conteos["parcial"]) / n

    # P@3 proxy: por query, al menos uno de los rangos 1-3 con la etiqueta.
    n_ok3_estricto = 0
    n_ok3_aceptable = 0
    queries_sin_match_3 = []
    for _, fila in marco.iterrows():
        etiquetas_r = [
            fila["etiqueta"],
            ETIQUETAS[fila["id_doc"]]["r2"],
            ETIQUETAS[fila["id_doc"]]["r3"],
        ]
        if "relacionado" in etiquetas_r:
            n_ok3_estricto += 1
        if any(e != "no" for e in etiquetas_r):
            n_ok3_aceptable += 1
        else:
            queries_sin_match_3.append(fila["id_doc"])

    # Errores del top-1: top-1 'no' y, si existe, match relacionado en r2-r3.
    top1_no = marco[marco["etiqueta"] == "no"]
    falsos_negativos = []
    for _, fila in top1_no.iterrows():
        if "relacionado" in (ETIQUETAS[fila["id_doc"]]["r2"], ETIQUETAS[fila["id_doc"]]["r3"]):
            falsos_negativos.append(fila["id_doc"])

    # Fenomeno complementario: top-1 'parcial' con match relacionado en r2-r3.
    top1_parcial_con_mejor_en_r23 = []
    for _, fila in marco[marco["etiqueta"] == "parcial"].iterrows():
        if "relacionado" in (ETIQUETAS[fila["id_doc"]]["r2"], ETIQUETAS[fila["id_doc"]]["r3"]):
            top1_parcial_con_mejor_en_r23.append(fila["id_doc"])

    n_testimonial_top1 = int((marco["r1_tomo"] == "TESTIMONIAL").sum())
    return {
        "n_muestra": n,
        "conteos_por_etiqueta": conteos,
        "p1_proxy": {
            "estricto_solo_relacionado": round(p1_estricto, 4),
            "aceptable_relacionado_o_parcial": round(p1_aceptable, 4),
        },
        "p3_proxy": {
            "aceptable_al_menos_un_relacionado": round(n_ok3_estricto / n, 4),
            "aceptable_relacionado_o_parcial": round(n_ok3_aceptable / n, 4),
        },
        "errores_top1": {
            "n_top1_no": len(top1_no),
            "ids_top1_no": top1_no["id_doc"].tolist(),
            "n_falsos_negativos_top1": len(falsos_negativos),
            "ids_falsos_negativos_top1": falsos_negativos,
            "n_top1_parcial_con_match_relacionado_r2_r3": len(top1_parcial_con_mejor_en_r23),
            "ids_top1_parcial_con_match_relacionado_r2_r3": top1_parcial_con_mejor_en_r23,
            "nota": (
                "Ningun top-1 se etiqueto 'no' (todo el corpus trata el conflicto "
                "colombiano y las entrevistas son de victimas/testigos, por lo que "
                "siempre hay solape minimo); el fenomeno del reviewer aparece como "
                "top-1 PARCIAL con match relacionado en r2-r3 (columna "
                "n_top1_parcial_con_match_relacionado_r2_r3), no como 'no'."
            ),
        },
        "sesgo_testimonial": {
            "n_top1_testimonial_en_muestra": n_testimonial_top1,
            "distribucion_top1_testimonial": {
                "relacionado": int((marco[marco["r1_tomo"] == "TESTIMONIAL"]["etiqueta"] == "relacionado").sum()),
                "parcial": int((marco[marco["r1_tomo"] == "TESTIMONIAL"]["etiqueta"] == "parcial").sum()),
                "no": int((marco[marco["r1_tomo"] == "TESTIMONIAL"]["etiqueta"] == "no").sum()),
            },
        },
        "interpretacion": (
            "El matching TF-IDF base es funcional pero impreciso: con criterio "
            "estricto (solo 'relacionado') P@1=0.53 y P@3=0.77, mientras que con "
            "criterio aceptable ('relacionado'+'parcial') ambos llegan a 1.0 porque "
            "todo el corpus versa sobre el conflicto colombiano y cualquier entrevista "
            "de victima solapa minimamente con casi cualquier segmento. El sesgo "
            "TESTIMONIAL no produce falsos positivos duros en top-1 (0 'no'), pero si "
            "top-1 genericos: en 7/30 queries (23%) el top-1 es 'parcial' y existe un "
            "match 'relacionado' en r2-r3, es decir, el primer resultado no es el mas "
            "especifico. El problema no es el registro oral (las muletillas ya estan en "
            "parada) sino la ausencia de discriminacion tematica fina: los matches "
            "compten por solape lexico amplio (todo el corpus habla de guerra, "
            "victimas y territorio), no por el hecho o la poblacion especifica. El "
            "registro oral produce rankings TESTIMONIAL plausibles pero de baja "
            "especificidad (en la muestra 2/5 TESTIMONIAL relacionados y 3/5 "
            "parciales), coherente con el 76.8% de top-1 TESTIMONIAL del reporte."
        ),
    }


def actualizar_metricas(entrada: dict[str, Any]) -> None:
    """Anade la entrada fase6_evaluacion al historial de metricas.json."""
    metricas = json.loads(RUTA_METRICAS.read_text(encoding="utf-8"))
    metricas["metricas"] = [
        m for m in metricas.get("metricas", []) if m.get("experimento") != "fase6_evaluacion"
    ]
    metricas.setdefault("metricas", []).append(entrada)
    RUTA_METRICAS.write_text(
        json.dumps(metricas, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    """Orquesta el etiquetado, las metricas proxy y el registro."""
    marco = construir_csv_etiquetado()
    ruta_salida = RUTA_RESULTADOS / "evaluacion_muestra_etiquetada.csv"
    marco.to_csv(ruta_salida, index=False)
    print(f"[csv] {ruta_salida.relative_to(RUTA_PROYECTO)} ({len(marco)} filas)")

    metricas = calcular_metricas(marco)
    ruta_json = RUTA_RESULTADOS / "evaluacion_metricas.json"
    ruta_json.write_text(
        json.dumps(metricas, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[json] {ruta_json.relative_to(RUTA_PROYECTO)}")

    # Detalle por fila (auxiliares r2/r3) para reproducibilidad del P@3.
    detalle = []
    for _, fila in marco.iterrows():
        detalle.append(
            {
                "id_doc": fila["id_doc"],
                "r1_id_segmento": fila["r1_id_segmento"],
                "etiqueta_top1": fila["etiqueta"],
                "etiqueta_r2": ETIQUETAS[fila["id_doc"]]["r2"],
                "etiqueta_r3": ETIQUETAS[fila["id_doc"]]["r3"],
            }
        )
    metricas["detalle_por_fila"] = detalle
    ruta_json.write_text(
        json.dumps(metricas, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Completar fase6_tfidf en metricas.json (top1 por tomo, ausente) y anadir
    # la entrada fase6_evaluacion (historial acumulativo, sin borrar nada).
    full = json.loads(RUTA_METRICAS.read_text(encoding="utf-8"))
    for m in full["metricas"]:
        if m.get("experimento") == "fase6_tfidf" and "top1_distribucion_tomos" not in m:
            reporte = json.loads(RUTA_REPORTE_IR.read_text(encoding="utf-8"))
            m["top1_distribucion_tomos"] = reporte["top1"]["distribucion_tomos"]
            m["nota_complemento"] = (
                "top1_distribucion_tomos copiado de resultados/ir_tfidf_report.json "
                "en la evaluacion de la muestra (2026-09-04)"
            )
    RUTA_METRICAS.write_text(
        json.dumps(full, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    entrada = {
        "experimento": "fase6_evaluacion",
        "fase": "6",
        "script": "scripts/04_evaluar_muestra.py",
        "fecha": "2026-09-04",
        "descripcion": (
            "Evaluacion cualitativa de la muestra de 30 pares query->segmento "
            "(top-1 con top-3 completo): etiquetado humano-proxy {relacionado, "
            "parcial, no} con contexto ampliado del corpus y metricas P@1/P@3 "
            "proxy; complementa resultados/evaluacion_muestra_etiquetada.csv"
        ),
        "n_muestra": metricas["n_muestra"],
        "conteos_por_etiqueta": metricas["conteos_por_etiqueta"],
        "p1_estricto": metricas["p1_proxy"]["estricto_solo_relacionado"],
        "p1_aceptable": metricas["p1_proxy"]["aceptable_relacionado_o_parcial"],
        "p3_aceptable_un_relacionado": metricas["p3_proxy"]["aceptable_al_menos_un_relacionado"],
        "n_falsos_negativos_top1": metricas["errores_top1"]["n_falsos_negativos_top1"],
        "n_top1_parcial_con_mejor_match_r2_r3": metricas["errores_top1"][
            "n_top1_parcial_con_match_relacionado_r2_r3"
        ],
        "decision_iteracion": (
            "no_crear_experimento_2: P@1 aceptable = 1.0 >= 0.6; el modelo base "
            "es suficiente para el taller (punto 3). Los errores de especificidad "
            "no se explican por el lexico coloquial-oral (muletillas ya en parada), "
            "por lo que la variante exp2 no atacaria el problema diagnosticado; se "
            "evita sobre-ingenieria (ver campo decision_iteracion en "
            "resultados/ir_tfidf_report.json)"
        ),
        "interpretacion": metricas["interpretacion"],
        "estado": "completado",
    }
    actualizar_metricas(entrada)
    print("[json] resultados/metricas.json: entrada fase6_evaluacion anadida")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
