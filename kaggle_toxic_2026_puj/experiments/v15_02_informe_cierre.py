"""V15-02: informe de cierre de la frontera final, con la estructura que el enunciado exige.

Que decide este script
---------------------
V15 cambio el objetivo de investigacion: dejo de buscar modelos y paso a preguntar **que
informacion disponible en los datos no esta usando V03**. Los dos experimentos anteriores
respondieron a esa pregunta, y este script la consolida en el estado de cierre que pide la
seccion 13, con sus seis apartados.

El veredicto, y por que no es el que cabia esperar
--------------------------------------------------
La respuesta tiene dos mitades que no hay que mezclar.

**La senal residual existe.** La posicion en el documento no esta representada por V03, y se
demuestra que es informacion genuinamente complementaria: AUC de 0.5547 en los estratos del margen
de V03 con un valor p empirico por debajo de 1 entre 500, frente a las 500 permutaciones. Es decir,
**el rasgo ve algo que V03 no ve**, que es literalmente la pregunta de la seccion 12.

**Y es inutilizable.** Anadido a V03, las tres candidatas que pasaron P3 dan deltas de -0.0014 a
-0.0030, y lo que decide no es el delta sino el **acierto cero** entre las celdas modificadas: de
50, 14 y 57 celdas, ninguna coincide con la verdad. El filtro de detectabilidad de V8-02 exige
0.5166.

Esa combinacion es el hallazgo de la fase: **una fuente de senal puede ser real, complementaria y
estadisticamente significativa, y a la vez ser inutilizable en el punto de decision.** Ordena bien
a los ejemplos y empuja en el sentido contrario en el umbral congelado.

Lo que la fase NO hace
----------------------
No propose features nuevas. Audita doce clases de senal, de las cuales dos no existen como campo,
siete ya estaban medidas, y tres eran la unica que V03 no puede representar por construccion. Y
esas tres se miden con la prueba que decide, no con una lista de 100 features.

Uso
---
    /home/carlos/venv/bin/python experiments/v15_02_informe_cierre.py
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v11_ayudas import fila, sello

EXPERIMENTO = "V15-02 (informe de cierre de la frontera final)"
REPORTE_MD = PROJECT_ROOT / "reports" / "v15_final_cierre.md"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v15_02_informe_cierre.json"

REFERENCIA_V03 = 0.7772114525165721
LB_V03 = 0.78306
DELTA_ORACULO = 0.046271
GATE_EXPLORATORIO = 0.002
GATE_CANDIDATO = 0.003
UMBRAL_DETECCION = 0.5166

FAMILIAS_CERRADAS: tuple[tuple[str, str], ...] = (
    ("Representacion basica TF-IDF", "V03 y todas las variantes de word y char"),
    ("Subpalabras", "cubierta exactamente por char_wb(3,5)"),
    ("Embeddings desde cero", "track v5, cerrado sin candidata"),
    ("Pooling, secuenciales, GRU y fine-tuning", "cerrados en el track de embeddings"),
    ("Meta-modelos y stacking", "E2-02 y E2-18, +0.0102 local y rechazo por transferencia"),
    ("Routing duro", "H2, -0.004187, la llave falla en 143 filas"),
    ("Factorizacion condicional", "V6-B, B1 -0.0107, B2 -0.0008, B3 -0.0115"),
    ("Label powerset", "V4-E, 0.771289 frente a 0.777211, NO_SIGNAL"),
    ("Objetivo de ranking o por pares", "H3, -0.002469, y V12-01, +0.002738 sostenido por un pliegue"),
    ("Perdidas a medida", "H2a, GCE +0.001616, marginal bajo el gate exploratorio"),
    ("Perdida enmascarada o partially-label", "V14, identica a H2-M con diferencia 0.000e+00"),
    ("Instance weighting y downweight", "V4-B05, V4-C, V6-03, V6-07, V8-07, todos negativos"),
    ("Rasgos estructurales", "E2-18, -0.0052 y -0.0051 en dos estimaciones"),
    ("Correccion de la estimacion de HS", "V13, la correccion deja TR y AG con delta exactamente cero"),
    ("Inferencia estructurada de etiquetas", "V13, cerrada formalmente"),
    ("Bagging y ensembles de repeticiones", "cerrados, no superan la referencia"),
    ("Prior shift y calibracion", "v11, cerrada"),
    ("Transduccion", "cerrada en su momento"),
)


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V15-02] {mensaje}", flush=True)


def main() -> None:
    """Consolida la fase y escribe el informe de cierre."""
    inicio = time.time()
    log("V15-02: informe de cierre de la frontera final")

    inventario = _leer("v15_00_inventario_senal.json")
    gap = _leer("v15_01_gap_y_ortogonalidad.json")

    log("1/5 A. estado del proyecto")
    registro = _estadisticas()
    log(f"   experimentos en el registro: {registro['total']}")
    log(f"   con adopcion positiva: {registro['adoptadas']}")
    log(f"   con adopcion negativa explicita: {registro['rechazadas']}")
    log(f"   mejor resultado local: {registro['mejor_local']} en {registro['mejor_local_id']}")
    log(f"   con suerte en el externo: {registro['mejor_local_suerte']}")
    log(f"   familias cerradas: {len(FAMILIAS_CERRADAS)}")

    log("2/5 B. diagnostico del cuello de botella")
    log(f"   techo del oraculo: {DELTA_ORACULO:+.6f}, y no es un techo de HS")

    log("3/5 C y D. inventario y senal residual")
    conteo = inventario["conteo"]
    log(
        f"   clases inventariadas: {conteo['total']} | inexistentes {conteo['inexistentes']} | "
        f"medidas o cerradas {conteo['medidas']}"
    )
    p3 = gap["p3_ortogonalidad"]
    p4 = gap["p4_adicion"]
    complementarias = [n for n, d in p3.items() if d["veredicto"] == "COMPLEMENTARIA"]
    log(f"   candidatas complementarias en P3: {len(complementarias)} de {len(p3)}")
    for d in p4["resultados"]:
        log(
            f"     {d['rasgo']:32s} delta {d['delta']:+.6f} | celdas {d['celdas_modificadas']:3d} "
            f"| acierto {d['acierto_entre_modificadas']:.4f}"
        )

    log("4/5 E. resultado de la fase")
    log(f"   experimentos: 2 | adoptables: {len(gap['candidatas_adoptables'])}")

    log("5/5 F. decision")
    decision = "CLOSED"
    informe = {
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "A_estado_del_proyecto": {
            "lb_v03": LB_V03,
            "corrected_cv_v03": REFERENCIA_V03,
            "mejor_resultado_local": registro["mejor_local"],
            "mejor_resultado_local_id": registro["mejor_local_id"],
            "mejor_resultado_local_suerte": registro["mejor_local_suerte"],
            "experimentos_en_el_registro": registro["total"],
            "adopciones_positivas": registro["adoptadas"],
            "rechazos_explicitos": registro["rechazadas"],
            "familias_cerradas": len(FAMILIAS_CERRADAS),
            "detalle_de_familias": [{"familia": f, "cierre": c} for f, c in FAMILIAS_CERRADAS],
        },
        "B_cuello_de_botella": {
            "delta_oraculo": DELTA_ORACULO,
            "origen_del_techo": (
                "El oraculo no cambia ni una sola celda de la columna HS: pone TR y AG a cero en las "
                "835 filas donde el HS verdadero es cero, borrando 196 falsos positivos, 78 de TR y "
                "118 de AG. Es un efecto aguas abajo de la condicion semantica"
            ),
            "por_que_mejorar_hs_no_sirve": (
                "V13 lo midio: corregir la etiqueta de HS deja TR y AG con delta exactamente cero, "
                "porque el meta produce una probabilidad para HS y no altera lo que el sistema "
                "decide sobre las otras dos etiquetas"
            ),
            "formulacion_correcta": (
                "El techo del oraculo procede de conocer la condicion semantica HS al evaluar TR y "
                "AG. Mejorar HS por si sola no produce beneficio. Y anadir la condicion como llave "
                "tampoco, porque la llave falla en 143 filas y da -0.004187"
            ),
        },
        "C_inventario_de_senal": {
            "clases": conteo["total"],
            "inexistentes": conteo["inexistentes"],
            "cubiertas_por_v03": conteo["cubiertas_por_v03"],
            "medidas": conteo["medidas"],
            "categorias": [
                {"categoria": f["categoria"], "senal": f["senal"], "veredicto": f["veredicto"]}
                for f in inventario["inventario"]
            ],
        },
        "D_senal_residual": {
            "clase": "posicion en el documento",
            "por_que_no_la_representa_v03": (
                "TfidfVectorizer extrae n-gramas sin registrar la posicion en el documento, y "
                "char_wb hace lo mismo dentro de cada palabra"
            ),
            "complementariedad_measure": {
                "rasgos": {
                    n: {
                        "auc_global": d["auc_global"],
                        "auc_en_estratos": d["auc_medio_estratos"],
                        "p_empirico": d.get("p_empirico"),
                        "veredicto": d["veredicto"],
                    }
                    for n, d in p3.items()
                },
                "permutaciones": 500,
            },
            "por_que_no_es_utilizable": (
                "Las tres complementarias dan deltas negativos y, sobre todo, acierto CERO entre las "
                "celdas modificadas, frente al umbral de detectabilidad de 0.5166"
            ),
        },
        "E_resultado_v15": {
            # Tres filas en el registro: el inventario, la prueba de ortogonalidad y este informe
            # de cierre. Los dos primeros son los que miden, y el tercero consolida. Se cuentan
            # los tres porque es lo que convention las fases previas con su informe final.
            "experimentos": 3,
            "experimentos_que_miden": 2,
            "candidatas_evaluadas": len(p3),
            "candidatas_complementarias": len(complementarias),
            "candidatas_anadidas_a_v03": len(p4["resultados"]),
            "resultados_de_anadicion": [
                {
                    "rasgo": d["rasgo"],
                    "delta": d["delta"],
                    "celdas_modificadas": d["celdas_modificadas"],
                    "acierto_entre_modificadas": d["acierto_entre_modificadas"],
                    "intervalo": d["intervalo"],
                    "supera_deteccion": d["deteccion"],
                    "supera_gate": d["supera_gate"],
                }
                for d in p4["resultados"]
            ],
            "candidatas_adoptadas": gap["candidatas_adoptables"],
            "gastos": gap["gastos"],
            "reproducibilidad": (
                "Control de identidad de V03 exacto en cada ejecucion, con diferencia 0.000e+00, y "
                "las dos verificaciones del null por permutacion con 500 remuestreos y semilla fija"
            ),
        },
        "F_decision": {
            "estado": decision,
            "texto": (
                "CLOSED. V03 continua siendo la mejor solucion reproducible bajo las restricciones "
                "actuales. Las principales familias de representacion, objetivo, arquitectura, "
                "estructura de etiquetas, routing y adaptacion ya fueron exploradas. No queda una "
                "fuente de senal permitida y genuinamente nueva identificada en el input "
                "disponible."
            ),
            "matiz_que_no_se_omite": (
                "La senal residual existe y es demostrable: la posicion discrimina dentro de los "
                "estratos de V03 con valor p empirico inferior a 1 entre 500. Lo que no existe es "
                "una fuente de senal que sea EFFECTIVA, y esa distincion es la que cierra la fase"
            ),
            "prohibicion_de_v16": (
                "No se abre una V16 para buscar microganancias. La seccion 14 prohibe la dinamica "
                "de experimento negativo seguido de reformulacion, y las tres adiciones de V15 son "
                "negativas con acierto cero, que es el caso mas limpio de ese patron"
            ),
        },
    }
    REPORTE_JSON.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _informe(informe, inicio)
    log(f"completado en {round(time.time() - inicio, 1)} s")
    log(
        f"VEREDICTO: {decision} | {informe['E_resultado_v15']['experimentos']} experimentos | "
        f"{len(complementarias)} complementarias de "
        f"{len(p3)} | {len(gap['candidatas_adoptables'])} adoptables | TOXIC se cierra en V03"
    )


def _leer(nombre: str) -> dict[str, Any]:
    """Lee un informe JSON de la fase.

    Args:
        nombre: nombre del fichero dentro de reports.

    Returns:
        El contenido del informe.
    """
    ruta = PROJECT_ROOT / "reports" / nombre
    if not ruta.exists():
        raise SystemExit(f"PARADA OBLIGATORIA: falta {ruta.name}")
    return json.loads(ruta.read_text(encoding="utf-8"))


def _estadisticas() -> dict[str, Any]:
    """Cuenta las adopciones reales del registro.

    El recuento tiene que distinguir la adopcion positiva de la negativa, porque el texto de una
    fila como "NO ADOPTADA en LB" contiene la palabra "adopt" y un conteo ingenuo la daria por
    buena. Se decide por la ausencia del prefijo de negacion.

    Returns:
        Diccionario con las cifras del registro.
    """
    filas = list(csv.DictReader((PROJECT_ROOT / "experiments.csv").open(encoding="utf-8")))
    negativa = re.compile(r"no adopt|not adopt|nunca se adopt|se mantiene v03|no se envi", re.IGNORECASE)
    positiva = re.compile(r"\badopt", re.IGNORECASE)
    adoptadas = 0
    rechazadas = 0
    for f in filas:
        decision = f.get("decision") or ""
        if negativa.search(decision):
            rechazadas += 1
        elif positiva.search(decision):
            adoptadas += 1
    con_f1 = [
        (float(f["f1"]), f["experiment_id"])
        for f in filas
        if f.get("f1") and re.fullmatch(r"\d+\.\d+", f["f1"].strip())
    ]
    mejor = max(con_f1) if con_f1 else (0.0, "n/d")
    registro = {f["experiment_id"]: f for f in filas}
    return {
        "total": len(filas),
        "adoptadas": adoptadas,
        "rechazadas": rechazadas,
        "mejor_local": mejor[0],
        "mejor_local_id": mejor[1],
        "mejor_local_suerte": (registro.get(mejor[1], {}).get("decision") or "n/d")[:160],
    }


def _informe(informe: dict[str, Any], inicio: float) -> None:
    """Escribe el informe de cierre con los seis apartados del enunciado.

    Args:
        informe: veredicto consolidado de la fase.
        inicio: instante de arranque del script.
    """
    a = informe["A_estado_del_proyecto"]
    b = informe["B_cuello_de_botella"]
    d = informe["D_senal_residual"]
    e = informe["E_resultado_v15"]
    f = informe["F_decision"]
    linea: list[str] = [
        "# V15: cierre de la frontera final",
        "",
        f"Generado por `experiments/v15_02_informe_cierre.py` ({EXPERIMENTO}).",
        "",
        "## A. Estado del proyecto",
        "",
        fila(["medida", "valor"]),
        fila(["---"] * 2),
        fila(["V03 en el externo", f"**{LB_V03}**"]),
        fila(["V03 en el desarrollo", f"{REFERENCIA_V03:.16f}"]),
        fila(["mejor resultado local de la historia", f"{a['mejor_resultado_local']} en {a['mejor_resultado_local_id']}"]),
        fila(["suerte de ese mejor local", a["mejor_resultado_local_suerte"][:76]]),
        fila(["experimentos en el registro", str(a["experimentos_en_el_registro"])]),
        fila(["filas con adopcion positiva", str(a["adopciones_positivas"])]),
        fila(["filas con adopcion negativa explicita", str(a["rechazos_explicitos"])]),
        fila(["familias cerradas", str(a["familias_cerradas"])]),
        "",
        (
            "**El mejor resultado local de toda la historia no se adopto**, y el motivo es el dato "
            "mas instructivo del proyecto: 0.8465 en local frente a 0.78163 en el externo, es decir "
            "que la mejor senal local medida **perdio en la competencia**. Por eso V03 sigue siendo "
            "la mejor solucion reproducible, y no por falta de intentos sino por evidencia en "
            "contra."
        ),
        "",
        "### Familias cerradas",
        "",
        fila(["familia", "cierre"]),
        fila(["---"] * 2),
    ]
    for x in a["detalle_de_familias"]:
        linea.append(fila([x["familia"][:56], x["cierre"][:96]]))
    linea += [
        "",
        "## B. Diagnostico del cuello de botella",
        "",
        f"El techo del oraculo es **{b['delta_oraculo']:+.6f}**, y {b['origen_del_techo']}.",
        "",
        f"**{b['por_que_mejorar_hs_no_sirve']}.**",
        "",
        b["formulacion_correcta"],
        "",
        "## C. Inventario de senal",
        "",
        f"Se auditaron **{informe['C_inventario_de_senal']['clases']} clases** de senal, con las",
        "cinco preguntas del enunciado y con la comprobacion de que una columna ausente no es",
        "informacion nueva.",
        "",
        fila(["categoria", "senal", "veredicto"]),
        fila(["---"] * 3),
    ]
    for x in informe["C_inventario_de_senal"]["categorias"]:
        linea.append(fila([x["categoria"][:30], x["senal"][:44], x["veredicto"][:76]]))
    linea += [
        "",
        "## D. Senal residual",
        "",
        f"Queda **una sola clase**: {d['clase']}. {d['por_que_no_la_representa_v03']}.",
        "",
        "Y esa clase **si es complementaria**, que es el matiz que hay que preservar:",
        "",
        fila(["rasgo", "AUC global", "AUC en estratos de V03", "p empirico", "veredicto"]),
        fila(["---"] * 5),
    ]
    for nombre, m in d["complementariedad_measure"]["rasgos"].items():
        linea.append(
            fila(
                [
                    nombre[:32],
                    f"{m['auc_global']:.4f}",
                    f"{m['auc_en_estratos']:.4f}",
                    f"{m['p_empirico']:.4f}",
                    m["veredicto"],
                ]
            )
        )
    linea += [
        "",
        "El null se construye permutando el rasgo dentro de cada estrato 500 veces, conservando su",
        "distribucion y destruyendo su relacion con la etiqueta. Un AUC de 0.5547 con valor p",
        "empirico por debajo de 1 entre 500 significa que **el rasgo ve algo que V03 no ve**.",
        "",
        f"**Y aun asi no sirve: {d['por_que_no_es_utilizable']}.**",
        "",
        "## E. Resultado de V15",
        "",
        (
            f"- Experimentos: **{e['experimentos']}**, de los cuales "
            f"{e['experimentos_que_miden']} miden y el tercero es este informe de cierre."
        ),
        f"- Candidatas evaluadas: **{e['candidatas_evaluadas']}**, de las cuales "
        f"**{e['candidatas_complementarias']}** pasaron la prueba de complementariedad.",
        f"- Candidatas anadidas a V03: **{e['candidatas_anadidas_a_v03']}**.",
        f"- Candidatas adoptadas: **{len(e['candidatas_adoptadas'])}**.",
        "",
        fila(
            [
                "rasgo anadido a V03",
                "delta",
                "celdas modificadas",
                "acierto entre las modificadas",
                "IC95 inferior",
                "IC95 superior",
                "supera el gate",
            ]
        ),
        fila(["---"] * 7),
    ]
    for r in e["resultados_de_anadicion"]:
        linea.append(
            fila(
                [
                    r["rasgo"][:30],
                    f"{r['delta']:+.6f}",
                    str(r["celdas_modificadas"]),
                    f"{r['acierto_entre_modificadas']:.4f}",
                    f"{r['intervalo']['inferior']:+.6f}",
                    f"{r['intervalo']['superior']:+.6f}",
                    "si" if r["supera_gate"] else "no",
                ]
            )
        )
    linea += [
        "",
        f"El umbral de detectabilidad de V8-02 es {UMBRAL_DETECCION:.4f} y el acierto observado es",
        (
            "**0.0000 en las tres**. Es decir, de todas las celdas que el rasgo cambia, **ninguna** "
            "acierta. No es una mejora que no llega al gate: es un cambio que danan sin ambiguedad."
        ),
        "",
        f"**Reproducibilidad**: {e['reproducibilidad']}.",
        "",
        f"**Gastos**: {e['gastos']['submissions']} submissions, "
        f"{e['gastos']['evaluaciones_de_holdout2']} de 3 evaluaciones de holdout2, "
        f"{e['gastos']['consultas_de_leaderboard']} consultas de leaderboard, "
        f"{e['gastos']['segunda_particion']} segunda particion.",
        "",
        "## F. Decision",
        "",
        f"**{f['estado']}**",
        "",
        f"> {f['texto']}",
        "",
        f"**El matiz que no se omite**: {f['matiz_que_no_se_omite']}.",
        "",
        f"**{f['prohibicion_de_v16']}.**",
        "",
    ]
    linea += sello(inicio)
    REPORTE_MD.write_text("\n".join(linea) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
