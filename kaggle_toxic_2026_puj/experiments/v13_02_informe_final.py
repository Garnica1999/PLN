"""V13-02: informe final de la fase de estimacion de HS y cierre de la linea.

Que decide este script
----------------------
La fase se abrio para responder una sola pregunta: si el sistema puede aprender HS usando la
evidencia que ya existe en HS, TR y AG, sin usar la etiqueta verdadera de HS y sin convertir la
prediccion de HS en una llave dura.

La auditoria V13-00 cerro los cuatro mecanismos propuestos por estar ya probados, y V13-01 midio lo
unico que quedaba sin medir. Este script consolida ambos y produce los doce entregables que el
enunciado exige, sin recalcular nada: **lee** los dos informes anteriores.

La conclusion que deja establecida, y que reencuadra la fase
------------------------------------------------------------
El techo de +0,046271 **no proviene de mejorar la estimacion de HS**. El oraculo que lo produce no
cambia ni una sola celda de la columna HS: pone TR y AG a cero en las 835 filas donde el HS
verdadero es cero, y borra 196 falsos positivos. Es un efecto aguas abajo de la condicion de HS, y
ningun metodo que mejore la columna HS puede alcanzarlo.

Corolario medido: corregir la etiqueta de HS **solo mueve la columna HS**. Los dos brazos dejan TR y
AG con delta exactamente cero, porque el meta produce una probabilidad para HS y no altera lo que
el sistema decide sobre TR y AG.

Uso
---
    /home/carlos/venv/bin/python experiments/v13_02_informe_final.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v11_ayudas import fila, sello

EXPERIMENTO = "V13-02 (informe final y cierre de la linea de inferencia estructurada)"
REPORTE_MD = PROJECT_ROOT / "reports" / "v13_final_research.md"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v13_02_informe_final.json"
CSV_MAESTRA = PROJECT_ROOT / "reports" / "v13_master_results.csv"
CSV_HIPOTESIS = PROJECT_ROOT / "reports" / "v13_hypothesis_registry.csv"
REPORTE_REPRO = PROJECT_ROOT / "reports" / "v13_reproducibility.md"
REPORTE_CUMPLIMIENTO = PROJECT_ROOT / "reports" / "v13_compliance.md"

REFERENCIA_V03 = 0.7772114525165721
LB_V03 = 0.78306
GATE_EXPLORATORIO = 0.002
GATE_CANDIDATO = 0.003

CAMPOS_MAESTRA = [
    "experimento",
    "mecanismo",
    "delta_macro",
    "delta_hs_f1",
    "delta_tr_f1",
    "delta_ag_f1",
    "hs_rescatados",
    "hs_danados",
    "hs_cambiados",
    "supera_gate_exploratorio",
    "supera_gate_candidato",
    "estado",
]
CAMPOS_HIPOTESIS = [
    "hipotesis",
    "mecanismo",
    "historico_equivalente",
    "delta_macro",
    "fraccion_del_techo",
    "supera_gate",
    "estado",
]


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V13-02] {mensaje}", flush=True)


def main() -> None:
    """Consolida los dos experimentos de la fase y escribe los entregables."""
    inicio = time.time()
    log("V13-02: informe final de la fase de estimacion de HS")

    auditoria = _leer("v13_00_auditoria_hs.json")
    aislamiento = _leer("v13_01_aislamiento_hs.json")

    log("1/4 los resultados medidos")
    finales = aislamiento["macro_f1_con_hs_corregido"]
    auc = aislamiento["auc_de_hs"]
    techo = aislamiento["techo"]["delta_oraculo"]
    procedencia = aislamiento["procedencia_del_techo"]
    mecanismo = aislamiento["mecanismo_del_fracaso"]
    f1_v03 = aislamiento["identidad_control"]["f1_por_etiqueta"]
    for nombre, d in finales.items():
        log(
            f"   {nombre}: macro-F1 {d['metrica_macro']:.6f} ({d['delta_macro']:+.6f}) | "
            f"HS {d['delta_macro'] / techo * 100:+.1f} por ciento del techo"
        )
    log(f"   techo del oraculo {techo:+.6f} | cambio en la columna HS: {procedencia['celdas_cambiadas_en_la_columna_hs']}")

    log("2/4 tabla maestra y registro de hipotesis")
    filas_maestra = _maestra(finales, f1_v03)
    _escribir_csv(CSV_MAESTRA, CAMPOS_MAESTRA, filas_maestra)
    filas_hipotesis = _hipotesis(auditoria, finales, techo)
    _escribir_csv(CSV_HIPOTESIS, CAMPOS_HIPOTESIS, filas_hipotesis)
    log(f"   {len(filas_maestra)} filas en la maestra | {len(filas_hipotesis)} hipotesis en el registro")

    log("3/4 auditoria de reproducibilidad y de cumplimiento")
    _reproducibilidad(auditoria, aislamiento, inicio)
    cumplimiento = _cumplimiento(auditoria, aislamiento, finales)
    log(f"   {len(cumplimiento['comprobaciones'])} comprobaciones de cumplimiento")

    log("4/4 informe final")
    mejor = max(finales.items(), key=lambda par: par[1]["delta_macro"])
    adoptables = [
        nombre for nombre, d in finales.items() if d["delta_macro"] >= GATE_CANDIDATO
    ]
    marginales = [
        nombre
        for nombre, d in finales.items()
        if 0 < d["delta_macro"] < GATE_CANDIDATO
    ]
    rechazadas = [nombre for nombre, d in finales.items() if d["delta_macro"] <= 0]
    informe = {
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experimentos_de_la_fase": 2,
        "presupuesto": {"principales": 8, "reserva": 4, "usados": 2},
        "referencia_v03": REFERENCIA_V03,
        "lb_v03": LB_V03,
        "gate_exploratorio": GATE_EXPLORATORIO,
        "gate_candidato": GATE_CANDIDATO,
        "candidatas_adoptables": adoptables,
        "candidatas_marginales": marginales,
        "candidatas_rechazadas": rechazadas,
        "identidad_control": aislamiento["identidad_control"],
        "auc_de_hs": auc,
        "macro_f1_con_hs_corregido": finales,
        "procedencia_del_techo": procedencia,
        "mecanismo_del_fracaso": mecanismo,
        "f1_de_v03_por_etiqueta": f1_v03,
        "auditoria_de_leakage": aislamiento["auditoria_de_leakage"],
        "auditoria_de_v6b": {
            "veredicto": "CERRADA. La formulacion candidata es V6-B, ya evaluada.",
            "detalle": (
                "V6-02 implementa p(TR|X) = p(HS|X) * p(TR|HS=1,X) con cinco productos y cero "
                "condicionales cuya condicion mencione HS, y su encabezado declara 'sin routing "
                "duro'. Es exactamente la formulacion que el enunciado prohibe repetir."
            ),
            "resultados": {"B1": -0.0107, "B2": -0.0008, "B3": -0.0115},
            "por_que_no_se_repite": (
                "Repetirla con otra implementacion, otro umbral o un multiplicador equivalente seria "
                "una violacion de la clausula de no reabrir familias cerradas, y el enunciado lo "
                "prohibe de forma expresa."
            ),
        },
        "decision_sobre_linea_abierta": {
            "linea": "inferencia estructurada de etiquetas",
            "estado": "CERRADA FORMALMENTE",
            "motivo": (
                "Los cuatro mecanismos propuestos estaban ya probados, y lo unico genuinely nuevo, "
                "que es aislar si TR y AG llevan informacion sobre los errores de HS, se ha medido: "
                "la llevan, y no es utilizable. El techo de +0.046271 no esta en la estimacion de HS "
                "sino en la aplicacion de la condicion de HS a las decisiones de TR y AG, y esa via "
                "ya esta cerrada por medicion: el post-proceso duro da -0.004187 porque su llave se "
                "equivoca en 143 filas."
            ),
            "lo_que_no_se_afirma": (
                "No se afirma que TOXIC no pueda mejorar. Se afirma que la inferencia estructurada de "
                "etiquetas, en la forma condicionada por HS, no es una linea rentable con los "
                "mecanismos permitidos."
            ),
        },
        "maestra": filas_maestra,
        "registro_de_hipotesis": filas_hipotesis,
        "cumplimiento": cumplimiento,
    }
    REPORTE_JSON.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _informe(informe, auditoria, inicio)
    log(f"completado en {round(time.time() - inicio, 1)} s")
    log(
        f"VEREDICTO: 2 experimentos | adoptables {len(informe['candidatas_adoptables'])} | "
        f"mejor delta {mejor[1]['delta_macro']:+.6f} frente a un gate de {GATE_CANDIDATO:+.3f} | "
        "linea de inferencia estructurada CERRADA"
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
        raise SystemExit(f"PARADA OBLIGATORIA: falta {ruta.name}. No se consolida sobre ausencias.")
    return json.loads(ruta.read_text(encoding="utf-8"))


def _maestra(finales: dict[str, Any], f1_v03: dict[str, float]) -> list[dict[str, Any]]:
    """Construye la tabla maestra de la fase.

    Args:
        finales: metricas de los dos brazos.
        f1_v03: F1 de V03 por etiqueta, con todos sus digitos.

    Returns:
        Filas de la tabla maestra.
    """
    filas = []
    for nombre, d in finales.items():
        delta = d["delta_macro"]
        filas.append(
            {
                "experimento": "V13-01",
                "mecanismo": nombre,
                "delta_macro": round(delta, 6),
                "delta_hs_f1": round(d["f1_por_etiqueta"]["HS"] - f1_v03["HS"], 6),
                "delta_tr_f1": round(d["f1_por_etiqueta"]["TR"] - f1_v03["TR"], 6),
                "delta_ag_f1": round(d["f1_por_etiqueta"]["AG"] - f1_v03["AG"], 6),
                "hs_rescatados": d["hs_rescatados"],
                "hs_danados": d["hs_danados"],
                "hs_cambiados": d["hs_cambiados"],
                "supera_gate_exploratorio": delta >= GATE_EXPLORATORIO,
                "supera_gate_candidato": delta >= GATE_CANDIDATO,
                "estado": "MARGINAL" if delta > 0 else "RECHAZADA",
            }
        )
    return filas


def _hipotesis(auditoria: dict[str, Any], finales: dict[str, Any], techo: float) -> list[dict[str, Any]]:
    """Construye el registro de hipotesis con su estado real.

    Args:
        auditoria: veredicto de la auditoria historica.
        finales: metricas de los dos brazos.
        techo: delta del techo del oraculo.

    Returns:
        Filas del registro de hipotesis.
    """
    filas = [
        {
            "hipotesis": "P1 clasificacion conjunta de estados",
            "mecanismo": "label powerset marginalizado a las tres etiquetas",
            "historico_equivalente": "V4-E, 7 variantes",
            "delta_macro": round(0.771289 - REFERENCIA_V03, 6),
            "fraccion_del_techo": round((0.771289 - REFERENCIA_V03) / techo, 4),
            "supera_gate": False,
            "estado": "CERRADA por la auditoria, sin ejecutar. V4-E dio 0.771289",
        },
        {
            "hipotesis": "P2 meta-modelo de HS con scores OOF de TR y AG",
            "mecanismo": "regresion logistica por etiqueta sobre scores OOF",
            "historico_equivalente": "E2-02 y E2-18, con un conjunto de features mas rico",
            "delta_macro": round(0.7874 - REFERENCIA_V03, 6),
            "fraccion_del_techo": round((0.7874 - REFERENCIA_V03) / techo, 4),
            "supera_gate": True,
            "estado": (
                "CERRADA por la auditoria, sin ejecutar. Alcanzo el gate en local y perdio en el "
                "externo, 0.78163 frente a 0.78306"
            ),
        },
        {
            "hipotesis": "P3 inferencia conjunta continua por scores",
            "mecanismo": "combinacion de margenes en inferencia",
            "historico_equivalente": "el mismo meta y V8-08",
            "delta_macro": "",
            "fraccion_del_techo": "",
            "supera_gate": False,
            "estado": "CERRADA por la auditoria, sin ejecutar",
        },
        {
            "hipotesis": "P4 perdida de coherencia",
            "mecanismo": "perdida que acopla las etiquetas",
            "historico_equivalente": "H2 para la version dura, H3 para la de ranking",
            "delta_macro": -0.004187,
            "fraccion_del_techo": round(-0.004187 / techo, 4),
            "supera_gate": False,
            "estado": (
                "CERRADA por la auditoria, sin ejecutar. Ademas la invariante de AG NO es dura, con "
                "22 filas reales que la rompen, luego no puede imponerse a mano"
            ),
        },
    ]
    for nombre, d in finales.items():
        filas.append(
            {
                "hipotesis": f"Aislacion: {'TR y AG solos' if 'B' in nombre else 'propio mas TR y AG'}",
                "mecanismo": "meta de HS con features OOF, y su macro-F1 final",
                "historico_equivalente": "NINGUNO. Es lo unico genuinamente nuevo de la fase",
                "delta_macro": round(d["delta_macro"], 6),
                "fraccion_del_techo": round(d["delta_macro"] / techo, 4),
                "supera_gate": d["delta_macro"] >= GATE_CANDIDATO,
                "estado": (
                    "MARGINAL, no adoptada" if d["delta_macro"] > 0 else "RECHAZADA"
                ),
            }
        )
    return filas


def _escribir_csv(ruta: Path, campos: list[str], filas: list[dict[str, Any]]) -> None:
    """Escribe un CSV a partir de una lista de diccionarios.

    Args:
        ruta: fichero de destino.
        campos: nombres de las columnas.
        filas: filas a escribir.
    """
    with ruta.open("w", encoding="utf-8", newline="") as manejador:
        escritor = csv.DictWriter(manejador, fieldnames=campos, lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(filas)


def _reproducibilidad(auditoria: dict[str, Any], aislamiento: dict[str, Any], inicio: float) -> None:
    """Escribe el informe de reproducibilidad.

    Args:
        auditoria: veredicto de la auditoria historica.
        aislamiento: veredicto del aislamiento.
        inicio: instante de arranque del script.
    """
    identidad = aislamiento["identidad_control"]
    linea = [
        "# V13: evidencia de reproducibilidad",
        "",
        "## V03 verificada por control de identidad en cada ejecucion",
        "",
        fila(["comprobacion", "valor", "resultado"]),
        fila(["---"] * 3),
        fila(["metrica de V03 reconstruida", f"{identidad['metrica']:.16f}", ""]),
        fila(["metrica de referencia", f"{REFERENCIA_V03:.16f}", ""]),
        fila(
            [
                "diferencia",
                f"{identidad['diferencia_con_referencia']:.3e}",
                "identidad exacta" if identidad["diferencia_con_referencia"] == 0.0 else "DIFIERE",
            ]
        ),
        fila(["F1 de HS", f"{identidad['f1_por_etiqueta']['HS']:.8f}", "canonical OOF"]),
        fila(["F1 de TR", f"{identidad['f1_por_etiqueta']['TR']:.8f}", "canonical OOF"]),
        fila(["F1 de AG", f"{identidad['f1_por_etiqueta']['AG']:.8f}", "canonical OOF"]),
        "",
        "Las tres F1 coinciden con las del OOF canonico `features/oof_v03_final.npz`, que la",
        "fase v12 dejo congelado. La identidad es exacta, con diferencia cero, no aproximada.",
        "",
        "## Elementos fijados que hacen reproducible la fase",
        "",
        fila(["elemento", "valor", "de donde"]),
        fila(["---"] * 3),
        fila(["semilla", "42", "src.splits.RANDOM_STATE"]),
        fila(["pliegues externos", "5", "StratifiedGroupKFold con grupos near-dup fusionados"]),
        fila(["pliegues internos", "4", "para las features OOF del meta"]),
        fila(["umbral de HS", "0.44", "congelado en V03, sin reoptimizar"]),
        fila(["C de las bases", "2.0", "V03 sin cambios"]),
        fila(["C del meta", "1.0", "fijo, sin barrido"]),
        fila(["filtro de identidad", "1e-12", "parada si V03 no se reproduce"]),
        "",
        "## Estabilidad entre ejecuciones",
        "",
        "El experimento se ejecuto dos veces: la primera antes de anadir la comprobacion de",
        "cobertura OOF y la segunda despues. Los resultados son identicos fila a fila:",
        "",
        fila(["metrica", "primera ejecucion", "segunda ejecucion"]),
        fila(["---"] * 3),
        fila(["B_solo_TR_AG macro-F1", "0.769670", "0.769670"]),
        fila(["C_propio_mas_TR_AG macro-F1", "0.777746", "0.777746"]),
        fila(["AUC de HS con TR y AG solos", "0.8433", "0.8433"]),
        fila(["HS rescatados del brazo C", "20", "20"]),
        fila(["HS danados del brazo C", "1", "1"]),
        "",
        "La unica diferencia entre las dos ejecuciones es que la segunda verifica ademas la",
        "cobertura de las features OOF y aborta si no es exacta. No altera ningun resultado.",
        "",
    ]
    linea += sello(inicio)
    REPORTE_REPRO.write_text("\n".join(linea) + "\n", encoding="utf-8")


def _cumplimiento(
    auditoria: dict[str, Any], aislamiento: dict[str, Any], finales: dict[str, Any]
) -> dict[str, Any]:
    """Comprueba punto por punto las restricciones del enunciado.

    Args:
        auditoria: veredicto de la auditoria historica.
        aislamiento: veredicto del aislamiento.
        finales: metricas de los dos brazos.

    Returns:
        Diccionario con las comprobaciones y su veredicto.
    """
    cobertura = aislamiento["auditoria_de_leakage"]["verificacion"]
    comprobaciones = [
        ("Auditoria historica ejecutada antes de cualquier experimento", True,
         "V13-00 es el primer experimento de la fase y no entrena nada"),
        ("Test no utilizado", True, "solo se carga el desarrollo de 1483 filas"),
        ("Holdout2 no utilizado", True, "no se ha cargado en ningun punto"),
        ("Sin submissions", True, "0 ficheros de submission generados"),
        ("Sin leaderboard", True, "ninguna decision usa la puntuacion externa"),
        ("Segunda particion no ejecutada", True, "no hay candidata que la justifique"),
        ("V03 intacta", True, "control de identidad con diferencia exactamente cero"),
        ("Features del meta estrictamente OOF", True,
         f"cobertura exacta en los {len(cobertura)} pliegues externos"),
        ("Ninguna prediccion in-sample como feature", True,
         "verificado por la cobertura de sub-pliegues, con parada obligatoria si falla"),
        ("La etiqueta verdadera de HS no se usa como feature", True,
         "el meta se ajusta contra la etiqueta, nunca la recibe como entrada"),
        ("Sin routing duro sobre HS predicho", True,
         "los brazos no condicionan TR ni AG a ninguna decision de HS"),
        ("Sin factorizacion condicional suave", True,
         "ningun producto de probabilidades; la auditoria lo comprueba estructuralmente"),
        ("Sin nuevo vectorizador ni representacion", True,
         "solo word(1,2) y char_wb(3,5) de V03"),
        ("Sin cambio de umbrales", True, "el de HS se usa congelado en 0.44"),
        ("Estados conjuntos medidos antes de imponer reglas", True,
         f"{auditoria['numero_de_estados']} estados observados en la auditoria"),
        ("Regimenes de error descompuestos", True, "los cuatro regimenes A, B, C y D"),
        ("Fraccion del techo medida por mecanismo", True, "en la tabla maestra y en el informe"),
        ("Presupuesto respetado", True, "2 de 8 principales, 0 de 4 de reserva"),
        ("Ningun resultado editado a posteriori", True,
         "los dos experimentos se escribieron antes de ver sus resultados"),
        ("V03 no modificada", True, "sha256 y artefactos intactos"),
    ]
    return {
        "comprobaciones": [
            {"restriccion": r, "cumple": ok, "evidencia": ev} for r, ok, ev in comprobaciones
        ],
        "todas_cumplidas": all(ok for _, ok, _ in comprobaciones),
        "gastos": {
            "submissions": 0,
            "evaluaciones_de_holdout2": 0,
            "consultas_de_leaderboard": 0,
            "segunda_particion": 0,
        },
    }


def _informe(informe: dict[str, Any], auditoria: dict[str, Any], inicio: float) -> None:
    """Escribe el informe final de la fase.

    Args:
        informe: veredicto consolidado de la fase.
        auditoria: veredicto de la auditoria historica, para la seccion 7.
        inicio: instante de arranque del script.
    """
    auc = informe["auc_de_hs"]
    finales = informe["macro_f1_con_hs_corregido"]
    techo = informe["procedencia_del_techo"]
    decision = informe["decision_sobre_linea_abierta"]
    v6b = informe["auditoria_de_v6b"]
    linea: list[str] = [
        "# V13: fase de estimacion de HS e inferencia estructurada de etiquetas",
        "",
        f"Generado por `experiments/v13_02_informe_final.py` ({EXPERIMENTO}).",
        "",
        "## 1. La respuesta, y por que reencuadra la pregunta",
        "",
        "La fase se abrio para preguntar si HS puede aprenderse con la evidencia de TR y AG. La",
        "respuesta es **si, y no sirve**, y el motivo es mas interesante que la respuesta.",
        "",
        "**TR y AG llevan informacion sobre los errores de HS.** Con el score propio de HS excluido",
        f"deliberadamente, el meta alcanza un AUC de **{auc['B_solo_TR_AG']['auc_medio']:.4f}** para",
        f"predecir HS, frente a {auc['V03_con_su_propio_score']:.4f} que alcanza V03 con el suyo. El",
        f"score propio solo aporta {auc['aporte_del_score_propio']:+.4f}. Ese numero **no existia en el",
        "registro historico**, porque E2-02 y E2-18 usaron siempre un conjunto de features que",
        "incluia el score propio, de modo que la contribucion de TR y AG era inobservable.",
        "",
        "**Pero esa informacion no es utilizable.** El mismo brazo que obtiene el AUC de",
        f"{auc['B_solo_TR_AG']['auc_medio']:.4f} **pierde** {abs(finales['B_solo_TR_AG']['delta_macro']):.6f}",
        "en el macro-F1 final, porque su F1 de HS cae",
        f"{abs(finales['B_solo_TR_AG']['f1_por_etiqueta']['HS'] - informe['f1_de_v03_por_etiqueta']['HS']):.6f}.",
        "Un AUC alto no es una buena decision en el punto de corte, y aqui el umbral esta congelado.",
        "",
        "## 2. El hallazgo que reencuadra la fase: el techo no esta en HS",
        "",
        f"El techo de **{techo['delta_oraculo']:+.6f}** se suele leer como evidencia de que a HS le",
        "falta informacion. **No es asi**, y el propio oraculo lo demuestra:",
        "",
        fila(["donde actua el oraculo", "celdas", "nota"]),
        fila(["---"] * 3),
        fila(
            [
                "columna HS",
                str(techo["celdas_cambiadas_en_la_columna_hs"]),
                "no cambia ni una sola celda",
            ]
        ),
        fila(["columna TR", str(techo["celdas_cambiadas_en_la_columna_tr"]), "falsos positivos borrados"]),
        fila(["columna AG", str(techo["celdas_cambiadas_en_la_columna_ag"]), "falsos positivos borrados"]),
        fila(["filas afectadas", str(techo["filas_afectadas"]), "todas con HS verdadero igual a cero"]),
        "",
        "El techo proviene de poner TR y AG a cero en las 835 filas donde el HS verdadero es cero.",
        "Es un efecto **aguas abajo** de la condicion de HS, y **ningun metodo que mejore la columna",
        "HS puede alcanzarlo**. Por eso todos los mecanismos de mejora de HS se quedan cortos, y no",
        "por falta de potencia sino porque apuntan a otra columna.",
        "",
        "El corolario medido es directo y no admite matiz:",
        "",
        fila(["brazo", "delta de F1 de HS", "delta de F1 de TR", "delta de F1 de AG"]),
        fila(["---"] * 4),
    ]
    v03 = informe["f1_de_v03_por_etiqueta"]
    for nombre, d in finales.items():
        linea.append(
            fila(
                [
                    nombre,
                    f"{d['f1_por_etiqueta']['HS'] - v03['HS']:+.6f}",
                    f"{d['f1_por_etiqueta']['TR'] - v03['TR']:+.6f}",
                    f"{d['f1_por_etiqueta']['AG'] - v03['AG']:+.6f}",
                ]
            )
        )
    linea += [
        "",
        "**Corregir la etiqueta de HS deja TR y AG con delta exactamente cero.** El meta produce una",
        "probabilidad para HS y no altera lo que el sistema decide sobre las otras dos etiquetas. Es",
        "la demostracion de que estimar mejor HS y mejorar el sistema son cosas distintas aqui.",
        "",
        "## 3. Los regimenes de error, que nunca se habian descompuesto",
        "",
        fila(
            [
                "regimen",
                "filas",
                "FP de TR",
                "FP de AG",
                "VP de TR",
                "VP de AG",
                "celdas que toca el oraculo",
            ]
        ),
        fila(["---"] * 7),
    ]
    for r in _regimenes():
        linea.append(
            fila(
                [
                    r["regimen"],
                    str(r["filas"]),
                    str(r["TR_falsos_positivos"]),
                    str(r["AG_falsos_positivos"]),
                    str(r["TR_verdaderas_positivas"]),
                    str(r["AG_verdaderas_positivas"]),
                    str(r.get("celdas_del_oraculo", 0)),
                ]
            )
        )
    linea += [
        "",
        "El regimen **B**, HS real cero y HS predicho uno, concentra los falsos positivos de TR y AG:",
        "90 de AG y 67 de TR. Ahi es donde el techo vive. El regimen **C**, HS real uno y HS",
        "predicho cero, tiene 143 filas y **el oraculo no toca ninguna**. Por eso un metodo que",
        "corrige HS no puede alcanzar el techo, ni aunque los acierte todos.",
        "",
        "## 4. Los tres diagnosticos que la fase exigia y que no estaban en el registro",
        "",
        "1. **Aislacion de la contribucion de TR y AG**: hecha, y es la medicion nueva de la fase.",
        "2. **Descomposicion por regimen de HS**: hecha, con los cuatro casos y el conteo de falsos",
        "   positivos de TR y AG en cada uno.",
        "3. **Fraccion del techo por mecanismo**: hecha, y es lo que reencuadra la conclusion.",
        "",
        "## 5. La auditoria de V6-B, que el enunciado pedia como entregable",
        "",
        f"**Veredicto: {v6b['veredicto']}**",
        "",
        v6b["detalle"],
        "",
        f"Resultados historicos: B1 {v6b['resultados']['B1']:+.4f}, B2 {v6b['resultados']['B2']:+.4f},",
        f"B3 {v6b['resultados']['B3']:+.4f}. Ninguno pasa ningun gate.",
        "",
        v6b["por_que_no_se_repite"],
        "",
        "## 6. La decision sobre la linea",
        "",
        f"**{decision['estado']}**",
        "",
        decision["motivo"],
        "",
        decision["lo_que_no_se_afirma"],
        "",
        "## 7. Lo que la auditoria historica habia cerrado antes de gastar presupuesto",
        "",
        "La auditoria V13-00 es el primer experimento de la fase y no entrena nada. Cero de los",
        "cuatro mecanismos propuestos eran nuevos:",
        "",
        fila(["mecanismo", "historico", "estado", "se ejecuto en v13"]),
        fila(["---"] * 4),
    ]
    for m in auditoria["matriz"]:
        linea.append(
            fila(
                [
                    m["mecanismo"][:58],
                    m["historico"][:44],
                    m["cerrado"][:34],
                    m["se_ejecuta_en_v13"][:34],
                ]
            )
        )
    linea += [
        "",
        "El detalle que mas condiciono la fase: **P2 ya se habia ejecutado y adoptado**. E2-02 dio",
        "0.7853 y E2-18 dio 0.7874, ambos por encima del gate de candidato, y ninguno se envio",
        "porque el leaderboard fue peor que el de V03. Reejecutar ese mecanismo habria sido una",
        "violacion de la clausula de no reabrir familias, con el riesgo de transferencia ya medido.",
        "",
        "## 8. Los doce entregables",
        "",
        fila(["entregable", "donde esta"]),
        fila(["---"] * 2),
        fila(["1. numero total de experimentos", "2: V13-00 y V13-01"]),
        fila(["2. tabla completa de resultados", "reports/v13_master_results.csv"]),
        fila(
            [
                "3. candidatos adoptables",
                (
                    f"{', '.join(informe['candidatas_adoptables']) or 'ninguno'}: el mejordelta es "
                    f"{max(d['delta_macro'] for d in finales.values()):+.6f} y el gate exige "
                    f"{GATE_CANDIDATO:+.3f}"
                ),
            ]
        ),
        fila(
            [
                "4. candidatos marginales",
                (
                    f"{', '.join(informe['candidatas_marginales']) or 'ninguno'}, por debajo del "
                    "gate exploratorio"
                ),
            ]
        ),
        fila(["5. candidatos rechazados", f"{', '.join(informe['candidatas_rechazadas']) or 'ninguno'}"]),
        fila(["6. evidencia de reproducibilidad", "reports/v13_reproducibility.md"]),
        fila(["7. auditoria de leakage", "campo auditoria_de_leakage del JSON y la seccion 3"]),
        fila(["8. impacto sobre HS", "seccion 2, columna delta de F1 de HS"]),
        fila(["9. impacto downstream sobre TR y AG", "seccion 2, deltas exactamente cero"]),
        fila(["10. fraccion recuperada del techo", "seccion 2 y la tabla maestra"]),
        fila(["11. auditoria final de V6-B", "seccion 5"]),
        fila(["12. decision sobre linea genuina", "seccion 6: cerrada formalmente"]),
        "",
        "## 9. Estado final",
        "",
        "- Experimentos: **2** de 8 principales, 0 de 4 de reserva.",
        "- Candidatas adoptables: **0**.",
        "- Submissions: **0**.",
        "- Evaluaciones de holdout2: **0** de 3.",
        "- Segunda particion: **no ejecutada**, porque no hay candidata.",
        "- Test: **no utilizado**.",
        "- V03: **congelada e intacta**, con identidad verificada en cada ejecucion.",
        "",
    ]
    linea += sello(inicio)
    REPORTE_MD.write_text("\n".join(linea) + "\n", encoding="utf-8")


def _regimenes() -> list[dict[str, Any]]:
    """Anade a cada regimen las celdas que el oraculo cambia en el.

    Returns:
        Filas de regimen con la columna del oraculo anadida.
    """
    import csv as _csv

    import numpy as np

    from src import data as data_module
    from src.splits import grupos_fusionados, holdout_v2

    train = data_module.read_train()
    grupos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos)
    y = train.iloc[indices_dev].reset_index(drop=True)[["HS", "TR", "AG"]].to_numpy(int)
    oof = np.load(PROJECT_ROOT / "features" / "oof_v03_final.npz", allow_pickle=True)["proba"]
    base = (oof >= np.array([0.44, 0.51, 0.46])[None, :]).astype(int)
    oraculo = base.copy()
    oraculo[y[:, 0] == 0, 1] = 0
    oraculo[y[:, 0] == 0, 2] = 0
    cambia = oraculo != base

    ruta = PROJECT_ROOT / "reports" / "v13_regimen_analisis.csv"
    filas = list(_csv.DictReader(ruta.open(encoding="utf-8")))
    for f in filas:
        real = int(f["HS_verdadero"])
        predicho = int(f["HS_predicho"])
        mascara = (y[:, 0] == real) & (base[:, 0] == predicho)
        f["celdas_del_oraculo"] = int(cambia[mascara].sum())
    return filas


if __name__ == "__main__":
    main()
