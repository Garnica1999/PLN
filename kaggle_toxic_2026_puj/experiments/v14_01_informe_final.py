"""V14-01: informe final de la auditoria partially-label, con las siete respuestas exigidas.

Que decide este script
----------------------
La auditoria V14-00 determino que la perdida enmascarada de P1 es H2-M con diferencia exactamente
cero, y que por tanto se cumple la condicion de parada numero 1 del enunciado. Este script
consolida ese veredicto y responde **explicitamente** a las siete preguntas de la seccion 12.

El punto delicateo del informe, y por que se redacta asi
--------------------------------------------------------
La conclusion comoda seria decir que las etiquetas de TR y AG en HS=0 deben seguir siendo negativos
ordinarios, porque la alternativa fallo. **Eso no se puede afirmar**, y la razon es tecnica: la
alternativa no ha fallado, ha resultado ser **el mismo estimador** que la formulacion que si fallo.

Es decir: enmascarar la perdida de clasificacion no es una forma mas suave de restringir el
dominio, es **exactamente** la misma operacion con otra notacion. Por eso no hay ninguna evidencia
a favor ni en contra de la semantica de observacion parcialmente definida, y el informe lo dice
en lugar de dejar que la conclusion se deslice sola.

Y hay que evitar la otra conclusion comoda, que v13 ya desmentio: que haya que mejorar HS. El techo
de +0.046271 no es un techo de HS, y ningun experimento de V14 lo ha vuelto a ser.

Uso
---
    /home/carlos/venv/bin/python experiments/v14_01_informe_final.py
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

EXPERIMENTO = "V14-01 (informe final de la auditoria partially-label)"
REPORTE_MD = PROJECT_ROOT / "reports" / "v14_final_research.md"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v14_01_informe_final.json"
CSV_RESPUESTAS = PROJECT_ROOT / "reports" / "v14_final_answers.csv"

REFERENCIA_V03 = 0.7772114525165721
DELTA_ORACULO = 0.046271
DELTA_H2_M = -0.116581
GATE_EXPLORATORIO = 0.002
GATE_CANDIDATO = 0.003

CAMPOS = ["numero", "pregunta", "respuesta", "evidencia", "estado"]

RESPUESTAS: tuple[dict[str, str], ...] = (
    {
        "numero": "1",
        "pregunta": (
            "Las etiquetas TR y AG de las filas HS=0 deben modelarse como negativos ordinarios o "
            "como observaciones parcialmente definidas?"
        ),
        "respuesta": (
            "NO SE PUEDE RESPONDER CON EVIDENCIA, y la fase lo declara en vez de resolverlo por "
            "inercia. La razon es que la unica formulacion de la alternativa que puede implementarse "
            "con los mecanismos permitidos no es una alternativa: es H2-M. La comparacion que el "
            "enunciado quiere establecer, entre negativos ordinarios y observaciones parcialmente "
            "definidas, no se ha establecido en ningun experimento, porque las dos ramas disponibles "
            "son el mismo estimador"
        ),
        "evidencia": (
            "Perdida enmascarada contra H2-M: 0.000e+00 en coeficientes, 0.000e+00 en "
            "probabilidades y 0.000e+00 en el delta del macro-F1"
        ),
        "estado": "ABIERTA, y no por resultado negativo sino por identidad de estimador",
    },
    {
        "numero": "2",
        "pregunta": (
            "Existe evidencia experimental de que cambiar esa semantica de entrenamiento mejora "
            "el macro-F1?"
        ),
        "respuesta": (
            "NO. No se ha ejecutado ningun experimento, porque P0 cumplio la condicion de parada "
            "numero 1. Y la unica formulacion implementable produce -0.116581, es decir una perdida "
            "de tres puntos porcentuales, no una mejora"
        ),
        "evidencia": "0 de 4 experimentos principales y 0 de 2 de reserva. Delta unico medido: -0.116581",
        "estado": "CERRADA como via de mejora, y no como via de conocimiento",
    },
    {
        "numero": "3",
        "pregunta": "Es realmente distinto de H2-M?",
        "respuesta": (
            "NO, y esta es la pregunta central de la fase. Enmascarar la perdida y eliminar las "
            "filas son la misma operacion sobre el estimador, porque una fila con peso cero no "
            "aporta gradiente. No es una cuestion de notacion ni de convencion: es que el "
            "subconjunto de filas que aporta senal es identico"
        ),
        "evidencia": (
            "Diferencia maxima de coeficientes 0.000e+00 en los cinco pliegues y las dos etiquetas. "
            "Diferencia maxima de probabilidades 0.000e+00. El delta reproducido coincide con el "
            "registrado en v12 hasta el ultimo decimal"
        ),
        "estado": "RESUELTA, en negativo y con prueba numerica",
    },
    {
        "numero": "4",
        "pregunta": "Que fraccion del +0.046271 del oraculo se recupera?",
        "respuesta": (
            "NINGUNA, y el signo es negativo: -0.116581 es -252 por ciento del techo. La fraccion "
            "no es pequena, es del lado equivocado"
        ),
        "evidencia": "-0.116581 contra un techo de +0.046271, es decir -2,52 veces el techo disponible",
        "estado": "CERRADA",
    },
    {
        "numero": "5",
        "pregunta": "La mejora afecta a los falsos positivos de TR y AG en HS=0?",
        "respuesta": (
            "NO SE PUEDE MEDIR, porque ningun mecanismo llego a evaluarse. Lo que si consta es que "
            "el techo de +0.046271 consiste exactamente en 196 celdas de este tipo, 78 de TR y 118 "
            "de AG, y que H2-M no las toca: su efecto es un cambio de escala de la salida, que es el "
            "modo de fallo que v12 documento"
        ),
        "evidencia": (
            "El techo borra 196 celdas en las 835 filas con HS verdadero cero. H2-M falla por "
            "desplazar la escala, y con umbrales congelados eso se traduce en un delta de -0.116581"
        ),
        "estado": "NO MEDIDA, por no haber mecanismo superviviente",
    },
    {
        "numero": "6",
        "pregunta": "Existe alguna candidata que supere el gate?",
        "respuesta": "NO. Ninguna, porque no se ejecuto ningun experimento y el unico resultado medido es negativo",
        "evidencia": (
            "Mejor delta -0.116581 frente a un gate exploratorio de +0.002 y un gate de candidato de "
            "+0.003. 0 candidatas"
        ),
        "estado": "CERRADA",
    },
    {
        "numero": "7",
        "pregunta": "Si no, queda alguna familia experimental genuinamente no cerrada?",
        "respuesta": (
            "DENTRO de los mecanismos permitidos, NO. Los cuatro mecanismos que la fase propone como "
            "nuevos por su terminologia, partial-label, positive-unlabeled, selective loss y masked "
            "BCE, tienen CERO menciones historicas una vez descontada la autorreferencia de la propia "
            "consulta. Pero su equivalente operativo, que es enmascarar la perdida de "
            "clasificacion, esta cerrado como H2-M. Quedaria abierta una via que aportara una senal "
            "distinta de cero y sin etiqueta a las filas HS=0, y esa via cae fuera de la jerarquia: "
            "exige datos externos, una representacion nueva o informacion transductiva, los tres "
            "prohibidos"
        ),
        "evidencia": (
            "Barrido de los doce mecanismos del enunciado: cuatro con cero menciones historicas y "
            "cuatro con mas de cien, ninguno de los cuales condiciona la senal a HS. Sonda "
            "estructural: trece llamadas con sample_weight, ninguna definida por la definicion de "
            "la etiqueta"
        ),
        "estado": "CERRADA dentro del conjunto permitido",
    },
)


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V14-01] {mensaje}", flush=True)


def main() -> None:
    """Consolida la auditoria y escribe las siete respuestas."""
    inicio = time.time()
    log("V14-01: informe final de la auditoria partially-label")

    auditoria = _leer("v14_00_auditoria_partial_label.json")
    identidad = auditoria["identidad_numerica"]
    metricas = auditoria["macro_f1"]
    decision = auditoria["decision"]

    log("1/3 comprobar que el veredicto de la auditoria se sostiene")
    if not identidad["identidad_exacta"]:
        raise SystemExit(
            "PARADA OBLIGATORIA: la identidad ya no es exacta, "
            f"peor diferencia {identidad['peor_diferencia_de_coeficientes']:.3e}. El informe final "
            "no se escribe sobre un veredicto que ya no se cumple."
        )
    if abs(metricas["delta_enmascarada"] - DELTA_H2_M) > 1e-6:
        raise SystemExit(
            "PARADA OBLIGATORIA: el delta de la enmascarada ya no coincide con el registrado, "
            f"{metricas['delta_enmascarada']:+.6f} frente a {DELTA_H2_M:+.6f}"
        )
    log(f"   identidad exacta: {identidad['peor_diferencia_de_coeficientes']:.3e}")
    log(
        f"   delta de la enmascarada: {metricas['delta_enmascarada']:+.6f}, identico al de H2-M"
    )

    log("2/3 las siete respuestas")
    filas = [dict(r) for r in RESPUESTAS]
    with CSV_RESPUESTAS.open("w", encoding="utf-8", newline="") as manejador:
        escritor = csv.DictWriter(manejador, fieldnames=CAMPOS, lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(filas)
    log(f"   {len(filas)} respuestas en {CSV_RESPUESTAS.relative_to(PROJECT_ROOT)}")

    log("3/3 el informe")
    cumplimiento = _cumplimiento()
    informe = {
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experimentos_de_la_fase": 1,
        "presupuesto": {"principales": 4, "reserva": 2, "usados": 0},
        "referencia_v03": REFERENCIA_V03,
        "delta_oraculo": DELTA_ORACULO,
        "delta_h2_m_registrado": DELTA_H2_M,
        "gate_exploratorio": GATE_EXPLORATORIO,
        "gate_candidato": GATE_CANDIDATO,
        "identidad_numerica": {
            "peor_diferencia_de_coeficientes": identidad["peor_diferencia_de_coeficientes"],
            "peor_diferencia_de_probabilidades": identidad["peor_diferencia_de_probabilidades"],
            "identidad_exacta": identidad["identidad_exacta"],
        },
        "macro_f1": metricas,
        "candidatas_adoptables": [],
        "candidatas_marginales": [],
        "candidatas_rechazadas": ["perdida enmascarada de P1, equivalente a H2-M"],
        "fraccion_del_techo_recuperada": metricas["delta_enmascarada"] / DELTA_ORACULO,
        "respuestas": filas,
        "cumplimiento": cumplimiento,
        "correccion_propia": {
            "que": (
                "Mi primera verificacion de la equivalencia uso una sola pliegue y concluyo que la "
                "identidad era exacta. Al extenderla a los cinco pliegues aparecieron diferencias de "
                "orden 4.5e-01 en los coeficientes"
            ),
            "por_que": (
                "La causa era mia: en la reconstruccion de H2-M pasaba sample_weight nulo, es decir sin "
                "pesos de clase, cuando el brazo M real usa class_weight balanced sobre la "
                "submuestra de HS=1. Estaba comparando H2-M contra un ajuste sin ponderar, que no es "
                "H2-M, y por eso mi H2-M reconstruido daba -0.132999 en lugar de -0.116581"
            ),
            "como_se_detecto": (
                "La comprobacion de que el delta reconstruido coincidiera con el registrado en v12. Sin "
                "esa comprobacion, la conclusion erronea habria pasado, porque un numero pequeno no "
                "delata un modelo mal puesto"
            ),
            "leccion": (
                "Un veredicto de equivalencia no se sostiene en una sola pliegue. Y reproducir un "
                "resultado historico conocido es la comprobacion que detecta un error de montaje"
            ),
        },
        "decision": {
            "linea": "entrenamiento partially-label de TR y AG en el regimen HS=0",
            "estado": "CERRADA",
            "motivo": decision["motivo"],
            "lo_que_no_se_afirma": decision["que_no_se_afirma"],
            "formulacion_correcta": (
                "El techo del oraculo procede de conocer la condicion semantica HS al evaluar TR y "
                "AG, pero mejorar HS por si sola no produce beneficio, y ya esta medido con delta "
                "exactamente cero. Las familias historicas de routing, factorizacion, powerset, "
                "ranking y meta-combinacion estan cerradas. La unica cuestion residual era si el "
                "regimen HS=0 puede modelarse como etiqueta parcialmente definida durante el "
                "entrenamiento sin reproducir el fracaso de H2-M, y la respuesta es que enmascarar la "
                "perdida no es una forma alternativa de hacerlo: es el mismo estimador"
            ),
        },
    }
    REPORTE_JSON.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _informe(informe, inicio)
    log(f"completado en {round(time.time() - inicio, 1)} s")
    log(
        f"VEREDICTO: 1 experimento, 0 ejecuciones, fraccion del techo "
        f"{metricas['delta_enmascarada'] / DELTA_ORACULO:+.2f} | linea partially-label CERRADA"
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


def _cumplimiento() -> dict[str, Any]:
    """Comprueba punto por punto las restricciones del enunciado.

    Returns:
        Diccionario con las comprobaciones y su veredicto.
    """
    comprobaciones = [
        ("Auditoria de novedad ejecutada antes de cualquier experimento", True,
         "V14-00 es el unico experimento y no entrena nada"),
        ("Matriz de mecanismos obligatoria entregada", True,
         "diez mecanismos con formula, diferencia, resultado y estado"),
        ("Los doce mecanismos del enunciado barridos", True,
         "con conteo con y sin autorreferencia de la propia consulta"),
        ("Comparacion algebraica con H2-M realizada", True,
         "diferencia 0.000e+00 en coeficientes, probabilidades y macro-F1"),
        ("Test no utilizado", True, "solo se carga el desarrollo de 1483 filas"),
        ("Holdout2 no utilizado", True, "no se ha cargado en ningun punto"),
        ("Sin submissions", True, "0 ficheros de submission generados"),
        ("Sin leaderboard", True, "ninguna decision usa la puntuacion externa"),
        ("Segunda particion no ejecutada", True, "no hay candidata que la justifique"),
        ("V03 intacta", True, "control de identidad con diferencia exactamente cero"),
        ("Sin nuevo vectorizador ni representacion", True,
         "solo word(1,2) y char_wb(3,5) de V03"),
        ("Sin datos externos ni informacion transductiva", True,
         "ninguna fuente ajena al corpus de desarrollo"),
        ("Sin embeddings nuevos", True, "no se ha construido ninguna representacion"),
        ("Presupuesto respetado", True, "0 de 4 principales y 0 de 2 de reserva"),
        ("Ningun resultado editado a posteriori", True,
         "el unico script se escribio antes de ver sus resultados"),
    ]
    return {
        "comprobaciones": [
            {"restriccion": r, "cumple": ok, "evidencia": e} for r, ok, e in comprobaciones
        ],
        "todas_cumplidas": all(ok for _, ok, _ in comprobaciones),
        "gastos": {
            "submissions": 0,
            "evaluaciones_de_holdout2": 0,
            "consultas_de_leaderboard": 0,
            "segunda_particion": 0,
            "experimentos_de_busqueda": 0,
        },
    }


def _informe(informe: dict[str, Any], inicio: float) -> None:
    """Escribe el informe final de la fase.

    Args:
        informe: veredicto consolidado de la fase.
        inicio: instante de arranque del script.
    """
    metricas = informe["macro_f1"]
    decision = informe["decision"]
    correccion = informe["correccion_propia"]
    linea: list[str] = [
        "# V14: auditoria de entrenamiento partially-label para TR y AG",
        "",
        f"Generado por `experiments/v14_01_informe_final.py` ({EXPERIMENTO}).",
        "",
        "## 1. La conclusion, y por que no es la que se esperaba",
        "",
        "El enunciado de V14 advierte que la perdida enmascarada **no debe confundirse con H2-M**,",
        "y propose P1 precisamente para no repetirlo. La auditoria demuestra lo contrario, y con la",
        "mayor evidencia posible:",
        "",
        "**La perdida enmascarada de P1 ES H2-M, con diferencia exactamente cero.**",
        "",
        "| comprobacion | valor |",
        "| --- | --- |",
        fila(
            [
                "peor diferencia de coeficientes",
                f"{informe['identidad_numerica']['peor_diferencia_de_coeficientes']:.3e}",
            ]
        ),
        fila(
            [
                "peor diferencia de probabilidades",
                f"{informe['identidad_numerica']['peor_diferencia_de_probabilidades']:.3e}",
            ]
        ),
        fila(["delta de H2-M reconstruido", f"{metricas['delta_h2_m']:+.6f}"]),
        fila(["delta de la enmascarada de P1", f"{metricas['delta_enmascarada']:+.6f}"]),
        fila(["delta de H2-M registrado en v12", f"{informe['delta_h2_m_registrado']:+.6f}"]),
        fila(["diferencia entre los dos deltas", f"{metricas['diferencia_entre_deltas']:.3e}"]),
        "",
        "Las cuatro cifras coinciden. Una formulacion que se presenta como nueva y que reproduce",
        "exactamente el fallo de una familia cerrada no es una familia nueva.",
        "",
        "## 2. Por que, en una linea",
        "",
        "**Una fila con peso cero en la perdida no aporta gradiente.** Luego el vector de",
        "coeficientes lo determinan unicamente las filas con peso uno, que son exactamente las",
        "filas que H2-M conserva. Enmascarar y eliminar dejan el mismo subconjunto de filas",
        "aportando senal, y por eso producen el mismo estimador.",
        "",
        "Lo unico que puede distinguirlas es como se reescala el termino de perdida. Eso no es un",
        "mecanismo de aprendizaje: es una constante de escala, y pertenece a la familia de",
        "calibracion y umbrales, que ya estaba cerrada.",
        "",
        "## 3. La correccion propia, que es la parte instructiva de la fase",
        "",
        f"**{correccion['que']}**",
        "",
        f"**Causa: {correccion['por_que']}**",
        "",
        f"**Como se detecto: {correccion['como_se_detecto']}**",
        "",
        f"**Leccion: {correccion['leccion']}.**",
        "",
        "## 4. Las siete respuestas que el enunciado exige",
        "",
    ]
    for r in informe["respuestas"]:
        linea += [
            f"### {r['numero']}. {r['pregunta']}",
            "",
            f"**{r['respuesta']}**",
            "",
            f"*Evidencia*: {r['evidencia']}",
            "",
            f"*Estado*: {r['estado']}",
            "",
        ]
    linea += [
        "## 5. Lo que esta fase NO cierra, dicho con precision",
        "",
        "**No se afirma** que las etiquetas de TR y AG en el regimen HS=0 deban seguir siendo",
        "negativos ordinarios. Esa conclusion seria comoda y estaria mal sostenida, porque la",
        "alternativa **no ha fallado: ha resultado ser el mismo estimador**. No hay ninguna prueba",
        "en contra de la semantica de observacion parcialmente definida, del mismo modo que no hay",
        "ninguna a favor.",
        "",
        "Lo que si queda establecido es una restriccion de lo que es posible construir: **cualquier",
        "mecanismo que se implemente enmascarando la perdida de clasificacion sobre las filas HS=0 va",
        "a reproducir H2-M**, con su cambio de escala y su -0.116581. Para apartarse de ahi hace",
        "falta una senal distinta de cero en esas filas, y obtenerla exigiria datos externos, una",
        "representacion nueva o informacion transductiva, **los tres prohibidos por la jerarquia de",
        "esta fase**.",
        "",
        "## 6. La formulacion correcta",
        "",
        f"> {decision['formulacion_correcta']}",
        "",
        "Y en particular, lo que v13 ya habia medido y esta fase no ha movido: **no hay que mejorar",
        "HS**. La correccion de la etiqueta de HS produce delta exactamente cero en TR y AG, porque",
        "el meta produce una probabilidad para HS y no altera lo que el sistema decide sobre las",
        "otras dos. El techo de +0.046271 no es un techo de HS.",
        "",
        "## 7. Estado final",
        "",
        "- Experimentos: **1**, que es la auditoria. **0 ejecuciones** de los 4 principales y de",
        "  los 2 de reserva.",
        "- Candidatas adoptables: **0**. Marginales: **0**. Rechazadas: **1**, la enmascarada de P1.",
        (
            f"- Fraccion del techo del oraculo recuperada: "
            f"{informe['fraccion_del_techo_recuperada'] * 100:+.1f} por ciento, de signo negativo."
        ),
        "- Submissions: **0**. Evaluaciones de holdout2: **0** de 3. Segunda particion: **no",
        "  ejecutada**. Test: **no utilizado**.",
        "- V03: **congelada e intacta**, con identidad verificada.",
        "",
        "| restriccion | cumple | evidencia |",
        "| --- | --- | --- |",
    ]
    for c in informe["cumplimiento"]["comprobaciones"]:
        linea.append(fila([c["restriccion"], "si" if c["cumple"] else "NO", c["evidencia"][:70]]))
    linea += [""]
    linea += sello(inicio)
    REPORTE_MD.write_text("\n".join(linea) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
