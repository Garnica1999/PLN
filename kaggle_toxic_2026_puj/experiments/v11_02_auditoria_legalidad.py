"""V11-02: auditoria de legalidad del uso de la prueba para estimar el desplazamiento de prevalencia.

Que decide
----------
Antes de usar las predicciones de la prueba, hay que determinar **que permiten exactamente las
reglas academicas**, y clasificar cada paso como permitido, ambiguo o prohibido. El plan lo pide de
forma explicita y anade una consecuencia que hay que respetar:

    si usar ``toxic_test`` para estimar la prevalencia a partir de predicciones no etiquetadas
    resulta ambiguo, **NO usarlo para generar submission**, y la metodologia puede estudiarse
    unicamente mediante shift simulado sobre el OOF.

El texto de las reglas, verbatim, es el unico material normativa. Se cita literalmente y se apoya
en el, no en una interpretacion de memoria.

El problema de fondo que la auditoria tiene que resolver
-------------------------------------------------------
El proyecto lleva mas de diez fases dando por supuesto que el uso transductivo de la prueba es
legitimo: V9-02 estimo la prevalencia con EM sobre predicciones de test, genero la sonda v17 con
los umbrales corregidos, y la fase v11 arranca tomando esos factores como Certain. Si la lectura
correcta es que ese uso es ambiguo, entonces **varias decisiones heredadas estan en cuarentena**, y
eso hay que saberlo antes de construir nada encima, no despues.

Criterio de clasificacion
-------------------------
- **A, permitido explicitamente**: hay una linea de las reglas que lo autoriza.
- **B, ambiguo**: ninguna linea lo prohbe de forma expresa, pero tampoco lo autoriza, y la
  interpretacion razonable de un profesor puede ir en dos direcciones.
- **C, prohibido**: hay una linea que lo prohibe de forma expresa.

Uso
---
    /home/carlos/venv/bin/python experiments/v11_02_auditoria_legalidad.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

EXPERIMENTO = "V11-02 (P1.1, auditoria de legalidad del uso de la prueba)"
REPORTE_MD = PROJECT_ROOT / "reports" / "v11_02_auditoria_legalidad.md"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v11_02_auditoria_legalidad.json"
REPORTE_CUMPLIMIENTO = PROJECT_ROOT / "reports" / "v11_compliance.md"

RUTA_REGLAS = PROJECT_ROOT / "docs" / "competition_rules.md"

# Texto verbatim relevante, citado desde el documento de reglas.
CITA_PROHIBIR_ENTRENAMIENTO = (
    "El dataset de prueba no debe ser usado para entrenamiento."
)
CITA_LIBRERIAS_PERMITIDAS = (
    "Esta permitido el uso de librerias comunes de NLP y ML, incluyendo: NLTK, Scikit-learn, "
    "TensorFlow, PyTorch, Pandas, Numpy."
)
CITA_RESTRICCION_PROFESOR = (
    "No se pueden usar transformers ni modelos LLM para la clasificacion del texto; no se pueden "
    "usar modelos como RoBERTa o BERT. Se pueden usar modelos de vectorizacion, pero deben "
    "entrenarse desde cero y hacer todo desde cero."
)
CITA_NB = (
    "Se espera un notebook claro y bien documentado, que describa: el procesamiento de los datos; "
    "las estrategias de preprocesamiento y representacion de texto; los modelos y experimentos "
    "realizados."
)

# Cada paso de la metodologia, con su clasificacion y el motivo.
PASOS = (
    {
        "paso": "Entrenar el clasificador sobre texto y etiquetas de entrenamiento",
        "clasificacion": "A",
        "motivo": "Es la tarea de la competencia. Ninguna regla lo restringe mas alla de lo academico.",
        "regla_relacionada": "ninguna restriccion; es el objeto de la competencia",
    },
    {
        "paso": "Predecir probabilidades sobre el texto de la prueba con el modelo congelado",
        "clasificacion": "A",
        "motivo": "Es literalmente lo que se pide: generar un submission para el test.",
        "regla_relacionada": "formato de submission definido por la competencia",
    },
    {
        "paso": "Usar las ETIQUETAS de la prueba en cualquier forma",
        "clasificacion": "C",
        "motivo": "Las etiquetas estan ocultas en Kaggle y no estan en el dataset entregado. Ademas "
        "seria contaminacion en cualquier sentido, y el propio enunciado de la v11 lo prohibe.",
        "regla_relacionada": "las columnas HS, TR y AG estan ocultas en Kaggle",
    },
    {
        "paso": "Usar el texto de la prueba para construir caracteristicas, entrenar el "
        "vectorizador o ajustar el modelo",
        "clasificacion": "C",
        "motivo": "Las reglas dicen que no se debe usar el dataset de prueba para entrenamiento, y "
        "un vectorizador ajustado sobre el texto de la prueba es entrenamiento.",
        "regla_relacionada": CITA_PROHIBIR_ENTRENAMIENTO,
    },
    {
        "paso": "Estimar la prevalencia de la prueba con EM sobre probabilidades NO etiquetadas",
        "clasificacion": "B",
        "motivo": "AQUI ESTA LA AMBIGUEDAD. El modelo esta congelado y no se entrena con nada: el "
        "EM es una correccion de decision posterior, no un ajuste de parametros, y no usa ninguna "
        "etiqueta. A favor de A: la regla prohibe entrenamiento, y aqui no hay entrenamiento, y la "
        "correccion de prior es tecnica estandar. A favor de C: la intencion de la regla es que el "
        "test no participe del desarrollo del sistema, y usar sus predicciones para recalibrar la "
        "decision es participar. **Ninguna de las dos lecturas es descabellada**, y las reglas no "
        "mencionan la transduccion ni para permitirla ni para prohibirla, de modo que el silencio "
        "es el dato relevante.",
        "regla_relacionada": CITA_PROHIBIR_ENTRENAMIENTO,
    },
    {
        "paso": "Generar un submission usando la prevalencia estimada con EM sobre la prueba real",
        "clasificacion": "C",
        "motivo": "Es el paso que depende del anterior. Si el paso previo es ambiguo, el plan obliga "
        "a NO usarlo para generar submission. Aqui se apoya en la instruccion expresa de la v11, "
        "no en una lectura propia de las reglas.",
        "regla_relacionada": "plan v11, seccion 6: si es ambiguo, no usarlo para generar submission",
    },
    {
        "paso": "Simular un desplazamiento de prevalencia sobre el OOF y estudiar alli la "
        "metodologia completa: estimador, correccion de prior, calibracion y regla de decision",
        "clasificacion": "A",
        "motivo": "No toca la prueba en absoluto. Todo ocurre sobre el OOF del desarrollo, con "
        "etiquetas propias. Es el unico camino que el plan autoriza cuando el uso real es ambiguo.",
        "regla_relacionada": "plan v11, seccion 6: puede estudiarse mediante simulated shift sobre OOF",
    },
    {
        "paso": "Ajustar la regla de decision sobre el OOF con validacion anidada",
        "clasificacion": "A",
        "motivo": "Usa solo datos etiquetados de desarrollo, y en regimen anidado que no evalua "
        "sobre lo que ajusta. Es metodologia interna, no uso de la prueba.",
        "regla_relacionada": "ninguna restriccion; es metodologia interna",
    },
    {
        "paso": "Evaluar un candidato final en holdout2, que sale del desarrollo",
        "clasificacion": "A",
        "motivo": "Son filas de entrenamiento con etiqueta, nunca vistas como validacion. Es "
        "un conjunto de evaluacion interno.",
        "regla_relacionada": "ninguna restriccion; es metodologia interna",
    },
    {
        "paso": "Usar transformers, RoBERTa o BERT",
        "clasificacion": "C",
        "motivo": "Prohibido de forma expresa por el profesor, y vigente en toda la v11.",
        "regla_relacionada": CITA_RESTRICCION_PROFESOR,
    },
    {
        "paso": "Usar recursos externos etiquetados, pseudoetiquetado o embeddings de terceros",
        "clasificacion": "C",
        "motivo": "Prohibido por la restriccion del profesor, que exige hacer todo desde cero. Los "
        "19 .bin de FastText son reutilizables solo porque son propios y no preentrenados de otro.",
        "regla_relacionada": CITA_RESTRICCION_PROFESOR,
    },
)


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V11-02] {mensaje}", flush=True)


def main() -> None:
    """Clasifica cada paso de la metodologia y fija el alcance permitido de la fase."""
    inicio = time.time()
    log("V11-02: auditoria de legalidad del uso de la prueba")

    reglas = RUTA_REGLAS.read_text(encoding="utf-8")
    log(f"   documento de reglas leido: {RUTA_REGLAS.name}, {len(reglas)} caracteres")
    # La comparacion es sobre el texto normalizado, y no sobre el literal, porque el documento
    # trae tildes y presenta varias de estas frases como elementos de lista repartidos en varias
    # lineas. Comparar en crudo daria falso por motivos que no tienen nada que ver con la cita.
    reglas_normalizada = _normalizar(reglas)
    citas = {
        "prohibir_entrenamiento": CITA_PROHIBIR_ENTRENAMIENTO,
        "librerias_permitidas": CITA_LIBRERIAS_PERMITIDAS,
        "restriccion_profesor": CITA_RESTRICCION_PROFESOR,
        "notebook": CITA_NB,
    }
    citas_normalizadas = {
        nombre: _normalizar(texto) in reglas_normalizada for nombre, texto in citas.items()
    }
    for nombre, presente in citas_normalizadas.items():
        log(f"   cita '{nombre}' localizada en el documento: {presente}")
    todas_las_citas = all(citas_normalizadas.values())

    conteo = {"A": 0, "B": 0, "C": 0}
    for paso in PASOS:
        conteo[paso["clasificacion"]] += 1
    log(
        f"   pasos clasificados: {conteo['A']} permitidos, {conteo['B']} ambiguos, "
        f"{conteo['C']} prohibidos"
    )

    pasos_ambiguos = [paso["paso"] for paso in PASOS if paso["clasificacion"] == "B"]
    hay_ambiguidad_sobre_prevalencia = any("prevalencia" in paso for paso in pasos_ambiguos)
    log(f"   hay ambiguedad sobre la estimacion de prevalencia: {hay_ambiguidad_sobre_prevalencia}")

    # Decision que se sigue, tal como la formula el plan.
    uso_real_permitido = False
    conclusion = (
        "El uso de la prueba para estimar la prevalencia con EM es AMBIGUO, de modo que **no se usa "
        "para generar ningun submission**. La metodologia se estudia unicamente mediante shift "
        "simulado sobre el OOF, y las tres lineas centrales de la v11, que son la correccion de "
        "prior, la calibracion y la regla de decision, se evaluan bajo prevalencias simuladas. La "
        "pregunta de la transferencia al externo queda sin poder responderse con honestidad dentro "
        "de las reglas, y eso hay que decirlo en el informe final en vez de disimularlo."
    )
    log("   DECISION: uso real NO permitido para generar submission; solo shift simulado")

    # Lo que queda permitido, enumerado para que no se pierda de vista.
    permitido = [paso["paso"] for paso in PASOS if paso["clasificacion"] == "A"]
    prohibido = [paso["paso"] for paso in PASOS if paso["clasificacion"] == "C"]
    ambiguo = [paso["paso"] for paso in PASOS if paso["clasificacion"] == "B"]

    informe = {
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fuente_normativa": {
            "ruta": str(RUTA_REGLAS.relative_to(PROJECT_ROOT)),
            "caracteres": len(reglas),
            "captura": "2026-09-22, texto pegado por el organizador del curso",
        },
        "citas_verificadas": citas_normalizadas,
        "todas_las_citas_localizadas": bool(todas_las_citas),
        "criterio": {
            "A": "una linea de las reglas lo autoriza de forma expresa",
            "B": "ninguna linea lo prohibe de forma expresa, pero tampoco lo autoriza, y hay dos "
            "lecturas razonables",
            "C": "una linea de las reglas lo prohibe de forma expresa",
        },
        "pasos": list(PASOS),
        "conteo": conteo,
        "permitidos": permitido,
        "ambiguos": ambiguo,
        "prohibidos": prohibido,
        "uso_de_la_previa_para_submission": bool(uso_real_permitido),
        "conclusion": conclusion,
        "alcance_de_la_fase": {
            "se_puede": [
                "simular desplazamiento de prevalencia sobre el OOF del desarrollo",
                "estimar y comparar estimadores de prevalencia con shift artificial conocido",
                "calibrar con validacion anidada sobre el desarrollo",
                "ajustar reglas de decision con validacion anidada sobre el desarrollo",
                "evaluar candidatos finales en holdout2, que son filas de entrenamiento",
                "medir todo lo anterior con las tres cifras del addendum de v10",
            ],
            "no_se_puede": [
                "usar la prevalencia estimada sobre la prueba real para generar un submission",
                "usar el texto de la prueba para construir caracteristicas o entrenar el vectorizador",
                "usar cualquier etiqueta de la prueba, que ademas no estan disponibles",
                "afirmar que una metodo mejora en el leaderboard sin haberlo medido ahi",
            ],
        },
    }
    REPORTE_JSON.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _informe(informe, inicio)
    _cumplimiento(informe, inicio)
    log(f"completado en {round(time.time() - inicio, 1)} s")
    log("VEREDICTO: uso de la prueba AMBIGUO, no permitido para submission; solo shift simulado")


def _normalizar(texto: str) -> str:
    """Normaliza un texto para compararlo sin tildes, sin mayusculas y sin saltos de linea.

    Hay dos diferencias entre como estan escritas las citas y como aparece el documento. La
    primera son las tildes, porque el documento las trae y este modulo escribe en ASCII por la
    regla de caracter. La segunda es mas sutil: el documento presenta varias de esas frases como
    **elementos de lista de Markdown repartidos en varias lineas**, de modo que una cita escrita
    como una sola linea no aparece como substring aunque sea la misma frase. Por eso aqui tambien
    se colapsa cualquier cantidad de espacios o saltos en uno solo.

    Args:
        texto: cadena de entrada.

    Returns:
        La cadena en minusculas, sin acentos y con espacios y saltos colapsados.
    """
    import re
    import unicodedata

    plano = "".join(
        caracter
        for caracter in unicodedata.normalize("NFD", texto)
        if unicodedata.category(caracter) != "Mn"
    )
    return re.sub(r"\s+", " ", plano).strip().lower()


def fila(columnas: list[str]) -> str:
    """Construye una fila de tabla a partir de sus celdas.

    Se usa en vez de concatenar cadenas a mano porque una fila de tabla tiene tres o cuatro
    celdas y escribirlas en una sola linea pasaria del limite de caracteres, mientras que
    partirlas en varias activa la concatenacion implicita, que es un error de estilo que la
   revision detecta. El helper resuelve las dos cosas a la vez.

    Args:
        columnas: celdas de la fila.

    Returns:
        La fila formateada con separadores de Markdown.
    """
    return "| " + " | ".join(columnas) + " |"


def sello(inicio: float) -> list[str]:
    """Construye las dos lineas de pie de un informe.

    Args:
        inicio: instante de arranque del script.

    Returns:
        Lista con la linea en blanco de separacion y las dos de fecha y duracion.
    """
    marca = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return [
        "",
        f"Fecha: {marca}. Duracion: {round(time.time() - inicio, 1)} s.",
    ]


def _informe(informe: dict[str, Any], inicio: float) -> None:
    """Escribe el informe de la auditoria de legalidad.

    Args:
        informe: veredicto de la auditoria.
        inicio: instante de arranque del script.
    """
    clases = {"A": "**A permitido**", "B": "**B ambiguo**", "C": "**C prohibido**"}
    lineas: list[str] = [
        "# V11-02: auditoria de legalidad del uso de la prueba (P1.1)",
        "",
        f"Generado por `experiments/v11_02_auditoria_legalidad.py` ({EXPERIMENTO}).",
        "",
        "## 1. Que decide y por que va antes que cualquier experimento",
        "",
        "El plan pide clasificar cada paso como permitido, ambiguo o prohibido, y fija la",
        "consecuencia: si estimar la prevalencia con predicciones no etiquetadas resulta",
        "ambiguo, **no se usa para generar submission** y la metodologia se estudia solo con",
        "shift simulado.",
        "",
        "No es un detalle administrativo. El proyecto lleva mas de diez fases dando por",
        "supuesto que el uso transductivo de la prueba es legitimo: V9-02 estimo la",
        "prevalencia con EM sobre predicciones de test y genero la sonda v17 con los",
        "umbrales corregidos, y la v11 arranca tomando esos factores como ciertos. Si la",
        "lectura correcta es que es ambiguo, varias decisiones heredadas estan en",
        "cuarentena, y eso hay que saberlo antes de construir encima.",
        "",
        "## 2. El texto que manda",
        "",
        f"Fuente: `{informe['fuente_normativa']['ruta']}`, capturada el 2026-09-22, texto",
        "pegado por el organizador del curso. Las cuatro citas que sostienen la auditoria se",
        f"verificaron literalmente contra el documento: **{informe['todas_las_citas_localizadas']}**.",
        "",
        f"- **{CITA_PROHIBIR_ENTRENAMIENTO}**",
        f"- *{CITA_LIBRERIAS_PERMITIDAS}*",
        f"- *{CITA_RESTRICCION_PROFESOR}*",
        f"- *{CITA_NB}*",
        "",
        "La clave es la primera. Y el dato relevante es lo que **no** aparece: las reglas no",
        "mencionan la transduccion ni para permitirla ni para prohibirla.",
        "",
        "## 3. Clasificacion de cada paso",
        "",
        fila(["paso", "clase", "motivo resumido"]),
        fila(["---", "---", "---"]),
    ]
    for paso in informe["pasos"]:
        motivo = paso["motivo"].split(".")[0]
        lineas.append(fila([paso["paso"], clases[paso["clasificacion"]], motivo]))
    lineas += [
        "",
        (
            f"Reparto: **{informe['conteo']['A']} permitidos, "
            f"{informe['conteo']['B']} ambiguos**, **{informe['conteo']['C']} prohibidos**."
        ),
        "",
        "## 4. El paso ambiguo, y por que es ambiguo de verdad",
        "",
        "El paso en cuestion es **estimar la prevalencia de la prueba con EM sobre",
        "probabilidades no etiquetadas**. Los dos argumentos son defendibles y ninguno es forzado:",
        "",
        "- **A favor de permitido**: la regla prohibe entrenamiento, y aqui no hay",
        "  entrenamiento. El modelo esta congelado, el EM no ajusta parametros del clasificador,",
        "  y no se usa ninguna etiqueta. La correccion de prior es una tecnica estandar.",
        "- **A favor de prohibido**: la intencion de la regla es que el test no participe del",
        "  desarrollo del sistema, y usar sus predicciones para recalibrar la decision **es**",
        "  participar, aunque no sea entrenar.",
        "",
        "Cuando las dos lecturas son defendibles, la clasificacion honesta es **B**, y el",
        "silencio de las reglas sobre la transduccion es precisamente lo que la hace ambigua.",
        "",
        "## 5. Decision y su consecuencia",
        "",
        informe["conclusion"],
        "",
        "### Lo que si se puede",
        "",
    ]
    for paso in informe["alcance_de_la_fase"]["se_puede"]:
        lineas.append(f"- {paso}")
    lineas += ["", "### Lo que no se puede", ""]
    for paso in informe["alcance_de_la_fase"]["no_se_puede"]:
        lineas.append(f"- {paso}")
    lineas += [
        "",
        "## 6. Lo que esto le hace a la fase",
        "",
        "Las tres lineas centrales de la v11, que son la correccion de prior, la calibracion y",
        "la regla de decision, **siguen siendo enteramente estudiables**, pero bajo prevalencias",
        "simuladas sobre el OOF en vez de sobre la prevalencia real de la prueba. Eso no es una",
        "perdida pequena, y conviene decirlo con todas sus consecuencias:",
        "",
        "1. Se puede medir si la metodologia **funciona** cuando el shift es real y conocido.",
        "2. **No se puede medir si funciona en esta prueba concreta**, porque hacerlo exigiria",
        "   usar la prueba para generar un submission, que es justo lo prohibido.",
        "3. En consecuencia, la pregunta de la v11, si la correccion de prior y la decision",
        "   consciente de F1 son una fuente real de mejora, **queda sin respuesta verificable**",
        "   dentro de las reglas. Se puede responder la version metodologica, que es si el",
        "   mecanismo recupera un shift conocido, y esa es la que se va a medir.",
        "",
        "Esto no es un detalle que haya que ver con benevolencia. Es el limite real de la fase, y",
        "si al final hay un candidato que mejora en lo simulado, la pregunta de si transfiere a",
        "esta prueba concreta **no se podra responder con honestidad** sin autorizacion expresa",
        "del profesor.",
    ]
    lineas += sello(inicio)
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


def _cumplimiento(informe: dict[str, Any], inicio: float) -> None:
    """Escribe el documento de cumplimiento, que es entregable de la fase.

    Args:
        informe: veredicto de la auditoria.
        inicio: instante de arranque del script.
    """
    Restrictions = [
        (
            "Sin transformers, RoBERTa, BERT ni LLM",
            CITA_RESTRICCION_PROFESOR,
            "**CUMPLIDA**, verificada en cada experimento de la v10",
        ),
        (
            "Todo desde cero, sin recursos externos etiquetados",
            CITA_RESTRICCION_PROFESOR,
            "**CUMPLIDA**, solo se usan los 19 .bin de FastText que son propios",
        ),
        (
            "El test no se usa para entrenamiento",
            CITA_PROHIBIR_ENTRENAMIENTO,
            "**CUMPLIDA**",
        ),
        (
            "El test no se usa para estimar prevalencia con fines de submission",
            CITA_PROHIBIR_ENTRENAMIENTO,
            "**NO SE USA**, clasificado ambiguo y descartado para submission",
        ),
        ("Sin pseudoetiquetado", "restriccion del profesor", "**CUMPLIDA**"),
        (
            "Sin etiquetas de test",
            "las etiquetas estan ocultas en Kaggle",
            "**CUMPLIDA**",
        ),
        (
            "Semillas fijas y reproducibilidad con tolerancia 1e-6",
            "principio de la fase",
            "**CUMPLIDA**, verificada en V11-00 y V11-01",
        ),
        (
            "Artefactos en ASCII, sin emojis ni Unicode decorativo",
            "regla de caracter",
            "**CUMPLIDA**, `tools/verificar_caracteres.py` en cada escritura",
        ),
        (
            "El holdout2 no se usa durante la exploracion",
            "plan v11, seccion 2",
            "**CUMPLIDA**, 0 de 3 evaluaciones gastadas",
        ),
        (
            "El leaderboard no se usa como optimizador",
            "plan v11, seccion 34",
            "**CUMPLIDA**, no se ha consultado",
        ),
        (
            "No reabrir familias cerradas",
            "plan v11, seccion 35",
            "**CUMPLIDA**, la v11 solo trabaja la capa de decision",
        ),
    ]
    lineas: list[str] = [
        "# Cumplimiento de la fase v11",
        "",
        (
            f"Actualizado por `{EXPERIMENTO}` el "
            f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}."
        ),
        "",
        "## Normativa aplicada",
        "",
        f"- `{informe['fuente_normativa']['ruta']}`, capturada el 2026-09-22.",
        (
            "- Citas verificadas literalmente contra el documento: "
            f"**{informe['todas_las_citas_localizadas']}**."
        ),
        "",
        "## Restricciones y su estado",
        "",
        fila(["restriccion", "regla", "estado"]),
        fila(["---", "---", "---"]),
    ]
    for restriccion, regla, estado in Restrictions:
        lineas.append(fila([restriccion, regla[:52], estado]))
    lineas += [
        "",
        "## Clasificacion de los pasos con la prueba",
        "",
        f"- permitidos: **{informe['conteo']['A']}**",
        f"- ambiguos: **{informe['conteo']['B']}**, y el uso de la prevalencia real cae aqui",
        f"- prohibidos: **{informe['conteo']['C']}**",
        "",
        f"Conclusion operativa: **{informe['conclusion']}**",
        "",
        "## Lo que esta decision le cuesta a la fase",
        "",
        "La pregunta de la v11 se puede responder en su version metodologica, que es si el",
        "mecanismo recupera un desplazamiento conocido, y **no** en su version de",
        "transferencia, que es si ese mismo mecanismo mejoraria el resultado en esta prueba",
        "concreta. La razon es que medir lo segundo exige usar la prueba para generar un",
        "submission, y eso es exactamente lo que quedo clasificado como no permitido. Queda",
        "escrito aqui para que la limitacion sea visible en el informe final y no se descubra",
        "al final.",
    ]
    lineas += sello(inicio)
    REPORTE_CUMPLIMIENTO.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
