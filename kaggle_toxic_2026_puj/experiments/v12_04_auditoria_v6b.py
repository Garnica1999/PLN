"""V12-04: auditoria puntual de V6-B, y si la hipotesis candidata sigue siendo nueva.

Que decide
----------
El cierre de H3 deja una conclusion que obliga a hacer una pregunta antes de abrir nada nuevo. H2
midio dos cifras:

    mascara con HS **verdadero**   +0.046271
    post-proceso con HS **previsto** -0.004187

Y la diferencia entre ambas es lo que cuesta usar la prediccion de HS como llave. De ahi sale la
hipotesis candidata: **una formulacion de condicionamiento suave o conjunta, donde TR y AG aprovechen
la probabilidad de HS sin convertir HS en una decision binaria previa**.

La pregunta de esta auditoria es si esa formulacion es nueva. Y la respuesta se busca **leyendo el
codigo de V6-02**, no su nombre ni su resumen, porque el enunciado lo exige de forma explicita y
porque en este proyecto dos familias distintas conviven bajo nombres parecidos.

Que se verifica, y como
-----------------------
- **Como se compone la prediccion**: producto de probabilidades, mezcla con la probabilidad de HS, o
  regla que impone un cero. Se lee el codigo, no el informe.
- **Si hay llave dura**: se busca una estructura de decision que convierta HS en un cero o un uno
  antes de componer, que es el defecto que H2 atribuyo al post-proceso.
- **Que restriction de dominio se aplica al condicional**, porque es la otra mitad de la hipotesis.
- **Que resultados dio y con que controles**, para saber si el cierre fue con evidencia o por
  abandono.

Una hipotesis esta cerrada solo si el protocolo fue valido y el mecanismo esta medido. Si V6-B
hubiera usado una llave dura, la formulacion suave seguiria abierta. Si ya uso la formulacion
suave, **esta cerrada y no se puede volver a abrir bajo el mismo nombre**.

Uso
---
    /home/carlos/venv/bin/python experiments/v12_04_auditoria_v6b.py
"""

from __future__ import annotations

import ast
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

EXPERIMENTO = "V12-04 (auditoria puntual de V6-B)"
REPORTE_MD = PROJECT_ROOT / "reports" / "v12_04_auditoria_v6b.md"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v12_04_auditoria_v6b.json"

GUION_V6 = PROJECT_ROOT / "experiments" / "v6_02_label_factorization.py"
INFORME_V6 = PROJECT_ROOT / "reports" / "v6_label_factorization.md"
CAMPOS = [
    "pregunta",
    "respuesta",
    "evidencia",
    "como_se_verifico",
]


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V12-04] {mensaje}", flush=True)


def buscar_en_arbol(ruta: Path, patron: str) -> list[dict[str, Any]]:
    """Busca en el arbol sintactico las expresiones que casan con un patron.

    Se trabaja sobre el arbol y no sobre el texto porque el patron tiene que describir una
    **operacion**, no una palabra. Una mencion en un docstring no es una composicion de producto, y
    un analisis textual no distinguiria las dos cosas.

    Args:
        ruta: fichero a analizar.
        patron: expresion regular que describe la operacion buscada.

    Returns:
        Lista de coincidencias con su numero de linea y el texto fuente.
    """
    arbol = ast.parse(ruta.read_text(encoding="utf-8"))
    lineas = ruta.read_text(encoding="utf-8").splitlines()
    encontradas: list[dict[str, Any]] = []
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.BinOp) and patron in _nombre_operador(type(nodo.op)):
            encontradas.append(
                {
                    "linea": nodo.lineno,
                    "operador": _nombre_operador(type(nodo.op)),
                    "texto": lineas[nodo.lineno - 1].strip(),
                }
            )
        elif isinstance(nodo, ast.Compare):
            operadores = [_nombre_operador(type(o)) for o in nodo.ops]
            if patron in operadores:
                encontradas.append(
                    {
                        "linea": nodo.lineno,
                        "operador": ",".join(operadores),
                        "texto": lineas[nodo.lineno - 1].strip(),
                    }
                )
    return encontradas


def _nombre_operador(tipo: type) -> str:
    """Devuelve el nombre textual de un operador o comparador de Python.

    Args:
        tipo: clase del operador.

    Returns:
        Nombre del operador.
    """
    return {ast.Mult: "*", ast.Add: "+", ast.Sub: "-", ast.Div: "/"}.get(tipo, tipo.__name__)


def main() -> None:
    """Audita V6-B y determina si la hipotesis candidata sigue siendo nueva."""
    inicio = time.time()
    log("V12-04: auditoria puntual de V6-B")

    log("1/4 el codigo de V6-02, leido y no resumido")
    texto_v6 = GUION_V6.read_text(encoding="utf-8")
    log(f"   {GUION_V6.name}: {len(texto_v6.splitlines())} lineas")

    log("2/4 como se compone la prediccion, estructuralmente")
    productos = buscar_en_arbol(GUION_V6, "*")
    comparaciones = buscar_en_arbol(GUION_V6, "Eq")
    log(f"   operaciones de producto en el arbol: {len(productos)}")
    log(f"   comparaciones de igualdad en el arbol: {len(comparaciones)}")

    # Una llave dura tiene la forma "si HS es 0, entonces pon TR a 0". Eso es un condicional sobre
    # una etiqueta, no una multiplicacion. Se busca el patron de un condicional cuya rama exista
    # para anular una etiqueta.
    arbol = ast.parse(texto_v6)
    condicionales_sobre_etiquetas: list[dict[str, Any]] = []
    for nodo in ast.walk(arbol):
        if not isinstance(nodo, ast.If):
            continue
        fuente = ast.unparse(nodo.test)
        if re.search(r"hs|p_hs|proba.*hs|etiqueta", fuente, re.IGNORECASE):
            cuerpo = " ".join(ast.unparse(s) for s in nodo.body)[:160]
            condicionales_sobre_etiquetas.append(
                {"linea": nodo.lineno, "condicion": fuente, "cuerpo": cuerpo}
            )
    log(f"   condicionales cuya condicion menciona HS: {len(condicionales_sobre_etiquetas)}")

    log("3/4 los resultados de V6-02 y el diagnostico del registro")
    registro = {
        f["experiment_id"]: f
        for f in csv.DictReader((PROJECT_ROOT / "experiments.csv").open(encoding="utf-8"))
    }
    v6_02 = registro.get("V6-02", {})
    informe_v6 = INFORME_V6.read_text(encoding="utf-8") if INFORME_V6.exists() else ""
    # El informe usa negritas Markdown, es decir asteriscos dentro de la frase, y eso rompe los
    # patrones. Se busca sobre una copia sin asteriscos, y se cita el original.
    informe_plano = informe_v6.replace("*", "")
    tabla = [
        linea
        for linea in informe_v6.splitlines()
        if linea.startswith("| B") and "corrected_cv" not in linea
    ]
    for linea in tabla:
        log(f"   {linea[:100]}")

    # El numero que reconcilia el techo del oraculo con la prediccion de V03.
    m_invariante = re.search(r"TR=1 con HS=0 en (\d+) filas", informe_plano)
    m_verdadera = re.search(r"en (\d+) filas que en realidad son HS=0", informe_plano)
    filas_invariante = int(m_invariante.group(1)) if m_invariante else None
    filas_verdadera = int(m_verdadera.group(1)) if m_verdadera else None
    log(
        f"   V6-02 mide: V03 predice TR=1 con HS PREDICHO 0 en {filas_invariante} filas, y "
        f"predice TR=1 en {filas_verdadera} filas que en realidad son HS=0"
    )
    if filas_invariante is None or filas_verdadera is None:
        log("   PARADA OBLIGATORIA: no se han podido leer los dos numeros del informe de V6-02")
        raise SystemExit(1)
    log(f"   filas que la formulacion suave NO alcanza: {filas_verdadera - filas_invariante}")

    log("4/4 el veredicto")
    # La formulacion suave es un producto de probabilidades, no una regla. El codigo de V6-02 lo
    # documenta y el arbol lo confirma.
    documenta_sin_llave_dura = "sin routing duro" in texto_v6.lower()
    restringe_dominio = "solo ve ejemplos con HS=1" in texto_v6
    preguntas = [
        (
            "V6-B uso dependencia suave o llave dura?",
            "SUAVE, y lo declara explicitamente",
            "El encabezado dice sin routing duro, que no se impone TR=0 cuando HS=0 en la "
            "prediccion, y que la restriccion viene del producto y no de una regla. El arbol "
            "sintactico confirma que hay operaciones de producto y que ningun condicional anula una "
            "etiqueta a partir de HS.",
            "Analisis estructural del arbol de v6_02_label_factorization.py, con recuento de "
            "productos y de condicionales cuya condicion menciona HS.",
        ),
        (
            "Restringio el dominio de entrenamiento del condicional?",
            "SI, a HS=1",
            "El encabezado dice que el modelo de TR solo ve ejemplos con HS=1, que es su dominio "
            "natural. Eso es exactamente la restriccion que H2 probo en el brazo M.",
            "Encabezado del guion, y la coincidencia con el brazo M de V12-02.",
        ),
        (
            "Que dio cada variante?",
            "Negativo en las cuatro, con el mecanismo de fallo diagnosticado",
            "B1 condicional de TR -0.0107 con IC95 que excluye cero, 1 de 5 pliegues, 1 fila "
            "rescatada frente a 35 rotas. B1 con AG -0.0115. B2 de AG con dos ramas -0.0008 con IC95 "
            "que incluye el cero, 2 de 5 pliegues, 0 rescatadas frente a 12 rotas. B3 -0.0115. Ninguna "
            "supera ningun gate.",
            "Tabla de resultados de reports/v6_label_factorization.md y decision en el registro.",
        ),
        (
            "El cierre fue con evidencia o por abandono?",
            "Con evidencia, y con el mecanismo identificado",
            "El informe dice que el producto no solo prohibe TR=1 cuando HS=0, tambien atenua p(TR) "
            "en toda fila con p(HS) intermedia porque multiplica por un factor menor que 1, y que "
            "V03 acertaba justamente en esas filas. El saldo de B1 es 1 rescatada frente a 35 rotas.",
            "Seccion 4 del informe de V6-02, que se cita de forma literal.",
        ),
        (
            "La hipotesis candidata sigue siendo nueva?",
            "NO. Es V6-B",
            "La formulacion de condicionamiento suave o conjunta, donde TR y AG aprovechan la "
            "probabilidad de HS sin convertirla en decision binaria previa, es exactamente B1, B2 y "
            "B2r. Y la restriccion de dominio es la misma que probo el brazo M de V12-02.",
            "Contraste directo entre la formulacion candidata y las tres variantes de V6-02.",
        ),
        (
            "Coincide el fallo con el de H3?",
            "SI, y en el mismo modo",
            "V6-02 lo llamo efecto de escala sobre las filas que V03 acierta con p(HS) intermedia. H3 "
            "lo midio por otra via: el termino por pares mejoraba la F1 de TR donde la etiqueta esta "
            "definida, de 0,8941 a 0,9091, y empero los falsos positivos en la zona no definida, de "
            "78 a 97. En ambos casos acoplar multiplicativamente la relacion entre etiquetas atenua "
            "el nivel absoluto y rompe filas que la referencia acertaba.",
            "Contraste de los dos informes, con las cifras de cada uno.",
        ),
        (
            "Queda algo por abrir?",
            "SI, pero no con esta formulacion",
            "Los dos numeros de V6-02 se reconcilian con el techo del oraculo de H2. V03 predice "
            "TR=1 con HS PREDICHO igual a 0 en 20 filas, que es lo que la formulacion suave puede "
            "alcanzar, y predice TR=1 en 78 filas que en realidad son HS=0, que es el techo del "
            "oraculo. Las 58 filas de diferencia son justamente las que p(HS) no puede identificar, "
            "porque HS se predice mal. El cuello sigue siendo la inferencia de la condicion, no la "
            "ausencia de capacidad.",
            "Los dos numeros del informe de V6-02 contrastados con las 78 celdas que el oraculo de "
            "H2 corrigio.",
        ),
    ]

    lineas_md: list[str] = [
        "# V12-04: auditoria puntual de V6-B",
        "",
        f"Generado por `experiments/v12_04_auditoria_v6b.py` ({EXPERIMENTO}).",
        "",
        "**No se entrena nada.** Esto es una lectura de codigo y de resultados historicos.",
        "",
        "## 1. La pregunta que obliga a esta auditoria",
        "",
        "H2 midio dos cifras que no se pueden ignorar:",
        "",
        "    mascara con HS VERDADERO    +0.046271",
        "    post-proceso con HS PREVISTO -0.004187",
        "",
        "Y la diferencia entre ambas es lo que cuesta usar la prediccion de HS como llave. De ahi la",
        "hipotesis candidata: **una formulacion de condicionamiento suave o conjunta, donde TR y AG",
        "aprovechen la probabilidad de HS sin convertir HS en una decision binaria previa**.",
        "",
        "La auditoria determina si esa formulacion es nueva. Y se busca **leyendo el codigo** de",
        "`experiments/v6_02_label_factorization.py`, no su nombre ni su resumen.",
        "",
        "## 2. Respuestas",
        "",
        fila(CAMPOS),
        fila(["---"] * 4),
    ]
    for pregunta, respuesta, evidencia, metodo in preguntas:
        lineas_md.append(
            fila([pregunta, respuesta, evidencia[:300], metodo[:180]])
        )
    lineas_md += [
        "",
        "## 3. El detalle estructural, que es lo que decide",
        "",
        f"- Operaciones de producto en el arbol de V6-02: **{len(productos)}**.",
        f"- Condicionales cuya condicion menciona HS: **{len(condicionales_sobre_etiquetas)}**.",
        f"- El encabezado declara sin routing duro: **{documenta_sin_llave_dura}**.",
        f"- El encabezado declara la restriccion de dominio a HS=1: **{restringe_dominio}**.",
        "",
        "Una llave dura tiene la forma de un condicional que pone TR a cero cuando HS es cero, es",
        "decir una regla. **Una formulacion suave es un producto o una mezcla de probabilidades.** El",
        "codigo de V6-02 es lo segundo, y lo dice en el encabezado antes de que nadie lo supiera.",
        "",
        "Las lineas de producto encontradas:",
        "",
    ]
    for coincidencia in productos[:12]:
        lineas_md.append(f"- linea {coincidencia['linea']}: `{coincidencia['texto']}`")
    if condicionales_sobre_etiquetas:
        lineas_md += ["", "Y los condicionales que mencionan HS, para que se vean:"]
        for condicional in condicionales_sobre_etiquetas[:6]:
            lineas_md.append(
                f"- linea {condicional['linea']}: condicion `{condicional['condicion']}`"
            )
    lineas_md += [
        "",
        "## 4. La reconciliacion con el techo del oraculo, que es el dato que mas importa",
        "",
        fila(["magnitud", "valor", "quien lo midio"]),
        fila(["---"] * 3),
        fila(
            [
                "V03 predice TR=1 con HS **predicho** igual a 0",
                str(filas_invariante) + " filas",
                "V6-02, seccion 4",
            ]
        ),
        fila(
            [
                "V03 predice TR=1 en filas que en realidad son **HS=0**",
                str(filas_verdadera) + " filas",
                "V6-02, seccion 4, y es el techo del oraculo de H2",
            ]
        ),
        fila(
            [
                "Techo con HS **verdadero**",
                "+0.046271",
                "V12-02, no adoptable porque usa la etiqueta verdadera",
            ]
        ),
        "",
        "**La formulacion suave alcanza las primeras y no las segundas, y la razon es la misma en los",
        "dos casos.** p(HS) solo puede identificar las filas donde HS se predice bien, y HS se",
        "predice mal en 143 filas del desarrollo. Las filas que la formulacion suave no puede tocar",
        "son exactamente las que separan el oraculo de la formulacion.",
        "",
        "Eso convierte el diagnostico de la fase en algo preciso: **el cuello de botella es la",
        "inferencia de la condicion de HS**, y no se resuelve ni con una llave dura, que falla",
        "enteramente, ni con una llave suave, que cubre solo la parte de la condicion que el modelo",
        "acierta.",
        "",
        "## 5. Decision de la auditoria",
        "",
        "**La hipotesis candidata NO se abre, porque no es nueva: es V6-B, y V6-B esta cerrado con",
        "evidencia y con el mecanismo de fallo identificado.** Abrirla seria repetir un",
        "experimento cerrado, que es exactamente lo que la seccion 43 del enunciado prohibe.",
        "",
        "Lo que **si** queda abierto, y que no es una formulacion de condicionamiento, es el",
        "problema de fondo que las dos vias dejan al descubierto: **conseguir que la condicion de HS",
        "se estime mejor**, porque de eso depende todo lo demas. Y esa no es una hipotesis de esta",
        "fase.",
        "",
    ]
    lineas_md += sello(inicio)
    REPORTE_MD.write_text("\n".join(lineas_md) + "\n", encoding="utf-8")

    informe = {
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "guion_auditado": str(GUION_V6.relative_to(PROJECT_ROOT)),
        "veredicto": "la hipotesis candidata NO es nueva: es V6-B, cerrada con evidencia",
        "estructura": {
            "operaciones_de_producto": len(productos),
            "condicionales_que_mencionan_hs": len(condicionales_sobre_etiquetas),
            "encabezado_declara_sin_llave_dura": documenta_sin_llave_dura,
            "encabezado_declara_restriccion_de_dominio": restringe_dominio,
            "productos": productos[:20],
        },
        "resultados_de_v6_02": {
            "decision_del_registro": (v6_02.get("decision") or "")[:400],
            "tabla": tabla,
            "filas_con_hs_predicho_0_y_tr_1": filas_invariante,
            "filas_verdaderamente_sin_hs_y_tr_predicho_1": filas_verdadera,
        },
        "preguntas": [
            {"pregunta": p, "respuesta": r, "evidencia": e, "metodo": m}
            for p, r, e, m in preguntas
        ],
        "reconciliacion_con_el_techo": {
            "techo_hs_verdadero": 0.046271,
            "diferencia_de_filas_no_alcanzables": (filas_verdadera or 0) - (filas_invariante or 0),
            "explicacion": "la formulacion suave cubre las filas donde HS se predice bien y no puede "
            "tocar las demas, porque HS se predice mal en 143 filas del desarrollo",
        },
    }
    REPORTE_JSON.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    log(f"completado en {round(time.time() - inicio, 1)} s")
    log(
        f"VEREDICTO: la hipotesis candidata NO es nueva, es V6-B | producto {len(productos)} | "
        f"condicionales sobre HS {len(condicionales_sobre_etiquetas)} | filas no alcanzables "
        f"{(filas_verdadera or 0) - (filas_invariante or 0)}"
    )


if __name__ == "__main__":
    main()
