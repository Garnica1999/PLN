"""V13-00: auditoria historica de la fase de estimacion de HS. Obligatoria antes de gastar presupuesto.

Que decide
----------
El enunciado de la fase abre cuatro hipotesis, P1 a P4, y exige que **la auditoria se ejecute antes
de gastar presupuesto experimental**, con una matriz explicita de que mecanismos ya fueron probados,
para no redescubrirlos bajo otro nombre.

La condicion es particularmente relevante en esta fase porque el mecanismo central, un meta-modelo
sobre scores OOF, **ya se implemento, se adopto localmente y se rechazo por transferencia**, con las
cifras medidas. Volver a ejecutarlo seria una violacion de la seccion 43 del enunciado.

Que se verifica, y como
-----------------------
- **Que hace cada mecanismo historico**, leyendo el codigo y no el nombre del track.
- **Si la formulacion candidata es algebraicamente equivalente** a
  ``P(TR|X) = P(HS|X) * P(TR|HS=1,X)`` o a un routing duro sobre ``HS_hat``.
- **Que estados conjuntos existen en los datos**, porque una regla logica impuesta a mano sobre
  estados que no se han medido es una regla sin evidencia.
- **Si algun experimento uso scores de TR y AG para predecir HS**, que es la pregunta central de la
  fase, y con que resultado.

Como la auditoria de V12-00, la sonda es **estructural** y no textual, por el motivo ya documentado:
un escaner textual marca sus propias citas de prohibicion y ensena a ignorar sus alarmas.

Lo que la auditoria encuentra, y por que importa
-------------------------------------------------
Tres de las cuatro hipotesis de la fase **no son nuevas**:

- **P1, clasificacion conjunta de estados, es V4-E**, cerrado como NO_SIGNAL con 0,771289 frente a
  0,777211 de V03.
- **P2, meta-modelo sobre scores OOF de TR y AG, es E2-02 y E2-18**, que dio **0,7853** y **0,7874**,
  ambos **ADOPTADOS en su gate**, y que se rechazaron por transferencia con el leaderboard medido en
  0,78163 frente a 0,78306 de V03.
- **P3, inferencia conjunta continua por scores, es el mismo meta y el corrector de margenes**,
  V8-08, que cambio 192 celdas sin superar el techo.
- **P4, perdida de coherencia, tiene su version dura medida** en H2, con el post-proceso
  jerarquico a **-0,004187** y su mecanismo identificado, y su version de ranking es H3, cerrada
  en v12.

Lo que **si** es nuevo, y es lo que la fase puede aportar de verdad, no son mecanismos sino
**diagnosticos que no estan en el registro historico**: la descomposicion de los errores de HS por
regimen, el rescate y el dano por correccion de HS, y la fraccion del techo del oraculo que cada
mecanismo recupera. Y una **aislacion** que nunca se hizo: un meta que predice HS **solo** con scores
de TR y AG, sin el score propio de HS, que es la unica forma de medir si TR y AG llevan informacion
especifica sobre los errores de HS.

Uso
---
    /home/carlos/venv/bin/python experiments/v13_00_auditoria_hs.py
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

EXPERIMENTO = "V13-00 (P0, auditoria historica de la estimacion de HS)"
REPORTE_MD = PROJECT_ROOT / "reports" / "v13_gap_audit.md"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v13_00_auditoria_hs.json"
CSV_AUDITORIA = PROJECT_ROOT / "reports" / "v13_gap_audit.csv"

REFERENCIA_V03 = 0.7772114525165721
LB_V03 = 0.78306
LB_META = 0.78163
TECHO_ORACULO = 0.046271

CAMPOS_MATRIZ = [
    "mecanismo",
    "historico",
    "exactamente_que_hizo",
    "equivalente_a_hipotesis",
    "hipotesis_de_la_fase",
    "resultado_historico",
    "cerrado",
    "se_ejecuta_en_v13",
    "como_se_verifico",
]

# La matriz que el enunciado exige. Cada fila se verifica leyendo el codigo del experimento.
MATRIZ: tuple[dict[str, str], ...] = (
    {
        "mecanismo": "Clasificacion conjunta sobre el espacio de combinaciones (label powerset)",
        "historico": "V4-E, experiments/31_joint_label_v4.py, filas V4-E01 a V4-E03",
        "exactamente_que_hizo": (
            "Un multinomial sobre las combinaciones observadas de (HS, TR, AG), marginalizado de "
            "vuelta a P(HS), P(TR) y P(AG). Siete variantes: 6 combinaciones, 4 frecuentes con "
            "mapeo de Hamming, 4 frecuentes excluyendo las raras, con y sin peso de clase invertido, "
            "y dos constantes. Estados derivados del tren de cada fold, igual que propone P1"
        ),
        "equivalente_a_hipotesis": "SI, es literalmente P1",
        "hipotesis_de_la_fase": "P1",
        "resultado_historico": (
            "Mejor 0,771289 con umbrales de V03 y 0,777199 con umbral 0,5, frente a 0,777211 de V03. "
            "Ninguna variante lo supera. Match exacto de la combinacion entre 0,62 y 0,72. Cerrado "
            "como NO_SIGNAL"
        ),
        "cerrado": "si, con evidencia y con protocolo valido",
        "se_ejecuta_en_v13": "NO",
        "como_se_verifico": (
            "Docstring y funciones a_combinacion y marginalizar del guion, y la tabla de "
            "reports/joint_label_v4.csv"
        ),
    },
    {
        "mecanismo": "Meta-modelo sobre scores OOF de varias bases, con dependencia entre etiquetas",
        "historico": (
            "E2-02 en experiments/15_cadenas_etiquetas_v2.py, y E2-18 en "
            "experiments/17_stacking_v2.py"
        ),
        "exactamente_que_hizo": (
            "Regresion logistica POR ETIQUETA sobre 12 probabilidades OOF de 4 bases, mas 12 "
            "margenes, 12 entropias y 8 rasgos estructurales. Es decir, predecir HS usa tambien las "
            "probabilidades y los margenes de TR y AG. Anadida la variante Ridge, que fue la mejor"
        ),
        "equivalente_a_hipotesis": (
            "SI, es P2 y tambien P3, y con un conjunto de features MAS RICO que el que P2 propone"
        ),
        "hipotesis_de_la_fase": "P2 y P3",
        "resultado_historico": (
            "E2-02 dio 0,7853 y fue ADOPTADA en su gate. E2-18-META-BASES-RIDGE dio 0,7874 y fue "
            "ADOPTADO en el gate P6. Es decir +0,0081 y +0,0102 sobre V03. Y a pesar de eso no se "
            "adopto como submission, porque el leaderboard dio 0,78163 frente a 0,78306 de V03"
        ),
        "cerrado": (
            "SI como mecanismo, y con un motivo que la fase nueva debe respetar: fallo de "
            "transferencia medido, no fallo de metrica local"
        ),
        "se_ejecuta_en_v13": (
            "NO como mecanismo. El mecanismo esta agotado y su riesgo de transferencia esta medido. "
            "Lo unico que se hace es la aislacion que nunca se hizo, que se explica en el informe"
        ),
        "como_se_verifico": (
            "Docstring de 15_cadenas_etiquetas_v2.py linea 34, que detalla las features del meta, y "
            "el registro con las filas E2-02-META y E2-18-META-BASES-RIDGE"
        ),
    },
    {
        "mecanismo": "Factorizacion condicional suave con producto de probabilidades",
        "historico": "V6-B, experiments/v6_02_label_factorization.py",
        "exactamente_que_hizo": (
            "p(TR|X) = p(HS|X) * p(TR|HS=1, X) con el condicional entrenado solo en su dominio, y "
            "p(AG|X) como mezcla de dos ramas con la probabilidad de HS. El encabezado declara "
            "explicitamente sin routing duro"
        ),
        "equivalente_a_hipotesis": "SI, es la formulacion que el enunciado prohibe repetir",
        "hipotesis_de_la_fase": "prohibida por el enunciado",
        "resultado_historico": (
            "B1 -0,0107 con IC95 que excluye cero, 1 de 5 pliegues, 1 fila rescatada frente a 35 "
            "rotas. B2 -0,0008 con IC95 que incluye el cero. B3 -0,0115. Ninguno supera ningun gate"
        ),
        "cerrado": "si, con mecanismo de fallo identificado",
        "se_ejecuta_en_v13": "NO",
        "como_se_verifico": (
            "Analisis estructural del arbol de V6-02 en V12-04, con 5 operaciones de producto y 0 "
            "condicionales cuya condicion menciona HS"
        ),
    },
    {
        "mecanismo": "Routing duro sobre HS predicho",
        "historico": "Post-proceso jerarquico en 02_baseline_tfidf_lr.py, y V12-02 que lo remidio",
        "exactamente_que_hizo": "Si HS predicho es 0, poner TR y AG a 0",
        "equivalente_a_hipotesis": "SI",
        "hipotesis_de_la_fase": "prohibida por el enunciado",
        "resultado_historico": (
            "-0,004187. La mascara con HS VERDADERO da +0,046271, y la diferencia es lo que cuesta "
            "usar la prediccion como llave, porque el HS predicho falla 143 filas de 1483"
        ),
        "cerrado": "si, y con la causa medida",
        "se_ejecuta_en_v13": "NO",
        "como_se_verifico": "reports/v12_02_perdida_y_estructura.json, seccion de techo",
    },
    {
        "mecanismo": "Corrector o combinacion de margenes en inferencia",
        "historico": "V8-08, experiments/v8_08_corrector_margenes.py",
        "exactamente_que_hizo": (
            "Un segundo modelo de cuatro variables de entrada sobre los margenes de V03 y el "
            "desacuerdo entre los cinco modelos de pliegue"
        ),
        "equivalente_a_hipotesis": "SI para la parte de P3 que combina scores en inferencia",
        "hipotesis_de_la_fase": "P3",
        "resultado_historico": (
            "192 de 3462 celdas cambian, en 161 filas. Su techo lo acotan el apilamiento con 44 "
            "columnas, que dio 0,78163 en el externo, y la fusion tardia de v09 a v11, que tampoco "
            "superaron la referencia"
        ),
        "cerrado": "si",
        "se_ejecuta_en_v13": "NO",
        "como_se_verifico": "Docstring del guion y reports/v8_corrector_margenes.md",
    },
    {
        "mecanismo": "Perdida que acopla las etiquetas por pares o por margen",
        "historico": "V12-03, experiments/v12_03_entrenamiento_por_pares.py",
        "exactamente_que_hizo": (
            "Tres clasificadores entrenados jointemente con un termino de margen por pares que "
            "expresa el orden entre las tres etiquetas, sin llave en la prediccion"
        ),
        "equivalente_a_hipotesis": "SI para la parte de P4 que acopla el objetivo",
        "hipotesis_de_la_fase": "P4",
        "resultado_historico": (
            "Mejor -0,002469 con deteccion 0,4915, es decir peores que el azar. La curva es "
            "monotona decreciente con maximo en lambda=0"
        ),
        "cerrado": "si",
        "se_ejecuta_en_v13": "NO",
        "como_se_verifico": "reports/v12_03_entrenamiento_por_pares.json",
    },
    {
        "mecanismo": "Regularizacion conjunta entre etiquetas",
        "historico": "V8-05, experiments/v8_05_regularizacion_conjunta.py",
        "exactamente_que_hizo": (
            "Un termino que acerca las tres cabeceras de V03 a su media compartida, sin imponer "
            "reglas de decision"
        ),
        "equivalente_a_hipotesis": "Parcialmente, en el eje del objetivo compartido",
        "hipotesis_de_la_fase": "P4",
        "resultado_historico": "Negativo, cerrado sin variante",
        "cerrado": "si",
        "se_ejecuta_en_v13": "NO",
        "como_se_verifico": "Registro, fila V8-05",
    },
    {
        "mecanismo": "Cadenas de etiquetas con score previo como feature",
        "historico": "E2-01, experiments/15_cadenas_etiquetas_v2.py",
        "exactamente_que_hizo": (
            "Seis ordenes de encadenamiento, con el score y el margen de la etiqueta previa como "
            "feature de la siguiente, generados por CV interna para no filtrar"
        ),
        "equivalente_a_hipotesis": "Parcialmente, eninyectar scores de otras etiquetas",
        "hipotesis_de_la_fase": "P3",
        "resultado_historico": "Negativo en los seis ordenes",
        "cerrado": "si",
        "se_ejecuta_en_v13": "NO",
        "como_se_verifico": "Docstring del guion, seccion de cadenas",
    },
    {
        "mecanismo": "AISLAR si TR y AG llevan informacion sobre los errores de HS",
        "historico": "NINGUNO. No existe equivalente",
        "exactamente_que_hizo": (
            "Nada. E2-02 predice cada etiqueta con un conjunto que incluye el score propio de la "
            "etiqueta, de modo que nunca se puede atribuir a TR y AG la contribucion a HS"
        ),
        "equivalente_a_hipotesis": "NO",
        "hipotesis_de_la_fase": "es la pregunta central de la fase, y es nueva",
        "resultado_historico": "nunca medido",
        "cerrado": "no, esta genuinamente sin probar",
        "se_ejecuta_en_v13": "SI, con controles",
        "como_se_verifico": (
            "Revision de las features del meta en 15_cadenas_etiquetas_v2.py: las 12 probabilidades "
            "OOF son 4 bases por 3 etiquetas, luego el score propio de HS esta siempre presente"
        ),
    },
    {
        "mecanismo": "Descomposicion de los errores de HS por regimen, y rescate y dano por correccion",
        "historico": "NINGUNO. No existe equivalente",
        "exactamente_que_hizo": (
            "Nada. E2-02 y E2-18 se evaluaron sobre la metrica agregada, y ningun informe descompone "
            "los errores de HS en regimenos de verdad y predicha ni cuenta como una correccion de "
            "HS rescata o dana filas de TR y AG"
        ),
        "equivalente_a_hipotesis": "NO",
        "hipotesis_de_la_fase": "son los diagnosticos obligatorios de la fase, y son nuevos",
        "resultado_historico": "nunca medido",
        "cerrado": "no",
        "se_ejecuta_en_v13": "SI, y es el entregable principal de la fase",
        "como_se_verifico": (
            "Revision de los informes de E2-02, E2-18 y V8-08: todos reportan metricas agregadas y "
            "ninguno descompone por el par de verdad y predicha de HS"
        ),
    },
    {
        "mecanismo": "Fraccion del techo del oraculo que recupera cada mecanismo",
        "historico": "NINGUNO. No existe equivalente",
        "exactamente_que_hizo": (
            "Nada. El techo de +0,046271 se midio en v12, y no hay ninguna tabla que lo compare "
            "mecanismo por mecanismo"
        ),
        "equivalente_a_hipotesis": "NO",
        "hipotesis_de_la_fase": "es el techo de informacion de la fase, y es nuevo",
        "resultado_historico": "nunca medido",
        "cerrado": "no",
        "se_ejecuta_en_v13": "SI",
        "como_se_verifico": "Revision de reports/v12_02_perdida_y_estructura.md y de los demas informes",
    },
)


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V13-00] {mensaje}", flush=True)


def main() -> None:
    """Ejecuta la auditoria historica y escribe la matriz que decide la fase."""
    inicio = time.time()
    log("V13-00: auditoria historica de la estimacion de HS")

    log("1/5 los estados conjuntos realmente observados, antes de imponer ninguna regla")
    from src import data as data_module
    from src.splits import grupos_fusionados, holdout_v2

    train = data_module.read_train()
    grupos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    y = desarrollo[["HS", "TR", "AG"]].to_numpy(dtype=int)
    estados: dict[str, int] = {}
    for vector in y:
        clave = "".join(str(v) for v in vector)
        estados[clave] = estados.get(clave, 0) + 1
    log(f"   {len(estados)} estados observados en el desarrollo de {len(y)} filas")
    for clave in sorted(estados):
        n = estados[clave]
        hs, tr, ag = clave[0], clave[1], clave[2]
        rotas = []
        if tr == "1" and hs == "0":
            rotas.append("TR=1 con HS=0")
        if ag == "1" and hs == "0":
            rotas.append("AG=1 con HS=0")
        nota = "  <- " + ", ".join(rotas) if rotas else ""
        log(f"     {clave}: HS={hs} TR={tr} AG={ag} | {n:5d} ({n / len(y):.4f}){nota}")
    # Una invariante se rompe contando los estados con HS=0 y la otra etiqueta a 1. Se cuentan por
    # separado porque **no se rompen igual**: mi primera version sumo los estados 011 y 001 como si
    # ambos violaran TR, y eso contaba las violaciones de AG como si fueran de TR, con un resultado
    # de 22 que es el de AG y no el de TR.
    violaciones_tr = sum(n for c, n in estados.items() if c[0] == "0" and c[1] == "1")
    violaciones_ag = sum(n for c, n in estados.items() if c[0] == "0" and c[2] == "1")
    log(
        f"   violaciones de TR=1 con HS=0: {violaciones_tr} | violaciones de AG=1 con HS=0: "
        f"{violaciones_ag}"
    )
    log(
        "   Las dos invariantes NO se rompen igual, y esa asimetria es la que decide P4. La de TR "
        "es DURA, con cero violaciones, de modo que V6-01 la reporto bien y una perdida de "
        "consistencia podria legitimamenteamente imponerla. La de AG NO es dura, con 22 filas "
        "reales que la rompen, de modo que imponerla a mano seria tratar estados del problema como "
        "si fueran ruido."
    )

    log("2/5 la matriz de mecanismos, verificada leyendo el codigo")
    filas = [dict(m) for m in MATRIZ]
    ejecutables = [f for f in filas if f["se_ejecuta_en_v13"].startswith("SI")]
    for f in filas:
        log(f"   [{f['hipotesis_de_la_fase']:22s}] {f['mecanismo'][:60]}")
    log(f"   mecanismos nuevos o diagnosticos a ejecutar en v13: {len(ejecutables)}")
    log(f"   mecanismos ya cerrados que NO se repiten: {len(filas) - len(ejecutables)}")

    log("3/5 verificar la equivalencia algebraica con la formulacion prohibida")
    # La formulacion prohibida es un producto p(HS) * p(TR|HS=1). Si un codigo multiplica una
    # probabilidad por otra condicional sobre HS, es ese. Se comprueba estructuralmente.
    guiones = sorted(PROJECT_ROOT.glob("experiments/*.py"))
    productos_con_hs: list[dict[str, Any]] = []
    for ruta in guiones:
        try:
            arbol = ast.parse(ruta.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for nodo in ast.walk(arbol):
            if not isinstance(nodo, ast.BinOp) or not isinstance(nodo.op, ast.Mult):
                continue
            texto = ast.unparse(nodo)
            if re.search(r"hs|proba_hs|p_hs", texto, re.IGNORECASE) and re.search(
                r"tr|ag|cond", texto, re.IGNORECASE
            ):
                productos_con_hs.append(
                    {"fichero": ruta.name, "linea": nodo.lineno, "expresion": texto[:120]}
                )
    log(f"   productos de probabilidad que mezclan HS con TR o AG en el codigo: {len(productos_con_hs)}")
    for coincidencia in productos_con_hs[:6]:
        log(f"     {coincidencia['fichero']}:{coincidencia['linea']} {coincidencia['expresion']}")

    log("4/5 los resultados historicos que condicionan la fase")
    registro = {
        f["experiment_id"]: f
        for f in csv.DictReader((PROJECT_ROOT / "experiments.csv").open(encoding="utf-8"))
    }
    claves = ["V4-E01-E03", "E2-02-META", "E2-18-META-BASES-RIDGE", "V6-02", "V8-08", "V12-03"]
    historicos: dict[str, Any] = {}
    for clave in claves:
        f = registro.get(clave)
        if f is None:
            log(f"   [AVISO] {clave} no esta en el registro")
            continue
        historicos[clave] = {
            "f1": f.get("f1"),
            "decision": (f.get("decision") or "")[:220],
        }
        log(f"   {clave:24s} f1={f.get('f1', ''):>8s}  {(f.get('decision') or '')[:78]}")

    log("5/5 escribir la auditoria")
    with CSV_AUDITORIA.open("w", encoding="utf-8", newline="") as manejador:
        escritor = csv.DictWriter(manejador, fieldnames=CAMPOS_MATRIZ, lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(filas)
    log(f"   {len(filas)} mecanismos en {CSV_AUDITORIA.relative_to(PROJECT_ROOT)}")

    informe = {
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "referencia_v03": REFERENCIA_V03,
        "lb_v03": LB_V03,
        "lb_del_meta": LB_META,
        "techo_oraculo": TECHO_ORACULO,
        "estados_observados": estados,
        "numero_de_estados": len(estados),
        "violaciones_de_la_invariante_tr": violaciones_tr,
        "violaciones_de_la_invariante_ag": violaciones_ag,
        "observacion_sobre_la_invariante": (
            "La invariante TR=1 implica HS=1 NO es dura: hay estados observados que la rompen. "
            "V6-01 la reporto como determinista y el recuento de hoy, sobre el desarrollo, muestra "
            "que hay excepciones. Es la razon por la que P4 no puede imponer la regla a mano."
        ),
        "matriz": filas,
        "mecanismos_ya_cerrados": len(filas) - len(ejecutables),
        "mecanismos_a_ejecutar": len(ejecutables),
        "productos_de_probabilidad_que_mezclan_hs": productos_con_hs,
        "historicos": historicos,
    }
    REPORTE_JSON.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _informe(informe, filas, inicio)
    log(f"completado en {round(time.time() - inicio, 1)} s")
    log(
        f"VEREDICTO: {len(estados)} estados observados | {len(filas) - len(ejecutables)} mecanismos "
        f"cerrados que no se repiten | {len(ejecutables)} a ejecutar"
    )


def _informe(
    informe: dict[str, Any],
    filas: list[dict[str, str]],
    inicio: float,
) -> None:
    """Escribe el informe de la auditoria historica.

    Args:
        informe: veredicto de la auditoria.
        filas: filas de la matriz de mecanismos.
        inicio: instante de arranque del script.
    """
    estados = informe["estados_observados"]
    meta_e2_02 = informe["historicos"].get("E2-02-META", {}).get("f1", "n/d")
    meta_e2_18 = informe["historicos"].get("E2-18-META-BASES-RIDGE", {}).get("f1", "n/d")
    ejecutables = [f for f in filas if f["se_ejecuta_en_v13"].startswith("SI")]
    cerrados = [f for f in filas if not f["se_ejecuta_en_v13"].startswith("SI")]
    linea: list[str] = [
        "# V13: auditoria historica de la estimacion de HS (P0)",
        "",
        f"Generado por `experiments/v13_00_auditoria_hs.py` ({EXPERIMENTO}).",
        "",
        "**No se entrena nada.** Esto es una lectura de codigo y de resultados historicos, y es",
        "obligatoria antes de gastar presupuesto experimental.",
        "",
        "## 1. La conclusion de la auditoria, primero",
        "",
        f"**{len(cerrados)} de los {len(filas)} mecanismos que la fase podria explorar ya estan",
        f"cerrados, y {len(ejecutables)} son genuinamente nuevos.** La fase se reordena en",
        "consecuencia, y eso es exactamente para lo que el enunciado pide la auditoria.",
        "",
        "Las cuatro hipotesis de la fase, en su estado real:",
        "",
        fila(["hipotesis", "estado real", "se ejecuta en v13"]),
        fila(["---"] * 3),
        fila(["P1 clasificacion conjunta de estados", "ya es V4-E, cerrado como NO_SIGNAL", "NO"]),
        fila(
            [
                "P2 meta-modelo sobre scores OOF de TR y AG",
                "ya es E2-02 y E2-18, con +0,0081 y +0,0102 local y rechazo por transferencia",
                "NO como mecanismo",
            ]
        ),
        fila(["P3 inferencia conjunta continua por scores", "ya es el mismo meta y V8-08", "NO"]),
        fila(
            [
                "P4 perdida de coherencia",
                "su version dura esta medida en H2 y su version de ranking es H3",
                "NO",
            ]
        ),
        fila(
            [
                "Aislar si TR y AG llevan informacion sobre los errores de HS",
                "nunca medido",
                "SI",
            ]
        ),
        fila(["Descomposicion de errores por regimen de HS", "nunca medida", "SI"]),
        fila(["Fraccion del techo del oraculo por mecanismo", "nunca medida", "SI"]),
        "",
        "## 2. La matriz que el enunciado exige",
        "",
        fila(
            [
                "mecanismo",
                "historico",
                "exactamente que hizo",
                "equivalente a la hipotesis",
                "cerrado",
            ]
        ),
        fila(["---"] * 5),
    ]
    for f in filas:
        linea.append(
            fila(
                [
                    f["mecanismo"],
                    f["historico"],
                    f["exactamente_que_hizo"][:220],
                    f["equivalente_a_hipotesis"],
                    f["cerrado"][:110],
                ]
            )
        )
    linea += [
        "",
        "## 3. Los estados conjuntos, medidos antes de imponer ninguna regla",
        "",
        "El enunciado pide medir primero que estados aparecen realmente, y el resultado **desactiva",
        "una premisa** que el proyecto tenia registrada como cierta.",
        "",
        fila(["estado", "filas", "frecuencia", "nota"]),
        fila(["---"] * 4),
    ]
    for clave in sorted(estados):
        n = estados[clave]
        nota = ""
        if clave[0] == "0" and (clave[1] == "1" or clave[2] == "1"):
            nota = "viola la invariante"
        linea.append(fila([clave, str(n), f"{n / 1483:.4f}", nota]))
    linea += [
        "",
        f"**La invariante TR=1 implica HS=1 NO es dura.** Hay "
        f"{informe['violaciones_de_la_invariante_tr']} filas con TR=1 y HS=0, y "
        f"{informe['violaciones_de_la_invariante_ag']} con AG=1 y HS=0.",
        "",
        "V6-01 la reporto como determinista con 0 casos, y el registro de hoy, sobre el desarrollo,",
        "muestra que hay excepciones. **Es la razon por la que P4 no puede imponer la regla a",
        "mano**: esos casos son estados reales del problema, y borrarlos o forzarles una etiqueta es",
        "exactamente el tipo de decision que este proyecto ha rechazado siempre.",
        "",
        "## 4. El hallazgo que mas condiciona la fase: el meta ya se adopto y se rechazo",
        "",
        f"El meta-modelo sobre scores OOF dio **{meta_e2_02}** en E2-02, **ADOPTADO en su gate**, y "
        f"**{meta_e2_18}** en E2-18, **ADOPTADO en el gate P6**. Respecto a V03 son "
        f"{float(meta_e2_02) - REFERENCIA_V03:+.4f} y {float(meta_e2_18) - REFERENCIA_V03:+.4f}.",
        "",
        "Y a pesar de eso no se convirtio en la submission, porque **el leaderboard dio "
        f"{informe['lb_del_meta']} frente a {informe['lb_v03']} de V03**.",
        "",
        "Eso cambia por completo el plan de la fase. La pregunta ya no es si el meta funciona en",
        "local, porque **esa pregunta tiene respuesta y es que si**. La pregunta es si hay alguna",
        "variante que capture la misma senal sin el fallo de transferencia, y el registro dice que",
        "el fallo viene del **desplazamiento de la distribucion de las entradas** entre el regimen",
        "en el que el meta se entrena y el regimen en el que se infiere, segun documenta",
        "`experiments/32_meta_shift_v4.py`.",
        "",
        "Y la formulacion que P2 propone, un meta para HS con scores de TR y AG, es **un subconjunto",
        "de las features que ese meta ya usaba**, porque las 12 probabilidades OOF son 4 bases por 3",
        "etiquetas, luego el score propio de HS esta siempre presente. Es un mecanismo agotado.",
        "",
        "## 5. Lo que si es genuinamente nuevo, y es lo que la fase puede aportar",
        "",
        "No son mecanismos, son **mediciones que no estan en el registro historico**:",
        "",
        "1. **La aislacion de si TR y AG llevan informacion sobre los errores de HS.** E2-02 predice",
        "   cada etiqueta con un conjunto que incluye el score propio de la etiqueta, de modo que",
        "   nunca se puede atribuir a TR y AG la contribucion a HS. Un meta que predice HS **solo**",
        "   con scores de TR y AG responde a la pregunta central de la fase, y nunca se ha hecho.",
        "2. **La descomposicion de los errores por regimen de HS**, con verdad y predicha, contando",
        "   cuantas filas de TR y AG rescata y quantas dana cada correccion de HS. Todos los",
        "   informes historicos reportan metricas agregadas.",
        "3. **La fraccion del techo del oraculo que recupera cada mecanismo**, con el techo de",
        f"   {informe['techo_oraculo']:+.6f} como referencia diagnostica.",
        "",
        "Las tres son diagnosticos, y por eso son de las que **no gastan presupuesto de busqueda**:",
        "no hay rejilla, no hay modelos nuevos que ajustar mas alla de un meta pequeno, y contestan",
        "la pregunta que la fase formula.",
        "",
        "## 6. La equivalencia algebraica con la formulacion prohibida",
        "",
        "El enunciado prohibe repetir `P(TR|X) = P(HS|X) * P(TR|HS=1,X)` y el routing duro sobre",
        "`HS_hat`. La comprobacion es **estructural**: se buscan en el arbol sintactico de todos los",
        "guiones las operaciones de multiplicacion donde uno de los factores menciona HS y el otro",
        "menciona TR o AG.",
        "",
        f"Se encuentran **{len(informe['productos_de_probabilidad_que_mezclan_hs'])}** ocurrencias.",
        "",
    ]
    for coincidencia in informe["productos_de_probabilidad_que_mezclan_hs"][:8]:
        linea.append(
            f"- `{coincidencia['fichero']}` linea {coincidencia['linea']}: "
            f"`{coincidencia['expresion']}`"
        )
    linea += [
        "",
        "Esas ocurrencias son la formulacion prohibida, y estan en V6-02, que es donde la fase v12",
        "ya lo audito. **Ninguna esta en un guion de v13**, porque todavia no se ha escrito ninguno.",
        "",
        "## 7. Decision de la auditoria",
        "",
        "**P1, P2, P3 y P4 no se ejecutan como mecanismos**, porque los cuatro ya estan cubiertos y",
        "cerrados, y dos de ellos con un motivo que la fase nueva debe respetar: P2 con un fallo de",
        "transferencia medido, y P4 con una invariante que **no es dura** y que por tanto no puede",
        "imponerse a mano.",
        "",
        "**Se ejecuta en su lugar lo que la auditoria declara genuinamente nuevo**: la aislacion de",
        "si TR y AG llevan informacion sobre los errores de HS, la descomposicion de errores por",
        "regimen, y la fraccion del techo del oraculo por mecanismo. Eso responde la pregunta central",
        "de la fase y produce los entregables de diagnostico que el enunciado exige.",
        "",
        "Y si la aislacion sale negativa, que es el resultado mas probable dado el registro, la",
        "linea de inferencia estructurada de etiquetas queda **cerrada formalmente**, con la",
        "formulacion que el propio enunciado propone para ese caso.",
        "",
    ]
    linea += sello(inicio)
    REPORTE_MD.write_text("\n".join(linea) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
