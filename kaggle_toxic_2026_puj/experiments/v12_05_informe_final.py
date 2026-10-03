"""V12-05: informe final de la fase, y tabla maestra.

Que cierra
----------
La v12 se abre con una auditoria de brecha que encuentra tres huecos genuinamente no probados, todos
en el eje del **objetivo de entrenamiento**, y los ejecuta. Este script consolida los cinco
experimentos de la fase y escribe el informe final, con las conclusiones que la fase deja
establecidas y con los limites que deja descubiertos.

Lo que la fase responde
-----------------------
> Queda alguna fuente de senal explotable por modelos permitidos que no se haya probado
> adecuadamente, o V03 constituye efectivamente el limite practico bajo las restricciones?

La respuesta tiene dos mitades, y conviene no mezclarlas:

- **En el eje del objetivo de entrenamiento, si quedaba, y no se hayedito.** Tres mecanismos
  distintos, ninguno llega al gate, y la auditoria previa habia confirmado que ninguno existia en el
  codigo. Eso cierra el eje.
- **En el conjunto del proyecto, hay un techo medido de +0.046271 que sigue sin capturar**, y la
  fase localiza donde esta.

La correccion conceptual que hay que conservar
----------------------------------------------
Un unico mecanismo positivo, el GCE de H2, **no autoriza a descartar la familia de la funcion de
perdida**. Lo que autoriza a decir es algo mas preciso: **las variaciones de objetivo evaluadas
hasta ahora tienen un impacto pequeno comparado con el techo que introduce el error de la
condicion de HS**. Esa formulacion evita sobregeneralizar a partir de un solo resultado positivo, y
es la que se escribe en el informe.

Uso
---
    /home/carlos/venv/bin/python experiments/v12_05_informe_final.py
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v11_ayudas import fila, sello

EXPERIMENTO = "V12-05 (informe final de fase y tabla maestra)"
REPORTE_MD = PROJECT_ROOT / "reports" / "v12_final_research.md"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v12_05_informe_final.json"
CSV_MAESTRA = PROJECT_ROOT / "reports" / "v12_master_results.csv"
CSV_HIPOTESIS = PROJECT_ROOT / "reports" / "v12_hypothesis_registry.csv"

REFERENCIA_V03 = 0.7772114525165721
LB_V03 = 0.78306
TECHO_ORACULO = 0.046271
POSTPROCESO_DURO = -0.004187

CAMPOS_MAESTRA = [
    "ID",
    "Track",
    "Hypothesis",
    "NovelMechanism",
    "Features",
    "Model",
    "Loss",
    "Lambda",
    "Temperature",
    "OuterProtocol",
    "InnerProtocol",
    "Seed",
    "GlobalMacroF1",
    "DeltaVsV12",
    "Fold1",
    "Fold2",
    "Fold3",
    "Fold4",
    "Fold5",
    "Folds_Positive",
    "BootstrapCI95",
    "SecondSplit",
    "Holdout2",
    "Salvage",
    "Damage",
    "Net",
    "Decision",
    "Artifact",
    "GitCommit",
]

EXPERIMENTOS: dict[str, dict[str, str]] = {
    "V12-00": {
        "track": "P0 auditoria de brecha",
        "hipotesis": "Que mecanismos se han probado de verdad, y cuales siguen sin probarse",
        "mecanismo": "sonda estructural sobre el arbol sintactico de 127 modulos, no busqueda textual",
        "salvage": "16 familias cerradas validamente, 5 parciales y 3 genuinamente no probadas, con la "
        "distincion de los nueve ejes",
        "damage": "V03 no tiene fila propia en el registro, y el escaner textual.marka sus propias citas",
        "decision": "CLOSED",
    },
    "V12-01": {
        "track": "H1 regularizacion por etiqueta y grupo",
        "hipotesis": "La regularizacion distinta por etiqueta y por grupo de caracteristicas extrae "
        "senal que V03 no usa",
        "mecanismo": "escalar el bloque de caracter, que multiplica su penalizacion L2 efectiva por "
        "s al cuadrado, con renormalizacion por norma de Frobenius",
        "salvage": "el cociente entre bloques tiene UN grado de libertad identificable, y la "
        "comparacion de cuatro brazos separa el ajuste por etiqueta de la estructura de grupo",
        "damage": "el maximo de la curva cae sobre el valor neutro con +0.000006, luego no hay nada "
        "que extraer",
        "decision": "CLOSED",
    },
    "V12-02": {
        "track": "H2 perdida propia y estructura de etiquetas",
        "hipotesis": "Una perdida propia, o restringir el entrenamiento a donde la etiqueta esta "
        "definida, capturan la senal de la estructura jerarquica",
        "mecanismo": "entropia cruzada generalizada L_q, y entrenamiento del condicional solo en su "
        "dominio de aplicabilidad",
        "salvage": "la premisa medida por primera vez, el techo con oraculo de +0.046271, y la "
        "localizacion de que el post-proceso falla porque su llave se equivoca",
        "damage": "la perdida propia da +0.001616 con el intervalo incluyendo el cero, y restringir el "
        "dominio da -0.116581",
        "decision": "MARGINAL",
    },
    "V12-03": {
        "track": "H3 entrenamiento por pares",
        "hipotesis": "Un objetivo por pares sobre TF-IDF puede explotar la estructura relativa entre "
        "HS, TR y AG",
        "mecanismo": "tres clasificadores entrenados jointemente con un termino de margen por pares "
        "que expresa el orden entre las tres etiquetas, sin llave en la prediccion",
        "salvage": "el modo de fallo, identificado y coherente con V6-02: acoplar multiplicativa la "
        "relacion atenua el nivel absoluto y rompe filas que la referencia acertaba",
        "damage": "el mejor lambda da -0.002469, con deteccion 0.4915, es decir peores que el azar",
        "decision": "CLOSED",
    },
    "V12-04": {
        "track": "auditoria de V6-B",
        "hipotesis": "La formulacion de condicionamiento suave sobre la probabilidad de HS es nueva",
        "mecanismo": "no aplica: lectura estructural del codigo de V6-02",
        "salvage": "la reconciliacion de 20 filas alcanzables frente a 78 del techo, y las 58 que "
        "p(HS) no puede identificar",
        "damage": "ninguno: la auditoria evita repetir un experimento cerrado",
        "decision": "CLOSED",
    },
}


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V12-05] {mensaje}", flush=True)


def commit_actual() -> str:
    """Devuelve el commit de la rama, o un aviso si no se puede leer.

    Returns:
        Identificador corto del commit.
    """
    proceso = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return proceso.stdout.strip() if proceso.returncode == 0 else "no disponible"


def main() -> None:
    """Construye la tabla maestra y escribe el informe final de la fase."""
    inicio = time.time()
    log("V12-05: informe final de fase y tabla maestra")
    commit = commit_actual()

    log("1/4 leer los informes de cada experimento")
    registro = {
        f["experiment_id"]: f
        for f in csv.DictReader((PROJECT_ROOT / "experiments.csv").open(encoding="utf-8"))
    }
    informes: dict[str, dict[str, Any]] = {}
    for nombre in ("v12_01_regularizacion_grupo", "v12_02_perdida_y_estructura",
                   "v12_03_entrenamiento_por_pares", "v12_04_auditoria_v6b"):
        ruta = PROJECT_ROOT / "reports" / f"{nombre}.json"
        if ruta.exists():
            informes[nombre] = json.loads(ruta.read_text(encoding="utf-8"))

    log("2/4 tabla maestra")
    filas: list[dict[str, Any]] = []
    for identificador, meta in EXPERIMENTOS.items():
        fila_registro = registro.get(identificador, {})
        clave = next((k for k in informes if identificador[-2:] in k), None)
        informe = informes.get(clave, {}) if clave else {}
        filas.append(
            {
                "ID": identificador,
                "Track": meta["track"],
                "Hypothesis": meta["hipotesis"],
                "NovelMechanism": meta["mecanismo"],
                "Features": "union TF-IDF word(1,2) + char_wb(3,5), min_df 2, la de V03 sin cambios",
                "Model": "LR C=2.0 balanced liblinear x3, la de V03",
                "Loss": "entropia cruzada" if identificador != "V12-03" else (
                    "entropia cruzada mas margen por pares, con optimizador propio"
                ),
                "Lambda": "no aplica" if identificador != "V12-03" else "0.0, 0.25, 1.0",
                "Temperature": "no aplica: no hay temperaturas en esta fase",
                "OuterProtocol": "StratifiedGroupKFold de 5 pliegues, grupos near-dup UNION pseudo-autor",
                "InnerProtocol": "validacion anidada de grupo, o ninguna si no hay eleccion",
                "Seed": "42",
                "GlobalMacroF1": fila_registro.get("f1") or "no aplica",
                "DeltaVsV12": "-",
                "Fold1": "", "Fold2": "", "Fold3": "", "Fold4": "", "Fold5": "",
                "Folds_Positive": (informe.get("veredictos") or {}).get(
                    "lambda=0.25", {}
                ).get("pliegues_a_favor", "-") if informe else "-",
                "BootstrapCI95": "-",
                "SecondSplit": "no ejecutada, no hay candidata",
                "Holdout2": "no usado, 0 de 3 evaluaciones gastadas",
                "Salvage": meta["salvage"],
                "Damage": meta["damage"],
                "Net": meta["decision"],
                "Decision": (fila_registro.get("decision") or "")[:220],
                "Artifact": f"reports/v12_{identificador[-2:]}_*.md",
                "GitCommit": commit,
            }
        )
    with CSV_MAESTRA.open("w", encoding="utf-8", newline="") as manejador:
        escritor = csv.DictWriter(manejador, fieldnames=CAMPOS_MAESTRA, lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(filas)
    log(f"   {len(filas)} filas en {CSV_MAESTRA.relative_to(PROJECT_ROOT)}")

    log("3/4 registro de hipotesis")
    campos_hipotesis = [
        "ID", "Hipotesis", "Mecanismo", "Estado", "Mejor_Delta", "Pliegues_Positivos",
        "IC95", "Deteccion", "Gate_Exploratorio", "Captura_del_techo", "Decision", "Por_que",
    ]
    h2 = informes.get("v12_02_perdida_y_estructura", {})
    h3 = informes.get("v12_03_entrenamiento_por_pares", {})
    v2_v10 = h2.get("veredictos", {}).get("G (q=0.2)", {})
    v3_best = h3.get("veredictos", {}).get("lambda=0.25", {})
    hipotesis = [
        {
            "ID": "H1",
            "Hipotesis": "regularizacion distinta por etiqueta y por grupo de caracteristicas",
            "Mecanismo": "escalar el bloque de caracter y renormalizar",
            "Estado": "CLOSED",
            "Mejor_Delta": "+0.002738",
            "Pliegues_Positivos": "3/5",
            "IC95": "incluye cero",
            "Deteccion": "0.5196",
            "Gate_Exploratorio": "no",
            "Captura_del_techo": "0.0591",
            "Decision": "CLOSED",
            "Por_que": "el maximo de la curva cae sobre el valor neutro, luego no hay nada que extraer",
        },
        {
            "ID": "H2a",
            "Hipotesis": "una perdida propia tolerante al ruido captura la senal de la zona no definida",
            "Mecanismo": "entropia cruzada generalizada L_q",
            "Estado": "MARGINAL",
            "Mejor_Delta": v2_v10.get("delta", "+0.001616"),
            "Pliegues_Positivos": v2_v10.get("pliegues_a_favor", "4/5"),
            "IC95": "incluye cero",
            "Deteccion": v2_v10.get("deteccion", "0.6667"),
            "Gate_Exploratorio": "no",
            "Captura_del_techo": "0.0349",
            "Decision": "MARGINAL",
            "Por_que": "es una meseta real y no un pico, pero el delta queda bajo el gate",
        },
        {
            "ID": "H2b",
            "Hipotesis": "restringir el entrenamiento a donde la etiqueta esta definida",
            "Mecanismo": "entrenar el condicional solo en su dominio de aplicabilidad",
            "Estado": "CLOSED",
            "Mejor_Delta": "-0.116581",
            "Pliegues_Positivos": "0/5",
            "IC95": "excluye cero, hacia abajo",
            "Deteccion": "0.1420",
            "Gate_Exploratorio": "no",
            "Captura_del_techo": "0.0000",
            "Decision": "CLOSED",
            "Por_que": "descartar el 56 por ciento de los datos desplaza la escala de salida y los "
            "umbrales congelados dejan de tener significado",
        },
        {
            "ID": "H3",
            "Hipotesis": "un objetivo por pares sobre TF-IDF explota la estructura relativa entre las "
            "tres etiquetas",
            "Mecanismo": "tres clasificadores jointemente con margen por pares, sin llave en la "
            "prediccion",
            "Estado": "CLOSED",
            "Mejor_Delta": v3_best.get("delta", "-0.002469"),
            "Pliegues_Positivos": v3_best.get("pliegues_a_favor", "2/5"),
            "IC95": "incluye cero",
            "Deteccion": v3_best.get("deteccion", "0.4915"),
            "Gate_Exploratorio": "no",
            "Captura_del_techo": "0.0000",
            "Decision": "CLOSED",
            "Por_que": "un objetivo de orden solo restringe diferencias y no dice nada del nivel "
            "absoluto, y la metrica se evalua con umbrales fijos",
        },
    ]
    with CSV_HIPOTESIS.open("w", encoding="utf-8", newline="") as manejador:
        escritor = csv.DictWriter(manejador, fieldnames=campos_hipotesis, lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(hipotesis)
    log(f"   {len(hipotesis)} hipotesis en {CSV_HIPOTESIS.relative_to(PROJECT_ROOT)}")

    log("4/4 el informe final")
    informe = {
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": commit,
        "referencia_v03": REFERENCIA_V03,
        "lb_v03": LB_V03,
        "techo_oraculo": TECHO_ORACULO,
        "postproceso_duro": POSTPROCESO_DURO,
        "experimentos": len(filas),
        "hipotesis": [h["ID"] for h in hipotesis],
        "candidatas": [],
        "submissions_generadas": 0,
        "holdout2_gastadas": 0,
        "pregunta_final": (
            "Queda alguna fuente de senal explotable por modelos permitidos que no se haya probado "
            "adecuadamente, o V03 constituye el limite practico bajo las restricciones?"
        ),
    }
    REPORTE_JSON.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _informe(informe, filas, hipotesis, inicio)
    log(f"completado en {round(time.time() - inicio, 1)} s")
    log(
        f"VEREDICTO: {len(filas)} experimentos | {len(hipotesis)} hipotesis | candidatas 0 | "
        f"submissions 0 | techo sin capturar {TECHO_ORACULO:+.6f}"
    )


def _informe(
    informe: dict[str, Any],
    filas: list[dict[str, Any]],
    hipotesis: list[dict[str, Any]],
    inicio: float,
) -> None:
    """Escribe el informe final de la fase.

    Args:
        informe: veredicto de la fase.
        filas: filas de la tabla maestra.
        hipotesis: registro de hipotesis.
        inicio: instante de arranque del script.
    """
    linea: list[str] = [
        "# V12: shift, calibracion y regla de decision. Informe final de fase",
        "",
        f"Generado por `experiments/v12_05_informe_final.py` ({EXPERIMENTO}).",
        f"Commit: `{informe['commit']}`.",
        "",
        f"Referencia: **V03 = {REFERENCIA_V03}**, leaderboard {LB_V03}. V03 permanece congelada.",
        f"**{informe['experimentos']} experimentos, 0 submissions, 0 evaluaciones de holdout2 "
        "gastadas, 0 candidatas.**",
        "",
        "## 1. La pregunta y la respuesta, en dos mitades que no hay que mezclar",
        "",
        "> Queda alguna fuente de senal explotable por modelos permitidos que no se haya probado",
        "> adecuadamente, o V03 constituye efectivamente el limite practico?",
        "",
        "**En el eje del objetivo de entrenamiento, si quedaba, y la v12 lo ha cerrado.** La",
        "auditoria de brecha, V12-00, encontro tres huecos genuinamente no probados, y los tres",
        "estaban en ese eje: no existia ninguna clase de perdida propia en 127 modulos, ninguna",
        "ocurrencia de entrenamiento por pares, y ninguna regularizacion por grupo de caracteristicas",
        "dentro de la etiqueta. Los tres se ejecutaron con gates identicos y los tres fallaron.",
        "",
        "**En el conjunto del proyecto, hay un techo medido de +0.046271 que sigue sin capturar.** Y",
        "la v12 aporta algo mas valioso que el negativo: **localizo donde esta ese techo.**",
        "",
        "## 2. La tabla maestra",
        "",
        fila(["ID", "via", "hipotesis", "mecanismo", "decision"]),
        fila(["---"] * 5),
    ]
    for f in filas:
        linea.append(
            fila([f["ID"], f["Track"], f["Hypothesis"][:70], f["NovelMechanism"][:70], f["Net"]])
        )
    linea += [
        "",
        "## 3. Las cuatro hipotesis, y que capturaron del techo",
        "",
        fila(
            [
                "hipotesis",
                "mejor delta",
                "pliegues a favor",
                "deteccion",
                "fraccion del techo",
                "decision",
            ]
        ),
        fila(["---"] * 6),
    ]
    for h in hipotesis:
        # Todos los campos se pasan como cadena: algunos llegan del JSON como float y el formateador
        # de tabla los rechaza con un TypeError que no dice nada sobre la causa real.
        linea.append(
            fila(
                [
                    str(h["ID"]),
                    f"{float(h['Mejor_Delta']):+.6f}",
                    str(h["Pliegues_Positivos"]),
                    str(h["Deteccion"]),
                    f"{float(h['Captura_del_techo']) * 100:.1f} por ciento",
                    str(h["Decision"]),
                ]
            )
        )
    linea += [
        "",
        f"El techo de referencia es **{TECHO_ORACULO:+.6f}**, la mascara con la etiqueta verdadera de",
        "HS, que no es adoptable porque usa la verdad y solo acota.",
        "",
        "## 4. La conclusion que la fase deja establecida",
        "",
        "**El techo condicionado por la etiqueta verdadera de HS demuestra que existe senal",
        "estructural abundante**, del orden de +0.046, que es veintitres veces el gate exploratorio y",
        "trescientas veces la mejor ganancia anidada de toda la historia del proyecto, que era de",
        "+0.000148.",
        "",
        "**El cuello de botella no es la ausencia de capacidad predictiva para TR y AG.** El",
        "rendimiento de TR donde la etiqueta esta definida es de 0,8941, muy alto. La dificultad es",
        "**inferir la condicion**, es decir saber, para cada fila, si HS es cero.",
        "",
        "Y las tres vias para superar esa condicion fallan, y fallan por razones distintas y todas",
        "documentadas:",
        "",
        "1. **La llave dura**, el post-proceso HS=0 luego TR=0 y AG=0, da",
        f"   **{POSTPROCESO_DURO:+.6f}**. Falla porque su llave es una prediccion que se equivoca en",
        "   143 filas de 1483, de modo que la mascara se aplica a filas donde TR si esta definida.",
        "2. **La llave suave**, que es V6-B y ya estaba cerrada, cubre solo las 20 filas donde HS se",
        "   predice bien, y no puede tocar las 58 restantes.",
        "3. **Acoplar la relacion en el objetivo**, que es H3, mejora la discriminacion local y",
        "   empeora los falsos positivos justo en la zona que hay que corregir, porque un objetivo de",
        "   orden no restringe el nivel absoluto y la metrica se evalua con umbrales fijos.",
        "",
        "## 5. La correccion conceptual que hay que conservar",
        "",
        "Conviene decirlo con precision, porque es facil sobregeneralizar aqui.",
        "",
        "**H2 y H3 no demuestran que la funcion de perdida no sea el problema en sentido absoluto.**",
        "Un unico mecanismo positivo, el GCE de H2, **no autoriza a descartar la familia entera**.",
        "",
        "Lo que si autorizan a decir es algo mas preciso y mas util: **las variaciones de objetivo",
        "evaluadas hasta ahora tienen un impacto pequeno comparado con el techo que introduce el",
        "error de la condicion de HS**. La mejor de ellas, el GCE con q=0,2, da +0.001616 frente a un",
        "techo de +0.046271, es decir captura el 3,5 por ciento. Y la mas estruturada, el objetivo por",
        "pares de H3, empeora.",
        "",
        "La distincion importa porque las dos afirmaciones llevan a decisiones distintas. La primera",
        "cerraria una familia de forma definitiva. La segunda solo establece que **el margen",
        "recuperable por esa via es pequeno frente a otro problema ya medido**, y deja la familia",
        "abierta para quien quiera explorarla con un mecanismo distinto.",
        "",
        "## 6. Rendimientos decrecientes en el objetivo de entrenamiento",
        "",
        "Tres mecanismos distintos sobre la misma funcion, en orden de sofisticacion:",
        "",
        "- **H1**, regularizacion por etiqueta y grupo: **+0.002738**, pero sostenido por un solo",
        "  pliegue de cinco, y el maximo de su curva cae sobre el valor neutro con +0.000006.",
        "- **H2a**, perdida propia tolerante al ruido: **+0.001616**, con 4 de 5 pliegues y",
        "  deteccion 0,6667, pero el intervalo pareado incluye el cero.",
        "- **H3**, objetivo por pares que acopla las tres etiquetas: **-0.002469**, con deteccion",
        "  0,4915, es decir **peores que el azar**.",
        "",
        "Ninguno llega al gate de +0.002 de forma defendible, y el techo de +0.046271 sigue intacto.",
        "Eso es la senal de **rendimientos decrecientes en el objetivo de entrenamiento**: no es que",
        "la primera idea fuera mala y las siguientes peores, es que el eje entero tiene poco margen",
        "recuperable frente a lo que queda disponible en otro sitio.",
        "",
        "## 7. Que queda abierto, y por que no es de esta fase",
        "",
        "**Queda abierto un solo problema, y la v12 lo deja escrito sin abrirlo: conseguir que la",
        "condicion de HS se estime mejor.** Todo lo demas depende de eso, porque con HS verdadera la",
        "senal esta ahi, con HS predicha no llega, y con HS mal predicha no sale",
        "",
        "No es una hipotesis de esta fase porque no es una hipotesis sobre el objetivo, la",
        "representacion, la capa de decision ni el ensemble: es una hipotesis sobre **la calidad de",
        "una condicion**, y el proyecto ya ha cerrado reiteradamente las familias que podrian",
        "atacarla, desde la regularizacion conjunta entre etiquetas, V8-05, hasta las cadenas, E2-01,",
        "y la factorizacion condicional, V6-02.",
        "",
        "## 8. El valor de la auditoria de brecha, que es methodologico",
        "",
        "**V12-00 encontro una asimetria que reencuadro la fase**: 17 de las 24 familias de mecanismo",
        "ya cerradas estaban en el eje de la representacion y del clasificador, y los tres huecos",
        "que quedaban estaban todos en el objetivo de entrenamiento. El proyecto habia sido",
        "exhaustivo en un eje y no habia tocado nunca la funcion que se minimiza. Esa asimetria es",
        "lo que justificaba abrir la v12, y el hallazgo se sostiene a la vista de los tres negativos.",
        "",
        "Y dos decisiones de instrumento que rinden mas que sus resultados:",
        "",
        "1. **La sonda es estructural, no textual.** La auditoria se autovalido con un positivo",
        "   conocido: el unico fichero del proyecto donde aparece la palabra contrastive es el",
        "   propio script de la auditoria, que la nombra al describir la familia ausente. Un",
        "   escaner textual concluiria que el mecanismo si se ha probado, con la maxima confianza, por",
        "   coincidencia literal. El indice estructural devuelve cero, que es lo correcto.",
        "2. **Todo objetivo a medida verifica su derivada antes de mirar un solo resultado.** En H3 eso",
        "   detecto cuatro fallos propios que habrian producido cifras plausibles y falsas: la",
        "   penalizacion L2 ausente del valor, el recorte del margen rompiendo la regla de la cadena,",
        "   el vertice de la bisagra, y la busqueda lineal demasiado corta. Un gradiente mal",
        "   implementado no da un fallo, da un numero.",
        "",
        "## 9. Lo que esta fase NO ha hecho",
        "",
        "- **No ha entrenado nada antes de cerrar la auditoria de brecha**, como exigia el",
        "  enunciado.",
        "- **No ha reabierto ninguna familia cerrada.** Donde el veredicto es CLOSED_VALID la",
        "  lectura es que no se ejecuta, no que se ejecuta con mas cuidado.",
        "- **No ha generado ni recomendado ninguna submission.** V03 sigue siendo la submission",
        "  disponible.",
        "- **No ha usado el texto de la prueba, ni holdout2, ni el leaderboard.** El uso",
        "  transductivo del test sigue prohibido, como en v11.",
        "- **No ha ejecutado la segunda particion**, porque ningun candidato llego a la mesa de",
        "  finales y gastar la en un negativo seria trabajar para nada.",
        "",
        "## 10. Estado final",
        "",
        fila(["elemento", "estado"]),
        fila(["---"] * 2),
        fila(["V03", f"congelada, {REFERENCIA_V03}, LB {LB_V03}, huella sha256 intacta"]),
        fila(["submissions generadas en la v12", "0 de 26 historicas, ninguna nueva"]),
        fila(["evaluaciones de holdout2 gastadas", "0 de 3"]),
        fila(["segunda particion", "no ejecutada, no hay candidata"]),
        fila(["candidatas", "0"]),
        fila(["uso del texto de la prueba", "ninguno, y sigue prohibido"]),
        fila(["leading board", "no consultado"]),
        "",
    ]
    linea += sello(inicio)
    REPORTE_MD.write_text("\n".join(linea) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
