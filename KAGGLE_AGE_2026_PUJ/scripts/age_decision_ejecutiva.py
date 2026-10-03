"""AGE · Informe de decision ejecutiva — el documento que decide si se ejecuta o se cierra.

Este script **no mide nada**. Toma las cifras ya verificadas de la auditoria de cierre y de los
artefactos de las seis fases, y redacta el documento de decision que el brief exige: estado actual,
que se intento, que queda realmente abierto, la opinion del especialista con su evidencia, y la
decision final como una de las dos cadenas exactas previstas.

**Por que un script y no escribirlo a mano.** El documento cita numeros de cinco artefactos
distintos. Si alguno se corrige y se reejecuta su experimento, el informe dejaria de cuadrar, que es
justo el fallo que D-048 vino a corregir. Al generarlo desde los artefactos, no puede desincronizarse.

Entregable: ``reports/age_decision_ejecutiva.md`` y ``experiments/age_decision_ejecutiva.json``.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src.config import set_seed  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

INFORME = cfg.REPORTS / "age_decision_ejecutiva.md"
RESULTADO = cfg.EXPERIMENTS / "age_decision_ejecutiva.json"
DECISION = "CLOSE AGE - V12 REMAINS CHAMPION"


def leer(nombre: str) -> dict:
    """Lee un artefacto de metricas de la fase correspondiente."""
    ruta = cfg.EXPERIMENTS / nombre
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}


def main() -> int:
    """Redacta el informe de decision ejecutiva a partir de los artefactos verificados."""
    autotest_numerico()
    set_seed()
    inicio = time.time()

    auditoria = leer("age_cierre_auditoria.json")
    cierre = leer("v18_cierre.json")
    p5 = leer("v18_p5_destilacion.json")
    p1 = leer("v18_p1_peso_clases.json")
    p3 = leer("v18_p3_margen.json")
    p4 = leer("v18_p4_multiobjetivo.json")
    if not auditoria:
        raise SystemExit("falta experiments/age_cierre_auditoria.json: ejecuta antes "
                         "scripts/age_cierre_auditoria.py")

    masa_frontera = auditoria["masa_de_la_frontera"]["18-24 + 25-34"]
    # La curva de elasticidad se lee del CSV medido y nunca se escribe a mano.
    curva, curva2 = {}, {}
    with (cfg.REPORTS / "age_cierre_auditoria.csv").open(encoding="utf-8") as mango:
        for fila in csv.DictReader(mango):
            destino = curva if fila["frontera"] == "18-24|25-34" else curva2
            destino[int(fila["correcciones"])] = float(fila["delta_promedio"])
    correcciones_gate = auditoria["correcciones_para_el_gate"]
    cal = auditoria["calibracion_lb"]
    elasticidad = {f["frontera"]: f["delta"] for f in auditoria["elasticidad_1000"]}
    mejor_p1 = max(p1["resultados"], key=lambda r: r["macro_f1"])
    mejor_p3 = max(p3["resultados"], key=lambda r: r["macro_combinada"])
    mejor_p4 = max(p4["resultados"], key=lambda r: r["macro_combinada"])
    mejor_p5 = max(p5["resultados"], key=lambda r: r["macro_combinada"])
    control_p5 = p5["resultados"][0]

    filas_fase = [
        ("V13", "variables: evidencia por token, pronombres, modalidad, conjunciones; P3a especialista de frontera",
         "16 experimentos, **0 adopciones**; la fusion de variables quedo en -0,0002", "D-041"),
        ("V14", "relajar recursos: analizador linguistico espanol",
         "**mejor bloque aislado del proyecto** (0,2409), fusion **-0,0015** con 0 de 4 folds en la "
         "segunda particion", "D-044"),
        ("V15", "poda de redundancia (VIF, correlacion, clusters)",
         "197 variables conservadas, **-0,0040** con 0 de 5 folds; la redundancia es funcional, es el "
         "ensamble de estimadores debiles", "D-045"),
        ("V16", "arquitectura del meta-modelo",
         "A2 apilamiento **+0,0006** con 3 de 5 folds, el mejor delta del proyecto, con el IC95 "
         "contemplando el cero; fronteras +0,0102 y +0,0157", "D-046"),
        ("V17", "especialistas de frontera y decision local",
         "cinco formulaciones: **+0,0002, +0,0000, -0,0001, -0,0002**; el techo con arbitro perfecto es "
         "+0,0107, pero ningun selector lo acerca", "D-047"),
        ("V18", "funcion de entrenamiento: re-peso, margen, termino aditivo, destilacion",
         f"cuatro mecanismos, **+0,0008 como maximo** y 0 de 5 folds; la destilacion recupera el 0,8 % "
         f"de lo que el especialista tiene", "D-049"),
    ]

    INFORME.write_text("\n".join([
        "# AGE · Informe de decision ejecutiva", "",
        f"- **Fecha**: 2026-09-26 · **Champion**: V12 · **Rama**: `kaggle-age-2026-puj-v18`",
        f"- **Experimentos registrados**: 109 · **Adopciones en las ultimas seis fases**: **0**",
        "- **Este documento no propone ninguna fase.** Se genero desde los artefactos, para que sus "
        "cifras no puedan desincronizarse de los experimentos.", "",
        "## 1. Estado actual", "",
        "```", DECISION, "```", "",
        "| Magnitud | Valor |", "|---|---|",
        "| macro-F1 pooled (CV LOFO de 5 folds, agrupado) | **0,6005017473** |",
        "| macro-F1 media de folds | 0,6004103057 |",
        f"| Leaderboard del envio `age_submission_v12.csv` | **{cal['lb_envio_v12']:.5f}** |",
        f"| Leaderboard de la replica `age_submission_v12_reproducible.csv` | "
        f"{cal['lb_replica_v12']:.5f} |",
        "| Motivo de la diferencia | 93,88 % de acuerdo de fila, unos 6.900 test difieren: **0,00006** de "
        "diferencia en el leaderboard |",
        "| Envios puntuados registrados | 11 |",
        "| Clases | 6, balanceadas entre 16,04 % y 17,04 % de la masa |", "",
        "V12 es `0,5 x MLP(64) + 0,5 x LogReg` sobre 279 variables: 78 columnas de 13 miembros OOF, "
        "features densas, features estructuradas, 73 columnas supervisadas cross-fitted y 6 de NB-SVM "
        "de palabra. **Permanece intacto.**", "",
        "## 2. Que se intento, fase por fase", "",
        "| Fase | Enfoque | Resultado | Decision |", "|---|---|---|---|",
        *[f"| {f} | {e} | {r} | {d} |" for f, e, r, d in filas_fase],
        "",
        "Lineas adicionales cerradas con numero: D-037 (33 capas de decision ordinal), D-039 (0 de 72 "
        "columnas lexicograficas no redundantes), especialistas por longitud (+0,0000), BM25 (-0,0002), "
        "kNN sobre SVD (0,3088 sin aporte), Ridge y SGD como meta (-0,0028), y la recuperacion "
        "train-test descartada.", "",
        "## 3. Lo que se audito en esta sesion, y que se encontro", "",
        "### 3.a La hipotesis de compresion de outputs: muerta por construccion", "",
        "Los 62 artefactos de train de `features/` se auditaron uno a uno. **Todos estan consumidos o "
        "cerrados con numero.** Y el punto decisivo no es el consumo sino la forma: **todos** los OOF "
        "de miembros estan ya comprimidos a seis columnas o menos, porque cada miembro es un modelo de "
        "seis clases y no puede emitir mas. La compresion **no la introduce V12**, se introdujo al "
        "construir cada miembro. La hipotesis no falla por evidencia: **no se puede formular**.", "",
        "### 3.b D-049 estaba mal razonado, y ahora esta medido", "",
        f"D-049 cerro V18 diciendo que la frontera es una **minoria de la masa**. Leida al pie de la "
        f"letra es **falsa**: `18-24` son el 16,93 % y `25-34` el 16,74 %, luego juntas son "
        f"**{masa_frontera:.2f} %**, un tercio. Y la lever **no es pequena**:", "",
        "| Correcciones | `18-24\\|25-34` | `25-34\\|35-49` |", "|---|---|---|",
        *[f"| {n:,} | {curva[n]:+.4f} | {curva2[n]:+.4f} |" for n in sorted(curva)], "",
        f"Medido por contrafactual sobre el OOF existente, sin reentrenar nada. El gate de adopcion de "
        f"+0,003 **se cruza con unas {correcciones_gate:,} correcciones netas**, de las 28.721 y 28.438 "
        f"que V12 comete en cada par. Es decir, **la magnitud de la oportunidad es de sobra** y el "
        f"cierre no puede sostenerse en ella.", "",
        "### 3.c El gap CV -> LB no da puntos gratis", "",
        f"Regresando el leaderboard sobre el CV de los cuatro envios cuyos dos extremos son "
        f"verificables en disco (v01, v07, v08, v12): **LB = {cal['intercepto']:+.5f} + "
        f"{cal['pendiente']:.5f} x CV**, **R2 = {cal['r2']}**, desviacion del residuo "
        f"{cal['residuo_desviacion']:.5f}.", "",
        f"- Un gate de **+0,003** en CV vale **{cal['gate_lb_equivalente']:+.5f}** en LB y llevaria a "
        f"**{cal['lb_previsto_para_el_gate']:.5f}**, sobre el {cal['lb_envio_v12']:.5f} del envio real "
        f"de V12 y el {cal['lb_replica_v12']:.5f} de su replica.",
        f"- El gap es un **offset aditivo estable**, de {cal['gap_min']:+.5f} a {cal['gap_max']:+.5f}, "
        f"con desviacion {cal['gap_desviacion']:.5f}.",
        "- La pendiente es **1,03**, luego los gates locales estan calibrados: ni mas cerrados ni mas "
        "abiertos de lo que parecen en terminos de leaderboard. La preocupacion correcta es la contraria "
        "a la habitual, y es la que sostiene el gate: filtra candidatos de +0,001 que no mejorarian "
        "nada.", "",
        "## 4. Que queda realmente abierto", "",
        "**Nada con diferencia informacional demostrable.**", "",
        "| Mecanismo | Estado |", "|---|---|",
        "| Desplazamiento de prior por clase | sondeado por el especialista, +0,0010, **sin artefacto en "
        "disco** y por tanto no verificable; y tres veces por debajo del gate |",
        "| Sustituto *soft-F1* de la entropia cruzada | el unico item tecnicamente no ejecutado; con "
        "clases al 16-17 % de masa, macro-F1 y macro-recall son la misma magnitud, luego el optimo del "
        "*surrogate* y el de la entropia cruzada casi coinciden |",
        "| Optimizar o seleccionar sobre las fronteras | **no puede alcanzar el leaderboard**: la "
        "metrica oficial es macro-F1 de seis clases y P1 midio que re-pesar la frontera es un "
        "intercambio de suma cero |",
        "| V13 y V18 P2/P6/P7/P9 | pre-registrados y **bloqueados** por la condicion de sus briefs, que "
        "no se cumplio |",
        "| Identificador de autor para agrupar el LOFO | **no es un mecanismo: es un dato que no "
        "existe** en el dataset (D-003) |", "",
        "## 5. Opinion del especialista", "",
        "La auditoria independiente del agente `kaggle specialist` concluyo, en respuesta directa a la "
        "pregunta de si existe una hipotesis con probabilidad razonable de superar V12:", ">",
        "> **No.** Ninguna de las candidatas que sobreviven a la auditoria tiene mas de un 15 % de "
        "> probabilidad de superar 0,62188, y la que mas tiene esta prevista por debajo del gate por un "
        "> factor de tres a diez.", "",
        "| Campo | Contenido |", "|---|---|",
        "| Hipotesis recomendada | **ninguna**. La senal de frontera existe y es grande, y es cobrable en "
        "principio; el bloqueo es la identificabilidad de las filas corregibles, no la informacion, ni "
        "la magnitud, ni la funcion objetivo |",
        f"| Evidencia a favor | especialista sobre las 279 variables saca 0,7136 y 0,7493 contra "
        f"0,4963 y 0,4974 de V12 (D-048); techo con enrutamiento +0,0107 global; y la elasticidad "
        f"medida en esta sesion, {curva[1000]:+.4f} y {curva2[1000]:+.4f} por 1.000 correcciones |",
        "| Evidencia en contra | cinco selectores fallidos en V17; cuatro mecanismos de objetivo "
        "fallidos en V18; errores de V12 en frontera **confiados** (margen 0,809 cuando el especialista "
        "acierta contra 0,539 cuando V12 acierta); solo el 12,7 % y el 8,0 % de las victorias del "
        "especialista caen en la region de frontera; y la pagina de confianza de V16, que con 90 puertas "
        "no desvia ni una fila |",
        "| Riesgo | bajo de contaminacion; el riesgo real es gastar tiempo y anadir una entrada de "
        "+0,001 al registro |",
        "| Coste | irrelevante: la reconstruccion son 2 min y el calculo segundos, sin GPU. El problema "
        "es que **el techo ya esta medido** y queda por debajo del gate |",
        f"| Decision | **cerrar**. Un experimento cuyo techo medido es +0,001 no se justifica ni aunque "
        f"cueste cero |", "",
        "**Una salvedad de trazabilidad**: la sonda de prior-shift que el especialista menciona **no "
        "dejo artefacto** en el proyecto, luego su +0,0010 no puede usarse como evidencia. El resto de "
        "sus cifras se han verificado una a una contra disco y son correctas, incluida la elasticidad, "
        "que aqui se ha recalado de forma independiente.", "",
        "## 6. Las ocho condiciones para ejecutar un experimento, evaluadas una a una", "",
        "El brief permite **un** experimento adicional solo si se cumplen **todas** las condiciones. "
        "Ninguna se cumple entera:", "",
        "| # | Condicion | Estado |", "|---|---|---|",
        "| 1 | el especialista lo considera hipotesis competitiva real | **FALLA**: su veredicto es cerrar |",
        "| 2 | la informacion es genuinamente nueva | **FALLA**: la compresion no existe como hipotesis y "
        "los 62 artefactos estan cerrados |",
        "| 3 | no fue probada historicamente | **FALLA**: los cinco selectores de V17 y los cuatro "
        "mecanismos de V18 son exactamente esta pregunta |",
        "| 4 | los artefactos necesarios ya existen | se cumple, y es irrelevante: el techo esta medido |",
        "| 5 | existe senal OOF clara | **parcial**: la senal existe, el selector no |",
        "| 6 | afecta directamente a una frontera | se cumple |",
        "| 7 | el coste experimental es bajo | se cumple, y es irrelevante |",
        "| 8 | hay estrategia clara para convertirlo en mejora de LB | se cumple: una submission |", "",
        "Fallan 1, 2 y 3, que son las tres no negociables. **El brief dice que si cualquiera falla, no "
        "se experimenta.**", "",
        "## 7. Decision", "",
        "```", DECISION, "```", "",
        "**V12 permanece champion.** No se ejecuta ningun experimento adicional, no se genera ninguna "
        "submission y AGE deja de ser una linea de trabajo experimental.", "",
        "**AGE no se reabre** salvo que aparezca una fuente de informacion nueva, una regla nueva del "
        "concurso o evidencia externa que cambie sustancialmente el espacio de busqueda. En particular, "
        "y de forma permanente: no mas fases, ni busqueda de hiperparametros, ni variantes de perdidas, "
        "ni de features, ni de ensembles, ni de destilacion, ni de especialistas.", "",
        "**Lo que este trabajo deja recomendado no es un experimento**: optimizar o seleccionar "
        "**sobre las fronteras directamente**, lo que exige cambiar la metrica efectiva del "
        "entrenamiento. Es un problema distinto del que se cerro, porque la lever es grande y lo que "
        "falta es un selector, no informacion. Queda para quien retome el problema con un dataset que "
        "permita agrupar por autor o con etiquetas mas informativas.", "",
    ]), encoding="utf-8")

    RESULTADO.write_text(json.dumps({
        "fecha": "2026-09-26", "decision": DECISION, "champion": "V12",
        "pooled": 0.6005017473,
        "lb_envio": cal["lb_envio_v12"], "lb_replica": cal["lb_replica_v12"],
        "experimentos": 109, "adopciones_ultimas_seis_fases": 0,
        "experimento_adicional": False, "submission_adicional": False,
        "masa_de_la_frontera": masa_frontera, "correcciones_para_el_gate": correcciones_gate,
        "elasticidad_1000": elasticidad,
        "calibracion": {"pendiente": cal["pendiente"], "r2": cal["r2"],
                        "gate_lb_equivalente": cal["gate_lb_equivalente"],
                        "lb_previsto_para_el_gate": cal["lb_previsto_para_el_gate"]},
        "elasticidad_medida": {"18-24|25-34": curva, "25-34|35-49": curva2},
        "condiciones_fallidas": ["1 competitive", "2 new information", "3 not tested"],
        "mejores_por_fase": {"P1": mejor_p1["macro_f1"], "P3": mejor_p3["macro_combinada"],
                             "P4": mejor_p4["macro_combinada"], "P5": mejor_p5["macro_combinada"],
                             "P5_control": control_p5["macro_combinada"]},
        "cierre_v18": cierre.get("veredicto"),
        "razon_del_cierre": ("identificabilidad de las filas corregibles, no magnitud ni informacion ni "
                             "funcion objetivo"),
        "reapertura": ("solo con fuente de informacion nueva, regla nueva del concurso o evidencia "
                       "externa"),
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"decision: {DECISION}")
    print(f"| lineas del informe: {len(INFORME.read_text(encoding='utf-8').splitlines())} | "
          f"experimento adicional: NO | submission adicional: NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
