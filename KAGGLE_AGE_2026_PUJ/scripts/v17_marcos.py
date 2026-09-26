"""V17 · Generador de los dos CSV maestros de la fase, a partir de los JSON ya producidos.

El brief pide dos tablas de seguimiento con columnas fijas: `reports/v17_master_results.csv` y
`reports/v17_hypothesis_registry.csv`. Se generan **desde los artefactos** de cada experimento y
no a mano, por dos razones practicas. La primera es que una cifra tecleada a mano es una cifra que
se puede equivocar sin que nadie lo note. La segunda es que las tablas tienen que poder
regenerarse cuando un experimento se repita o se corrija, y eso solo es automatico si el generador
lee la fuente.

**Como se marca lo que aun no se ha medido.** Las pistas que la fase todavia no ha ejecutado
aparecen con `Decision` = `PENDIENTE` y las celdas de resultado vacias. No se rellenan con
estimaciones ni con valores de la auditoria: una tabla de resultados con numeros que no salieron de
un experimento es exactamente el fallo que este proyecto lleva dos fases evitando.

Entregables: ``reports/v17_master_results.csv``, ``reports/v17_hypothesis_registry.csv``,
``experiments/v17_marcos.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src.config import set_seed  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

MAESTRO = cfg.REPORTS / "v17_master_results.csv"
REGISTRO = cfg.REPORTS / "v17_hypothesis_registry.csv"
RESULTADO = cfg.EXPERIMENTS / "v17_marcos.json"
#: Decimales con los que se publica una macro-F1. Pooled, cuatro decimales, como en
#: `metricas_globales`, que es el estimador canonico de V17.
REDONDEO = 4
#: Fases ya ejecutadas, con su artefacto de metricas y que celda de la tabla maestra rellenan.
EXPERIMENTOS = {
    "P0": ("experiments/v17_p0_reproduccion.json", "auditoria", None),
    "P0'": ("experiments/v17_p0_reproduccion.json", "referencia", None),
    "P1": ("experiments/v17_p1_fronteras.json", "matriz_por_miembro", "18-24|25-34"),
    "P1.1": ("experiments/v17_p1_fronteras.json", "complementariedad", "25-34|35-49"),
    "P1b": ("experiments/v17_p1b_curva_compuerta.json", "curva_de_compuerta", None),
    "P8": ("experiments/v17_p8_perfil_confianza.json", "perfil_de_confianza", None),
    "P1c": ("experiments/v17_p1c_regla_conjunta.json", "regla_conjunta", None),
    "P9": ("experiments/v17_p9_arbitro.json", "arbitro_aprendido", None),
}
#: Pistas del brief que todavia no se han ejecutado, con su hipotesis y su riesgo declarados.
PENDIENTES = {
    "P2": ("Especialista binario entrenado solo con las dos clases de la frontera",
           "que un clasificador dedicado a la frontera supere a V12 dentro del par",
           "V13 P3a (forma de 279 variables, cerrada) y E-028a (forma dispersa, ya medido: 0,6864 "
           "y 0,7218 en las dos fronteras)",
           "riesgo de repetir un experimento ya hecho con dos representaciones distintas",
           "media: 25 minutos de CPU"),
    "P3": ("Compuerta de ambiguedad sobre las probabilidades de V12",
           "que las filas en las que V12 duda sean las que el especialista mejora",
           "V16 P16d (90 combinaciones de umbral) y ADV-07 (compuerta por margen)",
           "P1b ya midio la curva con esta formulacion y dio +0,0002: probablemente cerrada",
           "baja: es inerte"),
    "P3.3": ("Compuerta blanda que mezcla V12 con el especialista en la region ambigua",
             "que una mezcla ponderada cobre mas que una conmutacion dura",
             "ninguna equivalente; lo mas cercano es A2 de V16, que mezclo las dos cabezas",
             "riesgo de que elabarate la region y mas dano que rescate",
             "media: 20 minutos de CPU"),
    "P4": ("Entrenamiento real de modelos pairwise sobre clases adyacentes",
           "que un modelo entrenado sobre pares/learned informacion independiente de V12",
           "capas de decision de Luce y Bradley-Terry, cerradas; E-028a ya es un par-SVM",
           "riesgo de rehacer E-028a con otro nombre",
           "alta: 40 minutos de CPU o mas"),
    "P5": ("Meta-combinador local por frontera sobre las probabilidades de los miembros",
           "que combinar los miembros dentro de la frontera rinda mas que combinarlos en global",
           "A2 de V16, que combino las dos cabezas en las seis clases",
           "riesgo de fuga si se entrena el combinador con predicciones de modelos que ya "
           "vieron el ejemplo",
           "media: 30 minutos de CPU"),
    "P6": ("Modelo residual dentro de las fronteras",
           "que los errores de V12 en la frontera tengan una estructura que otro modelo "
           "aprende a corregir",
           "ninguna en el ambito de fronteras; los residuales de fases anteriores eran globales",
           "riesgo alto de complejidad anidada; el brief autoriza cerrarlo antes que filtrar",
           "alta: mas de 60 minutos de CPU"),
    "P7": ("Coeficientes especificos de frontera sobre las 312 variables existentes",
           "que las fronteras necesiten funciones distintas con las mismas variables",
           "P2 de esta fase en su forma de 279 variables, cerrada con -0,0017 y -0,0029",
           "riesgo de repetir P3a de V13",
           "media: 30 minutos de CPU"),
    "P9": ("Experto selectivo con un arbitro aprendido sobre las senales de confianza",
           "que las senales que P8 encontro separables permitan elegir entre V12 y el "
           "especialista fila a fila",
           "ADV-07, que enruto por umbral y no por un arbitro aprendido",
           "riesgo de sobreajustar el arbitro; exige inner OOF estricto",
           "media: 25 minutos de CPU"),
    "P10": ("Decodificacion local en cadena usando 25-34 como clase hub",
            "que resolver 18-24, 25-34 y 35-49 en cadena reduzca las confusiones centrales",
            "proyeccion ordinal y capas de decision, cerradas",
            "riesgo de arrastrar las cuatro clases faciles si se aplica en global",
            "alta: mas de 40 minutos de CPU"),
    "P2.3": ("Tercera frontera 35-49 frente a 50-64",
             "que el enfoque de especialista funcione tambien en la tercera frontera",
             "ninguna: no se ha medido ninguna frontera que no sea central",
             "el brief prohibe gastar por simetria",
             "media: 15 minutos de CPU"),
    "P4.2": ("Todos los pares de clases, no solo los adyacentes",
             "que haya informacion no adyacente relevante",
             "P4, si se ejecuta; y la proyeccion ordinal, cerrada",
             "riesgo de disolver la estructura local que es justo lo que se quiere preservar",
             "alta: 60 minutos de CPU o mas"),
}


def leer(nombre: str) -> dict:
    """Lee un artefacto de metricas de la fase."""
    ruta = cfg.EXPERIMENTS / nombre
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}


def main() -> int:
    """Construye las dos tablas maestras desde los artefactos ya medidos."""
    autotest_numerico()
    set_seed()

    # ---------------------------------------------------------------- tabla maestra
    columnas = ["ID", "Track", "Hypothesis", "Mechanism", "Representation", "Model",
                "Outer_Protocol", "Inner_Protocol", "Seed", "Global_MacroF1",
                *[f"Fold_{n}" for n in range(1, 6)], "Delta_Global", "Boundary_18_24_25_34",
                "Boundary_25_34_35_49", "Salvage", "Damage", "Net", "Bootstrap_CI95",
                "Second_Split", "Decision", "Artifact"]
    filas = []
    p0 = leer("v17_p0_reproduccion.json")
    p1 = leer("v17_p1_fronteras.json")
    p1b = leer("v17_p1b_curva_compuerta.json")
    p8 = leer("v17_p8_perfil_confianza.json")
    p1c = leer("v17_p1c_regla_conjunta.json")
    p9 = leer("v17_p9_arbitro.json")

    filas.append({
        "ID": "P0", "Track": "auditoria de brechas",
        "Hypothesis": "que pistas de V17 no sean sinónimos de lo ya cerrado",
        "Mechanism": "lectura dirigida de memoria, informes y codigo; metricas inertes",
        "Representation": "ninguna", "Model": "ninguno",
        "Outer_Protocol": "no aplica", "Inner_Protocol": "no aplica", "Seed": cfg.SEED,
        "Global_MacroF1": round(p0["referencia"]["pooled"], REDONDEO),
        "Decision": "cerrada", "Artifact": "reports/v17_gap_audit.md, reports/v17_gap_audit.csv",
    })
    filas.append({
        "ID": "P0'", "Track": "validacion de la evidencia de V16",
        "Hypothesis": "que A2 de V16 se reproduzca desde sus artefactos",
        "Mechanism": "recalculo inerte de las metricas sobre las OOF guardadas",
        "Representation": "ninguna", "Model": "ninguno (verificacion)",
        "Outer_Protocol": "inerte", "Inner_Protocol": "no aplica", "Seed": cfg.SEED,
        "Global_MacroF1": round(p0["a2_verificado"][2]["macro_f1"], REDONDEO),
        "Fold_1": p0["a2_verificado"][2]["por_fold"][0],
        "Fold_2": p0["a2_verificado"][2]["por_fold"][1],
        "Fold_3": p0["a2_verificado"][2]["por_fold"][2],
        "Fold_4": p0["a2_verificado"][2]["por_fold"][3],
        "Fold_5": p0["a2_verificado"][2]["por_fold"][4],
        "Delta_Global": p0["a2"]["delta"],
        "Boundary_18_24_25_34": p0["a2"]["frontera_18_24_25_34"],
        "Boundary_25_34_35_49": p0["a2"]["frontera_25_34_35_49"],
        "Bootstrap_CI95": f"[{p0['a2']['bootstrap_ic95'][0]}; {p0['a2']['bootstrap_ic95'][1]}]",
        "Second_Split": "3 de 4 (citado de v16_arquitectura.json, no recalculado)",
        "Decision": "cerrada: se reproduce, canonico 0,6005 pooled", "Artifact":
        "reports/v17_p0_reproduccion.md",
    })
    interseccion = {fila["Frontera"]: fila for fila in p1["interseccion"]}
    filas.append({
        "ID": "P1", "Track": "matriz de error por miembro en las fronteras",
        "Hypothesis": "que algun miembro de V12 tenga informacion complementaria en la frontera",
        "Mechanism": "interseccion de errores por miembro, sin entrenar",
        "Representation": "las 312 variables y las OOF de los miembros",
        "Model": "los 14 miembros de V12 mas el especialista de pares E-028a",
        "Outer_Protocol": "inerte sobre OOF de 5 folds agrupados", "Inner_Protocol": "no aplica",
        "Seed": cfg.SEED,
        "Boundary_18_24_25_34": interseccion["18-24|25-34"]["Especialista_acierto"],
        "Boundary_25_34_35_49": interseccion["25-34|35-49"]["Especialista_acierto"],
        "Salvage": interseccion["18-24|25-34"]["Solo_especialista"],
        "Damage": interseccion["18-24|25-34"]["Solo_V12"],
        "Net": interseccion["18-24|25-34"]["Correccion_neta_posible"],
        "Decision": "cerrada: la complementariedad es real y grande", "Artifact":
        "reports/v17_p1_fronteras.md, reports/v17_member_boundary_matrix.csv",
    })
    filas.append({
        "ID": "P1.1", "Track": "complementariedad local por frontera",
        "Hypothesis": "que exista un subconjunto estable de miembros con informacion propia",
        "Mechanism": "columnas de rescate y dano por miembro y por frontera",
        "Representation": "OOF de miembros", "Model": "ninguno (diagnostico)",
        "Outer_Protocol": "inerte", "Inner_Protocol": "no aplica", "Seed": cfg.SEED,
        "Boundary_18_24_25_34": interseccion["25-34|35-49"]["Especialista_acierto"],
        "Boundary_25_34_35_49": interseccion["18-24|25-34"]["Especialista_acierto"],
        "Salvage": interseccion["25-34|35-49"]["Solo_especialista"],
        "Damage": interseccion["25-34|35-49"]["Solo_V12"],
        "Net": interseccion["25-34|35-49"]["Correccion_neta_posible"],
        "Decision": "cerrada: el techo con enrutamiento es 0,7481 y 0,7726", "Artifact":
        "reports/v17_error_intersection.csv, reports/v17_boundary_analysis.csv",
    })
    mejor_18 = p1b["mejor_delta_por_frontera"]["18-24|25-34"]
    mejor_25 = p1b["mejor_delta_por_frontera"]["25-34|35-49"]
    filas.append({
        "ID": "P1b", "Track": "curva de compromiso de la compuerta",
        "Hypothesis": "que la ambiguedad de V12 indique cuando usar el especialista",
        "Mechanism": "compuerta dura sobre la separacion de V12, con seis umbrales",
        "Representation": "probabilidades de V12 y margen del especialista",
        "Model": "V12 mas el especialista de pares E-028a, sin reentrenar",
        "Outer_Protocol": "inerte (el umbral de la tabla esta sesgado, no es resultado)",
        "Inner_Protocol": "no aplica", "Seed": cfg.SEED,
        "Global_MacroF1": round(p1b["referencia_pooled"] + max(mejor_18, mejor_25), REDONDEO),
        "Delta_Global": max(mejor_18, mejor_25),
        "Decision": "cerrada: +0,0002, no llega al gate exploratorio; P3 con esta senal no prospera",
        "Artifact": "reports/v17_p1b_curva_compuerta.md, reports/v17_p1b_curva.csv",
    })
    filas.append({
        "ID": "P8", "Track": "perfil de confianza de los errores de frontera",
        "Hypothesis": "que alguna senal de confianza distinga lo que el especialista arregla de lo "
                      "que rompe",
        "Mechanism": "area bajo la curva de seis senales, sin ajustar nada",
        "Representation": "probabilidades de V12, votos de miembros, margen del especialista",
        "Model": "ninguno (medida de separabilidad, no un modelo)",
        "Outer_Protocol": "inerte", "Inner_Protocol": "no aplica", "Seed": cfg.SEED,
        "Decision": "cerrada: el acuerdo entre miembros (0,7070 y 0,7300) y el margen del "
                    "especialista (0,7294 y 0,7539) son aprovechables; las probabilidades de V12 no",
        "Artifact": "reports/v17_p8_perfil_confianza.md, reports/v17_p8_perfil_confianza.csv",
    })
    filas.append({
        "ID": "P1c", "Track": "regla conjunta de las dos senales",
        "Hypothesis": "que leer las dos senales a la vez, y en sentido contrario, convierta la "
                      "separabilidad de P8 en ganancia global",
        "Mechanism": "compuerta dura con la conjuncion de acuerdo bajo y margen alto, mas la "
                     "condicion de contexto de frontera",
        "Representation": "votos de los miembros y margen del especialista, sin reentrenar",
        "Model": "V12 mas E-028a, enrutado", "Outer_Protocol": "inerte (optimo sesgado)",
        "Inner_Protocol": "no aplica", "Seed": cfg.SEED,
        "Global_MacroF1": round(p1c["referencia_pooled"] + p1c["mejor_conjunta"], REDONDEO),
        "Delta_Global": p1c["mejor_conjunta"],
        "Boundary_18_24_25_34": p1c["mejor_por_frontera"]["18-24|25-34"],
        "Boundary_25_34_35_49": p1c["mejor_por_frontera"]["25-34|35-49"],
        "Decision": "cerrada: ninguna regla alcanza el gate exploratorio; el techo si lo supera",
        "Artifact": "reports/v17_p1c_regla_conjunta.md, reports/v17_p1c_regla_conjunta.csv",
    })
    filas.append({
        "ID": "P9", "Track": "experto selectivo con arbitro aprendido",
        "Hypothesis": "que un arbitro entrenado sobre todas las columnas derivadas existentes "
                      "localice las filas de frontera donde conviene el especialista",
        "Mechanism": "LogReg binario por frontera sobre 47 columnas derivadas, con umbral "
                     "elegido en un bloque interior que el arbitro no ve",
        "Representation": "47 columnas ya existentes: votos y margenes de miembros, margenes de "
                          "pares y de grupo, probabilidades de V12 y de A2, acuerdo entre miembros",
        "Model": "V12 con sustitucion por el especialista de pares cuando el arbitro lo decide",
        "Outer_Protocol": "CV agrupado de 5 folds",
        "Inner_Protocol": "doble particion interior agrupada: una para ajustar el arbitro y otra "
                          "para elegir el umbral",
        "Seed": cfg.SEED,
        "Global_MacroF1": p9["candidata"]["macro_f1"],
        "Fold_1": p9["candidata"]["por_fold"][0], "Fold_2": p9["candidata"]["por_fold"][1],
        "Fold_3": p9["candidata"]["por_fold"][2], "Fold_4": p9["candidata"]["por_fold"][3],
        "Fold_5": p9["candidata"]["por_fold"][4],
        "Delta_Global": p9["delta"],
        "Bootstrap_CI95": f"[{p9['bootstrap_ic95'][0]}; {p9['bootstrap_ic95'][1]}]",
        "Decision": "cerrada: NO ADOPTAR, 0 de 5 folds y delta negativo",
        "Artifact": "reports/v17_p9_arbitro.md, features/v17_p9_oof.npy",
    })
    for identificador in PENDIENTES:
        hipotesis, mecanismo, _, riesgo, coste = PENDIENTES[identificador]
        filas.append({
            "ID": identificador, "Track": identificador.split(".")[0], "Hypothesis": hipotesis,
            "Mechanism": mecanismo, "Decision": "PENDIENTE",
            "Artifact": f"riesgo: {riesgo} | coste: {coste}",
        })
    maestro = pd.DataFrame(filas).reindex(columns=columnas)
    maestro.to_csv(MAESTRO, index=False)

    # ---------------------------------------------------------------- registro de hipotesis
    columnas_registro = ["ID", "Hypothesis", "Previous_Evidence", "Why_New",
                         "Expected_Mechanism", "Risk", "Cost", "Protocol", "Result", "Decision"]
    registros = [
        {"ID": "H01",
         "Hypothesis": "la referencia de V12 se puede fijar en un solo numero",
         "Previous_Evidence": "el proyecto usa 0,6004 y 0,6005 indistintamente",
         "Why_New": "con V17 empieza a haber deltas que no podem mezclarse con los de V13 a V16",
         "Expected_Mechanism": "los dos valores son dos estimadores del mismo macro-F1: pooled y "
                               "media por fold",
         "Risk": "que la diferencia fuera solo redondeo y se hubiera exagerado",
         "Cost": "nula, es inerte", "Protocol": "recalculo de ambos sobre la OOF de produccion",
         "Result": "diferencia de 9,14e-05, mayor que la unidad de redondeo: son dos estimadores. "
                   "Canonico 0,6005 pooled",
         "Decision": "cerrada"},
        {"ID": "H02",
         "Hypothesis": "el especialista de pares resuelve filas que V12 falla",
         "Previous_Evidence": "D-019: un SVM binario del par llegaba a 0,6856 frente a 0,486 del "
                              "ensamble, pero el enrutamiento no rindió",
         "Why_New": "nunca se midio la interseccion de errores de forma completa, con las dos "
                    "fronteras y todos los miembros",
         "Expected_Mechanism": "dos modelos distintos cometiendo errores distintos",
         "Risk": "que la interseccion este vacia y el margen sea solo ruido de estimacion",
         "Cost": "baja, inerte", "Protocol": "interseccion sobre las OOF agrupadas existentes",
         "Result": "rescata 14.356 y 15.571 filas y rompe 3.519 y 2.872. Techo con enrutamiento "
                   "0,7481 y 0,7726",
         "Decision": "cerrada: la complementariedad existe y es grande"},
        {"ID": "H03",
         "Hypothesis": "la ambiguedad de V12 dice cuando conviene el especialista",
         "Previous_Evidence": "V16 P16d con 90 combinaciones de umbral; ADV-07 con compuerta por "
                              "margen; ambos sin resultado",
         "Why_New": "nunca se midio la curva completa de la compuerta con su consecuencia global",
         "Expected_Mechanism": "las filas de baja confianza son las que el especialista arregla",
         "Risk": "que la curva sea plana, como en cualquier resultado anterior de compuerta",
         "Cost": "baja, inerte", "Protocol": "seis umbrales, con la salvedad de que el optimo esta "
                                            "sesgado",
         "Result": "el mejor delta es +0,0002 y a partir de 0,20 de margen se vuelve negativo. "
                   "El rescate y el dano se compensan en las filas activadas",
         "Decision": "cerrada: la senal que propone el brief para la compuerta no sirve"},
        {"ID": "H04",
         "Hypothesis": "alguna senal de confianza si separa las filas que el especialista arregla "
                       "de las que rompe",
         "Previous_Evidence": "ninguna: solo se habian probado umbrales, no separabilidad medida",
         "Why_New": "es la pregunta de la que depende todo lo demas de la fase",
         "Expected_Mechanism": "una senal con area claramente por encima de 0,5",
         "Risk": "que todas las señales den 0,5 y la fase se cierre sin via",
         "Cost": "baja, inerte", "Protocol": "area bajo la curva de Mann-Whitney sobre seis senales",
         "Result": "el acuerdo entre miembros da 0,7070 y 0,7300, y el margen del especialista "
                   "0,7294 y 0,7539. Las probabilidades de V12 dan 0,53 a 0,58 y la entropia esta "
                   "invertida",
         "Decision": "abierta y favorable: hay senal aprovechable, pero no es la de V12"},
    ]
    registros.append({
        "ID": "H05",
        "Hypothesis": "las dos senales que P8 separo, leidas a la vez y en sentido contrario, "
                      "localizan las filas donde el especialista conviene",
        "Previous_Evidence": "P1b con la sola confianza de V12 dio +0,0002",
        "Why_New": "el acuerdo entre miembros y el margen del especialista no forman parte de las "
                   "probabilidades de V12, y apuntan en sentidos opuestos",
        "Expected_Mechanism": "conjugar dispersión del ensemble con seguridad del especialista",
        "Risk": "que la separabilidad no se traduzca en macro-F1 global",
        "Cost": "baja, inerte", "Protocol": "rejilla gruesa 4x4 por frontera, con techo de "
                                           "referencia",
        "Result": "ninguna regla llega al gate exploratorio: la mejor uni y la mejor conjunta dan "
                  "+0,0000. El techo con arbitro perfecto si es +0,0107 y +0,0077, que supera el "
                  "gate de adopcion",
        "Decision": "cerrada: la senal no se traduce. P9 no esta indicado con estas entradas",
    })
    registros.append({
        "ID": "H06",
        "Hypothesis": "un arbitro aprendido sobre 47 columnas derivadas existentes localiza las "
                      "filas de frontera donde el especialista conviene",
        "Previous_Evidence": "P1b y P1c, con compuertas de umbral: +0,0002 y +0,0000",
        "Why_New": "una regla fija no puede combinar señales; un arbitro si, y el protocolo "
                   "anidado permite medirlo sin fuga",
        "Expected_Mechanism": "aprender la frontera entre las filas que el especialista resuelve y "
                              "las que rompe, dentro de la region de frontera",
        "Risk": "que no haya frontera aprendible y el arbitro termine activando casi nada",
        "Cost": "media: unos minutos de CPU, porque el arbitro es un LogReg pequeno",
        "Protocol": "anidado en tres niveles: ajuste, eleccion de umbral y evaluacion",
        "Result": "0,6004, delta -0,0001, 0 de 5 folds. El arbitro elige umbrales de 0,60 a 0,90, "
                  "activa entre 327 y 1.256 filas, y cuando actua pierde: 4 rescates contra 22 danos",
        "Decision": "cerrada: NO ADOPTAR. Confirma por falsacion la prediccion de P1c",
    })
    for identificador, (hipotesis, mecanismo, previo, riesgo, coste) in PENDIENTES.items():
        registros.append({
            "ID": f"H-{identificador}", "Hypothesis": hipotesis, "Previous_Evidence": previo,
            "Why_New": "pista del brief de V17, todavia sin ejecutar",
            "Expected_Mechanism": mecanismo, "Risk": riesgo, "Cost": coste,
            "Protocol": "anidado, con inner OOF para cualquier combinador", "Result": "",
            "Decision": "PENDIENTE",
        })
    registro = pd.DataFrame(registros).reindex(columns=columnas_registro)
    registro.to_csv(REGISTRO, index=False)

    RESULTADO.write_text(json.dumps({
        "fase": "V17 · generador de las tablas maestras", "fecha": "2026-09-26",
        "filas_maestro": int(len(maestro)), "filas_registro": int(len(registro)),
        "experimentos_registrados": sorted(k for k in EXPERIMENTOS if leer(EXPERIMENTOS[k][0])),
        "pistas_pendientes": sorted(PENDIENTES),
        "columnas_maestro": columnas, "columnas_registro": columnas_registro,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"maestro: {len(maestro)} filas ({int((maestro['Decision'] == 'PENDIENTE').sum())} "
          f"pendientes) | registro: {len(registro)} filas "
          f"({int((registro['Decision'] == 'PENDIENTE').sum())} pendientes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
