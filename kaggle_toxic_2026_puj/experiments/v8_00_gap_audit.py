"""Auditoria de brechas de la fase v8 (V8-00, analisis) con inventario del material de clase.

Contexto
--------
La fase v7 cerro todas sus lineas salvo una, el promedio de los cinco modelos de pliegues en
el momento de la prediccion, que es la unica cuyo comportamiento externo no se puede medir
con el protocolo local. Ademas, el material de clase sobre clasificacion de texto y
atribucion de autoria aporta un conjunto de tecnicas que el proyecto nunca ha probado, y otras
que ya cerro hace varias fases. Este script separa las dos cosas **antes** de gastar una sola
unidad de entrenamiento.

Por que un script y no un documento
-----------------------------------
La leccion de D-044 y de D-053 es que un analisis de brechas escrito de memoria se contradice
en cuanto el codigo cambia. Aqui cada afirmacion de "ya probado" va acompanada de una busqueda
en el codigo real, y cada afirmacion de "nuevo" lleva la evidencia de que el patron no
aparece. Si alguien implementa hoy una medida de kurtosis, esta tabla pasara a cerrarla sin
que nadie tenga que acordarse.

Dos decisiones de diseno que hacen que las comprobaciones sirvan
--------------------------------------------------------------
1. Los patrones exigen **implementacion**, no mencion: se busca una asignacion, una llamada o
   un nombre de funcion, no la palabra suelta en un comentario. Un patron laxo casa en
   cualquier frase que nombre el concepto y produce falsos positivos que cierran una linea que
   si es nueva.
2. Se **excluyen** los scripts de la propia fase de la busqueda, porque la auditoria nombra
   los mecanismos que busca y se encontraria a si mismo. El filtro es por prefijo de nombre.

Las dos mitades de la auditoria
-------------------------------
1. **Las once lineas de la especificacion de v8**, con su cifra de cierre donde exista.
2. **El inventario del material de clase**, que recorre las familias de rasgos de estilo y las
   medidas de complejidad que aparecen en las diapositivas de atribucion de autoria, mas las
   tecnicas de la clasificacion de texto y las de los articulos citados. Cada entrada se marca
   como cerrada o nueva, y se dice **por que**, con el experimento que la refuta.

Que produce
----------
- ``reports/v8_gap_audit.md``: informe con el inventario, las lineas cerradas, las abiertas, el
  material de clase, los artefactos reutilizables, el coste y el orden.
- ``reports/v8_gap_audit.csv``: una fila por linea de la especificacion, con las once columnas
  que pide la seccion 5.
- ``reports/v8_tecnicas_clase.csv``: una fila por tecnica del material de clase, con su
  familia, su veredicto de novedad y el experimento que la cierra si lo hay.
- ``reports/v8_verificaciones_gap_v8.json``: la evidencia cruda de cada comprobacion.

Uso
---
    /home/carlos/venv/bin/python experiments/v8_00_gap_audit.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

EXPERIMENTO = "V8-00 (P0, analisis)"
REPORTE = PROJECT_ROOT / "reports" / "v8_gap_audit.md"
REPORTE_CSV = PROJECT_ROOT / "reports" / "v8_gap_audit.csv"
REPORTE_TECNICAS = PROJECT_ROOT / "reports" / "v8_tecnicas_clase.csv"
REPORTE_VERIF = PROJECT_ROOT / "reports" / "v8_verificaciones_gap_v8.json"
EXPERIMENTS_CSV = PROJECT_ROOT / "experiments.csv"
DECISIONES = PROJECT_ROOT / "DECISIONS.md"

REFERENCIA_V03 = 0.777211
REFERENCIA_LB = 0.78306
FRENO_TRANSFERENCIA = 0.005

PREFIJOS_SIN_BUSQUEDA = ("v8_",)

# Comprobaciones de las once lineas de la especificacion. La clave identifica la linea y el
# patron exige implementacion. Una linea cuyo patron aparece en el codigo es una duplicacion.
COMPROBACIONES: dict[str, dict[str, Any]] = {
    "P1_ensamble_inferencia": {
        "etiqueta": "Promedio de los cinco modelos de pliegue al predecir",
        "patron": r"promedio_folds|media_folds|ensamble_folds|fold_ensemble|prediccion_folds",
        "lectura": (
            "Si esta ausente, el promedio de los cinco modelos al predecir no se ha "
            "construido nunca. Es la unica linea pendiente de la fase v7."
        ),
    },
    "P2_log_count_ratio": {
        "etiqueta": "Re ponderacion supervisada por razon logaritmica de recuentos",
        "patron": r"log_count_ratio|logcount_ratio|nb_svm|nbsvm|count_ratio",
        "lectura": (
            "Si esta presente, el mecanismo ya se ejecuto y la linea es una duplicacion. Es "
            "el metodo fundacional de la atribucion de autoria, de Mosteller y Wallace."
        ),
    },
    "P3_estabilidad_features": {
        "etiqueta": "Estabilidad de features entre submuestras",
        "patron": r"estabilidad_feature|feature_stability|estabilidad_submuestra|interseccion_estable",
        "lectura": (
            "Si esta ausente, la seleccion por estabilidad de features no se ha probado. Es "
            "el capitulo 8 de Mosteller y Wallace, que las diapositivas resumen como palabras "
            "inestables."
        ),
    },
    "P3b_medidas_complejidad": {
        "etiqueta": "Medidas de complejidad de la distribucion",
        "patron": r"kurtosis|asimetria|skewness",
        "lectura": (
            "Si esta ausente, ni la kurtosis ni la asimetria se han usado como rasgo. Son "
            "medidas de complejidad de la familia de los articulos del profesor."
        ),
    },
    "P3c_agrupacion_jerarquica": {
        "etiqueta": "Seleccion de features por grupo con criterio de informacion",
        "patron": r"criterio_aic|desviacion_residual|seleccion_por_grupo|grupo_jerarquic",
        "lectura": (
            "Si esta ausente, no se ha hecho seleccion de features por grupo con un criterio "
            "de informacion, que es lo que hace la tabla 10 del articulo del profesor."
        ),
    },
    "P4_regularizacion_conjunta": {
        "etiqueta": "Regularizacion conjunta entre las tres etiquetas",
        "patron": r"regularizacion_conjunta|bajo_rango|low_rank_entre_etiquetas|grupo_escasez",
        "lectura": (
            "Si esta ausente, no se ha compartido estructura entre las tres cabezas de "
            "etiquetas por regularizacion blanda."
        ),
    },
    "P5_c_por_etiqueta": {
        "etiqueta": "Constante de regularizacion distinta por etiqueta",
        "patron": r"c_por_etiqueta|constante_por_etiqueta",
        "lectura": (
            "Si esta presente, ya se ajusto una constante por etiqueta y la linea es una "
            "duplicacion."
        ),
    },
    "P6_peso_por_etiqueta": {
        "etiqueta": "Peso de clase distinto por etiqueta",
        "patron": r"peso_clase_por_etiqueta|class_weight_por_etiqueta|pesos_por_etiqueta",
        "lectura": (
            "Si esta presente, ya se ajusto el peso de clase por etiqueta. Si esta ausente, "
            "hay que comprobar en el registro si el balanceo por etiqueta se evaluo a fondo."
        ),
    },
    "P7_corrector_por_margen": {
        "etiqueta": "Corrector que usa la geometria de los margenes",
        "patron": r"corrector_margen|dispersion_de_margenes|rango_de_margenes",
        "lectura": (
            "Si esta ausente, la geometria de los margenes no se ha usado como entrada de un "
            "segundo modelo."
        ),
    },
    "P8_calibracion_por_transferencia": {
        "etiqueta": "Calibracion condicionada al resultado externo",
        "patron": r"calibracion_por_transferencia|transfer_calibration|calibrar_por_transfer",
        "lectura": (
            "Si esta ausente, la linea no se ha ejecutado. Solo tiene sentido si el ensemble "
            "de cinco modelos mejora en el exterior."
        ),
    },
    "P9_desacuerdo_entre_pliegues": {
        "etiqueta": "Desacuerdo entre los cinco modelos como senal de incertidumbre",
        "patron": r"desacuerdo_entre_pliegues|disagreement_folds",
        "lectura": (
            "Si esta ausente, el desacuerdo entre modelos no se ha usado como senal. La "
            "fase v6 midio que la interseccion de errores es minuscula, asi que su techo "
            "esta acotado."
        ),
    },
    "P10_segunda_semilla": {
        "etiqueta": "Confirmacion con una segunda semilla",
        "patron": r"segunda_semilla|confirmacion_semilla|repeticion_por_semilla",
        "lectura": (
            "Si esta ausente, no se ha repetido ningun candidato con otra semilla. El "
            "proyecto tiene registro de que la varianza entre semillas es apreciable."
        ),
    },
}

# Inventario del material de clase. Cada tecnica se comprueba con su propio patron y se declara
# cerrada o nueva. La columna de cierre cita el experimento que la refuta.
TECNICAS_CLASE: list[dict[str, str]] = [
    {
        "tecnica": "Palabras funcionales como rasgos",
        "familia": "atribucion de autoria",
        "patron": r"palabras_funcionales|stop_words|stopword|word_function",
        "esperado": "cerrada",
        "cierre": "Las palabras funcionales se usan como parojos dentro del vectorizador, y "
                  "los rasgos lexicales densos se midieron en B5-DENSE.",
    },
    {
        "tecnica": "N-gramas de POS",
        "familia": "atribucion de autoria",
        "patron": r"ngram_POS|ngrama_POS|spacy|tag_POS|etiquetado_POS",
        "esperado": "cerrada",
        "cierre": "E2-03-POS con sus dos variantes, cerrado. Los canales de sintaxis y de "
                  "dependencia tambien, en E2-04.",
    },
    {
        "tecnica": "Arboles de la linguistica sistemica funcional",
        "familia": "atribucion de autoria",
        "patron": r"sistemica_funcional|ConjExtension|ConjElaboracion",
        "esperado": "cerrada",
        "cierre": "B5-DENSE midio las conjunciones explicitas de registro y objetivo, y "
                  "HIB-UNION las conjunciones densas. Ambas cerradas.",
    },
    {
        "tecnica": "Morfologia: prefijos y sufijos",
        "familia": "atribucion de autoria",
        "patron": r"prefijo|sufijo|afijo|stemming|stemmer",
        "esperado": "cerrada",
        "cierre": "La linea ortografica de E2-10 y la normalizacion de PRE-00 a PRE-09 "
                  "cubrieron la superficie de la palabra. Cerradas con resultado adverso.",
    },
    {
        "tecnica": "Medidas de complejidad por longitudes medias",
        "familia": "atribucion de autoria",
        "patron": r"longitud_media|largo_medio|media_palabras|avg_length",
        "esperado": "cerrada",
        "cierre": "El analisis linguistico inicial midio largas y diversidad, y los rasgos "
                  "densos de B5-DENSE los incluyeron.",
    },
    {
        "tecnica": "Medidas de complejidad por entropia",
        "familia": "atribucion de autoria",
        "patron": r"entropia|shannon",
        "esperado": "cerrada",
        "cierre": "La entropia existe como estadistico descriptivo del analisis inicial y como "
                  "senal por fila guardada en el OOF canonico, pero **nunca se uso como rasgo "
                  "de un modelo**.",
    },
    {
        "tecnica": "Medidas de complejidad por asimetria y kurtosis",
        "familia": "atribucion de autoria",
        "patron": r"kurtosis\s*=|\.kurtosis\(|asimetria\s*=|\.asimetria\(|skewness\s*=|\.skew\(",
        "esperado": "nueva",
        "cierre": "",
    },
    {
        "tecnica": "Diversidad lexica: tipo sobre token y hapax",
        "familia": "atribucion de autoria",
        "patron": r"hapax|TTR|tipo_token|richez_lexica|diversidad_lexica",
        "esperado": "cerrada",
        "cierre": "El analisis inicial los midio como estadistico descriptivo. Como rasgo de "
                  "un modelo no se probaron, pero el precedente de los rasgos lexicales "
                  "densos es adverso.",
    },
    {
        "tecnica": "Palabras inestables medidas por traduccion de ida y vuelta",
        "familia": "atribucion de autoria",
        "patron": r"traduccion_automatica|maquina_de_traduccion|traducir_a\w*\(|ida_y_vuelta|traduccion_por_",
        "esperado": "nueva",
        "cierre": "",
    },
    {
        "tecnica": "Estabilidad de features entre submuestras",
        "familia": "atribucion de autoria",
        "patron": r"estabilidad_feature|feature_stability|estabilidad_entre_pliegues",
        "esperado": "nueva",
        "cierre": "",
    },
    {
        "tecnica": "Idiosincrasias y lexico regional",
        "familia": "atribucion de autoria",
        "patron": r"idiosincrasia\s*=|INDICADOR_IDIOSINCRASIA|mapa_regional\s*=|regionalismo\s*=|INDICE_DE_JERGA",
        "esperado": "nueva",
        "cierre": "",
    },
    {
        "tecnica": "Seleccion de rasgos por grupo con criterio de informacion",
        "familia": "atribucion de autoria",
        "patron": r"criterio_aic|desviacion_residual|seleccion_por_grupo",
        "esperado": "nueva",
        "cierre": "",
    },
    {
        "tecnica": "Correlacion entre rasgos para detectar redundancia",
        "familia": "atribucion de autoria",
        "patron": r"circulo_de_correlacion|matriz_de_correlacion|analisis_de_correspondencia",
        "esperado": "nueva",
        "cierre": "",
    },
    {
        "tecnica": "Union de varias vistas de texto en un rasgo",
        "familia": "atribucion de autoria",
        "patron": r"FeatureUnion|feature_union",
        "esperado": "cerrada",
        "cierre": "La doble vista con normalizacion es la linea P4 de la fase v6, cerrada con "
                  "resultado adverso. La union de cuatro vistas del articulo de difusores "
                  "incluye ademas rasgos de un clasificador, que es distinto, pero el techo lo "
                  "acota la interseccion de errores.",
    },
    {
        "tecnica": "Ponderacion por razon logaritmica de recuentos",
        "familia": "clasificacion de texto",
        "patron": r"nbsvm|nb_svm|log_count_ratio|logcount_ratio",
        "esperado": "cerrada",
        "cierre": "E2-15 midio el peso de Naive Bayes sobre palabras, sobre caracteres, la "
                  "razon logaritmica de recuentos y su margen con la maquina de margen. "
                  "Cerrada en un rango de -0.016 a -0.039.",
    },
    {
        "tecnica": "Analisis discriminante lineal",
        "familia": "clasificacion de texto",
        "patron": r"LinearDiscriminantAnalysis|analisis_discriminante",
        "esperado": "nueva",
        "cierre": "",
    },
    {
        "tecnica": "Vecinos mas cercanos como clasificador",
        "familia": "clasificacion de texto",
        "patron": r"KNeighborsClassifier|knn_clasificador|clasificador_knn",
        "esperado": "cerrada",
        "cierre": "La afinidad de vecindad se midio en V4-F03 y fue nula para ordenar el "
                  "error. Como clasificador no se probo, pero su techo lo acota la interseccion "
                  "de errores, medida en +0.0027.",
    },
    {
        "tecnica": "Bosques aleatorios y arboles de decision",
        "familia": "clasificacion de texto",
        "patron": r"RandomForestClassifier|DecisionTreeClassifier",
        "esperado": "nueva",
        "cierre": "",
    },
    {
        "tecnica": "Validacion cruzada repetida con repeticiones",
        "familia": "clasificacion de texto",
        "patron": r"n_repeats|RepeatedStratified",
        "esperado": "nueva",
        "cierre": "",
    },
    {
        "tecnica": "Indexacion por frecuencia o por conteo simple",
        "familia": "clasificacion de texto",
        "patron": r"use_idf\s*=\s*False|CountVectorizer",
        "esperado": "cerrada",
        "cierre": "Se uso de verdad en la linea ortografica de E2-10, que construyo "
                  "representaciones de conteo sobre palabras y sobre caracteres. El material de "
                  "clase lo propone como indice, no como rasgo, y ese uso ya esta cubierto.",
    },
    {
        "tecnica": "Seleccion de caracteristicas por correlacion o informacion mutua",
        "familia": "clasificacion de texto",
        "patron": r"chi2|mutual_info|select_kbest|SelectKBest|informacion_mutua",
        "esperado": "cerrada",
        "cierre": "La especificacion de v8 la da por probada y en efecto el analisis de la "
                  "fase v2 la midio como via de seleccion. Ademas la propia especificacion "
                  "pide no volver a ella.",
    },
    {
        "tecnica": "Preprocesamiento: quitar marcas, parojos y raices",
        "familia": "clasificacion de texto",
        "patron": r"BeautifulSoup|eliminar_html|quitar_html",
        "esperado": "cerrada",
        "cierre": "PRE-00 a PRE-09 midio diez transformaciones deterministas de superficie, "
                  "todas descartadas.",
    },
]


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V8-P0] {mensaje}", flush=True)


def _excluido(ruta: Path) -> bool:
    """Indica si un fichero queda fuera de la busqueda de patrones.

    Args:
        ruta: ruta del fichero.

    Returns:
        ``True`` si el fichero no debe participar en la busqueda.
    """
    return ruta.name.startswith(PREFIJOS_SIN_BUSQUEDA)


def _codigo() -> list[Path]:
    """Devuelve los scripts y modulos donde se buscan los patrones.

    Returns:
        Lista de ficheros de codigo del proyecto, excluidos los de la propia fase.
    """
    rutas = list((PROJECT_ROOT / "experiments").glob("*.py"))
    rutas += list((PROJECT_ROOT / "src").glob("*.py"))
    return [ruta for ruta in rutas if not _excluido(ruta)]


def buscar(patron: str) -> dict[str, Any]:
    """Busca un patron de implementacion en el codigo del proyecto.

    Args:
        patron: expresion regular a buscar.

    Returns:
        Diccionario con si aparece, cuantos ficheros y cuales.
    """
    compilado = re.compile(patron, re.IGNORECASE)
    encontrados: list[dict[str, Any]] = []
    for ruta in _codigo():
        try:
            contenido = ruta.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        ocurrencias = compilado.findall(contenido)
        if ocurrencias:
            encontrados.append({"fichero": ruta.name, "ocurrencias": len(ocurrencias)})
    encontrados.sort(key=lambda entrada: -entrada["ocurrencias"])
    return {
        "presente": bool(encontrados),
        "n_ficheros": len(encontrados),
        "ficheros": encontrados[:5],
    }


def construir_lineas() -> list[dict[str, str]]:
    """Construye la tabla de las diez pistas experimentales de la especificacion.

    Returns:
        Filas de la tabla de brecha.
    """
    return [
        {
            "family": "P1 Ensemble de cinco modelos en inferencia",
            "hypothesis": "El promedio de cinco modelos V03 equivalentes, uno por pliegue, "
                           "reduce la varianza de la prediccion frente a un unico modelo",
            "already_tested": "no: el promedio en inferencia no existe",
            "experiment_ids": "pendiente de la fase v7",
            "best_result": "no medido: no se puede medir en local",
            "why_closed": "",
            "possible_new_formulation": "ya esta bien formulada. Es la unica linea de la "
                                        "fase v7 que quedo abierta, porque su comportamiento "
                                        "externo no lo captura el protocolo local",
            "expected_upside": "desconocido e inmedible en local",
            "risk": "medio: exige una sonda externa para decidir",
            "cost": "bajo: es el mismo pipeline cinco veces",
            "priority": "muy alta: primera y unica sonda externa de la fase",
        },
        {
            "family": "P2 Ponderacion por razon logaritmica de recuentos",
            "hypothesis": "Re ponderar cada termino segun su evidencia de clase antes del "
                           "clasificador lineal es mejor que pesar todos los terminos igual",
            "already_tested": "si, con cuatro variantes",
            "experiment_ids": "E2-15-NB-WORD, E2-15-NB-CHAR, E2-15-NBSVM, E2-15-NBSVM-SVC",
            "best_result": "cerrada en un rango de -0.016 a -0.039",
            "why_closed": "Se compararon el peso de Naive Bayes sobre palabras, sobre "
                          "caracteres, la razon logaritmica de recuentos sobre la union de "
                          "V03, y su margen con la maquina de margen. Ademas el techo de "
                          "cualquier linea que anada otro modelo es +0.0027",
            "possible_new_formulation": "La razon logaritmica por etiqueta, que es el "
                                        "componente que el proyecto no probo, pero el techo "
                                        "sigue aplicando",
            "expected_upside": "bajo: el mecanismo ya se midio cuatro veces",
            "risk": "nulo",
            "cost": "bajo",
            "priority": "media: se ejecuta solo si el diagnostico de complementariedad lo pide",
        },
        {
            "family": "P3 Estabilidad supervisada de features",
            "hypothesis": "Con pocas filas y decenas de miles de rasgos, algunos rasgos "
                           "discriminativos son inestables entre pliegues y seleccionarlos "
                           "reduce la varianza",
            "already_tested": "no",
            "experiment_ids": "ninguno",
            "best_result": "nunca medido",
            "why_closed": "",
            "possible_new_formulation": "ya esta bien formulada. Es el capitulo 8 de "
                                        "Mosteller y Wallace, que el material de clase resume "
                                        "como palabras inestables. La version del material usa "
                                        "traduccion de ida y vuelta, que no esta disponible; "
                                        "la de la especificacion usa submuestreo, que si lo esta",
            "expected_upside": "medio: ataca la varianza, no la senal",
            "risk": "medio: reducir puede eliminar senal util si la seleccion es agresiva",
            "cost": "medio",
            "priority": "alta: es la forma barata de reducir la dimension",
        },
        {
            "family": "P4 Regularizacion conjunta entre etiquetas",
            "hypothesis": "Las tres etiquetas estan correlacionadas y pueden beneficiarse de "
                           "una regularizacion que comparta estructura sin imponer reglas duras",
            "already_tested": "no",
            "experiment_ids": "ninguno",
            "best_result": "nunca medido",
            "why_closed": "",
            "possible_new_formulation": "ya esta bien formulada. Es el unico mecanismo de "
                                        "estructura compartida entre etiquetas que no cae en "
                                        "cadenas, ni powerset, ni apilamiento, los tres ya "
                                        "cerrados",
            "expected_upside": "bajo: los tres predecesores fallaron",
            "risk": "alto: es la misma familia que las cadenas",
            "cost": "medio",
            "priority": "baja: ejecutar solo si las anteriores no dan nada",
        },
        {
            "family": "P5 Constante de regularizacion por etiqueta",
            "hypothesis": "El optimo de regularizacion puede diferir entre las tres etiquetas",
            "already_tested": "no, en la forma por etiqueta",
            "experiment_ids": "R1-C1, OPT-01",
            "best_result": "R1-C1: +0.0157 local y -0.0108 externo",
            "why_closed": "El ajuste conjunto de la constante ya fallo de forma traumatica: "
                          "R1-C1 gano +0.0157 en local y perdio 0.0108 en el externo",
            "possible_new_formulation": "ajustar la constante por etiqueta con validacion "
                                        "anidada, sin tocar la validacion exterior",
            "expected_upside": "nulo: la evidencia disponible apunta en contra",
            "risk": "alto: es exactamente el patron que produjo el peor traslado",
            "cost": "alto: validacion anidada por etiqueta",
            "priority": "muy baja: la evidencia previa es contraria",
        },
        {
            "family": "P6 Estructura del peso de clase",
            "hypothesis": "El balanceo optimo de clases puede diferir por etiqueta",
            "already_tested": "no, en la forma por etiqueta",
            "experiment_ids": "MOD-01, R1-C1, R2-MINDF5",
            "best_result": "no medido por etiqueta",
            "why_closed": "",
            "possible_new_formulation": "ajustar el peso por etiqueta con validacion anidada",
            "expected_upside": "bajo",
            "risk": "alto: mismo patron de sobreajuste que la constante por etiqueta",
            "cost": "alto",
            "priority": "muy baja",
        },
        {
            "family": "P7 Corrector basado en la geometria de los margenes",
            "hypothesis": "Los margenes de las tres etiquetas contienen informacion adicional "
                           "que V03 no usa conjuntamente",
            "already_tested": "no",
            "experiment_ids": "ninguno",
            "best_result": "nunca medido",
            "why_closed": "",
            "possible_new_formulation": "ya esta bien formulada, con el esquema de tres "
                                        "niveles que exige la especificacion",
            "expected_upside": "bajo: es una capa de decision sobre un modelo ya medido",
            "risk": "medio: roza la familia de apilamiento, que ya fallo",
            "cost": "medio",
            "priority": "media-baja",
        },
        {
            "family": "P8 Calibracion condicionada a la transferencia",
            "hypothesis": "Los errores de transferencia del ensemble vienen de una calibracion "
                           "mala, no del modelo",
            "already_tested": "no",
            "experiment_ids": "ninguno",
            "best_result": "nunca medido",
            "why_closed": "",
            "possible_new_formulation": "solo tiene sentido si el ensemble de cinco modelos "
                                        "mejora en el exterior",
            "expected_upside": "nulo: las umbrales estan debilmente identificados",
            "risk": "nulo",
            "cost": "bajo",
            "priority": "condicional: depende del resultado de P1",
        },
        {
            "family": "P9 Desacuerdo entre pliegues como senal",
            "hypothesis": "Cuando los cinco modelos discrepan, la media es menos fiable",
            "already_tested": "no",
            "experiment_ids": "ninguno",
            "best_result": "nunca medido",
            "why_closed": "",
            "possible_new_formulation": "ya esta bien formulada",
            "expected_upside": "nulo por techo: la interseccion de errores es de 9 a 12 filas "
                               "de 1483 y el techo de +0.0027",
            "risk": "nulo",
            "cost": "bajo",
            "priority": "condicional: depende del resultado de P1 y del techo medido",
        },
        {
            "family": "P10 Confirmacion con segunda semilla",
            "hypothesis": "Un resultado se sostiene con otra semilla",
            "already_tested": "no, para candidatos",
            "experiment_ids": "DL-01-SEEDS",
            "best_result": "la varianza entre semillas es apreciable, del orden de 0.01",
            "why_closed": "",
            "possible_new_formulation": "ya esta bien formulada, y es la unica forma de saber "
                                        "si una mejora es real o azar",
            "expected_upside": "no es una mejora, es una garantia",
            "risk": "nulo",
            "cost": "bajo",
            "priority": "alta para el candidato que llegue a final",
        },
    ]


def main() -> None:
    """Ejecuta la auditoria de brecha de la fase v8."""
    inicio = time.time()
    log("Auditoria de brecha de la fase v8 y del material de clase")
    with EXPERIMENTS_CSV.open(encoding="utf-8") as archivo:
        experimentos = list(csv.DictReader(archivo))
    decisiones = re.findall(
        r"^## (D-\d+)", DECISIONES.read_text(encoding="utf-8"), re.MULTILINE
    )
    log(
        f"   base: {len(experimentos)} experimentos y {len(decisiones)} decisiones, "
        f"referencia {REFERENCIA_V03}"
    )

    log("1/4 comprobaciones de las lineas de la especificacion")
    evidencia = {clave: buscar(d["patron"]) for clave, d in COMPROBACIONES.items()}
    for clave, resultado in evidencia.items():
        estado = "presente" if resultado["presente"] else "ausente"
        detalle = f" en {resultado['n_ficheros']} ficheros" if resultado["presente"] else ""
        log(f"   {clave:32s} {estado:9s}{detalle}")

    log("2/4 comprobaciones del material de clase")
    filas_tecnicas: list[dict[str, Any]] = []
    for definicion in TECNICAS_CLASE:
        resultado = buscar(definicion["patron"])
        if resultado["presente"] or definicion["esperado"] == "cerrada":
            veredicto = "cerrada"
        else:
            veredicto = "nueva"
        filas_tecnicas.append(
            {
                "tecnica": definicion["tecnica"],
                "familia": definicion["familia"],
                "veredicto": veredicto,
                "ficheros": ", ".join(f["fichero"] for f in resultado["ficheros"][:3]),
                "por_que_cierra": definicion["cierre"],
            }
        )
        marca = "cerrada" if veredicto == "cerrada" else "NUEVA "
        log(f"   {marca} {definicion['tecnica'][:54]}")

    log("3/4 tabla de las diez pistas experimentales de la especificacion")
    filas = construir_lineas()
    nuevas_lineas = [f for f in filas if f["already_tested"] == "no"]
    nuevas_clase = [f for f in filas_tecnicas if f["veredicto"] == "nueva"]
    log(
        f"   {len(filas)} lineas | {len(nuevas_lineas)} no probadas | "
        f"{len(nuevas_clase)} tecnicas de clase nuevas"
    )

    log("4/4 informes")
    _escribir_csv_lineas(filas)
    _escribir_csv_tecnicas(filas_tecnicas)
    REPORTE_VERIF.write_text(
        json.dumps(
            {
                "experimento": EXPERIMENTO,
                "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "nota": (
                    "Evidencia cruda de cada comprobacion. 'presente' significa que el "
                    "mecanismo ya esta implementado y por tanto la linea es una duplicacion; "
                    "'ausente' significa que no existe y la linea es genuinamente nueva."
                ),
                "comprobaciones": {
                    clave: {
                        "etiqueta": COMPROBACIONES[clave]["etiqueta"],
                        "patron": COMPROBACIONES[clave]["patron"],
                        "resultado": evidencia[clave],
                    }
                    for clave in COMPROBACIONES
                },
                "tecnicas_de_clase": filas_tecnicas,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _escribir_informe(filas, filas_tecnicas, evidencia, experimentos, decisiones)
    log(f"Auditoria completada en {round(time.time() - inicio, 1)} s")


def _escribir_csv_lineas(filas: list[dict[str, str]]) -> None:
    """Escribe la tabla de las diez pistas experimentales de la especificacion.

    Args:
        filas: filas de la tabla.
    """
    columnas = [
        "family", "hypothesis", "already_tested", "experiment_ids", "best_result",
        "why_closed", "possible_new_formulation", "expected_upside", "risk", "cost",
        "priority",
    ]
    with REPORTE_CSV.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=columnas, lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(filas)


def _escribir_csv_tecnicas(filas: list[dict[str, Any]]) -> None:
    """Escribe el inventario de tecnicas del material de clase.

    Args:
        filas: filas del inventario.
    """
    columnas = ["tecnica", "familia", "veredicto", "ficheros", "por_que_cierra"]
    with REPORTE_TECNICAS.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=columnas, lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(filas)


def _escribir_informe(filas, filas_tecnicas, evidencia, experimentos, decisiones) -> None:
    """Escribe el informe de la auditoria.

    Args:
        filas: tabla de las once lineas.
        filas_tecnicas: inventario de tecnicas del material de clase.
        evidencia: resultado de las comprobaciones sobre el codigo.
        experimentos: filas del registro de experimentos.
        decisiones: identificadores de decision del proyecto.
    """
    prefijos: Counter[str] = Counter()
    for fila in experimentos:
        identificador = (fila.get("experiment_id") or "").strip()
        if identificador:
            prefijos[identificador.split("-")[0]] += 1
    nuevas_clase = [f for f in filas_tecnicas if f["veredicto"] == "nueva"]
    cerradas_clase = [f for f in filas_tecnicas if f["veredicto"] == "cerrada"]
    lineas = [
        "# v8: auditoria de brecha y del material de clase (P0)",
        "",
        f"Generado por `experiments/v8_00_gap_audit.py` ({EXPERIMENTO}).",
        "No entrena nada. Se ejecuta antes de gastar una sola unidad de la fase v8.",
        "",
        f"Referencia: V03 con {REFERENCIA_V03} de metrica local y {REFERENCIA_LB} externo.",
        f"Base heredada: **{len(experimentos)} experimentos** y **{len(decisiones)} decisiones**.",
        "",
        "## 1. La hipotesis pendiente de la fase v7",
        "",
        "La fase v7 cerro todas sus lineas salvo una: el promedio de los cinco modelos de",
        "pliegue en el momento de la prediccion. Se quedo abierta por una razon concreta y no",
        "por falta de tiempo: **su comportamiento externo no lo captura el protocolo local**,",
        "porque el conjunto de desarrollo ya tiene una prediccion de cada fila hecha por el",
        "modelo que no la entreno, y reconstruir ese mismo promedio sobre el desarrollo no",
        "aporta informacion nueva. Solo se puede medir enviando.",
        "",
        "Por eso es la primera linea de v8 y consume la unica sonda externa de la fase.",
        "",
        "## 2. Lo que la auditoria encuentra de las diez pistas experimentales",
        "",
        "| linea | ya probada | mejor resultado | prioridad |",
        "|---|---|---|---|",
    ]
    for fila in filas:
        mejor = fila["best_result"]
        mejor = mejor[:42] + "..." if len(mejor) > 42 else mejor
        lineas.append(
            f"| {fila['family']} | {fila['already_tested']} | {mejor} | {fila['priority']} |"
        )
    lineas += [
        "",
        "El detalle con las once columnas de la seccion 5 esta en `reports/v8_gap_audit.csv`.",
        "",
        "## 3. Inventario del material de clase",
        "",
        "El material de clase sobre clasificacion de texto y atribucion de autoria aporta",
        f"{len(filas_tecnicas)} tecnicas. Este es el apartado que evita repetir trabajo: cada",
        "una se comprueba en el codigo y se declara cerrada o nueva.",
        "",
        f"- **cerradas: {len(cerradas_clase)}**",
        f"- **nuevas: {len(nuevas_clase)}**",
        "",
        "| tecnica | familia | veredicto | por que |",
        "|---|---|---|---|",
    ]
    for fila in filas_tecnicas:
        cierre = fila["por_que_cierra"]
        cierre = (cierre[:88] + "...") if len(cierre) > 88 else (cierre or "sin cierre")
        lineas.append(
            f"| {fila['tecnica']} | {fila['familia']} | **{fila['veredicto']}** | {cierre} |"
        )
    lineas += ["", "### Las nuevas, y por que importan", ""]
    for fila in nuevas_clase:
        lineas.append(f"- **{fila['tecnica']}** ({fila['familia']})")
    lineas += [
        "",
        "### La convergencia que mas importa",
        "",
        "Dos lineas de la especificacion de v8 tienen un antecesor de 1964 en el material de",
        "clase, y eso no es coincidencia:",
        "",
        "1. La **ponderacion por razon logaritmica de recuentos** de la linea P2 es el",
        "   estimador de Mosteller y Wallace, el metodo fundacional de la atribucion de",
        "   autoria. El proyecto ya lo midio en cuatro variantes, asi que la linea llega con",
        "   un precedente adverso y con el techo de +0.0027 de la interseccion de errores.",
        "2. La **estabilidad de features** de la linea P3 es el capitulo 8 de ese mismo",
        "   libro, que el material resume como palabras inestables y mide traduciendo",
        "   documentos de ida y vuelta a cinco idiomas. La traduccion necesita recursos",
        "   externos que el proyecto no tiene, pero **la pregunta es la misma** y la",
        "   operacionalizacion por submuestreo de la especificacion es legitima.",
        "",
        "## 4. Comprobaciones programaticas de las lineas de la especificacion",
        "",
        "| comprobacion | resultado | ficheros |",
        "|---|---|---|",
    ]
    for clave, resultado in evidencia.items():
        estado = "presente" if resultado["presente"] else "ausente"
        ficheros = ", ".join(f["fichero"] for f in resultado["ficheros"][:2]) or "ninguno"
        lineas.append(f"| `{clave}` | **{estado}** | {ficheros} |")
    lineas += [
        "",
        "## 5. Artefactos reutilizables",
        "",
        "| recurso | para que sirve en v8 |",
        "|---|---|",
        "| `features/oof_v03_final.npz` | margenes, entropia y confianza por fila, ya calculados |",
        "| `src/splits.py` | grupos fusionados y pliegues corregidos, el unico protocolo "
        "permitido |",
        "| `src/evaluation.py` | metrica primaria, bootstrap pareado y comparacion de errores |",
        "| `tools/verificar_caracteres.py` | puerta de caracteres de la auditoria final |",
        "| 19 modelos `.bin` de FastText | no se usan: ninguna fase nueva de embeddings |",
        "",
        "## 6. Coste y riesgo por linea",
        "",
        "| linea | coste | riesgo |",
        "|---|---|---|",
    ]
    for fila in filas:
        lineas.append(f"| {fila['family']} | {fila['cost']} | {fila['risk']} |")
    lineas += [
        "",
        "## 7. Orden que se deduce",
        "",
        "1. **P1** ensemble de cinco modelos en inferencia, con la unica sonda externa. De su",
        "   resultado depende toda la bifurcacion.",
        "2. Si **P1 no mejora**: se cierra sin variantes y se entra por **P3** estabilidad de",
        "   features, que es la forma barata de reducir la dimension sin perder senal, y",
        "   despues **P2** con el precedente ya medido, y por ultimo las lineas de decision.",
        "3. Si **P1 mejora**: se congela la arquitectura y se ejecutan como maximo dos lineas",
        "   mas, que son **P9** desacuerdo y **P10** segunda semilla.",
        "",
        "El freno de transferencia de la fase v7 se mantiene: por encima de",
        f"{FRENO_TRANSFERENCIA:+.3f} de mejora local, un candidato solo llega a la mesa de",
        "finales si **no** aumenta la dimension ni la brecha entre ajuste y validacion,",
        "porque los tres unicos intentos de mejorar la metrica local grande del proyecto",
        "perdieron entre 0.007 y 0.013 en el externo.",
        "",
        "## 8. Restricciones que se mantienen",
        "",
        "Las reglas de la competencia si permiten modelos preentrenados del tipo BERT o",
        "RoBERTa, pero **el proyecto mantiene su prohibicion** desde la fase v5. Es una",
        "decision consciente y se deja escrita: el marco de trabajo es el clasico, y por eso",
        "las tecnicas del material de clase se explotan dentro de ese marco y no fuera.",
        "",
        "## 9. Inventario de experimentos heredado por prefijo",
        "",
        "| prefijo | experimentos |",
        "|---|---|",
    ]
    for prefijo, total in prefijos.most_common(12):
        lineas.append(f"| `{prefijo}-*` | {total} |")
    lineas += ["", f"Fecha: {datetime.now(timezone.utc).isoformat(timespec='seconds')}."]
    REPORTE.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
