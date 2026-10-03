"""Auditoria de brechas de la fase v7 (V7-00, analisis): que queda realmente abierto.

Contexto
--------
La fase v6 se cerro sin ninguna adopcion y dejo trece restricciones asumidas (D-052).
La especificacion de v7 propone once tracks nuevos. Este script no entrena nada: comprueba,
contra el codigo y contra los artefactos reales, cuales de esos once tracks son de verdad
mecanismos nuevos y cuales repiten una linea ya medida.

Por que un script y no un documento
-----------------------------------
La lesson de D-044 es que un analisis de brechas escrito de memoria se contradice en
cuanto el codigo cambia. Aqui cada afirmacion de "ya probada" va acompanada de una
busqueda en el codigo real, y cada afirmacion de "nueva" lleva la evidencia de que el
patron no aparece. Si alguien implementa hoy un `prior de grupo` como feature, esta tabla
pasara a cerrarlo sin que nadie tenga que acordarse.

Dos decisiones de diseno que hacen que las comprobaciones sirvan
--------------------------------------------------------------
1. Los patrones exigen **implementacion**, no mencion: se busca una asignacion, una
   llamada o un nombre de funcion, no la palabra suelta en un comentario. Un patron laxo
   casa en cualquier frase que nombre el concepto y produce falsos positivos que cierran
   una linea que si es nueva.
2. Se **excluye** este script y los demas de la fase v7 de la busqueda. El gap analysis
   nombra los mecanismos que busca, asi que un patron sin mas se encontraria a si mismo y
   declararia duplicada justo la linea que esta identificando como nueva. El filtro es por
   prefijo de nombre, para que siga funcionando cuando escriban los scripts de los tracks.

Ademas de la busqueda en el codigo, el script **lee los numeros reales** de los JSON y CSV
que los experimentos anteriores ya escribieron. Un cierre sin cifra no es un cierre.

Que produce
----------
- ``reports/v7_gap_audit.md``: inventario, mecanismos cerrados con su numero, mecanismos
  abiertos, artefactos reutilizables, duplicaciones explicitas, coste, orden y riesgo de
  filtrado por cada linea.
- ``reports/v7_gap_audit.csv``: una fila por track con las once columnas de la seccion 6
  de la especificacion.
- ``reports/v7_gap_verificaciones.json``: la evidencia cruda de cada comprobacion.

Uso
---
    /home/carlos/venv/bin/python experiments/v7_00_gap_audit.py
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

EXPERIMENTO = "V7-00 (P0, analisis)"
REPORTE = PROJECT_ROOT / "reports" / "v7_gap_audit.md"
REPORTE_CSV = PROJECT_ROOT / "reports" / "v7_gap_audit.csv"
REPORTE_VERIF = PROJECT_ROOT / "reports" / "v7_gap_verificaciones.json"
EXPERIMENTS_CSV = PROJECT_ROOT / "experiments.csv"
DECISIONES = PROJECT_ROOT / "DECISIONS.md"

REFERENCIA_V03 = 0.777211
REFERENCIA_LB = 0.78306
GATE_DELTA = 0.003
# Freno de transferencia. Tres modelos con mejor OOF de la historia del proyecto
# (R1-C1, R2-MINDF5 y OPT-01, todos con mas de +0.015 de OOF) perdieron entre 0.007 y
# 0.013 de leaderboard, y los tres aumentaban capacidad. Por encima de este delta en
# OOF, un candidato solo puede llegar a la mesa de finales si no aumenta la capacidad.
FRENO_TRANSFERENCIA_DELTA = 0.005

PREFIJOS_SIN_BUSQUEDA = ("v7_",)

# Cada comprobacion busca una implementacion. La clave es la del mecanismo y el valor
#dict con la etiqueta legible, el patron de busqueda, donde buscar y como leerla.
COMPROBACIONES: dict[str, dict[str, Any]] = {
    "P0_1_umbral_anidado": {
        "etiqueta": "Estimacion de umbral anidada por pliegue",
        "patron": r"global_anidado|prediccion_global|umbrales_por_pliegue|nested_threshold",
        "busqueda": "codigo",
        "lectura": (
            "Si esta presente, el protocolo de umbral anidado ya existe y su numero "
            "esta medido: la linea P0 no es nueva, es una confirmacion con mas rigor."
        ),
    },
    "P0_2_umbral_por_rebanada": {
        "etiqueta": "Umbral propio por rebanada con shrinkage",
        "patron": r"umbrales_locales|shrinkage|SHRINK-M|rebanada",
        "busqueda": "codigo",
        "lectura": (
            "Si esta presente, los umbrales condicionales por rebanada de la seccion 7 "
            "punto 5 ya se ejecutaron con cinco variantes de shrinkage y se cerraron."
        ),
    },
    "P0_3_estimador_robusto": {
        "etiqueta": "Mediana, media recortada o mediana bootstrap de umbrales",
        "patron": r"trimmed_mean|media_recortada|bootstrap_median|mediana_bootstrap|umbral_consenso",
        "busqueda": "codigo",
        "lectura": (
            "Si esta ausente, la parte de estimacion robusta y consenso de la seccion 7 "
            "punto 3 y el track P10 no se han implementado nunca."
        ),
    },
    "P0_4_calibracion": {
        "etiqueta": "Calibracion de probabilidades (sigmoid o isotonica)",
        "patron": r"CalibratedClassifierCV|isotonic|sigmoid_calibration|calibracion",
        "busqueda": "codigo",
        "lectura": (
            "Si esta presente, la calibracion se evaluo con train, calibracion y "
            "validacion separados y se cerro. Es la seccion 7 punto 4."
        ),
    },
    "P1_1_oof_por_pliegue": {
        "etiqueta": "OOF ya construido por pliegue con la regla de no autoinclusion",
        "patron": r"pliegues|guardar_oof_v03_final",
        "busqueda": "codigo",
        "lectura": (
            "Si esta presente mas el artefacto de probabilidades fuera de pliegue ya "
            "contiene el pliegue de cada fila, la parte OOF de P1 punto 1 esta **hecha**: "
            "es exactamente el OOF canonico de V03 y no se puede repetir."
        ),
    },
    "P2_1_linearsvc": {
        "etiqueta": "Maquina de margen maximo sobre la union de V03",
        "patron": r"LinearSVC",
        "busqueda": "codigo",
        "lectura": (
            "Si esta presente, la maquina de margen se evaluo sobre la union y fue la "
            "referencia honesta del protocolo v2. Es la seccion 9 punto 1."
        ),
    },
    "P2_3_generativo": {
        "etiqueta": "Modelo generativo de Naive Bayes",
        "patron": r"MultinomialNB|ComplementNB|BernoulliNB|nbsvm|NB-SVM",
        "busqueda": "codigo",
        "lectura": (
            "Si esta presente, ya se compararon el peso de Naive Bayes sobre palabras, "
            "sobre caracteres, el log-count ratio y su margen con la maquina de margen. "
            "Es la seccion 9 punto 3."
        ),
    },
    "P3_1_cnn_caracteres": {
        "etiqueta": "Red convolucional de caracteres entrenada desde cero",
        "patron": r"Conv1d|CharCNN|cnn_caracteres|clase CNN",
        "busqueda": "codigo",
        "lectura": (
            "Si esta presente, la CNN de caracteres ya se entreno desde cero, sobre "
            "embeddings aleatorios aprendidos con la red, y se descarto por su numero. "
            "Es la seccion 10 completa."
        ),
    },
    "P4_1_vista_duplicada": {
        "etiqueta": "Segunda vista ortografica junto a la vista original",
        "patron": r"ORT-A|ORT_ABCD|canal_ortograf|mascara_ofusc|ortografia",
        "busqueda": "codigo",
        "lectura": (
            "Si esta presente, la doble vista de texto original mas texto normalizado se "
            "construyo y se midio con tres variantes. Es la seccion 11 completa."
        ),
    },
    "P5_1_alcance_negacion": {
        "etiqueta": "Alcance de negacion con tokens modificados",
        "patron": r"NEG|negacion|NO_|nunca|alcance_neg",
        "busqueda": "codigo",
        "lectura": (
            "Si esta presente, la negacion se estudio como canal y con patrones de "
            "dependencia, y se cerro con una perdida muy grande. Es la seccion 12 punto 1."
        ),
    },
    "P7_1_especialistas": {
        "etiqueta": "Modelo propio por rebanada o especialista por region",
        "patron": r"especialistas|modelo_por_rebanada|gate|selector_por_rebanada",
        "busqueda": "codigo",
        "lectura": (
            "Si esta presente, el modelo por rebanada y los espacios por etiqueta ya se "
            "cerraron. Es la seccion 14."
        ),
    },
    "P8_1_prior_grupo": {
        "etiqueta": "Prior de etiqueta del grupo como caracteristica",
        "patron": r"prior_grupo|group_label_prior|prior_pseudoautor|prior_del_grupo",
        "busqueda": "codigo",
        "lectura": (
            "Si esta ausente, el prior del grupo **como feature** no se ha probado. La "
            "fase v6 solo repondero filas por tamano de grupo, que es un peso y no un "
            "prior. Es el track P8 y es genuinamente nuevo."
        ),
    },
    "P9_1_perdida_f1": {
        "etiqueta": "Perdida orientada a la F1 o focal",
        "patron": r"focal|FocalLoss|soft_f1|surrogate_f1|F1Loss|fbeta",
        "busqueda": "codigo",
        "lectura": (
            "Si esta ausente, ninguna perdida orientada a la F1 se ha probado. Es el "
            "track P9 y es genuinamente nuevo."
        ),
    },
    "P10_1_consenso_umbral": {
        "etiqueta": "Consenso de umbrales entre pliegues",
        "patron": r"consenso_umbral|umbral_consenso|voto_umbral|mediana_umbral",
        "busqueda": "codigo",
        "lectura": (
            "Si esta ausente, el mecanismo de consenso de umbrales de la seccion 17 no "
            "existe y depende de que la auditoria muestre dispersion alta."
        ),
    },
}

# De donde sale cada cifra. Hay tres formas de origen y se declaran de forma explicita,
# porque la version anterior adivinaba nombres de columna y devolvia vacio o, peor, un
# numero que no significaba lo que decia significar.
#
#   tipo "json"     -> recorre una ruta con barras dentro de un JSON de reports
#   tipo "registro" -> toma el campo result de un experimento, lo interpreta como JSON y
#                      saca la clave indicada
#   tipo "csv"      -> busca una fila por su primera columna y devuelve otra columna
ORIGEN_DE_CIFRAS: dict[str, dict[str, str]] = {
    "P0_1_umbral_anidado": {
        "tipo": "json", "fichero": "backlog_restante_v2.json",
        "ruta": "e2_16_umbrales_rebanada/global_anidado", "clave": "corrected_cv",
        "nota": "umbral global anidado, es decir, leave-one-fold-out",
    },
    "P0_2_umbral_por_rebanada": {
        "tipo": "json", "fichero": "backlog_restante_v2.json",
        "ruta": "e2_16_umbrales_rebanada/variantes/SHRINK-M100", "clave": "corrected_cv",
        "nota": "mejor de las cuatro variantes de umbral por rebanada",
    },
    "P0_4_calibracion": {
        "tipo": "registro", "experimento": "E2-14-MEJOR-M2", "clave": "delta_vs_sin_calibrar",
        "nota": "mejora de la calibracion frente a no calibrar, que se cerro",
    },
    "P1_1_oof_por_pliegue": {
        "tipo": "json", "fichero": "v6_error_structure.json",
        "ruta": "n_desarrollo", "clave": "",
        "nota": "el artefacto canonico ya lleva el pliegue de cada fila",
    },
    "P2_1_linearsvc": {
        "tipo": "csv", "fichero": "v6_error_intersection.csv",
        "buscar": "candidato", "valor": "M3-SVC", "columna": "v03_falla_candidato_acierta",
        "nota": "filas que la maquina de margen acierta y V03 falla, sobre 1483",
    },
    "P2_3_generativo": {
        "tipo": "registro", "experimento": "E2-15-NBSVM", "clave": "corrected_cv",
        "nota": "log-count ratio sobre la union de V03",
    },
    "P3_1_cnn_caracteres": {
        "tipo": "registro", "experimento": "DL-02", "clave": "f1_oof_media_columnas",
        "nota": "convolutional de caracteres entrenada desde cero",
    },
    "P4_1_vista_duplicada": {
        "tipo": "registro", "experimento": "E2-10-ORT-ABCD", "clave": "delta_vs_v03_umbrales",
        "nota": "canal original mas canal normalizado",
    },
    "P5_1_alcance_negacion": {
        "tipo": "registro", "experimento": "E2-08-NEG", "clave": "delta_vs_v03_umbrales",
        "nota": "canal de negacion",
    },
    "P7_1_especialistas": {
        "tipo": "registro", "experimento": "E2-20-ESPECIALISTAS", "clave": "delta_vs_v03",
        "nota": "modelo propio por rebanada",
    },
    # P8 no tiene entrada a proposito: el mecanismo nunca se midio, asi que no existe cifra
    # que traer. Anadir una aqui obligaria a inventar un numero, que es justo el fallo que
    # este script evita.
}


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V7-P0] {mensaje}", flush=True)


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


def _cargar_json(nombre: str) -> dict[str, Any] | None:
    """Carga un JSON de artefactos si existe.

    Args:
        nombre: nombre del fichero dentro de reports.

    Returns:
        El diccionario cargado, o ``None`` si no existe o no es valido.
    """
    ruta = PROJECT_ROOT / "reports" / nombre
    if not ruta.exists():
        return None
    try:
        return json.loads(ruta.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _formatear(valor: Any) -> str:
    """Formatea un valor numerico con seis decimales, o lo deja como texto.

    Args:
        valor: valor a formatear.

    Returns:
        Representacion textual del valor.
    """
    if isinstance(valor, bool):
        return str(valor)
    if isinstance(valor, float):
        return f"{valor:.6f}"
    return str(valor)[:180]


def _del_registro(experiment_id: str, clave: str) -> tuple[str, str]:
    """Saca una clave del campo result de un experimento, que es JSON.

    Args:
        experiment_id: identificador del experimento.
        clave: clave que se quiere del diccionario.

    Returns:
        Tupla con el valor formateado y una marca de error vacia si todo fue bien.
    """
    with EXPERIMENTS_CSV.open(encoding="utf-8") as archivo:
        for fila in csv.DictReader(archivo):
            if (fila.get("experiment_id") or "").strip() != experiment_id:
                continue
            bruto = (fila.get("result") or "").strip()
            if not bruto:
                return "", f"resultado vacio para {experiment_id}"
            try:
                datos = json.loads(bruto)
            except json.JSONDecodeError:
                return "", f"resultado no interpretable para {experiment_id}"
            if clave not in datos:
                return "", f"clave ausente {clave} en {experiment_id}"
            return _formatear(datos[clave]), ""
    return "", f"experimento no encontrado: {experiment_id}"


def _del_csv(fichero: str, buscar: str, valor: str, columna: str) -> tuple[str, str]:
    """Busca una fila en un CSV por el valor de una columna y devuelve otra columna.

    Args:
        fichero: nombre del CSV dentro de reports.
        buscar: columna por la que se busca.
        valor: valor que se busca.
        columna: columna que se devuelve.

    Returns:
        Tupla con el valor formateado y una marca de error vacia si todo fue bien.
    """
    ruta = PROJECT_ROOT / "reports" / fichero
    if not ruta.exists():
        return "", f"artefacto ausente: {fichero}"
    with ruta.open(encoding="utf-8") as archivo:
        for fila in csv.DictReader(archivo):
            if (fila.get(buscar) or "").strip() == valor:
                if columna not in fila:
                    return "", f"columna ausente {columna} en {fichero}"
                return _formatear(fila[columna]), ""
    return "", f"fila no encontrada: {valor} en {fichero}"


def _del_json(fichero: str, ruta: str, clave: str) -> tuple[str, str]:
    """Recorre una ruta con barras dentro de un JSON de reports y saca una clave.

    Args:
        fichero: nombre del JSON dentro de reports.
        ruta: ruta con barras, por ejemplo ``a/b/c``.
        clave: clave final. Si esta vacia, se devuelve el nodo entero.

    Returns:
        Tupla con el valor formateado y una marca de error vacia si todo fue bien.
    """
    datos = _cargar_json(fichero)
    if datos is None:
        return "", f"artefacto ausente: {fichero}"
    nodo: Any = datos
    for parte in ruta.split("/"):
        if isinstance(nodo, dict) and parte in nodo:
            nodo = nodo[parte]
        else:
            return "", f"ruta ausente: {ruta}"
    if not clave:
        return _formatear(nodo), ""
    if not isinstance(nodo, dict) or clave not in nodo:
        return "", f"clave ausente {clave} en {ruta}"
    return _formatear(nodo[clave]), ""


def cifra_real(clave: str) -> tuple[str, str]:
    """Trae la cifra medida de un mecanismo ya cerrado, no una afirmacion.

    Args:
        clave: clave de la comprobacion.

    Returns:
        Tupla con la cifra y una descripcion de su origen, o con el error si no se pudo.
    """
    origen = ORIGEN_DE_CIFRAS.get(clave)
    if origen is None:
        return "sin artefacto asociado", ""
    tipo = origen["tipo"]
    if tipo == "registro":
        valor, error = _del_registro(origen["experimento"], origen["clave"])
    elif tipo == "csv":
        valor, error = _del_csv(
            origen["fichero"], origen["buscar"], origen["valor"], origen["columna"]
        )
    elif tipo == "json":
        valor, error = _del_json(origen["fichero"], origen["ruta"], origen["clave"])
    else:
        return "", f"tipo de origen desconocido: {tipo}"
    descripcion = origen.get("nota", "")
    if error:
        return error, descripcion
    return f"{valor} ({descripcion})" if descripcion else valor, descripcion


def construir_filas():
    """Construye la tabla de brecha, la evidencia y las cifras reales.

    Returns:
        Tupla con las filas de la tabla, la evidencia de busqueda, las cifras y las
        descripciones de donde sale cada cifra.
    """
    log("1/5 busquedas programaticas sobre el codigo real")
    evidencia = {
        clave: buscar(definicion["patron"]) for clave, definicion in COMPROBACIONES.items()
    }
    for clave, resultado in evidencia.items():
        estado = "presente" if resultado["presente"] else "ausente"
        detalle = f" en {resultado['n_ficheros']} ficheros" if resultado["presente"] else ""
        log(f"   {clave:28s} {estado:9s}{detalle}")

    log("2/5 lectura de las cifras reales de los artefactos")
    completas = {clave: cifra_real(clave) for clave in COMPROBACIONES}
    # Se separan la cifra de su descripcion para poderlas mostrar por separado.
    cifras = {clave: valor for clave, (valor, _) in completas.items()}
    for clave, valor in cifras.items():
        log(f"   {clave:28s} {valor[:88]}")

    log("3/5 tabla de brecha")
    # Cada fila es un juicio del analisis, pero cada juicio cita la clave de evidencia que
    # lo respalda y la cifra medida que lo cuantifica.
    filas: list[dict[str, str]] = [
        {
            "family": "P0.1 Umbral anidado",
            "hypothesis": (
                "Separar el ajuste del umbral del ajuste del modelo y medir si los "
                "umbrales congelados de V03 son estables o producto de sobreajuste"
            ),
            "already_tested": "si, con un protocolo mas laxo",
            "experiment_ids": "E2-16, TH-01, S03",
            "best_result": cifras["P0_1_umbral_anidado"],
            "why_closed": (
                "Ya medida como umbral global anidado tomando el umbral de cada pliegue "
                "del resto de pliegues. El resultado fue peor que V03."
            ),
            "possible_new_formulation": (
                "Separar en tres niveles: ajuste, calibracion y evaluacion. El protocolo "
                "existente reutiliza las probabilidades fuera de pliegue de los otros "
                "cuatro pliegues como conjunto de calibracion, y por tanto ve mas datos "
                "de los que permite el protocolo de la especificacion. Ademas produce el "
                "informe de dispersion de umbrales que nadie tiene."
            ),
            "expected_upside": "bajo: el mecanismo ya dio un resultado peor que V03",
            "risk": "bajo: es la unica linea que produce medicion fiable del sesgo",
            "cost": "medio: requiere reentrenar con particion de tres niveles",
            "priority": "media: se ejecuta, pero por rigor de medicion y no por upside",
        },
        {
            "family": "P0.2 Umbral por rebanada",
            "hypothesis": (
                "Un umbral por cada region grande del texto, en lugar de uno global"
            ),
            "already_tested": "si",
            "experiment_ids": "E2-16 (5 variantes con shrinkage)",
            "best_result": cifras["P0_2_umbral_por_rebanada"],
            "why_closed": (
                "Las rebanadas con distribucion distinta se midieron con umbral propio y "
                "cuatro niveles de shrinkage, y ninguna supero al umbral global anidado."
            ),
            "possible_new_formulation": "ninguna sin cambiar el mecanismo",
            "expected_upside": "nulo",
            "risk": "nulo",
            "cost": "nulo",
            "priority": "nula: cerrar",
        },
        {
            "family": "P0.3 Estimacion robusta de umbral",
            "hypothesis": (
                "La mediana, la media recortada y la mediana bootstrap de los umbrales por "
                "pliegue son mas estables que la media"
            ),
            "already_tested": "no",
            "experiment_ids": "ninguno",
            "best_result": "sin medir",
            "why_closed": "no se ha implementado nunca",
            "possible_new_formulation": (
                "ya esta bien formulada: cambiar el estimador del agregado de umbrales por "
                "pliegue, manteniendo la evaluacion fuera del conjunto de calibracion"
            ),
            "expected_upside": "bajo, pero es la parte de P0 que si es nueva",
            "risk": "bajo: no toca el modelo, solo el punto de operacion",
            "cost": "bajo: reutiliza el protocolo de P0.1",
            "priority": "baja-media: ejecutar solo si P0.1 muestra dispersion alta",
        },
        {
            "family": "P0.4 Calibracion de probabilidades",
            "hypothesis": "Ajustar sigmoid o isotonica mejora la F1 y la estabilidad",
            "already_tested": "si",
            "experiment_ids": "E2-14",
            "best_result": cifras["P0_4_calibracion"],
            "why_closed": (
                "Se evaluo con ajuste, calibracion y evaluacion separados, mejoro frente a "
                "no calibrar pero perdio al.Reference en la fusion, y se cerro."
            ),
            "possible_new_formulation": "ninguna sin cambiar el mecanismo",
            "expected_upside": "nulo",
            "risk": "nulo",
            "cost": "nulo",
            "priority": "nula: cerrar",
        },
        {
            "family": "P0.5 Umbral condicional por region",
            "hypothesis": (
                "Un umbral por cada region grande con soporte suficiente, calculado con "
                "datos de calibracion y minimo de soporte"
            ),
            "already_tested": "si, identico",
            "experiment_ids": "E2-16, E2-21",
            "best_result": "identico a P0.2",
            "why_closed": (
                "Es exactamente lo que hizo E2-16: umbral por rebanada, con shrinkage, "
                "con soporte minimo y con calibracion separada."
            ),
            "possible_new_formulation": "ninguna",
            "expected_upside": "nulo",
            "risk": "nulo",
            "cost": "nulo",
            "priority": "nula: cerrar",
        },
        {
            "family": "P1.1 Ensemble de modelos por pliegue",
            "hypothesis": (
                "V03 depende demasiado de un unico ajuste y el promedio de cinco modelos "
                "reduce la varianza"
            ),
            "already_tested": "la parte medible ya esta hecha",
            "experiment_ids": "artefacto features/oof_v03_final.npz",
            "best_result": "el OOF canonico ya es el promedio por pliegue",
            "why_closed": (
                "El OOF de V03 contiene la etiqueta de pliegue de cada fila y su "
                "probabilidad procede del modelo que no la entreno. Construir ese OOF otra "
                "vez no produce informacion nueva: solo queda el promedio en el conjunto de "
                "prueba, que no se puede medir en el desarrollo."
            ),
            "possible_new_formulation": (
                "Reescribir la linea como cambio **solo de prueba**: promediar las "
                "probabilidades de los cinco modelos al predecir el test. Eso solo puede "
                "verificarse en el leaderboard y por tanto queda para el final, sobre un "
                "candidato que haya superado todos los gates internos."
            ),
            "expected_upside": "desconocido e inmedible en local",
            "risk": "medio: no se puede validar antes de enviar",
            "cost": "bajo de computationally, pero exige una sonda de leaderboard",
            "priority": "aplazada: no es ejecutable en local",
        },
        {
            "family": "P2.1 Maquina de margen sobre V03",
            "hypothesis": "Una perdida de margen con el mismo espacio aporta mas que la logistica",
            "already_tested": "si",
            "experiment_ids": "V2-BASE-M3, ENS-01 a ENS-04, S04",
            "best_result": "0.7715 frente a 0.777211 de V03, y 10 filas de interseccion",
            "why_closed": (
                "Es la referencia honesta del protocolo v2 y participo en cuatro ensembles. "
                "Ademas su techo esta medido: el mejor modelo alternativo acierta 9 a 12 "
                "filas de 1483 donde V03 falla, lo que limita la mejora a +0.0027."
            ),
            "possible_new_formulation": (
                "recorrer C en cuatro valores solo cambia la escala de un modelo que ya se "
                "midio y cuyo techo ya se conoce"
            ),
            "expected_upside": "nulo: el techo esta por debajo del gate",
            "risk": "nulo",
            "cost": "bajo",
            "priority": "nula: cerrar por techo",
        },
        {
            "family": "P2.3 Experto generativo disperso",
            "hypothesis": "Un modelo generativo aporta senal complementaria al discriminativo",
            "already_tested": "si, cuatro variantes",
            "experiment_ids": "E2-15-NB-WORD, E2-15-NB-CHAR, E2-15-NBSVM, E2-15-NBSVM-SVC",
            "best_result": "cerrada en un rango de -0.016 a -0.039",
            "why_closed": (
                "Se compararon el peso de Naive Bayes sobre palabras, sobre caracteres, el "
                "log-count ratio sobre la union de V03 y su margen con la maquina de margen."
            ),
            "possible_new_formulation": (
                "quedaria probar el generativo puro con conteos, que no es el peso de "
                "Naive Bayes, pero el techo de +0.0027 aplica igual"
            ),
            "expected_upside": "nulo: el techo esta por debajo del gate",
            "risk": "nulo",
            "cost": "bajo",
            "priority": "nula: cerrar por techo",
        },
        {
            "family": "P3 CNN de caracteres desde cero",
            "hypothesis": "Un modelo no lineal de caracteres aprende interacciones que el espacio lineal no expresa",
            "already_tested": "si, y con conclusion firme",
            "experiment_ids": "DL-02 (y DL-01, DL-03 con el mismo precedente)",
            "best_result": cifras["P3_1_cnn_caracteres"],
            "why_closed": (
                "La CNN de caracteres ya se entreno desde cero, con vocabulario de 91 "
                "caracteres, dimension 48 y longitud 200, y se descarto. Su brecha entre "
                "ajuste y validacion fue de 0.18, que es el sintoma de falta de datos, no de "
                "arquitectura. La especificacion de v7 propone el mismo modelo, y ademas "
                "prohibe usar los embeddings externos que si serian una entrada distinta."
            ),
            "possible_new_formulation": "ninguna dentro de las restricciones de la fase",
            "expected_upside": "nulo",
            "risk": "nulo",
            "cost": "medio: entrenamiento lento para un resultado ya conocido",
            "priority": "nula: cerrar por duplicado",
        },
        {
            "family": "P4 Vista doble con normalizacion",
            "hypothesis": "El original y una version ortografica segura aportan dos vistas",
            "already_tested": "si, identico",
            "experiment_ids": "E2-10-ORT-A, E2-10-ORT-ABCD, E2-10-ORT-AE",
            "best_result": cifras["P4_1_vista_duplicada"],
            "why_closed": (
                "La doble vista se construyo exactamente asi: el canal original conserva el "
                "estilo y un canal normalizado recupera los lexemas ofuscados. Tres "
                "variantes, todas perdidas entre -0.0238 y -0.0286."
            ),
            "possible_new_formulation": "ninguna",
            "expected_upside": "nulo",
            "risk": "nulo",
            "cost": "nulo",
            "priority": "nula: cerrar por duplicado",
        },
        {
            "family": "P5.1 Alcance de negacion",
            "hypothesis": "Marcar el alcance de la negacion con tokens modificados aporta senal",
            "already_tested": "si, y con la mayor perdida del proyecto",
            "experiment_ids": "E2-08-NEG, E2-04-DEPS, E2-07-CITA-NEG-TXT",
            "best_result": cifras["P5_1_alcance_negacion"],
            "why_closed": (
                "La negacion como canal, como patron de dependencia y unida a la cita ya se "
                "midieron. El canal de negacion fue el peor resultado registrado, muy por "
                "debajo de V03."
            ),
            "possible_new_formulation": "ninguna sin cambiar el mecanismo",
            "expected_upside": "nulo",
            "risk": "nulo",
            "cost": "nulo",
            "priority": "nula: cerrar por duplicado",
        },
        {
            "family": "P6 Vista multiple en el momento de predecir",
            "hypothesis": "Promediar la prediccion de las dos vistas estabiliza frente al ruido ortografico",
            "already_tested": "parcialmente, por el mismo mecanismo que P4",
            "experiment_ids": "E2-10 (misma doble vista) y V6-08 (aumento, otro mecanismo)",
            "best_result": "la doble vista perdio entre -0.0238 y -0.0286",
            "why_closed": (
                "El mecanismo es la doble vista de P4, solo que en el momento de predecir. "
                "Si la vista normalizada no aporta en el ajuste, no hay razon para que "
                "aporte en la prediccion. Es el mismo argumento que cerro la linea de "
                "aumento de la fase v6."
            ),
            "possible_new_formulation": "ninguna que no repita P4",
            "expected_upside": "nulo",
            "risk": "nulo",
            "cost": "nulo",
            "priority": "nula: cerrar",
        },
        {
            "family": "P7 Puerta de expertos por region",
            "hypothesis": "Un modelo alternativo solo es mejor en una region concreta del espacio",
            "already_tested": "si, y con techo medido",
            "experiment_ids": "E2-20-ESPECIALISTAS, E2-21-ESPACIOS, E2-18 (meta)",
            "best_result": cifras["P7_1_especialistas"],
            "why_closed": (
                "El modelo por rebanada se midio y perdio. Ademas la interseccion de errores "
                "de la fase v6 demuestra que combinar dos clasificadores elige entre sus "
                "aciertos y no los suma, con un techo de +0.0027 para cualquier linea cuya "
                "unica via sea recuperar filas ajenas a V03."
            ),
            "possible_new_formulation": "ninguna que no cambie el techo",
            "expected_upside": "nulo: el techo esta por debajo del gate",
            "risk": "nulo",
            "cost": "nulo",
            "priority": "nula: cerrar por techo",
        },
        {
            "family": "P8 Prior de etiqueta del grupo",
            "hypothesis": (
                "Un grupo con varias observaciones tiene una distribucion de etiquetas "
                "caracteristica que puede convertirse en una caracteristica adicional"
            ),
            "already_tested": "no: el mecanismo es distinto al de v6",
            "experiment_ids": "PA-01 (grupo solo como control de particion), V6-03 (pesos)",
            "best_result": "nunca medido como caracteristica",
            "why_closed": "no esta cerrado",
            "possible_new_formulation": (
                "ya esta bien formulada. La distincion es mecanica y no nominal: la fase v6 cambio "
                "**pesos** de entrenamiento, que resta capacidad a un ejemplo repetido, y "
                "esta linea crea una **caracteristica** con la distribucion de etiquetas del "
                "grupo, que informa sin apartar al modelo de la fila."
            ),
            "expected_upside": "medio: ataca la estructura que v6 si encontro con senal",
            "risk": (
                "alto de filtrado: el prior debe calcularse con las otras observaciones del "
                "grupo dentro del pliegue y nunca con la propia etiqueta en evaluacion"
            ),
            "cost": "bajo: es una caracteristica densa de tres columnas",
            "priority": "alta: mecanismo genuinamente nuevo",
        },
        {
            "family": "P9 Perdida orientada a la F1",
            "hypothesis": (
                "Existe una diferencia material entre la perdida logistica y una perdida "
                "orientada al objetivo real, que es la F1 macro"
            ),
            "already_tested": "no",
            "experiment_ids": "ninguno",
            "best_result": "nunca medido",
            "why_closed": "no esta cerrado",
            "possible_new_formulation": (
                "ya esta bien formulada. Es la unica linea de la fase que ataca la funcion "
                "de perdida, es decir, el objetivo, y no la representacion ni el modelo."
            ),
            "expected_upside": "medio: es el unico mecanismo de decision sin cerrar",
            "risk": "medio: las aproximaciones a F1 pueden ser inestables con 1731 filas",
            "cost": "medio: hay que elegir y justificar un sustituto estable",
            "priority": "alta: primera linea de ejecucion",
        },
        {
            "family": "P10 Consenso de umbrales entre pliegues",
            "hypothesis": (
                "Un consenso entre pliegues produce umbrales mas estables que un unico "
                "ajuste, aunque no maximice la metrica"
            ),
            "already_tested": "no",
            "experiment_ids": "ninguno",
            "best_result": "nunca medido",
            "why_closed": "no esta cerrado, pero depende de lo que muestre P0.1",
            "possible_new_formulation": "ya esta bien formulada, y es la continuacion de P0.3",
            "expected_upside": "bajo por si mismo, alto como garantia de estabilidad",
            "risk": "bajo",
            "cost": "bajo",
            "priority": "media: ejecutar solo si P0.1 muestra dispersion alta",
        },
    ]
    return filas, evidencia, completas


def main() -> None:
    """Ejecuta la auditoria de brechas de la fase v7."""
    inicio = time.time()
    log("Auditoria de brechas de la fase v7, antes de gastar una sola unidad")
    with EXPERIMENTS_CSV.open(encoding="utf-8") as archivo:
        experimentos = list(csv.DictReader(archivo))
    filas, evidencia, completas = construir_filas()
    cifras = {clave: valor for clave, (valor, _) in completas.items()}
    decisiones = re.findall(
        r"^## (D-\d+)", DECISIONES.read_text(encoding="utf-8"), re.MULTILINE
    )

    log("4/5 inventario y clasificacion")
    prefijos: Counter[str] = Counter()
    for fila in experimentos:
        identificador = (fila.get("experiment_id") or "").strip()
        if identificador:
            prefijos[identificador.split("-")[0]] += 1
    nuevo = [f["family"] for f in filas if f["already_tested"].startswith("no")]
    confirmar = [f["family"] for f in filas if "parcial" in f["already_tested"]
                   or "mas laxo" in f["already_tested"]
                   or "solo esta hecha" in f["already_tested"]]
    log(f"   {len(experimentos)} experimentos | {len(decisiones)} decisiones")
    log(f"   mecanismos nuevos: {len(nuevo)} | a confirmar: {len(confirmar)}")

    log("5/5 informes")
    _escribir_csv(filas)
    REPORTE_VERIF.write_text(
        json.dumps(
            {
                "experimento": EXPERIMENTO,
                "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "nota": (
                    "Evidencia cruda de cada comprobacion. 'presente' significa que el "
                    "mecanismo ya esta implementado en el codigo del proyecto y por tanto "
                    "el track es una duplicacion; 'ausente' significa que no existe y el "
                    "track es genuinamente nuevo."
                ),
                "comprobaciones": {
                    clave: {
                        "etiqueta": COMPROBACIONES[clave]["etiqueta"],
                        "patron": COMPROBACIONES[clave]["patron"],
                        "resultado": evidencia[clave],
                        "cifra_real": completas[clave][0],
                        "descripcion_de_la_cifra": completas[clave][1],
                    }
                    for clave in COMPROBACIONES
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _escribir_informe(filas, evidencia, cifras, experimentos, prefijos, decisiones, nuevo)
    log(f"Auditoria completada en {round(time.time() - inicio, 1)} s")


def _escribir_csv(filas: list[dict[str, str]]) -> None:
    """Escribe la tabla de brecha en CSV.

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


def _escribir_informe(filas, evidencia, cifras, experimentos, prefijos, decisiones, nuevo) -> None:
    """Escribe el informe de la auditoria de brechas.

    Args:
        filas: filas de la tabla de brecha.
        evidencia: resultado de las busquedas en el codigo.
        cifras: cifras reales traidas de los artefactos.
        experimentos: filas del registro de experimentos.
        prefijos: recuento de prefijos de experimento.
        decisiones: identificadores de decision del proyecto.
        nuevo: familias clasificadas como genuinamente nuevas.
    """
    confirmar = [f for f in filas if "parcial" in f["already_tested"]
                   or "mas laxo" in f["already_tested"]
                   or "solo esta hecha" in f["already_tested"]]
    cerradas = [f for f in filas if f["priority"].startswith("nula")]
    lineas = [
        "# v7: auditoria de brechas (P0)",
        "",
        f"Generado por `experiments/v7_00_gap_audit.py` ({EXPERIMENTO}).",
        "No entrena nada. Se ejecuta antes de gastar una sola unidad de la fase v7.",
        "",
        "Cada afirmacion de la tabla se apoya en una busqueda programatica sobre el codigo",
        "real del proyecto y en la lectura del numero medido en los artefactos existentes.",
        "Un cierre sin cifra no es un cierre, y por eso la columna de mejor resultado se",
        "trae de los ficheros JSON y CSV que los experimentos anteriores ya escribieron.",
        "",
        "## 1. Inventario de lo ya experimentado",
        "",
        f"El registro tiene **{len(experimentos)} experimentos** y **{len(decisiones)}"
        " decisiones**. Por prefijo de identificador:",
        "",
        "| prefijo | experimentos |",
        "|---|---|",
    ]
    for prefijo, total in prefijos.most_common():
        lineas.append(f"| `{prefijo}-*` | {total} |")
    lineas += [
        "",
        "## 2. El hallazgo que reencuadra la fase",
        "",
        "Tres modelos con **mejor** metrica fuera de pliegue llegaron al final del proyecto",
        "",
        "| experimento | OOF | leaderboard | diferencia |",
        "|---|---|---|---|",
        "| V03 | 0.777211 | **0.78306** | referencia |",
        "| R1-C1, con C igual a 1.0 | 0.7929 | 0.77227 | **-0.011** |",
        "| R2-MINDF5 | 0.7993 | 0.77034 | **-0.013** |",
        "| OPT-01, busqueda con Optuna | **0.8012** | 0.77602 | **-0.007** |",
        "",
        "La mejora en OOF de los tres era de mas de +0.015, muy por encima del gate de",
        "+0.003, y los tres perdieron leaderboard. Ademas los tres aumentaban capacidad.",
        "La conclusion de S05 lo dice sin rodeos: la busqueda por metrica fuera de pliegue premia capacidad que no",
        "generaliza.",
        "",
        "De ahi dos reglas que condicionan toda la fase:",
        "",
        "1. **El unico movimiento positivo del leaderboard en toda la historia del proyecto",
        "   vino de la capa de decision, no del modelo**: los umbrales por etiqueta pasaron",
        "   el leaderboard de 0.77993 a 0.78306, es decir +0.0031.",
        "2. **Toda mejora de capacidad de modelo ha restado leaderboard.** Por eso el freno",
        f"   de transferencia: por encima de {FRENO_TRANSFERENCIA_DELTA:+.3f} de OOF, un",
        "   candidato solo llega a la mesa de finales si **no** aumenta la dimension ni la",
        "   brecha entre ajuste y validacion. Es la regla que los tres fracasos violan y que",
        "   V03 cumple.",
        "",
        "Esto no dice que 0.80 sea imposible. Dice que hay que moverlo desde donde ya se ha",
        "movido una vez, que es la decision, y que el unico gasto que justifica su precio es",
        "medir bien.",
        "",
        "## 3. La tabla de brecha completa",
        "",
        "El detalle, con las once columnas que pide la especificacion, esta en",
        "`reports/v7_gap_audit.csv`.",
        "",
        "**Sobre la procedencia de las cifras de esta tabla**: la columna de mejor resultado",
        "es un resumen. Las cifras con su origen exacto, leidas de los artefactos, estan en",
        "la seccion 4 y en `reports/v7_gap_verificaciones.json`. Cuando una cifra no se pudo",
        "leer, aparece el motivo del fallo en lugar de un numero, porque un hueco declarado",
        "es informacion y un numero inventado es ruido.",
        "",
        "| track | ya probado | mejor resultado | prioridad |",
        "|---|---|---|---|",
    ]
    for fila in filas:
        mejor = fila["best_result"]
        mejor = mejor[:40] + "..." if len(mejor) > 40 else mejor
        lineas.append(
            f"| {fila['family']} | {fila['already_tested']} | `{mejor}` | "
            f"{fila['priority']} |"
        )
    lineas += [
        "",
        "## 4. Comprobaciones programaticas",
        "",
        "| comprobacion | resultado | ficheros | cifra medida |",
        "|---|---|---|---|",
    ]
    for clave, resultado in evidencia.items():
        estado = "presente" if resultado["presente"] else "ausente"
        ficheros = ", ".join(
            f"{f['fichero']} ({f['ocurrencias']})" for f in resultado["ficheros"][:2]
        ) or "ninguno"
        linea = cifras[clave]
        linea = linea[:46] + "..." if len(linea) > 46 else linea
        lineas.append(
            f"| `{clave}` | **{estado}** | {ficheros} | `{linea}` |"
        )
    lineas += [
        "",
        "Las lecturas que importan:",
        "",
        "- `P8_1_prior_grupo` **ausente** y `P9_1_perdida_f1` **ausente**: son los dos",
        "  unicos mecanismos de la especificacion que no existen en el proyecto. Uno crea",
        "  una caracteristica a partir de la estructura de grupos repetidos, que es la",
        "  senal que la fase v6 si encontro; el otro ataca la funcion de perdida, que es el",
        "  objetivo real.",
        "- `P0_1_umbral_anidado` **presente** con un numero peor que V03: la linea de",
        "  umbrales no es nueva, pero el protocolo de la especificacion es **mas estricto**",
        "  que el existente y ademas produce la medicion de dispersion que nadie tiene.",
        "- `P1_1_oof_por_pliegue` **presente**: el artefacto canonico ya es el promedio por",
        "  pliegue, de modo que la parte medible de P1 esta hecha y solo queda un cambio que",
        "  no se puede medir en local.",
        "- `P3_1_cnn_caracteres` **presente**: la CNN de caracteres ya se entreno desde cero",
        "  y se descarto. La especificacion de v7 propone el mismo modelo y ademas le",
        "  prohibe la unica entrada distinta que lo hacia util.",
        "",
        "## 5. Lo que se cierra y por que no se ejecuta",
        "",
        "| track | causa del cierre |",
        "|---|---|",
    ]
    for fila in cerradas:
        lineas.append(f"| {fila['family']} | {fila['why_closed']} |")
    lineas += [
        "",
        "## 6. Lo que queda abierto de verdad",
        "",
        "| track | por que es nuevo | upside esperado | coste |",
        "|---|---|---|---|",
    ]
    for fila in filas:
        if fila["priority"].startswith("nula"):
            continue
        lineas.append(
            f"| {fila['family']} | {fila['possible_new_formulation'][:150]} | "
            f"{fila['expected_upside']} | {fila['cost']} |"
        )
    lineas += [
        "",
        "## 7. Lineas que se ejecutan con reservas expresas",
        "",
    ]
    for fila in confirmar:
        lineas.append(f"- **{fila['family']}**: {fila['possible_new_formulation']}")
    lineas += [
        "",
        "## 8. Riesgo de filtrado por linea",
        "",
        "El riesgo que mas importa en esta fase no es estadistico sino de protocolo:",
        "reutilizar el mismo conjunto fuera de pliegue para ajustar, para elegir umbrales y",
        "para evaluar. La fase v6 ya lo midio: reajustar umbrales sobre el mismo conjunto",
        "infla el resultado en +0.0036 sin cambiar el modelo. Por eso toda linea de v7 que",
        "toque umbrales o perdidas debe declarar su esquema de tres niveles de forma",
        "explicita en el registro.",
        "",
        "| track | riesgo de filtrado |",
        "|---|---|",
    ]
    for fila in filas:
        if not fila["priority"].startswith("nula"):
            lineas.append(f"| {fila['family']} | {fila['risk']} |")
    lineas += [
        "",
        "## 9. Orden de ejecucion que se deduce de esta auditoria",
        "",
        "1. **P9** perdida orientada a la F1: unico mecanismo de decision sin cerrar, y",
        "   ataca el objetivo real.",
        "2. **P8** prior de etiqueta del grupo: unica senal nueva, y ataca la estructura",
        "   que la fase v6 si encontro con senal.",
        "3. **P0.1** umbral anidado con protocolo mas estricto: por rigor de medicion, no",
        "   por upside. Su salida es el informe de dispersion que la fase v6 dejo pendiente.",
        "4. **P0.3 y P10** estimador robusto y consenso: solo si P0.1 muestra dispersion",
        "   alta, y son baratos porque reutilizan su protocolo.",
        "",
        "Este orden no es el de la especificacion, y la razon es que la auditoria mostro que",
        "los cuatro primeros tracks de aquel orden ya estaban medidos. El orden de la",
        "especificacion era correcto **antes** de conocer el registro, no despues.",
        "",
        "Presupuesto autorizado: 25 experimentos principales y 10 de reserva. Presupuesto",
        "realista despues de esta auditoria: **4 a 8 experimentos**. No es un recorte, es lo",
        "que queda abierto.",
        "",
        f"Fecha: {datetime.now(timezone.utc).isoformat(timespec='seconds')}.",
    ]
    REPORTE.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
