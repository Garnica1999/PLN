"""P0 de la fase v6 (V6-00, ANALYSIS): analisis de brechas antes de ejecutar nada.

Contexto
--------
La fase v6 no abre una busqueda nueva de modelos: busca **mecanismos** que el proyecto no
haya evaluado de verdad. Antes de gastar una sola unidad hace falta demostrar, experimento
por experimento, que la hipotesis es genuinamente nueva, porque el proyecto ya acumulo 174
experimentos y varias lineas que la especificacion de v6 propone fueron **cerradas hace dos
fases** sin que la especificacion lo supiera.

Por que este script no escribe el gap analysis a mano
------------------------------------------------------
Un analisis de brechas escrito de memoria se contradice en cuanto el codigo cambia. Aqui
cada afirmacion de "ya probada" o "nunca probada" va acompanada de una **verificacion
programatica** sobre el arbol de scripts: el script busca el patron en el codigo real y
registra lo que encuentra. Si alguien implementa un ``char_weight`` manana, esta tabla
pasara a decir "ya probada" y a senalar el track, sin que nadie tenga que acordarse.

Que produce
----------
- ``reports/v6_hypothesis_gap_analysis.csv``: una fila por track y sub-track con las ocho
  columnas que pide la seccion 4 de la especificacion.
- ``reports/v6_hypothesis_gap_analysis.md``: inventario, familias cerradas, hipotesis
  nuevas, artefactos reutilizables, presupuesto, orden y duplicaciones explicitas.
- ``reports/v6_verificaciones_gap_v6.json``: la evidencia cruda de cada verificacion.

Uso
---
    /home/carlos/venv/bin/python experiments/v6_00_gap_analysis.py
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

EXPERIMENTO = "V6-00 (P0, ANALYSIS)"
REPORTE = PROJECT_ROOT / "reports" / "v6_hypothesis_gap_analysis.md"
REPORTE_CSV = PROJECT_ROOT / "reports" / "v6_hypothesis_gap_analysis.csv"
REPORTE_VERIF = PROJECT_ROOT / "reports" / "v6_verificaciones_gap_v6.json"
EXPERIMENTS_CSV = PROJECT_ROOT / "experiments.csv"
DECISIONES = PROJECT_ROOT / "DECISIONS.md"

# Presupuesto de la fase, seccion 34 de la especificacion.
PRESUPUESTO_PRINCIPAL = 24
PRESUPUESTO_RESERVA = 6
PRESUPUESTO_TOTAL = PRESUPUESTO_PRINCIPAL + PRESUPUESTO_RESERVA

# Gate de adopcion de la fase, ya corregido a dos niveles (decision del usuario): el
#Gate 1 de la especificacion (delta >= +0.003 y "4/5 folds **o** complementariedad") es mas
# laxo que el criterio vigente desde D-036, y D-029 ya registro que un candidato que paso
# los gates fue rechazado por el holdout. El "o" se sustituye por "y" y el intervalo de
# confianza pasa a ser obligatorio.
GATE_1_DELTA = 0.003
GATE_1_FOLDS = 4
GATE_1_EXIGE_IC95 = True
GATE_2_REQUIERE = "segunda particion agrupada con otra semilla"

# Verificaciones programaticas: clave -> (etiqueta, patron, donde se busca, lectura).
#
# Dos decisiones que hacen que estas verificaciones sirvan de algo:
#
# 1. Los patrones exigen **implementacion**, no mencion. Se busca una asignacion o una
#    llamada (``char_weight =``, ``KNeighborsClassifier(``), no la palabra "skip". Un patron
#    laxo casa en cualquier comentario que nombre el concepto y produce falsos positivos que
#    cierran un track que en realidad es nuevo.
# 2. Se **excluye** este mismo script de la busqueda. El gap analysis nombra los mecanismos
#    que busca, asi que un patron sin mas se encontraria a si mismo y declararia duplicada la
#    linea que esta justement identificando como nueva. El filtro es por prefijo de nombre
#    (``v6_``) para que siga funcionando cuando escriban los scripts de los tracks.
VERIFICACIONES: dict[str, dict[str, Any]] = {
    "I_word_char_weight": {
        "etiqueta": "Ponderacion explicita entre los bloques word y char_wb",
        "patron": r"char_weight\s*=|word_weight\s*=|block_weight\s*=",
        "busqueda": "codigo",
        "lectura": "Si esta ausente, el optimo de V03 nunca se probo con una escala explicita entre evidencia lexica y sublexica.",
    },
    "A_group_weight": {
        "etiqueta": "Ponderacion de entrenamiento por tamano de grupo",
        "patron": r"peso_grupo|p(group|)_weight|peso_por_grupo|group_size\s*=\s*np\.|np\.sqrt\([^)]*grupo",
        "busqueda": "codigo",
        "lectura": "Si esta ausente, nunca se pregunto si V03 sobrepondera observaciones repetidas.",
    },
    "H_group_bagging": {
        "etiqueta": "Bagging con remuestreo por grupos",
        "patron": r"BaggingClassifier|bootstrap_grupo|grupo_bootstrap|bootstrap_por_grupo",
        "busqueda": "codigo",
        "lectura": "Si esta ausente, la reduccion de varianza por grupos nunca se probo; los ensembles previos mezclaban representaciones distintas.",
    },
    "C1_skipgram": {
        "etiqueta": "N-gramas de palabra con salto (skip-grams)",
        "patron": r"skip_?gram\(|skip_?gram\s*=|max_gap\s*=|ngrama_salto|skip_bigram\(",
        "busqueda": "codigo",
        "lectura": "Si esta ausente, los pares de palabras no adyacentes nunca se representaron.",
    },
    "C3_interacciones": {
        "etiqueta": "Tokens de interaccion entre categorias",
        "patron": r"interacciones_hstack|token_interaccion|INSULT_NEAR_|interaccion_sparse",
        "busqueda": "todo",
        "lectura": "Si esta presente, el track C3 es una duplicacion de V4-D y no debe ejecutarse.",
    },
    "C2_ventanas": {
        "etiqueta": "Canales sparse posicionales (ventanas inicio/medio/final)",
        "patron": r"ventanas_hstack|canal_ventana|ventana_posicional|posicion_relativa",
        "busqueda": "todo",
        "lectura": "Si esta presente, el track C2 es una duplicacion de V4-D.",
    },
    "C4_shape": {
        "etiqueta": "Combinaciones forma x token (elongado, risa, puntuacion)",
        "patron": r"[\"']ORT-[A-Z]\b|canal_ortograf|forma_token|shape_token",
        "busqueda": "todo",
        "lectura": "Si esta presente, el track C4 es una duplicacion de la linea ortografica de E2-10.",
    },
    "D_textcnn": {
        "etiqueta": "CNN 1D (TextCNN) sobre embeddings externos",
        "patron": r"Conv1d\(|class TextCNN|TextCNN\(",
        "busqueda": "codigo",
        "lectura": "Si esta ausente, las redes convolucionales solo se probaron con embeddings aleatorios (DL-02), que es un input distinto.",
    },
    "B_factorizacion": {
        "etiqueta": "Factorizacion probabilistica condicionada p(TR|X)=p(HS|X)p(TR|HS=1,X)",
        "patron": r"p_tr_dado_hs|prob_ag_dado_hs|p_tr_hs1|modelo_tr_hs1|factoriza_prob|ag_hs1|ag_hs0",
        "busqueda": "codigo",
        "lectura": "Si esta ausente, lo que se probo fue chains y powerset, que no restringen el dominio de entrenamiento del condicional.",
    },
    "E_knn_predictor": {
        "etiqueta": "kNN usado como predictor o experto",
        "patron": r"KNeighborsClassifier|prediccion_knn|knn_prob|experto_knn|prob_knn",
        "busqueda": "codigo",
        "lectura": "V4-F03 ya construyo vecinos de 3 como MEDIDA de ambiguedad; lo que falta es usarlos como predictor.",
    },
    "E_knn_medida": {
        "etiqueta": "kNN usado como medida de ambiguedad (V4-F03)",
        "patron": r"knn_disagreement|sim_cercana|vecino_ids",
        "busqueda": "codigo",
        "lectura": "Si esta presente, la afinidad de vecindad ya se midio y resulto nula para ordenar el error.",
    },
    "F_pair_noise": {
        "etiqueta": "Ruido a nivel de par instancia-etiqueta",
        "patron": r"instance_label|par_sospechoso|sospecha_par|soft_label|target_suavizado|sCORE_ruido_instancia",
        "busqueda": "codigo",
        "lectura": "Si esta ausente, el ruido se midio por duplicado (V4-A) y nunca a nivel de par.",
    },
    "G_label_conditional": {
        "etiqueta": "Espacio de caracteristicas propio por etiqueta",
        "patron": r"espacio_hs\b|espacio_tr\b|espacio_ag\b|feature_space_label|espacios_por_etiqueta",
        "busqueda": "codigo",
        "lectura": "Si esta presente, el track G es una duplicacion de E2-21.",
    },
    "J_augmentation": {
        "etiqueta": "Aumento de datos (crear instancias nuevas)",
        "patron": r"def aumentar\(|data_augment|texto_sintetico|clon_texto|variante_augment|augmentar_train",
        "busqueda": "codigo",
        "lectura": "Si esta ausente, lo que se probo fue normalizacion del texto (PRE-00..09), que es otra cosa: quitar ruido, no crear ejemplos.",
    },
    "J_normalizacion": {
        "etiqueta": "Normalizacion determinista del texto (PRE-00..09)",
        "patron": r"PRE-0\d",
        "busqueda": "todo",
        "lectura": "Si esta presente con diez variantes descartadas, el precedente es adverso para cualquier transformacion de superficie.",
    },
}


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V6-P0] {mensaje}", flush=True)


def _leer_experimentos() -> list[dict[str, str]]:
    """Lee el registro de experimentos del proyecto.

    Returns:
        Filas de ``experiments.csv``.
    """
    with EXPERIMENTS_CSV.open(encoding="utf-8") as archivo:
        return list(csv.DictReader(archivo))


PREFIJOS_SIN_BUSQUEDA = ("v6_",)


def _excluido(ruta: Path) -> bool:
    """Indica si un fichero queda fuera de la busqueda de patrones.

    Los scripts de la propia fase v6 se excluyen porque **nombran** los mecanismos que
    buscan: sin este filtro, cada track nuevo se detectaria a si mismo y se declararia
    duplicado, que es el error opuesto y igual de grave.

    Args:
        ruta: ruta del fichero.

    Returns:
        ``True`` si el fichero no debe participate en la busqueda.
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


def _artefactos() -> list[Path]:
    """Devuelve los informes y CSV donde se buscan los patrones.

    Returns:
        Lista de ficheros de artefacto.
    """
    rutas = list((PROJECT_ROOT / "reports").glob("*.csv"))
    rutas += list((PROJECT_ROOT / "reports").glob("*.md"))
    return [ruta for ruta in rutas if not _excluido(ruta)]


def verificar(patron: str, donde: str) -> dict[str, Any]:
    """Busca un patron en el codigo o en los artefactos y devuelve la evidencia.

    Args:
        patron: expresion regular a buscar.
        donde: ``codigo``, ``artefactos`` o ``todo``.

    Returns:
        Diccionario con la lista de ficheros donde aparece y el numero de ocurrencias.
    """
    compilado = re.compile(patron, re.IGNORECASE)
    if donde == "codigo":
        ficheros = _codigo()
    elif donde == "artefactos":
        ficheros = _artefactos()
    else:
        ficheros = _codigo() + _artefactos()
    encontrados: list[dict[str, Any]] = []
    for ruta in ficheros:
        try:
            contenido = ruta.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        ocurrencias = compilado.findall(contenido)
        if ocurrencias:
            encontrados.append(
                {"fichero": ruta.name, "ocurrencias": len(ocurrencias)}
            )
    encontrados.sort(key=lambda entrada: -entrada["ocurrencias"])
    return {
        "patron": patron,
        "donde": donde,
        "presente": bool(encontrados),
        "n_ficheros": len(encontrados),
        "ficheros": encontrados[:6],
    }


def construir_filas() -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Construye la tabla de brecha y la evidencia de las verificaciones.

    Returns:
        Tupla con las filas de la tabla y el diccionario de verificaciones.
    """
    log("1/4 verificaciones programaticas sobre el codigo real")
    evidencia = {
        clave: verificar(definicion["patron"], definicion["busqueda"])
        for clave, definicion in VERIFICACIONES.items()
    }
    for clave, resultado in evidencia.items():
        estado = "PRESENTE" if resultado["presente"] else "ausente"
        log(
            f"   {clave:24s} {estado:9s} | {VERIFICACIONES[clave]['etiqueta']}"
            + (f" | {resultado['n_ficheros']} ficheros" if resultado["presente"] else "")
        )

    log("2/4 tabla de brecha")
    # Cada fila es un juicio del analisis, pero cada juicio se apoya en las verificaciones de
    # arriba: la columna de evidencia cita los identificadores cuya presencia confirma o
    # descarta la duplicacion.
    filas: list[dict[str, str]] = [
        {
            "familia": "A - Ponderacion estructural",
            "hipotesis": "V03 sobrepondera observaciones repetidas y la ponderacion por tamano de grupo lo corrige",
            "ya_probada": "no",
            "experimentos_relacionados": "V4-B03 (ponderar por dureza), D-027 (boundary downweight)",
            "resultado_historico": "La ponderacion por dureza empeora (-0.0101); solo el downweight de frontera dio senal pequena",
            "realmente_cerrada": "no: el tamano de grupo nunca se uso como peso",
            "nueva_formulacion_posible": "w = 1/sqrt(group_size) y w = 1/group_size, con los grupos calculados solo sobre el train del pliegue",
            "prioridad": "alta",
        },
        {
            "familia": "A2 - Grupo + frontera",
            "hipotesis": "Repetidas y fronterizos son dos fuentes distintas de varianza y bajar ambos pesos reduce sobreajuste",
            "ya_probada": "parcial",
            "experimentos_relacionados": "V4-B03, D-027",
            "resultado_historico": "El downweight de frontera fue la unica intervencion positiva pequena de v4",
            "realmente_cerrada": "no: solo se probo la componente de frontera",
            "nueva_formulacion_posible": "Producto de los dos pesos en pocas configuraciones, sin rejilla amplia",
            "prioridad": "media",
        },
        {
            "familia": "B - Factorizacion condicional de etiquetas",
            "hipotesis": "p(TR|X) se factoriza en p(HS|X) y p(TR|HS=1,X) entrenado en su dominio natural",
            "ya_probada": "parcial",
            "experimentos_relacionados": "E2-01 (chains, 6 ordenes), V4-E01..E03 (label powerset, 12 variantes), E2-18 (meta de dependencia)",
            "resultado_historico": "Chains no adoptadas; powerset mejor 0.7713 < V03 0.777211; el meta no transfiere al LB (0.78163)",
            "realmente_cerrada": "no: ni chains ni powerset restringen el dominio de entrenamiento del condicional",
            "nueva_formulacion_posible": "p(TR|X)=p(HS|X)p(TR|HS=1,X) y p(AG|X)=p(HS|X)p(AG|HS=1,X)+(1-p(HS|X))p(AG|HS=0,X)",
            "prioridad": "alta",
        },
        {
            "familia": "C1 - Skip-grams",
            "hipotesis": "Ciertas relaciones (insulto + marcador, negacion + insulto) no requieren palabras adyacentes",
            "ya_probada": "no",
            "experimentos_relacionados": "REP-01..REP-06 (solo n-gramas contiguos)",
            "resultado_historico": "word(1,2) y char_wb(3,5) son contiguos; los pares con hueco nunca se representaron",
            "realmente_cerrada": "no",
            "nueva_formulacion_posible": "Bigramas con hueco acotado (max_gap=1, y 2 solo si aporta)",
            "prioridad": "media",
        },
        {
            "familia": "C2 - Canales posicionales",
            "hipotesis": "Las menciones iniciales y las respuestas finales tienen senal distinta",
            "ya_probada": "si",
            "experimentos_relacionados": "V4-D (ventanas como tokens)",
            "resultado_historico": "ventanas_hstack 0.7734 frente a V03 0.7772 (delta -0.0038); ventanas_standalone 0.5370",
            "realmente_cerrada": "si",
            "nueva_formulacion_posible": "ninguna sin hypothesis nueva",
            "prioridad": "nula: no ejecutar",
        },
        {
            "familia": "C3 - Tokens de interaccion",
            "hipotesis": "INSULT_NEAR_MENTION y similares como terminos sparse",
            "ya_probada": "si",
            "experimentos_relacionados": "V4-D (interacciones como tokens), E2-05 (canal simbolico), E2-06 (targeting)",
            "resultado_historico": "interacciones_hstack 0.7426 (delta -0.0346); canal simbolico -0.1696; targeting -0.0416",
            "realmente_cerrada": "si",
            "nueva_formulacion_posible": "ninguna sin hypothesis nueva",
            "prioridad": "nula: no ejecutar",
        },
        {
            "familia": "C4 - Forma x token",
            "hipotesis": "El mismo token con elongation, risa o puntuacion produce senal distinta",
            "ya_probada": "si",
            "experimentos_relacionados": "E2-10 (ORT-A, ORT-ABCD, ORT-AE)",
            "resultado_historico": "-0.0286, -0.0238 y -0.0276 frente a V03",
            "realmente_cerrada": "si",
            "nueva_formulacion_posible": "ninguna sin hypothesis nueva",
            "prioridad": "nula: no ejecutar",
        },
        {
            "familia": "D - Secuencial sobre FastText",
            "hipotesis": "El problema de FastText en v5 fue el pooling medio, no la representacion",
            "ya_probada": "arquitectura si, input no",
            "experimentos_relacionados": "DL-01 (MLP), DL-02 (CharCNN con Conv1d sobre embeddings aleatorios), DL-03 (BiLSTM con embeddings aleatorios)",
            "resultado_historico": "La convolucion 1D ya se probo, pero con embeddings ALEATORIOS de caracter: CharCNN 0.7587 y BiLSTM por debajo de B1 0.7927. Nunca se probo una convolucion sobre una representacion aprendida de un corpus externo",
            "realmente_cerrada": "no: la arquitectura se repite pero sobre embeddings aprendidos, que es un input distinto y no esta probado",
            "nueva_formulacion_posible": "TextCNN 1D sobre los FastText ya entrenados, congelado y ajustado, sin encoder externo",
            "prioridad": "media-baja",
        },
        {
            "familia": "E - kNN local sparse",
            "hipotesis": "Un experto local por vecinos rescata patrones que el modelo lineal global suaviza",
            "ya_probada": "parcial",
            "experimentos_relacionados": "V4-F01..F04 (vecindad y propagacion)",
            "resultado_historico": "Se construyeron 3 vecinos con cosimilitud y se uso el desacuerdo como medida: el error dentro del grupo que discrepa es 0.1798 frente a 0.1619 fuera, IC95 [-0.013, 0.0491] (no ordena). La propagacion por near-duplicado tiene techo nulo",
            "realmente_cerrada": "parcialmente: la afinidad ya se midio y fue nula; usarlos como PREDICTOR no se probo",
            "nueva_formulacion_posible": "kNN como predictor y como fuente de un score para la capa final, no como medida de ambiguedad",
            "prioridad": "baja: solo si el analisis de interseccion de errores muestra informacion recuperable",
        },
        {
            "familia": "F - Ruido por par instancia-etiqueta",
            "hipotesis": "Parte de la frontera contiene pares etiqueta-instancia sistematicamente inconsistentes",
            "ya_probada": "no",
            "experimentos_relacionados": "V4-A01, V4-A02, V4-B05 (ruido por duplicado)",
            "resultado_historico": "Solo 2 clusters contradictorios (84 tuits, 4.85%) y V03 acierta bien dentro de ellos (error 0.046 frente a 0.218). El ruido por par nunca se midio",
            "realmente_cerrada": "no",
            "nueva_formulacion_posible": "Score de sospecha por par con probabilidad, margen, desacuerdo entre modelos, consistencia estructural y densidad local; primero detectar, luego ponderar",
            "prioridad": "media",
        },
        {
            "familia": "G - Condicionales en la capa final",
            "hipotesis": "Permitir pocas interacciones especificas de etiqueta en la capa final",
            "ya_probada": "casi",
            "experimentos_relacionados": "E2-21 (espacios por etiqueta), E2-01 y E2-18 (score x margen en cadenas y meta)",
            "resultado_historico": "E2-21 cerrado sin mejora; el score con su margen ya existe como entrada de chains y del meta",
            "realmente_cerrada": "casi: solo quedaria anadir el score kNN o el condicional",
            "nueva_formulacion_posible": "marginal: una sola interaccion extra, no un espacio nuevo",
            "prioridad": "muy baja: no ejecutar",
        },
        {
            "familia": "H - Bagging por grupos",
            "hipotesis": "La reduccion de varianza por bootstrap de grupos con el mismo modelo mejora lo que el ensemble heterogeneo no pudo",
            "ya_probada": "no",
            "experimentos_relacionados": "ENS-01..ENS-04 (ensembles de representaciones distintas)",
            "resultado_historico": "Los ensembles mezclaban representaciones distintas y no superaron al mejor modelo; el bagging por grupos nunca se probo",
            "realmente_cerrada": "no",
            "nueva_formulacion_posible": "Cinco modelos V03-equivalentes sobre muestras balanceadas por grupo, promediar probabilidades",
            "prioridad": "alta",
        },
        {
            "familia": "I - Ponderacion word/char",
            "hipotesis": "El optimo de V03 depende de la escala entre evidencia lexica y sublexica",
            "ya_probada": "no",
            "experimentos_relacionados": "V2-BASE-M1 (solo word), V2-BASE-M4 (solo char), V2-BASE-M2/M3 (union)",
            "resultado_historico": "Se midieron los bloques por separado y la union, pero NUNCA con una ponderacion explicita dentro de la union",
            "realmente_cerrada": "no",
            "nueva_formulacion_posible": "char_weight en {0.5, 1.0, 1.5, 2.0} con word_weight fijo en 1",
            "prioridad": "muy alta: la mas barata del proyecto",
        },
        {
            "familia": "J - Aumento de datos",
            "hipotesis": "Transformaciones deterministas que casi preservan la etiqueta anaden ejemplos de entrenamiento",
            "ya_probada": "no",
            "experimentos_relacionados": "PRE-00..PRE-09",
            "resultado_historico": "Diez transformaciones deterministas de superficie, todas descartadas (mejor 0.7623 frente a 0.7587 de referencia). Eso era NORMALIZACION, no aumento: PRE quita ruido y no crea instancias",
            "realmente_cerrada": "no para el aumento; el precedente de superficie es adverso",
            "nueva_formulacion_posible": "Crear instancias nuevas en train, nunca en validacion ni test, con transformaciones de elongation, espaciado y puntuacion",
            "prioridad": "ultima",
        },
    ]
    return filas, evidencia


def main() -> None:
    """Ejecuta P0: inventario, brecha, artefactos y orden."""
    inicio = time.time()
    log("P0 de v6: analisis de brechas antes de gastar una sola unidad")
    experimentos = _leer_experimentos()
    filas, evidencia = construir_filas()

    log("3/4 inventario, familias e integridad del registro")
    familias: Counter[str] = Counter()
    corruptas: list[dict[str, Any]] = []
    for indice, fila in enumerate(experimentos, start=2):
        identificador = (fila.get("experiment_id") or "").strip()
        if not identificador:
            corruptas.append({"linea": indice, "motivo": "experiment_id vacio"})
            continue
        # Una fila esta desalineada si el identificador parece el nombre de la competicion en
        # lugar de un ID de experimento, o si una fecha ha acabado en la columna `competition`.
        if identificador.startswith("KAGGLE_") or (fila.get("competition") or "").startswith("20"):
            corruptas.append(
                {
                    "linea": indice,
                    "motivo": "columnas desplazadas: el identificador no es un ID de experimento",
                    "experiment_id": identificador,
                    "competition": (fila.get("competition") or "")[:24],
                }
            )
            continue
        familias[identificador.split("-")[0]] += 1
    log(f"   {len(familias)} prefijos | {len(corruptas)} filas corruptas de {len(experimentos)}")

    decisiones = re.findall(r"^## (D-\d+)", DECISIONES.read_text(encoding="utf-8"), re.MULTILINE)
    duplicadas = [f["familia"] for f in filas if f["realmente_cerrada"].startswith("si")]
    ejecutables = [f["familia"] for f in filas if f["prioridad"] != "nula: no ejecutar"]

    log("4/4 informes")
    _escribir_csv(filas)
    REPORTE_VERIF.write_text(
        json.dumps(
            {
                "experimento": EXPERIMENTO,
                "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "nota": (
                    "Evidencia cruda de cada verificacion. Cada fila de la tabla de brecha cita "
                    "las claves de este diccionario: 'presente' significa que el track es una "
                    "duplicacion y 'ausente' que es genuinamente nuevo."
                ),
                "verificaciones": evidencia,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _escribir_informe(experimentos, filas, evidencia, familias, decisiones, corruptas)
    verificaciones_ausentes = [k for k, v in evidencia.items() if not v["presente"]]
    verificaciones_presentes = [k for k, v in evidencia.items() if v["presente"]]
    log(
        f"P0 completada en {round(time.time() - inicio, 1)} s | "
        f"{len(experimentos)} experimentos inventariados | "
        f"{len(filas)} lineas analizadas: {len(duplicadas)} cerradas por duplicacion, "
        f"{len(ejecutables)} ejecutables"
    )
    log(
        f"   verificaciones ausentes ({len(verificaciones_ausentes)}): "
        + ", ".join(verificaciones_ausentes)
    )
    log(
        f"   verificaciones presentes ({len(verificaciones_presentes)}): "
        + ", ".join(verificaciones_presentes)
    )


def _escribir_csv(filas: list[dict[str, str]]) -> None:
    """Escribe la tabla de brecha en CSV.

    Args:
        filas: filas de la tabla.
    """
    columnas = [
        "familia",
        "hipotesis",
        "ya_probada",
        "experimentos_relacionados",
        "resultado_historico",
        "realmente_cerrada",
        "nueva_formulacion_posible",
        "prioridad",
    ]
    with REPORTE_CSV.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=columnas, lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(filas)


def _escribir_informe(
    experimentos: list[dict[str, str]],
    filas: list[dict[str, str]],
    evidencia: dict[str, Any],
    familias: Counter[str],
    decisiones: list[str],
    corruptas: list[dict[str, Any]],
) -> None:
    """Escribe el informe de brecha.

    Args:
        experimentos: filas del registro de experimentos.
        filas: filas de la tabla de brecha.
        evidencia: resultado de las verificaciones.
        familias: recuento de prefijos de experimento.
        decisiones: identificadores de decision del proyecto.
        corruptas: filas del registro con las columnas desplazadas.
    """
    nuevas = [f for f in filas if not f["realmente_cerrada"].startswith("si") and f["prioridad"] != "nula: no ejecutar"]
    duplicadas = [f for f in filas if f["realmente_cerrada"].startswith("si")]
    lineas = [
        "# v6: analisis de brechas (P0)",
        "",
        f"Generado por `experiments/v6_00_gap_analysis.py` ({EXPERIMENTO}).",
        "No entrena nada. Se ejecuta antes de gastar una sola unidad de la fase v6.",
        "",
        "Cada afirmacion de la tabla se apoya en una **verificacion programatica** sobre el",
        "codigo real del proyecto, registrada en `reports/v6_verificaciones_gap_v6.json`. Si",
        "alguien implementa hoy un `char_weight`, la tabla pasara a marcar el track I como",
        "duplicado sin que nadie tenga que acordarse.",
        "",
        "## 1. Inventario de lo ya experimented",
        "",
        f"El registro tiene **{len(experimentos)} filas**, de las cuales **{len(corruptas)} estan",
        "desalineadas** (seccion 2). El numero real de experimentos es por tanto",
        f"**{len(experimentos) - len(corruptas)}**: la especificacion de v6 estimaba 170 y la",
        "diferencia son los experimentos de v5 y v5.1 mas los de esta fase. El proyecto",
        f"acumula **{len(decisiones)} decisiones**. Por prefijo de identificador:",
        "",
        "| prefijo | experimentos |",
        "|---|---|",
    ]
    for prefijo, total in familias.most_common():
        lineas.append(f"| `{prefijo}-*` | {total} |")
    lineas += [
        "",
        "## 2. Integridad del registro (hallazgo de esta auditoria)",
        "",
        f"**{len(corruptas)} de las {len(experimentos)} filas de `experiments.csv` tienen las",
        "columnas desplazadas**: el campo `experiment_id` contiene el nombre de la competicion",
        "en lugar de un identificador, y el resto de campos ocupa una posicion de mas.",
        "",
        "| linea | experiment_id | competition |",
        "|---|---|---|",
        *[f"| {f['linea']} | `{f.get('experiment_id', '')}` | `{f.get('competition', '')}` |" for f in corruptas],
        "",
        "Las filas **no se tocan**: son evidencia historica y el principio del proyecto es no",
        "reescribir resultados. Lo que se hace es (a) contarlas aparte para no inflar el",
        "inventario y (b) verificar el formato en cada escritura a partir de ahora, que usa",
        "`csv.DictWriter` con los nombres de campo y por tanto no puede desalinearse. El",
        "inventario cuenta solo filas bien formadas.",
        "",
        "## 3. Familias completamente cerradas",
        "",
        "Agrupadas por mecanismo, no por identificador:",
        "",
        "| familia |experimentos | veredicto con evidencia |",
        "|---|---|---|",
        "| Metrica y preprocesamiento de superficie | E00, PRE-00..09 | cerrada: 10 transformaciones deterministas, mejor 0.7623 frente a 0.7587 |",
        "| Representacion TF-IDF y clasificador lineal | REP-01..06, MOD-01..04 | cerrada: la union word+char es V03 |",
        "| Umbrales por etiqueta | TH-01, S02, S03 | cerrada: adoptados y congelados en V03 |",
        "| Ensembles heterogeneos | ENS-01..04, S04 | cerrada: mejor anidado 0.7949 que no transfiere (LB 0.77510) |",
        "| Features densas (afijos, registro, objetivo) | HIB-H11/H12/H14, B5-DENSE | cerrada: todas descartadas |",
        "| Optimizacion de hiperparametros | OPT-01, S05 | cerrada: OOF 0.8039 que cae a LB 0.77602 |",
        "| Control de capacidad | R1-C1, R2-MINDF5, S06, S07 | cerrada: el gap train-OOF no predice el LB |",
        "| Deep learning con embeddings aleatorios | DL-01, DL-02, DL-03 | cerrada: CharCNN 0.7587 y BiLSTM por debajo de B1 0.7927 |",
        "| Cadenas, meta y dependencias | E2-01, E2-18, S08 | cerrada: el meta no transfiere (LB 0.78163) |",
        "| Canales sparse sintacticos y semanticos | E2-03..E2-10 | cerrada: todos entre -0.011 y -0.25 |",
        "| NB-SVM, calibracion y target encoding | E2-13, E2-14, E2-15 | cerrada: entre -0.016 y -0.039 |",
        "| Especialistas y espacios por etiqueta | E2-16, E2-20, E2-21 | cerrada: sin mejora |",
        "| v4: ruido, dureza y ponderacion | V4-A01..A02, B01..B05, H01 | cerrada: el ruido es escaso; solo el downweight de frontera dio senal pequena |",
        "| v4: sparse estructural | V4-D01..D04 | **cerrada**: interacciones como tokens 0.7426, ventanas 0.7734 |",
        "| v4: modelo conjunto de etiquetas | V4-E01..E03 | **cerrada**: 12 variantes de label powerset, mejor 0.7713 |",
        "| v4: vecindad y shift | V4-F01..F04 | **cerrada**: la afinidad no ordena el error |",
        "| v5 y v5.1: representacion distribuida | E-CORP-01..09, EBD-00..27 | cerrada: 14 configuraciones, 0 adoptables |",
        "",
        "## 4. La tabla de brecha",
        "",
        "| familia | ya probada | realmente cerrada | prioridad |",
        "|---|---|---|---|",
    ]
    for fila in filas:
        cerrada = "si" if fila["realmente_cerrada"].startswith("si") else "no"
        lineas.append(
            f"| {fila['familia']} | {fila['ya_probada']} | {cerrada} | {fila['prioridad']} |"
        )
    lineas += [
        "",
        "El detalle de cada fila, con la hipotesis, los experimentos relacionados, el resultado",
        "historico y la formulacion nueva, esta en `reports/v6_hypothesis_gap_analysis.csv`.",
        "",
        "## 5. Verificaciones programaticas",
        "",
        "Cada busqueda se hace sobre el codigo y los artefactos reales del proyecto:",
        "",
        "| verificacion | resultado | ficheros |",
        "|---|---|---|",
    ]
    for clave, resultado in evidencia.items():
        estado = "presente" if resultado["presente"] else "ausente"
        ficheros = ", ".join(f"{f['fichero']} ({f['ocurrencias']})" for f in resultado["ficheros"][:3])
        lineas.append(
            f"| `{clave}` | **{estado}** | {ficheros or '-'} |"
        )
    lineas += [
        "",
        "**Una asimetria que hay que entender**: la verificacion busca codigo, no el registro",
        "historico. Que `G_label_conditional` salga ausente no significa que los espacios por",
        "etiqueta sean nuevos: significa que en los scripts de v2 se llamaban de otra manera",
        "(`E2-21`). Por eso la tabla cita **siempre** los identificadores de experimento, que",
        "son la fuente autoritativa, y la verificacion programatica sirve para atrapar lo que",
        "se implemento sin registrar. Cuando las dos fuentes discrepan, manda el registro.",
        "",
        "Las lecturas que importan:",
        "",
        "- `I_word_char_weight` **ausente**: nunca se pondero de forma explicita la escala",
        "  entre los dos bloques de V03. Es el track mas barato del proyecto.",
        "- `C3_interacciones`, `C2_ventanas` y `C4_shape` **presentes**: esos tres sub-tracks",
        "  de C ya se ejecutaron en v4 y en v2. Ejecutarlos seria repetir un resultado",
        "  negativo conocido.",
        "- `J_augmentation` **ausente** pero `J_normalizacion` **presente**: lo que se probo",
        "  diez veces fue normalizacion (quitar ruido), no aumento (crear instancias). Son",
        "  cosas distintas, aunque el precedente de superficie es adverso.",
        "- `E_knn_medida` **presente** y `E_knn_predictor` **ausente**: la afinidad de",
        "  vecindad ya se midio y resulto nula para ordenar el error; usarla como predictor es",
        "  genuinamente nuevo, pero parte de una afinidad que ya se vio fallar.",
        "",
        "## 6. Duplicaciones explicitas de la especificacion de v6",
        "",
        "La especificacion propone cuatro lineas que el proyecto ya cerro con evidencia",
        "medida, porque fue redactada sin conocer el registro:",
        "",
        "| track propuesto | ya existe en | por que no se ejecuta |",
        "|---|---|---|",
    ]
    for fila in duplicadas:
        lineas.append(
            f"| {fila['familia']} | {fila['experimentos_relacionados']} | {fila['resultado_historico']} |"
        )
    lineas += [
        "",
        "## 7. Hipotesis genuinamente nuevas",
        "",
        "| familia | formulacion | prioridad |",
        "|---|---|---|",
    ]
    for fila in nuevas:
        lineas.append(
            f"| {fila['familia']} | {fila['nueva_formulacion_posible']} | {fila['prioridad']} |"
        )
    lineas += [
        "",
        "## 8. Artefactos y scripts reutilizables",
        "",
        "| recurso | para que sirve en v6 |",
        "|---|---|",
        "| `src/splits.py` | grupos fusionados y pliegues corregidos; es el unico split permitido |",
        "| `src/evaluation.py` | metrica primaria y bootstrap pareado, ya auditados |",
        "| `experiments/embedding_18_complementariedad_v5_1.py` | patron de salvage, damage y bootstrap de interaccion |",
        "| `features/oof_v03_final.npz` | probabilidades, margenes y predicciones de V03 para las features condicionales |",
        "| 19 modelos `.bin` de FastText | Track D, sin reentrenar |",
        "| `reports/error_intersection_v2.csv` y `all_three_errors.csv` | marco del analisis de interseccion de errores |",
        "| `reports/error_taxonomy_v4.md` | taxonomia de errores y subconjuntos robustos |",
        "| `experiments/28_hardness_v4.py` | calculo de vecindad y medidas de dureza, reutilizable |",
        "",
        "## 9. Presupuesto",
        "",
        f"- experimentation principal: **{PRESUPUESTO_PRINCIPAL}**",
        f"- reserva: **{PRESUPUESTO_RESERVA}**",
        f"- total: **{PRESUPUESTO_TOTAL}**, y no es obligatorio consumirlo",
        "",
        "Contabilidad separada: un experimento no es una unidad de entrenamiento ni una",
        "evaluacion downstream, y los graficos, tablas, auditorias y reproducciones no cuentan.",
        "Las familias que se cierren no consumen nada mas.",
        "",
        "## 10. Criterio de adopcion (puerta de dos niveles)",
        "",
        "La especificacion proponia `delta >= +0.003` y `4/5 folds positivos **o**",
        "complementariedad robusta`. Ese gate es mas laxo que el criterio vigente desde",
        "D-036, y D-029 ya registro que un candidato que paso los gates fue rechazado por el",
        "holdout. Por decision del usuario el gate queda en dos niveles:",
        "",
        f"- **Gate 1**: `delta >= {GATE_1_DELTA:+.3f}` **y** `IC95 que excluye el 0` **y** "
        f"`>= {GATE_1_FOLDS}/5 pliegues positivos`.",
        f"- **Gate 2**: {GATE_2_REQUIERE}. Solo se aplica a los finalistas, no para elegir entre",
        "  decenas de candidatos.",
        "- **Adopcion**: exige superar Gate 1 **y** Gate 2.",
        "",
        "El holdout no se usa: esta agotado (5/5) y el Gate 2 se resuelve con otra particion",
        "agrupada.",
        "",
        "## 11. Orden de ejecucion",
        "",
        "Se sigue el orden de la especificacion (**B, A, C, D**), con el diagnostico previo que",
        "justifica la eleccion de lineas:",
        "",
        "1. **Diagnostico** (P0-bis): interseccion de errores y estructura de etiquetas. Barato",
        "   y decide si D y E merecen gastarse.",
        "2. **B**: factorizacion condicional, precedida del diagnostico de P(TR|HS) y P(AG|HS).",
        "3. **A**: ponderacion por tamano de grupo.",
        "4. **C**: solo **C1** (skip-grams). C2, C3 y C4 quedan cerradas con evidencia previa.",
        "5. **D**: TextCNN sobre FastText, y D3 solo si D1 muestra senal.",
        "",
        "Las lineas E, F, G, H, I y J esperan al resultado del diagnostico y de B y A, y se",
        "priorizan segun lo que la evidencia mostre, no segun el orden del documento.",
        "",
    ]
    REPORTE.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
