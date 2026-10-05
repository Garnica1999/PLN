"""21_pack_validacion_m4.py
Fecha: 2026-10-03
Descripcion: Pack de validacion humana muestral de las etiquetas SILVER de M4
(bloque constitucional predominante entre las citas NER de la Constitucion).
Prepara la muestra estratificada del corpus etiquetado completo (23.654 docs),
la guia de anotacion y las utilidades para anotar (reanudable), resumir,
exportar y medir el acuerdo humano vs silver (exactitud global y por clase,
matriz de confusion, kappa de Cohen por pares y Fleiss global). Este script NO
entrena nada: la anotacion es diagnostico y por eso la muestra puede incluir
documentos del holdout sellado.

Modos (mutuamente excluyentes):
    --generar           construye muestra_200.jsonl + guia_anotacion.md
    --ver ID            muestra un documento (metadata + top3 + extracto 3000)
    --anotar            recorrido interactivo reanudable (requiere --anotador)
    --resumen           estado de la anotacion (respondidos, faltantes, distribucion)
    --exportar-csv      CSV plano (id, area_silver, respuesta, ...) para Excel
    --calcular-acuerdo  exactitud vs silver, matriz, kappa de Cohen y Fleiss
    --smoke             variante de 30 docs con TODO en /tmp/opencode/o1_smoke/
    --forzar            permite reescribir salidas existentes (no aplica a smoke)

Diseno de la muestra:
    Estratificada por area_primaria (16 clases del espacio fijo): minimo 5 por
    clase (o todos si la clase tiene menos de 5), el resto proporcional por el
    metodo del resto mayor (Hamilton); semilla 42. El orden final se baraja para
    no agrupar por clase. Se registra la particion canonica de cada documento
    (data/splits_m4.parquet) y si cae en el holdout sellado
    (data/holdout_m4_bloqueado.parquet), dato informativo: la anotacion no
    entrena ni toca el holdout.

Blinding en --anotar:
    Se muestra primero la metadata y el extracto (3000 caracteres) y el bloque
    SILVER (area + top3) al FINAL, para que la decision se tome leyendo el texto
    y el silver se use solo como verificacion (regla de oro de la guia).

Lectura del corpus:
    data/corpus_sentencias.parquet es un dataset Hive particionado por tipo/ano;
    se lee SOLO por id con filtros pyarrow (dataset.to_table(filter=id.isin(...)))
    y por lotes, sin cargar el corpus completo.

Estado del arte aplicado:
    - Validacion humana muestral para estimar la precision de etiquetas debiles
      (Northcutt et al., 2021; mismo espiritu del script 17 de auditoria).
    - Muestreo estratificado con asignacion por resto mayor (Hamilton) y semilla 42.
    - Acuerdo inter-anotador: kappa de Cohen (1960) por pares, Fleiss (1971)
      global sobre casos completos e interpretacion de Landis y Koch (1977).
    - Reutilizacion por importlib del script 17 (md5_archivo, versiones_entorno)
      y de su patron de guardas de no-sobrescritura; el script 06 no se importa
      porque este pack no entrena ni vectoriza.

Uso:
    /home/carlos/venv/bin/python scripts/21_pack_validacion_m4.py --generar
    /home/carlos/venv/bin/python scripts/21_pack_validacion_m4.py --ver 12345
    /home/carlos/venv/bin/python scripts/21_pack_validacion_m4.py --anotar --anotador carlos
    /home/carlos/venv/bin/python scripts/21_pack_validacion_m4.py --resumen
    /home/carlos/venv/bin/python scripts/21_pack_validacion_m4.py --exportar-csv
    /home/carlos/venv/bin/python scripts/21_pack_validacion_m4.py --calcular-acuerdo
    /home/carlos/venv/bin/python scripts/21_pack_validacion_m4.py --smoke --generar
"""

import argparse
import importlib.util
import json
import logging
import os
import random
import sys
import time
import traceback
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

# ===== SECCION 1: configuracion =====

RAIZ = Path(__file__).resolve().parent.parent

# Espacio fijo de 16 clases (orden alfabetico, identico al de los scripts 17/06).
CLASES_M4 = [
    "derechos_colectivos_ambiente",
    "derechos_fundamentales",
    "derechos_sociales_economicos_culturales",
    "elecciones",
    "habitantes_territorio",
    "organismos_control",
    "organizacion_estado",
    "organizacion_territorial",
    "participacion_democratica",
    "principios_fundamentales",
    "proteccion_aplicacion_derechos",
    "rama_ejecutiva",
    "rama_judicial",
    "rama_legislativa",
    "reforma_constitucion",
    "regimen_economico",
]

# Alias cortos por clase (normalizados sin tildes) para la anotacion interactiva.
ALIAS_CLASES = {
    "derechos_colectivos_ambiente": ("ambiente", "colectivos"),
    "derechos_fundamentales": ("fundamentales", "df"),
    "derechos_sociales_economicos_culturales": ("sociales", "desc"),
    "elecciones": ("elecciones", "elec"),
    "habitantes_territorio": ("habitantes", "fronteras"),
    "organismos_control": ("control", "organismos"),
    "organizacion_estado": ("estado",),
    "organizacion_territorial": ("territorial",),
    "participacion_democratica": ("participacion", "democracia"),
    "principios_fundamentales": ("principios", "pf"),
    "proteccion_aplicacion_derechos": ("proteccion", "tutela"),
    "rama_ejecutiva": ("ejecutiva", "gobierno"),
    "rama_judicial": ("judicial",),
    "rama_legislativa": ("legislativa", "congreso"),
    "reforma_constitucion": ("reforma",),
    "regimen_economico": ("economico", "regimen"),
}

# Respuestas especiales (fuera del espacio de 16 clases).
RESPUESTAS_ESPECIALES = ("ambiguo", "ninguno")

CONFIG = {
    "semilla": 42,
    "n_muestra": 200,
    "n_muestra_smoke": 30,
    "minimo_por_clase": 5,
    "minimo_por_clase_smoke": 1,
    "chars_extracto_pack": 1500,
    "chars_extracto_ver": 3000,
    "lote_lectura_corpus": 64,
    "dir_smoke": "/tmp/opencode/o1_smoke",
    "rutas_datos": {
        "etiquetas": "data/etiquetas_areas.parquet",
        "splits": "data/splits_m4.parquet",
        "corpus": "data/corpus_sentencias.parquet",
        "holdout": "data/holdout_m4_bloqueado.parquet",
    },
    "rutas_salida": {
        "muestra": "resultados/validacion_m4/muestra_200.jsonl",
        "guia": "resultados/validacion_m4/guia_anotacion.md",
        "respuestas": "resultados/validacion_m4/respuestas_{anotador}.jsonl",
        "export_csv": "resultados/validacion_m4/anotaciones_export.csv",
        "acuerdo": "resultados/validacion_m4/acuerdo_v1.json",
    },
    "rutas_salida_smoke": {
        "muestra": "/tmp/opencode/o1_smoke/muestra_smoke.jsonl",
        "guia": "/tmp/opencode/o1_smoke/guia_anotacion.md",
        "respuestas": "/tmp/opencode/o1_smoke/respuestas_{anotador}.jsonl",
        "export_csv": "/tmp/opencode/o1_smoke/anotaciones_export.csv",
        "acuerdo": "/tmp/opencode/o1_smoke/acuerdo_v1.json",
    },
}

# Premisas de diseno congeladas en acuerdo_v1.json (trazabilidad).
PREMISAS = [
    (
        "La muestra se toma del corpus etiquetado completo (23.654 documentos con "
        "area_primaria distinta de sin_area_definida), uniendo data/etiquetas_areas.parquet "
        "con data/splits_m4.parquet; incluye documentos de train/val/test y del holdout "
        "sellado (data/holdout_m4_bloqueado.parquet) porque la anotacion es diagnostico y "
        "NO entrena nada."
    ),
    (
        "El silver M4 es el bloque mayoritario de citas NER de la Constitucion "
        "(area_primaria, via=cita_cp) o el fallback lexico (via=fallback_derecho, menos "
        "fiable); top3_json lista los tres bloques con mas articulos distintos citados y "
        "su conteo. La validacion mide la precision de esa senal automatica, no un ground "
        "truth humano preexistente."
    ),
    (
        "Muestreo estratificado por area_silver: minimo 5 por clase (o todos si la clase "
        "tiene menos), el resto proporcional por resto mayor (Hamilton); semilla 42; el "
        "orden final se baraja para no agrupar por clase."
    ),
    (
        "Exactitud humana vs silver: sobre respuestas validas del espacio de 16 clases; "
        "ambiguo/ninguno se reportan aparte (porcentaje) y tambien se cuentan como error "
        "en la variante conservadora (exactitud_todas)."
    ),
    (
        "Kappa de Cohen por pares (categorias = 16 clases + ambiguo + ninguno, sobre ids "
        "respondidos por ambos) y Fleiss global solo sobre casos completos (ids "
        "respondidos por TODOS los anotadores); interpretacion Landis y Koch (1977)."
    ),
    (
        "El corpus se lee SOLO por id con filtros pyarrow sobre el dataset particionado "
        "tipo/ano; nunca se carga completo y no se modifica ningun archivo de datos ni el "
        "holdout sellado."
    ),
]

LOGGER_NOMBRE = "pack_validacion_m4_21"

# Cache del modulo 17 (su nombre empieza por digito: importlib por ruta).
_MODULO_AUDITORIA = None


# ===== SECCION 2: utilidades y reutilizacion del modulo 17 =====


def cargar_modulo_auditoria():
    """Importa scripts/17_auditoria_etiquetas_m4.py por ruta (cacheado).

    El nombre del modulo empieza por digito, por lo que no puede importarse
    con la sintaxis normal; se usa importlib.util.spec_from_file_location
    para reutilizar md5_archivo y versiones_entorno sin duplicarlas.

    Returns:
        Modulo 17_auditoria_etiquetas_m4 ya ejecutado.

    Raises:
        ImportError: Si el archivo no existe o no se puede cargar.
    """
    global _MODULO_AUDITORIA
    if _MODULO_AUDITORIA is None:
        ruta = RAIZ / "scripts" / "17_auditoria_etiquetas_m4.py"
        spec = importlib.util.spec_from_file_location("auditoria_etiquetas_m4_17", ruta)
        if spec is None or spec.loader is None:
            raise ImportError(f"no se pudo cargar {ruta}")
        modulo = importlib.util.module_from_spec(spec)
        sys.modules["auditoria_etiquetas_m4_17"] = modulo
        spec.loader.exec_module(modulo)
        _MODULO_AUDITORIA = modulo
    return _MODULO_AUDITORIA


def md5_archivo(ruta: str | Path) -> str | None:
    """Calcula el md5 de un archivo por bloques (delegado al modulo 17).

    Args:
        ruta: Ruta del archivo.

    Returns:
        Hexdigest md5, o None si el archivo no existe.
    """
    return cargar_modulo_auditoria().md5_archivo(str(ruta))


def versiones_entorno() -> dict:
    """Recolecta versiones de interprete y librerias clave (modulo 17).

    Returns:
        Dict con python, plataforma y versiones de librerias clave.
    """
    return cargar_modulo_auditoria().versiones_entorno()


def configurar_logger() -> logging.Logger:
    """Configura el logger de consola (mismo formato del modulo 17).

    Returns:
        Logger con formato fecha | nivel | mensaje, solo consola.
    """
    logger = logging.getLogger(LOGGER_NOMBRE)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formato = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    manejador = logging.StreamHandler()
    manejador.setFormatter(formato)
    logger.addHandler(manejador)
    logger.propagate = False
    return logger


def rutas_activas(smoke: bool) -> dict:
    """Construye el mapa de rutas de entrada/salida segun el modo smoke.

    Args:
        smoke: True para usar salidas exclusivamente en /tmp/opencode/o1_smoke.

    Returns:
        Dict combinado de rutas de datos y de salidas.
    """
    rutas = {}
    rutas.update(CONFIG["rutas_datos"])
    rutas.update(CONFIG["rutas_salida_smoke"] if smoke else CONFIG["rutas_salida"])
    return rutas


def verificar_salida_nueva(ruta: str | Path, forzar: bool, smoke: bool, logger: logging.Logger) -> Path:
    """Aborta si la salida final ya existe (patron de guarda del modulo 17).

    Args:
        ruta: Ruta de salida.
        forzar: True permite reescribir (siempre permitido en smoke).
        smoke: True si es corrida smoke (scratch reescribible).
        logger: Logger activo.

    Returns:
        Path de la salida validada.

    Raises:
        FileExistsError: Si la salida existe y no es smoke ni forzar.
    """
    destino = Path(ruta)
    if destino.exists() and not (smoke or forzar):
        raise FileExistsError(
            f"salida existente (no se sobrescribe; use --forzar para rehacer): {destino}"
        )
    if destino.exists():
        logger.warning("salida existente se reescribe (%s): %s", "smoke" if smoke else "forzar", destino)
    return destino


def normalizar_entrada(texto: str) -> str:
    """Normaliza texto de entrada: minusculas, sin tildes, espacios simples.

    Args:
        texto: Texto crudo del usuario.

    Returns:
        Texto normalizado (NFKD sin marcas combinantes, minusculas, strip).
    """
    descompuesto = unicodedata.normalize("NFKD", texto)
    sin_tildes = "".join(c for c in descompuesto if not unicodedata.combining(c))
    return " ".join(sin_tildes.lower().split())


def extracto_texto(texto: str, n_chars: int) -> str:
    """Extrae el inicio del documento normalizando finales de linea.

    Args:
        texto: Texto completo del documento.
        n_chars: Maximo de caracteres a conservar.

    Returns:
        Extracto del inicio (sin CR, con strip).
    """
    limpio = (texto or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    return limpio[:n_chars]


def formatear_top3(top3_json: str) -> str:
    """Formatea el top3_json de bloques citados para consola.

    Args:
        top3_json: String JSON con la lista [{bloque, n}, ...].

    Returns:
        Representacion legible "1) bloque (n=5); 2) ...".
    """
    try:
        lista = json.loads(top3_json)
    except (TypeError, ValueError):
        return str(top3_json)
    if not isinstance(lista, list):
        return str(top3_json)
    partes = []
    for posicion, item in enumerate(lista, 1):
        if isinstance(item, dict):
            partes.append(f"{posicion}) {item.get('bloque', '?')} (n={item.get('n', '?')})")
        else:
            partes.append(f"{posicion}) {item}")
    return "; ".join(partes)


def abrir_corpus(ruta_corpus: str | Path) -> ds.Dataset:
    """Abre el corpus particionado (tipo/ano) sin cargarlo en memoria.

    Args:
        ruta_corpus: Ruta del dataset parquet particionado.

    Returns:
        Dataset pyarrow listo para leer por id con filtros.
    """
    return ds.dataset(str(ruta_corpus), format="parquet", partitioning="hive")


def leer_textos_por_ids(
    dataset: ds.Dataset, ids: list[str], lote: int, logger: logging.Logger
) -> dict[str, str]:
    """Lee SOLO el texto de los ids pedidos con filtros pyarrow, por lotes.

    Args:
        dataset: Dataset del corpus ya abierto.
        ids: Ids a leer (strings).
        lote: Tamano de lote del filtro isin (limita memoria y tiempo).
        logger: Logger activo.

    Returns:
        Dict id -> texto (string vacio si el texto viniera nulo).
    """
    textos: dict[str, str] = {}
    for inicio in range(0, len(ids), lote):
        grupo = [str(v) for v in ids[inicio : inicio + lote]]
        tabla = dataset.to_table(filter=ds.field("id").isin(grupo), columns=["id", "texto"])
        for fila in tabla.to_pylist():
            textos[str(fila["id"])] = fila["texto"] or ""
    faltantes = [str(v) for v in ids if str(v) not in textos]
    if faltantes:
        logger.warning(
            "corpus sin texto para %d ids (primeros 10): %s", len(faltantes), faltantes[:10]
        )
    return textos


def escribir_jsonl(ruta: str | Path, registros: list[dict]) -> None:
    """Escribe una lista de registros como JSON Lines.

    Args:
        ruta: Ruta de salida.
        registros: Lista de dicts serializables.
    """
    destino = Path(ruta)
    destino.parent.mkdir(parents=True, exist_ok=True)
    with open(destino, "w", encoding="utf-8") as archivo:
        archivo.writelines(json.dumps(registro, ensure_ascii=False) + "\n" for registro in registros)


def escribir_jsonl_append(ruta: str | Path, registro: dict) -> None:
    """Agrega un registro al final de un JSON Lines (append con flush).

    Args:
        ruta: Ruta del archivo (se crea si no existe).
        registro: Dict serializable.
    """
    destino = Path(ruta)
    destino.parent.mkdir(parents=True, exist_ok=True)
    with open(destino, "a", encoding="utf-8") as archivo:
        archivo.write(json.dumps(registro, ensure_ascii=False) + "\n")
        archivo.flush()


def cargar_jsonl(ruta: str | Path) -> list[dict]:
    """Carga un JSON Lines validando linea por linea.

    Args:
        ruta: Ruta del archivo.

    Returns:
        Lista de registros.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si alguna linea no es JSON valido o el archivo esta vacio.
    """
    archivo = Path(ruta)
    if not archivo.exists():
        raise FileNotFoundError(f"no existe el archivo: {archivo}")
    registros = []
    with open(archivo, encoding="utf-8") as manejador:
        for numero, linea in enumerate(manejador, 1):
            linea = linea.strip()
            if not linea:
                continue
            try:
                registros.append(json.loads(linea))
            except json.JSONDecodeError as exc:
                raise ValueError(f"jsonl invalido en {archivo}:{numero}") from exc
    if not registros:
        raise ValueError(f"archivo vacio: {archivo}")
    return registros


def cargar_muestra(rutas: dict) -> list[dict]:
    """Carga el pack de la muestra activa.

    Args:
        rutas: Dict de rutas activas.

    Returns:
        Lista de registros del pack (orden de anotacion).

    Raises:
        FileNotFoundError: Si no existe la muestra (hay que ejecutar --generar).
    """
    archivo = Path(rutas["muestra"])
    if not archivo.exists():
        raise FileNotFoundError(f"no existe la muestra; ejecute --generar primero: {archivo}")
    return cargar_jsonl(archivo)


def normalizar_anotador(nombre: str) -> str:
    """Normaliza el nombre del anotador para usarlo como nombre de archivo.

    Args:
        nombre: Nombre crudo.

    Returns:
        Nombre con solo letras, digitos, guion y guion bajo.

    Raises:
        ValueError: Si el nombre queda vacio.
    """
    limpio = "".join(c if (c.isalnum() or c in "-_") else "_" for c in nombre.strip())
    limpio = limpio.strip("_")
    if not limpio:
        raise ValueError("nombre de anotador vacio o invalido")
    return limpio


def cargar_respuestas(ruta: str | Path) -> dict[str, dict]:
    """Carga las respuestas de un anotador (id -> ultimo registro).

    Args:
        ruta: Ruta del jsonl de respuestas (si no existe, dict vacio).

    Returns:
        Dict id -> registro (si un id aparece dos veces gana el ultimo).

    Raises:
        ValueError: Si alguna linea no es JSON valido.
    """
    archivo = Path(ruta)
    respuestas: dict[str, dict] = {}
    if not archivo.exists():
        return respuestas
    with open(archivo, encoding="utf-8") as manejador:
        for numero, linea in enumerate(manejador, 1):
            linea = linea.strip()
            if not linea:
                continue
            try:
                registro = json.loads(linea)
            except json.JSONDecodeError as exc:
                raise ValueError(f"jsonl invalido en {archivo}:{numero}") from exc
            if "id" in registro:
                respuestas[str(registro["id"])] = registro
    return respuestas


def cargar_todas_respuestas(rutas: dict) -> dict[str, dict[str, dict]]:
    """Carga todas las respuestas_*.jsonl del directorio de la muestra.

    Args:
        rutas: Dict de rutas activas.

    Returns:
        Dict anotador -> (id -> registro), ordenado por nombre.
    """
    directorio = Path(rutas["muestra"]).parent
    prefijo = "respuestas_"
    resultado: dict[str, dict[str, dict]] = {}
    for archivo in sorted(directorio.glob(f"{prefijo}*.jsonl")):
        anotador = archivo.stem[len(prefijo) :]
        if anotador:
            resultado[anotador] = cargar_respuestas(archivo)
    return resultado


def cargar_universo_etiquetado(logger: logging.Logger) -> pd.DataFrame:
    """Carga el universo etiquetado (area != sin_area_definida) con particion.

    Args:
        logger: Logger activo.

    Returns:
        DataFrame con id, tipo, ano, area_primaria, top3_json, via,
        n_citas_cp y particion.

    Raises:
        RuntimeError: Si algun documento etiquetado no tiene particion.
    """
    etiquetas = pq.read_table(CONFIG["rutas_datos"]["etiquetas"]).to_pandas()
    etiquetas = etiquetas[etiquetas["area_primaria"] != "sin_area_definida"].copy()
    splits = pq.read_table(CONFIG["rutas_datos"]["splits"]).to_pandas()
    universo = etiquetas.merge(splits, on="id", how="left")
    faltantes = int(universo["particion"].isna().sum())
    if faltantes:
        raise RuntimeError(
            f"{faltantes} documentos etiquetados sin particion en {CONFIG['rutas_datos']['splits']}"
        )
    columnas = ["id", "tipo", "ano", "area_primaria", "top3_json", "via", "n_citas_cp", "particion"]
    universo = universo[columnas].reset_index(drop=True)
    logger.info(
        "universo etiquetado: %d docs | particiones %s | clases %d",
        len(universo),
        universo["particion"].value_counts().to_dict(),
        universo["area_primaria"].nunique(),
    )
    return universo


def cargar_holdout_ids() -> set[str]:
    """Carga el conjunto de ids del holdout sellado (solo lectura).

    Returns:
        Set de ids del holdout.
    """
    tabla = pq.read_table(CONFIG["rutas_datos"]["holdout"], columns=["id"]).to_pandas()
    return set(tabla["id"].astype(str))


# ===== SECCION 3: cuotas, muestra y pack =====


def asignar_cuotas(conteos: dict[str, int], objetivo: int, minimo: int) -> dict[str, int]:
    """Asigna cuotas por clase: minimo por clase, resto proporcional (Hamilton).

    El minimo se recorta al tamano de la clase (si la clase tiene menos
    documentos que el minimo, se toman todos). El resto se reparte
    proporcionalmente por el metodo del resto mayor, con desempate
    determinista (mayor conteo, luego nombre de clase).

    Args:
        conteos: Dict clase -> documentos disponibles.
        objetivo: Tamano total deseado de la muestra.
        minimo: Cuota minima por clase.

    Returns:
        Dict clase -> cuota asignada (suma <= objetivo; exactamente objetivo
        si hay documentos suficientes).

    Raises:
        ValueError: Si el objetivo o el minimo son negativos.
    """
    if objetivo < 0 or minimo < 0:
        raise ValueError("objetivo y minimo deben ser no negativos")
    clases = sorted(conteos)
    objetivo = min(objetivo, int(sum(conteos.values())))
    cuotas = {clase: min(minimo, conteos[clase]) for clase in clases}
    restante = objetivo - sum(cuotas.values())
    while restante > 0:
        candidatos = [clase for clase in clases if cuotas[clase] < conteos[clase]]
        if not candidatos:
            break
        suma = float(sum(conteos[clase] for clase in candidatos))
        exactos = {clase: restante * conteos[clase] / suma for clase in candidatos}
        asignados = 0
        for clase in candidatos:
            extra = min(int(exactos[clase]), conteos[clase] - cuotas[clase])
            cuotas[clase] += extra
            asignados += extra
        restante -= asignados
        if restante > 0:
            sobrantes = [clase for clase in candidatos if cuotas[clase] < conteos[clase]]
            sobrantes.sort(
                key=lambda clase: (
                    -(exactos[clase] - int(exactos[clase])),
                    -conteos[clase],
                    clase,
                )
            )
            for clase in sobrantes:
                if restante == 0:
                    break
                cuotas[clase] += 1
                restante -= 1
    return cuotas


def construir_muestra(universo: pd.DataFrame, cuotas: dict[str, int], semilla: int) -> list[dict]:
    """Selecciona los documentos de la muestra y los baraja (semilla fija).

    Args:
        universo: DataFrame con el universo etiquetado.
        cuotas: Dict clase -> cuota.
        semilla: Semilla del generador (42).

    Returns:
        Lista de dicts con id, tipo, ano, area_silver, top3_json, n_citas_cp,
        via, particion e indice (1..N en el orden final barajado).

    Raises:
        RuntimeError: Si una clase no tiene documentos suficientes para su cuota.
    """
    rng = np.random.default_rng(semilla)
    elegidos: list[dict] = []
    for clase in sorted(cuotas):
        sub = universo[universo["area_primaria"] == clase].copy()
        if len(sub) < cuotas[clase]:
            raise RuntimeError(
                f"clase {clase}: cuota {cuotas[clase]} > documentos disponibles {len(sub)}"
            )
        sub["_id_num"] = pd.to_numeric(sub["id"], errors="raise").astype(np.int64)
        sub = sub.sort_values("_id_num", kind="stable").reset_index(drop=True)
        posiciones = np.sort(rng.choice(len(sub), size=int(cuotas[clase]), replace=False))
        for posicion in posiciones:
            fila = sub.iloc[int(posicion)]
            elegidos.append(
                {
                    "id": str(fila["id"]),
                    "tipo": str(fila["tipo"]),
                    "ano": int(fila["ano"]),
                    "area_silver": str(fila["area_primaria"]),
                    "top3_json": str(fila["top3_json"]),
                    "n_citas_cp": int(fila["n_citas_cp"]),
                    "via": str(fila["via"]),
                    "particion": str(fila["particion"]),
                }
            )
    orden = rng.permutation(len(elegidos))
    muestra = [elegidos[int(i)] for i in orden]
    for indice, registro in enumerate(muestra, 1):
        registro["indice"] = indice
    return muestra


def agregar_extractos(
    muestra: list[dict],
    textos: dict[str, str],
    holdout_ids: set[str],
    n_chars: int,
    logger: logging.Logger,
) -> list[dict]:
    """Agrega extracto, longitud y bandera de holdout a cada registro.

    Args:
        muestra: Registros de la muestra (se mutan).
        textos: Dict id -> texto completo del corpus.
        holdout_ids: Set de ids del holdout sellado.
        n_chars: Caracteres del extracto del pack.
        logger: Logger activo.

    Returns:
        La misma lista de registros con extracto, n_chars_texto y en_holdout.
    """
    sin_texto = 0
    for registro in muestra:
        texto = textos.get(registro["id"], "")
        if not texto:
            sin_texto += 1
        registro["en_holdout"] = registro["id"] in holdout_ids
        registro["n_chars_texto"] = len(texto)
        registro["extracto"] = extracto_texto(texto, n_chars)
    if sin_texto:
        logger.warning("%d documentos de la muestra sin texto en el corpus", sin_texto)
    return muestra


def modo_generar(args: argparse.Namespace, logger: logging.Logger, rutas: dict) -> None:
    """Construye la muestra estratificada y la guia de anotacion.

    Args:
        args: Argumentos CLI (smoke, forzar).
        logger: Logger activo.
        rutas: Dict de rutas activas.

    Raises:
        FileExistsError: Si la muestra o la guia existen y no hay --forzar.
    """
    t0 = time.perf_counter()
    destino_muestra = verificar_salida_nueva(rutas["muestra"], args.forzar, args.smoke, logger)
    destino_guia = verificar_salida_nueva(rutas["guia"], args.forzar, args.smoke, logger)
    universo = cargar_universo_etiquetado(logger)
    conteos = universo["area_primaria"].value_counts().to_dict()
    objetivo = CONFIG["n_muestra_smoke"] if args.smoke else CONFIG["n_muestra"]
    minimo = CONFIG["minimo_por_clase_smoke"] if args.smoke else CONFIG["minimo_por_clase"]
    cuotas = asignar_cuotas(conteos, objetivo, minimo)
    logger.info("cuotas de la muestra (objetivo=%d, minimo=%d):", objetivo, minimo)
    for clase in sorted(cuotas):
        logger.info("  %-46s %3d (clase n=%d)", clase, cuotas[clase], conteos[clase])
    muestra = construir_muestra(universo, cuotas, CONFIG["semilla"])
    t_lectura = time.perf_counter()
    dataset = abrir_corpus(rutas["corpus"])
    textos = leer_textos_por_ids(
        dataset, [registro["id"] for registro in muestra], CONFIG["lote_lectura_corpus"], logger
    )
    logger.info(
        "corpus leido por id: %d textos en %.2f s", len(textos), time.perf_counter() - t_lectura
    )
    holdout_ids = cargar_holdout_ids()
    muestra = agregar_extractos(
        muestra, textos, holdout_ids, CONFIG["chars_extracto_pack"], logger
    )
    escribir_jsonl(destino_muestra, muestra)
    escribir_guia(destino_guia, len(muestra), destino_muestra.name, args.smoke, logger)
    particiones = Counter(registro["particion"] for registro in muestra)
    en_holdout = sum(1 for registro in muestra if registro["en_holdout"])
    logger.info(
        "muestra guardada: %s (%d docs, md5 %s)",
        destino_muestra,
        len(muestra),
        md5_archivo(destino_muestra),
    )
    logger.info(
        "composicion: particiones=%s | holdout=%d | clases=%d",
        dict(particiones),
        en_holdout,
        len({registro["area_silver"] for registro in muestra}),
    )
    logger.info("--generar completado en %.1f s", time.perf_counter() - t0)


# ===== SECCION 4: guia de anotacion =====


def texto_guia(n_muestra: int, nombre_muestra: str, smoke: bool) -> str:
    """Construye la guia de anotacion en Markdown (espanol, sin emojis).

    Args:
        n_muestra: Tamano de la muestra.
        nombre_muestra: Nombre del archivo jsonl de la muestra.
        smoke: True si la guia corresponde a la corrida smoke.

    Returns:
        Contenido completo de la guia.
    """
    fecha = pd.Timestamp.now().strftime("%Y-%m-%d")
    menu = "\n".join(
        f"- {indice:2d}. {clase} (alias: {', '.join(ALIAS_CLASES[clase])})"
        for indice, clase in enumerate(CLASES_M4, 1)
    )
    nota_smoke = (
        "\nNOTA: esta guia se genero en modo smoke (30 documentos) y todas sus "
        "salidas viven en /tmp/opencode/o1_smoke/.\n"
        if smoke
        else ""
    )
    return f"""# Guia de anotacion - Validacion humana de etiquetas silver M4

Fecha de generacion: {fecha}
Script: scripts/21_pack_validacion_m4.py (semilla 42)
Muestra: {nombre_muestra} ({n_muestra} documentos)
{nota_smoke}
## 1. Objetivo

Medir la precision real del etiquetado silver de M4. El silver asigna a cada
sentencia el bloque tematico de la Constitucion (CP) al que pertenecen la
mayoria de las citas detectadas (via cita_cp) o, si no hay citas, un termino
lexico de respaldo (via fallback_derecho, menos fiable). La anotacion humana de
una muestra permite estimar la exactitud del silver y decidir si es una base
confiable para entrenar y evaluar.

Esta anotacion es diagnostico: NO entrena nada. Por eso la muestra se tomo del
corpus etiquetado completo (23.654 documentos) e incluye documentos de train,
val, test y del holdout sellado.

## 2. Que es el "bloque constitucional predominante"

La Constitucion esta dividida en 16 bloques tematicos (mapa articulo de la CP
hacia bloque, ADR-1). El bloque predominante de un documento es el bloque al que
pertenecen la mayor cantidad de articulos DISTINTOS de la CP citados en el
documento. Los empates se rompen por el articulo citado de menor numero. Su
tarea es decidir cual es el bloque predominante leyendo el documento.

## 3. Como leer la metadata

- via: cita_cp (etiqueta derivada de citas a la CP, ruta principal) o
  fallback_derecho (sin citas; se uso un termino lexico; menos fiable).
- n_citas_cp: numero de citas a articulos de la CP detectadas en el documento.
- top3: los tres bloques con mas articulos distintos citados y su conteo. Es
  parte del silver: uselo al final, solo para verificar.
- particion: train/val/test del protocolo M4; en_holdout marca documentos del
  holdout sellado (aparecen porque la anotacion no entrena).

## 4. Regla de oro

1. Lea el extracto completo (si necesita mas texto, use --ver ID en otra
   terminal).
2. Decida el bloque predominante SOLO con el texto.
3. Recien entonces compare con el bloque SILVER y registre su respuesta.

No consulte el producto, ni anotaciones previas, ni el silver de otros
documentos antes de decidir. El bloque SILVER se imprime al final del documento
justamente para facilitar esta regla.

## 5. Opciones de respuesta

- Una de las 16 clases del menu (numero o alias):
{menu}
- ambiguo (0): dos o mas bloques compiten sin predominio claro, o el texto no
  alcanza para decidir.
- ninguno (-): ningun bloque constitucional predomina (documento procesal,
  administrativo o sin anclaje tematico claro en la CP).

## 6. Confianza

- alta: el bloque predominante es evidente tras leer el documento.
- media: hay un bloque probable pero con dudas razonables.
- baja: decision al limite entre dos o mas bloques.

Enter = media. La confianza no cambia la respuesta, pero permite analisis
posteriores (por ejemplo, exactitud en el subconjunto de confianza alta).

## 7. Notas

Opcional. Recomendado anotar los articulos o temas que sustentan la decision, o
la razon de "ambiguo"/"ninguno". Enter para omitir.

## 8. Tiempo estimado

1 a 3 minutos por documento ({n_muestra} documentos = 3 a 10 horas). Se puede
anotar en tandas: el progreso se guarda despues de cada respuesta y --anotar
reanuda automaticamente donde quedo.

## 9. Duplicados

Si dos documentos parecen el mismo texto o muy similares, anote ambos: cada id
cuenta. Registre en la nota que es un posible duplicado. No salte documentos por
esta razon.

## 10. Flujo de comandos

    /home/carlos/venv/bin/python scripts/21_pack_validacion_m4.py --resumen
    /home/carlos/venv/bin/python scripts/21_pack_validacion_m4.py --anotar --anotador carlos
    /home/carlos/venv/bin/python scripts/21_pack_validacion_m4.py --ver 12345
    /home/carlos/venv/bin/python scripts/21_pack_validacion_m4.py --exportar-csv
    /home/carlos/venv/bin/python scripts/21_pack_validacion_m4.py --calcular-acuerdo

Durante --anotar: numero o alias para responder, 0 = ambiguo, - = ninguno,
s = saltar el documento (queda pendiente), q = guardar y salir, ? = menu.

## 11. Segundo anotador y acuerdo

Para medir acuerdo inter-anotador, un segundo anotador debe ejecutar --anotar
con otro nombre (por ejemplo --anotador ana) sobre la misma muestra. Luego
--calcular-acuerdo reporta exactitud vs silver por anotador, matriz de confusion
silver -> humano, porcentaje de ambiguo/ninguno, kappa de Cohen por pares y
Fleiss global (sobre documentos respondidos por todos).
"""


def escribir_guia(
    destino: str | Path,
    n_muestra: int,
    nombre_muestra: str,
    smoke: bool,
    logger: logging.Logger,
) -> None:
    """Escribe la guia de anotacion en Markdown.

    Args:
        destino: Ruta de salida.
        n_muestra: Tamano de la muestra.
        nombre_muestra: Nombre del archivo jsonl de la muestra.
        smoke: True si es corrida smoke.
        logger: Logger activo.
    """
    archivo = Path(destino)
    archivo.parent.mkdir(parents=True, exist_ok=True)
    contenido = texto_guia(n_muestra, nombre_muestra, smoke)
    with open(archivo, "w", encoding="utf-8") as manejador:
        manejador.write(contenido)
    logger.info("guia guardada: %s (%d lineas)", archivo, contenido.count("\n") + 1)


# ===== SECCION 5: modos ver y anotar =====


def imprimir_menu() -> None:
    """Imprime el menu de clases y comandos de la anotacion."""
    print("Clases (numero o alias):")
    for indice, clase in enumerate(CLASES_M4, 1):
        print(f"  {indice:2d}  {clase}  [{' / '.join(ALIAS_CLASES[clase])}]")
    print("   0  ambiguo  (varios bloques compiten o no es decidible)")
    print("   -  ninguno  (ningun bloque constitucional predomina)")
    print("   s  saltar este documento | q  guardar y salir | ?  mostrar este menu")


def imprimir_documento(
    doc: dict, texto: str, n_chars: int, indice: int | None = None, total: int | None = None
) -> None:
    """Imprime un documento para revision: metadata, extracto y bloque silver.

    El bloque SILVER se imprime al final (blinding: primero el texto).

    Args:
        doc: Registro del pack (metadata + top3 + area_silver).
        texto: Texto del documento (o extracto del pack como respaldo).
        n_chars: Caracteres a mostrar.
        indice: Posicion 1..N en la sesion (None en --ver).
        total: Total de documentos de la sesion (None en --ver).
    """
    cabecera = f"===== DOC {indice}/{total} | id {doc['id']} =====" if indice else f"===== DOC id {doc['id']} ====="
    print()
    print(cabecera)
    print(
        f"tipo: {doc['tipo']} | ano: {doc['ano']} | particion: {doc['particion']} | "
        f"via: {doc['via']} | n_citas_cp: {doc['n_citas_cp']}"
    )
    if doc.get("en_holdout"):
        print("(documento del holdout sellado; la anotacion no entrena nada)")
    extracto = extracto_texto(texto, n_chars)
    if not extracto:
        print("--- sin extracto disponible en el pack; use --ver ID para el texto completo ---")
    else:
        print(f"--- extracto ({len(extracto)} chars) ---")
        print(extracto)
        print("--- fin del extracto ---")
    print("SILVER (para verificacion; decida primero a partir del texto):")
    print(f"  area_silver: {doc['area_silver']}")
    print(f"  top3 citas CP: {formatear_top3(doc.get('top3_json', ''))}")


def interpretar_bloque(entrada: str) -> tuple[str, str | None, str | None]:
    """Interpreta la respuesta de bloque del anotador.

    Args:
        entrada: Texto crudo del usuario.

    Returns:
        Tupla (accion, valor, mensaje) con accion en
        respuesta | saltar | salir | ayuda | reintentar.
    """
    texto = normalizar_entrada(entrada)
    if texto == "":
        return "reintentar", None, "entrada vacia (use ? para el menu)"
    if texto in ("q", "salir", "quit"):
        return "salir", None, None
    if texto in ("s", "saltar", "skip"):
        return "saltar", None, None
    if texto in ("?", "menu", "ayuda", "help"):
        return "ayuda", None, None
    if texto in ("0", "ambiguo", "amb"):
        return "respuesta", "ambiguo", None
    if texto in ("-", "ninguno", "ning", "ninguna", "nada"):
        return "respuesta", "ninguno", None
    if texto.isdigit():
        numero = int(texto)
        if 1 <= numero <= len(CLASES_M4):
            return "respuesta", CLASES_M4[numero - 1], None
        return "reintentar", None, f"numero fuera de rango: {numero} (use 1-16)"
    if texto in CLASES_M4:
        return "respuesta", texto, None
    for clase, alias in ALIAS_CLASES.items():
        if texto in alias:
            return "respuesta", clase, None
    candidatos = []
    for clase in CLASES_M4:
        tokens = clase.split("_") + list(ALIAS_CLASES[clase])
        if any(token.startswith(texto) for token in tokens):
            candidatos.append(clase)
    if len(candidatos) == 1:
        return "respuesta", candidatos[0], None
    if len(candidatos) > 1:
        return "reintentar", None, "varias clases coinciden: " + ", ".join(candidatos)
    return "reintentar", None, f"no se reconocio la entrada: {entrada!r} (use ? para el menu)"


def pedir_bloque() -> tuple[str, str | None]:
    """Pide el bloque al anotador en bucle hasta una entrada valida.

    Returns:
        Tupla (accion, valor): accion respuesta | saltar | salir.
    """
    while True:
        entrada = input("Bloque (numero/alias, 0=ambiguo, -=ninguno, s=saltar, q=salir, ?=menu): ")
        accion, valor, mensaje = interpretar_bloque(entrada)
        if mensaje:
            print(mensaje)
        if accion == "ayuda":
            imprimir_menu()
            continue
        if accion == "reintentar":
            continue
        return accion, valor


def interpretar_confianza(entrada: str) -> str | None:
    """Interpreta el nivel de confianza (Enter = media).

    Args:
        entrada: Texto crudo del usuario.

    Returns:
        alta | media | baja, o None si la entrada no es valida.
    """
    texto = normalizar_entrada(entrada)
    if texto in ("", "m", "media", "2"):
        return "media"
    if texto in ("a", "alta", "1"):
        return "alta"
    if texto in ("b", "baja", "3"):
        return "baja"
    return None


def pedir_confianza() -> str:
    """Pide el nivel de confianza en bucle hasta una entrada valida.

    Returns:
        alta | media | baja.
    """
    while True:
        entrada = input("Confianza (a)lta / (m)edia / (b)aja [media]: ")
        confianza = interpretar_confianza(entrada)
        if confianza is not None:
            return confianza
        print("Use a, m o b (Enter = media).")


def modo_ver(id_doc: str, logger: logging.Logger, rutas: dict) -> None:
    """Muestra un documento completo por id (metadata + top3 + extracto).

    Args:
        id_doc: Id del documento.
        logger: Logger activo.
        rutas: Dict de rutas activas.

    Raises:
        SystemExit: Si el id no esta en el universo etiquetado.
    """
    t0 = time.perf_counter()
    id_doc = str(id_doc).strip()
    etiquetas = pq.read_table(
        CONFIG["rutas_datos"]["etiquetas"], filters=[("id", "==", id_doc)]
    ).to_pandas()
    if etiquetas.empty:
        logger.error("el id %s no esta en %s", id_doc, CONFIG["rutas_datos"]["etiquetas"])
        raise SystemExit(1)
    splits = pq.read_table(
        CONFIG["rutas_datos"]["splits"], filters=[("id", "==", id_doc)]
    ).to_pandas()
    particion = str(splits["particion"].iloc[0]) if not splits.empty else "desconocida"
    fila = etiquetas.iloc[0]
    doc = {
        "id": id_doc,
        "tipo": str(fila["tipo"]),
        "ano": int(fila["ano"]),
        "particion": particion,
        "area_silver": str(fila["area_primaria"]),
        "top3_json": str(fila["top3_json"]),
        "n_citas_cp": int(fila["n_citas_cp"]),
        "via": str(fila["via"]),
    }
    ruta_muestra = Path(rutas["muestra"])
    if ruta_muestra.exists():
        muestra = cargar_muestra(rutas)
        if id_doc not in {registro["id"] for registro in muestra}:
            logger.warning("el id %s no pertenece a la muestra activa (%s)", id_doc, ruta_muestra)
    dataset = abrir_corpus(rutas["corpus"])
    textos = leer_textos_por_ids(dataset, [id_doc], CONFIG["lote_lectura_corpus"], logger)
    imprimir_documento(doc, textos.get(id_doc, ""), CONFIG["chars_extracto_ver"])
    logger.info("--ver id=%s completado en %.2f s", id_doc, time.perf_counter() - t0)


def modo_anotar(nombre: str, logger: logging.Logger, rutas: dict) -> None:
    """Recorrido interactivo reanudable de anotacion del pack.

    Args:
        nombre: Nombre del anotador (se normaliza para el archivo).
        logger: Logger activo.
        rutas: Dict de rutas activas.

    Raises:
        FileNotFoundError: Si no existe la muestra (hay que ejecutar --generar).
    """
    t0 = time.perf_counter()
    anotador = normalizar_anotador(nombre)
    muestra = cargar_muestra(rutas)
    ruta_respuestas = Path(rutas["respuestas"].format(anotador=anotador))
    respondidos = cargar_respuestas(ruta_respuestas)
    pendientes = [doc for doc in muestra if doc["id"] not in respondidos]
    logger.info(
        "anotador=%s | respondidos=%d/%d | pendientes=%d | archivo=%s",
        anotador,
        len(respondidos),
        len(muestra),
        len(pendientes),
        ruta_respuestas,
    )
    if not pendientes:
        logger.info("no hay documentos pendientes: anotacion completa")
        return
    imprimir_menu()
    dataset = abrir_corpus(rutas["corpus"])
    nuevos = 0
    try:
        for posicion, doc in enumerate(pendientes, 1):
            textos = leer_textos_por_ids(
                dataset, [doc["id"]], CONFIG["lote_lectura_corpus"], logger
            )
            texto = textos.get(doc["id"]) or doc.get("extracto", "")
            t_doc = time.perf_counter()
            imprimir_documento(
                doc, texto, CONFIG["chars_extracto_ver"], indice=posicion, total=len(pendientes)
            )
            accion, respuesta = pedir_bloque()
            if accion == "salir":
                break
            if accion == "saltar":
                logger.info("documento %s saltado (queda pendiente)", doc["id"])
                continue
            confianza = pedir_confianza()
            nota = input("Nota (Enter para omitir): ").strip()
            registro = {
                "anotador": anotador,
                "id": doc["id"],
                "indice": int(doc.get("indice", 0)),
                "respuesta": respuesta,
                "confianza": confianza,
                "nota": nota,
                "fecha": pd.Timestamp.now().isoformat(timespec="seconds"),
                "tiempo_seg": round(time.perf_counter() - t_doc, 1),
            }
            escribir_jsonl_append(ruta_respuestas, registro)
            nuevos += 1
            logger.info(
                "[%d/%d] id=%s -> %s (%s) en %.1f s",
                len(respondidos) + nuevos,
                len(muestra),
                doc["id"],
                respuesta,
                confianza,
                time.perf_counter() - t_doc,
            )
    except EOFError:
        logger.warning("entrada terminada (EOF): se conserva lo respondido")
    except KeyboardInterrupt:
        logger.warning("interrumpido por el usuario: se conserva lo respondido")
    logger.info(
        "sesion: %d respuestas nuevas | total=%d/%d | faltantes=%d | %.1f s",
        nuevos,
        len(respondidos) + nuevos,
        len(muestra),
        len(muestra) - len(respondidos) - nuevos,
        time.perf_counter() - t0,
    )


# ===== SECCION 6: modos resumen y exportar =====


def modo_resumen(logger: logging.Logger, rutas: dict) -> None:
    """Imprime el estado de la anotacion (respondidos, faltantes, distribucion).

    Args:
        logger: Logger activo.
        rutas: Dict de rutas activas.
    """
    t0 = time.perf_counter()
    muestra = cargar_muestra(rutas)
    por_id = {doc["id"]: doc for doc in muestra}
    respuestas = cargar_todas_respuestas(rutas)
    particiones = Counter(doc["particion"] for doc in muestra)
    print(f"Muestra: {len(muestra)} documentos | {rutas['muestra']}")
    print(f"Particiones: {dict(particiones)} | holdout: {sum(1 for d in muestra if d.get('en_holdout'))}")
    print(f"Anotadores con archivo de respuestas: {len(respuestas)}")
    if not respuestas:
        print("Todavia no hay respuestas. Ejecute --anotar --anotador NOMBRE")
    for anotador in sorted(respuestas):
        registros = respuestas[anotador]
        validos = [registro for id_doc, registro in registros.items() if id_doc in por_id]
        huerfanos = len(registros) - len(validos)
        faltantes = len(muestra) - len(validos)
        distribucion = Counter(str(registro.get("respuesta", "?")) for registro in validos)
        ambiguo = distribucion.get("ambiguo", 0)
        ninguno = distribucion.get("ninguno", 0)
        porcentaje_ambiguo = 100.0 * ambiguo / len(validos) if validos else 0.0
        print(
            f"\n[{anotador}] respondidos {len(validos)}/{len(muestra)} | faltantes {faltantes}"
            f" | ambiguo {ambiguo} ({porcentaje_ambiguo:.1f} %) | ninguno {ninguno}"
        )
        if huerfanos:
            print(f"  aviso: {huerfanos} respuestas de ids fuera de la muestra actual")
        for respuesta, conteo in distribucion.most_common():
            print(f"    {respuesta}: {conteo}")
    respondidos_global = sum(
        1 for id_doc in por_id if any(id_doc in registros for registros in respuestas.values())
    )
    print(f"\nDocumentos con al menos una respuesta: {respondidos_global}/{len(muestra)}")
    logger.info("--resumen completado en %.2f s", time.perf_counter() - t0)


def modo_exportar_csv(
    forzar: bool, logger: logging.Logger, rutas: dict
) -> None:
    """Exporta las respuestas a CSV plano (una fila por anotador y documento).

    Args:
        forzar: True permite reescribir el CSV existente.
        logger: Logger activo.
        rutas: Dict de rutas activas.

    Raises:
        SystemExit: Si no hay respuestas que exportar.
    """
    t0 = time.perf_counter()
    muestra = cargar_muestra(rutas)
    por_id = {doc["id"]: doc for doc in muestra}
    respuestas = cargar_todas_respuestas(rutas)
    filas = []
    for anotador in sorted(respuestas):
        for id_doc, registro in respuestas[anotador].items():
            meta = por_id.get(id_doc)
            if meta is None:
                continue
            filas.append(
                {
                    "anotador": anotador,
                    "id": id_doc,
                    "area_silver": meta["area_silver"],
                    "respuesta": registro.get("respuesta"),
                    "confianza": registro.get("confianza"),
                    "nota": registro.get("nota", ""),
                    "particion": meta["particion"],
                    "_indice": int(meta.get("indice", 0)),
                }
            )
    if not filas:
        logger.error("no hay respuestas que coincidan con la muestra; ejecute --anotar primero")
        raise SystemExit(1)
    destino = verificar_salida_nueva(rutas["export_csv"], forzar, False, logger)
    destino.parent.mkdir(parents=True, exist_ok=True)
    columnas = ["anotador", "id", "area_silver", "respuesta", "confianza", "nota", "particion"]
    tabla = pd.DataFrame(filas).sort_values(["anotador", "_indice"])[columnas]
    tabla.to_csv(destino, index=False, encoding="utf-8")
    logger.info("CSV exportado: %s (%d filas) en %.2f s", destino, len(tabla), time.perf_counter() - t0)


# ===== SECCION 7: acuerdo humano vs silver =====


def nivel_kappa(kappa: float | None) -> str | None:
    """Interpreta el kappa segun Landis y Koch (1977).

    Args:
        kappa: Valor de kappa (None si no es calculable).

    Returns:
        Nivel textual, o None.
    """
    if kappa is None:
        return None
    if kappa < 0.0:
        return "pobre"
    if kappa < 0.21:
        return "leve"
    if kappa < 0.41:
        return "aceptable"
    if kappa < 0.61:
        return "moderada"
    if kappa < 0.81:
        return "sustancial"
    return "casi perfecta"


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> dict:
    """Calcula el kappa de Cohen sobre dos listas alineadas de etiquetas.

    Args:
        labels_a: Etiquetas del primer anotador (o del silver).
        labels_b: Etiquetas del segundo anotador (misma longitud).

    Returns:
        Dict con n, po (acuerdo observado), pe (esperado), kappa y nivel.

    Raises:
        ValueError: Si las listas tienen longitudes distintas.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("las listas de etiquetas deben tener la misma longitud")
    n = len(labels_a)
    if n == 0:
        return {"n": 0, "po": None, "pe": None, "kappa": None, "nivel": None}
    categorias = sorted(set(labels_a) | set(labels_b))
    po = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    conteo_a = Counter(labels_a)
    conteo_b = Counter(labels_b)
    pe = sum((conteo_a[categoria] / n) * (conteo_b[categoria] / n) for categoria in categorias)
    if abs(1.0 - pe) < 1e-12:
        kappa = None
    else:
        kappa = (po - pe) / (1.0 - pe)
    return {
        "n": n,
        "po": float(po),
        "pe": float(pe),
        "kappa": (float(kappa) if kappa is not None else None),
        "nivel": nivel_kappa(kappa),
    }


def fleiss_kappa(items: list[list[str]], categorias: list[str]) -> dict | None:
    """Calcula el kappa de Fleiss sobre items con el mismo numero de evaluadores.

    Args:
        items: Lista de items; cada item es la lista de etiquetas de TODOS los
            evaluadores (misma longitud en todos los items).
        categorias: Categorias posibles (incluye las observadas).

    Returns:
        Dict con kappa, p_bar, p_e, n_items y n_evaluadores; o None si no hay
        items o hay menos de 2 evaluadores.

    Raises:
        ValueError: Si los items no tienen todos el mismo numero de etiquetas.
    """
    if not items:
        return None
    n_evaluadores = len(items[0])
    if n_evaluadores < 2:
        return None
    if any(len(item) != n_evaluadores for item in items):
        raise ValueError("todos los items deben tener el mismo numero de evaluadores")
    n_items = len(items)
    p_i = []
    for item in items:
        conteo = Counter(item)
        suma_cuadrados = sum(valor * valor for valor in conteo.values())
        p_i.append((suma_cuadrados - n_evaluadores) / (n_evaluadores * (n_evaluadores - 1)))
    p_bar = sum(p_i) / n_items
    total = n_items * n_evaluadores
    p_j = {
        categoria: sum(1 for item in items for etiqueta in item if etiqueta == categoria) / total
        for categoria in categorias
    }
    p_e = sum(valor * valor for valor in p_j.values())
    if abs(1.0 - p_e) < 1e-12:
        kappa = None
    else:
        kappa = (p_bar - p_e) / (1.0 - p_e)
    return {
        "n_items": n_items,
        "n_evaluadores": n_evaluadores,
        "p_bar": float(p_bar),
        "p_e": float(p_e),
        "kappa": (float(kappa) if kappa is not None else None),
        "nivel": nivel_kappa(kappa),
    }


def evaluar_anotador(registros: dict[str, dict], por_id: dict[str, dict]) -> dict:
    """Evalua las respuestas de un anotador (o del pool) contra el silver.

    Args:
        registros: Dict id -> registro de respuesta.
        por_id: Dict id -> registro de la muestra (con area_silver, via, ...).

    Returns:
        Dict con cobertura, exactitudes (validas y con especiales como error),
        porcentajes de ambiguo/ninguno, desglose por clase silver, por via,
        distribucion de confianza, matriz de confusion y kappa vs silver.
    """
    respondidos = [id_doc for id_doc in por_id if id_doc in registros]
    n = len(respondidos)
    validas = []
    for id_doc in respondidos:
        respuesta = str(registros[id_doc].get("respuesta", ""))
        if respuesta in CLASES_M4:
            validas.append((por_id[id_doc]["area_silver"], respuesta))
    n_validas = len(validas)
    exactas = sum(1 for silver, respuesta in validas if silver == respuesta)
    n_ambiguo = sum(1 for id_doc in respondidos if str(registros[id_doc].get("respuesta")) == "ambiguo")
    n_ninguno = sum(1 for id_doc in respondidos if str(registros[id_doc].get("respuesta")) == "ninguno")
    n_otras = n - n_validas - n_ambiguo - n_ninguno
    columnas = CLASES_M4 + list(RESPUESTAS_ESPECIALES)
    matriz = {clase: {columna: 0 for columna in columnas} for clase in CLASES_M4}
    for id_doc in respondidos:
        respuesta = str(registros[id_doc].get("respuesta", ""))
        silver = por_id[id_doc]["area_silver"]
        if respuesta in matriz.get(silver, {}):
            matriz[silver][respuesta] += 1
    por_clase = {}
    for clase in CLASES_M4:
        fila = matriz[clase]
        n_clase = sum(fila.values())
        exactas_clase = fila[clase]
        especiales_clase = fila["ambiguo"] + fila["ninguno"]
        por_clase[clase] = {
            "n_respondidos": n_clase,
            "n_exactas": exactas_clase,
            "exactitud": (exactas_clase / (n_clase - especiales_clase))
            if (n_clase - especiales_clase) > 0
            else None,
            "n_ambiguo": fila["ambiguo"],
            "n_ninguno": fila["ninguno"],
        }
    por_via = {}
    for via in sorted({por_id[id_doc]["via"] for id_doc in respondidos}):
        ids_via = [id_doc for id_doc in respondidos if por_id[id_doc]["via"] == via]
        validas_via = [
            (por_id[id_doc]["area_silver"], str(registros[id_doc].get("respuesta", "")))
            for id_doc in ids_via
            if str(registros[id_doc].get("respuesta", "")) in CLASES_M4
        ]
        exactas_via = sum(1 for silver, respuesta in validas_via if silver == respuesta)
        por_via[via] = {
            "n_respondidos": len(ids_via),
            "n_validas": len(validas_via),
            "exactitud": (exactas_via / len(validas_via)) if validas_via else None,
        }
    kappa_silver = cohen_kappa(
        [silver for silver, _ in validas], [respuesta for _, respuesta in validas]
    )
    return {
        "n_respondidos": n,
        "n_validas": n_validas,
        "n_ambiguo": n_ambiguo,
        "n_ninguno": n_ninguno,
        "n_otras": n_otras,
        "cobertura_muestra": (n / len(por_id)) if por_id else None,
        "exactitud_validas": (exactas / n_validas) if n_validas else None,
        "exactitud_todas": (exactas / n) if n else None,
        "pct_ambiguo": (n_ambiguo / n) if n else None,
        "pct_ninguno": (n_ninguno / n) if n else None,
        "confianza": dict(Counter(str(registros[id_doc].get("confianza", "?")) for id_doc in respondidos)),
        "por_clase_silver": por_clase,
        "por_via": por_via,
        "matriz_confusion_silver_humano": matriz,
        "kappa_vs_silver": kappa_silver,
    }


def calcular_acuerdo(
    respuestas: dict[str, dict[str, dict]], por_id: dict[str, dict], logger: logging.Logger
) -> dict:
    """Calcula el acuerdo humano vs silver y entre anotadores.

    Args:
        respuestas: Dict anotador -> (id -> registro).
        por_id: Dict id -> registro de la muestra.
        logger: Logger activo.

    Returns:
        Dict con por_anotador, global_pooled, kappas por pares y Fleiss.
    """
    por_anotador = {
        anotador: evaluar_anotador(registros, por_id)
        for anotador, registros in sorted(respuestas.items())
    }
    pooled: dict[str, dict] = {}
    for anotador in sorted(respuestas):
        for id_doc, registro in respuestas[anotador].items():
            if id_doc in por_id:
                pooled[id_doc] = registro
    global_pooled = evaluar_anotador(pooled, por_id)
    pares = {}
    anotadores = sorted(respuestas)
    for i in range(len(anotadores)):
        for j in range(i + 1, len(anotadores)):
            anotador_a, anotador_b = anotadores[i], anotadores[j]
            comunes = [
                id_doc
                for id_doc in respuestas[anotador_a]
                if id_doc in respuestas[anotador_b] and id_doc in por_id
            ]
            comunes.sort(key=int)
            kappa = cohen_kappa(
                [str(respuestas[anotador_a][id_doc]["respuesta"]) for id_doc in comunes],
                [str(respuestas[anotador_b][id_doc]["respuesta"]) for id_doc in comunes],
            )
            pares[f"{anotador_a}|{anotador_b}"] = {
                "n_comunes": len(comunes),
                "po": kappa["po"],
                "kappa": kappa["kappa"],
                "nivel": kappa["nivel"],
            }
    fleiss = None
    if len(anotadores) >= 2:
        comunes_todos = [
            id_doc
            for id_doc in por_id
            if all(id_doc in respuestas[anotador] for anotador in anotadores)
        ]
        comunes_todos.sort(key=int)
        if comunes_todos:
            categorias = CLASES_M4 + list(RESPUESTAS_ESPECIALES)
            items = [
                [str(respuestas[anotador][id_doc]["respuesta"]) for anotador in anotadores]
                for id_doc in comunes_todos
            ]
            fleiss = fleiss_kappa(items, categorias)
            if fleiss is not None:
                fleiss["ids_completos"] = len(comunes_todos)
        else:
            logger.warning("Fleiss no calculable: ningun id respondido por TODOS los anotadores")
    return {
        "por_anotador": por_anotador,
        "global_pooled": global_pooled,
        "kappas_pares": pares,
        "fleiss": fleiss,
    }


def formatear_numero(valor: float | None, decimales: int = 3) -> str:
    """Formatea un numero opcional para consola.

    Args:
        valor: Numero o None.
        decimales: Decimales a mostrar.

    Returns:
        Texto del numero o "n/d".
    """
    if valor is None:
        return "n/d"
    return f"{valor:.{decimales}f}"


def modo_calcular_acuerdo(
    forzar: bool, logger: logging.Logger, rutas: dict
) -> None:
    """Calcula y persiste el acuerdo humano vs silver.

    Args:
        forzar: True permite reescribir acuerdo_v1.json.
        logger: Logger activo.
        rutas: Dict de rutas activas.

    Raises:
        SystemExit: Si no hay respuestas para calcular.
    """
    t0 = time.perf_counter()
    muestra = cargar_muestra(rutas)
    por_id = {doc["id"]: doc for doc in muestra}
    respuestas = cargar_todas_respuestas(rutas)
    if not respuestas:
        logger.error("no hay respuestas: ejecute --anotar antes de --calcular-acuerdo")
        raise SystemExit(1)
    resultado = calcular_acuerdo(respuestas, por_id, logger)
    resultado["meta"] = {
        "fecha": pd.Timestamp.now().isoformat(timespec="seconds"),
        "script": "scripts/21_pack_validacion_m4.py",
        "semilla_muestreo": CONFIG["semilla"],
        "n_muestra": len(muestra),
        "muestra": str(rutas["muestra"]),
        "md5_muestra": md5_archivo(rutas["muestra"]),
        "md5_respuestas": {
            anotador: md5_archivo(Path(rutas["muestra"]).parent / f"respuestas_{anotador}.jsonl")
            for anotador in sorted(respuestas)
        },
        "anotadores": sorted(respuestas),
        "versiones": versiones_entorno(),
        "premisas": PREMISAS,
    }
    destino = verificar_salida_nueva(rutas["acuerdo"], forzar, False, logger)
    destino.parent.mkdir(parents=True, exist_ok=True)
    with open(destino, "w", encoding="utf-8") as archivo:
        json.dump(resultado, archivo, ensure_ascii=False, indent=2, default=str)
    logger.info("acuerdo guardado: %s (md5 %s)", destino, md5_archivo(destino))
    logger.info("acuerdo humano vs silver (n muestra=%d):", len(muestra))
    for anotador in sorted(resultado["por_anotador"]):
        evaluacion = resultado["por_anotador"][anotador]
        logger.info(
            "%s: respondidos=%d/%d | exactitud(validas)=%s | con especiales=error=%s | "
            "ambiguo=%.1f %% | kappa vs silver=%s (%s)",
            anotador,
            evaluacion["n_respondidos"],
            len(muestra),
            formatear_numero(evaluacion["exactitud_validas"]),
            formatear_numero(evaluacion["exactitud_todas"]),
            100.0 * (evaluacion["pct_ambiguo"] or 0.0),
            formatear_numero(evaluacion["kappa_vs_silver"]["kappa"]),
            evaluacion["kappa_vs_silver"]["nivel"],
        )
    for par, datos in resultado["kappas_pares"].items():
        logger.info(
            "kappa par %s: %s (%s) sobre n=%d",
            par,
            formatear_numero(datos["kappa"]),
            datos["nivel"],
            datos["n_comunes"],
        )
    if resultado["fleiss"] is not None:
        logger.info(
            "Fleiss global: %s (%s) sobre %d items y %d evaluadores",
            formatear_numero(resultado["fleiss"]["kappa"]),
            resultado["fleiss"]["nivel"],
            resultado["fleiss"]["n_items"],
            resultado["fleiss"]["n_evaluadores"],
        )
    logger.info("--calcular-acuerdo completado en %.1f s", time.perf_counter() - t0)


# ===== SECCION 8: CLI y main =====


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define y parsea los argumentos de linea de comandos.

    Args:
        argv: Lista de argumentos (None usa sys.argv).

    Returns:
        Namespace con el modo pedido, anotador, smoke y forzar.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Pack de validacion humana muestral de las etiquetas silver M4 "
            "(bloque constitucional predominante). No entrena nada."
        )
    )
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--generar", action="store_true", help="construye muestra_200.jsonl + guia_anotacion.md")
    grupo.add_argument("--ver", metavar="ID", default=None, help="muestra un documento (metadata + top3 + extracto 3000)")
    grupo.add_argument("--anotar", action="store_true", help="recorrido interactivo reanudable (requiere --anotador)")
    grupo.add_argument("--resumen", action="store_true", help="estado de la anotacion")
    grupo.add_argument("--exportar-csv", action="store_true", dest="exportar_csv", help="CSV plano para revision")
    grupo.add_argument("--calcular-acuerdo", action="store_true", dest="calcular_acuerdo", help="exactitud vs silver, matriz, kappa y Fleiss")
    parser.add_argument("--anotador", default=None, help="nombre del anotador (requerido con --anotar)")
    parser.add_argument("--smoke", action="store_true", help="30 docs y todo en /tmp/opencode/o1_smoke")
    parser.add_argument("--forzar", action="store_true", help="permite reescribir salidas existentes")
    args = parser.parse_args(argv)
    if args.anotar and not args.anotador:
        parser.error("--anotar requiere --anotador NOMBRE")
    return args


def nombre_modo(args: argparse.Namespace) -> str:
    """Devuelve el nombre legible del modo activo.

    Args:
        args: Argumentos CLI.

    Returns:
        Nombre del modo (generar, ver, anotar, resumen, exportar_csv,
        calcular_acuerdo).
    """
    if args.generar:
        return "generar"
    if args.ver is not None:
        return "ver"
    if args.anotar:
        return "anotar"
    if args.resumen:
        return "resumen"
    if args.exportar_csv:
        return "exportar_csv"
    return "calcular_acuerdo"


def main(argv: list[str] | None = None) -> None:
    """Orquesta el modo pedido del pack de validacion.

    Args:
        argv: Argumentos CLI (None usa sys.argv).
    """
    args = parse_args(argv)
    os.chdir(RAIZ)
    logger = configurar_logger()
    random.seed(CONFIG["semilla"])
    np.random.seed(CONFIG["semilla"])
    rutas = rutas_activas(args.smoke)
    logger.info(
        "=== INICIO pack validacion M4 | modo=%s smoke=%s forzar=%s ===",
        nombre_modo(args),
        args.smoke,
        args.forzar,
    )
    if args.smoke:
        Path(CONFIG["dir_smoke"]).mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    if args.generar:
        modo_generar(args, logger, rutas)
    elif args.ver is not None:
        modo_ver(args.ver, logger, rutas)
    elif args.anotar:
        modo_anotar(args.anotador, logger, rutas)
    elif args.resumen:
        modo_resumen(logger, rutas)
    elif args.exportar_csv:
        modo_exportar_csv(args.forzar, logger, rutas)
    elif args.calcular_acuerdo:
        modo_calcular_acuerdo(args.forzar, logger, rutas)
    logger.info("=== FIN pack validacion M4 en %.1f s ===", time.perf_counter() - t0)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - protocolo: traza completa y salida 1 ante cualquier fallo
        traceback.print_exc()
        sys.exit(1)
