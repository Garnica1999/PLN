"""00_csv_a_parquet.py
Fecha: 2026-09-08
Descripcion: Migra el corpus Juris1992_2022.csv a Parquet particionado por tipo y ano.
Estado del arte aplicado: Limpieza E1 exacta portada del notebook primera_entrega.ipynb
    (minusculas, proteccion de la enie, NFD sin diacriticos, filtro [a-z], split y
    filtro de tokens de 1 letra no gramaticales mas abreviatura art).
"""

import csv
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    import psutil
except ImportError:
    psutil = None

CONFIG = {
    "semilla": 42,
    "ruta_csv": "/home/carlos/Git/PLN/PROYECTO/PRIMERA ENTREGA/data/Juris1992_2022.csv",
    "ruta_metadatos_e1": (
        "/home/carlos/Git/PLN/PROYECTO/PRIMERA ENTREGA/data/procesado/df_metadatos.csv"
    ),
    "ruta_parquet": "data/corpus_sentencias.parquet",
    "ruta_metricas": "resultados/metricas_formato.json",
    "ruta_log": "experimentos/migracion_parquet.log",
    "limite_campo_csv": 100_000_000,
    "n_muestra_esquema": 3,
    "spark_app": "juris-csv-a-parquet",
    "spark_master": "local[*]",
    "spark_driver_memory": "8g",
    "java17_home": "/usr/lib/jvm/java-17-openjdk",
    "columnas_parquet": ["id", "tipo", "numero", "ano", "texto", "texto_limpio"],
}

PLACEHOLDER_N = "\u0001"
PREFIJO_REGEX = re.compile(r"^(?:sentencia\s+)?([cts])(u)?\s*(\d{1,4})\s+(\d{2,4})")
MAPA_TIPO = {"t": "T", "c": "C", "s": "S"}
ABREV_CITA = {"art"}


def configurar_logging(ruta_log: str) -> logging.Logger:
    """Configura el logger con salida a consola y a archivo.

    Args:
        ruta_log: Ruta del archivo de log (se crea su carpeta padre).

    Returns:
        Logger configurado con dos handlers (consola y archivo).
    """
    Path(ruta_log).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("migracion_parquet")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formato = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    consola = logging.StreamHandler(sys.stdout)
    consola.setFormatter(formato)
    archivo = logging.FileHandler(ruta_log, mode="w", encoding="utf-8")
    archivo.setFormatter(formato)
    logger.addHandler(consola)
    logger.addHandler(archivo)
    return logger


def normalizar(texto: str, con_digitos: bool) -> str:
    """Normaliza un texto: minusculas, sin diacriticos y solo letras (y digitos).

    Portada verbatim de la celda de limpieza del notebook de la primera entrega:
    minusculas; proteccion de la enie antes de descomponer en NFD; eliminacion de
    marcas de acentuacion (rango Unicode U+0300-U+036F); restauracion de la enie;
    filtro de caracteres (letras a-z y enie, con o sin digitos) y colapso de espacios.

    Args:
        texto: Cadena original.
        con_digitos: True conserva digitos (necesario para parsear la clave);
            False los elimina (version final para el analisis lexico).

    Returns:
        Cadena normalizada sin espacios multiples ni espacios extremos.
    """
    s = texto.lower().replace("\u00f1", PLACEHOLDER_N)
    s = unicodedata.normalize("NFD", s)
    s = re.sub(r"[\u0300-\u036f]", "", s).replace(PLACEHOLDER_N, "\u00f1")
    patron = r"[^a-z\u00f10-9\s]" if con_digitos else r"[^a-z\u00f1\s]"
    s = re.sub(patron, " ", s)
    return re.sub(r"\s+", " ", s).strip()


def limpiar_texto(texto_normalizado: str) -> str:
    """Produce la version limpia final (sin digitos) del texto normalizado.

    Portada verbatim de la primera entrega: el texto normalizado con digitos solo
    contiene letras, digitos y espacios; basta retirar los digitos y colapsar los
    espacios resultantes para obtener la version de analisis.

    Args:
        texto_normalizado: Salida de normalizar(texto, con_digitos=True).

    Returns:
        Texto con unicamente letras [a-z] y enie, sin digitos.
    """
    return re.sub(r"\s+", " ", re.sub(r"\d", " ", texto_normalizado)).strip()


def interpretar_annio(digitos_annio: str) -> int:
    """Convierte los digitos del anio de la clave de relatoria a anio calendario.

    Portada verbatim de la primera entrega: con 4 digitos se usa el valor directo;
    con 2 digitos se aplica la regla 90-99 -> 19xx y 00-21 -> 20xx (periodo real
    del corpus: 1992-2021).

    Args:
        digitos_annio: Cadena de 2 o 4 digitos capturada por la expresion regular.

    Returns:
        Anio como entero.

    Raises:
        ValueError: Si los digitos no son convertibles a entero.
    """
    if len(digitos_annio) == 4:
        return int(digitos_annio)
    valor = int(digitos_annio)
    return 1900 + valor if 90 <= valor <= 99 else 2000 + valor


def parsear_prefijo(texto_normalizado: str) -> tuple | None:
    """Extrae (tipo, numero, anio) de la clave de relatoria al inicio del texto.

    Portada verbatim de la primera entrega: aplica el regex de prefijo sobre el
    texto normalizado con digitos y mapea t/c/s+u a T/C/SU.

    Args:
        texto_normalizado: Texto normalizado con digitos.

    Returns:
        Tupla (tipo, numero, anio) si el texto inicia con una clave valida de
        sentencia, o None si no la tiene (documento no sentencia).
    """
    coincidencia = PREFIJO_REGEX.match(texto_normalizado)
    if coincidencia is None:
        return None
    if coincidencia.group(2):
        tipo = "SU"
    else:
        tipo = MAPA_TIPO[coincidencia.group(1)]
    numero = int(coincidencia.group(3))
    annio = interpretar_annio(coincidencia.group(4))
    return tipo, numero, annio


def construir_stopwords_gramaticales() -> set:
    """Construye el conjunto de stopwords gramaticales normalizado como el texto.

    Portada verbatim de la primera entrega: normaliza la lista NLTK en espanol
    con la misma transformacion del corpus y descarta formas no alfabeticas. Las
    colisiones colapsan en un unico termino (313 originales -> 306 efectivas).

    Returns:
        Set con las stopwords gramaticales normalizadas.
    """
    from nltk.corpus import stopwords

    conjunto = set()
    for palabra in stopwords.words("spanish"):
        normalizada = normalizar(palabra, con_digitos=False)
        if re.fullmatch(r"[a-z\u00f1]+", normalizada):
            conjunto.add(normalizada)
    return conjunto


def filtrar_tokens(tokens: list, stopw_gramaticales: set) -> list:
    """Filtra tokens de ruido antes de contabilizar (portada de la primera entrega).

    Descarta los tokens de un solo caracter que no sean stopwords gramaticales
    (restos de claves T-001/21 -> t y citas de pagina p. 45 -> p) y la abreviatura
    de cita art. Las palabras funcionales de una letra que si estan en las
    stopwords (a, y, o, e) se conservan.

    Args:
        tokens: Lista de tokens del texto limpio.
        stopw_gramaticales: Set de stopwords gramaticales normalizadas.

    Returns:
        Lista de tokens conservados.
    """
    conservados = []
    for token in tokens:
        if token in ABREV_CITA:
            continue
        if len(token) == 1 and token not in stopw_gramaticales:
            continue
        conservados.append(token)
    return conservados


def construir_texto_limpio(texto_crudo: str, stopw_gramaticales: set) -> str:
    """Aplica la cadena completa de limpieza E1 y devuelve el texto limpio final.

    Cadena exacta E1: normalizar(con_digitos=True) -> limpiar_texto ->
    split() -> filtrar_tokens -> join con un espacio.

    Args:
        texto_crudo: Texto original de la sentencia.
        stopw_gramaticales: Set de stopwords gramaticales normalizadas.

    Returns:
        Texto limpio final (tokens filtrados unidos por espacios).
    """
    normalizado = normalizar(texto_crudo, con_digitos=True)
    limpio = limpiar_texto(normalizado)
    tokens = filtrar_tokens(limpio.split(), stopw_gramaticales)
    return " ".join(tokens)


def confirmar_esquema_csv(
    ruta_csv: str, n_muestra: int, logger: logging.Logger
) -> list:
    """Confirma el esquema del CSV leyendo el encabezado y las primeras filas.

    Args:
        ruta_csv: Ruta del CSV de entrada.
        n_muestra: Numero de filas de muestra a mostrar.
        logger: Logger de ejecucion.

    Returns:
        Encabezado del CSV como lista de columnas.

    Raises:
        FileNotFoundError: Si el CSV no existe.
        ValueError: Si el encabezado no es el esperado ['', 'Texto'].
    """
    csv.field_size_limit(CONFIG["limite_campo_csv"])
    with open(ruta_csv, "r", encoding="utf-8", newline="") as f:
        lector = csv.reader(f)
        encabezado = next(lector)
        logger.info("Esquema CSV confirmado: %s", encabezado)
        if encabezado != ["", "Texto"]:
            raise ValueError(f"Encabezado inesperado: {encabezado}")
        for i in range(n_muestra):
            fila = next(lector)
            logger.info(
                "Fila %d: id=%s ncols=%d inicio=%s",
                i,
                fila[0][:20],
                len(fila),
                fila[1][:120].replace("\n", " "),
            )
    return encabezado


def procesar_csv(
    ruta_csv: str, stopw_gramaticales: set, logger: logging.Logger
) -> tuple:
    """Recorre el CSV en streaming, parsea metadatos y aplica la limpieza E1.

    Replica la pasada completa de la primera entrega: por cada fila normaliza con
    digitos, parsea el prefijo (tipo, numero, anio) y, si es sentencia, genera el
    texto limpio. Las filas sin clave valida se registran como excluidas.

    Args:
        ruta_csv: Ruta del CSV de entrada.
        stopw_gramaticales: Set de stopwords gramaticales normalizadas.
        logger: Logger de ejecucion.

    Returns:
        Tupla (registros, excluidas, n_filas) donde registros es la lista de
        dicts con claves id/tipo/numero/ano/texto/texto_limpio, excluidas es la
        lista de tuplas (posicion_csv, comienzo_texto) y n_filas el total leido.
    """
    csv.field_size_limit(CONFIG["limite_campo_csv"])
    registros = []
    excluidas = []
    n_filas = 0
    inicio = time.time()
    with open(ruta_csv, "r", encoding="utf-8", newline="") as f:
        lector = csv.reader(f)
        next(lector)
        for fila in lector:
            if n_filas % 5000 == 0 and n_filas > 0:
                logger.info(
                    "Filas procesadas: %d (%.0f s acumulados)",
                    n_filas,
                    time.time() - inicio,
                )
            texto_crudo = fila[1]
            normalizado = normalizar(texto_crudo, con_digitos=True)
            metadatos = parsear_prefijo(normalizado)
            if metadatos is None:
                excluidas.append((n_filas, texto_crudo[:140].replace("\n", " ")))
                n_filas += 1
                continue
            tipo, numero, annio = metadatos
            texto_limpio = construir_texto_limpio(texto_crudo, stopw_gramaticales)
            registros.append(
                {
                    "id": str(fila[0]),
                    "tipo": tipo,
                    "numero": int(numero),
                    "ano": int(annio),
                    "texto": texto_crudo,
                    "texto_limpio": texto_limpio,
                }
            )
            n_filas += 1
    logger.info(
        "Procesamiento CSV terminado: %d filas en %.1f s", n_filas, time.time() - inicio
    )
    return registros, excluidas, n_filas


def clasificar_no_sentencia(comienzo_texto: str) -> str:
    """Clasifica el motivo de exclusion de una fila no parseada (criterio E1).

    Args:
        comienzo_texto: Primeros caracteres del texto crudo.

    Returns:
        auto si es un auto (clave A-xxx), basura_html si contiene marcado
        script, o no_clasificado en otro caso.
    """
    bajo = comienzo_texto.lower()
    if "<script" in bajo:
        return "basura_html"
    if "auto" in bajo:
        return "auto"
    return "no_clasificado"


def verificar_limpieza_e1(
    registros: list, ruta_metadatos_e1: str, logger: logging.Logger
) -> dict:
    """Verifica que la limpieza replica la de E1 comparando conteos por documento.

    Compara, para cada id, el numero de tokens del texto_limpio generado contra
    n_palabras_clean del df_metadatos.csv de la primera entrega (calculado con
    las mismas funciones). Reporta documentos comparados, coincidencias y las
    primeras divergencias si las hay.

    Args:
        registros: Lista de dicts con id y texto_limpio generados.
        ruta_metadatos_e1: Ruta del df_metadatos.csv de la primera entrega.
        logger: Logger de ejecucion.

    Returns:
        Dict con n_comparados, n_coincidentes, tasa_coincidencia y divergencias.
    """
    import pandas as pd

    df_e1 = pd.read_csv(ruta_metadatos_e1, dtype={"id": str})
    mapa_e1 = dict(zip(df_e1["id"].astype(str), df_e1["n_palabras_clean"].astype(int)))
    n_comp = 0
    n_ok = 0
    divergencias = []
    for reg in registros:
        esperado = mapa_e1.get(str(reg["id"]))
        if esperado is None:
            continue
        n_comp += 1
        obtenido = len(reg["texto_limpio"].split()) if reg["texto_limpio"] else 0
        if obtenido == int(esperado):
            n_ok += 1
        elif len(divergencias) < 5:
            divergencias.append(
                {
                    "id": str(reg["id"]),
                    "esperado_e1": int(esperado),
                    "obtenido": obtenido,
                }
            )
    tasa = n_ok / n_comp if n_comp else 0.0
    logger.info(
        "Verificacion E1: %d/%d documentos coinciden en n_palabras_clean (%.4f%%)",
        n_ok,
        n_comp,
        tasa * 100,
    )
    if divergencias:
        logger.warning("Divergencias de ejemplo: %s", divergencias)
    return {
        "n_comparados": n_comp,
        "n_coincidentes": n_ok,
        "tasa_coincidencia": tasa,
        "divergencias_ejemplo": divergencias,
    }


def obtener_version_java(logger: logging.Logger) -> dict:
    """Captura las versiones de Java por defecto y Java 17 (motor Spark).

    Args:
        logger: Logger de ejecucion.

    Returns:
        Dict con java_default, java17 y java_usado.
    """
    info = {"java_default": "desconocida", "java17": "desconocida", "java_usado": ""}
    try:
        r = subprocess.run(
            ["/usr/bin/java", "-version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        info["java_default"] = (r.stderr or r.stdout).strip().splitlines()[0]
    except Exception as error:  # noqa: BLE001 - diagnostico best-effort, no bloquea
        info["java_default"] = f"error: {error}"
    ruta17 = os.path.join(CONFIG["java17_home"], "bin", "java")
    try:
        r = subprocess.run(
            [ruta17, "-version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        info["java17"] = (r.stderr or r.stdout).strip().splitlines()[0]
    except Exception as error:  # noqa: BLE001 - diagnostico best-effort, no bloquea
        info["java17"] = f"error: {error}"
    info["java_usado"] = info["java17"]
    logger.info("Java por defecto: %s", info["java_default"])
    logger.info("Java 17 (Spark): %s", info["java17"])
    return info


def tamano_bytes(ruta: str) -> int:
    """Calcula el tamano en bytes de un archivo o directorio (recursivo).

    Args:
        ruta: Ruta de archivo o directorio.

    Returns:
        Tamano total en bytes (0 si la ruta no existe).
    """
    p = Path(ruta)
    if not p.exists():
        return 0
    if p.is_file():
        return p.stat().st_size
    total = 0
    for hijo in p.rglob("*"):
        if hijo.is_file():
            total += hijo.stat().st_size
    return total


class MedidorRamPico:
    """Mide el pico de RSS del proceso con muestreo periodico en background."""

    def __init__(self, intervalo: float = 0.5) -> None:
        """Inicializa el medidor con el intervalo de muestreo.

        Args:
            intervalo: Segundos entre muestras de RSS.
        """
        self.intervalo = intervalo
        self.pico_mb = 0.0
        self._activo = False
        self._hilo = None

    def _muestrear(self) -> None:
        """Bucle de muestreo de RSS hasta detener el medidor."""
        while self._activo:
            try:
                if psutil is not None:
                    rss = psutil.Process(os.getpid()).memory_info().rss
                    pico = rss / (1024 * 1024)
                    self.pico_mb = max(self.pico_mb, pico)
            except Exception:  # noqa: BLE001, S110 - muestreo best-effort
                pass
            time.sleep(self.intervalo)

    def iniciar(self) -> None:
        """Inicia el hilo de muestreo."""
        self._activo = True
        self._hilo = threading.Thread(target=self._muestrear, daemon=True)
        self._hilo.start()

    def detener(self) -> float:
        """Detiene el muestreo y devuelve el pico observado en MB.

        Returns:
            Pico de RSS en megabytes.
        """
        self._activo = False
        if self._hilo is not None:
            self._hilo.join(timeout=2.0)
        return self.pico_mb


def escribir_parquet_spark(
    registros: list, ruta_parquet: str, logger: logging.Logger
) -> dict:
    """Escribe los registros a Parquet particionado por tipo y ano con Spark.

    Crea una SparkSession local[*] con driver de 8g usando Java 17 (la Java 26
    por defecto es incompatible con Spark 4.x: ClassNotFoundException
    jdk.internal.ref.Cleaner en Platform.<clinit>). Si Spark falla, aplica el
    fallback documentado con pandas+pyarrow y particionado equivalente.

    Nota de menor riesgo (2026-09-08): el corpus completo (~1,6 GB de texto
    mas texto_limpio en una lista Python) supera el heap del driver al
    paralelizar con el particionado por defecto (tarea de ~58 MB, OOM
    java.lang.OutOfMemoryError: Java heap space). Por eso se paraleliza con
    256 slices y se evita el job extra de count(); si aun asi falla, el
    fallback pandas+pyarrow produce el mismo particionado tipo/ano.

    Args:
        registros: Lista de dicts con las columnas del parquet.
        ruta_parquet: Directorio de salida del dataset particionado.
        logger: Logger de ejecucion.

    Returns:
        Dict con motor, version_spark y detalle del motor usado.
    """
    os.environ["JAVA_HOME"] = CONFIG["java17_home"]
    spark = None
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql import types as t

        logger.info("Motor primario: Spark (JAVA_HOME=%s)", os.environ["JAVA_HOME"])
        spark = (
            SparkSession.builder.master(CONFIG["spark_master"])
            .appName(CONFIG["spark_app"])
            .config("spark.driver.memory", CONFIG["spark_driver_memory"])
            .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
            .config("spark.default.parallelism", "64")
            .config("spark.sql.shuffle.partitions", "64")
            .config("spark.driver.maxResultSize", "4g")
            .getOrCreate()
        )
        logger.info("SparkSession creada: version %s", spark.version)
        version_spark = spark.version
        esquema = t.StructType(
            [
                t.StructField("id", t.StringType(), False),
                t.StructField("tipo", t.StringType(), False),
                t.StructField("numero", t.IntegerType(), False),
                t.StructField("ano", t.IntegerType(), False),
                t.StructField("texto", t.StringType(), False),
                t.StructField("texto_limpio", t.StringType(), False),
            ]
        )
        rdd = spark.sparkContext.parallelize(registros, numSlices=256)
        df = spark.createDataFrame(rdd, schema=esquema)
        df = df.select(*CONFIG["columnas_parquet"])
        logger.info(
            "DataFrame Spark: %d filas (conteo Python), esquema:", len(registros)
        )
        for campo in df.schema.fields:
            logger.info("  %s: %s", campo.name, campo.dataType.simpleString())
        df.write.mode("overwrite").partitionBy("tipo", "ano").parquet(ruta_parquet)
        n_part = len([p for p in Path(ruta_parquet).rglob("*.parquet") if p.is_file()])
        logger.info("Parquet Spark escrito en %s (%d archivos)", ruta_parquet, n_part)
        spark.stop()
        spark = None
        return {"motor": "spark", "version_spark": version_spark, "fallback": False}
    except Exception as error:  # noqa: BLE001 - fallback documentado ante fallo Spark
        try:
            detalle_error = traceback.format_exc(limit=8)[-600:]
        except Exception:  # noqa: BLE001 - ultimo recurso si falla el formateo
            detalle_error = type(error).__name__
        logger.warning(
            "Spark fallo (%s); usando fallback pandas+pyarrow", type(error).__name__
        )
        logger.warning("Detalle Spark (traceback recortado): %s", detalle_error)
        if spark is not None:
            try:
                spark.stop()
            except Exception:  # noqa: BLE001, S110 - limpieza best-effort
                pass
        import pandas as pd
        import pyarrow as pa
        import pyarrow.dataset as ds

        version_pyarrow = pa.__version__
        df_pd = pd.DataFrame(registros, columns=CONFIG["columnas_parquet"])
        if Path(ruta_parquet).exists():
            shutil.rmtree(ruta_parquet)
        tabla = pa.Table.from_pandas(df_pd, preserve_index=False)
        ds.write_dataset(
            tabla,
            base_dir=ruta_parquet,
            format="parquet",
            partitioning=["tipo", "ano"],
        )
        del tabla
        logger.info("Parquet fallback escrito en %s", ruta_parquet)
        return {
            "motor": "fallback_pandas_pyarrow",
            "version_spark": None,
            "pyarrow": version_pyarrow,
            "fallback": True,
            "error_spark": detalle_error[:500],
        }


def main() -> None:
    """Orquesta la migracion CSV a Parquet particionado con metricas."""
    logger = configurar_logging(CONFIG["ruta_log"])
    medidor = MedidorRamPico()
    medidor.iniciar()
    inicio_total = time.time()
    try:
        random.seed(CONFIG["semilla"])
        np.random.seed(CONFIG["semilla"])
        logger.info("Semilla fijada: %d", CONFIG["semilla"])
        logger.info("CSV entrada: %s", CONFIG["ruta_csv"])
        logger.info("Parquet salida: %s", CONFIG["ruta_parquet"])

        confirmar_esquema_csv(CONFIG["ruta_csv"], CONFIG["n_muestra_esquema"], logger)

        stopw = construir_stopwords_gramaticales()
        logger.info(
            "Stopwords gramaticales: 313 originales -> %d formas normalizadas",
            len(stopw),
        )

        registros, excluidas, n_filas = procesar_csv(CONFIG["ruta_csv"], stopw, logger)

        conteo_motivos = Counter(clasificar_no_sentencia(t) for _, t in excluidas)
        logger.info(
            "Excluidas %d filas no-sentencia: %s", len(excluidas), dict(conteo_motivos)
        )
        for pos, comienzo in excluidas:
            logger.info(
                "  excluida pos=%d motivo=%s inicio=%s",
                pos,
                clasificar_no_sentencia(comienzo),
                comienzo[:100],
            )

        verificacion = verificar_limpieza_e1(
            registros, CONFIG["ruta_metadatos_e1"], logger
        )

        distrib_tipo = dict(Counter(r["tipo"] for r in registros))
        anos = [r["ano"] for r in registros]
        rango_anos = {"min": min(anos), "max": max(anos)} if anos else {}
        distrib_ano = dict(sorted(Counter(anos).items()))
        logger.info("Distribucion por tipo: %s", distrib_tipo)
        logger.info("Rango de anos: %s", rango_anos)

        info_java = obtener_version_java(logger)
        info_motor = escribir_parquet_spark(registros, CONFIG["ruta_parquet"], logger)

        tiempo_total = time.time() - inicio_total
        ram_pico_mb = medidor.detener()
        tam_csv = tamano_bytes(CONFIG["ruta_csv"])
        tam_parquet = tamano_bytes(CONFIG["ruta_parquet"])
        metricas = {
            "fecha": datetime.now(timezone.utc).date().isoformat(),
            "csv_bytes": tam_csv,
            "csv_gb": round(tam_csv / (1024**3), 3),
            "parquet_bytes": tam_parquet,
            "parquet_gb": round(tam_parquet / (1024**3), 3),
            "filas_entrada": n_filas,
            "filas_salida": len(registros),
            "filas_excluidas": len(excluidas),
            "excluidas_por_motivo": dict(conteo_motivos),
            "distribucion_tipo": distrib_tipo,
            "rango_anos": rango_anos,
            "distribucion_ano": {str(k): v for k, v in distrib_ano.items()},
            "tiempo_total_seg": round(tiempo_total, 1),
            "ram_pico_mb": round(ram_pico_mb, 1),
            "ram_pico_gb": round(ram_pico_mb / 1024, 2),
            "motor": info_motor.get("motor"),
            "version_spark": info_motor.get("version_spark"),
            "fallback": info_motor.get("fallback"),
            "java": info_java,
            "columnas": CONFIG["columnas_parquet"],
            "ruta_parquet": CONFIG["ruta_parquet"],
            "verificacion_limpieza_e1": verificacion,
            "stopwords_formas": len(stopw),
        }
        Path(CONFIG["ruta_metricas"]).parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG["ruta_metricas"], "w", encoding="utf-8") as fh:
            json.dump(metricas, fh, indent=2, ensure_ascii=False)
        logger.info("Metricas guardadas en %s", CONFIG["ruta_metricas"])
        logger.info(
            "RESUMEN: motor=%s filas=%d->%d excluidas=%d csv=%.3fGB parquet=%.3fGB "
            "tiempo=%.1fs ram_pico=%.2fGB verificacion=%.4f%%",
            metricas["motor"],
            metricas["filas_entrada"],
            metricas["filas_salida"],
            metricas["filas_excluidas"],
            metricas["csv_gb"],
            metricas["parquet_gb"],
            metricas["tiempo_total_seg"],
            metricas["ram_pico_gb"],
            verificacion["tasa_coincidencia"] * 100,
        )
    except Exception:  # noqa: BLE001 - main captura todo para log + exit 1
        medidor.detener()
        logger.error("Fallo la migracion:\n%s", traceback.format_exc())
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
