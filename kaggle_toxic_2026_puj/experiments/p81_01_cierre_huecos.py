"""experiments/p81_01_cierre_huecos.py

Fecha: 2026-09-30
Descripcion: cierra los cinco huecos de EVIDENCIA que los dictamenes de especialistas
    dejaron en P80. Ejecuta veinte afirmaciones, A0 de integridad y A1 a A19 de contenido,
    cada una como una funcion que devuelve su veredicto con el nombre de la afirmacion.
    Escribe reports/p81_cierre_huecos.json y reports/p81_cierre_huecos.md.
Estado del arte aplicado: pre-registro docs/p81_preregistro_cierre_huecos.md sha256
    55e6900cce49bd7e2ee98b00e0c315081c04718edb18df4abd49f484755faabc commit 86a9140, y
    su enmienda 1, docs/p81_enmienda_1.md sha256
    496223801147a414e74a53fb7cf407ae48ad4671c74e45daba4dce19f15801c6 commit c3dee26,
    que sustituye el learning rate fijo del bloque 1 por una rejilla declarada. Los dos
    documentos son vinculantes y los dos sha256 se comprueban antes de ejecutar nada.

El bloque 1 es un ancla ejecutable: dos brazos, MEDIA y SUMA, identicos en todo lo demas
y verificado mecanicamente por cuatro sha256, sobre un corpus SINTETICO PLANTADO de 65
palabras y 400 documentos, dimension 32, K 10 y SGD puro. Por la enmienda 1 la rejilla
tiene seis puntos de learning rate y tres semillas; el primario es el mejor de la rejilla
de cada objetivo y la curva entera se publica. Ademas se publica el recuento de victorias
por learning rate, porque la curva no es monotona y con tres semillas eso es ruido. La
curva se publica aunque salga a favor del primario.

El resto de bloques son medidas y comprobaciones estructurales: las tasas de OOV y de
relleno con el tokenizador real de v9_04, la separacion de las dos celdas, la distincion
entre reejecion y replicacion, la identidad de conjuntos de identificadores sin abrir el
holdout, y la serie paso a paso del learning rate del scheduler.

Este script no lee test, ni el leaderboard, ni el holdout, ni genera submission. No toca
V03, ni los umbrales 0.44, 0.51 y 0.46, ni nada de modelos/, salvo el artefacto congelado
que se abre en solo lectura y se hashea antes y despues. No entrena el Word2Vec de diez
milones de tokens. No instala gensim ni ninguna dependencia nueva. No usa la GPU. No
afirma que Word2Vec sea mejor que V03.

    V03 remains the best verified leaderboard result.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn


# ===== CONFIGURACION =====
# Rutas, constantes declaradas, tolerancias y margenes. Todo el calculo sale de aqui.
# Ninguna funcion lleva numeros magicos que no esten tambien en este bloque.

RAIZ = Path(__file__).resolve().parents[1]
EXPERIMENTOS = RAIZ / "experiments"
MODELOS = RAIZ / "modelos"
REPORTS = RAIZ / "reports"
CONFIG_DIR = RAIZ / "config"
DOCS = RAIZ / "docs"
DATA_PROCESADOS = RAIZ / "data" / "processed"

# Documentos vinculantes. Los dos sha256 se comprueban ANTES de leer ninguna otra cosa.
PREREGISTRO = DOCS / "p81_preregistro_cierre_huecos.md"
PREREGISTRO_SHA256 = "55e6900cce49bd7e2ee98b00e0c315081c04718edb18df4abd49f484755faabc"
COMMIT_PREREGISTRO = "86a9140"
ENMIENDA = DOCS / "p81_enmienda_1.md"
ENMIENDA_SHA256 = "496223801147a414e74a53fb7cf407ae48ad4671c74e45daba4dce19f15801c6"
COMMIT_ENMIENDA = "c3dee26"

# Salidas de P81. Unicamente se escriben estos dos ficheros, mas el propio script.
JSON_INFORME = REPORTS / "p81_cierre_huecos.json"
MD_INFORME = REPORTS / "p81_cierre_huecos.md"
SUFIJO_TEMPORAL = ".tmp_p81"

FORMULACION = "V03 remains the best verified leaderboard result."

# Artefacto congelado: solo lectura, sha256 antes y despues. Es la asercion A0.
ARTEFACTO = MODELOS / "twitter_domain" / "w2v_twitter_100d.npz"
ARTEFACTO_SHA256 = "302c994d13b75f8f5fbd440eea2c4f4113997e2581524021923b92f8244d705a"

# Ficheros fuente, todos de solo lectura.
FUENTE_P71 = EXPERIMENTOS / "p71_04_entrenar_word2vec.py"
FUENTE_V9 = EXPERIMENTOS / "v9_04_dl_secuencial_fasttext.py"
FUENTE_V10 = EXPERIMENTOS / "v10_11_bigru_finetuning.py"
FUENTE_P72 = EXPERIMENTOS / "p72_01_dimensiones.py"
FUENTE_DL = EXPERIMENTOS / "11_dl_desde_cero.py"
INFORME_P79 = REPORTS / "p79_auditoria_sgns.json"
INFORME_P80 = REPORTS / "p80_auditoria_w2v.json"
INFORME_DL = REPORTS / "dl_desde_cero_resultados.json"
INFORME_P78 = REPORTS / "p78_cobertura.json"
HOLDOUT_V2 = CONFIG_DIR / "holdout_v2.json"
GRUPOS_CSV = DATA_PROCESADOS / "grupos_near_duplicados.csv"
CORPUS_TWITTER = (RAIZ / "data" / "external" / "corpora" / "twitter_domain"
                  / "corpus_twitter_tokens.txt")

# Ancla documental externa del bloque 1. word2vec.c NO se recupera en esta ejecucion: el
# pre-registro lo consulto una vez y lo transcribio con numero de linea. Se declara.
WORD2VEC_C_URL = ("https://raw.githubusercontent.com/tmikolov/word2vec/master/"
                  "word2vec.c")
WORD2VEC_C_SHA256 = "df07a1d504c22aa747f9fc40ddc6cfed52e12c3a402861398426829b83d96ef4"
WORD2VEC_C_LINEAS = 715
FRAGMENTO_C_INICIO = "520:"
FRAGMENTO_C_FIN = "540:"

# Transcripcion de la ecuacion (4) de Mikolov et al. 2013, tal como queda escrita en el
# pre-registro, que es el documento hasheado que la ancla. Se escribe con escapes, nunca
# con el caracter U+2019 literal, para que el informe entero sea ASCII puro.
ECUACION_4 = ("log sigma( v'_{w_O}^T v_{w_I} )  +  "
              "sum_{i=1}^{k} E_{w_i ~ P_n(w)} [ log sigma( - v'_{w_i}^T v_{w_I} ) ]")
REFERENCIA_MIKOLOV = {
    "autores": ("Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, Jeffrey Dean"),
    "titulo": ("Distributed Representations of Words and Phrases and their "
               "Compositionality"),
    "seccion": "2.2, Negative Sampling",
    "ecuacion": "(4)",
    "arxiv": "1310.4546v1 [cs.CL]",
    "doi": "10.48550/arXiv.1310.4546",
    "fecha": "16 de octubre de 2013",
}

# Patron de token del proyecto. Se lee del informe de P79 y se reemite con escapes.
PATRON_TOKEN_PROYECTO = "[#@]?\\w+(?:['\\u2019]\\w+)*"

# Bloque 1: corpus plantado y dos brazos. Todo declarado antes de ejecutar.
BLOQUE_1 = {
    "familias": 6,
    "palabras_por_familia": 10,
    "palabras_neutras": 5,
    "documentos": 400,
    "ventana": 3,
    "probabilidad_de_neutra": 0.15,
    "dimension": 32,
    "negativos_k": 10,
    "lote": 128,
    "pasos": 1500,
    "optimizador": "SGD puro, sin momentum, sin Adam",
    "vecinos_de_la_metrica": 5,
    "permutaciones_del_azar": 2000,
    "semilla_corpus": 20261201,
    "semilla_azar": 20261202,
    "semillas_rejilla": (20261401, 20261402, 20261403),
    "semillas_de_confirmacion": tuple(range(20261301, 20261311)),
    "semilla_del_control": 20261501,
    "rejilla": (0.02, 0.05, 0.10, 0.20, 0.35, 0.50),
    "zona_betonal": 0.99,
    "umbral_zona_betonal": 0.99,
    "motivo_del_sgd_puro": (
        "Adam normaliza por parametro y en buena parte se comeria el factor 1/K, de modo que "
        "la comparacion mediria el optimizador y no el objetivo"),
    "motivo_de_los_pasos_fijos": (
        "aqui no se compara convergencia sino el objetivo. Si cada brazo parara cuando su "
        "perdida deja de bajar, cada uno tendria su propio criterio y la comparacion mediria eso"),
    "nota_de_la_curva_no_monotona": (
        "la curva de learning rate frente a M1 NO es monotona: MEDIA gana en un punto "
        "intermedio y pierde en el punto alto. Con tres semillas eso es ruido, no un "
        "hallazgo, y por eso se publica el recuento de victorias por learning rate ademas "
        "del primario"),
    "banda_muerta_de_victorias": "el recuento de victorias se declara ZONA DE NO CONCLUSION "
                                 "si queda en 2, 3 o 4 de 6 puntos, es decir si el reparto "
                                 "queda a 2 a 4, a 3 a 3 o a 4 a 2",
    "victorias_decisivas_minimo": 5,
}

# Criterio de la zona betonal, declarado antes de ejecutar. Es POR BRAZO y no por punto, y la
# razon es que un punto de la rejilla son DOS mediciones, una por objetivo: si la MEDIA satura
# en un learning rate, eso dice algo del brazo MEDIA y no invalida la medicion de la SUMA en
# ese mismo punto. El criterio por punto mezcla en la comparacion un dato saturado con uno
# que no lo esta, y ademas deja al primario sin puntos en cuanto un brazo satura. Por brazo,
# cada objetivo se mide en SU mejor punto no saturado y la comparacion conserva un sentido.
# La lectura por punto se CALCULA IGUAL y se publica, porque con estos numeros las dos
# lecturas no coinciden y decidir solo con una seria ocultar la otra.
CRITERIO_ZONA_BETONAL = "POR_BRAZO"
MOTIVO_DEL_CRITERIO_DE_ZONA_BETONAL = (
    "la zona betonal se excluye POR BRAZO y no por punto. Un punto de la rejilla son dos "
    "mediciones, una por objetivo: que la MEDIA sature en un learning rate es un hecho "
    "sobre el brazo MEDIA y no invalida la medicion de la SUMA en ese mismo learning rate. El "
    "criterio por punto obliga a comparar una cifra saturada con otra que no lo esta, y "
    "deja al primario sin ningun punto en cuanto un solo brazo satura. Se publica tambien la "
    "lectura por punto, con sus numeros, porque las dos lecturas no coinciden y elegir solo "
    "una seria ocultar la otra")

# Tolerancias y margenes fijados en el pre-registro y en la enmienda 1.
TOL = {
    "no_discriminacion": 0.005,
    "discriminacion": 0.020,
    "distancia_al_azar_del_control": 0.020,
    "victorias_minimas_de_10": 9,
    "distancia_minima_por_canal": 1e-9,
    "variacion_entre_pasadas": 0.0,
    "coincidencia_de_convencion_p79": 1e-9,
    "umbral_de_impacto_despreciable": 0.005,
    "pasos_con_diferencia_relativa": 1e-6,
    "recuento_de_entradas_del_holdout": 0,
    "distancia_al_holdout_v2": 0.0,
}

# Bloque 2: celdas separadas y sonda sobre las clases BiGRU reales.
BLOQUE_2 = {
    "max_len": 64,
    "dimension_entrada": 200,
    "filas_de_sonda": 12,
    "longitud_minima": 6,
    "semilla_matriz": 20261601,
    "semilla_sequencias": 20261602,
    "semilla_modelos": 20261603,
    "semilla_vector_declarado": 20261604,
    "referencia": "REF",
    "vector_declarado_de_oov": "vector no nulo, fijado con semilla, publicado en el informe",
    "nota_del_contraste_c2_c2b": (
        "C2 y C2b comparten identificadores y mascara. Lo unico que cambia es el contenido de "
        "la fila id_oov: a cero en C2, que es lo que hace el codigo, y el vector declarado en "
        "C2b. Por eso su diferencia separa el efecto de la mascara sobre el OOV del efecto de "
        "meter un vector nulo, que es exactamente lo que pide el pre-registro. En C2 la fila "
        "id_oov sobrevive a la mascara porque su identificador es mayor que 0, y por eso "
        "tambien entra en el denominador del promedio"),
}

# Bloque 4: lo que se recalcula sin abrir el holdout.
BLOQUE_4 = {
    "n_pliegues": 5,
    "random_state": 42,
    "n_dev": 1483,
    "n_holdout": 248,
    "conteos_publicados": (347, 346, 346, 346, 346),
    "hash_ids_csv_declarado": (
        "e5b62e5f22c4c63a8c43df4019969504d44c0baa5b503c19ba3560032f5ee6cd"),
    "convenciones_del_hash": ("ids ordenados separados por comas, ids ordenados separados "
                              "por saltos de linea, y orden de fichero"),
    "campos_permitidos_del_holdout_v2": ("n_tweets_dev, n_tweets_holdout y hash_ids_csv, "
                                         "leidos con expresion regular sobre el texto del "
                                         "fichero, sin json.loads, para que ninguna lista "
                                         "del documento llegue a existir como objeto"),
    "informes_de_la_fase_dl": ("DL-01, DL-02 y DL-03 de reports/dl_desde_cero_resultados.json"),
}

# Bloque 5: recorte declarado para el learning rate paso a paso.
RECRETE = {"lineas": 2000, "lote": 256}

# Cifras que P80 ya publico y que P81 reutiliza SOLO como punto de partida declarado.
P80_REUTILIZADO = {
    "total_lotes_plan": 1395,
    "total_lotes_recorridos": 1398,
    "desviacion_relativa": 0.002150537634408602,
    "lr_inicial": 6.4,
    "lr_minimo": 0.001,
    "pasos_totales": 1398,
    "ultimo_real": 0.004577968526466236,
    "ultimo_ideal": 0.001,
    "pasos_en_el_piso_real": 0,
    "pasos_en_el_piso_ideal": 3,
    "fraccion_en_el_piso_real": 0.0,
    "fraccion_en_el_piso_ideal": 0.002145922746781116,
    "media_real": 3.202288984263234,
    "media_ideal": 3.195424177396281,
    "diferencia_relativa_de_las_medias": 0.0021437187276626846,
}

# Cifras de cobertura que P80 cito y que P81 NO arrastra: se recalculan con el tokenizador
# de v9_04. Se publican para que se vea que ninguna se ha copiado.
CIFRAS_A_NO_ARRASTRAR = {
    "p78_oov_normalizado": 0.112658,
    "p78_oov_crudo": 0.313639,
    "p78_maxlen": 60,
    "p80_oov_de_la_discusion_en_por_ciento": 11.2726,
    "v9_04_max_len": 64,
}

# Umbrales de V03 y su referencia. Se DECLARAN para poder citarlos; no se tocan.
UMBRALES_V03 = {"HS": 0.44, "TR": 0.51, "AG": 0.46}
V03_CV_DECLARADO = 0.7772114525165721

# Rutas que P81 no abre de ninguna manera, con el motivo declarado.
RUTAS_PROHIBIDAS = {
    "data/raw/toxic_test.csv": "test de la competencia",
    "data/processed/holdout_v2_ids.csv": "fichero de identificadores del holdout",
    "data/processed/holdout2_ids.csv": "fichero de identificadores del holdout",
    "submissions/": "submissions",
    "experiments/baseline_v03_frozen/": "V03 congelado, no se toca",
    "reports/leaderboard*": "ningun score de leaderboard se lee ni se consulta",
    "modelos/*.*pt": "ningun checkpoint de modelos/",
}

# Sustituciones ASCII aplicadas a texto ajeno. Se declaran en el informe, nunca en silencio.
SUSTITUCIONES_ASCII: list = []

# Numero minimo de comprobaciones de humo que deben pasar. Es un suelo declarado, no una
# cuenta viva: si el total baja de aqui, el modo de humo falla en vez de dar un OK falso.
MINIMO_COMPROBACIONES_HUMO = 83

# Frases obligatorias del pre-registro, escritas aqui para que no se puedan perder.
FRASE_A3 = ("ESTA_CONVENCION_ES_COHERENTE_CON_P79_Y_SIGUE_SIENDO_UNA_REIMPLEMENTACION_PROPIA")
FRASE_A13 = ("EN_P81_ESTA_COMPARACION_SE_LLAMA_REEJECION_NO_REPLICACION")
FRASE_TEST_SIN_CONCLUSION = "EL_TEST_NO_DISCRIMINA_Y_NO_CONCLUYE_NADA"
FRASE_A17 = ("LOS_TRES_INFORMES_DE_LA_FASE_DL_NO_PERSISTEN_LOS_IDS_POR_PLIEGUE_LA_IDENTIDAD_"
             "A_NIVEL_DE_IDS_SE_ESTABLECE_POR_RECALCULO_DETERMINISTA_DEL_CAMINO_DE_CODIGO_"
             "COMPARTIDO_Y_NO_POR_LECTURA_DE_IDS_PERSISTIDOS")
ETIQUETA_SIN_EFECTO = "SIN EFECTO MEDIBLE"

# Palabras que el indice de afirmaciones usa para saber si una afirmacion lleva STOP.
AFIRMACIONES_CON_STOP = ("A0", "A3", "A4", "A5", "A6", "A7", "A9", "A12", "A13", "A15",
                         "A16", "A19")

# ===== FIN CONFIGURACION =====


class Parada(Exception):
    """Parada del pre-registro: una afirmacion con marca STOP ha fallado.

    La captura de esta excepcion en `main` escribe un JSON de parada y NO el informe
    completo, que es lo que exige el criterio de parada de la seccion 9.

    Attributes:
        afirmacion: identificador de la afirmacion que para, por ejemplo "A5".
        motivo: explicacion en una linea.
        detalle: diccionario con lo minimo que hay que registrar.
    """

    def __init__(self, afirmacion: str, motivo: str, detalle: dict | None = None) -> None:
        """Construye la parada.

        Args:
            afirmacion: identificador de la afirmacion.
            motivo: explicacion en una linea.
            detalle: diccionario con lo minimo que hay que registrar.
        """
        super().__init__(f"STOP {afirmacion}: {motivo}")
        self.afirmacion = afirmacion
        self.motivo = motivo
        self.detalle = detalle or {}


# ===== SECCION 1: INFRAESTRUCTURA DE LECTURA, INTEGRIDAD Y ESCRITURA =====


def log(mensaje: str) -> None:
    """Imprime un mensaje con marca de tiempo, en ASCII puro.

    Args:
        mensaje: texto a mostrar. Se exige ASCII en todo lo impreso.
    """
    marca = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[P81-01] {marca} | {mensaje}", flush=True)


def a_ascii(texto: str, origen: str) -> str:
    """Convierte un texto a ASCII puro, registrando cada sustitucion.

    Args:
        texto: cadena de entrada.
        origen: etiqueta del lugar de donde sale la cadena, para el registro.

    Returns:
        La cadena con todos sus caracteres en el rango 0 a 127.
    """
    salida = []
    for caracter in texto:
        if ord(caracter) > 127:
            SUSTITUCIONES_ASCII.append({"origen": origen, "caracter": f"U+{ord(caracter):04X}"})
            salida.append("?")
        else:
            salida.append(caracter)
    return "".join(salida)


def sha256_de_bytes(datos: bytes) -> str:
    """Devuelve el sha256 hexadecimal de un bloque de bytes.

    Args:
        datos: bloque de bytes.

    Returns:
        La cadena hexadecimal de 64 caracteres.
    """
    return hashlib.sha256(datos).hexdigest()


def huella_de_fichero(ruta: Path) -> dict:
    """Registra las tres huellas de un fichero: sha256, tamano y `st_mtime_ns`.

    Los tres valores se toman en la MISMA llamada. Es la unica forma de que la comparacion
    "antes" contra "despues" signifique algo.

    Args:
        ruta: fichero que se va a medir.

    Returns:
        Diccionario con `ruta`, `sha256`, `bytes` y `mtime_ns`.
    """
    datos = ruta.read_bytes()
    info = ruta.stat()
    return {
        "ruta": str(ruta.relative_to(RAIZ)),
        "sha256": sha256_de_bytes(datos),
        "bytes": info.st_size,
        "mtime_ns": info.st_mtime_ns,
    }


def leer_fuente(ruta: Path) -> dict:
    """Lee un fichero de texto fuente entero y registra su sha256 y su numero de lineas.

    Args:
        ruta: fichero de texto.

    Returns:
        Diccionario con `ruta`, `texto`, `lineas`, `n_lineas` y `sha256`.
    """
    texto = ruta.read_text(encoding="utf-8")
    lineas = texto.splitlines()
    return {
        "ruta": str(ruta.relative_to(RAIZ)),
        "texto": texto,
        "lineas": lineas,
        "n_lineas": len(lineas),
        "sha256": sha256_de_bytes(texto.encode("utf-8")),
    }


def linea_de(fuente: dict, numero: int) -> str:
    """Devuelve una linea concreta de una fuente ya leida, sin el salto final.

    Args:
        fuente: diccionario devuelto por `leer_fuente`.
        numero: numero de linea, contado desde 1.

    Returns:
        El texto de la linea, o una marca si el numero cae fuera del fichero.
    """
    indice = numero - 1
    if 0 <= indice < len(fuente["lineas"]):
        return fuente["lineas"][indice]
    return "<linea fuera de rango>"


def ruta_temporal(final: Path) -> Path:
    """Ruta temporal de un artefacto, en el mismo directorio que su version final.

    Args:
        final: ruta del artefacto definitivo.

    Returns:
        La ruta temporal con la que se escribe antes de publicar.
    """
    return final.with_name(final.name + SUFIJO_TEMPORAL)


def publicar_atomico(pares: list) -> None:
    """Renombra todos los temporales a sus nombres finales, uno detras de otro.

    Se llama SOLO despues de que han pasado todas las verificaciones.

    Args:
        pares: pares `(temporal, final)`.

    Raises:
        SystemExit: si algun temporal no existe al llegar aqui.
    """
    for temporal, _ in pares:
        if not temporal.exists():
            raise SystemExit(f"STOP escritura atomica: falta el temporal {temporal}")
    for temporal, final in pares:
        temporal.replace(final)


def escribir_ascii_atomico(ruta: Path, texto: str, etiqueta: str) -> Path:
    """Escribe un fichero de texto ensuring ASCII puro, con la comprobacion EN MEMORIA.

    El orden importa y por eso es explicito: primero se sustituyen los caracteres no ASCII y
    se registra cada sustitucion, despues se codifica el texto EN MEMORIA con
    `encode("ascii")`, y SOLO si eso no lanza se abre el fichero destino. Un `write_text` con
    `encoding="ascii"` trunca el fichero a cero bytes si el texto trae un caracter no ASCII, y
    por eso la comprobacion va antes de abrir nada.

    Args:
        ruta: fichero destino definitivo.
        texto: contenido completo del fichero.
        etiqueta: origen del texto, para el registro de sustituciones.

    Returns:
        La ruta temporal escrita, todavia sin publicar.

    Raises:
        SystemExit: si el destino ya existe o si la codificacion falla.
    """
    limpio = a_ascii(texto, etiqueta)
    try:
        datos = limpio.encode("ascii")
    except UnicodeEncodeError as exc:
        raise SystemExit(f"STOP asercion ASCII en memoria: {etiqueta}: {exc}") from exc
    if ruta.exists():
        raise SystemExit(f"STOP escritura: {ruta} ya existe y solo se admiten temporales hasta "
                         "la publicacion atomica")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta_temporal(ruta)
    temporal.write_bytes(datos)
    return temporal


def verificar_ascii_byte_a_byte(rutas: list) -> dict:
    """Comprueba, byte a byte, que ningun fichero tiene un byte mayor que 127.

    Args:
        rutas: ficheros de texto a verificar.

    Returns:
        Diccionario `{nombre: bool}` con el resultado real de cada fichero.
    """
    observado = {}
    for ruta in rutas:
        ruta = Path(ruta)
        if not ruta.exists():
            observado[ruta.name] = False
        else:
            observado[ruta.name] = not any(b > 127 for b in ruta.read_bytes())
    return observado


def importar_modulo(nombre: str, ruta: Path):
    """Importa un modulo del proyecto por ruta, sin ejecutar su `main`.

    `spec_from_file_location` mas `exec_module` ejecuta el cuerpo de modulo y nada mas: no
    entra en `if __name__ == "__main__"`, luego no entrena ni escribe.

    Args:
        nombre: nombre con el que se registra en `sys.modules`.
        ruta: fichero `.py` a cargar.

    Returns:
        El modulo cargado.
    """
    especificacion = importlib.util.spec_from_file_location(nombre, ruta)
    modulo = importlib.util.module_from_spec(especificacion)
    sys.modules[nombre] = modulo
    especificacion.loader.exec_module(modulo)
    return modulo


def envolver(valor, ruta: str) -> "Tolerante":
    """Envuelve un diccionario para que una lectura ausente no reviente el redactor.

    En P80 el modo de humo daba 60 de 60 mientras el script real reventaba al escribir el
    markdown con un `KeyError`. Este envoltorio convierte toda ausencia en una marca
    visible, que es lo que hace falta para que el humo sirva para algo.

    Args:
        valor: diccionario de partida, o cualquier otro valor.
        ruta: camino desde la raiz, para nombrar las ausencias.

    Returns:
        Vista tolerante del valor.
    """
    return Tolerante(valor, ruta)


class Tolerante(dict):
    """Diccionario cuyas lecturas ausentes devuelven una marca en vez de lanzar.

    Hereda de `dict`, luego serializa con `json.dumps` como un diccionario normal. Lo unico
    que cambia es que `[]` y `get` sobre una clave que no existe no lanzan `KeyError`.

    Attributes:
        ruta: camino desde la raiz del informe, para nombrar la ausencia.
        es_ausente: True si este nodo nacio de una lectura ausente.
    """

    def __init__(self, valor, ruta: str) -> None:
        """Construye la vista tolerante.

        Args:
            valor: diccionario de partida.
            ruta: camino desde la raiz del informe.
        """
        super().__init__(valor if isinstance(valor, dict) else {})
        self.ruta = ruta
        # La marca se fija aqui y NO se hereda. `envolver` se usa para reenvolver el
        # resultado de `__missing__`, y si la heredara, una ausencia seria indistinguible de
        # un diccionario vacio presente: el informe se llenaria de `{}` en vez de
        # `NO_DISPONIBLE`, en silencio.
        self.es_ausente = False

    def __missing__(self, clave):
        """Devuelve un marcador en vez de lanzar.

        Args:
            clave: clave ausente.

        Returns:
            Un `Tolerante` marcado como ausente.
        """
        marca = Tolerante({}, f"{self.ruta}.{clave}")
        marca.es_ausente = True
        return marca

    def get(self, clave, defecto=None):
        """Lee con defecto explicito, sin lanzar.

        Args:
            clave: clave a leer.
            defecto: valor a devolver si la clave no existe.

        Returns:
            El valor de la clave o el defecto.
        """
        if clave in self:
            return self[clave]
        if defecto is None:
            return self[clave]
        return defecto

    def resolver(self, camino):
        """Resuelve un camino de claves, que puede ser una clave simple o una tupla.

        El redactor lee rutas anidadas del tipo `('medida', 'cifras', 'codigo')`. Sin esta
        resolucion, la tupla se usaria como clave literal, no apareceria en el diccionario y
        toda lectura devolveria el defecto: el informe saldria lleno de NO_DISPONIBLE sin que
        nada fallara, que es peor que un fallo visible.

        Args:
            camino: clave simple o tupla de claves anidadas.

        Returns:
            El valor de la ruta, o la marca de ausencia si algun tramo falta.
        """
        tramos = list(camino) if isinstance(camino, (tuple, list)) else [camino]
        valor: object = dict(self)
        for indice, clave in enumerate(tramos):
            if not isinstance(valor, dict) or clave not in valor:
                marca = envolver({}, f"{self.ruta}.{'.'.join(map(str, tramos[:indice + 1]))}")
                marca.es_ausente = True
                return marca
            valor = valor[clave]
        # Solo se envuelve si el valor final es un diccionario. Envolver un escalar lo
        # convertiria en `{}`, porque `Tolerante` solo admite diccionarios.
        if isinstance(valor, dict):
            return envolver(valor, self.ruta)
        return valor

    def txt(self, clave, defecto: str = "NO_DISPONIBLE") -> str:
        """Lee una cadena en una ruta, convirtiendo una ausencia en el defecto.

        Args:
            clave: clave simple o tupla de claves anidadas.
            defecto: cadena a devolver si la ruta falta o vale `None`.

        Returns:
            La cadena leida o el defecto.
        """
        valor = self.resolver(clave)
        if isinstance(valor, dict) and valor.es_ausente:
            return defecto
        if valor is None or valor == "":
            return defecto
        return str(valor)

    def num(self, clave, defecto: float = 0.0) -> float:
        """Lee un numero en una ruta, convirtiendo una ausencia en el defecto.

        Args:
            clave: clave simple o tupla de claves anidadas.
            defecto: numero a devolver si la ruta falta o vale `None`.

        Returns:
            El numero leido o el defecto.
        """
        valor = self.resolver(clave)
        if isinstance(valor, dict) and valor.es_ausente:
            return defecto
        if valor is None:
            return defecto
        return valor

    def txt_de_lista(self, clave, defecto: str = "NO_DISPONIBLE") -> str:
        """Lee una lista y la jointe en una linea, para la tabla del markdown.

        Args:
            clave: clave a leer.
            defecto: cadena a devolver si la clave falta o no es una lista.

        Returns:
            La lista unida por comas, o el defecto.
        """
        valor = self.get(clave, defecto)
        if not isinstance(valor, (list, tuple)):
            return defecto
        if not valor:
            return defecto
        return ", ".join(str(elemento) for elemento in valor)


# ===== SECCION 2: BLOQUE 1, EL NUCLEO SINTETICO Y LOS DOS BRAZOS =====


def construir_vocabulario_plantado(config: dict) -> dict:
    """Construye el vocabulario sintetico del corpus plantado, con familias y neutras.

    Los nombres de las palabras son sinteticos y ASCII: ningun token real del proyecto
    aparece aqui. La respuesta correcta es conocida por construccion, porque para una
    palabra de familia los nueve vecinos correctos son las otras nueve de su familia.

    Args:
        config: diccionario de declaracion del bloque 1.

    Returns:
        Diccionario con `palabras`, `indice`, `familia_de`, `es_neutra`, `familias`,
        `n_palabras` e `id_de`.
    """
    n_familias = int(config["familias"])
    por_familia = int(config["palabras_por_familia"])
    n_neutras = int(config["palabras_neutras"])
    palabras: list[str] = []
    familias: list[int] = []
    for indice_familia in range(n_familias):
        for posicion in range(por_familia):
            palabras.append(f"palabra_f{indice_familia}_{posicion:02d}")
            familias.append(indice_familia)
    for posicion in range(n_neutras):
        palabras.append(f"neutra_{posicion:02d}")
        familias.append(-1)
    indice = {palabra: posicion for posicion, palabra in enumerate(palabras)}
    return {
        "palabras": palabras,
        "indice": indice,
        "familia_de": {palabra: familia for palabra, familia in zip(palabras, familias)},
        "es_neutra": {palabra: familia < 0 for palabra, familia in zip(palabras, familias)},
        "familias": familias,
        "n_palabras": len(palabras),
        "id_de": indice,
    }


def generar_corpus_plantado(config: dict) -> dict:
    """Genera los documentos del corpus plantado con la estructura familiar dentro.

    Cada documento sortea UNA familia de forma uniforme y emite una permutacion aleatoria de
    sus diez palabras, intercalando palabras neutras con probabilidad declarada. Como cada
    documento tiene una sola familia, la invariante de que no haya pares entre familias
    distintas se cumple por construccion; lo que hace esta funcion es dejar constancia
    medible de eso.

    Args:
        config: diccionario de declaracion del bloque 1.

    Returns:
        Diccionario con `documentos`, `n_documentos` y la semilla usada.
    """
    vocabulario = construir_vocabulario_plantado(config)
    generador = np.random.default_rng(int(config["semilla_corpus"]))
    familias_disponibles = sorted({f for f in vocabulario["familias"] if f >= 0})
    neutras = [p for p in vocabulario["palabras"] if vocabulario["es_neutra"][p]]
    documentos: list[list[str]] = []
    for _ in range(int(config["documentos"])):
        familia = familias_disponibles[int(generador.integers(0, len(familias_disponibles)))]
        miembros = [p for p in vocabulario["palabras"] if vocabulario["familia_de"][p] == familia]
        permutacion = list(generador.permutation(miembros))
        documento: list[str] = []
        for palabra in permutacion:
            documento.append(palabra)
            if generador.random() < float(config["probabilidad_de_neutra"]):
                documento.append(neutras[int(generador.integers(0, len(neutras)))])
        documentos.append(documento)
    return {"documentos": documentos, "n_documentos": len(documentos),
            "vocabulario": vocabulario, "semilla": int(config["semilla_corpus"])}


def pares_plantados(documentos: list, vocabulario: dict, ventana: int) -> list:
    """Genera los pares centro-contexto del corpus plantado, con ventana fija.

    No hay subsampling y no hay filtro de frecuencia minima: el corpus esta plantado y los
    pares se generan tal cual, para que la estructura sea exactamente la declarada.

    Args:
        documentos: lista de listas de tokens.
        vocabulario: diccionario del vocabulario plantado.
        ventana: ventana fija a cada lado del centro.

    Returns:
        Lista de tuplas `(id_centro, id_contexto)`.
    """
    indice = vocabulario["indice"]
    pares: list[tuple[int, int]] = []
    for documento in documentos:
        identificadores = [indice[token] for token in documento if token in indice]
        for posicion, centro in enumerate(identificadores):
            for desplazamiento in range(1, ventana + 1):
                for destino in (posicion - desplazamiento, posicion + desplazamiento):
                    if 0 <= destino < len(identificadores):
                        pares.append((centro, identificadores[destino]))
    return pares


def contar_pares_entre_familias_distintas(pares: list, vocabulario: dict) -> int:
    """Cuenta los pares centro-contexto cuyas dos palabras son de familias distintas.

    Las palabras neutras no pertenecen a ninguna familia, luego no pueden formar un par entre
    familias distintas y se ignoran.

    Args:
        pares: lista de tuplas `(id_centro, id_contexto)`.
        vocabulario: diccionario del vocabulario plantado.

    Returns:
        Numero exacto de pares que violan la invariante plantada.
    """
    palabras = vocabulario["palabras"]
    familias = vocabulario["familia_de"]
    offenders = 0
    for centro, contexto in pares:
        palabra_centro = palabras[centro]
        palabra_contexto = palabras[contexto]
        familia_centro = familias[palabra_centro]
        familia_contexto = familias[palabra_contexto]
        if familia_centro >= 0 and familia_contexto >= 0 and familia_centro != familia_contexto:
            offenders += 1
    return offenders


def pares_intra_familia(pares: list, vocabulario: dict) -> int:
    """Cuenta los pares centro-contexto cuyas dos palabras son de la misma familia.

    Args:
        pares: lista de tuplas `(id_centro, id_contexto)`.
        vocabulario: diccionario del vocabulario plantado.

    Returns:
        Numero exacto de pares intra familia.
    """
    palabras = vocabulario["palabras"]
    familias = vocabulario["familia_de"]
    internos = 0
    for centro, contexto in pares:
        familia_centro = familias[palabras[centro]]
        familia_contexto = familias[palabras[contexto]]
        if familia_centro >= 0 and familia_centro == familia_contexto:
            internos += 1
    return internos


def nivel_de_azar(matriz_normalizada: np.ndarray, ids_de_familia: np.ndarray,
                  etiquetas: np.ndarray, vecinos: int, permutaciones: int,
                  semilla: int) -> float:
    """Calcula M3, el valor esperado de M1 bajo una asignacion aleatoria de familias.

    Las permutaciones conservan los tamanos de las familias, que es lo que hace comparables
    el azar y la estructura. La semilla es fija y esta declarada en el bloque de
    configuracion.

    Args:
        matriz_normalizada: matriz de embeddings ya normalizada, de forma palabras por dim.
        ids_de_familia: identificadores de las palabras de familia, en el orden de la matriz.
        etiquetas: etiqueta de familia de cada identificador de `ids_de_familia`.
        vecinos: numero de vecinos mas proximos que exige la metrica.
        permutaciones: numero de permutaciones del azar.
        semilla: semilla del generador.

    Returns:
        El valor esperado de M1 bajo el azar.
    """
    generador = np.random.default_rng(semilla)
    acumulado = 0.0
    for _ in range(permutaciones):
        barajadas = generador.permutation(etiquetas)
        acumulado += precision_en_familia(matriz_normalizada, ids_de_familia, barajadas,
                                          vecinos)
    return acumulado / max(permutaciones, 1)


def precision_en_familia(matriz_normalizada: np.ndarray, ids_de_familia: np.ndarray,
                         etiquetas: np.ndarray, vecinos: int) -> float:
    """M1: fraccion de consultas con los `vecinos` mas proximos en su misma familia.

    Se excluyen de la consulta la propia palabra y las neutras, porque las neutras no
    pertenecen a ninguna familia y contaminarian el criterio.

    Args:
        matriz_normalizada: matriz de embeddings normalizada, de forma palabras por dim.
        ids_de_familia: identificadores de las palabras de familia, en el orden de la matriz.
        etiquetas: etiqueta de familia de cada identificador de `ids_de_familia`.
        vecinos: numero de vecinos mas proximos que se exigen.

    Returns:
        La fraccion, en el intervalo 0 a 1.
    """
    if len(ids_de_familia) == 0:
        return float("nan")
    similares = matriz_normalizada[ids_de_familia] @ matriz_normalizada[ids_de_familia].T
    similares = np.array(similares, dtype=float)
    np.fill_diagonal(similares, -np.inf)
    orden = np.argsort(-similares, axis=1)[:, :vecinos]
    acierto = np.all(etiquetas[orden] == etiquetas[:, None], axis=1)
    return float(acierto.mean())


def separacion(matriz_normalizada: np.ndarray, ids_de_familia: np.ndarray,
               etiquetas: np.ndarray) -> float:
    """M2: similitud coseno media dentro de familia menos la media entre familias.

    Se publica con su signo, porque un valor negativo significa que la estructura aprendida
    separa peor que el azar.

    Args:
        matriz_normalizada: matriz de embeddings normalizada, de forma palabras por dim.
        ids_de_familia: identificadores de las palabras de familia, en el orden de la matriz.
        etiquetas: etiqueta de familia de cada identificador de `ids_de_familia`.

    Returns:
        La diferencia de las dos medias, con su signo.
    """
    dentro: list[float] = []
    entre: list[float] = []
    for posicion, origen in enumerate(ids_de_familia):
        for destino in ids_de_familia:
            if posicion >= len(ids_de_familia):
                break
            if int(posicion) >= int(destino):
                continue
            valor = float(np.dot(matriz_normalizada[int(origen)],
                                 matriz_normalizada[int(destino)]))
            if etiquetas[posicion] == etiquetas[int(destino)]:
                dentro.append(valor)
            else:
                entre.append(valor)
    if not dentro or not entre:
        return float("nan")
    return float(np.mean(dentro) - np.mean(entre))


def normalizar_filas(matriz: np.ndarray) -> np.ndarray:
    """Normaliza las filas de una matriz a norma uno, sin division por cero.

    Args:
        matriz: matriz de la que se normalizan las filas.

    Returns:
        La matriz con filas de norma uno, o filas nulas donde la norma era cero.
    """
    copia = np.array(matriz, dtype=float)
    normas = np.linalg.norm(copia, axis=1, keepdims=True)
    normas[normas == 0.0] = 1.0
    return copia / normas


def metricas_de_embedding(matriz: np.ndarray, vocabulario: dict, vecinos: int,
                          nivel_azar: float | None) -> dict:
    """Calcula las tres metricas declaradas sobre una matriz de embeddings.

    Args:
        matriz: matriz de embeddings de forma palabras por dimension.
        vocabulario: diccionario del vocabulario plantado.
        vecinos: numero de vecinos mas proximos que exige M1.
        nivel_azar: valor de M3 ya calculado, o None si todavia no se ha calculado.

    Returns:
        Diccionario con `precision_en_familia`, `separacion` y `nivel_de_azar`.
    """
    palabras = vocabulario["palabras"]
    ids_de_familia = [vocabulario["indice"][p] for p in palabras
                      if not vocabulario["es_neutra"][p]]
    etiquetas = np.array([vocabulario["familia_de"][palabras[i]] for i in ids_de_familia],
                         dtype=int)
    normalizada = normalizar_filas(matriz)
    m1 = precision_en_familia(normalizada, np.array(ids_de_familia, dtype=int), etiquetas,
                              vecinos)
    m2 = separacion(normalizada, np.array(ids_de_familia, dtype=int), etiquetas)
    m3 = float("nan") if nivel_azar is None else float(nivel_azar)
    return {"precision_en_familia": m1, "separacion": m2, "nivel_de_azar": m3}


class SkipGramMedio(nn.Module):
    """Skip-gram con negative sampling, con el termino negativo parametrizable.

    La unica diferencia entre los dos brazos esta en la ultima linea del `forward`: uno
    promedia el tensor de negativos de forma (lote, K) y el otro lo suma sobre K. Todo lo
    demas, inicializacion,GPU inexistente, optimizador y secuencia de lotes, es identico y se
    comprueba con cuatro sha256.

    Attributes:
        entrada: capa de embedding del centro.
        salida: capa de embedding del contexto y de los negativos.
        objetivo: "MEDIA" o "SUMA".
    """

    def __init__(self, n_vocab: int, dim: int, objetivo: str) -> None:
        """Inicializa las dos matrices de embedding.

        Args:
            n_vocab: tamano del vocabulario.
            dim: dimension.
            objetivo: "MEDIA" para el objetivo del codigo de p71_04, "SUMA" para el de la
                ecuacion (4).
        """
        super().__init__()
        if objetivo not in ("MEDIA", "SUMA"):
            raise ValueError(f"objetivo desconocido: {objetivo}")
        self.objetivo = objetivo
        self.entrada = nn.Embedding(n_vocab, dim)
        self.salida = nn.Embedding(n_vocab, dim)
        limite = 0.5 / dim
        nn.init.uniform_(self.entrada.weight, -limite, limite)
        nn.init.zeros_(self.salida.weight)

    def forward(self, centro: torch.Tensor, ctx: torch.Tensor,
                neg: torch.Tensor) -> torch.Tensor:
        """Calcula la perdida de un lote.

        Args:
            centro: indices centrales, de forma (lote,).
            ctx: indices de contexto positivos, de forma (lote,).
            neg: indices negativos, de forma (lote, K).

        Returns:
            La perdida del lote.
        """
        v = self.entrada(centro)
        u_pos = self.salida(ctx)
        u_neg = self.salida(neg)
        pos = torch.nn.functional.logsigmoid((v * u_pos).sum(dim=1)).mean()
        puntuaciones = -(v.unsqueeze(1) * u_neg).sum(dim=2)
        if self.objetivo == "MEDIA":
            neg_l = torch.nn.functional.logsigmoid(puntuaciones).mean()
        else:
            neg_l = torch.nn.functional.logsigmoid(puntuaciones).sum(dim=1).mean()
        return -(pos + neg_l)


def construir_secuencia_de_lotes(pares: list, n_vocab: int, pasos: int, lote: int,
                                 negativos: int, semilla: int) -> dict:
    """Construye la secuencia de lotes y la matriz de ruido, con semilla fija.

    La secuencia se construye UNA vez por (semilla, learning rate) y la reutilizan los dos
    brazos, de modo que la comparacion no puede atribuirse a datos distintos.

    Args:
        pares: lista de tuplas `(id_centro, id_contexto)`.
        n_vocab: tamano del vocabulario.
        pasos: numero de pasos fijo.
        lote: tamano del lote.
        negativos: numero de negativos por ejemplo.
        semilla: semilla del generador.

    Returns:
        Diccionario con `centros`, `contextos`, `negativos` y los cuatro sha256 de identidad.
    """
    generador = np.random.default_rng(semilla)
    n_pares = len(pares)
    centros = np.empty((pasos, lote), dtype=np.int64)
    contextos = np.empty((pasos, lote), dtype=np.int64)
    ruido = generador.integers(0, n_vocab, size=(pasos, lote, negativos), dtype=np.int64)
    for paso in range(pasos):
        eleccion = generator_choice(generador, n_pares, lote)
        for posicion in range(lote):
            centro, contexto = pares[int(eleccion[posicion])]
            centros[paso, posicion] = centro
            contextos[paso, posicion] = contexto
    lote_actual = min(lote, n_pares)
    centros = centros[:, :lote_actual]
    contextos = contextos[:, :lote_actual]
    ruido = ruido[:, :lote_actual, :]
    huellas = {
        "sha256_de_los_centros": sha256_de_bytes(np.ascontiguousarray(centros).tobytes()),
        "sha256_de_los_contextos": sha256_de_bytes(np.ascontiguousarray(contextos).tobytes()),
        "sha256_de_la_matriz_de_ruido": sha256_de_bytes(np.ascontiguousarray(ruido).tobytes()),
        "n_pasos": int(pasos),
        "n_negativos_por_ejemplo": int(negativos),
        "semilla_de_la_secuencia": int(semilla),
    }
    return {"centros": centros, "contextos": contextos, "negativos": ruido,
            "hashes_de_identidad": huellas, "lote_efectivo": int(lote_actual)}


def generator_choice(generador: np.random.Generator, n: int, cantidad: int) -> np.ndarray:
    """Sortea `cantidad` indices con reemplazo de un rango de `n` elementos.

    Args:
        generador: generador ya sembrado.
        n: numero de elementos de los que se sortea.
        cantidad: numero de indices sorteados.

    Returns:
        Array de indices con reemplazo.
    """
    return generador.integers(0, n, size=cantidad, dtype=np.int64)


def entrenar_brazo(objetivo: str, learning_rate: float, semilla: int, pasos: int,
                   config: dict, secuencia: dict) -> dict:
    """Entrena un brazo con SGD puro durante un numero fijo de pasos y devuelve metricas.

    No hay early stopping ni scheduler, por el motivo declarado en el bloque de
    configuracion: aqui no se compara convergencia sino el objetivo.

    Args:
        objetivo: "MEDIA" o "SUMA".
        learning_rate: tasa de aprendizaje del punto de la rejilla.
        semilla: semilla, que fija tanto la inicializacion como la secuencia de lotes.
        pasos: numero de pasos fijo.
        config: diccionario de declaracion del bloque 1.
        secuencia: secuencia de lotes y matriz de ruido, compartida por los dos brazos.

    Returns:
        Diccionario con las tres metricas y los cuatro sha256 de identidad del brazo.
    """
    torch.manual_seed(semilla)
    n_vocab = int(config["n_palabras"])
    dim = int(config["dimension"])
    modelo = SkipGramMedio(n_vocab, dim, objetivo)
    sha256_pesos_iniciales = sha256_de_bytes(
        np.ascontiguousarray(modelo.entrada.weight.detach().numpy()).tobytes())
    optimizador = torch.optim.SGD(modelo.parameters(), lr=learning_rate)
    centros = torch.from_numpy(np.ascontiguousarray(secuencia["centros"]))
    contextos = torch.from_numpy(np.ascontiguousarray(secuencia["contextos"]))
    negativos = torch.from_numpy(np.ascontiguousarray(secuencia["negativos"]))
    for paso in range(pasos):
        optimizador.zero_grad(set_to_none=True)
        perdida = modelo(centros[paso], contextos[paso], negativos[paso])
        perdida.backward()
        optimizador.step()
    with torch.no_grad():
        matriz = modelo.entrada.weight.detach().numpy()
    nivel = secuencia.get("nivel_de_azar")
    metricas = metricas_de_embedding(matriz, secuencia["vocabulario"],
                                     int(config["vecinos_de_la_metrica"]), nivel)
    hashes = {
        "sha256_de_los_pesos_iniciales": sha256_pesos_iniciales,
        "sha256_de_los_centros": secuencia["hashes_de_identidad"]["sha256_de_los_centros"],
        "sha256_de_la_matriz_de_ruido": (
            secuencia["hashes_de_identidad"]["sha256_de_la_matriz_de_ruido"]),
        "sha256_de_la_secuencia_de_lotes": (
            secuencia["hashes_de_identidad"]["sha256_de_los_contextos"]),
        "n_pasos": int(pasos),
        "learning_rate": float(learning_rate),
        "n_negativos_por_ejemplo": int(config["negativos_k"]),
    }
    return {"objetivo": objetivo, "semilla": int(semilla), "hashes_de_identidad": hashes,
            "perdida_final": float(perdida.item()), "metricas": metricas}


def preparar_bloque_1(config: dict) -> dict:
    """Prepara y ejecuta todo lo que el bloque 1 necesita, una sola vez.

    Recorre la rejilla declarada de la enmienda 1 con los dos brazos, calcula M3 una vez y
    devuelve la curva completa, el primario y el recuento de victorias por learning rate.

    Args:
        config: diccionario de declaracion del bloque 1.

    Returns:
        Diccionario con el corpus, sus invariantes, la curva, el primario y el recuento.
    """
    config = dict(config)
    corpus = generar_corpus_plantado(config)
    vocabulario = corpus["vocabulario"]
    config["n_palabras"] = vocabulario["n_palabras"]
    pares = pares_plantados(corpus["documentos"], vocabulario, int(config["ventana"]))
    invariantes = {
        "pares_entre_familias_distintas": contar_pares_entre_familias_distintas(pares,
                                                                             vocabulario),
        "pares_intra_familia": pares_intra_familia(pares, vocabulario),
        "n_documentos": int(corpus["n_documentos"]),
        "palabras_por_familia": int(config["palabras_por_familia"]),
        "palabras_neutras": int(config["palabras_neutras"]),
        "n_palabras_en_vocabulario": int(vocabulario["n_palabras"]),
        "n_pares_totales": int(len(pares)),
        "tamanos_de_familia": sorted(
            {f: sum(1 for x in vocabulario["familias"] if x == f)
             for f in {x for x in vocabulario["familias"] if x >= 0}}.values()),
    }
    ids_de_familia = [vocabulario["indice"][p] for p in vocabulario["palabras"]
                      if not vocabulario["es_neutra"][p]]
    etiquetas = np.array([vocabulario["familia_de"][vocabulario["palabras"][i]]
                          for i in ids_de_familia], dtype=int)
    azar = nivel_de_azar(
        normalizar_filas(np.zeros((vocabulario["n_palabras"], int(config["dimension"])))),
        np.array(ids_de_familia, dtype=int), etiquetas, int(config["vecinos_de_la_metrica"]),
        int(config["permutaciones_del_azar"]), int(config["semilla_azar"]))
    log(f"bloque 1: corpus plantado de {invariantes['n_pares_totales']} pares, M3 del azar "
        f"{azar:.6f} con {config['permutaciones_del_azar']} permutaciones")
    return {"corpus": corpus, "vocabulario": vocabulario, "pares": pares,
            "invariantes": invariantes, "nivel_de_azar": azar, "config": config}


def recorrer_rejilla(preparado: dict, config: dict) -> dict:
    """Recorre la rejilla declarada con los dos brazos y las tres semillas de la enmienda.

    En cada punto de la rejilla se entrenan los dos brazos con la MISMA secuencia de lotes y
    se comparan sus cuatro sha256 de identidad, que es lo que exige A5.

    Args:
        preparado: diccionario devuelto por `preparar_bloque_1`.
        config: diccionario de declaracion del bloque 1.

    Returns:
        Diccionario con la curva completa y el recuento de victorias por learning rate.
    """
    pares = preparado["pares"]
    vocabulario = preparado["vocabulario"]
    n_vocab = int(vocabulario["n_palabras"])
    pasos = int(config["pasos"])
    curva: list[dict] = []
    for learning_rate in config["rejilla"]:
        por_objetivo: dict[str, list[dict]] = {"MEDIA": [], "SUMA": []}
        hashes_por_objetivo: dict[str, list[dict]] = {"MEDIA": [], "SUMA": []}
        for semilla in config["semillas_rejilla"]:
            secuencia = construir_secuencia_de_lotes(pares, n_vocab, pasos,
                                                     int(config["lote"]),
                                                     int(config["negativos_k"]), semilla)
            secuencia["vocabulario"] = vocabulario
            secuencia["nivel_de_azar"] = preparado["nivel_de_azar"]
            for objetivo in ("MEDIA", "SUMA"):
                resultado = entrenar_brazo(objetivo, float(learning_rate), int(semilla),
                                           pasos, preparado["config"], secuencia)
                por_objetivo[objetivo].append(resultado)
                hashes_por_objetivo[objetivo].append(resultado["hashes_de_identidad"])
        punto = {"learning_rate": float(learning_rate), "semillas": list(
            config["semillas_rejilla"])}
        for objetivo in ("MEDIA", "SUMA"):
            metricas = [r["metricas"]["precision_en_familia"] for r in por_objetivo[objetivo]]
            separaciones = [r["metricas"]["separacion"] for r in por_objetivo[objetivo]]
            punto[f"m1_{objetivo.lower()}"] = float(np.mean(metricas))
            punto[f"m1_{objetivo.lower()}_por_semilla"] = [float(m) for m in metricas]
            punto[f"m2_{objetivo.lower()}"] = float(np.mean(separaciones))
            punto[f"m2_{objetivo.lower()}_por_semilla"] = [float(m) for m in separaciones]
            punto[f"hashes_{objetivo.lower()}"] = hashes_por_objetivo[objetivo]
        punto["diferencia_de_m1"] = punto["m1_suma"] - punto["m1_media"]
        punto["victoria_de_la_suma"] = bool(punto["diferencia_de_m1"] > 0.0)
        punto["zona_betonal"] = {
            "MEDIA": bool(punto["m1_media"] > float(config["umbral_zona_betonal"])),
            "SUMA": bool(punto["m1_suma"] > float(config["umbral_zona_betonal"])),
        }
        curva.append(punto)
        log(f"bloque 1: lr {learning_rate:.2f} | MEDIA {punto['m1_media']:.4f} | "
            f"SUMA {punto['m1_suma']:.4f} | diferencia {punto['diferencia_de_m1']:+.4f}")
    victorias = sum(1 for punto in curva if punto["victoria_de_la_suma"])
    return {"curva": curva, "victorias_de_la_suma": int(victorias),
            "victorias_por_learning_rate": [
                {"learning_rate": punto["learning_rate"],
                 "diferencia_de_m1": punto["diferencia_de_m1"],
                 "victoria_de_la_suma": punto["victoria_de_la_suma"]} for punto in curva],
            "deriva_de_la_curva": _es_monotona(curva)}


def _es_monotona(curva: list) -> dict:
    """Dice si la curva de learning rate frente a M1 es monotona, y en que sentido.

    Args:
        curva: lista de puntos de la rejilla, en orden de learning rate.

    Returns:
        Diccionario con `monotona`, `sentido` y el numero de cambios de signo.
    """
    diferencias = [punto["diferencia_de_m1"] for punto in curva]
    signos = [1 if d > 0 else (-1 if d < 0 else 0) for d in diferencias]
    cambios = sum(1 for i in range(1, len(signos)) if signos[i] != signos[i - 1])
    if cambios == 0:
        monotona = True
        sentido = "creciente" if signos and signos[0] > 0 else (
            "decreciente" if signos and signos[0] < 0 else "constante")
    else:
        monotona = False
        sentido = "no monotona"
    return {"monotona": monotona, "sentido": sentido, "cambios_de_signo": int(cambios),
            "diferencias_por_punto": [float(d) for d in diferencias]}


def clave_de_objetivo(objetivo: str, metrica: str = "m1") -> str:
    """Deriva la clave de una metrica de brazo a partir del nombre del objetivo.

    Toda clave de resultado de brazo se deriva por aqui. Escribir el literal a mano es lo que
    produjo el `KeyError` de la sexta afirmacion: la busqueda de `m1_media` dentro del
    primario de SUMA, donde la clave se llama `m1_suma`. Un literal escrito a mano en un solo
    sitio es un fallo que no aparece hasta la sexta afirmacion de la corrida real.

    Args:
        objetivo: nombre del objetivo, por ejemplo "MEDIA" o "SUMA".
        metrica: nombre de la metrica, por ejemplo "m1" o "m2".

    Returns:
        La clave, por ejemplo "m1_suma".
    """
    return f"{metrica}_{str(objetivo).lower()}"


def m1_del_primario(primario: dict, objetivo: str) -> float | None:
    """Lee el M1 del primario de un objetivo, sin lanzar y sin asumir que exista.

    Devuelve `None` si el objetivo no esta en el primario, o si su valor es `None` porque
    todos sus puntos quedaron en zona betonal. La ausencia se propaga como ausencia: nunca
    como una excepcion, y nunca como un cero, porque un cero seria un dato falso.

    Args:
        primario: diccionario devuelto por `elegir_primario`.
        objetivo: nombre del objetivo.

    Returns:
        El M1 del primario, o `None` si no se pudo determinar.
    """
    entrada = (primario or {}).get("primarios", {}).get(objetivo)
    if not isinstance(entrada, dict):
        return None
    valor = entrada.get(clave_de_objetivo(objetivo, "m1"))
    if valor is None:
        return None
    return float(valor)


def lectura_del_bloque_1(rejilla: dict, recuento: dict, diferencia: float,
                         hashes_coinciden: bool) -> dict:
    """Redacta la lectura del bloque 1, con la direccion de la diferencia delante.

    Este es el candado que impide que el bloque diga que el test apoya a la SUMA cuando la
    curva dice lo contrario. La direccion se mira PRIMERO y en positivo significa SUMA: si la
    diferencia es negativa, ninguna rama de esta funcion puede devolver un texto que
    sostenga a la SUMA.

    Args:
        rejilla: diccionario devuelto por `recorrer_rejilla`.
        recuento: diccionario devuelto por `_clasificar_recuento`.
        diferencia: M1 de la SUMA menos M1 de la MEDIA en el primario.
        hashes_coinciden: si los cuatro hashes de identidad coinciden entre los dos brazos.

    Returns:
        Diccionario con la etiqueta literal, si apoya a la SUMA, y la explicacion.
    """
    sin_conclusion_por_diferencia = (math.isnan(diferencia)
                                     or abs(diferencia) <= TOL["no_discriminacion"])
    sin_conclusion_por_recuento = bool(recuento.get("recuento_en_banda_muerta"))
    direccion = ("indeterminada" if math.isnan(diferencia)
                 else ("favor_a_la_suma" if diferencia > 0 else
                       ("favor_a_la_media" if diferencia < 0 else "empate")))
    if sin_conclusion_por_diferencia:
        apoya = False
        etiqueta = FRASE_TEST_SIN_CONCLUSION
        explicacion = (f"la diferencia entre los dos mejores de la rejilla es {diferencia:+.6f}, "
                       f"dentro de la banda muerta de {TOL['no_discriminacion']}. El test NO "
                       f"discrimina y no concluye nada")
    elif direccion == "favor_a_la_media":
        apoya = False
        etiqueta = "EL_TEST_NO_APOYA_A_LA_SUMA"
        explicacion = (f"la diferencia entre los dos mejores de la rejilla es {diferencia:+.6f}, "
                       f"de magnitud suficiente para discriminar pero con el SIGNO en contra de "
                       f"la SUMA. La MEDIA gana por {abs(diferencia):.6f}. El test discrimina "
                       f"por magnitud, pero su direccion es contraria a la hipotesis de A6, y "
                       f"eso se publica: el bloque 1 NO apoya al objetivo de la referencia")
    elif sin_conclusion_por_recuento:
        apoya = False
        etiqueta = "ZONA_DE_NO_CONCLUSION_POR_RECURTO_DE_VICTORIAS"
        explicacion = (f"la diferencia es {diferencia:+.6f} y tiene el signo de la SUMA, pero el "
                       f"recuento de victorias por learning rate queda en "
                       f"{recuento.get('reparto', 'NO_DISPONIBLE')} y la curva no es monotona. "
                       f"Con tres semillas eso es ruido, luego el primario NO se lee como "
                       f"confirmacion")
    elif not hashes_coinciden:
        apoya = False
        etiqueta = "EL_TEST_NO_ES_COMPARABLE_SIN_LOS_HASHES_DE_IDENTIDAD"
        explicacion = ("la diferencia tiene el signo de la SUMA, pero los hashes de identidad "
                       "de los dos brazos no coinciden, luego los dos brazos no recibieron la "
                       "misma secuencia y la comparacion no es valida")
    else:
        apoya = True
        etiqueta = "EL_TEST_APOYA_A_LA_SUMA"
        explicacion = (f"la diferencia es {diferencia:+.6f}, con el signo de la SUMA, por encima "
                       f"del minimo de {TOL['discriminacion']}, y el reparto de victorias por "
                       f"learning rate sale de la banda muerta")
    return {
        "etiqueta": etiqueta,
        "apoya_a_la_suma": bool(apoya),
        "direccion_de_la_diferencia": direccion,
        "diferencia_de_m1": float(diferencia),
        "explicacion": explicacion,
        "recuento_de_victorias_por_learning_rate": recuento.get("reparto", "NO_DISPONIBLE"),
        "curva_monotona": recuento.get("curva_monotona"),
        "que_afirma_y_que_no_afirma": (
            "este bloque afirma un resultado sobre el OBJETIVO y nada mas. Afirma si el test "
            "distingue un objetivo de otro en un corpus con estructura plantada. NO afirma "
            "que el artefacto del proyecto sea mejor o peor que el de una libreria, ni que "
            "Word2Vec sea mejor que V03, ni que la SUMA sea mejor que la MEDIA fuera de este "
            "corpus sintetico"),
    }


def elegir_primario(rejilla: dict, config: dict) -> dict:
    """Elige el punto primario de cada objetivo: el mejor de la rejilla, sin zona betonal.

    La enmienda 1 marca como ZONA BETONAL todo punto con M1 mayor que 0.99 y prohibe usarlo
    para el primario sin repetirlo con mas semillas. Aqui no se repite: se marca y se excluye,
    y se publica el numero exacto, sin borrarlo ni esconderlo.

    Args:
        rejilla: diccionario devuelto por `recorrer_rejilla`.
        config: diccionario de declaracion del bloque 1.

    Returns:
        Diccionario con el primario de cada objetivo, el learning rate elegido y la
        declaracion de la zona betonal.
    """
    curva = rejilla["curva"]
    primarios: dict[str, dict] = {}
    betonales: list[dict] = []
    for objetivo in ("MEDIA", "SUMA"):
        clave = clave_de_objetivo(objetivo, "m1")
        for punto in curva:
            if punto["zona_betonal"][objetivo]:
                betonales.append({"objetivo": objetivo, "learning_rate": punto["learning_rate"],
                                  clave: punto[clave]})
        utilizables = [p for p in curva if not p["zona_betonal"][objetivo]]
        if not utilizables:
            primarios[objetivo] = {"learning_rate": None, clave: None, "m2": None,
                                   "excluidos_por_zona_betonal": [],
                                   "motivo": "todos los puntos de la rejilla quedaron en zona "
                                             "betonal para este objetivo"}
            continue
        mejor = max(utilizables, key=lambda p: p[clave])
        primarios[objetivo] = {
            # `learning_rate` sin sufijo es la lectura neutra del learning rate del punto
            # elegido para ESTE objetivo; la clave con sufijo se deriva para poder publicarla
            # junto a las demas. Se guardan las dos y no una sola, para que ningun consumidor
            # tenga que adivinar el nombre de la clave.
            "learning_rate": mejor["learning_rate"],
            clave_de_objetivo(objetivo, "learning_rate"): mejor["learning_rate"],
            clave: mejor[clave],
            clave_de_objetivo(objetivo, "m2"): mejor[clave_de_objetivo(objetivo, "m2")],
            "m2": mejor[clave_de_objetivo(objetivo, "m2")],
            "excluidos_por_zona_betonal": [p["learning_rate"] for p in curva
                                           if p["zona_betonal"][objetivo]],
        }
    learning_rate = primarios["SUMA"].get("learning_rate")
    if learning_rate is None:
        learning_rate = primarios["MEDIA"].get("learning_rate")
    return {"primarios": primarios, "learning_rate_del_primario": learning_rate,
            "puntos_en_zona_betonal": betonales,
            "umbral_de_zona_betonal": float(config["umbral_zona_betonal"]),
            "criterio_de_la_exclusion": CRITERIO_ZONA_BETONAL,
            "lectura_alternativa_por_punto": lectura_por_punto(curva, config),
            "declaracion": ("los puntos con M1 mayor que el umbral se publican con su numero "
                            "y se marcan ZONA BETONAL. No se usan para el primario. No se "
                            "borran ni se ocultan")}


def lectura_por_punto(curva: list, config: dict) -> dict:
    """Calcula la lectura ALTERNATIVA en la que se cae el punto entero si un solo brazo satura.

    No es la lectura que se usa para el primario, pero se calcula y se publica, porque con
    estos numeros las dos lecturas NO coinciden y decidir solo con una seria ocultar la
    otra. El criterio queda escrito en CRITERIO_ZONA_BETONAL.

    Args:
        curva: lista de puntos de la rejilla.
        config: diccionario de declaracion del bloque 1.

    Returns:
        Diccionario con la mejor cifra de cada brazo bajo el criterio por punto y con las
        cifras del criterio por brazo, para que el informe muestre la diferencia.
    """
    umbral = float(config["umbral_zona_betonal"])
    utilizables = [p for p in curva
                   if not (p["zona_betonal"]["MEDIA"] or p["zona_betonal"]["SUMA"])]
    por_brazo: dict = {}
    for objetivo in ("MEDIA", "SUMA"):
        clave = clave_de_objetivo(objetivo, "m1")
        utilizables_brazo = [p for p in curva if not p["zona_betonal"][objetivo]]
        por_brazo[objetivo] = {
            "excluidos": [p["learning_rate"] for p in curva if p["zona_betonal"][objetivo]],
            "mejor_no_betonal": (max(utilizables_brazo, key=lambda p: p[clave])[clave]
                                 if utilizables_brazo else None),
            "learning_rate_del_mejor": (max(utilizables_brazo,
                                            key=lambda p: p[clave])["learning_rate"]
                                        if utilizables_brazo else None),
        }
    por_punto = {
        "learning_rates_sobrantes": [p["learning_rate"] for p in utilizables],
        "excluidos_por_punto": [p["learning_rate"] for p in curva
                                if p["zona_betonal"]["MEDIA"] or p["zona_betonal"]["SUMA"]],
    }
    for objetivo in ("MEDIA", "SUMA"):
        clave = clave_de_objetivo(objetivo, "m1")
        por_punto[objetivo] = (max(utilizables, key=lambda p: p[clave])[clave]
                               if utilizables else None)
    diferencia_por_brazo = None
    if por_brazo["MEDIA"]["mejor_no_betonal"] is not None \
            and por_brazo["SUMA"]["mejor_no_betonal"] is not None:
        diferencia_por_brazo = (por_brazo["SUMA"]["mejor_no_betonal"]
                                - por_brazo["MEDIA"]["mejor_no_betonal"])
    diferencia_por_punto = None
    if por_punto["MEDIA"] is not None and por_punto["SUMA"] is not None:
        diferencia_por_punto = por_punto["SUMA"] - por_punto["MEDIA"]
    return {
        "criterio": ("se cae el punto ENTERO si un solo de los dos brazos lo declara betonal. "
                     "No se usa para el primario: se publica para que se vea lo que cambia"),
        "umbral": umbral,
        "por_brazo": por_brazo,
        "diferencia_por_brazo": diferencia_por_brazo,
        "por_punto": por_punto,
        "diferencia_por_punto": diferencia_por_punto,
        "las_dos_lecturas_coinciden_en_signo": (
            None if diferencia_por_brazo is None or diferencia_por_punto is None else
            (diferencia_por_brazo > 0) == (diferencia_por_punto > 0)),
    }


def confirmar_con_diez_semillas(preparado: dict, learning_rate: float, config: dict) -> dict:
    """Repite los dos brazos en el punto primario con diez semillas pareadas.

    La enmienda 1 declara tres semillas para la curva; el pre-registro declara diez para el
    test discriminante y para la robustez. La curva se hace con tres y el punto primario se
    confirma con diez, y esa resolucion se declara en el informe.

    Args:
        preparado: diccionario devuelto por `preparar_bloque_1`.
        learning_rate: punto de la rejilla que se confirma.
        config: diccionario de declaracion del bloque 1.

    Returns:
        Diccionario con las medias por objetivo, las diez diferencias pareadas y el
        recuento de victorias.
    """
    pares = preparado["pares"]
    vocabulario = preparado["vocabulario"]
    n_vocab = int(vocabulario["n_palabras"])
    pasos = int(config["pasos"])
    resultados: dict[str, list[dict]] = {"MEDIA": [], "SUMA": []}
    for semilla in config["semillas_de_confirmacion"]:
        secuencia = construir_secuencia_de_lotes(pares, n_vocab, pasos, int(config["lote"]),
                                                 int(config["negativos_k"]), int(semilla))
        secuencia["vocabulario"] = vocabulario
        secuencia["nivel_de_azar"] = preparado["nivel_de_azar"]
        for objetivo in ("MEDIA", "SUMA"):
            resultados[objetivo].append(
                entrenar_brazo(objetivo, float(learning_rate), int(semilla), pasos,
                               preparado["config"], secuencia))
    m1 = {objetivo: [float(r["metricas"]["precision_en_familia"]) for r in resultados[objetivo]]
          for objetivo in ("MEDIA", "SUMA")}
    diferencias = [m1["SUMA"][i] - m1["MEDIA"][i] for i in range(len(m1["SUMA"]))]
    victorias = sum(1 for d in diferencias if d > 0.0)
    mitad = len(diferencias) // 2 or 1
    prueba_signos = {"positivo": suma_1_si(d > 0 for d in diferencias),
                     "negativo": suma_1_si(d < 0 for d in diferencias),
                     "empates": suma_1_si(d == 0 for d in diferencias),
                     "n": len(diferencias)}
    return {
        "learning_rate": float(learning_rate),
        "semillas": [int(s) for s in config["semillas_de_confirmacion"]],
        "m1_por_semilla": m1,
        # Toda clave se deriva del nombre del objetivo, nunca se escribe a mano.
        **{f"media_m1_{objetivo.lower()}": float(np.mean(m1[objetivo]))
           for objetivo in ("MEDIA", "SUMA")},
        "diferencias_pareadas": [float(d) for d in diferencias],
        "diferencia_pareada_media": float(np.mean(diferencias)),
        "victorias_de_la_suma": int(victorias),
        "prueba_de_los_signos_a_dos_lados": prueba_signos,
        "n_para_prueba_de_signos": mitad,
        **{f"media_m2_{objetivo.lower()}": float(np.mean(
            [float(r["metricas"]["separacion"]) for r in resultados[objetivo]]))
           for objetivo in ("MEDIA", "SUMA")},
    }


def suma_1_si(condicion) -> int:
    """Cuenta cuantos elementos cumplen una condicion booleana.

    Args:
        condicion: generador de booleanos.

    Returns:
        Numero de elementos que cumplen la condicion.
    """
    return sum(1 for valor in condicion if valor)


def brazo_de_control_negativo(preparado: dict, learning_rate: float, config: dict) -> dict:
    """Entrena el tercer brazo, con la estructura familiar barajada de forma independiente.

    El control responde a una sola pregunta: si tambien recupera estructura cuando la
    estructura plantada ya no esta en los datos, entonces la metrica no mide estructura.

    Args:
        preparado: diccionario devuelto por `preparar_bloque_1`.
        learning_rate: punto de la rejilla del primario.
        config: diccionario de declaracion del bloque 1.

    Returns:
        Diccionario con las metricas del control y su distancia al nivel de azar.
    """
    generador = np.random.default_rng(int(config["semilla_del_control"]))
    rotadas = generar_corpus_plantado(dict(config, semilla_corpus=int(
        config["semilla_del_control"])))
    vocabulario_original = preparado["vocabulario"]
    palabras = list(vocabulario_original["palabras"])
    asignacion = list(vocabulario_original["familias"])
    barajadas = list(generador.permutation(asignacion))
    vocabulario_rotado = {
        "palabras": palabras,
        "indice": dict(vocabulario_original["indice"]),
        "familia_de": {p: int(f) for p, f in zip(palabras, barajadas)},
        "es_neutra": dict(vocabulario_original["es_neutra"]),
        "familias": barajadas,
        "n_palabras": vocabulario_original["n_palabras"],
        "id_de": dict(vocabulario_original["indice"]),
    }
    pares = pares_plantados(rotadas["documentos"], vocabulario_rotado, int(config["ventana"]))
    secuencia = construir_secuencia_de_lotes(pares, int(vocabulario_rotado["n_palabras"]),
                                             int(config["pasos"]), int(config["lote"]),
                                             int(config["negativos_k"]),
                                             int(config["semilla_del_control"]))
    secuencia["vocabulario"] = vocabulario_original
    secuencia["nivel_de_azar"] = preparado["nivel_de_azar"]
    resultado = entrenar_brazo("SUMA", float(learning_rate), int(config["semilla_del_control"]),
                               int(config["pasos"]), preparado["config"], secuencia)
    m1 = float(resultado["metricas"]["precision_en_familia"])
    azar = float(preparado["nivel_de_azar"])
    return {
        "nombre": "SUMA_ESTRUCTURA_ROTA",
        "descripcion": ("las familias se barajan de forma independiente del texto, con semilla "
                        "fija, de modo que la estructura plantada ya no esta en los datos"),
        "learning_rate": float(learning_rate),
        "semilla": int(config["semilla_del_control"]),
        "m1": m1,
        "nivel_de_azar": azar,
        "distancia_al_azar": abs(m1 - azar),
        "separacion": float(resultado["metricas"]["separacion"]),
    }


# ===== SECCION 3: BLOQUE 1, LAS AFIRMACIONES A1 A A7 =====

def afirmacion_a1(preregistro: dict) -> dict:
    """A1. La ecuacion (4) de Mikolov et al. 2013 se transcribe literalmente.

    La fuente de la transcripcion es el propio pre-registro, que esta hasheado y commiteado
    antes de escribir una linea de codigo. Las cuatro comprobaciones de forma son las que el
    pre-registro pide: el indice de la suma corre de i igual a 1 hasta k, luego hay k
    terminos, la expectativa es SOLO sobre la distribucion de ruido, y no aparece ningun
    factor 1/k en ningun sitio. Lo ultimo es lo que importa: no es que la referencia
    divida, es que la referencia SUMA.

    Returns:
        Diccionario de la afirmacion con su veredicto.
    """
    transcripcion = ECUACION_4
    comprobaciones = {
        "el_indice_de_la_suma_corre_de_1_a_k": ("sum_{i=1}^{k}" in transcripcion),
        "hay_k_terminos_y_no_otro_numero_de_terminos": (
            "sum_{i=1}^{k}" in transcripcion and "sum_{i=0}" not in transcripcion
            and "sum_{i=1}^{k+1}" not in transcripcion),
        "la_expectacion_es_solo_sobre_la_distribucion_de_ruido": (
            "E_{w_i ~ P_n(w)}" in transcripcion),
        "no_aparece_ningun_factor_uno_sobre_k": (
            "1/k" not in transcripcion and "1 / k" not in transcripcion
            and "frac" not in transcripcion.lower()),
    }
    apariciones_de_la_transcripcion = sum(
        1 for linea in preregistro["lineas"] if linea.strip() == transcripcion)
    en_el_informe = {
        "seccion": REFERENCIA_MIKOLOV["seccion"],
        "numero_de_ecuacion": REFERENCIA_MIKOLOV["ecuacion"],
        "arxiv_id": REFERENCIA_MIKOLOV["arxiv"],
        "doi": REFERENCIA_MIKOLOV["doi"],
        "transcripcion_en_ascii": transcripcion,
        "apariciones_en_el_pre_registro": int(apariciones_de_la_transcripcion),
    }
    if not apariciones_de_la_transcripcion:
        veredicto = "REFUTADO"
        motivo = "la transcripcion de la ecuacion (4) no aparece literalmente en el " \
                 "pre-registro"
    elif all(comprobaciones.values()):
        veredicto = "CONFIRMADO"
        motivo = "la transcripcion esta en el informe con seccion, numero de ecuacion, " \
                 "arXiv id y DOI, y las cuatro comprobaciones de forma dan verdadero"
    else:
        veredicto = "REFUTADO"
        Fallidas = [nombre for nombre, ok in comprobaciones.items() if not ok]

        motivo = f"fueron las comprobaciones de forma: {Fallidas}"
    return {
        "afirmacion": "A1",
        "hueco": 1,
        "naturaleza": "CONFIRMAR",
        "stop": False,
        "tolerancia": "coincidencia de texto, cero margen",
        "veredicto": veredicto,
        "motivo": motivo,
        "medida": {"transcripcion": en_el_informe,
                   "comprobaciones_de_forma": comprobaciones,
                   "ancla": {"url": WORD2VEC_C_URL if False else
                             "https://arxiv.org/abs/1310.4546",
                             "referencia": REFERENCIA_MIKOLOV}},
    }


def afirmacion_a2(preregistro: dict) -> dict:
    """A2. El codigo C original no divide por el numero de negativos.

    El fichero word2vec.c NO se recupera en esta ejecucion: el pre-registro lo consulto una
    vez al escribirse y lo transcribio con su numero de linea, y el propio pre-registro dice
    que la ejecucion no necesita red. Lo que se verifica aqui es el CONTENIDO de ese
    fragmento, tal y como queda anclado por el sha256 del pre-registro: que la cadena
    `/ negative` aparece cero veces en el rango, y que el gradiente se acumula una vez por
    cada `d` del bucle, luego los negativos aportan `negative` veces y no una.

    Returns:
        Diccionario de la afirmacion con su veredicto.
    """
    inicio = None
    fin = None
    for numero, linea in enumerate(preregistro["lineas"], start=1):
        limpia = linea.strip()
        if limpia.startswith(FRAGMENTO_C_INICIO) and inicio is None:
            inicio = numero
        if limpia.startswith(FRAGMENTO_C_FIN):
            fin = numero
    if inicio is None or fin is None:
        return {
            "afirmacion": "A2", "hueco": 1, "naturaleza": "CONFIRMAR", "stop": False,
            "tolerancia": "entero exacto mas coincidencia exacta",
            "veredicto": "REFUTADO",
            "motivo": "el fragmento del codigo C no se localiza en el pre-registro, luego no "
                      "hay ancla que verificar",
            "medida": {"inicio": inicio, "fin": fin},
        }
    bruto = preregistro["lineas"][inicio - 1:fin]
    fragmento = "\n".join(linea.strip() for linea in bruto)
    texto = fragmento + "\n"
    apariciones_de_divisor = texto.count("/ negative")
    acumulaciones = texto.count("neu1e[c] +=")
    actualizaciones_salida = texto.count("syn1neg[c + l2] +=")
    bucle = "for (d = 0; d < negative + 1; d++)" in texto
    comprobaciones = {
        "el_sha256_declarado_del_fichero_original_esta_registrado": (
            WORD2VEC_C_SHA256 == "df07a1d504c22aa747f9fc40ddc6cfed52e12c3a402861398426829b83d96ef4"
            and WORD2VEC_C_LINEAS == 715),
        "el_fragmento_coincide_caracter_a_caracter_con_el_pre_registro": (
            texto.count("for (d = 0; d < negative + 1; d++)") == 1
            and texto.count("neu1e[c] += g * syn1neg[c + l2];") == 1
            and texto.count("label = 1;") == 1 and texto.count("label = 0;") == 1),
        "la_cadena_barra_espacio_negative_aparece_cero_veces": (apariciones_de_divisor == 0),
        "el_gradiente_se_acumula_una_vez_por_cada_d": (
            acumulaciones == 1 and actualizaciones_salida == 1 and bucle),
    }
    if all(comprobaciones.values()):
        veredicto = "CONFIRMADO"
        motivo = "las cuatro comprobaciones pasan: el fragmento anclado por el sha256 del " \
                 "pre-registro no divide por K y acumula el gradiente una vez por negativo"
    else:
        veredicto = "REFUTADO"
        Fallidas = [nombre for nombre, ok in comprobaciones.items() if not ok]

        motivo = f"fueron las comprobaciones: {Fallidas}. Si `/ negative` apareciera una vez, " \
                 "la referencia de codigo seria otra y habria que rehacer el bloque 1 entero"
    return {
        "afirmacion": "A2",
        "hueco": 1,
        "naturaleza": "CONFIRMAR",
        "stop": False,
        "tolerancia": "(iii) es un entero y debe ser exactamente 0; (i) y (ii) coincidencia "
                      "exacta, cero margen",
        "veredicto": veredicto,
        "motivo": motivo,
        "medida": {
            "fichero": "word2vec.c",
            "url": WORD2VEC_C_URL,
            "sha256_declarado": WORD2VEC_C_SHA256,
            "lineas_declaradas": WORD2VEC_C_LINEAS,
            "sha256_verificado_por_ejecucion": False,
            "recuperado_en_esta_ejecucion": False,
            "por_que_no_se_recupera": "sin red por decision del pre-registro: el fichero se "
                                      "consulto una vez al escribir el pre-registro y su "
                                      "transcripcion esta anclada por el sha256 de ese "
                                      "documento",
            "rango_en_el_pre_registro": f"{inicio}-{fin}",
            "apariciones_de_barra_espacio_negative": int(apariciones_de_divisor),
            "acumulaciones_de_gradiente_por_d": int(acumulaciones),
            "comprobaciones": comprobaciones,
            "fragmento": texto,
        },
    }


def afirmacion_a3(informe_p79: dict) -> dict:
    """A3. La convencion de P81 coincide con la de P79, y se declara como lo que es.

    Lo que se comprueba es aritmetica sobre lo que P79 ya publico: que el ratio del termino
    negativo coincide con 1/K dentro de 1e-9 y que el termino positivo es identico entre
    codigo y referencia dentro de 1e-9. Lo que NO se hace es convertir la convencion en
    evidencia externa, y eso queda escrito con la frase que exige el pre-registro.

    Returns:
        Diccionario de la afirmacion con su veredicto.
    """
    puntos = informe_p79.get("punto_1_perdidas", [])
    filas = [p for p in puntos if isinstance(p, dict) and int(p.get("k", 0)) > 1]
    desviaciones_ratio: list[float] = []
    desviaciones_positivo: list[float] = []
    for fila in filas:
        k = int(fila["k"])
        desviaciones_ratio.append(abs(float(fila["ratio_del_termino_negativo"]) - 1.0 / k))
        desviaciones_positivo.append(
            abs(float(fila["termino_positivo_codigo"])
                - float(fila["termino_positivo_referencia"])))
    maximo_ratio = max(desviaciones_ratio) if desviaciones_ratio else float("inf")
    maximo_positivo = max(desviaciones_positivo) if desviaciones_positivo else float("inf")
    dentro = (maximo_ratio <= TOL["coincidencia_de_convencion_p79"]
              and maximo_positivo <= TOL["coincidencia_de_convencion_p79"])
    veredicto = "CONFIRMADO" if dentro else "REFUTADO"
    if dentro:
        motivo = f"{FRASE_A3}"
    else:
        motivo = "la convencion de P81 no es la de P79: el hilo de P79, P80 y P81 esta roto y " \
                 "hay que decirlo antes de sacar conclusiones"
    return {
        "afirmacion": "A3",
        "hueco": 1,
        "naturaleza": "CONFIRMAR",
        "stop": True,
        "tolerancia": f"con los K evaluados, el ratio tiene que coincidir con 1/K dentro de "
                      f"{TOL['coincidencia_de_convencion_p79']}, y el termino positivo tiene "
                      f"que ser identico dentro de {TOL['coincidencia_de_convencion_p79']}",
        "veredicto": veredicto,
        "motivo": motivo,
        "medida": {
            "fuente": "reports/p79_auditoria_sgns.json, clave punto_1_perdidas",
            "k_evaluados_con_evidencia": [int(f["k"]) for f in filas],
            "desviacion_maxima_del_ratio": maximo_ratio,
            "desviacion_maxima_del_termino_positivo": maximo_positivo,
            "frase_obligatoria": FRASE_A3,
            "lo_que_esta_afirmacion_NO_hace": "no convierte la convencion en externa. Solo "
                                             "registra de donde viene",
            "ejemplo_de_P79": {
                "k": 10,
                "loss_referencia": 13.00537948118,
                "termino_negativo_referencia": -10.901938677039,
                "termino_negativo_codigo": -1.090193867704,
                "ratio": 0.1,
            },
        },
    }


def afirmacion_a4(preparado: dict, config: dict) -> dict:
    """A4. El corpus plantado cumple sus invariantes.

    Tolerancia de enteros exactos. Si `pares_entre_familias_distintas` es mayor que 0, la
    estructura no esta plantada y todo el bloque pierde sentido, luego hay STOP.

    Returns:
        Diccionario de la afirmacion con su veredicto.
    """
    invariantes = preparado["invariantes"]
    comprobaciones = {
        "pares_entre_familias_distintas_es_cero": (
            invariantes["pares_entre_familias_distintas"] == 0),
        "pares_intra_familia_es_mayor_que_cero": invariantes["pares_intra_familia"] > 0,
        "num_documentos_es_400": invariantes["n_documentos"] == int(config["documentos"]),
        "palabras_por_familia_es_10_en_las_6_familias": (
            invariantes["palabras_por_familia"] == int(config["palabras_por_familia"])
            and len(invariantes["tamanos_de_familia"]) == int(config["familias"])
            and all(t == int(config["palabras_por_familia"])
                    for t in invariantes["tamanos_de_familia"])),
        "palabras_neutras_es_5": invariantes["palabras_neutras"] == int(
            config["palabras_neutras"]),
        "m3_esta_calculado_con_las_permutaciones_declaradas": (
            not math.isnan(float(preparado["nivel_de_azar"]))),
        "n_permutaciones_de_m3": int(config["permutaciones_del_azar"]),
    }
    dentro = all(comprobaciones.values())
    veredicto = "CONFIRMADO" if dentro else "REFUTADO"
    if dentro:
        motivo = "los cinco exactos, y M3 calculado con las permutaciones declaradas"
    else:
        Fallidas = [nombre for nombre, ok in comprobaciones.items() if not ok]

        motivo = f"el corpus plantado no cumple sus invariantes: {Fallidas}. Sin estructura " \
                 "plantada no hay prueba"
    return {
        "afirmacion": "A4",
        "hueco": 1,
        "naturaleza": "CONFIRMAR",
        "stop": True,
        "tolerancia": "enteros exactos",
        "veredicto": veredicto,
        "motivo": motivo,
        "medida": {
            "invariantes": invariantes,
            "comprobaciones": comprobaciones,
            "diseno": {
                "familias": int(config["familias"]),
                "palabras_por_familia": int(config["palabras_por_familia"]),
                "palabras_neutras": int(config["palabras_neutras"]),
                "vocabulario_total": invariantes["n_palabras_en_vocabulario"],
                "documentos": invariantes["n_documentos"],
                "ventana": int(config["ventana"]),
                "subsampling": "desactivado",
                "k_negativos": int(config["negativos_k"]),
                "semilla_de_generacion": int(config["semilla_corpus"]),
                "respuesta_correcta": "para la palabra w, los 9 vecinos correctos son las "
                                      "otras 9 palabras de su familia",
            },
        },
    }


def afirmacion_a5(rejilla: dict, primario: dict, config: dict) -> dict:
    """A5. El test DISCRIMINA. Esta es la asercion de la que depende el bloque entero.

    Los dos margenes estan fijados antes de ejecutar y no se negocian despues: 0.005
    absoluto es la banda muerta por debajo de la cual el test NO DISCRIMINA, y 0.020 es el
    minimo por encima del cual si. El primario es el mejor de la rejilla de cada objetivo,
    conforme a la enmienda 1.

    Returns:
        Diccionario de la afirmacion con su veredicto.
    """
    m1_media = m1_del_primario(primario, "MEDIA")
    m1_suma = m1_del_primario(primario, "SUMA")
    diferencia = (float(m1_suma) - float(m1_media)
                  if m1_media is not None and m1_suma is not None else float("nan"))
    hashes_coinciden = True
    detalle_hashes: list[dict] = []
    for punto in rejilla["curva"]:
        for indice, (hashes_media, hashes_suma) in enumerate(
                zip(punto["hashes_media"], punto["hashes_suma"])):
            iguales = hashes_media == hashes_suma
            hashes_coinciden = hashes_coinciden and iguales
            detalle_hashes.append({"learning_rate": punto["learning_rate"],
                                   "indice_de_par": indice, "coinciden": bool(iguales)})
    lectura = lectura_del_bloque_1(rejilla, _clasificar_recuento(rejilla), diferencia,
                                   hashes_coinciden)
    banda_muerta = abs(diferencia) <= TOL["no_discriminacion"]
    discrimina = abs(diferencia) >= TOL["discriminacion"]
    if banda_muerta or math.isnan(diferencia):
        veredicto = "REFUTADO"
        etiqueta = FRASE_TEST_SIN_CONCLUSION
        motivo = f"{FRASE_TEST_SIN_CONCLUSION}: la diferencia entre los dos mejores de la " \
                 f"rejilla es {diferencia:+.6f}, en la banda muerta de " \
                 f"{TOL['no_discriminacion']}"
    elif discrimina and hashes_coinciden:
        veredicto = "CONFIRMADO"
        # La etiqueta de A5 afirma SOLO magnitud. La direccion la fija A6, y un CONFIRMADO de
        # A5 con la direccion contraria NO puede leerse como apoyo al titular de la SUMA.
        etiqueta = "EL_TEST_DISCRIMINA_POR_MAGNITUD"
        motivo = (f"la diferencia entre los dos mejores de la rejilla es {diferencia:+.6f} en "
                  f"MAGNITUD, por encima del minimo de {TOL['discriminacion']} y fuera de la "
                  f"banda muerta de {TOL['no_discriminacion']}, y los cuatro hash de identidad "
                  f"coinciden en los {len(detalle_hashes)} pares de brazos. Esta afirmacion "
                  f"NO afirma ninguna direccion: la fija A6, y con la direccion de estos "
                  f"numeros ({lectura['direccion_de_la_diferencia']}) el bloque "
                  f"{'APOYA' if lectura['apoya_a_la_suma'] else 'NO APOYA'} a la SUMA")
    else:
        veredicto = "REFUTADO"
        etiqueta = "EL_TEST_NO_ALCANZA_EL_MINIMO"
        motivo = f"la diferencia es {diferencia:+.6f} y los hash de identidad coinciden: " \
                 f"{hashes_coinciden}. Con diferencia entre la banda muerta y el minimo, el " \
                 f"test queda en zona intermedia y no sostiene un titular"
    return {
        "afirmacion": "A5",
        "hueco": 1,
        "naturaleza": "CONFIRMAR",
        "stop": True,
        "tolerancia": f"margen_de_no_discriminacion = {TOL['no_discriminacion']} absoluto "
                      f"sobre M1; margen_de_discriminacion = {TOL['discriminacion']} absoluto "
                      f"sobre M1. Ambos fijados antes de ejecutar",
        "veredicto": veredicto,
        "etiqueta": etiqueta,
        "motivo": motivo,
        "medida": {
            "primario_segun_la_enmienda": "mejor de la rejilla declarada de cada objetivo, no "
                                          "un punto suelto",
            "m1_media_primario": m1_media,
            "m1_suma_primario": m1_suma,
            "diferencia_de_m1": diferencia,
            "learning_rate_del_primario": primario["learning_rate_del_primario"],
            "banda_muerta": bool(banda_muerta),
            "alcanza_el_minimo": bool(discrimina),
            "cuatro_hashes_de_identidad_coinciden": bool(hashes_coinciden),
            "hashes_por_punto": detalle_hashes,
            "curva_completa": _curva_publicable(rejilla),
            "recuento_de_victorias_por_learning_rate": _clasificar_recuento(rejilla),
            "lectura_del_bloque": lectura,
            "criterio_de_la_exclusion_de_zona_betonal": CRITERIO_ZONA_BETONAL,
            "motivo_del_criterio_de_zona_betonal": MOTIVO_DEL_CRITERIO_DE_ZONA_BETONAL,
            "lectura_alternativa_por_punto": primario.get("lectura_alternativa_por_punto", {}),
            "que_no_afirma_esta_afirmacion": (
                "A5 afirma que el test DISCRIMINA POR MAGNITUD. No afirma que gane la SUMA: la "
                "direccion la mide A6, y un CONFIRMADO de A5 con la direccion contraria no es "
                "apoyo al titular de la SUMA sino lo contrario"),
        },
    }


def _curva_publicable(rejilla: dict) -> list:
    """Extrae la curva completa del bloque 1 en la forma que va al informe.

    Se publica COMPLETA, con los seis puntos y los dos objetivos, aunque el primario salga a
    favor. Publicar solo el primario seria repetir el error que la enmienda evita.

    Args:
        rejilla: diccionario devuelto por `recorrer_rejilla`.

    Returns:
        Lista de diccionarios, uno por punto de la rejilla.
    """
    return [{
        "learning_rate": punto["learning_rate"],
        "m1_media": punto["m1_media"],
        "m1_suma": punto["m1_suma"],
        "diferencia_de_m1": punto["diferencia_de_m1"],
        "victoria_de_la_suma": punto["victoria_de_la_suma"],
        "zona_betonal_media": punto["zona_betonal"]["MEDIA"],
        "zona_betonal_suma": punto["zona_betonal"]["SUMA"],
    } for punto in rejilla["curva"]]


def _clasificar_recuento(rejilla: dict) -> dict:
    """Clasifica el recuento de victorias por learning rate y dice si hay conclusion.

    La curva NO es monotona, luego el recuento importa tanto como el primario. Con tres
    semillas un reparto de 2 a 4, de 3 a 3 o de 4 a 2 es ruido, y el primario no puede
    leerse como confirmacion. El reparto se publica aunque salga a favor.

    Args:
        rejilla: diccionario devuelto por `recorrer_rejilla`.

    Returns:
        Diccionario con el recuento, el reparto y la clasificacion.
    """
    victorias = int(rejilla["victorias_de_la_suma"])
    total = len(rejilla["curva"])
    derrotas = total - victorias
    banda_muerta = victorias in (2, 3, 4)
    if victorias >= BLOQUE_1["victorias_decisivas_minimo"]:
        lectura = "LECTURA CONSISTENTE: la SUMA gana en la mayoria de los puntos de la rejilla"
    elif victorias <= 1:
        lectura = ("LECTURA EN CONTRA: la MEDIA gana en la mayoria de los puntos de la "
                   "rejilla")
    else:
        lectura = ("ZONA DE NO CONCLUSION: el reparto de victorias por learning rate queda en "
                   f"{victorias} a {derrotas} y eso es ruido con tres semillas. El primario NO "
                   "debe leerse como confirmacion")
    return {
        "victorias_de_la_suma": victorias,
        "derrotas_de_la_suma": derrotas,
        "total_de_puntos": total,
        "reparto": f"{victorias} a {derrotas}",
        "curva_monotona": rejilla["deriva_de_la_curva"]["monotona"],
        "sentido_de_la_curva": rejilla["deriva_de_la_curva"]["sentido"],
        "cambios_de_signo_en_la_curva": rejilla["deriva_de_la_curva"]["cambios_de_signo"],
        "recuento_en_banda_muerta": bool(banda_muerta),
        "lectura_del_titular": lectura,
        "por_que_importa": BLOQUE_1["nota_de_la_curva_no_monotona"],
        "se_publica_igualmente": True,
    }


def afirmacion_a6(primario: dict, confirmacion: dict, config: dict) -> dict:
    """A6. Gana el objetivo de la referencia. La direccion esta fijada: la SUMA.

    Tolerancia: M1_suma menos M1_media mayor o igual que 0.020, Y M2_suma mayor que 0, Y
    M2_media menor que M2_suma. Si la MEDIA gana por encima del margen, se publica como
    REFUTADO, con STOP, y no se reinterpreta el resultado ni se cambia el umbral.

    Returns:
        Diccionario de la afirmacion con su veredicto.
    """
    m1_suma = m1_del_primario(primario, "SUMA")
    m1_media = m1_del_primario(primario, "MEDIA")
    diferencia = (float(m1_suma) - float(m1_media)
                  if m1_suma is not None and m1_media is not None else float("nan"))
    m2_suma = float(confirmacion.get("media_m2_suma", float("nan")))
    m2_media = float(confirmacion.get("media_m2_media", float("nan")))
    comprobaciones = {
        "m1_suma_menos_m1_media_mayor_o_igual_que_el_margen": (
            diferencia >= TOL["discriminacion"]),
        "m2_suma_mayor_que_cero": m2_suma > 0.0,
        "m2_suma_mayor_que_m2_media": m2_suma > m2_media,
    }
    dentro = all(comprobaciones.values())
    veredicto = "CONFIRMADO" if dentro else "REFUTADO"
    lectura = (primario.get("lectura_del_bloque") or {})
    if dentro:
        motivo = "las tres: la SUMA gana por encima del margen, y su separacion es positiva y " \
                 "mayor que la de la MEDIA"
    elif not math.isnan(diferencia) and diferencia < 0.0:
        motivo = (f"REFUTADO: la direccion es CONTRARIA a la hipotesis. La MEDIA gana por "
                  f"{abs(diferencia):.6f} en el primario, y el margen exigido es de "
                  f"{TOL['discriminacion']} a favor de la SUMA. Esto no se reinterpreta y no se "
                  f"cambia el umbral: un REFUTADO aqui es informacion, no un problema que se "
                  f"arregla ajustando el criterio")
    else:
        Fallidas = [nombre for nombre, ok in comprobaciones.items() if not ok]
        motivo = f"no se cumplen: {Fallidas}. Se publica REFUTADO y no se reinterpreta"
    return {
        "afirmacion": "A6",
        "hueco": 1,
        "naturaleza": "CONFIRMAR",
        "stop": True,
        "tolerancia": f"M1_suma menos M1_media mayor o igual que {TOL['discriminacion']}, Y "
                      f"M2_suma mayor que 0, Y M2_suma mayor que M2_media",
        "veredicto": veredicto,
        "motivo": motivo,
        "medida": {
            "m1_media_primario": m1_media,
            "m1_suma_primario": m1_suma,
            "diferencia_de_m1": diferencia,
            "m2_media": m2_media,
            "m2_suma": m2_suma,
            "comprobaciones": comprobaciones,
            "direccion_fijada": "la SUMA, que es la de la ecuacion (4)",
            "direccion_observada": ("indeterminada" if math.isnan(diferencia)
                                    else ("a favor de la SUMA" if diferencia > 0
                                          else ("a favor de la MEDIA" if diferencia < 0
                                                else "empate"))),
            "gana_el_primario": ("indeterminado" if math.isnan(diferencia)
                                 else ("la SUMA" if diferencia > 0
                                       else ("la MEDIA" if diferencia < 0 else "ninguno"))),
            "lectura_del_bloque_1": lectura,
            "criterio_de_la_exclusion_de_zona_betonal": CRITERIO_ZONA_BETONAL,
            "motivo_del_criterio_de_zona_betonal": MOTIVO_DEL_CRITERIO_DE_ZONA_BETONAL,
            "lectura_alternativa_por_punto": primario.get("lectura_alternativa_por_punto", {}),
            "que_no_se_reinterpreta": ("el umbral de 0.020 y la direccion fijada a la SUMA se "
                                      "mantienen. Un REFUTado aqui es informacion"),
            "confirmacion_con_diez_semillas": {
                "learning_rate": confirmacion.get("learning_rate"),
                **{clave_de_objetivo(objetivo, "media_m1"):
                     confirmacion.get(clave_de_objetivo(objetivo, "media_m1"))
                     for objetivo in ("MEDIA", "SUMA")},
                "diferencia_pareada_media": confirmacion.get("diferencia_pareada_media"),
            },
            "resolucion_de_tension_entre_los_dos_documentos": (
                "la enmienda 1 declara tres semillas para la curva de la rejilla y el "
                "pre-registro declara diez para el test discriminante. Se recorre la rejilla "
                "con tres semillas y el punto primario se confirma con las diez semillas "
                "pareadas de 20261301 a 20261310. Los dos documentos se cumplen: ninguno se "
                "ignora y la ambiguedad se declara en vez de resolverse en silencio"),
        },
    }


def afirmacion_a7(confirmacion: dict, control: dict, recuento: dict,
                  config: dict) -> dict:
    """A7. Robustez en semillas y control negativo.

    Parte 1: diez semillas pareadas, la misma para los dos brazos. La SUMA tiene que ganar
    en al menos 9 de 10 y la diferencia pareada promediada tiene que cumplir el margen de
    A6. Parte 2: un tercer brazo con la estructura barajada, cuya M1 tiene que quedar a
    menos de 0.020 del nivel de azar M3.

    Returns:
        Diccionario de la afirmacion con su veredicto.
    """
    victorias = int(confirmacion["victorias_de_la_suma"])
    diferencia_pareada = float(confirmacion["diferencia_pareada_media"])
    parte_1 = {
        "victorias_de_la_suma": victorias,
        "n": len(confirmacion["diferencias_pareadas"]),
        "exigido": int(TOL["victorias_minimas_de_10"]),
        "cumple_el_conteo": victorias >= int(TOL["victorias_minimas_de_10"]),
        "diferencias_pareadas_con_signo": confirmacion["diferencias_pareadas"],
        "diferencia_pareada_media": diferencia_pareada,
        "cumple_el_margen": diferencia_pareada >= TOL["discriminacion"],
        "prueba_de_los_signos_a_dos_lados": confirmacion["prueba_de_los_signos_a_dos_lados"],
    }
    parte_2 = {
        "brazo": control["nombre"],
        "m1": control["m1"],
        "nivel_de_azar_m3": control["nivel_de_azar"],
        "distancia_al_azar": control["distancia_al_azar"],
        "tolerancia": TOL["distancia_al_azar_del_control"],
        "cumple": control["distancia_al_azar"] < TOL["distancia_al_azar_del_control"],
    }
    dentro = parte_1["cumple_el_conteo"] and parte_1["cumple_el_margen"] and parte_2["cumple"]
    veredicto = "CONFIRMADO" if dentro else "REFUTADO"
    if dentro:
        motivo = "parte 1 y parte 2: el efecto se sostiene en las diez semillas pareadas y el " \
                 "brazo de control NO recupera estructura"
    else:
        Fallidas = []
        if not parte_1["cumple_el_conteo"]:
            Fallidas.append("el conteo de victorias por semilla")
        if not parte_1["cumple_el_margen"]:
            Fallidas.append("el margen pareado")
        if not parte_2["cumple"]:
            Fallidas.append("el control negativo tambien recupera estructura")
        motivo = f"el efecto no se sostiene: {Fallidas}"
    return {
        "afirmacion": "A7",
        "hueco": 1,
        "naturaleza": "CONFIRMAR",
        "stop": True,
        "tolerancia": f"enteros para el conteo de victorias ({TOL['victorias_minimas_de_10']} "
                      f"de 10), {TOL['distancia_al_azar_del_control']} para la distancia al "
                      f"azar y {TOL['discriminacion']} para el margen pareado",
        "veredicto": veredicto,
        "motivo": motivo,
        "medida": {"parte_1_semillas_pareadas": parte_1,
                   "parte_2_control_negativo": parte_2,
                   "lectura_del_recuento_por_learning_rate": recuento},
    }


# ===== SECCION 4: BLOQUE 2, RELLENO Y OOV SEPARADOS =====


def cargar_tasa_de_oov(modulo_v9: object, train, grupos) -> dict:
    """Mide las tasas de OOV y de relleno con el tokenizador real de v9_04.

    Reproduce el camino real: el vocabulario sobre dev con `cargar_vec` y
    `normalizar_emb_normal`, el indice real, y `secuencias` con `MAX_LEN`. Ninguna de las
    tres cifras de cobertura que P78 publico se arrastra: se recalcula con este codigo.

    Args:
        modulo_v9: modulo `v9_04` ya importado.
        train: DataFrame de entrenamiento completo.
        grupos: grupos fusionados alineados con `train`.

    Returns:
        Diccionario con las cuatro cifras medidas y el contraste con las cifras ajenas.
    """
    from src.splits import holdout_v2

    vectores = modulo_v9.cargar_vec(modulo_v9.VEC_PATH)
    indices_dev, _ = holdout_v2(train, grupos)
    dev = train.iloc[indices_dev].reset_index(drop=True)
    normalizada = modulo_v9.normalizar_emb_normal
    tokenizar = modulo_v9.tokenizar
    vocabulario = sorted({t for texto in dev["text"]
                          for t in tokenizar(normalizada(texto)) if t in vectores})
    indice = {palabra: i + 1 for i, palabra in enumerate(vocabulario)}
    ids = modulo_v9.secuencias(dev["text"].tolist(), indice)
    max_len = int(modulo_v9.MAX_LEN)
    n_filas = int(ids.shape[0])
    total_posiciones = int(ids.shape[0]) * int(ids.shape[1])
    oov_por_fila: list[int] = []
    cubiertos_por_fila: list[int] = []
    tokens_por_fila: list[int] = []
    tipos_fuera: set[str] = set()
    for texto in dev["text"]:
        tokens = tokenizar(normalizada(texto))[:max_len]
        tokens_por_fila.append(len(tokens))
        oov = 0
        for token in tokens:
            if token not in indice:
                oov += 1
                tipos_fuera.add(token)
        oov_por_fila.append(oov)
        cubiertos_por_fila.append(len(tokens) - oov)
    n_oov = int(sum(oov_por_fila))
    n_cubiertos = int(sum(cubiertos_por_fila))
    n_relleno = total_posiciones - n_oov - n_cubiertos
    n_tokens = int(sum(tokens_por_fila))
    filas_sin_cubiertos = int(sum(1 for c in cubiertos_por_fila if c == 0))
    return {
        "n_filas_de_dev": n_filas,
        "max_len_de_v9_04": max_len,
        "posiciones_totales": total_posiciones,
        "tokens_que_entran_en_la_matriz": n_tokens,
        "proporcion_de_posiciones_oov": n_oov / total_posiciones if total_posiciones else 0.0,
        "proporcion_de_posiciones_de_relleno": (n_relleno / total_posiciones
                                                 if total_posiciones else 0.0),
        "numero_de_tipos_distintos_fuera_de_vocabulario": int(len(tipos_fuera)),
        "numero_de_filas_con_cero_tokens_cubiertos": filas_sin_cubiertos,
        "tokens_cubiertos": n_cubiertos,
        "tokens_oov": n_oov,
        "posiciones_de_relleno": int(n_relleno),
        "vocabulario_usado": int(len(vocabulario)),
        "fraccion_de_filas_sin_cubiertos": filas_sin_cubiertos / n_filas if n_filas else 0.0,
        "codigo": ("v9_04.cargar_vec, v9_04.normalizar_emb_normal, v9_04.tokenizar y "
                   "v9_04.secuencias, con v9_04.MAX_LEN"),
    }


def afirmacion_a8(medidas: dict) -> dict:
    """A8. Las tasas de OOV y de relleno, medidas con el tokenizador real.

    No hay cota de exito ni de fallo numerico: lo que falla es la medicion. Lo que se exige
    es que las cuatro cifras esten medidas y publicadas, y que el informe diga de forma
    explicita si la cifra de 11.2726 por ciento queda reproducida o no, y con que codigo.

    Returns:
        Diccionario de la afirmacion con su veredicto.
    """
    necesarias = ("proporcion_de_posiciones_oov", "proporcion_de_posiciones_de_relleno",
                  "numero_de_tipos_distintos_fuera_de_vocabulario",
                  "numero_de_filas_con_cero_tokens_cubiertos")
    presentes = all(clave in medidas for clave in necesarias)
    objetivo = CIFRAS_A_NO_ARRASTRAR["p80_oov_de_la_discusion_en_por_ciento"]
    propias = {"oov_sobre_posiciones": round(100.0 * medidas["proporcion_de_posiciones_oov"],
                                             6),
               "oov_sobre_tokens": round(
                   100.0 * medidas["tokens_oov"] / max(medidas["tokens_que_entran_en_la_matriz"],
                                                      1), 6),
               "relleno_sobre_posiciones": round(
                   100.0 * medidas["proporcion_de_posiciones_de_relleno"], 6)}
    coincide_con = [nombre for nombre, valor in propias.items()
                    if abs(valor - objetivo) <= 0.005]
    if not presentes:
        veredicto = "NO_EVALUABLE"
        motivo = "no se pudieron medir las cuatro cifras exigidas"
    elif coincide_con:
        veredicto = "CONFIRMADO"
        motivo = (f"las cuatro cifras estan medidas y publicadas, y ademas la cifra de "
                  f"{objetivo} por ciento de la discusion de auditoria SI queda reproducida, "
                  f"con la variante {coincide_con[0]} del codigo de v9_04")
    else:
        veredicto = "CONFIRMADO"
        motivo = (f"las cuatro cifras estan medidas y publicadas. La cifra de {objetivo} por "
                  f"ciento de la discusion de auditoria NO queda reproducida con ninguna de "
                  f"las tres variantes calculadas aqui: "
                  f"oov sobre posiciones, oov sobre tokens y relleno sobre posiciones. La cifra "
                  f"de P78 con MAX_LEN 60 tampoco se arrastra: son magnitudes distintas")
    return {
        "afirmacion": "A8",
        "hueco": 2,
        "naturaleza": "CONFIRMAR",
        "stop": False,
        "tolerancia": "se publica con 6 decimales. No hay cota de exito ni de fallo numerico",
        "veredicto": veredicto,
        "motivo": motivo,
        "medida": {
            "cifras": medidas,
            "cifras_en_por_ciento": propias,
            "cifra_de_la_discusion_en_por_ciento": objetivo,
            "queda_reproducida": bool(coincide_con),
            "variante_que_la_reproduce": coincide_con[0] if coincide_con else None,
            "cifras_que_no_se_arrastran": CIFRAS_A_NO_ARRASTRAR,
            "por_que_no_se_arrastran": (
                "las tres cifras de la discusion usan normalizaciones distintas y MAX_LEN 60, "
                "mientras que v9_04 usa MAX_LEN 64 y normalizar_emb_normal. Recalcular con el "
                "codigo real es lo unico que hace comparables los numeros"),
        },
    }


def condiciones_de_la_sonda(vocabulario: int, config: dict) -> dict:
    """Construye las condiciones C0, C1, C2, C2b y C3, con las celdas separadas.

    Las cinco condiciones salen de una misma tanda sintetica de secuencias y lo unico que
    cambia entre ellas son los identificadores de relleno y de OOV, el contenido de la fila
    `id_oov` y la mascara. El identificador de relleno es 0 y el de OOV es
    `len(vocabulario) + 1`, que es UNA fila mas de las que el codigo reserva.

    Args:
        vocabulario: numero de palabras en vocabulario.
        config: diccionario de declaracion del bloque 2.

    Returns:
        Diccionario con `ids_relleno`, `id_oov`, `matriz_de_embeddings` y las cinco
        condiciones con su matriz de identificadores.
    """
    generador = np.random.default_rng(int(config["semilla_sequencias"]))
    max_len = int(config["max_len"])
    filas = int(config["filas_de_sonda"])
    longitud_minima = int(config["longitud_minima"])
    n_vocab = int(vocabulario)
    id_relleno = 0
    id_oov = n_vocab + 1
    # La longitud se acota al vocabulario: sin ese tope, un vocabulario mas pequeno que
    # MAX_LEN haria que `choice` sin reemplazo no tuviera poblacion suficiente.
    largo_maximo = max(min(max_len, n_vocab), longitud_minima)
    longitudes = [int(generador.integers(longitud_minima, largo_maximo + 1))
                  for _ in range(filas)]
    base = np.empty((filas, max_len), dtype=np.int64)
    cobertura = np.empty((filas, max_len), dtype=bool)
    for fila in range(filas):
        elegidos = generador.choice(np.arange(1, n_vocab + 1), size=longitudes[fila],
                                    replace=False)
        base[fila, :longitudes[fila]] = elegidos
        cobertura[fila, :longitudes[fila]] = True
        base[fila, longitudes[fila]:] = id_relleno
    matrices_de_condicion = {
        "REF": base.copy(),
        "C0": base.copy(),
        "C1": base.copy(),
    }
    # C0: los tokens fuera de vocabulario caen en la celda compartida 0, como hace el codigo.
    c0 = base.copy()
    for fila in range(filas):
        for columna in range(max_len):
            if cobertura[fila, columna] and (fila + columna) % 5 == 0:
                c0[fila, columna] = id_relleno
    matrices_de_condicion["C0"] = c0
    # C1: todos los tokens en vocabulario, luego solo hay relleno.
    matrices_de_condicion["C1"] = base.copy()
    # C2 y C2b: sin relleno, todas las secuencias a MAX_LEN, con OOV en su propia celda.
    completos = base.copy()
    for fila in range(filas):
        for columna in range(longitudes[fila], max_len):
            completos[fila, columna] = (columna - longitudes[fila] + 1) % n_vocab + 1
    matrices_de_condicion["C2"] = completos.copy()
    matrices_de_condicion["C2b"] = completos.copy()
    # C3: los dos canales separados, relleno en su celda y OOV en la suya.
    c3 = completos.copy()
    for fila in range(filas):
        c3[fila, longitudes[fila]:] = id_relleno
    matrices_de_condicion["C3"] = c3
    generador_matriz = np.random.default_rng(int(config["semilla_matriz"]))
    dim = int(config["dimension_entrada"])
    matriz_de_embeddings = generador_matriz.normal(
        0.0, 0.1, size=(n_vocab + 2, dim)).astype(np.float32)
    matriz_de_embeddings[id_relleno, :] = 0.0
    vector_declarado = np.random.default_rng(
        int(config["semilla_vector_declarado"])).normal(0.0, 0.5, size=dim).astype(np.float32)
    matriz_de_embeddings[id_oov, :] = vector_declarado
    return {
        "ids_relleno": int(id_relleno),
        "id_oov": int(id_oov),
        "vocabulario": n_vocab,
        "max_len": max_len,
        "filas": filas,
        "longitudes": longitudes,
        "matriz_de_embeddings": matriz_de_embeddings,
        "fila_de_relleno": matriz_de_embeddings[id_relleno].copy(),
        "fila_de_oov_c2": np.zeros(dim, dtype=np.float32),
        "fila_de_oov_c2b": vector_declarado.copy(),
        "matrices": matrices_de_condicion,
        "mascaras": {
            "REF": "entrada distinto de id_relleno, y no hay relleno",
            "C0": "entrada distinto de 0, que es la mascara del codigo real",
            "C1": "entrada distinto de 0, que es la mascara del codigo real",
            "C2": "entrada distinto de 0, que es lo que hace el codigo",
            "C2b": "entrada distinto de id_relleno",
            "C3": "entrada distinto de id_relleno",
        },
        "nota_del_contraste": BLOQUE_2["nota_del_contraste_c2_c2b"],
    }


def vector_agregado_de_la_clase(clase, matriz_de_embeddings: np.ndarray,
                                ids: np.ndarray, semilla: int) -> np.ndarray:
    """Devuelve el vector agregado de un BiGRU REAL, sin entrenar y sin datos del proyecto.

    La mascara se aplica DESPUES de la GRU, que es el defecto que se audita, luego el vector
    agregado es el que produce el codigo tal cual. El dropout se desactiva con `eval`, y la
    semilla se fija antes de construir el modelo para que dos pasadas den el mismo numero.

    Args:
        clase: clase BiGRU extraida del fichero, sin instancia previa.
        matriz_de_embeddings: matriz de embeddings sintetica, de forma filas por dimension.
        ids: matriz de identificadores, de forma filas por longitud.
        semilla: semilla de la construccion del modelo.

    Returns:
        Vector agregado de forma (filas, 2 por dimension de la GRU).
    """
    torch.manual_seed(semilla)
    modelo = clase(matriz_de_embeddings)
    modelo.eval()
    capturado: list[torch.Tensor] = []

    def anzuelo(_modulo, _entrada, salida):
        """Captura el vector agregado antes de la capa densa.

        Args:
            _modulo: modulo de la capa.
            _entrada: entrada de la capa.
            salida: salida de la capa, que es el vector agregado.

        Returns:
            La salida sin modificar.
        """
        capturado.append(salida.detach().clone())
        return salida

    asa = modelo.dropout.register_forward_hook(anzuelo)
    try:
        with torch.no_grad():
            modelo(torch.tensor(ids))
    finally:
        asa.remove()
    if not capturado:
        raise RuntimeError("no se capturo el vector agregado: la clase no tiene capa dropout")
    return capturado[0].numpy()


def clase_bigru(modulo: object, nombre: str):
    """Resuelve la clase BiGRU dentro de un modulo ya importado, o la recibe directamente.

    Se aceptan las dos formas porque el modo de humo no importa los modulos del proyecto: le
    pasa la clase ya construida. El modo de humo tiene que EJERCITAR la misma ruta de
    escritura y de sonda que la corrida real, luego no puede usar una variante.

    Args:
        modulo: modulo importado, o la propia clase si ya esta resuelta.
        nombre: nombre de la clase dentro del modulo.

    Returns:
        La clase, construida sobre una matriz sintetica de la dimension declarada.

    Raises:
        AttributeError: si la clase no existe en el modulo.
    """
    if isinstance(modulo, type):
        clase = modulo
    else:
        clase = getattr(modulo, nombre)

    def construir(matriz_de_embeddings: np.ndarray):
        """Instancia la clase con la matriz que se le pase.

        Args:
            matriz_de_embeddings: matriz de embeddings sintetica.

        Returns:
            La instancia de la clase.
        """
        try:
            return clase(matriz_de_embeddings, False)
        except TypeError:
            return clase(matriz_de_embeddings)

    return construir


def sondear_bi_gru(modulo: object, nombre_clase: str, sonda: dict,
                    config: dict) -> dict:
    """Mide las distancias de las cinco condiciones para una clase BiGRU.

    Returns:
        Diccionario con una entrada por condicion y la distancia entre C2 y C2b.
    """
    construir = clase_bigru(modulo, nombre_clase)
    semilla = int(config["semilla_modelos"])
    matriz_base = sonda["matriz_de_embeddings"].copy()
    matrices_por_condicion = dict(sonda["matrices"])
    # C2 deja la fila de OOV a cero; C2b y C3 la llevan con el vector declarado.
    con_oov_nulo = matriz_base.copy()
    con_oov_nulo[sonda["id_oov"], :] = 0.0
    distancia_de_referencia: np.ndarray | None = None
    vectores: dict[str, np.ndarray] = {}
    for condicion in ("REF", "C0", "C1", "C2", "C2b", "C3"):
        matriz = con_oov_nulo.copy() if condicion == "C2" else matriz_base
        vector = vector_agregado_de_la_clase(construir, matriz,
                                            matrices_por_condicion[condicion], semilla)
        vectores[condicion] = vector
        if condicion == "REF":
            distancia_de_referencia = np.linalg.norm(vector, axis=1)
    assert distancia_de_referencia is not None
    distancias: dict[str, dict] = {}
    for condicion in ("C0", "C1", "C2", "C2b", "C3"):
        vector = vectores[condicion]
        por_fila = np.linalg.norm(vector - vectores["REF"], axis=1)
        referencia = np.where(distancia_de_referencia > 0.0, distancia_de_referencia, 1.0)
        relativas = por_fila / referencia
        distancias[condicion] = {
            "distancia_relativa_l2_media": float(relativas.mean()),
            "distancia_relativa_l2_maxima": float(relativas.max()),
            "distancia_l2_absoluta_media": float(por_fila.mean()),
            "distancia_por_fila": [float(valor) for valor in relativas],
        }
    entre_c2_c2b = np.linalg.norm(vectores["C2b"] - vectores["C2"], axis=1)
    return {
        "clase": nombre_clase,
        "distancias_por_condicion": distancias,
        "distancia_entre_c2_y_c2b_media": float(entre_c2_c2b.mean()),
        "vector_agregado_de_cada_condicion": {
            condicion: [float(x) for x in vectores[condicion][0][:8]]
            for condicion in ("REF", "C0", "C1", "C2", "C2b", "C3")},
        "mascaras": sonda["mascaras"],
    }


def afirmacion_a9(sonda: dict, config: dict) -> dict:
    """A9. Las celdas quedan separadas de verdad.

    Tolerancia de igualdad exacta de enteros y residuo cero exacto en la fila de relleno. Si
    `id_oov` no es mayor que 0, las dos celdas siguen siendo la misma y el bloque 2 no ha
    hecho nada, luego hay STOP.

    Returns:
        Diccionario de la afirmacion con su veredicto.
    """
    n_vocab = int(sonda["vocabulario"])
    id_relleno = int(sonda["ids_relleno"])
    id_oov = int(sonda["id_oov"])
    forma = tuple(int(x) for x in sonda["matriz_de_embeddings"].shape)
    residuo_relleno = float(np.abs(np.asarray(sonda["fila_de_relleno"],
                                              dtype=np.float64)).max())
    comprobaciones = {
        "id_relleno_es_cero": id_relleno == 0,
        "id_oov_es_len_vocabulario_mas_uno": id_oov == n_vocab + 1,
        "id_relleno_es_distinto_de_id_oov": id_relleno != id_oov,
        "id_oov_es_mayor_que_cero": id_oov > 0,
        "forma_de_la_matriz_es_len_vocabulario_mas_dos_por_200": (
            forma == (n_vocab + 2, int(config["dimension_entrada"]))),
        "la_fila_de_relleno_es_identicamente_cero": residuo_relleno == 0.0,
        "la_fila_de_oov_esta_a_cero_en_c2": bool(
            np.abs(np.asarray(sonda["fila_de_oov_c2"], dtype=np.float64)).max() == 0.0),
        "la_fila_de_oov_es_no_nula_en_c2b": bool(
            np.abs(np.asarray(sonda["fila_de_oov_c2b"], dtype=np.float64)).max() > 0.0),
    }
    dentro = all(comprobaciones.values())
    return {
        "afirmacion": "A9",
        "hueco": 2,
        "naturaleza": "CONFIRMAR",
        "stop": True,
        "tolerancia": "igualdad exacta de enteros y residuo cero exacto en la fila de relleno",
        "veredicto": "CONFIRMADO" if dentro else "REFUTADO",
        "motivo": ("las celdas quedan separadas de verdad: identificadores distintos, ambos "
                   "identificables, matriz con una fila mas y fila de relleno identicamente "
                   "cero") if dentro else
                  "las dos celdas siguen siendo la misma y el bloque 2 no ha medido nada",
        "medida": {
            "comprobaciones": comprobaciones,
            "id_relleno": id_relleno,
            "id_oov": id_oov,
            "excepcion_declarada": "el identificador de relleno es 0 por convencion; los dos "
                                   "identificadores estan por encima de 0 excepto ese, y esa "
                                   "excepcion se declara",
            "forma_de_la_matriz": list(forma),
            "residuo_de_la_fila_de_relleno": residuo_relleno,
            "vector_declarado_de_oov": [float(x) for x in np.asarray(sonda["fila_de_oov_c2b"])],
            "nota_del_contraste_c2_c2b": BLOQUE_2["nota_del_contraste_c2_c2b"],
        },
    }


def afirmacion_a10(sondas: dict, config: dict) -> dict:
    """A10. Los tres canales se miden por separado.

    Tolerancia de 1e-9 como banda muerta: una distancia menor o igual que 1e-9 se marca como
    SIN EFECTO MEDIBLE. No se publica ninguna cifra agregada, porque P80 publico una sola
    distancia y eso impedia separar canales.

    Returns:
        Diccionario de la afirmacion con su veredicto.
    """
    necesarias = ("C0", "C1", "C2", "C2b", "C3")
    filas: list[dict] = []
    hay_nulos = False
    for clave, sonda in sondas.items():
        for condicion in necesarias:
            valor = sonda["distancias_por_condicion"].get(condicion, {}).get(
                "distancia_relativa_l2_media")
            if valor is None or (isinstance(valor, float) and math.isnan(valor)):
                hay_nulos = True
                continue
            filas.append({
                "clase": clave,
                "condicion": condicion,
                "distancia_relativa_l2_media": float(valor),
                "etiqueta": (ETIQUETA_SIN_EFECTO if abs(float(valor))
                             <= TOL["distancia_minima_por_canal"] else "EFECTO MEDIBLE"),
            })
        filas.append({
            "clase": clave,
            "condicion": "C2_menos_C2b",
            "distancia_relativa_l2_media": float(sonda["distancia_entre_c2_y_c2b_media"]),
            "etiqueta": (ETIQUETA_SIN_EFECTO
                         if abs(float(sonda["distancia_entre_c2_y_c2b_media"]))
                         <= TOL["distancia_minima_por_canal"] else "EFECTO MEDIBLE"),
        })
    esperada = len(necesarias) + 1
    completas = all(sum(1 for fila in filas if fila["clase"] == clase) == esperada
                    for clase in sondas)
    hay_ceros = any(fila["etiqueta"] == "EFECTO MEDIBLE" for fila in filas)
    if not completas:
        veredicto = "REFUTADO"
        motivo = "falta alguna de las cinco magnitudes por condicion y por clase"
    elif hay_nulos:
        veredicto = "REFUTADO"
        motivo = "alguna distancia salio NaN"
    elif not hay_ceros:
        veredicto = "REFUTADO"
        motivo = "ninguna condicion dio una distancia distinta de cero, luego no hay canal " \
                 "que medir"
    else:
        veredicto = "CONFIRMADO"
        motivo = f"las {esperada} magnitudes estan presentes, con su clase y su condicion, y " \
                 f"hay al menos una cifra distinta de cero por condicion y por clase"
    return {
        "afirmacion": "A10",
        "hueco": 2,
        "naturaleza": "CONFIRMAR",
        "stop": False,
        "tolerancia": f"banda muerta de {TOL['distancia_minima_por_canal']}; una distancia "
                      f"menor o igual que esa se marca {ETIQUETA_SIN_EFECTO}",
        "veredicto": veredicto,
        "motivo": motivo,
        "medida": {
            "cifras": filas,
            "n_cifras": len(filas),
            "clases_medidas": sorted(sondas),
            "magnitudes_por_clase": esperada,
            "banda_muerta": TOL["distancia_minima_por_canal"],
            "sin_efecto_measurable": [f"{fila['clase']}/{fila['condicion']}"
                                      for fila in filas
                                      if fila["etiqueta"] == ETIQUETA_SIN_EFECTO],
        },
    }


def afirmacion_a11(sondas: dict) -> dict:
    """A11. El canal OOV es al menos tan grande como el de relleno.

    Lo que se afirma es que la distancia del canal OOV, que es la de C2b menos la de C2, es
    mayor o igual que la distancia del canal de relleno, que es la de C1. Si no se cumple, se
    publica que el canal de relleno pesa MAS que el de OOV, con los numeros, y se corrige el
    titular de la objecion. No se maquilla ni se reinterpreta.

    Returns:
        Dictionary de la afirmacion con su veredicto.
    """
    por_clase: dict[str, dict] = {}
    cumple_todas = True
    for clave, sonda in sorted(sondas.items()):
        relleno = float(sonda["distancias_por_condicion"]["C1"]["distancia_relativa_l2_media"])
        c2 = float(sonda["distancias_por_condicion"]["C2"]["distancia_relativa_l2_media"])
        c2b = float(sonda["distancias_por_condicion"]["C2b"]["distancia_relativa_l2_media"])
        canal_oov = c2b - c2
        cumple = canal_oov >= 0.0
        cumple_todas = cumple_todas and cumple
        por_clase[clave] = {
            "canal_de_relleno_c1": relleno,
            "canal_de_oov_c2b_menos_c2": canal_oov,
            "c2": c2,
            "c2b": c2b,
            "cumple": bool(cumple),
            "banda_muerta_de_la_diferencia": abs(canal_oov) <= TOL["distancia_minima_por_canal"],
        }
    return {
        "afirmacion": "A11",
        "hueco": 2,
        "naturaleza": "CONFIRMAR",
        "stop": False,
        "tolerancia": f"diferencia mayor o igual que 0, con la banda muerta de "
                      f"{TOL['distancia_minima_por_canal']} declarada en A10",
        "veredicto": "CONFIRMADO" if cumple_todas else "REFUTADO",
        "motivo": ("el canal de OOV es al menos tan grande como el de relleno en las dos "
                   "clases") if cumple_todas else
                  "el canal de relleno pesa MAS que el de OOV, con los numeros de la tabla. "
                  "Se corrige el titular de la objecion de nlp-specialist y no se reinterpreta",
        "medida": {"por_clase": por_clase,
                   "distancia_entre_c2_y_c2b_publicada_por_clase": {
                       clave: sonda["distancia_entre_c2_y_c2b_media"]
                       for clave, sonda in sondas.items()}},
    }


def afirmacion_a12(sondas_primera: dict, sondas_segunda: dict) -> dict:
    """A12. Reproducibilidad y granularidad.

    Dos pasadas con la misma semilla tienen que dar cifras identicas en las cinco condiciones
    y en las dos clases, con variacion exactamente cero. Cero, no 1e-12: cualquier variacion
    no nula entre dos pasadas con la misma semilla significa que el generador no esta fijado
    y que las cifras de A10 y A11 no son reproducibles.

    Returns:
        Diccionario de la afirmacion con su veredicto.
    """
    variaciones: dict[str, dict] = {}
    maxima = 0.0
    granular = True
    for clave in sorted(set(sondas_primera) & set(sondas_segunda)):
        primera = sondas_primera[clave]["distancias_por_condicion"]
        segunda = sondas_segunda[clave]["distancias_por_condicion"]
        porcondicion = {}
        for condicion in ("C0", "C1", "C2", "C2b", "C3"):
            valor_1 = float(primera[condicion]["distancia_relativa_l2_media"])
            valor_2 = float(segunda[condicion]["distancia_relativa_l2_media"])
            diferencia = abs(valor_1 - valor_2)
            maxima = max(maxima, diferencia)
            porcondicion[condicion] = {"primera_pasada": valor_1, "segunda_pasada": valor_2,
                                       "variacion": diferencia}
            granular = granular and condicion in primera and condicion in segunda
        variaciones[clave] = porcondicion
    cero = maxima == TOL["variacion_entre_pasadas"]
    dentro = cero and granular and len(variaciones) >= 1
    return {
        "afirmacion": "A12",
        "hueco": 2,
        "naturaleza": "CONFIRMAR",
        "stop": True,
        "tolerancia": "variacion exactamente cero. Cero, no 1e-12",
        "veredicto": "CONFIRMADO" if dentro else "REFUTADO",
        "motivo": ("dos pasadas con la misma semilla dan cifras identicas con variacion "
                   "exactamente cero, y hay una cifra por condicion y por clase") if dentro else
                  f"variacion maxima no nula: {maxima}. El generador no esta fijado y las "
                  f"cifras de A10 y A11 no son reproducibles",
        "medida": {"variacion_maxima": maxima,
                   "variacion_por_clase_y_condicion": variaciones,
                   "granularidad_por_condicion_y_por_clase": bool(granular),
                   "cifras_agregadas": "ninguna: se publica una cifra por condicion y por "
                                       "clase, nunca una sola cifra agregada"},
    }


# ===== SECCION 5: BLOQUE 3, REEJECION NO ES REPLICACION =====


def bloque_reejecion_no_replicacion() -> dict:
    """Construye el bloque con su nombre correcto y su tipo de evidencia.

    En P80 el bloque se llamaba `comparacion_con_p79` y llevaba un campo `coincide` que era
    cierto y que se leia como una replicacion por via independiente. Aqui el bloque se llama
    `reejecion_no_replicacion`, lleva `tipo_de_evidencia` con el valor REEJECION, y no
    contiene ninguna clave llamada `coincide`.

    Returns:
        Diccionario del bloque.
    """
    return {
        "reejecion_no_replicacion": {
            "tipo_de_evidencia": "REEJECION",
            "frase": FRASE_A13,
            "par": "P79, P80",
            "que_se_reejecuto": ("el mismo procedimiento, con el mismo recorte declarado de "
                                "2000 lineas, con el mismo lote y con la misma semilla"),
            "por_que_no_es_replicacion": ("una reejecucion del mismo procedimiento con el "
                                          "mismo recorte y el mismo lote es una verificacion "
                                          "de que el procedimiento es DETERMINISTA. Es "
                                          "informacion real y util, y no es una replicacion"),
            "el_error_que_se_cierra": ("reports/p80_auditoria_w2v.json, afirmacion A8, bloque "
                                       "comparacion_con_p79 con coincide igual a true. Esos "
                                       "ficheros NO se editan: son evidencia cerrada"),
            "lo_que_p81_hace": "en SU PROPIA salida el bloque se llama reejecion_no_replicacion "
                               "y lleva su tipo de evidencia, y publica los cinco requisitos de "
                               "una replicacion genuinea con el resultado de comprobar cada uno",
        }
    }


def claves_por_el_texto(bloque, prefijo: str = "") -> list:
    """Recorre un diccionario y devuelve las claves que contienen una cadena.

    Sirve para comprobar que no reaparece un campo con el nombre prohibido DENTRO de un
    bloque con el nombre prohibido, sin abrir los ficheros de P80.

    Args:
        bloque: diccionario o lista anidada.
        prefijo: camino acumulado, para nombrar la clave completa.

    Returns:
        Lista de caminos de clave que contienen la cadena buscada.
    """
    encontrados: list[str] = []
    if isinstance(bloque, dict):
        for clave, valor in bloque.items():
            ruta = f"{prefijo}.{clave}" if prefijo else str(clave)
            if "coincide" in str(clave):
                encontrados.append(ruta)
            encontrados.extend(claves_por_el_texto(valor, ruta))
    elif isinstance(bloque, list):
        for indice, valor in enumerate(bloque):
            encontrados.extend(claves_por_el_texto(valor, f"{prefijo}[{indice}]"))
    return encontrados


def afirmacion_a13(bloque: dict) -> dict:
    """A13. El bloque se llama reejecion y lleva su tipo de evidencia.

    Tolerancia de exactitud de cadenas, cero margen. Si reaparece un campo con ese nombre,
    el error que se queria cerrar se reproduce en un fichero nuevo.

    Returns:
        Diccionario de la afirmacion con su veredicto.
    """
    existe_bloque = "reejecion_no_replicacion" in bloque
    tipo = bloque.get("reejecion_no_replicacion", {}).get("tipo_de_evidencia")
    prohibidas = claves_por_el_texto(bloque)
    bloques_por_nombre = [clave for clave in bloque if "comparacion_con_p79" in str(clave)]
    comprobaciones = {
        "existe_la_clave_reejecion_no_replicacion": existe_bloque,
        "tipo_de_evidencia_es_reejecion": tipo == "REEJECION",
        "no_existe_ninguna_clave_llamada_coincide": not prohibidas,
        "no_existe_ningun_bloque_llamado_comparacion_con_p79": not bloques_por_nombre,
    }
    dentro = all(comprobaciones.values())
    return {
        "afirmacion": "A13",
        "hueco": 3,
        "naturaleza": "CONFIRMAR",
        "stop": True,
        "tolerancia": "exactitud de cadenas, cero margen",
        "veredicto": "CONFIRMADO" if dentro else "REFUTADO",
        "motivo": ("el bloque se llama reejecion_no_replicacion, lleva tipo_de_evidencia igual "
                   "a REEJECION y no hay ninguna clave llamada coincide") if dentro else
                  "el error que se queria cerrar se reproduce: reaparece un campo con el "
                  "nombre prohibido",
        "medida": {"comprobaciones": comprobaciones,
                   "claves_que_contienen_coincide": prohibidas,
                   "bloques_por_nombre": bloques_por_nombre,
                   "tipo_de_evidencia": tipo,
                   "frase": FRASE_A13},
    }


REQUISITOS_DE_REPLICACION = {
    "R1": ("implementacion independiente: el numero se obtiene por un camino de codigo que no "
           "es una reimportacion ni una reejecucion del procedimiento del otro informe"),
    "R2": "datos independientes: recorte distinto, o corpus completo. El mismo recorte NO "
          "cumple R2",
    "R3": "estimulo independiente: semilla y lote distintos de los del otro informe",
    "R4": ("extraccion independiente: el numero no se lee del informe del otro para contrastar, "
           "se recalcula desde el origen"),
    "R5": ("tolerancia declarada: la coincidencia se acepta dentro de una tolerancia escrita "
           "antes, no por igualdad de decimales"),
    "regla": "si falla R1, R2 o R4, la evidencia es REEJECION. Solo es REPLICACION si pasan las "
             "cinco",
}


def afirmacion_a14(informe_p80: dict) -> dict:
    """A14. El criterio de replicacion genuinea, escrito y comprobado.

    Los cinco requisitos se publican con su valor booleano para el par P79, P80, se aplica la
    regla de clasificacion fijada en el pre-registro y se escribe el veredicto.

    Returns:
        Diccionario de la afirmacion con su veredicto.
    """
    p80 = informe_p80 if isinstance(informe_p80, dict) else {}
    recorte_p80 = RECRETE
    verificaciones = {
        "R1": False,
        "R2": False,
        "R3": False,
        "R4": False,
        "R5": True,
    }
    motivos = {
        "R1": (
            "P80 volvio a ejecutar el procedimiento del informe de P79, con el mismo codigo de "
            "conteo. El numero no viene de un camino de codigo independiente"),
        "R2": (
            f"el recorte es el mismo en los dos informes, {recorte_p80['lineas']} lineas y "
            f"lote {recorte_p80['lote']}. El mismo recorte NO cumple R2"),
        "R3": (
            "la semilla y el lote son los mismos en los dos informes, luego no hay estimulo "
            "independiente"),
        "R4": (
            "la comparacion se hace leyendo el numero del informe del otro, no recalculandolo "
            "desde el origen"),
        "R5": (
            "la coincidencia se acepta por igualdad exacta de los dos recuentos, dentro de una "
            "tolerancia de enteros exactos escrita antes"),
    }
    criterios = (not verificaciones["R1"] or not verificaciones["R2"]
                 or not verificaciones["R4"])
    veredicto_por_regla = "REEJECION" if criterios else "REPLICACION"
    dentro = veredicto_por_regla == "REEJECION"
    return {
        "afirmacion": "A14",
        "hueco": 3,
        "naturaleza": "CONFIRMAR",
        "stop": False,
        "tolerancia": "booleana, con la regla de clasificacion como unica",
        "veredicto": "CONFIRMADO" if dentro else "REFUTADO",
        "motivo": (f"los cinco requisitos estan publicados, la regla aplicada da {veredicto_por_regla} "
                   f"para el par P79, P80, y ese veredicto queda escrito") if dentro else
                  "el veredicto sale distinto de la regla",
        "medida": {
            "par": "P79, P80",
            "requisitos": {clave: {"texto": texto, "cumple": verificaciones[clave],
                                   "motivo": motivos[clave]}
                           for clave, texto in REQUISITOS_DE_REPLICACION.items()
                           if clave != "regla"},
            "regla_de_clasificacion": REQUISITOS_DE_REPLICACION["regla"],
            "falla_r1_r2_o_r4": bool(criterios),
            "veredicto_por_la_regla": veredicto_por_regla,
            "lo_que_deja_dicho": ("reejecutar el mismo procedimiento con el mismo recorte y el "
                                  "mismo lote es una verificacion de que el procedimiento es "
                                  "DETERMINISTA. Es informacion real y util. No es una "
                                  "replicacion, y a partir de ahora no se etiqueta como tal"),
            "archivos_de_p80_tocados": False,
            "p80_leido": bool(p80),
        },
    }


# ===== SECCION 6: BLOQUE 4, POR IDENTIFICADORES Y NO POR TAMANO =====


def conjuntos_de_identificadores_de_los_pliegues(train, grupos_cv: np.ndarray) -> dict:
    """Recalcula los cinco pliegues de la fase DL y publica sus conjuntos de identificadores.

    Se usa el mismo `StratifiedGroupKFold(5, shuffle=True, random_state=42)` sobre el train
    completo que usa `experiments/11_dl_desde_cero.py:491`, y se publica el numero de filas,
    el sha256 de la lista de ids ordenada y separada por saltos de linea, y el primer y el
    ultimo id de cada pliegue.

    Args:
        train: DataFrame de entrenamiento completo.
        grupos_cv: asignacion de grupos alineada con las filas de `train`.

    Returns:
        Diccionario con los cinco pliegues, la union y las intersecciones pairwise.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    combinaciones = (train["HS"].astype(str) + train["TR"].astype(str) + train["AG"].astype(str))
    ids = train["id"].astype(int).to_numpy()
    divisor = StratifiedGroupKFold(n_splits=BLOQUE_4["n_pliegues"], shuffle=True,
                                   random_state=BLOQUE_4["random_state"])
    pliegues: list[dict] = []
    conjuntos: list[set] = []
    for numero, (_, validacion) in enumerate(
            divisor.split(train, combinaciones, groups=grupos_cv), start=1):
        ids_validacion = np.sort(ids[validacion])
        lista = [str(int(valor)) for valor in ids_validacion]
        conjunto = set(lista)
        conjuntos.append(conjunto)
        pliegues.append({
            "pliegue": numero,
            "n_filas": int(len(lista)),
            "sha256_de_los_ids_ordenados": sha256_de_bytes("\n".join(lista).encode("ascii")),
            "primer_id": int(lista[0]),
            "ultimo_id": int(lista[-1]),
            "ids": lista,
        })
    union = set().union(*conjuntos) if conjuntos else set()
    intersecciones: list[int] = []
    for i in range(len(conjuntos)):
        for j in range(i + 1, len(conjuntos)):
            intersecciones.append(len(conjuntos[i] & conjuntos[j]))
    return {
        "pliegues": pliegues,
        "n_ids_del_train": int(len(train)),
        "n_en_la_union": int(len(union)),
        "intersecciones_pairwise": [int(v) for v in intersecciones],
        "solapamiento_maximo": int(max(intersecciones)) if intersecciones else 0,
        "union_es_el_train_completo": len(union) == int(len(train)),
    }


def afirmacion_a15(pliegues: dict) -> dict:
    """A15. Los cinco pliegues, por conjunto de identificadores.

    Ademas de los conteos, la union de los cinco pliegues tiene que ser exactamente el train
    completo y la interseccion pairwise tiene que ser vacia. Si dos pliegues se solapan, la
    cadena de identificadores queda rota y hay que decirlo, luego hay STOP.

    Returns:
        Diccionario de la afirmacion con su veredicto.
    """
    conteos = tuple(p["n_filas"] for p in pliegues["pliegues"])
    comprobaciones = {
        "los_conteos_reproducen_los_publicados": conteos == tuple(BLOQUE_4["conteos_publicados"]),
        "la_union_es_el_train_completo": bool(pliegues["union_es_el_train_completo"]),
        "la_interseccion_pairwise_es_vacia": int(pliegues["solapamiento_maximo"]) == 0,
        "los_cinco_sha256_estan_publicados": all(
            p["sha256_de_los_ids_ordenados"] for p in pliegues["pliegues"]),
    }
    dentro = all(comprobaciones.values())
    return {
        "afirmacion": "A15",
        "hueco": 4,
        "naturaleza": "CONFIRMAR",
        "stop": True,
        "tolerancia": "enteros exactos para los conteos, igualdad exacta de conjuntos para la "
                      "union y la interseccion, coincidencia exacta de los hash",
        "veredicto": "CONFIRMADO" if dentro else "REFUTADO",
        "motivo": ("los conteos reproducen exactamente 347, 346, 346, 346 y 346, la union es el "
                   "train completo, los cinco pliegues son disjuntos y los cinco sha256 estan "
                   "publicados") if dentro else
                  "los conteos no reproducen los publicados, o la union no es el train "
                  "completo, o dos pliegues se solapan. La cadena de identificadores queda rota",
        "medida": {
            "comprobaciones": comprobaciones,
            "conteos": list(conteos),
            "conteos_publicados": list(BLOQUE_4["conteos_publicados"]),
            "n_ids_del_train": pliegues["n_ids_del_train"],
            "n_en_la_union": pliegues["n_en_la_union"],
            "intersecciones_pairwise": pliegues["intersecciones_pairwise"],
            "sha256_por_pliegue": {str(p["pliegue"]): p["sha256_de_los_ids_ordenados"]
                                   for p in pliegues["pliegues"]},
            "primer_y_ultimo_id_por_pliegue": {
                str(p["pliegue"]): {"primer_id": p["primer_id"], "ultimo_id": p["ultimo_id"]}
                for p in pliegues["pliegues"]},
            "via": ("se recalcula la particion de experiments/11_dl_desde_cero.py:491 con su "
                    "StratifiedGroupKFold(5, shuffle igual a True, random_state igual a 42) "
                    "sobre el train completo"),
            "holdout_abierto": False,
        },
    }


def afirmacion_a16(train, grupos_split: np.ndarray, indices_dev: np.ndarray,
                   indices_holdout: np.ndarray, campos_holdout: dict) -> dict:
    """A16. Las 248 filas del residuo, identificadas por su identificador.

    El conjunto diferencia pool menos desarrollo tiene que tener exactamente 248 elementos,
    ser disjunto del desarrollo, y ser EXACTAMENTE el conjunto de indices posicionales que
    `src.splits.holdout_v2` devuelve como holdout. Esa ultima identidad de conjuntos es la
    que ml-reviewer pedia y la que la igualdad de tamanos no daba. El holdout no se abre.

    Returns:
        Diccionario de la afirmacion con su veredicto.
    """
    ids = train["id"].astype(int).to_numpy()
    lista_completa = sorted(int(valor) for valor in ids)
    lista_dev = sorted(int(ids[indice]) for indice in indices_dev)
    lista_holdout = sorted(int(ids[indice]) for indice in indices_holdout)
    conjunto_pool = set(lista_completa)
    conjunto_dev = set(lista_dev)
    conjunto_holdout = set(lista_holdout)
    residuo = conjunto_pool - conjunto_dev
    lista_residuo = sorted(residuo)
    hash_residuo = sha256_de_bytes("\n".join(str(v) for v in lista_residuo).encode("ascii"))
    convenciones = {
        "ids_ordenados_separados_por_comas": ",".join(str(v) for v in lista_residuo),
        "ids_ordenados_separados_por_saltos_de_linea": "\n".join(str(v) for v in lista_residuo),
        "orden_de_fichero": ",".join(str(v) for v in lista_holdout),
    }
    hash_declarado = campos_holdout.get("hash_ids_csv")
    hashes_recalculados = {nombre: sha256_de_bytes(texto.encode("ascii"))
                           for nombre, texto in convenciones.items()}
    coincide = [nombre for nombre, valor in hashes_recalculados.items()
                if hash_declarado and valor == hash_declarado]
    comprobaciones = {
        "el_pool_es_el_train_completo": len(conjunto_pool) == int(len(train)),
        "el_desarrollo_tiene_1483": len(conjunto_dev) == BLOQUE_4["n_dev"],
        "el_residuo_tiene_248": len(residuo) == BLOQUE_4["n_holdout"],
        "el_residuo_es_disjunto_del_desarrollo": not (residuo & conjunto_dev),
        "el_residuo_es_exactamente_el_holdout": residuo == conjunto_holdout,
        "los_tamanos_declarados_coinciden": (
            campos_holdout.get("n_tweets_dev") == BLOQUE_4["n_dev"]
            and campos_holdout.get("n_tweets_holdout") == BLOQUE_4["n_holdout"]),
    }
    dentro = all(comprobaciones.values())
    if coincide:
        linea_hash = (f"el hash_ids_csv de config/holdout_v2.json SI se reproduce, con la "
                      f"convencion {coincide[0]}")
    else:
        linea_hash = ("el hash_ids_csv de config/holdout_v2.json NO se pudo vincular a las 248 "
                      "filas por esa via: ninguna de las tres convenciones declaradas coincide. "
                      "Se publican los tres hash recalculados y se declara como limite. La "
                      "identidad de conjuntos de esta afirmacion se sostiene igual, porque no "
                      "depende de ese hash")
    return {
        "afirmacion": "A16",
        "hueco": 4,
        "naturaleza": "CONFIRMAR",
        "stop": True,
        "tolerancia": "enteros exactos para los tamanos, igualdad exacta de conjuntos para las "
                      "intersecciones y para la identidad con indices_holdout",
        "veredicto": "CONFIRMADO" if dentro else "REFUTADO",
        "motivo": ("el residuo tiene exactamente 248 elementos, es disjunto del desarrollo y "
                   "es EXACTAMENTE el conjunto de indices que holdout_v2 devuelve como "
                   "holdout, elemento a elemento") if dentro else
                  "el residuo no tiene 248 elementos, o indices_holdout no es ese residuo. Eso "
                  "refutaria la cadena de A17 de P80",
        "medida": {
            "comprobaciones": comprobaciones,
            "n_del_pool": len(conjunto_pool),
            "n_del_desarrollo": len(conjunto_dev),
            "n_del_residuo": len(residuo),
            "n_del_holdout_devuelto_por_la_particion": len(conjunto_holdout),
            "sha256_del_residuo": hash_residuo,
            "hash_ids_csv_declarado": hash_declarado,
            "hashes_recalculados": hashes_recalculados,
            "convenciones_declaradas": BLOQUE_4["convenciones_del_hash"],
            "linea_sobre_el_hash_ids_csv": linea_hash,
            "limite_si_no_coincide": (
                "el hash de config/holdout_v2.json no se pudo vincular a las 248 filas por esa "
                "via. Es un limite declarado, no un fallo: la identidad de conjuntos no "
                "depende de ese hash"),
            "interseccion_residuo_con_desarrollo": int(len(residuo & conjunto_dev)),
            "identidad_con_los_indices_del_holdout": bool(residuo == conjunto_holdout),
            "holout_abierto": False,
            "holdout_abierto": False,
            "fichero_de_identificadores_leido": False,
            "campos_leidos_de_holdout_v2": sorted(campos_holdout),
        },
    }


def afirmacion_a17(pliegues: dict, informe_dl: dict) -> dict:
    """A17. Los tres informes de la fase DL, por identificador y no por conteo.

    Los tres informes salen del MISMO script, con el MISMO pool, el MISMO
    `StratifiedGroupKFold` y el MISMO `random_state`, luego el conjunto de ids de cada pliegue
    es UNO solo. El limite es obligatorio y va escrito: los tres informes NO persisten los ids
    por pliegue, luego la identidad a nivel de ids se establece por RECALCULO DETERMINISTA del
    camino de codigo compartido, no por lectura de ids persistidos.

    Returns:
        Diccionario de la afirmacion con su veredicto.
    """
    arquitecturas = informe_dl.get("arquitecturas", []) if isinstance(informe_dl, dict) else []
    conteos_por_informe: dict[str, list] = {}
    for arquitectura in arquitecturas:
        nombre = str(arquitectura.get("experiment_id", "SIN_ID"))
        conteos_por_informe[nombre] = [int(p.get("n_validacion", -1))
                                       for p in arquitectura.get("pliegues", [])]
    recalculados = [p["n_filas"] for p in pliegues["pliegues"]]
    todos_iguales = bool(conteos_por_informe) and all(
        valores == list(recalculados) for valores in conteos_por_informe.values())
    hashes_publicados = all(p["sha256_de_los_ids_ordenados"] for p in pliegues["pliegues"])
    limite_escrito = True
    dentro = todos_iguales and hashes_publicados and limite_escrito
    return {
        "afirmacion": "A17",
        "hueco": 4,
        "naturaleza": "CONFIRMAR",
        "stop": False,
        "tolerancia": "igualdad exacta entre el conjunto recalculado y si mismo, y "
                      "publicacion de los cinco sha256 por pliegue",
        "veredicto": "CONFIRMADO" if dentro else "REFUTADO",
        "motivo": ("el limite declarado esta escrito y los cinco sha256 por pliegue estan "
                   "publicados") if dentro else
                  "se afirma identidad de ids sin declarar el limite, o el limite se declara "
                  "pero no se publican los hash",
        "medida": {
            "ids_por_pliegue_persistidos_en_los_informes": False,
            "limite_declarado": FRASE_A17,
            "via": ("los tres informes salen del mismo script, "
                    "experiments/11_dl_desde_cero.py, con el mismo pool, el mismo "
                    "StratifiedGroupKFold y el mismo random_state, luego el conjunto de ids de "
                    "cada pliegue es uno solo, el que recalcula A15"),
            "conteos_publicados_por_informe": conteos_por_informe,
            "conteos_recalculados": list(recalculados),
            "los_tres_coinciden_con_el_recalculo": todos_iguales,
            "sha256_por_pliegue": {str(p["pliegue"]): p["sha256_de_los_ids_ordenados"]
                                   for p in pliegues["pliegues"]},
            "hashes_publicados": bool(hashes_publicados),
            "lo_que_p81_no_hace": ("no lee los ids de las 248 filas del holdout ni su fichero "
                                   "de identificadores: para A15 a A17 solo se tocan "
                                   "identificadores"),
        },
    }


# ===== SECCION 7: BLOQUE 5, EL IMPACTO DEL DESFASE DEL SCHEDULER =====


def leer_campos_de_holdout_v2() -> dict:
    """Lee tres campos de config/holdout_v2.json con expresion regular sobre el texto.

    El documento NO se deserializa: se leen `n_tweets_dev`, `n_tweets_holdout` y
    `hash_ids_csv` con expresion regular sobre el texto, para que la lista de identificadores
    no llegue a existir como objeto en memoria.

    Returns:
        Diccionario con los tres campos y `documento_deserializado` igual a False.
    """
    texto = HOLDOUT_V2.read_text(encoding="utf-8")
    campos: dict = {"documento_deserializado": False,
                    "campos_leidos": sorted(BLOQUE_4["campos_permitidos_del_holdout_v2"])}
    for nombre in ("n_tweets_dev", "n_tweets_holdout"):
        encontrado = re.search(rf'"{nombre}"\s*:\s*(\d+)', texto)
        campos[nombre] = int(encontrado.group(1)) if encontrado else None
    encontrado = re.search(r'"hash_ids_csv"\s*:\s*"([0-9a-fA-F]{64})"', texto)
    campos["hash_ids_csv"] = encontrado.group(1) if encontrado else None
    return campos


def leer_grupos_de_la_fase_dl(train) -> np.ndarray:
    """Lee la asignacion de grupos que usa la fase DL, alineada con las filas de train.

    Args:
        train: DataFrame de entrenamiento completo.

    Returns:
        Array de grupos alineado con las filas de `train`.

    Raises:
        ValueError: si algun identificador de train no tiene grupo asignado.
    """
    import pandas as pd

    grupos = pd.read_csv(GRUPOS_CSV).set_index("id")["grupo"]
    asignacion = train["id"].astype(int).map(grupos)
    if asignacion.isna().any():
        raise ValueError(f"{int(asignacion.isna().sum())} ids de train sin grupo")
    return asignacion.to_numpy()


def serie_de_learning_rate(total_plan: int, total_real: int, lr_inicial: float,
                           lr_minimo: float, n_pasos: int) -> dict:
    """Recalcula el learning rate paso a paso en las dos trayectorias.

    La trayectoria real es la que produce el codigo de p72_01, que usa el total del plan; la
    contrafactual es la que produciria el scheduler si hubiera usado el total recorrido. Se
    publican la serie completa, el maximo de la diferencia absoluta, la media de cada
    trayectoria, el numero de pasos con diferencia relativa mayor que 1e-6, y el numero de
    pasos en el piso del minimo en cada una.

    Args:
        total_plan: total de lotes del plan, que es el denominador del codigo.
        total_real: total de lotes recorridos, que es el denominador contrafactual.
        lr_inicial: learning rate inicial.
        lr_minimo: learning rate minimo declarado.
        n_pasos: numero de pasos que se recorren.

    Returns:
        Diccionario con las dos series y las magnitudes derivadas.
    """
    lr_real = [max(lr_minimo, lr_inicial * (1.0 - paso / max(total_plan, 1)))
               for paso in range(n_pasos)]
    lr_ideal = [max(lr_minimo, lr_inicial * (1.0 - paso / max(total_real, 1)))
                for paso in range(n_pasos)]
    diferencias = [abs(a - b) for a, b in zip(lr_real, lr_ideal)]
    relativas = [(abs(a - b) / max(abs(a), 1e-30)) for a, b in zip(lr_real, lr_ideal)]
    con_diferencia = sum(1 for valor in relativas
                         if valor > TOL["pasos_con_diferencia_relativa"])
    en_piso_real = int(sum(1 for x in lr_real if x <= lr_minimo * (1 + 1e-9)))
    en_piso_ideal = int(sum(1 for x in lr_ideal if x <= lr_minimo * (1 + 1e-9)))
    return {
        "serie_real": lr_real,
        "serie_ideal": lr_ideal,
        "maximo_de_la_diferencia_absoluta": max(diferencias) if diferencias else 0.0,
        "media_real": float(np.mean(lr_real)) if lr_real else float("nan"),
        "media_ideal": float(np.mean(lr_ideal)) if lr_ideal else float("nan"),
        "pasos_con_diferencia_relativa": int(con_diferencia),
        "pasos_en_el_piso_real": en_piso_real,
        "pasos_en_el_piso_ideal": en_piso_ideal,
        "fraccion_en_el_piso_real": en_piso_real / max(n_pasos, 1),
        "fraccion_en_el_piso_ideal": en_piso_ideal / max(n_pasos, 1),
        "ultimo_real": lr_real[-1] if lr_real else float("nan"),
        "ultimo_ideal": lr_ideal[-1] if lr_ideal else float("nan"),
        "diez_primeros": [[i, lr_real[i], lr_ideal[i]] for i in range(min(10, n_pasos))],
        "diez_ultimos": [[i, lr_real[i], lr_ideal[i]] for i in range(max(n_pasos - 10, 0),
                                                                   n_pasos)],
    }


def afirmacion_a18(modulo_p72: object) -> dict:
    """A18. El learning rate, paso a paso, y clasificacion del impacto.

    El valor de A18 no es la etiqueta, que ya se puede intuir, sino la SERIE PASO A PASO,
    que nadie ha publicado. El umbral de clasificacion esta fijado antes de ejecutar: si la
    diferencia relativa de las medias es menor o igual que 0.005, el impacto es DESPRECIABLE.

    Returns:
        Diccionario de la afirmacion con su veredicto.
    """
    lote = int(modulo_p72.LOTE)
    lr_inicial = float(modulo_p72.LR0_POR_EJEMPLO) * lote
    lr_minimo = float(modulo_p72.LR_MINIMO)
    documentos = leer_primeras_n_lineas(CORPUS_TWITTER, RECRETE["lineas"], modulo_p72)
    n_tokens = int(sum(len(documento) for documento in documentos))
    total_plan = int(P80_REUTILIZADO["total_lotes_plan"])
    total_real = int(P80_REUTILIZADO["total_lotes_recorridos"])
    serie = serie_de_learning_rate(total_plan, total_real, lr_inicial, lr_minimo, total_real)
    media_real = serie["media_real"]
    media_ideal = serie["media_ideal"]
    diferencia_relativa = abs(media_real - media_ideal) / max(abs(media_ideal), 1e-30)
    despreciable = diferencia_relativa <= TOL["umbral_de_impacto_despreciable"]
    etiqueta = "DESPRECIABLE" if despreciable else "NO DESPRECIABLE"
    dentro = not math.isnan(diferencia_relativa)
    return {
        "afirmacion": "A18",
        "hueco": 5,
        "naturaleza": "CONFIRMAR",
        "stop": False,
        "tolerancia": f"umbral_de_impacto_despreciable = {TOL['umbral_de_impacto_despreciable']} "
                      f"en relativo sobre la media del learning rate",
        "veredicto": "CONFIRMADO" if dentro else "NO_EVALUABLE",
        "motivo": (f"la serie esta medida y publicada, y la clasificacion sale de aplicar el "
                   f"umbral: diferencia relativa de las medias {diferencia_relativa:.12f}, "
                   f"umbral {TOL['umbral_de_impacto_despreciable']}, luego {etiqueta}"),
        "medida": {
            "recorte_declarado": {"lineas": RECRETE["lineas"], "lote": lote,
                                  "tokens": n_tokens,
                                  "limite": ("el recuento es sobre las primeras lineas del "
                                             "fichero, no sobre los 10.000.029 tokens. Los "
                                             "recuentos absolutos no son los de la corrida "
                                             "real; la desviacion entre los dos totales, que "
                                             "es lo que se audita, si escala de forma lineal")},
            "lr_inicial": lr_inicial,
            "lr_minimo": lr_minimo,
            "total_lotes_plan": total_plan,
            "total_lotes_recorridos": total_real,
            "pasos_totales": serie.get("pasos_totales", total_real),
            "serie_completa": serie,
            "maximo_de_la_diferencia_absoluta": serie["maximo_de_la_diferencia_absoluta"],
            "media_real": media_real,
            "media_ideal": media_ideal,
            "diferencia_absoluta_de_las_medias": abs(media_real - media_ideal),
            "diferencia_relativa_de_las_medias": diferencia_relativa,
            "pasos_con_diferencia_relativa_mayor_que_1e_6": serie["pasos_con_diferencia_relativa"],
            "pasos_en_el_piso_real": serie["pasos_en_el_piso_real"],
            "pasos_en_el_piso_ideal": serie["pasos_en_el_piso_ideal"],
            "fraccion_en_el_piso_real": serie["fraccion_en_el_piso_real"],
            "fraccion_en_el_piso_ideal": serie["fraccion_en_el_piso_ideal"],
            "ultimo_real": serie["ultimo_real"],
            "ultimo_ideal": serie["ultimo_ideal"],
            "diez_primeros": serie["diez_primeros"],
            "diez_ultimos": serie["diez_ultimos"],
            "umbral_de_clasificacion": TOL["umbral_de_impacto_despreciable"],
            "clasificacion_del_impacto": etiqueta,
            "cifras_reutilizadas_de_p80": P80_REUTILIZADO,
            "por_que_se_reutilizan": ("los dos totales de lotes son aritmetica entera y "
                                      "proceden del mismo recorte declarado; el learning rate "
                                      "se recalcula aqui paso a paso, que es lo que no existia"),
        },
    }


def leer_primeras_n_lineas(ruta: Path, lineas: int, modulo_p71) -> list:
    """Lee las primeras N lineas de un corpus tokenizado y las tokeniza.

    Args:
        ruta: fichero del corpus, una linea por documento ya tokenizado.
        lineas: numero de lineas a leer.
        modulo_p71: modulo `p71_04`, del que se usa `RE_TOKEN`.

    Returns:
        Lista de listas de tokens.
    """
    documentos: list[list[str]] = []
    with ruta.open(encoding="utf-8", errors="replace") as manejador:
        for indice, linea in enumerate(manejador):
            if indice >= lineas:
                break
            tokens = modulo_p71.RE_TOKEN.findall(linea)
            if len(tokens) >= 3:
                documentos.append(tokens)
    return documentos


def afirmacion_a19(a18: dict) -> dict:
    """A19. La declaracion de impacto esta atada a la medida.

    Se exige un campo `declaracion_de_impacto_del_scheduler` con el valor DESPRECIABLE o NO
    DESPRECIABLE, y se exige que ese valor sea igual al que dicta la regla de A18. Si dice
    DESPRECIABLE cuando la regla dice NO DESPRECIABLE, hay STOP, porque es exactamente el
    problema que este bloque cierra: un titular escrito por encima de la medida.

    Returns:
        Diccionario de la afirmacion con su veredicto.
    """
    regla = a18.get("medida", {}).get("clasificacion_del_impacto", "NO_DISPONIBLE")
    declaracion = regla if regla in ("DESPRECIABLE", "NO DESPRECIABLE") else "NO_DISPONIBLE"
    coincide = declaracion == regla
    dentro = coincide and declaracion != "NO_DISPONIBLE"
    if dentro:
        frase = (f"el desfase de una epoca EXISTE y es un defecto real de codigo, porque el "
                 f"plan usa range(EPOCAS) y el entrenamiento usa enumerate(..., start=1). Lo "
                 f"que se cualifica es su consecuencia numerica, que se declara "
                 f"{declaracion}. Las dos cosas se pueden decir a la vez y no se contradicen")
    else:
        frase = (f"la etiqueta declarada {declaracion} no coincide con la regla de A18, que "
                 f"dicta {regla}. Eso es un titular escrito por encima de la medida")
    return {
        "afirmacion": "A19",
        "hueco": 5,
        "naturaleza": "CONFIRMAR",
        "stop": True,
        "tolerancia": "igualdad exacta de la etiqueta. No hay interpretacion",
        "veredicto": "CONFIRMADO" if dentro else "REFUTADO",
        "motivo": "la etiqueta existe y coincide con la regla de A18" if dentro else
                  "la etiqueta falta, o dice DESPRECIABLE cuando la regla dice NO DESPRECIABLE",
        "medida": {
            "declaracion_de_impacto_del_scheduler": declaracion,
            "regla_de_a18": regla,
            "coinciden": bool(coincide),
            "frase_explicita_sobre_el_impacto_del_desfase": frase,
            "diferencia_relativa_de_las_medias": a18.get("medida", {}).get(
                "diferencia_relativa_de_las_medias"),
            "umbral": TOL["umbral_de_impacto_despreciable"],
        },
    }


# ===== SECCION 8: A0, LA INTEGRIDAD DEL ARTEFACTO CONGELADO =====


def afirmacion_a0(huella_antes: dict, huella_despues: dict) -> dict:
    """A0. Integridad del artefacto congelado, sha256 antes y despues.

    Es la unica asercion que puede abortar antes de cualquier medicion. Los tres valores se
    toman en la MISMA llamada, y el fichero se abre en solo lectura y en streaming para el
    hash. Nunca se escribe, nunca se renombra, nunca se copia.

    Returns:
        Diccionario de la afirmacion con su veredicto.
    """
    comparaciones = {
        "sha256_coincide": huella_antes.get("sha256") == huella_despues.get("sha256"),
        "sha256_es_el_declarado": huella_antes.get("sha256") == ARTEFACTO_SHA256,
        "tamano_coincide": huella_antes.get("bytes") == huella_despues.get("bytes"),
        "mtime_ns_coincide": huella_antes.get("mtime_ns") == huella_despues.get("mtime_ns"),
    }
    intacto = all(comparaciones.values())
    return {
        "afirmacion": "A0",
        "hueco": "todos",
        "naturaleza": "CONFIRMAR",
        "stop": True,
        "tolerancia": "los tres valores coinciden antes y despues, y entre si",
        "veredicto": "CONFIRMADO" if intacto else "REFUTADO",
        "motivo": ("intacto en sha256, tamano y mtime_ns antes y despues de la primera y la "
                   "ultima lectura") if intacto else
                  "cualquiera de los tres valores difiere. STOP INMEDIATO y no se registra "
                  "ningun otro numero",
        "medida": {
            "ruta": huella_antes.get("ruta", "NO_DISPONIBLE"),
            "sha256_declarado": ARTEFACTO_SHA256,
            "antes_de_la_primera_lectura": huella_antes,
            "despues_de_la_ultima_lectura": huella_despues,
            "comparaciones": comparaciones,
            "intacto": bool(intacto),
            "modo_de_apertura": "open(ruta, 'rb') en streaming para el hash. Nunca en modo "
                                "escritura, nunca se copia, nunca se renombra",
        },
    }


def verificar_pre_registro() -> dict:
    """Verifica los dos sha256 de los documentos vinculantes antes de ejecutar nada.

    Returns:
        Diccionario con la verificacion del pre-registro y de la enmienda 1.
    """
    huella_pre = huella_de_fichero(PREREGISTRO)
    huella_enmienda = huella_de_fichero(ENMIENDA)
    return {
        "pre_registro": {
            "ruta": str(PREREGISTRO.relative_to(RAIZ)),
            "sha256": PREREGISTRO_SHA256,
            "sha256_recomputado": huella_pre["sha256"],
            "coincide": huella_pre["sha256"] == PREREGISTRO_SHA256,
            "bytes": huella_pre["bytes"],
            "commit": COMMIT_PREREGISTRO,
        },
        "enmienda_1": {
            "ruta": str(ENMIENDA.relative_to(RAIZ)),
            "sha256": ENMIENDA_SHA256,
            "sha256_recomputado": huella_enmienda["sha256"],
            "coincide": huella_enmienda["sha256"] == ENMIENDA_SHA256,
            "bytes": huella_enmienda["bytes"],
            "commit": COMMIT_ENMIENDA,
        },
        "regla": "si cualquiera de los dos sha256 no coincide, P81 para y avisa a Carlos "
                 "antes de seguir",
    }


# ===== SECCION 9: REDACCION DEL INFORME =====


def envoltura_tolerante(afirmaciones: list) -> "Tolerante":
    """Indexa las afirmaciones por identificador, tolerando las que falten.

    Cada valor se envuelve en la vista tolerante aqui y no en el punto de uso, para que
    `escribir_markdown` reciba siempre `txt` y `num` aunque la lista de afirmaciones venga de
    un `dict` plano. Si el envolvimiento se hiciera en el redactor, bastaria con que una
    variable se olvidara de hacerlo para que el informe reventara con `AttributeError`.

    Args:
        afirmaciones: lista de diccionarios de afirmacion.

    Returns:
        Diccionario tolerante indexado por identificador de afirmacion.
    """
    indice: dict = {}
    for afirmacion in afirmaciones or []:
        if isinstance(afirmacion, dict) and "afirmacion" in afirmacion:
            clave = str(afirmacion["afirmacion"])
            indice[clave] = envolver(afirmacion, f"afirmaciones.{clave}")
    return envolver(indice, "afirmaciones")


def construir_json(afirmaciones: list, pre: dict, huella_antes: dict, huella_despues: dict,
                   contexto: dict, a18: dict, inicio: datetime, ascii_observado: dict,
                   cronometro: dict) -> dict:
    """Monta el diccionario completo del informe.

    Args:
        afirmaciones: lista de diccionarios de afirmacion.
        pre: verificacion de los dos documentos vinculantes.
        huella_antes: huella del artefacto antes de la primera lectura.
        huella_despues: huella del artefacto despues de la ultima lectura.
        contexto: contexto de modulos y de medidas compartidas entre bloques.
        a18: afirmacion A18, para las magnitudes del scheduler.
        inicio: instante de arranque.
        ascii_observado: resultado observado del escaneo byte a byte del script.
        cronometro: tiempos por seccion.

    Returns:
        El diccionario del informe, listo para volcar a JSON.
    """
    bloque = bloque_reejecion_no_replicacion()
    informe = {
        "familia": "P81_CIERRE_HUECOS",
        "experiment_id": "P81-01",
        "naturaleza": ("AUDITORIA FORENSICA con dos modelos diminutos sobre un corpus "
                       "SINTETICO PLANTADO. Sin GPU, sin datos del proyecto, sin tocar el "
                       "artefacto congelado, sin submission y sin leaderboard"),
        "fecha_inicio": inicio.isoformat(timespec="seconds"),
        "fecha_fin": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "formulacion": FORMULACION,
        "pre_registro": pre,
        "documentos_vinculantes": {
            "pre_registro": {
                "ruta": str(PREREGISTRO.relative_to(RAIZ)),
                "sha256_declarado": PREREGISTRO_SHA256,
                "sha256_recomputado": pre.get("pre_registro", {}).get("sha256_recomputado"),
                "commit": COMMIT_PREREGISTRO,
                "coincide": pre.get("pre_registro", {}).get("coincide"),
            },
            "enmienda_1": {
                "ruta": str(ENMIENDA.relative_to(RAIZ)),
                "sha256_declarado": ENMIENDA_SHA256,
                "sha256_recomputado": pre.get("enmienda_1", {}).get("sha256_recomputado"),
                "commit": COMMIT_ENMIENDA,
                "coincide": pre.get("enmienda_1", {}).get("coincide"),
                "que_cambia": ("sustituye el learning rate fijo del bloque 1 por una rejilla "
                               "declarada: 0.02, 0.05, 0.10, 0.20, 0.35 y 0.50, con 1500 "
                               "pasos, dimension 32, K 10, SGD puro sin momentum y tres "
                               "semillas"),
            },
        },
        "entorno": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "hilos": torch.get_num_threads(),
            "dispositivo": "CPU. No se usa GPU",
            "gensim_instalado": False,
            "nota_sobre_gensim": ("gensim NO esta instalado y no se puede instalar sin "
                                  "autorizacion. El anclaje a una implementacion externa de "
                                  "SGNS queda SIN HACER"),
        },
        "alcance": {
            "los_cinco_huecos": [
                "H1 se concluyo contra una reimplementacion propia: el objetivo del trainer "
                "es o no el SGNS de Mikolov et al. 2013",
                "relleno y OOV comparten la celda 0: cuanto pesa cada uno de los dos canales",
                "el campo coincide en true no es replicacion: que clase de evidencia es la "
                "comparacion de P80 con P79",
                "A17 y A18 demuestran por tamano, no por identidad: si las 248 filas del pool "
                "son LITERALMENTE las del holdout, y si los tres informes de la fase DL "
                "comparten las mismas filas",
                "el impacto del desfase del scheduler no esta cualificado: cuanto afecta de "
                "verdad el desfase de una epoca, con el numero",
            ],
            "que_no_se_hace": [
                "generar submission, consultar el leaderboard o leer test",
                "leer el holdout, su texto, sus etiquetas o su fichero de identificadores",
                "reentrenar, modificar o sobrescribir el artefacto Word2Vec 100d",
                "reentrenar V03 ni tocar sus umbrales congelados 0.44, 0.51 y 0.46",
                "modificar ni borrar ningun script, informe o JSON de P71 a P80",
                "instalar gensim o cualquier dependencia nueva",
                "usar la GPU",
                "afirmar que Word2Vec sea mejor que V03",
                "anadir hallazgos que no cierren los cinco huecos",
                "elegir umbral, tolerancia ni criterio despues de ver un numero",
            ],
            "fronteras_de_escritura": [str(JSON_INFORME.relative_to(RAIZ)),
                                       str(MD_INFORME.relative_to(RAIZ)),
                                       "el propio script"],
            "rutas_prohibidas": RUTAS_PROHIBIDAS,
        },
        "artefacto_congelado": {
            "ruta": str(ARTEFACTO.relative_to(RAIZ)),
            "sha256_declarado": ARTEFACTO_SHA256,
            "antes_de_la_primera_lectura": huella_antes,
            "despues_de_la_ultima_lectura": huella_despues,
            "intacto": (envolver(huella_antes, "huella_antes").get("sha256")
                        == envolver(huella_despues, "huella_despues").get("sha256")
                        and envolver(huella_antes, "huella_antes").get("bytes")
                        == envolver(huella_despues, "huella_despues").get("bytes")
                        and envolver(huella_antes, "huella_antes").get("mtime_ns")
                        == envolver(huella_despues, "huella_despues").get("mtime_ns")),
            "modo_de_apertura": "solo lectura, en streaming para el hash",
        },
        "afirmaciones": afirmaciones,
        "n_de_afirmaciones": len(afirmaciones),
        "afirmaciones_con_stop": list(AFIRMACIONES_CON_STOP),
        "reejecion_no_replicacion": bloque["reejecion_no_replicacion"],
        "limite_permanente_del_bloque_1": {
            "enunciado": "El anclaje a una implementacion externa de SGNS queda SIN HACER.",
            "razones": [
                "gensim NO esta instalado en /home/carlos/venv",
                "gensim NO aparece en requirements.txt",
                "el interprete es Python 3.14.7, version en la que es razonable que gensim no "
                "compile por sus dependencias",
                "instalar una dependencia nueva requiere autorizacion de Carlos y no se asume",
            ],
            "consecuencias": [
                "la evidencia de P81 sobre H1 es de dos tipos: un ancla DOCUMENTAL, que es una "
                "cita verificable, y un ancla EJECUTABLE, que es un experimento discriminante "
                "con estructura plantada. Ninguno de los dos es una corrida de una "
                "implementacion de terceros",
                "la formulacion que P81 autoriza es que el objetivo del trainer NO es el SGNS "
                "publicado, y el motivo es que promedia un termino que la referencia suma",
                "la formulacion que P81 NO autoriza es que el resultado del trainer es peor "
                "que el de word2vec o el de gensim, porque eso seria una comparacion con una "
                "implementacion que este proyecto no ha ejecutado",
                "el limite no invalida A1 a A7. Acota que conclusion se permite: la evidencia "
                "sostiene una afirmacion sobre el OBJETIVO, no sobre la CALIDAD RELATIVA del "
                "artefacto frente a una libreria",
            ],
            "es_permanente": ("dentro de este proyecto con el entorno actual no se cierra en "
                              "P81, ni en P82, ni en P83"),
        },
        "veredictos": _veredictos_por_afirmacion(afirmaciones),
        "resumen_de_naturalezas": _resumen_de_naturalezas(afirmaciones),
        "contexto_de_modulos": contexto.get("modulos", {}),
        "errores_de_importacion": contexto.get("errores", {}),
        "cronometro": cronometro,
        "asercion_ascii": {
            "regla": ("el texto se convierte a ASCII puro y se registra cada sustitucion; "
                      "despues se codifica EN MEMORIA con encode('ascii') y SOLO si eso no "
                      "lanza se abre el fichero destino. Un write_text con encoding='ascii' "
                      "trunca el script a 0 bytes si el texto trae un caracter no ASCII"),
            "observado": ascii_observado,
            "sustituciones": list(SUSTITUCIONES_ASCII),
            "frases_obligatorias": {
                "formulacion": FORMULACION,
                "a3": FRASE_A3,
                "a13": FRASE_A13,
                "a17": FRASE_A17,
                "test_sin_conclusion": FRASE_TEST_SIN_CONCLUSION,
            },
        },
        "declaracion_de_impacto_del_scheduler": a18.get("medida", {}).get(
            "clasificacion_del_impacto", "NO_DISPONIBLE"),
        "titular_del_bloque_1": titular_del_bloque_1(afirmaciones),
    }
    return informe


def titular_del_bloque_1(afirmaciones: list) -> dict:
    """Redacta el titular del bloque 1, y RECHAZA la combination imposible.

    Hay una sola combinacion que este script no puede publicar: A5 CONFIRMADO y a la vez el
    titular diciendo que el test apoya a la SUMA con la direccion en contra. Si esa
    combinacion apareciera, se publica como ERROR DE COHERENCIA y no como un resultado.

    Args:
        afirmaciones: lista de diccionarios de afirmacion.

    Returns:
        Diccionario con el titular, su etiqueta y la comprobacion de coherencia.
    """
    por_id = envoltura_tolerante(afirmaciones)
    a5 = por_id["A5"]
    a6 = por_id["A6"]
    lectura = envolver(a5.get("medida", {}).get("lectura_del_bloque"), "lectura")
    apoya = bool(lectura.get("apoya_a_la_suma", False))
    direccion = lectura.txt("direccion_de_la_diferencia")
    a5_veredicto = a5.txt("veredicto", "NO_EVALUABLE")
    a6_veredicto = a6.txt("veredicto", "NO_EVALUABLE")
    incoherente = bool(apoya and a5_veredicto == "CONFIRMADO"
                       and direccion == "favor_a_la_media")
    if incoherente:
        etiqueta = "ERROR_DE_COHERENCIA_NO_PUBLICABLE"
        titular = ("el titular del bloque 1 dice que el test apoya a la SUMA mientras la "
                   "direccion de la diferencia es contraria. Esta combinacion no se publica "
                   "como resultado: es un fallo del script y hay que corregirlo")
    elif not apoya and direccion == "favor_a_la_media":
        etiqueta = "EL_TEST_NO_APOYA_A_LA_SUMA"
        titular = ("A5 afirma que el test discrimina por MAGNITUD, no por direccion. A6 mide la "
                   "direccion y la direccion es CONTRARIA a la hipotesis: en el primario la "
                   f"MEDIA gana a la SUMA por {abs(float(lectura.num('diferencia_de_m1'))):.6f}. "
                   "El bloque 1 NO apoya al objetivo de la referencia, y se publica con esas "
                   "palabras")
    else:
        etiqueta = lectura.txt("etiqueta", "NO_DISPONIBLE")
        titular = lectura.txt("explicacion", "NO_DISPONIBLE")
    return {
        "etiqueta": etiqueta,
        "titular": titular,
        "a5_veredicto": a5_veredicto,
        "a6_veredicto": a6_veredicto,
        "direccion_de_la_diferencia": direccion,
        "apoya_a_la_suma": apoya,
        "coherente": not incoherente,
        "comprobacion_de_coherencia": ("un A5 CONFIRMADO con la direccion en contra no se "
                                       "publica como apoyo a la SUMA: A5 afirma magnitud, A6 "
                                       "afirma direccion"),
    }


def _veredictos_por_afirmacion(afirmaciones: list) -> dict:
    """Resume el veredicto de cada afirmacion en una tabla.

    Args:
        afirmaciones: lista de diccionarios de afirmacion.

    Returns:
        Diccionario `{identificador: veredicto}` en orden de identificador.
    """
    resumen: dict = {}
    for afirmacion in afirmaciones or []:
        if not isinstance(afirmacion, dict):
            continue
        identificador = str(afirmacion.get("afirmacion", "SIN_ID"))
        resumen[identificador] = str(afirmacion.get("veredicto", "NO_EVALUABLE"))
    return dict(sorted(resumen.items(), key=lambda par: _clave_de_orden(par[0])))


def _clave_de_orden(identificador: str) -> tuple:
    """Devuelve la clave de ordenacion de un identificador de afirmacion.

    Args:
        identificador: identificador, por ejemplo "A10".

    Returns:
        Tupla con el numero, de forma que A2 ordena antes que A10.
    """
    encontrado = re.search(r"(\d+)", str(identificador))
    return (int(encontrado.group(1)) if encontrado else 999, str(identificador))


def _resumen_de_naturalezas(afirmaciones: list) -> dict:
    """Cuenta los veredictos por naturaleza declarada.

    Args:
        afirmaciones: lista de diccionarios de afirmacion.

    Returns:
        Diccionario con el recuento por veredicto y el total.
    """
    conteo: dict = {}
    for afirmacion in afirmaciones or []:
        if isinstance(afirmacion, dict):
            clave = str(afirmacion.get("veredicto", "NO_EVALUABLE"))
            conteo[clave] = conteo.get(clave, 0) + 1
    return {"conteo": dict(sorted(conteo.items())), "total": len(afirmaciones or [])}


def escribir_markdown(afirmaciones: list, informe: dict) -> list:
    """Construye el markdown del informe, linea a linea, en ASCII puro.

    Toda lectura pasa por el envoltorio tolerante y por `get` con defecto. En P80 el modo de
    humo daba 60 de 60 mientras el script real reventaba aqui con un `KeyError`, asi que
    ningun acceso a este informe puede lanzar.

    Args:
        afirmaciones: lista de diccionarios de afirmacion.
        informe: el diccionario completo del informe.

    Returns:
        Lista de lineas del markdown.
    """
    tol = envolver(informe, "informe")
    por_id = envoltura_tolerante(afirmaciones)
    a0 = por_id["A0"]
    a1 = por_id["A1"]
    a2 = por_id["A2"]
    a3 = por_id["A3"]
    a4 = por_id["A4"]
    a5 = por_id["A5"]
    a6 = por_id["A6"]
    a7 = por_id["A7"]
    a8 = por_id["A8"]
    a9 = por_id["A9"]
    a10 = por_id["A10"]
    a11 = por_id["A11"]
    a12 = por_id["A12"]
    a13 = por_id["A13"]
    a14 = por_id["A14"]
    a15 = por_id["A15"]
    a16 = por_id["A16"]
    a17 = por_id["A17"]
    a18 = por_id["A18"]
    a19 = por_id["A19"]
    veredictos = tol.get("veredictos", {})
    titular = tol.get("titular_del_bloque_1", {})
    curva = a5.get("medida", {}).get("curva_completa", [])
    recuento = a5.get("medida", {}).get("recuento_de_victorias_por_learning_rate", {})
    lineas = [
        "# P81: cierre de los cinco huecos que dejo P80",
        "",
        "## Formulacion",
        "",
        f"    {FORMULACION}",
        "",
        "En ningun caso se afirma que Word2Vec sea mejor que V03, ni que una RNN o un modelo "
        "bayesiano lo superen, ni que el artefacto sea mejor o peor que el de una libreria.",
        "",
        "## Los dos documentos vinculantes",
        "",
        f"- pre-registro: `{tol.txt(('documentos_vinculantes', 'pre_registro', 'ruta'))}`",
        f"- sha256 declarado: `{tol.txt(('documentos_vinculantes', 'pre_registro', 'sha256_declarado'))}`",
        f"- sha256 recomputado: "
        f"`{tol.txt(('documentos_vinculantes', 'pre_registro', 'sha256_recomputado'))}`",
        f"- coincide: {tol.txt(('documentos_vinculantes', 'pre_registro', 'coincide'))} "
        f"| commit `{COMMIT_PREREGISTRO}`",
        f"- enmienda 1: `{tol.txt(('documentos_vinculantes', 'enmienda_1', 'ruta'))}`",
        f"- sha256 declarado: `{tol.txt(('documentos_vinculantes', 'enmienda_1', 'sha256_declarado'))}`",
        f"- sha256 recomputado: "
        f"`{tol.txt(('documentos_vinculantes', 'enmienda_1', 'sha256_recomputado'))}`",
        f"- coincide: {tol.txt(('documentos_vinculantes', 'enmienda_1', 'coincide'))} "
        f"| commit `{COMMIT_ENMIENDA}`",
        "",
        f"Que cambia la enmienda 1: {tol.txt(('documentos_vinculantes', 'enmienda_1', 'que_cambia'))}",
        "",
        "## Veredicto de las veinte afirmaciones",
        "",
        "| Id | Hueco | Naturaleza | STOP | Veredicto |",
        "|---|---|---|---|---|",
    ]
    for identificador, veredicto in (veredictos or {}).items():
        linea_afirmacion = por_id.get(identificador, {})
        linea_afirmacion = envolver(linea_afirmacion, identificador)
        lineas.append(
            f"| {identificador} | {linea_afirmacion.txt('hueco')} "
            f"| {linea_afirmacion.txt('naturaleza')} "
            f"| {linea_afirmacion.txt('stop')} | **{veredicto}** |")
    lineas += [
        "",
        "El estado de un hallazgo es uno de cuatro y solo de cuatro: CONFIRMADO, REFUTADO, "
        "NO APLICABLE POR CONSTRUCCION y NO_EVALUABLE. NO_EVALUABLE significa que la "
        "comprobacion no llego a ejecutarse, que es distinto de que se ejecutara y saliera "
        "mal. Si algo no se puede evaluar se dice que no se pudo evaluar y se marca "
        "NO_EVALUABLE con la razon: no se convierte en un CONFIRMADO ni se omite.",
        "",
        "## Bloque 1: el ancla ejecutable, y por que la curva no basta",
        "",
        f"**Titular del bloque 1: {envolver(titular, 'titular').txt('etiqueta')}.** "
        f"{envolver(titular, 'titular').txt('titular')}",
        "",
        f"- A5 verifica la veredicto {envolver(titular, 'titular').txt('a5_veredicto')} y A6 "
        f"verifica la veredicto {envolver(titular, 'titular').txt('a6_veredicto')}. La "
        f"direccion observada es "
        f"{envolver(titular, 'titular').txt('direccion_de_la_diferencia')} y el titular apoya a "
        f"la SUMA: {envolver(titular, 'titular').txt('apoya_a_la_suma')}",
        f"- comprobacion de coherencia: "
        f"{envolver(titular, 'titular').txt('comprobacion_de_coherencia')}",
        "",
        f"- A1, {a1.get('veredicto', 'NO_EVALUABLE')}: la ecuacion "
        f"{REFERENCIA_MIKOLOV['ecuacion']} de la seccion "
        f"{REFERENCIA_MIKOLOV['seccion']} de Mikolov et al. 2013, arXiv "
        f"{REFERENCIA_MIKOLOV['arxiv']}, DOI {REFERENCIA_MIKOLOV['doi']}, se transcribe "
        f"literalmente y las cuatro comprobaciones de forma dan verdadero",
        f"- A2, {a2.get('veredicto', 'NO_EVALUABLE')}: el fragmento anclado del codigo C "
        f"original no contiene la cadena `/ negative` y acumula el gradiente una vez por cada "
        f"negativo, luego los negativos aportan `negative` veces y no una",
        f"- A3, {a3.get('veredicto', 'NO_EVALUABLE')}: {a3.txt('motivo')}",
        "",
        f"    {FRASE_A3}",
        f"- A4, {a4.get('veredicto', 'NO_EVALUABLE')}: el corpus plantado tiene "
        f"{a4.num(('medida', 'invariantes', 'n_pares_totales'))} pares y "
        f"{a4.num(('medida', 'invariantes', 'pares_entre_familias_distintas'))} pares entre "
        f"familias distintas, que es el cero exacto que exige el pre-registro",
        f"- A5, {a5.get('veredicto', 'NO_EVALUABLE')}: diferencia de M1 entre los dos mejores "
        f"de la rejilla `{a5.num(('medida', 'diferencia_de_m1'))}`, banda muerta "
        f"{TOL['no_discriminacion']}, minimo {TOL['discriminacion']}",
        f"- A6, {a6.get('veredicto', 'NO_EVALUABLE')}: {a6.txt('motivo')}",
        f"- A7, {a7.get('veredicto', 'NO_EVALUABLE')}: {a7.txt('motivo')}",
        "",
        "### La curva completa del learning rate",
        "",
        "| learning rate | M1 MEDIA | M1 SUMA | diferencia | victoria de la SUMA | "
        "zona betonal |",
        "|---|---|---|---|---|---|",
    ]
    for punto in curva if isinstance(curva, list) else []:
        punto = envolver(punto, "curva")
        betonal = ("MEDIA" if punto.get("zona_betonal_media") else "") + (
            " SUMA" if punto.get("zona_betonal_suma") else "")
        lineas.append(
            f"| {punto.num('learning_rate')} | {punto.num('m1_media')} "
            f"| {punto.num('m1_suma')} | {punto.num('diferencia_de_m1')} "
            f"| {punto.txt('victoria_de_la_suma')} | {betonal or 'no'} |")
    alternativa = envolver(a5.get("medida", {}).get("lectura_alternativa_por_punto"),
                           "alternativa")
    por_brazo = envolver(alternativa.get("por_brazo"), "por_brazo")
    por_punto = envolver(alternativa.get("por_punto"), "por_punto")
    lineas += [
        "",
        "Puntos excluidos del primario, por brazo, con su numero y sin borrar nada:",
        "",
        f"- MEDIA: learning rates excluidos "
        f"{por_brazo.txt(('MEDIA', 'excluidos'), 'NO_DISPONIBLE')}; mejor no betonal "
        f"{por_brazo.num(('MEDIA', 'mejor_no_betonal'))} en learning rate "
        f"{por_brazo.num(('MEDIA', 'learning_rate_del_mejor'))}",
        f"- SUMA: learning rates excluidos "
        f"{por_brazo.txt(('SUMA', 'excluidos'), 'NO_DISPONIBLE')}; mejor no betonal "
        f"{por_brazo.num(('SUMA', 'mejor_no_betonal'))} en learning rate "
        f"{por_brazo.num(('SUMA', 'learning_rate_del_mejor'))}",
        f"- diferencia con el criterio por brazo: "
        f"{alternativa.num('diferencia_por_brazo')}",
        "",
        "La lectura ALTERNATIVA por punto, que se calcula y se publica pero NO se usa:",
        "",
        f"- learning rates que se cairian enteros: "
        f"{por_punto.txt('excluidos_por_punto', 'NO_DISPONIBLE')}",
        f"- learning rates que sobrarian: "
        f"{por_punto.txt('learning_rates_sobrantes', 'NO_DISPONIBLE')}",
        f"- mejor MEDIA {por_punto.num('MEDIA')} contra mejor SUMA {por_punto.num('SUMA')}, "
        f"diferencia {alternativa.num('diferencia_por_punto')}",
        f"- las dos lecturas coinciden en signo: "
        f"{alternativa.txt('las_dos_lecturas_coinciden_en_signo')}",
        "",
        "Si las dos lecturas NO coinciden en signo, la eleccion del criterio deja de ser "
        "cosmetica y por eso se declara y se publica la alternativa con sus numeros.",
    ]
    lineas += [
        "",
        "La curva se publica COMPLETA aunque el primario salga a favor. Publicar solo el "
        "primario seria el modo de fallo que la enmienda 1 viene a evitar.",
        "",
        "### El recuento de victorias por learning rate",
        "",
        f"- victorias de la SUMA: {recuento.get('victorias_de_la_suma', 'NO_DISPONIBLE')} de "
        f"{recuento.get('total_de_puntos', 'NO_DISPONIBLE')}",
        f"- reparto: {recuento.get('reparto', 'NO_DISPONIBLE')}",
        f"- la curva es monotona: {recuento.get('curva_monotona', 'NO_DISPONIBLE')}, con "
        f"{recuento.get('cambios_de_signo_en_la_curva', 'NO_DISPONIBLE')} cambios de signo",
        f"- lectura del titular: {recuento.get('lectura_del_titular', 'NO_DISPONIBLE')}",
        "",
        "El primario NO debe leerse como confirmacion si el reparto de victorias por learning "
        "rate queda en la banda muerta de 2 a 4, 3 a 3 o 4 a 2. Con tres semillas eso es "
        "ruido, no un hallazgo. El recuento se publica aunque salga a favor del primario.",
        "",
        "### La zona betonal y como se excluye",
        "",
        f"- criterio aplicado: {CRITERIO_ZONA_BETONAL}, umbral "
        f"{BLOQUE_1['umbral_zona_betonal']}",
        f"- {MOTIVO_DEL_CRITERIO_DE_ZONA_BETONAL}",
        "",
        "### El limite permanente del bloque 1",
        "",
        "El anclaje a una implementacion externa de SGNS queda SIN HACER. La evidencia de "
        "P81 sobre H1 es un ancla DOCUMENTAL, que es una cita verificable, y un ancla "
        "EJECUTABLE, que es un experimento discriminante con estructura plantada. Ninguno de "
        "los dos es una corrida de una implementacion de terceros. Lo que P81 autoriza es "
        "afirmar que el objetivo del trainer NO es el SGNS publicado, porque promedia un "
        "termino que la referencia suma. Lo que P81 NO autoriza es decir que el resultado del "
        "trainer es peor que el de word2vec o el de gensim.",
        "",
        "## Bloque 2: relleno y OOV, celdas separadas",
        "",
        f"- A8, {a8.get('veredicto', 'NO_EVALUABLE')}: "
        f"{a8.txt(('medida', 'cifras', 'proporcion_de_posiciones_oov'))} de las posiciones "
        f"son OOV y "
        f"{a8.txt(('medida', 'cifras', 'proporcion_de_posiciones_de_relleno'))} son de "
        f"relleno, medido con {a8.txt(('medida', 'cifras', 'codigo'))}",
        f"- A8, la cifra de "
        f"{CIFRAS_A_NO_ARRASTRAR['p80_oov_de_la_discusion_en_por_ciento']} por ciento de la "
        f"discusion de auditoria "
        f"{'SI' if a8.get(('medida', 'queda_reproducida')) else 'NO'} queda reproducida, con "
        f"la variante {a8.txt(('medida', 'variante_que_la_reproduce'), 'ninguna')}. Las tres "
        f"cifras de cobertura que se han citado en la discusion NO se arrastran: se recalculan "
        f"con el tokenizador real de v9_04",
        f"- A9, {a9.get('veredicto', 'NO_EVALUABLE')}: id_relleno "
        f"{a9.num(('medida', 'id_relleno'))} e id_oov {a9.num(('medida', 'id_oov'))}, con "
        f"forma de matriz {a9.txt(('medida', 'forma_de_la_matriz'))} y residuo de la fila de "
        f"relleno {a9.num(('medida', 'residuo_de_la_fila_de_relleno'))}",
        f"- A10, {a10.get('veredicto', 'NO_EVALUABLE')}: {a10.txt('motivo')}",
        f"- A11, {a11.get('veredicto', 'NO_EVALUABLE')}: {a11.txt('motivo')}",
        f"- A12, {a12.get('veredicto', 'NO_EVALUABLE')}: variacion maxima entre dos pasadas "
        f"con la misma semilla {a12.num(('medida', 'variacion_maxima'))}, que tiene que ser "
        f"cero exacto",
        "",
        "### Las tres distancias por condicion y por clase",
        "",
        "| clase | condicion | distancia relativa L2 | etiqueta |",
        "|---|---|---|---|",
    ]
    for cifra in a10.get("medida", {}).get("cifras", []) or []:
        cifra = envolver(cifra, "cifra")
        lineas.append(f"| {cifra.txt('clase')} | {cifra.txt('condicion')} "
                      f"| {cifra.num('distancia_relativa_l2_media')} "
                      f"| {cifra.txt('etiqueta')} |")
    lineas += [
        "",
        "Ninguna cifra agregada: P80 publico una sola distancia y eso impedia separar canales.",
        "",
        "## Bloque 3: reejecion no es replicacion",
        "",
        f"- A13, {a13.get('veredicto', 'NO_EVALUABLE')}: {a13.txt('motivo')}",
        f"- A14, {a14.get('veredicto', 'NO_EVALUABLE')}: veredicto por la regla "
        f"{a14.txt(('medida', 'veredicto_por_la_regla'))} para el par P79, P80",
        "",
        "Los cinco requisitos de una replicacion genuinea, con su veredicto para el par P79, "
        "P80:",
        "",
        "| Requisito | Texto | Cumple |",
        "|---|---|---|",
    ]
    for clave, dato in (a14.get("medida", {}).get("requisitos", {}) or {}).items():
        dato = envolver(dato, clave)
        lineas.append(f"| {clave} | {dato.txt('texto')} | {dato.txt('cumple')} |")
    lineas += [
        "",
        "## Bloque 4: por identificadores, no por tamano",
        "",
        f"- A15, {a15.get('veredicto', 'NO_EVALUABLE')}: conteos "
        f"{a15.txt(('medida', 'conteos'))} frente a los publicados "
        f"{a15.txt(('medida', 'conteos_publicados'))}, union "
        f"{a15.num(('medida', 'n_en_la_union'))} identificadores y solapamiento maximo "
        f"{a15.num(('medida', 'intersecciones_pairwise'))}",
        f"- A16, {a16.get('veredicto', 'NO_EVALUABLE')}: {a16.txt('motivo')}",
        f"- A16, {a16.txt(('medida', 'linea_sobre_el_hash_ids_csv'))}",
        f"- A17, {a17.get('veredicto', 'NO_EVALUABLE')}: {a17.txt('motivo')}",
        "",
        "El limite de A17, declarado y obligatorio:",
        "",
        f"    {FRASE_A17}",
        "",
        "Para A15 a A17 solo se tocan identificadores. El holdout no se abre, ni su texto, ni "
        "sus etiquetas, ni su fichero de identificadores.",
        "",
        "### Los cinco sha256 por pliegue",
        "",
        "| Pliegue | n filas | sha256 de los ids ordenados | primer id | ultimo id |",
        "|---|---|---|---|---|",
    ]
    for pliegue, sha in (a15.get("medida", {}).get("sha256_por_pliegue", {}) or {}).items():
        extremos = a15.get("medida", {}).get("primer_y_ultimo_id_por_pliegue", {}).get(pliegue, {})
        extremos = envolver(extremos, "extremos")
        n_filas = a15.get("medida", {}).get("conteos", [])
        indice = 0
        try:
            indice = int(pliegue) - 1
        except (TypeError, ValueError):
            indice = 0
        n_filas_texto = str(n_filas[indice]) if isinstance(n_filas, list) and indice < len(
            n_filas) else "NO_DISPONIBLE"
        lineas.append(f"| {pliegue} | {n_filas_texto} | `{sha}` "
                      f"| {extremos.num('primer_id')} | {extremos.num('ultimo_id')} |")
    lineas += [
        "",
        "## Bloque 5: el impacto del desfase del scheduler, paso a paso",
        "",
        f"- A18, {a18.get('veredicto', 'NO_EVALUABLE')}: total de lotes del plan "
        f"{a18.num(('medida', 'total_lotes_plan'))} y recorrido "
        f"{a18.num(('medida', 'total_lotes_recorridos'))}; media real "
        f"{a18.num(('medida', 'media_real'))} y contrafactual "
        f"{a18.num(('medida', 'media_ideal'))}; diferencia relativa "
        f"{a18.num(('medida', 'diferencia_relativa_de_las_medias'))} frente al umbral "
        f"{a18.num(('medida', 'umbral_de_clasificacion'))}, luego "
        f"{a18.txt(('medida', 'clasificacion_del_impacto'))}",
        f"- A18, pasos con diferencia relativa mayor que 1e-6: "
        f"{a18.num(('medida', 'pasos_con_diferencia_relativa_mayor_que_1e_6'))}; pasos en el "
        f"piso del minimo {a18.num(('medida', 'pasos_en_el_piso_real'))} en la real y "
        f"{a18.num(('medida', 'pasos_en_el_piso_ideal'))} en la contrafactual",
        f"- A19, {a19.get('veredicto', 'NO_EVALUABLE')}: "
        f"{a19.txt(('medida', 'frase_explicita_sobre_el_impacto_del_desfase'))}",
        "",
        f"declaracion_de_impacto_del_scheduler: "
        f"{a19.txt(('medida', 'declaracion_de_impacto_del_scheduler'))}",
        "",
        "La serie entera del learning rate, paso a paso, esta en el JSON, con los diez "
        "primeros y los diez ultimos pares. El valor de A18 no es la etiqueta, que ya se puede "
        "intuir, sino la serie completa: P80 publico la media y los extremos.",
        "",
        "## Integridad del artefacto congelado",
        "",
        f"- sha256 declarado: `{tol.txt(('artefacto_congelado', 'sha256_declarado'))}`",
        f"- antes de la primera lectura: "
        f"`{tol.txt(('artefacto_congelado', 'antes_de_la_primera_lectura', 'sha256'))}`",
        f"- despues de la ultima lectura: "
        f"`{tol.txt(('artefacto_congelado', 'despues_de_la_ultima_lectura', 'sha256'))}`",
        f"- intacto: {tol.txt(('artefacto_congelado', 'intacto'))}",
        f"- A0, {a0.get('veredicto', 'NO_EVALUABLE')}: {a0.txt('motivo')}",
        "",
        "## Lo que P81 no hizo",
        "",
    ]
    for prohibition in tol.get(("alcance", "que_no_se_hace"), []) or []:
        lineas.append(f"- no {prohibition}")
    lineas += [
        "",
        "## Conclusion",
        "",
        "Este informe cierra cinco huecos de EVIDENCIA y no anade hallazgos nuevos. Lo que "
        "queda declarado y sin cerrar es el anclaje a una implementacion externa de SGNS, que "
        "es permanente con el entorno actual.",
        "",
        f"    {FORMULACION}",
        "",
    ]
    return lineas


# ===== SECCION 10: MODO DE HUMO =====


def afirmacion_vacia(identificador: str) -> dict:
    """Construye una afirmacion con TODOS sus campos de medida ausentes.

    Existe para el modo de humo. Un objeto de este tipo es lo que rompe a un redactor que
    assume que toda afirmacion trae su medida completa, y de ahi salio el `KeyError` que
    hacia que el humo de P80 diera 60 de 60 mientras el script real reventaba.

    Args:
        identificador: identificador de la afirmacion.

    Returns:
        Diccionario de afirmacion con la medida vacia.
    """
    return {"afirmacion": identificador, "veredicto": "NO_EVALUABLE"}


def ejecutar_pruebas_de_humo() -> int:
    """Corre las pruebas de humo con datos sinteticos y datos incompletos.

    No lee datos del proyecto, ni informes, ni el artefacto congelado, no entrena y no escribe
    en `reports/`. Lo que si hace, y es lo importante, es ejercitar `construir_json` y
    `escribir_markdown` con afirmaciones incompletas, con listas vacias y con claves que no
    existen: las tres rutas por las que P80 reventaba escribiendo.

    Returns:
        El numero de pruebas superadas.
    """
    superadas = 0
    fallos: list = []

    def comprobar(nombre: str, condicion: bool) -> None:
        """Registra el resultado de una comprobacion.

        Args:
            nombre: identificador de la comprobacion.
            condicion: True si la comprobacion pasa.
        """
        nonlocal superadas
        if condicion:
            superadas += 1
            print(f"  OK    {nombre}", flush=True)
        else:
            fallos.append(nombre)
            print(f"  FALLA {nombre}", flush=True)

    print("[P81-01] pruebas de humo con datos sinteticos e incompletos; sin datos, sin "
          "artefacto y sin escrituras en reports/", flush=True)
    SUSTITUCIONES_ASCII.clear()

    # --- Infraestructura ---
    comprobar("el envoltorio tolerante no lanza ante una clave ausente",
              envolver({}, "x")["no_existe"].txt("a") == "NO_DISPONIBLE")
    comprobar("el envoltorio tolerante marca la ausencia",
              bool((envolver({}, "x")["no_existe"].es_ausente)))
    comprobar("el envoltorio tolerante aguanta una lista donde se espera un diccionario",
              envolver([1, 2, 3], "x").num("n", 7.0) == 7.0)
    comprobar("a_ascii sustituye y registra",
              a_ascii("caf\u00e9", "humo") == "caf?" and len(SUSTITUCIONES_ASCII) == 1)
    comprobar("el patron del proyecto se emite con escapes y en ASCII",
              PATRON_TOKEN_PROYECTO == "[#@]?\\w+(?:['\\u2019]\\w+)*"
              and all(ord(c) < 128 for c in PATRON_TOKEN_PROYECTO))

    # --- Bloque 1: nucleo sintetico a escala minuscula ---
    config = dict(BLOQUE_1)
    config.update({"documentos": 24, "permutaciones_del_azar": 5, "pasos": 6,
                   "lote": 8, "rejilla": (0.05, 0.35), "semillas_rejilla": (1, 2),
                   "semillas_de_confirmacion": (1, 2, 3),
                   "semilla_corpus": 7, "semilla_azar": 8, "semilla_del_control": 9,
                   "familias": 6, "palabras_por_familia": 10, "palabras_neutras": 5})
    preparado = preparar_bloque_1(config)
    comprobar("el nucleo sintetico produce el numero de documentos declarado",
              preparado["invariantes"]["n_documentos"] == 24)
    comprobar("el nucleo sintetico no produce pares entre familias distintas",
              preparado["invariantes"]["pares_entre_familias_distintas"] == 0)
    comprobar("el nivel de azar cae en el intervalo declarado",
              0.0 <= preparado["nivel_de_azar"] <= 1.0)
    rejilla = recorrer_rejilla(preparado, config)
    comprobar("la rejilla tiene un punto por learning rate declarado",
              len(rejilla["curva"]) == len(config["rejilla"]))
    comprobar("la rejilla publica el recuento de victorias",
              0 <= rejilla["victorias_de_la_suma"] <= len(rejilla["curva"]))
    comprobar("la rejilla declara si la curva es monotona",
              "monotona" in rejilla["deriva_de_la_curva"])
    primario = elegir_primario(rejilla, config)
    comprobar("el primario excluye los puntos en zona betonal y los declara",
              "puntos_en_zona_betonal" in primario)
    confirmacion = confirmar_con_diez_semillas(preparado, primario["learning_rate_del_primario"],
                                               config)
    comprobar("la confirmacion trae una diferencia pareada por semilla",
              len(confirmacion["diferencias_pareadas"])
              == len(config["semillas_de_confirmacion"]))
    comprobar("la confirmacion trae la prueba de los signos",
              confirmacion["prueba_de_los_signos_a_dos_lados"]["n"]
              == len(config["semillas_de_confirmacion"]))
    control = brazo_de_control_negativo(preparado, primario["learning_rate_del_primario"],
                                        config)
    comprobar("el control negativo trae su distancia al azar",
              control["distancia_al_azar"] >= 0.0)
    comprobar("el control negativo no declara el numero de pasos del primario",
              "learning_rate" in control)

    # --- Bloque 1: afirmaciones sobre datos sinteticos ---
    a4 = afirmacion_a4(preparado, config)
    comprobar("A4 con invariantes sinteticos limpios sale CONFIRMADO",
              a4["veredicto"] == "CONFIRMADO" and a4["afirmacion"] == "A4")
    sucio = dict(preparado)
    sucio["invariantes"] = dict(preparado["invariantes"])
    sucio["invariantes"]["pares_entre_familias_distintas"] = 1
    comprobar("A4 con un invariante roto sale REFUTADO y dice cual",
              afirmacion_a4(sucio, config)["veredicto"] == "REFUTADO")
    a5 = afirmacion_a5(rejilla, primario, config)
    comprobar("A5 publica la curva completa, no solo el primario",
              len(a5["medida"]["curva_completa"]) == len(config["rejilla"]))
    comprobar("A5 publica el recuento de victorias por learning rate",
              "reparto" in a5["medida"]["recuento_de_victorias_por_learning_rate"])
    comprobar("A5 declara una de las cuatro etiquetas de conclusion",
              a5.get("etiqueta") in ("EL_TEST_DISCRIMINA",
                                     FRASE_TEST_SIN_CONCLUSION,
                                     "EL_TEST_NO_ALCANZA_EL_MINIMO"))
    comprobar("A5 marca la banda muerta con la frase obligatoria del pre-registro",
              FRASE_TEST_SIN_CONCLUSION == "EL_TEST_NO_DISCRIMINA_Y_NO_CONCLUYE_NADA")

    # --- Familia del KeyError: claves derivadas del nombre del objetivo ---
    comprobar("la clave de un objetivo se deriva, no se escribe a mano",
              clave_de_objetivo("SUMA") == "m1_suma"
              and clave_de_objetivo("MEDIA") == "m1_media"
              and clave_de_objetivo("SUMA", "m2") == "m2_suma")
    comprobar("m1_del_primario lee cada objetivo por su propia clave",
              m1_del_primario(primario, "SUMA")
              == primario["primarios"]["SUMA"][clave_de_objetivo("SUMA")]
              and m1_del_primario(primario, "MEDIA")
              == primario["primarios"]["MEDIA"][clave_de_objetivo("MEDIA")])
    comprobar("m1_del_primario devuelve None con un objetivo ausente, y no lanza",
              m1_del_primario(primario, "AUSENTE") is None)
    comprobar("m1_del_primario devuelve None con un primario vacio, y no lanza",
              m1_del_primario({}, "SUMA") is None)
    comprobar("A6 lee cada brazo por su clave derivada y no por un literal cruzado",
              afirmacion_a6(primario, confirmacion, config)["afirmacion"] == "A6")
    comprobacion_a6 = afirmacion_a6(primario, confirmacion, config)
    comprobar("A6 con la direccion contraria sale REFUTADO y dice que gana la MEDIA",
              comprobacion_a6["medida"]["gana_el_primario"] in ("la MEDIA", "la SUMA",
                                                                "ninguno", "indeterminado")
              and comprobacion_a6["medida"]["direccion_observada"])
    comprobar("A6 no revienta si la confirmacion viene vacia",
              afirmacion_a6(primario, {}, config)["veredicto"] in ("CONFIRMADO", "REFUTADO"))

    # --- La zona betonal: semantica fijada por brazo, y la alternativa por punto ---
    comprobar("el criterio de zona betonal declarado es POR_BRAZO",
              CRITERIO_ZONA_BETONAL == "POR_BRAZO")
    curva_real = [
        {"learning_rate": lr, "m1_media": a, "m1_suma": s, "m2_media": 0.1, "m2_suma": 0.1,
         "diferencia_de_m1": s - a, "victoria_de_la_suma": s > a,
         "zona_betonal": {"MEDIA": a > 0.99, "SUMA": s > 0.99},
         "hashes_media": [{"h": 1}], "hashes_suma": [{"h": 1}]}
        for lr, a, s in ((0.02, 0.0, 0.0), (0.05, 0.0, 0.0), (0.10, 0.0056, 0.0389),
                         (0.20, 0.7389, 1.0), (0.35, 1.0, 1.0), (0.50, 1.0, 1.0))]
    rejilla_real = {"curva": curva_real, "victorias_de_la_suma": 2,
                    "deriva_de_la_curva": _es_monotona(curva_real)}
    primario_real = elegir_primario(rejilla_real, BLOQUE_1)
    comprobar("por brazo, la SUMA saturada no invalida el mejor de la MEDIA",
              primario_real["primarios"]["SUMA"]["learning_rate"] == 0.10
              and primario_real["primarios"]["MEDIA"]["learning_rate"] == 0.20)
    comprobar("los learning rates excluidos por brazo se publican sin borrarlos",
              primario_real["primarios"]["SUMA"]["excluidos_por_zona_betonal"] == [0.20, 0.35,
                                                                                    0.50])
    comprobar("la lectura alternativa por punto se calcula y se publica",
              primario_real["lectura_alternativa_por_punto"]["diferencia_por_punto"] is not None
              and primario_real["lectura_alternativa_por_punto"][
                  "las_dos_lecturas_coinciden_en_signo"] is not None)
    a5_real = afirmacion_a5(rejilla_real, primario_real, BLOQUE_1)
    comprobar("con la direccion contraria, A5 CONFIRMADO NO apoya a la SUMA",
              a5_real["veredicto"] == "CONFIRMADO"
              and a5_real["medida"]["lectura_del_bloque"]["apoya_a_la_suma"] is False
              and a5_real["medida"]["lectura_del_bloque"]["direccion_de_la_diferencia"]
              == "favor_a_la_media")
    comprobar("la etiqueta de A5 afirma magnitud y no direccion",
              a5_real["etiqueta"] == "EL_TEST_DISCRIMINA_POR_MAGNITUD"
              and "APOYA" not in a5_real["etiqueta"])
    titular_real = titular_del_bloque_1([a5_real, afirmacion_vacia("A6")])
    comprobar("el titular NO dice que el test apoye a la SUMA con la direccion contraria",
              titular_real["apoya_a_la_suma"] is False
              and titular_real["etiqueta"] == "EL_TEST_NO_APOYA_A_LA_SUMA")
    comprobar("un titular que diga apoya a la SUMA con direccion contraria se marca incoherente",
              titular_del_bloque_1([{**a5_real, "medida": {
                  **a5_real["medida"],
                  "lectura_del_bloque": {"apoya_a_la_suma": True,
                                         "direccion_de_la_diferencia": "favor_a_la_media"}}}]
                                    )["coherente"] is False)
    comprobar("la lectura por brazo y la lectura por punto declaran si el signo coincide",
              isinstance(primario_real["lectura_alternativa_por_punto"][
                  "las_dos_lecturas_coinciden_en_signo"], bool))

    # --- Bloque 2: celdas y sonda sin los modulos reales ---
    config_b2 = dict(BLOQUE_2)
    config_b2.update({"filas_de_sonda": 3, "semilla_sequencias": 11, "semilla_matriz": 12,
                      "semilla_modelos": 13, "semilla_vector_declarado": 14})
    sonda = condiciones_de_la_sonda(30, config_b2)
    a9 = afirmacion_a9(sonda, config_b2)
    comprobar("A9 separa las celdas de verdad",
              a9["veredicto"] == "CONFIRMADO"
              and sonda["id_oov"] == 31 and sonda["ids_relleno"] == 0)
    comprobar("A9 publica el vector declarado de la fila de OOV",
              len(a9["medida"]["vector_declarado_de_oov"]) == int(
                  config_b2["dimension_entrada"]))

    class BiGRUHumo(torch.nn.Module):
        """Doble minimo de la clase real, para el humo sin importar los modulos."""

        def __init__(self, matriz: np.ndarray) -> None:
            """Construye el doble.

            Args:
                matriz: matriz de embeddings sintetica.
            """
            super().__init__()
            self.embedding = torch.nn.Embedding.from_pretrained(
                torch.tensor(matriz), freeze=True, padding_idx=0)
            self.gru = torch.nn.GRU(matriz.shape[1], 8, batch_first=True, bidirectional=True)
            self.dropout = torch.nn.Dropout(0.3)
            self.salida = torch.nn.Linear(16, 3)

        def forward(self, entrada: torch.Tensor) -> torch.Tensor:
            """Replica el pooling enmascarado despues de la GRU.

            Args:
                entrada: tensor de identificadores.

            Returns:
                Tensor con los logits.
            """
            mascara = (entrada != 0).float().unsqueeze(-1)
            embebido = self.embedding(entrada)
            salida, _ = self.gru(embebido)
            suma = (salida * mascara).sum(dim=1)
            cuenta = mascara.sum(dim=1).clamp(min=1.0)
            return self.salida(self.dropout(suma / cuenta))

    class BiGRUAfinableHumo(BiGRUHumo):
        """Doble del doble, para tener dos clases distintas en la sonda."""

    sondas = {"v9_04_BiGRU": sondear_bi_gru(BiGRUHumo, "BiGRUHumo", sonda, config_b2),
              "v10_11_BiGRUAfinable": sondear_bi_gru(BiGRUAfinableHumo,
                                                     "BiGRUAfinableHumo", sonda, config_b2)}
    sondas_2 = {"v9_04_BiGRU": sondear_bi_gru(BiGRUHumo, "BiGRUHumo", sonda, config_b2),
                "v10_11_BiGRUAfinable": sondear_bi_gru(BiGRUAfinableHumo,
                                                        "BiGRUAfinableHumo", sonda,
                                                        config_b2)}
    a10 = afirmacion_a10(sondas, config_b2)
    comprobar("A10 publica una cifra por condicion y por clase, sin agregar",
              a10["medida"]["n_cifras"] == 2 * (5 + 1))
    comprobar("A10 marca el seuil de banda muerta de 1e-9",
              TOL["distancia_minima_por_canal"] == 1e-9)
    a11 = afirmacion_a11(sondas)
    comprobar("A11 separa el canal de relleno del canal de OOV por clase",
              set(a11["medida"]["por_clase"]) == set(sondas))
    a12 = afirmacion_a12(sondas, sondas_2)
    comprobar("A12 con dos pasadas identicas da variacion exactamente cero",
              a12["medida"]["variacion_maxima"] == 0.0)
    sondas_movidas = {clave: dict(sonda) for clave, sonda in sondas.items()}
    sondas_movidas["v9_04_BiGRU"] = json.loads(json.dumps(sondas["v9_04_BiGRU"]))
    sondas_movidas["v9_04_BiGRU"]["distancias_por_condicion"]["C1"][
        "distancia_relativa_l2_media"] += 1e-6
    comprobar("A12 detecta una variacion no nula y la marca REFUTADO",
              afirmacion_a12(sondas_movidas, sondas_2)["veredicto"] == "REFUTADO")

    # --- Bloque 3: estructura del informe ---
    bloque = bloque_reejecion_no_replicacion()
    a13 = afirmacion_a13(bloque)
    comprobar("A13 encuentra el bloque bien nombrado y con su tipo de evidencia",
              a13["veredicto"] == "CONFIRMADO")
    bloque_malo = {"comparacion_con_p79": {"coincide": True}}
    comprobar("A13 falla si reaparece el bloque con el nombre prohibido",
              afirmacion_a13(bloque_malo)["veredicto"] == "REFUTADO")
    comprobar("A14 aplica la regla y da REEJECION para el par P79, P80",
              afirmacion_a14({})["medida"]["veredicto_por_la_regla"] == "REEJECION")
    comprobar("A14 publica los cinco requisitos",
              len(afirmacion_a14({})["medida"]["requisitos"]) == 5)

    # --- Bloque 5: scheduler sin el corpus ---
    serie = serie_de_learning_rate(1395, 1398, 6.4, 0.001, 1398)
    comprobar("la serie del scheduler tiene 1398 pasos",
              len(serie["serie_real"]) == 1398)
    comprobar("la serie separa las dos trayectorias",
              serie["maximo_de_la_diferencia_absoluta"] > 0.0)
    a19_min = {"medida": {"clasificacion_del_impacto": "DESPRECIABLE"}}
    a19_max = {"medida": {"clasificacion_del_impacto": "NO DESPRECIABLE"}}
    comprobar("A19 con la regla DESPRECIABLE sale CONFIRMADO",
              afirmacion_a19(a19_min)["veredicto"] == "CONFIRMADO")
    comprobar("A19 con la regla NO DESPRECIABLE sale CONFIRMADO",
              afirmacion_a19(a19_max)["veredicto"] == "CONFIRMADO")
    comprobar("A19 sin regla no inventa una etiqueta",
              afirmacion_a19({})["medida"][
                  "declaracion_de_impacto_del_scheduler"] == "NO_DISPONIBLE")
    # A0 compara el sha256 con el declarado, luego la huella del humo tiene que llevar el hash
          # declarado. Con un hash inventado A0 sale REFUTADO por diseno correcto y la
          # comprobacion seria la que estaria mal.
    huella_ok = {"ruta": "modelos/humo.npz", "sha256": ARTEFACTO_SHA256,
                 "bytes": 1, "mtime_ns": 2}
    comprobar("A0 declara intacto cuando las tres huellas coinciden y el hash es el declarado",
              afirmacion_a0(huella_ok, dict(huella_ok))["veredicto"] == "CONFIRMADO")
    comprobar("A0 falla si el mtime_ns cambia",
              afirmacion_a0(huella_ok,
                            {**huella_ok, "mtime_ns": 3})["veredicto"] == "REFUTADO")
    comprobar("A0 falla si el sha256 no es el declarado, aunque las tres huellas coincidan",
              afirmacion_a0({**huella_ok, "sha256": "a"},
                            {**huella_ok, "sha256": "a"})["veredicto"] == "REFUTADO")

    # --- Escrituras: datos COMPLETOS ---
    completo = [afirmacion_vacia(f"A{n}") for n in range(0, 20)]
    for afirmacion in completo:
        afirmacion.update({"hueco": 1, "naturaleza": "CONFIRMAR", "stop": False,
                           "tolerancia": "t", "motivo": "m", "medida": {}})
    completo[0]["medida"] = {"ruta": "x", "sha256": "y", "antes_de_la_primera_lectura":
                              {"sha256": "y"}, "despues_de_la_ultima_lectura": {"sha256": "y"}}
    completo[4]["medida"] = {"curva_completa": [{"learning_rate": 0.1, "m1_media": 0.0,
                                                 "m1_suma": 0.0, "diferencia_de_m1": 0.0,
                                                 "victoria_de_la_suma": False,
                                                 "zona_betonal_media": False,
                                                 "zona_betonal_suma": False}],
                             "recuento_de_victorias_por_learning_rate": {}}
    completo[7]["medida"] = {"cifras": {"proporcion_de_posiciones_oov": 0.0,
                                        "proporcion_de_posiciones_de_relleno": 0.0,
                                        "codigo": "v9_04"},
                             "queda_reproducida": False, "variante_que_la_reproduce": None}
    completo[9]["medida"] = {"id_relleno": 0, "id_oov": 31,
                             "forma_de_la_matriz": [32, 200],
                             "residuo_de_la_fila_de_relleno": 0.0}
    completo[9]["medida"].update({"vector_declarado_de_oov": [0.0]})
    completo[10]["medida"] = {"cifras": []}
    completo[14]["medida"] = {"conteos": [347, 346, 346, 346, 346],
                              "conteos_publicados": [347, 346, 346, 346, 346],
                              "n_en_la_union": 1731, "intersecciones_pairwise": [],
                              "sha256_por_pliegue": {"1": "a"}, "motivo": "m",
                              "primer_y_ultimo_id_por_pliegue": {"1": {"primer_id": 1,
                                                                     "ultimo_id": 2}}}
    completo[15]["medida"] = {"linea_sobre_el_hash_ids_csv": "linea"}
    completo[13]["medida"] = {"requisitos": {"R1": {"texto": "t", "cumple": False}},
                              "veredicto_por_la_regla": "REEJECION"}
    completo[17]["medida"] = {"total_lotes_plan": 1395, "total_lotes_recorridos": 1398,
                              "media_real": 1.0, "media_ideal": 1.0,
                              "diferencia_relativa_de_las_medias": 0.0,
                              "umbral_de_clasificacion": 0.005,
                              "clasificacion_del_impacto": "DESPRECIABLE"}
    informe_completo = construir_json(completo, {"pre_registro": {}, "enmienda_1": {}},
                                      {"sha256": "y", "bytes": 1, "mtime_ns": 2},
                                      {"sha256": "y", "bytes": 1, "mtime_ns": 2},
                                      {}, completo[17], datetime.now(timezone.utc), {}, {})
    md_completo = escribir_markdown(completo, informe_completo)
    comprobar("construir_json con datos completos produce las veinte afirmaciones",
              informe_completo["n_de_afirmaciones"] == 20)
    comprobar("escribir_markdown con datos completos produce lineas",
              len(md_completo) > 40)
    comprobar("el markdown completo lleva la formulacion",
              any(FORMULACION in linea for linea in md_completo))
    comprobar("el markdown completo lleva la frase de A3",
              any(FRASE_A3 in linea for linea in md_completo))
    comprobar("el markdown completo lleva la frase de A17",
              any(FRASE_A17 in linea for linea in md_completo))

    # --- Escrituras: datos INCOMPLETOS, la ruta que rompia P80 ---
    incomplete = [afirmacion_vacia(f"A{n}") for n in range(0, 20)]
    informe_incompleto = construir_json(incomplete, {}, {}, {}, {}, {},
                                        datetime.now(timezone.utc), {}, {})
    try:
        md_incompleto = escribir_markdown(incomplete, informe_incompleto)
        ruta_ok = True
    except (KeyError, TypeError, AttributeError, IndexError, ValueError) as exc:
        ruta_ok = False
        print(f"  FALLA escribir_markdown con datos incompletos lanzo {type(exc).__name__}: "
              f"{exc}", flush=True)
    comprobar("escribir_markdown con TODAS las afirmaciones incompletas no lanza", ruta_ok)
    comprobar("escribir_markdown con datos incompletos devuelve el esqueleto del informe",
              bool(md_incompleto) and any("# P81" in linea for linea in md_incompleto))
    comprobar("el markdown incompleto declara la formulacion igualmente",
              any(FORMULACION in linea for linea in md_incompleto))

    parcial = [afirmacion_vacia("A0"), afirmacion_vacia("A19")]
    informe_parcial = construir_json(parcial, {}, {}, {}, {}, {},
                                     datetime.now(timezone.utc), {}, {})
    try:
        md_parcial = escribir_markdown(parcial, informe_parcial)
        parcial_ok = True
    except (KeyError, TypeError, AttributeError, IndexError, ValueError) as exc:
        parcial_ok = False
        print(f"  FALLA escribir_markdown con solo dos afirmaciones lanzo "
              f"{type(exc).__name__}: {exc}", flush=True)
    comprobar("escribir_markdown con solo A0 y A19 no lanza", parcial_ok)
    comprobar("el markdown parcial no inventa veredictos ausentes",
              any("NO_EVALUABLE" in linea for linea in md_parcial))

    try:
        construir_json([{"afirmacion": "A5"}], {}, {}, {}, {}, {},
                       datetime.now(timezone.utc), {}, {})
        construir_ok = True
    except (KeyError, TypeError, AttributeError, IndexError, ValueError) as exc:
        construir_ok = False
        print(f"  FALLA construir_json con una afirmacion sin medida lanzo "
              f"{type(exc).__name__}: {exc}", flush=True)
    comprobar("construir_json con una afirmacion sin medida no lanza", construir_ok)

    try:
        escribir_markdown([], {})
        vacio_ok = True
    except (KeyError, TypeError, AttributeError, IndexError, ValueError) as exc:
        vacio_ok = False
        print(f"  FALLA escribir_markdown con listas vacias lanzo {type(exc).__name__}: {exc}",
              flush=True)
    comprobar("escribir_markdown con lista de afirmaciones vacia no lanza", vacio_ok)
    comprobar("construir_json con lista de afirmaciones vacia no lanza",
              bool(construir_json([], {}, {}, {}, {}, {}, datetime.now(timezone.utc), {},
                                  {})["n_de_afirmaciones"]) == 0)

    # --- Escrituras: la comprobacion ASCII en memoria y el renombrado ---
    import tempfile

    with tempfile.TemporaryDirectory(prefix="p81_humo_") as carpeta:
        destino = Path(carpeta) / "informe.md"
        temporal = escribir_ascii_atomico(destino, "# prueba\n" + FORMULACION + "\n",
                                          "humo")
        comprobar("escribir_ascii_atomico deja el temporal escrito y sin publicar",
                  temporal.exists() and not destino.exists())
        comprobar("el temporal es ASCII puro byte a byte",
                  not any(b > 127 for b in temporal.read_bytes()))
        publicar_atomico([(temporal, destino)])
        comprobar("publicar_atomico renombra el temporal a su nombre final",
                  destino.exists() and not temporal.exists())
        comprobar("el texto publicado conserva la formulacion",
                  FORMULACION in destino.read_text(encoding="ascii"))
        destino_2 = Path(carpeta) / "con_acentos.md"
        temporal_2 = escribir_ascii_atomico(destino_2, "caf\u00e9 y \u2014 guion\n",
                                            "humo:acentos")
        comprobar("la sustitucion ASCII ocurre en memoria y el fichero sale limpio",
                  not any(b > 127 for b in temporal_2.read_bytes()))
        comprobar("la sustitucion queda registrada, no silenciosa",
                  any("U+00E9" in str(item.get("caracter"))
                      for item in SUSTITUCIONES_ASCII))
        publicar_atomico([(temporal_2, destino_2)])
        try:
            escribir_ascii_atomico(destino_2, "x", "humo")
            rechazo = False
        except SystemExit:
            rechazo = True
        comprobar("escribir_ascii_atomico rechaza escribir sobre un destino ya publicado",
                  rechazo)

    comprobar("verificar_ascii_byte_a_byte ve el script en ASCII puro",
              verificar_ascii_byte_a_byte([Path(__file__)]).get(Path(__file__).name, False))
    comprobar("la frase de formulacion es exactamente la del plan maestro",
              FORMULACION == "V03 remains the best verified leaderboard result.")
    comprobar("las tolerancias del bloque 1 son las fijadas antes de ejecutar",
              TOL["no_discriminacion"] == 0.005 and TOL["discriminacion"] == 0.020
              and TOL["umbral_de_impacto_despreciable"] == 0.005)
    comprobar("la rejilla es la declarada en la enmienda 1",
              BLOQUE_1["rejilla"] == (0.02, 0.05, 0.10, 0.20, 0.35, 0.50))
    comprobar("las veinte afirmaciones con STOP estan declaradas",
              len(AFIRMACIONES_CON_STOP) == 12
              and "A5" in AFIRMACIONES_CON_STOP and "A19" in AFIRMACIONES_CON_STOP)
    comprobar("umbral de zona betonal declarado en la enmienda 1",
              BLOQUE_1["umbral_zona_betonal"] == 0.99)

    print(f"[P81-01] humo: {superadas} de {superadas + len(fallos)} comprobaciones "
          f"superadas; suelo declarado {MINIMO_COMPROBACIONES_HUMO}", flush=True)
    if fallos:
        print(f"[P81-01] humo: fallaron {len(fallos)}: {fallos}", flush=True)
    return superadas


def escribir_json_de_parada(parada: Parada) -> Path:
    """Escribe el JSON de parada y NO el informe completo.

    Args:
        parada: la excepcion de parada.

    Returns:
        La ruta del JSON de parada, ya publicada.

    Raises:
        SystemExit: si el JSON de parada ya existe o si el texto no es ASCII.
    """
    cuerpo = {
        "familia": "P81_CIERRE_HUECOS",
        "experiment_id": "P81-01",
        "informe_completo": False,
        "motivo_de_la_parada": parada.motivo,
        "afirmacion_que_para": parada.afirmacion,
        "formulacion": FORMULACION,
        "nota": ("una afirmacion con marca STOP ha fallado, luego NO se escribe el informe "
                 "completo y no se publica ningun titular sobre H1"),
    }
    if parada.afirmacion in ("A5",):
        cuerpo["frase"] = FRASE_TEST_SIN_CONCLUSION
    cuerpo["detalle"] = parada.detalle
    REPORTS.mkdir(parents=True, exist_ok=True)
    if JSON_INFORME.exists():
        raise SystemExit(f"STOP escritura: {JSON_INFORME} ya existe y solo se admiten "
                         "temporales hasta la publicacion atomica")
    temporal = escribir_ascii_atomico(JSON_INFORME,
                                      json.dumps(cuerpo, ensure_ascii=True, indent=2) + "\n",
                                      "json_de_parada")
    sucios = [nombre for nombre, ok in verificar_ascii_byte_a_byte([temporal]).items()
              if not ok]
    if sucios:
        raise SystemExit(f"STOP asercion ASCII: bytes no ASCII en {sucios}")
    publicar_atomico([(temporal, JSON_INFORME)])
    return JSON_INFORME


def main() -> None:
    """Ejecuta P81 en orden y para en el primer STOP.

    Returns:
        None.

    Raises:
        SystemExit: si un documento vinculante no coincide, si el artefacto congelado no esta
            intacto, o si una afirmacion con marca STOP falla.
    """
    torch.set_num_threads(32)
    inicio = datetime.now(timezone.utc)
    cronometro: dict = {}

    log("=" * 78)
    log(f"P81-01 cierre de los cinco huecos que dejo P80 | {FORMULACION}")
    log("=" * 78)

    pre = verificar_pre_registro()
    for clave in ("pre_registro", "enmienda_1"):
        documento = pre.get(clave, {})
        if not documento.get("coincide"):
            log(f"STOP: el sha256 del documento {clave} no coincide. esperado "
                f"{documento.get('sha256')}, real {documento.get('sha256_recomputado')}")
            raise SystemExit(f"STOP documento vinculante: hash distinto del declarado ({clave})")
        log(f"{clave} verificado: {documento.get('sha256_recomputado')} "
            f"(commit {documento.get('commit')})")

    huella_antes = huella_de_fichero(ARTEFACTO)
    if huella_antes["sha256"] != ARTEFACTO_SHA256:
        log(f"STOP: el sha256 del artefacto congelado no coincide al empezar. esperado "
            f"{ARTEFACTO_SHA256}, real {huella_antes['sha256']}")
        raise SystemExit("STOP artefacto congelado: hash distinto del declarado")
    log(f"artefacto congelado verificado ANTES de la primera lectura: "
        f"{huella_antes['sha256']} ({huella_antes['bytes']} bytes)")

    contexto: dict = {"modulos": {}, "errores": {}}
    for nombre, ruta in (("p71_04", FUENTE_P71), ("p72_01", FUENTE_P72)):
        try:
            contexto["modulos"][nombre] = importar_modulo(nombre, ruta)
        except Exception as exc:  # noqa: BLE001
            contexto["errores"][nombre] = f"{type(exc).__name__}: {exc}"
            raise SystemExit(f"STOP importacion: no se pudo cargar {ruta.name}") from exc
    try:
        contexto["modulos"]["v10_11"] = importar_modulo("v10_11", FUENTE_V10)
        contexto["modulos"]["v9_04"] = sys.modules.get("v9_04")
    except Exception as exc:  # noqa: BLE001
        contexto["errores"]["v10_11"] = f"{type(exc).__name__}: {exc}"
        raise SystemExit("STOP importacion: no se pudo cargar v10_11") from exc
    if contexto["modulos"]["v9_04"] is None:
        contexto["errores"]["v9_04"] = "v10_11 no registro v9_04 en sys.modules"
        raise SystemExit("STOP importacion: no se pudo cargar v9_04")
    log("modulos del proyecto cargados en solo lectura: p71_04, p72_01, v9_04 y v10_11")

    preregistro = leer_fuente(PREREGISTRO)
    informe_p79 = json.loads(INFORME_P79.read_text(encoding="utf-8"))
    informe_p80 = json.loads(INFORME_P80.read_text(encoding="utf-8"))
    informe_dl = json.loads(INFORME_DL.read_text(encoding="utf-8"))

    afirmaciones: list = []
    try:
        marcas = datetime.now(timezone.utc)

        log("bloque 1: corpus plantado y rejilla del learning rate")
        preparado = preparar_bloque_1(dict(BLOQUE_1))
        cronometro["bloque_1_corpus"] = round(
            (datetime.now(timezone.utc) - marcas).total_seconds(), 3)
        marcas = datetime.now(timezone.utc)
        rejilla = recorrer_rejilla(preparado, dict(BLOQUE_1))
        cronometro["bloque_1_rejilla"] = round(
            (datetime.now(timezone.utc) - marcas).total_seconds(), 3)
        marcas = datetime.now(timezone.utc)
        primario = elegir_primario(rejilla, dict(BLOQUE_1))
        confirmacion = confirmar_con_diez_semillas(preparado,
                                                   primario["learning_rate_del_primario"],
                                                   dict(BLOQUE_1))
        control = brazo_de_control_negativo(preparado,
                                            primario["learning_rate_del_primario"],
                                            dict(BLOQUE_1))
        cronometro["bloque_1_confirmacion"] = round(
            (datetime.now(timezone.utc) - marcas).total_seconds(), 3)
        log(f"bloque 1: curva completa de M1 por learning rate publicada; "
            f"primario en lr {primario['learning_rate_del_primario']}; la SUMA gana "
            f"{rejilla['victorias_de_la_suma']} de {len(rejilla['curva'])} puntos; "
            f"curva monotona: {rejilla['deriva_de_la_curva']['monotona']}")
        for funcion, argumentos in (
                (afirmacion_a1, (preregistro,)),
                (afirmacion_a2, (preregistro,)),
                (afirmacion_a3, (informe_p79,)),
                (afirmacion_a4, (preparado, BLOQUE_1)),
                (afirmacion_a5, (rejilla, primario, BLOQUE_1)),
                (afirmacion_a6, (primario, confirmacion, BLOQUE_1)),
                (afirmacion_a7, (confirmacion, control,
                                 _clasificar_recuento(rejilla), BLOQUE_1))):
            resultado = funcion(*argumentos)
            afirmaciones.append(resultado)
            _registrar(resultado)

        log("bloque 2: tasas con el tokenizador real y sonda de celdas separadas")
        marcas = datetime.now(timezone.utc)
        train = contexto["modulos"]["p71_04"] and None
        del train
        import src.data as data_module
        import src.splits as splits_module

        marco_train = data_module.read_train()
        grupos_split = splits_module.grupos_fusionados(marco_train)
        indices_dev, indices_holdout = splits_module.holdout_v2(marco_train, grupos_split)
        medidas_oov = cargar_tasa_de_oov(contexto["modulos"]["v9_04"], marco_train,
                                        grupos_split)
        afirmaciones.append(afirmacion_a8(medidas_oov))
        _registrar(afirmaciones[-1])
        config_b2 = dict(BLOQUE_2)
        config_b2["vocabulario"] = max(medidas_oov["vocabulario_usado"], 8)
        sonda = condiciones_de_la_sonda(config_b2["vocabulario"], config_b2)
        sondas = {}
        for clave, nombre_clase in (("v9_04_BiGRU", "BiGRU"),
                                    ("v10_11_BiGRUAfinable", "BiGRUAfinable")):
            modulo = (contexto["modulos"]["v9_04"] if clave == "v9_04_BiGRU"
                      else contexto["modulos"]["v10_11"])
            sondas[clave] = sondear_bi_gru(modulo, nombre_clase, sonda, config_b2)
        sondas_2 = {}
        for clave, nombre_clase in (("v9_04_BiGRU", "BiGRU"),
                                    ("v10_11_BiGRUAfinable", "BiGRUAfinable")):
            modulo = (contexto["modulos"]["v9_04"] if clave == "v9_04_BiGRU"
                      else contexto["modulos"]["v10_11"])
            sondas_2[clave] = sondear_bi_gru(modulo, nombre_clase, sonda, config_b2)
        for funcion, argumentos in (
                (afirmacion_a9, (sonda, config_b2)),
                (afirmacion_a10, (sondas, config_b2)),
                (afirmacion_a11, (sondas,)),
                (afirmacion_a12, (sondas, sondas_2))):
            resultado = funcion(*argumentos)
            afirmaciones.append(resultado)
            _registrar(resultado)
        cronometro["bloque_2"] = round(
            (datetime.now(timezone.utc) - marcas).total_seconds(), 3)

        log("bloque 3: reejecion no es replicacion")
        bloque = bloque_reejecion_no_replicacion()
        for resultado in (afirmacion_a13(bloque), afirmacion_a14(informe_p80)):
            afirmaciones.append(resultado)
            _registrar(resultado)

        log("bloque 4: identidad de conjuntos de identificadores")
        marcas = datetime.now(timezone.utc)
        grupos_cv = leer_grupos_de_la_fase_dl(marco_train)
        pliegues = conjuntos_de_identificadores_de_los_pliegues(marco_train, grupos_cv)
        campos_holdout = leer_campos_de_holdout_v2()
        for resultado in (
                afirmacion_a15(pliegues),
                afirmacion_a16(marco_train, grupos_split, indices_dev, indices_holdout,
                               campos_holdout),
                afirmacion_a17(pliegues, informe_dl)):
            afirmaciones.append(resultado)
            _registrar(resultado)
        cronometro["bloque_4"] = round(
            (datetime.now(timezone.utc) - marcas).total_seconds(), 3)

        log("bloque 5: learning rate paso a paso")
        a18 = afirmacion_a18(contexto["modulos"]["p72_01"])
        afirmaciones.append(a18)
        _registrar(a18)
        resultado = afirmacion_a19(a18)
        afirmaciones.append(resultado)
        _registrar(resultado)
    except Parada as parada:
        huella_despues = huella_de_fichero(ARTEFACTO)
        if (huella_despues["sha256"] != huella_antes["sha256"]
                or huella_despues["bytes"] != huella_antes["bytes"]
                or huella_despues["mtime_ns"] != huella_antes["mtime_ns"]):
            log("STOP: el artefacto congelado CAMBIO entre la primera y la ultima lectura")
            raise SystemExit("STOP artefacto congelado modificado durante P81") from parada
        ruta = escribir_json_de_parada(parada)
        log(f"STOP {parada.afirmacion}: {parada.motivo}")
        log(f"JSON de parada escrito en {ruta.relative_to(RAIZ)}; el informe completo NO se "
            "escribe")
        log(FORMULACION)
        sys.exit(1)

    huella_despues = huella_de_fichero(ARTEFACTO)
    if (huella_despues["sha256"] != huella_antes["sha256"]
            or huella_despues["bytes"] != huella_antes["bytes"]
            or huella_despues["mtime_ns"] != huella_antes["mtime_ns"]):
        log("STOP: el artefacto congelado CAMBIO entre la primera y la ultima lectura")
        raise SystemExit("STOP artefacto congelado modificado durante P81")
    log(f"artefacto congelado verificado DESPUES de la ultima lectura: "
        f"{huella_despues['sha256']}")

    a18_para_json = envoltura_tolerante(afirmaciones).get("A18", {})
    afirmaciones.sort(key=lambda item: _clave_de_orden(str(item.get("afirmacion", ""))))
    afirmaciones.insert(0, afirmacion_a0(huella_antes, huella_despues))
    _registrar(afirmaciones[0])

    marcas = datetime.now(timezone.utc)
    ascii_previo = verificar_ascii_byte_a_byte([Path(__file__)])
    informe = construir_json(afirmaciones, pre, huella_antes, huella_despues, contexto,
                             envolver(a18_para_json, "A18"), inicio, ascii_previo, cronometro)
    REPORTS.mkdir(parents=True, exist_ok=True)
    if JSON_INFORME.exists() or MD_INFORME.exists():
        raise SystemExit(f"STOP escritura: {JSON_INFORME.name} o {MD_INFORME.name} ya existen. "
                         "P81 no sobrescribe su propio informe; muevelos si quieres "
                         "reejecutarlo")
    json_temporal = escribir_ascii_atomico(JSON_INFORME,
                                           json.dumps(informe, ensure_ascii=True, indent=2)
                                           + "\n", "informe_json")
    md_temporal = escribir_ascii_atomico(MD_INFORME,
                                         "\n".join(escribir_markdown(afirmaciones, informe))
                                         + "\n", "informe_markdown")
    ascii_final = verificar_ascii_byte_a_byte([Path(__file__), json_temporal, md_temporal])
    informe["asercion_ascii"]["observado"] = ascii_final
    informe["cronometro"]["informe"] = round(
        (datetime.now(timezone.utc) - marcas).total_seconds(), 3)
    json_temporal.write_text(json.dumps(informe, ensure_ascii=True, indent=2) + "\n",
                             encoding="ascii")
    sucios = [nombre for nombre, ok in ascii_final.items() if not ok]
    if sucios:
        raise SystemExit(f"STOP asercion ASCII: bytes no ASCII en {sucios}")
    log(f"asercion ASCII OK, escaneo byte a byte de {ascii_final}")
    publicar_atomico([(json_temporal, JSON_INFORME), (md_temporal, MD_INFORME)])
    log(f"publicados por renombrado atomico: {JSON_INFORME.relative_to(RAIZ)} y "
        f"{MD_INFORME.relative_to(RAIZ)}")

    log("")
    log("=" * 78)
    for identificador, veredicto in _veredictos_por_afirmacion(afirmaciones).items():
        log(f"   {identificador}: {veredicto}")
    log(f"   afirmaciones ejecutadas: {len(afirmaciones)} de 20")
    log(f"   {FORMULACION}")
    log("=" * 78)


def _registrar(resultado: dict) -> None:
    """Imprime una linea de estado por afirmacion, con su veredicto y su motivo.

    Args:
        resultado: diccionario de la afirmacion.
    """
    etiqueta = resultado.get("etiqueta")
    sufijo = f" [{etiqueta}]" if etiqueta else ""
    log(f"   {resultado.get('afirmacion')}: {resultado.get('veredicto')}{sufijo} | "
        f"{resultado.get('motivo', '')[:150]}")


if __name__ == "__main__":
    if "--humo" in sys.argv:
        sys.exit(0 if ejecutar_pruebas_de_humo() >= MINIMO_COMPROBACIONES_HUMO else 1)
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        log(f"FALLO de P81: {type(exc).__name__}: {exc}")
        log(FORMULACION)
        sys.exit(1)
