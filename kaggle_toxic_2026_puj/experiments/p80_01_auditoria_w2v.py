"""p80_01_auditoria_w2v.py

Proposito
---------
Auditar el entrenador Word2Vec del proyecto (H1 a H7) sin reentrenar nada, sin tocar el
artefacto congelado y sin repetir los tres errores de instrumental de P79:

    1. la lectura del scheduler se hace por RECORRIDO DEL AST con mapa de padres, nunca
       buscando hacia atras un encabezado con `for` ni etiquetando el bucle del plan por la
       cadena `planes.append` en una ventana de cuatro lineas;
    2. la identidad de la loss se afirma con el SIGNO CORRECTO,
       `l_codigo + pos + neg_codigo = 0`, con tolerancia y parada, y ademas se comprueba que
       la identidad mal puesta FALLA, porque si tambien se cumpliera la asercion no tendria
       poder discriminante;
    3. la matriz de salida del artefacto congelado NO EXISTE: el npz solo guarda `W`, la tabla
       de entrada. No se sustituye por `W` en silencio: se declara el limite y la afirmacion
       queda como NO APLICABLE POR CONSTRUCCION.

Vinculante
----------
Pre-registro: `docs/p80_preregistro_auditoria_w2v.md`, sha256
`440ab13c610e1bcf35a0b00f096699f738bc59480240dbd934e6c5f3e2f7c81f`, commit `8726c9f`,
escrito y commiteado ANTES de implementar. Si algo de este script contradice al
pre-registro, GANA EL PRE-REGISTRO y la desviacion se declara en el informe.

Las diecinueve afirmaciones A1 a A19 del pre-registro se implementan una a una, en
`afirmacion_a1` ... `afirmacion_a19`. Cada una devuelve su veredicto con el nombre de la
afirmacion, y solo en los tres estados permitidos: CONFIRMADO, REFUTADO o
NO APLICABLE POR CONSTRUCCION.

Regla de alcance
----------------
Este script es la UNICA unidad ejecutable. NO edita ni borra P71 a P79. NO modifica V03 ni sus
umbrales congelados 0.44 / 0.51 / 0.46. NO lee test, ni holdout, ni leaderboard, ni genera
submission. NO escribe el artefacto Word2Vec: lo abre en solo lectura, con sha256, tamano y
`st_mtime_ns` comprobados ANTES de la primera lectura y DESPUES de la ultima, y para si
cualquiera de los tres cambia. NO escribe fuera de `reports/`, de `figuras/p80/` y del propio
script.

Dispositivo y coste
-------------------
CPU, `torch.set_num_threads(32)`. No se usa GPU: P80 es CPU y el presupuesto es de minutos.
No hay entrenamiento: solo producto punto a punto, diferenciacion automatica sobre tensores
diminutos, lectura de fuentes y enumeracion de ficheros.

Ejecucion
---------
    /home/carlos/venv/bin/python experiments/p80_01_auditoria_w2v.py            # el experimento
    /home/carlos/venv/bin/python experiments/p80_01_auditoria_w2v.py --humo     # test de humo

El modo `--humo` corre las pruebas de humo con datos sinteticos: no lee datos, ni informes, ni
el artefacto congelado, no entrena y no escribe nada. Lo unico que lee del proyecto es el
propio script y el texto de `p71_04_entrenar_word2vec.py`, porque A3 tiene que citar su linea
215. Sale antes de la seccion 1, de modo que no puede tener efectos.

Formulacion obligatoria, literal, en el log y en el informe:
    V03 remains the best verified leaderboard result.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
import re
import sys
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

# ===== CONFIGURACION =====
# Rutas, constantes declaradas y tolerancias. Todo el calculo sale de aqui; ninguna funcion
# lleva numeros magicos que no esten tambien en este bloque.

RAIZ = Path(__file__).resolve().parents[1]
EXPERIMENTOS = RAIZ / "experiments"
MODELOS = RAIZ / "modelos"
FEATURES = RAIZ / "features"
REPORTS = RAIZ / "reports"
FIGURAS = RAIZ / "figuras" / "p80"
CONFIG_DIR = RAIZ / "config"
DOCS = RAIZ / "docs"

PREREGISTRO = DOCS / "p80_preregistro_auditoria_w2v.md"
PREREGISTRO_SHA256 = "440ab13c610e1bcf35a0b00f096699f738bc59480240dbd934e6c5f3e2f7c81f"
COMMIT_PREREGISTRO = "8726c9f"

ARTEFACTO = MODELOS / "twitter_domain" / "w2v_twitter_100d.npz"
ARTEFACTO_SHA256 = "302c994d13b75f8f5fbd440eea2c4f4113997e2581524021923b92f8244d705a"

JSON_INFORME = REPORTS / "p80_auditoria_w2v.json"
MD_INFORME = REPORTS / "p80_auditoria_w2v.md"

SUFIJO_TEMPORAL = ".tmp_p80"
FORMULACION = "V03 remains the best verified leaderboard result."

# Ficheros fuente, todos de solo lectura.
FUENTE_P71 = EXPERIMENTOS / "p71_04_entrenar_word2vec.py"
FUENTE_P71_08 = EXPERIMENTOS / "p71_08_word2vec_corregido.py"
FUENTE_P72 = EXPERIMENTOS / "p72_01_dimensiones.py"
FUENTE_V9 = EXPERIMENTOS / "v9_04_dl_secuencial_fasttext.py"
FUENTE_V10 = EXPERIMENTOS / "v10_11_bigru_finetuning.py"
FUENTE_DL = EXPERIMENTOS / "11_dl_desde_cero.py"
INFORME_DL = REPORTS / "dl_desde_cero_resultados.json"
INFORME_P79 = REPORTS / "p79_auditoria_sgns.json"
HOLDOUT_V2 = CONFIG_DIR / "holdout_v2.json"
CORPUS_TWITTER = (RAIZ / "data" / "external" / "corpora" / "twitter_domain"
                  / "corpus_twitter_tokens.txt")

# Umbrales de V03 y su referencia. Se DECLARAN para poder citarlos y comparar; no se tocan.
UMBRALES_V03 = {"HS": 0.44, "TR": 0.51, "AG": 0.46}
V03_CV_DECLARADO = 0.7772114525165721

# Forma declarada del artefacto congelado, tal como la publica el pre-registro.
ARTEFACTO_ESPERADO = {"vocabulario": 77657, "dim": 100, "dtype": "float32",
                      "claves": ["W", "palabras"]}

# Red diminuta para H1, identica a la de P79 para que el contraste sea directo.
RED = {
    "dim": 5,
    "vocabulario": 7,
    "semilla": 20261140,
    "lote": 64,
    "valores_de_k": (1, 5, 10, 20),
    "k_con_evidencia": (5, 10, 20),
    "nota_k_1": "en K = 1 el ratio vale 1 y la asercion es trivialmente cierta",
}

# Tolerancias de la seccion 3 del pre-registro. La clave `tolerancia` de cada afirmacion del
# informe se toma de aqui, para que el JSON y el codigo no puedan separarse.
TOL = {
    "fidelidad_de_la_replica": 1e-12,
    "identidad_de_la_loss": 1e-12,
    "separacion_del_control_negativo": 1e-6,
    "ratio_del_termino_negativo": 1e-12,
    "componentes_de_gradiente_relativo": 1e-6,
    "residuo_de_descomposicion": 1e-9,
    "sonda_de_relleno": 1e-3,
    "recuentos_de_enteros": 0,
    "desvio_del_scheduler": 1e-9,
}

# Recorte declarado para la parte 2 de A8, comparacion con P79.
RECRETE = {"lineas": 2000, "lote": 256}

# Frase del informe de P79 que P80 sustituye, y las claves de cota que P80 no reproduce.
FRASE_SUSTITUIDA_P79 = (
    "el_ratio_entre_objetivos_es_propiedad_de_la_formula_para_ese_par_E_S, "
    "y_el_par_usado_no_es_el_del_entrenamiento"
)
CLAVES_DE_COTA_PROHIBIDAS = (
    "cota_inferior_del_ratio_total",
    "cota_superior_del_ratio_total",
)

# Rutas que P80 no abre de ninguna manera, con el motivo declarado.
RUTAS_PROHIBIDAS = {
    "data/raw/toxic_test.csv": "test de la competencia, prohibido por el pre-registro",
    "features/v2_structured_features.parquet": (
        "features estructuradas v2, prohibidas explicitamente por el pre-registro"),
    "experiments/baseline_v03_frozen/": "V03 congelado, no hace falta: V03 no se toca",
    "submissions/": "submissions, prohibidas por el pre-registro",
    "config/holdout_v2.json::identificadores": (
        "la lista de identificadores del holdout NO se lee ni se deserializa; de ese fichero "
        "solo se leen n_tweets_dev, n_tweets_holdout y hash_ids_csv"),
    "reports/leaderboard*": "ningun score de leaderboard se lee ni se consulta",
}

# Convencion que identifica parametros y variables de validacion. Es una convencion declarada,
# NO una prueba: el JSON publica la lista completa de nombres clasificados, para que un
# humano pueda auditar la clasificacion desde el informe. El componente tiene que ser
# exactamente `va` o `val`, para que `variante` NO se clasifique como validacion.
PATRON_VALIDACION = re.compile(r"^(?:x|t|ids|y|probas|probabilidades)_(?:va|val)$")
NOMBRES_VALIDACION_EXACTOS = {"va", "val", "validacion", "validas"}
# Simetrica de la anterior, para el desempate por nombre del call site. Tambien exige el
# componente entero: `tr_val` no es validacion porque `tr` no es un prefijo admitido, y
# `x_valida` tampoco, porque el componente es `valida` y no `val`.
PATRON_ENTRENAMIENTO = re.compile(r"^(?:x|t|ids|y|probas|probabilidades)_(?:train|tr)$")
NOMBRES_ENTRENAMIENTO_EXACTOS = {"tr", "train", "entrenamiento", "entrenamientos"}
# Llamadas que construirian una particion interna dentro de la funcion de entrenamiento. El
# patron se aplica SOLO a nombres que empiezan por minuscula: `CurvasPliegue()` es una clase y
# no una particion, y sin esta guarda `plieg` la detecta como repartido y marca REFUTADO donde
# no hay fuga, que es peor que no opinar.
PATRON_REPARTO = re.compile(r"split|part|fold|plieg|inner|train_test|kfold", re.IGNORECASE)
PREFIJOS_DE_CLASE = re.compile(r"^[A-Z]")

# Nombres que el pre-registro permite leer de config/holdout_v2.json, y nada mas. La lectura
# se hace con una expresion regular sobre el TEXTO, nunca deserializando el documento entero,
# para que la lista de identificadores no llegue a existir como objeto en memoria.
CAMPOS_PERMITIDOS_DEL_HOLDOUT = {
    "n_tweets_dev": int,
    "n_tweets_holdout": int,
    "hash_ids_csv": str,
}

# Numero minimo de comprobaciones de humo que deben pasar para que el modo --humo devuelva 0.
# Es un suelo declarado, no una cuenta viva: si el total de comprobaciones baja de aqui, el
# modo falla en vez de dar un OK deceptive. Si se anade o se quita una comprobacion, hay que
# actualizar este numero en el mismo commit; el valor actual es el inventario completo.
MINIMO_COMPROBACIONES_HUMO = 30

# Sustituciones ASCII aplicadas a texto ajeno. Se declaran en el informe, nunca en silencio.
SUSTITUCIONES_ASCII: list = []


# ===== FIN CONFIGURACION =====


class Parada(Exception):
    """Parada del pre-registro: una afirmacion con marca STOP ha fallado.

    La captura de esta excepcion en `main` escribe un JSON de parada y no el informe
    completo. Cuando `minimalista` es True, el JSON de parada lleva unicamente el residuo y
    las cadenas de la identidad, que es lo que exige el pre-registro para A9 y A10.

    Attributes:
        afirmacion: identificador de la afirmacion que para, por ejemplo "A9".
        motivo: explicacion en una linea.
        detalle: diccionario con lo minimo que hay que registrar.
        minimalista: si el JSON de parada debe llevar solo el residuo.
    """

    def __init__(self, afirmacion: str, motivo: str, detalle: dict | None = None,
                 minimalista: bool = False) -> None:
        """Construye la parada.

        Args:
            afirmacion: identificador de la afirmacion.
            motivo: explicacion en una linea.
            detalle: diccionario con lo minimo que hay que registrar.
            minimalista: si el JSON de parada debe llevar solo el residuo.
        """
        super().__init__(f"STOP {afirmacion}: {motivo}")
        self.afirmacion = afirmacion
        self.motivo = motivo
        self.detalle = detalle or {}
        self.minimalista = minimalista


# ===== SECCION 1: INFRAESTRUCTURA DE LECTURA, INTEGRIDAD Y ESCRITURA =====


def log(mensaje: str) -> None:
    """Imprime una linea de progreso, con marca de tiempo y flush.

    Args:
        mensaje: texto a mostrar. Se exige ASCII en todo lo impreso.
    """
    marca = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[P80-01] {marca} | {mensaje}", flush=True)


def a_ascii(texto: str, origen: str) -> str:
    """Convierte un texto a ASCII puro, registrando cada sustitucion.

    Un fichero de texto puede traer un acento o un simbolo en una linea citada. Se sustituye el
    caracter por `?` y se registra la sustitucion con su fichero, de forma que el informe
    declara lo que hizo en vez de fingir que el texto era limpio.

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

    Los tres valores se toman en la MISMA llamada, y el sha256 se calcula sobre el contenido
    leido en ese momento. Es la unica forma de que la comparacion "antes" contra "despues"
    signifique algo.

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

    Raises:
        FileNotFoundError: si el fichero no existe.
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


def mapa_de_padres(arbol: ast.AST) -> dict:
    """Construye el mapa de nodo a padre recorriendo el arbol.

    Es la pieza que P79 no tenia: sin este mapa, el unico modo de saber que bucle contiene una
    llamada es buscar hacia atras una linea que empiece por `for`, que es exactamente la
    heuristica que produjo el veredicto erroneo.

    Args:
        arbol: raiz del arbol sintactico.

    Returns:
        Diccionario que mapea cada nodo a su padre.
    """
    padres: dict = {}
    for nodo in ast.walk(arbol):
        for hijo in ast.iter_child_nodes(nodo):
            padres[hijo] = nodo
    return padres


def ancestro_de_tipo(nodo: ast.AST, tipo, padres: dict):
    """Sube por el mapa de padres hasta el primer nodo del tipo pedido.

    Args:
        nodo: nodo de partida.
        tipo: clase de nodo buscada, por ejemplo `ast.For`.
        padres: mapa de padres.

    Returns:
        El ancestro mas proximo del tipo pedido, o None si no hay ninguno.
    """
    actual = padres.get(nodo)
    while actual is not None:
        if isinstance(actual, tipo):
            return actual
        actual = padres.get(actual)
    return None


def ancestros_de_tipo(nodo: ast.AST, tipo, padres: dict) -> list:
    """Devuelve la cadena de ancestros del tipo pedido, del mas proximo al mas lejano.

    Args:
        nodo: nodo de partida.
        tipo: clase de nodo buscada.
        padres: mapa de padres.

    Returns:
        Lista de ancestros que cumplen el tipo, en orden de proximidad.
    """
    cadena = []
    actual = padres.get(nodo)
    while actual is not None:
        if isinstance(actual, tipo):
            cadena.append(actual)
        actual = padres.get(actual)
    return cadena


def ruta_de_atributo(nodo) -> list | None:
    """Descompone una expresion en su camino de nombres, si es una cadena de atributos.

    `np.random.default_rng` se convierte en `["np", "random", "default_rng"]`. Un nombre
    suelto se convierte en una lista de un elemento. Cualquier otra expresion devuelve None,
    incluida una llamada: para mirar el nombre de una llamada hay que pasarle `nodo.func`.

    Args:
        nodo: expresion a descomponer.

    Returns:
        La lista de componentes, o None si la expresion no es un camino de nombres.
    """
    partes: list = []
    actual = nodo
    while isinstance(actual, ast.Attribute):
        partes.append(actual.attr)
        actual = actual.value
    if isinstance(actual, ast.Name):
        partes.append(actual.id)
        partes.reverse()
        return partes
    return None


def es_llamada_de_reparto(nodo, lineas: list) -> dict | None:
    """Dice si un nodo es una llamada que podria construir una particion interna.

    Se aceptan las dos formas de llamada: un atributo (`divisor.split(...)`) y un nombre
    suelto (`partir_interna(...)`), que es como se ve una fuga real en este proyecto.

    Requisito de forma: el nombre de la funcion llamada tiene que empezar por minuscula. En
    este proyecto las clases se nombran en PascalCase y las funciones en minusculas, luego un
    nombre en mayuscula es un constructor, no un repartidor. Sin esta guarda, `CurvasPliegue()`
    casa con `plieg` y declara una particion interna donde no la hay, que es peor que no opinar.

    Args:
        nodo: nodo `ast.Call` a inspeccionar.
        lineas: lineas del fichero, para citar.

    Returns:
        Diccionario con la linea y el texto si es llamada de reparto, None si no.
    """
    if not isinstance(nodo, ast.Call):
        return None
    if isinstance(nodo.func, ast.Attribute):
        partes = ruta_de_atributo(nodo.func)
    elif isinstance(nodo.func, ast.Name):
        partes = [nodo.func.id]
    else:
        return None
    if not partes or not PATRON_REPARTO.search(partes[-1]):
        return None
    if PREFIJOS_DE_CLASE.match(partes[-1]):
        return None
    return {"linea": nodo.lineno,
            "texto": a_ascii(lineas[nodo.lineno - 1].strip(), f"llamada:{nodo.lineno}")}


def es_nombre_de_validacion(nombre: str) -> bool:
    """Clasifica un identificador como nombre de validacion, por convencion declarada.

    La clasificacion se publica en el informe, con la lista completa de nombres, para que no
    sea una caja negra. Los falsos positivos se evitan exigiendo que el componente sea
    exactamente `va` o `val`: asi `variante`, que empieza por `va`, NO se clasifica.

    Args:
        nombre: identificador a clasificar.

    Returns:
        True si el identificador pertenece a la convencion de validacion.
    """
    return nombre in NOMBRES_VALIDACION_EXACTOS or bool(PATRON_VALIDACION.match(nombre))


def es_nombre_de_entrenamiento(nombre: str) -> bool:
    """Clasifica un identificador como nombre de entrenamiento, por convencion declarada.

    Simetrica de `es_nombre_de_validacion`. Exige el componente entero, igual que la de
    validacion: `x_train` y `y_tr` cuentan, `x_trainable` no, y `tr_val` no es ni una cosa ni
    la otra.

    Args:
        nombre: identificador a clasificar.

    Returns:
        True si el identificador pertenece a la convencion de entrenamiento.
    """
    return nombre in NOMBRES_ENTRENAMIENTO_EXACTOS or bool(PATRON_ENTRENAMIENTO.match(nombre))


def verificar_ascii_byte_a_byte(rutas: list) -> dict:
    """Comprueba, byte a byte, que ningun fichero tiene un byte mayor que 127.

    El verificador oficial del proyecto no detecta Latin-1 Supplement, asi que aqui se hace el
    escaneo a pelo, sobre `read_bytes`, que es lo que exige el pre-registro. El resultado se
    devuelve OBSERVADO, nunca como constante.

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


def ruta_temporal(final: Path) -> Path:
    """Ruta temporal de un artefacto, en el mismo directorio que su version final.

    Un `Path.replace` dentro del mismo sistema de ficheros es atomico, y para que lo sea los
    dos nombres tienen que caer en el mismo.

    Args:
        final: ruta del artefacto definitivo.

    Returns:
        La ruta temporal con la que se escribe antes de publicar.
    """
    return final.with_name(final.name + SUFIJO_TEMPORAL)


def publicar_atomico(pares: list) -> None:
    """Renombra todos los temporales a sus nombres finales, uno detras de otro.

    Se llama SOLO despues de que han pasado todas las verificaciones. Los artefactos existen
    con su nombre temporal hasta ese momento, asi que un fallo anterior no deja ni un informe
    a medias ni una figura sin informe.

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


def importar_modulo(nombre: str, ruta: Path):
    """Importa un modulo del proyecto por ruta, sin ejecutar su `main`.

    `importlib.util.spec_from_file_location` mas `exec_module` ejecuta el cuerpo de modulo y
    nada mas: no entra en `if __name__ == "__main__"`, luego no entrena ni escribe.

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


# ===== SECCION 2: MODULOS DEL PROYECTO Y NUCLEO NUMERICO DE H1 =====


def contexto_de_modulos() -> dict:
    """Carga los modulos del proyecto que P80 necesita, y solo esos.

    Se cargan los cuatro que P80 necesita: `p71_04` (la clase `SkipGram` que se ejecuto),
    `p72_01` (el scheduler y sus constantes), y `v9_04` y `v10_11` (las clases de BiGRU que
    A16 sonda). Se cargan los cuatro aqui y no en el momento de sondarlos, para que un fallo de
    importacion quede registrado CON su causa en un solo sitio: si el error se captura donde se
    produce y despues se lee de otro diccionario, basta con que uno se lose para que A16 diga
    "motivo no registrado" y se trague la excepcion.

    Returns:
        Diccionario con las claves `p71`, `p72`, `v9` y `v10`, mas `errores`.
    """
    contexto: dict = {"errores": {}}
    for clave, ruta in (("p71", FUENTE_P71), ("p72", FUENTE_P72),
                        ("v9", FUENTE_V9), ("v10", FUENTE_V10)):
        try:
            contexto[clave] = importar_modulo(f"p80_{clave}", ruta)
        except Exception as exc:  # noqa: BLE001
            contexto["errores"][clave] = f"{type(exc).__name__}: {exc}"
    return contexto


def construir_red_de_prueba(modulo_p71) -> nn.Module:
    """Construye una `SkipGram` diminuta, inicializada como la de P72-01 y en float64.

    Se replica la inicializacion de `p72_01:82-85`: `SkipGram` pone la tabla de salida a cero en
    su constructor, y el entrenamiento la reinicializa con `uniform_(-0.05, 0.05)`. Sin ese
    paso los dos terminos de la loss valdrian lo mismo, y entonces la identidad de A9 no
    tendria poder discriminante.

    Args:
        modulo_p71: modulo `p71_04_entrenar_word2vec.py`.

    Returns:
        La red, en doble, con los parametros activos.
    """
    torch.manual_seed(RED["semilla"])
    modelo = modulo_p71.SkipGram(RED["vocabulario"], RED["dim"])
    with torch.no_grad():
        modelo.salida.weight.uniform_(-0.05, 0.05)
    return modelo.double()


def indices_de_prueba(negativos: int) -> tuple:
    """Dibuja los indices de centro, contexto y negativos de la red de prueba.

    Los indices se repiten a proposito, como en word2vec real, donde un mismo centro aparece
    muchas veces. Toda comparacion de gradientes posterior se hace sobre la matriz ACUMULADA,
    porque es lo que entrega autograd: indexar ese tensor por los indices volveria a
    multiplicar por el numero de repeticiones.

    Args:
        negativos: valor de K.

    Returns:
        Tupla de tres tensores `int64`: centro, ctx y neg de forma `(lote, K)`.
    """
    lote = RED["lote"]
    generador = np.random.default_rng([RED["semilla"], negativos])
    centro = torch.from_numpy(generador.integers(0, RED["vocabulario"], size=lote))
    ctx = torch.from_numpy(generador.integers(0, RED["vocabulario"], size=lote))
    neg = torch.from_numpy(generador.integers(0, RED["vocabulario"],
                                              size=(lote, negativos)))
    return centro.long(), ctx.long(), neg.long()


def tensores_de_punteria(modelo: nn.Module, centro: torch.Tensor, ctx: torch.Tensor,
                        neg: torch.Tensor) -> dict:
    """Calcula los productos punto intermedios del codigo de P71-04.

    Reproduce las lineas `p71_04:211-213`, que no admiten lectura ambigua: se proyecta el
    centro con la tabla de entrada, y el contexto y los negativos con la de salida.

    Args:
        modelo: la red.
        centro: indices centrales, forma `(lote,)`.
        ctx: indices de contexto, forma `(lote,)`.
        neg: indices negativos, forma `(lote, K)`.

    Returns:
        Diccionario con `v`, `u_pos`, `u_neg`, `s_pos` y `s_neg`.
    """
    v = modelo.entrada(centro)
    u_pos = modelo.salida(ctx)
    u_neg = modelo.salida(neg)
    s_pos = (v * u_pos).sum(dim=1)
    s_neg = (v.unsqueeze(1) * u_neg).sum(dim=2)
    return {"v": v, "u_pos": u_pos, "u_neg": u_neg, "s_pos": s_pos, "s_neg": s_neg}


def calcular_perdidas(modelo: nn.Module, centro: torch.Tensor, ctx: torch.Tensor,
                      neg: torch.Tensor) -> dict:
    """Construye tensores y objetivos nuevos, para un unico paso hacia atras.

    Se llama una vez por cada gradiente que se quiera pedir. Reutilizar el mismo objetivo para
    dos llamadas a `backward` falla, porque el grafo ya se consumio, y ademas los dos objetivos
    de una misma llamada comparten el subgrafo del termino positivo, luego una sola llamada no
    bastaria para los cuatro gradientes que A4 necesita.

    Returns:
        Diccionario con `tensores`, `objetivo_codigo`, `objetivo_referencia` y
        `objetivo_solo_positivo`, mas los tres terminos descompuestos.
    """
    tensores = tensores_de_punteria(modelo, centro, ctx, neg)
    tensores["centro"] = centro
    tensores["ctx"] = ctx
    tensores["neg"] = neg
    pos = torch.nn.functional.logsigmoid(tensores["s_pos"]).mean()
    neg_codigo = torch.nn.functional.logsigmoid(-tensores["s_neg"]).mean()
    neg_referencia = torch.nn.functional.logsigmoid(-tensores["s_neg"]).sum(dim=1).mean()
    return {
        "tensores": tensores,
        "objetivo_codigo": -(pos + neg_codigo),
        "objetivo_referencia": -(pos + neg_referencia),
        "objetivo_solo_positivo": -pos,
        "pos": pos,
        "neg_codigo": neg_codigo,
        "neg_referencia": neg_referencia,
    }


def loss_replica_local(modelo: nn.Module, centro: torch.Tensor, ctx: torch.Tensor,
                       neg: torch.Tensor) -> float:
    """Replica local de la loss de `SkipGram.forward`, escrita de forma independiente.

    Se escribe con la misma matematica pero con el codigo reordenado: se calcula el signo por
    separado y se compone la media al final. Si el resultado coincide con el del codigo que se
    ejecuto, la replica es fiel; si no, la replica esta mal.

    Args:
        modelo: la red.
        centro: indices centrales.
        ctx: indices de contexto.
        neg: indices negativos.

    Returns:
        El valor de la loss del codigo, como `float`.
    """
    with torch.no_grad():
        tensores = tensores_de_punteria(modelo, centro, ctx, neg)
        logsig_pos = torch.nn.functional.logsigmoid(tensores["s_pos"])
        logsig_neg = torch.nn.functional.logsigmoid(-tensores["s_neg"])
        replicado = -float(logsig_pos.sum() / logsig_pos.shape[0]
                           + logsig_neg.sum() / logsig_neg.numel())
    return replicado


def terminos_de_la_loss(modelo: nn.Module, centro: torch.Tensor, ctx: torch.Tensor,
                        neg: torch.Tensor) -> dict:
    """Descompone la loss del codigo y la de la referencia SGNS en sus terminos.

    La loss del codigo promedia el termino negativo sobre el LOTE y sobre K. La de la
    referencia lo SUMA sobre K y promedia sobre el lote. La descomposicion deja el termino
    positivo comun, que es lo que A2 comprueba.

    Args:
        modelo: la red.
        centro: indices centrales.
        ctx: indices de contexto.
        neg: indices negativos.

    Returns:
        Diccionario con `pos`, `neg_codigo`, `neg_referencia`, `loss_codigo` y
        `loss_referencia`.
    """
    with torch.no_grad():
        partes = calcular_perdidas(modelo, centro, ctx, neg)
        return {
            "pos": float(partes["pos"]),
            "neg_codigo": float(partes["neg_codigo"]),
            "neg_referencia": float(partes["neg_referencia"]),
            "loss_codigo": float(partes["objetivo_codigo"]),
            "loss_referencia": float(partes["objetivo_referencia"]),
            "lote": int(centro.shape[0]),
        }


def gradiente_autograd(modelo: nn.Module, indice: str, objetivo: torch.Tensor) -> torch.Tensor:
    """Devuelve el gradiente acumulado de una tabla del modelo para un objetivo dado.

    Se pide el gradiente de la matriz ENTERA, con las repeticiones de los indices ya sumadas,
    que es la magnitud con la que hay que comparar la descomposicion analitica.

    Args:
        modelo: la red.
        indice: `"entrada"` o `"salida"`, la tabla cuyo gradiente se quiere.
        objetivo: escalar del que se deriva.

    Returns:
        Tensor con la misma forma que la tabla, desacoplado del grafo.
    """
    modelo.zero_grad(set_to_none=True)
    objetivo.backward()
    tabla = getattr(modelo, indice).weight
    return tabla.grad.detach().clone()


def gradiente_del_termino_positivo(modelo: nn.Module, tensores: dict) -> dict:
    """Gradiente del termino POSITIVO de la loss, y solo del termino positivo.

    La forma cerrada es

        dv     = -sigma(-s_pos) u_pos / B
        du_pos = -sigma(-s_pos) v     / B

    El termino positivo no lleva factor: pesa 1 en el codigo y en la referencia. Se separa en
    su propia funcion, y no como el caso `f = 0` de una funcion unica, porque un componente
    tiene que ser ADITIVO: si la funcion devolviese el total, al sumar los dos componentes el
    termino positivo se contaria dos veces, ni la suma reprodujese a autograd ni la razon de
    los componentes negativos fuese 1/K.

    Args:
        modelo: la red.
        tensores: diccionario de `tensores_de_punteria` mas los tres indices.

    Returns:
        Diccionario con los tensores `entrada` y `salida` del componente positivo.
    """
    v = tensores["v"]
    u_pos = tensores["u_pos"]
    s_pos = tensores["s_pos"]
    lote = v.shape[0]
    sigma_pos = torch.sigmoid(-s_pos)
    dv = (-sigma_pos).unsqueeze(1) * u_pos / lote
    du_pos = (-sigma_pos).unsqueeze(1) * v / lote
    gradiente_entrada = torch.zeros_like(modelo.entrada.weight)
    gradiente_entrada.index_add_(0, tensores["centro"], dv)
    gradiente_salida = torch.zeros_like(modelo.salida.weight)
    gradiente_salida.index_add_(0, tensores["ctx"], du_pos)
    return {"entrada": gradiente_entrada, "salida": gradiente_salida}


def gradiente_del_termino_negativo(modelo: nn.Module, tensores: dict,
                                   factor_negativo: float) -> dict:
    """Gradiente del termino NEGATIVO de la loss, y solo del termino negativo.

    La forma cerrada es

        dv     = f SUM_k sigma(s_neg_bk) u_neg_bk / B
        du_neg = f sigma(s_neg_bk) v                   / B

    El signo es POSITIVO porque la loss lleva un menos delante: `-(pos + neg)`. Con
    `f = 1/K` el codigo queda con el peso 1/K de la repulsion, y con `f = 1` la referencia. El
    1/K es exacto y no hace falta ningun factor 1/K adicional: al multiplicar por `f` y dividir
    por `B` sale ya 1/(B*K), y `index_add_` es lineal, luego un factor 1/K por ocurrencia da
    exactamente 1/K tras la acumulacion.

    Args:
        modelo: la red.
        tensores: diccionario de `tensores_de_punteria` mas los tres indices.
        factor_negativo: el factor `f` del termino de repulsion.

    Returns:
        Diccionario con los tensores `entrada` y `salida` del componente negativo.
    """
    v = tensores["v"]
    u_neg = tensores["u_neg"]
    s_neg = tensores["s_neg"]
    lote = v.shape[0]
    sigma_neg = torch.sigmoid(s_neg)
    dv = (factor_negativo * (sigma_neg.unsqueeze(-1) * u_neg).sum(dim=1)) / lote
    du_neg = (factor_negativo * sigma_neg.unsqueeze(-1) * v.unsqueeze(1)) / lote
    gradiente_entrada = torch.zeros_like(modelo.entrada.weight)
    gradiente_entrada.index_add_(0, tensores["centro"], dv)
    gradiente_salida = torch.zeros_like(modelo.salida.weight)
    gradiente_salida.index_add_(0, tensores["neg"].reshape(-1),
                                du_neg.reshape(-1, RED["dim"]))
    return {"entrada": gradiente_entrada, "salida": gradiente_salida}


def sumar_componentes(componente_primero: dict, componente_segundo: dict) -> dict:
    """Suma dos componentes de gradiente tabla a tabla.

    Args:
        componente_primero: primer componente.
        componente_segundo: segundo componente.

    Returns:
        Diccionario con la suma de las dos tablas.
    """
    return {tabla: componente_primero[tabla] + componente_segundo[tabla]
            for tabla in ("entrada", "salida")}


def residuo_infinito(a: torch.Tensor, b: torch.Tensor) -> float:
    """Norma del maximo absoluto de la diferencia entre dos tensores.

    Args:
        a: primer tensor.
        b: segundo tensor, de la misma forma.

    Returns:
        La norma del maximo absoluto de `a - b`.
    """
    return float((a.detach() - b.detach()).abs().max())


def norma(tensor: torch.Tensor) -> float:
    """Norma euclidea de un tensor aplanado.

    Args:
        tensor: tensor de entrada.

    Returns:
        La norma euclidea, como `float`.
    """
    return float(torch.linalg.vector_norm(tensor.detach().reshape(-1)))


# ===== SECCION 3: LECTURA DEL SCHEDULER POR AST =====


def resolver_alias_de_modulo(arbol: ast.AST, alias: str, nombre_base: str,
                            modulo_baseado: object = None) -> object:
    """Resuelve que modulo de disco representa un alias, siguiendo el codigo, no la costumbre.

    En `p72_01`, `m` vale `cx["mod"]`, `cx["mod"]` vale `base`, y `base` es el modulo cargado
    desde `experiments/p71_04_entrenar_word2vec.py`. Resolver esa cadena por el AST evita tener
    que escribir a mano "el alias `m` es el modulo de P71-04", que seria una suposicion.

    Args:
        arbol: raiz del arbol sintactico del fichero que se lee.
        alias: nombre del alias a resolver, por ejemplo `"m"`.
        nombre_base: nombre de fichero del modulo que se espera.
        modulo_baseado: modulo ya resuelto. Si se pasa, la cadena de alias se sigue
            comprobando igual, pero no se importa nada: es lo que permite que la prueba de
            humo trabaje con un modulo sintetico sin abrir ficheros del proyecto.

    Returns:
        El modulo del alias.

    Raises:
        Parada: si la cadena no se resuelve o no apunta al modulo esperado.
    """
    valores_de_cx: dict = {}
    subscript_del_alias = None
    base_asignada = False
    exec_module_presente = False
    rutas_py = []
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Constant) and isinstance(nodo.value, str) \
                and nodo.value.endswith(".py"):
            rutas_py.append(nodo.value)
        if isinstance(nodo, ast.Call) and isinstance(nodo.func, ast.Attribute) \
                and nodo.func.attr == "exec_module":
            exec_module_presente = True
        if isinstance(nodo, ast.Assign) and len(nodo.targets) == 1 \
                and isinstance(nodo.targets[0], ast.Name):
            nombre = nodo.targets[0].id
            if nombre == alias and isinstance(nodo.value, ast.Subscript):
                subscript_del_alias = nodo.value
            if nombre == "base":
                base_asignada = True
            if nombre == "cx" and isinstance(nodo.value, ast.Dict):
                for clave, valor in zip(nodo.value.keys, nodo.value.values):
                    if isinstance(clave, ast.Constant) and isinstance(clave.value, str):
                        valores_de_cx[clave.value] = valor
    if subscript_del_alias is None or not isinstance(subscript_del_alias.slice, ast.Constant):
        raise Parada("A6", f"no se resuelve la asignacion de {alias} a cx[...mod...]",
                     {"alias": alias})
    clave_de_cx = subscript_del_alias.slice.value
    if clave_de_cx not in valores_de_cx:
        raise Parada("A6", f"la clave {clave_de_cx!r} no aparece en el diccionario cx",
                     {"clave": clave_de_cx})
    valor_de_mod = valores_de_cx[clave_de_cx]
    if not isinstance(valor_de_mod, (ast.Name, ast.Attribute)):
        raise Parada("A6", "la entrada de cx no es ni un nombre ni un atributo de modulo",
                     {"clave": clave_de_cx, "tipo": type(valor_de_mod).__name__})
    if not base_asignada or not exec_module_presente:
        raise Parada("A6", "no se encuentra la carga por importlib del modulo base",
                     {"base_asignada": base_asignada, "exec_module": exec_module_presente})
    candidatas = [ruta for ruta in rutas_py if ruta.endswith(nombre_base)]
    if not candidatas:
        raise Parada("A6", "no aparece ninguna ruta de modulo con el nombre esperado",
                     {"esperado": nombre_base, "rutas_en_el_fichero": rutas_py})
    if modulo_baseado is not None:
        return modulo_baseado
    modulo = sys.modules.get("p80_p71")
    if modulo is None:
        modulo = importar_modulo("p80_modulo_base", EXPERIMENTOS / candidatas[0])
    return modulo


def resolver_entero(nodo, modulos: dict) -> int:
    """Resuelve una expresion a entero, con una gramatica minima y cerrada.

    Se admiten solo tres formas: una constante entera, un atributo `<alias>.<NOMBRE>` de uno
    de los modulos dados, y una suma de formas admitidas. Cualquier otra cosa detiene la
    auditoria: no hay valor por omision y no hay suposicion.

    Args:
        nodo: expresion a resolver.
        modulos: diccionario `{alias: modulo}`.

    Returns:
        El valor entero.

    Raises:
        Parada: si la expresion no es de las formas admitidas.
    """
    if isinstance(nodo, ast.Constant) and isinstance(nodo.value, int) \
            and not isinstance(nodo.value, bool):
        return int(nodo.value)
    partes = ruta_de_atributo(nodo)
    if partes is not None and len(partes) == 2 and partes[0] in modulos:
        valor = getattr(modulos[partes[0]], partes[1], None)
        if isinstance(valor, int) and not isinstance(valor, bool):
            return int(valor)
    if isinstance(nodo, ast.BinOp) and isinstance(nodo.op, ast.Add):
        return resolver_entero(nodo.left, modulos) + resolver_entero(nodo.right, modulos)
    raise Parada("A6", f"expresion no resoluble a entero: {ast.dump(nodo)[:120]}",
                 {"tipo_de_nodo": type(nodo).__name__})


def es_append_sobre(nodo, nombre: str) -> bool:
    """Dice si el nodo es una llamada `nombre.append(...)`.

    Args:
        nodo: nodo a inspeccionar.
        nombre: nombre del objeto que se anade.

    Returns:
        True si el nodo es la llamada buscada.
    """
    return (isinstance(nodo, ast.Call) and isinstance(nodo.func, ast.Attribute)
            and nodo.func.attr == "append" and isinstance(nodo.func.value, ast.Name)
            and nodo.func.value.id == nombre)


def bucles_con_append_sobre(padre, nombre: str, padres: dict) -> dict:
    """Localiza los bucles cuyo cuerpo anade elementos al objeto indicado.

    Args:
        padre: funcion o modulo que contiene la construccion.
        nombre: nombre del objeto al que se anade.
        padres: mapa de padres.

    Returns:
        Diccionario `{id_bucle: nodo_For}` con los bucles donde hay `append` sobre el nombre.

    Raises:
        Parada: si hay `append` repartidos en mas de un bucle.
    """
    bucles: dict = {}
    n_appends = 0
    for nodo in ast.walk(padre):
        if es_append_sobre(nodo, nombre):
            n_appends += 1
            for ancestro in ancestros_de_tipo(nodo, ast.For, padres):
                bucles.setdefault(id(ancestro), ancestro)
    if n_appends == 0:
        raise Parada("A6", f"no se encuentra ningun append sobre {nombre}", {"objeto": nombre})
    if len(bucles) > 1:
        raise Parada("A6", f"los append sobre {nombre} estan repartidos en varios bucles",
                     {"n_bucles": len(bucles)})
    return bucles


def resolver_rango_de_iterador(nodo_for: ast.For, padres: dict, modulos: dict) -> tuple:
    """Interpreta el iterador de un `for`, y solo en las dos formas admitidas.

    Se admiten `range(N)`, que da `0 .. N - 1`, y `enumerate(x, start=k)`, que da
    `k .. k + len(x) - 1`. Cualquier otra forma de iterador detiene la auditoria.

    Args:
        nodo_for: el nodo `ast.For`.
        padres: mapa de padres del arbol.
        modulos: diccionario `{alias: modulo}` para resolver `range(m.EPOCAS)`.

    Returns:
        Tupla `(inicio, fin)` del rango de valores de la variable de iteracion.

    Raises:
        Parada: si el iterador no es de las dos formas admitidas o no se puede resolver.
    """
    iterador = nodo_for.iter
    if not isinstance(iterador, ast.Call):
        raise Parada("A6", "el iterador del bucle no es una llamada",
                     {"tipo": type(iterador).__name__})
    partes = ruta_de_atributo(iterador.func)
    if partes == ["range"]:
        argumentos = list(iterador.args)
        if len(argumentos) != 1:
            raise Parada("A6", f"range con {len(argumentos)} argumentos no esta admitido", {})
        n = resolver_entero(argumentos[0], modulos)
        return (0, n)
    if partes == ["enumerate"]:
        argumentos = list(iterador.args)
        palabras = {k.arg: k.value for k in iterador.keywords}
        if len(argumentos) != 1:
            raise Parada("A6", "enumerate con mas de un argumento posicional no esta admitido",
                         {})
        inicio = resolver_entero(palabras["start"], modulos) if "start" in palabras else 0
        objetivo = argumentos[0]
        if not isinstance(objetivo, ast.Name):
            raise Parada("A6", "el objeto de enumerate no es un nombre", {})
        padre = ancestro_de_tipo(nodo_for, (ast.FunctionDef, ast.Module), padres)
        if padre is None:
            raise Parada("A6", "no se encuentra la funcion que contiene el bucle", {})
        n, _rango_del_plan = resolver_longitud_por_append(padre, objetivo.id, padres, modulos)
        return (inicio, inicio + n)
    raise Parada("A6", f"forma de iterador no admitida: {ast.dump(iterador)[:120]}",
                 {"funcion_del_iterador": partes})


def resolver_longitud_por_append(padre, nombre: str, padres: dict, modulos: dict) -> tuple:
    """Resuelve cuantas veces se ejecuta el `append` que construye una lista.

    El objeto que se itera en `enumerate(planes, start=1)` esta vacio en su inicializador, asi
    que su longitud no se puede leer ahi. Lo que si se puede es lo siguiente, y todo por el
    codigo:

    1. localizar el `append` sobre ese nombre, y comprobar que todos los que hay estan dentro
       del MISMO bucle, que es el que construye la lista;
    2. resolver el rango de ese bucle, que es `range(m.EPOCAS)` con `m` resuelto desde el
       modulo importado;
    3. la longitud de la lista es el numero de EJECUCIONES del `append`, que es la longitud del
       rango del bucle, no el numero de sentencias `append` escritas: hay una sola sentencia
       `planes.append(c)` y se ejecuta EPOCAS veces.

    Args:
        padre: funcion o modulo que contiene la construccion.
        nombre: nombre de la lista.
        padres: mapa de padres.
        modulos: diccionario `{alias: modulo}`, que hace falta porque el bucle del plan se
            resuelve con `range(m.EPOCAS)`.

    Returns:
        Tupla con la longitud resuelta y el rango del bucle constructor.

    Raises:
        Parada: si no hay `append`, si estan en varios bucles o si el rango no se resuelve.
    """
    bucles = bucles_con_append_sobre(padre, nombre, padres)
    unico = next(iter(bucles.values()))
    rango = resolver_rango_de_iterador(unico, padres, modulos)
    n_appends = sum(1 for nodo in ast.walk(padre) if es_append_sobre(nodo, nombre))
    if rango[1] <= rango[0]:
        raise Parada("A6", "el bucle que construye la lista no produce ningun elemento",
                     {"rango": list(rango), "sentencias_append": n_appends})
    return rango[1] - rango[0], rango


def bucle_del_plan(nodo_for: ast.For, nombre_objeto: str) -> bool:
    """Dice si el cuerpo del bucle contiene un `append` sobre el objeto dado.

    Este es el discriminador entre el bucle del plan y el de entrenamiento. Sale del AST: no
    de buscar la cadena `planes.append` en una ventana de cuatro lineas alrededor de la
    llamada, que es la heuristica que produjo el error de P79.

    Args:
        nodo_for: el bucle que se inspecciona.
        nombre_objeto: nombre del objeto al que se anade, por ejemplo `"planes"`.

    Returns:
        True si el cuerpo del bucle anade al objeto.
    """
    for sentencia in nodo_for.body:
        for nodo in ast.walk(sentencia):
            if es_append_sobre(nodo, nombre_objeto):
                return True
    return False


def resolver_semilla(nodo, variable: str, valor: int, modulos: dict) -> int:
    """Evalua el argumento del generador para UN valor de la variable de iteracion.

    Se admiten la constante entera, el atributo de modulo, y la suma de ambos con la variable
    del bucle. Cualquier otra forma detiene la auditoria.

    Args:
        nodo: expresion del argumento.
        variable: nombre de la variable de iteracion del bucle.
        valor: valor concreto de esa variable.
        modulos: diccionario `{alias: modulo}`.

    Returns:
        El entero resuelto.

    Raises:
        Parada: si la expresion no es de las formas admitidas.
    """
    if isinstance(nodo, ast.Name) and nodo.id == variable:
        return int(valor)
    if isinstance(nodo, ast.Constant) and isinstance(nodo.value, int) \
            and not isinstance(nodo.value, bool):
        return int(nodo.value)
    if isinstance(nodo, ast.BinOp) and isinstance(nodo.op, ast.Add):
        return (resolver_semilla(nodo.left, variable, valor, modulos)
                + resolver_semilla(nodo.right, variable, valor, modulos))
    return resolver_entero(nodo, modulos)


def auditar_semillas_por_ast(texto: str, nombre: str, modulos: dict,
                            modulo_baseado: object = None) -> dict:
    """Lee las llamadas al generador de un fichero y resuelve el rango de semillas de cada una.

    Procedimiento, sin ninguna regla de proximidad:

    1. se lee el fichero entero y se registran sus lineas y su sha256;
    2. se localiza cada llamada `np.random.default_rng` recorriendo el AST con mapa de padres,
       comparando el camino de atributos del nodo `Call` con `['np', 'random', 'default_rng']`;
    3. el bucle contenedor se obtiene leyendo el padre `ast.For` del mapa de padres;
    4. el iterador se interpreta, y solo se admiten `range(N)` y `enumerate(x, start=k)`, con
       `N` resuelto desde el modulo importado y `len(x)` contado sobre el objeto que se itera;
    5. el bucle del plan se distingue porque su cuerpo contiene un `append` sobre `planes`;
    6. para cada llamada se devuelven su linea, su bucle contenedor con linea y texto, el rango
       resuelto y el conjunto de enteros resultante.

    Args:
        texto: contenido integro del fichero.
        nombre: nombre del fichero, para las citas.
        modulos: diccionario `{alias: modulo}` usado para resolver `m.SEMILLA` y `m.EPOCAS`.
        modulo_baseado: modulo ya resuelto para el alias `m`, si se tiene.

    Returns:
        Diccionario con las llamadas, el rango de cada una y el conjunto de semillas.

    Raises:
        Parada: si el iterador no es de las formas admitidas o si un valor no se resuelve.
    """
    arbol = ast.parse(texto, filename=nombre)
    padres = mapa_de_padres(arbol)
    lineas = texto.splitlines()
    modulos = dict(modulos)
    modulos["m"] = resolver_alias_de_modulo(arbol, "m", FUENTE_P71.name, modulo_baseado)

    llamadas: list = []
    for nodo in ast.walk(arbol):
        if not (isinstance(nodo, ast.Call) and ruta_de_atributo(nodo.func)
                == ["np", "random", "default_rng"]):
            continue
        registro = {
            "linea": nodo.lineno,
            "texto_literal": a_ascii(lineas[nodo.lineno - 1].strip(),
                                     f"{nombre}:{nodo.lineno}"),
            "argumento": ast.unparse(nodo.args[0]) if nodo.args else "",
        }
        bucle = ancestro_de_tipo(nodo, ast.For, padres)
        if bucle is None:
            registro.update({
                "bucle_contenedor": None,
                "rol": "fuera_de_bucle",
                "rango_de_epocas": None,
                "conjunto_de_semillas": None,
                "semillas_por_epoca": None,
                "nota": ("la llamada no esta dentro de ningun bucle, luego no tiene rango que "
                         "resolver: se registra y se deja tal cual"),
            })
            llamadas.append(registro)
            continue
        variable = bucle.target.elts[0] if isinstance(bucle.target, ast.Tuple) else bucle.target
        if not isinstance(variable, ast.Name):
            raise Parada("A6", "el objetivo del bucle no es un nombre", {})
        rango = resolver_rango_de_iterador(bucle, padres, modulos)
        semillas = [resolver_semilla(nodo.args[0], variable.id, valor, modulos)
                    for valor in range(rango[0], rango[1])]
        registro["bucle_contenedor"] = {
            "linea": bucle.lineno,
            "texto_literal": a_ascii(lineas[bucle.lineno - 1].strip(),
                                     f"{nombre}:{bucle.lineno}"),
            "iterador": ast.unparse(bucle.iter),
            "variable": variable.id,
        }
        registro["rol"] = "plan" if bucle_del_plan(bucle, "planes") else "entrenamiento"
        registro["rango_de_epocas"] = {"inicio": rango[0], "fin": rango[1],
                                       "longitud": rango[1] - rango[0]}
        registro["conjunto_de_semillas"] = sorted(set(semillas))
        registro["semillas_por_epoca"] = {str(valor): semilla for valor, semilla
                                          in zip(range(rango[0], rango[1]), semillas)}
        llamadas.append(registro)

    return {
        "fuente": nombre,
        "n_lineas": len(lineas),
        "sha256": sha256_de_bytes(texto.encode("utf-8")),
        "procedimiento": {
            "localizacion": "recorrido del AST con mapa de padres, comparando el camino de "
                            "atributos del nodo Call con ['np', 'random', 'default_rng']",
            "prohibido": ["buscar la cadena default_rng en el texto",
                          "buscar hacia atras un encabezado con for en una ventana de lineas",
                          "etiquetar el bucle del plan por la cadena planes.append en una "
                          "ventana de cuatro lineas"],
            "iteradores_admitidos": ["range(N)", "enumerate(x, start=k)"],
            "como_se_resuelve_len_de_enumerate": (
                "el objeto se construye con `append` dentro de un unico bucle, y su longitud es "
                "el numero de ejecuciones de ese `append`, que es la longitud del rango del "
                "bucle una vez resuelto `N` desde el modulo importado. No es el numero de "
                "sentencias `append` escritas"),
            "discriminador_del_bucle_del_plan": "el cuerpo del For contiene un Call "
                                                "Attribute(attr='append') sobre el Name "
                                                "'planes'",
            "resolucion_del_alias": "m <- cx['mod'] <- base <- experiments/"
                                    "p71_04_entrenar_word2vec.py, siguiendo el codigo por AST",
            "forma_admitida_para_la_entrada_mod_de_cx": "un Name o un Attribute; en p72_01 es "
                                                         "el Name `base`, que es el modulo "
                                                         "cargado por importlib",
        },
        "llamadas": llamadas,
    }


# ===== SECCION 4: H1, LA LOSS (A1, A2, A3) =====


def afirmacion_a1(modulo_p71) -> dict:
    """A1. La replica local de la loss es fiel al codigo que se ejecuto.

    Se ejecuta la `SkipGram.forward` REAL de `p71_04:199-216`, importada del fichero, con los
    mismos pesos y los mismos indices que la replica, para los cuatro valores de K.

    Returns:
        Diccionario de la afirmacion con su veredicto.

    Raises:
        Parada: si la clase importada no acepta la firma de llamada, o si falla al ejecutar.
    """
    diferencias = {}
    for k in RED["valores_de_k"]:
        modelo = construir_red_de_prueba(modulo_p71)
        centro, ctx, neg = indices_de_prueba(k)
        try:
            with torch.no_grad():
                loss_real = float(modelo(centro, ctx, neg))
        except TypeError as exc:
            raise Parada("A1", f"la clase importada no acepta la firma de llamada: {exc}",
                         {"k": k}) from exc
        except Exception as exc:  # noqa: BLE001
            raise Parada("A1", f"la clase importada falla al ejecutar forward: {exc}",
                         {"k": k}) from exc
        replicado = loss_replica_local(modelo, centro, ctx, neg)
        diferencias[str(k)] = abs(replicado - loss_real)
    maximo = max(diferencias.values())
    completa = len(diferencias) == len(RED["valores_de_k"])
    exito = completa and maximo <= TOL["fidelidad_de_la_replica"]
    if not exito:
        raise Parada("A1", "la replica local no reproduce la loss del codigo",
                     {"diferencias_por_k": diferencias, "maximo": maximo,
                      "tolerancia": TOL["fidelidad_de_la_replica"]})
    return {
        "afirmacion": "A1",
        "hallazgo": "H1",
        "naturaleza": "CONFIRMAR",
        "que_se_afirma": "la loss reimplementada en P80 coincide con SkipGram.forward de "
                         "p71_04:199-216 con los mismos pesos y los mismos indices",
        "tolerancia": TOL["fidelidad_de_la_replica"],
        "unidad": "diferencia absoluta sobre el valor de la loss, en float64",
        "medida": {
            "diferencias_por_k": diferencias,
            "maximo": maximo,
            "los_cuatro_k_evaluados": completa,
            "clase_importada_acepta_la_firma": completa,
            "clase": "p71_04_entrenar_word2vec.SkipGram",
        },
        "exito": "la diferencia maxima sobre los cuatro K es menor o igual que la tolerancia",
        "fallo": "cualquier diferencia mayor, o que la clase importada no acepte la firma",
        "stop_si_falla": True,
        "veredicto": "CONFIRMADO",
    }


def afirmacion_a2(modulo_p71) -> dict:
    """A2. El termino positivo del codigo es identico al de la referencia.

    Ademas de comparar el termino, se comprueba que NO depende de los negativos: se recalcula
    con la matriz de negativos en orden inverso y tiene que dar lo mismo. Esa segunda
    comprobacion es la que da poder discriminante, porque si el termino positivo dependiera de
    K, la identidad de A9 no separaria una formula de su negation.

    Returns:
        Diccionario de la afirmacion con su veredicto.

    Raises:
        Parada: si el pre-registro marca STOP para A2 y la comprobacion falla.
    """
    diferencias = {}
    insensibilidad = {}
    for k in RED["valores_de_k"]:
        modelo = construir_red_de_prueba(modulo_p71)
        centro, ctx, neg = indices_de_prueba(k)
        con_original = terminos_de_la_loss(modelo, centro, ctx, neg)
        con_invertidos = terminos_de_la_loss(modelo, centro, ctx, neg.flip(0))
        diferencias[str(k)] = abs(con_original["pos"] - con_invertidos["pos"])
        insensibilidad[str(k)] = {
            "pos_con_los_negativos_originales": con_original["pos"],
            "pos_con_los_negativos_invertidos": con_invertidos["pos"],
        }
    maximo = max(diferencias.values())
    exito = maximo <= TOL["fidelidad_de_la_replica"]
    if not exito:
        raise Parada("A2", "el termino positivo no es identico, o depende de los negativos",
                     {"diferencias_por_k": diferencias, "maximo": maximo,
                      "tolerancia": TOL["fidelidad_de_la_replica"]})
    return {
        "afirmacion": "A2",
        "hallazgo": "H1",
        "naturaleza": "CONFIRMAR",
        "que_se_afirma": "logsigmoid(v . u_pos).mean() es el termino positivo de la referencia, "
                         "no depende de K ni del termino negativo",
        "tolerancia": TOL["fidelidad_de_la_replica"],
        "unidad": "diferencia absoluta del termino positivo al recalcularlo con otros negativos",
        "medida": {
            "diferencias_por_k": diferencias,
            "maximo": maximo,
            "insensibilidad_al_negativo": insensibilidad,
        },
        "exito": "diferencia menor o igual que la tolerancia en los cuatro K",
        "fallo": "diferencia mayor, o el termino positivo depende de los negativos",
        "stop_si_falla": True,
        "veredicto": "CONFIRMADO",
    }


def afirmacion_a3(modulo_p71) -> dict:
    """A3. El termino negativo del codigo pesa EXACTAMENTE 1/K del de la referencia.

    El codigo promedia el termino negativo sobre el lote y sobre K, de modo que
    `neg_codigo = neg_referencia / K`, con la referencia definida con la SUMA sobre K. El 1/K
    es una division por un entero, no un estimador. En K = 1 el ratio vale 1 y la asercion es
    trivialmente cierta, luego K = 1 no cuenta como evidencia.

    Returns:
        Diccionario de la afirmacion con su veredicto.

    Raises:
        Parada: si el pre-registro marca STOP para A3 y la comprobacion falla.
    """
    ratios = {}
    desviaciones = {}
    for k in RED["valores_de_k"]:
        modelo = construir_red_de_prueba(modulo_p71)
        centro, ctx, neg = indices_de_prueba(k)
        terminos = terminos_de_la_loss(modelo, centro, ctx, neg)
        ratios[str(k)] = terminos["neg_codigo"] / terminos["neg_referencia"]
        desviaciones[str(k)] = abs(ratios[str(k)] - 1.0 / k)
    con_evidencia = max(desviaciones[str(k)] for k in RED["k_con_evidencia"])
    todos = max(desviaciones[str(k)] for k in RED["valores_de_k"])
    fuente = leer_fuente(FUENTE_P71)
    linea_215 = linea_de(fuente, 215).strip()
    sin_suma = ".sum(dim=1)" not in linea_215
    con_media = ".mean()" in linea_215
    exito = todos <= TOL["ratio_del_termino_negativo"] and sin_suma and con_media
    if not exito:
        raise Parada("A3", "el termino negativo no pesa exactamente 1/K, o la linea citada no "
                           "lo produce",
                     {"desviacion_de_1_sobre_k": desviaciones, "maximo": todos,
                      "linea_215": linea_215,
                      "contiene_sum_dim_1": not sin_suma,
                      "tolerancia": TOL["ratio_del_termino_negativo"]})
    return {
        "afirmacion": "A3",
        "hallazgo": "H1",
        "naturaleza": "CONFIRMAR",
        "que_se_afirma": "el codigo promedia el termino negativo sobre los ejes lote y K, de "
                         "modo que neg_codigo = neg_referencia / K de forma exacta",
        "tolerancia": TOL["ratio_del_termino_negativo"],
        "unidad": "|ratio_negativo - 1/K|, con ratio_negativo = neg_codigo / neg_referencia",
        "medida": {
            "ratio_negativo_por_k": ratios,
            "desviacion_de_1_sobre_k": desviaciones,
            "maximo_en_los_cuatro_k": todos,
            "maximo_en_k_con_evidencia": con_evidencia,
            "k_con_evidencia": list(RED["k_con_evidencia"]),
            "nota_de_k_1": RED["nota_k_1"],
            "linea_del_fuente": {
                "fichero": str(FUENTE_P71.relative_to(RAIZ)),
                "linea": 215,
                "texto_literal": a_ascii(linea_215, "p71_04:215"),
                "comprobacion": "la linea promedia y no suma sobre K antes de promediar",
                "contiene_sum_dim_1": not sin_suma,
                "contiene_mean": con_media,
            },
        },
        "exito": "se cumple en K = 5, K = 10 y K = 20, y la linea que lo produce esta citada",
        "fallo": "cualquier desviacion mayor que la tolerancia, o un ratio que no sea 1/K",
        "stop_si_falla": True,
        "veredicto": "CONFIRMADO",
    }


# ===== SECCION 5: H1, EL GRADIENTE (A4) Y LA AFIRMACION NEGATIVA (A5) =====


def afirmacion_a4(modulo_p71) -> dict:
    """A4. El gradiente del codigo es el de la referencia con la repulsion en 1/K.

    Se comprueban tres cosas: que el componente positivo pesa 1, que el componente negativo
    pesa 1/K, y que la suma de los dos reproduce a autograd con residuo por debajo de 1e-9.
    Toda la comparacion se hace sobre la matriz ACUMULADA, con `index_add_` y los mismos
    indices, porque autograd entrega el gradiente de la matriz entera con las repeticiones ya
    sumadas.

    Los dos componentes se calculan en funciones separadas y cada una devuelve SOLO su
    termino. No es cosmetico: si una sola funcion devolviera el total en funcion de `f`, el
    "componente negativo" llevaria tambien el termino positivo, la suma de los dos lo contaria
    dos veces, el residuo no seria nulo y la razon de los componentes negativos saldria igual a
    la razon de las normas TOTALES en lugar de 1/K.

    El componente positivo se compara contra el autograd del objetivo `-(pos)` aislado, no
    contra si mismo: comparar una magnitud consigo misma daria 1 por definicion y no seria una
    medida.

    Returns:
        Diccionario de la afirmacion con su veredicto.

    Raises:
        Parada: si el pre-registro marca STOP para A4 y algun residuo o razon se sale.
    """
    por_k = {}
    for k in RED["valores_de_k"]:
        modelo = construir_red_de_prueba(modulo_p71)
        centro, ctx, neg = indices_de_prueba(k)
        grafo_1 = calcular_perdidas(modelo, centro, ctx, neg)
        grad_entrada_codigo = gradiente_autograd(modelo, "entrada", grafo_1["objetivo_codigo"])
        grafo_2 = calcular_perdidas(modelo, centro, ctx, neg)
        grad_entrada_referencia = gradiente_autograd(modelo, "entrada",
                                                     grafo_2["objetivo_referencia"])
        grafo_3 = calcular_perdidas(modelo, centro, ctx, neg)
        grad_salida_codigo = gradiente_autograd(modelo, "salida", grafo_3["objetivo_codigo"])
        grafo_4 = calcular_perdidas(modelo, centro, ctx, neg)
        grad_salida_referencia = gradiente_autograd(modelo, "salida",
                                                    grafo_4["objetivo_referencia"])
        grafo_5 = calcular_perdidas(modelo, centro, ctx, neg)
        grad_positivo_entrada = gradiente_autograd(modelo, "entrada",
                                                   grafo_5["objetivo_solo_positivo"])
        grafo_6 = calcular_perdidas(modelo, centro, ctx, neg)
        grad_positivo_salida = gradiente_autograd(modelo, "salida",
                                                  grafo_6["objetivo_solo_positivo"])

        tensores = grafo_3["tensores"]
        componente_positivo = gradiente_del_termino_positivo(modelo, tensores)
        componente_negativo_codigo = gradiente_del_termino_negativo(modelo, tensores, 1.0 / k)
        componente_negativo_referencia = gradiente_del_termino_negativo(modelo, tensores, 1.0)
        total_codigo = sumar_componentes(componente_positivo, componente_negativo_codigo)
        total_referencia = sumar_componentes(componente_positivo,
                                            componente_negativo_referencia)

        normas = {
            "positivo_analitico": norma(componente_positivo["entrada"]),
            "positivo_autograd": norma(grad_positivo_entrada),
            "negativo_codigo": norma(componente_negativo_codigo["entrada"]),
            "negativo_referencia": norma(componente_negativo_referencia["entrada"]),
            "total_codigo": norma(grad_entrada_codigo),
            "total_referencia": norma(grad_entrada_referencia),
        }
        ratio_positivo = (normas["positivo_analitico"] / normas["positivo_autograd"]
                          if normas["positivo_autograd"] else float("nan"))
        ratio_negativo = (normas["negativo_codigo"] / normas["negativo_referencia"]
                          if normas["negativo_referencia"] else float("nan"))
        residuos = {
            "entrada_codigo": residuo_infinito(total_codigo["entrada"], grad_entrada_codigo),
            "entrada_referencia": residuo_infinito(total_referencia["entrada"],
                                                    grad_entrada_referencia),
            "salida_codigo": residuo_infinito(total_codigo["salida"], grad_salida_codigo),
            "salida_referencia": residuo_infinito(total_referencia["salida"],
                                                   grad_salida_referencia),
            "componente_positivo_contra_su_autograd": residuo_infinito(
                componente_positivo["entrada"], grad_positivo_entrada),
            "componente_positivo_contra_su_autograd_en_salida": residuo_infinito(
                componente_positivo["salida"], grad_positivo_salida),
            "la_diferencia_entre_los_dos_objetivos_es_solo_el_negativo": residuo_infinito(
                total_codigo["entrada"] - total_referencia["entrada"],
                componente_negativo_codigo["entrada"]
                - componente_negativo_referencia["entrada"]),
        }
        por_k[str(k)] = {
            "normas": normas,
            "ratio_positivo": ratio_positivo,
            "ratio_negativo": ratio_negativo,
            "objetivo_de_1_sobre_k": 1.0 / k,
            "residuos": residuos,
        }
    maximo_residuo = max(max(v["residuos"].values()) for v in por_k.values())
    tol_componentes = TOL["componentes_de_gradiente_relativo"]
    exito_positivo = all(
        abs(v["ratio_positivo"] - 1.0) <= tol_componentes for v in por_k.values())
    exito_negativo = all(
        abs(v["ratio_negativo"] - v["objetivo_de_1_sobre_k"]) <= tol_componentes
        for v in por_k.values())
    exito_residuos = maximo_residuo <= TOL["residuo_de_descomposicion"]
    exito = exito_positivo and exito_negativo and exito_residuos
    if not exito:
        raise Parada("A4", "la descomposicion del gradiente no se sostiene",
                     {"maximo_residuo": maximo_residuo,
                      "tolerancia_de_residuos": TOL["residuo_de_descomposicion"],
                      "ratio_positivo": {clave: v["ratio_positivo"] for clave, v
                                         in por_k.items()},
                      "ratio_negativo": {clave: v["ratio_negativo"] for clave, v
                                         in por_k.items()},
                      "objetivo_de_1_sobre_k": {clave: v["objetivo_de_1_sobre_k"] for clave, v
                                                in por_k.items()},
                      "tolerancia_de_componentes": tol_componentes,
                      "residuos_por_k": {clave: v["residuos"] for clave, v in por_k.items()}})
    return {
        "afirmacion": "A4",
        "hallazgo": "H1",
        "naturaleza": "CONFIRMAR",
        "que_se_afirma": "el componente positivo del gradiente pesa 1 y el negativo pesa 1/K, "
                         "y la suma de los dos reproduce a autograd",
        "tolerancia": {
            "componentes": f"{tol_componentes} relativa",
            "residuos": f"{TOL['residuo_de_descomposicion']} absoluta",
        },
        "unidad": "razones de normas y residuos de norma del maximo absoluto",
        "medida": {
            "por_k": por_k,
            "maximo_residuo": maximo_residuo,
            "matriz_de_comparacion": "matriz ACUMULADA, con index_add_ y los mismos indices",
            "como_se_define_ratio_positivo": (
                "norma del componente positivo analitico dividida por la norma del gradiente "
                "que autograd da para el objetivo `-(pos)` aislado. Medirlo contra si mismo "
                "daria 1 por definicion y no seria una medida"),
            "aditividad_de_los_componentes": (
                "positivo y negativo se calculan en dos funciones separadas y cada una devuelve "
                "solo su termino. Si una unica funcion devolviera el total en funcion de f, el "
                "componente negativo llevaria tambien el termino positivo, al sumarlos se "
                "contaria dos veces, el residuo dejaria de ser nulo y la razon de los "
                "componentes negativos no seria 1/K sino la razon de las normas totales"),
            "poder_discriminante": (
                "el peso 1 del componente positivo es identico por construccion, luego la "
                "prueba de verdad son los residuos: el del componente positivo contra su "
                "autograd, y el de la suma contra el objetivo completo"),
        },
        "exito": "ratio_positivo = 1 y ratio_negativo = 1/K, con los residuos por debajo de "
                 "la tolerancia en los cuatro K",
        "fallo": "cualquier ratio fuera de tolerancia, o cualquier residuo por encima del "
                 "limite, lo que significaria que la forma cerrada esta mal",
        "stop_si_falla": True,
        "veredicto": "CONFIRMADO",
    }


def contraejemplos_de_la_cota() -> dict:
    """Construye dos contraejemplos numericos explicitos a la cota que afirmo P79.

    Los dos vectores de aqui refutan los dos extremos con aritmetica exacta sobre hipotesis
    plausibles del termino negativo: uno anula el numerador y baja del limite inferior, el otro
    casi anula el denominador y lo supera.

    No son medidas del artefacto. Su unico papel es mostrar que la cota no se deduce de la
    forma de los dos objetivos.

    Returns:
        Diccionario con los dos contraejemplos y su aritmetica.
    """
    k = 10.0
    casos = {}
    g_pos = np.array([1.0, 0.0])
    g_neg = np.array([-k, 0.0])
    codigo = g_pos + g_neg / k
    referencia = g_pos + g_neg
    casos["por_debajo_del_limite_inferior"] = {
        "g_pos": g_pos.tolist(),
        "g_neg": g_neg.tolist(),
        "norma_total_codigo": float(np.linalg.norm(codigo)),
        "norma_total_referencia": float(np.linalg.norm(referencia)),
        "razon": float(np.linalg.norm(codigo) / np.linalg.norm(referencia)),
        "limite_inferior_que_P79_afirmaba": 1.0 / k,
        "lectura": "el termino negativo se opone al positivo con razon K, luego el numerador se "
                   "anula y la razon cae por debajo de 1/K",
    }
    g_pos2 = np.array([1.0, 0.0])
    g_neg2 = np.array([-(k - 0.001), 0.0])
    codigo2 = g_pos2 + g_neg2 / k
    referencia2 = g_pos2 + g_neg2
    casos["por_encima_del_limite_superior"] = {
        "g_pos": g_pos2.tolist(),
        "g_neg": g_neg2.tolist(),
        "norma_total_codigo": float(np.linalg.norm(codigo2)),
        "norma_total_referencia": float(np.linalg.norm(referencia2)),
        "razon": float(np.linalg.norm(codigo2) / np.linalg.norm(referencia2)),
        "limite_superior_que_P79_afirmaba": 1.0,
        "lectura": "el denominador casi se anula y el numerador no, luego la razon se dispara "
                   "muy por encima de 1",
    }
    return {
        "que_se_refuta": "la razon de la norma del gradiente TOTAL de entrada esta acotada",
        "porque_es_falso": "g_codigo = g_pos + g_neg/K y g_referencia = g_pos + g_neg no son "
                          "colineales, luego no se puede factorizar la razon en 1/K",
        "contraejemplos": casos,
        "naturaleza_de_los_numeros": "aritmetica exacta sobre dos hipotesis de vectores de "
                                     "gradiente; NO son medidas del artefacto congelado",
    }


def afirmacion_a5(por_k_a4: dict) -> dict:
    """A5. La razon de la NORMA total del gradiente NO se afirma que sea 1/K.

    Es una AFIRMACION NEGATIVA: P80 retira la cota de P79 y no la sustituye por ninguna otra.
    El ratio total se publica como numero informativo y se acompana de dos contraejemplos
    numericos que demuestran que la cota no se sostiene.

    Returns:
        Diccionario de la afirmacion con su veredicto.
    """
    ratio_total = {
        clave: por_k_a4[clave]["normas"]["total_codigo"]
        / por_k_a4[clave]["normas"]["total_referencia"] for clave in por_k_a4}
    bloque = {
        "ratio_de_la_norma_del_gradiente_total": {
            "etiqueta": "no_es_una_asercion",
            "valor_por_k": ratio_total,
            "por_que_no_es_una_asercion": "g_codigo y g_referencia no son colineales, luego la "
                                          "razon depende de la direccion del termino negativo "
                                          "y no se puede acotar",
        },
        "demostracion_de_la_cota": contraejemplos_de_la_cota(),
    }
    claves_presentes = claves_prohibidas_en_arbol(bloque)
    exito = not claves_presentes
    return {
        "afirmacion": "A5",
        "hallazgo": "H1",
        "naturaleza": "REFUTAR",
        "que_se_afirma": "P80 no asegura que ||g_entrada_codigo|| / ||g_entrada_referencia|| "
                         "valga 1/K, ni que este dentro de [1/K, 1]. La afirmacion de P79 se "
                         "retira por no demostrada y por incumplida por su propia medida",
        "tolerancia": "no aplica: es una correccion de una asercion indebida, no una medicion",
        "unidad": "razon de normas, publicada como numero informativo",
        "medida": {
            **bloque,
            "autocomprobacion": {
                "claves_de_cota_presentes": claves_presentes,
                "exigencia": "este bloque no puede contener ninguna clave de cota; si alguna "
                             "aparece, A5 falla",
            },
        },
        "exito": "el ratio aparece como numero informativo, sin ninguna cota, y con "
                 "contraejemplos numericos explicitos",
        "fallo": "que aparezca cualquier cota, o que el ratio se compare contra 1/K",
        "stop_si_falla": False,
        "veredicto": "CONFIRMADO" if exito else "REFUTADO",
    }


def claves_prohibidas_en_arbol(nodo, nivel: int = 0) -> list:
    """Busca claves prohibidas en un arbol de diccionarios y listas, recursivamente.

    Args:
        nodo: diccionario, lista o escalar.
        nivel: profundidad actual, usada para acotar la recursion.

    Returns:
        Lista de claves prohibidas encontradas.
    """
    if nivel > 12:
        return []
    encontradas: list = []
    if isinstance(nodo, dict):
        for clave, valor in nodo.items():
            if clave in CLAVES_DE_COTA_PROHIBIDAS:
                encontradas.append(clave)
            encontradas.extend(claves_prohibidas_en_arbol(valor, nivel + 1))
    elif isinstance(nodo, list):
        for valor in nodo:
            encontradas.extend(claves_prohibidas_en_arbol(valor, nivel + 1))
    return encontradas


def control_de_numeros_sospechosos(medidas: dict) -> dict:
    """Revisa las medidas por numeros sospechosamente perfectos.

    Solo se aplica a MEDIDAS. Los contraejemplos de A5 son hipotesis declaradas y no pasan
    por aqui, porque su funcion es ser absurdamente grandes. El criterio es el del criterio de
    parada 7: una perdida negativa o nula, o un ratio fuera del intervalo que la afirmacion
    obliga.

    Args:
        medidas: diccionario con las medidas ya calculadas.

    Returns:
        Diccionario con la revision.

    Raises:
        Parada: si aparece un numero que obliga a investigar replicacion antes de reportar.
    """
    hallazgos = []
    for clave, valor in medidas.items():
        if not isinstance(valor, (int, float)) or isinstance(valor, bool):
            continue
        if valor != valor or valor in (float("inf"), float("-inf")):
            hallazgos.append({"medida": clave, "valor": str(valor),
                              "problema": "la medida no es un numero finito"})
        if ("loss" in clave or "perdida" in clave) and valor <= 0:
            hallazgos.append({"medida": clave, "valor": valor,
                              "problema": "una perdida tiene que ser positiva"})
    if hallazgos:
        raise Parada("control", "ha aparecido un numero sospechosamente perfecto",
                     {"hallazgos": hallazgos})
    return {
        "criterio": "medidas finitas y perdidas positivas",
        "medidas_revisadas": len([v for v in medidas.values()
                                  if isinstance(v, (int, float)) and not isinstance(v, bool)]),
        "hallazgos": hallazgos,
        "exentos": "los contraejemplos de A5 son hipotesis declaradas, no medidas, y el ratio "
                   "total se publica como informativo y no se compara contra 1/K",
    }


# ===== SECCION 6: H2, EL SCHEDULER (A6, A7, A8) =====


def afirmacion_a6() -> dict:
    """A6. El scheduler se lee linea a linea, sin heuristica de busqueda de encabezado.

    Returns:
        Diccionario de la afirmacion con su veredicto.

    Raises:
        Parada: si el procedimiento no resuelve los dos rangos, porque el pre-registro marca
            STOP para A6.
    """
    modulo_base = sys.modules.get("p80_p71")
    lectura = auditar_semillas_por_ast(leer_fuente(FUENTE_P72)["texto"],
                                       str(FUENTE_P72.relative_to(RAIZ)), {}, modulo_base)
    contraste = auditar_semillas_por_ast(leer_fuente(FUENTE_P71_08)["texto"],
                                         str(FUENTE_P71_08.relative_to(RAIZ)), {}, modulo_base)
    llamadas = lectura["llamadas"]
    rangos_resueltos = [c for c in llamadas if c["rango_de_epocas"] is not None]
    roles = {c["rol"] for c in llamadas}
    exito = (len(llamadas) >= 2 and len(rangos_resueltos) >= 2
             and {"plan", "entrenamiento"}.issubset(roles))
    if not exito:
        raise Parada("A6", "la lectura por AST no resuelve los dos rangos del scheduler",
                     {"llamadas": len(llamadas), "rangos_resueltos": len(rangos_resueltos),
                      "roles_detectados": sorted(roles)})
    return {
        "afirmacion": "A6",
        "hallazgo": "H2",
        "naturaleza": "CONFIRMAR",
        "que_se_afirma": "el desfase se establece leyendo el fichero completo con el modulo ast "
                         "y un mapa de padres, sin busqueda de cadenas ni ventanas de lineas",
        "tolerancia": "igualdad exacta; cero margen",
        "unidad": "procedimiento, no numero",
        "medida": {
            "lectura_de_p72_01": lectura,
            "contraste_con_p71_08": contraste,
            "llamadas_encontradas": len(llamadas),
            "rangos_resueltos": len(rangos_resueltos),
            "roles_detectados": sorted(roles),
            "heuristicas_de_P79_que_no_se_reproducen": (
                "P79 buscaba el encabezado del bucle hacia atras en una ventana de 15 lineas "
                "y decidia si era el bucle del plan buscando la cadena planes.append en 4 "
                "lineas alrededor de la llamada. P80 no hace ninguna de las dos"),
        },
        "exito": "el procedimiento se completa sin ninguna regla de proximidad y los dos rangos "
                 "quedan resueltos",
        "fallo": "hace falta la heuristica de P79, o el iterador no es de las dos formas "
                 "admitidas",
        "stop_si_falla": True,
        "veredicto": "CONFIRMADO",
    }


def afirmacion_a7(lectura_a6: dict) -> dict:
    """A7. Los dos bucles usan rangos de semillas distintos.

    Los conjuntos tienen que ser DISTINTOS: que sean iguales es fallo, no exito.

    Returns:
        Diccionario de la afirmacion con su veredicto.
    """
    por_rol: dict = {}
    for llamada in lectura_a6["llamadas"]:
        if llamada["rango_de_epocas"] is not None:
            por_rol.setdefault(llamada["rol"], llamada)
    if "plan" not in por_rol or "entrenamiento" not in por_rol:
        return {
            "afirmacion": "A7",
            "hallazgo": "H2",
            "naturaleza": "CONFIRMAR",
            "que_se_afirma": "no se puede establecer el contraste sin los dos bucles",
            "tolerancia": "igualdad exacta de conjuntos",
            "unidad": "conjunto de enteros",
            "medida": {"por_rol": por_rol},
            "exito": "los conjuntos son distintos",
            "fallo": "son iguales",
            "stop_si_falla": False,
            "veredicto": "REFUTADO",
        }
    conjunto_plan = set(por_rol["plan"]["conjunto_de_semillas"])
    conjunto_entrenamiento = set(por_rol["entrenamiento"]["conjunto_de_semillas"])
    distintos = conjunto_plan != conjunto_entrenamiento
    return {
        "afirmacion": "A7",
        "hallazgo": "H2",
        "naturaleza": "CONFIRMAR",
        "que_se_afirma": "el bucle del plan resuelve ep en SEMILLA + {0..4} y el de "
                         "entrenamiento en SEMILLA + {1..5}; los conjuntos son distintos",
        "tolerancia": "igualdad exacta de conjuntos; que sean iguales es FALLO",
        "unidad": "conjunto de enteros",
        "medida": {
            "conjunto_plan": sorted(conjunto_plan),
            "conjunto_entrenamiento": sorted(conjunto_entrenamiento),
            "distintos": distintos,
            "solo_en_el_plan": sorted(conjunto_plan - conjunto_entrenamiento),
            "solo_en_el_entrenamiento": sorted(conjunto_entrenamiento - conjunto_plan),
            "semillas_por_epoca_plan": por_rol["plan"]["semillas_por_epoca"],
            "semillas_por_epoca_entrenamiento": por_rol["entrenamiento"]["semillas_por_epoca"],
            "que_implica": "el plan gasta una semilla que el entrenamiento nunca usa, luego "
                           "total_lotes se calibra sobre un numero de lotes que no es el que "
                           "se recorre",
            "por_que_P79_se_equivoco": "su heuristica de etiquetado metio los dos bucles en el "
                                       "mismo grupo, y por eso su unica lista tenia 6 elementos "
                                       "para EPOCAS = 5 y su booleano salia False no porque "
                                       "los rangos coincidieran, sino porque solo habia un "
                                       "grupo",
        },
        "exito": "los conjuntos son distintos y la diferencia se reporta con las listas "
                 "completas",
        "fallo": "son iguales, lo que refutaria H2 y explicaria el booleano de P79",
        "stop_si_falla": False,
        "veredicto": "CONFIRMADO" if distintos else "REFUTADO",
    }


def leer_recorte_de_corpus(ruta: Path, lineas: int, modulo_p71) -> list:
    """Lee solo las primeras lineas del corpus y las tokeniza como hace `p71_04.leer_tokens`.

    No se lee el fichero entero: se corta en el numero de lineas declarado. Se usa el
    `RE_TOKEN` del propio modulo de P71-04 y el mismo filtro de documentos con al menos tres
    tokens, de modo que el recorte sea el mismo que el de P79 y no una tokenizacion distinta.

    Args:
        ruta: fichero del corpus.
        lineas: numero de lineas a leer.
        modulo_p71: modulo `p71_04_entrenar_word2vec.py`.

    Returns:
        Lista de listas de tokens.
    """
    documentos = []
    with ruta.open(encoding="utf-8", errors="replace") as manejador:
        for indice, linea in enumerate(manejador):
            if indice >= lineas:
                break
            tokens = modulo_p71.RE_TOKEN.findall(linea)
            if len(tokens) >= 3:
                documentos.append(tokens)
    return documentos


def calendario_de_lr(lr_inicial: float, lr_minimo: float, total_lotes: int, n_pasos: int) -> dict:
    """Construye el calendario de learning rate que aplica el scheduler de P72-01.

    El scheduler hace `lr = max(LR_MINIMO, lr_ini * (1 - paso / max(total_lotes, 1)))`, una vez
    por lote.

    Args:
        lr_inicial: learning rate inicial.
        lr_minimo: suelo del learning rate.
        total_lotes: total con el que se calibra el decaimiento.
        n_pasos: numero de pasos a recorrer.

    Returns:
        Diccionario con la lista de valores y las metricas del calendario.
    """
    valores = [max(lr_minimo, lr_inicial * (1.0 - paso / max(total_lotes, 1)))
               for paso in range(n_pasos)]
    en_el_piso = sum(1 for valor in valores if valor <= lr_minimo)
    return {
        "valores": valores,
        "lr_inicial": lr_inicial,
        "lr_minimo": lr_minimo,
        "ultimo_real": valores[-1] if valores else None,
        "pasos_totales": n_pasos,
        "pasos_en_el_piso_del_minimo": en_el_piso,
        "fraccion_en_el_piso": (en_el_piso / n_pasos) if n_pasos else None,
        "media_real": float(np.mean(valores)) if valores else None,
    }


def comparar_con_p79(total_plan: int, total_real: int) -> dict:
    """Compara el par de conteos de P80 con el par que publico P79.

    Si el par coincide, la medida de P80 replica la de P79 en esos dos numeros, y eso se dice.
    Si no coincide, la diferencia se explica antes de seguir.

    Args:
        total_plan: total de lotes del bucle del plan.
        total_real: total de lotes del bucle de entrenamiento.

    Returns:
        Diccionario con el contraste y su explicacion.
    """
    registro = {
        "lectura_de_p79": "reports/p79_auditoria_sgns.json, punto_3_scheduler, solo para "
                          "contrastar; ninguna cifra de P79 se da por replicada",
        "p79_total_lotes_plan": None,
        "p79_total_lotes_recorridos": None,
        "p79_desvio_relativo": None,
    }
    if not INFORME_P79.exists():
        registro["explicacion"] = "no existe el informe de P79, luego no hay con que comparar"
        return registro
    informe = json.loads(INFORME_P79.read_text(encoding="utf-8"))
    scheduler = informe.get("punto_3_scheduler", {})
    registro["p79_total_lotes_plan"] = scheduler.get("total_lotes_plan")
    registro["p79_total_lotes_recorridos"] = scheduler.get("total_lotes_reales")
    registro["p79_desvio_relativo"] = scheduler.get("desvio_relativo")
    coincide = (registro["p79_total_lotes_plan"] == total_plan
                and registro["p79_total_lotes_recorridos"] == total_real)
    registro["coincide"] = bool(coincide)
    registro["explicacion"] = (
        "el par de conteos coincide con el de P79 sobre el mismo recorte, luego estos dos "
        "numeros quedan replicados. Lo que P80 corrige de P79 no es esta cuenta, sino el "
        "procedimiento que produjo la lista de semillas" if coincide else
        "el par de conteos NO coincide con el de P79. Antes de seguir: P80 lee las primeras "
        "lineas del fichero y tokeniza con el RE_TOKEN de p71_04, mientras que el recorte de "
        "P79 pudo tokenizar el fichero entero y truncar despues. La diferencia se explica por "
        "ahi y no por el scheduler")
    return registro


def afirmacion_a8(modulo_p71, modulo_p72) -> dict:
    """A8. Cuantificacion del efecto del desfase sobre el learning rate.

    Dos partes con alcances distintos. La parte 1 es la que sostiene el hallazgo y no depende
    del recorte: la diferencia entre las dos secuencias de lotes existe porque las semillas
    son distintas y `pares_de_epoca` depende de la semilla, lo cual se comprueba con aritmetica
    entera exacta sobre las semillas por epoca de los dos bucles. La parte 2 es la
    cuantificacion con el recorte declarado de 2000 lineas y lote 256, para poder comparar con
    P79.

    Returns:
        Diccionario de la afirmacion con su veredicto.
    """
    documentos = leer_recorte_de_corpus(CORPUS_TWITTER, RECRETE["lineas"], modulo_p71)
    n_tokens = sum(len(documento) for documento in documentos)
    vocab, p, _ = modulo_p71.construir_vocabulario(documentos)
    lote = int(modulo_p72.LOTE)
    epocas = int(modulo_p71.EPOCAS)
    semilla_base = int(modulo_p71.SEMILLA)

    lotes_por_epoca_plan = []
    for ep in range(epocas):
        generador = np.random.default_rng(semilla_base + ep)
        centros, _ = modulo_p71.pares_de_epoca(documentos, vocab, p, generador)
        lotes_por_epoca_plan.append(max(1, (len(centros) + lote - 1) // lote))
    total_plan = sum(lotes_por_epoca_plan)

    pares_por_epoca_reales = []
    lotes_por_epoca_reales = []
    for ep in range(1, epocas + 1):
        generador = np.random.default_rng(semilla_base + ep)
        centros, _ = modulo_p71.pares_de_epoca(documentos, vocab, p, generador)
        pares_por_epoca_reales.append(int(len(centros)))
        lotes_por_epoca_reales.append(max(1, (len(centros) + lote - 1) // lote))
    total_real = sum(lotes_por_epoca_reales)
    desvio = (total_real - total_plan) / total_plan if total_plan else None

    semillas_plan = [semilla_base + ep for ep in range(epocas)]
    semillas_entrenamiento = [semilla_base + ep for ep in range(1, epocas + 1)]
    difieren_en_todas = all(a != b for a, b in zip(semillas_plan, semillas_entrenamiento))
    primera_del_plan_ausente = semilla_base not in semillas_entrenamiento

    lr_inicial = float(modulo_p72.LR0_POR_EJEMPLO) * lote
    lr_minimo = float(modulo_p72.LR_MINIMO)
    reales = calendario_de_lr(lr_inicial, lr_minimo, total_real, total_real)
    ideales = calendario_de_lr(lr_inicial, lr_minimo, total_plan, total_real)
    comparacion = comparar_con_p79(total_plan, total_real)

    exito = (desvio is not None and abs(desvio) > TOL["desvio_del_scheduler"]
             and difieren_en_todas and primera_del_plan_ausente)
    return {
        "afirmacion": "A8",
        "hallazgo": "H2",
        "naturaleza": "CONFIRMAR",
        "que_se_afirma": "la diferencia entre las dos secuencias de lotes existe para "
                         "cualquier corpus, porque las semillas son distintas y pares_de_epoca "
                         "depende de la semilla; con el recorte declarado se cuantifica",
        "tolerancia": {"desvio_relativo": f"|desvio| > {TOL['desvio_del_scheduler']}",
                       "conteo_de_pares": "0, igualdad exacta con P79 si el recorte coincide"},
        "unidad": "lotes, y learning rate",
        "medida": {
            "parte_1_sin_recorte": {
                "que_es": "hechos aritmeticos enteros, independientes del corpus",
                "semillas_por_epoca_plan": semillas_plan,
                "semillas_por_epoca_entrenamiento": semillas_entrenamiento,
                "difieren_en_todas_las_epocas": difieren_en_todas,
                "la_primera_semilla_del_plan_no_aparece_en_el_entrenamiento":
                    primera_del_plan_ausente,
                "consecuencia": "en toda epoca el generador del entrenamiento se crea con otra "
                                "semilla, luego pares_de_epoca consume otro flujo y produce "
                                "otro conjunto de pares",
                "limite_de_la_afirmacion": "esto prueba el desfase de semillas, que es "
                                           "estructural. NO prueba que la desviacion numerica "
                                           "de lotes tenga el mismo valor para otro corpus: "
                                           "para eso esta la parte 2, con recorte declarado",
                "lectura_literal_del_pre_registro": (
                    "el pre-registro pide comprobar la parte 1 ejecutando los dos bucles con "
                    "el numero de filas del recorte; se hace asi, y despues se separa lo "
                    "estructural de lo numerico para no atribuirle al uno lo que solo sostiene "
                    "el otro"),
            },
            "parte_2_con_recorte_declarado": {
                "recorte": {
                    "lineas_leidas": RECRETE["lineas"],
                    "documentos": len(documentos),
                    "tokens": n_tokens,
                    "vocabulario": len(vocab),
                    "lote": lote,
                    "lote_tomado_de": "el LOTE de p72_01, que coincide con el recorte declarado",
                    "tokenizado_con": "el RE_TOKEN de p71_04, con el filtro de al menos tres "
                                      "tokens de p71_04.leer_tokens",
                },
                "limite_declarado": "los recuentos ABSOLUTOS de pares no son los de la corrida "
                                    "real, porque la frecuencia de los tipos cambia al "
                                    "recortar. Lo que se extrapola es la DESVIACION relativa, "
                                    "no los recuentos",
                "lotes_por_epoca_plan": lotes_por_epoca_plan,
                "lotes_por_epoca_recorridos": lotes_por_epoca_reales,
                "pares_por_epoca_recorridos": pares_por_epoca_reales,
                "total_lotes_plan": total_plan,
                "total_lotes_recorridos": total_real,
                "desvio_relativo": desvio,
                "learning_rate": {
                    "real": {clave: valor for clave, valor in reales.items()
                             if clave != "valores"},
                    "ideal_si_el_scheduler_hubiera_usado_el_total_real":
                        {clave: valor for clave, valor in ideales.items() if clave != "valores"},
                },
            },
            "comparacion_con_p79": comparacion,
        },
        "exito": "desvio distinto de cero, con la parte 1 declarada sin recorte",
        "fallo": "desvio exactamente cero, lo que significaria que el recorte borra el efecto",
        "stop_si_falla": False,
        "veredicto": "CONFIRMADO" if exito else "REFUTADO",
    }


# ===== SECCION 7: H3, LA IDENTIDAD DE LA LOSS (A9, A10) =====


def afirmacion_a9(modulo_p71) -> dict:
    """A9. La identidad de la loss se cumple con el SIGNO correcto.

    La loss del codigo es `-(pos + neg_codigo)`, luego la identidad correcta es
    `l_codigo + pos + neg_codigo = 0`. P79 comparo `pos + neg_codigo` contra `l_codigo`, es
    decir contra `-(pos + neg_codigo)`, y le faltaba el signo menos.

    Returns:
        Diccionario de la afirmacion con su veredicto.

    Raises:
        Parada: si la identidad no cuadra. El pre-registro prohibe entonces escribir el resto
            del informe: un objetivo cuya composicion no cuadra no se audita por partes.
    """
    residuos = {}
    for k in RED["valores_de_k"]:
        modelo = construir_red_de_prueba(modulo_p71)
        centro, ctx, neg = indices_de_prueba(k)
        terminos = terminos_de_la_loss(modelo, centro, ctx, neg)
        residuos[str(k)] = abs(terminos["loss_codigo"] + terminos["pos"]
                               + terminos["neg_codigo"])
    maximo = max(residuos.values())
    if maximo > TOL["identidad_de_la_loss"]:
        raise Parada("A9", "la identidad de la loss con el signo correcto no cuadra",
                     {"identidad": "l_codigo + pos + neg_codigo = 0",
                      "residuo_por_k": residuos,
                      "residuo_maximo": maximo,
                      "tolerancia": TOL["identidad_de_la_loss"]}, minimalista=True)
    return {
        "afirmacion": "A9",
        "hallazgo": "H3",
        "naturaleza": "CONFIRMAR",
        "que_se_afirma": "la loss del codigo es -(pos + neg_codigo), luego "
                         "l_codigo + pos + neg_codigo = 0 con el signo menos puesto",
        "tolerancia": TOL["identidad_de_la_loss"],
        "unidad": "|l_codigo + pos + neg_codigo| en float64",
        "medida": {
            "identidad": "l_codigo + pos + neg_codigo = 0",
            "residuo_por_k": residuos,
            "residuo_maximo": maximo,
            "nota_sobre_el_error_de_P79": "P79 comparo pos + neg_codigo contra l_codigo, que es "
                                          "-(pos + neg_codigo): al pasar de comparacion a "
                                          "identidad aparece el signo menos que faltaba, y "
                                          "por eso su indicador salia False en los cuatro K",
        },
        "exito": "se cumple en los cuatro K, con el signo menos puesto",
        "fallo": "no se cumple",
        "stop_si_falla": True,
        "veredicto": "CONFIRMADO",
    }


def afirmacion_a10(modulo_p71) -> dict:
    """A10. Control negativo: la identidad con el signo MAL puesto tiene que fallar.

    Sin esta comprobacion, A9 no distinguiria una formula de su negation. Se calcula la
    identidad que uso P79, `pos + neg_codigo == l_codigo`, y tiene que FALLAR con separacion
    suficiente. Si tambien se cumpliera, A9 quedaria sin poder discriminante.

    Returns:
        Diccionario de la afirmacion con su veredicto.

    Raises:
        Parada: si la identidad incorrecta tambien se cumple, con el JSON de parada minimo.
    """
    separaciones = {}
    losses = {}
    for k in RED["valores_de_k"]:
        modelo = construir_red_de_prueba(modulo_p71)
        centro, ctx, neg = indices_de_prueba(k)
        terminos = terminos_de_la_loss(modelo, centro, ctx, neg)
        separaciones[str(k)] = abs(terminos["pos"] + terminos["neg_codigo"]
                                   - terminos["loss_codigo"])
        losses[str(k)] = terminos["loss_codigo"]
    minimo = min(separaciones.values())
    if minimo < TOL["separacion_del_control_negativo"]:
        raise Parada("A10", "la identidad con el signo mal puesto tambien se cumple, luego A9 "
                            "no tiene poder discriminante",
                     {"identidad_incorrecta": "pos + neg_codigo == l_codigo",
                      "separacion_por_k": separaciones,
                      "separacion_minima": minimo,
                      "separacion_exigida": TOL["separacion_del_control_negativo"]},
                     minimalista=True)
    return {
        "afirmacion": "A10",
        "hallazgo": "H3",
        "naturaleza": "CONFIRMAR",
        "que_se_afirma": "la prueba de A9 tiene poder discriminante: la identidad con el signo "
                         "mal puesto, que es la que uso P79, falla en los cuatro K",
        "tolerancia": f"separacion minima {TOL['separacion_del_control_negativo']} en valor "
                      f"absoluto",
        "unidad": "|pos + neg_codigo - l_codigo|, la distancia de la identidad incorrecta",
        "medida": {
            "identidad_incorrecta": "pos + neg_codigo == l_codigo",
            "separacion_por_k": separaciones,
            "separacion_minima": minimo,
            "loss_codigo_por_k": losses,
            "relacion_teorica": "como l_codigo es el opuesto de pos + neg_codigo, la "
                                "separacion vale 2 por el valor absoluto de l_codigo. Si esa "
                                "relacion no se cumpliera, la medicion estaria mal",
            "por_que_es_una_prueba_de_poder": "si la identidad incorrecta fallara por poco, A9 "
                                               "no distinguiria una formula de su negation; por "
                                               "eso se exige una separacion minima y no solo "
                                               "que no sea cero",
        },
        "exito": "la identidad incorrecta falla en los cuatro K, con separacion suficiente",
        "fallo": "la identidad incorrecta tambien se cumple, y A9 quedaria sin poder "
                 "discriminante",
        "stop_si_falla": True,
        "veredicto": "CONFIRMADO",
    }


# ===== SECCION 8: H4, EL ARTEFACTO CONGELADO (A11, A12, A13) =====


def afirmacion_a11(huella_antes: dict) -> dict:
    """A11. El artefacto congelado solo guarda la tabla de entrada.

    Se abre en solo lectura, con un manejador de fichero, y se comprueban las claves, la forma,
    el tipo, la finitud, la unicidad del vocabulario y las filas nulas.

    Returns:
        Diccionario de la afirmacion con su veredicto.

    Raises:
        Parada: si hay una tercera clave o la forma no es la declarada.
    """
    with ARTEFACTO.open("rb") as manejador:
        paquete = np.load(manejador, allow_pickle=True)
        claves = sorted(paquete.files)
        matriz = paquete["W"]
        palabras = paquete["palabras"]
        en_doble = matriz.astype(np.float64, copy=False)
        integridad = {
            "vocabulario": int(matriz.shape[0]),
            "dim": int(matriz.shape[1]),
            "dtype": str(matriz.dtype),
            "todos_finitos": bool(np.isfinite(en_doble).all()),
            "palabras_unicas": int(len(set(palabras.tolist()))),
            "filas_nulas": int(int((np.abs(en_doble).sum(axis=1) == 0).sum())),
            "norma_media_de_fila": float(np.linalg.norm(en_doble, axis=1).mean()),
            "palabras_de_ejemplo": [a_ascii(str(p), "palabras") for p in palabras[:5]],
        }
    claves_ok = claves == sorted(ARTEFACTO_ESPERADO["claves"])
    forma_ok = (matriz.shape == (ARTEFACTO_ESPERADO["vocabulario"],
                                 ARTEFACTO_ESPERADO["dim"])
                and str(matriz.dtype) == ARTEFACTO_ESPERADO["dtype"])
    resto_ok = (integridad["todos_finitos"]
                and integridad["palabras_unicas"] == ARTEFACTO_ESPERADO["vocabulario"]
                and integridad["filas_nulas"] == 0)
    exito = claves_ok and forma_ok and resto_ok
    if not exito:
        raise Parada("A11", "el artefacto congelado no tiene la forma ni las claves declaradas",
                     {"claves": claves, "forma": list(matriz.shape), "dtype": str(matriz.dtype),
                      "todos_finitos": integridad["todos_finitos"],
                      "palabras_unicas": integridad["palabras_unicas"],
                      "filas_nulas": integridad["filas_nulas"]})
    return {
        "afirmacion": "A11",
        "hallazgo": "H4",
        "naturaleza": "CONFIRMAR",
        "que_se_afirma": "w2v_twitter_100d.npz contiene exactamente dos claves, W y palabras, "
                         "y una sola matriz; W es modelo.entrada.weight segun p72_01:124-127",
        "tolerancia": "igualdad exacta del conjunto de claves; cero margen",
        "unidad": "claves, forma, dtype y recuentos",
        "medida": {
            "claves": claves,
            "conjunto_de_claves_esperado": sorted(ARTEFACTO_ESPERADO["claves"]),
            "solo_una_matriz": claves_ok,
            "integridad": integridad,
            "nota_sobre_el_log_de_p72_01": (
                "el log de p72_01:129 anuncia parametros = vocabulario * dim * 2 porque el "
                "modelo tiene dos matrices; que las dos se guarden es otra cosa, y no se "
                "guardan"),
            "huella_antes_de_leer": huella_antes,
        },
        "exito": "exactamente esas dos claves, la forma declarada, todos los valores finitos, "
                 "vocabulario unico y 0 filas nulas",
        "fallo": "hay una tercera clave, o la forma no es la declarada",
        "stop_si_falla": True,
        "veredicto": "CONFIRMADO",
    }


def inspeccionar_npz_por_cabecera(fuente, nombre: str, limite: int = 4096) -> dict:
    """Inspecciona un `.npz` por su cabecera, sin leer el contenido de los arrays.

    Se abre el archivo ZIP, se listan sus miembros y de cada uno se leen los primeros bytes
    descomprimidos, que son justo la cabecera `.npy` con la forma y el tipo. El contenido de
    los arrays no se lee, y el registro lo declara.

    `fuente` es un manejador de fichero binario, no una ruta, por dos razones: la primera es
    que abrir con `open(ruta, "rb")` garantiza que el fichero nunca se abre en modo escritura,
    y la segunda es que la prueba de humo puede pasarle un `.npz` sintetico construido en
    memoria, sin escribir nada en disco.

    Args:
        fuente: manejador de fichero binario posicionado al principio.
        nombre: nombre del fichero, para el registro.
        limite: numero maximo de bytes descomprimidos por miembro.

    Returns:
        Diccionario con los miembros, sus formas y los bytes leidos.
    """
    registro = {
        "ruta": nombre,
        "miembros": [],
        "bytes_leidos": 0,
        "contenido_de_los_arrays_leido": False,
        "metodo": "cabecera .npy dentro del ZIP, sin descomprimir los arrays",
    }
    with zipfile.ZipFile(fuente) as paquete:
        for info in paquete.infolist():
            with paquete.open(info) as miembro:
                cabeza = miembro.read(limite)
            registro["bytes_leidos"] += len(cabeza)
            forma = None
            tipo = None
            try:
                with io.BytesIO(cabeza) as buffer:
                    version = np.lib.format.read_magic(buffer)
                    if version == (1, 0):
                        forma, _fortran, tipo = np.lib.format.read_array_header_1_0(buffer)
                    elif version == (2, 0):
                        forma, _fortran, tipo = np.lib.format.read_array_header_2_0(buffer)
            except Exception:  # noqa: BLE001
                forma, tipo = None, None
            registro["miembros"].append({
                "nombre": info.filename,
                "bytes_en_el_zip": info.file_size,
                "forma": list(forma) if forma else None,
                "dtype": str(tipo) if tipo else None,
                "cabecera_leida": len(cabeza),
            })
    return registro


def inspeccionar_npz_de_ruta(ruta: Path, limite: int = 4096) -> dict:
    """Abre un `.npz` en solo lectura y delega en `inspeccionar_npz_por_cabecera`.

    Args:
        ruta: fichero `.npz`.
        limite: numero maximo de bytes descomprimidos por miembro.

    Returns:
        Diccionario con los miembros, sus formas y los bytes leidos.
    """
    with ruta.open("rb") as manejador:
        return inspeccionar_npz_por_cabecera(manejador, str(ruta.relative_to(RAIZ)), limite)


def es_forma_de_salida(forma, n_vocab: int, dim: int) -> bool:
    """Dice si una forma puede ser la tabla de SALIDA del artefacto congelado.

    El criterio es dimensional COMPLETO, no por la primera dimension. Razon: en `SkipGram` las
    dos tablas son `nn.Embedding(n_vocab, dim)` con el MISMO `dim`, luego `salida.weight` tiene
    exactamente la forma `(n_vocab, dim)` de `entrada.weight`. Para el congelado de 100
    dimensiones eso es `(77657, 100)` y nada mas. Una matriz `(77657, 200)` o `(77657, 300)`
    pertenece a otra corrida y a otro numero de dimensiones, luego NO puede ser la tabla de
    salida de ESTE entrenamiento, por muy vocabulario que comparta.

    Por eso una coincidencia de la primera dimension no basta: es exactamente lo que hacia que
    P79 y la primera ejecucion de P80 vieran cuatro tablas de entrada de otras corridas como si
    fueran la tabla de salida.

    Args:
        forma: dimensiones a comprobar, o None si no se pudo leer.
        n_vocab: filas de la tabla de entrada del artefacto.
        dim: columnas de la tabla de entrada del artefacto.

    Returns:
        True si la forma puede ser la tabla de salida.
    """
    return bool(forma) and len(forma) == 2 and forma[0] == n_vocab and forma[1] == dim


def motivo_del_descarte(forma, n_vocab: int, dim: int) -> str:
    """Explica por que una forma NO es la tabla de salida del artefacto.

    Args:
        forma: dimensiones observadas, o None si no se pudo leer.
        n_vocab: filas de la tabla de entrada del artefacto.
        dim: columnas de la tabla de entrada del artefacto.

    Returns:
        Texto con el motivo, para que el negativo sea un dato y no una frase.
    """
    if not forma:
        return "no se pudo leer la forma del array"
    if len(forma) != 2:
        return f"la forma {forma} no es de rango 2, luego no es una tabla de embeddings"
    if forma[0] != n_vocab:
        return (f"la primera dimension es {forma[0]} y el vocabulario del artefacto es "
                f"{n_vocab}: es otra matriz")
    return (f"la primera dimension coincide ({n_vocab}) pero la segunda es {forma[1]} y la "
            f"tabla de salida de este entrenamiento tiene dim {dim}. Las dos tablas del modelo "
            f"comparten dim, luego esta forma pertenece a otra corrida de otras dimensiones")


def buscar_tablas_en_json(datos, n_vocab: int, dim: int, encontradas: list, descartadas: list,
                          prefijo: str, nivel: int = 0) -> None:
    """Busca en un JSON un valor que sea de verdad una matriz de la forma de la tabla de salida.

    Un JSON no puede contener la tabla de salida en la practica: haria falta un literal de
    7,7 millones de numeros. Aun asi se comprueba de forma estructural, y una clave `shape` o
    `forma` NUNCA cuenta: eso es metadato sobre la forma de otra cosa, no un array.

    Solo se mira una lista si su longitud es exactamente `n_vocab`, y de sus filas solo se
    necesita la primera para conocer la anchura, luego el coste es acotado.

    Args:
        datos: valor del JSON ya deserializado.
        n_vocab: filas de la tabla de entrada del artefacto.
        dim: columnas de la tabla de entrada del artefacto.
        encontradas: lista donde se acumulan las tablas reales.
        descartadas: lista donde se acumulan los descartes con su motivo.
        prefijo: camino textual hasta el valor actual.
        nivel: profundidad actual, para acotar la recursion.
    """
    if nivel > 14:
        return
    if isinstance(datos, dict):
        for clave, valor in datos.items():
            ruta = f"{prefijo}/{clave}"
            if clave in ("shape", "forma") and isinstance(valor, (list, tuple)):
                descartadas.append({
                    "ruta": ruta, "tipo": "clave_de_forma", "valor": list(valor),
                    "motivo": "una clave shape o forma es metadato sobre la forma de otra "
                              "cosa. No es un array, luego no puede ser una tabla de salida",
                })
                continue
            buscar_tablas_en_json(valor, n_vocab, dim, encontradas, descartadas, ruta,
                                  nivel + 1)
    elif isinstance(datos, list):
        if len(datos) == n_vocab:
            ancho = len(datos[0]) if datos and isinstance(datos[0], list) else None
            forma = [n_vocab, ancho] if ancho is not None else [n_vocab]
            if es_forma_de_salida(forma, n_vocab, dim):
                encontradas.append({"ruta": prefijo, "forma": forma, "n_filas": len(datos),
                                     "tipo": "lista_anidada_en_json"})
            else:
                descartadas.append({
                    "ruta": prefijo, "tipo": "lista_anidada_en_json", "forma": forma,
                    "motivo": motivo_del_descarte(forma, n_vocab, dim)})
        elif len(datos) >= 2 and all(isinstance(x, int) for x in datos[:2]):
            descartadas.append({
                "ruta": prefijo, "tipo": "lista_de_enteros", "valor": list(datos[:4]),
                "motivo": "es una lista de enteros, no una matriz de embeddings",
            })
        for indice, valor in enumerate(datos[:2]):
            buscar_tablas_en_json(valor, n_vocab, dim, encontradas, descartadas,
                                  f"{prefijo}[{indice}]", nivel + 1)


def clasificar_miembros_npz(miembros: list, ruta: str, n_vocab: int, dim: int,
                            es_el_artefacto: bool) -> dict:
    """Reparte los miembros de un `.npz` en candidatos y descartados, con el criterio de A12.

    Se separa del cuerpo de `afirmacion_a12` para que el humo pueda ejercitar EXACTAMENTE el
    mismo criterio sobre un npz sintetico en memoria. Un detector que solo se prueba a si mismo
    con el camino feliz no prueba nada.

    Args:
        miembros: lista de miembros con `nombre`, `forma` y `dtype`.
        ruta: nombre del fichero, para el registro.
        n_vocab: filas de la tabla de entrada del artefacto.
        dim: columnas de la tabla de entrada del artefacto.
        es_el_artefacto: True si el fichero es el artefacto congelado.

    Returns:
        Diccionario con `candidatos` y `descartados`.
    """
    candidatos = []
    descartados = []
    for miembro in miembros:
        forma = miembro.get("forma")
        if es_el_artefacto and miembro.get("nombre") == "W.npy":
            descartados.append({
                "ruta": ruta, "miembro": miembro["nombre"], "forma": forma,
                "motivo": "es la propia tabla de entrada W del artefacto congelado"})
            continue
        if not forma or forma[0] != n_vocab:
            continue
        if es_forma_de_salida(forma, n_vocab, dim):
            candidatos.append({"ruta": ruta, "miembro": miembro["nombre"], "forma": forma,
                               "dtype": miembro.get("dtype"),
                               "lectura": "miembro de un npz, array de rango 2"})
        else:
            descartados.append({
                "ruta": ruta, "miembro": miembro["nombre"], "forma": forma,
                "dtype": miembro.get("dtype"),
                "motivo": motivo_del_descarte(forma, n_vocab, dim)})
    return {"candidatos": candidatos, "descartados": descartados}


def afirmacion_a12() -> dict:
    """A12. La matriz de salida NO se puede reconstruir.

    La busqueda es exhaustiva y no heuristica: se recorren todos los `.npz` de `modelos/` y de
    `features/` por cabecera, y todas las claves de todos los `.json` de `reports/`. Se exige
    el recuento de ficheros inspeccionados y de candidatos descartados, para que el negativo
    sea un dato y no una frase.

    Returns:
        Diccionario de la afirmacion con su veredicto.

    Raises:
        Parada: si aparece una fuente con la tabla de salida, porque entonces el hallazgo H4
            cambia de contenido y hay que parar antes de publicar.
    """
    inspeccionados = []
    for base in (MODELOS, FEATURES):
        for ruta in sorted(base.rglob("*.npz")):
            inspeccionados.append(inspeccionar_npz_de_ruta(ruta))
    n_vocab = ARTEFACTO_ESPERADO["vocabulario"]
    dim = ARTEFACTO_ESPERADO["dim"]
    candidatos_npz = []
    descartados_npz = []
    for registro in inspeccionados:
        reparto = clasificar_miembros_npz(
            registro["miembros"], registro["ruta"], n_vocab, dim,
            registro["ruta"] == str(ARTEFACTO.relative_to(RAIZ)))
        candidatos_npz.extend(reparto["candidatos"])
        descartados_npz.extend(reparto["descartados"])
    n_json = 0
    candidatos_json = []
    descartados_json = []
    errores_json = []
    for ruta in sorted(REPORTS.rglob("*.json")):
        try:
            datos = json.loads(ruta.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            errores_json.append({"ruta": str(ruta.relative_to(RAIZ)),
                                 "error": f"{type(exc).__name__}: {exc}"})
            continue
        n_json += 1
        buscar_tablas_en_json(datos, n_vocab, dim, candidatos_json, descartados_json,
                              str(ruta.relative_to(RAIZ)))
    total_candidatos = len(candidatos_npz) + len(candidatos_json)
    resultado = {
        "afirmacion": "A12",
        "hallazgo": "H4",
        "naturaleza": "CONFIRMAR",
        "que_se_afirma": "no existe en el repositorio ninguna fuente que contenga la tabla de "
                         "salida del entrenamiento que produjo el congelado",
        "tolerancia": "no aplica: es una enumeracion con cobertura declarada",
        "unidad": "recuento de ficheros inspeccionados y de candidatos descartados",
        "medida": {
            "cobertura": {
                "npz_inspeccionados": len(inspeccionados),
                "npz_por_cabecera_sin_leer_el_contenido": True,
                "bytes_leidos_en_total": sum(r["bytes_leidos"] for r in inspeccionados),
                "json_inspeccionados": n_json,
                "json_con_error_de_lectura": errores_json,
                "nota_de_recorrido": "todo valor de reports/ se recorre. Un JSON no puede "
                                     "contener la tabla de salida en la practica, y aun asi "
                                     "se comprueba de forma estructural",
            },
            "criterio_de_deteccion": {
                "regla": "una fuente es candidata si contiene un array de rango 2 con forma "
                         f"exactamente ({n_vocab}, {dim})",
                "por_que_la_segunda_dimension": "en SkipGram las dos tablas son "
                                                "nn.Embedding(n_vocab, dim) con el MISMO dim, "
                                                "luego la tabla de salida tiene exactamente la "
                                                "forma de la de entrada. Una matriz con la "
                                                f"primera dimension {n_vocab} y otra segunda "
                                                "pertenece a otra corrida y no puede ser la "
                                                "tabla de salida de este entrenamiento",
                "por_que_un_json_no_sirve": "una clave shape o forma es metadato, no un array; "
                                            "y un array de esa forma en un JSON serian 7,7 "
                                            "millones de numeros. Aun asi se comprueba de "
                                            "forma estructural",
                "lo_que_se_descarta_y_se_registra": "todo lo que coincide en la primera "
                                                    "dimension pero no en la segunda, con su "
                                                    "motivo, para que el negativo sea un dato",
            },
            "candidatos_npz": candidatos_npz,
            "candidatos_json": candidatos_json,
            "candidatos_descartados_npz": descartados_npz,
            "candidatos_descartados_json": descartados_json,
            "n_descartados_npz": len(descartados_npz),
            "n_descartados_json": len(descartados_json),
            "n_candidatos": total_candidatos,
            "rutas_no_inspeccionadas_y_por_que": dict(RUTAS_PROHIBIDAS),
            "limites_de_la_enumeracion": [
                "A12 exige los .npz de modelos/ y features/ y los .json de reports/. Los "
                "checkpoints .pt de models/ y los ficheros de modelos/ que no son .npz no "
                "estan en esa lista y NO se inspeccionan: es un limite declarado de este "
                "negativo, no un olvido",
                "de features/oof_v03_final.npz se lee solo la cabecera del .npz, y eso se "
                "declara; su contenido no se lee",
            ],
        },
        "exito": "cero fuentes candidatas",
        "fallo": "aparece una fuente, y entonces A13 cambia de veredicto y H4 se reescribe",
        "stop_si_falla": False,
        "veredicto": "CONFIRMADO" if total_candidatos == 0 else "REFUTADO",
    }
    if total_candidatos:
        raise Parada("A12", "ha aparecido una fuente con la tabla de salida; H4 cambia de "
                            "contenido y hay que parar antes de publicar",
                     {"n_candidatos": total_candidatos,
                      "candidatos_npz": candidatos_npz[:5],
                      "candidatos_json": candidatos_json[:5]})
    return resultado


def verificar_cx_de_p72(texto_p72: str, modulo_p71, numero_de_linea_del_cx: int) -> dict:
    """Comprueba que `p72_01` usa la clase `SkipGram` de `p71_04` sin tocarla.

    Se lee el diccionario `cx` por el AST, se resuelve la entrada `"SkipGram"`, se comprueba que
    apunta al atributo del modulo cargado desde `p71_04`, y se comprueba en caliente que el
    atributo existe y es una clase con `forward`.

    Returns:
        Diccionario con la verificacion y las lineas citadas.
    """
    arbol = ast.parse(texto_p72, filename=FUENTE_P72.name)
    lineas = texto_p72.splitlines()
    entrada = None
    for nodo in ast.walk(arbol):
        if (isinstance(nodo, ast.Assign) and len(nodo.targets) == 1
                and isinstance(nodo.targets[0], ast.Name) and nodo.targets[0].id == "cx"
                and isinstance(nodo.value, ast.Dict)):
            for clave, valor in zip(nodo.value.keys, nodo.value.values):
                if isinstance(clave, ast.Constant) and clave.value == "SkipGram":
                    entrada = {"linea": nodo.lineno, "valor": ast.unparse(valor)}
    clase = getattr(modulo_p71, "SkipGram", None)
    citadas = [a_ascii(lineas[numero - 1].strip(), f"p72_01:{numero}")
               for numero in range(numero_de_linea_del_cx, numero_de_linea_del_cx + 3)]
    return {
        "que_se_afirma": "el artefacto lo produjo p72_01 usando la clase SkipGram de p71_04, sin "
                         "tocarla",
        "lineas_citadas": {"rango": f"p72_01:{numero_de_linea_del_cx}-"
                                    f"{numero_de_linea_del_cx + 2}",
                           "texto": citadas},
        "entrada_de_cx": entrada,
        "apunta_a_la_clase_de_p71_04": entrada_apunta_a_skipgram(entrada),
        "la_clase_existe_y_es_clase": isinstance(clase, type),
        "la_clase_tiene_forward": hasattr(clase, "forward"),
        "nota": "esta es la UNICA afirmacion admisible sobre el artefacto congelado, junto con "
                "la integridad descriptiva de W de A11",
    }


def entrada_apunta_a_skipgram(entrada: dict | None) -> bool:
    """Dice si la entrada de `cx` para SkipGram apunta a `base.SkipGram`.

    Args:
        entrada: diccionario con `linea` y `valor`, o None si no se encontro.

    Returns:
        True si el valor es el atributo SkipGram del modulo base.
    """
    return bool(entrada) and entrada["valor"] == "base.SkipGram"


def afirmacion_a13(cx_verificada: dict) -> dict:
    """A13. Las cifras absolutas de loss y de gradiente sobre el artefacto NO son evidencia.

    Es NO APLICABLE POR CONSTRUCCION: no es una medicion que P80 haga, es una afirmacion que
    P80 deja sin hacer. Como el artefacto no guarda la tabla de salida, cualquier evaluacion
    que necesite `u_pos` y `u_neg` tiene que inventar esa matriz. P79 puso `E = S = W`.

    Returns:
        Diccionario de la afirmacion con su veredicto.
    """
    return {
        "afirmacion": "A13",
        "hallazgo": "H4",
        "naturaleza": "NO APLICABLE POR CONSTRUCCION",
        "que_se_afirma": "las cifras absolutas de loss y de gradiente medidas sobre el artefacto "
                         "congelado NO son evidencia del estado final del entrenamiento, y no "
                         "pueden usarse para decir que el artefacto esta bien ni para decir que "
                         "esta mal",
        "tolerancia": "no aplica",
        "unidad": "no aplica: no se mide",
        "medida": {
            "evidencia_del_estado_final": False,
            "motivo": "el artefacto solo guarda W, que es la tabla de entrada. Para evaluar la "
                      "loss haria falta tambien la tabla de salida, y esa no existe en ninguna "
                      "parte del repositorio, segun A12",
            "tabla_de_salida_guardada": False,
            "tabla_de_salida_sustituida_en_silencio": False,
            "cifras_incluidas": False,
            "por_que_no_se_incluyen": "cualquier cifra exigiria el prefijo literal "
                                      "condicional_a_S_igual_a_W y aun asi invitaria a leerse "
                                      "como medida. P80 no pone ninguna, y por eso no hay nada "
                                      "que condicionalizar",
            "cifras_de_p79_que_no_se_replican": {
                "loss_codigo": 1.08412074,
                "loss_referencia": 9.78555861,
                "norma_del_gradiente_de_entrada_codigo": 0.02042704,
                "norma_del_gradiente_de_entrada_referencia": 0.25426385,
                "motivo": "describen el par (E, S) con E = S = W, que nunca existio durante el "
                          "entrenamiento",
                "prefijo_que_habria_que_poner": "condicional_a_S_igual_a_W",
            },
            "frase_que_sustituye_a_la_de_p79": FRASE_SUSTITUIDA_P79,
            "afirmaciones_retiradas_de_p79": [
                "P79 publico dos claves de cota sobre la razon de la norma del gradiente "
                "total, y afirmo que el ratio entre objetivos no depende del subconjunto. P80 "
                "retira las dos y no las sustituye por ninguna otra",
            ],
            "lo_unico_admisible_sobre_el_artefacto": cx_verificada,
        },
        "exito": "el bloque del artefacto declara evidencia_del_estado_final = false con su "
                 "motivo, y no incluye cifras condicionales",
        "fallo": "que P80 reporte esas cifras como medida del artefacto entrenado, o que "
                 "repita la frase sustituida",
        "stop_si_falla": False,
        "veredicto": "NO APLICABLE POR CONSTRUCCION",
    }


# ===== SECCION 9: H5 Y H6, LOS BIGRU (A14, A15, A16) =====


def analisis_de_origen(asignaciones: dict, parametros: list) -> dict:
    """Resuelve, para cada nombre de una funcion, de que depende hasta los parametros.

    Es un analisis de taintedness pequeno y declarado: se parte de una expresion y se siguen
    las asignaciones de la funcion hasta llegar a los parametros. No pretende ser un
    analizador de flujo de datos completo; pretende responder una pregunta concreta, que es si
    una linea depende de los datos de validacion del pliegue externo.

    Args:
        asignaciones: mapa de nombre a expresion de su asignacion.
        parametros: lista de parametros de la funcion.

    Returns:
        Diccionario `{nombre: lista_de_parametros}` con la dependencia de cada nombre.
    """
    cache: dict = {}

    def dependencia(nombre: str, en_curso: frozenset) -> set:
        """Parametros de los que depende un nombre.

        Args:
            nombre: identificador a resolver.
            en_curso: nombres ya visitados, para no entrar en ciclos.

        Returns:
            Conjunto de parametros de los que depende el nombre.
        """
        if nombre in cache:
            return cache[nombre]
        if nombre in parametros:
            return {nombre}
        if nombre in en_curso or nombre not in asignaciones:
            return set()
        try:
            nodo = ast.parse(asignaciones[nombre], mode="eval").body
        except SyntaxError:
            cache[nombre] = set()
            return cache[nombre]
        resultado: set = set()
        for hijo in ast.walk(nodo):
            if isinstance(hijo, ast.Name):
                resultado |= dependencia(hijo.id, en_curso | {nombre})
        cache[nombre] = resultado
        return resultado

    return {nombre: sorted(dependencia(nombre, frozenset()))
            for nombre in sorted(set(asignaciones) | set(parametros))}


def origen_nombre_or_empty(origen: dict, nombres_validacion: list,
                           parametros_de_entrenamiento: list) -> set:
    """Devuelve los parametros de entrenamiento de los que dependen los nombres dados.

    Args:
        origen: mapa de dependencia por nombre.
        nombres_validacion: nombres cuya dependencia se consulta.
        parametros_de_entrenamiento: parametros que no son de validacion.

    Returns:
        Conjunto de parametros de entrenamiento alcanzados, vacio si ninguno.
    """
    alcanzados: set = set()
    for nombre in nombres_validacion:
        alcanzados |= set(origen.get(nombre, [])) & set(parametros_de_entrenamiento)
    return alcanzados


def nombre_mas_a_la_derecha(patron) -> str | None:
    """Devuelve el nombre mas a la derecha de un patron de desestructuracion.

    El segundo elemento de `for k, (tr, va) in ...` es un Name, pero el de
    `for k, (tr, (va, w)) in ...` es una tupla. Se baja hasta la hoja de la derecha.

    Args:
        patron: nodo `ast.Tuple` o `ast.Name` del objetivo de un bucle.

    Returns:
        El identificador de la hoja derecha, o None si no hay ninguno.
    """
    if isinstance(patron, ast.Name):
        return patron.id
    if isinstance(patron, ast.Tuple) and patron.elts:
        return nombre_mas_a_la_derecha(patron.elts[-1])
    return None


def bucle_de_pliegues_de_la_llamada(llamada: ast.Call, padres: dict) -> ast.For | None:
    """Busca el bucle de validacion cruzada que envuelve a una llamada.

    No basta con el bucle mas proximo: en `v10_11` la llamada esta dentro de un bucle de
    variantes que no desestructura nada, y el bucle de pliegues es el de fuera. Se sube por la
    cadena de `For` hasta encontrar el primero cuyo objetivo se desestructura en dos.

    Args:
        llamada: nodo `ast.Call`.
        padres: mapa de padres.

    Returns:
        El `ast.For` de pliegues, o None si no hay ninguno.
    """
    for ancestro in ancestros_de_tipo(llamada, ast.For, padres):
        if isinstance(ancestro.target, ast.Tuple) and len(ancestro.target.elts) == 2:
            return ancestro
    return None


def resolver_origen_de_pliegues(alcance, nombre_de_pliegues: str | None) -> dict | None:
    """De donde sale el objeto `pliegues` dentro del ambito del llamante.

    Hay dos formas legitimas y las dos existen en los ficheros auditados: que `pliegues` sea una
    asignacion del propio llamante (`pliegues = pliegues_corregidos(dev, ...)`) o que sea un
    parametro suyo, cuando el llamante lo recibe ya construido. Se aceptan las dos y se declara
    cual se ha encontrado.

    Args:
        alcance: funcion o modulo que contiene la llamada.
        nombre_de_pliegues: nombre del objeto, o None si no se pudo resolver.

    Returns:
        Diccionario con el tipo de origen y su linea, o None si no se encuentra.
    """
    if nombre_de_pliegues is None:
        return None
    if isinstance(alcance, ast.FunctionDef):
        for argumento in alcance.args.args:
            if argumento.arg == nombre_de_pliegues:
                return {"tipo": "parametro_del_llamante", "linea": alcance.lineno,
                        "expresion": f"parametro {nombre_de_pliegues} de {alcance.name}"}
    for nodo in ast.walk(alcance):
        if isinstance(nodo, ast.Assign) and len(nodo.targets) == 1 \
                and isinstance(nodo.targets[0], ast.Name) \
                and nodo.targets[0].id == nombre_de_pliegues:
            return {"tipo": "asignacion_en_el_llamante", "linea": nodo.lineno,
                    "expresion": ast.unparse(nodo.value)}
    return None


def nombre_mas_a_la_izquierda(patron) -> str | None:
    """Devuelve el nombre mas a la izquierda de un patron de desestructuracion.

    En `for k, (tr, va) in ...` el elemento de entrenamiento NO es `k`, que es el contador, sino
    `tr`, que es la hoja izquierda del ultimo elemento. Sin esta funcion `ids[tr]` no casaba con
    nada y salia como desconocido.

    Args:
        patron: nodo `ast.Tuple` o `ast.Name`.

    Returns:
        El identificador de la hoja izquierda, o None si no hay ninguno.
    """
    if isinstance(patron, ast.Name):
        return patron.id
    if isinstance(patron, ast.Tuple) and patron.elts:
        return nombre_mas_a_la_izquierda(patron.elts[0])
    return None


def cadena_del_call_site(texto: str, nombre_funcion: str, nombres_validacion: list,
                        nombre: str) -> dict:  # noqa: ARG001
    """Resuelve el origen de los argumentos de validacion en el punto de llamada.

    A14 pregunta si la epoca se elige con el mismo pliegue que despues se puntua. Eso no se
    contesta dentro de la funcion de entrenamiento, sino mirando quien la llama: los argumentos
    de validacion tienen que acabar llegando al segundo elemento del pliegue que produce el
    bucle de validacion cruzada. La busqueda sigue las asignaciones del llamante hasta tres
    saltos, porque hay dos formas legitimas: un indice directo (`ids[va]`) y un tensor ya
    construido que viene de un `preparar(textos, entrenamiento, validacion)`.

    `nombres_validacion` se recibe y se declara, pero NO se usa para decidir que argumento es de
    validacion: esa decision es estructural, por donde llega el dato. Se publica igualmente para
    que el informe diga que parametros de la funcion son de validacion.

    Returns:
        Diccionario con la cadena resuelta, sus lineas y si es coherente.
    """
    arbol = ast.parse(texto, filename=nombre)
    padres = mapa_de_padres(arbol)
    lineas = texto.splitlines()
    llamada = None
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Call) and isinstance(nodo.func, ast.Name) \
                and nodo.func.id == nombre_funcion:
            llamada = nodo
    if llamada is None:
        return {"encontrada": False,
                "motivo": f"no se encuentra ninguna llamada a {nombre_funcion}"}

    bucle = bucle_de_pliegues_de_la_llamada(llamada, padres)
    segundo_del_bucle = None
    primer_del_bucle = None
    nombre_de_pliegues = None
    if bucle is not None:
        ultimo_del_objetivo = bucle.target.elts[-1]
        primer_del_bucle = nombre_mas_a_la_izquierda(ultimo_del_objetivo)
        segundo_del_bucle = nombre_mas_a_la_derecha(ultimo_del_objetivo)
        if isinstance(bucle.iter, ast.Call) and bucle.iter.args \
                and isinstance(bucle.iter.args[0], ast.Name):
            nombre_de_pliegues = bucle.iter.args[0].id

    alcance = ancestro_de_tipo(llamada, ast.FunctionDef, padres) or ast.Module
    asignaciones_del_llamante = {}
    for nodo in ast.walk(alcance):
        if isinstance(nodo, ast.Assign) and len(nodo.targets) == 1 \
                and isinstance(nodo.targets[0], ast.Name):
            asignaciones_del_llamante.setdefault(nodo.targets[0].id, ast.unparse(nodo.value))

    def nombres_de_expresion(expresion: str) -> set:
        """Nombres que aparecen en una expresion.

        Args:
            expresion: expresion en forma de texto.

        Returns:
            Conjunto de identificadores que aparecen en la expresion.
        """
        try:
            nodo = ast.parse(expresion, mode="eval").body
        except SyntaxError:
            return set()
        return {hijo.id for hijo in ast.walk(nodo) if isinstance(hijo, ast.Name)}

    def atributo_exterior(expresion: str) -> str | None:
        """Ultimo atributo de la cadena de una expresion, que identifica QUE array es.

        Args:
            expresion: expresion en forma de texto.

        Returns:
            El nombre del atributo exterior, o None si la expresion no termina en uno.
        """
        try:
            nodo = ast.parse(expresion, mode="eval").body
        except SyntaxError:
            return None
        while isinstance(nodo, ast.Call):
            nodo = nodo.func
        if isinstance(nodo, ast.Attribute):
            return nodo.attr
        return None

    def clase_por_nombre(terminal: str | None) -> str:
        """Clasifica un array por el nombre que lo identifica.

        Args:
            terminal: nombre del atributo exterior del array.

        Returns:
            `validacion`, `entrenamiento` o `desconocido`.
        """
        if not terminal:
            return "desconocido"
        bajo = terminal.lower()
        if "val" in bajo:
            return "validacion"
        if "train" in bajo:
            return "entrenamiento"
        return "desconocido"

    def evidencia_estructural(nodo: ast.expr) -> dict | None:
        """Clasifica una expresion por su propia estructura, sin mirar de donde viene.

        Se buscan dos cosas, en este orden:

        1. un subscript cuyo indice es el SEGUNDO elemento del pliegue, o el PRIMERO. Es la
           evidencia mas fuerte: son los indices los que dicen de que filas sale el array, y es
           lo que resuelve `ids[va]` y `etiquetas[validacion]`;
        2. un atributo cuyo nombre dice `val` o dice `train`. Es lo que resuelve `datos.x_val`,
           donde el indice no aparece porque el array ya venia construido.

        Si una expresion dice las dos cosas, o no dice ninguna, se devuelve None en vez de
        apostar: un detector que elige cuando no sabe es peor que uno que no opina.

        Args:
            nodo: expresion a clasificar.

        Returns:
            Diccionario con `clase` e `identificador`, o None si no hay evidencia.
        """
        subscripts = [n for n in ast.walk(nodo)
                      if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Name)]
        for sub in subscripts:
            if segundo_del_bucle is not None and sub.slice.id == segundo_del_bucle:
                return {"clase": "validacion", "identificador": sub.slice.id,
                        "evidencia": "subscript por el segundo elemento del pliegue"}
            if primer_del_bucle is not None and sub.slice.id == primer_del_bucle:
                return {"clase": "entrenamiento", "identificador": sub.slice.id,
                        "evidencia": "subscript por el primer elemento del pliegue"}
        atributos = sorted({n.attr for n in ast.walk(nodo) if isinstance(n, ast.Attribute)})
        dicen_val = [a for a in atributos if "val" in a.lower()]
        dicen_train = [a for a in atributos if "train" in a.lower()]
        if dicen_val and not dicen_train:
            return {"clase": "validacion", "identificador": dicen_val[0],
                    "evidencia": "atributo cuyo nombre dice val"}
        if dicen_train and not dicen_val:
            return {"clase": "entrenamiento", "identificador": dicen_train[0],
                    "evidencia": "atributo cuyo nombre dice train"}
        if dicen_val and dicen_train:
            return {"clase": "desconocido",
                    "identificador": ",".join(dicen_val + dicen_train),
                    "evidencia": "la expresion dice val y train a la vez, no se apuesta"}
        return None

    def vinculo_local(nombre: str) -> dict | None:
        """Clasifica un nombre desnudo siguiendo su unica asignacion en el llamante.

        Es el caso de `11_dl`: la llamada recibe `x_val` e `y_val` como nombres, y su
        procedencia esta unas lineas mas arriba, en `x_val = datos.x_val.to(dispositivo)` y en
        `y_val = torch.tensor(etiquetas[validacion], ...)`. Sin seguir ese vinculo los siete
        argumentos salen como desconocidos y la llamada parece no traer validacion.

        Args:
            nombre: identificador del argumento.

        Returns:
            Diccionario con la clasificacion y el vinculo, o None si no hay asignacion.
        """
        expresion = asignaciones_del_llamante.get(nombre)
        if expresion is None:
            return None
        try:
            nodo = ast.parse(expresion, mode="eval").body
        except SyntaxError:
            return None
        resultado = evidencia_estructural(nodo)
        if resultado is None:
            return None
        return {**resultado,
                "identificador": f"{nombre} -> {resultado['identificador']}",
                "evidencia": "vinculo local: " + resultado["evidencia"]}

    def clasificar_argumento(argumento: ast.expr) -> dict:
        """Clasifica UN argumento de la llamada: estructura, luego vinculo, luego nombre.

        Los tres pasos hacen falta y cada uno cubre un caso distinto. Con solo estructura
        funcionan `ids[va]` de v9_04 e `ids[validacion]` de v10_11, pero los nombres desnudos de
        `11_dl` salen todos como desconocidos. Con solo vinculo por grafo se arrastraban `modelo`
        y `x_train` a la clase de validacion, porque se construyen desde el mismo `datos` que
        produce `preparar(textos, entrenamiento, validacion)`. El nombre entra solo al final, y
        con la convencion declarada, no con un `in` crudo: `tr_val` NO es validacion.

        Args:
            argumento: nodo de la llamada.

        Returns:
            Diccionario con `clase`, `identificador` y `evidencia`.
        """
        directo = evidencia_estructural(argumento)
        if directo is not None:
            return {**directo, "identificador": ast.unparse(argumento)}
        if isinstance(argumento, ast.Name):
            local = vinculo_local(argumento.id)
            if local is not None:
                return local
            if es_nombre_de_validacion(argumento.id):
                return {"clase": "validacion", "identificador": argumento.id,
                        "evidencia": "desempate por el nombre, convencion de validacion"}
            if es_nombre_de_entrenamiento(argumento.id):
                return {"clase": "entrenamiento", "identificador": argumento.id,
                        "evidencia": "desempate por el nombre, convencion de entrenamiento"}
        return {"clase": "desconocido", "identificador": ast.unparse(argumento),
                "evidencia": "sin subscript, sin atributo que diga val o train, y sin nombre "
                             "que case con las convenciones"}

    argumentos_de_validacion = []
    clases = {}
    for indice_argumento, argumento in enumerate(llamada.args):
        texto_argumento = ast.unparse(argumento)
        veredicto = clasificar_argumento(argumento)
        registro = {"posicion": indice_argumento, "linea": argumento.lineno,
                    "expresion": texto_argumento,
                    "identificador_del_array": veredicto["identificador"],
                    "clase": veredicto["clase"],
                    "evidencia": veredicto["evidencia"]}
        clases[texto_argumento] = veredicto["clase"]
        if veredicto["clase"] == "validacion":
            argumentos_de_validacion.append(registro)

    origen_de_pliegues = resolver_origen_de_pliegues(alcance, nombre_de_pliegues)
    coherente = bool(alcance) and bool(argumentos_de_validacion) and origen_de_pliegues is not None
    return {
        "encontrada": True,
        "linea_de_la_llamada": llamada.lineno,
        "argumentos_de_validacion": argumentos_de_validacion,
        "n_argumentos_de_validacion": len(argumentos_de_validacion),
        "parametros_de_validacion_de_la_funcion": list(nombres_validacion),
        "clase_de_cada_argumento": clases,
        "segundo_elemento_del_bucle_de_pliegues": segundo_del_bucle,
        "linea_del_bucle_de_pliegues": bucle.lineno if bucle is not None else None,
        "texto_del_bucle_de_pliegues": (a_ascii(lineas[bucle.lineno - 1].strip(),
                                                f"{nombre}:{bucle.lineno}")
                                        if bucle is not None else None),
        "origen_de_los_pliegues": origen_de_pliegues,
        "coherente": bool(coherente),
        "primer_elemento_del_bucle_de_pliegues": primer_del_bucle,
        "criterio_estructural": {
            "tres_pasos": [
                "1. la estructura del propio argumento: un subscript por el segundo o el primer "
                "elemento del pliegue, o un atributo cuyo nombre dice val o train",
                "2. el vinculo local, si el argumento es un nombre desnudo: se sigue su unica "
                "asignacion en el llamante y se clasifica esa expresion con el paso 1",
                "3. el nombre, solo como desempate y solo con la convencion declarada, no con "
                "un `in` crudo: `tr_val` no es validacion",
            ],
            "por_que_no_se_sigue_el_grafo": "en 11_dl el modelo y `x_train` se construyen desde "
                                            "el mismo `datos` que produce `preparar(textos, "
                                            "entrenamiento, validacion)`. Seguir el grafo "
                                            "hasta el pliegue los marcaba a los dos como de "
                                            "validacion, y decia que cuatro argumentos lo eran "
                                            "cuando solo dos lo son",
            "por_que_A14_y_A19_se_comportan_distinto": (
                "A14 no sufre la limitacion de resolver solo por estructura porque v9_04 y "
                "v10_11 pasan los arrays como subscript del pliegue, `ids[va]` e "
                "`ids[validacion]`, y ahi la identidad del indice basta. A19 recibe en 11_dl "
                "nombres desnudos, `x_val` e `y_val`, con la procedencia unas lineas mas arriba, "
                "y por eso hace falta el paso 2. Es una diferencia de como esta escrito el "
                "codigo, no de protocolo: los dos entrenan y puntuan con el mismo pliegue"),
            "convenciones": {"validacion": PATRON_VALIDACION.pattern,
                             "entrenamiento": PATRON_ENTRENAMIENTO.pattern},
        },
        "busqueda_del_bucle": (
            "se sube por la cadena de For hasta el primero que se desestructura en dos, y no "
            "solo el mas proximo: en v10_11 la llamada esta dentro de un bucle de variantes"),
        "regla": "tiene que haber al menos un argumento de validacion, y el bucle tiene que "
                 "venir de un constructor de pliegues, ya sea una asignacion del llamante o un "
                 "parametro suyo",
    }


def analizar_entrenamiento_con_selector(texto: str, nombre: str, mejor: str) -> dict:
    """Analiza una funcion de entrenamiento que restaura pesos por mejor metrica de validacion.

    Se localiza la funcion por el hecho de llamar a `load_state_dict` sobre el nombre del mejor
    estado, no buscandola por texto. Dentro se localizan las cuatro piezas del pre-registro por
    linea exacta, y se calcula de que parametros dependen.

    Devuelve, ademas de las piezas, un bloque `por_que` que enumera CADA criterio con su valor
    observado y su veredicto. Sin ese bloque, un REFUTADO no se puede distinguir de un fallo
    del detector, y P73 y P74 si detectaron esta fuga: si aqui no se ve, lo sospechoso es el
    instrumento.

    Returns:
        Diccionario con las cuatro piezas, sus lineas, el analisis de origen y el bloque `por_que`.
    """
    arbol = ast.parse(texto, filename=nombre)
    lineas = texto.splitlines()
    funcion = None
    for nodo in ast.walk(arbol):
        if not isinstance(nodo, ast.FunctionDef):
            continue
        for hijo in ast.walk(nodo):
            if (isinstance(hijo, ast.Call) and isinstance(hijo.func, ast.Attribute)
                    and hijo.func.attr == "load_state_dict" and hijo.args
                    and isinstance(hijo.args[0], ast.Name) and hijo.args[0].id == mejor):
                funcion = nodo
    if funcion is None:
        return {"encontrada": False,
                "motivo": f"ninguna funcion llama a load_state_dict({mejor})"}
    parametros = [argumento.arg for argumento in funcion.args.args]
    asignaciones = {}
    for nodo in ast.walk(funcion):
        if isinstance(nodo, ast.Assign) and len(nodo.targets) == 1 \
                and isinstance(nodo.targets[0], ast.Name):
            asignaciones.setdefault(nodo.targets[0].id, ast.unparse(nodo.value))
    origen = analisis_de_origen(asignaciones, parametros)
    parametros_de_validacion = [p for p in parametros if es_nombre_de_validacion(p)]
    parametros_de_entrenamiento = [p for p in parametros
                                   if p not in parametros_de_validacion]

    def validacion_de_expresion(expresion: str) -> list:
        """Parametros de validacion de los que depende una expresion.

        Args:
            expresion: expresion en forma de texto.

        Returns:
            Lista ordenada de parametros de validacion de los que depende.
        """
        try:
            nodo = ast.parse(expresion, mode="eval").body
        except SyntaxError:
            return []
        nombres = {hijo.id for hijo in ast.walk(nodo) if isinstance(hijo, ast.Name)}
        resultado = set()
        for identificador in nombres:
            for dependencia in origen.get(identificador, []):
                if es_nombre_de_validacion(dependencia):
                    resultado.add(dependencia)
        return sorted(resultado)

    piezas: dict = {}
    for nodo in ast.walk(funcion):
        if (isinstance(nodo, ast.Assign) and len(nodo.targets) == 1
                and isinstance(nodo.targets[0], ast.Name) and nodo.targets[0].id == mejor
                and "inicializa_el_estado" not in piezas):
            piezas["inicializa_el_estado"] = {"nodo": nodo,
                                              "metricas": [ast.unparse(nodo.value)]}
        if isinstance(nodo, ast.If):
            sobrescribe = any(
                isinstance(sentencia, ast.Assign) and len(sentencia.targets) == 1
                and isinstance(sentencia.targets[0], ast.Name)
                and sentencia.targets[0].id == mejor
                for sentencia in nodo.body)
            if not sobrescribe:
                continue
            texto_test = ast.unparse(nodo.test)
            metricas = sorted({identificador for identificador in origen
                               if identificador in texto_test
                               and validacion_de_expresion(identificador)})
            if metricas:
                piezas["compara_y_sobrescribe"] = {"nodo": nodo, "metricas": metricas}
        if isinstance(nodo, ast.Call) and isinstance(nodo.func, ast.Attribute) \
                and nodo.func.attr == "load_state_dict":
            piezas["restaura_los_pesos"] = {"nodo": nodo,
                                            "metricas": [ast.unparse(nodo)]}

    candidatos_probabilidad = []
    for nodo in ast.walk(funcion):
        texto_objetivo = None
        if isinstance(nodo, ast.Return) and nodo.value is not None:
            texto_objetivo = ast.unparse(nodo.value)
        elif isinstance(nodo, ast.Assign) and len(nodo.targets) == 1:
            objetivo = nodo.targets[0]
            nombre_destino = objetivo.attr if isinstance(objetivo, ast.Attribute) \
                else objetivo.id if isinstance(objetivo, ast.Name) else ""
            if "proba" in nombre_destino.lower():
                texto_objetivo = ast.unparse(nodo.value)
        if texto_objetivo and validacion_de_expresion(texto_objetivo):
            candidatos_probabilidad.append(nodo)
    if candidatos_probabilidad:
        elegido = max(candidatos_probabilidad, key=lambda n: n.lineno)
        piezas["devuelve_probabilidades"] = {
            "nodo": elegido, "metricas": [ast.unparse(elegido.value)]}

    etiquetas = {
        "inicializa_el_estado": "inicializa el estado",
        "compara_y_sobrescribe": "compara la metrica y sobrescribe el estado",
        "restaura_los_pesos": "restaura los pesos",
        "devuelve_probabilidades": "devuelve las probabilidades de validacion",
    }
    faltan = []
    detalle_piezas = {}
    for clave, etiqueta in etiquetas.items():
        entrada = piezas.get(clave)
        if entrada is None:
            faltan.append(etiqueta)
            continue
        real = entrada["nodo"]
        expresion = " ".join(entrada["metricas"])
        detalle_piezas[clave] = {
            "linea": real.lineno,
            "texto_literal": a_ascii(lineas[real.lineno - 1].strip(),
                                     f"{nombre}:{real.lineno}"),
            "parametros_de_validacion_de_esta_pieza": validacion_de_expresion(expresion),
        }
    con_datos = {clave: valor["parametros_de_validacion_de_esta_pieza"]
                 for clave, valor in detalle_piezas.items()
                 if valor["parametros_de_validacion_de_esta_pieza"]}
    piezas_completas = not faltan
    dos_piezas_en_validacion = (
        len(con_datos) >= 2
        and all(valores and set(valores) <= set(parametros_de_validacion)
                for valores in con_datos.values())
        and all(not origen_nombre_or_empty(origen, valores, parametros_de_entrenamiento)
                for valores in con_datos.values()))

    nombres_de_validacion_usados = {nodo.id for nodo in ast.walk(funcion)
                                    if isinstance(nodo, ast.Name)
                                    and es_nombre_de_validacion(nodo.id)}
    no_autorizados = sorted(
        identificador for identificador in nombres_de_validacion_usados
        if identificador not in parametros
        and not any(es_nombre_de_validacion(d) for d in origen.get(identificador, [])))
    reparto = [registro for registro in
               (es_llamada_de_reparto(nodo, lineas) for nodo in ast.walk(funcion))
               if registro is not None]
    sin_particion_interna = not no_autorizados and not reparto
    call_site = cadena_del_call_site(texto, funcion.name, parametros_de_validacion, nombre)
    coherente_call_site = bool(call_site.get("coherente"))

    criterios = [
        {"criterio": "cuatro_piezas_localizadas",
         "observado": piezas_completas,
         "detalle": {"piezas_encontradas": sorted(detalle_piezas), "piezas_faltantes": faltan}},
        {"criterio": "las_piezas_que_portedatos_usan_la_validacion",
         "observado": dos_piezas_en_validacion,
         "detalle": {"piezas_que_portedatos": con_datos,
                     "parametros_de_validacion": parametros_de_validacion}},
        {"criterio": "sin_particion_interna",
         "observado": sin_particion_interna,
         "detalle": {"nombres_de_validacion_no_autorizados": no_autorizados,
                     "llamadas_de_reparto": reparto,
                     "nota": "una llamada cuyo nombre empieza por mayuscula es un constructor "
                             "de clase, no un repartidor, y no cuenta"}},
        {"criterio": "el_call_site_trae_la_validacion_del_pliegue_externo",
         "observado": coherente_call_site,
         "detalle": {"argumentos_de_validacion": call_site.get("argumentos_de_validacion"),
                     "clase_de_cada_argumento": call_site.get("clase_de_cada_argumento"),
                     "segundo_elemento_del_bucle":
                         call_site.get("segundo_elemento_del_bucle_de_pliegues"),
                     "origen_de_los_pliegues": call_site.get("origen_de_los_pliegues"),
                     "bucle_de_pliegues_encontrado":
                         call_site.get("linea_del_bucle_de_pliegues") is not None}},
    ]

    return {
        "encontrada": True,
        "funcion": funcion.name,
        "linea": funcion.lineno,
        "parametros": parametros,
        "parametros_de_validacion": parametros_de_validacion,
        "parametros_de_entrenamiento": parametros_de_entrenamiento,
        "dependencias_calculadas": origen,
        "piezas": detalle_piezas,
        "piezas_completas": piezas_completas,
        "piezas_faltantes": faltan,
        "piezas_que_portedatos": con_datos,
        "las_dos_piezas_que_portedatos_usan_validacion": dos_piezas_en_validacion,
        "nombres_de_validacion_en_el_cuerpo": sorted(nombres_de_validacion_usados),
        "nombres_de_validacion_no_autorizados": no_autorizados,
        "llamadas_de_reparto": reparto,
        "sin_particion_interna": sin_particion_interna,
        "call_site": call_site,
        "por_que": {
            "para_que_sirve": "sin este bloque, un REFUTADO no se puede distinguir de un fallo "
                              "del detector. P73 y P74 si detectaron esta fuga, luego si aqui "
                              "no se ve, lo sospechoso es el instrumento y no el codigo",
            "criterios": criterios,
            "todos_los_criterios_pasan": all(c["observado"] for c in criterios),
        },
        "clasificacion_de_parametros_publicada": {
            "convencion": PATRON_VALIDACION.pattern,
            "nombres_exactos": sorted(NOMBRES_VALIDACION_EXACTOS),
            "motivo_de_la_convencion": "el identificador tiene que tener un componente "
                                       "exactamente va o val, para que `variante` NO se "
                                       "clasifique como validacion",
            "parametros_clasificados_como_validacion": parametros_de_validacion,
            "parametros_clasificados_como_resto": parametros_de_entrenamiento,
        },
    }


def apariciones_de_una_frase(fuente: dict, frase: str) -> dict:
    """Localiza las apariciones de una cadena literal y cita sus lineas.

    Esta busqueda por cadena la exige el pre-registro: hay que registrar el texto literal con
    el que esos informes se describen a si mismos. No decide ningun veredicto.

    Args:
        fuente: diccionario devuelto por `leer_fuente`.
        frase: cadena a localizar.

    Returns:
        Diccionario con la frase, sus lineas y sus textos.
    """
    numeros = [numero for numero, linea in enumerate(fuente["lineas"], start=1)
               if frase in linea]
    return {
        "frase": frase,
        "n_apariciones": len(numeros),
        "lineas": numeros,
        "textos": [a_ascii(fuente["lineas"][numero - 1].strip(),
                           f"{fuente['ruta']}:{numero}") for numero in numeros],
        "metodo": "busqueda de cadena literal, exigida por el pre-registro; no decide ningun "
                  "veredicto",
    }


def diagnostico_por_hallazgo(afirmacion: str, ficheros: dict) -> dict:
    """Resume, para cada fichero, por que pasa o por que falla la afirmacion.

    Args:
        afirmacion: identificador de la afirmacion, "A14" o "A19".
        ficheros: diccionario `{clave: analisis}`.

    Returns:
        Diccionario con un veredicto por fichero y la lista de criterios que fallan.
    """
    salida = {}
    for clave, analisis in ficheros.items():
        criterios = analisis.get("por_que", {}).get("criterios", [])
        fallan = [c["criterio"] for c in criterios if not c["observado"]]
        salida[clave] = {
            "funcion_detectada": analisis.get("funcion"),
            "piezas_completas": analisis.get("piezas_completas"),
            "piezas_faltantes": analisis.get("piezas_faltantes"),
            "call_site_coherente": bool(analisis.get("call_site", {}).get("coherente")),
            "criterios_que_pasan": [c["criterio"] for c in criterios if c["observado"]],
            "criterios_que_fallan": fallan,
            "veredicto_del_fichero": "CONFIRMADO" if not fallan else "REFUTADO",
            "lectura": ("los cuatro criterios pasan" if not fallan else
                        "fallan estos criterios, y el detalle esta en por_que.criterios"),
        }
    return {"afirmacion": afirmacion, "por_fichero": salida}


def afirmacion_a14() -> dict:
    """A14. Los BiGRU eligen epoca con el mismo pliegue que despues puntuan.

    Se comprueba, sobre los dos ficheros y por linea exacta, que la metrica de validacion que
    decide cuando restaurar pesos es la del pliegue externo, y que las probabilidades que se
    reportan salen de ese mismo pliegue con esos mismos pesos restaurados. Ademas se comprueba
    que la funcion de entrenamiento no construye ninguna particion interna.

    Returns:
        Diccionario de la afirmacion con su veredicto y el bloque `por_que` que permite
        juzgar si un REFUTADO es real o un fallo del detector.
    """
    ficheros = {}
    for clave, ruta in (("v9_04", FUENTE_V9), ("v10_11", FUENTE_V10)):
        fuente = leer_fuente(ruta)
        analisis = analizar_entrenamiento_con_selector(fuente["texto"],
                                                       fuente["ruta"], "mejor_estado")
        analisis["apariciones_de_la_frase"] = apariciones_de_una_frase(fuente,
                                                                      "early stopping estricto")
        analisis["apariciones_de_la_frase"]["nota"] = (
            "la frase describe mal el protocolo: no hay particion interna, luego no es early "
            "stopping anidado. No se edita el fichero: la correccion va en este informe")
        analisis["fuente"] = {"ruta": fuente["ruta"], "sha256": fuente["sha256"],
                              "n_lineas": fuente["n_lineas"]}
        ficheros[clave] = analisis
    todos_ok = all(
        f.get("piezas_completas")
        and f.get("las_dos_piezas_que_portedatos_usan_validacion")
        and f.get("sin_particion_interna")
        and f.get("call_site", {}).get("coherente")
        for f in ficheros.values())
    numeros = {"V9-04": 0.694343, "V10-11": 0.711639}
    return {
        "afirmacion": "A14",
        "hallazgo": "H5",
        "naturaleza": "CONFIRMAR",
        "que_se_afirma": "en v9_04 y en v10_11 la metrica de validacion que decide cuando "
                         "restaurar pesos es la del pliegue externo, y las probabilidades "
                         "reportadas salen de ese mismo pliegue; no hay particion interna",
        "tolerancia": "no aplica: comprobacion estructural con lineas exactas",
        "unidad": "lineas y dependencia de datos",
        "medida": {
            "ficheros": ficheros,
            "por_que": diagnostico_por_hallazgo("A14", ficheros),
            "metodo": "la funcion se localiza por AST, por el hecho de llamar a "
                      "load_state_dict(mejor_estado). Las cuatro piezas se localizan por nodo "
                      "AST y se calculan sus dependencias hasta los parametros. La coherencia "
                      "del pliegue se resuelve en el punto de llamada, que es donde se ve de "
                      "donde salen los argumentos de validacion",
            "criterio_de_exito": [
                "las cuatro piezas existen, con su linea y su texto literal",
                "las dos piezas que portan datos dependen solo de parametros de validacion",
                "no hay ninguna particion interna en la funcion de entrenamiento",
                "los argumentos de validacion del call site son el segundo elemento del pliegue "
                "del bucle de validacion cruzada",
            ],
            "convenciones_usadas": {"validacion": [PATRON_VALIDACION.pattern,
                                                   sorted(NOMBRES_VALIDACION_EXACTOS)],
                                    "reparto": PATRON_REPARTO.pattern,
                                    "reparto_ignora_clases": PREFIJOS_DE_CLASE.pattern},
        },
        "exito": "las cuatro piezas existen, las que portan datos usan la validacion del "
                 "pliegue externo y no hay particion interna",
        "fallo": "hay particion interna, con lo cual H5 queda refutado y los numeros de V9-04 y "
                 "V10-11 serian comparables",
        "stop_si_falla": False,
        "veredicto": "CONFIRMADO" if todos_ok else "REFUTADO",
        "consecuencia_obligatoria": {
            "numeros_no_comparables": numeros,
            "motivo": "elegir la epoca con el pliegue que despues se puntua es una fuga, la "
                      "misma que condeno P73 y P74",
            "sesgo": "optimista, de magnitud NO MEDIDA",
            "lo_que_se_puede_afirmar": "sus numeros son mas bajos que los de V03, asi que la "
                                      "fuga no los convierte en comparables: no se pueden usar "
                                      "como evidencia en ninguna direccion",
        },
    }


def auditar_mascara_tardia_en_texto(texto: str, nombre: str) -> dict:
    """Comprueba el orden de las operaciones y el relleno sobre un texto fuente dado.

    Args:
        texto: fuente completa.
        nombre: nombre del fichero, para el AST.

    Returns:
        Diccionario con el conteo, el orden y el `padding_idx`.
    """
    arbol = ast.parse(texto, filename=nombre)
    lineas = texto.splitlines()
    linea_gru = None
    linea_multiplica = None
    padding_idx = None
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Call) and isinstance(nodo.func, ast.Attribute):
            partes = ruta_de_atributo(nodo.func)
            if partes and partes[-1] == "gru" and len(partes) >= 2 and partes[-2] == "self":
                linea_gru = nodo.lineno
            if partes and partes[-1] == "from_pretrained":
                for palabra in nodo.keywords:
                    if palabra.arg == "padding_idx":
                        padding_idx = palabra.value.value if isinstance(palabra.value,
                                                                      ast.Constant) \
                            else ast.unparse(palabra.value)
        if isinstance(nodo, ast.BinOp) and isinstance(nodo.op, ast.Mult):
            izquierda = ast.unparse(nodo.left)
            derecha = ast.unparse(nodo.right)
            if ("salida" in izquierda or "out" in izquierda) and "mascara" in derecha:
                linea_multiplica = nodo.lineno
    orden = bool(linea_gru and linea_multiplica and linea_gru < linea_multiplica)
    return {
        "apariciones_de_pack_padded_sequence": texto.count("pack_padded_sequence"),
        "linea_de_la_gru": linea_gru,
        "texto_de_la_gru": a_ascii(lineas[linea_gru - 1].strip(), nombre) if linea_gru else None,
        "linea_de_la_multiplicacion_por_la_mascara": linea_multiplica,
        "texto_de_la_multiplicacion": (a_ascii(lineas[linea_multiplica - 1].strip(), nombre)
                                       if linea_multiplica else None),
        "orden_gru_antes_que_mascara": orden,
        "padding_idx": padding_idx,
        "tres_comprobaciones": {
            "cero_pack_padded_sequence": texto.count("pack_padded_sequence") == 0,
            "gru_antes_que_la_mascara": orden,
            "padding_idx_es_cero": padding_idx == 0,
        },
    }


def afirmacion_a15() -> dict:
    """A15. El padding se enmascara DESPUES de la GRU.

    El numero de apariciones de `pack_padded_sequence` tiene que ser exactamente cero, el orden
    de las operaciones tiene que ser GRU y despues multiplicacion por la mascara, y el
    `padding_idx` del embedding tiene que ser 0.

    Returns:
        Diccionario de la afirmacion con su veredicto.

    Raises:
        Parada: si aparece `pack_padded_sequence`, porque entonces la lectura de estos dos
            ficheros cambia de interpretacion.
    """
    ficheros = {}
    for clave, ruta in (("v9_04", FUENTE_V9), ("v10_11", FUENTE_V10)):
        fuente = leer_fuente(ruta)
        analisis = auditar_mascara_tardia_en_texto(fuente["texto"], fuente["ruta"])
        analisis["fuente"] = {"ruta": fuente["ruta"], "sha256": fuente["sha256"],
                              "n_lineas": fuente["n_lineas"]}
        ficheros[clave] = analisis
    apariciones = sum(f["apariciones_de_pack_padded_sequence"] for f in ficheros.values())
    if apariciones:
        raise Parada("A15", "aparece pack_padded_sequence, luego la lectura de estos dos "
                            "ficheros cambia de interpretacion", ficheros)
    exito = all(all(f["tres_comprobaciones"].values()) for f in ficheros.values())
    return {
        "afirmacion": "A15",
        "hallazgo": "H6",
        "naturaleza": "CONFIRMAR",
        "que_se_afirma": "la mascara se construye a partir de los identificadores y se "
                         "multiplica por la SALIDA de la GRU, luego la recurrencia consume los "
                         "pasos de relleno",
        "tolerancia": "no aplica: comprobacion estructural",
        "unidad": "lineas y conteo de apariciones",
        "medida": {
            "ficheros": ficheros,
            "naturaleza_del_efecto": "con padding_idx = 0 el vector de relleno es el vector "
                                     "cero, luego el efecto no es de explosion sino de "
                                     "arrastre: los pasos de relleno se acumulan en los "
                                     "estados y, en la direccion inversa, se mezclan en las "
                                     "posiciones reales anteriores al relleno",
            "metodo": "el conteo de pack_padded_sequence es un recuento de cadena declarado por "
                      "el pre-registro; el orden de las operaciones y el padding_idx se "
                      "localizan por AST, no por ventana de lineas",
        },
        "exito": "las tres comprobaciones en los dos ficheros",
        "fallo": "aparece pack_padded_sequence, y H6 queda refutado",
        "stop_si_falla": True,
        "veredicto": "CONFIRMADO" if exito else "REFUTADO",
    }


def distancia_relativa(referencia: np.ndarray, otro: np.ndarray) -> float:
    """Distancia L2 relativa entre dos vectores.

    Args:
        referencia: vector de referencia.
        otro: vector a comparar.

    Returns:
        La distancia relativa, o infinito si el vector de referencia es nulo.
    """
    norma_ref = float(np.linalg.norm(referencia))
    if norma_ref == 0.0:
        return float("inf")
    return float(np.linalg.norm(referencia - otro) / norma_ref)


def lote_de_ids(secuencias: list, longitud: int) -> torch.Tensor:
    """Construye una tanda de identificadores rellenada con ceros hasta `longitud`.

    Args:
        secuencias: lista de listas de identificadores, cada fila con su longitud real.
        longitud: longitud maxima de la tanda.

    Returns:
        Tensor `int64` de forma `(tanda, longitud)`.
    """
    lote = torch.zeros((len(secuencias), longitud), dtype=torch.long)
    for fila, tokens in enumerate(secuencias):
        lote[fila, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)
    return lote


def sondear_relleno(clase, matriz: np.ndarray, longitudes: list, longitud_maxima: int,
                    semilla: int) -> dict:
    """Sonda controlada del efecto del relleno, sin entrenar y sin datos del proyecto.

    Se instancia la clase del propio fichero, se captura el vector agregado con un gancho que
    intercepta la entrada del dropout, y se evalua la misma fila de la tanda con dos longitudes
    maximas distintas: la longitud real y el maximo de la tanda.

    Args:
        clase: clase `BiGRU` o `BiGRUAfinable`.
        matriz: matriz de embeddings sintetica, con la fila 0 reservada al relleno.
        longitudes: longitudes reales de las filas de la tanda.
        longitud_maxima: longitud a la que se rellena en la segunda pasada.
        semilla: semilla de torch.

    Returns:
        Diccionario con las dos pasadas, las distancias y la fila observada.
    """
    torch.manual_seed(semilla)
    try:
        modelo = clase(matriz)
    except TypeError:
        modelo = clase(matriz, False)
    modelo.eval()
    capturado: list = []

    def gancho(_modulo, argumentos):
        """Captura la entrada del dropout, que es el vector agregado.

        Args:
            _modulo: el modulo al que se engancha.
            argumentos: tupla de entrada del dropout.

        Returns:
            None, para no alterar el calculo.
        """
        capturado.append(argumentos[0].detach().clone())

    manejador = modelo.dropout.register_forward_pre_hook(gancho)
    ids = [list(range(1, longitud + 1)) for longitud in longitudes]
    fila_observada = int(np.argmax([len(tokens) for tokens in ids]))
    try:
        with torch.no_grad():
            capturado.clear()
            modelo(lote_de_ids(ids, max(longitudes)))
            vector_real = capturado[-1][fila_observada].numpy()
            capturado.clear()
            modelo(lote_de_ids(ids, longitud_maxima))
            vector_maximo = capturado[-1][fila_observada].numpy()
    finally:
        manejador.remove()
    return {
        "longitudes_reales": longitudes,
        "longitud_maxima_real": max(longitudes),
        "longitud_maxima_de_la_tanda": longitud_maxima,
        "fila_observada": fila_observada,
        "pasos_de_relleno_anadidos": longitud_maxima - max(longitudes),
        "distancia_relativa_l2": distancia_relativa(vector_real, vector_maximo),
        "norma_del_vector_real": float(np.linalg.norm(vector_real)),
        "vector_real_primeros_6": [float(v) for v in vector_real[:6]],
        "vector_maximo_primeros_6": [float(v) for v in vector_maximo[:6]],
    }


def resolver_clase_bi_gru(nombre_modulo: str, modulo) -> object:
    """Localiza la clase de BiGRU en un modulo, y dice cual ha encontrado.

    `v9_04` define `BiGRU`; `v10_11` define `BiGRUAfinable` y importa `v9_04` como submodule,
    luego `getattr(modulo, "BiGRU")` falla y hay que ir a `BiGRUAfinable`. Si tampoco aparece
    ninguna, es un fallo del detector y hay que decirlo, no asumir.

    Args:
        nombre_modulo: clave del modulo, para el registro.
        modulo: modulo ya importado, o None.

    Returns:
        La clase encontrada, o None si el modulo no esta o no expone ninguna.

    Raises:
        Parada: si el modulo esta pero no expone ninguna de las dos clases.
    """
    if modulo is None:
        return None
    for nombre in ("BiGRU", "BiGRUAfinable"):
        clase = getattr(modulo, nombre, None)
        if clase is not None:
            return clase
    raise Parada("A16", f"el modulo {nombre_modulo} no expone BiGRU ni BiGRUAfinable",
                 {"modulo": nombre_modulo,
                  "atributos_publicos": sorted(n for n in dir(modulo)
                                               if not n.startswith("_"))[:40]})


def afirmacion_a16(contexto: dict) -> dict:
    """A16. Cuanto pesa la contaminacion por relleno, con umbral fijado antes.

    Sonda controlada: se importa la clase del propio fichero, se instancia con la misma
    dimension y con semilla fija, se toma una tanda de secuencias con longitudes conocidas y se
    compara el vector agregado de la fila mas larga con el relleno hasta la longitud real y con
    el relleno hasta el maximo de la tanda. El umbral de decision, 1e-3, esta fijado en el
    pre-registro y no se negocia despues.

    Un REFUTADO aqui tiene que poder distinguirse de una sonda que no llego a correr: si la
    sonda no se ejecuto, el veredicto lleva `estado_de_la_sonda: NO_EJECUTADA` con la causa
    real, y eso se dice antes que cualquier otra cosa.

    Returns:
        Diccionario de la afirmacion con su veredicto.
    """
    generador = np.random.default_rng(20261141)
    matriz = generador.normal(0.0, 0.1, size=(51, 200)).astype(np.float32)
    matriz[0, :] = 0.0
    longitudes = [12, 7, 4]
    sondas = {}
    errores = {}
    for clave, modulo in (("v9_04_BiGRU", contexto.get("v9")),
                          ("v10_11_BiGRUAfinable", contexto.get("v10"))):
        if modulo is None:
            causa = contexto.get("errores", {}).get(clave)
            if causa is None:
                causa = ("NO HAY CAUSA REGISTRADA: el modulo no se importo y tampoco "
                         "se registro el error. Eso es un fallo de P80, no del "
                         "proyecto, y hay que mirarlo antes de fiarse de este "
                         "veredicto")
            errores[clave] = f"el modulo {clave} no se pudo importar. Causa: {causa}"
            continue
            continue
        try:
            clase = resolver_clase_bi_gru(clave, modulo)
            primera = sondear_relleno(clase, matriz, longitudes, 30, 20261142)
            segunda = sondear_relleno(clase, matriz, longitudes, 30, 20261142)
        except Parada:
            raise
        except Exception as exc:  # noqa: BLE001
            errores[clave] = f"{type(exc).__name__}: {exc}"
            continue
        primera["reproducible"] = (primera["distancia_relativa_l2"]
                                   == segunda["distancia_relativa_l2"])
        primera["distancia_de_la_segunda_pasada"] = segunda["distancia_relativa_l2"]
        primera["clase_usada"] = clase.__name__
        sondas[clave] = primera
    if not sondas:
        return {
            "afirmacion": "A16",
            "hallazgo": "H6",
            "naturaleza": "CONFIRMAR",
            "que_se_afirma": "la sonda no llego a ejecutarse, luego la afirmacion no se puede "
                             "resolver y no se puede decir nada del relleno",
            "tolerancia": {"umbral_de_decision": TOL["sonda_de_relleno"]},
            "unidad": "distancia L2 relativa",
            "medida": {"sonda_ejecutada": False, "estado_de_la_sonda": "NO_EJECUTADA",
                       "causa": errores,
                       "por_que_importa": "un REFUTADO por no haber corrido la sonda no es lo "
                                          "mismo que un REFUTADO medido. Este veredicto no "
                                          "sostiene ninguna conclusion sobre el relleno"},
            "exito": "la sonda se ejecuta y devuelve un numero",
            "fallo": "distancia no calculable",
            "stop_si_falla": False,
            "veredicto": "NO_RESUELTA",
            "estado_del_hallazgo": "NO_EVALUABLE",
            "como_se_cierra_h6": "H6 queda sin evaluar en su parte de magnitud. La parte "
                                 "estructural de H6 la resuelve A15, que no depende de esta "
                                 "sonda",
        }
    distancias = {clave: sonda["distancia_relativa_l2"] for clave, sonda in sondas.items()}
    con_efecto = {clave: valor > TOL["sonda_de_relleno"] for clave, valor in distancias.items()}
    if all(con_efecto.values()):
        estado = "EFECTO_MEDIBLE"
    elif any(con_efecto.values()):
        estado = "EFECTO_MEDIBLE_EN_ALGUNAS_CLASES"
    else:
        estado = "SIN_EFECTO_MEDIBLE"
    reproducibles = all(sonda["reproducible"] for sonda in sondas.values())
    cierre = (
        "H6 queda presente en el codigo. La sonda sale SIN EFECTO MEDIBLE, luego se cierra "
        "como presente_en_el_codigo_sin_efecto_medible_a_1e-3 y NO se afirma contaminacion"
        if estado == "SIN_EFECTO_MEDIBLE" else
        "H6 queda confirmado en su parte estructural y con efecto medible por la sonda. La "
        "magnitud de la sonda acota el defecto en una red de prueba y NO se extrapola a los "
        "BiGRU reales sin medir")
    return {
        "afirmacion": "A16",
        "hallazgo": "H6",
        "naturaleza": "CONFIRMAR",
        "que_se_afirma": "el defecto estructural de A15 tiene un peso medible, y el peso se "
                         "mide antes de decir nada",
        "tolerancia": {
            "umbral_de_decision": TOL["sonda_de_relleno"],
            "regla": "menor o igual que el umbral es SIN EFECTO MEDIBLE; mayor que el umbral es "
                     "EFECTO MEDIBLE",
            "fijado": "en el pre-registro, antes de ejecutar, y no se negocia despues",
        },
        "unidad": "distancia L2 relativa entre los dos vectores agregados",
        "medida": {
            "sonda_ejecutada": True,
            "estado_de_la_sonda": "EJECUTADA",
            "condiciones": {
                "sin_entrenar": True,
                "sin_datos_del_proyecto": True,
                "matriz_sintetica": "51 filas por 200 columnas, fila 0 a cero, semilla 20261141",
                "instanciacion": "la clase del propio fichero, misma dimension y semilla fija",
                "captura_del_vector_agregado": "gancho previo en el dropout, que recibe el "
                                               "vector agregado y no lo recalcula",
            },
            "sondas": sondas,
            "distancias": distancias,
            "con_efecto_measurable": con_efecto,
            "reproducible": reproducibles,
            "errores": errores,
        },
        "exito": "la sonda se ejecuta y devuelve un numero, con los dos vectores y la distancia "
                 "registrados",
        "fallo": "distancia no calculable, o variacion no nula entre dos corridas con la misma "
                 "semilla",
        "stop_si_falla": False,
        "veredicto": "CONFIRMADO" if reproducibles else "REFUTADO",
        "estado_del_hallazgo": estado,
        "como_se_cierra_h6": cierre,
    }


# ===== SECCION 10: H7, LA FASE DL (A17, A18, A19) =====


def leer_recuentos_del_holdout() -> dict:
    """Lee del holdout v2 EXCLUSIVAMENTE los dos recuentos de filas y el hash de ids.

    El documento NO se deserializa. Se lee como texto y se extraen solo esas tres claves con
    una expresion regular, de forma que la lista de identificadores no llega a existir como
    objeto en memoria. El pre-registro lo dice literalmente: la lista de identificadores no se
    lee, no se deserializa y no se cruza con ninguna otra tabla. Un `json.loads` del fichero
    entero seria inocuo con este fichero, pero no cumpliria el contrato escrito.

    Returns:
        Diccionario con los tres campos leidos y la lista de claves no leidas.

    Raises:
        SystemExit: si falta algun campo permitido.
    """
    texto = HOLDOUT_V2.read_text(encoding="utf-8")
    leidos: dict = {}
    claves_vistas: list = []
    for clave, tipo in CAMPOS_PERMITIDOS_DEL_HOLDOUT.items():
        patron = re.compile(r'"' + re.escape(clave) + r'"\s*:\s*("(?:[^"\\]|\\.)*"|-?\d+)')
        encontrado = patron.search(texto)
        if encontrado is None:
            raise SystemExit(f"STOP: config/holdout_v2.json no trae la clave permitida {clave!r}")
        bruto = encontrado.group(1)
        leidos[clave] = tipo(bruto.strip('"')) if tipo is str else tipo(int(bruto))
    for clave in re.findall(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:', texto):
        if clave not in claves_vistas:
            claves_vistas.append(clave)
    leidos["campos_leidos"] = sorted(CAMPOS_PERMITIDOS_DEL_HOLDOUT)
    leidos["campos_no_leidos"] = sorted(set(claves_vistas) - set(CAMPOS_PERMITIDOS_DEL_HOLDOUT))
    leidos["documento_deserializado"] = False
    leidos["como_se_leyo"] = ("con expresion regular sobre el texto, sin json.loads, para que "
                             "ninguna lista del documento llegue a existir como objeto")
    return leidos


def nombres_que_vienen_de_read_train(arbol: ast.AST) -> set:
    """Nombres del fichero asignados desde una llamada a `read_train`.

    Es el criterio estructural que distingue el pool de la fase DL de cualquier otra llamada a
    `split`. Un `divisor.split(train, ...)` y un `linea.split()` tienen la misma forma en el
    AST, asi que la unica forma de separarlos sin heuristica de ventanas es mirar de donde
    sale el objeto que se reparte.

    Args:
        arbol: raiz del arbol sintactico.

    Returns:
        Conjunto de nombres asignados desde `read_train()`.
    """
    nombres = set()
    for nodo in ast.walk(arbol):
        if not (isinstance(nodo, ast.Assign) and len(nodo.targets) == 1
                and isinstance(nodo.targets[0], ast.Name)):
            continue
        for hijo in ast.walk(nodo.value):
            if isinstance(hijo, ast.Call):
                partes = ruta_de_atributo(hijo.func) or []
                if partes and partes[-1] == "read_train":
                    nombres.add(nodo.targets[0].id)
    return nombres


def resolver_pool_de_cv(texto: str, nombre: str) -> dict:
    """Localiza la llamada a `split` que reparte el pool de la fase DL.

    Se recorren TODAS las llamadas cuyo atributo se llama `split`, se registra cada una con su
    linea y su base, y se marca como la del pool de CV solo la que reparte un objeto que viene
    de `read_train()`. Tomar la ultima, o la que se encuentre primero, daria por resultado el
    `split` de una cadena, que es un metodo de texto y no un repartidor de filas.

    Las candidatas se ordenan por linea: `ast.walk` es un recorrido en anchura, luego sin
    ordenar la lista sale en orden de profundidad y no de linea, que es mas dificil de leer y
    de comparar con el fichero.

    Returns:
        Diccionario con la llamada elegida, todas las candidatas y el por que.
    """
    arbol = ast.parse(texto, filename=nombre)
    lineas = texto.splitlines()
    del_read_train = nombres_que_vienen_de_read_train(arbol)
    candidatas = []
    for nodo in ast.walk(arbol):
        if not (isinstance(nodo, ast.Call) and isinstance(nodo.func, ast.Attribute)
                and nodo.func.attr == "split"):
            continue
        base = ast.unparse(nodo.func.value)
        objeto = ast.unparse(nodo.args[0]) if nodo.args else None
        viene_de_read_train = objeto in del_read_train if objeto else False
        candidatas.append({
            "linea": nodo.lineno,
            "texto": a_ascii(lineas[nodo.lineno - 1].strip(), f"{nombre}:{nodo.lineno}"),
            "base": base,
            "objeto_repartido": objeto,
            "viene_de_read_train": viene_de_read_train,
            "es_el_pool_de_cv": viene_de_read_train,
            "clasificacion": ("pool de la fase DL" if viene_de_read_train else
                              "split que NO es el pool: no reparte nada que venga de read_train"),
        })
    candidatas.sort(key=lambda c: c["linea"])
    elegida = next((c for c in candidatas if c["es_el_pool_de_cv"]), None)
    return {
        "candidatas": candidatas,
        "n_candidatas": len(candidatas),
        "nombres_que_vienen_de_read_train": sorted(del_read_train),
        "linea_del_pool_de_cv": elegida["linea"] if elegida else None,
        "objeto_del_pool_de_cv": elegida["objeto_repartido"] if elegida else None,
        "encontrado": elegida is not None,
        "criterio": ("una llamada es el pool de la fase DL si el objeto que reparte es un "
                     "nombre asignado desde una llamada a read_train() en el mismo fichero. "
                     "No se decide por la forma de la llamada, porque `divisor.split(train, ...)` "
                     "y `linea.split()` son indistinguibles por forma"),
        "linea_del_read_train": next(
            (nodo.lineno for nodo in ast.walk(arbol)
             if isinstance(nodo, ast.Call)
             and (ruta_de_atributo(nodo.func) or [])[-1:] == ["read_train"]), None),
    }


def afirmacion_a17() -> dict:
    """A17. DL-03 entreno sobre el conjunto completo de etiquetas, holdout v2 incluido.

    Dos vias que deben concordar. La via numerica, que es la que decide: sumar los
    `n_validacion` de los cinco pliegues de DL-03 tiene que dar exactamente 1731, y 1731 tiene
    que ser 1483 + 248 con los dos recuentos publicados. La via estructural apoya pero no
    decide: se cita la linea del `read_train` y la del bucle de CV, y se comprueba que el pool
    que se reparte viene de `read_train()`.

    Returns:
        Diccionario de la afirmacion con su veredicto.

    Raises:
        Parada: si algo no cuadra. El motivo distingue la via numerica de la estructural, para
            no decir que la suma falla cuando lo que fallo fue la lectura del pool.
    """
    informe = json.loads(INFORME_DL.read_text(encoding="utf-8"))
    arquitectura = {a["experiment_id"]: a for a in informe["arquitecturas"]}
    pliegues = arquitectura["DL-03"]["pliegues"]
    suma = sum(int(p["n_validacion"]) for p in pliegues)
    holdout = leer_recuentos_del_holdout()
    dev = int(holdout["n_tweets_dev"])
    congelado = int(holdout["n_tweets_holdout"])
    identidad = dev + congelado

    fuente = leer_fuente(FUENTE_DL)
    pool = resolver_pool_de_cv(fuente["texto"], fuente["ruta"])
    arbol = ast.parse(fuente["texto"], filename=FUENTE_DL.name)
    lineas = fuente["lineas"]
    linea_bucle_cv = None
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.For) and isinstance(nodo.iter, ast.Call) \
                and ast.unparse(nodo.iter).startswith("enumerate(pliegues"):
            linea_bucle_cv = nodo.lineno
    via_numerica = suma == 1731 and identidad == suma
    pool_completo = bool(pool["encontrado"])
    bucle_presente = linea_bucle_cv is not None
    exito = via_numerica and pool_completo and bucle_presente
    if not exito:
        que_falla = []
        if not via_numerica:
            que_falla.append("la via numerica: la suma no es dev mas holdout")
        if not pool_completo:
            que_falla.append("la via estructural: no se encontro una llamada split que reparta "
                             "un objeto venga de read_train()")
        if not bucle_presente:
            que_falla.append("la via estructural: no se localizo el bucle de pliegues")
        raise Parada("A17", "A17 no se resuelve; falla " + "; ".join(que_falla),
                     {"que_falla": que_falla,
                      "via_numerica_cuadra": via_numerica,
                      "suma_n_validacion_DL_03": suma,
                      "n_tweets_dev": dev,
                      "n_tweets_holdout": congelado,
                      "dev_mas_holdout": identidad,
                      "pool_de_cv_encontrado": pool_completo,
                      "bucle_de_pliegues_encontrado": bucle_presente,
                      "candidatas_split": pool["candidatas"]})
    return {
        "afirmacion": "A17",
        "hallazgo": "H7",
        "naturaleza": "CONFIRMAR",
        "que_se_afirma": "el pool de validacion cruzada de la fase DL son las 1731 filas "
                         "etiquetadas completas, no las 1483 del desarrollo, luego el holdout "
                         "v2 estaba dentro del pool",
        "tolerancia": "cero, igualdad entera exacta",
        "unidad": "filas",
        "medida": {
            "via_numerica_que_decide": {
                "n_validacion_por_pliegue_DL_03": [int(p["n_validacion"]) for p in pliegues],
                "suma": suma,
                "esperado": 1731,
                "n_tweets_dev": dev,
                "n_tweets_holdout": congelado,
                "identidad_dev_mas_holdout": identidad,
                "identidad_se_cumple": identidad == suma,
                "campos_leidos_del_holdout": holdout["campos_leidos"],
                "campos_no_leidos_del_holdout": holdout["campos_no_leidos"],
                "documento_del_holdout_deserializado": holdout["documento_deserializado"],
                "como_se_leyo_el_holdout": holdout["como_se_leyo"],
            },
            "via_estructural_que_apoya": {
                "linea_del_read_train": pool["linea_del_read_train"],
                "texto_del_read_train": a_ascii(
                    lineas[pool["linea_del_read_train"] - 1].strip(), "11_dl:read_train")
                if pool["linea_del_read_train"] else None,
                "linea_del_bucle_de_cv": linea_bucle_cv,
                "texto_del_bucle_de_cv": (a_ascii(lineas[linea_bucle_cv - 1].strip(),
                                                  "11_dl:bucle_cv")
                                          if linea_bucle_cv else None),
                "busqueda_del_pool": pool,
                "el_pool_es_el_train_completo": pool_completo,
                "lectura": "el pool es `train` entero, sin ninguna mascara que saque el "
                           "holdout, luego el holdout v2 esta dentro de los cinco pliegues",
            },
            "nota_de_procedencia": {
                "campo_rutas_del_informe": informe["rutas"],
                "observacion": "el campo rutas apunta a otro arbol de ficheros. El fichero de "
                               "codigo SI existe aqui, luego la via estructural es posible y H7 "
                               "no queda como indeterminada. No se corrige el informe viejo",
            },
        },
        "exito": "la suma es 1731 y la identidad 1731 = 1483 + 248 se cumple exactamente",
        "fallo": "la suma es 1483, o cualquier otro numero que no se descomponga como dev mas "
                 "holdout",
        "stop_si_falla": True,
        "veredicto": "CONFIRMADO",
    }


def afirmacion_a18() -> dict:
    """A18. Lo mismo ocurre en DL-01 y DL-02, y su OOF no puede leerse como mejora sobre V03.

    Parte 1: los tres informes de la fase DL declaran los mismos `n_validacion` por pliegue, con
    lo que el pool es el mismo y la contaminacion del holdout alcanza a los tres.

    Parte 2, y es la que hay que dejar escrita: DL-01 declara un OOF por encima del
    `corrected_cv` de V03. Ese numero esta contaminado por el holdout y por una seleccion de
    epoca no anidada, y NO puede leerse como que un modelo supera a V03. La comparacion no es
    valida porque los dos numeros no se calcularon sobre el mismo conjunto de filas ni con el
    mismo protocolo, y por eso la diferencia se reporta como diferencia aritmetica, con doce
    decimales, sin llamarla delta ni mejora ni ventaja.

    Returns:
        Diccionario de la afirmacion con su veredicto.
    """
    informe = json.loads(INFORME_DL.read_text(encoding="utf-8"))
    arquitectura = {a["experiment_id"]: a for a in informe["arquitecturas"]}
    por_id = {clave: [int(p["n_validacion"]) for p in arquitectura[clave]["pliegues"]]
              for clave in ("DL-01", "DL-02", "DL-03")}
    iguales = len({tuple(valores) for valores in por_id.values()}) == 1
    oof_dl01 = float(arquitectura["DL-01"]["metricas_oof"]["f1_media_columnas"])
    diferencia = oof_dl01 - V03_CV_DECLARADO
    return {
        "afirmacion": "A18",
        "hallazgo": "H7",
        "naturaleza": "CONFIRMAR",
        "que_se_afirma": "parte 1, los tres informes declaran los mismos n_validacion por "
                         "pliegue; parte 2, el OOF de DL-01 no puede leerse como mejora sobre "
                         "V03",
        "tolerancia": {"parte_1": "igualdad exacta de los cinco n_validacion entre los tres "
                                  "informes",
                       "parte_2": "identidad aritmetica, reportada con 12 decimales"},
        "unidad": {"parte_1": "filas por pliegue", "parte_2": "columnas de F1"},
        "medida": {
            "parte_1_pool_compartido": {
                "n_validacion_por_pliegue": por_id,
                "iguales_entre_los_tres": iguales,
                "totales": {clave: sum(valores) for clave, valores in por_id.items()},
            },
            "parte_2_no_es_mejora": {
                "oof_de_DL_01": oof_dl01,
                "corrected_cv_de_V03": V03_CV_DECLARADO,
                "diferencia_aritmetica_con_12_decimales": f"{diferencia:.12f}",
                "como_se_etiqueta": "diferencia aritmetica entre dos cifras calculadas sobre "
                                    "distintos conjuntos de filas y con distinto protocolo",
                "como_NO_se_etiqueta": ["delta", "mejora", "ventaja", "ganancia", "avance"],
                "advertencia": (
                    "este numero esta contaminado por el holdout v2, por A17, y por una "
                    "seleccion de epoca no anidada, por A19. NO puede leerse como que un modelo "
                    "supera a V03: la comparacion no es valida porque los dos numeros no se "
                    "calcularon sobre el mismo conjunto de filas ni con el mismo protocolo"),
                "umbrales_de_V03_solo_como_referencia": dict(UMBRALES_V03),
            },
        },
        "exito": "parte 1 verificada y parte 2 escrita con la advertencia completa",
        "fallo": "los n_validacion difieren entre arquitecturas",
        "stop_si_falla": False,
        "veredicto": "CONFIRMADO" if iguales else "REFUTADO",
    }


def afirmacion_a19() -> dict:
    """A19. La epoca de la fase DL tambien se elige con el pliegue externo.

    El campo `no_es_early_stopping_anidado` NO es una afirmacion fija: se MEDE a partir de los
    mismos criterios que A14, de modo que no puede contradecir al veredicto. Si la medicion
    sale coherente, entonces no es anidado; si no sale coherente, el campo vale False y el
    veredicto REFUTADO explica por que en `por_que`.

    Returns:
        Diccionario de la afirmacion con su veredicto y su bloque `por_que`.
    """
    fuente = leer_fuente(FUENTE_DL)
    analisis = analizar_entrenamiento_con_selector(fuente["texto"], fuente["ruta"],
                                                   "mejor_estado")
    analisis["apariciones_de_la_frase"] = apariciones_de_una_frase(fuente,
                                                                   "early stopping estricto")
    analisis["fuente"] = {"ruta": fuente["ruta"], "sha256": fuente["sha256"],
                          "n_lineas": fuente["n_lineas"]}
    completo = bool(analisis.get("piezas_completas")
                    and analisis.get("las_dos_piezas_que_portedatos_usan_validacion")
                    and analisis.get("sin_particion_interna")
                    and analisis.get("call_site", {}).get("coherente"))
    return {
        "afirmacion": "A19",
        "hallazgo": "H7",
        "naturaleza": "CONFIRMAR",
        "que_se_afirma": "el entrenamiento de la fase DL restaura los pesos de la epoca con "
                         "mejor metrica de VALIDACION del propio pliegue que despues se "
                         "puntua, sin particion interna",
        "tolerancia": "no aplica: comprobacion estructural con lineas exactas",
        "unidad": "lineas y dependencia de datos",
        "medida": {
            "analisis": analisis,
            "por_que": diagnostico_por_hallazgo("A19", {"11_dl_desde_cero": analisis}),
            "criterio_que_decide": "la F1 media por columna de validacion, umbral 0.5, "
                                   "paciencia 5, mejores pesos restaurados",
            "no_es_early_stopping_anidado": completo,
            "como_se_calcula_no_es_early_stopping_anidado": (
                "es el resultado de los cuatro criterios del bloque por_que, no una "
                "afirmacion fija. Solo vale True si las cuatro piezas existen, las que portan "
                "datos usan la validacion del pliegue externo, no hay particion interna y el "
                "call site es coherente. Asi el campo no puede contradecir al veredicto"),
        },
        "exito": "las piezas existen, mismo pliegue, sin particion interna",
        "fallo": "existe particion interna, y H7 queda refutado en esta parte",
        "stop_si_falla": False,
        "veredicto": "CONFIRMADO" if completo else "REFUTADO",
        "consecuencia": "el sesgo de los tres OOF de la fase DL es optimista por dos causas "
                        "independientes: el holdout dentro del pool (A17) y la seleccion de "
                        "epoca no anidada (esta afirmacion)",
    }


# ===== SECCION 11: CONTROL DE NUMEROS, FIGURAS E INFORMES =====


def figuras(afirmaciones: list) -> dict:
    """Genera como maximo tres figuras, y solo si aportan.

    Se generan: el ratio frente a 1/K en escala logaritmica, el learning rate real frente al
    ideal, y la sonda de relleno. Si una figura no aporta, no se genera.

    Args:
        afirmaciones: lista de diccionarios de afirmacion ya calculadas.

    Returns:
        Diccionario con las figuras escritas y su sha256.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    por_id = {a["afirmacion"]: a for a in afirmaciones}
    escritas: dict = {}
    FIGURAS.mkdir(parents=True, exist_ok=True)

    a3 = por_id.get("A3")
    a4 = por_id.get("A4")
    if a3 and a4 and len(a3["medida"]["ratio_negativo_por_k"]) >= 2:
        ks = sorted(int(k) for k in a3["medida"]["ratio_negativo_por_k"])
        ratios_loss = [a3["medida"]["ratio_negativo_por_k"][str(k)] for k in ks]
        ratios_grad = [a4["medida"]["por_k"][str(k)]["ratio_negativo"] for k in ks]
        ratios_total = [a4["medida"]["por_k"][str(k)]["normas"]["total_codigo"]
                        / a4["medida"]["por_k"][str(k)]["normas"]["total_referencia"]
                        for k in ks]
        ideal = [1.0 / k for k in ks]
        figura, ejes = plt.subplots(figsize=(7, 4.5))
        ejes.loglog(ks, ideal, "k--", label="1/K exacto")
        ejes.loglog(ks, ratios_loss, "o-", label="ratio del termino negativo (loss)")
        ejes.loglog(ks, ratios_grad, "s-", label="componente negativo del gradiente")
        ejes.loglog(ks, ratios_total, "^--", label="norma total del gradiente (informativo)")
        ejes.set_xlabel("K")
        ejes.set_ylabel("razon, escala logaritmica")
        ejes.set_title("A3 y A4: el peso 1/K es exacto; el ratio total no es una asercion")
        ejes.legend(fontsize=8)
        ejes.grid(True, which="both", alpha=0.3)
        figura.tight_layout()
        destino = FIGURAS / "p80_ratio_negativo_y_total.png"
        figura.savefig(destino, dpi=130)
        plt.close(figura)
        escritas[destino.name] = {
            "tema": "ratio frente a 1/K en escala logaritmica",
            "nota": "el ratio de la norma total se dibuja como numero informativo, sin cota",
            "sha256": sha256_de_bytes(destino.read_bytes())}

    a8 = por_id.get("A8")
    if a8:
        parte2 = a8["medida"]["parte_2_con_recorte_declarado"]
        total_real = parte2["total_lotes_recorridos"]
        total_plan = parte2["total_lotes_plan"]
        lr_real = parte2["learning_rate"]["real"]
        if total_real and total_plan and lr_real.get("lr_inicial"):
            pasos = np.arange(total_real)
            escala_real = np.maximum(lr_real["lr_minimo"],
                                     lr_real["lr_inicial"] * (1.0 - pasos / max(total_real, 1)))
            escala_ideal = np.maximum(lr_real["lr_minimo"],
                                      lr_real["lr_inicial"] * (1.0 - pasos / max(total_plan, 1)))
            figura, ejes = plt.subplots(figsize=(7, 4.5))
            ejes.plot(pasos, escala_real, label=f"real, total_lotes = {total_real}")
            ejes.plot(pasos, escala_ideal, "--",
                      label=f"si el scheduler hubiera usado {total_plan}")
            ejes.axhline(lr_real["lr_minimo"], color="k", alpha=0.4, label="suelo del lr")
            ejes.set_xlabel("paso, lote a lote")
            ejes.set_ylabel("learning rate")
            ejes.set_title("A8: el desfase del scheduler desplaza el calendario de lr")
            ejes.legend(fontsize=8)
            ejes.grid(True, alpha=0.3)
            figura.tight_layout()
            destino = FIGURAS / "p80_lr_real_frente_a_ideal.png"
            figura.savefig(destino, dpi=130)
            plt.close(figura)
            escritas[destino.name] = {"tema": "learning rate real frente al ideal",
                                      "sha256": sha256_de_bytes(destino.read_bytes())}

    a16 = por_id.get("A16")
    if a16 and a16["medida"].get("distancias"):
        nombres = list(a16["medida"]["distancias"])
        valores = [a16["medida"]["distancias"][nombre] for nombre in nombres]
        figura, ejes = plt.subplots(figsize=(6.5, 4.2))
        ejes.bar(nombres, valores, color="#4477aa")
        ejes.axhline(TOL["sonda_de_relleno"], color="k", linestyle="--",
                     label=f"umbral {TOL['sonda_de_relleno']}")
        ejes.set_ylabel("distancia L2 relativa")
        ejes.set_title("A16: sonda del relleno en redes de prueba, sin entrenar")
        ejes.legend(fontsize=8)
        ejes.grid(True, axis="y", alpha=0.3)
        figura.tight_layout()
        destino = FIGURAS / "p80_sonda_de_relleno.png"
        figura.savefig(destino, dpi=130)
        plt.close(figura)
        escritas[destino.name] = {"tema": "sonda de relleno",
                                  "sha256": sha256_de_bytes(destino.read_bytes())}
    return escritas


def veredictos_por_hallazgo(afirmaciones: list) -> dict:
    """Resume los veredictos por hallazgo H1 a H7.

    Un hallazgo queda CLASIFICADO en cuatro estados y solo en cuatro: CONFIRMADO, REFUTADO, NO
    APLICABLE POR CONSTRUCCION o NO_EVALUABLE. NO_EVALUABLE es para la afirmacion que no llego
    a ejecutarse, que es distinto de la que se ejecuto y salio mal: sin ese estado, un
    veredicto que no se midio se confunde con uno que se midio y refute.

    Args:
        afirmaciones: lista de diccionarios de afirmacion.

    Returns:
        Diccionario con el estado de cada hallazgo y las afirmaciones que lo sostienen.
    """
    resumen: dict = {}
    for afirmacion in afirmaciones:
        resumen.setdefault(afirmacion["hallazgo"], []).append(afirmacion)
    salida = {}
    for hallazgo, items in sorted(resumen.items()):
        estados = {item["veredicto"] for item in items}
        if "REFUTADO" in estados:
            estado = "REFUTADO"
        elif "NO_EVALUABLE" in estados:
            estado = "NO_EVALUABLE"
        elif estados == {"NO APLICABLE POR CONSTRUCCION"}:
            estado = "NO APLICABLE POR CONSTRUCCION"
        else:
            estado = "CONFIRMADO"
        salida[hallazgo] = {
            "estado": estado,
            "afirmaciones": [item["afirmacion"] for item in items],
            "detalle": [f"{item['afirmacion']}: {item['veredicto']}" for item in items],
        }
    return salida


def correccion_de_p79(afirmaciones: list) -> dict:
    """Redacta la correccion de las tres afirmaciones erroneas de P79, en documento nuevo.

    Returns:
        Diccionario con las tres correcciones.
    """
    por_id = {a["afirmacion"]: a for a in afirmaciones}
    return {
        "principio": "P80 no edita P79 ni su informe. La correccion va en un documento nuevo, y "
                     "ninguna de las tres correcciones cambia los hallazgos de H1 que P79 si "
                     "acerto",
        "correcciones": [
            {
                "afirmacion_de_P79": "lectura de las semillas del scheduler: "
                                     "semillas_desalineadas = false, con una unica lista "
                                     "entrenamiento = [0, 1, 2, 3, 4, 5]",
                "por_que_mal": "su heuristica metio los dos bucles en el mismo grupo, y el "
                               "booleano salia False no porque los rangos coincidieran, sino "
                               "porque solo habia un grupo. Su propia lista tiene 6 elementos "
                               "para EPOCAS = 5, y eso ya lo contradicia",
                "veredicto_real": f"A7: {por_id['A7']['veredicto']}" if "A7" in por_id
                else "no alcanzado",
                "lo_que_dice_P80": "los dos bucles usan rangos distintos, leidos por AST con "
                                   "mapa de padres, sin buscar hacia atras un encabezado ni "
                                   "etiquetar por una ventana de cadena",
            },
            {
                "afirmacion_de_P79": "la_identidad_se_cumple = false en los cuatro valores de K",
                "por_que_mal": "comparaba pos + neg_codigo contra l_codigo, que es "
                               "-(pos + neg_codigo): faltaba el signo menos. Y el script no "
                               "paraba, con lo que un indicador que siempre falla no informa de "
                               "nada",
                "veredicto_real": f"A9: {por_id['A9']['veredicto']}" if "A9" in por_id
                else "no alcanzado",
                "lo_que_dice_P80": "la identidad se afirma con el signo correcto y con parada, y "
                                   "ademas se comprueba que la identidad mal puesta falla, para "
                                   "que la asercion tenga poder discriminante",
            },
            {
                "afirmacion_de_P79": "punto_5_artefacto_congelado, que mide loss y normas de "
                                     "gradiente sobre el artefacto",
                "por_que_mal": "asigno E = S = W, la misma matriz guardada, a la tabla de "
                               "entrada y a la de salida. El artefacto solo guarda W",
                "veredicto_real": f"A13: {por_id['A13']['veredicto']}" if "A13" in por_id
                else "no alcanzado",
                "lo_que_dice_P80": "no se mide el artefacto entrenado porque no se puede: se "
                                   "declara el limite y la afirmacion queda NO APLICABLE POR "
                                   "CONSTRUCCION, sin sustituir la tabla que falta por W",
            },
        ],
    }


def cifras_de_p79_que_p80_no_replica() -> dict:
    """Declara que cifras de P79 P80 no replica, y con que criterio las trata.

    Returns:
        Diccionario con las cifras y su tratamiento.
    """
    return {
        "por_que": "ninguna cifra de P79 se da por replicada ni por refutada de antemano. P80 "
                   "produce su propia medida o declara que no puede",
        "cifras": [
            {"cifra": 0.1, "que_era": "ratio del termino negativo con K = 10",
             "tratamiento_en_P80": "medida por A3, con la tolerancia del pre-registro y con la "
                                    "linea del fuente citada"},
            {"cifra": 0.08033795,
             "que_era": "ratio de la norma del gradiente de entrada, sobre un artefacto con "
                        "E = S = W",
             "tratamiento_en_P80": "RETIRADA. A5 no afirma ninguna cota y publica dos "
                                   "contraejemplos; ademas la cifra dependia de una matriz que "
                                   "no existe"},
            {"cifra": 1.08412074, "que_era": "loss del codigo sobre el artefacto",
             "tratamiento_en_P80": "NO MEDIBLE. A13 lo declara NO APLICABLE POR "
                                    "CONSTRUCCION"},
            {"cifra": 9.78555861, "que_era": "loss de la referencia sobre el artefacto",
             "tratamiento_en_P80": "NO MEDIBLE, por el mismo motivo"},
            {"cifra": [0.1, 1.0], "que_era": "cota inferior y superior del ratio total",
             "tratamiento_en_P80": "REFUTADA antes de ejecutar. A5 exige contraejemplo "
                                   "numerico y no publica ninguna cota"},
            {"cifra": 0.00215054,
             "que_era": "desvio relativo del total de lotes, con un recorte de 2000 lineas",
             "tratamiento_en_P80": "medida de nuevo por A8, en dos partes, con el recorte "
                                    "declarado y con la parte 1 sin recorte"},
            {"cifra": "los dos bucles usan el mismo rango de semillas",
             "que_era": "lectura del plan frente al entrenamiento",
             "tratamiento_en_P80": "PRESUMIDA ERRONEA antes de ejecutar. A7 la refuta con "
                                   "conjuntos de enteros exactos"},
        ],
    }


def construir_json(afirmaciones: list, contexto: dict, huella_antes: dict, huella_despues: dict,
                   pre: dict, figuras_hechas: dict, inicio: datetime, ascii_observado: dict,
                   cronometro: dict) -> dict:
    """Monta el diccionario completo del informe.

    Args:
        afirmaciones: lista de diccionarios de afirmacion.
        contexto: contexto de modulos, para el registro de errores de importacion.
        huella_antes: huella del artefacto antes de la primera lectura.
        huella_despues: huella del artefacto despues de la ultima lectura.
        pre: verificacion del pre-registro.
        figuras_hechas: figuras escritas con su sha256.
        inicio: instante de arranque.
        ascii_observado: resultado observado del escaneo byte a byte del script.
        cronometro: tiempos por seccion.

    Returns:
        El diccionario del informe, listo para volcar a JSON.
    """
    return {
        "familia": "P80_AUDITORIA_WORD2VEC",
        "experiment_id": "P80-01",
        "fase": "Fase 2 del plan p80_p82_auditoria_w2v_rnn_bayesiano",
        "naturaleza": "AUDITORIA FORENSE. Sin entrenamiento, sin reentrenamiento, sin "
                      "submission y sin tocar el artefacto congelado",
        "fecha_inicio": inicio.isoformat(timespec="seconds"),
        "fecha_fin": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "formulacion": FORMULACION,
        "pre_registro": pre,
        "alcance": {
            "que_se_audita": [
                "la loss de SkipGram de p71_04:182-216",
                "el gradiente y su descomposicion con el factor 1/K",
                "el calculo de total_lotes del scheduler de p72_01:91-96 frente al recorrido "
                "de p72_01:98-120",
                "la recuperabilidad de la matriz de salida del artefacto",
                "la eleccion de epoca de los BiGRU de v9_04 y v10_11",
                "el orden del enmascarado del relleno en esos BiGRU",
                "el pool de CV y la seleccion de epoca de la fase DL"],
            "que_no_se_hace": [
                "reentrenar el Word2Vec, ni de 10 millones de tokens ni de un corpus sintetico "
                "con fines de conclusion",
                "modificar, reentrenar ni sobrescribir el artefacto congelado",
                "modificar V03, sus umbrales 0.44 / 0.51 / 0.46 ni "
                "experiments/baseline_v03_frozen/",
                "generar submission, consultar el leaderboard, leer test o leer el holdout",
                "leer features/v2_structured_features.parquet",
                "editar o borrar P71 a P79 ni los informes que escribieron",
                "responder a si conviene una LSTM, un GRU con atencion o un modelo bayesiano: "
                "eso es P81 y P82",
                "escribir DECISIONS.md, RUNLOG.md ni PROJECT_STATE.md, que son entregables de "
                "la Fase 4 y no de este script"],
            "fronteras_de_escritura": ["reports/p80_auditoria_w2v.json",
                                       "reports/p80_auditoria_w2v.md",
                                       "figuras/p80/",
                                       "el propio script"],
        },
        "artefacto_congelado": {
            "sha256_declarado": ARTEFACTO_SHA256,
            "antes_de_la_primera_lectura": huella_antes,
            "despues_de_la_ultima_lectura": huella_despues,
            "intacto": (huella_antes["sha256"] == huella_despues["sha256"]
                        and huella_antes["bytes"] == huella_despues["bytes"]
                        and huella_antes["mtime_ns"] == huella_despues["mtime_ns"]),
            "modo_de_apertura": "open(ruta, 'rb') y np.load sobre el manejador: nunca en modo "
                                "escritura, nunca se copia, nunca se guarda encima",
            "matrices_derivadas_en_disco": "ninguna: todo lo calculado vive en memoria y sale "
                                           "solo por el JSON",
        },
        "entorno": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "hilos": torch.get_num_threads(),
            "dispositivo": "CPU. No se usa GPU: P80 no entrena nada",
        },
        "afirmaciones": afirmaciones,
        "veredictos_por_hallazgo": veredictos_por_hallazgo(afirmaciones),
        "correccion_de_p79": correccion_de_p79(afirmaciones),
        "cifras_de_p79_que_p80_no_replica": cifras_de_p79_que_p80_no_replica(),
        "limites_declarados": [
            "A12 exige recorrer todos los .npz de modelos/ y features/ y todos los .json de "
            "reports/, mientras que la seccion 2.3 punto 5 prohibe abrir otros ficheros de "
            "modelos/. P80 resuelve la tension por la via menos riesgosa: de cada .npz se leen "
            "solo las cabeceras .npy dentro del ZIP, sin descomprimir los arrays, y se declara "
            "el numero de bytes leidos",
            "la enumeracion de A12 no incluye los checkpoints .pt de models/ ni los ficheros "
            "de modelos/ que no son .npz: es un limite declarado del negativo, no un olvido",
            "P80 lee de config/holdout_v2.json exclusivamente n_tweets_dev, n_tweets_holdout "
            "y hash_ids_csv, y lo hace con expresion regular sobre el texto, sin deserializar "
            "el documento: la lista de identificadores no llega a existir como objeto",
            "la parte 1 de A8 establece el desfase de semillas con aritmetica entera, que es "
            "independiente del corpus; el valor numerico de la desviacion se mide con el "
            "recorte declarado y no se extrapola a la corrida real",
            "la convencion que clasifica nombres de validacion y la que detecta llamadas de "
            "reparto se publican en el informe, con sus expresiones regulares, para que sean "
            "auditables. La segunda ignora los nombres que empiezan por mayuscula, porque esos "
            "son constructores de clase y no repartidores",
            "el calculo del learning rate de A8 usa el LOTE de p72_01, que coincide con el "
            "recorte declarado, y el learning rate inicial como LR0_POR_EJEMPLO por LOTE",
        ],
        "rutas_prohibidas_no_abiertas": dict(RUTAS_PROHIBIDAS),
        "errores_de_importacion_de_modulos": contexto.get("errores", {}),
        "figuras": figuras_hechas,
        "asercion_ascii": {
            "metodo": "escaneo byte a byte con read_bytes, rechazando cualquier byte mayor que "
                      "127, ademas del verificador oficial del proyecto. Las figuras son "
                      "binarias por construccion, y lo que se les exige es el nombre en ASCII "
                      "y la firma PNG, no el contenido",
            "observado": ascii_observado,
            "sustituciones_ascii": SUSTITUCIONES_ASCII,
        },
        "cronometro": cronometro,
        "escritura": {
            "modo": "temporal con sufijo .tmp_p80, verificacion byte a byte del temporal, y "
                    "renombrado atomico al final",
            "directorio_de_figuras": str(FIGURAS.relative_to(RAIZ)),
        },
    }


def escribir_markdown(afirmaciones: list, informe: dict) -> list:
    """Construye el markdown del informe, linea a linea, en ASCII puro.

    Args:
        afirmaciones: lista de diccionarios de afirmacion.
        informe: el diccionario completo del informe.

    Returns:
        Lista de lineas del markdown.
    """
    por_id = {a["afirmacion"]: a for a in afirmaciones}
    a3, a4, a5 = por_id["A3"], por_id["A4"], por_id["A5"]
    a7, a8 = por_id["A7"], por_id["A8"]
    a9, a10 = por_id["A9"], por_id["A10"]
    a11, a12, a13 = por_id["A11"], por_id["A12"], por_id["A13"]
    a14, a15, a16 = por_id["A14"], por_id["A15"], por_id["A16"]
    a17, a18 = por_id["A17"], por_id["A18"]
    linea_215 = a3["medida"]["linea_del_fuente"]
    parte1 = a8["medida"]["parte_1_sin_recorte"]
    parte2 = a8["medida"]["parte_2_con_recorte_declarado"]
    via17 = a17["medida"]["via_numerica_que_decide"]
    via18 = a18["medida"]
    numeros_no_comparables = ", ".join(
        f"{clave} {valor}" for clave, valor
        in a14["consecuencia_obligatoria"]["numeros_no_comparables"].items())
    apariciones_packing = sum(f["apariciones_de_pack_padded_sequence"]
                              for f in a15["medida"]["ficheros"].values())
    linea_a16 = (f"- A16, sonda de relleno: estado {a16['medida'].get('estado_de_la_sonda')}, "
                 f"resultado {a16.get('estado_del_hallazgo', 'sin dato')}, con el umbral fijado "
                 f"en {TOL['sonda_de_relleno']} antes de ejecutar")
    linea_a18 = (f"- A18, el OOF de DL-01 es `{via18['parte_2_no_es_mejora']['oof_de_DL_01']}` "
                 f"y el `corrected_cv` de V03 es "
                 f"`{via18['parte_2_no_es_mejora']['corrected_cv_de_V03']}`. La diferencia "
                 f"aritmetica es "
                 f"`{via18['parte_2_no_es_mejora']['diferencia_aritmetica_con_12_decimales']}` "
                 "y NO se etiqueta como delta, mejora ni ventaja: los dos numeros no se "
                 "calcularon sobre el mismo conjunto de filas ni con el mismo protocolo.")
    lineas = [
        "# P80: auditoria del entrenador Word2Vec, sin repetir los errores de P79",
        "",
        "## Formulacion",
        "",
        f"    {FORMULACION}",
        "",
        "En ningun caso se afirma que Word2Vec sea mejor que V03, ni que una RNN o un modelo "
        "bayesiano lo superen.",
        "",
        "## Pre-registro",
        "",
        f"- fichero: `{informe['pre_registro']['ruta']}`",
        f"- sha256 declarado: `{informe['pre_registro']['sha256']}`",
        f"- sha256 recomputado: `{informe['pre_registro']['sha256_recomputado']}`",
        f"- coincide: {informe['pre_registro']['coincide']} | commit `{COMMIT_PREREGISTRO}`",
        "",
        "## Veredicto por hallazgo",
        "",
        "| Hallazgo | Estado | Afirmaciones |",
        "|---|---|---|",
    ]
    for hallazgo, datos in informe["veredictos_por_hallazgo"].items():
        lineas.append(f"| {hallazgo} | {datos['estado']} | "
                      f"{', '.join(datos['afirmaciones'])} |")
    lineas += [
        "",
        "Un hallazgo queda CLASIFICADO, no medido, en cuatro estados y solo en cuatro: "
        "CONFIRMADO, REFUTADO, NO APLICABLE POR CONSTRUCCION y NO_EVALUABLE. NO APLICABLE "
        "significa que la pregunta no tiene respuesta posible con los datos que existen. "
        "NO_EVALUABLE significa que la comprobacion no llego a ejecutarse, que es distinto de "
        "que se ejecutara y saliera mal.",
        "",
        "## Las diecinueve afirmaciones",
        "",
        "| Id | Hallazgo | Naturaleza | Tolerancia | Veredicto |",
        "|---|---|---|---|---|",
    ]
    for afirmacion in afirmaciones:
        tolerancia = afirmacion["tolerancia"]
        if isinstance(tolerancia, dict):
            tolerancia = "; ".join(f"{clave}: {valor}" for clave, valor in tolerancia.items())
        elif isinstance(tolerancia, float):
            tolerancia = str(tolerancia)
        lineas.append(f"| {afirmacion['afirmacion']} | {afirmacion['hallazgo']} | "
                      f"{afirmacion['naturaleza']} | {tolerancia} | "
                      f"**{afirmacion['veredicto']}** |")
    lineas += [
        "",
        "## H1: la loss no es SGNS, y el peso exacto es 1/K",
        "",
        f"- A3, desviacion maxima de 1/K en los K con evidencia: "
        f"`{a3['medida']['maximo_en_k_con_evidencia']}`, con tolerancia "
        f"`{TOL['ratio_del_termino_negativo']}`. En K = 1 el ratio vale 1 y no cuenta como "
        "evidencia.",
        f"- A3, linea citada: `{linea_215['fichero']}:{linea_215['linea']}`",
        f"- A3, texto: `{linea_215['texto_literal']}`",
        f"- A4, maximo residuo de la descomposicion: `{a4['medida']['maximo_residuo']}`, con "
        f"tolerancia `{TOL['residuo_de_descomposicion']}`",
        "- A4, los dos componentes se calculan en funciones separadas y cada una devuelve "
        "solo su termino. Si una sola devolviera el total, el componente negativo llevaria "
        "tambien el positivo, la suma lo contaria dos veces y la razon de los negativos no "
        "seria 1/K.",
        f"- A5, la razon de la norma total NO es una asercion: se publica con la etiqueta "
        f"`{a5['medida']['ratio_de_la_norma_del_gradiente_total']['etiqueta']}`, sin ninguna "
        "cota, y con dos contraejemplos numericos que la refutan",
        "",
        "## H2 y H3: el desfase del scheduler y la identidad de la loss",
        "",
        f"- A7, conjunto del plan: `{a7['medida']['conjunto_plan']}`",
        f"- A7, conjunto del entrenamiento: `{a7['medida']['conjunto_entrenamiento']}`",
        f"- A8 parte 1, sin recorte: las semillas difieren en todas las epocas: "
        f"`{parte1['difieren_en_todas_las_epocas']}`, y la primera semilla del plan no aparece "
        "en el entrenamiento: "
        f"`{parte1['la_primera_semilla_del_plan_no_aparece_en_el_entrenamiento']}`",
        f"- A8 parte 2, con recorte declarado de {parte2['recorte']['lineas_leidas']} lineas, "
        f"{parte2['recorte']['tokens']} tokens y lote {parte2['recorte']['lote']}: "
        f"`total_lotes_plan` = `{parte2['total_lotes_plan']}`, "
        f"`total_lotes_recorridos` = `{parte2['total_lotes_recorridos']}`, desviacion relativa "
        f"`{parte2['desvio_relativo']}`",
        f"- A8, comparacion con P79: `{a8['medida']['comparacion_con_p79']['coincide']}`. "
        f"{a8['medida']['comparacion_con_p79']['explicacion']}",
        f"- A9, residuo maximo de `l_codigo + pos + neg_codigo = 0`: "
        f"`{a9['medida']['residuo_maximo']}`, con tolerancia "
        f"`{TOL['identidad_de_la_loss']}`",
        f"- A10, separacion minima de la identidad MAL puesta: "
        f"`{a10['medida']['separacion_minima']}`. Tiene que fallar, y falla.",
        "",
        "## H4: el artefacto congelado",
        "",
        f"- A11, claves del npz: `{a11['medida']['claves']}`, y una sola matriz",
        f"- A12, ficheros `.npz` inspeccionados por cabecera: "
        f"{a12['medida']['cobertura']['npz_inspeccionados']}, con "
        f"{a12['medida']['cobertura']['bytes_leidos_en_total']} bytes leidos en total y sin "
        f"descomprimir ningun array. Ficheros JSON de reports/: "
        f"{a12['medida']['cobertura']['json_inspeccionados']}, con "
        f"{a12['medida']['cobertura']['claves_json_recorridas']} claves recorridas.",
        f"- A12, candidatos con forma de tabla de embeddings: "
        f"`{a12['medida']['n_candidatos']}`",
        f"- A13, `evidencia_del_estado_final = {a13['medida']['evidencia_del_estado_final']}`. "
        "La tabla de salida NO EXISTE y no se sustituye por `W`. En consecuencia, las cifras "
        "absolutas de P79 sobre el artefacto describen un par (E, S) que nunca existio, y P80 "
        "no publica ninguna cifra sobre el artefacto.",
        f"- A13, frase correcta: `{FRASE_SUSTITUIDA_P79}`",
        "",
        "## H5 y H6: los BiGRU",
        "",
        f"- A14, los numeros {numeros_no_comparables} quedan marcados como NO COMPARABLES con el "
        "protocolo actual, con un sesgo optimista de magnitud NO MEDIDA",
    ]
    for clave, datos in a14["medida"]["por_que"]["por_fichero"].items():
        lineas.append(f"- A14, {clave}: {datos['lectura']}"
                      + (f" Fallan: {', '.join(datos['criterios_que_fallan'])}."
                         if datos["criterios_que_fallan"] else ""))
    lineas += [
        "- A14 y A19, la cadena `early stopping estricto` con la que esos informes se describen "
        "a si mismos es una descripcion incorrecta: no hay particion interna, luego no es early "
        "stopping anidado. Los ficheros NO se editan; la correccion va aqui.",
        f"- A15, `pack_padded_sequence` aparece {apariciones_packing} veces en los dos "
        "ficheros, y la mascara se aplica DESPUES de la GRU",
        linea_a16,
        "",
        "## H7: la fase DL",
        "",
        f"- A17, `n_validacion` por pliegue de DL-03: "
        f"`{via17['n_validacion_por_pliegue_DL_03']}`, suma `{via17['suma']}`, y "
        f"`{via17['n_tweets_dev']} + {via17['n_tweets_holdout']} = "
        f"{via17['identidad_dev_mas_holdout']}`. El holdout v2 estaba dentro del pool.",
        "- A17, el pool de CV se localiza por el hecho de repartir un objeto que viene de "
        "`read_train()`, y no tomando la ultima llamada `split`: un `linea.split()` de una "
        "cadena es indistinguible por forma de un `divisor.split(train, ...)`. Todas las "
        "candidatas quedan listadas en el JSON con su clasificacion.",
        f"- A17, el holdout se lee con expresion regular sobre el texto, sin deserializar el "
        f"documento: `documento_deserializado = "
        f"{via17['documento_del_holdout_deserializado']}`",
        f"- A18, los tres informes declaran los mismos `n_validacion`: "
        f"`{via18['parte_1_pool_compartido']['iguales_entre_los_tres']}`, luego la "
        "contaminacion alcanza a los tres",
        linea_a18,
        "- A19, la epoca de la fase DL tambien se elige con el pliegue externo: el sesgo de los "
        "tres OOF es optimista por dos causas independientes. El campo "
        "`no_es_early_stopping_anidado` se calcula con los mismos cuatro criterios que A14, de "
        "modo que no puede contradecir al veredicto.",
        "",
        "## Correccion de las tres afirmaciones de P79",
        "",
    ]
    for correccion in informe["correccion_de_p79"]["correcciones"]:
        lineas += [
            f"### {correccion['afirmacion_de_P79']}",
            "",
            f"- por que estaba mal: {correccion['por_que_mal']}",
            f"- veredicto real: {correccion['veredicto_real']}",
            f"- lo que dice P80: {correccion['lo_que_dice_P80']}",
            "",
        ]
    lineas += [
        informe["correccion_de_p79"]["principio"] + ".",
        "",
        "## Limites declarados",
        "",
    ]
    for limite in informe["limites_declarados"]:
        lineas.append(f"- {limite}")
    linea_intacto = informe["artefacto_congelado"]["intacto"]
    lineas += [
        "",
        "## Artefacto congelado: integridad",
        "",
        f"- sha256 declarado: `{ARTEFACTO_SHA256}`",
        f"- antes de la primera lectura: "
        f"`{informe['artefacto_congelado']['antes_de_la_primera_lectura']['sha256']}`",
        f"- despues de la ultima lectura: "
        f"`{informe['artefacto_congelado']['despues_de_la_ultima_lectura']['sha256']}`",
        f"- intacto en sha256, tamano y mtime_ns: {linea_intacto}",
        "",
        "## Conclusion",
        "",
    ]
    estados = informe["veredictos_por_hallazgo"]
    if (estados.get("H1", {}).get("estado") == "CONFIRMADO"
            and estados.get("H2", {}).get("estado") == "CONFIRMADO"
            and estados.get("H4", {}).get("estado") == "NO APLICABLE POR CONSTRUCCION"):
        lineas.append(
            "Si H1, H2 y H4 quedan como los declara este informe, la conclusion sobre el "
            "artefacto congelado es que es UTILIZABLE pero PESIMISTA, no invalido, y que los "
            "resultados de V9-04, V10-11 y de la fase DL no son comparables con el protocolo "
            "actual.")
    lineas += [
        "",
        "En ningun caso se afirma que Word2Vec sea mejor que V03, y en ningun caso se propone "
        "reentrenar sin que lo decida Carlos.",
        "",
        f"    {FORMULACION}",
        "",
    ]
    return lineas


# ===== SECCION 12: PRUEBAS DE HUMO CON DATOS SINTETICOS =====


class ModuloFalso:
    """Modulo sintetico con las dos constantes que el scheduler resuelve.

    Attributes:
        EPOCAS: numero de epocas.
        SEMILLA: semilla base.
    """

    EPOCAS = 3
    SEMILLA = 100


def fuente_sintetica_del_scheduler() -> str:
    """Texto sintetico con el mismo patron de scheduler, para la prueba de humo de A6.

    Incluye la cadena de alias completa que `resolver_alias_de_modulo` sigue por el codigo,
    `m <- cx["mod"] <- base`, mas la carga por `importlib`, porque el detector no puede dar por
    hecho que el alias apunta al modulo de P71-04: tiene que verlo escrito en el fuente.

    Returns:
        Codigo fuente en Python, con las dos formas de iterador admitidas, una llamada fuera de
        bucle y un `for` con iterador no admitido para comprobar que ahi si se detiene.
    """
    return (
        "import importlib.util\n"
        "import numpy as np\n"
        "\n"
        "def cargar(nombre):\n"
        "    espec = importlib.util.spec_from_file_location(\n"
        "        'base', 'p71_04_entrenar_word2vec.py')\n"
        "    base = importlib.util.module_from_spec(espec)\n"
        "    espec.loader.exec_module(base)\n"
        "    cx = {'SkipGram': base.SkipGram, 'mod': base}\n"
        "    m = cx['mod']\n"
        "    return m\n"
        "\n"
        "def entrenar(m, docs, pares):\n"
        "    planes = []\n"
        "    for ep in range(m.EPOCAS):\n"
        "        r2 = np.random.default_rng(m.SEMILLA + ep)\n"
        "        c, _ = pares(docs, r2)\n"
        "        planes.append(c)\n"
        "    total_lotes = sum(max(1, (len(c) + 256 - 1) // 256) for c in planes)\n"
        "    r0 = np.random.default_rng(m.SEMILLA)\n"
        "    for ep, _ in enumerate(planes, start=1):\n"
        "        r2 = np.random.default_rng(m.SEMILLA + ep)\n"
        "        pares(docs, r2)\n"
        "    return total_lotes\n"
        "\n"
        "def no_admitido(m, docs):\n"
        "    for ep in reversed(range(m.EPOCAS)):\n"
        "        r2 = np.random.default_rng(m.SEMILLA + ep)\n"
        "        r2.random()\n"
        "    return None\n"
    )


def fuente_sintetica_del_entrenador(nombre_funcion: str = "entrenar") -> str:
    """Texto sintetico de una funcion de entrenamiento, para las pruebas de A14 y A19.

    Args:
        nombre_funcion: nombre con el que se define, para que el llamante de la fixture la
            pueda invocar con ese nombre.

    Returns:
        Codigo fuente en Python, con la perdida de validacion decidiendo el estado.
    """
    return (
        f"def {nombre_funcion}(ids_tr, y_tr, ids_va, y_va):\n"
        "    mejor_perdida = float('inf')\n"
        "    mejor_estado = None\n"
        "    for epoca in range(3):\n"
        "        perdida_va = float((y_va - epoca).mean())\n"
        "        if perdida_va < mejor_perdida - 1e-5:\n"
        "            mejor_perdida = perdida_va\n"
        "            mejor_estado = epoca\n"
        "    modelo.load_state_dict(mejor_estado)\n"
        "    probabilidades = torch.sigmoid(modelo(ids_va))\n"
        "    return probabilidades\n"
    )


def fuente_sintetica_con_particion_interna() -> str:
    """Texto sintetico de un entrenador que SI construye una particion interna.

    La particion interna se construye con una llamada a un repartidor, que es la firma que el
    detector reconoce por el nombre de la funcion. Es como se ve una fuga real en el codigo: no
    hace falta que el nombre de la variable nueva tenga forma de validacion.

    Returns:
        Codigo fuente en Python, con una particion interna construida dentro de la funcion.
    """
    return (
        "def entrenar(ids_tr, y_tr, ids_va, y_va):\n"
        "    mejor_perdida = float('inf')\n"
        "    mejor_estado = None\n"
        "    interno_tr, interno_va = partir_interna(ids_tr, y_tr)\n"
        "    for epoca in range(3):\n"
        "        perdida_va = float((y_va - epoca).mean())\n"
        "        if perdida_va < mejor_perdida - 1e-5:\n"
        "            mejor_perdida = perdida_va\n"
        "            mejor_estado = epoca\n"
        "    modelo.load_state_dict(mejor_estado)\n"
        "    probabilidades = torch.sigmoid(modelo(ids_va))\n"
        "    return probabilidades\n"
    )


def fuente_sintetica_con_clase_en_pascal() -> str:
    """Texto sintetico con un constructor de clase cuyo nombre casa con el patron de reparto.

    `CurvasPliegue` contiene `plieg`, luego sin la guarda de mayusculas el detector lo tomaria
    por una llamada de reparto y declararia una particion interna donde no la hay. Esta fuente
    existe para que el humo compruebe esa guarda.

    Returns:
        Codigo fuente en Python, con una clase en PascalCase y sin particion interna.
    """
    return (
        "def entrenar(ids_tr, y_tr, ids_va, y_va):\n"
        "    mejor_perdida = float('inf')\n"
        "    mejor_estado = None\n"
        "    curvas = CurvasPliegue()\n"
        "    for epoca in range(3):\n"
        "        perdida_va = float((y_va - epoca).mean())\n"
        "        if perdida_va < mejor_perdida - 1e-5:\n"
        "            mejor_perdida = perdida_va\n"
        "            mejor_estado = curvas\n"
        "    modelo.load_state_dict(mejor_estado)\n"
        "    probabilidades = torch.sigmoid(modelo(ids_va))\n"
        "    return probabilidades\n"
    )


def fuente_sintetica_con_call_site() -> str:
    """Texto sintetico con un llamante que indexa el segundo elemento del pliegue.

    Reproduce la forma de `v9_04`, donde el trainer recibe `ids[va]` directamente del bucle de
    validacion cruzada, para que la prueba de humo pueda ejercitar la resolucion del call site.

    Returns:
        Codigo fuente en Python, con el bucle de pliegues y la llamada.
    """
    return (
        "def main(ids, y, pliegues):\n"
        "    salida = []\n"
        "    for k, (tr, va) in enumerate(pliegues):\n"
        "        salida.append(entrenar(ids[tr], y[tr], ids[va], y[va]))\n"
        "    return salida\n"
        "\n"
        + fuente_sintetica_del_entrenador()
    )


def fuente_sintetica_con_call_site_anidado() -> str:
    """Llamante con un bucle de variantes entre la llamada y el bucle de pliegues.

    Reproduce la forma de `v10_11`, donde la llamada esta dentro de un `for nombre in
    VARIANTES`. El detector tiene que subir por la cadena de `For` hasta el bucle que se
    desestructura, no quedarse con el mas proximo.

    Returns:
        Codigo fuente en Python, con dos bucles anidados.
    """
    return (
        "def main(ids, y, pliegues):\n"
        "    salida = []\n"
        "    for indice_pliegue, (tr, va) in enumerate(pliegues):\n"
        "        for variante in VARIANTES:\n"
        "            salida.append(entrenar(ids[tr], y[tr], ids[va], y[va]))\n"
        "    return salida\n"
        "\n"
        + fuente_sintetica_del_entrenador()
    )


def fuente_sintetica_del_pool_de_cv() -> str:
    """Fuente sintetica con un `split` de framework y varios `split` de cadena.

    Reproduce el caso que hacia parar a A17 en falso: `linea.split()` de un texto es
    indistinguible por forma de `divisor.split(train, ...)`. La fuente tiene un `split` de texto
    DESPUES del del pool, que es justo el orden que rompia la lectura.

    Returns:
        Codigo fuente en Python, con la cadena de `read_train` y las dos clases de `split`.
    """
    return (
        "from sklearn.model_selection import StratifiedGroupKFold\n"
        "\n"
        "def cargar_datos():\n"
        "    train = data_module.read_train()\n"
        "    return train\n"
        "\n"
        "def construir_pliegues(train, grupos):\n"
        "    divisor = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)\n"
        "    pliegues = [(a, b) for a, b in divisor.split(train, None,\n"
        "                                          groups=grupos)]\n"
        "    for linea in ['uno dos', 'tres cuatro']:\n"
        "        partes = linea.split()\n"
        "        print(partes)\n"
        "    return pliegues\n"
    )


def fuente_sintetica_forma_de_11_dl() -> str:
    """Llamante con la FORMA de `11_dl`: argumentos como nombres desnudos.

    En `11_dl` la llamada no recibe subscripts sino `x_train`, `y_train`, `x_val` e `y_val`, y su
    procedencia esta unas lineas mas arriba. Esa es la forma que el detector por identidad de
    indice no resolvia, y por eso necesita un caso propio en el humo.

    Se incluye ademas `tr_val`, un nombre trampoloso que contiene la cadena `val` y que NO debe
    salir como validacion: no es un subscript, no se enlaza a nada que diga val y su nombre no
    casa con la convencion, luego tiene que quedar como desconocido.

    Returns:
        Codigo fuente en Python, con el llamante y el entrenador.
    """
    return (
        "def main(textos, etiquetas, pliegues, modelo_base, mascara_tr):\n"
        "    salida = []\n"
        "    for numero, (entrenamiento, validacion) in enumerate(pliegues):\n"
        "        datos = preparar(textos, entrenamiento, validacion)\n"
        "        modelo = modelo_base.to('cpu')\n"
        "        tr_val = mascara_tr\n"
        "        x_train = datos.x_train.to('cpu')\n"
        "        x_val = datos.x_val.to('cpu')\n"
        "        y_train = torch.tensor(etiquetas[entrenamiento], dtype=torch.float32)\n"
        "        y_val = torch.tensor(etiquetas[validacion], dtype=torch.float32)\n"
        "        curvas = entrenar_pliegue(modelo, x_train, y_train, x_val, y_val, tr_val)\n"
        "        salida.append(curvas)\n"
        "    return salida\n"
        "\n"
        + fuente_sintetica_del_entrenador("entrenar_pliegue")
    )


def tipos_de_modulo_minimo() -> object:
    """Construye un modulo sintetico con la clase `SkipGram` que H1 necesita.

    Reproduce la clase de `p71_04` linea a linea, para que las pruebas de humo puedan correr sin
    leer nada del proyecto.

    Returns:
        Un modulo con la clase `SkipGram`.
    """
    import types

    fuente = (
        "import torch\n"
        "from torch import nn\n"
        "class SkipGram(nn.Module):\n"
        "    def __init__(self, n_vocab, dim):\n"
        "        super().__init__()\n"
        "        self.entrada = nn.Embedding(n_vocab, dim)\n"
        "        self.salida = nn.Embedding(n_vocab, dim)\n"
        "        lim = 0.5 / dim\n"
        "        nn.init.uniform_(self.entrada.weight, -lim, lim)\n"
        "        nn.init.zeros_(self.salida.weight)\n"
        "    def forward(self, centro, ctx, neg):\n"
        "        v = self.entrada(centro)\n"
        "        u_pos = self.salida(ctx)\n"
        "        u_neg = self.salida(neg)\n"
        "        pos = torch.nn.functional.logsigmoid((v * u_pos).sum(dim=1)).mean()\n"
        "        neg_l = torch.nn.functional.logsigmoid(-(v.unsqueeze(1) * u_neg).sum(dim=2)).mean()\n"
        "        return -(pos + neg_l)\n"
    )
    modulo = types.ModuleType("p80_sintetico_p71")
    exec(compile(fuente, "<humo>", "exec"), modulo.__dict__)  # noqa: S102
    return modulo


def sondear_relleno_en_toy(longitudes: list, longitud_maxima: int, semilla: int) -> dict:
    """Sonda minima de relleno sobre una red sintetica, sin las clases del proyecto.

    Reproduce la misma logica que `sondear_relleno` con una red pequena, para que la prueba de
    humo compruebe el mecanismo de captura del vector agregado y el calculo de la distancia sin
    importar nada del proyecto.

    Args:
        longitudes: longitudes reales de las filas de la tanda.
        longitud_maxima: longitud a la que se rellena en la segunda pasada.
        semilla: semilla de torch.

    Returns:
        Diccionario con las dos pasadas y la distancia.
    """
    torch.manual_seed(semilla)
    entrada = torch.nn.Embedding(51, 8, padding_idx=0)
    gru = torch.nn.GRU(8, 6, batch_first=True, bidirectional=True)
    dropout = torch.nn.Dropout(0.0)
    capturado: list = []

    def gancho(_modulo, argumentos):
        """Captura la entrada del dropout, que es el vector agregado.

        Args:
            _modulo: el modulo al que se engancha.
            argumentos: tupla de entrada del dropout.

        Returns:
            None, para no alterar el calculo.
        """
        capturado.append(argumentos[0].detach().clone())

    manejador = dropout.register_forward_pre_hook(gancho)
    ids = [list(range(1, longitud + 1)) for longitud in longitudes]
    fila = int(np.argmax([len(tokens) for tokens in ids]))

    def agregado(secuencias, longitud: int) -> np.ndarray:
        """Pasa una tanda por la red y devuelve el vector agregado de la fila mas larga.

        Args:
            secuencias: lista de listas de identificadores.
            longitud: longitud maxima de la tanda.

        Returns:
            El vector agregado de la fila observada, como array numpy.
        """
        lote = lote_de_ids(secuencias, longitud)
        with torch.no_grad():
            capturado.clear()
            mascara = (lote != 0).float().unsqueeze(-1)
            salida, _ = gru(entrada(lote))
            dropout((salida * mascara).sum(dim=1) / mascara.sum(dim=1).clamp(min=1.0))
            return capturado[-1][fila].numpy()

    try:
        vector_real = agregado(ids, max(longitudes))
        vector_maximo = agregado(ids, longitud_maxima)
    finally:
        manejador.remove()
    return {
        "longitudes_reales": longitudes,
        "distancia_relativa_l2": distancia_relativa(vector_real, vector_maximo),
    }


def npz_sintetico_en_memoria() -> io.BytesIO:
    """Construye un `.npz` sintetico en memoria, sin escribir nada en disco.

    El modo `--humo` no puede crear ficheros en el proyecto, y `inspeccionar_npz_por_cabecera`
    recibe un manejador, luego el npz se fabrica con `np.savez` sobre un `BytesIO`.

    Returns:
        Buffer con el npz, posicionado al principio.
    """
    buffer = io.BytesIO()
    np.savez(buffer, W=np.zeros((4, 3), dtype=np.float32),
             palabras=np.array(["a", "b", "c", "d"], dtype=object))
    buffer.seek(0)
    return buffer


def npz_con_tabla_de_salida() -> io.BytesIO:
    """`.npz` en memoria con un miembro que SI es la tabla de salida del artefacto.

    El miembro se llama `salida` y tiene la forma exacta `(vocabulario, dim)` del congelado.
    Tiene que disparar la parada, porque es justo la condicion que P80 vigila.

    Returns:
        Buffer con el npz, posicionado al principio.
    """
    buffer = io.BytesIO()
    np.savez(buffer, salida=np.zeros((ARTEFACTO_ESPERADO["vocabulario"],
                                     ARTEFACTO_ESPERADO["dim"]), dtype=np.float32))
    buffer.seek(0)
    return buffer


def npz_con_falsos_positivos() -> io.BytesIO:
    """`.npz` en memoria con los dos falsos positivos reales de la primera ejecucion.

    Dos tablas de ENTRADA de otras corridas, con la primera dimension del vocabulario del
    artefacto y la segunda de otras dimensiones: `(77657, 200)` y `(77657, 300)`. Ninguna
    puede ser la tabla de salida del entrenamiento de 100 dimensiones, porque las dos tablas del
    modelo comparten `dim`.

    Returns:
        Buffer con el npz, posicionado al principio.
    """
    buffer = io.BytesIO()
    np.savez(buffer,
             W_200d=np.zeros((ARTEFACTO_ESPERADO["vocabulario"], 200), dtype=np.float32),
             W_300d=np.zeros((ARTEFACTO_ESPERADO["vocabulario"], 300), dtype=np.float32))
    buffer.seek(0)
    return buffer


def json_con_clave_de_forma() -> dict:
    """Informe sintetico con claves `shape` de dos enteros, como las que hay en reports/.

    Es el segundo falso positivo real: una clave que declara la forma de otra cosa. No es un
    array, luego no puede ser una tabla de salida.

    Returns:
        Diccionario con dos formas declaradas y un par de vectores.
    """
    return {"modelo": {"shape": [ARTEFACTO_ESPERADO["vocabulario"],
                                ARTEFACTO_ESPERADO["dim"]],
                       "forma": [ARTEFACTO_ESPERADO["vocabulario"], 200],
                       "pesos": [[0.1, 0.2, 0.3]]},
            "otro": {"shape": [3, 4]}}


def ejecutar_pruebas_de_humo() -> int:
    """Corre las pruebas de humo con datos sinteticos, sin leer nada del proyecto.

    No lee datos, ni informes, ni el artefacto congelado, no entrena y no escribe nada. Lo unico
    que lee del proyecto es el propio script y el texto de `p71_04_entrenar_word2vec.py`, del
    que A3 cita la linea 215. Sale antes de la seccion 1, de modo que no puede tener efectos.

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

    print("[P80-01] pruebas de humo con datos sinteticos; sin datos, sin artefacto y sin "
          "escrituras", flush=True)
    modulo_p71 = tipos_de_modulo_minimo()

    # --- H1, la loss ---
    a1 = afirmacion_a1(modulo_p71)
    comprobar("A1 la replica es fiel a SkipGram.forward", a1["veredicto"] == "CONFIRMADO")
    a2 = afirmacion_a2(modulo_p71)
    comprobar("A2 el termino positivo es identico e independiente de los negativos",
              a2["veredicto"] == "CONFIRMADO")
    a3 = afirmacion_a3(modulo_p71)
    comprobar("A3 el termino negativo pesa exactamente 1/K, sin `sum(dim=1)` antes del medio",
              a3["veredicto"] == "CONFIRMADO"
              and max(a3["medida"]["desviacion_de_1_sobre_k"][str(k)]
                      for k in RED["k_con_evidencia"])
              <= TOL["ratio_del_termino_negativo"])

    # --- H1, el gradiente ---
    a4 = afirmacion_a4(modulo_p71)
    comprobar("A4 gradiente con la repulsion en 1/K y residuos por debajo de 1e-9",
              a4["veredicto"] == "CONFIRMADO"
              and a4["medida"]["maximo_residuo"] <= TOL["residuo_de_descomposicion"])
    comprobar("A4 el componente negativo pesa 1/K y no la razon de las normas totales",
              all(abs(a4["medida"]["por_k"][str(k)]["ratio_negativo"] - 1.0 / k)
                  <= TOL["componentes_de_gradiente_relativo"]
                  for k in RED["valores_de_k"]))
    comprobar("A4 el componente positivo se mide contra autograd, no contra si mismo",
              all(abs(a4["medida"]["por_k"][str(k)]["ratio_positivo"] - 1.0)
                  <= TOL["componentes_de_gradiente_relativo"]
                  for k in RED["valores_de_k"])
              and max(a4["medida"]["por_k"][str(k)]["residuos"][
                  "componente_positivo_contra_su_autograd"]
                  for k in RED["valores_de_k"]) <= TOL["residuo_de_descomposicion"])
    comprobar("A4 los componentes son aditivos: la suma reproduce el objetivo completo",
              max(a4["medida"]["por_k"][str(k)]["residuos"]["entrada_codigo"]
                  for k in RED["valores_de_k"]) <= TOL["residuo_de_descomposicion"])

    # --- A5, la afirmacion negativa ---
    a5 = afirmacion_a5(a4["medida"]["por_k"])
    comprobar("A5 el ratio total se publica sin ninguna cota y con contraejemplos",
              a5["veredicto"] == "CONFIRMADO"
              and not a5["medida"]["autocomprobacion"]["claves_de_cota_presentes"]
              and len(a5["medida"]["demostracion_de_la_cota"]["contraejemplos"]) == 2)
    comprobar("A5 el contraejemplo del limite inferior da una razon por debajo de 1/K",
              a5["medida"]["demostracion_de_la_cota"]["contraejemplos"][
                  "por_debajo_del_limite_inferior"]["razon"]
              < 1.0 / RED["valores_de_k"][2])

    # --- H3, la identidad ---
    a9 = afirmacion_a9(modulo_p71)
    comprobar("A9 la identidad con el signo correcto cuadra",
              a9["veredicto"] == "CONFIRMADO"
              and a9["medida"]["residuo_maximo"] <= TOL["identidad_de_la_loss"])
    a10 = afirmacion_a10(modulo_p71)
    comprobar("A10 la identidad con el signo mal puesto falla de verdad",
              a10["veredicto"] == "CONFIRMADO"
              and a10["medida"]["separacion_minima"]
              >= TOL["separacion_del_control_negativo"])

    # --- H2, el scheduler por AST ---
    lectura = auditar_semillas_por_ast(
        fuente_sintetica_del_scheduler().split("def no_admitido")[0],
        "sintetico.py", {}, ModuloFalso())
    roles = {c["rol"] for c in lectura["llamadas"]}
    plan = [c for c in lectura["llamadas"] if c["rol"] == "plan"]
    entrenamiento = [c for c in lectura["llamadas"] if c["rol"] == "entrenamiento"]
    comprobar("A6 el AST distingue plan, entrenamiento y llamada fuera de bucle",
              roles == {"plan", "entrenamiento", "fuera_de_bucle"})
    comprobar("A6 resuelve el rango del plan por range(m.EPOCAS)",
              bool(plan) and plan[0]["conjunto_de_semillas"] == [100, 101, 102])
    comprobar("A6 resuelve el rango del entrenamiento por enumerate(..., start=1)",
              bool(entrenamiento)
              and entrenamiento[0]["conjunto_de_semillas"] == [101, 102, 103])
    comprobar("el recorrido del AST encuentra las llamadas a default_rng sin buscar cadenas",
              len(lectura["llamadas"]) == 3)
    iterador_no_admitido = False
    try:
        auditar_semillas_por_ast(fuente_sintetica_del_scheduler(), "sintetico_malo.py", {},
                                 ModuloFalso())
    except Parada as parada:
        iterador_no_admitido = parada.afirmacion == "A6"
    comprobar("A6 se detiene cuando el iterador no es range ni enumerate",
              iterador_no_admitido)

    # --- H6, el relleno: el detector ---
    fuente_mala = (
        "import torch\n"
        "class Red(torch.nn.Module):\n"
        "    def forward(self, x):\n"
        "        em = torch.nn.utils.rnn.pack_padded_sequence(x, None)\n"
        "        return em\n"
    )
    fuente_buena = (
        "import torch\n"
        "class Red(torch.nn.Module):\n"
        "    def __init__(self, m):\n"
        "        self.embedding = torch.nn.Embedding.from_pretrained(m, padding_idx=0)\n"
        "        self.gru = torch.nn.GRU(m.shape[1], 8, batch_first=True)\n"
        "    def forward(self, e):\n"
        "        mascara = (e != 0).float().unsqueeze(-1)\n"
        "        salida, _ = self.gru(self.embedding(e))\n"
        "        suma = (salida * mascara).sum(dim=1)\n"
        "        return suma\n"
    )
    deteccion_mala = auditar_mascara_tardia_en_texto(fuente_mala, "sintetico_packing.py")
    deteccion_buena = auditar_mascara_tardia_en_texto(fuente_buena, "sintetico_mascara.py")
    comprobar("A15 detecta pack_padded_sequence cuando esta",
              deteccion_mala["apariciones_de_pack_padded_sequence"] == 1
              and not deteccion_mala["tres_comprobaciones"]["cero_pack_padded_sequence"])
    comprobar("A15 valida el orden GRU antes que mascara y padding_idx 0 cuando esta sano",
              deteccion_buena["apariciones_de_pack_padded_sequence"] == 0
              and deteccion_buena["orden_gru_antes_que_mascara"]
              and deteccion_buena["padding_idx"] == 0
              and all(deteccion_buena["tres_comprobaciones"].values()))

    # --- H5, el call site y la particion interna ---
    sano = analizar_entrenamiento_con_selector(fuente_sintetica_del_entrenador(),
                                               "sintetico_entrenador.py", "mejor_estado")
    comprobar("A14 encuentra las cuatro piezas en un entrenador sano",
              sano["piezas_completas"])
    comprobar("A14 las dos piezas que portan datos dependen de la validacion",
              sano["las_dos_piezas_que_portedatos_usan_validacion"])
    comprobar("A14 no declara particion interna donde no la hay",
              sano["sin_particion_interna"])
    comprobar("A14 publica la convencion con la que clasifica los nombres",
              "convencion" in sano["clasificacion_de_parametros_publicada"]
              and "parametros_clasificados_como_validacion"
              in sano["clasificacion_de_parametros_publicada"])
    comprobar("A14 enumera los cuatro criterios y marca solo el que falla",
              len(sano["por_que"]["criterios"]) == 4
              and [c["criterio"] for c in sano["por_que"]["criterios"] if not c["observado"]]
              == ["el_call_site_trae_la_validacion_del_pliegue_externo"])
    Sick = analizar_entrenamiento_con_selector(fuente_sintetica_con_particion_interna(),
                                               "sintetico_interno.py", "mejor_estado")
    comprobar("A14 declara particion interna cuando la hay", not Sick["sin_particion_interna"])
    comprobar("A14 marca la particion interna con su linea exacta",
              bool(Sick["llamadas_de_reparto"])
              and Sick["llamadas_de_reparto"][0]["linea"] == 4)
    con_clase = analizar_entrenamiento_con_selector(fuente_sintetica_con_clase_en_pascal(),
                                                     "sintetico_clase.py", "mejor_estado")
    comprobar("A14 no confunde un constructor de clase con un repartidor de filas",
              con_clase["sin_particion_interna"] and not con_clase["llamadas_de_reparto"])
    con_call_site = analizar_entrenamiento_con_selector(
        fuente_sintetica_con_call_site(), "sintetico_callsite.py", "mejor_estado")
    comprobar("A14 resuelve el call site y lo ve coherente con el pliegue externo",
              con_call_site["call_site"]["encontrada"]
              and con_call_site["call_site"]["coherente"]
              and con_call_site["call_site"]["segundo_elemento_del_bucle_de_pliegues"] == "va")
    comprobar("A14 el argumento de validacion se identifica por el indice del pliegue",
              any(argumento["clase"] == "validacion"
                  and "segundo elemento del pliegue" in argumento["evidencia"]
                  for argumento in con_call_site["call_site"]["argumentos_de_validacion"]))
    comprobar("A14 el argumento de entrenamiento NO se clasifica como de validacion",
              con_call_site["call_site"]["clase_de_cada_argumento"].get("ids[tr]")
              != "validacion")
    comprobar("A14 el subscript de entrenamiento SI se clasifica como entrenamiento",
              con_call_site["call_site"]["clase_de_cada_argumento"].get("ids[tr]")
              == "entrenamiento")

    # --- la FORMA de 11_dl: argumentos como nombres desnudos ---
    forma_dl = analizar_entrenamiento_con_selector(fuente_sintetica_forma_de_11_dl(),
                                                    "sintetico_forma_dl.py", "mejor_estado")
    clases_dl = forma_dl["call_site"]["clase_de_cada_argumento"]
    comprobar("A19 por vinculo local: x_val e y_val salen como validacion",
              forma_dl["call_site"]["coherente"]
              and clases_dl.get("x_val") == "validacion"
              and clases_dl.get("y_val") == "validacion"
              and forma_dl["call_site"]["n_argumentos_de_validacion"] == 2)
    comprobar("A19 por vinculo local: x_train e y_train NO salen como validacion",
              clases_dl.get("x_train") == "entrenamiento"
              and clases_dl.get("y_train") == "entrenamiento")
    comprobar("A19 el nombre trampoloso tr_val NO sale como validacion",
              clases_dl.get("tr_val") == "desconocido")
    comprobar("A19 con nombres desnudos pasan los cuatro criterios, como en el fichero real",
              forma_dl["por_que"]["todos_los_criterios_pasan"])
    comprobar("A19 la evidencia dice de donde salio cada clasificacion",
              all("evidencia" in a for a in forma_dl["call_site"]["argumentos_de_validacion"]))
    anidado = analizar_entrenamiento_con_selector(
        fuente_sintetica_con_call_site_anidado(), "sintetico_anidado.py", "mejor_estado")
    comprobar("A14 sube por la cadena de bucles hasta el de pliegues, no se queda en el "
              "interior",
              anidado["call_site"]["coherente"]
              and anidado["call_site"]["segundo_elemento_del_bucle_de_pliegues"] == "va")
    comprobar("A14 con el call site resuelto pasan los cuatro criterios",
              con_call_site["por_que"]["todos_los_criterios_pasan"]
              and anidado["por_que"]["todos_los_criterios_pasan"])

    # --- A17, el pool de CV frente a los split de cadena ---
    pool = resolver_pool_de_cv(fuente_sintetica_del_pool_de_cv(), "sintetico_pool.py")
    comprobar("A17 encuentra el pool de CV aunque haya un split de cadena despues",
              pool["encontrado"]
              and pool["linea_del_pool_de_cv"] == 9
              and pool["objeto_del_pool_de_cv"] == "train")
    comprobar("A17 clasifica los split de cadena como lo que son",
              pool["n_candidatas"] == 2
              and sum(1 for c in pool["candidatas"] if not c["es_el_pool_de_cv"]) == 1
              and pool["candidatas"][1]["linea"] == 12)
    comprobar("A17 la busqueda del pool se apoya en read_train, no en la forma de la llamada",
              pool["nombres_que_vienen_de_read_train"] == ["train"]
              and pool["linea_del_read_train"] == 4)

    # --- A11/A12, la cabecera del npz, en memoria ---
    inspeccion = inspeccionar_npz_por_cabecera(npz_sintetico_en_memoria(), "sintetico.npz")
    comprobar("A12 inspecciona un npz por cabecera y ve las dos claves",
              len(inspeccion["miembros"]) == 2
              and sorted(m["nombre"] for m in inspeccion["miembros"]) == ["W.npy", "palabras.npy"])
    comprobar("A12 lee la forma y el tipo de la cabecera sin descomprimir el array",
              next(m for m in inspeccion["miembros"] if m["nombre"] == "W.npy")["forma"] == [4, 3]
              and next(m for m in inspeccion["miembros"]
                       if m["nombre"] == "W.npy")["dtype"] == "float32"
              and not inspeccion["contenido_de_los_arrays_leido"])
    comprobar("A12 declara cuantos bytes leyo",
              inspeccion["bytes_leidos"] > 0)

    # --- A12: el caso POSITIVO tiene que seguir disparando ---
    n_vocab = ARTEFACTO_ESPERADO["vocabulario"]
    dim = ARTEFACTO_ESPERADO["dim"]
    con_salida = clasificar_miembros_npz(
        inspeccionar_npz_por_cabecera(npz_con_tabla_de_salida(),
                                      "sintetico_salida.npz")["miembros"],
        "sintetico_salida.npz", n_vocab, dim, False)
    comprobar("A12 el caso positivo VE una tabla de salida de la forma exacta",
              len(con_salida["candidatos"]) == 1
              and con_salida["candidatos"][0]["forma"] == [n_vocab, dim])

    # --- A12: los dos falsos positivos reales NO tienen que verse ---
    falsos = clasificar_miembros_npz(
        inspeccionar_npz_por_cabecera(npz_con_falsos_positivos(),
                                      "sintetico_otros.npz")["miembros"],
        "sintetico_otros.npz", n_vocab, dim, False)
    comprobar("A12 NO ve la W de entrada de otra corrida con la misma primera dimension",
              not falsos["candidatos"] and len(falsos["descartados"]) == 2
              and all(d["forma"][0] == n_vocab and d["forma"][1] != dim
                      for d in falsos["descartados"]))
    comprobar("A12 el descarte dice por que: la segunda dimension es de otra corrida",
              all("segunda" in d["motivo"] and "dim" in d["motivo"]
                  for d in falsos["descartados"]))
    encontradas_json = []
    descartadas_json = []
    buscar_tablas_en_json(json_con_clave_de_forma(), n_vocab, dim, encontradas_json,
                          descartadas_json, "sintetico_forma.json")
    comprobar("A12 NO ve una clave shape de dos enteros como tabla de salida",
              not encontradas_json
              and sum(1 for d in descartadas_json if d["tipo"] == "clave_de_forma") == 3)
    comprobar("A12 el descarte de una clave shape dice que es metadato, no un array",
              all("metadato" in d["motivo"] for d in descartadas_json
                  if d["tipo"] == "clave_de_forma"))
    comprobar("A12 la primera dimension sola no basta para el criterio",
              not es_forma_de_salida([n_vocab, 200], n_vocab, dim)
              and not es_forma_de_salida([n_vocab + 1, dim], n_vocab, dim)
              and not es_forma_de_salida([n_vocab], n_vocab, dim)
              and es_forma_de_salida([n_vocab, dim], n_vocab, dim))
    anidada_ok = []
    anidada_ok_desc = []
    buscar_tablas_en_json([[[0.0] * dim] * n_vocab], n_vocab, dim, anidada_ok,
                          anidada_ok_desc, "sintetico_anidado.json")
    comprobar("A12 una lista de filas anidadas SI se ve cuando tiene la forma exacta",
              len(anidada_ok) >= 1
              and all(c["forma"] == [n_vocab, dim] for c in anidada_ok))
    anidada_mala = []
    anidada_mala_desc = []
    buscar_tablas_en_json([[[0.0] * 200] * n_vocab], n_vocab, dim, anidada_mala,
                          anidada_mala_desc, "sintetico_anidado_malo.json")
    comprobar("A12 una lista de filas anidadas de otra anchura NO se ve",
              not anidada_mala
              and any(d["tipo"] == "lista_anidada_en_json" for d in anidada_mala_desc))

    # --- A2 y A6: un fallo tiene que producir Parada, no un informe ---
    tolerancia_guardada = TOL["fidelidad_de_la_replica"]
    try:
        TOL["fidelidad_de_la_replica"] = -1.0
        try:
            resultado = afirmacion_a2(modulo_p71)
            a2_rota = "NO PARO: devolvio " + str(resultado.get("veredicto"))
        except Parada as parada_rota:
            a2_rota = parada_rota.afirmacion
    finally:
        TOL["fidelidad_de_la_replica"] = tolerancia_guardada
    comprobar("A2 con tolerancia imposible produce Parada y no un informe", a2_rota == "A2")

    # Se cambia la funcion en el diccionario del modulo y no con una asignacion local: en
    # Python el nombre quedaria local para TODO el cuerpo y las llamadas anteriores de este
    # mismo humo darian UnboundLocalError.
    lectura_guardada = globals()["auditar_semillas_por_ast"]

    def lectura_rota(*argumentos, **claves):
        """Lectura rota a proposito: dos llamadas con los rangos sin resolver.

        Args:
            argumentos: sin uso.
            claves: sin uso.

        Returns:
            Diccionario con dos llamadas sin rango resuelto.
        """
        return {"llamadas": [{"rol": "plan", "rango_de_epocas": None,
                              "conjunto_de_semillas": None, "linea": 1,
                              "texto_literal": "", "argumento": "",
                              "bucle_contenedor": None, "semillas_por_epoca": None},
                             {"rol": "entrenamiento", "rango_de_epocas": None,
                              "conjunto_de_semillas": None, "linea": 2,
                              "texto_literal": "", "argumento": "",
                              "bucle_contenedor": None, "semillas_por_epoca": None}],
                "contraste_con_p71_08": {"llamadas": []}}

    try:
        globals()["auditar_semillas_por_ast"] = lectura_rota
        try:
            resultado = afirmacion_a6()
            a6_rota = "NO PARO: devolvio " + str(resultado.get("veredicto"))
        except Parada as parada_rota:
            a6_rota = parada_rota.afirmacion
    finally:
        globals()["auditar_semillas_por_ast"] = lectura_guardada
    comprobar("A6 con los rangos sin resolver produce Parada y no un informe", a6_rota == "A6")

    # --- H6, la sonda ---
    sonda_1 = sondear_relleno_en_toy([12, 7, 4], 30, 20261142)
    sonda_2 = sondear_relleno_en_toy([12, 7, 4], 30, 20261142)
    comprobar("A16 la sonda de relleno es reproducible con la misma semilla",
              sonda_1["distancia_relativa_l2"] == sonda_2["distancia_relativa_l2"])
    comprobar("A16 la sonda devuelve un numero finito y no nulo",
              0.0 < sonda_1["distancia_relativa_l2"] < float("inf"))
    comprobar("A16 sin modulos importados la sonda dice NO_EJECUTADA, no REFUTADO",
              afirmacion_a16({"v9": None, "v10": None, "errores": {"v9": "X: Y"}})["medida"][
                  "estado_de_la_sonda"] == "NO_EJECUTADA")
    comprobar("el calculo de la distancia relativa es coherente en un caso conocido",
              abs(distancia_relativa(np.array([3.0, 4.0]), np.array([3.0, 4.0]))) < 1e-15
              and abs(distancia_relativa(np.array([3.0, 4.0]), np.array([0.0, 0.0])) - 1.0)
              < 1e-12)

    # --- auxiliares ---
    comprobar("el control de numeros sospechosos no ve nada raro",
              control_de_numeros_sospechosos({"loss_de_prueba": 1.5, "ratio": 0.25})["hallazgos"]
              == [])
    comprobar("el verificador ASCII no encuentra bytes mayores que 127 en el script",
              verificar_ascii_byte_a_byte([Path(__file__)])[Path(__file__).name])
    recuento = leer_recuentos_del_holdout()
    comprobar("la lectura del holdout no deserializa y solo lee los campos permitidos",
              recuento["documento_deserializado"] is False
              and sorted(recuento["campos_leidos"]) == sorted(CAMPOS_PERMITIDOS_DEL_HOLDOUT)
              and recuento["n_tweets_dev"] + recuento["n_tweets_holdout"] == 1731)

    print(f"[P80-01] humo: {superadas} comprobaciones superadas, {len(fallos)} fallidas",
          flush=True)
    if fallos:
        print(f"[P80-01] humo FALLIDO en: {fallos}", flush=True)
    return superadas


# ===== SECCION 13: ORQUESTACION =====


def test1_loss(contexto: dict) -> list:
    """Ejecuta H1 en su parte de loss: A1, A2 y A3, y para en el primer STOP.

    Args:
        contexto: contexto de modulos.

    Returns:
        Lista de diccionarios de afirmacion, en orden.

    Raises:
        Parada: si no se pudo importar el modulo que contiene la clase auditada.
    """
    log("test1_loss: H1, la loss (A1, A2, A3)")
    modulo_p71 = contexto.get("p71")
    if modulo_p71 is None:
        raise Parada("A1", "no se pudo importar p71_04, luego no hay codigo que auditar",
                     {"errores": contexto.get("errores", {})})
    afirmaciones = [afirmacion_a1(modulo_p71), afirmacion_a2(modulo_p71),
                    afirmacion_a3(modulo_p71)]
    for afirmacion in afirmaciones:
        log(f"  {afirmacion['afirmacion']} {afirmacion['veredicto']}")
    return afirmaciones


def test2_gradientes(contexto: dict) -> list:
    """Ejecuta H1 en su parte de gradiente: A4 y A5.

    Args:
        contexto: contexto de modulos.

    Returns:
        Lista de diccionarios de afirmacion, en orden.
    """
    log("test2_gradientes: H1, el gradiente (A4) y la afirmacion negativa (A5)")
    modulo_p71 = contexto["p71"]
    a4 = afirmacion_a4(modulo_p71)
    a5 = afirmacion_a5(a4["medida"]["por_k"])
    modelo = construir_red_de_prueba(modulo_p71)
    centro, ctx, neg = indices_de_prueba(RED["valores_de_k"][2])
    terminos = terminos_de_la_loss(modelo, centro, ctx, neg)
    control = control_de_numeros_sospechosos({
        "ratio_total_del_gradiente_k10":
            a5["medida"]["ratio_de_la_norma_del_gradiente_total"]["valor_por_k"]["10"],
        "loss_codigo_en_la_red_de_prueba": terminos["loss_codigo"],
        "loss_referencia_en_la_red_de_prueba": terminos["loss_referencia"],
    })
    log(f"  control de numeros sospechosos: {control['hallazgos']}")
    log(f"  A4 {a4['veredicto']} con maximo residuo {a4['medida']['maximo_residuo']}")
    log(f"  A5 {a5['veredicto']}")
    return [a4, a5]


def test3_scheduler(contexto: dict) -> list:
    """Ejecuta H2: A6, A7 y A8, leyendo el scheduler por AST.

    Args:
        contexto: contexto de modulos.

    Returns:
        Lista de diccionarios de afirmacion, en orden.

    Raises:
        Parada: si no se pudo importar el modulo del scheduler.
    """
    log("test3_scheduler: H2, el scheduler (A6, A7, A8)")
    modulo_p72 = contexto.get("p72")
    if modulo_p72 is None:
        raise Parada("A6", "no se pudo importar p72_01, luego no hay scheduler que auditar",
                     {"errores": contexto.get("errores", {})})
    a6 = afirmacion_a6()
    a7 = afirmacion_a7(a6["medida"]["lectura_de_p72_01"])
    a8 = afirmacion_a8(contexto["p71"], modulo_p72)
    for afirmacion in (a6, a7, a8):
        log(f"  {afirmacion['afirmacion']} {afirmacion['veredicto']}")
    return [a6, a7, a8]


def test4_identidad_con_signo(contexto: dict) -> list:
    """Ejecuta H3: A9 con el signo correcto y A10, su control negativo.

    Args:
        contexto: contexto de modulos.

    Returns:
        Lista de diccionarios de afirmacion, en orden.
    """
    log("test4_identidad_con_signo: H3, la identidad (A9) y su control (A10)")
    modulo_p71 = contexto["p71"]
    a9 = afirmacion_a9(modulo_p71)
    a10 = afirmacion_a10(modulo_p71)
    log(f"  A9 {a9['veredicto']} con residuo maximo {a9['medida']['residuo_maximo']}")
    log(f"  A10 {a10['veredicto']} con separacion minima {a10['medida']['separacion_minima']}")
    return [a9, a10]


def test5_artefacto(contexto: dict, huella_antes: dict) -> list:
    """Ejecuta H4: A11, A12 y A13, con el artefacto en solo lectura.

    Args:
        contexto: contexto de modulos.
        huella_antes: huella del artefacto antes de la primera lectura.

    Returns:
        Lista de diccionarios de afirmacion, en orden.
    """
    log("test5_artefacto: H4, el artefacto congelado (A11, A12, A13)")
    a11 = afirmacion_a11(huella_antes)
    log(f"  A11 {a11['veredicto']} | claves {a11['medida']['claves']}")
    a12 = afirmacion_a12()
    cobertura = a12["medida"]["cobertura"]
    log(f"  A12 {a12['veredicto']} | {cobertura['npz_inspeccionados']} npz y "
        f"{cobertura['json_inspeccionados']} json inspeccionados | "
        f"{a12['medida']['n_candidatos']} candidatos")
    linea_del_cx = siguiente_linea_con_texto(leer_fuente(FUENTE_P72), 'cx = {"sembrar"')
    cx = verificar_cx_de_p72(leer_fuente(FUENTE_P72)["texto"], contexto["p71"], linea_del_cx)
    a13 = afirmacion_a13(cx)
    log(f"  A13 {a13['veredicto']} | evidencia_del_estado_final "
        f"{a13['medida']['evidencia_del_estado_final']}")
    return [a11, a12, a13]


def siguiente_linea_con_texto(fuente: dict, fragmento: str) -> int:
    """Localiza la primera linea cuyo texto empieza por el fragmento dado.

    El numero de linea que hay que citar en el informe no se escribe a mano: se localiza en el
    fichero, para que el informe no pueda citar una linea que ya no exista.

    Args:
        fuente: diccionario devuelto por `leer_fuente`.
        fragmento: prefijo del texto a buscar.

    Returns:
        El numero de linea, contado desde 1, o 1 si no se encuentra.
    """
    for numero, linea in enumerate(fuente["lineas"], start=1):
        if linea.strip().startswith(fragmento):
            return numero
    return 1


def test6_bigru(contexto: dict) -> list:
    """Ejecuta H5 y H6: A14, A15 y A16.

    Los dos modulos de BiGRU ya estan importados por `contexto_de_modulos`, que es donde se
    captura el error de importacion con su causa. Aqui solo se registra el resultado, de modo
    que no haya dos sitios donde guardar el fallo y uno se quede sin el motivo.

    Args:
        contexto: contexto de modulos, para la sonda de A16.

    Returns:
        Lista de diccionarios de afirmacion, en orden.
    """
    log("test6_bigru: H5 y H6, los BiGRU (A14, A15, A16)")
    a14 = afirmacion_a14()
    for clave, datos in a14["medida"]["por_que"]["por_fichero"].items():
        log(f"  A14 {clave}: {datos['veredicto_del_fichero']} | "
            f"piezas_completas={datos['piezas_completas']} "
            f"call_site_coherente={datos['call_site_coherente']}"
            + (f" | fallan: {', '.join(datos['criterios_que_fallan'])}"
               if datos["criterios_que_fallan"] else ""))
    a15 = afirmacion_a15()
    log(f"  A15 {a15['veredicto']}")
    for clave in ("v9", "v10"):
        if contexto.get(clave) is not None:
            log(f"  modulo de {clave} importado")
        else:
            log(f"  FALLO: {clave} no se pudo importar. "
                f"{contexto.get('errores', {}).get(clave)}")
    a16 = afirmacion_a16(contexto)
    log(f"  A16 {a16['veredicto']} | sonda {a16['medida'].get('estado_de_la_sonda')} | "
        f"hallazgo {a16.get('estado_del_hallazgo')}")
    return [a14, a15, a16]


def test7_fase_dl() -> list:
    """Ejecuta H7: A17, A18 y A19.

    Returns:
        Lista de diccionarios de afirmacion, en orden.
    """
    log("test7_fase_dl: H7, la fase DL (A17, A18, A19)")
    afirmaciones = [afirmacion_a17(), afirmacion_a18(), afirmacion_a19()]
    for afirmacion in afirmaciones:
        log(f"  {afirmacion['afirmacion']} {afirmacion['veredicto']}")
    return afirmaciones


def escribir_json_de_parada(parada: Parada) -> Path:
    """Escribe el JSON de parada y NO el informe completo.

    Para A9 y A10 el JSON lleva unicamente el residuo y las cadenas de la identidad, que es lo
    que exige el pre-registro. Para las demas paradas lleva el motivo y el detalle. En los dos
    casos lleva la frase de formulacion, porque va en cualquier mensaje de salida de P80.

    Args:
        parada: la excepcion de parada.

    Returns:
        La ruta del JSON de parada, ya publicada.

    Raises:
        SystemExit: si el temporal contiene algun byte mayor que 127.
    """
    cuerpo = {
        "familia": "P80_AUDITORIA_WORD2VEC",
        "experiment_id": "P80-01",
        "informe_completo": False,
        "motivo_de_la_parada": parada.motivo,
        "afirmacion_que_para": parada.afirmacion,
        "formulacion": FORMULACION,
        "nota": "una afirmacion con marca STOP ha fallado, luego NO se escribe el informe "
                "completo. Abajo va el detalle minimo que exige el pre-registro",
    }
    if parada.minimalista:
        cuerpo["detalle_minimo"] = parada.detalle
        cuerpo["que_prohibe_este_parada"] = (
            "el pre-registro prohibe emitir ningun otro numero en A9 y A10")
    else:
        cuerpo["detalle"] = parada.detalle
    REPORTS.mkdir(parents=True, exist_ok=True)
    temporal = ruta_temporal(JSON_INFORME)
    temporal.write_text(json.dumps(cuerpo, ensure_ascii=True, indent=2) + "\n", encoding="ascii")
    sucios = [nombre for nombre, ok in verificar_ascii_byte_a_byte([temporal]).items() if not ok]
    if sucios:
        raise SystemExit(f"STOP asercion ASCII: bytes no ASCII en {sucios}")
    publicar_atomico([(temporal, JSON_INFORME)])
    return JSON_INFORME


def verificar_pre_registro() -> dict:
    """Verifica el sha256 del pre-registro antes de ejecutar nada.

    Returns:
        Diccionario con la verificacion.
    """
    huella = huella_de_fichero(PREREGISTRO)
    coincide = huella["sha256"] == PREREGISTRO_SHA256
    return {
        "ruta": str(PREREGISTRO.relative_to(RAIZ)),
        "sha256": PREREGISTRO_SHA256,
        "sha256_recomputado": huella["sha256"],
        "coincide": coincide,
        "bytes": huella["bytes"],
        "commit": COMMIT_PREREGISTRO,
        "regla": "si el sha256 no coincide, P80 para y avisa a Carlos antes de seguir",
    }


def main() -> None:
    """Ejecuta P80 en orden y para en el primer STOP.

    Returns:
        None.
    """
    torch.set_num_threads(32)
    inicio = datetime.now(timezone.utc)
    cronometro: dict = {}

    log("=" * 78)
    log(f"P80-01 auditoria del entrenador Word2Vec | {FORMULACION}")
    log("=" * 78)

    pre = verificar_pre_registro()
    if not pre["coincide"]:
        log(f"STOP: el sha256 del pre-registro no coincide. esperado {PREREGISTRO_SHA256}, real "
            f"{pre['sha256_recomputado']}")
        raise SystemExit("STOP pre-registro: hash distinto del declarado")
    log(f"pre-registro verificado: {pre['sha256_recomputado']} (commit {COMMIT_PREREGISTRO})")

    huella_antes = huella_de_fichero(ARTEFACTO)
    if huella_antes["sha256"] != ARTEFACTO_SHA256:
        log(f"STOP: el sha256 del artefacto congelado no coincide al empezar. esperado "
            f"{ARTEFACTO_SHA256}, real {huella_antes['sha256']}")
        raise SystemExit("STOP artefacto congelado: hash distinto del declarado")
    log(f"artefacto congelado verificado ANTES de la primera lectura: {huella_antes['sha256']} "
        f"({huella_antes['bytes']} bytes)")

    contexto = contexto_de_modulos()
    afirmaciones: list = []
    try:
        marcas = datetime.now(timezone.utc)
        for nombre, funcion in (
                ("test1_loss", lambda: test1_loss(contexto)),
                ("test2_gradientes", lambda: test2_gradientes(contexto)),
                ("test3_scheduler", lambda: test3_scheduler(contexto)),
                ("test4_identidad_con_signo", lambda: test4_identidad_con_signo(contexto)),
                ("test5_artefacto", lambda: test5_artefacto(contexto, huella_antes)),
                ("test6_bigru", lambda: test6_bigru(contexto)),
                ("test7_fase_dl", lambda: test7_fase_dl())):
            ahora = datetime.now(timezone.utc)
            cronometro[nombre] = round((ahora - marcas).total_seconds(), 3)
            marcas = ahora
            afirmaciones.extend(funcion())
    except Parada as parada:
        huella_despues = huella_de_fichero(ARTEFACTO)
        if (huella_despues["sha256"] != huella_antes["sha256"]
                or huella_despues["bytes"] != huella_antes["bytes"]
                or huella_despues["mtime_ns"] != huella_antes["mtime_ns"]):
            log("STOP: el artefacto congelado CAMBIO entre la primera y la ultima lectura; el "
                "informe se marca invalido")
            raise SystemExit("STOP artefacto congelado modificado durante P80") from parada
        ruta = escribir_json_de_parada(parada)
        log(f"STOP {parada.afirmacion}: {parada.motivo}")
        log(f"JSON de parada escrito en {ruta.relative_to(RAIZ)}; el informe completo NO se "
            "escribe")
        log(f"artefacto congelado intacto: {huella_despues['sha256']}")
        log(FORMULACION)
        sys.exit(1)

    huella_despues = huella_de_fichero(ARTEFACTO)
    if (huella_despues["sha256"] != huella_antes["sha256"]
            or huella_despues["bytes"] != huella_antes["bytes"]
            or huella_despues["mtime_ns"] != huella_antes["mtime_ns"]):
        log("STOP: el artefacto congelado CAMBIO entre la primera y la ultima lectura")
        raise SystemExit("STOP artefacto congelado modificado durante P80")
    log(f"artefacto congelado verificado DESPUES de la ultima lectura: "
        f"{huella_despues['sha256']} ({huella_despues['bytes']} bytes)")

    marcas = datetime.now(timezone.utc)
    figuras_hechas = figuras(afirmaciones)
    cronometro["figuras"] = round((datetime.now(timezone.utc) - marcas).total_seconds(), 3)

    marcas = datetime.now(timezone.utc)
    REPORTS.mkdir(parents=True, exist_ok=True)
    json_temporal = ruta_temporal(JSON_INFORME)
    md_temporal = ruta_temporal(MD_INFORME)
    ascii_previo = verificar_ascii_byte_a_byte([Path(__file__)])
    informe = construir_json(afirmaciones, contexto, huella_antes, huella_despues, pre,
                             figuras_hechas, inicio, ascii_previo, cronometro)
    json_temporal.write_text(json.dumps(informe, ensure_ascii=True, indent=2) + "\n",
                             encoding="ascii")
    md_temporal.write_text("\n".join(escribir_markdown(afirmaciones, informe)) + "\n",
                           encoding="ascii")

    nombres_de_figura = sorted(figuras_hechas)
    sucios_de_nombre = [nombre for nombre in nombres_de_figura
                        if any(ord(caracter) > 127 for caracter in nombre)]
    if sucios_de_nombre:
        raise SystemExit(f"STOP asercion ASCII: nombre de figura no ASCII en {sucios_de_nombre}")
    ascii_final = verificar_ascii_byte_a_byte([Path(__file__), json_temporal, md_temporal])
    informe["asercion_ascii"]["observado"] = ascii_final
    informe["asercion_ascii"]["figuras"] = {
        nombre: {
            "nombre_en_ascii": not any(ord(caracter) > 127 for caracter in nombre),
            "bytes": (FIGURAS / nombre).stat().st_size,
            "firma_png_valida": (FIGURAS / nombre).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n",
        } for nombre in nombres_de_figura}
    informe["cronometro"]["informe"] = round(
        (datetime.now(timezone.utc) - marcas).total_seconds(), 3)
    json_temporal.write_text(json.dumps(informe, ensure_ascii=True, indent=2) + "\n",
                             encoding="ascii")
    sucios = [nombre for nombre, ok in ascii_final.items() if not ok]
    if sucios:
        raise SystemExit(f"STOP asercion ASCII: bytes no ASCII en {sucios}")
    log(f"asercion ASCII OK, escaneo byte a byte de {ascii_final}")
    log(f"figuras: {nombres_de_figura or 'ninguna'}; los nombres van en ASCII y los PNG son "
        "binarios por construccion")

    publicar_atomico([(json_temporal, JSON_INFORME), (md_temporal, MD_INFORME)])
    log(f"publicados por renombrado atomico: {JSON_INFORME.relative_to(RAIZ)} y "
        f"{MD_INFORME.relative_to(RAIZ)}")

    log("")
    log("=" * 78)
    for hallazgo, datos in informe["veredictos_por_hallazgo"].items():
        log(f"   {hallazgo}: {datos['estado']} ({', '.join(datos['afirmaciones'])})")
    log(f"   afirmaciones ejecutadas: {len(afirmaciones)} de 19")
    log(f"   informe {JSON_INFORME.relative_to(RAIZ)} | markdown {MD_INFORME.relative_to(RAIZ)}")
    log(f"   {inicio.isoformat(timespec='seconds')} -> "
        f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    log(FORMULACION)
    log("=" * 78)


if __name__ == "__main__":
    if "--humo" in sys.argv:
        superadas = ejecutar_pruebas_de_humo()
        sys.exit(0 if superadas >= MINIMO_COMPROBACIONES_HUMO else 1)
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        log(f"FALLO de P80: {type(exc).__name__}: {exc}")
        log(FORMULACION)
        sys.exit(1)
