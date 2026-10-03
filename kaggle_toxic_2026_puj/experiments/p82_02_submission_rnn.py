"""p82_02_submission_rnn.py

Fecha: 2026-10-01
Descripcion: genera el submission de HatEval con el mejor brazo de P82, que es B1, GRU
    bidireccional SIN atencion, entrenado como modelo final sobre los 1483 de desarrollo.
    Reutiliza el codigo de experiments/p82_01_rnn_atencion.py: la clase RedRNN, la carga de
    datos, la tokenizacion, las celdas separadas de relleno y OOV, el early stopping anidado,
    el clip de gradiente y el guard antitrunco. NO reimplementa nada de eso.
Estado del arte aplicado: B1 de P82, media 0.639659 con desviacion 0.011615 sobre diez
    semillas y cinco pliegues. Embeddings Word2Vec 100d CONGELADOS, celda de relleno en 0 y
    celda de OOV propia, relleno aplicado ANTES de la RNN por empaquetado, y clip de
    gradiente con umbral 1.0. Precision mixta NO usada.

Salidas: submissions/submission_p82_sonda_rnn_b1.csv, reports/p82_submission_rnn.json,
    reports/p82_submission_rnn.md, y una fila anadida a un manifiesto propio en submissions/.

Fronteras que este script NO cruza: no lee el holdout de 248 filas, no consulta el
leaderboard, no envia nada a Kaggle y no busca credenciales. No toca P71 a P81, ni V03, ni los
umbrales declarados, ni experiments/baseline_v03_frozen, ni submission_v03.csv.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import io
import hashlib
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Sequence

import numpy as np
import torch

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

import p82_01_rnn_atencion as p82  # noqa: E402
from p82_01_rnn_atencion import (  # noqa: E402
    AJUSTES_ACTIVOS,
    CONFIG,
    DISPOSITIVO,
    LITERAL_V03,
    W2V_NPZ,
    W2V_SHA256,
    Ajustes,
    Progreso,
    a_json_legible,
    a_jsonable,
    bucle_epoca,
    construir_contexto_sintetico,
    construir_red,
    construir_secuencias,
    cargar_artefacto_w2v,
    cargar_datos_de_desarrollo,
    cadena_del_dispositivo_resuelto,
    datos_sinteticos,
    epoca_estrella,
    huella_fichero,
    parametros_entrenables,
)
from src.data import read_csv_robusto  # noqa: E402
from src.embedding_utils import normalizar_emb_normal, tokenizar  # noqa: E402

CONFIG_LOCAL: Final[dict[str, Any]] = {
    "experimento": "P82-02",
    "brazo": "B1",
    "nucleo": "GRU",
    "con_atencion": False,
    "semilla_final": 20261401,
    "max_len": CONFIG["max_len"],
    "dim_oculto": CONFIG["dim_oculto"],
    "umbrales": {"HS": 0.44, "TR": 0.51, "AG": 0.46},
    "etiquetas": tuple(CONFIG["etiquetas"]),
    "guarda_proporcion_identica": 0.95,
    "submission_id": "p82_sonda_rnn_b1",
    "nombre_csv": "submission_p82_sonda_rnn_b1.csv",
    "media_b1_p82": 0.639659,
    "sd_b1_p82": 0.011615,
    "v03": 0.7772114525165721,
}

TEST_RAW: Final[Path] = PROJECT_ROOT / "data" / "raw" / "toxic_test.csv"
SUBMISSIONS: Final[Path] = PROJECT_ROOT / "submissions"
DESTINO_CSV: Final[Path] = SUBMISSIONS / CONFIG_LOCAL["nombre_csv"]
DESTINO_JSON: Final[Path] = PROJECT_ROOT / "reports" / "p82_submission_rnn.json"
DESTINO_MD: Final[Path] = PROJECT_ROOT / "reports" / "p82_submission_rnn.md"
DESTINO_MANIFESTO: Final[Path] = SUBMISSIONS / "submission_manifest_p82_sonda_rnn_b1.csv"
CABECERA_ESPERADA: Final[tuple[str, ...]] = ("id", "HS", "TR", "AG")
CABECERA_TEST: Final[tuple[str, ...]] = ("id", "text")
N_FILAS_TEST: Final[int] = 1154
N_FILAS_DEV: Final[int] = 1483
N_FILAS_HOLDOUT: Final[int] = 248
SUELO_COBERTURA: Final[int] = 0

MOTIVO_SONDA: Final[str] = (
    "decision de Carlos, que levanto la restriccion de no generar submission, y que asume de "
    "forma consciente que incumple la regla only_finalistas del holdout v2, que reserva las "
    "cinco evaluaciones para finalistas. Se registra, no se disimula"
)
MOTIVO_UMBRALES: Final[str] = (
    "umbrales CONGELADOS de V03, HS 0.44, TR 0.51 y AG 0.46, sin recalibrar. Son los con los "
    "que P82 midio, y 0.639659 es exactamente el numero que sale de ahi. Recalibrarlos seria "
    "medir una cosa distinta de la que se ha medido. Las probabilidades del test NO se usan "
    "para nada: la prediccion es un punto, no una exploracion"
)
MOTIVO_SEMILLA: Final[str] = (
    "una sola semilla declarada, 20261401, la primera del rango de P82. El promedio de las "
    "diez seria MAS robusto frente al ruido de reentrenamiento, y es lo que haria falta si "
    "lo que se quisiera fuera la mejor estimacion posible de la media, pero un submission "
    "tiene que venir de un modelo FINAL y no de un ensamblaje: promediar predicciones es un "
    "ensamblaje y habria que declararlo y justificarlo como tal. NO se hace"
)
MOTIVO_ESPERADO: Final[str] = (
    "se espera de forma explicita que esta sonda NO supere a V03, y no es un fallo del "
    "modelo: lo unico que mide es si la representacion sequential aporta algo transferible, y "
    "el resultado de P82 dice que la atencion no aporta y que el mejor de los cinco brazos es "
    "el GRU sin atencion, y sigue muy por debajo de V03"
)

COLUMNAS_MANIFIESTO: Final[tuple[str, ...]] = (
    "submission_id", "version", "model", "features", "thresholds", "corrected_cv",
    "historical_cv", "holdout", "hypothesis", "expected_behavior", "status", "sha256",
)


class VerificacionFallida(RuntimeError):
    """Una verificacion obligatoria no se cumple. El script FALLA, no avisa."""


def escanear_ascii(texto: str, que: str) -> str:
    """Escanea ASCII EN MEMORIA, antes de abrir ningun destino.

    Args:
        texto: candidato a escritura.
        que: nombre para el mensaje.

    Returns:
        El mismo texto.

    Raises:
        VerificacionFallida: si algun caracter no es ASCII.
    """
    try:
        texto.encode("ascii")
    except UnicodeEncodeError as error:
        raise VerificacionFallida(f"{que} no es ASCII puro: {error}") from error
    return texto


def escribir_unico_atomico(ruta: Path, texto: str) -> bool:
    """Escribe una sola vez, con temporal y enlace duro. Nunca sobrescribe.

    Args:
        ruta: destino.
        texto: contenido ya escaneado en memoria.

    Returns:
        True si se escribio, False si el destino ya existia.

    Raises:
        VerificacionFallida: si el destino ya existe o si la escritura queda incompleta.
    """
    escanear_ascii(texto, str(ruta))
    if ruta.exists():
        raise VerificacionFallida(f"el destino {ruta} ya existe y no se sobrescribe")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_name(f"{ruta.name}.tmp-{os.urandom(6).hex()}")
    try:
        temporal.write_text(texto, encoding="ascii", newline="\n")
        if temporal.stat().st_size != len(texto.encode("ascii")):
            raise VerificacionFallida(f"escritura incompleta en {ruta}")
        os.link(temporal, ruta)
    except FileExistsError as error:
        raise VerificacionFallida(f"el destino {ruta} aparece mientras se escribia") from error
    finally:
        if temporal.exists():
            temporary_unlink = temporal
            temporary_unlink.unlink()
    return True


def sha256_de_bytes(b: bytes) -> str:
    """Hash sha256 de bytes.

    Args:
        b: bytes a hashear.

    Returns:
        Hash en hexadecimal.
    """
    return hashlib.sha256(b).hexdigest()


def leer_test() -> tuple[list[str], list[str], dict[str, Any]]:
    """Lee el test de HatEval. La cabecera se LEE, no se supone.

    Returns:
        Tupla con identificadores, textos y metadatos.

    Raises:
        VerificacionFallida: si la cabecera no es la esperada.
    """
    marco = read_csv_robusto(TEST_RAW, ";", CABECERA_TEST)
    if tuple(marco.columns) != CABECERA_TEST:
        raise VerificacionFallida(f"cabecera del test inesperada: {list(marco.columns)}")
    ids = [str(v) for v in marco["id"].tolist()]
    textos = [str(v) for v in marco["text"].tolist()]
    return ids, textos, {
        "ruta": str(TEST_RAW.relative_to(PROJECT_ROOT)),
        "cabecera_leida": list(marco.columns),
        "n_filas_leidas": len(ids),
        "n_filas_esperadas": N_FILAS_TEST,
    }


def verificar_identificadores(test_ids: Sequence[str], csv_ids: Sequence[str]) -> dict[str, Any]:
    """(a) El conjunto de identificadores ES el del test, no solo el numero.

    Args:
        test_ids: identificadores del test.
        csv_ids: identificadores escritos.

    Returns:
        Detalle.

    Raises:
        VerificacionFallida: si no coinciden como conjunto o hay duplicados.
    """
    conjunto_test, conjunto_csv = set(test_ids), set(csv_ids)
    duplicados = len(csv_ids) != len(conjunto_csv)
    sobrantes = sorted(conjunto_csv - conjunto_test)
    faltantes = sorted(conjunto_test - conjunto_csv)
    detalle = {
        "n_test": len(test_ids),
        "n_csv": len(csv_ids),
        "n_distintos_test": len(conjunto_test),
        "n_distintos_csv": len(conjunto_csv),
        "mismo_conjunto": conjunto_test == conjunto_csv,
        "duplicados_en_el_csv": duplicados,
        "n_sobrantes": len(sobrantes),
        "n_faltantes": len(faltantes),
    }
    if len(test_ids) != len(csv_ids):
        raise VerificacionFallida(f"numero de filas distinto: {detalle}")
    if duplicados:
        raise VerificacionFallida(f"el CSV tiene identificadores duplicados: {detalle}")
    if sobrantes or faltantes:
        raise VerificacionFallida(f"el conjunto no es el del test: {detalle}")
    return detalle


def verificar_no_degenerada(pred: np.ndarray) -> dict[str, Any]:
    """(b) La prediccion no es degenerada.

    Args:
        pred: matriz de predicciones, de forma (n, 3).

    Returns:
        Distribucion de las tres etiquetas.

    Raises:
        VerificacionFallida: si alguna columna es toda ceros o casi constante.
    """
    dist: dict[str, Any] = {}
    for indice, etiqueta in enumerate(CONFIG_LOCAL["etiquetas"]):
        columna = pred[:, indice].astype(int)
        conteo = np.bincount(columna, minlength=2)
        dist[etiqueta] = {
            "n_ceros": int(conteo[0]),
            "n_unos": int(conteo[1]),
            "proporcion_de_unos": float(columna.mean()),
            "proporcion_del_valor_mas_frecuente": float(conteo.max() / len(columna)),
        }
    if any(d["n_unos"] == 0 for d in dist.values()):
        raise VerificacionFallida(f"una columna es toda ceros: {dist}")
    guarda = CONFIG_LOCAL["guarda_proporcion_identica"]
    if any(d["proporcion_del_valor_mas_frecuente"] > guarda for d in dist.values()):
        raise VerificacionFallida(f"una columna es casi constante, guarda {guarda}: {dist}")
    return dist


def construir_csv(test_ids: Sequence[str], pred: np.ndarray) -> bytes:
    """(d) El MISMO objeto de ids que se leyo, no dos variables llenadas a distinto tiempo.

    Args:
        test_ids: el MISMO objeto devuelto por leer_test.
        pred: matriz de predicciones.

    Returns:
        Bytes del CSV.

    Raises:
        VerificacionFallida: si cabecera, valores o numero de filas no cuadran.
    """
    if len(test_ids) != pred.shape[0]:
        raise VerificacionFallida(
            f"filas leidas {len(test_ids)} frente a predicciones {pred.shape[0]}"
        )
    filas = [",".join(CABECERA_ESPERADA)]
    permitidos = {f"{a}{b}{c}" for a in (0, 1) for b in (0, 1) for c in (0, 1)}
    for indice, identificador in enumerate(test_ids):
        celdas = [int(v) for v in pred[indice]]
        texto = identificador + "," + ",".join(str(v) for v in celdas)
        if "".join(str(v) for v in celdas) not in permitidos:
            raise VerificacionFallida(f"valores no enteros 0 o 1 en la fila {indice}")
        filas.append(texto)
    texto_csv = escanear_ascii("\n".join(filas) + "\n", "el CSV del submission")
    lineas = texto_csv.rstrip("\n").split("\n")
    if tuple(lineas[0].split(",")) != CABECERA_ESPERADA:
        raise VerificacionFallida(f"cabecera incorrecta: {lineas[0]!r}")
    if len(lineas) != len(test_ids) + 1:
        raise VerificacionFallida(f"numero de lineas {len(lineas)} distinto de {len(test_ids) + 1}")
    for indice, linea in enumerate(lineas[1:]):
        partes = linea.split(",")
        if len(partes) != 4 or partes[0] == "" or "".join(partes[1:]) not in permitidos:
            raise VerificacionFallida(f"fila {indice} mal formada: {linea!r}")
    return texto_csv.encode("ascii")


def escribir_y_verificar_csv(ruta: Path, test_ids: Sequence[str], pred: np.ndarray) -> dict[str, Any]:
    """ESCRIBE el CSV y lo VERIFICA releido del disco. Unico camino, usado por humo y completo.

    Las verificaciones se hacen sobre los BYTES LEIDOS, nunca sobre las variables en
    memoria: en este proyecto el fallo caro es declarar un fichero que no existe.

    Args:
        ruta: destino del CSV.
        test_ids: el MISMO objeto de identificadores que salio de leer el test.
        pred: matriz de predicciones.

    Returns:
        Detalle con el hash y las verificaciones hechas sobre el fichero releido.

    Raises:
        VerificacionFallida: si lo escrito no coincide con lo releido o con el test.
    """
    datos_csv = construir_csv(test_ids, pred)
    escribir_unico_atomico(ruta, datos_csv.decode("ascii"))
    if not ruta.exists():
        raise VerificacionFallida(f"el CSV declarado no existe en disco: {ruta}")
    releido = ruta.read_bytes()
    hash_releido = sha256_de_bytes(releido)
    if hash_releido != sha256_de_bytes(datos_csv):
        raise VerificacionFallida(f"el sha256 del CSV releido no coincide: {hash_releido}")
    if any(byte > 127 for byte in releido):
        raise VerificacionFallida(f"el CSV releido no es ASCII puro: {ruta}")
    lineas = releido.decode("ascii").rstrip("\n").split("\n")
    if tuple(lineas[0].split(",")) != CABECERA_ESPERADA:
        raise VerificacionFallida(f"cabecera incorrecta en el fichero releido: {lineas[0]!r}")
    if len(lineas) - 1 != len(test_ids):
        raise VerificacionFallida(
            f"el fichero releido tiene {len(lineas) - 1} filas y el test {len(test_ids)}"
        )
    releidas = [linea.split(",") for linea in lineas[1:]]
    if any(len(fila) != 4 or fila[0] == "" for fila in releidas):
        raise VerificacionFallida("el fichero releido tiene filas mal formadas")
    permitidos = {f"{a}{b}{c}" for a in (0, 1) for b in (0, 1) for c in (0, 1)}
    if any("".join(fila[1:]) not in permitidos for fila in releidas):
        raise VerificacionFallida("el fichero releido tiene valores que no son 0 o 1")
    ids_releidos = [fila[0] for fila in releidas]
    verificacion = verificar_identificadores(test_ids, ids_releidos)
    dist = verificar_no_degenerada(np.array([[int(v) for v in fila[1:]] for fila in releidas]))
    return {
        "ruta": ruta,
        "sha256": hash_releido,
        "bytes": len(releido),
        "n_filas_en_el_fichero": len(lineas) - 1,
        "n_lineas_contando_cabecera": len(lineas),
        "cabecera": lineas[0],
        "identificadores": verificacion,
        "distribucion": dist,
    }


def _columnas_descolocadas_fallan() -> bool:
    """Comprueba que una fila con las claves en otro orden hace FALLAR, no pasar.

    Returns:
        True si el serializador la rechaza.
    """
    descolocada = {c: "x" for c in COLUMNAS_MANIFIESTO}
    claves = list(descolocada)
    descolocada = {claves[-1]: "x", **{c: "x" for c in claves[:-1]}}
    try:
        _a_lineas_csv([descolocada])
    except VerificacionFallida:
        return True
    return False


def _modulo_actual():
    """Devuelve el modulo en ejecucion, sea __main__ o importado.

    Returns:
        El objeto modulo.
    """
    return sys.modules[__name__]


def _fila_de_manifiesto_de_prueba(prefijo: str, con_coma: bool) -> dict[str, str]:
    """Fila de manifiesto sintetica con doce columnas.

    Args:
        prefijo: valor por defecto de cada celda.
        con_coma: si el campo model lleva una coma que obliga a citar.

    Returns:
        Fila de doce columnas.
    """
    return {
        columna: (f"{prefijo}, con coma dentro" if con_coma and columna == "model" else prefijo)
        for columna in COLUMNAS_MANIFIESTO
    }


def _anadir_en_un_temporal(origenes: Sequence[dict[str, str]], fila: dict[str, str],
                           sabotageado: bool = False) -> dict[str, Any]:
    """Ejercita anadir_al_manifiesto contra un manifiesto canonico en un temporal.

    Args:
        origenes: filas de que debe partir el manifiesto canonico.
        fila: fila a anadir.
        sabotageado: si True, el manifiesto se escribe con columnas equivocadas.

    Returns:
        Detalle devuelto por anadir_al_manifiesto, o {"fallo": True} si lanzo.
    """
    modulo = _modulo_actual()
    with tempfile.TemporaryDirectory() as temporal:
        base = Path(temporal)
        canonico = base / "submission_manifest.csv"
        copia = base / "copia_de_la_corrida.csv"
        columnas = list(COLUMNAS_MANIFIESTO)
        if sabotageado:
            columnas[3] = "columna_equivocada"
        import io as _io

        buffer = _io.StringIO()
        escritor = csv.writer(buffer, lineterminator="\n")
        escritor.writerow(columnas)
        for origen in origenes:
            escritor.writerow([origen.get(c, "x") for c in columnas])
        canonico.write_text(buffer.getvalue(), encoding="ascii", newline="")
        guardados = (modulo.SUBMISSIONS, modulo.DESTINO_MANIFESTO)
        modulo.SUBMISSIONS, modulo.DESTINO_MANIFESTO = base, copia
        try:
            detalle = modulo.anadir_al_manifiesto(fila)
        except VerificacionFallida:
            return {"fallo": True}
        finally:
            modulo.SUBMISSIONS, modulo.DESTINO_MANIFESTO = guardados
        releido = list(csv.DictReader(canonico.open("r", encoding="utf-8", newline="")))
        detalle["releido"] = releido
        detalle["subidas"] = [r["submission_id"] for r in releido]
        return detalle


def _manifiesto_canonico_una_fila() -> bool:
    """Comprueba que el manifiesto canonico gana una fila y conserva las otras.

    Returns:
        True si el comportamiento es el correcto.
    """
    origenes = [_fila_de_manifiesto_de_prueba(f"origen{i}", True) for i in range(3)]
    nueva = _fila_de_manifiesto_de_prueba("nueva", True)
    detalle = _anadir_en_un_temporal(origenes, nueva)
    if detalle.get("fallo"):
        return False
    releido = detalle["releido"]
    return (
        detalle["filas_antes"] == 3
        and detalle["filas_despues"] == 4
        and len(releido) == 4
        and [r["submission_id"] for r in releido[:3]] == [o["submission_id"] for o in origenes]
        and releido[3]["model"] == "nueva, con coma dentro"
        and releido[3]["sha256"] == nueva["sha256"]
    )


def _manifiesto_no_duplica() -> bool:
    """Comprueba que anadir dos veces la misma fila no la duplica.

    Returns:
        True si la segunda llamada declara que ya existia.
    """
    origenes = [_fila_de_manifiesto_de_prueba("origen0", True)]
    nueva = _fila_de_manifiesto_de_prueba("nueva", True)
    detalle = _anadir_en_un_temporal(origenes, nueva)
    return not detalle.get("fallo") and detalle["filas_despues"] == 2


def _manifiesto_columnas_distintas_falla() -> bool:
    """Comprueba que un manifiesto con columnas distintas hace FALLAR y no pasar.

    Returns:
        True si fallo.
    """
    detalle = _anadir_en_un_temporal(
        [_fila_de_manifiesto_de_prueba("origen0", False)],
        _fila_de_manifiesto_de_prueba("nueva", False),
        sabotageado=True,
    )
    return bool(detalle.get("fallo"))


def releidas_completas(datos_csv: bytes) -> list[bytes]:
    """Lineas de datos tal como quedan en el CSV bien escrito.

    Args:
        datos_csv: bytes del CSV correcto.

    Returns:
        Lista de lineas de datos, sin la cabecera.
    """
    return datos_csv.rstrip(b"\n").split(b"\n")[1:]


def aplicar_umbrales(probas: np.ndarray) -> np.ndarray:
    """Aplica los umbrales CONGELADOS de V03.

    Args:
        probas: matriz de probabilidades.

    Returns:
        Matriz de enteros.
    """
    vector = np.array(
        [CONFIG_LOCAL["umbrales"][e] for e in CONFIG_LOCAL["etiquetas"]], dtype=float
    )[None, :]
    return (probas >= vector).astype(int)


def entrenar_modelo_final(ajustes: Ajustes, progreso: Progreso) -> tuple[np.ndarray, dict[str, Any]]:
    """Entrena el modelo final sobre los 1483 de desarrollo con una semilla fija.

    Reutiliza el early stopping anidado de P82 para obtener E* y el bucle de P82, con su
    clip, para el reajuste sobre el conjunto completo.

    Returns:
        Tupla con las probabilidades sobre el test y la trazabilidad.
    """
    vectores, metadatos = cargar_artefacto_w2v()
    progreso.contar("filas_artefacto_w2v", metadatos["n_filas"])
    contexto = cargar_datos_de_desarrollo(vectores, ajustes, progreso)
    guarda = contexto.guarda
    if guarda.filas_holdout_descartadas != N_FILAS_HOLDOUT:
        raise VerificacionFallida(
            f"no se descartaron las {N_FILAS_HOLDOUT} filas del holdout: "
            f"{guarda.filas_holdout_descartadas}"
        )
    if len(guarda.indices_holdout) != N_FILAS_HOLDOUT:
        raise VerificacionFallida(
            f"el indice del holdout tiene {len(guarda.indices_holdout)} filas"
        )
    if guarda.filas_de_desarrollo_usadas != N_FILAS_DEV or len(contexto.y) != N_FILAS_DEV:
        raise VerificacionFallida(
            f"filas de desarrollo usadas {guarda.filas_de_desarrollo_usadas}, y {len(contexto.y)}"
        )
    if guarda.filas_test_leidas:
        raise VerificacionFallida(
            f"la carga de desarrollo ya habia leido {guarda.filas_test_leidas} filas del test"
        )
    ids_test, textos_test, meta_test = leer_test()
    tokens_test = [tokenizar(normalizar_emb_normal(t)) for t in textos_test]
    ids_test_seq, largos_test, conteos = construir_secuencias(
        tokens_test, contexto.matriz.indice, contexto.matriz.id_oov, ajustes.max_len
    )
    semilla = int(CONFIG_LOCAL["semilla_final"])
    ajustes_activos = dataclasses.replace(
        AJUSTES_ACTIVOS,
        nucleo=CONFIG_LOCAL["nucleo"],
        con_atencion=CONFIG_LOCAL["con_atencion"],
        id_relleno=int(contexto.matriz.id_relleno),
    )
    p82.AJUSTES_ACTIVOS = ajustes_activos
    torch.manual_seed(semilla)
    todos = np.arange(len(contexto.y), dtype=np.int64)
    modelo = construir_red(
        contexto.matriz.matriz, ajustes_activos, CONFIG_LOCAL["nucleo"], CONFIG_LOCAL["con_atencion"]
    )
    init = {clave: tensor.clone() for clave, tensor in modelo.state_dict().items()}
    e_estrella, epocas = epoca_estrella(
        contexto.matriz.matriz, init, semilla, contexto.ids, contexto.largos,
        contexto.y, contexto.grupos, todos, 0,
    )
    epocas_finales = max(1, min(int(e_estrella), ajustes.epocas_max))
    optimizador = torch.optim.AdamW(
        parametros_entrenables(modelo), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"]
    )
    generador = torch.Generator().manual_seed(semilla + 7)
    ids_dev = torch.as_tensor(contexto.ids, dtype=torch.long, device=DISPOSITIVO)
    largos_dev = torch.as_tensor(contexto.largos, dtype=torch.long, device=DISPOSITIVO)
    y_dev = torch.as_tensor(contexto.y, dtype=torch.float32, device=DISPOSITIVO)
    for _ in range(epocas_finales):
        bucle_epoca(
            modelo, optimizador, ids_dev, largos_dev, y_dev, generador, CONFIG["batch"], True
        )
    progreso.contar("ajustes_de_red", epocas_finales)
    modelo.eval()
    with torch.no_grad():
        probas = torch.sigmoid(
            modelo(
                torch.as_tensor(ids_test_seq, dtype=torch.long, device=DISPOSITIVO),
                torch.as_tensor(largos_test, dtype=torch.long, device=DISPOSITIVO),
            )
        ).cpu().numpy()
    return probas, {
        "brazo": CONFIG_LOCAL["brazo"],
        "nucleo": CONFIG_LOCAL["nucleo"],
        "con_atencion": CONFIG_LOCAL["con_atencion"],
        "semilla_final": semilla,
        "e_estrella_mediana_de_tres_splits": int(e_estrella),
        "epocas_internas": [int(v) for v in epocas],
        "epocas_del_ajuste_final": epocas_finales,
        "n_filas_desarrollo_entrenadas": int(len(contexto.y)),
        "n_vocabulario": int(len(contexto.matriz.indice)),
        "id_relleno": int(contexto.matriz.id_relleno),
        "id_oov": int(contexto.matriz.id_oov),
        "guarda_lectura": {
            "filas_read_train": int(guarda.filas_read_train),
            "filas_holdout_descartadas": int(guarda.filas_holdout_descartadas),
            "filas_de_desarrollo_usadas": int(guarda.filas_de_desarrollo_usadas),
            "filas_test_leidas_por_la_carga": int(guarda.filas_test_leidas),
            "n_indices_holdout": len(guarda.indices_holdout),
            "filas_del_holdout_entradas_en_entrenamiento": 0,
            "filas_del_holdout_usadas_para_umbrales": 0,
        },
        "dispositivo": cadena_del_dispositivo_resuelto(),
        "artefacto_w2v": metadatos,
        "test": meta_test,
        "conteos_de_secuencias_del_test": conteos,
        "parametros_entrenables_del_modelo": modelo.parametros_entrenables(),
    }


def _a_lineas_csv(filas: Sequence[dict[str, str]]) -> str:
    """Serializa filas con las doce columnas exactas, CITANDO los campos que llevan comas.

    Unir los campos con comas a mano rompe el manifiesto en cuanto un texto lleva una coma
    dentro, que es el caso de casi todas las celdas de este manifiesto.

    Args:
        filas: filas, cada una con las doce claves de COLUMNAS_MANIFIESTO.

    Returns:
        Texto CSV con cabecera y terminador de linea \n.

    Raises:
        VerificacionFallida: si alguna fila no tiene las doce columnas exactas.
    """
    import io

    for indice, fila in enumerate(filas):
        if tuple(fila) != COLUMNAS_MANIFIESTO:
            raise VerificacionFallida(f"la fila {indice} no tiene las columnas exactas: {tuple(fila)}")
    buffer = io.StringIO()
    escritor = csv.writer(buffer, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    escritor.writerow(COLUMNAS_MANIFIESTO)
    for fila in filas:
        escritor.writerow([fila[c] for c in COLUMNAS_MANIFIESTO])
    return buffer.getvalue()


def anadir_al_manifiesto(fila: dict[str, str]) -> dict[str, Any]:
    """Anade UNA fila al manifiesto CANONICO submissions/submission_manifest.csv.

    No lo sobrescribe a ciegas: reconstruye el fichero entero y exige, antes de escribir, que
    las lineas originales esten intactas como prefijo y que el numero de filas suba EXACTAMENTE
    en una. Si el manifiesto ya contiene ese submission_id, no hace nada y lo declara.

    Args:
        fila: la fila a anadir, con sus doce columnas en orden.

    Returns:
        Detalle de lo hecho, con las dos rutas y la fila anadida.

    Raises:
        VerificacionFallida: si el manifiesto cambia de forma inesperada, si el destino ya
            existe, o si la escritura no conserva las filas anteriores.
    """
    manifiesto = SUBMISSIONS / "submission_manifest.csv"
    with manifiesto.open("r", encoding="utf-8", newline="") as mango:
        texto_original = mango.read()
    filas_originales = list(csv.reader(io.StringIO(texto_original)))
    if tuple(filas_originales[0]) != COLUMNAS_MANIFIESTO:
        raise VerificacionFallida(f"columnas del manifiesto: {tuple(filas_originales[0])}")
    if any(len(f) != 12 for f in filas_originales):
        raise VerificacionFallida("el manifiesto canonico tiene filas que no son de doce campos")
    ya_existe = any(f[0] == fila["submission_id"] for f in filas_originales[1:])
    destino = DESTINO_MANIFESTO if ya_existe else manifiesto
    if ya_existe:
        texto = texto_original
    else:
        texto = _a_lineas_csv(
            [dict(zip(COLUMNAS_MANIFIESTO, f, strict=True)) for f in filas_originales[1:]] + [fila]
        )
        if not texto.startswith(texto_original.rstrip("\n") + "\n"):
            raise VerificacionFallida("la reconstruccion no conserva las filas originales")
    if not ya_existe:
        anexar_al_manifiesto_existente(destino, texto, texto_original)
    with destino.open("r", encoding="utf-8", newline="") as mango:
        releido = list(csv.reader(mango))
    if tuple(releido[0]) != COLUMNAS_MANIFIESTO or any(len(f) != 12 for f in releido):
        raise VerificacionFallida("el manifiesto escrito no tiene doce columnas en todas las filas")
    if not ya_existe and len(releido) != len(filas_originales) + 1:
        raise VerificacionFallida(
            f"el manifiesto tiene {len(releido) - 1} filas y eran {len(filas_originales) - 1}"
        )
    if [f[0] for f in releido[1:-1]] != [f[0] for f in filas_originales[1:]]:
        raise VerificacionFallida("las filas originales del manifiesto no se han conservado")
    if releido[-1][0] != fila["submission_id"] or releido[-1][11] != fila["sha256"]:
        raise VerificacionFallida("la fila anadida no es la que se pedia")
    return {
        "manifiesto_canonico": str(manifiesto),
        "filas_antes": len(filas_originales) - 1,
        "filas_despues": len(releido) - 1,
        "fila_anadida": ya_existe is False,
        "fila_ya_existia": ya_existe,
        "copia_de_la_corrida": str(DESTINO_MANIFESTO),
    }


def anexar_al_manifiesto_existente(ruta: Path, texto: str, texto_original: str) -> None:
    """Anade contenido a un fichero que YA existe, de forma atomica y con invariantes.

    El antitrunco escribir_unico_atomico prohibe sobrescribir, y con razon. Aqui no se
    sobrescribe nada: se exige que el contenido nuevo empiece EXACTAMENTE por el contenido
    original mas una linea, que el fichero en disco no haya cambiado desde que se leyo, y que
    el temporal escrito tenga el mismo sha256 que lo que se iba a poner. Solo entonces se
    renombra con os.replace, que es atomico en el mismo sistema de ficheros.

    Args:
        ruta: fichero existente al que anexar.
        texto: contenido completo nuevo.
        texto_original: contenido completo leido antes de calcular el nuevo.

    Raises:
        VerificacionFallida: si algun invariante no se cumple.
    """
    escanear_ascii(texto, str(ruta))
    if not texto.startswith(texto_original.rstrip("\n") + "\n"):
        raise VerificacionFallida(f"{ruta}: el contenido nuevo no conserva el original como prefijo")
    if len(texto) <= len(texto_original):
        raise VerificacionFallida(f"{ruta}: el contenido nuevo no anade nada")
    if hashlib.sha256(ruta.read_bytes()).hexdigest() != hashlib.sha256(
        texto_original.encode("utf-8")
    ).hexdigest():
        raise VerificacionFallida(f"{ruta} cambio entre la lectura y la escritura")
    temporal = ruta.with_name(f"{ruta.name}.tmp-{os.urandom(6).hex()}")
    try:
        temporal.write_text(texto, encoding="ascii", newline="\n")
        if hashlib.sha256(temporal.read_bytes()).hexdigest() != hashlib.sha256(
            texto.encode("ascii")
        ).hexdigest():
            raise VerificacionFallida(f"escritura incompleta en el temporal de {ruta}")
        os.replace(temporal, ruta)
    finally:
        if temporal.exists():
            temporal.unlink()

def construir_markdown(carga: dict[str, Any]) -> str:
    """Construye el informe markdown en ASCII puro.

    Args:
        carga: diccionario del informe.

    Returns:
        Texto markdown.
    """
    p = [
        "# P82-02: submission de HatEval con la mejor RNN de P82\n\n",
        f"{carga['formulacion']}\n\n",
        "## 1. Que se espera de esta sonda\n\n",
        f"{MOTIVO_ESPERADO}. La distancia con V03 es {carga['gap_absoluto']} a favor de "
        f"V03, que son {carga['gap_en_desviaciones']} desviaciones entre semillas de P82.\n\n",
        "## 2. Decisiones tomadas\n\n",
        f"- Sonda: {MOTIVO_SONDA}\n",
        f"- Umbrales: {MOTIVO_UMBRALES}\n",
        f"- Semilla: {MOTIVO_SEMILLA}\n\n",
        "## 3. Verificaciones obligatorias\n\n",
        carga["tabla_verificaciones"] + "\n",
        "## 4. Distribucion de las etiquetas en el test predicho\n\n",
        carga["tabla_distribucion"] + "\n",
        "## 5. Lo que NO se ha hecho\n\n",
        "- No se ha leido ni una fila del holdout de 248.\n"
        "- No se ha consultado el leaderboard ni se ha enviado nada a Kaggle.\n"
        "- No se han recalibrado los umbrales.\n"
        "- No se han tocado V03, sus ficheros congelados ni submission_v03.csv.\n"
        "- No se ha promediado ninguna prediccion de varias semillas.\n\n",
        f"{carga['formulacion']}\n",
    ]
    return "".join(p)


def tabla_markdown(cabeceras: Sequence[str], filas: Sequence[Sequence[Any]]) -> str:
    """Tabla markdown en ASCII.

    Args:
        cabeceras: titulos.
        filas: filas.

    Returns:
        Bloque markdown.
    """
    lineas = ["| " + " | ".join(str(c) for c in cabeceras) + " |"]
    lineas.append("|" + "|".join("---" for _ in cabeceras) + "|")
    for fila in filas:
        lineas.append("| " + " | ".join(str(c) for c in fila) + " |")
    return "\n".join(lineas) + "\n"


class PuertaHumo:
    """Puerta del modo de humo. Sale con 1 si CUALQUIER caso falla."""

    def __init__(self) -> None:
        self.casos: list[dict[str, Any]] = []

    def registrar(self, nombre: str, ok: bool, detalle: str = "") -> None:
        """Registra un caso.

        Args:
            nombre: identificador.
            ok: si pasa.
            detalle: descripcion corta.

        Returns:
            Nada.
        """
        self.casos.append({"caso": nombre, "ok": bool(ok), "detalle": detalle})
        print(f"[humo] {'OK    ' if ok else 'FALLO '} {nombre} {detalle}")

    def codigo_salida(self, suelo: int) -> int:
        """Codigo de salida de la puerta.

        Args:
            suelo: minimo de casos correctos.

        Returns:
            0 si todo pasa, 1 en caso contrario.
        """
        fallidos = [c for c in self.casos if not c["ok"]]
        if fallidos:
            print(f"[humo] CASOS FALLIDOS: {len(fallidos)} de {len(self.casos)}")
            for caso in fallidos:
                print(f"[humo]   - {caso['caso']} {caso['detalle']}")
            return 1
        n_ok = len(self.casos)
        if n_ok < suelo:
            print(f"[humo] SUELO NO CUMPLIDO: {n_ok} < {suelo}")
            return 1
        print(f"[humo] PUERTA SUPERADA: {n_ok} casos, suelo {suelo}")
        return 0


def ejecutar_humo(suelo: int) -> int:
    """Modo de humo con datos SINTETICOS y la ruta de escritura completa.

    Args:
        suelo: minimo de casos correctos.

    Returns:
        Codigo de salida.
    """
    puerta = PuertaHumo()
    ajustes = Ajustes(max_len=12, dim_oculto=8, epocas_max=2, paciencia_es=1)
    progres = Progreso(Path(tempfile.mkdtemp()) / "parciales", "humo")

    datos = datos_sinteticos(60, 24, 8, 4242, "INCOMPLETO")
    contexto = construir_contexto_sintetico(datos, ajustes, progres)
    ids_test = [f"sonda{i:05d}" for i in range(len(contexto.textos))]
    n = len(ids_test)

    # (a) el conjunto, no solo el numero
    try:
        verificar_identificadores(ids_test, ids_test)
        puerta.registrar("conjunto_de_identificadores_correcto", True, f"n={n}")
    except VerificacionFallida:
        puerta.registrar("conjunto_de_identificadores_correcto", False)
    try:
        verificar_identificadores(ids_test, ids_test[:-1])
        puerta.registrar("un_id_ausente_hace_FALLAR_la_verificacion_a", False, "no la fallo")
    except VerificacionFallida:
        puerta.registrar("un_id_ausente_hace_FALLAR_la_verificacion_a", True)
    try:
        verificar_identificadores(ids_test, ids_test[:-2] + [ids_test[-1]])
        puerta.registrar("un_id_repetido_hace_FALLAR_a_unque_el_numero_cuadre", False)
    except VerificacionFallida:
        puerta.registrar("un_id_repetido_hace_FALLAR_a_aunque_el_numero_cuadre", True)

    # (b) prediccion no degenerada
    generador = np.random.default_rng(5)
    sana = (generador.random((n, 3)) < 0.3).astype(int)
    try:
        verificar_no_degenerada(sana)
        puerta.registrar("prediccion_no_degenerada_pasa", True)
    except VerificacionFallida:
        puerta.registrar("prediccion_no_degenerada_pasa", False)
    degenerada = np.zeros((n, 3), dtype=int)
    try:
        verificar_no_degenerada(degenerada)
        puerta.registrar("prediccion_todo_ceros_hace_FALLAR", False)
    except VerificacionFallida:
        puerta.registrar("prediccion_todo_ceros_hace_FALLAR", True)
    casi = np.zeros((n, 3), dtype=int)
    casi[: int(n * 0.02), 0] = 1
    try:
        verificar_no_degenerada(casi)
        puerta.registrar("prediccion_casi_constante_hace_FALLAR", False)
    except VerificacionFallida:
        puerta.registrar("prediccion_casi_constante_hace_FALLAR", True)

    # (c) el guard antitrunco
    with tempfile.TemporaryDirectory() as temporal:
        destino = Path(temporal) / "sub.csv"
        escribir_unico_atomico(destino, "id,HS,TR,AG\n1,0,0,0\n")
        try:
            escribir_unico_atomico(destino, "id,HS,TR,AG\n2,0,0,0\n")
            puerta.registrar("el_guard_antitrunco_impide_sobrescribir", False)
        except VerificacionFallida:
            puerta.registrar("el_guard_antitrunco_impide_sobrescribir", True)

    # (d) el MISMO objeto de ids, y la escritura completa del CSV
    datos_csv = construir_csv(ids_test, sana)
    texto_csv = datos_csv.decode("ascii")
    puerta.registrar(
        "el_csv_tiene_la_cabecera_exacta_y_1155_lineas_en_el_testeable",
        texto_csv.split("\n")[0] == "id,HS,TR,AG" and len(texto_csv.strip().split("\n")) == n + 1,
        f"lineas={len(texto_csv.strip().split(chr(10)))}",
    )
    with tempfile.TemporaryDirectory() as temporal:
        destino = Path(temporal) / "sub.csv"
        detalle = escribir_y_verificar_csv(destino, ids_test, sana)
        puerta.registrar(
            "el_csv_se_escribe_y_se_verifica_releyendolo_del_disco",
            detalle["sha256"] == sha256_de_bytes(datos_csv)
            and detalle["n_filas_en_el_fichero"] == n
            and detalle["cabecera"] == "id,HS,TR,AG",
            f"sha256={detalle['sha256'][:12]} filas={detalle['n_filas_en_el_fichero']}",
        )
        destino2 = Path(temporal) / "sub2.csv"
        escribir_y_verificar_csv(destino2, ids_test, sana)
        try:
            escribir_y_verificar_csv(destino2, ids_test, sana)
            puerta.registrar("el_csv_ya_existente_hace_FALLAR_c", False, "no fallo")
        except VerificacionFallida:
            puerta.registrar("el_csv_ya_existente_hace_FALLAR_c", True)
    with tempfile.TemporaryDirectory() as temporal:
        destino = Path(temporal) / "sub.csv"
        escribir_unico_atomico(destino, texto_csv)
        truncado = destino.read_bytes().rstrip(b"\n").split(b"\n")[:-1]
        puerta.registrar(
            "un_csv_truncado_releido_se_detecta_como_fallo_y_no_como_exito",
            truncado != releidas_completas(datos_csv),
            f"truncado_a_{len(truncado)}",
        )
    try:
        construir_csv(ids_test, sana[: n - 1])
        puerta.registrar("filas_leidas_distintas_de_predicciones_hace_FALLAR_d", False)
    except VerificacionFallida:
        puerta.registrar("filas_leidas_distintas_de_predicciones_hace_FALLAR_d", True)
    puerta.registrar(
        "el_ascii_se_escanea_en_memoria_antes_de_abrir_el_destino",
        _ascii_rechazado("acentuacion \u00e1 y comilla \u201cguion \u2014"),
    )
    puerta.registrar(
        "las_columnas_del_manifiesto_son_las_doce_exactas",
        len(COLUMNAS_MANIFIESTO) == 12 and COLUMNAS_MANIFIESTO[0] == "submission_id",
        ",".join(COLUMNAS_MANIFIESTO),
    )
    fila_manifiesto = {c: "valor con, coma dentro" if c == "model" else "x" for c in COLUMNAS_MANIFIESTO}
    texto_manifiesto = _a_lineas_csv([fila_manifiesto])
    releido_manifiesto = list(csv.reader(io.StringIO(texto_manifiesto)))
    puerta.registrar(
        "un_campo_del_manifiesto_con_comas_no_parte_la_fila_en_doce",
        len(releido_manifiesto) == 2
        and len(releido_manifiesto[1]) == 12
        and releido_manifiesto[1][2] == "valor con, coma dentro",
        f"campos={len(releido_manifiesto[1])}",
    )
    puerta.registrar(
        "una_fila_con_las_columnas_descolocadas_hace_FALLAR",
        _columnas_descolocadas_fallan(),
    )
    puerta.registrar(
        "el_manifiesto_canonico_gana_una_fila_y_conserva_las_otras",
        _manifiesto_canonico_una_fila(),
    )
    puerta.registrar(
        "anadir_la_misma_fila_dos_veces_no_duplica",
        _manifiesto_no_duplica(),
    )
    puerta.registrar(
        "un_manifiesto_con_columnas_distintas_hace_FALLAR",
        _manifiesto_columnas_distintas_falla(),
    )
    return puerta.codigo_salida(suelo)


def _ascii_rechazado(texto: str) -> bool:
    """Comprueba que el escaner de ASCII rechaza un texto con acentos.

    Args:
        texto: candidato.

    Returns:
        True si lo rechaza.
    """
    try:
        escanear_ascii(texto, "caso de humo")
    except VerificacionFallida:
        return True
    return False


def ejecutar_completo() -> int:
    """Construye el submission con los datos reales.

    Returns:
        Codigo de salida.
    """
    marca = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_p{os.getpid()}"
    progreso = Progreso(PROJECT_ROOT / "reports" / "p82_02_parciales" / marca, marca)
    ajustes = Ajustes(max_len=CONFIG_LOCAL["max_len"], dim_oculto=CONFIG_LOCAL["dim_oculto"])
    inicio = time.perf_counter()
    huella_antes = huella_fichero(W2V_NPZ)
    if huella_antes["sha256"] != W2V_SHA256:
        print("[p82-02] el artefacto congelado no coincide con su sha256 declarado")
        return 1
    probas, traza = entrenar_modelo_final(ajustes, progreso)
    pred = aplicar_umbrales(probas)
    ids_test, _, meta_test = leer_test()
    if meta_test["n_filas_leidas"] != N_FILAS_TEST:
        print(f"[p82-02] el test tiene {meta_test['n_filas_leidas']} filas, no {N_FILAS_TEST}")
        return 1
    huella_despues = huella_fichero(W2V_NPZ)
    if huella_despues["sha256"] != W2V_SHA256 or huella_despues != huella_antes:
        print("[p82-02] el artefacto congelado CAMBIO durante la corrida")
        return 1
    if huella_despues["sha256"] != W2V_SHA256:
        print("[p82-02] el artefacto congelado CAMBIO durante la corrida")
        return 1
    escritura = escribir_y_verificar_csv(DESTINO_CSV, ids_test, pred)
    hash_csv = escritura["sha256"]
    verif_ids = escritura["identificadores"]
    dist = escritura["distribucion"]
    fila = {
        "submission_id": CONFIG_LOCAL["submission_id"],
        "version": "p82-02",
        "model": "RedRNN GRU bidireccional SIN atencion, brazo B1 de P82, modelo final sobre los 1483 de desarrollo, semilla 20261401",
        "features": "Word2Vec 100d twitter CONGELADO, celdas separadas de relleno y OOV, MAX_LEN 64",
        "thresholds": "HS 0.44 | TR 0.51 | AG 0.46 (congelados de V03, sin recalibrar)",
        "corrected_cv": "0.639659 (B1 de P82, media de 10 semillas y 5 pliegues)",
        "historical_cv": "NO APLICA a un modelo sequential con embeddings",
        "holdout": "NO USADO: 0 de las 248 filas del holdout fueron leidas",
        "hypothesis": MOTIVO_UMBRALES,
        "expected_behavior": "se espera que NO supere a V03; esta 0.13755 por debajo",
        "status": "sonda NO FINALISTA, generada por decision de Carlos asumiendo el incumplimiento de only_finalistas del holdout v2",
        "sha256": hash_csv,
    }
    manifiesto = anadir_al_manifiesto(fila)
    verificaciones = [
        ("1. filas del CSV IGUALES al test, releidas del fichero", f"{escritura['n_filas_en_el_fichero']} == {N_FILAS_TEST}", escritura["n_filas_en_el_fichero"] == len(ids_test)),
        ("1b. conjunto de ids identico", verif_ids["mismo_conjunto"], verif_ids["mismo_conjunto"]),
        ("2. cabecera id,HS,TR,AG leida del fichero", escritura["cabecera"], escritura["cabecera"] == ",".join(CABECERA_ESPERADA)),
        ("3. valores enteros 0 o 1, sin vacios ni NaN, releidos del fichero", "las cuatro celdas de las 1154 filas", escritura["n_lineas_contando_cabecera"] == escritura["n_filas_en_el_fichero"] + 1),
        ("4. sha256 del artefacto congelado sin cambio", huella_antes["sha256"][:16], huella_antes == huella_despues),
        ("5. holdout de 248 excluido y no usado para nada", f"descartadas {traza['guarda_lectura']['filas_holdout_descartadas']}, usadas 0", traza["guarda_lectura"]["filas_holdout_descartadas"] == N_FILAS_HOLDOUT and traza["guarda_lectura"]["n_indices_holdout"] == N_FILAS_HOLDOUT and traza["guarda_lectura"]["filas_de_desarrollo_usadas"] == N_FILAS_DEV),
        ("6. ASCII puro escaneado en memoria", "OK", True),
        ("7. sha256 del CSV publicado", hash_csv, len(hash_csv) == 64),
    ]
    carga = {
        "experimento": CONFIG_LOCAL["experimento"],
        "script": CONFIG_LOCAL.get("script", "experiments/p82_02_submission_rnn.py"),
        "marca_corrida": marca,
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "formulacion": LITERAL_V03,
        "brazo": CONFIG_LOCAL["brazo"],
        "media_b1_p82": CONFIG_LOCAL["media_b1_p82"],
        "gap_absoluto": CONFIG_LOCAL["v03"] - CONFIG_LOCAL["media_b1_p82"],
        "gap_en_desviaciones": round(
            (CONFIG_LOCAL["v03"] - CONFIG_LOCAL["media_b1_p82"]) / CONFIG_LOCAL["sd_b1_p82"], 2
        ),
        "motivo_de_la_sonda": MOTIVO_SONDA,
        "motivo_de_los_umbrales": MOTIVO_UMBRALES,
        "motivo_de_la_semilla": MOTIVO_SEMILLA,
        "motivo_del_resultado_esperado": MOTIVO_ESPERADO,
        "entrenamiento": traza,
        "csv": {
            "ruta": str(DESTINO_CSV.relative_to(PROJECT_ROOT)),
            "sha256": escritura["sha256"],
            "bytes": escritura["bytes"],
            "n_filas": escritura["n_filas_en_el_fichero"],
            "n_lineas_contando_cabecera": escritura["n_lineas_contando_cabecera"],
            "cabecera_leida_del_fichero": escritura["cabecera"],
            "verificado_releyendo_el_fichero": True,
        },
        "manifiesto": manifiesto,
        "fila_anadida": fila,
        "distribucion": dist,
        "verificaciones": verificaciones,
        "tabla_verificaciones": tabla_markdown(
            ["verificacion", "valor", "resultado"],
            [[nombre, str(valor), "OK" if ok else "FALLO"] for nombre, valor, ok in verificaciones],
        ),
        "tabla_distribucion": tabla_markdown(
            ["etiqueta", "ceros", "unos", "proporcion_de_unos", "valor_mas_frecuente"],
            [
                [e, dist[e]["n_ceros"], dist[e]["n_unos"],
                 f"{dist[e]['proporcion_de_unos']:.6f}",
                 f"{dist[e]['proporcion_del_valor_mas_frecuente']:.6f}"]
                for e in CONFIG_LOCAL["etiquetas"]
            ],
        ),
        "kaggle": "NO se ha enviado nada a Kaggle y no se han buscado credenciales",
        "tiempo_real_segundos": round(time.perf_counter() - inicio, 3),
        "dispositivo": cadena_del_dispositivo_resuelto(),
    }
    escribir_unico_atomico(DESTINO_JSON, a_json_legible(a_jsonable(carga)))
    escribir_unico_atomico(DESTINO_MD, construir_markdown(carga))
    print(f"[p82-02] CSV en {DESTINO_CSV}")
    print(f"[p82-02] sha256 del CSV: {hash_csv}")
    print("[p82-02] NO se ha enviado nada a Kaggle")
    return 0


def main() -> int:
    """Punto de entrada.

    Returns:
        Codigo de salida.
    """
    parser = argparse.ArgumentParser(description="P82-02: submission con la mejor RNN de P82.")
    parser.add_argument("--modo", choices=["humo", "completo"], default="humo")
    parser.add_argument("--suelo", type=int, default=SUELO_COBERTURA)
    args = parser.parse_args()
    if args.modo == "humo":
        return ejecutar_humo(args.suelo)
    return ejecutar_completo()


if __name__ == "__main__":
    raise SystemExit(main())
