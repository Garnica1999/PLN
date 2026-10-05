"""20_h2b_postproceso_m4.py
Fecha: 2026-10-03
Descripcion: H2b - refinamiento acotado SIN reentrenar del resultado H2. H2
dejo W2 (objetivo suave top3) con macro-F1 fijo16 en val de 0.3002 frente a
0.2658 de W1 (delta pareado +0.0344, IC95 [-0.0051, +0.0749]: roza cero). Este
experimento opera SOLO sobre las probabilidades ya persistidas en
resultados/m4_h2_w2_preds.npz y resultados/m4_h1_v1r_preds.npz (claves
ids_*/y_*/probs_* por bloque) y prueba tres variantes pre-registradas:

  - P1 = W2 + calibracion de sesgos aditivos por clase (log p + b_c), con los
    sesgos optimizados por optimizar_sesgos_logits (modulo 06, coordinate
    ascent estilo Fase C) sobre val completa.
  - P2 = ensemble simple: argmax del promedio (W1 + W2) / 2 de probabilidades.
  - P3 = ensemble + calibracion de sesgos por clase (sesgos sobre el promedio).
  - W1 = referencia interna, sin post-proceso.

Gate en val canonica (macro-F1 de espacio fijo de 16): para cada variante,
delta pareado vs W1 con IC bootstrap 1000/42 sobre la val usada y consistencia
A/B (los parametros ajustables se calibran en una mitad de la val, definida por
paridad de posicion en el orden de id, y se evaluan en la otra; y simetrico).
Gana la variante con delta > 0 e IC95 inferior > 0 en la val usada y delta > 0
en ambas mitades A/B; desempate por mayor delta, luego mayor macro-F1 fijo16 en
val, luego orden p1, p2, p3. Solo si hay ganador se leen UNA vez el test
canonico y el temporal con las probabilidades guardadas (marcado
development_test_reutilizado); si no hay ganador no hay lecturas de test. El
holdout sellado no esta en esos npz y no se toca.

Estado del arte aplicado:
  - Ajuste de sesgos/priors por clase sobre logits (logit adjustment; Menon et
    al., 2021) y calibracion de sesgos en validacion (reutilizada del modulo 06,
    Fase C/U del proyecto).
  - Soft voting por promedio de probabilidades (Zhou, 2012).
  - IC bootstrap percentil pareado (Efron y Tibshirani, 1993) y chequeo de
    robustez por mitades para no decidir con senales fragiles.

Uso:
    /home/carlos/venv/bin/python scripts/20_h2b_postproceso_m4.py --todo
    /home/carlos/venv/bin/python scripts/20_h2b_postproceso_m4.py --smoke
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
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ===== SECCION 1: configuracion =====

RAIZ = Path(__file__).resolve().parent.parent

CONFIG = {
    "semilla": 42,
    "n_bootstrap": 1000,
    "max_rondas_sesgos_fallback": 3,
    "eps_log": 1e-12,
    "smoke_n_val": 800,
    "dir_smoke": "/tmp/opencode/h2b_smoke",
    "rutas": {
        "w1_preds": "resultados/m4_h1_v1r_preds.npz",
        "w2_preds": "resultados/m4_h2_w2_preds.npz",
        "h2_json": "resultados/m4_h2_objetivo_suave_v1.json",
        "json": "resultados/m4_h2b_postproceso_v1.json",
        "figura": "figuras/fig_h2b_postproceso.png",
        "log": "experimentos/20_h2b_postproceso.log",
    },
}

# Referencia interna (sin post-proceso) y variantes candidatas.
REFERENCIA = "w1"
VARIANTES_POST = ("p1", "p2", "p3")
BLOQUES_PRED = ("val", "test", "temporal_val", "temporal_test")
BLOQUES_TEST = ("test", "temporal_val", "temporal_test")

# Premisas de diseno congeladas (trazabilidad del pre-registro).
PREMISAS = [
    (
        "H2 (registrado en resultados/m4_h2_objetivo_suave_v1.json): W2 dejo macro-F1 "
        "fijo16 en val de 0.3002 vs 0.2658 de W1 (delta pareado +0.0344, IC95 "
        "[-0.0051, +0.0749], p=0.128): el efecto roza cero y no alcanzo el gate. H2b "
        "refina SOLO el post-proceso de las probabilidades guardadas, sin reentrenar."
    ),
    (
        "Sin reentrenamiento: se reutilizan integramente las probabilidades persistidas "
        "de W2 (resultados/m4_h2_w2_preds.npz) y W1 (resultados/m4_h1_v1r_preds.npz). "
        "El holdout sellado no esta en esos npz y no se toca; no se reajusta ningun "
        "modelo, vectorizador ni split."
    ),
    (
        "P1/P3 (sesgos por clase): decision argmax(log p + b_c). log p difiere del "
        "decision_function en una constante por fila (denominador del softmax), lo que "
        "no altera ni el argmax sin sesgos ni la optimizacion aditiva de sesgos. Los "
        "sesgos se optimizan con optimizar_sesgos_logits del modulo 06 (rejilla "
        "[-2, 2] paso 0.05, coordinate ascent, maximo MAX_RONDAS_COORDENADA rondas) "
        "SIEMPRE sobre val (o sobre la mitad de ajuste en el chequeo A/B)."
    ),
    (
        "P2 (ensemble): promedio aritmetico (W1 + W2) / 2 de las probabilidades por "
        "fila y argmax; soft voting sin parametros ajustables."
    ),
    (
        "Gate: delta pareado de macro-F1 fijo16 vs W1 con IC bootstrap percentil 95 % "
        "(1000 remuestreos, semilla 42, mismos indices para ambas variantes) y "
        "consistencia A/B por mitades (A = posiciones pares y B = impares del orden de "
        "id). Para P1/P3 los sesgos se ajustan en la mitad complementaria y se evaluan "
        "en la otra; P2, sin parametros, se evalua directo en cada mitad. Gana la "
        "variante con delta > 0 e IC95 inferior > 0 en la val usada y delta > 0 en ambas "
        "mitades; desempate por mayor delta, luego mayor macro-F1 fijo16 en val, luego "
        "orden p1, p2, p3."
    ),
    (
        "Lecturas de test: solo si hay ganador se leen UNA vez el test canonico y el "
        "temporal (temporal_val y temporal_test) con las probabilidades ya guardadas, "
        "marcado development_test_reutilizado; W1 se reporta como comparador interno. "
        "Si no hay ganador no hay ninguna lectura de test/temporal."
    ),
    (
        "Metricas: macro-F1 de espacio fijo de 16 (zero_division=0) como metrica "
        "primaria; se reporta accuracy en todas las variantes y bloques como contexto "
        "de la compensacion macro-F1/accuracy."
    ),
    (
        "Modo smoke: submuestra aleatoria de 800 documentos de val (semilla 42) y "
        "salidas SOLO en /tmp/opencode/h2b_smoke/; el smoke verifica el codigo y no "
        "decide gates ni se mezcla con las salidas reales."
    ),
]

# Cache de modulos con nombre que empieza por digito (importlib por ruta).
_MODULO_19 = None
_MODULO_06 = None

LOGGER = logging.getLogger("h2b_postproceso_m4")


# ===== SECCION 2: utilidades (logger, modulos, md5, guardas, rutas) =====


def configurar_logger(ruta_log: str) -> logging.Logger:
    """Configura el logger del experimento (archivo append + consola).

    Args:
        ruta_log: Ruta del archivo de log (se crea el directorio padre).

    Returns:
        Logger configurado.
    """
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formato = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    Path(ruta_log).parent.mkdir(parents=True, exist_ok=True)
    manejador_archivo = logging.FileHandler(ruta_log, mode="a", encoding="utf-8")
    manejador_archivo.setFormatter(formato)
    LOGGER.addHandler(manejador_archivo)
    manejador_consola = logging.StreamHandler()
    manejador_consola.setFormatter(formato)
    LOGGER.addHandler(manejador_consola)
    LOGGER.propagate = False
    return LOGGER


def _cargar_modulo(nombre: str, ruta: Path):
    """Importa un script por ruta (su nombre empieza por digito).

    Args:
        nombre: Nombre interno para sys.modules.
        ruta: Ruta del script a importar.

    Returns:
        Modulo ejecutado.

    Raises:
        ImportError: Si la ruta no puede cargarse.
    """
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    if spec is None or spec.loader is None:
        raise ImportError(f"no se pudo cargar {ruta}")
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = modulo
    spec.loader.exec_module(modulo)
    return modulo


def modulo_19():
    """Carga (y cachea) scripts/19_h2_objetivo_suave_m4.py.

    Returns:
        Modulo 19 ejecutado (bootstrap pareado, helpers de md5/versiones,
        cargador de los modulos 17/18/06).
    """
    global _MODULO_19
    if _MODULO_19 is None:
        _MODULO_19 = _cargar_modulo(
            "h2_objetivo_suave_19_h2b", RAIZ / "scripts" / "19_h2_objetivo_suave_m4.py"
        )
    return _MODULO_19


def modulo_06():
    """Carga scripts/06_baseline_clasico.py reutilizando el cargador de 19.

    Returns:
        Modulo 06 ejecutado (evaluar_single_label, optimizar_sesgos_logits,
        GRID_SESGOS, MAX_RONDAS_COORDENADA).
    """
    global _MODULO_06
    if _MODULO_06 is None:
        _MODULO_06 = modulo_19().modulo_06()
    return _MODULO_06


def md5_archivo(ruta: str) -> str | None:
    """Delega el md5 por bloques en el modulo 19 (reutilizacion).

    Args:
        ruta: Ruta del archivo.

    Returns:
        Hexdigest md5 o None si no existe.
    """
    return modulo_19().md5_archivo(ruta)


def versiones_entorno() -> dict:
    """Delega el inventario de versiones en el modulo 19 (reutilizacion).

    Returns:
        Dict con python, plataforma y versiones de librerias clave.
    """
    return modulo_19().versiones_entorno()


def escribir_json(ruta: str, contenido: dict, permitir: bool, descripcion: str,
                  logger: logging.Logger) -> None:
    """Escribe un JSON nuevo (o lo reescribe solo si esta permitido).

    Args:
        ruta: Ruta de salida.
        contenido: Objeto serializable.
        permitir: True para permitir reescritura (smoke en scratch).
        descripcion: Etiqueta para el log.
        logger: Logger activo.

    Raises:
        FileExistsError: Si el archivo existe y no se permite reescribir.
    """
    destino = Path(ruta)
    if destino.exists() and not permitir:
        raise FileExistsError(f"{descripcion} ya existe (no se sobrescribe): {destino}")
    destino.parent.mkdir(parents=True, exist_ok=True)
    with open(destino, "w", encoding="utf-8") as archivo:
        json.dump(contenido, archivo, ensure_ascii=False, indent=2, default=str)
    logger.info("%s guardado: %s (md5 %s)", descripcion, destino, md5_archivo(ruta))


def guardar_figura(fig, ruta: str, logger: logging.Logger) -> None:
    """Guarda la figura en PNG (dpi 150) creando el directorio padre.

    Args:
        fig: Figura de matplotlib.
        ruta: Ruta del PNG.
        logger: Logger activo.
    """
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    logger.info("figura guardada: %s", ruta)


def rutas_activas(smoke: bool) -> dict:
    """Construye el mapa de rutas de entrada/salida segun el modo smoke.

    En smoke TODAS las salidas van a /tmp/opencode/h2b_smoke/; las entradas
    (npz de probabilidades y JSON de H2) se leen siempre de sus rutas
    canonicas de solo lectura.

    Args:
        smoke: True para usar salidas exclusivamente en /tmp/opencode/h2b_smoke.

    Returns:
        Dict combinado de rutas de entrada y de salida.
    """
    rutas = dict(CONFIG["rutas"])
    if smoke:
        base = Path(CONFIG["dir_smoke"])
        rutas.update(
            {
                "json": str(base / "m4_h2b_postproceso_v1.json"),
                "figura": str(base / "fig_h2b_postproceso.png"),
                "log": str(base / "20_h2b_postproceso.log"),
            }
        )
    return rutas


def verificar_entradas(rutas: dict, logger: logging.Logger) -> None:
    """Aborta si falta alguna entrada canonica (solo lectura).

    Args:
        rutas: Dict de rutas activas.
        logger: Logger activo.

    Raises:
        FileNotFoundError: Si falta algun npz de probabilidades requerido.
    """
    faltantes = [rutas[clave] for clave in ("w1_preds", "w2_preds")
                 if not Path(rutas[clave]).exists()]
    if faltantes:
        raise FileNotFoundError(
            "faltan entradas requeridas (probs persistidas de H1/H2): " + ", ".join(faltantes)
        )
    if not Path(rutas["h2_json"]).exists():
        logger.warning(
            "sin referencia H2 (%s): el JSON de H2b no incluira los valores registrados",
            rutas["h2_json"],
        )
    logger.info("entradas verificadas: %s | %s", rutas["w1_preds"], rutas["w2_preds"])


def verificar_salidas_nuevas(rutas: dict, smoke: bool) -> None:
    """Aborta si una salida final ya existe (no se sobrescribe nada).

    Args:
        rutas: Dict de rutas activas.
        smoke: True si es corrida smoke (salidas scratch, se permite reescribir).

    Raises:
        FileExistsError: Si existe el JSON o la figura final en modo real.
    """
    if smoke:
        return
    existentes = [rutas[clave] for clave in ("json", "figura")
                  if Path(rutas[clave]).exists()]
    if existentes:
        raise FileExistsError(
            "salidas ya existentes (este experimento nunca las sobrescribe): "
            + ", ".join(existentes)
        )


# ===== SECCION 3: carga y alineacion de las probabilidades guardadas =====


def cargar_preds_npz(ruta: str) -> dict:
    """Carga un npz de predicciones persistidas (variante, metadatos, bloques).

    Args:
        ruta: Ruta del .npz (probs guardadas por los experimentos 18/19).

    Returns:
        Dict con ruta, variante, metadatos, clases y bloques {ids, y, probs}.

    Raises:
        FileNotFoundError: Si la ruta no existe.
    """
    if not Path(ruta).exists():
        raise FileNotFoundError(f"falta el npz de predicciones: {ruta}")
    with np.load(ruta, allow_pickle=False) as datos:
        crudos = {clave: datos[clave] for clave in datos.files}
    salida = {
        "ruta": ruta,
        "variante": str(crudos["variante"]),
        "metadatos": json.loads(str(crudos["metadatos_json"])),
        "clases": [str(c) for c in crudos["clases"]],
    }
    for bloque in BLOQUES_PRED:
        salida[bloque] = {
            "ids": np.asarray([str(v) for v in crudos[f"ids_{bloque}"]]),
            "y": np.asarray([str(v) for v in crudos[f"y_{bloque}"]]),
            "probs": np.asarray(crudos[f"probs_{bloque}"], dtype=np.float64),
        }
    return salida


def verificar_clases(preds_w1: dict, preds_w2: dict, clases_ref: list[str],
                     logger: logging.Logger) -> None:
    """Verifica que ambas variantes usan el mismo espacio fijo de 16 clases.

    Args:
        preds_w1: Predicciones de W1.
        preds_w2: Predicciones de W2.
        clases_ref: Espacio fijo canonico (modulo 17).
        logger: Logger activo.

    Raises:
        RuntimeError: Si algun npz no declara exactamente el espacio canonico.
    """
    for nombre, preds in (("w1", preds_w1), ("w2", preds_w2)):
        if preds["clases"] != list(clases_ref):
            raise RuntimeError(
                f"el npz de {nombre} declara un espacio de clases distinto al canonico "
                f"de 16: {preds['clases']}"
            )
    logger.info("espacio de clases verificado en w1 y w2 (%d clases)", len(clases_ref))


def alinear_bloques(preds_w1: dict, preds_w2: dict, logger: logging.Logger) -> None:
    """Alinea W2 con el orden de ids/etiquetas de W1 (in-place).

    Los npz de H1/H2 se generaron con los mismos splits; si el orden de ids
    difiere pero el conjunto es identico se reordena W2. Si las etiquetas no
    coinciden se aborta (serian documentos distintos).

    Args:
        preds_w1: Predicciones de W1 (orden de referencia).
        preds_w2: Predicciones de W2 (se reordena si hace falta).
        logger: Logger activo.

    Raises:
        RuntimeError: Si los conjuntos de ids o las etiquetas no coinciden.
    """
    for bloque in BLOQUES_PRED:
        ids1 = preds_w1[bloque]["ids"]
        ids2 = preds_w2[bloque]["ids"]
        if not np.array_equal(ids1, ids2):
            if set(ids1) != set(ids2):
                raise RuntimeError(
                    f"{bloque}: los ids de w1 y w2 no coinciden "
                    f"(interseccion {len(set(ids1) & set(ids2))} de {len(ids1)})"
                )
            posicion = {valor: indice for indice, valor in enumerate(ids2)}
            orden = np.asarray([posicion[valor] for valor in ids1])
            preds_w2[bloque]["ids"] = ids2[orden]
            preds_w2[bloque]["y"] = preds_w2[bloque]["y"][orden]
            preds_w2[bloque]["probs"] = preds_w2[bloque]["probs"][orden]
            logger.warning("%s: w2 reordenado para alinear con w1", bloque)
        if not np.array_equal(preds_w1[bloque]["y"], preds_w2[bloque]["y"]):
            raise RuntimeError(f"{bloque}: las etiquetas de w1 y w2 no coinciden")
        n1 = preds_w1[bloque]["probs"].shape
        n2 = preds_w2[bloque]["probs"].shape
        if n1 != n2 or n1[0] != len(ids1):
            raise RuntimeError(f"{bloque}: shapes incoherentes w1={n1} w2={n2}")


def cargar_probabilidades(rutas: dict, logger: logging.Logger) -> dict:
    """Carga, valida y alinea las probabilidades guardadas de W1 y W2.

    Args:
        rutas: Dict de rutas activas.
        logger: Logger activo.

    Returns:
        Dict con w1, w2 (bloques {ids, y, probs}) y clases (espacio canonico).
    """
    t0 = time.perf_counter()
    preds_w1 = cargar_preds_npz(rutas["w1_preds"])
    preds_w2 = cargar_preds_npz(rutas["w2_preds"])
    clases = list(modulo_19().modulo_17().CLASES_M4)
    verificar_clases(preds_w1, preds_w2, clases, logger)
    alinear_bloques(preds_w1, preds_w2, logger)
    for nombre, preds in (("w1", preds_w1), ("w2", preds_w2)):
        for bloque in BLOQUES_PRED:
            suma = preds[bloque]["probs"].sum(axis=1)
            if not np.allclose(suma, 1.0, atol=1e-6):
                logger.warning(
                    "%s/%s: filas de probabilidad que no suman 1 (min=%.6f max=%.6f); "
                    "el metodo log-prob sigue siendo valido (solo importa el argmax)",
                    nombre, bloque, float(suma.min()), float(suma.max()),
                )
    logger.info(
        "probabilidades cargadas y alineadas en %.1f s: w1=%s w2=%s",
        time.perf_counter() - t0, preds_w1["variante"], preds_w2["variante"],
    )
    return {"w1": preds_w1, "w2": preds_w2, "clases": clases}


def indices_mitades_por_id(ids) -> tuple[np.ndarray, np.ndarray]:
    """Divide posiciones en mitades A/B por paridad en el orden de id.

    Ordena los ids (numericos si es posible, lexicograficos si no) y asigna a
    la mitad A las posiciones pares (0, 2, 4, ...) y a la B las impares.

    Args:
        ids: Arreglo de ids (strings).

    Returns:
        (indices_a, indices_b) como posiciones ordenadas en el arreglo original.
    """
    ids = np.asarray(ids)
    try:
        claves = ids.astype(np.int64)
    except ValueError:
        claves = ids
    orden = np.argsort(claves, kind="stable")
    return np.sort(orden[0::2]), np.sort(orden[1::2])


def indices_smoke(y, n: int, semilla: int) -> np.ndarray:
    """Submuestra aleatoria de val para el modo smoke (semilla fija).

    Args:
        y: Etiquetas de val (solo para conocer n).
        n: Tamano de la submuestra (si n >= len(y) se usa todo).
        semilla: Semilla del generador.

    Returns:
        Indices ordenados de la submuestra.
    """
    n_efectivo = min(int(n), len(y))
    rng = np.random.default_rng(semilla)
    return np.sort(rng.choice(len(y), size=n_efectivo, replace=False))


# ===== SECCION 4: post-procesos P1/P2/P3 y consistencia A/B =====


def probs_a_logits(probs: np.ndarray, eps: float) -> np.ndarray:
    """Convierte probabilidades en log-probabilidades (con recorte por eps).

    log p difiere del decision_function en una constante por fila (denominador
    del softmax); no cambia el argmax ni la optimizacion aditiva de sesgos.

    Args:
        probs: Matriz (n, k) de probabilidades.
        eps: Recorte inferior para evitar log(0).

    Returns:
        Matriz (n, k) de log-probabilidades.
    """
    return np.log(np.clip(probs, eps, None))


def ajustar_sesgos(logits: np.ndarray, y, clases: list[str],
                   logger: logging.Logger, etiqueta: str) -> tuple[np.ndarray, dict]:
    """Optimiza sesgos aditivos por clase con el coordinate ascent del modulo 06.

    Args:
        logits: Matriz (n, k) de log-probabilidades del bloque de ajuste.
        y: Etiquetas reales del bloque de ajuste.
        clases: Espacio fijo de clases (orden de columnas).
        logger: Logger activo.
        etiqueta: Etiqueta para el log.

    Returns:
        (sesgos, detalle) con la metrica auxiliar antes/despues y el tiempo.
    """
    base = modulo_06()
    max_rondas = int(getattr(base, "MAX_RONDAS_COORDENADA",
                            CONFIG["max_rondas_sesgos_fallback"]))
    t0 = time.perf_counter()
    sesgos, detalle = base.optimizar_sesgos_logits(logits, y, clases, max_rondas)
    detalle["n_ajuste"] = len(y)
    detalle["max_rondas"] = max_rondas
    detalle["tiempo_seg"] = time.perf_counter() - t0
    detalle["nota"] = (
        "macro_f1_val_antes/despues es macro-F1 observado del modulo 06 (metrica "
        "auxiliar); la metrica del gate se recalcula sobre el espacio fijo de 16"
    )
    logger.info(
        "calibracion %s: n=%d sesgos optimizados en %.1f s (macro obs %.4f -> %.4f)",
        etiqueta, len(y), detalle["tiempo_seg"],
        detalle["macro_f1_val_antes"], detalle["macro_f1_val_despues"],
    )
    return sesgos, detalle


def aplicar_variante(nombre: str, probs_w1: np.ndarray, probs_w2: np.ndarray,
                     sesgos: np.ndarray | None, clases: list[str]) -> np.ndarray:
    """Aplica la regla de decision de una variante a probabilidades dadas.

    P2 promedia y aplica argmax (sin sesgos); P1 usa W2 y P3 el promedio, con
    argmax(log p + sesgos) cuando hay sesgos calibrados.

    Args:
        nombre: p1, p2 o p3.
        probs_w1: Probabilidades de W1 (n, k).
        probs_w2: Probabilidades de W2 (n, k).
        sesgos: Sesgos por clase (None para p2).
        clases: Espacio fijo de clases (orden de columnas).

    Returns:
        Arreglo (n,) de etiquetas predichas (strings).
    """
    if nombre == "p2":
        promedio = 0.5 * (probs_w1 + probs_w2)
        return np.asarray(clases)[np.argmax(promedio, axis=1)]
    base = probs_w2 if nombre == "p1" else 0.5 * (probs_w1 + probs_w2)
    logits = probs_a_logits(base, CONFIG["eps_log"])
    return np.asarray(clases)[np.argmax(logits + sesgos, axis=1)]


def predecir_variante(nombre: str, probs_w1: np.ndarray, probs_w2: np.ndarray,
                      idx_ajuste: np.ndarray, idx_pred: np.ndarray, y_ajuste,
                      clases: list[str], logger: logging.Logger,
                      etiqueta: str) -> dict:
    """Ajusta (si aplica) y predice una variante sobre un subconjunto.

    P1/P3 ajustan sus sesgos SOLO con idx_ajuste y predicen idx_pred; P2 no
    tiene parametros ajustables y solo predice.

    Args:
        nombre: p1, p2 o p3.
        probs_w1: Probabilidades completas de W1 (n, k).
        probs_w2: Probabilidades completas de W2 (n, k).
        idx_ajuste: Posiciones del bloque de ajuste.
        idx_pred: Posiciones del bloque de prediccion.
        y_ajuste: Etiquetas del bloque de ajuste.
        clases: Espacio fijo de clases.
        logger: Logger activo.
        etiqueta: Etiqueta para el log.

    Returns:
        Dict con predicciones, sesgos (o None) y detalle de calibracion (o None).
    """
    t0 = time.perf_counter()
    if nombre == "p2":
        preds = aplicar_variante(nombre, probs_w1[idx_pred], probs_w2[idx_pred], None, clases)
        return {
            "predicciones": preds,
            "sesgos": None,
            "detalle_calibracion": None,
            "tiempo_seg": time.perf_counter() - t0,
        }
    base_ajuste = probs_w2[idx_ajuste] if nombre == "p1" else 0.5 * (
        probs_w1[idx_ajuste] + probs_w2[idx_ajuste]
    )
    logits_ajuste = probs_a_logits(base_ajuste, CONFIG["eps_log"])
    sesgos, detalle = ajustar_sesgos(logits_ajuste, y_ajuste, clases, logger, etiqueta)
    preds = aplicar_variante(nombre, probs_w1[idx_pred], probs_w2[idx_pred], sesgos, clases)
    return {
        "predicciones": preds,
        "sesgos": sesgos,
        "detalle_calibracion": detalle,
        "tiempo_seg": time.perf_counter() - t0,
    }


def evaluar_predicciones(y, preds, clases: list[str]) -> dict:
    """Metricas single-label comparables (delegadas al modulo 06).

    Args:
        y: Etiquetas reales.
        preds: Predicciones single-label.
        clases: Espacio fijo de 16 clases.

    Returns:
        Dict con macro_f1, macro_f1_espacio_fijo, accuracy y n.
    """
    metricas = modulo_06().evaluar_single_label(y, preds, clases)
    metricas["n"] = len(y)
    return metricas


def consistencia_ab(nombre: str, probs_w1: np.ndarray, probs_w2: np.ndarray,
                    y, idx_a: np.ndarray, idx_b: np.ndarray, clases: list[str],
                    logger: logging.Logger) -> dict:
    """Chequeo de robustez A/B por mitades complementarias.

    Para P1/P3 los sesgos se ajustan en una mitad y se evaluan en la otra (y
    simetrico); P2, sin parametros, se evalua directo. Las mitades se definen
    por paridad de posicion en el orden de id (A = pares, B = impares).

    Args:
        nombre: p1, p2 o p3.
        probs_w1: Probabilidades completas de W1 (n, k).
        probs_w2: Probabilidades completas de W2 (n, k).
        y: Etiquetas de val usadas.
        idx_a: Posiciones de la mitad A.
        idx_b: Posiciones de la mitad B.
        clases: Espacio fijo de clases.
        logger: Logger activo.

    Returns:
        Dict con deltas, metricas por mitad, sesgos ajustados y consistente.
    """
    pred_w1_a = np.asarray(clases)[np.argmax(probs_w1[idx_a], axis=1)]
    pred_w1_b = np.asarray(clases)[np.argmax(probs_w1[idx_b], axis=1)]
    met_w1_a = evaluar_predicciones(y[idx_a], pred_w1_a, clases)
    met_w1_b = evaluar_predicciones(y[idx_b], pred_w1_b, clases)

    salida_a = predecir_variante(
        nombre, probs_w1, probs_w2, idx_b, idx_a, y[idx_b], clases, logger,
        f"{nombre} A (ajuste en B)",
    )
    salida_b = predecir_variante(
        nombre, probs_w1, probs_w2, idx_a, idx_b, y[idx_a], clases, logger,
        f"{nombre} B (ajuste en A)",
    )
    met_a = evaluar_predicciones(y[idx_a], salida_a["predicciones"], clases)
    met_b = evaluar_predicciones(y[idx_b], salida_b["predicciones"], clases)
    delta_a = met_a["macro_f1_espacio_fijo"] - met_w1_a["macro_f1_espacio_fijo"]
    delta_b = met_b["macro_f1_espacio_fijo"] - met_w1_b["macro_f1_espacio_fijo"]
    resultado = {
        "mitades": "A = posiciones pares y B = impares del orden de id",
        "n_mitad_a": len(idx_a),
        "n_mitad_b": len(idx_b),
        "delta_mitad_a": float(delta_a),
        "delta_mitad_b": float(delta_b),
        "macro_f1_fijo16_mitad_a": {
            "w1": met_w1_a["macro_f1_espacio_fijo"],
            nombre: met_a["macro_f1_espacio_fijo"],
        },
        "macro_f1_fijo16_mitad_b": {
            "w1": met_w1_b["macro_f1_espacio_fijo"],
            nombre: met_b["macro_f1_espacio_fijo"],
        },
        "accuracy_mitad_a": {"w1": met_w1_a["accuracy"], nombre: met_a["accuracy"]},
        "accuracy_mitad_b": {"w1": met_w1_b["accuracy"], nombre: met_b["accuracy"]},
        "sesgos_ajustados_en_b_para_a": (
            None if salida_a["sesgos"] is None
            else [float(valor) for valor in salida_a["sesgos"]]
        ),
        "sesgos_ajustados_en_a_para_b": (
            None if salida_b["sesgos"] is None
            else [float(valor) for valor in salida_b["sesgos"]]
        ),
        "consistente": bool(delta_a > 0.0 and delta_b > 0.0),
    }
    logger.info(
        "A/B %s: delta A=%+.4f delta B=%+.4f consistente=%s",
        nombre, delta_a, delta_b, resultado["consistente"],
    )
    return resultado


# ===== SECCION 5: evaluacion en val, bootstrap pareado y decision =====


def evaluar_val_y_gate(datos: dict, indices: np.ndarray, logger: logging.Logger) -> dict:
    """Evalua las variantes en val, corre bootstrap pareado y consistencia A/B.

    Args:
        datos: Salida de cargar_probabilidades.
        indices: Posiciones de val a usar (completa o submuestra smoke).
        logger: Logger activo.

    Returns:
        Dict con metricas de val, comparaciones bootstrap, consistencia A/B,
        sesgos de val completa y tiempos de cada etapa.
    """
    clases = datos["clases"]
    y = datos["w1"]["val"]["y"][indices]
    probs_w1 = datos["w1"]["val"]["probs"][indices]
    probs_w2 = datos["w2"]["val"]["probs"][indices]

    t0 = time.perf_counter()
    pred_w1 = np.asarray(clases)[np.argmax(probs_w1, axis=1)]
    metricas = {REFERENCIA: evaluar_predicciones(y, pred_w1, clases)}
    sesgos_val: dict = {REFERENCIA: None}
    predicciones = {REFERENCIA: pred_w1}
    posiciones = np.arange(len(indices))
    for nombre in VARIANTES_POST:
        salida = predecir_variante(
            nombre, probs_w1, probs_w2, posiciones, posiciones, y, clases, logger,
            f"{nombre} val usada",
        )
        predicciones[nombre] = salida["predicciones"]
        metricas[nombre] = evaluar_predicciones(y, predicciones[nombre], clases)
        metricas[nombre]["sesgos"] = (
            None if salida["sesgos"] is None
            else [float(valor) for valor in salida["sesgos"]]
        )
        metricas[nombre]["detalle_calibracion"] = salida["detalle_calibracion"]
        sesgos_val[nombre] = salida["sesgos"]
    tiempo_val = time.perf_counter() - t0

    t0 = time.perf_counter()
    comparaciones = {}
    for nombre in VARIANTES_POST:
        comparaciones[nombre] = modulo_19().bootstrap_pareado_fijo16(
            y, predicciones[nombre], predicciones[REFERENCIA],
            CONFIG["n_bootstrap"], CONFIG["semilla"], logger, f"{nombre}_vs_w1",
        )
    tiempo_bootstrap = time.perf_counter() - t0

    t0 = time.perf_counter()
    idx_a, idx_b = indices_mitades_por_id(datos["w1"]["val"]["ids"][indices])
    consistencias = {
        nombre: consistencia_ab(
            nombre, probs_w1, probs_w2, y, idx_a, idx_b, clases, logger
        )
        for nombre in VARIANTES_POST
    }
    tiempo_ab = time.perf_counter() - t0

    return {
        "val": {
            "n": len(indices),
            "n_original": len(datos["w1"]["val"]["y"]),
            "submuestra_smoke": bool(len(indices) < len(datos["w1"]["val"]["y"])),
            "variantes": metricas,
        },
        "comparaciones_pareadas_val": comparaciones,
        "consistencia_ab": consistencias,
        "sesgos_val": sesgos_val,
        "tiempos_seg": {
            "val_y_calibracion_seg": tiempo_val,
            "bootstrap_seg": tiempo_bootstrap,
            "consistencia_ab_seg": tiempo_ab,
        },
    }


def decidir_ganador(val_metricas: dict, comparaciones: dict,
                    consistencias: dict) -> dict:
    """Aplica el gate pre-registrado y selecciona la variante ganadora.

    Criterio: delta puntual > 0 e IC95 inferior > 0 (macro-F1 fijo16 vs W1) y
    delta > 0 en ambas mitades A/B. Desempate: mayor delta puntual, luego mayor
    macro-F1 fijo16 en val, luego orden p1, p2, p3.

    Args:
        val_metricas: Dict variante -> metricas de val.
        comparaciones: Dict variante -> bootstrap pareado vs W1.
        consistencias: Dict variante -> chequeo A/B.

    Returns:
        Dict con ganador, resultado, criterio y detalle por variante.
    """
    detalle = {}
    candidatos = []
    for nombre in VARIANTES_POST:
        comp = comparaciones[nombre]["delta_macro_f1_espacio_fijo"]
        ab = consistencias[nombre]
        cumple_ic = bool(comp["punto"] > 0.0 and comp["ic95_inf"] > 0.0)
        cumple_ab = bool(ab["delta_mitad_a"] > 0.0 and ab["delta_mitad_b"] > 0.0)
        detalle[nombre] = {
            "delta_puntual": comp["punto"],
            "ic95_inf": comp["ic95_inf"],
            "ic95_sup": comp["ic95_sup"],
            "cumple_ic95_inf_positivo": cumple_ic,
            "cumple_consistencia_ab": cumple_ab,
        }
        if cumple_ic and cumple_ab:
            candidatos.append(nombre)
    if candidatos:
        ganador = max(
            candidatos,
            key=lambda n: (
                comparaciones[n]["delta_macro_f1_espacio_fijo"]["punto"],
                val_metricas[n]["macro_f1_espacio_fijo"],
                -VARIANTES_POST.index(n),
            ),
        )
    else:
        ganador = None
    return {
        "ganador": ganador,
        "resultado": "positivo" if ganador is not None else "negativo",
        "criterio": (
            "delta pareado (macro-F1 fijo16) vs W1 en val con IC95 inferior > 0 y "
            "delta > 0 en ambas mitades A/B; desempate por mayor delta puntual, "
            "luego mayor macro-F1 fijo16 en val, luego orden p1, p2, p3"
        ),
        "candidatos_que_cumplen": candidatos,
        "detalle_por_variante": detalle,
    }


# ===== SECCION 6: test canonico y temporal (solo con ganador) =====


def evaluar_test_temporal(ganador: str, datos: dict, sesgos_ganador,
                          clases: list[str], logger: logging.Logger) -> dict:
    """Lee UNA vez test/temporal con las probs guardadas (solo con ganador).

    Los sesgos del ganador ya estan ajustados en val completa (P1/P3); no se
    reajusta nada con test/temporal. Se marca development_test_reutilizado.

    Args:
        ganador: Variante ganadora (p1, p2 o p3).
        datos: Salida de cargar_probabilidades.
        sesgos_ganador: Sesgos ajustados en val (None para p2).
        clases: Espacio fijo de clases.
        logger: Logger activo.

    Returns:
        Dict bloque -> metricas del ganador y de W1, con la marca de reutilizacion.
    """
    salida = {
        "development_test_reutilizado": True,
        "n_lecturas_por_bloque": 1,
        "nota": (
            "test canonico y temporal leidos UNA sola vez, despues de declarar "
            "ganador, reutilizando probabilidades ya persistidas; los sesgos del "
            "ganador se ajustaron solo en val"
        ),
    }
    for bloque in BLOQUES_TEST:
        y = datos["w1"][bloque]["y"]
        probs_w1 = datos["w1"][bloque]["probs"]
        probs_w2 = datos["w2"][bloque]["probs"]
        pred_w1 = np.asarray(clases)[np.argmax(probs_w1, axis=1)]
        pred_ganador = aplicar_variante(ganador, probs_w1, probs_w2,
                                        sesgos_ganador, clases)
        salida[bloque] = {
            "ganador": evaluar_predicciones(y, pred_ganador, clases),
            "w1": evaluar_predicciones(y, pred_w1, clases),
        }
        logger.info(
            "%s: %s fijo16=%.4f acc=%.4f | w1 fijo16=%.4f acc=%.4f",
            bloque, ganador,
            salida[bloque]["ganador"]["macro_f1_espacio_fijo"],
            salida[bloque]["ganador"]["accuracy"],
            salida[bloque]["w1"]["macro_f1_espacio_fijo"],
            salida[bloque]["w1"]["accuracy"],
        )
    return salida


# ===== SECCION 7: figura de deltas con IC y A/B =====


def figura_deltas(resultado: dict, ruta: str, logger: logging.Logger) -> None:
    """Genera la figura de delta macro-F1 fijo16 vs W1 con IC95 por variante.

    Args:
        resultado: Resultado parcial con comparaciones y consistencia A/B.
        ruta: Ruta del PNG a guardar.
        logger: Logger activo.
    """
    nombres = list(VARIANTES_POST)
    puntos, inf, sup, ab_a, ab_b = [], [], [], [], []
    for nombre in nombres:
        comp = resultado["comparaciones_pareadas_val"][nombre]["delta_macro_f1_espacio_fijo"]
        puntos.append(comp["punto"])
        inf.append(comp["ic95_inf"])
        sup.append(comp["ic95_sup"])
        ab = resultado["consistencia_ab"][nombre]
        ab_a.append(ab["delta_mitad_a"])
        ab_b.append(ab["delta_mitad_b"])
    puntos_arr = np.asarray(puntos)
    errores = np.vstack([puntos_arr - np.asarray(inf), np.asarray(sup) - puntos_arr])
    x = np.arange(len(nombres))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, puntos_arr, width=0.5, color="#2F75B5",
           label="delta puntual (val usada)")
    ax.errorbar(x, puntos_arr, yerr=errores, fmt="none", ecolor="#1F1F1F",
                capsize=6, linewidth=1.4, label="IC95 bootstrap 1000/42")
    ax.scatter(x - 0.10, ab_a, marker="^", color="#ED7D31", zorder=3,
               label="delta mitad A (ajuste en B)")
    ax.scatter(x + 0.10, ab_b, marker="v", color="#70AD47", zorder=3,
               label="delta mitad B (ajuste en A)")
    ax.axhline(0.0, color="#C00000", linestyle="--", linewidth=1.2)
    for xi, punto in zip(x, puntos_arr):
        ax.annotate(f"{punto:+.4f}", (xi, punto), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([nombre.upper() for nombre in nombres])
    ax.set_ylabel("delta macro-F1 fijo16 vs W1")
    ax.set_title(
        "H2b: post-proceso sobre probabilidades guardadas (val)\n"
        "Barras: delta con IC95 pareado; triangulos: consistencia A/B"
    )
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    guardar_figura(fig, ruta, logger)


# ===== SECCION 8: orquestacion, resumen, CLI y main =====


def cargar_referencia_h2(rutas: dict, logger: logging.Logger) -> dict:
    """Lee los valores registrados de H2 (solo lectura, trazabilidad).

    Args:
        rutas: Dict de rutas activas.
        logger: Logger activo.

    Returns:
        Dict con disponibilidad, md5 y los valores de val/delta registrados.
    """
    ruta = rutas["h2_json"]
    if not Path(ruta).exists():
        return {"disponible": False, "ruta": ruta, "md5": None}
    with open(ruta, encoding="utf-8") as archivo:
        referencia = json.load(archivo)
    salida = {"disponible": True, "ruta": ruta, "md5": md5_archivo(ruta)}
    try:
        val_w1 = referencia["variantes"]["w1"]["val"]
        val_w2 = referencia["variantes"]["w2"]["val"]
        comp = referencia["comparaciones_pareadas_val"]["w2_vs_w1"][
            "delta_macro_f1_espacio_fijo"
        ]
        salida["val_w1"] = {
            "macro_f1_espacio_fijo": float(val_w1["macro_f1_espacio_fijo"]),
            "accuracy": float(val_w1["accuracy"]),
        }
        salida["val_w2"] = {
            "macro_f1_espacio_fijo": float(val_w2["macro_f1_espacio_fijo"]),
            "accuracy": float(val_w2["accuracy"]),
        }
        salida["delta_w2_vs_w1"] = {
            "punto": float(comp["punto"]),
            "ic95_inf": float(comp["ic95_inf"]),
            "ic95_sup": float(comp["ic95_sup"]),
        }
    except (KeyError, TypeError) as error:
        logger.warning("referencia H2 incompleta: %s", error)
    return salida


def ejecutar_h2b(args, logger: logging.Logger, rutas: dict) -> tuple[dict, float]:
    """Ejecuta el pipeline completo de H2b y guarda figura y JSON.

    Args:
        args: Argumentos CLI (smoke).
        logger: Logger activo.
        rutas: Dict de rutas activas.

    Returns:
        (resultado serializado, tiempo total en segundos).
    """
    t_total = time.perf_counter()
    smoke = bool(args.smoke)
    datos = cargar_probabilidades(rutas, logger)
    clases = datos["clases"]

    n_val_original = len(datos["w1"]["val"]["y"])
    if smoke:
        indices = indices_smoke(datos["w1"]["val"]["y"], CONFIG["smoke_n_val"],
                                CONFIG["semilla"])
        logger.info("smoke: val submuestreada a %d de %d docs", len(indices), n_val_original)
    else:
        indices = np.arange(n_val_original)

    evaluacion = evaluar_val_y_gate(datos, indices, logger)
    decision = decidir_ganador(
        evaluacion["val"]["variantes"],
        evaluacion["comparaciones_pareadas_val"],
        evaluacion["consistencia_ab"],
    )
    logger.info("decision H2b: %s (ganador=%s)", decision["resultado"], decision["ganador"])

    t0 = time.perf_counter()
    test_temporal = None
    if decision["ganador"] is not None:
        test_temporal = evaluar_test_temporal(
            decision["ganador"], datos,
            evaluacion["sesgos_val"][decision["ganador"]], clases, logger,
        )
    else:
        logger.info("sin ganador: NO se leen test canonico ni temporal (solo val)")
    tiempo_test = time.perf_counter() - t0

    resultado = {
        "fecha": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "script": "scripts/20_h2b_postproceso_m4.py",
        "smoke": smoke,
        "modo": "smoke" if smoke else "todo",
        "versiones": versiones_entorno(),
        "premisas": PREMISAS,
        "clases_espacio_fijo": list(clases),
        "parametros": {
            "semilla": CONFIG["semilla"],
            "n_bootstrap": CONFIG["n_bootstrap"],
            "eps_log": CONFIG["eps_log"],
            "max_rondas_sesgos": int(getattr(
                modulo_06(), "MAX_RONDAS_COORDENADA",
                CONFIG["max_rondas_sesgos_fallback"],
            )),
            "smoke_n_val": CONFIG["smoke_n_val"],
        },
        "md5_entradas": {
            "w1_preds": md5_archivo(rutas["w1_preds"]),
            "w2_preds": md5_archivo(rutas["w2_preds"]),
            "h2_json": md5_archivo(rutas["h2_json"]),
            "script": md5_archivo(str(RAIZ / "scripts" / "20_h2b_postproceso_m4.py")),
        },
        "referencia_h2": cargar_referencia_h2(rutas, logger),
        "val": evaluacion["val"],
        "comparaciones_pareadas_val": evaluacion["comparaciones_pareadas_val"],
        "consistencia_ab": evaluacion["consistencia_ab"],
        "decision": decision,
        "test_canonico_y_temporal": test_temporal,
        "tiempos_seg": {
            **evaluacion["tiempos_seg"],
            "test_temporal_seg": tiempo_test,
        },
        "artefactos": {
            "json": rutas["json"],
            "figura": rutas["figura"],
            "log": rutas["log"],
        },
    }

    figura_deltas(resultado, rutas["figura"], logger)
    resultado["md5_salidas"] = {"figura": md5_archivo(rutas["figura"])}
    total_seg = time.perf_counter() - t_total
    resultado["tiempos_seg"]["total_seg"] = total_seg
    escribir_json(rutas["json"], resultado, permitir=smoke, descripcion="JSON H2b",
                  logger=logger)
    return resultado, total_seg


def imprimir_resumen(resultado: dict, total_seg: float, logger: logging.Logger) -> None:
    """Imprime el resumen final en consola/log (maximo ~10 lineas).

    Args:
        resultado: Resultado de H2b.
        total_seg: Tiempo total en segundos.
        logger: Logger activo.
    """
    val = resultado["val"]
    logger.info("=== resumen H2b post-proceso (smoke=%s) ===", resultado["smoke"])
    logger.info(
        "val usada n=%d de %d | W1: fijo16=%.4f acc=%.4f",
        val["n"], val["n_original"],
        val["variantes"][REFERENCIA]["macro_f1_espacio_fijo"],
        val["variantes"][REFERENCIA]["accuracy"],
    )
    for nombre in VARIANTES_POST:
        met = val["variantes"][nombre]
        comp = resultado["comparaciones_pareadas_val"][nombre]["delta_macro_f1_espacio_fijo"]
        ab = resultado["consistencia_ab"][nombre]
        logger.info(
            "%s: fijo16=%.4f acc=%.4f | delta=%+.4f IC95=[%+.4f, %+.4f] | "
            "A/B=%+.4f/%+.4f consistente=%s",
            nombre, met["macro_f1_espacio_fijo"], met["accuracy"],
            comp["punto"], comp["ic95_inf"], comp["ic95_sup"],
            ab["delta_mitad_a"], ab["delta_mitad_b"], ab["consistente"],
        )
    logger.info("decision: %s (ganador=%s)", resultado["decision"]["resultado"],
                resultado["decision"]["ganador"])
    test_temporal = resultado.get("test_canonico_y_temporal")
    if test_temporal is not None:
        for bloque in BLOQUES_TEST:
            logger.info(
                "%s: ganador fijo16=%.4f acc=%.4f | W1 fijo16=%.4f acc=%.4f",
                bloque,
                test_temporal[bloque]["ganador"]["macro_f1_espacio_fijo"],
                test_temporal[bloque]["ganador"]["accuracy"],
                test_temporal[bloque]["w1"]["macro_f1_espacio_fijo"],
                test_temporal[bloque]["w1"]["accuracy"],
            )
    logger.info("salidas: %s | %s", resultado["artefactos"]["json"],
                resultado["artefactos"]["figura"])
    logger.info("tiempo total H2b: %.1f s", total_seg)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define y parsea los argumentos de linea de comandos.

    Args:
        argv: Lista de argumentos (None usa sys.argv).

    Returns:
        Namespace con todo o smoke (mutuamente excluyentes y obligatorios).
    """
    parser = argparse.ArgumentParser(
        description=(
            "H2b: post-proceso (sesgos por clase y ensemble) sobre las "
            "probabilidades guardadas de H1/H2, sin reentrenar."
        )
    )
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument(
        "--todo",
        action="store_true",
        help="run completo: val canonica completa, salidas reales en resultados/ y figuras/",
    )
    grupo.add_argument(
        "--smoke",
        action="store_true",
        help="verificacion: submuestra de 800 docs de val y salidas SOLO en /tmp/opencode/h2b_smoke",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Orquesta la ejecucion de H2b segun los flags.

    Args:
        argv: Argumentos CLI (None usa sys.argv).
    """
    args = parse_args(argv)
    os.chdir(RAIZ)
    rutas = rutas_activas(args.smoke)
    if args.smoke:
        Path(CONFIG["dir_smoke"]).mkdir(parents=True, exist_ok=True)
    logger = configurar_logger(rutas["log"])
    random.seed(CONFIG["semilla"])
    np.random.seed(CONFIG["semilla"])
    verificar_entradas(rutas, logger)
    verificar_salidas_nuevas(rutas, args.smoke)
    logger.info(
        "=== INICIO H2b post-proceso M4 | modo=%s smoke=%s ===",
        "smoke" if args.smoke else "todo", args.smoke,
    )
    resultado, total_seg = ejecutar_h2b(args, logger, rutas)
    imprimir_resumen(resultado, total_seg, logger)
    logger.info("=== FIN H2b post-proceso M4 en %.1f s ===", total_seg)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - protocolo: traza completa y salida 1 ante cualquier fallo
        traceback.print_exc()
        sys.exit(1)
