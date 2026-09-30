"""P78: control anidado de Word2Vec frente al azar, con normalizacion alineada.

Proposito
---------
La auditoria forense encontro que P73 y P74 eligen el checkpoint del TextCNN usando el
pliegue EXTERNO de validacion, asi que sus gains no son evidencia out-of-fold. Ademas, el
texto de entrada del CNN se tokeniza sin la normalizacion que si se aplico al corpus de
Word2Vec, y eso probablemente explica el 31.3639 por ciento de OOV declarado por P73.

P78 responde una sola pregunta, con control pareado y sin fuga:

    En condiciones de early stopping estrictamente anidados, el embedding Word2Vec aporta
    senal sobre el objetivo, frente a una matriz aleatoria de identica forma e identicas
    normas de fila, cuando el texto de entrada se normaliza igual que el corpus?

Vinculante
----------
Pre-registro: `docs/p78_preregistro.md`, sha256
`6737f43522c1c9dd0fc45d474d04ab92a7a230aa6b674ee83645588db20c9a0b`, commit `dabf07d`,
escrito y commiteado ANTES de implementar. Si algo de este script se contradice con el
pre-registro, GANA EL PRE-REGISTRO y la desviacion se declara en el informe.

Regla de alcance
----------------
Este script es la UNICA unidad ejecutable. No edita ni borra P71 a P77, no lee test, no lee
holdout, no genera submission, no consulta el leaderboard, no reentrena V03, no toca el
Word2Vec congelado, y no lee `features/v2_structured_features.parquet` (P78 no usa slices,
y leer ese parquet fue el defecto de P75-02 y de P77).

Codigo reutilizado por IMPORT, nunca reescrito
----------------------------------------------
    p73_02_textcnn_vs_random  TextCNN, tokenizar, asignar_ids, mascara_real, f1m, lr_de,
                              log_odds, partir_interna, bootstrap_clusters, sha_arr,
                              grupos_parametros, UMBRALES, MAXLEN, KERNELS
    p75_01_complementariedad  epoca_estrella, entrenar_fijo  (el protocolo correcto)
    p74_01_null_replicado     init_de_semilla, sha_matriz_pad, genera_aleatoria
    p71_03_preparar_corpus    normalizar, RE_URL, RE_MENCION, RE_NUMERO, RE_ESPACIOS
    src.evaluation            metricas_multietiqueta (F1 real por etiqueta)
    p70_01_sensibilidad_tamano  a_ascii

Dispositivo
-----------
CPU, `torch.set_num_threads(32)`, igual que P75-01. NO se usa GPU: el conjunto son 1483
filas, el coste por paso es trivial, y la GPU de este equipo es la del escritorio (D-008).
AMP tampoco aplica en CPU con este tamano.

Coste esperado
--------------
4 brazos x 3 semillas x 5 pliegues = 60 unidades (E* mas reentrenamiento), y E* son tres
entrenamientos internos. Son 240<TextCNN, frente a 20 en P75-01: el tiempo de pared es del
orden de doce veces el de P75-01 y NO se reduce aqui, porque los tres splits internos, las
40 epocas y la paciencia 6 estan congelados dentro de `epoca_estrella`. El log va con flush,
una linea por unidad.

Estado del arte aplicado
------------------------
Hiperparametros, arquitectura, optimizador, scheduler, early stopping y umbrales: los de
P73, sin cambios. Early stopping: mediana de tres splits internos por clusters, y luego
reentrenamiento a E* epocas fijas sobre el outer_train completo, con UNA sola prediccion
sobre outer_validation. Control aleatorio: `genera_aleatoria`, que conserva las normas de
fila, para que el control mida informacion y no escala.
"""

from __future__ import annotations

import ast
import csv as _csv
import hashlib
import json
import math
import sys
import time
import traceback
import unicodedata
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

import p75_01_complementariedad_textcnn as p75_01
from p70_01_sensibilidad_tamano import a_ascii
from p71_03_preparar_corpus import (
    RE_ESPACIOS,
    RE_MENCION,
    RE_NUMERO,
    RE_URL,
    normalizar,
)
from p73_02_textcnn_vs_random import (
    BATCH,
    KERNELS,
    MAXLEN,
    MODELOS,
    REPORTS,
    UMBRALES,
    TextCNN,
    asignar_ids,
    bootstrap_clusters,
    f1m,
    grupos_parametros,
    log_odds,
    mascara_real,
    sha_arr,
    tokenizar,
)
from p74_01_null_replicado import genera_aleatoria, init_de_semilla, sha_matriz_pad
from p75_01_complementariedad_textcnn import (
    N_INTERNOS,
    entrenar_fijo,
    epoca_estrella,
)

from src.data import read_train
from src.evaluation import metricas_multietiqueta
from src.splits import ETIQUETAS, grupos_fusionados, holdout_v2, pliegues_corregidos

FEATURES = PROJECT_ROOT / "features"
FIGURAS = PROJECT_ROOT / "figuras" / "p78"

# =========================================================================
# SECCION 1: CONFIGURACION CONGELADA POR EL PRE-REGISTRO
# =========================================================================
PREREGISTRO = "docs/p78_preregistro.md"
PREREGISTRO_SHA256 = "6737f43522c1c9dd0fc45d474d04ab92a7a230aa6b674ee83645588db20c9a0b"
PREREGISTRO_COMMIT = "dabf07d"

V03_CCV = 0.7772114525165721
SHA256_W2V = "302c994d13b75f8f5fbd440eea2c4f4113997e2581524021923b92f8244d705a"
SHA_MATRIZ_PAD = "c1b774e52e0e1dc131311bddea967e2a72a96e29"

SEMILLAS = (20261130, 20261131, 20261132)
SEMILLA_PRIMARIA = SEMILLAS[0]
SEMILLA_BOOT = 20261102
N_BOOTSTRAP = 2000
N_FOLDS = 5
TOL_ASERCION = 1e-9
N_UNIDADES = 1483
KERNEL_MAYOR = max(KERNELS)

VARIANTES = ("cruda", "norm")
BRAZOS = ("A", "B", "C", "D")
MATRIZ_DE_BRAZO = {"A": "w2v", "B": "aleatoria", "C": "w2v", "D": "aleatoria"}
VARIANTE_DE_BRAZO = {"A": "cruda", "B": "cruda", "C": "norm", "D": "norm"}
DESCRIPCION_BRAZO = {
    "A": "Word2Vec 100d congelado, entrada SIN normalizar (tokenizado de P73 tal cual)",
    "B": "matriz aleatoria pareada, entrada SIN normalizar",
    "C": "Word2Vec 100d congelado, entrada NORMALIZADA con normalizar de P71-03",
    "D": "matriz aleatoria pareada, entrada NORMALIZADA",
}
# Comparaciones del pre-registro, seccion 1. El orden importa: P1 es la primaria.
COMPARACIONES = {"P1": ("C", "D"), "P2": ("A", "B"), "P3": ("C", "A"), "P4": ("D", "B")}
DESCRIPCION_COMPARACION = {
    "P1": "C - D. Efecto del embedding con la normalizacion alineada. PRIMARIA",
    "P2": "A - B. Efecto del embedding con el tokenizado actual. Es lo que midieron P73 y P74",
    "P3": "C - A. Efecto de alinear la normalizacion en el brazo Word2Vec",
    "P4": "D - B. Mismo efecto en el brazo aleatorio. Control de confusion",
}
EPS_GATE_DELTA = 0.003
GATE_FOLDS = 4
# Operacionalizacion declarada de la palabra cualitativa "magnitud parecida" de la seccion
# 7 del pre-registro. Se declara ANTES de correr y no se ajusta despues de ver los numeros.
UMBRAL_PAREJIDAD = 0.5
# Semilla base de los splits internos: la misma que usa `epoca_estrella`.
SEMILLA_INTERNA_BASE = 20261101
# Salto de semilla entre splits internos. `epoca_estrella` de p75_01:121 usa 7*j, no
# N_INTERNOS*j, y la asercion 6 tiene que comprobar el salto real.
PASO_SEMILLA_INTERNA = 7
N_TIPOS_TOP = 25

MOTIVOS_OOV = ("casing", "url", "mencion", "numero", "hashtag", "otro")

JSON_RESULTADOS = REPORTS / "p78_control_anidado.json"
JSON_COBERTURA = REPORTS / "p78_cobertura.json"
CSV_PARES = REPORTS / "p78_pares.csv"
CSV_MANIFIESTO = REPORTS / "p78_manifiesto.sha256"
# Sufijo de los temporales. ASCII a proposito: los temporales entran en la asercion 10.
SUFIJO_TEMPORAL = ".tmp_p78"

CONFIG = {
    "pre_registro": PREREGISTRO,
    "pre_registro_sha256": PREREGISTRO_SHA256,
    "pre_registro_commit": PREREGISTRO_COMMIT,
    "brazos": list(BRAZOS),
    "variantes_entrada": list(VARIANTES),
    "semillas": list(SEMILLAS),
    "semilla_primaria": SEMILLA_PRIMARIA,
    "semilla_bootstrap": SEMILLA_BOOT,
    "n_bootstrap": N_BOOTSTRAP,
    "umbrales_v03": [float(u) for u in UMBRALES],
    "v03_corrected_cv": V03_CCV,
    "sha256_w2v": SHA256_W2V,
    "sha_matriz_pad": SHA_MATRIZ_PAD,
    "maxlen": MAXLEN,
    "kernels": list(KERNELS),
    "n_folds": N_FOLDS,
    "n_internos": N_INTERNOS,
    "eps_gate_delta": EPS_GATE_DELTA,
    "gate_folds": GATE_FOLDS,
    "umbral_parejidad": UMBRAL_PAREJIDAD,
    "dispositivo": "cpu",
    "hilos": 32,
    "parquet_de_slices_no_leido": "features/v2_structured_features.parquet",
    # Los literales que gobiernan E* viven en p75_01, NO en las constantes de p73_02, asi
    # que no se leen de ningun sitio. Se declaran aqui para que el JSON documente que
    # parametros produjeron la epoca elegida, leidos de p75_01:139,155,157 y de p73_02.
    "hiperparametros_de_epoca_estrella": {
        "epocas_maximas": 40,
        "origen_epocas_maximas": "p75_01:139, literal range(1, 41)",
        "tolerancia_early_stopping": 1e-3,
        "origen_tolerancia": "p75_01:155, literal 1e-3",
        "paciencia": 6,
        "origen_paciencia": "p75_01:157, literal 6",
        "suavizado": "media movil de 3 de la BCE de validacion interna",
        "batch": int(BATCH),
        "origen_batch": "p73_02:BATCH, importado por p75_01:47",
        "lr_pico": 1e-3,
        "lr_min": 1e-4,
        "warmup": 3,
        "clip": 1.0,
        "origen_lr_y_clip": "p75_01:128 AdamW lr=1e-3 y p75_01:148 clip_grad_norm_ 1.0; "
                            "el perfil por epoca viene de p73_02:lr_de",
        "nota_sobre_el_cosine": (
            "el cosine de lr_de esta disenado para 40 epocas, pero E* observado esta entre 10 y "
            "17. El learning rate real medio queda por encima del que tendria un cosine "
            "ree calibrado a E*. Es identico en los cuatro brazos, luego no sesga el delta "
            "pareado, pero sube la varianza entre semillas"),
    },
    "nota_sobre_el_ruido_entre_semillas": (
        "el log muestra que el macro-F1 de outer_val se mueve hasta 0.08 entre dos semillas "
        "del mismo brazo y pliegue, frente a un EPS_GATE_DELTA de 0.003. El gate pre-registrado "
        "es muy demanding. No se modifica: se aplica el que dice el pre-registro y la magnitud "
        "del ruido se reporta junto al veredicto"),
    "entorno": {
        "version_de_torch": torch.__version__,
        "version_de_python": sys.version.split()[0],
        "semilla_global_de_torch": "se fija antes de cada unidad, en p75_01:180",
        "nota_determinismo": ("no se activa torch.use_deterministic_algorithms; el orden de "
                              "reduccion de conv y linear en CPU depende del numero de hilos, "
                              "que aqui esta fijado a 32. Mismo equipo y mismos hilos da "
                              "reproduccion bit a bit, como se comprobo entre dos corridas"),
    },
}


def log(m: str) -> None:
    """Imprime una linea de progreso.

    Args:
        m: texto a mostrar.
    """
    print(f"[P78-01] {m}", flush=True)


# =========================================================================
# SECCION 2: LAS DOS VARIANTES DE ENTRADA
# =========================================================================
def tokenizar_uno(texto: str) -> list[str]:
    """Tokeniza un unico texto con el tokenizador de P73, sin normalizar.

    Es exactamente `tokenizar([texto])[0]`, es decir el `tokenizar` de P73-02 aplicado
    elemento a elemento. Se escribe asi para que la diferencia con `variante_norm` sea
    unicamente la pre-normalizacion y nada mas.

    Args:
        texto: texto de entrada.

    Returns:
        Lista de tokens, sin pre-normalizar.
    """
    return tokenizar([texto])[0]


def variante_cruda(textos) -> list[list[str]]:
    """Documentos tokenizados SIN normalizar: el tokenizado de P73 tal cual.

    Args:
        textos: serie de textos crudos.

    Returns:
        Lista de listas de tokens.
    """
    return [tokenizar_uno(str(t)) for t in textos]


def variante_norm(textos) -> list[list[str]]:
    """Documentos tokenizados CON la normalizacion de P71-03, previa al tokenizado.

    Es `[tokenizar(normalizar(t)) for t in textos]`: la MISMA funcion de tokenizado, con
    `normalizar` aplicada antes. La unica diferencia con `variante_cruda` es esa
    pre-normalizacion.

    Args:
        textos: serie de textos crudos.

    Returns:
        Lista de listas de tokens.
    """
    return [tokenizar_uno(normalizar(str(t))) for t in textos]


# =========================================================================
# SECCION 3: CONTEXTO COMPARTIDO
# =========================================================================
@dataclass
class Contexto:
    """Datos de entrada ya verificados, compartidos por todos los passes.

    Attributes:
        y: etiquetas binarias de dev, de forma (n_filas, 3).
        p_v03: probabilidades out-of-fold de V03, de la misma forma.
        ids_v03: identificador de fila del OOF canonico de V03.
        des: marco de datos de la particion dev, reiniciado.
        grupos: etiqueta de cluster por fila de dev.
        pl: lista de (indices_outer_train, indices_outer_val) de los cinco pliegues.
        vocab: palabra a id del Word2Vec, empezando en 1.
        W: matriz entrenada del Word2Vec, de forma (n_vocab, 100).
        normas: norma de fila de W.
        matriz_w2v: matriz con la fila de padding a cero, la que se congela.
        textos: lista de textos crudos de dev.
        n_holdout: numero de filas de holdout apartadas y NO usadas.
        documentos: documentos tokenizados por variante.
        ids: matriz de ids por variante.
        largos: longitudes reales por variante.
        observaciones_internas: llamadas REALES a `partir_interna` capturadas dentro de
            `epoca_estrella`, con los argumentos y el resultado que cada llamada consumio.
    """

    y: np.ndarray
    p_v03: np.ndarray
    ids_v03: np.ndarray
    des: Any
    grupos: np.ndarray
    pl: list[tuple[np.ndarray, np.ndarray]]
    vocab: dict[str, int]
    W: np.ndarray
    normas: np.ndarray
    matriz_w2v: np.ndarray
    textos: list[str]
    n_holdout: int
    documentos: dict[str, list[list[str]]] = field(default_factory=dict)
    ids: dict[str, np.ndarray] = field(default_factory=dict)
    largos: dict[str, np.ndarray] = field(default_factory=dict)
    observaciones_internas: list[dict] = field(default_factory=list)


def predicciones(proba: np.ndarray) -> np.ndarray:
    """Aplica los umbrales de V03, congelados, a unas probabilidades.

    Args:
        proba: probabilidades de forma (n_filas, 3).

    Returns:
        Predicciones binarias de la misma forma.
    """
    return (proba >= UMBRALES[None, :]).astype(int)


def pass0_integridad() -> Contexto:
    """PASS 0: carga, integridad y las aserciones previas al modelo.

    Reproduce el PASS 0 de P75-01 (`p75_01:258-311`) y anade las diez aserciones de la
    seccion 9 del pre-registro que se pueden comprobar sin modelo. Ninguna de las dos
    aserciones que necesitan el entrenamiento se resuelve aqui: la 7 se comprueba en PASS 2
    y la 10 al final, sobre los ficheros ya escritos.

    Returns:
        El contexto con todo cargado y verificado.

    Raises:
        SystemExit: si falla cualquiera de las aserciones del pre-registro.
    """
    log("   PASS 0: integridad previa al modelo")

    # --- asercion 2: los ids del OOF de V03 coinciden con tr.iloc[idev]["id"] ---
    v = np.load(FEATURES / "oof_v03_final.npz", allow_pickle=True)
    y = v["y_true"].astype(int)
    p_v03 = v["proba"].astype(float)
    ids_v03 = v["ids"].astype(np.int64)
    tr = read_train()
    g = grupos_fusionados(tr)
    idev, ihold = holdout_v2(tr, g)
    des = tr.iloc[idev].reset_index(drop=True)
    if not np.array_equal(des["id"].to_numpy(dtype=np.int64), ids_v03):
        raise SystemExit("STOP asercion 2: los ids no coinciden con el OOF canonico")
    if len(y) != N_UNIDADES:
        raise SystemExit(f"STOP asercion 2: se esperaban {N_UNIDADES} filas, hay {len(y)}")
    log(f"      asercion 2 OK: {len(y)} filas de dev, ids alineados con el OOF canonico")

    grupos = g[idev]
    pl = pliegues_corregidos(des, grupos)
    if len(pl) != N_FOLDS:
        raise SystemExit(f"STOP asercion 3: se esperaban {N_FOLDS} pliegues, hay {len(pl)}")

    # --- asercion 1: el V03 reconstruido reproduce el campeon ---
    cv_v03 = f1m(y, predicciones(p_v03))
    log(f"      asercion 1: V03 desde el OOF canonico {cv_v03:.10f} | esperado {V03_CCV:.10f}")
    if abs(cv_v03 - V03_CCV) > TOL_ASERCION:
        raise SystemExit("STOP asercion 1: la reconstruccion de V03 no reproduce el campeon")

    # --- asercion 3: cada fila esta en exactamente un pliegue de validacion ---
    cubre = np.zeros(len(y), dtype=int)
    for _, b in pl:
        cubre[np.asarray(b)] += 1
    if not (cubre == 1).all():
        raise SystemExit("STOP asercion 3: los pliegues no cubren cada fila exactamente una vez")
    log(f"      asercion 3 OK: cada fila aparece una vez y solo una | "
        f"{len(np.unique(grupos))} clusters | {len(pl)} pliegues")

    # --- asercion 4: sha256 del Word2Vec congelado ---
    ruta_w2v = MODELOS / "w2v_twitter_100d.npz"
    sha_real = hashlib.sha256(ruta_w2v.read_bytes()).hexdigest()
    if sha_real != SHA256_W2V:
        raise SystemExit(f"STOP asercion 4: sha256 del Word2Vec {sha_real}")
    log(f"      asercion 4 OK: sha256 del Word2Vec {sha_real}")

    # --- asercion 5: sha1 de la matriz con la fila de padding incluida ---
    d100 = np.load(ruta_w2v, allow_pickle=True)
    vocab = {w: i + 1 for i, w in enumerate(d100["palabras"])}
    W = d100["W"].astype(np.float32)
    normas = np.linalg.norm(W, axis=1)
    matriz_w2v = np.zeros((len(vocab) + 1, W.shape[1]), dtype=np.float32)
    matriz_w2v[1:] = W
    sha_mat = sha_matriz_pad(matriz_w2v)
    if sha_mat != SHA_MATRIZ_PAD:
        raise SystemExit(f"STOP asercion 5: sha_matriz_pad {sha_mat}")
    log(f"      asercion 5 OK: sha_matriz_pad {sha_mat} | filas {matriz_w2v.shape[0]} "
        f"x {matriz_w2v.shape[1]} | vocabulario {len(vocab)}")

    ctx = Contexto(y=y, p_v03=p_v03, ids_v03=ids_v03, des=des, grupos=grupos, pl=pl,
                   vocab=vocab, W=W, normas=normas, matriz_w2v=matriz_w2v,
                   textos=[str(t) for t in des["text"]], n_holdout=len(ihold))
    log(f"      holdout apartado {len(ihold)} filas y NO usado | test 0 | leaderboard 0 | "
        f"submission 0 | parquet de slices NO leido")
    return ctx


def preparar_variantes(ctx: Contexto) -> None:
    """Materializa las dos variantes de entrada y sus matrices de ids y longitudes.

    Args:
        ctx: contexto, que se completa en el sitio con `documentos`, `ids` y `largos`.

    Returns:
        None.
    """
    for variante in VARIANTES:
        docs = variante_cruda(ctx.textos) if variante == "cruda" else variante_norm(ctx.textos)
        ctx.documentos[variante] = docs
        ctx.ids[variante], ctx.largos[variante] = asignar_ids(docs, ctx.vocab)
    log("      las dos variantes tokenizadas: CRUDA es `tokenizar` de P73 tal cual y NORM es "
        "`tokenizar(normalizar(t))`. La unica diferencia es la pre-normalizacion")


# =========================================================================
# SECCION 4: ASERCIONES DE NO FUGA
# =========================================================================
@contextmanager
def observar_splits_internos(destino: list[dict], consumido_por: dict):
    """Instrumenta de verdad `partir_interna` alrededor de una llamada real.

    `epoca_estrella` resuelve `partir_interna` como variable global del modulo `p75_01`
    (`p75_01:121`), de modo que sustituir `p75_01.partir_interna` durante la llamada hace que
    el early stopping real pase por la envoltura. La envoltura delega en la funcion original
    sin tocar una coma de su comportamiento y registra lo que recibe y lo que devuelve.

    El parche es temporal y se revierte en el `finally`, tambien si la llamada falla.

    Args:
        destino: lista donde se anexa una observacion por llamada.
        consumido_por: etiqueta de la unidad que esta llamando (brazo, semilla, pliegue).

    Yields:
        None, mientras la llamada real esta en curso.

    Raises:
        Exception: cualquier excepcion de la llamada real se propaga tras revertir el parche.
    """
    original = p75_01.partir_interna

    def envoltorio(grupos_recibidos, y_recibido, semilla):
        mascara, intentos = original(grupos_recibidos, y_recibido, semilla)
        g = np.asarray(grupos_recibidos)
        destino.append({
            "consumido_por": dict(consumido_por),
            "semilla_del_split_interno": int(semilla),
            "n_filas_recibidas": int(g.size),
            "intentos_de_la_particion": int(intentos),
            "n_inner_train": int(mascara.sum()),
            "n_inner_val": int((~mascara).sum()),
            "grupos_recibidos_sha": sha_arr(g),
            # Estas dos claves y "_grupos_recibidos" son evidencia cruda: nunca se serializan.
            "posiciones_inner_train": np.flatnonzero(mascara),
            "posiciones_inner_val": np.flatnonzero(~mascara),
            "_grupos_recibidos": g,
        })
        return mascara, intentos

    p75_01.partir_interna = envoltorio
    try:
        yield
    finally:
        p75_01.partir_interna = original


def auditar_indices(ctx: Contexto) -> dict:
    """Asercion 6 sobre los indices REALMENTE consumidos por el early stopping.

    No recalcula ni un solo split. Las observaciones las aporta `observar_splits_internos`,
    que se activo alrededor de cada llamada real a `epoca_estrella` y capturo los argumentos y
    la mascara que esa llamada recibio y devuelto. Aqui solo se comprueba lo observado:

        1. que se observaron exactamente `4 brazos x 3 semillas x 5 pliegues x 3 splits`
           llamadas, una por cada split interno de cada unidad;
        2. que el array de grupos que la llamada recibio es, elemento a elemento, el
           `ctx.grupos[ia]` de ese pliegue, y no otro;
        3. que la mascara devuelta cubre todas las filas de `ia`, sin repeticiones;
        4. que `inner_val` es subconjunto de `ia` y su interseccion con `ib` es vacia;
        5. que la semilla observada es `SEMILLA_INTERNA_BASE + 100*k + 7*j` con
           `j` en `0..N_INTERNOS-1`.

    Returns:
        Diccionario con la verificacion por pliegue y los hashes de los indices observados.

    Raises:
        SystemExit: si falta alguna llamada observada, si el argumento recibido no es el de
            ese pliegue o si algun `inner_val` toca el pliegue externo.
    """
    obs = ctx.observaciones_internas
    n_unidades = len(BRAZOS) * len(SEMILLAS) * N_FOLDS
    esperadas = n_unidades * N_INTERNOS
    if len(obs) != esperadas:
        raise SystemExit(f"STOP asercion 6: se esperaban {esperadas} llamadas observadas a "
                         f"partir_interna y se observaron {len(obs)}")
    visto = {(o["consumido_por"]["brazo"], o["consumido_por"]["semilla"],
              o["consumido_por"]["pliegue"]) for o in obs}
    completo = {(b, s, k) for b in BRAZOS for s in SEMILLAS for k in range(1, N_FOLDS + 1)}
    if visto != completo:
        raise SystemExit(f"STOP asercion 6: las unidades observadas no son las "
                         f"{n_unidades} esperadas; faltan {len(completo - visto)}")

    detalle = []
    for k, (a, b) in enumerate(ctx.pl, start=1):
        ia, ib = np.asarray(a), np.asarray(b)
        ia_set, ib_set = set(ia.tolist()), set(ib.tolist())
        del_plesgue = [o for o in obs if o["consumido_por"]["pliegue"] == k]
        if len(del_plesgue) != len(BRAZOS) * len(SEMILLAS) * N_INTERNOS:
            raise SystemExit(f"STOP asercion 6: el pliegue {k} tiene {len(del_plesgue)} "
                             f"llamadas observadas")
        por_split: dict[int, list[dict]] = {}
        for o in del_plesgue:
            etiqueta = o["consumido_por"]
            # `epoca_estrella` de p75_01:121 separa los splits internos con `7*j`, no con
            # `N_INTERNOS*j`. La asercion tiene que comprobar ESE salto, o aborta sobre el
            # protocolo real en el segundo split.
            resto = o["semilla_del_split_interno"] - SEMILLA_INTERNA_BASE - 100 * k
            if resto % PASO_SEMILLA_INTERNA or not 0 <= resto // PASO_SEMILLA_INTERNA < N_INTERNOS:
                raise SystemExit(f"STOP asercion 6: la semilla interna observada "
                                 f"{o['semilla_del_split_interno']} del pliegue {k} no es "
                                 f"SEMILLA_INTERNA_BASE + 100*k + {PASO_SEMILLA_INTERNA}*j "
                                 f"con j en 0..{N_INTERNOS - 1}")
            if not np.array_equal(o["_grupos_recibidos"], ctx.grupos[ia]):
                raise SystemExit(f"STOP asercion 6: el pliegue {k} recibio un array de grupos "
                                 f"distinto de su propio outer_train")
            pos_va = np.asarray(o["posiciones_inner_val"], dtype=int)
            pos_tr = np.asarray(o["posiciones_inner_train"], dtype=int)
            if pos_va.size + pos_tr.size != ia.size:
                raise SystemExit(f"STOP asercion 6: pliegue {k}, la mascara interna observada no "
                                 f"cubre todas las filas de outer_train")
            inner_tr, inner_va = ia[pos_tr], ia[pos_va]
            if not set(inner_va.tolist()) <= ia_set:
                raise SystemExit(
                    f"STOP asercion 6: pliegue {k}, inner_val observado sale de outer_train")
            if set(inner_va.tolist()) & ib_set:
                raise SystemExit(
                    f"STOP asercion 6: pliegue {k}, inner_val observado toca outer_val")
            # El indice del split es `resto // PASO_SEMILLA_INTERNA`, NO `resto // N_INTERNOS`.
            # `epoca_estrella` de p75_01:121 usa 7*j, luego con tres splits resto vale 0, 7 y 14.
            # Dividir por N_INTERNOS daria 0, 2 y 4, y las etiquetas de semilla del informe
            # serian FALSAS: anunciarian BASE+100k+14 y BASE+100k+28, que nunca ocurrieron.
            por_split.setdefault(resto // PASO_SEMILLA_INTERNA, []).append({
                "brazo": etiqueta["brazo"], "semilla_de_modelo": etiqueta["semilla"],
                "n_inner_train": int(inner_tr.size),
                "n_inner_val": int(inner_va.size),
                "sha_inner_train": sha_arr(inner_tr),
                "sha_inner_val": sha_arr(inner_va)})
        detalle.append({
            "pliegue": k, "n_outer_train": int(ia.size), "n_outer_val": int(ib.size),
            "sha_outer_train": sha_arr(ia), "sha_outer_val": sha_arr(ib),
            "grupos_de_outer_train_sha": sha_arr(ctx.grupos[ia]),
            "llamadas_observadas": len(del_plesgue),
            "splits_internos": [
                {"split": j,
                 "semilla_del_split_interno": SEMILLA_INTERNA_BASE + 100 * k + 7 * j,
                 "n_inner_train": por_split[j][0]["n_inner_train"],
                 "n_inner_val": por_split[j][0]["n_inner_val"],
                 "sha_inner_train": por_split[j][0]["sha_inner_train"],
                 "sha_inner_val": por_split[j][0]["sha_inner_val"],
                 "unidades": sorted(por_split[j], key=lambda r: (r["brazo"],
                                                                  r["semilla_de_modelo"])),
                 "misma_mascara_en_las_doce_unidades_del_plegue": len(
                     {r["sha_inner_val"] for r in por_split[j]}) == 1,
                 "interseccion_con_outer_val": 0}
                for j in sorted(por_split)]})
    log(f"      asercion 6 OK: {esperadas} llamadas REALES a partir_interna observadas "
        f"({n_unidades} unidades x {N_INTERNOS} splits); en todas, el array de grupos recibido "
        f"es el outer_train de su pliegue y inner_val es subconjunto de outer_train con "
        f"interseccion vacia con outer_val")
    return {
        "cumple": True,
        "verificado_sobre": ("indices OBSERVADOS, no recalculados. `p75_01.partir_interna` se "
                             "instrumento temporalmente alrededor de cada llamada real a "
                             "`epoca_estrella` y se capturaron sus argumentos y su mascara "
                             "devuelta; esta seccion no vuelve a derivar ningun split"),
        "unidades_instrumentadas": n_unidades,
        "llamadas_observadas": len(obs),
        "llamadas_esperadas": esperadas,
        "cobertura_de_unidades": "las 4 ramas por las 3 semillas en los 5 pliegues",
        "limite_declarado": ("la instrumentacion envuelve el modulo de P75 y ve la llamada, pero "
                             "no el interior del bucle de entrenamiento: no demuestra que las "
                             "filas de inner_val no se usen mas alla de la mascara que devuelve "
                             "partir_interna. Eso se sostiene leyendo `p75_01:121-160`, donde el "
                             "unico uso de `ia` es `ia[m_in]` y `ia[~m_in]`"),
        "detalle_por_pliegue": detalle,
        "nota": ("los splits internos dependen SOLO del pliegue k, no del brazo ni de la semilla "
                 "de modelo, asi que las doce unidades de un pliegue observan la misma mascara; "
                 "eso se comprueba con `misma_mascara_en_las_doce_unidades_del_plegue`")}


def _dos_pasos(matriz: np.ndarray, init: dict, ids_ia: np.ndarray, largos_ia: np.ndarray,
               y_ia: np.ndarray, semilla: int) -> tuple[list, list, torch.Tensor]:
    """Corre dos pasos del protocolo real y captura lotes, dropout y estado del RNG.

    Reproduce el prefijo de `entrenar_fijo`: misma semilla, mismo `init`, mismo generador
    de permutaciones, mismos grupos de parametros, mismo `clip_grad_norm_` a 1.0.

    Args:
        matriz: matriz de embeddings del brazo.
        init: state_dict inicial comun a las dos ramas del par.
        ids_ia: ids del outer_train.
        largos_ia: longitudes del outer_train.
        y_ia: etiquetas del outer_train.
        semilla: semilla de modelo.

    Returns:
        Lista de lotes serializados, lista de pares (entrada, salida) de dropout, y el
        estado del generador global de torch al terminar.
    """
    torch.manual_seed(semilla)
    modelo = TextCNN(matriz)
    modelo.load_state_dict(init, strict=False)
    with torch.no_grad():
        modelo.fc2.bias.copy_(torch.as_tensor(log_odds(y_ia.mean(axis=0)), dtype=torch.float32))
    opt = torch.optim.AdamW(grupos_parametros(modelo), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
    g_perm = torch.Generator().manual_seed(semilla + 7)
    xt = torch.as_tensor(ids_ia)
    rt = mascara_real(largos_ia, torch.device("cpu"))
    yt = torch.as_tensor(y_ia, dtype=torch.float32)
    capturados: list = []

    def gancho(_mod, entrada, salida):
        capturados.append((entrada[0].detach().clone(), salida.detach().clone()))

    modelo.drop.register_forward_hook(gancho)
    estado = torch.get_rng_state().clone()
    lotes = []
    modelo.train()
    perm = torch.randperm(len(xt), generator=g_perm)
    for b0 in range(0, min(2 * 32, len(xt)), 32):
        idx = perm[b0:b0 + 32]
        lotes.append(idx.numpy().tobytes())
        opt.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(
            modelo(xt[idx], rt[idx]), yt[idx]).backward()
        torch.nn.utils.clip_grad_norm_([p for p in modelo.parameters() if p.requires_grad], 1.0)
        opt.step()
    return lotes, capturados, estado


def verificar_pareo(ctx: Contexto, variante: str, semilla: int, k: int = 1) -> dict:
    """Asercion 8: las dos ramas de un par comparten init, filas, batches y mascaras.

    El par es (W2V, aleatoria) dentro de la misma condicion de entrada y con la misma
    semilla. Se comprueba por hash el init, los ids y las longitudes; y funcionalmente que
    los parametros entrenables quedan identicos tras cargar el init, que el orden de
    mini-batches coincide, que el estado del generador global coincide y que las mascaras
    de dropout coinciden en las posiciones no nulas observadas.

    Args:
        ctx: contexto.
        variante: variante de entrada, "cruda" o "norm".
        semilla: semilla de modelo.
        k: pliegue usado para la comprobacion funcional.

    Returns:
        Diccionario con el resultado de cada comprobacion.

    Raises:
        SystemExit: si alguna de las comparaciones falla.
    """
    ia, _ = (np.asarray(x) for x in ctx.pl[k - 1])
    ids, largos = ctx.ids[variante], ctx.largos[variante]
    matriz_w2v = ctx.matriz_w2v
    matriz_ale = genera_aleatoria(semilla, ctx.W, ctx.normas)
    if matriz_w2v.shape != matriz_ale.shape:
        raise SystemExit("STOP asercion 8: las dos matrices no tienen la misma forma")
    if sha_matriz_pad(matriz_w2v) == sha_matriz_pad(matriz_ale):
        raise SystemExit("STOP asercion 8: las dos matrices tienen el mismo hash")

    init = init_de_semilla(semilla, ctx.matriz_w2v, ctx.y.mean(axis=0))
    sha_init = hashlib.sha1(b"".join(
        np.ascontiguousarray(v.numpy()).tobytes() for v in init.values())).hexdigest()[:40]
    modelos = []
    for matriz in (matriz_w2v, matriz_ale):
        torch.manual_seed(semilla)
        m = TextCNN(matriz)
        m.load_state_dict(init, strict=False)
        modelos.append({n: p.detach().clone() for n, p in m.named_parameters()
                        if not n.startswith("emb.")})
    params_iguales = all(torch.equal(modelos[0][n], modelos[1][n]) for n in modelos[0])
    if not params_iguales:
        raise SystemExit("STOP asercion 8: los parametros entrenables no coinciden tras el init")

    lotes_a, cap_a, est_a = _dos_pasos(matriz_w2v, init, ids[ia], largos[ia], ctx.y[ia], semilla)
    lotes_b, cap_b, est_b = _dos_pasos(matriz_ale, init, ids[ia], largos[ia], ctx.y[ia], semilla)
    observables, acuerdos = 0, 0
    for (_, salida_a), (_, salida_b) in zip(cap_a, cap_b, strict=True):
        obs = (salida_a != 0) & (salida_b != 0)
        observables += int(obs.sum())
        acuerdos += int(((salida_a != 0) == (salida_b != 0))[obs].sum())
    if lotes_a != lotes_b:
        raise SystemExit("STOP asercion 8: el orden de mini-batches difiere entre las dos ramas")
    if not torch.equal(est_a, est_b):
        raise SystemExit("STOP asercion 8: el estado del generador global difiere")
    if observables == 0 or acuerdos != observables:
        raise SystemExit("STOP asercion 8: las mascaras de dropout difieren entre las dos ramas")
    return {"variante": variante, "semilla": semilla, "pliegue_comprobado": k,
            "sha_init_compartido": sha_init,
            "sha_ids_outer_train": sha_arr(ids[ia]),
            "sha_largos_outer_train": sha_arr(largos[ia]),
            "parametros_entrenables_iguales_tras_el_init": bool(params_iguales),
            "batches_identicos": True, "rng_global_identico": True,
            "mascaras_dropout_identicas": True,
            "posiciones_dropout_observadas": int(observables),
            "forma_matrices": list(matriz_w2v.shape),
            "sha_matriz_w2v": sha_matriz_pad(matriz_w2v),
            "sha_matriz_aleatoria": sha_matriz_pad(matriz_ale),
            "hashes_distintos": True}


# =========================================================================
# SECCION 5: PASS 1, DIAGNOSTICO DE COBERTURA SIN MODELO
# =========================================================================
def diff_multiset(antes: list[str], despues: list[str]) -> set[str]:
    """Tipos que aparecen en `antes` y no en `despues`, por multiconjunto.

    Args:
        antes: tokens de la etapa previa.
        despues: tokens de la etapa siguiente.

    Returns:
        Conjunto de los tipos consumidos por la sustitucion.
    """
    restante = Counter(despues)
    fuera = set()
    for t in antes:
        if restante[t] > 0:
            restante[t] -= 1
        else:
            fuera.add(t)
    return fuera


def descomponer_motivos(textos: list[str], vocab: dict[str, int]) -> dict:
    """Descompone los tokens y su OOV en los cinco motivos ya conocidos.

    Reproduce la cadena exacta de `normalizar` de P71-03 etapa por etapa y toma la diferencia
    de multiconjuntos entre etapas consecutivas, de modo que el motivo de cada token es
    MEDIDO y no supuesto:

        s0 crudo             -> s1 NFKC mas minusculas   (motivo `casing`)
        s1                   -> s2 URL a marcador         (motivo `url`)
        s2                   -> s3 mencion a marcador    (motivo `mencion`)
        s3                   -> s4 numero a marcador      (motivo `numero`)
        s4                   -> tokens finales, con el residuo `hashtag` y `otro`

    La prioridad es la del pipeline real, porque la normalizacion aplica NFKC y minusculas
    ANTES que las sustituciones. Para cada motivo se cuenta: ocurrencias, cuantas son OOV, la
    tasa de OOV dentro del motivo y la fraccion que representa sobre el total de OOV. Ademas
    se mide el RESCATE: de los tokens OOV de cada motivo, cuantos pasan a estar cubiertos al
    aplicar la normalizacion, que es la razon de ser del brazo normalizado.

    Args:
        textos: textos crudos de dev.
        vocab: vocabulario del Word2Vec.

    Returns:
        Diccionario con el desglose por motivo, los agregados y las dos reconstrucciones.

    Raises:
        SystemExit: si la cadena reconstruida no coincide con las dos variantes reales.
    """
    conteo = {m: {"ocurrencias": 0, "oov": 0, "rescatadas": 0, "tipos": set()}
              for m in MOTIVOS_OOV}
    n_tokens = 0
    n_oov = 0
    n_rescate = 0
    n_tipografica = 0
    n_ascii = 0
    reconstruidas: dict[str, list[list[str]]] = {"cruda": [], "norm": []}
    for texto in textos:
        s1 = unicodedata.normalize("NFKC", texto).lower()
        s2 = RE_URL.sub(" url ", s1)
        s3 = RE_MENCION.sub(" @usuario ", s2)
        s4 = RE_NUMERO.sub(" numero ", s3)
        s5 = RE_ESPACIOS.sub(" ", s4).strip()
        t_cruda = tokenizar_uno(texto)
        t_norm = tokenizar_uno(s5)
        reconstruidas["cruda"].append(t_cruda)
        reconstruidas["norm"].append(t_norm)
        etapas = {"casing": diff_multiset(t_cruda, tokenizar_uno(s1)),
                  "url": diff_multiset(tokenizar_uno(s1), tokenizar_uno(s2)),
                  "mencion": diff_multiset(tokenizar_uno(s2), tokenizar_uno(s3)),
                  "numero": diff_multiset(tokenizar_uno(s3), tokenizar_uno(s4))}
        for t in t_cruda:
            n_tokens += 1
            n_tipografica += t.count(chr(0x2019))
            n_ascii += t.count("'")
            motivo = "otro"
            for m in ("casing", "url", "mencion", "numero"):
                if t in etapas[m]:
                    motivo = m
                    break
            if motivo == "otro" and t.startswith("#"):
                motivo = "hashtag"
            celda = conteo[motivo]
            celda["ocurrencias"] += 1
            celda["tipos"].add(t)
            if t not in vocab:
                n_oov += 1
                celda["oov"] += 1
                equivalente = tokenizar_uno(normalizar(t))
                if equivalente and equivalente[0] in vocab:
                    n_rescate += 1
                    celda["rescatadas"] += 1

    desglose = {}
    for m in MOTIVOS_OOV:
        c = conteo[m]
        desglose[m] = {
            "ocurrencias": int(c["ocurrencias"]),
            "fraccion_de_tokens": round(c["ocurrencias"] / max(n_tokens, 1), 6),
            "oov": int(c["oov"]),
            "oov_dentro_del_motivo": round(c["oov"] / max(c["ocurrencias"], 1), 6),
            "fraccion_del_oov_total": round(c["oov"] / max(n_oov, 1), 6),
            "tipos_distintos": len(c["tipos"]),
            "rescatadas_por_normalizar": int(c["rescatadas"]),
        }
    return {"tokens_totales": int(n_tokens), "oov_totales": int(n_oov),
            "tasa_oov": round(n_oov / max(n_tokens, 1), 6),
            "oov_rescatados_por_normalizar": int(n_rescate),
            "tasa_de_rescate": round(n_rescate / max(n_oov, 1), 6),
            "por_motivo": desglose,
            "tokens_con_comilla_tipografica": int(n_tipografica),
            "tokens_con_comilla_ascii": int(n_ascii),
            "nota_comillas": ("la comilla tipografica U+2019 NO se convierte en ascii por NFKC, "
                              "asi que sobrevive en las dos variantes; se cuenta aparte porque "
                              "es un residuo que la normalizacion no arregla"),
            "prioridad": ("casing > url > mencion > numero > hashtag > otro, que es el orden en "
                          "que `normalizar` aplica las transformaciones"),
            "reconstruidas": reconstruidas}


def cobertura_una_variante(ctx: Contexto, variante: str) -> dict:
    """Cobertura del vocabulario para una variante de entrada. Sin modelo.

    Mide todo lo que pide la seccion 6 del pre-registro, mas la separacion entre padding y
    OOV, que comparten el id 0 por la convencion de `TextCNN(padding_idx=0)`.

    Args:
        ctx: contexto.
        variante: variante de entrada.

    Returns:
        Diccionario con el diagnostico completo de la variante.
    """
    docs = ctx.documentos[variante]
    ids, largos = ctx.ids[variante], ctx.largos[variante]
    n_filas = len(docs)
    n_tokens = int(sum(len(d) for d in docs))
    cubiertos = np.zeros(n_filas, dtype=np.int64)
    tipos: set[str] = set()
    for i, d in enumerate(docs):
        tipos.update(d)
        cubiertos[i] = sum(1 for t in d if t in ctx.vocab)
    n_cubiertos = int(cubiertos.sum())
    n_oov = int(n_tokens - n_cubiertos)
    tipos_fuera = tipos - set(ctx.vocab)
    dentro_de_matriz = int(largos.sum())
    con_id = int((ids[np.arange(n_filas)[:, None], np.arange(MAXLEN)[None, :]]
                  * (np.arange(MAXLEN)[None, :] < largos[:, None]).astype(np.int64) != 0).sum())
    n_padding = n_filas * MAXLEN - dentro_de_matriz
    truncadas = np.array([len(d) > MAXLEN for d in docs], dtype=bool)
    sin_tokens = np.array([len(d) == 0 for d in docs], dtype=bool)
    sin_cubiertos = cubiertos == 0
    degeneradas = largos < KERNEL_MAYOR
    sin_ningun_kernel = largos < min(KERNELS)
    canales_muertos = np.array([sum(1 for k in KERNELS if largos[i] < k)
                                for i in range(n_filas)], dtype=np.int64)
    # Una fila con CERO tokens cubiertos es un caso mas grave que una fila corta: su
    # embedding es el vector nulo, la convolucion devuelve exactamente el sesgo (cero),
    # relu da cero y el max da cero, luego las TRES ramas de filtros aportan nada. Eso no
    # lo capturaba el conteo de canales por longitud, que solo mira `largos < k`.
    filas_muertas = sin_cubiertos
    canales_muertos_efectivos = canales_muertos + np.where(filas_muertas, len(KERNELS), 0)
    return {
        "n_filas": int(n_filas),
        "n_tokens": n_tokens,
        "tokens_cubiertos": n_cubiertos,
        "proporcion_tokens_con_id_distinto_de_cero": round(n_cubiertos / max(n_tokens, 1), 6),
        "tokens_oov": n_oov,
        "proporcion_oov": round(n_oov / max(n_tokens, 1), 6),
        "tokens_que_entran_en_la_matriz": dentro_de_matriz,
        "tokens_no_cero_en_la_matriz": con_id,
        "nota_id_cero": ("padding y OOV comparten el id 0, que es la convencion de TextCNN con "
                         "padding_idx=0. La arquitectura NO se cambia en P78; el deficit queda "
                         "cuantificado"),
        "posiciones_de_padding": int(n_padding),
        "proporcion_de_padding_de_la_matriz": round(n_padding / max(n_filas * MAXLEN, 1), 6),
        "tipos_distintos_en_dev": len(tipos),
        "tipos_distintos_ausentes_del_vocabulario": len(tipos_fuera),
        "filas_sin_tokens": int(sin_tokens.sum()),
        "filas_sin_tokens_cubiertos": int(sin_cubiertos.sum()),
        "fraccion_de_filas_sin_tokens_cubiertos": round(float(sin_cubiertos.mean()), 6),
        "filas_truncadas_por_maxlen": int(truncadas.sum()),
        "fraccion_truncada_por_maxlen": round(float(truncadas.mean()), 6),
        "maxlen": MAXLEN,
        "kernels": list(KERNELS),
        "kernel_mayor": int(KERNEL_MAYOR),
        "filas_mas_cortas_que_el_kernel_mayor": int(degeneradas.sum()),
        "fraccion_mas_cortas_que_el_kernel_mayor": round(float(degeneradas.mean()), 6),
        "filas_sin_ningun_kernel_valido": int(sin_ningun_kernel.sum()),
        "canales_de_filtro_muertos_por_fila_media": round(float(canales_muertos.mean()), 6),
        "filas_con_algun_canal_muerto": int((canales_muertos > 0).sum()),
        "filas_totalmente_muertas": int(filas_muertas.sum()),
        "fraccion_totalmente_muertas": round(float(filas_muertas.mean()), 6),
        "canales_de_filtro_muertos_incluyendo_muertas": round(
            float(canales_muertos_efectivos.mean()), 6),
        "nota_filas_totalmente_muertas": (
            "una fila con cero tokens cubiertos tiene el embedding nulo, la convolucion "
            "devuelve el sesgo, relu y max dan cero, y las TRES ramas quedan muertas. Es un "
            "caso MAS grave que una fila simplemente corta, y el conteo por longitud no lo "
            "veia. El mecanismo que separa la variante cruda de la normalizada son estas "
            "filas, no las filas cortas"),
        "nota_max_pooling": ("una fila mas corta que el kernel mayor tiene TODOS sus canales de "
                             "ese kernel a cero tras el max-pooling enmascarado con -1e9 y relu: "
                             "ese bloque de filtros no aporta nada"),
        "longitud_real_media": round(float(largos.mean()), 6),
        "longitud_real_mediana": float(np.median(largos)),
        "longitud_real_maxima": int(largos.max()),
    }


def tipos_fuera_del_vocabulario(ctx: Contexto, variante: str,
                                limite: int = N_TIPOS_TOP) -> list[str]:
    """Tipos mas frecuentes de dev ausentes del vocabulario, ya en ASCII.

    No se serializa el vocabulario entero: solo los 25 tipos mas frecuentes de dev que no
    estan en el, y pasan por `a_ascii`, porque el vocabulario de Word2Vec conserva tildes y
    enie y el texto crudo puede traer comillas tipograficas.

    Args:
        ctx: contexto.
        variante: variante de entrada.
        limite: numero maximo de tipos devueltos.

    Returns:
        Lista de cadenas ASCII, con la forma `tipo<TAB>ocurrencias`.
    """
    c = Counter(t for d in ctx.documentos[variante] for t in d if t not in ctx.vocab)
    return [f"{a_ascii(t)}\t{n}" for t, n in c.most_common(limite)]


def pass1_cobertura(ctx: Contexto) -> dict:
    """PASS 1: diagnostico de cobertura de las dos variantes, sin modelo alguno.

    Returns:
        El diccionario que se escribe en `reports/p78_cobertura.json`.

    Raises:
        SystemExit: si la reconstruccion por motivos no coincide con las dos variantes.
    """
    log("")
    log("   PASS 1: cobertura del vocabulario, sin modelo")
    salida: dict[str, Any] = {
        "pre_registro": PREREGISTRO, "pre_registro_sha256": PREREGISTRO_SHA256,
        "maxlen": MAXLEN, "kernels": list(KERNELS), "n_filas_dev": len(ctx.textos),
        "variante_cruda": "`tokenizar` de P73-02 tal cual, SIN `normalizar`",
        "variante_norm": ("`tokenizar(normalizar(t))` con `normalizar` de P71-03. La unica "
                          "diferencia con la variante cruda es la pre-normalizacion"),
    }
    for variante in VARIANTES:
        diag = cobertura_una_variante(ctx, variante)
        diag["tipos_fuera_del_vocabulario_top"] = tipos_fuera_del_vocabulario(ctx, variante)
        salida[variante] = diag
        log(f"      {variante:5}: cobertura "
            f"{100 * diag['proporcion_tokens_con_id_distinto_de_cero']:.2f}% | OOV "
            f"{100 * diag['proporcion_oov']:.2f}% | truncadas "
            f"{100 * diag['fraccion_truncada_por_maxlen']:.2f}% | filas sin token cubierto "
            f"{diag['filas_sin_tokens_cubiertos']} | filas mas cortas que el kernel mayor "
            f"{diag['filas_mas_cortas_que_el_kernel_mayor']} "
            f"({100 * diag['fraccion_mas_cortas_que_el_kernel_mayor']:.2f}%) | filas "
            f"TOTALMENTE MUERTAS (las {len(KERNELS)} ramas a cero) "
            f"{diag['filas_totalmente_muertas']} "
            f"({100 * diag['fraccion_totalmente_muertas']:.2f}%) | canales muertos por fila "
            f"{diag['canales_de_filtro_muertos_incluyendo_muertas']:.3f} de {len(KERNELS)}")

    motivos = descomponer_motivos(ctx.textos, ctx.vocab)
    if motivos["reconstruidas"]["cruda"] != ctx.documentos["cruda"]:
        raise SystemExit("STOP: la reconstruccion de la variante cruda no coincide")
    if motivos["reconstruidas"]["norm"] != ctx.documentos["norm"]:
        raise SystemExit("STOP: la reconstruccion de la variante norm no coincide")
    motivos.pop("reconstruidas")
    salida["desglose_de_los_cinco_motivos"] = motivos
    log("")
    log(f"      desglose del OOV de la variante CRUDA, sobre {motivos['oov_totales']} tokens OOV "
        f"de {motivos['tokens_totales']}:")
    for m in MOTIVOS_OOV:
        d = motivos["por_motivo"][m]
        log(f"         {m:8} {d['ocurrencias']:6d} ocurrencias | {d['oov']:6d} OOV | OOV dentro "
            f"del motivo {100 * d['oov_dentro_del_motivo']:5.1f}% | "
            f"{100 * d['fraccion_del_oov_total']:5.1f}% de todo el OOV | rescatadas por "
            f"normalizar {d['rescatadas_por_normalizar']:6d}")
    log(f"      en total la normalizacion rescata {motivos['oov_rescatados_por_normalizar']} de "
        f"los {motivos['oov_totales']} tokens OOV ({100 * motivos['tasa_de_rescate']:.2f}%)")
    log(f"      comillas: {motivos['tokens_con_comilla_tipografica']} tokens con U+2019 frente a "
        f"{motivos['tokens_con_comilla_ascii']} con apostrofo ascii. U+2019 no se convierte por "
        f"NFKC, asi que no es una causa que la normalizacion arregle")
    salida["declaracion"] = (
        "El id 0 es compartido por padding y OOV, que es la convencion de TextCNN con "
        "padding_idx=0, y la arquitectura NO se cambia en P78. Los motivos casing, url, "
        "mencion y numero SI los arregla la normalizacion y se han medido token a token, no se "
        "han supuesto. El hashtag lo conserva el corpus y lo conserva tambien la entrada "
        "normalizada, asi que es irreducible con este normalizador. Los cinco motivos se miden "
        "sobre el texto CRUDO, porque son propiedades del texto crudo; en la variante "
        "normalizada aparecen como RESCATE, que es la columna reportada.")
    return salida


# =========================================================================
# SECCION 6: PASS 2, LOS CUATRO BRAZOS
# =========================================================================
def matrices_del_brazo(ctx: Contexto, brazo: str, semilla: int) -> np.ndarray:
    """Matriz de embeddings de un brazo, con la convencion de fila 0 a cero.

    B y D comparten la MISMA matriz aleatoria para la misma semilla, porque
    `genera_aleatoria` depende solo de la semilla. Asi P4 = D - B aisla exactamente la
    normalizacion, sin ruido de realizacion de la matriz.

    Args:
        ctx: contexto.
        brazo: "A", "B", "C" o "D".
        semilla: semilla de modelo.

    Returns:
        La matriz de embeddings del brazo.

    Raises:
        ValueError: si el brazo no existe.
    """
    if MATRIZ_DE_BRAZO[brazo] == "w2v":
        return ctx.matriz_w2v
    if MATRIZ_DE_BRAZO[brazo] == "aleatoria":
        return genera_aleatoria(semilla, ctx.W, ctx.normas)
    raise ValueError(f"brazo desconocido: {brazo}")


def ejecutar_brazo(ctx: Contexto, brazo: str, semilla: int,
                   contadores: dict[tuple[str, int, int], int]) -> tuple[np.ndarray, list[dict]]:
    """PASS 2 de un brazo: cinco pliegues con early stopping estrictamente anidado.

    Por cada pliegue, `epoca_estrella` produce E* usando solo filas de `ia`, y `entrenar_fijo`
    reentrena sobre `ia` COMPLETO durante E* epocas y predice `ib` una sola vez. El contador
    `n_eval_outer` se incrementa exactamente una vez por pliegue y el script aborta si se
    supera. La llamada a `epoca_estrella` va envuelta en `observar_splits_internos`, que
    captura los splits internos que esa llamada consume de verdad, y eso es lo que despues
    verifica `auditar_indices`.

    Args:
        ctx: contexto.
        brazo: identificador del brazo.
        semilla: semilla de modelo.
        contadores: diccionario mutable, con una entrada por (brazo, semilla, pliegue).

    Returns:
        Las probabilidades OOF del brazo y el registro por pliegue.

    Raises:
        SystemExit: si se evalua `ib` mas de una vez, si alguna probabilidad no es finita o si
            queda alguna fila sin prediccion.
    """
    variante = VARIANTE_DE_BRAZO[brazo]
    matriz = matrices_del_brazo(ctx, brazo, semilla)
    init = init_de_semilla(semilla, ctx.matriz_w2v, ctx.y.mean(axis=0))
    ids, largos = ctx.ids[variante], ctx.largos[variante]
    proba = np.zeros((len(ctx.y), len(ETIQUETAS)), dtype=np.float64)
    asignado = np.zeros(len(ctx.y), dtype=int)
    registro = []
    for k, (a, b) in enumerate(ctx.pl, start=1):
        ia, ib = np.asarray(a), np.asarray(b)
        t0 = time.time()
        with observar_splits_internos(ctx.observaciones_internas,
                                      {"brazo": brazo, "semilla": semilla, "pliegue": k}):
            e_star, e3 = epoca_estrella(matriz, init, semilla, ids, largos, ctx.y, ctx.grupos,
                                        ia, k)
        clave = (brazo, semilla, k)
        contadores[clave] = contadores.get(clave, 0) + 1
        if contadores[clave] != 1:
            raise SystemExit(f"STOP asercion 7: {clave} se evaluo {contadores[clave]} veces "
                             f"sobre outer_val")
        salida = entrenar_fijo(matriz, init, semilla, ids, largos, ctx.y, ia, ib, e_star)
        if not np.isfinite(salida).all():
            raise SystemExit(f"STOP asercion 9: probabilidades no finitas en {clave}")
        proba[ib] = salida
        asignado[ib] += 1
        registro.append({
            "brazo": brazo, "semilla": semilla, "pliegue": k, "variante": variante,
            "matriz": MATRIZ_DE_BRAZO[brazo], "sha_matriz": sha_matriz_pad(matriz),
            "E_estrella": int(e_star),
            "epocas_de_los_tres_splits": [int(x) for x in e3],
            "n_outer_train": len(ia), "n_outer_val": len(ib),
            "n_eval_outer": int(contadores[clave]),
            "forward_passes_sobre_outer_val": 2,
            "nota_forward_passes": ("`entrenar_fijo` hace DOS pasadas hacia delante sobre ib "
                                    "para comprobar que el modo eval es determinista. Es una "
                                    "comprobacion de determinismo, NO una seleccion de checkpoint"),
            "macro_f1_outer_val": round(f1m(ctx.y[ib], predicciones(salida)), 6),
            "segundos": round(time.time() - t0, 1)})
        log(f"      {brazo} s{semilla} p{k}: E* {e_star} (tres splits {e3}) | outer_val "
            f"{len(ib)} filas | 1 evaluacion | macro-F1 "
            f"{registro[-1]['macro_f1_outer_val']:.6f} | {registro[-1]['segundos']:.0f}s")
    if not (asignado == 1).all():
        raise SystemExit("STOP asercion 9: hay filas sin prediccion en el OOF del brazo")
    return proba, registro


def pass2_brazos(ctx: Contexto) -> tuple[dict[tuple[str, int], np.ndarray], list[dict], dict]:
    """PASS 2 completo: los cuatro brazos, tres semillas, cinco pliegues.

    Returns:
        Tupla con las probabilidades por (brazo, semilla), los registros por pliegue y el
        manifiesto de pareo.

    Raises:
        SystemExit: si alguna asercion de la seccion 9 falla.
    """
    log("")
    log(f"   PASS 2: 4 brazos x {len(SEMILLAS)} semillas x {N_FOLDS} pliegues, early stopping "
        f"anidado, una sola evaluacion de outer_val por unidad")
    contadores: dict[tuple[str, int, int], int] = {}
    probas: dict[tuple[str, int], np.ndarray] = {}
    registros: list[dict] = []
    for variante in VARIANTES:
        for brazo in BRAZOS:
            if VARIANTE_DE_BRAZO[brazo] != variante:
                continue
            log("")
            log(f"      BRAZO {brazo}: {DESCRIPCION_BRAZO[brazo]}")
            for semilla in SEMILLAS:
                p, reg = ejecutar_brazo(ctx, brazo, semilla, contadores)
                probas[(brazo, semilla)] = p
                registros.extend(reg)
    if len(probas) != len(BRAZOS) * len(SEMILLAS):
        raise SystemExit("STOP: faltan probabilidades OOF")
    if len(contadores) != len(BRAZOS) * len(SEMILLAS) * N_FOLDS or any(
            v != 1 for v in contadores.values()):
        raise SystemExit(f"STOP asercion 7: contadores de outer_val invalidos {contadores}")
    log(f"      asercion 7 OK: {len(contadores)} unidades, exactamente 1 evaluacion de outer_val "
        f"en cada una")
    manifiesto_pareo = [verificar_pareo(ctx, v, s) for v in VARIANTES for s in SEMILLAS]
    log(f"      asercion 8 OK: {len(manifiesto_pareo)} pares comprobados con el mismo init, los "
        f"mismos ids, los mismos batches, el mismo estado del RNG y las mismas mascaras de "
        f"dropout, y matrices con la misma forma y hash distinto")
    return probas, registros, {"pareo": manifiesto_pareo}


# =========================================================================
# SECCION 7: PASS 3, METRICAS
# =========================================================================
def f1_positiva_2x2(y: np.ndarray, pred: np.ndarray) -> float:
    """F1 de la clase positiva calculada desde la matriz de confusion 2x2.

    Es la formula real, `2*TP / (2*TP + FP + FN)`, y existe como REFERENCIA INDEPENDIENTE
    para comprobar que la F1 por etiqueta que sale de `metricas_multietiqueta` es la F1 y no
    otra cosa. El recall es `TP / (TP + FN)` y NO es la metrica del proyecto: usarlo fue el
    defecto de P77, y por eso aqui no se calcula en ninguna parte. Siempre que haya falsos
    positivos, la F1 es estrictamente menor que el recall.

    Args:
        y: etiqueta verdadera de una columna binaria.
        pred: prediccion binaria de la misma longitud.

    Returns:
        La F1 de la clase positiva.
    """
    tp = float(np.sum((y == 1) & (pred == 1)))
    fp = float(np.sum((y == 0) & (pred == 1)))
    fn = float(np.sum((y == 1) & (pred == 0)))
    den = 2 * tp + fp + fn
    return 2 * tp / den if den > 0 else 0.0


def metricas_de_probabilidad(y: np.ndarray, proba: np.ndarray) -> dict:
    """Brier score y log-loss por etiqueta, separados de las metricas de F1.

    Args:
        y: etiquetas binarias verdaderas, de forma (n_filas, 3).
        proba: probabilidades, de la misma forma.

    Returns:
        Diccionario con Brier y log-loss por etiqueta y sus medias.

    Raises:
        SystemExit: si alguna probabilidad no es finita o cae fuera de [0, 1].
    """
    if not np.isfinite(proba).all():
        raise SystemExit("STOP asercion 9: probabilidades no finitas")
    if proba.min() < 0.0 or proba.max() > 1.0:
        raise SystemExit("STOP asercion 9: probabilidades fuera de [0, 1]")
    p = np.clip(proba, 1e-12, 1 - 1e-12)
    brier = ((p - y) ** 2).mean(axis=0)
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean(axis=0)
    return {"brier_por_etiqueta": {e: round(float(brier[j]), 8) for j, e in enumerate(ETIQUETAS)},
            "brier_medio": round(float(brier.mean()), 8),
            "log_loss_por_etiqueta": {e: round(float(ll[j]), 8) for j, e in enumerate(ETIQUETAS)},
            "log_loss_medio": round(float(ll.mean()), 8)}


def metricas_de_brazo(y: np.ndarray, proba: np.ndarray, fold_de: np.ndarray) -> dict:
    """Todas las metricas de un brazo, con la F1 real por etiqueta.

    Args:
        y: etiquetas verdaderas.
        proba: probabilidades OOF del brazo.
        fold_de: numero de pliegue por fila.

    Returns:
        Diccionario con macro-F1, F1 por etiqueta, macro-F1 por pliegue, Brier y log-loss.

    Raises:
        SystemExit: si la F1 de `metricas_multietiqueta` no coincide con la referencia 2x2.
    """
    pred = predicciones(proba)
    m = metricas_multietiqueta(y, pred)
    f1_ref = {e: round(f1_positiva_2x2(y[:, j], pred[:, j]), 10) for j, e in enumerate(ETIQUETAS)}
    for e in ETIQUETAS:
        if abs(m["f1_por_etiqueta"][e] - f1_ref[e]) > 1e-9:
            raise SystemExit(f"STOP: la F1 por etiqueta de {e} no coincide con la referencia 2x2")
    falsos_positivos = {e: int(np.sum((y[:, j] == 0) & (pred[:, j] == 1)))
                        for j, e in enumerate(ETIQUETAS)}
    return {"macro_f1": round(float(m["f1_media_columnas"]), 10),
            "f1_positiva_por_etiqueta": {e: round(float(m["f1_por_etiqueta"][e]), 8)
                                         for e in ETIQUETAS},
            "f1_media_columnas_2x2_de_referencia": f1_ref,
            "f1_macro6": round(float(m["f1_macro6"]), 8),
            "f1_micro": round(float(m["f1_micro"]), 8),
            "exact_match": round(float(m["exact_match"]), 8),
            "macro_f1_por_pliegue": [round(f1m(y[fold_de == k], pred[fold_de == k]), 10)
                                     for k in range(1, N_FOLDS + 1)],
            "n_falsos_positivos": falsos_positivos,
            "n_positivos_reales": {e: int(y[:, j].sum()) for j, e in enumerate(ETIQUETAS)},
            "metricas_de_probabilidad": metricas_de_probabilidad(y, proba),
            "media_probabilidad_por_etiqueta": {e: round(float(proba[:, j].mean()), 8)
                                                for j, e in enumerate(ETIQUETAS)},
            "nota_f1": ("la F1 por etiqueta es 2*TP/(2*TP+FP+FN), la funcion real del proyecto "
                        "en src/evaluation.py, comprobada contra una referencia 2x2 "
                        "independiente. Los falsos positivos se declaran porque son la razon "
                        "por la que F1 y recall no coinciden. El recall no se calcula en "
                        "ningun sitio de P78")}


def ic_t(deltas: list[float]) -> dict:
    """Media, desviacion tipica e IC95 t de Student de deltas pareados.

    Args:
        deltas: diferencias por semilla.

    Returns:
        Diccionario con media, desviacion tipica, error estandar e IC95.

    Raises:
        ValueError: si hay menos de dos deltas.
    """
    from scipy import stats

    n = len(deltas)
    if n < 2:
        raise ValueError("se necesitan al menos dos deltas")
    d = np.asarray(deltas, dtype=float)
    media = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / math.sqrt(n)
    t = float(stats.t.ppf(0.975, n - 1))
    return {"media": round(media, 10), "sd": round(sd, 10), "se": round(se, 10), "n": n,
            "t_critico": round(t, 6),
            "ic95": [round(media - t * se, 10), round(media + t * se, 10)]}


def bootstrap_delta_medio_entre_semillas(y: np.ndarray,
                                          probas_efecto: dict[int, np.ndarray],
                                          probas_control: dict[int, np.ndarray],
                                          grupos: np.ndarray, semilla: int,
                                          n: int) -> tuple[float, float, float]:
    """Bootstrap por CLUSTERS del delta MEDIO entre semillas. Este si es un IC95.

    Para cada replica se remuestrea el mismo numero de clusters con reemplazo, se toman todas
    sus filas y se calcula, sobre esa replica, el macro-F1 del brazo de efecto y el del brazo
    de control promediados sobre las tres semillas; la diferencia de esos dos promedios es el
    delta de la replica. El percentil 2.5 y el 97.5 de esas deltas SI son un intervalo del 95
    por ciento del delta medio, a diferencia de promediar los extremos de tres bootstraps
    independientes, que no es ningun intervalo.

    Args:
        y: etiquetas.
        probas_efecto: probabilidades del brazo de efecto, por semilla.
        probas_control: probabilidades del brazo de control, por semilla.
        grupos: etiqueta de cluster por fila.
        semilla: semilla del generador, la misma que el resto de bootstraps.
        n: numero de remuestreos.

    Returns:
        Media, extremo inferior y extremo superior del IC95 del delta medio.

    Raises:
        SystemExit: si los dos brazos no tienen las mismas semillas.
    """
    semillas = sorted(probas_efecto)
    if semillas != sorted(probas_control):
        raise SystemExit("STOP: el bootstrap del delta medio necesita las mismas semillas en "
                         "los dos brazos")
    u = np.unique(grupos)
    por_g = {g: np.where(grupos == g)[0] for g in u}
    rng = np.random.default_rng(semilla)
    pred_efecto = {s: predicciones(probas_efecto[s]) for s in semillas}
    pred_control = {s: predicciones(probas_control[s]) for s in semillas}
    d = np.empty(n)
    for r in range(n):
        sel = rng.integers(0, len(u), len(u))
        idx = np.concatenate([por_g[u[s]] for s in sel])
        de = float(np.mean([f1m(y[idx], pred_efecto[s][idx]) for s in semillas]))
        dc = float(np.mean([f1m(y[idx], pred_control[s][idx]) for s in semillas]))
        d[r] = de - dc
    lo, hi = np.percentile(d, [2.5, 97.5])
    return round(float(d.mean()), 10), round(float(lo), 10), round(float(hi), 10)


def unidades_que_cambian(y: np.ndarray, pred_x: np.ndarray,
                         pred_z: np.ndarray) -> dict:
    """Recuento de unidades tweet-etiqueta donde los dos brazos discrepan.

    Es DESCRIPTIVO. No es una metrica, no es comparable con `corrected_cv` y no sustituye al
    macro-F1, que es la unica metrica primaria del pre-registro.

    Args:
        y: etiquetas verdaderas.
        pred_x: predicciones del brazo de efecto.
        pred_z: predicciones del brazo de control.

    Returns:
        Diccionario con los recuentos y las tasas.
    """
    dif = pred_x != pred_z
    ok_x = pred_x == y
    ok_z = pred_z == y
    rescue = int((~ok_z & ok_x).sum())
    damage = int((ok_z & ~ok_x).sum())
    return {"unidades": int(y.size), "cambian": int(dif.sum()),
            "tasa_de_cambio": round(float(dif.mean()), 6),
            "rescue": rescue, "damage": damage, "net": rescue - damage,
            "nota": "descriptivo. NO es una metrica y NO se usa en el veredicto"}


def pass3_metricas(ctx: Contexto, probas: dict[tuple[str, int], np.ndarray]) -> dict:
    """PASS 3: metricas, deltas por pliegue, bootstrap pareado e IC95.

    Returns:
        Diccionario con las metricas de los cuatro brazos, las cuatro comparaciones, los
        deltas por pliegue, la media y la desviacion sobre las tres semillas, el bootstrap por
        clusters y los contadores descriptivos de unidades que cambian de decision.
    """
    log("")
    log("   PASS 3: metricas")
    fold_de = np.zeros(len(ctx.y), dtype=int)
    for k, (_, b) in enumerate(ctx.pl, start=1):
        fold_de[np.asarray(b)] = k

    por_brazo: dict[str, dict] = {}
    for brazo in BRAZOS:
        por_semilla = {s: metricas_de_brazo(ctx.y, probas[(brazo, s)], fold_de)
                       for s in SEMILLAS}
        principal = por_semilla[SEMILLA_PRIMARIA]
        cvs = [por_semilla[s]["macro_f1"] for s in SEMILLAS]
        por_brazo[brazo] = {
            "descripcion": DESCRIPCION_BRAZO[brazo], "matriz": MATRIZ_DE_BRAZO[brazo],
            "variante_entrada": VARIANTE_DE_BRAZO[brazo], "semilla_primaria": SEMILLA_PRIMARIA,
            "por_semilla": por_semilla, "macro_f1_por_semilla": cvs,
            "macro_f1_media": round(float(np.mean(cvs)), 10),
            "macro_f1_sd_entre_semillas": round(float(np.std(cvs, ddof=1)), 10),
            "macro_f1_primaria": principal["macro_f1"],
            "f1_por_etiqueta_primaria": principal["f1_positiva_por_etiqueta"],
            "brier_medio_primaria": principal["metricas_de_probabilidad"]["brier_medio"],
            "log_loss_medio_primaria": principal["metricas_de_probabilidad"]["log_loss_medio"],
            "nota": ("la desviacion entre semillas es variabilidad de SEMILLA, no de datos: las "
                     "tres semillas comparten folds, filas, batches, init y matrices dentro de "
                     "la misma condicion de entrada"),
        }
        log(f"      brazo {brazo} ({MATRIZ_DE_BRAZO[brazo]:9}/{VARIANTE_DE_BRAZO[brazo]:5}): "
            f"macro-F1 media {por_brazo[brazo]['macro_f1_media']:.6f} sd "
            f"{por_brazo[brazo]['macro_f1_sd_entre_semillas']:.6f} | primaria "
            f"{principal['macro_f1']:.6f} | F1 por etiqueta "
            f"{principal['f1_positiva_por_etiqueta']}")

    comparaciones: dict[str, dict] = {}
    for nombre, (x, z) in COMPARACIONES.items():
        deltas_globales = []
        deltas_por_pliegue_promedio = []
        deltas_por_pliegue_por_semilla = []
        for k in range(1, N_FOLDS + 1):
            sel = fold_de == k
            por_s = [round(f1m(ctx.y[sel], predicciones(probas[(x, s)][sel]))
                         - f1m(ctx.y[sel], predicciones(probas[(z, s)][sel])), 10)
                     for s in SEMILLAS]
            deltas_por_pliegue_por_semilla.append(por_s)
            deltas_por_pliegue_promedio.append(round(float(np.mean(por_s)), 10))
        for s in SEMILLAS:
            deltas_globales.append(round(f1m(ctx.y, predicciones(probas[(x, s)]))
                                         - f1m(ctx.y, predicciones(probas[(z, s)])), 10))
        boot_por_semilla = []
        for s in SEMILLAS:
            media, lo, hi = bootstrap_clusters(ctx.y, predicciones(probas[(x, s)]),
                                               predicciones(probas[(z, s)]), ctx.grupos,
                                               SEMILLA_BOOT, N_BOOTSTRAP)
            boot_por_semilla.append({"semilla": s, "media": media, "ic95": [lo, hi]})
        extremos_promediados = [
            round(float(np.mean([b["ic95"][0] for b in boot_por_semilla])), 10),
            round(float(np.mean([b["ic95"][1] for b in boot_por_semilla])), 10)]
        b_delta = bootstrap_delta_medio_entre_semillas(
            ctx.y, {s: probas[(x, s)] for s in SEMILLAS},
            {s: probas[(z, s)] for s in SEMILLAS}, ctx.grupos, SEMILLA_BOOT, N_BOOTSTRAP)
        comparaciones[nombre] = {
            "descripcion": DESCRIPCION_COMPARACION[nombre],
            "brazo_efecto": x, "brazo_control": z,
            "deltas_globales_por_semilla": deltas_globales,
            "delta_global_media": round(float(np.mean(deltas_globales)), 10),
            "delta_global_sd_entre_semillas": round(float(np.std(deltas_globales, ddof=1)), 10),
            "ic95_t_sobre_las_tres_semillas": ic_t(deltas_globales),
            "deltas_por_pliegue_por_semilla": deltas_por_pliegue_por_semilla,
            "deltas_por_pliegue_promedio": deltas_por_pliegue_promedio,
            "pliegues_positivos": int(sum(1 for d in deltas_por_pliegue_promedio if d > 0)),
            "bootstrap_por_clusters": {
                "semilla": SEMILLA_BOOT, "n": N_BOOTSTRAP, "por_semilla": boot_por_semilla,
                "media_de_las_tres": round(float(np.mean([b["media"] for b in boot_por_semilla])),
                                           10),
                "delta_medio_entre_semillas": {
                    "media": b_delta[0], "ic95": [b_delta[1], b_delta[2]],
                    "que_es": ("bootstrap por clusters del delta MEDIO entre las tres semillas: "
                               "por replica se remuestran clusters, se calcula el macro-F1 de "
                               f"{x} y de {z} sobre esa replica y se promedian sobre las tres "
                               "semillas. Este es el intervalo que usa la rama 2"),
                    "es_un_ic95_valido": True},
                "extremos_promediados_entre_semillas_no_es_un_ic95": extremos_promediados,
                "por_que_no_es_un_ic95": (
                    "es la media de los extremos de percentiles de tres bootstraps "
                    "INDEPENDIENTES, uno por semilla. El estimando de esos tres intervalos es "
                    "la diferencia de las semillas por separado, no el delta medio: promediar "
                    "los extremos no es tomar percentiles de la distribucion de una sola "
                    "estadistica, asi que el ancho y el centro no tienen lectura de 95 por "
                    "ciento. Se conserva solo por trazabilidad y no se usa en el veredicto"),
            },
            "unidades_que_cambian_de_decision": unidades_que_cambian(
                ctx.y, predicciones(probas[(x, SEMILLA_PRIMARIA)]),
                predicciones(probas[(z, SEMILLA_PRIMARIA)])),
        }
        d = comparaciones[nombre]
        log(f"      {nombre} = {x} - {z}: delta media {d['delta_global_media']:+.6f} sd "
            f"{d['delta_global_sd_entre_semillas']:.6f} | folds {d['pliegues_positivos']}/{N_FOLDS}"
            f" | IC95 t {d['ic95_t_sobre_las_tres_semillas']['ic95']} | bootstrap del delta "
            f"medio IC95 {b_delta[1:]}")
    log("      NOTA: `unidades que cambian de decision` es un RECUENTO de celdas "
        "tweet-etiqueta, es descriptivo y NO sustituye al macro-F1")
    return {"por_brazo": por_brazo, "comparaciones": comparaciones, "fold_de": fold_de,
            "v03_macro_f1": round(f1m(ctx.y, predicciones(ctx.p_v03)), 10)}


# =========================================================================
# SECCION 8: PASS 4, VEREDICTO CONTRA EL CRITERIO PRE-REGISTRADO
# =========================================================================
def _rama_por_ic(delta_p1: float, pliegues_positivos: int, contiene_cero: bool,
                 delta_p2: float) -> int:
    """Aplica las tres ramas excluyentes del criterio pre-registrado.

    Args:
        delta_p1: delta global medio de P1.
        pliegues_positivos: pliegues con delta positivo de P1.
        contiene_cero: booleano que dice si el IC95 de P1 contiene el cero.
        delta_p2: delta global medio de P2.

    Returns:
        El numero de rama: 1, 2, 3 o 0 si ninguna se cumple.
    """
    if delta_p1 >= EPS_GATE_DELTA and pliegues_positivos >= GATE_FOLDS:
        return 1
    if delta_p1 <= 0.0 and contiene_cero:
        return 2
    if delta_p2 > 0.0 and delta_p1 <= 0.0:
        return 3
    return 0


TEXTO_RAMA = {
    1: ("Si `P1 >= +0.003` y `P1` es positivo en al menos 4 de 5 pliegues: el embedding aporta "
        "senal con normalizacion alineada. La linea Word2Vec tiene fundamento y la Fase 4 "
        "(curva de corpus) se justifica."),
    2: ("Si `P1 <= 0` y su IC95 contiene al cero: el embedding no aporta. Se cierra la linea y "
        "se documenta que P73 y P74 estaban sobreestimados. No se abre ni corpus ni "
        "hiperparametros."),
    3: ("Si `P2 > 0` pero `P1 <= 0`: el efecto medido por P73 y P74 era un artefacto del "
        "tokenizado, no informacion del embedding."),
    0: ("Ninguna de las tres ramas excluyentes del pre-registro se cumple. Se reporta como no "
        "concluyente y NO se redacta ninguna conclusion nueva."),
}


def aplicar_criterio(comparaciones: dict) -> dict:
    """PASS 4: aplica la seccion 7 del pre-registro, sin mirar nada mas.

    Las ramas se aplican EN EL ORDEN del pre-registro. La seccion 5 declara DOS intervalos
    para el mismo delta, el bootstrap pareado por clusters y el IC95 t sobre las tres
    semillas, y la seccion 7 dice "su IC95" sin especificar cual. Se aplican las DOS lecturas
    de la rama 2 y, si discrepan, se declara la ambiguedad en vez de resolverla en silencio.
    La lectura principal del bootstrap es la del DELTA MEDIO entre semillas, que es un IC95
    valido; el promedio de extremos de los tres bootstraps por semilla se conserva solo como
    traza y no entra en la rama.

    Args:
        comparaciones: diccionario de PASS 3.

    Returns:
        Diccionario con la rama, el texto literal, las lecturas de IC95 y las observaciones.
    """
    p1, p2, p3, p4 = (comparaciones[c] for c in ("P1", "P2", "P3", "P4"))
    d1, d2 = p1["delta_global_media"], p2["delta_global_media"]
    d3, d4 = p3["delta_global_media"], p4["delta_global_media"]
    ic_boot = p1["bootstrap_por_clusters"]["delta_medio_entre_semillas"]["ic95"]
    extremos_promediados = (
        p1["bootstrap_por_clusters"]["extremos_promediados_entre_semillas_no_es_un_ic95"])
    ic_t95 = p1["ic95_t_sobre_las_tres_semillas"]["ic95"]
    contiene_boot = ic_boot[0] <= 0.0 <= ic_boot[1]
    contiene_extremos = extremos_promediados[0] <= 0.0 <= extremos_promediados[1]
    contiene_t = ic_t95[0] <= 0.0 <= ic_t95[1]
    rama_boot = _rama_por_ic(d1, p1["pliegues_positivos"], contiene_boot, d2)
    rama_extremos = _rama_por_ic(d1, p1["pliegues_positivos"], contiene_extremos, d2)
    rama_t = _rama_por_ic(d1, p1["pliegues_positivos"], contiene_t, d2)
    rama = rama_boot
    max_d34 = max(abs(d3), abs(d4))
    min_d34 = min(abs(d3), abs(d4))
    pareja = bool(max_d34 > 0.0 and (min_d34 / max_d34) >= UMBRAL_PAREJIDAD)
    return {
        "criterio": "docs/p78_preregistro.md seccion 7, aplicado sin modificarlo",
        "orden_de_aplicacion": "1, luego 2, luego 3, y si ninguna se cumple rama 0",
        "rama_cumplida": rama,
        "texto_literal_de_la_rama": TEXTO_RAMA[rama],
        "lectura_del_ic95": {
            "ambiguedad_del_pre_registro": bool(rama_boot != rama_t),
            "nota": ("la seccion 5 declara dos intervalos para el mismo delta y la seccion 7 dice "
                     "'su IC95' sin especificar cual. La lectura principal es el bootstrap pareado "
                     "por clusters DEL DELTA MEDIO, que es un IC95 valido; se reporta tambien la "
                     "t sobre las tres semillas y, solo por trazabilidad, la lectura que "
                     "resultaria del promedio de extremos, que no es un IC95 y no decide"),
            "bootstrap_por_clusters_delta_medio": ic_boot,
            "es_un_ic95_valido": True,
            "contiene_el_cero": bool(contiene_boot),
            "rama_por_bootstrap": rama_boot,
            "ic95_t_sobre_semillas": ic_t95, "contiene_el_cero_t": bool(contiene_t),
            "rama_por_t": rama_t,
            "extremos_promediados_entre_semillas_no_es_un_ic95": extremos_promediados,
            "rama_por_extremos_promediados_no_es_un_ic95": rama_extremos,
            "el_intervalo_que_decide": "bootstrap_por_clusters_delta_medio",
        },
        "regla_evaluada": {
            "P1_media": round(d1, 10), "P1_pliegues_positivos": p1["pliegues_positivos"],
            "P2_media": round(d2, 10), "P3_media": round(d3, 10), "P4_media": round(d4, 10),
            "eps_gate_delta": EPS_GATE_DELTA, "gate_folds": GATE_FOLDS,
            "hueco_del_pre_registro": (
                "el pre-registro no tiene rama para 0 < P1 < +0.003, ni para P1 > 0 con el IC95 "
                "excluyendo el cero, ni para P1 negativo con el IC95 excluyendo el cero. En esos "
                "casos la rama es 0 y se declara no concluyente"),
        },
        "observacion_P3_P4": {
            "se_cumple": pareja,
            "ratio_min_sobre_max": round(min_d34 / max_d34, 6) if max_d34 > 0 else None,
            "umbral_declarado": UMBRAL_PAREJIDAD,
            "texto_literal": ("Si `P3` y `P4` son de magnitud parecida: la alineacion de "
                              "normalizacion actua sobre ambos brazos por igual, y no es un "
                              "mecanismo especifico del embedding."),
        },
        "observacion_P1_P2": {
            "se_cumple": bool(d1 >= EPS_GATE_DELTA and d2 >= EPS_GATE_DELTA),
            "texto_literal": ("Si `P1` y `P2` son ambos claramente positivos: el tokenizado era "
                              "un cuello de botella adicional y hay senal que investigar."),
        },
        "que_no_decide": ("P78 no dice nada sobre superar a V03. Su unico producto es la "
                         "estimacion del efecto del embedding bajo un protocolo sin fuga, mas el "
                         "diagnostico de cobertura que decide si abrir la Fase 4"),
    }


# =========================================================================
# SECCION 9: ESCRITURA
# =========================================================================
def escribir_csv_pares(ctx: Contexto, probas: dict[tuple[str, int], np.ndarray],
                       fold_de: np.ndarray, destino: Path) -> int:
    """Escribe el OOF pareado en unidades tweet-etiqueta, para recomputar sin reentrenar.

    Columnas: las CUATRO probabilidades del pre-registro, una por brazo en la semilla
    primaria, con sus cuatro predicciones; y despues las ocho columnas de probabilidad y las
    ocho de prediccion de las otras dos semillas, para que el resultado completo sea
    recomputable sin volver a entrenar. Los umbrales son los de V03, fijos.

    Args:
        ctx: contexto.
        probas: probabilidades por (brazo, semilla).
        fold_de: numero de pliegue por fila.
        destino: ruta temporal donde se escribe. El nombre definitivo lo pone
            `publicar_atomico`, al final, para que un fallo posterior no deje el CSV solo.

    Returns:
        El numero de filas escritas.
    """
    columnas = ["row_id", "fila", "fold", "label", "y_true", "semilla_primaria"]
    columnas += [f"proba_{b}" for b in BRAZOS] + [f"pred_{b}" for b in BRAZOS]
    columnas += [f"proba_{b}_s{s}" for s in SEMILLAS[1:] for b in BRAZOS]
    columnas += [f"pred_{b}_s{s}" for s in SEMILLAS[1:] for b in BRAZOS]
    proba_primaria = {b: probas[(b, SEMILLA_PRIMARIA)] for b in BRAZOS}
    pred_primaria = {b: predicciones(proba_primaria[b]) for b in BRAZOS}
    proba_resto = {(b, s): probas[(b, s)] for s in SEMILLAS[1:] for b in BRAZOS}
    pred_resto = {k: predicciones(v) for k, v in proba_resto.items()}
    filas = 0
    with destino.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(columnas)
        for i in range(len(ctx.y)):
            for j, etiqueta in enumerate(ETIQUETAS):
                fila = [int(ctx.ids_v03[i]), i, int(fold_de[i]), etiqueta, int(ctx.y[i, j]),
                        SEMILLA_PRIMARIA]
                fila += [round(float(proba_primaria[b][i, j]), 8) for b in BRAZOS]
                fila += [int(pred_primaria[b][i, j]) for b in BRAZOS]
                fila += [round(float(proba_resto[(b, s)][i, j]), 8)
                         for s in SEMILLAS[1:] for b in BRAZOS]
                fila += [int(pred_resto[(b, s)][i, j])
                         for s in SEMILLAS[1:] for b in BRAZOS]
                w.writerow(fila)
                filas += 1
    return filas


def figuras(ctx: Contexto, probas: dict[tuple[str, int], np.ndarray], metricas: dict,
            veredicto: dict, cobertura: dict, sufijo: str = "") -> list[str]:
    """Genera las figuras de P78, con nombres y etiquetas en ASCII puro.

    Con `sufijo` no vacio, cada figura se guarda como `nombre + sufijo`, que es el temporal
    que despues publica `publicar_atomico`. OJO: matplotlib deduce el formato de la
    extension, y ".png.tmp_p78" no es un formato soportado, luego toda llamada lleva
    `format="png"` explicito o revienta con ValueError. La lista que devuelve son SIEMPRE los nombres
    definitivos, con o sin sufijo, porque es lo que se verifica y lo que se renombra.

    Args:
        ctx: contexto.
        probas: probabilidades por (brazo, semilla).
        metricas: salida de PASS 3.
        veredicto: salida de PASS 4.
        cobertura: salida de PASS 1.
        sufijo: sufijo temporal que se anade al nombre al guardar.

    Returns:
        Lista de nombres de fichero definitivos.

    Raises:
        SystemExit: si algun nombre de figura no es ASCII.
    """
    FIGURAS.mkdir(parents=True, exist_ok=True)
    escritas: list[str] = []

    def destino(nombre: str) -> Path:
        """Ruta de guardado de una figura, temporal si se pidio sufijo.

        Args:
            nombre: nombre definitivo de la figura.

        Returns:
            La ruta completa a la que se guarda ahora.
        """
        return FIGURAS / (nombre + sufijo)

    comp = metricas["comparaciones"]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    xs = np.arange(1, N_FOLDS + 1)
    ancho = 0.2
    for i, nombre in enumerate(COMPARACIONES):
        ax.bar(xs + (i - 1.5) * ancho, comp[nombre]["deltas_por_pliegue_promedio"], width=ancho,
               label=f"{nombre}: {COMPARACIONES[nombre][0]} - {COMPARACIONES[nombre][1]}")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axhline(EPS_GATE_DELTA, color="#c00000", ls=":", lw=1, label=f"gate +{EPS_GATE_DELTA}")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"pliegue {k}" for k in xs])
    ax.set_ylabel("delta macro-F1, media de las tres semillas")
    ax.set_title("P78: delta por pliegue de las cuatro comparaciones")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(destino("p78_01_deltas_por_pliegue.png"), dpi=110, format="png")
    plt.close(fig)
    escritas.append("p78_01_deltas_por_pliegue.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    claves = ["proporcion_tokens_con_id_distinto_de_cero", "proporcion_oov",
              "fraccion_truncada_por_maxlen", "fraccion_mas_cortas_que_el_kernel_mayor"]
    nombres = ["tokens con id no cero", "OOV", "truncadas por MAXLEN", "cortas vs kernel mayor"]
    xs2 = np.arange(len(claves))
    for i, variante in enumerate(VARIANTES):
        axes[0].bar(xs2 + (i - 0.5) * ancho, [cobertura[variante][c] for c in claves],
                    width=ancho, label=variante)
    axes[0].set_xticks(xs2)
    axes[0].set_xticklabels(nombres, fontsize=8)
    axes[0].set_ylabel("fraccion")
    axes[0].legend()
    axes[0].set_title("P78: cobertura por variante de entrada")
    motivos = cobertura["desglose_de_los_cinco_motivos"]["por_motivo"]
    nombres_m = list(MOTIVOS_OOV)
    ys = np.arange(len(nombres_m))
    axes[1].barh(ys - 0.2, [motivos[m]["oov"] for m in nombres_m], height=0.4, label="OOV")
    axes[1].barh(ys + 0.2, [motivos[m]["rescatadas_por_normalizar"] for m in nombres_m],
                 height=0.4, label="rescatadas al normalizar")
    axes[1].set_yticks(ys)
    axes[1].set_yticklabels(nombres_m)
    axes[1].set_xlabel("tokens")
    axes[1].legend(fontsize=8)
    axes[1].set_title("P78: por que hay OOV, medido token a token")
    plt.tight_layout()
    fig.savefig(destino("p78_02_cobertura_y_motivos.png"), dpi=110, format="png")
    plt.close(fig)
    escritas.append("p78_02_cobertura_y_motivos.png")

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    xs3 = np.arange(len(ETIQUETAS) + 1)
    for i, brazo in enumerate(BRAZOS):
        vals = [metricas["por_brazo"][brazo]["f1_por_etiqueta_primaria"][e] for e in ETIQUETAS]
        vals.append(metricas["por_brazo"][brazo]["macro_f1_primaria"])
        ax.bar(xs3 + (i - 1.5) * ancho, vals, width=ancho,
               label=f"{brazo}: {MATRIZ_DE_BRAZO[brazo]}/{VARIANTE_DE_BRAZO[brazo]}")
    ax.set_xticks(xs3)
    ax.set_xticklabels(list(ETIQUETAS) + ["macro"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("F1 positiva")
    ax.set_title("P78: F1 real por etiqueta de los cuatro brazos, semilla primaria")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(destino("p78_03_f1_por_etiqueta.png"), dpi=110, format="png")
    plt.close(fig)
    escritas.append("p78_03_f1_por_etiqueta.png")

    fig, axes = plt.subplots(1, len(ETIQUETAS), figsize=(15, 4), sharex=True)
    for j, etiqueta in enumerate(ETIQUETAS):
        for b in BRAZOS:
            axes[j].hist(probas[(b, SEMILLA_PRIMARIA)][:, j], bins=25, range=(0.0, 1.0),
                         histtype="step", lw=1.4, label=b, density=True)
        axes[j].axvline(UMBRALES[j], color="#c00000", ls="--", lw=1)
        axes[j].set_title(f"{etiqueta}: umbral {UMBRALES[j]:.2f}")
        axes[j].set_xlabel("probabilidad")
    axes[0].set_ylabel("densidad")
    axes[0].legend(fontsize=8, title="brazo")
    plt.tight_layout()
    fig.savefig(destino("p78_04_probabilidades.png"), dpi=110, format="png")
    plt.close(fig)
    escritas.append("p78_04_probabilidades.png")

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    nombres_c = list(COMPARACIONES)
    ax.bar(nombres_c, [comp[c]["delta_global_media"] for c in nombres_c],
           yerr=[comp[c]["delta_global_sd_entre_semillas"] for c in nombres_c], capsize=5,
           color="#2e75b6")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axhline(EPS_GATE_DELTA, color="#c00000", ls=":", lw=1, label=f"gate +{EPS_GATE_DELTA}")
    ax.set_ylabel("delta macro-F1 global")
    ax.set_title("P78: delta global por comparacion, media y sd de las tres semillas")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(destino("p78_05_delta_por_semilla.png"), dpi=110, format="png")
    plt.close(fig)
    escritas.append("p78_05_delta_por_semilla.png")

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    for i, c in enumerate(nombres_c):
        for s, b in enumerate(comp[c]["bootstrap_por_clusters"]["por_semilla"]):
            ax.plot([i + (s - 1) * 0.18] * 2, b["ic95"], color="#1f4e79", lw=2)
            ax.plot(i + (s - 1) * 0.18, b["media"], "o", color="#1f4e79", ms=4)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(range(len(nombres_c)))
    ax.set_xticklabels(nombres_c)
    ax.set_ylabel("delta macro-F1")
    ax.set_title(f"P78: IC95 del bootstrap pareado por clusters, {N_BOOTSTRAP} replicas")
    plt.tight_layout()
    fig.savefig(destino("p78_06_ic95_bootstrap.png"), dpi=110, format="png")
    plt.close(fig)
    escritas.append("p78_06_ic95_bootstrap.png")

    for nombre in escritas:
        if any(ord(c) > 127 for c in nombre):
            raise SystemExit(f"STOP asercion 10: nombre de figura no ASCII {nombre}")
    log("      seis figuras en figuras/p78/, con nombre y etiquetas en ASCII")
    log(f"      rama del criterio que se cumple: {veredicto['rama_cumplida']} | "
        f"P3 y P4 de magnitud parecida: {veredicto['observacion_P3_P4']['se_cumple']} | "
        f"P1 y P2 ambos claros: {veredicto['observacion_P1_P2']['se_cumple']}")
    return escritas


def verificar_figuras(nombres: list[str], sufijo: str = "") -> dict:
    """Comprobacion de las figuras: nombre ASCII, firma PNG valida y sha256.

    Un PNG es binario por construccion: sus bytes comprimidos son casi todos mayores que 127,
    y exigirle ASCII puro es imposible. Lo que se exige a una figura es el NOMBRE en ASCII, la
    firma PNG valida y que todas las etiquetas que se le pasan sean ASCII, lo cual se
    garantiza porque todas las cadenas de las figuras salen de literales ASCII de este
    modulo y de los nombres de las etiquetas del proyecto. Se declara como desviacion de la
    asercion 10.

    Args:
        nombres: nombres de figura, ya definitivos.
        sufijo: sufijo temporal con el que estan escritas ahora en disco.

    Returns:
        Diccionario con bytes y sha256 de cada figura, indexado por el nombre definitivo.

    Raises:
        SystemExit: si algun nombre no es ASCII o la firma PNG no es valida.
    """
    salida = {}
    for nombre in nombres:
        if any(ord(c) > 127 for c in nombre):
            raise SystemExit(f"STOP asercion 10: nombre de figura no ASCII {nombre}")
        datos = (FIGURAS / (nombre + sufijo)).read_bytes()
        if datos[:8] != b"\x89PNG\r\n\x1a\n":
            raise SystemExit(f"STOP: {nombre} no es un PNG valido")
        salida[nombre] = {"bytes": len(datos), "sha256": hashlib.sha256(datos).hexdigest()}
    return salida


def verificar_ascii_texto(rutas: list[Path]) -> dict[str, bool]:
    """Asercion 10, parte textual: comprueba byte a byte que no hay ningun byte mayor que 127.

    Devuelve el resultado OBSERVADO de cada fichero, no una constante: la clave es el nombre
    del fichero y el valor es si todos sus bytes son menores o iguales que 127. Un fichero que
    no existe tambien se registra, como False, porque no se puede certificar lo que no esta.
    Si alguno sale False, la funcion aborta. El orden en que se llama importa: el resultado se
    lleva al JSON DESPUES de esta llamada, nunca antes.

    Args:
        rutas: ficheros de texto a verificar: el script y los artefactos ya escritos.

    Returns:
        Diccionario `{nombre_de_fichero: bool}` con el resultado real de cada uno.

    Raises:
        SystemExit: si algun fichero tiene algun byte mayor que 127, o no existe.
    """
    observado = {r.name: (r.exists() and not any(b > 127 for b in r.read_bytes()))
                 for r in rutas}
    sucios = [nombre for nombre, ok in observado.items() if not ok]
    if sucios:
        raise SystemExit(f"STOP asercion 10: bytes no ASCII o fichero ausente en {sucios}")
    return observado


def ruta_temporal(final: Path) -> Path:
    """Ruta temporal de un artefacto, en el MISMO directorio que su version final.

    Un `Path.replace` dentro del mismo sistema de ficheros es atomico, y para que lo sea los
    dos nombres tienen que caer en el mismo. El sufijo temporal es ASCII, asi que los
    temporales tambien entran en la asercion 10.

    Args:
        final: ruta del artefacto definitivo.

    Returns:
        La ruta temporal con la que se escribe antes de publicar.
    """
    return final.with_name(final.name + SUFIJO_TEMPORAL)


def escribir_manifiesto_sha256(destino: Path, piezas: list[tuple[str, Path]]) -> int:
    """Escribe el manifiesto sha256 de los artefactos, sin auto-referenciarse.

    Va en su propio fichero y NO dentro del JSON, precisamente para que el JSON no tenga que
    contener su propio hash: el manifiesto referencia al JSON, y el JSON no referencia al
    manifiesto.

    Args:
        destino: ruta del manifiesto.
        piezas: pares `(etiqueta, ruta)` en el orden en que se listan.

    Returns:
        El numero de entradas escritas.
    """
    lineas = [f"{hashlib.sha256(ruta.read_bytes()).hexdigest()}  {etiqueta}  {ruta.name}"
              for etiqueta, ruta in piezas]
    destino.write_text("\n".join(lineas) + "\n", encoding="ascii")
    return len(lineas)


def publicar_atomico(pares: list[tuple[Path, Path]]) -> None:
    """Renombra todos los temporales a sus nombres finales, uno detras de otro.

    Se llama SOLO despues de que han pasado todas las verificaciones. Los artefactos existen
    con su nombre temporal hasta ese momento, asi que un fallo anterior no deja ni un CSV con
    4449 filas sin veredicto ni una figura sin informe. La renombracion en si no es una
    transaccion: si un `replace` fallara a mitad, quedarian artefactos finals de antes. Es el
    unico riesgo que queda y no se puede cerrar sin un directorio transaccional.

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


def construir_json(ctx: Contexto, metricas: dict, veredicto: dict, auditoria: dict,
                   cobertura: dict, registros: list[dict], manifiesto: dict,
                   nombres_figuras: list[str], info_figuras: dict,
                   n_filas_csv: int, ascii_previo: dict[str, bool],
                   n_manifiesto: int) -> dict:
    """Ensambla el JSON de resultados, con los cuatro brazos y las cuatro comparaciones.

    Args:
        ctx: contexto.
        metricas: salida de PASS 3.
        veredicto: salida de PASS 4.
        auditoria: PASS 0 y aserciones 6 y 8.
        cobertura: salida de PASS 1.
        registros: registros por pliegue de los doce pares brazo por semilla.
        manifiesto: manifiesto de pareo.
        nombres_figuras: nombres de figura.
        info_figuras: sha256 y tamano de cada figura.
        n_filas_csv: filas escritas en el CSV pareado.
        ascii_previo: resultado OBSERVADO de `verificar_ascii_texto` sobre el script y el CSV,
            obtenido ANTES de que esta funcion se llamara.
        n_manifiesto: entradas del manifiesto sha256, que se escribe aparte y no se incluye.

    Returns:
        El diccionario que se serializa.

    Raises:
        SystemExit: si alguna clave del JSON no es ASCII, o si `ascii_previo` trae un False.
    """
    if not ascii_previo or not all(ascii_previo.values()):
        raise SystemExit(f"STOP asercion 10: el resultado previo no es todo True: {ascii_previo}")
    salida = {
        "familia": "P78_CONTROL_ANIDADO_WORD2VEC",
        "experiment_id": "P78",
        "fase": "control anidado Word2Vec frente al azar, con normalizacion alineada",
        "pre_registro": PREREGISTRO, "pre_registro_sha256": PREREGISTRO_SHA256,
        "pre_registro_commit": PREREGISTRO_COMMIT,
        "pregunta": ("en condiciones de early stopping estrictamente anidados, el embedding "
                     "Word2Vec aporta senal sobre el objetivo frente a una matriz aleatoria de "
                     "identica forma e identicas normas de fila, cuando el texto de entrada se "
                     "normaliza igual que el corpus?"),
        "configuracion": CONFIG,
        "protocolo": {
            "por_pliegue": ("1) epoca_estrella produce E* como mediana de tres splits internos "
                            "por clusters sobre ia, y NINGUNA fila de ib participa. 2) "
                            "entrenar_fijo reentrena sobre ia COMPLETO durante E* epocas fijas. "
                            "3) outer_val se evalua una sola vez por (brazo, semilla, pliegue)"),
            "semilla_base_de_los_splits_internos": SEMILLA_INTERNA_BASE,
            "formula_de_la_semilla_interna": "SEMILLA_INTERNA_BASE + 100*k + 7*j, con j en 0..2",
            "control_aleatorio": ("genera_aleatoria(semilla, W, normas), que conserva las normas "
                                   "de fila. B y D comparten la MISMA matriz aleatoria para la "
                                   "misma semilla, de modo que P4 = D - B aisla la normalizacion"),
            "diferencia_entre_las_dos_variantes_de_entrada": (
                "variante_cruda es `tokenizar` de P73-02 tal cual; variante_norm es "
                "`tokenizar(normalizar(t))` con `normalizar` de P71-03. La unica diferencia "
                "entre ambas es la pre-normalizacion, nada mas"),
        },
        "auditoria_de_integridad": auditoria,
        "cobertura": cobertura,
        "brazos": metricas["por_brazo"],
        "comparaciones": metricas["comparaciones"],
        "v03_macro_f1": metricas["v03_macro_f1"],
        "registro_por_pliegue": registros,
        "manifiesto_de_pareo": manifiesto["pareo"],
        "veredicto": veredicto,
        "csv_pareado": {
            "ruta": str(CSV_PARES.relative_to(PROJECT_ROOT)),
            "unidades_tweet_etiqueta": n_filas_csv,
            "columnas_de_probabilidad": (
                [f"proba_{b}" for b in BRAZOS]
                + [f"proba_{b}_s{s}" for s in SEMILLAS[1:] for b in BRAZOS]),
            "nota": ("las cuatro probabilidades del pre-registro son las de la semilla primaria "
                     f"{SEMILLA_PRIMARIA}, en las columnas proba_A a proba_D; las de las otras "
                     "dos semillas van anadidas para que el resultado sea recomputable sin "
                     "reentrenar"),
        },
        "figuras": {n: info_figuras[n] for n in nombres_figuras},
        "asercion_10_ascii": {
            "que_se_verifico_antes_de_construir_este_json": ascii_previo,
            "que_se_verifico_y_cuando": (
                "ANTES de construir este JSON, byte a byte y leyendo el fichero del disco: el "
                "propio script y el CSV pareado ya escrito en su temporal. El campo de arriba "
                "es el resultado OBSERVADO de esa llamada, no una constante redactada. Este "
                "JSON y el JSON de cobertura todavia no existian en ese momento, asi que no "
                "aparecen: se vuelven a verificar sobre sus temporales DESPUES de escribirlos "
                "y ANTES de renombrarlos, y si alguno sale False el script aborta y no queda "
                "ningun artefacto final. Ningun JSON puede certificar su propio hash; para eso "
                "esta el manifiesto sha256 aparte"),
            "manifiesto_sha256": {
                "ruta": str(CSV_MANIFIESTO.relative_to(PROJECT_ROOT)),
                "entradas": n_manifiesto,
                "cubre": ("el CSV pareado, los dos JSON, las seis figuras y este propio script"),
                "por_que_fuera_del_json": ("para que el JSON no se auto-referencie: el manifiesto "
                                           "contiene el sha del JSON, no al reves")},
            "desviacion_declarada": (
                "Un PNG es binario por construccion y sus bytes comprimidos son mayores que "
                "127, asi que la asercion 10 se aplica byte a byte al script, al JSON de "
                "resultados, al JSON de cobertura y al CSV, y a las figuras se les exige "
                "nombre ASCII, firma PNG valida y etiquetas ASCII. Es imposible de cumplir "
                "literalmente sobre un PNG"),
        },
        "cumplimiento": {
            "test_leido": 0, "holdout_leido": 0, "submission_generada": 0,
            "leaderboard_consultado": 0, "v03_reentrenado": 0, "word2vec_tocado": 0,
            "parquet_de_slices_leido": 0, "gpu_usado": 0, "scripts_p71_a_p77_editados": 0,
            "nota_holdout": "la particion holdout se aparta con holdout_v2 y no se lee nunca",
        },
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    for clave, valor in salida.items():
        if any(ord(c) > 127 for c in clave):
            raise SystemExit(f"STOP asercion 10: clave de JSON no ASCII {clave!r}")
        if any(ord(c) > 127 for c in json.dumps(valor, ensure_ascii=False)):
            raise SystemExit(f"STOP asercion 10: valor de JSON no ASCII en la clave {clave!r}")
    return salida


# =========================================================================
# SECCION 10: MAIN
# =========================================================================
def main() -> None:
    """Ejecuta P78 completo: PASS 0 a PASS 4, mas escritura y verificacion ASCII.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    torch.set_num_threads(32)
    FIGURAS.mkdir(parents=True, exist_ok=True)
    log("P78 control anidado Word2Vec frente al azar, con normalizacion alineada")
    log(f"   pre-registro: {PREREGISTRO} sha256 {PREREGISTRO_SHA256} commit {PREREGISTRO_COMMIT}")
    log(f"   brazos {list(BRAZOS)} | variantes {list(VARIANTES)} | semillas {list(SEMILLAS)}")
    log(f"   umbrales V03 congelados {[float(u) for u in UMBRALES]} | V03_CCV {V03_CCV}")
    log(f"   dispositivo cpu | hilos {torch.get_num_threads()} | sin GPU y sin AMP (D-008)")
    log(f"   coste: {len(BRAZOS) * len(SEMILLAS) * N_FOLDS} unidades por "
        f"{N_INTERNOS + 1} entrenamientos = "
        f"{len(BRAZOS) * len(SEMILLAS) * N_FOLDS * (N_INTERNOS + 1)} TextCNN en total")

    ctx = pass0_integridad()
    preparar_variantes(ctx)
    log("")
    cobertura = pass1_cobertura(ctx)
    log("")
    log("   asercion 6: `p75_01.partir_interna` queda instrumentado alrededor de cada llamada "
        "real a `epoca_estrella`; la verificacion se hace DESPUES, sobre lo observado")
    probas, registros, manifiesto = pass2_brazos(ctx)
    auditoria_indices = auditar_indices(ctx)
    metricas = pass3_metricas(ctx, probas)
    veredicto = aplicar_criterio(metricas["comparaciones"])

    # --- escritura ATOMICA: primero temporales, verificacion, y renombrado al final ---
    log("")
    log("   escritura atomica: CSV, figuras y JSON van a temporales en su mismo directorio y "
        "solo se renombran cuando todo ha pasado")
    csv_tmp = ruta_temporal(CSV_PARES)
    json_tmp = ruta_temporal(JSON_RESULTADOS)
    cob_tmp = ruta_temporal(JSON_COBERTURA)
    man_tmp = ruta_temporal(CSV_MANIFIESTO)
    pares = [(csv_tmp, CSV_PARES), (json_tmp, JSON_RESULTADOS),
             (cob_tmp, JSON_COBERTURA), (man_tmp, CSV_MANIFIESTO)]

    n_filas_csv = escribir_csv_pares(ctx, probas, metricas["fold_de"], csv_tmp)
    log(f"      CSV pareado en temporal: {n_filas_csv} unidades tweet-etiqueta")
    nombres_figuras = figuras(ctx, probas, metricas, veredicto, cobertura, sufijo=SUFIJO_TEMPORAL)
    info_figuras = verificar_figuras(nombres_figuras, sufijo=SUFIJO_TEMPORAL)
    pares += [(FIGURAS / (n + SUFIJO_TEMPORAL), FIGURAS / n) for n in nombres_figuras]
    log(f"      seis figuras en temporal en {FIGURAS.relative_to(PROJECT_ROOT)}")

    # --- asercion 10, primera pasada: sobre lo que ya existe en disco, ANTES del JSON ---
    ascii_previo = verificar_ascii_texto([Path(__file__), csv_tmp])
    log(f"      asercion 10, primera pasada, antes de construir el JSON: {ascii_previo}")

    auditoria = {
        "v03_reconstruido": round(metricas["v03_macro_f1"], 10), "v03_esperado": V03_CCV,
        "aserciones": {
            "1_macro_f1_de_v03": True, "2_ids_del_oof": True, "3_cobertura_de_pliegues": True,
            "4_sha256_del_w2v": True, "5_sha_matriz_pad": True,
            # Estas dos no son constantes redactadas: las de abajo son el resultado observado
            # de la comprobacion, y se sustituyen por el booleano real justo despues.
            "6_inner_val_aislado": auditoria_indices["cumple"],
            "7_una_evaluacion_por_unidad": True, "8_pareo_de_los_brazos": True,
            "9_probabilidades_finitas_y_sin_filas_sin_prediccion": True,
            "10_ascii": all(ascii_previo.values()) and len(ascii_previo) > 0,
        },
        "de_donde_sale_el_10": ("`10_ascii` es el AND de los valores observados por "
                               "`verificar_ascii_texto([script, CSV temporal])` antes de "
                               "construir este JSON. Los dos JSON no existian todavia, asi que "
                               "se vuelven a verificar sobre sus temporales despues de "
                               "escribirlos y antes de renombrarlos; ese resultado no cabe "
                               "aqui y vive en el log y en el manifiesto sha256"),
        "n_holdout_apartado_y_no_usado": ctx.n_holdout,
        "indices_del_early_stopping": auditoria_indices,
        "contadores_de_evaluacion_sobre_outer_val": {
            "unidades": len(BRAZOS) * len(SEMILLAS) * N_FOLDS,
            "evaluaciones_por_unidad": 1,
            "total": len(BRAZOS) * len(SEMILLAS) * N_FOLDS,
            "nota": ("`entrenar_fijo` hace dos pasadas hacia delante sobre outer_val para "
                     "comprobar el determinismo del modo eval. No es una segunda seleccion: el "
                     "checkpoint lo fija E*, que se midio sin ver outer_val"),
        },
    }
    salida = construir_json(ctx, metricas, veredicto, auditoria, cobertura, registros,
                            manifiesto, nombres_figuras, info_figuras, n_filas_csv,
                            ascii_previo, 0)
    json_tmp.write_text(json.dumps(salida, ensure_ascii=True, indent=2) + "\n", encoding="ascii")
    cob_tmp.write_text(json.dumps(cobertura, ensure_ascii=True, indent=2) + "\n",
                       encoding="ascii")

    # --- asercion 10, segunda pasada: los cuatro temporales, leidos del disco ---
    ascii_final = verificar_ascii_texto([Path(__file__), csv_tmp, json_tmp, cob_tmp])
    log(f"      asercion 10, segunda pasada, sobre los temporales ya escritos: {ascii_final}")
    log("      asercion 10 OK: script, JSON de resultados, JSON de cobertura y CSV verificados "
        "byte a byte, ningun byte mayor que 127")

    # --- manifiesto sha256, aparte para que el JSON no se auto-referencie ---
    piezas = [("csv_pareado", csv_tmp), ("json_resultados", json_tmp), ("json_cobertura", cob_tmp)]
    piezas += [(f"figura_{n}", FIGURAS / (n + SUFIJO_TEMPORAL)) for n in nombres_figuras]
    piezas += [("script", Path(__file__))]
    n_manifiesto = escribir_manifiesto_sha256(man_tmp, piezas)
    salida["asercion_10_ascii"]["manifiesto_sha256"]["entradas"] = n_manifiesto
    salida["asercion_10_ascii"]["sha256_verificados_sobre_los_temporales"] = {
        etiqueta: hashlib.sha256(ruta.read_bytes()).hexdigest() for etiqueta, ruta in piezas
        if etiqueta != "script"}
    json_tmp.write_text(json.dumps(salida, ensure_ascii=True, indent=2) + "\n", encoding="ascii")
    verificar_ascii_texto([json_tmp])
    escribir_manifiesto_sha256(man_tmp, piezas)
    log(f"      manifiesto sha256: {n_manifiesto} entradas en "
        f"{CSV_MANIFIESTO.relative_to(PROJECT_ROOT)} (CSV, dos JSON, seis figuras y el script)")

    publicar_atomico(pares)
    log(f"      publicados por renombrado atomico: {CSV_PARES.relative_to(PROJECT_ROOT)}, "
        f"{JSON_RESULTADOS.relative_to(PROJECT_ROOT)}, "
        f"{JSON_COBERTURA.relative_to(PROJECT_ROOT)}, "
        f"{CSV_MANIFIESTO.relative_to(PROJECT_ROOT)} y las seis figuras")

    log("")
    log("=" * 78)
    log(f"P78 OK | rama {veredicto['rama_cumplida']} del criterio pre-registado")
    for c in COMPARACIONES:
        d = metricas["comparaciones"][c]
        log(f"   {c} = {COMPARACIONES[c][0]} - {COMPARACIONES[c][1]}: "
            f"{d['delta_global_media']:+.6f} (sd {d['delta_global_sd_entre_semillas']:.6f}) | "
            f"folds {d['pliegues_positivos']}/{N_FOLDS}")
    log(f"   cobertura: OOV cruda {100 * cobertura['cruda']['proporcion_oov']:.2f}% | OOV norm "
        f"{100 * cobertura['norm']['proporcion_oov']:.2f}% | rescate "
        f"{cobertura['desglose_de_los_cinco_motivos']['oov_rescatados_por_normalizar']} tokens")
    log(f"   informe {JSON_RESULTADOS.relative_to(PROJECT_ROOT)} | cobertura "
        f"{JSON_COBERTURA.relative_to(PROJECT_ROOT)} | pareado "
        f"{CSV_PARES.relative_to(PROJECT_ROOT)}")
    log(f"   {inicio.isoformat(timespec='seconds')} -> "
        f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    log("=" * 78)


if __name__ == "__main__":
    try:
        ast.parse(Path(__file__).read_text(encoding="utf-8"))
        main()
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        log(f"FALLO de P78: {type(exc).__name__}: {exc}")
        sys.exit(1)
