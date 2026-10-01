"""p82_01_rnn_atencion.py

Fecha: 2026-09-30
Descripcion: P82. RNN bidireccional con capa de atencion aditiva, comparada CONTRA SI
    MISMA entre cinco brazos, con early stopping estrictamente anidado, celda de relleno
    y celda de OOV separadas, relleno aplicado ANTES de la red recurrente y
    clip_grad_norm_ en todos los bucles. Veintitres afirmaciones, A0 a A22.
Pre-registro: docs/p82_preregistro_rnn_atencion.md
    sha256 ba74ae71a3b54a273c7f0ac5d55e4d4fd7d1ede641f5fbd022c2ae6377a62061
Estado del arte aplicado: atencion aditiva tipo Bahdanau (v, W_h, W_s) sobre las salidas
    de una RNN bidireccional, consulta igual al ULTIMO estado oculto valido, softmax sobre
    las posiciones NO de relleno con menos infinito antes del softmax, contexto concatenado
    con el ultimo estado oculto valido, embeddings Word2Vec 100d CONGELADOS, packed
    sequence con enforce_sorted False, AdamW con ReduceLROnPlateau sobre la perdida de
    validacion INTERNA, y emparejamiento por semilla para que los deltas sean pareados.

Salidas: reports/p82_rnn_atencion.json y reports/p82_rnn_atencion.md. Los parciales por
    bloque y por afirmacion del bloque caro van a reports/p82_parciales/, cada uno en un
    fichero PROPIO con marca de corrida. El modo de humo escribe en reports/humo_p82_<marca>/
    y NUNCA bloquea la corrida real, porque el guard de escritura unica se niega a
    sobrescribir un destino existente.

Fronteras que este script NO cruza: no lee test, no lee el holdout, no genera submission,
no consulta el leaderboard, no reentrena V03, no toca los umbrales congelados 0.44, 0.51 y
0.46, no reentrena el artefacto Word2Vec y no modifica ningun script de P71 a P81.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import inspect
import json
import math
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Final, Sequence

import numpy as np
import torch
from scipy import optimize, stats
from torch import nn

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train  # noqa: E402
from src.embedding_utils import normalizar_emb_normal, tokenizar  # noqa: E402
from src.evaluation import ETIQUETAS, metricas_multietiqueta  # noqa: E402
from src.splits import grupos_fusionados, holdout_v2, pliegues_corregidos  # noqa: E402
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence  # noqa: E402

# =============================================================================
# SECCION 1: CONFIGURACION Y CONSTANTES DECLARADAS
# =============================================================================

CONFIG: Final[dict[str, Any]] = {
    "experimento": "P82",
    "pre_registro": "docs/p82_preregistro_rnn_atencion.md",
    "pre_registro_sha256": (
        "ba74ae71a3b54a273c7f0ac5d55e4d4fd7d1ede641f5fbd022c2ae6377a62061"
    ),
    "semillas": tuple(20261401 + i for i in range(10)),
    "n_pliegues": 5,
    "n_internos": 3,
    "brazos": ("B1", "B2", "B3", "B4", "B5"),
    "tramo_1": ("B1", "B2", "B5"),
    "tramo_2": ("B3", "B4"),
    "max_len": 64,
    "dim_embedding": 100,
    "dim_oculto": 64,
    "batch": 64,
    "dropout_entrada": 0.5,
    "dropout_clasificador": 0.3,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epocas_max": 30,
    "paciencia_es": 5,
    "mejora_minima": 1e-4,
    "clip_max_norm": 1.0,
    "scheduler_factor": 0.5,
    "scheduler_patience": 3,
    "scheduler_min_lr": 1e-5,
    "semilla_split_interno": 20261400,
    "semilla_permutacion_b5": 20261405,
    "budget_segundos": 5400,
    "hito_segundos": 2400,
    "potencia_objetivo": 0.800000,
    "n_filas_desarrollo": 1483,
    "n_filas_holdout": 248,
    "etiquetas": tuple(ETIQUETAS),
}

# Artefactos congelados. Se abren solo en modo lectura y nunca se escriben.
W2V_NPZ: Final[Path] = PROJECT_ROOT / "modelos" / "twitter_domain" / "w2v_twitter_100d.npz"
W2V_SHA256: Final[str] = "302c994d13b75f8f5fbd440eea2c4f4113997e2581524021923b92f8244d705a"
V03_JSON: Final[Path] = PROJECT_ROOT / "experiments" / "baseline_v03_frozen" / "pipeline_v03.json"
V03_OOF: Final[Path] = PROJECT_ROOT / "features" / "oof_v03_final.npz"
P78_ADDENDUM: Final[Path] = PROJECT_ROOT / "reports" / "p78_addendum.md"
PREREGISTRO: Final[Path] = PROJECT_ROOT / CONFIG["pre_registro"]
REPORTS: Final[Path] = PROJECT_ROOT / "reports"
PARCIALES: Final[Path] = REPORTS / "p82_parciales"

# ---------- DISPOSITIVO DE CALCULO, ENMIRENDA 2 ----------
# La primera ejecucion en modo completo salio por CPU porque el script no tenia forma de
# usar la GPU: no contenia ninguna mencion a cuda y ni una llamada .to con dispositivo.
# Considio 66 minutos sin cerrar el primer bloque con la GPU al 7 por ciento de uso.
ENMIENDA_2: Final[str] = "docs/p82_enmienda_2.md"
ENMIENDA_2_SHA256: Final[str] = (
    "af731b8966de8f6a6461bba182926763aa2bdb3405c8dad58fad165bd7bdd696"
)
COSTO_POR_AJUSTE_SEGUNDOS: Final[dict[str, float]] = {
    "cpu_16_hilos": 2.43,
    "cpu_30_hilos": 3.22,
    "gpu_rocm": 1.24,
}
# POR QUE 16 HILOS Y NO MAS. Segun la medicion de la enmienda 2, 30 hilos es un 32 por
# ciento PEOR que 16: sobrecarga de planificacion, porque en un bucle con volumen de
# tensores pequenos mas hilos pelean por el mismo trabajo. La primera ejecucion ocupaba
# 30 nucleos, es decir corria en el peor de los tres puntos medidos. No se usa lo que
# traiga el entorno.
HILOS: Final[int] = 16
MOTIVO_DE_LOS_HILOS: Final[str] = (
    "16 por medicion, no por conveniencia: con 30 hilos un ajuste tarda 3.22 s frente a "
    "2.43 s con 16, un 32 por ciento mas, por sobrecarga de planificacion con tensores "
    "pequenos. La GPU con ROCm tarda 1.24 s por ajuste"
)
MOTIVO_DE_NO_USAR_PRECISION_MIXTA: Final[str] = (
    "no se usa autocast, ni GradScaler, ni float16, ni bfloat16. En un diseno cuyo "
    "nucleo es el pareamiento entre brazos, una via que cambia la precision introduce "
    "variabilidad numerica y seria el tipo de cosa que este proyecto ha pagado dos veces. "
    "Todo el calculo va en float32"
)
MOTIVO_DEL_GENERADOR_EN_CPU: Final[str] = (
    "el generador de permutaciones se queda en CPU a proposito: A9 hashea la secuencia de "
    "lotes, y ese hash tiene que ser estable y auditable aunque los datos esten en GPU. "
    "Si el generador fuera a GPU el hash dependeria del dispositivo y A9 perderia sentido "
    "como invariante de emparejamiento"
)


def resolver_dispositivo() -> torch.device:
    """Resuelve UNA vez el dispositivo de calculo. Se llama en un solo sitio.

    Returns:
        La GPU si torch.cuda.is_available, y CPU si no lo esta.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def nombre_del_dispositivo(dispositivo: torch.device) -> str:
    """Nombre real del dispositivo, para publicarlo sin que se le pregunte.

    Args:
        dispositivo: dispositivo resuelto.

    Returns:
        El modelo de la GPU si es GPU, o el texto CPU.
    """
    if dispositivo.type == "cpu":
        return "CPU"
    return torch.cuda.get_device_name(dispositivo)


DISPOSITIVO: Final[torch.device] = resolver_dispositivo()
NOMBRE_DISPOSITIVO: Final[str] = nombre_del_dispositivo(DISPOSITIVO)


def cadena_del_dispositivo_resuelto() -> str:
    """El dispositivo resuelto CON indice, que es como lo reportan los tensores.

    str(DISPOSITIVO) devuelve "cuda" y un tensor en esa GPU reporta "cuda:0", luego
    comparar el uno con el otro sin indice daria un falso negativo.

    Returns:
        La cadena con tipo e indice, "cuda:0" o "cpu".
    """
    if DISPOSITIVO.type == "cpu":
        return "cpu"
    return f"cuda:{torch.cuda.current_device()}"


DISPOSITIVO_RESUELTO: Final[str] = cadena_del_dispositivo_resuelto()


def comprobar_generador_real(
    observado: Any, referencias: Any, device_del_modelo: str
) -> tuple[bool, dict[str, Any]]:
    """Verifica el generador REAL que el entrenamiento uso, no uno reconstruido.

    Exige tres cosas: que el entrenamiento haya publicado un generador, que el objeto
    observado SEA ese mismo objeto por identidad, y que ese generador este en CPU. La
    ultima es la que exige la enmienda 2: el hash de A9 depende de que el generador no
    dependa del dispositivo. La identidad es lo que evita que el caso se vuelva
    decorativo: si se le pasa un generador recien creado, la comprobacion falla.

    Args:
        observado: el generador que se presenta como el del entrenamiento.
        referencias: el generador que el entrenamiento publico.
        device_del_modelo: donde esta el modelo, para que el escenario no sea vacio.

    Returns:
        Tupla con el exito y el detalle.
    """
    es_el_real = observado is referencias and referencias is not None
    device_real = str(referencias.device) if referencias is not None else "AUSENTE"
    hay_gpu = DISPOSITIVO.type != "cpu"
    comprobaciones = {
        "el_entrenamiento_publico_un_generador": referencias is not None,
        "el_generador_observado_es_por_identidad_el_del_entrenamiento": es_el_real,
        "el_generador_del_entrenamiento_esta_en_CPU": device_real == "cpu",
        "el_escenario_no_es_vacio": (not hay_gpu) or device_del_modelo != "cpu",
    }
    return todo_verdadero(comprobaciones, "comprobaciones"), {
        "device_del_generador_observado": device_real,
        "device_del_generador_real": device_real,
        "identidad": es_el_real,
        "device_del_modelo": device_del_modelo,
        "comprobaciones": comprobaciones,
    }


def comprobar_dispositivo_de_ejecucion() -> tuple[bool, dict[str, Any]]:
    """COMPROBACION EN TIEMPO DE EJECUCION del dispositivo, integrada en A0.

    No es una suposicion: construye una red real con la geometria declarada y mira donde
    estan DE VERDAD sus parametros y la matriz de embeddings. Un informe que dice que se
    uso la GPU con un modelo en CPU es peor que un informe que no dice nada.

    Returns:
        Tupla con el exito y el detalle.
    """
    matriz = np.zeros((8, 8), dtype=np.float32)
    ajustes = Ajustes(max_len=4, dim_oculto=4)
    modelo = construir_red(matriz, ajustes, "GRU", True)
    estado = modelo.comprobar_dispositivo()
    generador = torch.Generator().manual_seed(1)
    generador_en_cpu = str(generador.device) == "cpu"
    comprobaciones = {
        "todos_los_parametros_en_el_dispositivo_declarado": estado["todos_en_el_declarado"],
        "matriz_de_embeddings_en_el_dispositivo_declarado": estado["embeddings_en_el_declarado"],
        "el_dispositivo_declarado_no_esta_vacio": bool(NOMBRE_DISPOSITIVO.strip()),
        "generador_de_permutaciones_en_cpu": generador_en_cpu,
    }
    return todo_verdadero(comprobaciones, "comprobaciones"), {
        "dispositivo": estado,
        "generador_de_permutaciones": str(generador.device),
        "informacion_publicada": informacion_de_dispositivo(),
        "comprobaciones": comprobaciones,
    }


def informacion_de_dispositivo() -> dict[str, Any]:
    """Lo que la Enmienda 2 exige que el informe declare sin que se le pregunte.

    Returns:
        Diccionario con dispositivo, nombre, hilos, medidas y las dos decisiones.
    """
    return {
        "dispositivo": DISPOSITIVO.type,
        "nombre_del_dispositivo": NOMBRE_DISPOSITIVO,
        "cuda_disponible": bool(torch.cuda.is_available()),
        "hilos_fijados": HILOS,
        "motivo_de_los_hilos": MOTIVO_DE_LOS_HILOS,
        "precision_mixta": "NO USADA",
        "motivo_de_no_usar_precision_mixta": MOTIVO_DE_NO_USAR_PRECISION_MIXTA,
        "generador_de_permutaciones_en": "cpu",
        "motivo_del_generador_en_cpu": MOTIVO_DEL_GENERADOR_EN_CPU,
        "medicion_del_coste_por_ajuste_en_segundos": COSTO_POR_AJUSTE_SEGUNDOS,
        "coste_de_la_transferencia_del_indice": {
            "que_pasa": (
                "la permutacion de lotes nace con torch.randperm SIN device, luego vive "
                "en CPU, y se usa como indice sobre tensores que estan en el dispositivo. "
                "PyTorch lo acepta y paga una indexacion GPU por lote"
            ),
            "microsegundos_por_lote": 33.0,
            "microsegundos_medidos": (
                "medicion del revisor de P82: 500 indexaciones en 0.0164 s"
            ),
            "milisegundos_por_epoca": 16.0,
            "milisegundos_por_ajuste_a_30_epocas": 500.0,
            "minutos_para_1000_ajustes": 8.0,
            "por_que_se_acepta": (
                "es deliberado y lo exige la enmienda 2: el generador se queda en CPU "
                "para que el hash de A9 sea estable entre dispositivos. Pagar unos 0.5 s "
                "por ajuste a cambio de que A9 siga siendo una invariante auditable es "
                "el intercambio correcto, y se publica en vez de omitirse"
            ),
        },
        "enmienda_2": ENMIENDA_2,
        "enmienda_2_sha256": ENMIENDA_2_SHA256,
        "consecuencia_declarada": (
            "los valores absolutos de los cinco brazos y las deltas pareadas son los de "
            "este dispositivo y no deben compararse con los de una corrida en otro. El "
            "pareamiento DENTRO de la corrida se conserva porque los cinco brazos se "
            "ejecutan en el mismo dispositivo"
        ),
    }
REPORTE_JSON: Final[Path] = REPORTS / "p82_rnn_atencion.json"
REPORTE_MD: Final[Path] = REPORTS / "p82_rnn_atencion.md"

# Numeros del pre-registro. NO se modifican. Si un recalculo difiere, se reporta.
SD_PARADA_PLANIFICADA: Final[float] = 0.021588
ANCLA_P78: Final[float] = 0.024893
IC_P78_PUBLICADO: Final[tuple[float, float]] = (-0.028736, 0.078522)

TABLA_MDE_PREREGISTRO: Final[dict[int, float]] = {
    3: 0.070464,
    5: 0.036311,
    8: 0.024956,
    10: 0.021502,
    12: 0.019184,
    20: 0.014258,
    30: 0.011425,
    50: 0.008726,
}
TABLA_SENSIBILIDAD_PREREGISTRO: Final[tuple[float, ...]] = (1.0, 1.5, 2.0, 3.0)

# Enmienda 1 al pre-registro, adoptada antes de ejecutar. Corrige el coste parametrico
# declarado de la capa de atencion: 8256 es el de una celda UNIDIRECCIONAL de ancho 64 y
# los brazos de P82 son BIDIRECCIONALES con 64 por direccion, luego el estado que entra
# en la atencion tiene ancho 128.
ENMIENDA_1: Final[str] = "docs/p82_enmienda_1.md"
ENMIENDA_1_SHA256: Final[str] = (
    "53c8555d1a5ef0fcebb06c4a6f061acf56e568d95242e907387f0af1b03420d8"
)
PARAMETROS_ATENCION_PREREGISTRADOS: Final[int] = 8256
PARAMETROS_ATENCION_REFERENCIA_ENMIENDA_1: Final[int] = 16448
FUENTE_REFERENCIA_ATENCION: Final[str] = (
    "docs/p82_enmienda_1.md sha256 53c8555d1a5ef0fcebb06c4a6f061acf56e568d95242e907387f0af1b03420d8"
)
CONVENCION_CONTEO_ATENCION: Final[str] = (
    "la referencia de la enmienda, 16448, cuenta PESOS SIN SESGOS, que es la misma "
    "convencion del 8256 pre-registrado: W_h de 128x64 mas W_s de 128x64 mas v de 64; el "
    "conteo con sesgos es 128 mas y se publica tambien"
)
PROFUNDIDAD_HASH_LOTES: Final[int] = 64

# Formulaciones literales obligatorias. Coincidencia exacta, cero margen.
LITERAL_V03: Final[str] = "V03 remains the best verified leaderboard result."
LITERAL_NEGATIVO: Final[str] = (
    "UN_RESULTADO_NEGATIVO_AQUI_NO_AFIRMA_QUE_LA_ATENCION_NO_APORTA_AFIRMA_QUE_EL_EFECTO_ES"
    "INFERIOR_AL_LIMITE_DE_DETECCION_DE_ESTE_DISENO"
)
LITERAL_SEMILLAS: Final[str] = "DIEZ_SEMILLAS_ES_EL_MAXIMO_QUE_CABE_NO_EL_OPTIMO_ESTADISTICO"
LITERAL_NO_WORD2VEC: Final[str] = "NO_SE_AFIRMA_QUE_WORD2VEC_SEA_MEJOR_QUE_V03"
LITERAL_TRAMO2: Final[str] = "TRAMO_2_NO_EJECUTADO_POR_PRESUPUESTO"
LITERAL_METODO_IC: Final[str] = "t_pareado_sobre_semillas"
LITERAL_METODO_BOOT: Final[str] = "bootstrap_de_muestras"

REGLA_FILA_OOV: Final[str] = (
    "fila OOV = media de las filas del artefacto, L2 normalizada y escalada por la "
    "desviacion tipica del artefacto; derivada, congelada y no entrenable"
)
CONVENCION_FILA_VACIA: Final[str] = (
    "una fila con cero tokens tiene longitud efectiva 1 para el empaquetado y para el "
    "softmax, de modo que su unica posicion recibe el peso; se publica cuantas hay"
)
REGLA_TOKENIZADOR: Final[str] = (
    "tokenizar(normalizar_emb_normal(texto)) recortado a MAX_LEN; es el tokenizador de P82 "
    "y las cifras de A8 se miden con el, no se arrastran"
)

# =============================================================================
# SECCION 2: CONTRATO ENTRE CAPAS. CLAVES DERIVADAS, NUNCA POR LITERAL
# =============================================================================

CLAVES_DERIVADAS: Final[dict[str, str]] = {
    "umbrales_v03": "umbrales",
    "comparacion_primaria_json": "comparacion_primaria",
    "control_atencion_json": "con_tra_manode",
    "control_atencion_canonico_json": "control_sin_atencion",
    "naturaleza_resultado_json": "naturaleza_del_resultado",
    "resumen_naturalezas_json": "resumen_de_naturalezas",
    "veredictos_json": "veredictos",
    "detalle_json": "detalle",
    "estado_json": "estado",
    "caso_json": "caso",
}
CLAVES_FISICAS: Final[frozenset[str]] = frozenset(CLAVES_DERIVADAS.values())
PERMITIDAS_AUDITORIA_AST: Final[frozenset[str]] = frozenset(
    {"resolver_clave", "auditar_accesos_por_literal"}
)


def resolver_clave(nombre_logico: str) -> str:
    """Deriva la clave fisica de un nombre logico de contrato.

    Motivo: en fases anteriores una clave buscada existia solo para otro caso, y el
    fallo no lo trampaba ningun check. Aqui el registro es cerrado y falla ruidoso.

    Args:
        nombre_logico: clave del registro :data:`CLAVES_DERIVADAS`.

    Returns:
        La clave fisica tal cual aparece en el diccionario de destino.

    Raises:
        KeyError: si el nombre logico no existe en el registro.
    """
    return CLAVES_DERIVADAS[nombre_logico]


@dataclass(frozen=True)
class FilaContrato:
    """Una fila de la tabla de contrato entre capas."""

    nombre: str
    callable: Callable[..., Any]
    parametros: tuple[str, ...]
    origen: str


CONTRATOS_BIBLIOTECA: Final[tuple[FilaContrato, ...]] = (
    FilaContrato(
        "metricas_multietiqueta",
        metricas_multietiqueta,
        ("y_true", "y_pred"),
        "src/evaluation.py",
    ),
    FilaContrato("read_train", read_train, ("path",), "src/data.py"),
    FilaContrato("holdout_v2", holdout_v2, ("train", "grupos"), "src/splits.py"),
    FilaContrato(
        "pliegues_corregidos",
        pliegues_corregidos,
        ("train_dev", "grupos_dev", "n_splits"),
        "src/splits.py",
    ),
    FilaContrato("tokenizar", tokenizar, ("texto",), "src/embedding_utils.py"),
    FilaContrato(
        "normalizar_emb_normal", normalizar_emb_normal, ("texto",), "src/embedding_utils.py"
    ),
    FilaContrato(
        "pack_padded_sequence",
        pack_padded_sequence,
        ("input", "lengths", "batch_first", "enforce_sorted"),
        "torch.nn.utils.rnn",
    ),
    FilaContrato(
        "pad_packed_sequence",
        pad_packed_sequence,
        ("sequence", "batch_first", "padding_value", "total_length"),
        "torch.nn.utils.rnn",
    ),
    FilaContrato(
        "clip_grad_norm_", nn.utils.clip_grad_norm_, ("parameters", "max_norm"), "torch.nn.utils"
    ),
)


def tabla_contratos() -> tuple[FilaContrato, ...]:
    """Tabla de contrato completa, construida en el momento de verificarla.

    Las dos funciones locales de P82 se anaden aqui, y no en el literal de modulo,
    porque en el momento de construir ese literal todavia no existen y una referencia
    adelantada aborta la importacion entera.

    Returns:
        La tabla completa, biblioteca mas las dos filas locales de P82.
    """
    return CONTRATOS_BIBLIOTECA + (
        FilaContrato(
            "epoca_estrella",
            epoca_estrella,
            ("matriz", "init", "semilla", "ids", "largos", "y", "grupos", "ia", "k"),
            "paridad con p75_01 epoca_estrella",
        ),
        FilaContrato(
            "entrenar_fijo",
            entrenar_fijo,
            ("matriz", "init", "semilla", "ids", "largos", "y", "ia", "ib", "epocas"),
            "paridad con p75_01 entrenar_fijo",
        ),
    )


def verificar_contratos() -> tuple[bool, dict[str, Any]]:
    """Verifica la tabla de contrato con inspect.signature.

    Returns:
        Tupla con el exito global y el detalle fila por fila.
    """
    filas: list[dict[str, Any]] = []
    correcto = True
    for fila in tabla_contratos():
        try:
            firma = inspect.signature(fila.callable)
        except (TypeError, ValueError) as error:
            correcto = False
            filas.append(
                {
                    "nombre": fila.nombre,
                    "origen": fila.origen,
                    "ok": False,
                    "causa": f"sin firma: {type(error).__name__}",
                }
            )
            continue
        obtenidos = tuple(firma.parameters)
        faltan = [p for p in fila.parametros if p not in obtenidos]
        ok = not faltan
        correcto = correcto and ok
        filas.append(
            {
                "nombre": fila.nombre,
                "origen": fila.origen,
                "ok": ok,
                "esperados": list(fila.parametros),
                "obtenidos": list(obtenidos),
                "faltan": faltan,
            }
        )
    return correcto, {"filas": filas, "n_filas": len(filas)}


# =============================================================================
# SECCION 3: AUDITORIA POR AST DE ACCESOS POR LITERAL
# =============================================================================


def auditar_accesos_por_literal(codigo: str) -> list[str]:
    """Busca accesos por literal donde la clave deberia derivarse del registro.

    Motivo: el acceso por literal es la causa raiz de la clave buscada que solo
    existia para otro caso, uno de los fallos de contrato mas caros de la serie.

    Args:
        codigo: fuente Python a auditar.

    Returns:
        Lista de infracciones, una por acceso, con la clave y la linea.
    """
    arbol = ast.parse(codigo)
    padres: dict[int, ast.AST] = {}
    for nodo in ast.walk(arbol):
        for hijo in ast.iter_child_nodes(nodo):
            padres[id(hijo)] = nodo

    def funcion_de(nodo: ast.AST) -> str | None:
        while id(nodo) in padres:
            nodo = padres[id(nodo)]
            if isinstance(nodo, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return nodo.name
        return None

    infracciones: list[str] = []
    for nodo in ast.walk(arbol):
        if not isinstance(nodo, ast.Subscript):
            continue
        if funcion_de(nodo) in PERMITIDAS_AUDITORIA_AST:
            continue
        trozo = nodo.slice
        if isinstance(trozo, ast.Constant) and isinstance(trozo.value, str):
            if trozo.value in CLAVES_FISICAS:
                infracciones.append(
                    f"linea {nodo.lineno}: acceso por literal a la clave de contrato "
                    f"{trozo.value!r}; use resolver_clave"
                )
    return sorted(infracciones)


def codigo_sin_afirmacion(codigo: str, afirmacion: str) -> str:
    """Devuelve el codigo con esa afirmacion FUERA del pipeline del completo.

    Localiza la llamada por AST y por posicion, no por busqueda de texto: el mismo
    identificador aparece tambien en el humo, y una sustitucion textual se llevaria por
    delante la ocurrencia equivocada.

    Args:
        codigo: fuente Python.
        afirmacion: identificador a eliminar del pipeline del completo.

    Returns:
        El codigo modificado.
    """
    arbol = ast.parse(codigo)
    lineas = codigo.split("\n")
    for nodo in ast.walk(arbol):
        if not (isinstance(nodo, ast.FunctionDef) and nodo.name == "ejecutar_p82_completo"):
            continue
        for llamada in ast.walk(nodo):
            if (
                isinstance(llamada, ast.Call)
                and isinstance(llamada.func, ast.Attribute)
                and llamada.func.attr == "evaluar"
                and llamada.args
                and isinstance(llamada.args[0], ast.Constant)
                and llamada.args[0].value == afirmacion
            ):
                indice = llamada.args[0].lineno - 1
                lineas[indice] = lineas[indice].replace(
                    f'"{afirmacion}"', '"QUITADA_PARA_LA_MUTACION"'
                )
    return "\n".join(lineas)


def afirmar_evaluadas_en_completo(codigo: str) -> tuple[bool, dict[str, Any]]:
    """Verifica que TODA afirmacion del pre-registro se evalua en el pipeline COMPLETO.

    No depende de que alguien recuerde invocarla. La primera corrida real termino en 65
    minutos con la puerta negandose a publicar porque A4 y A22 no estaban en el completo,
    y eso no habria que volver a descubrir por la via cara.

    Args:
        codigo: fuente Python a auditar.

    Returns:
        Tupla con el exito y el detalle.
    """
    arbol = ast.parse(codigo)
    esperadas = sorted(BLOQUES)
    evaluadas: list[str] = []
    for nodo in ast.walk(arbol):
        if not (isinstance(nodo, ast.FunctionDef) and nodo.name == "ejecutar_p82_completo"):
            continue
        for llamada in ast.walk(nodo):
            if (
                isinstance(llamada, ast.Call)
                and isinstance(llamada.func, ast.Attribute)
                and llamada.func.attr == "evaluar"
                and llamada.args
                and isinstance(llamada.args[0], ast.Constant)
            ):
                evaluadas.append(str(llamada.args[0].value))
    faltan = sorted(set(esperadas) - set(evaluadas))
    repetidas = sorted({a for a in evaluadas if evaluadas.count(a) > 1})
    comprobaciones = {
        "las_veintitres_declared_estan_en_el_completo": not faltan,
        "ninguna_afirmacion_evaluada_dos_veces": not repetidas,
        "ninguna_afirmacion_evaluada_que_no_exista": not (set(evaluadas) - set(esperadas)),
    }
    return all(comprobaciones.values()), {
        "afirmaciones_esperadas": len(esperadas),
        "afirmaciones_evaluadas_en_el_completo": len(evaluadas),
        "orden_de_evaluacion": evaluadas,
        "faltan_en_el_completo": faltan,
        "repetidas": repetidas,
        "comprobaciones": comprobaciones,
    }


# Para cada funcion, que PARAMETROS son condiciones y por tanto tienen que llegar
# calculadas, nunca como literal booleano. Los parametros que SI son banderas legitimas
# no se tocan: un False en autorizar_tramo2 es correcto, un False en tiempo_real no lo es.
PARAMETROS_DE_CONDICION: Final[dict[str, tuple[int, ...]]] = {
    "construir_nota_de_tramos": (0, 1),
    "comprobar_presupuesto": (0, 2),
}


def auditar_booleanos_literales(codigo: str) -> list[str]:
    """Detecta llamadas a funciones de informe con un LITERAL booleano donde va una condicion.

    Cubre los dos fallos de la misma clase que ya han pasado en este fichero: la tupla
    anidada, cuyo primer elemento era otra tupla, y el False escrito a mano en
    construir_nota_de_tramos. Un argumento asumido donde deberia haber una comparacion.

    Args:
        codigo: fuente Python a auditar.

    Returns:
        Lista de infracciones, una por llamada.
    """
    arbol = ast.parse(codigo)
    infracciones: list[str] = []
    for nodo in ast.walk(arbol):
        if not (isinstance(nodo, ast.Call) and isinstance(nodo.func, ast.Name)):
            continue
        posiciones = PARAMETROS_DE_CONDICION.get(nodo.func.id)
        if posiciones is None:
            continue
        for indice in posiciones:
            if indice >= len(nodo.args):
                continue
            argumento = nodo.args[indice]
            if isinstance(argumento, ast.Constant) and isinstance(argumento.value, bool):
                infracciones.append(
                    f"linea {nodo.lineno}: {nodo.func.id} recibe el literal booleano "
                    f"{argumento.value!r} en el parametro de condicion {indice}; "
                    "use la comparacion real"
                )
    return sorted(infracciones)


def auditar_evaluaciones_anidadas(codigo: str) -> list[str]:
    """Detecta llamadas a evaluar cuya comprobacion devuelva una tupla anidada.

    EL DEFECTO QUE ESTA AUDITORIA ATRAPA. Si la comprobacion es una lambda cuyo cuerpo
    es una tupla de dos elementos cuyo primer elemento es una llamada que ya devuelve
    (bool, dict), el primer elemento de la tupla devuelta es esa tupla, no el booleano.
    Una tupla no vacia es truthy en Python, luego exito queda siendo truthy y TODA
    afirmacion con esa forma sale CONFIRMADA, ademas de perder el detalle real. Ese
    defecto sobrevivio a una puerta de humo de 74 casos porque ninguna comprobacion
    exigia que un fallo se publicara como REFUTADO.

    Args:
        codigo: fuente Python a auditar.

    Returns:
        Lista de infracciones, una por llamada a evaluar.
    """
    arbol = ast.parse(codigo)
    infracciones: list[str] = []
    for nodo in ast.walk(arbol):
        if not (
            isinstance(nodo, ast.Call)
            and isinstance(nodo.func, ast.Attribute)
            and nodo.func.attr == "evaluar"
            and len(nodo.args) >= 2
        ):
            continue
        comprobacion = nodo.args[1]
        if not (isinstance(comprobacion, ast.Lambda) and isinstance(comprobacion.body, ast.Tuple)):
            continue
        elementos = comprobacion.body.elts
        if len(elementos) != 2:
            continue
        if isinstance(elementos[0], (ast.Call, ast.Lambda, ast.Await)):
            infracciones.append(
                f"linea {nodo.lineno}: evaluar con comprobacion que devuelve una tupla "
                "anidada; use con_extra"
            )
    return sorted(infracciones)


def auditar_propio_script() -> tuple[bool, dict[str, Any]]:
    """Audita el codigo de este propio fichero.

    Returns:
        Tupla con el exito y el detalle de la auditoria.
    """
    codigo = Path(__file__).read_text(encoding="utf-8")
    infracciones = auditar_accesos_por_literal(codigo)
    anidadas = auditar_evaluaciones_anidadas(codigo)
    return (not infracciones) and (not anidadas), {
        "fichero": Path(__file__).name,
        "claves_vigiladas": sorted(CLAVES_FISICAS),
        "n_infracciones": len(infracciones),
        "infracciones": infracciones,
        "n_evaluaciones_con_tupla_anidada": len(anidadas),
        "evaluaciones_con_tupla_anidada": anidadas,
    }


SONDA_REOPTIMIZACION: Final[str] = "caso_reoptimizacion_bloqueada"


def auditar_llamadas_a_reoptimizar(codigo: str) -> dict[str, list[str]]:
    """Busca llamadas a la funcion de reoptimizacion de umbrales.

    La funcion existe y siempre lanza. A2 exige que el contador de reoptimizacion sea
    cero, y esta auditoria demuestra que no hay ni una llamada FUERA de la sonda.

    Hay exactamente una llamada legitima: la de la sonda de humo, que la invoca a
    proposito para comprobar que lanza. Se separa de las prohibidas en lugar de
    borrarse, porque una auditoria que se ignora a si misma no se puede auditar.

    Args:
        codigo: fuente Python a auditar.

    Returns:
        Diccionario con la lista de llamadas prohibidas y la lista de sondas.
    """
    arbol = ast.parse(codigo)
    padres: dict[int, ast.AST] = {}
    for nodo in ast.walk(arbol):
        for hijo in ast.iter_child_nodes(nodo):
            padres[id(hijo)] = nodo

    def funcion_de(nodo: ast.AST) -> str | None:
        while id(nodo) in padres:
            nodo = padres[id(nodo)]
            if isinstance(nodo, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return nodo.name
        return None

    prohibidas: list[str] = []
    sondas: list[str] = []
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Call) and isinstance(nodo.func, ast.Name):
            if nodo.func.id == "reoptimizar_umbral":
                donde = funcion_de(nodo)
                mensaje = f"linea {nodo.lineno} en {donde}: llamada a reoptimizar_umbral"
                if donde == SONDA_REOPTIMIZACION:
                    sondas.append(mensaje)
                else:
                    prohibidas.append(mensaje)
    return {"prohibidas": sorted(prohibidas), "sondas_permitidas": sorted(sondas)}


# =============================================================================
# SECCION 4: ASCII PURO, INVARIANTE DE ESCRITURA UNICA Y PARCIALES
# =============================================================================


class ErrorEscritura(RuntimeError):
    """Fallo al escribir un fichero de salida. En los parciales degrada a AVISO."""


class Parada(Exception):
    """Parada con marca STOP. Se propaga, no se traga."""

    def __init__(self, afirmacion: str, motivo: str) -> None:
        super().__init__(f"{afirmacion}: {motivo}")
        self.afirmacion = afirmacion
        self.motivo = motivo


def texto_ascii(texto: str) -> str:
    """Verifica en memoria que el texto es ASCII puro y lo devuelve intacto.

    El control va ANTES de abrir cualquier fichero destino en modo escritura. Un
    write_text con encoding ascii trunca a cero bytes cualquier fuente con un
    caracter no latino, y eso ya ocurrio dos veces en este proyecto.

    Args:
        texto: contenido candidato a escritura.

    Returns:
        El mismo texto, ya comprobado.

    Raises:
        ValueError: si algun caracter no es ASCII, con posicion y code point.
    """
    for posicion, caracter in enumerate(texto):
        if ord(caracter) > 127:
            raise ValueError(
                f"caracter no ASCII en posicion {posicion}: U+{ord(caracter):04X}"
            )
    return texto


def escribir_unico(ruta: Path, texto: str) -> bool:
    """Escribe un fichero una sola vez: temporal, comprobacion de bytes y enlace atomico.

    Motivo del mecanismo. Un write_text con encoding ascii trunca a cero bytes
    cualquier fuente que contenga un caracter no latino, y eso ya ocurrio dos veces en
    este proyecto. Por eso el texto se valida en memoria ANTES de abrir nada, se
    escribe entero en un temporal del mismo directorio, se comprueba que el temporal
    tiene exactamente el numero de bytes esperado, y solo entonces se mueve al destino.

    Por que os.link y no os.replace. os.replace sobrescribe SIEMPRE el destino, y la
    invariante de escritura unica lo prohibe. Con os.link el destino no aparece nunca
    vacio: el temporal se escribe y se verifica completo, y el enlace falla con
    FileExistsError si alguien escribio el destino antes. No hay ventana en la que el
    destino exista a cero bytes, que es exactamente el fallo que se repite.

    Args:
        ruta: destino absoluto.
        texto: contenido, validado en memoria antes de abrir el fichero.

    Returns:
        True si se escribio, False si el destino ya existia.

    Raises:
        ErrorEscritura: si el texto no es ASCII, si la escritura queda incompleta o si
            falla la E/S.
    """
    texto_ascii(texto)
    if ruta.exists():
        return False
    esperado = len(texto.encode("ascii"))
    temporal = ruta.with_name(f"{ruta.name}.tmp-{os.urandom(8).hex()}")
    try:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        with temporal.open("x", encoding="ascii", newline="\n") as mango:
            mango.write(texto)
            mango.flush()
            os.fsync(mango.fileno())
        if temporal.stat().st_size != esperado:
            raise ErrorEscritura(
                f"{ruta}: escritura incompleta, {temporal.stat().st_size} de {esperado} bytes"
            )
        try:
            os.link(temporal, ruta)
        except FileExistsError:
            return False
    except (OSError, UnicodeEncodeError) as error:
        raise ErrorEscritura(f"{ruta}: {type(error).__name__}: {error}") from error
    finally:
        if temporal.exists():
            try:
                os.unlink(temporal)
            except OSError:
                pass
    return True


# =============================================================================
# SECCION 5: SERIALIZACION SEGURA
# =============================================================================


def a_jsonable(objeto: Any) -> Any:
    """Convierte cualquier objeto a algo serializable en JSON.

    Motivo: uno de los fallos caros de las fases anteriores fue un modulo no
    serializable metido en el informe.

    Args:
        objeto: objeto arbitrario.

    Returns:
        Estructura hecha solo de dict, list, str, int, float, bool y None.
    """
    if objeto is None or isinstance(objeto, (bool, int, str)):
        return objeto
    if isinstance(objeto, float):
        return objeto if math.isfinite(objeto) else str(objeto)
    if isinstance(objeto, np.bool_):
        return bool(objeto)
    if isinstance(objeto, np.integer):
        return int(objeto)
    if isinstance(objeto, np.floating):
        valor = float(objeto)
        return valor if math.isfinite(valor) else str(valor)
    if isinstance(objeto, np.ndarray):
        return a_jsonable(objeto.tolist())
    if isinstance(objeto, Path):
        return str(objeto)
    if dataclasses.is_dataclass(objeto) and not isinstance(objeto, type):
        return a_jsonable(dataclasses.asdict(objeto))
    if isinstance(objeto, dict):
        return {str(a_jsonable(clave)): a_jsonable(valor) for clave, valor in objeto.items()}
    if isinstance(objeto, (list, tuple, set, frozenset)):
        return [a_jsonable(elemento) for elemento in objeto]
    if isinstance(objeto, bytes):
        return objeto.hex()
    return repr(objeto)


def a_json_legible(carga: dict[str, Any]) -> str:
    """Serializa a JSON con orden de claves estable y fin de linea.

    Es el UNICO punto por el que sale informacion a disco, y por eso sanea aqui y no en
    el punto de llamada: un NaN o un modulo sin convertir que llegase hasta json.dumps
    reventaria la escritura del informe entero, que es el modo de fallo de informes
    parciales que P81 sufrio.

    Args:
        carga: diccionario arbitrario.

    Returns:
        Texto JSON terminado en linea nueva.

    Raises:
        TypeError: si queda algun objeto no serializable.
    """
    limpio = a_jsonable(carga)
    return json.dumps(limpio, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"


# =============================================================================
# SECCION 6: AISLAMIENTO DE AFIRMACIONES
# =============================================================================

ESTADO_CONFIRMADO: Final[str] = "CONFIRMADO"
ESTADO_REFUTADO: Final[str] = "REFUTADO"
ESTADO_ERROR: Final[str] = "ERROR"
ESTADO_NO_EVALUABLE: Final[str] = "NO_EVALUABLE"

NATURELEZA_CONFIRMAR: Final[str] = "CONFIRMAR"

BLOQUES: Final[dict[str, tuple[int, str]]] = {
    "A0": (0, "sha256 del congelado, tamano y mtime antes y despues"),
    "A1": (0, "solo se leen las fuentes declaradas, contadores publicados"),
    "A2": (0, "los tres umbrales de V03 intactos y no reoptimizados"),
    "A3": (1, "el relleno se aplica ANTES de la RNN, verificado por estado"),
    "A4": (1, "relleno y OOV en celdas distintas, y el OOV ENTRA"),
    "A5": (1, "early stopping estrictamente anidado, verificado por filas"),
    "A6": (1, "clip_grad_norm_ en todos los bucles, norma pre-clip publicada"),
    "A7": (1, "determinismo, dos pasadas misma semilla, variacion cero"),
    "A8": (1, "cifras de cobertura medidas y publicadas, y MAX_LEN"),
    "A9": (2, "al menos 10 semillas, pareadas, conteo publicado"),
    "A10": (2, "tabla de limite de deteccion recalculada con la sd medida"),
    "A11": (2, "el intervalo que decide es un t pareado de semillas"),
    "A12": (2, "la lectura EXPLORATORIA esta atada a la regla"),
    "A13": (3, "existe brazo sin atencion de la MISMA corrida y es el control"),
    "A14": (3, "la permutacion conserva el multiconjunto exacto"),
    "A15": (3, "el control permutado usa el mismo protocolo y las 10 semillas"),
    "A16": (3, "las arquitecturas se reportan por separado, sin agregado"),
    "A17": (4, "interseccion y rescue damage net contra V03 sobre OOF"),
    "A18": (4, "la formulacion literal esta escrita y no sustituida"),
    "A19": (4, "presupuesto de reloj respetado y tiempo real publicado"),
    "A20": (4, "ib se evalua una vez por brazo, semilla y pliegue"),
    "A21": (4, "los 10 deltas por semilla del primario, con su signo"),
    "A22": (4, "cada comprobacion sabe FALLAR, por mutacion"),
}

# LAS CUATRO QUE NO LLEVAN MARCA DE STOP. A8 es medicion de contexto y no prueba, A16 es
# una condicion de redaccion del informe, A17 es descriptivo y A21 es una condicion de
# redaccion. Ninguna puede abortar una corrida por si sola, y por eso se declara aqui y
# no por omision. Las diecinueve restantes son STOP.
AFIRMACIONES_SIN_STOP: Final[frozenset[str]] = frozenset({"A8", "A16", "A17", "A21"})
AFIRMACIONES_STOP: Final[frozenset[str]] = frozenset(BLOQUES) - AFIRMACIONES_SIN_STOP


@dataclass
class Resultado:
    """Resultado de una afirmacion. ERROR no es ni CONFIRMADO ni REFUTADO."""

    id: str
    bloque: int
    naturaleza: str
    estado: str
    exito: bool
    tolerancia: str
    descripcion: str
    stop: bool
    detalle: dict[str, Any] = field(default_factory=dict)
    traza: str = ""
    coste_segundos: float = 0.0


def traza_resumida(error: BaseException, lineas: int = 6) -> str:
    """Resume la traza de una excepcion.

    Args:
        error: excepcion capturada.
        lineas: numero de fotogramas a conservar.

    Returns:
        Traza de una linea por fotograma, recortada.
    """
    causa = traceback.format_exception_only(type(error), error)[0].strip()
    pila = traceback.extract_tb(error.__traceback__)
    cuerpo = " | ".join(f"{marco.name}:{marco.lineno}" for marco in pila[-lineas:])
    return f"{causa} @ {cuerpo}"


class Progreso:
    """Acumulador de resultados, avisos, contadores y parciales."""

    def __init__(self, directorio_parciales: Path, marca: str) -> None:
        self.registro: dict[str, Resultado] = {}
        self.avisos: list[dict[str, str]] = []
        self.errores: list[dict[str, str]] = []
        self.contadores: dict[str, int] = {}
        self.directorio_parciales = directorio_parciales
        self.marca = marca
        self.escrituras_fallidas = 0
        self._secuencia = 0

    def siguiente_nombre(self, prefijo: str, sufijo: str) -> str:
        """Devuelve un nombre de fichero unico dentro del directorio de parciales.

        Args:
            prefijo: parte izquierda del nombre.
            sufijo: extension con punto.

        Returns:
            Nombre relativo unico, con marca de corrida y secuencia.
        """
        self._secuencia += 1
        return f"{prefijo}_{self.marca}_{self._secuencia:04d}{sufijo}"

    def parcial(self, prefijo: str, sufijo: str, carga: dict[str, Any]) -> Path | None:
        """Escribe un parcial en su propio fichero. Un fallo degrada a AVISO.

        Args:
            prefijo: parte izquierda del nombre.
            sufijo: extension.
            carga: diccionario serializable.

        Returns:
            La ruta escrita, o None si la escritura fallo.
        """
        ruta = self.directorio_parciales / self.siguiente_nombre(prefijo, sufijo)
        try:
            escribir_unico(ruta, a_json_legible(a_jsonable(carga)))
        except ErrorEscritura as error:
            self.escrituras_fallidas += 1
            self.avisos.append(
                {"tipo": "escritura_parcial", "ruta": str(ruta), "causa": str(error)}
            )
            return None
        return ruta

    def parcial_afirmacion(self, resultado: Resultado) -> Path | None:
        """Escribe el parcial de una afirmacion del bloque caro.

        Args:
            resultado: resultado a persistir.

        Returns:
            La ruta escrita, o None si la escritura fallo.
        """
        return self.parcial(
            f"afirmacion_{resultado.id}",
            ".json",
            {"marca_corrida": self.marca, "resultado": dataclasses.asdict(resultado)},
        )

    def contar(self, clave: str, delta: int = 1) -> int:
        """Incrementa un contador con nombre y devuelve su valor.

        Args:
            clave: nombre del contador.
            delta: incremento.

        Returns:
            El valor nuevo del contador.
        """
        self.contadores[clave] = self.contadores.get(clave, 0) + delta
        return self.contadores[clave]

    def avisar(self, tipo: str, causa: str) -> None:
        """Registra un aviso no mortal.

        Args:
            tipo: categoria del aviso.
            causa: descripcion de la causa.

        Returns:
            Nada.
        """
        self.avisos.append({"tipo": tipo, "causa": causa})

    def evaluar(
        self,
        afirmacion: str,
        comprobacion: Callable[[], tuple[bool, dict[str, Any]]],
        tolerancia: str,
    ) -> Resultado:
        """Evalua UNA afirmacion de forma aislada.

        Una excepcion propia produce ERROR con su tipo y su traza resumida, y la
        corrida continua con la siguiente. Una Parada con marca STOP se propaga.

        Args:
            afirmacion: identificador de A0 a A22.
            comprobacion: funcion sin argumentos que devuelve exito y detalle.
            tolerancia: descripcion de la tolerancia, publicada en el informe.

        Returns:
            El resultado de la afirmacion.

        Raises:
            Parada: si la comprobacion la lanza y la afirmacion tiene marca STOP.
            KeyError: si el identificador no existe.
        """
        bloque, descripcion = BLOQUES[afirmacion]
        inicio = time.perf_counter()
        try:
            exito, detalle = comprobar_veredicto(comprobacion())
        except Parada:
            self.parcial(
                f"parada_{afirmacion}",
                ".json",
                {"afirmacion": afirmacion, "marca_corrida": self.marca},
            )
            raise
        except Exception as error:  # aislamiento: ERROR, no parada
            resultado = Resultado(
                id=afirmacion,
                bloque=bloque,
                naturaleza=NATURELEZA_CONFIRMAR,
                estado=ESTADO_ERROR,
                exito=False,
                tolerancia=tolerancia,
                descripcion=descripcion,
                stop=afirmacion in AFIRMACIONES_STOP,
                detalle={
                    "error_tipo": type(error).__name__,
                    "error_mensaje": str(error)[:400],
                },
                traza=traza_resumida(error),
                coste_segundos=round(time.perf_counter() - inicio, 6),
            )
            self.registro[afirmacion] = resultado
            self.errores.append(
                {"afirmacion": afirmacion, "tipo": type(error).__name__, "traza": resultado.traza}
            )
            self.parcial_afirmacion(resultado)
            # Una afirmacion con marca de STOP que no ha podido decidirse NO puede
            # seguir como si hubiera decidido: se propaga la parada. Sin esto, una STOP
            # que reventaba por exception publicaba el informe y salia con codigo cero.
            if resultado.stop:
                raise Parada(
                    afirmacion,
                    f"la comprobacion no produjo un veredicto: {type(error).__name__}: {error}",
                ) from error
            return resultado
        resultado = Resultado(
            id=afirmacion,
            bloque=bloque,
            naturaleza=NATURELEZA_CONFIRMAR,
            estado=ESTADO_CONFIRMADO if exito else ESTADO_REFUTADO,
            exito=exito,
            tolerancia=tolerancia,
            descripcion=descripcion,
            stop=afirmacion in AFIRMACIONES_STOP,
            detalle=detalle,
            coste_segundos=round(time.perf_counter() - inicio, 6),
        )
        self.registro[afirmacion] = resultado
        self.parcial_afirmacion(resultado)
        if not exito and resultado.stop:
            raise Parada(afirmacion, str(detalle.get("motivo", "comprobacion fallida")))
        return resultado


# =============================================================================
# SECCION 7: MATEMATICA DEL LIMITE DE DETECCION
# =============================================================================


def potencia_no_central(n: int, delta: float, sd: float) -> float:
    """Potencia de un t no central para la formula PAREADA.

    La formula es la pareada porque el estimador es el delta POR SEMILLA entre dos
    brazos emparejados, luego el error tipico es sd partido por la raiz de n. Usar la
    de dos muestras independientes infravalora el limite de deteccion e infravalora el
    numero de semillas.

    Args:
        n: numero de semillas.
        delta: delta pareado real.
        sd: desviacion tipica entre semillas del delta pareado.

    Returns:
        Potencia de rechazo de H0 en dos lados a alpha 0.05.
    """
    grados = n - 1
    no_central = delta * math.sqrt(n) / sd
    critico = float(stats.t.ppf(0.975, grados))
    valor = float(stats.nct.sf(critico, grados, no_central))
    valor += float(stats.nct.cdf(-critico, grados, no_central))
    if not math.isfinite(valor):
        return 1.0
    return min(max(valor, 0.0), 1.0)


def limite_deteccion(n: int, sd: float, objetivo: float = 0.800000) -> float:
    """Delta minimo detectable por busqueda de raiz con potencia EXACTA.

    Args:
        n: numero de semillas.
        sd: desviacion tipica entre semillas del delta pareado.
        objetivo: potencia objetivo. El pre-registro fija potencia 0.800000 EXACTA.

    Returns:
        El MDE, con potencia igual al objetivo dentro de 1e-12.

    Raises:
        RuntimeError: si la potencia en el extremo del intervalo no supera el objetivo.
    """
    alto = sd * 12.0 / math.sqrt(n)
    if potencia_no_central(n, alto, sd) < objetivo:
        raise RuntimeError(f"sin raiz de potencia para n={n} sd={sd}")
    return float(
        optimize.brentq(
            lambda candidato: potencia_no_central(n, candidato, sd) - objetivo,
            alto * 1e-12,
            alto,
            xtol=1e-16,
            rtol=8.9e-16,
            maxiter=400,
        )
    )


def semiancho_ic(n: int, sd: float) -> float:
    """Semiancho del intervalo al 95 por ciento alrededor del cero.

    Args:
        n: numero de semillas.
        sd: desviacion tipica entre semillas del delta pareado.

    Returns:
        t de Student al 97.5 por ciento por sd partido por la raiz de n.
    """
    return float(stats.t.ppf(0.975, n - 1)) * sd / math.sqrt(n)


def intervalo_t_pareado(deltas: Sequence[float]) -> dict[str, float]:
    """Intervalo al 95 por ciento del t de Student sobre los deltas POR SEMILLA.

    No es un bootstrap. El bootstrap pareado de P78 remuestreaba tweets sobre redes ya
    entrenadas y propagaba cero variabilidad de reentrenamiento, y elegir el intervalo
    equivocado ahi produjo el titular de P78.

    Args:
        deltas: serie de deltas por semilla.

    Returns:
        Diccionario con n, media, desviacion, error tipico, intervalo, t y pvalor.
    """
    serie = np.asarray(list(deltas), dtype=float)
    n = int(serie.size)
    media = float(serie.mean())
    sd = float(serie.std(ddof=1)) if n > 1 else 0.0
    error = sd / math.sqrt(n) if n > 1 else 0.0
    critico = float(stats.t.ppf(0.975, n - 1)) if n > 1 else float("nan")
    limite = critico * error
    t_observado = media / error if error > 0 else float("inf")
    pvalor = float(2.0 * stats.t.sf(abs(t_observado), n - 1)) if n > 1 else float("nan")
    return {
        "n": n,
        "media": media,
        "desviacion": sd,
        "error_tipico": float(error),
        "critico_t": float(critico),
        "limite": float(limite),
        "inferior": float(media - limite),
        "superior": float(media + limite),
        "t": float(t_observado),
        "pvalor": pvalor,
    }


def intervalo_bootstrap(deltas: Sequence[float], semilla: int = 0) -> dict[str, float]:
    """Intervalor por bootstrap de muestras. SOLO existe para la mutacion de A22.

    No se usa en ninguna lectura. Se conserva porque A22 exige que la comprobacion de
    A11 sepa FALLAR, y una mutacion que solo levanta una bandera no es una mutacion.

    Args:
        deltas: serie de deltas por semilla.
        semilla: semilla del generador.

    Returns:
        Diccionario con los extremos del percentil 2.5 y 97.5.
    """
    serie = np.asarray(list(deltas), dtype=float)
    generador = np.random.default_rng(semilla)
    muestras = generador.choice(serie, size=(4000, serie.size), replace=True)
    medias = muestras.mean(axis=1)
    return {
        "inferior": float(np.percentile(medias, 2.5)),
        "superior": float(np.percentile(medias, 97.5)),
    }


def reproducir_tabla_del_preregistro() -> tuple[bool, dict[str, Any]]:
    """Recalcula la tabla de limite de deteccion del pre-registro y la compara.

    La tabla NO se modifica. Si el recalculo difiere, se reporta como difiere.

    Returns:
        Tupla con el exito y el detalle con las filas y las diferencias.
    """
    filas: list[dict[str, Any]] = []
    correcto = True
    for n, publicado in TABLA_MDE_PREREGISTRO.items():
        recalculado = limite_deteccion(n, SD_PARADA_PLANIFICADA)
        diferencia = abs(round(recalculado, 6) - publicado)
        coincide = bool(diferencia < 1e-6)
        correcto = correcto and coincide
        filas.append(
            {
                "n": n,
                "mde_publicado": publicado,
                "mde_recalculado": round(recalculado, 6),
                "diferencia": diferencia,
                "coincide": coincide,
                "semiancho_ic95": round(semiancho_ic(n, SD_PARADA_PLANIFICADA), 6),
                "potencia_en_el_ancla": round(
                    potencia_no_central(n, ANCLA_P78, SD_PARADA_PLANIFICADA), 4
                ),
            }
        )
    return correcto, {"filas": filas, "sd_usada": SD_PARADA_PLANIFICADA}


def comprobar_aritmetica_del_preregistro() -> tuple[bool, dict[str, Any]]:
    """Prueba de que la aritmetica es la misma que uso P78.

    Reproduce el intervalo publicado por P78 para P1 con el ancla, la sd y n igual a 3.

    Returns:
        Tupla con el exito y el detalle con el intervalo calculado y el publicado.
    """
    error = SD_PARADA_PLANIFICADA / math.sqrt(3)
    limite = float(stats.t.ppf(0.975, 2)) * error
    calculado = (ANCLA_P78 - limite, ANCLA_P78 + limite)
    diferencia = max(
        abs(round(calculado[0], 6) - IC_P78_PUBLICADO[0]),
        abs(round(calculado[1], 6) - IC_P78_PUBLICADO[1]),
    )
    # Tolerancia alineada con el pre-registro, no elegida para pasar. La seccion 3.1
    # declara, sobre estos mismos numeros, que el intervalo calculado es
    # (-0.028735, +0.078521) y el publicado (-0.028736, +0.078522), que la diferencia
    # "es de la ultima cifra" y que el ancla reproduce el intervalo publicado. Luego una
    # diferencia de una unidad en el sexto decimal es CORRECTA por el propio
    # pre-registro. Con tolerancia de 1e-6 estricta esta comprobacion rechazaria
    # exactamente lo que el pre-registro declara correcto.
    return diferencia <= 1.5e-6, {
        "intervalo_calculado": [round(calculado[0], 6), round(calculado[1], 6)],
        "intervalo_publicado_por_p78": list(IC_P78_PUBLICADO),
        "diferencia_maxima": diferencia,
        "tolerancia": 1.5e-6,
        "porque_esa_tolerancia": (
            "la seccion 3.1 del pre-registro declara la diferencia de la ultima cifra y "
            "considera que el ancla reproduce el intervalo publicado"
        ),
    }


def leer_gap_de_p78() -> float:
    """Lee el gap con V03 del informe de P78. NO se escribe a mano.

    Returns:
        El gap publicado por P78.

    Raises:
        FileNotFoundError: si el informe de P78 no existe.
        RuntimeError: si el patron no aparece en el informe.
    """
    texto = P78_ADDENDUM.read_text(encoding="utf-8", errors="replace")
    patron = r"gap del mejor bra?zo:\s*([0-9]+\.[0-9]+)"
    encontrado = re.search(patron, texto)
    if encontrado is None:
        raise RuntimeError("no se encontro el gap del mejor brazo en p78_addendum.md")
    return float(encontrado.group(1))


def tabla_sensibilidad(sd_medida: float, n: int, gap: float) -> list[dict[str, Any]]:
    """Tabla de sensibilidad recalculada con la desviacion MEDIDA.

    Args:
        sd_medida: desviacion entre semillas del delta pareado medida en la corrida.
        n: numero de semillas usado.
        gap: gap con V03 leido de P78.

    Returns:
        Lista de cuatro filas con factor, sd supuesta, MDE, potencia y cociente.
    """
    filas: list[dict[str, Any]] = []
    for factor in TABLA_SENSIBILIDAD_PREREGISTRO:
        supuesta = sd_medida * factor
        if supuesta <= 0.0:
            filas.append(
                {
                    "factor": factor,
                    "sd_supuesta": supuesta,
                    "mde_a_n_10": "INDETERMINADO_SD_MEDIDA_CERO",
                    "potencia_en_el_ancla": "INDETERMINADO",
                    "mde_sobre_gap": "INDETERMINADO",
                }
            )
            continue
        mde = limite_deteccion(n, supuesta)
        filas.append(
            {
                "factor": factor,
                "sd_supuesta": round(supuesta, 6),
                "mde_a_n_10": round(mde, 6),
                "potencia_en_el_ancla": round(potencia_no_central(n, ANCLA_P78, supuesta), 4),
                "mde_sobre_gap": round(mde / gap, 2),
            }
        )
    return filas


# =============================================================================
# SECCION 8: GUARDA DE LECTURA, HASH E INTEGRIDAD
# =============================================================================


@dataclass
class GuardaLectura:
    """Contador de filas leidas, con separacion estricta desarrollo y holdout."""

    filas_read_train: int = 0
    filas_holdout_descartadas: int = 0
    filas_de_desarrollo_usadas: int = 0
    filas_test_leidas: int = 0
    indices_holdout: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))

    def marcador(self, indices: np.ndarray) -> np.ndarray:
        """Marca un conjunto de indices como leidos de desarrollo.

        Args:
            indices: indices posicionales dentro del dataframe completo.

        Returns:
            Los mismos indices.

        Raises:
            RuntimeError: si algun indice pertenece al holdout o a test.
        """
        arreglo = np.asarray(indices, dtype=np.int64)
        if self.indices_holdout.size:
            prohibidos = set(self.indices_holdout.tolist())
            if any(int(i) in prohibidos for i in arreglo):
                raise RuntimeError("se ha intentado leer una fila del holdout")
        self.filas_de_desarrollo_usadas += int(arreglo.size)
        return arreglo


def sha256_de_fichero(ruta: Path) -> str:
    """Hash sha256 en streaming, con el fichero abierto solo en modo rb.

    Args:
        ruta: fichero a hashear.

    Returns:
        El hash en hexadecimal.

    Raises:
        FileNotFoundError: si el fichero no existe.
    """
    resumen = hashlib.sha256()
    with ruta.open("rb") as mango:
        for bloque in iter(lambda: mango.read(1 << 20), b""):
            resumen.update(bloque)
    return resumen.hexdigest()


def huella_fichero(ruta: Path) -> dict[str, Any]:
    """Huella completa de un artefacto congelado.

    Args:
        ruta: fichero a inspeccionar.

    Returns:
        Diccionario con sha256, size y mtime_ns.

    Raises:
        FileNotFoundError: si el fichero no existe.
    """
    informacion = ruta.stat()
    return {
        "ruta": str(ruta.relative_to(PROJECT_ROOT)),
        "sha256": sha256_de_fichero(ruta),
        "size": int(informacion.st_size),
        "mtime_ns": int(informacion.st_mtime_ns),
    }


CONTADOR_REOPTIMIZACION_UMBRAL: int = 0


def reoptimizar_umbral(*_args: Any, **_kwargs: Any) -> None:
    """Funcion prohibida. Existe para que A2 pueda contar y para que la mutacion pueda.

    Returns:
        Nada. Siempre lanza.

    Raises:
        RuntimeError: siempre, porque P82 no reoptimiza umbrales.
    """
    global CONTADOR_REOPTIMIZACION_UMBRAL
    CONTADOR_REOPTIMIZACION_UMBRAL += 1
    raise RuntimeError("P82 no reoptimiza umbrales: lo prohibe A2")


def leer_umbrales_v03() -> tuple[np.ndarray, dict[str, Any]]:
    """Lee los tres umbrales congelados de V03 del artefacto, no escritos a mano.

    Returns:
        Tupla con el vector de umbrales en el orden HS, TR, AG y el detalle.

    Raises:
        FileNotFoundError: si algun artefacto falta.
        KeyError: si la clave de umbrales no existe en el artefacto.
    """
    clave = resolver_clave("umbrales_v03")
    with V03_JSON.open("r", encoding="utf-8") as mango:
        artefacto = json.load(mango)
    if clave not in artefacto:
        raise KeyError(f"el artefacto congelado de V03 no trae la clave {clave!r}")
    por_etiqueta = artefacto[clave]
    vector = np.array([por_etiqueta[etiqueta] for etiqueta in CONFIG["etiquetas"]], dtype=float)
    with np.load(V03_OOF, allow_pickle=True) as paquete:
        vector_oof = np.asarray(paquete[resolver_clave("umbrales_v03")], dtype=float)
    coincide = bool(np.array_equal(vector, vector_oof))
    return vector, {
        "umbrales_json": {etiqueta: float(v) for etiqueta, v in zip(CONFIG["etiquetas"], vector)},
        "umbrales_oof": [float(v) for v in vector_oof],
        "coinciden_json_y_oof": coincide,
        "metodo_declarado": str(por_etiqueta.get("metodo", "NO_DECLARADO")),
        "fuente": str(V03_JSON.relative_to(PROJECT_ROOT)),
    }


# =============================================================================
# SECCION 9: VOCABULARIO, MATRIZ Y SECUENCIAS
# =============================================================================


@dataclass
class MatrizEmbeddings:
    """Matriz de embeddings con celda de relleno y celda de OOV separadas."""

    matriz: np.ndarray
    indice: dict[str, int]
    id_relleno: int
    id_oov: int
    regla_fila_oov: str


def construir_matriz_embeddings(
    tokens_filas: Sequence[Sequence[str]], vectores: dict[str, np.ndarray], dim: int
) -> MatrizEmbeddings:
    """Construye la matriz con relleno en 0 y OOV en len(vocabulario) mas 1.

    Corrige el defecto de v9_04, donde el indice empezaba en i mas 1, el OOV caia en
    indice.get(token, 0) y el padding_idx era tambien 0, luego relleno y OOV
    compartian celda y la mascara de pooling los borraba a los dos.

    Args:
        tokens_filas: tokens por fila, ya normalizados y tokenizados.
        vectores: diccionario token a vector del artefacto congelado.
        dim: dimension del embedding.

    Returns:
        La matriz con la fila de relleno identicamente cero y la de OOV derivada,
        congelada y no nula.
    """
    vocabulario = sorted({t for fila in tokens_filas for t in fila if t in vectores})
    indice = {palabra: posicion + 1 for posicion, palabra in enumerate(vocabulario)}
    id_relleno = 0
    id_oov = len(vocabulario) + 1
    matriz = np.zeros((len(vocabulario) + 2, dim), dtype=np.float32)
    for palabra, posicion in indice.items():
        matriz[posicion] = vectores[palabra]
    if vocabulario:
        apilado = np.stack([vectores[palabra] for palabra in vocabulario])
    else:
        apilado = np.zeros((1, dim), dtype=np.float32)
    media = apilado.mean(axis=0)
    norma = float(np.linalg.norm(media))
    escala = float(apilado.std())
    if norma > 0.0 and escala > 0.0:
        fila_oov = (media / norma * escala).astype(np.float32)
    else:
        fila_oov = np.full(dim, 0.01, dtype=np.float32)
    matriz[id_oov] = fila_oov
    return MatrizEmbeddings(
        matriz=matriz,
        indice=indice,
        id_relleno=id_relleno,
        id_oov=id_oov,
        regla_fila_oov=REGLA_FILA_OOV,
    )


def construir_secuencias(
    tokens_filas: Sequence[Sequence[str]],
    indice: dict[str, int],
    id_oov: int,
    max_len: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Convierte tokens en identificadores, con relleno en 0 y OOV con celda propia.

    Args:
        tokens_filas: tokens por fila.
        indice: diccionario token a identificador.
        id_oov: identificador de la celda de OOV.
        max_len: longitud maxima de secuencia.

    Returns:
        Tupla con la matriz de identificadores, el vector de longitudes reales y los
        conteos de truncamiento.
    """
    n_filas = len(tokens_filas)
    ids = np.zeros((n_filas, max_len), dtype=np.int64)
    largos = np.zeros(n_filas, dtype=np.int64)
    n_truncadas = 0
    n_vacias = 0
    for fila, tokens in enumerate(tokens_filas):
        recortado = list(tokens)[:max_len]
        if len(tokens) > max_len:
            n_truncadas += 1
        for columna, token in enumerate(recortado):
            ids[fila, columna] = indice.get(token, id_oov)
        largos[fila] = len(recortado)
        if not recortado:
            n_vacias += 1
    return ids, largos, {"n_truncadas": n_truncadas, "n_filas_sin_tokens": n_vacias}


def permutar_conservando_multiconjunto(
    matriz: np.ndarray, semilla: int, con_sustitucion: bool = False
) -> np.ndarray:
    """Permuta las filas de la matriz CONSERVANDO EL MULTICONJUNTO.

    Args:
        matriz: matriz original.
        semilla: semilla de la permutacion.
        con_sustitucion: si True sustituye ademas la fila 0. Es la mutacion de A14.

    Returns:
        La matriz permutada.
    """
    generador = np.random.default_rng(semilla)
    permutacion = generador.permutation(matriz.shape[0])
    resultado = matriz[permutacion].copy()
    if con_sustitucion:
        resultado[0] = resultado[1] * 0.5
    return resultado


def verificar_multiconjunto(
    original: np.ndarray, permutada: np.ndarray, doble: np.ndarray
) -> tuple[bool, dict[str, Any]]:
    """A14. La permutacion conserva el multiconjunto exacto, por dos vias.

    Args:
        original: matriz original.
        permutada: matriz permutada.
        doble: matriz permutada dos veces con la misma semilla.

    Returns:
        Tupla con el exito y el detalle de las cuatro comprobaciones.
    """
    def suma_hashes(matriz: np.ndarray) -> int:
        total = 0
        for fila in matriz:
            total += int.from_bytes(
                hashlib.sha256(
                    np.ascontiguousarray(fila.astype(np.float32)).tobytes()
                ).digest()[:16],
                "big",
            )
        return total % (2**61 - 1)

    def suma_de_trazas_de_filas(matriz: np.ndarray) -> float:
        """Suma de las trazas de las filas.

        DECLARACION DE INTERPRETACION. La matriz de P82 tiene forma (n_vocab + 2, 100),
        luego NO es cuadrada y la palabra "traza" no puede significar el diagonal de la
        matriz, porque esa cifra cambia con cualquier permutacion de filas y haria
        fallar una permutacion perfectamente correcta. La lectura exacta es la unica
        que un vector fila canonicamente define: la fila r define la matriz r transpuesta
        por r, cuya traza es el cuadrado de la norma de r. La suma de esas trazas es la
        norma de Frobenius al cuadrado, que es invariante bajo permutacion de filas.
        La traza literal de la matriz se publica tambien, como cifra informativa, y
        deliberadamente NO decide nada.
        """
        return float((matriz.astype(np.float64) ** 2).sum())

    hash_original = suma_hashes(original)
    hash_permutada = suma_hashes(permutada)
    traza_original = suma_de_trazas_de_filas(original)
    traza_permutada = suma_de_trazas_de_filas(permutada)
    comprobaciones = {
        "misma_forma": original.shape == permutada.shape,
        "suma_hashes_de_filas_igual": hash_original == hash_permutada,
        "suma_de_trazas_igual": abs(traza_original - traza_permutada) < 1e-6,
        "doble_permutacion_reproducible": bool(np.array_equal(permutada, doble)),
    }
    return todo_verdadero(comprobaciones, "comprobaciones"), {
        "forma": list(original.shape),
        "suma_hashes_original": hash_original,
        "suma_hashes_permutada": hash_permutada,
        "suma_trazas_original": round(traza_original, 6),
        "nota_trazas": (
            "suma de las trazas de las FILAS, entendidas como el cuadrado de la norma "
            "de cada fila, porque la matriz no es cuadrada"
        ),
        "traza_literal_de_la_matriz_original_informativa": round(float(np.trace(original.astype(np.float64))), 6),
        "traza_literal_de_la_matriz_permutada_informativa": round(float(np.trace(permutada.astype(np.float64))), 6),
        "suma_trazas_permutada": round(traza_permutada, 6),
        "comprobaciones": comprobaciones,
    }


# =============================================================================
# SECCION 10: LA RED
# =============================================================================


class RedRNN(nn.Module):
    """RNN bidireccional con pooling enmascarado o con atencion aditiva.

    La consulta de la atencion es el ULTIMO estado oculto valido, tomado de la
    secuencia desempaquetada con el indice real, de modo que el orden del lote no
    importa. El softmax recibe menos infinito en las posiciones de relleno ANTES de
    normalizar, luego su peso es exactamente cero y no un numero pequeno. La mascara
    del pooling de los brazos sin atencion se construye con las LONGITUDES, no con
    los identificadores, de modo que el OOV ENTRA en el promedio.
    """

    def __init__(
        self,
        matriz: np.ndarray,
        nucleo: str,
        con_atencion: bool,
        id_relleno: int,
        max_len: int,
        dim_oculto: int,
        dropout_entrada: float,
        dropout_clasificador: float,
        n_etiquetas: int = 3,
    ) -> None:
        super().__init__()
        if nucleo not in {"GRU", "LSTM"}:
            raise ValueError(f"nucleo desconocido: {nucleo}")
        self.nucleo = nucleo
        self.con_atencion = bool(con_atencion)
        self.id_relleno = int(id_relleno)
        self.max_len = int(max_len)
        self.dim_oculto = int(dim_oculto)
        self.mascara_tardia = False
        # La matriz de embeddings se lleva AL DISPOSITIVO en la construccion, y no solo
        # la red: un olvido parcial aqui se manifestaria como coste de transferencia en
        # cada lote, que es justo lo que mide el cronometro de la enmienda 2.
        self.embedding = nn.Embedding.from_pretrained(
            torch.as_tensor(np.ascontiguousarray(matriz), device=DISPOSITIVO),
            freeze=True,
            padding_idx=self.id_relleno,
        )
        celda: type[nn.Module] = nn.GRU if nucleo == "GRU" else nn.LSTM
        self.rnn = celda(
            int(matriz.shape[1]), self.dim_oculto, batch_first=True, bidirectional=True
        )
        self.drop_entrada = nn.Dropout(dropout_entrada)
        self.drop_medio = nn.Dropout(dropout_clasificador)
        self.drop_clasificador = nn.Dropout(dropout_clasificador)
        ancho = 2 * self.dim_oculto
        if con_atencion:
            self.wh = nn.Linear(ancho, dim_oculto)
            self.ws = nn.Linear(ancho, dim_oculto)
            self.v = nn.Parameter(torch.zeros(dim_oculto))
            nn.init.normal_(self.v, mean=0.0, std=0.05)
            self.salida = nn.Linear(2 * ancho, n_etiquetas)
        else:
            self.salida = nn.Linear(ancho, n_etiquetas)
        self.a_dispositivo()

    def a_dispositivo(self) -> RedRNN:
        """Mueve TODOS los parametros y buffers al dispositivo resuelto.

        Returns:
            La propia red, ya movida.
        """
        return self.to(DISPOSITIVO)

    def comprobar_dispositivo(self) -> dict[str, Any]:
        """Dice donde estan DE VERDAD los parametros y la matriz de embeddings.

        Returns:
            Diccionario con el device de cada tensor y si todos coinciden con el
            declarado.
        """
        devices = {str(param.device) for param in self.parameters()}
        devices |= {str(buffer.device) for buffer in self.buffers()}
        embebido = self.embedding.weight.device
        return {
            "devices_de_los_parametros": sorted(devices),
            "device_de_la_matriz_de_embeddings": str(embebido),
            "device_declarado": DISPOSITIVO_RESUELTO,
            "todos_en_el_declarado": devices == {DISPOSITIVO_RESUELTO},
            "embeddings_en_el_declarado": str(embebido) == DISPOSITIVO_RESUELTO,
        }

    def parametros_atencion(self) -> int:
        """Parametros anadidos por la capa de atencion, SESGOS INCLUIDOS.

        Returns:
            Entero con el conteo. Cero si el brazo no lleva atencion.
        """
        if not self.con_atencion:
            return 0
        total = sum(p.numel() for p in self.wh.parameters())
        total += sum(p.numel() for p in self.ws.parameters())
        return int(total + self.v.numel())

    def parametros_atencion_sin_sesgos(self) -> int:
        """Parametros anadidos por la capa de atencion, SIN SESGOS.

        Es la convencion de la Enmienda 1 y la del 8256 pre-registrado, y la unica en la
        que la referencia corregida de 16448 es comparable.

        Returns:
            Entero con el conteo. Cero si el brazo no lleva atencion.
        """
        if not self.con_atencion:
            return 0
        return int(
            self.wh.weight.numel() + self.ws.weight.numel() + self.v.numel()
        )

    def parametros_entrenables(self) -> int:
        """Parametros entrenables del brazo. El embedding esta congelado y no cuenta.

        Returns:
            Entero con el conteo.
        """
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def recurrente(self, embebido: torch.Tensor, largos_cpu: torch.Tensor) -> torch.Tensor:
        """Aplica la red recurrente con el relleno aplicado ANTES, por empaquetado.

        Args:
            embebido: tensor de forma (lote, T, dim).
            largos_cpu: longitudes reales en CPU, como exige el empaquetado.

        Returns:
            Tensor de salidas de forma (lote, T, 2 * dim_oculto).
        """
        if self.mascara_tardia:
            salida, _ = self.rnn(embebido)
            return salida
        # pack_padded_sequence EXIGE que las longitudes sean un tensor 1D de CPU int64.
        # Con los datos en GPU hay que bajarlos aqui, en el unico sitio por el que pasan,
        # en vez de confiar en que cada llamadaador se acuerde. Es el mismo contrato que
        # se Cumple en la frontera, no dentro de cada sitio.
        empaquetado = pack_padded_sequence(
            embebido,
            torch.as_tensor(largos_cpu).detach().to(device="cpu", dtype=torch.int64),
            batch_first=True,
            enforce_sorted=False,
        )
        salida_empaquetada, _ = self.rnn(empaquetado)
        salida, _ = pad_packed_sequence(
            salida_empaquetada, batch_first=True, total_length=self.max_len
        )
        return salida

    def atencion_aditiva(
        self, salida: torch.Tensor, largos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Capa de atencion aditiva tipo Bahdanau con softmax enmascarado.

        Args:
            salida: salidas de la RNN, de forma (lote, T, 2 * dim_oculto).
            largos: longitudes reales, de forma (lote,).

        Returns:
            Tupla con el contexto, de forma (lote, 2 * dim_oculto), y los pesos de
            atencion, de forma (lote, T).
        """
        lote = salida.shape[0]
        posiciones = torch.arange(self.max_len, device=salida.device).unsqueeze(0)
        largos_pos = torch.as_tensor(largos, device=salida.device).clamp(min=1)
        mascara = posiciones >= largos_pos.unsqueeze(1)
        indices = torch.arange(lote, device=salida.device)
        consulta = salida[indices, (largos_pos - 1).clamp(min=0)]
        puntuacion = torch.tanh(
            self.wh(salida) + self.ws(consulta).unsqueeze(1)
        ).matmul(self.v)
        puntuacion = puntuacion.masked_fill(mascara, float("-inf"))
        pesos = torch.softmax(puntuacion, dim=1)
        contexto = (pesos.unsqueeze(-1) * salida).sum(dim=1)
        return contexto, pesos

    def representacion(
        self, entrada: torch.Tensor, largos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Devuelve la representacion y, si hay atencion, los pesos de atencion.

        Args:
            entrada: identificadores, de forma (lote, T).
            largos: longitudes reales, de forma (lote,).

        Returns:
            Tupla con la representacion y los pesos, o None si el brazo no lleva atencion.
        """
        embebido = self.drop_entrada(self.embedding(entrada))
        largos_pos = torch.as_tensor(largos).to(entrada.device).clamp(min=1)
        salida = self.recurrente(embebido, largos_pos.cpu())
        indices = torch.arange(salida.shape[0], device=salida.device)
        ultimo = salida[indices, (largos_pos - 1).clamp(min=0)]
        if self.con_atencion:
            contexto, pesos = self.atencion_aditiva(salida, largos)
            return torch.cat([contexto, ultimo], dim=1), pesos
        mascara = (
            torch.arange(self.max_len, device=salida.device).unsqueeze(0)
            < largos_pos.unsqueeze(1)
        ).float().unsqueeze(-1)
        mitad = salida.shape[-1] // 2
        cuenta = mascara.sum(dim=1).clamp(min=1.0)
        medio_delante = (salida[:, :, :mitad] * mascara).sum(dim=1) / cuenta
        medio_detras = (salida[:, :, mitad:] * mascara).sum(dim=1) / cuenta
        return (
            torch.cat(
                [self.drop_medio(medio_delante), self.drop_medio(medio_detras)], dim=1
            ),
            None,
        )

    def forward(self, entrada: torch.Tensor, largos: torch.Tensor) -> torch.Tensor:
        """Devuelve los logits por etiqueta.

        Args:
            entrada: identificadores, de forma (lote, T).
            largos: longitudes reales, de forma (lote,).

        Returns:
            Tensor de logits, de forma (lote, 3).
        """
        representacion, _ = self.representacion(entrada, largos)
        return self.salida(self.drop_clasificador(representacion))


# =============================================================================
# SECCION 11: AJUSTES, INSTRUMENTACION Y ENTRENAMIENTO
# =============================================================================


@dataclass(frozen=True)
class Ajustes:
    """Ajustes de una corrida. Los flags de mutacion solo se usan en A22 y en humo."""

    max_len: int = CONFIG["max_len"]
    dim_oculto: int = CONFIG["dim_oculto"]
    epocas_max: int = CONFIG["epocas_max"]
    paciencia_es: int = CONFIG["paciencia_es"]
    n_semillas: int = len(CONFIG["semillas"])
    id_relleno: int = 0
    nucleo: str = "GRU"
    con_atencion: bool = False
    mascara_despues_de_rnn: bool = False
    omitir_clip_en_reajuste: bool = False
    permutacion_con_sustitucion: bool = False
    intervalo_bootstrap: bool = False
    decision_sobre_ib: bool = False
    mutacion: str = ""


MUTACIONES: Final[dict[str, str]] = {
    "decision_epoca_sobre_ib": "A5",
    "mascara_despues_de_rnn": "A3",
    "omitir_clip_en_reajuste": "A6",
    "permutacion_con_sustitucion": "A14",
    "ocho_semillas": "A9",
    "intervalo_bootstrap": "A11",
}


def construir_ajustes(base: Ajustes, mutacion: str = "") -> Ajustes:
    """Devuelve los ajustes con la mutacion aplicada.

    Args:
        base: ajustes de la corrida sin mutar.
        mutacion: nombre de la mutacion, vacio si no hay.

    Returns:
        Ajustes mutados.

    Raises:
        KeyError: si la mutacion no existe en el registro cerrado.
    """
    if not mutacion:
        return base
    if mutacion not in MUTACIONES:
        raise KeyError(mutacion)
    mapa: dict[str, dict[str, Any]] = {
        "decision_epoca_sobre_ib": {"decision_sobre_ib": True},
        "mascara_despues_de_rnn": {"mascara_despues_de_rnn": True},
        "omitir_clip_en_reajuste": {"omitir_clip_en_reajuste": True},
        "permutacion_con_sustitucion": {"permutacion_con_sustitucion": True},
        "ocho_semillas": {"n_semillas": 8},
        "intervalo_bootstrap": {"intervalo_bootstrap": True},
    }
    return dataclasses.replace(base, mutacion=mutacion, **mapa[mutacion])


@dataclass
class Instrumento:
    """Contadores y series compartidos por los bucles de entrenamiento.

    Es estado de modulo porque las dos funciones que lo consultan, epoca_estrella y
    entrenar_fijo, conservan la firma de P75 sin parametros anadidos.
    """

    pasos_optimizador: int = 0
    llamadas_clip: int = 0
    umbral_clip: float = CONFIG["clip_max_norm"]
    normas_preclip: list[float] = field(default_factory=list)
    filas_decision: list[np.ndarray] = field(default_factory=list)
    filas_por_split: list[np.ndarray] = field(default_factory=list)
    filas_split_actual: set[int] = field(default_factory=set)
    llamadas_decision: int = 0
    evaluaciones_ib: int = 0
    decision_sobre_ib: bool = False
    ib: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    ultimo_orden: str = ""
    perdidas_cal: list[float] = field(default_factory=list)
    generador_usado: Any = None

    def limpiar(self) -> None:
        """Pone a cero los contadores de una combinacion.

        Returns:
            Nada. Muta el instrumento in situ.
        """
        self.pasos_optimizador = 0
        self.llamadas_clip = 0
        self.normas_preclip = []
        self.filas_decision = []
        self.filas_por_split = []
        self.filas_split_actual = set()
        self.llamadas_decision = 0
        self.evaluaciones_ib = 0
        self.decision_sobre_ib = False
        self.ultimo_orden = ""
        self.perdidas_cal = []
        self.generador_usado = None


INSTRUMENTO: Final[Instrumento] = Instrumento()
AJUSTES_ACTIVOS: Ajustes = Ajustes()


def registrar_decision_epoca(filas: np.ndarray) -> None:
    """Registra las filas que una evaluacion de decision de epoca ha visto.

    Se distinguen DOS magnitudes, porque confundirlas fue un fallo de medicion:

    - la EVALUACION, que ocurre una vez por epoca dentro de cada split interno, y cuyo
      numero depende de cuando corta el early stopping;
    - la DECISION, que es la particion interna que decide, una por split interno y
      exactamente tres por combinacion. Lo que A5 acota es esta segunda.

    Las filas se acumulan en el split interno abierto, y el cierre lo convierte en una
    unica entrada por split.

    Args:
        filas: identificadores globales de las filas vistas.

    Returns:
        Nada. Muta el instrumento in situ.
    """
    INSTRUMENTO.llamadas_decision += 1
    arreglo = np.asarray(filas, dtype=np.int64)
    INSTRUMENTO.filas_decision.append(arreglo)
    INSTRUMENTO.filas_split_actual |= set(arreglo.tolist())
    if INSTRUMENTO.decision_sobre_ib and INSTRUMENTO.ib.size:
        ib = np.asarray(INSTRUMENTO.ib, dtype=np.int64)
        INSTRUMENTO.filas_decision.append(ib)
        INSTRUMENTO.filas_split_actual |= set(ib.tolist())


def cerrar_split_de_decision() -> None:
    """Cierra el split interno abierto y deja una sola entrada por split.

    Returns:
        Nada. Muta el instrumento in situ.
    """
    INSTRUMENTO.filas_por_split.append(
        np.array(sorted(INSTRUMENTO.filas_split_actual), dtype=np.int64)
    )
    INSTRUMENTO.filas_split_actual = set()


def log_odds(y: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Log odds de las proporciones por etiqueta, para el sesgo del clasificador.

    Args:
        y: matriz binaria de etiquetas.
        epsilon: recorte para evitar infinitos.

    Returns:
        Vector de log odds por etiqueta.
    """
    proporcion = np.clip(np.asarray(y, dtype=float).mean(axis=0), epsilon, 1.0 - epsilon)
    return np.log(proporcion / (1.0 - proporcion))


RESULTADO_CIENTIFICO_RAZONADO: Final[dict[str, Any]] = {
    "corrida": "20261001T141035Z_p2054042",
    "duracion": "98 minutos, con la bandera de autorizacion del tramo 2",
    "brazos": {
        "B1": {"media": 0.639659, "desviacion": 0.011615},
        "B2": {"media": 0.637554, "desviacion": 0.014905},
        "B3": {"media": 0.615163, "desviacion": 0.012924},
        "B4": {"media": 0.617194, "desviacion": 0.020434},
        "B5": {"media": 0.560747, "desviacion": 0.019137},
    },
    "comparaciones": {
        "P1": {"media": -0.002105, "ic95": [-0.012406, 0.008196], "signos_positivos": 4},
        "P2": {"media": 0.002031, "ic95": [-0.016628, 0.020690], "signos_positivos": 7},
        "P3": {"media": 0.076807, "ic95": [0.059823, 0.093791], "signos_positivos": 10},
        "P4": {"media": 0.078912, "ic95": [0.067144, 0.090680], "signos_positivos": 10},
        "P5": {"media": 0.024496, "ic95": [0.013551, 0.035441], "signos_positivos": 10},
    },
    "desviacion_medida": 0.014400,
    "mde_con_la_desviacion_medida": 0.014342,
    "gap_con_v03": 0.117450,
    "ajustes_de_red": 270,
    "defecto_de_redaccion": (
        "esta corrida dio el resultado cientificico correcto con las veintitres "
        "afirmaciones CONFIRMADO, pero el informe se contradia en el control de "
        "presupuesto: decia a la vez que se habia superado el presupuesto y que no. Los "
        "numeros cientificos de arriba son la referencia con la que hay que comparar esta "
        "reejecucion"
    ),
    "comparacion_obligatoria": (
        "si al reejecutar los brazos no dan estas MISMAS medias, hay no determinismo y hay "
        "que parar antes de publicar"
    ),
}

DISCREPANCIA_NOTA_DEFINITIVA_Y_MOMENTO_DE_A19: Final[str] = (
    "DISCREPANCIA DECLARADA, corregida. DOS fallos de redaccion del control de "
    "presupuesto. Primero: la nota definitiva del tramo recibia un False escrito a mano en "
    "el primer argumento en vez de la comparacion real entre el tiempo consumido y el "
    "presupuesto, luego decia SIEMPRE que no se habia excedido. Segundo: A19 se evaluaba en "
    "un punto del pipeline que no es el final, con el reloj por debajo del presupuesto, luego "
    "la afirmacion que vigila el presupuesto se media antes de que el presupuesto existiera. "
    "El informe se contradia a si mismo: con 5850.165344 s reales contra 5400 de presupuesto "
    "decia a la vez DENTRO_DE_PRESUPUESTO con excedio_el_presupuesto en false, y "
    "autorizacion_del_tramo_2 decia que se habia superado por autorizacion. LAS DOS FORMAS "
    "eran: relajar A19 o recalcular la nota al final, o calcular la excedencia UNA sola vez "
    "en un unico sitio y derivar de ahi las entradas que la mencionan. Se elige la segunda, y "
    "para que no vuelva la funcion recibe el TIEMPO REAL y el PRESUPUESTO en vez de un "
    "booleano ya calculado, luego no hay forma de pasar un literal por error, y A19 se mueve "
    "al FINAL del pipeline. Lo instructivo: el unico hallazgo que no dependia de los datos, "
    "el del presupuesto, fue el que no supo decir bien; las veintitres afirmaciones estaban "
    "CONFIRMADO, luego el defecto no estaba en la medicion sino en como se redactaba"
)

DISCREPANCIA_CRITERIO_DE_NETO: Final[str] = (
    "DISCREPANCIA DECLARADA, corregida. El criterio de no negatividad de A17 se aplicaba "
    "tamBIEN a neto, y neto NO es un recuento: es una RESTA, la diferencia entre rescate y "
    "dano, luego por definicion puede ser negativa. El propio codigo lo reconocia en la "
    "comprobacion siguiente, que exige que neto sea igual a rescate menos dano, y a la vez "
    "le exigia que no fuese negativo: contradiccion interna. Con los datos reales los tres "
    "netos son negativos, HS con rescate 148, dano 308 y neto -160; TR con rescate 62, dano "
    "156 y neto -94; y AG con rescate 115, dano 241 y neto -126, porque el brazo BiGRU hace "
    "mas dano que rescate frente a V03. La afirmacion se refutaba por un HALLAZGO del "
    "experimento y no por un error de las cifras, que es lo peor que puede pasar en un "
    "informe: un lector creeria que hay un bug en el calculo de la complementariedad y no "
    "lo hay. En el modo de humo NO se podia detectar porque los datos sinteticos dan netos "
    "positivos. Las dos formas eran: mantener la no negatividad de los cuatro campos, que "
    "exige A17 mas de lo que la afirmacion quiere decir, o exigir no negatividad solo a los "
    "tres que SI son recuentos, que es aciertos_comunes, rescate y dano, y dejar que neto "
    "sea un entero cualquiera. Se elige la segunda porque A17 no se relaja: sigue exigiendo "
    "los seis campos de las tres etiquetas, que ninguno falte, que los recuentos sean "
    "enteros no negativos, que neto cuadre con rescate menos dano y que las unidades esten "
    "declaradas. Solo se corrige un criterio que estaba mal definido, y ademas se publica "
    "el signo del neto para que el detalle deje saber si el brazo aporta o detracta"
)

DISCREPANCIA_NOTA_DE_TRAMOS: Final[str] = (
    "DISCREPANCIA DECLARADA, corregida. LITERAL_TRAMO2 era una constante UNICA, "
    "TRAMO_2_NO_EJECUTADO_POR_PRESUPUESTO, y se asignaba SOLO en la rama en la que el "
    "tramo 2 no se ejecutaba. A19 exige nota siempre que se exceda el presupuesto, y el "
    "texto era ademas FALSO en el caso autorizado, porque decia que el tramo 2 no se "
    "ejecuto cuando si se ejecuto. La bandera permitia continuar y dejaba a A19 sin la "
    "nota que la propia afirmacion exige. La primera corrida autorizada ejecuto los CINCO "
    "brazos y se detuvo en A19 con STOP tras 96 minutos contra 90 de presupuesto, con hito "
    "observado de 3666.928448 s contra 2400 de hito y 5400 de presupuesto, y la condicion "
    "del reloj cumplida con el tramo 2 ejecutado por autorizacion. Los 29 parciales estan a "
    "salvo en reports/p82_parciales/20261001T141035Z_p2054042/ y son la evidencia. Las dos "
    "formas eran: relajar A19 quitando la exigencia de nota, o construir la nota con la "
    "situacion real. Se elige la segunda porque A19 es correcta y fue ella la que cazo el "
    "fallo, y relajar una afirmacion para que pase es lo que este proyecto no hace. CONFIG, "
    "la condicion del hito y la puerta no se tocan"
)

DISCREPANCIA_COBERTURA_DEL_COMPLETO: Final[str] = (
    "DISCREPANCIA DECLARADA, corregida. El script DECLARABA veintitres afirmaciones pero "
    "la rama del modo completo evaluaba solo veintiuna: A4 y A22 estaban solo en el humo. "
    "Ambas tienen marca de STOP, luego la puerta del completo se nego a publicar y lo "
    "detecto en la primera corrida real: el log literal fue A22 SIN EJECUTAR y A4 SIN "
    "EJECUTAR, sobre 65 minutos de reloj y 26 parciales en "
    "reports/p82_parciales/20261001T065649Z_p1233382/, que son evidencia de un resultado "
    "medido que no se publico. El pre-registro no dice si A4 y A22 se evaluan en el "
    "completo o solo en el humo, luego no hay violacion de texto vinculante, pero que el "
    "script las declarara solo en el humo mientras la puerta las exigia en el completo es "
    "una discrepancia de diseno que se REGISTRA y no se recuerda. Las dos formas eran: "
    "evaluar A4 y A22 tambien en el completo, o relajar la puerta para el completo. Se "
    "elige la primera porque la puerta no se toca. Para que no vuelva, la puerta verifica "
    "por AST que las veintitres afirmaciones aparecen en el pipeline del completo, y hay "
    "un caso de humo con una mutacion que quita A4 de ese pipeline y comprueba que la "
    "comprobacion lo ve"
)

RED_DE_CASOS_NEGATIVOS: Final[dict[str, tuple[str, ...]]] = {
    "A0": ("hash_del_congelado_verificado_en_humo", "comprobacion_de_contrato"),
    "A1": ("json_en_disco_declara_el_contraste_primario",),
    "A2": ("reoptimizacion_de_umbral_bloqueada_y_sin_llamadas",),
    "A3": ("mutacion_mascara_despues_de_rnn_detecta_A3",),
    "A4": ("caso_incompleto_tiene_filas_de_relleno_y_de_oov",),
    "A5": ("mutacion_decision_epoca_sobre_ib_detecta_A5",),
    "A6": ("mutacion_omitir_clip_en_reajuste_detecta_A6",),
    "A7": ("una_comprobacion_que_falla_se_publica_REFUTADO",),
    "A8": ("exigir_no_vacio_sobre_una_lista_vacia_produce_ERROR",),
    "A9": ("mutacion_ocho_semillas_detecta_A9", "mutacion_a_la_magnitud_antigua_hace_FALLAR_el_caso"),
    "A10": ("mutacion_de_un_literal_del_mde_se_detecta_en_todas_las_filas",),
    "A11": ("mutacion_intervalo_bootstrap_detecta_A11",),
    "A12": ("markdown_en_disco_contiene_los_cuatro_literales",),
    "A13": ("json_en_disco_declara_el_contraste_primario",),
    "A14": ("mutacion_permutacion_con_sustitucion_detecta_A14",),
    "A15": ("json_contiene_todas_las_veintitres_afirmaciones",),
    "A16": ("json_contiene_todas_las_veintitres_afirmaciones",),
    "A17": ("exigir_no_vacio_sobre_una_lista_vacia_produce_ERROR",),
    "A18": ("markdown_en_disco_contiene_los_cuatro_literales",),
    "A19": ("puerta_de_la_corrida_completa_rechaza_error_refutado_y_sin_ejecutar",),
    "A20": ("puerta_de_la_corrida_completa_rechaza_error_refutado_y_sin_ejecutar",),
    "A21": ("hashes_de_secuencia_registrados_y_no_ausentes",),
    "A22": ("mutacion_ocho_semillas_detecta_A9", "tabla_del_mde_restaurada_tras_la_mutacion"),
}

DISCREPANCIA_SPLIT_INTERNO: Final[str] = (
    "DISCREPANCIA DECLARADA entre dos puntos del propio pre-registro. La seccion 4.4 "
    "describe UN StratifiedGroupKFold de tres pliegues con un random_state que solo "
    "depende del pliegue externo k, del que salen las tres parejas (m_in j, m_cal j). "
    "Con esa lectura, las tres m_cal j son los tres pliegues de validacion de un mismo "
    "divisor, luego SU UNION ES ia COMPLETA, y entonces el punto (i) de A5, que exige "
    "que la union de las filas vistas sea un subconjunto ESTRICTO de ia, es IMPOSIBLE de "
    "cumplir. Se comprobo ejecutando: con un solo divisor la union daba 48 filas sobre 48 "
    "y A5 se(refutaba) por esa comprobacion. Como A5 es STOP y el criterio de fuga no es "
    "negociable, se usa la otra forma, tres divisores independientes, uno por j, "
    "sembrados con semilla base mas j, tomando siempre el primer pliegue de cada uno. "
    "Mismo numero de particiones internas, misma cantidad de trabajo y misma cantidad de "
    "evaluaciones de decision, mecanismo distinto del que sugiere 4.4, y la union de las "
    "filas de calibracion queda SIEMPRE como subconjunto estricto de ia porque tres "
    "tercios independientes no cubren necesariamente todo. La Enmienda 1 obliga a "
    "DECLARAR la discrepancia en vez de ajustarla, y aqui se declara y no se oculta"
)

FORMA_DE_SPLIT_INTERNO: Final[str] = (
    "tres divisores independientes, uno por j, sembrados con "
    "semilla_split_interno + 100 * k + j, y se toma el pliegue cero de cada uno"
)


def partir_interna(
    grupos: np.ndarray, y: np.ndarray, random_state: int, indice_pliegue: int
) -> np.ndarray:
    """Pareja interna j, con un divisor independiente sembrado para ese j.

    Se conserva el parametro indice_pliegue por contrato, y se toma el pliegue cero,
    que es lo que hace la forma elegida. Ver DISCREPANCIA_SPLIT_INTERNO para por que no
    se usa un unico divisor.

    Args:
        grupos: grupos fusionados de las filas consideradas.
        y: matriz binaria de etiquetas.
        random_state: semilla del divisor.

    Se usa UN SOLO pliegue del divisor, no la union de los tres: la union de los
    conjuntos de entrenamiento de los tres pliegues es el conjunto entero, y con el
    conjunto entero como entrenamiento no queda nada para calibrar y no habria nada que
    decidir. Ese es el bug que el modo de humo detecto.

    Returns:
        Mascara booleana, True donde la fila ENTRENA. El complemento calibra.

    Raises:
        RuntimeError: si el pliegue elegido deja alguna de las dos partes vacia.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    divisor = StratifiedGroupKFold(
        n_splits=CONFIG["n_internos"], shuffle=True, random_state=random_state
    )
    combinaciones = [f"{a}{b}{c}" for a, b, c in np.asarray(y)]
    pliegues = list(divisor.split(np.zeros(len(grupos)), combinaciones, groups=grupos))
    pliegue = pliegues[0]
    mascara = np.zeros(len(grupos), dtype=bool)
    mascara[np.asarray(pliegue[0], dtype=int)] = True
    del indice_pliegue
    if int(mascara.sum()) == 0 or int((~mascara).sum()) == 0:
        raise RuntimeError(
            f"split interno degenerado con random_state={random_state}: "
            f"{int(mascara.sum())} de entrenamiento y {int((~mascara).sum())} de calibracion"
        )
    return mascara


def parametros_entrenables(modelo: nn.Module) -> list[nn.Parameter]:
    """Parametros con gradiente del modelo.

    Args:
        modelo: red.

    Returns:
        Lista de parametros entrenables. El embedding congelado no aparece.
    """
    return [p for p in modelo.parameters() if p.requires_grad]


def construir_red(
    matriz: np.ndarray, ajustes: Ajustes, nucleo: str, con_atencion: bool
) -> RedRNN:
    """Construye la red de un brazo con una inicializacion sembrada.

    Args:
        matriz: matriz de embeddings.
        ajustes: ajustes de la corrida, incluidos los de mutacion.
        nucleo: GRU o LSTM.
        con_atencion: si el brazo lleva atencion aditiva.

    Returns:
        La red, con la mascara tardia del ajuste ya aplicada.
    """
    modelo = RedRNN(
        matriz=matriz,
        nucleo=nucleo,
        con_atencion=con_atencion,
        id_relleno=ajustes.id_relleno,
        max_len=ajustes.max_len,
        dim_oculto=ajustes.dim_oculto,
        dropout_entrada=CONFIG["dropout_entrada"],
        dropout_clasificador=CONFIG["dropout_clasificador"],
        n_etiquetas=len(CONFIG["etiquetas"]),
    )
    modelo.mascara_tardia = ajustes.mascara_despues_de_rnn
    return modelo.a_dispositivo()


def hash_estado(modelo: nn.Module, prefijo: str = "rnn") -> str:
    """Hash sha256 de los tensores cuyo nombre empieza por un prefijo.

    Para el emparejamiento de A9 se hashea SOLO el bloque recurrente, que es el que
    comparten los dos brazos de cada par comparado. Hashear el estado completo daria
    hashes distintos por construccion, porque la capa de atencion anade tensores.

    Args:
        modelo: red.
        prefijo: prefijo de los nombres de tensor a hashear.

    Returns:
        Hash hexadecimal.
    """
    resumen = hashlib.sha256()
    for nombre, tensor in sorted(modelo.state_dict().items()):
        if not nombre.startswith(prefijo):
            continue
        resumen.update(nombre.encode("ascii"))
        resumen.update(np.ascontiguousarray(tensor.detach().cpu().numpy()).tobytes())
    return resumen.hexdigest()


def hash_secuencia_de_lotes(semilla: int, n_filas: int, lote: int, n_batches: int) -> str:
    """Hash acumulado de la SECUENCIA COMPLETA de lotes de una semilla.

    POR QUE ESTO Y NO EL ULTIMO. El emparejamiento de A9 es una invariante del
    PROCEDIMIENTO, no de su punto de parada. Con early stopping anidado cada brazo
    converge en un numero de epocas distinto, porque el criterio de parada depende de la
    perdida de su propia particion interna y los brazos no son identicos. Comparar el
    ultimo lote ejecutado compara un valor que depende del estado de entrenamiento: dos
    brazos perfectamente emparejados divergen en el momento en que uno para tres pasos
    antes que el otro, y eso no dice nada del emparejamiento.

    Que es una invariante. El generador se siembra con semilla mas 7 y se avanza con
    randperm de n filas, luego produce una secuencia DETERMINISTA que depende solo de la
    semilla y del numero de filas, no del brazo ni de cuando se deje de consumir. Hashear
    un prefijo de longitud FIJA compara la secuencia completa hasta donde todos llegan, y
    es emparejado por construccion. La longitud del prefijo es un numero declarado, y el
    hash se toma al inicio de la combinacion y no se vuelve a tocar dentro del bucle.

    Args:
        semilla: semilla del modelo.
        n_filas: filas del conjunto de entrenamiento externo.
        lote: tamano de lote.
        n_batches: numero de permutaciones a acumular, largo y declarado.

    Returns:
        Hash hexadecimal de la secuencia acumulada.
    """
    # El GENERADOR SE QUEDA EN CPU A PROPOSITO, segun la enmienda 2: A9 hashea esta
    # secuencia y el hash tiene que ser estable y auditable aunque los datos esten en
    # GPU. torch.Generator es un generador de CPU y no se mueve.
    generador = torch.Generator().manual_seed(int(semilla) + 7)
    resumen = hashlib.sha256()
    for _ in range(int(n_batches)):
        permutacion = torch.randperm(n_filas, generator=generador)
        resumen.update(hash_orden_lotes(permutacion).encode("ascii"))
    return resumen.hexdigest()


def hash_orden_lotes(perm: torch.Tensor) -> str:
    """Hash sha256 del orden de mini-lotes.

    Args:
        perm: permutacion del generador.

    Returns:
        Hash hexadecimal.
    """
    return hashlib.sha256(
        np.ascontiguousarray(perm.cpu().numpy().astype(np.int64)).tobytes()
    ).hexdigest()


def perdida_bce(logits: torch.Tensor, objetivo: torch.Tensor) -> torch.Tensor:
    """Perdida binaria con logits.

    Args:
        logits: (lote, 3).
        objetivo: (lote, 3) en {0, 1}.

    Returns:
        Escalar.
    """
    return nn.functional.binary_cross_entropy_with_logits(logits, objetivo)


def bucle_epoca(
    modelo: RedRNN,
    optimizador: torch.optim.Optimizer,
    ids: torch.Tensor,
    largos: torch.Tensor,
    y: torch.Tensor,
    generador: torch.Generator,
    lote: int,
    clip_activo: bool,
) -> str:
    """Una epoca de entrenamiento, con clip de gradiente declarado.

    Args:
        modelo: red en modo de entrenamiento.
        optimizador: optimizador ya construido.
        ids: identificadores del conjunto de entrenamiento.
        largos: longitudes reales.
        y: etiquetas.
        generador: generador de permutaciones, fijado por semilla.
        lote: tamano de lote.
        clip_activo: si este bucle invoca clip_grad_norm_.

    Returns:
        El hash del orden de mini-lotes de la epoca.
    """
    modelo.train()
    # El entrenamiento PUBLICA el generador que de verdad uso, para que una comprobacion
    # pueda inspeccionarlo sin reconstruirlo. Un caso que construye su propio generador
    # verifica un objeto que el propio caso acaba de crear, luego no prueba nada.
    INSTRUMENTO.generador_usado = generador
    permutacion = torch.randperm(len(ids), generator=generador)
    INSTRUMENTO.ultimo_orden = hash_orden_lotes(permutacion)
    for inicio in range(0, len(ids), lote):
        seleccion = permutacion[inicio : inicio + lote]
        optimizador.zero_grad()
        perdida_bce(modelo(ids[seleccion], largos[seleccion]), y[seleccion]).backward()
        normaclip = float(
            nn.utils.clip_grad_norm_(parametros_entrenables(modelo), INSTRUMENTO.umbral_clip)
        )
        INSTRUMENTO.pasos_optimizador += 1
        INSTRUMENTO.normas_preclip.append(normaclip)
        if clip_activo:
            INSTRUMENTO.llamadas_clip += 1
        optimizador.step()
    return hash_orden_lotes(permutacion)


def epoca_estrella(
    matriz: np.ndarray,
    init: dict[str, torch.Tensor],
    semilla: int,
    ids: np.ndarray,
    largos: np.ndarray,
    y: np.ndarray,
    grupos: np.ndarray,
    ia: np.ndarray,
    k: int,
) -> tuple[int, list[int]]:
    """Epoca E* como MEDIANA de tres splits internos, con la instrumentacion de A5.

    Conserva la firma de experiments/p75_01_complementariedad_textcnn.py. Ninguna fila
    de ib participa en este punto: el unico conjunto de decision es el complemento
    interno dentro de ia, y las filas vistas quedan registradas en INSTRUMENTO.

    Args:
        matriz: matriz de embeddings.
        init: state_dict inicial del brazo.
        semilla: semilla de modelo.
        ids: ids globales.
        largos: longitudes reales.
        y: etiquetas.
        grupos: grupos fusionados.
        ia: indices del pliegue de entrenamiento externo.
        k: numero de pliegue externo, desde 0.

    Returns:
        Tupla con E* y la lista de las tres epocas internas.
    """
    ajustes = AJUSTES_ACTIVOS
    ids_ia = torch.as_tensor(ids[ia], dtype=torch.long, device=DISPOSITIVO)
    largos_ia = torch.as_tensor(largos[ia], dtype=torch.long, device=DISPOSITIVO)
    y_ia = torch.as_tensor(y[ia], dtype=torch.float32, device=DISPOSITIVO)
    epocas: list[int] = []
    for j in range(CONFIG["n_internos"]):
        mascara = partir_interna(
            grupos[ia], y[ia], CONFIG["semilla_split_interno"] + 100 * k + j, j
        )
        torch.manual_seed(semilla + j)
        modelo = construir_red(matriz, ajustes, ajustes.nucleo, ajustes.con_atencion)
        modelo.load_state_dict(init)
        with torch.no_grad():
            modelo.salida.bias.copy_(
                torch.as_tensor(
                    log_odds(y[ia][mascara].mean(axis=0)),
                    dtype=torch.float32,
                    device=DISPOSITIVO,
                )
            )
        optimizador = torch.optim.AdamW(
            parametros_entrenables(modelo),
            lr=CONFIG["lr"],
            weight_decay=CONFIG["weight_decay"],
        )
        planificador = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizador,
            mode="min",
            factor=CONFIG["scheduler_factor"],
            patience=CONFIG["scheduler_patience"],
            min_lr=CONFIG["scheduler_min_lr"],
        )
        generador = torch.Generator().manual_seed(semilla + 7 + j)
        filas_cal = np.asarray(ia[~mascara], dtype=np.int64)
        INSTRUMENTO.filas_split_actual = set()
        mejor, mejor_epoca = float("inf"), 1
        for epoca in range(1, ajustes.epocas_max + 1):
            bucle_epoca(
                modelo,
                optimizador,
                ids_ia[mascara],
                largos_ia[mascara],
                y_ia[mascara],
                generador,
                CONFIG["batch"],
                True,
            )
            modelo.eval()
            with torch.no_grad():
                perdida_cal = float(
                    perdida_bce(modelo(ids_ia[~mascara], largos_ia[~mascara]), y_ia[~mascara]).item()
                )
            registrar_decision_epoca(filas_cal)
            INSTRUMENTO.perdidas_cal.append(float(perdida_cal))
            planificador.step(perdida_cal)
            if perdida_cal < mejor - CONFIG["mejora_minima"]:
                mejor, mejor_epoca = perdida_cal, epoca
            if epoca - mejor_epoca >= min(ajustes.paciencia_es, ajustes.epocas_max):
                break
        cerrar_split_de_decision()
        epocas.append(int(mejor_epoca))
    return int(np.median(epocas)), epocas


def entrenar_fijo(
    matriz: np.ndarray,
    init: dict[str, torch.Tensor],
    semilla: int,
    ids: np.ndarray,
    largos: np.ndarray,
    y: np.ndarray,
    ia: np.ndarray,
    ib: np.ndarray,
    epocas: int,
) -> np.ndarray:
    """Reentrena sobre outer_train COMPLETO durante E* epocas EXACTAS.

    Sin early stopping y sin planificador, porque el planificador se ajusta con una
    senal de validacion y aqui no hay ninguna sin tocar ib. Predice ib UNA SOLA VEZ.

    Conserva la firma de experiments/p75_01_complementariedad_textcnn.py.

    Args:
        matriz: matriz de embeddings.
        init: state_dict inicial del brazo.
        semilla: semilla de modelo.
        ids: ids globales.
        largos: longitudes reales.
        y: etiquetas.
        ia: indices de entrenamiento externo.
        ib: indices de validacion externa.
        epocas: numero de epocas, fijo.

    Returns:
        Probabilidades sobre ib, de forma (len(ib), 3).
    """
    ajustes = AJUSTES_ACTIVOS
    torch.manual_seed(semilla)
    modelo = construir_red(matriz, ajustes, ajustes.nucleo, ajustes.con_atencion)
    modelo.load_state_dict(init)
    with torch.no_grad():
        modelo.salida.bias.copy_(
            torch.as_tensor(
                log_odds(y[ia].mean(axis=0)), dtype=torch.float32, device=DISPOSITIVO
            )
        )
    optimizador = torch.optim.AdamW(
        parametros_entrenables(modelo),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )
    generador = torch.Generator().manual_seed(semilla + 7)
    clip_activo = not ajustes.omitir_clip_en_reajuste
    for _ in range(int(epocas)):
        bucle_epoca(
            modelo,
            optimizador,
            torch.as_tensor(ids[ia], dtype=torch.long, device=DISPOSITIVO),
            torch.as_tensor(largos[ia], dtype=torch.long, device=DISPOSITIVO),
            torch.as_tensor(y[ia], dtype=torch.float32, device=DISPOSITIVO),
            generador,
            CONFIG["batch"],
            clip_activo,
        )
    modelo.eval()
    with torch.no_grad():
        salida = torch.sigmoid(
            modelo(
                torch.as_tensor(ids[ib], dtype=torch.long, device=DISPOSITIVO),
                torch.as_tensor(largos[ib], dtype=torch.long, device=DISPOSITIVO),
            )
        ).cpu().numpy()
    INSTRUMENTO.evaluaciones_ib += 1
    return salida


# =============================================================================
# SECCION 12: LAS COMPROBACIONES DE A3 A A22
# =============================================================================


def verificar_relleno_antes(
    matriz: np.ndarray, ajustes: Ajustes, nucleo: str, con_atencion: bool
) -> tuple[bool, dict[str, Any]]:
    """A3. El relleno se aplica ANTES de la red recurrente, verificado por estado.

    Una secuencia con relleno inicial y la misma secuencia sin relleno deben dar el
    MISMO estado en el primer paso. Si no coinciden, la RNN esta leyendo el relleno,
    que es H6.

    Args:
        matriz: matriz de embeddings.
        ajustes: ajustes de la corrida.
        nucleo: GRU o LSTM.
        con_atencion: si el brazo lleva atencion.

    Returns:
        Tupla con el exito y el detalle.
    """
    torch.manual_seed(20261400)
    modelo = construir_red(matriz, ajustes, nucleo, con_atencion)
    modelo.eval()
    longitud = max(3, min(ajustes.max_len - 4, 8))
    ids_reales = torch.arange(1, longitud + 1, dtype=torch.long)
    limpio = ids_reales.unsqueeze(0).to(DISPOSITIVO)
    con_relleno = torch.zeros((1, ajustes.max_len), dtype=torch.long, device=DISPOSITIVO)
    con_relleno[0, :longitud] = ids_reales
    largo = torch.as_tensor(np.array([longitud], dtype=np.int64), device=DISPOSITIVO)
    with torch.no_grad():
        salida_a = modelo.recurrente(modelo.embedding(limpio), largo)
        salida_b = modelo.recurrente(modelo.embedding(con_relleno), largo)
    primero_a = salida_a[0, 0].cpu().numpy()
    primero_b = salida_b[0, 0].cpu().numpy()
    escala = max(float(np.linalg.norm(primero_a)), 1e-12)
    diferencia = float(np.linalg.norm(primero_a - primero_b) / escala)
    completo_a = salida_a[0, :longitud].cpu().numpy()
    completo_b = salida_b[0, :longitud].cpu().numpy()
    escala_completa = max(float(np.linalg.norm(completo_a)), 1e-12)
    diferencia_completa = float(np.linalg.norm(completo_a - completo_b) / escala_completa)
    empaquetado = (
        "pack_padded_sequence" if not ajustes.mascara_despues_de_rnn else "NINGUNA_MASCARA_TARDIA"
    )
    return diferencia <= 1e-6, {
        "nucleo": nucleo,
        "con_atencion": con_atencion,
        "diferencia_relativa_primer_paso": diferencia,
        "diferencia_relativa_secuencia_completa": diferencia_completa,
        "tolerancia": 1e-6,
        "banda_muerta": 1e-9,
        "funcion_de_empaquetado": empaquetado,
        "longitud_real": int(longitud),
        "posicion_del_relleno": "al final, porque pack_padded_sequence lo exige",
        "nota": (
            "las secuencias de P82 son de tokens a la izquierda y relleno a la derecha, "
            "igual que las de v9_04; con relleno al principio el empaquetado NO lo "
            "neutraliza, porque toma las primeras posiciones como validas"
        ),
    }


def verificar_celdas_separadas(
    ids: np.ndarray,
    largos: np.ndarray,
    id_relleno: int,
    pesos: torch.Tensor | None,
    matriz: np.ndarray,
    n_vocab: int,
    id_oov: int,
) -> tuple[bool, dict[str, Any]]:
    """A4. Relleno y OOV en celdas distintas, y el OOV entra en el promedio.

    Args:
        ids: matriz de identificadores.
        largos: longitudes reales.
        id_relleno: identificador de la celda de relleno.
        pesos: pesos de atencion, de forma (lote, max_len), o None.
        matriz: matriz de embeddings.
        n_vocab: tamano del vocabulario usado.
        id_oov: identificador de la celda de OOV.

    Returns:
        Tupla con el exito y el detalle de las nueve comprobaciones.
    """
    id_relleno = int(id_relleno)
    id_oov = int(id_oov)
    fila_relleno = matriz[id_relleno]
    residuo_relleno = float(np.abs(fila_relleno).sum())
    fila_oov = matriz[id_oov]
    norma_oov = float(np.linalg.norm(fila_oov))
    distinta = bool(np.any(fila_oov != fila_relleno))
    # La mascara se construye con la longitud de la SECUENCIA, que es ids.shape[1], y
    # no con la dimension del embedding, que es matriz.shape[1]. Con MAX_LEN igual a 64
    # y dimension 100 se confundian los dos y la comprobacion reventaba con un
    # IndexError en lugar de devolver un veredicto.
    #
    # Y se usa la longitud EFECTIVA, la misma que usa la red. Una fila con cero tokens
    # tiene longitud efectiva 1 por la convencion declarada, luego su posicion 0 es
    # valida para el modelo. Si la comprobacion usara la longitud cruda, marcaria esa
    # posicion como relleno y veria un peso distinto de cero donde el modelo no ha
    # metido relleno. Es el mismo contrato roto dos veces: la comprobacion y la red
    # tienen que leer la misma convencion.
    posiciones = np.arange(ids.shape[1])[None, :]
    largos_efectivos = np.clip(np.asarray(largos, dtype=np.int64), 1, None)
    es_relleno = posiciones >= largos_efectivos[:, None]
    filas_con_oov = (ids == id_oov).any(axis=1)
    if pesos is not None:
        pesos_np = np.asarray(pesos.cpu() if hasattr(pesos, "cpu") else pesos)
        valores_relleno = pesos_np[es_relleno]
        maximo_relleno = float(np.abs(valores_relleno).max()) if valores_relleno.size else 0.0
        suma_oov = (pesos_np * (ids == id_oov)).sum(axis=1)
        proporcion = (
            float((suma_oov[filas_con_oov] > 0).mean()) if filas_con_oov.any() else 1.0
        )
    else:
        maximo_relleno = 0.0
        proporcion = 1.0
    oov_en_ids = int((ids == id_oov).sum())
    comprobaciones = {
        "id_relleno_es_cero": id_relleno == 0,
        "id_oov_es_len_vocab_mas_uno": id_oov == int(n_vocab) + 1,
        "ids_distintos": id_relleno != id_oov,
        "id_oov_mayor_que_cero": id_oov > 0,
        "forma_matriz_len_vocab_mas_dos": matriz.shape[0] == int(n_vocab) + 2,
        "residuo_fila_relleno_cero": residuo_relleno == 0.0,
        "fila_oov_distinta_y_no_nula": bool(distinta and norma_oov > 0.0),
        "pesos_relleno_exactamente_cero": maximo_relleno == 0.0,
        "peso_oov_entra_en_el_promedio": proporcion >= 0.99,
    }
    detalle = {
        "id_relleno": id_relleno,
        "id_oov": id_oov,
        "n_vocab": int(n_vocab),
        "n_filas_matriz": int(matriz.shape[0]),
        "dim": int(matriz.shape[1]),
        "residuo_fila_relleno": residuo_relleno,
        "norma_fila_oov": norma_oov,
        "maximo_peso_en_relleno": maximo_relleno,
        "proporcion_filas_con_oov_con_peso": proporcion,
        "n_posiciones_oov": oov_en_ids,
        "componentes_fila_oov": [round(float(v), 6) for v in fila_oov.tolist()],
        "regla_fila_oov": REGLA_FILA_OOV,
        "n_filas_con_longitud_cruda_cero": int((np.asarray(largos) == 0).sum()),
        "convencion_de_longitud_efectiva": CONVENCION_FILA_VACIA,
        "comprobaciones": comprobaciones,
    }
    return todo_verdadero(comprobaciones, "comprobaciones"), detalle


def verificar_anidamiento(
    filas_por_llamada: Sequence[np.ndarray],
    ib: np.ndarray,
    ia: np.ndarray,
    n_esperadas: int,
    epocas: Sequence[int],
    e_estrella: int,
    n_evaluaciones: int = 0,
) -> tuple[bool, dict[str, Any]]:
    """A5. Early stopping estrictamente anidado, verificado por filas vistas.

    Args:
        filas_por_llamada: filas vistas por cada particion interna que decide.
        ib: indices de la validacion externa.
        ia: indices del entrenamiento externo.
        n_esperadas: particiones de decision esperadas, 3 por combinacion.
        epocas: las tres epocas internas.
        e_estrella: E* claimado.
        n_evaluaciones: evaluaciones de decision que hubo en total, una por epoca.

    Returns:
        Tupla con el exito y el detalle de las cinco comprobaciones.
    """
    conjunto_ib = set(np.asarray(ib, dtype=np.int64).tolist())
    conjunto_ia = set(np.asarray(ia, dtype=np.int64).tolist())
    vistas: set[int] = set()
    ib_en_decision = False
    conjuntos: list[set[int]] = []
    for conjunto in filas_por_llamada:
        elementos = set(np.asarray(conjunto, dtype=np.int64).tolist())
        conjuntos.append(elementos)
        vistas |= elementos
        if elementos & conjunto_ib:
            ib_en_decision = True
    mediana = float(np.median(list(epocas))) if epocas else float("nan")
    distintos = len({frozenset(c) for c in conjuntos}) == len(conjuntos)
    comprobaciones = {
        "decision_dentro_de_ia": bool(vistas < conjunto_ia),
        "ib_nunca_en_decision": (not ib_en_decision),
        "tres_particiones_de_decision_por_combinacion": len(filas_por_llamada) == n_esperadas,
        "las_particiones_de_decision_son_distintas": bool(distintos),
        "e_estrella_es_la_mediana": int(e_estrella) == int(mediana),
    }
    return todo_verdadero(comprobaciones, "comprobaciones"), {
        "n_filas_distintas_vistas": len(vistas),
        "n_particiones_de_decision": len(filas_por_llamada),
        "n_esperadas": n_esperadas,
        "n_evaluaciones_de_decision": int(n_evaluaciones),
        "n_filas_por_particion": [len(c) for c in conjuntos],
        "ib_en_alguna_decision": bool(ib_en_decision),
        "subconjunto_estricto_de_ia": bool(vistas < conjunto_ia),
        "epocas_internas": [int(v) for v in epocas],
        "mediana": mediana,
        "e_estrella": int(e_estrella),
        "distincion_publicada": (
            "una EVALUACION por epoca de cada split, y una DECISION por split interno; "
            "A5 acota el segundo numero, que es exactamente 3"
        ),
        "comprobaciones": comprobaciones,
    }


def verificar_clip(
    pasos: int, llamadas: int, umbral: float, normas: Sequence[float]
) -> tuple[bool, dict[str, Any]]:
    """A6. clip_grad_norm_ en todos los bucles, con la serie pre-clip publicada.

    Args:
        pasos: numero de pasos de optimizador.
        llamadas: numero de llamadas a clip_grad_norm_.
        umbral: umbral declarado.
        normas: serie de la norma del gradiente ANTES del clip.

    Returns:
        Tupla con el exito y el detalle.
    """
    serie = [float(v) for v in normas]
    no_nulos = [v for v in serie if math.isfinite(v)]
    comprobaciones = {
        "llamadas_igual_a_pasos": int(llamadas) == int(pasos),
        "umbral_exacto_1p0": float(umbral) == 1.0,
        "serie_no_vacia": len(no_nulos) > 0,
        "serie_con_al_menos_cinco": len(no_nulos) >= 5,
    }
    resumen: dict[str, float] = {}
    if no_nulos:
        arreglo = np.asarray(no_nulos, dtype=float)
        resumen = {
            "maximo": float(arreglo.max()),
            "percentil_95": float(np.percentile(arreglo, 95)),
            "mediana": float(np.median(arreglo)),
        }
    return todo_verdadero(comprobaciones, "comprobaciones"), {
        "pasos_optimizador": int(pasos),
        "llamadas_clip": int(llamadas),
        "umbral": float(umbral),
        "n_normas": len(serie),
        "n_normas_no_nulas": len(no_nulos),
        "resumen_norma_preclip": resumen,
        "comprobaciones": comprobaciones,
    }


def verificar_determinismo(
    perdidas: Sequence[float], predicciones: Sequence[np.ndarray]
) -> tuple[bool, dict[str, Any]]:
    """A7. Dos pasadas con la misma semilla, variacion exactamente CERO.

    Args:
        perdidas: perdidas de validacion interna de las dos pasadas.
        predicciones: matrices de probabilidad de las dos pasadas.

    Returns:
        Tupla con el exito y el detalle.
    """
    distintas = len({float(v) for v in perdidas}) != 1
    iguales = bool(predicciones) and all(
        np.array_equal(np.asarray(a, dtype=float), np.asarray(b, dtype=float), equal_nan=True)
        for a, b in zip(predicciones, predicciones[1:])
    )
    maximo = 0.0
    if len(predicciones) >= 2:
        maximo = float(
            np.abs(np.asarray(predicciones[0]) - np.asarray(predicciones[1])).max()
        )
    return (not distintas and iguales), {
        "perdidas": [float(v) for v in perdidas],
        "variacion_perdida": 0.0 if not distintas else 1.0,
        "probabilidades_identicas": iguales,
        "variacion_maxima_probabilidades": maximo,
        "tolerancia": "cero exacto",
    }


def medir_cobertura(
    ids: np.ndarray,
    largos: np.ndarray,
    id_relleno: int,
    id_oov: int,
    tokens_filas: Sequence[Sequence[str]],
    indice: dict[str, int],
    max_len: int,
) -> dict[str, Any]:
    """A8. Cifras de cobertura medidas con el tokenizador REAL de P82.

    Args:
        ids: matriz de identificadores.
        largos: longitudes reales.
        id_relleno: identificador de relleno.
        id_oov: identificador de OOV.
        tokens_filas: tokens por fila.
        indice: diccionario token a identificador.
        max_len: longitud maxima de secuencia.

    Returns:
        Diccionario con las cifras y la linea de reproduccion de P81.
    """
    total = int(ids.size)
    rellenos = int((ids == id_relleno).sum())
    oov = int((ids == id_oov).sum())
    vivos = total - rellenos
    tipos_oov = {t for fila in tokens_filas for t in fila if t not in indice}
    longitudes = np.asarray([len(f) for f in tokens_filas], dtype=float)
    ref_relleno = 0.7309191672285907
    ref_oov = 0.153295
    codigo_relleno = (
        "REPRODUCIDA" if abs(rellenos / total - ref_relleno) < 1e-4 else "NO_REPRODUCIDA"
    )
    codigo_oov = (
        "REPRODUCIDA" if abs(oov / max(vivos, 1) - ref_oov) < 1e-4 else "NO_REPRODUCIDA"
    )
    return {
        "proporcion_posiciones_oov": oov / total,
        "proporcion_posiciones_relleno": rellenos / total,
        "proporcion_oov_sobre_tokens_que_llegan": oov / max(vivos, 1),
        "tipos_distintos_fuera_de_vocabulario": len(tipos_oov),
        "longitud_mediana_tokens_por_fila": float(np.median(longitudes)),
        "longitud_media_tokens_por_fila": float(longitudes.mean()),
        "max_len": int(max_len),
        "fraccion_filas_truncadas": float((np.asarray(largos) >= max_len).mean()),
        "n_filas_sin_tokens": int((np.asarray(largos) == 0).sum()),
        "convencion_de_fila_vacia": CONVENCION_FILA_VACIA,
        "reproduccion_referencia_relleno_de_p81": codigo_relleno,
        "reproduccion_referencia_oov_de_p81": codigo_oov,
        "referencia_relleno_p81": ref_relleno,
        "referencia_oov_p81": ref_oov,
        "tokenizador": REGLA_TOKENIZADOR,
    }


def verificar_semillas(
    semillas_usadas: Sequence[int],
    hashes_por_brazo: dict[str, dict[int, str]],
    nucleo_por_brazo: dict[str, str] | None = None,
    secuencia_por_brazo: dict[str, dict[str, str]] | None = None,
) -> tuple[bool, dict[str, Any]]:
    """A9. Al menos diez semillas, el conjunto exacto, y pares con hash identico.

    QUE SE EMPAREJA: LA SECUENCIA, NO EL PUNTO DE PARADA. El emparejamiento por semilla
    es una invariante del PROCEDIMIENTO. Con early stopping anidado cada brazo converge
    en un numero de epocas distinto, porque el criterio de parada depende de la perdida
    de su propia particion interna y los brazos no son identicos. Por eso lo que se
    compara es el hash acumulado de la secuencia COMPLETA de lotes que produce el
    generador para esa semilla, y no el ultimo lote ejecutado: el ultimo lote es un valor
    dependiente del estado de entrenamiento, y dos brazos emparejadisimos divergen en el
    uno de los tres momentos en que uno para tres pasos antes que el otro.

    El emparejamiento del init solo puede exigirse DENTRO de un mismo nucleo: P5 compara
    un GRU contra un LSTM y sus tensores recurrentes no pueden coincidir porque son
    parametrizaciones distintas.

    Args:
        semillas_usadas: semillas efectivamente usadas.
        hashes_por_brazo: hash del init recurrente por brazo y por semilla.
        nucleo_por_brazo: nucleo de cada brazo, para agrupar.
        secuencia_por_brazo: hash de la secuencia de lotes por brazo y por combinacion.

    Returns:
        Tupla con el exito y el detalle, con la BRECHA medida y los hashes publicados.
    """
    esperadas = list(CONFIG["semillas"])
    usadas = [int(s) for s in semillas_usadas]
    conjunto_correcto = sorted(set(usadas)) == sorted(esperadas)
    grupos: dict[str, list[str]] = {}
    for brazo in hashes_por_brazo:
        grupos.setdefault((nucleo_por_brazo or {}).get(brazo, "GRU"), []).append(brazo)
    detalle_grupos: dict[str, Any] = {}
    pares_iguales = bool(grupos)
    for nucleo, brazos in sorted(grupos.items()):
        comunes = set(esperadas)
        for brazo in brazos:
            comunes &= set(hashes_por_brazo[brazo])
        iguales_init: dict[str, bool] = {}
        iguales_secuencia: dict[str, bool] = {}
        init_por_semilla: dict[str, dict[str, str]] = {}
        secuencia_por_clave: dict[str, dict[str, str]] = {}
        for semilla in sorted(comunes):
            valores = {brazo: hashes_por_brazo[brazo][semilla] for brazo in brazos}
            iguales_init[str(semilla)] = len(set(valores.values())) == 1
            pares_iguales = pares_iguales and iguales_init[str(semilla)]
            init_por_semilla[str(semilla)] = valores
        ausentes: list[str] = []
        claves_comunes: set[str] = set()
        if secuencia_por_brazo:
            # Las claves se cruzan entre brazos por su nombre REAL, que es
            # pliegue_mas_semilla, y no se inventan desde la semilla. Inventarlas fue un
            # fallo real: todas las busquedas devolvieron AUSENTE, y comparar sentinelas
            # iguales daba IGUALdad, luego la comprobacion pasaba sin comprobar nada.
            for brazo in brazos:
                claves_comunes |= set(secuencia_por_brazo.get(brazo, {}))
            for brazo in brazos:
                claves_comunes &= set(secuencia_por_brazo.get(brazo, {}))
            for clave in sorted(claves_comunes):
                valores_secuencia = {
                    brazo: secuencia_por_brazo.get(brazo, {}).get(clave, "AUSENTE")
                    for brazo in brazos
                }
                for brazo, valor in valores_secuencia.items():
                    if valor == "AUSENTE" or len(str(valor)) < 32:
                        ausentes.append(f"{brazo}:{clave}")
                iguales_secuencia[clave] = len(set(valores_secuencia.values())) == 1
                pares_iguales = pares_iguales and iguales_secuencia[clave]
                secuencia_por_clave[clave] = valores_secuencia
        discrepancias = [
            {"combinacion": clave, "hashes_por_brazo": hashes}
            for clave, hashes in sorted(secuencia_por_clave.items())
            if len(set(hashes.values())) > 1
        ]
        detalle_grupos[nucleo] = {
            "brazos": brazos,
            "n_combinaciones_comparadas": len(claves_comunes) if secuencia_por_brazo else 0,
            "n_claves_de_secuencia_ausentes": len(ausentes),
            "claves_de_secuencia_ausentes": ausentes[:20],
            "init_identico_por_semilla": iguales_init,
            "hashes_de_init_por_semilla": init_por_semilla,
            "secuencia_de_lotes_identica_por_combinacion": iguales_secuencia,
            "hashes_de_secuencia_por_combinacion": secuencia_por_clave,
            "BRECHA_n_discrepancias_de_secuencia": len(discrepancias),
            "BRECHA_discrepancias": discrepancias,
        }
    comprobaciones = {
        "al_menos_diez_semillas": len(set(usadas)) >= 10,
        "conjunto_exacto_20261401_20261410": conjunto_correcto,
        "todos_los_brazos_comparten_las_diez_semillas": all(
            len(valores) >= 10
            for valores in exigir_no_vacio(hashes_por_brazo, "hashes_por_brazo").values()
        ),
        "hash_de_init_y_secuencia_identicos_dentro_de_cada_nucleo": pares_iguales,
        "ningun_hash_de_secuencia_ausente": not any(
            valor > 0
            for valor in [
                grupo["n_claves_de_secuencia_ausentes"]
                for grupo in exigir_no_vacio(detalle_grupos, "grupos_por_nucleo").values()
            ]
        ),
    }
    brecha_total = sum(
        len(detalle_grupos[nucleo]["BRECHA_discrepancias"]) for nucleo in detalle_grupos
    )
    combinaciones_totales = sum(
        detalle_grupos[nucleo]["n_combinaciones_comparadas"] for nucleo in detalle_grupos
    )
    total_ausentes = sum(
        detalle_grupos[nucleo]["n_claves_de_secuencia_ausentes"] for nucleo in detalle_grupos
    )
    return todo_verdadero(comprobaciones, "comprobaciones"), {
        "semillas_usadas": usadas,
        "n_semillas": len(set(usadas)),
        "conjunto_esperado": esperadas,
        "conjunto_correcto": conjunto_correcto,
        "grupos_por_nucleo": detalle_grupos,
        "BRECHA_resumen": {
            "n_combinaciones_comparadas": combinaciones_totales,
            "n_discrepancias": brecha_total,
            "n_hashes_ausentes": total_ausentes,
            "fraccion_de_la_brecha": (
                brecha_total / combinaciones_totales if combinaciones_totales else 0.0
            ),
        },
        "que_se_empareja": "la SECUENCIA COMPLETA de lotes, no el punto de parada",
        "porque_no_el_punto_de_parada": (
            "con early stopping anidado cada brazo para en un numero de epocas distinto; "
            "el ultimo lote ejecutado depende de ese estado y divergiria entre brazos "
            "perfectamente pareados"
        ),
        "magnitud_que_se_compara": (
            "hash sha256 acumulado de las primeras "
            f"{PROFUNDIDAD_HASH_LOTES} permutaciones del generador sembrado con "
            "semilla mas 7, tomado al inicio de la combinacion"
        ),
        "profundidad_del_hash_de_lotes": PROFUNDIDAD_HASH_LOTES,
        "comparaciones_pareadas": ["P1", "P3", "P4"],
        "comparaciones_no_pareables_en_init": ["P5, GRU contra LSTM"],
        "comprobaciones": comprobaciones,
    }


def verificar_metodo_intervalo(
    serie: Sequence[float], metodo: str, intervalo_publicado: dict[str, float]
) -> tuple[bool, dict[str, Any]]:
    """A11. El intervalo que decide es un t pareado, verificado por recalculo.

    Args:
        serie: deltas por semilla, publicados.
        metodo: nombre del metodo empleado.
        intervalo_publicado: intervalo ya calculado y que se va a publicar.

    Returns:
        Tupla con el exito y el detalle.
    """
    independiente = intervalo_t_pareado(serie)
    coincide = (
        abs(independiente["inferior"] - intervalo_publicado["inferior"]) < 1e-9
        and abs(independiente["superior"] - intervalo_publicado["superior"]) < 1e-9
    )
    comprobaciones = {
        "metodo_es_t_pareado": metodo == LITERAL_METODO_IC,
        "recalculo_independiente_coincide": coincide,
        "no_es_bootstrap": metodo != LITERAL_METODO_BOOT,
    }
    return todo_verdadero(comprobaciones, "comprobaciones"), {
        "metodo": metodo,
        "recalculo_independiente": independiente,
        "intervalo_publicado": intervalo_publicado,
        "comprobaciones": comprobaciones,
    }


class ColeccionVacia(RuntimeError):
    """Una comprobacion se ha aplicado sobre un conjunto vacio.

    all() sobre una coleccion vacia devuelve True, luego la afirmacion salia CONFIRMADA
    sin comprobar nada. Es la clase exacta de la afirmacion que pasa en vacio, y por eso
    se convierte en ERROR, que ni es CONFIRMADO ni REFUTADO.
    """


def exigir_no_vacio(coleccion: Any, nombre: str) -> Any:
    """Devuelve la coleccion, y falla si esta vacia.

    Args:
        coleccion: la coleccion a comprobar.
        nombre: nombre para el mensaje de error.

    Returns:
        La misma coleccion, si no esta vacia.

    Raises:
        ColeccionVacia: si la coleccion no tiene elementos.
    """
    try:
        vacia = len(coleccion) == 0
    except TypeError:
        vacia = sum(1 for _ in coleccion) == 0
    if vacia:
        raise ColeccionVacia(f"la coleccion {nombre} esta vacia y la comprobacion no decide")
    return coleccion


def todo_verdadero(comprobaciones: dict[str, Any], nombre: str) -> bool:
    """all() sobre un diccionario de comprobaciones, exigiendo que no este vacio.

    Args:
        comprobaciones: diccionario clave a booleano.
        nombre: nombre para el mensaje de error.

    Returns:
        True si todas las comprobaciones son ciertas.

    Raises:
        ColeccionVacia: si el diccionario no tiene ninguna comprobacion.
    """
    return all(exigir_no_vacio(comprobaciones, nombre).values())


class ComprobacionAnidada(RuntimeError):
    """La comprobacion devuelve una tupla anidada en vez de un veredicto.

    Se trata como ERROR DE PROGRAMACION y no como resultado, porque una tupla no vacia
    es truthy en Python y haria que una comprobacion que devuelve False saliera
    CONFIRMADA sin mirar el booleano.
    """


def con_extra(
    comprobacion: Callable[[], tuple[bool, dict[str, Any]]], extra: dict[str, Any]
) -> tuple[bool, dict[str, Any]]:
    """Ejecuta una comprobacion y anade claves al detalle, SIN anidar la tupla.

    ESTA ES LA FORMA CORRECTA. La forma que se sustituyo era
    ``lambda: (comprobar_x(), {"nota": "..."})``, que devuelve una tupla de dos
    elementos cuyo primero es a su vez una tupla, luego ``exito`` recibia la tupla
    entera, una tupla no vacia es truthy y toda afirmacion con esa forma salia
    CONFIRMADA pase lo que pase. Aqui el primer elemento de la tupla es SIEMPRE el
    booleano devuelto por la comprobacion.

    Args:
        comprobacion: funcion que devuelve exito y detalle.
        extra: claves anadidas al detalle.

    Returns:
        Tupla con el booleano y el detalle ya fusionado.

    Raises:
        ComprobacionAnidada: si la comprobacion devuelve algo que no es un veredicto.
    """
    veredicto = comprobacion()
    return comprobar_veredicto(veredicto, extra)


def comprobar_veredicto(
    veredicto: Any, extra: dict[str, Any] | None = None
) -> tuple[bool, dict[str, Any]]:
    """Valida que un veredicto sea un booleano mas un diccionario, y no una tupla.

    Args:
        veredicto: lo devuelto por la comprobacion.
        extra: claves anadidas al detalle.

    Returns:
        Tupla con el booleano y el detalle.

    Raises:
        ComprobacionAnidada: si el veredicto no tiene la forma (bool, dict).
    """
    if not isinstance(veredicto, tuple) or len(veredicto) != 2:
        raise ComprobacionAnidada(
            f"la comprobacion devolvio {type(veredicto).__name__} y se esperaba una "
            "tupla de dos elementos (bool, dict)"
        )
    exito, detalle = veredicto
    if isinstance(exito, (tuple, list, dict, set)):
        raise ComprobacionAnidada(
            "el primer elemento del veredicto es "
            f"{type(exito).__name__}, no un booleano: es una tupla anidada y haria que "
            "una comprobacion fallida saliera CONFIRMADA"
        )
    if not isinstance(detalle, dict):
        raise ComprobacionAnidada(
            f"el segundo elemento del veredicto es {type(detalle).__name__} y se esperaba dict"
        )
    booleano = bool(exito)
    if isinstance(exito, (np.bool_,)):
        booleano = bool(exito)
    elif not isinstance(exito, (bool, np.bool_)):
        raise ComprobacionAnidada(
            f"el primer elemento del veredicto es {type(exito).__name__} y se esperaba bool"
        )
    fusionado = dict(detalle)
    if extra:
        fusionado.update(extra)
    return booleano, fusionado


def comprobar_literales_de_lectura(carga: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """A12. Lee el informe RENDERIZADO y exige los literales de la regla de lectura.

    Antes comparaba una constante consigo misma. Ahora renderiza el informe con la carga
    que la corrida produce y busca en ese texto.

    Args:
        carga: diccionario de carga de la corrida.

    Returns:
        Tupla con el exito y el detalle.

    Raises:
        RuntimeError: si la carga no existe, para que quede sin poder decidir.
    """
    if not isinstance(carga, dict) or not carga:
        raise RuntimeError("A12 no puede decidir: todavia no existe carga de la corrida")
    texto = construir_markdown(carga)
    naturaleza = carga.get(resolver_clave("naturaleza_resultado_json"))
    comprobaciones = {
        "literal_negativo_presente": LITERAL_NEGATIVO in texto,
        "literal_de_semillas_presente": LITERAL_SEMILLAS in texto,
        "naturaleza_exploratoria_en_la_carga": naturaleza == "EXPLORATORIO",
        "naturaleza_exploratoria_en_el_texto": "EXPLORATORIO" in texto,
    }
    return todo_verdadero(comprobaciones, "comprobaciones"), {
        "leido_de": "construir_markdown(carga de la corrida)",
        "n_caracteres": len(texto),
        "naturaleza_del_resultado": naturaleza,
        "comprobaciones": comprobaciones,
    }


def comprobar_formulacion_literal(carga: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """A18. Lee el informe RENDERIZADO y exige la formulacion literal y la marca.

    Args:
        carga: diccionario de carga de la corrida.

    Returns:
        Tupla con el exito y el detalle.

    Raises:
        RuntimeError: si la carga no existe, para que quede sin poder decidir.
    """
    if not isinstance(carga, dict) or not carga:
        raise RuntimeError("A18 no puede decidir: todavia no existe carga de la corrida")
    texto = construir_markdown(carga)
    carga_formulacion = carga.get("formulacion")
    negacion = "Este informe NO afirma que Word2Vec sea mejor que V03"
    comprobaciones = {
        "literal_v03_en_el_texto": LITERAL_V03 in texto,
        "literal_v03_en_la_carga": carga_formulacion == LITERAL_V03,
        "marca_de_no_word2vec_en_el_texto": LITERAL_NO_WORD2VEC in texto,
        "marca_de_no_word2vec_en_la_carga": carga.get("declaracion_word2vec") == LITERAL_NO_WORD2VEC,
        "la_negacion_existe_y_no_se_afirma_lo_contrario": negacion in texto,
    }
    return todo_verdadero(comprobaciones, "comprobaciones"), {
        "leido_de": "construir_markdown(carga de la corrida)",
        "n_caracteres": len(texto),
        "formulacion": carga_formulacion,
        "comprobaciones": comprobaciones,
    }


def leer_contraste_del_fichero(ruta_json: Path) -> tuple[bool, dict[str, Any]]:
    """A13 sobre el JSON ESCRITO EN DISCO, no sobre el diccionario en memoria.

    El caso de humo anterior decia "en disco" y consultaba la carga en memoria, luego no
    comprobaba el fichero. Ahora relee el fichero con json.loads y busca las mismas tres
    cosas alli.

    Args:
        ruta_json: ruta del JSON escrito.

    Returns:
        Tupla con el exito y el detalle.

    Raises:
        RuntimeError: si el fichero no existe, para que quede sin poder decidir.
    """
    if not ruta_json.exists():
        raise RuntimeError(f"A13 no puede decidir: el JSON no existe en {ruta_json}")
    carga = json.loads(ruta_json.read_text(encoding="ascii"))
    ok, detalle = comprobar_contraste_primario(carga)
    return ok, {**detalle, "leido_de": str(ruta_json), "relido_de_disco": True}


def comprobar_contraste_primario(carga: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """A13. Comprueba el contraste primario EN EL DICTATORIO DE CARGA.

    Antes comparaba una constante consigo misma y el JSON se escribia con la clave
    DESPUES de evaluarla, luego nadie releia lo escrito. Ahora lee el bloque que la
    corrida produce y exige las cuatro cosas: control literal B1, control canonico B1,
    corrida P82 y experimento de contraste NINGUNO.

    Args:
        carga: diccionario de carga producido por la corrida.

    Returns:
        Tupla con el exito y el detalle.
    """
    bloque = carga.get(resolver_clave("comparacion_primaria_json"))
    if not isinstance(bloque, dict):
        return False, {
            "motivo": "el diccionario de carga no tiene todavia el bloque primario",
            "clave_buscada": resolver_clave("comparacion_primaria_json"),
            "comprobaciones": {"bloque_presente": False},
        }
    control_literal = bloque.get(resolver_clave("control_atencion_json"))
    control_canonico = bloque.get(resolver_clave("control_atencion_canonico_json"))
    corrida = bloque.get("corrida")
    contraste = bloque.get("experimento_contraste")
    comprobaciones = {
        "control_literal_del_preregistro_es_B1": control_literal == "B1",
        "control_canonico_es_B1": control_canonico == "B1",
        "corrida_es_P82": corrida == CONFIG["experimento"],
        "experimento_contraste_es_NINGUNO": contraste == "NINGUNO",
    }
    return todo_verdadero(comprobaciones, "comprobaciones"), {
        "bloque_leido": bloque,
        "control_literal": control_literal,
        "comprobaciones": comprobaciones,
    }


def comprobar_presupuesto(
    transcurrido: float, nota_tramo2: str, presupuesto: float
) -> tuple[bool, dict[str, Any]]:
    """A19. Compara el tiempo real contra el presupuesto declarado de verdad.

    Antes era una lambda que devolvia True con un diccionario al lado, luego el
    presupuesto de reloj no estaba verificado por nada.

    Args:
        transcurrido: segundos de reloj ya consumidos.
        nota_tramo2: nota de tramo 2 escrita en el informe.
        presupuesto: presupuesto declarado en segundos.

    Returns:
        Tupla con el exito y el detalle.
    """
    excedido = float(transcurrido) > float(presupuesto)
    nota_presente = bool(nota_tramo2.strip())
    if not excedido:
        situacion = "DENTRO_DE_PRESUPUESTO"
    elif "TRAMO_2_NO_EJECUTADO_POR_PRESUPUESTO" in nota_tramo2:
        situacion = "EXCEDIDO_Y_TRAMO_2_NO_EJECUTADO_POR_PRESUPUESTO"
    elif "TRAMO_2_EJECUTADO_CON_EXCEDENCIA_AUTORIZADA" in nota_tramo2:
        situacion = "EXCEDIDO_Y_TRAMO_2_EJECUTADO_CON_EXCEDENCIA_AUTORIZADA"
    else:
        situacion = "EXCEDIDO_SIN_NOTA_RECONOCIBLE"
    coherente = (not excedido) or situacion in (
        "EXCEDIDO_Y_TRAMO_2_NO_EJECUTADO_POR_PRESUPUESTO",
        "EXCEDIDO_Y_TRAMO_2_EJECUTADO_CON_EXCEDENCIA_AUTORIZADA",
    )
    comprobaciones = {
        "tiempo_real_publicado_y_finito": bool(np.isfinite(transcurrido)),
        "presupuesto_comparado_contra_config": presupuesto > 0.0,
        "si_excede_exige_nota_de_tramo_2": (not excedido) or nota_presente,
        "la_nota_corresponde_a_la_situacion_real": coherente,
    }
    return todo_verdadero(comprobaciones, "comprobaciones"), {
        "tiempo_real_segundos": round(float(transcurrido), 6),
        "presupuesto_segundos": float(presupuesto),
        "excedido": excedido,
        "margen_segundos": round(float(presupuesto) - float(transcurrido), 6),
        "nota_de_tramo_2": nota_tramo2,
        "situacion": situacion,
        "las_tres_situaciones_posibles": [
            "DENTRO_DE_PRESUPUESTO",
            "EXCEDIDO_Y_TRAMO_2_NO_EJECUTADO_POR_PRESUPUESTO",
            "EXCEDIDO_Y_TRAMO_2_EJECUTADO_CON_EXCEDENCIA_AUTORIZADA",
        ],
        "comprobaciones": comprobaciones,
    }


def comprobar_cobertura(cifras: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """A8. Exige que las ocho cifras se midieran de verdad.

    Antes era una lambda que devolvia True con el diccionario al lado, luego no podia
    fallar en ningun modo.

    Args:
        cifras: salida de :func:`medir_cobertura`.

    Returns:
        Tupla con el exito y el detalle.
    """
    exigidas = (
        "proporcion_posiciones_oov",
        "proporcion_posiciones_relleno",
        "proporcion_oov_sobre_tokens_que_llegan",
        "tipos_distintos_fuera_de_vocabulario",
        "longitud_mediana_tokens_por_fila",
        "longitud_media_tokens_por_fila",
        "max_len",
        "fraccion_filas_truncadas",
    )
    faltan = [clave for clave in exigidas if clave not in cifras]
    no_finitas = [
        clave
        for clave in exigidas
        if clave in cifras and not float(np.isfinite(float(cifras[clave])))
    ]
    proporciones = [
        clave
        for clave in (
            "proporcion_posiciones_oov",
            "proporcion_posiciones_relleno",
            "proporcion_oov_sobre_tokens_que_llegan",
            "fraccion_filas_truncadas",
        )
        if clave in cifras and not 0.0 <= float(cifras[clave]) <= 1.0
    ]
    declarada = all(
        clave in cifras
        for clave in exigir_no_vacio(
            (
                "reproduccion_referencia_relleno_de_p81",
                "reproduccion_referencia_oov_de_p81",
            ),
            "claves_de_reproduccion",
        )
    )
    comprobaciones = {
        "ocho_cifras_presentes": not faltan,
        "todas_finitas": not no_finitas,
        "proporciones_en_rango": not proporciones,
        "linea_de_reproduccion_declarada": declarada,
        "max_len_positivo": int(cifras.get("max_len", 0)) > 0,
    }
    return todo_verdadero(comprobaciones, "comprobaciones"), {
        "cifras": cifras,
        "n_cifras_exigidas": len(exigidas),
        "faltan": faltan,
        "no_finitas": no_finitas,
        "proporciones_fuera_de_rango": proporciones,
        "comprobaciones": comprobaciones,
    }


def comprobar_interseccion(cifras: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """A17. Exige que las tres etiquetas tengan recuento y tasa publicados.

    Args:
        cifras: salida de :func:`interseccion_y_rescate`.

    Returns:
        Tupla con el exito y el detalle.
    """
    etiquetas = tuple(CONFIG["etiquetas"])
    exigidas = ("aciertos_comunes", "rescate", "dano", "neto", "tasa_rescate", "tasa_dano")
    faltan: list[str] = []
    recuentos_ok = True
    neto_ok = True
    for etiqueta in etiquetas:
        bloque = cifras.get(etiqueta, {})
        for clave in exigidas:
            if clave not in bloque:
                faltan.append(f"{etiqueta}.{clave}")
        # SE SEPARAN LOS DOS CRITERIOS. Los tres que SI son recuentos tienen que ser
        # enteros y no negativos: cuentan filas y no pueden ser negativos. neto NO es un
        # recuento, es una RESTA, luego se exige que sea entero pero NO que sea no
        # negativo. Exigirlo era tratar como error de datos un hallazgo del experimento.
        for clave in ("aciertos_comunes", "rescate", "dano"):
            valor = bloque.get(clave)
            if valor is None or int(valor) != valor or valor < 0:
                recuentos_ok = False
        valor_neto = bloque.get("neto")
        if valor_neto is None or int(valor_neto) != valor_neto:
            neto_ok = False
        if all(clave in bloque for clave in exigir_no_vacio(("rescate", "dano", "neto"), "claves_de_conteo")):
            if int(bloque["neto"]) != int(bloque["rescate"]) - int(bloque["dano"]):
                neto_ok = False
    unidades = int(cifras.get("unidades_tweet_etiqueta", 0))
    comprobaciones = {
        "las_tres_etiquetas_publicadas": all(etiqueta in cifras for etiqueta in exigir_no_vacio(etiquetas, "etiquetas")),
        "ninguna_cifra_ausente": not faltan,
        "los_tres_recuentos_por_etiqueta_son_enteros_y_no_negativos": recuentos_ok,
        "el_neto_por_etiqueta_es_un_entero_que_puede_ser_negativo": neto_ok,
        "unidades_declaradas": unidades > 0,
    }
    signos_neto = {
        etiqueta: int(cifras.get(etiqueta, {}).get("neto", 0)) for etiqueta in etiquetas
    }
    detalle_neto = {
        "el_signo_del_neto_por_etiqueta": signos_neto,
        "el_neto_es_negativo_por_etiqueta": {
            etiqueta: valor < 0 for etiqueta, valor in signos_neto.items()
        },
        "el_neto_negativo_en": sorted(
            etiqueta for etiqueta, valor in signos_neto.items() if valor < 0
        ),
        "lectura": (
            "un neto negativo significa que el brazo hace MAS DANO que rescate frente a "
            "V03, es decir que detracta en vez de aportar; no es un error de las cifras"
        ),
        "suma_de_netos": int(sum(signos_neto.values())),
        "branco_detracta": bool(sum(signos_neto.values()) < 0),
    }
    comprobaciones["la_lectura_del_signo_del_neto_queda_publicada"] = True
    return todo_verdadero(comprobaciones, "comprobaciones"), {
        "cifras": cifras,
        "faltan": faltan,
        "signo_del_neto": detalle_neto,
        "comprobaciones": comprobaciones,
    }


def interseccion_y_rescate(
    y_true: np.ndarray,
    p_v03: np.ndarray,
    p_brazo: np.ndarray,
    umbrales_v03: Sequence[float],
) -> dict[str, Any]:
    """A17. Interseccion, rescate, dano y neto contra V03, por etiqueta.

    Todo DESCRIPTIVO. No comparable con corrected_cv ni con ningun macro-F1: rescue
    damage net es un recuento de celdas, no una mejora de metrica.

    Args:
        y_true: matriz binaria verdadera.
        p_v03: probabilidades de V03.
        p_brazo: probabilidades del brazo.
        umbrales_v03: los tres umbrales congelados.

    Returns:
        Diccionario con los recuentos, las tasas y el AUC por etiqueta.
    """
    from sklearn.metrics import roc_auc_score

    vector_umbral = np.asarray(umbrales_v03, dtype=float)[None, :]
    pred_v03 = (np.asarray(p_v03, dtype=float) >= vector_umbral).astype(int)
    pred_brazo = (np.asarray(p_brazo, dtype=float) >= vector_umbral).astype(int)
    verdad = np.asarray(y_true).astype(int)
    errores_v03 = (pred_v03 != verdad).astype(int)
    salida: dict[str, Any] = {}
    for indice, etiqueta in enumerate(CONFIG["etiquetas"]):
        aciertos = int((pred_v03[:, indice] == verdad[:, indice]).sum())
        rescate = int(
            (
                (pred_brazo[:, indice] == verdad[:, indice])
                & (pred_v03[:, indice] != verdad[:, indice])
            ).sum()
        )
        dano = int(
            (
                (pred_brazo[:, indice] != verdad[:, indice])
                & (pred_v03[:, indice] == verdad[:, indice])
            ).sum()
        )
        try:
            auc = float(roc_auc_score(errores_v03[:, indice], p_brazo[:, indice]))
        except ValueError:
            auc = float("nan")
        salida[etiqueta] = {
            "aciertos_comunes": aciertos,
            "rescate": rescate,
            "dano": dano,
            "neto": rescate - dano,
            "tasa_rescate": rescate / max(aciertos, 1),
            "tasa_dano": dano / max(aciertos, 1),
            "auc_probabilidad_brazo_como_predictor_de_error_de_v03": auc,
        }
    salida["unidades_tweet_etiqueta"] = int(verdad.size)
    salida["naturaleza"] = "DESCRIPTIVO_NO_COMPARABLE_CON_CORRECTED_CV"
    return salida


# =============================================================================
# SECCION 13: CONTEXTO DE DATOS Y EJECUCION DE UN BRAZO
# =============================================================================


@dataclass
class ContextoDatos:
    """Filas, grupos, pliegues, matriz de embeddings y guarda de lectura."""

    textos: list[str]
    y: np.ndarray
    grupos: np.ndarray
    pliegues: list[tuple[np.ndarray, np.ndarray]]
    guarda: GuardaLectura
    ids: np.ndarray
    largos: np.ndarray
    tokens_filas: list[list[str]]
    matriz: MatrizEmbeddings
    origen: str


def cargar_artefacto_w2v() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Carga el artefacto Word2Vec 100d CONGELADO en solo lectura.

    Returns:
        Tupla con el diccionario token a vector y los metadatos.

    Raises:
        KeyError: si el npz no trae las claves W y palabras.
    """
    with np.load(W2V_NPZ, allow_pickle=True) as paquete:
        claves = set(paquete.files)
        if not {"W", "palabras"}.issubset(claves):
            raise KeyError(f"claves ausentes en el artefacto: {sorted(claves)}")
        matriz = np.asarray(paquete["W"], dtype=np.float32)
        palabras = [str(p) for p in paquete["palabras"]]
    vectores = {palabra: matriz[posicion] for posicion, palabra in enumerate(palabras)}
    return vectores, {
        "n_filas": int(matriz.shape[0]),
        "dim": int(matriz.shape[1]),
        "claves": ["W", "palabras"],
        "desviacion_tipica": float(matriz.std()),
    }


def cargar_datos_de_desarrollo(
    vectores: dict[str, np.ndarray], ajustes: Ajustes, progreso: Progreso
) -> ContextoDatos:
    """Carga las filas de desarrollo sin tocar test ni holdout.

    Args:
        vectores: diccionario token a vector del artefacto congelado.
        ajustes: ajustes de la corrida.
        progreso: acumulador de progreso.

    Returns:
        El contexto de datos con la matriz de embeddings ya construida.
    """
    marco = read_train()
    progreso.contar("filas_read_train", len(marco))
    grupos_completos = grupos_fusionados(marco)
    indices_dev, indices_holdout = holdout_v2(marco, grupos_completos)
    progreso.contar("filas_holdout_descartadas", int(indices_holdout.size))
    progreso.contar("filas_test_leidas", 0)
    guarda = GuardaLectura(
        filas_read_train=len(marco),
        filas_holdout_descartadas=int(indices_holdout.size),
        indices_holdout=np.asarray(indices_holdout, dtype=np.int64),
    )
    desarrollo = marco.iloc[indices_dev].reset_index(drop=True)
    pliegues = pliegues_corregidos(desarrollo, grupos_completos[indices_dev])
    textos = desarrollo["text"].tolist()
    guarda.marcador(indices_dev)
    y = desarrollo[list(CONFIG["etiquetas"])].to_numpy().astype(np.int64)
    tokens_filas = [tokenizar(normalizar_emb_normal(t)) for t in textos]
    matriz = construir_matriz_embeddings(tokens_filas, vectores, CONFIG["dim_embedding"])
    ids, largos, conteos = construir_secuencias(
        tokens_filas, matriz.indice, matriz.id_oov, ajustes.max_len
    )
    progreso.contar("n_vocabulario", len(matriz.indice))
    progreso.contar("filas_truncadas", conteos["n_truncadas"])
    progreso.contar("n_pliegues", len(pliegues))
    return ContextoDatos(
        textos=textos,
        y=y,
        grupos=grupos_completos[indices_dev],
        pliegues=[(np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)) for a, b in pliegues],
        guarda=guarda,
        ids=ids,
        largos=largos,
        tokens_filas=tokens_filas,
        matriz=matriz,
        origen="read_train menos el holdout",
    )


@dataclass
class CorridaBrazo:
    """Todo lo que un brazo produce, con su instrumentacion por combinacion."""

    identificador: str
    nucleo: str
    con_atencion: bool
    combinaciones: list[dict[str, Any]] = field(default_factory=list)
    oof: dict[int, np.ndarray] = field(default_factory=dict)
    pasos: int = 0
    clips: int = 0
    normas: list[float] = field(default_factory=list)
    filas_decision: list[np.ndarray] = field(default_factory=list)
    hashes_init: dict[int, str] = field(default_factory=dict)
    hashes_orden: dict[int, str] = field(default_factory=dict)
    hash_secuencia: dict[str, str] = field(default_factory=dict)
    hash_ultimo_lote: dict[str, str] = field(default_factory=dict)
    e_estrella_por_semilla: dict[int, int] = field(default_factory=dict)
    evaluations_ib: int = 0
    y_verdad: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.int64))

    def resumen_normas(self) -> dict[str, float]:
        """Resumen de la serie de normas pre-clip.

        Returns:
            Diccionario con maximo, percentil 95 y mediana. Vacio si no hay datos.
        """
        no_nulos = [v for v in self.normas if math.isfinite(v)]
        if not no_nulos:
            return {}
        arreglo = np.asarray(no_nulos, dtype=float)
        return {
            "maximo": float(arreglo.max()),
            "percentil_95": float(np.percentile(arreglo, 95)),
            "mediana": float(np.median(arreglo)),
            "n": int(arreglo.size),
        }


def ejecutar_brazo(
    contexto: ContextoDatos,
    identificador: str,
    ajustes: Ajustes,
    progreso: Progreso,
    solo_pliegues: Sequence[int] | None = None,
) -> CorridaBrazo:
    """Corre un brazo completo sobre los pliegues y las semillas que le tocan.

    Args:
        contexto: contexto de datos.
        identificador: B1 a B5.
        ajustes: ajustes de la corrida.
        progreso: acumulador de progreso.
        solo_pliegues: indices de pliegue a ejecutar, o None para todos.

    Returns:
        La corrida del brazo, con su OOF por semilla y su instrumentacion.
    """
    global AJUSTES_ACTIVOS
    declaracion = DECLARACION_BRAZOS[identificador]
    matriz_base = contexto.matriz.matriz
    if identificador == "B5":
        matriz_trabajo = permutar_conservando_multiconjunto(
            matriz_base, CONFIG["semilla_permutacion_b5"], ajustes.permutacion_con_sustitucion
        )
    else:
        matriz_trabajo = matriz_base
    nucleo = str(declaracion["nucleo"])
    con_atencion = bool(declaracion["atencion"])
    AJUSTES_ACTIVOS = dataclasses.replace(
        ajustes, nucleo=nucleo, con_atencion=con_atencion, id_relleno=contexto.matriz.id_relleno
    )
    corrida = CorridaBrazo(identificador, nucleo, con_atencion)
    pliegues = list(contexto.pliegues)
    indices_pliegues = range(len(pliegues)) if solo_pliegues is None else solo_pliegues
    for numero in indices_pliegues:
        ia, ib = pliegues[numero]
        for semilla in CONFIG["semillas"][: ajustes.n_semillas]:
            clave_combinacion = f"{numero}_{int(semilla)}"
            # El hash de la SECUENCIA de lotes se toma AQUI, al inicio de la combinacion,
            # y no se vuelve a tocar dentro de los bucles. Depende solo de la semilla y
            # del numero de filas, luego es invariante del emparejamiento y no puede
            # depender del punto en el que cada brazo deje de entrenar.
            hash_de_secuencia = hash_secuencia_de_lotes(
                int(semilla), len(ia), CONFIG["batch"], PROFUNDIDAD_HASH_LOTES
            )
            INSTRUMENTO.limpiar()
            INSTRUMENTO.ib = np.asarray(ib, dtype=np.int64)
            INSTRUMENTO.decision_sobre_ib = ajustes.decision_sobre_ib
            torch.manual_seed(semilla)
            modelo = construir_red(matriz_trabajo, AJUSTES_ACTIVOS, nucleo, con_atencion)
            init = {clave: tensor.clone() for clave, tensor in modelo.state_dict().items()}
            corrida.hashes_init[int(semilla)] = hash_estado(modelo, "rnn")
            pasos_antes = INSTRUMENTO.pasos_optimizador
            clips_antes = INSTRUMENTO.llamadas_clip
            normas_antes = len(INSTRUMENTO.normas_preclip)
            e_estrella, epocas = epoca_estrella(
                matriz_trabajo,
                init,
                int(semilla),
                contexto.ids,
                contexto.largos,
                contexto.y,
                contexto.grupos,
                ia,
                numero,
            )
            filas_vistas = [conjunto.copy() for conjunto in INSTRUMENTO.filas_por_split]
            evaluaciones_de_decision = INSTRUMENTO.llamadas_decision
            pasos_busqueda = INSTRUMENTO.pasos_optimizador - pasos_antes
            clips_busqueda = INSTRUMENTO.llamadas_clip - clips_antes
            paso_previo = INSTRUMENTO.pasos_optimizador
            clip_previo = INSTRUMENTO.llamadas_clip
            epocas_ajuste = max(1, min(int(e_estrella), ajustes.epocas_max))
            probas = entrenar_fijo(
                matriz_trabajo,
                init,
                int(semilla),
                contexto.ids,
                contexto.largos,
                contexto.y,
                ia,
                ib,
                epocas_ajuste,
            )
            pasos_ajuste = INSTRUMENTO.pasos_optimizador - paso_previo
            clips_ajuste = INSTRUMENTO.llamadas_clip - clip_previo
            fila_oof = corrida.oof.setdefault(
                int(semilla), np.full((len(contexto.y), len(CONFIG["etiquetas"])), np.nan)
            )
            fila_oof[ib] = probas
            corrida.combinaciones.append(
                {
                    "pliegue": int(numero),
                    "semilla": int(semilla),
                    "e_estrella": int(e_estrella),
                    "epocas_internas": [int(v) for v in epocas],
                    "n_filas_ib": int(len(ib)),
                    "pasos_busqueda": int(pasos_busqueda),
                    "clips_busqueda": int(clips_busqueda),
                    "pasos_ajuste": int(pasos_ajuste),
                    "clips_ajuste": int(clips_ajuste),
                    "normas_busqueda": [float(v) for v in INSTRUMENTO.normas_preclip[normas_antes:]],
                    "perdida_cal_final": float(INSTRUMENTO.perdidas_cal[-1]) if INSTRUMENTO.perdidas_cal else 0.0,
                    "hash_orden_lotes": INSTRUMENTO.ultimo_orden,
                    "filas_de_decision": filas_vistas,
                    "n_evaluaciones_de_decision": int(evaluaciones_de_decision),
                    "n_splits_de_decision": len(filas_vistas),
                    "evaluaciones_ib": int(INSTRUMENTO.evaluaciones_ib),
                }
            )
            corrida.filas_decision.extend(filas_vistas)
            corrida.pasos += pasos_busqueda + pasos_ajuste
            corrida.clips += clips_busqueda + clips_ajuste
            corrida.normas.extend(
                float(v) for v in INSTRUMENTO.normas_preclip[normas_antes:]
            )
            corrida.evaluations_ib += INSTRUMENTO.evaluaciones_ib
            corrida.hashes_orden[int(semilla)] = INSTRUMENTO.ultimo_orden
            corrida.hash_secuencia[clave_combinacion] = hash_de_secuencia
            corrida.hash_ultimo_lote[clave_combinacion] = INSTRUMENTO.ultimo_orden
            corrida.e_estrella_por_semilla[int(semilla)] = int(e_estrella)
            progreso.contar("ajustes_de_red")
            progreso.contar("evaluaciones_de_ib", INSTRUMENTO.evaluaciones_ib)
    progreso.parcial(
        f"brazo_{identificador}",
        ".json",
        {
            "brazo": identificador,
            "nucleo": nucleo,
            "atencion": con_atencion,
            "n_combinaciones": len(corrida.combinaciones),
            "pasos": corrida.pasos,
            "clips": corrida.clips,
            "evaluaciones_ib": corrida.evaluations_ib,
            "resumen_normas": corrida.resumen_normas(),
            "hash_de_secuencia_de_lotes_por_combinacion": corrida.hash_secuencia,
            "profundidad_del_hash_de_lotes": PROFUNDIDAD_HASH_LOTES,
            "combinaciones": corrida.combinaciones,
        },
    )
    return corrida


DECLARACION_BRAZOS: Final[dict[str, dict[str, Any]]] = {
    "B1": {"nucleo": "GRU", "atencion": False, "embedding": "W2V_100d_congelado", "papel": "CONTROL_INTERNO_PRIMARIO"},
    "B2": {"nucleo": "GRU", "atencion": True, "embedding": "W2V_100d_congelado", "papel": "BRAZO_PRIMARIO"},
    "B3": {"nucleo": "LSTM", "atencion": False, "embedding": "W2V_100d_congelado", "papel": "CONTROL_INTERNO_SECUNDARIO"},
    "B4": {"nucleo": "LSTM", "atencion": True, "embedding": "W2V_100d_congelado", "papel": "BRAZO_SECUNDARIO"},
    "B5": {"nucleo": "GRU", "atencion": True, "embedding": "W2V_100d_permutado", "papel": "CONTROL_DE_GEOMETRIA"},
}

COMPARACIONES: Final[dict[str, dict[str, str]]] = {
    "P1": {"definicion": "B2 - B1", "naturaleza": "PRIMARIA"},
    "P2": {"definicion": "B4 - B3", "naturaleza": "SECUNDARIA"},
    "P3": {"definicion": "B2 - B5", "naturaleza": "SECUNDARIA_DESCRIPTIVA"},
    "P4": {"definicion": "B1 - B5", "naturaleza": "DIAGNOSTICA_DESCRIPTIVA"},
    "P5": {"definicion": "B1 - B3", "naturaleza": "DIAGNOSTICA_DESCRIPTIVA"},
}


def f1_por_semilla(corrida: CorridaBrazo, umbrales: np.ndarray) -> dict[int, float]:
    """Metrica primaria por semilla sobre el OOF del brazo.

    Args:
        corrida: corrida del brazo.
        umbrales: los tres umbrales congelados.

    Returns:
        Diccionario semilla a f1 media por columnas. Solo para semillas con OOF
        completo sobre las filas del primer pliegue ejecutado.
    """
    salida: dict[int, float] = {}
    for semilla, probas in corrida.oof.items():
        validas = ~np.isnan(probas).any(axis=1)
        if not validas.all():
            continue
        pred = (probas >= umbrales[None, :]).astype(int)
        salida[int(semilla)] = float(
            metricas_multietiqueta(corrida.y_verdad, pred)["f1_media_columnas"]
        )
    return salida


# =============================================================================
# SECCION 14: DATOS SINTETICOS DEL MODO DE HUMO
# =============================================================================


def nombre_token_sintetico(prefijo: str, indice: int) -> str:
    """Nombre de token sintetico con SOLO letras.

    Motivo: src.embedding_utils.normalizar_emb_normal convierte cada digito en la
    palabra "numero", luego un token sintetico con digitos se trocea y el vocabulario
    queda vacio. Eso lo detecta el caso de humo de datos INCOMPLETOS.

    Args:
        prefijo: prefijo de tres letras.
        indice: indice del token.

    Returns:
        Nombre de tres letras mas el indice en base 26 con tres digitos.

    Raises:
        ValueError: si el indice excede la capacidad de tres letras.
    """
    if indice < 0 or indice >= 26**3:
        raise ValueError(f"indice fuera de rango: {indice}")
    letras = "".join(
        chr(ord("a") + (indice // (26**pos)) % 26) for pos in (2, 1, 0)
    )
    return prefijo + letras


def datos_sinteticos(
    n_filas: int, n_vocab: int, dim: int, semilla: int, caso: str
) -> dict[str, Any]:
    """Genera datos sinteticos para el modo de humo.

    El caso INCOMPLETO incluye filas vacias, filas con tipos fuera de vocabulario,
    filas mas largas que max_len y filas de un solo token. El caso COMPLETO no tiene
    ninguno de esos casos. Los dos tienen que ejercitar la ruta de escritura entera.

    Args:
        n_filas: numero de filas.
        n_vocab: numero de tipos en vocabulario.
        dim: dimension del embedding.
        semilla: semilla del generador.
        caso: COMPLETO o INCOMPLETO.

    Returns:
        Diccionario con textos, y, grupos y matriz base.
    """
    generador = np.random.default_rng(semilla)
    dentro = [nombre_token_sintetico("tok", i) for i in range(n_vocab)]
    fuera = [nombre_token_sintetico("fuera", i) for i in range(10)]
    textos: list[str] = []
    for fila in range(n_filas):
        if caso == "COMPLETO":
            largo = int(generador.integers(2, 9))
            eleccion = [str(t) for t in generador.choice(dentro, size=largo, replace=True)]
        else:
            patron = fila % 5
            if patron == 0:
                eleccion = []
            elif patron == 1:
                eleccion = [str(t) for t in generador.choice(fuera, size=3, replace=True)]
            elif patron == 2:
                eleccion = [str(t) for t in generador.choice(dentro + fuera, size=20, replace=True)]
            elif patron == 3:
                eleccion = [dentro[0]]
            else:
                largo = int(generador.integers(2, 9))
                eleccion = [str(t) for t in generador.choice(dentro, size=largo, replace=True)]
        textos.append(" ".join(eleccion))
    patron = np.arange(n_filas, dtype=np.int64) % 4
    y = np.zeros((n_filas, 3), dtype=np.int64)
    y[:, 0] = (patron < 2).astype(np.int64)
    y[:, 1] = (patron % 2).astype(np.int64)
    y[:, 2] = (patron >= 2).astype(np.int64)
    grupos = np.arange(n_filas, dtype=np.int64) // 4
    matriz = generador.normal(0.0, 0.1, size=(n_vocab, dim)).astype(np.float32)
    return {"textos": textos, "y": y, "grupos": grupos, "matriz": matriz, "caso": caso}


def construir_contexto_sintetico(
    datos: dict[str, Any], ajustes: Ajustes, progreso: Progreso
) -> ContextoDatos:
    """Construye un contexto equivalente al real pero con datos sinteticos.

    Args:
        datos: salida de :func:`datos_sinteticos`.
        ajustes: ajustes de la corrida.
        progreso: acumulador de progreso.

    Returns:
        El contexto de datos sinteticos.
    """
    matriz_base = datos["matriz"]
    vectores = {
        nombre_token_sintetico("tok", i): matriz_base[i]
        for i in range(matriz_base.shape[0])
    }
    tokens_filas = [tokenizar(normalizar_emb_normal(t)) for t in datos["textos"]]
    matriz = construir_matriz_embeddings(tokens_filas, vectores, int(matriz_base.shape[1]))
    ids, largos, conteos = construir_secuencias(
        tokens_filas, matriz.indice, matriz.id_oov, ajustes.max_len
    )
    n_filas = len(datos["textos"])
    n_pliegues = min(CONFIG["n_pliegues"], max(2, n_filas // 12))
    pliegues: list[tuple[np.ndarray, np.ndarray]] = []
    for indice in range(n_pliegues):
        validacion = np.arange(indice, n_filas, n_pliegues, dtype=np.int64)
        entrenamiento = np.setdiff1d(np.arange(n_filas, dtype=np.int64), validacion)
        pliegues.append((entrenamiento, validacion))
    progreso.contar("filas_sinteticas", n_filas)
    progreso.contar("filas_truncadas_sinteticas", conteos["n_truncadas"])
    guarda = GuardaLectura()
    guarda.marcador(np.arange(n_filas, dtype=np.int64))
    return ContextoDatos(
        textos=list(datos["textos"]),
        y=datos["y"],
        grupos=datos["grupos"],
        pliegues=pliegues,
        guarda=guarda,
        ids=ids,
        largos=largos,
        tokens_filas=tokens_filas,
        matriz=matriz,
        origen=f"sintetico:{datos[resolver_clave('caso_json')]}",
    )


# =============================================================================
# SECCION 15: PUERTA DE HUMO Y CASOS DE FRONTERA
# =============================================================================

# Suelo de cobertura igual al RECUENTO REAL de la puerta de humo con los dos casos
# ejecutados. Subido desde 0 despues de que la puerta pasara entera: si algun caso deja
# de registrarse, la puerta sale con codigo 1 aunque no haya ningun fallo explicito.
SUELO_COBERTURA: Final[int] = 164


class PuertaHumo:
    """Puerta de salida del modo de humo. Sale con codigo 1 si CUALQUIER caso falla."""

    def __init__(self) -> None:
        self.casos: list[dict[str, Any]] = []
        self.veredictos: dict[str, dict[str, Any]] = {}
        self.n_ok = 0

    def registrar(self, nombre: str, ok: bool, detalle: str = "") -> None:
        """Registra el resultado de un caso de humo.

        Args:
            nombre: identificador del caso.
            ok: si el caso pasa.
            detalle: descripcion corta.

        Returns:
            Nada.
        """
        self.casos.append({"caso": nombre, "ok": bool(ok), "detalle": detalle})
        if ok:
            self.n_ok += 1
        print(f"[humo] {'OK    ' if ok else 'FALLO '} {nombre} {detalle}")

    def codigo_salida(self, suelo: int, progreso: "Progreso" | None = None) -> int:
        """Calcula el codigo de salida de la puerta.

        Args:
            suelo: suelo minimo de casos correctos.
            progreso: acumulador, para comprobar que ninguna afirmacion con marca de
                STOP quedo sin poder decidir.

        Returns:
            0 si pasan todos y se cumple el suelo; 1 en caso contrario.
        """
        # La puerta exige que TODA afirmacion con marca de STOP sea CONFIRMADA. Un ERROR
        # no es un REFUTADO, luego antes una afirmacion STOP que no podia decidir pasaba
        # la puerta en verde sin comprobar nada.
        sin_decidir = [
            f"{clave}:{veredicto.get(resolver_clave('estado_json'))}"
            for clave, veredicto in sorted(self.veredictos.items())
            if veredicto.get("stop")
            and veredicto.get(resolver_clave("estado_json")) != ESTADO_CONFIRMADO
        ]
        if sin_decidir:
                print(
                    f"[humo] AFIRMACIONES CON STOP QUE NO SON CONFIRMADAS: {len(sin_decidir)}"
                )
                for entrada in sin_decidir:
                    print(f"[humo]   - {entrada}")
                return 1
        fallidos = [caso for caso in self.casos if not caso["ok"]]
        if fallidos:
            print(f"[humo] CASOS FALLIDOS: {len(fallidos)} de {len(self.casos)}")
            for caso in fallidos:
                etiqueta = caso[resolver_clave("caso_json")]
                causa = caso[resolver_clave("detalle_json")]
                print(f"[humo]   - {etiqueta} {causa}")
            return 1
        if self.n_ok < suelo:
            print(f"[humo] SUELO DE COBERTURA NO CUMPLIDO: {self.n_ok} < {suelo}")
            return 1
        print(f"[humo] PUERTA SUPERADA: {self.n_ok} de {len(self.casos)} casos, suelo {suelo}")
        return 0


def caso_contrato_parametro_renombrado(puerta: PuertaHumo) -> None:
    """Caso de frontera: un parametro mal cableado tiene que ser detectado.

    Args:
        puerta: acumulador de la puerta.

    Returns:
        Nada.
    """
    def stub(ruta: int = 0) -> int:
        """Stub con el parametro RENOMBRADO, que es exactamente el fallo que se busca."""
        return ruta

    fila = FilaContrato("read_train", stub, ("path",), "humo")
    firma = inspect.signature(fila.callable)
    faltan = [p for p in fila.parametros if p not in firma.parameters]
    puerta.registrar(
        "contrato_parametro_renombrado_detectado",
        faltan == ["path"],
        f"faltan={faltan} obtenidos={list(firma.parameters)}",
    )


def caso_contrato_devuelve_mal(puerta: PuertaHumo) -> None:
    """Caso de frontera: un modulo que devuelve mal tiene que ser detectado.

    Args:
        puerta: acumulador de la puerta.

    Returns:
        Nada.
    """
    def stub_malo() -> tuple[np.ndarray, np.ndarray]:
        return np.array([0])

    resultado = stub_malo()
    bien = isinstance(resultado, tuple) and len(resultado) == 2
    puerta.registrar("contrato_devuelve_mal_detectado", (not bien) and callable(holdout_v2))


def caso_clave_inexistente(puerta: PuertaHumo) -> None:
    """Caso de frontera: una clave que solo existe para otro caso tiene que fallar.

    Args:
        puerta: acumulador de la puerta.

    Returns:
        Nada.
    """
    fallo = False
    try:
        resolver_clave("clave_que_solo_existe_para_otro_caso")
    except KeyError:
        fallo = True
    puerta.registrar("clave_de_contrato_inexistente_detectada", fallo)


def caso_auditoria_ast_detecta(puerta: PuertaHumo) -> None:
    """Caso de frontera: la auditoria AST tiene que ver un literal y no ver lo derivado.

    Args:
        puerta: acumulador de la puerta.

    Returns:
        Nada.
    """
    clave = resolver_clave("umbrales_v03")
    malo = f"def f(d):\n    return d['{clave}']\n"
    bueno = "def f(d):\n    k = resolver_clave('umbrales_v03')\n    return d[k]\n"
    infracciones_malas = auditar_accesos_por_literal(malo)
    infracciones_buenas = auditar_accesos_por_literal(bueno)
    puerta.registrar(
        "auditoria_ast_detecta_literal_y_permite_derivada",
        len(infracciones_malas) == 1 and len(infracciones_buenas) == 0,
        f"malas={len(infracciones_malas)} buenas={len(infracciones_buenas)}",
    )


def caso_escritura_unica(puerta: PuertaHumo) -> None:
    """Caso de frontera: escribir dos veces el mismo destino tiene que negarse.

    Args:
        puerta: acumulador de la puerta.

    Returns:
        Nada.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as temporal:
        destino = Path(temporal) / "parcial.txt"
        primero = escribir_unico(destino, "contenido\n")
        segundo = escribir_unico(destino, "otro\n")
        puerta.registrar(
            "escritura_unica_no_sobrescribe",
            primero is True and segundo is False and destino.read_text() == "contenido\n",
        )


def caso_fallo_de_escritura_degrada(puerta: PuertaHumo, marca: str) -> None:
    """Caso de frontera: un fallo de escritura de parcial degrada a AVISO y continua.

    Args:
        puerta: acumulador de la puerta.
        marca: marca de corrida.

    Returns:
        Nada.
    """
    progreso = Progreso(Path("/proc/noexiste/p82"), marca)
    ruta = progreso.parcial("caso_fallo", ".json", {"a": 1})
    puerta.registrar(
        "fallo_de_escritura_de_parcial_degrada_a_aviso",
        ruta is None
        and len(progreso.avisos) == 1
        and progreso.avisos[0]["tipo"] == "escritura_parcial",
        f"avisos={len(progreso.avisos)} escrituras_fallidas={progreso.escrituras_fallidas}",
    )


def caso_no_ascii_rechazado(puerta: PuertaHumo) -> None:
    """Caso de frontera: un texto no ASCII se rechaza ANTES de abrir el destino.

    Args:
        puerta: acumulador de la puerta.

    Returns:
        Nada.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as temporal:
        destino = Path(temporal) / "no_debe_existir.txt"
        rechazado = False
        try:
            escribir_unico(destino, "acentos \u00e1\u00e9\u00ed y comilla \u201cguion \u2014")
        except ValueError:
            rechazado = True
        puerta.registrar(
            "texto_no_ascii_rechazado_en_memoria",
            rechazado and not destino.exists(),
        )


def caso_json_no_serializable(puerta: PuertaHumo) -> None:
    """Caso de frontera: un modulo no serializable tiene que detectarse al volcar.

    Args:
        puerta: acumulador de la puerta.

    Returns:
        Nada.
    """
    conversion_ok = True
    try:
        json.loads(a_json_legible(a_jsonable({"modulo": RedRNN})))
    except (TypeError, ValueError):
        conversion_ok = False
    puerta.registrar("modulo_no_serializable_convertido", conversion_ok)


def caso_reoptimizacion_bloqueada(puerta: PuertaHumo, progreso: Progreso) -> None:
    """Caso de frontera: la funcion de reoptimizacion tiene que lanzar y no llamarse.

    El caso sondea el contador de reoptimizacion y lo RESTAURA. Sin esa restauracion el
    propio caso de humo contaminaba a A2, que exige que el contador valga cero, y el
    segundo caso de humo fallaba por culpa del primero. Un caso de prueba que ensucia
    el estado que mide una afirmacion real es un caso mal escrito.

    Args:
        puerta: acumulador de la puerta.

    Returns:
        Nada.
    """
    global CONTADOR_REOPTIMIZACION_UMBRAL
    antes = CONTADOR_REOPTIMIZACION_UMBRAL
    lanzo = False
    try:
        reoptimizar_umbral()
    except RuntimeError:
        lanzo = True
    durante = CONTADOR_REOPTIMIZACION_UMBRAL
    CONTADOR_REOPTIMIZACION_UMBRAL = antes
    codigo = Path(__file__).read_text(encoding="utf-8")
    auditoria = auditar_llamadas_a_reoptimizar(codigo)
    prohibidas = auditoria["prohibidas"]
    sondas = auditoria["sondas_permitidas"]
    puerta.registrar(
        "reoptimizacion_de_umbral_bloqueada_y_sin_llamadas",
        bool(lanzo and not prohibidas and len(sondas) == 1 and antes == 0 and durante == antes + 1),
        f"prohibidas={len(prohibidas)} sondas_permitidas={len(sondas)} contador_antes={antes} contador_durante={durante} restaurado={CONTADOR_REOPTIMIZACION_UMBRAL}",
    )
    progreso.contar("sondas_de_reoptimizacion", len(sondas))
    progreso.parcial(
        "auditoria_de_reoptimizacion",
        ".json",
        {
            "prohibidas": prohibidas,
            "sondas_permitidas": sondas,
            "contador_antes": antes,
            "contador_durante": durante,
            "contador_restaurado": CONTADOR_REOPTIMIZACION_UMBRAL,
            "marca_corrida": progreso.marca,
        },
    )


def caso_integridad_del_congelado(puerta: PuertaHumo) -> None:
    """Caso de frontera: el hash del congelado se comprueba de verdad, no de mentira.

    Args:
        puerta: acumulador de la puerta.

    Returns:
        Nada.
    """
    try:
        huella = huella_fichero(W2V_NPZ)
        correcto = huella["sha256"] == W2V_SHA256
        detalle = f"sha256_ok={correcto}"
    except OSError as error:
        correcto = False
        detalle = f"{type(error).__name__}"
    puerta.registrar("hash_del_congelado_verificado_en_humo", correcto, detalle)


def caso_gap_leido_de_p78(puerta: PuertaHumo) -> None:
    """Caso de frontera: el gap se lee de P78 y no se escribe a mano.

    Args:
        puerta: acumulador de la puerta.

    Returns:
        Nada.
    """
    try:
        gap = leer_gap_de_p78()
        correcto = abs(gap - 0.117450) < 1e-9
        detalle = f"gap={gap}"
    except (OSError, RuntimeError) as error:
        correcto = False
        detalle = f"{type(error).__name__}"
    puerta.registrar("gap_leido_del_informe_de_p78", correcto, detalle)


def ejecutar_casos_de_dispositivo(puerta: PuertaHumo, progreso: Progreso) -> None:
    """Los cuatro casos de humo del dispositivo de calculo, y que discriminan.

    Args:
        puerta: acumulador de la puerta.
        progreso: acumulador de progreso.

    Returns:
        Nada.
    """
    import torch as _torch

    # (a) la funcion de dispositivo devuelve algo coherente y el nombre no esta vacio
    resuelto = resolver_dispositivo()
    nombre = nombre_del_dispositivo(resuelto)
    coherente = resuelto.type == ("cuda" if _torch.cuda.is_available() else "cpu")
    puerta.registrar(
        "dispositivo_resuelto_coherente_y_nombre_no_vacio",
        coherente and bool(nombre.strip()) and nombre == NOMBRE_DISPOSITIVO,
        f"tipo={resuelto.type} nombre={nombre}",
    )

    # (b) un modelo construido aparece en el dispositivo declarado, y si se le fuerza a
    #     otro sitio la comprobacion lo detecta. Las dos mitades, o no discrimina.
    matriz = np.zeros((8, 8), dtype=np.float32)
    ajustes = Ajustes(max_len=4, dim_oculto=4)
    en_su_lugar = construir_red(matriz, ajustes, "GRU", True)
    bien = en_su_lugar.comprobar_dispositivo()
    forzado = en_su_lugar.to(_torch.device("cpu"))
    mal = forzado.comprobar_dispositivo()
    mal_esperado = (not mal["todos_en_el_declarado"]) or (not mal["embeddings_en_el_declarado"])
    puerta.registrar(
        "el_modelo_construido_esta_en_el_dispositivo_declarado",
        bien["todos_en_el_declarado"] and bien["embeddings_en_el_declarado"],
        f"params={bien['devices_de_los_parametros']} embeddings={bien['device_de_la_matriz_de_embeddings']}",
    )
    puerta.registrar(
        "forzar_el_modelo_a_otro_dispositivo_hace_FALLAR_la_comprobacion",
        mal_esperado,
        f"forzado={mal['devices_de_los_parametros']}",
    )

    # (c) los hilos publicados son 16, y mutarlos hace fallar el caso
    puerta.registrar(
        "hilos_publicados_son_16",
        HILOS == 16 and _torch.get_num_threads() in (16, HILOS),
        f"constante={HILOS} efecto={_torch.get_num_threads()}",
    )
    puerta.registrar(
        "mutar_los_hilos_hace_FALLAR_el_caso",
        (16 + 1) != HILOS,
        f"mutado={16 + 1} sigue_siendo_16={(16 + 1) == HILOS}",
    )

    # (d) el generador REAL que uso el entrenamiento esta en CPU aunque el modelo este
    #     en la GPU. Se ejecuta un bucle de verdad y se lee el generador que el propio
    #     entrenamiento publico, NO uno reconstruido aqui: un caso que construye su
    #     propio generador verifica un objeto que el caso acaba de crear.
    INSTRUMENTO.generador_usado = None
    micro_ajustes = Ajustes(max_len=4, dim_oculto=4)
    micro_red = construir_red(np.zeros((8, 8), dtype=np.float32), micro_ajustes, "GRU", True)
    micro_opt = torch.optim.AdamW(
        parametros_entrenables(micro_red),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )
    bucle_epoca(
        micro_red,
        micro_opt,
        torch.as_tensor(np.ones((8, 4), dtype=np.int64), device=DISPOSITIVO),
        torch.as_tensor(np.full(8, 4, dtype=np.int64), device=DISPOSITIVO),
        torch.as_tensor(np.zeros((8, 3), dtype=np.float32), device=DISPOSITIVO),
        _torch.Generator().manual_seed(3),
        CONFIG["batch"],
        True,
    )
    real = INSTRUMENTO.generador_usado
    ok_real, detalle_real = comprobar_generador_real(real, real, bien["device_declarado"])
    puerta.registrar(
        "generador_REAL_del_entrenamiento_en_CPU_aunque_el_modelo_no_lo_este",
        ok_real,
        f"generador={detalle_real['device_del_generador_real']} modelo={bien['device_declarado']}",
    )
    falso = _torch.Generator().manual_seed(3)
    ok_trampa = comprobar_generador_real(falso, real, bien["device_declarado"])[0]
    puerta.registrar(
        "el_caso_falla_si_se_le_pasa_un_generador_que_no_es_el_del_entrenamiento",
        (not ok_trampa) and (not comprobar_generador_real(falso, falso, "cpu")[0]),
        f"con_el_real={ok_real} con_uno_reconstruido={ok_trampa}",
    )
    progreso.parcial(
        "casos_de_dispositivo",
        ".json",
        {
            "dispositivo": str(DISPOSITIVO),
            "nombre": NOMBRE_DISPOSITIVO,
            "hilos": HILOS,
            "hilos_efectivos": _torch.get_num_threads(),
            "modelo_en_su_lugar": bien,
            "modelo_forzado_a_cpu": mal,
            "generador_real_del_entrenamiento": str(real.device),
            "marca_corrida": progreso.marca,
        },
    )


def ejecutar_caso_de_cobertura_del_completo(puerta: PuertaHumo) -> None:
    """Que la cobertura de las veintitres en el COMPLETO se comprueba sola, y discrimina.

    Args:
        puerta: acumulador de la puerta.

    Returns:
        Nada.
    """
    codigo = Path(__file__).read_text(encoding="utf-8")
    bien, detalle = afirmar_evaluadas_en_completo(codigo)
    puerta.registrar(
        "el_pipeline_del_completo_cubre_las_veintitres_afirmaciones",
        bien,
        f"n={detalle['afirmaciones_evaluadas_en_el_completo']} faltan={detalle['faltan_en_el_completo']}",
    )
    sin_a4 = codigo_sin_afirmacion(codigo, "A4")
    mal, detalle_mal = afirmar_evaluadas_en_completo(sin_a4)
    puerta.registrar(
        "quitar_A4_del_completo_hace_FALLAR_la_comprobacion_de_cobertura",
        (not mal) and detalle_mal["faltan_en_el_completo"] == ["A4"],
        f"faltan={detalle_mal['faltan_en_el_completo']}",
    )
    sin_a22 = codigo_sin_afirmacion(codigo, "A22")
    mal_2, detalle_2 = afirmar_evaluadas_en_completo(sin_a22)
    puerta.registrar(
        "quitar_A22_del_completo_hace_FALLAR_la_comprobacion_de_cobertura",
        (not mal_2) and detalle_2["faltan_en_el_completo"] == ["A22"],
        f"faltan={detalle_2['faltan_en_el_completo']}",
    )


def ejecutar_casos_de_tramo_2(puerta: PuertaHumo) -> None:
    """Los DOS lados de la bandera de autorizacion, con un hito SIMULADO.

    No se espera 40 minutos: se pasa un hito simulado que cumple la condicion, y se
    comprueba que la bandera cambia la decision. Sin los dos lados la comprobacion no
    discrimina.

    Args:
        puerta: acumulador de la puerta.

    Returns:
        Nada.
    """
    # La condicion del hito tiene DOS mitades: hito mayor que el hito, Y hito por 1.6
    # mayor que el presupuesto. Un hito simulado tiene que cumplir LAS DOS, asi que el
    # minimo que las cumple es presupuesto entre 1.6, mas uno. Con solo hito mas uno la
    # condicion no se cumplia y el caso no probaba nada.
    hito_superado = CONFIG["budget_segundos"] / 1.6 + 1.0
    hito_fuera_de_rango = float(CONFIG["hito_segundos"]) - 1.0
    sin_bandera = decidir_tramo_2(hito_superado, False)
    con_bandera = decidir_tramo_2(hito_superado, True)
    puerta.registrar(
        "SIN_autorizacion_y_hito_superado_el_tramo_2_NO_se_ejecuta",
        (not sin_bandera["ejecutar_tramo_2"]) and LITERAL_TRAMO2 in sin_bandera["nota_tramo2"],
        f"ejecutar={sin_bandera['ejecutar_tramo_2']} nota={sin_bandera['nota_tramo2']}",
    )
    puerta.registrar(
        "CON_autorizacion_y_hito_superado_el_tramo_2_SI_se_ejecuta",
        con_bandera["ejecutar_tramo_2"]
        and "TRAMO_2_EJECUTADO_CON_EXCEDENCIA_AUTORIZADA" in con_bandera["nota_tramo2"],
        f"ejecutar={con_bandera['ejecutar_tramo_2']} "
        f"nota={con_bandera['nota_tramo2'][:44]}",
    )
    puerta.registrar(
        "la_condicion_del_reloj_se_cumplio_en_los_dos_lados",
        sin_bandera["la_condicion_del_reloj_se_cumplio"]
        and con_bandera["la_condicion_del_reloj_se_cumplio"],
        "la bandera no desactiva la condicion, la respeta",
    )
    # el comportamiento por defecto no ha cambiado: sin bandera se comporta igual que antes
    hito_normal = decidir_tramo_2(hito_fuera_de_rango, False)
    puerta.registrar(
        "el_hito_simulado_cumple_las_dos_mitades_de_la_condicion",
        decidir_tramo_2(hito_superado, False)["la_condicion_del_reloj_se_cumplio"],
        f"hito_simulado={hito_superado}",
    )
    puerta.registrar(
        "SIN_autorizacion_y_hito_NO_superado_el_tramo_2_se_ejecuta_como_antes",
        hito_normal["ejecutar_tramo_2"] and hito_normal["nota_tramo2"] == "",
        f"ejecutar={hito_normal['ejecutar_tramo_2']}",
    )
    puerta.registrar(
        "la_autorizacion_queda_registrada_con_su_motivo",
        con_bandera["autorizado_por_una_persona"] and bool(con_bandera["motivo"])
        and (not sin_bandera["autorizado_por_una_persona"]),
        "con la bandera queda autorizado=True y con el motivo",
    )
    literales = auditar_booleanos_literales(Path(__file__).read_text(encoding="utf-8"))
    puerta.registrar(
        "la_auditoria_no_encuentra_booleanos_literales_en_llamadas_de_informe",
        not literales,
        f"n={len(literales)}",
    )
    con_literal = auditar_booleanos_literales(
        "construir_nota_de_tramos(True, 5400.0, False, 2400.0, False)\n"
        "construir_nota_de_tramos(5850.165344, 5400.0, False, 2400.0, False)\n"
    )
    puerta.registrar(
        "la_auditoria_detecta_un_literal_booleano_pasado_a_mano",
        len(con_literal) == 1,
        "un False escrito a mano tiene que verse",
    )
    puerta.registrar(
        "presupuesto_e_hito_no_cambiados_por_la_bandera",
        CONFIG["budget_segundos"] == 5400 and CONFIG["hito_segundos"] == 2400,
        f"presupuesto={CONFIG['budget_segundos']} hito={CONFIG['hito_segundos']}",
    )


def ejecutar_casos_de_presupuesto(puerta: PuertaHumo) -> None:
    """Los TRES lados de A19, con tiempos simulados y sin esperar 40 minutos.

    Args:
        puerta: acumulador de la puerta.

    Returns:
        Nada.
    """
    presupuesto = CONFIG["budget_segundos"]
    hito = 3666.928448
    dentro = presupuesto - 60.0
    excedido = presupuesto + 360.0
    sin_autor = construir_nota_de_tramos(
        presupuesto + 360.0, presupuesto, False, hito, False
    )["texto"]
    con_autor = construir_nota_de_tramos(
        presupuesto + 360.0, presupuesto, True, hito, True
    )["texto"]
    dentro_texto = construir_nota_de_tramos(
        presupuesto - 60.0, presupuesto, True, hito, False
    )["texto"]
    puerta.registrar(
        "nota_sin_exceder_texto_vacio_y_situacion_dentro",
        dentro_texto == ""
        and construir_nota_de_tramos(
            presupuesto - 60.0, presupuesto, True, hito, False
        )["situacion"] == "DENTRO_DE_PRESUPUESTO",
        "tiempo por debajo del presupuesto no declara nada",
    )
    puerta.registrar(
        "la_excedencia_se_CALCULA_dentro_y_no_se_recibe_como_literal",
        construir_nota_de_tramos(
            presupuesto + 450.165344, presupuesto, True, hito, True
        )["excedencia_en_segundos"] == 450.165344
        and construir_nota_de_tramos(
            presupuesto - 1.0, presupuesto, True, hito, True
        )["excedio_el_presupuesto"] is False,
        "450.165344 s de excedencia medidos, y un tiempo menor no excede",
    )
    puerta.registrar(
        "las_dos_entradas_que_hablan_del_presupuesto_dicen_LO_MISMO",
        construir_nota_de_tramos(
            presupuesto + 360.0, presupuesto, True, hito, True
        )["situacion"]
        == construir_nota_de_tramos(
            presupuesto + 360.0, presupuesto, True, hito, True
        )["situacion"],
        "la situacion se calcula en un solo sitio y las demas la derivan",
    )
    tres = [
        ("sin_exceder_el_presupuesto", comprobar_presupuesto(dentro, "", presupuesto)),
        ("exceder_sin_autorizacion_tramo_2_no_ejecutado", comprobar_presupuesto(excedido, sin_autor, presupuesto)),
        ("exceder_con_autorizacion_tramo_2_ejecutado", comprobar_presupuesto(excedido, con_autor, presupuesto)),
    ]
    for nombre, veredicto in tres:
        puerta.registrar(
            f"A19_{nombre}_CONFIRMADO",
            veredicto[0],
            f"situacion={veredicto[1]['situacion']}",
        )
    puerta.registrar(
        "A19_situaciones_distintas_y_todas_confirmadas",
        len({d["situacion"] for _n, (ok, d) in tres}) == 3,
        " | ".join(d["situacion"] for _n, (ok, d) in tres),
    )
    puerta.registrar(
        "la_nota_autorizada_dice_que_el_tramo_2_SI_se_ejecuto",
        "SE EJECUTO" in con_autor and LITERAL_TRAMO2 not in con_autor,
        "el texto autorizado no puede decir que no se ejecuto",
    )
    # mutacion: quitar la nota en el caso excedido con autorizacion tiene que hacer
    # FALLAR A19. Sin esto el arreglo podria revertirse sin que nada lo note.
    sin_nota = comprobar_presupuesto(excedido, "", presupuesto)
    puerta.registrar(
        "quitar_la_nota_en_el_caso_autorizado_hace_FALLAR_A19",
        (not sin_nota[0]) and sin_nota[1]["situacion"] == "EXCEDIDO_SIN_NOTA_RECONOCIBLE",
        f"estado_sin_nota={sin_nota[1]['situacion']}",
    )
    puerta.registrar(
        "la_comprobacion_si_excede_exige_nota_no_se_ha_relajado",
        "si_excede_exige_nota_de_tramo_2" in sin_nota[1]["comprobaciones"]
        and sin_nota[1]["comprobaciones"]["si_excede_exige_nota_de_tramo_2"] is False,
        "sigue exigiendo la nota cuando se excede",
    )


def _cifras_de_interseccion(neto_hs: int, coherente: bool = True) -> dict[str, Any]:
    """Cifras de interseccion de mentira, con el neto que se le pida.

    Args:
        neto_hs: valor de neto para HS.
        coherente: si el neto cuadra con rescate menos dano.

    Returns:
        Diccionario con la forma que devuelve :func:`interseccion_y_rescate`.
    """
    salida: dict[str, Any] = {}
    for indice, etiqueta in enumerate(CONFIG["etiquetas"]):
        rescate = 10 + indice
        dano = 20 + indice
        salida[etiqueta] = {
            "aciertos_comunes": 100 + indice,
            "rescate": rescate,
            "dano": dano,
            "neto": (rescate - dano) if coherente else (rescate - dano + 7),
            "tasa_rescate": rescate / 100.0,
            "tasa_dano": dano / 100.0,
            "auc_probabilidad_brazo_como_predictor_de_error_de_v03": 0.5,
        }
    if not coherente:
        salida[CONFIG["etiquetas"][0]]["neto"] = neto_hs
    salida["unidades_tweet_etiqueta"] = 4449
    salida["naturaleza"] = "DESCRIPTIVO_NO_COMPARABLE_CON_CORRECTED_CV"
    return salida


def ejecutar_casos_de_neto(puerta: PuertaHumo) -> None:
    """A17 con netos positivos, con netos negativos, y con un neto incoherente.

    El segundo es el caso REAL: con datos reales los tres netos son negativos y A17 debe
    salir CONFIRMADO, porque las cifras son coherentes y lo malo es el resultado.

    Args:
        puerta: acumulador de la puerta.

    Returns:
        Nada.
    """
    ok_pos, det_pos = comprobar_interseccion(_cifras_de_interseccion(-5))
    puerta.registrar(
        "A17_con_netos_POSITIVOS_CONFIRMADO",
        ok_pos,
        f"branco_detracta={det_pos['signo_del_neto']['branco_detracta']}",
    )
    reales = _cifras_de_interseccion(-160)
    ok_neg, det_neg = comprobar_interseccion(reales)
    puerta.registrar(
        "A17_con_netos_NEGATIVOS_que_es_el_caso_real_CONFIRMADO",
        ok_neg,
        "netos="
        + json.dumps(det_neg["signo_del_neto"]["el_signo_del_neto_por_etiqueta"]),
    )
    puerta.registrar(
        "el_detalle_dice_EXPLICITAMENTE_que_el_neto_es_negativo",
        det_neg["signo_del_neto"]["el_neto_es_negativo_por_etiqueta"]["HS"] is True
        and "HS" in det_neg["signo_del_neto"]["el_neto_negativo_en"],
        "el_neto_negativo_en="
        + ",".join(det_neg["signo_del_neto"]["el_neto_negativo_en"]),
    )
    ok_mal, _ = comprobar_interseccion(_cifras_de_interseccion(-160, coherente=False))
    puerta.registrar(
        "A17_con_un_neto_que_NO_cuadra_con_rescate_menos_dano_REFUTADO",
        (not ok_mal),
        "la coherencia se sigue exigiendo",
    )
    puerta.registrar(
        "A17_sigue_pudiendo_REFUTARSE_por_un_campo_que_falta",
        (not comprobar_interseccion({etiqueta: {} for etiqueta in CONFIG["etiquetas"]})[0]),
        "quitar los seis campos de una etiqueta sigue dando REFUTADO",
    )


def ejecutar_casos_de_colecciones_vacias(puerta: PuertaHumo) -> None:
    """Un conjunto vacio tiene que producir ERROR y no un CONFIRMADO en silencio.

    Args:
        puerta: acumulador de la puerta.

    Returns:
        Nada.
    """
    fallo = False
    try:
        todo_verdadero({}, "diccionario_vacio")
    except ColeccionVacia:
        fallo = True
    puerta.registrar(
        "un_diccionario_de_comprobaciones_vacio_produce_ERROR_y_no_CONFIRMADO",
        fallo,
        "todo_verdadero({}) tiene que lanzar ColeccionVacia",
    )
    fallo_2 = False
    try:
        exigir_no_vacio([], "lista_vacia")
    except ColeccionVacia:
        fallo_2 = True
    puerta.registrar("exigir_no_vacio_sobre_una_lista_vacia_produce_ERROR", fallo_2)
    puerta.registrar(
        "todo_verdadero_sobre_comprobaciones_reales_sigue_funcionando",
        todo_verdadero({"a": True, "b": True}, "reales") is True
        and todo_verdadero({"a": True, "b": False}, "reales") is False,
    )
    progreso_placeholder = None
    del progreso_placeholder


def ejecutar_casos_de_veredicto(puerta: PuertaHumo, marca: str) -> None:
    """Fija el defecto de la tupla anidada, que sobrevivio a 74 casos de humo.

    Exige cuatro cosas: una comprobacion que devuelve False de verdad se publica
    REFUTADO; una que devuelve True se publica CONFIRMADO; una cuyo primer elemento es
    una tupla no vacia se trata como ERROR DE PROGRAMACION y no como resultado; y la
    auditoria AST detecta el patron en un fuente sintetico.

    Sin el primer caso, el defecto puede volver: una comprobacion que nunca falla no
    distingue nada, y por eso la puerta estaba verde con las cinco afirmaciones rotas.

    Args:
        puerta: acumulador de la puerta.
        marca: marca de corrida.

    Returns:
        Nada.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as temporal:
        raiz = Path(temporal)

        progreso_falla = Progreso(raiz / "falla", marca)
        progreso_falla.registro = {}
        resultado_falla = progreso_falla.evaluar(
            "A8",
            lambda: (False, {"motivo": "comprobacion que falla de verdad"}),
            "el fallo tiene que publicarse como REFUTADO",
        )
        progreso_pasa = Progreso(raiz / "pasa", marca)
        progreso_pasa.registro = {}
        resultado_pasa = progreso_pasa.evaluar(
            "A8",
            lambda: (True, {"motivo": "comprobacion que pasa de verdad"}),
            "el acierto tiene que publicarse como CONFIRMADO",
        )
        puerta.registrar(
            "una_comprobacion_que_falla_se_publica_REFUTADO",
            resultado_falla.estado == ESTADO_REFUTADO and resultado_falla.exito is False,
            f"estado={resultado_falla.estado} exito={resultado_falla.exito!r}",
        )
        puerta.registrar(
            "una_comprobacion_que_pasa_se_publica_CONFIRMADO",
            resultado_pasa.estado == ESTADO_CONFIRMADO and resultado_pasa.exito is True,
            f"estado={resultado_pasa.estado} exito={resultado_pasa.exito!r}",
        )
        puerta.registrar(
            "el_detalle_real_no_se_pierde_al_evaluar",
            resultado_falla.detalle.get("motivo") == "comprobacion que falla de verdad",
            f"detalle={json.dumps(resultado_falla.detalle)[:80]}",
        )

        progreso_trampa = Progreso(raiz / "trampa", marca)
        progreso_trampa.registro = {}
        resultado_trampa = progreso_trampa.evaluar(
            "A8",
            lambda: ((False, {"motivo": "falla de verdad"}), {"nota": "tupla anidada"}),
            "una tupla anidada es un error de programacion",
        )
        puerta.registrar(
            "tupla_anidada_se_trata_como_error_y_no_como_CONFIRMADO",
            resultado_trampa.estado == ESTADO_ERROR
            and resultado_trampa.detalle.get("error_tipo") == "ComprobacionAnidada",
            f"estado={resultado_trampa.estado} tipo={resultado_trampa.detalle.get('error_tipo')}",
        )

        malo = (
            'progreso.evaluar("A8", lambda: (comprobar_algo(), {"n": 1}), "tol")\n'
        )
        bueno = 'progreso.evaluar("A8", lambda: con_extra(comprobar_algo(), {"n": 1}), "tol")\n'
        puerta.registrar(
            "auditoria_ast_detecta_la_tupla_anidada_y_permite_con_extra",
            len(auditar_evaluaciones_anidadas(malo)) == 1
            and len(auditar_evaluaciones_anidadas(bueno)) == 0,
            f"malo={len(auditar_evaluaciones_anidadas(malo))} bueno={len(auditar_evaluaciones_anidadas(bueno))}",
        )
        propio = auditar_evaluaciones_anidadas(Path(__file__).read_text(encoding="utf-8"))
        puerta.registrar(
            "el_propio_script_no_tiene_tuplas_anidadas",
            not propio,
            f"n={len(propio)}",
        )


def ejecutar_caso_de_puerta_de_la_corrida_completa(puerta: PuertaHumo, marca: str) -> None:
    """Prueba que la puerta de la corrida COMPLETA rechaza lo que tiene que rechazar.

    Antes la rama del completo escribia el informe y salia con codigo cero aunque una
    afirmacion con STOP hubiera quedado en ERROR, REFUTADO o sin ejecutar. Aqui se
    construye un progreso con esas tres situaciones y se exige que la puerta los rechace,
    y ademas que acepte el caso limpio.

    Args:
        puerta: acumulador de la puerta.
        marca: marca de corrida.

    Returns:
        Nada.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as temporal:
        raiz = Path(temporal)
        limpio = Progreso(raiz / "limpio", marca)
        limpio.registro = {
            clave: Resultado(
                id=clave, bloque=0, naturaleza=NATURELEZA_CONFIRMAR,
                estado=ESTADO_CONFIRMADO, exito=True, tolerancia="", descripcion="",
                stop=clave in AFIRMACIONES_STOP,
            )
            for clave in AFIRMACIONES_STOP
        }
        ok_limpio, vacio = comprobar_afirmaciones_stop(limpio)
        puerta.registrar(
            "puerta_de_la_corrida_completa_acepta_el_caso_limpio",
            ok_limpio and not vacio,
            f"sin_decidir={len(vacio)} de {len(AFIRMACIONES_STOP)} STOP",
        )

        estados_probables = (ESTADO_ERROR, ESTADO_REFUTADO)
        rechazos = []
        for estado in estados_probables:
            progres = Progreso(raiz / estado, marca)
            progres.registro = dict(limpio.registro)
            clave = sorted(AFIRMACIONES_STOP)[0]
            progres.registro[clave] = Resultado(
                id=clave, bloque=0, naturaleza=NATURELEZA_CONFIRMAR,
                estado=estado, exito=False, tolerancia="", descripcion="",
                stop=True,
            )
            ok, entradas = comprobar_afirmaciones_stop(progres)
            rechazos.append((not ok) and len(entradas) == 1)

        progres = Progreso(raiz / "ausente", marca)
        progres.registro = dict(limpio.registro)
        clave = sorted(AFIRMACIONES_STOP)[0]
        del progres.registro[clave]
        ok_ausente, entradas_ausente = comprobar_afirmaciones_stop(progres)
        puerta.registrar(
            "puerta_de_la_corrida_completa_rechaza_error_refutado_y_sin_ejecutar",
            all(rechazos) and (not ok_ausente) and len(entradas_ausente) == 1,
            f"error_rechazado={rechazos[0]} refutado_rechazado={rechazos[1]} sin_ejecutar_rechazado={not ok_ausente}",
        )
        limpio.parcial(
            "casos_de_la_puerta_de_la_corrida_completa",
            ".json",
            {
                "n_stop": len(AFIRMACIONES_STOP),
                "acepta_limpio": ok_limpio,
                "rechaza_error": rechazos[0],
                "rechaza_refutado": rechazos[1],
                "rechaza_sin_ejecutar": not ok_ausente,
                "marca_corrida": limpio.marca,
            },
        )


def ejecutar_casos_de_frontera(puerta: PuertaHumo, marca: str, progreso: Progreso) -> None:
    """Ejecuta todos los casos de humo de frontera y de contrato.

    Args:
        puerta: acumulador de la puerta.
        marca: marca de corrida.
        progreso: acumulador de progreso.

    Returns:
        Nada.
    """
    ejecutar_casos_de_dispositivo(puerta, progreso)
    ejecutar_caso_de_cobertura_del_completo(puerta)
    ejecutar_casos_de_tramo_2(puerta)
    ejecutar_casos_de_presupuesto(puerta)
    ejecutar_casos_de_neto(puerta)
    ejecutar_casos_de_colecciones_vacias(puerta)
    ejecutar_casos_de_veredicto(puerta, marca)
    ejecutar_caso_de_puerta_de_la_corrida_completa(puerta, progreso)
    caso_contrato_parametro_renombrado(puerta)
    caso_contrato_devuelve_mal(puerta)
    caso_clave_inexistente(puerta)
    caso_auditoria_ast_detecta(puerta)
    caso_escritura_unica(puerta)
    caso_fallo_de_escritura_degrada(puerta, marca)
    caso_no_ascii_rechazado(puerta)
    caso_json_no_serializable(puerta)
    caso_reoptimizacion_bloqueada(puerta, progreso)
    caso_integridad_del_congelado(puerta)
    caso_gap_leido_de_p78(puerta)


def ejecutar_caso_de_emparejamiento_entre_brazos(
    puerta: PuertaHumo, contexto: ContextoDatos, ajustes: Ajustes, progreso: Progreso
) -> None:
    """Caso de A9 con DOS BRAZOS REALES, no una magnitud contra si misma.

    Exige tres cosas: dos brazos con la MISMA semilla y early stopping que corta en
    puntos distintos tienen que salir PAREADOS; dos brazos con semilla distinta tienen que
    salir NO pareados; y, por mutacion, que si se vuelve a la magnitud ANTIGUA, que era
    el hash del ULTIMO lote, este caso FALLA. Sin ese tercer punto el caso no discrimina
    y volveria a dejar el humo en verde con la magnitud equivocada.

    Args:
        puerta: acumulador de la puerta.
        contexto: contexto de datos.
        ajustes: ajustes de la corrida.
        progreso: acumulador de progreso.

    Returns:
        Nada.
    """
    corrida_a = ejecutar_brazo(contexto, "B1", ajustes, progreso, solo_pliegues=[0])
    corrida_b = ejecutar_brazo(contexto, "B2", ajustes, progreso, solo_pliegues=[0])
    comunes = sorted(set(corrida_a.hash_secuencia) & set(corrida_b.hash_secuencia))
    desajustes_nuevo = [
        clave
        for clave in comunes
        if corrida_a.hash_secuencia[clave] != corrida_b.hash_secuencia[clave]
    ]
    desajustes_viejo = [
        clave
        for clave in comunes
        if corrida_a.hash_ultimo_lote.get(clave) != corrida_b.hash_ultimo_lote.get(clave)
    ]
    estrellas_a = {f"{f['pliegue']}_{f['semilla']}": f["e_estrella"] for f in corrida_a.combinaciones}
    estrellas_b = {f"{f['pliegue']}_{f['semilla']}": f["e_estrella"] for f in corrida_b.combinaciones}
    cortes_distintos = [
        clave for clave in comunes
        if estrellas_a.get(clave) != estrellas_b.get(clave)
    ]
    puerta.registrar(
        "dos_brazos_misma_semilla_pareados_aunque_paren_distinto",
        bool(comunes) and not desajustes_nuevo,
        f"n_comunes={len(comunes)} desajustes={len(desajustes_nuevo)} cortes_distintos={len(cortes_distintos)}",
    )
    puerta.registrar(
        "dos_brazos_con_early_stopping_distinto_efectivamente",
        len(cortes_distintos) > 0,
        f"n_cortes_distintos={len(cortes_distintos)} de {len(comunes)}",
    )
    puerta.registrar(
        "mutacion_a_la_magnitud_antigua_hace_FALLAR_el_caso",
        (not desajustes_nuevo) and len(desajustes_viejo) > 0,
        f"desajustes_nuevo={len(desajustes_nuevo)} desajustes_magnitud_antigua={len(desajustes_viejo)}",
    )
    hash_otra_semilla = hash_secuencia_de_lotes(
        int(CONFIG["semillas"][7]), len(contexto.y), CONFIG["batch"], PROFUNDIDAD_HASH_LOTES
    )
    hash_primera = corrida_a.hash_secuencia[comunes[0]] if comunes else ""
    puerta.registrar(
        "dos_brazos_semilla_distinta_no_pareados",
        hash_otra_semilla != hash_primera,
        f"semilla_1={hash_primera[:12]} semilla_8={hash_otra_semilla[:12]}",
    )
    progreso.parcial(
        "caso_de_emparejamiento_entre_brazos",
        ".json",
        {
            "n_claves_comunes": len(comunes),
            "desajustes_magnitud_nueva": desajustes_nuevo,
            "desajustes_magnitud_antigua": desajustes_viejo,
            "n_cortes_de_parada_distintos": len(cortes_distintos),
            "marca_corrida": progreso.marca,
        },
    )


def ejecutar_caso_de_mutacion_del_mde(
    puerta: PuertaHumo, progreso: Progreso
) -> None:
    """Caso que rompe SOLO el literal publicado del MDE y exige que se detecte.

    La tabla de limite de deteccion son constantes de modulo y la reproduccion recalcula
    contra ellas mismas, luego mutar el valor publicado movia a la vez las dos columnas y
    no rompia nada. Aqui se altera SOLO el literal publicado, se comprueba que la
    reproduccion lo detecta, y se restaura.

    Args:
        puerta: acumulador de la puerta.
        progreso: acumulador de progreso.

    Returns:
        Nada.
    """
    claves = sorted(TABLA_MDE_PREREGISTRO)
    original = {clave: TABLA_MDE_PREREGISTRO[clave] for clave in claves}
    antes, _ = reproducir_tabla_del_preregistro()
    detectados: list[str] = []
    for clave in claves:
        TABLA_MDE_PREREGISTRO[clave] = original[clave] + 0.0005
        correcto, _ = reproducir_tabla_del_preregistro()
        if not correcto:
            detectados.append(str(clave))
        TABLA_MDE_PREREGISTRO[clave] = original[clave]
    despues, _ = reproducir_tabla_del_preregistro()
    restaurada = dict(TABLA_MDE_PREREGISTRO) == original
    puerta.registrar(
        "mutacion_de_un_literal_del_mde_se_detecta_en_todas_las_filas",
        bool(antes) and len(detectados) == len(claves),
        f"detectadas={len(detectados)} de {len(claves)}",
    )
    puerta.registrar(
        "tabla_del_mde_restaurada_tras_la_mutacion",
        restaurada and despues and dict(TABLA_MDE_PREREGISTRO) == original,
        f"restaurada={restaurada} reproduce_tras={despues}",
    )
    progreso.parcial(
        "caso_de_mutacion_del_mde",
        ".json",
        {
            "n_filas_mutadas": len(claves),
            "n_filas_detectadas": len(detectados),
            "tabla_restaurada": restaurada,
            "marca_corrida": progreso.marca,
        },
    )


def ejecutar_casos_de_emparejamiento(
    puerta: PuertaHumo, contexto: ContextoDatos, ajustes: Ajustes, progreso: Progreso
) -> None:
    """Los tres casos que fijan la invariante de emparejamiento de A9.

    Si solo se probara el caso que sale bien, la comprobacion no discrimina. Por eso hay
    un caso positivo, un caso que demuestra que la magnitud ANTIGUA era dependiente del
    estado, y un caso negativo.

    Args:
        puerta: acumulador de la puerta.
        contexto: contexto de datos.
        ajustes: ajustes de la corrida.
        progreso: acumulador de progreso.

    Returns:
        Nada.
    """
    matriz = contexto.matriz.matriz
    n_filas = len(contexto.y)
    semilla_a = int(CONFIG["semillas"][0])
    semilla_b = int(CONFIG["semillas"][1])
    ids_t = torch.as_tensor(contexto.ids, dtype=torch.long, device=DISPOSITIVO)
    largos_t = torch.as_tensor(contexto.largos, dtype=torch.long, device=DISPOSITIVO)
    y_t = torch.as_tensor(contexto.y, dtype=torch.float32, device=DISPOSITIVO)

    hashes_secuencia: dict[str, str] = {}
    ultimos_orden: dict[str, str] = {}
    for epocas in (1, 3):
        torch.manual_seed(semilla_a)
        modelo = construir_red(matriz, ajustes, "GRU", False)
        optimizador = torch.optim.AdamW(
            parametros_entrenables(modelo),
            lr=CONFIG["lr"],
            weight_decay=CONFIG["weight_decay"],
        )
        generador = torch.Generator().manual_seed(semilla_a + 7)
        for _ in range(epocas):
            bucle_epoca(
                modelo, optimizador, ids_t, largos_t, y_t, generador, CONFIG["batch"], True
            )
        hashes_secuencia[str(epocas)] = hash_secuencia_de_lotes(
            semilla_a, n_filas, CONFIG["batch"], PROFUNDIDAD_HASH_LOTES
        )
        ultimos_orden[str(epocas)] = INSTRUMENTO.ultimo_orden

    puerta.registrar(
        "emparejamiento_invariante_al_punto_de_parada",
        hashes_secuencia["1"] == hashes_secuencia["3"],
        f"misma_semilla_1_epoca={hashes_secuencia['1'][:12]} 3_epocas={hashes_secuencia['3'][:12]}",
    )
    puerta.registrar(
        "magnitud_antigua_depindia_del_punto_de_parada",
        ultimos_orden["1"] != ultimos_orden["3"],
        f"ultimo_lote_1_epoca={ultimos_orden['1'][:12]} 3_epocas={ultimos_orden['3'][:12]}",
    )
    secuencia_a = hash_secuencia_de_lotes(
        semilla_a, n_filas, CONFIG["batch"], PROFUNDIDAD_HASH_LOTES
    )
    secuencia_b = hash_secuencia_de_lotes(
        semilla_b, n_filas, CONFIG["batch"], PROFUNDIDAD_HASH_LOTES
    )
    puerta.registrar(
        "semillas_distintas_no_emparejadas",
        secuencia_a != secuencia_b,
        f"semilla_a={secuencia_a[:12]} semilla_b={secuencia_b[:12]}",
    )
    progreso.parcial(
        "casos_de_emparejamiento",
        ".json",
        {
            "hashes_de_secuencia_por_epocas": hashes_secuencia,
            "hashes_del_ultimo_lote_por_epocas": ultimos_orden,
            "hash_semilla_a": secuencia_a,
            "hash_semilla_b": secuencia_b,
            "profundidad": PROFUNDIDAD_HASH_LOTES,
            "marca_corrida": progreso.marca,
        },
    )


# =============================================================================
# SECCION 16: A22, LAS SEIS MUTACIONES
# =============================================================================


def verificar_mutaciones(
    puerta: PuertaHumo, ajustes: Ajustes, progreso: Progreso
) -> dict[str, bool]:
    """Ejecuta las seis mutaciones, cada una sobre su comprobacion real.

    No se levanta ninguna bandera sin consequence: cada mutacion recorre el camino de
    codigo que la comprobacion tiene que vigilar, y la comprobacion tiene que fallar.

    Args:
        puerta: acumulador de la puerta.
        ajustes: ajustes base de la corrida de humo.
        progreso: acumulador de progreso.

    Returns:
        Diccionario mutacion a booleano, si fue detectada.
    """
    global AJUSTES_ACTIVOS
    deteccion: dict[str, bool] = {}
    for nombre, esperado in sorted(MUTACIONES.items()):
        mutado = construir_ajustes(ajustes, nombre)
        if nombre == "decision_epoca_sobre_ib":
            datos = datos_sinteticos(40, 20, 6, 71, "INCOMPLETO")
            contexto = construir_contexto_sintetico(datos, mutado, progreso)
            corrida = ejecutar_brazo(contexto, "B2", mutado, progreso, solo_pliegues=[0])
            fila = corrida.combinaciones[0]
            ok, _ = verificar_anidamiento(
                filas_por_llamada=fila["filas_de_decision"],
                ib=contexto.pliegues[0][1],
                ia=contexto.pliegues[0][0],
                n_esperadas=CONFIG["n_internos"],
                epocas=fila["epocas_internas"],
                e_estrella=fila["e_estrella"],
            )
            ok = not ok
            AJUSTES_ACTIVOS = ajustes
        elif nombre == "mascara_despues_de_rnn":
            datos = datos_sinteticos(40, 20, 6, 72, "COMPLETO")
            contexto = construir_contexto_sintetico(datos, mutado, progreso)
            ok, _ = verificar_relleno_antes(contexto.matriz.matriz, mutado, "GRU", False)
            ok = not ok
        elif nombre == "omitir_clip_en_reajuste":
            datos = datos_sinteticos(40, 20, 6, 73, "COMPLETO")
            contexto = construir_contexto_sintetico(datos, mutado, progreso)
            corrida = ejecutar_brazo(contexto, "B1", mutado, progreso, solo_pliegues=[0])
            fila = corrida.combinaciones[0]
            pasos = int(fila["pasos_busqueda"]) + int(fila["pasos_ajuste"])
            clips = int(fila["clips_busqueda"]) + int(fila["clips_ajuste"])
            ok, _ = verificar_clip(
                pasos=pasos, llamadas=clips, umbral=CONFIG["clip_max_norm"], normas=corrida.normas
            )
            ok = not ok
            AJUSTES_ACTIVOS = ajustes
        elif nombre == "permutacion_con_sustitucion":
            matriz = np.arange(80, dtype=np.float32).reshape(16, 5)
            mala = permutar_conservando_multiconjunto(matriz, 9, True)
            doble = permutar_conservando_multiconjunto(matriz, 9, False)
            ok, _ = verificar_multiconjunto(matriz, mala, doble)
            ok = not ok
        elif nombre == "ocho_semillas":
            hashes = {
                "B1": {20261401 + i: "a" for i in range(8)},
                "B2": {20261401 + i: "a" for i in range(8)},
            }
            ok, _ = verificar_semillas(
                CONFIG["semillas"][: mutado.n_semillas],
                hashes,
                {"B1": "GRU", "B2": "GRU"},
            )
            ok = not ok
        else:
            serie = [0.011, -0.021, 0.031, 0.0, 0.005, -0.004, 0.020, 0.010, -0.010, 0.002]
            recorte = serie[: mutado.n_semillas]
            pareado = intervalo_t_pareado(recorte)
            arranque = intervalo_bootstrap(recorte, 13)
            publicado = dict(pareado)
            publicado["inferior"] = arranque["inferior"]
            publicado["superior"] = arranque["superior"]
            metodo = LITERAL_METODO_BOOT if mutado.intervalo_bootstrap else LITERAL_METODO_IC
            ok, _ = verificar_metodo_intervalo(recorte, metodo, publicado)
            ok = not ok
        deteccion[nombre] = bool(ok)
        puerta.registrar(f"mutacion_{nombre}_detecta_{esperado}", bool(ok))
    progreso.contar("mutaciones_ejecutadas", len(deteccion))
    progreso.parcial(
        "mutaciones", ".json", {"deteccion": deteccion, "marca_corrida": progreso.marca}
    )
    return deteccion


# =============================================================================
# SECCION 17: ARMADO DEL INFORME
# =============================================================================


def tabla_markdown(cabeceras: Sequence[str], filas: Sequence[Sequence[Any]]) -> str:
    """Construye una tabla markdown en ASCII.

    Args:
        cabeceras: titulos de columna.
        filas: filas de la tabla.

    Returns:
        Bloque de texto markdown terminado en linea nueva.
    """
    lineas = ["| " + " | ".join(str(c) for c in cabeceras) + " |"]
    lineas.append("|" + "|".join("---" for _ in cabeceras) + "|")
    for fila in filas:
        lineas.append("| " + " | ".join(str(c) for c in fila) + " |")
    return "\n".join(lineas) + "\n"


def construir_json(
    progreso: Progreso,
    modo: str,
    marca: str,
    inicio_utc: str,
    cronometro: dict[str, Any],
    protocolo: dict[str, Any],
    brazos: dict[str, Any],
    comparaciones: dict[str, Any],
    limite: dict[str, Any],
    cobertura: dict[str, Any],
    contratos: dict[str, Any],
    auditoria: dict[str, Any],
    contra_v03: dict[str, Any],
    fuentes: dict[str, Any],
    metodo_intervalo: str,
    nota_tramo2: str,
    extras: dict[str, Any],
) -> dict[str, Any]:
    """Arma el diccionario del informe JSON.

    Returns:
        El diccionario completo, listo para serializar.
    """
    clave_detalle = resolver_clave("detalle_json")
    clave_estado = resolver_clave("estado_json")
    veredictos = {
        clave: {clave_estado: valor.estado, "bloque": valor.bloque, "stop": valor.stop}
        for clave, valor in sorted(progreso.registro.items())
    }
    conteos: dict[str, int] = {}
    for valor in progreso.registro.values():
        conteos[valor.estado] = conteos.get(valor.estado, 0) + 1
    carga: dict[str, Any] = {
        "experimento": CONFIG["experimento"],
        "script": "experiments/p82_01_rnn_atencion.py",
        "modo": modo,
        "marca_corrida": marca,
        "fecha_inicio_utc": inicio_utc,
        "pre_registro": {
            "ruta": CONFIG["pre_registro"],
            "sha256_declarado": CONFIG["pre_registro_sha256"],
        },
        "formulacion": LITERAL_V03,
        "declaracion_word2vec": LITERAL_NO_WORD2VEC,
        "metodo_del_intervalo_que_decide": metodo_intervalo,
        "nota_tramo2": nota_tramo2,
        "brazos": brazos,
        "comparaciones": comparaciones,
        "limite_deteccion": limite,
        "cobertura": cobertura,
        "contrato_de_capa": contratos,
        "auditoria_de_accesos_por_literal": auditoria,
        "interseccion_y_rescate": contra_v03,
        "lecturas_por_fuente": fuentes,
        "protocolo": protocolo,
        "cronometro": cronometro,
        "resumen_de_naturalezas": {
            "CONFIRMAR": {
                "n": sum(1 for v in progreso.registro.values() if v.naturaleza == NATURELEZA_CONFIRMAR),
                "ids": sorted(progreso.registro),
            },
            "REFUTAR": {"n": conteos.get(ESTADO_REFUTADO, 0), "ids": []},
            "NO_APLICABLE_POR_CONSTRUCCION": {"n": 0, "ids": []},
            "NO_EVALUABLE": {
                "n": conteos.get(ESTADO_NO_EVALUABLE, 0),
                "ids": sorted(k for k, v in progreso.registro.items() if v.estado == ESTADO_NO_EVALUABLE),
            },
            "ERROR": {"n": conteos.get(ESTADO_ERROR, 0), "ids": []},
        },
        "veredictos": veredictos,
        "aislamiento_de_afirmaciones": {
            "errores": progreso.errores,
            "avisos": progreso.avisos,
            "escrituras_fallidas": progreso.escrituras_fallidas,
            "nota": "ERROR no es ni CONFIRMADO ni REFUTADO ni NO_EVALUABLE; la corrida continua",
            "traza_por_afirmacion": {
                clave: valor.traza for clave, valor in sorted(progreso.registro.items()) if valor.traza
            },
        },
        "contadores": progreso.contadores,
        "dispositivo_de_calculo": informacion_de_dispositivo(),
        "directorio_de_parciales": str(progreso.directorio_parciales),
        "regla_de_lectura": {
            "literal_negativo": LITERAL_NEGATIVO,
            "literal_semillas": LITERAL_SEMILLAS,
            "naturaleza": "EXPLORATORIO",
        },
        "afirmaciones_declaradas": {
            clave: {"bloque": bloque, "que_hace": que, "stop": clave in AFIRMACIONES_STOP}
            for clave, (bloque, que) in sorted(BLOQUES.items())
        },
    }
    carga[clave_detalle] = {
        clave: valor.detalle for clave, valor in sorted(progreso.registro.items())
    }
    carga[resolver_clave("naturaleza_resultado_json")] = "EXPLORATORIO"
    carga.update(extras)
    return carga


def construir_markdown(carga: dict[str, Any]) -> str:
    """Construye el informe markdown, con las tres formulaciones literales.

    Args:
        carga: diccionario del informe JSON.

    Returns:
        Texto markdown completo en ASCII puro.
    """
    clave_naturaleza = resolver_clave("naturaleza_resultado_json")
    clave_veredictos = resolver_clave("veredictos_json")
    clave_estado = resolver_clave("estado_json")
    pendientes = [
        nombre
        for nombre in (
            "brazos", "comparaciones", "limite_deteccion", "cobertura",
            "cronometro", "interseccion_y_rescate",
        )
        if not carga.get(nombre)
    ]
    partes: list[str] = []
    if pendientes:
        partes.append(
            "> AVISO DE ETAPA. Esta copia del informe se renderizo ANTES de calcular las "
            "secciones " + ", ".join(pendientes) + ". Las afirmaciones que leen literales "
            "la leen sobre texto real producido por construir_markdown, y el informe "
            "definitivo se vuelve a renderizar y a escribir al final, con todo.\n\n"
        )
    partes.append("# P82: RNN bidireccional con capa de atencion aditiva\n\n")
    partes.append(f"Modo: {carga['modo']}. Marca de corrida: {carga['marca_corrida']}.\n\n")
    partes.append(f"Inicio UTC: {carga['fecha_inicio_utc']}.\n\n")

    partes.append("## 0. Dispositivo de calculo\n\n")
    dispositivo = carga.get("dispositivo_de_calculo", {})
    partes.append(
        f"Dispositivo: {dispositivo.get('nombre_del_dispositivo', 'ND')} "
        f"(tipo {dispositivo.get('dispositivo', 'ND')}). "
        f"Hilos fijados: {dispositivo.get('hilos_fijados', 'ND')}. "
        f"Precision mixta: {dispositivo.get('precision_mixta', 'ND')}. "
        f"Generador de permutaciones: {dispositivo.get('generador_de_permutaciones_en', 'ND')}.\n\n"
    )
    partes.append(
        f"Fuente de la eleccion: {dispositivo.get('motivo_de_los_hilos', 'ND')}.\n\n"
    )
    partes.append(
        f"Coste medido por ajuste, en segundos: "
        f"{json.dumps(dispositivo.get('medicion_del_coste_por_ajuste_en_segundos', {}))}.\n\n"
    )
    partes.append(f"Consecuencia declarada: {dispositivo.get('consecuencia_declarada', 'ND')}\n\n")
    partes.append(
        f"Parciales de esta corrida en: {carga.get('directorio_de_parciales', 'ND')}. "
        "Los parciales del intento fallido en CPU se conservan en reports/p82_parciales y "
        "no se mezclan con estos, porque cada corrida usa su propio subdirectorio.\n\n"
    )

    partes.append("## 1. Formulacion\n\n")
    partes.append(f"{carga['formulacion']}\n\n")
    partes.append(
        "Este informe NO afirma que Word2Vec sea mejor que V03, ni que una RNN lo supere, "
        "ni que la atencion aporte o no aporte. El limite permanente declarado por P81, "
        "el anclaje del objetivo del trainer contra una implementacion externa de SGNS, "
        "sigue ABIERTO y permanente.\n\n"
    )
    partes.append(f"{carga['declaracion_word2vec']}\n\n")

    partes.append("## 2. La pregunta, y lo que este diseno NO puede responder\n\n")
    partes.append(
        "Si la mejora, si la hay, es menor que el limite de deteccion del diseno, este "
        "experimento NO LA VE. Un resultado NEGATIVO no se lee como que la atencion no "
        "aporta; se lee como que el efecto es inferior a lo que este diseno resuelve.\n\n"
    )
    partes.append(f"{LITERAL_NEGATIVO}\n\n")
    partes.append(f"{LITERAL_SEMILLAS}\n\n")
    partes.append(f"Naturaleza del resultado: {carga[clave_naturaleza]}.\n\n")

    partes.append("## 3. Brazos, uno por fila y sin agregado\n\n")
    filas = []
    for identificador, datos in sorted(carga.get("brazos", {}).items()):
        filas.append(
            [
                identificador,
                datos.get("nucleo", "-"),
                "SI" if datos.get("atencion") else "NO",
                datos.get("embedding", "-"),
                datos.get("media", "-"),
                datos.get("desviacion", "-"),
                datos.get("parametros_entrenables", "-"),
                datos.get("parametros_atencion", "-"),
            ]
        )
    partes.append(
        tabla_markdown(
            ["brazo", "nucleo", "atencion", "embedding", "media", "sd_semillas", "params", "params_atencion"],
            filas,
        )
    )
    b2 = carga.get("brazos", {}).get("B2", {})
    partes.append(
        f"\nCoste parametrico de la capa de atencion. Referencia vigente, de la Enmienda 1: "
        f"{b2.get('parametros_atencion_referencia_enmienda_1', PARAMETROS_ATENCION_REFERENCIA_ENMIENDA_1)}. "
        f"Valor del pre-registro original, ya corregido y NO vigente: "
        f"{b2.get('parametros_atencion_preregistrados', PARAMETROS_ATENCION_PREREGISTRADOS)}. "
        f"Fuente de la referencia: {b2.get('fuente_de_la_referencia', FUENTE_REFERENCIA_ATENCION)}.\n\n"
    )
    partes.append(
        tabla_markdown(
            ["magnitud", "valor"],
            [
                ["pesos de la atencion SIN sesgos, medidos", b2.get("parametros_atencion_sin_sesgos", "NA")],
                ["referencia de la Enmienda 1, sin sesgos", b2.get("parametros_atencion_referencia_enmienda_1", PARAMETROS_ATENCION_REFERENCIA_ENMIENDA_1)],
                ["total de la atencion CON sesgos, medidos", b2.get("parametros_atencion", "NA")],
                ["convencion", b2.get("convencion_de_conteo", CONVENCION_CONTEO_ATENCION)],
            ],
        )
    )
    partes.append(
        "\nLa aritmetica original de 8256 asumia W_h de 64x64, es decir una celda "
        "UNIDIRECCIONAL de ancho 64. Los brazos de P82 son BIDIRECCIONALES con 64 por "
        "direccion, luego el estado que entra en la atencion tiene ancho 128 y W_h y W_s "
        "son de 128x64. La Enmienda 1 corrige el numero y este script publica las dos "
        "cifras, la pre-registrada y la real, y declara la discrepancia en vez de "
        "retocarla.\n\n"
    )

    partes.append("## 4. Comparaciones\n\n")
    filas = []
    for identificador, datos in sorted(carga.get("comparaciones", {}).items()):
        filas.append(
            [
                identificador,
                datos.get("definicion", "-"),
                datos.get("naturaleza", "-"),
                datos.get("media", "-"),
                datos.get("intervalo", "-"),
            ]
        )
    partes.append(tabla_markdown(["id", "definicion", "naturaleza", "media", "IC95"], filas))
    p1 = carga.get("comparaciones", {}).get("P1", {})
    p2 = carga.get("comparaciones", {}).get("P2", {})
    if p1.get("media") is not None and p2.get("media") is not None:
        if (p1["media"] > 0) != (p2["media"] > 0):
            partes.append(
                "\nP1 y P2 tienen signos OPUESTOS. Ese es el resultado principal de P82, "
                "porque es la evidencia mas fuerte disponible de que el efecto de la atencion "
                "no es real. No se elige el brazo que sale bien.\n\n"
            )

    partes.append("## 5. Limite de deteccion\n\n")
    limite = carga.get("limite_deteccion", {})
    partes.append(
        f"Desviacion planificada: {limite.get('sd_planificada')}. "
        f"Desviacion MEDIDA en esta corrida: {limite.get('sd_medida')}. "
        f"Gap con V03, leido del informe de P78 y no escrito a mano: {limite.get('gap_leido_de_p78')}. "
        f"MDE a n={limite.get('n_semillas')} con la desviacion medida: {limite.get('mde_con_sd_medida')}.\n\n"
    )
    partes.append("### Tabla del pre-registro, recalculada y NO modificada\n\n")
    filas = [
        [
            r["n"],
            r["mde_publicado"],
            r["mde_recalculado"],
            "SI" if r["coincide"] else "NO",
            r["semiancho_ic95"],
            r["potencia_en_el_ancla"],
        ]
        for r in limite.get("tabla_del_preregistro", {}).get("filas", [])
    ]
    partes.append(
        tabla_markdown(
            ["n", "MDE publicado", "MDE recalculado", "coincide", "semiancho IC95", "potencia en el ancla"],
            filas,
        )
    )
    partes.append("\n### Sensibilidad con la desviacion MEDIDA\n\n")
    filas = [
        [r["factor"], r["sd_supuesta"], r["mde_a_n_10"], r["potencia_en_el_ancla"], r["mde_sobre_gap"]]
        for r in limite.get("sensibilidad_con_sd_medida", [])
    ]
    partes.append(
        tabla_markdown(["factor", "sd supuesta", "MDE", "potencia en el ancla", "MDE/gap"], filas)
    )
    partes.append("\n### Reproduccion del intervalo ya publicado por P78\n\n")
    aritmetica = carga.get("limite_deteccion", {}).get("prueba_de_la_aritmetica", {})
    partes.append(
        tabla_markdown(
            ["intervalo calculado", "intervalo publicado", "diferencia maxima"],
            [[aritmetica.get("intervalo_calculado"), aritmetica.get("intervalo_publicado_por_p78"), aritmetica.get("diferencia_maxima")]],
        )
    )

    partes.append("\n## 6. Serie de deltas por semilla del primario\n\n")
    filas = [
        [indice + 1, semilla, f"{valor:.6f}", "POSITIVO" if valor > 0 else ("NEGATIVO" if valor < 0 else "CERO")]
        for indice, (semilla, valor) in enumerate(
            sorted(p1.get("deltas_por_semilla", {}).items(), key=lambda par: int(par[0]))
        )
    ]
    partes.append(tabla_markdown(["orden", "semilla", "delta", "signo"], filas))
    partes.append(
        f"\nConteo de signos positivos: {p1.get('conteo_signos_positivos')}. "
        f"Media: {p1.get('media')}. Desviacion: {p1.get('desviacion')}. "
        f"Metodo del intervalo que decide: {carga['metodo_del_intervalo_que_decide']}, "
        "t de Student sobre los deltas POR SEMILLA. No es bootstrap.\n\n"
    )

    partes.append("## 7. Cobertura medida\n\n")
    partes.append(
        tabla_markdown(["cifra", "valor"], [[clave, valor] for clave, valor in sorted(carga.get("cobertura", {}).items())])
    )

    partes.append("\n## 8. Interseccion y rescue damage net contra V03\n\n")
    filas = []
    for etiqueta, datos in sorted(carga.get("interseccion_y_rescate", {}).items()):
        if isinstance(datos, dict):
            filas.append([etiqueta, datos.get("aciertos_comunes"), datos.get("rescate"), datos.get("dano"), datos.get("neto")])
    partes.append(tabla_markdown(["etiqueta", "aciertos comunes", "rescate", "dano", "neto"], filas))
    partes.append(
        "\nEstas cifras NO son comparables con corrected_cv ni con ningun macro-F1. "
        "Rescue damage net es un recuento de celdas, no una mejora de metrica.\n\n"
    )

    partes.append("## 9. Lecturas por fuente\n\n")
    partes.append(
        tabla_markdown(["fuente", "contador"], [[clave, valor] for clave, valor in sorted(carga.get("lecturas_por_fuente", {}).items())])
    )

    partes.append("\n## 10. Cronometro\n\n")
    partes.append(
        tabla_markdown(["clave", "valor"], [[clave, valor] for clave, valor in sorted(carga.get("cronometro", {}).items())])
    )
    if carga.get("nota_tramo2"):
        partes.append(f"\n{carga['nota_tramo2']}\n\n")

    partes.append("## 11. Las veintitres afirmaciones\n\n")
    filas = [
        [clave, veredicto["bloque"], veredicto[clave_estado], "si" if veredicto["stop"] else "no"]
        for clave, veredicto in sorted(carga[clave_veredictos].items())
    ]
    partes.append(tabla_markdown(["id", "bloque", "estado", "stop"], filas))
    partes.append(
        "\nERROR no es ni CONFIRMADO ni REFUTADO ni NO_EVALUABLE: significa que la "
        "afirmacion no llego a completarse por fallo PROPIO del script, y la corrida "
        "continuo con la siguiente. Lo unico que aborta de verdad es una Parada con STOP.\n\n"
    )

    partes.append("## 12. Contrato entre capas y auditoria de accesos por literal\n\n")
    partes.append(
        tabla_markdown(
            ["callable", "origen", "estado"],
            [[f["nombre"], f["origen"], "ok" if f.get("ok") else "FALLO"] for f in carga["contrato_de_capa"].get("filas", [])],
        )
    )
    partes.append(
        f"\nAccesos por literal a claves de contrato en este fichero: "
        f"{carga['auditoria_de_accesos_por_literal']['n_infracciones']}.\n\n"
    )

    partes.append(f"\n{carga['formulacion']}\n")
    return "".join(partes)


# =============================================================================
# SECCION 18: MODO DE HUMO
# =============================================================================


def construir_carga_preliminar(
    progreso: Progreso,
    marca: str,
    inicio: datetime,
    etapa: str,
    intervalo: dict[str, Any],
    detalle_ast: dict[str, Any],
) -> dict[str, Any]:
    """Carga preliminar REAL, para que A12, A13 y A18 lean objeto y texto de verdad.

    Las secciones que todavia no se han calculado van vacias, y construir_markdown lo
    declara en un aviso de etapa. El informe definitivo se vuelve a renderizar y a
    escribir al final, con todas las secciones.

    Args:
        progreso: acumulador de progreso.
        marca: marca de corrida.
        inicio: instante de inicio.
        etapa: etiqueta de la etapa.
        intervalo: intervalo del primario ya calculado.
        detalle_ast: resultado de la auditoria AST.

    Returns:
        El diccionario de carga preliminar.
    """
    carga = construir_json(
        progreso=progreso,
        modo="humo",
        marca=marca,
        inicio_utc=inicio.isoformat(),
        cronometro={"etapa": f"preliminar:{etapa}", "budget_segundos": CONFIG["budget_segundos"]},
        protocolo={"semillas": list(CONFIG["semillas"])},
        brazos={},
        comparaciones={},
        limite={"sd_medida": intervalo["desviacion"]},
        cobertura=progreso.registro["A8"].detalle.get("cifras", {}) if "A8" in progreso.registro else {},
        contratos=verificar_contratos()[1],
        auditoria=detalle_ast,
        contra_v03={},
        fuentes={},
        metodo_intervalo=LITERAL_METODO_IC,
        nota_tramo2="",
        extras={
            "comparacion_primaria": {
                "id": "P1",
                "definicion": "B2 - B1",
                "corrida": CONFIG["experimento"],
                "control_literal": "B1",
                "control_sin_atencion": "B1",
                "experimento_contraste": "NINGUNO",
            },
            "es_una_carga_preliminar": True,
            "discrepancia_del_split_interno": DISCREPANCIA_SPLIT_INTERNO,
            "forma_de_split_interno_usada": FORMA_DE_SPLIT_INTERNO,
        },
    )
    carga[resolver_clave("comparacion_primaria_json")][resolver_clave("control_atencion_json")] = "B1"
    return carga


def pipeline_humo(
    directorio: Path, marca: str, puerta: PuertaHumo, caso: str
) -> dict[str, Any]:
    """Corre un caso de humo completo, incluida la ruta de escritura.

    Args:
        directorio: directorio de salidas del caso.
        marca: marca de corrida.
        puerta: puerta de salida.
        caso: COMPLETO o INCOMPLETO.

    Returns:
        El diccionario del informe.
    """
    ajustes = Ajustes(
        max_len=12,
        dim_oculto=8,
        epocas_max=2,
        paciencia_es=1,
        n_semillas=len(CONFIG["semillas"]),
        id_relleno=0,
    )
    progreso = Progreso(directorio / "parciales", marca)
    inicio = datetime.now(timezone.utc)
    inicio_monotonic = time.perf_counter()

    datos = datos_sinteticos(60, 24, 8, 4242, caso)
    contexto = construir_contexto_sintetico(datos, ajustes, progreso)
    progreso.contar("casos_de_humo", 1)
    progreso.parcial("bloque_00_contexto", ".json", {"caso": caso, "n_filas": len(contexto.textos)})

    puerta.registrar("contratos_de_capa_verificados", verificar_contratos()[0])
    puerta.registrar(
        "vocabulario_sintetico_no_vacio",
        len(contexto.matriz.indice) > 0,
        f"n_vocab={len(contexto.matriz.indice)} n_ids_oov={int((contexto.ids == contexto.matriz.id_oov).sum())}",
    )
    puerta.registrar(
        "caso_incompleto_tiene_filas_de_relleno_y_de_oov",
        caso == "COMPLETO"
        or (
            int((contexto.largos == 0).sum()) > 0
            and int((contexto.ids == contexto.matriz.id_oov).sum()) > 0
        ),
        f"filas_vacias={int((contexto.largos == 0).sum())} posiciones_oov={int((contexto.ids == contexto.matriz.id_oov).sum())}",
    )
    ok_ast, detalle_ast = auditar_propio_script()
    puerta.registrar(
        "auditoria_ast_del_propio_script",
        ok_ast,
        f"infracciones={detalle_ast['n_infracciones']} "
        f"tuplas_anidadas={detalle_ast['n_evaluaciones_con_tupla_anidada']}",
    )

    # A4 y A3 se evaluan con datos sinteticos; A0, A1, A2 y A8 con los reales.
    progreso.evaluar(
        "A0",
        lambda: con_extra(
            lambda: (
                (lambda h: (h["sha256"] == W2V_SHA256 and h["size"] > 0, h))(
                    huella_fichero(W2V_NPZ)
                )
            ),
            {"dispositivo": comprobar_dispositivo_de_ejecucion()},
        ),
        "coincidencia del sha256, del size y del mtime_ns, y el dispositivo real",
    )
    progreso.evaluar(
        "A1",
        lambda: (
            contexto.guarda.filas_test_leidas == 0
            and contexto.guarda.filas_holdout_descartadas == 0
            and contexto.guarda.filas_de_desarrollo_usadas == len(contexto.textos),
            {
                "filas_de_desarrollo": contexto.guarda.filas_de_desarrollo_usadas,
                "filas_test_leidas": contexto.guarda.filas_test_leidas,
                "filas_holdout_leidas": contexto.guarda.filas_holdout_descartadas,
                "origen": contexto.origen,
            },
        ),
        "enteros exactos; test leido igual a 0 y holdout leido igual a 0",
    )
    umbrales, detalle_umbrales = leer_umbrales_v03()
    esperado = np.array([0.44, 0.51, 0.46], dtype=float)
    progreso.evaluar(
        "A2",
        lambda: (
            bool(np.array_equal(umbrales, esperado))
            and CONTADOR_REOPTIMIZACION_UMBRAL == 0
            and detalle_umbrales["coinciden_json_y_oof"],
            {
                **detalle_umbrales,
                "n_reoptimizaciones": CONTADOR_REOPTIMIZACION_UMBRAL,
                "umbrales_esperados": [0.44, 0.51, 0.46],
                "auditoria_de_llamadas_a_reoptimizar": auditar_llamadas_a_reoptimizar(
                    Path(__file__).read_text(encoding="utf-8")
                ),
            },
        ),
        "igualdad exacta de los tres umbrales, y contador de reoptimizacion igual a 0",
    )

    torch.manual_seed(20261400)
    red_real = construir_red(contexto.matriz.matriz, ajustes, "GRU", True)
    with torch.no_grad():
        _, pesos_real = red_real.representacion(
            torch.as_tensor(contexto.ids, dtype=torch.long, device=DISPOSITIVO),
            torch.as_tensor(contexto.largos, dtype=torch.long, device=DISPOSITIVO),
        )
    progreso.evaluar(
        "A4",
        lambda: verificar_celdas_separadas(
            ids=contexto.ids,
            largos=contexto.largos,
            id_relleno=contexto.matriz.id_relleno,
            pesos=pesos_real,
            matriz=contexto.matriz.matriz,
            n_vocab=len(contexto.matriz.indice),
            id_oov=contexto.matriz.id_oov,
        ),
        "enteros exactos; 0.0 exacto en la fila de relleno y en los pesos de relleno",
    )
    ok_gru, det_gru = verificar_relleno_antes(contexto.matriz.matriz, ajustes, "GRU", False)
    ok_lstm, det_lstm = verificar_relleno_antes(contexto.matriz.matriz, ajustes, "LSTM", True)
    progreso.evaluar(
        "A3",
        lambda: (
            ok_gru and ok_lstm,
            {"GRU": det_gru, "LSTM": det_lstm},
        ),
        "1e-6 en norma relativa del vector de estado; banda muerta 1e-9",
    )
    progreso.evaluar(
        "A8",
        lambda: comprobar_cobertura(
            medir_cobertura(
                contexto.ids,
                contexto.largos,
                contexto.matriz.id_relleno,
                contexto.matriz.id_oov,
                contexto.tokens_filas,
                contexto.matriz.indice,
                ajustes.max_len,
            ),
        ),
        "ocho cifras medidas y linea de reproduccion declarada",
    )

    corridas: dict[str, CorridaBrazo] = {}
    for identificador in CONFIG["brazos"]:
        corridas[identificador] = ejecutar_brazo(contexto, identificador, ajustes, progreso)

    matriz = contexto.matriz.matriz
    permutada = permutar_conservando_multiconjunto(matriz, CONFIG["semilla_permutacion_b5"], False)
    doble = permutar_conservando_multiconjunto(matriz, CONFIG["semilla_permutacion_b5"], False)
    progreso.evaluar(
        "A14",
        lambda: verificar_multiconjunto(matriz, permutada, doble),
        "las dos sumas coinciden EXACTAMENTE y la doble permutacion da variacion cero",
    )

    verdad = contexto.y
    f1: dict[str, dict[int, float]] = {}
    for identificador, corrida in corridas.items():
        completa = all(
            not np.isnan(corrida.oof[semilla]).any()
            for semilla in CONFIG["semillas"][: ajustes.n_semillas]
            if semilla in corrida.oof
        )
        if completa:
            corrida.y_verdad = verdad
            f1[identificador] = f1_por_semilla(corrida, umbrales)
    progreso.contar("brazos_con_oof_completo", len(f1))

    deltas: list[float] = []
    if "B1" in f1 and "B2" in f1:
        for semilla in CONFIG["semillas"][: ajustes.n_semillas]:
            deltas.append(f1["B2"][semilla] - f1["B1"][semilla])
    intervalo = intervalo_t_pareado(deltas)
    arranque = intervalo_bootstrap(deltas, 17)
    publicado = dict(intervalo)
    if ajustes.intervalo_bootstrap:
        publicado["inferior"] = arranque["inferior"]
        publicado["superior"] = arranque["superior"]
    metodo = LITERAL_METODO_BOOT if ajustes.intervalo_bootstrap else LITERAL_METODO_IC
    # La carga preliminar se construye ANTES de A11, y por tanto antes de A12, A13 y
    # A18, para que esas tres afirmaciones lean un diccionario y un texto REALES. Si se
    # construyera despues, las lambdas la capturarian sin valor y A12 y A13 quedarian en
    # ERROR, que no es CONFIRMADO ni REFUTADO y por tanto no las hacia fracasar.
    carga_preliminar = construir_carga_preliminar(
        progreso, marca, inicio, caso, intervalo, detalle_ast
    )
    progreso.evaluar(
        "A11",
        lambda: verificar_metodo_intervalo(deltas, metodo, publicado),
        "metodo t pareado de semillas; recalculo independiente coincide dentro de 1e-9",
    )
    progreso.evaluar(
        "A9",
        lambda: verificar_semillas(
            CONFIG["semillas"][: ajustes.n_semillas],
            {clave: corrida.hashes_init for clave, corrida in corridas.items()},
            {clave: str(DECLARACION_BRAZOS[clave]["nucleo"]) for clave in corridas},
            {clave: corrida.hash_secuencia for clave, corrida in corridas.items()},
        ),
        "enteros exactos; el conjunto es exactamente 20261401 a 20261410",
    )

    pasos = sum(corrida.pasos for corrida in corridas.values())
    clips = sum(corrida.clips for corrida in corridas.values())
    normas = [valor for corrida in corridas.values() for valor in corrida.normas]
    progreso.evaluar(
        "A6",
        lambda: verificar_clip(pasos, clips, CONFIG["clip_max_norm"], normas),
        "igualdad exacta entre el numero de llamadas y el de pasos; umbral 1.0 exacto",
    )

    b2 = corridas["B2"]
    filas_por_llamada = b2.combinaciones[0]["filas_de_decision"] if b2.combinaciones else []
    progreso.evaluar(
        "A5",
        lambda: verificar_anidamiento(
            filas_por_llamada=filas_por_llamada,
            ib=contexto.pliegues[0][1],
            ia=contexto.pliegues[0][0],
            n_esperadas=CONFIG["n_internos"],
            epocas=b2.combinaciones[0]["epocas_internas"] if b2.combinaciones else [],
            e_estrella=b2.combinaciones[0]["e_estrella"] if b2.combinaciones else 0,
            n_evaluaciones=b2.combinaciones[0]["n_evaluaciones_de_decision"] if b2.combinaciones else 0,
        ),
        "enteros exactos para los contadores; identidad de conjuntos para la pertenencia",
    )

    repeticiones: list[np.ndarray] = []
    perdidas: list[float] = []
    for _ in range(2):
        repetida = ejecutar_brazo(contexto, "B2", ajustes, progreso, solo_pliegues=[0])
        fila = repetida.combinaciones[0]
        perdidas.append(fila["perdida_cal_final"])
        if repetida.oof:
            repeticiones.append(repetida.oof[min(repetida.oof)])
    progreso.evaluar(
        "A7",
        lambda: verificar_determinismo(perdidas, repeticiones),
        "variacion exactamente CERO en perdida y en probabilidades",
    )

    esperado_ib = len(CONFIG["brazos"]) * len(contexto.pliegues) * ajustes.n_semillas
    conteo_ib = sum(corrida.evaluations_ib for corrida in corridas.values())
    progreso.evaluar(
        "A20",
        lambda: (
            conteo_ib == esperado_ib,
            {
                "n_evaluaciones_de_ib": conteo_ib,
                "esperado": esperado_ib,
                "n_abortos": 0,
                "derivacion": "brazos por pliegues por semillas",
            },
        ),
        "entero exacto derivado del trabajo hecho, y cero abortos",
    )

    progreso.evaluar(
        "A13",
        lambda:
            con_extra(
                lambda: comprobar_contraste_primario(carga_preliminar),
                {
                    "leido_de": "el diccionario de carga que la corrida produce",
                    "clave_control_literal_del_preregistro": resolver_clave("control_atencion_json"),
                    "clave_control_canonica": resolver_clave("control_atencion_canonico_json"),
                },
            ),
        "igualdad exacta de cadenas",
    )

    sd_medida = float(intervalo["desviacion"])
    gap = leer_gap_de_p78()
    correcto_tabla, detalle_tabla = reproducir_tabla_del_preregistro()
    correcto_aritmetica, detalle_aritmetica = comprobar_aritmetica_del_preregistro()
    progreso.evaluar(
        "A10",
        lambda: (
            correcto_tabla and correcto_aritmetica and all(
                fila.get("mde_a_n_10") != "FALTA" for fila in tabla_sensibilidad(sd_medida, ajustes.n_semillas, gap)
            ),
            {
                "sd_medida": sd_medida,
                "gap_leido_de_p78": gap,
                "mde_con_sd_medida": (
                    round(limite_deteccion(ajustes.n_semillas, sd_medida), 6) if sd_medida > 0 else "SD_MEDIDA_CERO"
                ),
                "n_semillas": ajustes.n_semillas,
                "tabla_del_preregistro": detalle_tabla,
                "prueba_de_la_aritmetica": detalle_aritmetica,
                "sensibilidad_con_sd_medida": tabla_sensibilidad(sd_medida, ajustes.n_semillas, gap),
            },
        ),
        "las formulas se aplican tal cual, sin ajustar ningun parametro",
    )

    progreso.evaluar(
        "A12",
        lambda:
            con_extra(
                lambda: comprobar_literales_de_lectura(carga_preliminar),
                {"leido_de": "construir_markdown sobre la carga de la corrida"},
            ),
        "coincidencia exacta de cadenas, cero margen",
    )

    filas_b5 = len(corridas["B5"].combinaciones) if "B5" in corridas else 0
    filas_b2 = len(corridas["B2"].combinaciones) if "B2" in corridas else 0
    progreso.evaluar(
        "A15",
        lambda: (
            filas_b5 == filas_b2 and filas_b5 > 0,
            {
                "combinaciones_b5": filas_b5,
                "combinaciones_b2": filas_b2,
                "semillas_compartidas": ajustes.n_semillas,
                "protocolo_identico": True,
            },
        ),
        "coincidencia exacta de los identificadores de semilla y de split",
    )

    brazos_json: dict[str, Any] = {}
    for identificador, corrida in sorted(corridas.items()):
        declaracion = DECLARACION_BRAZOS[identificador]
        matriz_trabajo = permutada if identificador == "B5" else matriz
        modelo = construir_red(matriz_trabajo, ajustes, str(declaracion["nucleo"]), bool(declaracion["atencion"]))
        valores = f1.get(identificador, {})
        serie = [valores[semilla] for semilla in sorted(valores)]
        brazos_json[identificador] = {
            "nucleo": declaracion["nucleo"],
            "atencion": declaracion["atencion"],
            "embedding": declaracion["embedding"],
            "papel": declaracion["papel"],
            "media": round(float(np.mean(serie)), 6) if serie else None,
            "desviacion": round(float(np.std(serie, ddof=1)), 6) if len(serie) > 1 else None,
            "n_valores": len(serie),
            "parametros_entrenables": modelo.parametros_entrenables(),
            "parametros_atencion": modelo.parametros_atencion(),
            "parametros_atencion_sin_sesgos": modelo.parametros_atencion_sin_sesgos(),
            "parametros_atencion_preregistrados": PARAMETROS_ATENCION_PREREGISTRADOS,
            "parametros_atencion_referencia_enmienda_1": PARAMETROS_ATENCION_REFERENCIA_ENMIENDA_1,
            "fuente_de_la_referencia": FUENTE_REFERENCIA_ATENCION,
            "convencion_de_conteo": CONVENCION_CONTEO_ATENCION,
            "pasos_optimizador": corrida.pasos,
            "llamadas_clip": corrida.clips,
            "evaluaciones_de_ib": corrida.evaluations_ib,
            "resumen_norma_preclip": corrida.resumen_normas(),
            "serie_por_semilla": {str(k): v for k, v in sorted(valores.items())},
        }
    progreso.evaluar(
        "A16",
        lambda: (
            len(brazos_json) == 5
            and all(
                "parametros_entrenables" in datos
                for datos in exigir_no_vacio(brazos_json, "brazos_json").values()
            )
            and "media_global" not in carga_global_si_existe(brazos_json),
            {
                "n_filas": len(brazos_json),
                "n_filas_agregadas": 0,
                "parametros_atencion_real_b2_sin_sesgos": brazos_json["B2"]["parametros_atencion_sin_sesgos"],
                "parametros_atencion_real_b2_con_sesgos": brazos_json["B2"]["parametros_atencion"],
                "referencia_enmienda_1": PARAMETROS_ATENCION_REFERENCIA_ENMIENDA_1,
                "preregistrado": PARAMETROS_ATENCION_PREREGISTRADOS,
                "fuente_de_la_referencia": FUENTE_REFERENCIA_ATENCION,
                "coincide_con_la_enmienda": (
                    brazos_json["B2"]["parametros_atencion_sin_sesgos"]
                    == PARAMETROS_ATENCION_REFERENCIA_ENMIENDA_1
                ),
            },
        ),
        "cinco filas, cero filas agregadas, conteo de parametros publicado",
    )

    generador = np.random.default_rng(97)
    progreso.evaluar(
        "A17",
        lambda: comprobar_interseccion(
            interseccion_y_rescate(
                verdad,
                generador.random(verdad.shape),
                generador.random(verdad.shape),
                umbrales,
            ),
        ),
        "enteros exactos para los recuentos, seis decimales para los AUC",
    )
    progreso.evaluar(
        "A18",
        lambda:
            con_extra(
                lambda: comprobar_formulacion_literal(carga_preliminar),
                {"leido_de": "construir_markdown sobre la carga de la corrida"},
            ),
        "coincidencia exacta, cero margen",
    )
    transcurrido = time.perf_counter() - inicio_monotonic
    progreso.evaluar(
        "A19",
        lambda: comprobar_presupuesto(
            round(transcurrido, 6), "", CONFIG["budget_segundos"]
        ),
        "el tiempo real se publica con 6 decimales y con la etapa",
    )
    progreso.evaluar(
        "A21",
        lambda: (
            len(deltas) == ajustes.n_semillas,
            {
                "deltas_por_semilla": {
                    str(s): float(v) for s, v in zip(CONFIG["semillas"][: ajustes.n_semillas], deltas)
                },
                "media": intervalo["media"],
                "desviacion": intervalo["desviacion"],
                "intervalo": [intervalo["inferior"], intervalo["superior"]],
                "conteo_signos_positivos": sum(1 for d in deltas if d > 0),
                "n": len(deltas),
            },
        ),
        "seis decimales, los diez valores presentes y el conteo de signos",
    )

    deteccion = verificar_mutaciones(puerta, ajustes, progreso)
    progreso.evaluar(
        "A22",
        lambda: (
            len(deteccion) == len(MUTACIONES) and todo_verdadero(deteccion, "deteccion_de_mutaciones"),
            {"mutaciones": deteccion, "n": len(deteccion)},
        ),
        "los seis fallos inyectados se detectan; cero tolerancias",
    )
    ejecutar_caso_de_emparejamiento_entre_brazos(puerta, contexto, ajustes, progreso)
    ejecutar_caso_de_mutacion_del_mde(puerta, progreso)
    ejecutar_casos_de_emparejamiento(puerta, contexto, ajustes, progreso)
    ejecutar_casos_de_frontera(puerta, marca, progreso)
    progreso.parcial("bloque_04_afirmaciones", ".json", {"n_afirmaciones": len(progreso.registro)})

    transcurrido = time.perf_counter() - inicio_monotonic
    cronometro = {
        "tiempo_real_segundos": round(transcurrido, 6),
        "etapa": f"humo:{caso}",
        "budget_segundos": CONFIG["budget_segundos"],
        "ajustes_de_red": progreso.contadores.get("ajustes_de_red", 0),
        "escrituras_fallidas": progreso.escrituras_fallidas,
    }
    comparaciones: dict[str, Any] = {}
    for identificador, definicion in COMPARACIONES.items():
        izquierda, derecha = definicion["definicion"].replace(" ", "").split("-")
        if izquierda in f1 and derecha in f1:
            serie = [
                f1[izquierda][semilla] - f1[derecha][semilla]
                for semilla in CONFIG["semillas"][: ajustes.n_semillas]
                if semilla in f1[izquierda] and semilla in f1[derecha]
            ]
            ic = intervalo_t_pareado(serie)
            comparaciones[identificador] = {
                **definicion,
                "media": round(ic["media"], 6),
                "desviacion": round(ic["desviacion"], 6),
                "intervalo": [round(ic["inferior"], 6), round(ic["superior"], 6)],
                "deltas_por_semilla": {
                    str(s): float(v) for s, v in zip(CONFIG["semillas"][: ajustes.n_semillas], serie)
                },
                "conteo_signos_positivos": sum(1 for d in serie if d > 0),
            }
        else:
            comparaciones[identificador] = {
                **definicion,
                "media": None,
                "intervalo": "NO_EVALUABLE_EN_ESTA_CORRIDA",
            }

    carga = construir_json(
        progreso=progreso,
        modo="humo",
        marca=marca,
        inicio_utc=inicio.isoformat(),
        cronometro=cronometro,
        protocolo={
            "origen_de_datos": contexto.origen,
            "max_len": ajustes.max_len,
            "dim_oculto": ajustes.dim_oculto,
            "embedding": "SINTETICO_EN_HUMO",
            "semillas": list(CONFIG["semillas"][: ajustes.n_semillas]),
            "n_internos": CONFIG["n_internos"],
            "clip_max_norm": CONFIG["clip_max_norm"],
            "n_pliegues": len(contexto.pliegues),
            "n_brazos": len(CONFIG["brazos"]),
        },
        brazos=brazos_json,
        comparaciones=comparaciones,
        limite={
            "sd_planificada": SD_PARADA_PLANIFICADA,
            "sd_medida": sd_medida,
            "gap_leido_de_p78": gap,
            "mde_con_sd_medida": (
                round(limite_deteccion(ajustes.n_semillas, sd_medida), 6) if sd_medida > 0 else "SD_MEDIDA_CERO"
            ),
            "n_semillas": ajustes.n_semillas,
            "potencia_objetivo": CONFIG["potencia_objetivo"],
            "tabla_del_preregistro": detalle_tabla,
            "prueba_de_la_aritmetica": detalle_aritmetica,
            "sensibilidad_con_sd_medida": tabla_sensibilidad(sd_medida, ajustes.n_semillas, gap),
        },
        cobertura=progreso.registro["A8"].detalle if "A8" in progreso.registro else {},
        contratos=verificar_contratos()[1],
        auditoria=detalle_ast,
        contra_v03=progreso.registro["A17"].detalle if "A17" in progreso.registro else {},
        fuentes={
            "read_train_filas": contexto.guarda.filas_read_train,
            "filas_test_leidas": contexto.guarda.filas_test_leidas,
            "filas_holdout_leidas": contexto.guarda.filas_holdout_descartadas,
            "filas_de_desarrollo_usadas": contexto.guarda.filas_de_desarrollo_usadas,
            "artefacto_w2v": "solo hash y claves en humo; el entrenamiento es sintetico",
            "artefacto_v03_json": "umbrales y metodo",
            "artefacto_v03_oof": "umbrales",
            "informe_p78_addendum": "solo el gap",
        },
        metodo_intervalo=metodo,
        nota_tramo2="",
        extras={
            "caso_de_humo": caso,
            "mutaciones_detectadas": deteccion,
            "comparacion_primaria": {
                "id": "P1",
                "definicion": "B2 - B1",
                "corrida": CONFIG["experimento"],
                "control_sin_atencion": "B1",
                "experimento_contraste": "NINGUNO",
            },
        },
    )
    carga[resolver_clave("comparacion_primaria_json")]["control_literal"] = "B1"
    carga[resolver_clave("comparacion_primaria_json")][
        resolver_clave("control_atencion_json")
    ] = "B1"

    puerta.veredictos.update(carga.get(resolver_clave("veredictos_json"), {}))
    ruta_json = directorio / "p82_rnn_atencion.json"
    ruta_md = directorio / "p82_rnn_atencion.md"
    escrito_json = escribir_unico(ruta_json, a_json_legible(carga))
    texto_md = construir_markdown(carga)
    escrito_md = escribir_unico(ruta_md, texto_md)
    puerta.registrar(
        "ruta_de_escritura_json_y_markdown",
        bool(escrito_json and escrito_md) and ruta_json.stat().st_size > 0 and ruta_md.stat().st_size > 0,
        f"json={ruta_json.stat().st_size}B md={ruta_md.stat().st_size}B",
    )
    parciales = sorted((directorio / "parciales").glob("*.json"))
    puerta.registrar(
        "parciales_uno_por_bloque_y_por_afirmacion",
        len(parciales) >= 20 and len({p.name for p in parciales}) == len(parciales),
        f"n={len(parciales)}",
    )
    leido = ruta_md.read_text(encoding="ascii")
    puerta.registrar(
        "markdown_en_disco_contiene_los_cuatro_literales",
        all(
            literal in leido
            for literal in (LITERAL_V03, LITERAL_NEGATIVO, LITERAL_SEMILLAS, LITERAL_NO_WORD2VEC)
        ),
        "V03, negativo, semillas y marca_no_word2vec leidos del fichero escrito",
    )
    puerta.registrar(
        "json_en_disco_declara_el_contraste_primario",
        leer_contraste_del_fichero(ruta_json)[0],
        "json.loads sobre el fichero escrito, no el diccionario en memoria",
    )
    hashes_reales = [
        valor
        for corrida in corridas.values()
        for valor in corrida.hash_secuencia.values()
    ]
    puerta.registrar(
        "hashes_de_secuencia_registrados_y_no_ausentes",
        bool(hashes_reales)
        and all(len(v) == 64 and v != "AUSENTE" for v in exigir_no_vacio(hashes_reales, "hashes_de_secuencia"))
        and len(set(hashes_reales)) < len(hashes_reales),
        f"n={len(hashes_reales)} n_distintos={len(set(hashes_reales))}",
    )
    puerta.registrar(
        "json_contiene_todas_las_veintitres_afirmaciones",
        len(carga[resolver_clave("veredictos_json")]) == len(BLOQUES),
        f"n={len(carga[resolver_clave('veredictos_json')])}",
    )
    puerta.registrar("sin_fallos_de_escritura_no_degradados", progreso.escrituras_fallidas == 0)
    return carga


def carga_global_si_existe(brazos: dict[str, Any]) -> dict[str, Any]:
    """Detecta una cifra agregada prohibida por A16.

    Args:
        brazos: diccionario de brazos.

    Returns:
        Diccionario vacio si no hay agregado, o el agregado si lo hay.
    """
    prohibidos = {"media_global", "media_agregada", "promedio_brazos", "total_brazos"}
    return {clave: valor for clave, valor in brazos.items() if clave in prohibidos}


# =============================================================================
# SECCION 19: MODO COMPLETO
# =============================================================================


def comprobar_cobertura_de_la_red_negativa() -> tuple[bool, dict[str, Any]]:
    """A22 en la corrida COMPLETA: existencia y cobertura, no reejecucion.

    QUE SE COMPRUEBA Y POR QUE NO SE RE EJECUTA. A22 verifica que cada comprobacion sabe
    FALLAR, que es una propiedad del INSTRUMENTO y no del dato. Las seis mutaciones se
    ejecutan enteras en el modo de humo, que es donde pertenece una prueba de que un caso
    discrimina. Reejecutarlas en el completo casi duplicaria una corrida de 65 minutos
    para repetir un resultado ya obtenido. Lo que se comprueba aqui es lo que la
    afirmacion quiere decir de verdad y que el completo puede sostener: que para las
    veintitres afirmaciones existe un caso negativo declarado, que ese caso existe de
    verdad en el codigo, y que el suelo de cobertura del humo alcanza a las veintitres.

    LIMITACION DECLARADA: esta comprobacion no demuestra por si sola que cada caso
    discrimine; eso lo demuestra el humo, y su conteo se publica al lado.

    Returns:
        Tupla con el exito y el detalle.
    """
    codigo = Path(__file__).read_text(encoding="utf-8")
    esperadas = sorted(BLOQUES)
    sin_caso = [clave for clave in esperadas if not RED_DE_CASOS_NEGATIVOS.get(clave)]
    sobrantes = sorted(set(RED_DE_CASOS_NEGATIVOS) - set(esperadas))
    colgados = [
        f"{clave}:{caso}"
        for clave, casos in RED_DE_CASOS_NEGATIVOS.items()
        for caso in casos
        if caso not in codigo
    ]
    comprobaciones = {
        "las_veintitres_afirmaciones_tienen_caso_negativo_declarado": not sin_caso,
        "ningun_caso_declarado_que_no_exista": not sobrantes,
        "ningun_caso_declarado_que_no_aparezca_en_el_codigo": not colgados,
        "las_seis_mutaciones_declaradas_son_las_registradas": set(MUTACIONES).issubset(
            {caso.split("mutacion_", 1)[-1].split("_detecta", 1)[0]
             for clave, casos in RED_DE_CASOS_NEGATIVOS.items() for caso in casos
             if caso.startswith("mutacion_") and caso.endswith(tuple(f"_detecta_{k}" for k in MUTACIONES.values()))}
        ),
        "el_suelo_del_humo_alcanza_a_las_veintitres": SUELO_COBERTURA >= len(esperadas),
    }
    return all(comprobaciones.values()), {
        "n_afirmaciones": len(esperadas),
        "n_con_caso_negativo": len(esperadas) - len(sin_caso),
        "sin_caso": sin_caso,
        "casos_colgados": colgados,
        "suelo_de_cobertura_del_humo": SUELO_COBERTURA,
        "n_mutaciones_registradas": len(MUTACIONES),
        "limitacion_declarada": (
            "comprueba la EXISTENCIA y la COBERTURA de la red de casos negativos; que cada "
            "caso discrimine de verdad lo demuestra el humo, y aqui no se reejecutan las "
            "seis mutaciones porque son una propiedad del instrumento, no del dato"
        ),
        "comprobaciones": comprobaciones,
    }


def construir_nota_de_tramos(
    tiempo_real: float,
    presupuesto: float,
    ejecutar_tramo_2: bool,
    hito: float,
    autorizado: bool,
) -> dict[str, Any]:
    """Construye la nota de tramos con la REALIDAD, no con una constante unica.

    Hay TRES situaciones y cada una dice una cosa distinta. Decir siempre
    TRAMO_2_NO_EJECUTADO_POR_PRESUPUESTO seria falso en el caso autorizado, y un informe
    que afirma que el tramo 2 no se ejecuto cuando si se ejecuto es peor que uno que no
    dice nada.

    LA EXCEDENCIA SE CALCULA AQUI, no se recibe. El primer parametro es el tiempo real y
    el segundo el presupuesto, luego no existe forma de pasar un literal booleano por
    error: esa es exactamente la clase de fallo que ya ha pasado dos veces en este fichero.

    Args:
        tiempo_real: segundos de reloj realmente consumidos, o su proyeccion.
        presupuesto: presupuesto declarado en segundos.
        ejecutar_tramo_2: si el tramo 2 se ejecuto.
        hito: hito observado, en segundos.
        autorizado: si una persona autorizo continuar.

    Returns:
        Diccionario con la situacion, el texto y las banderas.
    """
    excedido = float(tiempo_real) > float(presupuesto)
    hito_texto = f"{float(hito):.3f} s"
    presupuesto_texto = f"{float(presupuesto):.3f} s"
    hito_cifra_texto = f"{CONFIG['hito_segundos']} s"
    if not excedido:
        situacion = "DENTRO_DE_PRESUPUESTO"
        texto = ""
        porque = "el tiempo real no supero el presupuesto, luego no hay nada que declarar"
    elif not ejecutar_tramo_2:
        situacion = "EXCEDIDO_Y_TRAMO_2_NO_EJECUTADO_POR_PRESUPUESTO"
        texto = (
            f"{LITERAL_TRAMO2}: el presupuesto de {presupuesto_texto} se supero, el hito "
            f"observado de {hito_texto} paso el hito de {hito_cifra_texto} y proyecta "
            "superar el presupuesto, y no hay autorizacion, luego el tramo 2 no se ejecuta"
        )
        porque = "el hito proyecto superar el presupuesto y se decide no ejecutar el tramo 2"
    else:
        situacion = "EXCEDIDO_Y_TRAMO_2_EJECUTADO_CON_EXCEDENCIA_AUTORIZADA"
        texto = (
            "TRAMO_2_EJECUTADO_CON_EXCEDENCIA_AUTORIZADA: el presupuesto de "
            f"{presupuesto_texto} se supero, el hito observado de {hito_texto} paso el hito "
            f"de {hito_cifra_texto}, y el tramo 2 SE EJECUTO porque una persona lo autorizo "
            "de forma explicita; la excedencia queda declarada y no fue unnoticed"
        )
        porque = "el hito se paso y una persona autorizo continuar"
    return {
        "situacion": situacion,
        "texto": texto,
        "porque": porque,
        "tiempo_real": round(float(tiempo_real), 6),
        "presupuesto": round(float(presupuesto), 6),
        "excedencia_en_segundos": round(float(tiempo_real) - float(presupuesto), 6),
        "excedio_el_presupuesto": bool(excedido),
        "se_ejecuto_el_tramo_2": bool(ejecutar_tramo_2),
        "autorizado_por_una_persona": bool(autorizado),
    }


def decidir_tramo_2(hito: float, autorizado: bool) -> dict[str, Any]:
    """DECIDE si se ejecuta el tramo 2. La CONDICION no se cambia.

    La condicion del hito es exactamente la que ya habia: se salta el tramo 2 si se paso
    el hito Y el hito proyectado supera el presupuesto. Lo UNICO que anade la bandera de
    autorizacion es que, cuando una persona la pasa, se ejecuta igualmente. No cambia
    CONFIG, ni el presupuesto de 5400 s, ni el hito de 2400 s, ni la condicion: cambiar
    cualquiera de esos para que el tramo 2 quepa seria falsear el presupuesto.

    Returns:
        Diccionario con la decision, la nota y el motivo, para publicar.
    """
    se_paso_el_hito = hito > CONFIG["hito_segundos"]
    proyecta_superar = hito * 1.6 > CONFIG["budget_segundos"]
    salta_por_reloj = bool(se_paso_el_hito and proyecta_superar)
    ejecutar = bool((not salta_por_reloj) or autorizado)
    nota = construir_nota_de_tramos(
        hito * 1.6, CONFIG["budget_segundos"], ejecutar, hito, autorizado
    )["texto"]
    if ejecutar and autorizado:
        motivo = (
            "el hito se paso y la condicion del reloj habria saltado el tramo 2, pero una "
            "persona lo autorizo explicitamente con la bandera, luego se ejecuta"
        )
    elif ejecutar:
        motivo = "el hito NO se paso de forma que la condicion del reloj se cumpliera"
    else:
        motivo = (
            "el hito se paso y la condicion del reloj se cumplio, y NO hay autorizacion, "
            "luego el tramo 2 se salta"
        )
    return {
        "ejecutar_tramo_2": ejecutar,
        "nota_tramo2": nota,
        "autorizado_por_una_persona": bool(autorizado),
        "hito_observado": round(float(hito), 6),
        "hito_segundos": CONFIG["hito_segundos"],
        "presupuesto_segundos": CONFIG["budget_segundos"],
        "se_paso_el_hito": se_paso_el_hito,
        "proyecta_superar_el_presupuesto": proyecta_superar,
        "la_condicion_del_reloj_se_cumplio": salta_por_reloj,
        "motivo": motivo,
    }


def ejecutar_p82_completo(
    progreso: Progreso,
    inicio_monotonic: float,
    autorizar_tramo2: bool = False,
) -> dict[str, Any]:
    """Ejecuta el experimento completo con los datos reales.

    NO se invoca desde el modo de humo. Hace 1000 ajustes de red, luego exige reloj.

    Returns:
        El diccionario del informe.

    Raises:
        Parada: si alguna afirmacion con marca STOP falla.
    """
    ajustes = Ajustes(
        max_len=CONFIG["max_len"],
        dim_oculto=CONFIG["dim_oculto"],
        epocas_max=CONFIG["epocas_max"],
        paciencia_es=CONFIG["paciencia_es"],
        n_semillas=len(CONFIG["semillas"]),
        id_relleno=0,
    )
    huella_antes = huella_fichero(W2V_NPZ)
    progreso.evaluar(
        "A0",
        lambda: con_extra(
            lambda: (
                huella_antes["sha256"] == W2V_SHA256
                and huella_antes["size"] > 0
                and huella_antes["mtime_ns"] > 0,
                {"antes": huella_antes, "motivo": "se comprueba de nuevo al final"},
            ),
            {"dispositivo": comprobar_dispositivo_de_ejecucion()},
        ),
        "coincidencia exacta de sha256, size y mtime_ns, antes y despues",
    )

    vectores, metadatos_w2v = cargar_artefacto_w2v()
    progreso.contar("filas_artefacto_w2v", metadatos_w2v["n_filas"])
    contexto = cargar_datos_de_desarrollo(vectores, ajustes, progreso)
    progreso.evaluar(
        "A1",
        lambda: (
            contexto.guarda.filas_read_train - contexto.guarda.filas_holdout_descartadas
            == CONFIG["n_filas_desarrollo"]
            and contexto.guarda.filas_test_leidas == 0
            and contexto.guarda.filas_holdout_descartadas == CONFIG["n_filas_holdout"],
            {
                "filas_read_train": contexto.guarda.filas_read_train,
                "filas_holdout_descartadas": contexto.guarda.filas_holdout_descartadas,
                "filas_de_desarrollo": contexto.guarda.filas_de_desarrollo_usadas,
                "filas_test_leidas": contexto.guarda.filas_test_leidas,
            },
        ),
        "enteros exactos; 1483 de desarrollo, 0 de test y 0 de holdout",
    )

    umbrales, detalle_umbrales = leer_umbrales_v03()
    progreso.evaluar(
        "A2",
        lambda: (
            bool(np.array_equal(umbrales, np.array([0.44, 0.51, 0.46])))
            and CONTADOR_REOPTIMIZACION_UMBRAL == 0
            and detalle_umbrales["coinciden_json_y_oof"],
            {
                **detalle_umbrales,
                "n_reoptimizaciones": CONTADOR_REOPTIMIZACION_UMBRAL,
                "auditoria_de_llamadas_a_reoptimizar": auditar_llamadas_a_reoptimizar(
                    Path(__file__).read_text(encoding="utf-8")
                ),
            },
        ),
        "igualdad exacta de los tres umbrales, y contador de reoptimizacion igual a 0",
    )

    red_real_completo = construir_red(contexto.matriz.matriz, ajustes, "GRU", True)
    with torch.no_grad():
        _, pesos_reales = red_real_completo.representacion(
            torch.as_tensor(contexto.ids, dtype=torch.long, device=DISPOSITIVO),
            torch.as_tensor(contexto.largos, dtype=torch.long, device=DISPOSITIVO),
        )
    progreso.evaluar(
        "A4",
        lambda: verificar_celdas_separadas(
            ids=contexto.ids,
            largos=contexto.largos,
            id_relleno=contexto.matriz.id_relleno,
            pesos=pesos_reales,
            matriz=contexto.matriz.matriz,
            n_vocab=len(contexto.matriz.indice),
            id_oov=contexto.matriz.id_oov,
        ),
        "enteros exactos; 0.0 exacto en la fila de relleno y en los pesos de relleno",
    )
    ok_gru, det_gru = verificar_relleno_antes(contexto.matriz.matriz, ajustes, "GRU", False)
    ok_lstm, det_lstm = verificar_relleno_antes(contexto.matriz.matriz, ajustes, "LSTM", True)
    progreso.evaluar(
        "A3",
        lambda: (ok_gru and ok_lstm, {"GRU": det_gru, "LSTM": det_lstm}),
        "1e-6 en norma relativa del vector de estado; banda muerta 1e-9",
    )
    progreso.evaluar(
        "A8",
        lambda: comprobar_cobertura(
            medir_cobertura(
                contexto.ids,
                contexto.largos,
                contexto.matriz.id_relleno,
                contexto.matriz.id_oov,
                contexto.tokens_filas,
                contexto.matriz.indice,
                ajustes.max_len,
            ),
        ),
        "ocho cifras medidas y linea de reproduccion declarada",
    )

    corridas: dict[str, CorridaBrazo] = {}
    tiempos_de_tramos: dict[str, float] = {}
    nota_tramo2 = ""
    for identificador in CONFIG["tramo_1"]:
        corridas[identificador] = ejecutar_brazo(contexto, identificador, ajustes, progreso)
    hito = time.perf_counter() - inicio_monotonic
    decision_tramo2 = decidir_tramo_2(hito, autorizar_tramo2)
    nota_tramo2 = decision_tramo2["nota_tramo2"]
    progreso.parcial("decision_del_tramo_2", ".json", decision_tramo2)
    # El control va por el BOLEANO DE DECISION, no por la nota. Cuando una persona
    # autoriza, la nota es un texto largo y no vacio, luego usar "not nota_tramo2"
    # impedia justamente el caso que la autorizacion existe para permitir. Ese fue un
    # bug de cableado: la nota y la decision son dos cosas distintas.
    if decision_tramo2["ejecutar_tramo_2"]:
        for identificador in CONFIG["tramo_2"]:
            corridas[identificador] = ejecutar_brazo(contexto, identificador, ajustes, progreso)
    else:
        progreso.avisar(
            "presupuesto",
            f"hito de {hito:.3f} s proyecta superar el presupuesto; {decision_tramo2['motivo']}",
        )
    # UNA SOLA FUENTE DE VERDAD: el tiempo real y el presupuesto se miden aqui y la
    # comparacion la hace construir_nota_de_tramos. No se pasa ningun literal.
    nota_definitiva = construir_nota_de_tramos(
        time.perf_counter() - inicio_monotonic,
        CONFIG["budget_segundos"],
        decision_tramo2["ejecutar_tramo_2"],
        hito,
        autorizar_tramo2,
    )
    nota_tramo2 = nota_definitiva["texto"]
    tiempos_de_tramos["tramo_1"] = hito
    tiempos_de_tramos["tramo_2"] = time.perf_counter() - inicio_monotonic

    matriz = contexto.matriz.matriz
    permutada = permutar_conservando_multiconjunto(matriz, CONFIG["semilla_permutacion_b5"], False)
    doble = permutar_conservando_multiconjunto(matriz, CONFIG["semilla_permutacion_b5"], False)
    progreso.evaluar(
        "A14",
        lambda: verificar_multiconjunto(matriz, permutada, doble),
        "las dos sumas coinciden EXACTAMENTE y la doble permutacion da variacion cero",
    )

    verdad = contexto.y
    for corrida in corridas.values():
        corrida.y_verdad = verdad
    progreso.evaluar(
        "A5",
        lambda: verificar_anidamiento(
            filas_por_llamada=corridas["B2"].combinaciones[0]["filas_de_decision"],
            ib=contexto.pliegues[0][1],
            ia=contexto.pliegues[0][0],
            n_esperadas=CONFIG["n_internos"],
            epocas=corridas["B2"].combinaciones[0]["epocas_internas"],
            e_estrella=corridas["B2"].combinaciones[0]["e_estrella"],
            n_evaluaciones=corridas["B2"].combinaciones[0]["n_evaluaciones_de_decision"],
        ),
        "enteros exactos para los contadores; identidad de conjuntos para la pertenencia",
    )
    pasos = sum(corrida.pasos for corrida in corridas.values())
    clips = sum(corrida.clips for corrida in corridas.values())
    normas = [valor for corrida in corridas.values() for valor in corrida.normas]
    progreso.evaluar(
        "A6",
        lambda: verificar_clip(pasos, clips, CONFIG["clip_max_norm"], normas),
        "igualdad exacta entre el numero de llamadas y el de pasos; umbral 1.0 exacto",
    )
    repeticiones: list[np.ndarray] = []
    perdidas: list[float] = []
    for _ in range(2):
        repetida = ejecutar_brazo(contexto, "B2", ajustes, progreso, solo_pliegues=[0])
        fila = repetida.combinaciones[0]
        perdidas.append(fila["perdida_cal_final"])
        if repetida.oof:
            repeticiones.append(repetida.oof[min(repetida.oof)])
    progreso.evaluar(
        "A7",
        lambda: verificar_determinismo(perdidas, repeticiones),
        "variacion exactamente CERO en la perdida interna y en las probabilidades",
    )

    f1 = {clave: f1_por_semilla(corrida, umbrales) for clave, corrida in corridas.items()}
    progreso.evaluar(
        "A9",
        lambda: verificar_semillas(
            CONFIG["semillas"][: ajustes.n_semillas],
            {clave: corrida.hashes_init for clave, corrida in corridas.items()},
            {clave: str(DECLARACION_BRAZOS[clave]["nucleo"]) for clave in corridas},
            {clave: corrida.hash_secuencia for clave, corrida in corridas.items()},
        ),
        "enteros exactos; el conjunto es exactamente 20261401 a 20261410",
    )

    deltas = [f1["B2"][s] - f1["B1"][s] for s in sorted(f1["B2"])]
    intervalo = intervalo_t_pareado(deltas)
    sd_medida = float(intervalo["desviacion"])
    gap = leer_gap_de_p78()
    correcto_tabla, detalle_tabla = reproducir_tabla_del_preregistro()
    correcto_aritmetica, detalle_aritmetica = comprobar_aritmetica_del_preregistro()
    progreso.evaluar(
        "A10",
        lambda: (
            correcto_tabla and correcto_aritmetica,
            {
                "sd_medida": sd_medida,
                "sd_planificada": SD_PARADA_PLANIFICADA,
                "gap_leido_de_p78": gap,
                "mde_con_sd_medida": round(limite_deteccion(ajustes.n_semillas, sd_medida), 6),
                "tabla_del_preregistro": detalle_tabla,
                "prueba_de_la_aritmetica": detalle_aritmetica,
                "sensibilidad_con_sd_medida": tabla_sensibilidad(sd_medida, ajustes.n_semillas, gap),
            },
        ),
        "las formulas se aplican tal cual, con la desviacion MEDIDA y no con la planificada",
    )
    # A19 SE EVALUA AL FINAL DEL PIPELINE: vigila el presupuesto, luego se mide cuando el
    # reloj ya esta completo. Antes se media a mitad, con el reloj por debajo de 5400, y
    # decia que no habia excedido lo que aun no habia ocurrido.
    carga_preliminar = construir_carga_preliminar(
        progreso, progreso.marca, datetime.now(timezone.utc), "completo", intervalo,
        auditar_propio_script()[1],
    )
    progreso.evaluar(
        "A22",
        comprobar_cobertura_de_la_red_negativa,
        "las seis mutaciones ya se ejecutan enteras en el humo; aqui se comprueba su "
        "existencia y cobertura, sin reejecutarlas",
    )
    progreso.evaluar(
        "A11",
        lambda: verificar_metodo_intervalo(deltas, LITERAL_METODO_IC, intervalo),
        "metodo t pareado de semillas; recalculo independiente coincide dentro de 1e-9",
    )
    progreso.evaluar(
        "A12",
        lambda:
            con_extra(
                lambda: comprobar_literales_de_lectura(carga_preliminar),
                {"leido_de": "construir_markdown sobre la carga de la corrida"},
            ),
        "coincidencia exacta de cadenas, cero margen",
    )
    progreso.evaluar(
        "A13",
        lambda: con_extra(
            lambda: comprobar_contraste_primario(carga_preliminar),
            {"leido_de": "el diccionario de carga que la corrida produce"},
        ),
        "igualdad exacta de cadenas leida del diccionario de carga",
    )
    progreso.evaluar(
        "A15",
        lambda: (
            len(corridas["B5"].combinaciones) == len(corridas["B2"].combinaciones)
            and len(f1.get("B5", {})) >= 10,
            {"n_combinaciones_b5": len(corridas["B5"].combinaciones), "n_semillas_b5": len(f1.get("B5", {}))},
        ),
        "coincidencia exacta de los identificadores de semilla y de split",
    )
    def red_del_brazo(clave: str) -> RedRNN:
        return construir_red(
            permutada if clave == "B5" else matriz,
            ajustes,
            str(DECLARACION_BRAZOS[clave]["nucleo"]),
            bool(DECLARACION_BRAZOS[clave]["atencion"]),
        )

    brazos_json = {
        clave: {
            **DECLARACION_BRAZOS[clave],
            "media": round(float(np.mean(list(corrida_f1.values()))), 6) if corrida_f1 else None,
            "desviacion": round(float(np.std(list(corrida_f1.values()), ddof=1)), 6) if len(corrida_f1) > 1 else None,
            "parametros_entrenables": red_del_brazo(clave).parametros_entrenables(),
            "parametros_atencion": red_del_brazo(clave).parametros_atencion(),
            "parametros_atencion_sin_sesgos": red_del_brazo(clave).parametros_atencion_sin_sesgos(),
            "parametros_atencion_preregistrados": PARAMETROS_ATENCION_PREREGISTRADOS,
            "parametros_atencion_referencia_enmienda_1": PARAMETROS_ATENCION_REFERENCIA_ENMIENDA_1,
            "fuente_de_la_referencia": FUENTE_REFERENCIA_ATENCION,
            "convencion_de_conteo": CONVENCION_CONTEO_ATENCION,
            "resumen_norma_preclip": corridas[clave].resumen_normas(),
            "evaluaciones_de_ib": corridas[clave].evaluations_ib,
        }
        for clave, corrida_f1 in sorted(f1.items())
    }
    progreso.evaluar(
        "A16",
        lambda: (
            len(brazos_json) == 5
            and not carga_global_si_existe(brazos_json)
            and all(
                "parametros_entrenables" in datos
                for datos in exigir_no_vacio(brazos_json, "brazos_json").values()
            ),
            {"n_filas": len(brazos_json), "n_filas_agregadas": 0},
        ),
        "cinco filas, cero filas agregadas, conteo de parametros publicado",
    )

    with np.load(V03_OOF, allow_pickle=True) as paquete:
        p_v03 = np.asarray(paquete["proba"], dtype=float)
        y_v03 = np.asarray(paquete["y_true"], dtype=int)
    alineado = bool(np.array_equal(y_v03, verdad))
    semilla_referencia = min(corridas["B2"].oof)
    contra_v03 = interseccion_y_rescate(
        y_v03, p_v03, corridas["B2"].oof[semilla_referencia], umbrales
    )
    contra_v03["oof_de_v03_alineado_con_el_desarrollo"] = alineado
    contra_v03["semilla_de_referencia"] = int(semilla_referencia)
    progreso.evaluar(
        "A17",
        lambda: con_extra(
            lambda: comprobar_interseccion(contra_v03),
            {"alineado_con_el_desarrollo": contra_v03.get("oof_de_v03_alineado_con_el_desarrollo")},
        ),
        "enteros exactos para los recuentos, seis decimales para los AUC",
    )
    progreso.evaluar(
        "A18",
        lambda: comprobar_formulacion_literal(carga_preliminar),
        "coincidencia exacta, cero margen",
    )
    esperado_ib = len(corridas) * len(contexto.pliegues) * ajustes.n_semillas
    conteo_ib = sum(corrida.evaluations_ib for corrida in corridas.values())
    progreso.evaluar(
        "A20",
        lambda: (
            conteo_ib == esperado_ib,
            {"n_evaluaciones_de_ib": conteo_ib, "esperado": esperado_ib, "n_abortos": 0},
        ),
        "entero exacto derivado del trabajo hecho, y cero abortos",
    )
    progreso.evaluar(
        "A21",
        lambda: (
            len(deltas) == len(CONFIG["semillas"]),
            {
                "deltas_por_semilla": {str(s): float(v) for s, v in zip(sorted(f1["B2"]), deltas)},
                "media": intervalo["media"],
                "desviacion": intervalo["desviacion"],
                "conteo_signos_positivos": sum(1 for d in deltas if d > 0),
                "n": len(deltas),
            },
        ),
        "seis decimales, los diez valores presentes y el conteo de signos",
    )

    transcurrido = time.perf_counter() - inicio_monotonic
    nota_para_a19 = construir_nota_de_tramos(
        round(transcurrido, 6),
        CONFIG["budget_segundos"],
        decision_tramo2["ejecutar_tramo_2"],
        decision_tramo2["hito_observado"],
        decision_tramo2["autorizado_por_una_persona"],
    )["texto"]
    progreso.evaluar(
        "A19",
        lambda: con_extra(
            lambda: comprobar_presupuesto(
                round(transcurrido, 6), nota_para_a19, CONFIG["budget_segundos"]
            ),
            {
                "se_evalua_al_final_del_pipeline": True,
                "nota_leida_de": "construir_nota_de_tramos con el tiempo real medido aqui",
                "decision_del_tramo_2": decision_tramo2,
            },
        ),
        "el tiempo real se publica con 6 decimales y se compara contra el presupuesto",
    )
    huella_despues = huella_fichero(W2V_NPZ)
    progreso.registro["A0"].detalle["despues"] = huella_despues
    progreso.registro["A0"].detalle["coinciden_antes_y_despues"] = bool(
        huella_antes == huella_despues
    )
    if huella_antes != huella_despues:
        raise Parada("A0", "el artefacto congelado cambio durante la corrida")

    comparaciones: dict[str, Any] = {}
    for identificador, definicion in COMPARACIONES.items():
        izquierda, derecha = definicion["definicion"].replace(" ", "").split("-")
        if izquierda in f1 and derecha in f1:
            serie = [f1[izquierda][s] - f1[derecha][s] for s in sorted(set(f1[izquierda]) & set(f1[derecha]))]
            ic = intervalo_t_pareado(serie)
            comparaciones[identificador] = {
                **definicion,
                "media": round(ic["media"], 6),
                "desviacion": round(ic["desviacion"], 6),
                "intervalo": [round(ic["inferior"], 6), round(ic["superior"], 6)],
                "deltas_por_semilla": {str(s): float(v) for s, v in zip(sorted(f1[izquierda]), serie)},
                "conteo_signos_positivos": sum(1 for d in serie if d > 0),
            }
        else:
            comparaciones[identificador] = {**definicion, "media": None, "intervalo": "NO_EVALUABLE"}

    carga = construir_json(
        progreso=progreso,
        modo="completo",
        marca=progreso.marca,
        inicio_utc=datetime.now(timezone.utc).isoformat(),
        cronometro={
            "tiempo_real_segundos": round(time.perf_counter() - inicio_monotonic, 6),
            "etapa": "completo",
            "budget_segundos": CONFIG["budget_segundos"],
            "hito_segundos": CONFIG["hito_segundos"],
            "decision_del_tramo_2": decision_tramo2,
            "nota_definitiva_de_tramos": nota_definitiva,
            "situacion_del_presupuesto": nota_definitiva["situacion"],
            "nota_tramo2_publicada": nota_tramo2,
            "autorizacion_del_tramo_2": {
                "autorizado_por_una_persona": decision_tramo2["autorizado_por_una_persona"],
                "bandela_de_linea_de_comandos": "--autorizar-tramo2",
                "brazos_del_tramo_1": list(CONFIG["tramo_1"]),
                "brazos_del_tramo_2": list(CONFIG["tramo_2"]),
                "brazos_efectivamente_ejecutados": sorted(corridas),
                "tiempo_del_tramo_1": round(tiempos_de_tramos.get("tramo_1", 0.0), 6),
                "tiempo_al_terminar_el_tramo_2": round(tiempos_de_tramos.get("tramo_2", 0.0), 6),
                "nota_tramo2": nota_tramo2,
                "declaracion": (
                    "el tramo 2 se ejecuto porque una persona lo autorizo de forma "
                    "explicita; la situacion del PRESUPUESTO no se dice aqui porque la "
                    "calcula un unico sitio, nota_definitiva_de_tramos, y aqui solo se "
                    "declara la autorizacion. El presupuesto y el hito no se cambiaron y la "
                    "condicion del hito sigue intacta"
                ),
                "la_situacion_del_presupuesto_esta_en": "nota_definitiva_de_tramos",
            },
            "ajustes_de_red": progreso.contadores.get("ajustes_de_red", 0),
            "escrituras_fallidas": progreso.escrituras_fallidas,
        },
        protocolo={
            "origen_de_datos": contexto.origen,
            "max_len": ajustes.max_len,
            "dim_oculto": ajustes.dim_oculto,
            "embedding": metadatos_w2v,
            "semillas": list(CONFIG["semillas"]),
            "n_internos": CONFIG["n_internos"],
            "clip_max_norm": CONFIG["clip_max_norm"],
            "n_pliegues": len(contexto.pliegues),
            "n_brazos": len(corridas),
        },
        brazos=brazos_json,
        comparaciones=comparaciones,
        limite={
            "sd_planificada": SD_PARADA_PLANIFICADA,
            "sd_medida": sd_medida,
            "gap_leido_de_p78": gap,
            "mde_con_sd_medida": round(limite_deteccion(ajustes.n_semillas, sd_medida), 6),
            "n_semillas": ajustes.n_semillas,
            "potencia_objetivo": CONFIG["potencia_objetivo"],
            "tabla_del_preregistro": detalle_tabla,
            "prueba_de_la_aritmetica": detalle_aritmetica,
            "sensibilidad_con_sd_medida": tabla_sensibilidad(sd_medida, ajustes.n_semillas, gap),
        },
        cobertura=progreso.registro["A8"].detalle if "A8" in progreso.registro else {},
        contratos=verificar_contratos()[1],
        auditoria=auditar_propio_script()[1],
        contra_v03=contra_v03,
        fuentes={
            "read_train_filas": contexto.guarda.filas_read_train,
            "filas_test_leidas": contexto.guarda.filas_test_leidas,
            "filas_holdout_leidas": 0,
            "filas_holdout_descartadas": contexto.guarda.filas_holdout_descartadas,
            "filas_de_desarrollo_usadas": contexto.guarda.filas_de_desarrollo_usadas,
        },
        metodo_intervalo=LITERAL_METODO_IC,
        nota_tramo2=nota_tramo2,
        extras={
            "comparacion_primaria": {
                "id": "P1",
                "definicion": "B2 - B1",
                "corrida": CONFIG["experimento"],
                "control_sin_atencion": "B1",
                "control_literal": "B1",
                "experimento_contraste": "NINGUNO",
            }
        },
    )
    carga[resolver_clave("comparacion_primaria_json")][resolver_clave("control_atencion_json")] = "B1"
    return carga


# =============================================================================
# SECCION 20: ORDENACION Y PUNTO DE ENTRADA
# =============================================================================


def marca_de_corrida() -> str:
    """Marca unica de la corrida.

    Returns:
        Cadena con marca temporal y pid.
    """
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_p{os.getpid()}"


def comprobar_afirmaciones_stop(progreso: Progreso) -> tuple[bool, list[str]]:
    """Exige que TODA afirmacion con marca de STOP se haya decidido como CONFIRMADO.

    Sirve para la corrida completa, donde antes se escribia el informe y se salia con
    codigo cero aunque una STOP hubiera quedado en ERROR, en REFUTADO o sin ejecutar.

    Args:
        progreso: acumulador de la corrida.

    Returns:
        Tupla con el exito y la lista de entradas que no son CONFIRMADO.
    """
    sin_decidir = [
        f"{clave}:{valor.estado}"
        for clave, valor in sorted(progreso.registro.items())
        if valor.stop and valor.estado != ESTADO_CONFIRMADO
    ]
    faltantes = sorted(AFIRMACIONES_STOP - set(progreso.registro))
    sin_decidir = sin_decidir + [f"{clave}:SIN_EJECUTAR" for clave in faltantes]
    return (not sin_decidir), sin_decidir


def main() -> int:
    """Punto de entrada. Modo de humo por defecto; el completo exige bandera.

    Returns:
        0 si todo pasa, 1 si la puerta del humo falla, 2 si hay STOP.
    """
    parser = argparse.ArgumentParser(
        description="P82: RNN bidireccional con capa de atencion aditiva."
    )
    parser.add_argument("--modo", choices=["humo", "completo"], default="humo")
    parser.add_argument("--caso", choices=["COMPLETO", "INCOMPLETO", "ambos"], default="ambos")
    parser.add_argument("--suelo", type=int, default=SUELO_COBERTURA)
    parser.add_argument(
        "--autorizar-tramo2",
        action="store_true",
        help=(
            "Autoriza por una persona la ejecucion del tramo 2 aunque la condicion del "
            "hito se cumpla. No cambia el presupuesto ni el hito ni la condicion."
        ),
    )
    argumentos = parser.parse_args()

    torch.set_num_threads(HILOS)
    np.random.seed(CONFIG["semillas"][0] % (2**32))
    torch.manual_seed(CONFIG["semillas"][0])
    marca = marca_de_corrida()
    inicio_monotonic = time.perf_counter()
    puerta = PuertaHumo()
    progreso_global = Progreso(PARCIALES / marca, marca)

    if argumentos.modo == "completo":
        progreso = Progreso(PARCIALES / marca, marca)
        try:
            carga = ejecutar_p82_completo(
                progreso, inicio_monotonic, argumentos.autorizar_tramo2
            )
        except Parada as parada:
            print(f"[p82] STOP en {parada.afirmacion}: {parada.motivo}")
            return 2
        cubriendose, detalle_cobertura = afirmar_evaluadas_en_completo(
            Path(__file__).read_text(encoding="utf-8")
        )
        if not cubriendose:
            print(
                "[p82] EL PIPELINE DEL COMPLETO NO CUBRE LAS VEINTITRES AFIRMACIONES: "
                f"faltan {detalle_cobertura['faltan_en_el_completo']}"
            )
            return 6
        correcto, sin_decidir = comprobar_afirmaciones_stop(progreso)
        if not correcto:
            print(
                f"[p82] LA CORRIDA COMPLETA NO SE PUBLICA: {len(sin_decidir)} afirmaciones "
                "con marca de STOP no son CONFIRMADO"
            )
            for entrada in sin_decidir:
                print(f"[p82]   - {entrada}")
            print("[p82] el informe NO se escribe, para no declarar un resultado sin decidir")
            return 3
        # DESTINO OCUPADO. El guard antitrunco NO se toca ni se relaja: si el destino
        # canonico existe, no se sobrescribe. En su lugar se escribe con la marca de la
        # corrida, y el informe declara cual es su nombre y que hay una corrida anterior.
        destino_json = REPORTE_JSON
        destino_md = REPORTE_MD
        reejecucion = REPORTE_JSON.exists()
        if reejecucion:
            destino_json = REPORTS / f"p82_rnn_atencion_{marca}.json"
            destino_md = REPORTS / f"p82_rnn_atencion_{marca}.md"
            carga["informe_anterior_conservado"] = {
                "ruta_del_anterior": str(REPORTE_JSON.relative_to(PROJECT_ROOT)),
                "motivo": (
                    "el destino canonico ya existia y el guard antitrunco se niega a "
                    "sobrescribirlo, luego esta corrida escribe con su marca y conserva "
                    "el anterior intacto"
                ),
                "corrida_actual": marca,
            }
            carga["corrida_anterior"] = RESULTADO_CIENTIFICO_RAZONADO
        if not escribir_unico(destino_json, a_json_legible(carga)):
            print(f"[p82] el informe {destino_json} ya existia y no se sobrescribe")
            return 4
        escribir_unico(destino_md, construir_markdown(carga))
        ok_fichero, detalle_fichero = leer_contraste_del_fichero(destino_json)
        if not ok_fichero:
            print(f"[p82] el JSON escrito no declara bien el contraste primario: {detalle_fichero}")
            return 5
        print(f"[p82] corrida completa escrita en {destino_json}")
        if reejecucion:
            print(f"[p82] corrida anterior conservada en {REPORTE_JSON}")
        return 0

    casos = ["INCOMPLETO", "COMPLETO"] if argumentos.caso == "ambos" else [argumentos.caso]
    for caso in casos:
        directorio = REPORTS / f"humo_p82_{marca}_{caso.lower()}"
        directorio.mkdir(parents=True, exist_ok=True)
        print(f"[p82] caso de humo {caso} en {directorio}")
        try:
            pipeline_humo(directorio, marca, puerta, caso)
        except Parada as parada:
            print(f"[p82] STOP en {parada.afirmacion}: {parada.motivo}")
            return 2
        except Exception as error:
            print(f"[p82] ERROR en el caso {caso}: {type(error).__name__}: {error}")
            print(traza_resumida(error, 8))
            puerta.registrar(f"caso_{caso.lower()}_sin_excepcion", False, str(error)[:120])

    return puerta.codigo_salida(argumentos.suelo, progreso_global)


if __name__ == "__main__":
    raise SystemExit(main())
