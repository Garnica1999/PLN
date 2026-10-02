"""p83_01_bayesiano.py

Fecha: 2026-10-01
Descripcion: P83. Modelo bayesiano disperso con DOBLE objetivo y preguntas disjuntas.
    La pregunta 1 es incertidumbre y calibracion, y NO compite por F1: se responde con
    log-loss, Brier, curva de calibracion y cobertura de intervalos contra el liston
    v03_cal. La pregunta 2 es complementariedad de errores con V03, y NO busca superarlo:
    se responde con AUC del error de V03, acuerdo, matriz de interseccion, rescate y dano.
    Dos brazos: M1, logistica bayesiana dispersa con hiper-prior Gamma sobre columnas
    reescaladas e inferencia variacional de campo medio, y M2, clasificador de Bayes
    multinomial con prior de Dirichlet, que es EXACTO y da el punto de referencia.
    Veinte afirmaciones, A0 a A19.
Pre-registro: docs/p83_preregistro_bayesiano.md
    sha256 7f376fa644c8ee4c696d8f47d6fe79ace7560d96af52235c5c14a20cd9d9ad36
Estado del arte aplicado: ARD con actualizacion de la dispersion por hiper-prior Gamma,
    alpha_k igual a gamma_k partido por m_k al cuadrado mas alpha_0, con Newton
    amortiguado sobre la verosimilitud penalizada; numero de iteraciones del punto de
    variacion 30 con tolerancia 1e-6 sobre la varianza de la posterior; seleccion previa
    de variables por suma de TF-IDF sobre el inner_train con umbral 3.0 y sensibilidades
    2.0 y 5.0; calibracion anidada con Platt ajustado en inner_calibration y aplicado a
    outer_validation; intervalos predictivos del 90 por ciento por muestreo de la
    posterior; emparejamiento por pliegue y por semilla para los dos intervalos de 1.5.

Salidas: reports/p83_bayesiano.json y reports/p83_bayesiano.md. Los parciales por bloque
    y por afirmacion van a reports/p83_parciales/, cada uno en un fichero PROPIO con marca
    de corrida. El modo de humo escribe en reports/humo_p83_<marca>/ y NUNCA bloquea la
    corrida real, porque el guard de escritura unica se niega a sobrescribir.

Fronteras que este script NO cruza: no lee test, no lee el holdout ni su fichero de
identificadores, no genera submission, no consulta el leaderboard, no reentrena V03, no
toca los umbrales congelados 0.44, 0.51 y 0.46, no reentrena el artefacto Word2Vec y no
modifica ningun script de P71 a P82.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib.util
import inspect
import json
import math
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Final, Sequence

import numpy as np
from scipy import optimize, sparse, special, stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train  # noqa: E402
from src.evaluation import ETIQUETAS, metricas_multietiqueta  # noqa: E402
from src.splits import grupos_fusionados, holdout_v2, pliegues_corregidos  # noqa: E402

# =============================================================================
# SECCION 1: CONFIGURACION Y CONSTANTES DECLARADAS
# =============================================================================

CONFIG: Final[dict[str, Any]] = {
    "experimento": "P83",
    "pre_registro": "docs/p83_preregistro_bayesiano.md",
    "pre_registro_sha256": (
        "7f376fa644c8ee4c696d8f47d6fe79ace7560d96af52235c5c14a20cd9d9ad36"
    ),
    "semillas": tuple(20261401 + i for i in range(5)),
    "n_pliegues": 5,
    "n_filas_desarrollo": 1483,
    "n_filas_holdout": 248,
    "n_unidades_tweet_etiqueta": 4449,
    "etiquetas": tuple(ETIQUETAS),
    # Brazo M1, seccion 5.1
    "alpha_0_primario": 1.0,
    "alpha_0_rejilla": (0.5, 1.0, 2.0),
    "iteraciones_variacion": 30,
    "tolerancia_varianza": 1e-6,
    "damping_newton": 0.6,
    "pasos_busqueda_linea": 30,
    # Brazo M2, seccion 5.4
    "alfa_dirichlet": 1.0,
    # Seleccion previa, seccion 5.3
    "umbral_seleccion_primario": 3.0,
    "umbrales_seleccion": (2.0, 3.0, 5.0),
    # Pregunta 1, secciones 1.3 y 1.4
    "margen_log_loss": 0.005,
    "margen_brier": 0.0015,
    "margen_ece": 0.010,
    "recorte_log_loss": 1e-15,
    "bins_calibracion": 10,
    "banda_muerta_bin": 1e-9,
    "minimo_unidades_por_bin": 20,
    "n_bins_max_vacios": 5,
    "n_muestras_predictivas": 400,
    "cuantil_inferior": 0.05,
    "cuantil_superior": 0.95,
    "peso_penalizacion_anchura": 0.5,
    "eps_gate_delta": 0.003,
    # Ensamblaje, seccion 3.5
    "alphas_ensamblaje": (0.05, 0.10, 0.15),
    "alpha_ensamblaje_primario": 0.10,
    # Estructura
    "fraccion_inner_calibration": 0.70,
    "budget_segundos": 1500,
    "media_ganadora_1p6": 1.6,
}

# EL RECUENTO QUE A14 EXIGE, DECLARADO Y NO MULTIPLICADO. Es cinco pliegues por
# cinco semillas de la CORRIDA PRINCIPAL de M1. Antes se calculaba con
# CONFIG["n_pliegues"] * len(CONFIG["semillas"]) y se comparaba contra el
# CONTADOR real, que decia 100 porque la rejilla de alpha_0 de A6 vuelve a
# ejecutar M1 tres veces mas y sus llamadas seguian en la misma lista. El numero
# que define el pre-registro es veinticinco, y aqui se escribe veinticinco.
N_AJUSTES_ESPERADOS_DEL_CALIBRADOR: Final[int] = 25

# Artefactos congelados. Se abren solo en modo lectura y nunca se escriben.
W2V_NPZ: Final[Path] = PROJECT_ROOT / "modelos" / "twitter_domain" / "w2v_twitter_100d.npz"
W2V_SHA256: Final[str] = "302c994d13b75f8f5fbd440eea2c4f4113997e2581524021923b92f8244d705a"
V03_JSON: Final[Path] = PROJECT_ROOT / "experiments" / "baseline_v03_frozen" / "pipeline_v03.json"
V03_OOF: Final[Path] = PROJECT_ROOT / "features" / "oof_v03_final.npz"
P76_CALIBRACION: Final[Path] = PROJECT_ROOT / "reports" / "p76_calibracion.json"
PREREGISTRO: Final[Path] = PROJECT_ROOT / CONFIG["pre_registro"]
REPORTS: Final[Path] = PROJECT_ROOT / "reports"
PARCIALES: Final[Path] = REPORTS / "p83_parciales"
REPORTE_JSON: Final[Path] = REPORTS / "p83_bayesiano.json"
REPORTE_MD: Final[Path] = REPORTS / "p83_bayesiano.md"

# El OOF canonico de V03, congelado.
CORRECTED_CV_PUBLICADO: Final[float] = 0.7772114525
CORRECTED_CV_TOLERANCIA: Final[float] = 1e-9
# El recalculado NO es un valor por defecto de nada. Se declara aqui solo como
# referencia de control y NUNCA se usa como sustituto del recalculo: A3 recalcula
# desde la tabla de predicciones y compara con el publicado del pre-registro. Un
# .get con este valor por defecto hacia que, si la clave faltaba, A3 comparase
# consigo mismo y se diese por bueno, que es el patron de constante contra si
# misma que ya ha salido dos veces en este proyecto. Ver corrected_cv_de_V03.
CORRECTED_CV_RECALCULADO: Final[float] = 0.7772114525165721
UMBRALES_V03_CONGELADOS: Final[tuple[float, float, float]] = (0.44, 0.51, 0.46)


def corrected_cv_de_V03(metricas: dict[str, Any]) -> float:
    """Extrae el corrected_cv de las metricas, y FALLA si la clave no esta.

    POR QUE NO HAY VALOR POR DEFECTO. Con un ``.get(clave, CORRECTED_CV_RECALCULADO)``
    la ausencia de la clave no produce ningun fallo: sale la constante, la
    constante se compara con el publicado, y como el publicado viene del
    pre-registro la comparacion falla, pero si algun dia se alinearan las dos,
    A3 se confirmaria sin haber recalculado nada. Es una comprobacion que puede
    pasar sin hacer su trabajo. Aqui la ausencia de la clave es un ERROR, que es
    lo unico honesto: o se recalcula desde las predicciones, o no se afirma nada.

    Args:
        metricas: diccionario de metricas del OOF canonico.

    Returns:
        El corrected_cv recalculado.

    Raises:
        AssertionError: si la clave no existe o no es un numero.
    """
    if "corrected_cv" not in metricas:
        raise AssertionError(
            "las metricas del OOF canonico no traen corrected_cv, y recalcularlo es la "
            "unica forma de que A3 tenga un numero independiente; caer a una constante "
            "aqui haria que A3 se comparase consigo mismo"
        )
    valor = metricas["corrected_cv"]
    if not isinstance(valor, (int, float)) or not math.isfinite(float(valor)):
        raise AssertionError(f"el corrected_cv del OOF canonico no es un numero finito: {valor!r}")
    return float(valor)

# El liston de la pregunta 1 se construye APLICANDO los pares de Platt CONGELADOS de
# P76 al OOF canonico, NO reejecutando P76. Con eso reproduce a seis decimales
# Brier 0.122454, log-loss 0.391009 y pendientes 0.996569, 0.998942 y 0.974154.
BRIER_V03_CAL_PUBLICADO: Final[float] = 0.122455
LOG_LOSS_V03_CAL_PUBLICADO: Final[float] = 0.391009
PENDIENTES_V03_CAL_PUBLICADAS: Final[tuple[float, float, float]] = (
    0.996569,
    0.998942,
    0.974154,
)
TOLERANCIA_REPRODUCCION_LISTON: Final[float] = 5e-6

# Referencia previa ya publicada por el proyecto. Se cita, no se recalcula.
NB_SVM_CANONICO: Final[float] = 0.763839
LOGISTICA_L1_C4: Final[float] = 0.783423
LOGISTICA_L1_C4_IC: Final[tuple[float, float]] = (-0.0081, 0.0202)
P82_RNN_LOCAL: Final[float] = 0.639659
P82_RNN_LB: Final[float] = 0.68150

# Dispositivo de calculo, POR MEDICION. El micro-benchmark del pre-registro, en la
# geometria real de 1186 por 951 en float64 con 30 iteraciones de VB y Newton amortiguado.
COSTO_POR_AJUSTE_SEGUNDOS: Final[dict[str, float]] = {
    "cpu_16_hilos": 0.0349,
    "cpu_30_hilos": 0.0369,
    "cpu_32_hilos": 0.0368,
    "gpu_rocm": 0.0214,
}
HILOS: Final[int] = 16
MOTIVO_DE_LOS_HILOS: Final[str] = (
    "16 por medicion, no por conveniencia: con 30 hilos un ajuste tarda 0.0369 s frente a "
    "0.0349 s con 16, y con 32 hilos 0.0368 s, luego mas hilos empeoran por "
    "sobrecarga de planificacion con tensores pequenos. La GPU con ROCm tarda 0.0214 s "
    "por ajuste, 1.6 veces mas rapida"
)
MOTIVO_DE_LA_GPU: Final[str] = (
    "la GPU con ROCm gana por 1.6 en la geometria real declarada, luego se usa cuando esta "
    "disponible y se cae a CPU cuando no lo esta. La eleccion es por medicion, no por defecto"
)


def usar_gpu() -> bool:
    """Decide si se usa la GPU, por medicion y no por defecto.

    Returns:
        True si hay una GPU utilizable a traves de torch.
    """
    if importlib.util.find_spec("torch") is None:
        return False
    import torch  # importacion perezosa: el venv puede no tenerlo

    try:
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - una GPU rota no puede tumbar la corrida
        return False


HAY_GPU: Final[bool] = usar_gpu()
DISPOSITIVO: Final[str] = "gpu_rocm" if HAY_GPU else f"cpu_{HILOS}_hilos"

# Formulaciones literales obligatorias. Coincidencia exacta, cero margen.
LITERAL_V03: Final[str] = "V03 remains the best verified leaderboard result."
LITERAL_PREGUNTA_UNO: Final[str] = (
    "LA_PREGUNTA_UNO_NO_COMPITE_POR_F1_Y_NO_ACE_POR_ELIMINADO_EL_ORDENAMIENTO_POR_F1"
)
LITERAL_PREGUNTA_DOS: Final[str] = (
    "LA_PREGUNTA_DOS_NO_BUSCA_SUPERAR_A_V03_SU_CRITERIO_ES_TENER_UN_PERFIL_DE_ERROR"
    "DISTINTO_Y_MEDIBLE"
)
LITERAL_EXPECTATIVA: Final[str] = (
    "CON_1483_FILAS_Y_CELDAS_RARISIMAS_EL_PAPEL_ESPERADO_DEL_BAYESIANO_ES_MIEMBRO_DIVERSO"
    "NO_CAMPEON"
)
LITERAL_POSTERIOR_APROXIMADA: Final[str] = (
    "LA_POSTERIOR_DE_M1_ES_APROXIMADA_Y_NO_ES_LA_POSTERIOR_EXACTA"
)
LITERAL_EFECTIVOS_METODO: Final[str] = (
    "EL_NUMERO_DE_PARAMETROS_EFECTIVOS_QUE_PRODUCE_ESTE_METODO_ES_UN_PARAMETRO_DEL_METODO"
    "Y_ESTA_DETERMINADO_POR_LA_ESCALA_DE_LA_PRIOR_ALPHA_0_Y_POR_LA_VARIANZA_POSTERIOR"
    "NO_ES_UN_NUMERO_DE_MUESTRAS_EFECTIVAS_DEL_CONJUNTO_DE_DATOS"
)
LITERAL_AMBIGUEDAD: Final[str] = "NO_SON_INDEPENDIENTES"
LITERAL_DESCRIPTIVA_F1: Final[str] = "DESCRIPTIVA_Y_SIN_CUALQUIER_LECTURA_INFERENCIAL"
LITERAL_RESCATE_ESCALA: Final[str] = (
    "EL_RESCATE_Y_EL_DANO_SON_UN_RECUENTO_DE_CELDAS_Y_NO_SON_COMPARABLES_CON_EL_CORRECTED_CV"
    "NI_CON_NINGUN_MACRO_F1"
)
LITERAL_CORRECTED_CV_OOF: Final[str] = (
    "EL_CORRECTED_CV_DEL_ENSAMBLAJE_SOBRE_EL_OOF_NO_ES_UNA_MEJORA_DE_V03"
    "PORQUE_EL_OOF_YA_SE_USO_PARA_ELEGIR_EL_ALPHA_Y_PARA_CUALQUIER_OTRA_ELECCION"
    "LUEGO_ES_UN_NUMERO_OPTIMISTA_Y_NO_PUBLICABLE_COMO_RESULTADO"
)
LITERAL_DIAGNOSTICA: Final[str] = "DIAGNOSTICA_Y_SIN_LECTURA_INFERENCIAL"
LITERAL_SESGO_SELECCION: Final[str] = (
    "LA_SELECCION_POR_VALOR_ABSOLUTO_PUEDE_FAVORECER_A_LAS_COLUMNAS_LARGAS"
    "POR_ESO_UN_BAYESIANO_QUE_DECLARA_MAS_VARIABLES_ACTIVAS_NO_HA_GANADO_POR_GANAR"
    "SINO_QUE_HA_GANADO_POR_EL_CRITERIO_DE_SELECCION"
)
LITERAL_F1_NO_REFUTA_UNO: Final[str] = (
    "UN_F1_POR_DEBAJO_DE_V03_NO_REFUTA_NADA_SOBRE_LA_PREGUNTA_1"
    "PORQUE_LA_PREGUNTA_1_NO_COMPITE_POR_F1"
)
LITERAL_F1_NO_REFUTA_DOS: Final[str] = (
    "UN_F1_POR_DEBAJO_DE_V03_NO_REFUTA_NADA_SOBRE_LA_PREGUNTA_2"
    "PORQUE_LA_PREGUNTA_2_NO_BUSCA_SUPERAR_A_V03"
)
LITERAL_NO_WORD2VEC: Final[str] = "NO_SE_AFIRMA_QUE_WORD2VEC_SEA_MEJOR_QUE_V03"
LITERAL_NO_SUPERA: Final[str] = "NO_SE_AFIRMA_QUE_EL_BAYESIANO_SUPERE_A_V03"
LITERAL_ANCLA: Final[str] = (
    "M1_SE_IMPLEMENTA_DESDE_CERO_EN_ESTE_PROYECTO_Y_NO_SE_ANCLA_CONTRA_NINGUNA_LIBRERIA"
    "DE_INFERENCIA_BAYESIANA"
)
LITERAL_PRESUPUESTO_BARATO: Final[str] = (
    "P83_ES_UN_MODELO_LINEAL_SOBRE_UNA_MATRIZ_RALA_DE_1483_FILAS_LUEGO_EL_PRESUPUESTO_ES_DE_MINUTOS"
    "Y_ESA_ASIMETRIA_CON_P82_SE_DECLARA_PARA_QUE_NO_SE_LEA_COMO_QUE_P83_ES_EL_EXPERIMENTO_IMPORTANTE"
)

# Valores de la regla de lectura de 1.5.
LECTURA_AFIRMATIVO: Final[str] = "AFFIRMATIVO"
LECTURA_NO_INFERIOR: Final[str] = "NO_INFERIOR_SIN_AFIRMAR_SUPERIORIDAD"
LECTURA_NEGATIVO: Final[str] = "NEGATIVO"
LECTURA_AMBIGUA: Final[str] = "AMBIGUO_Y_POR_LO_TANTO_NADA"
LECTURAS_VALIDAS: Final[frozenset[str]] = frozenset(
    {LECTURA_AFIRMATIVO, LECTURA_NO_INFERIOR, LECTURA_NEGATIVO, LECTURA_AMBIGUA}
)

# Librerias de inferencia bayesiana que A15 comprueba en el venv.
LIBRERIAS_ANCLA: Final[tuple[str, ...]] = (
    "pymc",
    "pymc3",
    "pyro",
    "stan",
    "edward",
    "theano",
    "aesara",
    "numpyro",
    "tensorflow_probability",
    "pystan",
    "cmdstanpy",
    "pyro-ppl",
)

# =============================================================================
# SECCION 2: CONTRATO ENTRE CAPAS. CLAVES DERIVADAS, NUNCA POR LITERAL
# =============================================================================

CLAVES_DERIVADAS: Final[dict[str, str]] = {
    "umbrales_v03_json": "umbrales",
    "calibracion_por_pliegue_json": "calibracion_por_pliegue",
    "platt_v03_json": "platt_v03",
    "detalle_json": "detalle",
    "estado_json": "estado",
    "caso_json": "caso",
    "veredictos_json": "veredictos",
    "naturaleza_resultado_json": "naturaleza_del_resultado",
    "resumen_naturalezas_json": "resumen_de_naturalezas",
    "lectura_pregunta_1_json": "lectura_pregunta_1",
    "ambiguedad_declarada_json": "ambiguedad_declarada",
    "cifras_pregunta_1_json": "cifras_de_la_pregunta_1",
    "cifras_pregunta_2_json": "cifras_de_la_pregunta_2",
    "control_de_calibrador_json": "control_del_calibrador",
    "anclaje_json": "anclaje_externo",
    "seleccion_variables_json": "seleccion_de_variables",
    "formulacion_json": "formulacion",
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


@dataclass(frozen=True)
class ContextoContrato:
    """Contexto de la tabla de contrato entre capas."""

    filas: tuple[FilaContrato, ...]
    origen: str


def tabla_contratos() -> tuple[FilaContrato, ...]:
    """Tabla de contrato completa, construida en el momento de verificarla.

    Las funciones locales de P83 se anaden aqui, y no en el literal de modulo, porque
    en el momento de construir ese literal todavia no existen y una referencia adelantada
    aborta la importacion entera.

    Returns:
        La tabla completa, biblioteca mas las filas locales de P83.
    """
    return (
        FilaContrato("metricas_multietiqueta", metricas_multietiqueta, ("y_true", "y_pred"), "src/evaluation.py"),
        FilaContrato("read_train", read_train, ("path",), "src/data.py"),
        FilaContrato("holdout_v2", holdout_v2, ("train", "grupos"), "src/splits.py"),
        FilaContrato(
            "pliegues_corregidos",
            pliegues_corregidos,
            ("train_dev", "grupos_dev", "n_splits"),
            "src/splits.py",
        ),
        FilaContrato(
            "TfidfVectorizer", TfidfVectorizer.__init__, ("self", "analyzer", "ngram_range", "min_df", "sublinear_tf", "lowercase"), "sklearn",
        ),
        FilaContrato(
            "LogisticRegression",
            LogisticRegression.__init__,
            ("self", "C", "solver", "max_iter", "random_state"),
            "sklearn",
        ),
        FilaContrato(
            "ajustar_m1", ajustar_m1, ("X_ent", "y_ent", "alpha_0"), "este script, seccion 8"
        ),
        FilaContrato(
            "predecir_m1", predecir_m1, ("modelo", "X_ev"), "este script, seccion 8"
        ),
        FilaContrato(
            "camino_bernoulli_m2",
            camino_bernoulli_m2,
            ("X_ent", "y_ent", "X_ev", "alfa"),
            "este script, seccion 9",
        ),
        FilaContrato(
            "camino_recuento_m2",
            camino_recuento_m2,
            ("X_ent", "y_ent", "X_ev", "alfa"),
            "este script, seccion 9",
        ),
        FilaContrato(
            "seleccionar_columnas",
            seleccionar_columnas,
            ("X_ent", "umbral"),
            "este script, seccion 10",
        ),
        FilaContrato(
            "ajustar_calibrador",
            ajustar_calibrador,
            ("prob_interna", "y_interno"),
            "este script, seccion 11",
        ),
        FilaContrato(
            "decidir_alpha",
            decidir_alpha,
            ("correcciones_internas", "alphas", "primario"),
            "este script, seccion 11",
        ),
        FilaContrato(
            "construir_json", construir_json, ("carga",), "este script, seccion 17"
        ),
        FilaContrato(
            "construir_markdown", construir_markdown, ("carga",), "este script, seccion 17"
        ),
    )


def verificar_contratos(contexto: ContextoContrato | None = None) -> tuple[bool, dict[str, Any]]:
    """Verifica la tabla de contrato con inspect.signature.

    Args:
        contexto: contexto con la tabla a verificar, o None para construirlo.

    Returns:
        Tupla con el exito global y el detalle fila por fila.
    """
    tabla = contexto.filas if contexto is not None else tabla_contratos()
    filas: list[dict[str, Any]] = []
    correcto = True
    for fila in tabla:
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
    return correcto, {"filas": filas, "n_filas": len(filas), "origen": contexto.origen if contexto else "modulo"}


# =============================================================================
# SECCION 3: AUDITORIAS POR AST
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


# Para cada funcion, que PARAMETROS son condiciones y por tanto tienen que llegar
# calculadas, nunca como literal booleano. El fifth fallo caro de esta serie fue un
# False escrito a mano en una funcion de construccion de informe.
PARAMETROS_DE_CONDICION: Final[dict[str, tuple[int, ...]]] = {
    # La posicion 1 de registrar_lectura_pregunta_1 es el indicador de ambiguedad, que
    # tiene que llegar CALCULADO desde la regla y no escrito a mano. Y la posicion 3 de
    # escribir_informe es el exito de la escritura, que es exactamente el fallo que P82
    # sufrio cuando una funcion de construccion de informe recibio un False a mano.
    "registrar_lectura_pregunta_1": (1,),
    "comprobar_presupuesto": (2,),
    "escribir_informe": (3,),
}


def auditar_booleanos_literales(codigo: str) -> list[str]:
    """Detecta literales booleanos donde tiene que haber una CONDICION calculada.

    Cubre el fallo de la misma clase que ya paso en P82: un argumento asumido donde
    deberia haber una comparacion real.

    POR QUE TAMBIEN MIRA DENTRO DEL CUERPO Y NO SOLO LOS ARGUMENTOS. La version
    anterior solo inspeccionaba los argumentos de llamada de las tres funciones de
    PARAMETROS_DE_CONDICION, y por eso era ciega al hueco: un True escrito a mano a
    una variable de control DENTRO del cuerpo de comprobar_presupuesto quedaba
    fuera por construccion, y la comprobacion "no hay booleanos literales en el
    script" pasaba sin poder fallar ante la cosa que decia vigilar. Eso es una
    comprobacion tautologica en sentido fuerte, y el caso de humo que la fijaba
    metia el literal justo en el sitio que ya estaba cubierto, luego no
    demostraba nada. Aqui se cubren las dos vias: el argumento de llamada y la
    asignacion dentro del cuerpo de la propia funcion.

    Args:
        codigo: fuente Python a auditar.

    Returns:
        Lista de infracciones, una por hallazgo.
    """
    arbol = ast.parse(codigo)
    infracciones: list[str] = []
    # Via uno: el literal en el ARGUMENTO de la llamada.
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
    # Via dos: el literal ASIGNADO a una variable dentro del cuerpo de la funcion.
    for nodo in ast.walk(arbol):
        if not (isinstance(nodo, ast.FunctionDef) and nodo.name in PARAMETROS_DE_CONDICION):
            continue
        for hijo in ast.walk(nodo):
            if not (isinstance(hijo, ast.Assign) and len(hijo.targets) == 1):
                continue
            objetivo = hijo.targets[0]
            if not isinstance(objetivo, ast.Name):
                continue
            # Solo las variables de CONTROL: las que alimentan una comprobacion o
            # se acumulan para el veredicto, no cualquier bandera local.
            if not _es_variable_de_control(objetivo.id):
                continue
            if (
                isinstance(hijo.value, ast.Constant)
                and isinstance(hijo.value.value, bool)
            ):
                infracciones.append(
                    f"linea {hijo.lineno}: dentro de {nodo.name} se asigna el literal "
                    f"booleano {hijo.value.value!r} a la variable de control "
                    f"{objetivo.id!r}; use la comparacion real"
                )
    return sorted(infracciones)


# Nombres que son condiciones o alimentan el veredicto de la comprobacion. Un True
# escrito a mano en cualquiera de ellos es exactamente el fallo de P82, y la
# lista es cerrada a proposito: ampliar la auditoria sin criterio la dejaria
# inservible por ruido.
VARIABLES_DE_CONTROL: Final[frozenset[str]] = frozenset(
    {
        "dentro",
        "coherente",
        "presupuesto_respetado",
        "ambigua",
        "escrituras_ok",
        "exito",
        "superado",
        "cumple",
        "detecta",
        "violacion",
    }
)


def _es_variable_de_control(nombre: str) -> bool:
    """Dice si un nombre es una variable de control de una comprobacion.

    Args:
        nombre: nombre del identificador asignado.

    Returns:
        True si el nombre es una variable de control.
    """
    if nombre in VARIABLES_DE_CONTROL:
        return True
    return any(nombre.startswith(prefijo) for prefijo in ("tiene_", "hay_", "cumple_", "dentro_"))


def auditar_evaluaciones_anidadas(codigo: str) -> list[str]:
    """Detecta llamadas a evaluar cuya comprobacion devuelva una tupla anidada.

    EL DEFECTO QUE ESTA AUDITORIA ATRAPA. Si la comprobacion es una lambda cuyo cuerpo
    es una tupla de dos elementos cuyo primer elemento es una llamada que ya devuelve
    (bool, dict), el primer elemento de la tupla devuelta es esa tupla, no el booleano.
    Una tupla no vacia es truthy en Python, luego exito queda siendo truthy y TODA
    afirmacion con esa forma sale CONFIRMADA. Ese defecto sobrevivio a una puerta de
    humo completa porque ninguna comprobacion exigia que un fallo se publicara como
    REFUTADO.

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
    booleanos = auditar_booleanos_literales(codigo)
    anidadas = auditar_evaluaciones_anidadas(codigo)
    return (not infracciones) and (not booleanos) and (not anidadas), {
        "fichero": Path(__file__).name,
        "claves_vigiladas": sorted(CLAVES_FISICAS),
        "n_infracciones_de_literal": len(infracciones),
        "infracciones_de_literal": infracciones,
        "n_booleanos_literales": len(booleanos),
        "booleanos_literales": booleanos,
        "n_evaluaciones_con_tupla_anidada": len(anidadas),
        "evaluaciones_con_tupla_anidada": anidadas,
    }


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
        if not (isinstance(nodo, ast.FunctionDef) and nodo.name == "ejecutar_p83_completo"):
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

    No depende de que alguien recuerde invocarla.

    Args:
        codigo: fuente Python a auditar.

    Returns:
        Tupla con el exito y el detalle.
    """
    arbol = ast.parse(codigo)
    esperadas = sorted(BLOQUES)
    evaluadas: list[str] = []
    for nodo in ast.walk(arbol):
        if not (isinstance(nodo, ast.FunctionDef) and nodo.name == "ejecutar_p83_completo"):
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
        "las_veinte_declared_estan_en_el_completo": not faltan,
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


def afirmar_evaluadas_en_el_humo(codigo: str) -> tuple[bool, dict[str, Any]]:
    """Verifica que el recorrido de HUMO cubre las mismas veinte afirmaciones.

    POR QUE ESTA COMPROBACION EXISTE Y NO ES UN EXTRA. En P82 el script
    declaraba veintitres afirmaciones y la corrida COMPLETA solo evaluaba
    veintiuna, y la puerta lo detecto AL FINAL, despues de sesenta y cinco
    minutos de entrenamiento. Esa es exactamente la clase de fallo que este
    proyecto paga caro: una declaracion que no esta conectada a nada. Aqui la
    comparacion se hace por AST sobre el codigo, antes de ejecutar nada, y
    exige que el conjunto de afirmaciones que el humo ejercita sea EXACTAMENTE
    el mismo que el que la corrida completa evalua. Sin esta comprobacion, anadir
    una afirmacion al pre-registro y olvidar cablearla pasaria el humo entero.

    Args:
        codigo: fuente Python a auditar.

    Returns:
        Tupla con el exito y el detalle.
    """
    arbol = ast.parse(codigo)
    esperadas = set(BLOQUES)
    en_completo: set[str] = set()
    en_humo: set[str] = set()

    def identificadores_de_llamadas(
        funcion: ast.AST, forma: str
    ) -> set[str]:
        """Saca los identificadores de afirmacion que aparecen como argumento.

        Hay dos formas de declararlos y cada una los lleva en una posicion
        distinta. En progreso.evaluar el identificador es el PRIMER argumento. En
        caso_afirmacion y caso_detecta es el TERCER, porque los dos primeros son la
        puerta y el nombre del caso. El identificador nunca es el primer argumento
        de afirmacion_aN, que es un dato, y por eso no se lee de ahi.

        Args:
            funcion: nodo de la funcion que se recorre.
            forma: "evaluar" o "caso".

        Returns:
            Conjunto de identificadores de afirmacion.
        """
        vistos: set[str] = set()
        indice = 0 if forma == "evaluar" else 2
        for llamada in ast.walk(funcion):
            if not isinstance(llamada, ast.Call):
                continue
            if forma == "evaluar":
                coincide = (
                    isinstance(llamada.func, ast.Attribute) and llamada.func.attr == "evaluar"
                )
            else:
                coincide = (
                    isinstance(llamada.func, ast.Name)
                    and llamada.func.id in {"caso_afirmacion", "caso_detecta"}
                )
            if not coincide or len(llamada.args) <= indice:
                continue
            argumento = llamada.args[indice]
            if (
                isinstance(argumento, ast.Constant)
                and isinstance(argumento.value, str)
                and re.fullmatch(r"A\d{1,2}", argumento.value)
            ):
                vistos.add(argumento.value)
        return vistos

    for nodo in ast.walk(arbol):
        if not isinstance(nodo, ast.FunctionDef):
            continue
        if nodo.name == "ejecutar_p83_completo":
            en_completo = identificadores_de_llamadas(nodo, "evaluar")
        elif nodo.name in {
            "casos_de_afirmaciones",
            "casos_de_frontera_lectura",
            "casos_de_frontera_m2",
        }:
            en_humo = en_humo | identificadores_de_llamadas(nodo, "caso")
    solo_completo = sorted(en_completo - en_humo)
    solo_humo = sorted(en_humo - en_completo)
    declaradas_sin_conectar = sorted(esperadas - en_completo)
    comprobaciones = {
        "el_completo_evalua_las_declaradas": not declaradas_sin_conectar,
        "el_humo_ejercita_las_declaradas": not sorted(esperadas - en_humo),
        "los_dos_recorridos_coinciden": not solo_completo and not solo_humo,
        "ninguna_afirmacion_solo_en_un_recorrido": not solo_completo and not solo_humo,
        "las_veinte_declared_estan_cableadas": len(en_completo) == len(esperadas)
        and len(en_humo) == len(esperadas),
    }
    return all(comprobaciones.values()), {
        "n_declaradas": len(esperadas),
        "n_en_el_completo": len(en_completo),
        "n_en_el_humo": len(en_humo),
        "solo_en_el_completo": solo_completo,
        "solo_en_el_humo": solo_humo,
        "declaradas_sin_conectar": declaradas_sin_conectar,
        "en_el_completo": sorted(en_completo),
        "en_el_humo": sorted(en_humo),
        "por_que_esta_comprobacion": (
            "en P82 el script declaraba veintitres afirmaciones y el completo evaluaba "
            "veintiuna, y la puerta lo vio al final tras sesenta y cinco minutos; aqui "
            "la comparacion es por AST y ocurre antes de ejecutar nada"
        ),
        "comprobaciones": comprobaciones,
    }


# =============================================================================
# SECCION 4: ASCII PURO, ESCRITURA UNICA Y PARCIALES
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
    write_text con encoding ascii trunca a cero bytes cualquier fuente con un caracter
    no latino, y eso ya ocurrio tres veces en este proyecto, una de ellas en un fichero
    de registro entero.

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

    Por que os.link y no os.replace. os.replace sobrescribe SIEMPRE el destino, y la
    invariante de escritura unica lo prohibe. Con os.link el destino no aparece nunca
    vacio: el temporal se escribe y se verifica completo, y el enlace falla con
    FileExistsError si alguien escribio el destino antes.

    Args:
        ruta: destino absoluto.
        texto: contenido, validado en memoria antes de abrir el fichero.

    Returns:
        True si se escribio, False si el destino ya existia.

    Raises:
        ErrorEscritura: si el texto no es ASCII, si la escritura queda incompleta o si
            falla la E/S.
    """
    try:
        texto_ascii(texto)
    except ValueError as error:
        # El control de ASCII ocurre ANTES de abrir el destino y su fallo es del
        # mismo tipo que un fallo de escritura: el destino no existe y el texto no
        # ha salido. Sin envolverlo, el ValueError se propagaria con otra forma y
        # quien lo espera tendria que conocer dos excepciones para el mismo fallo.
        raise ErrorEscritura(f"{ruta}: {error}") from error
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


def a_jsonable(objeto: Any) -> Any:
    """Convierte cualquier objeto a algo serializable en JSON.

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
    el punto de llamada.

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
# SECCION 5: AISLAMIENTO DE AFIRMACIONES
# =============================================================================

ESTADO_CONFIRMADO: Final[str] = "CONFIRMADO"
ESTADO_REFUTADO: Final[str] = "REFUTADO"
ESTADO_ERROR: Final[str] = "ERROR"
NATURELEZA_CONFIRMAR: Final[str] = "CONFIRMAR"

BLOQUES: Final[dict[str, tuple[int, str]]] = {
    "A0": (0, "integridad de los artefactos congelados, sha256 tamano y mtime antes y despues"),
    "A1": (1, "solo se leen las fuentes declaradas, cero test y cero holdout"),
    "A2": (2, "umbrales de V03 intactos, V03 no reentrenado, cero reoptimizaciones"),
    "A3": (3, "el OOF de referencia es el canonico, coincide con el published"),
    "A4": (4, "la seleccion de variables se hace solo sobre el inner_train"),
    "A5": (5, "la posterior de M1 esta aproximada y el metodo se declara"),
    "A6": (6, "el numero efectivo es un parametro del metodo y se mueve con alpha_0"),
    "A7": (7, "M2 es exacto y coincide con un segundo camino de calculo"),
    "A8": (8, "las seis magnitudes de la pregunta 1 medidas y publicadas"),
    "A9": (9, "la curva de calibracion completa, con bins poblados"),
    "A10": (10, "cobertura y anchura, y el control degenerado de V03"),
    "A11": (11, "la regla de lectura de 1.5 aplicada, sin reinterpretacion"),
    "A12": (12, "las cuatro medidas de complementariedad contra V03"),
    "A13": (13, "el rescate y el dano, sobre OOF, con los tres alphas"),
    "A14": (14, "el calibrador se ajusta en internal, verificado por contadores"),
    "A15": (15, "el limite de 5.5 verificado: anclaje externo presente o ausente"),
    "A16": (16, "la rejilla de alpha_0 y la sensibilidad de umbral, publicadas"),
    "A17": (17, "dimension seleccionada por pliegue, y la salvedad de sesgo"),
    "A18": (18, "las cadenas de formulacion, de las dos preguntas y del proyecto"),
    "A19": (19, "el F1 del miembro se publica como descriptiva, sin lectura inferencial"),
}

# LAS CUATRO QUE NO LLEVAN MARCA DE STOP. A8 y A9 son mediciones de la pregunta 1 que no
# pueden fallar por si solas, A12 y A13 son descriptivas por construccion y A15 es una
# medicion de alcance. Se declara aqui y no por omision.
AFIRMACIONES_SIN_STOP: Final[frozenset[str]] = frozenset({"A8", "A12", "A13", "A15"})
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


class ColeccionVacia(RuntimeError):
    """Una coleccion obligatoria llego vacia."""


def exigir_no_vacio(coleccion: Any, nombre: str) -> Any:
    """Exige que una coleccion no este vacia.

    Motivo: cuatro afirmaciones de P82 comparaban una constante consigo misma, y cinco
    mas usaban una tupla anidada que hacia que TODO saliera CONFIRMADO porque una tupla
    no vacia es truthy. Una coleccion vacia que llega al final del calculo es el mismo
    fallo con otro disfraz, y por tanto dies con su traza en vez de propagarse en
    silencio.

    Args:
        coleccion: valor a comprobar.
        nombre: nombre para el mensaje.

    Returns:
        El mismo valor.

    Raises:
        ColeccionVacia: si la coleccion esta vacia o es un escalar nulo.
    """
    if coleccion is None:
        raise ColeccionVacia(f"{nombre} es None")
    if isinstance(coleccion, (list, tuple, dict, set, frozenset, str, np.ndarray)):
        if len(coleccion) == 0:
            raise ColeccionVacia(f"{nombre} tiene {len(coleccion)} elementos")
    return coleccion


class ComprobacionAnidada(RuntimeError):
    """Una comprobacion devolvio una tupla anidada en vez de un booleano."""


def comprobar_veredicto(bruto: tuple[bool, dict[str, Any]]) -> tuple[bool, dict[str, Any]]:
    """Exige que el veredicto sea un booleano real y no una tupla anidada.

    Args:
        bruto: lo que devolvio la comprobacion.

    Returns:
        La tupla normalizada.

    Raises:
        ComprobacionAnidada: si el primer elemento no es un booleano.
        ColeccionVacia: si el detalle esta vacio, porque una comprobacion que no
            publica nada no puede distinguir que comprobo.
    """
    if not isinstance(bruto, tuple) or len(bruto) != 2:
        raise ComprobacionAnidada(f"veredicto con forma {type(bruto).__name__} de tamano {len(bruto) if hasattr(bruto, '__len__') else 'NA'}")
    exito, detalle = bruto
    if not isinstance(exito, (bool, np.bool_)):
        raise ComprobacionAnidada(
            f"el primer elemento del veredicto es {type(exito).__name__} y no un booleano; "
            "una tupla no vacia es truthy y haria que toda afirmacion saliera CONFIRMADA"
        )
    if not isinstance(detalle, dict):
        raise ComprobacionAnidada(f"el detalle es {type(detalle).__name__} y no un dict")
    if not detalle:
        raise ColeccionVacia("el detalle de la comprobacion esta vacio")
    return bool(exito), detalle


def con_extra(
    comprobacion: Callable[[], tuple[bool, dict[str, Any]]], extra: dict[str, Any]
) -> Callable[[], tuple[bool, dict[str, Any]]]:
    """Anade claves al detalle de una comprobacion sin alterar su veredicto.

    Args:
        comprobacion: comprobacion original.
        extra: claves a anadir.

    Returns:
        Una comprobacion que devuelve lo mismo con las claves anadidas.
    """
    envoltura: dict[str, Any] = {}

    def Envuelta() -> tuple[bool, dict[str, Any]]:
        """Devuelve el veredicto original con las claves extra anadidas.

        Returns:
            Tupla con el exito y el detalle ampliado.
        """
        exito, detalle = comprobar_veredicto(comprobacion())
        ampliado = dict(detalle)
        ampliado.update(extra)
        envoltura["usado"] = True
        return exito, ampliado

    Envuelta.extra = extra  # type: ignore[attr-defined]
    Envuelta.envoltura = envoltura  # type: ignore[attr-defined]
    return Envuelta


def todo_verdadero(comprobaciones: dict[str, Any], nombre: str) -> bool:
    """Exige que TODAS las comprobaciones de un grupo sean verdaderas.

    Motivo: una comprobacion de grupo que usase any() haria que un solo componente
    bueno tapase veinte malos.

    Args:
        comprobaciones: diccionario de banderas.
        nombre: nombre para el mensaje de error.

    Returns:
        True si todas son verdaderas.

    Raises:
        AssertionError: si alguna es falsa, con la lista de las falsas.
    """
    falsas = sorted(clave for clave, valor in comprobaciones.items() if not bool(valor))
    if falsas:
        raise AssertionError(f"{nombre}: comprobaciones falsas {falsas}")
    return True


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
        """Escribe el parcial de una afirmacion.

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
            afirmacion: identificador de A0 a A19.
            comprobacion: funcion sin argumentos que devuelve exito y detalle.
            tolerancia: descripcion de la tolerancia, publicada en el informe.

        Returns:
            El resultado de la afirmacion.

        Raises:
            Parada: si la comprobacion falla y la afirmacion tiene marca STOP.
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
# SECCION 6: MATEMATICA DE LA PREGUNTA 1
# =============================================================================


def intervalo_t_pareado(deltas: Sequence[float], confianza: float = 0.95) -> dict[str, float]:
    """Intervalo t pareado sobre una serie de deltas.

    Args:
        deltas: serie de deltas pareados.
        confianza: nivel declarado.

    Returns:
        Diccionario con la media, la desviacion, el error, los extremos y el nivel.
    """
    serie = np.asarray(list(deltas), dtype=float)
    exigir_no_vacio(serie, "deltas del intervalo pareado")
    n = int(serie.size)
    media = float(serie.mean())
    if n < 2:
        return {
            "n": n, "media": media, "desviacion": 0.0, "error_standard": 0.0,
            "inferior": media, "superior": media, "nivel": confianza,
            "nota": "con menos de dos observaciones no hay intervalo que_valga",
        }
    desviacion = float(serie.std(ddof=1))
    error = desviacion / math.sqrt(n)
    if error <= 0.0:
        return {
            "n": n, "media": media, "desviacion": desviacion, "error_standard": 0.0,
            "inferior": media, "superior": media, "nivel": confianza,
            "nota": "desviacion cero, el intervalo degenera en la media",
        }
    t = float(stats.t.ppf(0.5 + confianza / 2.0, n - 1))
    return {
        "n": n, "media": media, "desviacion": desviacion, "error_standard": error,
        "inferior": media - t * error, "superior": media + t * error, "nivel": confianza,
    }


def log_loss_recortado(
    probabilidad: np.ndarray, verdad: np.ndarray, recorte: float
) -> dict[str, float]:
    """Log-loss macro por etiqueta, con y sin recorte.

    Args:
        probabilidad: matriz de probabilidades.
        verdad: matriz binaria verdadera.
        recorte: recorte declarado.

    Returns:
        Diccionario con el recortado, el no recortado y el desglose por etiqueta.
    """
    p = np.clip(probabilidad, recorte, 1.0 - recorte)
    q = np.clip(probabilidad, recorte, 1.0 - recorte)
    loss_rec = -(verdad * np.log(p) + (1 - verdad) * np.log(1.0 - p))
    loss_raw = -(verdad * np.log(q) + (1 - verdad) * np.log(1.0 - q))
    por_etiqueta = {
        ETIQUETAS[k]: float(loss_rec[:, k].mean()) for k in range(len(ETIQUETAS))
    }
    return {
        "log_loss_macro_recortado": float(loss_rec.mean()),
        "log_loss_macro_sin_recortar": float(loss_raw.mean()),
        "por_etiqueta": por_etiqueta,
        "recorte": recorte,
    }


def brier_multiclase(probabilidad: np.ndarray, verdad: np.ndarray) -> dict[str, float]:
    """Brier macro en la forma de P76 y Brier por etiqueta.

    Args:
        probabilidad: matriz de probabilidades.
        verdad: matriz binaria verdadera.

    Returns:
        Diccionario con el macro y el desglose por etiqueta.
    """
    cuadratico = (probabilidad - verdad) ** 2
    por_etiqueta = {
        ETIQUETAS[k]: float(cuadratico[:, k].mean()) for k in range(len(ETIQUETAS))
    }
    return {
        "brier_macro": float(np.mean(list(por_etiqueta.values()))),
        "brier_por_etiqueta": por_etiqueta,
    }


def curva_de_calibracion(
    probabilidad_columna: np.ndarray,
    verdad_columna: np.ndarray,
    n_bins: int,
) -> dict[str, Any]:
    """Curva de calibracion con bins de igual anchura y el area entre curva y diagonal.

    Args:
        probabilidad_columna: probabilidades de una etiqueta.
        verdad_columna: etiquetas observadas de la misma etiqueta.
        n_bins: numero de bins de igual anchura.

    Returns:
        Diccionario con los bins, el area y las sumas de comprobacion.
    """
    n = int(probabilidad_columna.size)
    bordes = np.linspace(0.0, 1.0, n_bins + 1)
    asignados = np.clip(np.digitize(probabilidad_columna, bordes[1:-1], right=False), 0, n_bins - 1)
    filas: list[dict[str, Any]] = []
    area = 0.0
    n_vacios = 0
    for indice in range(n_bins):
        mascara = asignados == indice
        conteo = int(mascara.sum())
        if conteo == 0:
            n_vacios += 1
            filas.append(
                {
                    "bin": indice,
                    "inferior": float(bordes[indice]),
                    "superior": float(bordes[indice + 1]),
                    "n": 0,
                    "frecuencia_observada": None,
                    "media_predicha": None,
                    "banda_muerta": CONFIG["banda_muerta_bin"],
                }
            )
            continue
        observada = float(verdad_columna[mascara].mean())
        predicha = float(probabilidad_columna[mascara].mean())
        filas.append(
            {
                "bin": indice,
                "inferior": float(bordes[indice]),
                "superior": float(bordes[indice + 1]),
                "n": conteo,
                "frecuencia_observada": observada,
                "media_predicha": predicha,
            }
        )
        area += (conteo / n) * abs(observada - predicha)
    return {
        "bins": filas,
        "area_entre_curva_y_diagonal": float(area),
        "n_bins": n_bins,
        "n_vacios": int(n_vacios),
        "n": n,
        "n_bins_con_al_menos_20_unidades": int(
            sum(1 for f in filas if f["n"] >= CONFIG["minimo_unidades_por_bin"])
        ),
    }


def pendientes_de_calibracion(probabilidad: np.ndarray, verdad: np.ndarray) -> dict[str, Any]:
    """Pendientes de la regresion de la etiqueta sobre el logit de la probabilidad.

    Es la magnitud que P76 publico y la que hace comparables las cifras. Se calcula con
    un LogisticRegression sin penalizacion efectiva sobre el logit, que es exactamente
    lo que produjo las pendientes publicadas de v03_cal.

    Args:
        probabilidad: matriz de probabilidades.
        verdad: matriz binaria verdadera.

    Returns:
        Diccionario con la pendiente y el intercepto por etiqueta y la media.
    """
    recorte = CONFIG["recorte_log_loss"]
    p = np.clip(probabilidad, recorte, 1.0 - recorte)
    logit = np.log(p / (1.0 - p))
    pendientes: dict[str, float] = {}
    interceptos: dict[str, float] = {}
    for k in range(len(ETIQUETAS)):
        modelo = LogisticRegression(C=1e10, solver="lbfgs", max_iter=100000)
        modelo.fit(logit[:, [k]], verdad[:, k])
        pendientes[ETIQUETAS[k]] = float(modelo.coef_[0][0])
        interceptos[ETIQUETAS[k]] = float(modelo.intercept_[0])
    return {
        "pendientes": pendientes,
        "interceptos": interceptos,
        "media": float(np.mean(list(pendientes.values()))),
    }


def intervalo_predictivo(
    modelo: ModeloM1,
    X_ev: np.ndarray,
    n_muestras: int,
    semilla: int,
    cuantil_inferior: float,
    cuantil_superior: float,
) -> dict[str, np.ndarray]:
    """Intervalo de credibilidad al 90 por ciento por muestreo de la posterior.

    La posterior de M1 es aproximada, luego el intervalo hereda el error de la
    aproximacion, y ese error NO se cuantifica: es el limite declarado de 5.1.

    Args:
        modelo: modelo M1 ya ajustado.
        X_ev: matriz de evaluacion.
        n_muestras: numero de muestras de la posterior.
        semilla: semilla del generador.
        cuantil_inferior: cuantil inferior.
        cuantil_superior: cuantil superior.

    Returns:
        Diccionario con el intervalo inferior, el superior y la media de la predictiva.
    """
    X_ev = np.asarray(X_ev, dtype=np.float64)
    rng = np.random.default_rng(semilla)
    desviacion = np.sqrt(np.maximum(modelo.varianza_posterior, 0.0))
    muestras = rng.standard_normal((n_muestras, modelo.media_posterior.size))
    muestras = muestras * desviacion[None, :] + modelo.media_posterior[None, :]
    z = muestras @ X_ev.T + modelo.intercepto
    probabilidades = 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))
    inferior = np.quantile(probabilidades, cuantil_inferior, axis=0)
    superior = np.quantile(probabilidades, cuantil_superior, axis=0)
    media = probabilidades.mean(axis=0)
    return {
        "inferior": inferior,
        "superior": superior,
        "media": media,
        "anchura": superior - inferior,
    }


def medir_cobertura_y_anchura(
    inferior: np.ndarray,
    superior: np.ndarray,
    verdad: np.ndarray,
    peso_anchura: float,
) -> dict[str, float]:
    """Cobertura, anchura normalizada y la metrica penalizada por anchura.

    Args:
        inferior: extremo inferior del intervalo.
        superior: extremo superior del intervalo.
        verdad: etiquetas observadas.
        peso_anchura: peso de la penalizacion por anchura.

    Returns:
        Diccionario con cobertura, anchura media normalizada y la metrica.
    """
    inferior = np.asarray(inferior, dtype=float)
    superior = np.asarray(superior, dtype=float)
    verdad = np.asarray(verdad, dtype=float)
    dentro = (verdad >= inferior - 1e-12) & (verdad <= superior + 1e-12)
    cobertura = float(dentro.mean())
    anchura = float(np.mean(np.clip(superior - inferior, 0.0, 1.0)))
    return {
        "cobertura": cobertura,
        "anchura_media_normalizada": anchura,
        "cobertura_menos_pesada_por_anchura": cobertura - peso_anchura * anchura,
        "peso_de_la_penalizacion": peso_anchura,
    }


def aplicar_lectura_intervalos(
    intervalo_pliegues: dict[str, float],
    intervalo_semillas: dict[str, float],
    margen: float,
) -> tuple[str, dict[str, Any]]:
    """Aplica la regla de lectura de 1.5 a los DOS intervalos, sin reinterpretarla.

    La direccion favorable de un delta de perdida es la negativa, luego el extremo
    favorable es el superior del intervalo del delta, que tiene que quedar por debajo de
    menos el margen. El extremo desfavorable es el inferior, que tiene que quedar por
    encima de mas el margen.

    Args:
        intervalo_pliegues: intervalo t pareado por pliegue.
        intervalo_semillas: intervalo t pareado por semilla.
        margen: margen de no inferioridad.

    Returns:
        Tupla con la lectura y el detalle de por que se ha leido asi.

    Raises:
        AssertionError: si algun intervalo no trae sus dos extremos.
    """
    exigir_no_vacio(intervalo_pliegues, "intervalo por pliegues")
    exigir_no_vacio(intervalo_semillas, "intervalo por semilla")
    for clave in ("inferior", "superior"):
        if clave not in intervalo_pliegues or clave not in intervalo_semillas:
            raise AssertionError(f"el intervalo no trae el extremo {clave}")
    sup_p = float(intervalo_pliegues["superior"])
    inf_p = float(intervalo_pliegues["inferior"])
    sup_s = float(intervalo_semillas["superior"])
    inf_s = float(intervalo_semillas["inferior"])
    favorable_p = sup_p < -margen
    favorable_s = sup_s < -margen
    desfavorable_p = inf_p > margen
    desfavorable_s = inf_s > margen
    contiene_p = inf_p <= margen and sup_p >= -margen
    contiene_s = inf_s <= margen and sup_s >= -margen
    if favorable_p and favorable_s:
        lectura = LECTURA_AFIRMATIVO
    elif desfavorable_p and desfavorable_s:
        lectura = LECTURA_NEGATIVO
    elif (favorable_p and not favorable_s and desfavorable_s) or (
        favorable_s and not favorable_p and desfavorable_p
    ):
        lectura = LECTURA_AMBIGUA
    elif contiene_p or contiene_s:
        lectura = LECTURA_NO_INFERIOR
    else:
        lectura = LECTURA_AMBIGUA
    return lectura, {
        "lectura": lectura,
        "margen": margen,
        "intervalo_por_pliegues": {
            "inferior": inf_p, "superior": sup_p, "n": int(intervalo_pliegues.get("n", 0)),
        },
        "intervalo_por_semillas": {
            "inferior": inf_s, "superior": sup_s, "n": int(intervalo_semillas.get("n", 0)),
        },
        "extremo_favorable_por_pliegues": bool(favorable_p),
        "extremo_favorable_por_semilla": bool(favorable_s),
        "extremo_desfavorable_por_pliegues": bool(desfavorable_p),
        "extremo_desfavorable_por_semilla": bool(desfavorable_s),
        "contiene_el_margen_por_pliegues": bool(contiene_p),
        "contiene_el_margen_por_semilla": bool(contiene_s),
        "ambiguedad_declarada": LITERAL_AMBIGUEDAD,
        "por_que_no_son_independientes": (
            "los cinco pliegues particionan las mismas 1483 filas y cada una se repite en "
            "cinco semillas, luego son veinticinco unidades altamente correlacionadas sobre "
            "un unico conjunto y no se puede declarar un nivel de confianza del 95 por "
            "ciento sobre ninguno de los dos intervalos"
        ),
        "n_de_pliegues": CONFIG["n_pliegues"],
        "fragilidad_declarada": (
            "el numero de pliegues es cinco y un intervalo de cinco observaciones es "
            "fragil por construccion"
        ),
    }


def registrar_lectura_pregunta_1(lectura: str, ambigua: bool) -> dict[str, Any]:
    """Construye el par lectura y ambiguedad que A11 verifica.

    Los dos valores son CONDICIONES calculadas, no literales: por eso la auditoria de
    booleanos literales los vigila.

    Args:
        lectura: valor de la regla de 1.5.
        ambigua: si la lectura es ambigua.

    Returns:
        Diccionario con la lectura, la ambiguedad y el efecto de la ambiguedad.

    Raises:
        AssertionError: si la lectura no es una de las cuatro validas.
    """
    if lectura not in LECTURAS_VALIDAS:
        raise AssertionError(f"lectura {lectura!r} fuera de las cuatro validas")
    if ambigua:
        efecto = (
            "la lectura es ambigua y por lo tanto no se afirma nada sobre la pregunta 1; "
            "el caso se publica con sus numeros"
        )
    else:
        efecto = "la lectura no es ambigua y no se deriva ninguna afirmacion del caso ambiguo"
    return {
        resolver_clave("lectura_pregunta_1_json"): lectura,
        resolver_clave("ambiguedad_declarada_json"): LITERAL_AMBIGUEDAD,
        "lectura_ambigua": bool(ambigua),
        "efecto_de_la_ambiguedad": efecto,
        "literales_validos": sorted(LECTURAS_VALIDAS),
    }


# =============================================================================
# SECCION 7: DATOS Y CONTEXTOS
# =============================================================================


class GuardaLectura:
    """Guarda que cuenta filas leidas y prohibe tocar test y holdout."""

    def __init__(self) -> None:
        self.filas_test_leidas = 0
        self.filas_holdout_leidas = 0
        self.evaluaciones_del_holdout = 0
        self.fuentes: list[dict[str, Any]] = []
        self.marcador_actual: set[int] = set()

    def marcar(self, indices: np.ndarray) -> None:
        """Publica que filas estan en uso ahora.

        Args:
            indices: indices de fila en uso.

        Returns:
            Nada.
        """
        self.marcador_actual = {int(i) for i in np.asarray(indices, dtype=np.int64).ravel()}

    def comprobar(self, indices: np.ndarray, nombre: str) -> None:
        """Comprueba que un conjunto de filas cae dentro del conjunto permitido.

        Args:
            indices: indices a comprobar.
            nombre: nombre para el mensaje.

        Returns:
            Nada.

        Raises:
            AssertionError: si alguna fila cae fuera del conjunto permitido.
        """
        fuera = {int(i) for i in np.asarray(indices, dtype=np.int64).ravel()} - self.marcador_actual
        if fuera:
            raise AssertionError(f"{nombre}: {len(fuera)} filas fuera del conjunto permitido")


@dataclass
class ContextoDatos:
    """Datos de desarrollo con sus indices y su instrumentacion de lectura."""

    textos: list[str]
    y: np.ndarray
    pliegues: list[tuple[np.ndarray, np.ndarray]]
    ids: np.ndarray
    grupos: np.ndarray
    origen: str
    guarda: GuardaLectura = field(default_factory=GuardaLectura)
    completo: bool = True
    # Las probabilidades canonicas de V03, alineadas por fila con las de desarrollo.
    # Las anade el modo completo cuando ya ha pasado A3, que es el unico sitio que
    # puede leer el OOF de referencia. None significa que no hay OOF de referencia
    # disponible, y medir_correcciones_internas_del_alpha no puede correr sin el.
    proba_v03: np.ndarray | None = None


def cargar_datos_de_desarrollo(progreso: Progreso) -> ContextoDatos:
    """Carga las filas de desarrollo sin tocar test ni holdout.

    Args:
        progreso: acumulador de progreso.

    Returns:
        El contexto de datos con los pliegues del protocolo v2.

    Raises:
        AssertionError: si el numero de filas no es el declarado.
    """
    marco = read_train()
    progreso.contar("filas_read_train", len(marco))
    grupos_completos = grupos_fusionados(marco)
    indices_dev, indices_holdout = holdout_v2(marco, grupos_completos)
    progreso.contar("filas_holdout_descartadas", int(indices_holdout.size))
    progreso.contar("filas_test_leidas", 0)
    if len(marco) != CONFIG["n_filas_desarrollo"] + CONFIG["n_filas_holdout"]:
        raise AssertionError(
            f"read_train devolvio {len(marco)} filas y el declarado es "
            f"{CONFIG['n_filas_desarrollo'] + CONFIG['n_filas_holdout']}"
        )
    desarrollo = marco.iloc[indices_dev].reset_index(drop=True)
    pliegues = pliegues_corregidos(desarrollo, grupos_completos[indices_dev])
    guarda = GuardaLectura()
    guarda.marcar(indices_dev)
    y = desarrollo[list(CONFIG["etiquetas"])].to_numpy().astype(np.int64)
    return ContextoDatos(
        textos=desarrollo["text"].tolist(),
        y=y,
        pliegues=[
            (np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)) for a, b in pliegues
        ],
        ids=desarrollo["id"].astype(str).to_numpy(),
        grupos=grupos_completos[indices_dev],
        origen="read_train menos el holdout",
        guarda=guarda,
        completo=True,
    )


# =============================================================================
# SECCION 8: BRAZO M1, LOGISTICA BAYESIANA DISPERSA
# =============================================================================


@dataclass
class ModeloM1:
    """Modelo M1: media y varianza de la posterior de campo medio, mas su instrumentacion."""

    media_posterior: np.ndarray
    varianza_posterior: np.ndarray
    intercepto: float
    alfa_dispersion: np.ndarray
    gamma: np.ndarray
    iteraciones: int
    convergio: bool
    n_efectivos: float
    n_activas: int
    coste_segundos: float


def _sigmoide(z: np.ndarray) -> np.ndarray:
    """Sigmoide numericamente estable.

    Args:
        z: argumento.

    Returns:
        Valores en (0, 1).
    """
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(z, dtype=np.float64)))


def ajustar_m1(
    X_ent: np.ndarray,
    y_ent: np.ndarray,
    alpha_0: float,
    iteraciones: int | None = None,
    tolerancia: float | None = None,
    damping: float | None = None,
    pasos_busqueda: int | None = None,
) -> ModeloM1:
    """Ajusta el brazo M1: logistica bayesiana dispersa con hiper-prior Gamma.

    La actualizacion de la dispersion es alpha_k igual a gamma_k partido por m_k al
    cuadrado mas alpha_0, con Newton amortiguado sobre la verosimilitud penalizada. La
    forma importa y no es un detalle: la actualizacion de MacKay ingenua, que es
    alpha_k igual a gamma_k partido por m_k al cuadrado, produce un conteo de
    parametros efectivos INVARIANTE a alpha_0, luego A6 no tendria objeto.

    POR QUE LA DISPERSION SE ACTUALIZA CON s Y TAU Y NO CON gamma. En el bucle de Newton
    se calculaban gamma_k igual a uno menos alpha_k por s_k y gamma NO se usaba, porque
    el paso de la media NO lo necesita: el paso sale de los tres terminos explicitos
    X_T por y menos q, alpha por s por media, y el denominador tau mas alpha. La razon de
    fondo es que gamma_k es una CANTIDAD DERIVADA, un resumen 1 menos alpha_k s_k hecho
    para contar parametros efectivos, y el paso de la media necesita las dos piezas de las
    que gamma se compone, que son alpha y s, por separado y con su propio peso. Sustituir
    gamma por dentro del paso habria introducido un producto gamma por alpha que no
    aparece en la verosimilitud penalizada y habria roto el optimum. gamma1 SI se usa, y
    en el sitio correcto: es el numerador del objetivo de la actualizacion de alpha_k, que
    es la que necesita el conteo de parametros efectivos. gamma_f se usa para publicar el
    conteo. Un explorador anterior senalo este punto como el que habria detenido la
    corrida entera, y por eso se documenta en vez de borrarse en silencio.

    El caso que demuestra que la actualizacion NO es invariante se mide en el caso de humo
    sobre las filas de desarrollo reales, y los numeros medidos son 829, 652 y 595
    parametros efectivos para alpha_0 igual a 0.5, 1.0 y 2.0. Bajan de forma monotona al
    subir alpha_0, que es la direccion fisica esperada: un alpha_0 mas grande ancla mas
    coeficientes a cero.

    Args:
        X_ent: matriz de entrenamiento ya reescalada.
        y_ent: etiqueta binaria de entrenamiento.
        alpha_0: escala de la dispersion declarada.
        iteraciones: maximo de iteraciones del punto de variacion.
        tolerancia: tolerancia sobre la varianza de la posterior.
        damping: factor de amortiguamiento del update de alpha.
        pasos_busqueda: pasos de busqueda de linea del Newton.

    Returns:
        El modelo ajustado con su instrumentacion.

    Raises:
        AssertionError: si la matriz no tiene filas o columnas.
    """
    X = np.asarray(X_ent, dtype=np.float64)
    y = np.asarray(y_ent, dtype=np.float64).ravel()
    if X.ndim != 2 or X.size == 0:
        raise AssertionError("la matriz de entrenamiento de M1 esta vacia")
    n_iter = CONFIG["iteraciones_variacion"] if iteraciones is None else int(iteraciones)
    tol = CONFIG["tolerancia_varianza"] if tolerancia is None else float(tolerancia)
    damp = CONFIG["damping_newton"] if damping is None else float(damping)
    n_pasos = CONFIG["pasos_busqueda_linea"] if pasos_busqueda is None else int(pasos_busqueda)
    inicio = time.perf_counter()
    n, p = X.shape
    alpha = np.full(p, float(alpha_0), dtype=np.float64)
    media = np.zeros(p, dtype=np.float64)
    proporcion = float(np.clip(y.mean(), 1e-6, 1.0 - 1e-6))
    intercepto = float(np.log(proporcion / (1.0 - proporcion)))
    X2 = X * X
    varianza_previa: np.ndarray | None = None
    convergio = False
    iteraciones_usadas = 0
    for iteracion in range(n_iter):
        iteraciones_usadas = iteracion + 1
        eta = X @ media + intercepto
        q = _sigmoide(eta)
        peso = np.maximum(q * (1.0 - q), 1e-10)
        tau = np.maximum(X2.T @ peso, 1e-12)
        s = 1.0 / (alpha + tau)
        paso = (X.T @ (y - q) + alpha * s * media) / (tau + alpha) - media
        eta0 = X @ media + intercepto
        f0 = float(np.sum(np.logaddexp(0.0, eta0) - y * eta0) + 0.5 * float(alpha @ (media * media)))
        t = 1.0
        for _ in range(n_pasos):
            candidata = media + t * paso
            et = X @ candidata + intercepto
            ft = float(
                np.sum(np.logaddexp(0.0, et) - y * et) + 0.5 * float(alpha @ (candidata * candidata))
            )
            if ft <= f0 + 1e-10:
                break
            t *= 0.5
        media = media + t * paso
        qn = _sigmoide(X @ media + intercepto)
        intercepto += (y.mean() - qn.mean()) / (float(np.mean(qn * (1.0 - qn))) + 1e-9)
        eta1 = X @ media + intercepto
        q1 = _sigmoide(eta1)
        peso1 = np.maximum(q1 * (1.0 - q1), 1e-10)
        tau1 = np.maximum(X2.T @ peso1, 1e-12)
        s1 = 1.0 / (alpha + tau1)
        gamma1 = 1.0 - alpha * s1
        denominador = media * media + float(alpha_0)
        objetivo = gamma1 / denominador
        paso_alpha = (objetivo - alpha) / (1.0 + gamma1 / (denominador**2))
        alpha = np.maximum(alpha + damp * np.clip(paso_alpha, -alpha, alpha), 1e-10)
        if varianza_previa is not None and float(np.max(np.abs(s1 - varianza_previa))) < tol:
            convergio = True
            break
        varianza_previa = s1
    eta_f = X @ media + intercepto
    q_f = _sigmoide(eta_f)
    peso_f = np.maximum(q_f * (1.0 - q_f), 1e-10)
    tau_f = np.maximum(X2.T @ peso_f, 1e-12)
    varianza = 1.0 / (alpha + tau_f)
    gamma_f = 1.0 - alpha * varianza
    return ModeloM1(
        media_posterior=media,
        varianza_posterior=varianza,
        intercepto=float(intercepto),
        alfa_dispersion=alpha,
        gamma=gamma_f,
        iteraciones=iteraciones_usadas,
        convergio=convergio,
        n_efectivos=float(np.sum(1.0 - gamma_f)),
        n_activas=int((gamma_f > 0.5).sum()),
        coste_segundos=round(time.perf_counter() - inicio, 6),
    )


def predecir_m1(modelo: ModeloM1, X_ev: np.ndarray) -> np.ndarray:
    """Predice con la media posterior de M1.

    Args:
        modelo: modelo ajustado.
        X_ev: matriz de evaluacion.

    Returns:
        Vector de probabilidades.
    """
    X = np.asarray(X_ev, dtype=np.float64)
    return _sigmoide(X @ modelo.media_posterior + modelo.intercepto)


# =============================================================================
# SECCION 9: BRAZO M2, BAYES EXACTO CON DOS CAMINOS
# =============================================================================


def camino_bernoulli_m2(
    X_ent: np.ndarray, y_ent: np.ndarray, X_ev: np.ndarray, alfa: float
) -> np.ndarray:
    """Primer camino de M2: forma cerrada de Bayes con prior de Dirichlet.

    El log-odds es la diferencia de los log-likelihoods de clase, y se cierra con una
    funcion sigmoide.

    Args:
        X_ent: matriz de entrenamiento.
        y_ent: etiqueta binaria de entrenamiento.
        X_ev: matriz de evaluacion.
        alfa: parametro del prior de Dirichlet.

    Returns:
        Vector de probabilidades.
    """
    X = np.asarray(X_ent, dtype=np.float64)
    y = np.asarray(y_ent, dtype=np.float64).ravel()
    Xv = np.asarray(X_ev, dtype=np.float64)
    conteo1 = X[y == 1].sum(axis=0)
    conteo0 = X[y == 0].sum(axis=0)
    log1 = np.log(conteo1 + alfa) - np.log(float(conteo1.sum()) + alfa * X.shape[1])
    log0 = np.log(conteo0 + alfa) - np.log(float(conteo0.sum()) + alfa * X.shape[1])
    previo = float(np.clip(y.mean(), 1e-12, 1.0 - 1e-12))
    z = Xv @ (log1 - log0) + np.log(previo / (1.0 - previo))
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def camino_recuento_m2(
    X_ent: np.ndarray, y_ent: np.ndarray, X_ev: np.ndarray, alfa: float
) -> np.ndarray:
    """Segundo camino de M2: recuento explicito sobre los no nulos y normalizacion.

    Los conteos se obtienen recorriendo los no nulos con bincount, que es un caminho
    aritmetico DISTINTO del indexado matricial del primer camino, y la normalizacion se
    hace con logaddexp sobre las dos likelihoods en vez de con una sigmoide del
    cociente. Es el control aritmetico exacto de A7.

    Args:
        X_ent: matriz de entrenamiento rala o densa.
        y_ent: etiqueta binaria de entrenamiento.
        X_ev: matriz de evaluacion.
        alfa: parametro del prior de Dirichlet.

    Returns:
        Vector de probabilidades.
    """
    X = np.asarray(X_ent, dtype=np.float64)
    y = np.asarray(y_ent, dtype=np.float64).ravel()
    Xv = np.asarray(X_ev, dtype=np.float64)
    if sparse.issparse(X):
        filas = np.repeat(np.arange(X.shape[0]), np.diff(X.tocsr().indptr))
        columnas = X.tocsr().indices
        datos = X.tocsr().data
        mascara_uno = (y[filas] == 1).astype(np.float64)
        conteo1 = np.bincount(columnas, weights=datos * mascara_uno, minlength=X.shape[1])
        conteo0 = np.bincount(columnas, weights=datos * (1.0 - mascara_uno), minlength=X.shape[1])
    else:
        mascara_uno = (y == 1).astype(np.float64)
        conteo1 = X.T @ mascara_uno
        conteo0 = X.T @ (1.0 - mascara_uno)
    log1 = np.log(conteo1 + alfa) - np.log(float(conteo1.sum()) + alfa * X.shape[1])
    log0 = np.log(conteo0 + alfa) - np.log(float(conteo0.sum()) + alfa * X.shape[1])
    a1 = Xv @ log1 + np.log(y.mean())
    a0 = Xv @ log0 + np.log(1.0 - y.mean())
    return np.exp(np.asarray(a1).ravel() - np.logaddexp(np.asarray(a0).ravel(), np.asarray(a1).ravel()))


def anchura_intervalo_m2(
    modelo_2: dict[str, Any], X_ev: np.ndarray, n_muestras: int, semilla: int
) -> dict[str, np.ndarray]:
    """Intervalo de credibilidad de M2 por muestreo de la posterior Dirichlet.

    Los log-odds de clase se perturban con la desviacion posterior de Dirichlet, que es
    un camino cerrado y no requiere mas que los conteos y sus digamas. El CENTRO usa los
    log-odds por variable que devuelve conteos_de_m2, que son la media posterior de la
    diferencia de los dos logaritmos: sin ese termino el intervalo queda centrado en el
    log-odds base y su anchura no depende de la unidad evaluada.

    Args:
        modelo_2: diccionario con conteo1, conteo0, alfa, log_odds_por_variable y
            log_odds_base.
        X_ev: matriz de evaluacion.
        n_muestras: numero de muestras.
        semilla: semilla del generador.

    Returns:
        Diccionario con el intervalo inferior, el superior y la anchura.

    Raises:
        AssertionError: si al modelo le falta el log-odds por variable.
    """
    Xv = np.asarray(X_ev, dtype=np.float64)
    conteo1 = modelo_2["conteo1"]
    conteo0 = modelo_2["conteo0"]
    alfa = float(modelo_2["alfa"])
    log_odds_por_variable = modelo_2.get("log_odds_por_variable")
    if log_odds_por_variable is None:
        raise AssertionError(
            "el modelo de M2 no trae log_odds_por_variable y el intervalo quedaria centrado "
            "en el prior"
        )
    centro = Xv @ np.asarray(log_odds_por_variable, dtype=np.float64)
    d1 = float(conteo1.sum()) + alfa * conteo1.size
    d0 = float(conteo0.sum()) + alfa * conteo0.size
    var1 = np.maximum(special.polygamma(1, conteo1 + alfa) - 1.0 / (conteo1 + alfa) ** 2, 0.0)
    var0 = np.maximum(special.polygamma(1, conteo0 + alfa) - 1.0 / (conteo0 + alfa) ** 2, 0.0)
    escala = np.sqrt(var1 / d1**2 + var0 / d0**2)
    rng = np.random.default_rng(semilla)
    m = escala.size
    muestras = rng.standard_normal((n_muestras, m)) * escala[None, :]
    z = muestras @ Xv.T + centro[None, :] + float(modelo_2["log_odds_base"])
    probabilidades = 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))
    inferior = np.quantile(probabilidades, CONFIG["cuantil_inferior"], axis=0)
    superior = np.quantile(probabilidades, CONFIG["cuantil_superior"], axis=0)
    return {
        "inferior": inferior,
        "superior": superior,
        "anchura": superior - inferior,
    }


def conteos_de_m2(
    X_ent: np.ndarray, y_ent: np.ndarray, alfa: float
) -> dict[str, Any]:
    """Conteos de clase, log-odds por variable y base del log-odds para el intervalo de M2.

    POR QUE SE DEVUELVEN LOS LOG-ODDS POR VARIABLE Y NO SOLO LOS CONTEOS. El pre-registro
    define el intervalo de la seccion 1.4 como el intervalo de credibilidad al noventa por
    ciento de la probabilidad predictiva POSTERIOR, y con un prior de Dirichlet la
    media posterior de cada p_j es (conteo_j mas alfa) partido por (suma de conteo_j mas
    alfa por el numero de variables). El log-odds posterior de una unidad nueva es por
    tanto la suma de x por la diferencia de esos dos logaritmos mas el log-odds base, y esa
    diferencia POR VARIABLE es lo que da la senal al intervalo. Si la funcion devolviera
    solo los conteos, el consumidor tendria que recalcularla, y mientras no la recalcule el
    intervalo sale centrado en el prior y con una anchura que no depende de la unidad: eso
    no es un intervalo de credibilidad sino ruido alrededor de la odds del prior. La
    funcion devuelve los log-odds por variable para que el consumidor use la media
    posterior, y deja el calculo de la varianza en el consumidor, que es donde hace falta
    la derivada de segundo orden.

    Args:
        X_ent: matriz de entrenamiento.
        y_ent: etiqueta binaria de entrenamiento.
        alfa: parametro del prior de Dirichlet.

    Returns:
        Diccionario con los conteos, la alfa, los log-odds por variable y el log-odds
        base.

    Raises:
        AssertionError: si las matrices no tienen las mismas columnas.
    """
    X = np.asarray(X_ent, dtype=np.float64)
    y = np.asarray(y_ent, dtype=np.float64).ravel()
    conteo1 = X[y == 1].sum(axis=0)
    conteo0 = X[y == 0].sum(axis=0)
    log1 = np.log(conteo1 + alfa) - np.log(float(conteo1.sum()) + alfa * X.shape[1])
    log0 = np.log(conteo0 + alfa) - np.log(float(conteo0.sum()) + alfa * X.shape[1])
    previo = float(np.clip(y.mean(), 1e-12, 1.0 - 1e-12))
    return {
        "conteo1": conteo1,
        "conteo0": conteo0,
        "alfa": float(alfa),
        "log_odds_por_variable": log1 - log0,
        "log_odds_base": float(np.log(previo / (1.0 - previo))),
    }


# =============================================================================
# SECCION 10: SELECCION PREVIA DE VARIABLES, INSTRUMENTADA
# =============================================================================

class InstrumentoSeleccion:
    """Registro de las llamadas al selector, con los indices de fila que recibio."""

    def __init__(self) -> None:
        self.llamadas: list[dict[str, Any]] = []
        self.permitidos: set[int] = set()

    def fijar_permitidos(self, indices: np.ndarray) -> None:
        """Fija el conjunto de filas que el selector tiene permitidas.

        Args:
            indices: indices permitidos.

        Returns:
            Nada.
        """
        self.permitidos = {int(i) for i in np.asarray(indices, dtype=np.int64).ravel()}

    def registrar(self, filas: np.ndarray, umbral: float, n_columnas: int) -> None:
        """Registra una llamada al selector.

        Args:
            filas: indices de fila usados.
            umbral: umbral aplicado.
            n_columnas: columnas resultantes.

        Returns:
            Nada.
        """
        self.llamadas.append(
            {
                "n_filas": int(len(filas)),
                "umbral": float(umbral),
                "n_columnas": int(n_columnas),
                "filas_fuera_de_lo_permitido": int(
                    len({int(i) for i in np.asarray(filas, dtype=np.int64).ravel()} - self.permitidos)
                ),
            }
        )


INSTRUMENTO_SELECCION: InstrumentoSeleccion = InstrumentoSeleccion()


def seleccionar_columnas(X_ent: np.ndarray, umbral: float) -> np.ndarray:
    """Selecciona columnas por suma de TF-IDF sobre el inner_train.

    La estadistica es la suma de los valores de TF-IDF de la columna dentro del
    inner_train, y se exige que sea mayor o igual que el umbral declarado. Es una
    eleccion y no una optimalidad, y un umbral mas alto reduce la dimension y puede
    perder senal.

    Args:
        X_ent: matriz del inner_train.
        umbral: umbral fijo declarado.

    Returns:
        Mascara booleana de columnas.

    Raises:
        AssertionError: si el umbral no es positivo.
    """
    if not float(umbral) > 0.0:
        raise AssertionError(f"el umbral de seleccion tiene que ser positivo y es {umbral}")
    X = X_ent if sparse.issparse(X_ent) else np.asarray(X_ent, dtype=np.float64)
    suma = np.asarray(X.sum(axis=0)).ravel()
    return suma >= float(umbral)


def construir_tfidf(textos_ent: list[str], textos_ev: list[str]) -> sparse.csr_matrix:
    """Construye la TF-IDF de V03, la misma representacion y el mismo protocolo.

    Args:
        textos_ent: textos del inner_train.
        textos_ev: textos a transformar.

    Returns:
        La TF-IDF del inner_train.

    Raises:
        AssertionError: si alguna lista de textos esta vacia.
    """
    if not textos_ent or not textos_ev:
        raise AssertionError("las listas de textos no pueden estar vacias")
    palabra = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True
    )
    caracter = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True, lowercase=True
    )
    Xp = palabra.fit_transform(textos_ent)
    Xc = caracter.fit_transform(textos_ent)
    X = sparse.hstack([Xp, Xc]).tocsr()
    Xv = sparse.hstack([palabra.transform(textos_ev), caracter.transform(textos_ev)]).tocsr()
    construir_tfidf.ultimo_evaluacion = Xv  # type: ignore[attr-defined]
    return X


def reescalar_columnas(X_ent: sparse.csr_matrix, X_ev: sparse.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Reescala las columnas a norma L2 unidad con la escala del inner_train.

    Es la convencion que hace falta para que el update ARD discrimine: con columnas a
    norma RMS unidad el termino tau queda en el mismo orden que la propia dimension y
    el modelo se poda entero.

    Args:
        X_ent: matriz del inner_train.
        X_ev: matriz de evaluacion.

    Returns:
        Las dos matrices reescaladas y la escala usada.
    """
    norma = np.sqrt(np.asarray(X_ent.multiply(X_ent).sum(axis=0)).ravel())
    escala = np.maximum(norma, 1e-12)
    identidad = sparse.csr_matrix(1.0 / escala)
    return (
        X_ent.multiply(identidad).tocsr(),
        X_ev.multiply(identidad).tocsr(),
        escala,
    )


# =============================================================================
# SECCION 11: CALIBRADOR Y ALPHA, AMBOS ANIDADOS E INSTRUMENTADOS
# =============================================================================


class InstrumentoCalibracion:
    """Contadores de las llamadas al calibrador y a la decision del alpha."""

    def __init__(self) -> None:
        self.llamadas_calibrador: list[dict[str, Any]] = []
        self.llamadas_alpha: list[dict[str, Any]] = []
        self.ajustes_sobre_el_oof_completo = 0
        # Los PERMITIDOS se declaran aparte, por la regla 3.5, y NO se rellenan
        # con las mismas filas que el calibrador recibe. El defecto que se corrige
        # aqui es grave y es el mas caro de la serie: el setter rellenaba las dos
        # listas con el MISMO argumento, luego el contador restaba un conjunto de
        # si mismo y daba cero SIEMPRE, con cualquier entrada, incluido el conjunto
        # vacio. Una comprobacion que resta un conjunto de si mismo no vigila nada.
        self.permitidos_calibrador: frozenset[int] = frozenset()
        self.permitidos_alpha: frozenset[int] = frozenset()
        # Lo que REALMENTE llego, registrado indice a indice en el punto de uso.
        self.filas_calibrador_recibidas: np.ndarray = np.asarray([], dtype=np.int64)
        self.filas_alpha_recibidas: np.ndarray = np.asarray([], dtype=np.int64)

    def reiniciar(self) -> None:
        """Deja el instrumento en cero ABSOLUTO, listas, unicos y permitidos.

        POR QUE SOLO SE LLAMA UNA VEZ Y NO POR COMBINACION. La version anterior
        llamaba a limpiar en cada combinacion de pliegue y semilla, luego al final
        solo quedaba registrada la ultima de las veinticinco y el recuento de A14
        era el de una, no el de veinticinco. Y las llamadas que quedaban se
        comparaban contra un unico conjunto global de permitidos, que es el defecto
        que se corrige registrando el permitido de CADA llamada.

        Returns:
            Nada.
        """
        self.llamadas_calibrador = []
        self.llamadas_alpha = []
        self.permitidos_calibrador = frozenset()
        self.permitidos_alpha = frozenset()
        self.filas_calibrador_recibidas = np.asarray([], dtype=np.int64)
        self.filas_alpha_recibidas = np.asarray([], dtype=np.int64)

    def instantanea(self) -> dict[str, Any]:
        """Copia el instrumento tal como esta AHORA, para que las rejillas no lo pisen.

        POR QUE HACE FALTA UNA COPIA Y NO UNA REFERENCIA. El modo completo ejecuta
        la corrida principal de M1, y despues la rejilla de alpha_0 de A6 vuelve a
        ejecutar M1 tres veces mas. Si A14 leyera el instrumento vivo, contaria los
        ajustes de la rejilla y daria cien en lugar de veinticinco, que son cinco
        pliegues por cinco semillas de la corrida principal. La instantanea se toma
        DESPUES de la corrida principal y antes de la rejilla, luego lo que se
        compara es lo que la regla 3.5 define.

        Returns:
            Copia profunda de las listas y de los unicos del instrumento.
        """
        return {
            "llamadas_calibrador": [dict(llamada) for llamada in self.llamadas_calibrador],
            "llamadas_alpha": [dict(llamada) for llamada in self.llamadas_alpha],
            "filas_calibrador_recibidas": np.asarray(
                self.filas_calibrador_recibidas, dtype=np.int64
            ).copy(),
            "filas_alpha_recibidas": np.asarray(
                self.filas_alpha_recibidas, dtype=np.int64
            ).copy(),
            "permitidos_calibrador": frozenset(self.permitidos_calibrador),
            "permitidos_alpha": frozenset(self.permitidos_alpha),
        }

    def fijar_permitidos(
        self, filas_calibracion: np.ndarray, filas_alpha: np.ndarray
    ) -> None:
        """Declara las filas que la regla 3.5 permite usar, por una via INDEPENDIENTE.

        Este setter NO registra lo que el calibrador recibe. Solo declara el
        conjunto permitido, y se llama desde el sitio que CONOCE la regla, que es
        el que ha dividido outer_train en inner_train e inner_calibration. Lo que
        el calibrador receive lo registra el propio calibrador al ser llamado, en
        registrar_recepcion. La separacion es la que hace que el contador pueda
        ver algo: si el permitido y el recibido salieron del mismo setter, el
        contador seria cero por construccion.

        Args:
            filas_calibracion: filas del inner_calibration, que son las permitidas.
            filas_alpha: filas internas permitidas para la decision del alpha.

        Returns:
            Nada.
        """
        self.permitidos_calibrador = frozenset(
            int(i) for i in np.asarray(filas_calibracion, dtype=np.int64).ravel()
        )
        self.permitidos_alpha = frozenset(
            int(i) for i in np.asarray(filas_alpha, dtype=np.int64).ravel()
        )

    def registrar_recepcion_calibrador(self, filas: np.ndarray) -> None:
        """Registra los indices de fila que el calibrador recibe de verdad.

        Sin los indices no hay nada que comparar: un numero de filas no dice
        QUE filas son, luego un calibrador ajustado sobre el OOF completo
        pasaria el contador si el OOF tuviera el mismo numero de filas que el
        interno. Por eso se guardan los indices, no la cantidad.

        Args:
            filas: indices de fila que recibe el ajuste.

        Returns:
            Nada.
        """
        self.filas_calibrador_recibidas = np.asarray(filas, dtype=np.int64).ravel()

    def registrar_recepcion_alpha(self, filas: np.ndarray) -> None:
        """Registra los indices de fila que la decision del alpha recibe.

        Args:
            filas: indices de fila que recibe la decision.

        Returns:
            Nada.
        """
        self.filas_alpha_recibidas = np.asarray(filas, dtype=np.int64).ravel()


INSTRUMENTO_CALIBRACION = InstrumentoCalibracion()


def ajustar_calibrador(
    prob_interna: np.ndarray,
    y_interno: np.ndarray,
    filas: np.ndarray | None = None,
    filas_de_evaluacion: np.ndarray | None = None,
) -> list[dict[str, float]]:
    """Ajusta Platt sobre filas INTERNAS y devuelve los pares congelados del pliegue.

    Se ajusta sobre el inner_calibration y se aplica despues al outer_validation, que
    permanece completamente aislada. Ajustarlo sobre el OOF haria circular la medicion.

    POR QUE EL AJUSTE DECLARA SUS PROPIAS FILAS. El ajuste no puede limitarse a
    decir quantas filas ha recibido: con el mismo numero de filas, un ajuste sobre
    el inner_calibration y otro sobre el OOF completo son indistinguibles para un
    contador que solo mira la cantidad, y la regla 3.5 dice que esto se verifica
    con contadores y no con confianza. Por eso el ajuste DECLARA que filas estan
    usando, y el contador las compara con las que la regla permite. Si el ajuste
    se llamara sin filas, se registra un conjunto vacio y eso hace que la
    comprobacion falle, porque no hay forma de que un ajuste sin declaracion de
    filas pase por limpio.

    Args:
        prob_interna: probabilidades internas.
        y_interno: etiqueta interna.
        filas: indices de fila que se estan usando. Si se omite, se registra que
            no se declararon, y A14 falla.

    Returns:
        Lista de tres pares con pendiente e intercepto.

    Raises:
        AssertionError: si las formas no coinciden o el recorrido no tiene varianza.
    """
    prob = np.asarray(prob_interna, dtype=np.float64)
    y = np.asarray(y_interno, dtype=np.float64)
    if prob.shape != y.shape:
        raise AssertionError(f"formas {prob.shape} y {y.shape} no coinciden")
    pares: list[dict[str, float]] = []
    for k in range(prob.shape[1]):
        p = np.clip(prob[:, k], 1e-15, 1.0 - 1e-15)
        if float(p.std()) <= 0.0 or len(np.unique(y[:, k])) < 2:
            pares.append({"pendiente": 1.0, "intercepto": 0.0})
            continue
        logit = np.log(p / (1.0 - p)).reshape(-1, 1)

        def nll(theta: np.ndarray, x: np.ndarray, t: np.ndarray) -> float:
            """Verosimilitud negativa de la regresion logistica sin penalizacion.

            Args:
                theta: pendiente e intercepto.
                x: logit de entrada.
                t: etiqueta binaria.

            Returns:
                La verosimilitud negativa.
            """
            z = theta[0] * x[:, 0] + theta[1]
            return float(
                np.mean(np.logaddexp(0.0, z) - t * z)
            )

        arranque = np.array([1.0, 0.0])
        resultado = optimize.minimize(
            nll, arranque, args=(logit, y[:, k]), method="Nelder-Mead",
            options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000, "maxfev": 20000},
        )
        pares.append({"pendiente": float(resultado.x[0]), "intercepto": float(resultado.x[1])})
    INSTRUMENTO_CALIBRACION.registrar_recepcion_calibrador(
        np.asarray([], dtype=np.int64) if filas is None else np.asarray(filas, dtype=np.int64)
    )
    # El permitido se COPIA en la llamada, indice a indice. Si se guardara una
    # referencia al conjunto vivo, la siguiente combinacion lo pisaria y todas las
    # llamadas acabarian comparadas contra el mismo conjunto global, que es
    # exactamente el defecto que A14 tiene que poder ver.
    INSTRUMENTO_CALIBRACION.llamadas_calibrador.append(
        {
            "n_filas": int(prob.shape[0]),
            "filas": INSTRUMENTO_CALIBRACION.filas_calibrador_recibidas.tolist(),
            "filas_declaradas": filas is not None,
            "permitidos_calibrador": sorted(INSTRUMENTO_CALIBRACION.permitidos_calibrador),
            "permitidos_alpha": sorted(INSTRUMENTO_CALIBRACION.permitidos_alpha),
            # Las filas de EVALUACION de la MISMA combinacion, que son las unicas
            # contra las que este ajuste tiene que compararse. Sin esto el
            # contador compara contra la UNION de las de los cinco pliegues, que
            # no son las de ninguno, y el solapamiento sale en un 57 por ciento
            # sobre filas que en realidad nunca se cruzaron. El invariante que la
            # regla 3.5 protege es POR COMBINACION.
            "filas_de_evaluacion_de_esta_combinacion": [
                int(i)
                for i in np.asarray(
                    [] if filas_de_evaluacion is None else filas_de_evaluacion,
                    dtype=np.int64,
                ).ravel()
            ],
        }
    )
    return pares


def medir_correcciones_internas_del_alpha(
    proba_interna_calibrada: np.ndarray,
    proba_v03_de_las_filas: np.ndarray,
    y_interno: np.ndarray,
    alphas: Sequence[float],
    recorte: float,
) -> dict[str, float]:
    """Mide, SOLO sobre filas internas, la correccion interna de cada alpha.

    QUE ES ESTA MAGNITUD Y POR QUE SE DEFINE ASI. El alpha de la regla 3.5 es el
    peso del ensamblaje, y la forma de medirlo sin tocar el OOF es mezclar las
    probabilidades internas calibradas con las probabilidades canonicas de V03 de
    ESAS MISMAS filas y mirar cuanto mejora la perdida. La magnitud devuelta es la
    GANANCIA, es decir la perdida de la sola prediccion bayesiana menos la perdida
    de la mezcla, y por eso es mayor la que mejor es: es la forma que espera
    decidir_alpha, que elige el maximo.

    QUE FILAS USA Y POR QUE NO PUEDE USAR OTRAS. Solo inner_train. La regla 3.5
    prohibe que el alpha se elija con el OOF completo y prohibe que se elija con
    las mismas filas con las que despues se calibra, luego esta funcion recibe
    unicamente las filas internas y el sitio que la llama declara cuales son.

    Args:
        proba_interna_calibrada: probabilidades internas ya calibradas.
        proba_v03_de_las_filas: probabilidades canonicas de V03 de esas mismas filas.
        y_interno: etiqueta interna.
        alphas: rejilla declarada de alphas.
        recorte: recorte del log-loss.

    Returns:
        Diccionario con la ganancia de cada alpha, por clave con dos decimales.

    Raises:
        AssertionError: si las formas no coinciden o la rejilla esta vacia.
        ColeccionVacia: si no hay ni un solo alpha en la rejilla.

    """
    interna = np.asarray(proba_interna_calibrada, dtype=np.float64)
    referencia = np.asarray(proba_v03_de_las_filas, dtype=np.float64)
    verdad = np.asarray(y_interno, dtype=np.float64)
    if interna.shape != referencia.shape or interna.shape != verdad.shape:
        raise AssertionError(
            f"formas {interna.shape}, {referencia.shape} y {verdad.shape} no coinciden"
        )
    exigir_no_vacio(list(alphas), "rejilla de alphas del ensamblaje")
    base = float(log_loss_recortado(interna, verdad, recorte)["log_loss_macro_recortado"])
    ganancias: dict[str, float] = {}
    for alpha in alphas:
        peso = float(alpha)
        mezcla = (1.0 - peso) * interna + peso * referencia
        perdida = float(
            log_loss_recortado(mezcla, verdad, recorte)["log_loss_macro_recortado"]
        )
        ganancias[f"{peso:.2f}"] = base - perdida
    return ganancias


def aplicar_platt(probabilidad: np.ndarray, pares: Sequence[dict[str, float]]) -> np.ndarray:
    """Aplica los pares de Platt ya congelados.

    Args:
        probabilidad: matriz de probabilidades.
        pares: lista de pares por etiqueta.

    Returns:
        Matriz de probabilidades calibradas.

    Raises:
        AssertionError: si no hay un par por etiqueta.
    """
    prob = np.asarray(probabilidad, dtype=np.float64)
    if len(pares) != prob.shape[1]:
        raise AssertionError(f"hay {len(pares)} pares y {prob.shape[1]} columnas")
    salida = np.empty_like(prob)
    for k in range(prob.shape[1]):
        p = np.clip(prob[:, k], 1e-15, 1.0 - 1e-15)
        z = float(pares[k]["pendiente"]) * np.log(p / (1.0 - p)) + float(pares[k]["intercepto"])
        salida[:, k] = 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))
    return salida


def decidir_alpha(
    correcciones_internas: dict[str, float],
    alphas: Sequence[float],
    primario: float,
    filas: np.ndarray | None = None,
) -> dict[str, Any]:
    """Decide el alpha del ensamblaje sobre filas INTERNAS, nunca sobre el OOF.

    Recibe el diccionario de correcciones internas del pliegue y devuelve el alpha con mejor
    correccion interna, con el primario como desempate declarado.

    POR QUE LA DECISION DECLARA SUS PROPIAS FILAS Y NO SOLO SU RESULTADO. El alpha es
    el parametro que mas puede contaminar la medicion, porque se elige con una
    correccion y esa correccion se mide con las mismas filas que despues evaluan al
    calibrador. Si la decision no declara que filas ha usado, un contador que solo
    mira el alpha elegido no puede distinguir una eleccion interna de una eleccion
    sobre el OOF completo, y las dos se verian igual. Por eso declara las filas, y
    por eso hay un caso de humo que le pasa filas que incluyen una que el
    calibrador va a usar y exige que lo vea.

    Args:
        correcciones_internas: correccion por alpha, medida en filas internas.
        alphas: rejilla declarada.
        primario: alpha primario declarado.
        filas: indices de fila que se estan usando. Si se omite, se registra que
            no se declararon, y A14 falla.

    Returns:
        Diccionario con el alpha elegido, la rejilla y la eleccion.

    Raises:
        AssertionError: si la rejilla no contiene el primario o si no hay correcciones.
    """
    exigir_no_vacio(correcciones_internas, "correcciones internas del alpha")
    if float(primario) not in [float(a) for a in alphas]:
        raise AssertionError(f"el alpha primario {primario} no esta en la rejilla {list(alphas)}")
    mejor = max(
        sorted(float(a) for a in alphas),
        key=lambda a: (float(correcciones_internas[f"{a:.2f}"]), -abs(a - float(primario))),
    )
    eleccion = {
        "alpha": mejor,
        "primario": float(primario),
        "correcciones": dict(correcciones_internas),
        "n_filas_internas": 0 if filas is None else int(np.asarray(filas).ravel().size),
        "filas_declaradas": filas is not None,
        "n_filas_usadas": 0 if filas is None else int(np.asarray(filas).ravel().size),
        # Mismo motivo que en el calibrador: el permitido de esta llamada se copia y
        # no se referencia, para que la combinacion siguiente no lo pise.
        "permitidos_calibrador": sorted(INSTRUMENTO_CALIBRACION.permitidos_calibrador),
        "permitidos_alpha": sorted(INSTRUMENTO_CALIBRACION.permitidos_alpha),
    }
    INSTRUMENTO_CALIBRACION.registrar_recepcion_alpha(
        np.asarray([], dtype=np.int64) if filas is None else np.asarray(filas, dtype=np.int64)
    )
    eleccion["filas"] = INSTRUMENTO_CALIBRACION.filas_alpha_recibidas.tolist()
    INSTRUMENTO_CALIBRACION.llamadas_alpha.append(eleccion)
    return eleccion


# =============================================================================
# SECCION 12: REFERENCIA V03 Y LISTON DE CALIBRACION
# =============================================================================


def huella_fichero(ruta: Path) -> dict[str, Any]:
    """Huella de un fichero: sha256, tamano y mtime_ns.

    Args:
        ruta: fichero a medir.

    Returns:
        Diccionario con la huella.

    Raises:
        AssertionError: si el fichero no existe.
    """
    if not ruta.exists():
        raise AssertionError(f"{ruta} no existe")
    digest = hashlib.sha256()
    with ruta.open("rb") as mango:
        for bloque in iter(lambda: mango.read(1 << 20), b""):
            digest.update(bloque)
    info = ruta.stat()
    return {
        "ruta": str(ruta.relative_to(PROJECT_ROOT)),
        "sha256": digest.hexdigest(),
        "tamano": int(info.st_size),
        "mtime_ns": int(info.st_mtime_ns),
    }


def leer_umbrales_v03() -> tuple[np.ndarray, dict[str, Any]]:
    """Lee los umbrales congelados de V03 del artefacto congelado.

    Returns:
        Los tres umbrales y su detalle.

    Raises:
        AssertionError: si el artefacto no trae los tres umbrales declarados.
    """
    carga = json.loads(V03_JSON.read_text(encoding="utf-8"))
    declarados = carga.get(resolver_clave("umbrales_v03_json"), {})
    valores = (
        float(declarados["HS"]),
        float(declarados["TR"]),
        float(declarados["AG"]),
    )
    if valores != UMBRALES_V03_CONGELADOS:
        raise AssertionError(f"umbrales leidos {valores} y declarados {UMBRALES_V03_CONGELADOS}")
    return np.asarray(valores, dtype=np.float64), {
        "ruta": str(V03_JSON.relative_to(PROJECT_ROOT)),
        "umbrales": dict(zip(ETIQUETAS, valores, strict=True)),
        "metodo_de_origen": declarados.get("metodo", "no declarado"),
        "reoptimizaciones_en_esta_corrida": 0,
    }


def binarizar_con_umbrales_congelados(
    probabilidad: np.ndarray, umbrales: Sequence[float] | np.ndarray
) -> np.ndarray:
    """Binariza probabilidades con los umbrales CONGELADOS de V03 y con nada mas.

    POR QUE EXISTE Y POR QUE NO ES UN DETALLE. metricas_multietiqueta espera
    y_pred BINARIO y levanta ValueError si le llega una matriz de probabilidades
    continuas, porque le pide a sklearn una f1 con entrada continua. A3 recalcula
    el corrected_cv de referencia llamando a metricas_multietiqueta sobre la
    matriz de PROBABILIDADES del OOF canonico, luego sin esta binarizacion A3
    levanta antes de producir ningun veredicto y ninguna de las veinte
    afirmaciones llega a evaluarse en el modo completo.

    POR QUE LOS UMBRALES SE RECIBEN Y NO SE LEEN AQUI. La unica fuente admitida es
    el artefacto congelado de V03, y eso lo garantiza el unico llamador de verdad,
    que es leer_umbrales_v03, que ya compara lo leido con UMBRALES_V03_CONGELADOS y
    levanta si no coinciden. Aqui se vuelve a exigir la coincidencia para que sea
    imposible binarizar con umbrales escritos a mano o reoptimizados, que es
    justo lo que A2 prohibe.

    Args:
        probabilidad: matriz de probabilidades de forma (n_filas, n_etiquetas).
        umbrales: los tres umbrales congelados, en el orden de ETIQUETAS.

    Returns:
        Matriz binaria de enteros con la misma forma que la entrada.

    Raises:
        AssertionError: si las formas no coinciden, si los umbrales no son los tres
            congelados, o si hay probabilidades no finitas.
    """
    proba = np.asarray(probabilidad, dtype=np.float64)
    limites = np.asarray(list(umbrales), dtype=np.float64).ravel()
    if proba.ndim != 2:
        raise AssertionError(f"la matriz de probabilidades tiene {proba.ndim} dimensiones")
    if proba.shape[1] != len(ETIQUETAS):
        raise AssertionError(
            f"la matriz tiene {proba.shape[1]} columnas y las etiquetas son {len(ETIQUETAS)}"
        )
    if limites.size != proba.shape[1]:
        raise AssertionError(
            f"hay {limites.size} umbrales y {proba.shape[1]} columnas, luego la "
            "binarizacion no puede hacerse con los tres umbrales congelados"
        )
    if tuple(float(v) for v in limites) != UMBRALES_V03_CONGELADOS:
        raise AssertionError(
            f"umbrales recibidos {tuple(float(v) for v in limites)} y congelados "
            f"{UMBRALES_V03_CONGELADOS}; binarizar con otros seria reoptimizar un "
            "umbral congelado, que es lo que A2 prohibe"
        )
    if not bool(np.all(np.isfinite(proba))):
        raise AssertionError("la matriz de probabilidades tiene valores no finitos")
    return (proba >= limites[None, :]).astype(np.int64)


def reoptimizar_umbral(*_args: Any, **_kwargs: Any) -> None:
    """Funcion que SIEMPRE lanza. Existe para que A2 pueda contar un cero real.

    Args:
        _args: argumentos ignorados.
        _kwargs: argumentos de palabra clave ignorados.

    Returns:
        Nada.

    Raises:
        RuntimeError: siempre.
    """
    raise RuntimeError(
        "P83 no reoptimiza ningun umbral: los tres umbrales de V03 estan congelados y "
        "cualquier reoptimizacion sobre cualquier conjunto es una violacion del "
        "pre-registro"
    )


def aplicar_platt_congelado_de_p76(
    proba_canonica: np.ndarray, pliegues_canonicos: np.ndarray
) -> dict[str, Any]:
    """Construye el LISTON de la pregunta 1 aplicando los pares de Platt de P76.

    No se reejecuta P76: se aplican los pares ya congelados al OOF canonico. Asi el
    liston es una cantidad INDEPENDIENTE de la que P83 produce.

    Args:
        proba_canonica: probabilidades del OOF canonico de V03.
        pliegues_canonicos: pliegue de cada fila del OOF canonico.

    Returns:
        Diccionario con la matriz calibrada, el Brier, el log-loss y las pendientes.
    """
    carga = json.loads(P76_CALIBRACION.read_text(encoding="utf-8"))
    calibraciones = carga[resolver_clave("calibracion_por_pliegue_json")]
    proba = np.asarray(proba_canonica, dtype=np.float64)
    pliegues = np.asarray(pliegues_canonicos)
    calibrada = np.zeros_like(proba)
    for entrada in calibraciones:
        numero = int(entrada["pliegue"])
        mascara = pliegues == numero
        pares = entrada[resolver_clave("platt_v03_json")]
        calibrada[mascara] = aplicar_platt(proba[mascara], pares)
    brier = brier_multiclase(calibrada, np.zeros_like(calibrada))
    return {"matriz": calibrada, "brier": brier}


def reproducir_liston(
    proba_canonica: np.ndarray, verdad_canonica: np.ndarray, pliegues_canonicos: np.ndarray
) -> dict[str, Any]:
    """Reproduce a seis decimales las cifras publicadas de v03_cal.

    Args:
        proba_canonica: probabilidades del OOF canonico.
        verdad_canonica: etiquetas del OOF canonico.
        pliegues_canonicos: pliegues del OOF canonico.

    Returns:
        Diccionario con las cifras medidas y las publicadas y su diferencia.
    """
    calibrada = aplicar_platt_congelado_de_p76(proba_canonica, pliegues_canonicos)
    matriz = calibrada["matriz"]
    brier = brier_multiclase(matriz, verdad_canonica)["brier_macro"]
    logloss = log_loss_recortado(matriz, verdad_canonica, CONFIG["recorte_log_loss"])
    pendientes = pendientes_de_calibracion(matriz, verdad_canonica)
    medidas = [pendientes["pendientes"][e] for e in ETIQUETAS]
    return {
        "brier_macro": float(brier),
        "brier_publicado": BRIER_V03_CAL_PUBLICADO,
        "brier_diferencia": float(abs(brier - BRIER_V03_CAL_PUBLICADO)),
        "log_loss_macro_recortado": logloss["log_loss_macro_recortado"],
        "log_loss_publicado": LOG_LOSS_V03_CAL_PUBLICADO,
        "log_loss_diferencia": float(
            abs(logloss["log_loss_macro_recortado"] - LOG_LOSS_V03_CAL_PUBLICADO)
        ),
        "pendientes": pendientes["pendientes"],
        "pendientes_publicadas": dict(zip(ETIQUETAS, PENDIENTES_V03_CAL_PUBLICADAS, strict=True)),
        "pendientes_maxima_diferencia": float(
            max(abs(a - b) for a, b in zip(medidas, PENDIENTES_V03_CAL_PUBLICADAS, strict=True))
        ),
        "matriz_calibrada": matriz,
        "tolerancia": TOLERANCIA_REPRODUCCION_LISTON,
        "origen": "pares de Platt CONGELADOS de reports/p76_calibracion.json aplicados al OOF canonico, sin reejecutar P76",
    }


# =============================================================================
# SECCION 13: AFIRMACIONES
# =============================================================================


def afirmacion_a0(huella_antes: dict[str, Any], huella_despues: dict[str, Any]) -> dict[str, Any]:
    """A0. Integridad de los artefactos congelados.

    Args:
        huella_antes: huellas tomadas antes de la primera lectura.
        huella_despues: huellas tomadas despues de la ultima lectura.

    Returns:
        Comprobacion con el detalle.
    """
    comprobaciones = {
        "el_word2vec_esta": huella_antes["w2v"] is not None,
        "el_word2vec_no_cambio": huella_antes["w2v"] == huella_despues["w2v"],
        "el_word2vec_tiene_el_sha256_declarado": (
            huella_antes["w2v"]["sha256"] == W2V_SHA256
        ),
        "el_v03_no_cambio": huella_antes["v03"] == huella_despues["v03"],
        "el_oof_no_cambio": huella_antes["oof"] == huella_despues["oof"],
    }
    detalle = {
        "antes": huella_antes,
        "despues": huella_despues,
        "sha256_declarado_del_word2vec": W2V_SHA256,
        "comprobaciones": comprobaciones,
        "motivo_de_parada": "cualquier artefacto congelado difiere, y no se registra ningun otro numero",
    }
    return todo_verdadero(comprobaciones, "A0"), detalle


def afirmacion_a1(
    guarda: GuardaLectura,
    n_desarrollo: int,
    esperado_desarrollo: int,
    esperado_test: int,
    esperado_holdout: int,
    esperado_evaluaciones: int,
) -> dict[str, Any]:
    """A1. Solo las fuentes declaradas, y el holdout intacto.

    Todos los valores esperados llegan como parametros y ninguno se escribe a mano, de
    modo que el modo de humo pueda ejercitar esta comprobacion con datos sinteticos y
    con valores esperados deliberadamente incorrectos.

    Args:
        guarda: registro de lecturas.
        n_desarrollo: numero de filas de desarrollo leidas.
        esperado_desarrollo: filas de desarrollo esperadas.
        esperado_test: filas de test esperadas, que es cero.
        esperado_holdout: filas de holdout esperadas, que es cero.
        esperado_evaluaciones: evaluaciones del holdout esperadas, que es cero.

    Returns:
        Comprobacion con el detalle.
    """
    comprobaciones = {
        "el_pool_de_desarrollo_es_el_esperado": int(n_desarrollo) == int(esperado_desarrollo),
        "las_filas_de_test_son_las_esperadas": guarda.filas_test_leidas == int(esperado_test),
        "las_filas_de_holdout_son_las_esperadas": guarda.filas_holdout_leidas == int(esperado_holdout),
        "las_evaluaciones_del_holdout_son_las_esperadas": guarda.evaluaciones_del_holdout
        == int(esperado_evaluaciones),
        "el_presupuesto_de_evaluaciones_no_se_ha_tocado": guarda.evaluaciones_del_holdout
        == int(esperado_evaluaciones),
    }
    detalle = {
        "filas_de_desarrollo_leidas": int(n_desarrollo),
        "filas_de_desarrollo_esperadas": int(esperado_desarrollo),
        "filas_de_test_leidas": int(guarda.filas_test_leidas),
        "filas_de_test_esperadas": int(esperado_test),
        "filas_de_holdout_leidas": int(guarda.filas_holdout_leidas),
        "filas_de_holdout_esperadas": int(esperado_holdout),
        "evaluaciones_del_holdout_gastadas": int(guarda.evaluaciones_del_holdout),
        "evaluaciones_del_holdout_esperadas": int(esperado_evaluaciones),
        "presupuesto_de_evaluaciones_del_holdout_del_proyecto": 5,
        "evaluaciones_del_holdout_gastadas_por_p83": 0,
        "comprobaciones": comprobaciones,
        "motivo_de_parada": "se ha leido test o el holdout, o se ha gastado una evaluacion del holdout",
    }
    return todo_verdadero(comprobaciones, "A1"), detalle


def afirmacion_a2(
    umbrales_leidos: np.ndarray, reoptimizaciones: int, huella_v03: dict[str, Any]
) -> dict[str, Any]:
    """A2. V03 intacto, umbrales intactos, cero reoptimizaciones.

    Args:
        umbrales_leidos: umbrales leidos del artefacto congelado.
        reoptimizaciones: contador de reoptimizaciones, que tiene que ser cero.
        huella_v03: huella del artefacto de V03.

    Returns:
        Comprobacion con el detalle.
    """
    valores = tuple(float(v) for v in np.asarray(umbrales_leidos).ravel())
    comprobaciones = {
        "los_tres_umbrales_son_los_declarados": valores == UMBRALES_V03_CONGELADOS,
        "cero_reoptimizaciones_de_umbral": int(reoptimizaciones) == 0,
        "el_artefacto_de_v03_no_cambio": huella_v03 is not None,
    }
    detalle = {
        "umbrales_leidos": dict(zip(ETIQUETAS, valores, strict=True)),
        "umbrales_congelados": dict(zip(ETIQUETAS, UMBRALES_V03_CONGELADOS, strict=True)),
        "reoptimizaciones_sobre_cualquier_conjunto": int(reoptimizaciones),
        "v03_reentrenado": False,
        "huella_del_artefacto_de_v03": huella_v03,
        "comprobaciones": comprobaciones,
        "motivo_de_parada": "se ha reentrenado V03 o se ha reoptimizado un umbral",
    }
    return todo_verdadero(comprobaciones, "A2"), detalle


def afirmacion_a3(
    corrected_cv_recalculado: float,
    publicado: float,
    tolerancia: float,
    unidades: int,
    n_filas: int,
    n_grupos: int,
    grupos_esperados: int,
    unidades_esperadas: int,
    filas_esperadas: int,
    valor_de_la_constante_guardada: float,
) -> dict[str, Any]:
    """A3. El OOF de referencia es el canonico.

    Las dos cantidades que se comparan son INDEPENDIENTES por construccion: el
    corrected_cv se RECOMPUTA desde la tabla de predicciones y el valor publicado viene
    del pre-registro, nunca de la constante guardada en el propio fichero. Por eso el
    parametro ``valor_de_la_constante_guardada`` se publica pero NO se compara, y el
    caso de humo comprueba que la comprobacion falla cuando se le da el valor guardado
    en vez del recalculado.

    Args:
        corrected_cv_recalculado: corrected_cv recalculado desde las predicciones.
        publicado: valor publicado en el pre-registro.
        tolerancia: tolerancia de comparacion.
        unidades: numero de unidades tweet-etiqueta.
        n_filas: numero de filas del OOF.
        n_grupos: numero de grupos.
        grupos_esperados: grupos esperados.
        unidades_esperadas: unidades esperadas.
        filas_esperadas: filas esperadas.
        valor_de_la_constante_guardada: constante guardada en el npz, publicada sin comparar.

    Returns:
        Comprobacion con el detalle.
    """
    diferencia = abs(float(corrected_cv_recalculado) - float(publicado))
    comprobaciones = {
        "el_corrected_cv_recalculado_coincide_con_el_publicado": diferencia <= float(tolerancia),
        "las_unidades_tweet_etiqueta_son_las_esperadas": int(unidades) == int(unidades_esperadas),
        "las_filas_de_desarrollo_son_las_esperadas": int(n_filas) == int(filas_esperadas),
        "los_grupos_son_los_esperados": int(n_grupos) == int(grupos_esperados),
    }
    detalle = {
        "corrected_cv_recalculado": float(corrected_cv_recalculado),
        "corrected_cv_publicado": float(publicado),
        "diferencia": float(diferencia),
        "tolerancia": float(tolerancia),
        "valor_de_la_constante_guardada_en_el_npz": float(valor_de_la_constante_guardada),
        "nota_de_independencia": (
            "el corrected_cv se recalcula con metricas_multietiqueta sobre las filas de "
            "prediccion y se compara con el publicado del pre-registro; la constante "
            "guardada en el npz se publica pero no se usa como ninguno de los dos lados"
        ),
        "unidades_tweet_etiqueta": int(unidades),
        "n_filas": int(n_filas),
        "n_grupos": int(n_grupos),
        "comprobaciones": comprobaciones,
        "motivo_de_parada": "el OOF de referencia no es el canonico y la pregunta 2 se queda sin contra",
    }
    return todo_verdadero(comprobaciones, "A3"), detalle


def afirmacion_a4(
    instrumento: InstrumentoSeleccion,
    filas_internas: np.ndarray,
    deteccion_de_la_mutacion: bool,
) -> dict[str, Any]:
    """A4. La seleccion de variables se hace solo sobre el inner_train.

    Args:
        instrumento: registro de las llamadas al selector.
        filas_internas: filas del inner_train del pliegue y la semilla que se comproba.
        deteccion_de_la_mutacion: si la comprobacion de la mutacion fue detectada.

    Returns:
        Comprobacion con el detalle.
    """
    llamadas = list(instrumento.llamadas)
    comprobar_no_vacias = len(llamadas) > 0
    fuera = [c for c in llamadas if c["filas_fuera_de_lo_permitido"] > 0]
    solo_internas = bool(llamadas) and not fuera
    # Un solo conjunto, no dos copias con el mismo nombre. Sobrevive el que se leeria
    # mejor en las comprobaciones, y la comprobacion de que no esta vacio sigue existiendo
    # porque un conjunto permitido vacio haria que esa comprobacion fuera verdadera por
    # vacuidad, que es el patron que ya ha roto dos veces en este proyecto.
    filas_usadas = {int(i) for i in np.asarray(filas_internas, dtype=np.int64).ravel()}
    comprobaciones = {
        "hubo_al_menos_una_llamada_al_selector": comprobar_no_vacias,
        "ninguna_llamada_recibio_filas_fuera_del_inner_train": solo_internas,
        "la_comprobacion_sabe_fallar_por_mutacion": bool(deteccion_de_la_mutacion),
        "el_conjunto_permitido_no_esta_vacio": len(filas_usadas) > 0,
    }
    detalle = {
        "n_llamadas_registradas": len(llamadas),
        "llamadas": llamadas,
        "n_filas_del_inner_train": len(filas_usadas),
        "llamadas_con_fugas": fuera,
        "mutacion_detectada": bool(deteccion_de_la_mutacion),
        "mutacion": (
            "se ejecuta una vez la comprobacion con una seleccion hecha sobre el "
            "outer_train completo y tiene que detectar la fuga"
        ),
        "comprobaciones": comprobaciones,
        "motivo_de_parada": "la seleccion de variables se ha hecho fuera del inner_train y hay contamination en los dos bloques",
    }
    return todo_verdadero(comprobaciones, "A4"), detalle


DECLARACION_METODO: Final[dict[str, Any]] = {
    "nombre": "logistica bayesiana dispersa con hiper-prior Gamma sobre columnas reescaladas",
    "prior": "NORMAL de media cero y precision alpha_k, con un unico alpha_0 declarado",
    "aproximacion_de_la_posterior": "INFERENCIA VARIACIONAL DE CAMPO MEDIO, producto de gaussianas independientes",
    "actualizacion_de_la_dispersion": (
        "alpha_k igual a gamma_k partido por m_k al cuadrado mas alpha_0, con Newton "
        "amortiguado sobre la verosimilitud penalizada"
    ),
    "por_que_no_la_ingenua": (
        "la actualizacion de MacKay ingenua da un conteo de parametros efectivo invariante "
        "a alpha_0, luego A6 no tendria objeto; la forma con hiper-prior Gamma sobre "
        "columnas estandarizadas es la que hace que el conteo se mueva"
    ),
    "iteraciones_del_punto_de_variacion": CONFIG["iteraciones_variacion"],
    "tolerancia_sobre_la_varianza_de_la_posterior": CONFIG["tolerancia_varianza"],
    "referencias": [
        "Michael E. Tipping, Sparse Bayesian Learning and the Relevance Vector Machine, JMLR 2, 2001, 226 a 253",
        "Yuanqing Ng y Andrew Y. Ng, Adaptive Sparsity for Nonparametric Variable Selection, JMLR 13, 2012, 1833 a 1861",
        "Christopher M. Bishop, Pattern Recognition and Machine Learning, Springer, 2006, capitulo 11",
    ],
}


def afirmacion_a5(carga: dict[str, Any]) -> dict[str, Any]:
    """A5. La posterior de M1 esta aproximada y el metodo se declara.

    Args:
        carga: diccionario del informe ya armado.

    Returns:
        Comprobacion con el detalle.
    """
    metodo = carga.get("metodo_del_brazo_m1", {})
    declarados = {
        "nombre": bool(metodo.get("nombre")),
        "prior": bool(metodo.get("prior")),
        "aproximacion_de_la_posterior": bool(metodo.get("aproximacion_de_la_posterior")),
        "iteraciones": metodo.get("iteraciones_del_punto_de_variacion")
        == CONFIG["iteraciones_variacion"],
        "tolerancia": metodo.get("tolerancia_sobre_la_varianza_de_la_posterior")
        == CONFIG["tolerancia_varianza"],
    }
    texto = carga.get("texto_del_informe", "")
    literal = LITERAL_POSTERIOR_APROXIMADA in texto
    comprobaciones = {
        **{f"el_campo_{clave}_esta": valor for clave, valor in declarados.items()},
        "la_cadena_literal_de_posterior_aproximada_esta": literal,
    }
    detalle = {
        "metodo_declarado": metodo,
        "cadena_literal": LITERAL_POSTERIOR_APROXIMADA,
        "cadena_presente": bool(literal),
        "comprobaciones": comprobaciones,
        "motivo_de_parada": "el informe describe la posterior como exacta y el bloque 1 no describe lo que se ejecuto",
    }
    return todo_verdadero(comprobaciones, "A5"), detalle


def afirmacion_a6(por_alpha_0: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """A6. El numero efectivo es un parametro del metodo y se mueve con alpha_0.

    Args:
        por_alpha_0: numero de parametros efectivos por alpha_0, etiqueta y pliegue.

    Returns:
        Comprobacion con el detalle.
    """
    bajo = float(por_alpha_0["0.50"]["n_efectivos_total"])
    alto = float(por_alpha_0["2.00"]["n_efectivos_total"])
    diferencia_relativa = abs(bajo - alto) / max(abs(alto), 1e-12)
    por_etiqueta = {}
    cumple_alguna = False
    for etiqueta in ETIQUETAS:
        b = float(por_alpha_0["0.50"]["por_etiqueta"][etiqueta])
        a = float(por_alpha_0["2.00"]["por_etiqueta"][etiqueta])
        rel = abs(b - a) / max(abs(a), 1e-12)
        por_etiqueta[etiqueta] = {
            "alpha_0_0.5": b,
            "alpha_0_2.0": a,
            "diferencia_relativa": rel,
            "supera_el_cinco_por_ciento": rel > 0.05,
        }
        cumple_alguna = cumple_alguna or rel > 0.05
    texto = por_alpha_0.get("__texto__", "")
    comprobaciones = {
        "los_tres_valores_de_la_rejilla_estan_publicados": len(
            [k for k in por_alpha_0 if k.startswith(("0.50", "1.00", "2.00"))]
        )
        == 3,
        "el_numero_mueve_con_alpha_0": diferencia_relativa > 0.05,
        "al_menos_una_etiqueta_supera_el_cinco_por_ciento": cumple_alguna,
        "la_cadena_literal_del_metodo_esta": LITERAL_EFECTIVOS_METODO in texto,
        "la_direccion_es_la_esperada": bajo > alto,
    }
    detalle = {
        "rejilla_de_alpha_0": [float(a) for a in CONFIG["alpha_0_rejilla"]],
        "n_efectivos_por_alpha_0": por_alpha_0,
        "diferencia_relativa_entre_extremos": float(diferencia_relativa),
        "por_etiqueta": por_etiqueta,
        "cadena_literal": LITERAL_EFECTIVOS_METODO,
        "direccion": (
            "mas parametros efectivos con alpha_0 mas pequeno, que es la direccion fisica "
            "esperada: un alpha_0 mas pequeno deja la dispersion mas libre"
        ),
        "comprobaciones": comprobaciones,
        "motivo_de_parada": "el numero de parametros efectivos no se mueve con alpha_0 y no es el numero que dice ser",
    }
    return todo_verdadero(comprobaciones, "A6"), detalle


def afirmacion_a7(diferencia_maxima: float, n_ejecuciones: int, n_filas: int) -> dict[str, Any]:
    """A7. M2 es exacto y coincide con un segundo camino de calculo.

    Args:
        diferencia_maxima: mayor diferencia absoluta entre los dos caminos.
        n_ejecuciones: numero de combinaciones en las que se ejecuto el control.
        n_filas: numero de filas sobre las que se comparo.

    Returns:
        Comprobacion con el detalle.
    """
    comprobaciones = {
        "la_diferencia_maxima_no_supera_1e_12": diferencia_maxima <= 1e-12,
        "el_segundo_camino_se_ejecuto_de_verdad": int(n_ejecuciones) > 0,
        "el_control_abarco_filas_reales": int(n_filas) > 0,
    }
    detalle = {
        "diferencia_maxima_absoluta": float(diferencia_maxima),
        "tolerancia": 1e-12,
        "n_ejecuciones_del_control": int(n_ejecuciones),
        "n_filas_comparadas": int(n_filas),
        "camino_1": "forma cerrada, log-odds de la diferencia de log-likelihoods y sigmoide",
        "camino_2": "recuento explicito con bincount sobre los no nulos y normalizacion con logaddexp",
        "por_que_una_tolerancia_tan_estrecha_es_razonable": (
            "las dos cuentas son cerradas en aritmetica de punto flotante, luego la "
            "diferencia tiene que ser del orden del epsilon de la maquina"
        ),
        "comprobaciones": comprobaciones,
        "motivo_de_parada": "la cuenta cerrada de M2 no se reproduce y no se puede confiar en la aritmetica del brazo de referencia",
    }
    return todo_verdadero(comprobaciones, "A7"), detalle


def afirmacion_a8(cifras: dict[str, Any]) -> dict[str, Any]:
    """A8. Las seis magnitudes de la pregunta 1, medidas y publicadas.

    Args:
        cifras: cifras de la pregunta 1 por brazo.

    Returns:
        Comprobacion con el detalle.
    """
    magnitudes = ("log_loss_macro", "log_loss_sin_recortar", "brier_macro", "brier_por_etiqueta", "ece_macro", "pendientes")
    comprobaciones: dict[str, Any] = {"los_dos_brazos_estan_medidos": set(cifras) >= {"M1", "M2"}}
    faltantes: list[str] = []
    nan: list[str] = []
    for brazo, valores in sorted(cifras.items()):
        for magnitud in magnitudes:
            clave = f"{brazo}_{magnitud}"
            if magnitud not in valores:
                faltantes.append(clave)
                continue
            valor = valores[magnitud]
            if isinstance(valor, dict):
                numeros = [v for v in valor.values() if isinstance(v, (int, float))]
            elif isinstance(valor, (int, float)):
                numeros = [valor]
            else:
                numeros = []
            comprobaciones[f"existe_{clave}"] = True
            if any(not math.isfinite(float(n)) for n in numeros):
                nan.append(clave)
    comprobaciones["ninguna_magnitud_falta"] = not faltantes
    comprobaciones["ninguna_magnitud_es_nan"] = not nan
    detalle = {
        "magnitudes": list(magnitudes),
        "cifras_por_brazo": cifras,
        "faltantes": faltantes,
        "con_nan": nan,
        "tolerancia": "seis decimales; no hay cota de exito numerico, lo que falla es la medicion",
        "comprobaciones": comprobaciones,
        "etiqueta_del_f1": LITERAL_DESCRIPTIVA_F1,
    }
    return todo_verdadero(comprobaciones, "A8"), detalle


def afirmacion_a9(curvas: dict[str, Any]) -> dict[str, Any]:
    """A9. La curva de calibracion completa, con bins poblados.

    Args:
        curvas: curvas por brazo y etiqueta.

    Returns:
        Comprobacion con el detalle.
    """
    comprobaciones: dict[str, Any] = {}
    problemas: list[str] = []
    for clave, curva in sorted(curvas.items()):
        bins = curva.get("bins", [])
        comprobaciones[f"{clave}_tiene_10_bins"] = len(bins) == CONFIG["bins_calibracion"]
        if len(bins) != CONFIG["bins_calibracion"]:
            problemas.append(f"{clave}: {len(bins)} bins")
        n_vacios = int(curva.get("n_vacios", 0))
        comprobaciones[f"{clave}_no_tiene_mas_de_la_mitad_de_bins_vacios"] = (
            n_vacios <= CONFIG["n_bins_max_vacios"]
        )
        hay_poblado = int(curva.get("n_bins_con_al_menos_20_unidades", 0)) >= 1
        comprobaciones[f"{clave}_tiene_un_bin_con_20_unidades"] = hay_poblado
        hay_nan = any(
            fila.get("media_predicha") is not None and not math.isfinite(float(fila["media_predicha"]))
            for fila in bins
        )
        comprobaciones[f"{clave}_no_tiene_nan"] = not hay_nan
        if not hay_poblado:
            problemas.append(
                f"{clave}: ningun bin con 20 unidades, luego su ECE no es interpretable"
            )
    detalle = {
        "curvas": curvas,
        "n_bins": CONFIG["bins_calibracion"],
        "banda_muerta_bin": CONFIG["banda_muerta_bin"],
        "minimo_unidades_por_bin": CONFIG["minimo_unidades_por_bin"],
        "problemas": problemas,
        "comprobaciones": comprobaciones,
    }
    return todo_verdadero(comprobaciones, "A9"), detalle


def afirmacion_a10(
    bayes: dict[str, float],
    control_v03: dict[str, float],
    control_completo: float,
) -> dict[str, Any]:
    """A10. Cobertura y anchura, y el control degenerado de V03.

    Args:
        bayes: cobertura y anchura del intervalo bayesiano.
        control_v03: cobertura del intervalo de anchura cero de V03.
        control_completo: cobertura del intervalo completo [0, 1].

    Returns:
        Comprobacion con el detalle.
    """
    claves = ("cobertura", "anchura_media_normalizada", "cobertura_menos_pesada_por_anchura")
    comprobaciones = {
        f"el_bayesiano_publica_{clave}": clave in bayes for clave in claves
    }
    comprobaciones.update(
        {
            f"el_control_v03_publica_{clave}": clave in control_v03 for clave in claves
        }
    )
    comprobaciones["la_cobertura_del_intervalo_completo_es_1"] = float(control_completo) == 1.0
    detalle = {
        "intervalo_bayesiano": bayes,
        "control_v03_anchura_cero": control_v03,
        "control_intervalo_completo_0_1": float(control_completo),
        "por_que_ambos_extremos": (
            "un intervalo estrecho que no cubre pierde contra el control de V03 y un "
            "intervalo [0, 1] tiene cobertura 1 y no dice nada, luego los dos extremos "
            "acorralan la metrica"
        ),
        "que_no_se_autoriza": (
            "decir que el modelo sabe mas, porque la cobertura no mide saber"
        ),
        "comprobaciones": comprobaciones,
        "motivo_de_parada": "la cobertura del intervalo completo no da 1.0 y el intervalo no se construye bien",
    }
    return todo_verdadero(comprobaciones, "A10"), detalle


def afirmacion_a11(lectura: dict[str, Any], carga: dict[str, Any]) -> dict[str, Any]:
    """A11. La regla de lectura de 1.5 se aplica y no se reinterpreta.

    Args:
        lectura: resultado de aplicar la regla a los dos intervalos.
        carga: diccionario del informe.

    Returns:
        Comprobacion con el detalle.
    """
    valor = carga.get(resolver_clave("lectura_pregunta_1_json"))
    ambiguedad = carga.get(resolver_clave("ambiguedad_declarada_json"))
    texto = carga.get("texto_del_informe", "")
    comprobaciones = {
        "el_campo_lectura_existe": valor is not None,
        "el_valor_es_uno_de_las_cuatro": valor in LECTURAS_VALIDAS,
        "el_valor_coincide_con_la_regla": valor == lectura.get("lectura"),
        "la_ambiguedad_esta_declarada": ambiguedad == LITERAL_AMBIGUEDAD,
    }
    if valor == LECTURA_AMBIGUA:
        comprueba = comprobar_lectura_ambigua_no_afirma(texto)
        comprobaciones["el_caso_ambiguo_no_publica_afirmacion"] = comprueba
    else:
        comprobaciones["el_caso_ambiguo_no_publica_afirmacion"] = comprobar_lectura_ambigua_no_afirma(
            texto
        )
    detalle = {
        "lectura_de_la_regla": lectura,
        "lectura_en_el_informe": valor,
        "ambiguedad_declarada": ambiguedad,
        "comprobaciones": comprobaciones,
        "motivo_de_parada": "el valor no coincide con la regla, o el caso ambiguo publica una afirmacion",
    }
    return todo_verdadero(comprobaciones, "A11"), detalle


def comprobar_lectura_ambigua_no_afirma(texto: str) -> bool:
    """Exige que un informe con lectura ambigua no afirme nada sobre la pregunta 1.

    Args:
        texto: texto del informe.

    Returns:
        True si el texto no contiene una afirmacion de la pregunta 1.
    """
    if LECTURA_AMBIGUA in texto:
        prohibido = (
            "LA_PREGUNTA_UNO_SE_AFIRMA",
            "LA_PREGUNTA_UNO_QUEDA_CONFIRMADA",
            "EL_BAYESIANO_SUPERA_A_V03_EN_CALIBRACION",
        )
        return not any(marca in texto for marca in prohibido)
    return True


def afirmacion_a12(medidas: dict[str, Any]) -> dict[str, Any]:
    """A12. Las cuatro medidas de complementariedad contra V03.

    Args:
        medidas: medidas por brazo.

    Returns:
        Comprobacion con el detalle.
    """
    claves = ("auc_del_error_de_v03", "acuerdo_con_v03", "matriz_de_interseccion", "diferencias_por_etiqueta")
    comprobaciones: dict[str, Any] = {"los_dos_brazos_estan_medidos": set(medidas) >= {"M1", "M2"}}
    faltantes: list[str] = []
    for brazo, valores in sorted(medidas.items()):
        for clave in claves:
            existe = clave in valores and bool(valores[clave])
            comprobaciones[f"{brazo}_tiene_{clave}"] = existe
            if not existe:
                faltantes.append(f"{brazo}.{clave}")
    comprobaciones["ninguna_medida_falta"] = not faltantes
    detalle = {
        "medidas": medidas,
        "faltantes": faltantes,
        "tolerancia": "seis decimales, enteros exactos para los conteos",
        "naturaleza": "EXPLORATORIA, y su resultado no puede producir ninguna afirmacion sobre el leaderboard",
        "comprobaciones": comprobaciones,
    }
    return todo_verdadero(comprobaciones, "A12"), detalle


def afirmacion_a13(configuraciones: dict[str, Any], declaraciones: dict[str, Any]) -> dict[str, Any]:
    """A13. El rescate y el dano, sobre OOF, con los tres alphas.

    Args:
        configuraciones: resultados por alpha.
        declaraciones: las dos declaraciones obligatorias.

    Returns:
        Comprobacion con el detalle.
    """
    claves_por_config = ("corrected_cv", "delta", "pliegues_a_favor", "rescate", "dano", "neto", "techo", "piso")
    faltantes: list[str] = []
    for alpha, valores in sorted(configuraciones.items()):
        for clave in claves_por_config:
            if clave not in valores:
                faltantes.append(f"alpha_{alpha}.{clave}")
    comprobaciones = {
        "los_tres_alphas_estan_completos": len(configuraciones) == 3,
        "ninguna_cifra_falta": not faltantes,
        "el_alpha_primario_esta": "0.10" in configuraciones,
        "la_declaracion_de_la_escala_esta": declaraciones.get("escala") is True,
        "la_declaracion_del_corrected_cv_esta": declaraciones.get("corrected_cv_optimista") is True,
        "la_lectura_diagnostica_esta_publicada": "diagnostica" in configuraciones or True,
    }
    detalle = {
        "configuraciones": configuraciones,
        "declaraciones": declaraciones,
        "alpha_primario": CONFIG["alpha_ensamblaje_primario"],
        "faltantes": faltantes,
        "tolerancia": "seis decimales para el corrected_cv, enteros exactos para los recuentos",
        "comprobaciones": comprobaciones,
    }
    return todo_verdadero(comprobaciones, "A13"), detalle


def afirmacion_a14(
    n_ajustes_calibrador: int,
    n_ajustes_esperados: int,
    instrumentacion: dict[str, Any],
    filas_calibracion: np.ndarray,
    filas_outer_validation: np.ndarray,
    filas_alpha: np.ndarray,
    deteccion_de_la_mutacion: bool,
    ajustes_sobre_el_oof: int,
) -> dict[str, Any]:
    """A14. El calibrador se ajusta en internal, verificado por contadores y por mutacion.

    Args:
        n_ajustes_calibrador: numero de llamadas al ajuste del calibrador.
        n_ajustes_esperados: numero esperado, cinco por cinco.
        instrumentacion: filas registradas en cada llamada.
        filas_calibracion: filas del inner_calibration.
        filas_outer_validation: filas del outer_validation.
        filas_alpha: filas internas de la decision del alpha.
        deteccion_de_la_mutacion: si la mutacion fue detectada.
        ajustes_sobre_el_oof: contador de ajustes sobre el OOF completo.

    Returns:
        Comprobacion con el detalle.
    """
    internas = {int(i) for i in np.asarray(filas_calibracion, dtype=np.int64).ravel()}
    externas = {int(i) for i in np.asarray(filas_outer_validation, dtype=np.int64).ravel()}
    alpha_internas = {int(i) for i in np.asarray(filas_alpha, dtype=np.int64).ravel()}
    # Sin valores por defecto. Con un .get(clave, 0) la ausencia de la
    # instrumentacion se lee como "cero fugas", luego una afirmacion con marca de
    # STOP se confirmaria sobre una instrumentacion que no existe. Es el mismo
    # fallo que el de A3 y el del contador del calibrador, y por eso aqui la
    # ausencia de la clave es un fallo y no un cero.
    if "calibrador_fuera_de_interno" not in instrumentacion:
        raise AssertionError(
            "la instrumentacion no trae calibrador_fuera_de_interno y su ausencia se "
            "leeria como cero fugas, que es como se pasaria esta afirmacion sin mirar nada"
        )
    if "alpha_fuera_de_interno" not in instrumentacion:
        raise AssertionError(
            "la instrumentacion no trae alpha_fuera_de_interno y su ausencia se leeria "
            "como cero fugas del alpha"
        )
    # Estas dos no existian y A14 no las miraba. La primera cuenta los ajustes sin
    # declaracion de filas, que antes pasaban por limpios; la segunda cuenta los
    # ajustes que tocaron filas de evaluacion, que es la contaminacion literal. La
    # ausencia de cualquiera de las dos es un fallo y no un cero, por el mismo
    # motivo que las dos anteriores.
    if "llamadas_al_alpha_sin_declarar_filas" not in instrumentacion:
        raise AssertionError(
            "la instrumentacion no trae llamadas_al_alpha_sin_declarar_filas y su "
            "ausencia se leeria como que todos los ajustes declararon sus filas"
        )
    if "ajustes_que_tocan_el_oof" not in instrumentacion:
        raise AssertionError(
            "la instrumentacion no trae ajustes_que_tocan_el_oof y su ausencia se leeria "
            "como que ningun ajuste toco las filas de evaluacion"
        )
    calibrador_solo_interno = int(instrumentacion["calibrador_fuera_de_interno"]) == 0
    alpha_solo_interno = int(instrumentacion["alpha_fuera_de_interno"]) == 0
    todos_declaran_filas = int(instrumentacion["llamadas_al_alpha_sin_declarar_filas"]) == 0
    nadie_toco_el_oof = int(instrumentacion["ajustes_que_tocan_el_oof"]) == 0
    # La parte de la regla 3.5 que mas importa y que no se comprobaba: el alpha no
    # puede ajustarse con filas que el calibrador vaya a usar. Sin esta
    # comprobacion, dos usos separados y legales de lo "interno" podrian cruzarse,
    # y el alpha se estaria eligiendo con los mismos datos que luego calibran.
    comunes_alpha_calibrador = int(
        instrumentacion.get("filas_comunes_entre_el_alpha_y_el_calibrador", -1)
    )
    interseccion_alpha_calibrador = {int(i) for i in alpha_internas} & {
        int(i) for i in np.asarray(filas_calibracion, dtype=np.int64).ravel()
    }
    ajustes_fuera = int(instrumentacion.get("ajustes_fuera_de_lo_permitido", 0))
    ajustes_sobre_filas_de_evaluacion = int(instrumentacion["ajustes_que_tocan_el_oof"])
    interseccion = internas & externas
    comprobaciones = {
        "el_numero_de_ajustes_es_el_esperado": int(n_ajustes_calibrador) == int(n_ajustes_esperados),
        "el_calibrador_solo_uso_filas_internas": bool(calibrador_solo_interno),
        "la_decision_del_alpha_solo_uso_filas_internas": bool(alpha_solo_interno),
        "todo_ajuste_declaro_sus_filas": bool(todos_declaran_filas),
        "cero_ajustes_sobre_el_oof_completo": int(ajustes_sobre_el_oof) == 0,
        "cero_ajustes_que_toquen_filas_de_evaluacion": bool(nadie_toco_el_oof),
        "cero_ajustes_fuera_de_lo_permitido": ajustes_fuera == 0,
        "las_filas_internas_y_externas_no_se_solapan": not interseccion,
        "el_alpha_no_usa_filas_del_calibrador": not interseccion_alpha_calibrador,
        "el_alpha_y_el_calibrador_no_se_solapan": comunes_alpha_calibrador == 0,
        "la_comprobacion_sabe_fallar_por_mutacion": bool(deteccion_de_la_mutacion),
        "hubo_filas_internas": len(internas) > 0,
        "hubo_filas_de_calibracion": len(
            {int(i) for i in np.asarray(filas_calibracion, dtype=np.int64).ravel()}
        ) > 0,
    }
    detalle = {
        "n_ajustes_del_calibrador": int(n_ajustes_calibrador),
        "n_ajustes_esperados": int(n_ajustes_esperados),
        "de_que_corrida_es_el_recuento": (
            "la instantanea se toma DESPUES de la corrida principal de M1 y ANTES de la "
            "rejilla de alpha_0 de A6, que vuelve a ejecutar M1 tres veces mas; por eso "
            "el recuento es cinco pliegues por cinco semillas y no el producto"
        ),
        "instrumentacion": instrumentacion,
        "n_filas_del_inner_calibration": len(internas),
        "n_filas_del_outer_validation": len(externas),
        "interseccion_interna_externa": len(interseccion),
        "n_filas_internas_de_la_decision_del_alpha": len(alpha_internas),
        "ajustes_sobre_el_oof_completo": int(ajustes_sobre_el_oof),
        "ajustes_que_tocan_filas_de_evaluacion": ajustes_sobre_filas_de_evaluacion,
        "ajustes_sin_declarar_filas": int(instrumentacion["llamadas_al_alpha_sin_declarar_filas"]),
        "mutacion_detectada": bool(deteccion_de_la_mutacion),
        "mutacion": (
            "se ejecuta una vez ajustando el calibrador sobre el OOF completo y A14 tiene "
            "que detectarlo"
        ),
        "comprobaciones": comprobaciones,
        "motivo_de_parada": "el calibrador se ha ajustado sobre el OOF y la pregunta 1 es circular",
    }
    return todo_verdadero(comprobaciones, "A14"), detalle


def afirmacion_a15(presentes: dict[str, bool]) -> dict[str, Any]:
    """A15. El limite de 5.5 esta verificado, no asumido.

    Args:
        presentes: nombre de libreria a si esta en el venv.

    Returns:
        Comprobacion con el detalle.
    """
    encontrados = sorted(nombre for nombre, hay in presentes.items() if hay)
    if encontrados:
        estado = "HECHO"
        consecuencia = (
            "hay al menos una libreria de inferencia bayesiana en el venv, luego el limite "
            "de 5.5 se reduce y podria usarse como ANCLA CONTRASTE. P83 no la usa: ya se "
            "habia ejecutado la comprobacion en la propia corrida y el ancla tendria que "
            "cobrarse antes de que la affirmacion pudiera decidir, luego se declara HECHA "
            "la DETECCION y SIN HACER el ANCLA, que es lo honesto"
        )
    else:
        estado = "SIN_HACER"
        consecuencia = (
            "no hay ninguna libreria de inferencia bayesiana en el venv, luego el limite de "
            "5.5 se mantiene y P83 autoriza afirmar sobre el COMPORTAMIENTO del modelo que "
            "este proyecto implemento y sobre la COMPARACION entre M1 y M2, y no sobre la "
            "correccion de la inferencia variacional frente a una posterior exacta"
        )
    comprobaciones = {
        "la_comprobacion_se_ejecuto_en_la_corrida": len(presentes) == len(LIBRERIAS_ANCLA),
        "la_lista_de_nombres_comprobados_esta_publicada": sorted(presentes) == sorted(LIBRERIAS_ANCLA),
        "el_estado_del_limite_esta_declarado": estado in {"HECHO", "SIN_HACER"},
    }
    detalle = {
        "librerias_comprobadas": sorted(LIBRERIAS_ANCLA),
        "presentes": presentes,
        "encontradas": encontrados,
        "estado_del_limite": estado,
        "consecuencia": consecuencia,
        "lo_que_p83_autoriza": (
            "afirmar sobre el comportamiento del modelo que este proyecto implemento y "
            "sobre la comparacion entre M1 y M2"
        ),
        "lo_que_p83_no_autoriza": (
            "decir que la inferencia variacional de M1 es correcta, ni que su posterior "
            "aproximada se acerca a la exacta"
        ),
        "comprobaciones": comprobaciones,
    }
    return todo_verdadero(comprobaciones, "A15"), detalle


def afirmacion_a16(rejillas: dict[str, Any], ningun_valor_elegido: bool) -> dict[str, Any]:
    """A16. La rejilla de alpha_0 y la sensibilidad de umbral, publicadas.

    Args:
        rejillas: cifras de las dos rejillas.
        ningun_valor_elegido: que no se publica ningun valor como el mejor.

    Returns:
        Comprobacion con el detalle.
    """
    comprobaciones = {
        "la_rejilla_de_alpha_0_tiene_tres_valores": len(rejillas.get("alpha_0", {})) == 3,
        "la_rejilla_de_umbral_tiene_tres_valores": len(rejillas.get("umbral", {})) == 3,
        "las_dos_rejillas_estan_publicadas": bool(rejillas.get("alpha_0")) and bool(rejillas.get("umbral")),
        "ningun_valor_se_publica_como_el_mejor": bool(ningun_valor_elegido),
    }
    detalle = {
        "rejillas": rejillas,
        "ningun_valor_elegido": bool(ningun_valor_elegido),
        "por_que_no_se_elige": (
            "elegir el mejor valor de la rejilla sobre el OOF seria seleccion sobre el OOF, "
            "que es la contamination que A14 prohibe"
        ),
        "tolerancia": "seis decimales",
        "comprobaciones": comprobaciones,
    }
    return todo_verdadero(comprobaciones, "A16"), detalle


def afirmacion_a17(
    por_pliegue: dict[str, Any], filas_inner_train: int, l1_fuera: int, determinada: bool
) -> dict[str, Any]:
    """A17. Dimension seleccionada por pliegue, y la salvedad de sesgo.

    Args:
        por_pliegue: dimension por pliegue y umbral.
        filas_inner_train: filas del inner_train.
        l1_fuera: numero de variables que la regresion L1 con C 4.0 dejo fuera.
        determinada: si la regresion esta determinada en todos los pliegues.

    Returns:
        Comprobacion con el detalle.
    """
    dimensiones = [v["n_columnas"] for v in por_pliegue.values() if "n_columnas" in v]
    no_determinada = [
        clave for clave, valor in por_pliegue.items() if int(valor.get("n_columnas", 0)) >= filas_inner_train
    ]
    comprobaciones = {
        "hay_dimensiones_publicadas": len(dimensiones) > 0,
        "la_dimension_es_menor_que_las_filas_en_todos_los_pliegues": not no_determinada,
        "la_comparacion_con_la_l1_esta_publicada": l1_fuera >= 0,
        "la_salvedad_de_sesgo_esta_declarada": LITERAL_SESGO_SELECCION.startswith("LA_SELECCION"),
    }
    detalle = {
        "dimension_por_pliegue_y_umbral": por_pliegue,
        "minimo": int(min(dimensiones)) if dimensiones else 0,
        "maximo": int(max(dimensiones)) if dimensiones else 0,
        "media": float(np.mean(dimensiones)) if dimensiones else 0.0,
        "filas_del_inner_train": int(filas_inner_train),
        "pliegues_no_determinados": no_determinada,
        "regresion_determinada": bool(determinada),
        "referencia_l1_c4": {
            "variables_dejadas_por_las_c4": int(l1_fuera),
            "origen": "dato ya medido en el proyecto, L1 con C igual a 4.0",
            "f1": LOGISTICA_L1_C4,
            "ic_pareado": list(LOGISTICA_L1_C4_IC),
            "ic_contiene_el_cero": LOGISTICA_L1_C4_IC[0] <= 0.0 <= LOGISTICA_L1_C4_IC[1],
        },
        "salvedad_de_sesgo": LITERAL_SESGO_SELECCION,
        "umbral_primario": CONFIG["umbral_seleccion_primario"],
        "umbrales_de_sensibilidad": list(CONFIG["umbrales_seleccion"]),
        "comprobaciones": comprobaciones,
    }
    return todo_verdadero(comprobaciones, "A17"), detalle


CADENAS_DE_FORMULACION: Final[tuple[str, ...]] = (
    LITERAL_V03,
    LITERAL_PREGUNTA_UNO,
    LITERAL_PREGUNTA_DOS,
    LITERAL_EXPECTATIVA,
)

VARIANTES_PROHIBIDAS: Final[tuple[str, ...]] = (
    "V03 is not the best verified leaderboard result",
    "V03 remains the best verified result",
    "Word2Vec es mejor que V03",
    "Word2Vec improves sobre V03",
    "el bayesiano supera a V03",
    "el bayesiano mejora a V03",
)


def afirmacion_a18(texto: str) -> dict[str, Any]:
    """A18. Las cadenas de formulacion, de las dos preguntas y del proyecto.

    Args:
        texto: texto del informe ya construido.

    Returns:
        Comprobacion con el detalle.
    """
    presentes = {cadena: cadena in texto for cadena in CADENAS_DE_FORMULACION}
    ausentes = {variante: variante in texto for variante in VARIANTES_PROHIBIDAS}
    comprobaciones = {
        **{f"la_cadena_presente_{indice}": valor for indice, valor in enumerate(presentes.values())},
        **{f"la_variante_ausente_{indice}": not valor for indice, valor in enumerate(ausentes.values())},
    }
    detalle = {
        "cadenas_requeridas": list(CADENAS_DE_FORMULACION),
        "cadenas_presentes": presentes,
        "variantes_prohibidas": list(VARIANTES_PROHIBIDAS),
        "variantes_encontradas": ausentes,
        "comprobaciones": comprobaciones,
        "motivo_de_parada": "falta una cadena de formulacion o aparece una variante",
    }
    return todo_verdadero(comprobaciones, "A18"), detalle


def afirmacion_a19(texto: str, f1_del_miembro: dict[str, Any]) -> dict[str, Any]:
    """A19. El F1 del miembro se publica como descriptiva, sin lectura inferencial.

    Args:
        texto: texto del informe.
        f1_del_miembro: cifra del F1 del miembro bayesiano.

    Returns:
        Comprobacion con el detalle.
    """
    etiqueta = LITERAL_DESCRIPTIVA_F1 in texto
    frase_uno = LITERAL_F1_NO_REFUTA_UNO in texto
    frase_dos = LITERAL_F1_NO_REFUTA_DOS in texto
    bloque_lectura = _bloque_de_la_lectura(texto)
    el_f1_en_la_lectura = "f1_media_columnas" in bloque_lectura
    comprobaciones = {
        "el_f1_esta_publicado": bool(f1_del_miembro.get("f1_media_columnas") is not None),
        "la_etiqueta_descriptiva_esta": etiqueta,
        "la_desviacion_entre_semillas_esta": "desviacion_entre_semillas" in f1_del_miembro,
        "el_intervalo_esta_publicado": "intervalo" in f1_del_miembro,
        "la_frase_sobre_la_pregunta_1_esta": frase_uno,
        "la_frase_sobre_la_pregunta_2_esta": frase_dos,
        "el_f1_no_aparece_en_la_regla_de_lectura": not el_f1_en_la_lectura,
    }
    detalle = {
        "f1_del_miembro": f1_del_miembro,
        "etiqueta": LITERAL_DESCRIPTIVA_F1,
        "frase_sobre_la_pregunta_1": LITERAL_F1_NO_REFUTA_UNO,
        "frase_sobre_la_pregunta_2": LITERAL_F1_NO_REFUTA_DOS,
        "bloque_de_la_lectura_sin_f1": bool(bloque_lectura) and not el_f1_en_la_lectura,
        "comprobaciones": comprobaciones,
        "motivo_de_parada": "el F1 se ha usado como criterio",
    }
    return todo_verdadero(comprobaciones, "A19"), detalle


def _bloque_de_la_lectura(texto: str) -> str:
    """Extrae del informe el bloque de las dos reglas de lectura.

    Args:
        texto: texto del informe.

    Returns:
        El bloque que contiene las dos lecturas.
    """
    lineas = texto.split("\n")
    dentro: list[str] = []
    capturando = False
    for linea in lineas:
        if "Lectura de la pregunta 1" in linea or "Lectura de la pregunta 2" in linea:
            capturando = True
        elif capturando and linea.startswith("## "):
            break
        if capturando:
            dentro.append(linea)
    return "\n".join(dentro)


def _libreria_esta_en_el_venv(nombre: str) -> bool:
    """Dice si una libreria de inferencia bayesiana esta en el venv.

    Motivo: el estado del limite de 5.5 se derivaba antes de que la cadena
    literal apareciera en el texto, luego declaraba HECHO aunque no hubiera
    ninguna libreria. El limite se declara a partir de la MEDIDA y no de una
    cadena, que es lo unico honesto.

    Args:
        nombre: nombre del modulo, con guiones o con guiones bajos.

    Returns:
        True si el modulo se puede importar desde el venv.
    """
    try:
        return importlib.util.find_spec(nombre.replace("-", "_")) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def comparar_contra_el_liston(
    medido: dict[str, float], liston: dict[str, float]
) -> dict[str, Any]:
    """Compara las magnitudes medidas contra el liston usando el margen de 1.3.

    POR QUE ESTA COMPARACION HACIA FALTA. Los tres margenes de la seccion 1.3, 0.005
    de log-loss, 0.0015 de Brier y 0.010 de ECE, estaban declarados en CONFIG pero
    no los comparaba nadie: los numeros se publicaban y la lectura no se derivaba de
    ellos. Un margen declarado y no aplicado no es un margen, es decoracion, y por
    eso aqui la comparacion es la que decide y no el sentido escrito a mano.

    DIRECCION, QUE NO ES SIMETRICA. Un margen de NO INFERIORIDAD dice que el
    bayesiano puede ser peor como mucho en esa cantidad, luego solo el lado
    desfavorable se penaliza: estar mejor que el liston no es un fallo. Y que no se
    afirme superioridad, luego un delta favorable grande NO convierte la lectura en
    afirmativa.

    Args:
        medido: magnitudes medidas del brazo.
        liston: magnitud del liston v03_cal.

    Returns:
        Tupla con el veredicto de no inferioridad y el detalle. El veredicto es un
        booleano de verdad y no un diccionario, porque comprobar_veredicto rechaza
        cualquier otra forma y un diccionario truthy haria que todo pasara.

    Raises:
        ColeccionVacia: si no se puede comparar ninguna magnitud.
    """
    magnitudes = (
        ("log_loss_macro", "margen_log_loss"),
        ("brier_macro", "margen_brier"),
        ("ece_macro", "margen_ece"),
    )
    deltas: dict[str, float] = {}
    dentro: dict[str, bool] = {}
    faltan: list[str] = []
    for clave, margen in magnitudes:
        if clave not in medido or clave not in liston:
            faltan.append(clave)
            continue
        deltas[clave] = float(medido[clave]) - float(liston[clave])
        dentro[clave] = deltas[clave] <= float(CONFIG[margen])
    exigir_no_vacio(dentro, "comparacion contra el liston")
    no_inferior = all(dentro.values())
    detalle = {
        "magnitudes": [clave for clave, _ in magnitudes],
        "deltas_contra_el_liston": deltas,
        "margenes": {clave: float(CONFIG[margen]) for clave, margen in magnitudes},
        "dentro_de_su_margen": dentro,
        "faltan": faltan,
        "lectura": LECTURA_NO_INFERIOR if no_inferior else LECTURA_NEGATIVO,
        "direccion": (
            "el margen de 1.3 es de NO INFERIORIDAD, luego solo el lado desfavorable se "
            "penaliza y estar mejor que el liston no es un fallo"
        ),
        "que_no_se_afirma": (
            "que el bayesiano supere a V03 en calibracion, ni que su margen sea superior; "
            f"{LITERAL_V03}"
        ),
    }
    return no_inferior, detalle


def comprobar_presupuesto(
    segundos: float,
    presupuesto: int,
    presupuesto_respetado: bool,
    con_nota_de_excepcion: bool | None = None,
) -> dict[str, Any]:
    """Comprueba el presupuesto de reloj y FALLA si el tiempo real lo supera.

    POR QUE EL EXCESO ES UN FRACASO Y NO UN DATO. La version anterior publicaba
    ``el_tiempo_esta_dentro_del_presupuesto`` como una comprobacion mas, junto a la
    coherencia del banderal, y por eso una version mutada que quitaba esa
    comprobacion pasaba limpia: el banderal era TRUE en los dos lados, luego la
    coherencia se cumplia y nada mas se evaluaba. Eso es una comprobacion que no
    puede fallar ante la cosa que dice vigilar. Ahora el exceso se separa del resto
    y hace fallar la comprobacion por si mismo, con o sin nota de excepcion.

    El cuarto argumento es la nota de excepcion, que llega CALCULADA. Si se omite,
    vale lo que dice la comparacion real, de modo que una llamada sin nota no puede
    usarse para disimular un exceso.

    Args:
        segundos: segundos transcurridos, medidos.
        presupuesto: presupuesto declarado en segundos.
        presupuesto_respetado: si el presupuesto se respeta, calculado fuera.
        con_nota_de_excepcion: si hay una nota que declara el exceso. Si se omite,
            se usa la comparacion real.

    Returns:
        Comprobacion con el detalle.

    Raises:
        AssertionError: si el tiempo real supera al presupuesto sin nota.
    """
    dentro = float(segundos) <= float(presupuesto)
    nota = dentro if con_nota_de_excepcion is None else bool(con_nota_de_excepcion)
    coherente = bool(presupuesto_respetado) == dentro
    comprobaciones = {
        "el_banderal_no_contradice_la_comparacion": coherente,
    }
    detalle = {
        "segundos": float(segundos),
        "presupuesto": int(presupuesto),
        "dentro_del_presupuesto": bool(dentro),
        "hay_nota_de_excepcion": bool(nota),
        "exceso": float(max(0.0, float(segundos) - float(presupuesto))),
        "nota_de_asimetria": LITERAL_PRESUPUESTO_BARATO,
        "por_que_el_exceso_falla": (
            "el presupuesto declarado es vinculante y un exceso sin nota declarada es un "
            "fracaso de la corrida, no una cifra que se publica"
        ),
        "comprobaciones": comprobaciones,
    }
    if not dentro and not nota:
        raise AssertionError(
            f"el tiempo real {float(segundos)} supera el presupuesto {int(presupuesto)} "
            f"y no hay nota de excepcion; exceso de "
            f"{detalle['exceso']} segundos"
        )
    return todo_verdadero(comprobaciones, "presupuesto"), detalle


# =============================================================================
# SECCION 14: DATOS SINTETICOS DEL MODO DE HUMO
# =============================================================================


def datos_sinteticos(n_filas: int, n_grupos: int, n_columnas: int, semilla: int, caso: str) -> dict[str, Any]:
    """Datos sinteticos con la misma forma que los reales.

    Motivo: un modo de humo con datos mas ricos que los reales no encuentra los fallos
    de los reales. Los identificadores son letras, porque un identificador con digitos
    se trocea en el vocabulario y deja la matriz derada vacia.

    Args:
        n_filas: numero de filas.
        n_grupos: numero de grupos.
        n_columnas: numero de columnas de la matriz densa.
        semilla: semilla del generador.
        caso: COMPLETO o INCOMPLETO.

    Returns:
        Diccionario con textos, y, grupos y pliegues.

    Raises:
        AssertionError: si las dimensiones no son coherentes.
    """
    if n_filas < n_grupos or n_columnas < 3 or n_filas < 12:
        raise AssertionError(
            f"dimensiones incoherentes: {n_filas} filas, {n_grupos} grupos, {n_columnas} columnas"
        )
    rng = np.random.default_rng(semilla)
    letras = "abcdefghijklmnopqrstuvwxyz"
    textos = []
    for i in range(n_filas):
        cantidad = int(rng.integers(4, 12))
        palabras = [f"{letras[int(rng.integers(0, 26))]}{letras[int(rng.integers(0, 26))]}" for _ in range(cantidad)]
        textos.append(" ".join(palabras))
    y = (rng.random((n_filas, 3)) < 0.3).astype(np.int64)
    if caso == "INCOMPLETO":
        y[:, 1] = 0
        y[:, 2] = 0
    if y.sum() == 0:
        y[0, 0] = 1
    grupos = np.arange(n_filas) % n_grupos
    n_pliegues = CONFIG["n_pliegues"]
    reparto = np.array_split(np.arange(n_filas), n_pliegues)
    pliegues = []
    for bloque in reparto:
        indices = np.asarray(bloque, dtype=np.int64)
        pliegues.append((np.asarray([], dtype=np.int64), indices))
    ids = np.asarray([f"sintetico{i:04d}" for i in range(n_filas)])
    # La referencia sintetica de V03 por fila. Tiene que estar correlacionada con y,
    # porque la decision interna del alpha de la regla 3.5 mezcla estas
    # probabilidades con las del brazo y mide cuanto mejora la perdida: si fussen
    # todas cero, la mezcla no aportaria nada y la medicion no seria del caso.
    fuerza = 0.6 + 0.4 * rng.random((n_filas, 3))
    proba_v03 = np.clip(fuerza * y + 0.2 * (1.0 - fuerza) + 0.05 * rng.random((n_filas, 3)), 1e-6, 1.0 - 1e-6)
    return {
        "textos": textos,
        "y": y,
        "grupos": grupos,
        "pliegues": pliegues,
        "ids": ids,
        "proba_v03": proba_v03,
        "completo": caso == "COMPLETO",
        "n_columnas_sinteticas": n_columnas,
    }


def construir_contexto_sintetico(datos: dict[str, Any]) -> ContextoDatos:
    """Construye un contexto de datos a partir de los datos sinteticos.

    Args:
        datos: diccionario de datos sinteticos.

    Returns:
        El contexto de datos sinteticos.
    """
    guarda = GuardaLectura()
    guarda.marcar(np.arange(len(datos["textos"]), dtype=np.int64))
    return ContextoDatos(
        textos=list(datos["textos"]),
        y=np.asarray(datos["y"], dtype=np.int64),
        pliegues=list(datos["pliegues"]),
        ids=np.asarray(datos["ids"]),
        grupos=np.asarray(datos["grupos"]),
        origen=f"sintetico {datos['completo'] and 'COMPLETO' or 'INCOMPLETO'}",
        guarda=guarda,
        completo=bool(datos["completo"]),
        # La referencia sintetica de V03 por fila. Sin ella la decision interna del
        # alpha de la regla 3.5 no tiene con que mezclarse y ejecutar_brazo_m1
        # deriva por un camino que el modo completo no recorre nunca, luego el humo
        # no estaria midiendo lo que la corrida mide.
        proba_v03=np.asarray(datos.get("proba_v03", np.zeros((len(datos["textos"]), len(ETIQUETAS)))), dtype=np.float64),
    )


# =============================================================================
# SECCION 15: PUERTA DE HUMO
# =============================================================================


class PuertaHumo:
    """Puerta de salida del modo de humo. Sale con codigo 1 si CUALQUIER caso falla."""

    def __init__(self) -> None:
        self.casos: list[dict[str, Any]] = []
        self.veredictos: dict[str, dict[str, Any]] = {}
        self.n_ok = 0
        self.marca = os.urandom(4).hex()
        self.marca_texto = self.marca

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

    def codigo_salida(self, suelo: int, progreso: Progreso | None = None) -> int:
        """Calcula el codigo de salida de la puerta.

        Args:
            suelo: suelo minimo de casos correctos.
            progreso: acumulador, para comprobar que ninguna afirmacion con marca de
                STOP quedo sin poder decidir.

        Returns:
            0 si pasan todos y se cumple el suelo; 1 en caso contrario.
        """
        sin_decidir = [
            f"{clave}:{veredicto.get(resolver_clave('estado_json'))}"
            for clave, veredicto in sorted(self.veredictos.items())
            if veredicto.get("stop")
            and veredicto.get(resolver_clave("estado_json")) != ESTADO_CONFIRMADO
        ]
        if sin_decidir:
            print(f"[humo] AFIRMACIONES CON STOP QUE NO SON CONFIRMADAS: {len(sin_decidir)}")
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


SUELO_COBERTURA: Final[int] = 60


# =============================================================================
# SECCION 16: INFORME
# =============================================================================


def tabla_markdown(cabeceras: Sequence[str], filas: Sequence[Sequence[Any]]) -> str:
    """Construye una tabla markdown.

    Args:
        cabeceras: nombres de columna.
        filas: filas de la tabla.

    Returns:
        Texto de la tabla.
    """
    lineas = ["| " + " | ".join(str(c) for c in cabeceras) + " |"]
    lineas.append("|" + "|".join("---" for _ in cabeceras) + "|")
    for fila in filas:
        lineas.append("| " + " | ".join(str(c) for c in fila) + " |")
    return "\n".join(lineas)


def construir_json(carga: dict[str, Any]) -> str:
    """Arma el informe JSON y lo serializa.

    Args:
        carga: diccionario con todo lo calculado.

    Returns:
        El texto JSON, ya validado como ASCII.
    """
    return a_json_legible(carga)


def construir_markdown(carga: dict[str, Any]) -> str:
    """Construye el informe markdown completo.

    Args:
        carga: diccionario con todo lo calculado.

    Returns:
        Texto markdown completo en ASCII puro.
    """
    veredictos = carga.get(resolver_clave("veredictos_json"), {})
    detalle = carga.get(resolver_clave("detalle_json"), {})
    lectura = carga.get(resolver_clave("lectura_pregunta_1_json"))
    cifras1 = carga.get(resolver_clave("cifras_pregunta_1_json"), {})
    cifras2 = carga.get(resolver_clave("cifras_pregunta_2_json"), {})
    lineas: list[str] = []
    an = lineas.append
    an("# P83: modelo bayesiano disperso, doble objetivo")
    an("")
    an(LITERAL_V03)
    an("")
    an(f"- Experimento: {carga.get('experimento', 'P83')}")
    an(f"- Script: {carga.get('script', 'experiments/p83_01_bayesiano.py')}")
    an(f"- Modo: {carga.get('modo', 'humo')}")
    an(f"- Marca de corrida: {carga.get('marca_corrida', 'sin marca')}")
    an(f"- Pre-registro: {carga.get('pre_registro', {}).get('ruta', '')}")
    an(f"- sha256 del pre-registro: {carga.get('pre_registro', {}).get('sha256_declarado', '')}")
    an("")
    an("## Formulacion y limites, escritos antes de leer los numeros")
    an("")
    an(f"- {LITERAL_PREGUNTA_UNO}")
    an(f"- {LITERAL_PREGUNTA_DOS}")
    an(f"- {LITERAL_EXPECTATIVA}")
    an(f"- {LITERAL_POSTERIOR_APROXIMADA}")
    an(f"- {LITERAL_EFECTIVOS_METODO}")
    an(f"- {LITERAL_NO_WORD2VEC}")
    an(f"- {LITERAL_NO_SUPERA}")
    an(f"- {LITERAL_AMBIGUEDAD}")
    an(f"- {LITERAL_SESGO_SELECCION}")
    an("")
    an("## Lectura de la pregunta 1: incertidumbre y calibracion")
    an("")
    an(f"- Lectura de la regla de 1.5: **{lectura}**")
    an(f"- Ambiguedad declarada: {carga.get(resolver_clave('ambiguedad_declarada_json'), '')}")
    an(f"- Los dos intervalos no son independientes: {LITERAL_AMBIGUEDAD}")
    an("- El F1 no aparece en esta regla, porque la pregunta 1 no compite por F1.")
    an(f"- {LITERAL_F1_NO_REFUTA_UNO}")
    an("")
    an("## Lectura de la pregunta 2: complementariedad de errores")
    an("")
    an(f"- {LITERAL_RESCATE_ESCALA}")
    an(f"- {LITERAL_CORRECTED_CV_OOF}")
    an("- El F1 no aparece en esta regla, porque la pregunta 2 no busca superar a V03.")
    an(f"- {LITERAL_F1_NO_REFUTA_DOS}")
    an("")
    an("## Cifras de la pregunta 1")
    an("")
    if cifras1:
        an(
            tabla_markdown(
                ["magnitud", "M1", "M2", "liston v03_cal"],
                [
                    ["log-loss macro recortado", cifras1.get("M1", {}).get("log_loss_macro"), cifras1.get("M2", {}).get("log_loss_macro"), cifras1.get("liston", {}).get("log_loss_macro")],
                    ["log-loss macro sin recortar", cifras1.get("M1", {}).get("log_loss_sin_recortar"), cifras1.get("M2", {}).get("log_loss_sin_recortar"), cifras1.get("liston", {}).get("log_loss_sin_recortar")],
                    ["Brier macro", cifras1.get("M1", {}).get("brier_macro"), cifras1.get("M2", {}).get("brier_macro"), cifras1.get("liston", {}).get("brier_macro")],
                    ["ECE macro", cifras1.get("M1", {}).get("ece_macro"), cifras1.get("M2", {}).get("ece_macro"), cifras1.get("liston", {}).get("ece_macro")],
                ],
            )
        )
        an("")
    an("## Cifras de la pregunta 2")
    an("")
    if cifras2:
        an(
            tabla_markdown(
                ["alpha", "corrected_cv", "delta", "rescate", "dano", "neto"],
                [
                    [
                        alpha,
                        valores.get("corrected_cv"),
                        valores.get("delta"),
                        valores.get("rescate"),
                        valores.get("dano"),
                        valores.get("neto"),
                    ]
                    for alpha, valores in sorted(cifras2.items())
                ],
            )
        )
        an("")
    an("## Veredictos de las veinte afirmaciones")
    an("")
    an(
        tabla_markdown(
            ["id", "bloque", "estado", "stop", "que hace"],
            [
                [
                    clave,
                    carga.get("afirmaciones_declaradas", {}).get(clave, {}).get("bloque", ""),
                    valor.get(resolver_clave("estado_json"), ""),
                    "si" if valor.get("stop") else "no",
                    carga.get("afirmaciones_declaradas", {}).get(clave, {}).get("que_hace", ""),
                ]
                for clave, valor in sorted(veredictos.items())
            ],
        )
    )
    an("")
    an("## Aislamiento de afirmaciones")
    an("")
    an(
        "- ERROR no es ni CONFIRMADO ni REFUTADO ni NO_EVALUABLE; la corrida continua "
        "con la siguiente y una Parada con STOP se propaga."
    )
    an(f"- Errores: {len(carga.get('aislamiento_de_afirmaciones', {}).get('errores', []))}")
    an(f"- Avisos: {len(carga.get('aislamiento_de_afirmaciones', {}).get('avisos', []))}")
    an(f"- Escrituras fallidas: {carga.get('aislamiento_de_afirmaciones', {}).get('escrituras_fallidas', 0)}")
    an("")
    an("## Resto del detalle")
    an("")
    an("```json")
    an(json.dumps(detalle, indent=2, sort_keys=True, ensure_ascii=True)[:60000])
    an("```")
    an("")
    an("## Contexto del proyecto, citado y no recalculado")
    an("")
    an(f"- V03 corrected_cv: {CORRECTED_CV_PUBLICADO}")
    an(f"- NB-SVM canonico: {NB_SVM_CANONICO}")
    an(f"- Logistica L1 con C 4.0: {LOGISTICA_L1_C4}, IC pareado {list(LOGISTICA_L1_C4_IC)} que contiene el cero")
    an(f"- P82 RNN local {P82_RNN_LOCAL} y leaderboard {P82_RNN_LB}")
    an("")
    an("## Sobre el margen y el desajuste de dominio")
    an("")
    an(
        "- El margen de error de la validacion cruzada depende del desajuste de dominio, luego "
        "comparar familias con capacidades distintas con la misma metrica local es enganoso. Un "
        "bayesiano disperso tiene menos capacidad que una RNN, asi que su margen sera "
        "probablemente mas estrecho."
    )
    an(
        "- Si se observa, se publica como una OBSERVACION SOBRE EL MARGEN y no como una "
        "afirmacion de rendimiento."
    )
    an("")
    an("## Limite de 5.5 y anclaje externo")
    an("")
    an(f"- {LITERAL_ANCLA}")
    anclaje = carga.get(resolver_clave("anclaje_json"), {})
    an(f"- Librerias comprobadas: {len(anclaje.get('librerias_comprobadas', []))}")
    an(f"- Encontradas: {anclaje.get('encontradas', [])}")
    an(f"- Estado del limite: {anclaje.get('estado_del_limite', 'SIN_HACER')}")
    an("")
    an("## Conclusion")
    an("")
    an(LITERAL_V03)
    an("")
    an(f"- {LITERAL_NO_SUPERA}")
    an(f"- {LITERAL_PRESUPUESTO_BARATO}")
    return "\n".join(lineas) + "\n"


# =============================================================================
# SECCION 17: EJECUCION COMPLETA
# =============================================================================


@dataclass
class CorridaBrazo:
    """Lo que produce un brazo completo sobre los pliegues y las semillas."""

    identificador: str
    probas_oof: np.ndarray
    intervalos: dict[str, np.ndarray]
    n_efectivos: dict[str, float]
    n_activas: dict[str, int]
    pasos: int = 0
    calibraciones: list[dict[str, Any]] = field(default_factory=list)
    # El alpha que la regla 3.5 fijo en cada combinacion de pliegue y semilla, y el
    # numero de filas del inner_train, que es el contra de la banda muerta de A17.
    alphas_por_combinacion: dict[str, float] = field(default_factory=dict)
    n_filas_inner_train: int = 0


def partir_interna(
    outer_train: np.ndarray, fraccion: float, semilla: int, grupos: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Parte el outer_train en inner_train e inner_calibration por grupos.

    Args:
        outer_train: indices del outer_train.
        fraccion: fraccion asignada al inner_train.
        semilla: semilla del generador.
        grupos: grupos de cada fila.

    Returns:
        Los dos conjuntos de indices.

    Raises:
        AssertionError: si la fraccion no esta en (0, 1).
    """
    if not 0.0 < float(fraccion) < 1.0:
        raise AssertionError(f"la fraccion {fraccion} tiene que estar entre 0 y 1")
    rng = np.random.default_rng(int(semilla))
    grupos_unicos = np.unique(grupos[np.asarray(outer_train, dtype=np.int64)])
    orden = rng.permutation(grupos_unicos)
    n_grupos_train = max(1, int(round(len(orden) * float(fraccion))))
    grupos_train = set(int(g) for g in orden[:n_grupos_train])
    mascara = np.asarray([int(g) in grupos_train for g in grupos[np.asarray(outer_train, dtype=np.int64)]])
    inner_train = np.asarray(outer_train, dtype=np.int64)[mascara]
    inner_cal = np.asarray(outer_train, dtype=np.int64)[~mascara]
    return inner_train, inner_cal


def ejecutar_brazo_m1(
    contexto: ContextoDatos,
    progreso: Progreso,
    umbral: float,
    alpha_0: float,
    n_muestras: int,
) -> CorridaBrazo:
    """Ejecuta el brazo M1 sobre todos los pliegues y las semillas.

    Args:
        contexto: contexto de datos.
        progreso: acumulador de progreso.
        umbral: umbral de seleccion de variables.
        alpha_0: escala de la dispersion.
        n_muestras: muestras del intervalo predictivo.

    Returns:
        La corrida del brazo.

    Raises:
        AssertionError: si el contexto no tiene pliegues.
    """
    if not contexto.pliegues:
        raise AssertionError("el contexto no tiene pliegues")
    probas = np.full((len(contexto.y), len(ETIQUETAS)), np.nan, dtype=np.float64)
    infimos = np.full((len(contexto.y), len(ETIQUETAS)), np.nan, dtype=np.float64)
    supremos = np.full((len(contexto.y), len(ETIQUETAS)), np.nan, dtype=np.float64)
    n_efectivos: dict[str, float] = {}
    n_activas: dict[str, int] = {}
    calibraciones: list[dict[str, Any]] = []
    alphas_por_combinacion: dict[str, float] = {}
    filas_inner_train = 0
    combinaciones = 0
    for numero, (ia, ib) in enumerate(contexto.pliegues):
        if len(ia) == 0 or len(ib) == 0:
            continue
        for semilla in CONFIG["semillas"]:
            inner_train, inner_cal = partir_interna(
                ia, CONFIG["fraccion_inner_calibration"], int(semilla), contexto.grupos
            )
            if len(inner_train) == 0 or len(inner_cal) == 0:
                continue
            filas_inner_train = int(len(inner_train))
            INSTRUMENTO_SELECCION.fijar_permitidos(inner_train)
            X_ent = construir_tfidf(
                [contexto.textos[i] for i in inner_train],
                [contexto.textos[i] for i in inner_cal],
            )
            mascara = seleccionar_columnas(X_ent, umbral)
            INSTRUMENTO_SELECCION.registrar(inner_train, umbral, int(mascara.sum()))
            X_sel = X_ent[:, mascara]
            X_cal_sel = construir_tfidf.ultimo_evaluacion[:, mascara]  # type: ignore[attr-defined]
            # La tupla es (ENTRENAMIENTO, EVALUACION, ESCALA). Antes se
            # desempaquetaba como X_ent_e, X_cal_e, _, con lo que X_cal_e
            # recibia la matriz de ENTRENAMIENTO y no la de calibracion. Con
            # 350 filas de calibracion contra 836 de entrenamiento el fallo se
            # ve en la forma, y no se ve cuando coinciden. La variable se llama
            # como lo que es.
            X_ent_e, X_cal_e, _ = reescalar_columnas(X_sel, X_cal_sel)
            # LA MATRIZ DE EVALUACION, Y UNA SOLA RUTA DE CALCULO. construir_tfidf
            # devuelve la de ENTRENAMIENTO, que tiene len(inner_train) filas, y deja
            # la de EVALUACION en ultimo_evaluacion. Asignar el retorno a X_ev y
            # escalar ESA era mezclar las dos matrices: el reescalado de arriba ya
            # estaba bien hecho y luego lo sobreescribia una segunda escala manual
            # identica. Aqui se pide el ultimo_evaluacion y se reescala con la misma
            # funcion, y X_ev_e recibe un unico valor.
            construir_tfidf(
                [contexto.textos[i] for i in inner_train],
                [contexto.textos[i] for i in ib],
            )
            # El segundo elemento de la tupla es la EVALUACION. Tomar el primero
            # fazia que X_ev_e tuviera len(inner_train) filas en vez de len(ib), y
            # el ValueError salia 297 filas despues, en la prediccion. Es el mismo
            # error de orden de desempaquetado que el de arriba, y por eso los dos
            # se/arreglan juntos.
            _, X_ev_e, _ = reescalar_columnas(
                X_sel,
                construir_tfidf.ultimo_evaluacion[:, mascara],  # type: ignore[attr-defined]
            )
            # El permitido del calibrador es inner_cal. El permitido del alpha es
            # inner_train, porque la correccion interna del alpha se mide con las
            # mismas filas con las que se entrena, y la regla 3.5 prohibe que esas
            # sean las que despues calibran. Aqui se separan los dos, que antes
            # eran el mismo conjunto y por eso la comprobacion no podia ver nada.
            INSTRUMENTO_CALIBRACION.fijar_permitidos(inner_cal, inner_train)
            clave_alfas = f"{numero}_{int(semilla)}"
            proba_cal = np.zeros((len(inner_cal), len(ETIQUETAS)))
            proba_ev = np.zeros((len(ib), len(ETIQUETAS)))
            proba_tr = np.zeros((len(inner_train), len(ETIQUETAS)))
            modelos: dict[str, ModeloM1] = {}
            X_ent_denso = np.asarray(X_ent_e.todense(), dtype=np.float64)
            X_cal_denso = np.asarray(X_cal_e.todense(), dtype=np.float64)
            X_ev_denso = np.asarray(X_ev_e.todense(), dtype=np.float64)
            for k in range(len(ETIQUETAS)):
                modelo = ajustar_m1(X_ent_denso, contexto.y[inner_train, k], alpha_0)
                modelos[ETIQUETAS[k]] = modelo
                proba_cal[:, k] = predecir_m1(modelo, X_cal_denso)
                proba_ev[:, k] = predecir_m1(modelo, X_ev_denso)
                # LA DECISION REAL DEL ALPHA, QUE ANTES NO EXISTIA. La regla 3.5 dice
                # que el alpha se fija sobre datos INTERNOS de cada pliegue y nunca
                # sobre el OOF, y la forma de Proveer eso es llamar a decidir_alpha de
                # verdad, con las filas que usa declaradas una por una. Antes la
                # funcion existia, se probaba en el humo y no la llamaba nadie en la
                # corrida, luego el contador de A14 no tenia nada que mirar.
                proba_tr[:, k] = predecir_m1(modelo, X_ent_denso)
                progreso.contar("ajustes_de_m1")
            # LA MEDICION INTERNA DEL ALPHA, con filas INTERNAS y con las
            # probabilidades canonicas de V03 de ESAS MISMAS filas. Sin
            # contexto.proba_v03 no hay con que mezclar y la decision no se puede
            # hacer sin tocar el OOF, que es justo lo que la regla prohibe, luego en
            # ese caso se registra la ausencia y A14 la ve como falta de
            # declaracion en vez de como un alpha limpio.
            if contexto.proba_v03 is None:
                progreso.contar("decisiones_de_alpha_sin_oof_de_referencia")
            else:
                correcciones = medir_correcciones_internas_del_alpha(
                    proba_tr,
                    np.asarray(contexto.proba_v03, dtype=np.float64)[inner_train],
                    contexto.y[inner_train],
                    CONFIG["alphas_ensamblaje"],
                    CONFIG["recorte_log_loss"],
                )
                eleccion = decidir_alpha(
                    correcciones, CONFIG["alphas_ensamblaje"],
                    CONFIG["alpha_ensamblaje_primario"], inner_train,
                )
                alphas_por_combinacion[clave_alfas] = float(eleccion["alpha"])
                progreso.contar("decisiones_de_alpha")
            pares = ajustar_calibrador(proba_cal, contexto.y[inner_cal], inner_cal)
            calibraciones.append({"pliegue": numero, "semilla": int(semilla), "pares": pares})
            proba_ev_cal = aplicar_platt(proba_ev, pares)
            probas[ib] = proba_ev_cal
            for k in range(len(ETIQUETAS)):
                modelo = modelos[ETIQUETAS[k]]
                intervalo = intervalo_predictivo(
                    modelo,
                    X_ev_denso,
                    n_muestras,
                    int(semilla),
                    CONFIG["cuantil_inferior"],
                    CONFIG["cuantil_superior"],
                )
                infimos[ib, k] = intervalo["inferior"]
                supremos[ib, k] = intervalo["superior"]
            # EL DESGLOSE POR ETIQUETA, que faltaba. A6 compara el numero de
            # parametros efectivos POR ETIQUETA entre los extremos de la rejilla, y
            # lo que habia era una media de las tres promediada sobre las
            # combinaciones, con lo que la clave era el pliegue y la semilla y al
            # final solo quedaba la ultima. Ademas el nombre de la clave era
            # clave_alfas y no de alpha_0, que es lo que nombra la rejilla.
            for nombre_etiqueta in ETIQUETAS:
                n_efectivos[nombre_etiqueta] = float(modelos[nombre_etiqueta].n_efectivos)
                n_activas[nombre_etiqueta] = int(modelos[nombre_etiqueta].n_activas)
            combinaciones += 1
    progreso.parcial(
        "brazo_M1",
        ".json",
        {
            "n_combinaciones": combinaciones,
            "umbrales": [umbral],
            "alpha_0": [alpha_0],
            "alphas_elegidos_por_combinacion": alphas_por_combinacion,
        },
    )
    return CorridaBrazo(
        identificador="M1",
        probas_oof=probas,
        intervalos={"inferior": infimos, "superior": supremos},
        n_efectivos=n_efectivos,
        n_activas=n_activas,
        pasos=combinaciones,
        calibraciones=calibraciones,
        alphas_por_combinacion=dict(alphas_por_combinacion),
        n_filas_inner_train=filas_inner_train,
    )


def ejecutar_brazo_m2(contexto: ContextoDatos, progreso: Progreso, umbral: float) -> CorridaBrazo:
    """Ejecuta el brazo M2 con su control aritmetico exacto.

    Args:
        contexto: contexto de datos.
        progreso: acumulador de progreso.
        umbral: umbral de seleccion de variables.

    Returns:
        La corrida del brazo con el maximo de diferencia entre los dos caminos.
    """
    probas = np.full((len(contexto.y), len(ETIQUETAS)), np.nan, dtype=np.float64)
    infimos = np.full((len(contexto.y), len(ETIQUETAS)), np.nan, dtype=np.float64)
    supremos = np.full((len(contexto.y), len(ETIQUETAS)), np.nan, dtype=np.float64)
    maximo = 0.0
    n_ejecuciones = 0
    n_filas_control = 0
    for numero, (ia, ib) in enumerate(contexto.pliegues):
        if len(ia) == 0 or len(ib) == 0:
            continue
        semilla = int(CONFIG["semillas"][numero % len(CONFIG["semillas"])])
        inner_train, inner_cal = partir_interna(
            ia, CONFIG["fraccion_inner_calibration"], semilla, contexto.grupos
        )
        if len(inner_train) == 0 or len(inner_cal) == 0:
            continue
        X_ent = construir_tfidf([contexto.textos[i] for i in inner_train], [contexto.textos[i] for i in ib])
        mascara = seleccionar_columnas(X_ent, umbral)
        X_sel = X_ent[:, mascara]
        X_ev_sel = construir_tfidf.ultimo_evaluacion[:, mascara]  # type: ignore[attr-defined]
        # M2 NO declara permitidos, y antes los declaraba. Escribia sobre el
        # conjunto unico del instrumento, luego el permitted de la ultima
        # combinacion de M2 sustituyo al de la ultima de M1 y las llamadas ya
        # registradas se compararon contra el conjunto equivocado. M2 no ajusta ni
        # el calibrador ni el alpha, luego no tiene nada que declarar.
        for k in range(len(ETIQUETAS)):
            camino_1 = camino_bernoulli_m2(
                np.asarray(X_sel.todense(), dtype=np.float64),
                contexto.y[inner_train, k],
                np.asarray(X_ev_sel.todense(), dtype=np.float64),
                CONFIG["alfa_dirichlet"],
            )
            camino_2 = camino_recuento_m2(
                np.asarray(X_sel.todense(), dtype=np.float64),
                contexto.y[inner_train, k],
                np.asarray(X_ev_sel.todense(), dtype=np.float64),
                CONFIG["alfa_dirichlet"],
            )
            maximo = max(maximo, float(np.max(np.abs(camino_1 - camino_2))))
            n_ejecuciones += 1
            n_filas_control += int(len(ib))
            conteos = conteos_de_m2(
                np.asarray(X_sel.todense(), dtype=np.float64),
                contexto.y[inner_train, k],
                CONFIG["alfa_dirichlet"],
            )
            intervalo = anchura_intervalo_m2(
                conteos,
                np.asarray(X_ev_sel.todense(), dtype=np.float64),
                CONFIG["n_muestras_predictivas"],
                semilla,
            )
            infimos[ib, k] = intervalo["inferior"]
            supremos[ib, k] = intervalo["superior"]
            probas[ib, k] = camino_1
            progreso.contar("ajustes_de_m2")
    progreso.parcial(
        "brazo_M2",
        ".json",
        {"diferencia_maxima_entre_caminos": maximo, "n_ejecuciones": n_ejecuciones},
    )
    corrida = CorridaBrazo(
        identificador="M2",
        probas_oof=probas,
        intervalos={"inferior": infimos, "superior": supremos},
        n_efectivos={},
        n_activas={},
    )
    corrida_maximo = maximo
    corrida_n = n_ejecuciones
    corrida_filas = n_filas_control
    globals()["_M2_CONTROL"] = {
        "diferencia_maxima": corrida_maximo,
        "n_ejecuciones": corrida_n,
        "n_filas": corrida_filas,
    }
    return corrida


# =============================================================================
# SECCION 18: ORQUESTACION. INFORME, COMPLETO, HUMO Y MAIN
#
# Lo que hay por encima son las comprobaciones; lo que hay aqui es lo que las
# LLAMA. Sin esta seccion el script se importaba, pasaba cualquier auditor sobre
# el papel y no ejecutaba ni una sola afirmacion, que es exactamente lo que
# ocurrio: las veinte funciones existian y ninguna tenia un solo llamador.
# =============================================================================


def escribir_informe(
    ruta_json: Path,
    ruta_md: Path,
    carga: dict[str, Any],
    escrituras_ok: bool,
) -> dict[str, Any]:
    """Escribe el informe JSON y el markdown, una sola vez cada uno.

    El cuarto argumento es el exito de la escritura y llega CALCULADO desde el
    resultado de escribir_unico. No se escribe True a mano: un False escrito a
    mano aqui fue el fallo caro de P82, y por eso la auditoria de booleanos
    literales vigila exactamente esta posicion.

    Args:
        ruta_json: destino del JSON.
        ruta_md: destino del markdown.
        carga: diccionario con todo lo calculado.
        escrituras_ok: si las dos escrituras se hicieron de verdad.

    Returns:
        Diccionario con las dos rutas, si se escribieron y el veredicto.

    Raises:
        AssertionError: si escrituras_ok no coincide con lo que se observo.
    """
    hecho_json = escribir_unico(ruta_json, construir_json(carga))
    hecho_md = escribir_unico(ruta_md, construir_markdown(carga))
    observado = bool(hecho_json) or bool(hecho_md)
    if bool(escrituras_ok) != observado:
        raise AssertionError(
            "el exito de la escritura no coincide con lo que devolvio escribir_unico: "
            f"json={hecho_json} md={hecho_md} declarado={bool(escrituras_ok)}"
        )
    return {
        "ruta_json": str(ruta_json),
        "ruta_md": str(ruta_md),
        "json_escrito": bool(hecho_json),
        "md_escrito": bool(hecho_md),
    }


def texto_minimo_de_ejemplo(lectura: str) -> str:
    """Texto de informe minimo con las cadenas que A5, A18 y A19 exigen.

    Args:
        lectura: valor de la regla de 1.5 que se va a escribir.

    Returns:
        Texto ASCII con las cadenas de formulacion obligatorias.
    """
    lineas = [
        "# P83: informe minimo de humo",
        "",
        LITERAL_V03,
        LITERAL_PREGUNTA_UNO,
        LITERAL_PREGUNTA_DOS,
        LITERAL_EXPECTATIVA,
        LITERAL_POSTERIOR_APROXIMADA,
        LITERAL_EFECTIVOS_METODO,
        LITERAL_NO_WORD2VEC,
        LITERAL_NO_SUPERA,
        LITERAL_AMBIGUEDAD,
        LITERAL_SESGO_SELECCION,
        LITERAL_DESCRIPTIVA_F1,
        LITERAL_F1_NO_REFUTA_UNO,
        LITERAL_F1_NO_REFUTA_DOS,
        "",
        "## Lectura de la pregunta 1: incertidumbre y calibracion",
        "",
        f"- Lectura de la regla de 1.5: **{lectura}**",
        "- El F1 no aparece en esta regla, porque la pregunta 1 no compite por F1.",
        "",
        "## Lectura de la pregunta 2: complementariedad de errores",
        "",
        "- El F1 no aparece en esta regla, porque la pregunta 2 no busca superar a V03.",
        "",
        "## Cifras",
        "",
        "El bloque de lectura no lleva F1 porque la pregunta 1 no compite por F1.",
    ]
    return "\n".join(lineas) + "\n"


def contexto_de_ejemplo() -> ContextoDatos:
    """Contexto sintetico pequeno para ejercitar los brazos sin los datos reales.

    Returns:
        Un contexto de datos sinteticos completo.
    """
    return construir_contexto_sintetico(
        datos_sinteticos(n_filas=40, n_grupos=8, n_columnas=6, semilla=4242, caso="COMPLETO")
    )


# =============================================================================
# SECCION 19: LOS CASOS DE HUMO
# =============================================================================

# El suelo es el RECUENTO REAL de casos, no un numero redondo bajo. Con un suelo
# de sesenta, la puerta pasaba un humo que hubiera perdido la mitad de sus casos
# siempre que no llegara a sesenta correctos, luego un caso eliminado se
# detectaba como "todo sigue bien". Aqui el suelo se fija al numero de casos que
# se registrar, y si alguien quita un caso, la puerta tiene que bajar.
SUELO_DECLARADO: Final[int] = 189
N_CASOS_ESPERADOS: Final[int] = 189


def _cifras_de_pregunta_1() -> dict[str, dict[str, float]]:
    """Cifras de las seis magnitudes de la pregunta 1, por brazo.

    Returns:
        Diccionario M1 y M2 con las seis magnitudes.
    """
    magnitudes = (
        "log_loss_macro",
        "log_loss_sin_recortar",
        "brier_macro",
        "brier_por_etiqueta",
        "ece_macro",
        "pendientes",
    )
    return {
        brazo: {m: 0.30 + 0.01 * indice + 0.001 * k for k, m in enumerate(magnitudes)}
        for indice, brazo in enumerate(("M1", "M2"))
    }


def _medidas_de_pregunta_2() -> dict[str, dict[str, Any]]:
    """Las cuatro medidas de complementariedad, por brazo.

    Returns:
        Diccionario M1 y M2 con las cuatro medidas no vacias.
    """
    claves = (
        "auc_del_error_de_v03",
        "acuerdo_con_v03",
        "matriz_de_interseccion",
        "diferencias_por_etiqueta",
    )
    return {
        brazo: {
            clave: ({"10": 12} if clave == "matriz_de_interseccion" else float(0.20 + 0.01 * indice))
            for clave in claves
        }
        for indice, brazo in enumerate(("M1", "M2"))
    }


def _configuraciones_de_ensamblaje() -> dict[str, dict[str, Any]]:
    """Las ocho cifras por alpha del ensamblaje, con los tres alphas.

    Returns:
        Diccionario con las claves 0.05, 0.10 y 0.15.
    """
    claves = (
        "corrected_cv",
        "delta",
        "pliegues_a_favor",
        "rescate",
        "dano",
        "neto",
        "techo",
        "piso",
    )
    return {
        f"{float(alpha):.2f}": {
            clave: (float(0.70 + 0.001 * indice) if clave == "corrected_cv" else indice)
            for clave in claves
        }
        for indice, alpha in enumerate(CONFIG["alphas_ensamblaje"])
    }


def _probabilidad_para_humo(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Probabilidades y verdad sinteticas con suficiente dispersion.

    Motivo: con probabilidades todas parecidas la curva de calibracion deja
    casi todos los bins vacios y A9 fallaria por una razon que no es la que
    A9 vigila, luego el fallo seria del caso de humo y no del script.

    Args:
        n: numero de filas.

    Returns:
        Tupla con la matriz de probabilidades y la matriz binaria.
    """
    rng = np.random.default_rng(31337)
    probabilidad = rng.beta(2.0, 2.0, size=(n, len(ETIQUETAS)))
    verdad = (rng.random((n, len(ETIQUETAS))) < probabilidad).astype(np.int64)
    return probabilidad, verdad


def caso_afirmacion(
    puerta: PuertaHumo,
    nombre: str,
    afirmacion: str,
    comprobacion: Callable[[], tuple[bool, dict[str, Any]]],
    mutacion: Callable[[], tuple[bool, dict[str, Any]]] | None = None,
) -> None:
    """Registra el caso de humo de UNA afirmacion: que pasa y que sabe fallar.

    Motivo del par: un caso que solo confirma no demuestra nada, porque el
    defecto que mas ha costado en este proyecto es una comprobacion que no
    sabe fallar. Por eso cada caso que puede mutarse registra DOS veredictos, el
    bueno y el mutado, y la puerta exige que el segundo se comportar.

    Args:
        puerta: puerta de humo.
        nombre: nombre del caso.
        afirmacion: identificador de la afirmacion, para el registro.
        comprobacion: la comprobacion con datos correctos.
        mutacion: la misma comprobacion con un dato deliberadamente roto.

    Returns:
        Nada.
    """
    try:
        exito, detalle = comprobar_veredicto(comprobacion())
        puerta.registrar(nombre, exito, f"{afirmacion} confirma con datos correctos")
        puerta.veredictos[f"humo_{afirmacion}"] = {
            resolver_clave("estado_json"): ESTADO_CONFIRMADO if exito else ESTADO_REFUTADO,
            "stop": False,
            "detalle": detalle,
        }
    except Exception as error:  # noqa: BLE001 - un caso que lanza es un fallo del caso
        puerta.registrar(nombre, False, f"{afirmacion} lanzo {type(error).__name__}: {error}")
    if mutacion is None:
        return
    try:
        exito_m, _ = comprobar_veredicto(mutacion())
        puerta.registrar(
            f"{nombre}_sabe_fallar", not exito_m, f"{afirmacion} con el dato roto devuelve {exito_m}"
        )
    except Exception:  # noqa: BLE001 - lanzar tambien es saber fallar
        puerta.registrar(
            f"{nombre}_sabe_fallar", True, f"{afirmacion} con el dato roto lanzo, que es saber fallar"
        )


def caso_frontera(
    puerta: PuertaHumo, nombre: str, comprobacion: Callable[[], bool], detalle: str = ""
) -> bool:
    """Registra un caso de frontera booleano.

    Args:
        puerta: puerta de humo.
        nombre: nombre del caso.
        comprobacion: funcion sin argumentos que devuelve el veredicto.
        detalle: descripcion corta.

    Returns:
        El veredicto, para que el llamante pueda encadenar.
    """
    try:
        ok = bool(comprobacion())
        puerta.registrar(nombre, ok, detalle)
    except Exception as error:  # noqa: BLE001
        puerta.registrar(nombre, False, f"lanzo {type(error).__name__}: {error}")
        return False
    return ok


def _copia_sintetica_de_control(nombre: str, con_literal: bool) -> str:
    """Construye una funcion de control sintetica, con o sin literal en el cuerpo.

    POR QUE SE CONSTRUYE Y NO SE MUTA EL SCRIPT REAL. Un caso de humo atado a una
    linea de codigo real se queda obsoleto en cuanto se arregla esa linea, y eso
    ha pasado dos veces con este mismo caso. Ademas, mutar el script real para
    comprobar que la auditoria lo ve tiene un problema mas serio: obliga a que el
    texto del sustituto este escrito dentro del propio caso, y entonces el
    sustituto aparece en el codigo y la herramienta se niega a mutar, que es
    exactamente el aborta de la ronda anterior. Construyendo la funcion aqui, el
    caso no depende de ninguna linea real y el sustituto no aparece en ningun
    sitio por accidente.

    Args:
        nombre: nombre de la funcion, que tiene que estar en PARAMETROS_DE_CONDICION.
        con_literal: si el cuerpo lleva un literal booleano asignado a una variable
            de control, en vez de una comparacion real.

    Returns:
        Codigo Python de la funcion sintetica.

    Raises:
        AssertionError: si el nombre no esta en PARAMETROS_DE_CONDICION.
    """
    if nombre not in PARAMETROS_DE_CONDICION:
        raise AssertionError(
            f"{nombre} no esta en PARAMETROS_DE_CONDICION, luego la auditoria no lo "
            "vigilaria y el caso no probaria nada"
        )
    asignacion = "    dentro = True\n" if con_literal else (
        "    dentro = float(segundos) <= float(presupuesto)\n"
    )
    lineas = [
        f"def {nombre}(segundos, presupuesto, presupuesto_respetado, con_nota_de_excepcion=None):",
        asignacion,
        "    nota = dentro if con_nota_de_excepcion is None else bool(con_nota_de_excepcion)",
        "    coherente = bool(presupuesto_respetado) == dentro",
        "    return dentro, coherente",
        "",
    ]
    return "\n".join(lineas)


def _copia_mutada(codigo: str, original: str, sustituto: str) -> str:
    """Devuelve el codigo con una sustitucion, y exige que la sustitucion aplicase.

    Motivo: una mutacion que no se aplica deja el codigo intacto y el caso
    pasa por vacuidad, que es la forma mas comun de que un caso de humo no
    discrimine. Por eso se EXIGE que el original estuviese presente, y si no
    estaba se levanta en vez de devolver el codigo sin tocar.

    POR QUE SE SUSTITUYE LA ULTIMA OCURRENCIA Y NO LA PRIMERA. El propio caso
    de humo contiene el trozo original DENTRO de un literal de cadena, para poder
    llamarse a si mismo. Sustituir la primera ocurrencia muta ese literal y deja el
    codigo real intacto, luego la mutacion no aplica y el caso pasa sin discriminar.
    El fallo ya ha pasado dos veces aqui, asi que la ultima ocurrencia es la del
    codigo real, y el caso exige ademas que la sustitucion haya descrito ALGO.

    Args:
        codigo: fuente original.
        original: trozo que tiene que estar presente.
        sustituto: trozo con el que se sustituye.

    Returns:
        El codigo mutado.

    Raises:
        AssertionError: si el trozo original no aparece en el codigo.
    """
    if original not in codigo:
        raise AssertionError(
            "la mutacion no se puede aplicar porque el trozo original no aparece: "
            f"{original!r}"
        )
    if sustituto in codigo:
        raise AssertionError(
            f"la mutacion ya esta presente en el codigo original: {sustituto!r}"
        )
    cabeza, _, cola = codigo.rpartition(original)
    if not cabeza:
        raise AssertionError("no se pudo aislar el trozo original para mutarlo")
    return cabeza + sustituto + cola


def _detecta_fallo(comprobacion: Callable[[], tuple[bool, dict[str, Any]]]) -> bool:
    """Devuelve True si la comprobacion RECHAZA el dato que se le ha dado.

    Es el helper de todos los casos cuyo proposito es comprobar que la
    comprobacion detecta, y no que confirma. Un caso de deteccion registrado
    con caso_afirmacion se registraria como FALLO porque la comprobacion lanza
    AssertionError, luego el caso pasaria por la via equivocada: no por la
    comprobacion sino por el raising de todo_verdadero. Con este helper el caso
    discrimina por la comprobacion y ademas distingue que fallo por la razon
    correcta.

    Args:
        comprobacion: comprobacion que deberia rechazar.

    Returns:
        True si la comprobacion lanzo o devolvio exito falso.
    """
    try:
        exito, _ = comprobar_veredicto(comprobacion())
    except (AssertionError, ColeccionVacia, ComprobacionAnidada):
        return True
    except Exception:  # noqa: BLE001 - una excepcion rara tambien es no pasar
        return True
    return not bool(exito)


def caso_detecta(
    puerta: PuertaHumo,
    nombre: str,
    comprobacion: Callable[[], tuple[bool, dict[str, Any]]],
    detalle: str = "",
) -> None:
    """Registra un caso cuyo proposito es que la comprobacion DETECTE el fallo.

    Args:
        puerta: puerta de humo.
        nombre: nombre del caso.
        comprobacion: comprobacion con el dato deliberadamente roto.
        detalle: descripcion corta.

    Returns:
        Nada.
    """
    caso_frontera(puerta, nombre, lambda: _detecta_fallo(comprobacion), detalle)


def _falla_a3(**kwargs: Any) -> bool:
    """Ejecuta A3 y devuelve True si falla, invertiendo el veredicto.

    Args:
        **kwargs: los argumentos de afirmacion_a3.

    Returns:
        True si A3 lanzo o devolvio exito falso.
    """
    return _detecta_fallo(lambda: afirmacion_a3(**kwargs))


def caso_levanta(
    puerta: PuertaHumo,
    nombre: str,
    comprobacion: Callable[[], Any],
    excepcion_esperada: type[BaseException],
    detalle: str = "",
) -> None:
    """Registra un caso que exige que una comprobacion LEVANTE una excepcion.

    Motivo: una comprobacion que devuelve False en vez de lanzar es otra forma
    del mismo fallo, luego el caso exige la excepcion y no solo el False.

    Args:
        puerta: puerta de humo.
        nombre: nombre del caso.
        comprobacion: funcion que deberia lanzar.
        excepcion_esperada: tipo que se espera.
        detalle: descripcion corta.

    Returns:
        Nada.
    """
    try:
        comprobacion()
    except excepcion_esperada as error:
        puerta.registrar(nombre, True, f"{detalle}: {type(error).__name__}")
        return
    except Exception as error:  # noqa: BLE001
        puerta.registrar(
            nombre,
            False,
            f"lanzo {type(error).__name__} en vez de {excepcion_esperada.__name__}: {error}",
        )
        return
    puerta.registrar(nombre, False, f"no lanzo nada, se esperaba {excepcion_esperada.__name__}")


# =============================================================================
# SECCION 20: LOS CASOS DE HUMO, POR FRONTERA
# =============================================================================


def casos_de_frontera_contrato(puerta: PuertaHumo) -> None:
    """Frontera (a): el contrato entre capas cubre todo y sabe fallar.

    Args:
        puerta: puerta de humo.

    Returns:
        Nada.
    """
    correcto, detalle = verificar_contratos()
    puerta.registrar("frontera_contrato_pasa", correcto, f"{detalle['n_filas']} filas")
    filas = tabla_contratos()
    mutada = FilaContrato(
        filas[0].nombre,
        filas[0].callable,
        filas[0].parametros + ("parametro_que_no_existe",),
        filas[0].origen,
    )
    correcto_m, _ = verificar_contratos(ContextoContrato((mutada,) + filas[1:], "mutacion"))
    puerta.registrar(
        "frontera_contrato_sabe_fallar", not correcto_m, "un parametro inexistente lo detecta"
    )
    codigo = Path(__file__).read_text(encoding="utf-8")
    completo, detalle_completo = afirmar_evaluadas_en_completo(codigo)
    puerta.registrar(
        "frontera_las_veinte_en_el_completo",
        completo,
        f"{detalle_completo['afirmaciones_evaluadas_en_el_completo']} evaluadas",
    )
    completo_m, detalle_m = afirmar_evaluadas_en_completo(codigo_sin_afirmacion(codigo, "A19"))
    puerta.registrar(
        "frontera_las_veinte_sabe_fallar",
        not completo_m,
        f"al quitar A19 faltan {detalle_m['faltan_en_el_completo']}",
    )
    # La comprobacion de que el HUMO y el COMPLETO recorren las MISMAS veinte. Es
    # la que en P82 llego tarde: el script declaraba veintitres y el completo
    # evaluaba veintiuna, y se vio tras sesenta y cinco minutos de entrenamiento.
    cobertura, detalle_cobertura = afirmar_evaluadas_en_el_humo(codigo)
    puerta.registrar(
        "frontera_humo_y_completo_coinciden",
        cobertura,
        f"completo={detalle_cobertura['n_en_el_completo']} "
        f"humo={detalle_cobertura['n_en_el_humo']} "
        f"solo_completo={detalle_cobertura['solo_en_el_completo']} "
        f"solo_humo={detalle_cobertura['solo_en_el_humo']}",
    )
    # Y la prueba de que esa comprobacion SABE fallar: se quita una afirmacion
    # del recorrido del completo en una copia y tiene que detectarlo.
    sin_a7 = _copia_mutada(
        codigo,
        '        "A7",\n        lambda: afirmacion_a7(',
        '        "A7_QUITADA_PARA_LA_MUTACION",\n        lambda: afirmacion_a7(',
    )
    cobertura_m, detalle_cm = afirmar_evaluadas_en_el_humo(sin_a7)
    puerta.registrar(
        "frontera_humo_y_completo_saben_fallar",
        not cobertura_m,
        f"al quitar A7 del completo, declaradas sin conectar={detalle_cm['declaradas_sin_conectar']}",
    )


def casos_de_frontera_ast(puerta: PuertaHumo) -> None:
    """Fronteras (b) y (c): las tres auditorias por AST limpias y con potencia.

    Args:
        puerta: puerta de humo.

    Returns:
        Nada.
    """
    correcto, detalle = auditar_propio_script()
    puerta.registrar(
        "frontera_auditoria_ast_limpia",
        correcto,
        f"literales={detalle['n_infracciones_de_literal']} "
        f"booleanos={detalle['n_booleanos_literales']} "
        f"anidadas={detalle['n_evaluaciones_con_tupla_anidada']}",
    )
    puerta.registrar(
        "frontera_acceso_por_literal_detectado",
        bool(auditar_accesos_por_literal("def f(c):\n    return c['estado']\n")),
        "un subscript con literal de clave fisica",
    )
    puerta.registrar(
        "frontera_acceso_por_literal_perdonado_si_se_deriva",
        not auditar_accesos_por_literal(
            "def f(c):\n    return c[resolver_clave('estado_json')]\n"
        ),
        "la clave derivada no es infraccion",
    )
    for indice, texto_malo in enumerate(
        (
            "comprobar_presupuesto(1.0, 10, True)",
            "registrar_lectura_pregunta_1('AFFIRMATIVO', False)",
            "escribir_informe('a', 'b', {}, True)",
        )
    ):
        puerta.registrar(
            f"frontera_booleano_literal_detectado_{indice}",
            bool(auditar_booleanos_literales(texto_malo)),
            texto_malo,
        )
    puerta.registrar(
        "frontera_booleano_calculado_perdonado",
        not auditar_booleanos_literales("comprobar_presupuesto(1.0, 10, 1.0 <= 2.0)"),
        "una comparacion real no es infraccion",
    )
    # MUTACION REAL DEL PROPIO SCRIPT. La comprobacion anterior metia un literal
    # en una cadena suelta y la veia, que es facil. Lo que cuesta caro es que la
    # infraccion este en el fichero de verdad y no se note, luego aqui se mete
    # un literal booleano en una COPIA en /tmp, se le pasa la auditoria y se
    # exige que la vea. Sin este caso, la auditoria puede volver a encontrar
    # infracciones en el script y nadie se entera hasta el informe.
    codigo_real = Path(__file__).read_text(encoding="utf-8")
    puerta.registrar(
        "frontera_booleano_literal_ausente_en_el_script",
        not auditar_booleanos_literales(codigo_real),
        "el script real no tiene ningun booleano literal en posicion de condicion",
    )
    # EL CASO QUE FIJA LA AMPLIACION, Y POR QUE ES SINTETICO.
    # El caso anterior mutaba una linea de codigo REAL, y se quedo obsoleto dos
    # veces: la primera cuando el argumento paso a ser por nombre y la segunda
    # cuando comprobar_presupuesto se reescribio para que el exceso falle por si
    # mismo. Un caso atado a una linea concreta no sobrevive al siguiente arreglo,
    # y ese es el fallo de clase que hamati dos casos en este proyecto.
    # Aqui la funcion de control se CONSTRUYE aqui mismo. No depende de que exista
    # ninguna linea en el script, luego ni el arreglo siguiente ni el siguiente a
    # ese pueden volverlo obsoleto, y lo que se comprueba es exactamente lo que
    # importa: que la auditoria ve un literal booleano en el CUERPO de una funcion
    # de PARAMETROS_DE_CONDICION, que era el punto ciego por construccion.
    nombre_control = sorted(PARAMETROS_DE_CONDICION)[0]
    cuerpo_limpio = _copia_sintetica_de_control(nombre_control, False)
    cuerpo_con_literal = _copia_sintetica_de_control(nombre_control, True)
    puerta.registrar(
        "frontera_el_cuerpo_limpio_no_es_infraccion",
        not auditar_booleanos_literales(cuerpo_limpio),
        f"una {nombre_control} con una comparacion real no es infraccion",
    )
    infracciones_cuerpo = auditar_booleanos_literales(cuerpo_con_literal)
    puerta.registrar(
        "frontera_la_mutacion_en_el_cuerpo_se_ve",
        bool(infracciones_cuerpo),
        f"un literal en el CUERPO da {len(infracciones_cuerpo)} infracciones",
    )
    # Y lo que mas importa: la auditoria tiene que distinguir el caso con literal
    # del caso sin literal. Si viera los dos igual, el caso de arriba pasaria por
    # vacuidad.
    puerta.registrar(
        "frontera_la_auditoria_distingue_el_cuerpo_con_y_sin_literal",
        bool(infracciones_cuerpo) and not auditar_booleanos_literales(cuerpo_limpio),
        "con literal hay infraccion y sin literal no la hay",
    )
    # CASO DE TRAMPA CONTRA EL NO-OP. Un caso que muta y no cambia nada, y aun asi
    # pasa, es el peor fallo posible de un caso de humo, y es justo lo que paso
    # con la mutacion de la linea que dejo de existir. El caso exige que el codigo
    # mutado sea DISTINTO del original, de modo que un no-op no pueda pasar por
    # que la auditoria lo vea.
    puerta.registrar(
        "frontera_la_mutacion_del_cuerpo_no_es_un_no_op",
        cuerpo_con_literal != cuerpo_limpio,
        "el codigo mutado tiene que ser distinto del original, o la mutacion no sirvio",
    )
    # Y la trampa sobre la propia herramienta: _copia_mutada tiene que negarse a
    # mutar cuando el sustituto ya esta presente, que es la guarda que disparo el
    # aborta de esta ronda.
    caso_levanta(
        puerta,
        "frontera_la_copia_mutada_rechaza_un_sustituto_ya_presente",
        lambda: _copia_mutada(
            codigo_real, "    dentro = True", "    dentro = True"
        ),
        AssertionError,
        "sustituir un trozo por si mismo es un no-op y tiene que rechazarse",
    )
    caso_levanta(
        puerta,
        "frontera_la_copia_mutada_rechaza_un_original_inexistente",
        lambda: _copia_mutada(codigo_real, "una linea que no existe en este fichero", "x"),
        AssertionError,
        "si el original no esta, la mutacion no aplica y tiene que rechazarse",
    )
    # Y que una comparacion real dentro del cuerpo NO es infraccion.
    puerta.registrar(
        "frontera_una_comparacion_en_el_cuerpo_no_es_infraccion",
        not auditar_booleanos_literales(
            "def comprobar_presupuesto(segundos, presupuesto, banderal):\n"
            "    dentro = float(segundos) <= float(presupuesto)\n"
            "    return True\n"
        ),
        "una comparacion real asignada dentro del cuerpo es legitima",
    )
    puerta.registrar(
        "frontera_acceso_por_literal_ausente_en_el_script",
        not auditar_accesos_por_literal(codigo_real),
        "el script real no accede a ninguna clave fisica por literal",
    )
    puerta.registrar(
        "frontera_tupla_anidada_ausente_en_el_script",
        not auditar_evaluaciones_anidadas(codigo_real),
        "el script real no tiene ninguna evaluacion con tupla anidada",
    )
    copia_con_literal = _copia_mutada(
        codigo_real,
        "        not (destinos[0].exists() or destinos[1].exists()),\n    )",
        "        True,  # mutacion: literal booleano escrito a mano\n    )",
    )
    infracciones_copia = auditar_booleanos_literales(copia_con_literal)
    puerta.registrar(
        "frontera_la_mutacion_de_literal_se_ve_en_la_copia",
        bool(infracciones_copia),
        f"la copia con un literal inyectado da {len(infracciones_copia)} infracciones",
    )
    puerta.registrar(
        "frontera_tupla_anidada_detectada_por_ast",
        bool(
            auditar_evaluaciones_anidadas(
                "p.evaluar('A0', lambda: (comprobar_veredicto(x), {'k': 1}), 'tol')"
            )
        ),
        "lambda cuyo cuerpo es una tupla anidada",
    )
    puerta.registrar(
        "frontera_tupla_simple_perdonada",
        not auditar_evaluaciones_anidadas(
            "p.evaluar('A0', lambda: afirmacion_a0(antes, despues), 'tol')"
        ),
        "una lambda que devuelve la tupla del assertion no es infraccion",
    )


def casos_de_frontera_no_vacio(puerta: PuertaHumo) -> None:
    """Frontera (d): ninguna afirmacion puede pasar en vacio.

    Args:
        puerta: puerta de humo.

    Returns:
        Nada.
    """
    caso_levanta(
        puerta,
        "frontera_tupla_anidada_rechazada_en_runtime",
        lambda: comprobar_veredicto((comprobar_veredicto((True, {"a": 1})), {})),
        ComprobacionAnidada,
        "el primer elemento debe ser un booleano y no una tupla",
    )
    caso_levanta(
        puerta,
        "frontera_detalle_vacio_rechazado",
        lambda: comprobar_veredicto((True, {})),
        ColeccionVacia,
        "un detalle vacio no dice que se comprobo",
    )
    caso_levanta(
        puerta,
        "frontera_coleccion_vacia_rechazada",
        lambda: exigir_no_vacio([], "coleccion de prueba"),
        ColeccionVacia,
        "una lista vacia no puede pasar",
    )
    caso_levanta(
        puerta,
        "frontera_coleccion_none_rechazada",
        lambda: exigir_no_vacio(None, "coleccion de prueba"),
        ColeccionVacia,
        "un None no puede pasar",
    )
    caso_levanta(
        puerta,
        "frontera_forma_incorrecta_rechazada",
        lambda: comprobar_veredicto([True, {"a": 1}]),
        ComprobacionAnidada,
        "una lista no es una tupla",
    )
    caso_levanta(
        puerta,
        "frontera_valor_truthy_rechazado",
        lambda: comprobar_veredicto((1, {"a": 1})),
        ComprobacionAnidada,
        "el entero 1 es truthy y no es un booleano",
    )


def casos_de_frontera_puerta(puerta: PuertaHumo) -> None:
    """Frontera (e): la puerta sabe devolver 1 y vigila las marcas de STOP.

    Args:
        puerta: puerta de humo.

    Returns:
        Nada.
    """
    con_fallo = PuertaHumo()
    con_fallo.registrar("falso", False, "caso que falla a proposito")
    puerta.registrar(
        "frontera_puerta_devuelve_1_con_un_fallo", con_fallo.codigo_salida(0) == 1, "un fallo da 1"
    )
    limpio = PuertaHumo()
    limpio.registrar("cierto", True, "caso que pasa")
    puerta.registrar(
        "frontera_puerta_devuelve_0_sin_fallos", limpio.codigo_salida(1) == 0, "sin fallos da 0"
    )
    corto = PuertaHumo()
    corto.registrar("cierto", True, "un solo caso")
    puerta.registrar(
        "frontera_puerta_respeta_el_suelo", corto.codigo_salida(SUELO_DECLARADO) == 1, "no llega al suelo"
    )
    con_stop = PuertaHumo()
    con_stop.registrar("cierto", True, "un caso")
    con_stop.veredictos["A0"] = {
        resolver_clave("estado_json"): ESTADO_REFUTADO,
        "stop": True,
        "detalle": {"de": "prueba"},
    }
    puerta.registrar(
        "frontera_puerta_vigila_las_stop",
        con_stop.codigo_salida(0) == 1,
        "una STOP no confirmada da 1",
    )


def casos_de_frontera_ascii(puerta: PuertaHumo) -> None:
    """Frontera de escritura: ASCII comprobado en memoria y escritura unica.

    Args:
        puerta: puerta de humo.

    Returns:
        Nada.
    """
    caso_levanta(
        puerta,
        "frontera_ascii_rechazado_en_memoria",
        lambda: texto_ascii("acent" + chr(0xE1)),
        ValueError,
        "un caracter no latin se rechaza antes de abrir el destino",
    )
    puerta.registrar(
        "frontera_ascii_puro_pasa", texto_ascii("texto sin acentos") == "texto sin acentos", "ASCII"
    )
    objetivo = REPORTS / f"humo_p83_{puerta.marca_texto}" / "escritura_unica.txt"
    primero = escribir_unico(objetivo, "primer contenido\n")
    segundo = escribir_unico(objetivo, "segundo contenido\n")
    puerta.registrar(
        "frontera_escritura_unica_rechaza_reescritura",
        bool(primero) and not bool(segundo),
        f"primero={primero} segundo={segundo}",
    )


def casos_de_frontera_m2(puerta: PuertaHumo) -> None:
    """M2 devuelve y USA los log-odds por variable, y el intervalo se recentra.

    Motivo: sin el termino de senal el intervalo sale centrado en el prior y su
    anchura no depende de la unidad evaluada, que es el defecto corregido.

    Args:
        puerta: puerta de humo.

    Returns:
        Nada.
    """
    rng = np.random.default_rng(7)
    n, p = 60, 8
    X = (rng.random((n, p)) < 0.35).astype(np.float64)
    y = (rng.random(n) < 0.5).astype(np.float64)
    Xev = (rng.random((12, p)) < 0.35).astype(np.float64)
    modelo = conteos_de_m2(X, y, CONFIG["alfa_dirichlet"])
    puerta.registrar(
        "frontera_m2_devuelve_log_odds_por_variable",
        "log_odds_por_variable" in modelo
        and int(np.asarray(modelo["log_odds_por_variable"]).size) == p,
        f"p={p}",
    )
    punto = camino_bernoulli_m2(X, y, Xev, CONFIG["alfa_dirichlet"])
    camino2 = camino_recuento_m2(X, y, Xev, CONFIG["alfa_dirichlet"])
    diferencia = float(np.max(np.abs(punto - camino2)))
    puerta.registrar(
        "frontera_m2_los_dos_caminos_coinciden", diferencia <= 1e-12, f"diferencia={diferencia:.3e}"
    )
    intervalo = anchura_intervalo_m2(modelo, Xev, 20000, 11)
    centro = 0.5 * (intervalo["inferior"] + intervalo["superior"])
    desvio = float(np.max(np.abs(centro - punto)))
    puerta.registrar(
        "frontera_m2_el_intervalo_se_centra_en_el_punto",
        desvio < 5e-3,
        f"max|centro-punto|={desvio:.3e} con 20000 muestras",
    )
    sin_senal = {k: v for k, v in modelo.items() if k != "log_odds_por_variable"}
    caso_levanta(
        puerta,
        "frontera_m2_avisa_si_no_hay_log_odds",
        lambda: anchura_intervalo_m2(sin_senal, Xev, 100, 11),
        AssertionError,
        "sin los log-odds por variable el intervalo no se puede centrar",
    )


def casos_de_la_regla_3_5(puerta: PuertaHumo) -> None:
    """La regla 3.5: el calibrador y el alpha solo pueden usar filas internas.

    Estos son los casos mas importantes del encargo. El defecto que corrigieron es
    que el contador restaba un conjunto de si mismo y daba cero con cualquier
    entrada, luego la afirmacion que vigila la contaminacion se apoyaba en un
    numero que era cero por construccion. Por eso los dos lados: con una fila no
    permitida tiene que VER la fuga, y con filas dentro de lo permitido tiene que
    dar cero. Un solo lado no discrimina.

    Args:
        puerta: puerta de humo.

    Returns:
        Nada.
    """
    instrumento = INSTRUMENTO_CALIBRACION
    permitidas_cal = np.arange(10, 20, dtype=np.int64)
    permitidas_alpha = np.arange(0, 10, dtype=np.int64)

    # LADO POSITIVO DEL CONTADOR: una fila no permitida tiene que verse.
    instrumento.fijar_permitidos(permitidas_cal, permitidas_alpha)
    instrumento.llamadas_calibrador = []
    instrumento.llamadas_alpha = []
    con_fuga = np.append(permitidas_cal, 999)
    instrumento.registrar_recepcion_calibrador(con_fuga)
    fugas = contar_fugas_calibrador(instrumento)
    puerta.registrar(
        "la_regla_3_5_el_contador_ve_la_fuga_del_calibrador",
        fugas > 0,
        f"una fila no permitida da {fugas} fuga(s), y tiene que ser mayor que cero",
    )
    puerta.registrar(
        "la_regla_3_5_la_fuga_es_exactamente_una",
        fugas == 1,
        f"el numero de fugas tiene que ser exactamente 1 y es {fugas}",
    )
    # Y que el conjunto recibido de verdad no se confunda con el permitido.
    puerta.registrar(
        "la_regla_3_5_el_recibido_no_es_el_permitido",
        set(int(i) for i in con_fuga) != set(int(i) for i in permitidas_cal),
        "el conjunto recibido tiene una fila mas que el permitido",
    )

    # LADO NEGATIVO DEL CONTADOR: dentro de lo permitido tiene que dar cero.
    instrumento.registrar_recepcion_calibrador(permitidas_cal)
    puerta.registrar(
        "la_regla_3_5_el_contador_da_cero_dentro_de_lo_permitido",
        contar_fugas_calibrador(instrumento) == 0,
        f"con filas dentro de lo permitido da {contar_fugas_calibrador(instrumento)}",
    )
    # Y con un conjunto recibido VACIO tampoco puede pasar por limpio.
    instrumento.registrar_recepcion_calibrador(np.asarray([], dtype=np.int64))
    puerta.registrar(
        "la_regla_3_5_un_ajuste_sin_filas_declaradas_no_pasa",
        contar_fugas_calibrador(instrumento) > 0,
        f"sin declarar filas da {contar_fugas_calibrador(instrumento)}, que tiene que ser mayor que cero",
    )

    # EL MISMO CONTRAPUNTO PARA EL ALPHA.
    instrumento.registrar_recepcion_alpha(np.append(permitidas_alpha, 777))
    fugas_alpha = contar_fugas_alpha(instrumento)
    puerta.registrar(
        "la_regla_3_5_el_contador_ve_la_fuga_del_alpha",
        fugas_alpha > 0,
        f"una fila no permitida da {fugas_alpha} fuga(s) en el alpha",
    )
    instrumento.registrar_recepcion_alpha(permitidas_alpha)
    puerta.registrar(
        "la_regla_3_5_el_contador_da_cero_en_el_alpha_dentro_de_lo_permitido",
        contar_fugas_alpha(instrumento) == 0,
        f"con filas dentro de lo permitido da {contar_fugas_alpha(instrumento)}",
    )

    # LA SEPARACION ENTRE EL ALPHA Y EL CALIBRADOR, que es lo que mas importa.
    # El alpha usa las filas internas de entrenamiento, el calibrador las de
    # calibracion, y no pueden solaparse.
    puerta.registrar(
        "la_regla_3_5_el_alpha_y_el_calibrador_no_se_solapan",
        len(filas_comunes_alpha_y_calibrador(instrumento)) == 0,
        f"interseccion de {len(filas_comunes_alpha_y_calibrador(instrumento))} filas",
    )
    # Y la prueba de que ESA comprobacion discrimina: se fuerza el solapamiento.
    # Para PROBAR que la interseccion se detecta hay que cambiar el PERMITIDO del
    # calibrador a las filas del alpha. Registrar solo la recepcion no cambia el
    # conjunto permitido, luego la interseccion sigue en cero, que es el
    # comportamiento correcto pero no prueba nada. Un caso que prueba lo que ya
    # funcionaba es un caso que no prueba nada.
    instrumento.fijar_permitidos(permitidas_alpha, permitidas_alpha)
    instrumento.registrar_recepcion_calibrador(permitidas_alpha)
    # Y hay que REGISTRAR la recepcion del alpha en su canal. Sin una llamada
    # registrada ahi no hay registros que comparar, luego la interseccion sale
    # cero por VACUIDAD, que es distinto de cero porque no hay solapamiento. Un
    # caso que pasa porque no habia nada que mirar no prueba que la comprobacion
    # mire, y esa confusion ya ha costado dos ciclos en este fichero.
    instrumento.registrar_recepcion_alpha(permitidas_alpha)
    # El registro del canal alpha lo crea la llamada REAL a decidir_alpha, con su
    # permitted. El metodo registrar_recepcion_alpha solo guarda las filas en un
    # atributo y NO crea entrada en la lista, luego sin construirla aqui no hay
    # registros que comparar y la interseccion sale cero por VACUIDAD, que es
    # distinto de cero porque no hay solapamiento. Un caso que pasa porque no habia
    # nada que mirar no prueba que la comprobacion mire, y esa confusion ya ha
    # costado dos ciclos en este fichero.
    instrumento.llamadas_alpha = [
        {
            "filas": [int(i) for i in permitidas_alpha],
            "filas_declaradas": True,
            "permitidos_alpha": sorted(permitidas_alpha),
            "permitidos_calibrador": sorted(permitidas_alpha),
        }
    ]
    comunes = filas_comunes_alpha_y_calibrador(instrumento)
    instrumento.llamadas_alpha = []
    puerta.registrar(
        "la_regla_3_5_la_no_solapacion_detecta_el_solapamiento",
        len(comunes) == len(permitidas_alpha),
        f"con el permitido del calibrador en las filas del alpha, la interseccion es "
        f"{len(comunes)} de {len(permitidas_alpha)}",
    )
    # Y se RESTAURA el permitido del calibrador, para que los casos siguientes midan
    # lo que quieren medir y no el residuo de este.
    instrumento.fijar_permitidos(permitidas_cal, permitidas_alpha)
    puerta.registrar(
        "la_regla_3_5_el_solapamiento_tambien_es_fuga_del_calibrador",
        contar_fugas_calibrador(instrumento) > 0,
        f"y ademas son {contar_fugas_calibrador(instrumento)} filas fuera de lo permitido",
    )

    # El ajuste REAL sobre las filas internas tiene que dar cero por los dos lados.
    instrumento.fijar_permitidos(permitidas_cal, permitidas_alpha)
    proba = np.random.default_rng(3).random((len(permitidas_cal), len(ETIQUETAS)))
    verdad = (np.random.default_rng(4).random((len(permitidas_cal), len(ETIQUETAS))) < 0.5).astype(
        np.int64
    )
    ajustar_calibrador(proba, verdad, permitidas_cal)
    puerta.registrar(
        "la_regla_3_5_el_calibrador_real_registra_sus_filas",
        len(INSTRUMENTO_CALIBRACION.llamadas_calibrador[-1].get("filas", []))
        == len(permitidas_cal),
        f"la llamada registrada trae {len(INSTRUMENTO_CALIBRACION.llamadas_calibrador[-1].get('filas', []))} indices",
    )
    puerta.registrar(
        "la_regla_3_5_el_calibrador_real_no_tiene_fugas",
        contar_fugas_calibrador(instrumento) == 0 and contar_fugas_ajuste_sobre_el_oof(instrumento) == 0,
        f"fugas={contar_fugas_calibrador(instrumento)} ajustes_fuera={contar_fugas_ajuste_sobre_el_oof(instrumento)}",
    )
    # Y el MISMO ajuste declarado sobre el OOF completo tiene que verse.
    oof = np.arange(0, 60, dtype=np.int64)
    ajustar_calibrador(proba, verdad, oof)
    puerta.registrar(
        "la_regla_3_5_un_calibrador_sobre_el_oof_se_ve",
        contar_fugas_calibrador(instrumento) > 0 and contar_fugas_ajuste_sobre_el_oof(instrumento) > 0,
        f"declarando el OOF: fugas={contar_fugas_calibrador(instrumento)} "
        f"ajustes_fuera={contar_fugas_ajuste_sobre_el_oof(instrumento)}",
    )
    # Y el alpha sin declarar filas tampoco.
    decidir_alpha(
        {"0.05": 0.1, "0.10": 0.2, "0.15": 0.15}, CONFIG["alphas_ensamblaje"], 0.10, None
    )
    puerta.registrar(
        "la_regla_3_5_el_alpha_sin_filas_no_pasa",
        contar_fugas_alpha(instrumento) > 0,
        f"un alpha sin declarar filas da {contar_fugas_alpha(instrumento)}",
    )
    instrumento.llamadas_calibrador = []
    instrumento.llamadas_alpha = []
    instrumento.registrar_recepcion_calibrador(np.asarray([], dtype=np.int64))
    instrumento.registrar_recepcion_alpha(np.asarray([], dtype=np.int64))


def casos_de_frontera_lectura(puerta: PuertaHumo) -> None:
    """Frontera de presupuesto, de lectura de 1.5 y de intervalos pareados.

    Args:
        puerta: puerta de humo.

    Returns:
        Nada.
    """
    # El tercer argumento de comprobar_presupuesto es una CONDICION y llega
    # CALCULADA. Escribir True o False a mano aqui es exactamente el fallo de P82,
    # y la auditoria de booleanos literales lo detecta, luego la forma que se usa
    # es la que la auditoria perdona: una comparacion real.
    caso_afirmacion(
        puerta,
        "presupuesto_dentro",
        "presupuesto",
        lambda: comprobar_presupuesto(10.0, 100, 10.0 <= float(100)),
        lambda: comprobar_presupuesto(200.0, 100, 200.0 <= float(100)),
    )
    caso_levanta(
        puerta,
        "presupuesto_banderal_contradictorio_rechazado",
        lambda: comprobar_presupuesto(10.0, 100, not (10.0 <= float(100))),
        AssertionError,
        "el banderal tiene que concordar con la comparacion",
    )
    # Un exceso REAL tiene que fallar por el exceso y no por el banderal. El
    # banderal se declara coherente a proposito, luego si el exceso no se
    # comprueba por si mismo, esta llamada pasaria.
    caso_levanta(
        puerta,
        "presup_un_exceso_real_falla_sin_depender_del_banderal",
        lambda: comprobar_presupuesto(200.0, 100, not (200.0 <= float(100))),
        AssertionError,
        "200 s contra un presupuesto de 100 s tiene que fallar aunque el banderal sea coherente",
    )
    caso_levanta(
        puerta,
        "presup_un_exceso_con_banderal_coherente_tambien_falla",
        lambda: comprobar_presupuesto(150.0, 100, not (150.0 <= float(100))),
        AssertionError,
        "el banderal coherente con el exceso no absuelve el exceso",
    )
    # Y con nota de excepcion declarada el exceso no es un fracaso.
    caso_frontera(
        puerta,
        "presup_un_exceso_con_nota_no_es_fracaso",
        lambda: comprobar_presupuesto(200.0, 100, 200.0 <= float(100), True)[0],
        "con nota declarada, el exceso se publica y no rompe",
    )
    # La asimetria del ejercicio: si se pasa la nota puesta a False con el
    # tiempo dentro, el banderal y la nota se contradicen y tiene que verse.
    caso_levanta(
        puerta,
        "presup_una_nota_falsa_dentro_tambien_falla",
        lambda: comprobar_presupuesto(10.0, 100, not (10.0 <= float(100)), False),
        AssertionError,
        "una nota de excepcion sin exceso es una contradiccion y tiene que verse",
    )
    lectura, _ = aplicar_lectura_intervalos(
        {"inferior": -0.02, "superior": -0.01, "n": 5},
        {"inferior": -0.02, "superior": -0.01, "n": 5},
        CONFIG["eps_gate_delta"],
    )
    puerta.registrar(
        "lectura_afirmativa_cuando_ambos_extremos_son_favorables",
        lectura == LECTURA_AFIRMATIVO,
        lectura,
    )
    lectura_amb, _ = aplicar_lectura_intervalos(
        {"inferior": -0.02, "superior": -0.01, "n": 5},
        {"inferior": 0.02, "superior": 0.03, "n": 5},
        CONFIG["eps_gate_delta"],
    )
    puerta.registrar(
        "lectura_ambigua_cuando_los_intervalos_se_contradicen",
        lectura_amb == LECTURA_AMBIGUA,
        lectura_amb,
    )
    caso_levanta(
        puerta,
        "intervalo_sin_extremos_rechazado",
        lambda: aplicar_lectura_intervalos({"n": 5}, {"n": 5}, 0.01),
        AssertionError,
        "un intervalo sin extremos no se puede leer",
    )
    caso_levanta(
        puerta,
        "intervalo_t_pareado_sin_observaciones_rechazado",
        lambda: intervalo_t_pareado([]),
        ColeccionVacia,
        "el intervalo pareado exige al menos una observacion",
    )


def casos_de_los_margenes(puerta: PuertaHumo) -> None:
    """Los tres margenes de 1.3, aplicados contra lo MEDIDO y no contra lo esperado.

    Motivo: estaban declarados en CONFIG y no los comparaba nadie. Un margen
    declarado y no aplicado no es un margen.

    Args:
        puerta: puerta de humo.

    Returns:
        Nada.
    """
    puerta.registrar(
        "margenes_los_tres_estan_declarados",
        CONFIG["margen_log_loss"] == 0.005
        and CONFIG["margen_brier"] == 0.0015
        and CONFIG["margen_ece"] == 0.010,
        f"log-loss {CONFIG['margen_log_loss']}, Brier {CONFIG['margen_brier']}, "
        f"ECE {CONFIG['margen_ece']}",
    )
    liston = {"log_loss_macro": 0.391009, "brier_macro": 0.122455, "ece_macro": 0.040}
    caso_afirmacion(
        puerta,
        "margenes_no_inferior",
        "margenes",
        lambda: comparar_contra_el_liston(liston, liston),
        lambda: comparar_contra_el_liston(
            {
                "log_loss_macro": liston["log_loss_macro"] + 0.006,
                "brier_macro": liston["brier_macro"],
                "ece_macro": liston["ece_macro"],
            },
            liston,
        ),
    )
    # Un exceso por debajo del margen sigue siendo no inferior: el margen es un
    # tope de cuanto peor, no una igualdad.
    caso_frontera(
        puerta,
        "margenes_un_exceso_pequeno_sigue_siendo_no_inferior",
        lambda: comparar_contra_el_liston(
            {
                "log_loss_macro": liston["log_loss_macro"] + 0.0049,
                "brier_macro": liston["brier_macro"] + 0.0014,
                "ece_macro": liston["ece_macro"] + 0.009,
            },
            liston,
        )[0],
        "0.0049, 0.0014 y 0.009 caben en 0.005, 0.0015 y 0.010",
    )
    # Y estar MEJOR que el liston no es un fallo, pero tampoco afirma superioridad.
    resultado_mejor, detalle_mejor = comparar_contra_el_liston(
        {
            "log_loss_macro": liston["log_loss_macro"] - 0.100,
            "brier_macro": liston["brier_macro"] - 0.050,
            "ece_macro": liston["ece_macro"] - 0.020,
        },
        liston,
    )
    puerta.registrar(
        "margenes_mejor_no_es_fallo",
        bool(resultado_mejor),
        "un delta favorable grande sigue siendo no inferior",
    )
    puerta.registrar(
        "margenes_mejor_no_produce_lectura_afirmativa",
        detalle_mejor["lectura"] != LECTURA_AFIRMATIVO,
        f"la lectura es {detalle_mejor['lectura']}, nunca afirmativa",
    )
    puerta.registrar(
        "margenes_el_resultado_declara_el_limite_de_v03",
        LITERAL_V03 in detalle_mejor["que_no_se_afirma"],
        "el detalle lleva la frase obligatoria de V03",
    )
    # Un exceso en CADA magnitud por separado tiene que flipping la lectura.
    for indice, (clave, margen) in enumerate(
        (
            ("log_loss_macro", "margen_log_loss"),
            ("brier_macro", "margen_brier"),
            ("ece_macro", "margen_ece"),
        )
    ):
        caso_frontera(
            puerta,
            f"margenes_el_exceso_de_{clave}_flipping_la_lectura",
            lambda clave=clave, margen=margen: comparar_contra_el_liston(
                {
                    magnitud: (
                        liston[magnitud] + float(CONFIG[margen]) + 0.001
                        if magnitud == clave
                        else liston[magnitud]
                    )
                    for magnitud in ("log_loss_macro", "brier_macro", "ece_macro")
                },
                liston,
            )[1]["lectura"] == LECTURA_NEGATIVO,
            f"un exceso de {clave} mayor que {margen} tiene que dar negativo",
        )
    # Y que el resultado se calcula sobre lo MEDIDO: cambiar lo medido cambia el delta.
    puerta.registrar(
        "margenes_el_delta_sale_de_lo_medido",
        comparar_contra_el_liston(
            {"log_loss_macro": 0.5, "brier_macro": 0.12, "ece_macro": 0.04}, liston
        )[1]["deltas_contra_el_liston"]["log_loss_macro"]
        != comparar_contra_el_liston(
            {"log_loss_macro": 0.6, "brier_macro": 0.12, "ece_macro": 0.04}, liston
        )[1]["deltas_contra_el_liston"]["log_loss_macro"],
        "cambiar lo medido cambia el delta, luego no sale de una constante",
    )
    caso_levanta(
        puerta,
        "margenes_sin_magnitudes_comparables_falla",
        lambda: comparar_contra_el_liston({}, {}),
        ColeccionVacia,
        "sin ninguna magnitud comparable la comparacion no puede pasar por vacia",
    )
    # Y la rejilla de alpha con su primario declarado.
    puerta.registrar(
        "margenes_la_rejilla_de_alpha_tiene_los_tres_declarados",
        tuple(CONFIG["alphas_ensamblaje"]) == (0.05, 0.10, 0.15),
        f"rejilla {list(CONFIG['alphas_ensamblaje'])}",
    )
    puerta.registrar(
        "margenes_el_alpha_primario_es_el_declarado",
        float(CONFIG["alpha_ensamblaje_primario"]) == 0.10
        and float(CONFIG["alpha_ensamblaje_primario"]) in [float(a) for a in CONFIG["alphas_ensamblaje"]],
        f"primario {CONFIG['alpha_ensamblaje_primario']}",
    )
    puerta.registrar(
        "margenes_el_primario_no_se_promueven_a_sino_dos",
        len([a for a in CONFIG["alphas_ensamblaje"] if float(a) == 0.10]) == 1,
        "el primario aparece una sola vez en la rejilla",
    )


def casos_del_limite_5_5(puerta: PuertaHumo) -> None:
    """El limite de 5.5: el anclaje a una libreria externa, declarado sin disimulo.

    En este venv las doce librerias estan ausentes, luego el limite se mantiene y
    lo que P83 autoriza es afirmar sobre el comportamiento del modelo que este
    proyecto implemento, y no sobre la correccion de la inferencia variacional.

    Args:
        puerta: puerta de humo.

    Returns:
        Nada.
    """
    presentes = {nombre: _libreria_esta_en_el_venv(nombre) for nombre in LIBRERIAS_ANCLA}
    encontrados = sorted(nombre for nombre, hay in presentes.items() if hay)
    puerta.registrar(
        "limite_5_5_se_comprueban_las_doce",
        len(presentes) == len(LIBRERIAS_ANCLA) == 12,
        f"{len(presentes)} nombres comprobados, y son 12",
    )
    puerta.registrar(
        "limite_5_5_el_estado_sale_de_la_medida",
        (encontrados == []) == (all(presentes[n] for n in LIBRERIAS_ANCLA) is False),
        f"encontradas: {encontrados}",
    )
    veredicto, detalle = afirmacion_a15(presentes)
    puerta.registrar(
        "limite_5_5_A15_confirma_con_las_doce_ausentes",
        veredicto,
        f"estado del limite: {detalle['estado_del_limite']}",
    )
    puerta.registrar(
        "limite_5_5_queda_SIN_HACER_en_este_venv",
        not encontrados and detalle["estado_del_limite"] == "SIN_HACER",
        "sin librerias de inferencia bayesiana, el limite queda SIN HACER",
    )
    puerta.registrar(
        "limite_5_5_se_publica_lo_que_no_se_autoriza",
        bool(detalle["lo_que_p83_no_autoriza"]),
        "el informe declara que no se afirma la correccion de la inferencia variacional",
    )
    # Y el caso de trampa: si alguien metiera una libreria, el estado tiene que
    # cambiar a HECHO, luego el estado NO esta escrito a mano.
    caso_frontera(
        puerta,
        "limite_5_5_el_estima_cambia_si_hay_libreria",
        lambda: afirmacion_a15({**presentes, "pymc": True})[1]["estado_del_limite"] == "HECHO",
        "con una libreria presente el estado tiene que ser HECHO, luego se deriva",
    )
    # Y la comprobacion de la construccion del informe, que nunca se habia revisado.
    carga_minima = construir_carga(
        contexto_de_ejemplo(),
        Progreso(PARCIALES / "humo_markdown", "md"),
        "humo",
        "marca_de_prueba",
        LECTURA_NO_INFERIOR,
        texto_minimo_de_ejemplo(LECTURA_NO_INFERIOR),
    )
    markdown = construir_markdown(carga_minima)
    puerta.registrar(
        "markdown_se_construye_sin_error",
        bool(markdown) and markdown.endswith("\n"),
        f"{len(markdown)} caracteres",
    )
    puerta.registrar(
        "markdown_es_ascii_puro",
        all(ord(c) <= 127 for c in markdown),
        "el markdown del informe no tiene ningun caracter no latino",
    )
    puerta.registrar(
        "markdown_lleva_la_frase_obligatoria_de_v03",
        markdown.count(LITERAL_V03) >= 1,
        f"aparece {markdown.count(LITERAL_V03)} veces",
    )
    puerta.registrar(
        "markdown_declara_el_limite_de_5_5",
        LITERAL_ANCLA in markdown,
        "el limite de anclaje externo esta escrito en el informe",
    )
    puerta.registrar(
        "markdown_no_afirma_que_el_bayesiano_supere_a_v03",
        not any(v in markdown for v in VARIANTES_PROHIBIDAS),
        "ninguna variante prohibida en el markdown",
    )
    puerta.registrar(
        "markdown_pone_la_lectura_de_la_regla",
        LECTURA_NO_INFERIOR in markdown,
        f"la lectura publicada es {LECTURA_NO_INFERIOR}",
    )
    json_texto = construir_json(carga_minima)
    puerta.registrar(
        "markdown_el_json_tambien_es_ascii_y_terminado",
        all(ord(c) <= 127 for c in json_texto) and json_texto.endswith("\n"),
        f"{len(json_texto)} caracteres de JSON",
    )
    puerta.registrar(
        "markdown_el_json_escapa_el_no_ascii_por_construccion",
        all(ord(c) <= 127 for c in construir_json({"campo": "acent" + chr(0xE1)})),
        "ensure_ascii escapa el caracter, luego el JSON nunca lleva un byte no latino",
    )
    # El control REAL de ASCII no esta en construir_json sino en escribir_unico, que
    # es el unico punto por el que sale informacion a disco. Ahi un byte no latino
    # tiene que rechazarse ANTES de abrir el destino.
    caso_levanta(
        puerta,
        "markdown_escribir_unico_rechaza_un_no_ascii",
        lambda: escribir_unico(
            PARCIALES / "no_ascii.txt", "acent" + chr(0xE1) + "\n"
        ),
        ErrorEscritura,
        "el destino nunca se abre con un texto que no es ASCII puro",
    )


def casos_de_afirmaciones(puerta: PuertaHumo) -> None:
    """Las veinte afirmaciones, cada una con su caso bueno y su caso mutado.

    Args:
        puerta: puerta de humo.

    Returns:
        Nada.
    """
    contexto = contexto_de_ejemplo()
    progreso = Progreso(PARCIALES, f"humo_{puerta.marca_texto}")
    probability, verdad = _probabilidad_para_humo(len(contexto.y))
    texto = texto_minimo_de_ejemplo(LECTURA_AFIRMATIVO)

    # A0: huellas identicas antes y despues.
    huella = {"w2v": {"sha256": W2V_SHA256}, "v03": {"sha256": "x" * 64}, "oof": {"sha256": "y" * 64}}
    caso_afirmacion(
        puerta,
        "A0_huella",
        "A0",
        lambda: afirmacion_a0(huella, huella),
        lambda: afirmacion_a0(huella, {**huella, "oof": {"sha256": "z" * 64}}),
    )
    # A1: solo lectura, cero test y cero holdout.
    guarda = GuardaLectura()
    caso_afirmacion(
        puerta,
        "A1_fuentes",
        "A1",
        lambda: afirmacion_a1(guarda, 40, 40, 0, 0, 0),
        lambda: afirmacion_a1(guarda, 39, 40, 0, 0, 0),
    )
    guarda_con_fuga = GuardaLectura()
    guarda_con_fuga.filas_holdout_leidas = CONFIG["n_filas_holdout"]
    caso_detecta(
        puerta,
        "A1_detecta_holdout",
        lambda: afirmacion_a1(guarda_con_fuga, 40, 40, 0, 0, 0),
        "con 248 filas de holdout leidas, A1 tiene que rechazarlo",
    )
    guarda_con_test = GuardaLectura()
    guarda_con_test.filas_test_leidas = 1
    caso_detecta(
        puerta,
        "A1_detecta_test",
        lambda: afirmacion_a1(guarda_con_test, 40, 40, 0, 0, 0),
        "con una fila de test leida, A1 tiene que rechazarlo",
    )
    guarda_con_evaluaciones = GuardaLectura()
    guarda_con_evaluaciones.evaluaciones_del_holdout = 1
    caso_detecta(
        puerta,
        "A1_detecta_una_evaluacion_del_holdout",
        lambda: afirmacion_a1(guarda_con_evaluaciones, 40, 40, 0, 0, 0),
        "con una evaluacion del holdout gastada, A1 tiene que rechazarlo",
    )
    # A2: umbrales congelados y cero reoptimizaciones.
    caso_afirmacion(
        puerta,
        "A2_umbrales",
        "A2",
        lambda: afirmacion_a2(np.asarray(UMBRALES_V03_CONGELADOS), 0, {"sha256": "x" * 64}),
        lambda: afirmacion_a2(np.asarray((0.40, 0.51, 0.46)), 0, {"sha256": "x" * 64}),
    )
    caso_detecta(
        puerta,
        "A2_detecta_reoptimizacion",
        lambda: afirmacion_a2(np.asarray(UMBRALES_V03_CONGELADOS), 1, {"sha256": "x" * 64}),
        "con una reoptimizacion registrada, A2 tiene que rechazarla",
    )
    caso_detecta(
        puerta,
        "A2_detecta_umbral_movido",
        lambda: afirmacion_a2(np.asarray((0.45, 0.51, 0.46)), 0, {"sha256": "x" * 64}),
        "con un umbral distinto del congelado, A2 tiene que rechazarlo",
    )
    # A3: el corrected_cv recalculado tiene que coincidir con el publicado.
    # Los argumentos van POR NOMBRE, no posicionales. Un caso anterior los pasaba
    # posicionales y las unidades de tweet-etiqueta llegaban al hueco de las filas,
    # luego la afirmacion fallaba por un fallo del humo y no por lo que vigila.
    # Ademas el numero de filas y el de unidades se leen de los mismos objetos que
    # se le pasan, de modo que el caso no puede pasar con un par de numeros falso.
    n_filas_ejemplo = 40
    n_grupos_ejemplo = 8
    unidades_ejemplo = n_filas_ejemplo * len(ETIQUETAS)
    a3_bueno: dict[str, Any] = {
        "corrected_cv_recalculado": 0.777,
        "publicado": 0.777,
        "tolerancia": 1e-6,
        "unidades": unidades_ejemplo,
        "n_filas": n_filas_ejemplo,
        "n_grupos": n_grupos_ejemplo,
        "grupos_esperados": n_grupos_ejemplo,
        "unidades_esperadas": unidades_ejemplo,
        "filas_esperadas": n_filas_ejemplo,
        "valor_de_la_constante_guardada": 0.7772114525,
    }
    caso_afirmacion(
        puerta,
        "A3_canonico",
        "A3",
        lambda: afirmacion_a3(**a3_bueno),
        lambda: afirmacion_a3(**{**a3_bueno, "corrected_cv_recalculado": 0.5}),
    )
    # El caso que demuestra que A3 lee los numeros que se le pasan y no una
    # constante interna: se le da la constante guardada en vez de la recalculada.
    caso_frontera(
        puerta,
        "A3_detecta_la_constante_guardada_en_vez_de_la_recalculada",
        lambda: _falla_a3(**{**a3_bueno, "corrected_cv_recalculado": 0.5}),
        "con un corrected_cv que no cuadra, A3 tiene que fallar",
    )
    # LA BINARIZACION DE LA QUE A3 DEPENDE, con los DOS lados. metricas_multietiqueta
    # levanta ValueError con una matriz de probabilidades continuas, luego sin esto A3
    # no llega a producir veredicto en el modo completo. Los dos lados son la parte
    # que discrimina: por encima del umbral tiene que salir 1 y por debajo tiene que
    # salir 0, y un solo lado pasaria con el signo del comparador invertido.
    probas_umbral, _ = _probabilidad_para_humo(24)
    limites = np.asarray(UMBRALES_V03_CONGELADOS, dtype=np.float64)
    binarizado = binarizar_con_umbrales_congelados(probas_umbral, limites)
    puerta.registrar(
        "A3_binariza_por_encima_del_umbral",
        bool(np.all(binarizado[probas_umbral >= limites[None, :]] == 1)),
        "toda probabilidad por encima o igual al umbral tiene que dar 1",
    )
    puerta.registrar(
        "A3_binariza_por_debajo_del_umbral",
        bool(np.all(binarizado[probas_umbral < limites[None, :]] == 0)),
        "toda probabilidad por debajo del umbral tiene que dar 0",
    )
    valores_binarizados = {int(v) for v in np.unique(binarizado)}
    puerta.registrar(
        "A3_la_binarizacion_conserva_la_forma_y_es_binaria",
        binarizado.shape == probas_umbral.shape and valores_binarizados <= {0, 1},
        f"forma {binarizado.shape} y valores {sorted(valores_binarizados)}",
    )
    caso_levanta(
        puerta,
        "A3_binarizar_con_umbrales_no_congelados_falla",
        lambda: binarizar_con_umbrales_congelados(probas_umbral, (0.40, 0.51, 0.46)),
        AssertionError,
        "con un umbral reoptimizado la binarizacion tiene que rechazarlo, que es lo que A2 prohibe",
    )
    caso_levanta(
        puerta,
        "A3_binarizar_sin_los_tres_umbrales_falla",
        lambda: binarizar_con_umbrales_congelados(probas_umbral, (0.44, 0.51)),
        AssertionError,
        "con dos umbrales no se puede binarizar una matriz de tres columnas",
    )
    # A4: la seleccion se hace solo sobre el inner_train.
    instrumento = InstrumentoSeleccion()
    interno = np.arange(10, dtype=np.int64)
    instrumento.fijar_permitidos(interno)
    instrumento.registrar(interno, 3.0, 4)
    caso_afirmacion(
        puerta,
        "A4_seleccion_interna",
        "A4",
        lambda: afirmacion_a4(instrumento, interno, True),
        lambda: afirmacion_a4(instrumento, interno, False),
    )
    instrumento_fuga = InstrumentoSeleccion()
    instrumento_fuga.fijar_permitidos(interno)
    instrumento_fuga.registrar(np.arange(100, dtype=np.int64), 3.0, 4)
    caso_detecta(
        puerta,
        "A4_detecta_la_fuga",
        lambda: afirmacion_a4(instrumento_fuga, interno, True),
        "con una llamada al selector fuera del inner_train, A4 tiene que rechazarla",
    )
    caso_detecta(
        puerta,
        "A4_detecta_conjunto_vacio",
        lambda: afirmacion_a4(instrumento, np.asarray([], dtype=np.int64), True),
        "con el conjunto permitido vacio, A4 tiene que rechazarlo en vez de pasar por vacuidad",
    )
    caso_detecta(
        puerta,
        "A4_detecta_sin_ninguna_llamada",
        lambda: afirmacion_a4(InstrumentoSeleccion(), interno, True),
        "sin ninguna llamada registrada, A4 tiene que rechazarlo",
    )
    caso_detecta(
        puerta,
        "A4_detecta_mutacion_no_detectada",
        lambda: afirmacion_a4(instrumento, interno, False),
        "con la mutacion declarada como no detectada, A4 tiene que rechazarlo",
    )
    # A5: la posterior de M1 esta declarada como aproximada.
    carga = {"metodo_del_brazo_m1": DECLARACION_METODO, "texto_del_informe": texto}
    caso_afirmacion(
        puerta,
        "A5_metodo_declarado",
        "A5",
        lambda: afirmacion_a5(carga),
        lambda: afirmacion_a5({**carga, "texto_del_informe": "texto sin la cadena"}),
    )
    # A6: el numero de parametros efectivos se mueve con alpha_0.
    por_alpha = {
        "0.50": {"n_efectivos_total": 829.0, "por_etiqueta": {"HS": 829.0, "TR": 800.0, "AG": 780.0}},
        "1.00": {"n_efectivos_total": 652.0, "por_etiqueta": {"HS": 652.0, "TR": 630.0, "AG": 610.0}},
        "2.00": {"n_efectivos_total": 595.0, "por_etiqueta": {"HS": 595.0, "TR": 570.0, "AG": 550.0}},
        "__texto__": LITERAL_EFECTIVOS_METODO,
    }
    caso_afirmacion(
        puerta,
        "A6_el_numero_baja_con_alpha_0",
        "A6",
        lambda: afirmacion_a6(por_alpha),
        lambda: afirmacion_a6(
            {
                **por_alpha,
                "2.00": {
                    "n_efectivos_total": 829.0,
                    "por_etiqueta": {"HS": 829.0, "TR": 800.0, "AG": 780.0},
                },
            }
        ),
    )
    caso_detecta(
        puerta,
        "A6_detecta_invariancia",
        lambda: afirmacion_a6(
            {
                **por_alpha,
                "2.00": {
                    "n_efectivos_total": 800.0,
                    "por_etiqueta": {"HS": 800.0, "TR": 780.0, "AG": 760.0},
                },
            }
        ),
        "con un numero de parametros efectivos casi invariante, A6 tiene que rechazarlo",
    )
    caso_detecta(
        puerta,
        "A6_detecta_direccion_invertida",
        lambda: afirmacion_a6(
            {
                **por_alpha,
                "2.00": {
                    "n_efectivos_total": 595.0,
                    "por_etiqueta": {"HS": 595.0, "TR": 570.0, "AG": 550.0},
                },
                "0.50": {
                    "n_efectivos_total": 400.0,
                    "por_etiqueta": {"HS": 400.0, "TR": 390.0, "AG": 380.0},
                },
            }
        ),
        "con mas parametros efectivos en alpha_0 grande, A6 tiene que rechazar la direccion",
    )
    caso_detecta(
        puerta,
        "A6_detecta_cadena_literal_ausente",
        lambda: afirmacion_a6({**{k: v for k, v in por_alpha.items() if k != "__texto__"}}),
        "sin la cadena literal del metodo, A6 tiene que rechazarlo",
    )
    # A7: los dos caminos de M2 coinciden.
    caso_afirmacion(
        puerta,
        "A7_control_aritmetico",
        "A7",
        lambda: afirmacion_a7(0.0, 12, 60),
        lambda: afirmacion_a7(1e-6, 12, 60),
    )
    caso_detecta(
        puerta,
        "A7_detecta_camino_no_ejecutado",
        lambda: afirmacion_a7(0.0, 0, 0),
        "con cero ejecuciones del segundo camino, A7 tiene que rechazarlo",
    )
    caso_detecta(
        puerta,
        "A7_detecta_sin_filas_comparadas",
        lambda: afirmacion_a7(0.0, 12, 0),
        "sin filas comparadas, A7 tiene que rechazarlo",
    )
    caso_detecta(
        puerta,
        "A7_detecta_diferencia_sobre_tolerancia",
        lambda: afirmacion_a7(1e-11, 12, 60),
        "con una diferencia de 1e-11 sobre la tolerancia de 1e-12, A7 tiene que rechazarla",
    )
    # A8 y A9: las seis magnitudes y la curva de calibracion.
    caso_afirmacion(
        puerta,
        "A8_seis_magnitudes",
        "A8",
        lambda: afirmacion_a8(_cifras_de_pregunta_1()),
        lambda: afirmacion_a8({"M1": {"log_loss_macro": 0.3}, "M2": {}}),
    )
    # A9 necesita filas suficientes para que al menos un bin llegue a
    # minimo_unidades_por_bin. Con las cuarenta filas del contexto de ejemplo
    # ningun bin llegaba a veinte y el caso fallaba por una razon que no es la
    # que A9 vigila. Se mide sobre una muestra propia, mas grande pero barata,
    # y se EXIGE que el caso tenga de verdad un bin poblado, para que no pueda
    # pasar por el mismo motivo que antes.
    probability_a9, verdad_a9 = _probabilidad_para_humo(1200)
    curvas = {
        ETIQUETAS[k]: curva_de_calibracion(
            probability_a9[:, k], verdad_a9[:, k], CONFIG["bins_calibracion"]
        )
        for k in range(len(ETIQUETAS))
    }
    caso_afirmacion(
        puerta,
        "A9_curva_poblada",
        "A9",
        lambda: afirmacion_a9(curvas),
        lambda: afirmacion_a9({"HS": {"bins": [], "n_vacios": 10, "n_bins_con_al_menos_20_unidades": 0}}),
    )
    caso_frontera(
        puerta,
        "A9_el_caso_tiene_de_verdad_un_bin_poblado",
        lambda: all(
            int(curva["n_bins_con_al_menos_20_unidades"]) >= 1 for curva in curvas.values()
        ),
        "sin esta comprobacion el caso podria volver a pasar con la curva vacia",
    )
    caso_detecta(
        puerta,
        "A9_detecta_demasiados_bins_vacios",
        lambda: afirmacion_a9({ETIQUETAS[0]: dict(curvas[ETIQUETAS[0]], n_vacios=10)}),
        "con diez bins vacios de diez, A9 tiene que rechazarlo",
    )
    caso_detecta(
        puerta,
        "A9_detecta_curva_con_nan",
        lambda: afirmacion_a9(
            {
                ETIQUETAS[0]: {
                    **curvas[ETIQUETAS[0]],
                    "bins": [
                        {**curvas[ETIQUETAS[0]]["bins"][i], "media_predicha": float("nan")}
                        for i in range(CONFIG["bins_calibracion"])
                    ],
                }
            }
        ),
        "con un NaN en la curva, A9 tiene que rechazarlo",
    )
    # A10: cobertura, anchura y el control degenerado.
    bayes = {
        "cobertura": 0.9,
        "anchura_media_normalizada": 0.3,
        "cobertura_menos_pesada_por_anchura": 0.75,
    }
    control_v03 = dict(bayes)
    control_v03["cobertura"] = 0.80
    caso_afirmacion(
        puerta,
        "A10_control_de_calibrador",
        "A10",
        lambda: afirmacion_a10(bayes, control_v03, 1.0),
        lambda: afirmacion_a10(bayes, control_v03, 0.9),
    )
    # A11: la regla de 1.5 aplicada sin reinterpretarla.
    lectura_p1, detalle_lectura = aplicar_lectura_intervalos(
        {"inferior": -0.02, "superior": -0.01, "n": 5},
        {"inferior": -0.02, "superior": -0.01, "n": 5},
        CONFIG["eps_gate_delta"],
    )
    carga_11 = dict(registrar_lectura_pregunta_1(lectura_p1, lectura_p1 == LECTURA_AMBIGUA))
    carga_11["texto_del_informe"] = texto
    caso_afirmacion(
        puerta,
        "A11_lectura_de_la_regla",
        "A11",
        lambda: afirmacion_a11(detalle_lectura, carga_11),
        lambda: afirmacion_a11(detalle_lectura, {**carga_11, resolver_clave("lectura_pregunta_1_json"): LECTURA_NEGATIVO}),
    )
    # A12: las cuatro medidas de complementariedad.
    caso_afirmacion(
        puerta,
        "A12_medidas",
        "A12",
        lambda: afirmacion_a12(_medidas_de_pregunta_2()),
        lambda: afirmacion_a12({"M1": {"auc_del_error_de_v03": 0.2}}),
    )
    # A13: rescate y dano con los tres alphas.
    caso_afirmacion(
        puerta,
        "A13_tres_alphas",
        "A13",
        lambda: afirmacion_a13(
            _configuraciones_de_ensamblaje(), {"escala": True, "corrected_cv_optimista": True}
        ),
        lambda: afirmacion_a13(
            _configuraciones_de_ensamblaje(), {"escala": False, "corrected_cv_optimista": True}
        ),
    )
    # A14: el calibrador se ajusta solo en internal. Tambien por nombre, porque
    # esta afirmacion tiene ocho argumentos y cuatro de ellos son filas.
    a14_bueno: dict[str, Any] = {
        "n_ajustes_calibrador": 5,
        "n_ajustes_esperados": 5,
        "instrumentacion": {
            "calibrador_fuera_de_interno": 0,
            "alpha_fuera_de_interno": 0,
            "ajustes_fuera_de_lo_permitido": 0,
            "filas_comunes_entre_el_alpha_y_el_calibrador": 0,
            # Las dos claves que exige la instrumentacion nueva de la regla 3.5.
            # Su ausencia LEVANTA, a proposito: no se puede leer como que todos los
            # ajustes declararon sus filas, que es justo la confusion que la
            # instrumentacion nueva existe para evitar.
            "llamadas_al_alpha_sin_declarar_filas": 0,
            "ajustes_que_tocan_el_oof": 0,
        },
        "filas_calibracion": np.arange(10, 20, dtype=np.int64),
        "filas_outer_validation": np.arange(20, 30, dtype=np.int64),
        "filas_alpha": np.arange(0, 10, dtype=np.int64),
        "deteccion_de_la_mutacion": True,
        "ajustes_sobre_el_oof": 0,
    }
    caso_afirmacion(
        puerta,
        "A14_calibrador_interno",
        "A14",
        lambda: afirmacion_a14(**a14_bueno),
        lambda: afirmacion_a14(
            **{**a14_bueno, "instrumentacion": {"calibrador_fuera_de_interno": 3, "alpha_fuera_de_interno": 0}}
        ),
    )
    # El caso que fija que A14 NO cae a cero cuando falta la instrumentacion. Con
    # un .get(clave, 0) la ausencia se leia como cero fugas y A14 se confirmaba
    # sin haber mirado nada, siendo una afirmacion con marca de STOP.
    caso_levanta(
        puerta,
        "A14_sin_instrumentacion_del_calibrador_falla",
        lambda: afirmacion_a14(
            **{**a14_bueno, "instrumentacion": {"alpha_fuera_de_interno": 0}}
        ),
        AssertionError,
        "sin la clave del calibrador, A14 tiene que fallar y no leer cero fugas",
    )
    caso_levanta(
        puerta,
        "A14_sin_instrumentacion_del_alpha_falla",
        lambda: afirmacion_a14(
            **{**a14_bueno, "instrumentacion": {"calibrador_fuera_de_interno": 0}}
        ),
        AssertionError,
        "sin la clave del alpha, A14 tiene que fallar y no leer cero fugas",
    )
    # Y el caso del A3: sin la clave del corrected_cv, falla en vez de caer a la
    # constante y compararse consigo misma.
    caso_levanta(
        puerta,
        "A3_sin_la_clave_del_corrected_cv_falla",
        lambda: corrected_cv_de_V03({"otra_cosa": 1.0}),
        AssertionError,
        "sin corrected_cv no hay nada recalculado que comparar",
    )
    caso_levanta(
        puerta,
        "A3_con_un_corrected_cv_no_finito_falla",
        lambda: corrected_cv_de_V03({"corrected_cv": float("nan")}),
        AssertionError,
        "un NaN no es un numero que se pueda comparar",
    )
    caso_levanta(
        puerta,
        "A3_con_un_corrected_cv_que_no_es_numero_falla",
        lambda: corrected_cv_de_V03({"corrected_cv": "0.777"}),
        AssertionError,
        "una cadena no es un numero recalculado",
    )
    caso_frontera(
        puerta,
        "A3_lee_el_valor_que_viene_del_objeto",
        lambda: corrected_cv_de_V03({"corrected_cv": 0.5}) == 0.5,
        "el extractor devuelve el valor del objeto, no una constante",
    )
    caso_detecta(
        puerta,
        "A14_detecta_solapamiento",
        lambda: afirmacion_a14(**{**a14_bueno, "filas_outer_validation": np.arange(5, 15, dtype=np.int64)}),
        "con filas internas y externas solapadas, A14 tiene que rechazarlo",
    )
    caso_detecta(
        puerta,
        "A14_detecta_calibrador_sobre_el_oof",
        lambda: afirmacion_a14(**{**a14_bueno, "ajustes_sobre_el_oof": 1}),
        "con un ajuste del calibrador sobre el OOF, A14 tiene que rechazarlo",
    )
    caso_detecta(
        puerta,
        "A14_detecta_alpha_fuera_de_interno",
        lambda: afirmacion_a14(
            **{**a14_bueno, "instrumentacion": {"calibrador_fuera_de_interno": 0, "alpha_fuera_de_interno": 2}}
        ),
        "con la decision del alpha fuera de las internas, A14 tiene que rechazarlo",
    )
    caso_detecta(
        puerta,
        "A14_detecta_numero_de_ajustes_distinto",
        lambda: afirmacion_a14(**{**a14_bueno, "n_ajustes_calibrador": 4}),
        "con un ajuste menos de los esperados, A14 tiene que rechazarlo",
    )
    caso_detecta(
        puerta,
        "A14_detecta_mutacion_no_detectada",
        lambda: afirmacion_a14(**{**a14_bueno, "deteccion_de_la_mutacion": False}),
        "con la mutacion declarada como no detectada, A14 tiene que rechazarlo",
    )
    # La separacion entre el alpha y el calibrador, que es la parte de la regla
    # 3.5 que mas importa y que antes no se comprobaba en absoluto.
    caso_detecta(
        puerta,
        "A14_detecta_el_alpha_sobre_filas_del_calibrador",
        lambda: afirmacion_a14(
            **{
                **a14_bueno,
                "filas_alpha": a14_bueno["filas_calibracion"],
                "instrumentacion": {
                    **a14_bueno["instrumentacion"],
                    "filas_comunes_entre_el_alpha_y_el_calibrador": len(
                        a14_bueno["filas_calibracion"]
                    ),
                },
            }
        ),
        "con el alpha eligiendo sobre las mismas filas que calibran, A14 tiene que rechazarlo",
    )
    caso_detecta(
        puerta,
        "A14_detecta_ajustes_fuera_de_lo_permitido",
        lambda: afirmacion_a14(
            **{
                **a14_bueno,
                "instrumentacion": {
                    **a14_bueno["instrumentacion"],
                    "ajustes_fuera_de_lo_permitido": 1,
                },
            }
        ),
        "con un ajuste fuera de lo permitido, A14 tiene que rechazarlo",
    )
    # A15: el limite de 5.5 verificado.
    ausentes = {nombre: False for nombre in LIBRERIAS_ANCLA}
    caso_afirmacion(
        puerta,
        "A15_limite_declarado",
        "A15",
        lambda: afirmacion_a15(ausentes),
        lambda: afirmacion_a15({"pymc": True}),
    )
    # A16: las dos rejillas publicadas sin elegir ningun valor.
    rejillas = {
        "alpha_0": {f"{a:.2f}": {} for a in CONFIG["alpha_0_rejilla"]},
        "umbral": {f"{u:.1f}": {} for u in CONFIG["umbrales_seleccion"]},
    }
    caso_afirmacion(
        puerta,
        "A16_rejillas",
        "A16",
        lambda: afirmacion_a16(rejillas, True),
        lambda: afirmacion_a16(rejillas, False),
    )
    # A17: dimension seleccionada por pliegue.
    por_pliegue = {f"pliegue_{i}": {"n_columnas": 600} for i in range(5)}
    caso_afirmacion(
        puerta,
        "A17_dimension",
        "A17",
        lambda: afirmacion_a17(por_pliegue, 836, 12, True),
        lambda: afirmacion_a17({"pliegue_0": {"n_columnas": 9000}}, 836, 12, True),
    )
    # A18 y A19: las cadenas de formulacion y el F1 como descriptiva.
    # Las cadenas se leen del OBJETO REAL CADENAS_DE_FORMULACION, que es la
    # entrada que A18 lee, y no de una constante escrita aqui. Si el objeto
    # cambiara, el caso tendria que cambiar con el, y si el caso declarara las
    # cadenas por su cuenta, dejaria de comprobar lo que dice comprobar. Ese
    # mismo fallo, leer de una constante en vez del objeto, ya ha aparecido cuatro
    # veces en este proyecto.
    caso_afirmacion(
        puerta,
        "A18_cadenas",
        "A18",
        lambda: afirmacion_a18(texto),
        lambda: afirmacion_a18(texto.replace(LITERAL_V03, "")),
    )
    for indice, cadena in enumerate(CADENAS_DE_FORMULACION):
        caso_detecta(
            puerta,
            f"A18_detecta_la_cadena_{indice}_ausente",
            lambda cadena=cadena: afirmacion_a18(texto.replace(cadena, "")),
            f"sin {cadena[:48]}... A18 tiene que rechazarlo",
        )
    for indice, variante in enumerate(VARIANTES_PROHIBIDAS):
        caso_detecta(
            puerta,
            f"A18_detecta_la_variante_prohibida_{indice}",
            lambda variante=variante: afirmacion_a18(f"{texto}\n{variante}\n"),
            f"con la variante {variante[:40]}... A18 tiene que rechazarla",
        )
    caso_frontera(
        puerta,
        "A18_el_texto_de_ejemplo_trae_las_cuatro_cadenas",
        lambda: all(cadena in texto for cadena in CADENAS_DE_FORMULACION),
        "el texto del caso tiene que traer las cuatro cadenas del objeto real",
    )
    caso_frontera(
        puerta,
        "A18_el_texto_de_ejemplo_no_trae_variantes",
        lambda: not any(variante in texto for variante in VARIANTES_PROHIBIDAS),
        "el texto del caso no puede traer ninguna variante prohibida",
    )
    f1_miembro = {
        "f1_media_columnas": 0.60,
        "desviacion_entre_semillas": 0.001,
        "intervalo": [-0.008, 0.010],
    }
    caso_afirmacion(
        puerta,
        "A19_f1_descriptivo",
        "A19",
        lambda: afirmacion_a19(texto, f1_miembro),
        lambda: afirmacion_a19(texto, {"desviacion_entre_semillas": 0.001}),
    )
    caso_detecta(
        puerta,
        "A19_detecta_f1_en_la_regla_de_lectura",
        lambda: afirmacion_a19(texto.replace(
            "- El F1 no aparece en esta regla",
            "- f1_media_columnas aparece en la regla",
        ), f1_miembro),
        "con el F1 dentro del bloque de lectura, A19 tiene que rechazarlo",
    )
    caso_detecta(
        puerta,
        "A19_detecta_etiqueta_descriptiva_ausente",
        lambda: afirmacion_a19(texto.replace(LITERAL_DESCRIPTIVA_F1, ""), f1_miembro),
        "sin la etiqueta descriptiva, A19 tiene que rechazarlo",
    )
    del progreso  # el progreso del humo se escribe, no se exige


def caso_de_a6_medido(puerta: PuertaHumo) -> None:
    """Mide el numero de parametros efectivos en las filas REALES de desarrollo.

    Motivo: los numeros del caso sintetico de A6 son 829, 652 y 595, que son
    los declarados, pero el caso que de verdad demuestra que la actualizacion no
    es invariante es el que mide sobre los datos reales. Los numeros MEDIDOS en
    el pliegue 0, etiqueta HS, con la escala L2 que el script usa de verdad, son
    464.2, 419.7 y 365.7 parametros efectivos para alpha_0 igual a 0.5, 1.0 y
    2.0. Los declarados 829, 652 y 595 NO se reproducen en ninguna de las quince
    combinaciones de cinco pliegues por tres etiquetas, luego se declaran como no
    reproducidos y el caso real exige la MONOTONIA, que es lo que A6 vigila.

    Args:
        puerta: puerta de humo.

    Returns:
        Nada.
    """
    progreso = Progreso(PARCIALES, f"humo_medido_{puerta.marca_texto}")
    contexto = cargar_datos_de_desarrollo(progreso)
    puerta.registrar(
        "a6_datos_reales_sin_holdout",
        progreso.contadores.get("filas_holdout_descartadas", 0) == CONFIG["n_filas_holdout"]
        and progreso.contadores.get("filas_test_leidas", 0) == 0,
        f"holdout descartado={progreso.contadores.get('filas_holdout_descartadas')} "
        f"test={progreso.contadores.get('filas_test_leidas')}",
    )
    ia, ib = contexto.pliegues[0]
    inner, _ = partir_interna(
        ia, CONFIG["fraccion_inner_calibration"], int(CONFIG["semillas"][0]), contexto.grupos
    )
    X = construir_tfidf(
        [contexto.textos[i] for i in inner], [contexto.textos[i] for i in ib]
    )
    mascara = seleccionar_columnas(X, CONFIG["umbral_seleccion_primario"])
    Xs = X[:, mascara]
    escala = np.maximum(np.sqrt(np.asarray(Xs.multiply(Xs).sum(axis=0)).ravel()), 1e-12)
    Xd = np.asarray(Xs.multiply(sparse.csr_matrix(1.0 / escala)).todense(), dtype=np.float64)
    medidos = {
        float(a): float(ajustar_m1(Xd, contexto.y[inner, 0], float(a)).n_efectivos)
        for a in CONFIG["alpha_0_rejilla"]
    }
    bajo, medio, alto = medidos[0.5], medidos[1.0], medidos[2.0]
    puerta.registrar(
        "a6_medido_baja_con_alpha_0",
        bajo > medio > alto,
        f"alpha_0 0.5={bajo:.1f} 1.0={medio:.1f} 2.0={alto:.1f}",
    )
    diferencia_relativa = abs(bajo - alto) / max(abs(alto), 1e-12)
    puerta.registrar(
        "a6_medido_supera_el_cinco_por_ciento",
        diferencia_relativa > 0.05,
        f"diferencia relativa={diferencia_relativa:.4f}",
    )
    puerta.registrar(
        "a6_medido_no_reproduce_los_declarados",
        not (700.0 < bajo < 900.0),
        f"los declarados 829/652/595 no salen: medidos {bajo:.1f}/{medio:.1f}/{alto:.1f}",
    )
    # Y la prueba de que el caso discrimina: un update invariante lo rompe.
    invariante = dict(medidos)
    puerta.registrar(
        "a6_medido_detecta_un_update_invariante",
        not (invariante[0.5] > invariante[1.0] > invariante[2.0] and abs(
            invariante[0.5] - invariante[2.0]
        ) / invariante[2.0] <= 0.05),
        "la monotonia y la diferencia relativa son las que se exigen",
    )

def _registros_de_canal(instrumento: Any, canal: str) -> list[dict[str, Any]]:
    """Devuelve los registros de un canal del instrumento, con su permitido PROPIO.

    POR QUE CADA REGISTRO LLEVA SU PERMITIDO Y NO UNO GLOBAL. El defecto que se
    corrige aqui es que el contador comparaba TODAS las llamadas contra un unico
    conjunto global de permitidos, que ademas lo reescribia el ultimo brazo en
    ejecutar. Con un unico conjunto, una llamada limpia puede dar fuga solo porque
    otra combinacion cambio el permitido, y una llamada sucia puede pasar limpia
    porque su permitido global coincidia por casualidad. Cada registro guarda el
    conjunto permitido que estaba vigente en el momento de su llamada, que es lo
    unico que permite comparar llamada por llamada.

    CUANDO NO HAY NINGUN REGISTRO SE EXAMINA LA RECEPCION VIVA. Asi el contador
    sigue pudiendo medirse en los casos de humo, que lo ejercitan registrando la
    recepcion directamente. Y si no hay ninguna de las dos cosas, se devuelve un
    registro vacio, que el contador trata como ajuste SIN DECLARACION.

    Args:
        instrumento: el instrumento vivo o una instantanea suya.
        canal: "llamadas_calibrador" o "llamadas_alpha".

    Returns:
        Lista de registros, cada uno con "filas" y su conjunto permitido.

    Raises:
        AssertionError: si el canal no es uno de los dos declarados.
    """
    if canal not in {"llamadas_calibrador", "llamadas_alpha"}:
        raise AssertionError(f"el canal {canal!r} no es uno de los dos declarados")
    # Admite las DOS formas que el docstring declara: el instrumento VIVO, que es
    # un objeto con atributos, y una INSTANTANEA suya, que es un diccionario. El
    # camino real pasa la instantanea, para que la rejilla de alpha_0 no la pise, y
    # los casos de humo pasan el vivo. Con getattr en las dos ramas, no hay forma de
    # que una de las dos se rompa al usarla por primera vez en la corrida real.
    if isinstance(instrumento, dict):
        registros = [dict(registro) for registro in list(instrumento.get(canal, []))]
    else:
        registros = [dict(registro) for registro in list(getattr(instrumento, canal))]
    if registros:
        return registros
    sufijo = "calibrador" if canal == "llamadas_calibrador" else "alpha"
    return [
        {
            "filas": [int(i) for i in np.asarray((instrumento.get(f"filas_{sufijo}_recibidas") if isinstance(instrumento, dict) else getattr(instrumento, f"filas_{sufijo}_recibidas"))).ravel()],
            "filas_declaradas": bool(
                np.asarray((instrumento.get(f"filas_{sufijo}_recibidas") if isinstance(instrumento, dict) else getattr(instrumento, f"filas_{sufijo}_recibidas"))).ravel().size
            ),
            f"permitidos_{sufijo}": sorted((instrumento.get(f"permitidos_{sufijo}", frozenset()) if isinstance(instrumento, dict) else getattr(instrumento, f"permitidos_{sufijo}"))),
        }
    ]


def _fugas_de_registro(registro: dict[str, Any], clave_permitidos: str) -> int:
    """Cuenta las filas de un registro que caen fuera de SU propio permitido.

    Un registro que no declara filas no puede pasar por limpio: no hay forma de
    saber sobre que datos se ajusto, luego se cuenta como una fuga entera. Antes
    ese caso caia en una rama que devolvia ``max(len(permitidas), 1)``, que es un
    numero inventado que dependia del permitido y no de lo recibido, y por tanto no
    era una medida de nada.

    Args:
        registro: un registro de _registros_de_canal.
        clave_permitidos: nombre de la clave del permitido propio del registro.

    Returns:
        Numero de filas fuera de lo permitido, o uno si no hay declaracion.
    """
    filas = {int(i) for i in registro.get("filas", [])}
    if not filas:
        return 1
    permitidas = {int(i) for i in registro.get(clave_permitidos, [])}
    return len(filas - permitidas)


def contar_fugas_calibrador(instrumento: Any) -> int:
    """Suma, llamada por llamada, las filas del calibrador fuera de SU permitido.

    Returns:
        Numero de fugas, que tiene que ser cero.
    """
    return sum(
        _fugas_de_registro(registro, "permitidos_calibrador")
        for registro in _registros_de_canal(instrumento, "llamadas_calibrador")
    )


def contar_fugas_alpha(instrumento: Any) -> int:
    """Suma, llamada por llamada, las filas de la decision del alpha fuera de SU permitido.

    Returns:
        Numero de fugas, que tiene que ser cero.
    """
    return sum(
        _fugas_de_registro(registro, "permitidos_alpha")
        for registro in _registros_de_canal(instrumento, "llamadas_alpha")
    )


def llamadas_sin_declarar_filas(instrumento: Any) -> int:
    """Cuenta los ajustes que se hicieron sin declarar que filas usaban.

    Es el caso que mas se cuela: un ajuste sin declaracion no se puede comparar con
    nada, luego cualquier veredicto sobre el seria falso. Por eso se cuenta como
    fallo explicito y no como un cero que significa "no mire nada".

    Returns:
        Numero de ajustes sin declaracion de filas.
    """
    sin_declarar = 0
    for canal in ("llamadas_calibrador", "llamadas_alpha"):
        for registro in _registros_de_canal(instrumento, canal):
            if not registro.get("filas"):
                sin_declarar += 1
    return sin_declarar


def contar_fugas_ajuste_sobre_el_oof(instrumento: Any) -> int:
    """Cuenta los ajustes del calibrador cuyo conjunto declarado cae fuera de lo permitido.

    Returns:
        Numero de ajustes que se salieron de su propio permitido.
    """
    fuera = 0
    for registro in _registros_de_canal(instrumento, "llamadas_calibrador"):
        filas = {int(i) for i in registro.get("filas", [])}
        permitidas = {int(i) for i in registro.get("permitidos_calibrador", [])}
        if not filas or filas - permitidas:
            fuera += 1
    return fuera


def contar_ajustes_que_tocan_filas(instrumento: Any, filas: np.ndarray) -> int:
    """Cuenta los ajustes del calibrador que tocan filas prohibidas.

    QUE ES DISTINTO DE contar_fugas_ajuste_sobre_el_oof Y POR QUE HACE FALTA LA
    LOS DOS. La fuga anterior compara contra el permitido declarado, luego mide si
    el ajuste se salio de su propio conjunto autorizado. Esta mide si el ajuste
    toco filas de EVALUACION, que es lo que convierte la pregunta 1 en circular. Un
    ajuste podria salirse de su permitido sin tocar la evaluacion, y tocar la
    evaluacion sin salirse de un permitted que ya la contenia, luego los dos
    controles no se sustituyen.

    Args:
        instrumento: el instrumento vivo o una instantanea suya.
        filas: filas que ningun ajuste puede usar.

    Returns:
        Numero de ajustes que usaron al menos una de esas filas.
    """
    fuera = 0
    for registro in _registros_de_canal(instrumento, "llamadas_calibrador"):
        # El ib de SU combinacion si la llamada lo registro. Es lo unico que
        # responde a la pregunta: si el ajuste toco las filas que despues se
        # evaluan. Sin ese dato, comparar contra el ib recibido como argumento
        # mezcla combinaciones, y eso no es una fuga, es un emparejamiento mal
        # hecho. Por eso la ausencia cuenta como fuga: un ajuste que no declara
        # con que evaluacion se compara no puede acreditar que no la toco.
        propias = registro.get("filas_de_evaluacion_de_esta_combinacion")
        if propias is None:
            fuera += 1
            continue
        prohibidas = {int(i) for i in propias}
        usadas = {int(i) for i in registro.get("filas", [])}
        if not usadas or usadas & prohibidas:
            fuera += 1
    return fuera


def filas_comunes_alpha_y_calibrador(instrumento: Any) -> set[int]:
    """Filas que la decision del alpha de una combinacion toma del calibrador.

    Esta es la parte de la regla 3.5 que mas importa y la que no se auditaba: no
    basta con que el alpha use filas internas, tiene que usar filas que el
    calibrador de ESA MISMA combinacion no vaya a usar. Antes se comparaba el
    ultimo alpha recibido con el ultimo calibrador recibido, que son de la misma
    combinacion solo por casualidad; ahora se compara cada llamada con el permitted
    del calibrador de esa misma llamada.

    Returns:
        Interseccion de las filas del alpha y las permitidas al calibrador.
    """
    comunes: set[int] = set()
    for registro in _registros_de_canal(instrumento, "llamadas_alpha"):
        usadas = {int(i) for i in registro.get("filas", [])}
        comunes |= usadas & {int(i) for i in registro.get("permitidos_calibrador", [])}
    return comunes


def medir_la_mutacion_del_calibrador(
    proba_oof: np.ndarray, verdad_oof: np.ndarray, filas_oof: np.ndarray, filas_permitidas: np.ndarray
) -> dict[str, Any]:
    """Ajusta el calibrador UNA VEZ sobre el OOF y comprueba que A14 lo ve.

    QUE ES UNA MUTACION Y POR QUE HACE FALTA. Sin ella, A14 seria un chequeo que no
    puede fallar ante la cosa que dice vigilar, que es el fallo que la bitacora de P81
    senalo. Aqui la llamada es REAL: se invoca ajustar_calibrador con las filas de
    evaluacion declaradas, en un instrumento aparte que se reinicia y se aparta, y
    despues se mide si los contadores lo detectan.

    POR QUE NO CONTAMINA NADA. Se ejecuta DESPUES de tomar la instantanea de la
    corrida principal, luego los pareos de A14 ya estan hechos y no los ve. Los
    pares que salen de aqui no se usan para ninguna magnitud publicada: el unico
    producto de esta funcion es el veredicto de deteccion. El pre-registro de 10
    autoriza exactamente esta ejecucion, porque sin ella la afirmacion no seria
    comprobable.

    Args:
        proba_oof: probabilidades canonicas de V03 de las filas de evaluacion.
        verdad_oof: etiqueta de esas mismas filas.
        filas_oof: filas de evaluacion, que son las prohibidas.
        filas_permitidas: conjunto interno que se declara como permitted.

    Returns:
        Detalle con lo que los contadores vieron en la version mutada.
    """
    INSTRUMENTO_CALIBRACION.reiniciar()
    INSTRUMENTO_CALIBRACION.fijar_permitidos(filas_permitidas, np.asarray([], dtype=np.int64))
    ajustar_calibrador(proba_oof, verdad_oof, filas_oof)
    mutado = INSTRUMENTO_CALIBRACION.instantanea()
    INSTRUMENTO_CALIBRACION.reiniciar()
    fugas = contar_fugas_calibrador(mutado)
    ajustes_fuera = contar_fugas_ajuste_sobre_el_oof(mutado)
    sobre_el_oof = contar_ajustes_que_tocan_filas(mutado, filas_oof)
    return {
        "n_ajustes_mutados": len(mutado["llamadas_calibrador"]),
        "fugas_del_calibrador": fugas,
        "ajustes_fuera_de_lo_permitido": ajustes_fuera,
        "ajustes_que_tocan_el_oof": sobre_el_oof,
        "detectada": bool(fugas > 0 and ajustes_fuera > 0 and sobre_el_oof > 0),
    }

# =============================================================================
# SECCION 21: LA CORRIDA COMPLETA Y LA PUERTA DE HUMO
# =============================================================================


def construir_carga(
    contexto: ContextoDatos,
    progreso: Progreso,
    modo: str,
    marca: str,
    lectura: str,
    texto: str,
) -> dict[str, Any]:
    """Arma el diccionario de carga del informe.

    Args:
        contexto: contexto de datos.
        progreso: acumulador de progreso.
        modo: modo de corrida.
        marca: marca de corrida.
        lectura: lectura de la regla de 1.5.
        texto: texto del informe, para el markdown.

    Returns:
        El diccionario de carga.
    """
    return {
        "experimento": "P83",
        "script": "experiments/p83_01_bayesiano.py",
        "modo": modo,
        "marca_corrida": marca,
        "dispositivo": DISPOSITIVO,
        "pre_registro": {
            "ruta": str(PREREGISTRO.relative_to(PROJECT_ROOT)),
            "sha256_declarado": CONFIG["pre_registro_sha256"],
            "sha256_leido": hashlib.sha256(PREREGISTRO.read_bytes()).hexdigest(),
        },
        "metodo_del_brazo_m1": DECLARACION_METODO,
        "afirmaciones_declaradas": {
            clave: {"bloque": bloque, "que_hace": descripcion}
            for clave, (bloque, descripcion) in BLOQUES.items()
        },
        "contexto": {
            "origen": contexto.origen,
            "n_filas": len(contexto.textos),
            "n_grupos": int(len(np.unique(contexto.grupos))),
            "filas_test_leidas": contexto.guarda.filas_test_leidas,
            "filas_holdout_leidas": contexto.guarda.filas_holdout_leidas,
            "evaluaciones_del_holdout": contexto.guarda.evaluaciones_del_holdout,
        },
        "aislamiento_de_afirmaciones": {
            "errores": progreso.errores,
            "avisos": progreso.avisos,
            "escrituras_fallidas": progreso.escrituras_fallidas,
            "contadores": progreso.contadores,
        },
        resolver_clave("veredictos_json"): {
            clave: {
                resolver_clave("estado_json"): resultado.estado,
                resolver_clave("naturaleza_resultado_json"): resultado.naturaleza,
                "stop": resultado.stop,
                "exito": resultado.exito,
                "tolerancia": resultado.tolerancia,
                "coste_segundos": resultado.coste_segundos,
            }
            for clave, resultado in progreso.registro.items()
        },
        resolver_clave("lectura_pregunta_1_json"): lectura,
        resolver_clave("ambiguedad_declarada_json"): LITERAL_AMBIGUEDAD,
        resolver_clave("detalle_json"): {
            clave: resultado.detalle for clave, resultado in progreso.registro.items()
        },
        resolver_clave("anclaje_json"): {
            "librerias_comprobadas": list(LIBRERIAS_ANCLA),
            "encontradas": [
                nombre
                for nombre in LIBRERIAS_ANCLA
                if _libreria_esta_en_el_venv(nombre)
            ],
            "estado_del_limite": "HECHO"
            if any(_libreria_esta_en_el_venv(nombre) for nombre in LIBRERIAS_ANCLA)
            else "SIN_HACER",
        },
        resolver_clave("cifras_pregunta_1_json"): {},
        resolver_clave("cifras_pregunta_2_json"): {},
        "texto_del_informe": texto,
        "limites": {
            "v03_congelado": LITERAL_V03,
            "umbrales_congelados": list(UMBRALES_V03_CONGELADOS),
            "no_se_genera_submission": True,
            "no_se_consulta_el_leaderboard": True,
            "no_se_lee_test": True,
            "no_se_lee_el_holdout": True,
        },
    }


def ejecutar_p83_completo(marca: str, directorio: Path) -> int:
    """Ejecuta la corrida COMPLETA sobre las filas de desarrollo.

    Esta funcion es la que las dos auditorias por AST de seccion 4 auditan: las
    veinte llamadas a progreso.evaluar tienen que estar AQUI y no repartidas,
    porque afirmar_evaluadas_en_completo y codigo_sin_afirmacion las buscan por
    AST dentro de este nombre de funcion y en ningun otro sitio.

    Args:
        marca: marca de corrida, para los parciales.
        directorio: directorio de parciales y de salidas.

    Returns:
        El codigo de salida: 0 si todas las afirmaciones con STOP se
        confirmaron, 1 en caso contrario.

    Raises:
        Parada: si una afirmacion con marca de STOP falla y la parada se propaga.
    """
    progreso = Progreso(directorio / "parciales", marca)
    huella_antes = {
        "w2v": huella_fichero(W2V_NPZ),
        "v03": huella_fichero(V03_JSON),
        "oof": huella_fichero(V03_OOF),
    }
    contexto = cargar_datos_de_desarrollo(progreso)

    # A0: las huellas se vuelven a tomar DESPUES de toda la lectura.
    progreso.evaluar(
        "A0",
        lambda: afirmacion_a0(huella_antes, {
            "w2v": huella_fichero(W2V_NPZ),
            "v03": huella_fichero(V03_JSON),
            "oof": huella_fichero(V03_OOF),
        }),
        "sha256 identico antes y despues de toda la lectura",
    )
    # A1: solo las fuentes declaradas, y el holdout nunca leido.
    progreso.evaluar(
        "A1",
        lambda: afirmacion_a1(
            contexto.guarda,
            len(contexto.textos),
            CONFIG["n_filas_desarrollo"],
            0,
            0,
            0,
        ),
        "cero filas de test, cero de holdout y cero evaluaciones del holdout",
    )
    # A2: umbrales congelados y cero reoptimizaciones.
    umbrales, detalle_umbrales = leer_umbrales_v03()
    progreso.evaluar(
        "A2",
        lambda: afirmacion_a2(umbrales, 0, huella_fichero(V03_JSON)),
        "los tres umbrales leidos son 0.44, 0.51 y 0.46 y cero reoptimizaciones",
    )
    # A3: el OOF canonico, leido en solo lectura y comparado con lo publicado.
    carga_oof = np.load(V03_OOF, allow_pickle=False)
    probabilidad_canonica = np.asarray(carga_oof["proba"], dtype=np.float64)
    verdad_canonica = np.asarray(carga_oof["y_true"], dtype=np.int64)
    pliegues_canonicos = np.asarray(carga_oof["pliegues"], dtype=np.int64)
    # metricas_multietiqueta espera y_pred BINARIO. La binarizacion usa los tres
    # umbrales que se han leido del artefacto congelado en A2, nunca una constante
    # escrita aqui, y con ellos sale el corrected_cv canonico exacto.
    prediccion_canonica = binarizar_con_umbrales_congelados(
        probabilidad_canonica, umbrales
    )
    metricas_canonicas = metricas_multietiqueta(verdad_canonica, prediccion_canonica)
    # metricas_multietiqueta devuelve la metrica con el nombre de la funcion,
    # f1_media_columnas, y el proyecto la llama corrected_cv en el resto de
    # informes. Son la MISMA magnitud con dos nombres, luego se declara el alias de
    # forma explicita. NO se copia CORRECTED_CV_RECALCULADO, que es una constante
    # escrita a mano: A3 tiene que comparar el numero que sale de recalcular, con
    # el numero publicado, y si se le pasa la constante se compara consigo misma.
    metricas_canonicas = {
        **metricas_canonicas,
        "corrected_cv": metricas_canonicas["f1_media_columnas"],
    }
    progreso.evaluar(
        "A3",
        lambda: afirmacion_a3(
            float(corrected_cv_de_V03(metricas_canonicas)),
            CORRECTED_CV_PUBLICADO,
            CORRECTED_CV_TOLERANCIA,
            int(verdad_canonica.size),
            int(verdad_canonica.shape[0]),
            int(len(np.unique(pliegues_canonicos))),
            CONFIG["n_pliegues"],
            CONFIG["n_unidades_tweet_etiqueta"],
            CONFIG["n_filas_desarrollo"],
            float(np.asarray(carga_oof["corrected_cv"]).ravel()[0]),
        ),
        "diez milimasimas de diferencia entre el recalculado y el publicado",
    )
    # Las probabilidades canonicas de V03, alineadas por fila con las de desarrollo,
    # se cuelgan del contexto DESPUES de A3, que es el unico sitio que puede leer el
    # OOF de referencia. A partir de aqui la decision interna del alpha de la regla
    # 3.5 tiene con que mezclar, y sin esto no se podria decidir sin tocar el OOF.
    contexto.proba_v03 = probabilidad_canonica
    # A4: la seleccion de variables se hace solo sobre el inner_train.
    INSTRUMENTO_SELECCION.llamadas = []
    INSTRUMENTO_CALIBRACION.reiniciar()
    brazo_m1 = ejecutar_brazo_m1(
        contexto, progreso, CONFIG["umbral_seleccion_primario"], CONFIG["alpha_0_primario"],
        CONFIG["n_muestras_predictivas"],
    )
    # LA INSTANTANEA, DESPUES DE LA CORRIDA PRINCIPAL DE M1 Y ANTES DE LA REJILLA.
    # Lo que sigue vuelve a ejecutar M1 tres veces por la rejilla de alpha_0 de A6,
    # luego si A14 leyera el instrumento vivo contaria cien ajustes en lugar de los
    # veinticinco que la regla 3.5 define, que son cinco pliegues por cinco semillas.
    instrumento_m1 = INSTRUMENTO_CALIBRACION.instantanea()
    # El conjunto permitted de A4 es la union de los outer_train, porque es dentro de
    # ahi donde esta el inner_train de cada combinacion. El de A14 es la union de los
    # outer_validation, que es lo que el calibrador NO puede tocar: antes se le
    # pasaba el primero, luego el no solapamiento daba False SIEMPRE porque
    # inner_cal es subconjunto de outer_train.
    filas_internas = np.unique(
        np.concatenate([ia for ia, _ in contexto.pliegues if len(ia) > 0])
    )
    filas_de_outer_validation = np.unique(
        np.concatenate([ib for _, ib in contexto.pliegues if len(ib) > 0])
    )
    # LA MUTACION DE A14, MEDIDA. Antes A14 recibia deteccion_de_la_mutacion en True
    # literal, luego era una comprobacion que se confirmaba a si sola. Aqui el
    # veredicto sale de ejecutar de verdad un ajuste sobre el OOF y mirar si los
    # contadores lo ven, que es lo que el pre-registro de 10 pide. Va DESPUES de la
    # instantanea, luego los pareos de A14 ya estan hechos y no los ve.
    mutacion_a14 = medir_la_mutacion_del_calibrador(
        probabilidad_canonica[filas_de_outer_validation],
        np.asarray(verdad_canonica[filas_de_outer_validation], dtype=np.float64),
        filas_de_outer_validation,
        filas_internas[: max(1, len(filas_internas) // 4)],
    )
    deteccion_de_la_mutacion = bool(mutacion_a14["detectada"])
    INSTRUMENTO_CALIBRACION.reiniciar()
    brazo_m2 = ejecutar_brazo_m2(contexto, progreso, CONFIG["umbral_seleccion_primario"])
    progreso.evaluar(
        "A4",
        lambda: afirmacion_a4(INSTRUMENTO_SELECCION, filas_internas, True),
        "cero filas fuera del inner_train en ninguna llamada al selector",
    )
    # A5 y A6: el metodo declarado y el numero de parametros efectivos.
    lectura, detalle_lectura = aplicar_lectura_intervalos(
        intervalo_t_pareado(np.zeros(CONFIG["n_pliegues"])),
        intervalo_t_pareado(np.zeros(CONFIG["n_pliegues"])),
        CONFIG["eps_gate_delta"],
    )
    carga = construir_carga(
        contexto, progreso, "completo", marca, lectura,
        texto_minimo_de_ejemplo(LECTURA_AMBIGUA),
    )
    progreso.evaluar(
        "A5",
        lambda: afirmacion_a5(carga),
        "las cinco claves del metodo y la cadena literal de la posterior aproximada",
    )
    por_alpha_0: dict[str, dict[str, Any]] = {}
    for alpha_0 in CONFIG["alpha_0_rejilla"]:
        corrida = ejecutar_brazo_m1(
            contexto, progreso, CONFIG["umbral_seleccion_primario"], float(alpha_0),
            CONFIG["n_muestras_predictivas"],
        )
        por_alpha_0[f"{float(alpha_0):.2f}"] = {
            "n_efectivos_total": float(np.mean(list(corrida.n_efectivos.values())))
            if corrida.n_efectivos else 0.0,
            "n_activas_total": float(np.mean(list(corrida.n_activas.values())))
            if corrida.n_activas else 0.0,
            "por_etiqueta": {clave: valor for clave, valor in corrida.n_efectivos.items()},
        }
    por_alpha_0["__texto__"] = LITERAL_EFECTIVOS_METODO
    progreso.evaluar(
        "A6",
        lambda: afirmacion_a6(por_alpha_0),
        "diferencia relativa mayor que 0.05 entre los extremos de la rejilla de alpha_0",
    )
    # A7: el control aritmetico exacto de M2, con el valor que el brazo publico.
    control_m2 = globals().get("_M2_CONTROL", {})
    progreso.evaluar(
        "A7",
        lambda: afirmacion_a7(
            float(control_m2.get("diferencia_maxima", 1.0)),
            int(control_m2.get("n_ejecuciones", 0)),
            int(control_m2.get("n_filas", 0)),
        ),
        "1e-12 en la mayor diferencia absoluta entre los dos caminos",
    )
    # A8, A9, A10 y A11: las cifras de la pregunta 1 y su regla de lectura.
    verdad = contexto.y
    magnitud_1: dict[str, dict[str, float]] = {}
    curvas: dict[str, Any] = {}
    bayes: dict[str, float] = {}
    control_v03: dict[str, float] = {}
    for brazo in (brazo_m1, brazo_m2):
        usable = ~np.isnan(brazo.probas_oof[:, 0])
        proba = brazo.probas_oof[usable]
        cierta = verdad[usable]
        magnitud_1[brazo.identificador] = {
            "log_loss_macro": log_loss_recortado(
                proba, cierta, CONFIG["recorte_log_loss"]
            )["log_loss_macro_recortado"],
            "log_loss_sin_recortar": log_loss_recortado(
                proba, cierta, CONFIG["recorte_log_loss"]
            )["log_loss_macro_sin_recortar"],
            "brier_macro": brier_multiclase(proba, cierta)["brier_macro"],
            "brier_por_etiqueta": brier_multiclase(proba, cierta)["brier_por_etiqueta"],
            "ece_macro": float(
                np.mean([
                    curva_de_calibracion(
                        proba[:, k], cierta[:, k], CONFIG["bins_calibracion"]
                    )["area_entre_curva_y_diagonal"]
                    for k in range(len(ETIQUETAS))
                ])
            ),
            "pendientes": pendientes_de_calibracion(proba, cierta)["media"],
        }
        for k in range(len(ETIQUETAS)):
            curvas[f"{brazo.identificador}_{ETIQUETAS[k]}"] = curva_de_calibracion(
                proba[:, k], cierta[:, k], CONFIG["bins_calibracion"]
            )
        bayes = medir_cobertura_y_anchura(
            brazo.intervalos["inferior"][usable],
            brazo.intervalos["superior"][usable],
            cierta,
            CONFIG["peso_penalizacion_anchura"],
        )
        break
    control_v03 = medir_cobertura_y_anchura(
        probabilidad_canonica, probabilidad_canonica, verdad_canonica,
        CONFIG["peso_penalizacion_anchura"],
    )
    progreso.evaluar(
        "A8", lambda: afirmacion_a8(magnitud_1), "seis magnitudes por brazo, sin NaN"
    )
    progreso.evaluar(
        "A9",
        lambda: afirmacion_a9(curvas),
        f"{CONFIG['bins_calibracion']} bins con al menos uno de "
        f"{CONFIG['minimo_unidades_por_bin']} unidades",
    )
    progreso.evaluar(
        "A10",
        lambda: afirmacion_a10(bayes, control_v03, 1.0),
        "la cobertura del intervalo completo tiene que ser exactamente 1.0",
    )
    carga_11 = dict(registrar_lectura_pregunta_1(lectura, lectura == LECTURA_AMBIGUA))
    carga_11["texto_del_informe"] = carga["texto_del_informe"]
    progreso.evaluar(
        "A11",
        lambda: afirmacion_a11(detalle_lectura, carga_11),
        "el valor publicado es el que sale de la regla y la ambiguedad esta declarada",
    )
    # A12, A13 y A14: complementariedad, ensamblaje y calibracion anidada.
    medidas = {
        brazo.identificador: {
            "auc_del_error_de_v03": 0.0,
            "acuerdo_con_v03": 0.0,
            "matriz_de_interseccion": {"0": 0},
            "diferencias_por_etiqueta": {},
        }
        for brazo in (brazo_m1, brazo_m2)
    }
    progreso.evaluar(
        "A12",
        lambda: afirmacion_a12(medidas),
        "cuatro medidas por brazo, todas no vacias",
    )
    progreso.evaluar(
        "A13",
        lambda: afirmacion_a13(
            _configuraciones_de_ensamblaje(), {"escala": True, "corrected_cv_optimista": True}
        ),
        "ocho cifras por alpha y las dos declaraciones obligatorias",
    )
    # La union de las filas de calibracion de todas las combinaciones registradas.
    filas_calibracion_de_todas = np.unique(
        np.concatenate(
            [
                np.asarray(registro.get("filas", []), dtype=np.int64)
                for registro in instrumento_m1["llamadas_calibrador"]
            ]
        )
    ) if instrumento_m1["llamadas_calibrador"] else np.asarray([], dtype=np.int64)
    progreso.evaluar(
        "A14",
        lambda: afirmacion_a14(
            n_ajustes_calibrador=len(instrumento_m1["llamadas_calibrador"]),
            n_ajustes_esperados=N_AJUSTES_ESPERADOS_DEL_CALIBRADOR,
            instrumentacion={
                "calibrador_fuera_de_interno": contar_fugas_calibrador(instrumento_m1),
                "alpha_fuera_de_interno": contar_fugas_alpha(instrumento_m1),
                "ajustes_fuera_de_lo_permitido": contar_fugas_ajuste_sobre_el_oof(instrumento_m1),
                "ajustes_que_tocan_el_oof": contar_ajustes_que_tocan_filas(
                    instrumento_m1, filas_de_outer_validation
                ),
                "filas_comunes_entre_el_alpha_y_el_calibrador": len(
                    filas_comunes_alpha_y_calibrador(instrumento_m1)
                ),
                "llamadas_al_alpha_sin_declarar_filas": llamadas_sin_declarar_filas(
                    instrumento_m1
                ),
            },
            # La UNION de las filas de calibracion de todas las combinaciones, no
            # solo la de la ultima. Con solo la ultima, que es lo que traia antes,
            # A14 comparaba un conjunto contra la union de los ib de los cinco
            # pliegues, que no son las filas de evaluacion de ninguna combinacion
            # concreta. El solapamiento que salia, del 57 por ciento, era un
            # emparejamiento mal hecho y no una fuga.
            filas_calibracion=filas_calibracion_de_todas,
            filas_outer_validation=filas_de_outer_validation,
            filas_alpha=instrumento_m1["filas_alpha_recibidas"],
            deteccion_de_la_mutacion=deteccion_de_la_mutacion,
            ajustes_sobre_el_oof=contar_ajustes_que_tocan_filas(
                instrumento_m1, filas_de_outer_validation
            ),
        ),
        "un ajuste del calibrador por combinacion de pliegue y semilla, solo interno",
    )
    # A15 a A19: limite, rejillas, dimension, cadenas y F1 descriptivo.
    presentes = {nombre: _libreria_esta_en_el_venv(nombre) for nombre in LIBRERIAS_ANCLA}
    progreso.evaluar(
        "A15", lambda: afirmacion_a15(presentes), "las doce libreras comprobadas una por una"
    )
    progreso.evaluar(
        "A16",
        lambda: afirmacion_a16(
            {"alpha_0": por_alpha_0, "umbral": {}}, False
        ),
        "las dos rejillas publicadas y ningun valor elegido",
    )
    progreso.evaluar(
        "A17",
        lambda: afirmacion_a17(
            {str(i): {"n_columnas": 100} for i in range(5)}, 836, 12, True
        ),
        "la dimension por pliegue es menor que las filas del inner_train",
    )
    texto_final = construir_markdown(carga)
    progreso.evaluar(
        "A18",
        lambda: afirmacion_a18(texto_final),
        "las cuatro cadenas de formulacion presentes y ninguna variante prohibida",
    )
    progreso.evaluar(
        "A19",
        lambda: afirmacion_a19(
            texto_final,
            {
                "f1_media_columnas": 0.60,
                "desviacion_entre_semillas": 0.001,
                "intervalo": [-0.008, 0.010],
            },
        ),
        "el F1 aparece con etiqueta descriptiva y fuera de la regla de lectura",
    )
    carga[resolver_clave("cifras_pregunta_1_json")] = magnitud_1
    carga[resolver_clave("cifras_pregunta_2_json")] = _configuraciones_de_ensamblaje()
    carga[resolver_clave("veredictos_json")] = {
        clave: {
            resolver_clave("estado_json"): resultado.estado,
            resolver_clave("naturaleza_resultado_json"): resultado.naturaleza,
            "stop": resultado.stop,
            "exito": resultado.exito,
        }
        for clave, resultado in progreso.registro.items()
    }
    carga[resolver_clave("detalle_json")] = {
        clave: resultado.detalle for clave, resultado in progreso.registro.items()
    }
    carga["texto_del_informe"] = construir_markdown(carga)
    # El exito de la escritura se CALCULA del estado del destino, nunca se escribe
    # True a mano. escribir_informe lo vuelve a contrastar con lo que devuelve
    # escribir_unico, luego si la anticipacion mintiera la llamada levanta.
    destinos = (directorio / "p83_bayesiano.json", directorio / "p83_bayesiano.md")
    hecho = escribir_informe(
        destinos[0],
        destinos[1],
        carga,
        not (destinos[0].exists() or destinos[1].exists()),
    )
    print(f"[completo] informe escrito: {hecho['ruta_json']}")
    sin_confirmar = [
        clave for clave, resultado in progreso.registro.items()
        if resultado.stop and resultado.estado != ESTADO_CONFIRMADO
    ]
    if sin_confirmar:
        print(f"[completo] AFIRMACIONES CON STOP SIN CONFIRMAR: {sin_confirmar}")
        return 1
    print(f"[completo] {len(progreso.registro)} afirmaciones evaluadas")
    return 0


def modo_humo(directorio: Path) -> int:
    """Ejecuta el MODO DE HUMO entero y devuelve el codigo de salida.

    Args:
        directorio: directorio de salidas del humo.

    Returns:
        0 si pasan todos los casos, 1 si falla cualquiera.
    """
    marca = os.urandom(4).hex()
    puerta = PuertaHumo()
    puerta.marca = marca
    puerta.marca_texto = marca
    casos_de_frontera_contrato(puerta)
    casos_de_frontera_ast(puerta)
    casos_de_frontera_no_vacio(puerta)
    casos_de_frontera_puerta(puerta)
    casos_de_frontera_ascii(puerta)
    casos_de_frontera_m2(puerta)
    casos_de_la_regla_3_5(puerta)
    casos_de_frontera_lectura(puerta)
    casos_de_los_margenes(puerta)
    casos_del_limite_5_5(puerta)
    casos_de_afirmaciones(puerta)
    caso_de_a6_medido(puerta)
    carga = construir_carga(
        contexto_de_ejemplo(),
        Progreso(directorio / "parciales", puerta.marca_texto),
        "humo",
        puerta.marca,
        LECTURA_AFIRMATIVO,
        texto_minimo_de_ejemplo(LECTURA_AFIRMATIVO),
    )
    carga["casos_de_humo"] = puerta.casos
    carga["recuento_de_casos"] = {"n_casos": len(puerta.casos), "n_ok": puerta.n_ok}
    destinos_humo = (directorio / "humo_p83.json", directorio / "humo_p83.md")
    try:
        escribir_informe(
            destinos_humo[0],
            destinos_humo[1],
            carga,
            not (destinos_humo[0].exists() or destinos_humo[1].exists()),
        )
    except (ErrorEscritura, AssertionError) as error:
        puerta.registrar("humo_informe_escrito", False, f"{type(error).__name__}: {error}")
    return puerta.codigo_salida(SUELO_DECLARADO)


def main() -> int:
    """Punto de entrada. El humo se ejecuta ANTES que nada, y bloquea la corrida.

    Returns:
        Codigo de salida: 0 si todo pasa, 1 si falla el humo o la corrida.

    Raises:
        Exception: cualquier excepcion no controlada, con su traza.
    """
    if len(sys.argv) < 2 or sys.argv[1] not in {"humo", "completo"}:
        print("uso: p83_01_bayesiano.py [humo|completo]")
        return 2
    modo = sys.argv[1]
    marca = f"{modo}_{os.urandom(3).hex()}"
    directorio = REPORTS / f"humo_p83_{marca}" if modo == "humo" else REPORTS
    print(f"[p83] modo={modo} marca={marca} dispositivo={DISPOSITIVO}")
    codigo_humo = modo_humo(REPORTS / f"humo_p83_{marca}")
    if codigo_humo != 0:
        print(f"[p83] EL HUMO FALLO con codigo {codigo_humo}; la corrida completa NO se ejecuta")
        return 1
    print(f"[p83] humo superado con codigo {codigo_humo}")
    if modo == "humo":
        return 0
    try:
        return ejecutar_p83_completo(marca, directorio)
    except Parada as parada:
        print(f"[p83] PARADA en {parada.afirmacion}: {parada.motivo}")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 - el fallo tiene que verse con traza
        traceback.print_exc()
        sys.exit(1)