"""v32_01_surrogate_macrof1.py
Fecha: 2026-09-30
Descripcion: implementacion exacta del pre-registro congelado V32-E1: challenger MLP
279->128->6 sobre la matriz 279 con perdida soft-macro-F1 (brazo A1), gemelo con
cross-entropy media (A2) y control negativo con etiquetas barajadas del fold-train (A3);
early stopping estricto sobre el split interno 90/10 (el fold-val jamas se usa para parar);
entrenamiento en GPU bajo el guardian D-007; integracion con el mecanismo P6/P7 congelado,
gates y arbol de decision del pre-registro.
Estado del arte aplicado: perdida surrogate diferenciable de macro-F1 por lote, AdamW con
weight decay desacoplado, early stopping estricto con restauracion de los mejores pesos,
determinismo de PyTorch (use_deterministic_algorithms, generadores por brazo/fold/epoca,
fork_rng por fold) y precision completa (sin AMP) porque el protocolo congela la numerica.

Fuentes: reports/v32_e1_preregistro.md y experiments/v32_00_preregistro.json.
"""

from __future__ import annotations

# ======================================================================================
# ---- 0. ENTORNO E IMPORTS (las variables de hilos van ANTES de importar torch) ----
# ======================================================================================
import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
#: Determinismo de GEMM (torch exige CUBLAS_WORKSPACE_CONFIG con use_deterministic_algorithms
#: en CUDA; en ROCm el equivalente es HIPBLAS_WORKSPACE_CONFIG; valor valido en ambos casos).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HIPBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg
from src.gpu import GpuBusyError, log_event, require_gpu
from src.verificacion_numerica import autotest_numerico

# ======================================================================================
# ---- 1. CONSTANTES CONGELADAS Y VERIFICACION DEL PRE-REGISTRO ----
# ======================================================================================
FECHA = "2026-09-30"
N_CLASES = 6
N_FILAS = 169336
N_FOLDS = 5
#: Hiperparametros congelados (experiments/v32_00_preregistro.json). No se cambian.
LOTE = 8192
LR = 1e-3
WD = 1e-4
EPOCAS_MAX = 60
PACIENCIA = 8
FRACCION_INTERNA = 0.10
GRAD_CLIP = 5.0
EPS_SOFT = 1e-6
SEMILLA = 20261010
SEMILLA_BARAJADO = 20261011
HILOS_CPU = 8
N_BOOTSTRAP = 2000
GATE_DELTA = 0.0015
GATE_FOLDS = 4
UMBRAL_ALTA_CONFIANZA = 0.60
PESOS = [1.0, 1.1749, 1.2721, 1.163, 1.001, 0.8128]
#: Referencias congeladas del campeon (V30 + plug-in) y tolerancias.
BASELINE = 0.6159732
NESTED = 0.6154643
VARIABILIDAD = 0.0008876
ABORTO_BASELINE = 5e-7
#: Sanity de 1 lote y guardas NaN.
SANITY_PASOS = 100
SANITY_REDUCCION = 0.30
NAN_MAX_EPOCA = 3
NAN_MAX_BRAZO = 10
#: Mecanismo de integracion P6/P7 congelado.
L2_CLIP = 1e-12
C_LOGREG = 1.0
MAX_ITER_LOGREG = 2000
CHUNKS_PRED = 16384
BRAZOS = ("A1_surrogate", "A2_ce", "A3_barajado")
OOF_ARCHIVO = {"A1_surrogate": "surrogate", "A2_ce": "ce", "A3_barajado": "barajado"}

FORMULA_SOFT = ("p=softmax(z) clamp [1e-6,1-1e-6]; TP_c=sum(p*y); FP_c=sum(p*(1-y)); "
                "FN_c=sum((1-p)*y); P=TP/(TP+FP+1e-6); R=TP/(TP+FN+1e-6); "
                "F1=2PR/(P+R+1e-6); L=1-media_c F1_c")
GUARDAS_NAN = (f"loss no finita: salta el paso; >={NAN_MAX_EPOCA} en una epoca o "
               f">={NAN_MAX_BRAZO} acumuladas: aborta el brazo")
CONTROL_GATE = "delta_int(A1) > delta_int(A2) (comparacion numerica pre-registrada)"
CONTROL_NEGATIVO = ("si delta_int(A3) >= delta_int(A1): NO_ATRIBUIBLE (no puede pasar a "
                    "segunda particion)")
MECANISMO = ("combinador logistico anidado sobre [L2(v30_prob) || L2(p_brazo)] (12 features, "
             f"L2 por fila con clip {L2_CLIP}); LogReg(C={C_LOGREG}, max_iter={MAX_ITER_LOGREG}, "
             "lbfgs) por fold (fit fold!=k, predice fold==k)")
DECISION = "argmax(p_comb * w_plugin CONGELADO)"
FOLD_VAL_TEXTO = "jamas se usa para parar (solo OOF)"
RESTAURAR_TEXTO = "mejor macro-F1 interna (hard argmax en el split interno)"
ARBOL = {
    "PROMISING": "pasa los gates -> segunda particion congelada (semilla 987654) antes de "
                 "cualquier otra cosa",
    "MARGINAL": "delta > 0 sin gates -> no se adopta; cierre documentado o P2 con justificacion",
    "CLOSED": "delta <= 0 -> se cierra P1; solo entonces se evalua P2 (CatBoost)",
    "NO_ATRIBUIBLE": "no puede pasar a segunda particion (control negativo barajado)",
}
ARBOL_SUBMISSION = ("jamas automatica; requeriria gate completo + segunda particion + "
                    "autorizacion explicita")

CANONICO = ROOT / "artifacts" / "p33_scores" / "todos"
PREREG_JSON = cfg.EXPERIMENTS / "v32_00_preregistro.json"
MANIFEST = cfg.REPORTS / "age_submission_plugin_manifest.json"
RESULTADOS_JSON = cfg.EXPERIMENTS / "v32_01_surrogate_macrof1.json"
CURVAS_JSON = cfg.EXPERIMENTS / "v32_e1_curvas.json"
FIGURA_CURVAS = cfg.FIGURES / "v32_e1_curvas.png"
INFORME = cfg.REPORTS / "v32_e1_surrogate_macrof1.md"
RUNLOG = ROOT / "RUNLOG.md"
EXPERIMENTS_CSV = ROOT / "experiments.csv"
ARCHIVO_PRE_REGISTRO = "reports/v32_e1_preregistro.md"
COMMIT_PRE_REGISTRO = "9c5cca2"
RUTAS_HASHES = {
    "canonico_y_true": CANONICO / "y_true.npy",
    "canonico_fold_idx": CANONICO / "fold_idx.npy",
    "canonico_v30_prob": CANONICO / "v30_prob.npy",
    "manifest_plugin": MANIFEST,
    "fuente_v19_b1_boosting.py": ROOT / "scripts" / "v19_b1_boosting.py",
    "fuente_v30_02_ensemble.py": ROOT / "scripts" / "v30_02_ensemble.py",
    "fuente_age_umbrales_por_clase.py": ROOT / "scripts" / "age_umbrales_por_clase.py",
}


class Detencion(RuntimeError):
    """Detencion pre-registrada: el mensaje se imprime y el proceso sale con ``codigo``."""

    def __init__(self, mensaje: str, codigo: int) -> None:
        super().__init__(mensaje)
        self.codigo = codigo


class AbortarBrazo(Detencion):
    """Aborta el brazo por guardas NaN (codigo de salida 5)."""

    def __init__(self, mensaje: str) -> None:
        super().__init__(mensaje, 5)


def sha256_bytes(datos: bytes) -> str:
    """SHA256 hexadecimal de un bloque de bytes."""
    return hashlib.sha256(datos).hexdigest()


def sha256_archivo(ruta: Path) -> str:
    """SHA256 hexadecimal de un archivo."""
    return sha256_bytes(ruta.read_bytes())


def ruta_relativa(ruta: Path) -> str:
    """Ruta de artefacto relativa al proyecto, para informes y CSV."""
    try:
        return str(ruta.relative_to(ROOT))
    except ValueError:
        return str(ruta)


def cargar_preregistro() -> dict:
    """Lee el JSON congelado del pre-registro.

    Returns
    -------
    dict
        Contenido completo de ``experiments/v32_00_preregistro.json``.

    Raises
    ------
    FileNotFoundError
        Si el pre-registro no existe.
    """
    if not PREREG_JSON.exists():
        raise FileNotFoundError(f"no existe el pre-registro {PREREG_JSON}")
    return json.loads(PREREG_JSON.read_text(encoding="utf-8"))


def verificar_constantes(prereg: dict) -> tuple[bool, dict]:
    """Compara cada constante local contra el JSON congelado, una por una.

    Parameters
    ----------
    prereg:
        Contenido del pre-registro.

    Returns
    -------
    tuple[bool, dict]
        Exito global y diccionario nombre -> booleano con el detalle de cada chequeo.
    """
    entrenamiento = prereg["entrenamiento"]
    determinismo = entrenamiento["determinismo"]
    parada = entrenamiento["early_stopping"]
    perdida = prereg["perdida_soft_macro_f1"]
    gates = prereg["gates"]
    integracion = prereg["integracion"]
    compliance = prereg["compliance"]
    checks = {
        "lote": entrenamiento["lote"] == LOTE,
        "optimizador": entrenamiento["optimizador"] == (
            f"AdamW lr={LR} betas=(0.9,0.999) eps=1e-8 wd={WD}"),
        "epocas_max": entrenamiento["epocas_max"] == EPOCAS_MAX,
        "paciencia": parada["paciencia"] == PACIENCIA,
        "split_interno": parada["split_interno"] == (
            f"{1 - FRACCION_INTERNA:.0%}/10% estratificado por clase del fold-train; "
            f"semilla = {SEMILLA} + fold"),
        "restaurar": parada["restaurar"] == RESTAURAR_TEXTO,
        "fold_val_nunca_para": parada["fold_val"] == FOLD_VAL_TEXTO,
        "grad_clip": entrenamiento["grad_clip"] == GRAD_CLIP,
        "guardas_nan": entrenamiento["guardas_nan"] == GUARDAS_NAN,
        "sanity_1_lote": entrenamiento["sanity_1_lote"] == (
            f"100 pasos sobre un lote de {LOTE}: PASS si loss finita y reduccion >= "
            f"{SANITY_REDUCCION:.2f}"),
        "seeds": determinismo["seeds"] == {"global": SEMILLA,
                                           "barajado_etiquetas": SEMILLA_BARAJADO},
        "shuffle_lote": determinismo["shuffle_lote"] == (
            f"torch.Generator por (brazo, fold, epoca); seed = {SEMILLA} + 10**6*brazo + "
            "1000*fold + epoca"),
        "hilos_cpu": determinismo["hilos_cpu"] == HILOS_CPU,
        "matmul_precision": prereg["dispositivo"]["matmul_precision"] == "highest",
        "eps_soft": perdida["eps"] == EPS_SOFT,
        "objetivos": perdida["objetivos"] == "one-hot float32",
        "formula": perdida["formula"] == FORMULA_SOFT,
        "pesos_plugin": prereg["pesos_plugin"] == PESOS,
        "baseline_primario": prereg["baseline_primario"] == BASELINE,
        "referencia_nested": prereg["referencia_nested"] == NESTED,
        "umbral_alta_confianza": prereg["umbral_alta_confianza"] == UMBRAL_ALTA_CONFIANZA,
        "variabilidad": prereg["variabilidad_entrenamiento"] == VARIABILIDAD,
        "brazos": sorted(prereg["brazos"]) == sorted(BRAZOS),
        "gates_delta_min": gates["delta_min"] == GATE_DELTA,
        "gates_folds_min": gates["folds_min"] == GATE_FOLDS,
        "gates_aborto_baseline": gates["baseline_abort"] == ABORTO_BASELINE,
        "gates_control": gates["control"] == CONTROL_GATE,
        "gates_control_negativo": gates["control_negativo"] == CONTROL_NEGATIVO,
        "gates_bootstrap": gates["bootstrap"] == {"n": N_BOOTSTRAP, "semilla": SEMILLA,
                                                  "ic95_sin_cero": True},
        "gates_sin_leakage": gates["sin_leakage"] is True,
        "integracion_mecanismo": integracion["mecanismo"] == MECANISMO,
        "integracion_decision": integracion["decision"] == DECISION,
        "arbol_promising": prereg["arbol"]["PROMISING"] == ARBOL["PROMISING"],
        "arbol_marginal": prereg["arbol"]["MARGINAL"] == ARBOL["MARGINAL"],
        "arbol_closed": prereg["arbol"]["CLOSED"] == ARBOL["CLOSED"],
        "arbol_submission": prereg["arbol"]["submission"] == ARBOL_SUBMISSION,
        "compliance_embeddings_0": compliance["embeddings"] == 0,
        "compliance_transformers_0": compliance["transformers"] == 0,
        "compliance_llm_0": compliance["LLM"] == 0,
        "compliance_test_0": compliance["test"] == 0,
        "compliance_holdout_0": compliance["holdout"] == 0,
        "compliance_leaderboard_0": compliance["leaderboard"] == 0,
        "compliance_submissions_0": compliance["submissions"] == 0,
        "compliance_datos_externos_0": compliance["datos_externos"] == 0,
    }
    fallos = [nombre for nombre, correcto in checks.items() if not correcto]
    for nombre in fallos:
        print(f"  [STOP] constante distinta a la congelada: {nombre}", flush=True)
    return (not fallos), checks


def verificar_hashes(prereg: dict) -> tuple[bool, dict]:
    """Verifica el SHA256 de cada artefacto leido contra el JSON congelado.

    Returns
    -------
    tuple[bool, dict]
        Exito global y detalle nombre -> booleano del hash.
    """
    checks = {}
    for nombre, ruta in RUTAS_HASHES.items():
        if not ruta.exists():
            print(f"  [STOP] falta el artefacto {ruta}", flush=True)
            checks[nombre] = False
            continue
        checks[nombre] = sha256_archivo(ruta) == prereg["hashes"][nombre]
        if not checks[nombre]:
            print(f"  [STOP] hash distinto en {nombre}", flush=True)
    return all(checks.values()), checks


# ======================================================================================
# ---- 2. CARGA DE ARTEFACTOS CANONICOS Y BASELINE ----
# ======================================================================================
def normaliza(p: np.ndarray) -> np.ndarray:
    """Clip [1e-12, 1] y renormalizacion L1 por fila (convencion del proyecto)."""
    p = np.clip(p, 1e-12, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def normaliza_l2(p: np.ndarray) -> np.ndarray:
    """Clip [1e-12, 1] y normalizacion L2 por fila (mecanismo P6/P7 congelado)."""
    p = np.clip(p, L2_CLIP, 1.0)
    norma = np.linalg.norm(p, axis=1, keepdims=True)
    return p / np.where(norma > L2_CLIP, norma, 1.0)


def macro(clase: np.ndarray, pred: np.ndarray) -> float:
    """macro-F1 de sklearn con las 6 clases canonicas declaradas."""
    return float(f1_score(clase, pred, average="macro", labels=list(range(N_CLASES))))


def macro_de_conteos(conteos: np.ndarray) -> float:
    """macro-F1 exacto a partir del histograma aplanado de la matriz de confusion 6x6."""
    matriz = conteos.reshape(N_CLASES, N_CLASES)
    f1s = []
    for c in range(N_CLASES):
        tp = float(matriz[c, c])
        fp = float(matriz[:, c].sum() - tp)
        fn = float(matriz[c, :].sum() - tp)
        denominador = 2.0 * tp + fp + fn
        f1s.append(2.0 * tp / denominador if denominador > 0 else 0.0)
    return float(np.mean(f1s))


def cargar_canonico() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Carga y valida ``y_true``, ``fold_idx`` y ``v30_prob`` de la particion canonica.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Etiquetas int64, folds int64 y probabilidades del campeon normalizadas (float64).

    Raises
    ------
    AssertionError
        Si los artefactos no tienen la alineacion congelada (169336 filas y 5 folds).
    """
    clase = np.load(CANONICO / "y_true.npy")
    fold_idx = np.load(CANONICO / "fold_idx.npy")
    v30 = np.load(CANONICO / "v30_prob.npy").astype(np.float64)
    assert clase.shape == (N_FILAS,), f"y_true con forma inesperada {clase.shape}"
    assert fold_idx.shape == (N_FILAS,), f"fold_idx con forma inesperada {fold_idx.shape}"
    assert v30.shape == (N_FILAS, N_CLASES), f"v30_prob con forma inesperada {v30.shape}"
    assert int(clase.min()) == 0 and int(clase.max()) == N_CLASES - 1, "clases fuera de rango"
    assert np.unique(fold_idx).size == N_FOLDS, "se esperan 5 folds canonicos"
    return clase.astype(np.int64), fold_idx.astype(np.int64), normaliza(v30)


def cargar_matriz(prereg: dict) -> np.ndarray:
    """Carga ``matriz_279`` de ``scripts/v19_b1_boosting.py`` y verifica su hash congelado.

    Raises
    ------
    Detencion
        Si el SHA256 de los bytes difiere del pre-registro (codigo 2).
    """
    from v19_b1_boosting import matriz_279

    matriz = matriz_279()
    sha = sha256_bytes(np.ascontiguousarray(matriz).tobytes())
    if sha != prereg["hashes"]["matriz_279_bytes"]:
        raise Detencion("[STOP] hash de matriz_279 distinto al congelado (exit 2)", 2)
    assert matriz.shape == (N_FILAS, 279), f"matriz_279 con forma inesperada {matriz.shape}"
    assert matriz.dtype == np.float32, f"matriz_279 deberia ser float32, es {matriz.dtype}"
    assert np.isfinite(matriz).all(), "matriz_279 contiene valores no finitos"
    return matriz


# ======================================================================================
# ---- 3. DETERMINISMO Y SANITY DE 1 LOTE (GPU) ----
# ======================================================================================
def configurar_determinismo() -> dict:
    """Fija semillas, precision y modo determinista; devuelve el estado efectivo.

    Returns
    -------
    dict
        ``{"modo": "estricto"|"warn_only", "eventos": [...]}``.
    """
    random.seed(SEMILLA)
    np.random.seed(SEMILLA)
    torch.manual_seed(SEMILLA)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEMILLA)
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(HILOS_CPU)
    estado = {"modo": "estricto", "eventos": []}
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as error:  # noqa: BLE001 - ROCm puede no ofrecer el modo estricto
        torch.use_deterministic_algorithms(True, warn_only=True)
        estado["modo"] = "warn_only"
        estado["eventos"].append(f"al fijar estricto: {type(error).__name__}: {error}")
    return estado


def degradar_determinismo(estado: dict, motivo: str) -> None:
    """Baja a ``warn_only=True`` y registra el motivo en el estado."""
    torch.use_deterministic_algorithms(True, warn_only=True)
    estado["modo"] = "warn_only"
    estado["eventos"].append(motivo)


def es_error_de_determinismo(error: Exception) -> bool:
    """True si el error es el RuntimeError de determinismo estricto de PyTorch/ROCm."""
    return isinstance(error, RuntimeError) and "deterministic" in str(error).lower()


def degradar_por_determinismo(estado: dict, error: Exception, fase: str) -> bool:
    """Degrada a ``warn_only`` si el error es de determinismo estricto; registra la fase.

    Parameters
    ----------
    estado:
        Estado del determinismo (``modo``/``eventos``).
    error:
        Excepcion capturada.
    fase:
        Donde ocurrio el fallo (``sanity`` o ``entrenamiento``).

    Returns
    -------
    bool
        True si degradó; False si el error no es de determinismo (el llamador lo propaga).
    """
    if estado["modo"] != "estricto" or not es_error_de_determinismo(error):
        return False
    degradar_determinismo(estado, f"{fase}: {type(error).__name__}: {error}")
    print(f"  [aviso] determinismo degradado a warn_only durante {fase}", flush=True)
    return True


def perdida_soft_macro_f1(logits: torch.Tensor, indice: torch.Tensor) -> torch.Tensor:
    """Perdida soft-macro-F1 EXACTA del pre-registro (sobre el lote).

    Parameters
    ----------
    logits:
        Salida lineal del modelo, forma ``(n, 6)``.
    indice:
        Etiquetas int64 del lote, forma ``(n,)``.

    Returns
    -------
    torch.Tensor
        Escalar ``1 - media_c F1_c`` con la formula congelada.
    """
    p = torch.softmax(logits, dim=1).clamp(EPS_SOFT, 1.0 - EPS_SOFT)
    y = F.one_hot(indice, N_CLASES).float()
    tp = (p * y).sum(dim=0)
    fp = (p * (1.0 - y)).sum(dim=0)
    fn = ((1.0 - p) * y).sum(dim=0)
    precision = tp / (tp + fp + EPS_SOFT)
    recall = tp / (tp + fn + EPS_SOFT)
    f1 = 2.0 * precision * recall / (precision + recall + EPS_SOFT)
    return 1.0 - f1.mean()


def macro_f1_hard(pred: torch.Tensor, verdad: torch.Tensor) -> float:
    """macro-F1 (argmax duro) sobre tensores; conteos exactos via one-hot (sin bincount CUDA)."""
    ojo_verdad = F.one_hot(verdad, N_CLASES).float()
    ojo_pred = F.one_hot(pred, N_CLASES).float()
    matriz = ojo_verdad.t() @ ojo_pred
    tp = matriz.diagonal()
    fp = matriz.sum(dim=0) - tp
    fn = matriz.sum(dim=1) - tp
    denominador = (2.0 * tp + fp + fn).clamp(min=1e-12)
    f1 = torch.where(2.0 * tp + fp + fn > 0, 2.0 * tp / denominador, torch.zeros_like(tp))
    return float(f1.mean())


def sanity_un_lote(matriz: np.ndarray, clase: np.ndarray, fold_idx: np.ndarray,
                   device: torch.device) -> dict:
    """Sanity de 1 lote (100 pasos AdamW sobre 8192 filas del fold 0).

    Parameters
    ----------
    matriz:
        Matriz de 279 features (169336 x 279).
    clase:
        Etiquetas canonicas int64.
    fold_idx:
        Fold canonico por fila.
    device:
        Dispositivo de entrenamiento (cuda).

    Returns
    -------
    dict
        Loss inicial/final, reduccion relativa y ``pass`` (finita y reduccion >=
        ``SANITY_REDUCCION``).

    Raises
    ------
    Detencion
        Si la loss deja de ser finita (codigo 4).
    """
    entrenamiento = np.flatnonzero(fold_idx == 0)
    tr_idx = entrenamiento[:LOTE]
    escalador = StandardScaler().fit(matriz[entrenamiento])
    x = escalador.transform(matriz[tr_idx]).astype(np.float32)
    x_gpu = torch.from_numpy(x).to(device)
    y_gpu = torch.from_numpy(clase[tr_idx].astype(np.int64)).to(device)
    torch.manual_seed(SEMILLA)
    modelo = nn.Sequential(nn.Linear(279, 128), nn.GELU(), nn.Dropout(0.1),
                           nn.Linear(128, N_CLASES)).to(device)
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=LR, betas=(0.9, 0.999),
                                    eps=1e-8, weight_decay=WD)
    modelo.train()
    perdida_inicial, perdida_final = None, None
    for _ in range(SANITY_PASOS):
        optimizador.zero_grad(set_to_none=True)
        perdida = perdida_soft_macro_f1(modelo(x_gpu), y_gpu)
        valor = float(perdida.detach())
        if not math.isfinite(valor):
            raise Detencion("[STOP] sanity de 1 lote: loss no finita (exit 4)", 4)
        if perdida_inicial is None:
            perdida_inicial = valor
        perdida.backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), GRAD_CLIP)
        optimizador.step()
        perdida_final = valor
    reduccion = (perdida_inicial - perdida_final) / perdida_inicial
    correcto = math.isfinite(perdida_inicial) and math.isfinite(perdida_final) \
        and reduccion >= SANITY_REDUCCION
    return {"lote": LOTE, "pasos": SANITY_PASOS, "loss_inicial": perdida_inicial,
            "loss_final": perdida_final, "reduccion": reduccion, "pass": bool(correcto)}


def entrenar_sanity_con_degradacion(matriz: np.ndarray, clase: np.ndarray,
                                    fold_idx: np.ndarray, device: torch.device,
                                    estado_det: dict) -> dict:
    """Ejecuta el sanity; si el modo estricto revienta por determinismo, degrada y reintenta."""
    try:
        return sanity_un_lote(matriz, clase, fold_idx, device)
    except Detencion:
        raise
    except Exception as error:
        if not degradar_por_determinismo(estado_det, error, "sanity"):
            raise
        return sanity_un_lote(matriz, clase, fold_idx, device)


# ======================================================================================
# ---- 4. ENTRENAMIENTO DE LOS 3 BRAZOS x 5 FOLDS (GPU, GUARDIAN D-007) ----
# ======================================================================================
def construir_modelo(entrada: int, brazo_idx: int, fold: int, device: torch.device) -> nn.Module:
    """Crea el MLP congelado; la init usa la semilla por (brazo, fold) del pre-registro.

    Parameters
    ----------
    entrada:
        Numero de features de entrada (279).
    brazo_idx:
        Indice del brazo (0, 1, 2).
    fold:
        Fold canonico (0..4).
    device:
        Dispositivo destino.

    Returns
    -------
    torch.nn.Module
        ``Linear(279,128) -> GELU -> Dropout(0.1) -> Linear(128,6)``.
    """
    torch.manual_seed(SEMILLA + 10 ** 6 * brazo_idx + 1000 * fold)
    modelo = nn.Sequential(nn.Linear(entrada, 128), nn.GELU(), nn.Dropout(0.1),
                           nn.Linear(128, N_CLASES))
    return modelo.to(device)


def etiquetas_objetivo(brazo: str, clase: np.ndarray, fold_idx: np.ndarray,
                       fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Indices del fold-train y fold-val, y etiquetas objetivo del brazo.

    A3 permuta las etiquetas del fold-train con ``default_rng(20261011 + fold)`` antes del
    split interno; el fold-val conserva las etiquetas reales (solo OOF).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        ``tr_idx``, ``va_idx`` y ``y_tr`` (etiquetas objetivo del fold-train).
    """
    tr_idx = np.flatnonzero(fold_idx != fold)
    va_idx = np.flatnonzero(fold_idx == fold)
    y_tr = clase[tr_idx].astype(np.int64)
    if brazo == "A3_barajado":
        generador = np.random.default_rng(SEMILLA_BARAJADO + fold)
        y_tr = y_tr[generador.permutation(len(tr_idx))]
    return tr_idx, va_idx, y_tr


def split_interno(y_objetivo: np.ndarray, fold: int) -> tuple[np.ndarray, np.ndarray]:
    """Split interno 90/10 estratificado por las etiquetas que entrenan el brazo.

    Parameters
    ----------
    y_objetivo:
        Etiquetas objetivo del fold-train (permutadas en A3).
    fold:
        Fold canonico, para la semilla congelada ``SEMILLA + fold``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Posiciones relativas del sub-train y del sub-val interno.

    Raises
    ------
    AssertionError
        Si el split no cubre el fold-train o los bloques se solapan.
    """
    divisor = StratifiedShuffleSplit(n_splits=1, test_size=FRACCION_INTERNA,
                                     random_state=SEMILLA + fold)
    dummy = np.zeros((len(y_objetivo), 1), dtype=np.float32)
    sub_tr, sub_va = next(divisor.split(dummy, y_objetivo))
    assert sub_tr.size + sub_va.size == len(y_objetivo), "el split interno no cubre el fold-train"
    assert np.intersect1d(sub_tr, sub_va).size == 0, "sub-train y sub-val se solapan"
    assert sub_tr.min() >= 0 and sub_va.max() < len(y_objetivo), "indices internos fuera de rango"
    return sub_tr, sub_va


@torch.no_grad()
def predecir_probabilidad(modelo: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Probabilidades softmax sin gradiente, por chunks de ``CHUNKS_PRED``.

    Parameters
    ----------
    modelo:
        Red en modo evaluacion.
    x:
        Entradas GPU, forma ``(n, 279)``.

    Returns
    -------
    torch.Tensor
        Probabilidades ``(n, 6)`` en el mismo dispositivo.
    """
    modelo.eval()
    salidas = []
    for inicio in range(0, x.shape[0], CHUNKS_PRED):
        salidas.append(torch.softmax(modelo(x[inicio:inicio + CHUNKS_PRED]), dim=1))
    return torch.cat(salidas, dim=0)


def f1_de_tensores(modelo: nn.Module, x: torch.Tensor, y_idx: torch.Tensor) -> float:
    """macro-F1 (hard argmax) del modelo sobre un tensor GPU de entrada.

    Parameters
    ----------
    modelo:
        Red a evaluar.
    x:
        Entradas GPU del bloque.
    y_idx:
        Etiquetas int64 del bloque.

    Returns
    -------
    float
        macro-F1 de las 6 clases (convencion sklearn: clase sin soporte vale 0).
    """
    modelo.eval()
    predicciones = []
    with torch.no_grad():
        for inicio in range(0, x.shape[0], LOTE):
            predicciones.append(modelo(x[inicio:inicio + LOTE]).argmax(dim=1))
    return macro_f1_hard(torch.cat(predicciones), y_idx)


def entrenar_fold(brazo: str, brazo_idx: int, fold: int, matriz: np.ndarray,
                  clase: np.ndarray, fold_idx: np.ndarray, device: torch.device) -> dict:
    """Entrena un (brazo, fold) con ES estricto y devuelve curvas, pesos y OOF del fold.

    El fold-val solo se usa para el reporte por epoca y para el OOF final; jamas para
    detener el entrenamiento ni para elegir pesos.

    Returns
    -------
    dict
        ``oof_fold`` (n_val, 6) float64, curvas por epoca (loss, f1_subtrain, f1_interna,
        f1_val, norma_grad, fraccion_recortada, nan_por_epoca), mejor epoca/F1, epocas,
        pasos por epoca, NaN y bandera de leakage del split interno.
    """
    tr_idx, va_idx, y_tr = etiquetas_objetivo(brazo, clase, fold_idx, fold)
    assert np.intersect1d(tr_idx, va_idx).size == 0, "train y val del fold se solapan"
    sub_tr, sub_va = split_interno(y_tr, fold)
    escalador = StandardScaler().fit(matriz[tr_idx])
    x_tr = escalador.transform(matriz[tr_idx]).astype(np.float32)
    x_va = escalador.transform(matriz[va_idx]).astype(np.float32)
    assert np.isfinite(x_tr).all() and np.isfinite(x_va).all(), "escalado con no finitos"
    # Tensores GPU residentes del fold: train completo, sub-train/sub-val internos y fold-val.
    x_ft = torch.from_numpy(x_tr).to(device)
    y_ft = torch.from_numpy(y_tr).to(device)
    idx_sub_tr = torch.from_numpy(sub_tr.astype(np.int64)).to(device)
    idx_sub_va = torch.from_numpy(sub_va.astype(np.int64)).to(device)
    x_sub_tr, y_sub_tr = x_ft[idx_sub_tr], y_ft[idx_sub_tr]
    x_sub_va, y_sub_va = x_ft[idx_sub_va], y_ft[idx_sub_va]
    x_va_gpu = torch.from_numpy(x_va).to(device)
    y_va_real = torch.from_numpy(clase[va_idx].astype(np.int64)).to(device)

    semilla_fold = SEMILLA + 10 ** 6 * brazo_idx + 1000 * fold
    n_sub = int(x_sub_tr.shape[0])
    pasos_por_epoca = math.ceil(n_sub / LOTE)
    dispositivos = [0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=dispositivos):
        torch.manual_seed(semilla_fold)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(semilla_fold)
        modelo = construir_modelo(279, brazo_idx, fold, device)
        optimizador = torch.optim.AdamW(modelo.parameters(), lr=LR, betas=(0.9, 0.999),
                                        eps=1e-8, weight_decay=WD)
        mejor_epoca, mejor_f1, mejor_estado = -1, -1.0, None
        sin_mejora = 0
        nan_total = 0
        curvas = {"loss": [], "f1_subtrain": [], "f1_interna": [], "f1_val": [],
                  "norma_grad": [], "fraccion_recortada": [], "nan_por_epoca": []}
        for epoca in range(EPOCAS_MAX):
            generador = torch.Generator().manual_seed(semilla_fold + epoca)
            permutacion = torch.randperm(n_sub, generator=generador).to(device)
            modelo.train()
            perdidas, normas, fracciones = [], [], []
            nan_epoca = 0
            for paso in range(pasos_por_epoca):
                lote_idx = permutacion[paso * LOTE:(paso + 1) * LOTE]
                optimizador.zero_grad(set_to_none=True)
                logits = modelo(x_sub_tr[lote_idx])
                objetivo = y_sub_tr[lote_idx]
                if brazo == "A2_ce":
                    perdida = F.cross_entropy(logits, objetivo)
                else:
                    perdida = perdida_soft_macro_f1(logits, objetivo)
                if not torch.isfinite(perdida):
                    nan_epoca += 1
                    nan_total += 1
                    if nan_epoca >= NAN_MAX_EPOCA or nan_total >= NAN_MAX_BRAZO:
                        raise AbortarBrazo(
                            f"[STOP] guardas NaN: {brazo} fold {fold} epoca {epoca} "
                            f"(epoca={nan_epoca}, total={nan_total}) (exit 5)")
                    continue
                perdida.backward()
                norma = float(torch.nn.utils.clip_grad_norm_(modelo.parameters(), GRAD_CLIP))
                optimizador.step()
                perdidas.append(float(perdida.detach()))
                normas.append(norma)
                fracciones.append(max(0.0, 1.0 - GRAD_CLIP / norma) if norma > 0.0 else 0.0)
            f1_subtrain = f1_de_tensores(modelo, x_sub_tr, y_sub_tr)
            f1_interna = f1_de_tensores(modelo, x_sub_va, y_sub_va)
            f1_val = f1_de_tensores(modelo, x_va_gpu, y_va_real)
            curvas["loss"].append(float(np.mean(perdidas)) if perdidas else None)
            curvas["f1_subtrain"].append(f1_subtrain)
            curvas["f1_interna"].append(f1_interna)
            curvas["f1_val"].append(f1_val)
            curvas["norma_grad"].append(float(np.mean(normas)) if normas else None)
            curvas["fraccion_recortada"].append(float(np.mean(fracciones)) if fracciones else 0.0)
            curvas["nan_por_epoca"].append(nan_epoca)
            if f1_interna > mejor_f1:
                mejor_f1 = f1_interna
                mejor_epoca = epoca
                mejor_estado = {clave: valor.detach().clone()
                                for clave, valor in modelo.state_dict().items()}
                sin_mejora = 0
            else:
                sin_mejora += 1
            print(f"    [{brazo}] fold {fold} epoca {epoca:02d}: "
                  f"loss {curvas['loss'][-1] if perdidas else float('nan'):.6f} | "
                  f"f1 interna {f1_interna:.6f} | f1 val (reporte) {f1_val:.6f} | "
                  f"norma {curvas['norma_grad'][-1] or 0.0:.4f} | nan {nan_epoca}", flush=True)
            if sin_mejora >= PACIENCIA:
                print(f"    [{brazo}] fold {fold}: ES estricto en la epoca {epoca} "
                      f"(mejor {mejor_epoca})", flush=True)
                break
        assert mejor_estado is not None, "el entrenamiento no registro ninguna mejora"
        modelo.load_state_dict(mejor_estado)
        probabilidades = predecir_probabilidad(modelo, x_va_gpu)
        oof_fold = probabilidades.detach().cpu().numpy().astype(np.float64)
    assert np.isfinite(oof_fold).all(), "OOF del fold con valores no finitos"
    return {"oof_fold": oof_fold, "mejor_epoca": mejor_epoca, "f1_interna_mejor": mejor_f1,
            "epocas": len(curvas["loss"]), "pasos_por_epoca": pasos_por_epoca,
            "nan_total": nan_total, "split_interno_ok": True, **curvas}


def entrenar_brazo(brazo: str, brazo_idx: int, matriz: np.ndarray, clase: np.ndarray,
                   fold_idx: np.ndarray, device: torch.device) -> dict:
    """Corre los 5 folds de un brazo y arma el OOF completo con contador por fila.

    Returns
    -------
    dict
        ``oof`` (169336, 6) float64, ``contador_uno``, NaN acumulado y detalle por fold.

    Raises
    ------
    AbortarBrazo
        Si el brazo acumula ``NAN_MAX_BRAZO`` losses no finitas (codigo 5).
    """
    oof = np.zeros((N_FILAS, N_CLASES), dtype=np.float64)
    contador = np.zeros(N_FILAS, dtype=np.int64)
    folds: dict[str, dict] = {}
    nan_acumulado = 0
    for fold in range(N_FOLDS):
        resultado = entrenar_fold(brazo, brazo_idx, fold, matriz, clase, fold_idx, device)
        va_idx = np.flatnonzero(fold_idx == fold)
        oof[va_idx] = resultado.pop("oof_fold")
        contador[va_idx] += 1
        nan_acumulado += int(resultado["nan_total"])
        assert np.isfinite(oof[va_idx]).all(), f"OOF {brazo} fold {fold} con no finitos"
        folds[str(fold)] = resultado
        print(f"  [{brazo}] fold {fold} listo: epocas {resultado['epocas']}, "
              f"mejor {resultado['mejor_epoca']}, f1 interna {resultado['f1_interna_mejor']:.6f}, "
              f"f1 val (reporte) {resultado['f1_val'][-1]:.6f}", flush=True)
        if nan_acumulado >= NAN_MAX_BRAZO:
            raise AbortarBrazo(f"[STOP] guardas NaN: {brazo} acumula {nan_acumulado} "
                               "losses no finitas (exit 5)")
    assert int(contador.min()) == 1 and int(contador.max()) == 1, \
        f"{brazo}: contador de relleno del OOF distinto de 1"
    assert np.isfinite(oof).all(), f"{brazo}: OOF con valores no finitos"
    return {"oof": oof, "contador_uno": bool((contador == 1).all()),
            "nan_acumulado": nan_acumulado, "folds": folds}


def guardar_oof(brazo: str, oof: np.ndarray) -> dict:
    """Persiste el OOF del brazo en ``features/`` y devuelve archivo y SHA256.

    Parameters
    ----------
    brazo:
        Nombre del brazo (clave de ``OOF_ARCHIVO``).
    oof:
        Matriz de probabilidades (169336, 6) float64.

    Returns
    -------
    dict
        Ruta relativa del ``.npy`` y su SHA256.
    """
    ruta = cfg.FEATURES / f"v32_e1_oof_{OOF_ARCHIVO[brazo]}.npy"
    np.save(ruta, oof)
    return {"archivo": ruta_relativa(ruta), "sha256": sha256_bytes(ruta.read_bytes())}


def sonda_determinismo(matriz: np.ndarray, clase: np.ndarray, fold_idx: np.ndarray,
                       device: torch.device, oof_a1: np.ndarray) -> dict:
    """Reentrena A1 fold 0 con config identica y compara contra el OOF guardado.

    Returns
    -------
    dict
        ``max_abs_diff``, ``identico``, SHA256 del probe persistido y comparacion contra la
        variabilidad de entrenamiento de referencia (0.0008876).
    """
    resultado = entrenar_fold("A1_surrogate", 0, 0, matriz, clase, fold_idx, device)
    probe = resultado["oof_fold"]
    referencia = oof_a1[fold_idx == 0]
    diferencia = float(np.abs(probe - referencia).max())
    ruta = cfg.FEATURES / "v32_e1_probe_a1_fold0.npy"
    np.save(ruta, probe)
    detalle = {"descripcion": "reentrenamiento de A1 fold 0 con config identica",
               "max_abs_diff": diferencia, "identico": diferencia == 0.0,
               "sha256_probe": sha256_bytes(ruta.read_bytes()),
               "archivo_probe": ruta_relativa(ruta),
               "variabilidad_referencia": VARIABILIDAD,
               "menor_o_igual_que_referencia": bool(diferencia <= VARIABILIDAD)}
    print(f"  sonda de determinismo A1 fold 0: max|delta p| = {diferencia:.3e} "
          f"(identico={detalle['identico']})", flush=True)
    return detalle


def entrenar_todo(matriz: np.ndarray, clase: np.ndarray, fold_idx: np.ndarray,
                  device: torch.device) -> dict:
    """Entrena los 3 brazos en orden (A1, sonda, A2, A3) y persiste cada OOF.

    Returns
    -------
    dict
        ``brazos``: detalle por brazo (folds, OOF persistido, sonda); ``oofs``: matrices
        (169336, 6) por brazo.
    """
    detalle_brazos: dict[str, dict] = {}
    oofs: dict[str, np.ndarray] = {}
    for brazo_idx, brazo in enumerate(BRAZOS):
        print(f"\n=== brazo {brazo} ===", flush=True)
        detalle = entrenar_brazo(brazo, brazo_idx, matriz, clase, fold_idx, device)
        oofs[brazo] = detalle.pop("oof")
        detalle["oof_artefacto"] = guardar_oof(brazo, oofs[brazo])
        detalle_brazos[brazo] = detalle
        if brazo == "A1_surrogate":
            detalle["sonda_determinismo"] = sonda_determinismo(matriz, clase, fold_idx,
                                                               device, oofs[brazo])
    return {"brazos": detalle_brazos, "oofs": oofs}


def entrenar_todo_con_degradacion(matriz: np.ndarray, clase: np.ndarray, fold_idx: np.ndarray,
                                  device: torch.device, estado_det: dict) -> dict:
    """Entrena los 3 brazos; si el modo estricto revienta por determinismo, degrada y repite.

    El reintento es unico y siempre corre ``entrenar_todo`` completo desde cero: un fallo a
    mitad de brazo no deja estado parcial util (el OOF solo se persiste al cerrar cada brazo
    y el reintento lo sobrescribe). Cualquier otra excepcion se propaga sin degradar.

    Returns
    -------
    dict
        El mismo resultado de ``entrenar_todo``.

    Raises
    ------
    AbortarBrazo
        Guardas NaN (salida 5); nunca se degrada por esto.
    RuntimeError
        Si el reintento vuelve a fallar por determinismo (salida 1, con mensaje claro).
    """
    try:
        return entrenar_todo(matriz, clase, fold_idx, device)
    except Detencion:
        raise
    except Exception as error:
        if not degradar_por_determinismo(estado_det, error, "entrenamiento"):
            raise
        print("  [aviso] reintento unico del entrenamiento tras degradar a warn_only",
              flush=True)
        try:
            return entrenar_todo(matriz, clase, fold_idx, device)
        except Detencion:
            raise
        except Exception as reintento:
            if es_error_de_determinismo(reintento):
                raise RuntimeError(
                    "[STOP] el reintento del entrenamiento volvio a fallar por determinismo "
                    f"({type(reintento).__name__}: {reintento})") from reintento
            raise


# ======================================================================================
# ---- 5. INTEGRACION P6/P7 (CPU, FUERA DEL LEASE) ----
# ======================================================================================
def bootstrap_delta(clase: np.ndarray, pred_base: np.ndarray, pred_cand: np.ndarray) -> dict:
    """Bootstrap pareado (N_BOOTSTRAP, semilla congelada) de F1(cand) - F1(base).

    Parameters
    ----------
    clase:
        Etiquetas canonicas (169336,).
    pred_base:
        Prediccion del campeon (regla congelada).
    pred_cand:
        Prediccion del candidato.

    Returns
    -------
    dict
        Delta medio, IC95 por percentiles, ``p_delta_positivo`` e ``ic95_sin_cero``.
    """
    generador = np.random.default_rng(SEMILLA)
    codigo_base = clase * N_CLASES + pred_base
    codigo_cand = clase * N_CLASES + pred_cand
    n = len(clase)
    deltas = np.empty(N_BOOTSTRAP, dtype=np.float64)
    for b in range(N_BOOTSTRAP):
        idx = generador.integers(0, n, n)
        deltas[b] = (macro_de_conteos(np.bincount(codigo_cand[idx], minlength=N_CLASES ** 2))
                     - macro_de_conteos(np.bincount(codigo_base[idx], minlength=N_CLASES ** 2)))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"delta_medio": float(deltas.mean()), "ic95": [float(lo), float(hi)],
            "p_delta_positivo": float((deltas > 0).mean()),
            "ic95_sin_cero": bool(lo > 0.0 or hi < 0.0)}


def combinador_oof(v30: np.ndarray, p_brazo: np.ndarray, clase: np.ndarray,
                   fold_idx: np.ndarray) -> tuple[np.ndarray, dict]:
    """Combinador logistico anidado congelado sobre [L2(v30) || L2(p_brazo)] (12 features).

    Returns
    -------
    tuple[numpy.ndarray, dict]
        ``p_comb`` (169336, 6) normalizada y el detalle del combinador (ratio del bloque del
        brazo e iteraciones por fold).

    Raises
    ------
    AssertionError
        Si el combinador de algun fold no ve las 6 clases en orden canonico.
    """
    caracteristicas = np.hstack([normaliza_l2(v30), normaliza_l2(p_brazo)])
    p = np.zeros((len(clase), N_CLASES), dtype=np.float64)
    ratio_por_fold, iteraciones = [], []
    for fold in range(N_FOLDS):
        tr = fold_idx != fold
        va = fold_idx == fold
        modelo = LogisticRegression(C=C_LOGREG, max_iter=MAX_ITER_LOGREG, solver="lbfgs")
        modelo.fit(caracteristicas[tr], clase[tr])
        assert list(modelo.classes_) == list(range(N_CLASES)), \
            "el combinador no vio las 6 clases en orden canonico"
        p[va] = modelo.predict_proba(caracteristicas[va])
        coeficientes = np.abs(modelo.coef_)
        ratio_por_fold.append(float(coeficientes[:, N_CLASES:].mean()
                                    / max(coeficientes[:, :N_CLASES].mean(), L2_CLIP)))
        iteraciones.append(int(np.max(modelo.n_iter_)))
    return normaliza(p), {"ratio_bloque_brazo_por_fold": ratio_por_fold,
                          "n_iter_por_fold": iteraciones}


def evaluar_combinacion(p_comb: np.ndarray, clase: np.ndarray, fold_idx: np.ndarray,
                        champ_pred: np.ndarray, pesos: np.ndarray) -> dict:
    """Metricas del combinador con la regla congelada (argmax p_comb * pesos) y diagnosticos.

    Returns
    -------
    dict
        macro-F1 global, delta vs baseline y nested, folds (global y por fold), folds
        positivos, IC95 del bootstrap, ``argmax_sin_regla`` y filas que cambian.
    """
    pred = np.argmax(p_comb * pesos, axis=1)
    cv = macro(clase, pred)
    folds = [macro(clase[fold_idx == k], pred[fold_idx == k]) for k in range(N_FOLDS)]
    folds_base = [macro(clase[fold_idx == k], champ_pred[fold_idx == k])
                  for k in range(N_FOLDS)]
    ic = bootstrap_delta(clase, champ_pred, pred)
    sin_regla = macro(clase, np.argmax(p_comb, axis=1))
    return {"corrected_cv": cv, "delta_vs_baseline": cv - BASELINE,
            "delta_vs_nested": cv - NESTED, "folds": folds, "folds_base": folds_base,
            "folds_positivos": int(sum(1 for a, b in zip(folds, folds_base) if a > b)),
            "filas_que_cambian": int((pred != champ_pred).sum()),
            "argmax_sin_regla": sin_regla, "delta_sin_regla_vs_baseline": sin_regla - BASELINE,
            "delta_medio_bootstrap": ic["delta_medio"], "ic95": ic["ic95"],
            "p_delta_positivo": ic["p_delta_positivo"], "ic95_sin_cero": ic["ic95_sin_cero"]}


# ======================================================================================
# ---- 6. COMPLEMENTARIEDAD Y DIAGNOSTICO ----
# ======================================================================================
def diagnostico_brazo(p_brazo: np.ndarray, clase: np.ndarray, fold_idx: np.ndarray,
                      champ_pred: np.ndarray, conf: np.ndarray) -> dict:
    """Standalone del brazo (sin pesos), matriz esperada 6x6 y rescate/dano vs campeon.

    Parameters
    ----------
    p_brazo:
        OOF del brazo (169336, 6).
    clase:
        Etiquetas canonicas.
    fold_idx:
        Fold canonico por fila.
    champ_pred:
        Decision del campeon (``argmax(v30 * pesos)``).
    conf:
        Confianza del campeon (``max v30``), para el subconjunto de alta confianza.

    Returns
    -------
    dict
        macro-F1 standalone global y por fold, F1 por clase, matriz ``E[p[:, j] | y = i]``
        con su diagonal, y rescates/danos/neto globales y por clase.
    """
    pred = np.argmax(p_brazo, axis=1)
    f1_por_clase = f1_score(clase, pred, average=None, labels=list(range(N_CLASES)))
    ok_cand = pred == clase
    ok_campeon = champ_pred == clase
    alta_confianza = (~ok_campeon) & (conf >= UMBRAL_ALTA_CONFIANZA)
    por_clase = []
    for c in range(N_CLASES):
        mascara = clase == c
        por_clase.append({
            "clase": cfg.CLASSES[c], "filas": int(mascara.sum()),
            "f1": float(f1_por_clase[c]),
            "rescate": int((~ok_campeon & ok_cand & mascara).sum()),
            "danio": int((ok_campeon & ~ok_cand & mascara).sum()),
            "rescates_alta_confianza": int((alta_confianza & ok_cand & mascara).sum())})
    esperada = np.stack([p_brazo[clase == i].mean(axis=0) for i in range(N_CLASES)])
    diagonal_ok = bool(all(int(np.argmax(esperada[i])) == i for i in range(N_CLASES)))
    rescate = int((~ok_campeon & ok_cand).sum())
    danio = int((ok_campeon & ~ok_cand).sum())
    return {"standalone_cv": macro(clase, pred),
            "standalone_folds": [macro(clase[fold_idx == k], pred[fold_idx == k])
                                 for k in range(N_FOLDS)],
            "f1_por_clase": [float(v) for v in f1_por_clase],
            "matriz_esperada": esperada.tolist(), "diagonal_ok": diagonal_ok,
            "vs_campeon": {"rescate": rescate, "danio": danio, "neto": rescate - danio,
                           "rescates_alta_confianza": int((alta_confianza & ok_cand).sum()),
                           "errores_alta_confianza_campeon": int(alta_confianza.sum()),
                           "por_clase": por_clase}}


# ======================================================================================
# ---- 7. GATES, VEREDICTO Y SALIDAS ----
# ======================================================================================
def como_nativo(valor: object) -> object:
    """Convierte recursivamente a escalares nativos de JSON (float/int/bool/str/None)."""
    if isinstance(valor, dict):
        return {str(clave): como_nativo(item) for clave, item in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [como_nativo(item) for item in valor]
    if isinstance(valor, np.ndarray):
        return como_nativo(valor.tolist())
    if isinstance(valor, bool):
        return bool(valor)
    if isinstance(valor, (float, np.floating)):
        numero = float(valor)
        return numero if math.isfinite(numero) else None
    if isinstance(valor, (int, np.integer)):
        return int(valor)
    if isinstance(valor, np.bool_):
        return bool(valor)
    return valor


def guardar_json(ruta: Path, datos: object) -> None:
    """Escribe JSON indentado con escalares nativos (ASCII)."""
    ruta.write_text(json.dumps(como_nativo(datos), indent=2, ensure_ascii=True) + "\n",
                    encoding="utf-8")


def construir_curvas(detalle_brazos: dict) -> dict:
    """Subconjunto congelado de curvas: brazo -> fold -> loss/f1_interna/f1_val/epocas."""
    curvas: dict[str, dict] = {}
    for brazo in BRAZOS:
        curvas[brazo] = {}
        for fold in range(N_FOLDS):
            detalle = detalle_brazos[brazo]["folds"][str(fold)]
            curvas[brazo][str(fold)] = {
                "loss": detalle["loss"], "f1_interna": detalle["f1_interna"],
                "f1_val": detalle["f1_val"], "epocas_entrenadas": detalle["epocas"],
                "mejor_epoca": detalle["mejor_epoca"]}
    return curvas


def figura_curvas(curvas: dict, ruta: Path) -> None:
    """Figura 2x3 (arriba F1 interna del ES por fold; abajo loss de train por fold)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figura, ejes = plt.subplots(2, 3, figsize=(15.0, 8.0))
    for columna, brazo in enumerate(BRAZOS):
        arriba, abajo = ejes[0][columna], ejes[1][columna]
        for fold in range(N_FOLDS):
            datos = curvas[brazo][str(fold)]
            arriba.plot(datos["f1_interna"], marker=".", linewidth=1.0, label=f"fold {fold}")
            abajo.plot(datos["loss"], marker=".", linewidth=1.0, label=f"fold {fold}")
        arriba.set_title(f"{brazo}: F1 interna (ES)")
        abajo.set_title(f"{brazo}: loss media de train")
        for eje, etiqueta_y in ((arriba, "macro-F1 interna"), (abajo, "loss")):
            eje.set_xlabel("epoca")
            eje.set_ylabel(etiqueta_y)
            eje.grid(alpha=0.3)
        arriba.legend(fontsize=7)
    figura.tight_layout()
    figura.savefig(ruta, dpi=140)
    plt.close(figura)
    print(f"  figura: {ruta_relativa(ruta)}", flush=True)


def invariantes_fold(detalle: dict) -> tuple[bool, bool]:
    """Invariantes que prueban que el ES se decidio solo con la metrica interna del fold.

    Parameters
    ----------
    detalle:
        Detalle de un (brazo, fold) con ``f1_interna`` (curva por epoca), ``mejor_epoca`` y
        ``epocas``.

    Returns
    -------
    tuple[bool, bool]
        ``(mejor_es_maximo_interno, parada_por_paciencia_interna)``: la epoca restaurada es
        el primer maximo de ``f1_interna`` y la parada se explica por la paciencia sobre esa
        misma curva (o por agotar ``EPOCAS_MAX``). Ninguna de las dos usa la curva del
        fold-val, que solo alimenta el reporte por epoca y el OOF final.
    """
    mejor_ok = int(np.argmax(detalle["f1_interna"])) == int(detalle["mejor_epoca"])
    parada_ok = (detalle["epocas"] == EPOCAS_MAX
                 or detalle["epocas"] - 1 - int(detalle["mejor_epoca"]) >= PACIENCIA)
    return bool(mejor_ok), bool(parada_ok)


def calcular_gates(baseline_ok: bool, integracion: dict) -> tuple[dict, dict, str]:
    """Aplica los 6 gates congelados y el arbol de decision (control negativo con precedencia).

    Returns
    -------
    tuple[dict, dict, str]
        Gates, detalle del control negativo y veredicto del arbol.
    """
    a1, a2, a3 = (integracion[brazo] for brazo in BRAZOS)
    gates = {
        "baseline_ok": bool(baseline_ok),
        "delta_A1_suficiente": a1["delta_vs_baseline"] >= GATE_DELTA,
        "folds_positivos_A1_suficiente": a1["folds_positivos"] >= GATE_FOLDS,
        "control_favorable_A1_sobre_A2": a1["delta_vs_baseline"] > a2["delta_vs_baseline"],
        "ic95_A1_sin_cero": bool(a1["ic95_sin_cero"]),
        "sin_leakage": True,
    }
    control = {"delta_A1": a1["delta_vs_baseline"], "delta_A3": a3["delta_vs_baseline"],
               "no_atribuible": bool(a3["delta_vs_baseline"] >= a1["delta_vs_baseline"])}
    if control["no_atribuible"]:
        veredicto = "NO_ATRIBUIBLE"
    elif all(gates.values()):
        veredicto = "PROMISING"
    elif a1["delta_vs_baseline"] > 0:
        veredicto = "MARGINAL"
    else:
        veredicto = "CLOSED"
    return gates, control, veredicto


def escribir_informe(resultado: dict) -> None:
    """Escribe el informe Markdown del experimento con tablas, gates y compliance."""
    integracion = resultado["integracion"]
    detalle_brazos = resultado["entrenamiento"]
    baseline = resultado["verificacion"]["baseline_recomputado"]
    lineas = [
        "# V32-E1 - Ejecucion: challenger con perdida surrogate de macro-F1",
        "",
        f"- Fecha de ejecucion: {resultado['fecha']}",
        (f"- Pre-registro: {ARCHIVO_PRE_REGISTRO} (commit {COMMIT_PRE_REGISTRO}) y "
         f"`experiments/v32_00_preregistro.json`"),
        f"- Commit de ejecucion: `{resultado['commit']}`",
        f"- Baseline primario recomputado: {baseline:.7f} (referencia nested {NESTED:.7f})",
        (f"- Dispositivo: {resultado['dispositivo']['gpu']}, torch "
         f"{resultado['dispositivo']['torch']}, matmul precision "
         f"{resultado['dispositivo']['matmul_precision']}, {HILOS_CPU} hilos CPU, guardian "
         "D-007 (require_gpu wait=10, poll=30 s; sin fallback a CPU)"),
        (f"- Determinismo: modo {resultado['determinismo']['modo']}; sonda A1 fold 0 "
         f"max|delta p| = {resultado['determinismo']['sonda']['max_abs_diff']:.3e} "
         f"(referencia de variabilidad {VARIABILIDAD:.7f})"),
        "",
        "## Integracion por brazo (combinador P6/P7 congelado, regla argmax(p_comb * w))",
        "",
        ("| Brazo | macro-F1 comb | delta vs baseline | folds+ | IC95 | delta sin regla | "
         "delta vs nested | sha256 OOF (16) |"),
        "|---|---|---|---|---|---|---|---|",
    ]
    for brazo in BRAZOS:
        met = integracion[brazo]
        sha = detalle_brazos[brazo]["oof_artefacto"]["sha256"][:16]
        lineas.append(
            f"| {brazo} | {met['corrected_cv']:.7f} | {met['delta_vs_baseline']:+.7f} | "
            f"{met['folds_positivos']}/5 | [{met['ic95'][0]:+.7f}, {met['ic95'][1]:+.7f}] | "
            f"{met['delta_sin_regla_vs_baseline']:+.7f} | {met['delta_vs_nested']:+.7f} | "
            f"`{sha}` |")
    lineas += [
        "",
        "## Entrenamiento (ES estricto; el fold-val solo se uso para el OOF/reporte)",
        "",
        ("| Brazo | fold | epocas | mejor epoca | F1 interna mejor | F1 fold-val (reporte) | "
         "NaN | pasos/epoca |"),
        "|---|---|---|---|---|---|---|---|",
    ]
    for brazo in BRAZOS:
        for fold in range(N_FOLDS):
            detalle = detalle_brazos[brazo]["folds"][str(fold)]
            lineas.append(
                f"| {brazo} | {fold} | {detalle['epocas']} | {detalle['mejor_epoca']} | "
                f"{detalle['f1_interna_mejor']:.6f} | {detalle['f1_val'][-1]:.6f} | "
                f"{detalle['nan_total']} | {detalle['pasos_por_epoca']} |")
    lineas += [
        "",
        "## Complementariedad standalone (argmax del OOF, sin pesos)",
        "",
        ("| Brazo | macro-F1 solo | diagonal 6x6 | rescate | dano | neto | rescate alta "
         "conf. | errores alta conf. campeon |"),
        "|---|---|---|---|---|---|---|---|",
    ]
    for brazo in BRAZOS:
        diag = resultado["complementariedad"][brazo]
        vs = diag["vs_campeon"]
        lineas.append(
            f"| {brazo} | {diag['standalone_cv']:.7f} | {diag['diagonal_ok']} | {vs['rescate']} | "
            f"{vs['danio']} | {vs['neto']:+d} | {vs['rescates_alta_confianza']} | "
            f"{vs['errores_alta_confianza_campeon']} |")
    lineas += [
        "",
        "## Gates",
        "",
        "| Gate | Resultado |",
        "|---|---|",
    ]
    for nombre, valor in resultado["gates"].items():
        lineas.append(f"| {nombre} | {valor} |")
    lineas += [
        "",
        "## Control negativo",
        "",
        (f"- delta_int(A3) = {resultado['control_negativo']['delta_A3']:+.7f} vs "
         f"delta_int(A1) = {resultado['control_negativo']['delta_A1']:+.7f}: "
         f"{'NO_ATRIBUIBLE' if resultado['control_negativo']['no_atribuible'] else 'atribuible'}"
         " a la perdida"),
        "",
        "## Veredicto",
        "",
        f"**{resultado['veredicto']}**: {ARBOL[resultado['veredicto']]}",
        "",
        "## Compliance",
        "",
        ("- 0 embeddings, 0 transformers, 0 LLM, 0 datos externos; sin test, sin holdout, sin "
         "leaderboard, sin submission."),
        ("- Early stopping estricto registrado: metrica interna 90/10; el fold-val jamas se uso "
         "para parar."),
        ("- Entrenamiento GPU bajo guardian D-007; sin fallback a CPU; un script nuevo; campeon "
         "V30 + plug-in intacto."),
        "",
    ]
    INFORME.write_text("\n".join(lineas), encoding="utf-8")
    print(f"  informe: {ruta_relativa(INFORME)}", flush=True)


def anotar_runlog(resultado: dict) -> bool:
    """Anade el bloque del experimento a RUNLOG.md una sola vez (append de texto)."""
    marca = "V32-E1 ejecutado"
    if marca in RUNLOG.read_text(encoding="utf-8"):
        return False
    a1 = resultado["integracion"]["A1_surrogate"]
    bloque = ("\n---\n\n## V32-E1 ejecutado: perdida surrogate de macro-F1 - "
              f"{FECHA}\n\n"
              f"- Veredicto: {resultado['veredicto']}. A1 comb {a1['corrected_cv']:.7f} "
              f"(delta {a1['delta_vs_baseline']:+.7f}, {a1['folds_positivos']}/5 folds); "
              f"generadores A2/A3 como controles.\n"
              f"- Baseline recomputado {BASELINE:.7f}; ES estricto; GPU D-007; sin test, sin "
              "holdout, sin submission.\n"
              f"- Artefactos: {ruta_relativa(RESULTADOS_JSON)} | {ruta_relativa(INFORME)} | "
              f"{ruta_relativa(CURVAS_JSON)}.\n")
    with RUNLOG.open("a", encoding="utf-8") as manejador:
        manejador.write(bloque)
    return True


def anotar_experiments_csv(resultado: dict) -> bool:
    """Anade la fila V32-E1 a experiments.csv con csv.writer (22 columnas de la cabecera)."""
    with EXPERIMENTS_CSV.open("r", encoding="utf-8", newline="") as manejador:
        filas = list(csv.reader(manejador))
    cabecera = filas[0]
    identificadores = {fila[0] for fila in filas[1:] if fila}
    if "V32-E1" in identificadores:
        return False
    a1 = resultado["integracion"]["A1_surrogate"]
    a2 = resultado["integracion"]["A2_ce"]
    a3 = resultado["integracion"]["A3_barajado"]
    resumen = (f"A1 comb {a1['corrected_cv']:.7f} (delta {a1['delta_vs_baseline']:+.7f}, "
               f"{a1['folds_positivos']}/5 folds, IC95 [{a1['ic95'][0]:+.7f}, "
               f"{a1['ic95'][1]:+.7f}]); A2 delta {a2['delta_vs_baseline']:+.7f}; "
               f"A3 delta {a3['delta_vs_baseline']:+.7f}")
    registro = {
        "experiment_id": "V32-E1", "fecha": FECHA, "bloque": "32.1",
        "etapa": "V32-E1 ejecucion: challenger MLP con perdida surrogate de macro-F1 "
                 "(3 brazos, GPU D-007)",
        "pregunta": "una perdida que optimiza macro-F1 directamente produce un challenger "
                    "complementario al campeon?",
        "hipotesis": "alinear entrenamiento y metrica aporta informacion que el campeon no explota",
        "features": "matriz_279 congelada (169336x279); 0 embeddings",
        "preprocessing": "StandardScaler por fold (fit solo fold-train); split interno 90/10 "
                         "estratificado",
        "representacion": "features numericas congeladas del meta-espacio",
        "modelo": "MLP 279-128-6 (GELU, dropout 0.1) en GPU; brazos A1 surrogate / A2 CE / "
                  "A3 barajado",
        "hiperparametros": "congelados: lote 8192, epochs<=60, ES paciencia 8, clip 5.0, "
                           "AdamW lr=1e-3 wd=1e-4",
        "validacion": "particion canonica; integracion P6/P7; gates +0.0015/4-5; bootstrap 2000",
        "macro_f1_mean": f"{float(np.mean(a1['folds'])):.7f}",
        "macro_f1_std": f"{float(np.std(a1['folds'])):.7f}",
        "weighted_f1_mean": "", "micro_f1_mean": "", "accuracy_mean": "",
        "resultado": resumen,
        "interpretacion": f"veredicto {resultado['veredicto']} por el arbol pre-registrado; "
                          "A2/A3 como controles de atribucion",
        "decision": ARBOL[resultado["veredicto"]],
        "duracion_seg": f"{float(resultado['duracion_seg']):.1f}",
        "artefactos": f"{ruta_relativa(RESULTADOS_JSON)} | {ruta_relativa(INFORME)} | "
                      f"{ruta_relativa(CURVAS_JSON)} | {ruta_relativa(FIGURA_CURVAS)}",
    }
    if set(cabecera) != set(registro):
        raise Detencion("[STOP] la fila nueva no respeta las columnas de experiments.csv", 1)
    with EXPERIMENTS_CSV.open("a", encoding="utf-8", newline="") as manejador:
        escritor = csv.DictWriter(manejador, fieldnames=cabecera, extrasaction="raise")
        escritor.writerow(registro)
    return True


def resumen_consola(resultado: dict) -> None:
    """Resumen final en consola (maximo 15 lineas)."""
    integracion = resultado["integracion"]
    print()
    print(f"[V32-E1] fin en {resultado['duracion_seg']:.0f} s | baseline "
          f"{resultado['verificacion']['baseline_recomputado']:.7f} | "
          f"determinismo {resultado['determinismo']['modo']}")
    for brazo in BRAZOS:
        met = integracion[brazo]
        print(f"  {brazo}: comb {met['corrected_cv']:.7f} "
              f"(delta {met['delta_vs_baseline']:+.7f}, {met['folds_positivos']}/5 folds, "
              f"IC95 [{met['ic95'][0]:+.7f}, {met['ic95'][1]:+.7f}])")
    print(f"  gates: {resultado['gates']}")
    print(f"  control negativo: A3 {resultado['control_negativo']['delta_A3']:+.7f} vs A1 "
          f"{resultado['control_negativo']['delta_A1']:+.7f} -> "
          f"{'NO_ATRIBUIBLE' if resultado['control_negativo']['no_atribuible'] else 'ok'}")
    print(f"  VEREDICTO: {resultado['veredicto']}")
    print(f"  artefactos: {ruta_relativa(RESULTADOS_JSON)} | {ruta_relativa(INFORME)} | "
          f"{ruta_relativa(CURVAS_JSON)} | {ruta_relativa(FIGURA_CURVAS)}")
    print(f"  sonda A1: max|delta p| {resultado['determinismo']['sonda']['max_abs_diff']:.3e}")


# ======================================================================================
# ---- 8. CLI ----
# ======================================================================================
def parsear_argumentos() -> argparse.Namespace:
    """Argumentos de linea de comandos."""
    analizador = argparse.ArgumentParser(
        description="V32-E1: implementacion del pre-registro congelado del challenger MLP "
                    "con perdida surrogate de macro-F1.")
    analizador.add_argument("--verificar", action="store_true",
                            help="solo verifica constantes, hashes, alineacion y baseline "
                                 "(no entrena; imprime VERIFICACION OK)")
    return analizador.parse_args()


def ejecutar(verificar: bool) -> int:
    """Pipeline completo: verificacion, baseline, entrenamiento GPU, integracion y gates."""
    inicio = time.time()
    autotest = autotest_numerico()
    prereg = cargar_preregistro()
    constantes_ok, detalle_constantes = verificar_constantes(prereg)
    hashes_ok, detalle_hashes = verificar_hashes(prereg)
    if not (constantes_ok and hashes_ok):
        raise Detencion("[STOP] el pre-registro congelado no se respeta (exit 2)", 2)
    clase, fold_idx, v30 = cargar_canonico()
    pesos = np.array(PESOS, dtype=np.float64)
    champ_pred = np.argmax(v30 * pesos, axis=1)
    conf = v30.max(axis=1)
    baseline = macro(clase, champ_pred)
    print(f"  baseline primario recomputado: {baseline:.7f} (esperado {BASELINE})", flush=True)
    if abs(baseline - BASELINE) > ABORTO_BASELINE:
        raise Detencion("[STOP] el baseline canonico no reproduce (exit 2)", 2)
    matriz = cargar_matriz(prereg)
    print(f"  matriz_279: {matriz.shape} {matriz.dtype} | hashes y constantes: OK", flush=True)

    if verificar:
        print("VERIFICACION OK", flush=True)
        return 0

    if not torch.cuda.is_available():
        raise Detencion("[STOP] GPU no disponible: el pre-registro exige GPU (exit 3)", 3)
    device = torch.device("cuda")
    estado_det = configurar_determinismo()
    sanity = None
    entrenamiento = None
    try:
        # El lease cubre el sanity y el entrenamiento (secciones 3 y 4); la integracion es CPU.
        with require_gpu(wait=10, poll_s=30.0) as lease:
            print(f"  guardian D-007: {lease.status.summary()}", flush=True)
            log_event("v32_01 inicio entrenamiento V32-E1 (3 brazos x 5 folds)")
            try:
                sanity = entrenar_sanity_con_degradacion(matriz, clase, fold_idx, device,
                                                         estado_det)
                print(f"  sanity 1 lote: loss {sanity['loss_inicial']:.6f} -> "
                      f"{sanity['loss_final']:.6f} (reduccion {sanity['reduccion']:.3f}) "
                      f"pass={sanity['pass']}", flush=True)
                if not sanity["pass"]:
                    raise Detencion("[STOP] sanity de 1 lote no pasa (reduccion < 0.30) "
                                    "(exit 4)", 4)
                entrenamiento = entrenar_todo_con_degradacion(matriz, clase, fold_idx, device,
                                                              estado_det)
                log_event("v32_01 fin entrenamiento V32-E1")
            finally:
                log_event("v32_01 fin del trabajo de GPU")
    except (GpuBusyError, BlockingIOError) as error:
        raise Detencion(f"[STOP] GPU ocupada: se devuelve el control sin fallback a CPU: "
                        f"{error} (exit 3)", 3) from error

    curvas = construir_curvas(entrenamiento["brazos"])
    guardar_json(CURVAS_JSON, curvas)
    figura_curvas(curvas, FIGURA_CURVAS)
    oofs = entrenamiento["oofs"]

    integracion = {}
    for brazo in BRAZOS:
        p_comb, detalle_combinador = combinador_oof(v30, oofs[brazo], clase, fold_idx)
        met = evaluar_combinacion(p_comb, clase, fold_idx, champ_pred, pesos)
        met["combinador"] = detalle_combinador
        integracion[brazo] = met
        print(f"  integracion {brazo}: comb {met['corrected_cv']:.7f} "
              f"(delta {met['delta_vs_baseline']:+.7f}, {met['folds_positivos']}/5)", flush=True)

    complementariedad = {brazo: diagnostico_brazo(oofs[brazo], clase, fold_idx, champ_pred, conf)
                         for brazo in BRAZOS}

    sin_leakage = {
        "alineacion_filas": bool(matriz.shape[0] == clase.shape[0] == N_FILAS),
        "cinco_folds_canonicos": bool(np.unique(fold_idx).size == N_FOLDS),
        "split_interno_disjunto_por_fold": all(
            entrenamiento["brazos"][brazo]["folds"][str(fold)]["split_interno_ok"]
            for brazo in BRAZOS for fold in range(N_FOLDS)),
        "oof_contadores_uno_por_brazo": all(entrenamiento["brazos"][brazo]["contador_uno"]
                                            for brazo in BRAZOS),
        # El ES interno solo mira y_sub_va (etiquetas objetivo del brazo); el fold-val solo se
        # usa para el reporte por epoca y para el OOF final. Los dos invariantes se derivan de
        # las curvas guardadas, no se declaran a mano.
        "fold_val_solo_reporte": all(
            all(invariantes_fold(entrenamiento["brazos"][brazo]["folds"][str(fold)]))
            for brazo in BRAZOS for fold in range(N_FOLDS)),
    }
    gates, control_negativo, veredicto = calcular_gates(True, integracion)
    gates["sin_leakage"] = bool(all(sin_leakage.values()))
    if control_negativo["no_atribuible"]:
        veredicto = "NO_ATRIBUIBLE"
    elif all(gates.values()):
        veredicto = "PROMISING"
    elif integracion["A1_surrogate"]["delta_vs_baseline"] > 0:
        veredicto = "MARGINAL"
    else:
        veredicto = "CLOSED"

    resultado = {
        "fase": "V32-E1 ejecucion",
        "fecha": FECHA,
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=False,
                                 capture_output=True, text=True, timeout=30).stdout.strip(),
        "prereg": {"json": "experiments/v32_00_preregistro.json",
                   "sha256": sha256_archivo(PREREG_JSON),
                   "informe": ARCHIVO_PRE_REGISTRO},
        "verificacion": {"constantes": detalle_constantes, "hashes": detalle_hashes,
                         "matriz_sha256": prereg["hashes"]["matriz_279_bytes"],
                         "autotest_numerico": autotest,
                         "baseline_recomputado": baseline, "baseline_ok": True},
        "dispositivo": {"device": str(device), "gpu": torch.cuda.get_device_name(0),
                        "torch": torch.__version__, "hilos_cpu": HILOS_CPU,
                        "matmul_precision": torch.get_float32_matmul_precision()},
        "determinismo": {**estado_det, "seeds": {"global": SEMILLA,
                                                 "barajado_etiquetas": SEMILLA_BARAJADO},
                         "sonda": entrenamiento["brazos"]["A1_surrogate"][
                             "sonda_determinismo"],
                         "probe_max_abs_diff": entrenamiento["brazos"]["A1_surrogate"][
                             "sonda_determinismo"]["max_abs_diff"],
                         "sha256_probe": entrenamiento["brazos"]["A1_surrogate"][
                             "sonda_determinismo"]["sha256_probe"]},
        "sanity": sanity,
        "entrenamiento": entrenamiento["brazos"],
        "integracion": integracion,
        "complementariedad": complementariedad,
        "sin_leakage": sin_leakage,
        "gates": gates,
        "control_negativo": control_negativo,
        "veredicto": veredicto,
        "siguiente_paso": ARBOL[veredicto],
        "compliance": {"embeddings": 0, "transformers": 0, "llm": 0, "datos_externos": 0,
                       "test": 0, "holdout": 0, "leaderboard": 0, "submission": 0,
                       "es_estricto": True, "gpu_d007": True},
        "duracion_seg": float(time.time() - inicio),
    }
    guardar_json(RESULTADOS_JSON, resultado)
    print(f"  resultados: {ruta_relativa(RESULTADOS_JSON)}", flush=True)
    escribir_informe(resultado)
    print(f"  RUNLOG.md: {'anotado' if anotar_runlog(resultado) else 'ya estaba'}", flush=True)
    print(f"  experiments.csv: "
          f"{'fila V32-E1 anadida' if anotar_experiments_csv(resultado) else 'fila ya estaba'}",
          flush=True)
    resumen_consola(resultado)
    return 0


def main() -> int:
    """Entrada del script con manejo de los codigos de salida pre-registrados."""
    argumentos = parsear_argumentos()
    try:
        return ejecutar(argumentos.verificar)
    except Detencion as detencion:
        print(str(detencion), flush=True)
        return detencion.codigo
    except Exception:  # noqa: BLE001 - se reporta el traceback y se sale con 1
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
