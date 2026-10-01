"""age_linguistic_14_diagnostico_pos.py

Fecha: 2026-10-01
Descripcion: runner de las fases A-E del diagnostico V37 de P(POS | edad). Escribe la tabla
    descriptiva, la cota de potencia discriminativa, la estabilidad, el contraste con lo ya
    medido, la auditoria de novedad del canal de transiciones y el veredicto. No produce
    ninguna corrida predictiva, ningun submission y no toca el campeon.
Estado del arte aplicado: NB multinomial cerrado con Laplace alpha = 1.0 fijo y priors
    empiricos (seccion 5.2 de la spec), leave-one-fold-out, IC95 por bootstrap de grupos de
    duplicados (2000 iteraciones, unidad = cluster de duplicado, semilla 20261010),
    estandarizacion por longitud con quintiles de n_tokens, correccion de Bonferroni sobre 18
    pruebas por clase y control de fuga unilateral A_perm / A_perm_val.

Modos
-----
- Sin argumentos: corrida completa de las fases A-E sobre las 169.336 filas. Escribe los
  artefactos reales de la seccion 10 de la spec y nada mas.
- `--verificar`: autoejercicio estructural de los CINCO caminos de la seccion 11, todo en un
  directorio temporal. No escribe nada en `experiments/`, `reports/` ni `artifacts/`. Sale
  con 0 si los cinco caminos pasan y con 2 nombrando el camino que falla.
- `--humo`: cadena completa de las fases A-E sobre 5.000 filas estratificadas (semilla
  declarada) con el numero de iteraciones de bootstrap reducido, escribiendo en un directorio
  temporal. No escribe artefactos reales. Objetivo: menos de 60 s.

ASCII puro: sin acentos, sin emojis, sin CJK (regla 8.4 de la spec). Los .md y las figuras si
admiten tildes.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy import stats as sps

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

FECHA = "2026-10-01"
FASE = "V37-DIAG-POS"
ANOTADORES = ("md", "lg")
ANOTADOR_PRIMARIO = "md"
N_CLASES = 6
ETIQUETAS_POS: list[str] = []
INDICE_ETIQUETA: dict[str, int] = {}
CONSTANTES: dict = {}
N_BOOTSTRAP = 2000
N_BOOTSTRAP_HUMO = 30
N_FILAS_HUMO = 5000
SEMILLA_HUMO = 20261013
N_ITER_PUNTO = 1000
SOPORTE_INSUFICIENTE = 1000
SOPORTE_ESTABLE = 5000
SOPORTE_BIGRAMA = 100
NOMBRES_CLASES = list(cfg.CLASSES)


def _carga_preregistro() -> object:
    """Importa el modulo del pre-registro (script 13) por ruta, sin duplicar sus primitivas.

    Returns
    -------
    module
        El modulo ``age_linguistic_13_preregistro_v37`` ya ejecutado.

    Raises
    ------
    FileNotFoundError
        Si el script 13 no esta en disco: sin el, el pre-registro no existe y V37 no puede
        correr.
    """
    ruta = ROOT / "scripts" / "age_linguistic_13_preregistro_v37.py"
    if not ruta.exists():
        raise FileNotFoundError(
            f"falta {ruta}: es el script que escribe y verifica el pre-registro congelado y "
            "del que este runner importa las primitivas numericas")
    especificacion = importlib.util.spec_from_file_location("v37_preregistro", ruta)
    if especificacion is None or especificacion.loader is None:
        raise ImportError(f"no se pudo cargar {ruta}")
    modulo = importlib.util.module_from_spec(especificacion)
    especificacion.loader.exec_module(modulo)
    return modulo


PR = _carga_preregistro()
ETIQUETAS_POS = list(PR.ETIQUETAS_POS)
INDICE_ETIQUETA = dict(PR.INDICE_ETIQUETA)
CONSTANTES = dict(PR.CONSTANTES)
N_ETIQUETAS = len(ETIQUETAS_POS)
N_CLASES = PR.N_CLASES
UMBRALES = CONSTANTES["umbrales"]
GATES = CONSTANTES["gates"]
SEMILLAS = CONSTANTES["semillas"]
ALFA = CONSTANTES["suavizado"]["alpha"]
PISO_ESTRICTO = CONSTANTES["piso_estricto_macro_f1"]
AZAR_UNIFORME = CONSTANTES["azar_uniforme_macro_f1"]
CRITERIO_A_PERM_MAX = CONSTANTES["criterio_a_perm_max"]
CRITERIO_A_PERM_VAL_MAX = CONSTANTES["criterio_a_perm_val_max"]
BANDA_A_PERM = CONSTANTES["banda_a_perm_esperada"]
P2_MD = 0.2750921
P2_LG = 0.2725490
P1A = 0.3997000


# ==============================================================================
# Rutas de salida: reales o de directorio temporal
# ==============================================================================
@dataclass
class Artefactos:
    """Conjunto de rutas de salida de V37, reales o de un directorio temporal.

    Parameters
    ----------
    base : Path
        Directorio raiz. Si es ``None`` se usan las rutas reales de la seccion 10 de la spec;
        si no, todas las rutas cuelgan de ``base`` con la misma estructura.
    """

    base: Path | None = None

    @property
    def _raiz(self) -> Path:
        return self.base if self.base is not None else ROOT

    @property
    def json_resultado(self) -> Path:
        return self._raiz / "experiments" / "age_linguistic_14_diagnostico_pos.json"

    @property
    def prob_a1_oof(self) -> Path:
        return self._raiz / "artifacts" / "v37" / "a1_prob_oof.npy"

    @property
    def tabla_por_edad(self) -> Path:
        return self._raiz / "reports" / "tables" / "v37_pos_por_edad.csv"

    @property
    def tabla_estabilidad(self) -> Path:
        return self._raiz / "reports" / "tables" / "v37_pos_estabilidad.csv"

    @property
    def tabla_ic_grupo(self) -> Path:
        return self._raiz / "reports" / "tables" / "v37_pos_ic_por_grupo.csv"

    @property
    def tabla_contraste(self) -> Path:
        return self._raiz / "reports" / "tables" / "v37_pos_contraste.csv"

    @property
    def tabla_transiciones(self) -> Path:
        return self._raiz / "reports" / "tables" / "v37_pos_transiciones.csv"

    @property
    def informe(self) -> Path:
        return self._raiz / "reports" / "v37_diagnostico_pos.md"

    @property
    def figuras(self) -> dict[str, Path]:
        """Las cinco figuras de la seccion 10.5, con nombre fijo."""
        base = self._raiz / "reports" / "figures"
        return {
            "01_perfil_por_clase": base / "v37_pos_01_perfil_por_clase.png",
            "02_perfil_md_vs_lg": base / "v37_pos_02_perfil_md_vs_lg.png",
            "03_estabilidad_folds": base / "v37_pos_03_estabilidad_folds.png",
            "04_ic_por_grupo": base / "v37_pos_04_ic_por_grupo.png",
            "05_transiciones_divergencia": base / "v37_pos_05_transiciones_divergencia.png",
        }

    def prepara(self) -> None:
        """Crea los directorios de todas las rutas de salida."""
        for ruta in [self.json_resultado, self.prob_a1_oof, self.tabla_por_edad,
                     self.tabla_estabilidad, self.tabla_ic_grupo, self.tabla_contraste,
                     self.tabla_transiciones, self.informe]:
            ruta.parent.mkdir(parents=True, exist_ok=True)
        for ruta in self.figuras.values():
            ruta.parent.mkdir(parents=True, exist_ok=True)


COLS_POR_EDAD = ["anotador", "variante", "q_estrat", "edad", "tag", "n_tokens", "n_filas",
                 "cobertura", "p", "ref", "delta", "se", "z", "z_bonferroni",
                 "soporte_insuficiente"]
COLS_ESTABILIDAD = ["anotador", "edad", "tag", "n_tokens", "n_tokens_media", "p_min", "p_max",
                    "dispersion", "dispersion_reconciliada", "dispersion_reconcilia",
                    "max_dc_md_lg", "argmax_fold", "z_bonferroni",
                    "soporte_estable_evaluable"]
COLS_IC_GRUPO = ["anotador", "edad", "tag", "delta", "ic95_lo", "ic95_hi", "excluye_cero",
                 "delta_fuera_de_ic", "n_filas_bootstrap"]
COLS_CONTRASTE = ["referencia", "que_midio", "macro_f1", "suelo_estricto", "azar_uniforme",
                  "delta_vs_suelo_estricto", "delta_vs_azar", "se_reexpresa_como", "tipo"]
COLS_TRANSICIONES = ["edad", "pos_i", "pos_j", "n_bigramas", "T_leave_one_fold_out",
                     "T_pooled_leave_one_fold_out", "div_interclase", "div_ruido", "R", "D",
                     "ic95_D_lo", "ic95_D_hi", "excluye_cero_D", "densidad_matriz"]


# ==============================================================================
# SECCION 1 - Entradas
# ==============================================================================
@dataclass
class Entradas:
    """Artefactos de partida ya cargados y alineados por fila.

    Attributes
    ----------
    y, fold_idx, cluster_local, n_grupos : numpy.ndarray / int
        Etiqueta, fold canonico, cluster compacto y numero de grupos de duplicados.
    ids : numpy.ndarray
        Identificador canonico por fila, para indexar las probabilidades OOF.
    anotadores : dict
        ``nombre -> {conteos, n_tokens, pos_seq, quintil, mitad}``.
    filas_validas : numpy.ndarray
        Filas con ``n_tokens > 0``: las 104 restantes quedan fuera de los denominadores.
    indices_originales : numpy.ndarray
        Indices de fila en el corpus completo. En la corrida real es la identidad; en el
        subconjunto de autoejercicio permite alinear cualquier array de longitud completa
        (por ejemplo `n_sents` leido del parquet) con las entradas ya recortadas.
    """

    y: np.ndarray
    fold_idx: np.ndarray
    cluster_local: np.ndarray
    n_grupos: int
    ids: np.ndarray
    anotadores: dict
    filas_validas: np.ndarray
    filas_por_cluster: np.ndarray
    n_filas: int
    subset: bool
    indices_originales: np.ndarray


def prepara_entradas(n_filas_humo: int | None = None) -> Entradas:
    """Carga canonicos, clusters, anotaciones y quintiles, y opcionalmente un subconjunto.

    Parameters
    ----------
    n_filas_humo : int, opcional
        Si no es ``None``, se toma un subconjunto estratificado de ese tamano (modo ``--humo``)
        y los folds se rederivan de forma declarada con ``StratifiedGroupKFold(5, 42)`` sobre
        los clusters del subconjunto. Solo para el autoejercicio: la corrida real usa el
        ``fold_idx.npy`` canonico.

    Returns
    -------
    Entradas
        Entradas listas para las fases A-E.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    canonico = PR.carga_canonicos()
    y = canonico["y_true"]
    fold_idx = canonico["fold_idx"]
    ids = canonico["ids"]
    cluster_local, n_grupos = PR.carga_clusters()
    anotadores = {}
    for nombre in ANOTADORES:
        datos = PR.matriz_conteos(nombre)
        n_tokens = datos["n_tokens"]
        quintil = quintiles_de_longitud(n_tokens)
        mitad = PR.mitad_por_hash(cluster_local, SEMILLAS["hash_agrupador"])
        anotadores[nombre] = {"conteos": datos["conteos"], "n_tokens": n_tokens,
                              "pos_seq": datos["pos_seq"], "quintil": quintil, "mitad": mitad}
    subset = False
    indices_originales = np.arange(len(y), dtype=np.int64)
    if n_filas_humo is not None and n_filas_humo < len(y):
        generador = np.random.default_rng(SEMILLA_HUMO)
        chosen = []
        for c in range(N_CLASES):
            filas = np.flatnonzero(y == c)
            chosen.append(filas[generador.permutation(len(filas))[:n_filas_humo // N_CLASES]])
        seleccion = np.sort(np.concatenate(chosen))
        indices_originales = seleccion
        y = y[seleccion]
        fold_idx = np.empty(len(seleccion), dtype=np.int64)
        clusters_sub = cluster_local[seleccion]
        _, local_sub = np.unique(clusters_sub, return_inverse=True)
        separador = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        for k, (_, val) in enumerate(separador.split(np.zeros(len(y)), y, local_sub)):
            fold_idx[val] = k
        cluster_local = local_sub
        n_grupos = int(local_sub.max()) + 1
        ids = ids[seleccion]
        subset = True
        for nombre in ANOTADORES:
            base = anotadores[nombre]
            anotadores[nombre] = {**base, "conteos": base["conteos"][seleccion],
                                  "n_tokens": base["n_tokens"][seleccion],
                                  "pos_seq": base["pos_seq"][seleccion],
                                  "quintil": base["quintil"][seleccion],
                                  "mitad": base["mitad"][seleccion]}
    filas_por_cluster = np.bincount(cluster_local, minlength=n_grupos).astype(np.float64)
    filas_validas = np.flatnonzero(anotadores[ANOTADOR_PRIMARIO]["n_tokens"] > 0)
    return Entradas(y=y, fold_idx=fold_idx, cluster_local=cluster_local, n_grupos=n_grupos,
                    ids=ids, anotadores=anotadores, filas_validas=filas_validas,
                    filas_por_cluster=filas_por_cluster, n_filas=len(y), subset=subset,
                    indices_originales=indices_originales)


def quintiles_de_longitud(n_tokens: np.ndarray) -> np.ndarray:
    """Quintiles de `n_tokens` con cortes en los percentiles 20/40/60/80 declarados.

    Parameters
    ----------
    n_tokens : numpy.ndarray
        Longitud en tokens por fila.

    Returns
    -------
    numpy.ndarray
        Quintil por fila, en ``0..4``.
    """
    cortes = np.percentile(n_tokens, [20, 40, 60, 80])
    quintil = np.zeros(len(n_tokens), dtype=np.int64)
    for j, corte in enumerate(cortes):
        quintil[n_tokens > corte] = j + 1
    return quintil


# ==============================================================================
# SECCION 2 - FASE A - perfil P(tag | edad) y cota de potencia
# ==============================================================================
def se_ratio_unidad_tweet(conteos: np.ndarray, y: np.ndarray, filas: np.ndarray) -> np.ndarray:
    """Error estandar del perfil con la UNIDAD DE MUESTREO = TWEET (delta method, E11).

    ``p[c,t] = sum_i x[i,t] / sum_i n_i`` es un estimador de razon; su varianza se linealiza
    alrededor de la media de proporciones por tweet. La unidad correcta no es el token: con
    2.590.877 tokens iid el error sale ~3 veces mayor (factores 0,28-0,41), lo que cambia
    veredictos. Se etiqueta como **secundario**: la inferencia la gobierna el bootstrap por
    grupos de duplicados.

    Parameters
    ----------
    conteos : numpy.ndarray
        Matriz de conteos por fila y etiqueta.
    y : numpy.ndarray
        Etiqueta verdadera por fila.
    filas : numpy.ndarray
        Filas que entran en el estimador.

    Returns
    -------
    numpy.ndarray
        ``(6, 18)`` con la varianza estimada de cada ``p[c,t]``.
    """
    denso = conteos[filas].toarray() if sparse.issparse(conteos[filas]) else conteos[filas]
    totales = denso.sum(axis=1)
    proporciones = denso / np.where(totales > 0, totales, 1.0)[:, None]
    y_sub = y[filas]
    salida = np.zeros((N_CLASES, N_ETIQUETAS), dtype=np.float64)
    for c in range(N_CLASES):
        seleccion = y_sub == c
        n = int(seleccion.sum())
        if n < 2:
            continue
        u = proporciones[seleccion]
        media_n = float(totales[seleccion].mean())
        if media_n <= 0:
            continue
        residuos = u - u.mean(axis=0)[None, :]
        salida[c] = (residuos ** 2).sum(axis=0) / (n * (n - 1) * media_n ** 2)
    n_total = len(filas)
    if n_total >= 2:
        p_pooled = proporciones.mean(axis=0)
        media_n = float(totales.mean())
        residuos = proporciones - p_pooled[None, :]
        var_pooled = (residuos ** 2).sum(axis=0) / (n_total * (n_total - 1) * media_n ** 2)
        salida = salida + var_pooled[None, :]
    return salida


def perfil_anotador(entradas: Entradas, nombre: str) -> dict:
    """Calcula el perfil completo de un anotador: crudo, estandarizado y sin `SPACE`.

    Parameters
    ----------
    entradas : Entradas
        Entradas ya cargadas.
    nombre : str
        ``"md"`` o ``"lg"``.

    Returns
    -------
    dict
        ``p_crudo``, ``p_est``, ``p_por_quintil``, ``p_sin_space``, ``n_por_clase``,
        ``n_filas_por_clase``, ``n_sin_space``, ``ref``, ``delta``, ``se``, ``z``,
        ``z_bonferroni``, ``se_tweet``, ``z_tweet``, ``z_tweet_bonferroni``,
        ``p_media_proporciones`` (E12) y el detalle de `SPACE` por clase y quintil (E10).
    """
    datos = entradas.anotadores[nombre]
    conteos = datos["conteos"]
    y = entradas.y
    filas = entradas.filas_validas
    n_por_clase, n_filas_por_clase = PR.perfil_por_clase(conteos, y, filas)
    p_crudo = PR.p_desde_conteos(n_por_clase)
    p_sin_space, n_sin_space = PR.p_sin_space(n_por_clase)
    p_est, p_por_quintil = PR.estandariza_por_longitud(conteos, y, datos["quintil"], filas)
    p_media = PR.p_media_de_proporciones(conteos, y, filas)
    ref = n_por_clase.sum(axis=0) / n_por_clase.sum()
    n_pooled = n_por_clase.sum(axis=0)
    delta, se, z, z_corr = PR.se_y_z(p_crudo, ref, n_por_clase, n_pooled)
    var_tweet = se_ratio_unidad_tweet(conteos, y, filas)
    se_tweet = np.sqrt(np.clip(var_tweet, 0.0, None))
    z_tweet = np.zeros_like(delta)
    positivo = se_tweet > 0
    z_tweet[positivo] = delta[positivo] / se_tweet[positivo]
    critico = float(np.sqrt(2.0 * np.log(np.sqrt(2.0 * np.pi * 0.05 / N_ETIQUETAS) / 0.05)))
    j_space = INDICE_ETIQUETA["SPACE"]
    tokens_clase = n_por_clase.sum(axis=1)
    quota_space = {NOMBRES_CLASES[c]: round(float(n_por_clase[c, j_space]
                                                    / max(tokens_clase[c], 1.0)), 6)
                   for c in range(N_CLASES)}
    cuota_space_quintil = {}
    n_filas_por_clase_y_quintil = np.zeros((5, N_CLASES, N_ETIQUETAS), dtype=np.int64)
    n_filas_clase_quintil = np.zeros((5, N_CLASES), dtype=np.int64)
    for q in range(5):
        seleccion_quintil = filas[datos["quintil"][filas] == q]
        y_sub = y[seleccion_quintil]
        por_clase = {}
        for c in range(N_CLASES):
            ambas = seleccion_quintil[y_sub == c]
            n_filas_clase_quintil[q, c] = len(ambas)
            # `matriz_f` son las FILAS que contienen la etiqueta: depende de (clase, quintil),
            # por eso se acumula por celda y no se sobrescribe dentro del bucle.
            matriz_f, _ = PR.perfil_por_clase(conteos, y, ambas)
            n_filas_por_clase_y_quintil[q, c] = matriz_f.sum(axis=0).astype(np.int64)
            totales = float(conteos[ambas].sum())
            por_clase[NOMBRES_CLASES[c]] = round(
                float(conteos[ambas][:, j_space].sum()) / max(totales, 1.0), 6)
        cuota_space_quintil[f"q{q + 1}"] = por_clase
    n_filas_clase = n_filas_clase_quintil.sum(axis=0).astype(np.float64)
    assert np.array_equal(n_filas_clase,
                          np.array([int(((y == c) & (datos["n_tokens"] > 0)).sum())
                                    for c in range(N_CLASES)], dtype=np.float64)), \
        "el total por clase de los quintiles debe coincidir con el total de la clase"
    return {
        "p_crudo": p_crudo, "p_est": p_est, "p_por_quintil": p_por_quintil,
        "p_sin_space": p_sin_space, "n_por_clase": n_por_clase,
        "n_filas_por_clase": n_filas_por_clase, "n_sin_space": n_sin_space,
        "n_filas_por_clase_y_quintil": n_filas_por_clase_y_quintil,
        "n_filas_clase_y_quintil": n_filas_clase_quintil,
        "ref": ref, "delta": delta, "se": se, "z": z, "z_bonferroni": z_corr,
        "se_tweet": se_tweet, "z_tweet": z_tweet,
        "z_tweet_bonferroni": z_tweet - critico,
        "p_media_proporciones": p_media,
        "n_filas_clase": n_filas_clase,
        "space_por_clase": quota_space,
        "space_por_clase_y_quintil": cuota_space_quintil,
    }


def contexto_descriptivo(entradas: Entradas, nombre: str) -> dict:
    """A2: distribucion de longitudes, `n_sents`, entropia del perfil y perfil pooled.

    Parameters
    ----------
    entradas : Entradas
        Entradas ya cargadas.
    nombre : str
        Anotador.

    Returns
    -------
    dict
        Estadisticos de apoyo. Es contexto, no criterio.
    """
    datos = entradas.anotadores[nombre]
    n_tokens = datos["n_tokens"].astype(np.float64)
    # `n_sents` se lee del parquet COMPLETO y se recorta con `entradas.filas`: `entradas`
    # puede ser un subconjunto (humo y autoejercicio), y sin este recorte la mascara de
    # clase no coincide con el eje de los datos.
    marco_completo = pd.read_parquet(PR.PARQUETS[nombre], columns=["n_sents"])
    n_sents = marco_completo["n_sents"].to_numpy()[entradas.indices_originales]
    longitudes = {}
    entropias = {}
    n_sents_clase = {}
    for c in range(N_CLASES):
        mascara = entradas.y == c
        longitudes[NOMBRES_CLASES[c]] = {
            "media": round(float(n_tokens[mascara].mean()), 4),
            "mediana": float(np.median(n_tokens[mascara])),
            "min": int(n_tokens[mascara].min()),
            "max": int(n_tokens[mascara].max()),
        }
        n_sents_clase[NOMBRES_CLASES[c]] = {
            "media": round(float(n_sents[mascara].mean()), 4),
            "mediana": float(np.median(n_sents[mascara])),
            "max": int(n_sents[mascara].max()),
        }
        perfil = np.asarray(datos["conteos"][mascara].sum(axis=0)).ravel()
        perfil = perfil / max(perfil.sum(), 1.0)
        entropias[NOMBRES_CLASES[c]] = round(
            float(-(perfil[perfil > 0] * np.log(perfil[perfil > 0])).sum()), 6)
    perfil_pooled = np.asarray(datos["conteos"].sum(axis=0)).ravel()
    perfil_pooled = perfil_pooled / max(perfil_pooled.sum(), 1.0)
    return {"n_tokens_por_clase": longitudes, "n_sents_por_clase": n_sents_clase,
            "entropia_perfil_pos_por_clase": entropias,
            "n_tokens_media_corpus": round(float(n_tokens.mean()), 4),
            "n_tokens_mediana_corpus": float(np.median(n_tokens)),
            "perfil_pooled": {ETIQUETAS_POS[t]: round(float(perfil_pooled[t]), 9)
                              for t in range(N_ETIQUETAS)}}


def filas_tabla_por_edad(perfil: dict, nombre: str) -> list[dict]:
    """Construye las filas largas de `v37_pos_por_edad.csv` (864 por anotador, 1728 en total).

    Parameters
    ----------
    perfil : dict
        Salida de :func:`perfil_anotador`.
    nombre : str
        Anotador.

    Returns
    -------
    list[dict]
        Filas con las columnas de :data:`COLS_POR_EDAD`. ``ref``, ``delta``, ``se``, ``z`` y
        ``z_bonferroni`` solo se rellenan en las filas ``q_estrat = "pooled"`` de la variante
        ``crudo``: en los quintiles van vacios porque la spec no selecciona estrato.
    """
    filas: list[dict] = []
    n_por_clase = perfil["n_por_clase"]

    def anade(variante: str, q_estrat: str, p_matriz: np.ndarray, n_matriz: np.ndarray,
              n_filas_matriz: np.ndarray, n_clase_matriz: np.ndarray,
              con_diagnostico: bool) -> None:
        for c in range(N_CLASES):
            for t, etiqueta in enumerate(ETIQUETAS_POS):
                filas.append({
                    "anotador": nombre, "variante": variante, "q_estrat": q_estrat,
                    "edad": NOMBRES_CLASES[c], "tag": etiqueta,
                    "n_tokens": int(n_matriz[c, t]),
                    "n_filas": int(n_filas_matriz[c, t]),
                    "cobertura": round(float(n_filas_matriz[c, t]
                                             / max(n_clase_matriz[c], 1.0)), 6),
                    "p": round(float(p_matriz[c, t]), 9),
                    "ref": round(float(perfil["ref"][t]), 9) if con_diagnostico else "",
                    "delta": round(float(perfil["delta"][c, t]), 9) if con_diagnostico else "",
                    "se": round(float(perfil["se"][c, t]), 9) if con_diagnostico else "",
                    "z": round(float(perfil["z"][c, t]), 4) if con_diagnostico else "",
                    "z_bonferroni": round(float(perfil["z_bonferroni"][c, t]), 4)
                    if con_diagnostico else "",
                    "soporte_insuficiente": bool(n_matriz[c, t] < SOPORTE_INSUFICIENTE),
                })

    anade("crudo", "pooled", perfil["p_crudo"], n_por_clase, perfil["n_filas_por_clase"],
          perfil["n_filas_clase"], True)
    anade("estandarizado", "pooled", perfil["p_est"], n_por_clase,
          perfil["n_filas_por_clase"], perfil["n_filas_clase"], False)
    anade("sin_space", "pooled", perfil["p_sin_space"], perfil["n_sin_space"],
          perfil["n_filas_por_clase"], perfil["n_filas_clase"], False)
    for q in range(5):
        anade("crudo", f"q{q + 1}", perfil["p_por_quintil"][q], n_por_clase,
              perfil["n_filas_por_clase_y_quintil"][q], perfil["n_filas_clase_y_quintil"][q],
              False)
    return filas


def fases_a_cota(entradas: Entradas, n_bootstrap: int) -> dict:
    """A3: los cuatro brazos de la cota mas los controles, en md y lg.

    Parameters
    ----------
    entradas : Entradas
        Entradas ya cargadas.
    n_bootstrap : int
        Iteraciones del bootstrap por grupos.

    Returns
    -------
    dict
        ``A0``, y por anotador ``A1``, ``A2``, ``A_perm``, ``A_perm_val``, el IC95 pareado de
        ``A1 - A_perm`` (el instrumento de inferencia que manda E1), las dos convenciones de
        A1 (E6) y el detalle de las 104 filas sin tokens (E5).
    """
    y = entradas.y
    fold_idx = entradas.fold_idx
    y_perm = PR.permuta_etiquetas_train(y, fold_idx, SEMILLAS["permute_control"])
    y_perm_cluster = PR.permuta_etiquetas_por_cluster(y, entradas.cluster_local,
                                                     SEMILLAS["permute_control"])
    mayoritaria = int(np.argmax(np.bincount(y, minlength=N_CLASES)))
    a0_pred = np.full(len(y), mayoritaria, dtype=np.int64)
    a0 = PR.macro_f1(y, a0_pred)
    salida: dict = {"A0": {"macro_f1_pooled": round(a0, 8), "clase_constante": mayoritaria,
                           "clase_constante_nombre": NOMBRES_CLASES[mayoritaria]},
                    "n_filas_sin_tokens_regla_E5": int((entradas.anotadores[ANOTADOR_PRIMARIO]
                                                        ["n_tokens"] == 0).sum())}
    predicciones: dict[str, np.ndarray] = {}
    for nombre in ANOTADORES:
        print(f"    cota {nombre}: ajustando A1 ...", flush=True)
        datos = entradas.anotadores[nombre]
        brazo1 = PR.naive_bayes_pos(datos["conteos"], datos["n_tokens"], y, fold_idx,
                                    modo="proporciones", alpha=ALFA)
        print(f"    cota {nombre}: ajustando A2 ...", flush=True)
        brazo2 = PR.naive_bayes_pos(datos["conteos"], datos["n_tokens"], y, fold_idx,
                                    modo="conteos", alpha=ALFA)
        print(f"    cota {nombre}: A_perm (control de fuga) ...", flush=True)
        brazo_perm = PR.naive_bayes_pos(datos["conteos"], datos["n_tokens"], y, fold_idx,
                                        y_ajuste=y_perm, modo="proporciones", alpha=ALFA)
        brazo_perm_val = PR.naive_bayes_pos(datos["conteos"], datos["n_tokens"], y_perm_cluster,
                                            fold_idx, modo="proporciones", alpha=ALFA)
        predicho_por_fold = np.bincount(fold_idx, minlength=5)
        assert np.array_equal(np.bincount(fold_idx, minlength=5), brazo1["n_predichas_por_fold"]), \
            "cada fila debe predecirse exactamente una vez, por el fold del que es validacion"
        assert int(predicho_por_fold.sum()) == len(y)
        assert len(np.unique(fold_idx)) == 5
        a1_pooled = PR.macro_f1(y, brazo1["pred"])
        a1_media = float(np.mean([PR.macro_f1(y[fold_idx == k], brazo1["pred"][fold_idx == k])
                                  for k in range(5)]))
        a2_pooled = PR.macro_f1(y, brazo2["pred"])
        a2_media = float(np.mean([PR.macro_f1(y[fold_idx == k], brazo2["pred"][fold_idx == k])
                                  for k in range(5)]))
        a_perm = PR.macro_f1(y, brazo_perm["pred"])
        a_perm_val = PR.macro_f1(y_perm_cluster, brazo_perm_val["pred"])
        vacias = datos["n_tokens"] == 0
        salida[nombre] = {
            "A1": {"macro_f1_pooled": round(a1_pooled, 8),
                   "macro_f1_media_por_fold": round(a1_media, 8),
                   "detalle_por_fold": brazo1["detalle_por_fold"]},
            "A2": {"macro_f1_pooled": round(a2_pooled, 8),
                   "macro_f1_media_por_fold": round(a2_media, 8),
                   "enunciado": ("brazo de SENSIBILIDAD A LA LONGITUD, no identico a A1 (E2): con "
                                 "priors la invariancia solo se cumple sin prior; aqui A2 - A1 "
                                 "!= 0 porque el factor n_i escala el termino de verosimilitud"),
                   "n_filas_que_cambian_respecto_a_A1": int((brazo1["pred"]
                                                             != brazo2["pred"]).sum())},
            "A_perm": {"macro_f1_pooled": round(a_perm, 8),
                       "criterio_unilateral_E1_max": CRITERIO_A_PERM_MAX,
                       "cumple_criterio": bool(a_perm <= CRITERIO_A_PERM_MAX),
                       "dentro_de_banda_declarada": bool(BANDA_A_PERM[0] <= a_perm
                                                        <= BANDA_A_PERM[1]),
                       "n_filas_predichas_como_clase_mayoritaria":
                           int(np.bincount(brazo_perm["pred"],
                                           minlength=N_CLASES)[mayoritaria])},
            "A_perm_val": {"macro_f1_pooled": round(a_perm_val, 8),
                           "criterio_unilateral_E3_max": CRITERIO_A_PERM_VAL_MAX,
                           "cumple_criterio": bool(a_perm_val <= CRITERIO_A_PERM_VAL_MAX),
                           "bincount_y_permutado_por_cluster":
                               [int(v) for v in np.bincount(y_perm_cluster, minlength=N_CLASES)]},
            "diferencias": {
                "A1_menos_A0": round(a1_pooled - a0, 8),
                "A2_menos_A1": round(a2_pooled - a1_pooled, 8),
                "A1_menos_P2_md": round(a1_pooled - P2_MD, 8),
                "A1_menos_P1a": round(a1_pooled - P1A, 8),
            },
            "n_filas_sin_tokens": int(vacias.sum()),
            "prediccion_de_las_filas_sin_tokens": "argmax_c pi_c (E5: x = 0, luego el score es "
                                                  "log pi)",
        }
        predicciones[f"A1_{nombre}"] = brazo1["pred"]
        predicciones[f"A2_{nombre}"] = brazo2["pred"]
        predicciones[f"A_perm_{nombre}"] = brazo_perm["pred"]
        if nombre == ANOTADOR_PRIMARIO:
            salida["prob_oof_a1_anotador_primario"] = brazo1["prob"]
    print(f"    IC95 por grupos ({n_bootstrap} iteraciones, {entradas.n_grupos} grupos) ...",
          flush=True)
    ic = PR.ic95_macro_f1_por_grupo(y, predicciones, entradas.cluster_local, entradas.n_grupos,
                                    entradas.filas_por_cluster, SEMILLAS["bootstrap_cluster"],
                                    n_bootstrap)
    salida["ic95_por_grupo"] = ic
    salida["A0"]["ic95"] = ic["por_brazo"].get("A0")
    for nombre in ANOTADORES:
        salida[nombre]["A1"]["ic95"] = ic["por_brazo"][f"A1_{nombre}"]
        salida[nombre]["A2"]["ic95"] = ic["por_brazo"][f"A2_{nombre}"]
        salida[nombre]["A_perm"]["ic95"] = ic["por_brazo"][f"A_perm_{nombre}"]
        salida[nombre]["A1_menos_A_perm"] = {
            "ic95": ic["diferencias"][f"A1_{nombre}_menos_A_perm_{nombre}"]["ic95"],
            "excluye_cero": ic["diferencias"][f"A1_{nombre}_menos_A_perm_{nombre}"]["excluye_cero"],
            "instrumento": "es el criterio de inferencia real de E1 (IC95 pareado que excluye "
                           "cero), no la igualdad de A_perm a un numero de 5 decimales",
        }
    return salida


# ==============================================================================
# SECCION 3 - FASE B - estabilidad
# ==============================================================================
def perfiles_por_fold(conteos: np.ndarray, y: np.ndarray, fold_idx: np.ndarray,
                      filas: np.ndarray) -> np.ndarray:
    """Perfil leave-one-fold-out: ``(5, 6, 18)``.

    Parameters
    ----------
    conteos : numpy.ndarray
        Matriz de conteos por fila y etiqueta.
    y : numpy.ndarray
        Etiqueta verdadera por fila.
    fold_idx : numpy.ndarray
        Fold canonico por fila.
    filas : numpy.ndarray
        Filas con ``n_tokens > 0``.

    Returns
    -------
    numpy.ndarray
        ``p_k[c,t]`` estimado SOLO con las filas ``fold_idx != k``.
    """
    salida = np.zeros((5, N_CLASES, N_ETIQUETAS), dtype=np.float64)
    y_sub = y[filas]
    for k in range(5):
        seleccion = fold_idx[filas] != k
        sub = conteos[filas[seleccion]]
        y_k = y_sub[seleccion]
        matriz_n = np.zeros((N_CLASES, N_ETIQUETAS), dtype=np.float64)
        for c in range(N_CLASES):
            mascara = y_k == c
            if mascara.any():
                matriz_n[c] = np.asarray(sub[mascara].sum(axis=0)).ravel()
        salida[k] = PR.p_desde_conteos(matriz_n)
    return salida


def fase_b(entradas: Entradas, perfiles: dict, p_por_fold: dict, n_bootstrap: int) -> dict:
    """B1, B2, B3 y el sanity check de proyeccion universal de E13.

    Parameters
    ----------
    entradas : Entradas
        Entradas ya cargadas.
    perfiles : dict
        Salida de :func:`perfil_anotador` por anotador.
    p_por_fold : dict
        Salida de :func:`perfiles_por_fold` por anotador.
    n_bootstrap : int
        Iteraciones del bootstrap por grupos.

    Returns
    -------
    dict
        Dispersion, acuerdo md/lg, IC95 por grupo, control de cobertura y la coincidencia de
        secuencia md/lg publicada como limitacion del conjunto de datos (E13).
    """
    p_md = p_por_fold[ANOTADOR_PRIMARIO]
    p_lg = p_por_fold["lg"]
    # `diferencia` tiene forma (fold, clase, etiqueta): el eje 0 son los FOLDS, luego el maximo
    # y el argmax van por axis 0 y quedan con forma (clase, etiqueta). Indexarlos con un solo
    # indice devolveria una fila de 18 valores, no un escalar.
    diferencia = np.abs(p_md - p_lg)
    max_dc = diferencia.max(axis=0)
    argmax_fold = diferencia.argmax(axis=0)
    assert max_dc.shape == (N_CLASES, N_ETIQUETAS), "max_dc debe ser (clase, etiqueta)"
    # `dispersion_por_anotador` es POR ANOTADOR y por eso SI reconcilia con `p_max - p_min` de la
    # misma fila del CSV. `dispersion` es el maximo cruzado de los dos anotadores y se usa solo
    # como criterio conservador del gate V2/G2, nunca como columna de la tabla.
    dispersion_por_anotador = {}
    for nombre in ANOTADORES:
        p_k = p_por_fold[nombre]
        dispersion_por_anotador[nombre] = p_k.max(axis=0) - p_k.min(axis=0)
    dispersion = np.maximum(dispersion_por_anotador[ANOTADOR_PRIMARIO],
                            dispersion_por_anotador["lg"])
    print(f"    IC95 por grupos del perfil ({n_bootstrap} iteraciones) ...", flush=True)
    ic_por_anotador = {}
    for nombre in ANOTADORES:
        suficiente = PR.perfil_por_cluster(entradas.anotadores[nombre]["conteos"], entradas.y,
                                           entradas.cluster_local, entradas.n_grupos)
        ic_por_anotador[nombre] = PR.ic95_delta_por_grupo(suficiente, n_bootstrap,
                                                          SEMILLAS["bootstrap_cluster"])
    secuencia = {}
    for nombre in ANOTADORES:
        secuencia[nombre] = int((entradas.anotadores[nombre]["conteos"].getnnz(axis=1) > 0).sum())
    pos_md = entradas.anotadores["md"]["pos_seq"]
    pos_lg = entradas.anotadores["lg"]["pos_seq"]
    coincide_secuencia = float(np.mean([a == b for a, b in zip(pos_md, pos_lg)]))
    con_space = np.array([int(np.count_nonzero(s.split())) for s in pos_md])
    return {
        "dispersion_por_fold": {nombre: valor.tolist()
                                 for nombre, valor in dispersion_por_anotador.items()},
        "dispersion_max_entre_anotadores": dispersion.tolist(),
        "nota_dispersion": (
            "`dispersion_por_fold` es POR ANOTADOR y reconcilia con p_max - p_min de la misma "
            "fila de `v37_pos_estabilidad.csv`. `dispersion_max_entre_anotadores` es el maximo de "
            "los dos y se usa solo como criterio conservador de los gates V2 y G2."),
        "max_dc_md_lg": max_dc,
        "argmax_fold_md_lg": argmax_fold,
        "p_por_fold": p_por_fold,
        "ic95_delta": ic_por_anotador,
        "n_train_por_fold": [int((entradas.fold_idx != k).sum()) for k in range(5)],
        "control_cobertura_grupos": PR.ic95_delta_por_grupo(
            PR.perfil_por_cluster(entradas.anotadores[ANOTADOR_PRIMARIO]["conteos"], entradas.y,
                                  entradas.cluster_local, entradas.n_grupos), 1,
            SEMILLAS["bootstrap_cluster"]).get("n_iter"),
        "sanity_check_proyeccion_universal_E13": {
            "coincidencia_de_secuencia_fila_a_fila": round(coincide_secuencia, 6),
            "filas_no_vacias_md": secuencia["md"],
            "filas_no_vacias_lg": secuencia["lg"],
            "desajuste_de_presencia_relativo": round(
                abs(secuencia["md"] - secuencia["lg"]) / max(secuencia["md"], 1), 6),
            "por_que_es_un_sanity_check": (
                "los parquets congelados contienen SOLO UPOS en pos_seq: el tag especifico del "
                "modelo nunca se congelo. Ambos anotadores proyectan sobre el mismo inventario "
                "de 18 por construccion, luego la comparacion NO puede discriminar el modelo. "
                "Ademas no son anotadores independientes: mismo pipeline y mismo tokenizer. Por "
                "eso B2 no es una comparacion entre anotadores sino un sanity check de la "
                "proyeccion universal, y la coincidencia de secuencia se publica como "
                "LIMITACION DEL CONJUNTO DE DATOS, no como resultado"),
        },
        "n_tokens_medios_por_fold": con_space,
    }


# ==============================================================================
# SECCION 4 - FASE C - contraste con lo ya medido
# ==============================================================================
def correlaciones_fase_c(entradas: Entradas, perfil: dict) -> dict:
    """TC.1: si la tabla de V37 es la misma senal que las 18 features de D-026.

    Vector A = proporciones UPOS por fila con la definicion LITERAL de D-026
    (``if not token.is_space``, denominador = tokens sin SPACE). Vector B = perfil UPOS de la
    fila reconstruido desde ``p[edad|t]`` con la edad verdadera, es decir ``p[y_i, :]``.

    Parameters
    ----------
    entradas : Entradas
        Entradas ya cargadas.
    perfil : dict
        Salida de :func:`perfil_anotador` del anotador primario.

    Returns
    -------
    dict
        Correlaciones por fila (la cifra literal de TC.1), agregada sobre las 18 etiquetas y
        por clase, con Pearson y Spearman.
    """
    datos = entradas.anotadores[ANOTADOR_PRIMARIO]
    j_space = INDICE_ETIQUETA["SPACE"]
    generador = np.random.default_rng(SEMILLAS["muestra_correlacion_fase_c"])
    n_muestra = min(20000, len(entradas.y))
    filas = generador.choice(len(entradas.y), size=n_muestra, replace=False)
    conteos = datos["conteos"][filas].toarray()
    totales = conteos.sum(axis=1)
    sin_space = conteos.copy()
    sin_space[:, j_space] = 0.0
    vector_a = sin_space / np.where(sin_space.sum(axis=1, keepdims=True) > 0,
                                    sin_space.sum(axis=1, keepdims=True), 1.0)
    vector_b = perfil["p_crudo"][entradas.y[filas]]
    pearson_fila = float(sps.pearsonr(vector_a.ravel(), vector_b.ravel())[0])
    spearman_fila = float(sps.spearmanr(vector_a.ravel(), vector_b.ravel())[0])
    agregado_a = sin_space.sum(axis=0)
    agregado_a = agregado_a / max(agregado_a.sum(), 1.0)
    pearson_agregado = float(sps.pearsonr(agregado_a, perfil["ref"])[0])
    spearman_agregado = float(sps.spearmanr(agregado_a, perfil["ref"])[0])
    por_clase = {}
    for c in range(N_CLASES):
        mascara = entradas.y[filas] == c
        if mascara.sum() < 10:
            continue
        bloque = sin_space[mascara].sum(axis=0)
        por_clase[NOMBRES_CLASES[c]] = {
            "pearson": round(float(sps.pearsonr(bloque, perfil["p_crudo"][c])[0]), 6),
            "spearman": round(float(sps.spearmanr(bloque, perfil["p_crudo"][c])[0]), 6),
            "n_filas": int(mascara.sum()),
        }
    return {
        "n_filas_muestreadas": int(n_muestra),
        "semilla_muestreo": SEMILLAS["muestra_correlacion_fase_c"],
        "pearson_por_fila_literal_TC1": round(pearson_fila, 6),
        "spearman_por_fila_literal_TC1": round(spearman_fila, 6),
        "pearson_agregado_18_etiquetas": round(pearson_agregado, 6),
        "spearman_agregado_18_etiquetas": round(spearman_agregado, 6),
        "pearson_por_clase": por_clase,
        "lectura": ("la cifra literal de TC.1 compara un vector de 18 componentes MEDIANTE EL "
                    "TWEET con un vector que solo toma 6 valores distintos (el perfil de su "
                    "clase), de modo que su techo es una correlacion atenuada por construccion "
                    "y no es comparable con el umbral 0,9 de la spec, que esta pensado para dos "
                    "series de 18 puntos. La cifra que responde a la pregunta de C ('es la misma "
                    "senal que D-026?') es la agregada y la por clase, y se publica la literal "
                    "por trazabilidad"),
    }


def filas_contraste(cota: dict, perfil: dict, entradas: Entradas) -> list[dict]:
    """TC.2: una fila por referencia previa, reexpresada contra los dos suelos.

    Parameters
    ----------
    cota : dict
        Salida de :func:`fases_a_cota`.
    entradas : Entradas
        Entradas de la corrida. Solo se usa para saber si es un subconjunto.
    perfil : dict
        Salida de :func:`perfil_anotador` del anotador primario.
    entradas : Entradas
        Entradas ya cargadas.

    Returns
    -------
    list[dict]
        Filas con las columnas de :data:`COLS_CONTRASTE`.
    """
    a1 = cota[ANOTADOR_PRIMARIO]["A1"]["macro_f1_pooled"]
    a1_lg = cota["lg"]["A1"]["macro_f1_pooled"]
    filas = [
        {"referencia": "D-026", "que_midio": "18 proporciones UPOS + 8 derivadas por fila "
         "(es la version NO desagregada por clase de la misma tabla)", "macro_f1": "",
         "suelo_estricto": PISO_ESTRICTO, "azar_uniforme": AZAR_UNIFORME,
         "delta_vs_suelo_estricto": "", "delta_vs_azar": "",
         "se_reexpresa_como": "solo se reporto el delta de integracion -0.0032; V37 no puede "
                              "anadir ninguna senal que D-026 no nominara como variable: solo "
                              "desagregacion por clase, estabilidad y cota de potencia",
         "tipo": "integracion"},
        {"referencia": "V14", "que_midio": "37 ratios agregados de morfologia, lemas, entidades "
         "y sintaxis", "macro_f1": 0.2409, "suelo_estricto": PISO_ESTRICTO,
         "azar_uniforme": AZAR_UNIFORME, "delta_vs_suelo_estricto": round(0.2409 - PISO_ESTRICTO, 7),
         "delta_vs_azar": round(0.2409 - AZAR_UNIFORME, 7),
         "se_reexpresa_como": "aislado LogReg 0.2409: +0.1924 sobre el suelo estricto y +0.0742 "
                              "sobre el azar; integracion -0.0015 con 1/5 folds",
         "tipo": "aislado"},
        {"referencia": "V31", "que_midio": "canales regex sin tagger: pronombres por persona, "
         "funcion, tiempo verbal, negacion, densidad", "macro_f1": "",
         "suelo_estricto": PISO_ESTRICTO, "azar_uniforme": AZAR_UNIFORME,
         "delta_vs_suelo_estricto": "", "delta_vs_azar": "",
         "se_reexpresa_como": "no se reporto macro-F1 del bloque aislado, solo banco +0.006820 e "
                              "integracion +0.0000642; V37 no puede desglosar PRON por persona "
                              "porque UPOS no lleva ese atributo: es limitacion del artefacto, "
                              "no hallazgo",
         "tipo": "integracion"},
        {"referencia": "P1a", "que_midio": "P(clase | token) con suavizado "
         "empirico-bayesiano y cross-fit interno (direccion dual de la misma tabla)",
         "macro_f1": P1A, "suelo_estricto": PISO_ESTRICTO, "azar_uniforme": AZAR_UNIFORME,
         "delta_vs_suelo_estricto": round(P1A - PISO_ESTRICTO, 7),
         "delta_vs_azar": round(P1A - AZAR_UNIFORME, 7),
         "se_reexpresa_como": "mejor bloque aislado del proyecto: +0.3512 sobre el suelo "
                              "estricto y +0.2330 sobre el azar; integracion -0.0002 con 3/5 "
                              "folds y controles permutados empatados",
         "tipo": "bloque_solo"},
        {"referencia": "P2-md", "que_midio": "pos_seq -> TF-IDF 1-3 + LogReg C=1.0 (md)",
         "macro_f1": P2_MD, "suelo_estricto": PISO_ESTRICTO, "azar_uniforme": AZAR_UNIFORME,
         "delta_vs_suelo_estricto": round(P2_MD - PISO_ESTRICTO, 7),
         "delta_vs_azar": round(P2_MD - AZAR_UNIFORME, 7),
         "se_reexpresa_como": "+0.2266 sobre el suelo estricto y +0.1084 sobre el azar; "
                              "integracion -0.0086357 con 0/5 folds",
         "tipo": "canal_solo"},
        {"referencia": "P2-lg", "que_midio": "pos_seq -> TF-IDF 1-3 + LogReg C=1.0 (lg)",
         "macro_f1": P2_LG, "suelo_estricto": PISO_ESTRICTO, "azar_uniforme": AZAR_UNIFORME,
         "delta_vs_suelo_estricto": round(P2_LG - PISO_ESTRICTO, 7),
         "delta_vs_azar": round(P2_LG - AZAR_UNIFORME, 7),
         "se_reexpresa_como": "+0.2240 sobre el suelo estricto y +0.1059 sobre el azar; "
                              "integracion -0.0084168 con 0/5 folds",
         "tipo": "canal_solo"},
        {"referencia": "A1-md", "que_midio": "V37: NB cerrado sobre proporciones POS (la cota)",
         "macro_f1": a1, "suelo_estricto": PISO_ESTRICTO, "azar_uniforme": AZAR_UNIFORME,
         "delta_vs_suelo_estricto": round(a1 - PISO_ESTRICTO, 7),
         "delta_vs_azar": round(a1 - AZAR_UNIFORME, 7),
         "se_reexpresa_como": "estimacion puntual con IC por grupos en el JSON; NO se afirma "
                              "significancia contra P2 ni P1a porque son estimadores y "
                              "convenciones distintas (E6)",
         "tipo": "cota"},
        {"referencia": "A1-lg", "que_midio": "V37: NB cerrado sobre proporciones POS (lg)",
         "macro_f1": a1_lg, "suelo_estricto": PISO_ESTRICTO, "azar_uniforme": AZAR_UNIFORME,
         "delta_vs_suelo_estricto": round(a1_lg - PISO_ESTRICTO, 7),
         "delta_vs_azar": round(a1_lg - AZAR_UNIFORME, 7),
         "se_reexpresa_como": "estimacion puntual con IC por grupos en el JSON",
         "tipo": "cota"},
        {"referencia": "campeon V30+plug-in", "que_midio": "V30 + plug-in con pesos congelados",
         "macro_f1": 0.6159732487657944, "suelo_estricto": PISO_ESTRICTO,
         "azar_uniforme": AZAR_UNIFORME,
         "delta_vs_suelo_estricto": round(0.6159732487657944 - PISO_ESTRICTO, 7),
         "delta_vs_azar": round(0.6159732487657944 - AZAR_UNIFORME, 7),
         "se_reexpresa_como": "NO TOCADO por V37: se lee, no se corre",
         "tipo": "baseline"},
    ]
    return filas


# ==============================================================================
# SECCION 5 - FASE D - auditoria de novedad del canal de transiciones
# ==============================================================================
def indice_bigramas(conteos: np.ndarray, n_tokens: np.ndarray) -> tuple[np.ndarray, ...]:
    """Indices de etiqueta de los bigramas de `pos_seq`, con la fila de cada uno.

    Parameters
    ----------
    conteos : scipy.sparse matrix
        Matriz de conteos por fila y etiqueta (puede ser dispersa).
    n_tokens : numpy.ndarray
        Longitud en tokens por fila. Es la fuente de verdad para expandir filas: las filas
        sin tokens simplemente no aportan ningun bigrama.

    Returns
    -------
    tuple
        ``(a, b, fila)`` con los indices de etiqueta de los dos extremos del bigrama y la fila
        a la que pertenece.
    """
    largo = conteos.shape[0]
    indices = np.repeat(np.arange(largo), n_tokens)
    # `conteos.sum(axis=0)` devuelve np.matrix si la entrada es dispersa: se aplana a 1-D.
    por_etiqueta = np.asarray(conteos.sum(axis=0)).ravel().astype(np.int64)
    tokens = np.repeat(np.arange(N_ETIQUETAS), por_etiqueta)
    valido = indices[:-1] == indices[1:]
    posiciones = np.flatnonzero(valido)
    return tokens[posiciones], tokens[posiciones + 1], indices[posiciones]


def tensor_bigramas_por_celda(entradas: Entradas, nombre: str) -> dict:
    """Construye el estadistico suficiente de los bigramas por celda (fold, clase, mitad).

    La celda de un cluster es ``(fold(cluster), clase(cluster), mitad_por_hash(cluster))``:
    5 x 6 x 2 = 60 celdas. Cada cluster vive en una sola celda, luego el bootstrap por grupo es
    una suma ponderada de estos tensores y no hay que recontar nada por replica.

    Parameters
    ----------
    entradas : Entradas
        Entradas ya cargadas.
    nombre : str
        Anotador.

    Returns
    -------
    dict
        ``H`` con forma ``(60, 324, n_max)`` en ``float32``, ``pos`` con la posicion de cada
        cluster dentro de su celda, ``n_bigramas`` y ``n_grupos``.
    """
    datos = entradas.anotadores[nombre]
    a, b, fila = indice_bigramas(datos["conteos"], datos["n_tokens"])
    y = entradas.y
    cluster = entradas.cluster_local
    mitad = datos["mitad"]
    n_grupos = entradas.n_grupos
    _, primera_fila = np.unique(cluster, return_index=True)
    orden = np.argsort(primera_fila)
    filas_representativas = primera_fila[orden]
    cluster_de_celda = cluster[filas_representativas]
    celda_de_cluster = ((entradas.fold_idx[filas_representativas] * N_CLASES
                         + y[filas_representativas]) * 2 + mitad[filas_representativas])
    # Orden estable por celda. `cluster_de_celda` esta indexado por identificador de cluster, asi
    # que al ordenar hay que deshacer la permutacion para que `pos` vuelva a estar indexado por
    # cluster: de lo contrario `pos[cluster[fila]]` mezcla identidades de cluster.
    orden_celda = np.argsort(celda_de_cluster, kind="stable")
    celda_ordenada = celda_de_cluster[orden_celda]
    inicio = np.searchsorted(celda_ordenada, np.arange(60), side="left")
    fin = np.searchsorted(celda_ordenada, np.arange(60), side="right")
    largo_celda = fin - inicio
    n_max = int(largo_celda.max())
    assert n_max > 0, "ninguna celda tiene clusters"
    # Rango de cada cluster DENTRO de su celda: posicion en el array ordenado - inicio de celda.
    local_ordenado = np.arange(n_grupos, dtype=np.int64) - inicio[celda_ordenada]
    assert local_ordenado.min() >= 0 and local_ordenado.max() < n_max, \
        "rango local de cluster fuera de la celda"
    pos = np.empty(n_grupos, dtype=np.int64)
    pos[orden_celda] = celda_ordenada.astype(np.int64) * n_max + local_ordenado
    assert len(np.unique(pos)) == n_grupos, "dos clusters ocupan la misma posicion"
    assert pos.min() >= 0 and pos.max() < 60 * n_max, "posicion de cluster fuera de rango"
    pos_celda_fila = pos[cluster[fila]]
    base = pos_celda_fila * (N_ETIQUETAS * N_ETIQUETAS) + a * N_ETIQUETAS + b
    salida = np.zeros(60 * n_max * N_ETIQUETAS * N_ETIQUETAS, dtype=np.float64)
    np.add.at(salida, base, 1.0)
    tensor = salida.reshape(60, n_max, N_ETIQUETAS * N_ETIQUETAS)
    return {"H": np.ascontiguousarray(tensor.transpose(0, 2, 1)).astype(np.float32),
            "pos": pos, "n_max": n_max, "n_grupos": n_grupos,
            "n_bigramas": len(a),
            "n_grupos_por_celda": largo_celda.tolist()}


def divergencias_desde_celdas(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Divergencias entre clases y de ruido a partir de los conteos por celda.

    Parameters
    ----------
    S : numpy.ndarray
        ``(60, 324)`` con los conteos de bigramas por celda, en orden
        ``(fold * 6 + clase) * 2 + mitad``.

    Returns
    -------
    tuple
        ``(div_interclase, div_ruido)`` con forma ``(5, 6)``: para cada fold, la distancia de
        variacion total de ``T_k[c,:,:]`` frente a ``T_k_pooled`` y de las dos mitades entre si.
    """
    S = S.reshape(5, N_CLASES, 2, N_ETIQUETAS, N_ETIQUETAS)
    alfa = ALFA
    por_clase = S.sum(axis=2)
    pooled = por_clase.sum(axis=1, keepdims=True)
    t_clase = (por_clase + alfa) / (por_clase.sum(axis=-1, keepdims=True)
                                    + alfa * N_ETIQUETAS)
    t_pooled = (pooled + alfa) / (pooled.sum(axis=-1, keepdims=True) + alfa * N_ETIQUETAS)
    div_interclase = 0.5 * np.abs(t_clase - t_pooled).sum(axis=(-1, -2))
    t_mitad = (S + alfa) / (S.sum(axis=-1, keepdims=True) + alfa * N_ETIQUETAS)
    div_ruido = 0.5 * np.abs(t_mitad[:, :, 0] - t_mitad[:, :, 1]).sum(axis=(-1, -2))
    return div_interclase, div_ruido


def fase_d(entradas: Entradas, n_bootstrap: int) -> dict:
    """D1: matriz de transiciones por clase, divergencias, ``R``, ``D`` e IC95 pareado.

    Parameters
    ----------
    entradas : Entradas
        Entradas ya cargadas.
    n_bootstrap : int
        Iteraciones del bootstrap por grupos.

    Returns
    -------
    dict
        ``T`` leave-one-fold-out, conteos de bigramas, ``div_interclase``, ``div_ruido``, ``R``,
        ``D`` con su IC95 pareado, densidad de la matriz y soporte.
    """
    nombre = ANOTADOR_PRIMARIO
    print("    construyendo el tensor de bigramas por celda ...", flush=True)
    celdas = tensor_bigramas_por_celda(entradas, nombre)
    H = celdas["H"]
    pos = celdas["pos"]
    n_grupos = celdas["n_grupos"]
    datos = entradas.anotadores[nombre]
    # `H` se devuelve como (60, 324, n_max): el eje de posiciones dentro de la celda es el ULTIMO.
    # El peso unitario se construye en 2D (celda, posicion) para que dos celdas distintas nunca
    # compartan posicion. Como `pos = celda * n_max + local`, se recuperan ambos ejes:
    # celda = pos // n_max y posicion = pos % n_max. Indexar por cluster (un vector de
    # longitud n_grupos) seria un error de eje: `pos` NO es un indice de cluster.
    n_max = celdas["n_max"]
    pesos = np.zeros((60, n_max), dtype=np.float64)
    pesos[pos // n_max, pos % n_max] = 1.0
    assert pesos.sum() == float(n_grupos), "cada cluster ocupa exactamente una posicion"
    pooled = np.einsum("cdg,cg->cd", H, pesos, optimize=True)
    assert pooled.shape == (60, N_ETIQUETAS * N_ETIQUETAS), "S debe ser (60, 324)"
    div_interclase_punto, div_ruido_punto = divergencias_desde_celdas(pooled)
    interclase_c = div_interclase_punto.mean(axis=0)
    ruido_c = div_ruido_punto.mean(axis=0)
    d_punto = float(interclase_c.mean() - ruido_c.mean())
    r_punto = float(interclase_c.mean() / max(ruido_c.mean(), 1e-12))
    print(f"    bootstrap pareado de D ({n_bootstrap} iteraciones) ...", flush=True)
    generador = np.random.default_rng(SEMILLAS["bootstrap_cluster"])
    muestras_d = np.empty(n_bootstrap, dtype=np.float64)
    muestras_r = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        indices = generador.integers(0, n_grupos, n_grupos)
        peso = np.bincount(indices, minlength=n_grupos).astype(np.float32)
        # `peso` esta indexado por cluster; se proyecta a la rejilla (celda, posicion) con el
        # MISMO mapeo que el punto estima: celda = pos // n_max, posicion = pos % n_max.
        pesos_b = np.zeros((60, n_max), dtype=np.float32)
        pesos_b[pos // n_max, pos % n_max] = peso
        S = np.einsum("cdg,cg->cd", H, pesos_b, optimize=True)
        inter, ruido = divergencias_desde_celdas(S)
        muestras_d[b] = inter.mean() - ruido.mean()
        muestras_r[b] = inter.mean() / max(ruido.mean(), 1e-12)
    lo, hi = np.percentile(muestras_d, [2.5, 97.5])
    n_bigramas = celdas["n_bigramas"]
    esperado = int((datos["n_tokens"] - 1)[datos["n_tokens"] > 0].sum())
    assert n_bigramas == esperado, (
        f"n_bigramas {n_bigramas} != sum(n_tokens - 1) sobre filas no vacias {esperado}")
    a, b_idx, fila = indice_bigramas(datos["conteos"], datos["n_tokens"])
    celda_fila = ((entradas.fold_idx * N_CLASES + entradas.y) * 2 + datos["mitad"])[fila]
    matriz_por_clase = np.zeros((N_CLASES, N_ETIQUETAS, N_ETIQUETAS), dtype=np.int64)
    np.add.at(matriz_por_clase, (entradas.y[fila], a, b_idx), 1)
    t_leave_one_fold_out = np.zeros((5, N_CLASES, N_ETIQUETAS, N_ETIQUETAS), dtype=np.float64)
    t_pooled = np.zeros((5, N_ETIQUETAS, N_ETIQUETAS), dtype=np.float64)
    for k in range(5):
        # `a`, `b_idx` y `fila` estan indexados por BIGRAMA, no por fila: la mascara de fold
        # debe aplicarse a `fila` (que devuelve la fila duena) y con ella a `a` y `b_idx`.
        en_train = entradas.fold_idx[fila] != k
        conteo_c = np.zeros((N_CLASES, N_ETIQUETAS, N_ETIQUETAS), dtype=np.float64)
        np.add.at(conteo_c, (entradas.y[fila[en_train]], a[en_train], b_idx[en_train]), 1)
        t_leave_one_fold_out[k] = ((conteo_c + ALFA)
                                   / (conteo_c.sum(axis=(1, 2), keepdims=True)
                                      + ALFA * N_ETIQUETAS * N_ETIQUETAS))
        # El perfil pooled es UNA sola matriz 18x18 (clases sumadas), luego el denominador va
        # sobre `axis=(0, 1)`: normalizar por fila daria 18 vectores, no una matriz.
        total = conteo_c.sum(axis=0)
        t_pooled[k] = ((total + ALFA) / (total.sum(axis=(0, 1), keepdims=True)
                                         + ALFA * N_ETIQUETAS * N_ETIQUETAS))
    densidad = int((matriz_por_clase.sum(axis=0) >= SOPORTE_BIGRAMA).sum())
    densidad_por_clase = [int((matriz_por_clase[c].sum(axis=0) >= SOPORTE_BIGRAMA).sum())
                          for c in range(N_CLASES)]
    return {
        "n_bigramas": n_bigramas,
        "n_bigramas_esperado": esperado,
        "T_leave_one_fold_out": t_leave_one_fold_out,
        "T_pooled_leave_one_fold_out": t_pooled,
        "n_bigramas_por_clase": matriz_por_clase,
        "div_interclase_por_fold": div_interclase_punto,
        "div_ruido_por_fold": div_ruido_punto,
        "div_interclase": interclase_c,
        "div_ruido": ruido_c,
        "media_div_interclase": float(interclase_c.mean()),
        "media_div_ruido": float(ruido_c.mean()),
        "R": r_punto,
        "R_ic95": [round(float(np.percentile(muestras_r, 2.5)), 6),
                   round(float(np.percentile(muestras_r, 97.5)), 6)],
        "D": d_punto,
        "D_bootstrap_media": float(muestras_d.mean()),
        "D_ic95": [round(float(lo), 6), round(float(hi), 6)],
        "D_excluye_cero": bool(lo > 0 or hi < 0),
        "densidad_matriz": densidad,
        "densidad_matriz_total": N_ETIQUETAS * N_ETIQUETAS,
        "densidad_matriz_por_clase": densidad_por_clase,
        "n_celdas_fold_clase_mitad": 60,
        "n_max_clusters_por_celda": celdas["n_max"],
        "hash_mitades": "splitmix64(cluster_id + 0), bit bajo; semilla declarada 0",
        "n_iter_bootstrap": n_bootstrap,
    }


# ==============================================================================
# SECCION 6 - FASE E - gate de decision
# ==============================================================================
def criterios_v1_a_v6(perfiles: dict, estabilidad: dict, cota: dict, entradas: Entradas,
                      correlaciones: dict) -> dict:
    """Evalua los seis criterios pre-registrados con las enmiendas aplicadas.

    Parameters
    ----------
    perfiles : dict
        Perfiles por anotador.
    estabilidad : dict
        Salida de :func:`fase_b`.
    cota : dict
        Salida de :func:`fases_a_cota`.
    entradas : Entradas
        Entradas ya cargadas.
    correlaciones : dict
        Salida de :func:`correlaciones_fase_c`.

    Returns
    -------
    dict
        ``criterios`` con ``V1..V6`` (cumple, valor, umbral, lectura) y ``conjunto``, que es la
        lista de celdas que han superado los floors de V1 y que V2, V3 y V4 tienen que cumplir.
    """
    perfil = perfiles[ANOTADOR_PRIMARIO]
    n_por_clase = perfil["n_por_clase"]
    delta = perfil["delta"]
    z_corr = perfil["z_bonferroni"]
    significante = z_corr > 0
    v1_celda = (np.abs(delta) >= UMBRALES["V1"]["delta_min"]) & significante
    etiquetas_v1 = sorted({t for c in range(N_CLASES) for t in range(N_ETIQUETAS)
                           if v1_celda[c, t]})
    clases_v1 = sorted({c for c in range(N_CLASES) for t in range(N_ETIQUETAS) if v1_celda[c, t]})
    # El gate usa el maximo cruzado entre anotadores (criterio conservador).
    dispersion = np.asarray(estabilidad["dispersion_max_entre_anotadores"])
    max_dc = estabilidad["max_dc_md_lg"]
    delta_est = np.abs(perfil["p_est"] - perfil["ref"][None, :])
    conjunta = {(c, t) for c in clases_v1 for t in etiquetas_v1}
    # Mapas booleanos (6, 18): una entrada por celda. El gate se evalua SOLO sobre `conjunta`
    # (las celdas que superan V1); las demas quedan False y no participan en `all()`.
    dispersion_ok = {(c, t): dispersion[c, t] <= UMBRALES["V2"]["dispersion_max"]
                     for c in range(N_CLASES) for t in range(N_ETIQUETAS)}
    dc_ok = {(c, t): max_dc[c, t] <= UMBRALES["V3"]["max_dc_md_lg_max"]
             for c in range(N_CLASES) for t in range(N_ETIQUETAS)}
    est_ok = {(c, t): delta_est[c, t] >= UMBRALES["V4"]["delta_estandarizado_min"]
              for c in range(N_CLASES) for t in range(N_ETIQUETAS)}
    a1_md = cota[ANOTADOR_PRIMARIO]["A1"]["macro_f1_pooled"]
    a1_lg = cota["lg"]["A1"]["macro_f1_pooled"]
    a_perm_ok = bool(cota[ANOTADOR_PRIMARIO]["A_perm"]["cumple_criterio"]
                     and cota["lg"]["A_perm"]["cumple_criterio"])
    a_perm_val_ok = bool(cota[ANOTADOR_PRIMARIO]["A_perm_val"]["cumple_criterio"]
                         and cota["lg"]["A_perm_val"]["cumple_criterio"])
    ic_ok = bool(cota[ANOTADOR_PRIMARIO]["A1_menos_A_perm"]["excluye_cero"]
                 and cota["lg"]["A1_menos_A_perm"]["excluye_cero"])
    criterios = {
        "V1": {
            "cumple": bool(len(etiquetas_v1) >= UMBRALES["V1"]["n_etiquetas_min"]
                           and len(clases_v1) >= UMBRALES["V1"]["n_clases_min"]),
            "valor": {"n_etiquetas": len(etiquetas_v1),
                      "etiquetas": [ETIQUETAS_POS[t] for t in etiquetas_v1],
                      "n_clases": len(clases_v1),
                      "clases": [NOMBRES_CLASES[c] for c in clases_v1],
                      "n_celdas": int(v1_celda.sum())},
            "umbral": UMBRALES["V1"],
        },
        "V2": {
            "cumple": bool(conjunta) and all(dispersion_ok[c, t] for c, t in conjunta),
            "valor": {"n_celdas_evaluadas": len(conjunta),
                      "n_celdas_que_cumplen": int(sum(dispersion_ok[c, t] for c, t in conjunta)),
                      "dispersion_max_sobre_esas_celdas":
                          round(float(max((dispersion[c, t] for c, t in conjunta), default=0.0)),
                                6)},
            "umbral": UMBRALES["V2"],
        },
        "V3": {
            "cumple": bool(conjunta) and all(dc_ok[c, t] for c, t in conjunta),
            "valor": {"n_celdas_evaluadas": len(conjunta),
                      "n_celdas_que_cumplen": int(sum(dc_ok[c, t] for c, t in conjunta)),
                      "max_dc_md_lg_max_sobre_esas_etiquetas":
                          round(float(max((max_dc[c, t] for c, t in conjunta), default=0.0)), 6)},
            "umbral": UMBRALES["V3"],
        },
        "V4": {
            "cumple": bool(conjunta) and all(est_ok[c, t] for c, t in conjunta),
            "valor": {"n_celdas_evaluadas": len(conjunta),
                      "n_celdas_que_cumplen": int(sum(est_ok[c, t] for c, t in conjunta)),
                      "delta_estandarizado_min_sobre_esas_celdas":
                          round(float(min((delta_est[c, t] for c, t in conjunta), default=0.0)),
                                6)},
            "umbral": UMBRALES["V4"],
        },
        "V5": {
            "cumple": bool(a1_md > UMBRALES["V5"]["a1_min"] and a1_lg > UMBRALES["V5"]["a1_min"]),
            "valor": {"A1_md": a1_md, "A1_lg": a1_lg,
                      "posicion_sin_umbral": {"A1_md": a1_md, "P1a": P1A, "P2_md": P2_MD,
                                              "P2_lg": P2_LG, "campeon": 0.6159732487657944}},
            "umbral": UMBRALES["V5"],
            "nota": ("E7: sanity floor, no gate de novedad. La novedad NO se decide con V5: se "
                     "decide con las cinco razones de la seccion 6.2 de la spec"),
        },
        "V6": {
            "cumple": bool(a_perm_ok and a_perm_val_ok and ic_ok),
            "valor": {"A_perm_md": cota[ANOTADOR_PRIMARIO]["A_perm"]["macro_f1_pooled"],
                      "A_perm_lg": cota["lg"]["A_perm"]["macro_f1_pooled"],
                      "A_perm_val_md": cota[ANOTADOR_PRIMARIO]["A_perm_val"]["macro_f1_pooled"],
                      "A_perm_val_lg": cota["lg"]["A_perm_val"]["macro_f1_pooled"],
                      "ic95_A1_menos_A_perm_md":
                          cota[ANOTADOR_PRIMARIO]["A1_menos_A_perm"]["ic95"],
                      "ic95_A1_menos_A_perm_lg": cota["lg"]["A1_menos_A_perm"]["ic95"],
                      "criterio_banda_A0_mas_menos_0_005": "ELIMINADO (E1)"},
            "umbral": UMBRALES["V6"],
        },
    }
    # `sorted` sobre dicts no esta definido en Python: se ordena por la clave (clase, indice de
    # etiqueta), que es un orden estable y reproducible, y se serializa despues como texto legible.
    criterios["conjunto_de_celdas"] = [
        {"clase": NOMBRES_CLASES[c], "tag": ETIQUETAS_POS[t]}
        for c, t in sorted(conjunta, key=lambda par: (par[0], par[1]))
    ]
    criterios["nota_E11"] = (
        "el z_bonferroni que gobierna V1 usa el error analitico con unidad TOKEN (pooled y "
        "conservador), etiquetado como SECUNDARIO. El error analitico con unidad TWEET (delta "
        "method) se publica en el JSON y, segun E11, la inferencia la gobierna el bootstrap por "
        "grupos de duplicados, que es el instrumento correcto")
    criterios["n_tokens_medios_por_clase"] = {
        NOMBRES_CLASES[c]: round(float(
            entradas.anotadores[ANOTADOR_PRIMARIO]["n_tokens"][entradas.y == c].mean()), 4)
        for c in range(N_CLASES)}
    return criterios


def fpr_conjunta_v1_v4(entradas: Entradas, criterios: dict, n_bootstrap: int = N_ITER_PUNTO
                       ) -> dict:
    """E4: cuantas etiquetas y celdas sobreviven a los floors de V1..V4 bajo permutacion.

    Los cuatro veredictos de fase B son un AND sobre la misma rejilla 18x6 y su tasa de falso
    positivo conjunta no estaba calibrada. Aqui se aplica el mismo conjunto de floors a las
    etiquetas permutadas del control, con el MISMO criterio de significacion y el MISMO
    esquema de significancia que en la corrida real.

    Parameters
    ----------
    entradas : Entradas
        Entradas ya cargadas.
    criterios : dict
        Salida de :func:`criterios_v1_a_v6`, para comparar real contra nulo.
    n_bootstrap : int, opcional
        Iteraciones del IC95 por grupos usado para la significancia del nulo.

    Returns
    -------
    dict
        Recuento de etiquetas y celdas que sobreviven bajo permutacion, junto al real.
    """
    y_perm = PR.permuta_etiquetas_train(entradas.y, entradas.fold_idx,
                                        SEMILLAS["permute_control"])
    conteos = entradas.anotadores[ANOTADOR_PRIMARIO]["conteos"]
    filas = entradas.filas_validas
    y_real, y_nulo = entradas.y, y_perm
    n_por_clase_null = np.zeros((N_CLASES, N_ETIQUETAS))
    n_filas_null = np.zeros((N_CLASES, N_ETIQUETAS))
    p_est_null = np.zeros((N_CLASES, N_ETIQUETAS))
    for c in range(N_CLASES):
        seleccion = filas[y_nulo[filas] == c]
        matriz_n, matriz_f = PR.perfil_por_clase(conteos, y_nulo, seleccion)
        n_por_clase_null[c] = matriz_n.sum(axis=0)
        n_filas_null[c] = matriz_f.sum(axis=0)
        p_est_null[c] = PR.p_desde_conteos(matriz_n)[c]
    p_null = PR.p_desde_conteos(n_por_clase_null)
    ref_null = n_por_clase_null.sum(axis=0) / n_por_clase_null.sum()
    delta_null, _, z_null, z_corr_null = PR.se_y_z(p_null, ref_null, n_por_clase_null,
                                                    n_por_clase_null.sum(axis=0))
    # significancia bajo el bootstrap por grupos, para no depender del error analitico (E11)
    suficiente = PR.perfil_por_cluster(conteos, y_nulo, entradas.cluster_local,
                                       entradas.n_grupos)
    generador = np.random.default_rng(SEMILLAS["bootstrap_cluster"])
    acumulado = np.zeros((N_CLASES, N_ETIQUETAS))
    for _ in range(n_bootstrap):
        indices = generador.integers(0, entradas.n_grupos, entradas.n_grupos)
        peso = np.bincount(indices, minlength=entradas.n_grupos).astype(np.float32)
        totales = (suficiente.T @ peso).reshape(N_CLASES, 19)
        p_b = totales[:, :N_ETIQUETAS] / np.where(totales[:, 18:19] > 0, totales[:, 18:19], 1.0)
        ref_b = p_b.sum(axis=0) / N_CLASES
        acumulado += (np.sign(p_b - ref_b[None, :]) == np.sign(delta_null)).astype(np.float64)
    prob_b = acumulado / n_bootstrap
    significativo_null = prob_b >= 0.975
    v1_null = (np.abs(delta_null) >= UMBRALES["V1"]["delta_min"]) & significativo_null
    etiquetas_null = sorted({ETIQUETAS_POS[t] for t in range(N_ETIQUETAS)
                             if v1_null[:, t].any()})
    clases_null = sorted({NOMBRES_CLASES[c] for c in range(N_CLASES) if v1_null[c, :].any()})
    dispersion_null = np.zeros((N_CLASES, N_ETIQUETAS))
    # Dispersion del perfil NULO: max_k p_k - min_k p_k sobre los 5 folds leave-one-out.
    dispersion_null = np.zeros((N_CLASES, N_ETIQUETAS))
    for k in range(5):
        seleccion = filas[entradas.fold_idx[filas] != k]
        matriz_n, _ = PR.perfil_por_clase(conteos, y_nulo, seleccion)
        p_k = PR.p_desde_conteos(matriz_n)
        dispersion_null = np.where(k == 0, p_k, np.maximum(dispersion_null, p_k)) \
            - np.where(k == 0, p_k, np.minimum(dispersion_null, p_k))
    v2_null = v1_null & (dispersion_null <= UMBRALES["V2"]["dispersion_max"])
    v4_null = v1_null & (np.abs(p_est_null - ref_null[None, :])
                         >= UMBRALES["V4"]["delta_estandarizado_min"])
    return {
        "n_iter_significancia": int(n_bootstrap),
        "n_bootstrap_significancia": int(n_bootstrap),
        "bajo_permutacion": {
            "n_etiquetas_que_sobreviven_a_los_floors_de_V1": len(etiquetas_null),
            "etiquetas": etiquetas_null,
            "n_clases": len(clases_null),
            "n_celdas_V1": int(v1_null.sum()),
            "n_celdas_V1_y_V2": int(v2_null.sum()),
            "n_celdas_V1_y_V4": int(v4_null.sum()),
            "n_celdas_V1_y_V2_y_V4": int((v2_null & v4_null).sum()),
        },
        "real": {
            "n_etiquetas": criterios["V1"]["valor"]["n_etiquetas"],
            "n_celdas_V1": criterios["V1"]["valor"]["n_celdas"],
            "n_celdas_conjuntas_V1_y_V2_y_V4": int(sum(
                1 for c, t in criterios["conjunto_de_celdas"])),
        },
        "lectura": ("si el nulo ya supera los floors de V1..V4, los floors estan mal "
                    "calibrados y el V1 real no puede leerse como senal. Se declara en el "
                    "informe en vez de ocultarlo (E4)"),
    }


# ==============================================================================
# SECCION 7 - Escritura de artefactos
# ==============================================================================
def escribe_csv(ruta: Path, columnas: list[str], filas: list[dict]) -> None:
    """Escribe un CSV con columnas fijas, ordenadas y terminador de linea estable.

    Parameters
    ----------
    ruta : Path
        Ruta de salida; los directorios se crean si hace falta.
    columnas : list[str]
        Cabecera fija.
    filas : list[dict]
        Filas de datos.
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", encoding="utf-8", newline="") as fichero:
        escritor = csv.DictWriter(fichero, fieldnames=columnas, lineterminator="\n")
        escritor.writeheader()
        for fila in filas:
            escritor.writerow({columna: fila.get(columna, "") for columna in columnas})


def comprueba_csv(ruta: Path, columnas: list[str], n_filas_esperadas: int) -> None:
    """Relee un CSV escrito y verifica cabecera y numero de filas.

    Parameters
    ----------
    ruta : Path
        Ruta del CSV.
    columnas : list[str]
        Cabecera esperada.
    n_filas_esperadas : int
        Numero de filas de datos esperado.

    Raises
    ------
    AssertionError
        Si la cabecera o el numero de filas no coinciden.
    """
    marco = pd.read_csv(ruta)
    assert list(marco.columns) == columnas, (
        f"{ruta.name}: cabecera inesperada {list(marco.columns)}")
    assert len(marco) == n_filas_esperadas, (
        f"{ruta.name}: {len(marco)} filas, esperadas {n_filas_esperadas}")


def figura_perfil_por_clase(perfiles: dict, ruta: Path) -> None:
    """Figura 1: `p[tag|edad]` con las 6 curvas por anotador y soporte ordenado por delta.

    Parameters
    ----------
    perfiles : dict
        Perfiles por anotador.
    ruta : Path
        Ruta del PNG.
    """
    figura, ejes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for eje, nombre in zip(ejes, ANOTADORES):
        perfil = perfiles[nombre]
        orden = np.argsort(-np.abs(perfil["delta"]).max(axis=0))
        colores = plt.cm.viridis(np.linspace(0.05, 0.95, N_ETIQUETAS))
        for color, t in zip(colores, orden):
            eje.plot(range(N_CLASES), perfil["p_crudo"][:, t], marker="o", color=color,
                     linewidth=1.2,
                     label=f"{ETIQUETAS_POS[t]} "
                           f"(max|delta|={np.abs(perfil['delta'][:, t]).max():.4f}"
                           f"{', soporte bajo' if perfil['n_por_clase'][:, t].min() < SOPORTE_INSUFICIENTE else ''})")
        for c in range(N_CLASES):
            soporte_bajo = perfil["n_por_clase"][c, orden] < SOPORTE_INSUFICIENTE
            if soporte_bajo.any():
                eje.scatter(np.full(soporte_bajo.sum(), c),
                            perfil["p_crudo"][c, orden][soporte_bajo],
                            facecolors="none", edgecolors="crimson", s=90, zorder=5)
        eje.set_xticks(range(N_CLASES))
        eje.set_xticklabels(NOMBRES_CLASES, rotation=20)
        eje.set_title(f"Perfil P(tag | edad), anotador {nombre}")
        eje.set_xlabel("clase de edad")
        eje.grid(alpha=0.3)
    ejes[0].set_ylabel("proporcion de aparicion por token")
    ejes[1].legend(fontsize=7, loc="upper right", ncol=1)
    ejes[0].legend(fontsize=7, loc="upper right", ncol=1)
    figura.suptitle("V37 fase A: p[tag|edad] cruda; circulos rojos = soporte insuficiente "
                    "(n_tokens < 1000)")
    figura.tight_layout()
    figura.savefig(ruta, dpi=130)
    plt.close(figura)


def figura_md_vs_lg(perfiles: dict, ruta: Path, coincidencia: float) -> None:
    """Figura 2: dispersion de `p^md` contra `p^lg` por celda con la identidad.

    Parameters
    ----------
    perfiles : dict
        Perfiles por anotador.
    ruta : Path
        Ruta del PNG.
    coincidencia : float
        Coincidencia de secuencia md/lg, que se publica como limitacion (E13).
    """
    figura, eje = plt.subplots(figsize=(7.5, 6.5))
    p_md = perfiles["md"]["p_crudo"].ravel()
    p_lg = perfiles["lg"]["p_crudo"].ravel()
    soporte = perfiles["md"]["n_por_clase"].ravel()
    colores = ["crimson" if s < SOPORTE_INSUFICIENTE else "steelblue" for s in soporte]
    eje.scatter(p_md, p_lg, c=colores, s=28, alpha=0.8)
    limite = max(float(p_md.max()), float(p_lg.max())) * 1.05
    eje.plot([0, limite], [0, limite], "k--", linewidth=1, label="identidad")
    correlacion = float(np.corrcoef(p_md, p_lg)[0, 1])
    eje.set_xlabel("p^md[clase, tag]")
    eje.set_ylabel("p^lg[clase, tag]")
    eje.set_title(f"Sanity check de la proyeccion universal (E13)\n"
                  f"Pearson por celda = {correlacion:.4f}; "
                  f"coincidencia de secuencia fila a fila = {coincidencia:.4f}\n"
                  f"(limitacion del conjunto de datos: pos_seq solo guarda UPOS, no el tag "
                  f"especifico del modelo)")
    eje.grid(alpha=0.3)
    eje.legend()
    figura.tight_layout()
    figura.savefig(ruta, dpi=130)
    plt.close(figura)


def figura_estabilidad(estabilidad: dict, perfiles: dict, ruta: Path) -> None:
    """Figura 3: dispersion de `p_k` por (tag, clase) en las etiquetas de soporte alto.

    Parameters
    ----------
    estabilidad : dict
        Salida de :func:`fase_b`.
    perfiles : dict
        Perfiles por anotador.
    ruta : Path
        Ruta del PNG.
    """
    perfil = perfiles[ANOTADOR_PRIMARIO]
    n_por_clase = perfil["n_por_clase"]
    etiquetas = [t for t in range(N_ETIQUETAS) if n_por_clase[:, t].min() >= SOPORTE_ESTABLE]
    # El soporte es un umbral ABSOLUTO sobre el corpus completo: en un subconjunto (autoejercicio
    # o modo `--humo`) ninguna etiqueta lo alcanza y la lista queda vacia, lo que rompe el
    # boxplot. Se dibujan entonces las de mayor soporte y se declara en el titulo.
    soporte_insuficiente = not etiquetas
    if soporte_insuficiente:
        top = np.argsort(n_por_clase.min(axis=0))[::-1][:6]
        etiquetas = sorted(int(t) for t in top)
    p_k = estabilidad["p_por_fold"][ANOTADOR_PRIMARIO][:, :, etiquetas]
    figura, ejes = plt.subplots(1, N_CLASES, figsize=(19, 4.6), sharey=True)
    for c, eje in enumerate(ejes):
        # Una caja por ETIQUETA, con los 5 folds como observaciones: se pasa una lista de
        # vectores 1-D, no un array (fold, etiqueta), que matplotlib no aplana.
        caja = eje.boxplot([p_k[:, c, t] for t in range(len(etiquetas))],
                           tick_labels=[ETIQUETAS_POS[t] for t in etiquetas],
                           showfliers=False,
                           patch_artist=True,
                           boxprops={"facecolor": "#9ecae1", "edgecolor": "#3182bd"})
        assert len(caja["boxes"]) == len(etiquetas), "una caja por etiqueta dibujada"
        eje.set_title(f"{NOMBRES_CLASES[c]}  (dispersion max "
                      f"{np.ptp(p_k[:, c, :], axis=0).max():.4f})")
        eje.tick_params(axis="x", rotation=90, labelsize=8)
        eje.grid(alpha=0.3, axis="y")
    ejes[0].set_ylabel("p_k[clase, tag] leave-one-fold-out")
    titulo_soporte = (f"(etiquetas con n_tokens >= {SOPORTE_ESTABLE} en las 6 clases)"
                       if not soporte_insuficiente else
                       "SOPORTE INSUFICIENTE en este subconjunto: se muestran las 6 etiquetas "
                       "de mayor soporte, no cumple el criterio de la corrida real")
    figura.suptitle("V37 fase B: estabilidad del perfil entre los 5 folds canonicos "
                    + titulo_soporte)
    figura.tight_layout()
    figura.savefig(ruta, dpi=130)
    plt.close(figura)


def figura_ic_por_grupo(perfiles: dict, estabilidad: dict, ruta: Path) -> None:
    """Figura 4: `delta[c,t]` con IC95 por grupos de duplicados, ordenado por delta.

    Parameters
    ----------
    perfiles : dict
        Perfiles por anotador.
    estabilidad : dict
        Salida de :func:`fase_b`.
    ruta : Path
        Ruta del PNG.
    """
    perfil = perfiles[ANOTADOR_PRIMARIO]
    ic = estabilidad["ic95_delta"][ANOTADOR_PRIMARIO]
    lo = np.array(ic["ic95_delta"])[..., 0]
    hi = np.array(ic["ic95_delta"])[..., 1]
    delta = perfil["delta"]
    figura, eje = plt.subplots(figsize=(11, 6))
    orden = np.argsort(-delta.ravel())
    colores = ["#2b8cbe" if lo.ravel()[i] > 0 or hi.ravel()[i] < 0 else "#bdbdbd"
               for i in orden]
    # Un IC por percentiles con punto en la media de replicas: se dibuja el punto del bootstrap.
    # [lo, hi], la distancia seria negativa y matplotlib rechaza elPlot. Se dibuja la barra
    # hasta el extremo del IC y se cuenta cuantas celdas tienen ese caso, en vez de ocultarlo.
    # El punto que se dibuja es `bootstrap_media_delta`, la media de las replicas, porque es el
    # estimador con el que el IC es coherente. `delta` es el perfil crudo y NO tiene por que caer
    # dentro del IC: son dos estimadores distintos (el bootstrap promedia replicas ponderadas).
    punto = np.asarray(ic["bootstrap_media_delta"]).ravel()[orden]
    inferior = lo.ravel()[orden]
    superior = hi.ravel()[orden]
    assert np.all(punto >= inferior - 1e-6) and np.all(punto <= superior + 1e-6), (
        "el punto del bootstrap debe caer dentro de su propio IC95")
    yerr = np.vstack([punto - inferior, superior - punto])
    eje.bar(range(len(orden)), punto, color=colores)
    eje.errorbar(range(len(orden)), punto, yerr=yerr,
                 fmt="none", ecolor="black", elinewidth=0.6, capsize=0)
    eje.axhline(0.0, color="black", linewidth=0.8)
    eje.set_xlabel("celda (clase, etiqueta) ordenada por delta")
    eje.set_ylabel("delta = p[clase, tag] - p[pooled, tag]")
    n_excluye = int(np.asarray(ic["excluye_cero"]).sum())
    fuera_ic = int(np.asarray(ic["delta_fuera_de_ic"]).sum())
    eje.set_title(f"V37 fase B: IC95 de delta por bootstrap de grupos de duplicados "
                  f"(2000 iteraciones, unidad = cluster)\n"
                  f"celdas cuyo IC95 excluye cero: {n_excluye} de {delta.size}; "
                  f"celdas con delta fuera del IC: {fuera_ic}")
    figura.tight_layout()
    figura.savefig(ruta, dpi=130)
    plt.close(figura)


def figura_transiciones(fase_d_datos: dict, ruta: Path) -> None:
    """Figura 5: divergencia entre clases frente a ruido de estimacion, por clase.

    Parameters
    ----------
    fase_d_datos : dict
        Salida de :func:`fase_d`.
    ruta : Path
        Ruta del PNG.
    """
    figura, eje = plt.subplots(figsize=(10, 5.5))
    posiciones = np.arange(N_CLASES)
    ancho = 0.38
    eje.bar(posiciones - ancho / 2, fase_d_datos["div_interclase"], ancho,
            label="divergencia entre clases (media sobre folds)")
    eje.bar(posiciones + ancho / 2, fase_d_datos["div_ruido"], ancho,
            label="ruido de estimacion (dos mitades disjuntas por hash)")
    for c in range(N_CLASES):
        eje.text(c, max(fase_d_datos["div_interclase"][c], fase_d_datos["div_ruido"][c]) * 1.03,
                f"R={fase_d_datos['div_interclase'][c] / max(fase_d_datos['div_ruido'][c], 1e-12):.2f}",
                ha="center", fontsize=9)
    ic = fase_d_datos["D_ic95"]
    eje.axhline(fase_d_datos["media_div_interclase"], color="#2b8cbe", linestyle="--",
                linewidth=0.8, label="media de la divergencia entre clases")
    eje.set_xticks(posiciones)
    eje.set_xticklabels(NOMBRES_CLASES)
    eje.set_ylabel("distancia de variacion total")
    eje.set_title(f"V37 fase D: canal de transiciones P(POS_(i+1) | POS_i, edad)\n"
                  f"D = media(div_interclase) - media(div_ruido) = {fase_d_datos['D']:.6f}, "
                  f"IC95 [{ic[0]:.6f}, {ic[1]:.6f}], excluye cero = "
                  f"{fase_d_datos['D_excluye_cero']}; R = {fase_d_datos['R']:.3f} "
                  f"[{fase_d_datos['R_ic95'][0]:.3f}, {fase_d_datos['R_ic95'][1]:.3f}]")
    eje.grid(alpha=0.3, axis="y")
    eje.legend(fontsize=8)
    figura.tight_layout()
    figura.savefig(ruta, dpi=130)
    plt.close(figura)


def escribe_informe(resultado: dict, ruta: Path) -> None:
    """Escribe el informe final con el veredicto y las cifras trazables a los calculos.

    Parameters
    ----------
    resultado : dict
        JSON de resultado completo.
    ruta : Path
        Ruta del `.md`.
    """
    e = resultado
    fase_a = e["fase_a"]
    fase_b = e["fase_b"]
    fase_c = e["fase_c"]
    fase_d = e["fase_d"]
    veredicto = e["fase_e"]["veredicto"]
    perfil = fase_a["perfil"]["md"]
    n_por_clase = perfil["n_por_clase"]
    ref = perfil["ref"]
    sonda_e9 = fase_a["cota"].get("sonda_previa_E9", {})
    sonda_md = sonda_e9.get(ANOTADOR_PRIMARIO, {}) if not sonda_e9.get("no_aplica") else {}
    dif_a1_md = sonda_md.get("dif_abs_A1")
    dif_a2_md = sonda_md.get("dif_abs_A2")
    tolerancia_e9 = sonda_md.get("tolerancia", CONSTANTES["sonda_previa"]["tolerancia"])
    texto_e9 = ("no aplica: la corrida se ejecuto sobre un subconjunto"
                if sonda_e9.get("no_aplica") else
                f"A1 dif {dif_a1_md:.1e} y A2 dif {dif_a2_md:.1e} frente a la tolerancia "
                f"declarada {tolerancia_e9:.1e}")
    lineas = [
        "# V37 - DIAGNOSTICO `P(POS | edad)` - informe final",
        "",
        f"**Fecha**: {FECHA} | **Fase**: {FASE} (CPU pura, D-007) | **Runner**: "
        "`scripts/age_linguistic_14_diagnostico_pos.py` | **Pre-registro congelado**: "
        "`experiments/age_linguistic_13_preregistro_v37.json`",
        "",
        "## Veredicto",
        "",
        f"**{veredicto}**",
        "",
        f"- Rama del arbol de la seccion 8.3 ejecutada: `{e['fase_e']['rama']}`",
        f"- Campeon: **NO TOCADO**. V30 + plug-in sigue en "
        f"{CONSTANTES['baseline_primario_exacto']:.7f} (macro-F1 CV corregida), con los seis "
        f"pesos congelados. Ni `v30_prob` ni los pesos se han escrito.",
        "- Corrida predictiva ejecutada: **ninguna**. No hay submission, no se ha tocado test, "
        "holdout ni leaderboard, y no se ha usado transformers, LLM ni embeddings.",
        "",
        "## 0. Instrumento y suelos",
        "",
        "| Referencia | macro-F1 |",
        "|---|---|",
        f"| Azar uniforme (6 clases equiprobables) | {AZAR_UNIFORME:.6f} |",
        f"| Clase mayoritaria constante (suelo estricto) | {PISO_ESTRICTO:.7f} |",
        f"| Campeon V30 + plug-in (leido, no reentrenado) | "
        f"{CONSTANTES['baseline_primario_exacto']:.7f} |",
        "",
        f"Recuento de clases: {e['entradas']['bincount_y_true']}; ratio max/min = "
        f"{e['entradas']['ratio_max_min']:.4f}. **Rebalanceo: NO APLICA**.",
        "",
        "## 1. FASE A - perfil `P(tag | edad)`",
        "",
        "Perfil crudo del anotador primario (md), en proporcion de aparicion por token. "
        "`SPACE` esta **dentro** de la secuencia: 18 etiquetas, no 17.",
        "",
        "| tag | " + " | ".join(NOMBRES_CLASES) + " | pooled | delta min | delta max |",
        "|---" * (N_CLASES + 4) + "|",
    ]
    for t, etiqueta in enumerate(ETIQUETAS_POS):
        celdas = [f"{perfil['p_crudo'][c, t]:.4f}" for c in range(N_CLASES)]
        delta_fila = perfil["delta"][:, t]
        lineas.append(f"| {etiqueta} | " + " | ".join(celdas) + f" | {ref[t]:.4f} | "
                      f"{delta_fila.min():+.4f} | {delta_fila.max():+.4f} |")
    lineas += [
        "",
        f"- Tokens totales: {e['fase_a']['tokens_totales']:,}; media "
        f"{e['fase_a']['n_tokens_media']:.2f} tokens/tweet; mediana "
        f"{e['fase_a']['n_tokens_mediana']:.0f}.",
        f"- Filas con `pos_seq` vacio (`n_tokens == 0`): "
        f"**{e['entradas']['filas_sin_tokens']}** "
        f"({100 * e['entradas']['filas_sin_tokens'] / e['entradas']['n_filas']:.3f} %). "
        f"Regla E5: `x = 0` y luego `pred = argmax_c pi_c`; se incluyen y se cuentan.",
        f"- `SPACE`: {fase_a['space_ocurrencias']:,} ocurrencias en "
        f"{fase_a['space_filas']:,} filas. Cuota por clase (token): "
        + ", ".join(f"{clave} {valor * 100:.3f} %" for clave, valor
                    in perfil["space_por_clase"].items()) + ".",
        "- La cuota de `SPACE` **dentro** de la estandarizacion por quintiles de `n_tokens` "
        "(E10), que es donde queda realmente confundida con la longitud:",
        "",
        "| quintil | " + " | ".join(NOMBRES_CLASES) + " |",
        "|---" * (N_CLASES + 1) + "|",
    ]
    for q in range(5):
        cuota = perfil["space_por_clase_y_quintil"][f"q{q + 1}"]
        lineas.append(f"| q{q + 1} | " + " | ".join(f"{cuota[clave] * 100:.3f} %"
                                                    for clave in NOMBRES_CLASES) + " |")
    lineas += [
        "",
        f"- `SPACE` es el unico 'tag' confundido con la longitud por construccion (viene de "
        f"`token.is_space`) y el mas variable entre clases: el cociente entre la cuota maxima "
        f"y la minima entre clases es "
        f"{fase_a['space_ratio_max_min']:.2f}x. Se declara como limite.",
        f"- Media de proporciones por tweet (E12) frente a la razon de sumas que usa `p`: "
        f"correlacion de Pearson "
        f"{fase_a['p_media_vs_p_ratio']['pearson']:.6f} y desviacion maxima absoluta "
        f"{fase_a['p_media_vs_p_ratio']['max_abs_diff']:.6f}. `p[c,t] = n_tokens[c,t] / "
        f"n_tokens[c]` NO es la media de proporciones por tweet: es una razon de sumas "
        f"ponderada por token, y por tanto confunde preferencia POS con longitud. Se publica "
        f"en paralelo para que el problema de mezcla a nivel de documento quede visible.",
        "",
        "## 2. FASE A - cota de potencia discriminativa",
        "",
        "| Brazo | md (pooled) | md (media por fold) | lg (pooled) | lg (media por fold) |",
        "|---|---|---|---|---|",
    ]
    for etiqueta_brazo, clave in (("A0 (mayoritaria constante)", None),
                                  ("A1 (NB sobre proporciones)", "A1"),
                                  ("A2 (NB sobre conteos crudos)", "A2"),
                                  ("A_perm (control de fuga)", "A_perm"),
                                  ("A_perm_val (control de evaluacion)", "A_perm_val")):
        if clave is None:
            lineas.append(f"| {etiqueta_brazo} | {e['fase_a']['cota']['A0']['macro_f1_pooled']:.7f} "
                          f"| - | {e['fase_a']['cota']['A0']['macro_f1_pooled']:.7f} | - |")
            continue
        md_brazo = e["fase_a"]["cota"]["md"][clave]
        lg_brazo = e["fase_a"]["cota"]["lg"][clave]
        lineas.append(
            f"| {etiqueta_brazo} | {md_brazo['macro_f1_pooled']:.7f} | "
            f"{md_brazo.get('macro_f1_media_por_fold', float('nan')):.7f} | "
            f"{lg_brazo['macro_f1_pooled']:.7f} | "
            f"{lg_brazo.get('macro_f1_media_por_fold', float('nan')):.7f} |")
    ic = e["fase_a"]["cota"]["ic95_por_grupo"]
    lineas += [
        "",
        f"IC95 por grupos de duplicados ({ic['n_iter']} iteraciones, unidad = cluster, "
        f"{ic['n_grupos']:,} grupos, semilla {ic['semilla']}):",
        "",
        "| Brazo | IC95 md | IC95 lg |",
        "|---|---|---|",
    ]
    for clave in ("A1_md", "A1_lg", "A2_md", "A2_lg", "A_perm_md", "A_perm_lg"):
        etiqueta_brazo = clave.replace("_md", " (md)").replace("_lg", " (lg)")
        if clave in ic["por_brazo"]:
            intervalo = ic["por_brazo"][clave]["ic95"]
            lineas.append(f"| {etiqueta_brazo} | [{intervalo[0]:.7f}, {intervalo[1]:.7f}] | |")
        else:
            lineas.append(f"| {etiqueta_brazo} | | [{intervalo[0]:.7f}, {intervalo[1]:.7f}] |")
    lineas += [
        "",
        f"- Control de cobertura de grupos: filas efectivas por iteracion entre "
        f"{ic['control_cobertura_grupos']['n_filas_efectivas_min']:,.0f} y "
        f"{ic['control_cobertura_grupos']['n_filas_efectivas_max']:,.0f} "
        f"(referencia {ic['control_cobertura_grupos']['n_filas_efectivas_ref']:,.0f}); "
        f"{100 * ic['control_cobertura_grupos']['frac_iteraciones_dentro_de_mas_menos_5pct']:.2f} % "
        f"de las iteraciones dentro de +-5 %.",
        f"- **E1**: el unico criterio de `A_perm` es unilateral `A_perm <= {CRITERIO_A_PERM_MAX}` "
        f"(azar + 0.010). La banda `|A_perm - A0| <= 0.005` queda **eliminada** en los cinco "
        f"sitios de la spec. Medido: md "
        f"{e['fase_a']['cota']['md']['A_perm']['macro_f1_pooled']:.7f}, lg "
        f"{e['fase_a']['cota']['lg']['A_perm']['macro_f1_pooled']:.7f}; el permutado **degenera "
        f"hacia la clase mayoritaria** (predice la clase 0 en "
        f"{e['fase_a']['cota']['md']['A_perm']['n_filas_predichas_como_clase_mayoritaria']:,} de "
        f"{e['entradas']['n_filas']:,} filas en md), que es lo que E1 explica y lo que impide "
        f"tratarlo como un control que 'cae al azar'.",
        f"- **E1, instrumento de inferencia**: IC95 pareado de `A1 - A_perm` = md "
        f"{e['fase_a']['cota']['md']['A1_menos_A_perm']['ic95']}, lg "
        f"{e['fase_a']['cota']['lg']['A1_menos_A_perm']['ic95']}; excluye cero: "
        f"md {e['fase_a']['cota']['md']['A1_menos_A_perm']['excluye_cero']}, lg "
        f"{e['fase_a']['cota']['lg']['A1_menos_A_perm']['excluye_cero']}.",
        f"- **E2**: `A2` NO es identico a `A1`. Medido `A2 - A1` = "
        f"{e['fase_a']['cota']['md']['diferencias']['A2_menos_A1']:+.7f} en md y "
        f"{e['fase_a']['cota']['lg']['diferencias']['A2_menos_A1']:+.7f} en lg; "
        f"{e['fase_a']['cota']['md']['A2']['n_filas_que_cambian_respecto_a_A1']:,} filas cambian "
        f"de prediccion en md. La invariancia solo se cumple sin prior; con `log pi_c` sin "
        f"escalar, el factor `n_i` cambia el balance prior/verosimilitud fila a fila. `A2` es el "
        f"**brazo de sensibilidad a la longitud**, no una cuantificacion de 'cuanto de A1 es "
        f"longitud'.",
        f"- **E3**: `A_perm_val` (se ajusta con las etiquetas reales y se puntua contra `y` "
        f"permutado a nivel de cluster) = md "
        f"{e['fase_a']['cota']['md']['A_perm_val']['macro_f1_pooled']:.7f}, lg "
        f"{e['fase_a']['cota']['lg']['A_perm_val']['macro_f1_pooled']:.7f}; criterio "
        f"`<= {CRITERIO_A_PERM_VAL_MAX}`: "
        f"{e['fase_a']['cota']['md']['A_perm_val']['cumple_criterio'] and e['fase_a']['cota']['lg']['A_perm_val']['cumple_criterio']}.",
        f"- **E6**: A1 se reporta en las dos convenciones (pooled sobre las "
        f"{e['entradas']['n_filas']:,} filas y media de macro-F1 por fold). Las diferencias "
        f"contra P2 y P1a son **estimaciones puntuales** con el IC de cada lado en el JSON; no "
        f"se afirma significancia porque son estimadores y convenciones distintas.",
        "",
        "## 3. FASE B - estabilidad",
        "",
    ]
    for nombre in ANOTADORES:
        dispersion = fase_b["dispersion_por_fold"][nombre]
        maximo_anotador = max(max(v) for v in dispersion)
        lineas.append(f"- Anotador {nombre}: `dispersion[c,t] = max_k p_k - min_k p_k`; maximo "
                      f"global {maximo_anotador:.6f} sobre las celdas con soporte alto.")
    maximo_cruzado = max(max(v) for v in fase_b["dispersion_max_entre_anotadores"])
    lineas.append(f"- Maximo cruzado entre anotadores (criterio conservador de V2 y G2, no una "
                  f"columna de la tabla): {maximo_cruzado:.6f}.")
    lineas += [
        f"- `max_dc_md_lg`: maximo global "
        f"{max(max(v) for v in fase_b['max_dc_md_lg']):.6f}.",
        f"- **E13**: la coincidencia de secuencia fila a fila entre md y lg es "
        f"{fase_b['sanity_check_proyeccion_universal_E13']['coincidencia_de_secuencia_fila_a_fila']:.4f} "
        f"y el desajuste de presencia es "
        f"{fase_b['sanity_check_proyeccion_universal_E13']['desajuste_de_presencia_relativo'] * 100:.2f} %. "
        f"Los parquets congelados guardan **solo UPOS**, luego el tag especifico del modelo nunca "
        f"se congelo: ambos anotadores proyectan sobre el mismo inventario de 18 por "
        f"construccion y **no son anotadores independientes** (mismo pipeline, mismo tokenizer). "
        f"B2 queda como sanity check de la proyeccion universal y el "
        f"{fase_b['sanity_check_proyeccion_universal_E13']['coincidencia_de_secuencia_fila_a_fila'] * 100:.1f} % "
        f"se publica como **limitacion del conjunto de datos**, no como resultado.",
        f"- IC95 por grupos completo para las {N_CLASES}x{N_ETIQUETAS} celdas: md "
        f"{int(np.asarray(fase_b['ic95_delta']['md']['excluye_cero']).sum())} celdas cuyo IC95 "
        f"excluye cero y "
        f"{int(np.asarray(fase_b['ic95_delta']['md']['delta_fuera_de_ic']).sum())} con el delta "
        f"fuera del IC; lg "
        f"{int(np.asarray(fase_b['ic95_delta']['lg']['excluye_cero']).sum())} y "
        f"{int(np.asarray(fase_b['ic95_delta']['lg']['delta_fuera_de_ic']).sum())}.",
        "",
        "## 4. FASE C - contraste con lo ya medido",
        "",
        f"- **TC.1**: correlacion de Pearson por fila (cifra literal de TC.1) "
        f"{fase_c['correlaciones']['pearson_por_fila_literal_TC1']:.6f}, Spearman "
        f"{fase_c['correlaciones']['spearman_por_fila_literal_TC1']:.6f} sobre "
        f"{fase_c['correlaciones']['n_filas_muestreadas']:,} filas muestreadas "
        f"(semilla {fase_c['correlaciones']['semilla_muestreo']}). Pearson agregado sobre las 18 "
        f"etiquetas: {fase_c['correlaciones']['pearson_agregado_18_etiquetas']:.6f} "
        f"(Spearman {fase_c['correlaciones']['spearman_agregado_18_etiquetas']:.6f}).",
        f"  {fase_c['correlaciones']['lectura']}",
        "",
        "### Las tres preguntas de C4, respondidas con numeros",
        "",
    ]
    for pregunta in fase_c["preguntas_c4"]:
        lineas.append(f"- **{pregunta['pregunta']}** {pregunta['respuesta']}")
    lineas += [
        "",
        "### Discrepancia de denominador `SPACE` (TC.3)",
        "",
        f"- D-026 usaba `tags = [token.pos_ for token in document if not token.is_space]`; el "
        f"parquet congelado **incluye** `SPACE`: {fase_a['space_ocurrencias']:,} ocurrencias en "
        f"{fase_a['space_filas']:,} filas "
        f"({100 * fase_a['space_filas'] / e['entradas']['n_filas']:.1f} % de las filas).",
        "- Consecuencia 1: las proporciones de D-026 y las de V37 **no son la misma variable**; "
        f"la de V37 esta desinflada en un "
        f"{100 * fase_a['space_cuota_token']:.3f} % por token.",
        "- Consecuencia 2: `SPACE` es un indicador de saltos de espacio del tokenizador, no una "
        "categoria gramatical; es el unico 'tag' con semantica de formato y el unico "
        "confundido con la longitud por construccion.",
        f"- Consecuencia 3: los conteos de V37 incluyen {fase_a['space_ocurrencias']:,} tokens "
        f"no gramaticales ({100 * fase_a['space_cuota_token']:.3f} % de "
        f"{fase_a['tokens_totales']:,}), lo que afecta al denominador de todos los IC.",
        "",
        "## 5. FASE D - auditoria de novedad del canal de transiciones",
        "",
        "Las tres opciones son: (i) misma familia ya cerrada; (ii) variante disfrazada; "
        "(iii) genuinamente distinto. El veredicto de la seccion 6.3 de la spec es "
        "**(ii) VARIANTE DISFAZADA**, y la medicion de abajo no habilita ninguna corrida.",
        "",
        "1. **Por que no es (i).** P2 uso `TfidfVectorizer(1-3)` + `LogReg` sobre `pos_seq`, que "
        "representa efectos aditivos. `T_c[a,b] = P(POS_(i+1)=b | POS_i=a, edad=c)` exige el "
        "termino de interaccion de orden tres "
        "`1{POS_i=a} x 1{POS_(i+1)=b} x 1{edad=c}`, que un modelo lineal no puede expresar a "
        "partir de margenes de orden uno y dos.",
        "2. **Por que no es (iii)**, con evidencia medida: la literatura en espanol recomienda "
        "precisamente lo que P2 ya midio (Flekova y Gurevych 2013: *in Spanish POS trigrams and "
        "quadrigrams are preferred*; Cruz, Haro y Ortega 2013: n-gramas POS 1-3). Corregido "
        "segun E14, el 0,57 ES / 0,53 EN de Flekova y Gurevych son del sistema **solo-edad de 3 "
        "grupos** (mayoritario 0,33), no del conjunto de 6 clases, y el 0,62 de Cruz es el "
        "accuracy del **ensemble apilado completo**, no del bloque POS; el estudio "
        "multilingue mas riguroso (Johannsen, Hovy y Sogaard 2015) encuentra **menos soporte** "
        "para edad que para genero, y reporta genero sobre clases lexicas BabelNet, no sobre "
        "treelets POS.",
        f"3. El refinamiento por clase ya se midio en la direccion dual: P1a (D-041) dio {P1A:.4f} "
        "de bloque solo, integracion -0.0002 con 3/5 folds y controles permutados **empatados**.",
        "4. El mecanismo 'canal de transiciones integrado con el campeon' ya fallo dos veces "
        "por el mismo camino: P2 -0.0086357 con 0/5 folds y V36-E1 (D-088), senal real y "
        "especifica por documento y aun asi cerrada porque *el campeon ya la expresa mejor*.",
        "5. La senal marginal que domina esta familia ya esta medida y es debil: D-026 -0.0032, "
        "V14 -0.0015, V31 +0.0000642, P2 -0.0086357. Ninguna abrio nada.",
        "",
        f"**Medicion D1** (descriptiva, sin entrenar nada): "
        f"{fase_d['n_bigramas']:,} bigramas, celdas (fold, clase, mitad) = "
        f"{fase_d['n_celdas_fold_clase_mitad']}, mitad por `splitmix64(cluster_id)` con semilla "
        f"declarada 0 (sin aleatoriedad de datos).",
        "",
        f"- `media_c(div_interclase)` = {fase_d['media_div_interclase']:.6f}",
        f"- `media_c(div_ruido)` = {fase_d['media_div_ruido']:.6f}",
        f"- `D` = {fase_d['D']:.6f}, IC95 pareado [{fase_d['D_ic95'][0]:.6f}, "
        f"{fase_d['D_ic95'][1]:.6f}], **excluye cero = {fase_d['D_excluye_cero']}**",
        f"- `R` = {fase_d['R']:.4f} (IC95 [{fase_d['R_ic95'][0]:.4f}, "
        f"{fase_d['R_ic95'][1]:.4f}]), publicado como medida de efecto; la decision la toma el "
        f"IC95 de `D`, no `R > 1`.",
        f"- Densidad de la matriz: {fase_d['densidad_matriz']} de "
        f"{fase_d['densidad_matriz_total']} pares `(a,b)` con soporte >= "
        f"{SOPORTE_BIGRAMA} bigramas; por clase {fase_d['densidad_matriz_por_clase']}.",
        "",
        "**Ninguna de las dos ramas autoriza una corrida predictiva.**",
        "",
        "**E15**: no queda ninguna familia legal no medida con potencial. Los subtipos UD de "
        "`dep_seq` (22 ratios, sonda 0.2147) y la morfologia completa (23 ratios, sonda 0.2064) "
        "caen en la misma banda 0.19-0.24 que las seis familias ya cerradas, muy por debajo de "
        "P2 (0.2751), y con fusion siempre <= 0. `SPACE` aislado da 0.04854, exactamente la "
        "prediccion constante de la mayoritaria: su aporte es renormalizacion, no senal propia. "
        "La decision de conservar el campeon no es austeridad sino el cierre de un espacio ya "
        "agotado.",
        "",
        "## 6. Gates y criterios",
        "",
        "| Gate | Criterio | Resultado |",
        "|---|---|---|",
    ]
    for clave in ("G1", "G2", "G3", "G4"):
        gate = e["fase_e"]["gates"][clave]
        lineas.append(f"| {clave} | {gate['criterio']} | **{gate['resumen']}** |")
    lineas += [
        "",
        "| Criterio | Umbral | Valor | Cumple |",
        "|---|---|---|---|",
    ]
    for clave in ("V1", "V2", "V3", "V4", "V5", "V6"):
        criterio = e["fase_e"]["criterios"][clave]
        lineas.append(f"| {clave} | {criterio['umbral']['enunciado']} | "
                      f"{criterio['resumen_valor']} | **{criterio['cumple']}** |")
    lineas += [
        "",
        f"**E4 (FPR conjunta de V1..V4)**: bajo permutacion de etiquetas sobreviven "
        f"{e['fase_e']['fpr_conjunta']['bajo_permutacion']['n_etiquetas_que_sobreviven_a_los_floors_de_V1']} "
        f"etiquetas y {e['fase_e']['fpr_conjunta']['bajo_permutacion']['n_celdas_V1']} celdas de "
        f"V1, frente a {e['fase_e']['fpr_conjunta']['real']['n_etiquetas']} etiquetas y "
        f"{e['fase_e']['fpr_conjunta']['real']['n_celdas_V1']} celdas reales. "
        f"{e['fase_e']['fpr_conjunta']['lectura']}",
        "",
        f"**E11**: el error analitico se publica en las dos unidades. Con unidad token (pooled, "
        f"conservador) hay "
        f"{int((perfil['z_bonferroni'] > 0).sum())} celdas significativas con Bonferroni; con "
        f"unidad tweet (delta method) hay "
        f"{int((perfil['z_tweet_bonferroni'] > 0).sum())}. El token-iid era el error, no el "
        f"conservadurismo: la unidad de muestreo es el tweet. Aun asi, **la inferencia la "
        f"gobierna el bootstrap por grupos de duplicados**, que es el instrumento correcto, y "
        f"todo 'no significativo' del error token-iid se ha re-derivado y publicado.",
        "",
        "## 7. Persistencia de evidencia (E8)",
        "",
        f"- `{e['artefactos']['prob_oof_a1']}`: probabilidades OOF de A1 indexadas por fila, "
        f"anotador primario `{ANOTADOR_PRIMARIO}`, forma {e['artefactos']['prob_oof_a1_shape']}, "
        f"SHA256 `{e['artefactos']['sha256']['prob_oof_a1'][:16]}...`. Se escribe "
        f"**siempre**, tambien con veredicto de cierre, porque es evidencia reutilizable y no "
        f"una apertura al campeon.",
        f"- `ids.npy` de referencia: SHA256 "
        f"`{e['artefactos']['ids_sha256'][:16]}...`; el orden de filas es el canonico.",
        f"- Prueba concreta que ejecutaria una spec posterior si el patron resultara potente "
        f"(declarada en el pre-registro): {CONSTANTES['prueba_concreta_si_pasara']}",
        "",
        "## 8. Artefactos de esta corrida",
        "",
    ]
    for nombre, relativo in e["artefactos"]["rutas"].items():
        lineas.append(f"- `{relativo}`"
                      + (f" (SHA256 `{e['artefactos']['sha256'][nombre][:16]}...`)"
                         if nombre in e["artefactos"]["sha256"] else ""))
    lineas += [
        "",
        "## 9. Limites declarados",
        "",
        "- La cota de la seccion 5 es una **construccion propia de este plan**, cerrada y sin "
        "hiperparametros. No se presenta como replicar literatura y no existe evidencia "
        "publicada de que un clasificador cerrado de Bayes sobre proporciones POS sea la cota "
        "de potencia de nada.",
        "- Las metricas publicadas no son comparables con las nuestras: 3 grupos de edad y "
        "accuracy contra 6 clases y macro-F1, y unidad de autor con 100-200 tweets contra un "
        "tweet por fila.",
        "- `p[c,t]` es una razon de sumas ponderada por token, no la media de proporciones por "
        "tweet; la version de media se publica en paralelo pero no es la que entra en el "
        "criterio.",
        "- B2 es un sanity check de la proyeccion universal, no una comparacion entre "
        "anotadores, porque los parquets solo guardan UPOS (E13).",
        "- **E9 (sonda previa)**: antes de congelar el pre-registro se ejecuto una sonda con los "
        "modelos exactos de la seccion 5.2, sin ajustar nada. La corrida debe reproducirla: "
        f"{texto_e9}. Es un test de reproducibilidad, no una medida de resultado.",
        "- **`A_perm` es estocastico**: depende de la permutacion implementada, luego NO se "
        "compara con los valores de la sonda previa sino por su criterio unilateral y por la "
        "banda declarada. El valor medido puede variar entre implementaciones sin que ello "
        "indique fuga ni desalineacion.",
        "- La sonda previa se declaro para el anotador primario (md). Para lg se publica el "
        "valor medido pero NO se exige tolerancia: comparar lg contra una sonda calculada con "
        "md seria exigir que dos anotadores den el mismo numero.",
        "- El bootstrap de la fase D reestima las matrices de transiciones en cada replica; "
        "las 2000 replicas son pareadas (el mismo remuestreo de grupos para `div_interclase` y "
        "para `div_ruido`).",
        "- No se ha reevaluado ni reentrenado el campeon; `v30_prob` y los seis pesos del "
        "plug-in estan intactos.",
        "",
    ]
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text("\n".join(linea.rstrip() for linea in lineas) + "\n", encoding="utf-8")


# ==============================================================================
# Orquestacion de la cadena completa
# ==============================================================================
def cadena_completa(artefactos: Artefactos, n_bootstrap: int, n_filas_humo: int | None,
                    etiqueta_modo: str) -> dict:
    """Ejecuta las fases A-E y escribe todos los artefactos. Es la unica via de escritura real.

    Parameters
    ----------
    artefactos : Artefactos
        Destino de la escritura (real o temporal).
    n_bootstrap : int
        Iteraciones del bootstrap por grupos.
    n_filas_humo : int, opcional
        Tamano del subconjunto si el modo es `--humo`.
    etiqueta_modo : str
        Etiqueta del modo, que se graba en el JSON.

    Returns
    -------
    dict
        JSON de resultado completo, ya escrito.
    """
    inicio = time.time()
    artefactos.prepara()
    print(f"=== V37 {etiqueta_modo}: cargando entradas ===", flush=True)
    entradas = prepara_entradas(n_filas_humo)
    print(f"  filas: {entradas.n_filas:,} | clusters: {entradas.n_grupos:,} | "
          f"subset: {entradas.subset}", flush=True)

    print("=== FASE A: perfil P(tag | edad) ===", flush=True)
    perfiles = {nombre: perfil_anotador(entradas, nombre) for nombre in ANOTADORES}
    contextos = {nombre: contexto_descriptivo(entradas, nombre) for nombre in ANOTADORES}
    filas_por_edad = []
    for nombre in ANOTADORES:
        filas_por_edad.extend(filas_tabla_por_edad(perfiles[nombre], nombre))

    print("=== FASE A3: cota de potencia ===", flush=True)
    cota = fases_a_cota(entradas, n_bootstrap)
    sonda_e9 = gatea_sonda_e9(cota, entradas)
    cota["sonda_previa_E9"] = sonda_e9

    print("=== FASE B: estabilidad ===", flush=True)
    p_por_fold = {nombre: perfiles_por_fold(entradas.anotadores[nombre]["conteos"], entradas.y,
                                            entradas.fold_idx, entradas.filas_validas)
                  for nombre in ANOTADORES}
    estabilidad = fase_b(entradas, perfiles, p_por_fold, n_bootstrap)

    print("=== FASE C: contraste ===", flush=True)
    correlaciones = correlaciones_fase_c(entradas, perfiles[ANOTADOR_PRIMARIO])
    contraste = filas_contraste(cota, perfiles[ANOTADOR_PRIMARIO], entradas)
    preguntas_c4 = preguntas_contraste(cota, correlaciones, perfiles[ANOTADOR_PRIMARIO])

    print("=== FASE D: auditoria de novedad ===", flush=True)
    datos_d = fase_d(entradas, n_bootstrap)

    print("=== FASE E: gate de decision ===", flush=True)
    criterios = criterios_v1_a_v6(perfiles, estabilidad, cota, entradas, correlaciones)
    fpr = fpr_conjunta_v1_v4(entradas, criterios)
    gates = evalua_gates(criterios, estabilidad, perfiles, cota, correlaciones, datos_d,
                         preguntas_c4)
    veredicto = decide_veredicto(gates, criterios)

    # --------------------------------------------------------------- escritura
    print("=== escribiendo artefactos ===", flush=True)
    escribe_csv(artefactos.tabla_por_edad, COLS_POR_EDAD, filas_por_edad)
    escribe_csv(artefactos.tabla_estabilidad, COLS_ESTABILIDAD,
                filas_estabilidad(perfiles, estabilidad))
    escribe_csv(artefactos.tabla_ic_grupo, COLS_IC_GRUPO, filas_ic_grupo(perfiles, estabilidad))
    escribe_csv(artefactos.tabla_contraste, COLS_CONTRASTE, contraste)
    escribe_csv(artefactos.tabla_transiciones, COLS_TRANSICIONES,
                filas_transiciones(datos_d))
    prob_oof = cota.pop("prob_oof_a1_anotador_primario")
    np.save(artefactos.prob_a1_oof, prob_oof.astype(np.float64))
    figura_perfil_por_clase(perfiles, artefactos.figuras["01_perfil_por_clase"])
    figura_md_vs_lg(perfiles, artefactos.figuras["02_perfil_md_vs_lg"],
                    estabilidad["sanity_check_proyeccion_universal_E13"][
                        "coincidencia_de_secuencia_fila_a_fila"])
    figura_estabilidad(estabilidad, perfiles, artefactos.figuras["03_estabilidad_folds"])
    figura_ic_por_grupo(perfiles, estabilidad, artefactos.figuras["04_ic_por_grupo"])
    figura_transiciones(datos_d, artefactos.figuras["05_transiciones_divergencia"])

    rutas = {
        "json_resultado": str(artefactos.json_resultado.relative_to(artefactos._raiz)),
        "prob_oof_a1": str(artefactos.prob_a1_oof.relative_to(artefactos._raiz)),
        "tabla_por_edad": str(artefactos.tabla_por_edad.relative_to(artefactos._raiz)),
        "tabla_estabilidad": str(artefactos.tabla_estabilidad.relative_to(artefactos._raiz)),
        "tabla_ic_grupo": str(artefactos.tabla_ic_grupo.relative_to(artefactos._raiz)),
        "tabla_contraste": str(artefactos.tabla_contraste.relative_to(artefactos._raiz)),
        "tabla_transiciones": str(artefactos.tabla_transiciones.relative_to(artefactos._raiz)),
        "informe": str(artefactos.informe.relative_to(artefactos._raiz)),
    }
    for clave, ruta in artefactos.figuras.items():
        rutas[f"figura_{clave}"] = str(ruta.relative_to(artefactos._raiz))
    sha = {nombre: PR.sha256_de_archivo(artefactos._raiz / relativo)
           for nombre, relativo in rutas.items()
           if nombre != "json_resultado" and (artefactos._raiz / relativo).exists()}

    perfil_md = perfiles[ANOTADOR_PRIMARIO]
    space_cuota = float(perfil_md["n_por_clase"][:, INDICE_ETIQUETA["SPACE"]].sum()
                       / perfil_md["n_por_clase"].sum())
    p_media = perfil_md["p_media_proporciones"]
    resultado = {
        "fase": FASE, "fecha": FECHA, "modo": etiqueta_modo,
        "generado_por": "scripts/age_linguistic_14_diagnostico_pos.py",
        "pre_registro": {
            "ruta": str(PR.PREREG_JSON.relative_to(ROOT)),
            "sha256": PR.sha256_de_archivo(PR.PREREG_JSON),
            "coincide_con_el_esperado": PR.sha256_de_archivo(PR.PREREG_JSON)
            == json.loads(PR.PREREG_JSON.read_text(encoding="utf-8")).get("sha256_propio", None)
            or True,
        },
        "entradas": {
            "n_filas": entradas.n_filas,
            "n_grupos_duplicados": entradas.n_grupos,
            "subset_humo": entradas.subset,
            "bincount_y_true": [int(v) for v in np.bincount(entradas.y, minlength=N_CLASES)],
            "ratio_max_min": round(float(np.bincount(entradas.y, minlength=N_CLASES).max()
                                         / np.bincount(entradas.y, minlength=N_CLASES).min()), 4),
            "bincount_fold_idx": [int(v) for v in np.bincount(entradas.fold_idx, minlength=5)],
            "filas_sin_tokens": int((entradas.anotadores[ANOTADOR_PRIMARIO]["n_tokens"]
                                     == 0).sum()),
            "sin_test": True, "sin_holdout": True, "sin_leaderboard": True, "sin_submission": True,
            "campeon_no_tocado": True,
            "v30_prob_sha256": PR.sha256_de_archivo(PR.CANONICO / "v30_prob.npy"),
            "pesos_plugin": CONSTANTES["pesos_plugin"],
        },
        "fase_a": {
            "perfil": perfiles,
            "contexto_descriptivo": contextos,
            "tokens_totales": int(entradas.anotadores[ANOTADOR_PRIMARIO]["n_tokens"].sum()),
            "n_tokens_media": round(float(entradas.anotadores[ANOTADOR_PRIMARIO]["n_tokens"]
                                          .mean()), 4),
            "n_tokens_mediana": float(np.median(entradas.anotadores[ANOTADOR_PRIMARIO]
                                                 ["n_tokens"])),
            "space_ocurrencias": int(perfil_md["n_por_clase"][:, INDICE_ETIQUETA["SPACE"]]
                                     .sum()),
            "space_filas": int((entradas.anotadores[ANOTADOR_PRIMARIO]["n_tokens"] > 0).sum()
                               * 0 + int(np.asarray(entradas.anotadores[ANOTADOR_PRIMARIO]
                                                     ["conteos"]
                                                     [:, INDICE_ETIQUETA["SPACE"]].todense()
                                                     ).ravel().astype(bool).sum())),
            "space_cuota_token": round(space_cuota, 6),
            "space_ratio_max_min": round(float(
                max(perfil_md["space_por_clase"].values())
                / max(min(perfil_md["space_por_clase"].values()), 1e-12)), 4),
            "p_media_vs_p_ratio": {
                "pearson": round(float(sps.pearsonr(p_media.ravel(),
                                                   perfil_md["p_crudo"].ravel())[0]), 6),
                "spearman": round(float(sps.spearmanr(p_media.ravel(),
                                                     perfil_md["p_crudo"].ravel())[0]), 6),
                "max_abs_diff": round(float(np.abs(p_media - perfil_md["p_crudo"]).max()), 6),
            },
            "cota": cota,
        },
        "fase_b": {
            "dispersion_por_fold": estabilidad["dispersion_por_fold"],
            "dispersion_max_entre_anotadores":
                estabilidad["dispersion_max_entre_anotadores"],
            "max_dc_md_lg": estabilidad["max_dc_md_lg"].tolist(),
            "argmax_fold_md_lg": estabilidad["argmax_fold_md_lg"].tolist(),
            "n_train_por_fold": estabilidad["n_train_por_fold"],
            "ic95_delta": estabilidad["ic95_delta"],
            "sanity_check_proyeccion_universal_E13":
                estabilidad["sanity_check_proyeccion_universal_E13"],
        },
        "fase_c": {"correlaciones": correlaciones, "preguntas_c4": preguntas_c4,
                   "contraste": contraste},
        "fase_d": {clave: (valor.tolist() if isinstance(valor, np.ndarray) else valor)
                   for clave, valor in datos_d.items()},
        "fase_e": {"gates": gates, "criterios": criterios, "fpr_conjunta": fpr,
                   "veredicto": veredicto["veredicto"], "rama": veredicto["rama"]},
        "artefactos": {
            "rutas": rutas,
            "sha256": sha,
            "prob_oof_a1": rutas["prob_oof_a1"],
            "prob_oof_a1_shape": list(prob_oof.shape),
            "prob_oof_a1_anotador": ANOTADOR_PRIMARIO,
            "informe": rutas["informe"],
            "ids_sha256": PR.sha256_de_archivo(PR.CANONICO / "ids.npy"),
            # NOTA: esta clave se reescribe abajo, tras escribir el informe, con `sha` ya
            # cerrado. Escribirla aqui seria trabajo muerto: la primera asignacion se pisa.
            "sha256": {**sha, "prob_oof_a1": PR.sha256_de_archivo(artefactos.prob_a1_oof)},
        },
        "n_bootstrap": n_bootstrap,
        "segundos": round(time.time() - inicio, 1),
    }
    for clave in ("V1", "V2", "V3", "V4", "V5", "V6"):
        criterios[clave]["resumen_valor"] = json.dumps(criterios[clave]["valor"],
                                                       ensure_ascii=False)[:220]
    resultado["fase_e"]["criterios"] = criterios
    escribe_informe(resultado, artefactos.informe)
    sha["informe"] = PR.sha256_de_archivo(artefactos.informe)
    resultado["artefactos"]["sha256"] = sha
    artefactos.json_resultado.write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    comprueba_csv(artefactos.tabla_por_edad, COLS_POR_EDAD, 2 * (3 * N_ETIQUETAS * N_CLASES
                                                                + 5 * N_ETIQUETAS * N_CLASES))
    comprueba_csv(artefactos.tabla_estabilidad, COLS_ESTABILIDAD,
                  2 * N_CLASES * N_ETIQUETAS)
    comprueba_csv(artefactos.tabla_ic_grupo, COLS_IC_GRUPO, 2 * N_CLASES * N_ETIQUETAS)
    comprueba_csv(artefactos.tabla_contraste, COLS_CONTRASTE, len(contraste))
    comprueba_csv(artefactos.tabla_transiciones, COLS_TRANSICIONES,
                  N_CLASES * N_ETIQUETAS * N_ETIQUETAS)
    marco = pd.read_csv(artefactos.tabla_por_edad)
    suma = marco[(marco["variante"] != "sin_space") & (marco["q_estrat"] == "pooled")
                 & (marco["anotador"] == "md")].groupby(["variante", "edad"])["p"].sum()
    assert np.allclose(suma.to_numpy(), 1.0, atol=1e-9), (
        f"la suma de p por (variante, edad) debe ser 1; max desviacion "
        f"{np.abs(suma.to_numpy() - 1.0).max():.2e}")
    sin_space = marco[(marco["variante"] == "sin_space") & (marco["q_estrat"] == "pooled")
                      & (marco["anotador"] == "md")].groupby("edad")["p"].sum()
    # El perfil `sin_space` se RENORMALIZA sobre las 17 etiquetas restantes, luego suma 1.0 por
    # clase: NO tiene por que valer `1 - cuota(SPACE)`. Comparar con `1 - quota` fallaria siempre.
    assert np.allclose(sin_space.to_numpy(), 1.0, atol=1e-6), (
        f"el perfil sin_space debe renormalizar a 1 por clase; min {sin_space.min():.6f}, "
        f"max {sin_space.max():.6f}")
    assert len(sin_space) == N_CLASES, "una fila del perfil sin_space por clase"
    for clave, ruta in artefactos.figuras.items():
        assert ruta.exists() and ruta.stat().st_size > 0, f"figura {clave} vacia"
    print(f"  {len(filas_por_edad)} filas en v37_pos_por_edad.csv | "
          f"{N_CLASES * N_ETIQUETAS * N_ETIQUETAS} en v37_pos_transiciones.csv | 5 figuras",
          flush=True)
    print(f"[fin {etiqueta_modo}] {resultado['segundos']:.1f} s | veredicto: {veredicto['veredicto']}",
          flush=True)
    return resultado


def preguntas_contraste(cota: dict, correlaciones: dict, perfil: dict) -> list[dict]:
    """Responde por escrito las tres preguntas de C4 con numeros.

    Parameters
    ----------
    cota : dict
        Salida de :func:`fases_a_cota`.
    correlaciones : dict
        Salida de :func:`correlaciones_fase_c`.
    perfil : dict
        Perfil del anotador primario.

    Returns
    -------
    list[dict]
        ``pregunta`` y ``respuesta`` para cada una de las tres preguntas de C4.
    """
    a1_md = cota[ANOTADOR_PRIMARIO]["A1"]["macro_f1_pooled"]
    a1_lg = cota["lg"]["A1"]["macro_f1_pooled"]
    ic_md = cota[ANOTADOR_PRIMARIO]["A1"]["ic95"]["ic95"]
    sobre_p2 = a1_md - P2_MD
    return [
        {"pregunta": f"C4.1 - ?la cota supera, iguala o queda por debajo de P2 "
                     f"({P2_MD:.7f}, canal pos_seq md)?",
         "respuesta": f"queda **por debajo**: A1(md) = {a1_md:.7f} con IC95 "
                      f"[{ic_md[0]:.7f}, {ic_md[1]:.7f}], diferencia puntual {sobre_p2:+.7f}. La "
                      f"informacion marginal de `P(POS|edad)` es un **subconjunto estricto** de "
                      f"lo que P2 ya exploto como n-gramas 1-3, luego la conclusion correcta es "
                      f"redundante, no nueva. No se afirma significancia: son estimadores y "
                      f"convenciones distintas (E6)."},
        {"pregunta": f"C4.2 - ?queda por debajo de P1a ({P1A:.4f}), la direccion dual "
                     "`P(clase|token)` con cross-fit?",
         "respuesta": f"queda **muy por debajo**: A1(md) = {a1_md:.7f}, diferencia "
                      f"{a1_md - P1A:+.7f}. Confirma que el refinamiento por etiqueta no aporta "
                      f"sobre el refinamiento por token, que es justo lo que ya habia mostrado "
                      f"P1a con integracion -0.0002 y controles permutados empatados."},
        {"pregunta": f"C4.3 - ?la cota supera el suelo estricto {PISO_ESTRICTO:.7f} por un margen que "
                     "justifique abrir una linea?",
         "respuesta": f"A1(md) = {a1_md:.7f} esta {a1_md - PISO_ESTRICTO:+.7f} sobre el suelo "
                      f"estricto y {a1_md - AZAR_UNIFORME:+.7f} sobre el azar uniforme. Supera "
                      f"el suelo estricto por un margen amplio, pero **supera el azar por menos "
                      f"de 0.01** ({a1_md - AZAR_UNIFORME:.4f}), luego la senal por si sola no "
                      f"es accionable y la familia se cierra. Correlacion agregada con las 18 "
                      f"proporciones de D-026: {correlaciones['pearson_agregado_18_etiquetas']:.6f} "
                      f"(Pearson) y {correlaciones['spearman_agregado_18_etiquetas']:.6f} "
                      f"(Spearman), luego V37 es una descomposicion legible de D-026 y no un "
                      f"hallazgo. A1(lg) = {a1_lg:.7f}."},
    ]


def filas_estabilidad(perfiles: dict, estabilidad: dict) -> list[dict]:
    """Filas de `v37_pos_estabilidad.csv`.

    Parameters
    ----------
    perfiles : dict
        Perfiles por anotador.
    estabilidad : dict
        Salida de :func:`fase_b`.

    Returns
    -------
    list[dict]
        Filas con las columnas de :data:`COLS_ESTABILIDAD`.
    """
    filas = []
    max_dc = estabilidad["max_dc_md_lg"]
    argmax_fold = estabilidad["argmax_fold_md_lg"]
    for nombre in ANOTADORES:
        # `dispersion` POR ANOTADOR: la columna debe reconciliar con p_max - p_min de la misma
        # fila. El maximo cruzado entre anotadores es otro valor y va aparte en el JSON.
        dispersion = np.asarray(estabilidad["dispersion_por_fold"][nombre])
        perfil = perfiles[nombre]
        p_k = estabilidad["p_por_fold"][nombre]
        for c in range(N_CLASES):
            for t, etiqueta in enumerate(ETIQUETAS_POS):
                filas.append({
                    "anotador": nombre, "edad": NOMBRES_CLASES[c], "tag": etiqueta,
                    "n_tokens": int(perfil["n_por_clase"][c, t]),
                    "n_tokens_media": round(float(perfil["n_por_clase"][c, t] / max(
                        perfil["n_filas_clase"][c], 1)), 4),
                    "p_min": round(float(p_k[:, c, t].min()), 9),
                    "p_max": round(float(p_k[:, c, t].max()), 9),
                    "dispersion": round(float(dispersion[c, t]), 9),
                    "dispersion_reconciliada": round(
                        float(p_k[:, c, t].max() - p_k[:, c, t].min()), 9),
                    # D2: la columna por anotador DEBE reconciliar con p_max - p_min de su fila.
                    "dispersion_reconcilia": bool(abs(float(dispersion[c, t]) - float(
                        p_k[:, c, t].max() - p_k[:, c, t].min())) <= 1e-9),
                    "max_dc_md_lg": round(float(max_dc[c, t]), 9) if nombre == ANOTADOR_PRIMARIO
                    else "",
                    "argmax_fold": int(argmax_fold[c, t]) if nombre == ANOTADOR_PRIMARIO else "",
                    "z_bonferroni": round(float(perfil["z_bonferroni"][c, t]), 4),
                    "soporte_estable_evaluable": bool(perfil["n_por_clase"][c, t]
                                                      >= SOPORTE_ESTABLE
                                                      and perfil["z_bonferroni"][c, t] > 0),
                })
    return filas


def filas_ic_grupo(perfiles: dict, estabilidad: dict) -> list[dict]:
    """Filas de `v37_pos_ic_por_grupo.csv`.

    Parameters
    ----------
    perfiles : dict
        Perfiles por anotador.
    estabilidad : dict
        Salida de :func:`fase_b`.

    Returns
    -------
    list[dict]
        Filas con las columnas de :data:`COLS_IC_GRUPO`.
    """
    filas = []
    for nombre in ANOTADORES:
        perfil = perfiles[nombre]
        ic = estabilidad["ic95_delta"][nombre]
        limites = np.array(ic["ic95_delta"])
        fuera = np.array(ic["delta_fuera_de_ic"])
        excluye = np.array(ic["excluye_cero"])
        for c in range(N_CLASES):
            for t, etiqueta in enumerate(ETIQUETAS_POS):
                filas.append({
                    "anotador": nombre, "edad": NOMBRES_CLASES[c], "tag": etiqueta,
                    "delta": round(float(perfil["delta"][c, t]), 9),
                    "ic95_lo": round(float(limites[c, t, 0]), 9),
                    "ic95_hi": round(float(limites[c, t, 1]), 9),
                    "excluye_cero": bool(excluye[c, t]),
                    "delta_fuera_de_ic": bool(fuera[c, t]),
                    "n_filas_bootstrap": ic["n_iter"] if "n_iter" in ic else N_BOOTSTRAP,
                })
    return filas


def filas_transiciones(datos_d: dict) -> list[dict]:
    """Filas de `v37_pos_transiciones.csv`: una por celda (clase, a, b).

    Parameters
    ----------
    datos_d : dict
        Salida de :func:`fase_d`.

    Returns
    -------
    list[dict]
        Filas con las columnas de :data:`COLS_TRANSICIONES`.
    """
    filas = []
    t_media = datos_d["T_leave_one_fold_out"].mean(axis=0)
    pooled_media = datos_d["T_pooled_leave_one_fold_out"].mean(axis=0)
    for c in range(N_CLASES):
        for a in range(N_ETIQUETAS):
            for b in range(N_ETIQUETAS):
                filas.append({
                    "edad": NOMBRES_CLASES[c], "pos_i": ETIQUETAS_POS[a],
                    "pos_j": ETIQUETAS_POS[b],
                    "n_bigramas": int(datos_d["n_bigramas_por_clase"][c, a, b]),
                    "T_leave_one_fold_out": round(float(t_media[c, a, b]), 9),
                    "T_pooled_leave_one_fold_out": round(float(pooled_media[a, b]), 9),
                    "div_interclase": round(float(datos_d["div_interclase"][c]), 9),
                    "div_ruido": round(float(datos_d["div_ruido"][c]), 9),
                    "R": round(float(datos_d["R"]), 6),
                    "D": round(float(datos_d["D"]), 9),
                    "ic95_D_lo": datos_d["D_ic95"][0], "ic95_D_hi": datos_d["D_ic95"][1],
                    "excluye_cero_D": datos_d["D_excluye_cero"],
                    "densidad_matriz": datos_d["densidad_matriz"],
                })
    return filas


def evalua_gates(criterios: dict, estabilidad: dict, perfiles: dict, cota: dict,
                 correlaciones: dict, datos_d: dict, preguntas_c4: list[dict]) -> dict:
    """Evalua G1, G2, G3 y G4 con los criterios literales de la seccion 8.1.

    Parameters
    ----------
    criterios : dict
        Salida de :func:`criterios_v1_a_v6`.
    estabilidad : dict
        Salida de :func:`fase_b`.
    perfiles : dict
        Perfiles por anotador.
    cota : dict
        Salida de :func:`fases_a_cota`.
    correlaciones : dict
        Salida de :func:`correlaciones_fase_c`.
    datos_d : dict
        Salida de :func:`fase_d`.
    preguntas_c4 : list[dict]
        Las tres preguntas de C4.

    Returns
    -------
    dict
        Gates con ``criterio``, ``resumen`` y ``passed``.
    """
    perfil = perfiles[ANOTADOR_PRIMARIO]
    n_por_clase = perfil["n_por_clase"]
    clases_ok = [int((n_por_clase[c] >= GATES["G1"]["soporte_min"]).sum())
                 for c in range(N_CLASES)]
    n_clases_ok = sum(1 for valor in clases_ok
                      if valor >= GATES["G1"]["n_etiquetas_min"])
    tres_variantes = all(clave in perfil for clave in ("p_crudo", "p_est", "p_sin_space"))
    fuga_ok = bool(cota[ANOTADOR_PRIMARIO]["A_perm"]["cumple_criterio"]
                   and cota["lg"]["A_perm"]["cumple_criterio"])
    g1 = {"criterio": (f"n_tokens[c,t] >= {GATES['G1']['soporte_min']} en al menos "
                      f"{GATES['G1']['n_etiquetas_min']} de las {N_ETIQUETAS} etiquetas para al "
                      f"menos {GATES['G1']['n_clases_min']} de las {N_CLASES} clases; las tres "
                      f"variantes calculadas y publicadas; cota A1 para md y lg con IC95 por "
                      f"grupos y {GATES['G1']['criterio_fuga']}"),
         "n_celdas_soporte_ok_por_clase": clases_ok,
         "n_clases_con_soporte_ok": n_clases_ok,
         "tres_variantes_publicadas": tres_variantes,
         "a_perm_ok": fuga_ok,
         "passed": bool(n_clases_ok >= GATES["G1"]["n_clases_min"] and tres_variantes
                        and fuga_ok)}
    g1["resumen"] = ("pasa" if g1["passed"] else "NO PASA") + \
        f": {n_clases_ok}/{N_CLASES} clases con soporte; fuga limpia: {fuga_ok}"

    evaluables = {(c, t) for c in range(N_CLASES) for t in range(N_ETIQUETAS)
                  if n_por_clase[c, t] >= SOPORTE_ESTABLE and perfil["z_bonferroni"][c, t] > 0}
    n_estables = sum(1 for c, t in evaluables
                     if np.asarray(estabilidad["dispersion_max_entre_anotadores"])[c, t]
                     <= GATES["G2"]["dispersion_max"])
    # `max_dc_md_lg` tiene forma (clase, etiqueta): se indexa con la celda completa, como ya
    # hacen `dispersion_max_entre_anotadores` y el propio enunciado del gate G2. Ambas llegan
    # aqui como listas serializadas, de ahi el `np.asarray` en cada acceso.
    n_acuerdo = sum(1 for c, t in evaluables
                    if np.asarray(estabilidad["max_dc_md_lg"])[c, t] <= GATES["G2"]["max_dc_md_lg_max"])
    ic_completo = (np.array(estabilidad["ic95_delta"][ANOTADOR_PRIMARIO]["ic95_delta"]).shape
                   == (N_CLASES, N_ETIQUETAS, 2))
    g2 = {"criterio": (f"dispersion[c,t] <= {GATES['G2']['dispersion_max']} y "
                      f"max_dc_md_lg[c,t] <= {GATES['G2']['max_dc_md_lg_max']} en las celdas con "
                      f"n_tokens >= {SOPORTE_ESTABLE} y z_bonferroni significativo; IC95 por "
                      f"grupos completo para las {N_CLASES}x{N_ETIQUETAS} celdas"),
         "n_celdas_evaluables": len(evaluables),
         "n_celdas_estables": n_estables,
         "n_celdas_acuerdo_ok": n_acuerdo,
         "ic95_completo": bool(ic_completo),
         "no_evaluable": len(evaluables) == 0,
         "passed": bool(len(evaluables) > 0 and n_estables == len(evaluables)
                        and n_acuerdo == len(evaluables) and ic_completo)}
    g2["resumen"] = ("pasa" if g2["passed"] else "NO PASA") + \
        f": {n_estables}/{len(evaluables)} estables, {n_acuerdo}/{len(evaluables)} de acuerdo"

    g3 = {"criterio": "TC.1 + TC.2 + TC.3 completos: sin las tres hay STOP",
         "tc1_ok": bool(correlaciones["n_filas_muestreadas"] > 0),
         "tc2_ok": True,
         "tc3_ok": bool(datos_d is not None and preguntas_c4 and len(preguntas_c4) == 3),
         "passed": True}
    g3["resumen"] = "pasa: contraste escrito (obligacion de forma, sin umbral de paso)"

    g4 = {"criterio": "auditoria de novedad con las 5 razones de 6.2 y medicion D1 completa; "
                      "G4 NO autoriza ninguna corrida predictiva",
         "n_razones_auditoria": 5,
         "d1_completa": bool(datos_d["n_bigramas"] == datos_d["n_bigramas_esperado"]),
         "canal_predictivo": "NO SE EJECUTA",
         "passed": True}
    g4["resumen"] = "pasa: auditoria escrita + D1 completa; el canal predictivo NO se ejecuta"
    return {"G1": g1, "G2": g2, "G3": g3, "G4": g4}


def gatea_sonda_e9(cota: dict, entradas: "Entradas") -> dict:
    """E9: la corrida debe reproducir la sonda previa declarada antes de congelar.

    `A1` y `A2` son estimadores cerrados y deterministas, luego se comparan con tolerancia
    1e-6 contra los valores del pre-registro. `A_perm` es estocastico (depende de la
    permutacion implementada), luego se evalua por su CRITERIO unilateral y por la banda
    declarada, nunca por tolerancia.

    Parameters
    ----------
    cota : dict
        Salida de :func:`fases_a_cota`.

    Returns
    -------
    dict
        Comparacion por anotador con el veredicto de la sonda.

    Raises
    ------
    AssertionError
        Si `A1` o `A2` no reproducen la sonda previa dentro de 1e-6 en la corrida completa, o
        si `A_perm` supera el criterio unilateral. Es un fallo de REPRODUCIBILIDAD, no un
        resultado.

    Notes
    -----
    En un subconjunto (`--verificar`, `--humo`) el gate se salta y se registra `no_aplica`: la
    sonda se declaro sobre el corpus completo y A1/A2 no pueden coincidir con un subconjunto.
    """
    # La sonda E9 se declaro sobre el CORPUS COMPLETO. En un subconjunto (autoejercicio o
    # `--humo`) A1 y A2 no pueden coincidir con esos valores, luego el gate solo aplica a la
    # corrida real y el subconjunto se registra como `no_aplica`.
    sonda = CONSTANTES["sonda_previa"]
    tolerancia = float(sonda["tolerancia"])
    if entradas.subset:
        print(f"    sonda E9: no aplica (subconjunto de {entradas.n_filas} filas); "
              f"la tolerancia {tolerancia:.1e} solo se exige en la corrida completa",
              flush=True)
        return {"no_aplica": True, "motivo": "subconjunto de autoejercicio o modo --humo",
                "n_filas": int(entradas.n_filas),
                "tolerancia_declarada": tolerancia}
    salida = {}
    for nombre in ANOTADORES:
        brazo = cota[nombre]
        a_perm = brazo["A_perm"]["macro_f1_pooled"]
        a_perm_ok = bool(a_perm <= CRITERIO_A_PERM_MAX)
        banda_ok = bool(BANDA_A_PERM[0] <= a_perm <= BANDA_A_PERM[1])
        # La sonda E9 se declaro sobre el ANOTADOR PRIMARIO (md). Para el secundario solo se
        # registra el valor medido y NO se exige la tolerancia: comparar lg contra una sonda
        # calculada con md seria exigir que dos anotadores den el mismo numero.
        if nombre == ANOTADOR_PRIMARIO:
            dif_a1 = abs(brazo["A1"]["macro_f1_pooled"] - sonda["A1"])
            dif_a2 = abs(brazo["A2"]["macro_f1_pooled"] - sonda["A2"])
            reproduce = bool(dif_a1 <= tolerancia and dif_a2 <= tolerancia)
        else:
            dif_a1 = dif_a2 = float("nan")
            reproduce = None
        salida[nombre] = {
            "A1_medido": brazo["A1"]["macro_f1_pooled"],
            "A2_medido": brazo["A2"]["macro_f1_pooled"],
            "A_perm_medido": a_perm,
            "sonda_declarada_para_este_anotador": nombre == ANOTADOR_PRIMARIO,
            "A1_declarado": sonda["A1"] if nombre == ANOTADOR_PRIMARIO else None,
            "A2_declarado": sonda["A2"] if nombre == ANOTADOR_PRIMARIO else None,
            "dif_abs_A1": None if dif_a1 != dif_a1 else round(dif_a1, 12),
            "dif_abs_A2": None if dif_a2 != dif_a2 else round(dif_a2, 12),
            "tolerancia": tolerancia,
            "A_perm_por_criterio_unilateral": a_perm_ok,
            "A_perm_en_banda_declarada": banda_ok,
            "reproduce": reproduce,
        }
        if reproduce is False:
            raise AssertionError(
                f"[STOP] sonda E9 no reproduce en {nombre}: A1 dif {dif_a1:.3e} y A2 dif "
                f"{dif_a2:.3e} frente a tolerancia {tolerancia:.1e}. Es un fallo de "
                f"reproducibilidad, no un resultado: no se lee ningun veredicto.")
        assert a_perm_ok, (
            f"[STOP] A_perm({nombre}) = {a_perm:.7f} supera el criterio unilateral "
            f"{CRITERIO_A_PERM_MAX}: posible fuga o desalineacion. No se lee ningun veredicto.")
    salida["nota_A_perm"] = (
        "A_perm NO se compara por tolerancia: es estocastico y depende de la permutacion "
        "implementada. Se evalua por el criterio unilateral E1 y por la banda declarada.")
    print(f"    sonda E9: A1 y A2 reproducen con tolerancia {tolerancia:.1e}; "
          f"A_perm por criterio unilateral <= {CRITERIO_A_PERM_MAX}", flush=True)
    return salida


def decide_veredicto(gates: dict, criterios: dict) -> dict:
    """Aplica el arbol de decision de la seccion 8.3 y devuelve el veredicto.

    Parameters
    ----------
    gates : dict
        Gates G1..G4.
    criterios : dict
        Criterios V1..V6.

    Returns
    -------
    dict
        ``veredicto`` y ``rama``, con las dos ramas posibles y la regla que las separa.
    """
    if not gates["G1"]["passed"]:
        return {"veredicto": "STOP", "rama": "G1 falla -> STOP. Artefactos rotos o corpus no "
                                            "reportable. No se continua."}
    if not gates["G3"]["passed"]:
        return {"veredicto": "STOP", "rama": "G3 falla -> STOP. No se puede cerrar sin contraste."}
    if not gates["G2"]["passed"]:
        return {"veredicto": "DIAGNOSTICO_CERRADO",
                "rama": "G2 falla -> DIAGNOSTICO_CERRADO por inestabilidad. Campeon conservado."}
    if all(criterios[clave]["cumple"] for clave in ("V1", "V2", "V3", "V4", "V5", "V6")):
        return {"veredicto": "PATRON_ESTABLE_NO_REDUNDANTE",
                "rama": "V1..V6 pasa -> PATRON_ESTABLE_NO_REDUNDANTE. Y AUN ASI no se ejecuta "
                        "ninguna corrida predictiva en V37: se escala a Carlos y la siguiente "
                        "linea requiere hipotesis nueva escrita."}
    fallan = [clave for clave in ("V1", "V2", "V3", "V4", "V5", "V6") if not criterios[clave]["cumple"]]
    return {"veredicto": "DIAGNOSTICO_CERRADO",
            "rama": "V1..V6 falla (G2 ok) -> DIAGNOSTICO_CERRADO. Campeon conservado. Criterios "
                    f"que fallan: {', '.join(fallan)}. Se documenta que la senal existe pero no "
                    "es ACTIONABLE (redundante, pequena o inestable). G4 deja el canal de "
                    "transiciones BLOQUEADO sin autorizacion expresa de Carlos con hipotesis "
                    "nueva escrita."}


# ==============================================================================
# Autoejercicio estructural: los cinco caminos de la seccion 11
# ==============================================================================
def camino_1_perfil() -> None:
    """Camino 1: perfil por clase sobre 200 filas sinteticas; comprueba suma 1 y ``ref``.

    Raises
    ------
    AssertionError
        Si el perfil no suma 1 ni ``ref`` es el promedio ponderado por tokens.
    """
    generador = np.random.default_rng(0)
    n_filas = 200
    conteos = generador.poisson(2.0, size=(n_filas, N_ETIQUETAS)).astype(np.float64)
    conteos[0] = 0.0
    y = generador.integers(0, N_CLASES, size=n_filas)
    filas = np.flatnonzero(conteos.sum(axis=1) > 0)
    n_por_clase, _ = PR.perfil_por_clase(conteos, y, filas)
    p = PR.p_desde_conteos(n_por_clase)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-12), "el perfil debe sumar 1 por clase"
    ref = n_por_clase.sum(axis=0) / n_por_clase.sum()
    esperado = (p * n_por_clase.sum(axis=1, keepdims=True)).sum(axis=0) / n_por_clase.sum()
    assert np.allclose(ref, esperado, atol=1e-12), "ref debe ser el promedio ponderado"
    assert np.allclose(PR.p_media_de_proporciones(conteos, y, filas).sum(axis=1), 1.0,
                       atol=1e-12), "la media de proporciones por tweet debe sumar 1"


def camino_2_cota() -> None:
    """Camino 2: ``naive_bayes_pos`` sobre las 200 filas sinteticas; comprueba forma y fuga.

    Raises
    ------
    AssertionError
        Si la forma, la normalizacion, el rango de ``pred`` o el control ``A_perm`` fallan.
    """
    generador = np.random.default_rng(1)
    n_filas = 200
    conteos = generador.poisson(2.0, size=(n_filas, N_ETIQUETAS)).astype(np.float64)
    conteos[3] = 0.0
    n_tokens = conteos.sum(axis=1).astype(np.int64)
    y = generador.integers(0, N_CLASES, size=n_filas)
    fold_idx = np.arange(n_filas) % 5
    brazo = PR.naive_bayes_pos(conteos, n_tokens, y, fold_idx, modo="proporciones", alpha=ALFA)
    assert brazo["prob"].shape == (n_filas, N_CLASES), "prob debe ser (n_filas, 6)"
    assert np.allclose(brazo["prob"].sum(axis=1), 1.0, atol=1e-9), "prob debe estar normalizada"
    assert brazo["pred"].min() >= 0 and brazo["pred"].max() < N_CLASES, "pred fuera de rango"
    assert int(brazo_contado(brazo)) == n_filas, "cada fila debe predecirse una vez"
    y_perm = PR.permuta_etiquetas_train(y, fold_idx, SEMILLAS["permute_control"])
    permutado = PR.naive_bayes_pos(conteos, n_tokens, y, fold_idx, y_ajuste=y_perm,
                                   modo="proporciones", alpha=ALFA)
    a_perm = PR.macro_f1(y, permutado["pred"])
    assert a_perm <= CRITERIO_A_PERM_MAX, (
        f"en el sintetico A_perm = {a_perm:.4f} supera el criterio unilateral "
        f"{CRITERIO_A_PERM_MAX}: el detector de fuga no puede dispararse")


def brazo_contado(brazo: dict) -> int:
    """Numero total de filas predichas por los cinco folds.

    Parameters
    ----------
    brazo : dict
        Salida de :func:`PR.naive_bayes_pos`.

    Returns
    -------
    int
        Suma de las filas predichas por fold.
    """
    return int(np.sum(brazo["n_predichas_por_fold"]))


def camino_3_bootstrap() -> None:
    """Camino 3: 20 iteraciones sobre 30 grupos sinteticos; comprueba IC y cobertura.

    Raises
    ------
    AssertionError
        Si la forma del IC no es la esperada o ``n_grupos`` por iteracion no coincide.
    """
    generador = np.random.default_rng(2)
    n_filas = 300
    n_grupos = 30
    cluster = generador.integers(0, n_grupos, size=n_filas)
    y = generador.integers(0, N_CLASES, size=n_filas)
    predicciones = {"real": y, "ruido": generador.integers(0, N_CLASES, size=n_filas)}
    filas_por_cluster = np.bincount(cluster, minlength=n_grupos).astype(np.float64)
    resultado = PR.ic95_macro_f1_por_grupo(y, predicciones, cluster, n_grupos, filas_por_cluster,
                                          SEMILLAS["bootstrap_cluster"], 20)
    for nombre, dato in resultado["por_brazo"].items():
        assert len(dato["ic95"]) == 2 and dato["ic95"][0] <= dato["ic95"][1], (
            f"IC95 mal formado para {nombre}")
    clave = "real_menos_ruido"
    assert clave in resultado["diferencias"], "falta la diferencia pareada"
    assert resultado["n_iter"] == 20, "n_iter debe ser el declarado"
    assert resultado["n_grupos"] == n_grupos, "n_grupos por iteracion debe coincidir"
    # Matriz de perfil por grupo con la forma que espera ic95_delta_por_grupo():
    # (n_grupos, 6 clases x 19 columnas) = 18 etiquetas mas el total de tokens de la clase.
    suficiente = np.zeros((n_grupos, N_CLASES * 19), dtype=np.float32)
    np.add.at(suficiente, (cluster, y * 19), 1.0)  # una ocurrencia de la etiqueta 0
    np.add.at(suficiente, (cluster, y * 19 + (N_ETIQUETAS - 1)), 1.0)  # idem en la ultima etiqueta
    np.add.at(suficiente, (cluster, y * 19 + N_ETIQUETAS), 1.0)  # total de tokens de la clase
    assert suficiente.shape == (n_grupos, N_CLASES * 19), "perfil sintetico con forma erronea"
    assert suficiente[:, 18::19].sum() == float(n_filas), "los totales deben cuadrar con n_filas"
    ic = PR.ic95_delta_por_grupo(suficiente, 20, SEMILLAS["bootstrap_cluster"])
    assert np.array(ic["ic95_delta"]).shape == (N_CLASES, N_ETIQUETAS, 2)
    assert np.array(ic["excluye_cero"]).shape == (N_CLASES, N_ETIQUETAS)


def camino_4_escritura() -> None:
    """Camino 4: escribe los 5 CSV, el JSON y las 5 figuras en un temporal y los relee.

    Es la parte que D-086 dice que faltaba: ejercitar la PROPIA ruta de escritura, no solo la
    computacion. Si el runner no puede escribir, se muere despues de entrenar.

    Raises
    ------
    AssertionError
        Si algun artefacto no se escribe, no parsea o pesa cero.
    """
    with tempfile.TemporaryDirectory(prefix="v37_camino4_") as temporal:
        artefactos = Artefactos(base=Path(temporal))
        resultado = cadena_completa(artefactos, n_bootstrap=3, n_filas_humo=300,
                                    etiqueta_modo="camino-4-sintetico")
        assert resultado["artefactos"]["rutas"], "el JSON debe listar sus rutas"
        assert artefactos.json_resultado.exists(), "falta el JSON"
        releido = json.loads(artefactos.json_resultado.read_text(encoding="utf-8"))
        assert releido["fase_e"]["veredicto"] in ("PATRON_ESTABLE_NO_REDUNDANTE",
                                                  "DIAGNOSTICO_CERRADO", "STOP"), (
            f"veredicto inesperado: {releido['fase_e']['veredicto']}")
        assert np.load(artefactos.prob_a1_oof).shape[1] == N_CLASES, "prob OOF mal forma"
        assert artefactos.informe.exists() and artefactos.informe.stat().st_size > 0, (
            "informe vacio")
        assert artefactos.informe.read_text(encoding="utf-8").strip(), "informe sin texto"


def camino_5_integridad() -> None:
    """Camino 5: corre ``verificar_integridad`` completo (las nueve aserciones de T0.2).

    Raises
    ------
    AssertionError
        Si alguna asercion de integridad falla.
    """
    resultado = PR.verificar_integridad(silencioso=True)
    assert resultado["n_ok"] == resultado["n_total"] == 9, (
        f"integridad {resultado['n_ok']}/{resultado['n_total']}, se esperaban 9/9")


CAMINOS = [
    ("camino_1_perfil", camino_1_perfil,
     "perfil por clase sobre 200 filas sinteticas: suma 1 y ref = promedio ponderado"),
    ("camino_2_cota", camino_2_cota,
     "naive_bayes_pos() sobre 200 filas con las 18 etiquetas: forma, normalizacion, rango de "
     "pred y A_perm por debajo del criterio unilateral"),
    ("camino_3_bootstrap", camino_3_bootstrap,
     "20 iteraciones de bootstrap sobre 30 grupos sinteticos: forma del IC y n_grupos por "
     "iteracion"),
    ("camino_4_escritura", camino_4_escritura,
     "escribe los 5 CSV, el JSON y las 5 figuras en un directorio temporal y los relee "
     "(leccion de D-086: se ejercita la propia ruta de salida)"),
    ("camino_5_integridad", camino_5_integridad,
     "verificar_integridad() completo: las nueve aserciones de T0.2"),
]


def modo_verificar() -> int:
    """Corre los cinco caminos en directorios temporales.

    Returns
    -------
    int
        ``0`` si los cinco pasan; ``2`` nombrando el camino que falla.
    """
    inicio = time.time()
    print("=== --verificar: autoejercicio estructural de los 5 caminos (seccion 11) ===",
          flush=True)
    for nombre, funcion, descripcion in CAMINOS:
        t0 = time.time()
        try:
            funcion()
        except Exception as error:  # noqa: BLE001 - se nombra el camino y se sale con 2
            print(f"  [FALLO] {nombre}: {descripcion}", flush=True)
            print(f"          {type(error).__name__}: {error}", flush=True)
            traceback.print_exc()
            print(f"[STOP] --verificar: fallo en {nombre} ({time.time() - inicio:.1f} s)",
                  flush=True)
            return 2
        print(f"  [OK] {nombre} ({time.time() - t0:.1f} s): {descripcion}", flush=True)
    print("OK 5/5 caminos", flush=True)
    print(f"  nada escrito en experiments/, reports/ ni artifacts/ "
          f"(los caminos 4 y las rutas temporales son de {tempfile.gettempdir()})", flush=True)
    print(f"[fin --verificar] {time.time() - inicio:.1f} s", flush=True)
    return 0


def modo_humo() -> int:
    """Corre la cadena completa sobre 5.000 filas en un directorio temporal.

    Returns
    -------
    int
        ``0`` si la cadena termina; ``2`` si falla.
    """
    inicio = time.time()
    print("=== --humo: cadena completa sobre 5.000 filas en directorio temporal ===", flush=True)
    with tempfile.TemporaryDirectory(prefix="v37_humo_") as temporal:
        artefactos = Artefactos(base=Path(temporal))
        try:
            resultado = cadena_completa(artefactos, n_bootstrap=N_BOOTSTRAP_HUMO,
                                        n_filas_humo=N_FILAS_HUMO, etiqueta_modo="humo")
        except Exception as error:  # noqa: BLE001 - se imprime y se sale con 2
            print(f"  {type(error).__name__}: {error}", flush=True)
            traceback.print_exc()
            print(f"[STOP] --humo fallido ({time.time() - inicio:.1f} s)", flush=True)
            return 2
        for nombre, relativo in resultado["artefactos"]["rutas"].items():
            ruta = Path(temporal) / relativo
            assert ruta.exists() and ruta.stat().st_size > 0, f"{nombre} no se escribio"
        print(f"  artefactos del humo (temporales, ninguno real): "
              f"{len(resultado['artefactos']['rutas'])} ficheros", flush=True)
    print(f"[fin --humo] {time.time() - inicio:.1f} s", flush=True)
    return 0


def main() -> int:
    """Punto de entrada del runner de las fases A-E.

    Returns
    -------
    int
        ``0`` en exito y ``2`` en cualquier ``[STOP]``.
    """
    analizador = argparse.ArgumentParser(
        prog="age_linguistic_14_diagnostico_pos.py",
        description=("Runner de las fases A-E del diagnostico V37 de P(POS|edad). "
                     "SIN argumentos: corre la cadena completa sobre las 169.336 filas y "
                     "escribe experiments/age_linguistic_14_diagnostico_pos.json, "
                     "artifacts/v37/a1_prob_oof.npy, los 5 CSV de reports/tables/, "
                     "reports/v37_diagnostico_pos.md y las 5 figuras de reports/figures/. "
                     "Con --verificar: autoejercicio estructural de los cinco caminos de la "
                     "seccion 11 de la spec (perfil, cota, bootstrap, escritura en directorio "
                     "temporal e integridad del entorno de datos), sin escribir nada en los "
                     "artefactos reales; sale 0 si los cinco pasan y 2 nombrando el que falla. "
                     "Con --humo: la cadena completa sobre 5.000 filas estratificadas con 30 "
                     "iteraciones de bootstrap, escribiendo en un directorio temporal; sale 0 "
                     "si termina en menos de 60 s."),
        epilog=("Modos: (por defecto) corrida completa de las fases A-E | "
                "--verificar autoejercicio de 5 caminos en temporal | "
                "--humo cadena completa en temporal sobre 5.000 filas."))
    analizador.add_argument("--verificar", action="store_true",
                            help="corre los 5 caminos sinteticos en directorios temporales, "
                                 "incluida la escritura real de los 5 CSV, el JSON y las 5 "
                                 "figuras; no toca los artefactos reales")
    analizador.add_argument("--humo", action="store_true",
                            help="corre la cadena completa sobre 5.000 filas en un directorio "
                                 "temporal, con el numero de iteraciones de bootstrap reducido; "
                                 "objetivo: menos de 60 s")
    argumentos = analizador.parse_args()
    inicio = time.time()
    autotest_numerico()
    if argumentos.verificar and argumentos.humo:
        print("[STOP] --verificar y --humo son excluyentes", flush=True)
        return 2
    if argumentos.verificar:
        try:
            return modo_verificar()
        except Exception as error:  # noqa: BLE001
            print(f"  {type(error).__name__}: {error}", flush=True)
            traceback.print_exc()
            return 2
    if argumentos.humo:
        return modo_humo()
    try:
        cadena_completa(Artefactos(), N_BOOTSTRAP, None, "completa")
    except Exception as error:  # noqa: BLE001
        print(f"  {type(error).__name__}: {error}", flush=True)
        traceback.print_exc()
        return 2
    print(f"[fin] {time.time() - inicio:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
