"""V29-03: los vecinos de similitud, pueden corregir errores de V03 o solo lo estorban?

Que responde
------------
Fases 5, 6, 7 y 8 de la directiva V29. V29-02 mostro que relajar la similitud produce grupos
**muy puros**: con Jaccard de char 3-5, los pares en la franja 0,50-0,60 comparten etiqueta
con probabilidad 0,8405 frente a 0,5080 que se esperaria si fueran independientes. Y al mismo
tiempo la cobertura es de 1,91 %.

Eso es exactamente el **resultado B** que la directiva describe: senal descriptiva que puede no
ser predictiva. Aqui se responde, y para responderla hacen falta los folds.

Lo que se mide
--------------
Para cada fila de validacion, usando **solo filas de su training fold** como vecinos:
similitud maxima, numero de vecinos, etiqueta mayoritaria de los vecinos, su confianza, si el
vecino contradice a V03 y si el vecino tiene la etiqueta correcta. Y de ahi los dos conteos que
la directiva exige: ``V03 wrong + vecino acierta`` (rescates) y
``V03 correct + vecino falla`` (danos).

Encontrar vecinos puros no basta. Lo que importa es si **cambian la decision de acierto**.

Protocolo anti-leakage (Fase 7, obligatorio)
-------------------------------------------
1. Cada fila de validacion se consulta **solo** contra filas de su training fold.
2. La etiqueta de una fila de validacion no se usa para nada, ni para ella ni para otra.
3. Las senales (etiqueta mayoritaria, confianza) se calculan **exclusivamente** con etiquetas
   del training fold.
4. El vectorizador se ajusta **dentro de cada training fold**, nunca con validacion.
5. Los grupos se construyen con el training fold; la validacion recibe la coincidencia calculada
   contra ese training, no una participacion propia.

Y hay una segunda capa, la que evita el overestimate de V12-01: **la regla se elige dentro del
training fold** con CV interna, y despues se aplica una sola vez a la validacion externa.
Elegir la mejor regla sobre el mismo corrected_cv que despues se reporta fue lo que produjo el
falso positivo de V12-01, y aqui queda cerrado por construccion.

La regla, que es la mas simple posible
--------------------------------------
``usar el vecino si hay al menos k vecinos por encima del umbral y la mayoria tiene confianza
al menos tau``

No hay meta-modelo, no hay pesos aprendidos, no hay blending. La unica decision es si la
confianza de la mayoria alcanza ``tau``.

Optimizacion que no cambia el resultado
---------------------------------------
Las senales del vecino no dependen de ``umbral`` ni de ``tau``: dependen solo del metodo y de
``k``. Se calculan una vez por combinacion y despues se recorre la rejilla de umbrales y tau
sobre ellas. Es un ahorro de tiempo, no una decision de diseno.

Uso
---
    /home/carlos/venv/bin/python experiments/v29_03_vecinos_corrigen_errores.py
"""

from __future__ import annotations

import json
import re
import resource
import sys
import time
from collections import namedtuple
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train
from src.evaluation import metricas_multietiqueta
from src.splits import grupos_fusionados, pliegues_corregidos

EXPERIMENTO = "V29-03 (los vecinos de similitud corrigen errores de V03 o solo los estorban?)"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v29_03_vecinos_corrigen_errores.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "v29_03_vecinos_corrigen_errores.md"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
OOF_CANONICO = PROJECT_ROOT / "features" / "oof_v03_final.npz"

ETIQUETAS = ("HS", "TR", "AG")
RE_URL_HISTORICO = re.compile(r"https?://\S+")
RE_MENCION = re.compile(r"@\w+")
RE_HASHTAG = re.compile(r"#\w+")
RE_PUNCT = re.compile(r"[!\u00bf?.,;:()\[\]{}\"'`\-_/\\|+*=<>$%&]")
CORRECTED_CV_V03 = 0.7772114525165721

# Rejilla pequena y justificada: las mejores franjas de V29-02 caen entre 0,40 y 0,60 para los
# tres metodos, asi que no tiene sentido explorar mas alla. k se limita a un rango corto porque
# los vecinos muy lejanos ya no son "el mismo texto".
REJILLA_UMBRAL = (0.60, 0.50, 0.40, 0.30)
REJILLA_K = (1, 3, 5, 10)
REJILLA_TAU = (0.50, 0.60, 0.75, 0.90, 1.00)
N_INTERNOS = 4
TAU_MAYORIA = 0.5
METODOS = ("B_coseno_char4_agresivo", "D_jaccard_char35_agresivo", "E_jaccard_tokens_agresivo")
FRANJAS_DESGLOSE = ((0.90, 1.01), (0.70, 0.90), (0.50, 0.70), (0.30, 0.50))

Senales = namedtuple("Senales", "pred conf vecinos sim_max sim_2")
Ajuste = namedtuple("Ajuste", "macro f1_por_label n_candidatas")


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V29-03] {mensaje}", flush=True)


def memoria_pico_mb() -> float:
    """Resident set maximum del proceso, en megabytes.

    Returns:
        La memoria maxima residente en MB.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def normalizar_agresiva(texto: str) -> str:
    """Quita URLs, menciones, hashtags y puntuacion, y pasa a minusculas.

    Args:
        texto: texto original.

    Returns:
        Texto normalizado de forma agresiva.
    """
    limpio = RE_MENCION.sub(" ", RE_HASHTAG.sub(" ", RE_PUNCT.sub(" ", texto.lower())))
    limpio = RE_URL_HISTORICO.sub(" ", limpio)
    return re.sub(r"\s+", " ", limpio).strip()


def construir_metodo(metodo: str, textos_train: list[str]) -> Any:
    """Ajusta el vectorizador del metodo **solo** sobre el training fold.

    Args:
        metodo: clave del metodo.
        textos_train: textos del training fold.

    Returns:
        Vectorizador ajustado.
    """
    if metodo.startswith("B_coseno"):
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 4), min_df=1).fit(textos_train)
    if metodo.startswith("D_jaccard_char"):
        return CountVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=1, binary=True
        ).fit(textos_train)
    return CountVectorizer(
        analyzer="word", ngram_range=(1, 1), binary=True, token_pattern=r"(?u)\b\w+\b"
    ).fit(textos_train)


def matriz_similitud(metodo: str, vectorizador: Any, consulta: list[str], train: list[str]) -> np.ndarray:
    """Similitud de las filas de consulta contra las del training.

    Args:
        metodo: clave del metodo.
        vectorizador: vectorizador ya ajustado en el training.
        consulta: textos a consultar.
        train: textos del training.

    Returns:
        Matriz de similitud de forma ``(len(consulta), len(train))``.
    """
    matriz_train = vectorizador.transform(train).astype(np.float32)
    matriz_consulta = vectorizador.transform(consulta).astype(np.float32)
    if metodo.startswith("B_coseno"):
        return np.asarray((matriz_consulta @ matriz_train.T).todense(), dtype=np.float32)
    interseccion = np.asarray((matriz_consulta @ matriz_train.T).todense(), dtype=np.float32)
    tamanos_consulta = np.asarray(matriz_consulta.sum(axis=1)).ravel()
    tamanos_train = np.asarray(matriz_train.sum(axis=1)).ravel()
    union = tamanos_consulta[:, None] + tamanos_train[None, :] - interseccion
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, interseccion / union, 0.0).astype(np.float32)


def senales_vecino(
    similitud: np.ndarray, etiquetas_train: np.ndarray, k: int
) -> Senales:
    """Senal de vecino para cada fila de consulta y cada label.

    Solo recibe etiquetas de training. Nunca ve las de la fila consultada.

    Args:
        similitud: matriz de similitud consulta contra training.
        etiquetas_train: matriz de etiquetas del training fold.
        k: numero maximo de vecinos a considerar.

    Returns:
        ``Senales`` con prediccion del vecino, confianza de la mayoria, numero de vecinos
        efectivos, similitud maxima y segunda similitud.
    """
    n_consulta = similitud.shape[0]
    n_labels = etiquetas_train.shape[1]
    k_efectivo = min(k, similitud.shape[1])
    orden = np.argsort(-similitud, axis=1)[:, :k_efectivo]
    filas = np.arange(n_consulta)[:, None]
    top = similitud[filas, orden]
    # Un vecino con similitud 0 no existe. Se marca con -1 para que no cuente ni para el
    # denominador de la confianza ni como vecino efectivo.
    indices = np.where(top > 0, orden, -1)
    vecinos_efectivos = (indices >= 0).sum(axis=1).astype(np.float32)
    prediccion = np.zeros((n_consulta, n_labels), dtype=np.int64)
    confianza = np.zeros((n_consulta, n_labels), dtype=np.float32)
    sim_max = np.zeros((n_consulta, n_labels), dtype=np.float32)
    sim_2 = np.zeros((n_consulta, n_labels), dtype=np.float32)
    for j in range(n_labels):
        etiquetas_vecinas = np.where(indices >= 0, etiquetas_train[indices, j], -1)
        positivos = (etiquetas_vecinas == 1).sum(axis=1).astype(np.float32)
        total = (etiquetas_vecinas >= 0).sum(axis=1).astype(np.float32)
        confianza[:, j] = np.where(total > 0, positivos / np.maximum(total, 1.0), 0.0)
        prediccion[:, j] = (confianza[:, j] >= 0.5).astype(np.int64)
        sim_max[:, j] = top[:, 0]
        sim_2[:, j] = top[:, 1] if top.shape[1] > 1 else 0.0
    return Senales(pred=prediccion, conf=confianza, vecinos=vecinos_efectivos,
                    sim_max=sim_max, sim_2=sim_2)


def aplicar_regla(senales: Senales, umbral: float, k: int, tau: float) -> np.ndarray:
    """Aplica la regla de decision del vecino.

    Args:
        senales: senales ya calculadas.
        umbral: similitud minima para considerar vecino.
        k: numero minimo de vecinos.
        tau: confianza minima de la mayoria.

    Returns:
        Matriz con 1 donde se usa el vecino del training.
    """
    # `vecinos` es un conteo por fila y el resto es por fila y label, asi que se
    # expande a una columna para que el broadcast sea explicito.
    vecinos = senales.vecinos[:, None]
    usa = (vecinos >= k) & (senales.sim_max >= umbral) & (senales.conf >= tau)
    return np.where(usa, senales.pred, 0).astype(np.int64)


def combinar(
    pred_base: np.ndarray, celdas: np.ndarray
) -> np.ndarray:
    """Sustituye la prediccion base por la del vecino donde la regla se cumple.

    Args:
        pred_base: prediccion de partida, de forma ``(n, n_labels)``.
        celdas: matriz con 1 donde se usa el vecino.

    Returns:
        Prediccion combinada.
    """
    salida = pred_base.copy()
    usar = celdas.sum(axis=1) > 0
    if usar.any():
        salida[usar] = celdas[usar]
    return salida


def macro_f1(y: np.ndarray, pred: np.ndarray) -> tuple[float, dict[str, float]]:
    """Macro-F1 y F1 por label.

    Args:
        y: etiquetas verdaderas.
        pred: predicciones.

    Returns:
        Tupla con el macro-F1 y el F1 de cada label.
    """
    # La metrica primaria acordada en el proyecto es `f1_media_columnas` (D-009), y el
    # desglose por label viene anidado en `f1_por_etiqueta`.
    m = metricas_multietiqueta(y, pred)
    return float(m["f1_media_columnas"]), {et: float(m["f1_por_etiqueta"][et]) for et in ETIQUETAS}


def pliegues_internos(
    grupos_tr: np.ndarray, y_tr: np.ndarray, n_splits: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Parte el training fold en pliegues internos respetando los grupos.

    Args:
        grupos_tr: grupos de las filas del training fold.
        y_tr: etiquetas de las filas del training fold, usadas solo para estratificar.
        n_splits: numero de pliegues internos.

    Returns:
        Lista de pares de indices, en posiciones relativas al training fold.
    """
    estrato = np.zeros(len(y_tr), dtype=np.int64)
    for j in range(y_tr.shape[1]):
        estrato = estrato * 2 + y_tr[:, j]
    divisor = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    salida = []
    for rel_tr, rel_va in divisor.split(np.zeros(len(y_tr)), estrato, groups=grupos_tr):
        comunes = set(grupos_tr[rel_tr]).intersection(grupos_tr[rel_va])
        if comunes:  # pragma: no cover - defensa; StratifiedGroupKFold lo garantiza
            raise RuntimeError("fuga entre pliegues internos")
        salida.append((np.sort(rel_tr), np.sort(rel_va)))
    return salida


def main() -> None:
    """Ejecuta la evaluacion de los vecinos de similitud sobre los folds de V03.

    Returns:
        None.
    """
    inicio = time.time()
    log(f"{EXPERIMENTO}. NINGUNA submission en esta fase")

    train = read_train()
    particion = pd.read_csv(PARTICION)
    ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
    desarrollo = train[train["id"].astype(int).isin(ids_dev)].reset_index(drop=True)
    grupos_dev = grupos_fusionados(desarrollo)
    pliegues = pliegues_corregidos(desarrollo, grupos_dev)
    y = desarrollo[list(ETIQUETAS)].to_numpy(dtype=np.int64)
    textos = [normalizar_agresiva(t) for t in desarrollo["text"].astype(str)]
    log(f"   desarrollo {len(desarrollo)} filas | {len(pliegues)} pliegues canonicos de V03")

    oof = np.load(OOF_CANONICO, allow_pickle=True)
    pred_oof = np.asarray(oof["pred"], dtype=np.int64)
    np.asarray(oof["ids"], dtype=np.int64)
    mapa = dict(zip(desarrollo["id"].astype(int), pred_oof))
    pred_v03 = np.array([mapa[int(i)] for i in desarrollo["id"].astype(int)])
    log(f"   OOF canonico de V03 cargado, corrected_cv de referencia {CORRECTED_CV_V03:.6f}")

    log("1/5 controles de la senal de vecino")
    vectorizador_ctrl = construir_metodo("D_jaccard_char35_agresivo", textos)
    sim_self = matriz_similitud(
        "D_jaccard_char35_agresivo", vectorizador_ctrl, textos, textos
    )
    senales_self = senales_vecino(sim_self, y, k=1)
    controles = {
        "similitud consigo mismo, minimo": float(np.diag(sim_self).min()),
        "simetria, asimetria maxima": float(np.max(np.abs(sim_self - sim_self.T))),
        "rango de similitud": [float(sim_self.min()), float(sim_self.max())],
        "vecinos efectivos en la diagonal, minimo": float(senales_self.vecinos.min()),
        "la senal de uno mismo reproduce su etiqueta": float(
            np.mean(senales_self.pred == y)
        ),
        "nota": (
            "la senal de vecino es lo bastante fuerte como para reproducir la etiqueta de la "
            "propia fila, exactamente en 1,0000. Por eso el aislamiento depende por completo de "
            "que la validacion solo consulte al training fold, y no de una limitacion del codigo. "
            "Un predictor de vecino sin folds seria fuga total"
        ),
    }
    log(f"   similitud consigo mismo: minimo {controles['similitud consigo mismo, minimo']:.4f}")
    log(f"   simetria: asimetria maxima {controles['simetria, asimetria maxima']:.1e}")
    log(
        f"   la senal de uno mismo reproduce su etiqueta: "
        f"{controles['la senal de uno mismo reproduce su etiqueta']:.4f}"
    )
    log("   por eso el aislamiento lo imponen los folds, no el codigo: queda declarado")

    log("2/5 senales de validacion para todo metodo y k, y selector anidado")
    oof_challenger = pred_v03.copy()
    senales_oof: list[Senales] = []
    reglas: list[dict[str, Any]] = []
    # Cache de las senales de validacion por (metodo, k). La matriz de similitud no depende de
    # las etiquetas, asi que calcularla una vez y reutilizarla para el selector y para el
    # barrido exhaustivo es un ahorro de tiempo, no una decision de diseno.
    senales_cache: dict[tuple[str, int, int], Senales] = {}
    for f, (idx_tr, idx_va) in enumerate(pliegues):
        y_tr = y[idx_tr]
        pred_v03_tr = pred_v03[idx_tr]
        textos_tr = [textos[i] for i in idx_tr]
        textos_va = [textos[i] for i in idx_va]
        for metodo_cache in METODOS:
            vec_cache = construir_metodo(metodo_cache, textos_tr)
            matriz_va = matriz_similitud(metodo_cache, vec_cache, textos_va, textos_tr)
            for k_cache in REJILLA_K:
                senales_cache[(metodo_cache, k_cache, f)] = senales_vecino(
                    matriz_va, y_tr, k=k_cache
                )
        internos = pliegues_internos(grupos_dev[idx_tr], y_tr, N_INTERNOS)
        mejor: dict[str, Any] | None = None
        for metodo in METODOS:
            vectorizador = construir_metodo(metodo, textos_tr)
            sim_tr = matriz_similitud(metodo, vectorizador, textos_tr, textos_tr)
            for k in REJILLA_K:
                senales_int = [
                    senales_vecino(sim_tr[np.ix_(rel_va, rel_tr)], y_tr[rel_tr], k)
                    for rel_tr, rel_va in internos
                ]
                for umbral in REJILLA_UMBRAL:
                    for tau in REJILLA_TAU:
                        pred_int = pred_v03_tr.copy()
                        n_candidatas = 0
                        for (rel_tr, rel_va), sen in zip(internos, senales_int, strict=True):
                            celdas = aplicar_regla(sen, umbral, k, tau)
                            n_candidatas += int(celdas.sum())
                            pred_int[rel_va] = combinar(pred_v03_tr[rel_va], celdas)
                        if n_candidatas == 0:
                            continue
                        macro, por_label = macro_f1(y_tr, pred_int)
                        if mejor is None or macro > mejor["macro_interna"]:
                            mejor = {
                                "metodo": metodo, "k": k, "umbral": umbral, "tau": tau,
                                "macro_interna": macro, "f1_interna": por_label,
                                "n_candidatas_internas": n_candidatas,
                            }
        if mejor is None:
            raise RuntimeError(
                f"pliegue {f}: ninguna regla produjo vecinos en el training fold. "
                "No se puede evaluar nada sin al menos un vecino"
            )
        # Aplicacion a validacion externa, una sola vez, con la regla ya elegida
        sen = senales_cache[(mejor["metodo"], mejor["k"], f)]
        celdas = aplicar_regla(sen, mejor["umbral"], mejor["k"], mejor["tau"])
        oof_challenger[idx_va] = combinar(pred_v03[idx_va], celdas)
        senales_oof.append(sen)
        reglas.append({**mejor, "pliegue": f, "n_filas_validadas": len(idx_va),
                       "n_celdas_cambiadas": int(celdas.sum())})
        log(
            f"   pliegue {f}: {mejor['metodo'][:18]:<18} k {mejor['k']:>2} "
            f"u {mejor['umbral']:.2f} tau {mejor['tau']:.2f} | macro interna "
            f"{mejor['macro_interna']:.4f} | celdas {int(celdas.sum()):>4} de {len(idx_va) * 3}"
        )

    log("3/5 evaluacion OOF: rescates, danos y delta")
    metricas = evaluar(y, pred_v03, oof_challenger)
    log(f"   corrected_cv V03         {metricas['macro_f1_v03']:.6f}")
    log(f"   corrected_cv con vecino  {metricas['macro_f1_challenger']:.6f}")
    log(f"   delta OOF                {metricas['delta_macro_f1']:+.6f}")
    log(f"   cobertura del predictor  {metricas['cobertura_celdas']:.4f}")
    log(f"   rescates {metricas['rescates']} | danos {metricas['danios']} | "
        f"neto {metricas['neto']:+d} | celdas {metricas['celdas_modificadas']}")
    for et in ETIQUETAS:
        d = metricas["por_label"][et]
        log(
            f"      {et}: {d['f1_v03']:.4f} -> {d['f1_challenger']:.4f} ({d['delta']:+.4f}) | "
            f"rescates {d['rescates']:>3} danos {d['danios']:>3} neto {d['neto']:+4d} | "
            f"celdas {d['celdas_modificadas']}"
        )

    log("4/5 desglose por franja de similitud del vecino")
    desglose = desglose_por_similitud(
        pliegues, desarrollo, y, pred_v03, reglas, senales_oof
    )
    for d in desglose:
        log(
            f"   franja {d['franja']:<12} celdas {d['n_celdas']:>5} | rescates "
            f"{d['rescates']:>4} | danos {d['danios']:>4} | neto {d['neto']:+5d}"
        )

    log("5/5 el detalle que pide la Fase 5, sobre las filas donde V03 falla")
    detalle = detalle_de_vecinos(pliegues, desarrollo, y, pred_v03, reglas, senales_oof)
    log(f"   filas de desarrollo con algun error de V03: {detalle['n_filas_con_error']}")
    log(f"   de ellas, con vecino por encima del umbral elegido: "
        f"{detalle['n_filas_con_error_y_vecino']} "
        f"({detalle['fraccion_con_error_y_vecino']:.4f})")
    log(f"   de ellas, el vecino trae la etiqueta correcta: "
        f"{detalle['n_filas_error_con_vecino_correcto']}")
    log(f"   y el vecino contradice a V03: {detalle['n_filas_error_vecino_contradice']}")
    log(f"   filas con V03 correcto y vecino que lo contradice: "
        f"{detalle['n_filas_correcto_vecino_contradice']}")

    barrido = barrido_exhaustivo(pliegues, y, pred_v03, senales_cache)
    log(f"   barrido exhaustivo: {len(barrido)} configuraciones sin selector")
    log(f"   mejor configuracion por net: {barrido[0]['metodo']} k {barrido[0]['k']} "
        f"u {barrido[0]['umbral']:.2f} -> net {barrido[0]['neto']:+d}")
    log(f"   peor configuracion por net: {barrido[-1]['metodo']} k {barrido[-1]['k']} "
        f"u {barrido[-1]['umbral']:.2f} -> net {barrido[-1]['neto']:+d}")
    with_rescates = [b for b in barrido if b["rescates"] > 0]
    log(f"   configuraciones con AL MENOS un rescate: {len(with_rescates)} de {len(barrido)}")
    mejor_con_rescate = max(with_rescates, key=lambda b: b["neto"]) if with_rescates else None
    if mejor_con_rescate:
        log(
            f"   la mejor con rescates: {mejor_con_rescate['metodo']} k "
            f"{mejor_con_rescate['k']} u {mejor_con_rescate['umbral']:.2f} -> rescates "
            f"{mejor_con_rescate['rescates']} danos {mejor_con_rescate['danios']} "
            f"neto {mejor_con_rescate['neto']:+d}"
        )
    else:
        log("   NINGUNA configuracion produce un solo rescate en todo el OOF")

    tabla = tabla_de_mecanismos(metricas)
    informe = {
        "familia": "FUZZY_TEMPLATE_MATCHING",
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estado": "OOF_EVALUATED",
        "pregunta": (
            "los vecinos de similitud pueden corregir errores reales de V03, o solo producen "
            "grupos puros que no cambian ninguna decision"
        ),
        "alcance": (
            "selector de vecino con una sola regla y sin meta-modelo. La regla se elige dentro "
            "del training fold por CV interna, de modo que el OOF reportado no se usa para "
            "elegirla"
        ),
        "reglas_anti_leakage": {
            "validacion_consulta_solo_a_training": True,
            "etiquetas_de_validacion_no_usadas": True,
            "senales_calculadas_solo_con_etiquetas_de_training": True,
            "vectorizador_ajustado_dentro_de_cada_training_fold": True,
            "grupos_construidos_con_el_training_fold": True,
            "regla_elegida_en_cv_interna_del_training_fold": True,
            "declaracion_de_fuga_no_label": (
                "dentro de la CV interna del training fold, las filas internas de validacion ya "
                "formaban parte del texto usado para ajustar el vectorizador. No toca etiquetas, "
                "pero es una fuga transductiva y se declara en vez de ocultarse"
            ),
        },
        "rejilla": {
            "metodos": list(METODOS),
            "umbrales": list(REJILLA_UMBRAL),
            "k": list(REJILLA_K),
            "tau": list(REJILLA_TAU),
            "n_pliegues_internos": N_INTERNOS,
            "configuraciones_por_pliegue": len(METODOS) * len(REJILLA_K) * len(REJILLA_UMBRAL) * len(REJILLA_TAU),
        },
        "controles": controles,
        "regla_por_pliegue": reglas,
        "metricas": metricas,
        "desglose_por_similitud": desglose,
        "detalle_de_vecinos": detalle,
        "barrido_exhaustivo_sin_selector": barrido,
        "configuraciones_con_rescates": len(with_rescates),
        "tabla_de_mecanismos": tabla,
        "gastos": {
            "submissions": 0,
            "predictores_auxiliares": 1,
            "memoria_pico_mb": round(memoria_pico_mb(), 1),
            "segundos": round(time.time() - inicio, 1),
        },
    }
    REPORTE_JSON.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _informe_md(informe)
    log("   informe escrito en reports/v29_03_vecinos_corrigen_errores.{json,md}")
    log(f"   memoria pico {memoria_pico_mb():.0f} MB | tiempo {round(time.time() - inicio, 1)} s")
    log("VEREDICTO: evaluacion OOF completada, 0 submissions")


def evaluar(y: np.ndarray, pred_v03: np.ndarray, pred_challenger: np.ndarray) -> dict[str, Any]:
    """Metricas OOF del selector de vecino contra V03.

    Args:
        y: etiquetas verdaderas.
        pred_v03: predicciones de V03.
        pred_challenger: predicciones con vecino.

    Returns:
        Diccionario con metricas, rescates, danos, cobertura y desglose por label.
    """
    macro_v03, f1_v03 = macro_f1(y, pred_v03)
    macro_ch, f1_ch = macro_f1(y, pred_challenger)
    cambios = pred_v03 != pred_challenger
    por_label: dict[str, Any] = {}
    rescues = dam = 0
    for j, et in enumerate(ETIQUETAS):
        bien_v03 = pred_v03[:, j] == y[:, j]
        bien_ch = pred_challenger[:, j] == y[:, j]
        r = int(np.count_nonzero(~bien_v03 & bien_ch))
        d = int(np.count_nonzero(bien_v03 & ~bien_ch))
        rescues += r
        dam += d
        donde = cambios[:, j]
        por_label[et] = {
            "f1_v03": f1_v03[et],
            "f1_challenger": f1_ch[et],
            "delta": f1_ch[et] - f1_v03[et],
            "rescates": r,
            "danios": d,
            "neto": r - d,
            "celdas_modificadas": int(donde.sum()),
            "acierto_entre_las_celdas_modificadas": (
                float(bien_ch[donde].mean()) if donde.any() else None
            ),
        }
    return {
        "macro_f1_v03": macro_v03,
        "macro_f1_challenger": macro_ch,
        "delta_macro_f1": macro_ch - macro_v03,
        "f1_por_label_v03": f1_v03,
        "f1_por_label_challenger": f1_ch,
        "rescates": rescues,
        "danios": dam,
        "neto": rescues - dam,
        "celdas_modificadas": int(cambios.sum()),
        "cobertura_celdas": float(cambios.sum() / cambios.size),
        "por_label": por_label,
    }


def detalle_de_vecinos(
    pliegues: list[tuple[np.ndarray, np.ndarray]],
    desarrollo: pd.DataFrame,
    y: np.ndarray,
    pred_v03: np.ndarray,
    reglas: list[dict[str, Any]],
    senales_oof: list[Senales],
) -> dict[str, Any]:
    """El detalle de la Fase 5 sobre las filas donde V03 falla.

    Args:
        pliegues: pliegues canonicos.
        desarrollo: tabla del desarrollo.
        y: etiquetas verdaderas.
        pred_v03: predicciones de V03.
        reglas: regla elegida en cada pliegue.
        senales_oof: senales calculadas contra el training de cada pliegue.

    Returns:
        Diccionario con los conteos que pide la directiva.
    """
    filas_error = 0
    filas_error_con_vecino = 0
    filas_error_vecino_correcto = 0
    filas_error_vecino_contradice = 0
    filas_correcto_vecino_contradice = 0
    for f, (idx_tr, idx_va) in enumerate(pliegues):
        sen = senales_oof[f]
        regla = reglas[f]
        error_v03 = pred_v03[idx_va] != y[idx_va]
        # `vecinos` es un conteo por fila, el resto es por fila y label.
        con_vecino = (sen.vecinos[:, None] >= regla["k"]) & (sen.sim_max >= regla["umbral"])
        filas_error += int(error_v03.any(axis=1).sum())
        filas_error_con_vecino += int((error_v03.any(axis=1) & con_vecino.any(axis=1)).sum())
        filas_error_vecino_correcto += int(
            (error_v03 & con_vecino & (sen.pred == y[idx_va])).sum()
        )
        filas_error_vecino_contradice += int(
            (error_v03 & con_vecino & (sen.pred != pred_v03[idx_va])).sum()
        )
        filas_correcto_vecino_contradice += int(
            (~error_v03 & con_vecino & (sen.pred != pred_v03[idx_va])).sum()
        )
    return {
        "n_filas_con_error": filas_error,
        "n_filas_con_error_y_vecino": filas_error_con_vecino,
        "fraccion_con_error_y_vecino": (
            filas_error_con_vecino / filas_error if filas_error else 0.0
        ),
        "n_filas_error_con_vecino_correcto": filas_error_vecino_correcto,
        "n_filas_error_vecino_contradice": filas_error_vecino_contradice,
        "n_filas_correcto_vecino_contradice": filas_correcto_vecino_contradice,
        "nota": (
            "estas filas son celdas, no filas: una fila aporta hasta tres celdas, una por label"
        ),
    }


def desglose_por_similitud(
    pliegues: list[tuple[np.ndarray, np.ndarray]],
    desarrollo: pd.DataFrame,
    y: np.ndarray,
    pred_v03: np.ndarray,
    reglas: list[dict[str, Any]],
    senales_oof: list[Senales],
) -> list[dict[str, Any]]:
    """Reparte los rescates y los danos por franja de similitud del vecino.

    Args:
        pliegues: pliegues canonicos.
        desarrollo: tabla del desarrollo.
        y: etiquetas verdaderas.
        pred_v03: predicciones de V03.
        reglas: regla elegida en cada pliegue.
        senales_oof: senales por pliegue.

    Returns:
        Lista de franjas con celdas, rescates y danos.
    """
    acc: dict[str, dict[str, int]] = {
        f"{a:.2f}_{b:.2f}": {"n_celdas": 0, "rescates": 0, "danios": 0}
        for a, b in FRANJAS_DESGLOSE
    }
    for f, (idx_tr, idx_va) in enumerate(pliegues):
        sen = senales_oof[f]
        regla = reglas[f]
        celdas = aplicar_regla(sen, regla["umbral"], regla["k"], regla["tau"])
        bien_v03 = pred_v03[idx_va] == y[idx_va]
        for j in range(len(ETIQUETAS)):
            usar = celdas[:, j] > 0
            if not usar.any():
                continue
            similitudes = sen.sim_max[usar, j]
            bien_ch = bien_v03[:, j].copy()
            bien_ch[usar] = celdas[usar, j] == y[idx_va, j][usar]
            v03 = bien_v03[:, j]
            for a, b in FRANJAS_DESGLOSE:
                mask = (similitudes >= a) & (similitudes < b)
                if not mask.any():
                    continue
                clave = f"{a:.2f}_{b:.2f}"
                acc[clave]["n_celdas"] += int(mask.sum())
                acc[clave]["rescates"] += int(np.count_nonzero(~v03[usar][mask] & bien_ch[usar][mask]))
                acc[clave]["danios"] += int(np.count_nonzero(v03[usar][mask] & ~bien_ch[usar][mask]))
    salida = [
        {"franja": clave, **d, "neto": d["rescates"] - d["danios"]}
        for clave, d in acc.items()
    ]
    return sorted(salida, key=lambda s: -s["n_celdas"])


def tabla_de_mecanismos(metricas: dict[str, Any]) -> list[dict[str, Any]]:
    """Construye la tabla de mecanismos que pide la directiva.

    Args:
        metricas: metricas globales ya calculadas.

    Returns:
        Filas de la tabla.
    """
    return [
        {
            "mecanismo": "V03",
            "cobertura": None,
            "rescates": None,
            "danios": None,
            "neto": None,
            "delta_oof": 0.0,
        },
        {
            "mecanismo": "vecinos_por_similitud_anidado",
            "cobertura": metricas["cobertura_celdas"],
            "rescates": metricas["rescates"],
            "danios": metricas["danios"],
            "neto": metricas["neto"],
            "delta_oof": metricas["delta_macro_f1"],
        },
    ]


def _precision_cambios(y: np.ndarray, pred_base: np.ndarray, pred_nueva: np.ndarray) -> float | None:
    """Acierto entre las celdas que el mecanismo cambia.

    Es la cifra que decide si un mecanismo tiene valor: si entre las celdas que cambia acierta
    mas que la proportion de acierto general, sirve; si acierta menos, solo rompe.

    Args:
        y: etiquetas verdaderas.
        pred_base: prediccion de partida.
        pred_nueva: prediccion tras aplicar el mecanismo.

    Returns:
        La precision entre las celdas modificadas, o None si no se modifico ninguna.
    """
    cambia = pred_base != pred_nueva
    if not cambia.any():
        return None
    return float(np.mean(pred_nueva[cambia] == y[cambia]))


def barrido_exhaustivo(
    pliegues: list[tuple[np.ndarray, np.ndarray]],
    y: np.ndarray,
    pred_v03: np.ndarray,
    senales_cache: dict[tuple[str, int, int], Senales],
) -> list[dict[str, Any]]:
    """Mide rescates y danos de TODA configuracion, sin selector.

    El selector anidado responde "que es lo mejor que puedo hacer con una regla elegida sin
    ver la validacion". Eso no responde a la pregunta de la Fase 5, que es "hay algun vecino
    que correcte un error de V03". Para esa pregunta hay que aplicar la mayoria simple en
    todas las configuraciones y contar, sin elegir nada.

    Se usa ``tau = 0.5``, es decir mayoria simple entre los k vecinos, y se barre el umbral y k.
    Cada fila informa ademas la precision entre las celdas que el mecanismo cambia, que es la
    cifra que decide si el mecanismo tiene algun valor: por encima de la precision de V03 en las
    celas que V03 ya erronea, sirve; por debajo, solo rompe.

    Args:
        pliegues: pliegues canonicos.
        y: etiquetas verdaderas.
        pred_v03: predicciones de V03.
        senales_cache: senales de validacion por (metodo, k, pliegue).

    Returns:
        Filas con cobertura, rescates, danos, net y delta de macro-F1.
    """
    filas: list[dict[str, Any]] = []
    for metodo in METODOS:
        for k in REJILLA_K:
            for umbral in REJILLA_UMBRAL:
                pred_oof = pred_v03.copy()
                for f, (idx_tr, idx_va) in enumerate(pliegues):
                    sen = senales_cache[(metodo, k, f)]
                    celdas = aplicar_regla(sen, umbral, k, TAU_MAYORIA)
                    pred_oof[idx_va] = combinar(pred_v03[idx_va], celdas)
                metricas = evaluar(y, pred_v03, pred_oof)
                filas.append({
                    "metodo": metodo,
                    "k": k,
                    "umbral": umbral,
                    "tau": TAU_MAYORIA,
                    "cobertura": metricas["cobertura_celdas"],
                    "celdas_modificadas": metricas["celdas_modificadas"],
                    "rescates": metricas["rescates"],
                    "danios": metricas["danios"],
                    "neto": metricas["neto"],
                    "delta_macro_f1": metricas["delta_macro_f1"],
                    "acierto_entre_celdas_modificadas": _precision_cambios(y, pred_v03, pred_oof),
                })
    return sorted(filas, key=lambda f: -f["neto"])


def _fmt(valor: Any) -> str:
    """Formatea un valor de la tabla, con guion si falta.

    Args:
        valor: valor a formatear.

    Returns:
        Cadena formateada.
    """
    return "-" if valor is None else f"{valor:.6f}"


def _informe_md(informe: dict[str, Any]) -> None:
    """Escribe el informe legible.

    Args:
        informe: informe con los datos.

    Returns:
        None.
    """
    m = informe["metricas"]
    g = informe["gastos"]
    d = informe["detalle_de_vecinos"]
    lineas = [
        "# V29-03: los vecinos de similitud, pueden corregir errores de V03?",
        "",
        f"Generado por `experiments/v29_03_vecinos_corrigen_errores.py` ({EXPERIMENTO}).",
        "",
        f"**Estado `{informe['estado']}`. Alcance: {informe['alcance']}**",
        "",
        "## 1. Reglas anti-leakage",
        "",
        "| regla | se cumple |",
        "|---|:--:|",
    ]
    for clave, valor in informe["reglas_anti_leakage"].items():
        if isinstance(valor, bool):
            lineas.append(f"| {clave.replace('_', ' ')} | {'si' if valor else 'NO'} |")
    lineas += [
        "",
        f"**Fuga declarada:** {informe['reglas_anti_leakage']['declaracion_de_fuga_no_label']}",
        "",
        (f"Rejilla: {informe['rejilla']['configuraciones_por_pliegue']} configuraciones por pliegue, "
        f"elegidas dentro del training fold por {informe['rejilla']['n_pliegues_internos']} "
        "pliegues internos."),
        "",
        "## 2. Controles",
        "",
        "| control | valor |",
        "|---|---|",
    ]
    for clave, valor in informe["controles"].items():
        if clave != "nota":
            lineas.append(f"| {clave} | {valor} |")
    lineas += [
        "",
        f"**{informe['controles']['nota']}**",
        "",
        "## 3. La regla elegida en cada pliegue",
        "",
        "| pliegue | metodo | k | umbral | tau | macro interna | celdas cambiadas |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for r in informe["regla_por_pliegue"]:
        lineas.append(
            f"| {r['pliegue']} | `{r['metodo']}` | {r['k']} | {r['umbral']:.2f} | "
            f"{r['tau']:.2f} | {r['macro_interna']:.4f} | {r['n_celdas_cambiadas']} |"
        )
    lineas += [
        "",
        "## 4. La tabla de mecanismos",
        "",
        "| mecanismo | cobertura | rescates | danos | net | delta OOF |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for fila in informe["tabla_de_mecanismos"]:
        lineas.append(
            f"| {fila['mecanismo']} | {_fmt(fila['cobertura'])} | {_fmt(fila['rescates'])} | "
            f"{_fmt(fila['danios'])} | {_fmt(fila['neto'])} | {_fmt(fila['delta_oof'])} |"
        )
    lineas += [
        "",
        (f"**corrected_cv V03 {m['macro_f1_v03']:.6f} | con vecino "
        f"{m['macro_f1_challenger']:.6f} | delta {m['delta_macro_f1']:+.6f}**"),
        "",
        "| label | F1 V03 | F1 con vecino | delta | rescates | danos | net | celdas |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for et in ETIQUETAS:
        pl = m["por_label"][et]
        lineas.append(
            f"| {et} | {pl['f1_v03']:.4f} | {pl['f1_challenger']:.4f} | {pl['delta']:+.4f} | "
            f"{pl['rescates']} | {pl['danios']} | {pl['neto']:+d} | {pl['celdas_modificadas']} |"
        )
    lineas += [
        "",
        "## 5. Rescates y danos por franja de similitud del vecino",
        "",
        "| franja | celdas | rescates | danos | net |",
        "|---|---:|---:|---:|---:|",
    ]
    for f in informe["desglose_por_similitud"]:
        lineas.append(
            f"| {f['franja']} | {f['n_celdas']} | {f['rescates']} | {f['danios']} | {f['neto']:+d} |"
        )
    lineas += [
        "",
        "## 6. Barrido exhaustivo: TODA configuracion, sin selector",
        "",
        "El selector de la seccion 4 responde \"que es lo mejor que puedo hacer con una regla",
        "elegida sin ver la validacion\". Esa no es la pregunta de la Fase 5, que es \"hay algun",
        "vecino que corrija un error de V03\". Aqui se aplica la mayoria simple en las",
        f"{len(informe['barrido_exhaustivo_sin_selector'])} configuraciones, sin elegir ninguna,",
        "y se cuentan los rescates y los danos.",
        "",
        "Las diez primeras por net, y las cinco ultimas:",
        "",
        "| metodo | k | umbral | cobertura | rescates | danos | net | delta OOF | acierto en lo que cambia |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ] + [""]
    barrido = informe["barrido_exhaustivo_sin_selector"]
    for b in barrido[:10] + ([{"_sep": True}] if len(barrido) > 15 else []) + barrido[-5:]:
        if b.get("_sep"):
            lineas.append("| ... | | | | | | | | |")
            continue
        acc = b["acierto_entre_celdas_modificadas"]
        lineas.append(
            f"| `{b['metodo'][:20]}` | {b['k']} | {b['umbral']:.2f} | {b['cobertura']:.4f} | "
            f"{b['rescates']} | {b['danios']} | {b['neto']:+d} | {b['delta_macro_f1']:+.6f} | "
            f"{(f'{acc:.4f}') if acc is not None else '-'} |"
        )
    lineas += [
        "",
        (f"**Configuraciones que producen al menos un rescate: "
        f"{informe['configuraciones_con_rescates']} de {len(barrido)}.**"),
        "",
        "## 7. El detalle de la Fase 5 sobre las filas donde V03 falla",
        "",
        f"{d['nota']}.",
        "",
        "| magnitud | valor |",
        "|---|---:|",
        f"| filas con algun error de V03 | {d['n_filas_con_error']} |",
        (f"| de ellas, con vecino por encima del umbral | {d['n_filas_con_error_y_vecino']} "
        f"({d['fraccion_con_error_y_vecino']:.4f}) |"),
        (f"| celdas de error donde el vecino trae la etiqueta correcta | "
        f"{d['n_filas_error_con_vecino_correcto']} |"),
        f"| celdas de error donde el vecino contradice a V03 | {d['n_filas_error_vecino_contradice']} |",
        (f"| celdas correctas donde el vecino contradice a V03 | "
        f"{d['n_filas_correcto_vecino_contradice']} |"),
        "",
        "## 8. Lo que este script no hace",
        "",
        "- No genera submission.",
        "- No entrena ningun modelo: la regla es un umbral de confianza.",
        "- No usa el test.",
        "- No consulta etiquetas de validacion para nada.",
        "",
        (f"Coste: {g['segundos']} s, memoria {g['memoria_pico_mb']:.0f} MB, "
        f"{g['predictores_auxiliares']} predictor auxiliar, {g['submissions']} submissions."),
        "",
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
