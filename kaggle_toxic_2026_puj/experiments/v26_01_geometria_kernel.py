"""V26-01: la geometria del kernel decide si la via no lineal tiene futuro.

La pregunta
-----------
V23 (`svd:rbf_svc`) gano en CV y perdio en el leaderboard. Su mecanismo medido: **1321 de
1731 muestras quedaron como vectores de soporte, 60-76 %**. La explicacion fue que la
geometria RBF sobre estos textos es de **memorizacion**: cada punto tiene un vecino tan
parecido que la maquina no puede generalizar.

Eso importa mas alla de V23, y por eso este script va primero. **Random Fourier Features y
Nystrom son aproximaciones del kernel RBF.** Si el RBF es un kernel de memorizacion, sus
aproximaciones lo son tambien, porque conservan la misma geometria. Es decir: F3 no es un
experimento mas, es el **test predictivo que decide si F1 tiene sentido antes de gastarlo**.

    mediana de similitud del vecino mas cercano > 0.60  ->  memorizacion, y toda la via no
                                                           lineal queda cerrada por teoria
    mediana < 0.30                                        ->  la premisa se cae, y la via
                                                           no lineal se reabre

Que mide, y por que el RBF exacto
---------------------------------
V23 no uso el RBF exacto: uso `TruncatedSVD(300)` y un SVC encima. El SVD de 300 componentes
**altera la geometria**, asi que el 70 % de vectores de soporte que se midio es un
diagnostico del pipeline truncado, no necesariamente del kernel. Este script mide **las tres**
geometrias sobre las mismas filas:

1. el **RBF exacto** sobre los TF-IDF normalizados de V03, que es la referencia;
2. el **RBF sobre SVD(300)**, que es exactamente lo que ejecuto V23;
3. el **kernel spectrum de n-gramas de caracteres**, un kernel de strings que no pasa por
   ningun SVM, para tener un segundo tipo de kernel no lineal.

Si el RBF exacto tiene vecino cercano y el de SVD(300) no, entonces el problema estaba en la
truncacion y no en la geometria, y F1 sobre el RBF exacto seria una via distinta a la de V23.

Controles, antes de interpretar nada
-------------------------------------
1. La identidad `||x-y||^2 = 2 - 2 x.y` se verifica contra el producto denso en un subconjunto
   pequeno, donde cabe.
2. Se verifica que la matriz de similitud tiene diagonal 1 y es simetrica.
3. Se comprueba que el numero de vectores de soporte que salen de un SVC con esa Gram
   geometrica reproduce el dato de V23, o se explica por que no.
4. Se registra la sensibilidad al ancho de banda, porque la mediana del vecino depende de
   `gamma` y un numero sin su `gamma` no significa nada.

Uso
---
    /home/carlos/venv/bin/python experiments/v26_01_geometria_kernel.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train
from src.splits import pliegues_corregidos

EXPERIMENTO = "V26-01 (geometria del kernel: test predictivo de la via no lineal)"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v26_01_geometria_kernel.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "v26_01_geometria_kernel.md"
OOF_CANONICO = PROJECT_ROOT / "features" / "oof_v03_final.npz"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"

PARAMETROS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
PARAMETROS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2, "sublinear_tf": True}
GAMMAS = (0.25, 0.5, 1.0, 2.0, 4.0)
N_COMPONENTES_SVD = 300
SEMILLA = 42
UMBRAL_MEMORIZACION = 0.60
UMBRAL_PREMISA_CAE = 0.30


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V26-01] {mensaje}", flush=True)


def filas_de(matriz: sparse.csr_matrix) -> np.ndarray:
    """Indice de fila de cada elemento no nulo de una CSR.

    Args:
        matriz: matriz dispersa CSR.

    Returns:
        Vector de longitud `nnz` con el indice de fila de cada no nulo.
    """
    return np.repeat(np.arange(matriz.shape[0]), np.diff(matriz.indptr))


def verificar_identidad(matriz: sparse.csr_matrix, n_prueba: int = 60, semilla: int = 0) -> dict[str, Any]:
    """Verifica que `||x-y||^2 = 2 - 2 x.y` sobre vectores normalizados.

    Se hace sobre un subconjunto pequeno, donde el producto denso cabe, para comparar la via
    rapida contra la directa.

    Args:
        matriz: matriz dispersa con filas normalizadas en L2.
        n_prueba: numero de filas del subconjunto.
        semilla: semilla del subconjunto.

    Returns:
        Diccionario con el error de la identidad.
    """
    generador = np.random.default_rng(semilla)
    indices = generador.choice(matriz.shape[0], size=min(n_prueba, matriz.shape[0]), replace=False)
    sub = matriz[indices].toarray()
    producto = sub @ sub.T
    directa = np.sum((sub[:, None, :] - sub[None, :, :]) ** 2, axis=2)
    error = float(np.max(np.abs((2.0 - 2.0 * producto) - directa)))
    return {"n_filas": len(indices), "error_maximo": error, "coincide": bool(error < 1e-9)}


def vecinos(gram: np.ndarray) -> dict[str, Any]:
    """Estadisticas del vecino mas cercano de cada fila, sin incluirse a si misma.

    Args:
        gram: matriz de similitud, cuadrada y simetrica.

    Returns:
        Diccionario con la mediana, los percentiles y las fracciones por umbral.
    """
    copia = gram.copy()
    np.fill_diagonal(copia, -np.inf)
    mejor = copia.max(axis=1)
    return {
        "mediana": float(np.median(mejor)),
        "percentil_10": float(np.percentile(mejor, 10)),
        "percentil_25": float(np.percentile(mejor, 25)),
        "percentil_75": float(np.percentile(mejor, 75)),
        "percentil_90": float(np.percentile(mejor, 90)),
        "maximo": float(mejor.max()),
        "minimo": float(mejor.min()),
        "fraccion_sobre_0_60": float(np.mean(mejor > UMBRAL_MEMORIZACION)),
        "fraccion_sobre_0_80": float(np.mean(mejor > 0.80)),
        "fraccion_sobre_0_90": float(np.mean(mejor > 0.90)),
    }


def gram_rbf_exacto(matriz: sparse.csr_matrix, gamma: float) -> np.ndarray:
    """Gram RBF exacto sobre filas normalizadas en L2, sin densificar la entrada.

    Args:
        matriz: matriz dispersa con filas normalizadas en L2.
        gamma: anchura de banda.

    Returns:
        Matriz de similitud densa de forma (n, n).
    """
    producto = np.asarray((matriz @ matriz.T).todense())
    producto = np.clip(producto, 0.0, 2.0)
    return np.exp(-gamma * (2.0 - 2.0 * producto))


def gram_spectrum(matriz_binaria: sparse.csr_matrix) -> np.ndarray:
    """Kernel spectrum: fraccion de n-gramas de caracteres compartidos.

    Es un kernel de strings explicito, sin SVM y sin paso por SVD, para tener un segundo
    punto de vista no lineal.

    Args:
        matriz_binaria: matriz binaria de n-gramas, forma (n, p).

    Returns:
        Matriz de similitud densa de forma (n, n).
    """
    Binaria = matriz_binaria.astype(np.float64)
    norma = np.sqrt(np.asarray(Binaria.multiply(Binaria).sum(axis=1)).ravel())
    norma[norma == 0] = 1.0
    escala = sparse.diags(1.0 / norma)
    producto = np.asarray((escala @ Binaria @ Binaria.T @ escala).todense())
    return np.clip(producto, 0.0, 1.0)


def main() -> None:
    """Mide la geometria de las tres familias de kernel y emite el veredicto.

    Returns:
        None.
    """
    inicio = time.time()
    log("V26-01: geometria del kernel. Ninguna submission en este script")
    log("   este script es el test predictivo: si el kernel es de memorizacion, sus")
    log("   aproximaciones (RFF, Nystrom) lo son tambien y la via no lineal se cierra")

    log("1/6 datos, representacion exacta de V03 y normalizacion L2")
    train = read_train()
    particion = pd.read_csv(PARTICION)
    ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
    desarrollo = train[train["id"].astype(int).isin(ids_dev)].reset_index(drop=True)
    textos = desarrollo["text"]
    log(f"   desarrollo: {len(desarrollo)} filas")

    palabra = TfidfVectorizer(**PARAMETROS_WORD)
    caracter = TfidfVectorizer(**PARAMETROS_CHAR)
    x_palabra = palabra.fit_transform(textos)
    x_caracter = caracter.fit_transform(textos)
    union = sparse.hstack([x_palabra, x_caracter]).tocsr()
    normada = normalize(union).tocsr()
    log(
        f"   TF-IDF de V03: {union.shape[1]} features | nnz {union.nnz} | "
        f"densidad {union.nnz / (union.shape[0] * union.shape[1]):.6f}"
    )

    log("2/6 control: la identidad ||x-y||^2 = 2 - 2 x.y sobre un subconjunto denso")
    identidad = verificar_identidad(normada)
    log(
        f"   {identidad['n_filas']} filas | error maximo {identidad['error_maximo']:.3e} | "
        f"coincide: {identidad['coincide']}"
    )
    if not identidad["coincide"]:
        raise SystemExit("PARADA OBLIGATORIA: la identidad de distancias no se cumple")

    normada.shape[0]
    log("3/6 control: la matriz de similitud tiene diagonal 1 y es simetrica")
    controles_gram: dict[str, Any] = {}
    for gamma in GAMMAS:
        gram = gram_rbf_exacto(normada, gamma)
        diag = float(np.max(np.abs(np.diag(gram) - 1.0)))
        asimetria = float(np.max(np.abs(gram - gram.T)))
        controles_gram[str(gamma)] = {
            "max_abs_diagonal_menos_1": diag,
            "max_abs_asimetria": asimetria,
            "coincide": bool(diag < 1e-12 and asimetria < 1e-12),
        }
    log(
        f"   diagonal: max |K_ii - 1| = {max(c['max_abs_diagonal_menos_1'] for c in controles_gram.values()):.3e}"
        f" | asimetria: max |K - K^T| = {max(c['max_abs_asimetria'] for c in controles_gram.values()):.3e}"
    )
    if not all(c["coincide"] for c in controles_gram.values()):
        raise SystemExit("PARADA OBLIGATORIA: el Gram no es simetrico ni tiene diagonal 1")

    resultados: dict[str, Any] = {}

    log("4/6 el RBF EXACTO sobre los TF-IDF de V03, con sensibilidad al ancho de banda")
    rbf: dict[str, Any] = {}
    for gamma in GAMMAS:
        t0 = time.time()
        gram = gram_rbf_exacto(normada, gamma)
        d = vecinos(gram)
        d["segundos"] = round(time.time() - t0, 2)
        rbf[str(gamma)] = d
        log(
            f"   gamma={gamma:<5} mediana {d['mediana']:.4f} | p90 {d['percentil_90']:.4f} | "
            f"max {d['maximo']:.4f} | >0.60: {d['fraccion_sobre_0_60'] * 100:5.1f} % | "
            f">0.90: {d['fraccion_sobre_0_90'] * 100:5.1f} % | {d['segundos']} s"
        )
    resultados["rbf_exacto"] = rbf
    mejor_gamma = max(GAMMAS, key=lambda g: rbf[str(g)]["mediana"])
    mediana_rbf = rbf[str(mejor_gamma)]["mediana"]
    log(f"   gamma con mayor mediana de similitud del vecino: {mejor_gamma} ({mediana_rbf:.4f})")

    log("5/6 el RBF sobre SVD(300), que es EXACTAMENTE el pipeline de V23")
    svd = TruncatedSVD(n_components=N_COMPONENTES_SVD, random_state=SEMILLA)
    reducido = svd.fit_transform(normada)
    varianza = float(svd.explained_variance_ratio_.sum())
    log(
        f"   SVD({N_COMPONENTES_SVD}): varianza explicada {varianza:.4f} | "
        f"primera componente {svd.explained_variance_ratio_[0]:.4f} | forma {reducido.shape}"
    )
    svd_d: dict[str, Any] = {}
    # El SVD NO deja las filas normalizadas, asi que la distancia al cuadrado es
    # ||x||^2 + ||y||^2 - 2 x.y y hay que dividir por una escala que la haga no negativa. La
    # primera version de este script usaba la media de ||x||^2, que puede ser menor que
    # ||x||^2 + ||y||^2, y eso producia exponentes positivos y similitudes mayores que 1: un
    # kernel RBF no puede dar eso. La escala correcta es la distancia maxima observada.
    for gamma in (0.25, 0.5, 1.0):
        norma = np.sum(reducido**2, axis=1)
        distancia = norma[:, None] + norma[None, :] - 2.0 * (reducido @ reducido.T)
        escala = float(np.max(distancia)) / 2.0
        np.fill_diagonal(distancia, 2.0 * escala)
        gram = np.exp(-2.0 * gamma * distancia / (2.0 * escala))
        np.fill_diagonal(gram, 1.0)
        if gram.max() > 1.0 + 1e-9:
            raise SystemExit(
                f"PARADA OBLIGATORIA: el Gram del SVD da {gram.max():.6f}, que no puede pasar de 1"
            )
        d = vecinos(gram)
        d["gamma"] = gamma
        d["escala_de_distancia"] = escala
        d["maximo_del_gram"] = float(gram.max())
        svd_d[str(gamma)] = d
        log(
            f"   gamma={gamma} (escala {escala:.4f}, max del Gram {gram.max():.6f}) | "
            f"mediana {d['mediana']:.4f} | p90 {d['percentil_90']:.4f} | "
            f">0.60: {d['fraccion_sobre_0_60'] * 100:5.1f} % | "
            f">0.90: {d['fraccion_sobre_0_90'] * 100:5.1f} %"
        )
    resultados["rbf_sobre_svd_300"] = {
        "varianza_explicada": varianza,
        "primera_componente": float(svd.explained_variance_ratio_[0]),
        "por_gamma": svd_d,
    }
    gamma_svd = min(svd_d, key=lambda g: abs(float(g) - 0.5))
    mediana_svd = svd_d[gamma_svd]["mediana"]

    log("6/6 el kernel spectrum de n-gramas de caracteres, un kernel de strings sin SVM")
    binaria = (union > 0).astype(np.int8).tocsr()
    t0 = time.time()
    gram_spec = gram_spectrum(binaria)
    d_spec = vecinos(gram_spec)
    d_spec["segundos"] = round(time.time() - t0, 2)
    resultados["spectrum_caracteres"] = d_spec
    log(
        f"   mediana {d_spec['mediana']:.4f} | p90 {d_spec['percentil_90']:.4f} | "
        f"max {d_spec['maximo']:.4f} | >0.60: {d_spec['fraccion_sobre_0_60'] * 100:5.1f} % | "
        f">0.90: {d_spec['fraccion_sobre_0_90'] * 100:5.1f} % | {d_spec['segundos']} s"
    )

    log("   control cruzado: support vectors de un SVC con el Gram RBF exacto")
    idx_tr, _idx_va = pliegues_corregidos(
        desarrollo, np.array([0] * len(desarrollo))
    )[0]
    y = desarrollo[["HS", "TR", "AG"]].to_numpy().astype(int)
    filas_tr = np.asarray(idx_tr)
    gram_tr = gram_rbf_exacto(normada[filas_tr], mejor_gamma)
    svc = SVC(C=2.0, kernel="precomputed", random_state=SEMILLA)
    svc.fit(gram_tr, y[filas_tr, 0])
    prop_svd = float(len(svc.support_) / len(filas_tr))
    log(
        f"   pliego 1, HS, RBF exacto con gamma={mejor_gamma}: "
        f"{len(svc.support_)} vectores de soporte sobre {len(filas_tr)} = {prop_svd * 100:.1f} %"
    )
    log("   dato de V23 con SVD(300): 60-76 % de vectores de soporte")

    int(svd_d["1.0"]["fraccion_sobre_0_60"] > UMBRAL_MEMORIZACION)
    if mediana_rbf >= UMBRAL_MEMORIZACION:
        veredicto = "MEMORIZACION_CONFIRMADA"
        lectura = (
            "El RBF exacto tiene vecino mas cercano con similitud mediana "
            f"{mediana_rbf:.4f}, por encima de {UMBRAL_MEMORIZACION}. La geometria es de "
            "memorizacion y, como RFF y Nystrom preservan esa misma geometria, sus "
            "aproximaciones lo son tambien. La via no lineal queda cerrada por teoria."
        )
    elif mediana_rbf <= UMBRAL_PREMISA_CAE:
        veredicto = "PREMISA_CAE"
        lectura = (
            f"El RBF exacto tiene vecino mas cercano con similitud mediana {mediana_rbf:.4f}, "
            f"por debajo de {UMBRAL_PREMISA_CAE}. La premisa de memorizacion se cae y la via "
            "no lineal queda reabierta."
        )
    else:
        veredicto = "ZONA_GRIS"
        lectura = (
            f"Mediana {mediana_rbf:.4f}, entre los dos umbrales. Habria que mirar el ancho de "
            "banda y el espectro de similitudes, no un solo numero."
        )
    log(f"VEREDICTO: {veredicto}")
    log(f"   {lectura}")
    log(f"   comparacion: RBF exacto {mediana_rbf:.4f} | RBF sobre SVD(300) {mediana_svd:.4f} | spectrum {d_spec['mediana']:.4f}")
    log(f"   tiempo total: {round(time.time() - inicio, 1)} s | submissions 0")

    informe = {
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pregunta": (
            "la mediana de similitud del vecino mas cercano con el kernel RBF exacto supera "
            "0.60, de modo que la via no lineal esta cerrada por teoria, o baja de 0.30 y la "
            "via se reabre"
        ),
        "controles": {
            "identidad_de_distancias": identidad,
            "gram_simetrico_y_diagonal_1": controles_gram,
            "support_vectors_rbf_exacto": {
                "gamma": mejor_gamma,
                "n_support": len(svc.support_),
                "n_entrenamiento": len(filas_tr),
                "proporcion": prop_svd,
            },
            "dato_de_v23_para_comparar": "60-76 % de vectores de soporte con SVD(300)",
            "gamma_svd": gamma_svd,
        },
        "resultados": resultados,
        "veredicto": veredicto,
        "umbral_memORIZacion": UMBRAL_MEMORIZACION,
        "umbral_premisa_cae": UMBRAL_PREMISA_CAE,
        "medianas": {
            "rbf_exacto": mediana_rbf,
            "rbf_sobre_svd_300": mediana_svd,
            "spectrum_caracteres": d_spec["mediana"],
        },
        "gastos": {"submissions": 0, "segundos": round(time.time() - inicio, 1)},
    }
    REPORTE_JSON.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _informe_md(informe)
    log("   informe escrito en reports/v26_01_geometria_kernel.{json,md}")


def _informe_md(informe: dict[str, Any]) -> None:
    """Escribe el informe legible.

    Args:
        informe: informe con los datos.

    Returns:
        None.
    """
    c = informe["controles"]
    m = informe["medianas"]
    sv = c["support_vectors_rbf_exacto"]
    lineas = [
        "# V26-01: la geometria del kernel, test predictivo de la via no lineal",
        "",
        f"Generado por `experiments/v26_01_geometria_kernel.py` ({EXPERIMENTO}).",
        "",
        "**Ninguna submission.**",
        "",
        "## 1. La pregunta",
        "",
        "```",
        informe["pregunta"],
        "```",
        "",
        "RFF y Nystrom **aproximan el kernel RBF**. Si el RBF es de memorizacion, sus",
        "aproximaciones lo son tambien, porque conservan la misma geometria. Por eso este",
        "script va antes que ellos: es el test que decide si la via tiene sentido.",
        "",
        "## 2. Controles",
        "",
        "| control | resultado |",
        "|---|---|",
        (f"| identidad `||x-y||^2 = 2 - 2 x.y` | error maximo "
        f"{c['identidad_de_distancias']['error_maximo']:.3e} |"),
        (f"| Gram simetrico con diagonal 1 | asimetria maxima "
        f"{max(v['max_abs_asimetria'] for v in c['gram_simetrico_y_diagonal_1'].values()):.3e} |"),
        (f"| support vectors con el RBF exacto | {sv['n_support']} de {sv['n_entrenamiento']}, "
        f"o sea {sv['proporcion'] * 100:.1f} % |"),
        f"| dato de V23 con SVD(300) | {c['dato_de_v23_para_comparar']} |",
        "",
        "## 3. La mediana de similitud del vecino mas cercano",
        "",
        "| kernel | mediana | p10 | p90 | maximo | > 0.60 | > 0.90 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for etiqueta, clave in (
        ("RBF exacto, mejor gamma", "rbf_exacto"),
        ("RBF sobre SVD(300)", "rbf_sobre_svd_300"),
        ("spectrum de caracteres", "spectrum_caracteres"),
    ):
        if clave == "rbf_exacto":
            d = informe["resultados"][clave][str(sv["gamma"])]
        elif clave == "rbf_sobre_svd_300":
            d = informe["resultados"][clave]["por_gamma"][str(c["gamma_svd"])]
        else:
            d = informe["resultados"][clave]
        lineas.append(
            f"| {etiqueta} | {d['mediana']:.4f} | {d['percentil_10']:.4f} | "
            f"{d['percentil_90']:.4f} | {d['maximo']:.4f} | "
            f"{d['fraccion_sobre_0_60'] * 100:.1f} % | {d['fraccion_sobre_0_90'] * 100:.1f} % |"
        )
    lineas += [
        "",
        "## 4. Sensibilidad al ancho de banda del RBF exacto",
        "",
        "Una mediana sin su `gamma` no significa nada, asi que se recorre.",
        "",
        "| gamma | mediana | p90 | > 0.60 | > 0.90 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for gamma, d in informe["resultados"]["rbf_exacto"].items():
        lineas.append(
            f"| {gamma} | {d['mediana']:.4f} | {d['percentil_90']:.4f} | "
            f"{d['fraccion_sobre_0_60'] * 100:.1f} % | {d['fraccion_sobre_0_90'] * 100:.1f} % |"
        )
    lineas += [
        "",
        (f"Varianza explicada por el SVD(300): "
        f"{informe['resultados']['rbf_sobre_svd_300']['varianza_explicada']:.4f}."),
        "",
        "## 5. Veredicto",
        "",
        f"**{informe['veredicto']}**",
        "",
        (f"Medianas: RBF exacto {m['rbf_exacto']:.4f}, RBF sobre SVD(300) "
        f"{m['rbf_sobre_svd_300']:.4f}, spectrum {m['spectrum_caracteres']:.4f}."),
        "",
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
