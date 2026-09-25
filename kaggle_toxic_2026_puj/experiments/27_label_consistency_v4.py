"""TRACK A v4: consistencia de etiquetas en textos identicos o casi identicos.

Contexto
--------
La fase v2 documento que existe ambiguedad de anotacion, pero nunca midio
**consistencia de etiquetas**: no se calculo, para los textos identicos o casi
identicos dentro de ``train``, quantas combinaciones de etiquetas distintas
conviven en el mismo cluster. Sin esa cifra no se puede responder a la pregunta
central de la spec v4 (§106): cuanto del error es ruido de etiqueta y cuanto es
informacion ausente.

Este script construye clusters de consistencia con cuatro criterios
independientes y calcula, para cada cluster, la pureza, la entropia, el numero
de combinaciones distintas y el desacuerdo, tanto global como por etiqueta
(HS, TR, AG). No entrena modelos, no lee etiquetas de test y no modifica nada
del baseline.

Criterios de agrupamiento
-------------------------
1. ``exacto``: texto identico byte a byte.
2. ``normalizado``: minusculas, menciones y URLs sustituidas por marcadores,
   espacios colapsados.
3. ``jaccard``: pares de tokens con Jaccard >= 0.8 (union-find).
4. ``cosimilitud``: cosimilitud TF-IDF (word 1-1 y char_wb 3-5) >= 0.85
   (union-find sobre los vecinos por encima del umbral).

Ademas persiste el vecindario mas cercano de cada tweet (top-3 por cosimilitud
combinada) porque lo reutiliza el mapa de dureza (`28_hardness_v4.py`).

Uso
---
    /home/carlos/venv/bin/python experiments/27_label_consistency_v4.py

Salidas
-------
- ``reports/label_consistency_v4.csv``: una fila por cluster con tamano >= 2.
- ``reports/label_consistency_v4_ejemplos.md``: clusters contradictorios con su
  texto, para revision manual de la ambiguedad.
- ``reports/label_consistency_v4.json``: resumen cuantitativo.
- ``features/label_consistency_v4.npz``: asignacion de cluster, banderas por
  etiqueta y vecindario mas cercano (reutilizado por el mapa de dureza).
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_test, read_train  # noqa: E402
from src.splits import ETIQUETAS  # noqa: E402

UMBRAL_JACCARD = 0.8
UMBRAL_COSIMILITUD = 0.85
N_VECINOS = 3
CSV_PATH = PROJECT_ROOT / "reports" / "label_consistency_v4.csv"
JSON_PATH = PROJECT_ROOT / "reports" / "label_consistency_v4.json"
EJEMPLOS_PATH = PROJECT_ROOT / "reports" / "label_consistency_v4_ejemplos.md"
NPZ_PATH = PROJECT_ROOT / "features" / "label_consistency_v4.npz"

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
ESPACIOS_PATTERN = re.compile(r"\s+")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def log(mensaje: str) -> None:
    """Imprime una linea de progreso con marca de seccion.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V4-A01] {mensaje}", flush=True)


# ---------------------------------------------------------------------------
# Utilidades de texto y de agrupamiento
# ---------------------------------------------------------------------------
def normalizar(texto: str) -> str:
    """Normaliza el texto para comparar versiones equivalentes.

    Sustituye URLs y menciones por marcadores estables, pasa a minusculas y
    colapsa espacios. No inventa tildes ni modifica el texto original.

    Args:
        texto: texto original del tweet.

    Returns:
        Texto normalizado.
    """
    limpio = URL_PATTERN.sub(" <url> ", texto.lower())
    limpio = MENTION_PATTERN.sub(" <men> ", limpio)
    return ESPACIOS_PATTERN.sub(" ", limpio).strip()


def tokenizar(texto: str) -> frozenset[str]:
    """Extrae tokens alfanumericos minusculos de un texto normalizado.

    Args:
        texto: texto ya normalizado con :func:`normalizar`.

    Returns:
        Conjunto de tokens.
    """
    return frozenset(TOKEN_PATTERN.findall(texto))


class UnionFind:
    """Estructura union-find para construir clusters transitivos.

    Attributes:
        padre: lista con el representante de cada elemento.
    """

    def __init__(self, n: int) -> None:
        """Inicializa cada elemento como su propia raiz.

        Args:
            n: numero de elementos.
        """
        self.padre = list(range(n))

    def encontrar(self, nodo: int) -> int:
        """Devuelve la raiz del nodo comprimiendo caminos.

        Args:
            nodo: indice del elemento.

        Returns:
            Indice de la raiz.
        """
        while self.padre[nodo] != nodo:
            self.padre[nodo] = self.padre[self.padre[nodo]]
            nodo = self.padre[nodo]
        return nodo

    def unir(self, a: int, b: int) -> None:
        """Une dos elementos en la misma componente.

        Args:
            a: primer indice.
            b: segundo indice.
        """
        raiz_a, raiz_b = self.encontrar(a), self.encontrar(b)
        if raiz_a != raiz_b:
            self.padre[max(raiz_a, raiz_b)] = min(raiz_a, raiz_b)

    def raices(self) -> dict[int, list[int]]:
        """Agrupa los indices por raiz.

        Returns:
            Diccionario ``raiz -> lista de indices``.
        """
        grupos: dict[int, list[int]] = {}
        for indice in range(len(self.padre)):
            grupos.setdefault(self.encontrar(indice), []).append(indice)
        return grupos


def clusters_por_llave(llaves: pd.Series) -> dict[Any, list[int]]:
    """Agrupa indices posicionales por un valor de llave.

    Args:
        llaves: serie con una llave por fila.

    Returns:
        Diccionario ``llave -> lista de indices``.
    """
    grupos: dict[Any, list[int]] = {}
    for indice, llave in enumerate(llaves):
        grupos.setdefault(llave, []).append(indice)
    return grupos


def clusters_jaccard(tokens: list[frozenset[str]], umbral: float) -> dict[int, list[int]]:
    """Construye clusters transitivos por similitud Jaccard de tokens.

    Compara todos los pares (n=1.731 son 1,5 millones de comparaciones) y une
    los que superan el umbral.

    Args:
        tokens: lista de conjuntos de tokens por fila.
        umbral: Jaccard minimo para unir dos filas.

    Returns:
        Diccionario ``raiz -> lista de indices``.
    """
    estructura = UnionFind(len(tokens))
    for i in range(len(tokens)):
        for j in range(i + 1, len(tokens)):
            conjunto_i, conjunto_j = tokens[i], tokens[j]
            if not conjunto_i or not conjunto_j:
                continue
            interseccion = len(conjunto_i & conjunto_j)
            if interseccion == 0:
                continue
            if interseccion / len(conjunto_i | conjunto_j) >= umbral:
                estructura.unir(i, j)
    return estructura.raices()


def matriz_cosimilitud(textos: list[str]) -> tuple[sparse.csr_matrix, Any]:
    """Calcula la cosimilitud TF-IDF word + char entre todos los pares.

    Args:
        textos: textos normalizados.

    Returns:
        Tupla con la matriz de cosimilitud densa y los dos vectorizadores
        ajustados (solo se usan dentro de train).
    """
    vectorizador_word = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), min_df=1)
    vectorizador_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)
    matriz_word = vectorizador_word.fit_transform(textos)
    matriz_char = vectorizador_char.fit_transform(textos)
    # La concatenacion ponderada evita que un canal domine al otro en la
    # similitud; los pesos son simetricos y no dependen de las etiquetas.
    matriz = sparse.hstack([matriz_word, 0.5 * matriz_char], format="csr")
    similitud = (matriz @ matriz.T).toarray()
    np.fill_diagonal(similitud, 0.0)
    return similitud, (vectorizador_word, vectorizador_char)


def clusters_cosimilitud(similitud: np.ndarray, umbral: float) -> dict[int, list[int]]:
    """Construye clusters transitivos por cosimilitud por encima del umbral.

    Args:
        similitud: matriz de cosimilitud cuadrada.
        umbral: cosimilitud minima para unir dos filas.

    Returns:
        Diccionario ``raiz -> lista de indices``.
    """
    estructura = UnionFind(similitud.shape[0])
    vecinos = np.argwhere(similitud >= umbral)
    for i, j in vecinos:
        if i < j:
            estructura.unir(int(i), int(j))
    return estructura.raices()


# ---------------------------------------------------------------------------
# Metricas de consistencia
# ---------------------------------------------------------------------------
def metricas_cluster(indices: list[int], etiquetas: np.ndarray) -> dict[str, Any]:
    """Calcula pureza, entropia y desacuerdo de un cluster de etiquetas.

    Args:
        indices: indices posicionales de los miembros del cluster.
        etiquetas: matriz binaria de etiquetas alineada con las filas de train.

    Returns:
        Diccionario con el tamano, las combinaciones distintas, la pureza, la
        entropia (en bits) y el desacuerdo, mas el detalle por etiqueta.
    """
    subconjunto = etiquetas[indices]
    combinaciones = ["".join(str(int(v)) for v in fila) for fila in subconjunto]
    conteo = Counter(combinaciones)
    n = len(indices)
    pureza = max(conteo.values()) / n
    entropia = -sum(
        (conteo[clave] / n) * np.log2(conteo[clave] / n) for clave in conteo
    )
    detalle = {}
    for posicion, etiqueta in enumerate(ETIQUETAS):
        columna = subconjunto[:, posicion]
        positivos = int(columna.sum())
        negativos = n - positivos
        if positivos and negativos:
            desajuste = min(positivos, negativos) / n
        else:
            desajuste = 0.0
        detalle[etiqueta] = {
            "positivos": positivos,
            "negativos": negativos,
            "desacuerdo": round(float(desajuste), 4),
        }
    return {
        "n": n,
        "combinaciones": len(conteo),
        "combinaciones_distintas": ";".join(sorted(conteo)),
        "pureza": round(float(pureza), 4),
        "entropia_bits": round(float(entropia), 4),
        "desacuerdo": round(float(1.0 - pureza), 4),
        "desacuerdo_por_etiqueta": detalle,
    }


def main() -> None:
    """Ejecuta el analisis de consistencia de etiquetas del TRACK A."""
    inicio = datetime.now(timezone.utc)
    log("Consistencia de etiquetas (TRACK A): clusters exacto, normalizado, Jaccard y cosimilitud")

    train = read_train()
    etiquetas = train[["HS", "TR", "AG"]].to_numpy(dtype=int)
    textos = train["text"].astype(str).tolist()
    normalizados = [normalizar(texto) for texto in textos]
    tokens = [tokenizar(texto) for texto in normalizados]
    log(f"   train: {len(train)} filas")

    # 1. Criterio exacto y 2. criterio normalizado.
    por_exacto = clusters_por_llave(pd.Series(textos))
    por_normalizado = clusters_por_llave(pd.Series(normalizados))
    log(f"   criterio exacto      : {sum(1 for v in por_exacto.values() if len(v) > 1)} clusters con 2+ miembros")
    log(f"   criterio normalizado : {sum(1 for v in por_normalizado.values() if len(v) > 1)} clusters con 2+ miembros")

    # 3. Criterio Jaccard de tokens.
    por_jaccard = clusters_jaccard(tokens, UMBRAL_JACCARD)
    log(f"   criterio jaccard>={UMBRAL_JACCARD}: {sum(1 for v in por_jaccard.values() if len(v) > 1)} clusters con 2+ miembros")

    # 4. Criterio de cosimilitud TF-IDF y vecindario mas cercano.
    similitud, _ = matriz_cosimilitud(normalizados)
    por_cosimilitud = clusters_cosimilitud(similitud, UMBRAL_COSIMILITUD)
    log(f"   criterio cosim>={UMBRAL_COSIMILITUD}: {sum(1 for v in por_cosimilitud.values() if len(v) > 1)} clusters con 2+ miembros")
    log(f"   pares con cosim>={UMBRAL_COSIMILITUD}: {int((similitud >= UMBRAL_COSIMILITUD).sum() // 2)}")

    # Vecindario mas cercano (se persiste para el mapa de dureza).
    orden = np.argsort(-similitud, axis=1)
    vecino_ids = orden[:, :N_VECINOS]
    vecino_sim = np.take_along_axis(similitud, orden[:, :N_VECINOS], axis=1)

    # Cluster combinado: transicion de cualquiera de los cuatro criterios.
    combinado = UnionFind(len(train))
    for grupos in (por_exacto, por_normalizado, por_jaccard, por_cosimilitud):
        for miembros in grupos.values():
            for otro in miembros[1:]:
                combinado.unir(miembros[0], otro)
    por_combinado = combinado.raices()

    # Fila por cluster con dos o mas miembros.
    filas: list[dict[str, Any]] = []
    asignacion = np.full(len(train), -1, dtype=int)
    discrepancia_etiqueta = np.zeros((len(train), len(ETIQUETAS)), dtype=int)
    for cluster_id, (raiz, miembros) in enumerate(sorted(por_combinado.items())):
        if len(miembros) < 2:
            continue
        metricas = metricas_cluster(miembros, etiquetas)
        for indice in miembros:
            asignacion[indice] = cluster_id
            for posicion, etiqueta in enumerate(ETIQUETAS):
                if metricas["desacuerdo_por_etiqueta"][etiqueta]["desacuerdo"] > 0:
                    discrepancia_etiqueta[indice, posicion] = 1
        filas.append(
            {
                "cluster_id": cluster_id,
                "criterio_dominante": _criterio_dominante(
                    miembros, por_exacto, por_normalizado, por_jaccard, por_cosimilitud
                ),
                "n_miembros": metricas["n"],
                "combinaciones_distintas": metricas["combinaciones_distintas"],
                "pureza": metricas["pureza"],
                "entropia_bits": metricas["entropia_bits"],
                "desacuerdo": metricas["desacuerdo"],
                "desacuerdo_HS": metricas["desacuerdo_por_etiqueta"]["HS"]["desacuerdo"],
                "desacuerdo_TR": metricas["desacuerdo_por_etiqueta"]["TR"]["desacuerdo"],
                "desacuerdo_AG": metricas["desacuerdo_por_etiqueta"]["AG"]["desacuerdo"],
                "ids": ";".join(str(int(train["id"].iloc[indice])) for indice in miembros),
                "textos": " || ".join(_recortar(textos[indice]) for indice in miembros),
            }
        )

    tabla = pd.DataFrame(filas).sort_values(
        ["desacuerdo", "n_miembros"], ascending=[False, False]
    )
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(CSV_PATH, index=False)

    # Resumen cuantitativo.
    en_cluster = asignacion >= 0
    contradictorios = tabla[tabla["desacuerdo"] > 0]
    por_etiqueta = {}
    for posicion, etiqueta in enumerate(ETIQUETAS):
        en_contradicto = discrepancia_etiqueta[:, posicion].astype(bool)
        por_etiqueta[etiqueta] = {
            "n_tweets_en_cluster_contradictorio": int(en_contradicto.sum()),
            "porcentaje_del_train": round(
                100.0 * float(en_contradicto.sum()) / len(train), 3
            ),
            "n_positivos": int(etiquetas[en_contradicto, posicion].sum()),
            "n_negativos": int((1 - etiquetas[en_contradicto, posicion]).sum()),
        }

    test = read_test()
    normalizados_test = {normalizar(texto) for texto in test["text"].astype(str)}
    coincidencias_test = sum(1 for texto in normalizados if texto in normalizados_test)

    resumen = {
        "experiment_id": "V4-A01",
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duracion_s": round((datetime.now(timezone.utc) - inicio).total_seconds(), 2),
        "n_train": int(len(train)),
        "umbrales": {
            "jaccard": UMBRAL_JACCARD,
            "cosimilitud": UMBRAL_COSIMILITUD,
        },
        "criterios": {
            "exacto": {
                "clusters_2plus": int(sum(1 for v in por_exacto.values() if len(v) > 1))
            },
            "normalizado": {
                "clusters_2plus": int(sum(1 for v in por_normalizado.values() if len(v) > 1))
            },
            "jaccard": {
                "clusters_2plus": int(sum(1 for v in por_jaccard.values() if len(v) > 1))
            },
            "cosimilitud": {
                "clusters_2plus": int(
                    sum(1 for v in por_cosimilitud.values() if len(v) > 1)
                )
            },
            "combinado": {"clusters_2plus": int(len(tabla))},
        },
        "tweets_en_un_cluster": int(en_cluster.sum()),
        "porcentaje_en_un_cluster": round(100.0 * float(en_cluster.sum()) / len(train), 3),
        "clusters_contradictorios": int(len(contradictorios)),
        "tweets_en_cluster_contradictorio": int(
            asignacion[discrepancia_etiqueta.max(axis=1) == 1].size
        ),
        "desacuerdo_global_por_etiqueta": por_etiqueta,
        "tweets_de_test_con_texto_normalizado_igual_a_algun_tweet_de_train": int(
            coincidencias_test
        ),
        "nota_test": (
            "El test solo se usa para CONTAR coincidencias de texto; no se leen "
            "sus etiquetas (no existen) ni se usa para ajustar nada."
        ),
    }
    JSON_PATH.write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    NPZ_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        NPZ_PATH,
        ids=train["id"].to_numpy(dtype=np.int64),
        cluster_id=asignacion,
        discrepancia_etiqueta=discrepancia_etiqueta,
        en_cluster=en_cluster.astype(np.int8),
        vecino_ids=vecino_ids.astype(np.int64),
        vecino_sim=vecino_sim.astype(np.float32),
        y_true=etiquetas,
    )

    _escribir_ejemplos(tabla, train)
    _imprimir_resumen(resumen, tabla)


def _criterio_dominante(
    miembros: list[int],
    por_exacto: dict[Any, list[int]],
    por_normalizado: dict[Any, list[int]],
    por_jaccard: dict[int, list[int]],
    por_cosimilitud: dict[int, list[int]],
) -> str:
    """Determina que criterio de agrupamiento produjo un cluster.

    Args:
        miembros: indices del cluster combinado.
        por_exacto: clusters por texto exacto.
        por_normalizado: clusters por texto normalizado.
        por_jaccard: clusters por Jaccard.
        por_cosimilitud: clusters por cosimilitud.

    Returns:
        Nombre del criterio mas restrictivo que cubre el cluster.
    """
    conjunto = set(miembros)
    if any(
        len(grupo) > 1 and conjunto <= set(grupo) for grupo in por_exacto.values()
    ):
        return "exacto"
    if any(
        len(grupo) > 1 and conjunto <= set(grupo) for grupo in por_normalizado.values()
    ):
        return "normalizado"
    if any(
        len(grupo) > 1 and conjunto <= set(grupo) for grupo in por_jaccard.values()
    ):
        return "jaccard"
    if any(
        len(grupo) > 1 and conjunto <= set(grupo) for grupo in por_cosimilitud.values()
    ):
        return "cosimilitud"
    return "mixto"


def _recortar(texto: str, limite: int = 110) -> str:
    """Recorta un texto para los informes legible.

    Args:
        texto: texto original.
        limite: longitud maxima.

    Returns:
        Texto recortado con puntos suspensivos si excede el limite.
    """
    limpio = ESPACIOS_PATTERN.sub(" ", texto).strip()
    return limpio if len(limpio) <= limite else limpio[: limite - 3] + "..."


def _escribir_ejemplos(tabla: pd.DataFrame, train: pd.DataFrame) -> None:
    """Escribe el informe de clusters contradictorios para revision manual.

    Args:
        tabla: tabla de clusters con dos o mas miembros.
        train: marco de entrenamiento original.
    """
    contradicciones = tabla[tabla["desacuerdo"] > 0].head(25)
    lineas = [
        "# Clusters contradictorios (revision manual de ambiguedad)",
        "",
        f"Generado por `experiments/27_label_consistency_v4.py` el "
        f"{datetime.now(timezone.utc).date().isoformat()}. Se listan hasta 25 "
        "clusters ordenados por desacuerdo y tamano; el objetivo es que la "
        "ambiguedad de anotacion sea auditable, no automatizada (§8).",
        "",
        f"Total de clusters con 2+ miembros: **{len(tabla)}**; "
        f"clusters con etiquetas contradictorias: **{int((tabla['desacuerdo'] > 0).sum())}**.",
        "",
    ]
    for _, fila in contradicciones.iterrows():
        lineas.append(
            f"## cluster {int(fila['cluster_id'])} | n={int(fila['n_miembros'])} | "
            f"pureza={fila['pureza']} | entropia={fila['entropia_bits']} bits | "
            f"desacuerdo={fila['desacuerdo']} | criterio={fila['criterio_dominante']}"
        )
        lineas.append("")
        for parte in str(fila["textos"]).split(" || "):
            lineas.append(f"- {parte}")
        lineas.append("")
    EJEMPLOS_PATH.write_text("\n".join(lineas) + "\n", encoding="utf-8")


def _imprimir_resumen(resumen: dict[str, Any], tabla: pd.DataFrame) -> None:
    """Imprime el resumen del analisis por consola.

    Args:
        resumen: diccionario de resumen.
        tabla: tabla de clusters.
    """
    log("   RESULTADO")
    log(f"   tweets en un cluster 2+      : {resumen['tweets_en_un_cluster']} ({resumen['porcentaje_en_un_cluster']} %)")
    log(f"   clusters contradictorios     : {resumen['clusters_contradictorios']}")
    for etiqueta, valores in resumen["desacuerdo_global_por_etiqueta"].items():
        log(
            f"   {etiqueta}: tweets en cluster contradictorio "
            f"{valores['n_tweets_en_cluster_contradictorio']} "
            f"({valores['porcentaje_del_train']} %)"
        )
    log(f"   test con texto igual a train: {resumen['tweets_de_test_con_texto_normalizado_igual_a_algun_tweet_de_train']}")
    log(f"   artefactos: {CSV_PATH.name}, {EJEMPLOS_PATH.name}, {JSON_PATH.name}, {NPZ_PATH.name}")


if __name__ == "__main__":
    main()
