"""V29-02: similitudes mas laxas, cobertura, pureza y relacion similitud -> misma etiqueta.

Que responde
------------
Fases 2, 3 y 4 de la directiva V29. La auditoria V29-01 demostro dos cosas: que el techo al que
se puede aspirar relajando **similitud** es 0,40 %, no 7,08 %, y que la **normalizacion** pesa
mas que el umbral (quitar menciones, hashtags y puntuacion multiplica por 7,5 la cobertura a
0,85). Este script mide, sobre esas alternativas, las tres cosas que la directiva pide:

- **Fase 3**: numero de pares, grupos, cobertura, tamano medio, mediana, p95, y por label el
  porcentaje de grupos con una sola etiqueta, con varias, y contradictorios, mas la pureza y
  la entropia.
- **Fase 4**: la relacion empirica ``similitud -> probabilidad de compartir etiqueta``, por
  label y por franja de similitud, **sin asumir ningun umbral**.

Y la pregunta de fondo, que es la de la directiva:

> Al relajar la similitud aparecen falsos vecinos (A), grupos grandes pero puros (B), o
> plantillas reales con informacion de etiqueta (C)?

Los cinco metodos
-----------------
Todos se calculan **directamente sobre el texto**, sin embeddings, sin representacion
aprendida y sin informacion externa. Se ordenan de mas a menos estricto:

| clave | metodo | normalizacion | analisis | proposito |
|---|---|---|---|---|
| A | coseno TF-IDF char_wb 4-4 | historica | char 4 | es el metodo de V03, sirve de referencia |
| B | coseno TF-IDF char_wb 4-4 | agresiva | char 4 | separa el efecto de la normalizacion |
| C | coseno TF-IDF char_wb 3-5 | agresiva | char 3-5 | mas contexto, mismo criterio |
| D | Jaccard de char 3-5 | agresiva | char 3-5 | conjunto, no producto: robusto a la longitud |
| E | Jaccard de tokens unigrama | agresiva | palabra | ignores la ortografia, compara vocabulario |

Por que no hay MinHash ni LSH
-----------------------------
Con 1731 filas de train hay 1 497 315 pares, y la matriz densa de similitud ocupa 12 MB. Un
indice aproximado no aportaria nada y anadiria una fuente de error. Carlos pidio usarlo "si
ayuda a escalar", y aqui no ayuda: se declara explicitamente.

Por que no hay similitud de edicion
------------------------------------
La directiva la admite **solo para candidatos prefiltrados**. Este script es puramente
descriptivo y no necesita ella, asi que no se implementa. Si la Fase 5 encuentra senal, la
distancia de edicion sobre los candidatos prefiltrados es el siguiente paso natural.

Ausencia de leakage
-------------------
Solo se lee ``train``. El test no se lee. Se usan las etiquetas **unicamente para medir** la
pureza de los grupos; ninguna etiqueta entra en el calculo de la similitud, de modo que la
similitud no puede filtrar nada del objetivo. Este script no produce ninguna senal para un
predictor: eso es trabajo de V29-03 con los folds.

Uso
---
    /home/carlos/venv/bin/python experiments/v29_02_similitudes_laxas.py
"""

from __future__ import annotations

import json
import re
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train

EXPERIMENTO = "V29-02 (similitudes mas laxas: cobertura, pureza y relacion similitud-etiqueta)"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v29_02_similitudes_laxas.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "v29_02_similitudes_laxas.md"

ETIQUETAS = ("HS", "TR", "AG")
RE_URL_HISTORICO = re.compile(r"https?://\S+")
RE_MENCION = re.compile(r"@\w+")
RE_HASHTAG = re.compile(r"#\w+")
RE_PUNCT = re.compile(r"[!\u00bf?.,;:()\[\]{}\"'`\-_/\\|+*=<>$%&]")
FRANJAS = ((0.95, 1.01), (0.90, 0.95), (0.85, 0.90), (0.80, 0.85), (0.70, 0.80),
           (0.60, 0.70), (0.50, 0.60), (0.40, 0.50), (0.30, 0.40), (0.00, 0.30))
UMBRALES_GRUPOS = (0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50, 0.40, 0.30)


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V29-01] {mensaje}", flush=True)


def memoria_pico_mb() -> float:
    """Resident set maximum del proceso, en megabytes.

    Returns:
        La memoria maxima residente en MB.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def normalizar_historico(texto: str) -> str:
    """Reproduce la normalizacion de ``experiments/00_auditoria_datos.py``.

    Args:
        texto: texto original.

    Returns:
        Texto en minusculas, sin URLs y con espacios colapsados.
    """
    limpio = RE_URL_HISTORICO.sub(" ", texto.lower())
    return re.sub(r"\s+", " ", limpio).strip()


def normalizar_agresiva(texto: str) -> str:
    """Quita ademas menciones, hashtags y puntuacion.

    Args:
        texto: texto original.

    Returns:
        Texto normalizado de forma mas agresiva.
    """
    limpio = RE_MENCION.sub(" ", RE_HASHTAG.sub(" ", RE_PUNCT.sub(" ", texto.lower())))
    limpio = RE_URL_HISTORICO.sub(" ", limpio)
    return re.sub(r"\s+", " ", limpio).strip()


def jaccard_por_interseccion(matriz: sparse.csr_matrix) -> np.ndarray:
    """Calcula la matriz de Jaccard a partir del tamano de las intersecciones.

    ``Jaccard(A, B) = |A inter B| / |A union B|``, y ``|A union B| = |A| + |B| - |A inter B|``.
    El producto de una matriz binaria dispersa por su traspuesta da exactamente el tamano de
    la interseccion, de modo que no hace falta materializar los conjuntos.

    Args:
        matriz: matriz binaria en formato CSR, una fila por documento.

    Returns:
        Matriz de Jaccard densa.
    """
    matriz = matriz.astype(np.float32)
    interseccion = (matriz @ matriz.T).toarray()
    tamanos = np.asarray(matriz.sum(axis=1)).ravel().astype(np.float32)
    union = tamanos[:, None] + tamanos[None, :] - interseccion
    with np.errstate(divide="ignore", invalid="ignore"):
        resultado = np.where(union > 0, interseccion / union, 0.0)
    return resultado.astype(np.float32)


def componentes_conexas(n: int, pares: np.ndarray) -> np.ndarray:
    """Agrupa por componentes conexas con union-find y compresion de caminos.

    Args:
        n: numero de nodos.
        pares: matriz ``(k, 2)`` con los indices de los pares a unir.

    Returns:
        Vector con el identificador de grupo de cada nodo.
    """
    padre = list(range(n))

    def encontrar(nodo: int) -> int:
        """Devuelve la raiz del nodo con compresion de caminos."""
        while padre[nodo] != nodo:
            padre[nodo] = padre[padre[padre[nodo]]]
            nodo = padre[nodo]
        return nodo

    for a, b in pares.tolist():
        raiz_a, raiz_b = encontrar(a), encontrar(b)
        if raiz_a != raiz_b:
            padre[max(raiz_a, raiz_b)] = min(raiz_a, raiz_b)
    return np.array([encontrar(i) for i in range(n)], dtype=np.int64)


def estadisticas_de_grupo(
    etiquetas: np.ndarray, grupos: np.ndarray, minimo: int
) -> dict[str, float]:
    """Calcula el tamano, la pureza y la entropia de los grupos de un label.

    Args:
        etiquetas: vector de etiquetas binarias.
        grupos: identificador de grupo por fila.
        minimo: tamano minimo para que un grupo cuente.

    Returns:
        Diccionario con el resumen de los grupos que alcanzan ``minimo``.
    """
    conteo = pd.Series(grupos).value_counts()
    utiles = conteo[conteo >= minimo].index.to_numpy()
    if len(utiles) == 0:
        return {"n_grupos": 0, "n_filas": 0}
    selector = np.isin(grupos, utiles)
    filas = int(selector.sum())
    tamanos, puras, multi, contradictorias = [], 0, 0, 0
    entropias: list[float] = []
    puras_maj: list[float] = []
    for grupo in utiles:
        sel = grupos == grupo
        valores, frequencies = np.unique(etiquetas[sel], return_counts=True)
        n = int(sel.sum())
        tamanos.append(n)
        if valores.size == 1:
            puras += 1
        elif n > 1 and valores.size > 1 and ((frequencies == 1).any()):
            contradictorias += 1
            multi += 1
        else:
            multi += 1
        proporciones = frequencies / n
        entropias.append(float(-(proporciones * np.log2(proporciones)).sum()))
        puras_maj.append(float(frequencies.max() / n))
    arr = np.asarray(tamanos)
    n_grupos = len(utiles)
    return {
        "n_grupos": n_grupos,
        "n_filas": filas,
        "tamano_medio": float(arr.mean()),
        "tamano_mediana": float(np.median(arr)),
        "tamano_p95": float(np.percentile(arr, 95)),
        "tamano_maximo": int(arr.max()),
        "frac_grupos_una_sola_etiqueta": puras / n_grupos,
        "frac_grupos_multiples_etiquetas": multi / n_grupos,
        "frac_grupos_contradictorios": contradictorias / n_grupos,
        "pureza_media_de_grupo": float(np.mean(puras_maj)),
        "entropia_media": float(np.mean(entropias)),
    }


def main() -> None:
    """Ejecuta el diagnostico descriptivo de las similitudes mas laxas.

    Returns:
        None.
    """
    inicio = time.time()
    log(f"{EXPERIMENTO}. NINGUN modelo, NINGUNA submission, test no leido")

    train = read_train()
    textos_crudos = train["text"].astype(str).tolist()
    etiquetas = train[list(ETIQUETAS)].to_numpy(dtype=np.int64)
    n = len(train)
    log(f"   train {n} filas | pares train-train {n * (n - 1) // 2} | solo train, test no se lee")
    log("   1 497 315 pares caben en una matriz densa de 12 MB: no hace falta MinHash ni LSH")

    hist = [normalizar_historico(t) for t in textos_crudos]
    agr = [normalizar_agresiva(t) for t in textos_crudos]

    log("1/6 controles de las cinco matrices de similitud")
    vectores = {
        "A_coseno_char4_historico": TfidfVectorizer(
            analyzer="char_wb", ngram_range=(4, 4), min_df=1
        ).fit(hist),
        "B_coseno_char4_agresivo": TfidfVectorizer(
            analyzer="char_wb", ngram_range=(4, 4), min_df=1
        ).fit(agr),
        "C_coseno_char35_agresivo": TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=1
        ).fit(agr),
    }
    # `binary=True` en TfidfVectorizer NO produce una matriz binaria: deja el term frequency a
    # uno y despues aplica idf y normalizacion L2, de modo que el producto matricial pesa cada
    # n-gram en vez de contar cuantos hay. Para Jaccard hace falta el conteo, asi que se usa
    # CountVectorizer con `binary=True`, que si entrega valores de exactamente 1. El control
    # explicito de este script es el que detecto el error: la diferencia llego a 6,7e-02.
    binarios = {
        "D_jaccard_char35_agresivo": CountVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=1, binary=True
        ).fit(agr),
        "E_jaccard_tokens_agresivo": CountVectorizer(
            analyzer="word", ngram_range=(1, 1), binary=True, token_pattern=r"(?u)\b\w+\b"
        ).fit(agr),
    }
    matrices: dict[str, np.ndarray] = {}
    for clave, vec in vectores.items():
        m = vec.transform(hist if "historico" in clave else agr)
        matrices[clave] = (m @ m.T).toarray().astype("float32")
    for clave, vec in binarios.items():
        matrices[clave] = jaccard_por_interseccion(vec.transform(agr))

    controles: dict[str, Any] = {}
    for clave, matriz in matrices.items():
        diag = np.diag(matriz).copy()
        # Se trabaja sobre una copia: fill_diagonal muta el array, y las matrices se reutilizan
        # mas adelante para los grupos y para las franjas de similitud.
        matrices[clave] = matriz.copy()
        np.fill_diagonal(matrices[clave], 0.0)
        controles[clave] = {
            "auto_max_desviacion": float(np.max(np.abs(diag - 1.0))),
            "simetria_max_asimetria": float(np.max(np.abs(matriz - matriz.T))),
            "minimo": float(matriz.min()),
            "maximo": float(matriz.max()),
            "nulos_o_nan": int(np.count_nonzero(~np.isfinite(matriz))),
        }
    # control explicito del Jaccard contra conjuntos de Python
    vec = binarios["D_jaccard_char35_agresivo"]
    m = vec.transform(agr)
    conjuntos = [set(m.indices[m.indptr[i] : m.indptr[i + 1]].tolist()) for i in range(n)]
    muestras = [(0, 1), (3, 40), (77, 500), (900, 1500), (1200, 1201)]
    errores = []
    for a, b in muestras:
        explicito = len(conjuntos[a] & conjuntos[b]) / max(len(conjuntos[a] | conjuntos[b]), 1)
        errores.append(abs(explicito - float(matrices["D_jaccard_char35_agresivo"][a, b])))
    controles["jaccard_explicito_vs_matriz_error_max"] = float(max(errores))
    # Control adicional: el producto de la matriz binaria debe contar los n-gramas comunes.
    vec_d = binarios["D_jaccard_char35_agresivo"]
    m_d = vec_d.transform(agr)
    errores_conteo = []
    for a, b in muestras:
        esperado = len(conjuntos[a] & conjuntos[b])
        producto = (m_d[a] @ m_d[b].T).toarray()[0, 0]
        errores_conteo.append(abs(esperado - float(producto)))
    controles["producto_matricial_igual_a_conteo_de_interseccion"] = float(max(errores_conteo))
    controles["valores_unicos_de_la_matriz_binaria"] = int(np.unique(m_d.data).size)
    for clave, d in controles.items():
        if isinstance(d, dict):
            log(
                f"   {clave:<32} auto {d['auto_max_desviacion']:.1e} | simetria "
                f"{d['simetria_max_asimetria']:.1e} | rango [{d['minimo']:.3f}, {d['maximo']:.3f}]"
            )
        else:
            log(f"   {clave}: {d:.2e}")

    log("2/6 cobertura por metodo y umbral, y descriptivo de grupos por label")
    descriptivo: dict[str, Any] = {}
    for clave, matriz in matrices.items():
        descriptivo[clave] = {}
        for umbral in UMBRALES_GRUPOS:
            pares = np.argwhere(np.triu(matriz, k=1) >= umbral)
            grupos = componentes_conexas(n, pares)
            conteo = pd.Series(grupos).value_counts()
            cubiertos = conteo >= 2
            entrada: dict[str, Any] = {
                "n_pares": len(pares),
                "n_grupos_2_o_mas": int(cubiertos.sum()),
                "cobertura": float(cubiertos.sum() / n),
                "filas_cubiertas": int(conteo[cubiertos].sum()),
                "grupo_maximo": int(conteo.max()) if len(conteo) else 1,
                "por_label": {},
            }
            for etiqueta in ETIQUETAS:
                j = ETIQUETAS.index(etiqueta)
                entrada["por_label"][etiqueta] = estadisticas_de_grupo(
                    etiquetas[:, j], grupos, minimo=2
                )
            descriptivo[clave][f"{umbral:.2f}"] = entrada
            log(
                f"   {clave:<32} u {umbral:.2f}: pares {entrada['n_pares']:>7} | "
                f"grupos {entrada['n_grupos_2_o_mas']:>4} | cobertura {entrada['cobertura']:.4f} | "
                f"max {entrada['grupo_maximo']:>4}"
            )

    log("3/6 relacion similitud -> misma etiqueta, por franja y por label (Fase 4)")
    # Se recorre el triangulo superior una vez y se acumula por franja. No se asume ningun
    # umbral: lo que se busca es si existe una zona donde compartir similitud elevate de forma
    # clara la probabilidad de compartir etiqueta, por encima de la linea base.
    base = etiquetas.mean(axis=0)
    relacion: dict[str, Any] = {}
    for clave, matriz in matrices.items():
        pares_i, pares_j = np.triu_indices(n, k=1)
        valores = matriz[pares_i, pares_j]
        franjas: dict[str, Any] = {}
        for j, etiqueta in enumerate(ETIQUETAS):
            iguales = etiquetas[pares_i, j] == etiquetas[pares_j, j]
            por_franja: dict[str, Any] = {}
            for baja, alta in FRANJAS:
                mask = (valores >= baja) & (valores < alta)
                m = int(mask.sum())
                if m == 0:
                    por_franja[f"{baja:.2f}_{alta:.2f}"] = {
                        "n_pares": 0,
                        "prob_misma_etiqueta": None,
                        "linea_base_esperada": float(
                            base[j] ** 2 + (1 - base[j]) ** 2
                        ),
                    }
                    continue
                por_franja[f"{baja:.2f}_{alta:.2f}"] = {
                    "n_pares": m,
                    "n_iguales": int(iguales[mask].sum()),
                    "prob_misma_etiqueta": float(iguales[mask].mean()),
                    "linea_base_esperada": float(base[j] ** 2 + (1 - base[j]) ** 2),
                }
            franjas[etiqueta] = por_franja
        relacion[clave] = franjas
        log(f"   {clave}: relacion calculada sobre {len(valores)} pares")

    log("4/6 lectura: donde se separa la probabilidad de compartir etiqueta")
    # Para cada metodo se busca la franja de mayor eficiencia: mayor proporcion de pares con la
    # misma etiqueta por par visitante, exigiendo una cobertura minima para que sirva de algo.
    MINIMO_PARES = 200
    hallazgos: list[dict[str, Any]] = []
    for clave, matriz in matrices.items():
        pares_i, pares_j = np.triu_indices(n, k=1)
        valores = matriz[pares_i, pares_j]
        for etiqueta in ETIQUETAS:
            j = ETIQUETAS.index(etiqueta)
            iguales = etiquetas[pares_i, j] == etiquetas[pares_j, j]
            mejor = None
            for baja, alta in FRANJAS:
                mask = (valores >= baja) & (valores < alta)
                m = int(mask.sum())
                if m < MINIMO_PARES:
                    continue
                prob = float(iguales[mask].mean())
                esperada = float(base[j] ** 2 + (1 - base[j]) ** 2)
                mejora = prob - esperada
                if mejor is None or mejora > mejor["mejora_sobre_linea_base"]:
                    mejor = {
                        "franja": f"{baja:.2f}_{alta:.2f}",
                        "n_pares": m,
                        "prob_misma_etiqueta": prob,
                        "linea_base_esperada": esperada,
                        "mejora_sobre_linea_base": mejora,
                    }
            hallazgos.append({"metodo": clave, "etiqueta": etiqueta, "mejor_franja": mejor})
    for h in hallazgos:
        m = h["mejor_franja"]
        if m is None:
            log(f"   {h['metodo']:<32} {h['etiqueta']}: ninguna franja alcanza {MINIMO_PARES} pares")
        else:
            log(
                f"   {h['metodo']:<32} {h['etiqueta']}: mejor franja {m['franja']} con "
                f"{m['prob_misma_etiqueta']:.4f} frente a {m['linea_base_esperada']:.4f} "
                f"esperado ({m['n_pares']} pares)"
            )

    log("5/6 la pregunta de la directiva: falsos vecinos, grupos puros, o plantillas")
    # Criterio explicito y declarado por adelantado:
    #   A) ruido: la probabilidad de compartir etiqueta no supera la linea base;
    #   B) grupos puros: la supera, pero la cobertura se queda en las decimas de punto;
    #   C) plantillas: la supera y hay cobertura suficiente para que un vecino exista.
    veredictos: dict[str, str] = {}
    for clave in matrices:
        mejor_mejora = max(
            (h["mejor_franja"]["mejora_sobre_linea_base"] for h in hallazgos
             if h["metodo"] == clave and h["mejor_franja"]), default=0.0
        )
        mejor_cobertura = max(
            d["cobertura"] for d in descriptivo[clave].values()
        )
        if mejor_mejora <= 0.0:
            veredictos[clave] = "A_RUIDO"
        elif mejor_cobertura < 0.10:
            veredictos[clave] = "B_GRUPOS_PUROS_POCA_COBERTURA"
        else:
            veredictos[clave] = "C_POTENCIAL_TEMPLATE"
        log(
            f"   {clave:<32} mejor mejora {mejor_mejora:+.4f} | mejor cobertura "
            f"{mejor_cobertura:.4f} -> {veredictos[clave]}"
        )

    log("6/6 informe")
    informe = {
        "familia": "FUZZY_TEMPLATE_MATCHING",
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estado": "DESCRIPTIVE_ONLY",
        "pregunta": (
            "al relajar la similitud, aparecen falsos vecinos, grupos grandes pero puros, o "
            "plantillas reales con informacion de etiqueta"
        ),
        "alcance": (
            "diagnostico descriptivo. No se entrena nada, no se construye predictor y no se "
            "produce submission. Las etiquetas solo se usan para MEDIR la pureza, nunca entran "
            "en el calculo de la similitud"
        ),
        "metodos": {
            "A_coseno_char4_historico": "coseno TF-IDF char_wb 4-4, normalizacion historica; es el metodo de V03",
            "B_coseno_char4_agresivo": "coseno TF-IDF char_wb 4-4, normalizacion agresiva",
            "C_coseno_char35_agresivo": "coseno TF-IDF char_wb 3-5, normalizacion agresiva",
            "D_jaccard_char35_agresivo": "Jaccard de char 3-5 binarios, normalizacion agresiva",
            "E_jaccard_tokens_agresivo": "Jaccard de unigramas de palabra binarios, normalizacion agresiva",
        },
        "por_que_no_minhash": (
            "con 1731 filas hay 1 497 315 pares y la matriz densa ocupa 12 MB; un indice "
            "aproximado no aporta nada y anade una fuente de error"
        ),
        "por_que_no_levenshtein": (
            "la directiva lo admite solo para candidatos prefiltrados. Este script es "
            "descriptivo y no lo necesita; queda como paso natural si la Fase 5 encuentra senal"
        ),
        "controles": controles,
        "descriptivo_por_umbral": descriptivo,
        "relacion_similitud_etiqueta": relacion,
        "mejor_franja_por_metodo_y_label": hallazgos,
        "veredictos_por_metodo": veredictos,
        "criterio_del_veredicto": (
            "A_RUIDO si ninguna franja mejora la linea base; B si la mejora existe pero la "
            f"cobertura se queda por debajo del 10 %; C si la mejora existe con cobertura "
            f"suficiente. Se exige al menos {MINIMO_PARES} pares por franja para considerar una"
        ),
        "gastos": {
            "submissions": 0,
            "modelos_entrenados": 0,
            "memoria_pico_mb": round(memoria_pico_mb(), 1),
            "segundos": round(time.time() - inicio, 1),
        },
    }
    REPORTE_JSON.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _informe_md(informe)
    log("   informe escrito en reports/v29_02_similitudes_laxas.{json,md}")
    log(f"   memoria pico {memoria_pico_mb():.0f} MB | tiempo {round(time.time() - inicio, 1)} s")
    log("VEREDICTO: diagnostico descriptivo completado, 0 modelos, 0 submissions")


def _informe_md(informe: dict[str, Any]) -> None:
    """Escribe el informe legible.

    Args:
        informe: informe con los datos.

    Returns:
        None.
    """
    d = informe["descriptivo_por_umbral"]
    rel = informe["relacion_similitud_etiqueta"]
    lineas = [
        "# V29-02: similitudes mas laxas, cobertura, pureza y relacion similitud-etiqueta",
        "",
        f"Generado por `experiments/v29_02_similitudes_laxas.py` ({EXPERIMENTO}).",
        "",
        f"**Estado `{informe['estado']}`. Alcance: {informe['alcance']}**",
        "",
        "## 1. Los cinco metodos",
        "",
        "| clave | metodo |",
        "|---|---|",
    ]
    for clave, desc in informe["metodos"].items():
        lineas.append(f"| `{clave}` | {desc} |")
    lineas += [
        "",
        f"Sin MinHash ni LSH: {informe['por_que_no_minhash']}.",
        "",
        f"Sin distancia de edicion: {informe['por_que_no_levenshtein']}.",
        "",
        "## 2. Controles",
        "",
        "| metodo | auto | simetria | rango | no finitos |",
        "|---|---:|---:|---|---:|",
    ]
    for clave, c in informe["controles"].items():
        if isinstance(c, dict):
            lineas.append(
                f"| `{clave}` | {c['auto_max_desviacion']:.1e} | {c['simetria_max_asimetria']:.1e} | "
                f"[{c['minimo']:.3f}, {c['maximo']:.3f}] | {c['nulos_o_nan']} |"
            )
    lineas += [
        "",
        (f"Jaccard explicito contra la matriz, error maximo: "
        f"`{informe['controles']['jaccard_explicito_vs_matriz_error_max']:.2e}`."),
        "",
        "## 3. Cobertura por metodo y umbral",
        "",
        "| metodo | " + " | ".join(f"u{u:.2f}" for u in UMBRALES_GRUPOS) + " |",
        "|---|" + "---:|" * len(UMBRALES_GRUPOS),
    ]
    for clave in informe["metodos"]:
        fila = " | ".join(
            f"{d[clave][f'{u:.2f}']['cobertura']:.4f}" for u in UMBRALES_GRUPOS
        )
        lineas.append(f"| `{clave}` | {fila} |")
    lineas += [
        "",
        "## 4. La relacion similitud -> misma etiqueta",
        "",
        "La linea base es la que se obtiene suponiendo las dos etiquetas independientes, es decir",
        "`p^2 + (1-p)^2` con `p` la frecuencia del positivo. Una franja solo es interesante si la",
        "probabilidad de compartir etiqueta supera esa linea base **y** tiene volumen de pares.",
        "",
    ]
    for clave in informe["metodos"]:
        lineas.append(f"### `{clave}`")
        lineas.append("")
        lineas.append("| franja | " + " | ".join(ETIQUETAS) + " |")
        lineas.append("|---|---:|---:|---:|")
        for baja, alta in FRANJAS:
            nombre = f"{baja:.2f}_{alta:.2f}"
            celdas = []
            for etiqueta in ETIQUETAS:
                f = rel[clave][etiqueta][nombre]
                if f["n_pares"] == 0:
                    celdas.append("-")
                else:
                    celdas.append(
                        f"{f['prob_misma_etiqueta']:.4f} (n={f['n_pares']})"
                    )
            lineas.append(f"| {nombre} | " + " | ".join(celdas) + " |")
        lineas.append("")
    lineas += [
        "## 5. La pregunta de la directiva: A, B o C",
        "",
        f"Criterio declarado por adelantado: {informe['criterio_del_veredicto']}.",
        "",
        "| metodo | veredicto |",
        "|---|---|",
    ]
    for clave, v in informe["veredictos_por_metodo"].items():
        lineas.append(f"| `{clave}` | **{v}** |")
    lineas += [
        "",
        "La mejor franja de cada metodo y label:",
        "",
        "| metodo | label | franja | prob. misma etiqueta | linea base | mejora | pares |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for h in informe["mejor_franja_por_metodo_y_label"]:
        m = h["mejor_franja"]
        if m is None:
            lineas.append(f"| `{h['metodo']}` | {h['etiqueta']} | ninguna con volumen | - | - | - | - |")
        else:
            lineas.append(
                f"| `{h['metodo']}` | {h['etiqueta']} | {m['franja']} | "
                f"{m['prob_misma_etiqueta']:.4f} | {m['linea_base_esperada']:.4f} | "
                f"**{m['mejora_sobre_linea_base']:+.4f}** | {m['n_pares']} |"
            )
    g = informe["gastos"]
    lineas += [
        "",
        "## 6. Lo que este script no hace",
        "",
        "- No entrena ningun modelo.",
        "- No construye predictor ni senal para un modelo.",
        "- No lee el test.",
        "- No produce submission.",
        "- Las etiquetas se usan solo para medir pureza, nunca en el calculo de similitud.",
        "",
        (f"Coste: {g['segundos']} s, memoria {g['memoria_pico_mb']:.0f} MB, "
        f"{g['modelos_entrenados']} modelos, {g['submissions']} submissions."),
        "",
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
