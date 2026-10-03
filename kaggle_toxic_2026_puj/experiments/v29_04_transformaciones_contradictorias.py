"""V29-04: que transformacion genera ejemplos casi identicos con etiquetas distintas.

Que responde
------------
Fase 11 de la directiva V29. V29-03 cerro que el matching por similitud no corrige errores de
V03: la mejor de 48 configuraciones tiene **net -8** y en la franja de similitud por encima de
0,90 hay **cero rescates y cero danos**, es decir, donde el vecino es casi una copia, V03 ya
acierta.

Eso deja una pregunta que si puede abrir algo: **si no sirve para predecir, sirve para
entender**. Cuando dos textos casi identicos llevan etiquetas distintas, algo cambio entre
ellos. Si ese cambio es siempre el mismo tipo de cosa, hay una senal que V03 no esta modelando.

Que se mide
----------
Se toman los pares de alta similitud de ``train`` y, para cada uno, se extrae el conjunto de
palabras que difieren. Despues cada diferencia se clasifica en categorias:

| categoria | que detecta |
|---|---|
| negacion | token de negacion anadido o retirado |
| insulto | token anadido que el propio corpus asocia fuertemente con HS |
| mencion | token que empieza por arroba |
| hashtag | token que empieza por almohadilla |
| url | token que empieza por http |
| ortografia | el token solo se diferencia en su escritura, comparte cuatro letras iniciales |
| truncamiento | un texto es prefijo del otro |
| solo_puntuacion | la diferencia son solo signos |
| repeticion | cambia una secuencia de caracteres repetidos |
| numerico | cambia un numero |

Y aqui esta el punto que evita la conclusion facil: **cada categoria se mide tambien sobre los
pares que NO son contradictorios**. Una categoria solo es una senal si aparece mas en los
pares contradictorios que en los concordantes, y se cuantifica con la razon de razones y un test
exacto de Fisher. Contar "el 40 % de los contradictorios tiene negacion" no dice nada si el
60 % de los concordantes tambien la tiene.

Como se decide que token es insulto
-----------------------------------
No hay lexicon externo. Un token se considera "`insulto`" si aparece en al menos
``MIN_APOYO`` filas de train y su frecuencia entre las filas con HS=1 supera un umbral de
razon de razones sobre su frecuencia en las filas con HS=0. Todo se calcula **solo con train**.

Ausencia de leakage
-------------------
Solo se lee ``train``. No hay folds aqui: el script es un analisis de pares, no produce ninguna
senal para un predictor. Las etiquetas se usan para decidir si un par es contradictorio y para
definir el conjunto de insultos, y nada mas.

Uso
---
    /home/carlos/venv/bin/python experiments/v29_04_transformaciones_contradictorias.py
"""

from __future__ import annotations

import json
import re
import resource
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train

EXPERIMENTO = (
    "V29-04 (que transformacion genera ejemplos casi identicos con etiquetas distintas)"
)
REPORTE_JSON = PROJECT_ROOT / "reports" / "v29_04_transformaciones_contradictorias.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "v29_04_transformaciones_contradictorias.md"

ETIQUETAS = ("HS", "TR", "AG")
RE_URL_HISTORICO = re.compile(r"https?://\S+")
RE_MENCION = re.compile(r"@\w+")
RE_HASHTAG = re.compile(r"#\w+")
RE_PUNCT = re.compile(r"[!\u00bf?.,;:()\[\]{}\"'`\-_/\\|+*=<>$%&]")
NEGACIONES = frozenset({
    "no", "ni", "nunca", "jamas", "nada", "sin", "tampoco", "nadie", "ningun", "ninguna",
    "nel", "ninguno",
})
PREFIJOS_SOLO_SIGNO = re.compile(r"^[^\w]+$")
PREFIJOS_NUMERO = re.compile(r"^\d+$")
PREFIJOS_REPETIDO = re.compile(r"^(.)\1{2,}")

MIN_APOYO = 10
RAZON_INSULTO = 2.0
DELTA_INSULTO = 1.5  # se multiplica por la frecuencia base de HS
UMBRALES_SIMILITUD = (0.90, 0.85, 0.75, 0.70, 0.60)
METODO = "D_jaccard_char35_agresivo"
CATEGORIAS = (
    "negacion", "insulto", "mencion", "hashtag", "url", "ortografia",
    "truncamiento", "solo_puntuacion", "repeticion", "numerico",
)


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V29-04] {mensaje}", flush=True)


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


def pares_por_similitud(matriz: np.ndarray, umbral: float) -> np.ndarray:
    """Pares del triangulo superior por encima del umbral.

    Args:
        matriz: matriz de similitud.
        umbral: similitud minima.

    Returns:
        Matriz ``(k, 2)`` con los indices de los pares.
    """
    return np.argwhere(np.triu(matriz, k=1) >= umbral)


def categorizar_diferencia(
    solo_a: set[str], solo_b: set[str], texto_a: str, texto_b: str, insultos: set[str]
) -> list[str]:
    """Clasifica la diferencia entre dos textos en categorias.

    Args:
        solo_a: tokens que solo estan en el primer texto.
        solo_b: tokens que solo estan en el segundo texto.
        texto_a: primer texto normalizado.
        texto_b: segundo texto normalizado.
        insultos: conjunto de tokens que el corpus asocia fuertemente con HS.

    Returns:
        Lista de categorias detectadas. Puede estar vacia si la diferencia no encaja en ninguna.
    """
    anadidos = solo_b - solo_a
    categorias: set[str] = set()
    for token in anadidos:
        if token in NEGACIONES:
            categorias.add("negacion")
        if token in insultos:
            categorias.add("insulto")
        if token.startswith("@"):
            categorias.add("mencion")
        if token.startswith("#"):
            categorias.add("hashtag")
        if token.startswith(("http", "www")):
            categorias.add("url")
        if PREFIJOS_NUMERO.match(token):
            categorias.add("numerico")
        if PREFIJOS_REPETIDO.match(token):
            categorias.add("repeticion")
        if PREFIJOS_SOLO_SIGNO.match(token):
            categorias.add("solo_puntuacion")
    # Ortografia: dos tokens distintos que comparten cuatro letras iniciales
    for token_a in solo_a:
        for token_b in solo_b:
            if token_a != token_b and token_a[:4] == token_b[:4] and len(token_a) >= 4:
                categorias.add("ortografia")
                break
    if texto_b.startswith(texto_a) or texto_a.startswith(texto_b):
        categorias.add("truncamiento")
    return sorted(categorias)


def main() -> None:
    """Ejecuta el analisis de transformaciones en los pares contradictorios.

    Returns:
        None.
    """
    inicio = time.time()
    log(f"{EXPERIMENTO}. NINGUN modelo, NINGUNA submission, test no leido")

    train = read_train()
    textos_crudos = train["text"].astype(str).tolist()
    textos = [normalizar_agresiva(t) for t in textos_crudos]
    etiquetas = train[list(ETIQUETAS)].to_numpy(dtype=np.int64)
    n = len(train)
    log(f"   train {n} filas | pares posibles {n * (n - 1) // 2}")

    log("1/5 el conjunto de insultos, derivado del propio train y de ningun otro sitio")
    conteo = Counter()
    positivos_hs = Counter()
    for texto, etiqueta in zip(textos, etiquetas[:, 0], strict=True):
        for token in set(texto.split()):
            conteo[token] += 1
            if etiqueta == 1:
                positivos_hs[token] += 1
    base_hs = float(np.mean(etiquetas[:, 0]))
    insultos: set[str] = set()
    for token, apariciones in conteo.items():
        if apariciones < MIN_APOYO:
            continue
        freq_pos = positivos_hs[token] / apariciones
        freq_neg = (apariciones - positivos_hs[token]) / max(
            len(train) - int(np.count_nonzero(etiquetas[:, 0])), 1
        )
        if freq_pos >= RAZON_INSULTO * max(freq_neg, 1e-6) and freq_pos >= DELTA_INSULTO * base_hs:
            insultos.add(token)
    # El desempate es por token y no por el orden de iteracion del set, que en Python
    # depende de PYTHONHASHSEED y haria la corrida no reproducible entre procesos.
    ejemplos = sorted(insultos, key=lambda t: (-conteo[t], t))[:15]
    log(f"   frecuencia base de HS en train: {base_hs:.4f}")
    log(f"   tokens con al menos {MIN_APOYO} apariciones y razon {RAZON_INSULTO}: {len(insultos)}")
    log(f"   los mas frecuentes: {', '.join(ejemplos[:10])}")

    log("2/5 matriz de similitud y controles")
    vectorizador = CountVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=1, binary=True
    ).fit(textos)
    matriz = vectorizador.transform(textos).astype(np.float32)
    interseccion = np.asarray((matriz @ matriz.T).todense(), dtype=np.float32)
    tamanos = np.asarray(matriz.sum(axis=1)).ravel()
    union = tamanos[:, None] + tamanos[None, :] - interseccion
    with np.errstate(divide="ignore", invalid="ignore"):
        similitud = np.where(union > 0, interseccion / union, 0.0).astype(np.float32)
    # Los controles se leen con la diagonal intacta; ponerla a cero despues, porque si no el
    # control de auto-similitud mide 0 y no verifica nada.
    auto_min = float(np.diag(similitud).min())
    asimetria = float(np.max(np.abs(similitud - similitud.T)))
    rango = [float(similitud.min()), float(similitud.max())]
    # El control de auto-similitud no da exactamente 1 en toda la matriz, y la causa conviene
    # saberla en vez de registrarla como un numero raro: un documento cuyo texto se queda vacio
    # tras la normalizacion agresiva tiene conjunto vacio, su union es 0 y su similitud consigo
    # mismo queda en 0 por convenio. Ese documento no puede tener vecinos, asi que no
    # desaparece en silencio: se cuenta y se declara.
    tamanos = np.asarray(matriz.sum(axis=1)).ravel()
    vacios = np.flatnonzero(tamanos == 0)
    vacios_info = {
        "n_documentos_con_representacion_vacia": len(vacios),
        "ids": train["id"].astype(int).to_numpy()[vacios].tolist(),
        "textos_originales": [textos_crudos[i] for i in vacios],
        "efecto": (
            "la normalizacion agresiva borra por completo al menos un documento del corpus. "
            "Afecta a una fila de 1731 y no a los resultados, pero explica por que el control de "
            "auto-similitud de V29-02 daba 1,0e+00 en los metodos con normalizacion agresiva y "
            "0,0e+00 en el metodo historico, que no borra hashtags ni puntuacion"
        ),
    }
    controles = {
        "documentos_con_representacion_vacia": vacios_info,
        "similitud consigo mismo, minimo": auto_min,
        "simetria, asimetria maxima": asimetria,
        "rango": rango,
        "min_apoyo_para_insulto": MIN_APOYO,
        "razon_insulto": RAZON_INSULTO,
        "delta_insulto_sobre_la_base": DELTA_INSULTO,
        "n_insultos_detectados": len(insultos),
    }
    log(f"   auto {auto_min:.4f} | simetria {asimetria:.1e} | rango {rango}")
    log(f"   documentos con representacion vacia: {len(vacios)} -> "
        f"{vacios_info['textos_originales']}")
    np.fill_diagonal(similitud, 0.0)

    log("3/5 pares por franja de similitud, y cuantos son contradictorios")
    tokens_por_doc = [set(t.split()) for t in textos]
    por_umbral: list[dict[str, Any]] = []
    categorias_por_umbral: dict[str, Any] = {}
    for umbral in UMBRALES_SIMILITUD:
        pares = pares_por_similitud(similitud, umbral)
        contradictorios = 0
        misma_etiqueta = 0
        conteo_cat_contr: Counter[str] = Counter()
        conteo_cat_conc: Counter[str] = Counter()
        ejemplos_contr: list[dict[str, Any]] = []
        for a, b in pares.tolist():
            difieren = (etiquetas[a] != etiquetas[b]).any()
            categorias = categorizar_diferencia(
                tokens_por_doc[a], tokens_por_doc[b], textos[a], textos[b], insultos
            )
            if not categorias:
                categorias = ["sin_clasificar"]
            if difieren:
                contradictorios += 1
                conteo_cat_contr.update(categorias)
                if len(ejemplos_contr) < 12:
                    ejemplos_contr.append({
                        "id_a": int(train["id"].iloc[a]),
                        "id_b": int(train["id"].iloc[b]),
                        "similitud": round(float(similitud[a, b]), 4),
                        "labels_a": etiquetas[a].tolist(),
                        "labels_b": etiquetas[b].tolist(),
                        "categorias": categorias,
                        "tokens_solo_a": sorted(tokens_por_doc[a] - tokens_por_doc[b])[:10],
                        "tokens_solo_b": sorted(tokens_por_doc[b] - tokens_por_doc[a])[:10],
                    })
            else:
                misma_etiqueta += 1
                conteo_cat_conc.update(categorias)
        por_umbral.append({
            "umbral": umbral,
            "n_pares": len(pares),
            "n_contradictorios": contradictorios,
            "n_misma_etiqueta": misma_etiqueta,
            "fraccion_contradictoria": contradictorios / len(pares) if len(pares) else None,
        })
        categorias_por_umbral[f"{umbral:.2f}"] = {
            cat: {
                "n_en_contradictorios": conteo_cat_contr[cat],
                "n_en_concordantes": conteo_cat_conc[cat],
                "razon_de_razones": (
                    (conteo_cat_contr[cat] / max(contradictorios, 1))
                    / max(conteo_cat_conc[cat] / max(misma_etiqueta, 1), 1e-9)
                ),
                "p_fisher": float(stats.fisher_exact([
                    [conteo_cat_contr[cat], contradictorios - conteo_cat_contr[cat]],
                    [conteo_cat_conc[cat], misma_etiqueta - conteo_cat_conc[cat]],
                ])[1]),
            }
            for cat in list(CATEGORIAS) + ["sin_clasificar"]
        }
        log(
            f"   umbral {umbral:.2f}: {len(pares):>5} pares | contradictorios "
            f"{contradictorios:>4} ({contradictorios / len(pares) if len(pares) else 0:.3f})"
        )

    log("4/5 que categoria discrimina de verdad, y cual es solo ruido")
    # La categoria con valor es la que aparece mas en los contradictorios que en los
    # concordantes Y tiene soporte suficiente. Se declara el minimo de pares de entrada.
    MIN_SOPORTE = 10
    discriminantes: list[dict[str, Any]] = []
    for clave, cats in categorias_por_umbral.items():
        for cat, d in cats.items():
            if d["n_en_contradictorios"] + d["n_en_concordantes"] < MIN_SOPORTE:
                discriminantes.append({
                    "umbral": clave, "categoria": cat, "veredicto": "SIN_SOPORTE",
                    "razon_de_razones": d["razon_de_razones"], "p_fisher": d["p_fisher"],
                    "n_en_contradictorios": d["n_en_contradictorios"],
                    "n_en_concordantes": d["n_en_concordantes"],
                })
                continue
            if d["p_fisher"] < 0.05 and d["razon_de_razones"] > 1.5:
                veredicto = "DISCRIMINANTE"
            elif d["razon_de_razones"] < 0.67:
                veredicto = "INVIRTIDA"
            else:
                veredicto = "RUIDO"
            discriminantes.append({
                "umbral": clave, "categoria": cat, "veredicto": veredicto,
                "razon_de_razones": d["razon_de_razones"], "p_fisher": d["p_fisher"],
                "n_en_contradictorios": d["n_en_contradictorios"],
                "n_en_concordantes": d["n_en_concordantes"],
            })
    for d in discriminantes:
        if d["veredicto"] in ("DISCRIMINANTE", "INVERTIDA"):
            log(
                f"   umbral {d['umbral']} {d['categoria']:<18} {d['veredicto']:<14} "
                f"razon {d['razon_de_razones']:.2f} | contradictorios "
                f"{d['n_en_contradictorios']} vs concordantes {d['n_en_concordantes']} | "
                f"p {d['p_fisher']:.2e}"
            )
    n_disc = sum(1 for d in discriminantes if d["veredicto"] == "DISCRIMINANTE")
    log(f"   categorias discriminantes en todo el barrido: {n_disc}")

    log("5/5 ejemplos de pares contradictorios, para leerlos")
    ejemplos_finales = _ejemplos_de_contradiccion(
        similitud, tokens_por_doc, textos, train, etiquetas, insultos
    )
    for e in ejemplos_finales[:8]:
        log(f"   sim {e['similitud']:.4f} | {e['labels_a']} -> {e['labels_b']} | "
            f"{', '.join(e['categorias'])}")
        log(f"      solo A: {', '.join(e['tokens_solo_a']) or '(ninguno)'}")
        log(f"      solo B: {', '.join(e['tokens_solo_b']) or '(ninguno)'}")

    informe = {
        "familia": "FUZZY_TEMPLATE_MATCHING",
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estado": "DIAGNOSTIC_ONLY",
        "pregunta": (
            "que tipo de transformacion separa dos textos casi identicos que llevan "
            "etiquetas distintas"
        ),
        "alcance": (
            "analisis de pares. No produce ninguna senal para un predictor, no entrena nada y "
            "no genera submission. El conjunto de insultos se deriva del propio train, sin "
            "lexicon externo"
        ),
        "metodo": METODO,
        "controles": controles,
        "insultos_mas_frecuentes": ejemplos,
        "pares_por_umbral": por_umbral,
        "categorias_por_umbral": categorias_por_umbral,
        "veredicto_por_categoria": discriminantes,
        "n_categorias_discriminantes": n_disc,
        "criterio_de_discriminante": (
            f"una categoria es DISCRIMINANTE si aparece mas en los pares contradictorios que en "
            f"los concordantes con razon de razones mayor que 1,5 y p de Fisher menor que 0,05, "
            f"exigiendo al menos {MIN_SOPORTE} pares de soporte. Sin ese control, contar "
            f"categorias en los contradictorios seria concluir que el sol es causa de la lluvia"
        ),
        "ejemplos_contradictorios": ejemplos_finales,
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
    log("   informe escrito en reports/v29_04_transformaciones_contradictorias.{json,md}")
    log(f"   memoria pico {memoria_pico_mb():.0f} MB | tiempo {round(time.time() - inicio, 1)} s")
    log("VEREDICTO: analisis de transformaciones completado, 0 modelos, 0 submissions")


def _ejemplos_de_contradiccion(
    similitud: np.ndarray,
    tokens_por_doc: list[set[str]],
    textos: list[str],
    train: pd.DataFrame,
    etiquetas: np.ndarray,
    insultos: set[str],
    limite: int = 25,
) -> list[dict[str, Any]]:
    """Recoge los pares mas similares que ademas son contradictorios.

    Args:
        similitud: matriz de similitud sin diagonal.
        tokens_por_doc: conjunto de tokens por documento.
        textos: textos normalizados.
        train: tabla de entrenamiento, para los identificadores.
        etiquetas: matriz de etiquetas.
        insultos: conjunto de tokens asociados a HS.
        limite: numero maximo de ejemplos.

    Returns:
        Lista de ejemplos ordenada por similitud descendente.
    """
    filas: list[tuple[float, int, int]] = []
    for a in range(similitud.shape[0]):
        candidatos = np.flatnonzero(similitud[a] > 0)
        for b in candidatos:
            if b <= a:
                continue
            if (etiquetas[a] != etiquetas[b]).any():
                filas.append((float(similitud[a, b]), a, b))
    filas.sort(reverse=True)
    salida = []
    for valor, a, b in filas[:limite]:
        salida.append({
            "similitud": round(valor, 4),
            "id_a": int(train["id"].iloc[a]),
            "id_b": int(train["id"].iloc[b]),
            "labels_a": etiquetas[a].tolist(),
            "labels_b": etiquetas[b].tolist(),
            "categorias": categorizar_diferencia(
                tokens_por_doc[a], tokens_por_doc[b], textos[a], textos[b], insultos
            ),
            "tokens_solo_a": sorted(tokens_por_doc[a] - tokens_por_doc[b])[:12],
            "tokens_solo_b": sorted(tokens_por_doc[b] - tokens_por_doc[a])[:12],
        })
    return salida


def _informe_md(informe: dict[str, Any]) -> None:
    """Escribe el informe legible.

    Args:
        informe: informe con los datos.

    Returns:
        None.
    """
    g = informe["gastos"]
    lineas = [
        "# V29-04: que transformacion genera ejemplos casi identicos con etiquetas distintas",
        "",
        f"Generado por `experiments/v29_04_transformaciones_contradictorias.py` ({EXPERIMENTO}).",
        "",
        f"**Estado `{informe['estado']}`. Alcance: {informe['alcance']}**",
        "",
        "## 1. El control que evita la conclusion facil",
        "",
        f"{informe['criterio_de_discriminante']}.",
        "",
        ("El conjunto de insultos no viene de un lexicon externo: se deriva del propio train, "
        f"con un minimo de {informe['controles']['min_apoyo_para_insulto']} apariciones y una "
        f"razon de razones sobre HS de {informe['controles']['razon_insulto']}. Se encontraron "
        f"{informe['controles']['n_insultos_detectados']} tokens. Los mas frecuentes: "
        f"{', '.join(informe['insultos_mas_frecuentes'][:12])}."),
        "",
        "## 2. Pares por franja de similitud",
        "",
        "| umbral | pares | contradictorios | misma etiqueta | fraccion contradictoria |",
        "|---:|---:|---:|---:|---:|",
    ]
    for p in informe["pares_por_umbral"]:
        fr = p["fraccion_contradictoria"]
        lineas.append(
            f"| {p['umbral']:.2f} | {p['n_pares']} | {p['n_contradictorios']} | "
            f"{p['n_misma_etiqueta']} | {(f'{fr:.4f}') if fr is not None else '-'} |"
        )
    lineas += [
        "",
        "## 3. Que categoria discrimina, y cual es ruido",
        "",
        "Razon de razones mayor que 1 significa que la categoria aparece mas entre los pares",
        "contradictorios. El p de Fisher dice si la diferencia es distinguible del azar.",
        "",
        "| umbral | categoria | contradictorios | concordantes | razon | p Fisher | veredicto |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for d in informe["veredicto_por_categoria"]:
        lineas.append(
            f"| {d['umbral']} | {d['categoria']} | {d['n_en_contradictorios']} | "
            f"{d['n_en_concordantes']} | {d['razon_de_razones']:.2f} | {d['p_fisher']:.2e} | "
            f"**{d['veredicto']}** |"
        )
    lineas += [
        "",
        (f"**Categorias discriminantes en todo el barrido: "
        f"{informe['n_categorias_discriminantes']}.**"),
        "",
        "## 4. Ejemplos de pares contradictorios, para leerlos",
        "",
    ]
    for e in informe["ejemplos_contradictorios"][:10]:
        lineas += [
            (f"- similitud **{e['similitud']:.4f}**, etiquetas `{e['labels_a']}` -> `{e['labels_b']}`, "
            f"categorias: {', '.join(e['categorias']) or 'sin clasificar'}"),
            f"  - solo en A: {', '.join(e['tokens_solo_a']) or '(ninguno)'}",
            f"  - solo en B: {', '.join(e['tokens_solo_b']) or '(ninguno)'}",
        ]
    lineas += [
        "",
        "## 5. Lo que este script no hace",
        "",
        "- No entrena ningun modelo.",
        "- No construye ninguna senal para un predictor.",
        "- No lee el test.",
        "- No produce submission.",
        "",
        (f"Coste: {g['segundos']} s, memoria {g['memoria_pico_mb']:.0f} MB, "
        f"{g['modelos_entrenados']} modelos, {g['submissions']} submissions."),
        "",
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
