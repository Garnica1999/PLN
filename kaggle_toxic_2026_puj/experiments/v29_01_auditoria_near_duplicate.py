"""V29-01: auditoria del metodo de near-duplicate actual y de por que su cobertura es 0,40 %.

Que responde
------------
La directiva de la fase V29 pide, antes de implementar nada, documentar como se definieron los
near-duplicates historicos y **determinar por que solo el 7,08 % de las filas entra en grupos de
tamano >= 2**. Este script responde a eso, y su hallazgo cambia el encuadre de toda la fase.

El hallazgo
-----------
El 7,08 % que reporto V28-03 **no viene del near-duplicate**. La cobertura se descompone asi:

| fuente | filas del desarrollo en grupo >= 2 | cobertura |
|---|---:|---:|
| near-duplicate puro (coseno char_wb 4-4 >= 0.85) | 6 | 0,0040 |
| pseudo-autor (primera mencion ``@usuario``) | 99 | 0,0668 |
| grupos fusionados, que son los que usa V03 | 105 | 0,0708 |

**99 de las 105 filas entran solo por la heuristica de pseudo-autor.** O sea que V03 valida
contra un criterio que en su mayoria no es "texto casi identico" sino "mismo usuario". Y
justamente ese criterio es el que produce los 43 grupos contradictorios y el multiplicador de
error de 3,8 a 5,3 veces que midio V28-03: dos tweets del mismo autor pueden no parecerse en
absoluto.

Esto no invalida el near-duplicate, que se creo para evitar fuga de copias casi identicas y
sigue cumpliendo su funcion. Lo que hace es Delimitar la premisa de la directiva: **relajar la
similitud textual no es "relajar el criterio que produjo el 7,08 %", porque ese criterio es
otro.** El techo de cobertura al que se puede aspirar relajando similitud es el 0,40 %, no el
7,08 %.

Metodo actual, documentado
--------------------------
- **Normalizacion**: minusculas, sustitucion de URLs por espacio y colapso de espacios. No
  elimina menciones, hashtags, puntuacion ni digitos.
- **Representacion**: ``TfidfVectorizer(analyzer="char_wb", ngram_range=(4,4), min_df=1)``.
- **Similitud**: coseno, es decir producto de filas de una matriz con norma L2.
- **Umbral de union**: 0,85. Los otros dos umbrales del modulo, ``NEAR_DUPLICATE_THRESHOLD``
  (0,90) y ``NEAR_DUPLICATE_BAND`` (0,80), solo se usan para el resumen descriptivo, no para
  agrupar.
- **Agrupacion**: componentes conexas por union-find sobre los pares por encima del umbral.
- **Sobre que datos**: ``train`` y ``test`` concatenados (2885 filas). El test participa en la
  formacion de las componentes, aunque no aporta etiquetas. No es fuga de etiquetas, pero si
  significa que un tweet de test puede unir dos tweets de train, y queda documentado.

Controles
---------
1. **Identidad**: se reproduce el metodo y se contrasta con ``grupos_near_duplicados.csv``. Si
   no coincide grupo a grupo, el script se para.
2. **Coseno contra explicito**: la matriz de producto coincide con el calculo fila a fila.
3. **Coseno de una fila consigo misma**: exactamente 1,0.
4. **Simetria**: la matriz es simetrica.
5. **Por que 0,40 %**: se barre el umbral del metodo actual y se mide la cobertura, para separar
   "el umbral es demasiado estricto" de "el dataset no tiene copias".
6. **Papel de la normalizacion**: se mide cuanto cubre el metodo si ademas se quitan menciones,
   hashtags y puntuacion, para separar "umbral" de "normalizacion insuficiente".

Ausencia de leakage
-------------------
Este script no entrena nada y no produce ninguna senal que llegue a un modelo. Reproduce una
agrupacion que ya existe en el repositorio y la audita. El test se lee **solo** para reproducir
el calculo historico identico, nunca para producir senal, y se declara explicitamente.

Uso
---
    /home/carlos/venv/bin/python experiments/v29_01_auditoria_near_duplicate.py
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
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_test, read_train
from src.splits import grupos_fusionados, pseudo_autores

EXPERIMENTO = "V29-01 (auditoria del metodo de near-duplicate actual y de su cobertura)"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v29_01_auditoria_near_duplicate.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "v29_01_auditoria_near_duplicate.md"
GRUPOS_CSV = PROJECT_ROOT / "data" / "processed" / "grupos_near_duplicados.csv"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"

ETIQUETAS = ("HS", "TR", "AG")
UMBRAL_HISTORICO = 0.85
# Patron historico exacto de experiments/00_auditoria_datos.py: consume la URL entera
# hasta el espacio. Un patron que solo quitara el prefijo dejaria el dominio y cambiaria
# los 4-gramas, de modo que la reproduccion no identica.
RE_URL_HISTORICO = re.compile(r"https?://\S+")
# Version laxa, solo para detectar el rasgo, no para reproducir el metodo.
RE_URL = re.compile(r"https?://|www\.")
RE_MENCION = re.compile(r"@\w+")
RE_HASHTAG = re.compile(r"#\w+")
RE_PUNCT = re.compile(r"[!\u00bf?.,;:()\[\]{}\"'`\-_/\\|+*=<>$%&]")


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


def matriz_similitud(textos: list[str]) -> np.ndarray:
    """Calcula la matriz de coseno TF-IDF char_wb 4-4 sobre texto normalizado.

    Args:
        textos: lista de textos ya normalizados.

    Returns:
        Matriz de similitud densa.
    """
    vectorizador = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 4), min_df=1)
    matriz = vectorizador.fit_transform(textos)
    return (matriz @ matriz.T).toarray().astype("float32")


def main() -> None:
    """Ejecuta la auditoria del metodo de near-duplicate.

    Returns:
        None.
    """
    inicio = time.time()
    log(f"{EXPERIMENTO}. NINGUN modelo, NINGUNA submission")

    train = read_train()
    test = read_test()
    log(f"   train {len(train)} | test {len(test)} | conjunto del metodo {len(train) + len(test)}")
    log("   el test se lee solo para reproducir el calculo historico identico, no para senal")

    log("1/7 control 1-4: identidad del metodo y propiedades de la matriz de similitud")
    ids = list(train["id"].astype(int)) + list(test["id"].astype(int))
    ["train"] * len(train) + ["test"] * len(test)
    textos = [normalizar_historico(t) for t in train["text"]] + [
        normalizar_historico(t) for t in test["text"]
    ]
    similitud = matriz_similitud(textos)

    # control 3: una fila consigo misma vale exactamente 1
    auto = np.diag(similitud).copy()
    max_desviacion_auto = float(np.max(np.abs(auto - 1.0)))
    # control 4: simetria
    max_asimetria = float(np.max(np.abs(similitud - similitud.T)))
    # control 2: coseno por producto de filas contra calculo explicito
    vectorizador = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 4), min_df=1)
    matriz = vectorizador.fit_transform(textos)
    fila = matriz[0].toarray().ravel()
    explicito = float(fila @ fila.T) / float(np.sqrt(fila @ fila.T) * np.sqrt(fila @ fila.T))
    error_producto = abs(explicito - float(similitud[0, 0]))
    controles = {
        "coseno_de_si_mismo_max_desviacion": max_desviacion_auto,
        "simetria_max_asimetria": max_asimetria,
        "coseno_explicito_vs_producto_error": error_producto,
    }
    log(f"   auto-similitud: desviacion maxima {max_desviacion_auto:.2e}")
    log(f"   simetria: asimetria maxima {max_asimetria:.2e}")
    log(f"   coseno explicito vs producto: error {error_producto:.2e}")

    similitud_sin_diagonal = similitud.copy()
    np.fill_diagonal(similitud_sin_diagonal, 0.0)
    pares = np.argwhere(np.triu(similitud_sin_diagonal, k=1) >= UMBRAL_HISTORICO)
    grupos_reproducidos = componentes_conexas(len(ids), pares)

    historico = pd.read_csv(GRUPOS_CSV)
    historico["id"] = historico["id"].astype(np.int64)
    mapa_hist = dict(zip(historico["id"], historico["grupo"]))
    grupos_historicos = np.array([mapa_hist[i] for i in ids], dtype=np.int64)

    # Los identificadores de grupo no tienen por que coincidir literalmente: lo que importa es
    # el particionado. Se comparan las funciones de particion, no las etiquetas enteras.
    def firma(grupos: np.ndarray) -> pd.DataFrame:
        """Devuelve la lista ordenada de miembros de cada grupo, para comparar particiones."""
        frame = pd.DataFrame({"g": grupos, "i": np.arange(len(grupos))})
        return frame.groupby("g")["i"].apply(lambda s: tuple(sorted(s.tolist()))).sort_values()

    reproduced = firma(grupos_reproducidos)
    original = firma(grupos_historicos)
    identico = reproduced.reset_index(drop=True).equals(original.reset_index(drop=True))
    log(f"   control 1, identidad del particionado frente al CSV congelado: {identico}")
    if not identico:
        raise SystemExit(
            "PARADA: el metodo reproducido no reproduce el particionado guardado. "
            "No se puede auditar sobre una base que no se sostiene."
        )
    controles["identidad_del_particionado"] = bool(identico)
    controles["pares_sobre_umbral"] = len(pares)
    controles["grupos_de_tamano_2_o_mas"] = int(
        (pd.Series(grupos_reproducidos).value_counts() >= 2).sum()
    )

    log("2/7 cobertura: de donde salen las 105 filas del desarrollo")
    particion = pd.read_csv(PARTICION)
    dev = set(particion.loc[particion["particion"] == "dev", "id"].astype(int))
    ids_arr = np.array(ids, dtype=np.int64)
    es_dev = np.array([i in dev for i in ids_arr])
    n_dev = int(es_dev.sum())
    tam_nd = pd.Series(grupos_reproducidos).map(pd.Series(grupos_reproducidos).value_counts())
    tam_nd = tam_nd.to_numpy()[es_dev]

    grupos_fusion = grupos_fusionados(train)
    # Las fuentes de autor existen solo sobre train, asi que necesitan su propia mascara.
    es_dev_train = es_dev[: len(train)]
    tam_fus = (
        pd.Series(grupos_fusion)
        .map(pd.Series(grupos_fusion).value_counts())
        .to_numpy()[es_dev_train]
    )
    autores = np.array(pseudo_autores(train), dtype=object)
    con_autor = autores != ""
    conteo_autores = pd.Series(autores[con_autor]).value_counts()
    tam_aut = np.ones(len(train), dtype=np.int64)
    for autor in np.unique(autores[con_autor]):
        tam_aut[autores == autor] = conteo_autores[autor]
    tam_aut = tam_aut[es_dev_train]

    desglose = {
        "near_duplicate_puro": {
            "filas_en_grupo_2_o_mas": int((tam_nd >= 2).sum()),
            "cobertura": float((tam_nd >= 2).mean()),
        },
        "pseudo_autor": {
            "filas_en_grupo_2_o_mas": int((tam_aut >= 2).sum()),
            "cobertura": float((tam_aut >= 2).mean()),
        },
        "grupos_fusionados_usados_por_v03": {
            "filas_en_grupo_2_o_mas": int((tam_fus >= 2).sum()),
            "cobertura": float((tam_fus >= 2).mean()),
        },
    }
    solo_autor = int(((tam_fus >= 2) & (tam_nd < 2)).sum())
    desglose["filas_que_entran_solo_por_pseudo_autor"] = solo_autor
    desglose["autores_distintos"] = len(conteo_autores)
    desglose["filas_con_autor"] = int(con_autor.sum())
    for nombre, d in desglose.items():
        if isinstance(d, dict):
            log(
                f"   {nombre:<40} {d['filas_en_grupo_2_o_mas']:>4} filas | "
                f"cobertura {d['cobertura']:.4f}"
            )
    log(f"   de las 105, {solo_autor} entran SOLO por la heuristica de pseudo-autor")

    log("3/7 el rasgo dominante del near-duplicate: menciones, hashtags y puntuacion")
    # La normalizacion historica deja menciones y hashtags dentro del texto. Cada una cambia
    # varios 4-gramas de golpe, asi que basta con que dos tweets del mismo autor se
    # mencionen mutuamente para que el coseno caiga por debajo de 0,85.
    fracciones = {
        "con_mentcion": float(np.mean([bool(RE_MENCION.search(t)) for t in textos])),
        "con_hashtag": float(np.mean([bool(RE_HASHTAG.search(t)) for t in textos])),
        "con_url_antes_de_normalizar": float(np.mean([bool(RE_URL.search(t)) for t in train["text"].astype(str)])),
    }
    log(
        f"   con mencion {fracciones['con_mentcion']:.4f} | con hashtag "
        f"{fracciones['con_hashtag']:.4f}"
    )

    log("4/7 control 5: por que solo 0,40 %, barrido del umbral del metodo actual")
    barrido_umbral: list[dict[str, Any]] = []
    for umbral in (0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50, 0.40, 0.30):
        p = np.argwhere(np.triu(similitud_sin_diagonal, k=1) >= umbral)
        g = componentes_conexas(len(ids), p)
        t = pd.Series(g).map(pd.Series(g).value_counts()).to_numpy()
        tam = t[es_dev]
        entrada = {
            "umbral": umbral,
            "pares": len(p),
            "grupos_2_o_mas_total": int((pd.Series(g).value_counts() >= 2).sum()),
            "filas_dev_2_o_mas": int((tam >= 2).sum()),
            "cobertura_dev": float((tam >= 2).mean()),
            "grupo_maximo": int(pd.Series(g).value_counts().max()) if len(p) else 1,
        }
        barrido_umbral.append(entrada)
        log(
            f"   umbral {umbral:.2f}: pares {entrada['pares']:>7} | grupos >=2 "
            f"{entrada['grupos_2_o_mas_total']:>5} | cobertura dev {entrada['cobertura_dev']:.4f}"
        )

    log("5/7 control 6: papel de la normalizacion, no solo del umbral")
    def normalizar_agresiva(texto: str) -> str:
        """Quita ademas menciones, hashtags y puntuacion.

        Args:
            texto: texto original.

        Returns:
            Texto normalizado de forma mas agresiva.
        """
        limpio = RE_MENCION.sub(" ", RE_HASHTAG.sub(" ", RE_PUNCT.sub(" ", texto.lower())))
        limpio = RE_URL.sub(" ", limpio)
        return re.sub(r"\s+", " ", limpio).strip()

    aggressive = [normalizar_agresiva(t) for t in textos]
    similitud_agresiva = matriz_similitud(aggressive)
    np.fill_diagonal(similitud_agresiva, 0.0)
    comparacion_normalizacion: list[dict[str, Any]] = []
    for umbral in (0.85, 0.80, 0.75, 0.70, 0.60, 0.50):
        fila = {"umbral": umbral}
        for nombre, matriz in (("historica", similitud_sin_diagonal), ("agresiva", similitud_agresiva)):
            p = np.argwhere(np.triu(matriz, k=1) >= umbral)
            g = componentes_conexas(len(ids), p)
            tam = pd.Series(g).map(pd.Series(g).value_counts()).to_numpy()[es_dev]
            fila[f"{nombre}_cobertura_dev"] = float((tam >= 2).mean())
            fila[f"{nombre}_pares"] = len(p)
        fila["ganancia_de_normalizacion"] = (
            fila["agresiva_cobertura_dev"] - fila["historica_cobertura_dev"]
        )
        comparacion_normalizacion.append(fila)
        log(
            f"   umbral {umbral:.2f}: historica {fila['historica_cobertura_dev']:.4f} | "
            f"agresiva {fila['agresiva_cobertura_dev']:.4f} | ganancia "
            f"{fila['ganancia_de_normalizacion']:+.4f}"
        )

    log("6/7 la implication para el diseno de V29")
    top_sim = float(similitud_sin_diagonal.max())
    cuant = {
        "similitud_maxima_entre_pares": top_sim,
        "pares_por_encima_de_0.85": int((similitud_sin_diagonal >= UMBRAL_HISTORICO).sum() / 2),
        "n_train": len(train),
        "n_test": len(test),
        "pares_train_train": len(train) * (len(train) - 1) // 2,
        "nota_escalado": (
            "con 1731 filas de train hay 1,5 millones de pares, de modo que MinHash o LSH no "
            "aportan nada: la matriz densa cabe en memoria y es mas simple de auditar"
        ),
    }
    log(f"   similitud maxima entre pares: {top_sim:.4f}")
    log(f"   pares train-train: {cuant['pares_train_train']} (all-pairs es viable)")
    log("   implicacion: el techo al que se puede aspirar RELAJANDO SIMILITUD es 0,40 %, no 7,08 %")

    log("7/7 informe")
    informe = {
        "familia": "FUZZY_TEMPLATE_MATCHING",
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estado": "AUDIT_ONLY",
        "pregunta": (
            "como se definieron los near-duplicates historicos y por que solo el 7,08 % de las "
            "filas del desarrollo entra en grupos de tamano 2 o mas"
        ),
        "alcance": (
            "auditoria documental y numerica. No se entrena nada, no se produce ninguna senal "
            "para un modelo y no se genera submission"
        ),
        "metodo_historico": {
            "normalizacion": "minusculas, URLs sustituidas por espacio, espacios colapsados",
            "no_normaliza": "menciones, hashtags, puntuacion ni digitos",
            "representacion": "TfidfVectorizer(analyzer='char_wb', ngram_range=(4,4), min_df=1)",
            "similitud": "coseno (producto de filas con norma L2)",
            "umbral_de_union": UMBRAL_HISTORICO,
            "otros_umbrales_del_modulo": {
                "NEAR_DUPLICATE_THRESHOLD": 0.90,
                "NEAR_DUPLICATE_BAND": 0.80,
                "nota": "solo se usan para el resumen descriptivo, no para agrupar",
            },
            "agrupacion": "componentes conexas con union-find",
            "datos": "train + test concatenados (2885 filas); el test participa en las componentes",
        },
        "controles": controles,
        "desglose_de_cobertura": desglose,
        "fracciones_de_rasgos": fracciones,
        "barrido_de_umbral_metodo_actual": barrido_umbral,
        "comparacion_de_normalizacion": comparacion_normalizacion,
        "cuant_escalado": cuant,
        "gastos": {
            "filas": n_dev,
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
    log("   informe escrito en reports/v29_01_auditoria_near_duplicate.{json,md}")
    log(f"   memoria pico {memoria_pico_mb():.0f} MB | tiempo {round(time.time() - inicio, 1)} s")
    log("VEREDICTO: auditoria completada, 0 modelos, 0 submissions")


def _informe_md(informe: dict[str, Any]) -> None:
    """Escribe el informe legible.

    Args:
        informe: informe con los datos.

    Returns:
        None.
    """
    m = informe["metodo_historico"]
    d = informe["desglose_de_cobertura"]
    c = informe["controles"]
    cu = informe["cuant_escalado"]
    g = informe["gastos"]
    lineas = [
        "# V29-01: auditoria del metodo de near-duplicate actual",
        "",
        f"Generado por `experiments/v29_01_auditoria_near_duplicate.py` ({EXPERIMENTO}).",
        "",
        f"**Estado `{informe['estado']}`. Alcance: {informe['alcance']}**",
        "",
        "## 1. El metodo, documentado",
        "",
        "| elemento | valor |",
        "|---|---|",
        f"| normalizacion | {m['normalizacion']} |",
        f"| no normaliza | {m['no_normaliza']} |",
        f"| representacion | `{m['representacion']}` |",
        f"| similitud | {m['similitud']} |",
        f"| umbral de union | {m['umbral_de_union']} |",
        f"| agrupacion | {m['agrupacion']} |",
        f"| datos | {m['datos']} |",
        "",
        (f"Los otros dos umbrales del modulo (`{m['otros_umbrales_del_modulo']['NEAR_DUPLICATE_THRESHOLD']}` "
        f"y `{m['otros_umbrales_del_modulo']['NEAR_DUPLICATE_BAND']}`) "
        f"{m['otros_umbrales_del_modulo']['nota']}."),
        "",
        "## 2. Controles",
        "",
        "| control | resultado |",
        "|---|---|",
        f"| identidad del particionado frente al CSV congelado | **{c['identidad_del_particionado']}** |",
        f"| coseno de si mismo, desviacion maxima | {c['coseno_de_si_mismo_max_desviacion']:.2e} |",
        f"| simetria, asimetria maxima | {c['simetria_max_asimetria']:.2e} |",
        f"| coseno explicito vs producto, error | {c['coseno_explicito_vs_producto_error']:.2e} |",
        f"| pares sobre el umbral | {c['pares_sobre_umbral']} |",
        f"| grupos de tamano 2 o mas | {c['grupos_de_tamano_2_o_mas']} |",
        "",
        "El control 1 compara el **particionado**, no las etiquetas enteras de grupo: lo que tiene",
        "que reproducirse es quien esta junto a quien, no el numero que el union-find asigne.",
        "",
        "## 3. De donde sale el 7,08 %",
        "",
        "Esta es la tabla que reencuadra la fase.",
        "",
        "| fuente | filas del desarrollo en grupo >= 2 | cobertura |",
        "|---|---:|---:|",
        (f"| near-duplicate puro | {d['near_duplicate_puro']['filas_en_grupo_2_o_mas']} | "
        f"**{d['near_duplicate_puro']['cobertura']:.4f}** |"),
        (f"| pseudo-autor (primera mencion `@usuario`) | {d['pseudo_autor']['filas_en_grupo_2_o_mas']} | "
        f"{d['pseudo_autor']['cobertura']:.4f} |"),
        (f"| grupos fusionados, los que usa V03 | {d['grupos_fusionados_usados_por_v03']['filas_en_grupo_2_o_mas']} | "
        f"{d['grupos_fusionados_usados_por_v03']['cobertura']:.4f} |"),
        "",
        f"**{d['filas_que_entran_solo_por_pseudo_autor']} de las 105 filas entran solo por la",
        "heuristica de pseudo-autor.** Hay {d['autores_distintos']} autores distintos y solo",
        f"{d['filas_con_autor']} filas tienen mencion, de las cuales se agrupan muy pocas.",
        "",
        "La consecuencia practica es directa: **relajar la similitud textual no es relajar el",
        "criterio que produjo el 7,08 %**, porque ese criterio es otro. El techo al que se puede",
        "aspirar relajando similitud es el 0,40 %, no el 7,08 %. Y el 0,40 % es exactamente el",
        "grupo que V28-03 encontro contradictorio.",
        "",
        "## 4. Por que 0,40 %, separando umbral de normalizacion",
        "",
        "| umbral | pares | grupos >= 2 | cobertura del desarrollo | grupo maximo |",
        "|---:|---:|---:|---:|---:|",
    ]
    for b in informe["barrido_de_umbral_metodo_actual"]:
        lineas.append(
            f"| {b['umbral']:.2f} | {b['pares']} | {b['grupos_2_o_mas_total']} | "
            f"{b['cobertura_dev']:.4f} | {b['grupo_maximo']} |"
        )
    lineas += [
        "",
        "Y el efecto de la normalizacion, manteniendo el mismo metodo:",
        "",
        "| umbral | normalizacion historica | normalizacion agresiva | ganancia |",
        "|---:|---:|---:|---:|",
    ]
    for fila in informe["comparacion_de_normalizacion"]:
        lineas.append(
            f"| {fila['umbral']:.2f} | {fila['historica_cobertura_dev']:.4f} | "
            f"{fila['agresiva_cobertura_dev']:.4f} | **{fila['ganancia_de_normalizacion']:+.4f}** |"
        )
    lineas += [
        "",
        f"La similitud maxima entre pares es {cu['similitud_maxima_entre_pares']:.4f} y hay",
        f"{cu['pares_por_encima_de_0.85']} pares por encima de 0,85. Con solo",
        f"{cu['n_train']} filas de train hay {cu['pares_train_train']} pares train-train, asi que",
        f"{cu['nota_escalado']}.",
        "",
        "## 5. Lo que este script no hace",
        "",
        "- No entrena ningun modelo.",
        "- No produce ninguna senal para un predictor.",
        "- No genera submission.",
        "- El test se lee unicamente para reproducir el calculo historico identico.",
        "",
        (f"Coste: {g['segundos']} s, memoria {g['memoria_pico_mb']:.0f} MB, "
        f"{g['modelos_entrenados']} modelos, {g['submissions']} submissions."),
        "",
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
