"""09_diacronia.py
Fecha: 2026-09-09
Descripcion: Evolucion diacronica del corpus de la Corte Constitucional
    1992-2021 (modulo M6) en tres ejes: (a) distribucion de areas
    tematicas por ano (etiquetas M4), con figura y tabla CSV; (b)
    diacronia lexica por log-odds ratio de lemas entre periodos
    consecutivos (1992-2001 -> 2002-2011 -> 2012-2021) sobre texto_limpio
    leido en streaming con pyarrow, con las 306 stopwords gramaticales
    de E1 reconstruidas localmente; (c) diversidad lexica anual como
    media por ano del MTLD por documento calculado en E1.
Estado del arte aplicado: Hamilton et al. (2016), diacronia lexica por
    cambio de uso relativo entre periodos; log-odds ratio binario con
    suavizado (add 0.5) para comparar vocabularios de corpus grandes sin
    sesgo del tamaño de cada periodo.
"""

import json
import random
import re
import sys
import time
import traceback
import unicodedata
from collections import Counter
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

CONFIG = {
    "semilla": 42,
    "ruta_corpus": "data/corpus_sentencias.parquet",
    "ruta_areas": "data/etiquetas_areas.parquet",
    "ruta_mtld": (
        "../PRIMERA ENTREGA/resultados/eda_extendido/"
        "lemas_stems_pos/mtld_por_doc.parquet"
    ),
    "ruta_json": "resultados/diacronia.json",
    "ruta_csv_areas": "resultados/diacronia_areas.csv",
    "ruta_csv_logodds": "resultados/diacronia_logodds.csv",
    "fig_areas": "figuras/fig_areas_evolucion.png",
    "fig_logodds": "figuras/fig_logodds_lexico.png",
    "fig_mtld": "figuras/fig_mtld_anual.png",
    "periodos": [(1992, 2001), (2002, 2011), (2012, 2021)],
    "top_logodds": 15,
    "min_frecuencia_lema": 50,
    "n_stopwords_esperado": 306,
}

PLACEHOLDER_ENIE = "\u0001"
RE_PARTICION = re.compile(r"ano=(\d{4})")


def normalizar(texto, con_digitos):
    """Normaliza un texto igual que el notebook E1 (minusculas, sin tildes).

    Pasos: minusculas; proteccion de la 'n' con virgulilla antes de
    descomponer en NFD; eliminacion de marcas de acentuacion (rango
    Unicode \\u0300-\\u036f); restauracion de la virgulilla; filtro de
    caracteres (letras a-z y 'n' con virgulilla, con o sin digitos) y
    colapso de espacios.

    Args:
        texto: Cadena original.
        con_digitos: True conserva digitos; False los elimina.

    Returns:
        Cadena normalizada sin espacios multiples ni extremos.
    """
    s = texto.lower().replace("\u00f1", PLACEHOLDER_ENIE)
    s = unicodedata.normalize("NFD", s)
    s = re.sub(r"[\u0300-\u036f]", "", s).replace(PLACEHOLDER_ENIE, "\u00f1")
    patron = r"[^a-z\u00f10-9\s]" if con_digitos else r"[^a-z\u00f1\s]"
    s = re.sub(patron, " ", s)
    return re.sub(r"\s+", " ", s).strip()


def construir_stopwords():
    """Construye el conjunto de stopwords normalizado de E1 (306 formas).

    Reconstruye localmente el conjunto de stopwords gramaticales en
    espanol de NLTK con la misma normalizacion del notebook E1 (celda 3):
    minusculas, NFD sin tildes, solo letras [a-z] y enie.

    Returns:
        Set con las 306 stopwords normalizadas.

    Raises:
        AssertionError: Si el conjunto no tiene exactamente 306 formas.
    """
    try:
        from nltk.corpus import stopwords

        stopwords.words("spanish")
    except (ImportError, LookupError):
        import nltk

        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords

    conjunto = set()
    for palabra in stopwords.words("spanish"):
        normalizada = normalizar(palabra, con_digitos=False)
        if re.fullmatch(r"[a-z\u00f1]+", normalizada):
            conjunto.add(normalizada)
    assert len(conjunto) == CONFIG["n_stopwords_esperado"], (
        f"Conjunto inesperado de stopwords: {len(conjunto)}"
    )
    return conjunto


def etiqueta_periodo(ano):
    """Devuelve la etiqueta del periodo que contiene un ano.

    Args:
        ano: Ano de emision.

    Returns:
        Cadena "AAAA-AAAA" del periodo.
    """
    for inicio, fin in CONFIG["periodos"]:
        if inicio <= ano <= fin:
            return f"{inicio}-{fin}"
    return "fuera_de_rango"


def analizar_areas():
    """Analiza la evolucion anual de areas tematicas (eje a).

    Returns:
        Tupla (tabla_areas, insight) donde tabla_areas es un DataFrame
        (indice ano, columnas areas, valores % de documentos) e insight
        es un dict con las areas de mayor crecimiento y caida entre el
        primer y el ultimo trienio.
    """
    df = pq.read_table(
        CONFIG["ruta_areas"], columns=["ano", "area_primaria"]
    ).to_pandas()
    tabla = (
        df.groupby("ano")["area_primaria"]
        .value_counts(normalize=True)
        .mul(100.0)
        .unstack(fill_value=0.0)
        .sort_index()
    )
    tabla = tabla.reindex(sorted(tabla.columns), axis=1)
    tabla.to_csv(CONFIG["ruta_csv_areas"], encoding="utf-8")
    print("CSV guardado:", CONFIG["ruta_csv_areas"])
    n_anos = len(tabla)
    k = min(3, n_anos)
    primer_trienio = tabla.iloc[:k].mean()
    ultimo_trienio = tabla.iloc[-k:].mean()
    delta = (ultimo_trienio - primer_trienio).sort_values()
    crecen = delta.tail(3).sort_values(ascending=False)
    decaen = delta.head(3)
    insight = {
        "primer_trienio": f"{int(tabla.index[0])}-{int(tabla.index[k - 1])}",
        "ultimo_trienio": f"{int(tabla.index[-k])}-{int(tabla.index[-1])}",
        "areas_mayor_crecimiento_pp": [
            {"area": area, "delta_pp": round(float(valor), 2)}
            for area, valor in crecen.items()
        ],
        "areas_mayor_caida_pp": [
            {"area": area, "delta_pp": round(float(valor), 2)}
            for area, valor in decaen.items()
        ],
    }
    print(
        "Areas: crecen",
        ", ".join(f"{a} (+{v:.1f}pp)" for a, v in crecen.items()),
        "| decaen",
        ", ".join(f"{a} ({v:.1f}pp)" for a, v in decaen.items()),
    )
    return tabla, insight


def contar_lemas_periodo(stopwords):
    """Cuenta lemas por periodo leyendo texto_limpio en streaming.

    Recorre el parquet (particionado Hive por tipo/ano) por fragmentos
    y row-groups con pyarrow.dataset; el ano se recupera de la ruta de
    la particion (columna virtual, no fisica). Tokeniza con split()
    sobre texto_limpio (ya normalizado sin digitos), descarta stopwords
    y acumula un Counter por periodo de emision. Por eficiencia cuenta
    primero los tokens unicos de cada documento (Counter del split) y
    luego descarta las stopwords del vocabulario unico, en lugar de
    filtrar token por token.

    Args:
        stopwords: Set de stopwords normalizadas.

    Returns:
        Dict etiqueta_periodo -> Counter de lemas.
    """
    contadores = {
        f"{inicio}-{fin}": Counter() for inicio, fin in CONFIG["periodos"]
    }
    n_docs = 0
    dataset = ds.dataset(CONFIG["ruta_corpus"], format="parquet")
    for frag in dataset.get_fragments():
        coincidencia = RE_PARTICION.search(frag.path)
        if coincidencia is None:
            raise ValueError(f"Ruta sin particion ano=: {frag.path}")
        ano = int(coincidencia.group(1))
        etiqueta = etiqueta_periodo(ano)
        if etiqueta not in contadores:
            continue
        pf = pq.ParquetFile(frag.path)
        for rg in range(pf.num_row_groups):
            tabla = pf.read_row_group(rg, columns=["texto_limpio"])
            for texto in tabla.column("texto_limpio").to_pylist():
                contador_doc = Counter(texto.split())
                for stop in contador_doc.keys() & stopwords:
                    del contador_doc[stop]
                contadores[etiqueta].update(contador_doc)
                n_docs += 1
    print(f"Documentos leidos en streaming: {n_docs}")
    return contadores


def log_odds_ratio(counter_base, counter_objetivo):
    """Calcula el log-odds ratio con suavizado entre dos periodos.

    Implementa el log-odds ratio binario con suavizado add-0.5: para
    cada lema presente en alguno de los dos periodos con frecuencia
    combinada suficiente,
    log_odds = log((f_obj + 0.5) / (N_obj - f_obj + 0.5))
             - log((f_base + 0.5) / (N_base - f_base + 0.5)).

    Args:
        counter_base: Counter del periodo anterior (base).
        counter_objetivo: Counter del periodo posterior (objetivo).

    Returns:
        DataFrame con columnas lema, freq_base, freq_objetivo y
        log_odds (positivo = sube en el periodo objetivo).
    """
    n_base = sum(counter_base.values())
    n_obj = sum(counter_objetivo.values())
    filas = []
    for lema in set(counter_base) | set(counter_objetivo):
        f_base = counter_base[lema]
        f_obj = counter_objetivo[lema]
        if f_base + f_obj < CONFIG["min_frecuencia_lema"]:
            continue
        odds_base = (f_base + 0.5) / (n_base - f_base + 0.5)
        odds_obj = (f_obj + 0.5) / (n_obj - f_obj + 0.5)
        filas.append(
            {
                "lema": lema,
                "freq_base": f_base,
                "freq_objetivo": f_obj,
                "log_odds": float(np.log(odds_obj) - np.log(odds_base)),
            }
        )
    return pd.DataFrame(filas)


def analizar_lexico(stopwords):
    """Analiza la diacronia lexica por log-odds entre periodos (eje b).

    Args:
        stopwords: Set de stopwords normalizadas.

    Returns:
        Tupla (tabla_logodds, transiciones) donde tabla_logodds es el
        DataFrame completo por transicion y transiciones es un dict con
        los top sube/baja por transicion.
    """
    contadores = contar_lemas_periodo(stopwords)
    for etiqueta, contador in contadores.items():
        print(
            f"  {etiqueta}: {sum(contador.values()):,} tokens, "
            f"{len(contador):,} lemas"
        )
    pares = [
        (
            contadores[f"{a}-{b}"],
            contadores[f"{c}-{d}"],
            f"{a}-{b}->{c}-{d}",
        )
        for (a, b), (c, d) in zip(CONFIG["periodos"], CONFIG["periodos"][1:])
    ]
    tablas = []
    transiciones = {}
    for base, objetivo, etiqueta in pares:
        tabla = log_odds_ratio(base, objetivo)
        tabla["transicion"] = etiqueta
        tablas.append(tabla)
        sube = tabla.sort_values("log_odds", ascending=False).head(
            CONFIG["top_logodds"]
        )
        baja = tabla.sort_values("log_odds").head(CONFIG["top_logodds"])
        transiciones[etiqueta] = {
            "top_sube": sube[["lema", "log_odds"]].round(3).to_dict("records"),
            "top_baja": baja[["lema", "log_odds"]].round(3).to_dict("records"),
        }
        print(
            f"Transicion {etiqueta}: top sube "
            f"{sube['lema'].iloc[0]} (+{sube['log_odds'].iloc[0]:.2f}) | "
            f"top baja {baja['lema'].iloc[0]} "
            f"({baja['log_odds'].iloc[0]:.2f})"
        )
    tabla_completa = pd.concat(tablas, ignore_index=True).sort_values(
        ["transicion", "log_odds"], ascending=[True, False]
    )
    tabla_completa.to_csv(
        CONFIG["ruta_csv_logodds"], index=False, encoding="utf-8"
    )
    print("CSV guardado:", CONFIG["ruta_csv_logodds"])
    return tabla_completa, transiciones


def analizar_mtld():
    """Analiza la diversidad lexica anual via MTLD por documento (eje c).

    Returns:
        DataFrame con indice ano y columnas mtld_media, mtld_std y
        n_docs.
    """
    df = pq.read_table(CONFIG["ruta_mtld"]).to_pandas()
    return (
        df.groupby("ano")["mtld"]
        .agg(["mean", "std", "count"])
        .rename(
            columns={"mean": "mtld_media", "std": "mtld_std", "count": "n_docs"}
        )
        .sort_index()
    )


def figura_areas(tabla):
    """Genera la figura de evolucion anual de areas (apilada al 100%).

    Args:
        tabla: DataFrame (indice ano, columnas areas, valores %).
    """
    _fig, eje = plt.subplots(figsize=(14, 7))
    tabla.plot(kind="area", stacked=True, ax=eje, cmap="tab20", alpha=0.85)
    eje.set_title("Evolucion de areas tematicas por ano (% de documentos)")
    eje.set_xlabel("Ano")
    eje.set_ylabel("% de documentos")
    eje.legend(
        loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, title="Area"
    )
    eje.set_xlim(tabla.index[0], tabla.index[-1])
    plt.tight_layout()
    plt.savefig(CONFIG["fig_areas"], dpi=150, bbox_inches="tight")
    plt.close()
    print("Figura guardada:", CONFIG["fig_areas"])


def figura_logodds(transiciones):
    """Genera la figura de top lemas sube/baja por transicion.

    Args:
        transiciones: Dict etiqueta_transicion -> top_sube/top_baja.
    """
    n_trans = len(transiciones)
    fig, ejes = plt.subplots(n_trans, 2, figsize=(14, 5 * n_trans))
    ejes = np.atleast_2d(ejes)
    for fila, (etiqueta, datos) in enumerate(transiciones.items()):
        for col, clave in enumerate(("top_sube", "top_baja")):
            eje = ejes[fila, col]
            registros = list(reversed(datos[clave]))
            lemas = [r["lema"] for r in registros]
            valores = [r["log_odds"] for r in registros]
            eje.barh(
                lemas,
                valores,
                color="#2f5d8a" if col == 0 else "#c0504d",
            )
            eje.set_title(f"{etiqueta} ({clave})", fontsize=10)
            eje.set_xlabel("log-odds ratio")
            eje.tick_params(axis="y", labelsize=8)
    fig.suptitle("Diacronia lexica: log-odds ratio de lemas por transicion")
    plt.tight_layout()
    plt.savefig(CONFIG["fig_logodds"], dpi=150, bbox_inches="tight")
    plt.close()
    print("Figura guardada:", CONFIG["fig_logodds"])


def figura_mtld(anual):
    """Genera la figura de MTLD medio anual con error estandar.

    Args:
        anual: DataFrame con mtld_media, mtld_std y n_docs por ano.
    """
    plt.figure(figsize=(12, 5))
    plt.plot(anual.index, anual["mtld_media"], marker="o", color="#2f5d8a")
    error = anual["mtld_std"] / np.sqrt(anual["n_docs"])
    plt.fill_between(
        anual.index,
        anual["mtld_media"] - error,
        anual["mtld_media"] + error,
        alpha=0.2,
        color="#2f5d8a",
    )
    plt.title("Diversidad lexica anual: MTLD medio por documento")
    plt.xlabel("Ano")
    plt.ylabel("MTLD (media por documento)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(CONFIG["fig_mtld"], dpi=150, bbox_inches="tight")
    plt.close()
    print("Figura guardada:", CONFIG["fig_mtld"])


def guardar_json(tabla_areas, insight_areas, transiciones, anual, tiempos):
    """Escribe el JSON de resultados de diacronia.

    Args:
        tabla_areas: DataFrame anual de areas (eje a).
        insight_areas: Dict de analizar_areas.
        transiciones: Dict de analizar_lexico.
        anual: DataFrame de analizar_mtld.
        tiempos: Dict con tiempos por seccion en segundos.
    """
    tabla_anual = {
        str(ano): {area: round(float(valor), 3) for area, valor in fila.items()}
        for ano, fila in tabla_areas.iterrows()
    }
    mtld_por_ano = {
        str(ano): {
            "mtld_media": round(float(fila.mtld_media), 2),
            "n_docs": int(fila.n_docs),
        }
        for ano, fila in anual.iterrows()
    }
    tendencia = float(
        np.polyfit(anual.index.astype(float), anual["mtld_media"], 1)[0]
    )
    resultado = {
        "fecha": datetime.now(timezone.utc).date().isoformat(),
        "areas": {
            "insight": insight_areas,
            "tabla_anual": tabla_anual,
        },
        "lexico": {
            "min_frecuencia_lema": CONFIG["min_frecuencia_lema"],
            "transiciones": transiciones,
        },
        "mtld": {
            "por_ano": mtld_por_ano,
            "tendencia_mtld_por_ano": round(tendencia, 3),
        },
        "tiempos_seg": tiempos,
    }
    with open(CONFIG["ruta_json"], "w", encoding="utf-8") as fh:
        json.dump(resultado, fh, ensure_ascii=False, indent=2)
    print("JSON guardado:", CONFIG["ruta_json"])


def main():
    """Orquesta el modulo M6: areas, lexico y MTLD con sus salidas."""
    random.seed(CONFIG["semilla"])
    np.random.seed(CONFIG["semilla"])
    t0 = time.time()
    stopwords = construir_stopwords()
    print(f"Stopwords reconstruidas: {len(stopwords)} formas")
    t_stop = time.time()
    tabla_areas, insight_areas = analizar_areas()
    figura_areas(tabla_areas)
    t_areas = time.time()
    _tabla_logodds, transiciones = analizar_lexico(stopwords)
    figura_logodds(transiciones)
    t_lexico = time.time()
    anual = analizar_mtld()
    figura_mtld(anual)
    t_mtld = time.time()
    guardar_json(
        tabla_areas,
        insight_areas,
        transiciones,
        anual,
        {
            "stopwords": round(t_stop - t0, 1),
            "areas": round(t_areas - t_stop, 1),
            "lexico": round(t_lexico - t_areas, 1),
            "mtld": round(t_mtld - t_lexico, 1),
            "total": round(t_mtld - t0, 1),
        },
    )
    print()
    print("RESUMEN M6 - DIACRONIA")
    print("-" * 56)
    crece = insight_areas["areas_mayor_crecimiento_pp"][0]
    decae = insight_areas["areas_mayor_caida_pp"][0]
    print(
        f"Areas: crece {crece['area']} (+{crece['delta_pp']}pp) | "
        f"decae {decae['area']} ({decae['delta_pp']}pp)"
    )
    for etiqueta, datos in transiciones.items():
        print(
            f"{etiqueta}: sube {datos['top_sube'][0]['lema']} | "
            f"baja {datos['top_baja'][0]['lema']}"
        )
    print(
        f"MTLD: {anual['mtld_media'].iloc[0]:.1f} ({anual.index[0]}) -> "
        f"{anual['mtld_media'].iloc[-1]:.1f} ({anual.index[-1]})"
    )
    print(f"Tiempo total: {t_mtld - t0:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - salida con traceback completo por convencion
        traceback.print_exc()
        sys.exit(1)
