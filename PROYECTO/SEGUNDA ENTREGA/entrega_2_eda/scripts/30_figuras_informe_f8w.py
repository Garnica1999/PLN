"""30_figuras_informe_f8w.py

Regenera las figuras del informe de la Entrega 2 para que sean legibles al
tamano de pagina del informe (ancho de linea de 6,26 in). Cada figura se
disena a su tamano de despliegue con fuentes de 9-10 pt y dpi 220, en lugar
de disenarse a ~14 in y reducirse al imprimir (lo que dejaba el texto interno
en ~4,5 pt). NO modifica el notebook sellado: lee los artefactos de
resultados/ y respalda las figuras anteriores en figuras/backup_f8w/.

Los textos visibles se reescriben con acentos, igual que el informe, y se
conservan titulos, etiquetas y colores originales de cada figura. Las cifras
se verifican contra resultados/metricas_entrega_ii.json antes de guardar.

Uso:
    /home/carlos/venv/bin/python scripts/30_figuras_informe_f8w.py
"""

from __future__ import annotations

import json
import re
import shutil
import unicodedata
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd
import pyarrow.parquet as pq_arrow

BASE = Path(__file__).resolve().parents[1]          # entrega_2_eda/
DATOS = BASE / "datos"
RES = BASE / "resultados"
EDA = RES / "eda_extendido"
LEMAS = EDA / "lemas_stems_pos"
FIGS = BASE / "figuras"
BACKUP = FIGS / "backup_f8w"

ANCHO = 6.26                                        # in: ancho de linea del informe
AZUL, ROJO, VERDE, NARANJA = "#2f5d8a", "#c0504d", "#3f8f5f", "#e8a33d"
COLORES_TIPO = {"T": AZUL, "C": ROJO, "SU": VERDE}
TAB_TRES = ["tab:blue", "tab:orange", "tab:green"]

plt.rcParams.update({
    "font.size": 9.0,
    "axes.titlesize": 10.0,
    "axes.labelsize": 9.0,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "figure.dpi": 220,
    "savefig.dpi": 220,
})

BACKUP.mkdir(exist_ok=True)

STOPW_DOMINIO_JURIDICO = [
    "sentencia", "sentencias", "corte", "constitucional", "constitucion", "derecho",
    "derechos", "justicia", "tribunal", "magistrado", "magistrados", "ponente", "demanda",
    "demandas", "expediente", "expedientes", "colombia", "bogota", "republica", "accion",
    "acciones", "tutela", "tutelas", "articulo", "articulos", "ley", "leyes", "norma",
    "normas", "codigo", "auto", "autos", "proceso", "procesos", "providencia",
    "providencias", "fallo", "fallos", "sala", "plena", "parte", "partes", "juez",
    "juzgado", "despacho", "relatoria", "secretaria", "actuacion", "actuaciones", "cargo",
    "cargos", "asunto", "asuntos", "caso", "casos", "tema", "temas", "prueba", "pruebas",
    "peticion", "solicitud", "solicitudes", "revision", "instancia", "instancias",
]


def guardar(fig, nombre):
    """Respalda la version anterior (una sola vez) y guarda la nueva figura."""
    destino = FIGS / nombre
    respaldo = BACKUP / nombre
    if destino.exists() and not respaldo.exists():
        shutil.copy2(destino, respaldo)
    fig.savefig(destino, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("  guardada:", nombre)


def miles(valor):
    """Formatea un entero con separador de miles espanol (punto)."""
    return "{:,}".format(int(valor)).replace(",", ".")


def num(valor, dec=1):
    """Formatea un decimal con coma decimal espanola."""
    return ("{:.%df}" % dec).format(valor).replace(".", ",")


def normalizar(texto):
    """Normaliza como la seccion 2: minusculas, sin diacriticos, ene protegida."""
    s = texto.lower().replace("\u00f1", "\u0001")
    s = unicodedata.normalize("NFD", s)
    s = re.sub(r"[\u0300-\u036f]", "", s).replace("\u0001", "\u00f1")
    s = re.sub(r"[^a-z\u00f1\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def stopwords_gramaticales():
    """Stopwords de NLTK normalizadas (306 formas efectivas, como la seccion 9.0)."""
    from nltk.corpus import stopwords
    conjunto = set()
    for palabra in stopwords.words("spanish"):
        forma = normalizar(palabra)
        if re.fullmatch(r"[a-z\u00f1]+", forma):
            conjunto.add(forma)
    return conjunto


def cargar_o_calcular_frecuencias(metadatos):
    """Devuelve (contador de contenido, contadores por anio).

    Reutiliza la cache resultados/eda_extendido/frecuencias_contenido*.csv;
    si no existe, recorre el corpus en streaming replicando la pasada de la
    seccion 3 del notebook (tokens de texto_limpio sin stopwords) y verifica
    los totales sellados antes de guardar.
    """
    ruta_global = EDA / "frecuencias_contenido.csv"
    ruta_anio = EDA / "frecuencias_contenido_por_anio.csv"
    if ruta_global.exists() and ruta_anio.exists():
        tabla = pd.read_csv(ruta_global)
        contador_sin = Counter(dict(zip(tabla["termino"], tabla["freq"].astype(int))))
        tabla_anio = pd.read_csv(ruta_anio)
        por_anio = {}
        for anio, grupo in tabla_anio.groupby("anio"):
            por_anio[int(anio)] = Counter(dict(zip(grupo["termino"], grupo["freq"].astype(int))))
        print("Frecuencias de contenido cargadas de cache.")
        return contador_sin, por_anio

    print("Recorriendo el corpus para recalcular frecuencias de contenido...")
    stopw = stopwords_gramaticales()
    assert len(stopw) == 306, "stopwords normalizadas: {}".format(len(stopw))
    contador_sin = Counter()
    por_anio = {}
    n_limpios, n_stop = 0, 0
    lector = pq_arrow.ParquetFile(DATOS / "corpus_medido.parquet")
    for lote in lector.iter_batches(batch_size=200, columns=["ano", "texto_limpio"]):
        for ano, limpio in zip(lote.column("ano").to_pylist(),
                               lote.column("texto_limpio").to_pylist()):
            tokens = limpio.split()
            n_limpios += len(tokens)
            contador_anio = por_anio.get(ano)
            if contador_anio is None:
                contador_anio = Counter()
                por_anio[ano] = contador_anio
            for token in tokens:
                if token in stopw:
                    n_stop += 1
                else:
                    contador_sin[token] += 1
                    contador_anio[token] += 1

    assert n_limpios == 250_012_101, "tokens limpios: {}".format(n_limpios)
    assert n_stop == 121_205_437, "stopwords: {}".format(n_stop)
    assert len(contador_sin) == 224_202, "tipos: {}".format(len(contador_sin))
    assert sum(contador_sin.values()) == 128_806_664, "tokens de contenido"
    hapax = sum(1 for v in contador_sin.values() if v == 1)
    assert hapax == 74_069, "hapax: {}".format(hapax)

    pd.DataFrame(contador_sin.most_common(), columns=["termino", "freq"]).to_csv(
        ruta_global, index=False, encoding="utf-8")
    filas = [{"anio": anio, "termino": termino, "freq": frec}
             for anio, contador_anio in sorted(por_anio.items())
             for termino, frec in contador_anio.most_common()]
    pd.DataFrame(filas).to_csv(ruta_anio, index=False, encoding="utf-8")
    print("Recuento verificado y guardado en cache.")
    return contador_sin, por_anio


def cargar_o_calcular_cola_tfidf():
    """Devuelve la masa acumulada de frecuencia documental del TF-IDF.

    Reutiliza resultados/eda_extendido/tfidf_cola_larga.csv; si no existe,
    reajusta el TfidfVectorizer con la configuracion documentada (min_df=3,
    100.000 rasgos, n-gramas 1-2, sublineal) sobre los lemas por documento.
    """
    ruta = EDA / "tfidf_cola_larga.csv"
    if ruta.exists():
        tabla = pd.read_csv(ruta)
        print("Cola larga TF-IDF cargada de cache.")
        return tabla["masa_acumulada"].to_numpy()

    from sklearn.feature_extraction.text import TfidfVectorizer
    print("Reajustando TF-IDF sobre lemas (unos 3-4 minutos)...")
    lemas = pd.read_parquet(DATOS / "lemas_por_doc.parquet", columns=["id", "lemas_texto"])
    lemas["id"] = lemas["id"].astype(str)
    lemas = lemas.sort_values("id", key=lambda s: s.astype(int)).reset_index(drop=True)
    textos = lemas["lemas_texto"].tolist()
    del lemas
    vectorizador = TfidfVectorizer(
        min_df=3, max_features=100_000, ngram_range=(1, 2),
        sublinear_tf=True, dtype=np.float32)
    matriz = vectorizador.fit_transform(textos)
    assert matriz.shape == (23_742, 100_000), "shape TF-IDF: {}".format(matriz.shape)
    assert matriz.nnz == 103_045_759, "nnz TF-IDF: {}".format(matriz.nnz)
    df_terminos = np.bincount(matriz.indices, minlength=matriz.shape[1])
    masa = np.cumsum(np.sort(df_terminos)[::-1]) / df_terminos.sum()
    pd.DataFrame({"rango": np.arange(1, len(masa) + 1),
                  "masa_acumulada": masa}).to_csv(ruta, index=False, encoding="utf-8")
    print("TF-IDF verificado y guardado en cache.")
    return masa


def figura_longitudes(metadatos):
    longitudes = metadatos["n_palabras_clean"].to_numpy()
    mediana, media = float(np.median(longitudes)), float(np.mean(longitudes))
    assert mediana == 7158 and abs(media - 10530.37) < 0.01
    fig, ejes = plt.subplots(1, 2, figsize=(ANCHO, 2.5))
    ejes[0].hist(longitudes, bins=80, color=AZUL, edgecolor="white")
    ejes[0].axvline(mediana, color=ROJO, linestyle="--", linewidth=1.6,
                    label="Mediana = {}".format(miles(mediana)))
    ejes[0].axvline(media, color=NARANJA, linestyle="--", linewidth=1.6,
                    label="Media = {}".format(miles(media)))
    ejes[0].xaxis.set_major_locator(MaxNLocator(5))
    ejes[0].tick_params(axis="x", rotation=30, labelsize=8)
    ejes[0].set_title("Distribución de longitudes por documento")
    ejes[0].set_xlabel("Palabras limpias por documento")
    ejes[0].set_ylabel("Número de documentos")
    ejes[0].legend()
    datos_tipo = [metadatos.loc[metadatos["tipo"] == t, "n_palabras_clean"].to_numpy()
                  for t in ["T", "C", "SU"]]
    cajas = ejes[1].boxplot(datos_tipo, tick_labels=["T", "C", "SU"],
                            showfliers=False, patch_artist=True)
    for caja, color in zip(cajas["boxes"], [AZUL, ROJO, VERDE]):
        caja.set_facecolor(color)
        caja.set_alpha(0.7)
    ejes[1].set_title("Longitud por tipo de providencia")
    ejes[1].set_xlabel("Tipo")
    ejes[1].set_ylabel("Palabras limpias por documento")
    fig.suptitle("Longitudes de los documentos (23.742 sentencias)")
    fig.tight_layout()
    guardar(fig, "fig_longitudes_documentos.png")


def figura_tipos(metadatos):
    conteo = metadatos["tipo"].value_counts().reindex(["T", "C", "SU"]).fillna(0).astype(int)
    assert conteo["T"] == 17099 and conteo["C"] == 6246 and conteo["SU"] == 397
    porcentaje = 100.0 * conteo / len(metadatos)
    tabla = (metadatos.groupby(["annio", "tipo"]).size().unstack(fill_value=0)
             .reindex(columns=["T", "C", "SU"], fill_value=0))
    fig, ejes = plt.subplots(1, 2, figsize=(ANCHO, 2.4))
    ejes[0].bar(conteo.index, conteo.values, color=[COLORES_TIPO[t] for t in conteo.index])
    for pos, (tipo_bar, valor) in enumerate(conteo.items()):
        ejes[0].text(pos, valor, "{}\n({} %)".format(miles(valor), num(porcentaje[tipo_bar])),
                     ha="center", va="bottom", fontsize=8.5)
    ejes[0].set_title("Documentos por tipo de providencia", fontsize=9.5)
    ejes[0].set_xlabel("Tipo")
    ejes[0].set_ylabel("Número de documentos")
    ejes[0].set_ylim(0, conteo.max() * 1.22)
    fondo = np.zeros(len(tabla))
    for tipo_bar in ["T", "C", "SU"]:
        valores = tabla[tipo_bar].to_numpy()
        ejes[1].bar(tabla.index, valores, bottom=fondo, label=tipo_bar,
                    color=COLORES_TIPO[tipo_bar])
        fondo += valores
    ejes[1].annotate("2021: {} documentos\n(año parcial)".format(int(fondo[-1])),
                     xy=(tabla.index[-1], fondo[-1]),
                     xytext=(tabla.index[-6], fondo[-1] + 0.28 * fondo.max()),
                     fontsize=8, bbox={"facecolor": "white", "alpha": 0.85,
                                       "edgecolor": "none"},
                     arrowprops={"arrowstyle": "->", "lw": 0.8})
    ejes[1].set_title("Documentos por año y tipo (1992-2021)", fontsize=9.5)
    ejes[1].set_xlabel("Año")
    ejes[1].set_ylabel("Número de documentos")
    ejes[1].legend(title="Tipo")
    fig.suptitle("Composición del corpus por tipo y por año")
    fig.tight_layout()
    guardar(fig, "fig_tipos_por_anio.png")


def figura_vocabulario_anio(metricas):
    por_anio = metricas["vocabulario_por_anio"]["por_anio"]
    anios = sorted(int(a) for a in por_anio)
    tipos = np.array([por_anio[str(a)]["tipos"] for a in anios])
    norm = np.array([por_anio[str(a)]["tipos_por_100k"] for a in anios])
    assert tipos.max() == 69822 and tipos[-1] == 17287
    colores = [ROJO if a == max(anios) else AZUL for a in anios]
    fig, ejes = plt.subplots(1, 2, figsize=(ANCHO, 2.4))
    for eje, valores, titulo, ymax in [
            (ejes[0], tipos, "Tipos únicos por año (sin stopwords)", tipos.max() * 1.12),
            (ejes[1], norm, "Tipos por 100.000 tokens (normalizado)", norm.max() * 1.18)]:
        eje.bar(anios, valores, color=colores)
        eje.set_title(titulo)
        eje.set_xlabel("Año")
        eje.tick_params(axis="x", rotation=60)
        eje.annotate("2021: parcial\n(20 documentos)", xy=(max(anios), valores[-1]),
                     xytext=(max(anios) - 12, ymax * 0.52),
                     fontsize=8, bbox={"facecolor": "white", "alpha": 0.85,
                                       "edgecolor": "none"},
                     arrowprops={"arrowstyle": "->", "lw": 0.8})
    ejes[0].set_ylabel("Tipos únicos")
    ejes[1].set_ylabel("Tipos por 100.000 tokens")
    fig.suptitle("Vocabulario por año de emisión (1992-2021)")
    fig.tight_layout()
    guardar(fig, "fig_vocabulario_por_anio.png")


def figura_mtld():
    mtld = pd.read_parquet(LEMAS / "mtld_por_doc.parquet")
    por_tipo = mtld.groupby("tipo")["mtld"].mean()
    por_ano = mtld.groupby("ano")["mtld"].mean()
    assert abs(por_tipo["T"] - 61.63) < 0.05 and abs(por_tipo["SU"] - 58.88) < 0.05
    fig, ejes = plt.subplots(1, 2, figsize=(ANCHO, 2.3))
    orden_tipo = ["T", "C", "SU"]
    valores_tipo = [float(por_tipo[t]) for t in orden_tipo]
    ejes[0].bar(orden_tipo, valores_tipo, color=[COLORES_TIPO[t] for t in orden_tipo])
    for pos, valor in enumerate(valores_tipo):
        ejes[0].text(pos, valor, num(valor, 2), ha="center", va="bottom", fontsize=8)
    ejes[0].set_ylim(0, max(valores_tipo) * 1.10)
    ejes[0].set_title("MTLD media por tipo de providencia")
    ejes[0].set_xlabel("Tipo")
    ejes[0].set_ylabel("MTLD")
    ejes[1].plot(por_ano.index.astype(int).tolist(), por_ano.tolist(), marker="o",
                 markersize=3)
    ejes[1].set_title("MTLD media por año (1992-2021)")
    ejes[1].set_xlabel("Año")
    ejes[1].set_ylabel("MTLD")
    fig.suptitle("Diversidad léxica MTLD (umbral 0,72)")
    fig.tight_layout()
    guardar(fig, "fig_mtld_distribucion.png")


def figura_mtld_vs_longitud():
    mtld = pd.read_parquet(LEMAS / "mtld_por_doc.parquet")
    rho = float(mtld["n_tokens"].corr(mtld["mtld"], method="spearman"))
    assert abs(rho - (-0.2628)) < 0.002, "Spearman: {}".format(rho)
    fig, eje = plt.subplots(figsize=(ANCHO, 3.9))
    dispersion = eje.hexbin(mtld["n_tokens"], mtld["mtld"], xscale="log", gridsize=60,
                            cmap="viridis", mincnt=1)
    fig.colorbar(dispersion, ax=eje, label="Documentos por celda")
    bordes = np.logspace(np.log10(mtld["n_tokens"].min()),
                         np.log10(mtld["n_tokens"].max()), 22)
    centros, medianas = [], []
    for inferior, superior in zip(bordes[:-1], bordes[1:]):
        sub = mtld[(mtld["n_tokens"] >= inferior) & (mtld["n_tokens"] < superior)]
        if len(sub) >= 10:
            centros.append(float(np.sqrt(inferior * superior)))
            medianas.append(float(sub["mtld"].median()))
    eje.plot(centros, medianas, "-o", color=ROJO, markersize=4,
             label="Mediana de MTLD por intervalo logarítmico")
    eje.set_title("MTLD frente a la longitud del documento (Spearman $\\rho$ = -0,2628)")
    eje.set_xlabel("Tokens del documento (log)")
    eje.set_ylabel("MTLD")
    eje.legend()
    fig.tight_layout()
    guardar(fig, "fig_mtld_vs_longitud.png")


def figura_logodds():
    diacronia = pd.read_csv(RES / "diacronia_logodds.csv")
    ultima = diacronia[diacronia["transicion"] == "2002-2011->2012-2021"]
    sube = ultima.nlargest(10, "log_odds")
    baja = ultima.nsmallest(10, "log_odds")
    assert sube.iloc[0]["lema"] == "uariv" and baja.iloc[0]["lema"] == "nilser"
    fig, ejes = plt.subplots(1, 2, figsize=(ANCHO, 2.9))
    for eje, datos, titulo, color in [
            (ejes[0], sube, "Lemas que suben (2002-2011 $\\rightarrow$ 2012-2021)", AZUL),
            (ejes[1], baja, "Lemas que bajan (2002-2011 $\\rightarrow$ 2012-2021)", ROJO)]:
        registros = list(reversed(datos.to_dict("records")))
        eje.barh([r["lema"] for r in registros], [r["log_odds"] for r in registros],
                 color=color)
        eje.set_xlabel("log-odds ratio")
        eje.set_title(titulo, fontsize=9)
        eje.tick_params(axis="y", labelsize=8)
    fig.suptitle("Evolución léxica por periodos: log-odds (1992-2001 / 2002-2011 / 2012-2021)")
    fig.tight_layout()
    guardar(fig, "fig_logodds_periodos.png")


def figura_ventana(metadatos):
    largos = metadatos["n_palabras_clean"].to_numpy()
    pct = 100.0 * float((largos > 512).mean())
    assert abs(pct - 99.95) < 0.01
    pct_tipo = {t: 100.0 * float((metadatos.loc[metadatos["tipo"] == t,
                                                 "n_palabras_clean"] > 512).mean())
                for t in ["T", "C", "SU"]}
    bordes = np.logspace(np.log10(max(largos.min(), 1)), np.log10(largos.max()), 60)
    fig, ejes = plt.subplots(1, 2, figsize=(ANCHO, 2.5))
    ejes[0].hist(largos, bins=bordes, color=AZUL, edgecolor="white")
    ejes[0].set_xscale("log")
    ejes[0].axvline(512, color=ROJO, linestyle="--", linewidth=1.8)
    cuentas = np.histogram(largos, bins=bordes)[0]
    ejes[0].annotate("99,95 % de los documentos\nexceden 512 tokens",
                     xy=(512, cuentas.max() * 0.35),
                     xytext=(512 * 2.2, cuentas.max() * 0.5), fontsize=8,
                     bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
                     arrowprops={"arrowstyle": "->", "lw": 0.8})
    ejes[0].set_ylim(0, cuentas.max() * 1.18)
    ejes[0].set_title("Longitudes por documento y ventana de 512 tokens")
    ejes[0].set_xlabel("Palabras limpias por documento (log)")
    ejes[0].set_ylabel("Número de documentos")
    tipos = ["T", "C", "SU"]
    ejes[1].bar(tipos, [pct_tipo[t] for t in tipos], color=[COLORES_TIPO[t] for t in tipos])
    for pos, t in enumerate(tipos):
        ejes[1].text(pos, pct_tipo[t] + 1.5, "{} %\n(n={})".format(
            num(pct_tipo[t], 2), miles((metadatos["tipo"] == t).sum())),
            ha="center", va="bottom", fontsize=8)
    ejes[1].set_title("Documentos que exceden 512 tokens por tipo")
    ejes[1].set_xlabel("Tipo")
    ejes[1].set_ylabel("Porcentaje de documentos (%)")
    ejes[1].set_ylim(0, 120)
    fig.suptitle("Cobertura de la ventana de 512 tokens sobre el corpus")
    fig.tight_layout()
    guardar(fig, "fig_longitud_vs_ventana.png")


def figura_cobertura():
    unigramas = pd.read_csv(EDA / "ngramas_1.csv")
    total = 250_012_101
    acumulado = np.cumsum(unigramas["freq"].to_numpy()) / float(total) * 100.0
    puntos = {k: float(acumulado[k - 1]) for k in (1000, 5000, 10000, 20000)}
    esperados = {1000: 77.90, 5000: 92.96, 10000: 96.64, 20000: 98.66}
    for k, valor in puntos.items():
        assert abs(valor - esperados[k]) < 0.05, "cobertura top-{}: {}".format(k, valor)
    fig, eje = plt.subplots(figsize=(ANCHO, 4.0))
    eje.plot(np.arange(1, len(acumulado) + 1), acumulado, color=AZUL, linewidth=2)
    for k, valor in puntos.items():
        eje.plot([k], [valor], "o", color=ROJO, markersize=5)
        eje.annotate("top-{}: {} %".format(miles(k), num(valor, 2)), xy=(k, valor),
                     xytext=(k * 1.15, valor - 5.0), fontsize=8.5,
                     arrowprops={"arrowstyle": "-", "lw": 0.7})
    eje.set_xscale("log")
    eje.set_xlabel("Tamaño del vocabulario top-k (log)")
    eje.set_ylabel("% acumulado de tokens cubiertos")
    eje.set_title("Cobertura acumulada del vocabulario (unigramas)")
    eje.grid(alpha=0.3, which="both")
    eje.set_ylim(float(acumulado.min()) - 2.0, 101.0)
    fig.tight_layout()
    guardar(fig, "fig_cobertura_vocabulario.png")


def figura_granularidad():
    gran = pd.read_csv(EDA / "granularidad.csv")
    nombres = ["sentencia", "párrafo", "subsentencia"]
    pcts = gran["pct_con_marcador"].tolist()
    medias = gran["media_marcadores"].tolist()
    assert pcts == [97.17, 97.17, 4.64]
    fig, ejes = plt.subplots(1, 2, figsize=(ANCHO, 2.5))
    ejes[0].bar(nombres, pcts, color=TAB_TRES)
    for pos, valor in enumerate(pcts):
        ejes[0].text(pos, valor, "{} %".format(num(valor, 2)), ha="center", va="bottom",
                     fontsize=8.5)
    ejes[0].set_title("Unidades con marcador por granularidad")
    ejes[0].set_ylabel("Porcentaje con ≥1 marcador")
    ejes[0].set_ylim(0, 112)
    ejes[1].bar(nombres, medias, color=TAB_TRES)
    ejes[1].set_title("Media de marcadores por unidad")
    ejes[1].set_ylabel("Marcadores por unidad")
    fig.suptitle("Granularidad y densidad de marcadores temáticos")
    fig.tight_layout()
    guardar(fig, "fig_granularidad.png")


def figura_ngramas():
    bigramas = pd.read_csv(EDA / "ngramas_2.csv").head(15)
    trigramas = pd.read_csv(EDA / "ngramas_3.csv").head(15)
    assert bigramas.iloc[0]["ngrama"] == "corte constitucional"
    assert bigramas.iloc[0]["freq"] == 286786 and trigramas.iloc[0]["freq"] == 344893
    fig, ejes = plt.subplots(1, 2, figsize=(ANCHO, 2.8))
    for eje, datos, titulo, color in [
            (ejes[0], bigramas, "Top-15 bigramas (sin stopwords en bordes)", None),
            (ejes[1], trigramas, "Top-15 trigramas (bordes de contenido)", "tab:orange")]:
        registros = list(reversed(datos.to_dict("records")))
        eje.barh([r["ngrama"] for r in registros], [r["freq"] for r in registros],
                 color=color)
        eje.set_xlabel("Frecuencia")
        eje.set_title(titulo, fontsize=9)
        eje.tick_params(axis="y", labelsize=8)
    fig.suptitle("N-gramas más frecuentes del corpus (23.742 sentencias)")
    fig.tight_layout()
    guardar(fig, "fig_ngramas_bi_tri.png")


def figura_colocaciones():
    col = pd.read_csv(EDA / "colocaciones.csv")
    llr = col[col["medida"] == "llr"]
    bigramas = llr[llr["tipo"] == "bigrama"].nsmallest(10, "rank")
    trigramas = llr[llr["tipo"] == "trigrama"].nsmallest(10, "rank")
    assert len(bigramas) == 10 and len(trigramas) == 10
    fig, ejes = plt.subplots(1, 2, figsize=(ANCHO, 2.8))
    for eje, datos, titulo, color in [
            (ejes[0], bigramas, "Top-10 bigramas por verosimilitud", None),
            (ejes[1], trigramas, "Top-10 trigramas por verosimilitud", "tab:green")]:
        registros = list(reversed(datos.to_dict("records")))
        eje.barh([r["ngrama"] for r in registros], [r["score"] for r in registros],
                 color=color)
        eje.set_xlabel("Razón de verosimilitud")
        eje.set_title(titulo, fontsize=9)
        eje.xaxis.set_major_formatter(FuncFormatter(
            lambda x, _: "0" if x == 0 else num(x / 1e6).rstrip("0").rstrip(",") + " M"))
        eje.tick_params(axis="y", labelsize=8)
    fig.suptitle("Colocaciones destacadas (verosimilitud, filtro freq ≥ 5)")
    fig.tight_layout()
    guardar(fig, "fig_colocaciones.png")


def figura_top20(contador_sin, metricas):
    unigramas = pd.read_csv(EDA / "ngramas_1.csv")
    top_con = unigramas.head(20)
    esperado_con = [(t, f) for t, f in metricas["top20_con_stopwords"]]
    obtenido_con = list(zip(top_con["ngrama"], top_con["freq"].astype(int)))
    assert obtenido_con == esperado_con, "top-20 con stopwords no coincide"
    esperado_sin = [(t, f) for t, f in metricas["top20_sin_stopwords"]]
    obtenido_sin = contador_sin.most_common(20)
    assert obtenido_sin == esperado_sin, "top-20 sin stopwords no coincide"
    for datos, titulo, nombre in [
            (list(zip(top_con["ngrama"], top_con["freq"].astype(int))),
             "Top-20 de palabras CON stopwords (corpus completo)",
             "fig_top20_con_stopwords.png"),
            (contador_sin.most_common(20),
             "Top-20 de palabras SIN stopwords gramaticales (corpus completo)",
             "fig_top20_sin_stopwords.png")]:
        registros = list(reversed(datos))
        fig, eje = plt.subplots(figsize=(ANCHO, 5.0))
        eje.barh([t for t, _ in registros], [f for _, f in registros],
                 color=AZUL, edgecolor="white")
        for pos, (_, valor) in enumerate(registros):
            eje.text(valor, pos, "  {}".format(miles(valor)), va="center", fontsize=8)
        eje.set_xlabel("Frecuencia absoluta")
        eje.set_title(titulo)
        eje.tick_params(axis="y", labelsize=9)
        eje.set_xlim(0, max(f for _, f in registros) * 1.16)
        fig.tight_layout()
        guardar(fig, nombre)


def figura_lemma_vs_stem():
    tabla = pd.read_csv(EDA / "lemma_vs_stem.csv")
    tabla = tabla[tabla["representacion"].isin(["tokens", "lemas", "stems"])]
    vocabulario = tabla["vocab_tipos"].tolist()
    compresion = tabla["compresion_pct"].tolist()
    assert vocabulario == [224468, 195959, 121696]
    fig, ejes = plt.subplots(1, 2, figsize=(ANCHO, 2.6))
    ejes[0].bar(["tokens", "lemas", "stems"], vocabulario, color=TAB_TRES)
    for pos, valor in enumerate(vocabulario):
        ejes[0].text(pos, valor, miles(valor), ha="center", va="bottom", fontsize=8.5)
    ejes[0].set_title("Tamaño de vocabulario por representación", fontsize=9.5)
    ejes[0].set_ylabel("Tipos únicos")
    ejes[0].set_ylim(0, max(vocabulario) * 1.12)
    ejes[1].bar(["tokens", "lemas", "stems"], compresion, color=TAB_TRES)
    for pos, valor in enumerate(compresion):
        ejes[1].text(pos, valor, "{} %".format(num(valor)), ha="center", va="bottom",
                     fontsize=8.5)
    ejes[1].set_title("Compresión respecto a tokens (= 100 %)", fontsize=9.5)
    ejes[1].set_ylabel("Porcentaje")
    ejes[1].set_ylim(0, 118)
    fig.suptitle("Lemas frente a stems: compresión de vocabulario")
    fig.tight_layout()
    guardar(fig, "fig_lemma_vs_stem.png")


def figura_pos():
    global_pos = pd.read_csv(LEMAS / "pos_global.csv")
    total = global_pos["freq"].sum()
    pct_noun = 100.0 * global_pos.loc[global_pos["pos"] == "NOUN", "freq"].iloc[0] / total
    pct_adp = 100.0 * global_pos.loc[global_pos["pos"] == "ADP", "freq"].iloc[0] / total
    print("  POS: NOUN {:.2f} %, ADP {:.2f} %".format(pct_noun, pct_adp))
    assert abs(pct_noun - 22.58) < 0.05 and abs(pct_adp - 20.45) < 0.05
    por_tipo = pd.read_csv(LEMAS / "pos_por_tipo_ano.csv")
    pivote = por_tipo.pivot_table(index="tipo", columns="pos", values="freq",
                                  aggfunc="sum", fill_value=0)
    top10 = global_pos.head(10)
    top6 = global_pos.head(6)["pos"].tolist()
    fig, ejes = plt.subplots(1, 2, figsize=(ANCHO, 2.8))
    ejes[0].barh(top10["pos"][::-1], top10["freq"][::-1])
    ejes[0].set_title("Top-10 etiquetas POS (global)")
    ejes[0].set_xlabel("Frecuencia")
    ejes[0].tick_params(axis="y", labelsize=8.5)
    pivote.loc[["T", "C", "SU"], top6].T.plot(
        kind="bar", ax=ejes[1], color=[COLORES_TIPO[t] for t in ["T", "C", "SU"]])
    ejes[1].set_title("Distribución POS por tipo (T/C/SU)")
    ejes[1].set_xlabel("Etiqueta POS")
    ejes[1].set_ylabel("Frecuencia")
    ejes[1].tick_params(axis="x", rotation=0)
    ejes[1].legend(title="Tipo")
    fig.suptitle("Distribución morfosintáctica del corpus")
    fig.tight_layout()
    guardar(fig, "fig_pos_distribucion.png")


def figura_espectro(contador_sin, metricas):
    frecuencias = np.fromiter(contador_sin.values(), dtype=np.int64)
    total_tipos = len(frecuencias)
    total_tokens = int(frecuencias.sum())
    hapax = int((frecuencias == 1).sum())
    bordes = np.logspace(0, np.log10(frecuencias.max()), 55)
    hist_tipos, _ = np.histogram(frecuencias, bins=bordes)
    centros = np.sqrt(bordes[:-1] * bordes[1:])
    estratos_json = metricas["espectro_frecuencias"]["estratos"]
    limites = [(1, 1), (2, 5), (6, 20), (21, 100), (101, 1000), (1001, int(frecuencias.max()))]
    etiquetas = ["1", "2-5", "6-20", "21-100", "101-1k", ">1k"]
    pct_tipos, pct_tokens = [], []
    for inferior, superior in limites:
        seleccion = frecuencias[(frecuencias >= inferior) & (frecuencias <= superior)]
        pct_tipos.append(100.0 * len(seleccion) / total_tipos)
        pct_tokens.append(100.0 * seleccion.sum() / total_tokens)
    for valor, esperado in zip(pct_tipos, estratos_json):
        assert abs(valor - esperado["pct_tipos"]) < 0.01, "estrato tipos: {}".format(valor)
    for valor, esperado in zip(pct_tokens, estratos_json):
        assert abs(valor - esperado["pct_tokens"]) < 0.01, "estrato tokens: {}".format(valor)

    fig, ejes = plt.subplots(1, 2, figsize=(ANCHO, 2.7))
    ejes[0].loglog(centros, hist_tipos, "o-", markersize=3, color=AZUL)
    ejes[0].axvline(1.0, color=ROJO, linestyle="--", linewidth=1.2)
    ejes[0].annotate("Hápax (frecuencia 1): {} tipos\n({} % del vocabulario)".format(
        miles(hapax), num(100.0 * hapax / total_tipos)),
        xy=(1.0, hist_tipos[0]), xytext=(1.8, hist_tipos[0] * 0.45), fontsize=8,
        arrowprops={"arrowstyle": "->", "lw": 0.8})
    ejes[0].set_title("Tipos por rango de frecuencia")
    ejes[0].set_xlabel("Frecuencia del término (log)")
    ejes[0].set_ylabel("Número de tipos (log)")
    ejes[0].grid(alpha=0.3, which="both")
    posiciones = np.arange(len(etiquetas))
    ancho = 0.38
    ejes[1].bar(posiciones - ancho / 2, pct_tipos, ancho, label="% de tipos", color=AZUL)
    ejes[1].bar(posiciones + ancho / 2, pct_tokens, ancho, label="% de tokens", color=ROJO)
    for pos, (valor_tipos, valor_tokens) in enumerate(zip(pct_tipos, pct_tokens)):
        ejes[1].text(pos - ancho / 2, valor_tipos + 1.2, num(valor_tipos),
                     ha="center", fontsize=8)
        ejes[1].text(pos + ancho / 2, valor_tokens + 1.2, num(valor_tokens),
                     ha="center", fontsize=8)
    ejes[1].set_xticks(posiciones)
    ejes[1].set_xticklabels(etiquetas)
    ejes[1].set_title("Concentración: tipos frente a tokens por estrato")
    ejes[1].set_xlabel("Estrato de frecuencia (ocurrencias)")
    ejes[1].set_ylabel("Porcentaje (%)")
    ejes[1].legend()
    ejes[1].set_ylim(0, 105)
    fig.suptitle("Espectro de frecuencias del vocabulario de contenido (sin stopwords)")
    fig.tight_layout()
    guardar(fig, "fig_espectro_frecuencias.png")


def figura_tfidf(masa):
    for rango in (100, 1000, 10000):
        print("  cola larga top-{}: {:.1f} %".format(miles(rango), 100.0 * masa[rango - 1]))
    ejemplos = pd.read_csv(RES / "ejemplos_vectorizacion_tfidf.csv")
    top15 = ejemplos[ejemplos["documento"] == "T-001/2021 (id 0)"].head(15)
    assert len(top15) == 15
    fig, ejes = plt.subplots(1, 2, figsize=(ANCHO, 2.7))
    ejes[0].plot(np.arange(1, len(masa) + 1), masa, color=AZUL)
    ejes[0].set_xscale("log")
    ejes[0].set_xlabel("Rango del término (escala log)")
    ejes[0].set_ylabel("Masa acumulada de document frequency")
    ejes[0].set_title("Cola larga del vocabulario TF-IDF")
    ejes[0].grid(alpha=0.3)
    for rango in (10, 100, 1000, 10000):
        ejes[0].annotate("{:.0f} %".format(100.0 * masa[rango - 1]),
                         xy=(rango, masa[rango - 1]),
                         xytext=(rango * 1.1, masa[rango - 1] - 0.08),
                         fontsize=8, arrowprops={"arrowstyle": "-", "lw": 0.6})
    ejes[0].axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ejes[0].text(1.2, 0.52, "50 % de la masa de frecuencia documental", fontsize=8,
                 color="gray")
    ejes[1].barh(top15["termino"][::-1], top15["peso_tfidf"][::-1], color=ROJO)
    ejes[1].set_xlabel("Peso TF-IDF")
    ejes[1].set_title("Top-15 rasgos del documento T-001/2021 (id 0)")
    ejes[1].tick_params(axis="y", labelsize=8)
    fig.suptitle("Representación TF-IDF sobre lemas (23.742 documentos)")
    fig.tight_layout()
    guardar(fig, "fig_representacion_tfidf.png")


def figura_embeddings():
    tabla = pd.read_csv(RES / "embeddings_e5_similitud.csv", index_col=0)
    etiquetas = tabla.index.tolist()
    similitud = tabla.to_numpy()
    assert similitud.shape == (5, 5)
    fig, eje = plt.subplots(figsize=(5.63, 4.9))
    imagen = eje.imshow(similitud, cmap="viridis", vmin=0.5, vmax=1.0)
    eje.set_xticks(range(len(etiquetas)))
    eje.set_xticklabels(etiquetas, rotation=45, ha="right", fontsize=8)
    eje.set_yticks(range(len(etiquetas)))
    eje.set_yticklabels(etiquetas, fontsize=8)
    for i in range(len(etiquetas)):
        for j in range(len(etiquetas)):
            eje.text(j, i, "{:.2f}".format(similitud[i, j]), ha="center", va="center",
                     fontsize=8, color="white" if similitud[i, j] < 0.8 else "black")
    fig.colorbar(imagen, fraction=0.046)
    eje.set_title("Similitud coseno e5-base (primeros 512 tokens)\ndimensión = 768")
    fig.tight_layout()
    guardar(fig, "fig_embeddings_ejemplo.png")


def figuras_nubes(metadatos, por_anio):
    """Nubes anuales por periodo (mismo estilo; top-250 por año para legibilidad)."""
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("  wordcloud no disponible; se conservan las nubes anteriores.")
        return
    min_docs, top_n, semilla = 50, 250, 42
    for inicio, fin in [(1992, 2001), (2002, 2011), (2012, 2020)]:
        anios = [a for a in sorted(por_anio)
                 if inicio <= a <= fin
                 and int((metadatos["annio"] == a).sum()) >= min_docs]
        n_filas = -(-len(anios) // 2)
        fig, ejes = plt.subplots(n_filas, 2, figsize=(2.75, 3.3))
        ejes = np.atleast_1d(ejes).ravel()
        for pos, anio in enumerate(anios):
            frecuencias = {t: f for t, f in por_anio[anio].most_common(top_n)
                           if t not in STOPW_DOMINIO_JURIDICO}
            nube = WordCloud(width=600, height=250, background_color="white",
                             colormap="viridis", random_state=semilla,
                             collocations=False).generate_from_frequencies(frecuencias)
            ejes[pos].imshow(nube, interpolation="bilinear")
            ejes[pos].axis("off")
            ejes[pos].set_title("Año {}".format(anio), fontsize=7.5)
        for extra in range(len(anios), len(ejes)):
            ejes[extra].axis("off")
        fig.tight_layout(pad=0.25)
        guardar(fig, "fig_nubes_por_anio_{}_{}.png".format(inicio, fin))


def main():
    with open(RES / "metricas_entrega_ii.json", encoding="utf-8") as fh:
        metricas = json.load(fh)
    metadatos = pd.read_csv(DATOS / "df_metadatos.csv")
    contador_sin, por_anio = cargar_o_calcular_frecuencias(metadatos)

    print("Generando figuras...")
    figura_longitudes(metadatos)
    figura_tipos(metadatos)
    figura_vocabulario_anio(metricas)
    figura_mtld()
    figura_mtld_vs_longitud()
    figura_logodds()
    figura_ventana(metadatos)
    figura_cobertura()
    figura_granularidad()
    figura_ngramas()
    figura_colocaciones()
    figura_top20(contador_sin, metricas)
    figura_lemma_vs_stem()
    figura_pos()
    figura_espectro(contador_sin, metricas)
    figura_tfidf(cargar_o_calcular_cola_tfidf())
    figura_embeddings()
    figuras_nubes(metadatos, por_anio)
    print("Listo. Versiones anteriores respaldadas en:", BACKUP)


if __name__ == "__main__":
    main()
