"""Regenera las figuras del articulo a tamano de imprenta (legibles sin zoom).

Salidas en informe/figuras/ (nombres nuevos fig_XX_*.png):
- fig_01_eda.png        : clases + longitudes (caracteres train/test; palabras train)
- fig_02_terminos.png   : terminos mas frecuentes por rango (ancho completo)
- fig_03_clusters.png   : tamanos de clusteres y test con gemelo
- fig_04_bloques.png    : tamano y varianza por familia de features
- fig_05_miembros.png   : aporte de los 14 miembros del meta
- fig_07_confusion.png  : confusion + F1 por rango del campeon (ancho completo)
- fig_08_progresion.png : estructura condicional + progresion de la validacion
- fig_10_familia_nn.png : familia neuronal V38-V43 + HGB
- fig_11_powermean.png  : concavidad Power-Mean (V57)
- fig_13_test.png       : distribucion de decisiones sobre el test
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
FIG = RAIZ / "informe" / "figuras"
AZUL, ROJO, VERDE, GRIS = "#2b6cb0", "#c53030", "#2f855a", "#718096"
CLASES = ["13-17", "18-24", "25-34", "35-49", "50-64", "65-xx"]

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "figure.dpi": 220, "savefig.bbox": "tight"})


def cargar_train() -> pd.DataFrame:
    train = pd.read_csv(RAIZ / "data" / "raw" / "age_train.csv", sep=";")
    train["texto"] = train["text"].fillna("").str.strip()
    return train


def figura_01_eda(train: pd.DataFrame, test: pd.DataFrame) -> None:
    figura = plt.figure(figsize=(3.45, 3.5))
    malla = figura.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.55, wspace=0.35)
    eje_a = figura.add_subplot(malla[0, :])
    eje_b = figura.add_subplot(malla[1, 0])
    eje_c = figura.add_subplot(malla[1, 1])
    conteo = train["age_range"].value_counts().reindex(CLASES)
    eje_a.bar(CLASES, conteo / conteo.sum() * 100, color=AZUL)
    eje_a.set_ylabel("filas train (%)")
    eje_a.set_title("(a) Distribuci\u00f3n de clases")
    eje_a.tick_params(axis="x", rotation=35)
    for i, valor in enumerate(conteo / conteo.sum() * 100):
        eje_a.text(i, valor + 0.25, f"{valor:.1f}", ha="center", fontsize=6.8)
    eje_a.set_ylim(0, 20)
    largos_tr = train["texto"].str.len().to_numpy()
    largos_te = test["text"].fillna("").str.strip().str.len().to_numpy()
    eje_b.hist([largos_tr, largos_te], bins=np.arange(0, 145, 5), histtype="step",
               color=[AZUL, ROJO], label=["train", "test"], linewidth=1.2)
    eje_b.set_xlabel("caracteres")
    eje_b.set_ylabel("tuits")
    eje_b.set_title("(b) Longitudes")
    eje_b.legend(fontsize=6.5)
    palabras = train["texto"].str.split().str.len().to_numpy()
    eje_c.hist(palabras, bins=np.arange(0, 45, 2), color=GRIS)
    eje_c.set_xlabel("palabras")
    eje_c.set_ylabel("tuits")
    eje_c.set_title("(c) Palabras (train)")
    figura.savefig(FIG / "fig_01_eda.png")
    plt.close(figura)


def figura_02_terminos(train: pd.DataFrame) -> None:
    figura, ejes = plt.subplots(2, 3, figsize=(7.16, 3.7))
    for eje, clase in zip(ejes.ravel(), CLASES):
        textos = train.loc[train["age_range"] == clase, "texto"]
        contador = Counter()
        for texto in textos:
            contador.update(t for t in re.findall(r"[a-z0-9]+", texto) if len(t) >= 4)
        comunes = contador.most_common(10)[::-1]
        etiquetas = [t for t, _ in comunes]
        valores = [c for _, c in comunes]
        eje.barh(etiquetas, valores, color=AZUL)
        eje.set_title(clase, fontsize=8)
        eje.tick_params(labelsize=6.6)
        eje.set_xlim(0, max(valores) * 1.15)
    figura.suptitle("T\u00e9rminos m\u00e1s frecuentes por rango (tokens de 4+ caracteres)",
                    fontsize=8.5)
    figura.tight_layout(rect=(0, 0, 1, 0.94))
    figura.savefig(FIG / "fig_02_terminos.png")
    plt.close(figura)


def figura_03_clusters() -> None:
    clusters = pd.read_parquet(RAIZ / "features" / "dup_clusters.parquet")
    tamanos = clusters["cluster"].value_counts()
    distribucion = tamanos.value_counts().sort_index()
    chicos = distribucion[distribucion.index <= 4]
    grandes = int(distribucion[distribucion.index > 4].sum())
    figura, ejes = plt.subplots(1, 2, figsize=(3.45, 1.9))
    ejes[0].bar(chicos.index.astype(str), chicos.values, color=AZUL)
    if grandes:
        ejes[0].bar(["5+"], [grandes], color=AZUL)
    ejes[0].set_yscale("log")
    ejes[0].set_xlabel("tama\u00f1o del cl\u00faster")
    ejes[0].set_ylabel("cl\u00fasteres (log)")
    ejes[0].set_title("(a) Cl\u00fasteres", fontsize=8)
    ejes[0].tick_params(labelsize=6.8)
    categorias = ["literal\nen train", "near-dup\n(casi igual)", "sin\ngemelo"]
    valores = [1.21, 4.26, 94.53]
    barras = ejes[1].barh(categorias, valores, color=[ROJO, "#b7791f", GRIS])
    for barra, valor in zip(barras, valores):
        ejes[1].text(valor + 2, barra.get_y() + barra.get_height() / 2, f"{valor:.2f}%",
                     va="center", fontsize=6.6)
    ejes[1].set_xlim(0, 118)
    ejes[1].set_title("(b) Test con gemelo", fontsize=8)
    ejes[1].tick_params(labelsize=6.8)
    figura.tight_layout()
    figura.savefig(FIG / "fig_03_clusters.png")
    plt.close(figura)
    print("clusters: distribucion de tamanos:", dict(distribucion.head(6)), "| 5+:", grandes)


def figura_04_bloques() -> None:
    meta = np.load(RAIZ / "features" / "meta_features_train_v11.npy", mmap_mode="r")
    bloques = [("miembros", 0, 78), ("densas", 78, 113), ("estruct.", 113, 200),
               ("supervis.", 200, 273)]
    varianzas = [float(np.var(np.asarray(meta[:, a:b], dtype=np.float64), axis=0).mean())
                 for _, a, b in bloques]
    figura, ejes = plt.subplots(1, 2, figsize=(3.45, 1.9))
    tamanos = [b - a for _, a, b in bloques]
    ejes[0].bar([n for n, _, _ in bloques], tamanos, color=AZUL)
    ejes[0].set_ylabel("columnas")
    ejes[0].set_title("(a) Tama\u00f1o del bloque", fontsize=8)
    ejes[0].tick_params(axis="x", rotation=25, labelsize=6.8)
    ejes[1].bar([n for n, _, _ in bloques], varianzas, color=VERDE)
    ejes[1].set_yscale("log")
    ejes[1].set_ylabel("varianza media (log)")
    ejes[1].set_title("(b) Varianza por familia", fontsize=8)
    ejes[1].tick_params(axis="x", rotation=25, labelsize=6.8)
    figura.tight_layout()
    figura.savefig(FIG / "fig_04_bloques.png")
    plt.close(figura)
    print("bloques: varianzas medias:", [f"{v:.4g}" for v in varianzas])


def figura_05_miembros() -> None:
    valores = {
        "e013": 0.5628, "e016": 0.5659, "e009": 0.5603, "e011": 0.5580, "e010": 0.5370,
        "e014": 0.5644, "e015_ord": 0.3973, "e015_soft": 0.3973, "e017": 0.5015,
        "e019b": 0.4075, "e021a": 0.5528, "e021b": 0.4017, "e028a": 0.0719, "adv05": 0.5641,
    }
    figura, eje = plt.subplots(figsize=(3.45, 2.3))
    eje.bar(list(valores), list(valores.values()), color=AZUL)
    eje.axhline(0.6005, color=VERDE, linestyle="--", linewidth=1.0,
                label="meta V12 (0.6005)")
    eje.axhline(0.5344, color=ROJO, linestyle="--", linewidth=1.0, label="baseline E-001")
    eje.set_ylabel("macro-F1 (OOF)")
    eje.set_ylim(0, 0.68)
    eje.tick_params(axis="x", rotation=55, labelsize=6.4)
    eje.legend(fontsize=6.4, loc="upper right")
    figura.tight_layout()
    figura.savefig(FIG / "fig_05_miembros.png")
    plt.close(figura)


def figura_07_confusion() -> None:
    oof_v30 = np.load(RAIZ / "features" / "age_plugin_segunda_particion_oof.npy")
    componentes = [np.load(RAIZ / "features" / nombre) for nombre in (
        "v43_oof_cramer_swa_tabularnet.npy", "v42_oof_smooth_swa_tabularnet.npy",
        "v40_oof_se_tabularnet.npy", "v39_oof_tabnet_ordinal.npy", "v39_oof_hgb.npy")]
    pesos = [0.5628, 0.0832, 0.1724, 0.0754, 0.0513, 0.0549]
    plug_in = np.array([1.0000, 1.1749, 1.2721, 1.1630, 1.0010, 0.8128])
    y_true = np.load(RAIZ / "artifacts" / "p33_scores" / "todos" / "y_true.npy")
    modelos = [oof_v30, *componentes]
    log_geom = sum(w * np.log(np.clip(m, 1e-12, 1.0)) for w, m in zip(pesos, modelos))
    prob = np.exp(log_geom - log_geom.max(axis=1, keepdims=True))
    prob = prob / prob.sum(axis=1, keepdims=True)
    pred = np.argmax(prob * plug_in, axis=1)
    confusion = np.zeros((6, 6), dtype=int)
    for real, predicho in zip(y_true, pred):
        confusion[real, predicho] += 1
    f1s, macro = [], 0.0
    for c in range(6):
        vp = confusion[c, c]
        fp = confusion[:, c].sum() - vp
        fn = confusion[c, :].sum() - vp
        f1 = 2 * vp / (2 * vp + fp + fn) if (2 * vp + fp + fn) else 0.0
        f1s.append(f1)
        macro += f1 / 6
    figura, ejes = plt.subplots(1, 2, figsize=(7.16, 2.9))
    imagen = ejes[0].imshow(confusion, cmap="Blues")
    for i in range(6):
        for j in range(6):
            valor = confusion[i, j]
            color = "white" if valor > confusion.max() * 0.55 else "#1a202c"
            ejes[0].text(j, i, f"{valor:,}".replace(",", "."), ha="center", va="center",
                         fontsize=6.8, color=color)
    ejes[0].set_xticks(range(6), CLASES, rotation=40, fontsize=7)
    ejes[0].set_yticks(range(6), CLASES, fontsize=7)
    ejes[0].set_xlabel("predicho")
    ejes[0].set_ylabel("real")
    ejes[0].set_title("(a) Matriz de confusi\u00f3n (OOF del campe\u00f3n)", fontsize=8.5)
    figura.colorbar(imagen, ax=ejes[0], shrink=0.85).ax.tick_params(labelsize=6.5)
    barras = ejes[1].bar(CLASES, f1s, color=AZUL)
    for barra, f1 in zip(barras, f1s):
        ejes[1].text(barra.get_x() + barra.get_width() / 2, f1 + 0.008, f"{f1:.3f}",
                     ha="center", fontsize=7)
    ejes[1].set_ylim(0, 0.83)
    ejes[1].set_ylabel("F1 por rango")
    ejes[1].set_title(f"(b) F1 por rango (macro-F1 = {macro:.4f})", fontsize=8.5)
    ejes[1].tick_params(axis="x", rotation=40, labelsize=7)
    figura.tight_layout()
    figura.savefig(FIG / "fig_07_confusion.png")
    plt.close(figura)
    print("V48: macro-F1", f"{macro:.7f}", "| F1 por clase:", [f"{f:.4f}" for f in f1s])
    print("confusion diagonal:", confusion.diagonal().tolist())


def figura_08_progresion() -> None:
    prob = np.load(RAIZ / "artifacts" / "p33_scores" / "todos" / "v30_prob.npy")
    y_true = np.load(RAIZ / "artifacts" / "p33_scores" / "todos" / "y_true.npy")
    condicional = np.zeros((6, 6))
    for fila in range(6):
        condicional[fila] = prob[y_true == fila].mean(axis=0)
    figura, ejes = plt.subplots(1, 2, figsize=(3.45, 2.4),
                                gridspec_kw={"width_ratios": [1.0, 1.25]})
    imagen = ejes[0].imshow(condicional, cmap="Blues", vmin=0, vmax=0.7)
    ejes[0].set_xticks(range(6), CLASES, rotation=55, fontsize=5.6)
    ejes[0].set_yticks(range(6), CLASES, fontsize=5.6)
    ejes[0].set_title("(a) P(clase | real)", fontsize=8)
    figura.colorbar(imagen, ax=ejes[0], shrink=0.85).ax.tick_params(labelsize=6)
    versiones = ["V12", "V20", "V30", "V30\nplug-in", "V48", "V56"]
    valores = [0.6005, 0.6097782, 0.6134471, 0.6159732, 0.6186451, 0.6189929]
    barras = ejes[1].bar(versiones, valores, color=AZUL)
    for barra, valor in zip(barras, valores):
        ejes[1].text(barra.get_x() + barra.get_width() / 2, valor + 0.0004,
                     f"{valor:.4f}", ha="center", fontsize=5.8)
    ejes[1].set_ylim(0.595, 0.6215)
    ejes[1].set_ylabel("CV local")
    ejes[1].set_title("(b) Progresi\u00f3n de la validaci\u00f3n", fontsize=8)
    ejes[1].tick_params(axis="x", labelsize=6.2)
    figura.tight_layout()
    figura.savefig(FIG / "fig_08_progresion.png")
    plt.close(figura)


def figura_10_familia_nn() -> None:
    valores = {"v38": 0.6086, "v39_ord": 0.6092, "v40": 0.6091, "v41": 0.6125,
               "v42": 0.6132, "v43": 0.6128, "v39_hgb": 0.6112}
    figura, eje = plt.subplots(figsize=(3.45, 2.0))
    barras = eje.bar(list(valores), list(valores.values()), color=AZUL)
    for barra, valor in zip(barras, valores.values()):
        eje.text(barra.get_x() + barra.get_width() / 2, valor + 0.0002, f"{valor:.4f}",
                 ha="center", fontsize=6.2)
    eje.axhline(0.6131, color=GRIS, linestyle="--", linewidth=1.0,
                label="\u00e1rbol troncal V30")
    eje.set_ylim(0.605, 0.616)
    eje.set_ylabel("macro-F1 (OOF)")
    eje.tick_params(axis="x", labelsize=6.8)
    eje.legend(fontsize=6.4, loc="lower right")
    figura.tight_layout()
    figura.savefig(FIG / "fig_10_familia_nn.png")
    plt.close(figura)


def figura_11_powermean() -> None:
    datos = json.loads((RAIZ / "experiments" / "v57_analisis_matematico_estadistico.json")
                       .read_text(encoding="utf-8"))
    curva = datos["curva_power_mean"]
    alphas = [punto["alpha"] for punto in curva]
    f1s = [punto["macro_f1"] for punto in curva]
    figura, eje = plt.subplots(figsize=(3.45, 1.9))
    eje.plot(alphas, f1s, marker="o", markersize=3.2, color=AZUL, linewidth=1.2)
    eje.axvline(0.0, color=VERDE, linestyle="--", linewidth=1.0)
    eje.annotate("PoE (alpha = 0)", xy=(0.0, min(f1s)), xytext=(0.025, min(f1s) + 0.00002),
                 fontsize=6.8, color=VERDE)
    eje.set_xlabel("alpha (familia Power-Mean)")
    eje.set_ylabel("macro-F1 (OOF)")
    figura.tight_layout()
    figura.savefig(FIG / "fig_11_powermean.png")
    plt.close(figura)


def figura_13_test() -> None:
    envio = pd.read_csv(RAIZ / "submissions" / "age_submission_v48_geometric_poe_blend.csv")
    conteo = envio["age_range"].value_counts().reindex(CLASES)
    figura, eje = plt.subplots(figsize=(3.45, 1.9))
    barras = eje.barh(CLASES[::-1], conteo[::-1], color=AZUL)
    for barra, valor in zip(barras, conteo[::-1]):
        eje.text(valor + 300, barra.get_y() + barra.get_height() / 2,
                 f"{valor:,}".replace(",", "."), va="center", fontsize=6.8)
    eje.set_xlim(0, conteo.max() * 1.22)
    eje.set_xlabel("filas del test")
    eje.tick_params(labelsize=7)
    figura.tight_layout()
    figura.savefig(FIG / "fig_13_test.png")
    plt.close(figura)


if __name__ == "__main__":
    train = cargar_train()
    test = pd.read_csv(RAIZ / "data" / "raw" / "age_test.csv", sep=";")
    figura_01_eda(train, test)
    figura_02_terminos(train)
    figura_03_clusters()
    figura_04_bloques()
    figura_05_miembros()
    figura_07_confusion()
    figura_08_progresion()
    figura_10_familia_nn()
    figura_11_powermean()
    figura_13_test()
    print("figuras del articulo regeneradas en", FIG)
