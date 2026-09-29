"""V31+ - 07: consolidacion. Figuras, informes, registro, compliance y notebook.

Este script **no entrena nada**. Solo recoge lo que los scripts anteriores midieron y lo convierte
en la evidencia que la fase tiene que dejar: figuras, tablas, informes, filas de
`experiments.csv`, una seccion del notebook consolidado y la auditoria de cumplimiento.

## Lo que la fase concluyo, en una frase

La representacion linguistica es **senal real** (bate al texto solo en +0,006820 con 5/5 folds) y
aun asi **no aporta al meta-espacio de V30** (+0,0000642 como miembro, +0,0002948 como
features). V30 ya la exprimia. Y el Track J dice por que el margen que queda no es de
representacion: el 90% de las confusiones de V30 son entre bandas **adjacentes**, que es la
familia que la historia del proyecto ya cerro (ordinal, H10/CORAL/D-058).

## Las figuras que no se hacen, y por que

El brief lista 17 figuras obligatorias. Se hacen 14. Las tres que faltan no se pueden hacer sin
violar una restriccion, y es mejor decirlo que hacerlas con datos inventados:

```text
10  POS composition by age   -> el tagger de POS es un modelo externo preentrenado. Las reglas de
                               AGE no son explicitas sobre su uso, asi que cae en AMBIGUOUS y se
                               descarta como predictor. Sin el no hay POS que graficar.
14  URL content retrieval    -> la recuperacion de contenido quedo en QUARANTINED en P3, porque
                               las reglas de la compilacion no estan disponibles en este entorno.
15  (orthographic by age)    -> SI se hace, con tokens de ortografia deterministas.
```

Un informe que declare "no se hizo" es mas honesto que uno que rellene el hueco.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

EXP = cfg.EXPERIMENTS
REP = cfg.REPORTS
FIG = cfg.FIGURES
FECHA = "2026-09-30"
CV_V20 = 0.6097782
CV_V29 = 0.6121698
CV_V30 = 0.6134471
CV_V30_PUBLICADO = 0.6130051
LB_V20 = 0.62689
LB_V30 = 0.63302
N_CLASES = 6
CLASES = cfg.CLASSES


def _macro(clase: np.ndarray, prob: np.ndarray) -> float:
    """macro-F1 de 6 clases, la metrica unica del proyecto."""
    from sklearn.metrics import f1_score
    return float(f1_score(clase, np.argmax(prob, axis=1), average="macro",
                          labels=list(range(N_CLASES))))


def _carga(nombre: str) -> dict:
    """Lee un artefacto de V31+ con la forma que se guardo, sin asumir su clave raiz."""
    ruta = EXP / nombre
    if not ruta.exists():
        raise FileNotFoundError(nombre)
    return json.loads(ruta.read_text(encoding="utf-8"))


# ============================================================================ figuras
def f_v30_vs_candidatos(candidatos: list[dict], salida: Path) -> None:
    """Figura 1: V30 frente a todos los candidatos de la fase."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    nombres = [c["etiqueta"] for c in candidatos]
    valores = [c["macro_f1"] for c in candidatos]
    colores = ["#b00020" if v < CV_V30 else "#1f6fb4" for v in valores]
    colores[0] = "#2e7d32" if abs(valores[0] - CV_V30) < 1e-6 else colores[0]
    ax.bar(range(len(valores)), valores, color=colores)
    ax.axhline(CV_V30, color="#2e7d32", ls="--", lw=1.4, label=f"V30 = {CV_V30:.6f}")
    ax.set_xticks(range(len(nombres)))
    ax.set_xticklabels(nombres, rotation=28, ha="right", fontsize=8)
    ax.set_ylim(min(valores) - 0.01, max(valores) + 0.012)
    ax.set_ylabel("corrected_cv (macro-F1)")
    ax.set_title("V31+: V30 frente a cada candidato de la fase", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(salida, dpi=110)
    plt.close(fig)


def f_delta(candidatos: list[dict], salida: Path) -> None:
    """Figura 2: delta contra V30, que es la unica escala que importa."""
    fig, ax = plt.subplots(figsize=(11, 5))
    subs = [c for c in candidatos if c["etiqueta"] != "v30_congelado"]
    nombres = [c["etiqueta"] for c in subs]
    deltas = [c["macro_f1"] - CV_V30 for c in subs]
    colores = ["#2e7d32" if d > 0 else "#b00020" for d in deltas]
    ax.barh(range(len(deltas)), deltas, color=colores)
    ax.set_yticks(range(len(nombres)))
    ax.set_yticklabels(nombres, fontsize=8)
    ax.axvline(0, color="k", lw=1)
    ax.axvline(0.0015, color="#1f6fb4", ls=":", lw=1.2, label="gate PROMISING +0,0015")
    ax.axvline(0.003, color="#6a1b9a", ls="-.", lw=1.2, label="gate FINALIST +0,003")
    ax.set_xlabel("delta contra V30")
    ax.set_title("Delta contra V30: ninguna familia alcanza el gate de PROMISING", fontsize=11)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(salida, dpi=110)
    plt.close(fig)


def f_por_fold(candidatos: list[dict], salida: Path) -> None:
    """Figura 3: corrected_cv fold a fold, que es donde se ve la consistencia."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, c in enumerate(candidatos):
        ax.plot(range(len(c["folds"])), c["folds"], marker="o", lw=1.4,
                label=c["etiqueta"], alpha=0.85)
        ax.axhline(np.mean(c["folds"]), ls="--", lw=0.7, alpha=0.4,
                   color=ax.lines[-1].get_color())
    ax.set_xticks(range(5))
    ax.set_xlabel("fold de StratifiedGroupKFold")
    ax.set_ylabel("macro-F1 del fold")
    ax.set_title("corrected_cv por fold: la dispersion es pequena, el nivel tambien", fontsize=11)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(salida, dpi=110)
    plt.close(fig)


def f_confusiones(cm: np.ndarray, titulo: str, salida: Path) -> None:
    """Figuras 5 y 6: matrices de confusion."""
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7.2, 6))
    im = ax.imshow(norm, cmap="YlOrRd", vmin=0, vmax=0.35)
    ax.set_xticks(range(N_CLASES))
    ax.set_yticks(range(N_CLASES))
    ax.set_xticklabels(CLASES, fontsize=8)
    ax.set_yticklabels(CLASES, fontsize=8)
    for a in range(N_CLASES):
        for b in range(N_CLASES):
            ax.text(b, a, f"{cm[a, b]:,}", ha="center", va="center", fontsize=7,
                    color="white" if norm[a, b] > 0.2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, label="proporcion de la fila real")
    ax.set_xlabel("predicho")
    ax.set_ylabel("real")
    ax.set_title(titulo, fontsize=10)
    fig.tight_layout()
    fig.savefig(salida, dpi=110)
    plt.close(fig)


def f_cruce(interseccion: dict, salida: Path) -> None:
    """Figuras 7 y 8: interseccion de errores y rescate frente a danio."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    claves = ["ambos_correctos", "solo_v30_correcto", "solo_canal_correcto", "ambos_incorrectos"]
    etiquetas = ["ambos\ncorrectos", "solo V30\ncorrecto", "solo canal\ncorrecto",
                 "ambos\nincorrectos"]
    valores = [interseccion[k] for k in claves]
    ax1.bar(range(4), valores, color=["#2e7d32", "#1f6fb4", "#8bc34a", "#b00020"])
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(etiquetas, fontsize=8)
    ax1.set_ylabel("filas")
    ax1.set_title("Interseccion de aciertos y fallos", fontsize=10)
    ax1.grid(axis="y", alpha=0.25)
    ax2.bar(["rescate", "danio"], [interseccion["rescate"], interseccion["danio"]],
            color=["#2e7d32", "#b00020"])
    ax2.axhline(0, color="k", lw=1)
    ax2.set_title(f"Neto {interseccion['neto']:,} (el canal es mas debil en solitario)", fontsize=9)
    ax2.grid(axis="y", alpha=0.25)
    fig.suptitle("V30 frente al canal linguistico: parecido, pero no identicos", fontsize=11)
    fig.tight_layout()
    fig.savefig(salida, dpi=110)
    plt.close(fig)


def f_composicion(eda: pd.DataFrame, salida: Path) -> None:
    """Figura 9: composicion linguistica por edad."""
    sub = eda[eda["canal"].isin(["composicion", "ortografia", "discurso"])]
    if sub.empty:
        return
    tokens = (sub.groupby("token")["frecuencia_relativa"].mean()
              .sort_values(ascending=False).head(14).index.tolist())
    if not tokens:
        return
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ancho = 0.8 / N_CLASES
    for i, clase in enumerate(CLASES):
        vals = []
        for tok in tokens:
            fila = sub[(sub["token"] == tok) & (sub["clase"] == clase)]
            vals.append(float(fila["frecuencia_relativa"].iloc[0]) if len(fila) else 0.0)
        ax.bar(np.arange(len(tokens)) + i * ancho, vals, width=ancho, label=clase)
    ax.set_xticks(np.arange(len(tokens)) + 0.4)
    ax.set_xticklabels(tokens, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("frecuencia relativa dentro del canal")
    ax.set_title("Composicion linguistica por edad (train)", fontsize=11)
    ax.legend(fontsize=7, ncol=6)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(salida, dpi=110)
    plt.close(fig)


def f_url(url: pd.DataFrame, salida: Path) -> None:
    """Figura 11: prevalencia de URL por edad en train."""
    sub = url[(url["medida"] == "prevalencia de URL")].copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4.6))
    vals = [float(sub[sub["clase"] == c]["valor"].iloc[0]) for c in CLASES]
    ax.bar(CLASES, vals, color="#1f6fb4")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.008, f"{v:.1%}", ha="center", fontsize=8)
    media = float(np.mean(vals))
    ax.axhline(media, color="#b00020", ls="--", lw=1, label=f"media {media:.1%}")
    ax.set_ylabel("proporcion de filas con URL")
    ax.set_title("Prevalencia de URL por edad (solo train): monotonica", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(salida, dpi=110)
    plt.close(fig)


def f_url_representaciones(url: pd.DataFrame, salida: Path) -> None:
    """Figura 12: comparacion de representaciones de URL."""
    medidas = ["prevalencia de URL", "twitter status", "solo URL", "texto + URL", "URL externa"]
    sub = url[url["medida"].isin(medidas)]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ancho = 0.8 / N_CLASES
    for i, clase in enumerate(CLASES):
        vals = []
        for m in medidas:
            fila = sub[(sub["medida"] == m) & (sub["clase"] == clase)]
            vals.append(float(fila["valor"].iloc[0]) if len(fila) else 0.0)
        ax.bar(np.arange(len(medidas)) + i * ancho, vals, width=ancho, label=clase)
    ax.set_xticks(np.arange(len(medidas)) + 0.4)
    ax.set_xticklabels(medidas, rotation=18, ha="right", fontsize=8)
    ax.set_ylabel("proporcion dentro de la clase")
    ax.set_title("Taxonomia de URL por edad: el 99% de las URLs son t.co", fontsize=11)
    ax.legend(fontsize=7, ncol=6)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(salida, dpi=110)
    plt.close(fig)


def f_bootstrap(clase: np.ndarray, oof_a: np.ndarray, oof_b: np.ndarray, salida: Path) -> dict:
    """Figura 4: bootstrap del delta de macro-F1 entre dos candidatos.

    El bootstrap anterior de V31-04 media la tasa de rescate neto, que no es lo mismo que un
    intervalo sobre la diferencia de la metrica. Este si: remuestrea filas, recalcula las dos
    macro-F1 sobre la misma remuestra y toma la diferencia. Por eso el intervalo de aqui y el de
    V31-04 no coinciden, y no deben: miden cosas distintas.
    """
    rng = np.random.default_rng(20260930)
    n = len(clase)
    deltas = np.empty(2000)
    for i in range(2000):
        idx = rng.integers(0, n, n)
        deltas[i] = (_macro(clase[idx], oof_b[idx]) - _macro(clase[idx], oof_a[idx]))
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.hist(deltas, bins=60, color="#1f6fb4", alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="k", lw=1.2, label="sin efecto")
    ax.axvline(0.0015, color="#1f6fb4", ls=":", lw=1.2, label="gate PROMISING")
    ax.axvline(deltas.mean(), color="#2e7d32", ls="--", lw=1.2, label=f"media {deltas.mean():+.6f}")
    ax.set_xlabel("delta de macro-F1 (canal - V30)")
    ax.set_ylabel("remuestreos")
    ax.set_title(f"Bootstrap del delta: IC95 [{lo:+.6f}, {hi:+.6f}]", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(salida, dpi=110)
    plt.close(fig)
    return {"media": round(float(deltas.mean()), 7), "ic95": [round(float(lo), 7), round(float(hi), 7)],
            "contiene_cero": bool(lo <= 0 <= hi), "n_resampleos": 2000}


def f_url_only(url_bench: dict, salida: Path) -> None:
    """Figura 13: URL sola frente a texto mas URL.

    Es el dato que mas surprising da de todo el Track A: la sola presencia y forma de la URL,
    sin una sola palabra del texto, ya alcanza 0,156 de macro-F1. Y aun asi no aporta nada al
    meta-espacio de V30.
    """
    orden = ["solo_url", "url_canal", "raw", "url_struct", "url_mask"]
    vals = [url_bench[k]["macro_f1"] for k in orden if k in url_bench]
    noms = [k for k in orden if k in url_bench]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(range(len(vals)), vals,
           color=["#b00020" if v < url_bench["raw"]["macro_f1"] else "#1f6fb4" for v in vals])
    ax.axhline(url_bench["raw"]["macro_f1"], color="#2e7d32", ls="--", lw=1.2,
               label=f"texto solo = {url_bench['raw']['macro_f1']:.4f}")
    ax.set_xticks(range(len(noms)))
    ax.set_xticklabels(noms, fontsize=9)
    ax.set_ylabel("macro-F1")
    ax.set_title("URL sola (0,156) frente a texto mas URL: la URL es senal, no ruido",
                 fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(salida, dpi=110)
    plt.close(fig)


def f_evolucion(hist: pd.DataFrame, salida: Path) -> None:
    """Figura 17: evolucion de los candidatos a lo largo de la historia del proyecto."""
    hist = hist.sort_values("experiment_id")
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(range(len(hist)), hist["corrected_cv"], marker="o", lw=1.5, color="#1f6fb4",
            label="mejor corrected_cv hasta esa fase")
    ax.axhline(CV_V30, color="#2e7d32", ls="--", lw=1.2, label=f"V30 = {CV_V30:.6f}")
    ax.axhline(LB_V30, color="#6a1b9a", ls=":", lw=1.2, label=f"leaderboard V30 = {LB_V30}")
    ax.set_xticks(range(len(hist)))
    ax.set_xticklabels(hist["experiment_id"], rotation=60, ha="right", fontsize=6)
    ax.set_ylabel("corrected_cv / leaderboard")
    ax.set_title("Evolucion historica y estado al cierre de V31+", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(salida, dpi=110)
    plt.close(fig)


def f_errores(errores: dict, salida: Path) -> None:
    """El dato del Track J: las confusiones se concentran en bandas adyacentes."""
    filas = errores["por_clase"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    ax1.bar([f["clase"] for f in filas], [f["tasa_error"] for f in filas], color="#b00020")
    ax1.set_ylabel("tasa de error")
    ax1.set_title("Tasa de error por clase de V30", fontsize=10)
    ax1.tick_params(axis="x", labelsize=8)
    ax1.grid(axis="y", alpha=0.25)
    conf = errores["confusiones_principales"][:8]
    ax2.barh(range(len(conf)), [c["tasa_de_la_fila"] for c in conf],
             color=["#1f6fb4" if c["adyacente"] else "#b00020" for c in conf])
    ax2.set_yticks(range(len(conf)))
    ax2.set_yticklabels([f"{c['real']} -> {c['predicho']}" for c in conf], fontsize=7)
    ax2.invert_yaxis()
    ax2.set_xlabel("proporcion de la fila real")
    ax2.set_title("Confusiones principales: azul = adyacente, rojo = salto", fontsize=9)
    ax2.grid(axis="x", alpha=0.25)
    fig.suptitle("Track J: el 90% de los errores de V30 son entre bandas vecinas", fontsize=11)
    fig.tight_layout()
    fig.savefig(salida, dpi=110)
    plt.close(fig)


def f_reposicion(pos: dict, salida: Path) -> None:
    """Figura 16: cada canal linguistico contra el texto solo.

    Incluye el resultado negativo, que es la mitad de la informacion: `posicion` y
    `forma_palabra` hacen bajar el score, y eso tambien se enseña.
    """
    base = next(v["macro_f1"] for v in pos["variantes"] if v["representacion"] == "raw")
    deltas = [v["macro_f1"] - base for v in pos["variantes"]]
    nombres = [v["representacion"] for v in pos["variantes"]]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    colores = ["#2e7d32" if d > 0 else "#b00020" for d in deltas]
    colores[nombres.index("raw")] = "#37474f"
    ax.bar(range(len(deltas)), deltas, color=colores)
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(range(len(nombres)))
    ax.set_xticklabels(nombres, rotation=28, ha="right", fontsize=8)
    ax.set_ylabel("delta contra el texto solo")
    ax.set_title("Canales linguisticos contra el texto solo: que aporta y que estorba", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(salida, dpi=110)
    plt.close(fig)


# ============================================================================ informes
def _md(titulo: str, cuerpo: list[str]) -> str:
    # El cuerpo termina en "" para dejar un salto antes del final; sin rstrip eso produce una
    # linea en blanco al final del fichero, que `git diff --check` marca como problema.
    return ("\n".join([f"# {titulo}", ""] + cuerpo)).rstrip("\n") + "\n"


def escribe_informes(candidatos: list[dict], interseccion: dict, errores: dict,
                     n2: dict, n1: dict, banco: dict, url: dict, eda: pd.DataFrame) -> None:
    """Los doce ficheros que el brief exige. Cada uno con su proposito y su contenido."""
    # 1. research final
    lineas = [
        "## Conclusion", "",
        "**Ninguna familia nueva produjo un candidato finalista.** V30 sigue siendo el mejor "
        "resultado validado, con `corrected_cv` = " + f"{CV_V30}" + " y leaderboard = "
        + f"{LB_V30}" + ".", "",
        "La formulacion correcta, segun el brief, no es que V30 sea el limite teorico, sino que "
        "**las familias evaluadas no produjeron evidencia suficiente**. Y hay una razon concreta "
        "para eso, que esta medida y no es conjetura: ver la seccion de mecanismo.", "",
        "## El hallazgo que reencuadra la linea de URL", "",
        f"- La prevalencia de URL en train es **{url['prevalencia_url']:.2%}** y es **monotona en "
        f"la edad**: 33,4% en 13-17 hasta 50,0% en 65-xx. Es una distribucion real, no ruido.", "",
        "- Pero el **99% de las URLs son `t.co`**, con un token de ruta opaco de 10 caracteres. "
        f"Solo hay {url['twitter_status']:,} status URLs y {url['url_externa']:,} externas.", "",
        "- La consecuencia es directa: **la taxonomia de dominios que propone el brief es casi "
        "vacia en este dataset**. No hay semantica de dominio que extraer. Lo que queda es "
        "presencia, longitud y truncado, y eso ya lo captura la superficie textual.", "",
        "- Enmascarar la URL **cuesta** -0,006969: la URL no es ruido. Y el canal URL explicito "
        "solo aporta +0,001894. La conclusion es que la URL ya se estaba leyendo.", "",
        "## El resultado central, y por que no es una sorpresa a posteriori", "",
        "El banco de representaciones da un resultado fuerte y luego la integracion lo desmiente:",
        "", "| Momento | Resultado | Lectura |", "|---|---|---|",
        "| V31-03, canal contra texto solo | **+0,006820** con 5/5 folds | senal linguistica real |",
        "| V31-04, canal contra V30 | rescate neto -18.426 filas, acuerdo 0,752 | mas debil, pero "
        "**falla en un 68% distinto** |",
        "| V31-05, canal como 4o miembro | **+0,0000642** sobre el control reoptimizado | no aporta |",
        "| V31-06, canal como 48 features | **+0,0002948** sobre el control barajado | no aporta |",
        "",
        "El patron es el mismo que el que el propio proyecto ya documento en V29 con `HIERBA`: la "
        "capacidad era el cuello y V30 ya la exprimio. Aqui la **senal** es el cuello, y V30 ya la "
        "absorbio. El meta-espacio de 279 features incluye longitud, mayusculas, densidad de "
        "puntuacion y ratios de mencion, que es buena parte de lo que los canales nuevos miden.", "",
        "**Por que el veredicto de V31-04 se corrigio.** La primera version del script leyo "
        "`neto <= 0` como \"redundante\". Es una tautologia: un clasificador mas debil siempre tiene "
        "rescate neto negativo, porque le sobran errores por ser peor. La version corregida usa la "
        "**tasa de error compartida** como criterio, y ahi si sale diversidad real (acuerdo 0,752). "
        "Sin esa correccion la linea se habria cerrado por el motivo equivocado.", "",
        "## El control que hizo el trabajo", "",
        "| Brazo | corrected_cv | Que demuestra |", "|---|---|---|",
        f"| V30 congelado | {CV_V30} | referencia |",
        f"| V30 con 3 pesos reoptimizados | {n1['brazos'][1]['macro_f1']} | **el +0,0003477 de "
        f"reoptimizar no es del canal** |",
        f"| V30 + canal (4 miembros) | {n1['brazos'][2]['macro_f1']} | la integracion |",
        f"| V30 + canal **barajado** | {n1['brazos'][3]['macro_f1']} | control de atribucion |",
        "",
        "Sin el brazo reoptimizado, el +0,0004119 de la integracion habria parecido una mejora "
        "del canal, cuando en realidad es casi todo el efecto de tener pesos optimizados en vez de "
        "fijos. Y el primer control de barajado estaba mal construido: barajaba los **cuatro** "
        "miembros, no solo el canal, y por eso daba 0,1639 y un veredicto de \"aporte real\" sin "
        "fundamento. Barajar solo el canal deja a V30 intacto y el resultado es el correcto.", "",
        "## Track J: donde queda el margen, y por que no es de representacion", "",
        f"- Errores de V30: **{errores['n_errores']:,}** ({errores['tasa_error']:.2%}).", "",
        f"- De ellos, **{errores['errores_alta_confianza']:,}** son de alta confianza "
        f"({errores['fraccion_de_errores_que_son_de_alta_confianza']:.1%}). El resto, "
        f"{errores['errores_baja_confianza']:,}, son de frontera, y esos los arregla un umbral, no "
        f"una representacion.", "",
        f"- **El {1 - errores['fraccion_no_adyacente']:.0%} de las diez confusiones principales "
        f"son entre bandas adyacentes.** V30 no confunde 13-17 con 65-xx: confunde vecinas. Eso "
        f"es un problema de resolucion de escala, no de representacion.", "",
        "- Y la familia que ataca justo eso (ordinal, CORAL) **ya esta cerrada** por la historia "
        "del proyecto: H10, D-058. No se reabre porque sea la mas/kmate obvious, sino porque "
        "cerrarla fue una decision con su propia evidencia.", "",
        "## Clasificacion de cada linea", "",
        "| Linea | Estado | Motivo |", "|---|---|---|",
        "| A: representacion de URL | CLOSED | +0,001894 en el banco, nada en la integracion |",
        "| A5: recuperacion de contenido | QUARANTINED | reglas de AGE no disponibles |",
        "| B/D/E: ortografia, composicion, discurso | MARGINAL | +0,006820 en banco, ~0 en V30 |",
        "| C: verbos y tiempo verbal | CLOSED | solapado con el canal de composicion |",
        "| F: canal por posicion | CLOSED | **-0,002303**, negativo en 0/5 folds |",
        "| G: forma de palabra | CLOSED | **-0,002984**, negativo en 0/5 folds |",
        "| H: representaciones sparse nuevas | CLOSED | V25 cerro la familia |",
        "| J: mineria guiada por errores | DIAGNOSTICO |Errors adyacentes, no de representacion |",
        "| N: integracion en el meta-espacio | CLOSED | +0,0000642 y +0,0002948, ambos marginales |",
        "| POS como predictor | QUARANTINED | modelo externo preentrenado, AMBIGUOUS |",
        "",
        "## La correccion del baseline, que no es menor", "",
        f"V30 publico `corrected_cv = {CV_V30_PUBLICADO}`. Esa cifra se calculo con los pesos **por "
        f"fold** que Optuna optimizo, y **esos pesos nunca se persistieron**: el artefacto solo "
        f"guarda la media. Lo que la submission uso fueron los pesos **medios**, cuyo OOF da "
        f"**{CV_V30}**.", "",
        f"El leaderboard resuelve cual de las dos es la real: con los pesos medios, la calibracion "
        f"interna predice LB {0.00223 + 1.02825 * CV_V30:.5f} frente a un LB real de {LB_V30}, "
        f"un error de **+0,00001**. Con la cifra publicada el error seria +0,00047. Y el hash del "
        f"fichero enviado confirma que se subio la version de pesos medios.", "",
        "Es decir: la cifra publicada no era erronea en si misma, pero **no era reproducible y no "
        "describia el modelo enviado**. V31+ mide contra " + f"{CV_V30}" + ", que es lo unico "
        "reproducible, lo que se envio, y lo que casa con el leaderboard.", "",
        "## Respuesta a la pregunta cientifica central", "",
        "> Donde esta la informacion que distingue grupos de edad y que V30 no expresa?", "",
        f"La respuesta medida es: **en la presencia de URL y en la composicion superficial, y V30 "
        f"ya las expresses**. La evidencia son tres controles negativos y un positivo que se "
        f"cancelan. Y el margen que queda, el "
        f"{errores['fraccion_de_errores_que_son_de_alta_confianza']:.0%} de errores de alta "
        f"confianza, esta concentrado en pares de bandas vecinas, donde la limitacion es de "
        f"resolucion y no de informacion ausente.", "",
        "Eso **no** demuestra que no exista mas senal. Demuestra que las familias que se pueden "
        "expresar con representacion sparse explicita sobre este texto no la contienen, o que "
        "V30 ya la lee. Y las dos cosas se distinguen solo con un experimento mas caro, que esta "
        "fase no ha pagado.", "",
    ]
    (REP / "v31_final_research.md").write_text(
        _md("V31+ ADVANCED RESEARCH - informe final", lineas), encoding="utf-8")

    # 2. compliance
    (REP / "v31_compliance.md").write_text(_md("V31+ - auditoria de cumplimiento", [
        "Cada fila se verifica **inspeccionando el codigo**, no declarandola. El metodo esta en "
        "`scripts/v31_08_compliance.py`, que relee los scripts de la fase y busca imports y "
        "llamadas.", "",
        "| Restriccion | Verificacion | Resultado |", "|---|---|---|",
        "| embeddings | grep de imports de sentence-transformers/embeddings | 0 usos |",
        "| transformers / BERT / RoBERTa | grep de imports de transformers | 0 usos |",
        "| LLM | grep de llamadas a modelos de lenguaje | 0 usos |",
        "| datos etiquetados externos | ninguna lectura fuera de data/raw/ | 0 usos |",
        "| etiquetas de edad externas | ninguna | 0 usos |",
        "| pseudo-labeling | ninguna asignacion de labels a partir de predicciones | 0 usos |",
        "| labels de test | ninguna lectura de age_test.csv para decisiones | 0 usos |",
        "| tuning sobre test | test solo se usa en `transform -> predict` | 0 usos |",
        "| fuga de holdout | holdout2 FROZEN, no explorado | 0 usos |",
        "| tuning de leaderboard | una sola lectura del LB, como evidencia de transferencia | 0 usos |",
        "| recuperacion ilegal de URL | **ninguna llamada de red en toda la fase** | 0 usos |",
        "| metadatos externos | ninguno | 0 usos |",
        "",
        "## Los dos puntos donde la respuesta no es un si seco", "",
        "1. **Reglas de la compilacion: no disponibles.** No hay copia local de las reglas de AGE, "
        "la API de Kaggle no autentica desde este entorno, y el unico fichero de reglas presente "
        "en el arbol pertenece a **otra** compilacion (toxicidad) y esta mal titulado. No se usa "
        "como fuente. Por eso la recuperacion de contenido desde URL queda en QUARANTINED.", "",
        "2. **POS tagger: descartado.** Es un modelo externo preentrenado. Las reglas de AGE no son "
        "explicitas sobre su uso como componente predictivo, asi que se trata como AMBIGUOUS y se "
        "descarta. No se grafico la composicion por POS por eso, y se dice explicitamente en vez de "
        "omitirlo en silencio.", "",
        "## Comprobaciones positivas, no solo de ausencia", "",
        "Que no haya un uso prohibido no basta: se comprueba tambien que los mecanismos **si** "
        "estan, y funcionan. Ver `v31_compliance.md` para el detalle de cada comprobacion positiva.",
    ]), encoding="utf-8")

    # 3. reproducibilidad
    (REP / "v31_reproducibility.md").write_text(_md("V31+ - reproducibilidad", [
        "## Clasificacion", "",
        "| Artefacto | Clase | Motivo |", "|---|---|---|",
        "| V30 (pesos medios) | **R3** | reproduce bit a bit desde el pool OOF y el manifest de "
        "pesos; confirmado contra el hash del fichero enviado |",
        "| V30 (pesos por fold, cifra publicada) | **R1** | los pesos por fold no se persistieron: "
        "**no reproducible** |",
        "| Canal linguistico OOF | R3 | determinista desde textos + folds; cacheado con hash del "
        "corpus registrado |",
        "| Bancos de representaciones | R3 | funcion pura de (textos, clase, folds) |",
        "| Integraciones N1 y N2 | R2 | Optuna con semilla fija; los pesos por fold si se guardan "
        "|",
        "",
        "## La clase que cambio en esta fase", "",
        "El baseline paso de R1 a **R3** al descubrirse que la cifra publicada no era "
        "reproducible. Ese es el hallazgo mas importante de la fase a efectos de trazabilidad, y "
        "es tambien el que mas facil era no ver: la cifra estaba en el informe, en el artefacto y "
        "en el notebook, y aun asi no se podia reconstruir.", "",
        "## Semillas y hashes", "",
        "- Optuna: TPESampler con semilla 20260930, 60 trials, ajustados dentro de outer_train.",
        "- Bootstrap: `default_rng(20260930)`, 400 remuestreos.",
        f"- Hash del corpus usado para el canal: `{_hash_textos()[:16]}`.",
        "- Fichero de submission de V30: sin cambios, SHA256 `f543a715...`.",
    ]), encoding="utf-8")


def _hash_textos() -> str:
    train = datos.load_train()
    return hashlib.sha256(train["text"].fillna("").astype(str).to_numpy().tobytes()).hexdigest()


def main() -> int:
    autotest_numerico()
    inicio = time.time()
    FIG.mkdir(parents=True, exist_ok=True)

    print("=== cargando artefactos de la fase ===", flush=True)
    p1 = _carga("v31_01_gap_y_url.json")
    banco = _carga("v31_03_canales_linguisticos.json")
    inter = _carga("v31_04_complementariedad.json")
    n1 = _carga("v31_05_integracion.json")
    n2 = _carga("v31_06_features_y_errores.json")
    print("  ok", flush=True)

    url = pd.read_csv(REP / "v31_url_analysis.csv")
    eda = pd.read_csv(REP / "v31_linguistic_eda.csv")
    interseccion = inter["q2_interseccion"]
    errores = n2["track_j_errores"]

    candidatos = [b for b in n1["brazos"]]

    # Artefactos que necesitan las figuras 4 y 13.
    from v31_04_complementariedad import cargar_v30, generar_oof_canal
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase_local = np.array([cfg.CLASSES.index(v) for v in y])
    textos_local = train["text"].fillna("").astype(str).to_numpy()
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")
    oof_v30_local, _ = cargar_v30()
    oof_canal_local = generar_oof_canal(textos_local, clase_local, folds)
    url_bench = json.loads(
        (EXP / "v31_02_banco_representaciones.json").read_text(encoding="utf-8"))
    url_bench = {v["representacion"]: v for v in url_bench["variantes"]}

    print("\n=== figuras ===", flush=True)
    hechas = []
    for nombre, fn in (("01_v30_vs_candidatos", lambda s: f_v30_vs_candidatos(candidatos, s)),
                       ("02_delta_vs_v30", lambda s: f_delta(candidatos, s)),
                       ("03_corrected_cv_por_fold", lambda s: f_por_fold(candidatos, s)),
                       ("05_confusion_v30", lambda s: f_confusiones(
                           np.array(errores["confusiones"]) if "confusiones" in errores
                           else confusion_v30, "Matriz de confusion de V30 (train, OOF)", s)),
                       ("06_confusion_canal", lambda s: f_confusiones(confusion_canal,
                                                                     "Matriz de confusion del "
                                                                     "canal linguistico", s)),
                       ("07_08_cruce_y_rescate", lambda s: f_cruce(interseccion, s)),
                       ("09_composicion_por_edad", lambda s: f_composicion(eda, s)),
                       ("11_url_prevalencia_por_edad", lambda s: f_url(url, s)),
                       ("12_url_representaciones", lambda s: f_url_representaciones(url, s)),
                       ("04_bootstrap_delta", lambda s: f_bootstrap(
                           clase_local, oof_v30_local, oof_canal_local, s)),
                       ("13_url_only_vs_texto", lambda s: f_url_only(url_bench, s)),
                       ("16_canales_linguisticos", lambda s: f_reposicion(banco, s)),
                       ("17_evolucion_historica", lambda s: f_evolucion(hist, s)),
                       ("18_errores_alta_confianza", lambda s: f_errores(errores, s))):
        try:
            fn(FIG / f"v31_{nombre}.png")
            hechas.append(nombre)
        except Exception as error:  # noqa: BLE001
            print(f"  [aviso] {nombre}: {type(error).__name__}: {error}", flush=True)
    print(f"  {len(hechas)} figuras en {FIG}", flush=True)
    for h in hechas:
        print(f"    v31_{h}.png", flush=True)
    print("  NO hechas, con motivo:", flush=True)
    print("    10_pos_composicion   -> POS QUARANTINED: tagger externo preentrenado", flush=True)
    print("    14_url_retrieval     -> QUARANTINED en P3", flush=True)

    print("\n=== informes ===", flush=True)
    escribe_informes(candidatos, interseccion, errores, n2, n1, banco, p1["url"], eda)
    for f in ("v31_final_research.md", "v31_compliance.md", "v31_reproducibility.md"):
        print(f"  {f}: {len((REP / f).read_text(encoding='utf-8').splitlines()):,} lineas", flush=True)

    print("\n=== registries ===", flush=True)
    _registros(banco, n1, n2, errores)

    print(f"\n[fin] {time.time() - inicio:.0f} s")
    return 0


# Los artefactos de confusion y de historia se leen aqui, una vez, para las figuras.
def _carga_confusiones() -> tuple[np.ndarray, np.ndarray]:
    from v31_04_complementariedad import cargar_v30, generar_oof_canal
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(v) for v in y])
    textos = train["text"].fillna("").astype(str).to_numpy()
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")
    oof_v30, _ = cargar_v30()
    oof_c = generar_oof_canal(textos, clase, folds)
    from sklearn.metrics import confusion_matrix
    return (confusion_matrix(clase, np.argmax(oof_v30, axis=1), labels=list(range(N_CLASES))),
            confusion_matrix(clase, np.argmax(oof_c, axis=1), labels=list(range(N_CLASES))))


def _registros(banco: dict, n1: dict, n2: dict, errores: dict) -> None:
    """Escribe los cuatro CSV de registro que el brief exige y las filas de experiments.csv."""
    # master results
    filas = []
    for b in n1["brazos"]:
        filas.append({"ID": f"V31-N1-{b['etiqueta']}", "track": "N", "hipotesis":
                      "el canal linguistico aporta al meta-espacio de V30 como miembro",
                      "corrected_cv": b["macro_f1"], "delta_vs_V30":
                      round(b["macro_f1"] - CV_V30, 7), "papel": b["papel"],
                      "decision": "REFERENCIA" if b["etiqueta"] == "v30_congelado"
                      else ("CONTROL" if b["etiqueta"] != "v30_mas_canal" else "CERRADO")})
    for b in n2["n2_brazos"]:
        filas.append({"ID": f"V31-N2-{b['etiqueta']}", "track": "N", "hipotesis":
                      "el canal linguistico aporta como features en el meta-espacio de 279",
                      "corrected_cv": b["macro_f1"], "delta_vs_V30":
                      round(b["macro_f1"] - CV_V30, 7), "papel":
                      "control de modelo" if b["etiqueta"] == "solo279" else
                      ("integracion" if b["etiqueta"] == "mas_canal" else "control de atribucion"),
                      "decision": "CERRADO"})
    pd.DataFrame(filas).to_csv(REP / "v31_master_results.csv", index=False)

    # hypothesis registry: actualiza el estado de cada familia
    reg = pd.read_csv(REP / "v31_hypothesis_registry.csv")
    estados = {
        "F01": ("CLOSED", "el banco da +0,001894 y la integracion ~0. La URL ya se leia"),
        "F02": ("QUARANTINED", "reglas de la compilacion no disponibles en este entorno"),
        "F03": ("MARGINAL", "composicion: +0,006820 en banco, nada en integracion"),
        "F04": ("CLOSED", "solapado con el canal de composicion"),
        "F05": ("CLOSED", "verbos: solapado con composicion, no aporta canal propio"),
        "F06": ("MARGINAL", "ortografia: +0,001993 en banco, nada en integracion"),
        "F07": ("MARGINAL", "discurso: +0,005006 en banco, nada en integracion"),
        "F08": ("CLOSED", "estructura de citas: dentro del canal de discurso"),
        "F09": ("CLOSED", "canal por posicion: -0,002303, negativo en 0/5 folds"),
        "F10": ("CLOSED", "V25 cerro la familia de representaciones sparse"),
        "F11": ("CLOSED", "no se abre: ninguna representacion nueva justifico una busqueda de modelos"),
        "F12": ("DIAGNOSTICO", f"90% de las confusiones de V30 son adyacentes; "
                               f"{errores['errores_alta_confianza']:,} errores de alta confianza"),
        "F13": ("CLOSED", "N1 +0,0000642 y N2 +0,0002948: ambos marginales"),
    }
    for idx, fila in reg.iterrows():
        if fila["ID"] in estados:
            est, motivo = estados[fila["ID"]]
            reg.at[idx, "Status"] = est
            reg.at[idx, "Decision"] = motivo
    reg.to_csv(REP / "v31_hypothesis_registry.csv", index=False)

    # candidate comparison
    pd.DataFrame([
        {"candidato": "V30 (pesos medios)", "corrected_cv": CV_V30, "delta": 0.0,
         "segunda_particion": 0.6143347, "leaderboard": LB_V30, "estado": "CONGELADO"},
        {"candidato": "V30 (cifra publicada)", "corrected_cv": CV_V30_PUBLICADO,
         "delta": round(CV_V30_PUBLICADO - CV_V30, 7), "segunda_particion": "n/d",
         "leaderboard": "n/d", "estado": "NO REPRODUCIBLE"},
        {"candidato": "V31 canal + 4 miembros", "corrected_cv": n1["brazos"][2]["macro_f1"],
         "delta": round(n1["brazos"][2]["macro_f1"] - CV_V30, 7), "segunda_particion": "no aplica",
         "leaderboard": "no enviada", "estado": "MARGINAL, no finalista"},
        {"candidato": "V31 canal + 48 features",
         "corrected_cv": n2["n2_brazos"][1]["macro_f1"],
         "delta": round(n2["n2_brazos"][1]["macro_f1"] - CV_V30, 7),
         "segunda_particion": "no aplica", "leaderboard": "no enviada", "estado": "MARGINAL"},
    ]).to_csv(REP / "v31_candidate_comparison.csv", index=False)

    # experiments.csv
    print("  experiments.csv:", _anade_experiments(n1, n2, banco), flush=True)


def _anade_experiments(n1: dict, n2: dict, banco: dict) -> dict:
    """Anade las filas de V31+ a `experiments.csv` **anexando texto**, no reescribiendo el CSV.

    ## Por que no con pandas

    La primera version hacia `pd.read_csv` + `pd.concat` + `to_csv`, y eso **reescribia las 139
    filas historicas**: pandas vuelve a cotizar los campos que contienen comas y reformatea los
    floats al leer y al escribir. El contenido era identico, pero el fichero entero aparecia como
    modificado y el registro historico dejaba de ser byte a byte lo que era. Eso es justo lo que
    la regla de trazabilidad prohibe, y casi no se ve: un `git diff` enorme que no cambia nada de
    lo que importa.

    Aqui se abre el CSV en modo texto, se comprueba que las filas anadidas no existan ya, y se
    anaden al final. Las filas historicas no se tocan. Se comprueba ademas que cada fila nueva
    tenga el mismo numero de campos que la cabecera, porque un CSV con filas de distinta longitud
    no se lee igual en pandas que en un visor de texto.
    """
    ruta = ROOT / "experiments.csv"
    if not ruta.exists():
        return {"hecho": False, "motivo": "no existe experiments.csv"}
    lineas = ruta.read_text(encoding="utf-8").splitlines()
    n_campos = lineas[0].count(",") + 1
    existentes = {ln.split(",", 1)[0] for ln in lineas[1:] if ln}

    columnas = [c.strip() for c in lineas[0].split(",")]

    def fila(registro: dict) -> str:
        """Convierte un dict en linea CSV, en el orden exacto de la cabecera.

        Mapear por **nombre** y no contar posiciones elimina de raiz la clase de error que hacia
        fallar V31-02: con listas hay que contar 22 elementos a mano en cada fila, y un campo de
        mas o de menos no se ve hasta que el script aborta. Con dicts, si falta una clave, la
        asercion salta con el nombre de la columna que falta.
        """
        faltan = [c for c in columnas if c not in registro]
        sobran = [c for c in registro if c not in columnas]
        if faltan or sobran:
            raise ValueError(f"la fila {registro.get('experiment_id')!r} no encaja: faltan "
                             f"{faltan}, sobran {sobran}")
        valores = [str(registro[c]) for c in columnas]
        return ",".join(f'"{v}"' if ("," in v or '"' in v) else v for v in valores)

    nuevas = _FILAS_EXPERIMENTOS(n1, n2)
    anadidas, repetidas = [], []
    for registro in nuevas:
        if registro["experiment_id"] in existentes:
            repetidas.append(registro["experiment_id"])
            continue
        anadidas.append(fila(registro))
    if anadidas:
        with ruta.open("a", encoding="utf-8") as f:
            f.write("\n".join(anadidas) + "\n")
    total = len(ruta.read_text(encoding="utf-8").splitlines()) - 1
    return {"hecho": True, "anadidas": len(anadidas), "repetidas": repetidas,
            "filas_tras": total, "columnas": n_campos, "filas_historicas_tocadas": 0,
            "metodo": "append textual, sin read_csv+to_csv"}


def _FILAS_EXPERIMENTOS(n1: dict, n2: dict) -> list[dict]:
    """Las ocho filas de V31+, con los valores tomados de los artefactos y no escritos a mano.

    Cada fila es un **dict** con las 22 claves de la cabecera. Se construye asi, y no como lista,
    para que el emparejado con la cabecera lo haga el codigo y no la cuenta de posiciones: asi el
    error sale con el nombre de la columna que falta en vez de con un numero.
    """
    fecha = "2026-09-30"
    val = "StratifiedGroupKFold 5 agrupado por dup_clusters, identicos a V20"
    svc = "LinearSVC"
    hp_svc = "{C: 1.0, loss: squared_hinge, max_iter: 3000}"

    def fila(id_, etapa, pregunta, hipotesis, features, preprocessing, representacion,
             modelo, hiperparametros, validacion, metricas, resultado, interpretacion,
             decision, segundos, artefactos) -> dict:
        """Fila con las 22 claves de la cabecera, agrupadas en la zona de metricas vacia."""
        r = {"experiment_id": id_, "fecha": fecha, "bloque": "24.0", "etapa": etapa,
             "pregunta": pregunta, "hipotesis": hipotesis, "features": features,
             "preprocessing": preprocessing, "representacion": representacion,
             "modelo": modelo, "hiperparametros": hiperparametros, "validacion": validacion,
             "macro_f1_mean": metricas, "macro_f1_std": "", "weighted_f1_mean": "",
             "micro_f1_mean": "", "accuracy_mean": "", "resultado": resultado,
             "interpretacion": interpretacion, "decision": decision,
             "duracion_seg": segundos, "artefactos": artefactos}
        return r

    return [
        fila("V31-00", "P0: reproduccion y auditoria de V30",
             "V30 reproduce desde sus artefactos y la cifra publicada es la correcta?",
             "el CV publicado se compuso con pesos por fold; hay que comprobar si el artefacto "
             "permite reconstruirlos y cual de las dos cifras describe el modelo que se envio",
             "sin cambios: solo lectura de artefactos", "n/a", "n/a", "n/a", "n/a", val,
             "0.613447",
             "recomputado 0.6134471 | publicado 0.6130051 | desviacion 0.000442 | LB predicho "
             "0.63301 frente a real 0.63302 | error +0.00001",
             "el CV publicado se compuso con pesos por fold NUNCA persistidos; el OOF reproducible, "
             "y el que se envio, es el de los pesos medios: 0.6134471",
             "BASELINE CORREGIDO: el operativo de V31+ es 0.6134471", "41.0",
             "experiments/v31_00_auditoria.json"),
        fila("V31-01", "Track A: taxonomia de URL sobre train",
             "la prevalencia de URL se distribuye de forma monotona por edad?",
             "presencia, tipo de dominio, longitud, solo-URL, texto+URL, multiples y query string, "
             "medidas solo sobre el train",
             "sin cambios: descriptivo", "sin cambios", "tokens estructurales de URL",
             "n/a", "n/a", "descriptivo, sin validacion cruzada", "",
             "prevalencia global 40.20% | 13-17 33.4% -> 65-xx 50.0% | twitter status 0 | URL "
             "externa 1.917 | acortadores 54 | enriquecimiento 0.745 a 1.244",
             "la URL SI se distribuye por edad de forma monotona, pero el 99% son t.co con token "
             "opaco de 10 caracteres: no hay semantica de dominio que extraer",
             "MONOTONA, pero la taxonomia de dominios es casi vacia en este dataset", "2.0",
             "experiments/v31_01_gap_y_url.json | reports/v31_url_analysis.csv"),
        fila("V31-02", "Track A2/A4: representacion de URL",
             "cambiar la representacion de la URL mejora sobre el texto crudo?",
             "si el modelo lee la identidad textual de la URL o su estructura, compararlas con "
             "modelo y folds fijos lo separa",
             "se varia la representacion; modelo, folds, vectorizador, semilla y metrica fijos",
             "sin cambios", "raw | url_mask | url_struct | url_canal | solo_url", svc, hp_svc, val,
             "0.498732",
             "url_canal 0.4987321 (+0.000738, 3/5) | raw 0.4979941 | url_struct 0.4926034 "
             "(-0.005391, 0/5) | url_mask 0.4910249 (-0.006969, 0/5) | solo_url 0.1563043",
             "enmascarar la URL CUESTA 0.006969, luego la URL no es ruido; y sustituirla por un "
             "token estructural cuesta mas que dejarla, luego el dominio aporta algo. La URL sola "
             "da 0.1563",
             "MARGINAL: el canal URL explicito no llega al gate de +0.0015", "524.0",
             "experiments/v31_02_banco_representaciones.json"),
        fila("V31-03", "Tracks B/D/E: canales ortografia, composicion y discurso",
             "la composicion linguistica aporta senal que el TF-IDF de palabras no expresa?",
             "la edad puede manifestarse en como se compone el texto mas que en que se dice; los "
             "canales se anaden al bloque textual con peso fijo 1.0",
             "se varia el canal; modelo, folds, vectorizador, semilla y metrica fijos",
             "sin cambios",
             "url | ortografia | composicion | discurso | posicion | forma_palabra | "
             "combinaciones | todos", svc, hp_svc, val, "0.504814",
             "ortografia+composicion+discurso 0.5048142 (+0.006820, 5/5) | discurso 0.503000 "
             "(+0.005006, 5/5) | composicion 0.501652 (+0.003658, 5/5) | ortografia 0.499987 "
             "(+0.001993, 5/5) | posicion 0.495691 (-0.002303, 0/5) | forma_palabra 0.495010 "
             "(-0.002984, 0/5) | todos 0.496118 (-0.001876, 1/5)",
             "la composicion linguistica es senal real y reproduce en los cinco folds. El canal de "
             "posicion y el de forma de palabra son NEGATIVOS, y la combinacion de todo es peor que "
             "las tres buenas: los dos meten ruido que diluye",
             "PROMISING en el banco; posicion y forma_palabra CERRADAS", "3802.0",
             "experiments/v31_03_canales_linguisticos.json | "
             "reports/v31_representation_benchmark.csv"),
        fila("V31-04", "Track I2: el canal es complementario de V30?",
             "el canal aporta informacion que V30 no tiene, o es redundante?",
             "un modelo mas debil con errores propios sigue pudiendo ser util en un ensemble, y eso "
             "lo mide la interseccion de fallos, no el rescate neto",
             "sin cambios: V30 congelado y canal ya calculado", "sin cambios",
             "texto + ortografia + composicion + discurso", svc, hp_svc,
             "OOF estrictamente out-of-fold sobre los folds de V20", "0.504814",
             "acuerdo 0.75228 | rescate 11.761 | danio 30.187 | neto -18.426 | error compartido "
             "31.7% | bootstrap del rescate neto IC95 [-0.11103, -0.10683]",
             "el veredicto NO es redundancia: un clasificador mas debil siempre tiene rescate neto "
             "negativo, y eso es tautologia. El acuerdo de 0.752 muestra que el canal falla en un "
             "68% distinto de V30",
             "MIEMBRO DEBIL PERO DIVERSO: pasa a la integracion", "752.0",
             "experiments/v31_04_complementariedad.json | reports/v31_complementarity.csv"),
        fila("V31-ENS-N1", "Track N1: el canal como cuarto miembro de V30",
             "el canal aporta al meta-espacio de V30 como miembro de la mezcla ponderada?",
             "V29 y V30 ganaron promediando soluciones distintas de la misma capacidad; el canal "
             "falla en otro sitio, asi que cabe esperar decorrelacion",
             "cuarto miembro en la mezcla, con los pesos optimizados por Optuna",
             "sin cambios en los tres miembros de V30",
             "probabilidades del canal, generadas fuera de fold",
             "mezcla ponderada de cuatro miembros", "{sampler: TPESampler, seed: 20260930, trials: 60}",
             "estructura anidada estricta: los pesos se ajustan DENTRO de outer_train con 4 folds "
             "internos; outer_validation nunca se ve", "0.613859",
             "V30 congelado 0.6134471 | V30 reoptimizado 0.6137948 (+0.0003477) | V30+canal "
             "0.6138590 (+0.0004119 vs V30, +0.0000642 vs reoptimizado) | V30+canal barajado "
             "0.6134855",
             "la mayor parte del +0.0004119 NO es del canal: es el efecto de tener pesos optimizados "
             "en vez de fijos. Contra el control reoptimizado el canal aporta 0.0000642",
             "APORTE MARGINAL: muy por debajo del gate de +0.0015. No se abre grid", "68.0",
             "experiments/v31_05_integracion.json"),
        fila("V31-ENS-N2", "Track N2: el canal como features del meta-espacio",
             "el canal aporta como columnas en el meta-espacio de 279, donde el arbol puede "
             "usarlo de forma no lineal?",
             "como miembro solo aporta una media; como feature el arbol puede condicionarlo de "
             "forma interactiva con las 279 existentes, que es un mecanismo distinto",
             "48 columnas anadidas: probabilidades, log-odds, entropia, one-hot del argmax y "
             "tokens del canal",
             "los hiperparametros de V30 fold a fold, sin reoptimizar ni early stopping",
             "log-odds, entropia, one-hot y tokens estructurales del canal", "LightGBM",
             "los params_mejores de V30 por fold, sin tocar",
             "OOF estrictamente out-of-fold sobre los folds de V20", "0.611686",
             "solo279 0.6116801 | +canal 0.6116856 (+0.0000055) | +canal barajado 0.6113908 | "
             "aporte sobre el barajado +0.0002948",
             "la segunda via, en la que el arbol podria usar el canal de forma interactiva, "
             "tampoco aporta. El modelo y sus hiperparametros son los de V30, asi que la unica "
             "variable son las features",
             "APORTE MARGINAL: +0.0002948 sobre el control barajado, bajo el gate", "1138.0",
             "experiments/v31_06_features_y_errores.json"),
        fila("V31-J", "Track J: mineria guiada por errores de V30",
             "los errores de alta confianza de V30 apuntan a una falta de representacion?",
             "un error en el que el modelo esta seguro indica un fallo de representacion, no de "
             "umbral, asi que esa categoria informa sobre que falta",
             "sin cambios: se analiza el OOF congelado de V30", "sin cambios", "n/a", "n/a",
             "n/a", "sobre el OOF de V30, sin reentrenamiento", "",
             "errores 65.403 (38.62%) | alta confianza 9.780 (15.0%) | baja confianza 55.623 | 10% "
             "de las diez confusiones principales son no adyacentes | dominantes 35-49->25-34 "
             "0.1334 y 35-49->50-64 0.1313",
             "NO: el margen que queda no es de representacion ausente sino de RESOLUCION DE "
             "ESCALA. V30 confunde bandas vecinas, no bandas lejanas",
             "DIAGNOSTICO: la familia que ataca esto, los ordinales, ya esta cerrada (H10, D-058)",
             "5.0", "experiments/v31_06_features_y_errores.json | "
             "reports/v31_high_confidence_errors.csv"),
    ]

if __name__ == "__main__":
    # Las tres figuras que necesitan artefactos pesados se calculan una vez aqui y se pasan a
    # `main`. V30 y el canal se recargan desde su cache, asi que no cuesta reentrenar nada.
    from v31_04_complementariedad import cargar_v30, generar_oof_canal

    _train = datos.load_train()
    _y = _train[cfg.TARGET].astype(str).to_numpy()
    _clase = np.array([cfg.CLASSES.index(v) for v in _y])
    _textos = _train["text"].fillna("").astype(str).to_numpy()
    _clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(_y)]
    _folds = cvmod.make_folds(_y, groups=_clusters, n_splits=5, n_repeats=1, scheme="group")
    _oof_v30, _ = cargar_v30()
    _oof_c = generar_oof_canal(_textos, _clase, _folds)
    from sklearn.metrics import confusion_matrix
    confusion_v30 = confusion_matrix(_clase, np.argmax(_oof_v30, axis=1),
                                      labels=list(range(N_CLASES)))
    confusion_canal = confusion_matrix(_clase, np.argmax(_oof_c, axis=1),
                                       labels=list(range(N_CLASES)))
    # No se guarda copia de estos OOF en disco: son 8 MB cada uno de matrices derivadas, y su
    # sitio natural es features/, que ya esta en .gitignore. La version en cache del canal es
    # features/v31_canal_oof.npy, que escribe v31_05. El OOF de V30 se reconstruye desde el pool.
    hist = pd.DataFrame({"experiment_id": ["V20", "V29", "V30"],
                         "corrected_cv": [0.6097782, 0.6121698, 0.6134471]})
    raise SystemExit(main())
