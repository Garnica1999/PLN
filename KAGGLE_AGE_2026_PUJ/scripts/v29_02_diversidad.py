"""V29 - 02 - P2: analisis de diversidad OOF sobre los 51 modelos historicos.

V29 se juega en esta tabla. La hipotesis central dice que V20 podria no estar limitado por la
capacidad de sus miembros, sino por la **falta de diversidad util** entre sus errores y por una
combinacion de pesos fija. Este script mide esa diversidad, sin entrenar nada.

## Las cinco medidas, y por que cinco

Una sola correlacion no alcanza, y puede actively mislead:

| Medida | Que responde |
|---|---|
| `corr_probabilidades` | Cuanto se parecen las distribuciones completas |
| `corr_errores` | Cuanto se parecen **los fallos**, que es lo que importa para un ensemble |
| `tasa_discrepancia` | Con que frecuencia difieren las decisiones |
| `rescate_unico` | Cuantas filas corrige este modelo y V20 no |
| `dano_unico` | Cuantas filas rompe este modelo que V20 acertaba |

`rescate_unico` alto con `corr_errores` bajo es la señal que_busca V29. `dano_unico` alto la
anula. Por eso se muestran **todos los componentes y no un score unico**: un numero resumen
ocultaria justo la tensión que hay que juzgar.

## La referencia: la diversidad interna de V20

V20 es un promedio de cuatro cuartos. Se mide la pairwise entre ellos. Si V20 ya tiene members
diverses, la hipotesis de V29 pierde fuerza; si losmembers son casi identicos, la hay gana. Esa
es la calibracion de todo lo demas, y se calcula primero.

## Lo que este script NO hace

No entrena, no optimiza pesos, no lee test, no lee leaderboard. Es medicion descriptiva sobre
artefactos congelados.
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
from sklearn.metrics import f1_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

RESULTADO = cfg.EXPERIMENTS / "v29_02_diversidad.json"
DIVERSIDAD = cfg.REPORTS / "v29_model_diversity.csv"
CONTRA_V20 = cfg.REPORTS / "v29_diversity_vs_v20.csv"
FIG_HEAT = cfg.FIGURES / "v29_model_diversity_heatmap.png"
FIG_RESCATE = cfg.FIGURES / "v29_rescate_vs_dano.png"
FIG_CV_DIV = cfg.FIGURES / "v29_cv_vs_diversidad.png"
N_CLASES = 6
CV_V20 = 0.6097782
#: Los cuatro cuartos de V20. Se usan como referencia de diversidad interna.
#: Los cuatro cuartos de V20. La cadena **no** se toma de un artefacto suelto: se reconstruye desde
#: `v20_f3_oof_cumuladas.npy` con la misma monotonia por maximo acumulado y las mismas diferencias
#: que usa el champion.Tomar `v22_seis_compartida` aqui seria un error: ese artefacto tiene el
#: argmax identico al arbol de V19, no al de la cadena, y ocultaria la diversidad real del cuarto.
CUARTOS_V20 = {
    "V20_logreg": "v19_b1_oof_logreg.npy",
    "V20_mlp": "v19_b1_oof_mlp.npy",
    "V20_arbol": "v19_oof_arbol_5folds.npy",
    "V20_cadena": "__reconstruida__",
}
V20 = "v20_f3b_oof_candidato.npy"


def _reconstruye_cadena() -> np.ndarray:
    """La sexta clase del cuarto de V20, reconstruida exactamente como la construye el champion."""
    acumuladas = np.load(cfg.FEATURES / "v20_f3_oof_cumuladas.npy")
    n_umbral = acumuladas.shape[1]
    mono = acumuladas.copy()
    for k in range(1, n_umbral):
        mono[:, k] = np.maximum(mono[:, k], mono[:, k - 1])
    seis = np.zeros((len(mono), n_umbral + 1))
    seis[:, 0] = mono[:, 0]
    for k in range(1, n_umbral):
        seis[:, k] = mono[:, k] - mono[:, k - 1]
    seis[:, -1] = 1.0 - mono[:, n_umbral - 1]
    seis = np.clip(seis, 0.0, None)
    return seis / np.clip(seis.sum(axis=1, keepdims=True), 1e-9, None)


def _cargar(nombre: str) -> np.ndarray:
    return np.load(cfg.FEATURES / nombre, allow_pickle=False)


def _pareja(a_correcto: np.ndarray, b_correcto: np.ndarray) -> dict:
    """Las cuatro celdas de una confusion de acierto, mas el solapamiento."""
    ac = int((a_correcto & b_correcto).sum())
    ad = int((a_correcto & ~b_correcto).sum())
    da = int((~a_correcto & b_correcto).sum())
    dd = int((~a_correcto & ~b_correcto).sum())
    return {"ambos_acierto": ac, "a_acierta_b_falla": ad, "b_acierta_a_falla": da,
            "ambos_fallan": dd}


def main() -> int:
    """Calcula la matriz de diversidad completa y la referencia interna de V20."""
    autotest_numerico()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(v) for v in y])
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")

    inventario = pd.read_csv(cfg.REPORTS / "v29_model_inventory.csv")
    reutilizables = inventario[inventario["reutilizable"]]["artefacto"].tolist()
    oof_v20 = _cargar(V20)
    pred_v20 = np.argmax(oof_v20, axis=1)
    bien_v20 = pred_v20 == clase
    print(f"[P2] {len(reutilizables)} modelos reutilizables | V20 CV "
          f"{f1_score(clase, pred_v20, average='macro', labels=list(range(N_CLASES))):.7f}", flush=True)

    # ---------------------------------------------------------------- 1. referencia interna de V20
    print("\n=== 1. DIVERSIDAD INTERNA DE V20 (los cuatro cuartos) ===", flush=True)
    ref = []
    nombres = list(CUARTOS_V20)
    argmax_por_miembro = {}
    correcto_por_miembro = {}
    prob_por_miembro = {}
    for nombre, artefacto in CUARTOS_V20.items():
        matriz = _reconstruye_cadena() if artefacto == "__reconstruida__" else _cargar(artefacto)
        argmax_por_miembro[nombre] = np.argmax(matriz, axis=1)
        correcto_por_miembro[nombre] = argmax_por_miembro[nombre] == clase
        prob_por_miembro[nombre] = matriz
    for i, a in enumerate(nombres):
        for b in nombres[i + 1:]:
            celdas = _pareja(correcto_por_miembro[a], correcto_por_miembro[b])
            fila = {"modelo_a": a, "modelo_b": b, "alcance": "dentro de V20",
                    "corr_probabilidades": round(float(np.corrcoef(
                        prob_por_miembro[a].ravel(), prob_por_miembro[b].ravel())[0, 1]), 6),
                    "corr_errores": round(float(np.corrcoef(
                        (~correcto_por_miembro[a]).astype(float),
                        (~correcto_por_miembro[b]).astype(float))[0, 1]), 6),
                    "tasa_discrepancia": round(float(
                        (argmax_por_miembro[a] != argmax_por_miembro[b]).mean()), 6),
                    **celdas}
            ref.append(fila)
            print(f"  {a:<12} vs {b:<12} corr_err {fila['corr_errores']:.4f} | "
                  f"discrepancia {fila['tasa_discrepancia']:.4f} | "
                  f"a_acierta_b_falla {celdas['a_acierta_b_falla']:>6,} | "
                  f"b_acierta_a_falla {celdas['b_acierta_a_falla']:>6,}", flush=True)
    tabla_ref = pd.DataFrame(ref)
    print(f"  media de la referencia: corr_errores {tabla_ref['corr_errores'].mean():.4f} | "
          f"discrepancia {tabla_ref['tasa_discrepancia'].mean():.4f} | "
          f"discrepancia maxima {tabla_ref['tasa_discrepancia'].max():.4f}", flush=True)

    # ---------------------------------------------------------------- 2. todos contra V20
    print("\n=== 2. CADA MODELO CONTRA V20 ===", flush=True)
    filas = list(ref)
    detalle = {}
    for artefacto in reutilizables:
        model_id = artefacto.replace(".npy", "")
        if artefacto == V20:
            continue
        matriz = _cargar(artefacto)
        pred = np.argmax(matriz, axis=1)
        bien = pred == clase
        celdas = _pareja(bien, bien_v20)
        info = inventario[inventario["artefacto"] == artefacto].iloc[0]
        filas.append({
            "modelo_a": model_id, "modelo_b": "V20", "alcance": "contra V20",
            "familia": info["familia"], "fase": info["fase"],
            "corrected_cv_propio": float(info["corrected_cv_pooled"]),
            "corr_probabilidades": round(float(np.corrcoef(matriz.ravel(), oof_v20.ravel())[0, 1]), 6),
            "corr_errores": round(float(np.corrcoef((~bien).astype(float),
                                                     (~bien_v20).astype(float))[0, 1]), 6),
            "tasa_discrepancia": round(float((pred != pred_v20).mean()), 6),
            # OJO con la orientacion: la llamada es _pareja(bien_del_modelo, bien_de_v20), asi que
            # `a_acierta_b_falla` es el modelo acertando donde V20 falla (rescate) y
            # `b_acierta_a_falla` es el modelo fallando donde V20 acierta (dano). Invertirlos
            # produce cifras imposibles: el rescate no puede superar los errores de V20.
            "rescate_unico": celdas["a_acierta_b_falla"],
            "dano_unico": celdas["b_acierta_a_falla"],
            "neto": celdas["a_acierta_b_falla"] - celdas["b_acierta_a_falla"],
            "ambos_acierto": celdas["ambos_acierto"],
            "ambos_fallan": celdas["ambos_fallan"],
            "a_acierta_b_falla": celdas["a_acierta_b_falla"],
            "b_acierta_a_falla": celdas["b_acierta_a_falla"],
            # Dos medidas NORMALIZADAS que evitan la trampa de leer "rescate" sin skill.
            # Un modelo que acierta el 6 % "rescata" decenas de miles de filas solo porque V20
            # falla el 39 %: eso no es diversidad util, es ruido. Estas dos lo dicen.
            "fraccion_de_los_errores_de_V20_que_corrige": round(
                celdas["a_acierta_b_falla"] / max(int((~bien_v20).sum()), 1), 6),
            "fraccion_de_los_aciertos_de_V20_que_rompe": round(
                celdas["b_acierta_a_falla"] / max(int(bien_v20.sum()), 1), 6),
            "discrepancia_por_fold": [round(float((pred[f["val_idx"]] != pred_v20[f["val_idx"]]).mean()),
                                           6) for f in folds],
            "rescate_por_fold": [int((bien[f["val_idx"]] & ~bien_v20[f["val_idx"]]).sum())
                                 for f in folds],
        })
        detalle[model_id] = filas[-1]
    contra = pd.DataFrame([f for f in filas if f["alcance"] == "contra V20"])
    print(f"  {len(contra)} modelos comparados contra V20", flush=True)

    # Los que mas rescatan, y los que menos danan, con su CV al lado.
    print("\n  -- top 12 por rescate unico (filas que corrigen y V20 no) --", flush=True)
    for _, f in contra.nlargest(12, "rescate_unico").iterrows():
        print(f"    {f['modelo_a']:<40} CV {f['corrected_cv_propio']:.4f} | "
              f"corr_err {f['corr_errores']:.4f} | rescate {int(f['rescate_unico']):>6,} | "
              f"dano {int(f['dano_unico']):>6,} | neto {int(f['neto']):>+7,}", flush=True)
    print("\n  -- top 12 por ratio rescate/dano (con CV >= 0,45: por debajo el modelo es ruido) --",
          flush=True)
    ratio = contra[(contra["dano_unico"] > 0) & (contra["corrected_cv_propio"] >= 0.45)].copy()
    ratio["ratio"] = ratio["rescate_unico"] / ratio["dano_unico"].replace(0, np.nan)
    for _, f in ratio.nlargest(12, "ratio").iterrows():
        print(f"    {f['modelo_a']:<40} CV {f['corrected_cv_propio']:.4f} | "
              f"ratio {f['ratio']:.3f} | rescate {int(f['rescate_unico']):>6,} | "
              f"dano {int(f['dano_unico']):>6,} | "
              f"corta {f['fraccion_de_los_errores_de_V20_que_corrige']:.3f} de los errores de V20",
              flush=True)
    print("\n  -- top 12 por fraccion de errores de V20 corregidos (con CV >= 0,45) --", flush=True)
    print("     es la misma idea normalizada: cuanto del error de V20 sabe arreglar este modelo",
          flush=True)
    utiles = contra[contra["corrected_cv_propio"] >= 0.45]
    for _, f in utiles.nlargest(12, "fraccion_de_los_errores_de_V20_que_corrige").iterrows():
        print(f"    {f['modelo_a']:<40} CV {f['corrected_cv_propio']:.4f} | "
              f"corrige {f['fraccion_de_los_errores_de_V20_que_corrige']:.3f} de los errores | "
              f"rompe {f['fraccion_de_los_aciertos_de_V20_que_rompe']:.3f} de los aciertos", flush=True)
    print(f"\n  [trampa evitada] el mayor rescate bruto ({int(contra['rescate_unico'].max()):,}) "
          f"pertenece a '{contra.nlargest(1, 'rescate_unico')['modelo_a'].iloc[0]}', con CV "
          f"{contra.nlargest(1, 'rescate_unico')['corrected_cv_propio'].iloc[0]:.4f}: resucita filas "
          f"por azar, no por Complementariedad.", flush=True)

    # ---------------------------------------------------------------- 3. pairwise completa
    print("\n=== 3. MATRIZ PAIRWISE COMPLETA ===", flush=True)
    objetivo = [a.replace(".npy", "") for a in reutilizables]
    prob = {m: _cargar(m + ".npy") for m in objetivo}
    argmax = {m: np.argmax(prob[m], axis=1) for m in objetivo}
    bien = {m: argmax[m] == clase for m in objetivo}
    pares = []
    for i, a in enumerate(objetivo):
        for b in objetivo[i + 1:]:
            celdas = _pareja(bien[a], bien[b])
            pares.append({
                "modelo_a": a, "modelo_b": b,
                "corr_probabilidades": round(float(np.corrcoef(prob[a].ravel(),
                                                               prob[b].ravel())[0, 1]), 6),
                "corr_errores": round(float(np.corrcoef((~bien[a]).astype(float),
                                                         (~bien[b]).astype(float))[0, 1]), 6),
                "tasa_discrepancia": round(float((argmax[a] != argmax[b]).mean()), 6),
                **celdas,
                "neto_a": celdas["a_acierta_b_falla"] - celdas["b_acierta_a_falla"]})
    completa = pd.DataFrame(pares).sort_values("tasa_discrepancia", ascending=False)
    completa.to_csv(DIVERSIDAD, index=False)
    contra.to_csv(CONTRA_V20, index=False)
    print(f"  {len(completa)} parejas | {DIVERSIDAD.name} escrito", flush=True)
    print(f"  mayor discrepancia entre dos modelos distintos: "
          f"{completa['tasa_discrepancia'].max():.4f}", flush=True)
    print("  -- las 10 parejas mas divergentes --", flush=True)
    for _, f in completa.head(10).iterrows():
        print(f"    {f['modelo_a']:<38} vs {f['modelo_b']:<38} "
              f"discrep {f['tasa_discrepancia']:.4f} | corr_err {f['corr_errores']:.4f}", flush=True)

    # Un artefacto con el MISMO argmax que otro no aporta(member's decisions) aunque sus
    # magnitudes difieran. Es una trampa de inventario y hay que contarla.
    por_argmax: dict[str, list[str]] = {}
    for m in objetivo:
        por_argmax.setdefault(hashlib.sha256(argmax[m].tobytes()).hexdigest()[:12],
                              []).append(m)
    gemelos = {k: v for k, v in por_argmax.items() if len(v) > 1}
    print(f"\n  [inventario] {len(gemelos)} grupos de artefactos con argmax IDENTICO:", flush=True)
    for _, grupo in gemelos.items():
        print(f"    {grupo}", flush=True)

    _figuras(contra, completa, tabla_ref, inventario, argmax, prob, clase)
    RESULTADO.write_text(json.dumps({
        "fase": "V29 02 P2 diversidad OOF", "fecha": "2026-09-29", "experimenta": False,
        "n_modelos": len(objetivo), "n_parejas": len(completa),
        "referencia_diversidad_interna_v20": {
            "corr_errores_media": round(float(tabla_ref["corr_errores"].mean()), 6),
            "tasa_discrepancia_media": round(float(tabla_ref["tasa_discrepancia"].mean()), 6),
            "tasa_discrepancia_maxima": round(float(tabla_ref["tasa_discrepancia"].max()), 6),
            "parejas": ref,
        },
        "rescate_unico_maximo": int(contra["rescate_unico"].max()),
        "modelo_con_mas_rescate": str(contra.nlargest(1, "rescate_unico")["modelo_a"].iloc[0]),
        "dano_unico_maximo": int(contra["dano_unico"].max()),
        "modelo_con_mas_dano": str(contra.nlargest(1, "dano_unico")["modelo_a"].iloc[0]),
        "corr_errores_vs_v20_minima": round(float(contra["corr_errores"].min()), 6),
        "modelo_mas_diverso_vs_v20": str(contra.nsmallest(1, "corr_errores")["modelo_a"].iloc[0]),
        "gemelos_de_argmax": list(gemelos.values()),
        "nota_gemelos": "artefactos con argmax identico: difieren en las magnitudes pero toman las "
                        "MISMAS decisiones, asi que en un ensemble con votedecisiones aportan cero",
        "criterio": "cinco medidas simultaneas; no se resumen en un numero porque un score unico "
                    "ocultaria la tension entre rescate y dano",
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[P2] {time.time() - inicio:.0f} s")
    return 0


def _figuras(contra: pd.DataFrame, completa: pd.DataFrame, ref: pd.DataFrame,
             inventario: pd.DataFrame, argmax: dict, prob: dict, clase: np.ndarray) -> None:
    """Tres figuras: CV contra diversidad, mapa de calor de correlacion de errores, rescate/dano."""
    # --- 1. CV individual contra diversidad frente a V20
    fig, ejes = plt.subplots(1, 2, figsize=(15, 5.5))
    eje = ejes[0]
    eje.scatter(contra["corr_errores"], contra["corrected_cv_propio"],
                c=contra["rescate_unico"], cmap="viridis", s=38, edgecolors="k", linewidths=0.4)
    mejor = contra.nlargest(4, "rescate_unico")
    for _, f in mejor.iterrows():
        eje.annotate(f["modelo_a"][:22], (f["corr_errores"], f["corrected_cv_propio"]),
                     fontsize=6.5, xytext=(3, 3), textcoords="offset points")
    eje.axhline(CV_V20, color="red", linestyle="--", linewidth=1, label=f"V20 = {CV_V20}")
    eje.set_xlabel("correlacion de errores con V20 (menor = mas diverso)")
    eje.set_ylabel("corrected_cv propio")
    eje.set_title("CV individual contra diversidad frente a V20\n"
                  "el color es rescate unico: filas que corrige y V20 no", fontsize=9)
    eje.legend(fontsize=7)

    eje = ejes[1]
    orden = contra.sort_values("corrected_cv_propio", ascending=False)
    eje.bar(range(len(orden)), orden["rescate_unico"], color="#2171b5", label="rescate unico")
    eje.bar(range(len(orden)), -orden["dano_unico"], color="#cb181d", label="dano unico")
    eje.set_xticks(range(len(orden)))
    eje.set_xticklabels(orden["modelo_a"], rotation=90, fontsize=4)
    eje.axhline(0, color="k", linewidth=1)
    eje.set_ylabel("filas")
    eje.set_title("Rescate unico frente a dano unico, todos los modelos", fontsize=9)
    eje.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_CV_DIV, dpi=130)
    plt.close(fig)

    # --- 2. mapa de calor de la correlacion de errores
    top = list(contra.nlargest(28, "corrected_cv_propio")["modelo_a"])
    extra = [m for m in ["v19_b1_oof_logreg", "v19_b1_oof_mlp", "v19_oof_arbol_5folds",
                         "v22_seis_compartida", "v20_f3b_oof_candidato"] if m not in top]
    ejes_modelos = top + extra
    matriz = np.eye(len(ejes_modelos))
    for i, a in enumerate(ejes_modelos):
        for j, b in enumerate(ejes_modelos):
            if i == j:
                continue
            x, z = (a, b) if a in argmax else (a + ".npy", b + ".npy")
            matriz[i, j] = float(np.corrcoef((~(argmax.get(x, np.argmax(prob[x])) == clase)).astype(float),
                                             (~(argmax.get(z, np.argmax(prob[z])) == clase)).astype(float))[0, 1])
    fig, eje = plt.subplots(figsize=(11, 9.5))
    imagen = eje.imshow(matriz, cmap="magma_r", vmin=0.0, vmax=1.0)
    eje.set_xticks(range(len(ejes_modelos)))
    eje.set_xticklabels(ejes_modelos, rotation=90, fontsize=5.5)
    eje.set_yticks(range(len(ejes_modelos)))
    eje.set_yticklabels(ejes_modelos, fontsize=5.5)
    for i in range(len(ejes_modelos)):
        for j in range(len(ejes_modelos)):
            if i != j:
                eje.text(j, i, f"{matriz[i, j]:.2f}", ha="center", va="center", fontsize=3.4,
                         color="white" if matriz[i, j] < 0.75 else "black")
    barra = fig.colorbar(imagen)
    barra.set_label("correlacion de errores (1 = identicos)", fontsize=8)
    eje.set_title("Correlacion de errores entre los 28 modelos con mejor CV mas los 4 cuartos de "
                  "V20\noscuro = errores casi identicos; claro = errores distintos", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_HEAT, dpi=130)
    plt.close(fig)

    # --- 3. rescate frente a dano de los 25 mas promises
    fig, eje = plt.subplots(figsize=(9, 6.5))
    sub = contra.nlargest(25, "rescate_unico").sort_values("neto", ascending=True)
    y = np.arange(len(sub))
    eje.barh(y, sub["rescate_unico"], color="#2171b5", label="rescate unico")
    eje.barh(y, -sub["dano_unico"], color="#cb181d", label="dano unico")
    eje.set_yticks(y)
    eje.set_yticklabels([f"{m}  (CV {c:.4f}, corr_err {e:.3f})"
                         for m, c, e in zip(sub["modelo_a"], sub["corrected_cv_propio"],
                                            sub["corr_errores"])], fontsize=6.5)
    eje.axvline(0, color="k", linewidth=1)
    eje.set_xlabel("filas (azul a la derecha: corrige; rojo a la izquierda: rompe)")
    eje.set_title("Los 25 con mas rescate unico, con su CV y su correlacion de errores", fontsize=10)
    eje.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_RESCATE, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
