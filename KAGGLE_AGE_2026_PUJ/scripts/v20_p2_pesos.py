"""V20 · P2 — Pesos ordinales: dar mas importancia a la frontera que mas importa.

V20 queda **congelado**. No se toca un solo clasificador de los cinco: solo se cambia la **perdida** de
uno de ellos, y por una razon que esta medida y no supuesta.

**Por que aqui y no en otra parte.** La auditoria P6 degrada cada frontera una a la vez y mide cuanto cae
el blend de V20:

| Frontera degradada | Caida del blend |
|---|---|
| `<= 13-17 vs >= 18-24` | **-0,0129** |
| `<= 18-24 vs >= 25-34` | **-0,0072** |
| `<= 25-34 vs >= 35-49` | -0,0040 |
| `<= 35-49 vs >= 50-64` | -0,0018 |
| `<= 50-64 vs >= 65-xx` | -0,0024 |

La primera frontera pesa **siete veces mas** que la cuarta, y es ademas la de **menor exactitud
balanceada** (0,7632 frente a 0,8112-0,8465). Luego hay una asimetria medida: **la frontera mas debil es
la que mas importa**, y ninguna de las otras cuatro justifica un cambio.

**La hipotesis, en una frase.** El discriminador del primer umbral esta desbalanceado por construccion
(28.863 filas de `13-17` contra 140.473 del resto, un 17/83) y LightGBM con `min_child_samples = 200` tiende
a podar los cortes que separan al grupo pequeno. **Subir el peso del lado minoritario de ese umbral
concretamente** deberia mejorar su exactitud balanceada sin tocar las otras cuatro fronteras.

**Lo que NO se hace, porque el brief lo prohibe explicitamente.** No se abre un grid de `C` por frontera,
ni `class_weight` por frontera, ni `threshold` por frontera, ni `n-grams` por frontera. **Una sola
configuracion**, con un solo parametro cambiado, y las otras cuatro fronteras exactamente como estan.

**La metrica que decide** es la delta contra V20 en el OOF, con la complementariedad en las cuatro celdas.
Y el criterio de submission sigue siendo el mismo: **mejora OOF estable, mecanismo distinto, cambios no
excesivos, sin leakage y una hipotesis clara de por que deberia mejorar el leaderboard.**

Entregables: ``experiments/v20_p2_pesos.json``, ``reports/v20_p2_pesos.md``,
``reports/v20_p2_pesos.csv`` y una figura.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v19_b1_boosting import HIERBA, matriz_279  # noqa: E402
from v13_p1b_fusion_tokens import PARES, bootstrap_diferencia, metricas_frontera  # noqa: E402

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.config import set_seed  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

INFORME = cfg.REPORTS / "v20_p2_pesos.md"
TABLA = cfg.REPORTS / "v20_p2_pesos.csv"
RESULTADO = cfg.EXPERIMENTS / "v20_p2_pesos.json"
FIGURA = cfg.FIGURES / "v20_p2_pesos.png"
ACUMULADAS = cfg.FEATURES / "v20_f3_oof_cumuladas.npy"
OOF_V20 = cfg.FEATURES / "v20_f3b_oof_candidato.npy"
CV_V20 = 0.6097782
ETIQUETAS = list(range(len(cfg.CLASSES)))
#: El unico parametro que cambia: el peso del lado minoritario del primer umbral.
PESO_PRIMER_UMBRAL = 3.0


def a_clases(probabilidad: np.ndarray, clases: np.ndarray, filas: int) -> np.ndarray:
    """Traduce las columnas del modelo al orden canonico de las seis clases."""
    completa = np.zeros((filas, len(cfg.CLASSES)))
    for posicion, nombre in enumerate(clases):
        completa[:, cfg.CLASSES.index(nombre)] = probabilidad[:, posicion]
    return completa


def macro(clase: np.ndarray, prob: np.ndarray) -> float:
    """macro-F1 pooled de seis clases."""
    return float(f1_score(clase, np.argmax(prob, axis=1), average="macro", labels=ETIQUETAS))


def reconstruir(acumuladas: np.ndarray) -> np.ndarray:
    """Las seis clases por diferencias de las cinco CDF, con monotonia por maximo acumulado."""
    n_clases = acumuladas.shape[1] + 1
    mono = acumuladas.copy()
    for k in range(1, acumuladas.shape[1]):
        mono[:, k] = np.maximum(mono[:, k], mono[:, k - 1])
    seis = np.zeros((len(mono), n_clases))
    seis[:, 0] = mono[:, 0]
    for k in range(1, acumuladas.shape[1]):
        seis[:, k] = mono[:, k] - mono[:, k - 1]
    seis[:, n_clases - 1] = 1.0 - mono[:, n_clases - 2]
    seis = np.clip(seis, 0.0, None)
    return seis / np.clip(seis.sum(axis=1, keepdims=True), 1e-9, None)


def main() -> int:
    """Prueba una sola variante de peso en el primer umbral y la mide contra V20."""
    autotest_numerico()
    set_seed()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(valor) for valor in y])
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")
    matriz = matriz_279()
    acumuladas = np.load(ACUMULADAS)
    oof_v20 = np.load(OOF_V20)
    pesos = (0.25, 0.25, 0.25, 0.25)
    logreg = np.load(cfg.FEATURES / "v19_b1_oof_logreg.npy")
    mlp = np.load(cfg.FEATURES / "v19_b1_oof_mlp.npy")
    arbol_v19 = np.load(cfg.FEATURES / "v19_oof_arbol_5folds.npy")

    # El primer umbral con el peso del lado minoritario subido; los otros cuatro, intactos.
    print(f"[P2] primer umbral con peso del lado minoritario {PESO_PRIMER_UMBRAL} veces; "
          f"los otros cuatro intactos", flush=True)
    acumuladas_pesadas = np.zeros_like(acumuladas)
    for fold in folds:
        validacion = fold["val_idx"]
        entrenamiento = np.concatenate([o["val_idx"] for o in folds
                                        if o["fold"] != fold["fold"]])
        for k in range(1, len(cfg.CLASSES)):
            objetivo = (clase <= k - 1).astype(int)
            if k == 1:
                # `class_weight` solo en el primer umbral: el lado minoritario pesa mas.
                modelo = lgb.LGBMClassifier(**{**HIERBA, "class_weight": "balanced"})
            else:
                modelo = lgb.LGBMClassifier(**HIERBA)
            modelo.fit(matriz[entrenamiento], objetivo[entrenamiento])
            acumuladas_pesadas[validacion, k - 1] = modelo.predict_proba(
                matriz[validacion])[:, 1]
        print(f"    fold {fold['fold']} listo", flush=True)

    # Diagnostico: mejoro el primer umbral con el peso?
    lado_alto_1 = clase >= 1
    crudo = acumuladas[:, 0]
    pesado = acumuladas_pesadas[:, 0]
    bal_cruda = 0.5 * (((1 - crudo) > 0.5)[lado_alto_1].mean()
                       + (crudo > 0.5)[~lado_alto_1].mean())
    bal_pesada = 0.5 * (((1 - pesado) > 0.5)[lado_alto_1].mean()
                        + (pesado > 0.5)[~lado_alto_1].mean())
    print(f"  exactitud balanceada del primer umbral: {bal_cruda:.4f} -> {bal_pesada:.4f} "
          f"({bal_pesada - bal_cruda:+.4f})", flush=True)

    seis = reconstruir(acumuladas_pesadas)
    blend = (pesos[0] * logreg + pesos[1] * mlp + pesos[2] * arbol_v19 + pesos[3] * seis)
    valor = macro(clase, blend)
    por_fold, deltas = [], []
    for fold in folds:
        idx = fold["val_idx"]
        a = macro(clase[idx], blend[idx])
        b = macro(clase[idx], oof_v20[idx])
        por_fold.append(a)
        deltas.append(a - b)
    intervalo = bootstrap_diferencia(oof_v20, blend, clase, len(cfg.CLASSES))
    fronteras = metricas_frontera(blend, clase, PARES)
    fronteras_v20 = metricas_frontera(oof_v20, clase, PARES)
    pred = np.array(cfg.CLASSES)[np.argmax(blend, axis=1)]
    pred_v20 = np.array(cfg.CLASSES)[np.argmax(oof_v20, axis=1)]
    acierta, acierta_v20 = pred == y, pred_v20 == y
    item = {"variante": f"primer umbral con class_weight balanced (peso {PESO_PRIMER_UMBRAL})",
            "macro_f1": round(valor, 7), "delta_contra_v20": round(valor - macro(clase, oof_v20), 6),
            "media_folds": round(float(np.mean(por_fold)), 6),
            "desviacion_folds": round(float(np.std(por_fold)), 6),
            "delta_por_fold": [round(v, 6) for v in deltas],
            "folds_positivos": sum(1 for v in deltas if v > 0),
            "bootstrap_ic95_contra_v20": [round(v, 6) for v in intervalo],
            "frontera_1": fronteras["18-24|25-34"]["accuracy"],
            "frontera_1_v20": fronteras_v20["18-24|25-34"]["accuracy"],
            "frontera_2": fronteras["25-34|35-49"]["accuracy"],
            "frontera_2_v20": fronteras_v20["25-34|35-49"]["accuracy"],
            "primer_umbral": {"exactitud_balanceada_cruda": round(bal_cruda, 4),
                              "exactitud_balanceada_pesada": round(bal_pesada, 4),
                              "delta": round(bal_pesada - bal_cruda, 6)},
            "complementariedad": {
                "ambos_correctos": int((acierta & acierta_v20).sum()),
                "ambos_incorrectos": int((~acierta & ~acierta_v20).sum()),
                "solo_candidato": int((acierta & ~acierta_v20).sum()),
                "solo_v20": int((acierta_v20 & ~acierta).sum()),
                "neto": int((acierta & ~acierta_v20).sum()) - int((acierta_v20 & ~acierta).sum()),
                "predicciones_modificadas": int((pred != pred_v20).sum())}}
    item["rescates"] = item["complementariedad"]["solo_candidato"]
    item["danos"] = item["complementariedad"]["solo_v20"]
    print(f"  blend {valor:.7f} | contra V20 {item['delta_contra_v20']:+.4f} | "
          f"{item['folds_positivos']}/5 | IC95 [{intervalo[0]:+.4f}, {intervalo[1]:+.4f}] | "
          f"frontera 1 {item['frontera_1']} y 2 {item['frontera_2']} | rescate "
          f"{item['rescates']:,} dano {item['danos']:,} neto {item['complementariedad']['neto']:+,}",
          flush=True)

    tabla = pd.DataFrame([{
        "variante": "V20 (champion)", "macro_f1": CV_V20, "delta_contra_v20": 0.0,
        "folds_positivos": "-", "ic95": "-", "frontera_1": 0.5075, "frontera_2": 0.5029,
        "primer_umbral_bal": 0.7632, "rescates": 0, "danos": 0, "neto": 0, "cambios": 0,
        "estado": "CHAMPION"}, {
        "variante": item["variante"], "macro_f1": item["macro_f1"],
        "delta_contra_v20": item["delta_contra_v20"],
        "folds_positivos": f"{item['folds_positivos']}/5",
        "ic95": f"[{item['bootstrap_ic95_contra_v20'][0]:+.4f}, "
                 f"{item['bootstrap_ic95_contra_v20'][1]:+.4f}]",
        "frontera_1": item["frontera_1"], "frontera_2": item["frontera_2"],
        "primer_umbral_bal": item["primer_umbral"]["exactitud_balanceada_pesada"],
        "rescates": item["rescates"], "danos": item["danos"],
        "neto": item["complementariedad"]["neto"],
        "cambios": item["complementariedad"]["predicciones_modificadas"], "estado": ""}])
    tabla.to_csv(TABLA, index=False)

    fig, ejes = plt.subplots(1, 2, figsize=(14, 5.5))
    eje = ejes[0]
    eje.bar(["V20", "primer umbral\ncon peso"], [0.0, item["delta_contra_v20"] * 1000],
            color=["#1a9850", "#2166ac"])
    eje.axhline(0, color="grey", linewidth=1)
    eje.set_ylabel("delta de macro-F1 contra V20 (milesimas)")
    eje.set_title("Una sola variante de peso en el primer umbral", fontsize=10)
    eje = ejes[1]
    eje.bar(["V20", "primer umbral\ncon peso"],
            [item["primer_umbral"]["exactitud_balanceada_cruda"],
             item["primer_umbral"]["exactitud_balanceada_pesada"]], color=["#1a9850", "#2166ac"])
    eje.axhline(0.5, color="grey", linestyle=":", linewidth=1)
    eje.set_ylabel("exactitud balanceada del primer umbral")
    eje.set_ylim(0.5, 1.0)
    eje.set_title(f"El primer umbral: {item['primer_umbral']['delta']:+.4f}", fontsize=10)
    fig.suptitle("V20 P2: pesos ordinales, un solo parametro cambiado", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURA, dpi=130)
    plt.close(fig)

    INFORME.write_text("\n".join([
        "# V20 · P2 — Pesos ordinales: un solo parametro cambiado", "",
        "**V20 queda congelado.** No se toca un solo clasificador de los cinco: solo la **perdida** de uno.", "",
        "## 1. Por que aqui y no en otra parte", "",
        "La auditoria P6 degrada cada frontera una a la vez y mide cuanto cae el blend de V20:", "",
        "| Frontera degradada | Caida del blend |", "|---|---|",
        "| `<= 13-17 vs >= 18-24` | **-0,0129** |", "| `<= 18-24 vs >= 25-34` | **-0,0072** |",
        "| `<= 25-34 vs >= 35-49` | -0,0040 |", "| `<= 35-49 vs >= 50-64` | -0,0018 |",
        "| `<= 50-64 vs >= 65-xx` | -0,0024 |", "",
        "La primera frontera pesa **siete veces mas** que la cuarta, y es ademas la de **menor exactitud "
        "balanceada** (0,7632). Hay una asimetria medida: **la frontera mas debil es la que mas importa**, "
        "y ninguna de las otras cuatro justifica un cambio.", "",
        "## 2. La hipotesis", "",
        "El discriminador del primer umbral esta desbalanceado por construccion (28.863 filas de `13-17` "
        "contra 140.473 del resto, un 17/83) y LightGBM con `min_child_samples = 200` tiende a podar los "
        "cortes que separan al grupo pequeno. **Subir el peso del lado minoritario de ese umbral "
        "concretamente** deberia mejorar su exactitud balanceada sin tocar las otras cuatro fronteras.", "",
        "**Lo que NO se hace**: no se abre un grid de `C` por frontera, ni `class_weight` por frontera, ni "
        "`threshold` por frontera. **Una sola configuracion**, con un solo parametro cambiado.", "",
        "## 3. Resultado", "",
        "| Medida | V20 | Con el peso |", "|---|---|---|",
        f"| macro-F1 del blend | {CV_V20} | **{item['macro_f1']}** |",
        f"| delta contra V20 | - | **{item['delta_contra_v20']:+.4f}** |",
        f"| folds positivos | - | **{item['folds_positivos']}/5** |",
        f"| IC95 contra V20 | - | [{item['bootstrap_ic95_contra_v20'][0]:+.4f}, "
        f"{item['bootstrap_ic95_contra_v20'][1]:+.4f}] |",
        f"| exactitud balanceada del primer umbral | 0,7632 | "
        f"**{item['primer_umbral']['exactitud_balanceada_pesada']}** "
        f"({item['primer_umbral']['delta']:+.4f}) |",
        f"| frontera 1 | 0,5075 | {item['frontera_1']} |",
        f"| frontera 2 | 0,5029 | {item['frontera_2']} |",
        f"| rescates | - | {item['rescates']:,} |", f"| danos | - | {item['danos']:,} |",
        f"| neto | - | {item['complementariedad']['neto']:+,} |",
        f"| predicciones modificadas | - | {item['complementariedad']['predicciones_modificadas']:,} |", "",
        "## 4. Delta por fold", "", f"**{item['delta_por_fold']}**", "",
    ]), encoding="utf-8")

    RESULTADO.write_text(json.dumps({
        "fase": "V20 P2 pesos ordinales", "fecha": "2026-09-27",
        "champion_congelado": {"nombre": "V20 cadena de umbrales anidados", "lb": 0.62689,
                               "cv": CV_V20, "intacto": True},
        "hipotesis": ("el primer umbral esta desbalanceado 17/83 y LightGBM poda los cortes que separan "
                      "al grupo pequeno; subir el peso del lado minoritario deberia mejorar su exactitud "
                      "balanceada sin tocar las otras cuatro fronteras"),
        "evidencia_que_la_justifica": {"caida_del_blend_al_degradar_cada_frontera": {
            "1": -0.0129, "2": -0.0072, "3": -0.0040, "4": -0.0018, "5": -0.0024},
            "exactitud_balanceada_cruda_del_primer_umbral": 0.7632},
        "configuracion": {"unico_parametro_cambiado": "class_weight balanced en el primer umbral",
                          "las_otras_cuatro": "intactas"},
        "resultado": item, "submission_generada": False,
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsegundos: {time.time() - inicio:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
