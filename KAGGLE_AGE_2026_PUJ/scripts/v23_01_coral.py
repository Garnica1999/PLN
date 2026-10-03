"""V23 · 01 — CORAL / cumulative-link con pesos compartidos frente a V20.

**Un solo experimento.** La unica variable es si las cinco fronteras ordinales comparten el
vector de pesos o cada una tiene el suyo. No se varia `lambda_w`, ni `lambda_umbral`, ni el
optimizador, ni el numero de columnas, ni el numero de folds. Esos valores estan
**pre-registrados** en `scripts/v23_00_tests_coral.py` y se congelan aqui.

## Los tres modelos que se comparan

| Modelo | Pesos | Umbrales | No lineal |
|---|---|---|---|
| **V20 (control A)** | cinco independientes | cinco binarios sin orden | si (LightGBM) |
| **control B** | cinco independientes | cinco binarios sin orden | no (lineal) |
| **CORAL (candidato)** | **uno compartido** | **cinco ordenados** | no (lineal) |

El control A es el que exige el brief: la referencia contra la que se decide el gate. El control
B es **imprescindible para atribuir**: sin el, una diferencia entre CORAL y V20 mezcla dos cosas
—compartir pesos y la perdida de no linealidad— y no sabriamos cual de las dos la causa. Con el, la
comparacion CORAL contra B aísla exactamente el efecto de compartir pesos, que es la hipotesis.

## Por que CORAL es distinto de las dos formulaciones ya medidas

- V20 son cinco binarios **independientes**: la monotonia no esta garantizada y el 11,18 % de las
  filas la viola (D-057).
- El modelo compartido de D-057 comparte la **representacion** pero sus cinco CDF salen de una
  salida multiclass **por suma**: no hay umbrales ordinales, luego no es un cumulative-link.
- CORAL tiene un unico `w` y cinco `theta` **ordenados por construccion**: `theta_k = gamma +
  suma_{j<k} softplus(delta_j)`, luego las cinco CDF son monotonicas siempre y no hace falta
  imponerlo despues.

## Protocolo

Mismas 279 features, mismos cinco `StratifiedGroupKFold` agrupados por duplicado que V20, misma
semilla, y el mismo blend de cuatro cuartos que usa V20. El test no se toca en ningun momento y
el holdout tampoco.

Entregables: ``experiments/v23_01_coral.json``, ``reports/v23_coral_results.csv``,
``reports/v23_coral_analysis.md`` y las figuras de monotonicidad y complementariedad.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.ordinal_coral import CoralOrdinal, cdf_a_clases  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402
from v13_p1b_fusion_tokens import bootstrap_diferencia, metricas_frontera  # noqa: E402
from v19_b1_boosting import matriz_279  # noqa: E402

INFORME = cfg.REPORTS / "v23_coral_analysis.md"
TABLA = cfg.REPORTS / "v23_coral_results.csv"
RESULTADO = cfg.EXPERIMENTS / "v23_01_coral.json"
FIG_MONO = cfg.FIGURES / "v23_coral_monotonicidad.png"
FIG_COMP = cfg.FIGURES / "v23_coral_complementariedad.png"
FIG_FOLD = cfg.FIGURES / "v23_coral_folds.png"
OOF_V20 = cfg.FEATURES / "v20_f3b_oof_candidato.npy"
CUMULADAS_V20 = cfg.FEATURES / "v20_f3_oof_cumuladas.npy"
CV_V20 = 0.6097782
LB_V20 = 0.62689
ETIQUETAS = list(range(len(cfg.CLASSES)))
#: Pre-registrados en `scripts/v23_00_tests_coral.py` y congelados para este experimento.
LAMBDA_W = 1.0
LAMBDA_UMBRAL = 0.0
#: Gate del brief: mejora minima y folds positivos exigidos.
GATE_DELTA = 0.001
GATE_FOLDS = 4
#: C de los cinco binarios del control B. Mismo valor para los cinco, luego el control no tiene
#: ninguna ventaja de ajuste sobre CORAL: los dos son lineales con la misma escala de penalty.
C_LOGISTICO = 1.0
CDF_CORAL = cfg.FEATURES / "v23_coral_cdf_oof.npy"
SEIS_B = cfg.FEATURES / "v23_binarios_lineales_oof.npy"
#: Filas del train; se usa en el informe para el porcentaje de violaciones.
N_FILAS = 169336
#: Pares de frontera que ya usa el proyecto, para no inventar metricas nuevas.
PARES = {("18-24", "25-34"): "18-24|25-34", ("25-34", "35-49"): "25-34|35-49"}


def macro(clase: np.ndarray, prob: np.ndarray) -> float:
    """macro-F1 pooled de las seis clases."""
    return float(f1_score(clase, np.argmax(prob, axis=1), average="macro", labels=ETIQUETAS))


def matriz_confusion(clase: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Matriz 6x6 de confusion, con filas reales y columnas predichas."""
    matriz = np.zeros((len(ETIQUETAS), len(ETIQUETAS)), dtype=np.int64)
    np.add.at(matriz, (clase, pred), 1)
    return matriz


def complementariedad(real: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """Las cuatro celdas mas el saldo, en el total y por clase.

    `rescata` es "el candidato acierta donde el control falla" y `danio` al reves. El `neto` es
    lo que decide, y por clase porque el macro-F1 pondera las seis por igual.
    """
    acierta_a = pred_a == real
    acierta_b = pred_b == real
    salida = {"ambos_aciertos": int((acierta_a & acierta_b).sum()),
              "ambos_fallos": int((~acierta_a & ~acierta_b).sum()),
              "solo_candidato": int((acierta_b & ~acierta_a).sum()),
              "solo_control": int((acierta_a & ~acierta_b).sum())}
    salida["neto"] = salida["solo_candidato"] - salida["solo_control"]
    salida["cambios_de_prediccion"] = int((pred_a != pred_b).sum())
    salida["por_clase"] = {}
    for indice, nombre in enumerate(cfg.CLASSES):
        filas = real == indice
        rescata = int((acierta_b & ~acierta_a & filas).sum())
        danio = int((acierta_a & ~acierta_b & filas).sum())
        salida["por_clase"][nombre] = {"filas": int(filas.sum()), "rescata": rescata,
                                        "danio": danio, "neto": rescata - danio}
    return salida


def main() -> int:
    """Entrena CORAL y el control B con los folds de V20, y los mide contra el control A."""
    autotest_numerico()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(valor) for valor in y])
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")
    matriz = matriz_279()
    n_clases = len(cfg.CLASSES)
    logreg = np.load(cfg.FEATURES / "v19_b1_oof_logreg.npy")
    mlp = np.load(cfg.FEATURES / "v19_b1_oof_mlp.npy")
    arbol_v19 = np.load(cfg.FEATURES / "v19_oof_arbol_5folds.npy")

    # ---------------------------------------------------------------- control A: V20
    oof_v20 = np.load(OOF_V20)
    valor_v20 = macro(clase, oof_v20)
    reproduce = abs(valor_v20 - CV_V20) < 1e-6
    print(f"[control A] V20 {valor_v20:.7f} frente a {CV_V20:.7f} | reproduce: {reproduce}",
          flush=True)
    if not reproduce:
        print("[ABORTO] V20 no reproduce. No se ejecuta CORAL hasta resolverlo.")
        return 1

    # Las CDF de V20 son las **antes** de la correccion de monotonia, que es donde se mide si
    # el modelo las garantiza o no. D-057 reporta 18.927 violaciones sobre 169.336 filas.
    acumuladas_v20_crudas = np.load(CUMULADAS_V20)
    violaciones_v20 = int((np.diff(acumuladas_v20_crudas, axis=1) < 0).any(axis=1).sum())
    print(f"[control A] violaciones de monotonia de las cinco CDF de V20: {violaciones_v20:,} "
          f"de {len(y):,} ({violaciones_v20 / len(y):.2%})", flush=True)

    # ---------------------------------------------------------------- CORAL y control B
    cdf_coral = np.zeros((len(y), n_clases - 1))
    seis_independiente = np.zeros((len(y), n_clases))
    fallos_coral: list[str] = []
    if CDF_CORAL.exists() and SEIS_B.exists():
        cdf_coral = np.load(CDF_CORAL)
        seis_independiente = np.load(SEIS_B)
        print(f"[cache] {CDF_CORAL.name} y {SEIS_B.name}", flush=True)
    for fold in [] if CDF_CORAL.exists() and SEIS_B.exists() else folds:
        validacion = fold["val_idx"]
        entrenamiento = np.concatenate([o["val_idx"] for o in folds
                                        if o["fold"] != fold["fold"]])

        # CORAL: un `w` y cinco `theta` ordenados.
        coral = CoralOrdinal(lambda_w=LAMBDA_W, lambda_umbral=LAMBDA_UMBRAL)
        coral.fit(matriz[entrenamiento], clase[entrenamiento])
        if not coral.exito_:
            fallos_coral.append(f"fold {fold['fold']}: L-BFGS-B no convergio")
        cdf_coral[validacion] = coral.predecir_cdf(matriz[validacion])

        # Control B: cinco binarios **independientes**, misma familia lineal y **el mismo
        # estandarizado por fold que recibe CORAL**, luego la unica diferencia entre los dos es
        # el peso compartido. Sin ese estandarizado `LogisticRegression` avisa de que no
        # converge, y un control que no converge no es un control.
        media = matriz[entrenamiento].mean(axis=0)
        desviacion = matriz[entrenamiento].std(axis=0)
        desviacion = np.where(desviacion < 1e-8, 1.0, desviacion)
        x_entrenamiento = (matriz[entrenamiento] - media) / desviacion
        x_validacion = (matriz[validacion] - media) / desviacion
        for k in range(1, n_clases):
            objetivo = (clase <= k - 1).astype(int)
            binario = LogisticRegression(C=C_LOGISTICO, max_iter=5000, tol=1e-8)
            binario.fit(x_entrenamiento, objetivo[entrenamiento])
            seis_independiente[validacion, k - 1] = binario.predict_proba(x_validacion)[:, 1]
        print(f"    fold {fold['fold']}: CORAL y control B listos "
              f"(|w| CORAL {coral.norma_pesos():.4f})", flush=True)
    # Se guarda el OOF de los dos candidatos para que el informe y las figuras se puedan rehacer
    # sin reentrenar, igual que hace V20 con `v20_f3b_oof_candidato.npy`.
    np.save(CDF_CORAL, cdf_coral)
    np.save(SEIS_B, seis_independiente)

    seis_coral = cdf_a_clases(cdf_coral)
    # Control B: los cinco binarios tampoco garantizan orden, luego se aplica **la misma**
    # correccion que V20 (maximo acumulado) para que la comparacion sea sobre el mecanismo y no
    # sobre una ventaja de reconstruccion.
    acumuladas_b = np.maximum.accumulate(seis_independiente, axis=1)
    seis_b = cdf_a_clases(acumuladas_b)
    violaciones_coral = int((np.diff(cdf_coral, axis=1) < 0).any(axis=1).sum())
    print(f"[CORAL] violaciones de monotonia: {violaciones_coral:,} de {len(y):,}", flush=True)

    # ---------------------------------------------------------------- metricas
    candidatos = {"V20 (control A)": oof_v20, "cinco binarios lineales (control B)": seis_b,
                  "CORAL pesos compartidos": seis_coral}
    filas: list[dict] = []
    detalle: dict = {}
    for nombre, prob in candidatos.items():
        pred = np.argmax(prob, axis=1)
        por_fold = [macro(clase[f["val_idx"]], prob[f["val_idx"]]) for f in folds]
        delta_fold = [round(v - valor_v20, 6) for v in por_fold]
        comp = complementariedad(clase, np.argmax(oof_v20, axis=1), pred)
        fronteras = metricas_frontera(prob, clase, PARES)
        item = {"modelo": nombre, "macro_f1": round(macro(clase, prob), 7),
                "delta_vs_v20": round(macro(clase, prob) - valor_v20, 6),
                "media_folds": round(float(np.mean(por_fold)), 6),
                "desviacion_folds": round(float(np.std(por_fold)), 6),
                "scores_por_fold": [round(v, 7) for v in por_fold],
                "delta_por_fold": delta_fold,
                "folds_positivos": sum(1 for v in delta_fold if v > 0),
                "violaciones_monotonia": (violaciones_v20 if nombre.startswith("V20")
                                          else (violaciones_coral
                                                if nombre.startswith("CORAL") else "no medido")),
                "salvage": comp["solo_candidato"], "damage": comp["solo_control"],
                "neto": comp["neto"], "cambios": comp["cambios_de_prediccion"],
                "ambos_aciertos": comp["ambos_aciertos"], "ambos_fallos": comp["ambos_fallos"],
                "f1_por_clase": {c: round(float(v), 4) for c, v in zip(
                    cfg.CLASSES, f1_score(clase, pred, average=None, labels=ETIQUETAS))},
                "frontera_18_24_25_34": round(float(fronteras["18-24|25-34"]["accuracy"]), 4),
                "frontera_25_34_35_49": round(float(fronteras["25-34|35-49"]["accuracy"]), 4),
                "complementariedad_por_clase": comp["por_clase"]}
        filas.append(item)
        detalle[nombre] = item
        print(f"[{nombre}] {item['macro_f1']:.7f} | delta {item['delta_vs_v20']:+.6f} | "
              f"folds +{item['folds_positivos']}/5 | neto {item['neto']:+,}", flush=True)
    pd.DataFrame([{k: v for k, v in f.items()
                   if k not in {"f1_por_clase", "complementariedad_por_clase", "scores_por_fold",
                                "delta_por_fold"}} for f in filas]).to_csv(TABLA, index=False)

    # ---------------------------------------------------------------- matrices de confusion
    matrices = {"V20 (control A)": matriz_confusion(clase, np.argmax(oof_v20, axis=1)),
                "CORAL pesos compartidos": matriz_confusion(clase, np.argmax(seis_coral, axis=1))}
    adyacentes = [("18-24", "25-34"), ("25-34", "35-49"), ("35-49", "50-64")]
    extremas = [("13-17", "50-64"), ("18-24", "65-xx"), ("13-17", "65-xx")]
    confusion = {}
    for nombre, matriz in matrices.items():
        fuera = {}
        for clase_a, clase_b in adyacentes + extremas:
            i, j = cfg.CLASSES.index(clase_a), cfg.CLASSES.index(clase_b)
            fuera[f"{clase_a}->{clase_b}"] = int(matriz[i, j])
            fuera[f"{clase_b}->{clase_a}"] = int(matriz[j, i])
        confusion[nombre] = fuera
    # Geometria del error. La suma de reducciones y la suma de aumentos son **iguales por
    # construccion** (cada fila de una matriz de confusion suma lo mismo en los dos modelos), luego
    # no dicen nada. Se reportan las magnitudes que si informen.
    m_v20 = matrices["V20 (control A)"]
    m_coral = matrices["CORAL pesos compartidos"]
    geometria = {
        "aciertos_v20": int(np.trace(m_v20)),
        "aciertos_coral": int(np.trace(m_coral)),
        "delta_aciertos": int(np.trace(m_coral) - np.trace(m_v20)),
        "celdas_modificadas": int((m_v20 != m_coral).sum()),
        "distancia_l1": int(np.abs(m_v20 - m_coral).sum()),
        "aciertos_fuera_de_la_frontera_1_v20": int(m_v20[1, 2] + m_v20[2, 1]),
        "aciertos_fuera_de_la_frontera_1_coral": int(m_coral[1, 2] + m_coral[2, 1]),
    }
    print(f"[geometria] aciertos V20 {geometria['aciertos_v20']:,} | CORAL "
          f"{geometria['aciertos_coral']:,} | delta {geometria['delta_aciertos']:+,} | "
          f"celdas que cambian {geometria['celdas_modificadas']}/36 | L1 "
          f"{geometria['distancia_l1']:,}", flush=True)

    # ---------------------------------------------------------------- gate y bootstrap
    coral_item = detalle["CORAL pesos compartidos"]
    b_item = detalle["cinco binarios lineales (control B)"]
    delta_coral = coral_item["delta_vs_v20"]
    cumple_delta = delta_coral >= GATE_DELTA
    cumple_folds = coral_item["folds_positivos"] >= GATE_FOLDS
    bootstrap = None
    if cumple_delta:
        intervalo = bootstrap_diferencia(oof_v20, seis_coral, clase, n_clases)
        # La tupla es (p2,5, p97,5, proporcion positiva) de la DIFERENCIA de macro-F1.
        bootstrap = {"delta": round(delta_coral, 6),
                     "ic95": [round(float(intervalo[1]), 6), round(float(intervalo[2]), 6)],
                     "proporcion_positiva": round(float(intervalo[2]), 6)}
        bootstrap["ic95_incluye_cero"] = bool(intervalo[1] <= 0.0 <= intervalo[2])
        print(f"[bootstrap] delta {bootstrap['delta']:+.6f} IC95 "
              f"[{bootstrap['ic95'][0]:+.6f}, {bootstrap['ic95'][1]:+.6f}]", flush=True)

    if not cumple_delta:
        decision = "CORAL = CLOSED"
        estado = "NO_SIGNAL"
    elif cumple_folds and bootstrap is not None and not bootstrap["ic95_incluye_cero"]:
        decision = "CORAL = CANDIDATE (local; requiere segunda particion y revision)"
        estado = "CANDIDATE"
    else:
        decision = "CORAL = MARGINAL"
        estado = "MARGINAL"

    # ---------------------------------------------------------------- figuras
    _figura_monotonia(acumuladas_v20_crudas, cdf_coral, violaciones_v20, violaciones_coral)
    _figura_complementariedad(detalle)
    _figura_folds(detalle, valor_v20)

    _escribe_informe(valor_v20, reproduce, violaciones_v20, violaciones_coral, filas, detalle,
                     matrices, confusion, geometria, delta_coral,
                     cumple_delta, cumple_folds, bootstrap, decision, fallos_coral)
    RESULTADO.write_text(json.dumps({
        "fase": "V23 01 CORAL", "fecha": "2026-09-27",
        "champion_congelado": {"nombre": "V20", "lb": LB_V20, "cv": CV_V20, "intacto": True},
        "hipotesis": ("compartir el vector de pesos entre las cinco fronteras ordinales y ordenar "
                      "los umbrales aporta senal util que cinco binarios independientes no capturan"),
        "variable_unica": "pesos compartidos frente a independientes",
        "pre_registrado": {"lambda_w": LAMBDA_W, "lambda_umbral": LAMBDA_UMBRAL,
                           "C_logistico_control_B": C_LOGISTICO, "n_splits": 5,
                           "agrupacion": "dup_clusters, StratifiedGroupKFold"},
        "control_a": {"macro_f1": round(valor_v20, 7), "reproducido": reproduce,
                      "violaciones_monotonia": violaciones_v20},
        "candidatos": detalle,
        "matrices_confusion": {k: v.tolist() for k, v in matrices.items()},
        "confusiones_fuera_de_diagonal": confusion,
        "geometria_del_error": geometria,
        "gate": {"delta_minimo": GATE_DELTA, "folds_minimos": GATE_FOLDS,
                 "delta_coral": delta_coral, "cumple_delta": cumple_delta,
                 "cumple_folds": cumple_folds, "bootstrap": bootstrap},
        "control_b_delta": b_item["delta_vs_v20"],
        "atribucion": ("CORAL contra el control B aísla el efecto de compartir pesos; el delta "
                       "contra V20 mezcla ese efecto con la perdida de no linealidad"),
        "fallos_de_convergencia": fallos_coral,
        "decision": decision, "estado": estado, "submission_generada": False,
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\ndecisión: {decision}")
    print(f"delta CORAL {delta_coral:+.6f} | gate delta {cumple_delta} | gate folds {cumple_folds}")
    return 0


def _figura_monotonia(crudas_v20: np.ndarray, cdf_coral: np.ndarray, vio_v20: int,
                      vio_coral: int) -> None:
    """Las cinco CDF de V20 frente a las de CORAL, y el reparto de violaciones."""
    fig, ejes = plt.subplots(1, 2, figsize=(14, 5.0))
    eje = ejes[0]
    muestra = np.argsort(crudas_v20[:, 0])[:4000]
    for k in range(crudas_v20.shape[1]):
        eje.plot(crudas_v20[muestra, k], marker=".", linestyle="none", markersize=2, alpha=0.35,
                 label=f"V20 umbral {k + 1}")
    for k in range(cdf_coral.shape[1]):
        eje.plot(cdf_coral[muestra, k], marker=".", linestyle="none", markersize=2, alpha=0.35,
                 label=f"CORAL umbral {k + 1}")
    eje.set_xlabel("4.000 filas ordenadas por la primera CDF")
    eje.set_ylabel("P(Y <= k)")
    eje.set_title("V20 se cruza; CORAL no puede cruzarse", fontsize=10)
    eje.legend(fontsize=6, ncol=2)
    eje = ejes[1]
    eje.bar(["V20", "CORAL"], [vio_v20, vio_coral], color=["#b2182b", "#1a9850"])
    for indice, valor in enumerate([vio_v20, vio_coral]):
        eje.text(indice, valor, f"{valor:,}", ha="center", va="bottom", fontsize=9)
    eje.set_ylabel("filas con la serie de CDF decreciente")
    eje.set_title("CORAL: 0 por construccion", fontsize=10)
    fig.suptitle("V23: violations de monotonia de las cinco CDF", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_MONO, dpi=130)
    plt.close(fig)


def _figura_complementariedad(detalle: dict) -> None:
    """Salvage, damage y neto de cada candidato frente a V20."""
    nombres = list(detalle)
    fig, ejes = plt.subplots(1, 2, figsize=(14, 5.0))
    eje = ejes[0]
    eje.bar(np.arange(len(nombres)) - 0.2, [detalle[n]["salvage"] for n in nombres], 0.4,
            label="rescata", color="#1a9850")
    eje.bar(np.arange(len(nombres)) + 0.2, [detalle[n]["damage"] for n in nombres], 0.4,
            label="danio", color="#b2182b")
    eje.set_xticks(range(len(nombres)))
    eje.set_xticklabels([n.replace(" (", "\n(") for n in nombres], fontsize=7)
    eje.set_ylabel("filas")
    eje.set_title("Rescates y danos frente a V20", fontsize=10)
    eje.legend(fontsize=8)
    eje = ejes[1]
    eje.bar(range(len(nombres)), [detalle[n]["neto"] for n in nombres],
            color=["#2166ac" if detalle[n]["neto"] < 0 else "#1a9850" for n in nombres])
    eje.axhline(0, color="grey", linewidth=1)
    eje.set_xticks(range(len(nombres)))
    eje.set_xticklabels([n.replace(" (", "\n(") for n in nombres], fontsize=7)
    eje.set_ylabel("neto de filas")
    eje.set_title("Neto", fontsize=10)
    fig.suptitle("V23: complementariedad de cada modelo con V20", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_COMP, dpi=130)
    plt.close(fig)


def _figura_folds(detalle: dict, valor_v20: float) -> None:
    """Delta por fold de cada candidato: la estabilidad es parte del gate."""
    nombres = list(detalle)
    fig, eje = plt.subplots(figsize=(9, 4.5))
    ancho = 0.35
    for indice, nombre in enumerate(nombres):
        eje.bar(np.arange(5) + (indice - 1) * ancho, detalle[nombre]["delta_por_fold"], ancho,
                label=nombre.replace(" (", "\n("))
    eje.axhline(0, color="grey", linewidth=1)
    eje.axhline(GATE_DELTA, color="#b2182b", linestyle="--", linewidth=1,
                label=f"gate delta {GATE_DELTA}")
    eje.set_xticks(range(5))
    eje.set_xticklabels([f"fold {i + 1}" for i in range(5)])
    eje.set_ylabel("delta de macro-F1 frente a V20")
    eje.set_title(f"V23: estabilidad por fold (V20 = {valor_v20:.7f})", fontsize=10)
    eje.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_FOLD, dpi=130)
    plt.close(fig)


def _escribe_informe(valor_v20, reproduce, vio_v20, vio_coral, filas, detalle, matrices,
                     confusion, geometria, delta_coral, cumple_delta, cumple_folds,
                     bootstrap, decision, fallos_coral) -> None:
    """Redacta el informe de analisis con las siete secciones que pide el brief."""
    cuerpo = [
        "# V23 · CORAL / cumulative-link con pesos compartidos", "",
        "## A. El control V20", "",
        f"- Reproduccion: **{valor_v20:.7f}** frente a {CV_V20:.7f} | coincide: **{reproduce}**.",
        f"- Leaderboard del champion: **{LB_V20}**. No se ha modificado V20.",
        f"- Violaciones de monotonia de sus cinco CDF: **{vio_v20:,}** de {N_FILAS:,} "
        f"({vio_v20 / N_FILAS:.2%}).", "",
        "## B. La formulacion", "",
        "```", "P(Y <= k | x) = sigmoid(theta_k - w' x),   k = 0..4",
        "theta_k = gamma + suma_{j<k} softplus(delta_j)", "```", "",
        "Un unico vector de pesos y cinco umbrales **ordenados por construccion**. Es distinto de "
        "V20 (cinco binarios independientes) y distinto del modelo compartido de D-057 (multiclass "
        "descompuesto por suma, sin umbrales ordinales).", "",
        "**Pre-registrado, sin tocar AGE**: `lambda_w = 1.0`, `lambda_umbral = 0.0`. Los fija el "
        "TEST B de `scripts/v23_00_tests_coral.py`, que demuestra que a la escala de AGE no hay "
        "separacion.", "",
        "## C. Resultado", "",
        "| Modelo | macro-F1 | Delta | Folds + | monotonia | rescate | dano | neto | cambios |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for item in filas:
        mono = item["violaciones_monotonia"]
        mono_texto = f"{mono:,}" if isinstance(mono, int) else str(mono)
        cuerpo.append(f"| {item['modelo']} | {item['macro_f1']:.7f} | {item['delta_vs_v20']:+.6f} "
                      f"| {item['folds_positivos']}/5 | {mono_texto} | {item['salvage']:,} "
                      f"| {item['damage']:,} | {item['neto']:+,} | {item['cambios']:,} |")
    cuerpo += ["", "### Scores por fold", "",
               "| Modelo | " + " | ".join(f"fold {i + 1}" for i in range(5)) + " | media |",
               "|---" * 7 + "|"]
    for item in filas:
        cuerpo.append(f"| {item['modelo']} | "
                      + " | ".join(f"{v:.7f}" for v in item["scores_por_fold"])
                      + f" | {item['media_folds']:.6f} |")
    cuerpo += ["", "### Delta por fold", "",
               "| Modelo | " + " | ".join(f"fold {i + 1}" for i in range(5)) + " | positivos |",
               "|---" * 7 + "|"]
    for item in filas:
        cuerpo.append(f"| {item['modelo']} | "
                      + " | ".join(f"{v:+.6f}" for v in item["delta_por_fold"])
                      + f" | {item['folds_positivos']}/5 |")
    cuerpo += ["", "## D. F1 por clase", "",
               "| Modelo | " + " | ".join(cfg.CLASSES) + " |", "|---" * 7 + "|"]
    for item in filas:
        cuerpo.append(f"| {item['modelo']} | "
                      + " | ".join(f"{item['f1_por_clase'][c]:.4f}" for c in cfg.CLASSES) + " |")
    cuerpo += ["", "## E. Monotonicidad: el problema que dice resolver", "",
               f"- **V20**: {vio_v20:,} filas con la serie de CDF decreciente.",
               f"- **CORAL**: **{vio_coral:,}** filas, y no por ajuste sino **por construccion**, "
               "porque `softplus(delta_j) > 0` obliga a `theta` a crecer y con el una sola `w` las "
               "cinco CDF heredan el orden.", "",
               "**La monotonia queda resuelta de verdad. La pregunta es si eso cambia el macro-F1.**",
               "", "## F. Intersecion de errores", "",
               "| Par | V20 | CORAL |", "|---|---|---|"]
    for clave in confusion["V20 (control A)"]:
        cuerpo.append(f"| {clave} | {confusion['V20 (control A)'][clave]:,} "
                      f"| {confusion['CORAL pesos compartidos'][clave]:,} |")
    cuerpo += ["", f"Masa en la diagonal (aciertos): V20 **{geometria['aciertos_v20']:,}**, "
                  f"CORAL **{geometria['aciertos_coral']:,}** "
                  f"(delta {geometria['delta_aciertos']:+,}). "
                  f"Celdas de la matriz de confusion que cambian: "
                  f"**{geometria['celdas_modificadas']}/36**, con distancia L1 "
                  f"**{geometria['distancia_l1']:,}**.", ""]
    cuerpo += ["## G. Decision", "",
               f"- Delta de CORAL frente a V20: **{delta_coral:+.6f}** (gate: >= {GATE_DELTA}) -> "
               f"**{cumple_delta}**.",
               f"- Folds positivos: **{detalle['CORAL pesos compartidos']['folds_positivos']}/5** "
               f"(gate: >= {GATE_FOLDS}) -> **{cumple_folds}**."]
    if bootstrap is not None:
        cuerpo += [f"- Bootstrap pareado: delta {bootstrap['delta']:+.6f}, IC95 "
                   f"[{bootstrap['ic95'][0]:+.6f}, {bootstrap['ic95'][1]:+.6f}], incluye cero: "
                   f"**{bootstrap['ic95_incluye_cero']}**."]
    else:
        cuerpo += ["- Bootstrap: **no se calcula**, porque el gate de delta no se cumple y el "
                   "brief lo reserva al caso en que CORAL lo supera."]
    cuerpo += ["", f"**{decision}**", ""]
    if fallos_coral:
        cuerpo += ["Aviso: L-BFGS-B no convergio en " + ", ".join(fallos_coral), ""]
    cuerpo += ["## H. Atribucion", "",
               f"- El control B (cinco binarios **independientes** y lineales) da "
               f"**{detalle['cinco binarios lineales (control B)']['delta_vs_v20']:+.6f}** frente a V20.",
               f"- CORAL (peso **compartido**, lineal) da **{delta_coral:+.6f}** frente a V20.",
               "- La diferencia entre CORAL y el control B es el efecto **puro** de compartir "
               "pesos. La diferencia entre el control B y V20 es el efecto de la no linealidad.", ""]
    INFORME.write_text("\n".join(cuerpo), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
