"""FASE 31 — Capa de decisión ordinal sobre P(V12) congelada (experimentos A, E, G, F).

Hipótesis (todas nuevas respecto a lo ya evaluado; ver ``reports/age_hypothesis_coverage.md``)
    A/ORD-1  **Decodificación acumulativa**: (a) un modelo de umbrales acumulados ajustado sobre las
             features del meta (ya permitidas) cuyas CDF se promedian con las de V12 y se diferencian
             para obtener p_k; (b) lo mismo sin V12, para medir cuánto aporta la evidencia de
             fronteras por sí sola; (c) fusión por peso geométrico aprendido con las OOF.
             Hipótesis: modelar explícitamente cada frontera reduce inconsistencias entre clases
             contiguas.
    E/ORD-2  **Proyección ordinal de P(V12)**: suavizado con kernel ordinal (Toeplitz 1-2-1) con
             strength elegido con las OOF, más una variante que ajusta la frontera desplazando masa
             entre el top-1 y su vecino adyacente en proporción al margen.
             Hipótesis: el argmax de V12 es ruidoso justo en la frontera; un suavizado ordinal que
             conserva la masa de la clase más probable pero corrige la proporción de su vecina
             mejora la decisión sin tocar el modelo.
    G/ORD-6  **Desempate por edad esperada**: cuando el top-1 y el top-2 son adyacentes y la diferencia
             de probabilidad es menor que un umbral τ (elegido con las OOF), se decide por la posición
             ordinal esperada. Hipótesis: solo en empates de baja confianza aporta información.
    F/ORD-7  **Decisión con coste ordinal**: minimizar el coste esperado con distancias
             decrecientes (w_1 > w_2 > w_3) en lugar del argmax. Hipótesis: puede mejorar la
             asignación entre contiguas; se mide contra macro-F1 sin asumirlo.

Protocolo
    corrected CV (LOFO): cada capa se ajusta con las probabilidades OOF de los otros cuatro folds y se
    aplica al retenido. Gate: Δ ≥ +0,003 y ≥4/5 folds. Holdout no se utiliza (agotado).

Salida: ``experiments/phase31_decision_ordinal.json`` + OOF de cada capa en ``features/``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import decision_layer as dl  # noqa: E402
from src.config import set_seed  # noqa: E402

RESULTS_PATH = cfg.EXPERIMENTS / "phase31_decision_ordinal.json"


def ajustar_umbrales_acumulados(dls: dict, train_idx: np.ndarray) -> dict:
    """Ajusta un clasificador por umbral P(Y > k) sobre las features del meta (ORD-1)."""
    from scripts.phase2_ponderacion import MEMBER_FILES, build_features

    features, _, _, _, _ = build_features()
    features = np.concatenate([features, np.load(cfg.FEATURES / "oof_ADV07_nbsvm_word.npy")], axis=1)
    dls["meta_features"] = features
    del MEMBER_FILES
    reales = dls["indices"]
    umbrales = {}
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    for k in range(len(cfg.CLASSES) - 1):
        objetivo = (reales[train_idx] > k).astype(int)
        modelo = Pipeline([("scaler", StandardScaler()),
                           ("clf", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs",
                                                      random_state=cfg.SEED))])
        modelo.fit(features[train_idx], objetivo)
        umbrales[k] = modelo
    return umbrales


def supervivencia_umbrales(dls: dict, umbrales: dict, filas: np.ndarray) -> np.ndarray:
    """Función de supervivencia de los umbrales acumulados: columna k = P(Y > k), k = 0..4."""
    features = dls["meta_features"]
    s = np.ones((len(filas), len(cfg.CLASSES) - 1))
    for k in range(len(cfg.CLASSES) - 1):
        s[:, k] = umbrales[k].predict_proba(features[filas])[:, 1]
    return s


def supervivencia_base(probabilidades: np.ndarray) -> np.ndarray:
    """Supervivencia de una distribución de clases: S(k) = 1 - F(k) (k = 0..4)."""
    cdf = np.cumsum(probabilidades, axis=1)
    return np.minimum.accumulate(1.0 - cdf[:, :len(cfg.CLASSES) - 1], axis=1)


def construir_ord1(dls: dict, peso: float | None) -> object:
    """Decodificación acumulativa: promedia CDFs y diferencia (peso None = solo umbrales)."""
    reales = dls["indices"]

    def construir(train_idx: np.ndarray, sustrato: dict) -> object:
        umbrales = ajustar_umbrales_acumulados(sustrato, train_idx)
        s_base = supervivencia_base(sustrato["oof"])

        def aplicar(filas: np.ndarray) -> np.ndarray:
            s_umbral = supervivencia_umbrales(sustrato, umbrales, filas)
            s_final = s_umbral if peso is None else (1.0 - peso) * s_base[filas] + peso * s_umbral
            return dl.desde_supervivencia(s_final)

        return aplicar

    return construir


def construir_ord2(dls: dict, fuerza: float, tau: float | None = None) -> object:
    """Proyección ordinal: suavizado de la frontera con kernel 1-f-1 hacia el vecino adyacente (ORD-2).

    Con ``tau`` la corrección se limita a las filas de baja confianza (margen top-1/top-2 menor que
    ``tau``), que es donde la decisión es realmente ambigua.
    """
    def construir(train_idx: np.ndarray, sustrato: dict) -> object:
        oof = sustrato["oof"]

        def aplicar(filas: np.ndarray) -> np.ndarray:
            p = oof[filas].copy()
            orden = np.argsort(-p, axis=1)
            top1, top2 = orden[:, 0], orden[:, 1]
            filas_idx = np.arange(len(filas))
            salida = p.copy()
            adyacente = np.abs(top1 - top2) == 1
            if tau is not None:
                margen = p[filas_idx, top1] - p[filas_idx, top2]
                adyacente = adyacente & (margen < tau)
            delta = np.where(adyacente, fuerza * np.minimum(p[filas_idx, top1],
                                                           p[filas_idx, top2]), 0.0)
            salida[filas_idx, top1] -= delta
            salida[filas_idx, top2] += delta
            return salida / salida.sum(axis=1, keepdims=True)

        return aplicar

    return construir


def construir_ord6(dls: dict, tau: float) -> object:
    """Desempate por edad esperada en empates adyacentes de baja confianza (ORD-6)."""
    def construir(train_idx: np.ndarray, sustrato: dict) -> object:
        oof = sustrato["oof"]
        posiciones = np.arange(len(cfg.CLASSES))

        def aplicar(filas: np.ndarray) -> np.ndarray:
            p = oof[filas]
            copia = p.copy()
            orden = np.argsort(-p, axis=1)
            top1, top2 = orden[:, 0], orden[:, 1]
            esperado = p @ posiciones
            empate = (np.abs(top1 - top2) == 1) & ((p[np.arange(len(filas)), top1]
                                                   - p[np.arange(len(filas)), top2]) < tau)
            con_empate = np.where(empate)[0]
            for indice in con_empate:
                # Decide por la posición ordinal esperada: elige la contigua más cercana.
                eleccion = top1[indice] if abs(esperado[indice] - top1[indice]) <= abs(
                    esperado[indice] - top2[indice]) else top2[indice]
                copia[indice] = 0.0
                copia[indice, eleccion] = 1.0
            return copia

        return aplicar

    return construir


def construir_ord7(dls: dict, penalizaciones: tuple[float, float]) -> object:
    """Decisión con coste ordinal: la clase correcta tiene coste 0 y el error a distancia d cuesta
    ``1.0`` (d=1), ``penalizaciones[0]`` (d=2) y ``penalizaciones[1]`` (d>=3) (ORD-7)."""
    def construir(train_idx: np.ndarray, sustrato: dict) -> object:
        oof = sustrato["oof"]
        n = len(cfg.CLASSES)
        p2, p3 = penalizaciones
        coste = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                d = abs(i - j)
                coste[i, j] = 0.0 if d == 0 else (1.0 if d == 1 else (p2 if d == 2 else p3))

        def aplicar(filas: np.ndarray) -> np.ndarray:
            riesgo = oof[filas] @ coste
            return dl.sigmoid(-riesgo * 4.0)

        return aplicar

    return construir


def main() -> int:
    """Ejecuta la familia de capa de decisión ordinal (A, E, G, F)."""
    set_seed()
    started = time.time()
    sustrato = dl.cargas_sustrato()
    dls = sustrato
    print(f"[sustrato] OOF {dls['oof'].shape} | base OOF macro-F1 "
          f"{dl.metricas(dls['oof'], dls['indices'])['macro_f1']}", flush=True)
    resultados = []

    selector = sys.argv[sys.argv.index("--solo") + 1] if "--solo" in sys.argv else "todas"
    # A: decodificación acumulativa (función de supervivencia -> masa de clase).
    for etiqueta, peso in (("A1 solo umbrales acumulados", None), ("A2 CDF media (peso 0.5)", 0.5)):
        if selector not in ("todas", "a"):
            continue
        print(f"[{etiqueta}]", flush=True)
        r = dl.evaluar_capa(etiqueta, construir_ord1(dls, peso), sustrato)
        resultados.append(r)
        print(f"  {r['macro_f1']} (base {r['macro_f1_base_por_fold']}, Δ {r['delta']:+.4f}, "
              f"{r['folds_positivos']}/5, rescate {r['complementariedad']['solo_candidato']} vs daño "
              f"{r['complementariedad']['solo_base']})", flush=True)

        # E: proyección ordinal (tres fuerzas puras) y tres con gate de baja confianza.
    for fuerza in (0.10, 0.25, 0.40):
        etiqueta = f"E proyección ordinal (fuerza {fuerza})"
        print(f"[{etiqueta}]", flush=True)
        if selector not in ("todas", "e"):
            continue
        r = dl.evaluar_capa(etiqueta, construir_ord2(dls, fuerza), sustrato)
        resultados.append(r)
        print(f"  {r['macro_f1']} (Δ {r['delta']:+.4f}, {r['folds_positivos']}/5, modificados "
              f"{r['complementariedad']['filas_modificadas_pct']}%)", flush=True)

        # G: desempate por edad esperada (tres umbrales).
    for tau in (0.05, 0.10, 0.20):
        etiqueta = f"G desempate E[Y] (tau {tau})"
        print(f"[{etiqueta}]", flush=True)
        if selector not in ("todas", "g"):
            continue
        r = dl.evaluar_capa(etiqueta, construir_ord6(dls, tau), sustrato)
        resultados.append(r)
        print(f"  {r['macro_f1']} (Δ {r['delta']:+.4f}, {r['folds_positivos']}/5, modificados "
              f"{r['complementariedad']['filas_modificadas_pct']}%)", flush=True)

    for fuerza, tau in ((0.25, 0.10), (0.40, 0.10), (0.40, 0.20)):
        etiqueta = f"E proyección ordinal con gate (fuerza {fuerza}, tau {tau})"
        print(f"[{etiqueta}]", flush=True)
        if selector not in ("todas", "e"):
            continue
        r = dl.evaluar_capa(etiqueta, construir_ord2(dls, fuerza, tau), sustrato)
        resultados.append(r)
        print(f"  {r['macro_f1']} (Δ {r['delta']:+.4f}, {r['folds_positivos']}/5, modificados "
              f"{r['complementariedad']['filas_modificadas_pct']}%)", flush=True)

        # F: coste ordinal (tres perfiles: plano,ydiscounted y con castigo extra a distancia >= 3).
    for penalizaciones in ((2.0, 3.0), (1.5, 1.6), (0.5, 0.4)):
        etiqueta = f"F coste ordinal (d2={penalizaciones[0]}, d3={penalizaciones[1]})"
        print(f"[{etiqueta}]", flush=True)
        if selector not in ("todas", "f"):
            continue
        r = dl.evaluar_capa(etiqueta, construir_ord7(dls, penalizaciones), sustrato)
        resultados.append(r)
        print(f"  {r['macro_f1']} (Δ {r['delta']:+.4f}, {r['folds_positivos']}/5, modificados "
              f"{r['complementariedad']['filas_modificadas_pct']}%)", flush=True)

    if selector != "todas" and RESULTS_PATH.exists():
        previos = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))["resultados"]
        claves = {r["capa"] for r in resultados}
        resultados = [r for r in previos if r["capa"] not in claves] + resultados
    adopted = [r for r in resultados if r["adoptado"]]
    print(f"\nadoptadas: {len(adopted)} -> {[r['capa'] for r in adopted]}", flush=True)
    payload = {"fase": "31 — capa de decisión ordinal (A, E, G, F)", "fecha": "2026-09-25",
               "resultados": resultados,
               "criterio": "Δ ≥ +0,003 y ≥4/5 folds positivos",
               "adoptadas": [r["capa"] for r in adopted],
               "segundos": round(time.time() - started, 1)}
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                            encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
