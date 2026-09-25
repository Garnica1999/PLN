"""FASE 32 — Corrección probabilística suave de fronteras (experimentos C, D, H y B).

Hipótesis (nuevas respecto a lo ya evaluado; ver ``reports/age_hypothesis_coverage.md``)
    C/ORD-4  **Corrección suave de fronteras**: en las filas cuyo top-1 y top-2 son adyacentes, la masa
             del par se redistribuye con un modelo logístico pequeño que combina la evidencia base
             (logit de las dos probabilidades) con señales de contexto que V12 ya conoce pero su
             argmax no convierte en clase. No hay routing ni umbrales duros: es una rec-calibración
             *dentro* del par, con regularización L2. Hipótesis: el error de frontera se concentra en
             pares concretos y el contexto (longitud, tipo, menciones, URL, retuit) informa la
             proporción correcta dentro del par.
    D/ORD-3  **Decodificación por matriz de transición**: los sesgos de confusión son sistemáticos;
             una matriz M estimada con OOF y encogida hacia la identidad corrige argmax_j (M' p).
             Hipótesis: la decodificación consciente del confusionamiento mejora el argmax.
    H/ORD-8  **Calibración por banda de confianza**: el escalado por banda (Platt por clase dentro de
             quintiles de confianza) puede cambiar el argmax a diferencia de una temperatura global
             (cerrada en D-029). Hipótesis: la fiabilidad del top-1 no es homogénea.
    B/ORD-B  **Combinación probabilística de los márgenes por pares**: los 5 márgenes adyacentes del
             miembro E-028a ya entrenado se combinan con un modelo de Luce/Bradley-Terry para obtener
             una distribución de 6 clases coherente, y se fusiona con P(V12). No se entrena nada nuevo:
             es la misma evidencia de fronteras, usada por fin con un mecanismo probabilístico.

Protocolo
    corrected CV (LOFO). Gate: Δ ≥ +0,003 y ≥4/5 folds. Holdout no se utiliza (agotado).

Salida: ``experiments/phase32_correccion_probabilistica.json``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import decision_layer as dl  # noqa: E402
from src.config import set_seed  # noqa: E402

RESULTS_PATH = cfg.EXPERIMENTS / "phase32_correccion_probabilistica.json"


def contexto_extra(dls: dict) -> np.ndarray:
    """Matriz de contexto estandarizada para la corrección de fronteras (solo señales existentes)."""
    contexto = dls["contexto"]
    columnas = [contexto["n_chars"], contexto["n_mention"], contexto["n_hashtag"],
                contexto["n_url"], contexto["is_rt"], contexto["es_grupo_solo_url"],
                contexto["es_grupo_muy_corto"], contexto["es_grupo_sin_menciones"]]
    matriz = np.column_stack(columnas).astype(float)
    escala = matriz.std(axis=0)
    escala = np.where(escala < 1e-9, 1.0, escala)
    return (matriz - matriz.mean(axis=0)) / escala


def construir_ord4(dls: dict, usar_estructura: bool, gamma: float,
                   alpha: float) -> object:
    """Corrección suave dentro del par adyacente top-1/top-2 (ORD-4), formulada como **residual**.

    El logit base del par se mantiene intacto y el modelo solo aporta un nudge
    ``logit_final = logit_base + gamma * f(contexto)``, con ``f`` un Ridge ajustado **solo** con las
    filas de entrenamiento del fold sobre el residuo ``y - sigmoid(logit_base)``. Así la corrección
    es pequeña, regularizada y no puede reescribir la decisión.
    """
    def construir(train_idx: np.ndarray, sustrato: dict) -> object:
        oof = sustrato["oof"]
        reales = sustrato["indices"]
        contexto = contexto_extra(sustrato)
        estructura = sustrato["estructura"]
        objetivo = (reales == np.argmax(oof, axis=1)).astype(int)
        orden = np.argsort(-oof, axis=1)
        filas_idx = np.arange(len(oof))
        adyacente = np.abs(orden[:, 0] - orden[:, 1]) == 1
        logit_base = np.log(np.clip(oof[filas_idx, orden[:, 0]], 1e-9, None)) - np.log(
            np.clip(oof[filas_idx, orden[:, 1]], 1e-9, None))
        residuo = objetivo - 1.0 / (1.0 + np.exp(-logit_base))
        bloques = [contexto]
        if usar_estructura and estructura is not None:
            bloques.append(np.column_stack([
                estructura[filas_idx, orden[:, 0]] - estructura[filas_idx, orden[:, 1]],
                np.log(np.clip(np.abs(estructura[filas_idx, orden[:, 0]]), 1e-9, None))]))
        matriz = np.column_stack(bloques)
        seleccion_entreno = adyacente[train_idx]
        modelo = Ridge(alpha=alpha, random_state=cfg.SEED)
        modelo.fit(matriz[train_idx][seleccion_entreno], residuo[train_idx][seleccion_entreno])

        def aplicar(filas: np.ndarray) -> np.ndarray:
            p = oof[filas]
            orden_fila = orden[filas]
            salida = p.copy()
            seleccion = adyacente[filas]
            if not seleccion.any():
                return salida
            nudge = gamma * modelo.predict(matriz[filas][seleccion])
            logit = logit_base[filas][seleccion] + nudge
            probabilidad = 1.0 / (1.0 + np.exp(-logit))
            locales = np.where(seleccion)[0]
            top1 = orden_fila[seleccion, 0]
            top2 = orden_fila[seleccion, 1]
            masa = p[locales, top1] + p[locales, top2]
            salida[locales, top1] = masa * probabilidad
            salida[locales, top2] = masa * (1.0 - probabilidad)
            return salida / salida.sum(axis=1, keepdims=True)

        return aplicar

    return construir


def construir_ord3(dls: dict, lambda_shrink: float, condicionar: bool) -> object:
    """Decodificación por matriz de transición con shrinkage hacia la identidad (ORD-3)."""
    def construir(train_idx: np.ndarray, sustrato: dict) -> object:
        oof = sustrato["oof"]
        reales = sustrato["indices"]
        contexto = sustrato["contexto"]
        matrices = {}
        predichos = np.argmax(oof, axis=1)
        grupos = {False: train_idx}
        if condicionar:
            grupo_riesgo = ((contexto["es_grupo_solo_url"].to_numpy()[train_idx] > 0)
                            | (contexto["es_grupo_muy_corto"].to_numpy()[train_idx] > 0))
            grupos = {False: train_idx[~grupo_riesgo], True: train_idx[grupo_riesgo]}
        for riesgo, filas in grupos.items():
            matriz = np.eye(len(cfg.CLASSES))
            if riesgo not in {True, False}:
                continue
            for i in range(len(cfg.CLASSES)):
                if not (reales[filas] == i).any():
                    continue
                seleccion = filas[reales[filas] == i]
                conteo = np.bincount(predichos[seleccion],
                                     minlength=len(cfg.CLASSES)).astype(float)
                matriz[i] = conteo / conteo.sum()
            matrices[riesgo] = (1.0 - lambda_shrink) * np.eye(len(cfg.CLASSES)) + \
                lambda_shrink * matriz

        def aplicar(filas: np.ndarray) -> np.ndarray:
            p = oof[filas]
            riesgo = ((contexto["es_grupo_solo_url"].to_numpy()[filas] > 0)
                      | (contexto["es_grupo_muy_corto"].to_numpy()[filas] > 0)) if condicionar \
                else np.zeros(len(filas), dtype=bool)
            salida = np.zeros((len(filas), len(cfg.CLASSES)))
            for etiqueta in (False, True):
                locales = np.where(riesgo == etiqueta)[0]
                if len(locales) == 0:
                    continue
                matriz = matrices.get(etiqueta, matrices.get(False))
                salida[locales] = p[locales] @ matriz.T
            return dl.sigmoid(salida * 4.0)

        return aplicar

    return construir


def construir_ord8(dls: dict, n_bandas: int) -> object:
    """Calibración por banda de confianza (Platt por clase dentro de cuantiles) (ORD-8)."""
    def construir(train_idx: np.ndarray, sustrato: dict) -> object:
        oof = sustrato["oof"]
        reales = sustrato["indices"]
        confianza = oof[train_idx].max(axis=1)
        cortes = np.quantile(confianza, np.linspace(0, 1, n_bandas + 1)[1:-1])
        banda_train = np.digitize(confianza, cortes)

        def aplicar(filas: np.ndarray) -> np.ndarray:
            p = oof[filas]
            confianza_fila = p.max(axis=1)
            banda = np.digitize(confianza_fila, cortes)
            salida = p.copy()
            for b in np.unique(banda):
                filas_b = np.where(banda == b)[0]
                entrena = train_idx[banda_train == b]
                for k in range(len(cfg.CLASSES)):
                    if len(entrena) < 200 or len(np.unique(reales[entrena] == k)) < 2:
                        continue
                    modelo = LogisticRegression(C=1.0, max_iter=2000)
                    modelo.fit(np.log(np.clip(oof[entrena, k:k + 1], 1e-9, None)),
                               (reales[entrena] == k).astype(int))
                    salida[filas_b, k] = modelo.predict_proba(
                        np.log(np.clip(p[filas_b, k:k + 1], 1e-9, None)))[:, 1]
            return salida / salida.sum(axis=1, keepdims=True)

        return aplicar

    return construir


def construir_ordb(dls: dict, peso: float | None) -> object:
    """Combinación probabilística de los márgenes adyacentes (Luce/Bradley-Terry) (ORD-B)."""
    def construir(train_idx: np.ndarray, sustrato: dict) -> object:
        oof = sustrato["oof"]
        margenes = sustrato["auxiliares"][dl.PARES_ADYACENTES]  # 5 margins, signo = P(Y > k)
        if margenes.shape[1] != len(cfg.CLASSES) - 1:
            margenes = margenes[:, :len(cfg.CLASSES) - 1]

        def aplicar(filas: np.ndarray) -> np.ndarray:
            s = 1.0 / (1.0 + np.exp(-margenes[filas]))  # P(Y > k) por margen
            log_masa = np.zeros((len(filas), len(cfg.CLASSES)))
            log_masa[:, 0] = np.log(np.clip(1.0 - s[:, 0], 1e-9, None))
            for k in range(1, len(cfg.CLASSES) - 1):
                log_masa[:, k] = np.log(np.clip(s[:, k - 1], 1e-9, None)) + np.log(
                    np.clip(1.0 - s[:, k], 1e-9, None))
            log_masa[:, -1] = np.log(np.clip(s[:, -1], 1e-9, None))
            masa = np.exp(log_masa - log_masa.max(axis=1, keepdims=True))
            p_luce = masa / masa.sum(axis=1, keepdims=True)
            if peso is None:
                return p_luce
            mezcla = (1.0 - peso) * np.sqrt(np.clip(oof[filas], 1e-12, None)) + peso * np.sqrt(
                np.clip(p_luce, 1e-12, None))
            mezcla = mezcla ** 2
            return mezcla / mezcla.sum(axis=1, keepdims=True)

        return aplicar

    return construir


def main() -> int:
    """Ejecuta las capas de corrección probabilística (C, D, H, B)."""
    set_seed()
    started = time.time()
    sustrato = dl.cargas_sustrato()
    dls = sustrato
    print(f"[sustrato] OOF {dls['oof'].shape} | base OOF macro-F1 "
          f"{dl.metricas(dls['oof'], dls['indices'])['macro_f1']}", flush=True)
    resultados = []

    def registrar(etiqueta, construir):
        print(f"[{etiqueta}]", flush=True)
        r = dl.evaluar_capa(etiqueta, construir, sustrato)
        resultados.append(r)
        c = r["complementariedad"]
        print(f"  {r['macro_f1']} (Δ {r['delta']:+.4f}, {r['folds_positivos']}/5 | rescate "
              f"{c['solo_candidato']} daño {c['solo_base']} | modif {c['filas_modificadas_pct']}%)",
              flush=True)
        return r

    # C: corrección suave de fronteras.
    registrar("C1 corrección residual (solo contexto, gamma 1.0)",
              construir_ord4(dls, False, 1.0, 1.0))
    registrar("C2 corrección residual (gamma 0.5)", construir_ord4(dls, False, 0.5, 1.0))
    registrar("C3 corrección residual (+vista estructural, gamma 0.5)",
              construir_ord4(dls, True, 0.5, 1.0))
    registrar("C4 corrección residual (gamma 0.25, alpha 10)", construir_ord4(dls, True, 0.25, 10.0))

    # D: decodificación por matriz de transición.
    for lam in (0.1, 0.25, 0.5, 1.0):
        registrar(f"D decodificación transición (lambda {lam})",
                  construir_ord3(dls, lam, False))
    registrar("D decodificación transición (lambda 0.5, condicionada)",
              construir_ord3(dls, 0.5, True))

    # H: calibración por banda de confianza.
    for bandas in (3, 5):
        registrar(f"H calibración por banda de confianza ({bandas} bandas)",
                  construir_ord8(dls, bandas))

    # B: combinación probabilística de márgenes adyacentes (Luce).
    registrar("B1 Luce solo (márgenes E-028a)", construir_ordb(dls, None))
    for peso in (0.25, 0.5):
        registrar(f"B2 Luce + V12 (peso {peso})", construir_ordb(dls, peso))

    adopted = [r for r in resultados if r["adoptado"]]
    payload = {"fase": "32 — corrección probabilística (C, D, H, B)", "fecha": "2026-09-25",
               "resultados": resultados, "criterio": "Δ ≥ +0,003 y ≥4/5 folds positivos",
               "adoptadas": [r["capa"] for r in adopted],
               "segundos": round(time.time() - started, 1)}
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                            encoding="utf-8")
    print(f"\nadoptadas: {len(adopted)} -> {[r['capa'] for r in adopted]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
