"""FASE 33 — Corrector de error muy pequeño sobre P(V12) congelada (experimento I).

Hipótesis
    I/ORD-5  Una capa de corrección **muy restringida**, entrenada solo con las predicciones OOF de
    V12, puede capturar lo que la regla de argmax no convierte en clase. Entrada: P(V12), top-1,
    top-2, margen, entropía, distancia ordinal entre las dos clases dominantes, longitud, tipo de
    tweet y la vista estructural ya evaluada. Salida: corrección ordinal.
    Modelos: LogReg multinomial (pequeño), LogReg sobre log-probabilidades, Ridge binario restringido
    al par top-2 e isotónica por clase. **No** se usa un MLP nuevo ni features textuales nuevas.

Protocolo
    corrected CV (LOFO): el corrector se ajusta con las filas OOF de los otros cuatro folds y se
    aplica al retenido. Gate: Δ ≥ +0,003 y ≥4/5 folds.

Advertencia de metodología (documentada en el informe)
    Las probabilidades base de las filas de entrenamiento proceden de modelos que sí vieron el fold de
    validación; es la estructura estándar de *stacking* y la misma que usa el combinador de V12, por lo
    que la comparación **pareada** contra V12 es coherente, aunque el corrector dispone de un nivel de
    información que el meta-modelo no tenía (las etiquetas de las filas de entrenamiento ya vistas por
    los modelos base). Se reporta como limitación.

Salida: ``experiments/phase33_corrector_minimo.json``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import decision_layer as dl  # noqa: E402
from src.config import set_seed  # noqa: E402

RESULTS_PATH = cfg.EXPERIMENTS / "phase33_corrector_minimo.json"
ENTROPIA = float(np.log(len(cfg.CLASSES)))


def matriz_caracteristicas(dls: dict) -> np.ndarray:
    """Features del corrector: P(V12) + confianza + contexto + vista estructural."""
    oof = dls["oof"]
    orden = np.argsort(-oof, axis=1)
    p1 = oof[np.arange(len(oof)), orden[:, 0]]
    p2 = oof[np.arange(len(oof)), orden[:, 1]]
    entropia = -(oof * np.log(np.clip(oof, 1e-12, None))).sum(axis=1)
    contexto = dls["contexto"]
    bloque_contexto = np.column_stack([
        contexto["n_chars"], contexto["n_mention"], contexto["n_hashtag"], contexto["n_url"],
        contexto["is_rt"], contexto["es_grupo_solo_url"], contexto["es_grupo_muy_corto"],
        contexto["es_grupo_sin_menciones"]]).astype(float)
    escala = bloque_contexto.std(axis=0)
    escala = np.where(escala < 1e-9, 1.0, escala)
    bloque_contexto = (bloque_contexto - bloque_contexto.mean(axis=0)) / escala
    columnas = [oof, (p1 - p2)[:, None], entropia[:, None],
                np.abs(orden[:, 0] - orden[:, 1])[:, None], bloque_contexto]
    if dls["estructura"] is not None:
        columnas.append(dls["estructura"])
    return np.column_stack(columnas)


def construir_logreg(dls: dict, usar_log: bool, regularizacion: float) -> object:
    """Corrector LogReg multinomial sobre P(V12) (o sobre log-probabilidades)."""
    matriz = matriz_caracteristicas(dls)
    if usar_log:
        matriz = np.log(np.clip(dls["oof"], 1e-9, None))
        matriz = np.column_stack([matriz, matriz_caracteristicas(dls)])
    escala = matriz.std(axis=0)
    escala = np.where(escala < 1e-9, 1.0, escala)
    matriz = (matriz - matriz.mean(axis=0)) / escala

    def construir(train_idx: np.ndarray, sustrato: dict) -> object:
        modelo = Pipeline([("scaler", StandardScaler()),
                           ("clf", LogisticRegression(C=1.0 / regularizacion, max_iter=3000,
                                                      random_state=cfg.SEED))])
        modelo.fit(matriz[train_idx], sustrato["indices"][train_idx])

        def aplicar(filas: np.ndarray) -> np.ndarray:
            probabilidades = modelo.predict_proba(matriz[filas])
            completa = np.zeros((len(filas), len(cfg.CLASSES)))
            completa[:, modelo.classes_.astype(int)] = probabilidades
            return completa

        return aplicar

    return construir


def construir_ridge_top2(dls: dict, regularizacion: float) -> object:
    """Corrector Ridge binario sobre el par top-1/top-2 (corrige solo la proporción del par)."""
    matriz = matriz_caracteristicas(dls)
    escala = matriz.std(axis=0)
    escala = np.where(escala < 1e-9, 1.0, escala)
    matriz = (matriz - matriz.mean(axis=0)) / escala

    def construir(train_idx: np.ndarray, sustrato: dict) -> object:
        oof = sustrato["oof"]
        orden = np.argsort(-oof, axis=1)
        par = np.abs(orden[:, 0] - orden[:, 1]) == 1
        filas = train_idx[par[train_idx]]
        objetivo = (sustrato["indices"][filas] == orden[filas, 0]).astype(int)
        if len(filas) < 500 or len(np.unique(objetivo)) < 2:
            objetivo = np.ones(len(filas), dtype=int)
        modelo = Ridge(alpha=regularizacion, random_state=cfg.SEED)
        modelo.fit(matriz[filas], objetivo.astype(float))

        def aplicar(filas_aplicar: np.ndarray) -> np.ndarray:
            salida = oof[filas_aplicar].copy()
            orden_fila = orden[filas_aplicar]
            seleccion = par[filas_aplicar]
            if not seleccion.any():
                return salida
            locales = np.where(seleccion)[0]
            predicciones = np.clip(modelo.predict(matriz[filas_aplicar][seleccion]), 0.02, 0.98)
            top1 = orden_fila[seleccion, 0]
            top2 = orden_fila[seleccion, 1]
            masa = salida[locales, top1] + salida[locales, top2]
            salida[locales, top1] = masa * predicciones
            salida[locales, top2] = masa * (1.0 - predicciones)
            return salida / salida.sum(axis=1, keepdims=True)

        return aplicar

    return construir


def construir_isotonica(dls: dict) -> object:
    """Calibración isotónica por clase sobre P(V12) (monótona, ajustada con OOF)."""
    def construir(train_idx: np.ndarray, sustrato: dict) -> object:
        oof = sustrato["oof"]
        indices = sustrato["indices"]
        modelos = []
        for k in range(len(cfg.CLASSES)):
            modelo = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            modelo.fit(oof[train_idx, k], (indices[train_idx] == k).astype(float))
            modelos.append(modelo)

        def aplicar(filas: np.ndarray) -> np.ndarray:
            salida = np.column_stack([modelos[k].predict(oof[filas, k])
                                      for k in range(len(cfg.CLASSES))])
            salida = np.clip(salida, 1e-6, None)
            return salida / salida.sum(axis=1, keepdims=True)

        return aplicar

    return construir


def main() -> int:
    """Evalúa el corrector mínimo (I) en sus cuatro parametrizaciones."""
    set_seed()
    started = time.time()
    sustrato = dl.cargas_sustrato()
    dls = sustrato
    print(f"[sustrato] base OOF macro-F1 {dl.metricas(dls['oof'], dls['indices'])['macro_f1']}",
          flush=True)
    resultados = []

    def registrar(etiqueta, construir):
        print(f"[{etiqueta}]", flush=True)
        r = dl.evaluar_capa(etiqueta, construir, sustrato)
        resultados.append(r)
        c = r["complementariedad"]
        print(f"  {r['macro_f1']} (Δ {r['delta']:+.4f}, {r['folds_positivos']}/5 | rescate "
              f"{c['solo_candidato']} daño {c['solo_base']} neto {c['neto']} | modif "
              f"{c['filas_modificadas_pct']}%)", flush=True)

    registrar("I1 corrector LogReg sobre P(V12)", construir_logreg(dls, False, 1.0))
    registrar("I2 corrector LogReg sobre log-probabilidades", construir_logreg(dls, True, 1.0))
    registrar("I3 corrector LogReg (más regularizado)", construir_logreg(dls, False, 0.1))
    registrar("I4 corrector Ridge sobre el par top-2", construir_ridge_top2(dls, 10.0))
    registrar("I5 calibración isotónica por clase", construir_isotonica(dls))

    adopted = [r for r in resultados if r["adoptado"]]
    payload = {"fase": "33 — corrector mínimo (I)", "fecha": "2026-09-25",
               "resultados": resultados, "criterio": "Δ ≥ +0,003 y ≥4/5 folds positivos",
               "adoptadas": [r["capa"] for r in adopted],
               "segundos": round(time.time() - started, 1)}
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                            encoding="utf-8")
    print(f"\nadoptadas: {len(adopted)} -> {[r['capa'] for r in adopted]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
