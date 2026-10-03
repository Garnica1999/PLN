"""FASE 22 — Refuerzo de la vista estructural (seguimiento de la única señal positiva de la fase).

Observación
    La fase 21 midió la vista **estructural** (secuencia de tipos de token, sin contenido): F1
    individual 0,2285, rescate del **18,17%** de los errores del v12 y acuerdo 27,96%; añadida como
    miembro del meta, **+0,0018 con 4/5 folds positivos** (0,6022 frente a 0,6004) — la única señal
    positiva de la búsqueda, pero por debajo del criterio de adopción (+0,003).

Hipótesis
    H58 (V1) Una alfabeto estructural más rico (longitud de palabra en cubetas, mayúsculas, dígitos,
         puntuación por clases y marcadores de frontera) mejora el rescate y el aporte al ensamble.
    H59 (V2) La vista estructural y la de extremos son complementarias entre sí (capturan forma global
         y registro de apertura/cierre).

Protocolo
    Mismas particiones agrupadas. Se cachean los márgenes OOF de cada vista en ``features/``. Puerta de
    diversidad: rescate ≥ 15% de los errores del v12. Valor en el ensemble: columnas añadidas al final
    del meta (mismo mecanismo que el NB-SVM de palabras), combinador de producción (promedio MLP +
    LogReg). Criterio de adopción: Δ ≥ +0,003 y ≥4/5 folds positivos.

Uso::

    python scripts/phase22_estructura_ampliada.py

Salida: ``experiments/phase22_estructura_ampliada.json``.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import cv, data  # noqa: E402
from src.config import set_seed  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from phase2_ponderacion import build_features  # noqa: E402
from phase21_vista_estructural import (PUERTA_RESCATE, TOKEN_RE, evaluar,  # noqa: E402
                                       oof_de_vista, vista_estructura, vista_extremos)

RESULTS_PATH = cfg.EXPERIMENTS / "phase22_estructura_ampliada.json"
PALABRA_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+")


def tipo_ampliado(token: str) -> str:
    """Tipo estructural con cubetas de longitud, mayúsculas y clases de puntuación."""
    if token.startswith("@"):
        return "@"
    if token.startswith("#"):
        return "#"
    if token.startswith(("http", "www")):
        return "U"
    if token.isdigit():
        return "N1" if len(token) == 1 else ("N2" if len(token) <= 3 else "N3")
    if PALABRA_RE.fullmatch(token):
        if token.isupper() and len(token) > 1:
            return "WU"
        if token[:1].isupper() and len(token) > 1:
            return "WC"
        if len(token) <= 3:
            return "W1"
        return "W2" if len(token) <= 6 else "W3"
    if token in ".,:;":
        return "PL"
    if token in "!?":
        return "PE"
    return "PS"


def vista_estructura_ampliada(textos: pd.Series) -> list[str]:
    """Secuencia de tipos estructurales ampliada, con marcadores de frontera del tweet."""
    documentos = []
    for valor in textos.fillna("").astype(str):
        tipos = [tipo_ampliado(token) for token in TOKEN_RE.findall(valor)]
        if not tipos:
            documentos.append("<S> </S>")
            continue
        documentos.append(" ".join(["<S>"] + tipos + ["</S>"]))
    return documentos


def margenes_de_vista(nombre: str, documentos: list[str], y: np.ndarray,
                      folds: list[dict]) -> np.ndarray:
    """Márgenes OOF de una vista, con caché en ``features/vista_<nombre>_margenes.npy``."""
    ruta = cfg.FEATURES / f"vista_{nombre}_margenes.npy"
    if ruta.exists():
        cacheado = np.load(ruta)
        if cacheado.shape == (len(documentos), len(cfg.CLASSES)):
            print(f"[{nombre}] márgenes cargados de caché", flush=True)
            return cacheado
    margenes = oof_de_vista(documentos, y, folds)
    np.save(ruta, margenes)
    return margenes


def main() -> int:
    """Evalúa las variantes reforzadas de la vista estructural."""
    set_seed()
    started = time.time()
    payload: dict = {"fase": "§13 multivista — refuerzo estructural", "fecha": "2026-09-25"}

    train = data.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    features, _, _, _, _ = build_features()
    features = np.concatenate([features, np.load(cfg.FEATURES / "oof_ADV07_nbsvm_word.npy")], axis=1)
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")
    folds = cv.make_folds(y, groups=clusters["cluster"].to_numpy()[:len(y)], n_splits=5,
                          n_repeats=1, scheme="group")
    referencia = json.loads((cfg.EXPERIMENTS / "phase12_auditoria_estado.json")
                            .read_text(encoding="utf-8"))["reproduccion_lofo_v12"]["resultado"]
    payload["baseline"] = {"macro_f1": referencia["macro_f1"], "std": referencia["std"],
                           "per_fold": referencia["per_fold"], "origen": "fase 12"}

    indice_real = np.array([cfg.CLASSES.index(valor) for valor in y])
    oof_v12 = np.load(cfg.FEATURES / "oof_ensemble_v12.npy")
    aciertos_v12 = np.argmax(oof_v12, axis=1) == indice_real
    errores_v12 = int((~aciertos_v12).sum())

    vistas = {
        "estructura": vista_estructura(train[cfg.TEXT_COL]),
        "ampliada": vista_estructura_ampliada(train[cfg.TEXT_COL]),
        "extremos": vista_extremos(train[cfg.TEXT_COL]),
    }
    margenes = {}
    for nombre, documentos in vistas.items():
        print(f"[{nombre}] OOF de la vista ({len(documentos):,} documentos)...", flush=True)
        margenes[nombre] = margenes_de_vista(nombre, documentos, y, folds)
        predicciones = np.argmax(margenes[nombre], axis=1)
        aciertos = predicciones == indice_real
        tasa = float((aciertos & ~aciertos_v12).sum() / max(1, errores_v12))
        payload.setdefault("vistas", []).append({
            "vista": nombre,
            "macro_f1_individual": round(float(f1_score(y, np.array(cfg.CLASSES)[predicciones],
                                                        average="macro", labels=cfg.CLASSES)), 4),
            "accuracy_individual": round(float(aciertos.mean()), 4),
            "rescate_pct_de_errores_v12": round(100 * tasa, 2),
            "acuerdo_con_v12_pct": round(
                100 * float((predicciones == np.argmax(oof_v12, axis=1)).mean()), 2),
            "puerta_superada": bool(tasa >= PUERTA_RESCATE),
        })
        print(f"[{nombre}] rescate {100 * tasa:.2f}% | puerta: {tasa >= PUERTA_RESCATE}", flush=True)

    variantes = {
        "V1 estructura ampliada": ["ampliada"],
        "V2 estructura + extremos": ["estructura", "extremos"],
        "V3 ampliada + extremos": ["ampliada", "extremos"],
    }
    resultados = []
    for etiqueta, componentes in variantes.items():
        arreglo = np.concatenate([features] + [margenes[nombre] for nombre in componentes], axis=1)
        print(f"[{etiqueta}] {arreglo.shape}", flush=True)
        veredicto = evaluar(arreglo, y, folds)
        positivos = sum(1 for nuevo, viejo in zip(veredicto["per_fold"], referencia["per_fold"])
                        if nuevo > viejo)
        delta = round(veredicto["macro_f1"] - referencia["macro_f1"], 4)
        resultados.append({"variante": etiqueta, "componentes": componentes,
                           "n_features": int(arreglo.shape[1]), **veredicto,
                           "delta_vs_v12": delta, "folds_positivos": positivos,
                           "adoptado": bool(delta >= 0.003 and positivos >= 4)})
        print(f"[{etiqueta}] {veredicto['macro_f1']:.4f} ({delta:+.4f}) · {positivos}/5 folds",
              flush=True)

    payload["variantes"] = resultados
    mejor = max(resultados, key=lambda fila: fila["macro_f1"])
    payload["mejor"] = mejor["variante"]
    payload["criterio"] = "Δ ≥ +0,003 y ≥4/5 folds positivos"
    payload["segundos"] = round(time.time() - started, 1)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                            encoding="utf-8")
    print(f"\nmejor: {mejor['variante']} = {mejor['macro_f1']:.4f} "
          f"({mejor['delta_vs_v12']:+.4f}) | adoptado: {mejor['adoptado']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
