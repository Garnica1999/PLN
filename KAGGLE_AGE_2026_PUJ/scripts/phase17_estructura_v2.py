"""FASE 17 — Estructura v2: RT con comentario, interior del hashtag, extremos y márgenes por frontera.

Observación
    El gap analysis v2 deja cuatro brechas explícitas que las features actuales no cubren:
    (a) ``rt_puro`` vs ``rt_con_comentario`` existe como **taxonomía** de segmentación (que fracasó)
    pero la estructura del comentario no entra al meta; (b) el interior del hashtag (longitud,
    camelCase, dígitos) no se representa, solo su perfil supervisado; (c) los márgenes de mención
    específicos por frontera nunca se calcularon; (d) los extremos del tweet (primeros/últimos
    caracteres) no tienen features explícitas.

Hipótesis
    H48 (E1) La estructura del comentario del retuit informa edad más allá de ``is_rt`` + longitud.
    H49 (E2) El interior del hashtag (forma, longitud, camelCase) informa más allá de su perfil.
    H50 (E3) El margen supervisado de las menciones **por par adyacente** (in-fold) ayuda en las
         fronteras donde se concentra el error.
    H51 (E4) Las transiciones de carácter en los extremos del tweet contienen estilo demográfico.

Protocolo
    Mismas particiones agrupadas; el combinador de producción (promedio MLP + LogReg) y referencia
    de la fase 12. Los bloques E1/E2/E4 son sin etiqueta (se calculan una vez sobre train+test); E3 se
    calcula in-fold. Uso: ``--bloque libres|supervisado|todos``.
    Criterio: Δ ≥ +0,003 y ≥4/5 folds positivos.

Salida: ``experiments/phase17_estructura_v2_<bloque>.json``.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import cv, data  # noqa: E402
from src.config import set_seed  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from phase2_ponderacion import EncodedMLP, build_features  # noqa: E402

#: Extracción del comentario propio en un retuit: todo lo anterior al marcador ``RT @``.
RT_PREFIX_RE = re.compile(r"^(?P<prefijo>.*?)\brt\s+@\w+", flags=re.IGNORECASE | re.DOTALL)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def entropia(valores: list[str]) -> float:
    """Entropía de Shannon (base 2) de los caracteres de una lista de cadenas."""
    if not valores:
        return 0.0
    texto = "".join(valores)
    if not texto:
        return 0.0
    conteos = np.array([texto.count(caracter) for caracter in set(texto)], dtype=float)
    proporcion = conteos / conteos.sum()
    return float(-(proporcion * np.log2(proporcion)).sum())


def features_libres(textos: pd.Series) -> tuple[np.ndarray, list[str]]:
    """Bloques E1 (RT con comentario), E2 (interior del hashtag) y E4 (extremos)."""
    valores = textos.fillna("").astype(str).to_numpy()
    filas = []
    for texto in valores:
        largo = max(len(texto), 1)
        coincidencia = RT_PREFIX_RE.match(texto)
        prefijo = coincidencia.group("prefijo") if coincidencia else ""
        sufijo = texto[len(prefijo):] if coincidencia else ""
        propio = re.match(r"\s*rt\s+@\w+\s*:?\s*", sufijo, flags=re.IGNORECASE)
        cuerpo = sufijo[propio.end():] if propio else ""
        prefijo_sin_espacios = prefijo.strip()
        etiquetas = HASHTAG_RE.findall(texto)
        extremo_izquierdo, extremo_derecho = texto[:5], texto[-5:]
        filas.append({
            # E1 — estructura del retuit
            "e1_es_rt": int(bool(coincidencia)),
            "e1_prefijo_chars": len(prefijo_sin_espacios),
            "e1_prefijo_ratio": len(prefijo_sin_espacios) / largo,
            "e1_prefijo_tokens": len(TOKEN_RE.findall(prefijo_sin_espacios)),
            "e1_prefijo_con_mencion": int(bool(MENTION_RE.search(prefijo_sin_espacios))),
            "e1_prefijo_con_url": int(bool(URL_RE.search(prefijo_sin_espacios))),
            "e1_prefijo_con_hashtag": int(bool(HASHTAG_RE.search(prefijo_sin_espacios))),
            "e1_prefijo_termina_puntuacion": int(bool(re.search(r"[^\w\s]$", prefijo_sin_espacios))),
            "e1_prefijo_inicia_simbolo": int(prefijo_sin_espacios[:1] in {"@", "#", "h"}),
            "e1_cuerpo_chars": len(cuerpo),
            "e1_cuerpo_tokens": len(TOKEN_RE.findall(cuerpo)),
            # E2 — interior del hashtag
            "e2_hashtag_n": len(etiquetas),
            "e2_hashtag_len_media": float(np.mean([len(etiqueta) - 1 for etiqueta in etiquetas]))
            if etiquetas else 0.0,
            "e2_hashtag_len_max": float(max([len(etiqueta) - 1 for etiqueta in etiquetas],
                                            default=0)),
            "e2_hashtag_camel_media": float(np.mean([len(re.findall(r"[a-z][A-Z]", etiqueta))
                                                     for etiqueta in etiquetas])) if etiquetas else 0.0,
            "e2_hashtag_digito_ratio": float(np.mean([sum(caracter.isdigit() for caracter in etiqueta)
                                                      / max(len(etiqueta), 1)
                                                      for etiqueta in etiquetas])) if etiquetas else 0.0,
            "e2_hashtag_entropia": entropia(etiquetas),
            # E4 — extremos del tweet
            "e4_izq_transiciones": len(re.findall(r"\W\w", extremo_izquierdo)),
            "e4_der_transiciones": len(re.findall(r"\w\W", extremo_derecho)),
            "e4_izq_digito": int(any(caracter.isdigit() for caracter in extremo_izquierdo)),
            "e4_der_digito": int(any(caracter.isdigit() for caracter in extremo_derecho)),
            "e4_izq_mayuscula": int(any(caracter.isupper() for caracter in extremo_izquierdo)),
            "e4_der_puntuacion": int(bool(re.search(r"[!?.,:;]$", texto.strip()))),
            "e4_der_puntos_suspensivos": int(texto.strip().endswith(("...", ".."))),
            "e4_izq_entidad": int(bool(re.match(r"[@#]", texto.strip()))),
        })
    marco = pd.DataFrame(filas)
    return marco.to_numpy(float), list(marco.columns)


def margenes_mencion(textos: pd.Series, y: np.ndarray, train_idx: np.ndarray,
                     alpha: float = 1.0) -> np.ndarray:
    """E3 — margen supervisado de las menciones por par adyacente (solo filas de entrenamiento).

    Para cada par adyacente ``k`` (``y <= k`` contra ``y > k``) se calcula el log-odds de cada
    mención y se promedia sobre las menciones del tweet. Sin menciones conocidas -> 0.
    """
    indices = np.array([cfg.CLASSES.index(valor) for valor in y])
    menciones = [set(MENTION_RE.findall(texto.lower()))
                 for texto in textos.fillna("").astype(str)]
    margenes = np.zeros((len(menciones), len(cfg.CLASSES) - 1))
    for borde in range(len(cfg.CLASSES) - 1):
        izquierda: dict[str, int] = {}
        derecha: dict[str, int] = {}
        for posicion in train_idx:
            for mencion in menciones[posicion]:
                if indices[posicion] <= borde:
                    izquierda[mencion] = izquierda.get(mencion, 0) + 1
                else:
                    derecha[mencion] = derecha.get(mencion, 0) + 1
        total_izquierda = sum(izquierda.values()) + alpha * len(izquierda)
        total_derecha = sum(derecha.values()) + alpha * len(derecha)
        log_odds = {mencion: np.log(((izquierda.get(mencion, 0) + alpha) / total_izquierda)
                                   / ((derecha.get(mencion, 0) + alpha) / total_derecha))
                    for mencion in set(izquierda) | set(derecha)}
        for posicion, conjunto in enumerate(menciones):
            if conjunto:
                margenes[posicion, borde] = float(np.mean([log_odds.get(mencion, 0.0)
                                                           for mencion in conjunto]))
    return margenes


def evaluar(features: np.ndarray, y: np.ndarray, folds: list[dict]) -> dict:
    """Macro-F1 LOFO del combinador de producción (promedio MLP + LogReg)."""
    puntajes = []
    for fold in folds:
        train_idx = np.concatenate([f["val_idx"] for f in folds if f["fold"] != fold["fold"]])
        valid_idx = fold["val_idx"]
        logreg = Pipeline([("scaler", StandardScaler()),
                           ("clf", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs",
                                                      random_state=cfg.SEED))])
        logreg.fit(features[train_idx], y[train_idx])
        mlp = EncodedMLP()
        mlp.fit(features[train_idx], y[train_idx])
        probabilidades = 0.5 * (logreg.predict_proba(features[valid_idx])
                                + mlp.model.predict_proba(mlp.scaler.transform(features[valid_idx])))
        predicciones = np.array(cfg.CLASSES)[np.argmax(probabilidades, axis=1)]
        puntajes.append(float(f1_score(y[valid_idx], predicciones, average="macro",
                                       labels=cfg.CLASSES)))
        print(f"    fold {fold['fold']}: {puntajes[-1]:.4f}", flush=True)
    return {"macro_f1": round(float(np.mean(puntajes)), 4), "std": round(float(np.std(puntajes)), 4),
            "per_fold": [round(valor, 4) for valor in puntajes]}


def main() -> int:
    """Evalúa los bloques estructurales solicitados."""
    set_seed()
    bloque = "libres"
    if "--bloque" in sys.argv:
        bloque = sys.argv[sys.argv.index("--bloque") + 1]
    started = time.time()
    payload: dict = {"fase": "E — estructura v2", "bloque": bloque, "fecha": "2026-09-25"}

    train = data.load_train()
    test = data.load_test()
    y = train[cfg.TARGET].astype(str).to_numpy()
    features, y_reconstruido, _, _, _ = build_features()
    features = np.concatenate([features, np.load(cfg.FEATURES / "oof_ADV07_nbsvm_word.npy")], axis=1)
    assert np.array_equal(y, y_reconstruido)
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")
    folds = cv.make_folds(y, groups=clusters["cluster"].to_numpy()[:len(y)], n_splits=5,
                          n_repeats=1, scheme="group")
    referencia = json.loads((cfg.EXPERIMENTS / "phase12_auditoria_estado.json")
                            .read_text(encoding="utf-8"))["reproduccion_lofo_v12"]["resultado"]
    payload["baseline"] = {"macro_f1": referencia["macro_f1"], "std": referencia["std"],
                           "per_fold": referencia["per_fold"], "origen": "fase 12"}

    if bloque in {"libres", "todos"}:
        bloque_libre, nombres = features_libres(train[cfg.TEXT_COL])
        arreglo = np.concatenate([features, bloque_libre], axis=1)
        print(f"[libres] {bloque_libre.shape} -> {arreglo.shape}", flush=True)
        resultado = evaluar(arreglo, y, folds)
        positivos = sum(1 for nuevo, viejo in zip(resultado["per_fold"], referencia["per_fold"])
                        if nuevo > viejo)
        payload["libres"] = {"n_features_nuevas": int(bloque_libre.shape[1]),
                             "nombres": nombres, **resultado,
                             "delta_vs_v12": round(resultado["macro_f1"]
                                                   - referencia["macro_f1"], 4),
                             "folds_positivos": positivos}

    if bloque in {"supervisado", "todos"}:
        # E3 requiere in-fold: se recalcula por fold y se combina con el mismo combinador.
        puntajes = []
        for fold in folds:
            train_idx = np.concatenate([f["val_idx"] for f in folds if f["fold"] != fold["fold"]])
            valid_idx = fold["val_idx"]
            margenes = margenes_mencion(train[cfg.TEXT_COL], y, train_idx)
            arreglo = np.concatenate([features, margenes], axis=1)
            logreg = Pipeline([("scaler", StandardScaler()),
                               ("clf", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs",
                                                          random_state=cfg.SEED))])
            logreg.fit(arreglo[train_idx], y[train_idx])
            mlp = EncodedMLP()
            mlp.fit(arreglo[train_idx], y[train_idx])
            probabilidades = 0.5 * (logreg.predict_proba(arreglo[valid_idx])
                                    + mlp.model.predict_proba(
                                        mlp.scaler.transform(arreglo[valid_idx])))
            predicciones = np.array(cfg.CLASSES)[np.argmax(probabilidades, axis=1)]
            puntajes.append(float(f1_score(y[valid_idx], predicciones, average="macro",
                                           labels=cfg.CLASSES)))
            print(f"    fold {fold['fold']}: {puntajes[-1]:.4f}", flush=True)
        resultado = {"macro_f1": round(float(np.mean(puntajes)), 4),
                     "std": round(float(np.std(puntajes)), 4),
                     "per_fold": [round(valor, 4) for valor in puntajes]}
        positivos = sum(1 for nuevo, viejo in zip(resultado["per_fold"], referencia["per_fold"])
                        if nuevo > viejo)
        payload["supervisado"] = {"n_features_nuevas": len(cfg.CLASSES) - 1, **resultado,
                                  "delta_vs_v12": round(resultado["macro_f1"]
                                                        - referencia["macro_f1"], 4),
                                  "folds_positivos": positivos}

    secciones = [clave for clave in ("libres", "supervisado") if clave in payload]
    for clave in secciones:
        payload[clave]["adoptado"] = bool(payload[clave]["delta_vs_v12"] >= 0.003)
    payload["segundos"] = round(time.time() - started, 1)
    RESULTS_PATH = cfg.EXPERIMENTS / f"phase17_estructura_v2_{bloque}.json"
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                            encoding="utf-8")
    print(json.dumps({clave: {k: valor for k, valor in payload[clave].items()
                              if k in ("macro_f1", "delta_vs_v12", "folds_positivos", "adoptado")}
                      for clave in secciones}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
