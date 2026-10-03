"""FASE 39 (P6) — Fusión de los bloques léxicos con V12 y control de permutación.

Compara, con el **mismo** combinador de producción (promedio LogReg + MLP sobre las 279 features de
V12) y los mismos folds agrupados, la referencia V12 contra cuatro candidatos:

* ``V12 + Registro`` (19 columnas)
* ``V12 + Vida`` (19 columnas)
* ``V12 + Temas`` (37 columnas)
* ``V12 + Registro + Vida + Temas`` (75 columnas). Se evalúa siempre, aunque §17 solo permitiría
  combinar si dos familias mostrasen evidencia independiente: medirlo es una ablación legítima y
  el veredicto se aplica después, con el resultado en la mano.

y, para cada bloque, su **control de permutación** (§DD-6 del plan): el mismo número de columnas
con las filas permutadas, es decir, con la señal destruida pero el tamaño del bloque intacto. Si un
bloque real no supera a su control permutado, su señal es nula aunque ambos mejoren: es el control
que protege contra el falso positivo de "más features = mejor generalización", demostrado en este
proyecto.

No se genera ninguna submission: un candidato que supere el gate queda como candidato y se decide
con revisión manual (§26).

Salida: ``experiments/phase39_fusion_v12.json``, ``reports/age_lexicon_results.csv``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from phase21_vista_estructural import evaluar  # noqa: E402  (combinador de produccion)
from phase2_ponderacion import EncodedMLP, build_features  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.config import set_seed  # noqa: E402

RESULTADO = cfg.EXPERIMENTS / "phase39_fusion_v12.json"
RESULTADOS_CSV = cfg.REPORTS / "age_lexicon_results.csv"
GATE_DELTA = 0.003
GATE_FOLDS = 4
#: Semillas del control de permutación. La varianza de inicialización del MLP al cambiar el ancho
#: del bloque es del orden de ±0,002 ( comparable al gate), así que una sola muestra de la nula
#: puede confundir señal con ruido: se usan varias y se compara con la media **y con el máximo**.
SEMILLAS_PERMUTACION = (1000, 1001, 1002)
#: Referencia canónica de producción (experiments/phase12_auditoria_estado.json).
V12_CANONICO = {"macro_f1": 0.6004, "per_fold": [0.6007, 0.605, 0.6032, 0.6004, 0.5928]}
ANCHO_BASE_ESPERADO = 279


def bloque(familia: str) -> tuple[np.ndarray, list[str]]:
    """Carga el bloque de una familia aplicando el cribado de redundancia del §12.

    Descarta las columnas marcadas como redundantes en ``reports/age_lexicon_redundancy.csv``
    (|rho| >= 0,8 contra las variables existentes o contra las 279 features de V12). En esta fase
    afecta a ``abreviaciones_internet``, que resulta ser un proxy del tamaño del tweet: sus
    conteos correlacionan 0,93-0,97 con la longitud.
    """
    matriz = np.load(cfg.FEATURES / f"lexicon_features_{familia}.npy").astype(np.float64)
    nombres = json.loads((cfg.FEATURES / f"lexicon_features_{familia}_nombres.json")
                        .read_text(encoding="utf-8"))
    redundancia = pd.read_csv(cfg.REPORTS / "age_lexicon_redundancy.csv")
    descartadas = set(redundancia.loc[redundancia["redundante"].astype(bool), "feature"])
    columnas = [i for i, nombre in enumerate(nombres) if nombre not in descartadas]
    print(f"  {familia}: {len(columnas)}/{len(nombres)} columnas tras cribado de redundancia "
          f"({len(nombres) - len(columnas)} descartadas: "
          f"{sorted({n.split('_lex_' + familia + '_')[1].rsplit('_', 1)[0] for n in descartadas if '_lex_' + familia + '_' in n})})",
          flush=True)
    return matriz[:, columnas], [nombres[i] for i in columnas]


def permuta(matriz: np.ndarray, semilla: int) -> np.ndarray:
    """Permuta las filas de un bloque: mismo tamaño, señal destruida (control DD-6)."""
    generador = np.random.default_rng(semilla)
    return matriz[generador.permutation(len(matriz))]


def evaluar_cabeza(features: np.ndarray, y: np.ndarray, folds: list[dict],
                   cabeza: str) -> dict:
    """Macro-F1 LOFO de una sola cabeza (LogReg o MLP), con el escalado de producción.

    Es la segunda mitad de DD-8 (el efecto no debe depender del tipo de cabeza) y solo se ejecuta
    si algún bloque supera el gate numérico.
    """
    puntajes = []
    for fold in folds:
        entrenamiento = np.concatenate([f["val_idx"] for f in folds
                                       if f["fold"] != fold["fold"]])
        valid_idx = fold["val_idx"]
        if cabeza == "logreg":
            modelo = Pipeline([("scaler", StandardScaler()),
                               ("clf", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs",
                                                          random_state=cfg.SEED))])
            modelo.fit(features[entrenamiento], y[entrenamiento])
            probabilidades = modelo.predict_proba(features[valid_idx])
        else:
            modelo = EncodedMLP()
            modelo.fit(features[entrenamiento], y[entrenamiento])
            probabilidades = modelo.model.predict_proba(
                modelo.scaler.transform(features[valid_idx]))
        predicciones = np.array(cfg.CLASSES)[np.argmax(probabilidades, axis=1)]
        puntajes.append(float(f1_score(y[valid_idx], predicciones, average="macro",
                                       labels=cfg.CLASSES)))
    return {"macro_f1": round(float(np.mean(puntajes)), 4), "std": round(float(np.std(puntajes)), 4),
            "per_fold": [round(valor, 4) for valor in puntajes]}


def evaluar_guardando(features: np.ndarray, y: np.ndarray, folds: list[dict]) -> dict:
    """Como :func:`evaluar`, pero devuelve además el argmax OOF de cada fila.

    El argmax OOF es el sustrato del análisis de complementariedad y de error de P7 (§19-§21): sin
    las predicciones por fila no se puede medir rescate, daño ni errores por frontera.

    El combinador se reimplementa aquí **a propósito** para poder acumular predicciones, y se
    verifica contra la función de producción: quien llama compara ambos macro-F1 y exige igualdad.
    """
    predicciones = np.empty(len(y), dtype=object)
    por_fold = []
    for fold in folds:
        entrenamiento = np.concatenate([f["val_idx"] for f in folds
                                       if f["fold"] != fold["fold"]])
        valid_idx = fold["val_idx"]
        logreg = Pipeline([("scaler", StandardScaler()),
                           ("clf", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs",
                                                      random_state=cfg.SEED))])
        logreg.fit(features[entrenamiento], y[entrenamiento])
        mlp = EncodedMLP()
        mlp.fit(features[entrenamiento], y[entrenamiento])
        probabilidades = 0.5 * (logreg.predict_proba(features[valid_idx])
                                + mlp.model.predict_proba(
                                    mlp.scaler.transform(features[valid_idx])))
        etiquetas = np.array(cfg.CLASSES)[np.argmax(probabilidades, axis=1)]
        predicciones[valid_idx] = etiquetas
        por_fold.append(float(f1_score(y[valid_idx], etiquetas, average="macro",
                                       labels=cfg.CLASSES)))
    return {"macro_f1": round(float(np.mean(por_fold)), 4), "std": round(float(np.std(por_fold)), 4),
            "per_fold": [round(valor, 4) for valor in por_fold],
            "oof_argmax": predicciones}


def main() -> int:
    """Evalúa la fusión de cada familia con V12 frente a su control de permutación."""
    set_seed()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    base, _, _, _, _ = build_features()
    base = np.concatenate([base, np.load(cfg.FEATURES / "oof_ADV07_nbsvm_word.npy")], axis=1)
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")
    assert base.shape[1] == ANCHO_BASE_ESPERADO, (
        f"la matriz base debe tener {ANCHO_BASE_ESPERADO} columnas, tiene {base.shape[1]}")
    print(f"matriz base V12: {base.shape}", flush=True)

    bloques = {familia: bloque(familia) for familia in ("registro", "vida", "temas")}
    for familia, (matriz, _) in bloques.items():
        assert matriz.shape[0] == len(y), f"{familia}: {matriz.shape[0]} filas != {len(y)} etiquetas"
    referencia = evaluar(base, y, folds)
    assert abs(referencia["macro_f1"] - V12_CANONICO["macro_f1"]) < 1e-3, (
        f"la referencia no reproduce V12: {referencia['macro_f1']} vs {V12_CANONICO['macro_f1']}")
    print(f"[V12] {referencia['macro_f1']} ± {referencia['std']} "
          f"per_fold={referencia['per_fold']}", flush=True)
    con_oof = evaluar_guardando(base, y, folds)
    assert con_oof["macro_f1"] == referencia["macro_f1"], (
        f"la version con OOF no reproduce el combinador de produccion: "
        f"{con_oof['macro_f1']} vs {referencia['macro_f1']}")
    oof_v12 = con_oof["oof_argmax"]
    np.save(cfg.FEATURES / "oof_argmax_v12.npy",
            np.array([cfg.CLASSES.index(v) for v in oof_v12], dtype=np.int8))
    print("  argmax OOF de V12 guardado y verificado contra el combinador de produccion", flush=True)

    candidatos = [("V12 + Registro", bloques["registro"][0]),
                  ("V12 + Vida", bloques["vida"][0]),
                  ("V12 + Temas", bloques["temas"][0]),
                  ("V12 + Registro + Vida + Temas", np.concatenate(
                      [bloques["registro"][0], bloques["vida"][0], bloques["temas"][0]], axis=1))]
    filas = [{"candidato": "V12 (referencia)", "tipo": "referencia",
              "n_columnas_anadidas": 0, "macro_f1": referencia["macro_f1"], "std": referencia["std"],
              "delta": 0.0, "folds_positivos": 0, "per_fold": referencia["per_fold"],
              "supera_gate_numerico": False, "supera_control_permutado": False, "adopta": False}]
    detalle = {}
    for nombre, matriz in candidatos:
        resultado = evaluar(np.concatenate([base, matriz], axis=1), y, folds)
        con_oof_candidato = evaluar_guardando(np.concatenate([base, matriz], axis=1), y, folds)
        assert con_oof_candidato["macro_f1"] == resultado["macro_f1"], (
            f"{nombre}: la version con OOF no reproduce la metrica reportada")
        etiqueta = ("".join(caracter if caracter.isalnum() else "_" for caracter in nombre)
                    .strip("_").lower())
        np.save(cfg.FEATURES / f"oof_argmax_{etiqueta}.npy",
                np.array([cfg.CLASSES.index(v) for v in con_oof_candidato["oof_argmax"]],
                         dtype=np.int8))
        positivos = int(sum(1 for nuevo, viejo in zip(resultado["per_fold"], referencia["per_fold"])
                            if nuevo > viejo))
        delta = round(resultado["macro_f1"] - referencia["macro_f1"], 4)
        gate_numerico = bool(delta >= GATE_DELTA and positivos >= GATE_FOLDS)
        # Control de permutación: siempre una muestra; si el bloque pasa el gate numérico se toman
        # las tres, porque un solo sorteo no distingue señal de ruido a este orden de magnitud.
        semillas = SEMILLAS_PERMUTACION if gate_numerico else SEMILLAS_PERMUTACION[:1]
        controles = [evaluar(np.concatenate([base, permuta(matriz, semilla)], axis=1), y, folds)
                     for semilla in semillas]
        mejor_control = max(c["macro_f1"] for c in controles)
        media_control = float(np.mean([c["macro_f1"] for c in controles]))
        # DD-6: la señal solo es real si el bloque supera a su control. Se compara con el máximo
        # de la nula (comparación conservadora).
        supera_control = bool(resultado["macro_f1"] > mejor_control)
        filas.append({"candidato": nombre, "tipo": "bloque real",
                      "n_columnas_anadidas": int(matriz.shape[1]),
                      "macro_f1": resultado["macro_f1"], "std": resultado["std"], "delta": delta,
                      "folds_positivos": positivos, "per_fold": resultado["per_fold"],
                      "supera_gate_numerico": gate_numerico, "supera_control_permutado": supera_control,
                      "adopta": bool(gate_numerico and supera_control)})
        for indice_control, control in enumerate(controles):
            filas.append({"candidato": f"CONTROL permutado[{indice_control}] de {nombre}",
                          "tipo": "permutado", "n_columnas_anadidas": int(matriz.shape[1]),
                          "macro_f1": control["macro_f1"], "std": control["std"],
                          "delta": round(control["macro_f1"] - referencia["macro_f1"], 4),
                          "folds_positivos": int(sum(
                              1 for nuevo, viejo in zip(control["per_fold"], referencia["per_fold"])
                              if nuevo > viejo)),
                          "per_fold": control["per_fold"],
                          "supera_gate_numerico": False, "supera_control_permutado": False,
                          "adopta": False})
        detalle[nombre] = {
            "real": resultado, "controles_permutados": controles, "delta": delta,
            "folds_positivos": positivos, "gate_numerico": gate_numerico,
            "media_control": round(media_control, 4), "mejor_control": mejor_control,
            "ventaja_sobre_control_medio": round(resultado["macro_f1"] - media_control, 4),
            "ventaja_sobre_control_maximo": round(resultado["macro_f1"] - mejor_control, 4),
            "supera_control": supera_control,
        }
        print(f"[{nombre}] {resultado['macro_f1']} (Δ {delta:+.4f}, {positivos}/5) | "
              f"control permutado: media {media_control:.4f} / max {mejor_control:.4f} | "
              f"ventaja {detalle[nombre]['ventaja_sobre_control_maximo']:+.4f} | "
              f"gate={gate_numerico} control={supera_control}", flush=True)
        if gate_numerico:
            for cabeza in ("logreg", "mlp"):
                detalle[nombre][f"cabeza_{cabeza}"] = evaluar_cabeza(
                    np.concatenate([base, matriz], axis=1), y, folds, cabeza)
            detalle[nombre]["cabeza_logreg_referencia"] = evaluar_cabeza(base, y, folds, "logreg")
            detalle[nombre]["cabeza_mlp_referencia"] = evaluar_cabeza(base, y, folds, "mlp")
            print(f"  DD-8: logreg {detalle[nombre]['cabeza_logreg']['macro_f1']} "
                  f"(ref {detalle[nombre]['cabeza_logreg_referencia']['macro_f1']}) | "
                  f"mlp {detalle[nombre]['cabeza_mlp']['macro_f1']} "
                  f"(ref {detalle[nombre]['cabeza_mlp_referencia']['macro_f1']})", flush=True)

    tabla = pd.DataFrame(filas)
    tabla.to_csv(RESULTADOS_CSV, index=False)
    adoptar = [fila["candidato"] for fila in filas if fila["adopta"]]
    signaled = [nombre for nombre, medida in detalle.items() if medida["supera_control"]]
    payload = {
        "fase": "P6 — fusion con V12 y control de permutacion",
        "fecha": "2026-09-25",
        "gate": f"delta >= +{GATE_DELTA} y >= {GATE_FOLDS}/5 folds positivos",
        "referencia_v12": referencia,
        "n_columnas_base": int(base.shape[1]),
        "candidatos": detalle,
        "tabla": filas,
        "adoptan": adoptar,
        "familias_con_señal_sobre_control": signaled,
        "semillas_permutacion": list(SEMILLAS_PERMUTACION),
        "criterio_adopcion": "gate numerico Y superar al control de permutacion (DD-6)",
        "hipotesis_independientes_para_combinacion": len(signaled) >= 2,
        "segundos": round(time.time() - inicio, 1),
    }
    RESULTADO.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nadoptan: {adoptar if adoptar else 'ninguno'}")
    print(f"familias con señal sobre su control: {signaled}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
