"""V13 · P1b — Fusion del bloque de evidencia por token con V12, con control de permutado y gate.

P1a demostro que ``P(clase | token)`` lleva senal propia: el control *block-only* da 0,3997 con
LogReg y 0,3917 con LinearSVC frente a un suelo de azar de 0,1667, el doble que el mejor bloque
lexico de la fase cerrada. Aqui se responde a la pregunta que de verdad decide: **¿eso se traduce
en una mejora de V12 por encima de su control?**

Protocolo:

* La referencia es el OOF de produccion ya calculado (``features/oof_ensemble_v12.npy``, el mismo
  combinador 0,5 LogReg + 0,5 MLP sobre las 279 variables, mismos folds y misma semilla). Reutilizar
  el OOF de produccion en vez de reentrenarlo hace que la comparacion sea **pareada fila a fila** y
  elimina una fuente de ruido, y su macro-F1 por fold es el 0,6007/0,6050/0,6032/0,6004/0,5928 que
  ya consta en ``experiments/phase39_fusion_v12.json``.
* El bloque de P1 entra **por fold** (cada fold exterior tiene su matriz, con las filas de train
  calculadas por ajuste cruzado y las de validacion con el perfil del train exterior completo).
* **Control de permutado (DD-6)**: se permutan las filas del bloque de P1 con tres semillas y se
  repite la fusion. Si la mejora real no supera al control permutado, la mejora viene de la forma de
  la matriz y no de su contenido, y no se adopta.
* **Gate (§24)**: diferencia >= +0,003, al menos 4 de 5 folds positivos, intervalo de confianza al
  95 % por bootstrap multinomial que excluya el cero, y una mejora que no se explique por una sola
  fold.
* **Doble metrica (§36)**: macro-F1 global y accuracy y F1 de las dos fronteras centrales
  ``18-24|25-34`` y ``25-34|35-49``, incluidos los cruces. Un bloque que mejora la global
  empeorando una frontera no se adopta.
* **Por cabeza (DD-8)**: LogReg sola y MLP sola, cada una contra su propia referencia, para ver si
  la senal entra por una cabeza y se diluye en la otra.
* **Complementariedad (§26)**: rescate, dano y neto, en global y en cada frontera.

El test no se toca en ninguna parte. Ninguna submission se genera aqui.

Entregables: ``experiments/v13_p1b_fusion_tokens.json``, ``reports/v13_p1_fusion.md``,
``reports/v13_p1_fusion_comparativa.csv``, ``features/v13_p1b_oof.npy``.
"""

from __future__ import annotations

import json
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
sys.path.insert(0, str(ROOT / "scripts"))

from phase2_ponderacion import EncodedMLP, build_features  # noqa: E402

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.config import set_seed  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

INFORME = cfg.REPORTS / "v13_p1_fusion.md"
COMPARATIVA = cfg.REPORTS / "v13_p1_fusion_comparativa.csv"
OOF = cfg.FEATURES / "v13_p1b_oof.npy"
RESULTADO = cfg.EXPERIMENTS / "v13_p1b_fusion_tokens.json"
REFERENCIA = cfg.EXPERIMENTS / "phase39_fusion_v12.json"

#: Gate de adopcion (§24).
DELTA_MINIMO = 0.003
FOLDS_POSITIVOS_MINIMOS = 4
#: Semillas del control de permutado (DD-6).
SEMILLAS = (1000, 1001, 1002)
#: Resmuestros del bootstrap multinomial del intervalo de confianza.
RESMUESTROS = 4000
#: Pares de frontera que exige la doble metrica (§36).
PARES = {("18-24", "25-34"): "18-24|25-34", ("25-34", "35-49"): "25-34|35-49"}


def logreg() -> Pipeline:
    """LogReg con tipificador, la cabeza lineal del combinador de produccion."""
    return Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs",
                                                random_state=cfg.SEED))])


def mlp() -> EncodedMLP:
    """La otra cabeza del combinador de produccion, con los mismos parametros que V12.

    No se puede usar un ``MLPClassifier`` pelado: ``EncodedMLP`` es el que escala, codifica las
    etiquetas y fija los hiperparametros (una capa de 64, alpha 1e-3, batch 512, early stopping),
    y cambiar cualquiera de ellos haria la comparacion con la referencia invalida.
    """
    return EncodedMLP()


def oof_combinador(matriz: np.ndarray, y: np.ndarray, folds: list[dict]) -> np.ndarray:
    """OOF del combinador de produccion (0,5 LogReg + 0,5 MLP) sobre una matriz de variables."""
    probabilidades = np.zeros((len(y), len(cfg.CLASSES)))
    for fold in folds:
        validacion = fold["val_idx"]
        entrenamiento = np.concatenate([otro["val_idx"] for otro in folds
                                        if otro["fold"] != fold["fold"]])
        lineal = logreg().fit(matriz[entrenamiento], y[entrenamiento])
        red = mlp().fit(matriz[entrenamiento], y[entrenamiento])
        probabilidades[validacion] = 0.5 * lineal.predict_proba(matriz[validacion]) + \
            0.5 * red.model.predict_proba(red.scaler.transform(matriz[validacion]))
        etiquetas = np.array(cfg.CLASSES)[np.argmax(probabilidades[validacion], axis=1)]
        print(f"    fold {fold['fold']}: "
              f"{f1_score(y[validacion], etiquetas, average='macro', labels=cfg.CLASSES):.4f}",
              flush=True)
    return probabilidades


def oof_una_cabeza(matriz: np.ndarray, y: np.ndarray, folds: list[dict],
                   cabeza: str) -> np.ndarray:
    """OOF de una sola cabeza (LogReg o MLP) sobre una matriz de variables."""
    probabilidades = np.zeros((len(y), len(cfg.CLASSES)))
    for fold in folds:
        validacion = fold["val_idx"]
        entrenamiento = np.concatenate([otro["val_idx"] for otro in folds
                                        if otro["fold"] != fold["fold"]])
        if cabeza == "logreg":
            estimador = logreg().fit(matriz[entrenamiento], y[entrenamiento])
            probabilidades[validacion] = estimador.predict_proba(matriz[validacion])
        else:
            estimador = mlp().fit(matriz[entrenamiento], y[entrenamiento])
            probabilidades[validacion] = estimador.model.predict_proba(
                estimador.scaler.transform(matriz[validacion]))
    return probabilidades


def metricas_globales(probabilidades: np.ndarray, clase: np.ndarray) -> dict:
    """Macro-F1, accuracy y F1 por clase de unas probabilidades OOF."""
    predicciones = np.argmax(probabilidades, axis=1)
    return {
        "macro_f1": round(float(f1_score(clase, predicciones, average="macro",
                                         labels=list(range(len(cfg.CLASSES))))), 4),
        "accuracy": round(float((predicciones == clase).mean()), 4),
        "f1_por_clase": {cfg.CLASSES[indice]: round(float(f1_score(
            clase, predicciones, average="macro", labels=[indice], zero_division=0)), 4)
            for indice in range(len(cfg.CLASSES))},
    }


def metricas_frontera(probabilidades: np.ndarray, clase: np.ndarray,
                      pares: dict[tuple[str, str], str]) -> dict:
    """Accuracy, macro-F1 y cruces resueltos en cada par de frontera (§36)."""
    predicciones = np.argmax(probabilidades, axis=1)
    salida = {}
    for (clase_a, clase_b), etiqueta in pares.items():
        indice_a, indice_b = cfg.CLASSES.index(clase_a), cfg.CLASSES.index(clase_b)
        en_par = np.isin(clase, (indice_a, indice_b))
        cruce = en_par & (((clase == indice_a) & (predicciones == indice_b))
                          | ((clase == indice_b) & (predicciones == indice_a)))
        aciertos = predicciones[en_par] == clase[en_par]
        salida[etiqueta] = {
            "filas": int(en_par.sum()),
            "accuracy": round(float(aciertos.mean()), 4),
            "macro_f1": round(float(f1_score(clase[en_par], predicciones[en_par],
                                              average="macro",
                                              labels=[indice_a, indice_b])), 4),
            "cruces": int(cruce.sum()),
            "cruces_resueltos": int((predicciones[cruce] == clase[cruce]).sum()),
            "pct_cruces_resueltos": round(
                100 * float((predicciones[cruce] == clase[cruce]).mean()), 3) if cruce.any() else 0.0,
        }
    return salida


def bootstrap_diferencia(referencia: np.ndarray, candidata: np.ndarray, clase: np.ndarray,
                         n_clases: int) -> tuple[float, float, float]:
    """Intervalo de confianza al 95 % de la diferencia de macro-F1 por bootstrap **pareado**.

    Remuestrear filas con reemplazo equivale a multiplicar la tabla conjunta por unos conteos
    multinomiales, porque el macro-F1 de cada modelo solo depende de los conteos por celda. La
    tabla que hay que remuestrar es la **conjunta** de (clase verdadera, prediccion de la
    referencia, prediccion de la candidata): asi los dos modelos se evaluan siempre sobre el
    **mismo** remuestreo de filas y la diferencia es pareada. Remuestrear por separado las dos
    tablas de confusion daria un intervalo sin sentido, y con predicciones identicas daria un
    intervalo que no contiene el cero.

    Devuelve
    -------
    tuple
        ``(percentil 2,5, percentil 97,5, proporcion de remuestrios con diferencia positiva)``.
    """
    predicciones_referencia = np.argmax(referencia, axis=1)
    predicciones_candidata = np.argmax(candidata, axis=1)
    lado = n_clases * n_clases
    celdas = np.zeros(n_clases * lado, dtype=np.int64)
    np.add.at(celdas, clase * lado + predicciones_referencia * n_clases + predicciones_candidata, 1)
    total = int(celdas.sum())
    generador = np.random.default_rng(cfg.SEED)
    # Ejes del tensor remuestreado: (remuestrio, clase verdadera, prediccion de la referencia,
    # prediccion de la candidata).
    muestreos = generador.multinomial(total, celdas / total,
                                      size=RESMUESTROS).reshape(RESMUESTROS, n_clases, n_clases,
                                                                n_clases)
    confusion_referencia = muestreos.sum(axis=3)   # se eliminan las predicciones de la candidata
    confusion_candidata = muestreos.sum(axis=2)    # se eliminan las de la referencia
    macro_por_modelo = [macro_f1_de_confusion(confusion_referencia),
                        macro_f1_de_confusion(confusion_candidata)]
    diferencias = macro_por_modelo[1] - macro_por_modelo[0]
    return (float(np.percentile(diferencias, 2.5)), float(np.percentile(diferencias, 97.5)),
            float((diferencias > 0).mean()))


def macro_f1_de_confusion(confusion: np.ndarray) -> np.ndarray:
    """Macro-F1 de cada matriz de confusion de un remuestreo.

    Parametros
    ----------
    confusion:
        Tensor de forma ``(remuestrios, n_clases, n_clases)`` con filas = clase verdadera y
        columnas = clase predicha.
    """
    verdadero = confusion.sum(axis=2)
    predicho = confusion.sum(axis=1)
    aciertos = np.diagonal(confusion, axis1=1, axis2=2)
    precision = aciertos / np.clip(predicho, 1, None)
    exhaustividad = aciertos / np.clip(verdadero, 1, None)
    f1 = 2 * precision * exhaustividad / np.clip(precision + exhaustividad, 1e-12, None)
    return f1.mean(axis=1)


def complementariedad(referencia: np.ndarray, candidata: np.ndarray, clase: np.ndarray) -> dict:
    """Rescate, dano y neto de la candidata sobre la referencia, en global y por frontera (§26)."""
    acierto_base = np.argmax(referencia, axis=1) == clase
    acierto_nueva = np.argmax(candidata, axis=1) == clase
    salida = {}
    conjuntos = {"global": np.ones(len(clase), dtype=bool)}
    for (clase_a, clase_b), etiqueta in PARES.items():
        conjuntos[etiqueta] = np.isin(clase, (cfg.CLASSES.index(clase_a),
                                              cfg.CLASSES.index(clase_b)))
    for nombre, mascara in conjuntos.items():
        rescate = int((~acierto_base & acierto_nueva & mascara).sum())
        dano = int((acierto_base & ~acierto_nueva & mascara).sum())
        salida[nombre] = {"rescate": rescate, "dano": dano, "neto": rescate - dano}
    return salida


def main() -> int:
    """Fusiona el bloque de P1 con V12 y decide la adopcion segun el gate."""
    autotest_numerico()
    set_seed()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(valor) for valor in y])
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")

    base, _, _, _, _ = build_features()
    base = np.concatenate([base, np.load(cfg.FEATURES / "oof_ADV07_nbsvm_word.npy")], axis=1)
    referencia = np.load(cfg.FEATURES / "oof_ensemble_v12.npy")
    metricas_referencia = metricas_globales(referencia, clase)
    fronteras_referencia = metricas_frontera(referencia, clase, PARES)
    por_fold_referencia = [round(float(f1_score(
        clase[fold["val_idx"]], np.argmax(referencia[fold["val_idx"]], axis=1),
        average="macro", labels=list(range(len(cfg.CLASSES))))), 4) for fold in folds]
    print(f"[referencia V12] macro-F1 {metricas_referencia['macro_f1']} | folds "
          f"{por_fold_referencia}", flush=True)

    nombres_p1 = json.loads((cfg.FEATURES / "v13_p1_tokens_nombres.json").read_text(encoding="utf-8"))
    bloques = {fold["fold"]: np.load(cfg.FEATURES / f"v13_p1_tokens_fold{fold['fold']}.npy")
               for fold in folds}

    def fusionar(bloques_fold: dict[int, np.ndarray]) -> np.ndarray:
        """OOF del combinador sobre V12 mas el bloque, usando la matriz de cada fold."""
        probabilidades = np.zeros((len(clase), len(cfg.CLASSES)))
        for fold in folds:
            validacion = fold["val_idx"]
            entrenamiento = np.concatenate([otro["val_idx"] for otro in folds
                                            if otro["fold"] != fold["fold"]])
            matriz = np.hstack([base, bloques_fold[fold["fold"]].astype(np.float64)])
            lineal = logreg().fit(matriz[entrenamiento], y[entrenamiento])
            red = mlp().fit(matriz[entrenamiento], y[entrenamiento])
            probabilidades[validacion] = 0.5 * lineal.predict_proba(matriz[validacion]) + \
                0.5 * red.model.predict_proba(red.scaler.transform(matriz[validacion]))
            etiquetas = np.array(cfg.CLASSES)[np.argmax(probabilidades[validacion], axis=1)]
        print(f"    fold {fold['fold']}: "
              f"{f1_score(y[validacion], etiquetas, average='macro', labels=cfg.CLASSES):.4f}",
              flush=True)
        return probabilidades

    print("[candidata] V12 + evidencia por token", flush=True)
    candidata = fusionar(bloques)
    metricas_candidata = metricas_globales(candidata, clase)
    fronteras_candidata = metricas_frontera(candidata, clase, PARES)
    por_fold_candidata = [round(float(f1_score(
        clase[fold["val_idx"]], np.argmax(candidata[fold["val_idx"]], axis=1),
        average="macro", labels=list(range(len(cfg.CLASSES))))), 4) for fold in folds]
    delta = metricas_candidata["macro_f1"] - metricas_referencia["macro_f1"]
    folds_positivos = sum(1 for a, b in zip(por_fold_candidata, por_fold_referencia) if b > a)
    print(f"[candidata] macro-F1 {metricas_candidata['macro_f1']} | delta {delta:+.4f} | "
          f"folds positivos {folds_positivos}/5 | folds {por_fold_candidata}", flush=True)
    np.save(OOF, candidata)

    # ---------------------------------------------------------------- control de permutado
    controles = []
    for semilla in SEMILLAS:
        # Cada fold recibe su propia permutacion (semilla + numero de fold) para que el control
        # no dependa de una unica ordenacion de filas.
        permutada = {numero: matriz[np.random.default_rng(semilla + numero).permutation(len(y))]
                     for numero, matriz in bloques.items()}
        probabilidades = fusionar(permutada)
        valor = metricas_globales(probabilidades, clase)["macro_f1"]
        controles.append({"semilla": semilla, "macro_f1": valor,
                          "delta": round(valor - metricas_referencia["macro_f1"], 4)})
        print(f"[control permutado] semilla {semilla}: {valor} (delta "
              f"{valor - metricas_referencia['macro_f1']:+.4f})", flush=True)
    mejor_control = max(controles, key=lambda control: control["macro_f1"])

    # ---------------------------------------------------------------- gate
    mejoras_por_fold = [nueva - vieja for nueva, vieja in zip(por_fold_candidata,
                                                             por_fold_referencia)]
    intervalo = bootstrap_diferencia(referencia, candidata, clase, len(cfg.CLASSES))
    print(f"[bootstrap] diferencia al 95 %: [{intervalo[0]:+.4f}, {intervalo[1]:+.4f}] | "
          f"proporcion de resmuestros positivos {intervalo[2]:.3f}", flush=True)
    complemento = complementariedad(referencia, candidata, clase)
    improved_frontiers = sum(
        1 for etiqueta in PARES.values()
        if fronteras_candidata[etiqueta]["accuracy"] > fronteras_referencia[etiqueta]["accuracy"])
    criterios = {
        "delta_minimo": bool(delta >= DELTA_MINIMO),
        "folds_positivos": bool(folds_positivos >= FOLDS_POSITIVOS_MINIMOS),
        "bootstrap_excluye_cero": bool(intervalo[0] > 0),
        "supera_control_permutado": bool(metricas_candidata["macro_f1"] > mejor_control["macro_f1"]),
        "no_explicada_por_una_fold": bool(max(mejoras_por_fold) <= sum(
           mejoras_por_fold) - max(mejoras_por_fold) + 1e-9),
        "fronteras_no_empeoran": bool(improved_frontiers >= len(PARES)),
    }
    adopta = all(criterios.values())

    # ---------------------------------------------------------------- por cabeza (DD-8)
    cabezas = {}
    for cabeza in ("logreg", "mlp"):
        base_cabeza = oof_una_cabeza(base, y, folds, cabeza)
        nueva_cabeza = oof_una_cabeza(np.hstack([base, bloques[0].astype(np.float64)]),
                                     y, folds, cabeza)
        valor_base = metricas_globales(base_cabeza, clase)["macro_f1"]
        valor_nueva = metricas_globales(nueva_cabeza, clase)["macro_f1"]
        cabezas[cabeza] = {"macro_f1_referencia": valor_base, "macro_f1_candidata": valor_nueva,
                           "delta": round(valor_nueva - valor_base, 4)}
        print(f"[cabeza {cabeza}] {valor_base} -> {valor_nueva} "
              f"({valor_nueva - valor_base:+.4f})", flush=True)

    # ---------------------------------------------------------------- tablas e informe
    comparativa = []
    for nombre, metricas, fronteras in (("V12 (referencia)", metricas_referencia, fronteras_referencia),
                                        ("V12 + P1 tokens", metricas_candidata, fronteras_candidata)):
        fila = {"modelo": nombre, "macro_f1": metricas["macro_f1"],
                "accuracy": metricas["accuracy"]}
        for etiqueta in PARES.values():
            fila[f"{etiqueta}_accuracy"] = fronteras[etiqueta]["accuracy"]
            fila[f"{etiqueta}_cruces_resueltos_pct"] = fronteras[etiqueta]["pct_cruces_resueltos"]
        comparativa.append(fila)
    for control in controles:
        comparativa.append({"modelo": f"control permutado (semilla {control['semilla']})",
                            "macro_f1": control["macro_f1"], "accuracy": None})
    pd.DataFrame(comparativa).to_csv(COMPARATIVA, index=False)

    lineas = [
        "# V13 · P1b — Fusion de la evidencia por token con V12", "",
        "## 1. Resultado global", "",
        "| Modelo | macro-F1 | accuracy |", "|---|---|---|",
        f"| V12 (referencia) | {metricas_referencia['macro_f1']} | "
        f"{metricas_referencia['accuracy']} |",
        f"| V12 + evidencia por token | {metricas_candidata['macro_f1']} | "
        f"{metricas_candidata['accuracy']} |",
    ]
    for control in controles:
        lineas.append(f"| Control permutado (semilla {control['semilla']}) | "
                      f"{control['macro_f1']} | - |")
    lineas += ["", f"**Diferencia: {delta:+.4f}** macro-F1 · folds "
                  f"{por_fold_candidata} frente a {por_fold_referencia} · "
                  f"**{folds_positivos} de 5 folds positivos**.", "",
              "## 2. Control de permutado (DD-6)", "",
              f"El mejor control permutado llega a {mejor_control['macro_f1']} "
              f"(delta {mejor_control['delta']:+.4f}). La cifra real "
              f"({metricas_candidata['macro_f1']}, delta {delta:+.4f}) "
              + ("**supera** al control." if metricas_candidata["macro_f1"] > mejor_control["macro_f1"]
                 else "**no supera** al control: la mejora no viene del contenido del bloque."), "",
              "## 3. Doble metrica (§36)", "",
              "| Frontera | Filas | accuracy referencia | accuracy candidata | Cruces | Cruces "
              "resueltos referencia | Cruces resueltos candidata |", "|---|---|---|---|---|---|---|"]
    for etiqueta in PARES.values():
        antes, ahora = fronteras_referencia[etiqueta], fronteras_candidata[etiqueta]
        lineas.append(f"| {etiqueta} | {antes['filas']:,} | {antes['accuracy']} | "
                      f"{ahora['accuracy']} | {antes['cruces']:,} | "
                      f"{antes['pct_cruces_resueltos']:.2f} % | "
                      f"{ahora['pct_cruces_resueltos']:.2f} % |")
    lineas += ["", "## 4. Bootstrap y complementariedad", "",
              f"- Diferencia con intervalo al 95 %: **[{intervalo[0]:+.4f}, {intervalo[1]:+.4f}]**, "
              f"proporcion de resmuestros positivos {intervalo[2]:.3f} "
              f"({RESMUESTROS} resmuestros multinomiales).", ""]
    for ambito, medidas in complemento.items():
        lineas.append(f"- {ambito}: rescate {medidas['rescate']:,} · dano {medidas['dano']:,} · "
                      f"**neto {medidas['neto']:+,}**")
    lineas += ["", "## 5. Por cabeza (DD-8)", "",
               "| Cabeza | Referencia | Candidata | Diferencia |", "|---|---|---|---|"]
    for cabeza, medidas in cabezas.items():
        lineas.append(f"| {cabeza} | {medidas['macro_f1_referencia']} | "
                      f"{medidas['macro_f1_candidata']} | {medidas['delta']:+.4f} |")
    lineas += ["", "## 6. Gate de adopcion (§24)", "",
               "| Criterio | Valor | Cumple |", "|---|---|---|"]
    etiquetas_criterio = {
        "delta_minimo": f"delta {delta:+.4f} >= {DELTA_MINIMO}",
        "folds_positivos": f"{folds_positivos}/5 >= {FOLDS_POSITIVOS_MINIMOS}",
        "bootstrap_excluye_cero": f"IC95 [{intervalo[0]:+.4f}, {intervalo[1]:+.4f}]",
        "supera_control_permutado": f"{metricas_candidata['macro_f1']} > "
                                     f"{mejor_control['macro_f1']}",
        "no_explicada_por_una_fold": f"la mejor fold aporta {max(mejoras_por_fold):+.4f} de {sum(mejoras_por_fold):+.4f} de mejora total",
        "fronteras_no_empeoran": f"{improved_frontiers}/{len(PARES)} fronteras mejoran",
    }
    for nombre, cumple in criterios.items():
        lineas.append(f"| {nombre.replace('_', ' ')} | {etiquetas_criterio[nombre]} | "
                      f"{'si' if cumple else '**no**'} |")
    lineas += ["", f"**Decision: {'ADOPTAR' if adopta else 'NO ADOPTAR'}**", ""]
    if adopta:
        lineas.append("El bloque supera el gate completo. Queda como candidato; la adopcion final "
                      "exige ademas una segunda particion o semilla (§25) antes de generar "
                      "submission.")
    else:
        fallidos = [nombre.replace("_", " ") for nombre, cumple in criterios.items() if not cumple]
        lineas.append(f"El bloque **no se adopta**: falla {', '.join(fallidos)}. La senal de P1a "
                      f"existe en el bloque aislado pero no se traduce en macro-F1 sobre V12.")
    lineas.append("")
    INFORME.write_text("\n".join(lineas) + "\n", encoding="utf-8")

    RESULTADO.write_text(json.dumps({
        "fase": "V13 · P1b fusion con V12 y control de permutado", "fecha": "2026-09-25",
        "n_variables_nuevas": len(nombres_p1), "nombres": nombres_p1,
        "referencia": {"macro_f1": metricas_referencia["macro_f1"],
                       "accuracy": metricas_referencia["accuracy"],
                       "per_fold": por_fold_referencia},
        "candidata": {"macro_f1": metricas_candidata["macro_f1"],
                      "accuracy": metricas_candidata["accuracy"],
                      "per_fold": por_fold_candidata},
        "delta": round(delta, 4), "folds_positivos": folds_positivos,
        "controles_permutados": controles, "mejor_control": mejor_control,
        "bootstrap_ic95": [round(valor, 4) for valor in intervalo],
        "bootstrap_proporcion_positiva": round(intervalo[2], 4),
        "complementariedad": complemento, "criterios": criterios, "adopta": adopta,
        "fronteras": {"referencia": fronteras_referencia, "candidata": fronteras_candidata},
        "por_cabeza": cabezas, "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\ndelta {delta:+.4f} | folds positivos {folds_positivos}/5 | bootstrap "
          f"[{intervalo[0]:+.4f}, {intervalo[1]:+.4f}] | DECISION: "
          f"{'ADOPTAR' if adopta else 'NO ADOPTAR'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
