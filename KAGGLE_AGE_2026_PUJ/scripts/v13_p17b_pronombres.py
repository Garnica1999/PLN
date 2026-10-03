"""V13 · P17b — Fusion de los pronombres de persona con V12, con pre-chequeo, permutado y gate.

P17a dio un control *block-only* de 0,1918 (LogReg) frente a un suelo de azar de 0,1667: hay senal,
pero es debil, por debajo del mejor bloque lexico de la fase cerrada (0,2033). Al mismo tiempo, el
cribado dio **0 de 26 variables redundantes** contra V12, con correlacion maxima 0,4757: el bloque
ocupa direcciones que V12 no usa. Ese es el mejor punto de partida posible para anadir informacion,
y tambien el que mas justifica no gastar una hora de CPU a ciegas.

Por eso este script tiene **dos etapas** y una puerta entre ellas:

**Etapa 1, pre-chequeo barato (sin meta-modelo).** Se entrena un LogReg sobre el bloque de
pronombres con los mismos folds y se mide su OOF. Luego se cuenta, sobre los errores de V12, a
cuantas de esas filas las acierta: el rescate, el dano y el neto (§26). Si el rescate neto es
marginal, la fusion no puede mover la macro-F1 y el experimento se detiene aqui, dejando constancia
de por que. Es la leccion de P15: antes de pagar una evaluacion cara, medir si hay algo que
recuperar.

**Etapa 2, fusion completa.** Solo si la etapa 1 pasa la puerta: combinador de produccion sobre
V12 mas el bloque, tres controles de permutado (DD-6), gate completo (§24), doble metrica (§36) y
reporte por cabeza (DD-8), que es donde se mide la prediccion pre-registrada del plan: el bloque
linguistico debe subir la rama LogReg y diluirse en la MLP.

Puerta de la etapa 1: rescate neto sobre los errores de V12 de al menos 1.000 filas (el 1,5 % de
los 67.610 errores) y rescate neto positivo en **las dos** fronteras centrales. El umbral esta
justificado en el propio tamaño del error: por debajo de 1.000 filas, ningun modelo de 279
variables y 6 clases puede mover la macro-F1 tres decimales.

El test no se toca. Ninguna submission se genera aqui.

Entregables: ``reports/v17_pronombres_fusion.md``, ``experiments/v17_p17b_pronombres.json``,
``features/v17_pronombres_fusion_oof.npy`` (solo si se llega a la etapa 2).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from phase2_ponderacion import build_features  # noqa: E402
from v13_p1b_fusion_tokens import (PARES, SEMILLAS, bootstrap_diferencia,  # noqa: E402
                                   complementariedad, logreg, metricas_frontera,
                                   metricas_globales, mlp, oof_una_cabeza)

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.config import set_seed  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

INFORME = cfg.REPORTS / "v17_pronombres_fusion.md"
OOF = cfg.FEATURES / "v17_pronombres_fusion_oof.npy"
RESULTADO = cfg.EXPERIMENTS / "v17_p17b_pronombres.json"

DELTA_MINIMO = 0.003
FOLDS_POSITIVOS_MINIMOS = 4
RESCATE_MINIMO = 1000
PORCENTAJE_ERROR = 0.015  # 1,5 % de los errores de V12


def f1(clase_true: np.ndarray, predicciones: np.ndarray) -> float:
    """Macro-F1 de unas predicciones expresadas como indice de clase."""
    return float(f1_score(clase_true, predicciones, average="macro",
                          labels=list(range(len(cfg.CLASSES)))))


def main() -> int:
    """Mide el rescate del bloque de pronombres y, si lo merece, lo fusiona con V12."""
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
    matriz = np.load(cfg.FEATURES / "v17_pronombres.npy")
    nombres = json.loads((cfg.FEATURES / "v17_pronombres_nombres.json").read_text(encoding="utf-8"))
    referencia = np.load(cfg.FEATURES / "oof_ensemble_v12.npy")
    metricas_referencia = metricas_globales(referencia, clase)
    por_fold_referencia = [f1(clase[fold["val_idx"]],
                               np.argmax(referencia[fold["val_idx"]], axis=1))
                           for fold in folds]
    errores = int((np.argmax(referencia, axis=1) != clase).sum())
    print(f"[referencia V12] macro-F1 {metricas_referencia['macro_f1']} | errores {errores:,} | "
          f"folds {[round(v, 4) for v in por_fold_referencia]}", flush=True)

    # ---------------------------------------------------------------- etapa 1: pre-chequeo
    oof_bloque = oof_una_cabeza(matriz, y, folds, "logreg")
    metricas_bloque = metricas_globales(oof_bloque, clase)
    acierto_v12 = np.argmax(referencia, axis=1) == clase
    acierto_bloque = np.argmax(oof_bloque, axis=1) == clase
    rescate = int((~acierto_v12 & acierto_bloque).sum())
    dano = int((acierto_v12 & ~acierto_bloque).sum())
    rescate_frontera = {}
    for (clase_a, clase_b), etiqueta in PARES.items():
        mascara = np.isin(clase, (cfg.CLASSES.index(clase_a), cfg.CLASSES.index(clase_b)))
        rescate_frontera[etiqueta] = {
            "rescate": int((~acierto_v12 & acierto_bloque & mascara).sum()),
            "dano": int((acierto_v12 & ~acierto_bloque & mascara).sum()),
        }
        rescate_frontera[etiqueta]["neto"] = (rescate_frontera[etiqueta]["rescate"]
                                             - rescate_frontera[etiqueta]["dano"])
    umbral = max(RESCATE_MINIMO, int(PORCENTAJE_ERROR * errores))
    pasa = (rescate - dano) >= umbral and all(
        medidas["neto"] > 0 for medidas in rescate_frontera.values())
    print(f"[pre-chequeo] bloque solo {metricas_bloque['macro_f1']} | rescate {rescate:,} | "
          f"dano {dano:,} | neto {rescate - dano:+,} | umbral {umbral:,} | "
          f"fronteras {[(k, v['neto']) for k, v in rescate_frontera.items()]}"
          f" | pasa {pasa}", flush=True)

    if not pasa:
        lineas = [
            "# V13 · P17b — Fusion de los pronombres con V12", "",
            "## 1. Pre-chequeo (sin meta-modelo)", "",
            f"El bloque de pronombres por si solo da {metricas_bloque['macro_f1']} de macro-F1 "
            f"(suelo de azar {1 / len(cfg.CLASSES):.4f}) y no toca ninguna de las 279 variables de "
            f"V12 (0 de 26 redundantes). Aun asi, antes de gastar una evaluacion del combinador se "
            f"mide cuanto del error de V12 sabe recuperar:", "",
            "| Medida | Filas |", "|---|---|",
            f"| Errores de V12 | {errores:,} |",
            f"| Rescates del bloque de pronombres | {rescate:,} |",
            f"| Danos del bloque de pronombres | {dano:,} |",
            f"| **Neto** | **{rescate - dano:+,}** |",
            f"| Umbral de paso | {umbral:,} |", "",
        ]
        for etiqueta, medidas in rescate_frontera.items():
            lineas.append(f"- {etiqueta}: rescate {medidas['rescate']:,} · dano {medidas['dano']:,} "
                          f"· neto {medidas['neto']:+,}")
        lineas += ["", "## 2. Decision", "",
                   f"El neto {rescate - dano:+,} **no alcanza** el umbral de {umbral:,} "
                   f"({PORCENTAJE_ERROR:.1%} de los errores), y ademas "
                   + ("ninguna" if not any(m["neto"] > 0 for m in rescate_frontera.values())
                      else "no todas") + " frontera central mejora. Un bloque que solo recupera "
                   f"{rescate:,} filas de {errores:,} no puede desplazar la macro-F1 tres decimales.",
                   "", "**La pista P17 se cierra sin fusionar.** No se ejecuta la etapa 2 y no se "
                   "consume el resto del presupuesto: el resultado ya es concluyente y el ahorro "
                   "de ~70 minutos de CPU es real. El lexicon queda documentado y el bloque "
                   "`features/v17_pronombres.npy` se conserva por si otra Combinacion lo "
                   "necesitara.", ""]
        INFORME.write_text("\n".join(lineas) + "\n", encoding="utf-8")
        RESULTADO.write_text(json.dumps({
            "fase": "V13 · P17b pronombres", "fecha": "2026-09-25",
            "etapa": "1 (pre-chequeo) y cierre: no procede fusion",
            "bloque_macro_f1": metricas_bloque["macro_f1"],
            "referencia_macro_f1": metricas_referencia["macro_f1"],
            "errores_v12": errores, "rescate": rescate, "dano": dano, "neto": rescate - dano,
            "umbral": umbral, "pasa_prechequeo": False,
            "rescate_por_frontera": rescate_frontera,
            "adopta": False, "segundos": round(time.time() - inicio, 1),
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nartefactos: {INFORME.name} (pista cerrada sin fusionar)")
        return 0

    # ---------------------------------------------------------------- etapa 2: fusion
    def fusionar(bloques: np.ndarray) -> np.ndarray:
        """OOF del combinador de produccion sobre V12 mas el bloque indicado."""
        probabilidades = np.zeros((len(clase), len(cfg.CLASSES)))
        matriz_completa = np.hstack([base, bloques.astype(np.float64)])
        for fold in folds:
            validacion = fold["val_idx"]
            entrenamiento = np.concatenate([otro["val_idx"] for otro in folds
                                            if otro["fold"] != fold["fold"]])
            lineal = logreg().fit(matriz_completa[entrenamiento], y[entrenamiento])
            red = mlp().fit(matriz_completa[entrenamiento], y[entrenamiento])
            probabilidades[validacion] = 0.5 * lineal.predict_proba(
                matriz_completa[validacion]) + 0.5 * red.model.predict_proba(
                    red.scaler.transform(matriz_completa[validacion]))
            etiquetas = np.array(cfg.CLASSES)[np.argmax(probabilidades[validacion], axis=1)]
            print(f"    fold {fold['fold']}: "
                  f"{f1_score(y[validacion], etiquetas, average='macro', labels=cfg.CLASSES):.4f}",
                  flush=True)
        return probabilidades

    candidata = fusionar(matriz)
    metricas_candidata = metricas_globales(candidata, clase)
    fronteras_candidata = metricas_frontera(candidata, clase, PARES)
    fronteras_referencia = metricas_frontera(referencia, clase, PARES)
    por_fold_candidata = [float(f1(clase[fold["val_idx"]],
                                   np.argmax(candidata[fold["val_idx"]], axis=1)))
                         for fold in folds]
    delta = metricas_candidata["macro_f1"] - metricas_referencia["macro_f1"]
    folds_positivos = sum(1 for a, b in zip(por_fold_candidata, por_fold_referencia) if b > a)
    mejoras = [a - b for a, b in zip(por_fold_candidata, por_fold_referencia)]
    np.save(OOF, candidata)

    controles = []
    for semilla in SEMILLAS:
        permutada = matriz[np.random.default_rng(semilla).permutation(len(y))]
        valor = metricas_globales(fusionar(permutada), clase)["macro_f1"]
        controles.append({"semilla": semilla, "macro_f1": valor,
                          "delta": round(valor - metricas_referencia["macro_f1"], 4)})
    mejor_control = max(controles, key=lambda control: control["macro_f1"])
    intervalo = bootstrap_diferencia(referencia, candidata, clase, len(cfg.CLASSES))
    fronteras_mejoras = sum(1 for etiqueta in PARES.values()
                            if fronteras_candidata[etiqueta]["accuracy"]
                            > fronteras_referencia[etiqueta]["accuracy"])
    criterios = {
        "delta_minimo": bool(delta >= DELTA_MINIMO),
        "folds_positivos": bool(folds_positivos >= FOLDS_POSITIVOS_MINIMOS),
        "bootstrap_excluye_cero": bool(intervalo[0] > 0),
        "supera_control_permutado": bool(metricas_candidata["macro_f1"]
                                         > mejor_control["macro_f1"]),
        "no_explicada_por_una_fold": bool(max(mejoras) <= sum(mejoras) - max(mejoras) + 1e-9),
        "fronteras_no_empeoran": bool(fronteras_mejoras >= len(PARES)),
    }
    adopta = all(criterios.values())
    cabezas = {}
    for cabeza in ("logreg", "mlp"):
        valor_base = metricas_globales(oof_una_cabeza(base, y, folds, cabeza), clase)["macro_f1"]
        valor_nueva = metricas_globales(
            oof_una_cabeza(np.hstack([base, matriz.astype(np.float64)]), y, folds, cabeza),
            clase)["macro_f1"]
        cabezas[cabeza] = {"referencia": valor_base, "candidata": valor_nueva,
                           "delta": round(valor_nueva - valor_base, 4)}
    complemento = complementariedad(referencia, candidata, clase)

    lineas = [
        "# V13 · P17b — Fusion de los pronombres con V12", "",
        "## 1. Pre-chequeo (sin meta-modelo)", "",
        f"| Medida | Filas |", "|---|---|",
        f"| Errores de V12 | {errores:,} |",
        f"| Rescates del bloque | {rescate:,} |",
        f"| Danos del bloque | {dano:,} |",
        f"| **Neto** | **{rescate - dano:+,}** |",
        f"| Umbral de paso | {umbral:,} |", "",
        f"El pre-chequeo {'pasa' if pasa else 'no pasa'}: con ese margen de rescate hay motivo para "
        f"pagar la evaluacion del combinador.", "",
        "## 2. Fusion", "",
        "| Modelo | macro-F1 |", "|---|---|",
        f"| V12 (referencia) | {metricas_referencia['macro_f1']} |",
        f"| V12 + pronombres | {metricas_candidata['macro_f1']} |",
    ]
    for control in controles:
        lineas.append(f"| Control permutado (semilla {control['semilla']}) | "
                      f"{control['macro_f1']} |")
    lineas += ["", f"Diferencia {delta:+.4f} · folds {[round(v, 4) for v in por_fold_candidata]} "
                  f"frente a {[round(v, 4) for v in por_fold_referencia]} · {folds_positivos}/5 "
                  f"positivos · bootstrap [{intervalo[0]:+.4f}, {intervalo[1]:+.4f}]", "",
              "## 3. Doble metrica (§36)", "",
              "| Frontera | accuracy referencia | accuracy candidata |",
              "|---|---|---|"]
    for etiqueta in PARES.values():
        lineas.append(f"| {etiqueta} | {fronteras_referencia[etiqueta]['accuracy']} | "
                      f"{fronteras_candidata[etiqueta]['accuracy']} |")
    lineas += ["", "## 4. Por cabeza (DD-8)", "",
               "La prediccion pre-registrada del plan es que un bloque linguistico sube la rama "
               "LogReg y se diluye en la MLP. La tabla lo dice.", "",
               "| Cabeza | Referencia | Candidata | Diferencia |", "|---|---|---|---|"]
    for cabeza, medidas in cabezas.items():
        lineas.append(f"| {cabeza} | {medidas['referencia']} | {medidas['candidata']} | "
                      f"{medidas['delta']:+.4f} |")
    lineas += ["", "## 5. Gate (§24)", "", "| Criterio | Cumple |", "|---|---|"]
    for nombre, cumple in criterios.items():
        lineas.append(f"| {nombre.replace('_', ' ')} | {'si' if cumple else '**no**'} |")
    lineas += ["", f"**Decision: {'ADOPTAR' if adopta else 'NO ADOPTAR'}**", ""]
    INFORME.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    RESULTADO.write_text(json.dumps({
        "fase": "V13 · P17b pronombres", "fecha": "2026-09-25", "etapa": "2 (fusion)",
        "pasa_prechequeo": True, "rescate": rescate, "dano": dano, "neto": rescate - dano,
        "referencia": {"macro_f1": metricas_referencia["macro_f1"],
                       "per_fold": [round(v, 4) for v in por_fold_referencia]},
        "candidata": {"macro_f1": metricas_candidata["macro_f1"],
                      "per_fold": [round(v, 4) for v in por_fold_candidata]},
        "delta": round(delta, 4), "folds_positivos": folds_positivos,
        "controles_permutados": controles, "mejor_control": mejor_control,
        "bootstrap_ic95": [round(v, 4) for v in intervalo],
        "complementariedad": complemento, "criterios": criterios, "adopta": adopta,
        "fronteras": {"referencia": fronteras_referencia, "candidata": fronteras_candidata},
        "por_cabeza": cabezas, "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\ndelta {delta:+.4f} | folds positivos {folds_positivos}/5 | DECISION: "
          f"{'ADOPTAR' if adopta else 'NO ADOPTAR'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
