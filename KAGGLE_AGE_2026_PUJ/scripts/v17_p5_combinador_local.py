"""V17 · P5 — Meta-combinador local por frontera: combinar en vez de seleccionar.

Esta es la ultima pista con una hipotesis genuinamente distinta, y la distincion es el objeto del
script.

- **P9 pregunto "a quien escucho"**. Es una pregunta de **seleccion**: entre V12 y el
  especialista, cual acierta. Su respuesta fue que la frontera que hay que aprender no existe con
  las señales disponibles, y por eso el arbitro se apaga.
- **P5 pregunta "que clase es"**. Es una pregunta de **combinacion**: dentro de la frontera, cual
  de las dos clases es, usando como evidencia las probabilidades de todos los miembros y los
  margenes de los pares. No hay dos modelos que elegir: hay una sola respuesta, y la evidencia se
  mezcla.

Por que la combinacion puede funcionar donde la seleccion no puede. P1 y P9 midieron que E-028a
gana 14.356 filas a V12 en la frontera, y que solo el 12,7 % de esas victorias caen dentro de la
region donde V12 tiene el par arriba. Pero hay un segundo hecho, medido en V15a al listar las
variables de la matriz: **el margen del par ya es una entrada de V12**, y V12 no lo explota. Hay
informacion accesible que el combinador global no usa. Un combinador local si la puede usar, porque
no tiene que decidir si V12 se equivoca: solo tiene que repartir mejor la masa dentro del par.

**Como se inyecta la evidencia, y por que asi.** El combinador local produce una probabilidad `q` de
que la fila sea de la clase superior del par, para **toda** fila, no solo para las de la frontera.
Esa probabilidad **no se usa como decision**: se usa para **redistribuir la masa que V12 ya pone en
las dos clases del par**. Concretamente, si V12 da masa `a` a la clase inferior y `b` a la
superior, laversion corregida reparte ese mismo total entre las dos en proporcion `1 - q` y `q` en
vez de en proporcion `a` y `b`. Tres propiedades importan y son deliberadas:

1. **La masa total del par no cambia**, asi que el combinador local solo puede repartir mejor
   lo que ya se atribuia a esas dos clases: no puede mejorar la situacion de las
   clases faciles.
2. **No puede mover masa hacia dentro del par desde fuera**, asi que no puede danar a las clases vecinas
   Es la limitacion que hace segura una inyeccion local.
3. **Actua en todas las filas**, no en una region activada, que es justo lo que hacia P9 y las
   compuertas.

La fuerza de la inyeccion se pondera con `w`, de modo que `w = 0` es V12 exacto y `w = 1` es
sustitucion completa. `w` se elige en una particion **interior** del train exterior, minimizando
criterio de llamada, y nunca con la validacion exterior.

**Protocolo**, anidado en tres niveles igual que P9: el combinador local se ajusta en las filas del
par del train exterior, `w` se elige en un bloque interior agrupado que el combinador no ha visto, y
la validacion exterior no participa en nada. Las salidas de V12 y de A2 son las OOF ya guardadas, de
modo que no hay que reentrenar ninguna cabeza y el unico modelo nuevo es el combinador local, que es
un LogReg de 47 entradas.

Entregables: ``experiments/v17_p5_combinador_local.json``, ``reports/v17_p5_combinador_local.md``,
``reports/v17_p5_combinador_local.csv``, ``features/v17_p5_oof.npy`` y una figura.
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

from v13_p1b_fusion_tokens import PARES, bootstrap_diferencia, metricas_frontera, metricas_globales  # noqa: E402
from v13_p15_matriz_errores_miembros import predicciones_de_miembros  # noqa: E402
from v17_p9_arbitro import FRACCION_UMBRAL, arbitro_entradas, partido_agrupado  # noqa: E402

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.config import set_seed  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

INFORME = cfg.REPORTS / "v17_p5_combinador_local.md"
CURVA = cfg.REPORTS / "v17_p5_combinador_local.csv"
RESULTADO = cfg.EXPERIMENTS / "v17_p5_combinador_local.json"
OOF = cfg.FEATURES / "v17_p5_oof.npy"
CENTRALES = (("18-24", "25-34", 1), ("25-34", "35-49", 2))
REFERENCIA_CANONICA = 0.6005
GATE_EXPLORATORIO = 0.0015
GATE_ADOPCION = 0.003
FOLDS_POSITIVOS_MINIMOS = 4
#: Forces of injection. w = 0 is exactly V12 and w = 1 is total replacement of the ordering inside
#: the pair. They are few on purpose: five points say the shape.
FUERZAS = (0.0, 0.25, 0.50, 0.75, 1.0)
C_COMBINADOR = 1.0


def etiquetas_de(matriz: np.ndarray) -> np.ndarray:
    """Nombres de clase a partir de una matriz de probabilidades.

    Existe porque `f1_score` no admite mezclar etiquetas de texto con indices numericos, y
    `np.argmax` devuelve indices. Comparar las dos cosas es un error que ha aparecido varias veces
    en este proyecto, asi que la conversion vive en un solo sitio.
    """
    return np.array(cfg.CLASSES)[np.argmax(matriz, axis=1)]


def redistribuir(probabilidades: np.ndarray, inferior: str, superior: str,
                 q: np.ndarray, fuerza: float) -> np.ndarray:
    """Reparte la masa del par entre sus dos clases en proporcion a la evidencia local.

    La masa total del par se conserva exactamente; solo cambia el reparto interno. Con `fuerza` 0
    el resultado es identico a V12, porque la proporcion original se conserva.
    """
    izq, der = cfg.CLASSES.index(inferior), cfg.CLASSES.index(superior)
    salida = probabilidades.copy()
    if fuerza == 0.0:
        return salida
    masa = probabilidades[:, izq] + probabilidades[:, der]
    proporcion_original = np.divide(probabilidades[:, der], masa, out=np.full(len(masa), 0.5),
                                    where=masa > 1e-12)
    proporcion_nueva = (1.0 - fuerza) * proporcion_original + fuerza * q
    salida[:, der] = masa * proporcion_nueva
    salida[:, izq] = masa * (1.0 - proporcion_nueva)
    return salida


def main() -> int:
    """Ajusta el combinador local por frontera y mide la inyeccion con el gate completo."""
    autotest_numerico()
    set_seed()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(valor) for valor in y])
    clusters_ids = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    v12 = np.load(cfg.FEATURES / "oof_ensemble_v12.npy")
    a2 = np.load(cfg.FEATURES / "v16_oof_combinador_logistico.npy")
    pares = np.load(cfg.FEATURES / "oof_E028a_pares.npy")
    grupos = np.load(cfg.FEATURES / "oof_E028b_grupos.npy")
    votos, nombres_miembros, margenes_miembros, _ = predicciones_de_miembros()
    entradas = arbitro_entradas(v12, a2, pares, grupos, votos, margenes_miembros)
    print(f"[entradas] {entradas.shape[1]} columnas derivadas ya existentes | "
          f"{len(nombres_miembros)} miembros | ninguna nueva", flush=True)
    referencia = v12
    metricas_referencia = metricas_globales(referencia, clase)
    fronteras_referencia = metricas_frontera(referencia, clase, PARES)

    folds = cvmod.make_folds(y, groups=clusters_ids, n_splits=5, n_repeats=1, scheme="group")
    generador = np.random.default_rng(cfg.SEED)
    probabilidades = v12.copy()
    filas: list[dict] = []
    por_fold: list[float] = []
    for fold in folds:
        validacion = fold["val_idx"]
        entrenamiento = np.concatenate([otro["val_idx"] for otro in folds
                                        if otro["fold"] != fold["fold"]])
        for inferior, superior, _ in CENTRALES:
            # El combinador local solo se ajusta en filas cuya etiqueta real es del par, que es lo
            # que exige el brief, y nunca ve la validacion exterior.
            en_par = np.flatnonzero(((y == inferior) | (y == superior))
                                    & np.isin(np.arange(len(y)), entrenamiento))
            if len(en_par) < 500:
                continue
            ajuste, eleccion = partido_agrupado(en_par, clusters_ids, FRACCION_UMBRAL, generador)
            objetivo = (y[ajuste] == superior).astype(int)
            combinador = LogisticRegression(C=C_COMBINADOR, max_iter=3000, random_state=cfg.SEED)
            combinador.fit(entradas.iloc[ajuste].to_numpy(), objetivo)
            # La fuerza se elige en el bloque que el combinador no ha visto, sobre la macro-F1
            # completa de las seis clases, que es la metrica que se va a reportar.
            q_eleccion = combinador.predict_proba(entradas.iloc[eleccion].to_numpy())[:, 1]
            mejor_fuerza, mejor_valor = 0.0, -np.inf
            for fuerza in FUERZAS:
                candidata = redistribuir(v12[eleccion], inferior, superior, q_eleccion, fuerza)
                valor = float(f1_score(y[eleccion], etiquetas_de(candidata),
                                       average="macro", labels=cfg.CLASSES))
                if valor > mejor_valor:
                    mejor_fuerza, mejor_valor = fuerza, valor
            filas.append({"Fold": fold["fold"], "Frontera": f"{inferior}|{superior}",
                          "Filas_par_train": int(len(en_par)),
                          "Fuerza_elegida": mejor_fuerza,
                          "MacroF1_interior": round(mejor_valor, 4),
                          "MacroF1_interior_sin_inyeccion": round(float(f1_score(
                              y[eleccion], etiquetas_de(v12[eleccion]), average="macro",
                              labels=cfg.CLASSES)), 4)})
            # Aplicacion a la validacion exterior, con el combinador reentrenado en todo el par
            objetivo_completo = (y[en_par] == superior).astype(int)
            combinador = LogisticRegression(C=C_COMBINADOR, max_iter=3000, random_state=cfg.SEED)
            combinador.fit(entradas.iloc[en_par].to_numpy(), objetivo_completo)
            q_validacion = combinador.predict_proba(entradas.iloc[validacion].to_numpy())[:, 1]
            probabilidades[validacion] = redistribuir(v12[validacion], inferior, superior,
                                                      q_validacion, mejor_fuerza)
        valor = float(f1_score(y[validacion], etiquetas_de(probabilidades[validacion]),
                               average="macro", labels=cfg.CLASSES))
        por_fold.append(valor)
        print(f"    fold {fold['fold']}: {valor:.4f} | fuerzas elegidas "
              + " · ".join(f"{f['Frontera']} {f['Fuerza_elegida']:.2f}"
                           for f in filas if f["Fold"] == fold["fold"]), flush=True)

    metricas = metricas_globales(probabilidades, clase)
    fronteras = metricas_frontera(probabilidades, clase, PARES)
    delta = metricas["macro_f1"] - metricas_referencia["macro_f1"]
    por_fold_referencia = [float(f1_score(y[fold["val_idx"]],
                                          etiquetas_de(v12[fold["val_idx"]]),
                                          average="macro", labels=cfg.CLASSES)) for fold in folds]
    folds_positivos = sum(1 for a, b in zip(por_fold, por_fold_referencia) if a > b)
    intervalo = bootstrap_diferencia(referencia, probabilidades, clase, len(cfg.CLASSES))
    np.save(OOF, probabilidades)
    tabla = pd.DataFrame(filas)
    tabla.to_csv(CURVA, index=False)
    mejoras_frontera = sum(1 for etiqueta in PARES.values()
                           if fronteras[etiqueta]["accuracy"] > fronteras_referencia[etiqueta]["accuracy"])
    criterios = {
        "efecto_minimo": bool(delta >= GATE_ADOPCION),
        "folds_positivos": bool(folds_positivos >= FOLDS_POSITIVOS_MINIMOS),
        "significacion": bool(intervalo[0] > 0),
        "fronteras": bool(mejoras_frontera >= len(PARES)),
    }
    adopta = all(criterios.values())
    print(f"[P5] macro-F1 {metricas['macro_f1']} | delta {delta:+.4f} | folds positivos "
          f"{folds_positivos}/5 | IC95 [{intervalo[0]:+.4f}, {intervalo[1]:+.4f}] | fronteras "
          f"{fronteras['18-24|25-34']['accuracy']} y {fronteras['25-34|35-49']['accuracy']}",
          flush=True)

    figura = cfg.FIGURES / "v17_p5_combinador_local.png"
    fig, eje = plt.subplots(figsize=(9, 5))
    posiciones = np.arange(len(por_fold))
    eje.bar(posiciones - 0.2, por_fold_referencia, 0.4, label="V12", color="#2166ac")
    eje.bar(posiciones + 0.2, por_fold, 0.4, label="V12 con combinador local", color="#1a9850")
    eje.axhline(REFERENCIA_CANONICA, color="grey", linestyle=":", linewidth=1)
    eje.set_xticks(posiciones)
    eje.set_xticklabels([f"fold {n + 1}" for n in range(len(por_fold))])
    eje.set_ylabel("macro-F1 del fold")
    eje.legend(fontsize=9)
    eje.set_title("V17 P5: inyeccion del combinador local, fold por fold", fontsize=10)
    fig.tight_layout()
    fig.savefig(figura, dpi=130)
    plt.close(fig)

    lineas = [
        "# V17 · P5 — Meta-combinador local por frontera", "",
        "## 1. Que cambia respecto a P9", "",
        "P9 preguntaba **a quien escucho** (seleccion entre dos modelos) y su respuesta fue que la "
        "frontera que hay que aprender no existe. P5 pregunta **que clase es** (combinacion), que "
        "es una hipotesis distinta: no hay dos modelos que elegir, hay una sola respuesta y la "
        "evidencia se mezcla.", "",
        "## 2. Como se inyecta la evidencia", "",
        "El combinador local da una probabilidad `q` de la clase superior del par para **toda** "
        "fila. Esa probabilidad **no se usa como decision**: se usa para **redistribuir la masa que "
        "V12 ya pone en las dos clases del par**, conservando el total. Por tanto (1) no puede "
        "mejorar la situacion de las clases faciles, porque no les mueve masa; (2) no puede "
        "empeorar las otras cuatro clases del par por entrar o salir de ellas; y (3) actua en todas "
        "las filas, no en una region activada.", "",
        "La fuerza `w` va de 0 (V12 exacto) a 1 (sustitucion completa del reparto interno) y se "
        "elige en un bloque interior agrupado que el combinador no ha visto, por macro-F1 completa.",
        "",
        "## 3. Resultado por fold", "",
        "| Fold | V12 | Con combinador local | Diferencia |", "|---|---|---|---|",
    ]
    for indice, (a, b) in enumerate(zip(por_fold_referencia, por_fold), start=1):
        lineas.append(f"| {indice} | {a:.4f} | {b:.4f} | {a - b:+.4f} |")
    lineas += ["", f"**Global: {metricas['macro_f1']} frente a {metricas_referencia['macro_f1']}, "
                  f"delta {delta:+.4f}** · folds positivos {folds_positivos}/5 · IC95 pareado "
                  f"[{intervalo[0]:+.4f}, {intervalo[1]:+.4f}]", "",
              "## 4. Doble metrica: las fronteras", "",
              "| Frontera | V12 | Con combinador local |", "|---|---|---|"]
    for etiqueta in PARES.values():
        lineas.append(f"| {etiqueta} | {fronteras_referencia[etiqueta]['accuracy']} | "
                      f"{fronteras[etiqueta]['accuracy']} |")
    lineas += ["", "## 5. Fuerza elegida por fold y frontera", "",
              "| Fold | Frontera | Filas del par en train | Fuerza elegida | macro-F1 interior | "
              "sin inyeccion |", "|---|---|---|---|---|---|"]
    for _, fila in tabla.iterrows():
        lineas.append(f"| {fila['Fold']} | {fila['Frontera']} | {fila['Filas_par_train']:,} | "
                      f"{fila['Fuerza_elegida']:.2f} | {fila['MacroF1_interior']} | "
                      f"{fila['MacroF1_interior_sin_inyeccion']} |")
    lineas += ["", "## 6. Gate de adopcion de V17", "", "| Criterio | Valor | Cumple |", "|---|---|---|"]
    valores = {"efecto_minimo": f"delta {delta:+.4f} >= {GATE_ADOPCION}",
               "folds_positivos": f"{folds_positivos}/5 >= {FOLDS_POSITIVOS_MINIMOS}",
               "significacion": f"IC95 [{intervalo[0]:+.4f}, {intervalo[1]:+.4f}]",
               "fronteras": f"{mejoras_frontera}/{len(PARES)} mejoran"}
    for nombre, cumple in criterios.items():
        lineas.append(f"| {nombre.replace('_', ' ')} | {valores[nombre]} | "
                      f"{'si' if cumple else '**no**'} |")
    lineas += ["", f"**Decision: {'ADOPTAR' if adopta else 'NO ADOPTAR'}**", ""]
    INFORME.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    RESULTADO.write_text(json.dumps({
        "fase": "V17 · P5 meta-combinador local por frontera", "fecha": "2026-09-26",
        "n_entradas": int(entradas.shape[1]), "entradas_nuevas": 0,
        "combinador": {"modelo": "LogReg", "C": C_COMBINADOR, "fuerzas": list(FUERZAS),
                       "inyeccion": "redistribucion de la masa del par conservando el total",
                       "protocolo": "anidado: ajuste en el par del train exterior, fuerza elegida "
                                    "en bloque interior agrupado"},
        "referencia": {"macro_f1": metricas_referencia["macro_f1"],
                       "por_fold": [round(v, 4) for v in por_fold_referencia]},
        "candidata": {"macro_f1": metricas["macro_f1"], "accuracy": metricas["accuracy"],
                      "por_fold": [round(v, 4) for v in por_fold]},
        "delta": round(delta, 6), "folds_positivos": folds_positivos,
        "bootstrap_ic95": [round(v, 6) for v in intervalo],
        "fronteras_referencia": {k: v["accuracy"] for k, v in fronteras_referencia.items()},
        "fronteras_candidata": {k: v["accuracy"] for k, v in fronteras.items()},
        "criterios": criterios, "adopta": adopta,
        "fuerzas": tabla.to_dict(orient="records"), "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\ndelta {delta:+.4f} | folds {folds_positivos}/5 | IC95 [{intervalo[0]:+.4f}, "
          f"{intervalo[1]:+.4f}] | DECISION: {'ADOPTAR' if adopta else 'NO ADOPTAR'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
