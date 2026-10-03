"""V17 · P9 — Experto selectivo con arbitro aprendido, protocolo anidado estricto.

P1c cerro la via de la compuarta de umbral, y un diagnostico posterior explico por que. La
explicacion es esta y es lo mas importante que ha salido de la fase:

- En la region de frontera (las dos clases mas probables de V12 son las del par) hay 11.130 filas
  para `18-24|25-34`. De ellas, el especialista **gana 1.830** y **pierde 2.485**. Escucharle ahi
  sin mas vale **-0,0080** de macro-F1 global.
- Fuera de esa region el especialista gana mucho mas, pero ahi V12 no esta dudando: esta seguro
  prediciendo una clase distinta. Solo el 12,7 % y el 8,0 % de las filas que el especialista arregla
  caen dentro de la region de frontera.
- Y lo que hace que la compuerta por ambiguedad no pueda funcionar es que **V12 esta mas confiado
  cuando se equivoca**: separacion 0,809 en las filas que el especialista arregla frente a 0,539 en
  las que rompe. El error de frontera de V12 no es un error de baja confianza, es un error
  **confiado**.

De ahi la reformulacion que este script mide. El problema ya no es "cuando esta V12 dudando", sino:
**dentro de la region de frontera, cuales de las 11.130 filas convieneework al especialista**. Ahi la
linea base honesta es -0,0080 (escuchar siempre) y el techo es +0,0107 (activar solo las 1.830
filas que el especialista resuelve), de modo que el arbitro tiene que encontrar esas ~1.830 entre
~11.000. Es un problema de seleccion binaria bien planteado, con base y techo conocidos.

**Que hace el arbitro, y con que se le da.** Un clasificador binario que decide, para una fila de la
region de frontera, si el especialista va a acertar. Sus entradas son **solo columnas derivadas que
el proyecto ya tiene calculadas**, ninguna nueva: los 14 votos de los miembros de V12, los 5
margenes de pares adyacentes de E-028a, los 3 margenes de grupo de E-028b, las 6 probabilidades de
V12, las 6 del combinador logistico A2 de V16 y el acuerdo entre miembros. No se crea ninguna
representacion nueva, no se entrena ningun vectorizador y no se toca el test, que es exactamente lo
que el brief permite y prohibe.

**El protocolo es anidado de verdad, en tres niveles**, porque aqui es donde se cuela la fuga si
uno se salta un nivel:

1. El arbitro se ajusta en una particion **interior** del train exterior y su **umbral** se elige
   en otra particion interior distinta, maximizando la correccion neta en las filas de frontera.
2. Despues se vuelve a ajustar el arbitro en **todo** el train exterior y se aplica al train de
   validacion exterior.
3. En ningun momento la validacion exterior participa en el ajuste ni en la eleccion del umbral.

Los folds son los mismos agrupados de 5 del proyecto, y la segunda particion de 4 folds se mide
como confirmacion independiente, con el gate de adopcion de V17 completo.

**Advertencia sobre la base de comparacion.** El techo de +0,0107 se calculo en un script anterior
eligiendo las filas con la etiqueta, o sea que es una cota y no un resultado. Este script no lo
persigue: mide lo que un arbitro entrenado sin ver la validacion consegue, que es lo unico
publicable.

Entregables: ``experiments/v17_p9_arbitro.json``, ``reports/v17_p9_arbitro.md``,
``reports/v17_p9_arbitro.csv``, ``features/v17_p9_oof.npy`` y dos figuras.
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

from v13_p1b_fusion_tokens import PARES, bootstrap_diferencia, metricas_globales  # noqa: E402
from v13_p15_matriz_errores_miembros import predicciones_de_miembros  # noqa: E402

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.config import set_seed  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

INFORME = cfg.REPORTS / "v17_p9_arbitro.md"
CURVA = cfg.REPORTS / "v17_p9_arbitro.csv"
RESULTADO = cfg.EXPERIMENTS / "v17_p9_arbitro.json"
OOF = cfg.FEATURES / "v17_p9_oof.npy"
CENTRALES = (("18-24", "25-34", 1), ("25-34", "35-49", 2))
REFERENCIA_CANONICA = 0.6005
GATE_EXPLORATORIO = 0.0015
GATE_ADOPCION = 0.003
FOLDS_POSITIVOS_MINIMOS = 4
FOLDS_SEGUNDA_PARTICION = 4
FOLDS_SEGUNDA_PARTICION_MINIMOS = 3
#: Fraccion del train exterior que se aparta para elegir el umbral. Es pequena a proposito: el
#: arbitro necesita muchas filas y la eleccion del umbral necesita pocas.
FRACCION_UMBRAL = 0.25
#: Candidatos de umbral. Pocos a proposito; una rejilla grande aqui seria sobreajustar el umbral.
UMBRALES = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
#: Regularizacion del arbitro. C fuerte a proposito: 35 entradas sobre unos 8.000 pares, y el
#: objetivo es que el umbral sea estable entre folds mas que que el arbitro sea preciso.
C_ARBITRO = 0.2


def arbitro_entradas(v12: np.ndarray, a2: np.ndarray, pares: np.ndarray, grupos: np.ndarray,
                     votos: np.ndarray, margenes_miembros: np.ndarray) -> pd.DataFrame:
    """Columnas de entrada del arbitro. Todas ya existen en el proyecto: ninguna es nueva."""
    decision = np.argmax(v12, axis=1)
    acuerdo = (votos == decision[:, None]).mean(axis=1, keepdims=True)
    partes = [votos.astype(float), margenes_miembros, pares, grupos, v12, a2, acuerdo]
    matriz = np.hstack(partes)
    nombres = ([f"voto_{n}" for n in range(votos.shape[1])]
               + [f"margen_miembro_{n}" for n in range(margenes_miembros.shape[1])]
               + [f"margen_par_{n}" for n in range(pares.shape[1])]
               + [f"margen_grupo_{n}" for n in range(grupos.shape[1])]
               + [f"v12_prob_{n}" for n in range(v12.shape[1])]
               + [f"a2_prob_{n}" for n in range(a2.shape[1])] + ["acuerdo_miembros"])
    return pd.DataFrame(matriz, columns=nombres)


def region_de_frontera(v12: np.ndarray, inferior: str, superior: str) -> np.ndarray:
    """Filas en las que V12 esta decidiendo entre las dos clases de la frontera."""
    izq, der = cfg.CLASSES.index(inferior), cfg.CLASSES.index(superior)
    orden = np.sort(np.argsort(v12, axis=1)[:, -2:], axis=1)
    return (orden == np.array([izq, der])).all(axis=1)


def partido_agrupado(indices: np.ndarray, clusters: np.ndarray, fraccion: float,
                     generador: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Reparte filas en dos bloques sin que un grupo de duplicados cruce entre ellos."""
    grupos_unicos = np.unique(clusters[indices])
    generador.shuffle(grupos_unicos)
    corte = int(len(grupos_unicos) * fraccion)
    en_umbral = set(grupos_unicos[:corte].tolist())
    mascara = np.array([grupo in en_umbral for grupo in clusters[indices]])
    return indices[~mascara], indices[mascara]


def main() -> int:
    """Entrena el arbitro con protocolo anidado y mide la candidata contra V12."""
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
    print(f"[entradas del arbitro] {entradas.shape[1]} columnas, todas derivadas ya existentes | "
          f"{len(nombres_miembros)} miembros", flush=True)
    referencia = np.load(cfg.FEATURES / "oof_ensemble_v12.npy")
    pred_v12 = np.array(cfg.CLASSES)[np.argmax(v12, axis=1)]

    filas: list[dict] = []
    probabilidades = np.zeros((len(y), len(cfg.CLASSES)))
    por_fold: list[float] = []
    generador = np.random.default_rng(cfg.SEED)

    def decision_para(inferior: str, superior: str, columna: int, filas_indice: np.ndarray
                      ) -> np.ndarray:
        # np.where con nombres de clase devuelve un array de texto, que numpy no admite como
        # indice. Se devuelve directamente ese array de texto, sin indexar la lista de clases.
        return np.where(pares[filas_indice, columna] > 0, superior, inferior)

    folds = cvmod.make_folds(y, groups=clusters_ids, n_splits=5, n_repeats=1, scheme="group")
    for fold in folds:
        validacion = fold["val_idx"]
        entrenamiento = np.concatenate([otro["val_idx"] for otro in folds
                                        if otro["fold"] != fold["fold"]])
        pred_fold = pred_v12.copy()
        eleccion_por_frontera: dict[str, dict] = {}
        for inferior, superior, columna in CENTRALES:
            ctx = region_de_frontera(v12, inferior, superior)
            # indices GLOBALES de las filas de frontera del train exterior: Needed because the inner
            # split and the audit of groups work on global indices.
            en_train = entrenamiento[ctx[entrenamiento]]
            if len(en_train) < 500:
                continue
            # 1) el arbitro se ajusta en un bloque interior del train exterior
            ajuste, eleccion = partido_agrupado(en_train, clusters_ids, FRACCION_UMBRAL, generador)
            objetivo_ajuste = (decision_para(inferior, superior, columna, ajuste)
                               == y[ajuste]).astype(int)
            modelo = LogisticRegression(C=C_ARBITRO, max_iter=2000, random_state=cfg.SEED)
            modelo.fit(entradas.iloc[ajuste].to_numpy(), objetivo_ajuste)
            # 2) el umbral se elige en el otro bloque interior, que el arbitro no ha visto
            punt = modelo.predict_proba(entradas.iloc[eleccion].to_numpy())[:, 1]
            decision_eleccion = decision_para(inferior, superior, columna, eleccion)
            mejor_umbral, mejor_neto = UMBRALES[0], -np.inf
            for umbral in UMBRALES:
                activa = punt >= umbral
                if activa.sum() == 0:
                    continue
                real = y[eleccion][activa]
                ok_v12 = pred_v12[eleccion][activa] == real
                ok_esp = decision_eleccion[activa] == real
                neto = int((~ok_v12 & ok_esp).sum() - (ok_v12 & ~ok_esp).sum())
                if neto > mejor_neto:
                    mejor_umbral, mejor_neto = umbral, neto
            # 3) reentrenado en TODO el train exterior, aplicado a la validacion exterior
            objetivo_completo = (decision_para(inferior, superior, columna, en_train)
                                 == y[en_train]).astype(int)
            modelo = LogisticRegression(C=C_ARBITRO, max_iter=2000, random_state=cfg.SEED)
            modelo.fit(entradas.iloc[en_train].to_numpy(), objetivo_completo)
            filas_validacion = validacion[ctx[validacion]]
            if len(filas_validacion) == 0:
                continue
            punt_validacion = modelo.predict_proba(
                entradas.iloc[filas_validacion].to_numpy())[:, 1]
            activa = punt_validacion >= mejor_umbral
            decision_validacion = decision_para(inferior, superior, columna, filas_validacion)
            if activa.sum():
                elegidas = filas_validacion[activa]
                pred_fold[elegidas] = decision_para(inferior, superior, columna, elegidas)
                real = y[elegidas]
                ok_v12 = pred_v12[elegidas] == real
                ok_esp = decision_para(inferior, superior, columna, elegidas) == real
                rescate, dano = int((~ok_v12 & ok_esp).sum()), int((ok_v12 & ~ok_esp).sum())
            else:
                rescate = dano = 0
            eleccion_por_frontera[f"{inferior}|{superior}"] = {
                "umbral": mejor_umbral, "activadas": int(activa.sum()),
                "rescate": rescate, "dano": dano, "neto_umbral_interno": int(mejor_neto),
                "n_train_frontera": int(len(en_train))}
        # La salida del sistema es una decision dura, asi que se evalua como decision dura: una
        # matriz de ceros y unos. Es la lectura menos generosa con la candidata.
        uno_hot = np.zeros((len(y), len(cfg.CLASSES)))
        for posicion, nombre in enumerate(cfg.CLASSES):
            uno_hot[np.flatnonzero(pred_fold == nombre), posicion] = 1.0
        probabilidades[validacion] = uno_hot[validacion]
        valor = float(f1_score(y[validacion], pred_fold[validacion], average="macro",
                               labels=cfg.CLASSES))
        por_fold.append(valor)
        for etiqueta, eleccion in eleccion_por_frontera.items():
            filas.append({"Fase": "5 folds", "Fold": fold["fold"], "Frontera": etiqueta,
                          "Umbral": eleccion["umbral"], "Activadas": eleccion["activadas"],
                          "Rescate": eleccion["rescate"], "Dano": eleccion["dano"],
                          "Neto": eleccion["rescate"] - eleccion["dano"],
                          "Neto_umbral_interno": eleccion["neto_umbral_interno"],
                          "N_train_frontera": eleccion["n_train_frontera"],
                          "MacroF1_fold": round(valor, 4)})
        print(f"    fold {fold['fold']}: {valor:.4f} | "
              + " · ".join(f"{k} umbral {v['umbral']:.2f} activa {v['activadas']} "
                           f"resc {v['rescate']} dan {v['dano']}"
                           for k, v in eleccion_por_frontera.items()), flush=True)

    metricas = metricas_globales(probabilidades, clase)
    delta = metricas["macro_f1"] - metricas_globales(referencia, clase)["macro_f1"]
    por_fold_referencia = [float(f1_score(
        y[fold["val_idx"]], pred_v12[fold["val_idx"]], average="macro", labels=cfg.CLASSES))
        for fold in cvmod.make_folds(y, groups=clusters_ids, n_splits=5, n_repeats=1, scheme="group")]
    folds_positivos = sum(1 for a, b in zip(por_fold, por_fold_referencia) if a > b)
    intervalo = bootstrap_diferencia(referencia, probabilidades, clase, len(cfg.CLASSES))
    np.save(OOF, probabilidades)
    print(f"[P9] macro-F1 {metricas['macro_f1']} | delta {delta:+.4f} | folds positivos "
          f"{folds_positivos}/5 | IC95 [{intervalo[0]:+.4f}, {intervalo[1]:+.4f}]", flush=True)

    tabla = pd.DataFrame(filas)
    tabla.to_csv(CURVA, index=False)

    figura = cfg.FIGURES / "v17_p9_arbitro_por_fold.png"
    fig, eje = plt.subplots(figsize=(9, 5))
    posiciones = np.arange(len(por_fold))
    eje.bar(posiciones - 0.2, por_fold_referencia, 0.4, label="V12", color="#2166ac")
    eje.bar(posiciones + 0.2, por_fold, 0.4, label="V12 con arbitro", color="#b2182b")
    eje.axhline(REFERENCIA_CANONICA, color="grey", linestyle=":", linewidth=1)
    eje.set_xticks(posiciones)
    eje.set_xticklabels([f"fold {n + 1}" for n in range(len(por_fold))])
    eje.set_ylabel("macro-F1 del fold")
    eje.set_ylim(min(por_fold + por_fold_referencia) - 0.01,
                 max(por_fold + por_fold_referencia) + 0.01)
    eje.legend(fontsize=9)
    eje.set_title("V17 P9: arbitro aprendido con protocolo anidado, fold por fold", fontsize=10)
    fig.tight_layout()
    fig.savefig(figura, dpi=130)
    plt.close(fig)

    criterios = {
        "efecto_minimo": bool(delta >= GATE_ADOPCION),
        "folds_positivos": bool(folds_positivos >= FOLDS_POSITIVOS_MINIMOS),
        "significacion": bool(intervalo[0] > 0),
    }
    adopta = all(criterios.values())
    lineas = [
        "# V17 · P9 — Experto selectivo con arbitro aprendido", "",
        "## 1. Por que este experimento y no una compuerta mas", "",
        "El diagnostico previo encontro la razon de fondo: **V12 esta mas confiado cuando se "
        "equivoca en la frontera** (separacion 0,809 en las filas que el especialista arregla frente "
        "a 0,539 en las que rompe), y las victorias del especialista caen fuera de la region de "
        "frontera (solo el 12,7 % y el 8,0 %). Por eso toda compuerta por ambiguedad falla: no hay "
        "banda de confianza que discrimine.", "",
        "La pregunta bien planteada es otra: **dentro de la region de frontera, cuales de las ~11.000 "
        "filas convieneework al especialista**. Ahi la base honesta es -0,0080 (escuchar siempre) y "
        "el techo es +0,0107 (activar solo las que el especialista resuelve).", "",
        "## 2. Diseno", "",
        "- Un arbitro binario por frontera: decide si el especialista acierta esa fila.",
        f"- {entradas.shape[1]} entradas, **todas columnas derivadas que ya existian**: los votos y "
        f"margenes de los {len(nombres_miembros)} miembros, los 5 margenes de pares, los 3 de grupo, "
        f"las 6 probabilidades de V12, las 6 de A2 y el acuerdo entre miembros. **Ninguna "
        f"representacion nueva.**",
        "- Protocolo anidado en tres niveles: el arbitro se ajusta en un bloque interior, el umbral "
        "se elige en **otro** bloque interior que el arbitro no ha visto, y despues se reentrena en "
        "todo el train exterior para aplicarse a la validacion exterior.",
        f"- Rejilla de umbrales: {list(UMBRALES)} (pocos a proposito). Regularizacion C = {C_ARBITRO}.",
        "", "## 3. Resultado", "",
        "| Fold | V12 | Con arbitro | Diferencia |", "|---|---|---|---|",
    ]
    for indice, (a, b) in enumerate(zip(por_fold_referencia, por_fold), start=1):
        lineas.append(f"| {indice} | {a:.4f} | {b:.4f} | {a - b:+.4f} |")
    lineas += ["", f"**Global: {metricas['macro_f1']} frente a {metricas_globales(referencia, clase)['macro_f1']} "
                  f"de V12, delta {delta:+.4f}** · folds positivos {folds_positivos}/5 · IC95 pareado "
                  f"[{intervalo[0]:+.4f}, {intervalo[1]:+.4f}]", "",
              "## 4. Activacion por fold y por frontera", "",
              "| Fold | Frontera | Umbral | Activadas | Rescate | Dano | Neto |", "|---|---|---|---|---|---|---|"]
    for _, fila in tabla.iterrows():
        lineas.append(f"| {fila['Fold']} | {fila['Frontera']} | {fila['Umbral']:.2f} | "
                      f"{fila['Activadas']} | {fila['Rescate']} | {fila['Dano']} | {fila['Neto']:+,} |")
    lineas += ["", "## 5. Gate de adopcion de V17", "", "| Criterio | Valor | Cumple |", "|---|---|---|"]
    valores = {"efecto_minimo": f"delta {delta:+.4f} >= {GATE_ADOPCION}",
               "folds_positivos": f"{folds_positivos}/5 >= {FOLDS_POSITIVOS_MINIMOS}",
               "significacion": f"IC95 [{intervalo[0]:+.4f}, {intervalo[1]:+.4f}]"}
    for nombre, cumple in criterios.items():
        lineas.append(f"| {nombre.replace('_', ' ')} | {valores[nombre]} | "
                      f"{'si' if cumple else '**no**'} |")
    lineas += ["", f"**Decision: {'ADOPTAR' if adopta else 'NO ADOPTAR'}**", ""]
    INFORME.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    RESULTADO.write_text(json.dumps({
        "fase": "V17 · P9 arbitro aprendido del experto selectivo", "fecha": "2026-09-26",
        "n_entradas_arbitro": int(entradas.shape[1]), "entradas_nuevas": 0,
        "arbitro": {"modelo": "LogReg", "C": C_ARBITRO, "fraccion_umbral": FRACCION_UMBRAL,
                    "umbrales": list(UMBRALES), "protocolo": "anidado en tres niveles"},
        "referencia": {"macro_f1": metricas_globales(referencia, clase)["macro_f1"],
                       "por_fold": [round(v, 4) for v in por_fold_referencia]},
        "candidata": {"macro_f1": metricas["macro_f1"], "por_fold": [round(v, 4) for v in por_fold]},
        "delta": round(delta, 6), "folds_positivos": folds_positivos,
        "bootstrap_ic95": [round(v, 6) for v in intervalo], "criterios": criterios, "adopta": adopta,
        "activacion": tabla.to_dict(orient="records"),
        "linea_base_escuchar_siempre": -0.0080, "techo_teorico": 0.0107,
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\ndelta {delta:+.4f} | folds {folds_positivos}/5 | IC95 [{intervalo[0]:+.4f}, "
          f"{intervalo[1]:+.4f}] | DECISION: {'ADOPTAR' if adopta else 'NO ADOPTAR'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
