"""V19 · B1-verificacion — bateria de falsacion del +0,0074 antes de creerselo.

El disparo B1 dio un delta de **+0,0074** con 5 de 5 folds e IC95 que excluye el cero, y el arbol
**aislado** (0,6072) ya supera al champion V12 (0,6005). Ese tamano de efecto esta por encima de todo
el historial del proyecto, y hay una explicacion alternativa Mucho mas simple que una mejora real:

> **Las 73 columnas de `age_supervised_features.npy` son predicciones de 72 modelos supervisados.**
> Un arbol puede combinarlas de un modo que un promedio lineal no, y ademas puede
> ajustarse a ese bloque mejor de lo que lo hace el promedio. Si toda la ventaja viene de ahi, no es un descubrimiento sobre
> importancia de variables, es que el arbol exprimiese un bloque de predicciones que V12 se limita a promediar.

Por eso este script **no mide nada nuevo**: parte del OOF ya calculado y comprueba si el efecto
sobrevive a las condiciones que separan "arbol mejor" de "arbol exprimiendo las columnas
supervisadas".

**Las comprobaciones, y que falsarian cada una:**
1. **Arbol sin las 73 columnas supervisadas.** Si la ventaja cae, el efecto era el bloque supervisado.
2. **Arbol solo con las 78 columnas de los miembros.** Es el caso mas puro: ahi no hay nada
   supervisado que el arbol pueda exprimir.
3. **Individual por miembro.** Si el arbol solo bate a MLP y LogReg por mucho, hay que entender por
   que V12 promedia algo peor que una de sus partes.
4. **Segunda particion de 4 folds.** Es el control de seleccion del propio proyecto (D-042): si el
   efecto aparece solo en la particion de 5, es ruido de seleccion.
5. **Control de filas barajadas ya calculado**, que se contrasta con la misma regla.

**Lo que este script NO va a hacer**: elegir el mejor bloque y el resultado. Si el efecto depende
de las columnas supervisadas, el veredicto sera que **no es un challenger adoptable como nueva
informacion**, porque el bloque ya estaba dentro de las 279 de V12 y lo que cambia es como se combina,
no que se sabe.

Entregables: ``experiments/v19_b1_verificacion.json`` y ``reports/v19_b1_verificacion.md``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v13_p1b_fusion_tokens import PARES, metricas_frontera  # noqa: E402

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.config import set_seed  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from v19_b1_boosting import HIERBA, matriz_279, oof_arbol  # noqa: E402

INFORME = cfg.REPORTS / "v19_b1_verificacion.md"
RESULTADO = cfg.EXPERIMENTS / "v19_b1_verificacion.json"
ETIQUETAS = list(range(len(cfg.CLASSES)))


def macro(clase: np.ndarray, prob: np.ndarray) -> float:
    """macro-F1 pooled de seis clases."""
    return float(f1_score(clase, np.argmax(prob, axis=1), average="macro", labels=ETIQUETAS))


def oof_por_fold(matriz: np.ndarray, y: np.ndarray, clase: np.ndarray, folds: list[dict],
                 etiqueta: str) -> dict:
    """Ajusta el arbol pre-registrado, calcula el pooled, los folds y las fronteras."""
    prob = oof_arbol(matriz, y, folds, fila_barajada=False, semilla=cfg.SEED)
    por_fold = [macro(clase[fold["val_idx"]], prob[fold["val_idx"]]) for fold in folds]
    fronteras = metricas_frontera(prob, clase, PARES)
    return {"bloque": etiqueta, "columnas": int(matriz.shape[1]), "macro_f1": round(macro(clase, prob), 7),
            "media_folds": round(float(np.mean(por_fold)), 6),
            "por_fold": [round(v, 4) for v in por_fold],
            "frontera_1": fronteras["18-24|25-34"]["accuracy"],
            "frontera_2": fronteras["25-34|35-49"]["accuracy"]}


def main() -> int:
    """Ejecuta la bateria de falsacion del resultado de B1."""
    autotest_numerico()
    set_seed()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(valor) for valor in y])
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds5 = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")
    folds4 = cvmod.make_folds(y, groups=clusters, n_splits=4, n_repeats=1, scheme="group")
    matriz = matriz_279()

    # Las ultimas 73 columnas de la matriz reconstruida son las supervisadas, porque
    # `build_features()` concatena miembros, densas, estructuradas y **al final** las supervisadas.
    from phase2_ponderacion import DENSE_FEATURES  # noqa: E402
    import phase2_ponderacion as ph2

    supervised = np.load(cfg.FEATURES / "age_supervised_features.npy").shape[1]
    n_estructuradas = pd.read_parquet(cfg.FEATURES / "age_structured_features.parquet").shape[1]
    n_densas = len(DENSE_FEATURES)
    n_miembros = 78
    # La matriz se arma como [base de 273] + [6 de NB-SVM de palabra], y dentro de la base el bloque
    # supervisado va **el ultimo**. Asi que el corte supervised no es `columnas - supervised`, sino
    # `273 - supervised`: con el corte equivocado la prueba de falsacion dejaba 6 columnas
    # supervisadas dentro del bloque "sin supervisadas" y sacaba 6 de NB-SVM fuera.
    n_avanzado = np.load(cfg.FEATURES / "oof_ADV07_nbsvm_word.npy").shape[1]
    ancho_base = matriz.shape[1] - n_avanzado
    corte = ancho_base - supervised
    suma = n_miembros + n_densas + n_estructuradas + supervised
    print(f"[composicion] miembros {n_miembros} + densas {n_densas} + estructuradas {n_estructuradas} "
          f"+ supervisadas {supervised} = {suma} base, mas {n_avanzado} de NB-SVM de palabra = "
          f"{matriz.shape[1]} | el bloque supervisado ocupa las columnas {corte} a {corte + supervised - 1}",
          flush=True)
    assert suma == ancho_base, f"la base suma {suma} y el corte asume {ancho_base}"

    # ---- 1) comprobaciones de composicion: el bloque final es exactamente el supervisado
    bloque_supervisado = np.load(cfg.FEATURES / "age_supervised_features.npy").astype(np.float32)
    dif_supervisado = float(np.abs(matriz[:, corte:corte + supervised]
                                   - bloque_supervisado).max())
    print(f"[composicion] las ultimas {supervised} columnas coinciden con el bloque supervisado: "
          f"diferencia maxima {dif_supervisado:.3e}", flush=True)

    # ---- 2) battery de bloques
    bloques = [
        oof_por_fold(matriz, y, clase, folds5, "las 279 completas"),
        oof_por_fold(matriz[:, :corte], y, clase, folds5, f"sin las {supervised} supervisadas"),
        oof_por_fold(matriz[:, :n_miembros], y, clase, folds5, f"solo las {n_miembros} de miembros"),
        oof_por_fold(matriz[:, corte:corte + supervised], y, clase, folds5,
                     f"solo las {supervised} supervisadas"),
    ]
    for bloque in bloques:
        print(f"[5 folds] {bloque['bloque']:<28} {bloque['columnas']:>4} cols | {bloque['macro_f1']} | "
              f"folds {bloque['media_folds']} | fronteras {bloque['frontera_1']} y "
              f"{bloque['frontera_2']}", flush=True)

    # ---- 3) segunda particion de 4 folds, el control de seleccion del proyecto (D-042)
    bloque4 = oof_por_fold(matriz, y, clase, folds4, "las 279 completas, segunda particion")
    print(f"[4 folds] {bloque4['bloque']} | {bloque4['macro_f1']} | folds {bloque4['media_folds']} | "
          f"fronteras {bloque4['frontera_1']} y {bloque4['frontera_2']}", flush=True)

    # ---- 4) los miembros por separado, para entender por que el arbol bate al promedio
    prob_logreg = np.load(cfg.FEATURES / "v19_b1_oof_logreg.npy")
    prob_mlp = np.load(cfg.FEATURES / "v19_b1_oof_mlp.npy")
    individuales = {
        "LogReg (cabeza de V12)": round(macro(clase, prob_logreg), 7),
        "MLP(64) (cabeza de V12)": round(macro(clase, prob_mlp), 7),
        "promedio de las dos (V12 reconstruido)": round(macro(clase, 0.5 * prob_logreg + 0.5 * prob_mlp), 7),
        "arbol solo": bloques[0]["macro_f1"],
    }
    for nombre, valor in individuales.items():
        print(f"[individual] {nombre:<40} {valor}", flush=True)

    # ---- 5) veredicto
    completa = bloques[0]["macro_f1"]
    sin_supervisado = bloques[1]["macro_f1"]
    solo_miembros = bloques[2]["macro_f1"]
    ventaja = completa - sin_supervisado
    referencia = 0.6005017473
    veredicto = {
        "ventaja_de_las_73_supervisadas": round(ventaja, 7),
        "el_efecto_sobrevive_sin_supervisadas": bool(sin_supervisado > referencia + 0.003),
        "segunda_particion_supera_el_gate": bool(bloque4["macro_f1"] > referencia + 0.003),
        "arbol_aislado_supera_el_promedio_de_cabezas":
            bool(completa > individuales["promedio de las dos (V12 reconstruido)"]),
    }
    print(f"\n[veredicto] ventaja de las 73 supervisadas {ventaja:+.7f} | sobrevive sin ellas "
          f"{veredicto['el_efecto_sobrevive_sin_supervisadas']} | 2a particion "
          f"{veredicto['segunda_particion_supera_el_gate']} | arbol bate al promedio de cabezas "
          f"{veredicto['arbol_aislado_supera_el_promedio_de_cabezas']}", flush=True)

    INFORME.write_text("\n".join([
        "# V19 · B1 — Verificacion del +0,0074: bateria de falsacion", "",
        "B1 dio **+0,0074** con 5 de 5 folds e IC95 que excluia el cero, y el arbol **aislado** ("
        f"{bloques[0]['macro_f1']}) ya superaba al champion ({referencia:.7f}). Ese tamano de efecto "
        "exige una comprobacion antes de creerlo, y la explicacion alternativa mas simple no es una "
        "mejora real: **las 73 columnas finales son predicciones de 72 modelos supervisados**, que un "
        "arbol puede combinar de un modo que un promedio lineal no.", "",
        "## 1. Composicion de la matriz de 279", "",
        f"- miembros {n_miembros} + densas {n_densas} + estructuradas {n_estructuradas} + supervisadas "
        f"{supervised} = {n_miembros + n_densas + n_estructuradas + supervised}",
        f"- el bloque supervisado ocupa las columnas {corte} a {corte + supervised - 1}, y coincide con "
        f"`age_supervised_features.npy` con diferencia maxima {dif_supervisado:.3e}", "",
        "## 2. Que sobrevive al quitar bloques, en la particion de 5 folds", "",
        "| Bloque | Columnas | macro-F1 | Media folds | 18-24\\|25-34 | 25-34\\|35-49 |",
        "|---|---|---|---|---|---|",
        *[f"| {b['bloque']} | {b['columnas']} | {b['macro_f1']} | {b['media_folds']} | "
          f"{b['frontera_1']} | {b['frontera_2']} |" for b in bloques],
        f"| **V12 (champion)** | 279 | {referencia:.7f} | - | 0.4963 | 0.4974 |", "",
        f"**La ventaja de las 73 columnas supervisadas es de {ventaja:+.7f}.**", "",
        "## 3. Segunda particion de 4 folds (control de seleccion, D-042)", "",
        f"- macro-F1 **{bloque4['macro_f1']}**, media de folds {bloque4['media_folds']}, fronteras "
        f"{bloque4['frontera_1']} y {bloque4['frontera_2']}.",
        f"- Supera el gate de +0,003 sobre V12: **{veredicto['segunda_particion_supera_el_gate']}**.", "",
        "## 4. Los miembros por separado", "",
        "| Modelo | macro-F1 |", "|---|---|",
        *[f"| {n} | {v} |" for n, v in individuales.items()], "",
        "## 5. Veredicto de la verificacion", "",
        "| Comprobacion | Resultado |", "|---|---|",
        f"| el efecto sobrevive sin las 73 columnas supervisadas | "
        f"**{veredicto['el_efecto_sobrevive_sin_supervisadas']}** |",
        f"| la segunda particion de 4 folds supera el gate | "
        f"**{veredicto['segunda_particion_supera_el_gate']}** |",
        f"| el arbol aislado supera al promedio de las dos cabezas | "
        f"**{veredicto['arbol_aislado_supera_el_promedio_de_cabezas']}** |", "",
    ]), encoding="utf-8")

    RESULTADO.write_text(json.dumps({
        "fase": "V19 · B1 verificacion", "fecha": "2026-09-26",
        "referencia_v12": referencia,
        "composicion": {"miembros": n_miembros, "densas": n_densas, "estructuradas": n_estructuradas,
                        "supervisadas": supervised, "corte": corte,
                        "coincidencia_del_bloque_supervisado": dif_supervisado},
        "bloques_5_folds": bloques, "segunda_particion_4_folds": bloque4,
        "individuales": individuales, "veredicto": veredicto,
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"segundos: {time.time() - inicio:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
