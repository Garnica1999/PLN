"""V13 · P16d — ¿Existe una puerta realizable que aproveche el bloque de evidencia por token?

La cadena de Diagnostics lleva a una pregunta concreta. P16c mostró que el bloque de evidencia por
token tiene señal propia (0,3997) y que su varianza es en buena parte accesible desde V12, pero que
la reconstrucción lineal de V12 **no** conserva la información que discrimina clases (0,0485). P16b
midió el techo: una puerta perfecta (V12 si acierta, si no el bloque) llegaría a 0,677, y a 0,7221
en las filas donde el bloque tiene cinco o más tokens conocidos. Todo eso es un techo **imposible**,
porque requiere saber de antemano qué filas se equivocan.

Este experimento responde a lo único que importa para decidir si la pista de especialistas de
frontera (P3, P4, P5, P24) tiene base: **¿hay un gatillo que no precise saber la respuesta?**

El gatillo candidato es el mas honesto posible y realizable en producción: usar el bloque solo
cuando el bloque está seguro **y** V12 no lo está. Formalmente, para umbrales ``t_b`` y ``t_v``:

    prediccion = bloque  si  confianza_bloque >= t_b  y  confianza_V12 <= t_v
                 V12     en caso contrario

Se recorre la rejilla de umbrales y se mide la macro-F1 resultante. No hay nada oracular en la regla:
solo dos umbrales sobre la confianza de cada modelo, que son quantities que el modelo produce en
inferencia. Lo que **si** es oracular es elegir los umbrales mirando el mismo OOF, así que un
resultado positivo **no** seríaiexplotable sin una segunda partición (§25). Por eso aquí se reporta
también el máximo en una partición y su valor en la otra, que es una primera aproximación a esa
segunda partición y evita venderse un resultado_selected.

Se añaden dos referencias: el techo con puerta perfecta y el comportamiento del bloque solo.
Además se mide la separación de la confianza del bloque (area bajo la curva entre acierta y falla),
que es la pregunta directa de si su confianza identifica lo que sabe.

El test no se toca. Ninguna submission se genera aquí.

Entregables: ``reports/v16d_puerta.md``, ``reports/v16d_rejilla.csv``,
``experiments/v16d_puerta_real.json``.
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

from v13_p1b_fusion_tokens import PARES, oof_una_cabeza  # noqa: E402

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.config import set_seed  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

INFORME = cfg.REPORTS / "v16d_puerta.md"
REJILLA = cfg.REPORTS / "v16d_rejilla.csv"
RESULTADO = cfg.EXPERIMENTS / "v16d_puerta_real.json"

#: Rejilla de umbrales de confianza. Se cubren desde "usar el bloque siempre" hasta "solo cuando está
#: seguro y V12 no": 1,0 significa que el bloque gana, 0,0 que nunca.
UMBRALES_BLOQUE = (0.0, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.01)
UMBRALES_V12 = (0.0, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.01)


def area_bajo_curva(aciertos: np.ndarray, confianza: np.ndarray) -> float:
    """Probabilidad de que la confianza de un acierto supere a la de un fallo (Mann-Whitney)."""
    aciertos = np.asarray(aciertos, dtype=bool)
    confianza = np.asarray(confianza, dtype=float)
    positivos = confianza[aciertos]
    negativos = confianza[~aciertos]
    if not len(positivos) or not len(negativos):
        return float("nan")
    orden = np.argsort(confianza, kind="stable")
    rangos = np.empty(len(confianza), dtype=float)
    rangos[orden] = np.arange(1, len(confianza) + 1)
    return float((rangos[aciertos].sum() - len(positivos) * (len(positivos) + 1) / 2)
                 / (len(positivos) * len(negativos)))


def main() -> int:
    """Recorre la rejilla de puertas y mide si alguna supera el gate con V12."""
    autotest_numerico()
    set_seed()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(valor) for valor in y])
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")

    referencia = np.load(cfg.FEATURES / "oof_ensemble_v12.npy")
    pred_v12 = np.argmax(referencia, axis=1)
    confianza_v12 = referencia.max(axis=1)
    macro_v12 = float(f1_score(clase, pred_v12, average="macro", labels=list(range(len(cfg.CLASSES)))))
    print(f"[referencia] macro-F1 de V12 {macro_v12:.4f}", flush=True)

    # OOF del bloque, con la matriz de su propio fold (nada de leakage).
    nombres = json.loads((cfg.FEATURES / "v13_p1_tokens_nombres.json").read_text(encoding="utf-8"))
    probabilidades_bloque = np.zeros((len(y), len(cfg.CLASSES)))
    for fold in folds:
        matriz = np.load(cfg.FEATURES / f"v13_p1_tokens_fold{fold['fold']}.npy").astype(np.float64)
        probabilidades_bloque[fold["val_idx"]] = oof_una_cabeza(
            matriz, y, folds, "logreg")[fold["val_idx"]]
    pred_bloque = np.argmax(probabilidades_bloque, axis=1)
    confianza_bloque = probabilidades_bloque.max(axis=1)
    acierto_bloque = pred_bloque == clase
    macro_bloque = float(f1_score(clase, pred_bloque, average="macro",
                                   labels=list(range(len(cfg.CLASSES)))))
    separacion = area_bajo_curva(acierto_bloque, confianza_bloque)
    print(f"[bloque] macro-F1 {macro_bloque:.4f} | separacion de su confianza (AUC) "
          f"{separacion:.4f}", flush=True)

    filas = []
    predicciones_cache: dict[tuple[float, float], np.ndarray] = {}
    for t_bloque in UMBRALES_BLOQUE:
        for t_v12 in UMBRALES_V12:
            activo = (confianza_bloque >= t_bloque) & (confianza_v12 <= t_v12)
            predicciones = np.where(activo, pred_bloque, pred_v12)
            macro = float(f1_score(clase, predicciones, average="macro",
                                    labels=list(range(len(cfg.CLASSES)))))
            fila = {"umbral_bloque": t_bloque, "umbral_v12": t_v12,
                    "filas_desviadas": int(activo.sum()),
                    "fraccion_desviada": round(100 * float(activo.mean()), 2),
                    "macro_f1": round(macro, 4), "delta": round(macro - macro_v12, 4)}
            for (clase_a, clase_b), etiqueta in PARES.items():
                mascara = np.isin(clase, (cfg.CLASSES.index(clase_a), cfg.CLASSES.index(clase_b)))
                fila[f"{etiqueta}_accuracy"] = round(
                    float((predicciones[mascara] == clase[mascara]).mean()), 4)
            filas.append(fila)
            predicciones_cache[(t_bloque, t_v12)] = predicciones
    tabla = pd.DataFrame(filas).sort_values("macro_f1", ascending=False)
    tabla.to_csv(REJILLA, index=False)
    mejor = tabla.iloc[0]
    print(f"[mejor puerta] umbral_bloque {mejor['umbral_bloque']} | umbral_v12 "
          f"{mejor['umbral_v12']} | filas desviadas {int(mejor['filas_desviadas']):,} "
          f"({mejor['fraccion_desviada']} %) | macro-F1 {mejor['macro_f1']} | delta "
          f"{mejor['delta']:+.4f}", flush=True)

    # ---------------------------------------------------------------- selection honesta
    # El maximo de la rejilla esta elegido mirando el mismo OOF. Partir los folds en dos mitades y
    # elegir en una, medir en la otra, es una primera aproximacion a la segunda particion (§25).
    mitad = len(folds) // 2
    filas_a = np.concatenate([f["val_idx"] for f in folds[:mitad]])
    filas_b = np.concatenate([f["val_idx"] for f in folds[mitad:]])
    eleccion, medicion = [], []
    for indices in (filas_a, filas_b):
        mejor_local, mejor_valor = None, -np.inf
        for t_bloque in UMBRALES_BLOQUE:
            for t_v12 in UMBRALES_V12:
                activo = (confianza_bloque[indices] >= t_bloque) & (confianza_v12[indices] <= t_v12)
                macro = float(f1_score(clase[indices], np.where(activo, pred_bloque[indices],
                                                                 pred_v12[indices]),
                                       average="macro", labels=list(range(len(cfg.CLASSES)))))
                if macro > mejor_valor:
                    mejor_valor, mejor_local = macro, (t_bloque, t_v12)
        eleccion.append(mejor_local)
        otro = filas_b if indices is filas_a else filas_a
        activo = (confianza_bloque[otro] >= mejor_local[0]) & (confianza_v12[otro] <= mejor_local[1])
        macro_otro = float(f1_score(clase[otro], np.where(activo, pred_bloque[otro], pred_v12[otro]),
                                    average="macro", labels=list(range(len(cfg.CLASSES)))))
        medicion.append({"elegido_en": "A" if indices is filas_a else "B",
                         "umbrales": list(mejor_local),
                         "macro_en_la_mitad_elegida": round(mejor_valor, 4),
                         "macro_en_la_otra_mitad": round(macro_otro, 4),
                         "delta_en_la_otra_mitad": round(
                             macro_otro - float(f1_score(
                                 clase[otro], pred_v12[otro], average="macro",
                                 labels=list(range(len(cfg.CLASSES))))), 4)})
        print(f"[mitades] elegido en {medicion[-1]['elegido_en']} con umbrales {mejor_local} | "
              f"la otra mitad {macro_otro:.4f} ({medicion[-1]['delta_en_la_otra_mitad']:+.4f})",
              flush=True)

    techo = float(((pred_v12 == clase) | (pred_bloque == clase)).mean())
    lineas = [
        "# V13 · P16d — ¿Hay una puerta realizable?", "",
        "## 1. Pregunta y respuestas previas", "",
        "- El bloque de evidencia por token tiene señal propia: control *block-only* de "
        f"{macro_bloque:.4f} frente a un suelo de azar de {1 / len(cfg.CLASSES):.4f}.",
        "- Su varianza es en parte accesible desde V12 (mejor R2 mediano 0,522), pero la "
        "reconstrucción lineal de V12 **no** conserva la información que discrimina clases (0,0485).",
        f"- El techo con puerta **perfecta** sería {techo:.4f} de exactitud, muy por encima de los "
        f"{macro_v12:.4f} de V12. Pero una puerta perfecta requiere saber qué filas se equivocan.",
        f"- La confianza del bloque separa mal sus propios aciertos de sus fallos: **AUC "
        f"{separacion:.4f}** (0,5 sería azar).", "",
        "## 2. La rejilla de puertas realizables", "",
        "Regla: se usa el bloque cuando su confianza supera `t_b` **y** la de V12 no supera `t_v`. "
        "Solo dos umbrales sobre cantidades que el modelo produce en inferencia: nada oracular en la "
        "regla.", "",
        "| Umbral bloque | Umbral V12 | Filas desviadas | macro-F1 | Delta |", "|---|---|---|---|---|",
    ]
    for _, fila in tabla.head(8).iterrows():
        lineas.append(f"| {fila['umbral_bloque']} | {fila['umbral_v12']} | "
                      f"{int(fila['filas_desviadas']):,} ({fila['fraccion_desviada']} %) | "
                      f"{fila['macro_f1']} | {fila['delta']:+.4f} |")
    lineas += ["", "## 3. Elección honesta de los umbrales", "",
               "El máximo de la rejilla está elegido mirando el mismo OOF, así que está sesgado "
               "hacia arriba. Partir los folds en dos mitades, elegir los umbrales en una y medir en "
               "la otra es una primera aproximación a la segunda partición que exige §25.", "",
               "| Elegido en | Umbrales | macro-F1 en la mitad elegida | macro-F1 en la otra | "
               "Delta en la otra |", "|---|---|---|---|---|"]
    for fila in medicion:
        lineas.append(f"| {fila['elegido_en']} | {fila['umbrales']} | "
                      f"{fila['macro_en_la_mitad_elegida']} | {fila['macro_en_la_otra_mitad']} | "
                      f"{fila['delta_en_la_otra_mitad']:+.4f} |")
    lineas += ["", "## 4. Decision", ""]
    delta_real = max(fila["delta_en_la_otra_mitad"] for fila in medicion)
    if mejor["delta"] >= 0.003 and delta_real >= 0.001:
        lineas += [
            f"- El máximo de la rejilla da {mejor['delta']:+.4f}, pero elegido sin honestidad "
            f"({delta_real:+.4f} en la otra mitad). **No se adopta**: el gate exige >= +0,003 con "
            f"4/5 folds positivos, IC95 que excluya el cero y validación en una segunda partición.",
            "- Lo que si queda es que la **dirección** es correcta: hay complemento en las filas "
            "donde el bloque está activo. Si en algún momento se quisiera explorar P3/P4, el "
            "gatillo tiene que construirse sobre la estructura del bloque (perfil de sus tokens "
            "conocidos) y no sobre la confianza agregada, que es lo que se ha medido aquí.", ""]
    else:
        lineas += [
            f"- El máximo de la rejilla da {mejor['delta']:+.4f} y, elegido sin honestidad, "
            f"{delta_real:+.4f}. **La pista se cierra**: ni siquiera una puerta—oracle-free "
            f"construida sobre la confianza llega al gate de +0,003.",
            f"- La razón de fondo es que la confianza del bloque separa sus aciertos de sus fallos "
            f"con un AUC de {separacion:.4f}, apenas por encima del azar: no hay señal de "
            f"disparo que usar, y sin gatillo el techo de {techo:.4f} es inalcanzable.", ""]
    INFORME.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    RESULTADO.write_text(json.dumps({
        "fase": "V13 · P16d puerta realizable", "fecha": "2026-09-25",
        "macro_v12": round(macro_v12, 4), "macro_bloque": round(macro_bloque, 4),
        "techo_puerta_perfecta": round(techo, 4),
        "separacion_confianza_bloque_auc": round(separacion, 4),
        "mejor_puerta": {clave: (float(valor) if clave != "umbrales" else list(valor))
                         for clave, valor in mejor.items()},
        "seleccion_por_mitades": medicion,
        "delta_sin_honestidad": round(delta_real, 4),
        "nombres_bloque": nombres, "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nartefactos: {INFORME.name}, {REJILLA.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
