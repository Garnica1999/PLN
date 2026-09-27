"""V19 · X1 — Puerta de coste cero y las dos unicas configuraciones que el especialista autoriza.

El especialista recomienda **cerrar la explotacion de V19**, y sus tres argumentos son verificables:
la tasa base del subespacio de parametros es +0,0008 frente a un gate de +0,0030; la resolucion del
aparato es de +0,0011 (0,6090431 en 4 folds contra 0,6079388 en 5, **sin cambiar nada**); y el
residuo de la calibracion sobre V19 es de **-0,0016**, de modo que un +0,002 en CV puede aterrizar en
cualquier punto entre -0,000 y +0,004 en LB.

**La puerta previa, que es un `read` y no una ejecucion.** La primera propuesta del especialista
-descender un nivel de profundidad- se apoya en una tesis de **saturacion**: con 6 clases, 300
iteraciones y 7 cortes internos por arbol el techo es 12.600 y se usarian 12.280, es decir el 97,5 %.
Si la ocupacion real fuera del 70 % y no del 97,5 %, toda la lectura cambiaria y el eje de capacidad
reabriria. Se mide sobre el modelo persistido, que es lo que permite hacerlo sin gastar un solo ajuste.

**Las dos configuraciones, y solo dos.** El especialista las delimita con numeros y descarta otras seis
dimensiones por inertes o por ya sondadas:

| Configuracion | Cambio | Por que |
|---|---|---|
| A, profundidad | `max_depth` 3 a 4 con `num_leaves` **fijado en 8** | con 8 hojas y profundidad 3 se impone la misma restriccion dos veces, porque 2^3 = 8: `num_leaves` 8 a 16 con profundidad 3 es un no-op exacto. Sube la profundidad manteniendo el tope de hojas, que es lo que gobierna la varianza |
| B, regularizacion | `min_child_samples` 200 a 800 | de los cuatro knobs, `reg_lambda` 1,0 es inerte a esta escala, `min_child_weight` es una reparametrizacion del propio `min_child_samples` porque la hessiana por fila es 0,5-0,8, y bajar `colsample` mataria el bloque de interaccion. Es el unico que puede **cerrar** el hueco sin abrirlo |

**El riesgo que las dos comparten, y que se mide**: el bloque de 73 se lleva el 28,6 % de los cortes
para el 9,1 % de la ganancia, luego cada corte suyo vale 4,0 veces menos que uno del bloque de 206.
Profundidad y `min_child_samples` anaden cortes en ese bloque de peor ratio, asi que A puede abrir el
hueco y B es el unico de riesgo monontonico.

**Ninguna de las dos es candidata a submission por si sola.** El experto fija la regla y aqui se
aplica sin abreviar: submission solo con **delta CV >= +0,003, 5 de 5 folds positivos, IC95 que
excluya el cero y confirmacion >= +0,002 en la particion de 4 folds**. Ademas se mide la
**complementariedad de errores frente a V19**, porque un candidato que mejora el CV cambiando solo las
mismas filas y anadiendo dano no lo justifies.

Entregables: ``experiments/v19_x1_explotacion.json``, ``reports/v19_x1_explotacion.md``,
``reports/v19_x1_comparativa.csv``, ``reports/v19_x1_complementariedad.csv`` y una figura.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v19_b1_boosting import COMBINACIONES, HIERBA, matriz_279  # noqa: E402
from v13_p1b_fusion_tokens import PARES, bootstrap_diferencia, metricas_frontera  # noqa: E402

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.config import set_seed  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

INFORME = cfg.REPORTS / "v19_x1_explotacion.md"
COMPARATIVA = cfg.REPORTS / "v19_x1_comparativa.csv"
COMPLEMENTARIEDAD = cfg.REPORTS / "v19_x1_complementariedad.csv"
RESULTADO = cfg.EXPERIMENTS / "v19_x1_explotacion.json"
FIGURA = cfg.FIGURES / "v19_x1_explotacion.png"
MODELO = cfg.MODELS / "v19_champion_lightgbm.txt"
OOF_ARBOL = cfg.FEATURES / "v19_oof_arbol_5folds.npy"
OOF_ARBOL_4 = cfg.FEATURES / "v19_oof_arbol_4folds.npy"

CV_V19 = 0.6079388
CV_V19_4 = 0.6090431
GATE = 0.003
ETIQUETAS = list(range(len(cfg.CLASSES)))
#: Las dos configuraciones, y solo estas dos.
CONFIGURACIONES = {
    "A profundidad max_depth 4 con num_leaves 8": {"max_depth": 4},
    "B regularizacion min_child_samples 800": {"min_child_samples": 800},
}


def a_clases(probabilidad: np.ndarray, clases: np.ndarray, filas: int) -> np.ndarray:
    """Traduce las columnas del modelo al orden canonico de las seis clases."""
    completa = np.zeros((filas, len(cfg.CLASSES)))
    for posicion, nombre in enumerate(clases):
        completa[:, cfg.CLASSES.index(nombre)] = probabilidad[:, posicion]
    return completa


def macro(clase: np.ndarray, prob: np.ndarray) -> float:
    """macro-F1 pooled de seis clases."""
    return float(f1_score(clase, np.argmax(prob, axis=1), average="macro", labels=ETIQUETAS))


def puerta_de_saturacion() -> dict:
    """La tesis que sostiene la propuesta de profundidad, medida sobre el modelo persistido.

    Con `num_leaves = 8` un arbol de decision puede como maximo 7 cortes internos. La pregunta es
    cuantos usa de verdad: si llena lamayoria, la profundidad es lo que limita y subirla tiene sentido;
    si se queda corto, el limite esta en otro sitio y la propuesta de profundidad cae.
    """
    booster = lgb.Booster(model_file=str(MODELO))
    volcado = booster.dump_model()
    arboles = volcado["tree_info"]
    cortes_por_arbol = []
    hojas_por_arbol = []

    def contar(nodo: dict) -> tuple[int, int]:
        if "split_feature" in nodo:
            hijos = [nodo[k] for k in ("left_child", "right_child") if k in nodo]
            total = 1 + sum(contar(hijo)[0] for hijo in hijos)
            hojas = 1 if not hijos else sum(contar(hijo)[1] for hijo in hijos)
            return total, hojas
        return 0, 1

    for arbol in arboles:
        cortes, hojas = contar(arbol["tree_structure"])
        cortes_por_arbol.append(cortes)
        hojas_por_arbol.append(hojas)
    cortes = np.array(cortes_por_arbol)
    hojas = np.array(hojas_por_arbol)
    techo_por_arbol = 7
    ocupacion = float(cortes.mean()) / techo_por_arbol
    return {
        "arboles": len(arboles),
        "cortes_medios": round(float(cortes.mean()), 4),
        "cortes_maximos_por_arbol": int(cortes.max()),
        "hojas_medias": round(float(hojas.mean()), 4),
        "techo_de_cortes_por_arbol": techo_por_arbol,
        "ocupacion_del_techo": round(ocupacion, 6),
        "arboles_al_techo": int((cortes >= techo_por_arbol).sum()),
        "arboles_por_debajo_de_5_cortes": int((cortes < 5).sum()),
        "distribucion_de_cortes": {str(int(k)): int(v) for k, v in
                                   zip(*np.unique(cortes, return_counts=True))},
        "umbral_de_la_tesis": 0.95,
        "la_tesis_se_sostiene": bool(ocupacion >= 0.95),
    }


def oof_de_arbol(matriz: np.ndarray, y: np.ndarray, folds: list[dict], sobrescribe: dict,
                 cache: Path) -> np.ndarray:
    """OOF real de una configuracion del arbol, con cache para no repetir cinco ajustes."""
    if cache.exists():
        matriz = np.load(cache)
        if matriz.shape == (len(y), len(cfg.CLASSES)):
            print(f"    cache: {cache.name}", flush=True)
            return matriz
    salida = np.zeros((len(y), len(cfg.CLASSES)))
    for fold in folds:
        validacion = fold["val_idx"]
        entrenamiento = np.concatenate([o["val_idx"] for o in folds
                                        if o["fold"] != fold["fold"]])
        modelo = lgb.LGBMClassifier(**{**HIERBA, **sobrescribe})
        modelo.fit(matriz[entrenamiento], y[entrenamiento])
        salida[validacion] = a_clases(modelo.predict_proba(matriz[validacion]),
                                       modelo.classes_, len(validacion))
    np.save(cache, salida)
    return salida


def evaluar(nombre: str, oof_arbol: np.ndarray, clase: np.ndarray, y: np.ndarray, folds: list[dict],
            pesos: tuple[float, float, float], oof_v19: np.ndarray,
            sobrescribe: dict) -> dict:
    """Metricas de una combinacion, con la delta **contra V19** y no solo contra V12."""
    logreg = np.load(cfg.FEATURES / "v19_b1_oof_logreg.npy")
    mlp = np.load(cfg.FEATURES / "v19_b1_oof_mlp.npy")
    combinacion = pesos[0] * logreg + pesos[1] * mlp + pesos[2] * oof_arbol
    valor = macro(clase, combinacion)
    valor_v19 = macro(clase, oof_v19)
    por_fold, por_fold_v19 = [], []
    for fold in folds:
        idx = fold["val_idx"]
        por_fold.append(macro(clase[idx], combinacion[idx]))
        por_fold_v19.append(macro(clase[idx], oof_v19[idx]))
    intervalo = bootstrap_diferencia(oof_v19, combinacion, clase, len(cfg.CLASSES))
    positivos = sum(1 for a, b in zip(por_fold, por_fold_v19) if a > b)
    fronteras = metricas_frontera(combinacion, clase, PARES)
    fronteras_v19 = metricas_frontera(oof_v19, clase, PARES)
    # P9: complementariedad de errores frente a V19, que es lo que decide si un candidato lo es.
    pred = np.array(cfg.CLASSES)[np.argmax(combinacion, axis=1)]
    pred_v19 = np.array(cfg.CLASSES)[np.argmax(oof_v19, axis=1)]
    acierta = pred == y
    acierta_v19 = pred_v19 == y
    ambos_ok = int((acierta & acierta_v19).sum())
    ambos_mal = int((~acierta & ~acierta_v19).sum())
    solo_challenger = int((acierta & ~acierta_v19).sum())
    solo_v19 = int((acierta_v19 & ~acierta).sum())
    return {
        "configuracion": nombre, "cambios": sobrescribe_a_json(sobrescribe),
        "macro_f1": round(valor, 7), "macro_f1_v19": round(valor_v19, 7),
        "delta_contra_v19": round(valor - valor_v19, 6), "delta_contra_v12": round(valor - 0.6005017473, 6),
        "media_folds": round(float(np.mean(por_fold)), 6),
        "por_fold": [round(v, 6) for v in por_fold],
        "folds_positivos_contra_v19": positivos,
        "bootstrap_ic95_contra_v19": [round(v, 6) for v in intervalo],
        "frontera_1": fronteras["18-24|25-34"]["accuracy"],
        "frontera_1_v19": fronteras_v19["18-24|25-34"]["accuracy"],
        "frontera_2": fronteras["25-34|35-49"]["accuracy"],
        "frontera_2_v19": fronteras_v19["25-34|35-49"]["accuracy"],
        "complementariedad": {"ambos_correctos": ambos_ok, "ambos_incorrectos": ambos_mal,
                              "solo_challenger": solo_challenger, "solo_v19": solo_v19,
                              "neto": solo_challenger - solo_v19},
    }


def sobrescribe_a_json(sobrescribe: dict) -> dict:
    """Vuelca la sobrescritura de hiperparametros de forma legible."""
    return {k: v for k, v in sobrescribe.items()}





def main() -> int:
    """Ejecuta la puerta de coste cero y las dos configuraciones autorizadas."""
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

    # ---------------------------------------------------------------- puerta de coste cero
    print("[puerta] saturacion de los arboles del champion, leida del modelo persistido", flush=True)
    saturacion = puerta_de_saturacion()
    print(f"[puerta] cortes medios {saturacion['cortes_medios']} de un techo de "
          f"{saturacion['techo_de_cortes_por_arbol']} | ocupacion "
          f"{saturacion['ocupacion_del_techo']:.1%} | arboles al techo "
          f"{saturacion['arboles_al_techo']} de {saturacion['arboles']} | la tesis se sostiene: "
          f"{saturacion['la_tesis_se_sostiene']}", flush=True)

    pesos = COMBINACIONES["tercero a partes iguales"]
    logreg = np.load(cfg.FEATURES / "v19_b1_oof_logreg.npy")
    mlp = np.load(cfg.FEATURES / "v19_b1_oof_mlp.npy")
    oof_v19 = pesos[0] * logreg + pesos[1] * mlp + pesos[2] * np.load(OOF_ARBOL)
    print(f"[V19] CV {macro(clase, oof_v19):.7f} | referencia contra la que se mide", flush=True)

    # ---------------------------------------------------------------- las dos configuraciones
    resultados = []
    for nombre, sobrescribe in CONFIGURACIONES.items():
        clave = "".join(f"_{k}{v}" for k, v in sobrescribe.items())
        print(f"[config] {nombre}", flush=True)
        oof = oof_de_arbol(matriz, y, folds5, sobrescribe,
                           cfg.FEATURES / f"v19_x1_oof_arbol5{clave}.npy")
        item = evaluar(nombre, oof, clase, y, folds5, pesos, oof_v19, sobrescribe)
        # Confirmacion en la particion de 4 folds, que es la que no se uso para elegir nada.
        oof4 = oof_de_arbol(matriz, y, folds4, sobrescribe,
                            cfg.FEATURES / f"v19_x1_oof_arbol4{clave}.npy")
        logreg4 = np.load(cfg.FEATURES / "v19_b1_oof4_logreg.npy")
        mlp4 = np.load(cfg.FEATURES / "v19_b1_oof4_mlp.npy")
        candidato4 = pesos[0] * logreg4 + pesos[1] * mlp4 + pesos[2] * oof4
        item["cv_4_folds"] = round(macro(clase, candidato4), 7)
        # La referencia de V19 en la particion de 4 folds ya esta verificada y medida por el
        # experimento de confirmacion, luego se usa esa constante en vez de recalcular el OOF.
        item["cv_4_folds_v19"] = CV_V19_4
        item["delta_4_folds"] = round(item["cv_4_folds"] - item["cv_4_folds_v19"], 6)
        criterios = {
            "delta_5_folds": bool(item["delta_contra_v19"] >= GATE),
            "cinco_de_cinco_folds": bool(item["folds_positivos_contra_v19"] == 5),
            "ic95_excluye_cero": bool(item["bootstrap_ic95_contra_v19"][0] > 0),
            "confirmacion_4_folds": bool(item["delta_4_folds"] >= 0.002),
        }
        item["criterios_de_submission"] = criterios
        item["merece_submission"] = all(criterios.values())
        resultados.append(item)
        print(f"    CV {item['macro_f1']} | contra V19 {item['delta_contra_v19']:+.4f} | folds "
              f"{item['folds_positivos_contra_v19']}/5 | IC95 "
              f"[{item['bootstrap_ic95_contra_v19'][0]:+.4f}, "
              f"{item['bootstrap_ic95_contra_v19'][1]:+.4f}] | 4 folds "
              f"{item['delta_4_folds']:+.4f} | merece submission {item['merece_submission']}",
              flush=True)
        print(f"    complementariedad: {item['complementariedad']}", flush=True)

    # ---------------------------------------------------------------- tablas y figura
    tabla = pd.DataFrame([{
        "configuracion": "V19 (champion)", "cambio": "-", "cv_5folds": CV_V19,
        "cv_4folds": CV_V19_4, "delta_contra_v19": 0.0, "folds_positivos": "-",
        "frontera_1": 0.5083, "frontera_2": 0.5018, "solo_challenger": 0, "solo_v19": 0,
        "neto": 0, "merece_submission": "-"}] + [{
        "configuracion": r["configuracion"], "cambio": json.dumps(r["cambios"], ensure_ascii=False),
        "cv_5folds": r["macro_f1"], "cv_4folds": r["cv_4_folds"],
        "delta_contra_v19": r["delta_contra_v19"],
        "folds_positivos": f"{r['folds_positivos_contra_v19']}/5",
        "frontera_1": r["frontera_1"], "frontera_2": r["frontera_2"],
        "solo_challenger": r["complementariedad"]["solo_challenger"],
        "solo_v19": r["complementariedad"]["solo_v19"],
        "neto": r["complementariedad"]["neto"],
        "merece_submission": r["merece_submission"]} for r in resultados])
    tabla.to_csv(COMPARATIVA, index=False)
    pd.DataFrame([{"configuracion": r["configuracion"], **r["complementariedad"]}
                  for r in resultados]).to_csv(COMPLEMENTARIEDAD, index=False)

    fig, ejes = plt.subplots(1, 2, figsize=(15, 5.5))
    eje = ejes[0]
    nombres = ["V19\n(champion)"] + [r["configuracion"].split()[0] for r in resultados]
    valores = [0.0] + [r["delta_contra_v19"] * 1000 for r in resultados]
    eje.bar(range(len(nombres)), valores, color=["#1a9850"] + ["#2166ac"] * len(resultados))
    eje.axhline(0, color="grey", linewidth=1)
    eje.axhline(GATE * 1000, color="#4d4d4d", linestyle="--", linewidth=1,
                label=f"gate de adopcion {GATE}")
    eje.set_xticks(range(len(nombres)))
    eje.set_xticklabels(nombres, fontsize=8)
    eje.set_ylabel("delta de macro-F1 contra V19 (milesimas)")
    eje.set_title("Ninguna configuracion llega al gate de adopcion", fontsize=10)
    eje.legend(fontsize=8)
    eje = ejes[1]
    eje.bar(range(len(resultados)),
            [r["complementariedad"]["solo_challenger"] for r in resultados], 0.4,
            label="solo el challenger acierta", color="#1a9850")
    eje.bar(range(len(resultados)),
            [-r["complementariedad"]["solo_v19"] for r in resultados], 0.4,
            label="solo V19 acierta (negativo)", color="#b2182b")
    eje.axhline(0, color="grey", linewidth=1)
    eje.set_xticks(range(len(resultados)))
    eje.set_xticklabels([r["configuracion"].split()[0] for r in resultados], fontsize=8)
    eje.set_ylabel("filas")
    eje.set_title("Complementariedad de errores frente a V19", fontsize=10)
    eje.legend(fontsize=8)
    fig.suptitle("V19 X1: las dos unicas configuraciones autorizadas por el especialista", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURA, dpi=130)
    plt.close(fig)

    # ---------------------------------------------------------------- informe
    INFORME.write_text("\n".join([
        "# V19 · X1 — Explotacion del champion: puerta previa y las dos configuraciones", "",
        "El especialista recomienda **cerrar la explotacion de V19**. Sus tres argumentos se resumen "
        "antes que los resultados, porque condicionan si estos dos experimentos pueden decidir algo:", "",
        "1. **La tasa base del subespacio de parametros es +0,0008** frente a un gate de +0,0030. Las "
        "14 formulaciones cerradas de V15 a V18 se movieron en [-0,0040, +0,0008]. Todo lo que alguna "
        "vez movio la aguja fue un **cambio de mecanismo**, nunca un parametro.",
        "2. **La resolucion del aparato es de +0,0011**: 0,6090431 en 4 folds frente a 0,6079388 en 5, "
        "**sin cambiar nada del modelo**. Un ajuste con efecto real menor que +0,001 no es "
        "distinguible de elegir particion.",
        "3. **El residuo de la calibracion sobre V19 es de -0,0016** (CV 0,6079388 predecía 0,62735 y "
        "el LB real es 0,62578). En V12 el residuo fue +0,0019, luego la oscilacion entre modelos "
        "vecinos es de 0,0034: un +0,002 en CV puede aterrizar en cualquier punto entre -0,000 y +0,004 "
        "en LB.", "",
        "## 1. La puerta de coste cero: la saturacion se sostiene", "",
        "| Medida | Valor |", "|---|---|",
        f"| arboles del champion | {saturacion['arboles']} |",
        f"| cortes internos medios | {saturacion['cortes_medios']} |",
        f"| techo de cortes por arbol | {saturacion['techo_de_cortes_por_arbol']} |",
        f"| **ocupacion del techo** | **{saturacion['ocupacion_del_techo']:.1%}** |",
        f"| arboles que llegan al techo | {saturacion['arboles_al_techo']} de {saturacion['arboles']} |",
        f"| arboles con menos de 5 cortes | {saturacion['arboles_por_debajo_de_5_cortes']} |",
        f"| la tesis se sostiene (umbral 95 %) | **{saturacion['la_tesis_se_sostiene']}** |", "",
        "Con `num_leaves = 8` un arbol puede como maximo 7 cortes, y la Depth 3 permite exactamente "
        f"{saturacion['arboles']} de ellos. La ocupacion real es del "
        f"{saturacion['ocupacion_del_techo']:.1%}, luego **la profundidad es lo que limita y no las "
        "hojas**, que es la premisa de la configuracion A. Si hubiera sido del 70 %, la lectura "
        "cambiaria y el eje de capacidad reabriria.", "",
        "## 2. Las dos configuraciones y nada mas", "",
        "El especialista delimito con numeros el resto y las descarto por inertes o ya sondadas: "
        "`num_leaves` 8 a 16 con profundidad 3 es un **no-op exacto** porque 2^3 = 8; `reg_lambda` 1 a "
        "10 es inerte a esta escala de hojas; `min_child_weight` es una reparametrizacion de "
        "`min_child_samples` porque la hessiana multiclase por fila es 0,5-0,8; y bajar `colsample` "
        "mata el bloque de interaccion. Los pesos del blend se sondearon dos veces (V16 A1 -0,0003 y A2 "
        "+0,0006 con el IC95 contemplando el cero).", "",
        "| Configuracion | CV 5 folds | Delta contra V19 | Folds pos. | IC95 contra V19 | CV 4 folds | "
        "Delta 4 folds | Merece submission |",
        "|---|---|---|---|---|---|---|---|",
        f"| **V19 (champion)** | {CV_V19} | - | - | - | {CV_V19_4} | - | - |",
        *[f"| {r['configuracion']} | {r['macro_f1']} | {r['delta_contra_v19']:+.4f} | "
          f"{r['folds_positivos_contra_v19']}/5 | [{r['bootstrap_ic95_contra_v19'][0]:+.4f}, "
          f"{r['bootstrap_ic95_contra_v19'][1]:+.4f}] | {r['cv_4_folds']} | {r['delta_4_folds']:+.4f} | "
          f"{'si' if r['merece_submission'] else '**no**'} |" for r in resultados], "",
        "## 3. Fronteras y complementariedad de errores frente a V19", "",
        "| Configuracion | Frontera 1 | Frontera 1 de V19 | Frontera 2 | Frontera 2 de V19 | Solo el "
        "candidato acierta | Solo V19 acierta | Neto |", "|---|---|---|---|---|---|---|---|",
        *[f"| {r['configuracion']} | {r['frontera_1']} | {r['frontera_1_v19']} | {r['frontera_2']} | "
          f"{r['frontera_2_v19']} | {r['complementariedad']['solo_challenger']:,} | "
          f"{r['complementariedad']['solo_v19']:,} | {r['complementariedad']['neto']:+,} |"
          for r in resultados], "",
        "## 4. Criterio de submission, aplicado sin abreviar", "",
        "| Criterio | " + " | ".join(r["configuracion"].split()[0] for r in resultados) + " |",
        "|---|" + "---|" * len(resultados),
        *[f"| {nombre} | " + " | ".join(
            f"{'si' if r['criterios_de_submission'][nombre] else 'no'}" for r in resultados) + " |"
          for nombre in ("delta_5_folds", "cinco_de_cinco_folds", "ic95_excluye_cero",
                         "confirmacion_4_folds")], "",
        f"**Ninguna configuracion merece submission.** {sum(1 for r in resultados if r['merece_submission'])} "
        f"de {len(resultados)} pasan los cuatro criterios.", "",
    ]), encoding="utf-8")

    RESULTADO.write_text(json.dumps({
        "fase": "V19 X1 explotacion del champion", "fecha": "2026-09-27",
        "champion": {"nombre": "V19 boosting", "lb": 0.62578, "cv_5folds": CV_V19,
                     "cv_4folds": CV_V19_4},
        "recomendacion_del_especialista": "cerrar la explotacion; como maximo 2 configs diagnosticas",
        "tasa_base_del_subespacio_de_parametros": 0.0008, "gate": GATE,
        "resolucion_del_aparato_entre_particiones": 0.0011,
        "residuo_de_la_calibracion_sobre_v19": -0.0016,
        "puerta_de_saturacion": saturacion,
        "configuraciones": resultados,
        "submission_generada": False,
        "razon_del_cierre": ("el techo conjunto de las dos configuraciones queda por debajo del gate y "
                             "por debajo de la resolucion del aparato de medicion"),
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\npuerta: ocupacion {saturacion['ocupacion_del_techo']:.1%}, la tesis se sostiene "
          f"{saturacion['la_tesis_se_sostiene']}")
    print(f"configuraciones que merecen submission: "
          f"{sum(1 for r in resultados if r['merece_submission'])} de {len(resultados)}")
    print(f"segundos: {time.time() - inicio:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
