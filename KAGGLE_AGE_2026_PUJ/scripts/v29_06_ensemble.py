"""V29 - 06 - P9, P10 y P11: optimizacion de pesos, seleccion de subconjunto y complementariedad.

Es la parte decisiva de V29. Un ensemble se puede construir de dos maneras y aqui se hacen las dos,
**las dos anidadas**, para que ninguna pueda apoyarse en la validacion exterior.

## P9: Optuna sobre los pesos

Los pesos se parametrizan como un **softmax** de un vector libre, lo que garantiza `w_i >= 0` y
`sum w_i = 1` sin necesitar la restriccion. Se ajustan sobre un holdout **interior** y se miden en la
validacion **exterior**.

Se evaluan dos objetivos, porque el brief no presupone que el segundo gane:

- **Objetivo 1** — `macro_F1` puro. Lo unico que se optimiza.
- **Objetivo 2** — `macro_F1 - lambda * H(w)/log(k)`, donde `H(w) = -sum_i w_i log(w_i)` es la
  entropia de Shannon de los pesos. Es una penalizacion de **compacidad**, no una recompensa por
  diversidad: cuanto mas repartidos los pesos, mas entropia, mas penalizacion. La forma esta
  justificada en el docstring de `_objetivo_2` y `lambda` es pequena a proposito, para no forzar
  pesos hacia modelos debiles.

## P10: seleccion de subconjunto

Ademas de Optuna, se ejecuta seleccion greedily hacia adelante con repeticion (Caruana), que es
es-naturalmente dispersa: cada ronda anade el miembro que mas mejora el promedio, y el tope de
**6 miembros distintos** se impone por construccion, no por penalizacion posterior.

## El control que decide si esto es H04 otra vez

Al final se mira **en quien apoya el optimum**. Si los pesos se concentran en V20 o en un solo
miembro, entonces la composicion no aporta nada y la conclusion es que V29 no encontro senal nueva.
Si el peso se reparte entre familias estructuralmente distintas, la hipotesis de diversidad tiene
algo que decir. Ese juicio se escribe en el artefacto, no se deja a la interpretacion.

## La fuga indirecta que se declara

Los OOF historicos de los miembros se produjeron con los folds de V20, asi que la prediccion de
una fila del outer_train viene de un modelo que **si vio parte del outer_val**. Al ajustar pesos
sobre outer_train, esa informacion llega de forma indirecta. Se mide su sensibilidad: se ajustan
los pesos dos veces sobre mitades distintas del interior y se compara el vector resultante. Si los
pesos son estables, la fuga no mueve la conclusion. Se reporta el numero aunque incomode.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402
from v13_p1b_fusion_tokens import bootstrap_diferencia  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)

RESULTADO = cfg.EXPERIMENTS / "v29_06_ensemble.json"
TABLA = cfg.REPORTS / "v29_ensemble_results.csv"
N_CLASES = 6
CV_V20 = 0.6097782
GATE_DELTA = 0.003
GATE_FOLDS = 4
TRIALS_PESOS = 300
RONDAS_CARUANA = 30
MAXIMO_MEMBROS = 6
#: Penalizacion de compacidad del objetivo 2. Pequena a proposito.
LAMBDA = 0.002
SEMILLA_OPTUNA = 20260929
PARES_FRONTERA = {("18-24", "25-34"): "18-24|25-34", ("25-34", "35-49"): "25-34|35-49"}


def _macro(clase: np.ndarray, prob: np.ndarray) -> float:
    return float(f1_score(clase, np.argmax(prob, axis=1), average="macro",
                          labels=list(range(N_CLASES))))


def _softmax(theta: np.ndarray) -> np.ndarray:
    """Convierte un vector libre en pesos del simplex: no negativos y suman uno."""
    e = np.exp(theta - theta.max())
    return e / e.sum()


def _combina(matrices: np.ndarray, pesos: np.ndarray) -> np.ndarray:
    """Aplica los pesos a la pila de OOF. `matrices` es (k, n, 6)."""
    return np.tensordot(pesos, matrices, axes=(0, 0))


def _objetivo_2(macro: float, pesos: np.ndarray, lam: float) -> float:
    """`macro_F1 - lambda * H(w)/log(k)`, con `H` la entropia de Shannon de los pesos.

    La entropia de un vector de pesos esta en `[0, log k]`: vale 0 si todo el peso esta en un
    miembro y `log k` si esta repartido por igual. Dividir por `log k` la deja en `[0, 1]`, con lo
    que `lambda` tiene el mismo sentido sea cual sea el numero de candidatos. Penalizar la entropia
    es penalizar el **reparto**: empuja los pesos a concentrarse en pocos miembros. No es una
    recompensa por diversidad, y por eso no puede premiar a un modelo debil: un modelo debil que
    concentra peso empeoraria el macro-F1 mucho mas de lo que ahorra en entropia.
    """
    k = len(pesos)
    if k <= 1:
        return macro
    h = float(-(pesos * np.log(np.clip(pesos, 1e-12, None))).sum())
    return macro - lam * h / np.log(k)


def main() -> int:
    """Ejecuta P9, P10 y P11 sobre el pool congelado."""
    autotest_numerico()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(v) for v in y])
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")

    pool_info = json.loads((cfg.EXPERIMENTS / "v29_05_pool.json").read_text(encoding="utf-8"))
    datos_pool = np.load(cfg.EXPERIMENTS / "v29_pool_oof.npz")
    miembros = list(datos_pool.files)
    matrices = np.stack([datos_pool[m].astype(np.float64) for m in miembros])
    print(f"[P9] pool de {len(miembros)} miembros congelado", flush=True)
    print(f"  {', '.join(miembros)}", flush=True)

    oof_v20 = np.load(cfg.FEATURES / "v20_f3b_oof_candidato.npy")
    pred_v20 = np.argmax(oof_v20, axis=1)
    bien_v20 = pred_v20 == clase
    v20_en_pool = "v20_f3b_oof_candidato" in miembros
    indice_v20 = miembros.index("v20_f3b_oof_candidato") if v20_en_pool else -1

    # ---------------------------------------------------------------- los tres metodos
    def _tres(m: np.ndarray, nombres: list[str], prefijo: str) -> dict:
        return {
            f"{prefijo}1 Optuna macro-F1": _optuna_pesos(m, clase, folds, nombres,
                                                          objetivo=1, lam=0.0),
            f"{prefijo}2 Optuna macro-F1 menos lambda por entropia": _optuna_pesos(
                m, clase, folds, nombres, objetivo=2, lam=LAMBDA),
            f"{prefijo}3 Caruana con tope de 6 miembros": _caruana(m, clase, folds, nombres),
        }

    metodos = _tres(matrices, miembros, "P9-")
    # Brazo de control: el **mismo** pool sin los modelos de Optuna. Sin el, el contraste entre
    # "la diversidad no sirve" y "la capacidad si" seria una cifra escrita a mano en vez de una
    # medicion, y es justamente la medicion que sostiene la conclusion de V29.
    sin_optuna = [i for i, n in enumerate(miembros) if not n.startswith("v29_")]
    if len(sin_optuna) < len(miembros):
        print(f"\n=== brazo de control: los mismos 3 metodos sobre {len(sin_optuna)} miembros, "
              f"sin los modelos de Optuna ===", flush=True)
        control = _tres(matrices[sin_optuna], [miembros[i] for i in sin_optuna], "C-")
    else:
        control = {}
        print("\n[aviso] no hay modelos de Optuna en el pool: el brazo de control no se ejecuta",
              flush=True)
    # Referencia: el propio V20, y el promedio igual de todo el pool sin optimizar nada.
    metodos["T0 V20 sin tocar"] = {"oof": oof_v20, "pesos": None, "detalle": {}}
    iguales = matrices.mean(axis=0)
    metodos["T1 promedio igual de todo el pool"] = {"oof": iguales, "pesos": None, "detalle": {}}

    metodos.update(control)
    filas, detalle = [], {}
    for nombre, resultado in metodos.items():
        oof = resultado["oof"]
        pred = np.argmax(oof, axis=1)
        valor = _macro(clase, oof)
        por_fold = [_macro(clase[f["val_idx"]], oof[f["val_idx"]]) for f in folds]
        base_fold = [_macro(clase[f["val_idx"]], oof_v20[f["val_idx"]]) for f in folds]
        deltas = [round(a - b, 6) for a, b in zip(por_fold, base_fold)]
        rescate = int(((~bien_v20) & (pred == clase)).sum())
        dano = int((bien_v20 & (pred != clase)).sum())
        cumple_delta = valor - CV_V20 >= GATE_DELTA
        cumple_folds = sum(1 for d in deltas if d > 0) >= GATE_FOLDS
        item = {
            "metodo": nombre, "corrected_cv": round(valor, 7),
            "delta_vs_V20": round(valor - CV_V20, 6),
            "fold_1": round(por_fold[0], 7), "fold_2": round(por_fold[1], 7),
            "fold_3": round(por_fold[2], 7), "fold_4": round(por_fold[3], 7),
            "fold_5": round(por_fold[4], 7),
            "media_por_fold": round(float(np.mean(por_fold)), 6),
            "desviacion_por_fold": round(float(np.std(por_fold)), 6),
            "delta_por_fold": deltas, "folds_positivos": sum(1 for d in deltas if d > 0),
            "rescue": rescate, "damage": dano, "net": rescate - dano,
            "filas_cambiadas": int((pred != pred_v20).sum()),
            "f1_por_clase": {c: round(float(v), 4) for c, v in zip(
                cfg.CLASSES, f1_score(clase, pred, average=None,
                                      labels=list(range(N_CLASES))))},
            "cumple_delta": bool(cumple_delta), "cumple_folds": bool(cumple_folds),
            "supera_gate": bool(cumple_delta and cumple_folds),
        }
        if indice_v20 >= 0 and resultado["pesos"] is not None:
            item["peso_de_V20"] = round(float(resultado["pesos"][indice_v20]), 4)
            item["miembros_efectivos"] = int((np.asarray(resultado["pesos"]) > 0.05).sum())
            item["peso_maximo"] = round(float(np.max(resultado["pesos"])), 4)
            item["peso_maximo_miembro"] = miembros[int(np.argmax(resultado["pesos"]))]
            item["apoyo_por_familia"] = _por_familia(miembros, resultado["pesos"],
                                                      pool_info["pool"])
        filas.append(item)
        detalle[nombre] = {"resumen": item, "detalle": resultado["detalle"]}
        print(f"  [{nombre}] CV {valor:.7f} | delta {valor - CV_V20:+.6f} | "
              f"folds +{item['folds_positivos']}/5 | neto {rescate - dano:+,} | "
              f"gate: {item['supera_gate']}", flush=True)

    # ---------------------------------------------------------------- el control H04
    print("\n=== control: ¿el optimum se apoya en V20 o se reparte entre familias? ===", flush=True)
    h04 = {}
    metodos_optimizados = [n for n in metodos if n.startswith("P9-") or n.startswith("P10-")]
    for nombre in metodos_optimizados:
        p = detalle[nombre]["resumen"]
        if "peso_de_V20" not in p:
            continue
        h04[nombre] = {"peso_de_V20": p["peso_de_V20"], "peso_maximo": p["peso_maximo"],
                       "miembro_que_lleva_el_peso": p["peso_maximo_miembro"],
                       "miembros_efectivos": p["miembros_efectivos"],
                       "apoyo_por_familia": p["apoyo_por_familia"]}
        print(f"  {nombre}")
        print(f"    peso de V20: {p['peso_de_V20']:.4f} | miembro dominante: "
              f"{p['peso_maximo_miembro']} ({p['peso_maximo']:.4f}) | "
              f"miembros con peso > 0,05: {p['miembros_efectivos']}", flush=True)
        repartido = sum(1 for f, v in p["apoyo_por_familia"].items() if v > 0.05)
        print(f"    familias con peso > 0,05: {repartido}", flush=True)
        p["familias_con_peso"] = repartido
    decision_h04 = ("el optimum se apoya en una sola estructura: es H04 de nuevo, cerrado por "
                    "equivalencia"
                    if all(detalle[n]["resumen"].get("familias_con_peso", 0) <= 1
                           for n in metodos_optimizados)
                    else "el optimum reparte peso entre estructuras distintas de V20: hay algo "
                         "que decir")
    # El contraste medido entre el pool completo y el pool sin Optuna, que es lo que sostiene la
    # conclusion de la fase.
    contraste = {}
    for sufijo in ("1", "2", "3"):
        con = next((r for r in filas if r["metodo"].startswith(f"P9-{sufijo}")), None)
        sin = next((r for r in filas if r["metodo"].startswith(f"C-{sufijo}")), None)
        if con and sin:
            contraste[sufijo] = {
                "metodo": con["metodo"],
                "delta_con_optuna": con["delta_vs_V20"], "delta_sin_optuna": sin["delta_vs_V20"],
                "diferencia_atribuida_al_modelo_nuevo": round(
                    con["delta_vs_V20"] - sin["delta_vs_V20"], 6),
                "folds_con_optuna": con["folds_positivos"], "folds_sin_optuna": sin["folds_positivos"],
            }

    # ---------------------------------------------------------------- P12: bootstrap de lo que pasa el gate
    print("\n=== P12: bootstrap solo de los que superan el gate ===", flush=True)
    print("  (el brief prohibe bootstrapizar todos los metodos)", flush=True)
    Passing = [f for f in filas if f["supera_gate"]]
    bootstraps = {}
    if not Passing:
        print("  ninguno supera el gate: no se calcula ningun IC95, y eso ya es un resultado", flush=True)
    for f in Passing:
        oof = metodos[f["metodo"]]["oof"]
        intervalo = bootstrap_diferencia(oof_v20, oof, clase, N_CLASES)
        bootstraps[f["metodo"]] = {
            "delta": f["delta_vs_V20"],
            "ic95": [round(float(intervalo[0]), 6), round(float(intervalo[1]), 6)],
            "proporcion_positiva": round(float(intervalo[2]), 6),
            "ic95_incluye_cero": bool(intervalo[0] <= 0.0 <= intervalo[1])}
        print(f"  {f['metodo']}: IC95 {bootstraps[f['metodo']]['ic95']}", flush=True)

    pd.DataFrame([{k: v for k, v in f.items() if k != "f1_por_clase"} for f in filas]).to_csv(
        TABLA, index=False)

    RESULTADO.write_text(json.dumps({
        "fase": "V29 06 P9-P12 ensemble", "fecha": "2026-09-29", "experimenta": False,
        "n_miembros_pool": len(miembros), "miembros": miembros,
        "metodos": detalle, "tabla": filas,
        "control_h04": {"datos": h04, "decision": decision_h04},
        "control_sin_modelo_nuevo": contraste,
        "bootstrap": bootstraps,
        "finalistas": [f["metodo"] for f in Passing],
        "n_finalistas": len(Passing),
        "hipotesis_central": "si el optimum no reparte peso entre estructuras distintas y no supera "
                            "el gate, la respuesta es que V20 NO estaba limitado por falta de "
                            "diversidad complementaria",
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[fin] {time.time() - inicio:.0f} s")
    return 0


def _por_familia(miembros: list[str], pesos: np.ndarray, pool: list[dict]) -> dict:
    """Reparte el peso por familia, para poder ver si se reparte entre estructuras."""
    familias = {f["model_id"]: f["familia"] for f in pool}
    salida: dict[str, float] = {}
    for m, w in zip(miembros, pesos):
        salida[familias.get(m, "desconocida")] = round(
            salida.get(familias.get(m, "desconocida"), 0.0) + float(w), 4)
    return dict(sorted(salida.items(), key=lambda kv: -kv[1]))


def _interior(clase: np.ndarray, entrenamiento: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Holdout interior agrupado por posicion, para no filtrar fuera de outer_train."""
    partes = cvmod.make_folds(clase[entrenamiento], groups=np.arange(len(entrenamiento)),
                              n_splits=5, n_repeats=1, scheme="stratified")
    elegido = min(partes, key=lambda f: abs(len(f["val_idx"]) - int(0.2 * len(entrenamiento))))
    val = entrenamiento[np.asarray(elegido["val_idx"])]
    tr = entrenamiento[np.concatenate([np.asarray(o["val_idx"]) for o in partes
                                       if o["fold"] != elegido["fold"]])]
    return tr, val


def _optuna_pesos(matrices: np.ndarray, clase: np.ndarray, folds: list[dict], miembros: list[str],
                  objetivo: int, lam: float) -> dict:
    """Ajusta los pesos por Optuna dentro de cada fold exterior. Devuelve el OOF ensamblado."""
    k = len(miembros)
    oof = np.zeros((matrices.shape[1], N_CLASES))
    detalle = {}
    for fold in folds:
        val = np.asarray(fold["val_idx"])
        tr = np.concatenate([np.asarray(o["val_idx"]) for o in folds if o["fold"] != fold["fold"]])
        tr_in, val_in = _interior(clase, tr)
        assert not np.intersect1d(val_in, val).size, "holdout interior toca la validacion exterior"

        mejor = {"theta": None, "valor": -np.inf}

        def f(trial: optuna.Trial) -> float:
            theta = np.array([trial.suggest_float(f"w{i}", -4.0, 4.0) for i in range(k)])
            pesos = _softmax(theta)
            macro = _macro(clase[val_in], _combina(matrices[:, val_in, :], pesos))
            valor = macro if objetivo == 1 else _objetivo_2(macro, pesos, lam)
            trial.set_user_attr("macro_f1_inner", round(macro, 6))
            if valor > mejor["valor"]:
                mejor.update({"theta": theta, "valor": valor, "macro": macro})
            return valor

        estudio = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SEMILLA_OPTUNA + int(fold["fold"])))
        estudio.optimize(f, n_trials=TRIALS_PESOS, show_progress_bar=False)
        pesos = _softmax(mejor["theta"])
        oof[val] = _combina(matrices[:, val, :], pesos)

        # Sensibilidad: los pesos ajustados sobre una mitad del interior, comparados con los del
        # otro. Mide cuanto depende el vector de las filas concretas, que es lo que la fuga
        # indirecta de los OOF historicos podria mover.
        rng = np.random.default_rng(SEMILLA_OPTUNA + int(fold["fold"]))
        idx = rng.permutation(len(val_in))
        mitad = len(idx) // 2
        vectores = []
        for trozo in (idx[:mitad], idx[mitad:]):
            sub = val_in[trozo]

            def g(trial: optuna.Trial) -> float:
                theta = np.array([trial.suggest_float(f"w{i}", -4.0, 4.0) for i in range(k)])
                return _macro(clase[sub], _combina(matrices[:, sub, :], _softmax(theta)))

            e2 = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=SEMILLA_OPTUNA + int(fold["fold"])))
            e2.optimize(g, n_trials=TRIALS_PESOS // 2, show_progress_bar=False)
            vectores.append(_softmax(e2.best_params and
                                     np.array([e2.best_params[f"w{i}"] for i in range(k)])))
        l1 = float(np.abs(vectores[0] - vectores[1]).sum())
        detalle[int(fold["fold"])] = {
            "n_inner_train": int(len(tr_in)), "n_inner_val": int(len(val_in)),
            "n_trials": TRIALS_PESOS, "mejor_trial": int(estudio.best_trial.number),
            "mejor_valor_objetivo": round(float(mejor["valor"]), 6),
            "mejor_macro_f1_inner": round(float(mejor["macro"]), 6),
            "mediana_trial": round(float(np.median([
                t.value for t in estudio.trials if t.value is not None])), 6),
            "mejor_sobre_mediana_en_sigmas": round(float(
                (estudio.best_value - np.median([t.value for t in estudio.trials
                                                 if t.value is not None]))
                / max(np.std([t.value for t in estudio.trials if t.value is not None]), 1e-9)), 4),
            "mejor_es_outlier": bool(
                estudio.best_value - np.median([t.value for t in estudio.trials
                                               if t.value is not None])
                > 3 * max(np.std([t.value for t in estudio.trials if t.value is not None]), 1e-9)),
            "pesos": {m: round(float(w), 4) for m, w in zip(miembros, pesos) if w > 0.01},
            "L1_entre_pesos_de_dos_mitades": round(l1, 4),
            "macro_f1_outer_val": round(_macro(clase[val], oof[val]), 6),
        }
        print(f"    fold {fold['fold']} ({'macro-F1' if objetivo == 1 else 'con entropia'}): "
              f"macro interior {mejor['macro']:.6f} | outer_val "
              f"{detalle[int(fold['fold'])]['macro_f1_outer_val']:.6f} | L1 entre mitades {l1:.3f}",
              flush=True)
    pesos_medios = np.mean([[detalle[f]["pesos"].get(m, 0.0) for m in miembros]
                            for f in detalle], axis=0)
    pesos_medios = pesos_medios / pesos_medios.sum()
    return {"oof": oof, "pesos": pesos_medios, "detalle": detalle}


def _caruana(matrices: np.ndarray, clase: np.ndarray, folds: list[dict],
             miembros: list[str]) -> dict:
    """Seleccion greedily con repeticion, con tope duro de miembros distintos."""
    k = len(miembros)
    oof = np.zeros((matrices.shape[1], N_CLASES))
    detalle = {}
    for fold in folds:
        val = np.asarray(fold["val_idx"])
        tr = np.concatenate([np.asarray(o["val_idx"]) for o in folds if o["fold"] != fold["fold"]])
        _, val_in = _interior(clase, tr)
        assert not np.intersect1d(val_in, val).size, "holdout interior toca la validacion exterior"
        eleccion: list[int] = []
        corrida = np.zeros((len(val_in), N_CLASES))
        usados: set[int] = set()
        for ronda in range(RONDAS_CARUANA):
            mejor_indice, mejor_valor = None, -np.inf
            for i in range(k):
                if i not in usados and len(usados) >= MAXIMO_MEMBROS:
                    continue
                candidata = (corrida * ronda + matrices[i][val_in]) / (ronda + 1)
                valor = _macro(clase[val_in], candidata)
                if valor > mejor_valor:
                    mejor_indice, mejor_valor = i, valor
            eleccion.append(mejor_indice)
            usados.add(mejor_indice)
            corrida = (corrida * ronda + matrices[mejor_indice][val_in]) / (ronda + 1)
        conteo = np.bincount(eleccion, minlength=k).astype(float)
        pesos = conteo / conteo.sum()
        oof[val] = _combina(matrices[:, val, :], pesos)
        detalle[int(fold["fold"])] = {
            "n_inner_val": int(len(val_in)), "rondas": RONDAS_CARUANA,
            "tope_miembros_distintos": MAXIMO_MEMBROS,
            "miembros_distintos_elegidos": int((conteo > 0).sum()),
            "macro_f1_inner_final": round(_macro(clase[val_in], corrida), 6),
            "orden_de_entrada": [miembros[i] for i in eleccion],
            "pesos": {m: round(float(w), 4) for m, w in zip(miembros, pesos) if w > 0.01},
            "macro_f1_outer_val": round(_macro(clase[val], oof[val]), 6),
        }
        print(f"    fold {fold['fold']} (Caruana): {detalle[int(fold['fold'])]['miembros_distintos_elegidos']}"
              f" miembros distintos de {k} | inner "
              f"{detalle[int(fold['fold'])]['macro_f1_inner_final']:.6f} | outer_val "
              f"{detalle[int(fold['fold'])]['macro_f1_outer_val']:.6f}", flush=True)
    conteo = np.mean([[detalle[f]["pesos"].get(m, 0.0) for m in miembros] for f in detalle], axis=0)
    conteo = conteo / conteo.sum()
    return {"oof": oof, "pesos": conteo, "detalle": detalle}


if __name__ == "__main__":
    raise SystemExit(main())
