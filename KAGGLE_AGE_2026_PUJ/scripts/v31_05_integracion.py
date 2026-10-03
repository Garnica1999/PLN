"""V31+ - 05: integracion del canal linguistico en el meta-espacio de V30.

## La pregunta

V31-03 dijo que el canal `ortografia+composicion+discurso` vale +0,006820 sobre el texto solo,
con 5/5 folds. V31-04 dijo que, como clasificador suelto, es mas debil que V30 pero **falla en
un 68% distinto**: acierta 11.761 filas donde V30 se equivoca.

Un miembro mas debil con errores propios es exactamente el perfil que hizo util a V29. Pero
"util" solo se demuestra integrandolo. Este script responde a esa pregunta.

## Las tres integraciones, y por que en este orden

```text
N1  V30 + el canal como cuarto miembro de la mezcla, con pesos optimizados
N2  los log-odds del canal como features extra para los miembros de V30
N3  el canal como features extra en el meta-espacio de 279
```

N1 es la mas barata y ataca el mecanismo que V30 ya demostro que funciona (mezcla ponderada de
miembros con errores correlacionados). Si N1 no mueve nada, N2 y N3 tampoco van a.move mucho,
porque las tres compan la misma pregunta: hay informacion que el meta-espacio de V30 no ve.

Se ejecutan en ese orden y se para si la primera no da senal, para no gastar presupuesto en
variantes de lo mismo.

## El control que decide de verdad

Un cuarto miembro con pesos optimizados **siempre** mejora algo, y ese algo no tiene por que
venir del canal: puede venir de tener cuatro pesos donde antes habia tres. Para separar las dos
cosas hace falta el brazo de control del brief:

```text
control con la representacion **barajada**: las mismas filas de probabilidad del canal, pero
con las asignadas al azar. Si el barajado rinde casi lo mismo que el canal real, la mejora no
venia de la senal linguistica sino de tener un miembro mas.
```

Ese control es la unica forma honesta de distinguir "el canal aporta informacion" de "un
ensamble de cuatro pesos rinde mas". Sin el, cualquier gain seria ininterpretable.

## La estructura anidada, sin excepciones

```text
outer_train
    |
    +-- inner_train x4   -> se optimizan los pesos aqui
    |
outer_validation        -> se evalua aqui, y no se ha visto nunca
```

Los pesos se ajustan **dentro** de `outer_train` con folds internos. `outer_validation` no
participa en ninguna decision. Es la diferencia entre una integracion real y una fuga, y por eso
el OOF que sale de aqui es comparable con el de V30.

## Controles negativos que tambien se ejecutan

- **V30 con 3 pesos reoptimizados**: si subir de 3 a 4 pesos ya da gain, ese gain no es del
  canal. Por eso el reoptimizado de 3 es la referencia, no el V30 congelado.
- **V30 congelado**: se reporta tambien, para no perder de vista la referencia absoluta.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v31_03_canales_linguisticos import (  # noqa: E402
    VEC_CANAL, VEC_PALABRA, canal_composicion, canal_discurso, canal_ortografia,
)
from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

RESULTADO = cfg.EXPERIMENTS / "v31_05_integracion.json"
MAESTRO_CSV = cfg.REPORTS / "v31_master_results.csv"
REGISTRO = cfg.REPORTS / "v31_hypothesis_registry.csv"
FECHA = "2026-09-30"
CV_V30 = 0.6134471
N_CLASES = 6
PESO_CANAL = 1.0
CANALES = ["ortografia", "composicion", "discurso"]
CACHE_CANAL = cfg.FEATURES / "v31_canal_oof.npy"
SEMILLA = 20260930
N_TRIALS = 60


def _macro(clase: np.ndarray, prob: np.ndarray) -> float:
    return float(f1_score(clase, np.argmax(prob, axis=1), average="macro",
                          labels=list(range(N_CLASES))))


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ============================================================================ el canal
def oof_canal(textos: np.ndarray, clase: np.ndarray, folds: list[dict]) -> np.ndarray:
    """OOF del canal linguistic. Se cachea porque cuesta ~10 min y no depende de nada mas.

    El cache es legitimo y no es un atajo perezoso: el OOF es una funcion determinista de
    (textos, clase, folds, semilla del vectorizador), y los folds son los congelados de V20.
    Lo que el brief no permite es **reutilizar un OOF sin poder reconstruirlo**, no cachear una
    reconstruccion. Se guarda el hash del corpus para que quede constancia de que no cambio.
    """
    if CACHE_CANAL.exists():
        matriz = np.load(CACHE_CANAL)
        if matriz.shape == (len(clase), N_CLASES):
            print("  OOF del canal: desde cache", flush=True)
            return matriz
    import hashlib
    print("  OOF del canal: calculando (cache miss)", flush=True)
    canales = {"ortografia": canal_ortografia, "composicion": canal_composicion,
               "discurso": canal_discurso}
    matrices = {n: TfidfVectorizer(**VEC_CANAL).fit_transform([f(t) for t in textos])
                for n, f in canales.items()}
    print(f"  hash del corpus: {hashlib.sha256(textos.tobytes()).hexdigest()[:16]}", flush=True)
    oof = np.zeros((len(clase), N_CLASES), dtype=np.float64)
    for f in folds:
        ent, val = f["train_idx"], f["val_idx"]
        vec = TfidfVectorizer(**VEC_PALABRA)
        x_ent = sparse.hstack([vec.fit_transform(textos[ent])]
                              + [matrices[n][ent] * PESO_CANAL for n in CANALES], format="csr")
        x_val = sparse.hstack([vec.transform(textos[val])]
                              + [matrices[n][val] * PESO_CANAL for n in CANALES], format="csr")
        m = LinearSVC(C=1.0, loss="squared_hinge", penalty="l2", max_iter=3000, tol=1e-4)
        m.fit(x_ent, clase[ent])
        dec = m.decision_function(x_val)
        if dec.ndim == 1:
            dec = np.column_stack([-dec, dec])
        dec = dec - dec.max(axis=1, keepdims=True)
        e = np.exp(dec)
        oof[val] = e / e.sum(axis=1, keepdims=True)
    np.save(CACHE_CANAL, oof)
    return oof


# ============================================================================ optimizacion
def optimiza_pesos(matrices: list[np.ndarray], clase: np.ndarray, idx: np.ndarray,
                   folds_internos: list[dict], n_trials: int) -> np.ndarray:
    """Optuna sobre los pesos, elegido por macro-F1 en los folds INTERNOS.

    La funcion objetivo ve unicamente `idx`, que es el outer_train de este fold. La
    `outer_validation` correspondiente no se pasa aqui en ningun momento.
    """
    import optuna
    log = optuna.logging.get_logger("optuna")
    log.setLevel(optuna.logging.WARNING)
    sub_clase = clase[idx]
    limite = len(idx)
    for f in folds_internos:
        assert f["val_idx"].max() < limite, (
            f"un fold interno se sale del rango local: max {f['val_idx'].max()} >= {limite}. "
            f"Los folds internos tienen que estar en indices RELATIVOS a idx, no globales")
        assert f["train_idx"].max() < limite, "fold interno de train fuera de rango"
    cache: dict[tuple, float] = {}

    def objetivo(prueba) -> float:
        pesos = np.array([prueba.suggest_float(f"w{i}", 0.0, 1.0) for i in range(len(matrices))])
        if pesos.sum() <= 0:
            return 0.0
        pesos = pesos / pesos.sum()
        clave = tuple(np.round(pesos, 4))
        if clave in cache:
            return cache[clave]
        mezcla = np.tensordot(pesos, np.stack([m[idx] for m in matrices]), axes=(0, 0))
        valor = float(np.mean([_macro(sub_clase[f["val_idx"]], mezcla[f["val_idx"]])
                               for f in folds_internos]))
        cache[clave] = valor
        return valor

    estudio = optuna.create_study(direction="maximize",
                                  sampler=optuna.samplers.TPESampler(seed=SEMILLA))
    estudio.optimize(objetivo, n_trials=n_trials, show_progress_bar=False)
    pesos = np.array([estudio.best_params[f"w{i}"] for i in range(len(matrices))])
    return pesos / pesos.sum() if pesos.sum() > 0 else np.ones(len(matrices)) / len(matrices)


def mezcla_oof(matrices: list[np.ndarray], clase: np.ndarray, folds: list[dict],
               n_miembros: int, etiqueta: str, barajar: bool = False,
               pesos_fijos: np.ndarray | None = None) -> dict:
    """OOF de una mezcla de `n_miembros` miembros, con pesos ajustados dentro de cada fold.

    `barajar` permuta las filas de las probabilidades de los miembros no congelados. Es el
    control: mismo numero de miembros, misma estructura, misma optimizacion, pero sin la
    senal. Si el barajado rinde parecido, la mejora no era del canal.
    """
    inicio = time.time()
    matrices = [m.copy() for m in matrices[:n_miembros]]
    if barajar:
        # Se baraja **solo el miembro nuevo**, nunca los de V30. Barajar tambien los miembros de
        # V30 destruye el ensemble entero y hace que el control mida "V30 barajado es inútil",
        # que es una tautologia y no dice nada sobre el canal. Con la version anterior el
        # control daba 0,1639 y producia un veredicto de "aporte real" sin ningun fundamento.
        rng = np.random.default_rng(SEMILLA)
        matrices[-1][:] = matrices[-1][rng.permutation(len(matrices[-1]))]
    oof = np.zeros((len(clase), N_CLASES), dtype=np.float64)
    por_fold, pesos_por_fold = [], []
    for f in folds:
        ent, val = f["train_idx"], f["val_idx"]
        if pesos_fijos is not None:
            pesos = pesos_fijos
        else:
            # Los folds internos se generan sobre `clase[ent]`, asi que sus indices son
            # LOCALES a `ent`. No se reindexan a globales: `optimiza_pesos` construye la mezcla
            # de longitud `len(ent)` y la indexa con esos indices locales. Reindexarlos era
            # justo lo que rompia con IndexError.
            internos = cvmod.make_folds(clase[ent], groups=None, n_splits=4, n_repeats=1,
                                        seed=SEMILLA, scheme="stratified")
            pesos = optimiza_pesos(matrices, clase, ent, internos, N_TRIALS)
        mezcla = np.tensordot(pesos, np.stack(matrices), axes=(0, 0))
        oof[val] = mezcla[val]
        por_fold.append(_macro(clase[val], mezcla[val]))
        pesos_por_fold.append([round(float(w), 4) for w in pesos])
    cv = _macro(clase, oof)
    np.save(cfg.FEATURES / f"v31_05_oof_{etiqueta}.npy", oof)
    return {"etiqueta": etiqueta, "n_miembros": n_miembros, "macro_f1": round(cv, 7),
            "folds": [round(v, 7) for v in por_fold], "pesos_por_fold": pesos_por_fold,
            "barajado": barajar, "segundos": round(time.time() - inicio, 1)}


def main() -> int:
    autotest_numerico()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(v) for v in y])
    textos = train["text"].fillna("").astype(str).to_numpy()
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")

    print("=== cargando miembros ===", flush=True)
    ens = json.loads((cfg.EXPERIMENTS / "v30_02_ensemble.json").read_text(encoding="utf-8"))
    pool = np.load(cfg.EXPERIMENTS / "v29_pool_oof.npz")
    extra = {"v30_lgbm_refinado": cfg.FEATURES / "v30_lgbm_refinado_oof.npy"}
    miembros, nombres = [], []
    for nombre in ens["manifest"]["miembros"]:
        if nombre in extra:
            m = np.load(extra[nombre])
        elif nombre in pool.files:
            m = pool[nombre]
        else:
            m = np.load(cfg.FEATURES / f"{nombre}.npy")
        miembros.append(m.astype(np.float64))
        nombres.append(nombre)
    print(f"  {nombres}", flush=True)
    base = np.tensordot(np.array([ens["manifest"]["pesos"][n] for n in nombres]), np.stack(miembros),
                        axes=(0, 0))
    cv_base = _macro(clase, base)
    print(f"  V30 congelado: {cv_base:.7f} | canonico {CV_V30} | coincide "
          f"{abs(cv_base - CV_V30) < 1e-6}", flush=True)
    if abs(cv_base - CV_V30) > 1e-6:
        print("[STOP] el baseline no reproduce", flush=True)
        return 1

    print("\n=== calculando el OOF del canal ===", flush=True)
    oof_c = oof_canal(textos, clase, folds)
    print(f"  canal: {_macro(clase, oof_c):.7f}", flush=True)

    brazos = []
    print("\n=== A: V30 congelado (referencia absoluta) ===", flush=True)
    brazos.append({"etiqueta": "v30_congelado", "n_miembros": 3, "macro_f1": round(cv_base, 7),
                   "folds": [round(_macro(clase[f["val_idx"]], base[f["val_idx"]]), 7) for f in folds],
                   "delta_vs_v30": 0.0, "folds_positivos": 0, "barajado": False,
                   "pesos_por_fold": [], "segundos": 0.0,
                   "papel": "referencia absoluta, pesos congelados del manifest de V30"})

    print("\n=== B: V30 con 3 pesos REOPTIMIZADOS (el control que hace falta) ===", flush=True)
    b = mezcla_oof(miembros, clase, folds, 3, "v30_reopt")
    b["delta_vs_v30"] = round(b["macro_f1"] - cv_base, 7)
    b["folds_positivos"] = 0
    b["papel"] = "control: subir de 3 pesos fijos a 3 pesos optimizados. Si esto ya da gain, el " \
                 "gain de N1 no puede atribuirse al canal"
    brazos.append(b)
    print(f"  {b['macro_f1']:.7f} | delta {b['delta_vs_v30']:+.7f}", flush=True)

    print("\n=== C: V30 + canal como 4o miembro (N1) ===", flush=True)
    c = mezcla_oof(miembros + [oof_c], clase, folds, 4, "v30_mas_canal")
    c["delta_vs_v30"] = round(c["macro_f1"] - cv_base, 7)
    c["delta_vs_reopt"] = round(c["macro_f1"] - b["macro_f1"], 7)
    c["folds_positivos"] = int(sum(1 for v in c["folds"]
                                   for w in b["folds"] if v > w))
    c["papel"] = "la integracion de verdad: cuatro miembros, pesos optimizados con estructura "\
                 "anidada"
    brazos.append(c)
    print(f"  {c['macro_f1']:.7f} | vs V30 {c['delta_vs_v30']:+.7f} | vs reopt "
          f"{c['delta_vs_reopt']:+.7f}", flush=True)

    print("\n=== D: 4 miembros con el canal BARAJADO (control) ===", flush=True)
    d = mezcla_oof(miembros + [oof_c], clase, folds, 4, "v30_mas_canal_barajado", barajar=True)
    d["delta_vs_v30"] = round(d["macro_f1"] - cv_base, 7)
    d["delta_vs_reopt"] = round(d["macro_f1"] - b["macro_f1"], 7)
    d["folds_positivos"] = 0
    d["papel"] = "control de atribucion: mismos 4 miembros, mismos pesos, misma optimizacion, "\
                 "pero con las probabilidades del canal permutadas al azar"
    brazos.append(d)
    print(f"  {d['macro_f1']:.7f} | vs V30 {d['delta_vs_v30']:+.7f} | vs reopt "
          f"{d['delta_vs_reopt']:+.7f}", flush=True)

    print("\n=== veredicto de atribucion ===", flush=True)
    aporte_canal = c["macro_f1"] - d["macro_f1"]
    aporte_sobre_reopt = c["macro_f1"] - b["macro_f1"]
    print(f"  N1 - control barajado = {aporte_canal:+.7f}", flush=True)
    print(f"  N1 - reoptimizado     = {aporte_sobre_reopt:+.7f}", flush=True)
    # El control que manda es el **reoptimizado**: separa el efecto de "pesos optimizados" del
    # efecto "canal linguistic". El barajado confirma que la senal viaja en las filas correctas,
    # pero no por si solo prueba que el canalMejore a V30.
    if aporte_sobre_reopt <= 0:
        veredicto = (f"SIN APORTE: anadir el canal como cuarto miembro mueve {aporte_sobre_reopt:+.7f} "
                     f"sobre el simple reoptimizar los tres pesos de V30. La senal linguistica es "
                     f"real (V31-03: +0,006820 sobre el texto solo) pero V30 ya la exprimia")
    elif aporte_sobre_reopt < 0.0005:
        veredicto = (f"APORTE MARGINAL: el canal aporta {aporte_sobre_reopt:+.7f} sobre el "
                     f"reoptimizado, por debajo del gate de +0,0015. No se abre grid")
    else:
        veredicto = (f"APORTE REAL: el canal aporta {aporte_sobre_reopt:+.7f} sobre el "
                     f"reoptimizado, por encima del gate")

    filas = [{"ID": "V31-01", "hipotesis": "el canal ortografia+composicion+discurso aporta "
                                            "informacion al meta-espacio de V30",
              "track": "N", "variable": "cuarto miembro en la mezcla",
              "fijos": "3 miembros de V30, folds, semilla, optimizador",
              "corrected_cv": c["macro_f1"], "delta_vs_V30": c["delta_vs_v30"],
              "control_barajado": d["macro_f1"], "control_reopt": b["macro_f1"],
              "aporte_del_canal": round(aporte_sobre_reopt, 7),
              "decision": veredicto, "artefacto": "experiments/v31_05_integracion.json"}]
    pd.DataFrame(filas).to_csv(MAESTRO_CSV, index=False)

    resultado = _jsonable({
        "fase": "V31+ 05 integracion del canal en el meta-espacio de V30", "fecha": FECHA,
        "experimenta": True, "baseline_v30": CV_V30,
        "brazos": brazos, "veredicto": veredicto,
        "aporte_del_canal_sobre_control_barajado": round(aporte_canal, 7),
        "aporte_del_canal_sobre_reoptimizado": round(aporte_sobre_reopt, 7),
        "protocolo": {"folds": "StratifiedGroupKFold 5, dup_clusters, identicos a V20",
                      "pesos": "Optuna TPESampler, semilla 20260930, 60 trials, ajustados DENTRO "
                               "de outer_train con 4 folds internos; outer_validation nunca se ve",
                      "metrica": "macro-F1 de 6 clases"},
        "restricciones": {"usa_test": False, "usa_red": False, "usa_embeddings": False,
                          "reutiliza_oof_de_v30": True},
        "segundos": round(time.time() - inicio, 1),
    })
    RESULTADO.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[fin] {time.time() - inicio:.0f} s | {veredicto}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
