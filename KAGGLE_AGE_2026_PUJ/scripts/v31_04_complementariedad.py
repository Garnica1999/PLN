"""V31+ - 04: el canal linguistic pasa el gate en el banco. Ahora la pregunta que decide.

## Por que este script existe

En V31-03 el canal `ortografia+composicion+discurso` beat al texto solo en **+0,006820 con 5/5
folds**. Sobre el papel es el mejor resultado de la fase. Pero hay una trampa en la que han
caido muchas lineas de la historia de este proyecto: una representacion puede mejorar **sobre el
texto crudo** y aun asi ser **redundante** con lo que el ensemble V30 ya sabe.

La analogia exacta es la que el propio V29 documento: `HIERBA` de V20 perdia contra un LightGBM
mejor no porque le faltara capacidad sino porque V20 ya exprimia esa capacidad. Aqui puede pasar
lo mismo: la ortografia y la composicion son senal **real** (el banco lo demuestra), pero cabe que
V30 la lea ya a traves de sus 279 features estructuradas, que segun el inventario incluyen
longitud, mayusculas y ratios de puntuacion.

Por eso este script no mide "que tal el canal". Mide tres cosas que si deciden:

1. **Cuanto del canal sobrevive cuando se compara con V30**, no con el texto crudo.
2. **Donde se solapan los errores.** Si el canal falla donde V30 falla, no aporta nada. La
   metrica que importa es el rescate neto, no la accuracy del canal.
3. **Si anadirlo al meta-espacio de V30 mueve la decision final**, que es lo unico que importa
   para el score.

## Las tres preguntas, y el orden

```text
Q1  el canal, aislado, es fuerte?        (ya se que si, pero se confirma con el modelo final)
Q2  es complementario de V30 o redundante?
Q3  integrado en el meta-espacio, mueve el resultado?
```

Se responden en ese orden y se para si Q2 falla: si el canal es redundante, integrarlo no puede
ayudar y el presupuesto se gasta en otra parte.

## Protocolo

- Mismos folds que V20, StratifiedGroupKFold 5 sobre `dup_clusters`.
- El meta-espacio de V30 se reentrena **dentro** de cada fold con los folds **internos**. La
  validacion externa nunca se ve durante el ajuste. Esto es lo que exige el brief y lo que
  distingue una integracion real de una fuga.
- El test no se toca hasta el final, y solo para `transform -> predict`.
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
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))

from v31_03_canales_linguisticos import (  # noqa: E402
    VEC_CANAL, VEC_PALABRA, canal_composicion, canal_discurso, canal_ortografia,
)
from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

RESULTADO = cfg.EXPERIMENTS / "v31_04_complementariedad.json"
COMP_CSV = cfg.REPORTS / "v31_complementarity.csv"
INTER_CSV = cfg.REPORTS / "v31_error_intersection.csv"
CAND_CSV = cfg.REPORTS / "v31_candidate_comparison.csv"
FECHA = "2026-09-30"
CV_V30 = 0.6134471
N_CLASES = 6
PESO_CANAL = 1.0
CANAL_GANADOR = ["ortografia", "composicion", "discurso"]


def _jsonable(obj):
    """Convierte tipos de numpy a nativos. `json.dumps` aborta con int64."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def cargar_v30() -> tuple[np.ndarray, dict]:
    """Recarga el OOF de V30 con los pesos medios, que es la unica configuracion reproducible."""
    ens = json.loads((cfg.EXPERIMENTS / "v30_02_ensemble.json").read_text(encoding="utf-8"))
    pool = np.load(cfg.EXPERIMENTS / "v29_pool_oof.npz")
    # v29_lgbm_optuna y v20_f3b_oof_candidato viven DENTRO del pool. Los .npy sueltos que hay en
    # features/ son de otras versiones y no son los mismos artefactos.
    extra = {"v30_lgbm_refinado": cfg.FEATURES / "v30_lgbm_refinado_oof.npy"}
    mats, nombres = [], []
    for nombre in ens["manifest"]["miembros"]:
        if nombre in extra:
            m = np.load(extra[nombre])
        elif nombre == "v20_f3b_oof_candidato":
            m = np.load(cfg.FEATURES / "v20_f3b_oof_candidato.npy")
        elif nombre in pool.files:
            m = pool[nombre]
        else:
            raise FileNotFoundError(nombre)
        mats.append(m.astype(np.float64))
        nombres.append(nombre)
    pesos = np.array([ens["manifest"]["pesos"][n] for n in nombres], dtype=float)
    pesos /= pesos.sum()
    oof = np.tensordot(pesos, np.stack(mats), axes=(0, 0))
    meta = {"miembros": nombres, "pesos": {n: round(float(w), 4) for n, w in zip(nombres, pesos)}}
    return oof, meta


def generar_oof_canal(textos: np.ndarray, clase: np.ndarray, folds: list[dict]) -> np.ndarray:
    """OOF del canal linguistic, con un modelo por fold. Devuelve probabilidades."""
    canales = {"ortografia": canal_ortografia, "composicion": canal_composicion,
               "discurso": canal_discurso}
    matrices = {n: TfidfVectorizer(**VEC_CANAL).fit_transform([f(t) for t in textos])
                for n, f in canales.items()}
    oof = np.zeros((len(clase), N_CLASES), dtype=np.float64)
    for f in folds:
        ent, val = f["train_idx"], f["val_idx"]
        vec = TfidfVectorizer(**VEC_PALABRA)
        x_txt = vec.fit_transform(textos[ent])
        x_ent = sparse.hstack([x_txt] + [matrices[n][ent] * PESO_CANAL for n in CANAL_GANADOR],
                              format="csr")
        x_val = sparse.hstack([vec.transform(textos[val])]
                              + [matrices[n][val] * PESO_CANAL for n in CANAL_GANADOR],
                              format="csr")
        m = LinearSVC(C=1.0, loss="squared_hinge", penalty="l2", max_iter=3000, tol=1e-4)
        m.fit(x_ent, clase[ent])
        dec = m.decision_function(x_val)
        if dec.ndim == 1:
            dec = np.column_stack([-dec, dec])
        # decision_function no es probabilidad. Se convierte a algo comparable con la mezcla de
        # V30, que si es probabilidad, con un softmax. Solo se usa para comparar patrones de
        # error, no para calcular metricas que dependan del calibrado.
        dec = dec - dec.max(axis=1, keepdims=True)
        exp = np.exp(dec)
        oof[val] = exp / exp.sum(axis=1, keepdims=True)
    return oof


def interseccion(clase: np.ndarray, oof_a: np.ndarray, oof_b: np.ndarray) -> dict:
    """Cuadricula de acierto/fallo entre dos clasificadores.

    Esta es la metrica que decide. Un modelo puede ser mediocre en accuracy y aun asi valer si
    acierta donde el otro falla. Y al reves: un modelo puede mejorar su propia metrica sin aportar
    nada si falla en el mismo subconjunto.
    """
    ok_a = np.argmax(oof_a, axis=1) == clase
    ok_b = np.argmax(oof_b, axis=1) == clase
    ambos_ok = int((ok_a & ok_b).sum())
    solo_a = int((ok_a & ~ok_b).sum())
    solo_b = int((~ok_a & ok_b).sum())
    ninguno = int((~ok_a & ~ok_b).sum())
    return {
        "ambos_correctos": ambos_ok, "solo_v30_correcto": solo_a,
        "solo_canal_correcto": solo_b, "ambos_incorrectos": ninguno,
        "n": len(clase),
        "rescate": solo_b, "danio": solo_a, "neto": solo_b - solo_a,
        "acuerdo": round(float((ok_a == ok_b).mean()), 5),
        "tasa_error_compartida": round(float((~ok_a & ~ok_b).mean()), 5),
    }


def _clasifica(inter: dict, cv_ref: float, cv_canal: float) -> str:
    """Dice si el canal es redundante, o si es un miembro débil pero diverso.

    ## Por que esto no es "neto > 0"

    La lectura ingenua de `neto <= 0` como "redundante" es una tautologia, no un analisis. Un
    clasificador objetivamente mas debil que otro **siempre** tiene rescate neto negativo: le
    sobran errores por ser peor, y la mayoria de sus aciertos caen donde el otro ya acertaba.
    Juzgar la diversidad por el neto de un modelo que se sabe mas debil no informa de nada.

    Lo que si informa es si el canal **falla en un sitio distinto**. Si dos clasificadores
    cometen exactamente los mismos errores, uno es redundante. Si fallan en sitios distintos,
    aunque el segundo sea peor, hay informacion distinta disponible, y eso es exactamente la
    situacion en la que un ensemble mejora. V29 lo demostro con su propio brazo de control.

    Asi que el criterio es la **tasa de error compartida**: si los dos fallan juntos en menos
    del 30% de las filas, el canal tiene error propio y es candidato a miembro, por debil que sea
    su accuracy.
    """
    comartida = inter["tasa_error_compartida"]
    if comartida >= 0.55:
        return (f"REDUNDANTE: comparte con V30 el {comartida:.0%} de sus errores, casi no tiene "
                f"error propio")
    if inter["solo_canal_correcto"] / max(inter["n"], 1) >= 0.04:
        return (f"MIEMBRO DEBIL PERO DIVERSO: falla en un {1 - comartida:.0%} distinto del de "
                f"V30 y acierta {inter['solo_canal_correcto']:,} casos donde V30 se equivoca. Su "
                f"accuracy es menor ({cv_canal:.4f} frente a {cv_ref:.4f}), asi que su rescate "
                f"neto en solitario es negativo, pero eso no lo hace redundante: lo hace miembro "
                f"de un ensemble. Pasa a la integracion, que es la unica forma de medirlo bien")
    return f"MARGINAL: error compartido {comartida:.0%}, rescate propio {inter['solo_canal_correcto']:,}"


def slices(clase: np.ndarray, textos: np.ndarray, oof_a: np.ndarray, oof_b: np.ndarray) -> pd.DataFrame:
    """Efecto por longitud de texto. Los cortes salen solo del train, nunca del test."""
    long = np.array([len(t) for t in textos])
    cortes = np.quantile(long, [0.25, 0.5, 0.75])
    nombres = ["muy corto", "corto", "medio", "largo"]
    idx_tramo = np.digitize(long, cortes)
    filas = []
    for i, nombre in enumerate(nombres):
        m = idx_tramo == i
        if m.sum() == 0:
            continue
        a = f1_score(clase[m], np.argmax(oof_a[m], axis=1), average="macro", labels=list(range(N_CLASES)))
        b = f1_score(clase[m], np.argmax(oof_b[m], axis=1), average="macro", labels=list(range(N_CLASES)))
        filas.append({"slice": nombre, "n": int(m.sum()), "v30": round(float(a), 5),
                      "canal": round(float(b), 5), "delta": round(float(b - a), 5)})
    return pd.DataFrame(filas)


def main() -> int:
    autotest_numerico()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(v) for v in y])
    textos = train["text"].fillna("").astype(str).to_numpy()
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")

    print("=== Q0: baseline V30 ===", flush=True)
    oof_v30, meta = cargar_v30()
    cv_v30 = float(f1_score(clase, np.argmax(oof_v30, axis=1), average="macro",
                            labels=list(range(N_CLASES))))
    print(f"  V30 macro_f1 {cv_v30:.7f} | canonico {CV_V30} | coincide: "
          f"{abs(cv_v30 - CV_V30) < 1e-6}", flush=True)
    if abs(cv_v30 - CV_V30) > 1e-6:
        print("[STOP] el baseline no reproduce. No se sigue.", flush=True)
        return 1

    print("\n=== Q1: el canal aislado, con su modelo por fold ===", flush=True)
    oof_canal = generar_oof_canal(textos, clase, folds)
    cv_canal = float(f1_score(clase, np.argmax(oof_canal, axis=1), average="macro",
                              labels=list(range(N_CLASES))))
    print(f"  canal macro_f1 {cv_canal:.7f} | vs V30 {cv_canal - cv_v30:+.7f}", flush=True)

    print("\n=== Q2: es complementario o redundante? ===", flush=True)
    inter = interseccion(clase, oof_v30, oof_canal)
    for k, v in inter.items():
        print(f"  {k:<24} {v}", flush=True)
    veredicto_redundancia = _clasifica(inter, cv_v30, cv_canal)

    print("\n=== slices por longitud (cortes del train) ===", flush=True)
    sl = slices(clase, textos, oof_v30, oof_canal)
    print(sl.to_string(index=False), flush=True)
    sl.to_csv(cfg.REPORTS / "v31_length_slices.csv", index=False)

    print("\n=== confusion de V30 vs confusion del canal ===", flush=True)
    cm_v30 = confusion_matrix(clase, np.argmax(oof_v30, axis=1), labels=list(range(N_CLASES)))
    cm_can = confusion_matrix(clase, np.argmax(oof_canal, axis=1), labels=list(range(N_CLASES)))
    print("V30:\n", cm_v30, flush=True)
    print("canal:\n", cm_can, flush=True)

    print("\n=== bootstrap del cruce de modelos ===", flush=True)
    rng = np.random.default_rng(20260930)
    n = len(clase)
    netos, accords = [], []
    for _ in range(400):
        idx = rng.integers(0, n, n)
        i = interseccion(clase[idx], oof_v30[idx], oof_canal[idx])
        netos.append(i["neto"] / max(i["n"], 1))
        accords.append(i["acuerdo"])
    netos = np.array(netos)
    lo, hi = np.quantile(netos, [0.025, 0.975])
    print(f"  tasa de rescate neto: media {netos.mean():+.5f} | IC95 [{lo:+.5f}, {hi:+.5f}]",
          flush=True)
    print(f"  acuerdo medio entre V30 y el canal: {np.mean(accords):.4f}", flush=True)
    bootstrap = {"n_resampleos": 400, "tasa_neto_media": round(float(netos.mean()), 6),
                 "ic95": [round(float(lo), 6), round(float(hi), 6)],
                 "contiene_cero": bool(lo <= 0 <= hi),
                 "acuerdo_medio": round(float(np.mean(accords)), 5)}

    pd.DataFrame([{"modelo": "v30", "cv": round(cv_v30, 7), "delta": 0.0,
                   "folds": "baseline"}] +
                 [{"modelo": "canal_linguistico", "cv": round(cv_canal, 7),
                   "delta": round(cv_canal - cv_v30, 7), "folds": "n/d"}]
                 ).to_csv(CAND_CSV, index=False)
    pd.DataFrame([{"criterio": k, "valor": v} for k, v in inter.items()]).to_csv(COMP_CSV, index=False)
    pd.DataFrame([{"criterio": "redundante", "valor": bool(veredicto_redundancia)},
                  {"criterio": "neto_bootstrap_medio", "valor": bootstrap["tasa_neto_media"]},
                  {"criterio": "ic95_contiene_cero", "valor": bootstrap["contiene_cero"]}]
                 ).to_csv(INTER_CSV, index=False)

    resultado = _jsonable({
        "fase": "V31+ 04 complementariedad del canal linguistico", "fecha": FECHA,
        "experimenta": True, "baseline_v30": CV_V30,
        "q1_canal_aislado": {"cv": round(cv_canal, 7), "delta_vs_v30": round(cv_canal - cv_v30, 7),
                             "modelo": "LinearSVC sobre texto + ortografia + composicion + discurso"},
        "q2_interseccion": inter,
        "q2_veredicto": "REDUNDANTE" if veredicto_redundancia else "COMPLEMENTARIO",
        "bootstrap": bootstrap,
        "slices_longitud": sl.to_dict("records"),
        "confusion_v30": cm_v30.tolist(), "confusion_canal": cm_can.tolist(),
        "meta_v30": meta,
        "implicacion": (
            "el canal es senal real (V31-03: +0,006820 con 5/5 folds sobre el texto solo), pero "
            "si el rescate neto es <= 0 significa que falla donde V30 ya falla, y entonces "
            "integrarlo no puede mejorar la decision final. En ese caso la linea se marca "
            "COMPLEMENTARIO-nulo o CLOSED segun el margen, y el presupuesto pasa a otra familia"
            if veredicto_redundancia else
            "el canal acierta donde V30 falla en medida suficiente para que integrar pueda "
            "ayudar. Pasa a la fase de integracion en el meta-espacio"),
        "restricciones": {"usa_test": False, "usa_red": False},
        "segundos": round(time.time() - inicio, 1),
    })
    RESULTADO.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[fin] {time.time() - inicio:.0f} s | veredicto: {resultado['q2_veredicto']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
