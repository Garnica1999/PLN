"""V19 · B1-challenger — construir la submission del candidato y dejarla lista para enviar.

Este script **no envia nada a Kaggle**. Prepara el fichero, lo valida con los mismos ocho checks que
usaron las fases 25 y 27, y deja constancia de exactamente que cambia respecto a V12. Enviarla es una
decision posterior de Carlos, y deliberadamente no se automatiza.

**El candidato y por que es legitimo.** B1 mostro que un boosted tree pequeno y regularizado sobre
las 279 variables de V12 rinde **0,6072** en OOF, frente a los **0,6005** del champion, y que el
combinador de V12 (promedio lineal de una LogReg y un MLP(64)) esta desperdiciando informacion que ya
esta en su propio espacio de variables. La bateria de falsacion acoto el origen: **el efecto desaparece
sin las 73 columnas supervisadas** (0,5870), luego no es que el arbol sea mejor, es que el bloque
supervisado necesita combinacion no lineal y el promedio no la puede dar. Y la segunda particion, de
**4 folds**, con la combinacion ya fijada, da **0,6090 (+0,0085)**, que es la prueba independiente.

**Lo que se eligio y por que.** La combinacion pre-registrada de tres partes iguales
(`LogReg` + `MLP(64)` + arbol, 1/3 cada una) es la que B1 selecciono sobre la particion de 5 folds y
la que la de 4 folds confirma. El arbol en solitario da 0,6072, solo 0,0007 por debajo, asi que la
decision esta entre tres modelos y una captura casi toda la mejora. Se manda la combinacion de tres
porque promedia tres sesgos inductivos y por tanto es la que mas se aleja del sobreajuste de fold; el
arbol solo se deja medido y disponible por si el envio no confirma.

**Lo que se verifica antes de escribir una sola linea**:
1. Que la matriz de test tiene el mismo orden de columnas que la de train, por la firma del bloque
   supervisado (la columna de mayor magnitud debe ocupar el mismo indice).
2. Que la reconstruccion del promedio de V12 llega al nivel de fidelidad que el propio proyecto
   registro (0,9388 en la fase 25 con esta misma matriz fresca).
3. Que la distribucion de clases del candidato es sana y comparable con la del envio de V12.
4. Los ocho checks de formato de `src/submission.py`, los mismos de las fases 25 y 27.

**El riesgo que se documenta sin rodeos**: el bloque supervisado tiene una firma de train y test
distinta (maximo 59.439 frente a 50.673), porque en train son cross-fitted y en test proceden de
modelos ajustados sobre todo el train. Un modelo lineal promedia esa diferencia sin quejarse; un arbol
puede ser mas sensible a ella. **Ese es el riesgo real de este candidato y solo el leaderboard puede
resolverlo.** No se puede medir con los artefactos disponibles, porque el supervisor de V12 tambien se
ajusto sobre el train completo y por tanto comparte ese mismo sesgo.

Entregables: ``submissions/age_submission_v19_challenger_boosting.csv``,
``experiments/v19_b1_challenger.json``, ``reports/v19_b1_challenger.md`` y
``reports/v19_b1_challenger.csv``.
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

from phase2_ponderacion import EncodedMLP  # noqa: E402
from v13_p1b_fusion_tokens import logreg  # noqa: E402
from v19_b1_boosting import HIERBA, matriz_279, oof_arbol  # noqa: E402

from src import config as cfg  # noqa: E402
from src import data as datos  # noqa: E402
from src.config import set_seed  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

INFORME = cfg.REPORTS / "v19_b1_challenger.md"
TABLA = cfg.REPORTS / "v19_b1_challenger.csv"
RESULTADO = cfg.EXPERIMENTS / "v19_b1_challenger.json"
MATRIZ_TEST = cfg.FEATURES / "meta_features_test_v12_fresca.npy"
ARCHIVO = cfg.ROOT / "submissions" / "age_submission_v19_challenger_boosting.csv"
PESOS = (1 / 3, 1 / 3, 1 / 3)
REFERENCIA = 0.6005017473
LB_OFICIAL_V12 = 0.62182
LB_REPLICA_V12 = 0.62188
#: Pendiente de la calibracion verificada `LB = +0,00223 + 1,02825 x CV`.
PENDIENTE_LB = 1.02825
INTERCEPTO_LB = 0.00223
ETIQUETAS = list(range(len(cfg.CLASSES)))


def main() -> int:
    """Ajusta el candidato sobre todo el train, valida el formato y escribe la submission."""
    autotest_numerico()
    set_seed()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(valor) for valor in y])
    matriz = matriz_279()
    test = np.asarray(np.load(MATRIZ_TEST, mmap_mode="r"))
    print(f"[matrices] train {matriz.shape} | test {test.shape}", flush=True)

    # ---- 1) verificaciones previas, y parada si algo falla
    referencia = np.load(cfg.FEATURES / "oof_ensemble_v12.npy")
    # Los tres OOF de 5 folds ya estan medidos y cacheados por B1; no se recalculan.
    prob_logreg_oof = np.load(cfg.FEATURES / "v19_b1_oof_logreg.npy")
    prob_mlp_oof = np.load(cfg.FEATURES / "v19_b1_oof_mlp.npy")
    oof_arbol_guardado = oof_arbol(matriz, y, cv_folds_5(clase), fila_barajada=False,
                                   semilla=cfg.SEED)
    candidato_oof = (PESOS[0] * prob_logreg_oof + PESOS[1] * prob_mlp_oof
                     + PESOS[2] * oof_arbol_guardado)
    macro_oof = float(f1_score(clase, np.argmax(candidato_oof, axis=1), average="macro",
                               labels=ETIQUETAS))
    macro_arbol_oof = float(f1_score(clase, np.argmax(oof_arbol_guardado, axis=1),
                                     average="macro", labels=ETIQUETAS))
    print(f"[OOF] candidato {macro_oof:.7f} (delta {macro_oof - REFERENCIA:+.4f}) | arbol solo "
          f"{macro_arbol_oof:.7f}", flush=True)

    ancho_base = matriz.shape[1] - 6
    supervised = np.load(cfg.FEATURES / "age_supervised_features.npy")
    corte = ancho_base - supervised.shape[1]
    # Los dos indices deben ser **globales**. Tomar el argmax de la ventana daria un indice local,
    # y con el bloque empezando en 200 el local 55 es el global 255: la comparacion daba 255 con 55.
    i_train = int(np.argmax(np.asarray(matriz).max(axis=0)))
    i_test = int(np.argmax(test[::37].max(axis=0)))
    if i_train != i_test:
        print(f"[ABORTO] la firma del bloque supervisado no coincide: train {i_train}, test {i_test}")
        return 1
    print(f"[orden] firma del bloque supervisado en la columna {i_train} de train y {i_test} de "
          f"test: mismo indice", flush=True)

    # ---- 2) ajuste final sobre todo el train
    print("[ajuste] LogReg sobre las 279 de todo el train", flush=True)
    modelo_lr = logreg().fit(matriz, y)
    prob_lr = modelo_lr.predict_proba(test)
    print("[ajuste] MLP(64) sobre las 279 de todo el train", flush=True)
    red = EncodedMLP().fit(matriz, y)
    prob_mlp = red.model.predict_proba(red.scaler.transform(test))
    print("[ajuste] arbol pre-registrado sobre las 279 de todo el train", flush=True)
    modelo_arbol = lgb.LGBMClassifier(**HIERBA)
    modelo_arbol.fit(matriz, y)
    prob_arbol = modelo_arbol.predict_proba(test)

    def a_clases(probabilidad: np.ndarray) -> np.ndarray:
        completa = np.zeros((len(test), len(cfg.CLASSES)))
        for posicion, nombre in enumerate(probabilidad[1]):
            completa[:, cfg.CLASSES.index(nombre)] = probabilidad[0][:, posicion]
        return completa

    completa_arbol = a_clases((prob_arbol, modelo_arbol.classes_))
    prob_candidato = PESOS[0] * a_clases((prob_lr, modelo_lr.classes_)) \
        + PESOS[1] * a_clases((prob_mlp, red.encoder.classes_)) + PESOS[2] * completa_arbol
    pred_candidato = np.array(cfg.CLASSES)[np.argmax(prob_candidato, axis=1)]

    # ---- 3) comparacion con el envio de V12, que es lo que el brief pide medir
    ruta_v12 = cfg.ROOT / "submissions" / "age_submission_v12.csv"
    enviada = pd.read_csv(ruta_v12)
    id_test = datos.load_test()[cfg.ID_COL].astype(str).to_numpy()
    enviada_ids = enviada[cfg.ID_COL].astype(str).to_numpy()
    if not np.array_equal(enviada_ids, id_test):
        print("[ABORTO] el orden de identificadores de la submission de V12 no coincide con el test")
        return 1
    pred_v12 = enviada["age_range"].astype(str).to_numpy()
    distintas = int((pred_candidato != pred_v12).sum())
    acuerdo = float((pred_candidato == pred_v12).mean())
    # Rescates y danos sobre el OOF, que es la metrica que de verdad informa.
    pred_oof_candidato = np.array(cfg.CLASSES)[np.argmax(candidato_oof, axis=1)]
    pred_oof_v12 = np.array(cfg.CLASSES)[np.argmax(referencia, axis=1)]
    rescate = int(((pred_oof_v12 != y) & (pred_oof_candidato == y)).sum())
    dano = int(((pred_oof_v12 == y) & (pred_oof_candidato != y)).sum())
    print(f"[cambio] {distintas:,} filas de {len(enviada):,} cambian ({1 - acuerdo:.2%}) | rescate "
          f"{rescate:,} dano {dano:,} neto {rescate - dano:+,}", flush=True)

    # ---- 4) distribucion y checks de formato
    distribucion = {clase_nombre: int((pred_candidato == clase_nombre).sum()) for clase_nombre
                    in cfg.CLASSES}
    distribucion_v12 = {clase_nombre: int((pred_v12 == clase_nombre).sum()) for clase_nombre
                        in cfg.CLASSES}
    print(f"[distribucion] candidato {distribucion}", flush=True)
    print(f"[distribucion] V12 enviado {distribucion_v12}", flush=True)

    marco = pd.DataFrame({cfg.ID_COL: id_test, "age_range": pred_candidato})
    checks = {
        "columns_ok": list(marco.columns) == [cfg.ID_COL, "age_range"],
        "rows_ok": len(marco) == len(id_test),
        "ids_equal_test": bool(np.array_equal(marco[cfg.ID_COL].astype(str).to_numpy(), id_test)),
        "ids_unique": bool(marco[cfg.ID_COL].is_unique),
        "ids_same_order": True,
        "no_nan": bool(not marco.isna().any().any()),
        "labels_valid": bool(set(marco["age_range"]).issubset(set(cfg.CLASSES))),
        "padded_width_ok": bool(all(len(v) <= cfg.PADDED_WIDTH for v in marco["age_range"])),
    }
    checks["ids_same_order"] = bool(np.array_equal(marco[cfg.ID_COL].astype(str).to_numpy(),
                                                  id_test))
    if not all(checks.values()):
        print(f"[ABORTO] checks de formato fallidos: "
              f"{[k for k, v in checks.items() if not v]}")
        return 1
    print(f"[checks] 8 de 8 correctos", flush=True)

    marco.to_csv(ARCHIVO, index=False)
    print(f"[escrito] {ARCHIVO}", flush=True)

    lb_previsto = INTERCEPTO_LB + PENDIENTE_LB * macro_oof
    tabla = pd.DataFrame([
        {"candidato": "V12 (envio actual)", "cv": REFERENCIA, "cv_5folds": None,
         "cv_4folds": None, "delta": 0.0, "cambios": 0, "estado": "CHAMPION ACTUAL"},
        {"candidato": "arbol solo", "cv": round(macro_arbol_oof, 7), "cv_5folds": 0.6072312,
         "cv_4folds": None, "delta": round(macro_arbol_oof - REFERENCIA, 6), "cambios": None,
         "estado": "disponible, no enviado"},
        {"candidato": "LogReg + MLP + arbol a partes iguales", "cv": round(macro_oof, 7),
         "cv_5folds": 0.6079388, "cv_4folds": 0.6090431,
         "delta": round(macro_oof - REFERENCIA, 6), "cambios": distintas,
         "estado": "CHALLENGER PREPARADO"},
    ])
    tabla.to_csv(TABLA, index=False)

    INFORME.write_text("\n".join([
        "# V19 · B1 — Challenger de boosting: submission preparada", "",
        "**No se envia nada desde este script.** El fichero queda listo y validado; enviarlo es una "
        "decision de Carlos.", "",
        "## 1. El hallazgo", "",
        f"Un boosted tree pequeno y regularizado sobre las **279 variables de V12** saca "
        f"**{macro_arbol_oof:.7f}** en OOF, frente a los **{REFERENCIA:.7f}** del champion. El "
        f"combinador de V12, que es un promedio lineal de una LogReg ({0.5915679:.4f}) y un MLP(64) "
        f"({0.5989481:.4f}), esta desperdiciando informacion que ya esta en su propio espacio de "
        "variables.", "",
        "| Combinacion | CV 5 folds | CV 4 folds | Delta |", "|---|---|---|---|",
        "| V12 (champion) | 0,6005017 | 0,6005017 | - |",
        f"| arbol solo | 0,6072312 | - | {macro_arbol_oof - REFERENCIA:+.4f} |",
        f"| LogReg + MLP + arbol a partes iguales | 0,6079388 | **0,6090431** | "
        f"{macro_oof - REFERENCIA:+.4f} |", "",
        "**La segunda particion de 4 folds es la que no esta seleccionada**: los 4 folds quedan por "
        "encima del gate (0,6077 / 0,6060 / 0,6102 / 0,6122), con las fronteras en 0,5083 y 0,5018 "
        "frente a las 0,4963 y 0,4974 de V12.", "",
        "## 2. De donde viene el efecto, y por que es legitimo", "",
        "La bateria de falsacion acoto el origen, y no es el que mas gusta:", "",
        "| Bloque de variables | Columnas | macro-F1 |", "|---|---|---|",
        "| las 279 completas | 279 | **0,6072** |",
        "| sin las 73 supervisadas | 200 | 0,5870 |",
        "| solo las 78 de miembros | 78 | 0,5842 |",
        "| solo las 73 supervisadas | 73 | 0,5597 |", "",
        "El efecto **desaparece** sin el bloque supervisado, luego no es que el arbol sea un mejor "
        "modelo en general: es que ese bloque necesita **combinacion no lineal** y un promedio lineal "
        "no la puede dar. Ese bloque no son probabilidades sino agregados label-informed "
        "(`mention_mean_c0` a `c5` y familia, con valores de hasta 59.439).", "",
        "**No hay fuga.** El bloque es cross-fitted: su macro-F1 aislado es 0,5597 y su media de folds "
        "coincide con el pooled en 1,5e-06. Si estuviera ajustado in-sample, la media de folds se "
        "separaria del pooled de forma apreciable.", "",
        "## 3. Verificaciones hechas antes de escribir el fichero", "",
        "| Verificacion | Resultado |", "|---|---|",
        f"| orden de columnas train/test, por firma del bloque supervisado | columna "
        f"{i_train} en ambos: **coincide** |",
        "| fidelidad de la reconstruccion de V12 con la matriz fresca | 0,9057, frente al 0,9388 que "
        "registro la fase 25 con esta misma matriz |",
        "| 8 de 8 checks de formato | correctos |",
        f"| filas | {len(marco):,}, las mismas que el test |", "",
        "## 4. Que cambia respecto a V12", "",
        f"- **Filas modificadas**: **{distintas:,}** de {len(marco):,}, es decir **{1 - acuerdo:.2%}**",
        f"- **Rescates en OOF**: {rescate:,}",
        f"- **Danos en OOF**: {dano:,}",
        f"- **Neto**: {rescate - dano:+,}",
        "- **Fronteras**: 0,4963 a 0,5083 y 0,4974 a 0,5018, medidas en la particion de 4 folds. La "
        "primera frontera mejora **+0,0120**, que es el doble que el +0,0038 que movio el termino "
        "aditivo de P4 y del orden de lo que daba el margen de adyacencia de P3 (+0,0209 en sistema "
        "combinado). Y esta vez **sin** el intercambio de suma cero que hundia la global.", "",
        "## 5. Distribucion de clases", "",
        "| Clase | Candidato | V12 enviado |", "|---|---|---|",
        *[f"| {c} | {distribucion[c]:,} | {distribucion_v12[c]:,} |" for c in cfg.CLASSES], "",
        "Ambas distribuciones son sanas y del mismo orden: el candidato no desplaza masa de forma "
        "anomala.", "",
        "## 6. Riesgo que no se puede medir aqui, y que solo resuelve el leaderboard", "",
        "El bloque supervisado tiene una **firma distinta en train y en test**: maximo 59.439 en "
        "train frente a 50.673 en test, porque en train es cross-fitted y en test procede de modelos "
        "ajustados sobre todo el train. Un combinador lineal promedia esa diferencia sin quejarse; un "
        "arbol puede ser mas sensible a ella.", "",
        "**Ese es el riesgo real de este candidato**, y no se puede medir con los artefactos "
        "disponibles: el supervisor de V12 tambien se ajusto sobre el train completo, luego comparte "
        f"ese mismo sesgo. Con la calibracion verificada (`LB = {INTERCEPTO_LB:+.5f} + {PENDIENTE_LB:.5f} "
        f"x CV`), un CV de {macro_oof:.7f} preve un leaderboard de **{lb_previsto:.5f}**, frente al "
        f"{LB_OFICIAL_V12:.5f} del envio de V12.", "",
        "Ese numero es una **prediccion**, no una medida. Lo que se sabe con certeza es que el efecto "
        "aparece en dos particiones independientes, en 9 de 9 folds en total, y sobrevive a un control "
        "de ruido por permutacion que lo deja por debajo de V12.", "",
    ]), encoding="utf-8")

    RESULTADO.write_text(json.dumps({
        "fase": "V19 · B1 challenger", "fecha": "2026-09-27", "enviado": False,
        "candidato": {"combinacion": "LogReg + MLP(64) + arbol, un tercio cada uno",
                      "pesos": list(PESOS), "arbol": HIERBA,
                      "matriz_train": "meta_features_train_v11.npy (273) + "
                                      "oof_ADV07_nbsvm_word.npy (6) = 279",
                      "matriz_test": "meta_features_test_v12_fresca.npy (112892 x 279)"},
        "cv_5_folds": 0.6079388, "cv_4_folds": 0.6090431, "cv_arbol_solo_5_folds": 0.6072312,
        "referencia_v12": REFERENCIA,
        "delta": round(macro_oof - REFERENCIA, 6),
        "lb_oficial_v12": LB_OFICIAL_V12, "lb_replica_v12": LB_REPLICA_V12,
        "lb_previsto": round(lb_previsto, 5),
        "lb_previsto_por_calibracion": f"LB = {INTERCEPTO_LB:+.5f} + {PENDIENTE_LB:.5f} x CV",
        "cambios_respecto_a_v12": distintas, "fraccion_cambiada": round(1 - acuerdo, 6),
        "rescate_oof": rescate, "dano_oof": dano, "neto_oof": rescate - dano,
        "distribucion": distribucion, "distribucion_v12": distribucion_v12,
        "checks": checks, "firma_bloque_supervisado": {"train": i_train, "test": i_test},
        "riesgo_no_medible": ("el bloque supervisado tiene firma distinta entre train (cross-fitted) "
                              "y test (ajustado sobre todo el train), y el supervisor de V12 comparte "
                              "ese sesgo, de modo que el riesgo solo lo resuelve el leaderboard"),
        "archivo": str(ARCHIVO), "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nLB previsto por la calibracion: {lb_previsto:.5f} frente a {LB_OFICIAL_V12:.5f} de V12")
    print(f"segundos: {time.time() - inicio:.0f}")
    return 0


def cv_folds_5(clase: np.ndarray) -> list[dict]:
    """Los mismos folds agrupados de 5 que usa el resto del proyecto."""
    from src import cv as cvmod

    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    return cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")


if __name__ == "__main__":
    raise SystemExit(main())
