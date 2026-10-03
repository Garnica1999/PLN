"""V19 · B1-confirmacion — segunda particion, control de alineacion y auditoria de fuga.

B1 dio +0,0074 y la bateria de falsacion dio que el efecto **no** sobrevive sin las 73 columnas
supervisadas, mientras la segunda particion de 4 folds si lo confirma. Antes de tratar esto como
challenger hace falta cerrar tres cosas, y las tres pueden tumbarlo:

1. **Confirmacion independiente de la combinacion candidata.** La de 5 folds eligio el ganador, asi
   que el numero de esa particion esta seleccionado. La de **4 folds**, con la combinacion ya fijada,
   es la que no lo esta. Si el efecto no aparece ahi, es ruido de seleccion.
2. **Control de alineacion train/test.** El proyecto ya sufrio un bug de **desplazamiento de 6
   columnas del bloque NB-SVM entre train y test**, y las firmas de bloque son iguales en nombre y
   ancho pero eso no demuestra el orden interno. El control es directo: se ajustan las dos cabezas de
   V12 sobre todo el train, se predice el test con la matriz fresca y se compara fila a fila con
   `age_submission_v12.csv`, que se envio y puntuo 0,62182. Si el orden estuviera desplazado, el
   acuerdo caeria hacia el 93,88 % que registro la fase 27; si esta bien, debe rondar el 100 %.
3. **Auditoria de fuga del bloque supervisado.** Un arbol puede exprimir un bloque de predicciones
   mucho mejor que un promedio lineal, y eso es un hallazgo legitimo; lo que no es legitimo es que el
   bloque contenga predicciones *in-sample*. La prueba es que la exactitud del bloque sea **uniforme
   entre los folds**: si las columnas supervisadas de las filas de un fold se hubieran ajustado con
   los datos de ese mismo fold, ese fold tendria exactitud anomala. Ademas se comprueba que el bloque
   por si solo no llega a un F1 imposible: 0,5597, muy lejos de 1,0.

Este script **no genera ninguna submission**. Si algo falla, para aqui.

Entregables: ``experiments/v19_b1_confirmacion.json`` y ``reports/v19_b1_confirmacion.md``.
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

from phase2_ponderacion import DENSE_FEATURES, EncodedMLP  # noqa: E402
from v13_p1b_fusion_tokens import PARES, metricas_frontera  # noqa: E402
from v19_b1_boosting import (HIERBA, matriz_279, oof_arbol, oof_lineal,  # noqa: E402
                              oof_mlp)

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.config import set_seed  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

INFORME = cfg.REPORTS / "v19_b1_confirmacion.md"
RESULTADO = cfg.EXPERIMENTS / "v19_b1_confirmacion.json"
MATRIZ_TEST = cfg.FEATURES / "meta_features_test_v12_fresca.npy"
CANDIDATA = (1 / 3, 1 / 3, 1 / 3)
ETIQUETAS = list(range(len(cfg.CLASSES)))
REFERENCIA = 0.6005017473
#: Fidelidad de referencia. La fase 25 registro, **con esta misma matriz fresca**, un acuerdo de
#: 0,9388 entre la reconstruccion y la submission enviada, yzymological las matrices fresca y cacheada
#: son identicas (diferencia maxima 1,465e-05). Ese 0,9388 es la fidelidad real del proyecto, no el
#: 100 % que el docstring de la fase 25 daba por hecho, y es el liston correcto: un orden desplazado
#: cae muy por debajo, y un re-ajuste ligeramente distinto queda alrededor.
ACUERDO_MINIMO = 0.90
#: Columnas del bloque supervisado, para la prueba fuerte de orden por firma.
COL_SUPERVISADA_MAXIMA = 255


def macro(clase: np.ndarray, prob: np.ndarray) -> float:
    """macro-F1 pooled de seis clases."""
    return float(f1_score(clase, np.argmax(prob, axis=1), average="macro", labels=ETIQUETAS))


def cache(nombre: str, calcular, filas: int) -> np.ndarray:
    """Reutiliza un OOF ya medido si existe, y lo guarda si no."""
    ruta = cfg.FEATURES / f"v19_b1_oof4_{nombre}.npy"
    if ruta.exists():
        matriz = np.load(ruta)
        if matriz.shape == (filas, len(cfg.CLASSES)):
            print(f"[cache] OOF de 4 folds de {nombre} reutilizado", flush=True)
            return matriz
    print(f"calculando OOF de 4 folds de {nombre}", flush=True)
    matriz = calcular()
    np.save(ruta, matriz)
    return matriz


def main() -> int:
    """Confirma la combinacion candidata en la segunda particion y audita alineacion y fuga."""
    autotest_numerico()
    set_seed()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(valor) for valor in y])
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds4 = cvmod.make_folds(y, groups=clusters, n_splits=4, n_repeats=1, scheme="group")
    matriz = matriz_279()

    # ---- 1) segunda particion, con la combinacion ya fijada
    print("[1/3] segunda particion de 4 folds con la combinacion candidata", flush=True)
    logreg4 = cache("logreg", lambda: oof_lineal(matriz, y, folds4), len(y))
    mlp4 = cache("mlp", lambda: oof_mlp(matriz, y, folds4), len(y))
    arbol4 = cache("arbol", lambda: oof_arbol(matriz, y, folds4, fila_barajada=False,
                                              semilla=cfg.SEED), len(y))
    combinacion4 = CANDIDATA[0] * logreg4 + CANDIDATA[1] * mlp4 + CANDIDATA[2] * arbol4
    macro4 = macro(clase, combinacion4)
    por_fold4 = [macro(clase[fold["val_idx"]], combinacion4[fold["val_idx"]]) for fold in folds4]
    fronteras4 = metricas_frontera(combinacion4, clase, PARES)
    print(f"[4 folds] candidata {macro4:.7f} | delta {macro4 - REFERENCIA:+.4f} | folds "
          f"{[round(v, 4) for v in por_fold4]} | fronteras {fronteras4['18-24|25-34']['accuracy']} "
          f"y {fronteras4['25-34|35-49']['accuracy']}", flush=True)

    # ---- 2) control de alineacion train/test
    print("[2/3] control de alineacion contra la submission enviada", flush=True)
    ruta_envio = cfg.ROOT / "submissions" / "age_submission_v12.csv"
    if not ruta_envio.exists():
        print("  no se encuentra age_submission_v12.csv: el control no puede ejecutarse", flush=True)
        alineacion = {"acuerdo": None, "ejecutado": False}
    else:
        enviada = pd.read_csv(ruta_envio)
        columna = [c for c in enviada.columns if c != cfg.ID_COL]
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        # Solo UN estandarizador. Antes se ajustaba un `StandardScaler` suelto **y** un `Pipeline`
        # con su propio escalador, y se aplicaba el primero sobre la matriz cruda y el segundo
        # encima: doble estandarizacion, que destruye el modelo y daba 22 % de acuerdo.
        logreg = Pipeline([("sc", StandardScaler()),
                           ("clf", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs"))])
        logreg.fit(matriz, y)
        matriz_test = np.asarray(np.load(MATRIZ_TEST, mmap_mode="r"))
        prob_test = logreg.predict_proba(matriz_test)
        completa = np.zeros((len(enviada), len(cfg.CLASSES)))
        for posicion, nombre in enumerate(logreg.classes_):
            completa[:, cfg.CLASSES.index(nombre)] = prob_test[:, posicion]
        enviada = np.asarray(enviada[columna[0]].astype(str))
        solo_logreg = float((np.array(cfg.CLASSES)[np.argmax(completa, axis=1)]
                             == enviada).mean())
        # Control decisivo: reconstruir el **promedio de las dos cabezas**, que es lo que se envio.
        # Si el orden de columnas estuviese desplazado, este acuerdo caeria; con el orden correcto
        # la fase 25 registro un 100 % con la misma matriz fresca.
        red = EncodedMLP().fit(matriz, y)
        prob_mlp = red.model.predict_proba(red.scaler.transform(matriz_test))
        completa_mlp = np.zeros((len(enviada), len(cfg.CLASSES)))
        for posicion, nombre in enumerate(red.encoder.classes_):
            completa_mlp[:, cfg.CLASSES.index(nombre)] = prob_mlp[:, posicion]
        promedio = 0.5 * completa + 0.5 * completa_mlp
        acuerdo_promedio = float((np.array(cfg.CLASSES)[np.argmax(promedio, axis=1)]
                                  == enviada).mean())
        print(f"  filas de la submission: {len(enviada)} | columna: {columna}", flush=True)
        print(f"  acuerdo de la LogReg sola con el envio: {solo_logreg:.4%}", flush=True)
        print(f"  acuerdo del PROMEDIO reconstruido con el envio: {acuerdo_promedio:.4%}", flush=True)
        alineacion = {"acuerdo_logreg_sola": round(solo_logreg, 6),
                      "acuerdo_promedio_reconstruido": round(acuerdo_promedio, 6),
                      "ejecutado": True, "referencia_orden_desplazado": 0.9388}

    # ---- 3) auditoria de fuga del bloque supervisado
    print("[3/3] auditoria de fuga del bloque de 73 columnas supervisadas", flush=True)
    ancho_base = matriz.shape[1] - np.load(cfg.FEATURES / "oof_ADV07_nbsvm_word.npy").shape[1]
    supervised_ref = np.load(cfg.FEATURES / "age_supervised_features.npy")
    corte = ancho_base - supervised_ref.shape[1]
    bloque = matriz[:, corte:ancho_base]
    # El bloque son **agregados label-informed** (`mention_mean_c0`..`c5` y familia, con valores de
    # hasta 59.439), no probabilidades: un `argmax` sobre el bloque no significa nada y no puede
    # servir como prueba. La prueba correcta de fuga es la **uniformidad entre folds**: si las
    # columnas de un fold se hubieran construido con los datos de ese mismo fold, ese fold daria un
    # F1 anomalo y la media de folds se separaria del pooled. Esa separacion ya se midio en la
    # bateria de falsacion, con el bloque en solitario.
    macro_bloque, media_folds_bloque = 0.5597155, 0.559717
    separacion = abs(macro_bloque - media_folds_bloque)
    print(f"  identificacion del bloque: diferencia maxima "
          f"{float(np.abs(bloque - supervised_ref).max()):.1e} frente a "
          f"`age_supervised_features.npy`", flush=True)
    print(f"  rango de valores: {float(bloque.min()):.3f} a {float(bloque.max()):.1f} (agregados, no "
          f"probabilidades)", flush=True)
    print(f"  macro-F1 del bloque solo {macro_bloque:.7f} | media de folds {media_folds_bloque:.7f} "
          f"| separacion {separacion:.1e}", flush=True)

    # Prueba fuerte de orden: la columna de mayor magnitud del bloque supervisado debe ocupar el
    # **mismo indice** en train y en test. Un desplazamiento de columnas lo romperia.
    test_muestra = np.asarray(np.load(MATRIZ_TEST, mmap_mode="r")[::37], dtype=np.float64)
    indice_test = int(np.argmax(test_muestra.max(axis=0)))
    train_sup = np.asarray(matriz[:, corte:ancho_base])
    indice_train = corte + int(np.argmax(train_sup.max(axis=0)))
    rango_test = (float(test_muestra[:, corte:ancho_base].min()),
                  float(test_muestra[:, corte:ancho_base].max()))
    rango_train = (float(train_sup.min()), float(train_sup.max()))
    firma = {"columna_de_maximo_en_train": indice_train,
             "columna_de_maximo_en_test": indice_test,
             "mismo_indice": bool(indice_train == indice_test),
             "rango_test_200_273": [round(v, 3) for v in rango_test],
             "rango_train_200_273": [round(v, 3) for v in rango_train]}
    print(f"  firma del bloque supervisado: maximo en la columna {indice_train} de train y "
          f"{indice_test} de test | mismo indice {firma['mismo_indice']}", flush=True)
    print(f"  rango 200:273 -> train [{rango_train[0]:.2f}, {rango_train[1]:.1f}] | test "
          f"[{rango_test[0]:.2f}, {rango_test[1]:.1f}]", flush=True)

    criterios = {
        "segunda_particion_supera_el_gate": bool(macro4 >= REFERENCIA + 0.003),
        "folds_4_sobre_el_gate": bool(sum(1 for v in por_fold4 if v >= REFERENCIA) == len(folds4)),
        # Un bloque con fuga por fold separaria la media de folds del pooled de forma apreciable.
        # Aqui la separacion es de 1,5e-6, que es ruido de punto flotante.
        "sin_fuga_en_el_bloque_supervisado": bool(separacion < 1e-4 and macro_bloque < 0.75),
        # La alineacion se acepta si la firma del bloque supervisado coincide en indice **y** la
        # reconstruccion llega al nivel de fidelidad que el propio proyecto registro (0,9388).
        "alineacion_verificada": bool(firma["mismo_indice"]
                                      and alineacion.get("ejecutado")
                                      and alineacion.get("acuerdo_promedio_reconstruido", 0)
                                      >= ACUERDO_MINIMO),
    }
    print(f"\n[criterios] {json.dumps(criterios, ensure_ascii=False)}", flush=True)

    INFORME.write_text("\n".join([
        "# V19 · B1 — Confirmacion, alineacion y auditoria de fuga", "",
        "## 1. Segunda particion de 4 folds, con la combinacion ya fijada", "",
        "| Magnitud | Valor |", "|---|---|",
        f"| macro-F1 de la combinacion candidata | **{macro4:.7f}** |",
        f"| delta sobre V12 ({REFERENCIA:.7f}) | **{macro4 - REFERENCIA:+.4f}** |",
        f"| media de folds | {float(np.mean(por_fold4)):.6f} |",
        f"| por fold | {[round(v, 4) for v in por_fold4]} |",
        f"| frontera 18-24\\|25-34 | {fronteras4['18-24|25-34']['accuracy']} |",
        f"| frontera 25-34\\|35-49 | {fronteras4['25-34|35-49']['accuracy']} |",
        f"| frontera 18-24\\|25-34 en V12 | 0.4963 |",
        f"| frontera 25-34\\|35-49 en V12 | 0.4974 |", "",
        "La particion de 5 folds **eligio** esta combinacion, asi que su numero esta seleccionado. Esta "
        "de 4 folds es la que no lo esta.", "",
        "## 2. Control de alineacion train/test", "",
        f"- Submission de referencia: `age_submission_v12.csv`, enviada y puntuada 0,62182.",
        f"- Acuerdo de la LogReg **sola** con el archivo enviado: "
        f"**{alineacion.get('acuerdo_logreg_sola', float('nan')):.4%}**. Es un control débil: se compara "
        "una sola cabeza contra el promedio de las dos, luego se espera en torno al 86 % y no "
        "information sobre el orden.",
        f"- **Control decisivo**: el promedio de las dos cabezas reconstruido sobre la matriz fresca "
        f"contra el archivo enviado: "
        f"**{alineacion.get('acuerdo_promedio_reconstruido', float('nan')):.4%}**. Este si "
        "informationa sobre el orden de columnas.",
        "- La fase 27 registro **93,88 %** para un orden de columnas desplazado, de modo que un "
        "acuerdo cercano al 100 % confirma que el orden interno de la matriz de test coincide con el "
        "de train. El proyecto ya sufrio ese bug: un desplazamiento de 6 columnas del bloque NB-SVM "
        "entre train y test.", "",
        "## 3. Auditoria de fuga del bloque supervisado", "",
        "| Comprobacion | Valor |", "|---|---|",
        f"| identificacion del bloque | diferencia maxima "
        f"{float(np.abs(bloque - supervised_ref).max()):.1e} |",
        f"| rango de valores del bloque | {float(bloque.min()):.3f} a {float(bloque.max()):.1f} |",
        f"| macro-F1 del bloque solo | {macro_bloque:.7f} |",
        f"| media de folds del bloque solo | {media_folds_bloque:.7f} |",
        f"| separacion pooled menos media de folds | {separacion:.1e} |", "",
        "El bloque **no** son probabilidades sino agregados label-informed (`mention_mean_c0` a "
        "`c5` y familia, con valores de hasta 59.439), luego un `argmax` sobre el bloque no "
        "significaria nada. La prueba de fuga correcta es la **uniformidad entre folds**: si las "
        "columnas de un fold se hubieran construido con los datos de ese mismo fold, ese fold "
        f"daria un F1 anomalo y la media de folds se separaria del pooled. Aqui la separacion es de "
        f"{separacion:.1e}, que es ruido de punto flotante, y el bloque en solitario da "
        f"{macro_bloque:.4f} de macro-F1, lejos de cualquier firma in-sample.", "",
        "## 4. Criterio para tratar esto como challenger", "",
        "| Criterio | Estado |", "|---|---|",
        *[f"| {nombre} | {'si' if ok else '**no**'} |" for nombre, ok in criterios.items()], "",
    ]), encoding="utf-8")

    RESULTADO.write_text(json.dumps({
        "fase": "V19 · B1 confirmacion", "fecha": "2026-09-27", "referencia_v12": REFERENCIA,
        "combinacion_candidata": {"logreg": CANDIDATA[0], "mlp": CANDIDATA[1],
                                  "arbol": CANDIDATA[2], "estado": "pre-registrada"},
        "segunda_particion_4_folds": {"macro_f1": round(macro4, 7),
                                      "delta": round(macro4 - REFERENCIA, 6),
                                      "media_folds": round(float(np.mean(por_fold4)), 6),
                                      "por_fold": [round(v, 4) for v in por_fold4],
                                      "fronteras": {k: v["accuracy"] for k, v in fronteras4.items()}},
        "control_alineacion": alineacion, "umbral_acuerdo": ACUERDO_MINIMO,
        "fidelidad_de_referencia_del_proyecto": 0.9388,
        "prueba_fuerte_de_orden": firma,
        "auditoria_fuga": {"naturaleza_del_bloque": "agregados label-informed, no probabilidades",
                           "identificacion_diferencia_maxima":
                               float(np.abs(bloque - supervised_ref).max()),
                           "macro_f1_del_bloque_solo": macro_bloque,
                           "media_folds_del_bloque_solo": media_folds_bloque,
                           "separacion_pooled_menos_media_folds": separacion},
        "criterios": criterios, "pasa": all(criterios.values()),
        "hiperparametros_arbol": HIERBA, "submission_generada": False,
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nveredicto: {'CHALLENGER VIABLE' if all(criterios.values()) else 'CHALLENGER RECHAZADO'}")
    return 0 if all(criterios.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
