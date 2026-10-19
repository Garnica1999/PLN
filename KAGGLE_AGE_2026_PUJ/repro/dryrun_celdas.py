"""Dry-run de las celdas restantes del notebook final (validacion de logica sin computo pesado).

Ejecuta las celdas del notebook EN ORDEN en un espacio compartido: las celdas de cabecera
(setup, resolucion de datos, modulos de la cadena, folds) corren de verdad; despues se SUSTITUYEN
las funciones pesadas por stubs que leen los artefactos locales del proyecto (o devuelven formas
correctas), y se ejecutan las celdas restantes celda a celda reportando errores.

Objetivo: cazar errores de logica/API en las ~40 celdas que aun no se han ejecutado en ningun
sitio, en minutos y en local, antes de la corrida completa en el servidor.

Uso (desde la raiz del proyecto):
    MPLBACKEND=Agg /home/carlos/venv/bin/python repro/dryrun_celdas.py
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
NOTEBOOK = RAIZ / "notebooks" / "kaggle_final_pipeline.ipynb"
FEATURES = RAIZ / "features"


def cargar_celdas() -> list[dict]:
    """Celdas de codigo del notebook, con su indice original."""
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return [(i, "".join(c["source"])) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]


def stub_heavies(espacio: dict) -> None:
    """Sustituye las funciones pesadas por versiones que leen artefactos locales."""
    import numpy as np
    import pandas as pd

    def _parquet_por_tamano(df, nombre_train, nombre_test):
        archivo = nombre_train if len(df) == 169336 else nombre_test
        return pd.read_parquet(FEATURES / archivo)

    def extraer_textometria_stub(df):
        return _parquet_por_tamano(df, "dense_train.parquet", "dense_test.parquet")

    def extraer_estructuradas_stub(df):
        return _parquet_por_tamano(df, "age_structured_features.parquet",
                                   "age_structured_features_test.parquet")

    def supervisadas_infold_stub(textos, y, fold_idx):
        return np.load(FEATURES / "age_supervised_features.npy")

    def supervisadas_test_stub(textos_train, y, textos_test):
        return np.zeros((len(textos_test), 73), dtype=np.float64)

    def construir_miembros_oof_stub(textos, y, fold_idx, dense):
        nombres = {"e013": "oof_E013_margenes.npy", "e016": "oof_E016_margenes.npy",
                   "e009": "oof_E-009.npy", "e011": "oof_E-011.npy", "e010": "oof_E-010.npy",
                   "e014": "oof_E014_margenes.npy", "e015_ordinal": "oof_E015_ordinal.npy",
                   "e015_softmax": "oof_E015_softmax.npy", "e017": "oof_E017_probabilidades.npy",
                   "e019b": "oof_E019b_densas_svd.npy", "e021a": "oof_E021a_ordinal.npy",
                   "e021b": "oof_E021b_rocchio.npy", "e028a": "oof_E028a_pares.npy",
                   "adv05": "oof_ADV05_nbsvm.npy"}
        return {clave: np.load(FEATURES / archivo) for clave, archivo in nombres.items()}

    def nb_svm_margins_stub(textos, y, train_idx, valid_idx, config):
        if np.asarray(valid_idx).max() >= 169336:
            return np.zeros((len(valid_idx), 6), dtype=np.float32)
        archivo = "oof_ADV05_nbsvm.npy" if "mentions" in config else "oof_ADV07_nbsvm_word.npy"
        return np.load(FEATURES / archivo)[valid_idx]

    def representaciones_test_stub(train, test):
        return {}

    def base_models_test_stub(reps, y, densa_tr, densa_te, textos_tr, textos_te):
        bloques = [np.zeros((112892, 6), dtype=np.float64) for _ in range(11)]
        bloques.append(np.zeros((112892, 1), dtype=np.float64))
        bloques.append(np.zeros((112892, 5), dtype=np.float64))
        return bloques

    def nb_word_test_stub(textos_train, y, textos_test):
        return np.zeros((len(textos_test), 6), dtype=np.float32)

    def v19_oofs_stub(matriz, y, clase, fold_idx):
        return (np.load(FEATURES / "v19_b1_oof_logreg.npy"),
                np.load(FEATURES / "v19_b1_oof_mlp.npy"),
                np.load(FEATURES / "v19_oof_arbol_5folds.npy"))

    def v20_cadena_oof_stub(matriz, clase, fold_idx):
        lr, mlp, arbol = espacio["v19_oofs_stub"](matriz, clase, clase, fold_idx)
        v20 = np.load(FEATURES / "v20_f3b_oof_candidato.npy")
        return 4.0 * v20 - lr - mlp - arbol

    def lgbm_refit_oof_stub(matriz, clase, fold_idx, configs, base=None):
        primero = configs[0].get("n_estimators")
        archivo = "v29_lgbm_optuna_oof.npy" if primero == 816 else "v30_lgbm_refinado_oof.npy"
        return np.load(FEATURES / archivo)

    def cuartos_v20_oof_stub(matriz, y, clase, fold_idx):
        return np.load(FEATURES / "v20_f3b_oof_candidato.npy")

    def lgbm_refit_oof_sin_es_stub(matriz, clase, y, fold_idx, configs):
        primero = configs[0].get("n_estimators")
        archivo = "v29_lgbm_optuna_oof.npy" if primero == 816 else "v30_lgbm_refinado_oof.npy"
        return np.load(FEATURES / archivo)

    def v20_test_stub(matriz, matriz_test, y, clase):
        return np.zeros((112892, 6), dtype=np.float64)

    def lgbm_refit_test_stub(matriz, clase, matriz_test, configs):
        return np.zeros((112892, 6), dtype=np.float64)

    def entrenar_componente_stub(nombre, X_train, X_test, y, fold_idx, dispositivo=None):
        pares = {"v38": ("v38_oof_tabularnet.npy", "v38_test_prob_tabularnet.npy"),
                 "v39_ordinal": ("v39_oof_tabnet_ordinal.npy", "v39_test_tabnet_ordinal.npy"),
                 "v40": ("v40_oof_se_tabularnet.npy", "v40_test_se_tabularnet.npy"),
                 "v41": ("v41_oof_swa_tabularnet.npy", "v41_test_swa_tabularnet.npy"),
                 "v42": ("v42_oof_smooth_swa_tabularnet.npy", "v42_test_smooth_swa_tabularnet.npy"),
                 "v43": ("v43_oof_cramer_swa_tabularnet.npy", "v43_test_cramer_swa_tabularnet.npy")}
        f_oof, f_test = pares[nombre]
        return np.load(FEATURES / f_oof), np.load(FEATURES / f_test)

    def hgb_componente_stub(X_train, X_test, y, fold_idx):
        return np.load(FEATURES / "v39_oof_hgb.npy"), np.load(FEATURES / "v39_test_hgb.npy")

    sustituciones = {
        "extraer_textometria": extraer_textometria_stub,
        "extraer_estructuradas": extraer_estructuradas_stub,
        "supervisadas_infold": supervisadas_infold_stub,
        "supervisadas_test": supervisadas_test_stub,
        "construir_miembros_oof": construir_miembros_oof_stub,
        "nb_svm_margins": nb_svm_margins_stub,
        "representaciones_test": representaciones_test_stub,
        "base_models_test": base_models_test_stub,
        "nb_word_test": nb_word_test_stub,
        "v19_oofs": v19_oofs_stub,
        "v20_cadena_oof": v20_cadena_oof_stub,
        "lgbm_refit_oof": lgbm_refit_oof_stub,
        "cuartos_v20_oof": cuartos_v20_oof_stub,
        "lgbm_refit_oof_sin_es": lgbm_refit_oof_sin_es_stub,
        "v20_test": v20_test_stub,
        "lgbm_refit_test": lgbm_refit_test_stub,
        "entrenar_componente": entrenar_componente_stub,
        "hgb_componente": hgb_componente_stub,
    }
    sustituciones["v19_oofs_stub"] = v19_oofs_stub
    espacio.update(sustituciones)


def main() -> int:
    """Ejecuta el dry-run completo."""
    celdas = cargar_celdas()
    indice_folds = None
    for posicion, (indice, texto) in enumerate(celdas):
        if "cargar_folds_embebidos" in texto:
            indice_folds = posicion
            break
    if indice_folds is None:
        print("no se encontro la celda de folds")
        return 2
    print(f"celdas de codigo: {len(celdas)} | cabeza hasta la posicion {indice_folds}")

    import numpy as np  # noqa: F401
    espacio: dict = {"__name__": "dryrun"}
    fallos: list[str] = []
    for posicion, (indice, texto) in enumerate(celdas):
        if posicion == indice_folds + 1:
            stub_heavies(espacio)
            print("--- stubs activados; ejecutando celdas restantes ---")
        try:
            exec(compile(texto, f"<celda {indice}>", "exec"), espacio)
        except Exception as error:  # noqa: BLE001
            resumen = f"celda {indice}: {type(error).__name__}: {error}"
            fallos.append(resumen)
            print(f"  [FALLA] {resumen}")
            traceback.print_exc(limit=3)
    print(f"\n=== dry-run: {len(celdas)} celdas, {len(fallos)} fallos ===")
    for fallo in fallos:
        print("  -", fallo)
    return 0 if not fallos else 2


if __name__ == "__main__":
    sys.exit(main())
