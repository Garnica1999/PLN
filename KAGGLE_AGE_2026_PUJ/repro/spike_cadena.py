"""Spike cadena - verifica la reconstruccion completa del campeon contra los artefactos guardados.

Etapas (seleccionables con --etapas; cache por etapa en --cache):
  features  -> dense/estructuradas/supervisadas (train y test) vs parquet/npy guardados
  members   -> 14 OOF de miembros vs features/oof_*.npy
  meta      -> meta 273 + ADV07 + matriz 279 vs meta_features_train_v11 / oof_ADV07
  test      -> lado test (representaciones + 13 modelos + ADV05/ADV07) vs meta_features_test_v12_fresca
  v30       -> V20/V29/V30 (OOF y test) + plug-in vs features/*.npy y age_plugin_test_prob.npy
  nn        -> componentes V38-V43 + HGB vs features/v4*.npy / v39_*
  sub       -> submission V48 (PoE + plug-in) vs submissions/age_submission_v48_geometric_poe_blend.csv

NO escribe artefactos del proyecto: cache en disco aparte y solo imprime comparaciones. ASCII puro.

Uso:
    python repro/spike_cadena.py --etapas features,members,meta
    python repro/spike_cadena.py --etapas test,v30
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd

import sys

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "repro"))

from cadena import comun  # noqa: E402
from cadena import dense as dense_mod  # noqa: E402
from cadena import structured as structured_mod  # noqa: E402
from cadena import supervised as supervised_mod  # noqa: E402
from cadena import meta as meta_mod  # noqa: E402
from cadena import nb_svm  # noqa: E402
from cadena import test_side  # noqa: E402
from cadena import v30 as v30_mod  # noqa: E402

FEATURES = RAIZ / "features"
CACHE_DEFECTO = Path.home() / "age_notebook" / "runs" / "cadena_cache"


def comparar(nombre: str, recon: np.ndarray, ref: np.ndarray) -> None:
    """Imprime el diagnostico de diferencias entre reconstruccion y referencia."""
    recon = np.asarray(recon)
    ref = np.asarray(ref)
    if recon.shape != ref.shape:
        print(f"[cadena] {nombre}: FORMA DISTINTA recon={recon.shape} ref={ref.shape}")
        return
    dif = np.abs(recon.astype(np.float64) - ref.astype(np.float64))
    linea = (f"[cadena] {nombre}: max|d|={dif.max():.3e} media|d|={dif.mean():.3e} "
             f"dtype recon={recon.dtype} ref={ref.dtype}")
    if recon.ndim == 2 and recon.shape[1] > 1:
        ac = float((np.argmax(recon, axis=1) == np.argmax(ref, axis=1)).mean())
        linea += f" argmax igual={ac:.8f}"
    print(linea, flush=True)


def guardar(cache: Path, nombre: str, **arrays: np.ndarray) -> None:
    """Guarda arrays de una etapa en la cache del spike."""
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache / f"{nombre}.npz", **arrays)


def cargar(cache: Path, nombre: str) -> dict[str, np.ndarray]:
    """Carga los arrays de una etapa desde la cache del spike."""
    with np.load(cache / f"{nombre}.npz", allow_pickle=False) as datos:
        return {clave: datos[clave] for clave in datos.files}


#: Posiciones de las 3 columnas de emoji dentro de las 38 densas (TEXTOMETRY + STYLE).
INDICES_EMOJI = (25, 26, 27)
#: Nombres de las 35 columnas densas usadas por el meta (sin las 3 de emoji).
DENSE35_COLS = [c for c in (dense_mod.TEXTOMETRY_FEATURES + dense_mod.STYLE_FEATURES)
                if c not in ("n_emoji", "emoji_diversity", "has_emoji")]


def densa35(densa38: np.ndarray) -> np.ndarray:
    """Selecciona las 35 columnas densas usadas por el meta (excluye las 3 de emoji)."""
    return np.delete(densa38, INDICES_EMOJI, axis=1)


def etapa_features(cache: Path) -> None:
    """Dense, estructuradas y supervisadas (train y test) vs los artefactos guardados."""
    train = comun.cargar_train(RAIZ)
    test = comun.cargar_test(RAIZ)
    y = train["age_range"].astype(str).to_numpy()
    fold_idx = comun.cargar_folds(RAIZ)

    t0 = time.time()
    densa = dense_mod.extraer_textometria(train)
    densa_te = dense_mod.extraer_textometria(test)
    comparar("dense train (38)", densa.to_numpy(float), pd.read_parquet(FEATURES / "dense_train.parquet").to_numpy(float))
    comparar("dense test  (38)", densa_te.to_numpy(float), pd.read_parquet(FEATURES / "dense_test.parquet").to_numpy(float))
    print(f"[cadena] dense: {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    estr = structured_mod.extraer_estructuradas(train)
    estr_te = structured_mod.extraer_estructuradas(test)
    comparar("estructuradas train (87)", estr.to_numpy(float),
             pd.read_parquet(FEATURES / "age_structured_features.parquet").to_numpy(float))
    comparar("estructuradas test  (87)", estr_te.to_numpy(float),
             pd.read_parquet(FEATURES / "age_structured_features_test.parquet").to_numpy(float))
    print(f"[cadena] estructuradas: {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    sup = supervised_mod.supervisadas_infold(train["text"], y, fold_idx)
    sup_te = supervised_mod.supervisadas_test(train["text"], y, test["text"])
    comparar("supervisadas train (73)", sup, np.load(FEATURES / "age_supervised_features.npy"))
    print(f"[cadena] supervisadas: {time.time() - t0:.1f}s", flush=True)

    guardar(cache, "features",
            densa=densa.to_numpy(np.float64), densa_te=densa_te.to_numpy(np.float64),
            estr=estr.to_numpy(np.float64), estr_te=estr_te.to_numpy(np.float64),
            sup=sup, sup_te=sup_te)


def etapa_members(cache: Path) -> None:
    """Los 14 OOF de miembros vs los .npy guardados."""
    train = comun.cargar_train(RAIZ)
    y = train["age_range"].astype(str).to_numpy()
    fold_idx = comun.cargar_folds(RAIZ)
    datos = cargar(cache, "features")
    densa_df = pd.DataFrame(densa35(datos["densa"]), columns=DENSE35_COLS)
    oofs = meta_mod.construir_miembros_oof(train["text"], y, fold_idx, densa_df)
    referencia = {
        "e013": "oof_E013_margenes.npy", "e016": "oof_E016_margenes.npy",
        "e009": "oof_E-009.npy", "e011": "oof_E-011.npy", "e010": "oof_E-010.npy",
        "e014": "oof_E014_margenes.npy", "e015_ordinal": "oof_E015_ordinal.npy",
        "e015_softmax": "oof_E015_softmax.npy", "e017": "oof_E017_probabilidades.npy",
        "e019b": "oof_E019b_densas_svd.npy", "e021a": "oof_E021a_ordinal.npy",
        "e021b": "oof_E021b_rocchio.npy", "e028a": "oof_E028a_pares.npy",
        "adv05": "oof_ADV05_nbsvm.npy",
    }
    for nombre, fichero in referencia.items():
        comparar(f"miembro {nombre}", oofs[nombre], np.load(FEATURES / fichero))
    guardar(cache, "members", **oofs)


def etapa_meta(cache: Path) -> None:
    """Meta 273 + ADV07 + matriz 279 vs los artefactos guardados."""
    train = comun.cargar_train(RAIZ)
    y = train["age_range"].astype(str).to_numpy()
    fold_idx = comun.cargar_folds(RAIZ)
    datos = cargar(cache, "features")
    oofs = cargar(cache, "members")

    adv07 = np.zeros((len(y), 6), dtype=np.float32)
    for fold in range(5):
        tr = np.flatnonzero(fold_idx != fold)
        va = np.flatnonzero(fold_idx == fold)
        adv07[va] = nb_svm.nb_svm_margins(train["text"], y, tr, va, comun.WORD)
    comparar("ADV07 word", adv07, np.load(FEATURES / "oof_ADV07_nbsvm_word.npy"))

    meta273 = meta_mod.build_features_273(oofs, densa35(datos["densa"]), datos["estr"], datos["sup"])
    comparar("meta 273", meta273, np.load(FEATURES / "meta_features_train_v11.npy"))

    matriz = meta_mod.matriz_279(meta273, adv07)
    print(f"[cadena] matriz 279: forma={matriz.shape} dtype={matriz.dtype}", flush=True)
    guardar(cache, "meta", meta273=meta273, adv07=adv07, matriz=matriz)


def _adv05_test(train_text, y, test_text) -> np.ndarray:
    """ADV05 lado test: NB-SVM char+menciones ajustado sobre todo el train (phase3:121-128)."""
    textos_todo = pd.concat([train_text, test_text], ignore_index=True)
    y_todo = np.concatenate([y, np.array(["13-17"] * len(test_text))])
    tr = np.arange(len(train_text))
    va = np.arange(len(train_text), len(train_text) + len(test_text))
    config = {**comun.CHAR_300K, "mentions": {"config": comun.MENTION, "pattern": r"@\w+"}}
    return nb_svm.nb_svm_margins(textos_todo, y_todo, tr, va, config)


def etapa_test(cache: Path) -> None:
    """Lado test completo vs meta_features_test_v12_fresca.npy."""
    train = comun.cargar_train(RAIZ)
    test = comun.cargar_test(RAIZ)
    y = train["age_range"].astype(str).to_numpy()
    datos = cargar(cache, "features")
    densa_te = pd.DataFrame(datos["densa_te"])

    reps = test_side.representaciones_test(train, test)
    bloques = test_side.base_models_test(reps, y, densa35(datos["densa"]), densa35(datos["densa_te"]),
                                         train["text"], test["text"])
    nb_test = _adv05_test(train["text"], y, test["text"])
    nb_word = test_side.nb_word_test(train["text"], y, test["text"])
    matriz_test = test_side.matriz_test_fresca(bloques, nb_test, densa35(datos["densa_te"]),
                                               datos["estr_te"], datos["sup_te"], nb_word)
    comparar("matriz test 279", matriz_test, np.load(FEATURES / "meta_features_test_v12_fresca.npy"))
    guardar(cache, "test", matriz_test=matriz_test, nb_test=nb_test, nb_word=nb_word)


def etapa_v30(cache: Path) -> None:
    """V20/V29/V30 (OOF y test) + plug-in vs los artefactos guardados."""
    train = comun.cargar_train(RAIZ)
    y = train["age_range"].astype(str).to_numpy()
    clase = np.array([comun.CLASSES.index(v) for v in y])
    fold_idx = comun.cargar_folds(RAIZ)
    datos_meta = cargar(cache, "meta")
    matriz = datos_meta["matriz"]
    matriz_test = cargar(cache, "test")["matriz_test"]

    v20 = v30_mod.v20_oof(matriz, y, clase, fold_idx)
    comparar("V20 OOF", v20, np.load(FEATURES / "v20_f3b_oof_candidato.npy"))
    v29 = v30_mod.lgbm_refit_oof(matriz, clase, fold_idx, v30_mod.V29_CONFIGS,
                                 base={"random_state": comun.SEED})
    comparar("V29 OOF", v29, np.load(FEATURES / "v29_lgbm_optuna_oof.npy"))
    v30 = v30_mod.lgbm_refit_oof(matriz, clase, fold_idx, v30_mod.V30_CONFIGS)
    comparar("V30 OOF", v30, np.load(FEATURES / "v30_lgbm_refinado_oof.npy"))

    oof_v30 = v30_mod.mezcla(v20, v29, v30)
    comparar("V30 blend OOF", oof_v30, np.load(RAIZ / "artifacts" / "p33_scores" / "todos" / "v30_prob.npy"))
    cv_v30 = v30_mod.macro_f1(clase, np.argmax(oof_v30, axis=1))
    print(f"[cadena] corrected_cv V30 reconstruido = {cv_v30:.7f} (documentado 0.6134471)", flush=True)
    cv_plug = v30_mod.macro_f1(clase, v30_mod.regla_plug_in(oof_v30))
    print(f"[cadena] plug-in OOF reconstruido = {cv_plug:.10f} (documentado 0.6159732487657944)",
          flush=True)

    # Segunda particion (semilla 987654): linaje V30 fuera de la canonica.
    from sklearn.model_selection import StratifiedGroupKFold
    clusters = pd.read_parquet(FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[: len(y)]
    divisor = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=987654)
    fold_idx2 = np.zeros(len(y), dtype=np.int64)
    for numero, (_, validacion) in enumerate(divisor.split(np.zeros(len(y)), y, groups=clusters)):
        fold_idx2[validacion] = numero
    v20s = v30_mod.cuartos_v20_oof(matriz, y, clase, fold_idx2)
    v29s = v30_mod.lgbm_refit_oof_sin_es(matriz, clase, y, fold_idx2, v30_mod.V29_CONFIGS)
    v30s = v30_mod.lgbm_refit_oof_sin_es(matriz, clase, y, fold_idx2, v30_mod.V30_CONFIGS)
    oof_segunda = v30_mod.mezcla(v20s, v29s, v30s)
    comparar("V30 segunda OOF", oof_segunda,
             np.load(FEATURES / "age_plugin_segunda_particion_oof.npy"))
    print(f"[cadena] V30 segunda: {v30_mod.macro_f1(clase, np.argmax(oof_segunda, axis=1)):.7f} "
          f"(doc 0.6144120) | plug-in: {v30_mod.macro_f1(clase, v30_mod.regla_plug_in(oof_segunda)):.7f} "
          f"(doc 0.6164700)", flush=True)

    t0 = time.time()
    p20 = v30_mod.v20_test(matriz, matriz_test, y, clase)
    p29 = v30_mod.lgbm_refit_test(matriz, clase, matriz_test, v30_mod.V29_CONFIGS)
    p30 = v30_mod.lgbm_refit_test(matriz, clase, matriz_test, v30_mod.V30_CONFIGS)
    test_v30 = v30_mod.mezcla(p20, p29, p30)
    print(f"[cadena] V20/V29/V30 test: {time.time() - t0:.1f}s", flush=True)
    comparar("age_plugin_test_prob", test_v30, np.load(FEATURES / "age_plugin_test_prob.npy"))
    guardar(cache, "v30", v20_oof=v20, v29_oof=v29, v30_oof=v30, oof_blend=oof_v30,
            test_blend=test_v30, oof_segunda=oof_segunda)


def etapa_nn(cache: Path) -> None:
    """Componentes V38-V43 + HGB vs los artefactos guardados."""
    from cadena import neuronales  # noqa: PLC0415

    train = comun.cargar_train(RAIZ)
    y = train["age_range"].astype(str).to_numpy()
    clase = np.array([comun.CLASSES.index(v) for v in y])
    fold_idx = comun.cargar_folds(RAIZ)
    datos_meta = cargar(cache, "meta")
    matriz = datos_meta["matriz"]
    matriz_test = cargar(cache, "test")["matriz_test"]

    componentes = {
        "v38": ("v38_oof_tabularnet.npy", "v38_test_prob_tabularnet.npy"),
        "v39_ordinal": ("v39_oof_tabnet_ordinal.npy", "v39_test_tabnet_ordinal.npy"),
        "v40": ("v40_oof_se_tabularnet.npy", "v40_test_se_tabularnet.npy"),
        "v41": ("v41_oof_swa_tabularnet.npy", "v41_test_swa_tabularnet.npy"),
        "v42": ("v42_oof_smooth_swa_tabularnet.npy", "v42_test_smooth_swa_tabularnet.npy"),
        "v43": ("v43_oof_cramer_swa_tabularnet.npy", "v43_test_cramer_swa_tabularnet.npy"),
    }
    resultados: dict[str, np.ndarray] = {}
    for nombre, (f_oof, f_test) in componentes.items():
        t0 = time.time()
        oof, test = neuronales.entrenar_componente(nombre, matriz, matriz_test, clase, fold_idx)
        print(f"[cadena] {nombre}: {time.time() - t0:.1f}s", flush=True)
        comparar(f"{nombre} OOF", oof, np.load(FEATURES / f_oof))
        comparar(f"{nombre} TEST", test, np.load(FEATURES / f_test))
        resultados[f"{nombre}_oof"] = oof
        resultados[f"{nombre}_test"] = test

    t0 = time.time()
    hgb_oof, hgb_test = neuronales.hgb_componente(matriz, matriz_test, clase, fold_idx)
    print(f"[cadena] v39_hgb: {time.time() - t0:.1f}s", flush=True)
    comparar("v39_hgb OOF", hgb_oof, np.load(FEATURES / "v39_oof_hgb.npy"))
    comparar("v39_hgb TEST", hgb_test, np.load(FEATURES / "v39_test_hgb.npy"))
    resultados["v39_hgb_oof"] = hgb_oof
    resultados["v39_hgb_test"] = hgb_test

    # V48: evaluacion publicada (miembro V30 = OOF de la segunda particion) y variante canonica.
    pesos = np.array([0.5628, 0.0832, 0.1724, 0.0754, 0.0513, 0.0549])
    pesos = pesos / pesos.sum()

    def _poe(modelos):
        log_geom = sum(w * np.log(np.clip(m, 1e-12, 1.0)) for w, m in zip(pesos, modelos))
        p_geom = np.exp(log_geom - np.max(log_geom, axis=1, keepdims=True))
        return p_geom / p_geom.sum(axis=1, keepdims=True)

    datos_v30 = cargar(cache, "v30")
    componentes = [resultados["v43_oof"], resultados["v42_oof"], resultados["v40_oof"],
                   resultados["v39_ordinal_oof"], resultados["v39_hgb_oof"]]
    v48_publicada = v30_mod.regla_plug_in(_poe([datos_v30["oof_segunda"], *componentes]))
    v48_canonica = v30_mod.regla_plug_in(_poe([datos_v30["oof_blend"], *componentes]))
    print(f"[cadena] V48 publicado (doc 0.6186451): {v30_mod.macro_f1(clase, v48_publicada):.7f}",
          flush=True)
    print(f"[cadena] V48 variante canonica       : {v30_mod.macro_f1(clase, v48_canonica):.7f}",
          flush=True)
    guardar(cache, "nn", **resultados)


def etapa_sub(cache: Path) -> None:
    """Submission V48 (PoE geometrico + plug-in) vs el fichero sellado."""
    train = comun.cargar_train(RAIZ)
    test = comun.cargar_test(RAIZ)
    y = train["age_range"].astype(str).to_numpy()
    clase = np.array([comun.CLASSES.index(v) for v in y])

    datos_nn = cargar(cache, "nn")
    test_v30 = cargar(cache, "v30")["test_blend"]

    pesos = np.array([0.5628, 0.0832, 0.1724, 0.0754, 0.0513, 0.0549])
    pesos = pesos / pesos.sum()
    modelos = [test_v30, datos_nn["v43_test"], datos_nn["v42_test"], datos_nn["v40_test"],
               datos_nn["v39_ordinal_test"], datos_nn["v39_hgb_test"]]
    eps = 1e-12
    log_geom = sum(w * np.log(np.clip(m, eps, 1.0)) for w, m in zip(pesos, modelos))
    p_geom = np.exp(log_geom - np.max(log_geom, axis=1, keepdims=True))
    p_geom /= p_geom.sum(axis=1, keepdims=True)
    pred = np.argmax(p_geom * v30_mod.PESOS_PLUG_IN, axis=1)

    submission = pd.DataFrame({"id": test["id"].to_numpy(),
                               "age_range": [comun.CLASSES[i] for i in pred]})
    ruta = RAIZ / "submissions" / "age_submission_v48_geometric_poe_blend.csv"
    referencia = pd.read_csv(ruta)
    iguales = float((submission["age_range"].to_numpy() == referencia["age_range"].to_numpy()).mean())
    distintas = int((submission["age_range"].to_numpy() != referencia["age_range"].to_numpy()).sum())
    print(f"[cadena] submission V48: celdas distintas = {distintas} de {len(submission)} "
          f"| coincidencia = {iguales:.8f}", flush=True)

    temporal = Path("/tmp/age_submission_v48_reconstruida.csv")
    submission.to_csv(temporal, index=False)
    sha = hashlib.sha256(temporal.read_bytes()).hexdigest()
    print(f"[cadena] SHA256 reconstruido = {sha}", flush=True)
    print("[cadena] SHA256 documentado  = bf2554571ca6676195af66a053e28411fc1cfc9e3946fd95bac1cae0ce71df62",
          flush=True)
    print(f"[cadena] SHA identico: {sha == 'bf2554571ca6676195af66a053e28411fc1cfc9e3946fd95bac1cae0ce71df62'}",
          flush=True)


ETAPAS = {"features": etapa_features, "members": etapa_members, "meta": etapa_meta,
          "test": etapa_test, "v30": etapa_v30, "nn": etapa_nn, "sub": etapa_sub}


def main() -> int:
    """Ejecuta las etapas pedidas del spike de cadena."""
    ap = argparse.ArgumentParser(description="Spike de la cadena completa AGE")
    ap.add_argument("--etapas", default=",".join(ETAPAS), help="lista separada por comas")
    ap.add_argument("--cache", default=str(CACHE_DEFECTO), help="directorio de cache")
    args = ap.parse_args()
    cache = Path(args.cache)
    pedidas = [e.strip() for e in args.etapas.split(",") if e.strip()]
    for nombre in pedidas:
        if nombre not in ETAPAS:
            print(f"[cadena] etapa desconocida: {nombre}")
            return 2
        print(f"\n===== ETAPA {nombre} =====", flush=True)
        t0 = time.time()
        ETAPAS[nombre](cache)
        print(f"===== ETAPA {nombre} FIN: {time.time() - t0:.1f}s =====", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
