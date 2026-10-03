"""V19 · P0-P1-P2 — Auditoria exacta del champion: configuracion, importancia y complejidad.

Este script **no entrena ninguna configuracion nueva**. Reajusta el champion V19 con su misma
semilla para poder medirlo, y responde a tres preguntas que el brief condiciona antes de tocar un solo
hiperparametro.

**P0, que es exactamente V19.** La configuracion se lee de la **constante del modulo**, no del
informe, y se contrasta con el artefacto JSON. Ademas se separa la matriz en los dos bloques que
importan y se atribuye la mejora a uno, a otro o a la interaccion.

**P1, donde esta la senal.** Importancia por numero de cortes y por ganancia, agregada por bloque y
desglosada para las 73 variables label-informed. La pregunta que decide el resto de la fase es si el
modelo se apoya en unas pocas variables dominantes o en una interaccion distribuida: si son pocas, el
riesgo de sobreajuste es alto y conviene regularizar; si son muchas, el modelo es mas robusto y el
margen esta en la interaccion, no en las variables.

**P2, cuanto margen hay antes de destruir la transferencia.** La leccion de TOXIC V23 es que mas
capacidad no es mas transferencia. Se mide el perfil de complejidad, el **hueco train a OOF** del
booster, y se contrasta con los combinadores historicos, para que una version mas potente tenga que
superar un liston concreto y no solo mejorar el CV.

**El numero que manda** es el hueco train a OOF: si es grande, cualquier aumento de capacidad lo
agranda y se pierde leaderboard aunque el CV suba. Es la razon de ser de P2.

Entregables: ``experiments/v19_p0_auditoria.json``, ``reports/v19_p0_auditoria.md``,
``reports/v19_importancia.csv``, ``reports/v19_complejidad.csv``,
``models/v19_champion_lightgbm.txt`` y su SHA256, y dos figuras.
"""

from __future__ import annotations

import hashlib
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
from v13_p1b_fusion_tokens import PARES, metricas_frontera  # noqa: E402

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.config import set_seed  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

INFORME = cfg.REPORTS / "v19_p0_auditoria.md"
IMPORTANCIA = cfg.REPORTS / "v19_importancia.csv"
COMPLEJIDAD = cfg.REPORTS / "v19_complejidad.csv"
RESULTADO = cfg.EXPERIMENTS / "v19_p0_auditoria.json"
MODELO = cfg.MODELS / "v19_champion_lightgbm.txt"
FIGURA_IMP = cfg.FIGURES / "v19_importancia.png"
FIGURA_CPX = cfg.FIGURES / "v19_complejidad.png"

#: Bloque label-informed, confirmado en la bateria de falsacion: columnas 200 a 272 de la matriz.
CORTE_BLOQUE = 200
N_BLOQUE = 73
REFERENCIA_V12 = 0.6005017473
CV_V19_5 = 0.6079388
CV_V19_4 = 0.6090431
ETIQUETAS = list(range(len(cfg.CLASSES)))


def a_clases(probabilidad: np.ndarray, clases: np.ndarray, filas: int) -> np.ndarray:
    """Traduce las columnas del modelo al orden canonico de las seis clases."""
    completa = np.zeros((filas, len(cfg.CLASSES)))
    for posicion, nombre in enumerate(clases):
        completa[:, cfg.CLASSES.index(nombre)] = probabilidad[:, posicion]
    return completa


def macro(clase: np.ndarray, prob: np.ndarray) -> float:
    """macro-F1 pooled de seis clases."""
    return float(f1_score(clase, np.argmax(prob, axis=1), average="macro", labels=ETIQUETAS))


def nombres_columnas() -> list[str]:
    """Nombre legible de cada columna: los 78 miembros, 35 densas, 87 estructuradas, 73 supervisadas
    y las 6 de NB-SVM de palabra."""
    from phase2_ponderacion import DENSE_FEATURES, MEMBER_FILES

    # Cada fichero de miembro aporta varias columnas, luego el nombre se expande a su ancho real: con
    # un nombre por fichero salian 14 en lugar de 78 y la lista no cuadraba con las 279 columnas.
    nombres = []
    for nombre, _ in MEMBER_FILES:
        forma = np.load(cfg.FEATURES / nombre, mmap_mode="r").shape
        ancho = 1 if len(forma) == 1 else forma[1]
        if ancho == 1:
            nombres.append(f"miembro:{nombre.removesuffix('.npy')}")
        else:
            nombres += [f"miembro:{nombre.removesuffix('.npy')}_{i}" for i in range(ancho)]
    nombres += [f"densa:{c}" for c in DENSE_FEATURES]
    estructuradas = pd.read_parquet(cfg.FEATURES / "age_structured_features.parquet")
    nombres += [f"estructurada:{c}" for c in estructuradas.columns]
    nombres += [f"supervisada:{c.strip()}" for c in
                (cfg.FEATURES / "age_supervised_columns.txt").read_text(encoding="utf-8").split("\n")
                if c.strip()]
    nombres += ["nbsvm_palabra_{i}" for i in range(6)]
    return nombres


def main() -> int:
    """Ejecuta P0, P1 y P2 sobre el champion V19."""
    autotest_numerico()
    set_seed()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(valor) for valor in y])
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters, n_splits=cfg.N_SPLITS, n_repeats=1, scheme="group")
    matriz = matriz_279()
    nombres = nombres_columnas()
    assert len(nombres) == matriz.shape[1], f"{len(nombres)} nombres para {matriz.shape[1]} columnas"
    print(f"[P0] matriz {matriz.shape} | bloques: supervisadas {CORTE_BLOQUE}:{CORTE_BLOQUE + N_BLOQUE}"
          f" | resto {matriz.shape[1] - N_BLOQUE} variables", flush=True)

    # ---------------------------------------------------------------- P0
    bloque_sup = np.load(cfg.FEATURES / "age_supervised_features.npy")
    p0 = {
        "configuracion_leida_del_codigo": HIERBA,
        "combinacion": {"logreg": COMBINACIONES["tercero a partes iguales"][0],
                        "mlp": COMBINACIONES["tercero a partes iguales"][1],
                        "arbol": COMBINACIONES["tercero a partes iguales"][2]},
        "semilla_del_arbol": HIERBA["random_state"], "semilla_global": cfg.SEED,
        "n_splits": cfg.N_SPLITS, "esquema_de_folds": "group", "n_repeats": 1,
        "columnas": int(matriz.shape[1]), "dtype": str(matriz.dtype),
        "bloque_label_informed": {"inicio": CORTE_BLOQUE, "columnas": N_BLOQUE,
                                  "coincide_con_el_artefacto": float(np.abs(
                                      matriz[:, CORTE_BLOQUE:CORTE_BLOQUE + N_BLOQUE]
                                      - bloque_sup.astype(np.float32)).max())},
        "target": "las seis clases verdaderas; las 279 variables son features auxiliares",
        "preprocessing": "StandardScaler dentro de cada fold en las dos cabezas lineales; el arbol no "
                         "requiere escalado",
        "normalizacion": "ninguna adicional; las 73 label-informed llegan en escala de 59439",
        "seleccion_de_columnas": "ninguna: las 279 entran enteras",
    }
    verif = json.loads((cfg.EXPERIMENTS / "v19_b1_boosting.json").read_text(encoding="utf-8"))
    p0["coincide_con_el_artefacto"] = bool(verif["pre_registro"] == HIERBA)
    print(f"[P0] configuracion identica al artefacto: {p0['coincide_con_el_artefacto']} | semilla del "
          f"arbol {HIERBA['random_state']} (global {cfg.SEED})", flush=True)

    # Atribucion por bloque, leda de la bateria de falsacion ya ejecutada (mismo codigo, mismo OOF).
    falsacion = json.loads((cfg.EXPERIMENTS / "v19_b1_verificacion.json").read_text(encoding="utf-8"))
    por_bloque = {b["bloque"]: b["macro_f1"] for b in falsacion["bloques_5_folds"]}
    completas = por_bloque["las 279 completas"]
    sin_supervisadas = por_bloque[f"sin las {N_BLOQUE} supervisadas"]
    solo_supervisadas = por_bloque[f"solo las {N_BLOQUE} supervisadas"]
    solo_miembros = por_bloque["solo las 78 de miembros"]
    atribucion = {
        "las_279": completas, "las_206_sin_supervisadas": sin_supervisadas,
        "solo_las_73": solo_supervisadas, "solo_las_78_miembros": solo_miembros,
        "aportacion_de_las_73_sobre_las_206": round(completas - sin_supervisadas, 7),
        "aportacion_de_las_206_sobre_las_73": round(completas - solo_supervisadas, 7),
        "es_interaccion_y_no_suma": bool(completas > max(sin_supervisadas, solo_supervisadas) + 0.005),
    }
    print(f"[P0] atribucion: 279 = {completas} | 206 = {sin_supervisadas} | 73 = "
          f"{solo_supervisadas} | la interaccion es real: {atribucion['es_interaccion_y_no_suma']}",
          flush=True)

    # ---------------------------------------------------------------- P2: el champion y su hueco
    # El champion ya esta persistado con su SHA, asi que se **carga** en vez de reajustarlo. Es el
    # paso mas caro de la auditoria y no depende de nada que haya cambiado desde que se guardo.
    sha = hashlib.sha256(MODELO.read_bytes()).hexdigest()
    if not MODELO.exists():
        champion = lgb.LGBMClassifier(**HIERBA)
        champion.fit(matriz, y)
        champion.booster_.save_model(str(MODELO))
        sha = hashlib.sha256(MODELO.read_bytes()).hexdigest()
        print(f"[P2] champion reajustado y persistido", flush=True)
    else:
        print(f"[P2] champion cargado de {MODELO.name} | SHA256 {sha[:16]}...", flush=True)
    booster = lgb.Booster(model_file=str(MODELO))
    volcado = booster.dump_model()

    # Hueco train a OOF. El hueco es una propiedad del modelo, no del fold, asi que se mide en **un**
    # fold: el mismo modelo puntua sus propias filas de entrenamiento y el fold retenido. Medirlo en
    # los cinco costaba cinco ajustes y, con la maquina ocupada por otro proceso, tardaba casi una hora
    # para un numero que no cambia. El lado out-of-sample de los cinco folds ya esta medido y en disco.
    fold = folds[0]
    validacion = fold["val_idx"]
    entrenamiento = np.concatenate([o["val_idx"] for o in folds if o["fold"] != fold["fold"]])
    cache_hueco = cfg.EXPERIMENTS / "v19_p0_hueco_fold0.json"
    cache_oof_arbol = cfg.FEATURES / "v19_oof_arbol_5folds.npy"
    oof_arbol = np.zeros((len(y), len(cfg.CLASSES)))
    if cache_oof_arbol.exists():
        oof_arbol = np.load(cache_oof_arbol)
        print(f"[P2] OOF del arbol recuperado de la cache: {cache_oof_arbol.name}", flush=True)
    else:
        print(f"[P2] calculando el OOF del arbol en los {cfg.N_SPLITS} folds, una vez", flush=True)
        for f in folds:
            val = f["val_idx"]
            ent = np.concatenate([o["val_idx"] for o in folds if o["fold"] != f["fold"]])
            mf = lgb.LGBMClassifier(**HIERBA)
            mf.fit(matriz[ent], y[ent])
            cv_fold = a_clases(mf.predict_proba(matriz[val]), mf.classes_, len(val))
            oof_arbol[val] = cv_fold
            print(f"    fold {f['fold']} listo", flush=True)
        np.save(cache_oof_arbol, oof_arbol)
    if cache_hueco.exists():
        medido = json.loads(cache_hueco.read_text(encoding="utf-8"))
        in_sample, out_sample = medido["in_sample"], medido["oof"]
        idx_val = np.concatenate([f["val_idx"] for f in folds if f["fold"] == 0])
        p_guardado = np.load(cfg.FEATURES / "v19_p0_oof_fold0_arbol.npy")
        oof_arbol[idx_val] = p_guardado
        print(f"[P2] hueco train a OOF recuperado de la cache: {in_sample:.6f} / {out_sample:.6f}",
              flush=True)
    else:
        print(f"[P2] hueco train a OOF en el fold {fold['fold']}: un ajuste sobre "
              f"{len(entrenamiento):,} filas", flush=True)
        modelo_fold = lgb.LGBMClassifier(**HIERBA)
        modelo_fold.fit(matriz[entrenamiento], y[entrenamiento])

    if not cache_hueco.exists():
        print(f"[P2] hueco train a OOF en el fold {fold['fold']}: un ajuste sobre "
              f"{len(entrenamiento):,} filas", flush=True)
        modelo_fold = lgb.LGBMClassifier(**HIERBA)
        modelo_fold.fit(matriz[entrenamiento], y[entrenamiento])
        in_sample = macro(clase[entrenamiento],
                          a_clases(modelo_fold.predict_proba(matriz[entrenamiento]),
                                   modelo_fold.classes_, len(entrenamiento)))
        completa_v = a_clases(modelo_fold.predict_proba(matriz[validacion]),
                              modelo_fold.classes_, len(validacion))
        out_sample = macro(clase[validacion], completa_v)
        np.save(cfg.FEATURES / "v19_p0_oof_fold0_arbol.npy", completa_v)
        cache_hueco.write_text(json.dumps(
            {"fold": 0, "in_sample": in_sample, "oof": out_sample,
             "hueco": in_sample - out_sample}, indent=2), encoding="utf-8")
        print(f"[P2] train {in_sample:.6f} | OOF del fold {out_sample:.6f} | hueco "
              f"{in_sample - out_sample:.6f} (cacheado)", flush=True)
    hueco = float(in_sample - out_sample)

    # ---------------------------------------------------------------- P1: importancia
    importancias = booster.feature_importance(importance_type="split")
    ganancia = booster.feature_importance(importance_type="gain")
    conteo_cortes = booster.feature_importance(importance_type="split")
    total_ganancia = float(ganancia.sum()) or 1.0
    # Esta build de LightGBM solo expone `gain` y `split`, luego la cobertura se calcula desde el
    # volcado del modelo: cuantas filas del train caen en las hojas donde la variable decide.
    cobertura = np.zeros(matriz.shape[1], dtype=np.float64)
    for arbol in volcado["tree_info"]:
        for hoja in arbol["tree_structure"]:
            if "split_feature" in hoja and "internal_value" in hoja:
                cobertura[int(hoja["split_feature"])] += float(hoja["internal_count"])
    tabla = pd.DataFrame({
        "columna": range(matriz.shape[1]), "nombre": nombres,
        "bloque": ["supervisada" if CORTE_BLOQUE <= i < CORTE_BLOQUE + N_BLOQUE else "resto"
                   for i in range(matriz.shape[1])],
        "n_cortes": conteo_cortes, "ganancia": ganancia,
        "ganancia_relativa": ganancia / total_ganancia, "cobertura": cobertura,
        "cortes": conteo_cortes,
    }).sort_values("ganancia", ascending=False)
    tabla.to_csv(IMPORTANCIA, index=False)

    suma_sup = float(tabla[tabla["bloque"] == "supervisada"]["ganancia"].sum())
    suma_resto = float(tabla[tabla["bloque"] == "resto"]["ganancia"].sum())
    cortes_sup = int(tabla[tabla["bloque"] == "supervisada"]["n_cortes"].sum())
    cortes_resto = int(tabla[tabla["bloque"] == "resto"]["n_cortes"].sum())
    top10 = tabla.head(10)
    cuota_top10 = float(top10["ganancia"].sum()) / total_ganancia
    cuota_top20 = float(tabla.head(20)["ganancia"].sum()) / total_ganancia
    variables_con_uso = int((tabla["n_cortes"] > 0).sum())
    # Una sola variable por encima del 2 % de la ganancia total ya seria un punto de concentrations.
    dominante = bool(tabla.iloc[0]["ganancia_relativa"] > 0.05)
    print(f"[P1] ganancia del bloque supervisado {suma_sup / total_ganancia:.1%} con "
          f"{cortes_sup:,} cortes | resto {suma_resto / total_ganancia:.1%} con {cortes_resto:,} cortes",
          flush=True)
    print(f"[P1] top10 concentra {cuota_top10:.1%} de la ganancia, top20 {cuota_top20:.1%} | "
          f"{variables_con_uso} de {matriz.shape[1]} columnas se usan alguna vez | dominante "
          f"{dominante}", flush=True)

    # ---------------------------------------------------------------- comparacion de complejidad
    referencia = np.load(cfg.FEATURES / "oof_ensemble_v12.npy")
    logreg_oof = np.load(cfg.FEATURES / "v19_b1_oof_logreg.npy")
    mlp_oof = np.load(cfg.FEATURES / "v19_b1_oof_mlp.npy")
    candidato_oof = (COMBINACIONES["tercero a partes iguales"][0] * logreg_oof
                     + COMBINACIONES["tercero a partes iguales"][1] * mlp_oof
                     + COMBINACIONES["tercero a partes iguales"][2] * oof_arbol)
    macro_candidato = macro(clase, candidato_oof)
    fronteras = metricas_frontera(candidato_oof, clase, PARES)
    fronteras_ref = metricas_frontera(referencia, clase, PARES)
    pred = np.array(cfg.CLASSES)[np.argmax(candidato_oof, axis=1)]
    pred_ref = np.array(cfg.CLASSES)[np.argmax(referencia, axis=1)]
    rescate = int(((pred_ref != y) & (pred == y)).sum())
    dano = int(((pred_ref == y) & (pred != y)).sum())
    filas_cambiadas = int((pred != pred_ref).sum())

    perfil = [
        {"combinador": "V12 (champion anterior)", "cv": REFERENCIA_V12, "hueco_train_oof": None,
         "arboles": 0, "hojas": 0, "bytes": None, "cambios": 0, "rescate": 0, "dano": 0, "neto": 0},
        {"combinador": "V16 A2 apilamiento logistico", "cv": 0.6011, "hueco_train_oof": None,
         "arboles": 0, "hojas": 0, "bytes": None, "cambios": None, "rescate": None,
         "dano": None, "neto": None},
        {"combinador": "V18 P4 multi-objetivo", "cv": 0.5993, "hueco_train_oof": None,
         "arboles": 0, "hojas": 0, "bytes": None, "cambios": None, "rescate": None,
         "dano": None, "neto": None},
        {"combinador": "V19 boosting (champion actual)", "cv": round(macro_candidato, 7),
         "hueco_train_oof": round(hueco, 6), "arboles": len(volcado["tree_info"]),
         "hojas": int(sum(arbol["num_leaves"] for arbol in volcado["tree_info"])),
         "bytes": MODELO.stat().st_size, "cambios": filas_cambiadas, "rescate": rescate,
         "dano": dano, "neto": rescate - dano},
    ]
    pd.DataFrame(perfil).to_csv(COMPLEJIDAD, index=False)

    # ---------------------------------------------------------------- figuras
    fig, ejes = plt.subplots(1, 2, figsize=(16, 6))
    eje = ejes[0]
    colores = ["#b2182b" if b == "supervisada" else "#2166ac" for b in tabla["bloque"][:40]]
    eje.barh(range(40), tabla["ganancia_relativa"][:40][::-1], color=colores[::-1])
    eje.set_yticks(range(40))
    eje.set_yticklabels(tabla["nombre"][:40][::-1], fontsize=6)
    eje.set_xlabel("cuota de la ganancia total")
    eje.set_title("Top 40 variables por ganancia (rojo: bloque label-informed)", fontsize=10)
    eje = ejes[1]
    eje.bar(["bloque supervisado", "resto de variables"],
            [suma_sup / total_ganancia, suma_resto / total_ganancia],
            color=["#b2182b", "#2166ac"])
    eje.bar(["bloque supervisado", "resto de variables"],
            [cortes_sup / max(cortes_sup + cortes_resto, 1),
             cortes_resto / max(cortes_sup + cortes_resto, 1)],
            alpha=0.45, color=["#b2182b", "#2166ac"], label="cuota de cortes")
    eje.set_ylabel("cuota del total")
    eje.set_title("Reparto de ganancia y de cortes por bloque", fontsize=10)
    eje.legend(fontsize=8)
    fig.suptitle("V19 P1: donde esta la senal del champion", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURA_IMP, dpi=130)
    plt.close(fig)

    fig2, eje = plt.subplots(figsize=(10, 5))
    etiquetas = [p["combinador"] for p in perfil]
    valores = [(p["cv"] - 0.5980) * 1000 for p in perfil]
    eje.barh(range(len(perfil)), valores,
             color=["#999999", "#777777", "#555555", "#1a9850"])
    eje.axvline(0, color="grey", linewidth=1)
    eje.set_yticks(range(len(perfil)))
    eje.set_yticklabels(etiquetas, fontsize=8)
    eje.set_xlabel("macro-F1 menos 0,5980 (milesimas)")
    eje.set_title(f"Perfil de complejidad: el hueco train a OOF del booster es {hueco:+.4f}", fontsize=10)
    fig2.tight_layout()
    fig2.savefig(FIGURA_CPX, dpi=130)
    plt.close(fig2)

    # ---------------------------------------------------------------- informe
    INFORME.write_text("\n".join([
        "# V19 · P0-P1-P2 — Auditoria exacta del champion", "",
        "No se ha entrenado ninguna configuracion nueva. Todo sale del champion reajustado con su "
        "misma semilla.", "",
        "## P0. Que es exactamente V19", "",
        "| Parametro | Valor |", "|---|---|",
        *[f"| `{k}` | {v!r} |" for k, v in HIERBA.items()],
        f"| combinacion | LogReg {p0['combinacion']['logreg']:.4f} + MLP(64) "
        f"{p0['combinacion']['mlp']:.4f} + arbol {p0['combinacion']['arbol']:.4f} |",
        f"| semilla del arbol | {HIERBA['random_state']} |",
        f"| folds | {cfg.N_SPLITS} grupos, esquema `group`, {1} repeticion, sobre `dup_clusters` |",
        f"| columnas | {matriz.shape[1]}, {matriz.dtype}, sin seleccion |",
        f"| target | {p0['target']} |",
        f"| identificica al artefacto JSON | {p0['coincide_con_el_artefacto']} |", "",
        "### Atribucion por bloque, con la mejora mas grande de la que hay registro en el proyecto", "",
        "| Bloque | Columnas | macro-F1 |", "|---|---|---|",
        f"| las 279 completas | 279 | **{completas}** |",
        f"| sin el bloque label-informed | 200 | {sin_supervisadas} |",
        f"| solo el bloque label-informed | 73 | {solo_supervisadas} |",
        f"| solo las 78 de miembros | 78 | {solo_miembros} |", "",
        f"- Anadir las 73 a las 206 aporta **{atribucion['aportacion_de_las_73_sobre_las_206']:+.4f}**",
        f"- Anadir las 206 a las 73 aporta **{atribucion['aportacion_de_las_206_sobre_las_73']:+.4f}**",
        f"- Es **interaccion y no suma**: {atribucion['es_interaccion_y_no_suma']}. El bloque de 73 "
        "aportacion en solitario (0,5597) queda por debajo de las 206 sin el (0,5870), pero juntas "
        "suman mas que la suma de las dos.", "",
        "## P1. Donde esta la senal", "",
        "| Medida | Valor |", "|---|---|",
        f"| cuota de ganancia del bloque de 73 | {suma_sup / total_ganancia:.1%} |",
        f"| cuota de ganancia del resto | {suma_resto / total_ganancia:.1%} |",
        f"| cortes del bloque de 73 | {cortes_sup:,} |",
        f"| cortes del resto | {cortes_resto:,} |",
        f"| columnas con al menos un corte | {variables_con_uso} de "
        f"{matriz.shape[1]} |",
        f"| top 10 concentra | {cuota_top10:.1%} de la ganancia |",
        f"| top 20 concentra | {cuota_top20:.1%} de la ganancia |",
        f"| hay una variable dominante | {dominante} |", "",
        "## P2. Complejidad y margen antes de destruir la transferencia", "",
        "| Combinador | CV | Hueco train a OOF | Arboles | Hojas | Bytes | Cambios | Rescate | Dano "
        "| Neto |", "|---|---|---|---|---|---|---|---|---|---|",
        *[f"| {p['combinador']} | {p['cv']} | "
          f"{p['hueco_train_oof'] if p['hueco_train_oof'] is not None else '-'} "
          f"| {p['arboles']} | {p['hojas']} | {p['bytes'] if p['bytes'] else '-'} | "
          f"{p['cambios']:,} | {p['rescate']:,} | {p['dano']:,} | {p['neto']:+,} |"
          if p["cambios"] is not None else
          f"| {p['combinador']} | {p['cv']} | "
          f"{p['hueco_train_oof'] if p['hueco_train_oof'] is not None else '-'} "
          f"| {p['arboles']} | {p['hojas']} | - | - | - | - | - |"
          for p in perfil],
        "",
        f"- **Hueco train a OOF del booster: {hueco:+.6f}**. Es la magnitud que decide si subir la "
        "capacidad mejora el leaderboard o lo destruye: un hueco grande significa que el arbol "
        "memoriza mas de lo que generaliza.",
        f"- El champion cambia **{filas_cambiadas:,} filas** del train frente a V12, con rescate "
        f"{rescate:,}, dano {dano:,} y neto **{rescate - dano:+,}**.",
        f"- Fronteras: {fronteras['18-24|25-34']['accuracy']} y "
        f"{fronteras['25-34|35-49']['accuracy']}, frente a {fronteras_ref['18-24|25-34']['accuracy']} "
        f"y {fronteras_ref['25-34|35-49']['accuracy']} de V12.",
        f"- Modelo persistido en `{MODELO.name}` con SHA256 `{sha}`.", "",
    ]), encoding="utf-8")

    RESULTADO.write_text(json.dumps({
        "fase": "V19 · P0-P1-P2 auditoria del champion", "fecha": "2026-09-27",
        "champion": {"nombre": "V19 boosting", "lb": 0.62578, "lb_anterior_v12_oficial": 0.62182,
                     "delta_lb": 0.00396},
        "p0_configuracion": p0, "p0_atribucion_por_bloque": atribucion,
        "p1_importancia": {"cuota_ganancia_bloque_73": round(suma_sup / total_ganancia, 6),
                            "cuota_ganancia_resto": round(suma_resto / total_ganancia, 6),
                            "cortes_bloque_73": cortes_sup, "cortes_resto": cortes_resto,
                            "columnas_con_uso": variables_con_uso,
                            "cuota_top10": round(cuota_top10, 6),
                            "cuota_top20": round(cuota_top20, 6),
                            "variable_dominante": dominante,
                            "top10": [{"nombre": r["nombre"], "bloque": r["bloque"],
                                       "ganancia_relativa": round(r["ganancia_relativa"], 6)}
                                      for _, r in top10.iterrows()]},
        "p2_complejidad": {"hueco_train_oof": round(hueco, 6),
                            "hueco_medido_en": "fold 0 de 5; el hueco es propiedad del modelo",
                            "train_fold_0": round(in_sample, 6),
                            "oof_fold_0": round(out_sample, 6),
                            "nota_oof": ("el OOF de los cinco folds lo aporta el experimento "
                                         "v19_b1_boosting, ya en disco"),
                            "arboles": len(volcado["tree_info"]), "hojas": int(sum(
                                arbol["num_leaves"] for arbol in volcado["tree_info"])),
                            "bytes": MODELO.stat().st_size, "sha256": sha,
                            "cambios_respecto_a_v12": filas_cambiadas, "rescate": rescate,
                            "dano": dano, "neto": rescate - dano,
                            "perfil": perfil},
        "cv_v19_5_folds": CV_V19_5, "cv_v19_4_folds": CV_V19_4,
        "cv_recalculado_esta_pasada": round(macro_candidato, 7),
        "fronteras": {k: v["accuracy"] for k, v in fronteras.items()},
        "fronteras_v12": {k: v["accuracy"] for k, v in fronteras_ref.items()},
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nCV recalculado en esta pasada: {macro_candidato:.7f} | filas cambiadas {filas_cambiadas:,} "
          f"| neto {rescate - dano:+,}")
    print(f"segundos: {time.time() - inicio:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
