"""V29 - 05 - P7 y P8: construccion y congelacion del pool de candidatos.

Este script decide **que modelos entran al ensemble**. Es la decision mas importante de V29 y la
que mas facil de hacer mal, porque hay dos trampas:

## Trampa 1: el rescate bruto

Un modelo con CV 0,06 "rescata" decenas de miles de filas, porque V20 falla el 39 % del train y
cualquier moneda acierta parte de eso. Eso no es diversidad util, es ruido. La metrica de decision
es la **fraccion de los errores de V20 que el modelo corrige**, que esta normalizada, y no el
numero absoluto de filas.

## Trampa 2: el CV maximo

Elegir por mejor CV individual es justo lo que el brief prohibe. Un modelo con CV 0,610 y errores
casi identicos a los de V20 **no** es mejor miembro; uno con CV 0,556 y errores claramente
distintos puede aportar mas al conjunto.

## El criterio, declarado y auditable

Cuatro filtros **consecutivos**, cada uno con su motivo, y sin formula resumen que oculte la
tension:

1. **Util**: CV >= 0,50. Por debajo no hay senal que combinar.
2. **Diverso**: o bien su familia es distinta de la de los cuatro cuartos de V20, o bien su
   correlacion de errores con V20 esta por debajo de la media de los pares internos de V20.
3. **No redundante**: se eliminan los que toman las **mismas decisiones** que otro ya elegido
   (mismo argmax), porque en un ensemble con voto no aportan nada aunque difieran en magnitud.
4. **Tope de tamano**: como maximo `MAX_CANDIDATOS`, para que la seleccion de subconjunto tenga un
   espacio de busqueda manejable y para no construir el ensemble gigante que el brief prohibe.

Los filtros se aplican en ese orden y se registra **cuantos cae cada uno**, para que un pool
demasiado pequeño se vea como lo que es y no como un criterio afortunado.
"""

from __future__ import annotations

import hashlib
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

from src import config as cfg  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

RESULTADO = cfg.EXPERIMENTS / "v29_05_pool.json"
POOL = cfg.REPORTS / "v29_candidate_pool.csv"
OOF_POOL = cfg.EXPERIMENTS / "v29_pool_oof.npz"
N_CLASES = 6
CV_V20 = 0.6097782
CV_MINIMO = 0.50
MAX_CANDIDATOS = 16
#: Los cuatro cuartos de V20, que son la referencia de diversidad y el modelo a batir.
MIEMBROS_V20 = ["v19_b1_oof_logreg", "v19_b1_oof_mlp", "v19_oof_arbol_5folds",
                "v20_cadena_reconstruida", "v20_f3b_oof_candidato"]
FAMILIAS_V20 = {"logistica multiclase", "perceptron multicapa", "arbol / boosting",
                "cuarto de V20"}
#: Los dos modelos nuevos de Optuna, si P4 y P5 los-produjeron.
NUEVOS_OPTUNA = {"v29_lgbm_optuna": "v29_lgbm_optuna_oof.npy",
                 "v29_mlp_optuna": "v29_mlp_optuna_oof.npy"}


def _cargar(nombre: str) -> np.ndarray:
    return np.load(cfg.FEATURES / nombre, allow_pickle=False)


def main() -> int:
    """Aplica los cuatro filtros, escribe el pool y congela sus OOF."""
    autotest_numerico()
    inicio = time.time()
    train = __import__("src.data", fromlist=["load_train"]).load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(v) for v in y])
    contra = pd.read_csv(cfg.REPORTS / "v29_diversity_vs_v20.csv")
    inventario = pd.read_csv(cfg.REPORTS / "v29_model_inventory.csv")
    diversidad = json.loads((cfg.EXPERIMENTS / "v29_02_diversidad.json").read_text(encoding="utf-8"))
    liston = diversidad["referencia_diversidad_interna_v20"]["corr_errores_media"]

    oof_v20 = _cargar("v20_f3b_oof_candidato.npy")
    pred_v20 = np.argmax(oof_v20, axis=1)
    bien_v20 = pred_v20 == clase
    errores_v20 = int((~bien_v20).sum())
    print(f"[P7] V20 CV {CV_V20} | {errores_v20:,} errores | liston de diversidad {liston:.4f}",
          flush=True)

    # Los candidatos se empiezan con todos los reutilizables mas los nuevos de Optuna.
    filas = []
    for _, f in contra.iterrows():
        artefacto = f["modelo_a"] + ".npy"
        if not (cfg.FEATURES / artefacto).exists():
            continue
        info = inventario[inventario["model_id"] == f["modelo_a"]]
        filas.append({
            "model_id": f["modelo_a"], "artefacto": artefacto,
            "familia": info["familia"].iloc[0] if not info.empty else "desconocida",
            "fase": info["fase"].iloc[0] if not info.empty else "desconocida",
            "individual_cv": float(f["corrected_cv_propio"]),
            "corr_errores_vs_V20": float(f["corr_errores"]),
            "corr_prob_vs_V20": float(f["corr_probabilidades"]),
            "tasa_discrepancia_vs_V20": float(f["tasa_discrepancia"]),
            "rescate_unico": int(f["rescate_unico"]),
            "dano_unico": int(f["dano_unico"]),
            "neto": int(f["neto"]),
            "fraccion_errores_de_V20_que_corrige": float(
                f["fraccion_de_los_errores_de_V20_que_corrige"]),
            "fraccion_aciertos_de_V20_que_rompe": float(
                f["fraccion_de_los_aciertos_de_V20_que_rompe"]),
            "origen": "historico",
        })
    for nombre, artefacto in NUEVOS_OPTUNA.items():
        if not (cfg.FEATURES / artefacto).exists():
            print(f"  [aviso] {artefacto} no existe: P4 o P5 no completo, se omite", flush=True)
            continue
        matriz = _cargar(artefacto)
        pred = np.argmax(matriz, axis=1)
        bien = pred == clase
        rescate = int((bien & ~bien_v20).sum())
        dano = int((~bien & bien_v20).sum())
        filas.append({
            "model_id": nombre, "artefacto": artefacto, "familia": "Optuna (nueva capacidad)",
            "fase": "V29 P4/P5", "individual_cv": round(float(f1_score(
                clase, pred, average="macro", labels=list(range(N_CLASES)))), 7),
            "corr_errores_vs_V20": round(float(np.corrcoef((~bien).astype(float),
                                                          (~bien_v20).astype(float))[0, 1]), 6),
            "corr_prob_vs_V20": round(float(np.corrcoef(matriz.ravel(),
                                                        oof_v20.ravel())[0, 1]), 6),
            "tasa_discrepancia_vs_V20": round(float((pred != pred_v20).mean()), 6),
            "rescate_unico": rescate, "dano_unico": dano, "neto": rescate - dano,
            "fraccion_errores_de_V20_que_corrige": round(rescate / max(errores_v20, 1), 6),
            "fraccion_aciertos_de_V20_que_rompe": round(dano / max(int(bien_v20.sum()), 1), 6),
            "origen": "Optuna V29",
        })
    tabla = pd.DataFrame(filas)
    # ---------------------------------------------------------------- V20 entra en su propio pool
    # La tabla `contra` es "modelos comparados CONTRA V20", asi que V20 no esta en ella por
    # construccion. Sin anadirlo aqui, el ensemble no podria ser "V20 mas miembros diversos", que
    # es justamente la pregunta de V29, y el control de H04 no tendria sobre que pronunciarse.
    # Se anade V20 y su cuarto de cadena, que son la referencia del ensemble a batir.
    acumuladas = np.load(cfg.FEATURES / "v20_f3_oof_cumuladas.npy")
    n_umbral = acumuladas.shape[1]
    mono = acumuladas.copy()
    for k in range(1, n_umbral):
        mono[:, k] = np.maximum(mono[:, k], mono[:, k - 1])
    seis = np.zeros((len(mono), n_umbral + 1))
    seis[:, 0] = mono[:, 0]
    for k in range(1, n_umbral):
        seis[:, k] = mono[:, k] - mono[:, k - 1]
    seis[:, -1] = 1.0 - mono[:, n_umbral - 1]
    seis = np.clip(seis, 0.0, None)
    seis /= np.clip(seis.sum(axis=1, keepdims=True), 1e-9, None)
    np.save(cfg.FEATURES / "v20_cadena_reconstruida.npy", seis)
    for nombre, matriz in (("v20_f3b_oof_candidato", oof_v20),
                          ("v20_cadena_reconstruida", seis)):
        pred = np.argmax(matriz, axis=1)
        bien = pred == clase
        rescate = int((bien & ~bien_v20).sum())
        dano = int((~bien & bien_v20).sum())
        tabla = pd.concat([tabla, pd.DataFrame([{
            "model_id": nombre, "artefacto": nombre + ".npy", "familia": "cuarto de V20",
            "fase": "V20 (congelado)", "individual_cv": round(float(f1_score(
                clase, pred, average="macro", labels=list(range(N_CLASES)))), 7),
            "corr_errores_vs_V20": round(float(np.corrcoef(
                (~bien).astype(float), (~bien_v20).astype(float))[0, 1]), 6),
            "corr_prob_vs_V20": round(float(np.corrcoef(matriz.ravel(),
                                                        oof_v20.ravel())[0, 1]), 6),
            "tasa_discrepancia_vs_V20": round(float((pred != pred_v20).mean()), 6),
            "rescate_unico": rescate, "dano_unico": dano, "neto": rescate - dano,
            "fraccion_errores_de_V20_que_corrige": round(rescate / max(errores_v20, 1), 6),
            "fraccion_aciertos_de_V20_que_rompe": round(dano / max(int(bien_v20.sum()), 1), 6),
            "origen": "V20 (referencia)"}])], ignore_index=True)
    print(f"[P7] {len(tabla)} candidatos de partida "
          f"({int((tabla['origen'] == 'historico').sum())} historicos, "
          f"{int((tabla['origen'] == 'Optuna V29').sum())} de Optuna, "
          f"{int(tabla['origen'].str.contains('V20').sum())} de referencia V20)", flush=True)

    # ---------------------------------------------------------------- filtro 1: util
    antes = len(tabla)
    tabla = tabla[tabla["individual_cv"] >= CV_MINIMO].copy()
    cae_1 = antes - len(tabla)
    # ---------------------------------------------------------------- filtro 2: diverso
    antes = len(tabla)
    es_miembro_v20 = tabla["model_id"].isin(MIEMBROS_V20)
    diverso = (~tabla["familia"].isin(FAMILIAS_V20)) | (tabla["corr_errores_vs_V20"] < liston)
    tabla = tabla[diverso | es_miembro_v20].copy()
    # A los miembros de V20 se les marca para que el pool los contenga como referencia, pero no
    # cuentan como candidatos nuevos.
    cae_2 = antes - len(tabla)
    # ---------------------------------------------------------------- filtro 3: no redundante
    antes = len(tabla)
    argmax_por_modelo = {f["model_id"]: np.argmax(_cargar(f["artefacto"]), axis=1)
                        for _, f in tabla.iterrows()}
    hashes = {m: hashlib.sha256(a.tobytes()).hexdigest()[:12] for m, a in argmax_por_modelo.items()}
    vistos, quedan = set(), []
    # Primero los miembros de V20, que son la referencia, y despues por indice de rescate
    # normalizado, que es la senal util y no la bruta.
    orden = tabla.assign(_es_v20=es_miembro_v20).sort_values(
        ["_es_v20", "fraccion_errores_de_V20_que_corrige", "individual_cv"],
        ascending=[False, False, False])
    for _, f in orden.iterrows():
        h = hashes[f["model_id"]]
        if h in vistos:
            continue
        vistos.add(h)
        quedan.append(f["model_id"])
    redundantes = [m for m in tabla["model_id"] if m not in quedan]
    tabla = tabla[tabla["model_id"].isin(quedan)].copy()
    cae_3 = antes - len(tabla)
    # ---------------------------------------------------------------- filtro 4: tamano
    antes = len(tabla)
    orden4 = tabla.assign(_es_v20=es_miembro_v20).sort_values(
        ["_es_v20", "fraccion_errores_de_V20_que_corrige", "individual_cv"],
        ascending=[False, False, False])
    if len(orden4) > MAX_CANDIDATOS:
        Cuartos = orden4[orden4["_es_v20"]]
        candidatos = orden4[~orden4["_es_v20"]]
        # El tope de tamano se reparte en tres tramos, y el orden **no** puede dejar fuera al mejor
        # modelo individual por el simple hecho de que sea poco diverso. Ordenar solo por rescate
        # normalizado hizo exactamente eso: el LightGBM de Optuna, con el mejor CV del proyecto,
        # se quedo fuera porque sus errores se parecen a los del arbol de V20. Ser el mas diverso
        # y ser el mejor no son la misma propiedad, y un pool que excluye al mejor no puede
        # responder a la pregunta de V29.
        #
        # Tramo 1: el de mayor CV individual, siempre. Tramo 2: los de mayor fraccion de errores
        # de V20 corregidos. Tramo 3: el resto por CV. Y todo se declara aqui.
        elegido = list(candidatos.nlargest(1, "individual_cv")["model_id"])
        for m in candidatos.sort_values(
                ["fraccion_errores_de_V20_que_corrige", "individual_cv"],
                ascending=False)["model_id"]:
            if m not in elegido:
                elegido.append(m)
        huecos = MAX_CANDIDATOS - len(Cuartos)
        tabla = pd.concat([Cuartos, candidatos[candidatos["model_id"].isin(elegido[:huecos])]]
                          ).copy()
    cae_4 = antes - len(tabla)

    tabla["seleccionado_para_pool"] = True
    tabla["motivo_seleccion"] = [
        "cuarto de V20: referencia del ensemble a batir" if m in MIEMBROS_V20
        else ("Optuna: capacidad nueva" if f.origen != "historico"
              else f"diverso (corr_err {f.corr_errores_vs_V20:.3f} < {liston:.3f}) y "
                   f"corrige {f.fraccion_errores_de_V20_que_corrige:.3f} de los errores de V20")
        for m, f in zip(tabla["model_id"], tabla.itertuples())]
    tabla = tabla.sort_values("individual_cv", ascending=False).reset_index(drop=True)
    tabla.to_csv(POOL, index=False)

    # ---------------------------------------------------------------- P8: congelar el pool
    # Se guardan los OOF de todos los candidatos en un unico archivo. A partir de aqui la
    # combinacion no puede reentrenar a nadie: solo ponderar estas matrices congeladas.
    matrices, nombres = [], []
    for _, f in tabla.iterrows():
        matrices.append(_cargar(f["artefacto"]).astype(np.float32))
        nombres.append(f["model_id"])
    np.savez_compressed(OOF_POOL, **{m: v for m, v in zip(nombres, matrices)})
    print(f"\n[P8] pool congelado en {OOF_POOL.name}: {len(nombres)} miembros, "
          f"{OOF_POOL.stat().st_size / 1e6:.1f} MB", flush=True)

    print(f"\n  filtros aplicados: util -{cae_1} | diverso -{cae_2} | "
          f"no redundante -{cae_3} | tamano -{cae_4}", flush=True)
    if redundantes:
        print(f"  eliminados por argmax identico: {redundantes}", flush=True)
    print(f"\n  {'modelo':<32}{'CV':>8}{'corr_err':>10}{'corta%err':>11}{'rompe%ac':>10}  origen")
    for _, f in tabla.iterrows():
        print(f"  {f['model_id']:<32}{f['individual_cv']:>8.4f}{f['corr_errores_vs_V20']:>10.4f}"
              f"{f['fraccion_errores_de_V20_que_corrige']:>11.3f}"
              f"{f['fraccion_aciertos_de_V20_que_rompe']:>10.3f}  {f['origen']}", flush=True)

    RESULTADO.write_text(json.dumps({
        "fase": "V29 05 P7-P8 pool de candidatos", "fecha": "2026-09-29", "experimenta": False,
        "criterio": ["util: CV >= 0,50", "diverso: familia distinta de la de V20 o correlacion de "
                     "errores por debajo del liston interno de V20", "no redundante: se eliminan "
                     "los que toman las mismas decisiones que otro ya elegido",
                     f"tamano: como maximo {MAX_CANDIDATOS}"],
        "liston_diversidad_v20": round(liston, 6),
        "candidatos_de_partida": int(len(filas)), "cain_por_filtro": {"util": cae_1,
                                                                     "diverso": cae_2,
                                                                     "no_redundante": cae_3,
                                                                     "tamano": cae_4},
        "redundantes_por_argmax": redundantes,
        "pool": tabla.to_dict("records"), "n_miembros": len(nombres),
        "pool_congelado": OOF_POOL.name,
        "nuevos_de_optuna_en_pool": [m for m in nombres if m in NUEVOS_OPTUNA],
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[fin] {time.time() - inicio:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
