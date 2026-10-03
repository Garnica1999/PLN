"""V31+ - 00 - P0: auditoria de partida y reproduccion verificable de V30.

V30 es ahora el baseline absoluto y **queda congelado**. Esta fase no lo modifica: lo reproduce, lo
audita y lo usa como referencia contra la que medir cualquier representacion nueva.

## Que se comprueba

1. **V30 reproduce** desde sus artefactos: corrected_cv, folds, segunda particion y pesos.
2. **El fichero de submission de V30 esta intacto**: mismo hash, mismo numero de filas, mismas
   clases. Su SHA256 es `f543a715...` y debe seguir siendolo.
3. **El leaderboard de V30** se registra como evidencia de transferencia **externa**, y se contrasta
   con lo que predice la calibracion del propio proyecto. Es la unica vez que se usa en V31+, y no
   como funcion objetivo: sirve para saber si la calibracion interna es fiable, que es informacion
   sobre la que se decide el **diseno** de los cortes, nunca sobre que se prueba.
4. **La auditoria de legalidad de URL** se hace aqui, antes de construir nada, porque su resultado
   puede cerrar un track entero y eso hay que saberlo antes de trabajarlo.

## El estado congelado

```text
V20 corrected_cv   = 0,6097782   LB = 0,62689
V29 LightGBM       = 0,6121698
V30 ensemble       = 0,6130051   LB = 0,63302
V30 2a particion   = 0,6143347
```

V30 mejoró el leaderboard en **+0,00613** sobre V20. Ese numero es un hecho, no un objetivo.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
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
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

RESULTADO = cfg.EXPERIMENTS / "v31_00_auditoria.json"
CV_V20 = 0.6097782
CV_LGBM_V29 = 0.6121698
CV_V30 = 0.6130051
#: El baseline **operativo** de V31+: el CV de la configuracion que se envio y que
#: reproduce desde los artefactos. La cifra publicada en V30 venia de una configuracion
#: de pesos por fold que no quedo persistida. Se explica en `reproduce_v30`.
CV_V30_REPRO = 0.6134471
CV_V30_SEGUNDA = 0.6143347
LB_V20 = 0.62689
LB_V30 = 0.63302
HASH_V30 = "f543a715a1881c6128a860301b84a7273c3db8f8a709545be1e6267822e713be"
#: Calibracion interna LB = a + b * CV, ajustada por el propio proyecto en fases anteriores.
CALIBRACION = (0.00223, 1.02825)
N_CLASES = 6
FECHA = "2026-09-30"
SUBMISSION_V30 = ROOT / "submissions" / "age_submission_v30_final_candidate.csv"
COMPETICION = "tweet-age-2026-puj"


def _macro(clase: np.ndarray, prob: np.ndarray) -> float:
    return float(f1_score(clase, np.argmax(prob, axis=1), average="macro",
                          labels=list(range(N_CLASES))))


def reproduce_v30() -> dict:
    """Recomputa el OOF de V30 desde el pool congelado y audita que cifra es la buena.

    ## Un hallazgo que corrige el baseline

    V30 publico `0,6130051`, pero ese numero se calculo **componiendo el OOF con los pesos que
    Optuna optimizo en cada fold por separado**. Esos pesos **no se persistieron en el artefacto**:
    el JSON solo guarda la media entre folds. Y lo que la submission uso fueron **los pesos
    medios**, que dan un OOF distinto.

    Las dos cifras, y cual manda:

    | Configuracion | CV | LB predicho | LB real 0,63302 | error |
    |---|---|---|---|---|
    | pesos por fold (**no persistida, no reproducible**) | 0,6130051 | 0,63255 | 0,63302 | +0,00047 |
    | pesos medios (**lo que se envio, reproducible**) | 0,6134471 | 0,63301 | 0,63302 | **+0,00001** |

    El leaderboard dice cual de las dos es la real, y dice que es la de los **pesos medios**. Por
    tanto el baseline operativo de V31+ es **0,6134471**, no 0,6130051: es el unico que se puede
    reproducir desde los artefactos, es exactamente lo que se envio, y es el que casa con el
    leaderboard hasta la quinta decimal.

    La cifra publicada no era erronea en si misma, pero **no era reproducible y no describia el
    modelo enviado**. Se declara en lugar de elegir silenciosamente una de las dos.
    """
    ens = json.loads((cfg.EXPERIMENTS / "v30_02_ensemble.json").read_text(encoding="utf-8"))
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(v) for v in y])
    datos_pool = np.load(cfg.EXPERIMENTS / "v29_pool_oof.npz")
    extra = {"v29_lgbm_optuna": cfg.FEATURES / "v29_lgbm_optuna_oof.npy",
             "v30_lgbm_refinado": cfg.FEATURES / "v30_lgbm_refinado_oof.npy"}
    matrices, nombres = [], []
    for nombre in ens["manifest"]["miembros"]:
        if nombre in extra:
            matriz = np.load(extra[nombre])
        elif nombre == "v20_f3b_oof_candidato":
            matriz = np.load(cfg.FEATURES / "v20_f3b_oof_candidato.npy")
        elif nombre in datos_pool.files:
            matriz = datos_pool[nombre]
        else:
            raise FileNotFoundError(nombre)
        matrices.append(matriz.astype(np.float64))
        nombres.append(nombre)
    pesos = np.array([ens["manifest"]["pesos"].get(n, 0.0) for n in nombres], dtype=float)
    pesos = pesos / pesos.sum()
    oof = np.tensordot(pesos, np.stack(matrices), axes=(0, 0))
    valor = _macro(clase, oof)
    publicado = next(f["corrected_cv"] for f in ens["tabla"]
                     if f["metodo"] == ens["candidato_principal"])
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")
    por_fold = [round(_macro(clase[f["val_idx"]], oof[f["val_idx"]]), 7) for f in folds]
    return {
        "configuracion_reproducible": "pesos medios entre folds, que es lo que uso la submission",
        "miembros": nombres,
        "pesos_usados": {n: round(float(w), 4) for n, w in zip(nombres, pesos)},
        "corrected_cv_recomputado": round(valor, 7),
        "reproduce": abs(valor - CV_V30_REPRO) < 1e-6,
        "desviacion": round(abs(valor - CV_V30_REPRO), 9),
        "folds": por_fold,
        "cv_publicado_en_v30": publicado,
        "cv_publicado_es_reproducible": False,
        "motivo_de_la_diferencia": (
            "el CV publicado se compuso con los pesos **por fold** que Optuna optimizo, y esos "
            "pesos no se persistieron en el artefacto; el JSON solo guarda la media. La "
            "submission uso los pesos medios, cuyo OOF es el que se reproduce aqui"),
        "baseline_operativo": CV_V30_REPRO,
        "cifra_publicada": CV_V30,
        "delta_entre_las_dos_cifras": round(valor - publicado, 7),
        "cual_manda": "la de los pesos medios, porque es la unica reproducible, es exactamente la "
                      "que se envio, y la que casa con el leaderboard hasta la quinta decimal",
    }


def auditoria_submission() -> dict:
    """El fichero de V30 tiene que estar byte a byte igual que cuando se selló."""
    if not SUBMISSION_V30.exists():
        return {"existe": False}
    sha = hashlib.sha256(SUBMISSION_V30.read_bytes()).hexdigest()
    marco = pd.read_csv(SUBMISSION_V30)
    return {
        "existe": True, "sha256": sha, "hash_esperado": HASH_V30,
        "intacto": sha == HASH_V30,
        "filas": int(len(marco)), "columnas": list(marco.columns),
        "ids_unicos": bool(marco[cfg.ID_COL].is_unique),
        "sin_nan": bool(not marco.isna().any().any()),
        "clases": sorted(marco["age_range"].astype(str).unique().tolist()),
        "bytes": int(SUBMISSION_V30.stat().st_size),
    }


def transferencia() -> dict:
    """Contrasta el LB de V30 con lo que predice la calibracion interna.

    Es **informacion de diseno**, no una funcion objetivo: sirve para saber si la calibracion
    interna es fiable y, con ello, que tamanho de delta en CV hace falta para mover el
    leaderboard. No se usa para elegir nada.
    """
    a, b = CALIBRACION
    predicciones = {f"CV {round(cv, 7)}": round(a + b * cv, 5)
                    for cv in (CV_V20, CV_LGBM_V29, CV_V30, CV_V30_REPRO, CV_V30_SEGUNDA)}
    # La prediccion que se compara con el LB real es la del baseline **operativo**, que es el
    # unico que reproduce y el que se envio.
    predicho = a + b * CV_V30_REPRO
    predicho_publicado = a + b * CV_V30
    return {
        "calibracion": f"LB = {a} + {b} * CV",
        "lb_predicho_para_v30": round(predicho, 5),
        "lb_predicho_con_la_cifra_publicada": round(predicho_publicado, 5),
        "error_si_se_usara_la_cifra_publicada": round(LB_V30 - predicho_publicado, 5),
        "lb_real_de_v30": LB_V30,
        "error_de_transferencia": round(LB_V30 - predicho, 5),
        "nota_sobre_la_cifra": "la comparacion se hace contra el baseline **operativo** "
                               "(0,6134471), no contra el 0,6130051 publicado. Con el operativo el "
                               "error cae a +0,00001; con el publicado seria +0,00047. Esa "
                               "diferencia es la evidencia de que lo que se envio fue el "
                               "operativo",
        "mejora_de_lb_sobre_v20": round(LB_V30 - LB_V20, 5),
        "caso_del_punto_32": "A: CV robusto, segunda particion robusta y LB mejora. "
                             "**successful transfer**",
        "implicacion_para_el_diseno": (
            "la calibracion interna acierta con un error de "
            f"{abs(LB_V30 - predicho):.5f} con el baseline operativo, asi que un delta de +0,003 "
            f"en CV equivale a unos "
            f"{0.003 * b:.5f} en el leaderboard. Los cortes de esta fase se pueden fijar en CV sin "
            "convertir el leaderboard en un objetivo"),
        "lb_por_candidado": predicciones,
        "uso_del_leaderboard": "una sola lectura, como evidencia de transferencia. No es funcion "
                               "objetivo, no se usa para seleccionar nada y no se volvera a "
                               "consultar durante la fase",
    }


def auditoria_url_legality() -> dict:
    """Clasifica formalmente la hipotesis de recuperar el contenido de una URL.

    El brief exige clasificarlo ALLOWED / AMBIGUOUS / PROHIBITED **antes** de implementarlo, y
    ponerlo en QUARANTINED si hay ambiguedad. Aqui se comprueba si las reglas estan disponibles.
    """
    hallazgos = {"reglas_locales_de_AGE": [], "api_kaggle_accesible": None,
                 "archivo_de_reglas_en_otro_proyecto": None}
    # 1. ¿Hay una copia de las reglas de ESTA competencia en algun sitio?
    for ruta in list(ROOT.rglob("*rule*")) + list(ROOT.rglob("*competi*")):
        if ruta.is_file() and "__pycache__" not in str(ruta):
            hallazgos["reglas_locales_de_AGE"].append(str(ruta.relative_to(ROOT)))
    # 2. ¿La API responde? Es una llamada de solo lectura, sin enviar nada.
    try:
        proceso = subprocess.run(
            ["/home/carlos/venv/bin/kaggle", "competitions", "files", "-c", COMPETICION],
            capture_output=True, text=True, timeout=90)
        accesible = proceso.returncode == 0
        sin_auth = "Authentication required" in (proceso.stdout + proceso.stderr)
        hallazgos["api_kaggle_accesible"] = accesible
        if not accesible:
            hallazgos["sin_autenticacion"] = sin_auth
            hallazgos["motivo_api"] = (
                "la API de Kaggle rechaza la llamada por autenticacion, asi que las reglas "
                "oficiales no se pueden leer desde aqui"
                if "Authentication required" in (proceso.stdout + proceso.stderr) else
                "la API de Kaggle fallo por otra causa")
    except Exception as error:  # noqa: BLE001
        hallazgos["api_kaggle_accesible"] = False
        hallazgos["motivo_api"] = type(error).__name__
    # 3. El unico fichero de reglas local pertenece a OTRA competicion y esta mal titulado.
    otro = ROOT.parent / "kaggle_toxic_2026_puj" / "docs" / "competition_rules.md"
    if otro.exists():
        texto = otro.read_text(encoding="utf-8", errors="ignore")
        es_otra = ("Hate Speech" in texto or "toxicidad" in texto.lower()
                   or "tweet-toxic" in texto)
        hallazgos["archivo_de_reglas_en_otro_proyecto"] = {
            "ruta": str(otro.relative_to(ROOT.parent)), "pertenece_a_otra_competicion": es_otra,
            "nota": "el fichero se titula 'Reglas oficiales de la competencia' pero su contenido "
                    "describe la competencia de toxicidad (HS, TR, AG), no la de edad. **No se usa "
                    "como fuente de reglas de AGE**, porque daria por cubierto un requisito que no "
                    "esta cubierto" if es_otra else "revisar",
        }
    reglas_disponibles = bool(hallazgos["reglas_locales_de_AGE"]) and \
        hallazgos["api_kaggle_accesible"]
    estado = "QUARANTINED"
    return {
        "hipotesis": "A5: si una fila contiene la URL del propio tweet, recuperar el texto original "
                     "y reemplazar la representacion por el contenido real",
        "estado": estado,
        "clasificacion": "AMBIGUOUS" if not reglas_disponibles else "ALLOWED",
        "motivo": ("las reglas oficiales de la compilacion no estan disponibles: no hay copia local "
                   "de AGE, la API de Kaggle no autentica desde este entorno, y el unico fichero de "
                   "reglas presente pertenece a otra compilacion y esta mal titulado. Sin el texto "
                   "de las reglas no se puede establecer que la recuperacion de contenido externo "
                   "este permitida, y recuperarla podria ser informacion externa, "
                   "transductiva o fuga de metadatos"),
        "decision": "QUARANTINED. No se implementa la recuperacion, no se hace ninguna llamada de "
                    "red y ningun candidato que dependa de contenido recuperado puede llegar a una "
                    "submission. Los tracks A1 a A4, que usan **solo el texto del train**, no se "
                    "ven afectados y se ejecutan con normalidad",
        "como se levanta el cuarentena": "que el organizador confirme por escrito que la "
                                        "recuperacion de contenido desde URLs propias esta "
                                        "permitida y que no se use junto a metadatos",
        "hallazgos": hallazgos,
    }


def main() -> int:
    """Ejecuta la auditoria de partida y detiene V31+ si V30 no esta integro."""
    autotest_numerico()
    inicio = time.time()
    print("=== P0.1 reproduccion de V30 ===", flush=True)
    v30 = reproduce_v30()
    print(f"  corrected_cv recomputado {v30['corrected_cv_recomputado']} | registrado "
          f"{v30['cv_publicado_en_v30']} | operativo {CV_V30_REPRO} | reproduce: {v30['reproduce']}",
          flush=True)
    print(f"  pesos: {v30['pesos_usados']}", flush=True)

    print("\n=== P0.2 integridad del fichero de submission de V30 ===", flush=True)
    sub = auditoria_submission()
    print(f"  existe: {sub['existe']} | SHA256 {sub.get('sha256', '')[:16]}... | intacto: "
          f"{sub.get('intacto')} | filas {sub.get('filas'):,}", flush=True)

    print("\n=== P0.3 evidencia de transferencia de V30 ===", flush=True)
    tr = transferencia()
    print(f"  LB predicho por la calibracion: {tr['lb_predicho_para_v30']} | LB real: "
          f"{tr['lb_real_de_v30']} | error: {tr['error_de_transferencia']}", flush=True)
    print(f"  mejora sobre V20 en el leaderboard: {tr['mejora_de_lb_sobre_v20']:+.5f}", flush=True)
    print(f"  {tr['caso_del_punto_32']}", flush=True)

    print("\n=== P0.4 auditoria de legalidad del retrieval desde URLs ===", flush=True)
    url = auditoria_url_legality()
    print(f"  estado: **{url['estado']}** (clasificacion {url['clasificacion']})", flush=True)
    print(f"  reglas de AGE disponibles: "
          f"{'si' if url['hallazgos']['api_kaggle_accesible'] and url['hallazgos']['reglas_locales_de_AGE'] else 'NO'}",
          flush=True)
    print(f"  {url['decision']}", flush=True)

    problemas = []
    if not v30["reproduce"]:
        problemas.append("V30 no reproduce desde sus artefactos")
    if abs(v30["corrected_cv_recomputado"] - CV_V30_REPRO) > 1e-6:
        problemas.append("el baseline operativo de V30 no reproduce")
    if not sub.get("intacto"):
        problemas.append("el fichero de submission de V30 cambio de hash")
    if not sub.get("sin_nan") or not sub.get("ids_unicos"):
        problemas.append("el fichero de submission de V30 esta corrupto")
    if abs(tr["error_de_transferencia"]) > 0.002:
        problemas.append("la calibracion interna no explica el LB de V30: habria que auditar")

    RESULTADO.write_text(json.dumps({
        "fase": "V31+ 00 P0 auditoria y baseline", "fecha": FECHA, "experimenta": False,
        "baseline_v30": v30, "integridad_submission_v30": sub,
        "evidencia_de_transferencia": tr, "auditoria_url_legality": url,
        "estado_congelado": {"V20": CV_V20, "V29_lgbm": CV_LGBM_V29, "V30": CV_V30, "V30_operativo": CV_V30_REPRO,
                             "V30_segunda": CV_V30_SEGUNDA, "LB_V20": LB_V20, "LB_V30": LB_V30},
        "artefactos_de_v30_tocados": [],
        "problemas": problemas, "v31_autorizado": not problemas,
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    if problemas:
        print("\n[STOP] ", problemas, flush=True)
        return 1
    print("\n[P0] APROBADO. V31+ puede empezar.", flush=True)
    print(f"[P0] {time.time() - inicio:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
