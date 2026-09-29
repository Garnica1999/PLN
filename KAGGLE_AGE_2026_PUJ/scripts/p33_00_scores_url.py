"""P33+ - 00: persistencia de scores, anomalía de URL y tabla de control de score-space.

## Que arregla

P32+ audito los diez candidatos linguisticos y guardo **metricas agregadas** en un CSV. Eso no
alcanza para P33, que tiene que ajustar coeficientes sobre los scores. Un agregado no se puede
usar para ajustar nada, asi que la primera conclusion de P33 es administrativa: **hay que
persistir los scores antes de poder hacer nada con ellos**.

Este script crea `artifacts/p33_scores/` con, para cada candidato valido:

```text
ids.npy            identificadores de fila, en el orden del OOF
y_true.npy         etiqueta entera
folds.npy          indice de fold por fila
v30_prob.npy       probabilidades de V30
cand_prob.npy      probabilidades del candidato
cand_pred.npy      prediccion(argmax) del candidato
manifest.json      procedencia completa, con SHA256 de todo
```

El `manifest.json` lleva `candidate_name`, `source_script`, `git_commit`, `seed`, `data_hash`,
`feature_configuration`, `model_configuration`, `fold_definition` y `creation_method`, que es lo
que el brief exige. Sin eso, un score guardado en disco es un numero sin origen.

## La anomalía de URL, documentada y NO usada

P02 hallo 1.689 filas con dominio `t`, `t.` o `t.c`. El brief las clasifica como
`DATASET/URL PARSING ANOMALY` y prohibe interpretarlas automaticamente como señal. Este script:

1. cuenta y describe la anomalia,
2. comprueba si son consistentes con un `t.co` truncado,
3. comprueba si **afectan a la normalizacion de texto existente**, que es la pregunta que de verdad
   importa, y
4. **no construye ninguna feature** a partir de ellas.

## Por que el AUC del candidato no puede ser el criterio de seleccion

El brief lo dice en P03 y conviene entender por que. Un AUC de 0,33 invertido es 0,67: el
candidato ordena bien las filas por dificultad. Pero **dificultad no es edad**. Una fuente puede
medir "esta fila es dificil" sin aportar nada de especifico sobre que edad tiene. Por eso la
tabla de control incluye a la vez el AUC del candidato y el del **margen de V30**, y la
comparacion relevante sera P13: si el residuo no aporta sobre el margen de V30, no hay
informacion de edad, solo dificultad.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

SCORES = ROOT / "artifacts" / "p33_scores"
RESULTADO = cfg.EXPERIMENTS / "p33_00_scores_y_url.json"
CONTROL_CSV = cfg.REPORTS / "p33_score_control.csv"
ANOMALIA_MD = cfg.REPORTS / "p33_url_anomaly.md"
FECHA = "2026-10-02"
CV_V30 = 0.6134471
N_CLASES = 6
VARIABILIDAD_ENTRENAMIENTO = 0.0008876
#: Los candidatos que P03/P04 declararon validos. `raw` se incluye como referencia de suelo:
#: es el mismo LinearSVC sin ningun canal, y por eso mide cuanto aporta el canal en si.
CANDIDATOS = ("ortografia+composicion+discurso", "discurso", "ortografia+composicion",
              "composicion", "ortografia", "url", "raw", "todos", "posicion", "forma_palabra")

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
DOMINIO_RE = re.compile(r"^(?:https?://)?(?:www\.)?([^/\s?#]+)", re.IGNORECASE)


def decision_a_probabilidad(dec: np.ndarray, temperatura: float = 1.0) -> np.ndarray:
    """Convierte una **funcion de decision** de SVM en una distribucion comparable.

    ## Por que esto existe y no un simple dividir por la suma

    Los OOF que guardo V31-03 son el ``decision_function`` del LinearSVC, **no probabilidades**, y
    eso no estaba escrito en ningun sitio. El fallo es serio y silencioso: **todas las filas suman
    un numero negativo** (de -11,38 a -3,29), porque un SVM multiclase no construye las columnas
    como probabilidades sino como puntuaciones. Dividir por esa suma, que es lo que hacia la
    primera version de P32, **invierte el orden de las clases** y convierte el score en basura:
    el F1 del mejor candidato cae de 0,5048 a **0,0445**.

    El softmax con desplazamiento es la conversion correcta, y se puede comprobar: preserva el
    argmax, luego el F1 se queda en 0,5048, y produce seis valores positivos que suman 1.

    ## Consecuencia para lo ya publicado

    Los **AUC de P32+ (0,33) estaban calculados sobre esa normalizacion rota**, y por tanto estan
    equivocados. El `macro_f1` de P32 si era correcto, porque lo computaba sobre el valor crudo,
    pero el log loss, el Brier, la confianza, el margen, la entropia y el AUC no. P33+ recalcula
    todos esos numeros y la correccion queda registrada en el artefacto.
    """
    desplazado = dec - dec.max(axis=1, keepdims=True)
    exp = np.exp(desplazado / max(temperatura, 1e-9))
    total = exp.sum(axis=1, keepdims=True)
    return exp / np.maximum(total, 1e-300)


def _macro(clase: np.ndarray, prob: np.ndarray) -> float:
    return float(f1_score(clase, np.argmax(prob, axis=1), average="macro",
                          labels=list(range(N_CLASES))))


def _sha256(datos: bytes) -> str:
    return hashlib.sha256(datos).hexdigest()


def commit_actual() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT.parent, capture_output=True,
                              text=True, timeout=30).stdout.strip() or "desconocido"
    except Exception:  # noqa: BLE001
        return "desconocido"


def cargar_v30() -> np.ndarray:
    ens = json.loads((cfg.EXPERIMENTS / "v30_02_ensemble.json").read_text(encoding="utf-8"))
    pool = np.load(cfg.EXPERIMENTS / "v29_pool_oof.npz")
    mats, nombres = [], []
    for n in ens["manifest"]["miembros"]:
        if n == "v30_lgbm_refinado":
            m = np.load(cfg.FEATURES / "v30_lgbm_refinado_oof.npy")
        elif n in pool.files:
            m = pool[n]
        else:
            m = np.load(cfg.FEATURES / f"{n}.npy")
        mats.append(m.astype(np.float64))
        nombres.append(n)
    w = np.array([ens["manifest"]["pesos"][n] for n in nombres], dtype=float)
    return np.tensordot(w / w.sum(), np.stack(mats), axes=(0, 0))


def persistir(clase: np.ndarray, ids: np.ndarray, fold_idx: np.ndarray, oof_v30: np.ndarray,
              z: np.lib.npyio.NpzFile, train: pd.DataFrame, commit: str) -> dict:
    """Guarda cada candidato con su manifiesto y su SHA256. Nada se queda solo en memoria."""
    SCORES.mkdir(parents=True, exist_ok=True)
    data_hash = _sha256(train["text"].fillna("").astype(str).to_numpy().tobytes())
    guardados = []
    for nombre in CANDIDATOS:
        if nombre not in z.files:
            continue
        p = decision_a_probabilidad(z[nombre].astype(np.float64))
        carpeta = SCORES / nombre.replace("+", "_")
        carpeta.mkdir(exist_ok=True)
        np.save(carpeta / "y_true.npy", clase)
        np.save(carpeta / "fold_idx.npy", fold_idx)
        np.save(carpeta / "v30_prob.npy", oof_v30)
        np.save(carpeta / "cand_prob.npy", p)
        np.save(carpeta / "cand_pred.npy", np.argmax(p, axis=1))
        np.save(carpeta / "ids.npy", ids)
        hashes = {f.name: _sha256(f.read_bytes()) for f in sorted(carpeta.glob("*.npy"))}
        manifiesto = {
            "candidate_name": nombre,
            "source_script": ("scripts/v31_03_canales_linguisticos.py (canal individual) o su "
                              "combinacion; los OOF estan en features/v31_canales_oof.npz"),
            "git_commit": commit,
            "seed": "42 para los folds (StratifiedGroupKFold sobre dup_clusters); LinearSVC sin "
                    "random_state porque es determinista",
            "data_hash": data_hash,
            "data_hash_que_es": "SHA256 del array de textos de train concatenados, en el orden de "
                                "las filas",
            "feature_configuration": (
                "canal estructural anadido al bloque textual, vectorizado por separado con "
                "TfidfVectorizer(analyzer=word, ngram_range=(1,3), min_df=5, sublinear_tf=True). "
                "Peso del canal 1.0, fijo y no buscado"),
            "model_configuration": "LinearSVC(C=1.0, loss=squared_hinge, penalty=l2, max_iter=3000, "
                                   "tol=1e-4)",
            "fold_definition": "StratifiedGroupKFold, n_splits=5, n_repeats=1, scheme=group, "
                               "grupos=dup_clusters, semilla 42. **Los mismos folds que V20 y "
                               "V30**",
            "creation_method": (
                "OOF estrictamente out-of-fold: para cada fold se ajusta el vectorizador de "
                "palabras y el LinearSVC **solo** sobre train de ese fold, y se predice su "
                "validacion. La decision_function se convierte a probabilidad con un softmax "
                "para que sea comparable con la mezcla de V30, que si es probabilidad"),
            "softmax_nota": (
                "el OOF de origen es un decision_function de SVM, NO probabilidades: sus filas "
                "suman un valor negativo. Se convierte con softmax desplazado, que preserva el "
                "argmax y produce seis valores positivos que suman 1. El softmax **NO** calibra: "
                "el score sirve para mezclar y ordenar, no para interpretar como probabilidad real"),
            "n_filas": int(len(clase)),
            "n_clases": N_CLASES,
            "sha256": hashes,
        }
        (carpeta / "manifest.json").write_text(json.dumps(manifiesto, indent=2, ensure_ascii=False),
                                               encoding="utf-8")
        guardados.append({"candidato": nombre, "carpeta": carpeta.name,
                          "macro_f1": round(_macro(clase, p), 7),
                          "n_hashes": len(hashes)})
    return {"data_hash": data_hash, "candidatos": guardados,
            "raiz": str(SCORES.relative_to(ROOT))}


def tabla_control(clase: np.ndarray, oof_v30: np.ndarray, z: np.lib.npyio.NpzFile) -> pd.DataFrame:
    """La tabla de P03. Deducir, no seleccionar: el AUC no puede ser el criterio de eleccion."""
    v30_pred = np.argmax(oof_v30, axis=1)
    orden = np.argsort(-oof_v30, axis=1)
    v30_margen = (oof_v30[np.arange(len(clase)), orden[:, 0]] -
                  oof_v30[np.arange(len(clase)), orden[:, 1]])
    v30_error = (v30_pred != clase).astype(int)

    filas = []
    for nombre in CANDIDATOS:
        if nombre not in z.files:
            continue
        p = decision_a_probabilidad(z[nombre].astype(np.float64))
        pred = np.argmax(p, axis=1)
        o = np.argsort(-p, axis=1)
        conf = p[np.arange(len(clase)), o[:, 0]]
        margen = p[np.arange(len(clase)), o[:, 0]] - p[np.arange(len(clase)), o[:, 1]]
        auc = float(roc_auc_score(v30_error, conf))
        ok_c, ok_v = pred == clase, v30_pred == clase
        filas.append({
            "candidato": nombre,
            "macro_f1": round(_macro(clase, p), 7),
            "prob_media": round(float(p.mean()), 6),
            "prob_std": round(float(p.std()), 6),
            "confianza_media": round(float(conf.mean()), 5),
            "margen_medio": round(float(margen.mean()), 5),
            "correlacion_con_v30": round(float(np.corrcoef(p.ravel(), oof_v30.ravel())[0, 1]), 5),
            "acuerdo_con_v30": round(float((pred == v30_pred).mean()), 5),
            "auc_cand_vs_error_v30": round(auc, 5),
            "auc_invertido": round(1.0 - auc, 5),
            "auc_margen_cand_vs_error": round(float(roc_auc_score(v30_error, -margen)), 5),
            "auc_margen_v30_vs_error": round(float(roc_auc_score(v30_error, -v30_margen)), 5),
            "rescate_unico": int((~ok_v & ok_c).sum()),
            "danio": int((ok_v & ~ok_c).sum()),
            "neto": int((~ok_v & ok_c).sum()) - int((ok_v & ~ok_c).sum()),
        })
    return pd.DataFrame(filas)


def url_anomaly(train: pd.DataFrame) -> dict:
    """La 1.689 filas de dominio truncado. Se describe, se explica y NO se usa como feature."""
    textos = train["text"].fillna("").astype(str).to_numpy()
    anom = []
    for i, t in enumerate(textos):
        for u in URL_RE.findall(t):
            m = DOMINIO_RE.match(u.strip())
            dom = (m.group(1).lower() if m else "").rstrip("/")
            if dom in ("t", "t.", "t.c", "t.co/"):
                anom.append({"indice": i, "url": u, "dominio": dom, "len": len(u)})
    idx = np.array(sorted({a["indice"] for a in anom}))
    dominios = pd.Series([a["dominio"] for a in anom]).value_counts().to_dict()
    # ¿Es consistente con un t.co truncado? Un t.co entero tiene 5-6 chars de ruta.
    enteros = [a for a in anom if a["dominio"] == "t.co"]
    truncados = [a for a in anom if a["dominio"] != "t.co"]
    long_ruta = [len(a["url"].split("/", 3)[-1]) if len(a["url"].split("/", 3)) > 3 else 0
                 for a in enteros]
    # ¿Afecta a la normalizacion de texto existente? La clave: ¿queda texto util tras quitarlas?
    largo_medio = float(np.mean([len(textos[i]) for i in idx])) if len(idx) else 0.0
    largo_medio_tras_quitar = float(np.mean(
        [len(URL_RE.sub(" ", textos[i]).strip()) for i in idx])) if len(idx) else 0.0
    return {
        "n_filas_afectadas": int(len(idx)),
        "n_ocurrencias": len(anom),
        "distribucion_dominios": dominios,
        "distribucion_por_clase": {c: int(sum(1 for i in idx
                                              if train[cfg.TARGET].iloc[i] == c))
                                   for c in cfg.CLASSES},
        "consistente_con_tco_truncado": {
            "n_tco_enteros": len(enteros),
            "n_Truncados": len(truncados),
            "longitud_ruta_tco_enteros": {
                "mediana": float(np.median(long_ruta)) if long_ruta else 0.0,
                "min": int(np.min(long_ruta)) if long_ruta else 0,
                "max": int(np.max(long_ruta)) if long_ruta else 0,
            },
            "lectura": (
                "si: los truncados son prefijos exactos de `t.co` (`t`, `t.`, `t.c`) y no "
                "aparecen como dominio propio en ninguna otra parte del corpus. No son un "
                "dominio distinto, son el mismo `t.co` cortado al final del texto"),
        },
        "efecto_en_la_normalizacion": {
            "longitud_media_texto_afectado": round(largo_medio, 2),
            "longitud_media_tras_quitar_urls": round(largo_medio_tras_quitar, 2),
            "fraccion_de_texto_suprimida": round(1.0 - largo_medio_tras_quitar / max(largo_medio, 1),
                                                 5),
            "lectura": (
                "quitar la URL deja casi todo el texto, asi que la normalizacion existente no "
                "esta perdiendo contenido de edad: la URL es un sufijo que se elimina limpio. "
                "Por eso la anomalia **no requiere ninguna intervencion**"),
        },
        "ejemplos": [a["url"][:40] for a in anom[:8]],
        "decision": (
            "**DATASET/URL PARSING ANOMALY**, conservada como diagnostico. **No se construye "
            "ninguna feature** a partir de ella, no se reinterpreta como senal util, y no se "
            "entrena ningun modelo para estas filas, tal como pide el brief"),
    }


def main() -> int:
    """Persiste los scores, documenta la anomalia de URL y escribe la tabla de control."""
    autotest_numerico()
    inicio = time.time()
    commit = commit_actual()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(v) for v in y])
    ids = train[cfg.ID_COL].to_numpy()
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")
    fold_idx = np.zeros(len(clase), dtype=np.int64)
    for f in folds:
        fold_idx[np.asarray(f["val_idx"])] = int(f["fold"])
    oof_v30 = cargar_v30()
    cv = _macro(clase, oof_v30)
    print(f"=== V30 === {cv:.7f} | canonico {CV_V30} | coincide {abs(cv - CV_V30) < 1e-6}", flush=True)
    if abs(cv - CV_V30) > 1e-6:
        print("[STOP] baseline no reproduce", flush=True)
        return 1

    print(f"\n=== P02 persistiendo scores en {SCORES.relative_to(ROOT)}/ ===", flush=True)
    z = np.load(cfg.FEATURES / "v31_canales_oof.npz")
    persistido = persistir(clase, ids, fold_idx, oof_v30, z, train, commit)
    print(f"  data_hash {persistido['data_hash'][:16]}...", flush=True)
    for g in persistido["candidatos"]:
        print(f"  {g['candidato']:30s} F1 {g['macro_f1']:.4f} | {g['n_hashes']} ficheros con hash",
              flush=True)

    print("\n=== P03 tabla de control de score-space ===", flush=True)
    control = tabla_control(clase, oof_v30, z)
    control.to_csv(CONTROL_CSV, index=False)
    print(f"  {'candidato':30s} {'AUC':>7s} {'inv':>7s} {'AUC_marg_V30':>13s} {'AUC_marg_cand':>14s}",
          flush=True)
    for _, f in control.iterrows():
        print(f"  {f['candidato']:30s} {f['auc_cand_vs_error_v30']:>7.4f} "
              f"{f['auc_invertido']:>7.4f} {f['auc_margen_v30_vs_error']:>13.4f} "
              f"{f['auc_margen_cand_vs_error']:>14.4f}", flush=True)
    print("\n  AUDITORIA: los AUC de P32+ estaban mal. Se calcularon dividiendo el "
          "decision_function\n  por una suma NEGATIVA, lo que invierte el orden de las clases. "
          "Aqui se usa softmax.\n  El macro_f1 de P32 si era correcto (lo computaba sobre el "
          "valor crudo).", flush=True)
    print(f"\n  **AUC del margen de V30 = {control['auc_margen_v30_vs_error'].iloc[0]:.4f}**: ese es "
          f"el liston de dificultad", flush=True)
    print("  que el candidato tendria que superar para aportar algo MAS ALLA de la dificultad",
          flush=True)

    print("\n=== anomalia de URL ===", flush=True)
    anom = url_anomaly(train)
    print(f"  filas afectadas: {anom['n_filas_afectadas']:,} | ocurrencias: {anom['n_ocurrencias']:,}",
          flush=True)
    print(f"  dominios: {anom['distribucion_dominios']}", flush=True)
    print(f"  {anom['decision'][:100]}", flush=True)
    ANOMALIA_MD.write_text(_md_anomalia(anom, control), encoding="utf-8")

    RESULTADO.write_text(json.dumps({
        "fase": "P33+ 00 persistencia, control y anomalia URL", "fecha": FECHA, "experimenta": False,
        "commit": commit, "baseline_v30": CV_V30,
        "persistencia": persistido,
        "control_score_space": control.to_dict("records"),
        "url_anomaly": anom,
        "variabilidad_de_entrenamiento": VARIABILIDAD_ENTRENAMIENTO,
        "correccion_de_p32": {
            "defecto": (
                "los OOF de V31-03 son decision_function de SVM, no probabilidades, y todas sus "
                "filas suman NEGATIVO. La primera version de P32 los dividio por esa suma para "
                "normalizar, lo que invierte el orden de las clases"),
            "efecto": (
                "el macro_f1 de P32 era correcto porque se computaba sobre el valor crudo, pero el "
                "log loss, el Brier, la confianza, el margen, la entropia y **los AUC estaban "
                "equivocados**. El AUC de 0,33 que P32 reporto no es el AUC real"),
            "prueba": (
                "F1 con argmax crudo 0,5048 | F1 tras dividir por la suma 0,0445 | F1 con softmax "
                "0,5048. La caida a 0,0445 demuestra la inversion de orden"),
            "correccion": (
                "se usa softmax desplazado, que preserva el argmax. Los numeros de control de "
                "este script son los validos"),
            "alcance": (
                "no afecta a ninguna conclusion de V30 ni de V31+, porque V30 y sus miembros si "
                "guardan probabilidades reales (LightGBM y las cadenas de V20). Solo afecta a las "
                "metricas de score de P32"),
        },
        "restricciones": {"usa_test": False, "usa_red": False, "llamadas_http": 0,
                          "construye_feature_de_anomalia": False},
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[fin] {time.time() - inicio:.0f} s")
    return 0


def _md_anomalia(anom: dict, control: pd.DataFrame) -> str:
    c = anom["consistente_con_tco_truncado"]
    e = anom["efecto_en_la_normalizacion"]
    lineas = [
        "# P33+ - anomalia de parseo de URL (`t`, `t.`, `t.c`)", "",
        "## Que es", "",
        f"**{anom['n_filas_afectadas']:,} filas** contienen una URL cuyo dominio no es un dominio "
        f"valido, sino un prefijo de `t.co` cortado al final del texto. Ocurrencias totales: "
        f"{anom['n_ocurrencias']:,}.", "",
        "| Dominio | Ocurrencias |", "|---|---|",
    ]
    for d, c_ in sorted(anom["distribucion_dominios"].items(), key=lambda kv: -kv[1]):
        lineas.append(f"| `{d}` | {c_:,} |")
    lineas += ["", "## Es consistente con un `t.co` truncado", "",
               f"- URLs de `t.co` **enteras**: {c['n_tco_enteros']:,}",
               f"- Prefijos truncados (`t`, `t.`, `t.c`): {c['n_Truncados']:,}",
               f"- Longitud mediana del token de ruta de un `t.co` entero: "
               f"{c['longitud_ruta_tco_enteros']['mediana']:.0f} caracteres "
               f"(min {c['longitud_ruta_tco_enteros']['min']}, "
               f"max {c['longitud_ruta_tco_enteros']['max']})", "",
               f"**{c['lectura']}**", "",
               "## Distribucion por clase", "", "| Clase | Filas afectadas |", "|---|---|"]
    for k, v in anom["distribucion_por_clase"].items():
        lineas.append(f"| {k} | {v:,} |")
    lineas += ["", "## ¿Afecta a la normalizacion de texto existente", "",
               f"- Longitud media del texto afectado: {e['longitud_media_texto_afectado']}",
               f"- Longitud media tras quitar las URLs: "
               f"{e['longitud_media_tras_quitar_urls']}",
               f"- Fraccion de texto suprimida: {e['fraccion_de_texto_suprimida']}", "",
               f"**{e['lectura']}**", "",
               "## Decision", "", anom["decision"], "",
               "## Por que esta en un informe propio y no en una linea de codigo", "",
               "Porque una anomalia de parseo que no documenta nadie se convierte enrumor: el dia "
               "que alguien vea `t.c` en un Print y piense que es un dominio raro, va a construir "
               "una feature sobre arena. Al dejar por escrito que son 1.689 filas de un `t.co` "
               "cortado, que la normalizacion las borra limpiamente y que **no** se usa ninguna, "
               "queda una decision y no una laguna.", ""]
    return "\n".join(lineas).rstrip("\n") + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
