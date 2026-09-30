"""V32 - E2 (P2) - Pre-registro CONGELADO: challenger CatBoost sobre la matriz 279.

## Que congela

El segundo experimento de la reapertura V32: un challenger **CatBoost** multiclass entrenado
sobre la matriz de 279 features congelada, comparado contra el campeon V30 + plug-in
(baseline interno 0.6159732) con el mecanismo de integracion establecido (combinador logistico
anidado [p_V30 || p_challenger] y decision con la regla y pesos del plug-in CONGELADOS).

## Decisiones incorporadas (revisiones de agentes + usuario)

- ml-architect: A1 CatBoost MultiClass (una config, sin grid); A2 control de atribucion =
  gemelo LightGBM con la MISMA matriz/folds/split interno/ES sobre macro-F1 y los parametros
  congelados del ancla afinada V29/V30 del pool (aisla lo CatBoost-especifico frente a la
  familia boosting ya usada; control conservador); A3 = CatBoost con etiquetas barajadas
  (control negativo no decisional). Guarda de truncamiento y calibracion de fold 0.
- kaggle-specialist: APTO con condiciones: hash de la matriz verificado + re-entreno desde
  cero por fold en ambos brazos; ES 90/10 interno suficiente; gate de control con bootstrap
  pareado de la diferencia (delta_A1 - delta_A2) sin cero; tolerancias de determinismo
  declaradas; AST-audit de compliance; JSON con best_iteration/truncamiento (A1 cap MARGINAL).
- Usuario: autorizo el ciclo 2 (P2 CatBoost) tras cerrar P1 como NO_ATRIBUIBLE (D-081).

## Invariantes

- Sin test, sin holdout, sin leaderboard, sin submission, sin tuning posterior.
- El campeon V30 + plug-in no se modifica; el challenger es un candidato separado.
- Un experimento = un script nuevo; nada se ajusta mirando el resultado.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg
from src.verificacion_numerica import autotest_numerico

FECHA = "2026-09-30"
CANONICO = ROOT / "artifacts" / "p33_scores" / "todos"
MANIFEST = ROOT / "reports" / "age_submission_plugin_manifest.json"
PREREG_MD = cfg.REPORTS / "v32_p2_preregistro.md"
PREREG_JSON = cfg.EXPERIMENTS / "v32_02_preregistro_catboost.json"

BASELINE = 0.6159732
NESTED = 0.6154643
PESOS = [1.0, 1.1749, 1.2721, 1.163, 1.001, 0.8128]
N_CLASES = 6
SEMILLA = 20261010
SEMILLA_BARAJADO = 20261011
FRACCION_INTERNA = 0.10
N_BOOTSTRAP = 2000
GATE_DELTA = 0.0015
GATE_FOLDS = 4
UMBRAL_ALTA_CONFIANZA = 0.60
VARIABILIDAD = 0.0008876
MATRIZ_SHA_PREVIO = "175b6d1ee242eef95eec0f7c7c2e1bf5aa50578e20f006000f015e1ce634bf58"
HILOS = 32
ITERACIONES = 3000
OD_WAIT = 150
PROYECCION_FACTOR = 9.0
LIMITE_PROYECTADO_H = 20.0
LIMITE_GLOBAL_H = 24.0
FUENTES_A_HASHEAR = ("scripts/v19_b1_boosting.py", "scripts/v32_01_surrogate_macrof1.py",
                     "scripts/v30_02_ensemble.py", "scripts/age_umbrales_por_clase.py")

CATBOOST_CONFIG = {
    "loss_function": "MultiClass", "iterations": ITERACIONES, "learning_rate": 0.03,
    "depth": 6, "l2_leaf_reg": 3.0, "bootstrap_type": "Bernoulli", "subsample": 0.8,
    "random_strength": 1.0, "border_count": 254, "boosting_type": "Plain",
    "grow_policy": "SymmetricTree", "thread_count": HILOS, "random_seed": SEMILLA,
    "task_type": "CPU", "allow_writing_files": False, "od_type": "Iter", "od_wait": OD_WAIT,
    "use_best_model": True, "eval_metric": "TotalF1", "verbose": 100, "metric_period": 50,
}
LGBM_CONFIG = {
    "objective": "multiclass", "num_class": N_CLASES, "num_leaves": 56, "max_depth": 7,
    "learning_rate": 0.03, "min_data_in_leaf": 54, "bagging_fraction": 0.78,
    "bagging_freq": 1, "feature_fraction": 0.6, "lambda_l1": 0.015, "lambda_l2": 0.0005,
    "num_threads": HILOS, "seed": SEMILLA, "deterministic": True, "verbosity": -1,
    "num_boost_round": ITERACIONES, "early_stopping": OD_WAIT, "feval": "macro-F1",
}


def sha256_bytes(datos: bytes) -> str:
    return hashlib.sha256(datos).hexdigest()


def sha256_archivo(ruta: Path) -> str:
    return sha256_bytes(ruta.read_bytes())


def main() -> int:
    autotest_numerico()
    inicio = time.time()

    clase = np.load(CANONICO / "y_true.npy")
    fold_idx = np.load(CANONICO / "fold_idx.npy")
    v30 = np.load(CANONICO / "v30_prob.npy").astype(np.float64)
    v30 = np.clip(v30, 1e-12, 1.0)
    v30 = v30 / v30.sum(axis=1, keepdims=True)
    w = np.array(PESOS, dtype=float)
    baseline = float(f1_score(clase, np.argmax(v30 * w, axis=1), average="macro",
                              labels=list(range(N_CLASES))))
    print(f"baseline primario recomputado: {baseline:.7f} (esperado {BASELINE})", flush=True)
    if abs(baseline - BASELINE) > 5e-7:
        print("[STOP] el baseline canonico no reproduce", flush=True)
        return 1
    assert clase.shape == fold_idx.shape == (169336,), "artefactos canonicos desalineados"
    assert np.unique(fold_idx).size == 5, "se esperan 5 folds canonicos"

    from v19_b1_boosting import matriz_279
    matriz = matriz_279()
    print(f"matriz_279: {matriz.shape} {matriz.dtype}", flush=True)
    sha_matriz = sha256_bytes(np.ascontiguousarray(matriz).tobytes())
    if sha_matriz != MATRIZ_SHA_PREVIO:
        print("[STOP] la matriz 279 no coincide con la congelada en V32-E1", flush=True)
        return 1

    import catboost
    import lightgbm
    versiones = {"catboost": catboost.__version__, "lightgbm": lightgbm.__version__,
                 "numpy": np.__version__}

    hashes = {
        "canonico_y_true": sha256_archivo(CANONICO / "y_true.npy"),
        "canonico_fold_idx": sha256_archivo(CANONICO / "fold_idx.npy"),
        "canonico_v30_prob": sha256_archivo(CANONICO / "v30_prob.npy"),
        "manifest_plugin": sha256_archivo(MANIFEST),
        "matriz_279_bytes": sha_matriz,
    }
    for fuente in FUENTES_A_HASHEAR:
        hashes[f"fuente_{Path(fuente).name}"] = sha256_archivo(ROOT / fuente)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT.parent, check=False,
                            capture_output=True, text=True, timeout=30).stdout.strip()

    congelado = {
        "fase": "V32-E2 (P2) pre-registro congelado", "fecha": FECHA, "commit": commit,
        "experimento": "challenger CatBoost MultiClass sobre matriz_279 (ciclo 2 de la reapertura)",
        "baseline_primario": round(baseline, 7), "referencia_nested": NESTED,
        "pesos_plugin": PESOS, "hashes": hashes, "versiones": versiones,
        "brazos": {
            "A1_catboost": {"config": CATBOOST_CONFIG, "rol": "decisional"},
            "A2_lgbm_twin": {"config": LGBM_CONFIG, "rol": "control de atribucion decisional"},
            "A3_barajado": {"config": "CatBoost A1 con etiquetas del fold-train permutadas "
                                      "(semilla 20261011+fold) antes del split interno; el "
                                      "fold-val conserva etiquetas reales para el OOF",
                            "rol": "control negativo no decisional"},
        },
        "entrenamiento": {
            "split_interno": "90/10 estratificado del fold-train; semilla 20261010+fold; "
                             "StratifiedShuffleSplit(test_size=0.10)",
            "fold_val": "jamas se usa para parar/seleccionar (solo predict_proba para el OOF)",
            "parada": f"ES con od_wait/early_stopping = {OD_WAIT} sobre TotalF1 (CatBoost) y "
                      f"macro-F1 feval (LGBM) en el split interno; use_best_model",
            "truncamiento": "best_iteration >= iterations-150 => flag presupuesto agotado; "
                            "A1 queda capado a MARGINAL; A2 se registra y anota como control "
                            "debilitado",
            "hilos": f"thread_count/num_threads = {HILOS} FIJO (determinismo)",
        },
        "determinismo": {
            "seeds": {"global": SEMILLA, "barajado_etiquetas": SEMILLA_BARAJADO},
            "sonda": "reentrenar fold 0 de A1 y de A2 con la misma config y comparar el OOF "
                     "bit a bit (max|delta p|)",
            "tolerancias": "0 -> exacto; <=1e-9 -> tolerancia declarada; >1e-9 -> segunda "
                           "repeticion, rango publicado y se exige delta_gate > rango; rango "
                           f">= {VARIABILIDAD} o solape con el gate => no concluyente",
        },
        "integracion": {
            "mecanismo": "combinador logistico anidado sobre [L2(v30_prob) || L2(p_brazo)] "
                         "(12 features, L2 por fila con clip 1e-12); LogReg(C=1.0, "
                         "max_iter=2000, lbfgs) por fold (fit fold!=k, predice fold==k)",
            "decision": "argmax(p_comb * w_plugin CONGELADO)",
            "diagnosticos": ["argmax_sin_regla (sin pesos)", "delta vs nested 0.6154643"],
        },
        "gates": {
            "baseline_abort": 5e-7, "delta_min": GATE_DELTA, "folds_min": GATE_FOLDS,
            "control": "delta_int(A1) > delta_int(A2) Y bootstrap pareado de la diferencia "
                       "(F1_combA1 - F1_combA2; 2000; semilla 20261010) con IC95 sin cero",
            "control_negativo": "si delta_int(A3) >= delta_int(A1): NO_ATRIBUIBLE (precedencia)",
            "bootstrap": {"n": N_BOOTSTRAP, "semilla": SEMILLA, "ic95_sin_cero": True},
            "sin_leakage": True,
            "cap_truncamiento": "flag presupuesto_A1_agotado => veredicto no puede ser PROMISING",
        },
        "arbol": {
            "PROMISING": "pasa los gates -> segunda particion congelada (semilla 987654) antes "
                         "de cualquier otra cosa",
            "MARGINAL": "delta > 0 sin gates -> no se adopta; cierre documentado",
            "CLOSED": "delta <= 0 -> se cierra P2; seria el ultimo candidato priorizado "
                      "(P3 FTRL solo con autorizacion)",
            "submission": "jamas automatica; requeriria gate completo + segunda particion + "
                          "autorizacion explicita",
        },
        "ejecucion": {
            "checkpoints": "experiments/v32_03_estado.json por fold (reanudable)",
            "calibracion": f"tras el fold 0 de A1: proyeccion_total = {PROYECCION_FACTOR} x "
                           f"t_fold0; si > {LIMITE_PROYECTADO_H} h -> parar y reportar (exit 6)",
            "timeout_global": f"{LIMITE_GLOBAL_H} h -> parar con estado guardado (exit 7)",
        },
        "compliance": {
            "embeddings": 0, "transformers": 0, "llm": 0, "datos_externos": 0, "test": 0,
            "holdout": 0, "leaderboard": 0, "submission": 0,
            "nota": "0 embeddings; el gemelo LGBM se reentrena desde cero sobre la matriz "
                    "congelada (control interno, no reutiliza pesos del pool); auditoria AST "
                    "del runner: 0 referencias prohibidas y 0 lecturas de test/holdout",
        },
        "presupuesto": "15 fits CPU secuenciales (~9-19 h con contencion; guardas de "
                       "calibracion y timeout declaradas)",
        "variabilidad_entrenamiento": VARIABILIDAD,
        "umbral_alta_confianza": UMBRAL_ALTA_CONFIANZA,
        "documentos_fuente": ["plan/V32_REAPERTURA_PLAN.md", "reports/v32_e1_preregistro.md"],
    }

    lineas = [
        "# V32 - E2 (P2) - Pre-registro congelado: challenger CatBoost sobre la matriz 279",
        "",
        f"**Fecha**: {FECHA}. **Estado**: CONGELADO antes de ejecutar (commit `{commit[:7]}`).",
        (f"**Baseline primario recomputado**: {baseline:.7f} "
         f"(referencia secundaria nested {NESTED})."),
        "",
        "## La pregunta y la hipotesis",
        "",
        "> El boosting con ordered boosting (CatBoost) sobre la matriz 279 congelada produce",
        "> un challenger con informacion complementaria al campeon V30 + plug-in, mas alla de",
        "> la familia boosting ya usada (LightGBM del pool)?",
        "",
        "Ciclo 2 de la reapertura V32 (tras cerrar P1 como NO_ATRIBUIBLE, D-081). Auditoria por",
        "operacion: CatBoost 1.2.10 no aparece en el historial y v20 lo registro como nunca",
        "ejecutado sobre matriz_279.",
        "",
        "## Materiales congelados (SHA256)",
        "",
        "| Artefacto | SHA256 (16) |",
        "|---|---|",
    ]
    for nombre, valor in hashes.items():
        lineas.append(f"| {nombre} | `{valor[:16]}...` |")
    lineas += [
        "",
        "Versiones: " + ", ".join(f"{k} {v}" for k, v in versiones.items()) + ".",
        "",
        "## Brazos",
        "",
        "| Brazo | Config | Rol |",
        "|---|---|---|",
        ("| A1_catboost | CatBoost MultiClass: 3000 iter, lr 0.03, depth 6, l2 3.0, Bernoulli "
         "0.8, border 254, Plain/SymmetricTree, seed 20261010, 32 hilos | decisional |"),
        ("| A2_lgbm_twin | LightGBM con la MISMA matriz/folds/split/ES(150) macro-F1 y el ancla "
         "afinada V29/V30 (56 hojas, depth 7, lr 0.03, min 54, bag 0.78, col 0.6, l1 0.015, "
         "l2 0.0005, deterministic, 32 hilos) | control de atribucion (decisional) |"),
        ("| A3_barajado | CatBoost A1 con etiquetas del fold-train permutadas (20261011) | "
         "control negativo (no decisional) |"),
        "",
        "## Entrenamiento congelado",
        "",
        "- Split interno 90/10 estratificado del fold-train (semilla 20261010+fold); el",
        "  fold-val jamas se usa para parar/seleccionar; solo predict_proba para el OOF.",
        f"- Early stopping con od_wait/early_stopping = {OD_WAIT} sobre TotalF1 (CatBoost) y",
        "  macro-F1 feval (LGBM); use_best_model; la equivalencia TotalF1 <-> macro-F1 se",
        "  verifica en el sanity.",
        "- Guarda de truncamiento: best_iteration >= iterations-150 => flag; A1 capado a",
        "  MARGINAL; A2 se registra como control debilitado.",
        "- Determinismo: semillas fijas; 32 hilos fijos; sonda de reentreno del fold 0 de A1 y",
        "  de A2 con tolerancias declaradas (0 exacto; <=1e-9 declarada; >1e-9 rango y ",
        "  delta_gate > rango; rango >= 0.0008876 => no concluyente).",
        "- Ejecucion resumible por fold (checkpoint JSON); calibracion tras fold 0 de A1",
        (f"  (proyeccion 9 x t_fold0 > {LIMITE_PROYECTADO_H:g} h => parar) y timeout global "
         f"{LIMITE_GLOBAL_H:g} h."),
        "",
        "## Integracion y evaluacion (mecanismo P6/P7, sin cambios)",
        "",
        "- Combinador logistico anidado [L2(v30) || L2(p_brazo)] (12 features); LogReg C=1.0 por",
        "  fold; decision argmax(p_comb * w) con los pesos del plug-in CONGELADOS.",
        "- Standalone macro-F1 del OOF por brazo; complementariedad rescate/dano/neto y alta",
        f"  confianza (max p del campeon >= {UMBRAL_ALTA_CONFIANZA}); diagnosticos sin regla.",
        "",
        "## Gates (todos obligatorios)",
        "",
        "1. baseline recomputado en la misma corrida (aborta si |delta| > 5e-7);",
        f"2. delta_int(A1) >= +{GATE_DELTA:.4f};",
        f"3. folds positivos de A1 >= {GATE_FOLDS}/5;",
        "4. control favorable: delta_int(A1) > delta_int(A2) Y bootstrap pareado de la",
        "   diferencia (F1_combA1 - F1_combA2) con IC95 sin cero;",
        "5. control negativo: si delta_int(A3) >= delta_int(A1) -> NO_ATRIBUIBLE;",
        "6. bootstrap pareado 2000 (semilla 20261010) del delta de A1 sin cero;",
        "7. sin leakage (asserts del runner) y sin truncamiento de A1 para PROMISING.",
        "",
        "## Arbol de decision",
        "",
        "| Resultado | Estado | Siguiente paso |",
        "|---|---|---|",
        "| gates completos | PROMISING | segunda particion congelada (semilla 987654) |",
        "| delta > 0 sin gates | MARGINAL | no se adopta; cierre documentado |",
        ("| delta <= 0 | CLOSED | se cierra P2 (ultimo candidato priorizado; P3 solo con "
         "autorizacion) |"),
        "",
        "## Compliance y limites",
        "",
        "- 0 embeddings, 0 transformers, 0 LLM, 0 datos externos; sin test, sin holdout, sin",
        "  leaderboard, sin submission; un script nuevo; campeon intacto.",
        "- El gemelo LGBM es un control interno reentrenado desde cero (no reutiliza pesos del",
        "  pool); auditoria AST del runner (0 prohibidos, 0 lecturas de test/holdout).",
        f"- Referencia de variabilidad de entrenamiento: {VARIABILIDAD}.",
    ]
    texto = "\n".join(lin.rstrip() for lin in lineas) + "\n"
    PREREG_MD.write_text(texto, encoding="utf-8")
    print(f"  {PREREG_MD.name}: {len(texto.splitlines())} lineas", flush=True)

    PREREG_JSON.write_text(json.dumps(congelado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  {PREREG_JSON.name}", flush=True)

    runlog = ROOT / "RUNLOG.md"
    marca = "Pre-registro de V32-E2"
    if marca not in runlog.read_text(encoding="utf-8"):
        bloque = ("\n---\n\n## Pre-registro de V32-E2 (P2 CatBoost) · " + FECHA + "\n\n"
                  "- Congelado: A1 CatBoost MultiClass sobre matriz_279; A2 gemelo LightGBM con "
                  "el ancla afinada\n  V29/V30 (control decisional); A3 etiquetas barajadas; "
                  "ES interno (od_wait 150); 32 hilos;\n  integracion P6/P7; gates + bootstrap "
                  "de la diferencia A1-A2; calibracion y timeout declarados.\n"
                  f"- Baseline recomputado: {baseline:.7f}. Revisiones de ml-architect y "
                  "kaggle-specialist incorporadas.\n"
                  "- Sin test, sin holdout, sin submission.\n")
        with runlog.open("a", encoding="utf-8") as f:
            f.write(bloque)
        print("  RUNLOG.md: pre-registro anotado", flush=True)

    ruta_exp = ROOT / "experiments.csv"
    lineas_exp = ruta_exp.read_text(encoding="utf-8").splitlines()
    cols = [c.strip() for c in lineas_exp[0].split(",")]
    if "V32-E2-PREREG" not in {lin.split(",", 1)[0] for lin in lineas_exp[1:] if lin}:
        reg = {"experiment_id": "V32-E2-PREREG", "fecha": FECHA, "bloque": "32.2",
               "etapa": "V32-E2 (P2) pre-registro congelado del challenger CatBoost + gemelo LGBM",
               "pregunta": "CatBoost sobre matriz_279 aporta complementariedad al campeon mas "
                           "alla de la familia boosting ya usada?",
               "hipotesis": "el ordered boosting aporta informacion distinta del LGBM afilado "
                            "del pool",
               "features": "matriz_279 congelada (169336x279); 0 embeddings",
               "preprocessing": "StandardScaler no aplica (GBDT); split interno 90/10",
               "representacion": "features numericas congeladas del meta-espacio",
               "modelo": "CatBoost 1.2.10 MultiClass (A1) / LightGBM gemelo ancla V29-V30 (A2) "
                         "/ CatBoost barajado (A3)",
               "hiperparametros": "congelados: 3000 iter cap, lr 0.03, depth 6, l2 3.0, ES 150, "
                                  "32 hilos, seed 20261010",
               "validacion": "particion canonica; integracion P6/P7; bootstrap de la diferencia "
                             "A1-A2; gates +0.0015/4-5; calibracion y timeout",
               "macro_f1_mean": "", "macro_f1_std": "", "weighted_f1_mean": "",
               "micro_f1_mean": "", "accuracy_mean": "",
               "resultado": "pre-registro escrito y congelado; sin resultados aun",
               "interpretacion": "protocolo del ml-architect + condiciones del kaggle-specialist",
               "decision": "ejecutar V32-E2 tal cual",
               "duracion_seg": "0.0",
               "artefactos": "reports/v32_p2_preregistro.md | "
                             "experiments/v32_02_preregistro_catboost.json"}
        import csv as _csv
        import io as _io
        buf = _io.StringIO()
        esc = _csv.writer(buf, lineterminator="\n")
        esc.writerow([reg[c] for c in cols])
        with ruta_exp.open("a", encoding="utf-8") as f:
            f.write(buf.getvalue())
        print("  experiments.csv: fila V32-E2-PREREG", flush=True)

    print(f"\n[fin] {time.time() - inicio:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
