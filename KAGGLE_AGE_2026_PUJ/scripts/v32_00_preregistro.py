"""V32 - E1 - Pre-registro CONGELADO: challenger MLP con perdida surrogate de macro-F1.

## Que congela

El primer experimento de la reapertura V32: un challenger MLP pequeno entrenado sobre la
matriz de 279 features congelada con una perdida **soft-macro-F1** diferenciable, comparado
contra el campeon V30 + plug-in (baseline interno 0.6159732) con el mecanismo de integracion
establecido (combinador logistico anidado [p_V30 || p_challenger] y decision con la regla y
pesos del plug-in CONGELADOS).

## Decisiones incorporadas (revisiones de agentes + Carlos)

- ml-architect: challenger MLP 279->128->6; perdida soft-macro-F1; control gemelo con
  cross-entropy; integracion P6/P7; coste < 30 min.
- kaggle-specialist: APTO con salvaguardas: hashes y asserts de alineacion; assert de OOF por
  fold; early stopping solo con split interno; determinismo; deltas tambien con argmax puro;
  tercer brazo barajado como control negativo.
- dl-specialist: AdamW (weight decay desacoplado), one-hot float32, pasos por epoca calculados
  (ceil(n/lote), no hardcodear), guardas NaN, torch.use_deterministic_algorithms, hilos fijos,
  sanity-check de 1 lote, y recomendacion de early stopping estricto.
- Carlos: **early stopping estricto** (opcion A) y **entrenamiento en GPU** (D-007, guardian).

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
PREREG_MD = cfg.REPORTS / "v32_e1_preregistro.md"
PREREG_JSON = cfg.EXPERIMENTS / "v32_00_preregistro.json"

BASELINE = 0.6159732
NESTED = 0.6154643
PESOS = [1.0, 1.1749, 1.2721, 1.163, 1.001, 0.8128]
N_CLASES = 6
LOTE = 8192
LR = 1e-3
WD = 1e-4
EPOCAS_MAX = 60
PACIENCIA = 8
FRACCION_INTERNA = 0.10
GRAD_CLIP = 5.0
EPS_SOFT = 1e-6
SEMILLA = 20261010
SEMILLA_BARAJADO = 20261011
HILOS_CPU = 8
N_BOOTSTRAP = 2000
GATE_DELTA = 0.0015
GATE_FOLDS = 4
UMBRAL_ALTA_CONFIANZA = 0.60
VARIABILIDAD = 0.0008876
FUENTES_A_HASHEAR = ("scripts/v19_b1_boosting.py", "scripts/v30_02_ensemble.py",
                     "scripts/age_umbrales_por_clase.py")


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

    import torch
    gpu_nombre = "n/a"
    try:
        if torch.cuda.is_available():
            gpu_nombre = torch.cuda.get_device_name(0)
    except Exception as error:  # noqa: BLE001
        gpu_nombre = f"no disponible ({type(error).__name__})"

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
        "fase": "V32-E1 pre-registro congelado", "fecha": FECHA, "commit": commit,
        "experimento": "challenger MLP con perdida surrogate de macro-F1",
        "baseline_primario": round(baseline, 7), "referencia_nested": NESTED,
        "pesos_plugin": PESOS, "hashes": hashes,
        "dispositivo": {
            "entrenamiento": "GPU (ROCm) con guardian D-007 (require_gpu, wait acotado; "
                             "sin fallback silencioso a CPU)",
            "gpu_detectada": gpu_nombre, "torch": torch.__version__,
            "matmul_precision": "highest",
        },
        "arquitectura": "StandardScaler(fit solo fold-train) -> Linear(279,128) -> GELU -> "
                        "Dropout(0.1) -> Linear(128,6) -> softmax; orden canonico de clases",
        "perdida_soft_macro_f1": {
            "formula": "p=softmax(z) clamp [1e-6,1-1e-6]; TP_c=sum(p*y); FP_c=sum(p*(1-y)); "
                       "FN_c=sum((1-p)*y); P=TP/(TP+FP+1e-6); R=TP/(TP+FN+1e-6); "
                       "F1=2PR/(P+R+1e-6); L=1-media_c F1_c",
            "objetivos": "one-hot float32", "eps": EPS_SOFT},
        "entrenamiento": {
            "lote": LOTE, "pasos_por_epoca": "calculado: ceil(n_train/lote), sin hardcodear",
            "optimizador": f"AdamW lr={LR} betas=(0.9,0.999) eps=1e-8 wd={WD}",
            "epocas_max": EPOCAS_MAX,
            "early_stopping": {
                "split_interno": f"{1-FRACCION_INTERNA:.0%}/10% estratificado por clase del "
                                 "fold-train; semilla = 20261010 + fold",
                "paciencia": PACIENCIA,
                "restaurar": "mejor macro-F1 interna (hard argmax en el split interno)",
                "fold_val": "jamas se usa para parar (solo OOF)",
            },
            "grad_clip": GRAD_CLIP,
            "guardas_nan": "loss no finita: salta el paso; >=3 en una epoca o >=10 acumuladas: "
                           "aborta el brazo",
            "sanity_1_lote": "100 pasos sobre un lote de 8192: PASS si loss finita y reduccion "
                             ">= 0.30",
            "determinismo": {
                "seeds": {"global": SEMILLA, "barajado_etiquetas": SEMILLA_BARAJADO},
                "shuffle_lote": "torch.Generator por (brazo, fold, epoca); seed = "
                                f"{SEMILLA} + 10**6*brazo + 1000*fold + epoca",
                "algoritmos": "torch.use_deterministic_algorithms(True); si ROCm lanza, "
                              "se registra el fallback warn_only(True) y la sonda de "
                              "determinismo lo mide",
                "hilos_cpu": HILOS_CPU,
                "sonda": "reentrenar fold 0 del brazo A1 con la misma config; comparar el "
                         "OOF bit a bit; si difiere, reportar max|delta p| frente a 8.9e-04",
            },
        },
        "brazos": {
            "A1_surrogate": "MLP con perdida soft-macro-F1 (decisional)",
            "A2_ce": "gemelo identico con cross-entropy media (control de atribucion)",
            "A3_barajado": "gemelo A1 con etiquetas del fold-train permutadas "
                           "(semilla 20261011; control negativo no decisional)",
        },
        "integracion": {
            "mecanismo": "combinador logistico anidado sobre [L2(v30_prob) || L2(p_brazo)] "
                         "(12 features, L2 por fila con clip 1e-12); LogReg(C=1.0, max_iter=2000,"
                         " lbfgs) por fold (fit fold!=k, predice fold==k)",
            "decision": "argmax(p_comb * w_plugin CONGELADO)",
            "diagnosticos": ["argmax_sin_regla (sin pesos)", "delta vs nested 0.6154643"],
        },
        "gates": {
            "baseline_abort": 5e-7, "delta_min": GATE_DELTA, "folds_min": GATE_FOLDS,
            "control": "delta_int(A1) > delta_int(A2) (comparacion numerica pre-registrada)",
            "control_negativo": "si delta_int(A3) >= delta_int(A1): NO_ATRIBUIBLE (no puede "
                                "pasar a segunda particion)",
            "bootstrap": {"n": N_BOOTSTRAP, "semilla": SEMILLA, "ic95_sin_cero": True},
            "sin_leakage": True,
        },
        "arbol": {
            "PROMISING": "pasa los gates -> segunda particion congelada (semilla 987654) antes "
                         "de cualquier otra cosa",
            "MARGINAL": "delta > 0 sin gates -> no se adopta; cierre documentado o P2 con "
                        "justificacion",
            "CLOSED": "delta <= 0 -> se cierra P1; solo entonces se evalua P2 (CatBoost)",
            "submission": "jamas automatica; requeriria gate completo + segunda particion + "
                          "autorizacion explicita",
        },
        "compliance": {
            "embeddings": 0, "transformers": 0, "LLM": 0, "test": 0, "holdout": 0,
            "leaderboard": 0, "submissions": 0, "datos_externos": 0,
            "nota": "0 embeddings en ambos brazos; las activaciones ocultas nunca se exportan; "
                    "el integrador solo consume las 6 probabilidades",
        },
        "presupuesto": "3 brazos x 5 folds con ES (<=60 epocas, ~17 pasos/epoca) en GPU: minutos; "
                       "combinadores + bootstrap en CPU: ~5-10 min; total < 1 h",
        "variabilidad_entrenamiento": VARIABILIDAD,
        "umbral_alta_confianza": UMBRAL_ALTA_CONFIANZA,
        "documentos_fuente": ["plan/V32_REAPERTURA_PLAN.md"],
    }

    lineas = [
        "# V32 - E1 - Pre-registro congelado: challenger con perdida surrogate de macro-F1",
        "",
        f"**Fecha**: {FECHA}. **Estado**: CONGELADO antes de ejecutar (commit `{commit[:7]}`).",
        ("**Baseline primario recomputado**: "
         f"{baseline:.7f} (referencia secundaria nested {NESTED})."),
        "",
        "## La pregunta y la hipotesis",
        "",
        "> Entrenar un modelo con una perdida que **directamente** optimiza macro-F1 (surrogate",
        "> diferenciable) produce un challenger con informacion complementaria al campeon",
        "> V30 + plug-in, y mejora el macro-F1 corregido al integrarse con el mecanismo establecido?",
        "",
        "Es la unica operacion priorizada que nunca se implemento (familia 'objetivo': solo",
        "cross-entropy en 95 scripts). P2 (CatBoost) y P3 (FTRL) solo se evaluan si P1 falla.",
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
        "## Dispositivo y arquitectura",
        "",
        "- Entrenamiento en **GPU (ROCm)** con guardian D-007: `require_gpu(wait acotado)`; si la",
        "  GPU esta ocupada, el script avisa y se detiene (sin fallback silencioso a CPU).",
        f"- GPU detectada: `{gpu_nombre}`; torch `{torch.__version__}`; matmul precision `highest`.",
        "- Red: StandardScaler (fit solo fold-train) -> Linear(279,128) -> GELU -> Dropout(0.1)",
        "  -> Linear(128,6) -> softmax. Orden canonico de clases (demostrado por diagonal).",
        "",
        "## Perdida (soft-macro-F1)",
        "",
        "```text",
        "p = softmax(z), clamp [1e-6, 1-1e-6]; objetivos one-hot float32",
        "TP_c = sum(p * y);  FP_c = sum(p * (1-y));  FN_c = sum((1-p) * y)",
        "P = TP/(TP+FP+1e-6);  R = TP/(TP+FN+1e-6);  F1 = 2PR/(P+R+1e-6)",
        "L = 1 - media_c F1_c",
        "```",
        "",
        "## Entrenamiento congelado",
        "",
        (f"- Lote {LOTE}; pasos por epoca **calculados** (ceil(n/lote)); optimizador "
         f"**AdamW** lr={LR},"),
        f"  wd={WD}; max {EPOCAS_MAX} epocas; grad-clip {GRAD_CLIP}; shuffle con generador por",
        "  (brazo, fold, epoca) con semilla fija.",
        "- **Early stopping estricto (decision de Carlos)**: split interno 90/10 estratificado del",
        "  fold-train (semilla 20261010+fold), paciencia 8, se restauran los mejores pesos por",
        "  macro-F1 interna. El fold-val jamas se usa para parar.",
        "- Guardas: loss no finita -> salta paso; >=3 en una epoca o >=10 acumuladas -> aborta brazo.",
        "- Sanity-check de 1 lote (100 pasos): PASS si loss finita y reduccion >= 0.30.",
        "- Determinismo: seeds fijos; `torch.use_deterministic_algorithms(True)` (fallback",
        "  warn_only registrado si ROCm lo exige); `torch.set_float32_matmul_precision('highest')`;",
        f"  {HILOS_CPU} hilos CPU; sonda de determinismo sobre fold 0 del brazo A1.",
        "",
        "## Brazos",
        "",
        "| Brazo | Perdida | Rol |",
        "|---|---|---|",
        "| A1_surrogate | soft-macro-F1 | decisional |",
        "| A2_ce | cross-entropy media (gemelo identico) | control de atribucion |",
        ("| A3_barajado | soft-macro-F1 con etiquetas permutadas (20261011) | control negativo "
         "no decisional |"),
        "",
        "## Integracion y evaluacion (mecanismo P6/P7, sin cambios)",
        "",
        "- Combinador logistico anidado sobre [L2(v30_prob) || L2(p_brazo)] (12 features); LogReg",
        "  C=1.0 fit por fold; decision `argmax(p_comb * w)` con los pesos del plug-in CONGELADOS.",
        "- Standalone: macro-F1 del argmax del OOF por brazo (global y por fold).",
        "- Complementariedad vs campeon: rescate/daño/neto; alta confianza = max p del campeon",
        f"  >= {UMBRAL_ALTA_CONFIANZA}.",
        "- Diagnosticos: `argmax_sin_regla` y delta vs nested. No se re-derivan los pesos.",
        "",
        "## Gates (todos obligatorios)",
        "",
        "1. baseline recomputado en la misma corrida (aborta si |delta| > 5e-7);",
        f"2. delta_int(A1) >= +{GATE_DELTA:.4f};",
        f"3. folds positivos de A1 >= {GATE_FOLDS}/5;",
        "4. control favorable: delta_int(A1) > delta_int(A2) (comparacion pre-registrada);",
        "5. control negativo: si delta_int(A3) >= delta_int(A1) -> NO_ATRIBUIBLE;",
        "6. bootstrap pareado 2000 (semilla 20261010) sin cero;",
        "7. sin leakage (asserts del runner: alineacion, OOF por fold, split interno).",
        "",
        "## Arbol de decision",
        "",
        "| Resultado | Estado | Siguiente paso |",
        "|---|---|---|",
        ("| gates completos | PROMISING | segunda particion congelada (semilla 987654) antes de "
         "cualquier otra cosa |"),
        ("| delta > 0 sin gates | MARGINAL | no se adopta; cierre documentado o P2 con "
         "justificacion |"),
        "| delta <= 0 | CLOSED | se cierra P1; solo entonces se evalua P2 (CatBoost) |",
        "",
        "## Compliance y limites",
        "",
        "- 0 embeddings (ambos brazos), 0 transformers, 0 LLM, 0 datos externos; sin test, sin",
        "  holdout, sin leaderboard, sin submission; un script nuevo; campeon intacto.",
        "- Las activaciones ocultas nunca se exportan como features (solo las 6 probabilidades).",
        f"- Referencia de variabilidad de entrenamiento: {VARIABILIDAD}.",
    ]
    texto = "\n".join(l.rstrip() for l in lineas) + "\n"
    PREREG_MD.write_text(texto, encoding="utf-8")
    print(f"  {PREREG_MD.name}: {len(texto.splitlines())} lineas", flush=True)

    PREREG_JSON.write_text(json.dumps(congelado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  {PREREG_JSON.name}", flush=True)

    runlog = ROOT / "RUNLOG.md"
    marca = "Pre-registro de V32-E1"
    if marca not in runlog.read_text(encoding="utf-8"):
        bloque = ("\n---\n\n## Pre-registro de V32-E1: challenger con perdida surrogate de "
                  "macro-F1 · " + FECHA + "\n\n"
                  "- Congelado: MLP 279->128->6 sobre matriz_279; perdida soft-macro-F1; control "
                  "gemelo CE;\n  tercer brazo barajado; early stopping estricto (90/10 "
                  "interno) y GPU (D-007);\n  integracion P6/P7 con regla y pesos del plug-in "
                  "congelados.\n"
                  f"- Baseline recomputado: {baseline:.7f}. Revisiones de ml-architect, "
                  "kaggle-specialist y dl-specialist incorporadas.\n"
                  "- Sin test, sin holdout, sin submission.\n")
        with runlog.open("a", encoding="utf-8") as f:
            f.write(bloque)
        print("  RUNLOG.md: pre-registro anotado", flush=True)

    ruta_exp = ROOT / "experiments.csv"
    lineas_exp = ruta_exp.read_text(encoding="utf-8").splitlines()
    cols = [c.strip() for c in lineas_exp[0].split(",")]
    if "V32-E1-PREREG" not in {l.split(",", 1)[0] for l in lineas_exp[1:] if l}:
        reg = {"experiment_id": "V32-E1-PREREG", "fecha": FECHA, "bloque": "32.0",
               "etapa": "V32-E1 pre-registro congelado del challenger con perdida surrogate de macro-F1",
               "pregunta": "una perdida que optimiza macro-F1 directamente produce un challenger "
                           "complementario al campeon?",
               "hipotesis": "alinear entrenamiento y metrica aporta informacion que el campeon "
                            "no explota",
               "features": "matriz_279 congelada (169336x279)",
               "preprocessing": "StandardScaler por fold (fit solo fold-train)",
               "representacion": "features numericas congeladas del meta-espacio",
               "modelo": "MLP 279-128-6 (GELU, dropout 0.1) en PyTorch GPU; AdamW lr=1e-3",
               "hiperparametros": "congelados: lote 8192, epochs<=60, ES paciencia 8, clip 5.0",
               "validacion": "particion canonica; integracion P6/P7; control gemelo CE; control "
                             "barajado; gates +0.0015/4-5/bootstrap",
               "macro_f1_mean": "", "macro_f1_std": "", "weighted_f1_mean": "", "micro_f1_mean": "",
               "accuracy_mean": "",
               "resultado": "pre-registro escrito y congelado; sin resultados aun",
               "interpretacion": "decisiones de Carlos (ES + GPU) y revisiones de agentes registradas",
               "decision": "ejecutar V32-E1 tal cual",
               "duracion_seg": "0.0",
               "artefactos": "reports/v32_e1_preregistro.md | experiments/v32_00_preregistro.json"}
        with ruta_exp.open("a", encoding="utf-8") as f:
            f.write(",".join(f'"{reg[c]}"' if ("," in str(reg[c]) or '"' in str(reg[c]))
                             else str(reg[c]) for c in cols) + "\n")
        print("  experiments.csv: fila V32-E1-PREREG", flush=True)

    print(f"\n[fin] {time.time() - inicio:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
