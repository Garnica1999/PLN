"""P2 - pre-registro CONGELADO: canal POS-secuencia contra el campeon adoptado.

Este documento se escribe ANTES de ejecutar P2 y no se modifica despues. Congela la pregunta,
los materiales (con HAS256 de todo lo que se va a leer), el protocolo exacto, el alfa, los
controles, los gates y el arbol de decision. Cualquier desviacion posterior queda visible por
comparacion contra este pre-registro.

Regla de la fase: nada se ajusta mirando el resultado (ni C, ni n-gramas, ni min_df, ni alfa, ni
los pesos del plug-in, que estan congelados). No se toca test ni holdout. Sin submission
automatica.
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

from src import config as cfg  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

FECHA = "2026-09-29"
ANOTACION_JSON = cfg.EXPERIMENTS / "age_linguistic_01_anotacion.json"
CANONICO = ROOT / "artifacts" / "p33_scores" / "todos"
MANIFEST = ROOT / "reports" / "age_submission_plugin_manifest.json"
PREREG_MD = cfg.REPORTS / "age_linguistic_pos_preregistro.md"
PREREG_JSON = cfg.EXPERIMENTS / "age_linguistic_02_preregistro.json"

N_CLASES = 6
SEMILLA_BOOTSTRAP = 20261010
VARIABILIDAD = 0.0008876
GATE_DELTA = 0.0015
GATE_FOLDS = 4
ALFA_MEZCLA = 0.5
UMBRAL_ALTA_CONFIANZA = 0.60
PARAMS_CANAL = {
    "vectorizador": "TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df=5, "
                    "sublinear_tf=True, lowercase=True, token_pattern=r'(?u)\\S+')",
    "modelo": "LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')",
    "fit": "SOLO en el train de cada fold (vectorizador y modelo por fold)",
    "entrada": "pos_seq del parquet congelado por anotador",
}


def sha256(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def main() -> int:
    autotest_numerico()
    inicio = time.time()
    for ruta in (ANOTACION_JSON, MANIFEST):
        if not ruta.exists():
            print(f"[STOP] falta {ruta}", flush=True)
            return 1

    anot = json.loads(ANOTACION_JSON.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    pesos = [float(x) for x in manifest["weights"]]
    if len(pesos) != N_CLASES:
        print("[STOP] los pesos del manifest no tienen 6 clases", flush=True)
        return 1

    hashes = {
        "anotacion_md_parquet": anot["fichas"]["md"]["sha256_parquet"],
        "anotacion_lg_parquet": anot["fichas"]["lg"]["sha256_parquet"],
        "modelo_md_meta_json": anot["fichas"]["md"]["sha256_meta_json"],
        "modelo_lg_meta_json": anot["fichas"]["lg"]["sha256_meta_json"],
        "canonico_y_true": sha256(CANONICO / "y_true.npy"),
        "canonico_fold_idx": sha256(CANONICO / "fold_idx.npy"),
        "canonico_v30_prob": sha256(CANONICO / "v30_prob.npy"),
        "manifest_plugin": sha256(MANIFEST),
    }

    clase = np.load(CANONICO / "y_true.npy")
    fold_idx = np.load(CANONICO / "fold_idx.npy")
    conteo_folds = dict(zip(*[x.tolist() for x in np.unique(fold_idx, return_counts=True)]))
    print(f"particion canonica: {conteo_folds}", flush=True)
    v30 = np.load(CANONICO / "v30_prob.npy").astype(np.float64)
    v30 = np.clip(v30, 1e-12, 1.0)
    v30 = v30 / v30.sum(axis=1, keepdims=True)
    w = np.array(pesos, dtype=float)
    baseline = float(f1_score(clase, np.argmax(v30 * w, axis=1), average="macro",
                              labels=list(range(N_CLASES))))
    print(f"baseline congelado global recomputado: {baseline:.7f} (esperado 0.6159732)", flush=True)
    if abs(baseline - 0.6159732) > 5e-7:
        print("[STOP] el baseline no reproduce; los artefactos no son los canonicos", flush=True)
        return 1

    lineas = [
        "# Pre-registro congelado - P2: canal POS-secuencia contra el campeon adoptado",
        "",
        f"**Fecha**: {FECHA}. **Estado**: CONGELADO antes de ejecutar (el commit y los hashes se",
        "registran en `experiments/age_linguistic_02_preregistro.json`).",
        "",
        "Autorizado por **D-076** (etiquetas si, vectores no) y por el brief de la fase.",
        "",
        "## La pregunta, una sola",
        "",
        "> `pos_seq` (secuencia de etiquetas POS) contiene informacion que el campeon adoptado",
        "> (V30 + plug-in por clase) **no explota de forma equivalente**, y su integracion por",
        "> promedio neutral de probabilidades mejora el macro-F1 corregido?",
        "",
        "No se pregunta si POS tiene alguna senal (casi cualquier representacion la tiene): se",
        "pregunta si aporta sobre el campeon, medida contra las decisiones de EL.",
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
        f"Pesos congelados del plug-in (manifest oficial): `{pesos}`",
        "",
        f"Referencia primaria (la que decide): **baseline congelado global = {baseline:.7f}**",
        "(la regla que ejecuta la submission, recomputada hoy sobre la particion canonica).",
        "Referencia secundaria de continuidad: nested 0.6154643.",
        "",
        "## Protocolo exacto (sin tuning)",
        "",
        "1. **Particion**: `fold_idx` canonico leido del artefacto persistido (StratifiedGroup",
        "   KFold 5 sobre `dup_clusters`, el mismo con el que se valido el plug-in). Nada se",
        "   reagrupa: los folds se LEEN, no se regeneran.",
        "2. **Canal** (por anotador, md y lg): entrada = `pos_seq` del parquet congelado.",
        f"   - Vectorizador: `{PARAMS_CANAL['vectorizador']}`",
        f"   - Modelo: `{PARAMS_CANAL['modelo']}`",
        "   - Vectorizador y modelo se ajustan **solo** en el train de cada fold; el OOF son las",
        "     probabilidades en el fold de validacion. Filas normalizadas.",
        "3. **Medicion 1** (diagnostico, sin gate): macro-F1 del canal solo.",
        "4. **Medicion 2** (diagnostico): rescate unico / compartido / dano / neto del canal",
        "   contra las decisiones del campeon; rescates unicos sobre errores de alta confianza",
        f"   del campeon (max `p_v30` >= {UMBRAL_ALTA_CONFIANZA}).",
        "5. **Medicion 3 (la que decide)**: `p_mix = 0.5 * p_v30 + 0.5 * p_canal` (filas",
        "   normalizadas); decision = `argmax_c(w_c * p_mix_c)`. Delta primario = macro-F1 de la",
        "   decision menos el baseline primario, sobre los mismos folds. **Alfa congelado en 0.5**",
        "   (neutro; no se busca su optimo).",
        "6. **Controles, definidos ANTES de ver el resultado**:",
        "   - `barajado_filas` (primario): `pos_seq` permutada ENTRE documentos (mismo multiset",
        "     de secuencias, sin asociacion fila-texto); el canal se reentrena igual.",
        "   - `barajado_intra` (secundario): etiquetas permutadas DENTRO de cada documento",
        "     (mismo bag por documento, sin orden).",
        "   Semillas de los controles: barajado_filas = 20261010, barajado_intra = 20261011.",
        "",
        "## Gates (todos obligatorios, para cada anotador)",
        "",
        f"- delta >= **+{GATE_DELTA:.4f}** sobre el baseline primario;",
        f"- folds positivos >= **{GATE_FOLDS}/5**;",
        "- IC95 del bootstrap pareado (2.000 remuestreos, semilla 20261010, unidad la fila) sin",
        "  incluir el cero;",
        "- delta real > delta de `barajado_filas` (el control no puede empatar ni ganar).",
        "",
        "## Que NO se hara",
        "",
        "- No se ajustan C, n-gramas, min_df, vocabulario, alfa ni los pesos del plug-in.",
        "- No se anaden features de longitud, URL o posicion al canal: es POS y solo POS.",
        "- No se toca el test. No se usa el holdout. No se usa el leaderboard. Sin submission",
        "  automatica.",
        "",
        "## Arbol de decision pre-registrado",
        "",
        "| Resultado | Estado | Siguiente paso |",
        "|---|---|---|",
        "| Gate completo | **PROMISING** | segunda particion agrupada (misma receta, semilla 987654,"
        " reejecucion declarada); sin submission sin autorizacion explicita |",
        "| delta positivo pero < gate, o folds < 4/5 | **MARGINAL** | no se adopta; se cierra POS"
        " documentado |",
        "| delta <= 0 | **NEGATIVO** | POS se cierra documentado (regla §38 del brief) |",
        "",
        "## Limites declarados",
        "",
        "- Alfa = 0.5 es el punto neutro; si el resultado es MARGINAL o NEGATIVO, la conclusion",
        "  valida es sobre ESTA integracion (y sobre el canal como fuente), no una refutacion",
        "  universal de 'POS'. El diagnostico de las mediciones 1 y 2 acota la lectura.",
        "- La particion es la canonica; la segunda particion, si se llega a ella, es reejecucion",
        "  (mismos `dup_clusters`, semilla 987654), no una particion independiente.",
    ]
    texto = "\n".join(l.rstrip() for l in lineas) + "\n"
    PREREG_MD.write_text(texto, encoding="utf-8")
    print(f"  {PREREG_MD.name}: {len(texto.splitlines())} lineas", flush=True)

    PREREG_JSON.write_text(json.dumps({
        "fase": "P2 pre-registro congelado", "fecha": FECHA,
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT.parent,
                                 capture_output=True, text=True, timeout=30).stdout.strip(),
        "pregunta": "pos_seq aporta sobre el campeon adoptado y mejora el macro-F1 al integrarse "
                    "por promedio neutral?",
        "hashes": hashes, "pesos_plugin": pesos,
        "baseline_primario": round(baseline, 7), "referencia_nested": 0.6154643,
        "alfa_mezcla": ALFA_MEZCLA, "params_canal": PARAMS_CANAL,
        "gates": {"delta_min": GATE_DELTA, "folds_min": GATE_FOLDS,
                  "bootstrap_semilla": SEMILLA_BOOTSTRAP,
                  "control_primario": "barajado_filas", "control_secundario": "barajado_intra"},
        "semillas_control": {"barajado_filas": 20261010, "barajado_intra": 20261011},
        "umbral_alta_confianza": UMBRAL_ALTA_CONFIANZA,
        "variabilidad_entrenamiento": VARIABILIDAD,
        "no_hacer": ["ajustar C/ngramas/min_df/alfa/pesos", "features extra en el canal",
                     "test", "holdout", "leaderboard", "submission automatica"],
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("  experiments/age_linguistic_02_preregistro.json", flush=True)

    # registros
    runlog = ROOT / "RUNLOG.md"
    marca = "Pre-registro de P2"
    if marca not in runlog.read_text(encoding="utf-8"):
        bloque = ("\n---\n\n## Pre-registro de P2: canal POS-secuencia · " + FECHA + "\n\n"
                  "- Congelado el protocolo: particion canonica leida, canal TF-IDF 1-3 + LogReg "
                  "(C=1.0),\n  alfa=0.5, control de barajado de filas e intra-documento, gates "
                  "(+0.0015, 4/5,\n  bootstrap, control) y arbol de decision.\n"
                  f"- Baseline primario recomputado: {baseline:.7f} (regla de la submission).\n"
                  "- Sin tuning, sin test, sin holdout, sin submission automatica.\n")
        with runlog.open("a", encoding="utf-8") as f:
            f.write(bloque)
        print("  RUNLOG.md: pre-registro anotado", flush=True)

    ruta_exp = ROOT / "experiments.csv"
    lineas_exp = ruta_exp.read_text(encoding="utf-8").splitlines()
    cols = [c.strip() for c in lineas_exp[0].split(",")]
    if "AGE-LING-P2-PREREG" not in {l.split(",", 1)[0] for l in lineas_exp[1:] if l}:
        reg = {"experiment_id": "AGE-LING-P2-PREREG", "fecha": FECHA, "bloque": "30.2",
               "etapa": "P2 pre-registro congelado del canal POS-secuencia",
               "pregunta": "pos_seq aporta sobre el campeon y mejora al integrarse por promedio neutral?",
               "hipotesis": "la secuencia de etiquetas POS contiene informacion no explotada por V30",
               "features": "pos_seq (md y lg) -> TF-IDF 1-3", "preprocessing": "ninguno",
               "representacion": "canal sparse de etiquetas POS",
               "modelo": "LogisticRegression C=1.0 (por fold)",
               "hiperparametros": "congelados: (1,3), min_df=5, C=1.0, alfa=0.5, pesos plug-in fijos",
               "validacion": "fold_idx canonico persistido; delta vs baseline congelado global "
                             f"{baseline:.7f}; gates +0.0015, 4/5, bootstrap 2000/20261010, control barajado",
               "macro_f1_mean": "", "macro_f1_std": "", "weighted_f1_mean": "", "micro_f1_mean": "",
               "accuracy_mean": "",
               "resultado": "pre-registro escrito y congelado; sin resultados aun",
               "interpretacion": "nada se ajustara mirando el resultado",
               "decision": "ejecutar P2 tal cual",
               "duracion_seg": "0.0",
               "artefactos": "reports/age_linguistic_pos_preregistro.md | "
                             "experiments/age_linguistic_02_preregistro.json"}
        with ruta_exp.open("a", encoding="utf-8") as f:
            f.write(",".join(f'"{reg[c]}"' if ("," in str(reg[c]) or '"' in str(reg[c]))
                             else str(reg[c]) for c in cols) + "\n")
        print("  experiments.csv: fila AGE-LING-P2-PREREG", flush=True)

    print(f"\n[fin] {time.time() - inicio:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
