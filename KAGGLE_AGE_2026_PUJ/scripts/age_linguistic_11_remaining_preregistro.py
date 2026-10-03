"""P3/P4/P5 - pre-registro CONGELADO de las familias restantes de la serie.

Se escribe ANTES de ejecutar el ciclo final. Congela las tres representaciones (R3 lemma_POS,
R4 morfologia, R5 dependencias), el protocolo (identico a P6/P7), el control, los gates y el
arbol de decision. Registra la clasificacion previa de cada familia, la decision del usuario de
ejecutarlas y los puntos adoptados de las revisiones de agentes. Nada se ajusta mirando el
resultado. No toca test, no toca holdout, sin leaderboard, sin submission.

Diferencia declarada con P2/P6: el control barajado permuta DENTRO de cada fold (como P7), por
lo que sus deltas barajados no son numericamente comparables con los globales de P2/P6; el gate
no cambia.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

FECHA = "2026-09-30"
ANOTACION_JSON = cfg.EXPERIMENTS / "age_linguistic_01_anotacion.json"
DEP_JSON = cfg.EXPERIMENTS / "age_linguistic_10_dep_edges.json"
CANONICO = ROOT / "artifacts" / "p33_scores" / "todos"
MANIFEST = ROOT / "reports" / "age_submission_plugin_manifest.json"
REGISTRO_CSV = cfg.REPORTS / "linguistic_annotation_registry.csv"
PREREG_MD = cfg.REPORTS / "age_remaining_channels_preregistro.md"
PREREG_JSON = cfg.EXPERIMENTS / "age_linguistic_11_remaining_preregistro.json"

N_CLASES = 6
SEMILLA_BOOTSTRAP = 20261010
SEMILLA_BARAJADO = 20261031
GATE_DELTA = 0.0015
GATE_FOLDS = 4
UMBRAL_ALTA_CONFIANZA = 0.60
VARIABILIDAD = 0.0008876
N_BOOTSTRAP = 2000
ANOTADORES = ("md", "lg")
REPRESENTACIONES = {
    "R3_lemma_POS": {
        "definicion": "tokens lemma_POS de la columna congelada lemma_pos_seq (querer_VERB)",
        "vec": "TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df=5, sublinear_tf=True, "
               "lowercase=True, token_pattern=r'(?u)\\S+')",
        "nota": "es el bloque B de P7 SIN el bloque de texto: canal tipado aislado"},
    "R4_morfologia": {
        "definicion": "tokens de morph_seq con los vacios '_' EXCLUIDOS (bundles tipo "
                      "Tense=Past|Person=1, en orden; los bigramas son adyacencia en la "
                      "secuencia FILTRADA)",
        "vec": "identico a R3",
        "nota": "excluir '_' evita que el canal mida longitud/posicion; los docs vacios suben "
                "(se cuentan y se declaran; uniforme 1/6)"},
    "R5_dependencias": {
        "definicion": "tokens 'dep->POS_cabeza' por token (nsubj->VERB, det->NOUN; raiz "
                      "'ROOT->POS' como codificacion propia) del artefacto derivado determinista "
                      "dep_edges (mismo modelo congelado; validacion 0/0 contra lo almacenado)",
        "vec": "identico a R3",
        "nota": "etiqueta de arco (relacion + categoria de la cabeza); no es un triple historico; "
                "pierde el POS del dependiente por decision fija"},
}
APRENDIZ = "LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs', tol=1e-3)"
INTEGRACION = ("combinador logistico anidado sobre [p_V30 || p_canal] L2 por fila (12 features), "
               "decision argmax(w_plug-in * p_combinada); identico a P6/P7")
CONTROL = ("secuencias del canal permutadas DENTRO DE CADA FOLD (semilla 20261031; mismo mapeo "
           "md/lg); pipeline completo; NO comparable numericamente con los barajados globales "
           "de P2/P6 (declarado)")
CLASIFICACION_PREVIA = {
    "P3_lemma_POS": "PARTIALLY_COVERED (auditoria del ciclo P6: lemas solo como agregados en "
                    "v14a; incremento lexical cubierto; incremento POS cerrado en P2)",
    "P4_morfologia": "PARTIALLY_COVERED (ratios en v14a; sufijos redundantes con char_wb medido "
                     "en V13)",
    "P5_dependencias": "GENUINELY_NEW por operacion; sin justificacion independiente tras P7 "
                       "(revision unica del gap)",
}


def sha256(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def vacios_por_canal() -> dict:
    """Cuenta documentos con secuencia vacia por canal y anotador (dato congelado)."""
    resultado = {}
    for clave in ("md", "lg"):
        frame = pd.read_parquet(ROOT / "features" / "linguistic"
                                / f"anotacion_{clave}_train.parquet",
                                columns=["lemma_pos_seq", "morph_seq"])
        r3 = frame["lemma_pos_seq"].astype(str)
        r4 = frame["morph_seq"].astype(str).apply(
            lambda b: " ".join(t for t in b.split() if t != "_"))
        dep = pd.read_parquet(ROOT / "features" / "linguistic"
                              / f"dep_edges_{clave}_train.parquet", columns=["dep_edge_seq"])
        r5 = dep["dep_edge_seq"].astype(str)
        resultado[clave] = {"R3": int((r3.str.len() == 0).sum()),
                            "R4": int((r4.str.len() == 0).sum()),
                            "R5": int((r5.str.len() == 0).sum())}
    return resultado


def main() -> int:
    autotest_numerico()
    inicio = time.time()
    anot = json.loads(ANOTACION_JSON.read_text(encoding="utf-8"))
    dep = json.loads(DEP_JSON.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    pesos = [float(x) for x in manifest["weights"]]
    if len(pesos) != N_CLASES:
        print("[STOP] pesos sin 6 clases", flush=True)
        return 1

    hashes = {
        "anotacion_md_parquet": anot["fichas"]["md"]["sha256_parquet"],
        "anotacion_lg_parquet": anot["fichas"]["lg"]["sha256_parquet"],
        "dep_edges_md_parquet": dep["fichas"]["md"]["sha256"],
        "dep_edges_lg_parquet": dep["fichas"]["lg"]["sha256"],
        "corpus_train": sha256(ROOT / "data" / "raw" / "age_train.csv"),
        "canonico_y_true": sha256(CANONICO / "y_true.npy"),
        "canonico_fold_idx": sha256(CANONICO / "fold_idx.npy"),
        "canonico_v30_prob": sha256(CANONICO / "v30_prob.npy"),
        "manifest_plugin": sha256(MANIFEST),
    }
    vacios = vacios_por_canal()

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
    print(f"baseline primario: {baseline:.7f} (esperado 0.6159732)", flush=True)
    if abs(baseline - 0.6159732) > 5e-7:
        print("[STOP] baseline no reproduce", flush=True)
        return 1

    lineas = [
        "# Pre-registro congelado - ciclo final: familias restantes (P3 / P4 / P5)",
        "",
        f"**Fecha**: {FECHA}. **Estado**: CONGELADO antes de ejecutar (commit y hashes en",
        "`experiments/age_linguistic_11_remaining_preregistro.json`).",
        "",
        "## Contexto y decision del usuario",
        "",
        "Las tres familias quedaron sin ejecutar en los ciclos anteriores con estas",
        "clasificaciones:",
        "",
        "| Familia | Clasificacion previa (auditorias por operacion) |",
        "|---|---|",
    ]
    for clave, valor in CLASIFICACION_PREVIA.items():
        lineas.append(f"| {clave} | {valor} |")
    lineas += [
        "",
        "**Decision del usuario (2026-09-30)**: ejecutar las experimentaciones restantes de la",
        "serie, con el mismo protocolo congelado de P6/P7, para cerrar el expediente con evidencia",
        "propia de cada familia. Se registra como override explicito de la recomendacion de no",
        "ejecutar (se documenta en D-080).",
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
        f"Pesos congelados del plug-in: `{pesos}`",
        f"Baseline primario recomputado: **{baseline:.7f}** (brazo A = ablacion sin canal, "
        "D-076 cond.4).",
        "",
        "## Documentos con secuencia vacia por canal (congelado; uniforme 1/6 declarado)",
        "",
        "| Anotador | R3 | R4 (tras excluir '_') | R5 |",
        "|---|---|---|---|",
    ]
    for clave in ANOTADORES:
        v = vacios[clave]
        lineas.append(f"| {clave} | {v['R3']} | {v['R4']} | {v['R5']} |")
    lineas += [
        "",
        "## Representaciones (congeladas, sin grid)",
        "",
        "| ID | Definicion | Vectorizador |",
        "|---|---|---|",
    ]
    for rid, rep in REPRESENTACIONES.items():
        lineas.append(f"| {rid} | {rep['definicion']} | {rep['vec']} |")
    lineas += [
        "",
        f"Aprendiz (los tres): {APRENDIZ}, ajustado SOLO en el train de cada fold; OOF en",
        "validacion. Orden de ejecucion congelado: **R3 -> R4 -> R5**.",
        "",
        "## Integracion y control (pre-registrados)",
        "",
        f"- Integracion: {INTEGRACION}.",
        f"- Control: {CONTROL}.",
        "- Filas con secuencia vacia: probabilidad uniforme 1/6 (convencion P2/P6), declarado.",
        "- Anotadores: md y lg (mandato D-076).",
        "",
        "## Gates (identicos a la serie; por familia y anotador)",
        "",
        f"- delta >= **+{GATE_DELTA:.4f}** sobre el baseline primario;",
        f"- folds positivos >= **{GATE_FOLDS}/5**;",
        "- delta real > delta barajado (mismo canal y anotador);",
        "- IC95 bootstrap pareado (2.000, semilla 20261010) sin cero;",
        "- sin leakage (canal OOF por fold; combinador solo con train).",
        "",
        "Veredicto por familia: PROMISING si algun anotador pasa los cinco; MARGINAL si positivo",
        "por debajo del gate; CLOSED si <=0 o el control empata/gana. Si algo pasara: segunda",
        "particion congelada (semilla 987654) y jamas submission sin autorizacion (D-076 cond.5).",
        "",
        "## Revisiones de agentes (puntos adoptados)",
        "",
        "| Agente | Punto | Decision |",
        "|---|---|---|",
        "| Kaggle Specialist | re-analisis legal bajo D-076; registrar artefacto hijo en el registro; "
        "ROOT->POS como codificacion propia | adoptado |",
        "| Kaggle Specialist | declarar barajado intra-fold y su no-comparabilidad con P2/P6 | "
        "adoptado |",
        "| Revisor ML | excluir '_' en R4; contabilizar vacios (uniformes) | adoptado |",
        "| Revisor ML | R5: re-validar dep_seq/n_tokens en el runner; trampas declaradas | "
        "adoptado |",
        "| Revisor ML | R3 vs P7 limpio; vs P2 no comparable (protocolos distintos) | registro |",
        "| Revisor ML | asserts de orden de clases, pesos, SHA de particion y parquet | adoptado |",
        "| Revisor ML | alternativa de rasgos atomicos en R4 | NO adoptada: otra hipotesis, no un "
        "brazo extra |",
        "",
        "## Que NO se hara",
        "",
        "- No hay grids, ni variantes de n-gramas, ni escalas, ni combinadores, ni semillas extra.",
        "- No se toca el test, ni el holdout, ni el leaderboard. Sin submission automatica.",
        "- No se reabren P2, P6 ni P7. El campeon (V30 + plug-in, LB 0.63575) permanece intacto.",
        "",
        "## Interpretacion pre-registrada",
        "",
        "- Un delta plano con barajado igual o mejor es un cierre valido de la familia, no un",
        "  fracaso: significa que el canal no aporta atribucion positiva bajo el mecanismo.",
        "- Si una familia falla: 'el canal X no aporta mejora al campeon bajo el mecanismo",
        "  evaluado' (no una afirmacion sobre toda la informacion linguistica).",
    ]
    texto = "\n".join(l.rstrip() for l in lineas) + "\n"
    PREREG_MD.write_text(texto, encoding="utf-8")
    print(f"  {PREREG_MD.name}: {len(texto.splitlines())} lineas", flush=True)

    PREREG_JSON.write_text(json.dumps({
        "fase": "P3/P4/P5 pre-registro congelado (familias restantes)", "fecha": FECHA,
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT.parent,
                                 capture_output=True, text=True, timeout=30).stdout.strip(),
        "hashes": hashes, "pesos_plugin": pesos,
        "baseline_primario": round(baseline, 7), "representaciones": REPRESENTACIONES,
        "aprendiz": APRENDIZ, "integracion": INTEGRACION, "control": CONTROL,
        "semillas": {"bootstrap": SEMILLA_BOOTSTRAP, "barajado": SEMILLA_BARAJADO},
        "gates": {"delta_min": GATE_DELTA, "folds_min": GATE_FOLDS},
        "vacios_por_canal": vacios, "anotadores": list(ANOTADORES),
        "orden": ["R3_lemma_POS", "R4_morfologia", "R5_dependencias"],
        "clasificacion_previa": CLASIFICACION_PREVIA,
        "override_usuario": "ejecutar las familias restantes pese a P3/P4 PARTIALLY_COVERED y "
                            "P5 sin justificacion independiente; cierre con evidencia propia",
        "umbral_alta_confianza": UMBRAL_ALTA_CONFIANZA, "n_bootstrap": N_BOOTSTRAP,
        "variabilidad_entrenamiento": VARIABILIDAD, "sklearn_version": sklearn.__version__,
        "no_hacer": ["grid", "sub-hipotesis extra", "test", "holdout", "leaderboard",
                     "submission"],
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("  experiments/age_linguistic_11_remaining_preregistro.json", flush=True)

    # registro del artefacto derivado (requisito del Kaggle Specialist)
    filas = list(csv.DictReader(REGISTRO_CSV.open(encoding="utf-8")))
    if not any(f["recurso"].startswith("dep_edges") for f in filas):
        with REGISTRO_CSV.open("a", encoding="utf-8", newline="") as f:
            escritor = csv.DictWriter(f, fieldnames=list(filas[0].keys()), lineterminator="\n")
            escritor.writerow({
                "recurso": "dep_edges (derivacion propia: dep->POS_cabeza)",
                "tipo": "artefacto derivado del anotador autorizado (no es un modelo)",
                "version": "n/a", "idioma": "es", "licencia": "derivado de es_core_news_* (GPL-3.0)",
                "fecha_alta": FECHA,
                "hash_sha256": f"md:{hashes['dep_edges_md_parquet'][:16]}...; "
                               f"lg:{hashes['dep_edges_lg_parquet'][:16]}...",
                "clasificacion": "ALLOWED (etiquetas del mismo anotador D-076)",
                "evidencia": "re-analisis determinista; validacion 0/0 contra dep_seq y n_tokens "
                             "almacenados; script age_linguistic_10_dep_edges.py",
                "condiciones": "congelado antes de modelar; nunca se calcula en test; ROOT->POS es "
                               "codificacion propia"})
        print("  registro: fila de dep_edges anadida", flush=True)
    else:
        print("  registro: dep_edges ya estaba", flush=True)

    runlog = ROOT / "RUNLOG.md"
    if "Pre-registro del ciclo final (P3/P4/P5)" not in runlog.read_text(encoding="utf-8"):
        bloque = ("\n---\n\n## Pre-registro del ciclo final (P3/P4/P5) · " + FECHA + "\n\n"
                  "- Congeladas las tres familias restantes con protocolo P6/P7 (combinador "
                  "anidado, control\n  intra-fold semilla 20261031, gates) y orden R3->R4->R5.\n"
                  f"- Baseline recomputado: {baseline:.7f}. Artefacto dep_edges registrado "
                  "(md 39832126..., lg ed32f5d2...).\n"
                  "- Override del usuario registrado (P3/P4 PARTIALLY_COVERED, P5 sin "
                  "justificacion independiente).\n"
                  "- Sin tuning, sin test, sin holdout, sin submission.\n")
        with runlog.open("a", encoding="utf-8") as f:
            f.write(bloque)
        print("  RUNLOG.md: pre-registro anotado", flush=True)

    ruta_exp = ROOT / "experiments.csv"
    lineas_exp = ruta_exp.read_text(encoding="utf-8").splitlines()
    cols = [c.strip() for c in lineas_exp[0].split(",")]
    if "AGE-LING-REMAINING-PREREG" not in {l.split(",", 1)[0] for l in lineas_exp[1:] if l}:
        reg = {"experiment_id": "AGE-LING-REMAINING-PREREG", "fecha": FECHA, "bloque": "30.5",
               "etapa": "pre-registro congelado de las familias restantes (P3/P4/P5)",
               "pregunta": "los canales restantes aportan al campeon bajo el mecanismo evaluado?",
               "hipotesis": "lemma_POS / morfologia / dependencias como canales tipados",
               "features": "R3 lemma_POS; R4 morph sin '_'; R5 dep->POS_cabeza (derivado)",
               "preprocessing": "columnas congeladas; filtrado '_' en R4; artefacto propio en R5",
               "representacion": "tres canales sparse de tokens tipados",
               "modelo": "LogReg C=1.0 tol=1e-3 (por fold) + combinador anidado + regla congelada",
               "hiperparametros": "congelados por pre-registro (AGE-LING-REMAINING-PREREG)",
               "validacion": "particion canonica; baseline 0.6159732; control intra-fold "
                             "(20261031); gates +0.0015, 4/5, real>barajado, bootstrap",
               "macro_f1_mean": "", "macro_f1_std": "", "weighted_f1_mean": "", "micro_f1_mean": "",
               "accuracy_mean": "",
               "resultado": "pre-registro escrito y congelado; sin resultados aun",
               "interpretacion": "override del usuario registrado; revisiones de agentes adoptadas",
               "decision": "ejecutar el ciclo tal cual",
               "duracion_seg": "0.0",
               "artefactos": "reports/age_remaining_channels_preregistro.md | "
                             "experiments/age_linguistic_11_remaining_preregistro.json"}
        with ruta_exp.open("a", encoding="utf-8") as f:
            f.write(",".join(f'"{reg[c]}"' if ("," in str(reg[c]) or '"' in str(reg[c]))
                             else str(reg[c]) for c in cols) + "\n")
        print("  experiments.csv: fila AGE-LING-REMAINING-PREREG", flush=True)

    print(f"\n[fin] {time.time() - inicio:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
