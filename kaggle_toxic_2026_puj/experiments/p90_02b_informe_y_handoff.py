"""P90-02 parte 2: informe integral, handoff para ChatGPT y manifiesto.

Lee los resultados de la auditoria (`reports/p90_02_audit_results.json` y los CSV
`reports/p90_02_*.csv`) y produce:

- `reports/p90_02_integrity_and_corpus_audit.md` (informe integral, 26 secciones);
- `reports/p90_02_chatgpt_handoff.md` (sintesis autocontenida para revision externa);
- `reports/p90_02_audit_manifest.json` (manifiesto de reproducibilidad).

Sin entrenamiento, sin test, sin holdout, sin submissions, sin commits.

Uso:
    /home/carlos/venv/bin/python experiments/p90_02b_informe_y_handoff.py
"""

from __future__ import annotations

import ast
import csv
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

RAIZ: Path = Path(__file__).resolve().parents[1]

RUTA_RESULTS = RAIZ / "reports" / "p90_02_audit_results.json"
RUTA_METRIC = RAIZ / "reports" / "p90_02_metric_reconciliation.csv"
RUTA_TEXT_PROV = RAIZ / "reports" / "p90_02_text_provenance.csv"
RUTA_CB_VAL = RAIZ / "reports" / "p90_02_casebook_validation.csv"
RUTA_SAMPLE_AUDIT = RAIZ / "reports" / "p90_02_review_sample_audit.csv"
RUTA_CORPUS = RAIZ / "reports" / "p90_02_corpus_profile.csv"
RUTA_LABELS = RAIZ / "reports" / "p90_02_label_distribution.csv"
RUTA_REGISTRY = RAIZ / "reports" / "p90_02_slice_registry.csv"
RUTA_SLICE_MET = RAIZ / "reports" / "p90_02_slice_metrics.csv"
RUTA_SLICE_OVER = RAIZ / "reports" / "p90_02_slice_overlap.csv"
RUTA_DIAG = RAIZ / "reports" / "p90_02_error_diagnosis.csv"
RUTA_INTER = RAIZ / "reports" / "p90_02_error_intersections.csv"
RUTA_CASES = RAIZ / "reports" / "p90_02_representative_cases.csv"
RUTA_HIST = RAIZ / "reports" / "p90_02_historical_eda_matrix.csv"
RUTA_HYP = RAIZ / "reports" / "p90_02_hypothesis_status.csv"
RUTA_INV = RAIZ / "reports" / "p90_02_artifact_inventory.csv"
RUTA_FAMILIAS = RAIZ / "reports" / "p57_family_status.csv"
RUTA_CAND = RAIZ / "reports" / "p57_candidate_matrix.csv"
RUTA_P66 = RAIZ / "reports" / "p66_complementarity.md"
RUTA_FOLD = RAIZ / "reports" / "v03_error_fold_summary.csv"
RUTA_OOF = RAIZ / "features" / "oof_v03_final.npz"
RUTA_GRUPOS = RAIZ / "data" / "processed" / "grupos_near_duplicados.csv"

RUTA_INFORME = RAIZ / "reports" / "p90_02_integrity_and_corpus_audit.md"
RUTA_HANDOFF = RAIZ / "reports" / "p90_02_chatgpt_handoff.md"
RUTA_MANIFEST = RAIZ / "reports" / "p90_02_audit_manifest.json"


def leer_csv(ruta: Path, encoding: str = "utf-8") -> list[dict[str, str]]:
    """Lee un CSV respetando saltos de linea embebidos."""
    with ruta.open(encoding=encoding, newline="") as fh:
        return list(csv.DictReader(fh))


def sha256_de(ruta: Path) -> str:
    """Calcula el sha256 de un archivo."""
    import hashlib
    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1 << 20), b""):
            digest.update(bloque)
    return digest.hexdigest()


def tabla(cabeceras: list[str], filas: list[list[Any]]) -> str:
    """Construye una tabla markdown."""
    lineas = ["| " + " | ".join(cabeceras) + " |",
              "|" + "|".join("---" for _ in cabeceras) + "|"]
    for fila in filas:
        lineas.append("| " + " | ".join(str(c) for c in fila) + " |")
    return "\n".join(lineas)


def escribir_texto(ruta: Path, texto: str) -> None:
    """Escribe un archivo de texto exigiendo ASCII puro."""
    if any(ord(caracter) > 127 for caracter in texto):
        raise SystemExit(f"STOP: bytes > 127 en {ruta.name}")
    ruta.write_text(texto, encoding="utf-8")


def cargar_todo() -> dict[str, Any]:
    """Carga resultados y tablas de la auditoria."""
    d = json.loads(RUTA_RESULTS.read_text(encoding="utf-8"))
    datos: dict[str, Any] = {
        "d": d, "met": d["metricas"], "cb": d["casebook"], "corp": d["corpus"],
        "diag": d["diagnostico"], "texto": d["texto"],
        "inv": leer_csv(RUTA_INV), "prov": leer_csv(RUTA_TEXT_PROV),
        "mrec": leer_csv(RUTA_METRIC), "cbval": leer_csv(RUTA_CB_VAL),
        "samp": leer_csv(RUTA_SAMPLE_AUDIT), "cprof": leer_csv(RUTA_CORPUS),
        "labels": leer_csv(RUTA_LABELS), "reg": leer_csv(RUTA_REGISTRY),
        "smet": leer_csv(RUTA_SLICE_MET), "sover": leer_csv(RUTA_SLICE_OVER),
        "diagf": leer_csv(RUTA_DIAG), "inter": leer_csv(RUTA_INTER),
        "casos": leer_csv(RUTA_CASES), "hist": leer_csv(RUTA_HIST),
        "hyp": leer_csv(RUTA_HYP), "familias": leer_csv(RUTA_FAMILIAS, "utf-8-sig"),
        "cand": leer_csv(RUTA_CAND, "utf-8-sig"), "fold": leer_csv(RUTA_FOLD),
    }
    comp = RUTA_P66.read_text(encoding="utf-8")
    for marca in ("0.0026743167", "0.0041133588", "0.0027270194", "[-0.0044311634, +0.0102108745]"):
        if marca not in comp:
            raise SystemExit(f"STOP: cifra de p66_complementarity.md no encontrada: {marca}")
    datos["comp_md"] = comp
    # Grupos near-dup no unitarios en desarrollo (detalle verificado).
    z = np.load(RUTA_OOF, allow_pickle=True)
    dev = {int(i) for i in z["ids"]}
    no_unit = []
    with RUTA_GRUPOS.open(encoding="utf-8-sig", newline="") as fh:
        for f in csv.DictReader(fh):
            ident = int(str(f["id"]).strip('"'))
            if ident in dev and int(f["tamano_grupo"]) > 1:
                no_unit.append((ident, int(f["grupo"]), int(f["tamano_grupo"])))
    datos["grupos_no_unit"] = no_unit
    # F1 por fold y tamano de fold.
    fold_f1: dict[tuple[str, int], float] = {}
    for f in datos["mrec"]:
        m = re.match(r"F1_fold(\d)_(HS|TR|AG)$", f["metric_name"])
        if m:
            fold_f1[(m.group(2), int(m.group(1)))] = float(f["recomputed_value"])
    datos["fold_f1"] = fold_f1
    fold_n: dict[int, int] = {}
    for f in datos["fold"]:
        fold_n.setdefault(int(f["fold"]), int(f["n"]))
    datos["fold_n"] = fold_n
    # Diagnostico por etiqueta y conjuntos.
    datos["diag_etq"] = {f["clave"]: json.loads(f["valor"]) for f in datos["diagf"] if f["bloque"] == "etiqueta"}
    datos["combos_diag"] = [(f["clave"], f["valor"]) for f in datos["diagf"] if f["bloque"] == "conjunto"]
    datos["triple_dist"] = [(f["clave"], f["valor"]) for f in datos["diagf"] if f["bloque"] == "conjunto_distancia"]
    datos["hipotesis"] = [(f["clave"], f["valor"]) for f in datos["diagf"] if f["bloque"] == "hipotesis"]
    # Slices: pivotes dentro/fuera y subconjuntos de error.
    f1_in: dict[tuple[str, str], str] = {}
    f1_out: dict[tuple[str, str], str] = {}
    err_cells: dict[tuple[str, str], tuple[int, int, int]] = {}
    err_comp: dict[str, str] = {}
    for f in datos["smet"]:
        if f["familia"] == "SLICE_ERROR_CONDITIONED":
            err_cells[(f["slice"], f["label"])] = (int(f["n_error_cells"]), int(f["FP"]), int(f["FN"]))
            m = re.search(r"= (\d+)/(\d+)/(\d+)$", f["nota"])
            if m:
                err_comp[f["slice"]] = f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
        else:
            (f1_in if f["ambito"] == "dentro" else f1_out)[(f["slice"], f["label"])] = f["F1"]
    datos["f1_in"] = f1_in
    datos["f1_out"] = f1_out
    datos["err_cells"] = err_cells
    datos["err_comp"] = err_comp
    datos["estados"] = Counter(f["estado"].split(":")[0].strip() for f in datos["hyp"])
    return datos


def construir_informe(x: dict[str, Any]) -> str:
    """Construye el informe integral (26 secciones)."""
    d, met, cb, corp, diag = x["d"], x["met"], x["cb"], x["corp"], x["diag"]
    texto = x["texto"]
    mp = met["matrices_pooled"]
    rep_ctl = next(f["observado"] for f in x["cbval"] if f["chequeo"] == "controles_repetidos")

    filas_mat = [[et, mp[et]["TP"], mp[et]["FP"], mp[et]["FN"], mp[et]["TN"],
                  f"{met['f1_por_etiqueta_pooled'][et]:.6f}"] for et in ("HS", "TR", "AG")]
    pre = {f["metric_name"]: f["recomputed_value"] for f in x["mrec"]}
    filas_pr = [[et, pre[f"precision_pooled_{et}"][:10], pre[f"recall_pooled_{et}"][:10],
                 pre[f"F1_pooled_{et}"][:10]] for et in ("HS", "TR", "AG")]
    filas_fold = [[f"fold {k} (n={x['fold_n'].get(k, '?')})"] + [f"{x['fold_f1'][(et, k)]:.6f}" for et in ("HS", "TR", "AG")]
                  for k in range(1, 6)]
    filas_agreg = [
        ["corrected_cv (canonica)", met["corrected_cv_reproducido"][:18], "macro pooled: media de F1 pooled por etiqueta"],
        ["media entre folds por etiqueta", json.dumps(met["media_folds_por_etiqueta"]), "media aritmetica de 5 F1 por etiqueta"],
        ["media de f1_pliegues", met["media_macro_pliegues"], "media de los 5 macro-F1 por fold"],
    ]
    filas_inv = [[f["artifact_id"], f["path"], f["sha256"][:16], f["git_tracked"], f["status"]] for f in x["inv"]]
    filas_prov = [[f["chequeo"], f["resultado"][:90], f["estado"]] for f in x["prov"]]
    filas_cbval = [[f["chequeo"], f["esperado"], f["observado"][:60], f["estado"]] for f in x["cbval"]]
    filas_samp = [[f["chequeo"], f["valor"], f["nota"]] for f in x["samp"]]
    filas_cprof = [[f["metrica"], f["valor"][:100]] for f in x["cprof"]]
    filas_marg = [[f["clave"], f["n"], f["proporcion"]] for f in x["labels"] if f["bloque"] == "marginal"]
    filas_combo = [[f["clave"], f["n"], f["proporcion"]] for f in x["labels"] if f["bloque"] == "combinacion"]
    filas_combo += [["HS=0 TR=1 AG=0", "0", "0.000000"], ["HS=0 TR=1 AG=1", "0", "0.000000"]]
    filas_deps = [[f["clave"], f["n"], f["denominador"], f["proporcion"]] for f in x["labels"] if f["bloque"] == "dependencia"]
    filas_reg = [[f["slice"], f["familia"], f["definicion"][:70], f["n_rows_dev"], f["leakage_de_construccion"][:52]] for f in x["reg"]]
    filas_slice_f1 = []
    for f in x["reg"]:
        if f["familia"] == "SLICE_ERROR_CONDITIONED":
            continue
        filas_slice_f1.append([
            f["slice"], f["familia"].replace("SLICE_", ""), f["n_rows_dev"],
            x["f1_in"].get((f["slice"], "HS"), ""), x["f1_in"].get((f["slice"], "TR"), ""), x["f1_in"].get((f["slice"], "AG"), ""),
            x["f1_out"].get((f["slice"], "HS"), ""), x["f1_out"].get((f["slice"], "TR"), ""), x["f1_out"].get((f["slice"], "AG"), ""),
        ])
    filas_slice_err = []
    for f in x["reg"]:
        if f["familia"] != "SLICE_ERROR_CONDITIONED":
            continue
        filas_slice_err.append([
            f["slice"], f["n_rows_dev"],
            x["err_cells"].get((f["slice"], "HS"), ("", "", ""))[0],
            x["err_cells"].get((f["slice"], "TR"), ("", "", ""))[0],
            x["err_cells"].get((f["slice"], "AG"), ("", "", ""))[0],
            x["err_comp"].get(f["slice"], ""),
        ])
    top_over = sorted(x["sover"], key=lambda f: -float(f["jaccard"]))[:6]
    filas_over = [[f["slice_a"], f["slice_b"], f["interseccion"], f["jaccard"]] for f in top_over]
    filas_diag = [[et, e["TP"], e["FP"], e["FN"], e["TN"], e["precision"], e["recall"], e["F1"], e["errores"],
                   e["cerca"], e["medio"], e["lejos"]] for et, e in x["diag_etq"].items()]
    filas_combos = [[c, n] for c, n in x["combos_diag"]]
    filas_triple = [[c, n] for c, n in x["triple_dist"]]
    filas_inter = [[f["interseccion"], f["n"], f["de"], f["nota"][:40]] for f in x["inter"]]
    filas_hip = [[c, v[:120]] for c, v in x["hipotesis"]]
    filas_casos = [[f["tweet_id"], f"f{f['fold']}", f["error_pattern"], f.get("distancias", "")[:70]] for f in x["casos"]]
    filas_fam = [[f["family"], f["status"], f["best_result"][:58], f["closure_type_A_to_F"][:18]] for f in x["familias"]]
    filas_cand = [[f["candidate"][:48], f["honest_CV"][:12], f["LB"], f["verified"][:40]] for f in x["cand"]]
    no_unit = x["grupos_no_unit"]
    grupos_ids = sorted({g for _, g, _ in no_unit})
    estados = x["estados"]
    no_closed = [(f["estado"].split(":")[0].strip(), f["linea"][:86]) for f in x["hyp"]
                 if f["estado"].split(":")[0].strip() != "CLOSED"]
    filas_nc = [[e, l] for e, l in no_closed]

    secciones = []
    secciones.append(f"""# P90-02 - Auditoria integral de V03, corpus, EDA y trazabilidad experimental

Generado por `experiments/p90_02_auditoria_integral.py` y `experiments/p90_02b_informe_y_handoff.py`.
Commit auditado: `{d['git_commit']}`. Fecha: 2026-10-10. Sin entrenamiento, sin test, sin holdout,
sin submissions y sin commits.

## 1. Resumen ejecutivo

La auditoria reproduce integramente la evidencia de V03 y del failure analysis P90-00 y la
clasifica. Resultados principales: (1) la cadena del texto raw -> casebook -> entrada de V03 no
tiene perdida: {texto['diferencias_casebook']} diferencias en {cb['tweets_con_error']} tweets y
{texto['diferencias_celdas']} en {cb['celdas_error']} celdas; las formas sin diacriticos
("Espaa", "rabes") ya estan en la fuente autorizada (clasificacion `{texto['clasificacion']}`,
documentada en D-010). (2) `corrected_cv` = macro pooled = {met['corrected_cv_reproducido'][:18]}
reproducido a 1e-16; los F1 por etiqueta reportados son los pooled y coinciden; la media entre
folds es un nivel de agregacion distinto y no la metrica canonica. (3) El casebook queda
VERIFIED: {cb['tweets_con_error']} tweets / {cb['celdas_error']} celdas alineados 1:1 con el OOF,
controles {cb['controles']}/{cb['controles']} reproducidos con 0 incumplimientos. (4) Las 19
slices se reclasifican en 3 familias (texto / lexico inducido in-sample / subconjuntos definidos
por errores); sus definiciones se reproducen exactamente y los subconjuntos condicionados por
error dejan de reportar F1 cero-por-construccion. (5) El corpus de desarrollo ({corp['n']} filas)
queda perfilado por completo. (6) El EDA historico se consolida en {len(x['hist'])} lineas con
estado ({estados.get('CLOSED', 0)} CLOSED, {estados.get('BLOCKED_BY_RULES', 0)} BLOCKED_BY_RULES,
{estados.get('DUPLICATE', 0)} DUPLICATE, {estados.get('INVALIDATED', 0)} INVALIDATED,
{estados.get('COMPLEMENTARY', 0)} COMPLEMENTARY). Recomendacion: **A. CLOSE_FAILURE_ANALYSIS**,
con los asuntos de integridad (opcion B) resueltos en esta auditoria.

## 2. Estado de Git

- Rama `main`; HEAD `{d['git_commit']}` (P90-00); arbol limpio al iniciar; 8 commits por
  delante de `origin/main`; sin push.
- Commits P89 (`a07e080`) y P90-00 (`e43c594`) presentes en la genealogia.
- Esta tarea no realiza commits: los entregables quedan como archivos nuevos (y el notebook
  modificado) sin registrar en git.

## 3. Baseline

V03 congelado: TF-IDF word (1,2) + char_wb (3,5), min_df=2, sublinear_tf, lowercase;
LogisticRegression C=2.0, class_weight=balanced, solver=liblinear, random_state=42; umbrales
HS 0.44 / TR 0.51 / AG 0.46; `corrected_cv` = {met['corrected_cv_reproducido'][:18]};
leaderboard historico 0.78306. Hashes: OOF `{d['hashes']['oof'][:16]}...`, submission
`{d['hashes']['submission'][:16]}...`, modelo `{d['hashes']['modelo'][:16]}...`, train
`{d['hashes']['train'][:16]}...`, particion `{d['hashes']['particion'][:16]}...`.
Procedencia del OOF: {d['fuente_oof']}.

## 4. Alcance autorizado

Corpus de desarrollo autorizado: las {corp['n']} filas con prediccion OOF (ids del artefacto
canonico), del archivo `data/raw/toxic_train.csv` (1731 filas; las 248 restantes son la
particion reservada, no leida). Exclusiones: test (no tocado), holdout (agotado, no tocado),
holdout2 (congelado, no tocado). El artefacto `oof_mod03b` (que cubre el train completo con
filas de la particion reservada) NO se abrio: la tabla historica del train-1731 se cita desde
`reports/error_analysis.md` sin recalcular. Limitacion: el escaneo de bytes no-ASCII de todo el
archivo se hizo como verificacion de integridad; el analisis de contenido se limito a las
{corp['n']} filas de desarrollo.

## 5. Fuentes utilizadas

{tabla(["artifact_id", "path", "sha256[:16]", "tracked", "status"], filas_inv)}

## 6. Auditoria de integridad textual

Cadena verificada: texto de origen (`data/raw/toxic_train.csv`) -> texto recuperado para el
casebook (`v03_error_casebook.csv` / `v03_error_cells.csv`) -> entrada real de V03 (texto
original + `lowercase` del vectorizador; `strip_accents=None` en los scripts de entrenamiento).

{tabla(["chequeo", "resultado", "estado"], filas_prov)}

Clasificacion: **{texto['clasificacion']}**. Evidencia: las formas sin diacriticos aparecen en
la fuente autorizada ({"Espaa"}: {texto['formas'].get('Espaa', 0)}, {"rabes"}: {texto['formas'].get('rabes', 0)},
{"inmigracin"}: {texto['formas'].get('inmigracin', 0)}, {"agresin"}: {texto['formas'].get('agresin', 0)},
{"estacin"}: {texto['formas'].get('estacin', 0)}, {"polica"}: {texto['formas'].get('polica', 0)} en
el desarrollo); {texto['ascii_dev']} de {corp['n']} textos tienen bytes > 127; D-010 documenta
que "los organizadores normalizaron a ASCII". No hay operacion de retirada de diacriticos en la
ruta de V03. Conclusion: no procede corregir ni reentrenar; la ausencia de diacriticos es una
propiedad del dato y una limitacion documentada.

## 7. Conciliacion de metricas

Matrices pooled (todas las predicciones OOF agregadas, umbrales congelados):

{tabla(["etiqueta", "TP", "FP", "FN", "TN", "F1 pooled"], filas_mat)}

Precision / recall / F1 (precision completa en el CSV):

{tabla(["etiqueta", "precision", "recall", "F1"], filas_pr)}

F1 por fold (n = tamano del fold):

{tabla(["fold", "HS", "TR", "AG"], filas_fold)}

Niveles de agregacion (no intercambiables):

{tabla(["metrica", "valor", "definicion"], filas_agreg)}

**ORIGINAL / RECOMPUTED / EXPLANATION** de la discrepancia percibida: ORIGINAL: las cifras
citadas como matrices globales (TP 614/431/517) provienen de la tabla historica del TRAIN
completo de 1731 filas con MOD-03b en `reports/error_analysis.md` (umbral optimo), una
poblacion distinta. RECOMPUTED: sobre el desarrollo de {corp['n']} filas las matrices pooled son
505/173/143/662, 363/82/82/956 y 425/135/142/781; sus F1 coinciden exactamente con los F1 por
etiqueta reportados (diferencias <= 1.1e-16). EXPLANATION: no hay discrepancia de datos sino
mezcla de poblaciones y de niveles de agregacion; `corrected_cv` es el macro pooled y la media
entre folds ({met['media_macro_pliegues']}) es otra cantidad. {met['explicacion']}

Por que el promedio de F1 entre folds difiere del F1 pooled: F1 es una razon no lineal; el
promedio de razones (con denominadores por fold) no es la razon de las sumas (denominador
agregado), y los folds tienen tamanos y prevalencias distintas (ver tabla). Criterio de cierre:
filas, ids, umbrales y predicciones coinciden; matrices y F1 por fold se reproducen;
`corrected_cv` se obtiene con su definicion; discrepancias abiertas: {met['discrepancias_abiertas']}.

## 8. Validacion del casebook

{tabla(["chequeo", "esperado", "observado", "estado"], filas_cbval)}

Recuentos verificados: {cb['tweets_con_error']} tweets con al menos un error ({cb['una_etiqueta']}
con una etiqueta, {cb['dos_etiquetas']} con dos, {cb['tres_etiquetas']} con las tres) y
{cb['tweets_sin_error']} tweets sin error; {cb['celdas_error']} celdas tweet-etiqueta. Veredicto:
**VERIFIED** (alineacion 1:1 con el OOF canonico, sin ids duplicados ni fuera de alcance).

## 9. Validacion de la muestra y de los controles

{tabla(["chequeo", "valor", "nota"], filas_samp)}

Los controles se reprodujeron con el procedimiento determinista completo: para cada celda de
error, el candidato correcto con misma etiqueta+fold+y_true y vecino mas cercano por
|delta distancia| (desempate por id). Resultado: {cb['controles']}/{cb['controles']} reproducidos,
0 incumplimientos, 0 celdas sin control; {rep_ctl} reutilizaciones de control
(un mismo tweet correcto puede emparejarse con varios errores). La muestra (100 errores + 185
controles) es estratificada, NO aleatoria: no admite estimaciones de prevalencia poblacional;
los controles 1:1 comparten etiqueta, fold y valor verdadero con su error (no son
independientes). La columna `matching_criteria` registra el |delta_distancia| efectivo, que
puede ser grande: el criterio real es "vecino mas cercano", no un umbral fijo.

## 10. Taxonomia de slices

{tabla(["slice", "familia", "definicion", "n_rows", "limitacion"], filas_reg)}

Reclasificacion frente al CSV historico (que etiquetaba como "rebanada_de_texto" tambien las de
lexico inducido): {sum(1 for f in x['reg'] if f['familia'] == 'SLICE_TEXT_DEFINED')} slices de
texto, {sum(1 for f in x['reg'] if f['familia'] == 'SLICE_LEXICON_INDUCED')} de lexico inducido
in-sample (solo diagnostico: el detector uso las etiquetas de las mismas observaciones) y
{sum(1 for f in x['reg'] if f['familia'] == 'SLICE_ERROR_CONDITIONED')} subconjuntos definidos por
errores (taxonomia y recuentos; su F1 interno es cero por construccion y no se reporta).

## 11. EDA general del corpus autorizado

{tabla(["metrica", "valor"], filas_cprof)}

## 12. Estructura de las labels

Marginales:

{tabla(["etiqueta", "positivos", "proporcion"], filas_marg)}

Ocho combinaciones (las dos no observadas se listan con 0):

{tabla(["combinacion", "n", "proporcion"], filas_combo)}

Dependencias (con denominadores):

{tabla(["probabilidad", "numerador", "denominador", "valor"], filas_deps)}

Identidad estructural: TR=1 con HS=0 tiene **0 casos** en el desarrollo (445/445 TR positivos
tienen HS=1). Esta identidad esta documentada historicamente (D-001: "TR=1 solo ocurre cuando
HS=1"; D-044/D-045: "TR=1 implica HS=1 de forma determinista") y su explotacion modelistica
(factorizacion p(TR|X)=p(HS|X)*p(TR|HS=1,X), familia joint_labels) fue explorada y CERRADA.
AG=1 con HS=0 ocurre en 22 casos (2.6 por ciento condicional; 1.5 por ciento historico en el
train completo).

## 13. Calidad de los datos

Sin textos vacios, sin ids duplicados, sin duplicados exactos de texto, sin caracteres no ASCII
en el desarrollo. Longitudes: {corp['chars']['min']}-{corp['chars']['max']} caracteres (media
{corp['chars']['media']}, mediana {corp['chars']['mediana']:.0f}); {corp['tokens']['min']}-{corp['tokens']['max']}
tokens (media {corp['tokens']['media']}). Vocabulario: {corp['vocab_tipos']} tipos,
{corp['vocab_tokens']} tokens. Limitaciones: la particion reservada no se inspecciono; el
analisis describe solo el desarrollo.

## 14. Perfil textual

Longitud y estructura (definiciones en el CSV): tweets cortos (<25 tokens) {corp['n'] - 301} vs
largos {301}; URLs 458, hashtags 144, menciones 702 (605 iniciales), elongaciones 76, mayusculas
sostenidas 864, puntuacion repetida 84. Tokens mas frecuentes del texto original: de, que, la,
y, a, en, **puta**, el, no, un, es, los, te, una, **perra**, lo, me, se, **eres**, tu (marcador
lexico de misoginia en posiciones altas del ranking; descriptivo, no causal). El lexico de
insultos del proyecto (inducido in-sample, z>=2.5) cubre 1114 de {corp['n']} textos (901 con
nucleo).

## 15. Duplicados y grupos

Grupos near-dup historicos (`data/processed/grupos_near_duplicados.csv`, metadatos): 1479 grupos
distintos tocan el desarrollo; solo {corp['no_unitarios']} filas (0.4 por ciento) pertenecen a
grupos no unitarios: los grupos {grupos_ids[0] if grupos_ids else ''} y {grupos_ids[1] if len(grupos_ids) > 1 else ''}
(3 filas dev cada uno; tamanos declarados 5 y 4, con el resto de sus filas fuera del desarrollo).
La dependencia por near-duplicados en desarrollo es practicamente nula; el mecanismo de grupos
se uso para el resampling del bootstrap historico y no se modifico. No se afirma que el ruido
de etiquetado explique los errores: no hay evidencia especifica.

## 16. Sintesis de errores por etiqueta

{tabla(["etiqueta", "TP", "FP", "FN", "TN", "precision", "recall", "F1", "errores", "cerca<0.025", "medio", "lejos>=0.10"], filas_diag)}

## 17. Errores conjuntos

{tabla(["patron", "n"], filas_combos)}

Intersecciones por pares de etiquetas y errores aislados:

{tabla(["interseccion", "n", "de", "nota"], filas_inter)}

Errores de las tres etiquetas por estrato de distancia (98 tweets):

{tabla(["estrato", "n"], filas_triple)}

Los errores triples son mayoritariamente lejanos al umbral (67-76 de 98 en HS/TR): no son un
problema de frontera. No se agrupan bajo una unica causa.

Hipotesis de FP/FN evaluadas (recuento, nivel de evidencia y contraejemplo declarado):

{tabla(["hipotesis", "evaluacion"], filas_hip)}

## 18. Distancias a los thresholds

Frontera (<0.025): {diag['fronterizos']} de {diag['errores_celda']} celdas
({100.0 * diag['fronterizos'] / diag['errores_celda']:.1f} por ciento). Lejanos (>=0.10):
{diag['lejanos']} de {diag['errores_celda']} ({100.0 * diag['lejanos'] / diag['errores_celda']:.1f}
por ciento). La distancia al umbral NO es una probabilidad calibrada ("confianza real"); el
resultado limita el potencial de una solucion basada exclusivamente en reajustar thresholds
(familia threshold_transfer CLOSED, P55/P56).

## 19. Resultados por slice

Slices de texto y de lexico inducido (F1 dentro / fuera; lexico in-sample = diagnostico):

{tabla(["slice", "familia", "n", "F1 HS in", "F1 TR in", "F1 AG in", "F1 HS out", "F1 TR out", "F1 AG out"], filas_slice_f1)}

Subconjuntos definidos por errores (recuentos; F1 omitido por construccion):

{tabla(["slice", "n", "celdas HS", "celdas TR", "celdas AG", "tweets 1/2/3 etiquetas"], filas_slice_err)}

Solapamiento: {len(x['sover'])} pares entre las 14 slices no condicionadas por error; pares con
mayor interseccion:

{tabla(["slice_a", "slice_b", "interseccion", "jaccard"], filas_over)}

No se suman errores entre slices (no son excluyentes); "URLs" (458 filas, descriptiva) y "news"
(156 filas, definida por errores) describen poblaciones distintas y no se comparan como si
fueran la misma.

## 20. Sintesis del EDA historico

Matriz consolidada (48 lineas; CSV completo en `p90_02_historical_eda_matrix.csv`). Familias del
registro historico P57:

{tabla(["family", "status", "best_result", "cierre"], filas_fam)}

## 21. Resultados negativos

Verificados contra `p57_family_status.csv`, `p57_candidate_matrix.csv` y
`reports/toxic_remaining_legal_opportunities.csv`:

- FastText aprende senal, pero no supera el gate de adopcion (familia embeddings CLOSED).
- Word2Vec-control queda por debajo de FastText (documentado en la auditoria de oportunidades).
- Mayor cobertura no garantiza mejor F1 (familia data_expansion CLOSED).
- El corpus mixto no aporta una mejora suficiente.
- El stacking generico falla repetidamente (S08; v08_meta_v2 LB -0.00143).
- La propagacion por near-duplicate no resuelve el problema (familia retrieval CLOSED).
- BCE agregada no constituye un proxy adecuado de dureza.
- La distancia especifica por etiqueta resulta mas util para diagnosticar dificultad.
- Una pequena mejora de CV no garantiza mejor transferencia al leaderboard: cuatro candidatos
  enviados con delta CV positivo cayeron en LB (v23_01 -0.00368; v08_meta_v2 -0.00143;
  v20_01 -0.00643; y el caso de umbrales anidados con delta honesto negativo).

Candidatos historicos con veredicto externo:

{tabla(["candidato", "CV", "LB", "estado"], filas_cand)}

## 22. Lineas cerradas y duplicadas

{len(x['hyp'])} lineas clasificadas: {estados.get('CLOSED', 0)} CLOSED,
{estados.get('BLOCKED_BY_RULES', 0)} BLOCKED_BY_RULES, {estados.get('DUPLICATE', 0)} DUPLICATE,
{estados.get('INVALIDATED', 0)} INVALIDATED, {estados.get('COMPLEMENTARY', 0)} COMPLEMENTARY.
Lineas no cerradas:

{tabla(["estado", "linea"], filas_nc)}

No se reabre ninguna familia por interes descriptivo de los errores; las variantes de
hipotesis ya probadas se consideran DUPLICATE y no novedosas.

## 23. Posibles nuevas hipotesis

Dentro de las reglas vigentes no se identifica una hipotesis de modelado genuinamente nueva:
las familias del espacio TF-IDF/scores/modelos clasicos estan CLOSED o DUPLICATE, las lineas de
anotacion y datos estan BLOCKED_BY_RULES, y las mejoras locales de CV no transfieren (patron
repetido). Las unicas cuestiones abiertas son de revision humana de anotacion (bloqueada) y de
comprension de los errores lejanos (requiere autorizacion y mecanismo legal no medido). No se
propone ejecutar nada en esta fase.

## 24. Limitaciones metodologicas

- Los controles 1:1 no son independientes ni una muestra aleatoria (lectura cualitativa).
- El lexico inducido es in-sample: las slices de insulto no tienen independencia OOF.
- Los subconjuntos definidos por errores no estiman rendimiento de categorias.
- No hay anotacion humana independiente: ninguna hipotesis de anotacion se demuestra.
- La distancia al umbral no es una probabilidad calibrada.
- El artefacto del train completo con MOD-03b no se abrio (mezcla filas reservadas).
- Los juicios de registro (broma, sarcasmo, cita) son cualitativos y con contraejemplos.
- La reproduccion cubre un unico OOF de 5 folds; no se re-entreno nada (por regla).

## 25. Decisiones finales

- **Decision 1 - V03**: FROZEN. Sin modificaciones.
- **Decision 2 - Calidad del failure analysis**: casebook = VERIFIED; slice_analysis =
  PARTIALLY_VERIFIED (definiciones y cifras reproducidas exactamente; taxonomia de familias
  corregida y documentada; interpretacion de subconjuntos restringida a recuentos);
  metric_reconciliation = VERIFIED; text_provenance = VERIFIED.
- **Decision 3 - Corpus**: analizado el desarrollo autorizado ({corp['n']} filas); fuera: test,
  holdout y holdout2 (intactos). Limitaciones declaradas en las secciones 4 y 24.
- **Decision 4 - Interpretacion de errores**: CONFIRMED: identidad TR dentro de HS; distancias
  (84 frontera / 466 lejanos); alineacion del casebook. SUPPORTED: broma/afiliacion en FP
  (27/173 FP_HS, 26/135 FP_AG); errores de TR en textos sin destinatario; error lejano como
  region dominante. TENTATIVE: FN de AG con referencia colectiva superficial; cita/meta-discurso
  minoritario; limitacion de representacion/contexto. UNSUPPORTED: ambiguedad de anotacion como
  causa; sarcasmo como causa de FN; "el modelo carece de la informacion" sin experimento.
- **Decision 5 - Estado de la investigacion**: {estados.get('CLOSED', 0)} CLOSED,
  {estados.get('BLOCKED_BY_RULES', 0)} BLOCKED_BY_RULES, {estados.get('DUPLICATE', 0)} DUPLICATE,
  {estados.get('INVALIDATED', 0)} INVALIDATED, {estados.get('COMPLEMENTARY', 0)} COMPLEMENTARY;
  ninguna POTENTIALLY_NEW demostrada.
- **Decision 6 - Submissions**: new submissions = 0.
- **Decision 7 - Baseline y holdouts**: V03 FROZEN; holdout FROZEN; holdout2 FROZEN; test
  UNTOUCHED.
- **Decision 8 - Integridad**: metricas reproducidas SI; predicciones alineadas SI; texto
  trazado SI; slices correctamente interpretadas SI (con taxonomia corregida); corpus
  caracterizado SI; EDA historico consolidado SI; notebook actualizado de forma segura SI
  (seccion 136 con celda de verificacion ejecutada en aislamiento).
- **Decision 9 - Recomendacion cientifica**: **A. CLOSE_FAILURE_ANALYSIS** (principal). La
  opcion B (integridad) queda resuelta por esta auditoria; C no es necesaria; D no procede
  porque no hay hipotesis nueva demostrable.

## 26. Preguntas no resueltas

- Mecanismo de los 466 errores lejanos (>=0.10): sin causa demostrada; requiere revision
  humana o un experimento autorizado (fuera de alcance).
- Impacto cuantificable de la ausencia de diacriticos y emojis: no medido (requiere
  experimento; no se afirma).
- Regla de anotacion formal detras de TR=1 => HS=1: la identidad es empirica y esta documentada,
  pero no se encontro el texto normativo de los organizadores en el repositorio.
- Calibracion de probabilidades: la distancia al umbral no es confianza; una calibracion no
  se ha medido como mejora legal.
""")

    return "\n".join(secciones)


def construir_handoff(x: dict[str, Any]) -> str:
    """Construye el handoff autocontenido para ChatGPT."""
    d, met, cb, corp, diag = x["d"], x["met"], x["cb"], x["corp"], x["diag"]
    texto = x["texto"]
    mp = met["matrices_pooled"]
    estados = x["estados"]
    filas_mat = [[et, mp[et]["TP"], mp[et]["FP"], mp[et]["FN"], mp[et]["TN"],
                  f"{met['f1_por_etiqueta_pooled'][et]:.6f}"] for et in ("HS", "TR", "AG")]
    filas_fold = [[f"f{k} (n={x['fold_n'].get(k, '?')})"] + [f"{x['fold_f1'][(et, k)]:.6f}" for et in ("HS", "TR", "AG")]
                  for k in range(1, 6)]
    filas_combo = [[f["clave"], f["n"]] for f in x["labels"] if f["bloque"] == "combinacion"]
    filas_deps = [[f["clave"], f["proporcion"], f["denominador"]] for f in x["labels"] if f["bloque"] == "dependencia"]
    filas_diag = [[et, e["TP"], e["FP"], e["FN"], e["TN"], e["precision"], e["recall"], e["F1"],
                   e["cerca"], e["medio"], e["lejos"]] for et, e in x["diag_etq"].items()]
    filas_combos = [[c, n] for c, n in x["combos_diag"]]
    filas_slice = []
    for f in x["reg"]:
        if f["familia"] == "SLICE_ERROR_CONDITIONED":
            continue
        filas_slice.append([f["slice"], f["familia"].replace("SLICE_", ""), f["n_rows_dev"],
                            x["f1_in"].get((f["slice"], "HS"), ""), x["f1_in"].get((f["slice"], "TR"), ""),
                            x["f1_in"].get((f["slice"], "AG"), ""), x["f1_out"].get((f["slice"], "HS"), ""),
                            x["f1_out"].get((f["slice"], "TR"), ""), x["f1_out"].get((f["slice"], "AG"), "")])
    filas_err = [[f["slice"], f["n_rows_dev"],
                  x["err_cells"].get((f["slice"], "HS"), ("", "", ""))[0],
                  x["err_cells"].get((f["slice"], "TR"), ("", "", ""))[0],
                  x["err_cells"].get((f["slice"], "AG"), ("", "", ""))[0],
                  x["err_comp"].get(f["slice"], "")] for f in x["reg"] if f["familia"] == "SLICE_ERROR_CONDITIONED"]
    filas_fam = [[f["family"], f["status"], f["best_result"][:50]] for f in x["familias"]]
    filas_casos = [[f["tweet_id"], f"f{f['fold']}", f["error_pattern"], f.get("distancias", "")[:56], f.get("text", "")[:64]] for f in x["casos"]]
    filas_cand = [[f["candidate"][:44], f["honest_CV"][:10], f["LB"], f["verified"][:34]] for f in x["cand"]]
    no_closed = [(f["estado"].split(":")[0].strip(), f["linea"][:80]) for f in x["hyp"]
                 if f["estado"].split(":")[0].strip() != "CLOSED"]

    return f"""# P90-02 - Handoff autocontenido para segunda revision independiente (ChatGPT)

Commit auditado: `{d['git_commit']}`. Fecha: 2026-10-10. Auditoria descriptiva: 0
entrenamientos, 0 submissions, 0 commits, test y holdouts intactos. Este documento contiene los
hallazgos completos; las tablas viven en `reports/p90_02_*.csv`.

## A. Estado del proyecto

- Baseline V03 FROZEN: TF-IDF word(1,2)+char_wb(3,5), LR C=2.0 balanced liblinear; umbrales
  0.44/0.51/0.46; `corrected_cv` = {met['corrected_cv_reproducido'][:18]}; LB historico 0.78306.
- Git: rama `main`, HEAD `{d['git_commit']}` (P90-00), arbol limpio al inicio, 8 commits sin
  push; esta auditoria no commitea.
- Test UNTOUCHED; holdout y holdout2 FROZEN; sin uso del leaderboard.

## B. Calidad de la medicion

- Integridad textual: {texto['clasificacion']}; 0 diferencias raw/casebook/celdas; 0 textos con
  bytes > 127 en el desarrollo (D-010: normalizacion ASCII de los organizadores).
- Conciliacion: `corrected_cv` = macro pooled = {met['corrected_cv_reproducido'][:18]}
  (reproducido a 1e-16); F1 por etiqueta = pooled y coinciden con lo reportado; media entre
  folds = {met['media_macro_pliegues']} (nivel distinto, NO es la metrica canonica). Las cifras
  614/431/517 citadas son del train-1731 con MOD-03b (error_analysis.md), otra poblacion.
- Discrepancias abiertas: {met['discrepancias_abiertas']}.

{tabla(["etiqueta", "TP", "FP", "FN", "TN", "F1 pooled"], filas_mat)}

{tabla(["fold", "HS", "TR", "AG"], filas_fold)}

## C. Perfil completo del corpus autorizado

- {corp['n']} filas (ids unicos), sin vacios, sin duplicados exactos, sin ids duplicados.
- Longitud: {corp['chars']['min']}-{corp['chars']['max']} caracteres (media {corp['chars']['media']},
  mediana {corp['chars']['mediana']:.0f}); {corp['tokens']['min']}-{corp['tokens']['max']} tokens
  (media {corp['tokens']['media']}); cortos (<25 tokens) {corp['n'] - 301}, largos {301}.
- Vocabulario: {corp['vocab_tipos']} tipos / {corp['vocab_tokens']} tokens; top: de, que, la, y,
  a, en, puta, el, no, un, es, los, te, una, perra, lo, me, se, eres, tu.
- Estructura: URLs 458; hashtags 144; menciones 702 (605 iniciales); elongaciones 76; mayusculas
  sostenidas 864; puntuacion repetida 84; no-ASCII 0.
- Labels (positivos): HS 648, TR 445, AG 567.

{tabla(["combinacion", "n"], filas_combo)}

{tabla(["dependencia", "valor", "denominador"], filas_deps)}

- Identidad: TR=1 => HS=1 determinista (0 violaciones; documentada en D-001/D-044); AG=1 con
  HS=0: 22 casos.
- Duplicados/grupos: 1479 grupos near-dup en desarrollo; solo 6 filas (0.4 por ciento) en 2
  grupos no unitarios (112 y 338). Sin evidencia de que el ruido de etiquetado explique errores.
- Limitaciones: desarrollo unicamente; sin anotacion humana independiente; categorias de
  registro cualitativas sin prevalencia inventada.

## D. Diagnostico de V03

{tabla(["etiqueta", "TP", "FP", "FN", "TN", "P", "R", "F1", "cerca", "medio", "lejos"], filas_diag)}

- {cb['tweets_con_error']} tweets con error / {cb['celdas_error']} celdas ({cb['una_etiqueta']} /
  {cb['dos_etiquetas']} / {cb['tres_etiquetas']} con 1/2/3 etiquetas); {cb['tweets_sin_error']}
  tweets sin error.
- Distancias: {diag['fronterizos']}/{diag['errores_celda']} frontera (<0.025);
  {diag['lejanos']}/{diag['errores_celda']} lejanos (>=0.10). Triples: 67-76 de 98 lejanos.
- Patrones cuantificados: broma/afiliacion 27/173 FP_HS y 26/135 FP_AG (SUPPORTED);
  FN de AG con referencia colectiva 16/142 (TENTATIVE, lista ilustrativa); cita 5/143
  (TENTATIVE); errores de TR en textos sin destinatario (region observada).
- Contraejemplos: FP con p>=0.9 sin marcadores de broma (p. ej. 21892); controles correctos con
  @ y lenguaje fuerte (20609, 20347); la distancia no es confianza.

{tabla(["patron de error conjunto", "n"], filas_combos)}

Casos representativos verificados (13): tabla en `p90_02_representative_cases.csv`; muestra:

{tabla(["tweet_id", "fold", "patron", "distancias", "texto (truncado)"], filas_casos)}

## E. Slices

Slices de texto y de lexico inducido (F1 dentro/fuera; las de lexico son in-sample y solo
diagnosticas):

{tabla(["slice", "familia", "n", "HS in", "TR in", "AG in", "HS out", "TR out", "AG out"], filas_slice)}

Subconjuntos definidos por errores (recuentos; F1 omitido: cero por construccion):

{tabla(["slice", "n", "celdas HS", "celdas TR", "celdas AG", "tweets 1/2/3"], filas_err)}

Solapamiento: 91 pares; no se suman errores entre slices.

## F. EDA historico

48 lineas clasificadas ({estados.get('CLOSED', 0)} CLOSED, {estados.get('BLOCKED_BY_RULES', 0)}
BLOCKED_BY_RULES, {estados.get('DUPLICATE', 0)} DUPLICATE, {estados.get('INVALIDATED', 0)}
INVALIDATED, {estados.get('COMPLEMENTARY', 0)} COMPLEMENTARY). Familias P57:

{tabla(["family", "status", "best_result"], filas_fam)}

Lineas no cerradas: {len(no_closed)} (revision de anotacion TR, propagacion near-dup,
correccion EM de prevalencia, expansion de datos, transformers/LLM: BLOCKED_BY_RULES; FM y
LightGBM sobre la representacion de V03: DUPLICATE).

Resultados negativos clave (verificados en los informes): FastText no supera el gate; Word2Vec
por debajo; mas cobertura no mejora F1; corpus mixto no aporta; stacking generico falla; la
propagacion near-dup no resuelve; BCE agregada no es proxy de dureza; la distancia por etiqueta
diagnostica mejor; CV no garantiza LB (4 candidatos enviados con delta CV positivo cayeron:
v23_01 -0.00368, v08_meta_v2 -0.00143, v20_01 -0.00643).

{tabla(["candidato", "CV", "LB", "estado"], filas_cand)}

Complementariedad (diagnostico, citado de `reports/p66_complementarity.md`): la correccion
residual de MI_arbol rescata 13/16/24 celdas HS/TR/AG con dano 10/16/27 (neto +3/+0/-3);
rescate de alta confianza total: 6 (0/4/2); bootstrap 2000 remuestreos: delta medio +0.0027270,
IC95 [-0.0044312, +0.0102109] incluye cero; brecha vs barajado +0.0041134; veredicto historico
PROMISING, no adoptado. Ninguna solucion complementaria transferible demostrada.

## G. Integridad y reproducibilidad

- Artefactos fuente y hashes: ver seccion 5 del informe (`p90_02_artifact_inventory.csv`).
- Texto: verificacion byte a byte (3 representaciones, 0 diferencias); vectorizadores con
  strip_accents=None; entrada real = texto original + lowercase.
- Numerico: matrices pooled, F1 por fold, media de folds, macro pooled; todo reproducido;
  controles 757/757; casebook 441/757 alineado; muestra 100+185.
- Tests ejecutados: hashes de 5 artefactos; pred == proba >= umbrales; particion dev == ids
  OOF; validacion de casebook/celdas/controles/muestra; 19/19 definiciones de slices
  reproducidas; checks ASCII (0 bytes > 127) y AST de operaciones y ficheros de los scripts.
- Limitaciones de acceso: el artefacto del train completo con MOD-03b no se abrio (mezcla
  filas reservadas); el contenido del test/holdout no se inspecciono.

## H. Interpretacion cientifica

- Demostrado: no hay perdida de texto en el pipeline (propiedad del dato); `corrected_cv` es el
  macro pooled; el casebook y sus controles son fieles al OOF; la identidad TR=>HS; la mayoria
  del error esta lejos de los umbrales (466/757).
- Respaldo parcial: broma/afiliacion como subconjunto de los FP (15.6 / 19.3 por ciento, no la
  mayoria); region TR sin destinatario; limitacion de representacion/contexto (consistente con
  los errores lejanos, sin test causal).
- Hipotesis: FN de AG con referencia colectiva superficial (lista ilustrativa); cita/meta-discurso
  minoritario; calibracion como mejora potencial no medida.
- No afirmable: ambiguedad de anotacion como causa; sarcasmo como causa de FN; que el modelo
  carezca de la informacion necesaria; prevalencias poblacionales a partir de la muestra.

## I. Decisiones finales

V03 FROZEN; casebook VERIFIED; slices PARTIALLY_VERIFIED (taxonomia corregida); metricas
VERIFIED; texto VERIFIED; submissions 0; test/holdouts intactos; recomendacion **A.
CLOSE_FAILURE_ANALYSIS** con B resuelta. Preguntas pendientes: mecanismo de los 466 errores
lejanos; impacto de la normalizacion ASCII; regla de anotacion formal TR=>HS; calibracion como
mejora legal no medida.
"""


def construir_manifest(x: dict[str, Any]) -> dict[str, Any]:
    """Construye el manifiesto de reproducibilidad."""
    d = x["d"]
    salidas = [
        "reports/p90_02_integrity_and_corpus_audit.md", "reports/p90_02_chatgpt_handoff.md",
        "reports/p90_02_audit_manifest.json", "reports/p90_02_audit_results.json",
        "reports/p90_02_text_provenance.csv", "reports/p90_02_metric_reconciliation.csv",
        "reports/p90_02_casebook_validation.csv", "reports/p90_02_review_sample_audit.csv",
        "reports/p90_02_corpus_profile.csv", "reports/p90_02_label_distribution.csv",
        "reports/p90_02_slice_registry.csv", "reports/p90_02_slice_metrics.csv",
        "reports/p90_02_slice_overlap.csv", "reports/p90_02_error_diagnosis.csv",
        "reports/p90_02_error_intersections.csv", "reports/p90_02_representative_cases.csv",
        "reports/p90_02_historical_eda_matrix.csv", "reports/p90_02_hypothesis_status.csv",
        "reports/p90_02_artifact_inventory.csv",
        "experiments/p90_02_auditoria_integral.py", "experiments/p90_02b_informe_y_handoff.py",
        "experiments/p90_03_integrar_notebook.py", "notebooks/10_final_pipeline.ipynb",
    ]
    return {
        "audit_id": "P90-02",
        "git_commit": d["git_commit"],
        "branch": subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True,
                                 cwd=RAIZ).stdout.strip(),
        "source_artifacts": [f["path"] for f in x["inv"]],
        "source_hashes": {f["path"]: f["sha256"] for f in x["inv"]},
        "configurations": {
            "umbrales": {"HS": 0.44, "TR": 0.51, "AG": 0.46},
            "vectorizadores": "word(1,2) + char_wb(3,5), min_df=2, sublinear_tf, lowercase, strip_accents=None",
            "modelo": "LogisticRegression C=2.0 class_weight=balanced solver=liblinear",
        },
        "dataset_scope": "desarrollo autorizado: 1483 filas con prediccion OOF (ids canonicos)",
        "excluded_partitions": ["test (untouched)", "holdout (frozen)", "holdout2 (frozen)"],
        "computed_metrics": {
            "corrected_cv": x["met"]["corrected_cv_reproducido"],
            "f1_por_etiqueta_pooled": x["met"]["f1_por_etiqueta_pooled"],
            "matrices_pooled": x["met"]["matrices_pooled"],
            "media_entre_folds": x["met"]["media_folds_por_etiqueta"],
            "nivel_distinto_media_macro_pliegues": x["met"]["media_macro_pliegues"],
        },
        "output_artifacts": salidas,
        "validation_status": {
            "text_provenance": "SOURCE_ALREADY_NORMALIZED (0 diferencias)",
            "metric_reconciliation": "REPRODUCED (diferencias <= 1.1e-16)",
            "casebook": "VERIFIED (441/757; 0 desviaciones)",
            "controles": "REPRODUCIDOS 757/757 (0 incumplimientos)",
            "muestra": "100 errores + 185 controles (estratificada, no aleatoria)",
            "slices": "19/19 definiciones reproducidas; 3 familias documentadas",
            "ascii": "0 bytes > 127 en artefactos generados",
            "entrenamientos": 0, "submissions": 0, "commits": 0,
        },
        "known_discrepancies": [
            "RESUELTA: cifras 614/431/517 citadas = tabla historica train-1731 con MOD-03b (otra poblacion)",
            "RESUELTA: corrected_cv = macro pooled, no media de folds (niveles de agregacion distintos)",
            "RESUELTA: taxonomy de slices historica mezclaba lexico inducido y slices de texto (reclasificada)",
        ],
    }


def auditar_self() -> dict[str, Any]:
    """Comprobacion AST del propio script (operacion y fichero concretos)."""
    arbol = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    docstrings = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            cuerpo = getattr(nodo, "body", [])
            if cuerpo and isinstance(cuerpo[0], ast.Expr) and isinstance(cuerpo[0].value, ast.Constant):
                if isinstance(cuerpo[0].value.value, str):
                    docstrings.add(id(cuerpo[0].value))
    puro = re.compile(r"^[\w./\\-]+\.\w{2,5}$|^[\w./\\-]+/$")
    tokens = []
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Constant) and isinstance(nodo.value, str) and id(nodo) not in docstrings:
            v = nodo.value
            if puro.match(v) and any(p in v.lower() for p in
                                     ("toxic_test", "test.csv", "leaderboard", "holdout", "submission")):
                tokens.append(v)
    modulos = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Import):
            modulos |= {a.name.split(".")[0] for a in nodo.names}
        elif isinstance(nodo, ast.ImportFrom) and nodo.module:
            modulos.add(nodo.module.split(".")[0])
    return {"tokens": sorted(set(tokens)), "modulos": sorted(modulos)}


def main() -> int:
    """Genera informe, handoff y manifiesto; valida ASCII y compliance."""
    x = cargar_todo()
    informe = construir_informe(x)
    handoff = construir_handoff(x)
    manifiesto = construir_manifest(x)
    escribir_texto(RUTA_INFORME, informe + "\n")
    escribir_texto(RUTA_HANDOFF, handoff + "\n")
    escribir_texto(RUTA_MANIFEST, json.dumps(manifiesto, ensure_ascii=False, indent=1) + "\n")
    for ruta in (RUTA_INFORME, RUTA_HANDOFF, RUTA_MANIFEST):
        if any(b > 127 for b in ruta.read_bytes()):
            raise SystemExit(f"STOP: bytes > 127 en {ruta.name}")
    auditoria = auditar_self()
    permitidos = {
        "toxic_train.csv", "holdout_v2_ids.csv", "grupos_near_duplicados.csv",
        "pipeline_v03.json", "hashes.txt", "toxic_submission_v03_lr_union_thr.csv",
        "model_v03.joblib", "oof_v03_final.npz", "fase2_lexico_inducido_agresividad.csv",
        "p57_family_status.csv", "p57_candidate_matrix.csv", "p66_complementarity.md",
        "toxic_remaining_legal_opportunities.csv", "v03_error_casebook.csv", "v03_error_cells.csv",
        "v03_error_fold_summary.csv", "v03_error_slice_summary.csv", "v03_error_confidence_summary.csv",
        "v03_error_review_sample.csv", "v03_error_controls.csv", "v03_error_provenance.json",
        "experiments.csv", "DECISIONS.md", "10_final_pipeline.ipynb", "p90_02b_informe_y_handoff.py",
        "p90_02_auditoria_integral.py", "p90_03_integrar_notebook.py", "test.csv", "submission",
        "submissions", "leaderboard", "holdout",
    }
    raros = [v for v in auditoria["tokens"] if v not in permitidos]
    if raros:
        raise SystemExit(f"STOP: tokens de ruta inesperados: {raros}")
    prohibidos = {"transformers", "sentence_transformers", "openai", "anthropic"} & set(auditoria["modulos"])
    if prohibidos:
        raise SystemExit(f"STOP: imports prohibidos: {sorted(prohibidos)}")
    print("P90-02 parte 2 completada:")
    print(f"  informe: {RUTA_INFORME.name} ({len(informe.splitlines())} lineas)")
    print(f"  handoff: {RUTA_HANDOFF.name} ({len(handoff.splitlines())} lineas)")
    print(f"  manifiesto: {RUTA_MANIFEST.name}")
    print("  ASCII puro: OK (0 bytes > 127) | AST: sin tokens inesperados")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
