"""P7 - texto tipado (texto original + lemma_POS): ejecucion del pre-registro congelado.

## Orden de ejecucion

1. Verifica hashes y constantes contra `experiments/age_linguistic_08_texto_tipado_preregistro.json`.
2. Reproduce el baseline (V30 + plug-in = 0,6159732 en la particion canonica); si no reproduce, para.
3. Construye el canal T por fold: bloque A = texto original (VEC_PALABRA de V31-03), bloque B =
   lemma_POS; vectorizadores separados ajustados SOLO en el train de cada fold; bloques apilados;
   LogReg C=1.0 tol=1e-3. OOF por fold. Igual para md y lg.
4. Control: bloque B permutado DENTRO DE CADA FOLD (semilla 20261030); el pipeline completo se repite.
5. Integracion: combinador anidado sobre [p_V30 || p_T] normalizadas por fila; decision con la
   regla y pesos del plug-in congelados. Metricas, bootstrap, complementariedad y cubos de longitud.
6. Aplica gates y arbol de decision; escribe reportes, documentos y la seccion del notebook.

No toca test, no toca holdout, no usa leaderboard y no genera ninguna submission.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import data as datos  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

FECHA = "2026-09-30"
N_CLASES = 6
PREREG_JSON = cfg.EXPERIMENTS / "age_linguistic_08_texto_tipado_preregistro.json"
CANONICO = ROOT / "artifacts" / "p33_scores" / "todos"
PARQUETS = {"md": ROOT / "features" / "linguistic" / "anotacion_md_train.parquet",
            "lg": ROOT / "features" / "linguistic" / "anotacion_lg_train.parquet"}
RESULTADOS_CSV = cfg.REPORTS / "age_typed_text_results.csv"
COMPL_CSV = cfg.REPORTS / "age_typed_text_complementarity.csv"
DECISION_MD = cfg.REPORTS / "age_typed_text_decision.md"
RESULTADO = cfg.EXPERIMENTS / "age_linguistic_09_texto_tipado.json"
NOTEBOOK = ROOT / "notebooks" / "99_entrega_consolidado.ipynb"
MARCADOR_NB = "PHASE - TYPED TEXT REPRESENTATION"

SEMILLA_BOOTSTRAP = 20261010
SEMILLA_BARAJADO = 20261030
GATE_DELTA = 0.0015
GATE_FOLDS = 4
UMBRAL_ALTA_CONFIANZA = 0.60
VARIABILIDAD = 0.0008876
N_BOOTSTRAP = 2000
ANOTADORES = ("md", "lg")
VEC_A_STR = ('TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=3, max_df=0.5, '
             'sublinear_tf=True, lowercase=True, strip_accents=None, token_pattern=r"(?u)\\b\\w+\\b")')
VEC_B_STR = ('TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=5, sublinear_tf=True, '
             'lowercase=True, token_pattern=r"(?u)\\S+")')
VEC_A = dict(analyzer="word", ngram_range=(1, 2), min_df=3, max_df=0.5, sublinear_tf=True,
             lowercase=True, strip_accents=None, token_pattern=r"(?u)\b\w+\b")
VEC_B = dict(analyzer="word", ngram_range=(1, 2), min_df=5, sublinear_tf=True, lowercase=True,
             token_pattern=r"(?u)\S+")
CUBOS = [("cortos<=40", 0, 40), ("medios_41-93", 41, 93), ("largos_94-130", 94, 130),
         ("muy_largos>130", 131, 10**9)]


def sha256(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def normaliza(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def normaliza_l2(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0)
    norma = np.linalg.norm(p, axis=1, keepdims=True)
    return p / np.where(norma > 1e-12, norma, 1.0)


def macro(clase: np.ndarray, pred: np.ndarray, mascara: np.ndarray | None = None) -> float:
    if mascara is not None:
        return float(f1_score(clase[mascara], pred[mascara], average="macro",
                              labels=list(range(N_CLASES))))
    return float(f1_score(clase, pred, average="macro", labels=list(range(N_CLASES))))


def macro_de_conteos(conteos: np.ndarray) -> float:
    matriz = conteos.reshape(N_CLASES, N_CLASES)
    f1s = []
    for c in range(N_CLASES):
        tp = float(matriz[c, c])
        fp = float(matriz[:, c].sum() - tp)
        fn = float(matriz[c, :].sum() - tp)
        denom = 2.0 * tp + fp + fn
        f1s.append(2.0 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(f1s))


def bootstrap_delta(clase: np.ndarray, pred_base: np.ndarray, pred_cand: np.ndarray) -> dict:
    rng = np.random.default_rng(SEMILLA_BOOTSTRAP)
    pb = clase * N_CLASES + pred_base
    pc = clase * N_CLASES + pred_cand
    n = len(clase)
    deltas = np.empty(N_BOOTSTRAP, dtype=np.float64)
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        cb = np.bincount(pb[idx], minlength=N_CLASES * N_CLASES)
        cc = np.bincount(pc[idx], minlength=N_CLASES * N_CLASES)
        deltas[b] = macro_de_conteos(cc) - macro_de_conteos(cb)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"delta_medio": round(float(deltas.mean()), 7),
            "ic95": [round(float(lo), 7), round(float(hi), 7)],
            "p_delta_positivo": round(float((deltas > 0).mean()), 4)}


def canal_oof(textos: np.ndarray, tipado: np.ndarray, clase: np.ndarray, fold_idx: np.ndarray,
              etiqueta: str) -> tuple[np.ndarray, dict]:
    """OOF del canal T: bloques A (texto) y B (lemma_POS) ajustados SOLO en train de cada fold."""
    p = np.zeros((len(clase), N_CLASES), dtype=np.float64)
    vocab_a, vocab_b, segundos, iters, warns_total = [], [], [], [], 0
    for k in range(5):
        t0 = time.time()
        tr = fold_idx != k
        va = fold_idx == k
        vec_a = TfidfVectorizer(**VEC_A)
        vec_b = TfidfVectorizer(**VEC_B)
        xa_tr = vec_a.fit_transform(textos[tr])
        xb_tr = vec_b.fit_transform(tipado[tr])
        x_tr = sparse.hstack([xa_tr, xb_tr]).tocsr()
        x_va = sparse.hstack([vec_a.transform(textos[va]),
                              vec_b.transform(tipado[va])]).tocsr()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mod = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", tol=1e-3)
            mod.fit(x_tr, clase[tr])
            warns_total += sum(1 for x in w if "Convergence" in str(x.message))
        p[va] = mod.predict_proba(x_va)
        vocab_a.append(int(xa_tr.shape[1]))
        vocab_b.append(int(xb_tr.shape[1]))
        iters.append(int(mod.n_iter_[0]))
        segundos.append(round(time.time() - t0, 1))
        print(f"    [{etiqueta}] fold {k}: A {xa_tr.shape[1]:,} + B {xb_tr.shape[1]:,} features | "
              f"{segundos[-1]} s | n_iter {iters[-1]}", flush=True)
    p = normaliza(p)
    return p, {"vocabulario_A": vocab_a, "vocabulario_B": vocab_b,
               "segundos_por_fold": segundos, "n_iter_por_fold": iters,
               "warnings_convergencia": warns_total}


def combinador_oof(v30: np.ndarray, p_canal: np.ndarray, clase: np.ndarray,
                   fold_idx: np.ndarray) -> tuple[np.ndarray, dict]:
    features = np.hstack([normaliza_l2(v30), normaliza_l2(p_canal)])
    p = np.zeros((len(clase), N_CLASES), dtype=np.float64)
    razones = []
    for k in range(5):
        tr = fold_idx != k
        va = fold_idx == k
        mod = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        mod.fit(features[tr], clase[tr])
        p[va] = mod.predict_proba(features[va])
        coef = np.abs(mod.coef_)
        razones.append(round(float(coef[:, N_CLASES:].mean()
                                   / max(coef[:, :N_CLASES].mean(), 1e-12)), 4))
    p = normaliza(p)
    return p, {"peso_bloque_canal_ratio_por_fold": razones,
               "peso_bloque_canal_ratio_media": round(float(np.mean(razones)), 4)}


def evalua(p_comb: np.ndarray, clase: np.ndarray, fold_idx: np.ndarray, champ_pred: np.ndarray,
           pesos: np.ndarray, baseline: float) -> dict:
    pred = np.argmax(p_comb * pesos, axis=1)
    cv = macro(clase, pred)
    folds = [macro(clase, pred, fold_idx == k) for k in range(5)]
    folds_base = [macro(clase, champ_pred, fold_idx == k) for k in range(5)]
    ic = bootstrap_delta(clase, champ_pred, pred)
    return {"corrected_cv": round(cv, 7), "delta_vs_baseline": round(cv - baseline, 7),
            "delta_vs_nested": round(cv - 0.6154643, 7),
            "folds": [round(v, 6) for v in folds],
            "folds_positivos": int(sum(1 for a, b in zip(folds, folds_base) if a > b)),
            "filas_que_cambian": int((pred != champ_pred).sum()),
            "ic95": ic["ic95"], "p_delta_positivo": ic["p_delta_positivo"],
            "argmax_sin_regla": round(macro(clase, np.argmax(p_comb, axis=1)), 7),
            "_pred": pred}


def complementariedad(clave: str, pred: np.ndarray, clase: np.ndarray, champ_pred: np.ndarray,
                       conf: np.ndarray, filas: list[dict]) -> None:
    ok_c = pred == clase
    ok_ch = champ_pred == clase
    alta = (~ok_ch) & (conf >= UMBRAL_ALTA_CONFIANZA)
    filas.append({
        "anotador": clave, "clase": "TODAS", "filas": int(len(clase)),
        "campeon_ok_cand_no": int((ok_ch & ~ok_c).sum()),
        "campeon_no_cand_ok": int((~ok_ch & ok_c).sum()),
        "ambos_ok": int((ok_ch & ok_c).sum()), "ambos_no": int((~ok_ch & ~ok_c).sum()),
        "rescate_unico": int((~ok_ch & ok_c).sum()), "danio": int((ok_ch & ~ok_c).sum()),
        "neto": int((~ok_ch & ok_c).sum()) - int((ok_ch & ~ok_c).sum()),
        "rescates_alta_confianza": int((alta & ok_c).sum()),
        "errores_alta_confianza_campeon": int(alta.sum())})
    for c, nombre in enumerate(cfg.CLASSES):
        mascara = clase == c
        rescate = int((~ok_ch & ok_c & mascara).sum())
        danio = int((ok_ch & ~ok_c & mascara).sum())
        filas.append({"anotador": clave, "clase": nombre, "filas": int(mascara.sum()),
                      "campeon_ok_cand_no": danio, "campeon_no_cand_ok": rescate,
                      "ambos_ok": int((ok_ch & ok_c & mascara).sum()),
                      "ambos_no": int((~ok_ch & ~ok_c & mascara).sum()),
                      "rescate_unico": rescate, "danio": danio, "neto": rescate - danio,
                      "rescates_alta_confianza": "", "errores_alta_confianza_campeon": ""})


def cubos_longitud(pred: np.ndarray, champ_pred: np.ndarray, clase: np.ndarray,
                   largos: np.ndarray) -> list[dict]:
    filas = []
    for nombre, lo, hi in CUBOS:
        mascara = (largos >= lo) & (largos <= hi)
        if mascara.sum() == 0:
            continue
        filas.append({"cubo": nombre, "documentos": int(mascara.sum()),
                      "macro_f1_campeon": round(macro(clase, champ_pred, mascara), 6),
                      "macro_f1_candidato": round(macro(clase, pred, mascara), 6),
                      "delta": round(macro(clase, pred, mascara)
                                     - macro(clase, champ_pred, mascara), 7)})
    return filas


def auditoria_compliance() -> dict:
    """Verificacion negativa sobre los scripts de la fase; el propio auditor se autoexcluye."""
    propio = Path(__file__).name
    problemas = []
    for ruta in sorted((ROOT / "scripts").glob("age_linguistic_*.py")):
        if ruta.name == propio:
            continue
        texto = ruta.read_text(encoding="utf-8")
        for patron, nombre in ((r"read_csv\([^)]*age_test|load_test\(", "lectura de test"),
                               (r"import\s+transformers", "transformers"),
                               (r"sentence_transformers|SentenceTransformer", "embeddings"),
                               (r"openai\.|anthropic\.|from\s+llama", "LLM"),
                               (r"requests\.(get|post)|urllib\.request", "red")):
            if re.search(patron, texto):
                problemas.append({"script": ruta.name, "hallazgo": nombre})
    return {"hallazgos": problemas, "limpio": len(problemas) == 0, "auditor_excluido": propio}


def escribe_results_csv(filas: list[dict]) -> None:
    cols = ["anotador", "brazo", "corrected_cv", "delta_vs_baseline", "delta_vs_nested",
            "folds_positivos", "folds", "ic95_lo", "ic95_hi", "filas_que_cambian",
            "vocab_A_medio", "vocab_B_medio", "segundos_canal", "n_iter_medio",
            "warnings", "peso_bloque_canal_ratio", "texto_tipado_solo",
            "argmax_sin_regla", "pasa_gate"]
    with RESULTADOS_CSV.open("w", encoding="utf-8", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        escritor.writeheader()
        for fila in filas:
            escritor.writerow({c: fila.get(c, "") for c in cols})


def escribe_decision_md(lote: dict, baseline: float) -> None:
    """Informe de decision con veredicto, lectura, cubos, compliance y limites."""
    lineas = [
        "# P7 - texto tipado (texto original + lemma_POS): decision",
        "",
        f"**Fecha**: {FECHA} | **Pre-registro**: `reports/age_typed_text_preregistro.md` | "
        f"**Baseline**: {baseline:.7f} (brazo A = ablacion sin canal, D-076 condicion 4)",
        "",
        "## Veredicto",
        "",
        "| Anotador | T solo (macro-F1) | delta real | folds+ | IC95 | delta barajado | Gate |",
        "|---|---|---|---|---|---|---|",
    ]
    for clave in ANOTADORES:
        r = lote["brazos"][clave]
        rb = lote["brazos_barajados"][clave]
        ic = r["ic95"]
        estado = "**PASA**" if r["pasa_gate"] else "no pasa"
        lineas.append(f"| {clave} | {lote['canal_solo'][clave]:.7f} | "
                      f"{r['delta_vs_baseline']:+.7f} | {r['folds_positivos']}/5 | "
                      f"[{ic[0]:+.7f}; {ic[1]:+.7f}] | {rb['delta_vs_baseline']:+.7f} | {estado} |")
    global_ = ("PROMISING" if lote["promising"] else
               ("MARGINAL" if lote["marginal"] else "CLOSED"))
    lineas += [
        "",
        f"**Resultado global: {global_}**",
        "",
        "## Lectura",
        "",
        "- La comparacion decisiva es real vs barajado por el mismo pipeline completo. Si el ",
        "  control empata o gana, no hay atribucion positiva.",
    ]
    for clave in ANOTADORES:
        r = lote["brazos"][clave]
        det = lote["detalle_canal"][clave]
        lineas.append(
            f"- **{clave}**: T solo {lote['canal_solo'][clave]:.7f}; delta "
            f"{r['delta_vs_baseline']:+.7f} vs barajado "
            f"{lote['brazos_barajados'][clave]['delta_vs_baseline']:+.7f}; peso del bloque canal "
            f"en el combinador {lote['combinador'][clave]['peso_bloque_canal_ratio_media']}; "
            f"vocab medios A/B {round(float(np.mean(det['vocabulario_A']))):,}/"
            f"{round(float(np.mean(det['vocabulario_B']))):,}; n_iter medio "
            f"{round(float(np.mean(det['n_iter_por_fold'])), 1)}; warnings "
            f"{det['warnings_convergencia']}; B vacios {det['B_vacios']}; diagonal de clases OK.")
    lineas += [
        "",
        "## Complementariedad (resumen TODAS)",
        "",
        "| Anotador | rescate unico | danio | neto | campeon ok / cand no | campeon no / cand ok | "
        "ambos ok | ambos no | rescates alta confianza |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for clave in ANOTADORES:
        fila = next(f for f in lote["filas_comp"] if f["anotador"] == clave and f["clase"] == "TODAS")
        lineas.append(
            f"| {clave} | {fila['rescate_unico']:,} | {fila['danio']:,} | {fila['neto']:+,} | "
            f"{fila['campeon_ok_cand_no']:,} | {fila['campeon_no_cand_ok']:,} | "
            f"{fila['ambos_ok']:,} | {fila['ambos_no']:,} | "
            f"{fila['rescates_alta_confianza']:,} / {fila['errores_alta_confianza_campeon']:,} |")
    lineas += [
        "",
        "Detalle por clase de edad en `reports/age_typed_text_complementarity.csv`.",
        "",
        "## Cubos de longitud (descriptivo, no decisorio)",
        "",
        "| Anotador | Cubo | Docs | macro-F1 campeon | macro-F1 candidato | delta |",
        "|---|---|---|---|---|---|",
    ]
    for clave in ANOTADORES:
        for fila in lote["cubos"][clave]:
            lineas.append(f"| {clave} | {fila['cubo']} | {fila['documentos']:,} | "
                          f"{fila['macro_f1_campeon']} | {fila['macro_f1_candidato']} | "
                          f"{fila['delta']:+.7f} |")
    lineas += [
        "",
        "## Compliance (registrado)",
        "",
        "| Restriccion | Estado |",
        "|---|---|",
        "| lectura de test / labels de test | 0 |",
        "| entrenamiento o calibracion con test | 0 |",
        "| holdout / leaderboard (tuning) | 0 |",
        "| embeddings / transformers / LLM | 0 |",
        "| datos o etiquetas externas | 0 |",
        f"| auditoria automatica de scripts (auditor excluido) | {lote['compliance']} |",
        "",
        "## Provenance",
        "",
        "Modelos y anotaciones: SHA256 en `experiments/age_linguistic_08_texto_tipado_preregistro.json`",
        "y en el resultado del runner; particion canonica y manifest del plug-in verificados al",
        "arrancar (hashes_ok = True). Semillas: bootstrap 20261010, barajado 20261030.",
        "",
        "## Limites declarados",
        "",
        "- Una sola semilla de barajado y un solo mecanismo de integracion (presupuesto del brief).",
        "- PII registrada: los lemas PROPN provienen del propio tweet (ya presentes en el bloque A).",
        "- El claim es empirico y acotado: el resultado rige para ESTA representacion y ESTE",
        "  protocolo; no se extrapola a P2, P6 ni a 'la informacion linguistica' en general.",
    ]
    texto = "\n".join(l.rstrip() for l in lineas) + "\n"
    DECISION_MD.write_text(texto, encoding="utf-8")


def anade_notebook(lote: dict, baseline: float) -> dict:
    import nbformat
    nb = nbformat.read(NOTEBOOK, as_version=4)
    antes = len(nb.cells)
    for i, c in enumerate(nb.cells):
        if c.cell_type == "markdown" and MARCADOR_NB in c.source:
            del nb.cells[i:]
            break

    def md(t):
        nb.cells.append(nbformat.v4.new_markdown_cell(t))

    def code(t):
        nb.cells.append(nbformat.v4.new_code_cell(t))

    estado = ("PROMISING" if lote["promising"] else ("MARGINAL" if lote["marginal"] else "CLOSED"))
    md(f"""## PHASE - TYPED TEXT REPRESENTATION

Hipotesis: una representacion que conserva el **texto original** y anade la **funcion
linguistica por token** (lexema + POS: `querer_VERB`, `casa_NOUN`, `muy_ADV`) puede aportar
informacion complementaria al campeon (V30 + plug-in, LB 0,63575).

- Gap analysis por operacion: **GENUINELY_NEW** (0 consumidores previos de `lemma_pos_seq` como
  representacion; detalle en `reports/age_typed_text_gap_analysis.md`).
- Legalidad: D-076 (etiquetas y lemas si; vectores no); anotaciones congeladas, sin anotadores
  nuevos. PII: los lemas PROPN vienen del propio tweet (ya en el texto original).
- Representacion: dos bloques sparse - A = texto original (TF-IDF word 1-2, clon de V31-03),
  B = lemma_POS (1-2 gramas, min_df=5); vectorizadores separados por fold; escala 1.0 por bloque.
- Baseline: brazo A = V30 + plug-in = {baseline:.7f} (ablacion sin canal, D-076 cond.4).
- Control: bloque B permutado dentro de cada fold (semilla 20261030) por el pipeline completo.

**Resultado: {estado}.**""")

    code("""import json, pathlib
import pandas as pd
RAIZ = pathlib.Path("..")
res = json.loads((RAIZ / "experiments" / "age_linguistic_09_texto_tipado.json").read_text(encoding="utf-8"))
print("verificacion:", res["verificacion"], "| baseline:", res["baseline_primario"])
tabla = pd.read_csv(RAIZ / "reports" / "age_typed_text_results.csv")
print()
print(tabla.to_string(index=False))
print()
comp = pd.read_csv(RAIZ / "reports" / "age_typed_text_complementarity.csv")
todas = comp[comp["clase"] == "TODAS"]
print(todas[["anotador","rescate_unico","danio","neto","rescates_alta_confianza","errores_alta_confianza_campeon"]].to_string(index=False))
assert res["verificacion"]["hashes_ok"] is True
assert res["verificacion"]["constantes_ok"] is True
assert abs(res["baseline_primario"] - 0.6159732) < 1e-6
assert res["sin_test"] is True and res["sin_submission"] is True
print()
print("[OK] P7 ejecutado contra el pre-registro congelado; sin tocar test")""")
    md("""### Decision y limites

La decision sigue el arbol pre-registrado (PROMISING / MARGINAL / CLOSED). Limites declarados:
una semilla de barajado y un mecanismo de integracion (presupuesto); resultado acotado a esta
representacion y este protocolo, sin extrapolar a P2/P6. Detalle en
`reports/age_typed_text_decision.md`.""")
    nbformat.write(nb, NOTEBOOK)
    return {"antes": antes, "despues": len(nb.cells), "anadidas": len(nb.cells) - antes}


def d_079(lote: dict, baseline: float) -> str:
    if lote["promising"]:
        estado, siguiente = ("PROMISING",
                             "bootstrap y complementariedad ya calculados; segunda particion "
                             "agrupada con configuracion congelada (semilla 987654) como "
                             "siguiente paso; sin submission sin autorizacion")
    elif lote["marginal"]:
        estado, siguiente = ("MARGINAL",
                             "cierre documentado sin abrir grid; P5 se evaluara una sola vez "
                             "solo si el gap la justifica")
    else:
        estado, siguiente = ("CLOSED",
                             "cierre documentado; P5 pendiente de una unica evaluacion del gap")
    lineas = ["---", ""]
    if lote["promising"]:
        titulo = f"## D-079 - P7: el texto tipado (texto original + lemma_POS) pasa el gate ({estado})"
    else:
        titulo = f"## D-079 - P7: el texto tipado (texto original + lemma_POS) queda en {estado}"
    lineas += [titulo, ""]
    partes = " | ".join(
        f"{clave}: delta {lote['brazos'][clave]['delta_vs_baseline']:+.7f} "
        f"({lote['brazos'][clave]['folds_positivos']}/5), barajado "
        f"{lote['brazos_barajados'][clave]['delta_vs_baseline']:+.7f}"
        for clave in ANOTADORES)
    lineas += [
        f"- **Observation**: se ejecuto P7 con el pre-registro congelado. Baseline primario "
        f"recomputado: {baseline:.7f} (brazo A = ablacion sin canal, D-076 cond.4). "
        f"Resultados: {partes}.",
        f"- **Result**: texto tipado solo (md) macro-F1 {lote['canal_solo']['md']:.7f}; "
        f"peso del bloque canal en el combinador "
        f"{lote['combinador']['md']['peso_bloque_canal_ratio_media']} (md).",
        f"- **Decision**: {siguiente}.",
        "- **Consequences**: si el resultado no es positivo, la conclusion es 'la representacion "
        "lexema+POS no aporta mejora al campeon bajo el mecanismo evaluado' (no es una afirmacion "
        "sobre toda la informacion linguistica). El campeon adoptado (V30 + plug-in, LB 0.63575) "
        "permanece intacto. P2 y P6 no se reabren.",
        "- **Evidence**: `experiments/age_linguistic_09_texto_tipado.json`, "
        "`reports/age_typed_text_results.csv`, `reports/age_typed_text_complementarity.csv`, "
        "`reports/age_typed_text_decision.md`.",
    ]
    return "\n".join(lineas) + "\n"


def anade(ruta: Path, marca: str, bloque: str) -> str:
    if not ruta.exists():
        return f"{ruta.name}: no existe"
    texto = ruta.read_text(encoding="utf-8")
    if marca in texto:
        return f"{ruta.name}: ya estaba"
    cuerpo = (texto if texto.endswith("\n") else texto + "\n") + bloque
    cuerpo = "\n".join(l.rstrip() for l in cuerpo.splitlines()) + "\n"
    ruta.write_text(cuerpo, encoding="utf-8")
    return f"{ruta.name}: +{len(bloque.splitlines())} lineas"


def main() -> int:
    autotest_numerico()
    inicio = time.time()
    prereg = json.loads(PREREG_JSON.read_text(encoding="utf-8"))

    print("=== verificacion contra el pre-registro ===", flush=True)
    constantes_ok = (prereg["gates"]["delta_min"] == GATE_DELTA
                     and prereg["gates"]["folds_min"] == GATE_FOLDS
                     and prereg["semillas"] == {"bootstrap": SEMILLA_BOOTSTRAP,
                                                "barajado": SEMILLA_BARAJADO}
                     and prereg["vec_a"] == VEC_A_STR and prereg["vec_b"] == VEC_B_STR
                     and prereg["escala_por_bloque"] == 1.0)
    rutas_hashes = {
        "anotacion_md_parquet": PARQUETS["md"],
        "anotacion_lg_parquet": PARQUETS["lg"],
        "corpus_train": ROOT / "data" / "raw" / "age_train.csv",
        "canonico_y_true": CANONICO / "y_true.npy",
        "canonico_fold_idx": CANONICO / "fold_idx.npy",
        "canonico_v30_prob": CANONICO / "v30_prob.npy",
        "manifest_plugin": ROOT / "reports" / "age_submission_plugin_manifest.json",
    }
    hashes_ok = True
    for nombre, ruta in rutas_hashes.items():
        if sha256(ruta) != prereg["hashes"][nombre]:
            hashes_ok = False
            print(f"  [STOP] hash distinto en {nombre}", flush=True)
    print(f"  constantes ok: {constantes_ok} | hashes ok: {hashes_ok} | sklearn {sklearn.__version__}",
          flush=True)
    if not (constantes_ok and hashes_ok):
        print("[STOP] el pre-registro no se respeta", flush=True)
        return 1

    pesos = np.array(prereg["pesos_plugin"], dtype=np.float64)
    clase = np.load(CANONICO / "y_true.npy")
    fold_idx = np.load(CANONICO / "fold_idx.npy")
    v30 = normaliza(np.load(CANONICO / "v30_prob.npy").astype(np.float64))
    champ_pred = np.argmax(v30 * pesos, axis=1)
    conf = v30.max(axis=1)
    baseline = macro(clase, champ_pred)
    print(f"  baseline (brazo A): {baseline:.7f} (esperado {prereg['baseline_primario']})",
          flush=True)
    if abs(baseline - prereg["baseline_primario"]) > 1e-6:
        print("[STOP] el baseline no reproduce", flush=True)
        return 1

    train = datos.load_train()
    textos = train[cfg.TEXT_COL].fillna("").astype(str).to_numpy()
    largos = np.array([len(t) for t in textos])

    # control: bloque B permutado dentro de cada fold (mismo mapeo para md y lg)
    rng = np.random.default_rng(SEMILLA_BARAJADO)
    perm = np.arange(len(clase))
    for k in range(5):
        idx = np.flatnonzero(fold_idx == k)
        perm[idx] = idx[rng.permutation(len(idx))]

    lote = {"brazos": {}, "brazos_barajados": {}, "detalle_canal": {},
            "detalle_canal_barajado": {}, "combinador": {}, "combinador_barajado": {},
            "canal_solo": {}, "canal_solo_barajado": {}, "cubos": {}, "filas_csv": [],
            "filas_comp": []}
    for clave in ANOTADORES:
        frame = pd.read_parquet(PARQUETS[clave], columns=["lemma_pos_seq", "n_tokens"])
        tipado = frame["lemma_pos_seq"].astype(str).to_numpy()
        vacios = int(sum(1 for s in tipado if not s.strip()))
        assert not frame["lemma_pos_seq"].isna().any(), "nulos en el bloque B"
        print(f"\n=== canal {clave}: {len(tipado):,} docs | B vacios {vacios} ===", flush=True)

        print(f"--- {clave} real ---", flush=True)
        p_t, info = canal_oof(textos, tipado, clase, fold_idx, f"T-{clave}")
        # chequeo de orden de clases (el bug silencioso tipico): diagonal dominante
        diagonal = []
        for i in range(N_CLASES):
            fila = p_t[clase == i].mean(axis=0)
            diagonal.append(int(np.argmax(fila)) == i)
        if not all(diagonal):
            print(f"[STOP] orden de clases roto en {clave}: {diagonal}", flush=True)
            return 1
        p_comb, comb = combinador_oof(v30, p_t, clase, fold_idx)
        r = evalua(p_comb, clase, fold_idx, champ_pred, pesos, baseline)
        canal_solo = macro(clase, np.argmax(p_t, axis=1))

        print(f"--- {clave} barajado ---", flush=True)
        tipado_shuf = tipado[perm]
        p_t_b, info_b = canal_oof(textos, tipado_shuf, clase, fold_idx, f"T-{clave}-barajado")
        p_comb_b, comb_b = combinador_oof(v30, p_t_b, clase, fold_idx)
        r_b = evalua(p_comb_b, clase, fold_idx, champ_pred, pesos, baseline)

        pasa = (r["delta_vs_baseline"] >= GATE_DELTA and r["folds_positivos"] >= GATE_FOLDS
                and r["ic95"][0] > 0 and r["delta_vs_baseline"] > r_b["delta_vs_baseline"])
        r["pasa_gate"] = bool(pasa)
        print(f"  {clave}: T solo {canal_solo:.7f} | delta {r['delta_vs_baseline']:+.7f} "
              f"({r['folds_positivos']}/5) | barajado {r_b['delta_vs_baseline']:+.7f} -> "
              f"{'PASA' if pasa else 'no pasa'}", flush=True)

        lote["brazos"][clave] = r
        lote["brazos_barajados"][clave] = r_b
        lote["detalle_canal"][clave] = {**info, "B_vacios": vacios,
                                        "diagonal_clases": diagonal}
        lote["detalle_canal_barajado"][clave] = info_b
        lote["combinador"][clave] = comb
        lote["combinador_barajado"][clave] = comb_b
        lote["canal_solo"][clave] = round(canal_solo, 7)
        lote["canal_solo_barajado"][clave] = round(macro(clase, np.argmax(p_t_b, axis=1)), 7)
        pred_cand = r.pop("_pred")
        lote["cubos"][clave] = cubos_longitud(pred_cand, champ_pred, clase, largos)
        complementariedad(clave, pred_cand, clase, champ_pred, conf, lote["filas_comp"])
        r_b.pop("_pred", None)
        lote["filas_csv"].append({
            "anotador": clave, "brazo": "real", "corrected_cv": r["corrected_cv"],
            "delta_vs_baseline": r["delta_vs_baseline"], "delta_vs_nested": r["delta_vs_nested"],
            "folds_positivos": r["folds_positivos"],
            "folds": ";".join(f"{v:.6f}" for v in r["folds"]),
            "ic95_lo": r["ic95"][0], "ic95_hi": r["ic95"][1],
            "filas_que_cambian": r["filas_que_cambian"],
            "vocab_A_medio": round(float(np.mean(info["vocabulario_A"])), 0),
            "vocab_B_medio": round(float(np.mean(info["vocabulario_B"])), 0),
            "segundos_canal": sum(info["segundos_por_fold"]),
            "n_iter_medio": round(float(np.mean(info["n_iter_por_fold"])), 0),
            "warnings": info["warnings_convergencia"],
            "peso_bloque_canal_ratio": comb["peso_bloque_canal_ratio_media"],
            "texto_tipado_solo": lote["canal_solo"][clave],
            "argmax_sin_regla": r["argmax_sin_regla"], "pasa_gate": r["pasa_gate"]})
        lote["filas_csv"].append({
            "anotador": clave, "brazo": "barajado", "corrected_cv": r_b["corrected_cv"],
            "delta_vs_baseline": r_b["delta_vs_baseline"],
            "delta_vs_nested": r_b["delta_vs_nested"],
            "folds_positivos": r_b["folds_positivos"],
            "folds": ";".join(f"{v:.6f}" for v in r_b["folds"]),
            "ic95_lo": r_b["ic95"][0], "ic95_hi": r_b["ic95"][1],
            "filas_que_cambian": r_b["filas_que_cambian"],
            "vocab_A_medio": round(float(np.mean(info_b["vocabulario_A"])), 0),
            "vocab_B_medio": round(float(np.mean(info_b["vocabulario_B"])), 0),
            "segundos_canal": sum(info_b["segundos_por_fold"]),
            "n_iter_medio": round(float(np.mean(info_b["n_iter_por_fold"])), 0),
            "warnings": info_b["warnings_convergencia"],
            "peso_bloque_canal_ratio": comb_b["peso_bloque_canal_ratio_media"],
            "texto_tipado_solo": lote["canal_solo_barajado"][clave],
            "argmax_sin_regla": r_b["argmax_sin_regla"], "pasa_gate": ""})

    lote["promising"] = any(r["pasa_gate"] for r in lote["brazos"].values())
    lote["marginal"] = (not lote["promising"]
                        and any(r["delta_vs_baseline"] > 0 for r in lote["brazos"].values()))
    lote["compliance"] = auditoria_compliance()
    print(f"\n  PROMISING: {lote['promising']} | MARGINAL: {lote['marginal']} | "
          f"compliance limpio: {lote['compliance']['limpio']}", flush=True)

    escribe_results_csv(lote["filas_csv"])
    cols_c = ["anotador", "clase", "filas", "campeon_ok_cand_no", "campeon_no_cand_ok",
              "ambos_ok", "ambos_no", "rescate_unico", "danio", "neto",
              "rescates_alta_confianza", "errores_alta_confianza_campeon"]
    with COMPL_CSV.open("w", encoding="utf-8", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=cols_c, lineterminator="\n")
        escritor.writeheader()
        for fila in lote["filas_comp"]:
            escritor.writerow({c: fila.get(c, "") for c in cols_c})

    RESULTADO.write_text(json.dumps({
        "fase": "P7 texto tipado", "fecha": FECHA,
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT.parent,
                                 capture_output=True, text=True, timeout=30).stdout.strip(),
        "verificacion": {"hashes_ok": hashes_ok, "constantes_ok": constantes_ok},
        "baseline_primario": round(baseline, 7),
        "brazos": lote["brazos"], "brazos_barajados": lote["brazos_barajados"],
        "detalle_canal": lote["detalle_canal"], "detalle_canal_barajado":
            lote["detalle_canal_barajado"],
        "combinador": lote["combinador"], "canal_solo": lote["canal_solo"],
        "canal_solo_barajado": lote["canal_solo_barajado"], "cubos_longitud": lote["cubos"],
        "promising": lote["promising"], "marginal": lote["marginal"],
        "compliance": lote["compliance"],
        "sin_test": True, "sin_holdout": True, "sin_submission": True,
        "segundos_total": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("  experiments/age_linguistic_09_texto_tipado.json", flush=True)

    print("\n=== documentacion ===", flush=True)
    escribe_decision_md(lote, baseline)
    print("  " + anade(ROOT / "DECISIONS.md", "## D-079", d_079(lote, baseline)), flush=True)
    print(f"  {DECISION_MD.name} | {RESULTADOS_CSV.name} | {COMPL_CSV.name}", flush=True)

    runlog = ROOT / "RUNLOG.md"
    if "P7 ejecutado: texto tipado" not in runlog.read_text(encoding="utf-8"):
        resumen = " | ".join(f"{k}: {v['delta_vs_baseline']:+.7f} ({v['folds_positivos']}/5)"
                             for k, v in sorted(lote["brazos"].items()))
        estado = ("PROMISING" if lote["promising"] else
                  ("MARGINAL" if lote["marginal"] else "CLOSED"))
        bloque = ("\n---\n\n## P7 ejecutado: texto tipado (texto original + lemma_POS) · "
                  + FECHA + "\n\n"
                  f"- Resultado: {estado}. Brazos: {resumen}. T solo md "
                  f"{lote['canal_solo']['md']:.4f}.\n"
                  "- Control: bloque B permutado dentro de cada fold por el pipeline completo.\n"
                  f"- Baseline {baseline:.7f}; sin tuning; sin test; sin holdout; sin submission.\n")
        with runlog.open("a", encoding="utf-8") as f:
            f.write(bloque)
        print("  RUNLOG.md: P7 anotado", flush=True)

    ruta_exp = ROOT / "experiments.csv"
    lineas_exp = ruta_exp.read_text(encoding="utf-8").splitlines()
    cols = [c.strip() for c in lineas_exp[0].split(",")]
    if "AGE-LING-P7-TYPED" not in {l.split(",", 1)[0] for l in lineas_exp[1:] if l}:
        estado = ("PROMISING" if lote["promising"] else
                  ("MARGINAL" if lote["marginal"] else "CLOSED"))
        resumen = " | ".join(f"{k}: delta {v['delta_vs_baseline']:+.7f} "
                             f"({v['folds_positivos']}/5)" for k, v in sorted(lote["brazos"].items()))
        reg = {"experiment_id": "AGE-LING-P7-TYPED", "fecha": FECHA, "bloque": "30.4",
               "etapa": "P7 ejecucion: texto tipado (texto original + lemma_POS) contra el campeon",
               "pregunta": "una representacion lexica tipada anade informacion complementaria al "
                           "campeon?",
               "hipotesis": "lexema + funcion por token, conservando el texto original",
               "features": "bloque A texto original (word 1-2, V31-03) + bloque B lemma_POS (1-2)",
               "preprocessing": "dos vectorizadores separados por fold; escala 1.0 por bloque",
               "representacion": "texto tipado en dos bloques sparse",
               "modelo": "LogReg C=1.0 tol=1e-3 + combinador anidado (12 feats) + regla congelada",
               "hiperparametros": "congelados por pre-registro (AGE-LING-P7-PREREG)",
               "validacion": "particion canonica; baseline 0.6159732; control B permutado dentro "
                             "de fold (20261030) por pipeline completo; gates +0.0015, 4/5, "
                             "real>barajado, bootstrap",
               "macro_f1_mean": "", "macro_f1_std": "", "weighted_f1_mean": "",
               "micro_f1_mean": "", "accuracy_mean": "",
               "resultado": f"{estado} | {resumen} | T solo md {lote['canal_solo']['md']:.7f}",
               "interpretacion": "lectura por el arbol pre-registrado; sin extrapolar a P2/P6",
               "decision": ("segunda particion congelada" if lote["promising"] else
                            "cierre documentado; P5 se evaluara una sola vez"),
               "duracion_seg": f"{round(time.time() - inicio, 1)}",
               "artefactos": "reports/age_typed_text_decision.md | "
                             "reports/age_typed_text_results.csv | "
                             "reports/age_typed_text_complementarity.csv | "
                             "experiments/age_linguistic_09_texto_tipado.json"}
        with ruta_exp.open("a", encoding="utf-8") as f:
            f.write(",".join(f'"{reg[c]}"' if ("," in str(reg[c]) or '"' in str(reg[c]))
                             else str(reg[c]) for c in cols) + "\n")
        print("  experiments.csv: fila AGE-LING-P7-TYPED", flush=True)

    estado = ("PROMISING" if lote["promising"] else ("MARGINAL" if lote["marginal"] else "CLOSED"))
    bloque_estado = ("\n---\n\n## P7 ejecutado: texto tipado (" + FECHA + ")\n\n"
                     f"- Resultado: {estado}. T solo (md) {lote['canal_solo']['md']:.7f}; "
                     f"delta real/barajado por anotador en el informe.\n"
                     f"- Baseline {baseline:.7f}; sin test, sin holdout, sin submission; "
                     "campeon intacto.\n")
    print("  " + anade(ROOT / "PROJECT_STATE.md", "## P7 ejecutado: texto tipado",
                       bloque_estado), flush=True)
    print("  " + anade(ROOT / "FINAL_REPORT.md", "## P7 ejecutado: texto tipado",
                       bloque_estado), flush=True)
    print("  " + str(anade_notebook(lote, baseline)), flush=True)

    print(f"\n[fin] {time.time() - inicio:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
