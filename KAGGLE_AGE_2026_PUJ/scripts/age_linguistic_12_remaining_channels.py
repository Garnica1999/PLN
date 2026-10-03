"""Ciclo final - familias restantes (R3 lemma_POS, R4 morfologia, R5 dependencias).

## Orden de ejecucion

1. Verifica hashes y constantes contra `experiments/age_linguistic_11_remaining_preregistro.json`.
2. Reproduce el baseline (V30 + plug-in = 0,6159732); re-valida el artefacto dep_edges contra las
   columnas almacenadas (0 discrepancias exigidas).
3. Construye las secuencias de cada canal:
   R3 = lemma_pos_seq (congelada); R4 = morph_seq sin '_'; R5 = dep->POS_cabeza (derivado).
4. Por familia (orden R3 -> R4 -> R5), anotador (md, lg) y condicion (real, barajado intra-fold
   semilla 20261031): canal OOF por fold (LogReg C=1.0 tol=1e-3), combinador anidado, metricas,
   bootstrap y complementariedad.
5. Aplica gates y arbol de decision; escribe reportes, registros y la seccion del notebook.

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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

FECHA = "2026-09-30"
N_CLASES = 6
PREREG_JSON = cfg.EXPERIMENTS / "age_linguistic_11_remaining_preregistro.json"
CANONICO = ROOT / "artifacts" / "p33_scores" / "todos"
ANOT_DIR = ROOT / "features" / "linguistic"
RESULTADOS_CSV = cfg.REPORTS / "age_remaining_results.csv"
COMPL_CSV = cfg.REPORTS / "age_remaining_complementarity.csv"
DECISION_MD = cfg.REPORTS / "age_remaining_decision.md"
RESULTADO = cfg.EXPERIMENTS / "age_linguistic_12_remaining_channels.json"
NOTEBOOK = ROOT / "notebooks" / "99_entrega_consolidado.ipynb"
MARCADOR_NB = "PHASE - REMAINING STRUCTURED CHANNELS"

SEMILLA_BOOTSTRAP = 20261010
SEMILLA_BARAJADO = 20261031
GATE_DELTA = 0.0015
GATE_FOLDS = 4
UMBRAL_ALTA_CONFIANZA = 0.60
VARIABILIDAD = 0.0008876
N_BOOTSTRAP = 2000
ANOTADORES = ("md", "lg")
FAMILIAS = {
    "R3_lemma_POS": {"nombre": "lemma_POS", "vec": "identico a R3"},
    "R4_morfologia": {"nombre": "morfologia", "vec": "identico a R3"},
    "R5_dependencias": {"nombre": "dependencias", "vec": "identico a R3"},
}
VEC_STR = "TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df=5, sublinear_tf=True, " \
          "lowercase=True, token_pattern=r'(?u)\\S+')"
VEC = dict(analyzer="word", ngram_range=(1, 2), min_df=5, sublinear_tf=True, lowercase=True,
           token_pattern=r"(?u)\S+")
APRENDIZ_STR = "LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs', tol=1e-3)"
PATRONES_COMPLIANCE = ((r"read_csv\([^)]*age_test|load_test\(", "lectura de test"),
                       (r"import\s+transformers", "transformers"),
                       (r"sentence_transformers|SentenceTransformer", "embeddings"),
                       (r"openai\.|anthropic\.|from\s+llama", "LLM"),
                       (r"requests\.(get|post)|urllib\.request", "red"))


def sha256(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def normaliza(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def normaliza_l2(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0)
    norma = np.linalg.norm(p, axis=1, keepdims=True)
    return p / np.where(norma > 1e-12, norma, 1.0)


def macro(clase: np.ndarray, pred: np.ndarray) -> float:
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


def canal_oof(secuencias: np.ndarray, clase: np.ndarray, fold_idx: np.ndarray,
              etiqueta: str) -> tuple[np.ndarray, dict]:
    p = np.zeros((len(clase), N_CLASES), dtype=np.float64)
    vocab, segundos, iters, warns = [], [], [], 0
    for k in range(5):
        t0 = time.time()
        tr = fold_idx != k
        va = fold_idx == k
        vec = TfidfVectorizer(**VEC)
        x_tr = vec.fit_transform(secuencias[tr])
        x_va = vec.transform(secuencias[va])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mod = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", tol=1e-3)
            mod.fit(x_tr, clase[tr])
            warns += sum(1 for x in w if "Convergence" in str(x.message))
        p[va] = mod.predict_proba(x_va)
        vocab.append(int(x_tr.shape[1]))
        iters.append(int(mod.n_iter_[0]))
        segundos.append(round(time.time() - t0, 1))
        print(f"    [{etiqueta}] fold {k}: {x_tr.shape[1]:,} features | {segundos[-1]} s | "
              f"n_iter {iters[-1]}", flush=True)
    vacias = np.flatnonzero(np.array([not str(s).strip() for s in secuencias]))
    if len(vacias):
        p[vacias] = 1.0 / N_CLASES
    p = normaliza(p)
    return p, {"vocabularios": vocab, "segundos_por_fold": segundos, "n_iter_por_fold": iters,
               "warnings_convergencia": warns, "filas_vacias": int(len(vacias))}


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
    folds = [macro(clase[fold_idx == k], pred[fold_idx == k]) for k in range(5)]
    folds_base = [macro(clase[fold_idx == k], champ_pred[fold_idx == k]) for k in range(5)]
    ic = bootstrap_delta(clase, champ_pred, pred)
    return {"corrected_cv": round(cv, 7), "delta_vs_baseline": round(cv - baseline, 7),
            "delta_vs_nested": round(cv - 0.6154643, 7),
            "folds": [round(v, 6) for v in folds],
            "folds_positivos": int(sum(1 for a, b in zip(folds, folds_base) if a > b)),
            "filas_que_cambian": int((pred != champ_pred).sum()),
            "ic95": ic["ic95"], "p_delta_positivo": ic["p_delta_positivo"],
            "argmax_sin_regla": round(macro(clase, np.argmax(p_comb, axis=1)), 7),
            "_pred": pred}


def complementariedad(familia: str, clave: str, pred: np.ndarray, clase: np.ndarray,
                       champ_pred: np.ndarray, conf: np.ndarray, filas: list[dict]) -> None:
    ok_c = pred == clase
    ok_ch = champ_pred == clase
    alta = (~ok_ch) & (conf >= UMBRAL_ALTA_CONFIANZA)
    filas.append({
        "familia": familia, "anotador": clave, "clase": "TODAS", "filas": int(len(clase)),
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
        filas.append({"familia": familia, "anotador": clave, "clase": nombre,
                      "filas": int(mascara.sum()), "campeon_ok_cand_no": danio,
                      "campeon_no_cand_ok": rescate,
                      "ambos_ok": int((ok_ch & ok_c & mascara).sum()),
                      "ambos_no": int((~ok_ch & ~ok_c & mascara).sum()),
                      "rescate_unico": rescate, "danio": danio, "neto": rescate - danio,
                      "rescates_alta_confianza": "", "errores_alta_confianza_campeon": ""})


def auditoria_compliance() -> dict:
    """Escaneo por operacion; clasifica hallazgos de patrones embebidos (auto-coincidencias)."""
    propio = Path(__file__).name
    reales, patrones = [], []
    for ruta in sorted((ROOT / "scripts").glob("age_linguistic_*.py")):
        if ruta.name == propio:
            continue
        for i, linea in enumerate(ruta.read_text(encoding="utf-8").splitlines(), 1):
            for patron, nombre in PATRONES_COMPLIANCE:
                if re.search(patron, linea):
                    registro = {"script": ruta.name, "linea": i, "hallazgo": nombre}
                    if "r\"" in linea or "r'" in linea:
                        patrones.append(registro)
                    else:
                        reales.append(registro)
    return {"reales": reales, "auto_coincidencias_patron": patrones,
            "limpio": len(reales) == 0, "auditor_excluido": propio}


def serie_previa() -> list[dict]:
    """Resumen de la serie P2/P6/P7 leido de sus artefactos (md como referencia)."""
    filas = []
    try:
        p2 = json.loads((cfg.EXPERIMENTS / "age_linguistic_03_pos_secuencia.json")
                        .read_text(encoding="utf-8"))
        filas.append({"fase": "P2 POS-secuencia",
                      "canal_solo": p2["anotadores"]["md"]["medicion1_canal_solo"]
                      ["canal_solo_corrected_cv"],
                      "delta_md": p2["anotadores"]["md"]["medicion3_integracion"]["real"]
                      ["delta_vs_baseline"],
                      "veredicto": p2["anotadores"]["md"]["veredicto"]})
    except Exception as error:  # noqa: BLE001
        filas.append({"fase": "P2 POS-secuencia", "nota": f"ver artefacto ({error})"})
    try:
        p6 = json.loads((cfg.EXPERIMENTS / "age_linguistic_06_ner_estructura.json")
                        .read_text(encoding="utf-8"))
        filas.append({"fase": "P6 NER estructural",
                      "canal_solo": p6["detalle_canal"]["N1|md"]["canal_solo"],
                      "delta_md": p6["brazos"]["N1|md"]["delta_vs_baseline"],
                      "veredicto": "CLOSED"})
    except Exception as error:  # noqa: BLE001
        filas.append({"fase": "P6 NER estructural", "nota": f"ver artefacto ({error})"})
    try:
        p7 = json.loads((cfg.EXPERIMENTS / "age_linguistic_09_texto_tipado.json")
                        .read_text(encoding="utf-8"))
        filas.append({"fase": "P7 texto tipado",
                      "canal_solo": p7["canal_solo"]["md"],
                      "delta_md": p7["brazos"]["md"]["delta_vs_baseline"],
                      "veredicto": "CLOSED"})
    except Exception as error:  # noqa: BLE001
        filas.append({"fase": "P7 texto tipado", "nota": f"ver artefacto ({error})"})
    return filas


def escribe_decision_md(lote: dict, baseline: float) -> None:
    lineas = [
        "# Ciclo final - familias restantes (P3/P4/P5): decision",
        "",
        f"**Fecha**: {FECHA} | **Pre-registro**: `reports/age_remaining_channels_preregistro.md` | "
        f"**Baseline**: {baseline:.7f} (brazo A = ablacion sin canal, D-076 cond.4)",
        "",
        "## Veredicto por familia",
        "",
        "| Familia | Anotador | canal solo | delta real | folds+ | IC95 | delta barajado | Gate |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for familia in FAMILIAS:
        for clave in ANOTADORES:
            r = lote["brazos"][(familia, clave)]
            rb = lote["brazos_barajados"][(familia, clave)]
            ic = r["ic95"]
            estado = "**PASA**" if r["pasa_gate"] else "no pasa"
            lineas.append(f"| {familia} | {clave} | {lote['canal_solo'][(familia, clave)]:.7f} | "
                          f"{r['delta_vs_baseline']:+.7f} | {r['folds_positivos']}/5 | "
                          f"[{ic[0]:+.7f}; {ic[1]:+.7f}] | {rb['delta_vs_baseline']:+.7f} | "
                          f"{estado} |")
    lineas += [
        "",
        f"**Resultado global: {'PROMISING (hay brazo que pasa)' if lote['promising'] else 'NINGUN BRAZO PASA'}**",
        "",
        "## Lectura",
        "",
        "- La comparacion decisiva es real vs barajado por el mismo pipeline completo.",
    ]
    for familia in FAMILIAS:
        for clave in ANOTADORES:
            r = lote["brazos"][(familia, clave)]
            det = lote["detalle_canal"][(familia, clave)]
            lineas.append(
                f"- **{familia}/{clave}**: delta {r['delta_vs_baseline']:+.7f} vs barajado "
                f"{lote['brazos_barajados'][(familia, clave)]['delta_vs_baseline']:+.7f}; peso "
                f"del bloque canal {lote['combinador'][(familia, clave)]['peso_bloque_canal_ratio_media']}; "
                f"vocab medio {round(float(np.mean(det['vocabularios']))):,}; vacios "
                f"{det['filas_vacias']}; n_iter medio "
                f"{round(float(np.mean(det['n_iter_por_fold'])), 1)}; warnings "
                f"{det['warnings_convergencia']}.")
    lineas += [
        "",
        "## Serie completa de canales linguisticos estructurados (referencia md)",
        "",
        "| Fase | Canal solo | delta (md) | Veredicto |",
        "|---|---|---|---|",
    ]
    for fila in lote["serie"]:
        if "nota" in fila:
            lineas.append(f"| {fila['fase']} | - | - | {fila['nota']} |")
        else:
            lineas.append(f"| {fila['fase']} | {fila['canal_solo']} | {fila['delta_md']:+.7f} | "
                          f"{fila['veredicto']} |")
    for familia in FAMILIAS:
        r = lote["brazos"][(familia, "md")]
        lineas.append(f"| {familia} ({FAMILIAS[familia]['nombre']}) | "
                      f"{lote['canal_solo'][(familia, 'md')]:.7f} | "
                      f"{r['delta_vs_baseline']:+.7f} | "
                      f"{'PROMISING' if r['pasa_gate'] else ('MARGINAL' if r['delta_vs_baseline'] > 0 else 'CLOSED')} |")
    lineas += [
        "",
        "## Complementariedad (resumen TODAS)",
        "",
        "| Familia | Anotador | rescate | danio | neto | alta confianza |",
        "|---|---|---|---|---|---|",
    ]
    for fila in lote["filas_comp"]:
        if fila["clase"] == "TODAS":
            lineas.append(f"| {fila['familia']} | {fila['anotador']} | {fila['rescate_unico']:,} | "
                          f"{fila['danio']:,} | {fila['neto']:+,} | "
                          f"{fila['rescates_alta_confianza']:,} / "
                          f"{fila['errores_alta_confianza_campeon']:,} |")
    lineas += [
        "",
        "## Compliance (registrado)",
        "",
        f"- Auditoria por operacion: {lote['compliance']}",
        "- test = 0; holdout = 0; leaderboard = 0; submissions = 0; modelos nuevos = 0; "
        "licencias nuevas = 0; descargas = 0.",
        "",
        "## Provenance",
        "",
        "Hashes de anotacion, dep_edges y particion canonica verificados al arrancar",
        "(`hashes_ok`); validacion del artefacto derivado re-hecha en el runner (0 discrepancias",
        "exigidas). Semillas: bootstrap 20261010, barajado intra-fold 20261031.",
        "",
        "## Limites declarados",
        "",
        "- Control intra-fold (como P7): sus deltas barajados no son comparables numericamente",
        "  con los globales de P2/P6.",
        "- R5 pierde el POS del dependiente y sus bigramas son adyacencia lineal, no caminos del",
        "  arbol (decision fija pre-registrada).",
        "- El claim es empirico y acotado: 'el canal X no aporta mejora al campeon bajo el",
        "  mecanismo evaluado'; no se extrapola a toda la informacion linguistica.",
    ]
    texto = "\n".join(l.rstrip() for l in lineas) + "\n"
    DECISION_MD.write_text(texto, encoding="utf-8")


def d_080(lote: dict, baseline: float) -> str:
    def resumen_familia(familia: str) -> str:
        partes = []
        for clave in ANOTADORES:
            r = lote["brazos"][(familia, clave)]
            partes.append(f"{clave} {r['delta_vs_baseline']:+.7f} ({r['folds_positivos']}/5)")
        return " | ".join(partes)
    lineas = ["---", "",
              "## D-080 - Ciclo final (P3/P4/P5): las familias restantes quedan medidas y cerradas"
              if not lote["promising"] else
              "## D-080 - Ciclo final (P3/P4/P5): hay un brazo que pasa el gate (PROMISING)", ""]
    lineas += [
        f"- **Observation**: el usuario ordeno ejecutar las experimentaciones restantes; se "
        f"ejecutaron las tres familias con el protocolo congelado de P6/P7 (baseline "
        f"{baseline:.7f}; control intra-fold 20261031; artefacto dep_edges derivado y re-validado "
        f"0/0). Resultados (md/lg):",
        f"  - R3 lemma_POS: {resumen_familia('R3_lemma_POS')};",
        f"  - R4 morfologia: {resumen_familia('R4_morfologia')};",
        f"  - R5 dependencias: {resumen_familia('R5_dependencias')}.",
        f"- **Result**: {'algun brazo pasa: segunda particion congelada como siguiente paso.' if lote['promising'] else 'ningun brazo pasa el gate: las tres familias se cierran documentadas.'}",
        "- **Decision**: con esto quedan MEDIDAS las seis representaciones de la fase (P2, P6, P7 "
        "y las tres restantes), todas bajo gates identicos y con control propio. "
        + ("El siguiente paso obligatorio es la segunda particion del brazo que pase; sin "
           "submission sin autorizacion (D-076 cond.5)." if lote["promising"] else
           "El campeon adoptado (V30 + plug-in, LB 0.63575) permanece intacto; no se reabre "
           "ninguna familia; sin encadenar variantes."),
        "- **Consequences**: el expediente linguistico-estructurado de AGE queda cerrado con "
        "evidencia propia para cada familia; la regla de interpretacion pre-registrada se aplica "
        "por canal ('no aporta mejora bajo el mecanismo evaluado'), sin extrapolar a la "
        "informacion linguistica en general.",
        "- **Evidence**: `experiments/age_linguistic_12_remaining_channels.json`, "
        "`reports/age_remaining_results.csv`, `reports/age_remaining_complementarity.csv`, "
        "`reports/age_remaining_decision.md`.",
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

    veredictos = {f"{f}/{c}": ("PASA" if lote["brazos"][(f, c)]["pasa_gate"] else
                               ("MARGINAL" if lote["brazos"][(f, c)]["delta_vs_baseline"] > 0
                                else "CLOSED"))
                  for f in FAMILIAS for c in ANOTADORES}
    md(f"""## PHASE - REMAINING STRUCTURED CHANNELS

Orden del usuario: ejecutar las familias restantes de la serie con el mismo protocolo congelado
de P6/P7 (baseline {baseline:.7f}; control intra-fold; regla y pesos del plug-in congelados).
Registro: P3/P4 estaban PARTIALLY_COVERED y P5 sin justificacion independiente; se ejecutan como
override explicito del usuario para cerrar el expediente con evidencia propia.

- R3 lemma_POS (canal aislado; el bloque B de P7 sin el texto) -> {veredictos['R3_lemma_POS/md']}/{veredictos['R3_lemma_POS/lg']} (md/lg).
- R4 morfologia (sin tokens '_'; vacios -> uniforme 1/6) -> {veredictos['R4_morfologia/md']}/{veredictos['R4_morfologia/lg']}.
- R5 dependencias (dep->POS_cabeza; artefacto derivado re-validado 0/0) -> {veredictos['R5_dependencias/md']}/{veredictos['R5_dependencias/lg']}.

**Resultado global: {'PROMISING' if lote['promising'] else 'NINGUN BRAZO PASA'}.**""")

    code("""import json, pathlib
import pandas as pd
RAIZ = pathlib.Path("..")
res = json.loads((RAIZ / "experiments" / "age_linguistic_12_remaining_channels.json").read_text(encoding="utf-8"))
print("verificacion:", res["verificacion"], "| baseline:", res["baseline_primario"])
tabla = pd.read_csv(RAIZ / "reports" / "age_remaining_results.csv")
print()
print(tabla.to_string(index=False))
comp = pd.read_csv(RAIZ / "reports" / "age_remaining_complementarity.csv")
todas = comp[comp["clase"] == "TODAS"]
print()
print(todas[["familia","anotador","rescate_unico","danio","neto","rescates_alta_confianza"]].to_string(index=False))
assert res["verificacion"]["hashes_ok"] is True
assert res["verificacion"]["constantes_ok"] is True
assert res["verificacion"]["dep_edges_ok"] is True
assert abs(res["baseline_primario"] - 0.6159732) < 1e-6
assert res["sin_test"] is True and res["sin_submission"] is True
print()
print("[OK] ciclo final ejecutado contra el pre-registro congelado; sin tocar test")""")
    md("""### Serie completa y decision

La tabla comparativa de las seis representaciones (P2, P6, P7, R3, R4, R5) vive en
`reports/age_remaining_decision.md`. La decision sigue el arbol pre-registrado; el campeon
(V30 + plug-in, LB 0,63575) permanece intacto y no se reabre ninguna familia.""")
    nbformat.write(nb, NOTEBOOK)
    return {"antes": antes, "despues": len(nb.cells), "anadidas": len(nb.cells) - antes}


def main() -> int:
    autotest_numerico()
    inicio = time.time()
    prereg = json.loads(PREREG_JSON.read_text(encoding="utf-8"))

    print("=== verificacion contra el pre-registro ===", flush=True)
    constantes_ok = (prereg["gates"]["delta_min"] == GATE_DELTA
                     and prereg["gates"]["folds_min"] == GATE_FOLDS
                     and prereg["semillas"] == {"bootstrap": SEMILLA_BOOTSTRAP,
                                                "barajado": SEMILLA_BARAJADO}
                     and prereg["aprendiz"] == APRENDIZ_STR
                     and all(prereg["representaciones"][f]["vec"] in ("identico a R3", VEC_STR)
                             for f in prereg["representaciones"]))
    rutas_hashes = {
        "anotacion_md_parquet": ANOT_DIR / "anotacion_md_train.parquet",
        "anotacion_lg_parquet": ANOT_DIR / "anotacion_lg_train.parquet",
        "dep_edges_md_parquet": ANOT_DIR / "dep_edges_md_train.parquet",
        "dep_edges_lg_parquet": ANOT_DIR / "dep_edges_lg_train.parquet",
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

    # re-validacion del artefacto derivado (requisito del revisor ML)
    dep_edges_ok = True
    for clave in ANOTADORES:
        almacen = pd.read_parquet(ANOT_DIR / f"anotacion_{clave}_train.parquet",
                                  columns=["dep_seq", "n_tokens"])
        borde = pd.read_parquet(ANOT_DIR / f"dep_edges_{clave}_train.parquet",
                                columns=["dep_edge_seq", "n_tokens"])
        if not np.array_equal(almacen["n_tokens"].to_numpy(), borde["n_tokens"].to_numpy()):
            dep_edges_ok = False
        distintos = sum(1 for guardado, derivado in
                        zip(almacen["dep_seq"].astype(str), borde["dep_edge_seq"].astype(str))
                        if guardado.split() != [t.split("->")[0] for t in derivado.split()])
        if distintos:
            dep_edges_ok = False
            print(f"  [STOP] dep_edges {clave}: {distintos} documentos con dep distinta", flush=True)
    print(f"  dep_edges re-validado: {dep_edges_ok}", flush=True)
    if not dep_edges_ok:
        print("[STOP] el artefacto derivado no reproduce lo almacenado", flush=True)
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

    # secuencias por anotador y familia
    canales = {}
    for clave in ANOTADORES:
        an = pd.read_parquet(ANOT_DIR / f"anotacion_{clave}_train.parquet",
                             columns=["lemma_pos_seq", "morph_seq"])
        dep = pd.read_parquet(ANOT_DIR / f"dep_edges_{clave}_train.parquet",
                              columns=["dep_edge_seq"])
        canales[(clave, "R3_lemma_POS")] = an["lemma_pos_seq"].astype(str).to_numpy()
        canales[(clave, "R4_morfologia")] = an["morph_seq"].astype(str).apply(
            lambda b: " ".join(t for t in b.split() if t != "_")).to_numpy()
        canales[(clave, "R5_dependencias")] = dep["dep_edge_seq"].astype(str).to_numpy()

    rng = np.random.default_rng(SEMILLA_BARAJADO)
    perm = np.arange(len(clase))
    for k in range(5):
        idx = np.flatnonzero(fold_idx == k)
        perm[idx] = idx[rng.permutation(len(idx))]

    lote = {"brazos": {}, "brazos_barajados": {}, "detalle_canal": {}, "combinador": {},
            "canal_solo": {}, "filas_csv": [], "filas_comp": []}
    for familia in FAMILIAS:
        for clave in ANOTADORES:
            etiqueta = f"{familia}-{clave}"
            print(f"\n=== {etiqueta} (real) ===", flush=True)
            sec = canales[(clave, familia)]
            p_c, det = canal_oof(sec, clase, fold_idx, etiqueta)
            diagonal = [int(np.argmax(p_c[clase == i].mean(axis=0))) == i for i in range(N_CLASES)]
            if not all(diagonal):
                print(f"[STOP] orden de clases roto en {etiqueta}: {diagonal}", flush=True)
                return 1
            p_comb, comb = combinador_oof(v30, p_c, clase, fold_idx)
            r = evalua(p_comb, clase, fold_idx, champ_pred, pesos, baseline)
            solo = macro(clase, np.argmax(p_c, axis=1))

            print(f"=== {etiqueta} (barajado) ===", flush=True)
            p_cb, det_b = canal_oof(sec[perm], clase, fold_idx, f"{etiqueta}-barajado")
            p_comb_b, comb_b = combinador_oof(v30, p_cb, clase, fold_idx)
            r_b = evalua(p_comb_b, clase, fold_idx, champ_pred, pesos, baseline)

            pasa = (r["delta_vs_baseline"] >= GATE_DELTA and r["folds_positivos"] >= GATE_FOLDS
                    and r["ic95"][0] > 0 and r["delta_vs_baseline"] > r_b["delta_vs_baseline"])
            r["pasa_gate"] = bool(pasa)
            print(f"  {etiqueta}: canal solo {solo:.7f} | delta {r['delta_vs_baseline']:+.7f} "
                  f"({r['folds_positivos']}/5) | barajado {r_b['delta_vs_baseline']:+.7f} -> "
                  f"{'PASA' if pasa else 'no pasa'}", flush=True)

            pred_cand = r.pop("_pred")
            r_b.pop("_pred", None)
            lote["brazos"][(familia, clave)] = r
            lote["brazos_barajados"][(familia, clave)] = r_b
            lote["detalle_canal"][(familia, clave)] = det
            lote["combinador"][(familia, clave)] = comb
            lote["canal_solo"][(familia, clave)] = round(solo, 7)
            complementariedad(familia, clave, pred_cand, clase, champ_pred, conf, lote["filas_comp"])
            for nombre, metricas, condicion in (("real", r, "real"),
                                                ("barajado", r_b, "barajado")):
                detalle = det if nombre == "real" else det_b
                combo = comb if nombre == "real" else comb_b
                lote["filas_csv"].append({
                    "familia": familia, "anotador": clave, "brazo": condicion,
                    "corrected_cv": metricas["corrected_cv"],
                    "delta_vs_baseline": metricas["delta_vs_baseline"],
                    "delta_vs_nested": metricas["delta_vs_nested"],
                    "folds_positivos": metricas["folds_positivos"],
                    "folds": ";".join(f"{v:.6f}" for v in metricas["folds"]),
                    "ic95_lo": metricas["ic95"][0], "ic95_hi": metricas["ic95"][1],
                    "filas_que_cambian": metricas["filas_que_cambian"],
                    "vocab_medio": round(float(np.mean(detalle["vocabularios"])), 0),
                    "segundos_canal": sum(detalle["segundos_por_fold"]),
                    "n_iter_medio": round(float(np.mean(detalle["n_iter_por_fold"])), 0),
                    "warnings": detalle["warnings_convergencia"],
                    "vacios": detalle["filas_vacias"],
                    "peso_bloque_canal_ratio": combo["peso_bloque_canal_ratio_media"],
                    "canal_solo": round(solo, 7) if nombre == "real" else "",
                    "argmax_sin_regla": metricas["argmax_sin_regla"],
                    "pasa_gate": metricas.get("pasa_gate", "")})

    lote["promising"] = any(r["pasa_gate"] for r in lote["brazos"].values())
    lote["serie"] = serie_previa()
    lote["compliance"] = auditoria_compliance()
    print(f"\n  PROMISING: {lote['promising']} | compliance limpio: {lote['compliance']['limpio']}",
          flush=True)

    cols = ["familia", "anotador", "brazo", "corrected_cv", "delta_vs_baseline",
            "delta_vs_nested", "folds_positivos", "folds", "ic95_lo", "ic95_hi",
            "filas_que_cambian", "vocab_medio", "segundos_canal", "n_iter_medio", "warnings",
            "vacios", "peso_bloque_canal_ratio", "canal_solo", "argmax_sin_regla", "pasa_gate"]
    with RESULTADOS_CSV.open("w", encoding="utf-8", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        escritor.writeheader()
        for fila in lote["filas_csv"]:
            escritor.writerow({c: fila.get(c, "") for c in cols})
    cols_c = ["familia", "anotador", "clase", "filas", "campeon_ok_cand_no", "campeon_no_cand_ok",
              "ambos_ok", "ambos_no", "rescate_unico", "danio", "neto",
              "rescates_alta_confianza", "errores_alta_confianza_campeon"]
    with COMPL_CSV.open("w", encoding="utf-8", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=cols_c, lineterminator="\n")
        escritor.writeheader()
        for fila in lote["filas_comp"]:
            escritor.writerow({c: fila.get(c, "") for c in cols_c})

    escribe_decision_md(lote, baseline)
    RESULTADO.write_text(json.dumps({
        "fase": "ciclo final P3/P4/P5", "fecha": FECHA,
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT.parent,
                                 capture_output=True, text=True, timeout=30).stdout.strip(),
        "verificacion": {"hashes_ok": hashes_ok, "constantes_ok": constantes_ok,
                         "dep_edges_ok": dep_edges_ok},
        "baseline_primario": round(baseline, 7),
        "brazos": {"|".join(k): v for k, v in lote["brazos"].items()},
        "brazos_barajados": {"|".join(k): v for k, v in lote["brazos_barajados"].items()},
        "detalle_canal": {"|".join(k): v for k, v in lote["detalle_canal"].items()},
        "combinador": {"|".join(k): v for k, v in lote["combinador"].items()},
        "canal_solo": {"|".join(k): v for k, v in lote["canal_solo"].items()},
        "serie": lote["serie"], "promising": lote["promising"],
        "compliance": lote["compliance"],
        "sin_test": True, "sin_holdout": True, "sin_submission": True,
        "segundos_total": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("  experiments/age_linguistic_12_remaining_channels.json", flush=True)

    print("\n=== documentacion ===", flush=True)
    print("  " + anade(ROOT / "DECISIONS.md", "## D-080", d_080(lote, baseline)), flush=True)
    print(f"  {DECISION_MD.name} | {RESULTADOS_CSV.name} | {COMPL_CSV.name}", flush=True)

    runlog = ROOT / "RUNLOG.md"
    if "Ciclo final ejecutado (P3/P4/P5)" not in runlog.read_text(encoding="utf-8"):
        resumen = " | ".join(f"{f}/{c}: {lote['brazos'][(f, c)]['delta_vs_baseline']:+.7f}"
                             for f in FAMILIAS for c in ANOTADORES)
        estado = "PROMISING" if lote["promising"] else "ningun brazo pasa"
        bloque = ("\n---\n\n## Ciclo final ejecutado (P3/P4/P5) · " + FECHA + "\n\n"
                  f"- Resultado: {estado}. {resumen}.\n"
                  f"- Baseline {baseline:.7f}; control intra-fold 20261031; dep_edges re-validado."
                  "\n- Sin test, sin holdout, sin submission; campeon intacto.\n")
        with runlog.open("a", encoding="utf-8") as f:
            f.write(bloque)
        print("  RUNLOG.md: ciclo anotado", flush=True)

    ruta_exp = ROOT / "experiments.csv"
    lineas_exp = ruta_exp.read_text(encoding="utf-8").splitlines()
    cols_exp = [c.strip() for c in lineas_exp[0].split(",")]
    filas_nuevas = []
    for familia, nombre in (("R3_lemma_POS", "P3-LEMMA-POS"), ("R4_morfologia", "P4-MORFOLOGIA"),
                            ("R5_dependencias", "P5-DEPENDENCIAS")):
        eid = f"AGE-LING-{nombre}"
        if eid in {l.split(",", 1)[0] for l in lineas_exp[1:] if l}:
            continue
        resumen = " | ".join(f"{c}: delta {lote['brazos'][(familia, c)]['delta_vs_baseline']:+.7f} "
                             f"({lote['brazos'][(familia, c)]['folds_positivos']}/5)"
                             for c in ANOTADORES)
        reg = {"experiment_id": eid, "fecha": FECHA, "bloque": "30.5",
               "etapa": f"ciclo final: {nombre} contra el campeon",
               "pregunta": "el canal aporta al campeon bajo el mecanismo evaluado?",
               "hipotesis": f"{nombre} como canal tipado",
               "features": f"{familia} -> TF-IDF 1-2 + LogReg por fold",
               "preprocessing": "columnas congeladas (R4 sin '_'; R5 artefacto derivado)",
               "representacion": f"canal sparse {familia}",
               "modelo": "LogReg + combinador anidado + regla congelada",
               "hiperparametros": "congelados por AGE-LING-REMAINING-PREREG",
               "validacion": "particion canonica; baseline 0.6159732; control intra-fold "
                             "(20261031); gates +0.0015/4-5/real>barajado/bootstrap",
               "macro_f1_mean": "", "macro_f1_std": "", "weighted_f1_mean": "",
               "micro_f1_mean": "", "accuracy_mean": "",
               "resultado": resumen,
               "interpretacion": "lectura por el arbol pre-registrado",
               "decision": "cierre documentado de la familia" if not lote["promising"] else
                           "segunda particion del brazo que pasa",
               "duracion_seg": f"{round(time.time() - inicio, 1)}",
               "artefactos": "reports/age_remaining_decision.md | "
                             "experiments/age_linguistic_12_remaining_channels.json"}
        filas_nuevas.append(reg)
    if filas_nuevas:
        with ruta_exp.open("a", encoding="utf-8") as f:
            for reg in filas_nuevas:
                f.write(",".join(f'"{reg[c]}"' if ("," in str(reg[c]) or '"' in str(reg[c]))
                                 else str(reg[c]) for c in cols_exp) + "\n")
        print(f"  experiments.csv: {len(filas_nuevas)} filas", flush=True)

    estado = "PROMISING" if lote["promising"] else "ningun brazo pasa"
    bloque_estado = ("\n---\n\n## Ciclo final ejecutado: P3/P4/P5 (" + FECHA + ")\n\n"
                     f"- Resultado: {estado}; serie completa de seis representaciones medida.\n"
                     f"- Baseline {baseline:.7f}; sin test, sin holdout, sin submission; "
                     "campeon intacto.\n")
    print("  " + anade(ROOT / "PROJECT_STATE.md", "## Ciclo final ejecutado: P3/P4/P5",
                       bloque_estado), flush=True)
    print("  " + anade(ROOT / "FINAL_REPORT.md", "## Ciclo final ejecutado: P3/P4/P5",
                       bloque_estado), flush=True)
    print("  " + str(anade_notebook(lote, baseline)), flush=True)

    print(f"\n[fin] {time.time() - inicio:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
