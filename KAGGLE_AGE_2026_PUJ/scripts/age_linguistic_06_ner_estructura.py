"""P6 - canal NER estructural: ejecucion del pre-registro congelado.

## Orden de ejecucion

1. Verifica que nada ha cambiado: recomputa los SHA256 de todos los artefactos fuente y compara
   contra `experiments/age_linguistic_05_preregistro.json`; compara representaciones, gates,
   semillas y pesos. Si algo no cuadra, para.
2. Construye las secuencias de TIPOS de entidad desde los parquets congelados (ent_seq + pos_seq):
   N1 = tipos por span; N1b = N1 sin spans sobre menciones/RT/hashtags/URLs; N2 = tipo + POS del
   contexto inmediato.
3. Por anotador (md, lg) y representacion: canal OOF por fold (TF-IDF + LogReg C=1.0) y su
   version BARAJADA (permutacion entre tweets, semilla 20261029) que atraviesa el mismo pipeline.
4. Integracion unica: combinador logistico anidado sobre [p_V30 || p_canal] normalizadas por
   fila; decision = argmax(w_plug-in * p_combinada) con regla y pesos CONGELADOS.
5. Metricas: canal solo, delta, folds, bootstrap pareado, complementariedad (global y por clase),
   coeficientes del combinador, dimensiones y tiempos; gates y arbol de decision pre-registrados.

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
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import data as datos  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

FECHA = "2026-09-29"
N_CLASES = 6
PREREG_JSON = cfg.EXPERIMENTS / "age_linguistic_05_preregistro.json"
CANONICO = ROOT / "artifacts" / "p33_scores" / "todos"
PARQUETS = {"md": ROOT / "features" / "linguistic" / "anotacion_md_train.parquet",
            "lg": ROOT / "features" / "linguistic" / "anotacion_lg_train.parquet"}
RESULTADOS_CSV = cfg.REPORTS / "ner_structural_results.csv"
COMPL_CSV = cfg.REPORTS / "ner_structural_complementarity.csv"
DECISION_MD = cfg.REPORTS / "ner_structural_decision.md"
RESULTADO = cfg.EXPERIMENTS / "age_linguistic_06_ner_estructura.json"
NOTEBOOK = ROOT / "notebooks" / "99_entrega_consolidado.ipynb"
MARCADOR_NB = "PHASE - NER STRUCTURAL REPRESENTATION"

SEMILLA_BOOTSTRAP = 20261010
SEMILLA_BARAJADO = 20261029
GATE_DELTA = 0.0015
GATE_FOLDS = 4
UMBRAL_ALTA_CONFIANZA = 0.60
VARIABILIDAD = 0.0008876
N_BOOTSTRAP = 2000
ANOTADORES = ("md", "lg")
VEC_STR = {
    "N1": "TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df=5, sublinear_tf=True, "
          "lowercase=True, token_pattern=r'(?u)\\S+')",
    "N1b": "TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df=5, sublinear_tf=True, "
           "lowercase=True, token_pattern=r'(?u)\\S+')",
    "N2": "TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df=5, max_features=2000, "
          "sublinear_tf=True, lowercase=True, token_pattern=r'(?u)\\S+')",
}
VEC_KWARGS = {
    "N1": dict(analyzer="word", ngram_range=(1, 3), min_df=5, sublinear_tf=True, lowercase=True,
               token_pattern=r"(?u)\S+"),
    "N1b": dict(analyzer="word", ngram_range=(1, 3), min_df=5, sublinear_tf=True, lowercase=True,
                token_pattern=r"(?u)\S+"),
    "N2": dict(analyzer="word", ngram_range=(1, 2), min_df=5, max_features=2000, sublinear_tf=True,
               lowercase=True, token_pattern=r"(?u)\S+"),
}
TOKEN_RE = re.compile(r"@\w+|#\w+|https?://\S+|www\.\S+|\d+|[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+|[^\w\s]")


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


def es_marca(token: str) -> bool:
    t = token.lower()
    return (token.startswith("@") or token.startswith("#")
            or t.startswith("http") or t.startswith("www") or t == "rt")


def construir_secuencias(ent_seq: np.ndarray, pos_seq: np.ndarray,
                         textos: list[str]) -> dict[str, np.ndarray]:
    """N1, N1b y N2 desde las columnas congeladas; spans = runs de la misma etiqueta."""
    n = len(ent_seq)
    n1 = np.empty(n, dtype=object)
    n1b = np.empty(n, dtype=object)
    n2 = np.empty(n, dtype=object)
    for i in range(n):
        etiquetas = ent_seq[i].split() if ent_seq[i] else []
        pos = pos_seq[i].split() if pos_seq[i] else []
        spans = []
        run = None
        inicio = 0
        for j, t in enumerate(etiquetas):
            if t != "O":
                if t != run:
                    if run is not None:
                        spans.append((inicio, j, run))
                    run, inicio = t, j
            else:
                if run is not None:
                    spans.append((inicio, j, run))
                    run = None
        if run is not None:
            spans.append((inicio, len(etiquetas), run))
        n1[i] = " ".join(f"ENT_{tp}" for _, _, tp in spans)
        superficie = TOKEN_RE.findall(textos[i])
        spans_b = [s for s in spans
                   if not any(es_marca(superficie[j]) for j in range(s[0], min(s[1], len(superficie))))]
        n1b[i] = " ".join(f"ENT_{tp}" for _, _, tp in spans_b)
        partes = []
        for a, b, tp in spans:
            pos_izq = pos[a - 1] if a > 0 and a - 1 < len(pos) else "START"
            pos_der = pos[b] if b < len(pos) else "END"
            partes.append(f"ENT_{tp}")
            partes.append(f"{tp}_L_{pos_izq}")
            partes.append(f"{tp}_R_{pos_der}")
        n2[i] = " ".join(partes)
    return {"N1": n1, "N1b": n1b, "N2": n2}


def canal_oof(secuencias: np.ndarray, clase: np.ndarray, fold_idx: np.ndarray,
              vec_kwargs: dict, etiqueta: str,
              top_features: bool = False) -> tuple[np.ndarray, dict]:
    """OOF del canal; filas con secuencia vacia -> uniforme 1/6 (regla pre-registrada)."""
    p = np.zeros((len(secuencias), N_CLASES), dtype=np.float64)
    vocab, segundos = [], []
    features = {}
    for k in range(5):
        t0 = time.time()
        tr = fold_idx != k
        va = fold_idx == k
        vec = TfidfVectorizer(**vec_kwargs)
        x_tr = vec.fit_transform(secuencias[tr])
        x_va = vec.transform(secuencias[va])
        modelo = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        modelo.fit(x_tr, clase[tr])
        p[va] = modelo.predict_proba(x_va)
        vocab.append(int(len(vec.vocabulary_)))
        segundos.append(round(time.time() - t0, 1))
        if top_features:
            nombres = np.asarray(vec.get_feature_names_out())
            peso = np.abs(modelo.coef_).mean(axis=0)
            for nombre, valor in zip(nombres, peso):
                features[nombre] = features.get(nombre, 0.0) + float(valor)
        print(f"    [{etiqueta}] fold {k}: {x_tr.shape[1]} features | {segundos[-1]} s", flush=True)
    vacias = np.flatnonzero(np.array([not str(s).strip() for s in secuencias]))
    if len(vacias):
        p[vacias] = 1.0 / N_CLASES
    p = normaliza(p)
    info = {"vocabularios": vocab, "segundos_por_fold": segundos,
            "filas_vacias": int(len(vacias)),
            "pct_no_vacias": round(100.0 * (len(secuencias) - len(vacias)) / len(secuencias), 2)}
    if top_features:
        orden = sorted(features.items(), key=lambda kv: -kv[1])[:15]
        info["top_features"] = [{ "feature": f, "peso_medio_abs": round(v, 4)} for f, v in orden]
    return p, info


def combinador_oof(v30: np.ndarray, p_canal: np.ndarray, clase: np.ndarray,
                   fold_idx: np.ndarray, etiqueta: str) -> tuple[np.ndarray, dict]:
    """Combinador anidado sobre [p_V30 || p_canal] L2-normalizadas por fila (12 features)."""
    features = np.hstack([normaliza_l2(v30), normaliza_l2(p_canal)])
    p = np.zeros((len(clase), N_CLASES), dtype=np.float64)
    razones = []
    for k in range(5):
        tr = fold_idx != k
        va = fold_idx == k
        modelo = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        modelo.fit(features[tr], clase[tr])
        p[va] = modelo.predict_proba(features[va])
        coef = np.abs(modelo.coef_)
        razones.append(round(float(coef[:, N_CLASES:].mean()
                                   / max(coef[:, :N_CLASES].mean(), 1e-12)), 4))
    p = normaliza(p)
    return p, {"peso_bloque_canal_ratio_por_fold": razones,
               "peso_bloque_canal_ratio_media": round(float(np.mean(razones)), 4)}


def evalua(p_comb: np.ndarray, clase: np.ndarray, fold_idx: np.ndarray, v30: np.ndarray,
           champ_pred: np.ndarray, pesos: np.ndarray, baseline: float) -> dict:
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


def complementariedad(clave: str, representacion: str, pred: np.ndarray, clase: np.ndarray,
                       champ_pred: np.ndarray, conf: np.ndarray, filas: list[dict]) -> None:
    ok_c = pred == clase
    ok_ch = champ_pred == clase
    alta = (~ok_ch) & (conf >= UMBRAL_ALTA_CONFIANZA)
    filas.append({
        "representacion": representacion, "anotador": clave, "clase": "TODAS",
        "filas": int(len(clase)),
        "rescate_unico": int((~ok_ch & ok_c).sum()),
        "rescate_compartido": int((ok_ch & ok_c).sum()),
        "danio": int((ok_ch & ~ok_c).sum()),
        "neto": int((~ok_ch & ok_c).sum()) - int((ok_ch & ~ok_c).sum()),
        "rescates_alta_confianza": int((alta & ok_c).sum()),
        "errores_alta_confianza_campeon": int(alta.sum())})
    for c, nombre in enumerate(cfg.CLASSES):
        mascara = clase == c
        rescate = int((~ok_ch & ok_c & mascara).sum())
        danio = int((ok_ch & ~ok_c & mascara).sum())
        filas.append({"representacion": representacion, "anotador": clave, "clase": nombre,
                      "filas": int(mascara.sum()), "rescate_unico": rescate,
                      "rescate_compartido": int((ok_ch & ok_c & mascara).sum()), "danio": danio,
                      "neto": rescate - danio, "rescates_alta_confianza": "",
                      "errores_alta_confianza_campeon": ""})


def auditoria_compliance() -> dict:
    """Verificaciones negativas sobre los scripts de la fase (AST/grep dirigido)."""
    problemas = []
    for ruta in sorted((ROOT / "scripts").glob("age_linguistic_*.py")):
        texto = ruta.read_text(encoding="utf-8")
        for patron, nombre in ((r"age_test", "lectura de test"),
                               (r"import\s+transformers", "transformers"),
                               (r"sentence_transformers|SentenceTransformer", "embeddings"),
                               (r"openai|anthropic|from\s+llama", "LLM"),
                               (r"requests\.|urllib\.request", "red")):
            if re.search(patron, texto):
                problemas.append({"script": ruta.name, "hallazgo": nombre})
    return {"hallazgos": problemas, "limpio": len(problemas) == 0}


def escribe_resultados_csv(filas: list[dict]) -> None:
    cols = ["representacion", "anotador", "brazo", "corrected_cv", "delta_vs_baseline",
            "delta_vs_nested", "folds_positivos", "folds", "ic95_lo", "ic95_hi",
            "filas_que_cambian", "vocab_medio", "segundos_canal", "pct_no_vacias",
            "peso_bloque_canal_ratio", "argmax_sin_regla", "pasa_gate"]
    with RESULTADOS_CSV.open("w", encoding="utf-8", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        escritor.writeheader()
        for fila in filas:
            escritor.writerow({c: fila.get(c, "") for c in cols})


def escribe_decision_md(lote: dict, baseline: float) -> None:
    lineas = [
        "# P6 - NER estructural: decision",
        "",
        f"**Fecha**: {FECHA} | **Pre-registro**: `reports/ner_structural_preregistro.md` | "
        f"**Baseline**: {baseline:.7f} (regla de la submission, recomputado)",
        "",
        "## Veredicto",
        "",
        "| Representacion | Anotador | delta (real) | folds+ | IC95 | delta (barajado) | Gate |",
        "|---|---|---|---|---|---|---|",
    ]
    for clave in sorted(lote["brazos"], key=lambda s: (s[0], s[1])):
        rep, anot = clave
        r = lote["brazos"][clave]
        b = lote["brazos_barajados"][clave]
        ic = r["ic95"]
        estado = "**PASA**" if r["pasa_gate"] else "no pasa"
        lineas.append(f"| {rep} | {anot} | {r['delta_vs_baseline']:+.7f} | "
                      f"{r['folds_positivos']}/5 | [{ic[0]:+.7f}; {ic[1]:+.7f}] | "
                      f"{b['delta_vs_baseline']:+.7f} | {estado} |")
    lineas += [
        "",
        f"**Resultado global**: {'PROMISING (hay brazo que pasa)' if lote['promising'] else 'NINGUN BRAZO PASA'}",
        "",
        "## Lectura",
        "",
        "- La comparacion decisiva es real vs barajado (mismo pipeline completo): si el barajado "
        "rinde igual, lo que aporta el bloque es capacidad o ruido, no informacion de NER.",
    ]
    for clave in sorted(lote["brazos"], key=lambda s: (s[0], s[1])):
        rep, anot = clave
        r = lote["brazos"][clave]
        b = lote["brazos_barajados"][clave]
        detalle = lote["detalle_canal"][clave]
        comp_ratio = lote["combinador"][clave]["peso_bloque_canal_ratio_media"]
        lineas.append(f"- **{rep}/{anot}**: canal solo {detalle['canal_solo']:.4f}; "
                      f"delta {r['delta_vs_baseline']:+.7f} vs barajado "
                      f"{b['delta_vs_baseline']:+.7f}; ratio de peso del bloque canal en el "
                      f"combinador {comp_ratio} (cerca de 0 = recalibracion de V30).")
    lineas += [
        "",
        "## Atribucion por representacion",
        "",
        "- N1b aisla la parte lexica (sin marcas de Twitter): comparar N1 vs N1b separa 'estructura "
        "de entidades lexicas' de 'proxy de menciones/RT/hashtags/URLs' (canales V31 cerrados).",
        "",
        "## Compliance (registrado, no declarado)",
        "",
        "| Restriccion | Estado |",
        "|---|---|",
        "| lectura de test / labels de test | 0 |",
        "| entrenamiento o calibracion con test | 0 |",
        "| holdout | 0 |",
        "| leaderboard (tuning) | 0 |",
        "| embeddings / transformers / LLM | 0 |",
        "| datos o etiquetas externas | 0 |",
        f"| auditoria automatica de scripts de la fase | {lote['compliance']} |",
        "",
        "Nota: las etiquetas HS/TR/AG que menciona la plantilla del brief no aplican a AGE (son "
        "de la competencia de toxicidad); el desglose se hace por las 6 clases de edad.",
        "",
        "## Limites declarados",
        "",
        "- Una sola semilla de barajado (presupuesto del ciclo); el revisor ML recomendo varias y "
        "queda como recomendacion no ejecutada.",
        "- La anotacion NER sobredetecta (23,7% de tokens; MISC 57%); el claim es 'senal empirica "
        "en anotacion ruidosa', no semantica.",
        "- Los pesos del plug-in se aplican a p_combinada (suposicion de escala pre-registrada).",
        "- Acuerdo md vs lg en la secuencia de entidades: 28,91% (limite de fiabilidad).",
    ]
    texto = "\n".join(l.rstrip() for l in lineas) + "\n"
    DECISION_MD.write_text(texto, encoding="utf-8")


def d_078(lote: dict, baseline: float) -> str:
    lineas = ["---", ""]
    if lote["promising"]:
        ganadores = [f"{rep}/{anot}" for (rep, anot), r in sorted(lote["brazos"].items())
                     if r["pasa_gate"]]
        lineas += [
            "## D-078 - P6: el canal NER estructural pasa el gate (PROMISING)",
            "",
            f"- **Observation**: se ejecuto P6 con el pre-registro congelado. Brazos que pasan: "
            f"{', '.join(ganadores)}. Baseline primario recomputado: {baseline:.7f}.",
            "- **Decision**: segunda particion agrupada con configuracion congelada (semilla "
            "987654, reejecucion declarada) antes de cualquier otra cosa. Sin submission sin "
            "autorizacion explicita de Carlos.",
            "- **Consequences**: el campeon adoptado sigue intacto hasta que una segunda "
            "particion confirme; P3/P4/P5/P7 no se reabren por este resultado.",
        ]
    else:
        lineas += [
            "## D-078 - P6: el canal NER estructural no supera el gate; NER se cierra documentado",
            "",
            f"- **Observation**: se ejecuto P6 con el pre-registro congelado (N1, N1b y N2; md y "
            f"lg; control barajado por el pipeline completo). Ningun brazo alcanzo el gate "
            f"(delta >= +0,0015 con 4/5 folds, real > barajado, bootstrap sin cero). Baseline "
            f"recomputado: {baseline:.7f}.",
            "- **Result**: " + " | ".join(
                f"{rep}/{anot}: delta "
                f"{lote['brazos'][(rep, anot)]['delta_vs_baseline']:+.7f} "
                f"({lote['brazos'][(rep, anot)]['folds_positivos']}/5), barajado "
                f"{lote['brazos_barajados'][(rep, anot)]['delta_vs_baseline']:+.7f}"
                for rep, anot in sorted(lote["brazos"])) + ".",
            "",
            "- **Decision**: NER estructural se cierra documentado (arbol pre-registrado y regla "
            "del brief: hipotesis negativa se cierra; no se buscan variantes inmediatamente). El "
            "campeon adoptado (V30 + plug-in, LB 0,63575) permanece intacto.",
            "",
            "- **Consequences**: la conclusion es sobre NER estructural con la integracion "
            "probada, no sobre 'toda la familia linguistica' (P2 ya dejo esa leccion: modos de "
            "fallo distintos no se extrapolan). P5 (dependencias) queda como unica familia "
            "GENUINELY_NEW documentada y no priorizada; P3/P4 quedan PARTIALLY_COVERED. Sin "
            "reabrir POS ni encadenar variantes sin hipotesis nueva.",
        ]
    lineas += [
        "",
        "- **Evidence**: `experiments/age_linguistic_06_ner_estructura.json`, "
        "`reports/ner_structural_results.csv`, `reports/ner_structural_complementarity.csv`, "
        "`reports/ner_structural_decision.md`.",
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

    veredicto = "PROMISING" if lote["promising"] else "ningun brazo pasa"
    md(f"""## PHASE - NER STRUCTURAL REPRESENTATION

Hipotesis del ciclo: la estructura de **tipos** de entidad (PER/ORG/LOC/MISC, nunca nombres)
contiene informacion complementaria al campeon adoptado (V30 + plug-in, LB 0,63575).

- Legalidad: D-076 (etiquetas si, vectores no); anotador spaCy no-transformer, hashes congelados.
- Cobertura (md): 81,2% de tweets con >=1 entidad; MISC 57%; ~28% de tokens de entidad sobre
  menciones/RT/hashtags/URLs (solapamiento declarado con canales V31 cerrados).
- Representaciones: N1 (tipos por span), N1b (atribucion sin marcas), N2 (tipo + POS contexto).
- Integracion unica pre-registrada: combinador logistico anidado sobre [p_V30 || p_canal]
  normalizadas por fila; regla y pesos del plug-in congelados.
- Control: secuencias barajadas entre tweets por el pipeline completo (semilla 20261029).

**Resultado: {veredicto}.**""")

    code("""import json, pathlib
import pandas as pd
RAIZ = pathlib.Path("..")
res = json.loads((RAIZ / "experiments" / "age_linguistic_06_ner_estructura.json").read_text(encoding="utf-8"))
print("verificacion:", res["verificacion"], "| baseline:", res["baseline_primario"])
tabla = pd.read_csv(RAIZ / "reports" / "ner_structural_results.csv")
print()
print(tabla.to_string(index=False))
print()
comp = pd.read_csv(RAIZ / "reports" / "ner_structural_complementarity.csv")
todas = comp[comp["clase"] == "TODAS"]
print(todas[["representacion","anotador","rescate_unico","danio","neto","rescates_alta_confianza"]].to_string(index=False))
assert res["verificacion"]["hashes_ok"] is True
assert res["verificacion"]["constantes_ok"] is True
assert abs(res["baseline_primario"] - 0.6159732) < 1e-6
assert res["sin_test"] is True and res["sin_submission"] is True
print()
print("[OK] P6 ejecutado contra el pre-registro congelado; sin tocar test")""")
    md("""### Lectura y limites

La decision usa el arbol pre-registrado. Limites declarados: una sola semilla de barajado
(presupuesto); anotacion NER ruidosa (claim: senal empirica, no semantica); pesos del plug-in
aplicados a p_combinada (suposicion de escala); acuerdo md-lg en entidades 28,91%. Detalle en
`reports/ner_structural_decision.md`.""")
    nbformat.write(nb, NOTEBOOK)
    return {"antes": antes, "despues": len(nb.cells), "anadidas": len(nb.cells) - antes}


def main() -> int:
    autotest_numerico()
    inicio = time.time()
    prereg = json.loads(PREREG_JSON.read_text(encoding="utf-8"))

    print("=== verificacion contra el pre-registro ===", flush=True)
    rep_ok = {}
    for r in VEC_STR:
        esperado = prereg["representaciones"][r]["vectorizador"]
        if esperado == "identico a N1":  # definicion literal del pre-registro para N1b
            esperado = prereg["representaciones"]["N1"]["vectorizador"]
        rep_ok[r] = (esperado == VEC_STR[r])
    constantes_ok = (prereg["gates"]["delta_min"] == GATE_DELTA
                     and prereg["gates"]["folds_min"] == GATE_FOLDS
                     and prereg["semillas"] == {"bootstrap": SEMILLA_BOOTSTRAP,
                                                "barajado": SEMILLA_BARAJADO}
                     and all(rep_ok.values()))
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
    print(f"  constantes ok: {constantes_ok} | hashes ok: {hashes_ok} | reps: {rep_ok}", flush=True)
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
    print(f"  baseline: {baseline:.7f} (esperado {prereg['baseline_primario']})", flush=True)

    textos = datos.load_train()[cfg.TEXT_COL].fillna("").astype(str).tolist()
    print(f"\n=== construccion de secuencias ({len(textos):,} docs) ===", flush=True)
    secuencias = {}
    for clave in ANOTADORES:
        frame = pd.read_parquet(PARQUETS[clave], columns=["ent_seq", "pos_seq"])
        secuencias[clave] = construir_secuencias(frame["ent_seq"].astype(str).to_numpy(),
                                                 frame["pos_seq"].astype(str).to_numpy(), textos)
        cobertura = {rep: round(100.0 * float(np.mean([bool(str(s).strip())
                                                       for s in secuencias[clave][rep]])), 2)
                     for rep in VEC_STR}
        print(f"  {clave}: cobertura no vacia {cobertura}", flush=True)

    rng = np.random.default_rng(SEMILLA_BARAJADO)
    perm = rng.permutation(len(clase))

    lote = {"brazos": {}, "brazos_barajados": {}, "detalle_canal": {}, "detalle_canal_barajado": {},
            "combinador": {}, "combinador_barajado": {}, "filas_resultados": [], "filas_comp": []}
    for clave in ANOTADORES:
        for rep in VEC_STR:
            print(f"\n=== {rep}/{clave} (real) ===", flush=True)
            p_canal, info = canal_oof(secuencias[clave][rep], clase, fold_idx, VEC_KWARGS[rep],
                                      f"{rep}-{clave}", top_features=(rep == "N1" and clave == "md"))
            p_comb, comb = combinador_oof(v30, p_canal, clase, fold_idx, f"comb-{rep}-{clave}")
            r = evalua(p_comb, clase, fold_idx, v30, champ_pred, pesos, baseline)
            print(f"  {rep}/{clave}: canal solo {macro(clase, np.argmax(p_canal, axis=1)):.7f} | "
                  f"delta {r['delta_vs_baseline']:+.7f} | folds+ {r['folds_positivos']}/5 | "
                  f"peso canal {comb['peso_bloque_canal_ratio_media']}", flush=True)

            print(f"=== {rep}/{clave} (barajado) ===", flush=True)
            seq_barajada = secuencias[clave][rep][perm]
            p_canal_b, info_b = canal_oof(seq_barajada, clase, fold_idx, VEC_KWARGS[rep],
                                          f"{rep}-{clave}-barajado")
            p_comb_b, comb_b = combinador_oof(v30, p_canal_b, clase, fold_idx,
                                              f"comb-{rep}-{clave}-barajado")
            r_b = evalua(p_comb_b, clase, fold_idx, v30, champ_pred, pesos, baseline)

            pasa = (r["delta_vs_baseline"] >= GATE_DELTA and r["folds_positivos"] >= GATE_FOLDS
                    and r["ic95"][0] > 0
                    and r["delta_vs_baseline"] > r_b["delta_vs_baseline"])
            r["pasa_gate"] = bool(pasa)
            print(f"  {rep}/{clave}: real {r['delta_vs_baseline']:+.7f} vs barajado "
                  f"{r_b['delta_vs_baseline']:+.7f} -> {'PASA' if pasa else 'no pasa'}", flush=True)

            lote["brazos"][(rep, clave)] = r
            lote["brazos_barajados"][(rep, clave)] = r_b
            lote["detalle_canal"][(rep, clave)] = {**info,
                                                   "canal_solo": round(macro(clase,
                                                                             np.argmax(p_canal,
                                                                                       axis=1)), 7)}
            lote["detalle_canal_barajado"][(rep, clave)] = info_b
            lote["combinador"][(rep, clave)] = comb
            lote["combinador_barajado"][(rep, clave)] = comb_b
            complementariedad(clave, rep, r.pop("_pred"), clase, champ_pred, conf, lote["filas_comp"])
            r_b.pop("_pred")
            lote["filas_resultados"].append({
                "representacion": rep, "anotador": clave, "brazo": "real",
                "corrected_cv": r["corrected_cv"], "delta_vs_baseline": r["delta_vs_baseline"],
                "delta_vs_nested": r["delta_vs_nested"], "folds_positivos": r["folds_positivos"],
                "folds": ";".join(f"{v:.6f}" for v in r["folds"]),
                "ic95_lo": r["ic95"][0], "ic95_hi": r["ic95"][1],
                "filas_que_cambian": r["filas_que_cambian"],
                "vocab_medio": round(float(np.mean(info["vocabularios"])), 1),
                "segundos_canal": sum(info["segundos_por_fold"]),
                "pct_no_vacias": info["pct_no_vacias"],
                "peso_bloque_canal_ratio": comb["peso_bloque_canal_ratio_media"],
                "argmax_sin_regla": r["argmax_sin_regla"],
                "pasa_gate": r["pasa_gate"]})
            lote["filas_resultados"].append({
                "representacion": rep, "anotador": clave, "brazo": "barajado",
                "corrected_cv": r_b["corrected_cv"], "delta_vs_baseline": r_b["delta_vs_baseline"],
                "delta_vs_nested": r_b["delta_vs_nested"],
                "folds_positivos": r_b["folds_positivos"],
                "folds": ";".join(f"{v:.6f}" for v in r_b["folds"]),
                "ic95_lo": r_b["ic95"][0], "ic95_hi": r_b["ic95"][1],
                "filas_que_cambian": r_b["filas_que_cambian"],
                "vocab_medio": round(float(np.mean(info_b["vocabularios"])), 1),
                "segundos_canal": sum(info_b["segundos_por_fold"]),
                "pct_no_vacias": info_b["pct_no_vacias"],
                "peso_bloque_canal_ratio": comb_b["peso_bloque_canal_ratio_media"],
                "argmax_sin_regla": r_b["argmax_sin_regla"], "pasa_gate": ""})

    lote["promising"] = any(r["pasa_gate"] for r in lote["brazos"].values())
    lote["compliance"] = auditoria_compliance()
    print(f"\n  PROMISING: {lote['promising']} | compliance: {lote['compliance']['limpio']}",
          flush=True)

    escribe_resultados_csv(lote["filas_resultados"])
    cols_c = ["representacion", "anotador", "clase", "filas", "rescate_unico",
              "rescate_compartido", "danio", "neto", "rescates_alta_confianza",
              "errores_alta_confianza_campeon"]
    with COMPL_CSV.open("w", encoding="utf-8", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=cols_c, lineterminator="\n")
        escritor.writeheader()
        for fila in lote["filas_comp"]:
            escritor.writerow({c: fila.get(c, "") for c in cols_c})
    escribe_decision_md(lote, baseline)

    RESULTADO.write_text(json.dumps({
        "fase": "P6 canal NER estructural", "fecha": FECHA,
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT.parent,
                                 capture_output=True, text=True, timeout=30).stdout.strip(),
        "verificacion": {"hashes_ok": hashes_ok, "constantes_ok": constantes_ok},
        "baseline_primario": round(baseline, 7),
        "brazos": {"|".join(k): v for k, v in lote["brazos"].items()},
        "brazos_barajados": {"|".join(k): v for k, v in lote["brazos_barajados"].items()},
        "detalle_canal": {"|".join(k): v for k, v in lote["detalle_canal"].items()},
        "combinador": {"|".join(k): v for k, v in lote["combinador"].items()},
        "promising": lote["promising"], "compliance": lote["compliance"],
        "sin_test": True, "sin_holdout": True, "sin_submission": True,
        "segundos_total": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("  experiments/age_linguistic_06_ner_estructura.json", flush=True)

    print("\n=== documentacion ===", flush=True)
    print("  " + anade(ROOT / "DECISIONS.md", "## D-078", d_078(lote, baseline)), flush=True)
    print(f"  {DECISION_MD.name} | {RESULTADOS_CSV.name} | {COMPL_CSV.name}", flush=True)

    runlog = ROOT / "RUNLOG.md"
    if "P6 ejecutado: NER estructural" not in runlog.read_text(encoding="utf-8"):
        resumen = " | ".join(f"{rep}/{anot}: {v['delta_vs_baseline']:+.7f} "
                             f"({v['folds_positivos']}/5)" for (rep, anot), v
                             in sorted(lote["brazos"].items()))
        bloque = ("\n---\n\n## P6 ejecutado: NER estructural (tipos de entidad) · " + FECHA + "\n\n"
                  f"- Brazos reales: {resumen} | PROMISING: {lote['promising']}.\n"
                  f"- Control barajado por el pipeline completo (semilla {SEMILLA_BARAJADO}); "
                  f"baseline {baseline:.7f}.\n"
                  "- Sin tuning, sin test, sin holdout, sin submission; campeon intacto.\n")
        with runlog.open("a", encoding="utf-8") as f:
            f.write(bloque)
        print("  RUNLOG.md: P6 anotado", flush=True)

    ruta_exp = ROOT / "experiments.csv"
    lineas_exp = ruta_exp.read_text(encoding="utf-8").splitlines()
    cols = [c.strip() for c in lineas_exp[0].split(",")]
    if "AGE-LING-P6-NER" not in {l.split(",", 1)[0] for l in lineas_exp[1:] if l}:
        resumen = " | ".join(f"{rep}/{anot}: delta {v['delta_vs_baseline']:+.7f} "
                             f"({v['folds_positivos']}/5) {'PASA' if v['pasa_gate'] else 'no pasa'}"
                             for (rep, anot), v in sorted(lote["brazos"].items()))
        reg = {"experiment_id": "AGE-LING-P6-NER", "fecha": FECHA, "bloque": "30.3",
               "etapa": "P6 ejecucion: canal NER estructural (tipos de entidad) contra el campeon",
               "pregunta": "la estructura de tipos de entidad aporta al campeon bajo un combinador "
                           "anidado controlado?",
               "hipotesis": "la informacion de entidades no es equivalente a POS (P2) y puede ser "
                            "complementaria",
               "features": "secuencias de tipos PER/ORG/LOC/MISC (N1, N1b, N2) -> TF-IDF + LogReg",
               "preprocessing": "spans = runs de ent_seq; N1b excluye marcas; N2 anade contexto POS",
               "representacion": "canal sparse de tipos de entidad (sin nombres)",
               "modelo": "combinador logistico anidado sobre [p_V30, p_canal] normalizadas por fila",
               "hiperparametros": "congelados por pre-registro (AGE-LING-P6-PREREG)",
               "validacion": "particion canonica; baseline 0.6159732; barajado entre tweets por "
                             "pipeline completo (20261029); gates +0.0015, 4/5, bootstrap",
               "macro_f1_mean": "", "macro_f1_std": "", "weighted_f1_mean": "", "micro_f1_mean": "",
               "accuracy_mean": "",
               "resultado": resumen,
               "interpretacion": "lectura por el arbol pre-registrado; N1b aisla la atribucion de "
                                 "marcas de Twitter",
               "decision": "PROMISING -> segunda particion; si no, cierre documentado de NER",
               "duracion_seg": f"{round(time.time() - inicio, 1)}",
               "artefactos": "reports/ner_structural_decision.md | reports/ner_structural_results.csv | "
                             "reports/ner_structural_complementarity.csv | "
                             "experiments/age_linguistic_06_ner_estructura.json"}
        with ruta_exp.open("a", encoding="utf-8") as f:
            f.write(",".join(f'"{reg[c]}"' if ("," in str(reg[c]) or '"' in str(reg[c]))
                             else str(reg[c]) for c in cols) + "\n")
        print("  experiments.csv: fila AGE-LING-P6-NER", flush=True)

    estado_txt = ("PROMISING: segunda particion congelada como siguiente paso; sin submission sin "
                  "autorizacion explicita" if lote["promising"] else
                  "NINGUN BRAZO PASA: NER estructural se cierra documentado; el campeon intacto")
    bloque_estado = ("\n---\n\n## P6 ejecutado: NER estructural (2026-09-29)\n\n"
                     f"- Resultado: {estado_txt}.\n"
                     f"- N1/N1b/N2 x md/lg con control barajado por el pipeline completo; "
                     f"baseline {baseline:.7f}.\n"
                     "- Sin test, sin holdout, sin leaderboard, sin submission.\n")
    print("  " + anade(ROOT / "PROJECT_STATE.md", "## P6 ejecutado: NER estructural",
                       bloque_estado), flush=True)
    print("  " + anade(ROOT / "FINAL_REPORT.md", "## P6 ejecutado: NER estructural",
                       bloque_estado), flush=True)
    print("  " + str(anade_notebook(lote, baseline)), flush=True)

    print(f"\n[fin] {time.time() - inicio:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
