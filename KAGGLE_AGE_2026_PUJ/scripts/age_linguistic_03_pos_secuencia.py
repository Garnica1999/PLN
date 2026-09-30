"""P2 - canal POS-secuencia contra el campeon adoptado: ejecucion del pre-registro congelado.

## Que hace, en el orden del pre-registro

1. Verifica que nada ha cambiado: recomputa los SHA256 de todos los artefactos fuente y compara
   contra `experiments/age_linguistic_02_preregistro.json`; compara las constantes (alfa, gates,
   semillas, parametros del canal, pesos). Si algo no cuadra, para.
2. Construye el OOF del canal `pos_seq` por fold (TF-IDF 1-3 + LogReg C=1.0, ajustado solo en el
   train de cada fold), para cada anotador (md y lg) y para sus dos controles de barajado.
3. Mide: (1) macro-F1 del canal solo; (2) complementariedad contra las decisiones del campeon;
   (3) integracion neutral `p_mix = 0.5*p_v30 + 0.5*p_canal` con los pesos del plug-in
   congelados, y su delta con bootstrap pareado.
4. Aplica los gates y el arbol de decision pre-registrados.

No toca test, no toca holdout, no usa el leaderboard y no genera ninguna submission.
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

FECHA = "2026-09-29"
N_CLASES = 6
PREREG_JSON = cfg.EXPERIMENTS / "age_linguistic_02_preregistro.json"
CANONICO = ROOT / "artifacts" / "p33_scores" / "todos"
PARQUETS = {
    "md": ROOT / "features" / "linguistic" / "anotacion_md_train.parquet",
    "lg": ROOT / "features" / "linguistic" / "anotacion_lg_train.parquet",
}
RESULTADO = cfg.EXPERIMENTS / "age_linguistic_03_pos_secuencia.json"
INFORME = cfg.REPORTS / "age_linguistic_pos_secuencia.md"
BRAZOS_CSV = cfg.REPORTS / "age_linguistic_pos_secuencia_brazos.csv"
NOTEBOOK = ROOT / "notebooks" / "99_entrega_consolidado.ipynb"
MARCADOR_NB = "P2 - CANAL POS-SECUENCIA"

ALFA_MEZCLA = 0.5
SEMILLA_BOOTSTRAP = 20261010
SEMILLA_BARAJADO_FILAS = 20261010
SEMILLA_BARAJADO_INTRA = 20261011
GATE_DELTA = 0.0015
GATE_FOLDS = 4
UMBRAL_ALTA_CONFIANZA = 0.60
VARIABILIDAD = 0.0008876
N_BOOTSTRAP = 2000
PARAMS_CANAL = {
    "vectorizador": "TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df=5, "
                    "sublinear_tf=True, lowercase=True, token_pattern=r'(?u)\\S+')",
    "modelo": "LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')",
    "fit": "SOLO en el train de cada fold (vectorizador y modelo por fold)",
    "entrada": "pos_seq del parquet congelado por anotador",
}
VEC_KWARGS = dict(analyzer="word", ngram_range=(1, 3), min_df=5, sublinear_tf=True,
                  lowercase=True, token_pattern=r"(?u)\S+")
LOGREG_KWARGS = dict(C=1.0, max_iter=2000, solver="lbfgs")


def sha256(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def normaliza(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def macro(clase: np.ndarray, pred: np.ndarray, mascara: np.ndarray | None = None) -> float:
    if mascara is not None:
        return float(f1_score(clase[mascara], pred[mascara], average="macro",
                              labels=list(range(N_CLASES))))
    return float(f1_score(clase, pred, average="macro", labels=list(range(N_CLASES))))


def macro_de_conteos(conteos: np.ndarray) -> float:
    """macro-F1 a partir de la matriz 6x6 de conteos (para el bootstrap)."""
    matriz = conteos.reshape(N_CLASES, N_CLASES)
    f1s = []
    for c in range(N_CLASES):
        tp = float(matriz[c, c])
        fp = float(matriz[:, c].sum() - tp)
        fn = float(matriz[c, :].sum() - tp)
        denom = 2.0 * tp + fp + fn
        f1s.append(2.0 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(f1s))


def bootstrap_delta(clase: np.ndarray, pred_base: np.ndarray, pred_cand: np.ndarray,
                    semilla: int, n_boot: int = N_BOOTSTRAP) -> dict:
    """IC95 del delta pareado de macro-F1 (semilla pre-registrada, unidad la fila)."""
    rng = np.random.default_rng(semilla)
    pb = clase * N_CLASES + pred_base
    pc = clase * N_CLASES + pred_cand
    n = len(clase)
    deltas = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
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
    """OOF del canal: vectorizador y modelo ajustados SOLO en el train de cada fold."""
    p = np.zeros((len(secuencias), N_CLASES), dtype=np.float64)
    segundos = []
    vocabularios = []
    for k in range(5):
        t0 = time.time()
        tr = fold_idx != k
        va = fold_idx == k
        vec = TfidfVectorizer(**VEC_KWARGS)
        x_tr = vec.fit_transform(secuencias[tr])
        x_va = vec.transform(secuencias[va])
        modelo = LogisticRegression(**LOGREG_KWARGS)
        modelo.fit(x_tr, clase[tr])
        p[va] = modelo.predict_proba(x_va)
        segundos.append(round(time.time() - t0, 1))
        vocabularios.append(int(len(vec.vocabulary_)))
        print(f"    [{etiqueta}] fold {k}: {x_tr.shape[1]} features | {segundos[-1]} s", flush=True)
    return normaliza(p), {"segundos_por_fold": segundos, "vocabularios": vocabularios}


def barajar_filas(secuencias: np.ndarray, semilla: int) -> np.ndarray:
    rng = np.random.default_rng(semilla)
    return secuencias[rng.permutation(len(secuencias))]


def barajar_intra(secuencias: np.ndarray, semilla: int) -> np.ndarray:
    rng = np.random.default_rng(semilla)
    salida = np.empty(len(secuencias), dtype=object)
    for i, secuencia in enumerate(secuencias):
        etiquetas = secuencia.split()
        if len(etiquetas) < 2:
            salida[i] = secuencia
            continue
        orden = rng.permutation(len(etiquetas))
        salida[i] = " ".join(etiquetas[j] for j in orden)
    return salida


def evalua(clave: str, p_canal: np.ndarray, p_filas: np.ndarray, p_intra: np.ndarray,
           clase: np.ndarray, fold_idx: np.ndarray, v30: np.ndarray, champ_pred: np.ndarray,
           conf: np.ndarray, pesos: np.ndarray, baseline: float) -> dict:
    """Las tres mediciones y los controles, tal como los congela el pre-registro."""
    pred_canal = np.argmax(p_canal, axis=1)
    ok_canal = pred_canal == clase
    ok_champ = champ_pred == clase
    rescate_unico = int((~ok_champ & ok_canal).sum())
    rescate_compartido = int((ok_champ & ok_canal).sum())
    danio = int((ok_champ & ~ok_canal).sum())
    err_alta = ~ok_champ & (conf >= UMBRAL_ALTA_CONFIANZA)
    resc_alta = int((err_alta & ok_canal).sum())

    medicion1 = {"canal_solo_corrected_cv": round(macro(clase, pred_canal), 7)}
    medicion2 = {
        "rescate_unico": rescate_unico, "rescate_compartido": rescate_compartido,
        "danio": danio, "neto": rescate_unico - danio,
        "errores_alta_confianza_campeon": int(err_alta.sum()),
        "rescates_unicos_alta_confianza": resc_alta,
        "tasa_rescate_alta_confianza": round(resc_alta / max(int(err_alta.sum()), 1), 5),
    }

    def brazo(p_canal_brazo: np.ndarray) -> dict:
        p_mix = normaliza(ALFA_MEZCLA * v30 + (1 - ALFA_MEZCLA) * p_canal_brazo)
        pred = np.argmax(p_mix * pesos, axis=1)
        cv = macro(clase, pred)
        folds = [macro(clase, pred, fold_idx == k) for k in range(5)]
        folds_base = [macro(clase, champ_pred, fold_idx == k) for k in range(5)]
        return {"corrected_cv": round(cv, 7), "delta_vs_baseline": round(cv - baseline, 7),
                "folds": [round(v, 6) for v in folds],
                "folds_positivos": int(sum(1 for a, b in zip(folds, folds_base) if a > b)),
                "filas_que_cambian": int((pred != champ_pred).sum()), "_pred": pred}

    real = brazo(p_canal)
    control_filas = brazo(p_filas)
    control_intra = brazo(p_intra)
    ic = bootstrap_delta(clase, champ_pred, real.pop("_pred"), SEMILLA_BOOTSTRAP)
    control_filas.pop("_pred")
    control_intra.pop("_pred")

    pasa = (real["delta_vs_baseline"] >= GATE_DELTA and real["folds_positivos"] >= GATE_FOLDS
            and ic["ic95"][0] > 0 and real["delta_vs_baseline"] > control_filas["delta_vs_baseline"])
    if pasa:
        veredicto = "PROMISING"
    elif real["delta_vs_baseline"] > 0:
        veredicto = "MARGINAL"
    else:
        veredicto = "NEGATIVO"

    return {
        "anotador": clave, "medicion1_canal_solo": medicion1, "medicion2_complementariedad": medicion2,
        "medicion3_integracion": {
            "real": {k: v for k, v in real.items()},
            "barajado_filas": control_filas, "barajado_intra": control_intra,
            "bootstrap": ic,
            "delta_vs_nested_6154643": round(real["corrected_cv"] - 0.6154643, 7),
            "delta_menos_control_filas": round(real["delta_vs_baseline"]
                                               - control_filas["delta_vs_baseline"], 7),
            "ratio_sobre_variabilidad": round(abs(real["delta_vs_baseline"]) / VARIABILIDAD, 3),
        },
        "gate": {"delta_min": GATE_DELTA, "folds_min": GATE_FOLDS,
                 "control_primario": "barajado_filas", "pasa": bool(pasa)},
        "veredicto": veredicto,
    }


def escribe_informe(resultados: dict, baseline: float) -> None:
    lineas = [
        "# P2 - canal POS-secuencia contra el campeon adoptado",
        "",
        f"**Fecha**: {FECHA} | **Pre-registro**: `reports/age_linguistic_pos_preregistro.md` "
        "(congelado antes de ejecutar) | **Baseline primario**: "
        f"{baseline:.7f} (regla de la submission, recomputado)",
        "",
        "## Veredicto",
        "",
        "| Anotador | Canal solo (macro-F1) | delta integracion | folds+ | IC95 delta | "
        "delta control barajado | Veredicto |",
        "|---|---|---|---|---|---|---|",
    ]
    for clave in ("md", "lg"):
        r = resultados["anotadores"][clave]
        m1 = r["medicion1_canal_solo"]["canal_solo_corrected_cv"]
        m3 = r["medicion3_integracion"]
        ic = m3["bootstrap"]["ic95"]
        lineas.append(f"| {clave} | {m1:.7f} | {m3['real']['delta_vs_baseline']:+.7f} | "
                      f"{m3['real']['folds_positivos']}/5 | [{ic[0]:+.7f}; {ic[1]:+.7f}] | "
                      f"{m3['barajado_filas']['delta_vs_baseline']:+.7f} | **{r['veredicto']}** |")
    lineas += [
        "",
        "## Medicion 2 - complementariedad contra las decisiones del campeon",
        "",
        "| Anotador | rescate unico | rescate compartido | danio | neto | rescates unicos de "
        "alta confianza | tasa sobre errores de alta confianza |",
        "|---|---|---|---|---|---|---|",
    ]
    for clave in ("md", "lg"):
        m2 = resultados["anotadores"][clave]["medicion2_complementariedad"]
        lineas.append(f"| {clave} | {m2['rescate_unico']:,} | {m2['rescate_compartido']:,} | "
                      f"{m2['danio']:,} | {m2['neto']:+,} | "
                      f"{m2['rescates_unicos_alta_confianza']:,} / "
                      f"{m2['errores_alta_confianza_campeon']:,} | "
                      f"{m2['tasa_rescate_alta_confianza']:.5f} |")
    lineas += [
        "",
        "## Medicion 3 - integracion neutral (alfa 0.5) y controles",
        "",
        "| Anotador | brazo | delta vs baseline | folds+ | filas que cambian |",
        "|---|---|---|---|---|",
    ]
    for clave in ("md", "lg"):
        m3 = resultados["anotadores"][clave]["medicion3_integracion"]
        for nombre, brazo in (("real", m3["real"]), ("barajado_filas", m3["barajado_filas"]),
                              ("barajado_intra", m3["barajado_intra"])):
            lineas.append(f"| {clave} | {nombre} | {brazo['delta_vs_baseline']:+.7f} | "
                          f"{brazo['folds_positivos']}/5 | {brazo['filas_que_cambian']:,} |")
    lineas += [
        "",
        "## Lectura y decision (arbol pre-registrado)",
        "",
    ]
    for clave in ("md", "lg"):
        r = resultados["anotadores"][clave]
        m3 = r["medicion3_integracion"]
        lineas += [
            f"### Anotador {clave}: {r['veredicto']}",
            "",
            f"- delta de la integracion: {m3['real']['delta_vs_baseline']:+.7f} "
            f"(ratio sobre la variabilidad 8,9e-04: {m3['ratio_sobre_variabilidad']:.2f});",
            f"- bootstrap IC95: [{m3['bootstrap']['ic95'][0]:+.7f}; "
            f"{m3['bootstrap']['ic95'][1]:+.7f}], P(delta>0) = "
            f"{m3['bootstrap']['p_delta_positivo']:.4f};",
            f"- control barajado de filas: {m3['barajado_filas']['delta_vs_baseline']:+.7f}; "
            f"control intra-documento: {m3['barajado_intra']['delta_vs_baseline']:+.7f};",
            f"- canal solo: {r['medicion1_canal_solo']['canal_solo_corrected_cv']:.7f}; "
            f"rescate unico {r['medicion2_complementariedad']['rescate_unico']:,} vs danio "
            f"{r['medicion2_complementariedad']['danio']:,}.",
            "",
        ]
    lineas += [
        "## Limites declarados",
        "",
        "- Alfa = 0.5 es el punto neutro pre-registrado; ningun parametro se ajusta mirando el",
        "  resultado. La conclusion vale sobre ESTA integracion y este canal.",
        "- El test no se ha tocado; el holdout tampoco. No hay submission.",
    ]
    texto = "\n".join(l.rstrip() for l in lineas) + "\n"
    INFORME.write_text(texto, encoding="utf-8")


def escribe_brazos_csv(resultados: dict) -> None:
    filas = []
    for clave in ("md", "lg"):
        r = resultados["anotadores"][clave]
        m3 = r["medicion3_integracion"]
        for nombre, brazo in (("real", m3["real"]), ("barajado_filas", m3["barajado_filas"]),
                              ("barajado_intra", m3["barajado_intra"])):
            filas.append({"anotador": clave, "brazo": nombre,
                          "corrected_cv": brazo["corrected_cv"],
                          "delta_vs_baseline": brazo["delta_vs_baseline"],
                          "folds_positivos": brazo["folds_positivos"],
                          "filas_que_cambian": brazo["filas_que_cambian"]})
    pd.DataFrame(filas).to_csv(BRAZOS_CSV, index=False, lineterminator="\n")


def anade_notebook(resultados: dict, baseline: float) -> dict:
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

    veredictos = {k: resultados["anotadores"][k]["veredicto"] for k in ("md", "lg")}
    md(f"""## P2 - CANAL POS-SECUENCIA (md y lg)

Resultado de la integracion neutral pre-registrada (alfa 0.5, pesos del plug-in congelados):
**md: {veredictos['md']}** | **lg: {veredictos['lg']}**. Sin ajuste posterior de ningun
parametro; el test y el holdout siguen intactos y no hay submission.""")

    code("""import json, pathlib
import pandas as pd
RAIZ = pathlib.Path("..")
res = json.loads((RAIZ / "experiments" / "age_linguistic_03_pos_secuencia.json").read_text(encoding="utf-8"))
print("verificacion previa:", res["verificacion"])
brazos = pd.read_csv(RAIZ / "reports" / "age_linguistic_pos_secuencia_brazos.csv")
print()
print(brazos.to_string(index=False))
print()
for clave in ("md", "lg"):
    m2 = res["anotadores"][clave]["medicion2_complementariedad"]
    print(f"{clave}: rescate unico {m2['rescate_unico']:,} | danio {m2['danio']:,} | "
          f"neto {m2['neto']:+,} | veredicto {res['anotadores'][clave]['veredicto']}")
assert res["verificacion"]["hashes_ok"] is True
assert res["verificacion"]["constantes_ok"] is True
assert abs(res["baseline_primario"] - 0.6159732) < 1e-6
assert res["sin_test"] is True and res["sin_submission"] is True
print()
print("[OK] P2 ejecutado contra el pre-registro congelado, sin tocar test")""")
    nbformat.write(nb, NOTEBOOK)
    return {"antes": antes, "despues": len(nb.cells), "anadidas": len(nb.cells) - antes}


def d_077(resultados: dict, baseline: float) -> str:
    pivotes = {k: resultados["anotadores"][k] for k in ("md", "lg")}
    ganador = max(pivotes, key=lambda k: pivotes[k]["medicion3_integracion"]
                  ["real"]["delta_vs_baseline"])
    estado = pivotes[ganador]["veredicto"]
    lineas = [
        "---",
        "",
        f"## D-077 - P2: el canal POS-secuencia queda en {estado}",
        "",
        "- **Observation**: se ejecuto P2 con el pre-registro congelado "
        "(`reports/age_linguistic_pos_preregistro.md`): canal `pos_seq` (md y lg) -> TF-IDF 1-3 + "
        "LogReg C=1.0 ajustado por fold sobre la particion canonica, integracion neutral "
        f"`p_mix = 0,5*p_V30 + 0,5*p_canal` con los pesos del plug-in congelados. Baseline "
        f"primario recomputado: {baseline:.7f}.",
        "",
        "- **Result**: " + " | ".join(
            f"**{k}**: delta {pivotes[k]['medicion3_integracion']['real']['delta_vs_baseline']:+.7f} "
            f"({pivotes[k]['medicion3_integracion']['real']['folds_positivos']}/5 folds, IC95 "
            f"[{pivotes[k]['medicion3_integracion']['bootstrap']['ic95'][0]:+.7f}; "
            f"{pivotes[k]['medicion3_integracion']['bootstrap']['ic95'][1]:+.7f}], control "
            f"barajado {pivotes[k]['medicion3_integracion']['barajado_filas']['delta_vs_baseline']:+.7f}) "
            f"-> {pivotes[k]['veredicto']}" for k in ("md", "lg")) + ".",
        "",
        "- **Decision**: " + (
            "PROMISING: la segunda particion agrupada (semilla 987654, reejecucion declarada) es "
            "el siguiente paso obligatorio antes de cualquier otra cosa. Sin submission sin "
            "autorizacion explicita."
            if estado == "PROMISING" else
            "no se adopta. El canal POS-secuencia se cierra documentado conforme al arbol "
            "pre-registrado (la regla del brief: una hipotesis negativa se cierra; no se encadenan "
            "variantes de POS sin una hipotesis nueva)."),
        "",
        "- **Consequences**: el candidato adoptado (V30 + plug-in, LB 0,63575) permanece intacto. "
        "El modo de fallo queda acotado por las mediciones 1 y 2 (senal en aislamiento y "
        "complementariedad contra el campeon), que acotan la lectura: no es 'POS no tiene "
        "informacion', es lo que el gate midio sobre ESTA integracion.",
        "",
        "- **Evidence**: `experiments/age_linguistic_03_pos_secuencia.json`, "
        "`reports/age_linguistic_pos_secuencia.md`, "
        "`reports/age_linguistic_pos_secuencia_brazos.csv`.",
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
    constantes_ok = (prereg["alfa_mezcla"] == ALFA_MEZCLA
                     and prereg["gates"]["delta_min"] == GATE_DELTA
                     and prereg["gates"]["folds_min"] == GATE_FOLDS
                     and prereg["gates"]["bootstrap_semilla"] == SEMILLA_BOOTSTRAP
                     and prereg["semillas_control"]["barajado_filas"] == SEMILLA_BARAJADO_FILAS
                     and prereg["semillas_control"]["barajado_intra"] == SEMILLA_BARAJADO_INTRA
                     and prereg["params_canal"] == PARAMS_CANAL)
    rutas_hashes = {
        "anotacion_md_parquet": PARQUETS["md"],
        "anotacion_lg_parquet": PARQUETS["lg"],
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
    # los hashes de los meta.json de los modelos se verificaron al construir la anotacion
    print(f"  constantes ok: {constantes_ok} | hashes ok: {hashes_ok}", flush=True)
    if not (constantes_ok and hashes_ok):
        print("[STOP] el pre-registro no se respeta; no se ejecuta nada", flush=True)
        return 1

    pesos = np.array(prereg["pesos_plugin"], dtype=np.float64)
    clase = np.load(CANONICO / "y_true.npy")
    fold_idx = np.load(CANONICO / "fold_idx.npy")
    v30 = normaliza(np.load(CANONICO / "v30_prob.npy").astype(np.float64))
    champ_pred = np.argmax(v30 * pesos, axis=1)
    conf = v30.max(axis=1)
    baseline = macro(clase, champ_pred)
    print(f"  baseline primario: {baseline:.7f} (esperado {prereg['baseline_primario']})", flush=True)

    resultados = {"anotadores": {}}
    for clave in ("md", "lg"):
        print(f"\n=== canal {clave} (real) ===", flush=True)
        pos = pd.read_parquet(PARQUETS[clave], columns=["pos_seq"])["pos_seq"].astype(str).to_numpy()
        p_real, info_real = canal_oof(pos, clase, fold_idx, f"{clave}-real")
        print(f"  controles {clave}...", flush=True)
        p_filas, _ = canal_oof(barajar_filas(pos, SEMILLA_BARAJADO_FILAS), clase, fold_idx,
                               f"{clave}-filas")
        p_intra, _ = canal_oof(np.asarray(barajar_intra(pos, SEMILLA_BARAJADO_INTRA)), clase,
                               fold_idx, f"{clave}-intra")
        r = evalua(clave, p_real, p_filas, p_intra, clase, fold_idx, v30, champ_pred, conf, pesos,
                   baseline)
        r["detalle_canal"] = info_real
        resultados["anotadores"][clave] = r
        m3 = r["medicion3_integracion"]
        print(f"  {clave}: canal solo {r['medicion1_canal_solo']['canal_solo_corrected_cv']:.7f} | "
              f"delta {m3['real']['delta_vs_baseline']:+.7f} | folds+ "
              f"{m3['real']['folds_positivos']}/5 | control "
              f"{m3['barajado_filas']['delta_vs_baseline']:+.7f} | {r['veredicto']}", flush=True)

    resultados.update({"fecha": FECHA, "baseline_primario": round(baseline, 7),
                       "verificacion": {"hashes_ok": hashes_ok, "constantes_ok": constantes_ok},
                       "sin_test": True, "sin_holdout": True, "sin_submission": True,
                       "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT.parent,
                                                capture_output=True, text=True,
                                                timeout=30).stdout.strip()})
    RESULTADO.write_text(json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")
    escribe_informe(resultados, baseline)
    escribe_brazos_csv(resultados)
    print(f"\n  {INFORME.name} | {BRAZOS_CSV.name} | {RESULTADO.name}", flush=True)

    print("  " + anade(ROOT / "DECISIONS.md", "## D-077", d_077(resultados, baseline)), flush=True)
    runlog = ROOT / "RUNLOG.md"
    marca = "P2 ejecutado: canal POS-secuencia"
    if marca not in runlog.read_text(encoding="utf-8"):
        v = {k: resultados["anotadores"][k]["veredicto"] for k in ("md", "lg")}
        bloque = ("\n---\n\n## P2 ejecutado: canal POS-secuencia · " + FECHA + "\n\n"
                  f"- md: {v['md']} | lg: {v['lg']} (arbol pre-registrado aplicado).\n"
                  f"- Baseline primario recomputado: {baseline:.7f}; sin ajuste de ningun parametro.\n"
                  "- Sin test, sin holdout, sin submission; candidato adoptado intacto.\n")
        with runlog.open("a", encoding="utf-8") as f:
            f.write(bloque)
        print("  RUNLOG.md: P2 anotado", flush=True)

    ruta_exp = ROOT / "experiments.csv"
    lineas_exp = ruta_exp.read_text(encoding="utf-8").splitlines()
    cols = [c.strip() for c in lineas_exp[0].split(",")]
    if "AGE-LING-P2-POS-SEQ" not in {l.split(",", 1)[0] for l in lineas_exp[1:] if l}:
        md_ = resultados["anotadores"]["md"]
        lg_ = resultados["anotadores"]["lg"]
        reg = {"experiment_id": "AGE-LING-P2-POS-SEQ", "fecha": FECHA, "bloque": "30.2",
               "etapa": "P2 ejecucion: canal POS-secuencia contra el campeon adoptado",
               "pregunta": "pos_seq aporta sobre el campeon y mejora al integrarse por promedio neutral?",
               "hipotesis": "la secuencia de etiquetas POS contiene informacion no explotada por V30",
               "features": "pos_seq (md y lg) -> TF-IDF 1-3 por fold",
               "preprocessing": "ninguno",
               "representacion": "canal sparse de secuencias POS",
               "modelo": "LogisticRegression C=1.0 + mezcla alfa 0.5 con V30 y pesos plug-in congelados",
               "hiperparametros": "congelados por pre-registro (AGE-LING-P2-PREREG)",
               "validacion": f"particion canonica; baseline primario {baseline:.7f}; gates +0.0015, "
                             "4/5, bootstrap 2000/20261010, control barajado de filas",
               "macro_f1_mean": f"{md_['medicion3_integracion']['real']['corrected_cv']}",
               "macro_f1_std": "", "weighted_f1_mean": "", "micro_f1_mean": "",
               "accuracy_mean": "",
               "resultado": f"md: delta {md_['medicion3_integracion']['real']['delta_vs_baseline']:+.7f} "
                            f"({md_['medicion3_integracion']['real']['folds_positivos']}/5) {md_['veredicto']} | "
                            f"lg: delta {lg_['medicion3_integracion']['real']['delta_vs_baseline']:+.7f} "
                            f"({lg_['medicion3_integracion']['real']['folds_positivos']}/5) {lg_['veredicto']}",
               "interpretacion": "lectura acotada por las mediciones 1 y 2 del pre-registro",
               "decision": "aplicado el arbol pre-registrado; sin submission",
               "duracion_seg": f"{round(time.time() - inicio, 1)}",
               "artefactos": "reports/age_linguistic_pos_secuencia.md | "
                             "reports/age_linguistic_pos_secuencia_brazos.csv | "
                             "experiments/age_linguistic_03_pos_secuencia.json"}
        with ruta_exp.open("a", encoding="utf-8") as f:
            f.write(",".join(f'"{reg[c]}"' if ("," in str(reg[c]) or '"' in str(reg[c]))
                             else str(reg[c]) for c in cols) + "\n")
        print("  experiments.csv: fila AGE-LING-P2-POS-SEQ", flush=True)

    print("\n=== notebook ===", flush=True)
    print("  " + str(anade_notebook(resultados, baseline)), flush=True)

    print(f"\n[fin] {time.time() - inicio:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
