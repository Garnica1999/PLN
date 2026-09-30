"""P7 - pre-registro CONGELADO: canal de texto tipado (texto original + lemma_POS) contra el campeon.

Se escribe ANTES de ejecutar P7. Congela representacion, vectorizadores exactos, escala por
bloque, integracion, control, gates, arbol de decision y revisiones de agentes. Nada se ajusta
mirando el resultado. No toca test, no toca holdout, sin leaderboard, sin submission.

Diferencias declaradas con P2 y P6:
- P2 probo POS-solamente (sin lexema); P6 probo tipos de entidad. P7 conserva el lexema y anade
  su funcion: la representacion es texto original + tokens lemma_POS, como dos bloques.
- P7 reutiliza el combinador anidado de P6 (procedimiento equivalente); no repite la mezcla fija
  de P2 (la dilucion quedo documentada en D-077).
- P7 corre md y lg (mandato de comparacion de D-076), como P2 y P6.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

FECHA = "2026-09-30"
ANOTACION_JSON = cfg.EXPERIMENTS / "age_linguistic_01_anotacion.json"
CANONICO = ROOT / "artifacts" / "p33_scores" / "todos"
MANIFEST = ROOT / "reports" / "age_submission_plugin_manifest.json"
PREREG_MD = cfg.REPORTS / "age_typed_text_preregistro.md"
PREREG_JSON = cfg.EXPERIMENTS / "age_linguistic_08_texto_tipado_preregistro.json"

N_CLASES = 6
SEMILLA_BOOTSTRAP = 20261010
SEMILLA_BARAJADO = 20261030
GATE_DELTA = 0.0015
GATE_FOLDS = 4
UMBRAL_ALTA_CONFIANZA = 0.60
VARIABILIDAD = 0.0008876
N_BOOTSTRAP = 2000
CUBOS_LONGITUD = {"cortos<=40": (0, 40), "medios_41-93": (41, 93),
                  "largos_94-130": (94, 130), "muy_largos>130": (131, 10**9)}
VEC_A_STR = ('TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=3, max_df=0.5, '
             'sublinear_tf=True, lowercase=True, strip_accents=None, token_pattern=r"(?u)\\b\\w+\\b")')
VEC_B_STR = ('TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=5, sublinear_tf=True, '
             'lowercase=True, token_pattern=r"(?u)\\S+")')
MODELO_STR = "LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs', tol=1e-3)"
INTEGRACION = {
    "mecanismo": "combinador logistico anidado sobre dos bloques de probabilidad (identico a P6)",
    "features": "[p_V30 L2 por fila (6) || p_texto_tipado L2 por fila (6)] = 12 features",
    "modelo": "LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs') ajustado SOLO en el "
              "train de cada fold, con predicciones OOF del canal (nunca in-sample)",
    "decision": "argmax_c(w_c * p_combinada_c) con regla y pesos del plug-in CONGELADOS",
    "justificacion": "igual que P6 (procedimiento equivalente); evita la mezcla fija que diluyo "
                     "en P2; V30 queda como caja negra; sin features externas ni residuos",
    "ablacion_sin_canal": "el brazo A (V30 + plug-in, 0.6159732) ES la ablacion sin canal que "
                          "exige la condicion 4 de D-076; se reporta explicitamente como tal",
}
CONTROL = {
    "que": "el bloque B (lemma_POS) se permuta DENTRO DE CADA FOLD (5 permutaciones, semilla "
           "20261030; el mismo mapeo para md y lg); el bloque A (texto) queda intacto",
    "que_preserva": "misma dimensionalidad, misma frecuencia de tokens, misma distribucion "
                    "marginal por fold (el IDF por fold queda identico al original)",
    "que_rompe": "solo la correspondencia tweet <-> estructura tipada",
    "por_que_dentro_de_fold": "recomendacion del revisor ML: aisla el emparejamiento "
                              "token-etiqueta sin mover la marginal de ningun split; queda "
                              "registrado tambien el barajado global como alternativa no usada",
}


def sha256(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def main() -> int:
    autotest_numerico()
    inicio = time.time()
    anot = json.loads(ANOTACION_JSON.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    pesos = [float(x) for x in manifest["weights"]]
    if len(pesos) != N_CLASES:
        print("[STOP] pesos sin 6 clases", flush=True)
        return 1

    hashes = {
        "anotacion_md_parquet": anot["fichas"]["md"]["sha256_parquet"],
        "anotacion_lg_parquet": anot["fichas"]["lg"]["sha256_parquet"],
        "modelo_md_meta_json": anot["fichas"]["md"]["sha256_meta_json"],
        "modelo_lg_meta_json": anot["fichas"]["lg"]["sha256_meta_json"],
        "corpus_train": sha256(ROOT / "data" / "raw" / "age_train.csv"),
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
    print(f"baseline primario: {baseline:.7f} (esperado 0.6159732)", flush=True)
    if abs(baseline - 0.6159732) > 5e-7:
        print("[STOP] baseline no reproduce", flush=True)
        return 1

    lineas = [
        "# Pre-registro congelado - P7: texto tipado (texto original + lemma_POS)",
        "",
        f"**Fecha**: {FECHA}. **Estado**: CONGELADO antes de ejecutar (commit y hashes en",
        "`experiments/age_linguistic_08_texto_tipado_preregistro.json`). Autorizado por D-076",
        "(etiquetas y lemas si; vectores no) y por el brief del ciclo.",
        "",
        "## La pregunta",
        "",
        "> Puede una representacion lexica tipada conservar el contenido que hace fuerte al",
        "> TF-IDF y, simultaneamente, incorporar la funcion linguistica de cada token de una",
        "> forma que el campeon no capture?",
        "",
        "A = V30 + plug-in (campeon). B = A + canal de texto tipado. **El unico cambio es el",
        "canal**: bloques de texto original + lemma_POS.",
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
        f"Baseline primario recomputado: **{baseline:.7f}** (brazo A; tambien la ablacion sin "
        "canal de D-076 condicion 4). Referencia secundaria: nested 0.6154643.",
        "",
        "## Representacion (congelada)",
        "",
        "```text",
        "bloque A: texto original        -> " + VEC_A_STR,
        "bloque B: tokens lemma_POS      -> " + VEC_B_STR,
        "escala por bloque: 1.0 fija (neutra; definida antes; sin busqueda de escala)",
        "vectorizadores SEPARADOS (sin colision de vocabulario); cada uno se ajusta SOLO",
        "en el train de cada fold; cada bloque L2 por fila (TF-IDF por defecto)",
        "modelo del canal: " + MODELO_STR,
        "```",
        "",
        "- Bloque A clonado bit a bit de V31-03 (VEC_PALABRA); bloque B: unigramas y bigramas de",
        "  lemma_POS, sin trigramas. No se anaden NER, dependencias, POS puro, morfologia,",
        "  longitud, URL, hashtags ni features densas (la finalidad es aislar P7).",
        "- El parquet de anotacion se valida en el runner (sin nulos, tokenizacion coherente,",
        "  documentos con B vacio: se cuentan y se reportan; no hay mas tratamiento).",
        "",
        "## Integracion (unica, pre-registrada)",
        "",
    ]
    for clave, valor in INTEGRACION.items():
        lineas.append(f"- **{clave}**: {valor}")
    lineas += [
        "",
        "## Control (definido ANTES de ver el resultado)",
        "",
    ]
    for clave, valor in CONTROL.items():
        lineas.append(f"- **{clave}**: {valor}")
    lineas += [
        "",
        "## Gates (todos obligatorios, por anotador)",
        "",
        f"- delta >= **+{GATE_DELTA:.4f}** sobre el baseline primario;",
        f"- folds positivos >= **{GATE_FOLDS}/5**;",
        "- delta real > delta barajado (misma representacion y anotador);",
        "- IC95 bootstrap pareado (2.000, semilla 20261010) sin cero;",
        "- sin leakage (canal OOF por fold; combinador solo con train; vectorizadores por fold).",
        "",
        "Resultado: PROMISING si pasa; MARGINAL si es positivo por debajo del gate; CLOSED si es",
        "negativo o el control empata o gana. Si falla: no se abre grid (brief).",
        "",
        "## Revisiones de agentes (brief seccion 5) - puntos adoptados",
        "",
        "| Agente | Punto | Decision |",
        "|---|---|---|",
        "| Kaggle Specialist | legalidad OK; lemas/etiquetas son anotacion de forma | adoptado |",
        "| Kaggle Specialist | PII: los lemas PROPN vienen del propio tweet (ya en bloque A) | registrado |",
        "| Kaggle Specialist | hashes de particion y anotacion; join 1:1 por fila; IDF por fold | adoptado |",
        "| Kaggle Specialist | barajar dentro del split | adoptado (dentro de cada fold) |",
        "| Kaggle Specialist | brazo neutral 0.5 extra | NO adoptado: presupuesto 1 experimento + 1 control |",
        "| Kaggle Specialist | correr md y lg (D-076) | adoptado |",
        "| Revisor ML | clonar VEC_PALABRA de V31-03 bit a bit | adoptado |",
        "| Revisor ML | tol=1e-3, max_iter=2000, registrar n_iter_ y warnings | adoptado (smoke: 123 iter) |",
        "| Revisor ML | escala 1.0 por bloque, no convertirla en grid | adoptado |",
        "| Revisor ML | barajar dentro de cada split, mismo criterio md/lg | adoptado |",
        "| Revisor ML | asserts de orden de clases; diagnostico T sola y sin regla | adoptado |",
        "| Revisor ML | smoke test previo de 1 fold | hecho: 19,1 s/fold, ~6 min los 20 fits |",
        "",
        "## Diagnostico adicional (no decisorio)",
        "",
        "- macro-F1 del canal T solo; macro-F1 del combinador con argmax puro (sin regla).",
        "- Delta descriptivo por cubos de longitud: " + ", ".join(CUBOS_LONGITUD) + ".",
        "- Si T sola supera a V30 en la particion canonica se registra como hallazgo, sin tocar",
        "  al campeon congelado.",
        "",
        "## Que NO se hara",
        "",
        "- No hay grid de n-gramas, escala, tokenizadores, dimensiones, semillas ni combinadores.",
        "- No se toca el test, ni el holdout, ni el leaderboard. Sin submission automatica.",
        "- No se reabren P2, P6 ni P5 en este ciclo; P5 se evaluara SOLO tras clasificar P7.",
        "",
        "## Arbol de decision pre-registrado",
        "",
        "| Resultado | Estado | Siguiente paso |",
        "|---|---|---|",
        "| pasa los 5 gates | **PROMISING** | bootstrap + complementariedad ya calculados; segunda "
        "particion agrupada con configuracion congelada (semilla 987654); sin submission |",
        "| positivo < +0.0015 | **MARGINAL** | cierre documentado; solo P5 si el gap la justifica |",
        "| <=0 o control empata/gana | **CLOSED** | cierre documentado; revisar P5 una sola vez |",
        "",
        "## Interpretacion pre-registrada (regla del brief)",
        "",
        "- Si P7 falla: 'la representacion lexema+POS no aporta mejora al campeon bajo el",
        "  mecanismo evaluado' (NO 'la informacion linguistica no sirve').",
        "- Si P7 mejora: 'la representacion lexema+POS aporta informacion complementaria al",
        "  campeon bajo el protocolo evaluado' (NO 'el POS funciona').",
    ]
    texto = "\n".join(l.rstrip() for l in lineas) + "\n"
    PREREG_MD.write_text(texto, encoding="utf-8")
    print(f"  {PREREG_MD.name}: {len(texto.splitlines())} lineas", flush=True)

    PREREG_JSON.write_text(json.dumps({
        "fase": "P7 pre-registro congelado", "fecha": FECHA,
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT.parent,
                                 capture_output=True, text=True, timeout=30).stdout.strip(),
        "hashes": hashes, "pesos_plugin": pesos,
        "baseline_primario": round(baseline, 7), "referencia_nested": 0.6154643,
        "vec_a": VEC_A_STR, "vec_b": VEC_B_STR, "modelo_canal": MODELO_STR,
        "escala_por_bloque": 1.0, "integracion": INTEGRACION, "control": CONTROL,
        "semillas": {"bootstrap": SEMILLA_BOOTSTRAP, "barajado": SEMILLA_BARAJADO},
        "gates": {"delta_min": GATE_DELTA, "folds_min": GATE_FOLDS},
        "umbral_alta_confianza": UMBRAL_ALTA_CONFIANZA, "n_bootstrap": N_BOOTSTRAP,
        "cubos_longitud": {k: list(v) for k, v in CUBOS_LONGITUD.items()},
        "variabilidad_entrenamiento": VARIABILIDAD, "sklearn_version": sklearn.__version__,
        "smoke": {"segundos_por_fold": 19.1, "n_iter": 123, "warnings": 0},
        "anotadores": ["md", "lg"],
        "no_hacer": ["grid", "multiples semillas", "multiples tokenizadores", "multiples "
                     "dimensiones", "multiples combinadores", "test", "holdout", "leaderboard",
                     "submission"],
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("  experiments/age_linguistic_08_texto_tipado_preregistro.json", flush=True)

    runlog = ROOT / "RUNLOG.md"
    marca = "Pre-registro de P7"
    if marca not in runlog.read_text(encoding="utf-8"):
        bloque = ("\n---\n\n## Pre-registro de P7: texto tipado (texto original + lemma_POS) · "
                  + FECHA + "\n\n"
                  "- Congelado: dos bloques (A = texto original, clon de VEC_PALABRA de V31-03; "
                  "B = lemma_POS 1-2\n  gramas), vectorizadores separados por fold, escala 1.0; "
                  "combinador anidado identico a P6;\n  control = bloque B permutado dentro de "
                  "cada fold (semilla 20261030); gates +0.0015/4-5 folds/\n  real>barajado/"
                  "bootstrap; md y lg.\n"
                  f"- Baseline primario recomputado: {baseline:.7f}.\n"
                  "- Revisiones de Kaggle Specialist y revisor ML incorporadas; smoke test de "
                  "1 fold: 19,1 s.\n"
                  "- Sin tuning, sin test, sin holdout, sin submission.\n")
        with runlog.open("a", encoding="utf-8") as f:
            f.write(bloque)
        print("  RUNLOG.md: pre-registro anotado", flush=True)

    ruta_exp = ROOT / "experiments.csv"
    lineas_exp = ruta_exp.read_text(encoding="utf-8").splitlines()
    cols = [c.strip() for c in lineas_exp[0].split(",")]
    if "AGE-LING-P7-PREREG" not in {l.split(",", 1)[0] for l in lineas_exp[1:] if l}:
        reg = {"experiment_id": "AGE-LING-P7-PREREG", "fecha": FECHA, "bloque": "30.4",
               "etapa": "P7 pre-registro congelado del texto tipado (texto original + lemma_POS)",
               "pregunta": "una representacion lexica tipada conserva el contenido TF-IDF y anade "
                           "funcion linguistica que el campeon no captura?",
               "hipotesis": "lexema + funcion por token produce representacion complementaria",
               "features": "bloque A texto original (word 1-2, V31-03) + bloque B lemma_POS "
                           "(1-2 gramas) -> TF-IDF + LogReg por fold",
               "preprocessing": "dos vectorizadores separados por fold; escala 1.0 por bloque",
               "representacion": "texto tipado en dos bloques sparse",
               "modelo": "LogReg C=1.0 tol=1e-3 sobre bloques apilados; combinador anidado 12 feats",
               "hiperparametros": "congelados por pre-registro (AGE-LING-P7-PREREG)",
               "validacion": "particion canonica; baseline 0.6159732; control B permutado dentro "
                             "de fold (20261030); gates +0.0015, 4/5, real>barajado, bootstrap",
               "macro_f1_mean": "", "macro_f1_std": "", "weighted_f1_mean": "", "micro_f1_mean": "",
               "accuracy_mean": "",
               "resultado": "pre-registro escrito y congelado; sin resultados aun",
               "interpretacion": "revisiones de agentes incorporadas; smoke 19,1 s/fold",
               "decision": "ejecutar P7 tal cual",
               "duracion_seg": "0.0",
               "artefactos": "reports/age_typed_text_preregistro.md | "
                             "experiments/age_linguistic_08_texto_tipado_preregistro.json"}
        with ruta_exp.open("a", encoding="utf-8") as f:
            f.write(",".join(f'"{reg[c]}"' if ("," in str(reg[c]) or '"' in str(reg[c]))
                             else str(reg[c]) for c in cols) + "\n")
        print("  experiments.csv: fila AGE-LING-P7-PREREG", flush=True)

    print(f"\n[fin] {time.time() - inicio:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
