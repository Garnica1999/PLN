"""P6 - pre-registro CONGELADO: canal NER estructural (tipos de entidad) contra el campeon.

Se escribe ANTES de ejecutar P6. Congela: representaciones, canal, integracion, controles,
gates, arbol de decision, revisiones de agentes y hashes de todo lo que se va a leer. Nada se
ajusta mirando el resultado. No toca test, no toca holdout, sin leaderboard, sin submission.

Diferencias con P2 (declaradas):
- P2 probaba la INFORMACION de POS en secuencia; P6 prueba la informacion de ENTIDADES (tipos).
- P2 usaba mezcla fija 50/50 (dilucion); P6 usa un combinador logistico anidado sobre bloques
  de probabilidad normalizados por fila, con la regla final y los pesos del plug-in congelados.
- P6 anade N1b (atribucion: sin spans sobre marcas de Twitter) y N2 (contexto POS inmediato).
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

FECHA = "2026-09-29"
ANOTACION_JSON = cfg.EXPERIMENTS / "age_linguistic_01_anotacion.json"
CANONICO = ROOT / "artifacts" / "p33_scores" / "todos"
MANIFEST = ROOT / "reports" / "age_submission_plugin_manifest.json"
PREREG_MD = cfg.REPORTS / "ner_structural_preregistro.md"
PREREG_JSON = cfg.EXPERIMENTS / "age_linguistic_05_preregistro.json"

N_CLASES = 6
SEMILLA_BOOTSTRAP = 20261010
SEMILLA_BARAJADO = 20261029
GATE_DELTA = 0.0015
GATE_FOLDS = 4
UMBRAL_ALTA_CONFIANZA = 0.60
VARIABILIDAD = 0.0008876
N_BOOTSTRAP = 2000
REPRESENTACIONES = {
    "N1": {"definicion": "un token ENT_<TIPO> por span de entidad, en orden; vacio si no hay",
           "vectorizador": "TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df=5, "
                           "sublinear_tf=True, lowercase=True, token_pattern=r'(?u)\\S+')",
           "rol": "primaria (brief seccion 8)"},
    "N1b": {"definicion": "N1 excluyendo spans que contienen menciones (@), hashtags (#), URL o RT",
            "vectorizador": "identico a N1", "rol": "atribucion (revision Kaggle Specialist)"},
    "N2": {"definicion": "por span: ENT_<TIPO>, <TIPO>_L_<POSizq|START>, <TIPO>_R_<POSder|END>",
           "vectorizador": "TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df=5, "
                           "max_features=2000, sublinear_tf=True, lowercase=True, "
                           "token_pattern=r'(?u)\\S+')",
           "rol": "extension (brief seccion 9; guarda de dimensionalidad del revisor ML)"},
}
CANAL_MODELO = "LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')"
CANAL_VACIO = "filas con secuencia vacia -> probabilidad uniforme 1/6 (identico en todos los brazos)"
INTEGRACION = {
    "mecanismo": "combinador logistico anidado sobre dos bloques de probabilidad",
    "features": "[p_V30 normalizadas por fila (L2, 6 dims) concatenadas con p_canal "
                "normalizadas por fila (L2, 6 dims)] = 12 features",
    "modelo": "LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs') ajustado SOLO en las "
              "filas de train de cada fold (con OOF estricto del canal, no in-sample)",
    "decision": "argmax_c(w_c * p_combinada_c) con la regla y los pesos del plug-in CONGELADOS",
    "suposicion_declarada": "los pesos del plug-in no son invariantes a escala; aplicarlos a "
                            "p_combinada es una suposicion pre-registrada (regla congelada, "
                            "brief seccion 24)",
    "por_que_no_es_la_familia_cerrada": "V31-06 anadio features del canal al meta-modelo de V30 "
                                        "(refit de V30); P33/P34/G12 inyectaron un residuo en el "
                                        "score. Aqui V30 es caja negra (solo interfaz de "
                                        "probabilidad), sin features externas ni residuo, y la "
                                        "regla de decision queda congelada. Es un candidato nuevo.",
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
        "corpus_train": sha256(ROOT / "data" / "raw" / "age_train.csv"),
        "canonico_y_true": sha256(CANONICO / "y_true.npy"),
        "canonico_fold_idx": sha256(CANONICO / "fold_idx.npy"),
        "canonico_v30_prob": sha256(CANONICO / "v30_prob.npy"),
        "manifest_plugin": sha256(MANIFEST),
    }

    clase = np.load(CANONICO / "y_true.npy")
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
        "# Pre-registro congelado - P6: canal NER estructural (tipos de entidad)",
        "",
        f"**Fecha**: {FECHA}. **Estado**: CONGELADO antes de ejecutar (commit y hashes en",
        "`experiments/age_linguistic_05_preregistro.json`). Autorizado por D-076 (etiquetas si,",
        "vectores no) y por el brief del ciclo.",
        "",
        "## La pregunta",
        "",
        "> La estructura de entidades (TIPOS PER/ORG/LOC/MISC, nunca nombres) contiene informacion",
        "> complementaria al campeon adoptado (V30 + plug-in), y un combinador anidado la explota",
        "> para mejorar el macro-F1 corregido?",
        "",
        "P2 cerro POS-secuencia (senal aislada 0.2751; integracion fija 50/50 negativa por",
        "dilucion). P6 responde otra pregunta: entidades + integracion CONTROLADA. No se extrapola",
        "el resultado de P2 a P6.",
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
        f"Baseline primario recomputado: **{baseline:.7f}** (la regla que ejecuta la submission).",
        "Referencia secundaria de continuidad: nested 0.6154643.",
        "",
        "## Representaciones (todas sin nombres concretos: solo tipos)",
        "",
        "| ID | Definicion | Vectorizador | Rol |",
        "|---|---|---|---|",
    ]
    for rid, rep in REPRESENTACIONES.items():
        lineas.append(f"| {rid} | {rep['definicion']} | {rep['vectorizador']} | {rep['rol']} |")
    lineas += [
        "",
        f"Canal: {CANAL_MODELO}, ajustado SOLO en el train de cada fold; {CANAL_VACIO}.",
        "Anotadores: md y lg (D-076 ordeno comparacion; los dos corren el mismo protocolo).",
        "",
        "## Integracion (unica, pre-registrada)",
        "",
        f"- Mecanismo: {INTEGRACION['mecanismo']}.",
        f"- Features: {INTEGRACION['features']}.",
        f"- Modelo: {INTEGRACION['modelo']}.",
        f"- Decision: {INTEGRACION['decision']}.",
        f"- Suposicion declarada: {INTEGRACION['suposicion_declarada']}.",
        f"- Por que no es la familia cerrada: {INTEGRACION['por_que_no_es_la_familia_cerrada']}",
        "",
        "Diagnostico no decisorio: se reportara tambien argmax(p_combinada) sin regla, para",
        "lectura; la decision usa la regla congelada.",
        "",
        "## Controles (definidos ANTES de ver el resultado)",
        "",
        "- `barajado`: la secuencia de cada representacion se permuta ENTRE tweets (una",
        f"  permutacion, semilla {SEMILLA_BARAJADO}), preservando marginal exacta. El pipeline",
        "  COMPLETO (canal + combinador + regla) se repite con la version barajada.",
        "- `N1b`: variante que excluye spans sobre menciones/RT/hashtags/URLs (~28% de los tokens",
        "  de entidad; canales de V31 cerrados). Si el aporte solo sobrevive con marcas de Twitter,",
        "  se clasifica como redundancia, no como senal nueva.",
        "- Se registran los coeficientes del combinador por fold (peso del bloque canal): un peso",
        "  ~0 significa recalibracion de V30, no informacion nueva de NER.",
        "",
        "## Gates (todos obligatorios, por brazo representacion x anotador)",
        "",
        f"- delta >= **+{GATE_DELTA:.4f}** sobre el baseline primario;",
        f"- folds positivos >= **{GATE_FOLDS}/5**;",
        "- delta real > delta barajado (misma representacion y anotador);",
        "- IC95 bootstrap pareado (2.000, semilla 20261010) sin cero;",
        "- sin leakage (auditado por diseno: canal OOF por fold, combinador solo con train).",
        "",
        "## Revisor de agentes (auditoria del brief seccion 5) - puntos adoptados",
        "",
        "| Agente | Punto | Decision |",
        "|---|---|---|",
        "| Kaggle Specialist | legalidad OK bajo D-076; documentar solapamiento de marcas | adoptado (N1b + doc) |",
        "| Kaggle Specialist | correr md y lg (mandato D-076) | adoptado |",
        "| Kaggle Specialist | segunda particion antes de cualquier submission | ya en el arbol |",
        "| Revisor ML | N2 con (1,2) + max_features=2000 (dimensionalidad) | adoptado |",
        "| Revisor ML | combinador distinto de la familia cerrada; documentar en ADR | adoptado |",
        "| Revisor ML | registrar coeficientes del combinador | adoptado |",
        "| Revisor ML | varias semillas de barajado | NO adoptado: brief seccion 26 prohibe "
        "multiples semillas; queda como limite declarado |",
        "| Revisor ML | control barajado por pipeline completo | adoptado |",
        "",
        "## Que NO se hara",
        "",
        "- No hay grid de NER, ni variantes de pesos, ni varios modelos NER, ni multiples semillas,",
        "  ni tuning del anotador (brief seccion 26).",
        "- No se toca el test, ni el holdout, ni el leaderboard. Sin submission automatica.",
        "- El campeon adoptado (V30 + plug-in, LB 0.63575) no se modifica.",
        "",
        "## Arbol de decision pre-registrado",
        "",
        "| Resultado | Estado | Siguiente paso |",
        "|---|---|---|",
        "| algun brazo pasa los 5 gates | **PROMISING** | segunda particion agrupada con "
        "configuracion congelada (semilla 987654, reejecucion declarada); sin submission sin "
        "autorizacion |",
        "| ningun brazo pasa | **CIERRE** | NER se cierra documentado; no se buscan otras "
        "variantes inmediatamente (brief seccion 22) |",
        "",
        "## Limites declarados",
        "",
        "- La anotacion NER esta sobredetectando sobre este corpus (23,7% de tokens en entidades;",
        "  MISC 57%; ~28% sobre marcas). El claim futuro sera 'senal empirica en anotacion",
        "  ruidosa', no semantica.",
        "- Una sola semilla de barajado (presupuesto); el revisor ML recomendo varias y queda",
        "  registrado como recomendacion no ejecutada.",
        "- El esquema de stacking usa OOF estricto del canal para las filas de train del combinador",
        "  (practica estandar del proyecto: V12/V20); el control pasa por el mismo esquema.",
    ]
    texto = "\n".join(l.rstrip() for l in lineas) + "\n"
    PREREG_MD.write_text(texto, encoding="utf-8")
    print(f"  {PREREG_MD.name}: {len(texto.splitlines())} lineas", flush=True)

    PREREG_JSON.write_text(json.dumps({
        "fase": "P6 pre-registro congelado", "fecha": FECHA,
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT.parent,
                                 capture_output=True, text=True, timeout=30).stdout.strip(),
        "hashes": hashes, "pesos_plugin": pesos,
        "baseline_primario": round(baseline, 7), "referencia_nested": 0.6154643,
        "representaciones": REPRESENTACIONES, "canal_modelo": CANAL_MODELO,
        "canal_vacio": CANAL_VACIO, "integracion": INTEGRACION,
        "semillas": {"bootstrap": SEMILLA_BOOTSTRAP, "barajado": SEMILLA_BARAJADO},
        "gates": {"delta_min": GATE_DELTA, "folds_min": GATE_FOLDS,
                  "control": "barajado entre tweets, pipeline completo"},
        "umbral_alta_confianza": UMBRAL_ALTA_CONFIANZA, "n_bootstrap": N_BOOTSTRAP,
        "variabilidad_entrenamiento": VARIABILIDAD,
        "sklearn_version": sklearn.__version__,
        "no_hacer": ["grid NER", "variantes de pesos", "varios modelos NER", "multiples semillas",
                     "tuning del anotador", "test", "holdout", "leaderboard", "submission"],
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("  experiments/age_linguistic_05_preregistro.json", flush=True)

    runlog = ROOT / "RUNLOG.md"
    marca = "Pre-registro de P6"
    if marca not in runlog.read_text(encoding="utf-8"):
        bloque = ("\n---\n\n## Pre-registro de P6: canal NER estructural · " + FECHA + "\n\n"
                  "- Congelado: representaciones N1 (tipos por span), N1b (atribucion sin marcas), "
                  "N2 (tipo + POS\n  de contexto, (1,2)+2000 features); combinador logistico "
                  "anidado sobre bloques de\n  probabilidad normalizados; regla y pesos del "
                  "plug-in congelados; barajado entre tweets por el\n  pipeline completo; gates "
                  "(+0.0015, 4/5, real>barajado, bootstrap) y arbol de decision.\n"
                  f"- Baseline primario recomputado: {baseline:.7f}.\n"
                  "- Revisiones de Kaggle Specialist y revisor ML incorporadas; limites declarados.\n"
                  "- Sin tuning, sin test, sin holdout, sin submission automatica.\n")
        with runlog.open("a", encoding="utf-8") as f:
            f.write(bloque)
        print("  RUNLOG.md: pre-registro anotado", flush=True)

    ruta_exp = ROOT / "experiments.csv"
    lineas_exp = ruta_exp.read_text(encoding="utf-8").splitlines()
    cols = [c.strip() for c in lineas_exp[0].split(",")]
    if "AGE-LING-P6-PREREG" not in {l.split(",", 1)[0] for l in lineas_exp[1:] if l}:
        reg = {"experiment_id": "AGE-LING-P6-PREREG", "fecha": FECHA, "bloque": "30.3",
               "etapa": "P6 pre-registro congelado del canal NER estructural",
               "pregunta": "la estructura de tipos de entidad aporta al campeon bajo un combinador "
                           "anidado controlado?",
               "hipotesis": "la informacion social/tematica de las entidades no es equivalente a POS",
               "features": "secuencias de tipos PER/ORG/LOC/MISC (N1, N1b, N2) -> TF-IDF + LogReg",
               "preprocessing": "spans = runs de ent_seq; N1b excluye marcas; N2 anade contexto POS",
               "representacion": "canal sparse de tipos de entidad (sin nombres)",
               "modelo": "combinador logistico anidado sobre [p_V30, p_canal] normalizadas por fila",
               "hiperparametros": "congelados: C=1.0, (1,3)/(1,2)+2000, min_df=5, sin tuning",
               "validacion": "particion canonica; baseline 0.6159732; barajado entre tweets "
                             "(semilla 20261029) por pipeline completo; gates +0.0015, 4/5, bootstrap",
               "macro_f1_mean": "", "macro_f1_std": "", "weighted_f1_mean": "", "micro_f1_mean": "",
               "accuracy_mean": "",
               "resultado": "pre-registro escrito y congelado; sin resultados aun",
               "interpretacion": "revisiones de agentes incorporadas; limites declarados",
               "decision": "ejecutar P6 tal cual",
               "duracion_seg": "0.0",
               "artefactos": "reports/ner_structural_preregistro.md | "
                             "experiments/age_linguistic_05_preregistro.json"}
        with ruta_exp.open("a", encoding="utf-8") as f:
            f.write(",".join(f'"{reg[c]}"' if ("," in str(reg[c]) or '"' in str(reg[c]))
                             else str(reg[c]) for c in cols) + "\n")
        print("  experiments.csv: fila AGE-LING-P6-PREREG", flush=True)

    print(f"\n[fin] {time.time() - inicio:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
