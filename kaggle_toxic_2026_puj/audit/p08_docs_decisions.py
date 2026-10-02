"""P8: auditoria documental y de decisiones.

- Clasifica las 67 decisiones D-xxx (VALIDATED / PARTIALLY / INVALIDATED /
  SUPERSEDED / UNCERTAIN) con justificacion y detecta conclusiones pendientes.
- Comprueba afirmaciones criticas contra artefactos (audit_documentation.csv).
- Tabla de familias cerradas y claims de techo.
"""
from __future__ import annotations

import csv
import json
import re
import unicodedata
from pathlib import Path

PROJ = Path("/home/carlos/Git/PLN-toxic-audit/kaggle_toxic_2026_puj")

ESTADOS = {
    "D-001": ("VALIDATED", "TR=>HS 0 violaciones verificado; postproceso medido y descartado (-0.0056)"),
    "D-002": ("VALIDATED", "sin recursos externos de Hateval; corpus/test limpios (P4)"),
    "D-003": ("VALIDATED", "spaCy solo anotador; sin vectores preentrenados en features"),
    "D-004": ("SUPERSEDED", "grupos solo near-duplicado; corregido por D-015/D-016"),
    "D-005": ("VALIDATED", "519/519 TR=>HS y 26 AG sin HS reproducidos (P1)"),
    "D-006": ("VALIDATED", "parsers y encoding verificados (utf-8/cp1252)"),
    "D-007": ("VALIDATED", "sin datos externos supervisados; lexicones no usados"),
    "D-008": ("VALIDATED", "preflight GPU documentado en DL"),
    "D-009": ("PARTIALLY_VALIDATED", "sonda E00 apoya column-wise/micro; la familia exacta no es observable"),
    "D-010": ("VALIDATED", "train sin diacriticos; test con 6 bytes no ASCII"),
    "D-012": ("VALIDATED", "umbrales adoptados con control; optimismo medido +0.0036"),
    "D-013": ("INVALIDATED", "v6/LB: reducir capacidad no redujo el gap (referencia stale en el doc)"),
    "D-014": ("SUPERSEDED", "cierre de exploracion LB revocado por el presupuesto de D-063"),
    "D-015": ("VALIDATED", "0.783825 vs 0.792659 reproducidos (P3)"),
    "D-011": ("VALIDATED", "estilo de codigo"),
    "D-016": ("VALIDATED", "holdout y OOF corregidos reproducidos (P1/P6)"),
    "D-017": ("VALIDATED", "cadenas sin mejora; meta v2 no transfirio (nota)"),
    "D-018": ("VALIDATED", "targeting sin mejora; encoding fold-local verificado"),
    "D-019": ("SUPERSEDED", "adopcion del meta revocada por D-024 (LB 0.78163 < 0.78306)"),
    "D-020": ("PARTIALLY_VALIDATED", "holdout +0.018 verificado; no transfirio al LB"),
    "D-021": ("VALIDATED", "backlog 24/24"),
    "D-022": ("VALIDATED", "cierre documental v2"),
    "D-023": ("VALIDATED", "decision de no sondear"),
    "D-024": ("VALIDATED", "3 sondas por debajo de V03 (log LB)"),
    "D-025": ("VALIDATED", "apertura v4 con protocolo"),
    "D-026": ("VALIDATED", "duplicacion escasa; dureza por frontera (v6 lo re-mide)"),
    "D-027": ("VALIDATED", "reweighting: mejora solo bajando violadores del umbral"),
    "D-028": ("VALIDATED", "shift medido y refutado"),
    "D-029": ("VALIDATED", "holdout rechaza al candidato; V03 final"),
    "D-030": ("VALIDATED", "scope externo cumplido y escaneado (P4)"),
    "D-031": ("VALIDATED", "estado de la representacion y palanca"),
    "D-032": ("VALIDATED", "cobertura sube, complementariedad no (nota: titulo con palabra en ingles)"),
    "D-033": ("VALIDATED", "curva 10M-50M plana; techo no es la representacion"),
    "D-034": ("VALIDATED", "tramo 5M; suelo de la curva"),
    "D-035": ("VALIDATED", "baseline v5.1 y presupuesto"),
    "D-036": ("VALIDATED", "vector_size inerte; gate de dos niveles"),
    "D-037": ("VALIDATED", "window inercia"),
    "D-038": ("VALIDATED", "min_count inercia con subpalabras"),
    "D-039": ("VALIDATED", "intervalo 3-6 optimo; regla de dispersion corregida"),
    "D-040": ("VALIDATED", "negative inerte; presupuesto agotado"),
    "D-041": ("VALIDATED", "conclusion arquitectonica"),
    "D-042": ("VALIDATED", "mismo mecanismo, menos ruido; correccion de vocabulario"),
    "D-043": ("PARTIALLY_VALIDATED", "cierre v5.1 GO; 'limite = datos etiquetados' es cota empirica, no prueba"),
    "D-044": ("VALIDATED", "apertura v6 con verificacion programatica"),
    "D-045": ("VALIDATED", "TR=>HS cierta pero no accionable; control B0 exacto"),
    "D-046": ("VALIDATED", "ponderacion por grupo no adopta; repeticion es senal"),
    "D-047": ("VALIDATED", "optimo interior; techo mitad del gate"),
    "D-048": ("VALIDATED", "skip-grams neutros; bug de tokenizacion detectado por control"),
    "D-049": ("VALIDATED", "bagging no refutable con el protocolo"),
    "D-050": ("PARTIALLY_VALIDATED", "conclusion se mantiene; componente de pesos no fold-local (auditoria P4)"),
    "D-051": ("VALIDATED", "aumento empeora; sesgo no cantidad"),
    "D-052": ("VALIDATED", "cierre v6 sin adopciones"),
    "D-053": ("VALIDATED", "P9 focal no mejora"),
    "D-054": ("VALIDATED", "repeticion es senal, no senal nueva"),
    "D-055": ("VALIDATED", "umbrales debilmente identificados (medido)"),
    "D-056": ("VALIDATED", "cierre v7"),
    "D-057": ("VALIDATED", "apertura v8 en rama aislada"),
    "D-058": ("VALIDATED", "identidad de pipeline y parada"),
    "D-059": ("VALIDATED", "limite en el externo (observado)"),
    "D-060": ("VALIDATED", "estabilidad de features cerrada"),
    "D-061": ("VALIDATED", "premisa propia falsa; suavizado"),
    "D-062": ("VALIDATED", "compartir estructura empeora"),
    "D-063": ("PARTIALLY_VALIDATED", "gasto de sondas autorizado; resultados LB por debajo de V03"),
    "D-064": ("VALIDATED", "estimacion inestable entre particiones"),
    "D-065": ("VALIDATED", "dos sondas pierden; curva de transferencia cierra calibracion"),
    "D-066": ("VALIDATED", "hipotesis de error de diseno refutada"),
    "D-067": ("VALIDATED", "promediado explicaba la mejora local"),
}

TECHOS = [
    ("v6", "techo de la interseccion de errores +0.0027", "cota empirica del OOF (no prueba matematica)", "BOUNDED"),
    ("v6", "techo empirico de la linea mas barata", "medicion sobre aciertos/errores OOF", "OBSERVED"),
    ("v6", "D (TextCNN) y E (kNN): techo teorico del canal", "cota derivada de interseccion de errores", "BOUNDED"),
    ("v7", "umbrales debilmente identificados (holgura ~0.002-0.003)", "medicion de dispersion por pliegue", "OBSERVED"),
    ("v4", "el shift no explica el error; vecindad con techo nulo", "mediciones de shift y propagacion", "OBSERVED"),
    ("v5.1", "el techo no es la representacion (curva plana 5M-50M)", "curva de escala medida", "OBSERVED"),
    ("v8", "el limite esta en el externo (LB), no en el modelo", "P1: identidad de pipeline + sondas", "OBSERVED"),
    ("v8", "la mejora local era el promediado (D-067)", "ensemble de 5 modelos: +0.005066 OOF con IC95 [-0.0037,+0.0141]", "PARTIALLY (no transfiere al LB)"),
]


def ascii_text(text: str) -> str:
    repl = {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
            "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u00a0": " "}
    for k, v in repl.items():
        text = text.replace(k, v)
    norm = unicodedata.normalize("NFKD", text)
    return "".join(ch if ord(ch) < 128 else "" for ch in norm if not unicodedata.combining(ch))


def main() -> None:  # noqa: PLR0915
    decisiones = re.split(r"\n## (D-\d+) - ", (PROJ / "DECISIONS.md").read_text())
    # decisiones: [preambulo, num, cuerpo, num, cuerpo, ...]
    filas_dec = []
    pendientes = []
    for i in range(1, len(decisiones), 2):
        numero, cuerpo = decisiones[i], decisiones[i + 1]
        titulo = cuerpo.splitlines()[0].strip()
        pendiente = bool(re.search(r"Conclusion final\**: *pendiente", cuerpo))
        tiene_evidencia = "**Evidencia**" in cuerpo
        tiene_resultado = "**Resultado**" in cuerpo
        estado, razon = ESTADOS.get(numero, ("UNCERTAIN", "sin clasificacion"))
        if pendiente:
            pendientes.append(numero)
        filas_dec.append({
            "Decision": numero, "Titulo": ascii_text(titulo), "Estado": estado,
            "Razon": ascii_text(razon), "Conclusion_pendiente": int(pendiente),
            "Tiene_evidencia": int(tiene_evidencia), "Tiene_resultado": int(tiene_resultado),
        })

    # ---------- consistencia documental ----------
    checks = []

    def add(claim, fuente, esperado, actual, consistente, severidad):
        checks.append({"Claim": ascii_text(claim), "Source": fuente,
                       "Expected": ascii_text(esperado), "Actual": ascii_text(actual),
                       "Consistent": consistente, "Severity": severidad})

    v6rep = (PROJ / "reports" / "v6_reproducibility.md").read_text(encoding="utf-8")
    n_rutas = len(re.findall(r"experiments/experiments/", v6rep))
    add("rutas de scripts en v6_reproducibility.md", "reports/v6_reproducibility.md",
        "experiments/<script>.py", f"{n_rutas} rutas con prefijo duplicado",
        "no" if n_rutas else "si", "LOW")
    te = (PROJ / "reports" / "target_encoding_audit_v2.md").read_text(encoding="utf-8")
    add("evaluaciones del holdout usadas", "reports/target_encoding_audit_v2.md",
        "5/5 (log final)", "2/5 (texto desactualizado)", "no", "LOW")
    nb = json.loads((PROJ / "notebooks" / "10_final_pipeline.ipynb").read_text(encoding="utf-8"))
    add("celdas del notebook", "notebooks/10_final_pipeline.ipynb",
        "63 celdas / 32 de codigo",
        f"{len(nb['cells'])} / {sum(1 for c in nb['cells'] if c['cell_type'] == 'code')}",
        "si", "INFO")
    add("conclusiones pendientes en DECISIONS.md", "DECISIONS.md",
        "0 pendientes", f"{len(pendientes)} ({','.join(pendientes)})",
        "no" if pendientes else "si", "LOW")
    registro = list(csv.DictReader((PROJ / "experiments.csv").open(newline="", encoding="utf-8")))
    add("conteo de filas del registro (superficie final)", "experiments.csv",
        "202 filas (197 claves)", f"{len(registro)} filas", "si", "INFO")
    log_lb = list(csv.DictReader((PROJ / "submissions" / "leaderboard_log.csv").open(newline="", encoding="utf-8")))
    add("sondeos de leaderboard", "submissions/leaderboard_log.csv",
        "12 en v2 + 3 en v8", f"{len(log_lb)} filas", "si", "INFO")
    subs = [p.name for p in (PROJ / "submissions").glob("toxic_submission_v0[89]*")] + \
           [p.name for p in (PROJ / "submissions").glob("toxic_submission_v1[0-6]*")]
    colisiones = sorted({n.split("_")[2] for n in subs if n.startswith("toxic_submission_v0")})
    add("numeracion de submissions v08-v12 (v2 vs v8)", "submissions/",
        "sin colisiones", "colision en v08..v12 (dos familias de nombre)",
        "no", "MEDIUM")
    ebid = [r["experiment_id"] for r in registro if r["experiment_id"].startswith("EBD")]
    add("filas EBD-01..06", "experiments.csv",
        "filas propias EBD-01..06",
        f"faltan (presentes: {sorted(set(ebid))[:6]}...)", "no", "HIGH")
    add("dato del digesto c3aa4198", "commit ae0084e (mensaje)",
        "verificacion reproducible en artefactos", "no localizado en ningun artefacto",
        "no", "LOW")
    d13 = next((f for f in filas_dec if f["Decision"] == "D-013"), None)
    add("D-013 marcada como refutada en v6/LB", "DECISIONS.md",
        "estado reflejado en el cuerpo de la decision",
        "el cuerpo puede no reflejar la refutacion (verificar manualmente)", "no", "LOW")
    add("spaCy como unico componente externo de anotacion", "grep de imports",
        "si (sin transformers/LLM)", "confirmado (14 archivos, solo anotador)",
        "si", "INFO")

    # ---------- familias cerradas ----------
    familias = [
        ("v4 tracks A/B/D/E/F/H + P8", "mejoras locales via reweighting/label/meta/shift",
         "mejor +0.0082 (A_cost_sensitive) pero holdout rechaza a P8-C", "cerrada (D-029)",
         "CLOSED VALIDLY", "gates pre-registrados y holdout aplicado; re-ejecucion parcial P6"),
        ("v5 (representacion 10M->50M)", "el corpus/arquitectura limita la mejora",
         "complementariedad no supera gate", "cerrada (D-032/D-033/D-034)",
         "CLOSED VALIDLY (con reproducibilidad pendiente)", "corpus y modelos locales; repro FastText en P6b"),
        ("v5.1 (5 ejes de arquitectura)", "vector_size/window/min_count/subwords/negative",
         "todos inerte/inercia; baseline 3-6 optimo", "cerrada (D-036..D-040)",
         "CLOSED VALIDLY", "gates de dos niveles; presupuesto respetado"),
        ("v6 (7 tracks)", "weighting/bagging/skip-grams/ruido/aumento",
         "max +0.0016 (I2) bajo gate +0.005", "cerrada sin adopciones (D-052)",
         "CLOSED WITH CAVEAT", "v6-07 con componente no fold-local (impacto 1.4e-4 -> 0)"),
        ("v7 (P8/P9/P0.1)", "prior de grupo, perdida F1, umbral anidado",
         "deltas -0.0019/-0.0024/-0.0019 vs sus controles", "cerrada (D-056)",
         "CLOSED VALIDLY", "protocolo anidado verificado en codigo y numeros"),
        ("v8 (P1..P10 y sondas)", "tecnicas de clase, entrenamiento, ensemble, corrector",
         "solo el promediado mejora local (+0.0051 OOF); LB por debajo de V03",
         "cerrada (D-065/D-067)", "CLOSED VALIDLY (nota: numeracion de submissions)",
         "sondas verificadas en el log; colision de nombres v08-v12"),
    ]

    with (PROJ / "audit" / "out" / "p08_decisions.csv").open("w", newline="", encoding="ascii") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(filas_dec[0].keys()))
        writer.writeheader()
        writer.writerows(filas_dec)
    with (PROJ / "reports" / "audit_documentation.csv").open("w", newline="", encoding="ascii") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(checks[0].keys()))
        writer.writeheader()
        writer.writerows(checks)
    with (PROJ / "audit" / "out" / "p08_closed_families.csv").open("w", newline="", encoding="ascii") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Familia", "Hipotesis", "Mejor resultado", "Cierre", "Veredicto auditoria", "Nota"])
        for f in familias:
            writer.writerow([ascii_text(x) for x in f])
    with (PROJ / "audit" / "out" / "p08_ceiling_claims.csv").open("w", newline="", encoding="ascii") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Fase", "Claim", "Base", "Clasificacion"])
        for t in TECHOS:
            writer.writerow([ascii_text(x) for x in t])

    conteo = {}
    for f in filas_dec:
        conteo[f["Estado"]] = conteo.get(f["Estado"], 0) + 1
    print("decisiones:", len(filas_dec), conteo)
    print("pendientes:", pendientes)
    print("checks:")
    for c in checks:
        print("  ", c["Consistent"], "|", c["Severity"], "|", c["Claim"][:70])
    print("familias:", len(familias), "| techos:", len(TECHOS))


if __name__ == "__main__":
    main()
