"""P96-05: integra la seccion 156 (P96 AUDITORIA FORENSE) en el notebook.

Lee los entregables `reports/p96_*`, construye una subseccion compacta con los
hashes reales de esta corrida y una celda de verificacion ejecutable en
aislamiento (solo lectura del proyecto). Idempotente y ASCII puro.

Uso:
    /home/carlos/venv/bin/python experiments/p96_05_integrar_notebook.py
"""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RUTA_NOTEBOOK = RAIZ / "notebooks" / "10_final_pipeline.ipynb"
MARCA_SECCION = "## 156. P96 - AUDITORIA FORENSE INTEGRAL"
MARCA_CODIGO = "Verificacion de la seccion 156 (P96"


def sha16(ruta: Path) -> str:
    """sha256 truncado a 16 hexdigitos."""
    x = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            x.update(bloque)
    return x.hexdigest()[:16]


def nueva_celda_markdown(texto: str) -> dict:
    """Celda de markdown con identificador unico."""
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {},
            "source": texto.splitlines(keepends=True)}


def nueva_celda_codigo(texto: str) -> dict:
    """Celda de codigo vacia de salidas y sin ejecutar."""
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
            "execution_count": None, "outputs": [],
            "source": texto.splitlines(keepends=True)}


CODIGO = '''"""Verificacion de la seccion 156 (P96, auditoria forense integral).

Comprueba los entregables de la fase, la integridad de las fuentes canonicas y
que V03 se reproduce exactamente. No entrena nada.
"""

import csv
import hashlib
import json
from pathlib import Path

R = PROJECT_ROOT / "reports"
ENTREGABLES = ("p96_forensic_audit.md", "p96_experiment_integrity_matrix.csv",
               "p96_code_and_metric_findings.csv", "p96_reproducibility_matrix.csv",
               "p96_validation_and_leakage_audit.md",
               "p96_historical_conclusions_review.md",
               "p96_reproduction_candidates.csv", "p96_manifest.json")
for nombre in ENTREGABLES:
    assert (R / nombre).exists(), f"falta {nombre}"


def leer(nombre):
    """Lee un CSV de reportes con encabezado estandar."""
    with open(R / nombre, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


integridad = leer("p96_experiment_integrity_matrix.csv")
hallazgos = leer("p96_code_and_metric_findings.csv")
matriz = leer("p96_reproducibility_matrix.csv")
repro = leer("p96_reproduction_candidates.csv")
assert len(integridad) == 459 and len({f["experiment_id"] for f in integridad}) == 459
assert len(hallazgos) == 17
assert all(f["severity"] not in ("P0", "P1") for f in hallazgos)
assert len(repro) == 2 and all("condicional" in f["experiment_id"] for f in repro)
estados = {}
for f in matriz:
    estados[f["status"]] = estados.get(f["status"], 0) + 1
assert estados.get("NUMERIC_MISMATCH", 0) == 0
assert estados.get("COMPARED_WITH_DOCUMENTED_EXACTLY") == 20
assert estados.get("EXCLUDED_BY_POLICY") == 5

manifest = json.load(open(R / "p96_manifest.json", encoding="utf-8"))
assert all(manifest["canonicos_intactos"].values())
assert len(manifest["bootstrap"]) == 4

folds = json.load(open(R / "p96_folds_evidence.json", encoding="utf-8"))
assert folds["pliegues"]["particion_identica_tras_traducir_numeracion"]
assert folds["pliegues"]["pliegues_con_grupos_compartidos"] == []
assert folds["grupos"]["iguales_etiqueta_a_etiqueta"]

import numpy as np

z = np.load(PROJECT_ROOT / "features" / "oof_v03_final.npz", allow_pickle=True)
y = z["y_true"].astype(int)
pred = (z["proba"] >= z["umbrales"]).astype(int)
from sklearn.metrics import f1_score

f1 = float(np.mean([f1_score(y[:, i], pred[:, i], zero_division=0) for i in range(3)]))
assert f1 == 0.7772114525165721, f"V03 no reproduce: {f1!r}"

fuentes = {
    "data/raw/toxic_train.csv": "3d40e7345b588860",
    "features/oof_v03_final.npz": "559265816ce4f631",
    "experiments/baseline_v03_frozen/submission_v03.csv": "7d7478f271e6edfd",
    "experiments/baseline_v03_frozen/model_v03.joblib": "263ed5bd0fc0de7c",
    "reports/p93r5_label_correction_candidates.csv": "a884cf91467aa3ea",
}
for ruta, pref in fuentes.items():
    h = hashlib.sha256((PROJECT_ROOT / ruta).read_bytes()).hexdigest()
    assert h.startswith(pref), f"fuente alterada: {ruta}"

for nombre in ENTREGABLES:
    datos = (R / nombre).read_bytes()
    assert all(b <= 127 for b in datos), f"byte > 127 en {nombre}"

print("P96 (auditoria forense integral) verificado:")
print("  V03 reproduce exacto: 0.7772114525165721 | folds identicos | 20/20 curadas")
print("  17 hallazgos (0 P0/P1) | 2 reproducciones condicionales | sin suspensiones")
print("  AUDIT_CLEAN_WITHIN_VERIFIED_SCOPE | MODEL_EXPLORATION_REMAINS_PAUSED")'''


def construir_seccion() -> str:
    """Construye el markdown de la seccion 156 con hashes reales."""
    h = {n: sha16(RAIZ / "reports" / n) for n in (
        "p96_forensic_audit.md", "p96_reproducibility_matrix.csv",
        "p96_code_and_metric_findings.csv", "p96_manifest.json")}
    return (
        "## 156. P96 - AUDITORIA FORENSE INTEGRAL (2026-10-13)\n"
        "\n"
        "**Fase forense de solo lectura, sin entrenamiento, sin predicciones y sin "
        "accesos a particiones reservadas.** Objetivo: determinar si la evidencia "
        "acumulada justifica las conclusiones historicas, verificando codigo, "
        "metricas, comparabilidad y reproducibilidad contra los artefactos reales.\n"
        "\n"
        "- **P0/P1**: 459 filas de registro confirmadas (459 ids unicos, 0 "
        "duplicados); 434/459 con script existente; las 25 sin script real son "
        "auditorias/envios/demonstraciones (ninguna fila de modelado).\n"
        "- **P2/P3**: la cadena canonica de V03 (04_refit -> 14_oof -> "
        "23_candidatos -> oof_v03_final, con autoverificacion que aborta si no "
        "reproduce) queda verificada de punta a punta; la metrica canonica "
        "reproduce `0.7772114525165721` exacto (por etiqueta y por pliegue), con "
        "`y_true` alineado a las etiquetas originales.\n"
        "- **P4**: particion 1731 = 1483 + 248 completa y disjunta; grupos "
        "fusionados recomputados identicos etiqueta a etiqueta; pliegues "
        "corregidos con la misma particion (numeracion 0..4 vs 1..5, biyeccion); "
        "0 grupos compartidos; 0 vecindades de test con fit; artefactos que "
        "cubren holdout excluidos por politica. `NO_LEAKAGE_FOUND_IN_AUDITED_SCOPE`.\n"
        "- **P5/P10**: 20/20 comparaciones curadas reproducen valores "
        "documentados (V4 A/B/C/D, meta_shift x4 x2, P84-01 semillas 42/43/44, "
        "P84-02 control x2, P84-03 control=V03, V03@0.5) y 159 variantes "
        "recomputadas sin discrepancia; la submission de V32-05 es byte-identica "
        "a V03 (no transfirio porque no cambio ninguna decision).\n"
        "- **P8**: la banda de 0.017 (D-070), el gate de tres cifras (D-071) y el "
        "techo de +0.0376 (regresion sobre 19 puntos; D-291) quedan documentados "
        "como derivados; se verifico ademas que el criterio de IC de la fase V4 "
        "uso un estimando proxy (F96-17) y que el bootstrap de `src` remuestrea "
        "filas y no clusters (F96-06); ninguno de los dos cambia una direccion de "
        "cierre.\n"
        "- **Hallazgos**: 17 registrados (8 P4, 5 P3, 4 P2; **0 P0/P1**); 2 "
        "reproducciones futuras condicionales (V6 provenance; IC por clusters) y "
        "ninguna justificada por un defecto confirmado. Ninguna conclusion "
        "historica se suspende; el benchmark interno queda reforzado.\n"
        "\n"
        f"**Entregables**: p96_forensic_audit.md ({h['p96_forensic_audit.md']}), "
        f"p96_reproducibility_matrix.csv ({h['p96_reproducibility_matrix.csv']}), "
        f"p96_code_and_metric_findings.csv ({h['p96_code_and_metric_findings.csv']}), "
        f"p96_manifest.json ({h['p96_manifest.json']}) y companions.\n"
        "\n"
        "**Decision**: `AUDIT_CLEAN_WITHIN_VERIFIED_SCOPE` + "
        "`MODEL_EXPLORATION_REMAINS_PAUSED` + `V03 = FROZEN` + "
        "`NO_TRAINING_AUTHORIZED` + `NO_SUBMISSIONS_AUTHORIZED`.\n"
        "\n"
        "> La pregunta no era cuantos experimentos terminaron negativos, sino "
        "cuantos de esos resultados constituyen evidencia fiable, comparable y "
        "reproducible. La respuesta, en el alcance auditado, es: la evidencia "
        "central lo es.")


def validar(cells: list[dict]) -> None:
    """Compila las celdas de codigo y exige ASCII puro en lo generado."""
    for indice, celda in enumerate(cells):
        fuente = "".join(celda.get("source", []))
        if celda["cell_type"] == "code":
            compile(fuente, f"<celda {indice}>", "exec")
    if any(ord(c) > 127 for c in CODIGO):
        raise SystemExit("Byte > 127 en el texto generado (codigo)")
    if any(b > 127 for b in Path(__file__).read_bytes()):
        raise SystemExit("El propio script contiene bytes > 127")


def ejecutar_verificacion() -> None:
    """Ejecuta la celda de verificacion en aislamiento (solo lee el proyecto)."""
    espacio: dict[str, object] = {"PROJECT_ROOT": RAIZ, "__name__": "__verificacion__"}
    exec(compile(CODIGO, "<seccion 156>", "exec"), espacio)


def main() -> None:
    """Integra la seccion 156, ejecuta su verificacion y guarda el notebook."""
    seccion = construir_seccion()
    if any(ord(c) > 127 for c in seccion):
        raise SystemExit("Byte > 127 en el texto generado (seccion)")
    ejecutar_verificacion()
    notebook = json.loads(RUTA_NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    celdas_antes = len(cells)
    while any("".join(c.get("source", [])).startswith(MARCA_SECCION) for c in cells):
        cells.pop(next(i for i, c in enumerate(cells)
                       if "".join(c.get("source", [])).startswith(MARCA_SECCION)))
    while any(MARCA_CODIGO in "".join(c.get("source", [])) for c in cells):
        cells.pop(next(i for i, c in enumerate(cells)
                       if MARCA_CODIGO in "".join(c.get("source", []))))
    cells.append(nueva_celda_markdown(seccion))
    cells.append(nueva_celda_codigo(CODIGO))
    validar(cells)
    notebook["cells"] = cells
    RUTA_NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                             encoding="utf-8")
    print("Notebook actualizado con la seccion 156:")
    print(f"  celdas antes: {celdas_antes} | despues: {len(cells)}")
    print("  verificacion de la fase en aislamiento: OK")
    print("  celdas de codigo compiladas: OK | ASCII puro en lo generado: OK")


if __name__ == "__main__":
    main()
