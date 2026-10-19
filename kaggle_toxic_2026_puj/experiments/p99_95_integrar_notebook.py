"""P99-95: integra la seccion 162 (P99: S99-01 y gate de escala).

Idempotente y ASCII puro; celda de verificacion ejecutada en aislamiento.

Uso:
    /home/carlos/venv/bin/python experiments/p99_95_integrar_notebook.py
"""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RUTA_NOTEBOOK = RAIZ / "notebooks" / "10_final_pipeline.ipynb"
MARCA_SECCION = "## 162. P99 - ESCALA"
MARCA_CODIGO = "Verificacion de la seccion 162 (P99"


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


CODIGO = '''"""Verificacion de la seccion 162 (P99: escala a 60M y gate).

No entrena nada; solo lee artefactos.
"""

import csv
import hashlib
import json
from pathlib import Path

PROY = PROJECT_ROOT
R = PROY / "reports"
man = json.load(open(PROY / "artifacts" / "p99_s99_01_manifest.json",
                     encoding="utf-8"))
assert man["tokens_finales"] > 60_000_000
assert man["tokens_nuevos_unicos"] > 30_000_000
assert man["coincidencias_exactas_vs_train"] == 0
assert man["val_presentes"] == 3300 + 20

pre = json.load(open(R / "p99_pretraining.json", encoding="utf-8"))
assert pre["checkpoints"]["ltr"]["epocas"] == 2
assert pre["checkpoints"]["rtl"]["epocas"] == 2
assert abs(pre["correccion_gauge_rtl_ep1"]["valor_corregido"] - 135.6) < 1.0
assert pre["gpu_horas_totales"] < 6.0

can = json.load(open(R / "p99_downstream_canonica.json", encoding="utf-8"))
p47 = json.load(open(R / "p99_downstream_part47.json", encoding="utf-8"))
assert can["gate"]["veredicto"] == "GATE_FAILED"
assert abs(can["comparacion"]["delta60_vs_v03"] - 0.004375) < 1e-4
assert can["comparacion"]["delta60_vs_30"] < 0
assert p47["comparacion"]["delta60_vs_v03"] < 0
assert can["control_reproduccion_30M"]["ok"]

filas = list(csv.DictReader(open(PROY / "experiments.csv",
                                 encoding="utf-8-sig")))
ids = {f["experiment_id"] for f in filas}
assert len(filas) == 485 and {"P99-00","P99-11","P99-12","P99-13","P99-14"} <= ids

fuentes = {
    "features/oof_v03_final.npz": "559265816ce4f631",
    "data/external/corpora/p98_c98_1/tokens_30m.txt": "9f562718ef4e7d45",
    "models/p98_c98_1_ltr.pt": "472891a7598a1daa",
    "models/p98_c98_1_rtl.pt": "cff68d45bf683cad",
}
for ruta, pref in fuentes.items():
    h = hashlib.sha256((PROY / ruta).read_bytes()).hexdigest()
    assert h.startswith(pref), f"fuente alterada: {ruta}"

for nombre in ("p99_preregistration.md", "p99_corpus_scaling_plan.md",
               "p99_corpus_results.csv", "p99_pretraining_results.csv",
               "p99_downstream_results.csv", "p99_complementarity.csv"):
    datos = (R / nombre).read_bytes()
    assert all(b <= 127 for b in datos), f"byte > 127 en {nombre}"

print("P99 verificada:")
print("  S99-01: 60.707.427 tokens | +30.970.244 nuevos | 0 solapamiento train")
print("  LTR ppl 100.6 | RTL ppl 98.3 | gauge ep1 corregido (135.6)")
print("  downstream: delta60 vs V03 canonica +0.004375 (4/5) pero -0.00068 vs")
print("  30M y negativo en la particion 47 -> GATE_FAILED, hipotesis de escala")
print("  CERRADA con evidencia (sin rescates).")'''


def construir_seccion() -> str:
    """Construye el markdown de la seccion 162 con metricas reales."""
    man = json.loads((RAIZ / "artifacts" / "p99_s99_01_manifest.json").read_text(
        encoding="utf-8"))
    pre = json.loads((RAIZ / "reports" / "p99_pretraining.json").read_text(
        encoding="utf-8"))
    can = json.loads((RAIZ / "reports" / "p99_downstream_canonica.json").read_text(
        encoding="utf-8"))
    p47 = json.loads((RAIZ / "reports" / "p99_downstream_part47.json").read_text(
        encoding="utf-8"))
    h = {n: sha16(RAIZ / "reports" / n) for n in (
        "p99_pretraining.json", "p99_downstream_canonica.json",
        "p99_downstream_part47.json")}
    rc = can["reglas"]; rp = p47["reglas"]
    cc = can["comparacion"]; cp = p47["comparacion"]
    return (
        "## 162. P99 - ESCALA A 60M (S99-01) Y GATE DE ESCALA (2026-10-13)\n"
        "\n"
        "**H99-01 pre-registrada** (`p99_preregistration.md` + "
        "`p99_corpus_scaling_plan.md`) ANTES de construir y entrenar. Corpus "
        f"S99-01: {man['tokens_finales']:,} tokens, "
        f"{man['documentos_finales']:,} documentos, "
        f"+{man['tokens_nuevos_unicos']:,} tokens nuevos unicos "
        "(C98-1 subconjunto integro; 0 solapamiento vs train). Composicion "
        "efectiva 69.2/16.5/14.3 (wiki/mC4/twitter): proporciones nominales "
        "60/25/15 no alcanzables con material local (tope mC4 = 10M); "
        "desviacion registrada antes de entrenar; sin descargas.\n"
        "\n"
        "**Preentrenamiento** (arquitectura y vocab CONGELADOS de C98-1; "
        "2 epocas/direccion; lr por tokens 75.1%): LTR ppl 100.6 (ep1 136.9) | "
        "RTL ppl 98.3 (ep1 135.6, gauge corregido con checkpoint preservado) | "
        f"{pre['gpu_horas_totales']} GPU-h (tope 6). Erratas documentadas: "
        "gauge RTL ep1 (orden de validacion), reconstruccion LTR desde el log "
        "persistente tras la reanudacion.\n"
        "\n"
        "**Downstream (mismas recetas de D98-1; umbrales V03)**\n"
        "\n"
        "| Particion | V03 | fusion 30M | fusion 60M | delta60 vs V03 | "
        "delta60 vs 30M |\n|---|---|---|---|---|---|\n"
        f"| Canonica | {rc['v03']['corrected_cv']:.6f} | "
        f"{rc['fusion_30M']['corrected_cv']:.6f} (control OK) | "
        f"{rc['fusion_60M']['corrected_cv']:.6f} | "
        f"{cc['delta60_vs_v03']:+.6f} ({cc['pliegues60_mejorados_vs_v03']}/5) | "
        f"{cc['delta60_vs_30']:+.6f} |\n"
        f"| Particion 47 | {rp['v03_equivalente_47']['corrected_cv']:.6f} | "
        f"{rp['fusion_30M_47']['corrected_cv']:.6f} | "
        f"{rp['fusion_60M_47']['corrected_cv']:.6f} | "
        f"{cp['delta60_vs_v03']:+.6f} ({cp['pliegues60_mejorados_vs_v03']}/5) | "
        f"{cp['delta60_vs_30']:+.6f} |\n"
        "\n"
        "**GATE_FAILED**: 60M no mejora sobre 30M (delta negativo en ambas "
        "particiones) y su delta vs V03 no sobrevive a la particion 47 "
        "(negativo). Condiciones incumplidas: positivo vs 30M y positivo en la "
        "47 (y neto 47 no positivo). Por los criterios pre-registrados de P99 "
        "se **CIERRA la hipotesis de escala**: sin rescates (no mas tokens, "
        "sin cambios de arquitectura, sin D98-2; P99 seccion 15: no se abre la "
        "hipotesis de arquitectura). La doble dosis de documentos unicos con "
        "exposicion constante no mejoro la clasificacion: mas documentos y "
        "ppl comparable no equivalen a mejor F1.\n"
        "\n"
        f"**Artefactos**: p99_pretraining.json ({h['p99_pretraining.json']}), "
        f"p99_downstream_canonica.json ({h['p99_downstream_canonica.json']}), "
        f"p99_downstream_part47.json ({h['p99_downstream_part47.json']}), "
        "p99_corpus_results.csv, p99_pretraining_results.csv, "
        "p99_downstream_results.csv, p99_complementarity.csv, "
        "features/p99_reps_dev.npz, modelos p99_s99_01_{ltr,rtl}.pt; filas "
        "P99-00/11/12/13/14 (total 485).")


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
    """Ejecuta la celda de verificacion en aislamiento."""
    espacio: dict[str, object] = {"PROJECT_ROOT": RAIZ, "__name__": "__verificacion__"}
    exec(compile(CODIGO, "<seccion 162>", "exec"), espacio)


def main() -> None:
    """Integra la seccion 162, ejecuta su verificacion y guarda el notebook."""
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
    print("Notebook actualizado con la seccion 162:")
    print(f"  celdas antes: {celdas_antes} | despues: {len(cells)}")
    print("  verificacion de la fase en aislamiento: OK")
    print("  celdas de codigo compiladas: OK | ASCII puro en lo generado: OK")


if __name__ == "__main__":
    main()
