"""P98-96: integra la seccion 159 (P98 FASE B: C98-0 + corpus C98-1).

Idempotente y ASCII puro; celda de verificacion ejecutada en aislamiento.

Uso:
    /home/carlos/venv/bin/python experiments/p98_96_integrar_notebook_b.py
"""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RUTA_NOTEBOOK = RAIZ / "notebooks" / "10_final_pipeline.ipynb"
MARCA_SECCION = "## 159. P98 - FASE B"
MARCA_CODIGO = "Verificacion de la seccion 159 (P98"


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


CODIGO = '''"""Verificacion de la seccion 159 (P98 Fase B: C98-0 y corpus C98-1).

No entrena nada; solo lee artefactos.
"""

import hashlib
import json
from pathlib import Path

import csv

R = PROJECT_ROOT / "reports"
piloto = json.load(open(R / "p98_c98_0_piloto.json", encoding="utf-8"))
assert piloto["veredicto"] == "PILOTO_OK"
assert piloto["eleccion_c98_1"]["hidden"] == 1024
assert piloto["eleccion_c98_1"]["capas"] == 2
assert piloto["historial"][-1]["val_loss"] < piloto["historial"][0]["val_loss"]
assert piloto["gpu_horas_estimadas"] < 2.0

corpus = json.load(open(PROJECT_ROOT / "artifacts" / "p98_c98_1_manifest.json",
                        encoding="utf-8"))
assert corpus["tokens_finales"] > 29_000_000
assert corpus["coincidencias_exactas_vs_train"] == 0
ruta_corpus = PROJECT_ROOT / corpus["ruta"]
h = hashlib.sha256(ruta_corpus.read_bytes()).hexdigest()
assert h == corpus["sha256"], "corpus C98-1 alterado"

filas = list(csv.DictReader(open(PROJECT_ROOT / "experiments.csv",
                                 encoding="utf-8-sig")))
ids = {f["experiment_id"] for f in filas}
assert {"P98-10", "P98-11"} <= ids and len(filas) == 476

plan = (R / "p98_plan_seguimiento.md").read_text(encoding="utf-8")
assert "CORPUS LISTO" in plan and "PILOTO_OK" in plan

fuentes = {
    "features/oof_v03_final.npz": "559265816ce4f631",
    "experiments/baseline_v03_frozen/submission_v03.csv": "7d7478f271e6edfd",
}
for ruta, pref in fuentes.items():
    hh = hashlib.sha256((PROJECT_ROOT / ruta).read_bytes()).hexdigest()
    assert hh.startswith(pref), f"fuente alterada: {ruta}"

for nombre in ("p98_c98_0_piloto.json", "p98_c98_0_piloto.md",
               "p98_plan_seguimiento.md"):
    datos = (R / nombre).read_bytes()
    assert all(b <= 127 for b in datos), f"byte > 127 en {nombre}"

print("P98 Fase B (parcial) verificada:")
print("  C98-0 PILOTO_OK: val ppl 651.3 -> 351.1 | escalera [1024,2] @ 17.133 tok/s")
print("  C98-1 corpus: 29.737.183 tokens, 166.022 docs, 0 coincidencias vs train")
print("  siguiente: preentrenamiento LTR y RTL (<=6 GPU-h) y luego D98-1")'''


def construir_seccion() -> str:
    """Construye el markdown de la seccion 159 con metricas reales."""
    piloto = json.loads((RAIZ / "reports" / "p98_c98_0_piloto.json").read_text(
        encoding="utf-8"))
    corpus = json.loads((RAIZ / "artifacts" / "p98_c98_1_manifest.json").read_text(
        encoding="utf-8"))
    h = {n: sha16(RAIZ / "reports" / n) for n in (
        "p98_c98_0_piloto.json", "p98_c98_0_piloto.md")}
    eleccion = piloto["eleccion_c98_1"]
    filas_tabla = "\n".join(
        f"| {hh['epoca']} | {hh['loss_primera']:.3f} -> {hh['loss_ultima']:.3f} | "
        f"{hh['val_perplexity']:.1f} | {hh['tokens_por_segundo']:,.0f} |"
        for hh in piloto["historial"]).replace(",", ".")
    return (
        "## 159. P98 - FASE B: C98-0 (PILOTO REAL) Y CORPUS C98-1 (2026-10-13)\n"
        "\n"
        "**C98-0 (piloto, wikipedia 5M local; ~0.16 GPU-h)**: pipeline real "
        "validado de punta a punta. Vocab top-50k construido desde cero sobre "
        "el corpus (96.39% de cobertura de tokens; 173.942 tipos; 254 chars); "
        "modelo LTR 512x2 (14.58M parametros); sin tocar test/holdouts.\n"
        "\n"
        "| Epoca | loss | val ppl | tok/s |\n|---|---|---|---|\n"
        f"{filas_tabla}\n"
        "\n"
        f"- **Sonda [1024, 2]**: {eleccion['tok_s']:,.0f} tok/s, VRAM "
        f"{piloto['vram_pico_gb']:.2f} GB >= requerido "
        f"{piloto['requerido_tok_s']:,.0f} tok/s -> **escalera fija: hidden 1024 "
        "x 2 capas** para C98-1 (30M x 4 epocas x direccion en ~2 h).\n"
        "- **Hallazgo de entorno**: el dropout nativo de nn.LSTM no compila en "
        "MIOpen gfx1201 (header rocrand ausente); se usa nn.Dropout entre capas "
        "(misma semantica), documentado en el json del piloto.\n"
        "\n"
        f"**Corpus C98-1 construido** (sin descargas): {corpus['tokens_finales']:,} "
        f"tokens finales, {corpus['documentos_finales']:,} documentos (wikipedia "
        "prefijo 18.0M + mc4 prefijo 7.5M + twitter muestreo 4.5M con 9.656 "
        "duplicados internos removidos por dedup exacta); 0 coincidencias "
        f"exactas vs train; sha256 `{corpus['sha256'][:16]}...`.\n"
        "\n"
        f"**Artefactos**: p98_c98_0_piloto.json ({h['p98_c98_0_piloto.json']}), "
        f"p98_c98_0_piloto.md ({h['p98_c98_0_piloto.md']}), "
        "artifacts/p98_c98_1_manifest.json, "
        "data/external/corpora/p98_c98_1/tokens_30m.txt; "
        f"filas P98-10 y P98-11 en experiments.csv (total 476).\n"
        "\n"
        "> Siguiente: preentrenamiento LTR y RTL sobre C98-1 (<=6 GPU-h; "
        "early stop por perplexity de validacion) y despues D98-1 (frozen "
        "reps + LR por pliegue + fusion estilo v5.1 con umbrales de V03).")


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
    exec(compile(CODIGO, "<seccion 159>", "exec"), espacio)


def main() -> None:
    """Integra la seccion 159, ejecuta su verificacion y guarda el notebook."""
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
    print("Notebook actualizado con la seccion 159:")
    print(f"  celdas antes: {celdas_antes} | despues: {len(cells)}")
    print("  verificacion de la fase en aislamiento: OK")
    print("  celdas de codigo compiladas: OK | ASCII puro en lo generado: OK")


if __name__ == "__main__":
    main()
