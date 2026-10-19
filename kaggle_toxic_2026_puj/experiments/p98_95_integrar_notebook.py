"""P98-95: integra la seccion 158 (P98 FASE A) en el notebook.

Lee los entregables `reports/p98_*` y `artifacts/p98_*`, construye una
subseccion compacta con hashes reales y una celda de verificacion ejecutable en
aislamiento (solo lectura). Idempotente y ASCII puro.

Uso:
    /home/carlos/venv/bin/python experiments/p98_95_integrar_notebook.py
"""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RUTA_NOTEBOOK = RAIZ / "notebooks" / "10_final_pipeline.ipynb"
MARCA_SECCION = "## 158. P98 - FASE A"
MARCA_CODIGO = "Verificacion de la seccion 158 (P98"


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


CODIGO = '''"""Verificacion de la seccion 158 (P98 Fase A: gobernanza e infraestructura).

Comprueba registros, cuarentenas, auditorias, prueba de infraestructura y
fuentes canonicas. No entrena nada.
"""

import csv
import hashlib
import json
from pathlib import Path

R = PROJECT_ROOT / "reports"
entregables = ("p98_corpus_registry.csv", "p98_corpus_licenses.md",
               "p98_architecture.md", "p98_infra_test.json",
               "p98_auditoria_solapamiento.json", "p98_plan_seguimiento.md")
for nombre in entregables:
    assert (R / nombre).exists(), f"falta {nombre}"
assert (PROJECT_ROOT / "artifacts" / "p98_corpus_manifest.json").exists()
assert (PROJECT_ROOT / "docs" / "p98_provenance_autorizacion_bilstm.md").exists()

reg = list(csv.DictReader(open(R / "p98_corpus_registry.csv", encoding="utf-8")))
assert len(reg) == 17
estados = {}
for f in reg:
    estados[f["permission_status"]] = estados.get(f["permission_status"], 0) + 1
assert estados["ALLOWED"] == 4 and estados["ALLOWED_WITH_RESTRICTIONS"] == 4
assert estados["NEEDS_EXPLICIT_PERMISSION"] == 9
cuarentena = [f for f in reg if f["source_id"] in
              ("C98-SRC-40", "C98-SRC-41", "C98-SRC-42", "C98-SRC-43",
               "C98-SRC-44")]
assert len(cuarentena) == 5 and all(
    "CUARENTENA" in f["notas"] for f in cuarentena)

infra = json.load(open(R / "p98_infra_test.json", encoding="utf-8"))
assert infra["checkpoint_roundtrip_ok"] and infra["determinismo_delta_30_pasos"] == 0.0
assert infra["converge"] and infra["tokens_por_segundo"] > 10000

audit = json.load(open(R / "p98_auditoria_solapamiento.json", encoding="utf-8"))
assert audit["veredicto"] == "CUMPLE"
assert audit["coincidencias_exactas_total"] == 0
assert all(d["categoria"] == "puntuacion"
           for d in audit["diagnostico_coincidencias"])

man = json.load(open(PROJECT_ROOT / "artifacts" / "p98_corpus_manifest.json",
                     encoding="utf-8"))
assert man["plan_estado"]["fase_A"]["infra_sintetica"].startswith("hecha")
assert man["plan_estado"]["fase_A"]["auditoria_solapamiento"].startswith("CUMPLE")

filas = list(csv.DictReader(open(PROJECT_ROOT / "experiments.csv",
                                 encoding="utf-8-sig")))
p98 = [f for f in filas if f["experiment_id"].startswith("P98")]
assert len(p98) == 3 and len(filas) == 474

fuentes = {
    "data/raw/toxic_train.csv": "3d40e7345b588860",
    "features/oof_v03_final.npz": "559265816ce4f631",
    "experiments/baseline_v03_frozen/submission_v03.csv": "7d7478f271e6edfd",
}
for ruta, pref in fuentes.items():
    h = hashlib.sha256((PROJECT_ROOT / ruta).read_bytes()).hexdigest()
    assert h.startswith(pref), f"fuente alterada: {ruta}"

for nombre in entregables:
    datos = (R / nombre).read_bytes()
    assert all(b <= 127 for b in datos), f"byte > 127 en {nombre}"

print("P98 Fase A verificada:")
print("  17 fuentes registradas | 5 cuarentenas de odio activas")
print("  auditoria de solapamiento CUMPLE | infra ROCm OK (determinismo exacto)")
print("  arquitectura v1 pre-registrada | 3 filas nuevas en experiments.csv")'''


def construir_seccion() -> str:
    """Construye el markdown de la seccion 158 con hashes y metricas reales."""
    h = {n: sha16(RAIZ / "reports" / n) for n in (
        "p98_corpus_registry.csv", "p98_corpus_licenses.md",
        "p98_architecture.md", "p98_auditoria_solapamiento.json")}
    infra = json.loads((RAIZ / "reports" / "p98_infra_test.json").read_text(
        encoding="utf-8"))
    tok_s = f"{infra['tokens_por_segundo']:,.0f}".replace(",", ".")
    return (
        "## 158. P98 - FASE A: GOBERNANZA DE CORPUS E INFRAESTRUCTURA "
        "(2026-10-13)\n"
        "\n"
        "**Autorizacion**: el mecanismo (BiLSTM desde cero, objetivo secuencial "
        "bidireccional, corpus externos no etiquetados no-competencia) queda "
        "confirmado por el propietario (nota de provenance en "
        "`docs/p98_provenance_autorizacion_bilstm.md`); la restriccion del "
        "profesor cubre transformers/LLM con pesos preentrenados, no un modelo "
        "desde cero. Cuarentena de corpus de odio vigente (no se libera con "
        "resultados positivos).\n"
        "\n"
        "- **Registro de fuentes** (17 filas): Wikipedia es (CC BY-SA 4.0), mC4 "
        "(ODC-BY 1.0), twitter_domain COVID-es (CC BY-SA 4.0, 361.994 docs, "
        "autorizacion + auditoria de solapamiento) = ALLOWED; mixto/libros = "
        "ALLOWED_WITH_RESTRICTIONS; OSCAR/CulturaX/HPLT (familia Common Crawl "
        "unica, sin descarga), TASS y 5 corpus de odio = "
        "NEEDS_EXPLICIT_PERMISSION. 6 hashes de corpus re-verificados contra "
        "manifiesto v5/P73.\n"
        "- **Auditoria de solapamiento vs train** (P98-02): **CUMPLE** - 0 "
        "coincidencias exactas a nivel de documento en los 4 corpus; la unica "
        "coincidencia de 8-gramas (twitter completo, 10.3M tokens) resulto ser "
        "`\".\".\".\"` (puntuacion, sin contenido). Frente a toxic_test no se "
        "escanea (regla vigente; incertidumbre residual documentada).\n"
        "- **Infraestructura** (P98-01, datos sinteticos): ROCm estable; loss "
        f"{infra['perdida_primera']:.3f} -> {infra['perdida_ultima']:.3f} (val "
        f"{infra['val_loss']:.3f}); {tok_s} tokens/s (corrida en caliente; la "
        "primera corrida midio 50.805 por cache de kernels ROCm); VRAM "
        f"{infra['vram_pico_gb']:.2f} GB; checkpoint roundtrip identico; "
        "**determinismo exacto** (delta 0.0 en dos corridas de 30 pasos).\n"
        "- **Arquitectura v1 pre-registrada** (`p98_architecture.md`): "
        "char-aware (conv 3/5) + LSTM 2x1024 por direccion + softmax atado "
        "(~40M parametros); escalera de dimensionado por piloto; C98-1 = 30M "
        "tokens (wikipedia 60% / mC4 25% / twitter 15%), dedup cruzada; D98-1 = "
        "reps congeladas + LR por pliegue + fusion estilo v5.1.\n"
        "\n"
        f"**Entregables**: p98_corpus_registry.csv "
        f"({h['p98_corpus_registry.csv']}), p98_corpus_licenses.md "
        f"({h['p98_corpus_licenses.md']}), p98_architecture.md "
        f"({h['p98_architecture.md']}), p98_auditoria_solapamiento.json "
        f"({h['p98_auditoria_solapamiento.json']}), p98_infra_test.json, "
        "p98_plan_seguimiento.md, artifacts/p98_corpus_manifest.json; 3 filas "
        "nuevas en experiments.csv (total 474).\n"
        "\n"
        "> Proximo: C98-0 (piloto real sobre wikipedia 5M local, <=2 GPU-h) "
        "para fijar la escalera de dimensiones; despues C98-1 (corpus 30M + "
        "preentrenamiento LTR/RTL) y D98-1 (evaluacion downstream vs V03 y "
        "control estatico v5.1).")


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
    exec(compile(CODIGO, "<seccion 158>", "exec"), espacio)


def main() -> None:
    """Integra la seccion 158, ejecuta su verificacion y guarda el notebook."""
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
    print("Notebook actualizado con la seccion 158:")
    print(f"  celdas antes: {celdas_antes} | despues: {len(cells)}")
    print("  verificacion de la fase en aislamiento: OK")
    print("  celdas de codigo compiladas: OK | ASCII puro en lo generado: OK")


if __name__ == "__main__":
    main()
