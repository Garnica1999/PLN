"""P97-95: integra la seccion 157 (P97 INVESTIGACION INDEPENDIENTE) en el notebook.

Lee los entregables `reports/p97_*`, construye una subseccion compacta con los
hashes reales y una celda de verificacion ejecutable en aislamiento (solo
lectura). Idempotente y ASCII puro.

Uso:
    /home/carlos/venv/bin/python experiments/p97_95_integrar_notebook.py
"""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RUTA_NOTEBOOK = RAIZ / "notebooks" / "10_final_pipeline.ipynb"
MARCA_SECCION = "## 157. P97 - INVESTIGACION INDEPENDIENTE"
MARCA_CODIGO = "Verificacion de la seccion 157 (P97"


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


CODIGO = '''"""Verificacion de la seccion 157 (P97, investigacion independiente).

Comprueba entregables, veredictos, registro y fuentes canonicas. No entrena.
"""

import csv
import hashlib
import json
from pathlib import Path

R = PROJECT_ROOT / "reports"
for nombre in ("p97_independent_research.md", "p97_method_registry.csv",
               "p97_novelty_matrix.csv", "p97_01_reglas_por_segmento.json",
               "p97_02_group_dro.json", "p97_03_nnpu_tr.json",
               "p97_04_replicacion.json"):
    assert (R / nombre).exists(), f"falta {nombre}"

reg = list(csv.DictReader(open(R / "p97_method_registry.csv", encoding="utf-8")))
mat = list(csv.DictReader(open(R / "p97_novelty_matrix.csv", encoding="utf-8")))
assert len(reg) == 25 and len(mat) == 8
assert sum(1 for f in mat if f["decision"].startswith("SELECCIONADA")) == 3

v01 = json.load(open(R / "p97_01_reglas_por_segmento.json", encoding="utf-8"))
v02 = json.load(open(R / "p97_02_group_dro.json", encoding="utf-8"))
v03 = json.load(open(R / "p97_03_nnpu_tr.json", encoding="utf-8"))
v04 = json.load(open(R / "p97_04_replicacion.json", encoding="utf-8"))
assert v01["veredicto_pre_registrado"] == "FALSIFICADA"
assert v02["veredicto_pre_registrado"] == "FALSIFICADA"
assert v03["veredicto_pre_registrado"] == "SENAL"
assert v04["veredicto_pre_registrado"] == "FALSIFICADA"
assert v02["control_A0"]["reproduce_v03_exacto"]

filas = list(csv.DictReader(open(PROJECT_ROOT / "experiments.csv",
                                 encoding="utf-8-sig")))
p97 = [f for f in filas if f["experiment_id"].startswith("P97")]
assert len(p97) == 12 and len(filas) == 471

for artefacto in ("features/p97_01_pred_segmentos.npz",
                  "features/p97_02_oof_dro.npz",
                  "features/p97_03_oof_nnpu.npz",
                  "features/p97_04_oof_replica.npz",
                  "features/p97_02_oof_dro_escala_invalida.npz",
                  "features/p97_03_oof_nnpu_l2_escala_invalida.npz",
                  "features/p97_03_oof_nnpu_pn_idx_invalido.npz"):
    assert (PROJECT_ROOT / artefacto).exists(), f"falta {artefacto}"

fuentes = {
    "data/raw/toxic_train.csv": "3d40e7345b588860",
    "features/oof_v03_final.npz": "559265816ce4f631",
    "experiments/baseline_v03_frozen/submission_v03.csv": "7d7478f271e6edfd",
}
for ruta, pref in fuentes.items():
    h = hashlib.sha256((PROJECT_ROOT / ruta).read_bytes()).hexdigest()
    assert h.startswith(pref), f"fuente alterada: {ruta}"

for nombre in ("p97_independent_research.md", "p97_method_registry.csv",
               "p97_novelty_matrix.csv", "p97_01_reglas_por_segmento.json",
               "p97_01_reglas_por_segmento.md", "p97_02_group_dro.json",
               "p97_02_group_dro.md", "p97_03_nnpu_tr.json", "p97_03_nnpu_tr.md",
               "p97_04_replicacion.json", "p97_04_replicacion.md"):
    datos = (R / nombre).read_bytes()
    assert all(b <= 127 for b in datos), f"byte > 127 en {nombre}"

print("P97 (investigacion independiente) verificado:")
print("  25 mecanismos externos | 8 candidatos | 3 hipotesis ejecutadas")
print("  H97-A FALSIFICADA | H97-B FALSIFICADA | H97-C SENAL/MARGINAL")
print("  H97-C replicacion FALSIFICADA | sin via hacia 0.80 con la evidencia actual")'''


def construir_seccion() -> str:
    """Construye el markdown de la seccion 157 con hashes reales."""
    h = {n: sha16(RAIZ / "reports" / n) for n in (
        "p97_independent_research.md", "p97_method_registry.csv",
        "p97_novelty_matrix.csv", "p97_03_nnpu_tr.json")}
    return (
        "## 157. P97 - INVESTIGACION INDEPENDIENTE: MECANISMOS NUEVOS HACIA 0.80 "
        "(2026-10-13)\n"
        "\n"
        "**Fase de investigacion externa con ejecucion controlada.** 16 busquedas "
        "en fuentes primarias (arXiv, ACL Anthology, PMLR, NeurIPS, JASA) "
        "cubriendo las ocho areas del encargo; 25 mecanismos registrados con "
        "clasificacion A-E; 8 candidatos evaluados; 3 hipotesis pre-registradas "
        "y ejecutadas; 4 experimentos principales, 3 controles y 2 "
        "instanciaciones invalidas preservadas; 0 submissions, 0 usos de "
        "test/holdout/leaderboard.\n"
        "\n"
        "- **H97-A (decision condicional por segmento)**: FALSIFICADA. El umbral "
        "por cuartil de longitud pierde -0.007845 frente a la regla global "
        "anidada (1/5 pliegues). El control global-nested reproduce el +0.000925 "
        "historico (P55/V56-01).\n"
        "- **H97-B (Group DRO minimax)**: FALSIFICADA. DRO 0.7720416 vs control "
        "A0 0.7772115 (-0.005170, 3/5); sin mejora de cola por grupo y con mas "
        "dispersion. La primera instanciacion (pesos sin normalizar frente a la "
        "escala de C de liblinear) colapso y quedo preservada como "
        "`*_escala_invalida`.\n"
        "- **H97-C (nnPU/PNU para TR; HS=0 como mezcla no supervisada)**: SENAL "
        "DENTRO DEL ARNES / MARGINAL. Con el mecanismo valido (dos "
        "instanciaciones invalidas previas preservadas), +0.002150 pooled con "
        "5/5 pliegues y TR +0.006450 (5/5), IC95 de clusters "
        "[-0.000461,+0.004896], por debajo del gate +0.003 y sin superar a V03 "
        "en crudo (-0.0002).\n"
        "- **H97-C replicacion (particion agrupada nueva, seed 43)**: "
        "FALSIFICADA (-0.001738, 1/5). La senal no es robusta a la particion; "
        "H97-C queda MARGINAL sin mas pruebas (parada pre-registrada).\n"
        "\n"
        "**Clasificacion final**: H97-A INVALIDATED | H97-B INVALIDATED | "
        "H97-C MARGINAL. Sin via plausible hacia 0.80 con la evidencia actual: "
        "los tres mecanismos genuinamente nuevos compatibles con las reglas "
        "fueron ejecutados y ninguno supera el gate de adopcion; V03 sigue "
        "siendo el mejor resultado verificado (corrected_cv 0.7772114525165721; "
        "LB 0.78306).\n"
        "\n"
        f"**Entregables**: p97_independent_research.md "
        f"({h['p97_independent_research.md']}), p97_method_registry.csv "
        f"({h['p97_method_registry.csv']}), p97_novelty_matrix.csv "
        f"({h['p97_novelty_matrix.csv']}), resultados P97-01..04 con figura "
        f"cada uno, 12 filas nuevas en experiments.csv (total 471).\n"
        "\n"
        "> No se confundio \"no encontramos nuevas ideas en la auditoria "
        "anterior\" con \"no quedan ideas nuevas\": se buscaron fuera del mapa "
        "previo, se registro la novedad contra el codigo y se ejecutaron las "
        "mejores hipotesis validas. La conclusion negativa es informativa: "
        "incluso la formulacion no supervisada de la semantica TR/AG, la mas "
        "prometedora del registro, solo produce una senal direccional fragil.")


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
    exec(compile(CODIGO, "<seccion 157>", "exec"), espacio)


def main() -> None:
    """Integra la seccion 157, ejecuta su verificacion y guarda el notebook."""
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
    print("Notebook actualizado con la seccion 157:")
    print(f"  celdas antes: {celdas_antes} | despues: {len(cells)}")
    print("  verificacion de la fase en aislamiento: OK")
    print("  celdas de codigo compiladas: OK | ASCII puro en lo generado: OK")


if __name__ == "__main__":
    main()
