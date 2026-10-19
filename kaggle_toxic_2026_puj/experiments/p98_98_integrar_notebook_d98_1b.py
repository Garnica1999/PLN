"""P98-98: integra la seccion 161 (P98: D98-1b replica y cierre contextual).

Idempotente y ASCII puro; celda de verificacion ejecutada en aislamiento.

Uso:
    /home/carlos/venv/bin/python experiments/p98_98_integrar_notebook_d98_1b.py
"""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RUTA_NOTEBOOK = RAIZ / "notebooks" / "10_final_pipeline.ipynb"
MARCA_SECCION = "## 161. P98 - D98-1b"
MARCA_CODIGO = "Verificacion de la seccion 161 (P98"


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


CODIGO = '''"""Verificacion de la seccion 161 (P98 D98-1b: replica y cierre).

No entrena nada; solo lee artefactos.
"""

import csv
import hashlib
import json
from pathlib import Path

R = PROJECT_ROOT / "reports"
pre = json.load(open(R / "p98_14_preregistro.json", encoding="utf-8"))
assert pre["particion"]["semilla"] == 43
assert pre["particion"]["sin_fuga_de_grupos"] is True

rep = json.load(open(R / "p98_14_replica_d98_1b.json", encoding="utf-8"))
assert rep["veredicto"] == "FALSIFICADA"
assert rep["comparacion"]["pliegues_positivos"] == 2
assert abs(rep["comparacion"]["delta_pooled"] - 0.001151) < 1e-5
assert rep["comparacion"]["bootstrap_clusters"]["ic95"][0] < 0

d98 = json.load(open(R / "p98_13_downstream.json", encoding="utf-8"))
assert d98["comparaciones"]["fusion_vs_v03"]["delta"] == 0.005055
h13 = hashlib.sha256((R / "p98_13_downstream.json").read_bytes()).hexdigest()
assert h13.startswith("56240d433e7237ad"), "D98-1 alterado"

filas = list(csv.DictReader(open(PROJECT_ROOT / "experiments.csv",
                                 encoding="utf-8-sig")))
assert len(filas) == 480
assert any(f["experiment_id"] == "P98-15" for f in filas)
assert not (PROJECT_ROOT / "reports" / "p98_15").exists()

plan = (R / "p98_plan_seguimiento.md").read_text(encoding="utf-8")
assert "FALSIFICADA" in plan and "CERRADO" in plan

fuentes = {
    "features/oof_v03_final.npz": "559265816ce4f631",
    "data/raw/toxic_train.csv": "3d40e7345b588860",
}
for ruta, pref in fuentes.items():
    h = hashlib.sha256((PROJECT_ROOT / ruta).read_bytes()).hexdigest()
    assert h.startswith(pref), f"fuente alterada: {ruta}"

for nombre in ("p98_14_preregistro.json", "p98_14_replica_d98_1b.json",
               "p98_14_replica_d98_1b.md", "p98_downstream_results_b.csv",
               "p98_complementarity_b.csv"):
    datos = (R / nombre).read_bytes()
    assert all(b <= 127 for b in datos), f"byte > 127 en {nombre}"

print("P98 D98-1b verificada:")
print("  canonica: +0.005055 (4/5) | replica 47: +0.001151 (2/5) -> FALSIFICADA")
print("  linea contextual actual CERRADA por criterios pre-registrados")
print("  D98-1 intacto | 480 filas | sin D98-2 | sin submissions")'''


def construir_seccion() -> str:
    """Construye el markdown de la seccion 161 con metricas reales."""
    rep = json.loads((RAIZ / "reports" / "p98_14_replica_d98_1b.json").read_text(
        encoding="utf-8"))
    r = rep["reglas"]
    c = rep["comparacion"]
    h = sha16(RAIZ / "reports" / "p98_14_replica_d98_1b.json")
    return (
        "## 161. P98 - D98-1b: REPLICA DE ROBUSTEZ EN SEGUNDA PARTICION "
        "(2026-10-13)\n"
        "\n"
        "**Autorizado con condiciones estrictas**; pre-registro de folds "
        "(`p98_14_preregistro.json`: semilla 43, tamanos [297, 297, 297, 296, "
        "296], sin fuga de grupos, hash de folds registrado ANTES de evaluar). "
        "Todo congelado de D98-1: reps, fusion, umbrales; sin blendings; sin "
        "test/holdouts/leaderboard.\n"
        "\n"
        "| Regla | corrected_cv | delta | F1 HS/TR/AG |\n|---|---|---|---|\n"
        f"| V03-equivalente (part47) | "
        f"{r['v03_equivalente_part47']['corrected_cv']:.10f} | - | "
        f"{r['v03_equivalente_part47']['f1_por_etiqueta']['HS']:.4f}/"
        f"{r['v03_equivalente_part47']['f1_por_etiqueta']['TR']:.4f}/"
        f"{r['v03_equivalente_part47']['f1_por_etiqueta']['AG']:.4f} |\n"
        f"| Fusion contextual (part47) | "
        f"{r['fusion_contextual_part47']['corrected_cv']:.10f} | "
        f"{c['delta_pooled']:+.6f} | "
        f"{r['fusion_contextual_part47']['f1_por_etiqueta']['HS']:.4f}/"
        f"{r['fusion_contextual_part47']['f1_por_etiqueta']['TR']:.4f}/"
        f"{r['fusion_contextual_part47']['f1_por_etiqueta']['AG']:.4f} |\n"
        "\n"
        f"- Pliegues positivos: {c['pliegues_positivos']}/5; neto de celdas "
        f"{c['neto_celdas']:+d}; IC95 clusters "
        f"{c['bootstrap_clusters']['ic95']} (incluye cero).\n"
        f"- Canonica (D98-1): fusion 0.782266, delta +0.005055, 4/5, IC95 "
        "[-0.004058, +0.014721]. El patron por slices tampoco replico: "
        "q4 +0.011 (era +0.027), url -0.005 y hashtag -0.009 (eran positivos).\n"
        "\n"
        "**Veredicto: FALSIFICADA.** Por los criterios pre-registrados del "
        "propietario (falsificacion = delta <= 0 o < 4/5 pliegues o neto <= 0), "
        "la senal contextual de D98-1 NO sobrevive al cambio de particion y se "
        "CIERRA la linea contextual actual: sin escalar el corpus, sin "
        "incrementar complejidad y sin D98-2. Consistente con el freno "
        "historico del proyecto (deltas locales < +0.02 no transfieren).\n"
        "\n"
        f"**Artefactos**: p98_14_preregistro.json, p98_14_replica_d98_1b.json "
        f"({h}), p98_downstream_results_b.csv, p98_complementarity_b.csv, "
        "features/p98_14_oof_b.npz, figura p98_14_replica.png; fila P98-15 "
        "(total 480). D98-1 y el OOF canonico permanecen intactos.")


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
    exec(compile(CODIGO, "<seccion 161>", "exec"), espacio)


def main() -> None:
    """Integra la seccion 161, ejecuta su verificacion y guarda el notebook."""
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
    print("Notebook actualizado con la seccion 161:")
    print(f"  celdas antes: {celdas_antes} | despues: {len(cells)}")
    print("  verificacion de la fase en aislamiento: OK")
    print("  celdas de codigo compiladas: OK | ASCII puro en lo generado: OK")


if __name__ == "__main__":
    main()
