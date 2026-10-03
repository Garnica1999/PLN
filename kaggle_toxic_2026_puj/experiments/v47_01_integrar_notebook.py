"""Integra la fase P47 (auditoria de datos) en el notebook consolidado.

Anade la seccion 43 al final de `notebooks/10_final_pipeline.ipynb`. La celda de
codigo lee `reports/p47_data_inventory.json` en tiempo de ejecucion, de modo que
ninguna cifra queda escrita a mano.

No se anade seccion de augmentation porque no se ejecuto ningun experimento de
augmentation: la auditoria demostro que ya existia (V6-08).

Uso
---
    /home/carlos/venv/bin/python experiments/v47_01_integrar_notebook.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

NOTEBOOK = PROJECT_ROOT / "notebooks" / "10_final_pipeline.ipynb"
INFORME = PROJECT_ROOT / "reports" / "p47_data_inventory.json"

TITULO = """\
## 43. P47 - Data Availability Audit (2026-09-27)

La pregunta: existe alguna fuente de datos etiquetados legalmente utilizable que
V03 no haya usado, y la augmentation train-only ya fue probada?

0 modelos, 0 submissions, 0 consultas a Kaggle. El holdout y el test se abrieron
solo para contar filas y calcular hashes; ninguna etiqueta se leyo para ninguna
decision.

No se anade seccion de augmentation porque **no se ejecuto ningun experimento**:
la auditoria demostro que la familia ya existia y estaba cerrada.
"""

CODIGO = """\
from pathlib import Path
import json

RAIZ = Path.cwd()
while not (RAIZ / "reports").exists() and RAIZ != RAIZ.parent:
    RAIZ = RAIZ.parent
with (RAIZ / "reports" / "p47_data_inventory.json").open(encoding="utf-8") as f:
    d = json.load(f)

p = d["pregunta_central"]
c = d["reconciliacion_de_cifras"]
a = d["precedente_augmentation"]
v = d["veredicto"]

print("43.1  INVENTARIO DE FUENTES OFICIALES Y DERIVADAS")
print(f"  {'path':44}{'clase':24}{'filas':>7}")
for s in d["inventario_oficial"]:
    print(f"  {s['path']:44}{s['clase']:24}{s['filas']:>7}")
print()
print("43.2  CORPUS EXTERNOS (medidos en tokens, no en filas)")
print(f"  {'id':12}{'origen':16}{'documentos':>12}{'tokens':>14}  sup. emb.")
for e in d["inventario_externo"]:
    docs = "" if e["documentos"] is None else f"{e['documentos']}"
    sup = "si" if e["legal_for_supervised_training"] else "no"
    emb = "si" if e["legal_for_embedding_training"] else "no"
    print(f"  {e['id']:12}{e['origen']:16}{docs:>12}{e['tokens']:>14}  {sup:4} {emb}")
print(f"  artefactos de embeddings en disco: {d['artefactos_embeddings_en_disco']}")
print()
print("43.3  RECONCILIACION DE LAS CIFRAS CITADAS")
for clave, x in c.items():
    print(f"  {clave:16} afirmado {x['afirmado']:>7} | real {x['real']:>7} | "
          f"{'OK' if x['coincide'] else 'CORREGIDO'}")
print()
print("43.4  HAY TRAIN ETIQUETADO SIN USAR?")
print(f"  train oficial etiquetado : {p['n_official_labeled_train']}")
print(f"  dev (usado por V03)      : {p['n_dev']}")
print(f"  holdout (agotado)        : {p['n_holdout']}")
print(f"  test oficial             : {p['n_official_test']} | con labels: "
      f"{p['n_test_tiene_labels']}")
print(f"  TRAIN SIN ASIGNAR        : {p['n_train_sin_asignar']}")
print(f"  holdout2                 : {p['n_holdout2']} filas, de las cuales "
      f"{p['n_holdout2_dentro_del_holdout']} dentro del holdout")
print()
print("43.5  PRECEDENTE DE AUGMENTATION")
print(f"  experimento: {a['experimento']} | script existe: {a['script_existe']}")
print(f"  transformaciones: {a['transformaciones']}")
print(f"  candidatos de la directiva cubiertos: "
      f"{a['coinciden_con_los_candidatos_de_la_directiva']}")
print(f"  colocacion: {a['colocacion_del_pipeline']}")
print(f"  grupos preservados: {a['grupos_preservados']} | control reproduce V03: "
      f"{a['control_reproduce_v03']}")
print(f"  resultado J1 (una copia): {a['resultado_j1_una_copia']}")
print(f"  decision: {a['decision_registrada']}")
print()
print("43.6  VEREDICTO")
print(f"  CASE                 : {v['CASE']}")
print(f"  DATA_EXPANSION       : {v['DATA_EXPANSION']}")
print(f"  TRAIN_AUGMENTATION   : {v['TRAIN_AUGMENTATION']}")
print(f"  experimento nuevo    : {v['ejecutado_experimento_nuevo']}")
print(f"  modelos / submissions / consultas: "
      f"{v['gastos']['modelos_entrenados']} / {v['gastos']['submissions']} / "
      f"{v['gastos']['consultas_kaggle']}")
"""


def markdown(texto: str) -> dict[str, Any]:
    """Construye una celda markdown.

    Args:
        texto: fuente de la celda.

    Returns:
        La celda como diccionario de notebook.
    """
    return {"cell_type": "markdown", "metadata": {}, "source": texto.splitlines(keepends=True)}


def codigo(texto: str) -> dict[str, Any]:
    """Construye una celda de codigo sin outputs.

    Args:
        texto: fuente de la celda.

    Returns:
        La celda como diccionario de notebook.
    """
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": texto.splitlines(keepends=True),
    }


def main() -> None:
    """Anade la seccion 43 al notebook consolidado.

    Returns:
        None.
    """
    if not INFORME.exists():
        raise SystemExit("PARADA: falta reports/p47_data_inventory.json")
    json.loads(INFORME.read_text(encoding="utf-8"))
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    antes = len(notebook["cells"])
    marcas = " ".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown"
    )
    if "P47 - Data Availability Audit" in marcas:
        print("  la seccion 43 ya existe, no se duplica")
        return
    notebook["cells"].extend([markdown(TITULO), codigo(CODIGO)])
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"  notebook: {antes} -> {len(notebook['cells'])} celdas")
    print("  seccion 43 integrada en el notebook consolidado")


if __name__ == "__main__":
    main()
