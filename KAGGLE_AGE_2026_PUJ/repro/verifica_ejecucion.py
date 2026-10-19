"""Verificador post-ejecucion del notebook final AGE.

Dos modos:
  --checklist <ejecutado.ipynb>   comprueba los valores clave contra la documentacion (con tolerancia).
  --comparar <a.ipynb> <b.ipynb>  compara los valores clave de dos ejecuciones (idempotencia).

Extrae los textos de las salidas del notebook y busca cada valor con tolerancia numerica.
ASCII puro.

Uso:
    /home/carlos/venv/bin/python repro/verifica_ejecucion.py --checklist ejecutado.ipynb
    /home/carlos/venv/bin/python repro/verifica_ejecucion.py --comparar corrida1.ipynb corrida2.ipynb
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

#: (etiqueta, patron regex con un grupo numerico, valor esperado, tolerancia)
CHECKLIST = [
    ("folds embebidos == regenerados", r"coincidencia folds regenerados vs embebidos: ([0-9.]+)", 1.0, 0.0),
    ("V20 OOF (doc 0.6097782)", r"V20 \(cuartos\)\s+macro-F1 ([0-9.]+)", 0.6097782, 1e-3),
    ("V29 individual (doc 0.6121698)", r"V29 individual\s+: ([0-9.]+)", 0.6121698, 1e-3),
    ("V30 individual (doc 0.6122387)", r"V30 individual\s+: ([0-9.]+)", 0.6122387, 1e-3),
    ("V30 mezcla (doc 0.6134471)", r"V30 mezcla\s+: ([0-9.]+)", 0.6134471, 1e-3),
    ("V30 + plug-in (doc 0.6159732487657944)", r"V30 \+ plug-in\s+: ([0-9.]+)", 0.6159732487657944, 1e-3),
    ("V30 segunda (doc 0.6144120)", r"V30 segunda\s+: ([0-9.]+)", 0.6144120, 1e-3),
    ("V30 + plug-in segunda (doc 0.6164700)", r"V30 \+ plug-in segunda: ([0-9.]+)", 0.6164700, 1e-3),
    ("V48 OOF (doc 0.6186451)", r"V48 reconstruido \(OOF, con plug-in\): ([0-9.]+)", 0.6186451, 2e-3),
    ("V53 consenso (doc 0.6188259)", r"V53 CV ([0-9.]+)", 0.6188259, 2e-3),
    ("V54 consenso (doc 0.6188887)", r"V54 CV ([0-9.]+)", 0.6188887, 2e-3),
    ("V56 consenso (doc 0.6189929)", r"V56 CV ([0-9.]+)", 0.6189929, 2e-3),
    ("coincidencia con la submission sellada", r"coincidencia con la submission sellada: [0-9,]+ de [0-9,]+ \(([0-9.]+)%\)", None, None),
]

#: Valores clave que se extraen en ambos modos para comparar idempotencia.
CLAVES = [
    ("folds", r"coincidencia folds regenerados vs embebidos: ([0-9.]+)"),
    ("v20_oof", r"V20 \(cuartos\)\s+macro-F1 ([0-9.]+)"),
    ("v30_oof", r"V30 mezcla\s+: ([0-9.]+)"),
    ("plugin_oof", r"V30 \+ plug-in\s+: ([0-9.]+)"),
    ("v30_segunda", r"V30 segunda\s+: ([0-9.]+)"),
    ("plugin_segunda", r"V30 \+ plug-in segunda: ([0-9.]+)"),
    ("v48_oof", r"V48 reconstruido \(OOF, con plug-in\): ([0-9.]+)"),
    ("v53_cv", r"V53 CV ([0-9.]+)"),
    ("v54_cv", r"V54 CV ([0-9.]+)"),
    ("v56_cv", r"V56 CV ([0-9.]+)"),
    ("coincidencia_sub", r"coincidencia con la submission sellada: [0-9,]+ de [0-9,]+ \(([0-9.]+)%\)"),
    ("sha_reconstruida", r"SHA256 reconstruida : ([0-9a-f]{64})"),
]


def salidas(notebook: Path) -> str:
    """Concatena todos los textos de salida del notebook ejecutado."""
    nb = json.loads(notebook.read_text(encoding="utf-8"))
    partes = []
    for celda in nb.get("cells", []):
        for salida in celda.get("outputs", []):
            if salida.get("output_type") == "stream":
                partes.append("".join(salida.get("text", [])))
            elif salida.get("output_type") == "execute_result":
                partes.append("".join(salida.get("data", {}).get("text/plain", [])))
    return "\n".join(partes)


def extrae(texto: str, patron: str) -> str | None:
    """Primer grupo capturado por el patron, o None."""
    coincidencia = re.search(patron, texto)
    return coincidencia.group(1) if coincidencia else None


def modo_checklist(notebook: Path) -> int:
    """Comprueba los valores clave del checklist contra la documentacion."""
    texto = salidas(notebook)
    fallos = 0
    print(f"=== checklist sobre {notebook.name} ===")
    for etiqueta, patron, esperado, tolerancia in CHECKLIST:
        crudo = extrae(texto, patron)
        if crudo is None:
            print(f"  [FALTA] {etiqueta}: no aparece en las salidas")
            fallos += 1
            continue
        if esperado is None:
            print(f"  [INFO ] {etiqueta}: {crudo}")
            continue
        valor = float(crudo)
        ok = abs(valor - esperado) <= tolerancia
        print(f"  [{'OK' if ok else 'FALLA'}] {etiqueta}: {valor} (delta {valor - esperado:+.2e})")
        fallos += 0 if ok else 1
    print(f"  {len(CHECKLIST) - fallos}/{len(CHECKLIST)} pasan")
    return 0 if fallos == 0 else 2


def modo_comparar(primero: Path, segundo: Path) -> int:
    """Compara los valores clave de dos ejecuciones (idempotencia)."""
    texto_a, texto_b = salidas(primero), salidas(segundo)
    fallos = 0
    print("=== comparacion de idempotencia ===")
    for etiqueta, patron in CLAVES:
        valor_a, valor_b = extrae(texto_a, patron), extrae(texto_b, patron)
        iguales = valor_a is not None and valor_a == valor_b
        print(f"  [{'OK' if iguales else 'DIFIERE'}] {etiqueta}: {valor_a} vs {valor_b}")
        fallos += 0 if iguales else 1
    print(f"  {len(CLAVES) - fallos}/{len(CLAVES)} identicos")
    return 0 if fallos == 0 else 2


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada con los dos modos."""
    ap = argparse.ArgumentParser(description="Verificador post-ejecucion del notebook AGE")
    ap.add_argument("--checklist", type=Path, help="notebook ejecutado a comprobar")
    ap.add_argument("--comparar", nargs=2, type=Path, metavar=("A", "B"), help="dos ejecuciones")
    args = ap.parse_args(argv)
    if args.checklist:
        return modo_checklist(args.checklist)
    if args.comparar:
        return modo_comparar(*args.comparar)
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
