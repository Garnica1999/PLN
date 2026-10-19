"""P89-01: correccion del bloque P61 duplicado en el notebook consolidado.

Discrepancia detectada en la auditoria final de consistencia (P89-00): el bloque
de la fase P61 (secciones 81 a 88, 24 celdas) aparece DOS veces en el notebook.
La copia gestionada vive entre los marcadores `<!-- P61-NOTEBOOK-INICIO -->` y
`<!-- P61-NOTEBOOK-FIN -->` (la que regenera `p61_10_notebook.py`); delante de
esos marcadores quedo una copia huerfana de la primera version, con dos celdas
que son las versiones CON BUG corregidas despues (el SyntaxError del f-string en
la tabla de cuantiles y la clave 'pool' inexistente del ensemble).

Esta correccion elimina la copia huerfana y conserva la copia gestionada (la
corregida). Es idempotente: si no hay duplicado, no hace nada. No toca ningun
otro documento, no modifica codigo de experimentos y no regenera el notebook.

Uso:
    /home/carlos/venv/bin/python experiments/p89_01_correccion_duplicado_p61.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Final

RAIZ: Final[Path] = Path(__file__).resolve().parents[1]
RUTA_NOTEBOOK: Final[Path] = RAIZ / "notebooks" / "10_final_pipeline.ipynb"
MARCA_INICIO: Final[str] = "<!-- P61-NOTEBOOK-INICIO -->"
MARCA_FIN: Final[str] = "<!-- P61-NOTEBOOK-FIN -->"


def log(mensaje: str) -> None:
    """Imprime una linea de progreso."""
    print(f"[P89-01] {mensaje}", flush=True)


def fuente(celda: dict) -> str:
    """Devuelve el texto de una celda."""
    return "".join(celda.get("source", []))


def firma(celda: dict) -> str:
    """Hash corto del contenido de una celda."""
    return hashlib.sha256(fuente(celda).encode()).hexdigest()


def detectar_huerfano(cells: list[dict]) -> dict[str, Any] | None:
    """Localiza la copia huerfana del bloque P61, si existe.

    La copia huerfana son las N celdas inmediatamente anteriores al marcador de
    INICIO, donde N es el tamano del bloque gestionado (entre INICIO y FIN).

    Args:
        cells: celdas del notebook.

    Returns:
        Descripcion del huerfano (rango, celdas divergentes) o None si no hay
        duplicado.
    """
    indices_inicio = [i for i, c in enumerate(cells) if fuente(c).strip() == MARCA_INICIO]
    indices_fin = [i for i, c in enumerate(cells) if fuente(c).strip() == MARCA_FIN]
    if len(indices_inicio) != 1 or len(indices_fin) != 1:
        return None
    inicio = indices_inicio[0]
    fin = indices_fin[0]
    if fin <= inicio:
        return None
    gestionado = cells[inicio + 1:fin]
    n = len(gestionado)
    if inicio - n < 0:
        return None
    huerfano = cells[inicio - n:inicio]
    if len(huerfano) != n:
        return None
    # Verificacion de equivalencia: mismos tipos, mismas cabeceras y mayoria de
    # celdas identicas; las divergencias deben ser las dos celdas corregidas.
    tipos_ok = all(a["cell_type"] == b["cell_type"] for a, b in zip(huerfano, gestionado))
    if not tipos_ok:
        return None
    identicas = sum(1 for a, b in zip(huerfano, gestionado) if firma(a) == firma(b))
    if identicas < n - 2:
        return None
    divergentes = [
        (inicio - n + j, inicio + 1 + j)
        for j, (a, b) in enumerate(zip(huerfano, gestionado))
        if firma(a) != firma(b)
    ]
    for j_h, j_g in divergentes:
        texto_h = fuente(cells[j_h])
        texto_g = fuente(cells[j_g])
        # Las versiones corregidas documentan los bugs en comentarios.
        if not ("SyntaxError" in texto_g or "no tiene una clave 'pool'" in texto_g):
            return None
        if "SyntaxError" in texto_h or "no tiene una clave 'pool'" in texto_h:
            return None
    return {
        "inicio_huerfano": inicio - n,
        "fin_huerfano": inicio - 1,
        "inicio_gestionado": inicio + 1,
        "fin_gestionado": fin - 1,
        "n_celdas": n,
        "identicas": identicas,
        "divergentes": divergentes,
    }


def validar(cells: list[dict]) -> None:
    """Compila las celdas de codigo del notebook resultante."""
    for indice, celda in enumerate(cells):
        if celda["cell_type"] == "code":
            compile(fuente(celda), f"<celda {indice}>", "exec")


def main() -> int:
    """Elimina la copia huerfana del bloque P61 si existe."""
    notebook = json.loads(RUTA_NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    celdas_antes = len(cells)
    huerfano = detectar_huerfano(cells)
    if huerfano is None:
        log("sin duplicado del bloque P61: no hay nada que corregir")
        print(json.dumps({"corregido": False, "celdas": celdas_antes}, ensure_ascii=False))
        return 0

    log(
        f"copia huerfana detectada: celdas {huerfano['inicio_huerfano']}-{huerfano['fin_huerfano']} "
        f"({huerfano['n_celdas']} celdas, {huerfano['identicas']} identicas a la gestionada)"
    )
    for j_h, j_g in huerfano["divergentes"]:
        log(f"  divergente: celda {j_h} (primera version) frente a la gestionada {j_g} (corregida)")

    del cells[huerfano["inicio_huerfano"]:huerfano["fin_huerfano"] + 1]

    validar(cells)
    restante = detectar_huerfano(cells)
    if restante is not None:
        raise SystemExit("STOP: el duplicado persiste tras la correccion")
    marcas = sum(1 for c in cells if fuente(c).strip() in (MARCA_INICIO, MARCA_FIN))
    if marcas != 2:
        raise SystemExit("STOP: los marcadores P61 no quedaron intactos")

    notebook["cells"] = cells
    RUTA_NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    log(f"corregido: {celdas_antes} -> {len(cells)} celdas (eliminadas {huerfano['n_celdas']})")
    print(json.dumps({
        "corregido": True,
        "celdas_antes": celdas_antes,
        "celdas_despues": len(cells),
        "huerfano": [huerfano["inicio_huerfano"], huerfano["fin_huerfano"]],
        "divergentes": huerfano["divergentes"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
