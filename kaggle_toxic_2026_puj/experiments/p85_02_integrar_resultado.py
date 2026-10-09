"""Integra la seccion 131 (P85-01: SGNS corregido sobre Twitter/X) en el notebook.

Lee `reports/p85_01_sgns_twitter.json` y construye la seccion 131 con el resultado
de la reapertura autorizada (embedding-only y fusion por brazo, complementariedad
y decision preregistrada) mas una celda de verificacion que comprueba los
entregables, la fila P85-01 del registro y el escaneo byte a byte.

Idempotente y ASCII puro (politica de generacion del proyecto).

Uso:
    /home/carlos/venv/bin/python experiments/p85_02_integrar_resultado.py
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RUTA_NOTEBOOK = RAIZ / "notebooks" / "10_final_pipeline.ipynb"
RUTA_JSON = RAIZ / "reports" / "p85_01_sgns_twitter.json"
RUTA_MD = RAIZ / "reports" / "toxic_twitter_sgns_corrected.md"
RUTA_CSV = RAIZ / "reports" / "toxic_twitter_sgns_corrected.csv"
MARCA_SECCION = "## 131. P85-01"
MARCA_CODIGO = "Verificacion de la seccion 131 (P85-01)"


def nueva_celda_markdown(texto: str) -> dict:
    """Crea una celda de markdown con identificador unico."""
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": texto.splitlines(keepends=True),
    }


def nueva_celda_codigo(texto: str) -> dict:
    """Crea una celda de codigo vacia de salidas y sin ejecutar."""
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": texto.splitlines(keepends=True),
    }


CODIGO = '''"""Verificacion de la seccion 131 (P85-01).

Comprueba los entregables de la reapertura, la fila P85-01 del registro y que
los ficheros generados son ASCII puro (0 bytes > 127).
"""

import csv
import json

ruta_json = PROJECT_ROOT / "reports" / "p85_01_sgns_twitter.json"
ruta_md = PROJECT_ROOT / "reports" / "toxic_twitter_sgns_corrected.md"
ruta_csv = PROJECT_ROOT / "reports" / "toxic_twitter_sgns_corrected.csv"
for ruta in (ruta_json, ruta_md, ruta_csv):
    assert ruta.exists(), f"falta {ruta.name}"

informe = json.loads(ruta_json.read_text(encoding="utf-8"))
assert informe["experiment_id"] == "P85-01"
assert informe["caso"] in ("A", "B", "C")
assert informe["compliance"]["test_usado"] == 0
assert informe["compliance"]["holdout_usado"] == 0
assert informe["compliance"]["entrenamientos_nuevos"] == 1

registro = list(csv.DictReader((PROJECT_ROOT / "experiments.csv").read_text(encoding="utf-8-sig").splitlines()))
assert any(fila["experiment_id"] == "P85-01" for fila in registro), "falta la fila P85-01"

sucios = [ruta.name for ruta in (ruta_json, ruta_md, ruta_csv) if any(b > 127 for b in ruta.read_bytes())]
assert not sucios, f"bytes > 127 en {sucios}"

twitter = informe["brazos"]["twitter"]
general = informe["brazos"]["general"]
print("P85-01 (SGNS corregido sobre Twitter/X) verificado:")
print(f"  caso {informe['caso']} | fusion twitter {twitter['delta_fusion_congelada']:+.6f} vs V03 | general {general['delta_fusion_congelada']:+.6f}")
print("  entregables OK | fila P85-01 OK | escaneo byte a byte: 0 bytes > 127")'''


def construir_seccion(informe: dict) -> str:
    """Devuelve el markdown de la seccion 131 a partir del informe JSON.

    Args:
        informe: contenido de reports/p85_01_sgns_twitter.json.

    Returns:
        Texto markdown de la seccion.
    """
    tw = informe["brazos"]["twitter"]
    gen = informe["brazos"]["general"]
    cond = informe["condiciones"]
    net_f = tw["matriz_fusion_vs_v03"]["net"]
    net_c = tw["matriz_candidato_vs_v03"]["net"]
    return f"""## 131. P85-01: SGNS corregido sobre corpus Twitter/X (reapertura unica, 2026-10-08)

La reapertura autorizada comprobo la unica combinacion no ejecutada: SGNS correctamente implementado (suma del termino negativo, implementacion anclada de P84-04) sobre el corpus legal de Twitter/X (mmaguero, CC BY-SA 4.0, periodo cerrado). Control general: el embedding S de P84-04, re-puntuado con la normalizacion documentada (EMB-NORMAL-TW v1) porque su pooling publicado uso texto crudo (cobertura 0.6585). Semilla unica 20261016 (limitacion declarada: no se estima la varianza de reentrenamiento).

| brazo | embedding-only congelada | fusion congelada | delta fusion vs V03 | pliegues + |
|---|---|---|---|---|
| Twitter/X | {tw['embedding_only']['f1_congelada']:.6f} | {tw['fusion']['f1_congelada']:.6f} | {tw['delta_fusion_congelada']:+.6f} | {tw['pliegues_positivos_fusion']}/5 |
| general | {gen['embedding_only']['f1_congelada']:.6f} | {gen['fusion']['f1_congelada']:.6f} | {gen['delta_fusion_congelada']:+.6f} | {gen['pliegues_positivos_fusion']}/5 |

Bootstrap del delta de la fusion (Twitter): IC95 {tw['bootstrap_delta_fusion_congelada']['ic95']}. ensemble_gain 0.5: {tw['ensemble_gain_05']['ensemble_gain']:+.6f} (IC95 {tw['ensemble_gain_05']['ic95']}). Complementariedad de la fusion vs V03: salvage {net_f['global']['salvage']}, damage {net_f['global']['damage']}, net {net_f['global']['net']} (candidato vs V03: {net_c['global']['salvage']}/{net_c['global']['damage']}/{net_c['global']['net']}).

Condiciones preregistradas: C1 {cond['C1_delta_fusion_ge_0015']} | C2 {cond['C2_cuatro_de_cinco_pliegues']} | C3 {cond['C3_net_fusion_positivo']} | C4 {cond['C4_bootstrap_ic95_inferior_positivo']} | C5 {cond['C5_dominio_twitter_supera_general']}. **Caso {informe['caso']}: {informe['decision']}.**

Entregables: `reports/toxic_twitter_sgns_corrected.md` y `.csv`; `reports/p85_01_sgns_twitter.json`; fila P85-01 en `experiments.csv`; D-290. Sin submissions, sin test, sin holdout, sin leaderboard, sin push. No se abren mas experimentos tras esta comparacion.

V03 remains the best verified leaderboard result."""


def validar(cells: list[dict]) -> None:
    """Compila las celdas de codigo y exige ASCII puro en lo generado."""
    for indice, celda in enumerate(cells):
        fuente = "".join(celda.get("source", []))
        if celda["cell_type"] == "code":
            compile(fuente, f"<celda {indice}>", "exec")
    if any(ord(caracter) > 127 for caracter in CODIGO):
        raise SystemExit("Byte > 127 en el texto generado (codigo)")
    if any(byte > 127 for byte in Path(__file__).read_bytes()):
        raise SystemExit("El propio script contiene bytes > 127")


def main() -> None:
    """Integra la seccion 131 y guarda el notebook."""
    informe = json.loads(RUTA_JSON.read_text(encoding="utf-8"))
    seccion = construir_seccion(informe)
    if any(ord(caracter) > 127 for caracter in seccion):
        raise SystemExit("Byte > 127 en el texto generado (seccion)")

    notebook = json.loads(RUTA_NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    celdas_antes = len(cells)

    while any("".join(c.get("source", [])).startswith(MARCA_SECCION) for c in cells):
        cells.pop(next(i for i, c in enumerate(cells) if "".join(c.get("source", [])).startswith(MARCA_SECCION)))
    while any(MARCA_CODIGO in "".join(c.get("source", [])) for c in cells):
        cells.pop(next(i for i, c in enumerate(cells) if MARCA_CODIGO in "".join(c.get("source", []))))

    cells.append(nueva_celda_markdown(seccion))
    cells.append(nueva_celda_codigo(CODIGO))

    validar(cells)
    notebook["cells"] = cells
    RUTA_NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )

    print("Notebook actualizado con la seccion 131:")
    print(f"  celdas antes: {celdas_antes} | despues: {len(cells)}")
    print(f"  caso {informe['caso']} | delta fusion twitter {informe['brazos']['twitter']['delta_fusion_congelada']:+.6f}")
    print("  celdas de codigo compiladas: OK | ASCII puro en lo generado: OK (0 bytes > 127)")


if __name__ == "__main__":
    main()
