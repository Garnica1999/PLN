"""auditoria_v1v37_notebook.py

Fecha: 2026-10-01
Descripcion: anade al notebook consolidado `notebooks/99_entrega_consolidado.ipynb` el capitulo de
    auditoria del historial V1-V37. Inserta celdas al FINAL del notebook, sin tocar ninguna celda
    existente, y actualiza el titulo de los metadatos.

Que NO hace: no reejecuta nada, no lee test/holdout/leaderboard, no genera submissions, no modifica
scripts de experimentos, el CSV de registro ni los artefactos de metricas.

Modo `--verificar`: comprueba en una COPIA en directorio temporal que el parseo del notebook, la
insercion, la unicidad de ids y el round-trip JSON funcionan. No toca el notebook real. Sale 0 si
pasa, 2 si falla.

ASCII puro en el codigo; el texto de las celdas markdown si admite tildes.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "99_entrega_consolidado.ipynb"
INVENTARIO_JSON = ROOT / "experiments" / "auditoria_v1v37_inventario.json"
MATRIZ = ROOT / "reports" / "tables" / "auditoria_v1v37_inventario.csv"
INFORME = ROOT / "reports" / "auditoria_experimentos_v1_v37.md"
PLAN = ROOT / "plan" / "auditoria"

TITULO_NUEVO = "99 - Entrega consolidada (Parte I + Parte II + auditoria del historial V1-V37)"
TITULO_HISTORICO = "99 - Entrega consolidada (Parte I + Parte II: busqueda de senal residual)"

# Los capitulos declaran su propio codigo con bloques ```python dentro del .md fuente. El script
# los extrae y los inserta como celdas de codigo justo despues del texto del capitulo.
RE_BLOQUE_PYTHON = re.compile(r"```python\n(.*?)```", re.DOTALL)

# Los capitulos visibles viven en un .md externo para que este script sea ASCII puro (regla 8.4)
# y el texto del notebook pueda llevar tildes. SEPARADOR entre celdas:
# Fuentes de los capitulos, EN ORDEN de insercion. Primero las fases V36 y V37 (que son del
# proyecto), y al final el capitulo de auditoria, que cierra el notebook.
CAPITULOS_MD = (
    ROOT / "plan" / "notebook_fases_v36_v37.md",
    ROOT / "plan" / "auditoria" / "notebook_capitulos.md",
)
SEP = "    ===CELDA==="
# Las lineas anteriores al primer SEP son cabecera del archivo .md y NUNCA son un capitulo.
CABECERA = "<!-- cabecera -->"
# Primera linea de los capitulos tal como los insertaba la version anterior, que no ponia
# encabezado. Sirven para retirar el bloque antiguo al migrar y no dejarlo duplicado.
MARCADORES_HEREDADOS = (
    "Esta parte documenta una **auditor",
    "El recuento que sigue se **lee**",
    # El capitulo se inserto primero como PARTE III, y ya existia otra PARTE III (V16 y V17),
    # asi que se renombro a PARTE IV. El punto medio se escribe escapado para que este archivo
    # siga siendo ASCII puro; el prefijo lleva "AUDITOR" para no confundirse con la otra PARTE III.
    "# PARTE III \u00b7 AUDITOR",
    "# PARTE IV \u00b7 AUDITOR",
    "# PHASE - V36 (VIA C1-A)",
    "# PHASE - V37: DIAGNOSTICO",
    "## Inventario congelado de la auditor",
)


def cargar_capitulos() -> list[str]:
    """Lee los capitulos visibles de TODOS los .md de `CAPITULOS_MD`, en orden.

    Todo lo que precede al primer SEP de cada archivo es cabecera y se descarta: sin esta regla, la
    cabecera se contaria como un capitulo mas y la idempotencia dejaria de cumplirse.
    """
    cuerpos: list[str] = []
    for ruta in CAPITULOS_MD:
        texto = ruta.read_text(encoding="utf-8")
        cuerpo = texto.split(SEP, 1)[1] if SEP in texto else texto
        for bruto in cuerpo.split(SEP):
            lineas = [ln for ln in bruto.splitlines() if not ln.startswith("<!-- titulo:")]
            while lineas and not lineas[0].strip():
                lineas.pop(0)
            while lineas and not lineas[-1].strip():
                lineas.pop()
            if lineas:
                cuerpos.append("\n".join(lineas))
    return cuerpos


def md(source: str) -> dict:
    """Construye una celda markdown con `source` como lista de lineas (formato nbformat)."""
    lineas = source.splitlines(keepends=True)
    return {"cell_type": "markdown", "metadata": {}, "source": lineas}


def code(source_lines: list[str]) -> dict:
    """Construye una celda de codigo a partir de lineas ya separadas por \\n."""
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": list(source_lines)}


def verifica_estructura(nb: dict) -> dict:
    """Comprobaciones estructurales del notebook, aplicables al real y a una copia."""
    celdas = nb["cells"]
    ids = [c.get("id") for c in celdas if c.get("id")]
    informe = {
        "nbformat": f"{nb['nbformat']}.{nb.get('nbformat_minor')}",
        "celdas": len(celdas),
        "markdown": sum(1 for c in celdas if c["cell_type"] == "markdown"),
        "code": sum(1 for c in celdas if c["cell_type"] == "code"),
        "ids_unicos": len(set(ids)) == len(ids),
        "errores_guardados": sum(
            1 for c in celdas for o in c.get("outputs", []) if o.get("output_type") == "error"
        ),
    }
    return informe


def construye_celdas() -> list[dict]:
    """Celdas del bloque, en orden: el texto de cada capitulo y sus bloques ```python.

    Un capitulo puede llevar varios bloques de codigo; se insertan intercalados en el orden en
    que aparecen. Si un capitulo no lleva ninguno, solo aporta su celda markdown.
    """
    out: list[dict] = []
    for cuerpo in cargar_capitulos():
        partes = RE_BLOQUE_PYTHON.split(cuerpo)
        for indice, parte in enumerate(partes):
            if indice % 2 == 0:
                texto = parte.strip()
                if texto:
                    out.append(md(texto))
            else:
                lineas = parte.strip().splitlines(keepends=True)
                if lineas:
                    out.append(code(lineas))
    return out


def primera_linea(c: dict) -> str:
    """Primera linea del source de una celda, o cadena vacia."""
    fuente = c.get("source") or []
    return fuente[0].strip() if fuente else ""


def marcadores() -> list[str]:
    """Primera linea de cada capitulo: es la clave con la que se detecta el bloque."""
    return [b.splitlines()[0].strip() for b in cargar_capitulos() if b.splitlines()]




def retira_bloque(celdas: list[dict]) -> int:
    """Retira el bloque insertado previamente y devuelve cuantas celdas retiro.

    Se apoya en la FIRMA del bloque (tipo y primera linea de cada celda), de modo que no depende
    de que el bloque este al final ni de marcadores sueltos. Cubre tambien las dos versiones
    heredadas: la primera no llevaba encabezado, y la segunda decia "# PARTE III \u00b7 AUDITOR".
    """
    esperadas = construye_celdas()
    firma = [(c["cell_type"], primera_linea(c)) for c in esperadas]
    n = len(firma)
    reales = [(c["cell_type"], primera_linea(c)) for c in celdas]
    for ini in range(len(celdas) - n + 1):
        if reales[ini:ini + n] == firma:
            del celdas[ini:ini + n]
            return n

    # Migracion de versiones anteriores: markdown de capitulo (sin encabezado o con el encabezado
    # viejo) seguida de su celda de codigo.
    fuera: list[int] = []
    i = 0
    while i < len(celdas):
        primera = primera_linea(celdas[i])
        if celdas[i]["cell_type"] == "markdown" and any(
            primera.startswith(p) for p in MARCADORES_HEREDADOS
        ):
            fuera.append(i)
            i += 1
            if i < len(celdas) and celdas[i]["cell_type"] == "code":
                fuera.append(i)
                i += 1
            continue
        i += 1
    for indice in reversed(fuera):
        del celdas[indice]
    return len(fuera)


def aplicar(nb_path: Path) -> dict:
    """Aplica el bloque de capitulos a un notebook dado en disco. Idempotente."""
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    antes = verifica_estructura(nb)
    retiradas = retira_bloque(nb["cells"])
    nuevas = construye_celdas()
    nb["cells"].extend(nuevas)
    nb.setdefault("metadata", {})["title"] = TITULO_NUEVO
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    despues = verifica_estructura(json.loads(nb_path.read_text(encoding="utf-8")))
    return {
        "celdas_antes": antes["celdas"],
        "celdas_despues": despues["celdas"],
        "celdas_nuevas": len(nuevas),
        "retiradas_previas": retiradas,
        "antes": antes,
        "despues": despues,
    }


def main(argv: list[str] | None = None) -> int:
    """Aplica el bloque al notebook real, o lo verifica en una copia temporal."""
    ap = argparse.ArgumentParser(description="Capitulos V36/V37 y auditoria en el notebook consolidado.")
    ap.add_argument("--verificar", action="store_true", help="prueba en copia temporal; no toca el real")
    args = ap.parse_args(argv)
    inicio = time.time()

    if args.verificar:
        return _verificar()

    r = aplicar(NB)
    print(json.dumps(r, indent=2, ensure_ascii=False))
    print(f"\n[fin] {time.time() - inicio:.1f} s")
    return 0


def _verificar() -> int:
    """Autoejercicio del bloque, sobre COPIAS. No toca el notebook real."""
    print("=== verificacion del bloque de capitulos (copia temporal) ===")
    fallos = 0
    esperadas = construye_celdas()
    firma = [(c["cell_type"], primera_linea(c)) for c in esperadas]

    with tempfile.TemporaryDirectory() as tmp:
        # 1: el notebook real parsea y su md5 de celdas no cambia durante la verificacion
        real_antes = json.loads(NB.read_text(encoding="utf-8"))["cells"]
        try:
            json.loads(NB.read_text(encoding="utf-8"))
            print(f"  [OK] el notebook real parsea como JSON ({len(real_antes)} celdas)")
        except Exception as exc:  # noqa: BLE001
            print(f"  [FALLA] el notebook real no parsea: {exc}")
            fallos += 1

        # 2: el bloque tiene las celdas esperadas y al menos un bloque de codigo por capitulo
        n_md = sum(1 for c in esperadas if c["cell_type"] == "markdown")
        n_code = sum(1 for c in esperadas if c["cell_type"] == "code")
        ok = n_md >= len(cargar_capitulos()) and n_code >= 1 and len(firma) == len(esperadas)
        print(f"  [{'OK' if ok else 'FALLA'}] bloque construido: {len(esperadas)} celdas "
              f"({n_md} markdown, {n_code} codigo) para {len(cargar_capitulos())} capitulos")
        fallos += 0 if ok else 1

        # 3: todo el codigo del bloque compila
        malas = []
        for c in esperadas:
            if c["cell_type"] != "code":
                continue
            try:
                ast.parse("".join(c["source"]))
            except SyntaxError as exc:
                malas.append(f"{exc.msg} (linea {exc.lineno})")
        ok = not malas
        print(f"  [{'OK' if ok else 'FALLA'}] el codigo del bloque compila: {malas or 'sin errores'}")
        fallos += 0 if ok else 1

        # 3b: el codigo del bloque EJECUTA de verdad contra los artefactos reales. Compilar no
        # basta: un KeyError por una clave que no existe solo aparece al ejecutar. Es la leccion
        # de D-086 aplicada a este bloque.
        import contextlib
        import io as _io
        espacio: dict = {"__name__": "__main__"}
        fallos_ejecucion = []
        for n, c in enumerate(esperadas):
            if c["cell_type"] != "code":
                continue
            try:
                with contextlib.redirect_stdout(_io.StringIO()):
                    exec(compile("".join(c["source"]), f"bloque{n}", "exec"), espacio)
            except Exception as exc:  # noqa: BLE001
                fallos_ejecucion.append(f"celda {n}: {type(exc).__name__}: {exc}")
        ok = not fallos_ejecucion
        print(f"  [{'OK' if ok else 'FALLA'}] el codigo del bloque EJECUTA contra los artefactos: "
              f"{fallos_ejecucion or 'sin errores'}")
        fallos += 0 if ok else 1

        # 4: aplicar a una copia anade el bloque y es idempotente en tres pasadas
        copia = Path(tmp) / "nb.ipynb"
        shutil.copy2(NB, copia)
        r1 = aplicar(copia)
        r2 = aplicar(copia)
        r3 = aplicar(copia)
        nb_final = json.loads(copia.read_text(encoding="utf-8"))
        reales = [(c["cell_type"], primera_linea(c)) for c in nb_final["cells"]]
        n_bloques = sum(1 for i in range(len(reales) - len(firma) + 1) if reales[i:i + len(firma)] == firma)
        ok = (r1["celdas_despues"] == r3["celdas_despues"]
              and r2["retiradas_previas"] == r2["celdas_nuevas"]
              and r3["retiradas_previas"] == r3["celdas_nuevas"]
              and n_bloques == 1)
        print(f"  [{'OK' if ok else 'FALLA'}] idempotencia en 3 pasadas: celdas "
              f"{r1['celdas_despues']} -> {r3['celdas_despues']} (estable); retiradas "
              f"{r2['retiradas_previas']}/{r3['retiradas_previas']} por +{r3['celdas_nuevas']} cada vez; "
              f"bloques presentes={n_bloques}")
        fallos += 0 if ok else 1

        # 5: el notebook real no fue tocado
        real_despues = json.loads(NB.read_text(encoding="utf-8"))["cells"]
        ok = real_antes == real_despues
        print(f"  [{'OK' if ok else 'FALLA'}] el notebook real queda intacto en modo verificar "
              f"({len(real_despues)} celdas, sin cambios)")
        fallos += 0 if ok else 1

    print(f"  {6 - fallos}/6 caminos pasan; el notebook real no fue modificado")
    return 0 if fallos == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
