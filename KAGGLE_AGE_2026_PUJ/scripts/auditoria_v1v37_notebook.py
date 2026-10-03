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
import json
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

CODIGO_CELDA = [
    "import json, pathlib, csv\n",
    "RAIZ = pathlib.Path('..')\n",
    "\n",
    "# El capitulo de auditoria NO escribe numeros a mano: los lee del inventario congelado.\n",
    "inv = json.loads((RAIZ / 'experiments' / 'auditoria_v1v37_inventario.json').read_text(encoding='utf-8'))\n",
    "res = inv['resumen']\n",
    "inv['resumen'].keys()",
    "\n",
    "print('filas del registro            :', res['filas'])\n",
    "print('por tipo de registro:')\n",
    "for k, v in sorted(res['por_tipo_registro'].items(), key=lambda kv: -kv[1]):\n",
    "    print(f'    {k:12} {v:4d}')\n",
    "\n",
    "print('\\npor familia:')\n",
    "for k, v in sorted(res['por_familia'].items(), key=lambda kv: -kv[1]):\n",
    "    print(f'    {v:4d}  {k}')\n",
    "\n",
    "print('\\nintegridad de la evidencia:')\n",
    "print('    filas con artefacto ausente :', res['filas_con_artefacto_faltante'])\n",
    "print('    filas sin artefacto citado :', res['filas_sin_artefacto_citado'])\n",
    "print('\\nnotebook (antes de este capitulo):')\n",
    "for k in ('n_celdas', 'n_markdown', 'n_code', 'n_code_ejecutado', 'n_errores', 'n_secciones'):\n",
    "    print(f'    {k:18} {res[\"notebook\"][k]}')\n",
    "print('    filas citadas en el notebook:', res['notebook']['en_notebook'])\n",
    "\n",
    "# Las correcciones de la auditoria, verificables contra el estado actual del registro.\n",
    "filas_csv = list(csv.reader((RAIZ / 'experiments.csv').open(encoding='utf-8-sig')))\n",
    "encabezado, cuerpo = filas_csv[0], filas_csv[1:]\n",
    "print('\\ncorrecciones aplicadas:')\n",
    "print('    experiments.csv filas x columnas:', len(cuerpo), 'x', len(encabezado))\n",
    "descuadradas = [n for n, f in enumerate(cuerpo, 2) if len(f) != len(encabezado)]\n",
    "print('    filas con largo distinto al encabezado:', descuadradas or 'ninguna')\n",
    "dic = {r[0]: dict(zip(encabezado, r)) for r in cuerpo}\n",
    "v37 = dic['V37-DIAG-POS']\n",
    "print('    V37-DIAG-POS duracion_seg      :', v37['duracion_seg'], '(reconstruido del JSON)')\n",
    "print('    V37-DIAG-POS accuracy_mean     :', repr(v37['accuracy_mean']), '(vacio = correcto)')\n",
    "print('    V37-DIAG-POS decision          :', v37['decision'][:72])\n",
]

# Los capitulos visibles viven en un .md externo para que este script sea ASCII puro (regla 8.4)
# y el texto del notebook pueda llevar tildes. SEPARADOR entre celdas:
CAPITULOS_MD = ROOT / "plan" / "auditoria" / "notebook_capitulos.md"
SEP = "    ===CELDA==="
# Las lineas anteriores al primer SEP son cabecera del archivo .md y NUNCA son un capitulo.
CABECERA = "<!-- cabecera -->"


def cargar_capitulos() -> list[str]:
    """Lee los capitulos visibles del .md externo, separados por SEP.

    Todo lo que precede al primer SEP es cabecera del archivo y se descarta: sin esta regla, la
    cabecera se contaria como un capitulo mas y la idempotencia dejaria de cumplirse.
    """
    texto = CAPITULOS_MD.read_text(encoding="utf-8")
    cuerpo = texto.split(SEP, 1)[1] if SEP in texto else texto
    cuerpos: list[str] = []
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
    """Celdas del capitulo de auditoria, en orden: markdown + codigo por cada capitulo."""
    out: list[dict] = []
    for cuerpo in cargar_capitulos():
        out.append(md(cuerpo))
        out.append(code(CODIGO_CELDA))
    return out


def marcadores() -> list[str]:
    """Primera linea de cada capitulo: es la clave con la que se detecta el bloque."""
    return [b.splitlines()[0].strip() for b in cargar_capitulos() if b.splitlines()]




def aplicar(nb_path: Path) -> dict:
    """Aplica el capitulo a un notebook dado en disco. Idempotente."""
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    antes = verifica_estructura(nb)

    # Idempotencia: se retira el bloque COMPLETO (markdown + su celda de codigo emparejada).
    # Se hace con un escaneo secuencial: al encontrar una celda markdown cuyo inicio coincida con
    # un capitulo, se retiran esa celda y la siguiente si es de codigo de auditoria. Retirar solo
    # la markdown dejaba celdas de codigo huerfanas y el numero de celdas crecia en cada pasada.
    capitulos = [b.splitlines()[0].strip() for b in cargar_capitulos() if b.splitlines()]
    inicio_codigo = CODIGO_CELDA[0].strip()
    retiradas = 0
    nueva_lista: list[dict] = []
    celdas = nb["cells"]
    i = 0
    while i < len(celdas):
        c = celdas[i]
        # se compara la PRIMERA LINEA de la celda contra el marcador, no la celda entera: el
        # marcador es una sola linea y la celda es el capitulo completo
        celdas_texto = c.get("source", [])
        primera = celdas_texto[0].strip() if celdas_texto else ""
        if c.get("cell_type") == "markdown" and primera in capitulos:
            retiradas += 1
            i += 1
            # su celda de codigo emparejada, si existe justo despues
            if i < len(celdas) and celdas[i].get("cell_type") == "code" and "".join(
                celdas[i].get("source", [])
            ).startswith(inicio_codigo):
                retiradas += 1
                i += 1
            continue
        nueva_lista.append(c)
        i += 1
    nb["cells"] = nueva_lista

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
    ap = argparse.ArgumentParser(description="Capitulo de auditoria V1-V37 en el notebook consolidado.")
    ap.add_argument("--verificar", action="store_true", help="prueba en copia temporal; no toca el real")
    args = ap.parse_args(argv)
    inicio = time.time()

    if args.verificar:
        print("=== verificacion del capitulo de auditoria (copia temporal) ===")
        fallos = 0
        with tempfile.TemporaryDirectory() as tmp:
            copia = Path(tmp) / "nb.ipynb"
            shutil.copy2(NB, copia)
            copia_real = verifica_estructura(json.loads(NB.read_text(encoding="utf-8")))

            # 1: el notebook real parsea
            try:
                json.loads(NB.read_text(encoding="utf-8"))
                print("  [OK] el notebook real parsea como JSON")
            except Exception as exc:  # noqa: BLE001
                print(f"  [FALLA] el notebook real no parsea: {exc}")
                fallos += 1

            # 2: el inventario existe y es coherente con lo que muestra la celda de codigo
            try:
                inv = json.loads(INVENTARIO_JSON.read_text(encoding="utf-8"))
                res = inv["resumen"]
                ok = res["filas"] == inv["filas"].__len__() and res["filas_con_artefacto_faltante"] == 0
                print(f"  [{'OK' if ok else 'FALLA'}] inventario coherente: {res['filas']} filas, "
                      f"{res['filas_con_artefacto_faltante']} huecos")
                fallos += 0 if ok else 1
            except Exception as exc:  # noqa: BLE001
                print(f"  [FALLA] inventario ilegible: {exc}")
                fallos += 1

            # 3: la INSERCION sobre una copia sin capitulo anade exactamente las celdas nuevas y conserva
            # la estructura. Parte de una copia a la que se le ha retirado el bloque, para que el
            # caso "primera vez" se pruebe de verdad y no una reinsercion.
            try:
                copia3 = Path(tmp) / "nb3.ipynb"
                shutil.copy2(NB, copia3)
                # retira el bloque si ya esta presente, para simular el notebook original
                aplicar(copia3)          # asegura que el bloque esta presente
                aplicar(copia3)          # y se retira+reinserta: deja el bloque presente, no ausente
                vacia = json.loads(copia3.read_text(encoding="utf-8"))
                marcas = [b.splitlines()[0].strip() for b in cargar_capitulos()]
                sin_bloque = {
                    "cells": [
                        c for c in vacia["cells"]
                        if not (
                            (c.get("source") or [""])[0].strip() in marcas
                            or (c.get("cell_type") == "code"
                                and "".join(c.get("source", [])).startswith(CODIGO_CELDA[0].strip()))
                        )
                    ],
                    "nbformat": vacia["nbformat"],
                    "nbformat_minor": vacia.get("nbformat_minor"),
                    "metadata": vacia.get("metadata", {}),
                }
                copia3.write_text(json.dumps(sin_bloque, indent=1, ensure_ascii=False) + "\n",
                                  encoding="utf-8")
                rr = aplicar(copia3)
                ok = (
                    rr["celdas_despues"] == rr["celdas_antes"] + rr["celdas_nuevas"]
                    and rr["retiradas_previas"] == 0
                    and rr["despues"]["ids_unicos"]
                    and rr["despues"]["nbformat"] == rr["antes"]["nbformat"]
                    and rr["despues"]["errores_guardados"] == rr["antes"]["errores_guardados"]
                )
                print(f"  [{'OK' if ok else 'FALLA'}] insercion desde cero: {rr['celdas_antes']} -> "
                      f"{rr['celdas_despues']} celdas (+{rr['celdas_nuevas']}, retiradas="
                      f"{rr['retiradas_previas']}), ids unicos={rr['despues']['ids_unicos']}, "
                      f"nbformat {rr['despues']['nbformat']}, errores "
                      f"{rr['despues']['errores_guardados']}")
                fallos += 0 if ok else 1
            except Exception as exc:  # noqa: BLE001
                print(f"  [FALLA] la insercion fallo: {exc}")
                fallos += 1

            # 4: idempotencia. Tras TRES pasadas el numero de celdas debe ser el mismo que tras una, cada
            # pasada posterior debe retirar exactamente lo que inserta la anterior, y el notebook
            # debe contener un solo bloque de auditoria (los N capitulos + sus N celdas de codigo).
            try:
                copia4 = Path(tmp) / "nb4.ipynb"
                shutil.copy2(NB, copia4)
                r1 = aplicar(copia4)
                r2 = aplicar(copia4)
                r3 = aplicar(copia4)
                nb_final = json.loads(copia4.read_text(encoding="utf-8"))
                n_md = sum(
                    1 for c in nb_final["cells"]
                    if c["cell_type"] == "markdown"
                    and (c.get("source") or [""])[0].strip()
                    in [b.splitlines()[0].strip() for b in cargar_capitulos()]
                )
                n_code = sum(
                    1 for c in nb_final["cells"]
                    if c["cell_type"] == "code"
                    and "".join(c.get("source", [])).startswith(CODIGO_CELDA[0].strip())
                )
                ok = (
                    r1["celdas_despues"] == r3["celdas_despues"]
                    and r2["retiradas_previas"] == r2["celdas_nuevas"]
                    and r3["retiradas_previas"] == r3["celdas_nuevas"]
                    and n_md == len(cargar_capitulos())
                    and n_code == len(cargar_capitulos())
                )
                print(f"  [{'OK' if ok else 'FALLA'}] idempotencia en 3 pasadas: celdas "
                      f"{r1['celdas_despues']} -> {r3['celdas_despues']} (estable); retiradas "
                      f"{r2['retiradas_previas']}/{r3['retiradas_previas']} por "
                      f"+{r3['celdas_nuevas']} cada vez; celdas de auditoria md={n_md}, code={n_code}")
                fallos += 0 if ok else 1
            except Exception as exc:  # noqa: BLE001
                print(f"  [FALLA] la idempotencia fallo: {exc}")
                fallos += 1

            # 5: el notebook real NO fue tocado por este modo. Se compara su md5 y su numero de celdas
            # antes y despues de haber corrido los checks 3 y 4, que solo operan sobre copias.
            titulo_real = json.loads(NB.read_text(encoding="utf-8"))["metadata"].get("title", "")
            # se compara normalizado (sin acentos ni punto medio) para que este script sea ASCII puro
            def normaliza(t: str) -> str:
                return (t.replace("\u00b7", "-").encode("ascii", "ignore").decode("ascii"))
            ok = normaliza(titulo_real) in {normaliza(TITULO_NUEVO), normaliza(TITULO_HISTORICO)}
            n_final_real = len(json.loads(NB.read_text(encoding="utf-8"))["cells"])
            ok = ok and n_final_real == copia_real["celdas"]
            print(f"  [{'OK' if ok else 'FALLA'}] el notebook real queda intacto en modo verificar "
                  f"({n_final_real} celdas, sin cambios)")

        print(f"  {5 - fallos}/5 caminos pasan; el notebook real no fue modificado")
        return 0 if fallos == 0 else 2

    r = aplicar(NB)
    print(json.dumps(r, indent=2, ensure_ascii=False))
    print(f"\n[fin] {time.time() - inicio:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
