"""generar_notebook_kaggle_age.py - construye notebooks/kaggle_final_pipeline.ipynb (TweetAge_2026_PUJ).

Ensambla el notebook final desde tres fuentes:
  1. `plan/kaggle_notebook/secciones/*.md`  - markdown (con tildes) + bloques ```python = celdas de codigo.
  2. `repro/cadena/*.py`                     - modulos de la cadena, embebidos y transformados
                                               (se quitan `from __future__` y los imports `cadena.*`).
  3. Celdas fijas definidas aqui (portada tecnica, resolucion de datos, folds embebidos, etc.).

Reglas: este script es ASCII puro; el codigo de las celdas se valida ASCII; los .md llevan tildes.

Uso:
    /home/carlos/venv/bin/python scripts/generar_notebook_kaggle_age.py [--parcial] [--verificar]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
SECCIONES = RAIZ / "plan" / "kaggle_notebook" / "secciones"
CADENA = RAIZ / "repro" / "cadena"
SALIDA = RAIZ / "notebooks" / "kaggle_final_pipeline.ipynb"
FOLD_PAYLOAD = RAIZ / "plan" / "kaggle_notebook" / "fold_idx_b85.txt"
FOLD2_PAYLOAD = RAIZ / "plan" / "kaggle_notebook" / "fold_idx2_b85.txt"
SUBMISSION_PAYLOAD = RAIZ / "plan" / "kaggle_notebook" / "submission_sellada_b85.txt"

#: Orden de ensamblaje: ("md", fichero) o ("cadena", modulo). Los modulos de la cadena se
#: insertan juntos tras la portada; el resto de secciones va en orden CRISP-ML.
CONTENIDO = [
    ("md", "00_portada.md"),
    ("md", "01_fase1.md"),
    ("md", "02_fase2.md"),
    ("md", "fase2_5_auditoria_historial.md"),
    ("md", "03_fase3.md"),
    ("md", "fase4_intro.md"),
    ("md", "fase4_1_v18_v29.md"),
    ("md", "04_fase4.md"),
    ("md", "fase4_3_fracasos.md"),
    ("md", "04b_fase4_nn.md"),
    ("md", "fase4_5_v44_v52.md"),
    ("md", "fase4_6_v53_v57.md"),
    ("md", "05_fase5.md"),
    ("md", "fase5_4_matematicos.md"),
    ("md", "fase5_5_calibracion.md"),
    ("md", "06_fase6.md"),
    ("md", "anexo_a_inventario.md"),
    ("md", "anexo_b_referencias.md"),
    ("md", "anexo_c_glosario.md"),
]

#: Modulos de la cadena, en orden de dependencias.
MODULOS = ["comun", "clusters", "dense", "structured", "supervised", "nb_svm",
           "miembros_oof_a", "miembros_oof_b", "meta", "v30", "neuronales",
           "consenso_v2", "analisis_v57", "test_side"]

FORBIDDEN = ["from src", "import src", "from scripts", "import scripts",
             "from cadena", "import cadena", "git clone", "subprocess"]


def celda_markdown(texto: str) -> dict:
    """Celda markdown con identificador unico."""
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {},
            "source": texto.splitlines(keepends=True)}


def celda_codigo(texto: str) -> dict:
    """Celda de codigo vacia de salidas y sin ejecutar."""
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
            "execution_count": None, "outputs": [],
            "source": texto.splitlines(keepends=True)}


def divide_markdown(texto: str, limite: int = 120) -> list[str]:
    """Divide la prosa en trozos manejables: por encabezados (## / ###) y por tamano.

    Los trozos mayores de `limite` lineas se parten en lineas en blanco; si el corte cae dentro de
    una tabla (sin lineas en blanco), se repite su cabecera con una nota de continuacion.
    """
    lineas = texto.split("\n")
    trozos: list[list[str]] = []
    actual: list[str] = []
    for linea in lineas:
        if (linea.startswith("## ") or linea.startswith("### ")) and any(x.strip() for x in actual):
            trozos.append(actual)
            actual = [linea]
        else:
            actual.append(linea)
    if actual:
        trozos.append(actual)

    finales: list[str] = []
    for trozo in trozos:
        while len(trozo) > limite:
            corte = None
            for i in range(limite, 0, -1):
                if not trozo[i].strip():
                    corte = i
                    break
            es_tabla = corte is None
            if corte is None:
                corte = limite
            cabeza, resto = trozo[:corte], trozo[corte:]
            if es_tabla:
                cabecera = [linea for linea in cabeza[:4] if linea.startswith("|")][:2]
                if len(cabecera) == 2:
                    resto = cabecera + ["", "*Tabla (continua).*", ""] + resto
            finales.append("\n".join(cabeza))
            trozo = resto
        finales.append("\n".join(trozo))
    return [trozo.strip("\n") for trozo in finales if trozo.strip()]


def parte_md(fichero: str) -> list[dict]:
    """Convierte un .md de seccion en celdas: markdown (troceado) + bloques ```python."""
    texto = (SECCIONES / fichero).read_text(encoding="utf-8")
    celdas: list[dict] = []
    patron = re.compile(r"```python\n(.*?)```", re.DOTALL)
    posicion = 0
    for coincidencia in patron.finditer(texto):
        antes = texto[posicion:coincidencia.start()].strip("\n")
        for trozo in divide_markdown(antes):
            celdas.append(celda_markdown(trozo))
        codigo = coincidencia.group(1).strip("\n")
        celdas.append(celda_codigo(codigo))
        posicion = coincidencia.end()
    resto = texto[posicion:].strip("\n")
    for trozo in divide_markdown(resto):
        celdas.append(celda_markdown(trozo))
    return celdas


def transforma_modulo(nombre: str) -> str:
    """Lee un modulo de la cadena y quita `from __future__` y los imports internos `cadena.*`.

    Soporta imports multilinea con parentesis (se salta el bloque completo).
    """
    texto = (CADENA / f"{nombre}.py").read_text(encoding="utf-8")
    lineas = []
    saltando = False
    for linea in texto.split("\n"):
        limpia = linea.strip()
        if saltando:
            if ")" in limpia:
                saltando = False
            continue
        if re.match(r"^from cadena(\.[a-z_0-9]+)? import ", limpia):
            if "(" in limpia and ")" not in limpia:
                saltando = True
            continue
        if re.match(r"^import cadena(\.[a-z_0-9]+)?$", limpia):
            continue
        lineas.append(linea)
    texto_final = "\n".join(lineas).strip("\n")
    # En el notebook los nombres viven en el espacio global: los accesos cualificados a los
    # modulos de la cadena (`comun.CLASSES`, `miembros_oof_a.oof_e013`, ...) pierden el prefijo.
    # Nota: `from __future__ import annotations` SE CONSERVA (las anotaciones auto-referentes
    # como `-> EncodedMLP` lo necesitan; es legal al inicio de cada celda).
    for prefijo in ("comun.", "miembros_oof_a.", "miembros_oof_b."):
        texto_final = re.sub(r"\b" + prefijo, "", texto_final)
    return texto_final


def celda_folds() -> str:
    """Celda del protocolo congelado de folds (payload comprimido + reconstruccion)."""
    payload = FOLD_PAYLOAD.read_text(encoding="ascii").strip()
    return (
        "# Protocolo congelado de folds (169.336 valores 0-4): asignacion canonica del proyecto.\n"
        "# Se embebe comprimida para inmunizar el notebook frente a la version de scikit-learn\n"
        "# (StratifiedGroupKFold cambia su salida entre versiones; se verifico contra la 1.8.0).\n"
        "import base64\n"
        "import zlib\n"
        "\n"
        "FOLD_IDX_B85 = (\n"
        f"    \"{payload}\"\n"
        ")\n"
        "\n"
        "\n"
        "def cargar_folds_embebidos() -> np.ndarray:\n"
        "    \"\"\"Decodifica el fold_idx canonico embebido (zlib + base85).\"\"\"\n"
        "    crudo = zlib.decompress(base64.b85decode(FOLD_IDX_B85.encode(\"ascii\")))\n"
        "    return np.frombuffer(crudo, dtype=np.uint8).astype(np.int64)\n"
        "\n"
        "\n"
        "fold_idx = cargar_folds_embebidos()\n"
        "assert fold_idx.shape == (169336,), fold_idx.shape\n"
        "assert sorted(np.unique(fold_idx).tolist()) == [0, 1, 2, 3, 4]\n"
        "print(\"folds canonicos cargados:\", np.bincount(fold_idx).tolist())"
    )


def construir(parcial: bool) -> list[dict]:
    """Ensambla la lista completa de celdas del notebook (cadena insertada tras la portada)."""
    celdas: list[dict] = []
    faltantes: list[str] = []
    for tipo, nombre in CONTENIDO:
        if tipo == "cadena":
            continue
        ruta = SECCIONES / nombre
        if not ruta.is_file():
            faltantes.append(nombre)
            if parcial:
                continue
            raise FileNotFoundError(f"falta la seccion: {nombre}")
        celdas.extend(parte_md(nombre))
        if nombre == "00_portada.md":
            celdas.extend(bloque_cadena())
            celdas.extend(bloque_folds())
    if faltantes and parcial:
        print(f"[generador] AVISO: secciones ausentes (modo parcial): {faltantes}")
    return celdas


def bloque_cadena() -> list[dict]:
    """Bloque de la cadena (modulos embebidos), en orden de dependencias."""
    bloque = [celda_markdown(
        "## 0.3 La cadena de reconstrucción (código embebido)\n\n"
        "Todo el código de reconstrucción viaja dentro de este notebook: cada módulo es un porte\n"
        "fiel del script original del proyecto (la referencia exacta aparece en su docstring).\n"
        "No hay dependencias externas: con el corpus descargado, la cadena se reconstruye entera.\n"
        "Los módulos se presentan en orden de dependencias.")]
    for modulo in MODULOS:
        bloque.append(celda_markdown(f"### Módulo `{modulo}.py`"))
        bloque.append(celda_codigo(transforma_modulo(modulo)))
    return bloque


def bloque_folds() -> list[dict]:
    """Celdas del protocolo congelado de folds (canonica y segunda particion)."""
    return [celda_markdown("### Protocolo congelado de folds (asignación canónica embebida)"),
            celda_codigo(celda_folds()),
            celda_markdown("### Segunda partición (semilla 987654)"),
            celda_codigo(celda_folds2())]


def celda_folds2() -> str:
    """Celda del protocolo de la segunda particion (semilla 987654), embebida."""
    payload = FOLD2_PAYLOAD.read_text(encoding="ascii").strip()
    return (
        "# Segunda particion del proyecto (semilla 987654): particion agrupada alternativa del\n"
        "# linaje V30, embebida por la misma razon que la canonica (inmunidad a la version de\n"
        "# scikit-learn).\n"
        "FOLD_IDX2_B85 = (\n"
        f"    \"{payload}\"\n"
        ")\n"
        "\n"
        "\n"
        "def cargar_folds2_embebidos() -> np.ndarray:\n"
        "    \"\"\"Decodifica el fold_idx de la segunda particion (zlib + base85).\"\"\"\n"
        "    crudo = zlib.decompress(base64.b85decode(FOLD_IDX2_B85.encode(\"ascii\")))\n"
        "    return np.frombuffer(crudo, dtype=np.uint8).astype(np.int64)\n"
        "\n"
        "\n"
        "fold_idx2 = cargar_folds2_embebidos()\n"
        "assert fold_idx2.shape == (169336,), fold_idx2.shape\n"
        "assert sorted(np.unique(fold_idx2).tolist()) == [0, 1, 2, 3, 4]\n"
        "print(\"folds de la segunda particion cargados:\", np.bincount(fold_idx2).tolist())"
    )


def insertar_payload_submission(celdas: list[dict]) -> list[dict]:
    """Reemplaza el marcador __PAYLOAD_SUBMISSION__ por el payload de la submission sellada."""
    payload = SUBMISSION_PAYLOAD.read_text(encoding="ascii").strip()
    for celda in celdas:
        if celda["cell_type"] != "code":
            continue
        texto = "".join(celda["source"])
        if "__PAYLOAD_SUBMISSION__" in texto:
            celda["source"] = texto.replace("__PAYLOAD_SUBMISSION__", payload).splitlines(keepends=True)
    return celdas


def verificar(celdas: list[dict]) -> int:
    """Comprobaciones estructurales del notebook (0 = todo bien)."""
    fallos = 0
    codigos = [c for c in celdas if c["cell_type"] == "code"]
    marcas = [c for c in celdas if c["cell_type"] == "markdown"]

    def check(ok: bool, mensaje: str) -> None:
        nonlocal fallos
        print(f"  [{'OK' if ok else 'FALLA'}] {mensaje}")
        if not ok:
            fallos += 1

    check(len(celdas) >= 80, f"celdas totales: {len(celdas)} (markdown {len(marcas)}, codigo {len(codigos)})")
    no_ascii = []
    for celda in codigos:
        texto = "".join(celda["source"])
        if any(ord(c) > 127 for c in texto):
            no_ascii.append(texto[:60])
    check(not no_ascii, f"codigo 100% ASCII (violaciones: {len(no_ascii)})")

    prohibidos = []
    for celda in codigos:
        texto = "".join(celda["source"])
        for patron in FORBIDDEN:
            if patron in texto:
                prohibidos.append(patron)
    check(not prohibidos, f"sin imports del proyecto ni git clone (violaciones: {sorted(set(prohibidos))})")

    todos = "\n".join("".join(c["source"]) for c in celdas)
    for fase in ["FASE 1", "FASE 2", "FASE 3", "FASE 4", "FASE 5", "FASE 6"]:
        check(fase in todos, f"fase presente: {fase}")
    check("FOLD_IDX_B85" in todos, "payload de folds embebido")
    check("FOLD_IDX2_B85" in todos, "payload de la segunda particion embebido")
    check("SUBMISSION_SELLADA_B85" in todos and "__PAYLOAD_SUBMISSION__" not in todos,
          "payload de la submission sellada inyectado")
    check("plt.show()" in todos, "figuras mostradas con plt.show()")

    codigo_total = "\n".join("".join(c["source"]) for c in codigos)
    check("from cadena" not in codigo_total and "import cadena" not in codigo_total,
          "imports internos de la cadena eliminados")
    check(len(codigos) >= len(MODULOS), f"modulos de la cadena presentes: {len(MODULOS)}")

    sin_compilar = []
    for posicion, celda in enumerate(codigos):
        texto = "".join(celda["source"])
        try:
            compile(texto, f"<celda {posicion}>", "exec")
        except SyntaxError as error:
            sin_compilar.append(f"celda {posicion}: {error.msg} (linea {error.lineno})")
    check(not sin_compilar, f"todas las celdas de codigo compilan (fallos: {sin_compilar})")

    # Nombres indefinidos: se concatenan todas las celdas (espacio global compartido) y se
    # consulta a ruff por F821. Detecta restos de la transformacion (p. ej. `comun.CLASSES`).
    import subprocess
    try:
        resultado = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "--select", "F821", "--no-cache",
             "--stdin-filename", "notebook_concatenado.py", "-"],
            input=codigo_total, capture_output=True, text=True, timeout=120, check=False)
        if resultado.returncode == 0:
            check(True, "sin nombres indefinidos (ruff F821)")
        else:
            detalles = [l for l in resultado.stdout.splitlines() if "F821" in l][:6]
            check(False, f"nombres indefinidos (ruff F821): {detalles}")
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"  [AVISO] no se pudo ejecutar ruff F821: {error}")
    return fallos


def main(argv: list[str] | None = None) -> int:
    """Construye (o verifica) el notebook final."""
    ap = argparse.ArgumentParser(description="Generador del notebook final de Kaggle (AGE)")
    ap.add_argument("--parcial", action="store_true", help="tolera secciones ausentes")
    ap.add_argument("--verificar", action="store_true", help="solo comprueba estructura")
    args = ap.parse_args(argv)

    celdas = construir(parcial=args.parcial)
    celdas = insertar_payload_submission(celdas)

    if args.verificar:
        print("=== verificacion del notebook generado (en memoria) ===")
        fallos = verificar(celdas)
        print(f"  {fallos} fallos")
        return 0 if fallos == 0 else 2

    notebook = {"cells": celdas,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                            "name": "python3"},
                             "language_info": {"name": "python", "version": "3.14"}},
                "nbformat": 4, "nbformat_minor": 5}
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[generador] notebook escrito: {SALIDA.relative_to(RAIZ)} ({len(celdas)} celdas)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
