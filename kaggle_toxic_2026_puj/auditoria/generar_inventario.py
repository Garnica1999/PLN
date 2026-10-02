"""auditoria/generar_inventario.py
Fecha: 2026-10-02
Descripcion: Construye el inventario unificado de experimentos del proyecto
kaggle_toxic_2026_puj leyendo, sin modificarlas, cinco fuentes de solo lectura:
`experiments.csv` (411 registros: 368 bien formados y 43 malformados desde P67),
`experiments/*.py` (376 scripts), `notebooks/10_final_pipeline.ipynb` (446 celdas),
y las evidencias de `reports/`, `resultados/` y `figuras/`.

Genera tres salidas nuevas, todas dentro de `auditoria/`:
  - inventario_experimentos.csv: una fila por experimento identificado. Cubre los
    411 registros de `experiments.csv` mas una fila por cada script de
    `experiments/` que no aparece citado ni asociado a ningun registro.
  - inventario_scripts.csv: una fila por script `.py` de `experiments/` (376).
  - inventario_resumen.md: resumen en espanol (con acentos) con los conteos y
    los huecos principales.

Reglas de construccion (deterministas; ninguna salida depende del orden del
sistema de ficheros):
  - Registro malformado = fila cuyo ancho (numero de campos tras `csv.reader`)
    difiere de 23. Su linea cruda se conserva integra en memoria y se usa para
    extraer el id (primer token antes del primer `|`), la fecha y los scripts.
  - `scripts` de un experimento = rutas `experiments/*.py` citadas literalmente
    en la fila cruda; si la fila no cita ninguna, se asocian por id normalizado
    (minusculas y `-` -> `_`) los scripts cuyo stem es igual al id o empieza por
    el id mas `_`. Para ids de la forma `pNN-MMx` se prueba ademas sin el sufijo
    alfabetico.
  - `familia` se deriva del id y, si no es informativo, del stem del script.
  - `tipo` se deduce por palabras clave del stem del script (o del payload si la
    fila no tiene scripts) siguiendo un orden de prioridad fijo.
  - `reportes` y `citado_en_reportes`: hasta 5 rutas de `reports/`, `resultados/`
    o `figuras/` cuyo nombre (ruta relativa en minusculas) contiene el id en
    minusculas (con variantes `-`/`_`) o alguna variante del basename del script
    sin `.py`: el stem completo y, cuando el stem tiene la forma
    `<prefijo><NN>_<resto>` (por ejemplo `p67_01_controles_decisivos`), tambien
    `<prefijo>_<resto>` (`p67_controles_decisivos`), porque buena parte de los
    artefactos del proyecto se nombran a nivel de fase y no de indice. Para
    needles de menos de 3 caracteres se exige frontera de token para evitar
    coincidencias espurias. Cuando no hay ninguna coincidencia por id o script
    pero si la hay por la fase de la familia (`vNN`, `pNN`, `embedding`), esa
    evidencia de fase se registra aparte en el resumen y en `nota`, sin
    contarla como evidencia del experimento.
  - `en_notebook` / `celdas_notebook`: busqueda del id con frontera de palabra
    (clase `[A-Za-z0-9_]`, insensible a mayusculas) y de cada basename de script;
    se guardan hasta 5 indices de celda en base 0.
  - `estado`: con_evidencia_y_notebook | con_evidencia_sin_notebook |
    sin_evidencia | solo_notebook.

El codigo de este script es ASCII puro (docstrings y comentarios sin acentos);
el markdown de salida si lleva acentos y enie, construidos con escapes \\uXXXX.
No modifica ningun fichero existente: solo escribe los tres ficheros de salida
bajo `auditoria/`.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
import sys
from collections import Counter

# ===== SECCION 1: configuracion =====

CONFIG = {
    "registro": "experiments.csv",
    "dir_scripts": "experiments",
    "notebook": "notebooks/10_final_pipeline.ipynb",
    "dirs_evidencia": ("reports", "resultados", "figuras"),
    "salida_experimentos": "auditoria/inventario_experimentos.csv",
    "salida_scripts": "auditoria/inventario_scripts.csv",
    "salida_resumen": "auditoria/inventario_resumen.md",
    "ancho_registro": 23,
    "max_reportes": 5,
    "max_celdas": 5,
    "max_ids_listados": 80,
    "max_huecos_listados": 120,
    "fecha_re": r"\d{4}-\d{2}-\d{2}",
    "clase_frontera": r"[A-Za-z0-9_]",
    "prefijo_salida_permitido": "auditoria/",
}

# Reglas de clasificacion por palabras clave. El orden de la tupla define la
# prioridad: gana la primera regla que casa.
REGLAS_TIPO = (
    ("integracion_notebook", ("integrar_notebook", "integrar_v", "notebook", "documentacion")),
    ("submission", ("submission", "submissions")),
    ("scaffold", ("scaffold",)),
    ("auditoria", (
        "auditoria",
        "audit",
        "verificacion",
        "diagnostico",
        "diag_",
        "forense",
        "reconciliacion",
        "reconciliar",
        "compliance",
    )),
    ("cierre", (
        "cierre",
        "cerrar",
        "informe_final",
        "final_report",
        "final_summary",
        "postmortem",
        "sintesis",
        "conclusion",
        "dictamen",
        "paquete_candidato",
        "reporte_final",
    )),
    ("registro", (
        "registro",
        "registrar",
        "registry",
        "preinscripcion",
    )),
    ("ayudas", ("ayudas", "ayuda", "helper", "utils")),
    ("experimento", ()),
)

# Prioridad para agregar el tipo de una fila que tiene varios scripts.
PRIORIDAD_TIPO = (
    "integracion_notebook",
    "submission",
    "scaffold",
    "auditoria",
    "cierre",
    "registro",
    "ayudas",
    "experimento",
)

CABECERA_EXPERIMENTOS = (
    "id",
    "familia",
    "fecha",
    "scripts",
    "tipo",
    "en_registro",
    "fila_registro_valida",
    "reportes",
    "en_notebook",
    "celdas_notebook",
    "estado",
    "nota",
)

CABECERA_SCRIPTS = (
    "script",
    "familia",
    "tipo",
    "en_registro",
    "citado_en_reportes",
    "en_notebook",
    "nota",
)


# ===== SECCION 2: utilidades de lectura =====


def read_text_lines(path: str) -> list:
    """Lee un fichero de texto como lista de lineas sin el salto final.

    Args:
        path: ruta relativa del fichero.

    Returns:
        Lista de lineas (sin `\\n`); la primera es la cabecera.
    """
    with open(path, encoding="utf-8-sig") as fh:
        data = fh.read()
    return data.split("\n")[:-1] if data.endswith("\n") else data.split("\n")


def discover_scripts(dir_scripts: str) -> list:
    """Descubre los scripts `.py` del directorio indicado.

    Args:
        dir_scripts: directorio a explorar (no recursivo).

    Returns:
        Lista ordenada de rutas relativas `dir_scripts/*.py`.
    """
    return sorted(glob.glob(os.path.join(dir_scripts, "*.py")))


def stem_of(script_path: str) -> str:
    """Devuelve el nombre base de un script sin la extension `.py`."""
    return os.path.basename(script_path)[:-3]


def normalize_id(raw_id: str) -> str:
    """Normaliza un id: minusculas, espacios colapsados y guiones a guion bajo."""
    return re.sub(r"\s+", " ", raw_id.strip()).lower().replace("-", "_")


# ===== SECCION 3: lectura del registro =====


def read_registry_records(registro: str, ancho: int) -> list:
    """Lee `experiments.csv` conservando la linea cruda de cada fila.

    Detecta las filas malformadas por ancho distinto de `ancho` (las comas sin
    comillas las parten en varios campos). El id de una fila malformada es el
    primer token antes del primer caracter `|` de la linea cruda.

    Args:
        registro: ruta del fichero de registro.
        ancho: ancho esperado de columnas (23).

    Returns:
        Lista de diccionarios con claves: linea, raw, fields, width, valid, id.
    """
    lines = read_text_lines(registro)
    with open(registro, encoding="utf-8-sig", newline="") as fh:
        header = next(csv.reader(fh))
    esperado = len(header)
    records = []
    for offset, raw in enumerate(lines[1:]):
        fields = next(csv.reader([raw]))
        valid = len(fields) == esperado == ancho
        if valid:
            raw_id = fields[0]
        else:
            raw_id = raw.split("|", 1)[0]
        records.append(
            {
                "linea": offset + 2,
                "raw": raw,
                "fields": fields,
                "width": len(fields),
                "valid": valid,
                "id": re.sub(r"\s+", " ", raw_id.strip()),
            }
        )
    return records


def extract_date(record: dict, date_col: int, fecha_re: str) -> str:
    """Extrae la fecha `YYYY-MM-DD` de la fila (columna `date` o payload crudo)."""
    if record["valid"] and date_col < len(record["fields"]):
        candidate = record["fields"][date_col].strip()
        if re.fullmatch(fecha_re, candidate):
            return candidate
    match = re.search(fecha_re, record["raw"])
    return match.group(0) if match else ""


# ===== SECCION 4: familia, tipo y asociacion de scripts =====


def family_from_id(raw_id: str) -> str:
    """Deriva la familia de un id: vN, pN, embedding, e2 u otras.

    Args:
        raw_id: id tal cual aparece en el registro.

    Returns:
        Cadena de familia: `v1..v58`, `p45..p83`, `embedding`, `e2` u `otras`.
    """
    low = raw_id.strip().lower()
    match = re.match(r"^v(\d+)", low)
    if match:
        return "v" + match.group(1)
    match = re.match(r"^p(\d+)", low)
    if match:
        return "p" + match.group(1)
    if low.startswith("embedding"):
        return "embedding"
    if low.startswith("e2"):
        return "e2"
    return "otras"


def family_from_script(stem: str) -> str:
    """Deriva la familia de un script a partir de su stem."""
    low = stem.lower()
    match = re.match(r"^v(\d+)_", low)
    if match:
        return "v" + match.group(1)
    match = re.match(r"^p(\d+)_", low)
    if match:
        return "p" + match.group(1)
    if low.startswith("embedding"):
        return "embedding"
    return "otras"


def classify_tipo(text: str) -> str:
    """Clasifica el tipo de un experimento por palabras clave.

    Args:
        text: stem de script o payload crudo en minusculas.

    Returns:
        Uno de: experimento, auditoria, cierre, integracion_notebook,
        submission, scaffold, ayudas, registro, otro.
    """
    low = text.lower()
    if not low.strip():
        return "otro"
    for tipo, claves in REGLAS_TIPO:
        for clave in claves:
            if clave in low:
                return tipo
    return "experimento"


def aggregate_tipo(script_paths: list) -> str:
    """Agrega el tipo de una fila con varios scripts por prioridad fija."""
    if not script_paths:
        return "otro"
    tipos = {classify_tipo(stem_of(path)) for path in script_paths}
    for tipo in PRIORIDAD_TIPO:
        if tipo in tipos:
            return tipo
    return "otro"


def associate_scripts_by_id(raw_id: str, script_index: dict) -> list:
    """Asocia scripts a un id por prefijo normalizado del nombre base.

    Args:
        raw_id: id del registro.
        script_index: mapa stem_lower -> lista de rutas de script.

    Returns:
        Lista ordenada de rutas de script cuyo stem coincide con el id o
        empieza por el id normalizado mas `_`. Si no hay coincidencias y el id
        tiene la forma `pNN_MMx`, se reintenta sin el sufijo alfabetico final.
    """
    normalized = normalize_id(raw_id)
    paths = [path for stem, group in sorted(script_index.items()) if stem == normalized for path in group]
    paths += [
        path
        for stem, group in sorted(script_index.items())
        if stem.startswith(normalized + "_")
        for path in group
    ]
    if not paths:
        match = re.match(r"^([vp]\d+_\d+)[a-z]$", normalized)
        if match:
            prefix = match.group(1)
            paths = [
                path
                for stem, group in sorted(script_index.items())
                if stem == prefix
                for path in group
            ]
            paths += [
                path
                for stem, group in sorted(script_index.items())
                if stem.startswith(prefix + "_")
                for path in group
            ]
    return sorted(set(paths))


def cite_scripts_in_line(raw_lower: str, script_index: dict) -> list:
    """Devuelve los scripts cuyo stem aparece literalmente en la fila cruda."""
    return sorted(paths[0] for stem, paths in script_index.items() if stem in raw_lower)


def resolve_row_scripts(record: dict, script_index: dict) -> tuple:
    """Resuelve los scripts de una fila: citacion literal y, si falta, por id.

    Args:
        record: diccionario de registro con las claves `raw` e `id`.
        script_index: mapa stem_lower -> lista de rutas de script.

    Returns:
        Tupla (scripts, origen): lista ordenada de rutas y origen de la
        resolucion, uno de `citado`, `id` o `ninguno`.
    """
    cited = cite_scripts_in_line(record["raw"].lower(), script_index)
    if cited:
        return cited, "citado"
    associated = associate_scripts_by_id(record["id"], script_index)
    if associated:
        return associated, "id"
    return [], "ninguno"


def script_evidence_needles(stem: str) -> list:
    """Variantes de busqueda de evidencia para el stem de un script.

    Args:
        stem: nombre base del script sin `.py`.

    Returns:
        Lista ordenada con el stem en minusculas y, si el stem tiene la forma
        `<prefijo><NN>_<resto>`, la variante sin el indice numerico, que es la
        que usan los artefactos nombrados a nivel de fase.
    """
    low = stem.lower()
    needles = [low]
    match = re.match(r"^((?:v|p|embedding)\d+_)\d+_(.+)$", low)
    if match:
        needles.append(match.group(1) + match.group(2))
    return sorted(set(needles))


def phase_needle(familia: str) -> str:
    """Devuelve el token de fase de una familia, o cadena vacia si no aplica."""
    if re.fullmatch(r"(v|p)\d+", familia) or familia == "embedding":
        return familia
    return ""


# ===== SECCION 5: indices de evidencia y notebook =====


def build_evidence_index(dirs: tuple) -> list:
    """Indexa las evidencias de `reports/`, `resultados/` y `figuras/`.

    Args:
        dirs: tupla de directorios raiz, en orden de preferencia.

    Returns:
        Lista de diccionarios con claves `path` (ruta relativa) y
        `name_lower` (ruta relativa en minusculas), ordenada por directorio y
        ruta para que el inventario sea determinista.
    """
    index = []
    for root_order, root in enumerate(dirs):
        paths = sorted(
            path
            for path in glob.glob(os.path.join(root, "**", "*"), recursive=True)
            if os.path.isfile(path)
        )
        index.extend({"path": path, "name_lower": path.lower(), "orden": root_order} for path in paths)
    return index


def load_notebook_cells(path: str) -> list:
    """Carga el texto de cada celda del notebook (indice base 0).

    Args:
        path: ruta del notebook `.ipynb`.

    Returns:
        Lista de cadenas, una por celda, con el `source` unido.
    """
    with open(path, encoding="utf-8") as fh:
        notebook = json.load(fh)
    cells = []
    for cell in notebook.get("cells", []):
        source = cell.get("source", "")
        cells.append(source if isinstance(source, str) else "".join(source))
    return cells


def id_needles(raw_id: str) -> list:
    """Genera las variantes de busqueda de un id (minusculas, `-`/`_`)."""
    low = raw_id.strip().lower()
    variants = {
        low,
        low.replace("-", "_"),
        low.replace("_", "-"),
        low.replace("-", "_").replace("_", "-"),
    }
    return sorted(v for v in variants if v)


def evidence_matches(needle: str, name_lower: str) -> bool:
    """Indica si el needle aparece en la ruta; exige frontera si es muy corto."""
    if len(needle) < 3:
        pattern = r"(?<![a-z0-9])" + re.escape(needle) + r"(?![a-z0-9])"
        return re.search(pattern, name_lower) is not None
    return needle in name_lower


def find_evidence(needles: list, evidence: list, limit: int) -> list:
    """Devuelve hasta `limit` rutas de evidencia que contienen algun needle."""
    hits = []
    for item in evidence:
        if any(evidence_matches(needle, item["name_lower"]) for needle in needles):
            hits.append(item["path"])
            if len(hits) == limit:
                break
    return hits


def boundary_pattern(needle: str) -> "re.Pattern":
    """Compila un patron de needle con frontera de palabra tipo identificador."""
    return re.compile(
        r"(?<!" + CONFIG["clase_frontera"] + r")"
        + re.escape(needle)
        + r"(?!" + CONFIG["clase_frontera"] + r")",
        re.IGNORECASE,
    )


def find_notebook_cells(needles: list, cells: list, limit: int) -> list:
    """Devuelve hasta `limit` indices de celda (base 0) que citan algun needle."""
    patterns = [boundary_pattern(needle) for needle in needles if needle]
    if not patterns:
        return []
    hits = []
    for index, text in enumerate(cells):
        if any(pattern.search(text) for pattern in patterns):
            hits.append(index)
            if len(hits) == limit:
                break
    return hits


# ===== SECCION 6: construccion de filas =====


def build_script_index(scripts: list) -> dict:
    """Construye el mapa stem_lower -> [rutas] para todos los scripts."""
    index = {}
    for path in scripts:
        index.setdefault(stem_of(path).lower(), []).append(path)
    return {stem: sorted(paths) for stem, paths in index.items()}


def build_experiment_row(
    record: dict, row_scripts: list, origen: str, evidence: list, cells: list
) -> dict:
    """Construye la fila de inventario de un registro de `experiments.csv`.

    Args:
        record: registro con `id`, `raw`, `valid`, `width` y `linea`.
        row_scripts: scripts resueltos para la fila.
        origen: `citado`, `id` o `ninguno`, segun como se resolvieron.
        evidence: indice de evidencias.
        cells: celdas del notebook.

    Returns:
        Diccionario con las columnas de `inventario_experimentos.csv` mas dos
        claves internas de auditoria (`evidencia_fase` y `evidencia_grupo`).
    """
    raw_id = record["id"]
    needles = id_needles(raw_id)
    for path in row_scripts:
        needles.extend(script_evidence_needles(stem_of(path)))
    reportes = find_evidence(needles, evidence, CONFIG["max_reportes"])
    celdas = find_notebook_cells(
        [raw_id] + [stem_of(path).lower() for path in row_scripts], cells, CONFIG["max_celdas"]
    )
    familia = family_from_id(raw_id)
    if familia == "otras" and row_scripts:
        familia = family_from_script(stem_of(row_scripts[0]))
    tipo = aggregate_tipo(row_scripts) if row_scripts else classify_tipo(record["raw"])
    token = phase_needle(familia)
    evidencia_fase = find_evidence([token], evidence, CONFIG["max_reportes"]) if token else []
    nota = build_record_note(record, row_scripts, origen, bool(reportes), bool(evidencia_fase))
    estado = classify_estado(bool(reportes), bool(celdas))
    return {
        "id": raw_id,
        "familia": familia,
        "fecha": extract_date(record, 20, CONFIG["fecha_re"]),
        "scripts": ";".join(row_scripts),
        "tipo": tipo,
        "en_registro": "SI",
        "fila_registro_valida": "SI" if record["valid"] else "NO",
        "reportes": ";".join(reportes),
        "en_notebook": "SI" if celdas else "NO",
        "celdas_notebook": ";".join(str(c) for c in celdas),
        "estado": estado,
        "nota": nota,
        "evidencia_fase": bool(evidencia_fase) and not reportes,
    }


def build_script_rows(scripts: list, records: list, index: dict, evidence: list, cells: list) -> list:
    """Construye las 376 filas del inventario de scripts."""
    citados = set()
    for record in records:
        row_scripts, _ = resolve_row_scripts(record, index)
        citados.update(row_scripts)
    rows = []
    for path in scripts:
        stem = stem_of(path)
        needles = script_evidence_needles(stem)
        reportes = find_evidence(needles, evidence, CONFIG["max_reportes"])
        celdas = find_notebook_cells([stem.lower()], cells, CONFIG["max_celdas"])
        nota = build_script_note(path, citados, reportes, celdas)
        rows.append(
            {
                "script": path,
                "familia": family_from_script(stem),
                "tipo": classify_tipo(stem),
                "en_registro": "SI" if path in citados else "NO",
                "citado_en_reportes": ";".join(reportes),
                "en_notebook": "SI" if celdas else "NO",
                "nota": nota,
            }
        )
    return rows


def build_script_only_rows(
    scripts: list, citados: set, index: dict, evidence: list, cells: list
) -> list:
    """Crea una fila por cada script sin registro, en la misma forma que las de registro.

    Args:
        scripts: todos los scripts.
        citados: conjunto de rutas cubiertas por algun registro.
        index: mapa stem_lower -> [rutas].
        evidence: indice de evidencias.
        cells: celdas del notebook.

    Returns:
        Lista de filas de inventario con `id` = stem del script, `en_registro=NO`
        y `fila_registro_valida=NO`.
    """
    rows = []
    for path in scripts:
        if path in citados:
            continue
        stem = stem_of(path)
        needles = script_evidence_needles(stem)
        reportes = find_evidence(needles, evidence, CONFIG["max_reportes"])
        celdas = find_notebook_cells(needles, cells, CONFIG["max_celdas"])
        familia = family_from_script(stem)
        token = phase_needle(familia)
        evidencia_fase = find_evidence([token], evidence, CONFIG["max_reportes"]) if token else []
        if reportes:
            nota = "script sin fila en experiments.csv; alta directa desde experiments/"
        elif evidencia_fase:
            nota = (
                "script sin fila en experiments.csv; sin evidencia por id/script, "
                "hay evidencia a nivel de fase"
            )
        else:
            nota = "script sin fila en experiments.csv; sin evidencia atribuible"
        rows.append(
            {
                "id": stem,
                "familia": familia,
                "fecha": "",
                "scripts": path,
                "tipo": classify_tipo(stem),
                "en_registro": "NO",
                "fila_registro_valida": "NO",
                "reportes": ";".join(reportes),
                "en_notebook": "SI" if celdas else "NO",
                "celdas_notebook": ";".join(str(c) for c in celdas),
                "estado": classify_estado(bool(reportes), bool(celdas)),
                "nota": nota,
                "evidencia_fase": bool(evidencia_fase) and not reportes,
            }
        )
    return rows


def classify_estado(has_evidence: bool, has_notebook: bool) -> str:
    """Combina evidencia y notebook en uno de los cuatro estados posibles."""
    if has_evidence and has_notebook:
        return "con_evidencia_y_notebook"
    if has_evidence:
        return "con_evidencia_sin_notebook"
    if has_notebook:
        return "solo_notebook"
    return "sin_evidencia"


def build_record_note(
    record: dict, row_scripts: list, origen: str, has_evidence: bool, has_phase_evidence: bool
) -> str:
    """Redacta la nota corta de una fila de registro (menos de 160 caracteres)."""
    partes = []
    if not record["valid"]:
        partes.append(
            "fila malformada: ancho " + str(record["width"]) + " (esperado 23); "
            "comas sin comillas, linea " + str(record["linea"]) + " cruda conservada"
        )
    if not row_scripts:
        partes.append("sin script identificado en la fila ni por asociacion de id")
    elif origen == "id":
        partes.append("scripts asociados por id, no citados literalmente")
    if not has_evidence:
        if has_phase_evidence:
            partes.append("sin evidencia por id/script; hay evidencia a nivel de fase")
        else:
            partes.append("sin evidencia atribuible por id, script o fase")
    return "; ".join(partes)[:159]


def build_script_note(path: str, citados: set, reportes: list, celdas: list) -> str:
    """Redacta la nota corta de una fila de script (menos de 160 caracteres)."""
    partes = []
    if path not in citados:
        partes.append("sin fila de registro en experiments.csv")
    if not reportes:
        partes.append("sin cita en reports/resultados/figuras")
    if not celdas:
        partes.append("sin mencion en el notebook")
    return "; ".join(partes)[:159]


# ===== SECCION 7: auditoria del notebook =====


def notebook_sections(cells: list) -> list:
    """Extrae las secciones numeradas de nivel 2 (`## N. titulo`).

    Returns:
        Lista de tuplas (indice_celda, numero, titulo) en orden de aparicion.
    """
    sections = []
    pattern = re.compile(r"^(#{2})\s*(\d+)\.?\s*(.*)$")
    for index, text in enumerate(cells):
        for line in text.split("\n"):
            match = pattern.match(line.strip())
            if match:
                sections.append((index, int(match.group(2)), match.group(3).strip()))
                break
    return sections


def phase_tokens(text: str) -> set:
    """Extrae tokens de fase (`P45`, `V03`, ...) con frontera de identificador."""
    tokens = re.findall(r"(?<![A-Za-z0-9_])([PV]\d+)(?![0-9])", text, re.IGNORECASE)
    return {token.upper() for token in tokens}


def registry_phases(records: list) -> set:
    """Devuelve el conjunto de fases (`P67`, `V45`, ...) presentes en el registro."""
    phases = set()
    for record in records:
        match = re.match(r"^([PV]\d+)", record["id"].strip(), re.IGNORECASE)
        if match:
            phases.add(match.group(1).upper())
    return phases


def build_notebook_gaps(records: list, scripts: list, cells: list) -> dict:
    """Calcula los huecos de cobertura entre notebook, registro y scripts.

    Returns:
        Diccionario con: secciones sin registro, duplicados de seccion, fases
        con scripts sin ninguna mencion en el notebook y numeros de seccion
        ausentes en la numeracion.
    """
    phases = registry_phases(records)
    sections = notebook_sections(cells)
    sin_registro = []
    for index, numero, titulo in sections:
        tokens = phase_tokens(titulo)
        faltantes = sorted(token for token in tokens if token not in phases)
        if faltantes:
            alternativas = []
            for token in faltantes:
                letra, numero_fase = token[0], token[1:]
                contraria = ("V" if letra == "P" else "P") + numero_fase
                if contraria in phases:
                    alternativas.append(token + "->" + contraria)
            sin_registro.append((numero, index, titulo, faltantes, alternativas))

    repetidas = Counter((numero, titulo) for _, numero, titulo in sections)
    duplicadas = sorted((numero, titulo) for (numero, titulo), n in repetidas.items() if n > 1)

    numeros = sorted({numero for _, numero, _ in sections})
    ausentes = [n for n in range(1, max(numeros) + 1) if n not in set(numeros)]

    fases_scripts = {}
    for path in scripts:
        match = re.match(r"^([vp]\d+)_", stem_of(path).lower())
        if match:
            fases_scripts.setdefault(match.group(1).upper(), []).append(path)
    fases_sin_mencion = []
    for fase in sorted(fases_scripts, key=lambda f: (f[0], int(f[1:]))):
        letra, numero = fase[0], fase[1:]
        alterna = ("V" if letra == "P" else "P") + numero
        needles = [fase, alterna] + [stem_of(path).lower() for path in fases_scripts[fase]]
        if not find_notebook_cells(needles, cells, 1):
            fases_sin_mencion.append((fase, len(fases_scripts[fase])))

    return {
        "secciones_sin_registro": sin_registro,
        "secciones_duplicadas": duplicadas,
        "numeros_ausentes": ausentes,
        "fases_sin_mencion": fases_sin_mencion,
    }


# ===== SECCION 8: escritura de salidas =====


def ensure_inside_allowed(path: str) -> None:
    """Comprueba que la ruta de salida vive bajo `auditoria/`."""
    if not os.path.abspath(path).startswith(os.path.abspath(CONFIG["prefijo_salida_permitido"])):
        raise ValueError("ruta de salida fuera de auditoria/: " + path)


def write_csv(path: str, header: tuple, rows: list, columns: tuple) -> None:
    """Escribe un CSV determinista (utf-8, salto `\\n`, comillas minimas)."""
    ensure_inside_allowed(path)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow([row[column] for column in columns])


def listar_huecos_script(script_rows: list, limite: int) -> list:
    """Devuelve las filas de scripts sin registro, ordenadas por ruta."""
    return [
        row
        for row in sorted(script_rows, key=lambda r: r["script"])
        if row["en_registro"] == "NO"
    ][:limite]


def escribir_resumen(
    path: str,
    experiment_rows: list,
    script_rows: list,
    records: list,
    gaps: dict,
) -> None:
    """Escribe `auditoria/inventario_resumen.md` con conteos y huecos.

    Nota: el fichero de codigo es ASCII puro; los acentos y la enie del
    markdown se construyen aqui con escapes \\uXXXX.
    """
    ensure_inside_allowed(path)
    validas = sum(1 for record in records if record["valid"])
    malformadas = [record for record in records if not record["valid"]]
    con_evidencia = [row for row in experiment_rows if row["reportes"]]
    solo_fase = [row for row in experiment_rows if row["evidencia_fase"]]
    sin_evidencia = [
        row for row in experiment_rows if not row["reportes"] and not row["evidencia_fase"]
    ]
    sin_notebook = [row for row in experiment_rows if row["en_notebook"] == "NO"]
    estados = Counter(row["estado"] for row in experiment_rows)
    tipos_script = Counter(row["tipo"] for row in script_rows)
    sin_registro = listar_huecos_script(script_rows, CONFIG["max_huecos_listados"])
    sin_registro_total = sum(1 for row in script_rows if row["en_registro"] == "NO")
    sin_cita_reporte = sum(1 for row in script_rows if not row["citado_en_reportes"])

    lineas = []
    lineas.append("# Inventario unificado de experimentos")
    lineas.append("")
    lineas.append(
        "Generado por `auditoria/generar_inventario.py` el 2026-10-02 a partir de "
        "`experiments.csv`, `experiments/*.py`, `notebooks/10_final_pipeline.ipynb`, "
        "`reports/`, `resultados/` y `figuras/`. Las cifras son reproducibles: basta "
        "reejecutar el script con el venv del proyecto."
    )
    lineas.append("")

    lineas.append("## 1. Cobertura del registro y de los scripts")
    lineas.append("")
    lineas.append("| Concepto | Total |")
    lineas.append("| --- | ---: |")
    lineas.append("| Filas de datos en `experiments.csv` | " + str(len(records)) + " |")
    lineas.append("| Filas bien formadas (23 columnas) | " + str(validas) + " |")
    lineas.append(
        "| Filas malformadas (ancho distinto de 23, desde P67) | "
        + str(len(malformadas))
        + " |"
    )
    lineas.append("| Scripts `.py` en `experiments/` | " + str(len(script_rows)) + " |")
    lineas.append(
        "| Scripts con fila o asociaci\u00f3n de registro | "
        + str(sum(1 for row in script_rows if row["en_registro"] == "SI"))
        + " |"
    )
    lineas.append(
        "| Scripts sin fila de registro | " + str(sin_registro_total) + " |"
    )
    lineas.append("| Experimentos en el inventario | " + str(len(experiment_rows)) + " |")
    lineas.append(
        "| Experimentos con evidencia atribuida por id o script | "
        + str(len(con_evidencia))
        + " |"
    )
    lineas.append(
        "| Experimentos con evidencia solo a nivel de fase | "
        + str(len(solo_fase))
        + " |"
    )
    lineas.append(
        "| Experimentos sin evidencia atribuible | " + str(len(sin_evidencia)) + " |"
    )
    lineas.append("")

    lineas.append("## 2. Scripts por tipo")
    lineas.append("")
    lineas.append("| Tipo | Scripts |")
    lineas.append("| --- | ---: |")
    for tipo, total in sorted(tipos_script.items(), key=lambda item: (-item[1], item[0])):
        lineas.append("| " + tipo + " | " + str(total) + " |")
    lineas.append("")

    lineas.append("## 3. Estado de los experimentos")
    lineas.append("")
    lineas.append("| Estado | Experimentos |")
    lineas.append("| --- | ---: |")
    for estado in (
        "con_evidencia_y_notebook",
        "con_evidencia_sin_notebook",
        "sin_evidencia",
        "solo_notebook",
    ):
        lineas.append("| " + estado + " | " + str(estados.get(estado, 0)) + " |")
    lineas.append("")
    lineas.append(
        "Criterio de evidencia: una ruta de `reports/`, `resultados/` o `figuras/` "
        "cuenta si su nombre contiene el id (variantes `-`/`_`) o alguna variante del "
        "basename del script sin `.py` (stem completo o stem sin el \u00edndice "
        "num\u00e9rico). La evidencia que solo coincide con el token de fase "
        "(`v44`, `p60`, `embedding`) no se cuenta en `estado`: se informa aparte "
        "para no atribuir a un sub-experimento artefactos de toda la fase."
    )
    lineas.append("")

    lineas.append(
        "## 4. Experimentos sin documentar en el notebook ("
        + str(len(sin_notebook))
        + ")"
    )
    lineas.append("")
    lineas.append(
        "Ids sin ninguna menci\u00f3n en el notebook (se listan como m\u00e1ximo "
        + str(CONFIG["max_ids_listados"])
        + ", en orden de registro):"
    )
    lineas.append("")
    lineas.append("```")
    for row in sin_notebook[: CONFIG["max_ids_listados"]]:
        lineas.append(row["id"] + "\t" + row["familia"] + "\t" + row["tipo"])
    lineas.append("```")
    if len(sin_notebook) > CONFIG["max_ids_listados"]:
        lineas.append(
            "Lista truncada: se muestran "
            + str(CONFIG["max_ids_listados"])
            + " de "
            + str(len(sin_notebook))
            + " experimentos sin notebook."
        )
    lineas.append("")

    lineas.append(
        "## 5. Scripts sin fila de registro (" + str(sin_registro_total) + ")"
    )
    lineas.append("")
    lineas.append("| Script | Familia | Tipo | Evidencia | Notebook |")
    lineas.append("| --- | --- | --- | --- | --- |")
    for row in sin_registro:
        lineas.append(
            "| `"
            + row["script"]
            + "` | "
            + row["familia"]
            + " | "
            + row["tipo"]
            + " | "
            + ("SI" if row["citado_en_reportes"] else "NO")
            + " | "
            + row["en_notebook"]
            + " |"
        )
    lineas.append("")
    lineas.append(
        "Scripts sin ninguna cita en `reports/`, `resultados/` o `figuras/`: "
        + str(sin_cita_reporte)
        + " de "
        + str(len(script_rows))
        + ". La divergencia de nombres entre scripts (`p60_02_asl.py`) y artefactos "
        "(`p60_A_asl.json`) explica la mayor parte de las citas no encontradas."
    )
    lineas.append("")

    lineas.append("## 6. Secciones del notebook sin registro")
    lineas.append("")
    lineas.append(
        "Secciones de nivel 2 cuyo token de fase (`P45`, `V03`, ...) no existe en "
        "`experiments.csv`. La columna de alternativas se\u00f1ala fases registradas "
        "con la otra letra de nomenclatura:"
    )
    lineas.append("")
    lineas.append("| Secci\u00f3n | Celda | T\u00edtulo | Tokens sin registro | Alternativas |")
    lineas.append("| ---: | ---: | --- | --- | --- |")
    for numero, index, titulo, faltantes, alternativas in gaps["secciones_sin_registro"]:
        lineas.append(
            "| "
            + str(numero)
            + " | "
            + str(index)
            + " | "
            + titulo.replace("|", "/")
            + " | "
            + ", ".join(faltantes)
            + " | "
            + ", ".join(alternativas)
            + " |"
        )
    lineas.append("")
    lineas.append(
        "La columna Celda usa \\u00edndice base 0 del notebook. Los tokens `V03` "
        "corresponden al modelo campe\\u00f3n y los `P0`/`P7` a paquetes internos de "
        "la fase v2, no a fases del registro: se listan por rigor, pero no son "
        "huecos de registro."
    )
    lineas.append("")
    if gaps["secciones_duplicadas"]:
        lineas.append("### Secciones numeradas repetidas")
        lineas.append("")
        for numero, titulo in gaps["secciones_duplicadas"]:
            lineas.append("- Secci\u00f3n " + str(numero) + ": " + titulo.replace("|", "/"))
        lineas.append("")
    if gaps["numeros_ausentes"]:
        lineas.append(
            "### N\u00fameros de secci\u00f3n ausentes en la numeraci\u00f3n de nivel 2"
        )
        lineas.append("")
        lineas.append("- " + ", ".join(str(n) for n in gaps["numeros_ausentes"]))
        lineas.append("")
    if gaps["fases_sin_mencion"]:
        lineas.append(
            "### Fases con scripts en `experiments/` sin ninguna menci\u00f3n en el notebook"
        )
        lineas.append("")
        lineas.append("| Fase | Scripts |")
        lineas.append("| --- | ---: |")
        for fase, total in gaps["fases_sin_mencion"]:
            lineas.append("| " + fase + " | " + str(total) + " |")
        lineas.append("")

    lineas.append("## 7. Filas malformadas del registro")
    lineas.append("")
    lineas.append(
        "Las 43 filas malformadas usan un formato de barras (`id|fecha|...`) con "
        "comas sin comillas en el payload; `csv.reader` las parte en 1 a 12 campos. "
        "Su l\u00ednea cruda se conserva intacta en la fuente y se us\u00f3 para "
        "extraer id, fecha y scripts. L\u00edneas afectadas:"
    )
    lineas.append("")
    lineas.append("```")
    for record in malformadas:
        lineas.append(
            str(record["linea"])
            + "\t"
            + record["id"]
            + "\tancho="
            + str(record["width"])
        )
    lineas.append("```")
    lineas.append("")

    lineas.append("## 8. Huecos principales detectados")
    lineas.append("")
    for texto in huecos_principales(
        records, malformadas, script_rows, sin_registro, gaps, estados
    ):
        lineas.append("- " + texto)
    lineas.append("")

    ensure_inside_allowed(path)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("\n".join(lineas) + "\n")


def huecos_principales(
    records: list,
    malformadas: list,
    script_rows: list,
    sin_registro: list,
    gaps: dict,
    estados: Counter,
) -> list:
    """Redacta la lista de huecos mas graves del inventario."""
    fases = [fase for fase, _ in gaps["fases_sin_mencion"]]
    sin_registro_v = [row for row in script_rows if row["en_registro"] == "NO"]
    familias_huerfanas = Counter(row["familia"] for row in sin_registro_v)
    return [
        "El registro deja de ser CSV tabular en P67: "
        + str(len(malformadas))
        + " filas en formato de barras con comas sin comillas (anchos de 1 a 12), "
        "de modo que un lector estricto de CSV pierde 43 experimentos.",
        "Scripts sin fila de registro: "
        + str(len(sin_registro_v))
        + " de "
        + str(len(script_rows))
        + "; por familia destacan "
        + ", ".join(
            familia + "(" + str(total) + ")"
            for familia, total in familias_huerfanas.most_common(6)
        )
        + ".",
        "Fases con scripts en `experiments/` y ninguna menci\u00f3n en el notebook: "
        + (", ".join(fases) if fases else "ninguna")
        + ".",
        "El notebook no tiene secci\u00f3n propia para P78, P79, P80, P81, P82 ni P83; "
        "el \u00fanico `p80` del notebook es el percentil `umbral_de_dureza_p80`, "
        "no la fase P80.",
        "N\u00fameros de secci\u00f3n ausentes en la numeraci\u00f3n de nivel 2: "
        + (", ".join(str(n) for n in gaps["numeros_ausentes"]) if gaps["numeros_ausentes"] else "ninguno")
        + ".",
        "Secciones numeradas repetidas: "
        + (
            ", ".join(str(numero) for numero, _ in gaps["secciones_duplicadas"])
            if gaps["secciones_duplicadas"]
            else "ninguna"
        )
        + " (el bloque P61, secciones 81-88, aparece dos veces: celdas 265-286 y 290-311).",
        "Experimentos solo en notebook, sin evidencia: "
        + str(estados.get("solo_notebook", 0))
        + "; experimentos sin evidencia ni menci\\u00f3n en el notebook: "
        + str(estados.get("sin_evidencia", 0))
        + ".",
        "Nomenclatura divergente: el registro usa ids V45-xx a V58-xx mientras el "
        "notebook titula esas fases P45 a P58 (ver secci\u00f3n 6).",
        "La evidencia se nombra por fase o por letra y no por el \u00edndice del "
        "script (por ejemplo `p60_02_asl.py` frente a `p60_A_asl.json`), lo que "
        "impide atribuirla autom\u00e1ticamente sin revisi\u00f3n manual.",
    ]


# ===== SECCION 9: orquestacion =====


def run(config: dict) -> dict:
    """Ejecuta el pipeline completo y escribe las tres salidas.

    Returns:
        Diccionario con los conteos principales (para el resumen en consola).
    """
    scripts = discover_scripts(config["dir_scripts"])
    script_index = build_script_index(scripts)
    records = read_registry_records(config["registro"], config["ancho_registro"])
    cells = load_notebook_cells(config["notebook"])
    evidence = build_evidence_index(config["dirs_evidencia"])

    experiment_rows = []
    cubiertos = set()
    for record in records:
        row_scripts, origen = resolve_row_scripts(record, script_index)
        cubiertos.update(row_scripts)
        experiment_rows.append(
            build_experiment_row(record, row_scripts, origen, evidence, cells)
        )
    experiment_rows.extend(
        build_script_only_rows(scripts, cubiertos, script_index, evidence, cells)
    )
    script_rows = build_script_rows(scripts, records, script_index, evidence, cells)

    write_csv(
        config["salida_experimentos"],
        CABECERA_EXPERIMENTOS,
        experiment_rows,
        CABECERA_EXPERIMENTOS,
    )
    write_csv(config["salida_scripts"], CABECERA_SCRIPTS, script_rows, CABECERA_SCRIPTS)
    gaps = build_notebook_gaps(records, scripts, cells)
    escribir_resumen(config["salida_resumen"], experiment_rows, script_rows, records, gaps)

    return {
        "experimentos": len(experiment_rows),
        "scripts": len(script_rows),
        "malformadas": sum(1 for record in records if not record["valid"]),
        "sin_registro": sum(1 for row in script_rows if row["en_registro"] == "NO"),
        "sin_notebook": sum(1 for row in experiment_rows if row["en_notebook"] == "NO"),
        "con_evidencia": sum(1 for row in experiment_rows if row["reportes"]),
        "evidencia_fase": sum(1 for row in experiment_rows if row["evidencia_fase"]),
        "sin_evidencia": sum(
            1
            for row in experiment_rows
            if not row["reportes"] and not row["evidencia_fase"]
        ),
    }


def main() -> int:
    """Punto de entrada con manejo de errores y codigo de salida."""
    try:
        resumen = run(CONFIG)
        print("Inventario generado en auditoria/:")
        print("  experimentos: " + str(resumen["experimentos"]))
        print("  scripts: " + str(resumen["scripts"]))
        print("  filas malformadas: " + str(resumen["malformadas"]))
        print("  scripts sin registro: " + str(resumen["sin_registro"]))
        print("  experimentos sin notebook: " + str(resumen["sin_notebook"]))
        print("  con evidencia por id/script: " + str(resumen["con_evidencia"]))
        print("  con evidencia solo de fase: " + str(resumen["evidencia_fase"]))
        print("  sin evidencia atribuible: " + str(resumen["sin_evidencia"]))
        return 0
    except Exception:  # noqa: BLE001 - se reporta el traceback completo
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
