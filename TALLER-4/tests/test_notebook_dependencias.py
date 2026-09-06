"""test_notebook_dependencias.py

Fecha: 2026-09-05 (v4)
Descripcion: Tests de independencia y robustez del notebook entregable
    (notebooks/Taller_4_IR_CEV.ipynb, ensamblado desde notebook_cells/):

    (a) ninguna celda code importa scripts/ ni experimentos/ (ni usa
        importlib / sys.path+append hacia ellos); los docstrings pueden
        MENCIONAR la independencia ("no usa importlib..."), por eso el test
        detecta uso real (import_module/spec_from_file/append) y no la
        mera palabra.
    (b) ninguna celda markdown referencia ../imagenes/ (no renderiza en
        GitHub/Colab; las figuras van inline o via mostrar_png).
    (c) todo literal de archivo (.jsonl/.txt/.csv/.json/.png) citado entre
        comillas en celdas code es un archivo versionado en git o esta en la
        lista blanca de archivos >100 MB con guard .exists() en la misma
        celda. En los bloques expandidos de scripts/01-03 (celdas [auto] del
        ensamblador, v4) el guard se exige a nivel de BLOQUE: el ensamblador
        parte el script canonico en varias celdas, asi que la unidad con
        sentido es el bloque completo (equivale a la celda unica de la v3).
        Lista blanca (crece documentada): el JSON crudo de entrevistas y las
        normalizadas/tokens (generados por scripts/02; >100 MB, no
        versionables). Se ignoran literales con interpolacion (llaves) y con
        espacios (nombres descriptivos).
    (d) las celdas que resuelven la raiz (def buscar_raiz / _raiz_standalone)
        lanzan FileNotFoundError con el marcador segmentos_libros.jsonl y no
        contienen la asignacion silenciosa del ultimo candidato.
    (e) expansion v4 completa: (e1) el ipynb NO contiene CODIGO_EMBED_ ni
        "exec(" (v3 embebia los scripts como cadenas ejecutadas con exec; v4
        los expande como celdas de codigo normales); (e2) hay celdas code con
        el codigo de scripts/01-03 (distintivos ORDEN_TOMOS de 01, def
        construir_nlp / cargar_stopwords_habla de 02 y def etapa_estadisticas
        de 03); (e3) estan los alias [auto] main_extraccion / main_preprocesar
        / main_eda; (e4) REGEN_AUTOMATICO esta definido en False; (e5) no
        quedan marcadores de expansion ni de embebido sin resolver.
    (f) el ipynb contiene la celda de diagnostico de prerrequisitos.

El notebook debe estar ensamblado (scripts/ensamblar_notebook_t4.py); si no
existe, el modulo se omite con pytest.skip.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

RUTA_PROYECTO = Path(__file__).resolve().parents[1]
RUTA_NOTEBOOK = RUTA_PROYECTO / "notebooks" / "Taller_4_IR_CEV.ipynb"

# Archivos >100 MB no versionados con respaldo estatico o regenerables;
# cualquier literal de estos exige guard .exists() en la misma celda.
LISTA_BLANCA = {
    "entrevistas_all_2023-03-21_14-24_05.json",  # JSON crudo (Drive del taller)
    "entrevistas_normalizadas.jsonl",  # genera scripts/02 normalizar
    "entrevistas_tokens.txt",  # genera scripts/02 spacy (191 MB)
    "paginas.jsonl",  # intermedio de scripts/01 en data/corpus/paginas/ (.gitignore)
}

_PATRON_LITERAL = re.compile(r"""(['"])([^'"\n]*?\.(?:jsonl|txt|csv|json|png))\1""")
_PATRON_IMPORT_RELATIVO = re.compile(
    r"^\s*(?:import|from)\s+(?:scripts|experimentos)(?:\s|\.|$)", re.MULTILINE
)

EXTENSIONES = (".jsonl", ".txt", ".csv", ".json", ".png")


def _celdas_code() -> list[str]:
    """Devuelve el source de todas las celdas code del notebook.

    Returns:
        Lista con el texto de cada celda code.

    Raises:
        pytest.skip: si el notebook no existe (falta ensamblar).
    """
    if not RUTA_NOTEBOOK.exists():
        pytest.skip(
            f"No existe {RUTA_NOTEBOOK}: ejecuta primero "
            "scripts/ensamblar_notebook_t4.py"
        )
    with RUTA_NOTEBOOK.open(encoding="utf-8") as fh:
        nb = json.load(fh)
    return [
        "".join(celda["source"])
        for celda in nb["cells"]
        if celda["cell_type"] == "code"
    ]


def _celdas_markdown() -> list[str]:
    """Devuelve el source de todas las celdas markdown del notebook."""
    with RUTA_NOTEBOOK.open(encoding="utf-8") as fh:
        nb = json.load(fh)
    return [
        "".join(celda["source"])
        for celda in nb["cells"]
        if celda["cell_type"] == "markdown"
    ]


def _bloques_expandidos(celdas: list[str]) -> list[tuple[int, int]]:
    """Rangos de celdas de cada bloque expandido [auto] del ensamblador.

    Un bloque va desde la celda con el comentario [auto] del ensamblador
    hasta la celda de alias ('main_<etapa> = main') que lo cierra (v4: los
    scripts/01-03 viajan partidos en varias celdas code normales).

    Args:
        celdas: Sources de las celdas code del notebook.

    Returns:
        Lista de (inicio, fin) inclusivos por bloque expandido.
    """
    bloques: list[tuple[int, int]] = []
    inicio: int | None = None
    for indice, celda in enumerate(celdas):
        if celda.startswith("# [auto] Codigo expandido desde scripts/"):
            inicio = indice
        elif inicio is not None and re.match(r"^main_\w+ = main", celda):
            bloques.append((inicio, indice))
            inicio = None
    return bloques


def _archivos_versionados() -> tuple[set[str], set[str]]:
    """Consulta git ls-files y devuelve basenames y rutas relativas a TALLER-4.

    Returns:
        Tupla (basenames, rutas sin el prefijo TALLER-4/) de archivos
        versionados en el repositorio.
    """
    proc = subprocess.run(
        ["git", "ls-files"],
        cwd=RUTA_PROYECTO,
        capture_output=True,
        text=True,
        check=False,
    )
    rutas = [linea for linea in proc.stdout.splitlines() if linea.strip()]
    basenames = {Path(ruta).name for ruta in rutas}
    relativas = {ruta.removeprefix("TALLER-4/") for ruta in rutas}
    return basenames, relativas


def test_a_sin_imports_de_scripts_ni_experimentos() -> None:
    """(a) Ninguna celda code importa scripts/, experimentos/, importlib o
    sys.path con append hacia esas carpetas (uso real, no menciones)."""
    for celda in _celdas_code():
        assert not _PATRON_IMPORT_RELATIVO.search(celda), (
            "import relativo de scripts/experimentos en una celda code"
        )
        if "sys.path" in celda:
            for mutacion in (
                "sys.path.append",
                "sys.path.insert",
                "sys.path.extend",
                "sys.path[0:0]",
            ):
                assert mutacion not in celda, f"{mutacion} presente en una celda code"
        if "importlib" in celda:
            for uso in ("import_module", "spec_from_file", "exec_module"):
                assert uso not in celda, f"importlib.{uso} presente en una celda code"


def test_b_sin_referencias_imagenes_relativas() -> None:
    """(b) Ninguna celda markdown referencia ../imagenes/."""
    for celda in _celdas_markdown():
        assert "../imagenes/" not in celda, "referencia ../imagenes/ en markdown"


def test_c_literales_de_archivo_versionados_o_lista_blanca() -> None:
    """(c) Literales de archivo en celdas code: versionados o lista blanca
    con guard .exists() (en la misma celda o en su bloque expandido [auto]).

    El ensamblador v4 parte cada script canonico (scripts/01-03) en varias
    celdas code normales (bloque [auto] + alias); un literal de la lista
    blanca citado en una de esas celdas (p. ej. en un docstring) se valida
    contra el guard .exists() presente en CUALQUIER celda del MISMO bloque,
    porque la unidad con sentido es el script completo (equivale a la celda
    unica de la v3). Fuera de bloques [auto] la regla sigue siendo estricta
    por celda.
    """
    basenames, rutas_relativas = _archivos_versionados()
    celdas = _celdas_code()
    bloques = _bloques_expandidos(celdas)

    def texto_para(indice: int) -> str:
        """Devuelve el texto de la celda o el de su bloque expandido [auto].

        Args:
            indice: Indice de la celda code en la lista de sources.

        Returns:
            Source de la celda si no pertenece a un bloque expandido; si
            pertenece, el texto del bloque completo (celdas [auto] + alias).
        """
        for inicio, fin in bloques:
            if inicio <= indice <= fin:
                return "\n".join(celdas[inicio : fin + 1])
        return celdas[indice]

    for indice, celda in enumerate(celdas):
        texto_validado = texto_para(indice)
        for literal in _PATRON_LITERAL.findall(texto_validado):
            candidato = literal[1]
            if (
                not candidato
                or "{" in candidato
                or "}" in candidato
                or "$" in candidato
                or " " in candidato
                or candidato.startswith(".")
                or not candidato.endswith(EXTENSIONES)
            ):
                continue  # interpolacion o no es un archivo
            nombre = Path(candidato).name
            versionado = nombre in basenames or candidato in rutas_relativas
            if versionado:
                continue
            assert nombre in LISTA_BLANCA, (
                f"literal de archivo no versionado y fuera de la lista blanca: "
                f"{candidato!r}"
            )
            assert ".exists()" in texto_validado, (
                f"literal en lista blanca {candidato!r} sin guard .exists() "
                "en la misma celda ni en su bloque expandido [auto]"
            )


def test_d_resolucion_de_raiz_con_error_claro() -> None:
    """(d) Las celdas que resuelven la raiz lanzan FileNotFoundError claro y
    no asignan silenciosamente el ultimo candidato sin verificar."""
    celdas_resolutoras = [
        celda
        for celda in _celdas_code()
        if "def buscar_raiz" in celda or "def _raiz_standalone" in celda
    ]
    assert celdas_resolutoras, "no se encontro ninguna celda que defina buscar_raiz"
    for celda in celdas_resolutoras:
        assert "FileNotFoundError" in celda, "falta FileNotFoundError en la celda"
        assert "segmentos_libros.jsonl" in celda, "falta el marcador en la celda"
        assert "raise FileNotFoundError" in celda, "la celda debe lanzar el error"
    for celda in _celdas_code():
        assert "RUTA_RAIZ = _candidato" not in celda, (
            "asignacion silenciosa del ultimo candidato presente en una celda"
        )


def test_e_expansion_completa_sin_embebido() -> None:
    """(e) Expansion v4: sin CODIGO_EMBED_ ni exec(; celdas con el codigo de
    los scripts 01-03, aliases [auto] presentes, REGEN_AUTOMATICO en False y
    sin marcadores de expansion/embebido sin resolver."""
    code = _celdas_code()
    todas = code + _celdas_markdown()
    texto_total = "\n".join(todas)
    # (e1) v4 expande los scripts como celdas normales: nada de codigo-como-
    # string (CODIGO_EMBED_*) ni llamadas ejecutadas con exec().
    assert "CODIGO_EMBED_" not in texto_total, (
        "queda codigo-como-string (CODIGO_EMBED_*) en el notebook"
    )
    assert "exec(" not in texto_total, (
        "queda una llamada exec( en el notebook (v4 usa celdas normales)"
    )
    # (e2) distintivos del codigo de cada script expandido (celdas code).
    celdas_01 = [
        celda
        for celda in code
        if "# [auto] Codigo expandido desde scripts/01_extraer_libros.py" in celda
    ]
    assert celdas_01 and any("ORDEN_TOMOS" in celda for celda in celdas_01), (
        "no aparece el codigo de scripts/01 (ORDEN_TOMOS) en celdas expandidas"
    )
    assert any(
        "def construir_nlp" in celda or "def cargar_stopwords_habla" in celda
        for celda in code
    ), "no aparece el codigo de scripts/02 (construir_nlp/stopwords) expandido"
    assert any("def etapa_estadisticas" in celda for celda in code), (
        "no aparece el codigo de scripts/03 (etapa_estadisticas) expandido"
    )
    # (e3) aliases [auto] que cierran el bloque de cada script.
    assert any("main_extraccion = main" in celda for celda in code), (
        "falta el alias main_extraccion (scripts/01)"
    )
    assert any("main_preprocesar = main" in celda for celda in code), (
        "falta el alias main_preprocesar (scripts/02)"
    )
    assert any("main_eda = main" in celda for celda in code), (
        "falta el alias main_eda (scripts/03)"
    )
    # (e4) flags de regeneracion apagados por defecto.
    assert any(
        re.search(r"REGEN_AUTOMATICO\s*=\s*False", celda) for celda in code
    ), "REGEN_AUTOMATICO debe estar definido en False"
    # (e5) sin marcadores de expansion/embebido sin resolver.
    assert not any("[expand:" in celda or "[embed:" in celda for celda in todas), (
        "quedo un marcador [expand: o [embed: sin resolver en el notebook"
    )


def test_f_celda_de_diagnostico_de_prerrequisitos() -> None:
    """(f) El notebook contiene la celda de diagnostico de prerrequisitos."""
    assert any("prerrequisitos" in celda for celda in _celdas_code()), (
        "no aparece la celda de diagnostico de prerrequisitos"
    )
