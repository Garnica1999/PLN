"""test_notebook_dependencias.py

Fecha: 2026-09-05 (v5)
Descripcion: Tests de independencia total y robustez del notebook entregable
    (notebooks/Taller_4_IR_CEV.ipynb, ensamblado desde notebook_cells/):

    (a) ninguna celda code importa scripts/ ni experimentos/ (ni usa
        importlib / sys.path+append hacia ellos); los docstrings pueden
        MENCIONAR la independencia ("no usa importlib..."), por eso el test
        detecta uso real (import_module/spec_from_file/append) y no la
        mera palabra.
    (b) ninguna celda markdown referencia ../imagenes/ (no renderiza en
        GitHub/Colab; las figuras van inline o via mostrar_png).
    (c) politica v5 de lecturas: lista blanca de lecturas directas SIN
        guard = SOLO los archivos de data/ de los 3 insumos permitidos
        (entrevistas_all_2023-03-21_14-24_05.json, zip/PDFs de libros,
        Resource-tesauro.rdf) y los caches derivados de data/corpus
        (segmentos_libros.jsonl, segmentos/<TOMO>.jsonl,
        {libros,entrevistas}_{lemas,tokens}.txt, *_docs.json,
        entrevistas_normalizadas.jsonl). TODO lo demas (config/,
        resultados/, imagenes/) solo puede aparecer en contexto OPCIONAL:
        una celda code (o su bloque expandido [auto]) que contenga
        `.exists()`, `try:`, o las funciones con fallback documentado
        `verificar_consistencia_config`/`mostrar_png`. En los bloques
        expandidos de scripts/01-03 (celdas [auto] del ensamblador) la
        unidad con sentido es el bloque completo: sus lecturas solo
        ocurren cuando el driver de regeneracion los invoca.
    (d) las celdas que resuelven la raiz (def buscar_raiz / _raiz_standalone)
        lanzan FileNotFoundError con el marcador segmentos_libros.jsonl y no
        contienen la asignacion silenciosa del ultimo candidato.
    (e) expansion v4/v5 completa: (e1) el ipynb NO contiene CODIGO_EMBED_ ni
        "exec("; (e2) hay celdas code con el codigo de scripts/01-03; (e3)
        estan los alias [auto] main_extraccion / main_preprocesar /
        main_eda; (e4) REGEN_AUTOMATICO esta definido en True (v5: la
        regeneracion automatica es el comportamiento por defecto); (e5) no
        quedan marcadores de expansion ni de embebido sin resolver.
    (f) el ipynb contiene la celda de diagnostico de prerrequisitos con la
        semantica v5 (caches REQUERIDOS autogenerables y opcionales).
    (g) SANDBOX de independencia: se ejecuta el notebook con nbclient
        (timeout 600 s/celda) desde un arbol temporal que contiene SOLO el
        .ipynb + los caches derivados de data/corpus (symlinks a los
        archivos reales, sin duplicar disco) y un data/raw VACIO: sin
        config/, resultados/, imagenes/, scripts/ ni notebook_cells/. El
        resultado esperado es 0 errores, con los avisos "[aviso]/[info]"
        de archivos opcionales ausentes y los calculos inline completos
        (figuras PNG inline y auto-checks con valores).

El notebook debe estar ensamblado (scripts/ensamblar_notebook_t4.py); si no
existe, el modulo se omite con pytest.skip.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

RUTA_PROYECTO = Path(__file__).resolve().parents[1]
RUTA_NOTEBOOK = RUTA_PROYECTO / "notebooks" / "Taller_4_IR_CEV.ipynb"

# (c) Lista blanca v5: lecturas directas permitidas SIN guard. Solo
# archivos de data/: los 3 insumos (descarga manual del Drive del taller) y
# los caches derivados de data/corpus que el propio notebook genera en la
# seccion Regeneracion. TODO lo demas exige contexto opcional.
LISTA_BLANCA_DIRECTA = {
    # --- 3 insumos permitidos (data/raw, descarga manual) ---
    "entrevistas_all_2023-03-21_14-24_05.json",
    "books-20260904T025322Z-1-001.zip",
    "Resource-tesauro.rdf",
    # --- caches derivados de data/corpus (autogenerados) ---
    "segmentos_libros.jsonl",
    "entrevistas_normalizadas.jsonl",
    "libros_lemas.txt",
    "libros_tokens.txt",
    "entrevistas_lemas.txt",
    "entrevistas_tokens.txt",
    "libros_docs.json",
    "entrevistas_docs.json",
    "validacion_entrevistas_lemas.txt",
    "validacion_entrevistas_tokens.txt",
    "validacion_entrevistas_docs.json",
}
# Basenames bajo config/, resultados/ e imagenes/ (git): su literal en una
# celda code SOLO puede aparecer en contexto opcional.
_DIRS_OPCIONALES = ("config/", "resultados/", "imagenes/")

_PATRON_LITERAL = re.compile(r"""(['"])([^'"\n]*?\.(?:jsonl|txt|csv|json|png|rdf|zip))\1""")
_PATRON_IMPORT_RELATIVO = re.compile(
    r"^\s*(?:import|from)\s+(?:scripts|experimentos)(?:\s|\.|$)", re.MULTILINE
)

EXTENSIONES = (".jsonl", ".txt", ".csv", ".json", ".png", ".rdf", ".zip")


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
    hasta la celda de alias ('main_<etapa> = main') que lo cierra (los
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


def _archivos_basename_bajo_dirs_opcionales() -> set[str]:
    """Basenames versionados en git bajo config/, resultados/ e imagenes/.

    Returns:
        Conjunto de basenames; los literales con estos nombres en celdas
        code exigen contexto opcional (politica v5).
    """
    proc = subprocess.run(
        ["git", "ls-files"],
        cwd=RUTA_PROYECTO,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        Path(linea).name
        for linea in proc.stdout.splitlines()
        if linea.strip()
        and linea.startswith(_DIRS_OPCIONALES)
        and Path(linea).suffix in EXTENSIONES
    }


def _es_archivo_data_permitido(candidato: str) -> bool:
    """True si el literal apunta a data/ (insumo o cache derivado).

    Args:
        candidato: literal de archivo citado en una celda code.

    Returns:
        True si el literal es un nombre de la lista blanca, un PDF suelto
        de libros (data/raw/books/*.pdf) o un segmentos/<TOMO>.jsonl.
    """
    nombre = Path(candidato).name
    if nombre in LISTA_BLANCA_DIRECTA:
        return True
    if candidato.endswith(".pdf"):
        return True  # PDFs sueltos de libros en data/raw/books
    if re.fullmatch(r"[A-Z]+\.jsonl", nombre) and candidato.endswith(".jsonl"):
        # data/corpus/segmentos/<TOMO>.jsonl (celdas [auto] de scripts/01)
        return nombre in {
            "HALLAZGOS.jsonl", "CONVOCATORIA.jsonl", "NARRATIVA.jsonl",
            "SUFRIR.jsonl", "TESTIMONIAL.jsonl", "CUERPO.jsonl", "NNA.jsonl",
            "VIOLACIONES.jsonl", "ETNICO.jsonl", "FUERA.jsonl",
        }
    return False


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


def test_c_politica_v5_de_lecturas_de_archivo() -> None:
    """(c) Politica v5: solo data/ (insumos + caches) sin guard; todo lo
    demas (config/, resultados/, imagenes/) en contexto OPCIONAL.

    Contexto opcional = la celda code (o su bloque expandido [auto])
    contiene `.exists()`, `try:`, o las funciones con fallback documentado
    `verificar_consistencia_config`/`mostrar_png`. Fuera de bloques [auto]
    la regla se aplica por celda; dentro de un bloque [auto] se aplica al
    bloque completo (sus lecturas solo ocurren si el driver de la seccion
    Regeneracion los invoca).
    """
    opcionales_por_nombre = _archivos_basename_bajo_dirs_opcionales()
    celdas = _celdas_code()
    bloques = _bloques_expandidos(celdas)

    def texto_para(indice: int) -> str:
        """Texto de la celda o de su bloque expandido [auto] completo.

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

    def contexto_opcional(texto_bloque: str) -> bool:
        """True si el bloque contiene un guard o funcion con fallback."""
        return (
            ".exists(" in texto_bloque
            or "try:" in texto_bloque
            or "verificar_consistencia_config" in texto_bloque
            or "mostrar_png" in texto_bloque
        )

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
            if _es_archivo_data_permitido(candidato):
                continue  # lista blanca v5: data/ insumos + caches derivados
            nombre = Path(candidato).name
            if nombre not in opcionales_por_nombre:
                # Literal desconocido: exigir guard por prudencia.
                assert ".exists()" in texto_validado, (
                    f"literal de archivo fuera de la lista blanca y sin guard: "
                    f"{candidato!r}"
                )
                continue
            assert contexto_opcional(texto_validado), (
                f"literal de config/resultados/imagenes {candidato!r} sin "
                "contexto opcional (.exists(), try:, verificar_consistencia_"
                "config o mostrar_png con fallback) en la celda ni en su "
                "bloque expandido [auto]"
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
    """(e) Expansion v4/v5: sin CODIGO_EMBED_ ni exec(; celdas con el codigo
    de los scripts 01-03, aliases [auto] presentes, REGEN_AUTOMATICO en True
    y sin marcadores de expansion/embebido sin resolver."""
    code = _celdas_code()
    todas = code + _celdas_markdown()
    texto_total = "\n".join(todas)
    # (e1) los scripts viajan como celdas normales: nada de codigo-como-
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
    # (e4) v5: regeneracion automatica por defecto (REGEN_AUTOMATICO=True).
    assert any(
        re.search(r"REGEN_AUTOMATICO\s*=\s*True", celda) for celda in code
    ), "REGEN_AUTOMATICO debe estar definido en True (v5)"
    # (e5) sin marcadores de expansion/embebido sin resolver.
    assert not any("[expand:" in celda or "[embed:" in celda for celda in todas), (
        "quedo un marcador [expand: o [embed: sin resolver en el notebook"
    )


def test_f_celda_de_diagnostico_de_prerrequisitos() -> None:
    """(f) Celda de diagnostico con la semantica v5: checklist de caches
    REQUERIDOS autogenerables y OPCIONALES (config embebida, resultados e
    imagenes nunca requeridos)."""
    code = _celdas_code()
    assert any("prerrequisitos" in celda for celda in code), (
        "no aparece la celda de diagnostico de prerrequisitos"
    )
    assert any("REQUERIDOS" in celda for celda in code), (
        "el diagnostico debe listar los caches REQUERIDOS"
    )
    assert any("OPCIONALES" in celda for celda in code), (
        "el diagnostico debe listar los OPCIONALES"
    )
    assert any("Regeneracion" in celda and "descarga manual" in celda for celda in code), (
        "el aviso de regeneracion/descarga manual no aparece en el diagnostico"
    )


# ---------------------------------------------------------------------------
# (g) Sandbox de independencia: ejecutar el notebook sin config/, resultados/,
# imagenes/, scripts/ ni notebook_cells/, con un data/raw vacio.
# ---------------------------------------------------------------------------

_SCRIPT_EJECUTOR = """
import json
import sys
import nbformat
from nbclient import NotebookClient

ruta = sys.argv[1]
nb = nbformat.read(ruta, as_version=4)
cliente = NotebookClient(nb, timeout=600, kernel_name="python3",
                         resources={"metadata": {"path": "."}})
try:
    cliente.execute()
except Exception as exc:  # noqa: BLE001
    print("SANDBOX_EXCEPTION:", type(exc).__name__, exc)
    sys.exit(3)
errores = 0
texto_salidas = []
for celda in nb.cells:
    for salida in celda.get("outputs", []):
        if salida.get("output_type") == "error":
            errores += 1
            texto_salidas.append(
                "ERROR " + "".join(salida.get("traceback", []))[:500]
            )
        elif salida.get("output_type") == "stream":
            texto_salidas.append(salida.get("text", ""))
nbformat.write(nb, ruta)
print("SANDBOX_ERRORES:", errores)
print("SANDBOX_TEXTO:")
print("\\n".join(texto_salidas))
sys.exit(0 if errores == 0 else 4)
"""


def _construir_arbol_sandbox(destino: Path) -> None:
    """Construye el arbol temporal del sandbox (symlinks, sin copiar GB).

    Solo se enlazan: el .ipynb y los caches derivados de data/corpus que la
    seccion de pruebas necesita para los calculos inline (segmentos, lemas,
    tokens y docs de ambos corpus, y las normalizadas). NO se enlazan
    config/, resultados/, imagenes/, scripts/ ni notebook_cells/, y data/raw
    queda VACIO (sin los 3 insumos).

    Args:
        destino: Raiz del arbol temporal (tmp_path de pytest).
    """
    destino.joinpath("data", "raw").mkdir(parents=True)
    destino.joinpath("data", "corpus", "preprocesado").mkdir(parents=True)
    enlaces = [
        "data/corpus/segmentos_libros.jsonl",
        "data/corpus/entrevistas_normalizadas.jsonl",
        "data/corpus/preprocesado/libros_lemas.txt",
        "data/corpus/preprocesado/libros_tokens.txt",
        "data/corpus/preprocesado/libros_docs.json",
        "data/corpus/preprocesado/entrevistas_lemas.txt",
        "data/corpus/preprocesado/entrevistas_tokens.txt",
        "data/corpus/preprocesado/entrevistas_docs.json",
    ]
    for relativa in enlaces:
        origen = RUTA_PROYECTO / relativa
        if not origen.exists():
            pytest.skip(f"falta el cache {relativa} para el sandbox")
        (destino / relativa).symlink_to(origen)
    # Copia del .ipynb (pocos MB; evita problemas de lectura por symlink).
    import shutil

    shutil.copyfile(RUTA_NOTEBOOK, destino / "Taller_4_IR_CEV.ipynb")


def _ejecutar_sandbox(tmp_path: Path) -> None:
    """Ejecuta el notebook en el arbol temporal y verifica la salida.

    Args:
        tmp_path: Directorio temporal de pytest.
    """
    _construir_arbol_sandbox(tmp_path)
    ruta_nb = tmp_path / "Taller_4_IR_CEV.ipynb"
    env = dict(os.environ)
    env.pop("TALLER4_ROOT", None)
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT_EJECUTOR, str(ruta_nb)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
        check=False,
    )
    salida = proc.stdout + proc.stderr
    assert proc.returncode in (0, 4), (
        f"nbclient fallo (rc={proc.returncode}): {salida[-4000:]}"
    )
    if proc.returncode == 4:
        pytest.fail(f"el notebook tuvo celdas con error en el sandbox: {salida[-4000:]}")
    # Verificaciones de la salida esperada (independencia total).
    esperados = [
        "[aviso]",
        "[info]",
        "Config embebida (independencia total)",
        "Muestra determinista inline: 30 pares",
        "P@1 estricto (solo 'relacionado')         : 0.5333",
        "Segmentos de libro por tomo (calculado inline",
    ]
    faltantes = [texto for texto in esperados if texto not in salida]
    assert not faltantes, (
        "el sandbox no produjo la salida esperada; faltan: "
        f"{faltantes}. Salida:\n{salida[-6000:]}"
    )


def test_g_sandbox_independencia(tmp_path: Path) -> None:
    """(g) Sandbox real (nbclient, ~1-2 min): 0 errores sin
    config/resultados/imagenes/scripts y con avisos [aviso]/[info]."""
    try:
        import nbclient  # noqa: F401
    except ImportError:
        pytest.skip("nbclient no esta instalado")
    _ejecutar_sandbox(tmp_path)
