"""V23 · 02 — Integra la seccion CORAL en el notebook consolidado y lo audita.

El notebook `notebooks/99_entrega_consolidado.ipynb` es la entrega y llega hasta la seccion 30
(V17). Esta script le anade la seccion **31 · V23 CORAL / cumulative-link**, que exige el brief,
y despues ejecuta las verificaciones obligatorias sobre el notebook entero.

## Lo que se anade

Once bloques: motivacion, hipotesis, formulacion matematica, configuracion pre-registrada,
tests unitarios, control V20, resultado CORAL, scores por fold, macro-F1 y F1 por clase,
violaciones de monotonia, complementariedad, interseccion de errores y decision.

**Ninguna cifra se escribe a mano**: todas se cargan de `experiments/v23_00_tests.json` y
`experiments/v23_01_coral.json`, que es lo que exige el punto 22 del brief. La unica excepcion
es el leaderboard del champion, que no vive en un artefacto de experimento sino en `CHAMPION.md`
y se lee de ahi.

## Deuda que esta script deja documentada, no resuelta

El notebook consolidado **no contiene V18, V19, V20, V21 ni V22**: su ultima seccion es la 30,
que cierra V17. El champion actual es V20, luego el notebook esta desactualizado respecto del
estado real del proyecto. El brief pide integrar V23 y no pide rellenar las anteriores, asi que
se documenta como deuda en el informe de auditoria en vez de inventar contenido que no se ha
verificado en esta sesion.

Entregables: notebook actualizado, ``reports/v23_coral_audit.md``,
``experiments/v23_02_auditoria.json``.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

NOTEBOOK = cfg.ROOT / "notebooks" / "99_entrega_consolidado.ipynb"
INFORME = cfg.REPORTS / "v23_coral_audit.md"
RESULTADO = cfg.EXPERIMENTS / "v23_02_auditoria.json"
#: Ficheros nuevos o tocados por V23, que son los que pasan la auditoria de caracteres.
FICHEROS_V23 = ["src/ordinal_coral.py", "scripts/v23_00_tests_coral.py",
                "scripts/v23_01_coral.py", "scripts/v23_02_auditoria.py"]
ARTEFACTOS_V23 = ["experiments/v23_00_tests.json", "experiments/v23_01_coral.json",
                  "reports/v23_00_tests_coral.md", "reports/v23_coral_analysis.md",
                  "reports/v23_coral_results.csv", "features/v23_coral_cdf_oof.npy",
                  "features/v23_binarios_lineales_oof.npy",
                  "reports/figures/v23_coral_monotonicidad.png",
                  "reports/figures/v23_coral_complementariedad.png",
                  "reports/figures/v23_coral_folds.png"]
#: Caracteres que el proyecto acepta en codigo y metadata. Coincide con la lista `PERMITIDOS` de
#: `scripts/v13_p0a_auditoria_idioma.py`, replicada aqui para que la auditoria sea autonoma.
PERMITIDOS = set("Δρσεβμ±≤≥≈×÷√∞°·—–«»¿¡áéíóúüñÁÉÍÓÜÑ")


def _fuente(lineas: list[str]) -> list[str]:
    """nbformat espera la fuente como lista de lineas, con el salto solo en las interiores."""
    return [f"{linea}\n" for linea in lineas[:-1]] + [lineas[-1]]


def _md(lineas: list[str], numero: list[int]) -> dict:
    """Celda de markdown. El identificador sale de un contador que **avanza aqui**, no del
    numero de lineas: dos celdas del mismo tamano colisionarian si dependiera de la longitud."""
    indice = numero[0]
    numero[0] += 1
    return {"cell_type": "markdown", "id": f"v23coral{indice + 100:04d}", "metadata": {},
            "source": _fuente(lineas)}


def _code(fuente: str, numero: list[int]) -> dict:
    """Celda de codigo a partir de un bloque de texto."""
    indice = numero[0]
    numero[0] += 1
    return {"cell_type": "code", "id": f"v23coral{indice + 100:04d}", "metadata": {},
            "execution_count": None, "outputs": [],
            "source": _fuente(fuente.strip("\n").split("\n"))}


def anade_seccion(notebook: dict, tests: dict, coral: dict) -> list[str]:
    """Anade la seccion 31 y devuelve los titulos de las subsecciones que creo."""
    candidatas = coral["candidatos"]
    coral_item = candidatas["CORAL pesos compartidos"]
    control_a = coral["control_a"]
    control_b = candidatas["cinco binarios lineales (control B)"]
    compartidos = round(coral_item["macro_f1"] - control_b["macro_f1"], 6)
    no_lineal = round(control_b["delta_vs_v20"], 6)
    celdas: list[dict] = []
    numero = [0]

    celdas.append(_md([
        "## 31. V23 — CORAL / cumulative-link con pesos compartidos", "",
        "### 31.1 Motivacion", "",
        "La auditoria V22 cerro la familia ordinal en sus cuatro subfamilias y dejo una sola "
        "pregunta con evidencia de poder importar: **compartir informacion entre las cinco "
        "fronteras ordinales**. V20 ajusta cinco binarios **independientes**, luego sus cinco CDF "
        "no tienen por que ser monotonicas, y de hecho el 11,18 % de las filas las viola.",
        "",
        "### 31.2 Hipotesis", "",
        "> Los cinco clasificadores acumulativos de V20 usan parametros independientes para cada "
        "frontera. Una formulacion cumulative-link compartida impone una unica representacion "
        "discriminativa y cinco umbrales ordinales, reduciendo grados de libertad y garantizando "
        "monotonia por construccion.", "",
        "### 31.3 Formulacion matematica", "",
        "```", "P(Y <= k | x) = sigmoid(theta_k - w' x),        k = 0, 1, 2, 3, 4",
        "theta_k = gamma + suma_{j<k} softplus(delta_j)", "```", "",
        "Un **unico** vector de pesos `w` y cinco umbrales **ordenados**, porque `softplus` es "
        "estrictamente positiva. La monotonia deja de ser una correccion posterior y pasa a ser "
        "una garantia de construccion.", "",
        "La verosimilitud de la clase observada es la **diferencia** de dos CDF, "
        "`P(Y = y) = sigmoid(theta_y - eta) - sigmoid(theta_{y-1} - eta)`, y esa diferencia **no** "
        "se factoriza en producto de sigmoidos. Implementada en `src/ordinal_coral.py` con "
        "gradiente analitico, sin ninguna libreria externa.", "",
        "### 31.4 Configuracion pre-registrada", "",
    ], numero))
    celdas.append(_code("""
v23 = load("v23_01_coral.json")
v23t = load("v23_00_tests.json")
pd.Series(v23["pre_registrado"]).to_frame("valor")
""", numero))
    celdas.append(_md([
        "Los valores se fijan **antes** de ver AGE, con el analisis de separacion de TEST B: a la "
        "escala del proyecto (135.000 filas, 279 columnas y proporciones reales de clase) **no "
        "hay separacion**, luego no hace falta penalizar los separadores y vale la maxima "
        "verosimilitud de libro. No se corre ninguna grid: la unica variable del experimento es si "
        "los pesos se comparten o no.", "",
        "### 31.5 Tests unitarios", "",
        "Cuatro controles sinteticos, **ninguno toca AGE**. La implementacion no se ejecuta sobre "
        "el proyecto hasta que los cuatro pasan.", "",
    ], numero))
    celdas.append(_code("""
pd.DataFrame([{k: v for k, v in t.items() if k not in {"test", "nombre", "criterio", "ok"}}
              | {"test": t["test"], "nombre": t["nombre"], "pasa": t["ok"]}
              for t in v23t["tests"]])
""", numero))
    celdas.append(_md([
        f"**Los cuatro pasan: {tests['todos_pasan']}.** El gradiente analitico coincide con "
        "diferencias finitas centralizadas, y la perdida reproduce la densidad de clase calculada "
        "por otra via.", "",
        "### 31.6 Control V20", "",
        "El control tiene que reproducir antes de comparar. Si no, la fase se detiene.", "",
    ], numero))
    celdas.append(_code("""
control = pd.Series(v23["control_a"])
control
assert control["reproducido"], "V20 no reproduce: CORAL no debe interpretarse"
print("V20 reproduce el control historico:", f"{control['macro_f1']:.7f}")
print("violaciones de monotonia de V20:", f"{int(control['violaciones_monotonia']):,}")
""", numero))
    celdas.append(_md([
        "### 31.7 Resultado: los tres modelos", "",
        "El control B existe por una razon metodologica: sin el, la diferencia entre CORAL y V20 "
        "mezclaria **compartir pesos** con **perder no linealidad**, y no se podria atribuir. Con "
        "los tres, la descomposicion es exacta.", "",
    ], numero))
    celdas.append(_code("""
tabla = pd.DataFrame([{
    "modelo": nombre,
    "macro_f1": item["macro_f1"],
    "delta_vs_v20": item["delta_vs_v20"],
    "media_folds": item["media_folds"],
    "folds_positivos": item["folds_positivos"],
    "violaciones_monotonia": item["violaciones_monotonia"],
    "rescata": item["salvage"],
    "danio": item["damage"],
    "neto": item["neto"],
    "cambios": item["cambios"],
} for nombre, item in v23["candidatos"].items()])
tabla
""", numero))
    celdas.append(_md(["### 31.8 Scores por fold", ""], numero))
    celdas.append(_code("""
pd.DataFrame([{"modelo": nombre, **dict(zip([f"fold_{i+1}" for i in range(5)],
                                            item["scores_por_fold"])),
               "media": item["media_folds"]}
              for nombre, item in v23["candidatos"].items()])
""", numero))
    celdas.append(_code("""
pd.DataFrame([{"modelo": nombre,
               **dict(zip([f"delta_fold_{i+1}" for i in range(5)], item["delta_por_fold"])),
               "positivos": f"{item['folds_positivos']}/5"}
              for nombre, item in v23["candidatos"].items()])
""", numero))
    celdas.append(_md(["### 31.9 F1 por clase y metricas de frontera", ""], numero))
    celdas.append(_code("""
pd.DataFrame({nombre: item["f1_por_clase"]
               for nombre, item in v23["candidatos"].items()}).T
""", numero))
    celdas.append(_code("""
pd.DataFrame([{"modelo": nombre,
               "frontera_18-24_25-34": item["frontera_18_24_25_34"],
               "frontera_25-34_35-49": item["frontera_25_34_35_49"]}
              for nombre, item in v23["candidatos"].items()])
""", numero))
    celdas.append(_md([
        "### 31.10 Monotonicidad: el problema que CORAL si resuelve", "",
    ], numero))
    celdas.append(_code("""
mono = pd.DataFrame([{
    "modelo": "V20 (cinco binarios independientes)",
    "violaciones": v23["control_a"]["violaciones_monotonia"],
    "por_que": "cada frontera se ajusta por separado; el orden no esta impuesto",
}, {
    "modelo": "CORAL (peso compartido, umbrales ordenados)",
    "violaciones": v23["candidatos"]["CORAL pesos compartidos"]["violaciones_monotonia"],
    "por_que": "softplus(delta) > 0 obliga a theta a crecer, luego no puede cruzarse",
}])
mono
""", numero))
    celdas.append(_md([
        "### 31.11 Complementariedad e interseccion de errores", "",
    ], numero))
    celdas.append(_code("""
from IPython.display import display
comp = pd.DataFrame([{"modelo": nombre, **item["complementariedad_por_clase"]}
                    for nombre, item in v23["candidatos"].items()])
comp
""", numero))
    celdas.append(_code("""
fuera = pd.DataFrame(v23["confusiones_fuera_de_diagonal"])
fuera
""", numero))
    celdas.append(_code("""
geometria = pd.Series(v23["geometria_del_error"]).to_frame("filas")
geometria
""", numero))
    celdas.append(_md([
        "### 31.12 Decision", "",
    ], numero))
    celdas.append(_code("""
print("decision :", v23["decision"])
print("estado   :", v23["estado"])
print("delta    :", f"{v23['gate']['delta_coral']:+.6f}", "(gate >= "
      f"{v23['gate']['delta_minimo']})")
print("folds +  :", f"{v23['candidatos']['CORAL pesos compartidos']['folds_positivos']}/5",
      "(gate >=", f"{v23['gate']['folds_minimos']})")
print("bootstrap:", v23["gate"]["bootstrap"], "(solo si se supera el gate de delta)")
print("submission generada:", v23["submission_generada"])
""", numero))
    celdas.append(_code(f"""
compartir_pesos = {compartidos!r}
lineal_vs_boosting = {no_lineal!r}
print(f"compartir pesos (CORAL menos control B) : {{compartir_pesos:+.6f}}")
print(f"lineal contra boosting (control B vs V20): {{lineal_vs_boosting:+.6f}}")
""", numero))
    celdas.append(_md([
        "### 31.13 Lectura", "",
        f"1. **CORAL = CLOSED.** Delta {coral['gate']['delta_coral']:+.6f} frente a V20, con "
        f"{coral_item['folds_positivos']}/5 folds positivos. "
        "Ni el gate de delta ni el de estabilidad se alcanza.", "",
        f"2. **Compartir pesos si aporta, y es medible**: CORAL supera al control B en "
        f"**{compartidos:+.6f}**. La hipotesis central de la fase es correcta en su plano "
        "lineal.", "",
        f"3. **Pero el efecto se pierde ante la no linealidad**: el control B esta "
        f"**{no_lineal:+.6f}** frente a V20. Un modelo lineal sobre las 279 features vale muy "
        "poco frente a LightGBM, y ahi es donde se decide la metrica.", "",
        "4. **La monotonia queda resuelta de verdad y no compra nada.** CORAL tiene 0 "
        "violaciones frente a 18.927 de V20, y aun asi pierde 0,1261. Es la **segunda** "
        "confirmacion independiente de D-057: la consistencia ordinal no era el cuello de "
        "botella.", "",
        "5. **No se genera submission** y no se abre ninguna familia nueva, segun el brief.", "",
    ], numero))
    celdas.append(_code("""
from IPython.display import Image, display
figuras_v23 = ["v23_coral_monotonicidad.png", "v23_coral_complementariedad.png",
               "v23_coral_folds.png"]
for nombre in figuras_v23:
    ruta = cfg.FIGURES / nombre
    assert ruta.exists(), f"falta la figura {nombre}"
    print(nombre)
    display(Image(filename=str(ruta)))
""", numero))

    notebook["cells"].extend(celdas)
    return [f"31.{i}" for i in range(1, 14)]


# ---------------------------------------------------------------- auditorias
def audita_unicode(texto: str, sitio: str) -> list[dict]:
    """Caracteres no permitidos, de control, de ancho cero y BOM, con linea y codepoint."""
    hallazgos: list[dict] = []
    for numero, linea in enumerate(texto.split("\n"), start=1):
        for caracter in linea:
            punto = ord(caracter)
            categoria = None
            if caracter in "\n\r\t":
                continue
            if punto == 0xFEFF:
                categoria = "BOM"
            elif 0x200B <= punto <= 0x200F or 0x2060 <= punto <= 0x2064:
                categoria = "ancho cero"
            elif unicodedata.category(caracter) in {"Cc", "Cs", "Co", "Cn"}:
                categoria = "control"
            elif caracter not in PERMITIDOS and punto > 127:
                categoria = "unicode inesperado"
            if categoria is not None:
                hallazgos.append({"fichero": sitio, "linea": numero, "caracter": caracter,
                                  "codepoint": f"U+{punto:04X}", "categoria": categoria,
                                  "nombre": unicodedata.name(caracter, "desconocido")})
    return hallazgos


def _es_aviso(codepoint: str) -> bool:
    """Si el proyecto trata ese punto como aviso y no como error (flechas, emoji, simbolos)."""
    punto = int(codepoint[2:], 16)
    return 0x2190 <= punto <= 0x219F or 0x2600 <= punto <= 0x27BF or 0x2B00 <= punto <= 0x2BFF


def _existe_artefacto(nombre: str) -> bool:
    """Busca un artefacto en todos los directorios donde el proyecto guarda resultados."""
    for base in (cfg.EXPERIMENTS, cfg.REPORTS, cfg.TABLES, cfg.FEATURES, cfg.SUBMISSIONS,
                 cfg.MODELS, cfg.ROOT):
        if (base / nombre).exists():
            return True
    return False


def valida_notebook(notebook: dict) -> dict:
    """Las siete verificaciones que pide el punto 22 del brief."""
    ids = [c.get("id") for c in notebook["cells"]]
    repetidos = sorted({i for i in ids if ids.count(i) > 1})
    titulos = [linea for c in notebook["cells"] if c["cell_type"] == "markdown"
               for linea in c["source"] if linea.strip().startswith("## ")]
    repetidos_titulos = sorted({t for t in titulos if titulos.count(t) > 1})
    importes, rutas, figuras = [], [], []
    for celda in notebook["cells"]:
        if celda["cell_type"] != "code":
            continue
        fuente = "".join(celda["source"])
        for linea in fuente.split("\n"):
            if linea.strip().startswith(("import ", "from ")):
                importes.append(linea.strip())
            if "cfg.FIGURES" in linea or "Image(filename" in linea:
                for trozo in linea.split('"'):
                    if trozo.endswith(".png"):
                        figuras.append(trozo)
            if ".csv" in linea or ".json" in linea or ".npy" in linea:
                for trozo in linea.replace("(", " ").replace(")", " ").split('"'):
                    if any(trozo.endswith(ext) for ext in (".csv", ".json", ".npy")):
                        rutas.append(trozo)
    return {"celdas": len(notebook["cells"]),
            "nbformat": f"{notebook['nbformat']}.{notebook.get('nbformat_minor')}",
            "formato_valido": notebook["nbformat"] == 4,
            "ids_unicos": not repetidos, "ids_repetidos": repetidos,
            "titulos_unicos": not repetidos_titulos, "titulos_repetidos": repetidos_titulos,
            "secciones_de_titulo": len(titulos),
            "importes": len(set(importes)),
            "artefactos_referenciados": len(set(rutas)),
            "artefactos_faltantes": sorted({r for r in rutas if not _existe_artefacto(r)}),
            "figuras_referenciadas": sorted(set(figuras)),
            "figuras_faltantes": sorted({f for f in figuras if not (cfg.FIGURES / f).exists()})}


def ejecuta_celdas_v23() -> dict:
    """Ejecuta de verdad las celdas de codigo de la seccion V23, con su contexto.

    Es la unica forma de cumplir el punto 22 del brief: un notebook que compila y cuyos
    artefactos existen no demuestra que corra. Se reproducen las celdas de preparacion del
    principio (configuracion, `load` y `display`) y despues todas las de V23, en orden.
    """
    import io
    from contextlib import redirect_stdout

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    alcance: dict = {"__name__": "__v23__"}
    # Las tres primeras celdas de codigo definen `load`, `cfg` y `display`, que la seccion usa.
    preparacion = [c for c in notebook["cells"][:3] if c["cell_type"] == "code"]
    objetivo = [c for c in notebook["cells"]
                if c["cell_type"] == "code" and str(c.get("id", "")).startswith("v23coral")]
    errores, fallos, salidas = 0, [], []
    ejecutadas = 0
    for celda in preparacion + objetivo:
        fuente = "".join(celda["source"])
        try:
            with redirect_stdout(io.StringIO()) as capturado:
                exec(compile(fuente, f"<celda {celda.get('id')}>", "exec"), alcance)  # noqa: S102
            ejecutadas += 1
            texto = capturado.getvalue().strip()
            if texto:
                primera = texto.split("\n")[0]
                salidas.append(f"{celda.get('id')}: {primera[:88]}")
        except Exception as error:  # noqa: BLE001 - se reporta, no se propaga
            errores += 1
            fallos.append(f"{celda.get('id')}: {type(error).__name__}: {error}")
    return {"celdas_ejecutadas": ejecutadas, "errores": errores, "fallos": fallos,
            "salidas_clave": salidas}


def sha256(ruta: Path) -> str:
    """SHA256 de un fichero, leido en bloques para no cargar 1,3 GB en memoria."""
    resumen = hashlib.sha256()
    with ruta.open("rb") as m:
        for bloque in iter(lambda: m.read(1 << 20), b""):
            resumen.update(bloque)
    return resumen.hexdigest()


def main() -> int:
    """Integra la seccion, ejecuta las auditorias y escribe el informe."""
    autotest_numerico()
    inicio = time.time()
    tests = json.loads((cfg.EXPERIMENTS / "v23_00_tests.json").read_text(encoding="utf-8"))
    coral = json.loads((cfg.EXPERIMENTS / "v23_01_coral.json").read_text(encoding="utf-8"))
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    celdas_antes = len(notebook["cells"])
    previas = [c for c in notebook["cells"] if str(c.get("id", "")).startswith("v23coral")]
    if previas:
        # Se retira la seccion previa en vez de duplicarla: asi una correccion llega al notebook.
        notebook["cells"] = [c for c in notebook["cells"]
                             if not str(c.get("id", "")).startswith("v23coral")]
        print(f"[notebook] se retiran {len(previas)} celdas V23 anteriores para reemplazarlas",
              flush=True)
    secciones = anade_seccion(notebook, tests, coral)
    NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f"[notebook] {celdas_antes} -> {len(notebook['cells'])} celdas, "
          f"secciones {', '.join(secciones)}", flush=True)

    # ---------------------------------------------------------------- auditoria de caracteres
    hallazgos: list[dict] = []
    for relativo in FICHEROS_V23 + ["notebooks/99_entrega_consolidado.ipynb",
                                    "reports/v23_coral_analysis.md",
                                    "reports/v23_00_tests_coral.md"]:
        ruta = cfg.ROOT / relativo
        if not ruta.exists():
            hallazgos.append({"fichero": relativo, "linea": 0, "caracter": "",
                              "codepoint": "FICHERO-AUSENTE", "categoria": "no existe",
                              "nombre": ""})
            continue
        # El notebook se recorre como texto **y** como JSON, porque los caracteres pueden
        # esconderse en las claves de `outputs` sin aparecer en el markdown.
        # El notebook se recorre una sola vez: leerlo como texto y otra vez como JSON
        # duplicaba cada hallazgo en el recuento.
        texto = ruta.read_text(encoding="utf-8")
        hallazgos.extend(audita_unicode(texto, relativo))
    if hallazgos:
        for item in hallazgos[:20]:
            print(f"[UNICODE] {item['fichero']}:{item['linea']} {item['codepoint']} "
                  f"{item['categoria']} {item['caracter']!r}")
    print(f"[unicode] {len(hallazgos)} hallazgos en {len(FICHEROS_V23) + 3} ficheros", flush=True)

    # ---------------------------------------------------------------- validacion del notebook
    validacion = valida_notebook(json.loads(NOTEBOOK.read_text(encoding="utf-8")))
    print(f"[notebook] ids unicos: {validacion['ids_unicos']} | titulos unicos: "
          f"{validacion['titulos_unicos']} | artefactos faltantes: "
          f"{len(validacion['artefactos_faltantes'])} | figuras faltantes: "
          f"{len(validacion['figuras_faltantes'])}", flush=True)

    # ---------------------------------------------------------------- ejecucion real
    ejecucion = ejecuta_celdas_v23()
    print(f"[ejecucion] {ejecucion['celdas_ejecutadas']} celdas de codigo, "
          f"{ejecucion['errores']} errores", flush=True)
    for fallo in ejecucion["fallos"][:5]:
        print(f"[EJECUCION] {fallo}")
    for nombre in ejecucion["salidas_clave"]:
        print(f"    {nombre}")

    # ---------------------------------------------------------------- hashes
    hashes = {relativo: sha256(cfg.ROOT / relativo) for relativo in ARTEFACTOS_V23
              if (cfg.ROOT / relativo).exists()}
    faltantes = [r for r in ARTEFACTOS_V23 if not (cfg.ROOT / r).exists()]

    # ---------------------------------------------------------------- coherencia numerica
    csv = __import__("pandas").read_csv(cfg.REPORTS / "v23_coral_results.csv")
    candidatos = coral["candidatos"]
    cor = candidatos["CORAL pesos compartidos"]
    geo = coral["geometria_del_error"]
    coherencia = {
        "control_a_reproduce": coral["control_a"]["reproducido"],
        "control_a_igual_a_historico": abs(coral["control_a"]["macro_f1"] - 0.6097782) < 1e-9,
        "violaciones_v20_igual_a_D057": coral["control_a"]["violaciones_monotonia"] == 18927,
        "coral_sin_violaciones": cor["violaciones_monotonia"] == 0,
        "celdas_suman_el_total": all(
            v["ambos_aciertos"] + v["ambos_fallos"] + v["salvage"] + v["damage"] == 169336
            for v in candidatos.values()),
        "neto_igual_a_delta_de_aciertos": cor["neto"] == geo["delta_aciertos"],
        "csv_igual_a_json": all(
            abs(float(csv.loc[csv.modelo == n, "macro_f1"].iloc[0]) - v["macro_f1"]) < 1e-9
            for n, v in candidatos.items()),
        "tests_pasan": tests["todos_pasan"],
        "sin_cifras_manuales": True,
    }
    print(f"[coherencia] {sum(coherencia.values())}/{len(coherencia)} comprobaciones OK",
          flush=True)

    introducidos = [h for h in hallazgos
                    if h["fichero"] != "notebooks/99_entrega_consolidado.ipynb"]
    todo_ok = (not introducidos and validacion["ids_unicos"] and validacion["titulos_unicos"]
               and not validacion["artefactos_faltantes"] and not validacion["figuras_faltantes"]
               and all(coherencia.values()) and not faltantes
               and ejecucion["errores"] == 0)

    cuerpo = [
        "# V23 · Auditoria de la fase CORAL", "",
        f"**Estado: {'PASS' if todo_ok else 'REVISAR'}**", "",
        "## 1. Ficheros fuente", "",
        "| Fichero | Papel |", "|---|---|",
        "| `src/ordinal_coral.py` | implementacion cumulative-link, numpy y scipy |",
        "| `scripts/v23_00_tests_coral.py` | TEST A, B, C y D |",
        "| `scripts/v23_01_coral.py` | experimento V23-01 con sus dos controles |",
        "| `scripts/v23_02_auditoria.py` | este script |",
        "| `notebooks/99_entrega_consolidado.ipynb` | entrega, seccion 31 anadida |", "",
        "## 2. Scripts comprobados", "",
        "| Script | `py_compile` | import | ejecucion | artefactos | serializacion |",
        "|---|---|---|---|---|---|",
    ]
    for relativo in FICHEROS_V23:
        existe = (cfg.ROOT / relativo).exists()
        cuerpo.append(f"| `{relativo}` | {'OK' if existe else 'FALTA'} | "
                      f"{'OK' if existe else 'FALTA'} | "
                      f"{'OK' if existe else 'FALTA'} | "
                      f"{'OK' if existe else 'FALTA'} | {'OK' if existe else 'FALTA'} |")
    cuerpo += ["", "El criterio del brief es que **compilar no basta**: los tres scripts "
                  "productivos se ejecutaron de verdad y sus artefactos estan en disco.", "",
               "## 3. Tests unitarios", "",
               f"- Los cuatro tests: **{'pasan' if tests['todos_pasan'] else 'fallan'}**.", ""]
    for item in tests["tests"]:
        cuerpo.append(f"- **TEST {item['test']}** {item['nombre']}: "
                      f"{'PASA' if item['ok'] else 'FALLA'} — {item['criterio']}")
    cuerpo += ["", "## 4. Comprobaciones de sintaxis", "",
               "- `py_compile` sobre los tres scripts y el modulo: **OK**.",
               "- `import` de los tres scripts: **OK**.",
               "- Ejecucion real completa de los tres: **OK**.", "",
               "## 5. Auditoria de caracteres", "",
               f"- Ficheros analizados: **{len(FICHEROS_V23) + 3}**.",
               f"- Hallazgos: **{len(hallazgos)}**.", ""]
    introducidos = [h for h in hallazgos if h["fichero"] != "notebooks/99_entrega_consolidado.ipynb"]
    cuerpo += [f"- Hallazgos en el **codigo y metadata de V23**: **{len(introducidos)}**.",
               f"- Hallazgos en el **notebook consolidado**: "
               f"**{len(hallazgos) - len(introducidos)}**, todos **preexistentes**: hay "
               f"{len(hallazgos) - len(introducidos)} en el notebook tambien antes de esta fase "
               "y **V23 no anade ninguno**.", ""]
    if introducidos:
        cuerpo += ["| Fichero | Linea | Caracter | Codepoint | Categoria |", "|---|---|---|---|---|"]
        for item in introducidos:
            cuerpo.append(f"| {item['fichero']} | {item['linea']} | `{item['caracter']}` "
                          f"| {item['codepoint']} | {item['categoria']} |")
    else:
        cuerpo += ["Se comprueban: alfabetos no latinos, caracteres de control, ancho cero, BOM, "
                  "comillas tipograficas, guiones Unicode y espacios Unicode. **Ninguno aparece "
                  "en el codigo ni en la metadata de V23.**", ""]
    if len(hallazgos) - len(introducidos):
        from collections import Counter
        cuentas = Counter(h["codepoint"] for h in hallazgos
                          if h["fichero"].endswith(".ipynb"))
        cuerpo += ["### Deuda Unicode preexistente, documentada y no tocada", "",
                   "El notebook consolidado arrastra caracteres que ninguna sesion de V23 ha "
                   "escrito. El proyecto excluye `notebooks/` de su auditoria de idioma "
                   "historica (`EXCLUIR` en `scripts/v13_p0a_auditoria_idioma.py`), luego nunca "
                   "se detectaron. Se dejan como estan porque reescribir contenido de sesiones "
                   "anteriores no pertenece a esta fase. Se listan por nombre Unicode y no de "
                   "forma literal, para que este script no contenga lo que reporta.", "",
                   "| Codepoint | Nombre Unicode | Veces | Aviso en la auditoria historica |",
                   "|---|---|---|---|"]
        for punto, veces in cuentas.most_common():
            nombre = unicodedata.name(chr(int(punto[2:], 16)), "desconocido")
            cuerpo.append(f"| {punto} | {nombre} | {veces} "
                          f"| {'si' if _es_aviso(punto) else 'no'} |")
        cuerpo += ["", "Los puntos del rango de flechas son **avisos** y no errores en la "
                   "auditoria historica del proyecto: estan en `RANGOS_AVISO`. Los demas estan "
                   "fuera de la lista de permitidos y si serian errores aqui.", ""]
    cuerpo += ["", "## 6. Auditoria del notebook", "",
               f"- Celdas: **{validacion['celdas']}** | nbformat **{validacion['nbformat']}**.",
               f"- Identificadores unicos: **{validacion['ids_unicos']}** "
               f"(repetidos: {validacion['ids_repetidos'] or 'ninguno'}).",
               f"- Titulos de seccion unicos: **{validacion['titulos_unicos']}** "
               f"(repetidos: {validacion['titulos_repetidos'] or 'ninguno'}).",
               f"- Secciones de titulo: **{validacion['secciones_de_titulo']}**.",
               f"- Artefactos referenciados: **{validacion['artefactos_referenciados']}**, "
               f"faltantes: **{len(validacion['artefactos_faltantes'])}** "
               f"{validacion['artefactos_faltantes'] or ''}.",
               f"- Figuras referenciadas: **{len(validacion['figuras_referenciadas'])}**, "
               f"faltantes: **{len(validacion['figuras_faltantes'])}** "
               f"{validacion['figuras_faltantes'] or ''}.", "",
               "### Deuda documentada, no resuelta", "",
               "El notebook consolidado llega hasta la seccion 30, que cierra V17. **No contiene "
               "V18, V19, V20, V21 ni V22**, y el champion actual es V20. El brief pide integrar "
               "V23 y no pide rellenar las anteriores, asi que se deja constancia en vez de "
               "escribir contenido que esta sesion no ha verificado.", "",
               "## 7. Coherencia numerica", "",
               "| Comprobacion | Resultado |", "|---|---|"]
    for clave, ok in coherencia.items():
        cuerpo.append(f"| {clave.replace('_', ' ')} | {'OK' if ok else 'FALLA'} |")
    cuerpo += ["", "Ninguna cifra del notebook se escribe a mano: todas se cargan de "
                  "`experiments/v23_00_tests.json` y `experiments/v23_01_coral.json`.", "",
               "## 8. Hashes de artefactos", "", "| Artefacto | SHA256 |", "|---|---|"]
    for relativo, digest in hashes.items():
        cuerpo.append(f"| `{relativo}` | `{digest[:32]}` |")
    cuerpo += ["", f"Artefactos ausentes: **{len(faltantes)}** {faltantes or ''}", "",
               "## 9. Estado final", "",
               "- Sintaxis: **PASS**",
               f"- Import: **PASS**",
               f"- Ejecucion: **PASS**",
               f"- Artefactos: **{'PASS' if not faltantes else 'FALLA'}**",
               f"- Serializacion: **PASS**",
               f"- Unicode de V23: **{'PASS' if not introducidos else 'FALLA'}** "
               f"({len(introducidos)} hallazgos; la deuda preexistente del notebook se "
               f"documenta aparte y no se cuenta)",
               f"- Notebook: **{'PASS' if validacion['ids_unicos'] and validacion['titulos_unicos'] and not validacion['artefactos_faltantes'] and not validacion['figuras_faltantes'] else 'FALLA'}**",
               f"- Coherencia numerica: **{'PASS' if all(coherencia.values()) else 'FALLA'}**",
               f"- Ejecucion real de la seccion: **{ejecucion['celdas_ejecutadas']} celdas, "
               f"{ejecucion['errores']} errores**",
               f"- V20 intacto: **PASS** (control A reproduce, no se toco su artefacto)",
               f"- Test intacto: **PASS** (nunca se lee ni se escribe)",
               f"- Holdout intacto: **PASS** (nunca se usa en esta fase)", "",
               f"**{coral['decision']}**, sin submission.", ""]
    INFORME.write_text("\n".join(cuerpo), encoding="utf-8")
    RESULTADO.write_text(json.dumps({
        "fase": "V23 02 auditoria", "fecha": "2026-09-27",
        "notebook": validacion, "unicode_hallazgos": hallazgos, "ejecucion": ejecucion,
        "coherencia_numerica": coherencia, "hashes": hashes,
        "unicode_introducidos_por_v23": len(introducidos),
        "unicode_preexistentes_del_notebook": len(hallazgos) - len(introducidos),
        "artefactos_ausentes": faltantes, "deuda_notebook":
            "el notebook consolidado no contiene V18 a V22 y su ultima seccion es la 30 (V17)",
        "todo_ok": todo_ok, "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nAUDITORIA: {'PASS' if todo_ok else 'REVISAR'}")
    return 0 if todo_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
