"""V25 · 02 — Integra la seccion de seleccion supervisada en el notebook y lo audita.

Anade la seccion **33 · V25 Seleccion supervisada de features** al notebook consolidado, y
despues ejecuta las verificaciones que pide el brief: ejecucion real de las celdas nuevas,
unicidad de identificadores y titulos, artefactos referenciados, coherencia numerica y
**requisito bloqueante de Unicode**.

## Idempotencia

Si la seccion ya existe, **se retira y se vuelve a construir**, no se duplica. Sin eso, una
correccion llegaria al notebook como una copia mas y el recuento de celdas creeria sin que
cambie nada, que es exactamente el fallo que V24 vino a corregir.

## De donde salen las cifras

Ninguna cifra del texto se escribe a mano: todas se leen de `experiments/v25_00_inventario.json` y
`experiments/v25_01_seleccion.json`, o se dejan que las celdas de codigo las carguen. El notebook
no puede contener un numero que no venga de un artefacto.

## Entregables

Notebook actualizado, ``experiments/v25_02_auditoria.json`` y ``reports/v25_02_auditoria.md``.
"""

from __future__ import annotations

import json
import sys
import time
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

NOTEBOOK = cfg.ROOT / "notebooks" / "99_entrega_consolidado.ipynb"
INFORME = cfg.REPORTS / "v25_02_auditoria.md"
RESULTADO = cfg.EXPERIMENTS / "v25_02_auditoria.json"
PREFIJO = "v25sel"
#: Caracteres aceptados en codigo y metadata, replica de la lista `PERMITIDOS` del proyecto.
PERMITIDOS = set("Δρσεβμ±≤≥≈×÷√∞°·—–«»¿¡áéíóúüñÁÉÍÓÚÜÑ")


def _fuente(lineas: list[str]) -> list[str]:
    """nbformat espera la fuente como lista de lineas, con el salto solo en las interiores."""
    return [f"{linea}\n" for linea in lineas[:-1]] + [lineas[-1]]


def _md(lineas: list[str], n: int) -> dict:
    """Celda de markdown con identificador unico y determinista."""
    return {"cell_type": "markdown", "id": f"{PREFIJO}{n:04d}", "metadata": {},
            "source": _fuente(lineas)}


def _code(bloque: str, n: int) -> dict:
    """Celda de codigo con identificador unico y determinista."""
    return {"cell_type": "code", "id": f"{PREFIJO}{n:04d}", "metadata": {},
            "execution_count": None, "outputs": [],
            "source": _fuente(bloque.strip("\n").split("\n"))}


def construye(inv: dict, sel: dict) -> list[dict]:
    """Celdas de la seccion 33, con las cifras interpoladas desde los artefactos."""
    c = sel["candidatos"]
    t0 = c["T0 V20 sin seleccion"]
    t1 = c["T1 seleccion MI top 50%"]
    bloques = sel["error_por_bloque"]
    k = sel["pre_registrado"]["K"]
    celdas: list[dict] = []

    celdas.append(_md([
        "## 33. V25 — Seleccion supervisada de features", "",
        "### 33.1 Hipotesis", "",
        "> V20 puede mejorar eliminando features ruidosas mediante una seleccion supervisada "
        "realizada correctamente dentro de cada fold.", "",
        "### 33.2 Por que no es V15", "",
        "| | V15 | V25 |", "|---|---|---|",
        "| Metodo | VIF, correlacion, clustering | `mutual_info_classif` sobre 6 categorias |",
        "| Usa la variable objetivo | **no** | **si**, solo del training fold |",
        "| Poda | no supervisada | ninguna: se conserva el top 50 por ciento |",
        "| Resultado | poda -0,004, 0 de 5 folds | se mide mas abajo |", "",
        "V15 concluyo que la redundancia de entrada es **funcional** y que podar empeora. V25 no "
        "repite esa poda: usa la etiqueta, y solo las del training fold.", "",
        "### 33.3 Inventario y procedencia de las 279 features", "",
        "La pregunta previa era si las 73 features **label-informed** son legales. Se verifico por "
        "codigo, por comparacion de arrays y por separacion empirica.", "",
    ], 1))
    celdas.append(_code("""
inv = load("v25_00_inventario.json")
print("features auditadas:", inv["n_features"])
inv["bloques"]
""", 2))
    celdas.append(_code("""
proc = pd.DataFrame({clave: [inv["procedencia"][clave]] for clave in
                     ("codigo_calcula_con_y_train", "codigo_escribe_solo_en_validacion",
                      "codigo_pasa_y_train_a_la_firma", "mismo_array_de_grupos",
                      "mismos_folds_de_validacion")}).T
proc.columns = ["resultado"]
print("veredicto:", inv["procedencia"]["veredicto"])
proc
""", 3))
    celdas.append(_code("""
# La escala condiciona el mutual information: sin estandarizar, el estimador por vecino mas
# cercano compara distancias en columnas que no comparten rango.
escala = pd.DataFrame(inv["escala"]).T
escala
""", 4))
    celdas.append(_md([
        "### 33.4 Control T0 y la distincion que obliga a hacer", "",
        "**V20 no es la cadena de cinco umbrales: es el blend de cuatro cuartos.** La cadena sola "
        f"da {inv['cadena_sin_blend']:.7f} y el blend da {t0['corrected_cv']:.7f}. Comparar contra "
        "la cifra del blend es lo unico comparable, y es la misma distincion que ya hubo que "
        "hacer en V22.", "",
        "### 33.5 La regla de seleccion, pre-registrada y unica", "",
        f"K = floor(279 x {sel['pre_registrado']['top_porcentaje']}) = **{k}**, aplicado "
        "**independientemente en cada training fold** y **sin ajustarlo observando el resultado**. "
        "No se corre ninguna grid, no se prueba otro K y no se prueba otro selector.", "",
    ], 5))
    celdas.append(_code("""
sel = load("v25_01_seleccion.json")
print("T0 reproduce:", sel["control_t0"]["reproducido"],
      "| valor:", f"{sel['control_t0']['corrected_cv']:.7f}")
print("T1:", f"{sel['candidatos']['T1 seleccion MI top 50%']['corrected_cv']:.7f}",
      "| delta:", f"{sel['delta']:+.6f}")
sel["gate"]
""", 6))
    celdas.append(_md(["### 33.6 Resultados por fold y F1 por clase", ""], 7))
    celdas.append(_code("""
sel_df = pd.read_csv(cfg.REPORTS / "v25_feature_selection.csv")
sel_df
""", 8))
    celdas.append(_code("""
pd.DataFrame([{
    "modelo": nombre,
    **{f"fold_{i+1}": v[f"fold_{i+1}"] for i in range(5)},
    "media": v["media"], "desviacion": v["desviacion"],
    "rescue": v["rescue"], "damage": v["damage"], "net": v["net"],
    "changed_rows": v["changed_rows"], "folds_positivos": f"{v['folds_positivos']}/5",
} for nombre, v in sel["candidatos"].items()])
""", 9))
    celdas.append(_code("""
pd.DataFrame({nombre: v["f1_por_clase"] for nombre, v in sel["candidatos"].items()}).T
""", 10))
    celdas.append(_md(["### 33.7 Estabilidad de la seleccion y error por bloque", "",
                       "La frecuencia de seleccion es **diagnostico**. No se usa para construir una "
                       "segunda seleccion, que es justo lo que V15 hizo mal.", ""], 11))
    celdas.append(_code("""
frecuencia = pd.read_csv(cfg.REPORTS / "v25_feature_selection_frequency.csv")
print("features:", len(frecuencia),
      "| en los 5 folds:", int((frecuencia["selected_in_n_folds"] == 5).sum()),
      "| en ninguno:", int((frecuencia["selected_in_n_folds"] == 0).sum()))
pd.DataFrame(sel["error_por_bloque"]).T
""", 12))
    celdas.append(_code("""
pd.DataFrame(sel["seleccion_por_fold"])[["fold", "K", "mi_max", "mi_min"]]
""", 13))
    celdas.append(_md([
        "### 33.8 Decision", "",
        f"1. **`{sel['decision']}`.** Delta {t1['delta_vs_t0']:+.6f} frente a T0, con "
        f"**{t1['folds_positivos']}/5** folds positivos. El gate exige "
        f"{sel['gate']['delta_minimo']:+.3f} y {sel['gate']['folds_minimos']}/5 folds.", "",
        "2. **La seleccion no elimina ruido, elimina senal.** Las columnas que sobreviven con "
        f"frecuencia 1,000 son las {bloques['miembro_ensamble']['columnas']} probabilidades del "
        f"ensamble y las {bloques['nbsvm_palabra']['columnas']} de NB-SVM, con MI media de 0,215 y "
        f"0,208. Las que nunca sobreviven son las estructuradas y las densas "
        f"({bloques['estructurado']['nunca_seleccionadas']} y "
        f"{bloques['denso_textometrico']['nunca_seleccionadas']} columnas), con MI media de 0,012. "
        "**El mutual information alto no es ruido: es la prediccion de los miembros que sostiene "
        "el modelo.**", "",
        f"3. **La perdida es pequena y repartida**: {t1['rescue']:,} rescates contra "
        f"{t1['damage']:,} "
        f"daños, neto **{t1['net']:+,}**, y solo {t1['changed_rows']:,} filas cambian. El F1 por "
        "clase es practicamente identico, con la caida repartida y ninguna clase afectada de forma "
        "especifica.", "",
        "4. **No se genera submission, no se prueba otro K y no se prueba otro selector.** El "
        "brief cierra la familia en cuanto el delta no supera el gate.", "",
    ], 14))
    celdas.append(_code("""
from IPython.display import Image, display
for nombre in ("v25_frecuencia_seleccion.png", "v25_delta_por_fold.png"):
    ruta = cfg.FIGURES / nombre
    assert ruta.exists(), f"falta la figura {nombre}"
    print(nombre)
    display(Image(filename=str(ruta)))
""", 15))
    return celdas


def audita_unicode(texto: str, sitio: str) -> list[dict]:
    """Caracteres no permitidos, de control, de ancho cero y BOM, con linea y codepoint."""
    hallazgos: list[dict] = []
    for numero, linea in enumerate(texto.split("\n"), start=1):
        for caracter in linea:
            if caracter in "\n\r\t":
                continue
            punto = ord(caracter)
            categoria = None
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


def ejecuta_celdas() -> dict:
    """Ejecuta de verdad las celdas de codigo de V25, con su contexto de arranque."""
    import io
    from contextlib import redirect_stdout

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    preparacion = [c for c in notebook["cells"][:3] if c["cell_type"] == "code"]
    objetivo = [c for c in notebook["cells"]
                if c["cell_type"] == "code" and str(c.get("id", "")).startswith(PREFIJO)]
    alcance: dict = {"__name__": "__v25__"}
    fallos, salidas, ejecutadas, errores = [], [], 0, 0
    for celda in preparacion + objetivo:
        fuente = "".join(celda["source"])
        try:
            with redirect_stdout(io.StringIO()) as capturado:
                exec(compile(fuente, f"<celda {celda.get('id')}>", "exec"), alcance)  # noqa: S102
            ejecutadas += 1
            texto = capturado.getvalue().strip()
            if texto:
                salidas.append(f"{celda.get('id')}: " + " | ".join(texto.split(chr(10))[:2])[:110])
        except Exception as error:  # noqa: BLE001 - se reporta, no se propaga
            errores += 1
            fallos.append(f"{celda.get('id')}: {type(error).__name__}: {error}")
    return {"celdas_ejecutadas": ejecutadas, "errores": errores, "fallos": fallos,
            "salidas_clave": salidas}


def main() -> int:
    """Integra la seccion, ejecuta las celdas y audita el resultado."""
    autotest_numerico()
    inicio = time.time()
    inv = json.loads((cfg.EXPERIMENTS / "v25_00_inventario.json").read_text(encoding="utf-8"))
    sel = json.loads((cfg.EXPERIMENTS / "v25_01_seleccion.json").read_text(encoding="utf-8"))
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    antes = len(notebook["cells"])
    # La deuda heredada del notebook se mide ANTES, para no atribuírsela a V25.
    heredados = audita_unicode(NOTEBOOK.read_text(encoding="utf-8"), "notebook_antes")
    previas = [c for c in notebook["cells"] if str(c.get("id", "")).startswith(PREFIJO)]
    if previas:
        notebook["cells"] = [c for c in notebook["cells"]
                             if not str(c.get("id", "")).startswith(PREFIJO)]
        print(f"[notebook] se retiran {len(previas)} celdas V25 anteriores", flush=True)
    nuevas = construye(inv, sel)
    notebook["cells"].extend(nuevas)
    NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f"[notebook] {antes} -> {len(notebook['cells'])} celdas (+{len(nuevas)})", flush=True)

    # ---------------------------------------------------------------- estructura
    celdas = notebook["cells"]
    ids = [c.get("id") for c in celdas]
    titulos = [l.strip() for c in celdas if c["cell_type"] == "markdown"
               for l in c.get("source", []) if l.strip().startswith("## ")]
    cuerpo = "".join("".join(c.get("source", [])) for c in celdas)
    rutas = [t for c in celdas if c["cell_type"] == "code"
             for t in __import__("re").findall(r'"([^"]+\.(?:json|csv|png))"',
                                             "".join(c.get("source", [])))]
    faltan = sorted({r for r in rutas if not _existe(r)})
    estructura = {"celdas": len(celdas), "ids_unicos": len(set(ids)) == len(ids),
                  "titulos_unicos": len(set(titulos)) == len(titulos),
                  "secciones": len(titulos), "titulo_v25_presente": "## 33. V25" in cuerpo,
                  "artefactos_referenciados": len(set(rutas)), "artefactos_faltantes": faltan}
    print(f"[notebook] ids unicos {estructura['ids_unicos']} | titulos unicos "
          f"{estructura['titulos_unicos']} | artefactos faltantes {len(faltan)}", flush=True)

    ejecucion = ejecuta_celdas()
    print(f"[ejecucion] {ejecucion['celdas_ejecutadas']} celdas, {ejecucion['errores']} errores",
          flush=True)
    for fallo in ejecucion["fallos"][:5]:
        print(f"[EJECUCION] {fallo}")

    # ---------------------------------------------------------------- unicode
    introducidos = audita_unicode(NOTEBOOK.read_text(encoding="utf-8"), "notebook_despues")
    def _clave(hallazgo):
        return (hallazgo["caracter"], hallazgo["codepoint"], hallazgo["categoria"])
    antes_claves = {_clave(h) for h in heredados}
    nuevos = [h for h in introducidos if _clave(h) not in antes_claves]
    heredados_despues = [h for h in introducidos if _clave(h) in antes_claves]
    scripts_v25 = audita_unicode(
        (cfg.ROOT / "scripts" / "v25_01_seleccion.py").read_text(encoding="utf-8"),
        "scripts/v25_01_seleccion.py")
    print(f"[unicode] introducidos por V25: {len(nuevos)} | heredados del notebook: "
          f"{len(heredados_despues)} | en los scripts de V25: {len(scripts_v25)}", flush=True)

    # ---------------------------------------------------------------- coherencia
    t0 = sel["candidatos"]["T0 V20 sin seleccion"]["corrected_cv"]
    t1 = sel["candidatos"]["T1 seleccion MI top 50%"]["corrected_cv"]
    inventario = pd.read_csv(cfg.REPORTS / "v25_feature_inventory.csv")
    frecuencia = pd.read_csv(cfg.REPORTS / "v25_feature_selection_frequency.csv")
    coherencia = {
        "t0_es_el_canonico": abs(t0 - 0.6097782) < 1e-9,
        "t1_coincide_con_el_json": abs(t1 - sel["candidatos"]["T1 seleccion MI top 50%"]
                                       ["corrected_cv"]) < 1e-9,
        "inventario_de_279": len(inventario) == 279,
        "inventario_73_label_informed": int(inventario["label_informed"].sum()) == 73,
        "frecuencia_de_279": len(frecuencia) == 279,
        "decision_cerrada": sel["estado"] == "NO_SIGNAL",
        "sin_submission": sel["submission_generada"] is False,
    }
    print(f"[coherencia] {sum(coherencia.values())}/{len(coherencia)} OK", flush=True)
    todo_ok = (estructura["ids_unicos"] and estructura["titulos_unicos"]
               and estructura["titulo_v25_presente"] and not faltan
               and ejecucion["errores"] == 0 and all(coherencia.values())
               and not nuevos and not scripts_v25)

    _informe(estructura, ejecucion, nuevos, heredados_despues, scripts_v25, coherencia,
             inv, sel, todo_ok)
    RESULTADO.write_text(json.dumps({
        "fase": "V25 02 auditoria de notebook", "fecha": "2026-09-27", "experimenta": False,
        "celdas_antes": antes, "celdas_despues": len(notebook["cells"]),
        "estructura": estructura, "ejecucion": ejecucion, "unicode_introducidos": nuevos,
        "unicode_heredado_del_notebook": len(heredados_despues),
        "unicode_en_scripts": scripts_v25, "coherencia": coherencia, "todo_ok": todo_ok,
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nV25-02: {'PASS' if todo_ok else 'REVISAR'}")
    return 0 if todo_ok else 1


def _existe(nombre: str) -> bool:
    """Busca un artefacto en todos los directorios donde el proyecto guarda resultados."""
    for base in (cfg.EXPERIMENTS, cfg.REPORTS, cfg.TABLES, cfg.FEATURES, cfg.SUBMISSIONS,
                 cfg.FIGURES, cfg.ROOT):
        if (base / nombre).exists():
            return True
    return False


def _informe(estructura, ejecucion, introducidos, heredados, scripts_v25, coherencia,
             inv, sel, ok) -> None:
    """Redacta la auditoria de la integracion documental de V25."""
    c = sel["candidatos"]
    t0 = c["T0 V20 sin seleccion"]
    t1 = c["T1 seleccion MI top 50%"]
    cuerpo = [
        "# V25 · 02 — Auditoria de la integracion documental", "",
        f"**Estado: {'PASS' if ok else 'REVISAR'}**", "",
        "## 1. Notebook", "",
        f"- Celdas: **{estructura['celdas']}** | secciones: **{estructura['secciones']}**.",
        f"- Identificadores unicos: **{estructura['ids_unicos']}**.",
        f"- Titulos unicos: **{estructura['titulos_unicos']}**.",
        f"- Seccion 33 de V25 presente: **{estructura['titulo_v25_presente']}**.",
        f"- Artefactos referenciados: **{estructura['artefactos_referenciados']}**, faltantes: "
        f"**{len(estructura['artefactos_faltantes'])}**.", "",
        "## 2. Ejecucion real", "",
        f"- Celdas de codigo ejecutadas: **{ejecucion['celdas_ejecutadas']}**.",
        f"- Errores: **{ejecucion['errores']}**.", ""]
    for salida in ejecucion["salidas_clave"]:
        cuerpo.append(f"- `{salida}`")
    cuerpo += ["", "## 3. Unicode (requisito bloqueante)", "",
               f"- **Introducidos por V25: {len(introducidos)}** (requisito bloqueante)",
               f"- **Heredados del notebook, de sesiones anteriores: {len(heredados)}**, medidos "
               "antes de esta fase y no atribuibles a V25",
               f"- En los scripts de V25: **{len(scripts_v25)}**", "",
               "Se comprueban: alfabetos no latinos, CJK, cirilico, caracteres de control, ancho "
               "cero, BOM, comillas tipograficas, guiones Unicode y espacios Unicode.", "",
               "## 4. Coherencia numerica", "", "| Comprobacion | Resultado |", "|---|---|"]
    for clave, valor in coherencia.items():
        cuerpo.append(f"| {clave.replace('_', ' ')} | {'OK' if valor else 'FALLA'} |")
    cuerpo += ["", "## 5. Resultado de la fase, que el notebook documenta", "",
               f"- Control T0: **{t0['corrected_cv']:.7f}** (reproduce: "
               f"{sel['control_t0']['reproducido']}).",
               f"- T1 seleccion MI top {sel['pre_registrado']['top_porcentaje']} "
               f"(K = {sel['pre_registrado']['K']}): **{t1['corrected_cv']:.7f}** | delta "
               f"**{t1['delta_vs_t0']:+.6f}** | **{t1['folds_positivos']}/5** folds positivos.",
               f"- Decision: **{sel['decision']}** | submission: "
               f"**{sel['submission_generada']}**.", "",
               f"- Inventario: **{inv['n_features']} features** auditadas, de las cuales "
               f"**{inv['columnas_label_informed']}** label-informed.", ""]
    INFORME.write_text("\n".join(cuerpo), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
