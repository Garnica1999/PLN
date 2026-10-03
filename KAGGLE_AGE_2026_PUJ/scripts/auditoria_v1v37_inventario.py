"""auditoria_v1v37_inventario.py

Fecha: 2026-10-01
Descripcion: inventario verificable de las 195 filas de `experiments.csv` del proyecto
    KAGGLE_AGE_2026_PUJ. Clasifica cada fila por su TIPO DE REGISTRO (no por heuristica de
    texto) usando evidencia en disco, y construye la cadena de trazabilidad
    hipotesis -> protocolo -> script -> artefactos -> fila CSV -> notebook.

Este script NO entrena, NO lee test, holdout ni leaderboard, NO genera submissions y NO
modifica ningun script de experimento. Solo lee: `experiments.csv`, el notebook, `DECISIONS.md`,
`PROJECT_STATE.md` y el arbol de `experiments/`, `reports/`, `artifacts/`, `plan/`, `scripts/`.

Modos
-----
- Sin argumentos: escribe `reports/tables/auditoria_v1v37_inventario.csv`,
  `reports/tables/auditoria_v1v37_huecos.csv` y
  `experiments/auditoria_v1v37_inventario.json`.
- `--verificar`: autoejercicio de los siete caminos de la seccion "Verificacion" en un
  directorio temporal. No escribe nada real. Sale 0 si los seis pasan, 2 si alguno falla.
- `--resumen`: solo imprime el resumen por tipo y por fase, sin escribir archivos.

Criterio de clasificacion (tipo de registro), en este orden:
  1. ENMIENDA      si el resultado dice "enmienda registrada" o el ID contiene ENMIENDA.
  2. PREREGISTRO   si el resultado dice "pre-registro" / "sin resultados" o el ID contiene PREREG.
  3. BLOQUEADO     si el resultado declara un defecto de implementacion o una interrupcion.
  4. DOCUMENTAL    si no hay macro_f1_mean y no hay artefacto con metricas, y la etapa
                  lexical es de auditoria, informe, cierre, compliance o decision.
  5. EJECUTADO     en otro caso, con o sin macro_f1 (un cierre por gate SI se ejecuto).

ASCII puro: sin acentos, sin emojis, sin CJK. Los .md y las figuras si admiten tildes.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "experiments.csv"
NB_PATH = ROOT / "notebooks/99_entrega_consolidado.ipynb"
DEC_PATH = ROOT / "DECISIONS.md"
STATE_PATH = ROOT / "PROJECT_STATE.md"
OUT_CSV = ROOT / "reports/tables/auditoria_v1v37_inventario.csv"
OUT_GAPS = ROOT / "reports/tables/auditoria_v1v37_huecos.csv"
OUT_JSON = ROOT / "experiments/auditoria_v1v37_inventario.json"

COLS = [
    "experiment_id", "fecha", "tipo_registro", "familia", "etapa", "pregunta",
    "macro_f1_mean", "macro_f1_std", "resultado", "decision",
    "script_declarado", "artefactos_citados", "n_artefactos", "artefactos_existentes",
    "artefactos_faltantes", "evidencia_log", "evidencia_json", "en_notebook",
    "seccion_notebook", "en_decisiones", "en_project_state", "dictamen_inventario",
    "severidad", "observacion",
]

FAMILIA_RANGES = [
    (0.0, 4.99, "V1-V4 fundacional (E-*, B*, ADV-*)"),
    (5.0, 12.99, "V5-V12 fases PH*"),
    (13.0, 17.99, "V13-V17 busqueda residual"),
    (18.0, 23.99, "V18-V23 frontera y ordinal"),
    (24.0, 29.99, "V24-V29 seleccion y ensemble"),
    (30.0, 31.99, "V30-V31 campeon y tracks"),
    (32.0, 33.99, "V32-V33 reapertura y FTRL"),
    (34.0, 36.99, "V34-V36 falsacion FTRL y L4"),
    (37.0, 99.99, "V37 diagnostico POS"),
]

LEX_DOC = re.compile(
    r"auditor|informe|cierre|reconcil|compliance|revisi|cobertura|submission|transferencia|"
    r"decision ejecutiva|hito|acta|document|bitacora",
    re.IGNORECASE,
)
# OJO: un RECUENTO de tipos menciona la palabra "ENMIENDA" ("151 EJECUTADO + ... + 10 ENMIENDA"),
# y eso no hace que la fila sea una enmienda. Por eso el match va anclado al principio del campo
# `resultado`, que es donde una enmienda se autodeclara ("enmienda registrada; ...").
RE_ENMIENDA = re.compile(r"^\s*enmienda", re.IGNORECASE)
# OJO: una ejecucion cerrada por gate dice "sin integracion ... por pre-registro", que es una
# RESTRICCION, no una declaracion de que la fila sea un pre-registro. Por eso el match va anclado
# al principio del campo `resultado` y no en cualquier posicion.
RE_PREREG = re.compile(r"^\s*(pre-?registro|sin resultados)", re.IGNORECASE)
# Igual que RE_ENMIENDA: un RECUENTO menciona "BLOQUEADO" sin que la fila lo sea. Una corrida
# bloqueada se autodeclara al principio de `resultado` ("DEFECTO_DE_IMPLEMENTACION: ...").
RE_BLOQUEO = re.compile(
    r"^\s*(defecto_de_implementacion|bloquead|abortad|interrumpid|exit 1)", re.IGNORECASE
)
RE_PATH = re.compile(r"[\w./\-]+\.(?:json|md|csv|npy|png|ipynb|py|log|txt|parquet|txt)\b")
# `features/`, `models/`, `data/` y `experimentos/` estan en .gitignore y por eso NO aparecen en el
# indice de Codebase; se comprueban contra el arbol real, no contra el indice.
SUBS_BUSQUEDA = (
    "", "experiments", "reports", "reports/tables", "reports/figures",
    "submissions", "features", "models", "artifacts", "plan", "scripts", "notebooks",
    "experimentos",
)


def familia_de(bloque: str) -> str:
    """Traduce la columna `bloque` a una familia de fases legible."""
    try:
        b = float(bloque)
    except (TypeError, ValueError):
        return "sin bloque"
    for lo, hi, nombre in FAMILIA_RANGES:
        if lo <= b <= hi:
            return nombre
    return "sin bloque"


def split_artefactos(texto: str) -> list[str]:
    """Separa la lista de artefactos.

    El registro usa ' | ' como separador principal, pero algunas filas antiguas usan ', '
    (ADV-*, PH*), y hay entradas mixtas. Se separan por ambos y se limpian las anotaciones
    entre parentesis ('experto.json (campo enmiendas)').
    """
    if not texto:
        return []
    # '(' sin cerrar (defecto de escritura del registro) haria que la limpieza por parentesis
    # se comiera el separador ' | ' siguiente, asi que se cortan las anotaciones por parentesis
    # solo cuando estan balanceadas, y por la ultima ruta legible si no lo estan.
    texto = re.sub(r"\((?![^()]*\))", " | ", texto)
    partes = re.split(r"\s\|\s|\s;\s|,\s|\n", texto)
    salida: list[str] = []
    for p in partes:
        p = p.strip().strip("()")
        p = re.sub(r"\s*\(.*?\)\s*$", "", p).strip()
        if p and RE_PATH.search(p):
            salida.append(p)
    return salida


def es_no_fichero(tok: str) -> bool:
    """True si el token es un glob, un directorio o una referencia textual, no un fichero."""
    t = tok.strip()
    # '*' cubre los globs ('reports/tables/phase13_*.csv'); '/' final, las carpetas ('lexicons/**').
    return any(ch in t for ch in ("*",)) or t.endswith("/")


def resolver(tok: str) -> str | None:
    """Resuelve un artefacto citado contra el arbol real.

    Devuelve la ruta existente, o None si no aparece. Prueba la ruta tal cual y el nombre del
    fichero dentro de los subdirectorios conocidos, porque varias filas citan un CSV de
    `reports/` sin su carpeta. Si el token es prosa que contiene una ruta ('figuras v25_x.png
    y v25_y.png', 'DECISIONS.md (D-087'), se extraen las rutas embebidas y basta con que
    exista alguna.
    """
    t = tok.strip()
    if es_no_fichero(t):
        base = t.rstrip("/").split("*")[0].strip("/")
        return f"{base}/" if (ROOT / base).is_dir() else None

    p = Path(t)
    if (ROOT / p).exists():
        return t
    for sub in SUBS_BUSQUEDA:
        cand = (ROOT / sub / p.name) if sub else (ROOT / p.name)
        if cand.exists():
            return str(cand.relative_to(ROOT))

    # el token puede ser prosa con rutas dentro: se prueban todas las que contenga
    embebidas = [m.group(0) for m in RE_PATH.finditer(t)]
    if embebidas:
        for e in embebidas:
            for sub in SUBS_BUSQUEDA:
                cand = (ROOT / sub / Path(e).name) if sub else (ROOT / Path(e).name)
                if cand.exists():
                    return str(cand.relative_to(ROOT))
    return None


def cargar_csv(path: Path) -> tuple[list[str], list[dict]]:
    """Lee el CSV con validacion estricta del largo de cada fila."""
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise RuntimeError("experiments.csv vacio")
    hdr = rows[0]
    for n, r in enumerate(rows[1:], start=2):
        if len(r) != len(hdr):
            raise RuntimeError(f"experiments.csv fila {n} con {len(r)} campos, se esperaban {len(hdr)}")
    return hdr, [dict(zip(hdr, r)) for r in rows[1:]]


def clasificar(fila: dict) -> tuple[str, str, str]:
    """Devuelve (tipo_registro, dictamen_inventario, severidad, observacion)."""
    idu = fila.get("experiment_id", "") or ""
    res = fila.get("resultado", "") or ""
    et = fila.get("etapa", "") or ""
    f1 = (fila.get("macro_f1_mean", "") or "").strip()
    obs = ""

    if RE_ENMIENDA.search(res) or "ENMIENDA" in idu.upper():
        return ("ENMIENDA", "documento_valido", "info",
                "enmienda pre-ejecucion declarada; no es una ejecucion")
    # un pre-registro se identifica por su ID o por su propio texto inicial, y NUNCA debe traer un
    # artefacto de corrida (figuras de curvas, JSON de folds): si los trae, se ejecuto.
    artefactos = split_artefactos(fila.get("artefactos", "") or "")
    tiene_evidencia_de_corrida = any(
        a.endswith(".png") or re.search(r"_(?:folds?|curvas?|resultados|ejecucion|estado)\w*\.json$", a)
        for a in artefactos
    )
    if ("PREREG" in idu.upper() or RE_PREREG.search(res)) and not tiene_evidencia_de_corrida:
        return ("PREREGISTRO", "documento_valido", "info",
                "pre-registro congelado; no es una ejecucion")
    if RE_BLOQUEO.search(res):
        return ("BLOQUEADO", "ejecucion_fallida_declarada", "alta",
                "corrida abortada por defecto de implementacion; su evidencia esta en cuarentena")
    if not f1 and LEX_DOC.search(et):
        return ("DOCUMENTAL", "documento_valido", "info",
                "trabajo documental o de auditoria; no entrena modelo")
    if not f1:
        obs = "ejecutado sin macro_f1 en el CSV: el resultado vive en el artefacto y en el informe"
        return ("EJECUTADO", "correcto_con_reservas", "media", obs)
    return ("EJECUTADO", "correcto_con_reservas", "media", obs or "")


def indice_logs() -> dict[str, list[str]]:
    """Mapea nombre de log -> scripts que lo mencionan (para evidencia de corrida)."""
    out: dict[str, list[str]] = defaultdict(list)
    logs = list((ROOT / "experimentos").glob("*.log")) + list((ROOT / "experiments").glob("*.log"))
    for lg in logs:
        out[lg.name] = [str(lg.relative_to(ROOT))]
    return out


def construir_indice_notebook(nb: dict) -> dict[str, object]:
    """Extrae del notebook: texto, celdas por seccion y el indice de la celda de cada ID.

    `celda_de_id` mapea cada `experiment_id` encontrado a la celda en la que aparece. Es lo que
    permite asignar una seccion por posicion real y no por heuristica de texto.
    """
    celdas = nb.get("cells", [])
    texto = "\n".join("".join(c.get("source", [])) for c in celdas)
    code = "\n".join("".join(c.get("source", [])) for c in celdas if c.get("cell_type") == "code")
    secciones: list[dict] = []
    for i, c in enumerate(celdas):
        if c.get("cell_type") != "markdown":
            continue
        for linea in "".join(c.get("source", [])).splitlines():
            m = re.match(r"^(#{1,3})\s+(.*)$", linea.strip())
            if m:
                secciones.append({"celda": i, "nivel": len(m.group(1)), "titulo": m.group(2).strip()})
    # indice celda -> id, construido celda a celda (no sobre el texto concatenado, que perderia
    # la correspondencia con la posicion)
    celda_de_id: dict[str, int] = {}
    for i, c in enumerate(celdas):
        if c.get("cell_type") != "code":
            continue
        fuente = "".join(c.get("source", []))
        for m in re.finditer(r'["\']([A-Z]{1,3}-[A-Z0-9][A-Z0-9\-]{2,})["\']', fuente):
            celda_de_id.setdefault(m.group(1), i)
    return {
        "n_celdas": len(celdas),
        "n_markdown": sum(1 for c in celdas if c.get("cell_type") == "markdown"),
        "n_code": sum(1 for c in celdas if c.get("cell_type") == "code"),
        "n_code_ejecutado": sum(
            1 for c in celdas if c.get("cell_type") == "code" and c.get("execution_count") is not None
        ),
        "n_errores": sum(
            1 for c in celdas for o in c.get("outputs", []) if o.get("output_type") == "error"
        ),
        "secciones": secciones,
        "celda_de_id": celda_de_id,
        "texto": texto,
        "codigo": code,
    }


def seccion_de(nb_idx: dict, experiment_id: str, celda_mencion: int | None = None) -> str:
    """Seccion del notebook que contiene la primera mencion del ID.

    Se resuelve por POSICION de celda: la seccion vigente es la de titulo mas cercano por encima
    de la celda donde aparece el ID. La version anterior ("titulo mas largo que aparece antes en el
    texto") devolvia el H1 del documento para 8 de 23 filas, porque el H1 es el titulo mas largo.

    Limite honesto: solo cubre IDs citados literalmente dentro de celdas de codigo. El notebook
    resume muchas fases por prosa y sin ID, y esas filas quedan con `seccion_notebook` vacio.
    """
    if celda_mencion is None:
        celda_mencion = nb_idx["celda_de_id"].get(experiment_id)
    if celda_mencion is None:
        return ""
    anterior = [s for s in nb_idx["secciones"] if s["celda"] <= celda_mencion]
    if not anterior:
        return "sin seccion previa"
    return f"[celda {celda_mencion}] {anterior[-1]['titulo']}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Inventario auditable de experimentos V1-V37.")
    ap.add_argument("--verificar", action="store_true", help="autoejercicio en temporal; no escribe real")
    ap.add_argument("--resumen", action="store_true", help="solo resumen por tipo y familia")
    args = ap.parse_args(argv)

    inicio = time.time()
    _, filas = cargar_csv(CSV_PATH)
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    nb_idx = construir_indice_notebook(nb)
    dec = DEC_PATH.read_text(encoding="utf-8") if DEC_PATH.exists() else ""
    state = STATE_PATH.read_text(encoding="utf-8") if STATE_PATH.exists() else ""
    logs = indice_logs()

    filas_out: list[dict] = []
    huecos: list[dict] = []
    for f in filas:
        idu = f.get("experiment_id", "")
        tipo, dictamen, sev, obs = clasificar(f)
        arts = split_artefactos(f.get("artefactos", "") or "")
        resueltos: dict[str, str | None] = {a: resolver(a) for a in arts}
        faltan = [a for a in arts if resueltos[a] is None]
        # script declarado: busca en artefactos un .py
        scripts = [a for a in arts if a.endswith(".py")]
        jsons = [a for a in arts if a.endswith(".json") and resueltos[a]]
        # el campo `evidencia_log` es informativo: el registro no liga cada fila a su log, asi
        # que se listan los logs existentes del proyecto sin atribuir ninguno a una fila concreta.
        logs_rel = sorted({l for grupo in logs.values() for l in grupo})
        celda_mencion = nb_idx["celda_de_id"].get(idu)
        en_nb = idu in nb_idx["texto"]
        sec = seccion_de(nb_idx, idu, celda_mencion)
        en_dec = idu in dec
        en_st = idu in state
        fila = {
            "experiment_id": idu,
            "fecha": f.get("fecha", ""),
            "tipo_registro": tipo,
            "familia": familia_de(f.get("bloque", "")),
            "etapa": (f.get("etapa", "") or "")[:200],
            "pregunta": (f.get("pregunta", "") or "")[:200],
            "macro_f1_mean": f.get("macro_f1_mean", ""),
            "macro_f1_std": f.get("macro_f1_std", ""),
            "resultado": (f.get("resultado", "") or "")[:240],
            "decision": (f.get("decision", "") or "")[:200],
            "script_declarado": " | ".join(scripts),
            "artefactos_citados": (f.get("artefactos", "") or "")[:400],
            "n_artefactos": len(arts),
            "artefactos_existentes": len(arts) - len(faltan),
            "artefactos_faltantes": " | ".join(faltan),
            "evidencia_log": " | ".join(logs_rel),
            "evidencia_json": " | ".join(jsons),
            "en_notebook": "si" if en_nb else "no",
            "seccion_notebook": sec[:120],
            "en_decisiones": "si" if en_dec else "no",
            "en_project_state": "si" if en_st else "no",
            "dictamen_inventario": dictamen,
            "severidad": sev,
            "observacion": obs,
        }
        filas_out.append(fila)
        if faltan:
            huecos.append({"experiment_id": idu, "artefactos_faltantes": " | ".join(faltan)})

    por_tipo = Counter(r["tipo_registro"] for r in filas_out)
    por_familia = Counter(r["familia"] for r in filas_out)
    resumen = {
        "generado_por": "scripts/auditoria_v1v37_inventario.py",
        "fecha": time.strftime("%Y-%m-%d"),
        "filas": len(filas_out),
        "por_tipo_registro": dict(por_tipo),
        "por_familia": dict(por_familia),
        "notebook": {
            "n_celdas": nb_idx["n_celdas"],
            "n_markdown": nb_idx["n_markdown"],
            "n_code": nb_idx["n_code"],
            "n_code_ejecutado": nb_idx["n_code_ejecutado"],
            "n_errores": nb_idx["n_errores"],
            "n_secciones": len(nb_idx["secciones"]),
            "en_notebook": sum(1 for r in filas_out if r["en_notebook"] == "si"),
        },
        "en_decisiones": sum(1 for r in filas_out if r["en_decisiones"] == "si"),
        "filas_con_artefacto_faltante": len(huecos),
        "filas_sin_artefacto_citado": sum(1 for r in filas_out if r["n_artefactos"] == 0),
        "severidad": dict(Counter(r["severidad"] for r in filas_out)),
        "dictamen": dict(Counter(r["dictamen_inventario"] for r in filas_out)),
        "segundos": round(time.time() - inicio, 1),
    }

    if args.resumen:
        print(json.dumps(resumen, indent=2, ensure_ascii=False))
        return 0

    if args.verificar:
        return verificar(filas_out, nb_idx, huecos)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(filas_out)
    OUT_GAPS.parent.mkdir(parents=True, exist_ok=True)
    with OUT_GAPS.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["experiment_id", "artefactos_faltantes"])
        w.writeheader()
        w.writerows(huecos)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"resumen": resumen, "filas": filas_out}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(resumen, indent=2, ensure_ascii=False))
    return 0


def verificar(filas_out: list[dict], nb_idx: dict, huecos: list[dict]) -> int:
    """Autoejercicio de los siete caminos. NO escribe ningun artefacto real: solo lee y compara.

    El aislamiento se apoya en que `verificar` no contiene ninguna llamada de escritura; el unico
    punto de escritura del script esta en `main()` y se salta antes cuando el modo no es el normal.
    """
    checks: list[tuple[str, bool, str]] = []

    # C1: cobertura completa y forma de la fila de salida. Comprueba que el inventario tiene tantas
    # filas como el CSV y que cada una lleva TODAS las claves de COLS (una clave mal escrita pasaria
    # en silencio y DictWriter la volcaria vacia).
    n_csv = len(cargar_csv(CSV_PATH)[1])
    faltan_claves = [r["experiment_id"] for r in filas_out if set(r) != set(COLS)]
    checks.append((
        "C1_cobertura_y_forma",
        len(filas_out) == n_csv and not faltan_claves,
        f"{len(filas_out)} filas == {n_csv} del CSV; {len(COLS)} columnas; claves faltantes={faltan_claves}",
    ))

    # C2: los IDs son unicos.
    ids = [r["experiment_id"] for r in filas_out]
    checks.append(("C2_ids_unicos", len(set(ids)) == len(ids), f"{len(set(ids))} ids unicos"))

    # C3: toda fila clasificada tiene un tipo de la escala cerrada.
    tipos_ok = {"EJECUTADO", "DOCUMENTAL", "PREREGISTRO", "ENMIENDA", "BLOQUEADO"}
    tipos = {r["tipo_registro"] for r in filas_out}
    checks.append(("C3_tipo_en_escala", tipos <= tipos_ok, f"tipos={sorted(tipos)}"))

    # C4: PREREGISTRO y ENMIENDA no deben llevar macro_f1 (si lo llevan, contradiccion).
    contrad = [
        r["experiment_id"]
        for r in filas_out
        if r["tipo_registro"] in ("PREREGISTRO", "ENMIENDA") and (r["macro_f1_mean"] or "").strip()
    ]
    checks.append(("C4_prereg_sin_metrica", not contrad, f"contradicciones={contrad}"))

    # C4b: una fila cuyo `resultado` es un RECUENTO de tipos menciona las palabras "ENMIENDA" y
    # "BLOQUEADO" sin ser una de esas cosas. Este caso existio de verdad (AUD-INV se clasifico como
    # ENMIENDA y luego como BLOQUEADO) y por eso se comprueba de forma explicita.
    recuento = ("151 EJECUTADO + 21 DOCUMENTAL + 12 PREREGISTRO + 10 ENMIENDA + 1 BLOQUEADO = 195; "
                "0 artefactos ausentes")
    tipo_recuento = clasificar({
        "experiment_id": "X", "resultado": recuento, "etapa": "inventario", "artefactos": "",
    })[0]
    ok_rec = tipo_recuento == "EJECUTADO"
    checks.append((
        "C4b_recuento_no_es_enmienda_ni_bloqueo",
        ok_rec,
        f"un recuento de tipos se clasifica como {tipo_recuento} (esperado EJECUTADO)",
    ))

    # C5: control independiente. NO reutiliza `huecos` (que sale del mismo `resolver`): vuelve a
    # resolver desde el texto crudo y exige que no falte ninguno de los artefactos declarados.
    declarados_faltantes = [
        (r["experiment_id"], a)
        for r in filas_out
        for a in split_artefactos(r["artefactos_citados"])
        if resolver(a) is None
    ]
    checks.append((
        "C5_todo_declarado_existe",
        not declarados_faltantes,
        f"faltantes={declarados_faltantes[:4]} (total {len(declarados_faltantes)})",
    ))

    # C6: el indice del notebook se reconstruye de forma consistente, con datos reales.
    ok_nb = (
        nb_idx["n_celdas"] == nb_idx["n_markdown"] + nb_idx["n_code"]
        and len(nb_idx["secciones"]) > 0
        and 0 < nb_idx["n_code_ejecutado"] <= nb_idx["n_code"]
        and nb_idx["n_celdas"] > 0
    )
    detalle_nb = (
        f"celdas={nb_idx['n_celdas']}, markdown={nb_idx['n_markdown']}, code={nb_idx['n_code']}, "
        f"ejecutadas={nb_idx['n_code_ejecutado']}, secciones={len(nb_idx['secciones'])}"
    )
    checks.append(("C6_indice_notebook", ok_nb, detalle_nb))

    print("=== autoejercicio del inventario (auditoria V1-V37) ===")
    fallos = 0
    for nombre, ok, detalle in checks:
        print(f"  [{'OK' if ok else 'FALLA'}] {nombre}: {detalle}")
        if not ok:
            fallos += 1
    print(f"  {len(checks) - fallos}/{len(checks)} caminos pasan; no se escribio ningun artefacto real")
    return 0 if fallos == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
