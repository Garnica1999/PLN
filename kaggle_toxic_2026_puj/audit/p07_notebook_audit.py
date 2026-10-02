"""P7: auditoria del notebook consolidado 10_final_pipeline.ipynb.

- Inventario de celdas con salidas, errores y avisos.
- Cobertura por familia de experimentos (codigo / resultados / decision).
- Coherencia de numeros clave frente a los artefactos.
Escribe reports/audit_notebook.csv, reports/audit_notebook_gaps.md y
audit/out/p07_cells.csv.
"""
from __future__ import annotations

import csv
import json
import re
import unicodedata
from pathlib import Path

PROJ = Path("/home/carlos/Git/PLN-toxic-audit/kaggle_toxic_2026_puj")
NB = PROJ / "notebooks" / "10_final_pipeline.ipynb"

NUMEROS = ["0.777211", "0.7772", "0.78306", "0.7989", "0.7927", "0.8285", "0.8465",
           "0.78163", "0.77895", "0.77978", "0.78150", "0.77602", "0.77227", "0.77034",
           "0.7751", "0.75677", "0.77993", "0.51494", "0.7800", "0.8364", "0.8336"]
FAMILIAS = [
    ("V03 (referencia)", ["TfidfVectorizer", "LogisticRegression"], ["V03", "0.777211"],
     ["0.777211", "0.78306"]),
    ("v2 (correccion de protocolo)", ["grupos_fusionados", "pseudo"], ["D-015", "0.7838"],
     ["0.783825", "0.792659"]),
    ("v4 (holdout)", [], ["D-029", "holdout"], ["0.8163", "0.8285"]),
    ("v5/v5.1 (embeddings)", ["fasttext", "entrenar_fasttext"], ["EBD", "D-030", "D-031"],
     ["0.776613"]),
    ("v6 (tracks cerrados)", [], ["V6-", "track"], ["0.777211"]),
    ("v7 (umbral anidado)", [], ["V7-", "anidado"], ["0.767228", "0.769123"]),
    ("v8 (tecnicas)", [], ["V8-", "curva de transferencia"], ["0.780036"]),
    ("finales (FIN/meta v08)", [], ["FIN-", "meta"], ["0.8285", "0.8465", "0.78163"]),
]


def ascii_text(text: str) -> str:
    repl = {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
            "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u00a0": " "}
    for k, v in repl.items():
        text = text.replace(k, v)
    norm = unicodedata.normalize("NFKD", text)
    return "".join(ch if ord(ch) < 128 else "" for ch in norm if not unicodedata.combining(ch))


def texto_de_salidas(cell: dict) -> str:
    partes = []
    for out in cell.get("outputs", []):
        tipo = out.get("output_type")
        if tipo == "stream":
            partes.append("".join(out.get("text", [])))
        elif tipo in {"execute_result", "display_data"}:
            datos = out.get("data", {})
            if "text/plain" in datos:
                partes.append("".join(datos["text/plain"]))
        elif tipo == "error":
            partes.append("ERROR: " + " | ".join(out.get("ename", ""),
                                                  out.get("evalue", "")))
    return "\n".join(partes)


def main() -> None:  # noqa: PLR0915
    nb = json.loads(NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    filas_celdas = []
    texto_total_md, texto_total_code, texto_total_out = [], [], []
    ejecucion_no_monotona = []
    anterior = None
    for idx, cell in enumerate(cells):
        fuente = "".join(cell["source"])
        salida = texto_de_salidas(cell) if cell["cell_type"] == "code" else ""
        cifras = [n for n in NUMEROS if n in salida]
        errores = any(o.get("output_type") == "error" for o in cell.get("outputs", []))
        avisos = bool(re.search(r"Warning|warning", salida))
        trazas = salida.count("Traceback")
        exec_count = cell.get("execution_count")
        if cell["cell_type"] == "code" and isinstance(exec_count, int):
            if anterior is not None and exec_count < anterior:
                ejecucion_no_monotona.append(idx)
            anterior = exec_count
        archivos = sorted(set(re.findall(r"[\w\-/]+\.(?:py|csv|json|npz|md|png)",
                                         fuente + " " + salida)))[:8]
        filas_celdas.append({
            "cell": idx, "tipo": cell["cell_type"],
            "primera_linea": ascii_text(fuente.strip().splitlines()[0] if fuente.strip() else "")[:110],
            "n_outputs": len(cell.get("outputs", [])),
            "exec_count": exec_count if cell["cell_type"] == "code" else "",
            "tiene_error": int(errores), "tiene_warning": int(avisos),
            "trazas": trazas,
            "numeros_clave_en_salida": ",".join(cifras),
            "archivos_referenciados": ";".join(archivos),
        })
        if cell["cell_type"] == "markdown":
            texto_total_md.append(fuente)
        else:
            texto_total_code.append(fuente)
            texto_total_out.append(salida)
    full_md = "\n".join(texto_total_md)
    full_code = "\n".join(texto_total_code)
    full_out = "\n".join(texto_total_out)

    filas_familias = []
    for nombre, marcadores_code, marcadores_texto, cifras in FAMILIAS:
        code_hits = {m: full_code.count(m) for m in marcadores_code} if marcadores_code else {}
        texto_hits = {m: (full_md + full_out).count(m) for m in marcadores_texto}
        cifras_hits = {c: (full_md + full_out).count(c) for c in cifras}
        tiene_codigo = any(v > 0 for v in code_hits.values()) if code_hits else False
        tiene_texto = any(v > 0 for v in texto_hits.values())
        tiene_cifras = any(v > 0 for v in cifras_hits.values())
        filas_familias.append({
            "familia": nombre,
            "codigo_ejecutable": "si" if tiene_codigo else "no",
            "resultados_en_notebook": "si" if tiene_cifras else "no",
            "mencion_de_decision": "si" if tiene_texto else "no",
            "detalle": ("code=" + str(code_hits) + "; texto=" + str(texto_hits)
                        + "; cifras=" + str(cifras_hits)),
            "estado": ("reproducible_en_notebook" if tiene_codigo and tiene_cifras
                       else ("documentado_sin_codigo" if tiene_texto or tiene_cifras
                             else "ausente")),
        })

    # Coherencia de numeros clave en salidas frente a los artefactos
    referencias = {
        "0.777211": "corrected_cv V03", "0.78306": "LB V03", "0.7989": "OOF historico con umbrales",
        "0.7927": "OOF historico 0.5", "0.8285": "holdout V03", "0.8465": "holdout meta",
        "0.78163": "LB meta v08", "0.8163": "holdout meta36 (rechazado)",
    }
    coherentes = {n: (n in full_out) for n in referencias}

    salida_gaps = ["# P7 - Auditoria del notebook 10_final_pipeline.ipynb", ""]
    salida_gaps.append(f"- Celdas totales: {len(cells)} "
                       f"(codigo {sum(1 for c in cells if c['cell_type'] == 'code')}, "
                       f"markdown {sum(1 for c in cells if c['cell_type'] == 'markdown')})")
    errores = [f for f in filas_celdas if f["tiene_error"]]
    trazas = [f for f in filas_celdas if f["trazas"]]
    salida_gaps.append(f"- Celdas con error en salida: {len(errores)}; con Traceback: {len(trazas)}")
    if ejecucion_no_monotona:
        salida_gaps.append(f"- execution_count no monotono en celdas: {ejecucion_no_monotona[:12]}")
    salida_gaps.append("")
    salida_gaps.append("## Cobertura por familia")
    salida_gaps.append("")
    salida_gaps.append("| familia | codigo | resultados | decision | estado |")
    salida_gaps.append("|---|---|---|---|---|")
    for f in filas_familias:
        salida_gaps.append(f"| {f['familia']} | {f['codigo_ejecutable']} | "
                           f"{f['resultados_en_notebook']} | {f['mencion_de_decision']} | {f['estado']} |")
    salida_gaps.append("")
    salida_gaps.append("## Numeros clave presentes en SALIDAS del notebook")
    salida_gaps.append("")
    for n, desc in referencias.items():
        salida_gaps.append(f"- {n} ({desc}): {'presente' if coherentes[n] else 'AUSENTE'}")
    salida_gaps.append("")
    salida_gaps.append("## Nota")
    salida_gaps.append("El notebook resume la investigacion (markdown + tablas); las familias "
                       "v4-v8 estan documentadas con cifras pero no contienen su codigo de "
                       "ejecucion completo. Estado global: 'documentado, parcialmente "
                       "reproducible desde el propio notebook'.")

    with (PROJ / "reports" / "audit_notebook.csv").open("w", newline="", encoding="ascii") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(filas_familias[0].keys()))
        writer.writeheader()
        writer.writerows(filas_familias)
    with (PROJ / "audit" / "out" / "p07_cells.csv").open("w", newline="", encoding="ascii") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(filas_celdas[0].keys()))
        writer.writeheader()
        writer.writerows(filas_celdas)
    (PROJ / "reports" / "audit_notebook_gaps.md").write_text("\n".join(salida_gaps), encoding="ascii")

    print("celdas:", len(cells), "| errores:", len(errores), "| trazas:", len(trazas))
    for f in filas_familias:
        print(f["familia"], "->", f["estado"], "|", f["detalle"][:120])
    print("numeros en salidas:", {k: v for k, v in coherentes.items()})
    faltantes = [n for n, v in coherentes.items() if not v]
    print("ausentes en salidas:", faltantes)


if __name__ == "__main__":
    main()
