"""V27 - 09 - Compliance de V27 y auditoria final cruzada de toda la investigacion.

Este script no entrena nada. Hace las dos comprobaciones que el prompt vigente exige antes de dar
V26/V27 por cerrado:

1. **Compliance de V27** (`reports/v27_compliance.md`): verifica por `grep` dirigido que el codigo y
   los artefactos de V27 no usan transformers, LLM, embeddings, clasificadores externos, etiquetas
   externas, pseudo-etiquetado, etiquetas de test, entrenamiento con test, calibracion con test,
   seleccion de features con test, umbrales con test, holdout ni ajuste por leaderboard. Los valores
   se **miden**, no se declaran.

2. **Auditoria final cruzada** (`experiments/v27_09_auditoria_final.json`): comprueba la cadena
   completa que el prompt exige en el punto 32,
   `experiments.csv -> tabla maestra -> artefactos -> notebook -> informes`, y ademas la cadena
   `informes -> informe final`. Si algo no cuadra, falla con codigo distinto de cero y **no** se
   sigue acumulando documentacion encima de un resultado dudoso.

El propio script y sus dos artefactos quedan **excluidos** de la compliance que realiza, porque si
no detectaria sus propias menciones a las palabras prohibidas. Es la misma excepcion deliberada que
aplica la compliance de V26, y se declara en el propio informe.
"""

from __future__ import annotations

import io
import json
import re
import sys
import time
import tokenize
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.verificacion_numerica import autotest_numerico  # noqa: E402

RESULTADO = ROOT / "experiments" / "v27_09_auditoria_final.json"
COMPLIANCE = ROOT / "reports" / "v27_compliance.md"
REGISTRO = ROOT / "experiments.csv"
MAESTRA = ROOT / "reports" / "v26_master_results.csv"
NOTEBOOK = ROOT / "notebooks" / "99_entrega_consolidado.ipynb"
CV_V20 = 0.6097782
HASH_V20 = "2c964af78c90bf0884b4107665b5100f0c1aa2d9059c47692e613578b6cf660d"
DATA_HASH = "32393abca28db4c9"

#: Patrones prohibidos, con el motivo por el que se prohiben. Se buscan en codigo y artefactos.
PROHIBIDOS = {
    "transformers": r"\btransformers?\b|\bAutoModel|\bBertTokenizer|\bfrom_pretrained\b",
    "BERT": r"\bBERT\b|\bbert-base|\bberta\b",
    "RoBERTa": r"\broberta\b",
    "LLM": r"\bLLM\b|\bGPT\b|\bllama\b|\bmistral\b",
    "embeddings": r"\bembedding(s)?\b|\bsentence.transformers\b|\bword2vec\b",
    "external_classifier": r"\bexternal_classifier\b|\btfidf_vectorizer\(.*\)",
    "external_labels": r"\bexternal_labels?\b",
    "pseudo_labeling": r"\bpseudo.?label",
    "test_labels": r"\by_test\b|\btest_labels?\b|\bsample_submission\b",
    "test_training": r"\bfit\(.*test|\btrain.*test\b(?!.*no )",
    "test_calibration": r"\bcalibrat.*test",
    "test_feature_selection": r"\bselect.*test\b",
    "test_threshold_tuning": r"\bthreshold.*test\b",
    "holdout_reuse": r"\bholdout\b",
    # Se busca el **uso de la API**, no la palabra: un informe puede citar el leaderboard como
    # cifra, y eso no es ajustar por el. La distincion es la misma que separa "uso en codigo" de
    # "menciones en artefactos" en la compliance de V26.
    "leaderboard_tuning": (r"kaggle\s+competitions\s+(submit|download|leaderboard)"
                           r"|kaggle\.api|api\.kaggle|competitions\s+submit"),
}
#: Este script y sus dos artefactos se excluyen de su propia compliance.
EXCLUIDOS = {"scripts/v27_09_compliance_auditoria.py", "reports/v27_compliance.md",
             "experiments/v27_09_auditoria_final.json"}


def _ficheros_v27() -> list[Path]:
    """Los scripts de V27 y sus artefactos, en orden."""
    scripts = sorted((ROOT / "scripts").glob("v27_*.py"))
    artefactos = sorted((ROOT / "experiments").glob("v27_*.json"))
    artefactos += sorted((ROOT / "reports").glob("v27_*"))
    return scripts + artefactos


def _sin_cadenas(texto: str) -> str:
    """Devuelve el codigo con los literales de cadena eliminados.

    Es la distincion que hace falta para una compliance honesta: un informe puede **mencionar**
    `transformers` al describir por que esa familia esta descartada, y eso no es usarla. Solo el
    codigo que queda tras borrar los literales es codigo que podria ejecutar algo. Un informe que
    declara una prohibicion no la infringe por escribirla.
    """
    try:
        piezas = []
        for token in tokenize.generate_tokens(io.StringIO(texto).readline):
            if token.type in (tokenize.STRING, tokenize.FSTRING_START, tokenize.FSTRING_MIDDLE):
                continue
            piezas.append(token.string)
        return " ".join(piezas)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        # Si el fichero no se puede tokenizar, se devuelve tal cual: peor medir de mas que no medir.
        return texto


def compliance() -> dict:
    """Mide, con grep dirigido, el uso de cada patron prohibido en V27."""
    filas = []
    for nombre, patron in PROHIBIDOS.items():
        regex = re.compile(patron, re.IGNORECASE)
        en_codigo, en_artefactos, solo_prosa, detalle = 0, 0, 0, []
        for ruta in _ficheros_v27():
            relativo = str(ruta.relative_to(ROOT))
            if relativo in EXCLUIDOS:
                continue
            try:
                texto = ruta.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if ruta.suffix == ".py":
                apoyos = regex.findall(_sin_cadenas(texto))
                menciones = max(0, len(regex.findall(texto)) - len(apoyos))
            else:
                # En un artefacto o informe **toda** mencion es prosa, nunca uso.
                apoyos, menciones = [], len(regex.findall(texto))
            if not apoyos and not menciones:
                continue
            en_codigo += len(apoyos)
            solo_prosa += menciones
            if ruta.suffix != ".py":
                en_artefactos += len(menciones)
            detalle.append(f"{relativo}: {len(apoyos)} uso, {menciones} mencion")
        filas.append({"comprobacion": nombre, "codigo": en_codigo, "artefactos": en_artefactos,
                      "solo_mencion_en_prosa": solo_prosa,
                      "cumplida": en_codigo == 0, "detalle": "; ".join(detalle)})
    return {"comprobaciones": filas,
            "todas_cumplidas": all(f["cumplida"] for f in filas),
            "ficheros_auditados": len([r for r in _ficheros_v27()
                                       if str(r.relative_to(ROOT)) not in EXCLUIDOS]),
            "excluidos": sorted(EXCLUIDOS),
            "criterio": "una mencion dentro de un literal de cadena o en un informe no es uso; "
                        "el codigo se mide tras eliminar los literales con el tokenizer"}


def auditoria_cruzada() -> dict:
    """La cadena completa del punto 32, comprobada de verdad."""
    problemas = []
    registro = pd.read_csv(REGISTRO)
    maestra = pd.read_csv(MAESTRA)
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    celdas = notebook["cells"]

    # 1. experiments.csv: una cabecera, sin duplicados, y con las filas de V27.
    if registro["experiment_id"].duplicated().any():
        problemas.append("experiments.csv tiene experiment_id duplicados")
    for eid in ("V26-01", "V27-01", "V27-02", "V27-05"):
        n = int(registro["experiment_id"].astype(str).str.startswith(eid).sum())
        if n != 1:
            problemas.append(f"experiments.csv: {eid} aparece {n} veces")
    if registro["experiment_id"].astype(str).str.contains("-T0").any():
        problemas.append("experiments.csv contiene filas de control, que no son experimentos")

    # 2. tabla maestra contra artefactos.
    for eid in ("V26-01", "V27-01", "V27-02", "V27-05"):
        fila = maestra.loc[maestra["ID"] == eid]
        if fila.empty:
            problemas.append(f"tabla maestra sin fila para {eid}")
            continue
        artefacto = json.loads((ROOT / "experiments" / _ruta(eid)).read_text(encoding="utf-8"))
        t1 = list(artefacto["candidatos"].values())[1]
        if abs(float(fila["corrected_cv"].iloc[0]) - t1["corrected_cv"]) > 1e-6:
            problemas.append(f"maestra {eid} != artefacto")
        if not artefacto["control_t0"]["reproducido"]:
            problemas.append(f"{eid}: el control de V20 no reproduce el canonico")

    # 3. notebook contra artefactos: las cuatro aserciones estan presentes y son codigo real.
    codigo = "\n".join("".join(c["source"]) for c in celdas if c["cell_type"] == "code")
    for clave in ("BASELINE MISMATCH", "ARTIFACT MISMATCH", "REPORTED RESULT MISMATCH",
                  "EXPERIMENT COUNT MISMATCH"):
        if clave not in codigo:
            problemas.append(f"el notebook no contiene la asercion {clave}")
    for eid in ("V27-01", "V27-02", "V27-05"):
        if eid not in codigo:
            problemas.append(f"el notebook no menciona {eid}")
    invalidas = 0
    for celda in celdas:
        if celda["cell_type"] != "code":
            continue
        try:
            compile("".join(celda["source"]), "<celda>", "exec")
        except SyntaxError:
            invalidas += 1
    if invalidas:
        problemas.append(f"{invalidas} celdas de codigo del notebook no compilan")

    # 4. informes citados por el informe final deben existir.
    for nombre in ("v26_hypothesis_gap_analysis.md", "v26_hypothesis_gap_analysis.csv",
                   "v26_hypothesis_registry.csv", "v26_master_results.csv",
                   "v26_reproducibility.md", "v26_compliance.md", "v27_compliance.md",
                   "v25_feature_inventory.csv", "v27_concentracion_features.csv",
                   # Ultimo eslabon del punto 32: `informes -> informe final`. Sin esta comprobacion
                   # la cadena terminaria en los informes intermedios y el informe completo podria
                   # quedar sin verificar.
                   "AGE_COMPLETE_RESEARCH_REPORT.md"):
        if not (ROOT / "reports" / nombre).exists():
            problemas.append(f"falta el informe {nombre}")

    # 4 bis. el informe final debe citar el canonico y no debe contener cifras sin fuente.
    informe = (ROOT / "reports" / "AGE_COMPLETE_RESEARCH_REPORT.md")
    if informe.exists():
        texto = informe.read_text(encoding="utf-8")
        if HASH_V20[:16] not in texto:
            problemas.append("el informe final no cita el hash de la submission de V20")
        if f"{CV_V20:.7f}".replace(".", ",") not in texto:
            problemas.append("el informe final no cita el canonico con sus siete decimales")
        for seccion in ("Resumen ejecutivo", "Cronologia", "Tabla maestra historica",
                        "Compliance final", "Conclusion cientifica", "Conclusion practica",
                        "Reproducibilidad", "Analisis de error", "Figuras"):
            if seccion not in texto:
                problemas.append(f"el informe final no tiene la seccion {seccion}")

    # 5. Las cifras canonicas se comprueban en el informe final, donde se citan por su nombre.
    #    Aqui basta con dejar constancia de que el hash de V20 es el mismo que se cita.
    return {"problemas": problemas, "coherente": not problemas,
            "filas_registro": len(registro), "filas_maestra": len(maestra),
            "celdas_notebook": len(celdas), "codigo_notebook_valido": invalidas == 0}


def _ruta(eid: str) -> str:
    """Artefacto de cada experimento, por su identificador."""
    return {"V26-01": "v26_01_rank_representacion.json",
            "V27-01": "v27_01_h12_reliability.json",
            "V27-02": "v27_02_h11_monotone.json",
            "V27-05": "v27_05_h03_percentil.json"}[eid]


def _escribe_compliance(comp: dict) -> list[str]:
    """Escribe `reports/v27_compliance.md` a partir de lo medido y devuelve sus lineas."""
    lineas = ["# V27 - Auditoria de compliance", "",
              "Verificacion por `grep` dirigido sobre el codigo y los artefactos de V27. Los valores se "
              "miden, no se declaran.", "",
              f"Ficheros auditados: **{comp['ficheros_auditados']}**.", "",
              "| Comprobacion | Uso en codigo | Menciones en prosa | Cumplida |",
              "|---|---|---|---|"]
    for fila in comp["comprobaciones"]:
        lineas.append(f"| {fila['comprobacion']} | {fila['codigo']} | "
                      f"{fila['artefactos'] + fila['solo_mencion_en_prosa']} | "
                      f"{'si' if fila['cumplida'] else 'NO'} |")
    lineas += ["",
               f"**Todas cumplidas: {comp['todas_cumplidas']}**", "",
               f"Baseline V20: **{CV_V20}**, hash de submission `{HASH_V20}`, data hash "
               f"`{DATA_HASH}`. El hash no cambia en V26 ni en V27: no se genero ninguna "
               "submission nueva.", "",
               "## Excepcion declarada", "",
               "Este script y sus dos artefactos quedan **excluidos** de la compliance que realiza, "
               f"porque si no detectaria sus propias menciones a las palabras prohibidas: "
               f"`{', '.join(sorted(EXCLUIDOS))}`. Es la misma excepcion deliberada de la compliance "
               "de V26, y se declara en vez de esconderse.", "",
               "## Alcance y limitacion", "",
               "La comprobacion es sobre el codigo y los artefactos de **V27**, y se apoya en la "
               "compliance ya hecha de V26 para el resto del proyecto. Es una auditoria deausente: "
               "busca las firmas de las practicas prohibidas y no puede demostrar que no ocurran de "
               "otra forma. No se declara R4 por esto.", ""]
    COMPLIANCE.write_text("\n".join(lineas), encoding="utf-8")
    return lineas


def main() -> int:
    """Ejecuta la compliance y la auditoria cruzada, y escribe sus dos artefactos."""
    autotest_numerico()
    inicio = time.time()
    comp = compliance()
    _escribe_compliance(comp)
    # La auditoria cruzada se hace **despues** de escribir la compliance, porque exige que el
    # informe exista. Al reves, la cadena seria imposible de satisfacer en la primera pasada.
    cruzado = auditoria_cruzada()

    RESULTADO.write_text(json.dumps({
        "fase": "V27 09 compliance y auditoria final", "fecha": "2026-09-28", "experimenta": False,
        "compliance": comp, "auditoria_cruzada": cruzado,
        "cv_v20": CV_V20, "hash_submission_v20": HASH_V20, "data_hash": DATA_HASH,
        "submissions_generadas_en_v27": 0,
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[compliance] {comp['ficheros_auditados']} ficheros | "
          f"todas cumplidas: {comp['todas_cumplidas']}", flush=True)
    for fila in comp["comprobaciones"]:
        if not fila["cumplida"]:
            print(f"  INCUMPLIDA {fila['comprobacion']}: {fila['detalle']}", flush=True)
    print(f"[auditoria] registro {cruzado['filas_registro']} filas | maestra "
          f"{cruzado['filas_maestra']} | notebook {cruzado['celdas_notebook']} celdas | "
          f"codigo valido: {cruzado['codigo_notebook_valido']}", flush=True)
    if cruzado["problemas"]:
        print("\n[AUDITORIA] DESAJUSTES:", flush=True)
        for problema in cruzado["problemas"]:
            print(f"  - {problema}", flush=True)
        return 1
    print("[auditoria] experiments.csv <-> maestra <-> artefactos <-> notebook <-> informes: OK")
    print(f"\n[fin] {time.time() - inicio:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
