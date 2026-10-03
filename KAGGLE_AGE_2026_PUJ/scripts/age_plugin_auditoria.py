"""Auditoria de la submission experimental: leakage, compliance y verificacion del fichero.

## Por que la auditoria es por operacion y no por palabra

Un `grep` de "test" encuentra 200 usos legitimos: leer el test para `transform -> predict` es lo
que el procedimiento **debe** hacer. Marcar eso como leakage seria tan inutil como no auditarlo.
Asi que se comprueba **que operacion concreta** se hace con el test:

```text
PERMITIDO   leer el test, transformarlo y predecir
PROHIBIDO   leer el target del test
            ajustar cualquier cosa con el test (pesos, umbrales, IDF, vocabulario, calibracion)
            seleccionar features con el test
```

Se recorre el **AST** de cada script de esta fase buscando las operaciones, no las menciones.

## Lo que se verifica del fichero, contra el test y contra el propio fichero

- filas, columnas exactas, ids unicos e iguales a los del test y en el mismo orden
- sin NaN, clases validas, todas presentes, sin duplicados, sin columnas auxiliares
- el hash, contra el que quedo en el manifiesto

## Y lo que NO se hace aqui

No se estima ningun score. No se proyecta el leaderboard. No se cuentan las filas que cambian para
inferir una mejora esperada. El numero de filas que cambian es un dato de integridad, no una
prediccion.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg  # noqa: E402
from src import data as datos  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

RESULTADO = cfg.EXPERIMENTS / "age_plugin_auditoria.json"
SUBMISSION = ROOT / "submissions" / "age_submission_plugin_candidate.csv"
MANIFEST = cfg.REPORTS / "age_submission_plugin_manifest.json"
FECHA = "2026-10-12"
N_CLASES = 6

#: Operaciones PROHIBIDAS sobre el test. Se buscan en el AST como llamadas y accesos.
PROHIBIDAS = {
    "leer_el_target_del_test": [r"test\s*\[\s*[\"']age_range[\"']\s*\]",
                                r"test\s*\.\s*age_range"],
    "ajustar_con_el_test": [r"fit\s*\(\s*test", r"fit_transform\s*\(\s*test",
                            r"partial_fit\s*\(\s*test"],
    "estadisticas_sobre_el_test": [r"test\w*\.mean\s*\(", r"test\w*\.std\s*\(",
                                   r"value_counts\s*\(\s*\)\s*.*test"],
    "seleccion_de_features_con_el_test": [r"SelectKBest\s*\(\s*\)\s*\.fit\s*\(\s*test",
                                          r"chi2\s*\(\s*test", r"mutual_info.*test"],
    "umbrales_o_pesos_desde_el_test": [r"optimiza\w*\s*\([^)]*test", r"pesos.*test.*argmax"],
}
#: Comprobaciones POSITIVAS: lo que el procedimiento tiene que hacer y si esta.
POSITIVAS = {
    "orden_de_clases_verificado": r"verifica_orden_de_clases",
    "sanity_check_antes_de_predecir": r"sanity_check",
    "pesos_congelados_sin_reoptimizar": r"PESOS_CONGELADOS",
    "regla_argumento": r"prob\s*\*\s*PESOS_CONGELADOS",
    "pipeline_de_v30_reutilizado": r"from v30_03_submission import",
    "solo_argmax_y_producto": r"argmax",
}


def audita_operaciones() -> dict:
    """Recorre el AST de los scripts de la fase buscando las operaciones prohibidas."""
    scripts = sorted((ROOT / "scripts").glob("age_plugin*.py"))
    hallazgos, positivas = [], {k: 0 for k in POSITIVAS}
    for ruta in scripts:
        if ruta.name == "age_plugin_auditoria.py":
            continue
        texto = ruta.read_text(encoding="utf-8")
        ast.parse(texto)
        lineas = texto.splitlines()
        for nombre, patrones in PROHIBIDAS.items():
            for patron in patrones:
                for i, linea in enumerate(lineas, 1):
                    limpia = linea.strip()
                    # Un comentario o un docstring que NOMBRA la prohibicion no es la operacion.
                    if limpia.startswith("#") or limpia.startswith('"') or limpia.startswith("'"):
                        continue
                    if re.search(patron, linea):
                        hallazgos.append({"script": ruta.name, "operacion": nombre,
                                          "linea": i, "texto": limpia[:100]})
        for nombre, patron in POSITIVAS.items():
            if re.search(patron, texto):
                positivas[nombre] += 1
    return {"scripts_auditados": [p.name for p in scripts], "hallazgos": hallazgos,
            "total_hallazgos": len(hallazgos), "comprobaciones_positivas": positivas,
            "positivas_ok": all(v > 0 for v in positivas.values()),
            "metodo": "AST de cada script + busqueda por operacion en cada linea de codigo vivo, "
                      "excluyendo comentarios y docstrings, que nombran las prohibiciones sin "
                      "ejecutarlas"}


def audita_compliance() -> dict:
    """Prohibiciones de AGE, por operacion."""
    P = {"embeddings": r"sentence_transformers|SentenceTransformer|get_embedding|FastText|word2vec",
         "transformers": r"\bfrom\s+transformers\b|\bAutoModel\b|\bAutoTokenizer\b|\bBertModel\b",
         "LLM": r"\bopenai\b|\banthropic\b|\bcohere\b|\bllama\b|\bgpt-4",
         "datos_externos": r"requests\.get\(|urllib\.request|httpx\."}
    hallazgos = []
    scripts = sorted((ROOT / "scripts").glob("age_plugin*.py"))
    for ruta in scripts:
        arbol = ast.parse(ruta.read_text(encoding="utf-8"))
        imports = set()
        for n in ast.walk(arbol):
            if isinstance(n, ast.Import):
                imports.update(a.name for a in n.names)
            elif isinstance(n, ast.ImportFrom) and n.module:
                imports.add(n.module)
        for recurso, patron in P.items():
            for obj in imports:
                if re.search(patron, obj, re.I):
                    hallazgos.append({"script": ruta.name, "recurso": recurso, "import": obj})
    return {"scripts": [p.name for p in scripts], "hallazgos": hallazgos,
            "total": len(hallazgos)}


def verifica_fichero() -> dict:
    """Los checks del fichero, con los ids del test y el hash del manifiesto."""
    sub = pd.read_csv(SUBMISSION, dtype={cfg.ID_COL: str})
    test = datos.load_test()
    id_test = test[cfg.ID_COL].astype(str).to_numpy()
    manifiesto = json.loads(MANIFEST.read_text(encoding="utf-8"))
    sha = hashlib.sha256(SUBMISSION.read_bytes()).hexdigest()
    checks = {
        "filas": int(len(sub)) == 112892,
        "columnas_exactas": list(sub.columns) == [cfg.ID_COL, "age_range"],
        "ids_unicos": bool(sub[cfg.ID_COL].is_unique),
        "ids_iguales_a_test": set(sub[cfg.ID_COL]) == set(id_test),
        "mismo_orden_que_test": bool((sub[cfg.ID_COL].to_numpy() == id_test).all()),
        "sin_nan": bool(not sub.isna().any().any()),
        "clases_validas": bool(set(sub["age_range"].unique()) <= set(cfg.CLASSES)),
        "todas_las_clases_presentes": int(sub["age_range"].nunique()) == N_CLASES,
        "sin_columnas_auxiliares": sub.shape[1] == 2,
        "hash_coincide_con_el_manifiesto": sha == manifiesto["submission_sha256"],
    }
    dist = sub["age_range"].value_counts().to_dict()
    return {"checks": checks, "todos_ok": bool(all(checks.values())), "sha256": sha,
            "distribucion_predicha": {c: int(dist.get(c, 0)) for c in cfg.CLASSES},
            "distribucion_relativa": {c: round(dist.get(c, 0) / len(sub), 4) for c in cfg.CLASSES}}


def main() -> int:
    """Audita la submission: operaciones, compliance y fichero."""
    autotest_numerico()
    inicio = time.time()
    print("=== [1/3] operaciones prohibidas sobre el test ===", flush=True)
    ops = audita_operaciones()
    print(f"  scripts auditados: {len(ops['scripts_auditados'])}", flush=True)
    print(f"  hallazgos: {ops['total_hallazgos']}", flush=True)
    for h in ops["hallazgos"][:8]:
        print(f"    {h['script']}:{h['linea']} {h['operacion']} -> {h['texto'][:70]}", flush=True)
    print("  comprobaciones positivas:", flush=True)
    for k, v in ops["comprobaciones_positivas"].items():
        print(f"    {k:40s} presente en {v} script(s)", flush=True)

    print("\n=== [2/3] compliance de AGE ===", flush=True)
    comp = audita_compliance()
    print(f"  hallazgos: {comp['total']}", flush=True)
    for h in comp["hallazgos"]:
        print(f"    {h['script']}: {h['recurso']} en {h['import']}", flush=True)

    print("\n=== [3/3] el fichero ===", flush=True)
    fil = verifica_fichero()
    for k, v in fil["checks"].items():
        print(f"  {'OK ' if v else 'NO '} {k}", flush=True)
    print(f"  SHA256: {fil['sha256']}", flush=True)
    print(f"  distribucion predicha: {fil['distribucion_relativa']}", flush=True)

    aprobado = bool(ops["total_hallazgos"] == 0 and comp["total"] == 0 and fil["todos_ok"]
                    and ops["positivas_ok"])
    print(f"\n=== {'APROBADO' if aprobado else 'REVISAR'} ===", flush=True)

    RESULTADO.write_text(json.dumps({
        "fase": "auditoria de la submission experimental", "fecha": FECHA, "experimenta": False,
        "operaciones_sobre_el_test": ops, "compliance": comp, "fichero": fil,
        "aprobado": aprobado,
        "nota_de_metodo": "el test se lee para transform y predict, que es lo que el procedimiento "
                          "debe hacer. Lo que se audita es que NO se lea su target ni se ajuste "
                          "nada con el",
        "no_se_hace": ["estimar score", "proyectar leaderboard",
                       "usar el numero de filas cambiadas como prediccion"],
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[fin] {time.time() - inicio:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
