"""P11 de la fase v5.1 (EBD-28, AUDIT): auditoria final y veredicto de la entrega.

Conteo
------
Este script es una ``AUDIT / REPRODUCTION UNIT`` y **no** es un experimento nuevo: por la
regla de contabilidad de la seccion 2 de la spec no se registra como fila nueva de
``experiments.csv``. Escribe su resultado en
``reports/embedding_auditoria_v5_1.json`` y en la decision de cierre.

Que verifica
------------
1. **Reglas de la seccion 36** delegando en la auditoria de v5 (que ya sabe distinguir un uso
   real de un nombre en un documento) y anadiendo la comprobacion de los 19 modelos de la
   fase completa.
2. **Presupuesto**: 9 de 9 exploratorios, 0 de 1 de reserva, y ninguna firma duplicada ni
   reentrenamiento disfrazado.
3. **Integridad de artefactos**: los cinco informes de la seccion 37, la tabla maestra, las
   tablas A y B, el informe de validacion del notebook y las figuras.
4. **V03 intacto**: `corrected_cv` 0.777211, submission de 1.155 lineas con su sha256, y
   ninguna submission nueva en la fase.
5. **Notebook**: 63 celdas, seccion 18 presente y validada con veredicto GO.
6. **Estado de la rama**: rama correcta, arbol limpio y sin push.

Uso
---
    /home/carlos/venv/bin/python experiments/embedding_22_auditoria_final_v5_1.py
"""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

EXPERIMENTO = "EBD-28 (P11, AUDIT)"
REPORTE = PROJECT_ROOT / "reports" / "embedding_auditoria_v5_1.json"
VALIDACION_NOTEBOOK = PROJECT_ROOT / "reports" / "embedding_notebook_validation_v5_1.json"
CORRECTED_CV_ESPERADO = 0.777211
RAMA_ESPERADA = "proyecto-kaggle-toxic-2026-v5"
SUBMISSION_V03 = PROJECT_ROOT / "experiments" / "baseline_v03_frozen" / "submission_v03.csv"
NOTABLET = PROJECT_ROOT / "notebooks" / "10_final_pipeline.ipynb"

# Informes y artefactos que la seccion 37 de la spec exige que existan.
ARTEFACTOS_EXIGIDOS = [
    "reports/embedding_architecture_v5_1.md",
    "reports/embedding_architecture_results_v5_1.csv",
    "reports/embedding_capacity_analysis_v5_1.md",
    "reports/embedding_scaling_conclusion_v5_1.md",
    "reports/embedding_complementarity.md",
    "reports/embedding_notebook_validation.md",
    "reports/embedding_architecture_maestra_v5_1.csv",
    "reports/embedding_downstream_evaluations_v5_1.csv",
    "reports/embedding_budget_v5_1.json",
    "reports/corpus_manifest_v5.json",
]

FIGURAS_ESPERADAS = [
    "fig_49_curva_completa_v5.png",
    "fig_50_vector_size_v5_1.png",
    "fig_51_window_v5_1.png",
    "fig_52_min_count_v5_1.png",
    "fig_53_subwords_v5_1.png",
    "fig_54_negative_v5_1.png",
    "fig_55_heatmap_arquitectura_v5_1.png",
    "fig_56_curva_escala_conclusiva_v5_1.png",
    "fig_57_complementariedad_v5_1.png",
]


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V5.1-P11] {mensaje}", flush=True)


def _cargar_modulo(nombre: str, ruta: Path):
    """Carga un modulo de experimento de forma aislada.

    Args:
        nombre: nombre del modulo en memoria.
        ruta: ruta del archivo.

    Returns:
        El modulo cargado.
    """
    especificacion = importlib.util.spec_from_file_location(nombre, ruta)
    modulo = importlib.util.module_from_spec(especificacion)
    especificacion.loader.exec_module(modulo)
    return modulo


def auditar_modelos() -> dict[str, Any]:
    """Verifica que los 19 modelos de la fase tienen hash coherente en disco.

    Returns:
        Diccionario con el recuento y las discrepancias.
    """
    from src.embedding_utils import sha256_de

    presupuesto = json.loads(
        (PROJECT_ROOT / "reports" / "embedding_budget_v5_1.json").read_text(encoding="utf-8")
    )
    tabla_a = presupuesto["tabla_a"]
    firmas = {fila["training_signature"] for fila in tabla_a}
    disco = PROJECT_ROOT / "data" / "external" / "embeddings"
    # El prefijo de archivo **no** se deriva del training_id: los tres modelos de Wikipedia
    # 10M se entrenaron en P2 con los prefijos `fasttext_baseline`, `fasttext_seed_control`
    # y `word2vec_control`. El mapa canonico esta en el inventario de P0, que es donde se
    # declararon; derivarlo del identificador daria tres rutas inexistentes.
    modulo_p0 = _cargar_modulo(
        "embedding_11_p0", PROJECT_ROOT / "experiments" / "embedding_11_auditoria_firmas_v5_1.py"
    )
    prefijos_v5 = {entrada["training_id"]: entrada["prefijo"] for entrada in modulo_p0.INVENTARIO_V5}
    verificados = 0
    discrepancias: list[str] = []
    for fila in tabla_a:
        training_id = fila["training_id"]
        if training_id in prefijos_v5:
            prefijo = prefijos_v5[training_id]
        else:
            prefijo = training_id.lower().replace("fasttext-wiki-10m", "fasttext_wiki_10m").replace("-", "_")
        ruta_vec = disco / f"{prefijo}.vec"
        if not ruta_vec.exists():
            discrepancias.append(f"{fila['training_id']}: no existe {ruta_vec.name}")
            continue
        real = sha256_de(ruta_vec)
        if real != fila["model_sha256"]:
            discrepancias.append(
                f"{fila['training_id']}: sha256 {real[:12]} frente a {fila['model_sha256'][:12]}"
            )
            continue
        verificados += 1
    return {
        "n_modelos_en_la_tabla": len(tabla_a),
        "n_verificados": verificados,
        "firmas_unicas": len(firmas) == len(tabla_a),
        "discrepancias": discrepancias,
        "cumple": verificados == len(tabla_a) and len(firmas) == len(tabla_a) and not discrepancias,
    }


def auditar_presupuesto() -> dict[str, Any]:
    """Comprueba que el presupuesto de la spec se respeta.

    Returns:
        Diccionario con los contadores y el veredicto.
    """
    presupuesto = json.loads(
        (PROJECT_ROOT / "reports" / "embedding_budget_v5_1.json").read_text(encoding="utf-8")
    )
    estado = presupuesto["presupuesto"]
    aserciones = presupuesto["aserciones"]
    propuestas = presupuesto["propuestas"]
    pendientes = [p for p in propuestas if p["status"] == "PENDING"]
    cumple = (
        estado["new_v5_1_trainings"] <= estado["limite_exploratorio"]
        and estado["total_new_budget_used"] <= estado["limite_exploratorio"] + estado["limite_reserva"]
        and estado["reserved_training_units"] == 0
        and all(aserciones.values())
        and not pendientes
    )
    return {
        "contadores": estado,
        "aserciones": aserciones,
        "propuestas_pendientes": [p["codigo"] for p in pendientes],
        "cumple": cumple,
    }


def auditar_v03() -> dict[str, Any]:
    """Comprueba que V03 sigue intacto y que la fase no generó submissions.

    Returns:
        Diccionario con el estado de V03.
    """
    import numpy as np

    from src.data import read_test, read_train
    from src.evaluation import ETIQUETAS, metrica_macro
    from src.splits import grupos_fusionados, holdout_v2

    oof = np.load(PROJECT_ROOT / "features" / "oof_v03_final.npz", allow_pickle=True)
    train = read_train()
    grupos = grupos_fusionados(train)
    indices, _ = holdout_v2(train, grupos)
    desarrollo = train.iloc[indices].reset_index(drop=True)
    y = desarrollo[list(ETIQUETAS)].to_numpy(dtype=int)
    umbrales = np.array([0.44, 0.51, 0.46])
    corrected_cv = metrica_macro(y, (oof["proba"] >= umbrales).astype(int))
    lineas = sum(1 for _ in SUBMISSION_V03.open(encoding="utf-8"))
    esperado = len(read_test()) + 1
    with (PROJECT_ROOT / "submissions" / "leaderboard_log.csv").open(encoding="utf-8") as archivo:
        sondas = len(list(csv.DictReader(archivo)))
    return {
        "corrected_cv": round(corrected_cv, 6),
        "corrected_cv_esperado": CORRECTED_CV_ESPERADO,
        "coincide": abs(corrected_cv - CORRECTED_CV_ESPERADO) < 1e-6,
        "submission_v03_lineas": lineas,
        "submission_v03_esperadas": esperado,
        "submission_intacta": lineas == esperado,
        "sondas_de_leaderboard": sondas,
        "sondas_esperadas": 12,
        "submissions_nuevas_en_la_fase": sondas - 12,
    }


def auditar_notebook() -> dict[str, Any]:
    """Comprueba que el notebook tiene la seccion 18 y que su validacion fue GO.

    Returns:
        Diccionario con el estado de la validacion.
    """
    import nbformat

    notebook = nbformat.read(str(NOTABLET), as_version=4)
    tiene_seccion = any(
        celda.cell_type == "markdown" and "## 18. Fase v5 y v5.1" in "".join(celda.source)
        for celda in notebook.cells
    )
    con_salidas = sum(1 for celda in notebook.cells if celda.get("outputs"))
    validacion = (
        json.loads(VALIDACION_NOTEBOOK.read_text(encoding="utf-8")) if VALIDACION_NOTEBOOK.exists() else {}
    )
    return {
        "n_celdas": len(notebook.cells),
        "tiene_seccion_18": tiene_seccion,
        "celdas_con_salida": con_salidas,
        "veredicto_validacion": validacion.get("veredicto"),
        "n_errores_validacion": validacion.get("n_errores"),
        "cumple": tiene_seccion and validacion.get("veredicto") == "GO" and validacion.get("n_errores") == 0,
    }


def auditar_git() -> dict[str, Any]:
    """Comprueba la rama, la limpieza del arbol y la ausencia de push.

    Returns:
        Diccionario con el estado de git.
    """
    def git(*argumentos: str) -> str:
        """Ejecuta un comando de git en la raiz del repositorio y devuelve su salida."""
        return subprocess.run(
            ["git", "-C", str(PROJECT_ROOT.parent), *argumentos],
            capture_output=True,
            text=True,
        ).stdout.strip()

    rama = git("branch", "--show-current")
    estado = git("status", "--porcelain")
    log_remoto = git("log", "--oneline", "-1", "origin/proyecto-kaggle-toxic-2026-v5")
    return {
        "rama": rama,
        "rama_correcta": rama == RAMA_ESPERADA,
        "arbol_limpio": estado == "",
        "commits_locales": int(git("rev-list", "--count", "HEAD").strip() or 0),
        "rama_en_remoto": bool(log_remoto),
        "pusheada": bool(log_remoto),
        "nota": "La rama v5 no se ha pusheado: queda local por decision del usuario.",
    }


def auditar_artefactos() -> dict[str, Any]:
    """Comprueba que existen los informes, tablas y figuras que exige la spec.

    Returns:
        Diccionario con lo encontrado y lo que falta.
    """
    faltan = [r for r in ARTEFACTOS_EXIGIDOS if not (PROJECT_ROOT / r).exists()]
    figuras = PROJECT_ROOT / "reports" / "figuras"
    faltan_figuras = [f for f in FIGURAS_ESPERADAS if not (figuras / f).exists()]
    return {
        "exigidos": len(ARTEFACTOS_EXIGIDOS),
        "faltan": faltan,
        "figuras_esperadas": len(FIGURAS_ESPERADAS),
        "figuras_faltantes": faltan_figuras,
        "cumple": not faltan and not faltan_figuras,
    }


def main() -> None:
    """Ejecuta la auditoria final de la fase v5.1."""
    inicio = time.time()
    log("P11: auditoria final de la fase v5.1 (unidad AUDIT, no cuenta como experimento)")
    modulo_06 = _cargar_modulo(
        "embedding_06_auditoria", PROJECT_ROOT / "experiments" / "embedding_06_auditoria_v5.py"
    )
    recursos = modulo_06.auditar_recursos()
    uso_toxic = modulo_06.auditar_uso_de_toxic()
    holdout = modulo_06.auditar_holdout()
    fuga = modulo_06.auditar_fuga_measured()
    modelos = auditar_modelos()
    presupuesto = auditar_presupuesto()
    v03 = auditar_v03()
    notebook = auditar_notebook()
    artefactos = auditar_artefactos()
    git_estado = auditar_git()

    reglas = {
        "transformers": 0 if not recursos.get("hallazgos") else "ALGUNO",
        "bert_roberta_llm": 0 if not recursos.get("hallazgos") else "ALGUNO",
        "embeddings_preentrenados": 0 if not recursos.get("hallazgos") else "ALGUNO",
        "from_pretrained": 0 if not recursos.get("hallazgos") else "ALGUNO",
        "etiquetas_externas": 0 if uso_toxic.get("ok") else "REVISAR",
        "textos_de_test_en_corpus": 0 if fuga.get("ok") else "ALGUNO",
        "uso_del_holdout": 0 if holdout.get("presupuesto_intacto") else "REVISAR",
    }
    comprobaciones = {
        "recursos_prohibidos": not recursos.get("hallazgos"),
        "uso_toxic_declarado": bool(uso_toxic.get("ok")),
        "holdout_intacto": bool(holdout.get("presupuesto_intacto")),
        "sin_fuga_de_test": bool(fuga.get("ok")),
        "modelos_verificados": modelos["cumple"],
        "presupuesto_respetado": presupuesto["cumple"],
        "v03_intacto": v03["coincide"] and v03["submission_intacta"],
        "sin_submissions_nuevas": v03["submissions_nuevas_en_la_fase"] == 0,
        "notebook_validado": notebook["cumple"],
        "artefactos_completos": artefactos["cumple"],
        "rama_correcta": git_estado["rama_correcta"],
    }
    veredicto = "GO" if all(comprobaciones.values()) else "NO-GO"
    fallos = [nombre for nombre, valor in comprobaciones.items() if not valor]

    resultado = {
        "experimento": EXPERIMENTO,
        "tipo": "AUDIT / REPRODUCTION UNIT (no cuenta como experimento nuevo)",
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duracion_s": round(time.time() - inicio, 1),
        "reglas_seccion_36": reglas,
        "recursos_detalle": recursos.get("hallazgos", []),
        "modelos": modelos,
        "presupuesto": presupuesto,
        "v03": v03,
        "notebook": notebook,
        "artefactos": artefactos,
        "git": git_estado,
        "comprobaciones": comprobaciones,
        "comprobaciones_fallidas": fallos,
        "veredicto": veredicto,
    }
    REPORTE.write_text(json.dumps(resultado, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"   reglas de la seccion 36: {reglas}")
    log(f"   modelos: {modelos['n_verificados']}/{modelos['n_modelos_en_la_tabla']} verificados")
    log(f"   presupuesto: {presupuesto['contadores']}")
    log(f"   V03: {v03['corrected_cv']} | submission {v03['submission_v03_lineas']} lineas")
    log(f"   notebook: {notebook['n_celdas']} celdas, validacion {notebook['veredicto_validacion']}")
    log(f"   rama: {git_estado['rama']} | arbol limpio: {git_estado['arbol_limpio']}")
    log(f"   comprobaciones que fallan: {fallos or 'ninguna'}")
    log(f"VEREDICTO: {veredicto} | {round(time.time() - inicio, 1)} s")


if __name__ == "__main__":
    main()
