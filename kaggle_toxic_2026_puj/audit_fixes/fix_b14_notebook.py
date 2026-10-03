"""B-14: corrige y re-ejecuta el notebook consolidado.

- Actualiza los recuentos (15 sondas; 207 filas de registro) y la seccion 11.
- Anade las 3 sondas de v8 a las tablas de descripciones/decisiones.
- Anade la seccion 19: fase v9 y cierre post-auditoria, con verificacion.
- Ejecuta el notebook completo (nbconvert --execute --inplace) y registra el
  resultado en reports/notebook_ejecucion_v9.json.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

PROJ = Path("/home/carlos/Git/PLN-v8/kaggle_toxic_2026_puj")
NB_PATH = PROJ / "notebooks" / "10_final_pipeline.ipynb"
REGISTRO_EJECUCION = PROJ / "reports" / "notebook_ejecucion_v9.json"
JUPYTER = "/home/carlos/venv/bin/jupyter"


def reemplazar(texto: str, viejo: str, nuevo: str, etiqueta: str) -> str:
    """Reemplazo con verificacion de unicidad."""
    assert viejo in texto, f"no encontrado: {etiqueta}"
    assert texto.count(viejo) == 1, f"multiple: {etiqueta}"
    return texto.replace(viejo, nuevo)


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    celdas = nb["cells"]

    # --- celda 0: cabecera ---
    src = "".join(celdas[0]["source"])
    src = reemplazar(src, "F1 público **0.78306** (v03, mejor de 12 sondeos)",
                     "F1 público **0.78306** (v03, mejor de 15 sondeos)", "celda 0 cabecera")
    src = reemplazar(
        src,
        "| **Alcance** | Dos fases: pipeline clásico (fases 0-14) y Advanced Research v2 (paquetes P0-P7, 24 líneas de investigación) |",
        "| **Alcance** | Pipeline clásico (fases 0-14), Advanced Research v2 (P0-P7), fases v4/v5/v5.1, "
        "v6-v8 y fase v9 guiada por la auditoría forense (2026-09-26) |", "celda 0 alcance")
    celdas[0]["source"] = src.splitlines(keepends=True)

    # --- celda 29: prosa de las sondas ---
    src = "".join(celdas[29]["source"])
    src = reemplazar(src, "## 11. Las 12 sondas de leaderboard",
                     "## 11. Las 15 sondas de leaderboard", "celda 29 titulo")
    viejo_obs = ("El proyecto gastó 12 envíos en el leaderboard público: 1 sonda de metrología "
                 "(E00), 1 baseline, 5 variantes de la familia unión de la fase 1, 2 del ciclo v2 "
                 "(v03 revalidada y el meta) y 3 de la fase final de candidatos (metas "
                 "restringidos sobre V03 + canales). El máximo observado es 0.78306 (v03).")
    nuevo_obs = ("El proyecto gastó 15 envíos en el leaderboard público: 1 sonda de metrología "
                 "(E00), 1 baseline, 5 variantes de la familia unión de la fase 1, 2 del ciclo v2 "
                 "(v03 revalidada y el meta), 3 de la fase final de candidatos (metas restringidos "
                 "sobre V03 + canales) y 3 de la fase v8 (promedio de pliegues, C por etiqueta y "
                 "promedio de repeticiones). El máximo observado sigue siendo 0.78306 (v03): "
                 "ninguna sonda posterior lo superó.")
    src = reemplazar(src, viejo_obs, nuevo_obs, "celda 29 observacion")
    src = reemplazar(src, "El máximo de 12 mediciones ruidosas",
                     "El máximo de 15 mediciones ruidosas", "celda 29 hipotesis")
    celdas[29]["source"] = src.splitlines(keepends=True)

    # --- celda 30: tabla de sondas ---
    src = "".join(celdas[30]["source"])
    src = reemplazar(src, "Tabla completa de las 12 sondas", "Tabla completa de las 15 sondas",
                     "celda 30 docstring")
    ancla_desc = '    "v11_candidate_c": "C-EMO-SYN: V03 + emocion + sintaxis (meta restringido)",\n'
    nuevas_desc = (ancla_desc
                   + '    "v08_p1_fold_ensemble": "Promedio de modelos por pliegue (fase v8, P1)",\n'
                   + '    "v09_p5_c_por_etiqueta": "C por etiqueta con correccion (fase v8, P5)",\n'
                   + '    "v12_p7_promedio_repeticiones": "Promedio de repeticiones (fase v8, P7)",\n')
    src = reemplazar(src, ancla_desc, nuevas_desc, "celda 30 descripciones")
    ancla_dec = ('    "v11_candidate_c": "Sonda enviada: LB 0.78150 (no supera V03, D-024)",\n')
    nuevas_dec = (ancla_dec
                  + '    "v08_p1_fold_ensemble": "Sonda v8 enviada: LB 0.77470 (no supera V03)",\n'
                  + '    "v09_p5_c_por_etiqueta": "Sonda v8 enviada: LB 0.77781 (no supera V03)",\n'
                  + '    "v12_p7_promedio_repeticiones": "Sonda v8 enviada: LB 0.77122 (no supera V03)",\n')
    src = reemplazar(src, ancla_dec, nuevas_dec, "celda 30 decisiones")
    src = reemplazar(
        src,
        'tabla["gap_recalculado"] = (tabla["kaggle_f1"] - tabla["local_f1"]).round(5)',
        'tabla["gap_recalculado"] = (tabla["kaggle_f1"] - tabla["local_f1"]).round(5)\n'
        'tabla["definicion"] = leaderboard["definicion_diferencia"]  # B-12: semantica declarada del log',
        "celda 30 gap")
    src = reemplazar(src, "LAS 12 SONDAS DE LEADERBOARD", "LAS 15 SONDAS DE LEADERBOARD",
                     "celda 30 print")
    src = reemplazar(src, 'assert len(tabla) == 12, "El proyecto declara 12 sondas de leaderboard"',
                     'assert len(tabla) == 15, "El proyecto declara 15 sondas de leaderboard"\n'
                     'assert float(tabla["kaggle_f1"].max()) == 0.78306\n'
                     'assert tabla.loc[tabla["kaggle_f1"].idxmax(), "submission_id"] == "v03_lr_union_thr"',
                     "celda 30 assert")
    src = reemplazar(
        src,
        'familia_union = familia[familia["submission_id"].str.contains("v0[2-8]")]',
        'NOMBRES_UNION = ["v02_lr_union", "v03_lr_union_thr", "v04_ens04_mediana", "v05_optuna",\n'
        '                 "v06_robustez_c1", "v07_robustez_mindf5", "v08_meta_v2"]\n'
        'familia_union = familia[familia["submission_id"].isin(NOMBRES_UNION)]',
        "celda 30 familia")
    celdas[30]["source"] = src.splitlines(keepends=True)

    # --- celda 38: verificacion de numeros clave ---
    src = "".join(celdas[38]["source"])
    src = reemplazar(src, "(143 experimentos, 12 sondas, 39 scripts)",
                     "(207 filas de registro, 15 sondas, 39 scripts)", "celda 38 docstring")
    src = reemplazar(src, 'verificar("filas de experiments.csv", len(experimentos), 174)',
                     'verificar("filas de experiments.csv", len(experimentos), 207)', "celda 38 filas")
    src = reemplazar(src, 'verificar("sondas de leaderboard", len(sondas), 12)',
                     'verificar("sondas de leaderboard", len(sondas), 15)', "celda 38 sondas")
    celdas[38]["source"] = src.splitlines(keepends=True)

    # --- seccion 19: fase v9 y cierre post-auditoria ---
    markdown_19 = (
        "## 19. Fase v9 y cierre post-auditoria forense (2026-09-26)\n\n"
        "### OBSERVACIÓN\n\n"
        "Tras la fase v8, una auditoría forense completa verificó integridad, fugas, "
        "reproducibilidad, documentación y notebook (rama `proyecto-kaggle-toxic-2026-audit`). "
        "Hallazgos principales: V03 reproducible exacto (0.777211) y submissions v01-v03 "
        "regeneradas byte a byte; dos mecanismos de información con impacto medido nulo o "
        "marginal (near-duplicados train+test en los grupos: 0.0; pesos de v6 Track F: "
        "+0.000142 -> 0 con versión fold-local); registro reparado (EBD-01..06); notebook "
        "corregido; 0 secretos y 0 emojis. Con esa base se abrió la fase v9 (H1-H5).\n\n"
        "### HIPÓTESIS\n\n"
        "H5: la fuga estructural no cambia V03. H4: el test tiene ~7 % menos positivos (E00) "
        "y una corrección de prior por EM recupera parte del gap. H2: los pares casi idénticos "
        "train-test permiten propagar etiquetas. H1: un mecanismo secuencial sobre los "
        "embeddings propios supera al pooling. H3: una compresión supervisada supera a SVD.\n\n"
        "### RESULTADO\n\n"
        "| experimento | resultado | veredicto |\n"
        "|---|---|---|\n"
        "| V9-01 (H5, sensibilidad de fugas) | V03 con grupos solo-train = 0.7772114525 "
        "(delta 0.0; ARI 1.0) | REPRODUCE EXACTO: la fuga no cambia nada |\n"
        "| V9-02 (H4, prior shift EM) | EM test: HS 0.4092 / TR 0.2865 / AG 0.3587 "
        "(factores 0.936/0.955/0.938, coherentes con E00); ganancia anidada media +0.000574 "
        "(HS +0.0028; TR -0.0008; AG -0.0003) | mecanismo real pero pequeño; sonda v17 "
        "pre-registrada (62 celdas) |\n"
        "| V9-03 (H2, near-duplicados) | 8 pares >= 0.85 en 4 tuits de test; la propagación "
        "deja la submission idéntica a V03 (0 celdas) | sin efecto accionable; sonda v18 |\n"
        "| V9-04 (H1, secuencial) | BiGRU con embeddings congelados 0.6943 vs media+LR 0.6636 "
        "vs V03 0.7772 | el secuencial mejora al pooling (+0.031) pero no a V03 |\n"
        "| V9-05 (H3, compresión supervisada) | direcciones discriminantes 0.7066 vs SVD(6) "
        "0.6531 vs V03 0.7772 | la supervisión ayuda frente a su control, no frente a V03 |\n\n"
        "### DECISIÓN\n\n"
        "V03 se mantiene como modelo final y como la mejor submission (0.78306). Las sondas "
        "v17/v18 quedan generadas y pre-registradas, sin enviar. El cuello de botella sigue "
        "siendo de distribución/calibración, no de representación.\n")

    code_19 = (
        '"""Verificación de la fase v9 y del cierre post-auditoría.\n\n'
        "Comprueba el registro reparado (207 filas, EBD-01..06 y V9-01..05), las 15 sondas,\n"
        "los informes de la fase v9 y la integridad de las sondas v17/v18 (generadas, no\n"
        "enviadas).\n"
        '"""\n\n'
        "import json\n\n"
        "import pandas as pd\n\n"
        "registro = pd.read_csv(PROJECT_ROOT / \"experiments.csv\")\n"
        "assert len(registro) == 207, \"el registro debe tener 207 filas tras la reparacion\"\n"
        "assert registro[\"experiment_id\"].nunique() == 207, \"los ids deben ser unicos\"\n"
        "ids = set(registro[\"experiment_id\"])\n"
        "for esperado in (\"EBD-01\", \"EBD-02\", \"EBD-03\", \"EBD-04\", \"EBD-05\", \"EBD-06\",\n"
        "                 \"V9-01\", \"V9-02\", \"V9-03\", \"V9-04\", \"V9-05\"):\n"
        "    assert esperado in ids, f\"falta la fila {esperado}\"\n\n"
        "sondas = pd.read_csv(RUTAS[\"submissions\"] / \"leaderboard_log.csv\")\n"
        "assert len(sondas) == 15\n"
        "assert \"definicion_diferencia\" in sondas.columns, \"el log debe declarar la semantica de diferencia\"\n\n"
        "v9_01 = json.loads((RUTAS[\"reports\"] / \"v9_01_sensibilidad_fugas.json\").read_text(encoding=\"utf-8\"))\n"
        "assert abs(v9_01[\"delta_preciso_vs_canon\"]) < 1e-12\n\n"
        "v9_02 = json.loads((RUTAS[\"reports\"] / \"v9_02_prior_shift_em.json\").read_text(encoding=\"utf-8\"))\n"
        "assert set(v9_02[\"em_por_etiqueta\"]) == {\"HS\", \"TR\", \"AG\"}\n"
        "assert 0.85 < min(d[\"factor_vs_train\"] for d in v9_02[\"em_por_etiqueta\"].values()) < 1.0\n\n"
        "v9_04 = json.loads((RUTAS[\"reports\"] / \"v9_04_dl_secuencial_fasttext.json\").read_text(encoding=\"utf-8\"))\n"
        "v9_05 = json.loads((RUTAS[\"reports\"] / \"v9_05_proyecciones_supervisadas.json\").read_text(encoding=\"utf-8\"))\n\n"
        "for sonda in (\"toxic_submission_v17_prior_shift_em.csv\",\n"
        "              \"toxic_submission_v18_near_dup_propagacion.csv\"):\n"
        "    marco = pd.read_csv(RUTAS[\"submissions\"] / sonda)\n"
        "    assert len(marco) == 1154 and marco[\"id\"].is_unique\n"
        "    assert set(marco[[\"HS\", \"TR\", \"AG\"]].to_numpy().ravel()) <= {0, 1}\n\n"
        "print(f\"V9-04 BiGRU {v9_04['gru_umbrales_v03']:.4f} vs media {v9_04['media_lr_umbrales_v03']:.4f} \"\n"
        "      f\"| V9-05 SDR {v9_05['sdr_6_features_umbrales_v03']:.4f} vs SVD {v9_05['svd_6_componentes_umbrales_v03']:.4f}\")\n"
        "print(\"V9-02 EM:\", {e: d[\"prevalencia_em_test\"] for e, d in v9_02[\"em_por_etiqueta\"].items()})\n"
        "print(\"Sondas v17/v18 generadas y verificadas (no enviadas).\")\n"
        "print(\"Todas las verificaciones de la fase v9 pasaron.\")\n")

    celdas.append({"cell_type": "markdown", "metadata": {}, "source": markdown_19.splitlines(keepends=True)})
    celdas.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                   "source": code_19.splitlines(keepends=True)})

    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("notebook parcheado; ejecutando completo...")

    inicio = time.perf_counter()
    proc = subprocess.run(
        [JUPYTER, "nbconvert", "--to", "notebook", "--execute", "--inplace",
         "notebooks/10_final_pipeline.ipynb", "--ExecutePreprocessor.timeout=1800"],
        cwd=str(PROJ), capture_output=True, text=True)
    duracion = round(time.perf_counter() - inicio, 1)

    nb2 = json.loads(NB_PATH.read_text(encoding="utf-8"))
    errores = []
    celdas_codigo = 0
    for i, c in enumerate(nb2["cells"]):
        if c["cell_type"] != "code":
            continue
        celdas_codigo += 1
        for salida in c.get("outputs", []):
            if salida.get("output_type") == "error":
                errores.append({"celda": i, "ename": salida.get("ename"),
                                "evalue": salida.get("evalue", "")[:200]})
    registro = {
        "fecha": "2026-09-26",
        "notebook": "notebooks/10_final_pipeline.ipynb",
        "celdas_totales": len(nb2["cells"]),
        "celdas_codigo": celdas_codigo,
        "returncode": proc.returncode,
        "errores": errores,
        "duracion_s": duracion,
        "stderr_tail": proc.stderr[-600:],
    }
    REGISTRO_EJECUCION.write_text(json.dumps(registro, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in registro.items() if k != "stderr_tail"}, indent=2, ensure_ascii=False))
    if errores or proc.returncode != 0:
        raise SystemExit("la ejecucion del notebook fallo")


if __name__ == "__main__":
    main()
