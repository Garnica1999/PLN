"""Fase de correcciones B-02..B-12 en el proyecto (v8).

- B-03: rutas duplicadas en v6_reproducibility.md
- B-04: '2/5' del holdout -> 5/5 en target_encoding_audit_v2.md
- B-05: conclusiones pendientes en DECISIONS.md (8)
- B-06/B-07/B-10: notas de auditoria en scripts (sin cambiar logica)
- B-12: columna 'definicion_diferencia' en leaderboard_log.csv (aditiva)
- B-02: documenta los huecos EBD-11/12 en RUNLOG
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

PROJ = Path("/home/carlos/Git/PLN-v8/kaggle_toxic_2026_puj")

CONCLUSIONES = {
    "D-001": ("formulacion multi-etiqueta adoptada; el post-proceso jerarquico se midio en las "
              "Fases 4-5 y se descarto (empeora -0.0056); el modelo final (V03) no lo aplica."),
    "D-003": ("se mantuvo sin excepciones: spaCy se uso solo como anotador simbolico "
              "(POS/dependencias) y las features no incorporaron vectores preentrenados; la "
              "auditoria forense lo verifico (P4)."),
    "D-004": ("superada por D-015/D-016: agrupar solo por near-duplicado daba una estimacion "
              "optimista (0.7927 frente a 0.7838 al anadir pseudo-autor); el protocolo vigente "
              "usa grupos fusionados."),
    "D-005": ("caracterizacion confirmada con los datos finales (519/519 TR=>HS; 26 casos AG=1 "
              "con HS=0) y usada como base de la jerarquia y de los canales TR/AG."),
    "D-007": ("cumplido: no se uso ningun recurso externo etiquetado ni lexicones externos "
              "(pysentimiento/VADER-es descartados por prohibicion); el unico corpus externo fue "
              "no etiquetado y solo para embeddings (D-030)."),
    "D-008": ("protocolo aplicado en la fase 11 (preflight rocm-smi y limite de uso); el resto "
              "del proyecto es CPU; sin incidentes."),
    "D-010": ("verificado (train sin bytes no ASCII; test con 6 bytes cp1252); las variantes de "
              "normalizacion se documentaron en v2 y no se intento reconstruir diacriticos."),
    "D-017": ("cadenas cerradas sin mejora (0/5); el meta de dependencia no se adopto como modelo "
              "final (D-019 superada por D-024); los canales derivados quedan disponibles para "
              "metas restringidos."),
}

NOTAS_SCRIPTS = {
    "experiments/v6_07_ruido_par.py": (
        "# Nota de auditoria forense (B-06, 2026-09-26): los pesos de sospecha se calculan con\n"
        "# etiquetas de TODO el desarrollo, incluidas las del pliegue evaluado (mecanismo no\n"
        "# fold-local). Contrafactual medido: delta +0.000142 -> 0.000000 con deteccion\n"
        "# fold-local (audit/p04b_suspicion_foldlocal.json). Si se reabre la familia, calcular\n"
        "# densidad/estructural por pliegue con etiquetas solo del entrenamiento. No se modifico\n"
        "# la logica historica.\n"),
    "experiments/00_auditoria_datos.py": (
        "# Nota de auditoria forense (B-07, 2026-09-26): el clustering de near-duplicados se\n"
        "# ajusta sobre train+test (2885 textos). Impacto medido sobre las particiones: 0.0\n"
        "# (ARI=1.0; holdout y pliegues identicos). Si se repara, clusterizar solo con train y\n"
        "# regenerar data/processed/grupos_near_duplicados.csv. No se modifico la logica\n"
        "# historica.\n"),
    "experiments/23_candidatos_finales.py": (
        "# Nota de auditoria forense (B-10, 2026-09-26): la columna delta_vs_v03 de\n"
        "# final_candidates.csv se calculo contra la referencia redondeada 0.7772 (desvios de\n"
        "# hasta 1.1e-5 frente a la canonica 0.777211). En futuras corridas usar 0.777211.\n"
        "# No se modifico la logica historica.\n"),
}


def insertar_nota(ruta: Path, nota: str) -> bool:
    texto = ruta.read_text(encoding="utf-8")
    if "Nota de auditoria forense" in texto:
        return False
    ancla = "from __future__ import annotations"
    if ancla in texto:
        texto = texto.replace(ancla, nota + ancla, 1)
    else:
        texto = nota + texto
    ruta.write_text(texto, encoding="utf-8")
    return True


def main() -> None:
    cambios = []

    # B-03
    p = PROJ / "reports" / "v6_reproducibility.md"
    t = p.read_text(encoding="utf-8")
    n = t.count("experiments/experiments/")
    if n:
        p.write_text(t.replace("experiments/experiments/", "experiments/"), encoding="utf-8")
        cambios.append(f"B-03: {n} rutas corregidas en v6_reproducibility.md")

    # B-04
    p = PROJ / "reports" / "target_encoding_audit_v2.md"
    t = p.read_text(encoding="utf-8")
    viejo = "solo 2 de 5 evaluaciones usadas: P7-01 (V03, 0.8285) y P7-02 (meta, 0.8465); ninguna decisión de desarrollo lo consultó."
    if viejo in t:
        nuevo = ("5 de 5 evaluaciones usadas en total (P7-01 V03 0.8285; P7-02 meta 0.8465; "
                 "FIN-02 0.8364; FIN-04 0.8336; P8-C-meta36 0.8163, rechazado); todas con "
                 "umbrales congelados y registradas en reports/holdout_evaluaciones_v2.csv.")
        p.write_text(t.replace(viejo, nuevo), encoding="utf-8")
        cambios.append("B-04: conteo del holdout actualizado a 5/5")
    else:
        cambios.append("B-04: texto esperado no encontrado (revisar)")

    # B-05
    p = PROJ / "DECISIONS.md"
    t = p.read_text(encoding="utf-8")
    aplicadas = 0
    for decision, conclusion in CONCLUSIONES.items():
        patron = re.compile(
            rf"(## {decision} -.*?)(- \*\*Conclusion final\*\*: ?pendiente\.?)",
            re.DOTALL)
        nuevo_texto, cuenta = patron.subn(
            lambda m: m.group(1) + "- **Conclusion final**: " + conclusion, t, count=1)
        if cuenta:
            t = nuevo_texto
            aplicadas += 1
    if aplicadas:
        p.write_text(t, encoding="utf-8")
    cambios.append(f"B-05: {aplicadas}/8 conclusiones completadas en DECISIONS.md")

    # B-06/B-07/B-10
    for ruta_rel, nota in NOTAS_SCRIPTS.items():
        if insertar_nota(PROJ / ruta_rel, nota):
            cambios.append(f"nota de auditoria insertada en {ruta_rel}")

    # B-12 (aditiva)
    p = PROJ / "submissions" / "leaderboard_log.csv"
    with p.open(newline="", encoding="utf-8") as fh:
        filas = list(csv.reader(fh))
    header = filas[0]
    if "definicion_diferencia" not in header:
        header.append("definicion_diferencia")
        for fila in filas[1:]:
            sid = fila[0]
            if sid in {"v08_p1_fold_ensemble", "v09_p5_c_por_etiqueta",
                       "v12_p7_promedio_repeticiones"}:
                fila.append("LB - LB(V03)")
            else:
                fila.append("LB - local")
        with p.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(filas[1:])
        cambios.append("B-12: columna definicion_diferencia anadida al log (aditiva)")

    # B-02 + registro de la fase de correcciones en RUNLOG
    p = PROJ / "RUNLOG.md"
    t = p.read_text(encoding="utf-8")
    entrada = ("\n## Fase de correcciones de la auditoria forense (2026-09-26)\n\n"
               "Aplicadas en este arbol las correcciones B-01..B-14 de la auditoria "
               "(rama `proyecto-kaggle-toxic-2026-audit`):\n\n"
               "- B-01: bloque EBD-01..06 reconstruido en `experiments.csv` (202 filas, "
               "202 claves unicas, 23 campos por fila). Contenido derivado de los "
               "artefactos v5 y marcado como reconstruccion en `notes`.\n"
               "- B-02: EBD-11 y EBD-12 no existen como experimentos (huecos de "
               "numeracion historicos); se documenta aqui y no se renumeran ids.\n"
               "- B-03: rutas `experiments/experiments/` corregidas en "
               "`reports/v6_reproducibility.md`.\n"
               "- B-04: conteo del holdout actualizado a 5/5 en "
               "`reports/target_encoding_audit_v2.md`.\n"
               "- B-05: conclusiones finales completadas en DECISIONS.md (D-001, D-003, "
               "D-004, D-005, D-007, D-008, D-010, D-017).\n"
               "- B-06/B-07/B-10: notas de auditoria en los scripts (sin cambiar la "
               "logica historica).\n"
               "- B-08: retirada la copia anidada `features/features/` (higiene).\n"
               "- B-12: columna `definicion_diferencia` en `submissions/leaderboard_log.csv`.\n"
               "- B-14: notebook corregido y re-ejecutado (ver commit).\n\n")
    if "Fase de correcciones de la auditoria" not in t:
        p.write_text(t + entrada, encoding="utf-8")
        cambios.append("RUNLOG: entrada de correcciones + nota B-02")

    print("\n".join(cambios))


if __name__ == "__main__":
    main()
