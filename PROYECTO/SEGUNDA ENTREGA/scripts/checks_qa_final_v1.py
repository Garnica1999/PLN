"""check_qa_final.py
Fecha: 2026-09-09
Descripcion: Verificacion ejecutada del QA final de entrega (Fase N).
    Valida los 10 JSON de resultados/qa_final/ contra los criterios de
    aceptacion: es_sentencia correcto, area top-1 con prob, articulos
    extraidos > 0, BM25 == 5, citas resueltas coherentes (ano/pagerank),
    sin campos null inesperados y tiempo < 30 s. Emite tabla compacta y
    exit != 0 si cualquier check critico falla.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QA = ROOT / "resultados" / "qa_final"

# Casos esperados: (archivo, id, tipo, ano, es_sentencia esperado)
CASOS = [
    ("pos_2337.json", "2337", "T", 1997, True),
    ("pos_2728.json", "2728", "SU", 1997, True),
    ("pos_6251.json", "6251", "T", 2001, True),
    ("pos_6968.json", "6968", "C", 2001, True),
    ("pos_13015.json", "13015", "C", 2007, True),
    ("pos_13494.json", "13494", "T", 2006, True),
    ("pos_19890.json", "19890", "T", 2013, True),
    ("pos_20178.json", "20178", "C", 2013, True),
    ("neg_gaitan.json", None, None, None, False),
    ("neg_petro.json", None, None, None, False),
]

CAMPOS_OBLIGATORIOS = [
    "es_sentencia",
    "score_validacion",
    "id_detectado",
    "ponente",
    "area",
    "articulos_cp",
    "sentencias_citadas",
    "resumen",
    "tiempo_segundos",
]

fallos: list[str] = []
print(
    f"{'caso':<18} {'es_sent':<8} {'area top-1':<28} "
    f"{'p_top1':<7} {'arts':<5} {'bm25':<5} {'citas':<6} "
    f"{'res':<5} {'t(s)':<6}"
)
for archivo, id_doc, tipo, ano, esperado in CASOS:
    ruta = QA / archivo
    if not ruta.is_file():
        fallos.append(f"{archivo}: no existe")
        continue
    with open(ruta, encoding="utf-8") as fh:
        salida = json.load(fh)

    # 1. Campos obligatorios presentes y no-null inesperado
    for campo in CAMPOS_OBLIGATORIOS:
        if campo not in salida:
            fallos.append(f"{archivo}: falta campo {campo}")
    if salida.get("es_sentencia") != esperado:
        fallos.append(
            f"{archivo}: es_sentencia={salida.get('es_sentencia')} "
            f"esperado {esperado}"
        )

    # 2. Checks solo aplicables a positivos
    if esperado:
        area = salida.get("area") or {}
        top3 = area.get("top3") or []
        if not top3:
            fallos.append(f"{archivo}: area sin top3")
        top1 = top3[0] if top3 else {}
        prob = top1.get("prob")
        if prob is None or not (0 < prob <= 1):
            fallos.append(f"{archivo}: prob top-1 invalida {prob}")
        extraidos = salida["articulos_cp"]["extraidos"]
        if len(extraidos) == 0:
            fallos.append(f"{archivo}: 0 articulos extraidos")
        bm25 = salida["articulos_cp"]["recuperados_bm25"]
        if len(bm25) != 5:
            fallos.append(f"{archivo}: BM25 {len(bm25)} != 5")
        if salida.get("id_detectado") != id_doc:
            fallos.append(
                f"{archivo}: id_detectado={salida.get('id_detectado')} != {id_doc}"
            )
        # Coherencia citas resueltas: ano en canon == ano resuelto, pagerank > 0
        for cita in salida["sentencias_citadas"]:
            if cita["resuelta"]:
                if cita["ano"] is None or cita["pagerank"] is None:
                    fallos.append(
                        f"{archivo}: cita resuelta con ano/pagerank None"
                    )
                elif cita["pagerank"] <= 0:
                    fallos.append(
                        f"{archivo}: pagerank no positivo {cita['pagerank']}"
                    )
        n_res = sum(1 for c in salida["sentencias_citadas"] if c["resuelta"])
        n_citas = len(salida["sentencias_citadas"])
        fila = (
            f"{archivo:<18} {salida['es_sentencia']!s:<8} "
            f"{top1.get('area')!s:<28} {prob!s:<7} "
            f"{len(extraidos):<5} {len(bm25):<5} {n_citas:<6} "
            f"{n_res:<5} {salida['tiempo_segundos']:<6}"
        )
    else:
        # Negativos: componentes deben quedar vacios/None
        if salida.get("area") is not None:
            fallos.append(f"{archivo}: negativo con area != None")
        if salida["articulos_cp"]["extraidos"]:
            fallos.append(f"{archivo}: negativo con articulos extraidos")
        if salida["articulos_cp"]["recuperados_bm25"]:
            fallos.append(f"{archivo}: negativo con BM25 no vacio")
        if salida["sentencias_citadas"]:
            fallos.append(f"{archivo}: negativo con citas no vacias")
        fila = (
            f"{archivo:<18} {salida['es_sentencia']!s:<8} "
            f"{'-':<28} {'-':<7} {'-':<5} {'-':<5} {'-':<6} "
            f"{'-':<5} {salida['tiempo_segundos']:<6}"
        )

    # 3. Tiempo < 30 s (todos los casos)
    if salida["tiempo_segundos"] >= 30:
        fallos.append(
            f"{archivo}: tiempo {salida['tiempo_segundos']} >= 30 s"
        )
    print(fila)

print()
if fallos:
    print(f"FALLOS ({len(fallos)}):")
    for f in fallos:
        print(f"  - {f}")
    sys.exit(1)
print("TODOS LOS CHECKS PASARON (10/10 casos)")
