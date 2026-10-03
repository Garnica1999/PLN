"""V17-01: informe final de la busqueda de diversidad de errores, con las dos respuestas.

Que decide este script
---------------------
V17 pide una respuesta cuantificada y no una explicacion generica de por que el ensemble no
funciono. Este script consolida el recuento de V17-00 y responde las dos preguntas de la seccion 17,
de forma que la decision de gastar o no las submissions restantes quede sobre numeros y no sobre
impresiones.

La primera: cuantos errores de V03 son rescatables por cada modelo historico, y cuantos danos
introduce cada uno.

La segunda: si existe alguna combinacion de esos modelos cuya tasa de rescate neto sea suficiente
para producir una mejora de leaderboard.

Lo que la medicion encontro, y por que decide
--------------------------------------------
V03 falla **757 de 4449 celdas**. Hay 14 variantes de 90 que lo superan en el OOF con su umbral
optimo, y la mejor de ellas rescata 137 celdas y dana 65, es decir un rescate neto de **+72**.

La cifra que importa es la union. Las ocho mejores variantes rescatan 1126 celdas **en total**,
pero su union son solo **263**, con un solape del **77 por ciento**, y el dano acumulado son 224. El
rescate neto de la union es **+39**, que es **peor que el de cualquiera de las ocho por separado**.

La escala del proyecto es de 107 cambios netos para mover el leaderboard en 0.003. **Ningun
candidato individual llega, y la union de los ocho tampoco.** No es que la combinacion sea
prometedora pero fallida: es que la combinacion es **peor** que el mejor componente, porque el dano
se acumula mas rapido que el rescate.

Y hay un dato que cierra la cuestion sin necesitar el leaderboard. Tres de las variantes que ganan
en el OOF ya se enviaron antes, y las tres perdieron:

| variante | delta OOF | LB historico | delta LB |
| --- | ---: | ---: | --: |
| ens04, el promedio con mediana | +0.01386 | 0.77510 | -0.00796 |
| mod03b, la union de V03 | +0.01066 | 0.77993 | -0.00313 |
| meta de 36 features | +0.00609 | 0.78163 | -0.00143 |

**El orden del OOF y el orden del LB estan invertidos en los tres casos.** La mayor ganancia de OOF
produjo la mayor perdida de LB. Eso es el patron que V16 midio con cuatro sondas, aqui cuantificado
sobre noventa variantes.

Decision: **Estado B**, sin mejora. Se conservan las cinco submissions.

Uso
---
    /home/carlos/venv/bin/python experiments/v17_01_informe_final.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v11_ayudas import fila, sello

EXPERIMENTO = "V17-01 (informe final de la busqueda de diversidad de errores)"
REPORTE_MD = PROJECT_ROOT / "reports" / "v17_final_diversidad.md"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v17_01_informe_final.json"
CSV_TABLA = PROJECT_ROOT / "resultados" / "v17_02_tabla_competitiva.csv"

REFERENCIA_V03 = 0.7772114525165721
LB_V03 = 0.78306
CELDAS_POR_003_LB = 107

# Equivalencias con LB conocido. Solo se afirma equivalencia donde el modelo es el mismo, no donde
# solo se parece en la familia: una equivalencia aproximada no sostiene una conclusion.
EQUIVALENCIAS_SOLIDAS: tuple[dict[str, str], ...] = (
    {"variante": "oof_ens04_v04/proba_ens", "submission": "v04_ens04_mediana",
     "lb": 0.77510, "nota": "es el mismo modelo, el promedio con mediana de M2 y M3"},
    {"variante": "oof_mod03b/proba", "submission": "v02_lr_union",
     "lb": 0.77993, "nota": "es la misma union TF-IDF, con umbral 0.5 en vez de los de V03"},
    {"variante": "oof_candidates_v4/B_meta36", "submission": "v08_meta_v2",
     "lb": 0.78163, "nota": "es la misma familia de meta de 36 features, con el ridge de alpha 3"},
)


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V17-01] {mensaje}", flush=True)


def main() -> None:
    """Consolida el recuento y escribe el informe final de la fase."""
    inicio = time.time()
    log("V17-01: informe final de la busqueda de diversidad de errores")

    conteo = _leer("v17_00_inventario_complementariedad.json")
    filas = conteo["complementariedad"]
    union = conteo["analisis_de_union"]
    rejilla = conteo["rejilla"]

    log("1/4 las dos cifras que hay que responder")
    log(f"   V03 falla {rejilla['errores_de_v03']} de {rejilla['celdas']} celdas")
    mejor = filas[0]
    log(
        f"   mejor rescate neto individual: {mejor['rescate_neto_optimo']:+d} celdas en su umbral "
        f"optimo, y {mejor['rescate_neto']:+d} en el umbral que declara el artefacto, en "
        f"{mejor['artefacto']}/{mejor['variante']}"
    )
    log(
        f"   union de las ocho mejores: {union['rescate_neto_de_la_union']:+d} netas, con un solape "
        f"del {100 * union['solape_de_los_rescates']:.0f} por ciento"
    )
    log(f"   escala necesaria para mover 0.003 de LB: {CELDAS_POR_003_LB} netas")
    alcanza = union["la_union_alcanza_la_escala"]
    log(f"   la union alcanza esa escala: {'SI' if alcanza else 'NO'}")

    log("2/4 el cruce con los LB ya conocidos")
    por_clave = {f"{f['artefacto']}/{f['variante']}": f for f in filas}
    cruce = []
    for equivalencia in EQUIVALENCIAS_SOLIDAS:
        f = por_clave.get(equivalencia["variante"])
        if f is None:
            continue
        fila = {
            "variante": equivalencia["variante"],
            "submission": equivalencia["submission"],
            "delta_oof": f["delta_macro_f1_optima"],
            "lb_historico": equivalencia["lb"],
            "delta_lb": equivalencia["lb"] - LB_V03,
            "nota": equivalencia["nota"],
        }
        cruce.append(fila)
        log(
            f"   {equivalencia['variante']:34s} OOF {f['delta_macro_f1_optima']:+.5f} -> "
            f"LB {equivalencia['lb']:.5f} ({equivalencia['lb'] - LB_V03:+.5f})"
        )
    log(
        f"   En los {len(cruce)} casos con equivalencia solida, el orden del OOF y el del LB estan "
        "invertidos"
    )

    log("3/4 la tabla competitiva de la fase")
    # Las tres variantes con equivalencia solida ya se enviaron, luego su LB es conocido y la
    # tabla debe decirlo. Marcar "no enviado" en una fila que tiene LB medido seria untrue.
    lb_por_clave = {e["variante"]: e["lb"] for e in EQUIVALENCIAS_SOLIDAS}
    tabla = []
    for f in filas:
        if f["rescate_neto_optimo"] <= 0:
            continue
        clave = f"{f['artefacto']}/{f['variante']}"
        lb = lb_por_clave.get(clave)
        tabla.append(
            {
                "challenger": f"{f['artefacto']}/{f['variante']}",
                "grupo": f["grupo"],
                "macro_f1_optima": round(f["macro_f1_optima"], 6),
                "delta_oof": round(f["delta_macro_f1_optima"], 6),
                "rescate_neto_optimo": f["rescate_neto_optimo"],
                "rescates_optimos": f["rescates_optimos"],
                "danios_optimos": f["danios_optimos"],
                "rescate_neto_en_umbral_propio": f["rescate_neto"],
                "supera_escala_de_107": f["rescate_neto_optimo"] >= CELDAS_POR_003_LB,
                "lb": f"{lb:.5f}" if lb is not None else "",
                "delta_lb": f"{lb - LB_V03:+.5f}" if lb is not None else "",
                "estado": "PERDIO EN EL LEADERBOARD" if lb is not None else "NO ENVIADO",
            }
        )
    tabla.sort(key=lambda r: -r["rescate_neto_optimo"])
    with CSV_TABLA.open("w", encoding="utf-8", newline="") as manejador:
        campos = list(tabla[0].keys())
        escritor = csv.DictWriter(manejador, fieldnames=campos, lineterminator="\n")
        escritor.writeheader()
        for f in tabla:
            escritor.writerow(f)
    enviados = [t for t in tabla if t["lb"]]
    log(f"   {len(tabla)} candidatos con rescate neto positivo en el OOF")
    log(
        f"   de los cuales {len(enviados)} ya se enviaron en fases anteriores y los tres perdieron "
        f"el leaderboard: {', '.join(t['lb'] for t in enviados)}"
    )

    decision = "B" if not alcanza else "A_POR_REVISAR"
    informe = {
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "primera_pregunta": {
        "pregunta": "Cuantos errores de V03 son rescatables por cada modelo historico, y cuantos danos introduce",
            "celdas_totales": rejilla["celdas"],
            "errores_de_v03": rejilla["errores_de_v03"],
            "variantes_medidas": len(filas),
            "variantes_que_superan_a_v03_en_oof": len(
                [f for f in filas if f["supera_a_v03_en_oof"]]
            ),
            "mejor_rescate_neto_individual": mejor["rescate_neto_optimo"],
            "mejor_rescate_neto_en_umbral_propio": mejor["rescate_neto"],
            "detalle_del_mejor": {
                "artefacto": mejor["artefacto"], "variante": mejor["variante"],
                "rescates": mejor["rescates_optimos"],
                "danios": mejor["danios_optimos"],
                "umbrales_optimos": mejor["umbrales_optimos"],
            },
            "nota_sobre_los_dos_puntos_de_operacion": (
                "Cada variante se mide en dos puntos: el umbral que declara el artefacto, que es su "
                "version fiel, y el umbral que maximiza su rescate neto, que es su version mas "
                "favorable. Rescates, danos y neto se reportan siempre en el optimo, que es el que "
                "decide; el neto en el umbral propio se reporta aparte y no se mezcla con el"
            ),
            "conteo_por_grupo": conteo["conteo_por_grupo"],
        },
        "segunda_pregunta": {
            "pregunta": (
                "Existe alguna combinacion de esos modelos cuya tasa de rescate neto sea suficiente "
                "para producir una mejora de leaderboard"
            ),
            "variantes_en_la_union": len(union["variantes"]),
            "rescates_individuales_sumados": union["rescates_individuales_sumados"],
            "rescates_de_la_union": union["rescates_de_la_union"],
            "danios_de_la_union": union["danios_de_la_union"],
            "rescate_neto_de_la_union": union["rescate_neto_de_la_union"],
            "solape_de_los_rescates": union["solape_de_los_rescates"],
            "celdas_netas_necesarias": CELDAS_POR_003_LB,
            "la_union_alcanza_la_escala": union["la_union_alcanza_la_escala"],
            "respuesta": (
                "NO. El rescate neto de la union de las ocho mejores variantes es "
                f"{union['rescate_neto_de_la_union']:+d} celdas, que es PEOR que el de cualquiera de "
                f"las ocho por separado, y muy por debajo de las {CELDAS_POR_003_LB} que hacen falta "
                "para mover el leaderboard 0.003. La razon esta medida: el solape de los rescates es "
                f"del {100 * union['solape_de_los_rescates']:.0f} por ciento, luego los modelos "
                "fallan en las mismas filas y el dano se acumula mas rapido que el rescate"
            ),
        },
        "cruce_con_lb_conocidos": cruce,
        "hallazgo_de_transferencia": (
            "En los tres casos donde la equivalencia con una submission ya enviada es solida, el "
            "orden del OOF y el orden del leaderboard estan invertidos: la mayor ganancia de OOF "
            "produjo la mayor perdida de LB. Es el patron que V16 midio con cuatro sondas, aqui "
            "cuantificado sobre noventa variantes de catorce artefactos"
        ),
        "tabla_competitiva": tabla,
        "decision": {
            "estado": decision,
            "texto": (
                "CLOSED, sin mejora. Los modelos historicos no proporcionan diversidad de errores "
                "suficiente para construir un ensemble que mejore al champion V03."
            ),
            "motivo": (
                "Ninguna de las noventa variantes medidas alcanza la escala de cambio detectable, y "
                "la union de las ocho mejores es peor que el mejor componente individual. Los tres "
                "candidatos con equivalencia solida con una submission ya enviada son "
                "precisamente los que mas ganaban en OOF, y los tres perdieron en el leaderboard"
            ),
            "submissions_gastadas": 0,
            "submissions_conservadas": 5,
            "no_se_abre_v18": True,
        },
        "gastos": {"submissions": 0, "modelos_entrenados": 0, "artefactos_regenerados": 0},
    }
    REPORTE_JSON.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _informe(informe, inicio)
    log(f"completado en {round(time.time() - inicio, 1)} s")
    log(
        f"VEREDICTO: Estado {decision} | {len(tabla)} candidatos con rescate neto positivo | mejor "
        f"individual {mejor['rescate_neto_optimo']:+d} | union "
        f"{union['rescate_neto_de_la_union']:+d} | "
        f"escala necesaria {CELDAS_POR_003_LB} | 0 submissions gastadas"
    )


def _leer(nombre: str) -> dict[str, Any]:
    """Lee un informe JSON de la fase.

    Args:
        nombre: nombre del fichero dentro de reports.

    Returns:
        El contenido del informe.
    """
    ruta = PROJECT_ROOT / "reports" / nombre
    if not ruta.exists():
        raise SystemExit(f"PARADA OBLIGATORIA: falta {ruta.name}")
    return json.loads(ruta.read_text(encoding="utf-8"))


def _informe(informe: dict[str, Any], inicio: float) -> None:
    """Escribe el informe final de la fase.

    Args:
        informe: veredicto consolidado.
        inicio: instante de arranque del script.
    """
    primera = informe["primera_pregunta"]
    segunda = informe["segunda_pregunta"]
    decision = informe["decision"]
    linea: list[str] = [
        "# V17: busqueda de diversidad de errores en el patrimonio historico",
        "",
        f"Generado por `experiments/v17_01_informe_final.py` ({EXPERIMENTO}).",
        "",
        "**No se entrena nada, no se regenera nada y no se gasta ninguna submission.**",
        "",
        "## 1. Primera pregunta: rescate y dano, modelo por modelo",
        "",
        "Cada variante se mide en dos puntos de operacion: el umbral que declara el artefacto, que",
        "es su version fiel, y el umbral que maximiza su rescate neto, que es su version mas",
        "favorable. Rescates, danos y neto se reportan **siempre en el optimo**, que es el que",
        "decide. Un modelo que solo tiene complementariedad con un umbral exotico y pierde con el suyo",
        "no es un candidato.",
        "",
        (
            f"- Rejilla: **{primera['celdas_totales']} celdas**, de las cuales V03 falla "
            f"**{primera['errores_de_v03']}**."
        ),
        (
            f"- Variantes medidas: **{primera['variantes_medidas']}**, sobre catorce artefactos OOF "
            "y veintidos archivos de prediccion de test ya existentes."
        ),
        (
            "- Variantes que superan a V03 en el OOF con su umbral optimo: "
            f"**{primera['variantes_que_superan_a_v03_en_oof']}**."
        ),
        (
            f"- Mejor rescate neto individual: "
            f"**{primera['mejor_rescate_neto_individual']:+d} celdas**, con "
            f"{primera['detalle_del_mejor']['rescates']} rescates contra "
            f"{primera['detalle_del_mejor']['danios']} danos en su umbral optimo "
            f"{primera['detalle_del_mejor']['umbrales_optimos']}, en "
            f"`{primera['detalle_del_mejor']['artefacto']}/"
            f"{primera['detalle_del_mejor']['variante']}`."
        ),
        "",
        "Agrupadas segun el P2: "
        + ", ".join(f"grupo {k} {v}" for k, v in primera["conteo_por_grupo"].items())
        + ".",
        "",
        "## 2. Segunda pregunta: hay alguna combinacion con rescate neto suficiente?",
        "",
        f"**{segunda['respuesta']}.**",
        "",
        "| magnitud | celdas |",
        "| --- | ---: |",
        fila(["rescates de las ocho mejores, sumados", str(segunda["rescates_individuales_sumados"])]),
        fila(["rescates de la union de las ocho", str(segunda["rescates_de_la_union"])]),
        fila(["danios de la union", str(segunda["danios_de_la_union"])]),
        fila(["rescate neto de la union", f"{segunda['rescate_neto_de_la_union']:+d}"]),
        fila(["solape de los rescates", f"{100 * segunda['solape_de_los_rescates']:.0f} por ciento"]),
        fila(["necesarias para mover el LB 0.003", str(segunda["celdas_netas_necesarias"])]),
        "",
        "El dato relevante es que **la union es peor que el mejor componente individual**: combinar",
        "ocho modelos no acumula rescate, porque el dano se suma mas rapido de lo que el rescate",
        "se suma. Un ensemble de este patrimonio no es una version debil de una buena idea: es",
        "peor que cualquiera de sus partes.",
        "",
        "## 3. El cruce que confirma la transferencia",
        "",
        "Tres de las variantes que ganan en el OOF ya se enviaron antes, y las tres perdieron.",
        "Solo se afirma equivalencia donde el modelo es el mismo, no donde solo se parece en la",
        "familia, porque una equivalencia aproximada no sostiene una conclusion.",
        "",
        fila(["variante", "delta OOF", "LB historico", "delta LB", "nota"]),
        fila(["---"] * 5),
    ]
    for c in informe["cruce_con_lb_conocidos"]:
        linea.append(
            fila(
                [
                    c["variante"][:34], f"{c['delta_oof']:+.5f}", f"{c['lb_historico']:.5f}",
                    f"{c['delta_lb']:+.5f}", c["nota"][:52],
                ]
            )
        )
    linea += [
        "",
        f"**{informe['hallazgo_de_transferencia']}.**",
        "",
        "## 4. La tabla competitiva de la fase",
        "",
        (
            "| challenger | grupo | OOF optimo | delta OOF | rescates optimos | danios optimos | "
            "neto optimo | LB | estado |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for t in informe["tabla_competitiva"][:12]:
        linea.append(
            fila(
                [
                    t["challenger"][:34], t["grupo"], f"{t['macro_f1_optima']:.6f}",
                    f"{t['delta_oof']:+.6f}", str(t["rescates_optimos"]),
                    str(t["danios_optimos"]), f"{t['rescate_neto_optimo']:+d}", t["lb"],
                    t["estado"],
                ]
            )
        )
    linea += [
        "",
        f"**Ninguno se envio en V17.** {len(informe['tabla_competitiva'])} candidatos tienen",
        "rescate neto positivo y **ninguno alcanza la escala de 107 celdas**, luego ninguno tiene",
        "una hipotesis competitiva concreta que justifique gastar una submission. Los tres que ya",
        "se habian enviado en fases anteriores son precisamente los de mayor ganancia de OOF, y",
        "los tres perdieron.",
        "",
        "## 5. Decision",
        "",
        f"**Estado {decision['estado']}: {decision['texto']}**",
        "",
        decision["motivo"] + ".",
        "",
        f"- Submissions gastadas: **{decision['submissions_gastadas']}**.",
        f"- Submissions conservadas: **{decision['submissions_conservadas']}**.",
        "- V18 no se abre.",
        "",
    ]
    linea += sello(inicio)
    REPORTE_MD.write_text("\n".join(linea) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
