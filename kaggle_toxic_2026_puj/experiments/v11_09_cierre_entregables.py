"""V11-09: completa los entregables de la v11 que quedaron sin contenido, y explica por que.

Que hace
--------
Al cerrar la fase, V11-08 dejo siete de los once entregables pedidos con contenido real. Cuatro
quedaron sin producir:

- ``reports/v11_gap_audit.md``: la auditoria de brechas **si se ejecuto**, fue V11-00, y su
  informe canonico es ``reports/v11_00_reconciliacion.md``. Este script escribe el documento con
  el nombre que se pidio, que es un **puntero** al informe con su alcance y su unico hallazgo
  real, no una copia: copiar el contenido crearia dos fuentes de verdad para la misma
  observacion, y dos fuentes de verdad divergen.
- ``reports/v11_calibration.csv``: corresponde a la linea P4, que **ya estaba medida y cerrada**
  en v10, con V10-04. En v11 no se ejecuto ningun experimento de calibracion, asi que no hay
  cifras nuevas que escribir.
- ``reports/v11_threshold_stability.csv``: corresponde a la linea P8, igualmente cerrada en v10
  con V10-06. Sin experimento propio, no hay cifras.
- ``reports/v11_error_intersection.csv``: corresponde a la interseccion de errores **entre
  candidatos**, y en v11 no hay dos candidatos. Solo hay uno vivo, V03. La interseccion de un
  conjunto consigo mismo es el propio conjunto, asi que la tabla seria trivial y vacia de
  contenido.

Por que se escriben igualmente
------------------------------
Un entregable ausente se confunde con un entregable olvidado, y esa confusion es exactamente el
tipo de cosa que la seccion 13 del enunciado obliga a registrar en lugar de pasar por alto. Un
fichero con una cabecera y **una fila que explica por que esta vacio** es mas honesto que un
fichero que no existe, porque documenta la decision y la razon.

Lo que NO se hace
-----------------
No se rellenan las tablas con cifras de v10 reetiquetadas como si fueran de v11. Seria
inventar trazabilidad: el lector veria una tabla de la v11 con resultados de la v10, y no hay
forma de distinguirlo al abrir el fichero. Se cite la ruta del artefacto de v10 que si contiene
las cifras, para que quien las necesite vaya a su sitio.

Uso
---
    /home/carlos/venv/bin/python experiments/v11_09_cierre_entregables.py
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

EXPERIMENTO = "V11-09 (cierre de entregables)"
REPORTE_MD = PROJECT_ROOT / "reports" / "v11_09_cierre_entregables.md"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v11_09_cierre_entregables.json"
REPORTE = PROJECT_ROOT / "reports"

# Los cuatro entregables sin contenido, con la razon y la ruta de quien si lo tiene.
PENDIENTES = {
    "v11_gap_audit.md": {
        "tipo": "documento",
        "linea": "P0",
        "motivo": (
            "La auditoria de brechas SI se ejecuto: es V11-00, con 25 de 25 comprobaciones "
            "superadas. Este documento es un puntero a su informe canonico, no una copia, "
            "porque dos copias de la misma observacion divergen"
        ),
        "donde_esta": "reports/v11_00_reconciliacion.md, secciones 2 a 6",
    },
    "v11_calibration.csv": {
        "tipo": "tabla",
        "linea": "P4",
        "motivo": (
            "La linea de calibracion ya estaba medida y cerrada en v10 con V10-04, que dio NO "
            "ACCIONABLE porque la calibracion mejora el ECE de 0.0583 a 0.0174 y no la F1. En "
            "v11 no se ejecuto ningun experimento de calibracion, luego no hay cifras nuevas"
        ),
        "donde_esta": "reports/v10_04_calibracion_fuerte.md y reports/v10_04_calibracion_fuerte.json",
    },
    "v11_threshold_stability.csv": {
        "tipo": "tabla",
        "linea": "P8",
        "motivo": (
            "La parametrizacion estable de umbrales ya estaba medida y cerrada en v10 con "
            "V10-06, que no supero su gate: no hay parametrizacion a la vez igual de buena y mas "
            "estable. Sin experimento propio, no hay cifras"
        ),
        "donde_esta": "reports/v10_06_umbrales_estables.md y reports/v10_06_umbrales_estables.json",
    },
    "v11_error_intersection.csv": {
        "tipo": "tabla",
        "linea": "transversal",
        "motivo": (
            "La interseccion de errores es una medida entre DOS o mas candidatos, y en v11 solo "
            "hay uno vivo, V03. La interseccion de un conjunto consigo mismo es el propio "
            "conjunto, luego la tabla seria trivial y sin contenido informativo"
        ),
        "donde_esta": "no aplica, no hay segundo candidato con el que intersectar",
    },
}

CAMPOS_VACIOS = [
    "Artefacto",
    "Linea_del_plan",
    "Estado",
    "Motivo_de_vacio",
    "Donde_esta_el_contenido_real",
]


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V11-09] {mensaje}", flush=True)


def main() -> None:
    """Escribe los cuatro entregables pendientes con su razon documentada."""
    inicio = time.time()
    log("V11-09: cierre de entregables, con la razon de cada ausencia")

    log("1/4 escribir los tres CSV con una fila que documenta el vacio")
    for nombre, meta in PENDIENTES.items():
        if meta["tipo"] != "tabla":
            continue
        with (REPORTE / nombre).open("w", encoding="utf-8", newline="") as manejador:
            escritor = csv.DictWriter(
                manejador, fieldnames=CAMPOS_VACIOS, lineterminator="\n"
            )
            escritor.writeheader()
            escritor.writerow(
                {
                    "Artefacto": nombre,
                    "Linea_del_plan": meta["linea"],
                    "Estado": "VACIO CON MOTIVO DOCUMENTADO",
                    "Motivo_de_vacio": meta["motivo"],
                    "Donde_esta_el_contenido_real": meta["donde_esta"],
                }
            )
        log(f"   {nombre}: cabecera y 1 fila de motivo")

    log("2/4 escribir el puntero de la auditoria de brechas")
    _gap_audit(inicio)

    log("3/4 comprobar que los once entregables existen ahora")
    esperados = sorted(
        [
            "v11_gap_audit.md",
            "v11_master_results.csv",
            "v11_shift_analysis.csv",
            "v11_calibration.csv",
            "v11_decision_rules.csv",
            "v11_threshold_stability.csv",
            "v11_error_intersection.csv",
            "v11_candidate_comparison.csv",
            "v11_reproducibility.md",
            "v11_compliance.md",
            "v11_final_research.md",
        ]
    )
    faltan = [nombre for nombre in esperados if not (REPORTE / nombre).exists()]
    log(f"   {len(esperados) - len(faltan)} de {len(esperados)} presentes | faltan: {faltan}")

    log("4/4 informe del experimento")
    informe = {
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "entregables_esperados": len(esperados),
        "entregables_ausentes": faltan,
        "detalle": [
            {"artefacto": nombre, **meta} for nombre, meta in PENDIENTES.items()
        ],
        "regla_aplicada": (
            "Un entregable ausente se confunde con uno olvidado. Se escribe con una fila que "
            "documenta el motivo del vacio, porque documentar una decision es mas honesto que "
            "callarla."
        ),
        "lo_que_no_se_hizo": (
            "No se copiaron cifras de v10 a tablas de v11. Seria inventar trazabilidad: el "
            "lector veria resultados de v10 etiquetados como de v11, sin forma de distinguirlo "
            "al abrir el fichero."
        ),
    }
    REPORTE_JSON.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _informe(informe, inicio)
    log(f"completado en {round(time.time() - inicio, 1)} s")
    log(f"VEREDICTO: {len(esperados) - len(faltan)} de {len(esperados)} entregables presentes")


def _gap_audit(inicio: float) -> None:
    """Escribe el documento de la auditoria de brechas como puntero a V11-00.

    Args:
        inicio: instante de arranque del script.
    """
    informe_v00 = json.loads(
        (REPORTE / "v11_00_reconciliacion.json").read_text(encoding="utf-8")
    )
    superadas = informe_v00.get("supercadas", "?")
    total = informe_v00.get("total", "?")
    lineas = [
        "# V11: auditoria de brechas (P0)",
        "",
        f"Generado por `experiments/v11_09_cierre_entregables.py` ({EXPERIMENTO}).",
        "",
        "## 1. Este documento es un puntero, y hay que decir por que",
        "",
        "La auditoria de brechas **si se ejecuto**: es **V11-00**, con "
        f"{superadas} de {total} comprobaciones superadas. Su informe canonico es",
        "**`reports/v11_00_reconciliacion.md`**, con las cinco",
        "secciones de detalle, y su registro estructurado esta en",
        "`reports/v11_00_reconciliacion.json`.",
        "",
        "Este documento existe con el nombre que pidio el plan, y **no copia el contenido**: dos",
        "copias de la misma observacion divergen, y cuando divergen ya no se sabe cual es la",
        "verdadera. El informe de V11-00 es la fuente; este solo indica donde esta.",
        "",
        "## 2. Que preguntaba la auditoria",
        "",
        "Antes de medir nada nuevo, comprobar que **las tres referencias de las que depende la",
        "v11 son coherentes y reproducibles**: la metrica corregida de V03, su huella de",
        "submission, sus umbrales congelados, la integridad del holdout2, y las cifras de",
        "desplazamiento de prevalencia que V9-02 reporto y que la v11 iba a dar por buenas.",
        "",
        "## 3. Los tres hallazgos, y que hacer con cada uno",
        "",
    ]
    for titulo, texto in (
        (
            "Hallazgo 1, propio y sin consecuencia: la comprobacion buscaba la misma clave",
            "buscaba la clave en el informe y en el registro, y en dos experimentos no se llaman "
            "igual: en V10-05 es `gate_a` en el registro y `gate_a_superado` en el informe, y en "
            "V10-10 el informe anida el veredicto en `gate_b_de_mejora.superado`. Se corrigio la "
            "comprobacion, no el dato.",
        ),
        (
            "Hallazgo 2, propio y con consecuencia real: el registro retrocedio",
            "el commit de v10 y el merge con el que se trajo a v11 **retrocedieron el "
            "`experiments.csv`** y las cuatro filas V11 se perdieron del registro. V11-05 lo "
            "resuelve derivando las "
            "filas de los informes JSON en lugar de teclearlas, con la regla de que una clave "
            "ausente queda vacia y no rellenada.",
        ),
        (
            "Hallazgo 3, de un artefacto historico: el nombre del campo no significa lo que dice",
            "el campo `factor_vs_train` de V9-02 usa como denominador la prevalencia del "
            "**desarrollo** de 1483 filas y no la del **train completo** de 1731, porque el codigo "
            "define `prevalencia_train = y_dev.mean(axis=0)`. La diferencia es de 0.0002, "
            "pequena, pero es el tipo de detalle que hace fallar una cadena de resultados. **V9-02 "
            "es internamente consistente y no tiene defecto; lo que no significa lo que dice es "
            "el nombre del campo.** Ver D-079.",
        ),
    ):
        lineas += [f"### {titulo}", "", texto, ""]
    lineas += [
        "## 4. Lo que la auditoria NO pudo comprobar",
        "",
        "**No pudo comprobar la transferencia a la prueba real**, y no es un hueco de la",
        "auditoria: es una limitacion de las reglas, que V11-02 clasifico como ambigua y NIVEL 2",
        "elevo a prohibida. Auditar si el mecanismo mejora en esta prueba concreta exigiria",
        "usar la prueba, que esta prohibido. La pregunta queda sin respuesta verificable, y se",
        "dice aqui para que no se lea mas de lo que vale.",
        "",
        "## 5. Veredicto",
        "",
        "Las tres referencias son coherentes y reproducibles. La auditoria **no encontro ningun",
        "defecto en V03 ni en el holdout2**: la huella de V03 es identica a la congelada y el",
        "holdout2 conserva su integridad con sus tres evaluaciones sin gastar. El unico",
        "hallazgo de fondo es el nombre del campo de V9-02, que se documento y serodea usando la",
        "prevalencia absoluta en lugar del factor.",
    ]
    lineas += sello(inicio)
    (REPORTE / "v11_gap_audit.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")


def _informe(informe: dict[str, Any], inicio: float) -> None:
    """Escribe el informe del cierre de entregables.

    Args:
        informe: veredicto del experimento.
        inicio: instante de arranque del script.
    """
    lineas = [
        "# V11-09: cierre de entregables",
        "",
        f"Generado por `experiments/v11_09_cierre_entregables.py` ({EXPERIMENTO}).",
        "",
        "## 1. Que faltaba y por que",
        "",
        "Al cerrar la fase con V11-08, siete de los once entregables pedidos tinham contenido",
        "real. **Cuatro no se produjeron, y no por olvido.**",
        "",
        fila(["entregable", "linea", "tipo", "motivo del vacio"]),
        fila(["---"] * 4),
    ]
    for detalle in informe["detalle"]:
        lineas.append(
            fila(
                [
                    detalle["artefacto"],
                    detalle["linea"],
                    detalle["tipo"],
                    detalle["motivo"][:150],
                ]
            )
        )
    lineas += [
        "",
        "El patron es el mismo en tres de los cuatro: **la linea correspondiente ya estaba",
        "medida y cerrada en v10**, de modo que en v11 no se ejecuto ningun experimento propio y",
        "no hay cifras nuevas que escribir.",
        "",
        "El cuarto es distinto: la interseccion de errores es una medida **entre dos o mas",
        "candidatos**, y en v11 solo hay uno vivo, V03. La interseccion de un conjunto consigo",
        "mismo es el propio conjunto, luego la tabla seria trivial.",
        "",
        "## 2. La regla aplicada",
        "",
        f"**{informe['regla_aplicada']}**",
        "",
        "Por eso los tres CSV se escriben con cabecera y **una fila que documenta el motivo**,",
        "y la auditoria de brechas se escribe como **puntero** a su informe canonico.",
        "",
        "## 3. Lo que no se hizo, y por que",
        "",
        f"**{informe['lo_que_no_se_hizo']}**",
        "",
        "Ese es el motivo de que los tres CSV **no tengan cifras**: vacios con el motivo",
        "escrito es mas honesto que llenos con numeros que no son de aqui.",
        "",
        "## 4. Estado final de los once entregables",
        "",
        f"**{informe['entregables_esperados'] - len(informe['entregables_ausentes'])} de",
        f"{informe['entregables_esperados']} presentes.** Ausentes: "
        f"{informe['entregables_ausentes'] or 'ninguno'}.",
    ]
    lineas += sello(inicio)
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
