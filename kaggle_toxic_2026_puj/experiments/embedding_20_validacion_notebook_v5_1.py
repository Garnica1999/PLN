"""P10 de la fase v5.1 (EBD-26): validacion del notebook consolidado por ejecucion completa.

Contexto
--------
La seccion 18 del notebook se genero con ``experiments/embedding_19_notebook_v5_1.py`` sin
ejecutarla. Este script la **ejecuta de verdad**: lanza el notebook entero con el kernel del
proyecto, mide cuanto tarda cada celda, comprueba que no hay errores y que las aserciones de
la seccion 18 se cumplen, y escribe el informe de validacion.

Por que el notebook entero y no solo las celdas nuevas
-------------------------------------------------------
Ejecutar solo la seccion 18 daria una verificacion parcial: el notebook comparte estado entre
celdas (la celda de configuracion define `RUTAS` y los ayudantes de presentacion, y las 44
celdas anteriores se ejecutan antes). Un notebook que corre por separado su ultima seccion
puede fallar por una dependencia de las anteriores. El coste es que las celdas viejas tambien
se ejecutan, y por eso se ejecuta en segundo plano con un limite de tiempo por celda.

Que hace
--------
1. Localiza el notebook y comprueba que contiene la seccion 18.
2. Lo ejecuta con ``nbconvert`` en un temporal, con el kernel del venv del proyecto.
3. Extrae por celda el estado, el tiempo y los errores.
4. Comprueba que la ultima celda de la seccion 18 se ejecuto sin excepcion y que las
   aserciones de presupuesto se imprimieron.
5. Guarda el notebook **ejecutado** (con salidas) en la ruta canonica, que es como el
   proyecto versiona el notebook.
6. Escribe ``reports/embedding_notebook_validation.md``.

Uso
---
    /home/carlos/venv/bin/python experiments/embedding_20_validacion_notebook_v5_1.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

EXPERIMENTO = "EBD-26 (P10)"
NOTABLET_RELATIVO = "notebooks/10_final_pipeline.ipynb"
MARKER_SECCION = "## 18. Fase v5 y v5.1"
MARKER_ASERCION = "Aserciones de presupuesto: OK"
TIMEOUT_POR_CELDA = 1800
TIEMPO_MAXIMO_NOTEBOOK = 5400


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V5.1-P10] {mensaje}", flush=True)


def ejecutar_notebook(ruta: Path) -> dict[str, Any]:
    """Ejecuta el notebook completo con nbconvert y devuelve el notebook ejecutado.

    Args:
        ruta: ruta del notebook a ejecutar.

    Returns:
        Diccionario con el notebook ejecutado, el codigo de salida, el tiempo y el log.
    """
    destino = Path(tempfile.mkdtemp(prefix="validacion_notebook_")) / ruta.name
    comando = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        f"--ExecutePreprocessor.timeout={TIMEOUT_POR_CELDA}",
        "--ExecutePreprocessor.kernel_name=python3",
        "--output",
        str(destino),
        str(ruta),
    ]
    inicio = time.time()
    log(f"   ejecutando {ruta.name} con el kernel del proyecto (limite {TIEMPO_MAXIMO_NOTEBOOK} s)")
    proceso = subprocess.run(
        comando, capture_output=True, text=True, cwd=str(ruta.parent), timeout=TIEMPO_MAXIMO_NOTEBOOK
    )
    duracion = time.time() - inicio
    log(f"   ejecucion terminada en {round(duracion, 1)} s con codigo {proceso.returncode}")
    notebook = None
    if destino.exists():
        import nbformat

        notebook = nbformat.read(str(destino), as_version=4)
    return {
        "notebook": notebook,
        "returncode": proceso.returncode,
        "duracion_s": round(duracion, 1),
        "stdout": proceso.stdout[-4000:],
        "stderr": proceso.stderr[-6000:],
        "ruta_temporal": str(destino),
    }


def analizar(notebook_ejecutado, ejecucion: dict[str, Any]) -> dict[str, Any]:
    """Extrae el estado de cada celda y busca errores.

    Args:
        notebook_ejecutado: notebook ya ejecutado.
        ejecucion: resultado de :func:`ejecutar_notebook`.

    Returns:
        Diccionario con celdas, errores y hallazgos.
    """
    if notebook_ejecutado is None:
        # nbconvert no escribe el fichero de salida cuando una celda lanza una excepcion, asi
        # que el diagnostico tiene que salir de su propio stderr. Sin esto el informe solo
        # diria "no se genero", que no es accionable.
        stderr = ejecucion.get("stderr", "")
        diagnostico = [
            linea
            for linea in stderr.splitlines()
            if any(marca in linea for marca in ("Error", "error:", "Traceback", "assert", "NameError", "Line"))
        ]
        return {
            "celdas": [],
            "errores": diagnostico[-12:] or ["nbconvert no genero salida y no informo de la causa"],
            "hallazgos": {"seccion_18_ejecutada": False, "aserciones_ok": False},
            "n_errores": 1,
        }
    celdas: list[dict[str, Any]] = []
    errores: list[str] = []
    texto_salida: list[str] = []
    inicio_indice = None
    for indice, celda in enumerate(notebook_ejecutado.cells):
        fuente = "".join(celda.source)
        if celda.cell_type == "markdown" and fuente.strip().startswith(MARKER_SECCION):
            inicio_indice = indice
        if celda.cell_type != "code":
            continue
        tiene_error = False
        for salida in celda.get("outputs", []):
            if salida.get("output_type") == "error":
                tiene_error = True
                errores.append(
                    f"celda {indice}: {salida.get('ename')}: {str(salida.get('evalue'))[:200]}"
                )
            elif salida.get("output_type") == "stream":
                texto_salida.append(str(salida.get("text", "")))
            elif salida.get("output_type") in ("display_data", "execute_result"):
                # Los mensajes de `display(...)` y el resultado de la ultima expresion no
                # llegan como flujo de texto: viven en el contenido de la salida. Buscar solo
                # en `stream` daria un falso negativo de validacion.
                contenido = salida.get("data", {})
                # Para un objeto de `display` el campo `text/plain` es la repr del objeto
                # (`<IPython.core.display.Markdown object>`) y el texto real esta en
                # `text/markdown`. Se recogen **ambos**: el orden importa porque el primero
                # que exista se usaria, y el repr no contiene nada legible.
                partes = []
                for clave in ("text/markdown", "text/plain"):
                    valor = contenido.get(clave)
                    if valor:
                        partes.append(valor if isinstance(valor, str) else "".join(valor))
                if partes:
                    texto_salida.append("\n".join(partes))
        celdas.append(
            {
                "indice": indice,
                "en_seccion_18": inicio_indice is not None and indice > inicio_indice,
                "ejecutada": celda.get("execution_count") is not None,
                "error": tiene_error,
                "n_salidas": len(celda.get("outputs", [])),
                "primera_linea": fuente.strip().splitlines()[0][:70] if fuente.strip() else "",
            }
        )
    salida_completa = "\n".join(texto_salida)
    codigo_v03_ok = "0.777211" in salida_completa
    hallazgos = {
        "seccion_18_ejecutada": inicio_indice is not None
        and any(c["en_seccion_18"] and c["ejecutada"] for c in celdas),
        "aserciones_ok": MARKER_ASERCION in salida_completa,
        "v03_verificado": codigo_v03_ok,
        "presupuesto_impreso": "budget_remaining" in salida_completa,
    }
    return {
        "celdas": celdas,
        "errores": errores,
        "hallazgos": hallazgos,
        "n_errores": len(errores),
        "n_celdas_codigo": len(celdas),
        "n_celdas_ejecutadas": sum(1 for c in celdas if c["ejecutada"]),
        "celdas_sin_ejecutar": [c["indice"] for c in celdas if not c["ejecutada"]],
    }


def main() -> None:
    """Ejecuta el notebook completo y escribe el informe de validacion."""
    inicio = time.time()
    log("P10: validacion del notebook consolidado por ejecucion completa")
    ruta = PROJECT_ROOT / NOTABLET_RELATIVO
    if not ruta.exists():
        raise RuntimeError(f"No existe {NOTABLET_RELATIVO}")
    import nbformat

    original = nbformat.read(str(ruta), as_version=4)
    if not any(
        celda.cell_type == "markdown" and MARKER_SECCION in "".join(celda.source)
        for celda in original.cells
    ):
        raise RuntimeError("El notebook no contiene la seccion 18; ejecuta antes embedding_19")
    log(f"   notebook: {len(original.cells)} celdas, seccion 18 presente")

    ejecucion = ejecutar_notebook(ruta)
    analisis = analizar(ejecucion["notebook"], ejecucion)
    log(
        f"   {analisis.get('n_celdas_ejecutadas', 0)}/{analisis.get('n_celdas_codigo', 0)} celdas "
        f"de codigo ejecutadas | {analisis['n_errores']} errores | hallazgos: {analisis['hallazgos']}"
    )

    if ejecucion["returncode"] != 0 or analisis["n_errores"] > 0:
        log("   el notebook no termina sin errores; se registra igualmente el resultado")
    else:
        # El proyecto versiona el notebook con salidas: se guarda el ejecutado.
        destino = Path(ejecucion["ruta_temporal"])
        if destino.exists():
            shutil.copy2(destino, ruta)
            log(f"   notebook ejecutado guardado en {NOTABLET_RELATIVO} (con salidas)")

    veredicto = (
        "GO"
        if ejecucion["returncode"] == 0
        and analisis["n_errores"] == 0
        and all(analisis["hallazgos"].values())
        else "NO-GO"
    )
    _informe(original, ejecucion, analisis, veredicto, time.time() - inicio)
    log(f"P10 completada en {round(time.time() - inicio, 1)} s | veredicto de validacion: {veredicto}")


def _informe(
    original,
    ejecucion: dict[str, Any],
    analisis: dict[str, Any],
    veredicto: str,
    duracion: float,
) -> None:
    """Escribe el informe de validacion del notebook.

    Args:
        original: notebook antes de ejecutar.
        ejecucion: resultado de la ejecucion.
        analisis: analisis de las celdas.
        veredicto: veredicto de la validacion.
        duracion: segundos totales.
    """
    ruta = PROJECT_ROOT / "reports" / "embedding_notebook_validation.md"
    hallazgos = analisis["hallazgos"]
    lineas = [
        "# Validacion del notebook consolidado por ejecucion (EBD-26)",
        "",
        f"Generado por `experiments/embedding_20_validacion_notebook_v5_1.py` ({EXPERIMENTO}).",
        "",
        "## 1. Que se valida",
        "",
        "El **notebook entero**, no solo la seccion nueva. El notebook comparte estado entre",
        "celdas (la de configuracion define `RUTAS` y los ayudantes de presentacion) y las 44",
        "celdas anteriores se ejecutan antes, de modo que ejecutar la seccion 18 por separado",
        "daria una verificacion parcial que podria fallar por una dependencia de las previas.",
        "",
        "## 2. Resultado de la ejecucion",
        "",
        "| magnitud | valor |",
        "|---|---|",
        f"| notebook | `{NOTABLET_RELATIVO}` |",
        f"| celdas totales | {len(original.cells)} |",
        f"| celdas de codigo | {analisis.get('n_celdas_codigo', 0)} |",
        f"| celdas ejecutadas | {analisis.get('n_celdas_ejecutadas', 0)} |",
        f"| celdas sin ejecutar | {len(analisis.get('celdas_sin_ejecutar', []))} |",
        f"| errores | {analisis['n_errores']} |",
        f"| duracion | {ejecucion['duracion_s']} s |",
        f"| codigo de salida | {ejecucion['returncode']} |",
        f"| limite por celda | {TIMEOUT_POR_CELDA} s |",
        "",
        "## 3. Comprobaciones exigidas por la seccion 18",
        "",
        "| comprobacion | resultado |",
        "|---|---|",
        f"| la seccion 18 se ejecuta | {'OK' if hallazgos.get('seccion_18_ejecutada') else 'FALLA'} |",
        f"| las aserciones de presupuesto se cumplen | {'OK' if hallazgos.get('aserciones_ok') else 'FALLA'} |",
        f"| el contador de presupuesto se imprime | {'OK' if hallazgos.get('presupuesto_impreso') else 'FALLA'} |",
        f"| V03 verificado en 0.777211 | {'OK' if hallazgos.get('v03_verificado') else 'FALLA'} |",
        "",
        "## 4. Errores",
        "",
    ]
    if analisis["errores"]:
        for error in analisis["errores"]:
            lineas.append(f"- {error}")
    else:
        lineas.append("Ninguno.")
    if ejecucion["stderr"].strip():
        lineas += ["", "### Ultimas lineas de stderr", "", "```", ejecucion["stderr"][-1500:], "```"]
    lineas += [
        "",
        "## 5. Cobertura de lo que pide la seccion 23 de la spec",
        "",
        "| requisito | donde esta |",
        "|---|---|",
        "| Motivacion | 18.0.1 |",
        "| Restricciones y autorizacion | 18.0.2 |",
        "| Corpus: construccion, limpieza, tokenizacion | 18.1 |",
        "| Las tres variantes de corpus con escalas | 18.1 y 18.5 |",
        "| Coverage y OOV | 18.2 y 18.3 |",
        "| FastText frente a Word2Vec-control | 18.3 |",
        "| Pooling y downstream | 18.4 |",
        "| Downstream y fusion | 18.4 y 18.5 |",
        "| Curva de aprendizaje con linea de V03 | 18.5 |",
        "| Complementariedad y slices | 18.6 |",
        "| Vector size, window, min count, subwords, negative | 18.7 |",
        "| Tabla maestra A y B con doble contabilidad | 18.8 |",
        "| Contador automatico de presupuesto con asercion | 18.8 |",
        "| Conclusiones separadas (representacion y tarea) | 18.9 |",
        "",
        "## 6. Veredicto",
        "",
        f"**{veredicto}**",
        "",
        "GO significa que el notebook completo se ejecuta sin errores y que las",
        "comprobaciones de la seccion 18 se cumplen: la seccion se ejecuta, las aserciones de",
        "presupuesto pasan y V03 se verifica en 0.777211 desde la celda de configuracion.",
        "",
        f"Fecha: {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
        f"Duracion total: {round(duracion, 1)} s.",
    ]
    ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    (PROJECT_ROOT / "reports" / "embedding_notebook_validation_v5_1.json").write_text(
        json.dumps(
            {
                "experimento": EXPERIMENTO,
                "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "notebook": NOTABLET_RELATIVO,
                "celdas_totales": len(original.cells),
                "duracion_s": ejecucion["duracion_s"],
                "returncode": ejecucion["returncode"],
                "n_errores": analisis["n_errores"],
                "errores": analisis["errores"],
                "hallazgos": hallazgos,
                "veredicto": veredicto,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
