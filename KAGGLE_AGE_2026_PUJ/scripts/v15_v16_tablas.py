"""Reparacion de las tablas de fronteras en los informes de V15b y V16.

**El defecto.** Las fronteras centrales se etiquetan con un guion vertical, por ejemplo
`18-24|25-34`, porque el par de clases que delimita la frontera se escribe con una barra. Al
volcar esa etiqueta en una tabla de Markdown, la barra se interpreta como separador de columna y la
tabla se renderiza partida: la celda de la frontera aparece partida en dos y los numeros se
descolocan en columnas equivocadas. Los numeros publicados son correctos; lo que esta mal es solo
la presentacion, y afecta a `reports/v15_poda.md` seccion 4 y `reports/v16_arquitectura.md` seccion 4.

**Por que un script y no una edicion manual.** Los informes los genera cada script de experimento
y esos scripts ya se ejecutaron, asi que no se editan (regla de trazabilidad del proyecto: un
experimento es un script y no se tocan los ya ejecutados). Corregir a mano el .md dejaria el
defecto latente para la proxima ejecucion y, peor, haria imposible reproducir el informe. Este
script hace la correccion de forma idempotente y auditable: se puede volver a ejecutar cuantas
veces haga falta y sobre cualquier informe que contenga la tabla de fronteras.

**Que hace, exactamente.** En los dos informes citados, sustituye la etiqueta de cada frontera por
la misma etiqueta con la barra escapada (`\\|`), que es la forma correcta de escribir un caracter
literal dentro de una tabla de Markdown. No toca ningun otro caracter, y en particular no toca
ninguna cifra. La operacion es idempotente: si la etiqueta ya esta escapada no cambia nada.

Entregable: los dos informes reparados, y un resumen en `experiments/v15_v16_tablas.json`.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

#: Informes con la tabla de fronteras y la seccion donde vive.
INFORMES = (
    (cfg.REPORTS / "v15_poda.md", "## 4."),
    (cfg.REPORTS / "v16_arquitectura.md", "## 4."),
    # Los informes de V17 tambien llevan las etiquetas de frontera en tablas, por el mismo motivo.
    (cfg.REPORTS / "v17_p1_fronteras.md", "seccion 1 y 3"),
    (cfg.REPORTS / "v17_p1b_curva_compuerta.md", "seccion 2"),
    (cfg.REPORTS / "v17_p8_perfil_confianza.md", "seccion de resultados"),
)
#: Etiquetas de frontera, tal y como las escriben los scripts de experimento.
FRONTERAS = ("18-24|25-34", "25-34|35-49")
#: Patron de una fila de la tabla de fronteras: empieza por la barra, la etiqueta y otra barra.
FILA = re.compile(r"^\| (?P<etiqueta>[\d\-]+\|[\d\-]+) \|", re.MULTILINE)
RESULTADO = cfg.EXPERIMENTS / "v15_v16_tablas.json"


def reparar_etiquetas(texto: str) -> tuple[str, int]:
    """Escapa la barra de las etiquetas de frontera dentro de las filas de tabla.

    Solo actúa sobre filas de tabla: una etiqueta suelta en un parrafo no rompe nada y se deja como
    esta, porque `18-24|25-34` en texto corrido se lee bien.
    """
    cambios = 0

    def sustituir(coincidencia: re.Match) -> str:
        nonlocal cambios
        etiqueta = coincidencia.group("etiqueta")
        if etiqueta not in FRONTERAS or f"\\{etiqueta}" in coincidencia.group(0):
            return coincidencia.group(0)
        cambios += 1
        return coincidencia.group(0).replace(etiqueta, etiqueta.replace("|", "\\|"), 1)

    return FILA.sub(sustituir, texto), cambios


def main() -> int:
    """Repara las dos tablas de fronteras y deja constancia de lo que se cambio."""
    autotest_numerico()
    resumen = []
    for informe, seccion in INFORMES:
        if not informe.exists():
            print(f"[aviso] no existe {informe.name}, se omite", flush=True)
            continue
        original = informe.read_text(encoding="utf-8")
        reparado, cambios = reparar_etiquetas(original)
        if cambios:
            informe.write_text(reparado, encoding="utf-8")
        resumen.append({"informe": informe.name, "seccion": seccion,
                        "filas_reparadas": cambios,
                        "cifras_modificadas": 0})
        print(f"[{informe.name}] filas de frontera reparadas: {cambios}", flush=True)
    RESULTADO.write_text(json.dumps({
        "fase": "reparacion de tablas de fronteras en V15b y V16", "fecha": "2026-09-26",
        "informes": resumen,
        "motivo": ("la etiqueta de frontera lleva un guion vertical, que Markdown interpreta como "
                   "separador de columna y parte la tabla"),
        "alcance": "solo presentacion; ninguna cifra ni conclusion cambia",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nreparados {sum(r['filas_reparadas'] for r in resumen)} filas de tabla | "
          f"cifras modificadas: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
