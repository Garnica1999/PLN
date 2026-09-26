"""V13 · P0a — Auditoría de idioma de los artefactos del proyecto.

Comprueba que el codigo, la documentacion y los datos derivados estan **en español** y sin
simbolos ni caracteres procedentes de otros alfabetos. Motivo: durante la fase de lexicones se
colaron caracteres CJK, cirilicos y signos de puntuacion chinos en listas de palabras y en
docstrings, y en V13 eso ya no puede pasar inadvertido.

Clasifica cada hallazgo en:

* **ERROR**: alfabeto no latino (CJK, cirilico, arabe, hebreo, devanagari,.tailandés), puntuacion
  CJK, mojibake (secuencias tipo ``Ã``/``â€``) o sustituciones corruptas.
* **AVISO**: emoji y dingbats (no son otro idioma, pero ensucian un artefacto tecnico).
* **OK**: latin con tildes espanolas, signos matematicos deliberados (delta, rho, sigma, epsilon,
  beta, menos-mas, menor-igual, mayor-igual) y puntuacion habitual.

Salida: ``reports/v13_auditoria_idioma.md``, ``experiments/v13_p0a_auditoria_idioma.json``.
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402

INFORME = cfg.REPORTS / "v13_auditoria_idioma.md"
RESULTADO = cfg.EXPERIMENTS / "v13_p0a_auditoria_idioma.json"

#: Rangos de alfabetos no latinos: cualquier ahi es un error.
RANGOS_PROHIBIDOS = [
    (0x0400, 0x04FF, "cirilico"),
    (0x0590, 0x05FF, "hebreo"),
    (0x0600, 0x06FF, "arabe"),
    (0x0700, 0x074F, "sirio"),
    (0x0900, 0x097F, "devanagari"),
    (0x0E00, 0x0E7F, "tailandes"),
    (0x3040, 0x30FF, "japones (kana)"),
    (0x4E00, 0x9FFF, "chino (han)"),
    (0xAC00, 0xD7AF, "coreano (hangul)"),
    (0x3000, 0x303F, "puntuacion CJK"),
    (0xFF00, 0xFFEF, "formas de anchura completa"),
]
#: Emoji y dingbats: aviso, no error.
RANGOS_AVISO = [
    (0x1F300, 0x1FAFF, "emoji"),
    (0x2600, 0x27BF, "dingbat"),
    (0x2B00, 0x2BFF, "simbolo"),
    (0x2190, 0x21FF, "flecha"),
    (0x2705, 0x2705, "marca de estado"),
]
#: Matematicas deliberadas que si se usan en los informes del proyecto.
PERMITIDOS = set("Δρσεβμ±≤≥≈×÷√∞°·—–«»¿¡áéíóúüñÁÉÍÓÚÜÑ")
MOJIBAKE = re.compile(r"[ÃÂâ€][\x80-\xbf\u0080-\u00bf]")

EXTENSIONES = {".py", ".md", ".txt", ".csv"}
#: El propio informe de idioma se excluye: cita los hallazgos y se auto-referenciaria.
EXCLUIR = {"__pycache__", ".git", "data", "notebooks", "features", "models", "submissions"}
EXCLUIR_FICHEROS = {"v13_auditoria_idioma.md"}


def clasifica(caracter: str) -> str:
    """Devuelve la categoria de un caracter: ERROR, AVISO u OK."""
    if caracter in PERMITIDOS or caracter in "\n\r\t":
        return "OK"
    punto = ord(caracter)
    for inicio, fin, nombre in RANGOS_PROHIBIDOS:
        if inicio <= punto <= fin:
            return f"ERROR:{nombre}"
    for inicio, fin, nombre in RANGOS_AVISO:
        if inicio <= punto <= fin:
            return f"AVISO:{nombre}"
    if unicodedata.category(caracter) in {"Cc", "Cs", "Co", "Cn"}:
        return "ERROR:control"
    return "OK"


def audita_ruta(ruta: Path) -> list[dict]:
    """Audita un fichero de texto y devuelve los hallazgos."""
    try:
        texto = ruta.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        return [{"fichero": str(ruta.relative_to(ROOT)), "linea": 0, "categoria": "ERROR:lectura",
                 "detalle": str(error)[:80]}]
    hallazgos = []
    for numero, linea in enumerate(texto.splitlines(), start=1):
        for caracter in linea:
            categoria = clasifica(caracter)
            if categoria != "OK":
                faltan = [h for h in hallazgos if h["linea"] == numero
                          and h["categoria"] == categoria]
                if not faltan:
                    hallazgos.append({
                        "fichero": str(ruta.relative_to(ROOT)), "linea": numero,
                        "categoria": categoria,
                        "detalle": unicodedata.name(caracter, "desconocido")[:60],
                        "contexto": linea.strip()[:90]})
        if MOJIBAKE.search(linea):
            hallazgos.append({"fichero": str(ruta.relative_to(ROOT)), "linea": numero,
                              "categoria": "ERROR:mojibake", "detalle": "secuencia UTF-8 corrupta",
                              "contexto": linea.strip()[:90]})
    return hallazgos


def main() -> int:
    """Audita todo el proyecto y escribe el informe."""
    hallazgos: list[dict] = []
    ficheros = 0
    for patron in ("**/*.py", "**/*.md", "**/*.txt", "**/*.csv"):
        for ruta in sorted(ROOT.glob(patron)):
            if any(parte in EXCLUIR for parte in ruta.parts):
                continue
            if ruta.suffix.lower() not in EXTENSIONES:
                continue
            if ruta.name in EXCLUIR_FICHEROS:
                continue
            ficheros += 1
            hallazgos += audita_ruta(ruta)

    errores = [h for h in hallazgos if h["categoria"].startswith("ERROR")]
    avisos = [h for h in hallazgos if h["categoria"].startswith("AVISO")]

    lineas = [
        "# V13 · P0a — Auditoría de idioma", "",
        "Comprueba que el código, la documentación y los datos derivados están en español y sin "
        "caracteres de otros alfabetos. Motivo: en la fase de lexicones se colaron ideogramas CJK, "
        "cirílicos y puntuación china en listas de palabras y docstrings; en V13 eso se detecta "
        "de forma automática.", "",
        f"- **Ficheros auditados**: {ficheros} (`.py`, `.md`, `.txt`, `.csv`)", 
        f"- **Errores**: {len(errores)} · **Avisos**: {len(avisos)}", "",
    ]
    if errores:
        lineas += ["## Errores (alfabeto no latino, mojibake o control)", "",
                   "| Fichero | Línea | Categoría | Detalle | Contexto |", "|---|---|---|---|---|"]
        for hallazgo in sorted(errores, key=lambda h: (h["fichero"], h["linea"])):
            lineas.append(f"| `{hallazgo['fichero']}` | {hallazgo['linea']} | "
                          f"**{hallazgo['categoria']}** | {hallazgo['detalle']} | "
                          f"{hallazgo['contexto']} |")
    else:
        lineas.append("**Sin errores**: no hay ideogramas, cirílicos, árabe, hebreo, devanagari, "
                      "tailandés, coreano, puntuación CJK ni mojibake en el proyecto.")
    if avisos:
        lineas += ["", "## Avisos (emoji y dingbats: no son otro idioma, pero ensucian)", "",
                   "| Fichero | Línea | Categoría | Contexto |", "|---|---|---|---|"]
        for hallazgo in sorted(avisos, key=lambda h: (h["fichero"], h["linea"]))[:60]:
            lineas.append(f"| `{hallazgo['fichero']}` | {hallazgo['linea']} | "
                          f"{hallazgo['categoria']} | {hallazgo['contexto']} |")
    lineas += ["", "## Regla aplicada en V13", "",
              "Todo el código, los comentarios, los docstrings, los informes y los nombres de "
              "variable se escriben en español sin caracteres de otros alfabetos. Los únicos "
              "signos no latinos permitidos son los matemáticos deliberados (delta, rho, sigma, "
              "epsilon, beta, mas/menos, menor-igual, mayor-igual) y la puntuación española.", ""]
    INFORME.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    RESULTADO.write_text(json.dumps({
        "fase": "V13 · P0a auditoria de idioma", "fecha": "2026-09-25",
        "ficheros_auditados": ficheros, "errores": len(errores), "avisos": len(avisos),
        "hallazgos": sorted(errores + avisos, key=lambda h: (h["fichero"], h["linea"])),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"ficheros auditados: {ficheros} | errores: {len(errores)} | avisos: {len(avisos)}")
    for hallazgo in sorted(errores, key=lambda h: (h["fichero"], h["linea"]))[:25]:
        print(f"  ERROR {hallazgo['fichero']}:{hallazgo['linea']} {hallazgo['categoria']} "
              f"| {hallazgo['contexto'][:60]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
