"""P10: auditoria de seguridad - busqueda de secretos/credenciales (redactado).

Escanea el arbol actual del proyecto y la historia git del path toxic. Nunca
imprime el valor del secreto: solo archivo/patron/ubicacion.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

PROJ = Path("/home/carlos/Git/PLN-toxic-audit/kaggle_toxic_2026_puj")
REPO = Path("/home/carlos/Git/PLN")
OUT = PROJ / "audit" / "out"

PATRONES = {
    "kaggle_token": r"KGAT_[A-Za-z0-9]{10,}",
    "github_pat": r"(ghp_|github_pat_)[A-Za-z0-9_]{20,}",
    "aws_key": r"(AKIA|ASIA)[A-Z0-9]{16}",
    "openai_key": r"sk-[A-Za-z0-9]{20,}",
    "slack": r"xox[baprs]-[A-Za-z0-9-]{10,}",
    "google_api": r"AIza[0-9A-Za-z\-_]{30,}",
    "private_key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "generic_password": r"(?i)(password|passwd|contrasena|contrasena_)\s*[:=]\s*[\"'][^\"']{6,}[\"']",
    "generic_secret": r"(?i)(api[_-]?key|secret[_-]?key|access[_-]?token)\s*[:=]\s*[\"'][^\"']{8,}[\"']",
}
EXCLUIDOS = {".git", "__pycache__", "node_modules", "venv", "audit", "data", "features", "modelos", "tools/fastText"}


def escanear_arbol() -> list[dict]:
    hallazgos = []
    for ruta in sorted(PROJ.rglob("*")):
        if not ruta.is_file():
            continue
        rel = ruta.relative_to(PROJ).as_posix()
        if any(rel == ex or rel.startswith(ex + "/") for ex in EXCLUIDOS):
            continue
        if ruta.suffix.lower() not in {".py", ".md", ".json", ".csv", ".txt", ".ipynb", ".yml", ".yaml", ".toml", ".cfg", ".ini", ".env"}:
            continue
        try:
            texto = ruta.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for nombre, patron in PATRONES.items():
            for m in re.finditer(patron, texto):
                linea = texto[:m.start()].count("\n") + 1
                hallazgos.append({"ambito": "arbol", "archivo": rel, "patron": nombre,
                                  "linea": linea})
    return hallazgos


def escanear_historia() -> list[dict]:
    log = subprocess.run(
        ["git", "-C", str(REPO), "log", "--all", "-p", "--format=C|%h|%s",
         "--", "kaggle_toxic_2026_puj"],
        capture_output=True, text=True, check=True).stdout
    hallazgos = []
    commit = ""
    for linea in log.splitlines():
        if linea.startswith("C|"):
            commit = linea.split("|", 2)[1]
            continue
        if linea.startswith("+++") or linea.startswith("---"):
            continue
        if not linea.startswith("+"):
            continue
        for nombre, patron in PATRONES.items():
            if re.search(patron, linea):
                hallazgos.append({"ambito": "historia", "commit": commit, "patron": nombre})
    return hallazgos


def main() -> None:
    arbol = escanear_arbol()
    historia = escanear_historia()
    resultado = {"arbol": arbol, "historia": historia}
    (OUT / "p10_security.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=True), encoding="ascii")
    print("arbol:", len(arbol), "hallazgos")
    for h in arbol[:10]:
        print("  ", h["archivo"], "|", h["patron"], "| linea", h["linea"])
    print("historia:", len(historia), "hallazgos")
    for h in historia[:10]:
        print("  ", h["commit"], "|", h["patron"])
    if not arbol and not historia:
        print("SIN SECRETOS DETECTADOS (patrones escaneados: "
              + ", ".join(PATRONES) + ")")


if __name__ == "__main__":
    main()
