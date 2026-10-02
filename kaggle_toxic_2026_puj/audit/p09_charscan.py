"""P9: escaneo independiente de caracteres (complementa tools/verificar_caracteres.py).

Categorias: emoji/dingbats, flechas, box drawing, geometricos, zero-width,
selectores de variacion, acentos latinos, otros simbolos, emoticonos ASCII.
Distingue artefactos generados del dataset fuente (data/raw) y de corpora externos.
Escribe audit/out/p09_charscan.csv y audit/out/p09_charscan_resumen.json.
"""
from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

PROJ = Path("/home/carlos/Git/PLN-toxic-audit/kaggle_toxic_2026_puj")
OUT = PROJ / "audit" / "out"
EXTENSIONES = {".md", ".py", ".csv", ".json", ".txt", ".ipynb"}
EXCLUIDOS = {".git", "__pycache__", "node_modules", "venv", "tools/fastText", "features", "modelos", "audit/out", "audit/out_p06"}

RANGOS = {
    "emoji_dingbats": [(0x1F300, 0x1FAFF), (0x2600, 0x27BF), (0x2B00, 0x2BFF), (0x1F000, 0x1F2FF)],
    "flechas": [(0x2190, 0x21FF), (0x2900, 0x297F)],
    "box_drawing": [(0x2500, 0x257F)],
    "geometricos": [(0x25A0, 0x25FF)],
    "zero_width": [(0x200B, 0x200F), (0x202A, 0x202E), (0x2060, 0x2064), (0xFEFF, 0xFEFF)],
    "selectores_variacion": [(0xFE00, 0xFE0F)],
    "acentos_latinos": [(0x00C0, 0x00FF), (0x0100, 0x017F), (0x1E00, 0x1EFF)],
    "otros_simbolos": [(0x2100, 0x214F), (0x2200, 0x22FF), (0x2460, 0x24FF), (0x2000, 0x200A), (0x2018, 0x201F), (0x2013, 0x2014)],
    "alfabetos_no_latinos": [(0x0370, 0x03FF), (0x0400, 0x04FF), (0x0590, 0x05FF), (0x0600, 0x06FF),
                             (0x0900, 0x097F), (0x0E00, 0x0E7F), (0x3000, 0x9FFF), (0xAC00, 0xD7AF)],
}
PERMITIDOS_MAT = {0x0394, 0x03B1, 0x03B2, 0x03B5, 0x03B9, 0x03BA, 0x03BB, 0x03BC,
                  0x03BD, 0x03C0, 0x03C3, 0x03C4, 0x2212, 0x2264, 0x2265, 0x00B1, 0x00D7}
EXCEPCIONES = {"reports/embedding_vocabulary_coverage.md", "reports/embedding_vocabulary_coverage.csv"}
EMOTICONOS = [":)", ":-)", ":D", ":-D", ";)", ";-)", ":P", ":-P", "xD", "XD", "T_T", "-_-", "^_^", "u_u", "<3"]


def categoria(ch: str) -> str | None:
    cp = ord(ch)
    if cp < 128:
        return None
    if cp in PERMITIDOS_MAT:
        return None
    for nombre, rangos in RANGOS.items():
        for inicio, fin in rangos:
            if inicio <= cp <= fin:
                return nombre
    return "otro_no_ascii"


def fold_ascii(text: str) -> str:
    repl = {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
            "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u00a0": " "}
    for k, v in repl.items():
        text = text.replace(k, v)
    norm = unicodedata.normalize("NFKD", text)
    return "".join(ch if ord(ch) < 128 else "?" for ch in norm if not unicodedata.combining(ch))


def main() -> None:  # noqa: PLR0915
    hallazgos: dict[tuple[str, str], dict] = {}
    emoticones: dict[str, list[str]] = defaultdict(list)
    archivos = 0
    for ruta in sorted(PROJ.rglob("*")):
        if not ruta.is_file() or ruta.suffix not in EXTENSIONES:
            continue
        rel = ruta.relative_to(PROJ).as_posix()
        if any(rel == ex or rel.startswith(ex + "/") for ex in EXCLUIDOS):
            continue
        if rel.startswith("data/") and not rel.startswith("data/raw"):
            continue
        try:
            texto = ruta.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        archivos += 1
        contadores: dict[str, int] = defaultdict(int)
        contextos: dict[str, str] = {}
        for pos, ch in enumerate(texto):
            cat = categoria(ch)
            if cat is None:
                continue
            contadores[cat] += 1
            if cat not in contextos:
                ini = max(0, pos - 30)
                contextos[cat] = texto[ini:pos + 31].replace("\n", " ")
        for cat, cuenta in contadores.items():
            fuente = "dataset" if rel.startswith("data/raw") else "generado"
            hallazgos[(rel, cat)] = {
                "archivo": rel, "categoria": cat, "conteo": cuenta,
                "fuente": fuente, "contexto": contextos.get(cat, ""),
            }
        for emo in EMOTICONOS:
            if emo in texto:
                emoticones[rel].append(emo)

    filas = []
    resumen: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for (rel, cat), datos in sorted(hallazgos.items()):
        filas.append({
            "archivo": rel, "categoria": cat, "conteo": datos["conteo"],
            "fuente": datos["fuente"], "contexto": datos["contexto"][:120],
        })
        resumen[cat][datos["fuente"]] += datos["conteo"]
    with (OUT / "p09_charscan.csv").open("w", newline="", encoding="ascii") as fh:
        writer = csv.DictWriter(fh, fieldnames=["archivo", "categoria", "conteo", "fuente", "contexto"])
        writer.writeheader()
        writer.writerows([{k: fold_ascii(str(v)) for k, v in f.items()} for f in filas])
    with (OUT / "p09_emoticones.csv").open("w", newline="", encoding="ascii") as fh:
        writer = csv.writer(fh)
        writer.writerow(["archivo", "emoticones"])
        for rel, lista in sorted(emoticones.items()):
            writer.writerow([rel, "|".join(sorted(set(lista)))])
    generados = {c: v.get("generado", 0) for c, v in resumen.items()}
    dataset = {c: v.get("dataset", 0) for c, v in resumen.items()}
    salida = {"archivos_escaneados": archivos, "generados": generados, "dataset": dataset,
              "archivos_con_emoticones": sorted(emoticones)}
    (OUT / "p09_charscan_resumen.json").write_text(
        json.dumps(salida, indent=2, ensure_ascii=True), encoding="ascii")
    print("archivos:", archivos)
    print("generados por categoria:", dict(generados))
    print("dataset por categoria:", dict(dataset))
    print("archivos con emoticonos:", len(emoticones))
    peores = sorted(filas, key=lambda f: -f["conteo"])[:12]
    for f in peores:
        print("  ", f["conteo"], f["categoria"], f["fuente"], f["archivo"])


if __name__ == "__main__":
    main()
