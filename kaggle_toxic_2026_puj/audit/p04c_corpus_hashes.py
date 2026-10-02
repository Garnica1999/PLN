"""P4c: verificacion de hashes de corpus procesados vs manifiesto."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROJ = Path("/home/carlos/Git/PLN-toxic-audit/kaggle_toxic_2026_puj")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads(
        (PROJ / "data" / "external" / "corpora" / "corpus_manifest.json").read_text())
    entradas = [e for e in manifest.get("corpus_construidos", []) if isinstance(e, dict)]
    mixto_raw = manifest.get("corpus_construidos_mixto", [])
    if isinstance(mixto_raw, dict):
        entradas.append(mixto_raw)
    else:
        no_dict = [e for e in mixto_raw if not isinstance(e, dict)]
        entradas += [e for e in mixto_raw if isinstance(e, dict)]
        if no_dict:
            (PROJ / "audit" / "out" / "p04c_mixto_no_dict.txt").write_text(
                "\n".join(str(e)[:300] for e in no_dict), encoding="ascii", errors="replace")
    resultados = []
    for entrada in entradas:
        ruta = entrada.get("processed_ruta") or ""
        esperado = (entrada.get("processed_sha256") or "").lower()
        path = PROJ / ruta
        if not ruta or not esperado or not path.exists():
            resultados.append({"id": entrada.get("id"), "archivo": ruta,
                               "estado": "AUSENTE_O_SIN_HASH"})
            continue
        real = sha256_of(path)
        resultados.append({"id": entrada.get("id"), "archivo": ruta,
                           "estado": "OK" if real == esperado else "MISMATCH",
                           "sha256_ok": real[:16], "sha256_manifest": esperado[:16],
                           "bytes_ok": path.stat().st_size,
                           "bytes_manifest": entrada.get("processed_bytes")})
    (PROJ / "audit" / "out" / "p04c_corpus_hashes.json").write_text(
        json.dumps(resultados, indent=2, ensure_ascii=True), encoding="ascii")
    for fila in resultados:
        print(fila["id"], fila["estado"], fila.get("bytes_ok"), fila.get("bytes_manifest"),
              fila.get("sha256_ok"), fila.get("sha256_manifest"))


if __name__ == "__main__":
    main()
