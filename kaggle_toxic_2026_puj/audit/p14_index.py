"""P14: indice final de los CSV de la auditoria (para la entrega)."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

PROJ = Path("/home/carlos/Git/PLN-toxic-audit/kaggle_toxic_2026_puj")


def sha16(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def filas_csv(path: Path) -> int:
    try:
        with path.open(newline="", encoding="utf-8", errors="replace") as fh:
            return max(0, sum(1 for _ in csv.reader(fh)) - 1)
    except OSError:
        return -1


def main() -> None:
    objetivos = sorted((PROJ / "reports").glob("audit_*.csv")) + \
        sorted((PROJ / "audit" / "out").glob("*.csv"))
    filas = []
    for path in objetivos:
        filas.append({
            "csv": str(path.relative_to(PROJ)),
            "filas": filas_csv(path),
            "sha256_16": sha16(path),
            "bytes": path.stat().st_size,
        })
    salida = PROJ / "reports" / "audit_csv_index.csv"
    with salida.open("w", newline="", encoding="ascii") as fh:
        writer = csv.DictWriter(fh, fieldnames=["csv", "filas", "sha256_16", "bytes"])
        writer.writeheader()
        writer.writerows(filas)
    print("CSVs indexados:", len(filas))
    for f in filas:
        print(f"  {f['filas']:>5} filas  {f['csv']}")


if __name__ == "__main__":
    main()
