"""P2: forense de experiments.csv.

- Parseo robusto por superficie (A=128, B=188, C=202 filas esperadas).
- Filas malformadas (campos != 23), ids duplicados, ids ausentes.
- Commit de introduccion y commits de modificacion por fila (git).
- Cobertura: script existe, id citado en reports/docs, relacion con submissions.
- Guiones huerfanos (scripts sin fila que los referencie).

Escribe reports/audit_experiment_coverage.csv y audit/p02_findings.json.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

AUDIT_PROJ = Path("/home/carlos/Git/PLN-toxic-audit/kaggle_toxic_2026_puj")
REPO = Path("/home/carlos/Git/PLN")
PROJ = "kaggle_toxic_2026_puj"
SURFACES = {
    "A": Path("/home/carlos/Git/PLN") / PROJ,
    "B": Path("/home/carlos/Git/PLN-v7") / PROJ,
    "C": Path("/home/carlos/Git/PLN-v8") / PROJ,
}
TIPS = [
    ("abd4ec4", "v2"), ("5d06ce7", "v4"), ("a719153", "v5"),
    ("d370977", "v6"), ("2f8fac8", "v7"), ("b0edf7a", "v8"),
]
STRAY = {"c8e220f", "f68637c", "32ab2f2"}


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args],
                          capture_output=True, text=True, check=True).stdout


def parse_rows(path: Path) -> tuple[list[str], list[list[str]], list[int]]:
    raw = list(csv.reader(path.open(newline="", encoding="utf-8")))
    header = raw[0]
    rows = []
    bad = []
    for idx, row in enumerate(raw[1:], start=2):
        if len(row) != len(header):
            bad.append(idx)
        rows.append(row)
    return header, rows, bad


def row_hash(row: list[str], width: int) -> str:
    padded = (row + [""] * width)[:width]
    return hashlib.sha256("\x1f".join(padded).encode("utf-8")).hexdigest()[:16]


def phase_of(short: str, seq: list[str]) -> str:
    """Fase de un commit segun posicion respecto a los tips (secuencia cronologica)."""
    idx = 0
    for c in seq:
        if idx < len(TIPS) and c == TIPS[idx][0]:
            if c == short:
                return "age-v13(stray)" if short in STRAY else TIPS[idx][1]
            idx += 1
        if c == short:
            if short in STRAY:
                return "age-v13(stray)"
            if short == "bb87158":
                return "v1-inicial"
            return TIPS[idx][1] if idx < len(TIPS) else "post-v8"
    return "desconocida"


def main() -> None:  # noqa: PLR0915
    findings: dict = {}
    parsed = {}
    for tag, root in SURFACES.items():
        header, rows, bad = parse_rows(root / "experiments.csv")
        parsed[tag] = {"header": header, "rows": rows, "bad": bad}
    findings["conteos"] = {
        tag: {"filas": len(parsed[tag]["rows"]),
              "filas_malformadas": len(parsed[tag]["bad"]),
              "filas_malformadas_lineas": parsed[tag]["bad"]}
        for tag in parsed
    }
    width = len(parsed["C"]["header"])
    header = parsed["C"]["header"]
    id_col = header.index("experiment_id")
    script_col = header.index("script")

    # ---------- ids por superficie ----------
    ids_by_tag: dict[str, list[str]] = {
        tag: [(r + [""] * width)[id_col] for r in parsed[tag]["rows"]]
        for tag in parsed
    }
    findings["ids_duplicados"] = {
        tag: sorted({i for i in ids if i and ids.count(i) > 1})
        for tag, ids in ids_by_tag.items()
    }
    findings["ids_vacios"] = {tag: sum(1 for i in ids if not i) for tag, ids in ids_by_tag.items()}

    # ---------- introduccion por commit (git) ----------
    log = git("log", "--all", "--reverse", "--date=short",
              "--pretty=format:C|%H|%h|%ad|%s", "--", f"{PROJ}/experiments.csv")
    commits = []
    for line in log.splitlines():
        if line.startswith("C|"):
            _, full, short, date, subject = line.split("|", 4)
            commits.append({"full": full, "short": short, "date": date, "subject": subject})
    seq = git("log", "--all", "--reverse", "--pretty=format:%h", "--",
              PROJ).splitlines()
    intro: dict[str, tuple[str, str]] = {}
    changes: dict[str, list[str]] = {}
    removed_rows: list[tuple[str, int, str]] = []
    prev_state: dict[str, tuple[str, str]] = {}
    states: list[dict] = []
    for commit in commits:
        blob = git("show", f"{commit['full']}:{PROJ}/experiments.csv")
        rows = list(csv.reader(blob.splitlines()))
        state: dict[str, str] = {}
        for row in rows[1:]:
            padded = (row + [""] * width)[:width]
            the_id = padded[id_col]
            key = the_id if the_id else f"<vacio@{row_hash(padded, width)}>"
            state[key] = row_hash(padded, width)
        states.append({"commit": commit["short"], "state": state})
        for key, digest in state.items():
            if key not in prev_state:
                intro[key] = (commit["short"], commit["date"])
            elif prev_state[key] != digest:
                changes.setdefault(key, []).append(commit["short"])
        for key in prev_state:
            if key not in state:
                removed_rows.append((commit["short"], prev_state[key], key))
        prev_state = state
    findings["commits_sobre_experiments_csv"] = len(commits)
    findings["filas_introducidas_por_fase"] = {}
    for key, (short, _date) in intro.items():
        ph = phase_of(short, seq)
        findings["filas_introducidas_por_fase"][ph] = (
            findings["filas_introducidas_por_fase"].get(ph, 0) + 1)
    findings["filas_eliminadas_en_alguna_revision"] = removed_rows
    findings["filas_modificadas_tras_introduccion"] = {
        k: v for k, v in sorted(changes.items())}

    # ---------- lectura robusta por fila (superficie C como canon) ----------
    c_rows = []
    c_meta = []
    bad_counter = 0
    for raw_row in parsed["C"]["rows"]:
        malformed = len(raw_row) != width
        padded = (raw_row + [""] * width)[:width]
        row = dict(zip(header, padded, strict=True))
        if malformed:
            bad_counter += 1
            row["_bad_id"] = f"BAD-{bad_counter:02d}"
            row["_bad_fields"] = str(len(raw_row))
        c_rows.append(row)
        c_meta.append(malformed)

    # ---------- textos de reportes para busqueda de ids ----------
    search_text = []
    search_files = []
    for sub in ("reports", "docs", "tasks"):
        for path in sorted((SURFACES["C"] / sub).glob("**/*")):
            if path.is_file() and path.suffix.lower() in {".md", ".json", ".csv", ".txt"}:
                search_files.append(path)
    for name in ("DECISIONS.md", "PROJECT_STATE.md", "RUNLOG.md", "FINAL_REPORT.md", "README.md"):
        path = SURFACES["C"] / name
        if path.exists():
            search_files.append(path)
    for path in search_files:
        try:
            search_text.append(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            pass
    big = "\n".join(search_text)
    findings["archivos_texto_indexados"] = len(search_files)

    # ---------- scripts existentes y huerfanos ----------
    scripts_disk = {p.name: str(p.relative_to(SURFACES["C"]))
                    for p in (SURFACES["C"] / "experiments").glob("*.py")}
    referenced: set[str] = set()
    coverage_rows = []
    for row, malformed in zip(c_rows, c_meta, strict=True):
        if malformed:
            exp_id = row["_bad_id"]
            status_bits = [f"FILA_MALFORMADA({row['_bad_fields']}campos;"
                           f"id_original={row['experiment_id']})"]
            original_id = row["experiment_id"]
        else:
            exp_id = row["experiment_id"] or "<vacio>"
            original_id = exp_id
            status_bits = []
            if not row["experiment_id"]:
                status_bits.append("ID_VACIO")
            elif ids_by_tag["C"].count(exp_id) > 1:
                status_bits.append(f"ID_DUPLICADO(x{ids_by_tag['C'].count(exp_id)})")
        script_field = row["script"].strip()
        candidates = re.findall(r"experiments/[\w\-.]+\.py", script_field)
        missing = [c for c in candidates if not (SURFACES["C"] / c).exists()]
        for c in candidates:
            referenced.add(Path(c).name)
        in_report = bool(re.search(rf"(?<![A-Za-z0-9-]){re.escape(original_id)}(?![A-Za-z0-9-])", big)) \
            if original_id and not original_id.startswith("BAD-") else False
        sub_related = bool(re.search(r"submission", script_field, re.I)) or bool(
            re.search(r"submission", row.get("notes", ""), re.I))
        row_text = " ".join(row.values())
        art_refs = re.findall(
            r"(?:reports|submissions)/(?:[\w\-]+/)*[\w\-]+\.(?:md|json|csv|png)", row_text)
        missing_art = sorted({a for a in art_refs if not (SURFACES["C"] / a).exists()})
        artifact_exists = ("si" if art_refs and not missing_art
                           else ("faltan:" + "|".join(missing_art) if missing_art else "no_citado"))
        is_manual_probe = "submissions/" in script_field and not candidates
        if not malformed:
            if not candidates and not is_manual_probe and script_field.lower() not in {"", "n/a", "no aplica", "-"}:
                status_bits.append("SIN_RUTA_DE_CODIGO")
            if missing:
                status_bits.append("SCRIPT_AUSENTE:" + ",".join(Path(m).name for m in missing))
            if missing_art:
                status_bits.append("ARTEFACTO_CITADO_AUSENTE:" + ",".join(Path(m).name for m in missing_art))
        coverage_rows.append({
            "Experiment_ID": exp_id,
            "In_experiments_csv": ("ABC" if original_id in ids_by_tag["A"]
                                   and original_id in ids_by_tag["B"] and not malformed
                                   else ("BC" if original_id in ids_by_tag["B"] and not malformed
                                         else ("C" if not malformed else "B/C(malformada)"))),
            "In_report": "si" if in_report else "no",
            "Code_exists": ("n/a_sonda_manual" if is_manual_probe else
                            ("si" if candidates and not missing
                             else ("no" if missing else ("n/a" if script_field.lower() in {"", "n/a", "no aplica", "-"} else "sin_ruta")))),
            "Artifact_exists": artifact_exists,
            "Notebook_cell": "pendiente",
            "Submission_related": "si" if sub_related else "no",
            "Reproduced": "pendiente",
            "Numerically_consistent": "pendiente",
            "Decision_consistent": "pendiente",
            "Status": "OK" if not status_bits else "REVIEW(" + ";".join(status_bits) + ")",
        })
    orphans = sorted(set(scripts_disk) - referenced)
    findings["scripts_sin_fila_que_los_referencie"] = orphans
    findings["n_scripts_en_disco_C"] = len(scripts_disk)
    findings["huerfanos_citados_en_texto"] = {name: (name in big) for name in orphans}

    # ---------- huecos de numeracion por familia ----------
    fam_nums: dict[str, list[int]] = {}
    for key in intro:
        match = re.match(r"^([A-Z]{1,6})-?(\d+)$", key)
        if match:
            fam_nums.setdefault(match.group(1), []).append(int(match.group(2)))
    gaps = {}
    for fam, nums in fam_nums.items():
        nums = sorted(set(nums))
        missing = [n for n in range(min(nums), max(nums) + 1) if n not in nums]
        if missing:
            gaps[fam] = missing
    findings["huecos_numeracion"] = gaps

    # ---------- filas malformadas: cuarentena ----------
    quarantine = []
    for tag in ("B", "C"):
        for line_no, row in zip(parsed[tag]["bad"], [
                r for r in parsed[tag]["rows"] if len(r) != width], strict=False):
            quarantine.append({"surface": tag, "line": line_no, "fields": len(row),
                               "preview": " | ".join(c[:60] for c in row[:4])})
    findings["filas_cuarentena"] = quarantine

    # ---------- escritura ----------
    out = AUDIT_PROJ / "reports" / "audit_experiment_coverage.csv"
    with out.open("w", newline="", encoding="ascii") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(coverage_rows[0].keys()))
        writer.writeheader()
        writer.writerows(coverage_rows)
    with (AUDIT_PROJ / "audit" / "out" / "p02_introduction_map.csv").open(
            "w", newline="", encoding="ascii") as fh:
        writer = csv.writer(fh)
        writer.writerow(["experiment_id", "intro_commit", "intro_date", "fase", "modificado_en"])
        for key, (short, date) in sorted(intro.items()):
            writer.writerow([key, short, date, phase_of(short, seq),
                             "|".join(changes.get(key, []))])
    with (AUDIT_PROJ / "audit" / "p02_findings.json").open("w", encoding="ascii") as fh:
        json.dump(findings, fh, indent=2, ensure_ascii=True, default=str)

    print("conteos:", findings["conteos"])
    print("ids duplicados:", findings["ids_duplicados"])
    print("ids vacios:", findings["ids_vacios"])
    print("commits sobre experiments.csv:", len(commits))
    print("filas introducidas por fase:", findings["filas_introducidas_por_fase"])
    print("filas eliminadas:", len(removed_rows))
    print("filas modificadas tras introduccion:", len(changes))
    print("filas en cuarentena:", len(quarantine))
    print("scripts en disco:", len(scripts_disk), "sin referencia:", len(orphans))
    print("huerfanos (primeros 20):", orphans[:20])
    sin_reporte = [r for r in coverage_rows if r["In_report"] == "no"]
    sin_codigo = [r for r in coverage_rows if r["Status"] != "OK"]
    print("filas sin cita en reports:", len(sin_reporte))
    print("filas con Status != OK:", len(sin_codigo))
    for r in sin_codigo[:15]:
        print("  ", r["Experiment_ID"], r["Status"])


if __name__ == "__main__":
    main()
