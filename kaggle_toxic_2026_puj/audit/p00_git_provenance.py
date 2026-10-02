"""P0: provenance git, hashes de referencia y entorno para la auditoria TOXIC.

Script de auditoria (no modifica el repo fuente). Escribe en
/home/carlos/Git/PLN-toxic-audit/kaggle_toxic_2026_puj/reports/.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import unicodedata
from importlib import metadata
from pathlib import Path

REPO = Path("/home/carlos/Git/PLN")
AUDIT_PROJ = Path("/home/carlos/Git/PLN-toxic-audit/kaggle_toxic_2026_puj")
REPORTS = AUDIT_PROJ / "reports"
PROJ = "kaggle_toxic_2026_puj"

SURFACES = {
    "A_actual": Path("/home/carlos/Git/PLN") / PROJ,
    "B_v7": Path("/home/carlos/Git/PLN-v7") / PROJ,
    "C_v8": Path("/home/carlos/Git/PLN-v8") / PROJ,
    "D_audit": AUDIT_PROJ,
}

BRANCHES = [
    "proyecto-kaggle-toxic-2026",
    "proyecto-kaggle-toxic-2026-v2",
    "proyecto-kaggle-toxic-2026-v4",
    "proyecto-kaggle-toxic-2026-v5",
    "proyecto-kaggle-toxic-2026-v6",
    "proyecto-kaggle-toxic-2026-v7",
    "proyecto-kaggle-toxic-2026-v8",
    "proyecto-kaggle-toxic-2026-audit",
    "main",
    "kaggle-age-2026-puj-v13",
]

STRAY_AGE = {"c8e220f", "f68637c", "32ab2f2"}


def ascii_text(text: str) -> str:
    """Normaliza a ASCII: pliega acentos y reemplaza comillas/dashes tipograficos."""
    repl = {
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u00a0": " ",
        "\u2192": "->", "\u2265": ">=", "\u2264": "<=", "\u00d7": "x",
    }
    for k, v in repl.items():
        text = text.replace(k, v)
    norm = unicodedata.normalize("NFKD", text)
    out = "".join(ch for ch in norm if not unicodedata.combining(ch))
    return "".join(ch if ord(ch) < 128 else "?" for ch in out)


def git(*args: str) -> str:
    res = subprocess.run(
        ["git", "-C", str(REPO), *args],
        capture_output=True, text=True, check=True,
    )
    return res.stdout


def sha256_file(path: Path) -> tuple[int, str]:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return path.stat().st_size, h.hexdigest()


def classify(relpath: str) -> str:
    p = relpath
    if p.startswith("experiments/baseline_v03_frozen/"):
        return "artifact"
    if p.startswith("experiments/") and p.endswith(".py"):
        return "script"
    if p.startswith("experiments/") and p.endswith(".log"):
        return "log"
    if p.startswith("experiments/"):
        return "experiments_other"
    if p.startswith("reports/figuras/"):
        return "figure"
    if p.startswith("reports/"):
        return "report"
    if p.startswith("submissions/"):
        return "submission"
    if p.startswith("notebooks/"):
        return "notebook"
    if p.startswith("src/"):
        return "code"
    if p.startswith("docs/"):
        return "doc"
    if p.startswith("tasks/"):
        return "task_plan"
    if p.startswith("tools/"):
        return "tool"
    if p.startswith("config/"):
        return "config"
    if p.startswith("data/"):
        return "data"
    return "root_doc"


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)

    # --- 1. Inventario de commits que tocan el proyecto (todas las refs) ---
    raw = git(
        "log", "--all", "--reverse", "--name-status", "--date=short",
        "--pretty=format:C|%H|%h|%ad|%s", "--", PROJ,
    )
    commits: list[dict] = []
    current: dict | None = None
    for line in raw.splitlines():
        if line.startswith("C|"):
            _, full, short, date, subject = line.split("|", 4)
            current = {
                "full": full, "short": short, "date": date,
                "subject": ascii_text(subject), "files": [],
            }
            commits.append(current)
        elif line.strip() and current is not None:
            parts = line.split("\t")
            status = parts[0]
            path = parts[-1] if status.startswith(("R", "C")) else parts[1]
            current["files"].append((status, path))

    # Fase por posicion respecto a los tips decorados en el log ordenado.
    tips_order = [
        ("abd4ec4", "v2"), ("5d06ce7", "v4"), ("a719153", "v5"),
        ("d370977", "v6"), ("2f8fac8", "v7"), ("b0edf7a", "v8"),
    ]
    phase_by_commit: dict[str, str] = {}
    idx = 0
    for commit in commits:
        if idx < len(tips_order) and commit["short"] == tips_order[idx][0]:
            label = tips_order[idx][1]
            idx += 1
        elif commit["short"] == "bb87158":
            label = "v1-inicial"
        else:
            label = tips_order[idx][1] if idx < len(tips_order) else "post-v8"
        if commit["short"] in STRAY_AGE:
            label = "age-v13(stray)"
        elif label == "v5" and "v5.1" in commit["subject"]:
            label = "v5.1"
        phase_by_commit[commit["short"]] = label

    # --- 2. Primer/ultimo commit por ruta ---
    path_stats: dict[str, dict] = {}
    for commit in commits:
        for status, path in commit["files"]:
            rel = path.removeprefix(PROJ + "/") if path.startswith(PROJ) else path
            st = path_stats.setdefault(rel, {
                "first": commit["short"], "first_date": commit["date"],
                "last": commit["short"], "last_date": commit["date"],
                "n": 0, "statuses": set(), "phases": set(),
            })
            st["last"] = commit["short"]
            st["last_date"] = commit["date"]
            st["n"] += 1
            st["statuses"].add(status.split("\t")[0][:1] if not status.startswith(("R", "C")) else status[:1])
            st["phases"].add(phase_by_commit[commit["short"]])

    extra_untracked = [
        "features/oof_v03_final.npz",
        "features/v2_structured_features.parquet",
        "tools/fastText",
        "tools/verificar_caracteres.py",
        "data/raw/toxic_train.csv",
        "data/raw/toxic_test.csv",
    ]
    for rel in extra_untracked:
        st = path_stats.setdefault(rel, {
            "first": "", "first_date": "", "last": "", "last_date": "",
            "n": 0, "statuses": set(), "phases": set(),
        })
        st["statuses"].add("u")

    prov_rows = []
    for rel, st in sorted(path_stats.items()):
        presence = []
        for name, root in SURFACES.items():
            if (root / rel).exists():
                presence.append(name.split("_")[0])
        notes = ""
        if "age-v13(stray)" in st["phases"]:
            notes = "tocado en commits de la sesion AGE v13 (fuera de las fases toxic)"
        prov_rows.append({
            "path": rel,
            "kind": classify(rel),
            "first_commit": st["first"],
            "first_date": st["first_date"],
            "last_commit": st["last"],
            "last_date": st["last_date"],
            "n_commits": st["n"],
            "change_types": "".join(sorted(st["statuses"])),
            "phases": "|".join(sorted(st["phases"])),
            "present_in": "".join(presence),
            "notes": notes,
        })
    with (REPORTS / "audit_git_provenance.csv").open("w", newline="", encoding="ascii") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(prov_rows[0].keys()))
        writer.writeheader()
        writer.writerows(prov_rows)

    # --- 3. Tabla de ramas ---
    worktrees = {}
    wt_raw = git("worktree", "list", "--porcelain")
    wt_path = ""
    for line in wt_raw.splitlines():
        if line.startswith("worktree "):
            wt_path = line.split(" ", 1)[1]
        elif line.startswith("branch refs/heads/"):
            worktrees[line.split("refs/heads/", 1)[1]] = wt_path
    branch_rows = []
    for br in BRANCHES:
        try:
            full = git("rev-parse", br).strip()
        except subprocess.CalledProcessError:
            continue
        info = git("log", "-1", "--date=short", "--format=%h|%ad|%s", br).strip().split("|", 2)
        n_toxic = git("rev-list", "--count", br, "--", PROJ).strip()
        try:
            git("rev-parse", "--verify", f"origin/{br}")
            pushed = "si"
        except subprocess.CalledProcessError:
            pushed = "no"
        branch_rows.append({
            "branch": br,
            "tip": info[0],
            "tip_date": info[1],
            "tip_subject": ascii_text(info[2]),
            "commits_toxic": n_toxic,
            "pushed_to_origin": pushed,
            "worktree": worktrees.get(br, ""),
        })

    # --- 4. Estado de cada superficie ---
    surface_rows = []
    for name, root in SURFACES.items():
        if not root.exists():
            continue
        wt = next((p for b, p in worktrees.items() if Path(p) == root.parent), "")
        try:
            branch = subprocess.run(
                ["git", "-C", str(root), "branch", "--show-current"],
                capture_output=True, text=True, check=True).stdout.strip()
        except subprocess.CalledProcessError:
            branch = ""
        try:
            dirty = subprocess.run(
                ["git", "-C", str(root.parent), "status", "--porcelain", "--", root.name],
                capture_output=True, text=True, check=True).stdout.strip()
        except subprocess.CalledProcessError:
            dirty = "?"
        exp = root / "experiments.csv"
        n_rows = ""
        if exp.exists():
            with exp.open(newline="", encoding="utf-8") as fh:
                n_rows = str(sum(1 for _ in csv.reader(fh)) - 1)
        nb = root / "notebooks" / "10_final_pipeline.ipynb"
        nb_info = ""
        if nb.exists():
            size, digest = sha256_file(nb)
            nb_info = f"{size}B sha256:{digest[:16]}"
        surface_rows.append({
            "surface": name,
            "path": str(root),
            "branch": branch or ("[worktree audit]" if name == "D_audit" else ""),
            "experiments_csv_rows": n_rows,
            "notebook": nb_info,
            "uncommitted": ascii_text(dirty) or "limpio",
        })

    # --- 5. Hashes de referencia ---
    hash_targets: dict[str, list[str]] = {
        "data/raw/toxic_train.csv": ["A_actual"],
        "data/raw/toxic_test.csv": ["A_actual"],
        "data/raw/submission.csv": ["A_actual"],
        "data/raw/tweet-toxic-2026-puj.zip": ["A_actual"],
        "data/external/corpora/corpus_manifest.json": ["A_actual"],
        "features/oof_v03_final.npz": ["B_v7", "C_v8", "D_audit"],
        "features/v2_structured_features.parquet": ["B_v7", "C_v8", "D_audit"],
        "experiments/baseline_v03_frozen/model_v03.joblib": ["B_v7", "C_v8"],
        "experiments/baseline_v03_frozen/oof_v03_historical.npz": ["B_v7", "C_v8"],
        "experiments/baseline_v03_frozen/pipeline_v03.json": ["B_v7", "C_v8"],
        "experiments/baseline_v03_frozen/submission_v03.csv": ["B_v7", "C_v8"],
        "experiments/baseline_v03_frozen/hashes.txt": ["B_v7", "C_v8"],
        "notebooks/10_final_pipeline.ipynb": ["A_actual", "B_v7", "C_v8"],
        "experiments.csv": ["A_actual", "B_v7", "C_v8"],
        "DECISIONS.md": ["A_actual", "B_v7", "C_v8"],
        "PROJECT_STATE.md": ["A_actual", "B_v7", "C_v8"],
        "RUNLOG.md": ["A_actual", "B_v7", "C_v8"],
        "FINAL_REPORT.md": ["A_actual", "B_v7", "C_v8"],
        "src/data.py": ["A_actual", "B_v7", "C_v8"],
        "src/splits.py": ["A_actual", "B_v7", "C_v8"],
        "src/evaluation.py": ["A_actual", "B_v7", "C_v8"],
        "tools/verificar_caracteres.py": ["A_actual", "B_v7", "C_v8"],
        "README.md": ["A_actual", "B_v7", "C_v8"],
    }
    for name, root in SURFACES.items():
        subs = root / "submissions"
        if subs.is_dir():
            for f in sorted(subs.glob("*.csv")):
                hash_targets.setdefault(f"submissions/{f.name}", []).append(name)

    hash_rows = []
    for rel, surfaces in sorted(hash_targets.items()):
        for surface in sorted(set(surfaces)):
            root = SURFACES[surface]
            path = root / rel
            if not path.exists():
                hash_rows.append({"surface": surface, "path": rel, "exists": "no",
                                  "bytes": "", "mtime": "", "sha256": ""})
                continue
            size, digest = sha256_file(path)
            hash_rows.append({
                "surface": surface, "path": rel, "exists": "si",
                "bytes": size,
                "mtime": subprocess.run(
                    ["date", "-d", f"@{path.stat().st_mtime:.0f}", "+%Y-%m-%d %H:%M:%S"],
                    capture_output=True, text=True, check=True).stdout.strip(),
                "sha256": digest,
            })
    with (REPORTS / "audit_hashes_reference.csv").open("w", newline="", encoding="ascii") as fh:
        writer = csv.DictWriter(fh, fieldnames=["surface", "path", "exists", "bytes", "mtime", "sha256"])
        writer.writeheader()
        writer.writerows(hash_rows)

    # Diferencias entre superficies para rutas comunes
    by_path: dict[str, dict[str, str]] = {}
    for row in hash_rows:
        if row["exists"] == "si":
            by_path.setdefault(row["path"], {})[row["surface"]] = row["sha256"]
    diffs = []
    for rel, shas in sorted(by_path.items()):
        if len(set(shas.values())) > 1:
            diffs.append((rel, shas))

    # --- 6. Entorno ---
    libs = {}
    for lib in ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "xgboost",
                "catboost", "nltk", "optuna", "joblib", "pyarrow", "matplotlib"]:
        try:
            libs[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            libs[lib] = "AUSENTE"
    try:
        import torch  # noqa: PLC0415
        torch_info = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "ninguno",
        }
    except Exception as exc:  # noqa: BLE001
        torch_info = {"error": str(exc)}
    mem_kb = 0
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                mem_kb = int(line.split()[1])
    except OSError:
        pass
    env = {
        "fecha": subprocess.run(["date", "-Iseconds"], capture_output=True, text=True, check=True).stdout.strip(),
        "python": platform.python_version(),
        "python_exe": "/home/carlos/venv/bin/python",
        "os": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor(),
        "ram_gb": round(mem_kb / 1024 / 1024, 1),
        "librerias": libs,
        "gpu": torch_info,
        "head_repo": git("rev-parse", "HEAD").strip(),
    }
    (AUDIT_PROJ / "audit" / "env_audit.json").write_text(
        json.dumps(env, indent=2, ensure_ascii=True), encoding="ascii")

    # --- 7. Narrativa MD ---
    lines = [
        "# P0 - Provenance git y estado del repositorio (borrador denso generado por script)",
        "",
        "Generado por `audit/p00_git_provenance.py`. Todos los datos son read-only sobre",
        "`/home/carlos/Git/PLN`. Este documento se refina manualmente en la fase P0.",
        "",
        "## Commits que tocan el proyecto (todas las refs)",
        f"- Total de commits que tocan `{PROJ}/`: {len(commits)}",
        f"- Primer commit: {commits[0]['short']} {commits[0]['date']} {commits[0]['subject']}",
        f"- Ultimo commit: {commits[-1]['short']} {commits[-1]['date']} {commits[-1]['subject']}",
        "",
        "## Distribucion por fase",
    ]
    phase_counts: dict[str, int] = {}
    for commit in commits:
        phase_counts[phase_by_commit[commit["short"]]] = phase_counts.get(
            phase_by_commit[commit["short"]], 0) + 1
    for phase, count in sorted(phase_counts.items()):
        lines.append(f"- {phase}: {count} commits")
    lines += [
        "",
        "## Ramas",
        "",
        "| Rama | Tip | Fecha | Commits que tocan el proyecto | En origin | Worktree |",
        "|---|---|---|---|---|---|",
    ]
    for row in branch_rows:
        lines.append(
            f"| {row['branch']} | {row['tip']} | {row['tip_date']} | {row['commits_toxic']} "
            f"| {row['pushed_to_origin']} | {row['worktree'] or '-'} |")
    lines += [
        "",
        "## Superficies auditadas",
        "",
        "| Superficie | Ruta | Rama | Filas experiments.csv | Notebook | Sin commitear |",
        "|---|---|---|---|---|---|",
    ]
    for row in surface_rows:
        lines.append(
            f"| {row['surface']} | {row['path']} | {row['branch']} | {row['experiments_csv_rows']} "
            f"| {row['notebook']} | {row['uncommitted']} |")
    lines += ["", "## Archivos que difieren entre superficies (hashes de referencia)", ""]
    if diffs:
        for rel, shas in diffs:
            detail = "; ".join(f"{s}={v[:12]}" for s, v in sorted(shas.items()))
            lines.append(f"- {rel}: {detail}")
    else:
        lines.append("- Ninguna diferencia entre superficies para las rutas de referencia.")
    lines += [
        "",
        "## Stash",
        "- Existe 1 stash: `stash@{0}: On kaggle-age-2026-puj-lexicons: WIP AGE lexicones fases 36-41`.",
        "  Pertenece al proyecto AGE, no al toxic. NO se toca.",
        "",
        "## Incidencias de commits cruzados (sesion AGE v13)",
        "- c8e220f, f68637c, 32ab2f2 tocaron `kaggle_toxic_2026_puj/tools/` desde la sesion AGE:",
        "  f68637c retira del indice `tools/fastText` (gitlink anadido por `git add -A`);",
        "  32ab2f2 retira del indice `tools/verificar_caracteres.py` (213 lineas, queda en disco sin versionar).",
        "",
        "## Entorno",
        f"- Python {env['python']} ({env['python_exe']}), {env['os']}, RAM {env['ram_gb']} GB",
        f"- GPU: {torch_info}",
        f"- Librerias: {json.dumps(libs, ensure_ascii=True)}",
        "",
        "## Hashes de referencia",
        "- Ver `audit_hashes_reference.csv` (sha256 por superficie).",
        "",
    ]
    (REPORTS / "audit_git_provenance.md").write_text("\n".join(lines), encoding="ascii")

    print(f"commits={len(commits)} paths={len(prov_rows)} branches={len(branch_rows)}")
    print(f"surfaces={[r['surface'] for r in surface_rows]}")
    print(f"md_lines={len(lines)} diffs={len(diffs)} hash_rows={len(hash_rows)}")
    for rel, shas in diffs[:20]:
        print("DIFF", rel, shas)


if __name__ == "__main__":
    main()
