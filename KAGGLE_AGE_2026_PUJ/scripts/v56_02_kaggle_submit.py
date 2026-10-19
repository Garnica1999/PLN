"""V56 · 02 — Envío oficial de la submission V56 a Kaggle y verificación de leaderboard."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg

SUBMISSION = cfg.ROOT / "submissions" / "age_submission_v56_nonlinear_poe_consensus.csv"
RESULTADO = cfg.EXPERIMENTS / "v56_02_kaggle_submission.json"
COMPETITION = "tweet-age-2026-puj"
MENSAJE = "V56 Non-Linear Multi-Modal PoE Consensus (9-model Bayes consensus w=[0.5594, 0.0602, 0.0209, 0.1687, 0.0608, 0.0438, 0.0520, 0.0104, 0.0238]) + Plug-in"


def main() -> int:
    inicio = time.time()
    print(f"[V56-SUBMIT] Iniciando envío de {SUBMISSION.name} a Kaggle...", flush=True)

    if not SUBMISSION.exists():
        print(f"[ERROR] No existe {SUBMISSION}", file=sys.stderr)
        return 1

    cmd_submit = [
        "/home/carlos/venv/bin/kaggle",
        "competitions",
        "submit",
        "-c",
        COMPETITION,
        "-f",
        str(SUBMISSION),
        "-m",
        MENSAJE,
    ]

    print(f"[V56-SUBMIT] Ejecutando: {' '.join(cmd_submit)}", flush=True)
    res = subprocess.run(cmd_submit, capture_output=True, text=True)
    print(f"[V56-SUBMIT] Salida:\n{res.stdout}", flush=True)
    if res.stderr:
        print(f"[V56-SUBMIT] Stderr:\n{res.stderr}", flush=True)

    print("[V56-SUBMIT] Esperando procesamiento en Kaggle...", flush=True)
    time.sleep(15)

    ref = None
    score = None
    status = None

    for intento in range(12):
        cmd_list = ["/home/carlos/venv/bin/kaggle", "competitions", "submissions", COMPETITION]
        r_list = subprocess.run(cmd_list, capture_output=True, text=True)
        lines = r_list.stdout.strip().split("\n")
        for line in lines[2:]:
            parts = line.split()
            if len(parts) >= 6 and SUBMISSION.name in parts[1]:
                ref = int(parts[0])
                status = parts[4] if len(parts) >= 5 else "UNKNOWN"
                if "COMPLETE" in status:
                    score = float(parts[5]) if len(parts) >= 6 else None
                    break
        if score is not None:
            break
        print(f"  [Intento {intento+1}] Status: {status}, esperando 10s...", flush=True)
        time.sleep(10)

    print(f"[V56-SUBMIT] Resultado final: Ref={ref}, Status={status}, Score={score}", flush=True)

    import hashlib
    sha256 = hashlib.sha256(SUBMISSION.read_bytes()).hexdigest()

    resultado = {
        "fase": "V56-02",
        "fecha": "2026-10-18",
        "archivo": str(SUBMISSION),
        "sha256": sha256,
        "filas": 112892,
        "ref": ref,
        "status": status,
        "public_score": score,
        "mensaje": MENSAJE,
        "duracion_seg": round(time.time() - inicio, 1),
    }

    with open(RESULTADO, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2)

    return 0 if score is not None else 1


if __name__ == "__main__":
    sys.exit(main())
