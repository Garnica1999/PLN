"""V54 · 02 — Envío oficial de la submission V54 Hierarchical Multi-Loss PoE Consensus a Kaggle.

Sube submissions/age_submission_v54_hierarchical_poe_consensus.csv a tweet-age-2026-puj,
monitorea el estado hasta completar y registra el score oficial en experiments/v54_02_kaggle_submission.json.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg
from src.verificacion_numerica import autotest_numerico

SUBMISSION_FILE = cfg.ROOT / "submissions" / "age_submission_v54_hierarchical_poe_consensus.csv"
RESULTADO = cfg.EXPERIMENTS / "v54_02_kaggle_submission.json"
COMPETITION = "tweet-age-2026-puj"
MENSAJE = "V54 Hierarchical Multi-Loss PoE Consensus (8-model Bayes consensus w=[0.5515, 0.0652, 0.0163, 0.1690, 0.0739, 0.0503, 0.0538, 0.0200]) + Plug-in"


def main() -> int:
    autotest_numerico()
    inicio = time.time()
    print("[V54 SUBMIT] Iniciando envío de V54 Hierarchical PoE Consensus a Kaggle...", flush=True)

    if not SUBMISSION_FILE.exists():
        print(f"[ERROR] No existe {SUBMISSION_FILE}", flush=True)
        return 1

    sha256 = hashlib.sha256(SUBMISSION_FILE.read_bytes()).hexdigest()
    with open(SUBMISSION_FILE, encoding="utf-8") as f:
        filas = sum(1 for _ in f) - 1
    print(f"[V54 SUBMIT] Archivo: {SUBMISSION_FILE.name} | Filas: {filas:,} | SHA256: {sha256}", flush=True)

    cred = Path.home() / ".kaggle" / "kaggle.json"
    if not cred.exists():
        print("[ERROR] No existe ~/.kaggle/kaggle.json", flush=True)
        return 1
    data = json.loads(cred.read_text())
    if "token" in data:
        os.environ["KAGGLE_API_TOKEN"] = data["token"]

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    print("[V54 SUBMIT] Autenticado en Kaggle API.", flush=True)

    print(f"[V54 SUBMIT] Enviando a {COMPETITION}...", flush=True)
    res = api.competition_submit(file_name=str(SUBMISSION_FILE), message=MENSAJE, competition=COMPETITION)
    print(f"[V54 SUBMIT] Respuesta: {res}", flush=True)

    # Esperar calificación
    print("[V54 SUBMIT] Esperando calificación en Kaggle...", flush=True)
    ref = None
    public_score = None
    status = None

    for intento in range(30):
        time.sleep(10)
        subs = api.competition_submissions(COMPETITION)
        if not subs:
            continue
        latest = subs[0]
        ref = getattr(latest, "ref", None)
        status = str(getattr(latest, "status", ""))
        public_score = getattr(latest, "public_score", None)
        print(f"  Intento {intento + 1}/30: ref={ref}, status={status}, public_score={public_score}", flush=True)

        if "COMPLETE" in status or (public_score is not None and str(public_score) != "None"):
            break
        if "ERROR" in status:
            print("[ERROR] Kaggle reportó error en la submission", flush=True)
            break

    duracion = round(time.time() - inicio, 2)
    print("\n" + "=" * 80)
    print(f"[V54 SUBMIT] RESULTADO OFICIAL KAGGLE:")
    print(f"  - Ref          : {ref}")
    print(f"  - Status       : {status}")
    print(f"  - Public Score : {public_score}")
    print(f"  - Duración     : {duracion}s")
    print("=" * 80, flush=True)

    resultado = {
        "fase": "V54-02",
        "fecha": time.strftime("%Y-%m-%d"),
        "archivo": str(SUBMISSION_FILE),
        "sha256": sha256,
        "filas": filas,
        "ref": ref,
        "status": status,
        "public_score": float(public_score) if public_score is not None else None,
        "mensaje": MENSAJE,
        "duracion_seg": duracion,
    }

    RESULTADO.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTADO, "w") as f:
        json.dump(resultado, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
