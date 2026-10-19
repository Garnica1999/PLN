"""V38 · 03 — Envío oficial de la submission V38 a Kaggle y seguimiento de puntuación.

Sube submissions/age_submission_v38_tabular_blend.csv a la competencia tweet-age-2026-puj,
monitorea el estado hasta SubmissionStatus.COMPLETE y registra el resultado en experiments/.
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

SUBMISSION_FILE = cfg.ROOT / "submissions" / "age_submission_v38_tabular_blend.csv"
RESULTADO = cfg.EXPERIMENTS / "v38_03_kaggle_submission.json"
COMPETITION = "tweet-age-2026-puj"
MENSAJE = "V38 TabularNet Blend (0.88 V30 + 0.12 TabNet ResNet) + Plug-in"


def main() -> int:
    autotest_numerico()
    inicio = time.time()
    print("[V38 SUBMIT] Iniciando proceso de envío a Kaggle...", flush=True)

    if not SUBMISSION_FILE.exists():
        print(f"[ERROR] No existe el archivo {SUBMISSION_FILE}", flush=True)
        return 1

    sha256 = hashlib.sha256(SUBMISSION_FILE.read_bytes()).hexdigest()
    with open(SUBMISSION_FILE, encoding="utf-8") as f:
        filas = sum(1 for _ in f) - 1
    print(f"[V38 SUBMIT] Archivo: {SUBMISSION_FILE.name} | Filas: {filas:,} | SHA256: {sha256}", flush=True)

    # Autenticación Kaggle
    cred = Path.home() / ".kaggle" / "kaggle.json"
    if not cred.exists():
        print("[ERROR] No existe ~/.kaggle/kaggle.json", flush=True)
        return 1
    os.environ["KAGGLE_API_TOKEN"] = json.loads(cred.read_text())["token"]

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    print("[V38 SUBMIT] Autenticado con éxito en Kaggle API.", flush=True)

    print(f"[V38 SUBMIT] Enviando archivo a {COMPETITION}...", flush=True)
    res = api.competition_submit(file_name=str(SUBMISSION_FILE), message=MENSAJE, competition=COMPETITION)
    print(f"[V38 SUBMIT] Respuesta de envío: {res}", flush=True)

    # Esperar puntuación
    print("[V38 SUBMIT] Esperando procesamiento en Kaggle...", flush=True)
    score = None
    status = None
    ref = None

    for intento in range(30):
        time.sleep(10)
        subs = api.competition_submissions(COMPETITION)
        if not subs:
            continue
        latest = subs[0]
        status = str(latest.status)
        score = latest.public_score
        ref = latest.ref
        print(f"  Intento {intento+1}: Ref {ref} | Estado {status} | Score {score}", flush=True)
        if "COMPLETE" in status and score not in (None, "", "None"):
            break

    duracion = round(time.time() - inicio, 1)
    record = {
        "fase": "V38 03 Kaggle Submission",
        "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
        "archivo": SUBMISSION_FILE.name,
        "sha256": sha256,
        "competition": COMPETITION,
        "message": MENSAJE,
        "ref": str(ref),
        "status": status,
        "public_score": float(score) if score and score != "None" else None,
        "duracion_seg": duracion
    }
    RESULTADO.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[V38 SUBMIT] ¡FINALIZADO en {duracion}s! Public Score: {score}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
