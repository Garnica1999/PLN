"""V43 · 03 — Envío oficial de la submission V43 Cramer-Smooth Blend a Kaggle.

Sube submissions/age_submission_v43_cramer_blend.csv a tweet-age-2026-puj,
monitorea el estado hasta completar y registra el score oficial.
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

SUBMISSION_FILE = cfg.ROOT / "submissions" / "age_submission_v43_cramer_blend.csv"
RESULTADO = cfg.EXPERIMENTS / "v43_03_kaggle_submission.json"
COMPETITION = "tweet-age-2026-puj"
MENSAJE = "V43 Cramer-Smooth Blend (0.555 V30 + 0.085 Cramer-SWA + 0.175 Smooth-SWA + 0.08 SE + 0.05 TabOrd + 0.055 HGB) + Plug-in"


def main() -> int:
    autotest_numerico()
    inicio = time.time()
    print("[V43 SUBMIT] Iniciando envío de V43 Cramer-Smooth Blend a Kaggle...", flush=True)

    if not SUBMISSION_FILE.exists():
        print(f"[ERROR] No existe {SUBMISSION_FILE}", flush=True)
        return 1

    sha256 = hashlib.sha256(SUBMISSION_FILE.read_bytes()).hexdigest()
    with open(SUBMISSION_FILE, encoding="utf-8") as f:
        filas = sum(1 for _ in f) - 1
    print(f"[V43 SUBMIT] Archivo: {SUBMISSION_FILE.name} | Filas: {filas:,} | SHA256: {sha256}", flush=True)

    cred = Path.home() / ".kaggle" / "kaggle.json"
    if not cred.exists():
        print("[ERROR] No existe ~/.kaggle/kaggle.json", flush=True)
        return 1
    os.environ["KAGGLE_API_TOKEN"] = json.loads(cred.read_text())["token"]

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    print("[V43 SUBMIT] Autenticado en Kaggle API.", flush=True)

    print(f"[V43 SUBMIT] Enviando a {COMPETITION}...", flush=True)
    res = api.competition_submit(file_name=str(SUBMISSION_FILE), message=MENSAJE, competition=COMPETITION)
    print(f"[V43 SUBMIT] Respuesta: {res}", flush=True)

    print("[V43 SUBMIT] Esperando procesamiento en Kaggle...", flush=True)
    score = None
    status = None
    ref_id = None

    for _ in range(30):
        time.sleep(10)
        subs = api.competition_submissions(COMPETITION)
        if subs:
            latest = subs[0]
            ref_id = str(getattr(latest, "ref", ""))
            status = getattr(latest, "status", "")
            score_str = getattr(latest, "public_score", None)
            print(f"  Consulta status={status} publicScore={score_str} ref={ref_id}", flush=True)
            if "complete" in str(status).lower():
                score = float(score_str)
                break
            elif "error" in str(status).lower():
                print(f"[ERROR] La sumisión falló en Kaggle: {getattr(latest, 'error_description', '')}")
                return 1

    duracion = time.time() - inicio
    print(f"\n[V43 SUBMIT] ¡Procesamiento completado!")
    print(f"  Score obtenido (Macro-F1) : {score}")
    print(f"  Referencia Kaggle         : {ref_id}")
    print(f"  Tiempo total              : {duracion:.1f}s")

    resultado = {
        "fase": "V43-03",
        "fecha": time.strftime("%Y-%m-%d"),
        "duracion_seg": round(duracion, 2),
        "archivo": str(SUBMISSION_FILE),
        "sha256": sha256,
        "filas": filas,
        "mensaje": MENSAJE,
        "competencia": COMPETITION,
        "kaggle_ref": ref_id,
        "public_score": score,
        "status": str(status),
    }

    with open(RESULTADO, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2)

    print(f"[V43 SUBMIT] Resultado registrado en {RESULTADO}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
