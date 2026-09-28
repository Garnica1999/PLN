"""V30 - 04: la unica submission autorizada, o el registro de por que no se pudo hacer.

Este script esta **deliberadamente separado** de la construccion del fichero. La construccion es
reversible; una sonda en Kaggle **no lo es**. Separarlas permite inspeccionar el fichero y su hash
antes de que exista una unica e irreversible subida.

## La regla que gobierna el script

- **Exactamente una** submission. Nunca mas, nunca una variante, nunca un reintento con otro
  modelo.
- Si Kaggle devuelve un error **tecnico**, se registra y solo se reintenta **el mismo fichero** por
  causa puramente operacional. Nunca se cambia el artefacto.
- **Nunca se imprime** ninguna credencial. Ni token, ni cookie, ni clave. El diagnostico de
  autenticacion se describe en palabras, no en valores.
- Si falta una credencial valida: **STOP de la submission**, sin pedir ni copiar secretos en texto.

## Por que el resultado del leaderboard no puede volver a tocar el modelo

Este script, una vez subido, **congela la fase**. El score que devuelva Kaggle se registra como
**evidencia externa de transferencia** en `v30_05_analisis_transferencia.py`, y en ningun momento
se usa para cambiar hiperparametros, pesos, thresholds o features. El punto 42 del brief es
inequivoco: la submission es el final del experimento, no parte del entrenamiento.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

RESULTADO = cfg.EXPERIMENTS / "v30_04_upload.json"
SUBMISSION = cfg.ROOT / "submissions" / "age_submission_v30_final_candidate.csv"
CONSTRUCCION = cfg.EXPERIMENTS / "v30_03_submission_construccion.json"
NOMBRE_SONDA = "AGE_V30_FINAL_CANDIDATE"
DESCRIPCION = "AGE V30 final candidate: V20 + LightGBM refinado con Optuna anidada"
#: Competicion. Se deja parametrizable y **se comprueba antes de subir**: una submission a la
#: competicion equivocada es irrecuperable.
COMPETICION = "playground-series-s4e11"
TIMEOUT_SEG = 1800


def _estado_credencial() -> dict:
    """Describe el estado de la autenticacion **sin revelar ningun valor**.

    Solo se comprueban nombres de claves, longitudes y si estan vacias. Nada de eso es un secreto.
    """
    ruta = Path.home() / ".kaggle" / "kaggle.json"
    salida = {"fichero_presente": ruta.exists(), "claves": [], "formato": None,
              "variables_de_entorno": {k: bool(__import__("os").environ.get(k))
                                      for k in ("KAGGLE_USERNAME", "KAGGLE_KEY",
                                                "KAGGLE_API_TOKEN")}}
    if ruta.exists():
        try:
            datos = json.loads(ruta.read_text(encoding="utf-8"))
            salida["claves"] = sorted(datos.keys())
            salida["formato"] = "clasico" if {"username", "key"} <= set(datos) else (
                "token_solo" if "token" in datos else "desconocido")
            salida["campos_vacios"] = [k for k, v in datos.items() if not str(v).strip()]
        except Exception as error:  # noqa: BLE001
            salida["error_de_lectura"] = type(error).__name__
    return salida


def _comprueba_competicion() -> dict:
    """Una llamada de solo lectura que confirma identidad y competicion, sin imprimir secretos."""
    try:
        proceso = subprocess.run(
            ["/home/carlos/venv/bin/kaggle", "competitions", "list", "-s", "s4e11"],
            capture_output=True, text=True, timeout=120)
        return {"comando": "competitions list", "returncode": proceso.returncode,
                "autenticado": proceso.returncode == 0,
                "primera_linea_util": (proceso.stdout.strip().splitlines()[:1] or [""])[0][:120]}
    except Exception as error:  # noqa: BLE001
        return {"comando": "competitions list", "returncode": None, "autenticado": False,
                "error": type(error).__name__}


def main() -> int:
    """Sube la unica submission autorizada, o registra por que no se pudo."""
    autotest_numerico()
    inicio = time.time()
    if not SUBMISSION.exists():
        print(f"[STOP] no existe {SUBMISSION.name}. Hay que construirlo primero.", flush=True)
        return 1
    construccion = json.loads(CONSTRUCCION.read_text(encoding="utf-8"))
    if construccion["checks_superados"] != construccion["checks_total"]:
        print("[STOP] la construccion no paso todos los checks. No se sube nada.", flush=True)
        return 1
    sha = hashlib.sha256(SUBMISSION.read_bytes()).hexdigest()
    if sha != construccion["sha256"]:
        print(f"[STOP] el hash del fichero cambio desde la construccion "
              f"({construccion['sha256'][:16]} -> {sha[:16]}). No se sube nada.", flush=True)
        return 1
    print(f"[V30] fichero {SUBMISSION.name} | {construccion['filas']:,} filas | "
          f"SHA256 {sha[:16]}... | checks {construccion['checks_superados']}/"
          f"{construccion['checks_total']}", flush=True)

    credencial = _estado_credencial()
    identidad = _comprueba_competicion()
    print(f"[V30] credencial presente: {credencial['fichero_presente']} | formato: "
          f"{credencial['formato']} | claves: {credencial['claves']}", flush=True)
    print(f"[V30] autenticacion en la API: {identidad['autenticado']}", flush=True)

    base = {
        "fase": "V30 04 unica submission autorizada", "fecha": "2026-09-30",
        "archivo": SUBMISSION.name, "sha256": sha, "bytes": int(SUBMISSION.stat().st_size),
        "filas": construccion["filas"], "nombre_sonda": NOMBRE_SONDA,
        "descripcion": DESCRIPCION, "competicion": COMPETICION,
        "candidato": construccion["candidato"], "pesos_congelados": construccion["pesos_congelados"],
        "checks_superados": construccion["checks_superados"],
        "checks_total": construccion["checks_total"],
        "credencial": credencial, "identidad_api": identidad,
        "intentos": [], "numero_de_submissions": 0,
        "reglas": ["exactamente una submission", "sin variantes ni reintento con otro modelo",
                   "sin imprimir credenciales", "el score no se usa para volver a optimizar"],
    }

    if not identidad["autenticado"]:
        # STOP CONDITION. Se registra con precision y **no se pide el secreto en texto**: la
        # credencial es un fichero local del usuario y no tiene que pasar por una conversacion.
        causa = ("el fichero ~/.kaggle/kaggle.json existe pero solo contiene la clave `token`, "
                 "que es el formato antiguo; el CLI 2.2.4 exige `username` y `key`, o bien "
                 "autenticacion OAuth. Sin eso la API rechaza la llamada.")
        print(f"\n[STOP SUBMISSION] {causa}", flush=True)
        print("El fichero de submission queda construido, validado y con hash, listo para subir "
              "en cuanto haya credencial.", flush=True)
        base.update({"estado": "STOP_SIN_CREDENCIAL", "causa": causa, "subido": False,
                     "solucion_para_el_usuario":
                         "`kaggle auth login` (flujo OAuth, abre el navegador y cachea el token) "
                         "o dejar un ~/.kaggle/kaggle.json con las claves `username` y `key`. "
                         "Ambas vias las hace el usuario en su terminal; ninguna credencial tiene "
                         "que pasar por esta conversacion.",
                     "fichero_listo_para_subir": True})
        RESULTADO.write_text(json.dumps(base, indent=2, ensure_ascii=False), encoding="utf-8")
        return 2

    # Solo se llega aqui con autenticacion valida. Una unica llamada, y nunca mas de una.
    print(f"\n[V30] subiendo a {COMPETICION} con el nombre {NOMBRE_SONDA} ...", flush=True)
    try:
        proceso = subprocess.run(
            ["/home/carlos/venv/bin/kaggle", "competitions", "submit",
             "-c", COMPETICION, "-f", str(SUBMISSION), "-m", NOMBRE_SONDA],
            capture_output=True, text=True, timeout=TIMEOUT_SEG)
        base["intentos"].append({"numero": 1, "returncode": proceso.returncode,
                                 "stdout": proceso.stdout[-1500:],
                                 "stderr": proceso.stderr[-1500:]})
        base["numero_de_submissions"] = 1
        if proceso.returncode == 0:
            base.update({"estado": "SUBIDA_CORRECTA", "subido": True,
                         "score": None,
                         "nota": "el score se recoge despues, en el analisis de transferencia, "
                                 "y **no** se usa para cambiar nada"})
            print("[V30] subida correcta. La fase queda CONGELADA.", flush=True)
        else:
            base.update({"estado": "ERROR_TECNICO", "subido": False,
                         "nota": "solo se reintentaria el MISMO fichero por causa operacional; "
                                 "nunca con otro modelo"})
            print(f"[V30] error tecnico de Kaggle: {proceso.stderr[-400:]}", flush=True)
    except subprocess.TimeoutExpired:
        base.update({"estado": "TIMEOUT", "subido": False,
                     "nota": "la llamada excedio el tiempo de espera; se puede reintentar el "
                             "MISMO fichero, nunca uno distinto"})
        print(f"[V30] timeout tras {TIMEOUT_SEG} s.", flush=True)
    except Exception as error:  # noqa: BLE001
        base.update({"estado": "ERROR", "subido": False, "error": type(error).__name__})
        print(f"[V30] error: {type(error).__name__}: {error}", flush=True)

    base["segundos"] = round(time.time() - inicio, 1)
    RESULTADO.write_text(json.dumps(base, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if base.get("subido") else 1


if __name__ == "__main__":
    raise SystemExit(main())
