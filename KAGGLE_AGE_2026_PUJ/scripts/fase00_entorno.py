"""FASE 00 — Exploración dinámica del entorno de ejecución (portable).

Detecta intérprete de Python, CPU, RAM y GPU (dedicada o integrada) y escribe el perfil en
``config/hardware.json`` más el resumen legible en ``plan/00_maquina.md``.

Reglas que se derivan del perfil detectado (no se asumen):
    - intérprete: el primer Python funcional de la lista de candidatos (venv de Carlos por defecto);
    - hilos: ``n_jobs=-1`` y número de hilos lógicos para BLAS/torch;
    - GPU: se reserva solo si hay PyTorch con ``cuda`` disponible (esta fase de AGE es CPU pura,
      por lo que no se usa ninguna capacidad de GPU).

Ejecutar: ``/home/carlos/venv/bin/python scripts/fase00_entorno.py``
"""

from __future__ import annotations

import json
import platform
import re
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANDIDATOS_PYTHON = ["/home/carlos/venv/bin/python", ".venv/bin/python", "venv/bin/python"]
GPU_POR_DEFECTO = {"nombre": "ninguna", "tipo": "ninguna", "vram_gb": 0.0, "torch_cuda": False}


def detector_python() -> str:
    """Devuelve el primer intérprete utilizable de la lista de candidatos."""
    for candidato in CANDIDATOS_PYTHON:
        ruta = Path(candidato)
        if ruta.exists() and not ruta.is_dir():
            completa = ruta if ruta.is_absolute() else (ROOT / ruta)
            if completa.exists():
                return str(completa)
    return shutil.which("python3") or "python3"


def _normaliza_etiqueta(etiqueta: str) -> str:
    """Normaliza la etiqueta de ``lscpu``: minúsculas, sin acentos y sin comillas.

    Necesario porque ``lscpu`` localize al español escribe «Núcleo(s) por «socket»».
    """
    limpia = etiqueta.replace("«", "").replace("»", "").replace('"', "").strip().lower()
    return "".join(caracter for caracter in unicodedata.normalize("NFD", limpia)
                   if unicodedata.category(caracter) != "Mn").strip()


def informacion_cpu() -> dict:
    """Nombre del procesador, núcleos físicos e hilos lógicos.

    ``lscpu`` puede venir en inglés o en español (y entrecomilla los sockets), así que se comparan
    las etiquetas ya normalizadas en lugar de literales exactos.
    """
    nombre, nucleos_fisicos, hilos = platform.processor() or "desconocido", None, None
    try:
        salida = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=10).stdout
        valores: dict[str, int] = {}
        for linea in salida.splitlines():
            etiqueta, _, valor = linea.partition(":")
            etiqueta, valor = _normaliza_etiqueta(etiqueta), valor.strip()
            if not valor:
                continue
            if "model name" in etiqueta or "nombre del modelo" in etiqueta:
                nombre = valor
            elif "nucleo" in etiqueta and "socket" in etiqueta and "por" in etiqueta:
                valores["nucleos_por_socket"] = int(re.sub(r"[^0-9]", "", valor) or 0)
            elif "nucleo" in etiqueta and "por" in etiqueta:
                valores["hilos_por_nucleo"] = int(re.sub(r"[^0-9]", "", valor) or 0)
            elif etiqueta in {"socket(s)", "zocalo(s)"}:
                valores["sockets"] = int(re.sub(r"[^0-9]", "", valor) or 0)
            elif etiqueta == "cpu(s)":
                valores["hilos"] = int(re.sub(r"[^0-9]", "", valor) or 0)
        nucleos_fisicos = valores.get("nucleos_por_socket", 0) * valores.get("sockets", 0)
        hilos = (nucleos_fisicos * valores.get("hilos_por_nucleo", 1)) or valores.get("hilos")
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    hilos = hilos or os_cpu_count()
    if nucleos_fisicos is None and hilos:
        nucleos_fisicos = max(1, hilos // 2)
    return {"nombre": nombre, "nucleos_fisicos": nucleos_fisicos, "hilos": hilos}


def os_cpu_count() -> int | None:
    """Número de procesadores lógicos con degradación segura."""
    try:
        return len(getattr(__import__("os"), "sched_getaffinity")(0))
    except (AttributeError, OSError):
        return None


def informacion_ram() -> float:
    """RAM total en GB desde /proc/meminfo."""
    try:
        for linea in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if linea.startswith("MemTotal"):
                return round(int(re.sub(r"[^0-9]", "", linea)) / 1024 / 1024, 1)
    except (OSError, ValueError):
        pass
    return 0.0


def informacion_gpu() -> dict:
    """GPU dedicada o integrada: lspci + PyTorch. No lanza/downloads nada."""
    import torch

    if not torch.cuda.is_available():
        return dict(GPU_POR_DEFECTO)
    nombre = torch.cuda.get_device_name(0)
    total_gb = round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1)
    return {"nombre": nombre, "tipo": "dedicada", "vram_gb": total_gb, "torch_cuda": True}


def main() -> int:
    """Detecta el entorno, escribe ``config/hardware.json`` y ``plan/00_maquina.md``."""
    python = detector_python()
    cpu = informacion_cpu()
    ram = informacion_ram()
    try:
        gpu = informacion_gpu()
    except Exception:  # noqa: BLE001 - la detección de GPU no debe impedir la fase 00
        gpu = dict(GPU_POR_DEFECTO)
    perfil = {
        "fecha": "2026-09-25",
        "python": python,
        "version_python": platform.python_version(),
        "cpu": cpu,
        "ram_gb": ram,
        "gpu": gpu,
        "sistema": f"{platform.system()} {platform.release()}",
        "uso_en_esta_fase": "CPU pura: la fase de lexicones es de conteo y agregados + combinador "
                            "lineal; no se usa GPU (D-007).",
        "reglas_derivadas": {
            "n_jobs": -1,
            "hilos_blas": cpu["hilos"],
            "batch_na": "no aplica (sin redes neuronales en esta fase)",
        },
    }
    ruta_json = ROOT / "config" / "hardware.json"
    ruta_json.parent.mkdir(parents=True, exist_ok=True)
    ruta_json.write_text(json.dumps(perfil, indent=2, ensure_ascii=False), encoding="utf-8")

    lineas = [
        "# Fase 00 — Perfil del entorno de ejecución", "",
        f"- **Fecha**: {perfil['fecha']}",
        f"- **Intérprete canónico**: `{python}` (Python {perfil['version_python']})",
        f"- **CPU**: {cpu['nombre']} — {cpu['nucleos_fisicos']} núcleos físicos, "
        f"{cpu['hilos']} hilos lógicos",
        f"- **RAM**: {ram} GB",
        f"- **GPU**: {gpu['nombre']} ({gpu['tipo']}, {gpu['vram_gb']} GB, torch_cuda={gpu['torch_cuda']})",
        f"- **Sistema**: {perfil['sistema']}", "",
        "## Decisiones de recursos derivadas del perfil",
        f"- `n_jobs=-1` y {cpu['hilos']} hilos para BLAS/torch (CPUs con {cpu['hilos']} hilos).",
        "- **La fase de lexicones es CPU pura**: los agregados léxicos y el combinador lineal no "
        "requieren GPU (coherente con D-007).",
        "- No se instalan paquetes: el venv ya contiene nltk 3.9.4, pandas 2.3.3, scikit-learn 1.8.0 "
        "y scipy 1.17.1, que es todo lo necesario.",
        "", "_Generado por `scripts/fase00_entorno.py`; no asumir hardware sin reejecutarlo._",
    ]
    ruta_md = ROOT / "plan" / "00_maquina.md"
    ruta_md.parent.mkdir(parents=True, exist_ok=True)
    ruta_md.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"python: {python} | CPU: {cpu['nombre']} ({cpu['nucleos_fisicos']}/{cpu['hilos']}) "
          f"| RAM: {ram} GB | GPU: {gpu['nombre']}")
    print(f"escrito: config/hardware.json, plan/00_maquina.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
