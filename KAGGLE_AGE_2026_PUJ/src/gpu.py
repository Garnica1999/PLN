"""Guardián de la GPU dedicada (decisión D-007).

Regla operativa: ningún paso del proyecto usa la GPU sin verificar antes que está libre.
Si está ocupada, el módulo NO inicia trabajo, NO interrumpe procesos ajenos y devuelve el
control informando la situación (modo espera acotado y opcional).

Detección (en este orden):
1. ``rocm-smi --showpids`` → señal primaria de cómputo (PIDs en la cola KFD).
2. ``rocm-smi --showuse --showmemuse`` → uso y VRAM contra umbrales calibrados sobre la
   línea base medida del escritorio (≈7% uso / ≈13% VRAM el 2026-09-22).
3. ``torch.cuda.mem_get_info()`` → confirmación de VRAM libre a nivel de dispositivo.

Nota: escanear handles de ``/dev/kfd`` produce falsos positivos (Plasma, Firefox, Steam
mantienen el dispositivo abierto sin cargar cómputo), por eso no se usa como señal.
"""

from __future__ import annotations

import fcntl
import os
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

#: Umbrales calibrados sobre la línea base del escritorio (ver D-007).
USE_THRESHOLD_PCT = 25.0
VRAM_THRESHOLD_PCT = 40.0
CONFIRMATIONS = 3
CONFIRMATION_INTERVAL_S = 5.0
LOCK_FILE = Path.home() / ".cache" / "kaggle_age_2026_puj_gpu.lock"

ROCM_SMI = shutil.which("rocm-smi") or "/opt/rocm/bin/rocm-smi"


class GpuBusyError(RuntimeError):
    """La GPU está siendo usada por otro proceso: se devuelve el control al usuario."""


@dataclass
class GpuStatus:
    """Estado observado de la GPU."""

    free: bool
    reason: str
    compute_pids: list[str] = field(default_factory=list)
    processes: list[str] = field(default_factory=list)
    use_pct: float | None = None
    vram_pct: float | None = None
    vram_free_gb: float | None = None
    source: str = "rocm-smi"
    timestamp: float = field(default_factory=time.time)

    def summary(self) -> str:
        """Descripción legible de una línea para logs."""
        return (
            f"free={self.free} reason={self.reason} pids={self.compute_pids} "
            f"use={self.use_pct}% vram={self.vram_pct}% vram_libre={self.vram_free_gb}GB"
        )

    def to_dict(self) -> dict:
        """Serializa el estado a diccionario."""
        return asdict(self)


def _run(cmd: list[str], timeout: int = 20) -> str:
    """Ejecuta un comando de solo lectura y devuelve su salida combinada."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return (result.stdout or "") + (result.stderr or "")
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return f"__error__ {exc}"


def _parse_pids(output: str) -> tuple[list[str], list[str]]:
    """Extrae PIDs de cómputo y descripciones de proceso de la salida de ``rocm-smi``."""
    if "No KFD PIDs currently running" in output:
        return [], []
    pids: list[str] = []
    processes: list[str] = []
    for line in output.splitlines():
        match = re.match(r"\s*(\d+)\s+(\d+)\s+(\S+)\s+(.*)", line)
        if match:
            pids.append(match.group(1))
            processes.append(f"{match.group(4).strip()} (pid={match.group(1)})")
    return pids, processes


def _parse_pct(output: str, marker: str) -> float | None:
    """Extrae un porcentaje de la salida de ``rocm-smi`` dado su marcador textual."""
    for line in output.splitlines():
        if marker in line and ":" in line:
            tail = line.split(":", 1)[1]
            match = re.search(r"(\d+(?:\.\d+)?)", tail)
            if match:
                return float(match.group(1))
    return None


def gpu_status(confirmations: int = CONFIRMATIONS,
               interval_s: float = CONFIRMATION_INTERVAL_S) -> GpuStatus:
    """Inspecciona la GPU sin modificar nada.

    La señal de cómputo (PIDs KFD) es inmediata. Los umbrales de uso/VRAM requieren
    ``confirmations`` lecturas separadas ``interval_s`` segundos para descartar picos
    transitorios del escritorio.

    Returns
    -------
    GpuStatus
        Estado de la GPU con el motivo de la decisión.
    """
    if not Path(ROCM_SMI).exists():
        return GpuStatus(free=False, reason="rocm-smi no disponible: no se puede verificar la GPU",
                         source="unavailable")

    pid_output = _run([ROCM_SMI, "--showpids"])
    if pid_output.startswith("__error__"):
        return GpuStatus(free=False, reason=f"no se pudo consultar rocm-smi: {pid_output}",
                         source="error")

    pids, processes = _parse_pids(pid_output)
    own_pid = str(os.getpid())
    foreign = [pid for pid in pids if pid != own_pid]

    vram_free_gb = None
    try:  # confirmación opcional de VRAM (solo si torch y la GPU están operativos)
        import torch

        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            vram_free_gb = round(free_bytes / 1024 ** 3, 2)
    except Exception:  # noqa: BLE001 - la ausencia de torch no invalida el chequeo
        vram_free_gb = None

    if foreign:
        return GpuStatus(free=False, reason="hay procesos de cómputo activos en la GPU",
                         compute_pids=foreign, processes=processes, vram_free_gb=vram_free_gb)

    # Sin PIDs de cómputo: se confirman uso y VRAM contra la línea base del escritorio.
    highest_use = 0.0
    highest_vram = 0.0
    for index in range(max(1, confirmations)):
        use_output = _run([ROCM_SMI, "--showuse", "--showmemuse"])
        use_pct = _parse_pct(use_output, "GPU use") or 0.0
        vram_pct = _parse_pct(use_output, "VRAM%") or _parse_pct(use_output, "VRAM") or 0.0
        highest_use = max(highest_use, use_pct)
        highest_vram = max(highest_vram, vram_pct)
        if use_pct >= USE_THRESHOLD_PCT or vram_pct >= VRAM_THRESHOLD_PCT:
            return GpuStatus(free=False, reason=f"uso elevado sostenido (uso={use_pct}%, vram={vram_pct}%)",
                             use_pct=use_pct, vram_pct=vram_pct, vram_free_gb=vram_free_gb)
        if index < confirmations - 1:
            time.sleep(interval_s)

    return GpuStatus(free=True, reason="sin cómputo ajeno y por debajo de umbrales",
                     use_pct=highest_use, vram_pct=highest_vram, vram_free_gb=vram_free_gb)


class GpuLease:
    """Context manager que mantiene el lock de GPU mientras dure el trabajo."""

    def __init__(self, status: GpuStatus) -> None:
        self.status = status
        self._handle = None

    def __enter__(self) -> "GpuLease":
        LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        self._handle = LOCK_FILE.open("w")
        fcntl.flock(self._handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self._handle.write(f"pid={os.getpid()} {self.status.summary()}\n")
        self._handle.flush()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._handle is not None:
            fcntl.flock(self._handle, fcntl.LOCK_UN)
            self._handle.close()


def require_gpu(wait: int = 0, poll_s: float = 30.0) -> GpuLease:
    """Devuelve un ``GpuLease`` si la GPU está libre; si no, lanza ``GpuBusyError``.

    Parameters
    ----------
    wait : int
        Número máximo de reintentos (cada ``poll_s`` segundos). ``0`` (por defecto) no
        espera: devuelve el control inmediatamente. La espera nunca es indefinida.
    poll_s : float
        Segundos entre reintentos cuando ``wait > 0``.

    Raises
    ------
    GpuBusyError
        Si la GPU está ocupada tras agotar los reintentos; el mensaje incluye el estado
        completo para que el usuario decida.
    """
    status = gpu_status()
    attempts = 0
    while not status.free and attempts < max(0, wait):
        attempts += 1
        time.sleep(poll_s)
        status = gpu_status()
    if not status.free:
        raise GpuBusyError(
            "GPU OCUPADA: se devuelve el control al usuario y se entra en modo espera. "
            f"Estado: {status.summary()}"
        )
    return GpuLease(status)


def log_event(message: str) -> None:
    """Añade un evento de GPU al log de experimentos."""
    from src import config as cfg

    log_path = cfg.EXPERIMENTS / "gpu_events.log"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{message}\n")


if __name__ == "__main__":
    info = gpu_status()
    print(info.summary())
