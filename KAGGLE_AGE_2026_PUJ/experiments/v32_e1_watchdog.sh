#!/usr/bin/env bash
# V32-E1 - watchdog de relanzamiento bajo el guardian D-007.
# Espera (acotado) a que el proceso p75_01 del otro proyecto termine Y la GPU
# quede libre de forma estable (2 lecturas consecutivas del guardian), y
# entonces ejecuta el runner v32_01. Hasta 2 intentos de lanzamiento.
# No mata procesos; solo observa y lanza cuando el protocolo lo permite.
set -u
cd /home/carlos/Git/PLN/KAGGLE_AGE_2026_PUJ
VENV=/home/carlos/venv/bin/python
LOG=experiments/v32_e1_run2.log
say() { echo "[watchdog $(date +%H:%M:%S)] $*"; }

libre() { "$VENV" -c "import sys; sys.path.insert(0,'.'); from src import gpu; sys.exit(0 if gpu.gpu_status().free else 1)"; }
p75_vivo() { pgrep -f "p75_01_complementariedad[_]textcnn" >/dev/null; }

fin=$(( $(date +%s) + 5400 ))
say "inicio (limite 90 min); p75_vivo=$(p75_vivo && echo si || echo no)"
intento=0
while [ "$intento" -lt 2 ]; do
  intento=$((intento+1))
  ok=0
  while [ "$(date +%s)" -lt "$fin" ]; do
    if ! p75_vivo && libre; then ok=$((ok+1)); else ok=0; fi
    if [ "$ok" -ge 2 ]; then break; fi
    sleep 45
  done
  if [ "$ok" -lt 2 ]; then
    say "timeout: la GPU no quedo libre (p75_vivo=$(p75_vivo && echo si || echo no))"
    exit 9
  fi
  say "GPU libre estable (p75 ausente); lanzando runner, intento $intento"
  "$VENV" scripts/v32_01_surrogate_macrof1.py 2>&1 | tee "$LOG"
  rc=${PIPESTATUS[0]}
  say "runner termino rc=$rc"
  if [ "$rc" -ne 3 ]; then exit "$rc"; fi
  say "rc=3: la GPU volvio a ocuparse; espero otra ventana (60 min)"
  sleep 120
  fin=$(( $(date +%s) + 3600 ))
done
say "dos intentos agotados sin exito"
exit 10
