#!/bin/bash
# Runner del spike de cadena completa (Fase 0) con el venv exacto del proyecto.
# Uso: bash repro/run_cadena.sh [etapas]   (default: features,members,meta)
# Politica de hardware: pre-vuelo; si esta ocupado, no ejecuta (exit 3). Nunca mata procesos.
set -u
cd ~/age_notebook/proyecto || exit 2
ETAPAS="${1:-features,members,meta}"
GPU=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
LOAD=$(cut -d' ' -f1 /proc/loadavg)
echo "PRE-VUELO [$(date -Iseconds)]: gpu_procs=$GPU load1=$LOAD etapas=$ETAPAS"
if [ "${SKIP_GPU_CHECK:-0}" != "1" ] && [ "${GPU:-1}" != "0" ]; then
    echo "HARDWARE OCUPADO (GPU en uso): no ejecuto. Esperar y reintentar."
    exit 3
fi
LIBRE=$(awk -v l="$LOAD" 'BEGIN{print (l<6)?1:0}')
if [ "$LIBRE" != "1" ]; then
    echo "HARDWARE OCUPADO (CPU load1=$LOAD): no ejecuto. Esperar y reintentar."
    exit 3
fi
echo "HARDWARE LIBRE: ejecutando spike de cadena ($ETAPAS)..."
PYBIN="${PYBIN:-$HOME/age_notebook/venv_exacto/bin/python}"
echo "PYTHON: $PYBIN"
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    "$PYBIN" repro/spike_cadena.py --etapas "$ETAPAS"
CODIGO=$?
echo "CADENA FIN ($ETAPAS): exit=$CODIGO"
exit $CODIGO
