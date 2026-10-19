#!/bin/bash
# Runner generico Fase 0: ejecuta repro/<NN>_*.py con la politica de hardware del servidor.
# Uso: bash repro/run_spike.sh <NN>   (p. ej. bash repro/run_spike.sh 04)
# Si el hardware esta ocupado por procesos ajenos, NO ejecuta (exit 3). Nunca mata procesos.
set -u
cd ~/age_notebook/proyecto || exit 2
SPIKE="${1:?uso: run_spike.sh <NN>}"
GPU=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
LOAD=$(cut -d' ' -f1 /proc/loadavg)
echo "PRE-VUELO [$(date -Iseconds)]: spike=$SPIKE gpu_procs=$GPU load1=$LOAD"
if [ "${SKIP_GPU_CHECK:-0}" != "1" ]; then
    if [ "${GPU:-1}" != "0" ]; then
        echo "HARDWARE OCUPADO (GPU en uso): no ejecuto. Esperar y reintentar."
        exit 3
    fi
fi
LIBRE=$(awk -v l="$LOAD" 'BEGIN{print (l<6)?1:0}')
if [ "$LIBRE" != "1" ]; then
    echo "HARDWARE OCUPADO (CPU load1=$LOAD): no ejecuto. Esperar y reintentar."
    exit 3
fi
echo "HARDWARE LIBRE: ejecutando spike $SPIKE..."
PYBIN="${PYBIN:-$HOME/miniconda3/envs/ia_stable/bin/python}"
echo "PYTHON: $PYBIN"
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
    "$PYBIN" repro/${SPIKE}_*.py
CODIGO=$?
echo "SPIKE $SPIKE FIN: exit=$CODIGO"
exit $CODIGO
