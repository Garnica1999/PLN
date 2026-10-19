#!/bin/bash
# Runner Fase 0 - spike 01 (reproducibilidad de dup_clusters + folds canonicos).
# Politica de hardware del servidor: verifica GPU/CPU antes de ejecutar; si esta
# ocupado por otros procesos, NO ejecuta y sale con codigo 3 (esperar y reintentar).
# Nunca mata procesos ajenos.
set -u
cd ~/age_notebook/proyecto || exit 2
GPU=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
LOAD=$(cut -d' ' -f1 /proc/loadavg)
FECHA=$(date -Iseconds)
echo "PRE-VUELO [$FECHA]: gpu_procs=$GPU load1=$LOAD"
if [ "${GPU:-1}" != "0" ]; then
    echo "HARDWARE OCUPADO (GPU en uso): no ejecuto. Esperar y reintentar."
    exit 3
fi
LIBRE=$(awk -v l="$LOAD" 'BEGIN{print (l<6)?1:0}')
if [ "$LIBRE" != "1" ]; then
    echo "HARDWARE OCUPADO (CPU load1=$LOAD): no ejecuto. Esperar y reintentar."
    exit 3
fi
echo "HARDWARE LIBRE: ejecutando spike 01..."
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
    ~/miniconda3/envs/ia_stable/bin/python repro/01_dup_clusters_folds.py
CODIGO=$?
echo "SPIKE01 FIN: exit=$CODIGO"
exit $CODIGO
