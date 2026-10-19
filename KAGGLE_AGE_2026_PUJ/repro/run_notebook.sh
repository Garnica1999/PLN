#!/bin/bash
# Runner de ejecucion del notebook final en el servidor (nbconvert + kernel del venv exacto).
# Uso: bash repro/run_notebook.sh <notebook.ipynb> [dir_trabajo]
# Politica de hardware: pre-vuelo; si esta ocupado, no ejecuta (exit 3). Nunca mata procesos.
set -u
NOTEBOOK="${1:?uso: run_notebook.sh <notebook.ipynb> [dir]}"
WORKDIR="${2:-$HOME/age_notebook/runs/notebook_exec}"
GPU=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
LOAD=$(cut -d' ' -f1 /proc/loadavg)
echo "PRE-VUELO [$(date -Iseconds)]: gpu_procs=$GPU load1=$LOAD"
if [ "${SKIP_GPU_CHECK:-0}" != "1" ] && [ "${GPU:-1}" != "0" ]; then
    echo "HARDWARE OCUPADO (GPU en uso): no ejecuto. Esperar y reintentar."
    exit 3
fi
LIBRE=$(awk -v l="$LOAD" 'BEGIN{print (l<6)?1:0}')
if [ "$LIBRE" != "1" ]; then
    echo "HARDWARE OCUPADO (CPU load1=$LOAD): no ejecuto. Esperar y reintentar."
    exit 3
fi
mkdir -p "$WORKDIR"
cp -f "$NOTEBOOK" "$WORKDIR/ejecucion.ipynb"
cd "$WORKDIR" || exit 2
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
echo "ejecutando $(basename "$NOTEBOOK") en $WORKDIR ..."
~/miniconda3/envs/ia_stable/bin/jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.kernel_name=age_exacto \
    --ExecutePreprocessor.timeout=43200 \
    --ExecutePreprocessor.allow_errors=False \
    --output ejecutado.ipynb ejecucion.ipynb
CODIGO=$?
echo "NOTEBOOK FIN: exit=$CODIGO"
exit $CODIGO
