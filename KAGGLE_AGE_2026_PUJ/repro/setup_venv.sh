#!/bin/bash
# Setup del venv exacto del notebook AGE en el servidor (Python 3.14, versiones del proyecto).
# Uso: bash repro/setup_venv.sh   (en el servidor; NO mata procesos; no toca otros entornos)
set -e
VENV="$HOME/age_notebook/venv_exacto"
echo "=== reconstruccion limpia del venv exacto ==="
rm -rf "$VENV"
/usr/bin/python3.14 -m venv "$VENV"
V="$VENV/bin"
"$V/pip" install --upgrade pip
echo "--- pip listo ---"
"$V/pip" install numpy==1.26.4 pandas==2.3.3 scipy==1.17.1 scikit-learn==1.8.0 \
    lightgbm==4.6.0 pyarrow==24.0.0 matplotlib==3.10.9 ipykernel==7.2.0
echo "--- stack sklearn instalado ---"
"$V/pip" install torch==2.14.0 || { echo 'torch==2.14.0 no disponible; instalando torch generico'; "$V/pip" install torch; }
echo "--- torch instalado ---"
"$V/python" -c "import numpy,pandas,scipy,sklearn,lightgbm,pyarrow,matplotlib,torch; print('numpy',numpy.__version__,'| pandas',pandas.__version__,'| scipy',scipy.__version__,'| sklearn',sklearn.__version__,'| lightgbm',lightgbm.__version__,'| torch',torch.__version__,'| cuda?',torch.cuda.is_available())"
echo "=== venv exacto LISTO ==="
