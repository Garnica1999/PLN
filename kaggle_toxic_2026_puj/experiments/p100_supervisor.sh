#!/usr/bin/env bash
# P100: supervisor detached del entrenamiento B1 (p100_11_preentrenar_b1.py).
#
# Contexto: los shells en segundo plano de la sesion del harness mueren
# cuando el servidor de opencode evicta/reinicia los "location services" del
# directorio (evento observado 2026-10-13 20:57; ciclo ~cada 30-60 min).
# Este supervisor corre DETACHED (setsid) fuera del arbol de procesos del
# servidor y reanuda el entrenamiento si el proceso muere sin completar.
# La reanudacion es la del propio script (estado atomico cada 1000 pasos);
# no cambia el pipeline ni ningun artefacto del experimento.
cd /home/carlos/Git/PLN/kaggle_toxic_2026_puj || exit 1
LOG=experiments/p100_supervisor.log
echo "[supervisor] inicio $(date)" >> "$LOG"
intentos=0
while true; do
  if [ -f reports/p100_b1_pretraining.json ]; then
    echo "[supervisor] COMPLETO $(date)" >> "$LOG"
    break
  fi
  if pgrep -f '^/home/carlos/venv/bin/python -u experiments/p100_11_preentrenar_b1\.py$' > /dev/null; then
    sleep 45
    continue
  fi
  intentos=$((intentos + 1))
  if [ "$intentos" -gt 30 ]; then
    echo "[supervisor] tope de reintentos ($intentos) $(date)" >> "$LOG"
    break
  fi
  echo "[supervisor] lanzando intento $intentos $(date)" >> "$LOG"
  /home/carlos/venv/bin/python -u experiments/p100_11_preentrenar_b1.py >> experiments/p100_pretraining.log 2>&1
  echo "[supervisor] intento $intentos termino rc=$? $(date)" >> "$LOG"
  sleep 15
done
