# Historial de bugs - E2 Juris-PLN

## 2026-09-10 - Deadlock DataLoader multiproceso en arranque de 07_transformer.py (chunk_mean, run completo)

- **Sintoma**: Run completo (PID 258323, lanzado 22:59) congela tras cargar modelo BETO en GPU
  y forkear 8 workers. Main en `futex_wait` (0 CPU en ventana 30 s), 8 workers en `sys_poll`
  (0 CPU, ~0.07 s acumulado en 11 min), GPU 4-7%, sin lineas de log de epoca en 12+ min.
  Ultima actividad: ~7 warnings de tokenizacion (workers tokenizaron ~7 docs y se congelaron).
- **Evidencia**: `experimentos/m4_transformer_chunks_deadlock_evidencia.log` y
  `experimentos/m4_transformer_chunks_stdout_deadlock_evidencia.log` (copias preservadas).
- **Diagnostico (triage ml-debugging)**:
  - Smoke del propio script (`--limite 64 --epocas 1`): COMPLETA (EXIT 0, 139 s, ambos escenarios).
  - Probe A (DataLoader identico: fork, 8 workers, pin_memory, modelo en GPU, dataset completo,
    2000 docs): PASA a 200 docs/s.
  - Probe B (bucle de entrenamiento completo: esperar_vram, AMP, GradScaler, acumulacion 32,
    500 docs): PASA a 5.8 docs/s.
  - Causa raiz: condicion de carrera NO determinista en el arranque del DataLoader
    multiproceso (`fork`) con runtime ROCm/HIP ya inicializado (hilos internos + pools OMP
    heredados con locks en vuelo al momento del fork). 1 fallo en 4 arranques.
- **Run 2 (00:12, PID 479303, con OMP_NUM_THREADS=1)**: main murio SIN traceback a los ~15 min.
  CAUSA RAIZ: error del operador (ml-trainer) — senal SIGUSR1 enviada al main durante intento de
  diagnostico con py-spy (permiso denegado); la accion por defecto de SIGUSR1 sin handler es
  terminar el proceso. NO fue OOM ni bug del script. Evidencia preservada en
  `experimentos/*_run2_sigusr1_evidencia.log`. Leccion: NUNCA enviar senales a un run en curso;
  py-spy requiere sudo en este equipo (ptrace_scope=1).
- **Nota de medicion**: `pgrep -f 07_transformer` matchea los propios comandos bash de monitoreo
  (patron en su cmdline) -> usar `ps -eo pid,cmd | awk '/[0]7_transformer\\.py/'`.
- **Fix aplicado (sin tocar el script aprobado)**: relanzamiento run 3 SIN env vars (fidelidad
  al script revisado; probes A/B pasaron sin ellas). Monitoreo reforzado: criterio de cuelgue =
  sin linea "epoca 1" en 15 min Y main con 0 CPU en ventana 60 s (medido con /proc/<pid>/stat,
  PIDs verificados con ps+awk, sin senales).
- **Si recurre**: la mitigacion robusta (fork ANTES de .to(cuda), num_workers=0, o
  start_method spawn) requiere modificar `scripts/07_transformer.py` REVISADO -> escalar al
  orquestador para re-revision (NO modificar sin gate).
- **Guard propuesto**: verificar progreso de log (linea de epoca) en los primeros 15 min de
  cualquier run largo de 07_transformer.py antes de darlo por lanzado.
