# P84 - Auditoria de rendimiento (modo Quick, verificacion de artefactos)

- Fecha: 2026-10-02
- Alcance: fase P84 (EXP-01..EXP-05). No se ejecutaron entrenamientos nuevos y no habia procesos P84 activos.
- Modo: Quick con verificacion de las cifras ya registradas. No se midio GPU/CPU/RAM en vivo; los valores citados provienen de logs, JSON y derivados aritmeticos de artefactos, y se etiquetan con su fuente.
- Maquina: 32 nucleos logicos, 31 GB RAM, AMD RX 9070 XT 17,095,983,104 B VRAM (gfx1201), ROCm 7.2 + torch 2.14.0+rocm7.2 (config/hardware.json).
- Fuentes: reports/p84_01_dl01.json/.md, p84_02_fasttext.json/.md, p84_03_ros.json/.md, p84_04_sgns.json/.md, p84_04_ancla.json, p84_05_em.json/.md, RUNLOG.md (entradas P84, lineas 5057-5093), /tmp/opencode/p84_01_run2.log, p84_02_run.log, p84_03_run(2).log, p84_04_run.log, p84_04_humo*.log, p84_05_run.log, config/hardware.json, experiments.csv (filas P84), reports/twitter_word2vec_training_lrfix.json (referencia p71_08).

## Scorecard

| Metrica | Valor | Fuente | Estado |
|---|---|---|---|
| Tiempo P84-01 (GPU, 15 combos + determinismo) | 248.33 s | log 248.3s + JSON tiempo_total_segundos | OK |
| Tiempo P84-02 (CPU, -thread 1, 90 ajustes + determinismo) | 206.36 s (146.09 s de entrenamiento) | log 206.4s + JSON | OK |
| Tiempo P84-03 (CPU liblinear) | 20.24 s (1a corrida 17.48 s; reejecucion por Enmienda 3bis) | log run2 + JSON | OK |
| Tiempo P84-04 (GPU, 3 semillas x S/M pareados + R) | 15511.17 s = 4.31 h (15434.9 s de entrenamiento; 76.3 s de overhead) | log 15511.2s + JSON + parciales | a mejorar (presupuesto) |
| Tiempo P84-05 (CPU numpy) | 1.0 s | log + JSON | OK |
| Suma de corridas finales P84 | 15987.10 s = 4 h 26 min (4.44 h) | derivado de los JSON | - |
| Epoca P84-04 | 1010-1056 s por epoca pareada S+M | log (15 lineas) | OK |
| Throughput P84-04 | 124.0-126.0 lotes/s (643,264 pasos / 5106.1 s, seed 20261016) | derivado de parciales (pasos/segundos) | - |
| VRAM pico P84-01 | no medido (solo baseline pre-vuelo en reposo: 1.99 GB, uso 5-9%) | log run2 | a mejorar (telemetria) |
| VRAM pico P84-04 | no medido | - | a mejorar (telemetria) |
| Uso CPU durante corridas | no medido | - | a mejorar (telemetria) |
| AMP P84-01 | desactivado; evidencia medida previa citada (BiLSTM 3.42 vs 7.48 s; CharCNN 0.41 vs 1.73 s) | JSON gpu.amp/amp_evidencia | OK |
| Hilos | script: torch.set_num_threads(32) en P84-01:2549 y P84-04:283; informe solo declara -thread 1 en P84-02 | scripts + informes | documentacion |

## 1. Coherencia de tiempos, device, hilos, AMP y memoria

| Exp | Device declarado | Hilos en informe | AMP en informe | Memoria en informe | Tiempo informe vs log |
|---|---|---|---|---|---|
| P84-01 | cuda (RX 9070 XT), inicial y final, fallback_cpu False | no | False con evidencia medida | baseline pre-vuelo 2.03e9 B; pico no | 248.33 vs 248.3 s OK |
| P84-02 | cpu (fastText -thread 1; sin torch) | si, thread 1 | no aplica (sin torch) | no aplica | 206.36 vs 206.4 s OK |
| P84-03 | cpu (liblinear; sin torch) | no | no aplica | no aplica | 20.24 vs 20.2 s OK |
| P84-04 | cuda (RX 9070 XT) | no | no declarado | no | 15511.17 vs 15511.2 s OK |
| P84-05 | cpu (numpy puro, sin entrenamiento) | no aplica | no aplica | no aplica | 1.0 vs 1.0 s OK |

- Todos los totales coinciden entre informe y log. No hay procesos P84 activos al auditar.
- La unica cifra GPU de P84-01 (pre-vuelo, 5 muestras, uso 5-9%) es de la GPU en reposo antes de entrenar; no es utilizacion durante el entrenamiento y no debe leerse como tal.

## 2. Honestidad de metricas

- No se detectan cifras de rendimiento presentadas como medidas sin fuente. Cada tiempo tiene log asociado y cada informe declara su device.
- La evidencia de AMP de P84-01 (BiLSTM/CharCNN) es una medicion real de una fase previa (reports/dl_desde_cero_resultados.json, reports/contexto.md:166) en la misma GPU; el informe la cita como evidencia y no como re-medicion de P84-01. Correcto, pero es un dato de otro modelo (mismo orden de tamano).
- VRAM pico y utilizacion GPU durante P84-01 y P84-04: no medidas. Se reportan aqui como "no medido"; no se extrapola nada de ellas.
- La spec P84 cita "~23 min por corrida de 10M tokens/5 epocas/dim 200, medido en p71_08: 1402.9 s" (docs/p84_spec.md:647). Verificado: 1402.9 s es real pero corresponde al corpus twitter de p71_08 (10,000,029 tokens, vocab 77,657); el mismo script midio el corpus general (10,002,400 tokens, vocab 100,097, identico en tokens/vocab al de P84-04) en 2491.5 s (reports/twitter_word2vec_training_lrfix.json). La estimacion se anclo en la variante rapida del corpus. Limitacion: p71_08 no publica hash del corpus, luego la equivalencia con el de P84-04 se apoya en tokens/vocabulario.

## 3. Hallazgos

### [ALTO] Presupuesto de P84-04 anclado a un corpus no comparable
- Area: GPU / Datos / Planificacion
- Ubicacion: docs/p84_spec.md:647; reports/twitter_word2vec_training_lrfix.json; reports/p84_04_sgns.json
- P84-04 entreno 6 brazos-semilla en 15434.9 s (4.29 h) + 76.3 s de overhead (0.5%). La referencia comparable por corpus es 2491.5 s por brazo (5 epocas, dim 200): 6 x 2491.5 = 14949 s. El resultado real queda a +3.2% de esa referencia; no hay defecto de eficiencia en la implementacion. El desvio percibido (~1.87x sobre 23 min/brazo) viene de comparar contra la corrida twitter (1402.9 s).
- Recomendacion: en proximas specs, citar la referencia por corpus exacto (tokens/vocab/hash) y registrar el tiempo de la corrida p71_08 del corpus que se va a usar. Mantener el entrenamiento pareado S/M: el ahorro de generacion de datos es real (flujo verificado por hash).

### [MEDIO] Sin telemetria de VRAM ni de utilizacion GPU en P84-01 y P84-04
- Area: GPU
- Ubicacion: reports/p84_01_dl01.json (solo pre-vuelo); reports/p84_04_sgns.json (solo device)
- Sin muestreo no puede afirmarse ni descartarse cuello de botella GPU vs CPU ni subutilizacion de VRAM (impacto potencial, no medido).
- Recomendacion: en corridas GPU, anexar al JSON muestras de torch.cuda.mem_get_info() y de uso del device (rocm-smi) por epoca o cada N minutos; coste despreciable.

### [MEDIO] Reanudacion de P84-04 con granularidad de brazo-semilla (~85 min) y hito de decision no equivalente
- Area: GPU / Operacion
- Ubicacion: p84_04_run.log (parcial por unidad); experiments/p84_04_sgns_corregido_anclado.py:154 y :1186
- La reanudacion existe y se probo en humo (p84_04_humo_reanudar1/2.log), pero el punto de guardado es por unidad completa: una caida a mitad de semilla cuesta hasta ~85 min. El hito preregistrado era "40 min por brazo-semilla"; el codigo lo implementa como "epoca > 2400 s", condicion que con epocas de ~1020 s nunca se activo (no hay linea HITO en el log).
- Recomendacion (opcional, no cambia el analisis): checkpoint por epoca (modelo + optimizador + contador de paso) reutilizando el hash de flujo ya implementado, y mover el hito a tiempo transcurrido desde el inicio de la unidad.

### [BAJO] Hilos no declarados en los informes de P84-01 y P84-04
- Area: CPU / Documentacion
- Ubicacion: experiments/p84_01_dl01_corregido.py:2549; experiments/p84_04_sgns_corregido_anclado.py:283; informes .md/.json
- Los scripts fijan 32 hilos (coherente con hardware.json) pero los informes no lo declaran; P84-02 si declara -thread 1.
- Recomendacion: anadir una linea "hilos" en la seccion de entorno de los informes.

### [BAJO] Grilla de epocas de fastText reentrenada desde cero (sin accion)
- Area: CPU / Datos
- Ubicacion: p84_02_run.log; reports/p84_02_fasttext.json
- Cada split entrena 3 veces (5/10/20 epocas), ~35 unidades de epoca por split frente a 20 minimas; hasta ~40% de los 146.09 s de entrenamiento (~1 min). fastText CLI no facilita snapshots intermedios y la magnitud es irrelevante.
- Recomendacion: no invertir esfuerzo; dejar constancia de que es deliberado.

### [INFO] Buenas practicas verificadas
- Determinismo comprobado con doble pasada identica (P84-01 y P84-02) y flujo pareado S/M verificado por hash de pares y negativos (3/3 semillas).
- Pre-vuelo GPU D-008 de 5 muestras con umbrales y decision "libre" registrados antes de entrenar.
- Ancla algebraica/gradiente/ejecutable escrita antes de entrenar (p84_04_ancla.json), parciales por unidad, 15 checkpoints en P84-01, OOF y hashes; 0 incidencias en las 5 corridas finales.
- P84-03 reejecutado tras corregir el bug de conteo (Enmienda 3bis) con informe y fila CSV regenerados; el tiempo vigente es 20.24 s.

### [INFO] GPU vs CPU por experimento (sin evidencia de mal uso)
- P84-01: GPU, MLP de 110,339 parametros, batch 32, 15 combos en 248 s (13.1-15.5 s por combo en la corrida final; 212.0 s de los 3 seeds). AMP desactivado con evidencia medida; impacto potencial no medido de usar CPU, no se recomienda cambiar.
- P84-02: CPU -thread 1 deliberado por determinismo; 206 s.
- P84-03: CPU liblinear (1 hilo por defecto); 20 s.
- P84-04: GPU con SGNS manual, lote 256, 643k pasos por semilla (~126 lotes/s con 2 brazos por lote); epoca pareada 1010-1056 s, equivalente a ~505-528 s por brazo frente a 498.3 s/epoca de p71_08 en el mismo corpus (+1.3% a +6.0% por epoca; +2.5% a +4.1% por semilla).
- P84-05: CPU numpy puro; 1 s.

## 4. Recomendaciones acotadas para futuras corridas (sin tocar el diseno cerrado)

1. Presupuesto por corpus: referenciar tokens/vocab/hash y usar 2491.5 s/brazo para wiki-10M dim 200 (no 1402.9 s, que es twitter); anadir 0.5-3% de overhead de ancla+downstream.
2. Telemetria GPU: VRAM pico y uso del device por epoca en cada corrida GPU; anexarlos al JSON.
3. Operacion de corridas largas: checkpoint por epoca y hito de decision por tiempo transcurrido de la unidad.
4. Documentacion: declarar hilos y AMP/no-AMP en todos los informes, aunque sea "no aplica".
5. No optimizar P84-02/P84-03/P84-05 (206 s, 20 s, 1 s): cualquier micro-optimizacion es ruido.

## 5. Limitaciones del analisis

- Modo Quick: sin medicion en vivo de GPU/CPU/RAM; los hallazgos sin log se etiquetan como "impacto potencial" y no como "medido".
- Los tiempos son auto-reportados con time.perf_counter dentro de los scripts; no hay verificacion externa con time de shell.
- La corrida 1 de P84-01 (crasheo al escribir el MD tras entrenar), la 1a de P84-03 y los humos de P84-04 no registran tiempo total en artefacto: no se contabilizan en la suma de 15987.10 s; su gasto real no esta medido.
- La equivalencia de corpus entre p71_08-general y P84-04 se basa en tokens/vocabulario; falta hash en p71_08.
