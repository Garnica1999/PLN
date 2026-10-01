# Auditoria V1-V37 · Fase 00 - Entorno, alcance y reglas

## Fecha y estado

- Fecha de inicio: 2026-10-01.
- Rama: `proyecto-kaggle-age-2026-v32`.
- Commit de referencia: `06c8828` ("AGE: V37 cerrado - acta D-089: P(POS|edad) CERRADA POR POTENCIA").
- Arbol del proyecto: limpio. Las unicas entradas sin seguimiento son `../home/` y `../kaggle_toxic_2026_puj/tools/`,
  que pertenecen a otros proyectos y quedan fuera del alcance.

## Entorno detectado (Fase 00)

| Recurso | Valor detectado |
|---|---|
| Python canonico | `/home/carlos/venv/bin/python` (3.14.7) |
| Hilos logicos | 32 |
| RAM | 31 GB |
| GPU | AMD Radeon RX 9070 XT, 15,9 GB VRAM, `torch.cuda.is_available() == True` (ROCm 7.2) |
| torch / sklearn / numpy / pandas | 2.14.0+rocm7.2 / 1.8.0 / 1.26.4 / 2.3.3 |

Perfil persistido en `config/hardware.json`.

**Derivacion de recursos para esta auditoria**: la auditoria no entrena, por lo que no hay batch, ni
`num_workers`, ni presupuesto de GPU que decidir. El presupuesto real es de tiempo de lectura y analisis.
La GPU queda declarada pero **no se usa** para repetir ninguna corrida.

## Alcance

- Historico completo V1 a V37.
- Las 195 filas de `experiments.csv` mas cualquier corrida documentada fuera del registro que aparezca
  en scripts, resultados, logs o artefactos.
- `DECISIONS.md`, `PROJECT_STATE.md`, planes y pre-registros, informes de `reports/`, JSON de
  `experiments/`, logs de `experimentos/` y `experiments/`, y `notebooks/99_entrega_consolidado.ipynb`.

## Reglas que gobiernan esta auditoria

| Regla | Origen |
|---|---|
| No reejecutar scripts ya corridos ni entrenar | plan de auditoria + politica del proyecto |
| No tocar el campeon V30 + plug-in (`v30_prob`, seis pesos) | D-072 a D-074, acta D-089 |
| No abrir test, holdout, leaderboard ni enviar submissions | brief de la competencia, decisiones previas |
| No editar scripts de experimentos ya ejecutados | metodologia operativa del proyecto |
| Ninguna afirmacion sin artefacto o ruta verificable | precedente D-086, D-089 |
| Un resultado negativo no es un fallo de ejecucion | criterio de la auditoria |
| Pre-registro congelado manda; enmiendas pre-ejecucion declaradas mandan sobre la spec | D-086, D-089 |

## Verificacion de integridad antes de auditar

- Submissions: 22 ficheros, el mas reciente con fecha 2026-09-28. Ninguno modificado durante V29-V37.
- Campeon: `artifacts/p33_scores/todos/v30_prob.npy` con SHA256 `3d77352486605971` y mtime 2026-09-29,
  anterior a las fases V32-V37.
- Verificacion del campeon recomputada al inicio: macro-F1 = `0.6159732` con los seis pesos congelados,
  identico al valor de D-089.

## Disposicion de la salida

- Informe principal largo: `reports/auditoria_experimentos_v1_v37.md`.
- Matriz trazable: `reports/tables/auditoria_v1_v37_matriz.csv`.
- Resumen en el notebook consolidado: capitulo de auditoria con enlace al informe y a la matriz.
- Acta propia en `DECISIONS.md` y filas nuevas en `experiments.csv` (`AUD-INV`, `AUD-EJEC`, ...).
- Memoria del proyecto en `plan/contexto.md` y en Graphiti (grupo `kaggle_age_2026_puj`).
