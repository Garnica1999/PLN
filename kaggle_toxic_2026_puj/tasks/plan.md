# PLAN v2 - KAGGLE_TOXIC_2026_PUJ (Advanced Research)

Aprobado por Carlos el 2026-09-25. Objetivo: encontrar una fuente de senal nueva
que supere el LB 0.78306 de V03, con evidencia experimental por encima del
numero aislado.

## Reglas del proyecto

- V03 congelada en `experiments/baseline_v03_frozen/`; no se modifica. Todos los
  experimentos nuevos se comparan contra ella.
- Protocolo de validacion v2 (D-016):
  - Grupos = near-duplicado (Fase 0) UNION pseudo-autor (primera mencion).
  - Holdout congelado: `StratifiedGroupKFold(7)` sobre los grupos; 1 pliegue es
    holdout (~14 %), 6/7 es desarrollo. Se congela ANTES de mirar resultados;
    maximo 5 evaluaciones al holdout en todo el proyecto, solo finalistas.
  - `corrected_cv` (desarrollo, grupos fusionados) es la metrica de seleccion;
    `historical_cv` se reporta para continuidad. Baseline de referencia en el
    desarrollo: la propia V03 reevaluada.
  - Contingencia: si el corrected_cv tiene std entre pliegues > 0.025, pasar a CV
    repetida (3 semillas x 5 pliegues); si aun no discrimina, holdout al 20 %.
- Regla de adopcion: corrected_cv >= +0.005 y mejora en >= 3/5 pliegues; o bien
  complementariedad demostrada (`ensemble_gain` >= 0.0015 con IC bootstrap > 0).
- Todo OOF; prohibido usar test para fit/vocabulario/seleccion/umbrales.
- Presupuesto LB: maximo 3 sondas, solo finalistas, cada una con hipotesis.
- Entregables academicos (notebook y FINAL_REPORT) se actualizan si un candidato
  supera a V03.

## Paquetes y gates

| Paquete | Contenido | Gate de salida |
|---|---|---|
| P0 | Congelar V03; holdout v2; OOF corregido de bases; gap analysis v2; backlog v2; D-016 | Gate P0 publicado con numeros |
| P1 | Label dependency suave (cadenas en 3 ordenes) + meta-modelo regularizado | corrected_cv o complementariedad |
| P2 | Target-insult distance; target profiling; colectivo + agresion (FN de AG) | idem + mejora en rebanadas |
| P3 | POS/dependencias/morfologia como representacion SPARSE | idem |
| P4 | Cita / atribucion / negacion (H13), canales autor vs citado | idem |
| P5 | Emocion H5 + lexicones documentados + NB-SVM + multicanal ortografico | idem |
| P6 | Late fusion / stacking v2 + bootstrap por tweet | adopcion pre-registrada |
| P7 | Finalistas -> holdout -> auditoria -> 2-3 sondas LB -> entregables | GO/NO-GO |

## Lineas cerradas (no repetir sin hipotesis nueva)

Reduccion de capacidad (v06/v07 peores), ensembles indiscriminados (v04
0.77510), Optuna masivo (v05: OOF 0.8039 -> LB 0.77602), post-proceso duro
(-0.0056), DL secuencial desde cero (0.7949/0.7410/0.7364), concatenacion densa
de features (-0.009 a -0.032).

## Criterios de cierre de linea

Una linea se cierra cuando la hipotesis fue probada, las variantes razonables
fueron evaluadas, no hay mejora y el resultado es estable (documentado en
RUNLOG + experiments.csv + DECISIONS).

## P80 - Auditoria del entrenador Word2Vec (Fase 1 del plan maestro p80_p82)

Estado: PRE-REGISTRADO, NO EJECUTADO. Se anade al final de este fichero; lo
anterior no se modifica. Plan maestro: `/home/carlos/.opencode/plan/p80_p82_auditoria_w2v_rnn_bayesiano.md`,
Fase 1. Pre-registro: `docs/p80_preregistro_auditoria_w2v.md`.

### Pregunta

Esta bien entrenado el Word2Vec 100d congelado, o su trainer tiene defectos
que lo invalidan como representacion reutilizable? Se responde con AUDITORIA
pura: cero entrenamientos, cero cambios en el artefacto.

### Fijado antes de ejecutar

- 19 afirmaciones numeradas (A1 a A19), cada una con tolerancia, criterio de
  exito, criterio de fallo y marca de STOP. Tabla de Naturaleza: CONFIRMAR,
  REFUTAR o NO APLICABLE POR CONSTRUCCION.
- Hallazgos cubiertos: H1 ratio 1/K exacto de la loss y del gradiente (A1-A5),
  H2 desfase de semillas del scheduler leido por AST sin heuristica de
  encabezado (A6-A8), H3 identidad de la loss con el signo correcto y control
  negativo (A9-A10), H4 matriz de salida no reconstruible y cifras absolutas
  que no son evidencia (A11-A13), H5 pooling y epoca de los BiGRU (A14),
  H6 padding enmascarado despues de la GRU con umbral fijado (A15-A16),
  H7 DL-03 con todo train sin excluir el holdout v2 (A17-A19).
- A5 retira la cota [1/K, 1] sobre la razon de la norma total del gradiente:
  P79 la afirmo sin demostrarla y su propia medida del artefacto la incumple
  (0.08033795 con K=10, por debajo de 0.1).
- A16 fija el unico umbral que puede cambiar un veredicto: diferencia relativa
  1e-3 entre el relleno a longitud real y el relleno al maximo de tanda.
- Formulacion literal obligatoria: `V03 remains the best verified leaderboard result.`
  Prohibido afirmar que Word2Vec sea mejor que V03.

### Alcance de lectura

Lee: `p71_04`, `p71_08`, `p72_01`, `v9_04`, `v10_11`, `11_dl_desde_coro.py`,
`reports/p79_auditoria_sgns.json` (contraste, no evidencia),
`reports/dl_desde_cero_resultados.json`, los informes de procedencia del
congelado, las primeras 2000 lineas del corpus tokenizado, y de
`config/holdout_v2.json` SOLO los dos recuentos de filas.

No lee: test, ninguna particion del holdout, la lista de ids del holdout,
leaderboard, submissions, el parquet de features, ni el OOF de V03.

El artefacto congelado se abre solo en lectura, con sha256, `st_size` y
`st_mtime_ns` comprobados antes de la primera lectura y despues de la ultima.
sha256 declarado: `302c994d13b75f8f5fbd440eea2c4f4113997e2581524021923b92f8244d705a`.

### Secuencia

1. P80.1 Pre-registro escrito, verificado en ASCII byte a byte y commiteado.
   Gate 1: sin commit no se ejecuta.
2. P80.2 `experiments/p80_01_auditoria_w2v.py`, script unico y completo.
3. P80.3 Revision de codigo por `ml-reviewer` (5 ejes ML), OBLIGATORIA, con
   atencion a replicacion circular, comparaciones con el signo mal puesto, uso
   de una matriz que no existe y aserciones que no abortan. Gate 3: lint OK,
   sin leakage, venv correcto.
4. P80.4 Ejecucion por `ml-trainer`, veredicto por hallazgo, y correccion de
   las tres afirmaciones erroneas de P79 en documento NUEVO. P79 no se edita.
5. P80.5 Dictamenes de `dl-specialist`, `nlp-specialist`, `ml-reviewer` y
   `kaggle-specialist`, en paralelo, registrados como dictamenes. Si contradicen
   a los datos, ganan los datos y se documenta por que.
6. P80.6 Gate 4 cerrado y actualizados DECISIONS, RUNLOG y PROJECT_STATE.

### Decision que P80 desbloquea, y solo si los datos la dan

Si H5 y H6 invalidan los BiGRU previos, la via RNN se cierra con evidencia y
no se abre P81. Si son reinterpretables y H1 deja margen real en el vector, se
pre-registra P81 con early stopping estrictamente anidado y relleno correcto,
sin atencion. El gap a V03 es de 0.117 y son horas de GPU: lo decide Carlos.
