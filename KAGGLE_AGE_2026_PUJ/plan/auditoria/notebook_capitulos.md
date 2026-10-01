# Capitulos del notebook de auditoria (fuente para `scripts/auditoria_v1v37_notebook.py`)

Este archivo existe para que el script sea **ASCII puro** (regla 8.4 del proyecto) y el texto
visible del notebook pueda llevar tildes. Los dos capitulos van separados por una linea

    ===CELDA===

<!-- titulo: PARTE III · AUDITORÍA DEL HISTORIAL V1-V37 -->

Esta parte documenta una **auditoría de evidencia** del historial completo del proyecto,
ejecutada el 2026-10-01 y registrada en el acta **D-090** de `DECISIONS.md`.

No es una nueva búsqueda de señal ni una familia de experimentos: es una revisión de lo ya hecho.
La pregunta era: *de las 195 filas del registro, ¿cuáles son ejecuciones reales, se ejecutaron
correctamente, usaron parámetros lógicos, están justificadas, y están documentadas con exactitud?*

Restricciones que rigieron la auditoría: **no reejecutar** ningún script, **no abrir** test, holdout ni
leaderboard, **no generar** submissions, **no tocar** el campeón ni los scripts ya ejecutados.

### Qué encontró

Lo primero, y más relevante para leer bien el registro: **195 filas no son 195 experimentos**. La
clasificación por tipo de registro da **151 ejecuciones**, 33 documentos (21 documentales + 12
pre-registros), 10 enmiendas y 1 corrida abortada (`V35-E1-INTENTO-1`, `UnboundLocalError`, con la
evidencia en cuarentena). Toda la evidencia citada existe en disco: **0 artefactos ausentes**.

Y conviene decir lo bueno sin rodeos: **el protocolo de competencia está limpio en lo esencial**. No
hubo `fit` sobre test (verificado por barrido AST de 308 scripts, no por declaración propia), ni
calibración de umbrales sobre test, ni uso del leaderboard como conjunto de validación o como función
objetivo. `id` nunca se usó como característica, no hay fuga por autor ni temporal, y el campeón
congelado es reproducible bit a bit (SHA256 verificado, 112.892 filas, 6 clases, 0 nulos).

### Tabla de hallazgos

| # | Hallazgo | Severidad | Estado |
|---|---|---|---|
| K-01 | El techo de LB 0,70 (D-087) se apoya en una rejilla de umbrales que nunca exploró por debajo de 0,30; el "techo" era el máximo de una rejilla incompleta | **alta** | **refutado** |
| A-01 | La multiplicidad de ~151 corridas no está cuantificada en ningún documento maestro | **alta** | registrada como reserva estructural |
| K-02 | El holdout de V12 está quemado: declara criterio de aceptación sobre el propio holdout | media | registrada |
| K-03 | El registro de envíos cuenta 15/12 cuando hay 16 registros puntuados reales | media | registrada |
| N-02 | `es_core_news_lg` no está fijado en `requirements.txt`, luego el anotador de todos los resultados md/lg no es instalable desde el entorno declarado | media | **acción pendiente** |
| A-03 | `max_iter=40` en el MLP, congelado 37 fases y miembro del campeón con peso 0,25 | media | reserva |
| A-02 | El inner split de Optuna no agrupa near-duplicates; medido **sin impacto** en solapamiento exacto | media | impacto en la métrica **sin medir** |
| A-04 | V32-E1 atribuye el cierre "a la pérdida" cuando la causa operativa es el presupuesto de épocas | media | reserva (el veredicto se sostiene) |
| N-01 | El brazo `N1b` del canal NER filtra spans con un tokenizador desalineado (~55,8 % de coincidencia) | media | sin impacto (bajo el gate) |
| N-03 | El 16,35 % de acuerdo md-lg se publica sin controlar por longitud | media | reserva |
| A-06 | `V18-P3/P4/P5` no siembran `torch.manual_seed` | baja | reserva |
| K-04 | `v36_00_preregistro.py` rotula `StratifiedKFold` cuando lee la partición canónica agrupada | baja | reserva |
| N-05 | SHA256 del pre-registro V37 mal transcrito en dos documentos maestros | baja | **corregido** |

### El hallazgo K-01 en detalle

Es el único que cambia una **conclusión publicada**, y merece su propia celda porque el número es
concreto. `scripts/v35_00g_techo_analisis.py:69` define:

```python
TAUS = (0.30, 0.35, 0.40, 0.45, 0.50)
```

La rejilla **empieza en 0,30 y sube**, y el "techo" se tomó como su máximo, mientras la curva del oráculo
es monótona creciente al bajar τ. Reproducción sobre `artifacts/p33_scores/todos/` (base 0,6159732,
65.683 errores):

| τ | filas corregibles | ganancia | ¿supera +0,0626? |
|---|---|---|---|
| 0,30 (máximo publicado) | 10.384 | +0,0593677 | no |
| **0,2930 (el punto que el propio D-087 usa como objetivo)** | 11.298 | **+0,0646335** | **sí** |

D-087 afirmaba que un modelo perfecto "no llegaría". Es un artefacto del redondeo de la rejilla. **Lo que
sí se sostiene, y es más fuerte**: un recalibrador tendría que acertar el ~16 % de los errores con
`p(y_true) ≥ 0,293`, es decir conocimiento casi-oracular. LB 0,70 sigue siendo inalcanzable en la
práctica; la prueba publicada no lo demuestra.

### El hallazgo A-01 en detalle

Con `macro_f1_std` mediano 0,0042 sobre las 76 filas que lo declaran, el error estándar de la media de
5 folds es **SE ≈ 0,00188**. Un gate de `Δ ≥ +0,0015` implica z ≈ 0,80 y **≈ 32 falsos positivos
esperados** en 151 corridas; `Δ ≥ +0,0030` implica z ≈ 1,60 y **≈ 8**. Un Bonferroni de familia 151
exigiría `Δ ≥ 0,0147`.

`grep` de *multiplicidad / multiple comparisons / FWER / family-wise* en `DECISIONS.md`,
`PROJECT_STATE.md`, `FINAL_REPORT.md` e `INFORME_COMPLETO.md` devuelve **cero**.

**Por qué no invalida las conclusiones**: las cuatro adopciones reales (V19-B1, V20, V30, plug-in)
tienen **IC95 pareado excluyendo 0** y confirmación externa. La afirmación correcta es "ninguno pasó el
gate *y* su bootstrap pareado excluyendo 0", no "ninguno pasó el gate".

### Qué NO encontró la auditoría

Para que el informe sea legible también por lo que no hay:

- **Ningún control negativo degenerado.** El patrón de V36-E1 (permutación biyectiva, o barajado
  intra-documento que deja la matriz idéntica al ser bolsa de n-gramas) **no se repite** en las familias
  lingüísticas. `barajar_intra` de P2 **sí** discrimina (−0,008031 intra vs −0,0039528 filas).
- **Los cierres distinguen "cero señal" de "efecto negativo".** P2 cierra con IC95 que excluye cero; P7
  cierra como "no detectable" porque su IC **incluye** cero; R3 cierra con IC que excluye cero. Ningún
  cierre se apoya en un delta negativo sin intervalo.
- **Cero adopciones sin pre-registro.** El plug-in incluso tiene commit verificable anterior al
  resultado.
- **Continuidad de la partición**: los 32+ scripts de las familias no lingüísticas usan el esquema
  agrupado; ninguno recrea splits aleatorios.
- **Sin fuga** en ninguna red ni meta-modelo: escalador ajustado solo en fold-train, combinador anidado
  sobre probabilidades OOF.

### Dónde está el detalle

| Documento | Contenido |
|---|---|
| `reports/auditoria_experimentos_v1_v37.md` | informe principal, con las tablas y los números |
| `reports/tables/auditoria_v1v37_inventario.csv` | matriz de 197 filas × 24 columnas, 0 huecos |
| `plan/auditoria/01_inventario.md` | fase 1 y hallazgos H-01 a H-04 |
| `plan/auditoria/02a_familias_linguisticas.md` | fase 2a, defectos N-01 a N-08 |
| `plan/auditoria/02b_arquitectura_y_dl.md` | fase 2b, defectos A-01 a A-07 |
| `plan/auditoria/02c_protocolo_y_qa.md` | fase 2c, defectos K-01 a K-04 y el QA independiente |
| `DECISIONS.md` | acta **D-090** |

La siguiente celda lee el inventario congelado y muestra el recuento, de modo que **ninguna cifra de
esta auditoría está escrita a mano**.

    ===CELDA===

<!-- titulo: Inventario congelado de la auditoría -->

El recuento que sigue se **lee** de `experiments/auditoria_v1v37_inventario.json`, que
produce `scripts/auditoria_v1v37_inventario.py`. Ese script tiene un modo `--verificar` con 6 checks
que **no escribe nada**, y su aislamiento está comprobado por hash del árbol antes y después.

Regeneración:

```bash
/home/carlos/venv/bin/python scripts/auditoria_v1v37_inventario.py --verificar   # 6/6, exit 0
/home/carlos/venv/bin/python scripts/auditoria_v1v37_inventario.py             # escribe los 3 artefactos
```

**Nota sobre el alcance de este capítulo**: el notebook llega hasta V35. Las fases **V36** (cierre de
L4, D-088) y **V37** (diagnóstico `P(POS|edad)`, D-089) están documentadas en `reports/v36_e1_*.md` y
`reports/v37_diagnostico_pos.md` y en el acta D-090, pero **no tienen sección propia aquí**. Queda
registrado como deuda documental pendiente de reconciliación.

    ===CELDA===
