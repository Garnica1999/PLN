# Auditoría del historial experimental V1–V37

**Proyecto**: `KAGGLE_AGE_2026_PUJ` · **Competición**: `tweet-age-2026-puj` (Kaggle, host `NLP_Course_PUJ`)
**Fecha**: 2026-10-01 · **Acta**: `DECISIONS.md` D-090 · **Rama**: `proyecto-kaggle-age-2026-v32` (commit base `06c8828`)

---

## 1. Respuesta corta

El protocolo del proyecto está limpio en lo esencial, y conviene decirlo sin rodeos: **no hubo
`fit` sobre test, no hubo calibración de umbrales sobre test, no hubo uso del leaderboard como conjunto
de validación ni como función objetivo, `id` nunca se usó como característica, no hay fuga por autor ni
temporal, y el campeón congelado es reproducible bit a bit.**

Sobre 195 registros: **no son 195 experimentos**. Son **151 ejecuciones**, 33 documentos
(21 documentales + 12 pre-registros), 10 enmiendas y 1 corrida abortada. Toda la evidencia citada
existe en disco: **0 huecos**.

Encontré **20 defectos confirmados**. Ninguno invalida una adopción. El más grave no es de ejecución
sino de razonamiento: la prueba de que el leaderboard 0,70 es inalcanzable **no demuestra lo que
afirma**, porque su rejilla de umbrales nunca exploró por debajo de 0,30.

El campeón sigue congelado en **0,6159732487657944**, verificado por hash y recomputado.

---

## 2. Alcance y método

Se auditó el historial completo, no solo las fases recientes, porque el peso del registro está en las
fases iniciales (113 de 195 filas en V1–V12) y ahí es donde los parámetros heredados carries más riesgo.

| Revisión | Alcance | Resultado |
|---|---|---|
| `ml-architect` | diseño experimental transversal, 7 familias no lingüísticas | 7 defectos |
| `nlp-specialist` | POS, NER, morfología, dependencias, V37 | 8 defectos |
| `dl-specialist` | régimen de entrenamiento de las redes | 2 defectos |
| `kaggle-specialist` | reglas, envíos, calibración CV↔LB | 4 defectos |
| `ml-reviewer` | el script de inventario de esta auditoría | 3 bloqueantes, corregidos |
| `ml-qa-engineer` | correcciones del auditor y dictámenes ajenos | 1 defecto crítico, corregido |

Ningún hallazgo se aceptó sin **verificación propia contra la fuente primaria** con ruta y línea. Tres
hallazgos ajenos se matizaron al comprobarlos (dos cifras y un alcance). El más relevante: un hallazgo
que parecía decir "el pre-registro se mutó" resultó ser un error de transcripción en dos documentos.

**Restricciones respetadas**: no se reejecutó ningún script, no se abrió test/holdout/leaderboard, no se
generó ninguna submission, no se tocó el campeón ni ningún script de experimento ya corrido.

---

## 3. Inventario: qué se ejecutó realmente

`scripts/auditoria_v1v37_inventario.py` clasifica cada fila por su **tipo de registro**, no por
heurística de texto, y resuelve cada artefacto citado contra el árbol real.

| Tipo | Filas | Qué es |
|---|---|---|
| `EJECUTADO` | **151** | hubo corrida; 113 con macro-F1 en el registro y 38 cerradas por gate o verbal |
| `DOCUMENTAL` | 21 | auditoría, informe, cierre, compliance, submission; no entrena |
| `PREREGISTRO` | 12 | pre-registro congelado, sin ejecución |
| `ENMIENDA` | 10 | enmienda pre-ejecución declarada |
| `BLOQUEADO` | 1 | `V35-E1-INTENTO-1`, `UnboundLocalError`, evidencia en cuarentena |

Distribución por familia:

| Familia | Filas | | Familia | Filas |
|---|---|---|---|---|
| V1–V4 fundacional | 37 | | V24–V29 selección y ensemble | 15 |
| V5–V12 fases PH* | 76 | | V30–V31 campeón y tracks | 14 |
| V13–V17 búsqueda residual | 13 | | V32–V33 reapertura y FTRL | 10 |
| V18–V23 frontera y ordinal | 12 | | V34–V36 falsación y L4 | 16 |

**Los 95 huecos iniciales de artefactos no eran huecos.** Eran rutas citadas con coma en vez de ` | `,
CSVs de `reports/` citados sin su carpeta, o prosa entre paréntesis. Con un resolvedor correcto: de 262
artefactos citados, 230 existen y los 32 restantes son globs (`lexicons/**`), carpetas o referencias
textuales. **`features/` y `models/` están en `.gitignore` y no aparecen en el índice de Codebase**, por
lo que se comprobaron contra el árbol real.

---

## 4. El hallazgo más grave: el techo de LB 0,70 no demuestra lo que afirma

`DECISIONS.md` D-087 afirmó que un modelo perfecto con la información que el campeón ya tiene
"no llegaría" al objetivo, porque el oráculo rendía **+0,0568** frente a un objetivo de **+0,0626**.

El problema está en `scripts/v35_00g_techo_analisis.py:69`:

```python
TAUS = (0.30, 0.35, 0.40, 0.45, 0.50)
```

**La rejilla empieza en 0,30 y sube.** El "techo" se tomó como el máximo de una rejilla que nunca
exploró por debajo de 0,30, mientras la curva del oráculo es monótona creciente al bajar τ.

Reproducción propia sobre `artifacts/p33_scores/todos/` (base 0,6159732, 65.683 errores):

| τ | filas corregibles | ganancia macro-F1 | ¿supera +0,0626? |
|---|---|---|---|
| 0,30 (máximo publicado) | 10.384 | +0,0593677 | no |
| **0,2930 (el punto que el propio D-087 usa como objetivo)** | 11.298 | **+0,0646335** | **sí** |
| 0,29 | 11.725 | +0,0670929 | sí |
| 0,28 | 13.199 | +0,0755813 | sí |
| 0,25 | 18.345 | +0,1054505 | sí |

En el punto exacto que D-087 define como objetivo, el oráculo da **+0,0646**, un 3 % por encima de lo
exigido. La afirmación "no llegaría" es un artefacto del redondeo de la rejilla.

**Lo que sí se sostiene, y es un argumento más fuerte**: un recalibrador que reordenara la salida de
V30 tendría que acertar el ~16 % de los errores con `p(y_true) ≥ 0,293`, es decir conocimiento
casi-oracular. La formulación correcta es *"el objetivo queda exactamente en el borde de lo que un
oráculo sobre la salida de V30 podría entregar"*, no *"no llegaría"*.

**LB 0,70 sigue siendo inalcanzable en la práctica; la prueba publicada no lo demuestra.**

Relacionado: el `IC95 [0.635, 0.653]` de D-087 **no está calculado en ningún artefacto**. Sale de una
"tasa de captura del 4 %–9 %" aplicada a mano. Es un rango **declarado, no estimado**.

---

## 5. Defecto de marco analítico: multiplicidad sin cuantificar

No es un defecto de ejecución, sino el único punto donde el proyecto afirma más de lo que su diseño
sostiene.

`grep` de *multiplicidad / multiple comparisons / FWER / family-wise / sesgo de selección* sobre
`DECISIONS.md`, `PROJECT_STATE.md`, `FINAL_REPORT.md` e `INFORME_COMPLETO.md` devuelve **cero**. El único
control de multiplicidad del proyecto es el Bonferroni interno de V37 (18 pruebas por clase).

Con `macro_f1_std` mediano 0,0042 en las 76 filas que lo declaran, el error estándar de la media de 5
folds es **SE ≈ 0,00188**:

| Gate | z aproximado | Falsos positivos esperados en 151 corridas |
|---|---|---|
| Δ ≥ +0,0015 | z ≈ 0,80 | **≈ 32** |
| Δ ≥ +0,0030 | z ≈ 1,60 | **≈ 8** |

Un Bonferroni de familia 151 exigiría Δ ≥ **0,0147**, tres veces el gate vigente.

**Por qué no invalida las conclusiones**: las cuatro adopciones reales (V19-B1, V20, V30, plug-in) tienen
**IC95 pareado excluyendo 0** y confirmación externa. El bootstrap pareado corrige la varianza de
estimación; no corrige la selección del máximo sobre 151 corridas, pero las tres adopciones que importan
no dependen de esa corrección.

**La afirmación correcta es "ninguno pasó el gate *y* su bootstrap pareado excluyendo 0"**, no "ninguno
pasó el gate".

---

## 6. El protocolo de competencia: limpio, verificado por AST

No por las declaraciones del propio proyecto, sino por barrido del árbol sintáctico de 308 scripts:

- **Cero `fit` sobre test.** Solo 2 `fit()` con receptor de tipo test
  (`phase32_correccion_probabilistica.py:181`, `phase33_corrector_minimo.py:144`), y ambos calibran
  sobre **OOF de train**.
- **Los ~20 vectorizadores** ajustan solo sobre train; el test solo recibe `.transform`.
- **`id` nunca es característica**; aparece solo en armado y validación de submission.
- **Near-duplicates protegidos**: 4,2554 % del test cae en un cluster de train
  (`experiments/audit_01_resumen.json:65`) y los clusters se construyen sobre train+test sin usar
  etiquetas. `reports/tables/audit_memorizacion.csv` muestra 0,00 % de memorización con el esquema
  agrupado frente a 3,99 % con el ingenuo.
- **Campeón reproducible bit a bit**: SHA256 `29cbe6cc…bd6c0` verificado, 112.892 filas, 2 columnas,
  ids 1…112.892 en orden, 6 clases válidas, 0 nulos.
- **Cero adopciones sin pre-registro.** El plug-in incluso tiene commit verificable (`bd08aba`,
  2026-09-29 14:54) anterior al resultado.

---

## 7. Defectos de código confirmados

| # | Defecto | Severidad | Impacto |
|---|---|---|---|
| **N-01** | El brazo `N1b` del canal NER filtra spans con `TOKEN_RE`, que coincide con `len(pos_seq.split())` solo en **55,8 %** de los documentos (spaCy parte `#TFZ` en `#`+`TFZ`+espacio) | media | **Ninguno**: los seis brazos quedan bajo el gate. Re-declarar N1b exigiría re-ejecución, que el brief prohíbe |
| **N-02** | `es_core_news_lg` **no está fijado** en `requirements.txt:49` (solo `md`) | media | **Alto y accionable**: el anotador usado en **todos** los resultados md/lg no es instalable desde el entorno declarado |
| **A-06** | `src/config.py:75-85` siembra `random` y `numpy` pero **no `torch.manual_seed`**, y `v18_p3/p4/p5_destilacion.py` construyen `nn.Linear` | baja | Esos tres experimentos no son reproducibles; son cabezas lineales con resultado negativo en 5/3/3 brazos, robustos al ruido de init |
| **A-02** | El inner split de Optuna usa `scheme="stratified"` con `groups=np.arange(...)` en 5 scripts del camino al campeón, aunque `src/cv.py:11` lo documenta como *control de leakage* | media | **Medido sin impacto**: 0,963 % ingenuo vs 1,030 % agrupado de solapamiento exacto. El impacto en la métrica de selección queda **sin medir** |
| **A-03** | `max_iter=40` en el MLP de `phase2_ponderacion.py:116`, congelado 37 fases y **miembro del campeón con peso 0,25** | media | La vía de revisión se diseñó (`v29_04_optuna_individual.py:130-141`) pero `experiments/v29_05_optuna_mlp.json` **no existe**: se diseñó y no se ejecutó |
| **A-04** | V32-E1 declara `NO_ATRIBUIBLE` **a la pérdida**, pero A1 agotó las 60 épocas en 5/5 folds mientras A2 paró por early stopping | media | El **veredicto** se sostiene por otra vía (MLP 0,5989/0,6028 vs campeón 0,6160); la **causa atribuida** no |
| **K-02** | El holdout de V12 está **quemado**: declara criterio `Δ ≥ +0,003` sobre el propio holdout y `confirmado: true` | media | Es una puerta de selección, no confirmación. Agravante mitigado: el CV ya había dado +0,0031 en 5/5 folds |
| **K-03** | El registro de envíos está obsoleto: cuenta 15/12 cuando hay **16 registros puntuados** | media | 4 envíos posteriores (V19, V20, V30, plug-in) no están en ningún registro |
| **K-04** | `v36_00_preregistro.py:338` rotula `StratifiedKFold` cuando lee la partición canónica `StratifiedGroupKFold` | baja | Solo el rótulo; el código es correcto |
| **N-03** | El 16,35 % de acuerdo md-lg se publica sin controlar por longitud | media | El acuerdo **por token** es 86,269 %, y la coincidencia exacta cae de 36,76 % (1–3 tokens) a 9,89 % (16–23). Mide longitud, no calidad de anotación |
| **N-05** | SHA256 del pre-registro V37 mal transcrito en dos documentos maestros | baja | **Corregido**: el real es `1d6d6573…`, que es el que registra el propio JSON. El pre-registro no se mutó |

---

## 8. Lo que se declara correcto, explícitamente

Para que el informe sea legible también por lo que **no** encontró:

- **Ningún control negativo degenerado.** El patrón de V36-E1 (permutación biyectiva o barajado intra
  documento que deja la matriz idéntica) **no se repite** en las familias lingüísticas. `barajar_intra`
  de P2 **sí** discrimina (−0,008031 intra vs −0,0039528 filas) porque el vectorizador usa
  `ngram_range=(1,3)` sobre las secuencias POS.
- **Hashes reproducibles.** El defecto `data_hash` con `tobytes()` de V36 **no se propagó**.
- **Los cierres distinguen "cero señal" de "efecto negativo".** P2 cierra con IC95 que excluye cero;
  P7 cierra como "no detectable" porque su IC **incluye** cero; R3 cierra con IC que excluye cero.
  Ningún cierre se apoya en un delta negativo sin intervalo.
- **V18–V23 y V32–V37 son el estándar del proyecto**: pre-registros con SHA256, controles barajados,
  gemelos con presupuesto idéntico, sondas bit a bit, y auto-falsaciones que se disparan solas (D-073
  paró la submission por confundir tres números; D-066 detectó que la cifra publicada no era
  reproducible).
- **Continuidad de la partición**: los 32+ scripts de las familias no lingüísticas usan
  `scheme="group"`; **ninguno recrea splits aleatorios**. Las semillas alternativas usan los mismos
  grupos repartidos de otro modo, y está **declarado**: el script de confirmación no contiene ningún
  optimizador.
- **Sin fuga en ninguna red ni meta-modelo**: escalador ajustado solo en fold-train, combinador anidado
  sobre probabilidades OOF, `V18-P5` con cross-fitting explícito del teacher.
- **La política de early stopping se respetó** en V32-E1, con fecha y autor registrados.
- **D-065 ya sabía que `HIERBA` estaba sub-parametrizado** y lo documentó. No es un defecto desconocido.

---

## 9. Correcciones aplicadas por esta auditoría

| Corrección | Detalle |
|---|---|
| `experiments.csv` → `V37-DIAG-POS` | escrita con 21 de 22 campos: una coma de más hizo que la narrativa de métricas aterrizara en `accuracy_mean` y que `decision` nunca se escribiera. Reparada con el patrón de `V36-E1` |
| `DECISIONS.md` y `PROJECT_STATE.md` | SHA256 del pre-registro V37 corregido de `30d5d3cf…` a `1d6d6573…` |
| `scripts/auditoria_v1v37_inventario.py` | 3 defectos bloqueantes detectados por `ml-reviewer`: el regex de pre-registro clasificaba **V33/V34/V35 como no ejecutados** (afirmación falsa en mi artefacto), `seccion_de()` devolvía el H1 para 8 de 23 filas, y los checks C1/C4/C5 eran tautológicos o código muerto |
| `experiments.csv` | 2 filas nuevas: `AUD-INV` y `AUD-REV` |

Sobre la primera reparación: **la primera versión fue incorrecta** (metió la narrativa en
`accuracy_mean` e inventó texto en `decision`) y el QA lo detectó. La versión final mueve la narrativa a
`resultado`, deja `accuracy_mean` vacía y recupera el cierre ya escrito en `interpretacion`, verificando
que los 17 campos descriptivos coinciden con el original.

---

## 10. Reservas abiertas

Tres cosas quedan **sin medir**, y se declaran como tales:

1. **Sesgo del inner split de Optuna** (A-02): medido sin impacto en solapamiento de texto exacto, pero
   su efecto en la métrica de selección no está cuantificado.
2. **`max_iter=40` del MLP** (A-03): miembro del campeón con peso 0,25 sin revisar.
3. **IC95 [0,635, 0,653] de D-087**: rango declarado, no estimado.

Ninguna exige cambiar el campeón.

---

## 11. Artefactos

| Fichero | Contenido |
|---|---|
| `scripts/auditoria_v1v37_inventario.py` | inventario reproducible, 6/6 checks, ruff limpio, ASCII puro |
| `reports/tables/auditoria_v1v37_inventario.csv` | 197 filas × 24 columnas, 0 huecos |
| `experiments/auditoria_v1v37_inventario.json` | resumen y filas del inventario |
| `plan/auditoria/00_entorno_y_alcance.md` | entorno, alcance y reglas de la auditoría |
| `plan/auditoria/01_inventario.md` | fase 1 con los hallazgos H-01 a H-04 |
| `plan/auditoria/02a_familias_linguisticas.md` | fase 2a: 8 defectos N-01 a N-08 |
| `plan/auditoria/02b_arquitectura_y_dl.md` | fase 2b: 7 defectos A-01 a A-07 |
| `plan/auditoria/02c_protocolo_y_qa.md` | fase 2c: 4 defectos K-01 a K-04 + QA independiente |
| `DECISIONS.md` | acta **D-090** |

### Cómo reproducir el inventario

```bash
/home/carlos/venv/bin/python scripts/auditoria_v1v37_inventario.py --verificar   # 6/6, no escribe nada
/home/carlos/venv/bin/python scripts/auditoria_v1v37_inventario.py             # escribe los 3 artefactos
```

---

*Auditoría de evidencia, no de opinión: cada afirmación cita su artefacto. Verificado sobre el commit
`06c8828`, rama `proyecto-kaggle-age-2026-v32`. Campeón V30 + plug-in congelado e intacto en
0,6159732487657944.*
