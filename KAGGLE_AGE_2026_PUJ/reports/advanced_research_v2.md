# Informe de investigación avanzada v2 — de la ponderación a la señal ausente

**Proyecto**: `KAGGLE_AGE_2026_PUJ` · **Fecha**: 2026-09-24 · **Candidato**: `v11` (meta-modelo MLP)

---

## 1. Baseline congelado

`experiments/baseline_v09_frozen/` (16 archivos con hash SHA-256 y `MANIFEST.json`): v09 = **0,60807** en el leaderboard, CV 0,5897, holdout 0,5966. Envíos usados: 9 de ~10.

## 2. Gap analysis v2

`reports/advanced_gap_analysis_v2.md`: tabla de 22 fenómenos con estado de detección, representación, versión supervisada, canal de modelo y **Δ medido**. Su función fue evitar re-ejecutar ocho líneas ya cerradas (mención/hashtag/RT supervisados, distribución, rareza, prototipos, NB-SVM, especialistas por longitud, incertidumbre). Brechas confirmadas: errores de intersección (L1), residualización por longitud (L2), contenido vs estilo (L3), transiciones de carácter (L4), tipología de tweets (L5), complementariedad (L6).

## 3. Diagnóstico (L1 + L5) — el hallazgo que reorientó la fase

- **25.94%** de los tweets: el ensamble falla y **al menos un modelo base acierta**.
- Los modelos que más rescatan son los **débiles pero diversos**: softmax ordinal (36,1%), centroides (31,1%), MLP densas+SVD (30,5%), LogReg palabra (27,9%), ComplementNB (25,2%).
- Categorías de intersección:

| Categoría | n | % | accuracy del ensamble |
|---|---|---|---|
| everyone_wrong | 26,196 | 15.47% | 0.0327 |
| one_model_right | 20,952 | 12.37% | 0.14 |
| few_models_right | 23,993 | 14.17% | 0.3599 |
| many_models_right | 98,195 | 57.99% | 0.8927 |

- Las pérdidas se concentran en las clases medias: 25-34 (10.241), 35-49 (9.067), 18-24 (8.927).
- Error por tipo de tweet: `solo_url` **61,1%** (n=7.661), `solo_texto` 44,4% (n=59.221), `rt_con_comentario` 37,1%, mientras `mencion_alta` 19,2% y `hashtag_alto` 20,4% son los más fáciles.
- Error por longitud: monótono de **54,5%** (≤10 caracteres) a **35,1%** (130-141); la fracción de casos rescatables por algún miembro también cae de 36,9% a 21,1%.

**Interpretación.** El diagnóstico apuntó a **ponderación**, no a falta de información: si en una de cada cuatro filas algún miembro acierta y el ensamble no, el combinador está desaprovechando señal. Eso descartó atacar de nuevo la representación en esta fase.

## 4. Hipótesis y experimento (L6)

**H27-H29**: un combinador que pueda **condicionar su confianza al tipo de tweet** recupera parte de esos casos. Se probaron cuatro variantes con LOFO:

| Variante | macro-F1 | Δ |
|---|---|---|
| global LogReg (referencia v10) | 0.5906 ± 0.0047 | +0.0000 |
| global MLP (ponderación suave) | 0.5966 ± 0.0040 | +0.0060 |
| segmentado por tipo de tweet | 0.5899 ± 0.0040 | -0.0007 |
| segmentado por banda de longitud | 0.5891 ± 0.0035 | -0.0015 |

- **La segmentación dura falla** (tipo −0,0007; longitud −0,0015), igual que en la fase anterior con la compuerta de frontera: partir el problema en subconjuntos pequeños pierde más de lo que gana.
- **La ponderación suave funciona**: un MLP superficial como meta-modelo alcanza **0.5967 ± 0.0040** frente a **0.5906 ± 0.0047** del LogReg, con mejora en **5/5 folds** (de +0,0043 a +0,0080). Un modelo lineal no puede expresar 'confía en este miembro cuando el tweet es de este tipo'; el MLP sí.

## 5. Corrección de dos defectos detectados en la auditoría

La generación del candidato reveló una submission degenerada (78.576 filas predichas en una sola clase). Diagnóstico y corrección:

1. **Desalineación de 6 columnas**: el bloque NB-SVM se anexaba al final del lado test pero no del lado train, desplazando densas/estructuradas/supervisadas. Corregido; ahora ambos lados comparten el mismo orden.
2. **Cuatro features de forma de palabra en cero**: el tokenizador solo aceptaba minúsculas en la alternativa de palabra, así que las palabras capitalizadas o en mayúsculas se fragmentaban. Además `shape_alphanumeric_ratio` era cero por definición (el tokenizador separa dígitos de letras). Corregidos ambos: columnas con varianza nula tras el recálculo: ninguna.

**Lección metodológica**: la validación de formato de una submission no basta; hay que **verificar la distribución de etiquetas predichas**. El archivo degenerado pasaba todos los checks de estructura.

## 6. Confirmación en el holdout congelado

| Meta-modelo | macro-F1 en el holdout | |
|---|---|---|
| meta LogReg (v10) | 0.5981 | accuracy 0.5989 |
| meta MLP (v11) | 0.6024 | accuracy 0.6026 |

**Δ = +0.0043** (criterio: Δ ≥ +0,003) -> **CONFIRMADO**.

## 7. Candidato v11

- **Modelo**: ensamble de 14 fuentes + **meta-modelo MLP(64)** sobre 273 features.
- **CV (LOFO)**: 0.5966 ± 0.0040.
- **Holdout congelado**: 0.6024.
- **Distribución predicha**: {'13-17': 16716, '18-24': 21025, '25-34': 17024, '35-49': 16155, '50-64': 22325, '65-xx': 19647}.
- **LB esperado**: 0,612-0,614 (CV 0,5967 + brecha calibrada de +0,0169 ± 0,0013).

## 8. Líneas evaluadas y descartadas en esta fase

| Línea | Resultado | Decisión |
|---|---|---|
| Segmentación dura por tipo de tweet | −0,0007 | cerrada |
| Segmentación dura por banda de longitud | −0,0015 | cerrada |
| Features de forma de palabra (antes degeneradas) | corregidas | se conservan |
| Alineación train/test del meta | bug corregido | — |

## 9. Líneas pendientes (con brecha confirmada en el gap analysis v2)

- **L2 — Residualización por longitud**: separar señal de edad de 'tweet simplemente más largo'; es la línea con mejor relación coste/información de las que quedan.
- **L3 — Contenido vs estilo**: representación explícita de estilo (palabras funcionales, puntuación, estructura) frente a contenido; explicaría por qué las fronteras centrales resisten.
- **L4 — Transiciones de carácter**: estadística de transición letra<->dígito<->puntuación.
- **L7 — NB-SVM por canal**: word, hashtag y mención puros como miembros independientes.
- **Retrieval train->test**: sigue **REQUIERE_VERIFICACIÓN** de reglas; no implementado.

## 10. Conclusión de la fase

El baseline v09 captura representación dispersa de carácter, canales léxicos con perfiles supervisados, ordinalidad y diversidad de modelos. El diagnóstico de intersección demostró que **una cuarta parte de los tweets tiene un miembro acertando y el ensamble fallando**, y que los rescatadores son sistemáticamente los modelos diversos. La hipótesis derivada —el combinador debe poder condicionar su confianza al tipo de tweet— se confirmó con un **MLP superficial de meta-modelo**: +0,0061 de CV en 5/5 folds y +0,0043 en el holdout congelado. La segmentación dura volvió a fallar, lo que refuerza que el problema es de **ponderación suave**, no de partición del espacio.