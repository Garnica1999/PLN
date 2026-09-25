# Gap analysis v2 — espacio de features y señal ausente

Generado por `scripts/phase0_freeze_gap_v2.py`. Cada fila indica si el fenómeno ya está detectado, representado, si tiene versión supervisada, en qué canal entra al modelo, qué dice el análisis de errores y **qué brecha queda**. La última columna es la hipótesis nueva.

> **Propósito metodológico.** Este documento evita re-ejecutar líneas ya medidas. La columna `Δ medido` recoge el aporte real al ensamble cuando existe.

| Fenómeno | Detectado | Representado | Representación actual | Supervisado | Canal | Análisis de errores | Brecha | Hipótesis nueva |
|---|---|---|---|---|---|---|---|---|
| longitud | sí | sí | n_chars + 26 de distribución | no aplica | todos los miembros | domina el error (52,5% en Q1) | ninguna | ¿queda señal tras descontar longitud? |
| riqueza léxica | sí | sí | TTR, CTTR, hapax, entropía, MATTR, Yule's K, Maas, Simpson | no | miembros + meta | redundante con longitud | baja | residualización (L2) |
| rareza léxica | sí | sí | 12 features (frecuencia de corpus in-fold) | sí (corpus de train) | meta | aporta poco | +0,0003 medido | ¿por tipo de tweet? |
| menciones | sí | sí | TF-IDF + 17 features supervisadas | sí P(clase|mención) | canal + meta | la señal más fuerte (−0,125 al eliminarlas) | baja | ¿margen específico por frontera? |
| hashtags | sí | sí | TF-IDF + 17 supervisadas | sí | canal + meta | neutro | −0,0001 medido | ¿representación sin # o n-gramas del hashtag? |
| RT | sí | sí | is_rt + 17 de fuente supervisada + pares | sí P(clase|fuente) | miembros + meta | RT más frecuente en mayores | +0,0007 medido | ¿RT puro vs RT con comentario? |
| URLs | sí | sí | conteo, densidad, posición, longitud | no | meta | más URL en los errores | medido | sin dominio (todas t.co) |
| dominios | sí (EDA) | no aplica | — | no | — | no hay dominios distintos: todo es t.co | vacío por datos | cerrada |
| puntuación | sí | sí | densidad por tipo, transiciones, entropía | no | meta | quitarla cuesta −0,008 | medido | ¿transiciones de carácter (L4)? |
| ortografía | sí | sí | 10 de forma de palabra, elongaciones, mayúsculas | no | meta | informalidad en clases jóvenes | medido | ¿distribución y no solo frecuencia? |
| repetición | sí | sí | runs, ratios de unigramas/bigramas/trigramas | no | meta | jajaja característico de jóvenes | medido | cerrada salvo repetición de largo alcance |
| entropía | sí | sí | carácter, token, puntuación, entidad | no | meta | aportó poco | medido | cerrada |
| distribución | sí | sí | percentiles, asimetría, curtosis, CV, IQR, MAD | no | meta | colas distintas por clase | +0,0006 (bloque) | residualización (L2) |
| posición | sí | sí | 20 features de posición normalizada | no | meta | nunca explotada antes | medido | cerrada |
| OOV | sí | sí | ratio OOV y de novedad léxica | sí (vocabulario de train) | meta | no explorado antes | +0,0003 (bloque) | cerrada |
| ordinalidad | sí | sí | acumulativo, pares adyacentes, jerarquía, léxico por clase | parcial | miembros + meta | fronteras medidas (0,486-0,601) | +0,0006 (pares) | ¿scores pairwise como features del meta? |
| perfiles léxicos | sí (EDA) | sí | log-odds por clase (10 features) | sí | meta | términos distintivos muy interpretables | +0,0001 | cerrada |
| tipos de tweet | parcial | no | — | no | — | especialistas por longitud dieron 0 | **sin taxonomía explícita** | **L5: tipología + F1 por tipo** |
| incertidumbre | sí | sí | 51 features de confianza, margen, entropía, acuerdo | no | meta | aportó poco | +0,0002 | cerrada |
| diversidad de modelos | sí | parcial | tasas de desacuerdo puntuales | no | — | ComplementNB y MLP aportan | **sin matriz formal** | **L6/§48: matriz de complementariedad** |
| near duplicates | sí | sí (CV) | clústeres Jaccard ≥ 0,8 para agrupar folds | no | — | 1,21% literal / 4,26% en clústeres | usado en CV, no como feature | retrieval bloqueado por reglas |
| errores universales | no | no | — | no | — | nunca analizado | **ausente** | **L1: intersección de errores** |

## Líneas con brecha confirmada (las únicas que aborda esta fase)

1. **L1 — Errores universales y de intersección** (ausente): nunca se midió cuántos tweets tienen al menos un modelo base acertando mientras el ensamble falla. Es el diagnóstico que decide si el problema es de información o de ponderación.
2. **L2 — Residualización por longitud** (ausente): la longitud domina el error y confunde a las features de riqueza; el residuo puede contener señal de edad ortogonal.
3. **L3 — Separación contenido/estilo** (sin representación explícita): el contenido se comparte entre cohortes (RT de humor y deportes en 18-24 y 25-34, ya documentado); el estilo podría conservar señal demográfica.
4. **L4 — Transiciones de caracteres** (ausente): patrones letra→dígito, letra→puntuación, espacio→puntuación que el TF-IDF no expresa como feature explícita.
5. **L5 — Tipología de tweets y umbral de información** (sin taxonomía explícita): el error puede depender del tipo estructural, no solo de la longitud.
6. **L6 — Matriz de complementariedad y ponderación condicional** (sin matriz formal): guiada por L1.
7. **L7 — NB-SVM por canal** (extensión barata de algo ya validado: +0,0007 en char+mención).

## Líneas cerradas con evidencia (no se repiten)

| Línea | Δ medido | Evidencia |
|---|---|---|
| Mención supervisada | +0,0016 | ADV-02 (ya en v09) |
| RT source supervisado | +0,0007 | ADV-02 |
| Rareza léxica | +0,0003 | ADV-02 |
| Léxico por clase | +0,0001 | ADV-02 |
| Hashtag supervisado | −0,0001 | ADV-02 |
| Distribución/formas/posición | +0,0006 | ADV-02 |
| Incertidumbre e interacciones | +0,0002 | ADV-04 |
| Selección de features | 0,0000 | ADV-05 |
| NB-SVM como modelo | −0,0017 | ADV-05 (se conserva como miembro) |
| Especialistas por longitud | +0,0000 | E-023/E-024 |
| Frontera con compuerta dura | −0,0036 | ADV-07 |
| Enrutamiento por margen (par-SVM) | 0,0000 | ADV-07 |
| BM25 sin normalización · kNN sobre SVD · meta LightGBM · jerarquía gruesa | ≤ 0 | E-022/E-025/E-026/E-028b |
