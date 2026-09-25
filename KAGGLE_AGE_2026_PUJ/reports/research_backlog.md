# Research backlog — fase avanzada

Ordenado por **contenido informativo esperado**, justificado con la evidencia del proyecto (no por intuición). El orden reproduce el del brief (§86) y cada línea cita su evidencia.

## Contenido informativo ALTO

| ID | Hipótesis | Señal esperada | Coste | Evaluación | Estado |
|---|---|---|---|---|---|
| RB-01 | Las features supervisadas de menciones aportan lo que el TF-IDF no expresa | la identidad de la cuenta mencionada es la señal más fuerte medida (−0,125 al eliminarla) | medio | LOFO sobre el meta-modelo | pendiente |
| RB-02 | El perfil supervisado de la cuenta fuente de un RT aporta | el corpus tiene 30-53% de RTs y la fuente es un handle distinto del autor | medio | idem | pendiente |
| RB-03 | La rareza/novedad léxica (frecuencia de corpus in-fold) aporta sobre longitud y TTR | no existe ninguna feature de frecuencia de corpus hoy | bajo | idem | pendiente |
| RB-04 | NB-SVM (log-count ratio) supera a TF-IDF+SVM | método clásico de referencia en clasificación de texto corto, no evaluado aún | bajo | CV individual + ensamble | pendiente |
| RB-05 | Las distribuciones internas (percentiles, asimetría, curtosis) aportan sobre las medias | el EDA muestra diferencias de cola, no solo de media | bajo | idem | pendiente |

## Contenido informativo MEDIO

| ID | Hipótesis | Señal esperada | Coste | Evaluación | Estado |
|---|---|---|---|---|---|
| RB-06 | La posición de menciones/hashtags/URLs aporta estructura | solo se cuenta, no se ubica | bajo | LOFO | pendiente |
| RB-07 | La forma de palabra (ALLCAPS, capitalización, alfanuméricos) aporta ortografía | solo hay ratios agregados | bajo | LOFO | pendiente |
| RB-08 | La entropía por tipo (puntuación, vocabulario, entidades) aporta concentración | solo existen 2 entropías | bajo | LOFO | pendiente |
| RB-09 | La compresibilidad aporta estilo | no evaluada | bajo | LOFO | pendiente |
| RB-10 | El tokenizador propio de tweets mejora la representación de palabra | no evaluado | medio | CV individual | pendiente |
| RB-11 | Las features de incertidumbre (margen, entropía de scores, desacuerdo) mejoran el meta-modelo | el meta ya ve los scores pero no derivados explícitos | bajo | LOFO | pendiente |
| RB-12 | Un holdout congelado confirma los candidatos fuera del CV de desarrollo | metodología del brief §50 | bajo | evaluación aparte | pendiente |

## Contenido informativo BAJO (ya explorado con resultado negativo documentado)

| ID | Línea | Resultado previo | Decisión |
|---|---|---|---|
| RB-13 | Especialistas por longitud | +0,0000 (E-023/E-024) | cerrada salvo con features nuevas |
| RB-14 | BM25 sin normalización | −0,0002 (E-025) | cerrada |
| RB-15 | kNN sobre SVD | 0,3088 individual, sin aporte (E-026) | cerrada |
| RB-16 | Meta-modelo LightGBM | 0,5858 vs 0,5860 LogReg (E-022) | cerrada |
| RB-17 | Jerarquía gruesa→fina | +0,0003 redundante (E-028b) | cerrada |
| RB-18 | Deep learning denso como clasificador | 0,3381/0,4071 (E-019) | cerrada (se conserva solo como miembro diverso) |

## Líneas que REQUIEREN verificación de reglas (no iniciadas)

| ID | Línea | Riesgo | Estado |
|---|---|---|---|
| RB-R1 | Retrieval train→test en inferencia (exact/near-duplicate lookup) | El brief lo autoriza solo si las reglas lo permiten; la competencia prohíbe usar el test para entrenar, pero un lookup train→test es inferencia, no entrenamiento | **REQUIERE_VERIFICACIÓN** |
| RB-R2 | POS tagger preentrenado de spaCy | recurso preentrenado en zona gris (D-001) | **REQUIERE_VERIFICACIÓN** |
| RB-R3 | Lexicones externos | no aportan con corpus ASCII y su licencia es dudosa | no iniciada |