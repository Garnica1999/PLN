# Gate A — Hipótesis priorizadas

Generado por `notebooks/03_linguistic_analysis.ipynb` (Fase 3).

## H1 — La textometría (longitud, riqueza léxica) aporta señal directa (prioridad alta)
- **Evidencia**: Kruskal epsilon^2: n_chars 0,126; cttr 0,066; lexical_entropy 0,065
- **Cómo se prueba**: E-002/E-010 (ablación textometría)

## H2 — El estilo informal separa sobre todo las clases extremas (prioridad media)
- **Evidencia**: mayúsculas, exclamaciones y elongaciones máximas en 13-17
- **Cómo se prueba**: E-011 (ablación estilo) + Fase 17 (error analysis)

## H3 — El comportamiento de retuit es señal de actividad por edad (prioridad media)
- **Evidencia**: RT: 53% en 65-xx vs 30% en 18-24
- **Cómo se prueba**: E-011

## H4 — Parte de la ganancia al añadir features es redundancia, no señal (prioridad alta)
- **Evidencia**: ttr~hapax_ratio rho=0,999; cttr~lexical_entropy 0,997
- **Cómo se prueba**: Fase 18 (ablación incremental obligatoria)

## H5 — Los 166 textos vacíos deben predecirse por prior de clase (prioridad baja)
- **Evidencia**: 104 vacíos en train, 62 en test
- **Cómo se prueba**: pipeline final

## H6 — Los char n-grams aportan señal ortográfica complementaria (prioridad alta)
- **Evidencia**: cuatrigramas distintivos con z alto (abreviaturas, terminaciones)
- **Cómo se prueba**: E-006 (word vs char vs union)

## H7 — El léxico temático discrimina clases (deportes/música vs noticias) (prioridad alta)
- **Evidencia**: términos distintivos por clase con z > 5
- **Cómo se prueba**: E-004/E-005

## H8 — La ortografía informal (tildes omitidas, letras repetidas) es señal (prioridad media)
- **Evidencia**: elongaciones y errores aparentes concentrados en 13-17
- **Cómo se prueba**: E-007 (variantes de normalización)

## H9 — Emojis y hashtags aportan sobre todo en clases extremas (prioridad media)
- **Evidencia**: emoji_diversity y has_emoji con chi2 alto; hashtags temáticos por clase
- **Cómo se prueba**: E-012 (ablación emojis/hashtags)

## H10 — URLs y menciones son marcadores de registro (más en mayores) (prioridad media)
- **Evidencia**: n_url 0,33 en 13-17 vs 0,51 en 65-xx
- **Cómo se prueba**: E-011

## H11 — La validación agrupada por clúster es obligatoria (prioridad alta)
- **Evidencia**: CV estratificado regala 0,061 de macro-F1 por memorización de duplicados
- **Cómo se prueba**: todos (protocolo fijo)

## H12 — El ajuste de prior/umbral por clase mejora macro-F1 (prioridad media)
- **Evidencia**: clases balanceadas pero errores asimétricos (65-xx con más RT)
- **Cómo se prueba**: Fase 19 (optimización de F1 multiclase)
