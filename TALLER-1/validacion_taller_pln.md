# Informe de validación — Taller PLN "Análisis de discursos presidenciales en Colombia"

Fecha inicial: 2026-08-19 · Actualizado: 2026-08-20
Notebook original: `Copia_de_Taller_PLN.ipynb` (81 celdas)
Notebook final: `Taller_Presidentes_final.ipynb` (134 celdas, 10/10 puntos, 0 errores de ejecución)
Enunciado: `Taller_Presidentes_2610.pdf` (10 puntos)
Método: revisión técnica puntual + verificación empírica ejecutando el corpus real
(entorno: NLTK 3.9.4, spaCy 3.8.13, es_core_news_md, textstat 0.7.13, textdescriptives 2.8.4, corpus presidentes 1942-2026).

---

## 1. Veredicto final por punto del taller

| Punto | Estado inicial | Estado final |
|---|---|---|
| 1. Exploración inicial | CUMPLE PARCIAL (sin value_counts) | ✅ CUMPLE (value_counts agregado) |
| 2. Limpieza + doble vía NLTK/spaCy | CUMPLE PARCIAL (tildes dañaban spaCy; sin conclusión) | ✅ CUMPLE (vía paralela con tildes; conclusión explícita: lemas de spaCy para análisis léxicos) |
| 3. Frecuencia léxica | CUMPLE | ✅ CUMPLE (+ términos distintivos por keyness, normalización) |
| 4. Nube de palabras | CUMPLE PARCIAL (solo por corriente) | ✅ CUMPLE (corriente + por década, como pide "presidente o periodo") |
| 5. N-gramas | CUMPLE PARCIAL (sin Matcher) | ✅ CUMPLE (n-gramas NLTK vs spaCy + Matcher con 3 patrones + comparación) |
| 6. Diversidad léxica | NO CUMPLE | ✅ CUMPLE (TTR, MTLD, Entropía de Shannon por presidente) |
| 7. Estilometría | NO CUMPLE | ✅ CUMPLE (textstat: Flesch, Fog, SMOG, ARI, Coleman-Liau, Dale-Chall + textdescriptives + comparación) |
| 8. Análisis temporal | NO CUMPLE | ✅ CUMPLE (por año: total palabras, TTR, términos clave, legibilidad; gráficos y CSV) |
| 9. Entidades nombradas (NER) | NO CUMPLE | ✅ CUMPLE (PER/LOC/ORG por tipo y por década, 1.009 entidades) |
| 10. Grafos NetworkX | NO CUMPLE | ✅ CUMPLE (sujeto-verbo nsubj y sujeto-adjetivo, lemas, aristas ponderadas) |

## 2. Cifras verificadas empíricamente (estado final, post-limpieza del corpus)

- Conteos del punto 2 (discurso de Petro): 4.573 palabras = tokens_nltk 2.261 + stopwords_nltk 2.312 = tokens_spacy 1.884 + stopwords_spacy 2.707. Consistente.
- Sección 5.2 (stopwords): intersección 35,6-55,1 %; únicas spaCy 44,9-61,8 %; únicas NLTK 0-4,4 %.
- Corpus tras limpieza: 53.938 tokens NLTK (antes 53.998); 1009 entidades NER; 525+108+8+10+11 correcciones de artefactos aplicadas (ver `procesamiento_corpus.md`).
- Reproducibilidad: CSVs byte-idénticos en re-ejecuciones; cifras de markdowns verificadas contra CSVs por revisor independiente.

## 3. Dictamen final de la limpieza de segmentos y tokens

**La limpieza quedó CORRECTA para ambos pipelines tras las correcciones:**

1. `limpiar_texto` (NLTK): minúsculas, sin puntuación/números, sin tildes — correcta para tokens crudos y conteos (enunciado literal del punto 2).
2. Vía paralela `discurso_spacy` (spaCy): minúsculas, CON tildes y puntuación — correcta para lemas, POS, NER, dependencias, Matcher y textdescriptives.
3. **Limpieza avanzada del corpus** (de-hyphenation + P1-P4): corrige artefactos del scan original (guiones de fin de línea, pies de página, errores OCR de dígitos, "n" suelta, duplicados funcionales) — todo con validación léxica para evitar pseudopalabras.

## 4. Impacto de la limpieza avanzada en los puntos (resumen)

- 9 de 10 puntos con cambios materiales; el punto 1 (exploración) no depende del texto.
- Cambios principales: top-20 de Turbay (10 posiciones: salen "n", "agosto", "1978") y López Michelsen (5); total de palabras 1974: 5.500→5.485, 1978: 7.574→7.543; NER 1.007→1.009 entidades; Flesch de los 3 discursos con scan ajustado (±0,1-0,2 puntos).
- Detalle completo por punto en `procesamiento_corpus.md`.

## 5. Artefactos de salida

- `Taller_Presidentes_final.ipynb` — notebook completo (134 celdas, ejecutado sin errores).
- `images/` — 20 PNG (todas las figuras, dpi=150) + 16 CSV (datos tabulares, utf-8-sig).
- `errores_encontrados.md` — registro de errores y correcciones.
- `procesamiento_corpus.md` — detalle de la limpieza del corpus y su impacto.
