# V14 · A — Variables lingüísticas del analizador español

## 1. Qué cambia respecto a V13

D-043 relaja §3 para permitir un recurso lingüístico externo. El recurso es `es_core_news_md` de spaCy: un analizador morfosintáctico (**no** transformer, **no** LLM, **no** embeddings de clasificación) que aporta análisis morfológico, lemas, entidades nombradas y dependencias. El bloque es determinista y **no** usa etiquetas, así que no tiene ninguna vía de leakage.

Procedencia: `es_core_news_md` instalado en el venv, SHA256 de `meta.json` = `889a2fd05d217fa7c1bcffa867af7fc30911bc7b48523f87b5b2525846553ef5`.

**No es la línea de D-026** (26 recuentos de etiqueta POS, −0,0032): aquí se usan los otros componentes del mismo modelo, que son otro tipo de variable.

## 2. Cobertura del análisis

| Magnitud | Media por documento | Documentos con valor | Máximo |
|---|---|---|---|
| verbos | 1.92 | 77.90 % | 18 |
| sustantivos | 5.895 | 97.39 % | 32 |
| entidades | 3.372 | 81.13 % | 58 |
| tokens | 12.966 | 99.91 % | 58 |
| lemas | 12.018 | 99.91 % | 31 |
| frases | 1.541 | 99.94 % | 16 |
| personas | 0.488 | 24.96 % | 18 |
| organizaciones | 0.492 | 27.16 % | 23 |
| lugares | 0.525 | 26.15 % | 22 |

## 3. Características

37 columnas en cinco familias: morfología verbal (tiempo, aspecto, modo, persona), morfología nominal (género, número, diminutivo, aumentativo), lemas (diversidad y hapax por lema), entidades nombradas (reparto por tipo) y sintaxis (profundidad de dependencia, coordinación, subordinación, pasiva, longitud de frase).

## 4. Cribado de redundancia contra las 279 variables de V12

- Variables redundantes: **4** de 37
- Correlación máxima: **0.9824** · mediana 0.3069

## 5. Control block-only

| Modelo | macro-F1 | std | Suelo de azar | Supera el azar |
|---|---|---|---|---|
| logreg | 0.2409 | 0.0009 | 0.1667 | si |
| linearsvc | 0.227 | 0.0019 | 0.1667 | si |

## 6. Puerta de rescate

- Rescate **12,444** · daño **71,126** · **neto -58,682** (umbral 1,014)

- 18-24|25-34: rescate 4,228 · daño 22,260 · neto -18,032
- 25-34|35-49: rescate 2,760 · daño 24,894 · neto -22,134

## 7. Decisión

**La puerta no se supera**: el bloque no justifica pagar la evaluación del meta-modelo. Se cierra la pista.

