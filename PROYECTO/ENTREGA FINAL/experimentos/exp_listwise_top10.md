# Sonda listwise sobre el top-10 (M7 v13)

Fecha: 2026-11-01T01:51:34-05:00. Casos: 90. Modelo: `qwen3.8:diario-8k-q3_k_s`.

| Metrica | Valor |
|---|---:|
| nDCG@10 de v2.2 | 0.6926 |
| nDCG@10 del orden del LLM | 0.6764 |
| Delta pareado | -0.0162 |
| IC 95 % | [-0.0331; +0.0004] |
| P(mejora) | 0.029 |
| Parseos validos | 87 de 90 |
| Latencia media | 3380 ms |
| Techo de reorden perfecto | 0.7359 |

## Lectura

Si el LLM no supera a v2.2, el orden actual ya es competitivo y el margen
solo se recupera cambiando el conjunto de candidatos (promover los articulos
de las posiciones 11 a 50), no reordenando el top-10.
