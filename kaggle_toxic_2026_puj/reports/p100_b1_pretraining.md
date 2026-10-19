# P100-11 (P100-B1): preentrenamiento sobre 30M (social 15% + conversacional 30% + general 55%)

- Corpus: 29,727,450 tokens (162977 documentos; sha 5111a1832fe9567b...); composicion {'social': 4227450, 'mc4': 4850000, 'conversacional': 9000000, 'wikipedia': 11650000}; vocab congelado de C98-1.
- Val: medidor fijo de 3320 documentos (mismos de P98, materializados desde C98-1); 3189 excluidos del stream de entrenamiento.
- Diagnostico chatsubs: 500 docs (sin efecto en la parada).

| Dir | Epoca | val ppl | ppl wiki/mC4/twitter | ppl chatsubs | tok/s |
|---|---|---|---|---|---|
| LTR | 3 | 111.4 | n/a (reconstruida) | n/a | 17,325 |
| LTR | 4 | 98.7 | 90.7/207.1/68.3 | n/a | 17,340 |
| RTL | 1 | 205.7 | 198.7/338.5/108.1 | 30.0 | 17,354 |
| RTL | 2 | 138.7 | 130.8/255.7/79.6 | 25.3 | 17,354 |
| RTL | 3 | 112.0 | 104.3/219.8/69.2 | 23.5 | 17,424 |
| RTL | 4 | 98.6 | 90.9/201.0/64.0 | 22.3 | 17,300 |

- Checkpoints: LTR ppl 98.7; RTL ppl 98.6; GPU-h 2.719 (esperado ~3.9; tope de fase 10).

La perplexity NO es criterio de adopcion; el gate downstream es el de reports/p100_preregistration.md seccion 8.
