# P99-11 (S99-01): preentrenamiento sobre 60M tokens unicos

- Corpus: 60,707,427 tokens (340219 documentos; sha e627a7f33886c0a1...); vocab congelado de C98-1.
- Val: 3320 documentos (mismos de P98), por fuente.

| Dir | Epoca | val ppl | ppl wiki/mC4/twitter | tok/s |
|---|---|---|---|---|
| LTR | 1 | 136.9 | n/a (reconstruida) | 17,385 |
| LTR | 2 | 100.6 | 81.8/187.4/67.4 | 17,354 |
| RTL | 1 | 2408.3 | 2703.1/2465.5/1320.2 | 17,337 |
| RTL | 2 | 98.3 | 81.2/181.4/63.5 | 17,189 |

- Checkpoints: LTR ppl 100.6; RTL ppl 98.3; GPU-h 3.876 (tope 6).

La perplexity NO es criterio de adopcion (P99 seccion 9).

## Errata del gauge RTL (epoca 1)

El valor original de la epoca 1 del RTL (ppl 2408.3) era invalido (evaluado en orden LTR). Corregido con el checkpoint preservado y el stream RTL correcto: **ppl 135.6** (wiki 116.6 / mc4 244.8 / twitter 80.9). Controles: rtl final 98.3, ltr final 100.6.
