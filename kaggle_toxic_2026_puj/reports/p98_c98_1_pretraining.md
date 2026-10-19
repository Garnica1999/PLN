# P98-12 (C98-1): preentrenamiento LTR y RTL sobre 30M tokens

- Corpus: 29,737,183 tokens, 166022 documentos (sha256 9f562718ef4e7d45...).
- Vocab top-50k: cobertura 93.76%; 254 chars.
- Validacion: 3320 documentos (2% por fuente).

| Direccion | Epoca | loss (muestra ini -> fin) | val ppl | tok/s | s |
|---|---|---|---|---|---|
| LTR | 1 | 10.831 -> 5.291 | 185.0 | 17,316 | 1693 |
| LTR | 2 | 5.290 -> 4.868 | 126.4 | 17,404 | 1684 |
| LTR | 3 | 4.937 -> 4.641 | 105.6 | 17,435 | 1681 |
| LTR | 4 | 4.774 -> 4.462 | 94.8 | 17,529 | 1672 |
| RTL | 1 | 10.828 -> 5.227 | 180.4 | 17,059 | 1718 |
| RTL | 2 | 5.265 -> 4.784 | 123.4 | 17,344 | 1690 |
| RTL | 3 | 4.935 -> 4.550 | 102.2 | 17,261 | 1698 |
| RTL | 4 | 4.748 -> 4.375 | 91.7 | 17,275 | 1697 |

- Checkpoints: LTR val ppl 94.8 (4 epocas); RTL val ppl 91.7 (4 epocas).
- GPU-h C98-1: 3.775 (tope 6). La perplexity NO es criterio de adopcion (P98 seccion 10).

Siguiente: D98-1 (representaciones congeladas + clasificador por pliegue + fusion estilo v5.1 con umbrales de V03).
