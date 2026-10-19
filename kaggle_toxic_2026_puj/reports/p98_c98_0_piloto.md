# P98-10 (C98-0): piloto real sobre Wikipedia 5M

- Corpus: wikipedia_es_5M (5,011,404 tokens, 1694 docs; val 2% por documento). Vocab top-50000 desde cero: 173,942 tipos; cobertura 96.39%; 254 chars.
- Modelo piloto LTR: 14,580,696 parametros (512x2).

| Epoca | loss (ini -> fin) | val ppl | tok/s | s |
|---|---|---|---|---|
| 1 | 10.907 -> 6.487 | 651.3 | 23,546 | 207 |
| 2 | 6.484 -> 6.094 | 448.9 | 27,205 | 179 |
| 3 | 6.111 -> 5.828 | 351.1 | 27,406 | 178 |

- VRAM pico: 9.81 GB | umbral requerido para C98-1: 13,333 tok/s (30M x 4 epocas x direccion en <=2.5 h).
- **Eleccion C98-1**: hidden 1024 x 2 capas (17,133 tok/s medidos).
- Veredicto: **PILOTO_OK** | tiempo total 588.3 s (~0.163 GPU-h, dentro del tope de 2 GPU-h del piloto).

Sin corpus nuevos; sin etiquetas de la competencia; mismo pipeline que se reutilizara en C98-1.
