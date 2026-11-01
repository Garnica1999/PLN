# Cobertura NER juridico (M2)

Fecha: 2026-09-08
Documentos: 23742 (T/C/SU)
Tiempo total: 249.0s

## Precision muestral (100 docs seed 42, 20 casos por tipo)
- arts_cp: 1.00
- normas: 1.00
- sents_resuelve_corpus: 0.90
- sents_rango_valido: 1.00
- ponente: 1.00

## Cobertura global (% docs con >=1)
- cita CP: 98.71
- norma: 99.67
- sentencia citada: 99.94
- ponente: 85.38

## Por tipo T/C/SU (%)
| tipo | cita_CP | norma | sentencia | ponente | n |
| --- | ---: | ---: | ---: | ---: | ---: |
| C | 100.00 | 99.70 | 99.98 | 86.20 | 6246 |
| SU | 99.50 | 100.00 | 99.75 | 75.57 | 397 |
| T | 98.22 | 99.65 | 99.93 | 85.31 | 17099 |

## Por periodo (%)
| periodo | cita_CP | norma | sentencia | ponente | n |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1992-1996 | 99.42 | 98.65 | 99.53 | 92.30 | 2739 |
| 1997-2001 | 95.39 | 99.21 | 100.00 | 96.05 | 4427 |
| 2002-2006 | 98.47 | 99.96 | 99.98 | 90.44 | 4918 |
| 2007-2011 | 99.81 | 99.98 | 100.00 | 88.60 | 4861 |
| 2012-2016 | 100.00 | 100.00 | 100.00 | 76.61 | 4360 |
| 2017-2021 | 99.92 | 99.88 | 100.00 | 57.28 | 2437 |

## Top-20 articulos CP citados (n_docs)
- Art. 241: 20289 docs
- Art. 86: 16405 docs
- Art. 13: 8757 docs
- Art. 2: 7818 docs
- Art. 1: 7710 docs
- Art. 9: 7338 docs
- Art. 31: 5483 docs
- Art. 29: 5322 docs
- Art. 33: 5117 docs
- Art. 4: 4937 docs
- Art. 48: 3789 docs
- Art. 53: 3645 docs
- Art. 5: 3636 docs
- Art. 6: 3565 docs
- Art. 150: 3336 docs
- Art. 44: 2878 docs
- Art. 11: 2812 docs
- Art. 34: 2795 docs
- Art. 35: 2740 docs
- Art. 25: 2624 docs

## Top-10 ponentes (n_docs)
- Alvaro Tafur Galvis: 931 docs
- Marco Gerardo Monroy Cabra: 862 docs
- Humberto Antonio Sierra Porto: 846 docs
- Rodrigo Escobar Gil: 844 docs
- Alfredo Beltrán Sierra: 840 docs
- Jaime Córdoba Triviño: 803 docs
- Jorge Ignacio Pretelt Chaljub: 767 docs
- Nilson Pinilla Pinilla: 756 docs
- Gabriel Eduardo Mendoza Martelo: 728 docs
- Luis Ernesto Vargas Silva: 719 docs

## Grafo sentencia-sentencia (M5)
- Aristas unicas doc->cita (con self-loops): 540181
- Aristas unicas sin self-loops: 516529
- Menciones totales con duplicados: 854641
- Menciones unicas por doc sumadas: 540181
- Menciones que resuelven al corpus: 88.85%
- Docs que se autocitan: 23652 (99.62%)
- Nota M5: la autocita bruta incluye la mencion del propio encabezado; los self-loops deben excluirse del grafo.
- Docs con >=2 menciones propias (proxy autocita en cuerpo): 9960 (41.95%)

## Descartes y diagnosticos
- decreto_legislativo_mencion: 7564
- norma_ano_fuera_rango: 112
- norma_num_invalido: 4
- sent_ano_fuera_rango: 1351
- sent_ano_invalido: 533
- sent_num_invalido: 13
