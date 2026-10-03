# Taxonomia de errores v4

Generado por `experiments/35_discovery_v4.py` el 2026-09-26. Conteos sobre el desarrollo corregido (n=1.483) con la referencia V03 y sus umbrales congelados.

| Categoria | n | % del desarrollo | ids de ejemplo |
|---|---|---|---|
| FP_HS | 173 | 11.67 | 23930;21532;24427;23475;23543;22307;20157;24128 |
| FN_HS | 143 | 9.64 | 22678;20379;21145;20144;20956;21904;21643;23631 |
| FP_TR | 82 | 5.53 | 23930;23543;23414;22049;24101;20935;23839;24213 |
| FN_TR | 82 | 5.53 | 20956;21904;22883;22620;22163;22168;22065;24864 |
| FP_AG | 135 | 9.1 | 22434;23930;21532;24427;23475;22307;23414;22634 |
| FN_AG | 142 | 9.58 | 22678;20379;23719;20144;20337;20956;24441;21643 |
| ALL_THREE | 98 | 6.61 | 23930;23414;22049;20956;20935;23320;21892;23068 |
| IMPLICIT | 143 | 9.64 | 22678;20379;21145;20144;20956;21904;21643;23631 |
| QUOTED | 16 | 1.08 | 24039;24596;24441;20707;24638;22203;22228;21522 |
| COLLECTIVE | 142 | 9.58 | 22678;20379;23719;20144;20337;20956;24441;21643 |
| DIRECTED | 82 | 5.53 | 20956;21904;22883;22620;22163;22168;22065;24864 |
| AFFILIATIVE | 83 | 5.6 | 23930;21532;23475;22307;20034;23414;24982;20935 |
| NEWS | 156 | 10.52 | 22434;21532;24427;20379;22307;20157;24128;24596 |

## Robustez por subconjunto de duplicados (V4-F02)

| Subconjunto | n | F1 media | F1 HS/TR/AG | error medio por etiqueta |
|---|---|---|---|---|
| todos | 1483 | 0.7772 | 0.7617/0.8157/0.7542 | 0.1702 |
| textos_unicos | 1415 | 0.7568 | 0.7435/0.7929/0.7342 | 0.1767 |
| en_cluster_near_duplicado | 68 | 0.9769 | 0.9804/0.9903/0.96 | 0.0343 |

## Desacuerdo con el vecino mas cercano (V4-F03)

- Ejemplos cuyo vecino mas cercano discrepa en 2 de 3 etiquetas o mas: **684**.
- Error medio dentro: **0.1798**; fuera: **0.1619**; delta 0.018 con IC95 [-0.013, 0.0491].

## Lectura

Las categorias QUOTED, COLLECTIVE, AFFILIATIVE y NEWS son subconjuntos solapados de los errores por etiqueta, no categorias mutuamente excluyentes: sirven para leer donde se concentra el error, no para cuantificar una particion.
