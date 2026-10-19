# P107-01: Exploracion de arquitecturas no lineales, aumento de datos desde cero y FastText/CharCNN

Fecha: 2026-10-14. Duracion: 329.1 s. Dispositivo: cuda.

## Resumen de Resultados

| Variante | delta Canonico | Pliegues Pos. | Celdas Cambiadas | IC95 Bootstrap | delta Part. 47 | Veredicto Gate A3 |
|---|---:|---:|---:|---|---:|---|
| V1_aumento_caracteres | -0.001101 | 2/5 | 93 | [-0.007384, +0.005009] | -0.001698 | **GATE_FAILED** |
| V2_mixup_features | -0.016623 | 1/5 | 273 | [-0.028634, -0.004500] | -0.014387 | **GATE_FAILED** |
| V3_char_cnn_lightweight | -0.000093 | 3/5 | 19 | [-0.002479, +0.002352] | +0.001152 | **GATE_FAILED** |
| V4_fasttext_scratch | -0.001586 | 1/5 | 13 | [-0.003713, +0.000392] | +0.001041 | **GATE_FAILED** |

## Conclusiones Cientificas
- V1 (Aumento de datos ortografico con reajuste de vocabulario): Reajustar los vectorizadores sobre datos perturbados sintetizados dentro del pliegue expande la dimensionalidad con n-gramas ruidosos de baja frecuencia y diluye los pesos TF-IDF optimos de los terminos discriminantes.
- V2 (Mixup continuo en TF-IDF): La interpolacion lineal sobre representaciones dispersas con objetivos suaves reduce la separabilidad n-grama natural de los tweets sin ofrecer ganancia medible.
- V3 (Lightweight 1D CharCNN en PyTorch, libre de fuga): La red convolucional de caracteres entrenada con validacion interna anidada alcanza ~0.53 OOF standalone y al combinarse residualmente (90/10) con V03 no supera el gate A3 (-0.000093 en canonica).
- V4 (FastText desde cero con PyTorch): La arquitectura FastText basada en EmbeddingBag de palabras y subpalabras n-gramas (3-4 ngrams) entrenada desde cero sin embeddings externos obtiene ~0.55 OOF standalone y al combinarse de forma residual con V03 no supera a la combinacion lineal analitica de V03.

V03 remains the best verified leaderboard result.
