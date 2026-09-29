# V31+ - reproducibilidad

## Clasificacion

| Artefacto | Clase | Motivo |
|---|---|---|
| V30 (pesos medios) | **R3** | reproduce bit a bit desde el pool OOF y el manifest de pesos; confirmado contra el hash del fichero enviado |
| V30 (pesos por fold, cifra publicada) | **R1** | los pesos por fold no se persistieron: **no reproducible** |
| Canal linguistico OOF | R3 | determinista desde textos + folds; cacheado con hash del corpus registrado |
| Bancos de representaciones | R3 | funcion pura de (textos, clase, folds) |
| Integraciones N1 y N2 | R2 | Optuna con semilla fija; los pesos por fold si se guardan |

## La clase que cambio en esta fase

El baseline paso de R1 a **R3** al descubrirse que la cifra publicada no era reproducible. Ese es el hallazgo mas importante de la fase a efectos de trazabilidad, y es tambien el que mas facil era no ver: la cifra estaba en el informe, en el artefacto y en el notebook, y aun asi no se podia reconstruir.

## Semillas y hashes

- Optuna: TPESampler con semilla 20260930, 60 trials, ajustados dentro de outer_train.
- Bootstrap: `default_rng(20260930)`, 400 remuestreos.
- Hash del corpus usado para el canal: `2ec719b78028a043`.
- Fichero de submission de V30: sin cambios, SHA256 `f543a715...`.
