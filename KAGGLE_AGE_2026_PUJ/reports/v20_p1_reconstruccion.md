# V20 · P1 — ORDINAL_RECONSTRUCTION: cuatro formas de reconstruir las seis clases

**V20 queda congelado.** No se toca un solo clasificador: solo la capa de reconstruccion.

## 1. Lo que la auditoria P0 dejo establecido, y por que aqui hay poco margen

- Las cinco CDFs estan **esencialmente perfectamente calibradas**: descalibracion entre **-0,0003 y +0,0002**. **No hay nada que ganar recalibrando.**
- Solo el **11,18 %** de las filas violan la monotonia, con magnitud media **0,002457** y maxima **0,518606**. **La proyeccion monotona tiene poco que corregir.**
- **Consecuencia**: el margen de esta subfamilia es pequeno, y eso es una hipotesis medible, no una corazonada. Si las cuatro variantes salen planas, el hallazgo es que la consistencia ordinal ya esta saturada por la forma en que V20 la resolvio.

## 2. Las cuatro variantes

| Variante | Mecanismo | Por que es distinta |
|---|---|---|
| **A** | la combinacion actual de V20 | **Control exacto**: debe reproducir 0,6097782 |
| **B** | proyeccion monotona por **PAVA** | el maximo acumulado empuja el valor posterior hacia arriba; PAVA reparte la violacion promediando el bloque, que es la proyeccion L2 sobre el cono |
| **C** | **conteo de umbrales cruzados** | no usa las probabilidades reconstruidas: decide por la estructura acumulada |
| **D** | **edad esperada** | introduce la distancia ordinal en la decision, que el argmax no tiene |

## 3. Resultados contra V20

| Variante | macro-F1 | Delta contra V20 | Folds pos. | IC95 | Frontera 1 | Frontera 2 | Rescates | Danos | Neto | Cambios |
|---|---|---|---|---|---|---|---|---|---|---|
| **V20 (champion)** | 0.6097782 | - | - | - | 0,5075 | 0,5029 | - | - | - | - | CHAMPION |
| A control exacto: maximo acumulado | 0.6097782 | +0.0000 | 0/5 | [+0.0000, +0.0000] | 0.5075 | 0.5029 | 0 | 0 | +0 | 0 |
| B proyeccion monotona por PAVA | 0.6097927 | +0.0000 | 3/5 | [-0.0000, +0.0001] | 0.5074 | 0.5028 | 16 | 11 | +5 | 41 |
| C decision por conteo de umbrales | 0.5983871 | -0.0114 | 0/5 | [-0.0130, -0.0098] | 0.545 | 0.5456 | 8,361 | 11,680 | -3,319 | 33,182 |
| D decision por edad esperada | 0.5834541 | -0.0263 | 0/5 | [-0.0282, -0.0244] | 0.541 | 0.5738 | 11,295 | 17,709 | -6,414 | 47,138 |

**El control A reproduce V20 exacto: True** (0.6097782 frente a 0.6097782).

## 4. Delta por fold

| Variante | Delta por fold |
|---|---|
| A control exacto: maximo acumulado | [0.0, 0.0, 0.0, 0.0, 0.0] |
| B proyeccion monotona por PAVA | [-4.2e-05, 7.3e-05, -4e-05, 3.5e-05, 4.8e-05] |
| C decision por conteo de umbrales | [-0.011546, -0.011731, -0.009882, -0.010998, -0.012812] |
| D decision por edad esperada | [-0.023341, -0.028723, -0.025632, -0.025422, -0.028597] |
