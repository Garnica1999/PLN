# V13 · P17b — Fusion de los pronombres con V12

## 1. Pre-chequeo (sin meta-modelo)

El bloque de pronombres por si solo da 0.1922 de macro-F1 (suelo de azar 0.1667) y no toca ninguna de las 279 variables de V12 (0 de 26 redundantes). Aun asi, antes de gastar una evaluacion del combinador se mide cuanto del error de V12 sabe recuperar:

| Medida | Filas |
|---|---|
| Errores de V12 | 67,610 |
| Rescates del bloque de pronombres | 12,043 |
| Danos del bloque de pronombres | 76,118 |
| **Neto** | **-64,075** |
| Umbral de paso | 1,014 |

- 18-24|25-34: rescate 2,199 · dano 25,281 · neto -23,082
- 25-34|35-49: rescate 2,289 · dano 25,627 · neto -23,338

## 2. Decision

El neto -64,075 **no alcanza** el umbral de 1,014 (1.5% de los errores), y ademas ninguna frontera central mejora. Un bloque que solo recupera 12,043 filas de 67,610 no puede desplazar la macro-F1 tres decimales.

**La pista P17 se cierra sin fusionar.** No se ejecuta la etapa 2 y no se consume el resto del presupuesto: el resultado ya es concluyente y el ahorro de ~70 minutos de CPU es real. El lexicon queda documentado y el bloque `features/v17_pronombres.npy` se conserva por si otra combinacion lo necesitara.

