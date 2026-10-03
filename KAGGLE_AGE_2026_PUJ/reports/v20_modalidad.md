# V13 · P20 — Incertidumbre y modalidad

## 1. Que se construye

Marcadores de incertidumbre y modalidad con vocabulario cerrado y propio (R-1), en seis categorías: duda, certeza, modalidad, negación, grado e imposibilidad. Se excluyen a propósito los marcadores de risa: no son incertidumbre ni modalidad, y reabrirían la línea de lexicones que D-039 cerró.

Este bloque es de un tipo de variable que P0b declaró con margen: funciones y conectores del discurso, no medidas de longitud ni de variedad. P17 dio 0 de 26 redundantes para pronombres y aquí se espera lo mismo.

## 2. Cobertura por categoría

| Categoría | Documentos | Apariciones | Formas sobre el mínimo |
|---|---|---|---|
| duda | 2,775 | 2,853 | 4/13 |
| certeza | 734 | 742 | 1/10 |
| modalidad | 4,689 | 4,877 | 7/13 |
| negacion | 27,431 | 34,058 | 7/10 |
| grado | 16,866 | 18,946 | 10/12 |

Umbral de permanencia de una forma: 200 documentos.

## 3. Características

23 columnas: conteo, ratio por cien tokens y presencia para cada categoría, más ocho derivadas. La más informativa es **índice de duda relativo** (duda entre duda más certeza), que separa a quien duda de quien afirma.

## 4. Cribado de redundancia contra las 279 variables de V12

- Variables redundantes: **0** de 23
- Correlación máxima: **0.4067** · mediana 0.1455

## 5. Control block-only

| Modelo | macro-F1 | std | Suelo de azar | Supera el azar |
|---|---|---|---|---|
| logreg | 0.1270 | 0.0013 | 0.1667 | no |
| linearsvc | 0.1259 | 0.0012 | 0.1667 | no |

## 6. Puerta de rescate (§26)

Antes de pagar la evaluación del combinador se mide cuánto del error de V12 recupera el bloque y cuánto daña. Umbral: neto de al menos 1,014 filas y positivo en las dos fronteras centrales.

| Medida | Filas |
|---|---|
| Errores de V12 | 67,610 |
| Rescates | 10,390 |
| Daños | 80,413 |
| **Neto** | **-70,023** |

- 18-24|25-34: rescate 1,510 · daño 25,774 · neto -24,264
- 25-34|35-49: rescate 266 · daño 27,834 · neto -27,568

## 7. Decisión

**La puerta no se supera**: el bloque no justifica pagar la evaluación del combinador. La pista P20 se cierra sin fusionar, con la evidencia del pre-chequeo, igual que P17.

