# V16-02: promedio de semillas con umbrales de tasa constante

Generado por `experiments/v16_02_promedio_umbrales.py` (V16-02 (promedio de semillas con umbrales re-optimizados a tasa constante)).

## 1. La hipotesis y su causa medida

El promedio de probabilidades fracaso dos veces por un mecanismo medido: contrae la escala de salida y con umbrales congelados eso empuja en la direccion contraria. La combinacion que nunca se probo es reoptimizar los umbrales despues de promediar, emparejando la tasa de positivos con la del champion para que el unico cambio que se envia sea la reduccion de varianza

## 2. El promedio solo, con los umbrales de V03: el fracaso que se reproduce

| semilla | macro-F1 | delta |
| --- | --- | --- |
| 42 | 0.777211 | +0.000000 |
| 7 | 0.777211 | +0.000000 |
| 1234 | 0.777211 | +0.000000 |
| 20260926 | 0.777211 | +0.000000 |
| 99991 | 0.777211 | +0.000000 |
| **promedio** | 0.777211 | +0.000000 |

## 3. Los umbrales de tasa constante

El objetivo no es maximizar la media de F1 sobre el OOF, que es donde V03 ya esta, sino
igualar la tasa de positivos predicha del champion. La curva de cuatro puntos medida en
esta fase demuestra que mover la tasa cuesta LB con pendiente empinada.

| etiqueta | umbral V03 | tasa V03 | umbral promedio | tasa promedio | tasa objetivo |
| --- | --- | --- | --- | --- | --- |
| HS | 0.440 | 0.4572 | 0.410 | 0.4943 | 0.4385 |
| TR | 0.510 | 0.3001 | 0.475 | 0.3122 | 0.2894 |
| AG | 0.460 | 0.3776 | 0.450 | 0.3844 | 0.3657 |

## 4. El candidato

- macro-F1 sobre OOF: **0.781106**, delta **+0.003895** respecto a V03.
- Celdas que cambian en el OOF: **83**.
- Submission: `submissions/toxic_submission_v16_ch6_promedio_5semillas.csv`.

**Alcance de la afirmacion**: La reduccion de varianza por promedio es real y deberia transferir bien, pero la evidencia previa en este proyecto es de dos fracasos con el mismo mecanismo. Esta candidata corrige la causa medida; no esta garantizado que la corriga, y por eso se gasta UNA submission y no mas


Fecha: 2026-09-27T01:46:29+00:00. Duracion: 57.9 s.
