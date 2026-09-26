# V9-04 (H1): secuencial (BiGRU) sobre embeddings FastText propios

| modelo | umbrales V03 | umbral 0.5 |
|---|---|---|
| BiGRU (congelados) | 0.6943 | 0.6829 |
| Media + LR (v5) | 0.6636 | 0.6616 |
| V03 (referencia) | 0.777211 | - |

Veredicto: **SECUENCIAL SUPERA A LA MEDIA; comparar contra V03** (delta secuencial - media +0.0307; vs V03 -0.0829).

Cierra la brecha de la seccion 24 de la auditoria: el mecanismo secuencial se
probo con los vectores propios congelados y early stopping estricto.

Fecha: 2026-09-26. Duracion: 25.59 s.