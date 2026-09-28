# V30 - Analisis de transferencia

## El estado de la sonda

- Estado de la subida: **STOP_SIN_CREDENCIAL**
- Numero de submissions: **0**
- Subido: **False**
- Caso del punto 32: **SIN_TRANSFERENCIA_MEDIDA**

## La comparacion que el punto 31 pide

| Magnitud | Valor |
|---|---|
| corrected_cv interno del candidato | 0,6130051 |
| delta interno frente a V20 | 0,003227 |
| folds positivos internos | 5/5 |
| IC95 interno | [0.00229, 0.004219] |
| LB historico de V20 | 0.62689 |
| LB del candidato | **no disponible**: No se midio: **la submission no llego a Kaggle**, por falta de credencial valida.

## Por que no se infiere nada de la ausencia

La ausencia de score **no es informacion sobre la transferencia**. Es informacion sobre la infraestructura. Confundir las dos cosas seria el error clasico de este tipo de fase: inventar una expectativa a partir de un dato que no existe.

Y la regla del punto 42 se respeta por construccion: **no hay una segunda submission**, así que no existe la possibility de un bucle de retroalimentacion con el leaderboard. La fase queda cerrada con la evidencia interna que hay, y la externa pendiente de una credencial que es del usuario.

## Que pasaria si la credencial estuviera disponible

El fichero `submissions/age_submission_v30_final_candidate.csv` esta construido, con sus diez checks en verde y el hash `f543a715a1881c6128a860301b84a7273c3db8f8a709545be1e6267822e713be`. `v30_04_upload.py` lo sube **tal cual, una vez**, con el nombre de sonda `AGE_V30_FINAL_CANDIDATE`. No habria que cambiar ni un parametro: el candidato ya esta congelado.
