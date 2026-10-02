# Revision adversarial P82 (RNN con atencion) y su sonda

Revisor: ml-reviewer (sesión del 2026-10-02). Alcance: `experiments/p82_01_rnn_atencion.py`,
`experiments/p82_02_submission_rnn.py`, informe vigente
`reports/p82_rnn_atencion_20261001T194000Z_p2731658.json/.md`, `reports/p82_submission_rnn.json/.md`,
pre-registro y enmiendas, `submissions/submission_p82_sonda_rnn_b1.csv` y la fila del
`leaderboard_log.csv`. No se modifico ningun archivo existente; el artefacto congelado del
Word2Vec quedo verificado intacto.

## Veredictos

1. "23 de 23 CONFIRMADAS": EXACTO en el conteo (23 CONFIRMADO, 0 REFUTADO o NO_EVALUABLE),
   pero con matiz material grave en A7 (ver H1). El conteo no basta: A7 se declara cumplida
   con una medicion que no puede fallar.
2. "Determinismo bit a bit entre dos corridas independientes": REFUTADO. A7
   (`p82_01:2966-2994`, invocada en 6097 y 6830) compara dos pasadas del mismo brazo B2
   dentro de la MISMA corrida, y `variacion_maxima_probabilidades` es NaN; la igualdad se
   declara con `np.array_equal(..., equal_nan=True)` sobre matrices de mascaras. La
   continuidad entre corridas es un contraste contra un diccionario hardcodeado
   (`p82_01:2164-2196`), no una comprobacion automatica.
3. Brazos y comparaciones: CONFIRMADOS exactos. B1 0.639659, B2 0.637554, B3 0.615163,
   B4 0.617194, B5 0.560747; P1 (B2-B1) -0.002105 IC95 [-0.012406, 0.008196] con 4/10
   positivos; P2 (B4-B3) +0.002031 con 7/10; MDE 0.014342. Los signos opuestos de P1 y P2
   están publicados correctamente.
4. Dispositivo: CONFIRMADO. `resolver_dispositivo` (151-157), `DISPOSITIVO` (174) y
   `.to(DISPOSITIVO)` (1811, 1839, 2576-2578); `HILOS=16` (131) con
   `torch.set_num_threads(HILOS)` (7191). El 1.24 s (GPU ROCm) contra 2.43 s (CPU 16
   hilos) coincide con la enmienda 2 y con el JSON; el "x2" es 1.96, sostenido como
   benchmark de un ajuste.
5. Sonda: sha256 verificado con `sha256sum` (3315b252...d02322f, OK); CSV con 1154 filas de
   datos, ids unicos, cabecera y valores binarios. Umbrales congelados sin recalibrar
   (`p82_02:81`, `MOTIVO_UMBRALES`); holdout excluido por guardas (`p82_02:555-571`, 0
   usadas). La igualdad de ids contra el test queda NO_EVALUABLE de forma independiente
   (el test no es UTF-8 ni ASCII y el presupuesto corto la verificacion binaria); el script
   interno declara `true` en su verificacion 1b.
6. "LB 0.68150": registrado y explicito en la fila `p82_sonda_rnn_b1` del leaderboard
   (kaggle_f1 0.68150, local 0.639659, delta +0.041841 con su interpretacion de anomalia).
   Manda el registro. `reports/p82_submission_rnn.md:39` ("No se ha consultado el
   leaderboard ni se ha enviado nada a Kaggle") quedo desactualizado. Severidad ALTA
   documental; no invalida el modelo.
7. Escaner: D1 presente (15 funciones en p82_01 y 12 en p82_02, solo-humo, MEDIA), D6 por
   menciones (MEDIA) y D5 umbral hardcodeado (`p82_02:81`, BAJA). D3 y D4 no aparecen en
   P82. Ninguno invalida conclusiones publicadas; D1 es deuda de testabilidad.

## Hallazgos nuevos

- H1 (MEDIA-ALTA, metodológico): `p82_01:2979-2987` y la afirmacion A7 declaran variacion
  cero de probabilidades sin medirla (NaN mas `equal_nan=True` sobre mascaras). Contamina
  la afirmacion 2 y debilita A7.
- H2 (MEDIA, documental): `p82_01:2164-2196`: la comparación obligatoria entre corridas es
  contra un diccionario hardcodeado, no automatica.
- H3 (MEDIA, reproducibilidad): `p82_02:591-594`: el submission recalcula E* con cinco
  pliegues internos sobre los 1483 y no reutiliza los pliegues externos de la CV, de modo
  que el modelo final no replica exactamente el pipeline de B1. Debe declararse.
- H4 (ALTA, documental): contradiccion entre el LB 0.68150 registrado y el MD que niega el
  envio.

## Pendiente

- Verificacion independiente de igualdad de ids test contra submission (NO_EVALUABLE por
  encoding y presupuesto).
- Volcar las correcciones H1, H3 y H4 al notebook y al registro antes de dar P82 por
  cerrado documentalmente.
