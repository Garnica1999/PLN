# Revision adversarial P69 a P77

Revisor: ml-reviewer (sesión del 2026-10-02). Alcance: experiments/p69_*.py a
p77_*.py, reports/p69_* a p77_*, filas P69 a P77 de experiments.csv.

## Veredictos

1. P69: CONFIRMADA CON SALVEDADES. CAND=0.7854160387 re-derivado; diferencia
   contra el histórico -3.2392e-11 (coincide con el 3.2e-11 declarado); delta
   +0.0082045862 exacto. PERO la afirmacion "R3 re-ejecutado" sobreafirma:
   `p69_01_auditoria_downweight.py:434-481` no re-entrena, carga
   `features/oof_weighting_v4.npz`. La reproducibilidad es de scoring,
   umbrales y splits, no del entrenamiento. El criterio 6
   (A_falta_de_artefacto) esta verificado en
   `reports/p58_final_candidate_audit.json` y no invalida la adopcion
   experimental, pero impide FINALIST.
2. P71: CONFIRMADA. El bug existe y esta bien citado: `p71_04:65` LOTE=8192
   con Adam lr=0.025 y perdida .mean() (gradiente dividido por 8192);
   corregido en `p71_08:100-101` con lr_ini=6.4. La conclusion MARGINAL no
   sobreafirma.
3. P72: CONFIRMADA CON ERROR DE ESTADISTICO. Numeros exactos: A=0.6037188662,
   media de nulas=0.5932546209, sd=0.0122425664, maximo=0.6179044898. El
   delta publicado -0.0141856236 es A menos el MAXIMO, no A menos la media.
   A menos la media es +0.0104642453 y solo 1 de 10 nulas supera a A. El
   pre-registro define la regla contra el maximo, luego el veredicto
   SENAL_DEBIL es procedimentalmente correcto, pero la lectura publica
   "no supera a las nulas" es enganosa. P72-04: el defecto del
   StandardScaler existio (presente en `p72_02:199`, ausente en `p72_03`,
   añadido en `p72_04:60-64`) y no cambia el veredicto.
4. P73: CONFIRMADA CON CALIFICACION. A=0.6090700099, B=0.5774440627,
   delta=+0.0316259472, 4/5 pliegues. Segunda particion: 20261103
   delta=+0.0344649141 IC [+0.014094,+0.055680]; 20261104 delta=+0.0112854167
   IC [-0.009276,+0.031672]. El signo se sostiene; la magnitud no. Las tres
   particiones reutilizan el mismo par de matrices y no miden la varianza del
   sorteo.
5. P74: CONFIRMADA. Pareo real en `p74_01:493-496` con semilla comun y
   `torch.Generator().manual_seed(semilla+7)`; verificacion instrumentada en
   `verificar_pareo:274-335`. IC95 [+0.0139795,+0.0345616], A gana 10/10.
6. P75: CONFIRMADA. rescue=268, damage=813, net=-545; tres alphas medidos
   (0.05 -> -0.0026005064; 0.10 -> -0.0052438938; 0.15 -> -0.0080173237);
   gate NO PASA; veredicto CASO 5 CNN_PERJUDICA.
7. P76: CONFIRMADA. Ocho configuraciones verificadas; mejor
   C_V03raw_CNNcal_alpha0.05 delta=-0.0017743921 (2/5). La calibración mejora
   Brier y pendientes pero no transfiere; calibrar V03 lo empeora 0.0086
   (0.7772 -> 0.7686): su sobreconfianza es portante.
8. P77: CONFIRMADA. AUC del gate 0.61-0.74; tres operadores monotonos
   (154 -> -0.0192385535; 393 -> -0.0512720253; 543 -> -0.0556959478), 0/5
   pliegues, IC excluyendo cero; TR dana mas; segunda particion NO ejecutada.

## Hallazgos nuevos

- CRITICO (metodológico): `reports/twitter_word2vec_control_aleatorio.json` y
  `experiments/p72_03_control_aleatorio.py:337` publican
  `delta_contra_el_maximo` como un delta generico; la fila P72-03 de
  `experiments.csv:403` lo presenta sin aclarar que es contra el maximo. Debe
  corregirse la etiqueta en el registro y en el notebook.
- IMPORTANTE: `p69_01_auditoria_downweight.py:434-481` carga el OOF
  histórico; la etiqueta correcta es "re-scoring trazable", no
  "reproduccion R3 del entrenamiento".
- IMPORTANTE: `p69_01:539-543`: TOLERANCIA y coincide_folds se calculan pero
  el nivel R3 no los exige.
- SUGERENCIA: la adenda de P73 corrige el rango de la nula pero no mide la
  varianza del sorteo de matrices.

## Piezas no ejecutadas de P69 a P77

- P77: segunda particion NO ejecutada; sin finalista.
- P69-10: submission generada y auditada, NO subida.
- P74-PRE: preinscripcion; luego ejecutada como P74.
- V03, holdout, test y leaderboard: 0 usos declarados en los JSON de P69-P77.

## Pendiente de completar (declarado por el revisor)

- Transversal A: pre-registros de P73/P74/P77 contra constantes del código
  (EPOCAS, PACIENCIA, TOL_ES, BATCH) línea a línea.
- Transversal B: early stopping sobre el pliegue evaluado en P75/P76/P77.
- Transversal C: auditoría AST completa de fugas test/holdout en los 11
  scripts.
- Transversal D: cruce con los hallazgos D1/D3/D4 del escaner (lo completa el
  lider en la bitácora).
