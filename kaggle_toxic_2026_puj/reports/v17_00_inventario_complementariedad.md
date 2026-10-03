# V17-00: inventario del patrimonio y complementariedad de errores frente a V03

Generado por `experiments/v17_00_inventario_y_complementariedad.py` (V17-00 (P0 inventario y P1 complementariedad de errores frente a V03)).

**No se entrena nada, no se regenera nada y no se gasta ninguna submission.** Esto es un
recuento sobre artefactos que ya existen, y decide si un ensemble merece construirse.

## 1. La rejilla

- **1483 filas x 3 etiquetas = 4449 celdas.**
- Errores de V03: **757**, el 17.02 por ciento de las celdas.
- La rejilla es de celdas y no de filas: la metrica es la media de las tres F1, y
  corregir HS no cambia el acierto de TR ni de AG.

## 2. La primera pregunta, con numeros

Cuantos errores de V03 rescata cada modelo historico y cuantos danos introduce. Orden
por rescate neto en el umbral optimo, que es la cifra que decide.

| artefacto | variante | macro F1 optima | rescates optimos | danios optimos | rescate neto optimo | cobertura | correlacion de errores |
| --- | --- | --- | --- | --- | --- | --- | --- |
| oof_candidates_v4 | C_meta36_downweight | 0.7900 | 137 | 65 | +72 | 19.4 por ciento | 0.806 |
| oof_ens04_v04 | proba_ens | 0.7911 | 173 | 114 | +59 | 22.3 por ciento | 0.766 |
| oof_meta_shift_v4 | meta_control | 0.7826 | 133 | 77 | +56 | 20.9 por ciento | 0.765 |
| oof_mod03b | proba | 0.7879 | 162 | 107 | +55 | 21.1 por ciento | 0.781 |
| oof_candidates_v4 | B_meta36 | 0.7833 | 126 | 73 | +53 | 19.0 por ciento | 0.800 |
| oof_meta_shift_v4 | meta_shift | 0.7833 | 126 | 73 | +53 | 19.0 por ciento | 0.800 |
| oof_meta_shift_v4 | meta_shift_cuantil | 0.7832 | 124 | 72 | +52 | 19.0 por ciento | 0.797 |
| oof_meta_shift_v4 | meta_shift_platt | 0.7756 | 145 | 99 | +46 | 16.2 por ciento | 0.748 |
| oof_candidates_v4 | D_svd300_rbf | 0.7886 | 134 | 90 | +44 | 18.1 por ciento | 0.803 |
| oof_tracks_g_v4 | svd__rbf_svc | 0.7886 | 134 | 90 | +44 | 18.1 por ciento | 0.803 |
| oof_candidates_v4 | A_cost_sensitive_downwei | 0.7862 | 104 | 63 | +41 | 13.1 por ciento | 0.847 |
| oof_joint_label_v4 | joint__lr4_excluir_bal_C | 0.7755 | 165 | 124 | +41 | 16.8 por ciento | 0.784 |
| oof_weighting_v4 | combo__downweight_violac | 0.7862 | 104 | 63 | +41 | 13.1 por ciento | 0.847 |
| v2_meta_oof | proba | 0.7789 | 137 | 101 | +36 | 17.4 por ciento | 0.802 |
| oof_joint_label_v4 | joint__lr6_bal_C2.0 | 0.7765 | 155 | 120 | +35 | 18.1 por ciento | 0.761 |
| oof_joint_label_v4 | joint__lr6_unb_C2.0 | 0.7725 | 155 | 122 | +33 | 25.8 por ciento | 0.700 |
| oof_reweight_v4 | svc__down_violacion | 0.7813 | 96 | 63 | +33 | 12.9 por ciento | 0.820 |
| oof_joint_label_v4 | joint__lr4_mapeo_bal_C2. | 0.7702 | 190 | 158 | +32 | 17.6 por ciento | 0.770 |

## 3. La segunda pregunta: hay alguna combinacion con rescate neto suficiente?

**14 variantes superan a V03 en OOF con su umbral optimo.** Esas son las
unicas que podrian entrar en un ensemble. Todas pertenecen a artefactos que ya se
habian enviado, de modo que la pregunta no es si el OOF mejora, sino si esa mejora
transferiria, que es justamente lo que V16 demostro que no hace.

| artefacto | variante | OOF optimo | delta | umbrales optimos |
| --- | --- | --- | --- | --- |
| oof_ens04_v04 | proba_ens | 0.791075 | +0.013864 | 0.47/0.49/0.52 |
| oof_candidates_v4 | C_meta36_downweight | 0.790001 | +0.012790 | 0.47/0.44/0.50 |
| oof_candidates_v4 | D_svd300_rbf | 0.788564 | +0.011352 | 0.46/0.51/0.49 |
| oof_tracks_g_v4 | svd__rbf_svc | 0.788564 | +0.011352 | 0.46/0.51/0.49 |
| oof_mod03b | proba | 0.787876 | +0.010664 | 0.48/0.51/0.50 |
| oof_candidates_v4 | A_cost_sensitive_downwei | 0.786199 | +0.008987 | 0.44/0.61/0.47 |
| oof_weighting_v4 | combo__downweight_violac | 0.786199 | +0.008987 | 0.44/0.61/0.47 |
| oof_candidates_v4 | B_meta36 | 0.783306 | +0.006094 | 0.46/0.50/0.47 |
| oof_meta_shift_v4 | meta_shift | 0.783306 | +0.006094 | 0.46/0.50/0.47 |
| oof_meta_shift_v4 | meta_shift_cuantil | 0.783234 | +0.006023 | 0.46/0.48/0.47 |

## 5. Lo que este script NO hace

- No entrena ningun modelo, como pide la instruccion de la fase.
- No construye ningun ensemble: la condicion para construirlo es que la
  complementibilidad se sostenga, y eso se evalua aqui.
- No gasta ninguna submission. Las cinco restantes se conservan para un candidato con
  hipotesis concreta.


Fecha: 2026-09-27T02:20:29+00:00. Duracion: 0.4 s.
