# P102 - Sprint de mejoras (triage de 18 ideas): pre-registro

Fecha: 2026-10-14. Origen: triage de auditoria (18 ideas; 17 YA_REALIZADO con
evidencia; 1 AJUSTE limpia = D4 char sin 'wb'; esquinas menores). Este
pre-registro fija los experimentos del sprint y sus gates ANTES de ejecutar.

## 1. Alcance (orden del ranking del triage)

1. **P102-01 (D4)**: variantes char sin 'wb' sobre el arnes REP (control = V03).
2. **P102-02 (C1-corner)**: promedio V03 + LinearSVC + ComplementNB con
   umbrales recalibrados a la escala del promedio.
3. **P102-03 (B2-corner)**: regla dura TR=0 si HS=0, aislada.
4. **P102-04**: micro-barrido de umbrales alrededor de V03 sobre OOF
   (analisis; sin candidato salvo gate y >=200 celdas).
5. **P102-05 (D1-corner)**: tokens tipograficos (emojis) — condicionado: el
   texto oficial esta ASCII-normalizado por los organizadores; se verifica
   primero y solo se ejecuta si existe material.

## 2. Reglas comunes

- V03 congelado (corrected_cv 0.7772114525165721; umbrales 0.44/0.51/0.46);
  sin test; sin holdouts; sin leaderboard para decidir; sin submissions
  automaticas; sin transformers/LLM; sin reentrenar la referencia.
- Un experimento = un script nuevo; artefactos historicos intactos; ASCII en
  artefactos generados.
- **Gate A3 (unico criterio de aceptacion)**: delta canonico >= +0.003;
  >= 4/5 pliegues; IC95 de bootstrap por clusters que excluya cero; delta
  positivo en la particion 47 (recreada seed 43 + verificacion de hash);
  >= 200 celdas cambiadas vs V03 (presupuesto de deteccion). Todo con los
  umbrales de V03 congelados.
- Parada: si un item falla el gate se cierra y se pasa al siguiente; sin
  rescates por hiperparametro ni barridos de consuelo.

## 3. P102-01 (D4) - diseno exacto

- Arnes: folds canonicos congelados (pliegues de `features/oof_v03_final.npz`;
  verificados contra `src.splits.pliegues_corregidos`); vectorizador y LR
  ajustados DENTRO del pliegue; LR balanced C=2.0 liblinear seed 42 por
  etiqueta; umbrales V03.
- **Control obligatorio**: M2 (word(1,2) + char_wb(3,5), min_df=2, sublinear)
  debe reproducir el OOF congelado (proba de `v2_oof_base_corrected.npz` y
  corrected_cv 0.7772114525165721); si no reproduce, se detiene y se reporta.
- **Variantes**: V1 = word(1,2) + char(1,6) sin wb; V2 = triple union
  (wb + sin wb); V3 = V1 con lowercase=False (case-sensible).
- Evaluacion: corrected_cv, 5 fold scores, delta vs V03, celdas cambiadas,
  IC95 bootstrap por clusters (2000 remuestreos, seed 20260925), delta en la
  particion 47.
- Salidas: `reports/p102_01_char_variantes.json` y `.md`.

## 4. Presupuesto y reglas de riesgo

- CPU solamente; < 1 h por item; sin GPU.
- Sin sondas de leaderboard sin decision explicita del propietario.
- Cualquier hallazgo que sugiera uso de test/holdout se detiene.
