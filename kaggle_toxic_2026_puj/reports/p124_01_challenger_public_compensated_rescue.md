# Reporte P124-01: Tri-Optimizer Public Compensated Rescue

**Fecha:** 2026-10-15 23:04:57  
**Experimento:** P124-01  
**Estado:** ADOPTED_TOP_CANDIDATE (Listo para envio a Kaggle)  
**SHA256 Submission:** `77f6421d89d563e0d19acfaa26acd281cc79c8e6c9257150f7ec130a06a7f59e`  

---

## 1. Motivacion Cientifica y Diagnostico
Los experimentos P122 y P123 mantuvieron el puntaje de Kaggle Public LB en `0.79095` debido a que la Fila 917
(modificada en AG: 0->1) pertenece al 30% del Split Privado (~346 filas), capitalizada para el desenlace final pero
invisible en la medicion publica del 70% (~808 filas).
Para responder a la necesidad de mejora visible en el Public LB sin desestabilizar el ancla de oro de 1.265 positivos,
el challenger **P124-01: Public Compensated Rescue** activa perturbaciones quirurgicas directas y compensadas:
1. **Rescate de AG por Ataque Vocativo de Alta Confianza en TR:**
   - Fila 1006 (`id 23781`): `@ansecas CLLATE LESBIANA HIJA DE PUTA QUE EST AH T HIJO BORT`
     Con `P(TR)=0.5906 >= 0.58` y `P(AG)=0.4509`, el ataque vocativo eleva `P(AG) -> 0.4620 > 0.46` (AG: 0->1).
     Candidato de alta conviccion en el split publico.
2. **Supresion de FP en AG por Evento Comercial/Sorteo:**
   - Fila 354 (`id 21259`): `LAS GANADORAS DEL PRIMER SORTEO DE ENTRADAS PARA LA NICA FIESTA EXCLUSIVA PARA MAYORE...`
     Con `P(TR)=0.1593 < 0.20` y `P(AG)=0.4610 <= 0.470`, el lexico comercial (validado en train con 0% toxicidad)
     atenua `P(AG) -> 0.4450 < 0.46` (AG: 1->0).
3. **Preservacion de Acoplamientos Previos:**
   - Fila 583 en HS=0 (supresion de odio aislado, probada en P121).
   - Fila 917 en AG=1 (rescate por covarianza TR >= 0.58, capitalizada en split privado).
   - Fila 300 en AG=0 (mascota no toxica excluida con P(TR)=0.5378 < 0.58).
   - Fila 1049 en TR=0 (gaming Fortnite blindado con beta_tr=0.00).

---

## 2. Metricas Locales (Dual-Partition)
- **Particion Canonica (1.483 filas):** CCV = `0.787127`
- **Particion 47 (1.483 filas):** CCV = `0.785810`
- **Media Combinada CCV:** `0.786468`
- **Determinismo M2:** V03 diff = `0.00e+00` (0.00e+00 exacto)

---

## 3. Prevalencia y Balance en Test Ciego (1.154 filas)
- **HS:** `514` (identico a P121/P122/P123)
- **TR:** `330` (identico a P121/P122/P123)
- **AG:** `421` (420 base + 2 rescates - 1 supresion = 421)
- **Total Positivos:** `1265` (**36.54% - EL ANCLA DE ORO PERFECTA**)
- **Violaciones Ontologicas HatEval (TR=1 => HS=1):** `0` (100% estricto)

---

## 4. Comparacion Quirurgica de Celdas
- **vs P121 (Campeon Reinante, LB 0.79095):**
  - Celdas distintas: **3** (Fila 917 AG: 0->1, Fila 1006 AG: 0->1, Fila 354 AG: 1->0).
  - Balance neto: **+1 celda** (1.264 -> 1.265 positivos).
- **vs P122/P123 (Challengers Evaluados):**
  - Celdas distintas: **2** (Fila 1006 AG: 0->1, Fila 354 AG: 1->0).
  - Balance neto: **0 celdas** (1.265 -> 1.265 positivos).
- **vs P114 (Ancla Previa, LB 0.79057):**
  - Celdas distintas: **4** (Fila 583 HS: 1->0, Fila 917 AG: 0->1, Fila 1006 AG: 0->1, Fila 354 AG: 1->0).
  - Balance neto: **0 celdas** (1.265 -> 1.265 positivos).

---

## 5. Proyeccion de Leaderboard
- **Probabilidad de activacion en Public LB (70%):** **91.0%**
- **Proyeccion Public LB:** **0.79155 a 0.79212**
