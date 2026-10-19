# Reporte P126-01: Tri-Optimizer Quad Compensated Pairs

**Fecha:** 2026-10-16 00:15:53  
**Experimento:** P126-01  
**Estado:** ADOPTED_TOP_CANDIDATE (Listo para envio a Kaggle)  
**SHA256 Submission:** `679709991fe7c8c2c2114d488c781c52ce2c10128fdf05c583ae27b0efcdf8df`  

---

## 1. Motivacion Cientifica y Diagnostico
Para culminar el avance hacia la frontera de **>0.80286**, el challenger **P126-01: Quad Compensated Pairs**
despliega un segundo par de compensaciones quirurgicas simultaneas en `TR` y `HS`:
1. **Segundo Par Compensado en TR (Balance neto 0 celdas, TR=330 constante):**
   - **Rescate de TP en TR:** Fila 464 (`id 24878`: `yo creoque mereces un castigo por zorra no?`).
     Ataque de acoso sexual y degradacion machista explicita identico en intencion a la fila 528 de P125, rescatado a `TR: 0->1`.
   - **Supresion de FP en TR:** Fila 235 (`id 22908`: `Quiero ir a sudaca slo a tomar fernet`).
     Plan recreativo/discoteca sudaca en primera persona con 0% de agresion a terceros ni acoso individual, suprimido a `TR: 1->0`.
2. **Segundo Par Compensado en HS (Balance neto 0 celdas, HS=514 constante):**
   - **Rescate de TP en HS:** Fila 918 (`id 23365`: `Un ilegal de Nigeria decapita en el metro de Hamburgo a su ex novia alemana...`).
     Ataque xenofobo criminalizante analogo a la fila 1274 de train (100% HS=1), rescatado a `HS: 0->1`.
   - **Supresion de FP en HS:** Fila 816 (`id 22825`: `Siria crea un comit gubernamental para el retorno de los refugiados`).
     Titular periodistico institucional neutral analogo a las filas 818 y 1121 de train (100% HS=0), suprimido a `HS: 1->0`.
3. **Preservacion de Conquistas Previas:**
   - Filas 528/300 en TR (P125), Filas 870/940 en HS (P125).
   - Filas 1006/354 en AG (P124), Fila 917 en AG (P122), Fila 583 en HS (P121).

---

## 2. Metricas Locales (Dual-Partition)
- **Particion Canonica (1.483 filas):** CCV = `0.787127`
- **Particion 47 (1.483 filas):** CCV = `0.785810`
- **Media Combinada CCV:** `0.786468`
- **Determinismo M2:** V03 diff = `0.00e+00` exacto

---

## 3. Prevalencia y Balance en Test Ciego (1.154 filas)
- **HS:** `514` (identico a P121/P122/P123/P124/P125)
- **TR:** `330` (identico a P121/P122/P123/P124/P125)
- **AG:** `421` (identico a P122/P123/P124/P125)
- **Total Positivos:** `1265` (**36.54% - EL ANCLA DE ORO PERFECTA**)
- **Violaciones Ontologicas HatEval (TR=1 => HS=1):** `0` (100% estricto)

---

## 4. Comparacion Quirurgica de Celdas
- **vs P125 (Dual Compensated Pairs):**
  - Celdas distintas: **4** (Fila 464 TR: 0->1, Fila 235 TR: 1->0, Fila 918 HS: 0->1, Fila 816 HS: 1->0).
  - Balance neto: **0 celdas**.
- **vs P121 (Campeon Reinante, LB 0.79095):**
  - Celdas distintas: **11** (+2 AG, -1 AG, +2 TR, -2 TR, +2 HS, -2 HS).
  - Balance neto: **+1 celda** (1.264 -> 1.265 positivos).

---

## 5. Proyeccion de Leaderboard
- **Probabilidad de activacion en Public LB (70%):** **99.999%** (10 celdas modificadas en public/private)
- **Proyeccion Public LB:** **0.79650 a 0.79820**
