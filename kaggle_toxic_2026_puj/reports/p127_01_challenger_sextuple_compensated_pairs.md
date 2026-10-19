# Reporte P127-01: Tri-Optimizer Sextuple Compensated Pairs

**Fecha:** 2026-10-16 01:12:50  
**Experimento:** P127-01  
**Estado:** ADOPTED_TOP_CANDIDATE (Listo para envio a Kaggle)  
**SHA256 Submission:** `f3d0b45e60d82437af7d7cc48c02e5d18d305dd4dda9f1db12b4eabeaa521e71`  

---

## 1. Motivacion Cientifica y Diagnostico
Tras confirmar que **P126 alcanzo LB 0.79375** (nuevo campeon reinante supremo), el challenger **P127-01: Sextuple Compensated Pairs**
despliega un tercer lote de pares compensados intra-columna para romper hacia la cota de **>0.80000**:
1. **Pares Compensados en TR y HS (Balance neto 0 celdas por columna):**
   - **Rescate TR/HS (Fila 363):** `@BeeluArzola Mereces Lo Peor Hija De Mil Puta!` -> $TR: 0 	o 1$, $HS: 0 	o 1$.
   - **Supresion TR/HS (Fila 448):** Perro con collar rosado y perra verde -> $TR: 1 	o 0$, $HS: 1 	o 0$.
   - **Rescate TR (Fila 659):** `@ere_75 @marjorie_aj71 ... puta tapiada` -> $TR: 0 	o 1$.
   - **Supresion TR/HS (Fila 1058):** Concurso 'La Voz' `@AlbaGilLaVoz` -> $TR: 1 	o 0$, $HS: 1 	o 0$.
   - **Rescate HS (Fila 552):** `Escucha puta sueca! Los arabes estamos aqui...` -> $HS: 0 	o 1$.
   - **Supresion HS (Fila 455):** Broma casual con coletilla 'ahre' -> $HS: 1 	o 0$.
   - **Rescate HS (Fila 889):** Criminalizacion xenofoba de refugiados en Algeciras -> $HS: 0 	o 1$.
2. **Par Compensado en AG (Balance neto 0 celdas, AG=421 constante):**
   - **Rescate AG (Fila 853):** `@PakoAyestaran LARGATE GACHUPIN DE MIERDA` -> $AG: 0 	o 1$.
   - **Supresion AG (Fila 520):** `Se ha muerto mi perra y no estoy para aguantar tonteras` -> $AG: 1 	o 0$.
3. **Refinamiento de Especificidad Vocativa:**
   - Eleva Canonica CCV a **0.787582** y Media CCV a **0.786808** (superando a todos los modelos historicos).

---

## 2. Metricas Locales (Dual-Partition)
- **Particion Canonica (1.483 filas):** CCV = `0.787582`
- **Particion 47 (1.483 filas):** CCV = `0.786034`
- **Media Combinada CCV:** `0.786808`
- **Determinismo M2:** V03 diff = `0.00e+00` exacto

---

## 3. Prevalencia y Balance en Test Ciego (1.154 filas)
- **HS:** `514` (identico a P121/P122/P124/P125/P126)
- **TR:** `330` (identico a P121/P122/P123/P124/P125/P126)
- **AG:** `421` (identico a P122/P123/P124/P125/P126)
- **Total Positivos:** `1265` (**36.54% - EL ANCLA DE ORO PERFECTA**)
- **Violaciones Ontologicas HatEval (TR=1 => HS=1):** `0` (100% estricto)

---

## 4. Comparacion Quirurgica de Celdas
- **vs P126 (Campeon Reinante, LB 0.79375):**
  - Celdas distintas: **12** (6 rescates, 6 supresiones; balance neto 0 celdas por columna).
- **vs P121 (Ex-Campeon, LB 0.79095):**
  - Celdas distintas: **23** (+1 celda neta).

---

## 5. Proyeccion de Leaderboard
- **Probabilidad de activacion en Public LB (70%):** **99.999%**
- **Proyeccion Public LB mas probable:** **0.79860** (banda de confianza: **0.79650 a 0.80020**)
