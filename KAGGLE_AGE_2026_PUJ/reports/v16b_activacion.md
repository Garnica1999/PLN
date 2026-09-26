# V13 · P16b — ¿Dónde es útil el bloque de evidencia por token?

## 1. La paradoja que hay que explicar

El bloque de P1 tiene señal propia (0,3997 en control aislado), su información **no** está en V12 (P16: R2 mediano 0,09 y el control sobre la reconstrucción cae al suelo de azar) y, aun así, fusionarlo da -0,0002, igual que un bloque permutado. La explicación más económica es que la señal esté concentrada en una minoría de filas y el promedio la diluya. Eso es lo que se mide aquí, sin entrenar nada.

## 2. Actividad del bloque

- Tokens conocidos de la banda específica por documento: media 1.65
- Documentos con al menos uno: **122,870** (72.6 % del total)

## 3. Exactitud por franja de actividad

Umbral = número mínimo de tokens conocidos de la banda específica para considerar el documento activo. **Techo de puerta** es el máximo alcanzable con una puerta perfecta (V12 si acierta, si no el bloque).

| Umbral | Ámbito | Frontera | Filas | V12 | Bloque | Techo de puerta | Rescate | Daño | Neto |
|---|---|---|---|---|---|---|---|---|---|
| 0 | todas | global | 169,336 | 0.6007 | 0.4028 | **0.677** | 12,910 | 46,425 | -33,515 |
| 0 | activas | global | 169,336 | 0.6007 | 0.4028 | **0.677** | 12,910 | 46,425 | -33,515 |
| 1 | todas | global | 169,336 | 0.6007 | 0.4028 | **0.677** | 12,910 | 46,425 | -33,515 |
| 1 | activas | global | 122,870 | 0.6216 | 0.4391 | **0.6924** | 8,697 | 31,130 | -22,433 |
| 1 | inactivas | global | 46,466 | 0.5455 | 0.307 | **0.6361** | 4,213 | 15,295 | -11,082 |
| 2 | todas | global | 169,336 | 0.6007 | 0.4028 | **0.677** | 12,910 | 46,425 | -33,515 |
| 2 | activas | global | 77,872 | 0.633 | 0.4647 | **0.6998** | 5,200 | 18,302 | -13,102 |
| 2 | inactivas | global | 91,464 | 0.5733 | 0.3501 | **0.6576** | 7,710 | 28,123 | -20,413 |
| 3 | todas | global | 169,336 | 0.6007 | 0.4028 | **0.677** | 12,910 | 46,425 | -33,515 |
| 3 | activas | global | 43,620 | 0.642 | 0.4867 | **0.7063** | 2,806 | 9,581 | -6,775 |
| 3 | inactivas | global | 125,716 | 0.5864 | 0.3737 | **0.6668** | 10,104 | 36,844 | -26,740 |
| 5 | todas | global | 169,336 | 0.6007 | 0.4028 | **0.677** | 12,910 | 46,425 | -33,515 |
| 5 | activas | global | 8,856 | 0.6551 | 0.5194 | **0.7221** | 593 | 1,795 | -1,202 |
| 5 | inactivas | global | 160,480 | 0.5977 | 0.3964 | **0.6745** | 12,317 | 44,630 | -32,313 |

## 4. Las dos fronteras centrales (§36)

| Umbral | Frontera | Filas | V12 | Bloque | Techo de puerta | Neto |
|---|---|---|---|---|---|---|
| 0 | 18-24|25-34 | 57,023 | 0.4963 | 0.3059 | 0.5898 | -10,858 |
| 0 | 25-34|35-49 | 56,577 | 0.4974 | 0.278 | 0.5681 | -12,410 |
| 1 | 18-24|25-34 | 38,283 | 0.4973 | 0.3183 | 0.5851 | -6,856 |
| 1 | 25-34|35-49 | 41,255 | 0.5205 | 0.3147 | 0.5939 | -8,491 |
| 2 | 18-24|25-34 | 21,975 | 0.4804 | 0.3057 | 0.5574 | -3,839 |
| 2 | 25-34|35-49 | 26,008 | 0.5265 | 0.3238 | 0.5941 | -5,271 |
| 3 | 18-24|25-34 | 11,094 | 0.4588 | 0.2932 | 0.5263 | -1,837 |
| 3 | 25-34|35-49 | 14,235 | 0.5267 | 0.3279 | 0.5874 | -2,830 |
| 5 | 18-24|25-34 | 1,951 | 0.4449 | 0.2783 | 0.5095 | -325 |
| 5 | 25-34|35-49 | 2,693 | 0.5158 | 0.319 | 0.5774 | -530 |

## 5. Lectura

- En las 169,336 filas activas, el bloque acierta 0.4028 frente a 0.6007 de V12, y una puerta perfecta llegaria a 0.677.
- El techo de puerta sobre las filas activas es la cota de §37: si esa cifra no supera al 0,6004 de V12 por un margen amplio, **ninguna puerta puede adopter el bloque** y la pista P1 se cierra de forma definitiva, no solo por no cumplir el gate de la fusion.
- Con una puerta imperfecta el techo es inalcanzable: la regla de §37 solo activa el especialista cuando V12 es ambiguo, y P15 ya mostro que la confianza de los miembros no identifica las filas que rescuarian. El margen entre el techo y una puerta real es precisamente lo que hay que mirar antes de gastar una hora de CPU en P3.

