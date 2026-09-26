# V17 · P8 — Perfil de confianza de los errores de frontera

Inerte: ninguna señal se ajusta aqui. La separabilidad se mide con el area bajo la curva, que es una medida de orden y no un modelo, asi que no puede sobreajustar.

## Que se compara

En cada frontera, las filas que el especialista de pares **rescata** (V12 falla y el especialista acierta) frente a las que **rompe** (V12 acierta y el especialista falla). Si una senal de confianza sirviera para enrutar, tendria que dar valores altos a las rescatables y bajos a las danables. El area bajo la curva mide exactamente eso, y **0,50 significa que la senal no dice nada**.

## Resultados

| Frontera | Señal | AUC rescate contra daño | Lectura |
|---|---|---|---|
| 18-24\|25-34 | Separacion_V12 | 0.578 | debil pero real |
| 18-24\|25-34 | Entropia_V12 | 0.4323 | inutil |
| 18-24\|25-34 | Masa_dos_primeras | 0.5746 | debil pero real |
| 18-24\|25-34 | Acuerdo_miembros | 0.707 | aprovechable |
| 18-24\|25-34 | Margen_especialista | 0.7294 | aprovechable |
| 18-24\|25-34 | Desacuerdo | 0.5 | inutil |
| 25-34\|35-49 | Separacion_V12 | 0.5268 | inutil |
| 25-34\|35-49 | Entropia_V12 | 0.423 | inutil |
| 25-34\|35-49 | Masa_dos_primeras | 0.5673 | debil pero real |
| 25-34\|35-49 | Acuerdo_miembros | 0.73 | aprovechable |
| 25-34\|35-49 | Margen_especialista | 0.7539 | aprovechable |
| 25-34\|35-49 | Desacuerdo | 0.5 | inutil |

## Lectura

**4 señales superan el area de 0.62**: quedan vias abiertas para P9 (experto selectivo) y P3 (compuerta blanda). El nombre de la mejor es `Margen_especialista` en la frontera `18-24|25-34`, con area 0.7294.

Figuras: `figuras/v17_p8_area_por_senal.png` y `figuras/v17_p8_distribuciones.png`.

