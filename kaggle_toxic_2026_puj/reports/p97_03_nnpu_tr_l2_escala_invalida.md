# P97-03 (H97-C): nnPU/PNU para la cabeza TR

Pre-registro en `p97_independent_research.md` seccion 5. Forma no negativa con P, N y U; HS y AG estandar en ambos brazos.

| Regla | corrected_cv | F1 TR | std pliegues |
|---|---|---|---|
| propios control | 0.6888590064 | 0.729839 | 0.01682 |
| propios nnpu | 0.5994521889 | 0.461618 | 0.01234 |
| frozen control | 0.4567237512 | 0.208835 | 0.00770 |
| frozen nnpu | 0.5409847232 | 0.461618 | 0.00160 |

Comparacion principal (nnpu - control, regla propia): delta -0.089407, pliegues 0/5, delta F1 TR -0.268220, pliegues TR 0/5, IC95 clusters [-0.099985, -0.079065].

Zona no definida (HS=0): tasa de TR=1 predicha - V03 0.0934 | control 0.2156 | nnpu 1.0000.

**Veredicto pre-registrado: FALSIFICADA**
