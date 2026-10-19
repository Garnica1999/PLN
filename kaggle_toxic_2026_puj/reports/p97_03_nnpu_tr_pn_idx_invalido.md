# P97-03 (H97-C): nnPU/PNU para la cabeza TR

Pre-registro en `p97_independent_research.md` seccion 5. Forma no negativa con P, N y U; HS y AG estandar en ambos brazos.

| Regla | corrected_cv | F1 TR | std pliegues |
|---|---|---|---|
| propios control | 0.7748502494 | 0.813115 | 0.01966 |
| propios nnpu | 0.6576847505 | 0.461618 | 0.01479 |
| frozen control | 0.7772114525 | 0.815730 | 0.02309 |
| frozen nnpu | 0.6591740926 | 0.461618 | 0.01672 |

Comparacion principal (nnpu - control, regla propia): delta -0.117165, pliegues 0/5, delta F1 TR -0.351496, pliegues TR 0/5, IC95 clusters [-0.128232, -0.106364].

Zona no definida (HS=0): tasa de TR=1 predicha - V03 0.0934 | control 0.1126 | nnpu 1.0000.

**Veredicto pre-registrado: FALSIFICADA**
