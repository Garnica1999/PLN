# P97-03 (H97-C): nnPU/PNU para la cabeza TR

Pre-registro en `p97_independent_research.md` seccion 5. Forma no negativa con P, N y U; HS y AG estandar en ambos brazos.

| Regla | corrected_cv | F1 TR | std pliegues |
|---|---|---|---|
| propios control | 0.7748502494 | 0.813115 | 0.01966 |
| propios nnpu | 0.7770004038 | 0.819565 | 0.01874 |
| frozen control | 0.7772114525 | 0.815730 | 0.02309 |
| frozen nnpu | 0.7704992186 | 0.795594 | 0.02278 |

Comparacion principal (nnpu - control, regla propia): delta +0.002150, pliegues 5/5, delta F1 TR +0.006450, pliegues TR 5/5, IC95 clusters [-0.000461, 0.004896].

Zona no definida (HS=0): tasa de TR=1 predicha - V03 0.0934 | control 0.1126 | nnpu 0.1114.

**Veredicto pre-registrado: SENAL**

## Nota de ejecuciones (trazabilidad)

Dos instanciaciones previas invalidas, preservadas sin sobrescribir:

- `*_l2_escala_invalida.*`: Adam wd=0.5 (= 2400x el equivalente de C=2 en
  liblinear) producia subentrenamiento del arnes completo.
- `*_pn_idx_invalido.*`: codificacion P/N/U erronea (N entro como positivo y
  el conjunto de negativos quedo vacio).

La ejecucion presente corrige ambas (wd = 1/(C*n); P = TR=1, N = TR=0 y HS=1,
U = HS=0) y es la ejecucion valida del mecanismo tal como se pre-registro.
