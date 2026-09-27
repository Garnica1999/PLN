# V18 · P5 — Destilacion del conocimiento de frontera

## 1. Diseno

P5 no repite P1, P3 ni P4: aqui el objetivo auxiliar no es la etiqueta verdadera (que ya esta en la entropia cruzada) sino la **prediccion suave del teacher**, que es informacion nueva. Y **no hay routing en inferencia**: la senal se descarga durante el entrenamiento.

**Teacher**: el especialista binario sobre las **279 variables de V12**, ya validado en P0' con el protocolo completo (0.7136 y 0.7493 en las filas de cada frontera), no el de TF-IDF disperso, porque es mejor y esta en la misma representacion que el student, que es la condicion para que el student pueda expresar lo que se le ensena.

**Cross-fitting**: el teacher se cross-fita en cinco inner folds **agrupados** del train exterior, de modo que cada ejemplo del train exterior fue predicho por un teacher que no lo uso para entrenarse. El teacher ajustado sobre todo el train exterior predice la validacion exterior **solo para diagnostico**, y esas predicciones nunca entran en el entrenamiento del student.

**Cuadrica** de 6 combinaciones de peso y temperatura, **seleccionada dentro del train exterior** en una particion interior del 25%. La validacion exterior no participa en la eleccion.

**Senales**: la probabilidad de la clase superior de cada pareja, equivalente a su margen por la sigmoida, aplicada **solo a las filas de esa pareja**. Sin pseudo-etiqueta dura en ningun punto, y la etiqueta verdadera es siempre el objetivo principal.

## 2. Resultado

| Brazo | Lambda | T | macro-F1 | Delta | Media folds | Folds pos. | IC95 | 18-24\|25-34 | 25-34\|35-49 | F1 25-34 | Rescate | Dano | Adopta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| control_sin_destilacion | 0.0 | 1 | 0.5983 | -0.0022 | 0.598127 | 0/5 | [-0.0029, -0.0014] | 0.4957 | 0.495 | 0.4846 | 2,112 | 2,473 | **no** |
| student_destilado | 0.3 | 1 | 0.599 | -0.0015 | 0.598885 | 0/5 | [-0.0023, -0.0008] | 0.498 | 0.4966 | 0.4856 | 1,979 | 2,255 | **no** |
| student_destilado | 0.3 | 2 | 0.5991 | -0.0014 | 0.599006 | 0/5 | [-0.0022, -0.0007] | 0.4969 | 0.4973 | 0.486 | 1,947 | 2,210 | **no** |
| student_destilado | 0.3 | 4 | 0.5988 | -0.0017 | 0.598653 | 0/5 | [-0.0024, -0.0009] | 0.496 | 0.4943 | 0.4832 | 1,940 | 2,235 | **no** |
| student_destilado | 1.0 | 1 | 0.5986 | -0.0019 | 0.598516 | 0/5 | [-0.0027, -0.0012] | 0.4956 | 0.4949 | 0.4847 | 1,984 | 2,309 | **no** |
| student_destilado | 1.0 | 2 | 0.5991 | -0.0014 | 0.59906 | 0/5 | [-0.0022, -0.0006] | 0.4964 | 0.4937 | 0.4834 | 2,067 | 2,307 | **no** |
| student_destilado | 1.0 | 4 | 0.5986 | -0.0019 | 0.598421 | 0/5 | [-0.0027, -0.0012] | 0.4936 | 0.495 | 0.483 | 1,982 | 2,301 | **no** |
| V12 (referencia) | - | - | 0.6005 | - | 0.6004 | - | - | 0.4963 | 0.4974 | 0.4858 | - | - | - |

## 3. Cuanto recupera el student del teacher disponible

| Brazo | Frontera | V12 | Student | Teacher validado | Recuperacion |
|---|---|---|---|---|---|
| control_sin_destilacion | 18-24|25-34 | 0.4963 | 0.4957 | 0.7136 | -0.3% |
| control_sin_destilacion | 25-34|35-49 | 0.4974 | 0.495 | 0.7493 | -0.9% |
| student_destilado | 18-24|25-34 | 0.4963 | 0.498 | 0.7136 | 0.8% |
| student_destilado | 25-34|35-49 | 0.4974 | 0.4966 | 0.7493 | -0.3% |
| student_destilado | 18-24|25-34 | 0.4963 | 0.4969 | 0.7136 | 0.3% |
| student_destilado | 25-34|35-49 | 0.4974 | 0.4973 | 0.7493 | -0.0% |
| student_destilado | 18-24|25-34 | 0.4963 | 0.496 | 0.7136 | -0.1% |
| student_destilado | 25-34|35-49 | 0.4974 | 0.4943 | 0.7493 | -1.2% |
| student_destilado | 18-24|25-34 | 0.4963 | 0.4956 | 0.7136 | -0.3% |
| student_destilado | 25-34|35-49 | 0.4974 | 0.4949 | 0.7493 | -1.0% |
| student_destilado | 18-24|25-34 | 0.4963 | 0.4964 | 0.7136 | 0.1% |
| student_destilado | 25-34|35-49 | 0.4974 | 0.4937 | 0.7493 | -1.5% |
| student_destilado | 18-24|25-34 | 0.4963 | 0.4936 | 0.7136 | -1.2% |
| student_destilado | 25-34|35-49 | 0.4974 | 0.495 | 0.7493 | -0.9% |

## 4. Combinacion elegida por fold (seleccion interior)

| Brazo | Fold | Lambda | Temperatura |
|---|---|---|---|
| control_sin_destilacion | 0 | 0.0 | 1 |
| control_sin_destilacion | 1 | 0.0 | 1 |
| control_sin_destilacion | 2 | 0.0 | 1 |
| control_sin_destilacion | 3 | 0.0 | 1 |
| control_sin_destilacion | 4 | 0.0 | 1 |
| student_destilado | 0 | 1.0 | 1 |
| student_destilado | 1 | 0.3 | 2 |
| student_destilado | 2 | 0.3 | 2 |
| student_destilado | 3 | 1.0 | 1 |
| student_destilado | 4 | 0.3 | 4 |
| student_destilado | 0 | 0.3 | 1 |
| student_destilado | 1 | 0.3 | 1 |
| student_destilado | 2 | 0.3 | 1 |
| student_destilado | 3 | 0.3 | 1 |
| student_destilado | 4 | 1.0 | 4 |
| student_destilado | 0 | 0.3 | 1 |
| student_destilado | 1 | 0.3 | 4 |
| student_destilado | 2 | 1.0 | 1 |
| student_destilado | 3 | 1.0 | 4 |
| student_destilado | 4 | 0.3 | 2 |
| student_destilado | 0 | 1.0 | 4 |
| student_destilado | 1 | 0.3 | 4 |
| student_destilado | 2 | 1.0 | 1 |
| student_destilado | 3 | 1.0 | 2 |
| student_destilado | 4 | 0.3 | 2 |
| student_destilado | 0 | 0.3 | 4 |
| student_destilado | 1 | 0.3 | 4 |
| student_destilado | 2 | 0.3 | 2 |
| student_destilado | 3 | 1.0 | 2 |
| student_destilado | 4 | 1.0 | 4 |
| student_destilado | 0 | 1.0 | 2 |
| student_destilado | 1 | 0.3 | 2 |
| student_destilado | 2 | 1.0 | 1 |
| student_destilado | 3 | 0.3 | 2 |
| student_destilado | 4 | 0.3 | 1 |
