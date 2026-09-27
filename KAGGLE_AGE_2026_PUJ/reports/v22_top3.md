# V22 · TOP 3 — candidatos ejecutables

Seleccionados por `novedad x plausibilidad x bajo coste x transferibilidad`, **no** por ser los mas sencillos.

## 1. coste de error proporcional a la distancia ordinal

- **Novedad**: es el unico **objetivo de entrenamiento** que queda sin probar; V18 cerro las distribuciones de masa, los margenes y la destilacion, pero nunca un coste ordinal
- **Plausibilidad**: alta: la metrica es macro-F1 de seis clases y un error de cinco categorias deberia pesar mas que uno de una
- **Coste**: bajo: una matriz de costes sobre la salida existente, sin reentrenar los cinco discriminadores
- **Riesgo**: medio: podria reproducir la leccion de P1 (mejorar la frontera y empeorar la global)
- **Relacion con V20**: -
- **Potencial de complementariedad**: si el coste ordinal mejora la global sin degradar las fronteras, seria un mecanismo nuevo
- **Evidencia externa**: Frank & Hall 2001; O'Connor 2024 en edad y genero desde Twitter

## 2. CORAL / cumulative-link con pesos compartidos

- **Novedad**: E-021a usa K-1 binarios **independientes**; compartir pesos entre los cinco umbrales es una formulacion ordinal distinta
- **Plausibilidad**: media: las cinco tareas estan relacionadas por el orden de edad
- **Coste**: medio: un unico modelo con cinco salidas o una representacion lineal compartida
- **Riesgo**: medio: podria no superar a los cinco independientes, que ya funcionan
- **Relacion con V20**: -
- **Potencial de complementariedad**: si mejora el macro-F1, seria una via nueva
- **Evidencia externa**: Frank & Hall 2001 es el antecedente; la version con pesos compartidos no esta en el historial

## 3. feature selection supervisada de las 279 features

- **Novedad**: V15 podo **sin** supervision; la seleccion supervisada no se ha hecho
- **Plausibilidad**: media-baja: puede reducir ruido, pero LightGBM ya maneja 279 features
- **Coste**: bajo: informacion mutua o chi-cuadrado dentro de cada fold
- **Riesgo**: medio: puede perder senal que el boosting si aprovecha
- **Relacion con V20**: -
- **Potencial de complementariedad**: limitado, pero es barato y legal
- **Evidencia externa**: estandar en clasificacion de texto; no es novedoso
