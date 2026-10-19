# P93-R5 - Recomendacion cuantificada para una fase experimental futura

Fecha: 2026-10-13. Insumo: comparacion con el gold completada (P93-R5, sin tercera
revision). Esta recomendacion NO autoriza nada por si misma.

## 1. Cifras

- Discrepancias adjudicadas (decision humana binaria distinta del gold): 171 de 171 comparadas.
- Posibles correcciones con justificacion documentada (elegibles): 14.
- Registradas sin justificacion documentada: 157.
- Confirmadas como error de anotacion: 0.
- Casos sin resolver que se conservan: 5.

## 2. Distribucion por etiqueta

| etiqueta | comparadas (acuerdo) | elegibles | direccion |
|---|---|---|---|
| HS | 23 | 2 | retirar etiqueta 2 / anadir 0 |
| TR | 85 | 4 | retirar etiqueta 1 / anadir 3 |
| AG | 63 | 8 | retirar etiqueta 7 / anadir 1 |
| total | 171 | 14 | retirar 10 / anadir 4 |

Candidatas por componente de muestreo: {'B': 14}; involucran 14 tweets con justificacion documentada.

Nota de dependencia de muestreo: las 14 candidatas provienen del componente B
(enriquecido; razon dominante HIGH_CONFIDENCE_OOF_DISAGREEMENT); el componente A
(referencia aleatoria) no aporto candidatas con justificacion documentada.

## 3. Tamano del efecto y riesgos

- Escala: 14 celdas elegibles sobre ~5193 celdas de train (aprox. 0.27%).
- Muestreo enriquecido por discrepancia: el conjunto de candidatas no es una muestra
  aleatoria y su tasa no es extrapolable.
- Dependencia de A/B: las decisiones finales vieron A/B en ronda 2; corregir el train
  con estas celdas introduce la vision de A/B en el modelo.
- Riesgo de circularidad: evaluar la correccion sobre las mismas celdas usadas para
  definirla infla artificialmente cualquier mejora.
- Riesgo de interpretacion alternativa: en las 14 candidatas no se excluyo formalmente
  una lectura distinta del gold; siguen siendo 'posibles', no confirmadas.
- Grupos near-duplicate: las celdas corregidas pertenecen a tweets con grupos; hay que
  evitar fuga entre particiones si se experimenta.
- Expectativa realista de mejora de Kaggle con 14 celdas: insignificante.

## 4. Protocolo requerido (si el propietario autoriza un experimento)

1. Autorizacion separada; V03 y su gold permanecen intactos y congelados.
2. Crear una COPIA versionada del train (por ejemplo
   data/processed/toxic_train_p93r5_variant.csv) aplicando SOLO las celdas
   elegibles, con manifiesto de cambios celda a celda y hash propio.
3. Diseno agrupado: usar dup_group_id / grupos near-duplicate para que ningun grupo
   quede repartido entre ajuste y evaluacion; la particion canonica no se toca.
4. Comparacion pre-registrada: mismo pipeline/codigo para V03-original y variante;
   la metrica final se calcula en una particion NO usada para definir las correcciones
   (OOF/particiones existentes sin rehacer splits; holdout solo con autorizacion aparte).
5. Nada de ajustar hiperparametros ni umbrales sobre la particion de evaluacion de esta
   comparacion; sin pseudo-labeling; sin submissions.
6. Reportar el delta con incertidumbre y el recuento de celdas cambiadas; con 14 celdas
   se espera un efecto indistinguible de ruido (registrarlo como tal).

## 5. Recomendacion

Evidencia: 171 discrepancias adjudicadas, 14 con justificacion documentada, 0 confirmadas,
5 sin resolver. La evidencia es SUFICIENTE para documentar candidatas concretas y
SUFICIENTE para un experimento LIMITADO de diagnostico bajo el protocolo de la seccion 4,
pero NO es suficiente para esperar una mejora medible de Kaggle ni para modificar el
train oficial: la muestra esta enriquecida, las decisiones dependen de A/B y ninguna
discrepancia fue confirmada tras excluir interpretaciones alternativas.

Veredicto: el experimento limitado es opcional y solo con el protocolo completo;
la via principal recomendada es NO corregir el train con esta base y priorizar nuevas
evidencias (mas anotacion) antes de cualquier reentrenamiento orientado a Kaggle.
