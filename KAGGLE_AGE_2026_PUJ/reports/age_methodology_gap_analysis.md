# AGE - gap analysis de mecanismos

**Conclusion: `AGE_RESEARCH_CLOSED`**

## La pregunta que queda contestada

> ¿Queda algun mecanismo de aprendizaje realmente distinto que pueda convertir informacion ya disponible en una correccion robusta, sin repetir una familia que ya fallo?

**No dentro de las restricciones del proyecto.** El razonamiento, en tres pasos.

### Paso 1: la informacion disponible esta localizada

- Grupo A: **27,430** errores sin ninguna fuente que los corrija (41.9%).
- Grupo B: **37,973** con al menos una fuente (58.1%), de los cuales **30,038** son de rescate compartido.
- Rescates exclusivos: **8,029** entre **23** fuentes, sin estructura comun.

### Paso 2: las vias de integracion estan cerradas con control

| Via | Experimento | Resultado |
|---|---|---|
| feature anadida al meta-espacio | V31-06 | +0,0002948 sobre el control barajado |
| miembro de la mezcla | V31-05 | +0,0000642 sobre el reoptimizado |
| residuo global | P33 | -0,0002152; el ruido aleatorio gana |
| residuo local por pares | P34 | -0,0002039; cero rescates de alta confianza |
| residuo con pendiente en el margen | G12 | +0,0003018 sin superar a la constante, y baja el AUC |

Las cinco tienen **control de atribucion**, y en las cinco el control gana o empata. No es que la integracion se haya intentado poco: es que se intento cinco veces con la misma conclusion.

### Paso 3: los mecanismos sin usar atacan poblaciones sin senal

Los cinco mecanismos que no aparecen en el codigo son focal loss, hard example mining, label smoothing, ranking objective y bootstrap de grupos. Los tres primeros apuntan a los ejemplos dificiles y a la frontera, que es donde la medicion dice que el ruido gana. El cuarto no mueve la metrica. El quinto es una variante del bagging ya cerrado.

## Lo que habria que cambiar para que algo funcionara

No es un mecanismo, y por eso no se propone como hipotesis: haria falta **informacion que no esta en los datos disponibles**. Las tres fuentes que lo aportarian estan prohibidas o no existen:

1. **Metadatos de la cuenta** (fecha de registro, ubicacion, historial): las reglas de AGE son de texto, y el brief los prohibe.
2. **El contenido de las URLs**: no hay ninguna URL de status en el dataset, y las 1.685 con dominio truncado no apuntan a ningun sitio resoluble.
3. **Modelos semanticos** (embeddings, transformers): prohibidos por las reglas de la competicion.

## Cierre

La investigacion de AGE se cierra con V30 como mejor resultado validado y todas las vias metodologicas disponibles recorridas o cerradas con evidencia. El detalle completo esta en `reports/age_final_methodology_audit.md` y la matriz en `reports/age_methodology_coverage_matrix.csv`.
