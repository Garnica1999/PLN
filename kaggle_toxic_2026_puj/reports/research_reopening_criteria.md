# Criterios de reapertura de la investigacion (definidos, no ejecutados)

Fecha: 2026-10-14. Estado actual: MODEL_EXPLORATION_STATUS = PAUSED; V03
congelado (corrected_cv 0.7772114525165721; LB 0.78306).

La reapertura futura solo puede proponerse si se cumple AL MENOS UNA de estas
condiciones:

```text
A. nueva autorizacion academica escrita que modifique explicitamente el
   espacio permitido;
B. nueva evidencia supervisada admisible (p. ej. adjudicacion de labels con
   revisores reales y gobernanza resuelta);
C. evidencia concreta de un defecto de implementacion en el flujo vigente;
D. una hipotesis genuinamente nueva con mecanismo falsable, que no sea
   reformulacion de lineas cerradas.
```

## 1. Que NO cuenta como nuevo (prohibido reabrir por esta via)

- Aumentar tokens o documentos sin una hipotesis distinta.
- Probar otras proporciones de ChatSubs.
- Incrementar o reducir dimensiones de la misma BiLSTM sin evidencia
  especifica.
- Cambiar seeds para buscar una particion favorable.
- Ajustar thresholds repetidamente.
- Optimizar la fusion sobre el mismo OOF.
- Probar nuevas variantes de familias cerradas sin un cambio metodologico
  justificable.

## 2. Protocolo si se reabre

- Si una autorizacion amplia el espacio permitido, definir PRIMERO una prueba
  pequena y falsable; no lanzar una grid general.
- Toda hipotesis nueva debe demostrar su diferencia frente a los experimentos
  existentes mediante codigo, configuraciones y resultados historicos
  (regla P2: mecanismo distinto, no otra implementacion del mismo mecanismo).
- Gate estandar vigente (A3): delta canonico >= +0.003; >= 4/5 pliegues; IC95
  de bootstrap por clusters que excluya cero; replicacion positiva en la
  particion 47; >= 200 celdas cambiadas; y control de reproduccion de V03
  (diff 0.0) en ambas particiones.
- Piso de deteccion del arnes (medido en P102-P105): se ~0.0035; los efectos
  menores se reportan como "sin senal detectable" y no deben usarse como
  argumento de cambio.
