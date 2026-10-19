# P93-R3 - Auditoria de cegamiento de la ronda 1

Fecha: 2026-10-13. **Resultado: ROUND1_BLINDING_NOT_FULLY_VERIFIABLE**
(evidencia de diseno completa; comportamiento de acceso no verificable).
No se afirma independencia total de la ronda 1.

## 1. Verificado (nivel de diseno y contenido)

- El paquete de ronda 1 no contenia gold, respuestas ni notas de A/B, lista
  de candidatos, motivos de seleccion, componente, grupos, OOF, V03 ni
  umbrales (auditoria de P93-R: 15/15, verificada de nuevo en P93-R3).
- IDs opacos P93R-* sin correspondencia visible con review_id de la ronda
  principal; la clave de union permanece en reports/ (fuera del paquete).
- El README instruia no consultar ningun otro material.
- Las notas de C1/C2 no contienen referencias a A/B, gold ni candidatos.

## 2. No verificable (limitaciones registradas)

- No existen registros de acceso ni acuse de recibo firmado: el
  comportamiento real de consulta no puede demostrarse.
- El paquete incluia un manifest.json con RECUENTOS AGREGADOS de celdas
  candidatas (127/176 y 23/87/66 por etiqueta; sin lista, sin gold, sin
  A/B). No puede verificarse si los revisores lo leyeron.
- Las dos respuestas se guardaron en la MISMA carpeta compartida con 10
  segundos de diferencia; la visibilidad mutua de archivos ya guardados no
  puede descartarse.

## 3. Firma de posible contaminacion (verificada: sin senal)

- Las tasas de coincidencia de C1/C2 con A/B en celdas A==B binarias se
  compararon entre candidatas (mismatch gold) y no candidatas (match gold):
  C1: {'cand_utd': 2, 'no_cand_coincide': 109, 'cand_coincide': 174}
  C2: {'cand_coincide': 159, 'no_cand_coincide': 96, 'cand_utd': 8, 'no_cand_utd': 8, 'cand_difiere': 9, 'no_cand_difiere': 5}
- Una contaminacion por copia produciria coincidencia anomala concentrada
  en las candidatas; los patrones observados se documentan en la auditoria
  de mapeo (p93r3_mapping_integrity_audit.md).

## 4. Consecuencia

- La ronda 2 procede, pero las conclusiones futuras deberan arrastrar la
  limitacion: la independencia de la ronda 1 no es demostrable al 100 por
  ciento. La ronda 2 mantiene el cegamiento cruzado entre revisores.
