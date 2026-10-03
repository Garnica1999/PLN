# Bitácora de la auditoría integral

Registro cronologico de fases, agentes y resultados. Este fichero lo actualiza
el lider al cerrar cada pieza y cada subagente deja aquí su resumen (maximo
20 líneas).

## 2026-10-02

- F0 completada: estado sucio de P67 (P82 y P83, 187 archivos) asegurado en
  el commit `1026ff1`. Verificado que el verificador oficial admite acentos y
  enie (solo persigue alfabetos no latinos).
- F1 completada: commit `9663afb`. Importados 67 archivos unicos de la rama
  de auditoría v8 y 85 artefactos del rerun (16 submissions regeneradas
  incluidas); fuente de fastText (commit `1142dc4`) copiada a `tools/fastText`
  sin `build/`. Manifiesto con 154 filas. Verificador de caracteres en
  0 errores. Sin conflictos ni sobrescrituras.
- F2 completada: commit `41f8a3f`. Inventario unificado: 483 experimentos
  (411 de registro mas 72 scripts sin fila), 376 scripts clasificados, 43
  filas malformadas confirmadas (P67 a P77), 277 experimentos sin documentar
  en el notebook y P78 a P83 sin seccion propia.
- F3a completada: escaner estático AST sobre 376 scripts
  (`auditoria/escaner_estatico.py`), 2.165 hallazgos (54 ALTA, 319 MEDIA,
  342 BAJA, 1.450 INFO). D1 (funciones probadas sin llamador real) da 154
  casos: 54 sin ningun llamador y 100 solo desde humo. El caso histórico
  `decidir_alpha` ya aparece conectado, lo que valida el detector contra el
  fallo conocido; `con_extra` de P83 sigue sin llamador. D2 (aserciones
  vacias) da 0 con el detector implementado. Sensibilidad probada con
  archivo plantado en /tmp/opencode.
- F3b completada: revisión adversarial del núcleo (`auditoria/revision_nucleo.md`).
  6 de 7 afirmaciones CONFIRMADAS, incluida la recomputacion exacta de
  `0.7772114525165721` y el +0.0036286 de la rejilla de umbrales sobre el
  mismo OOF. REFUTADA parcialmente la afirmacion 5: `v33_01_validacion.py`
  evaluo el MISMO holdout v2 el 2026-09-27 sin registrarlo, de modo que el
  consumo real es 6 de 5; el registro de 5/5 solo cuenta lo declarado.
  Hallazgos nuevos: refits de submission entrenan con las 1731 filas
  (incluye holdout; es el diseño de refit final, se documenta), y el
  `leaderboard_log.csv` no es UTF-8 (cp1252). D2=0 sostenido con AST
  independiente (64 asserts, 0 tautologicos).
- F3c completada: revisión P69-P77 (`auditoria/revision_p69_p77.md`). Ocho
  conclusiones verificadas. Hallazgo CRITICO: el delta de P72-03 se publica
  contra el maximo de las nulas, no contra la media; A queda por encima de la
  media (+0.0105) y solo 1 de 10 nulas lo supera. Hallazgos: P69 es
  re-scoring, no reentrenamiento; P73 sostiene el signo pero no la magnitud.
- F3d completada parcialmente: revisión P78-P81
  (`auditoria/revision_p78_p81.md`). P78 confirmado (E* real 8-17, no 10-17);
  P79 con el recuento del scheduler no confirmado; P80 "19/19" impreciso
  (18 mas 1 no aplicable) y la frase del pool de CV mezclaba fases.
- F3d2 completada: revisión P82 y su sonda (`auditoria/revision_p82.md`).
  Conteo 23/23 exacto pero A7 contaminada: el determinismo "bit a bit" compara
  dos pasadas de la misma corrida y declara igualdad con NaN y
  `equal_nan=True` sobre mascaras. H3: el submission recalcula E* con
  pliegues internos y no replica el pipeline de B1. H4: el MD de la sonda
  niega un envio que el registro si tiene (LB 0.68150).

- F4 COMPLETADA: P83 corregido y ejecutado en completo. Commit `bfa8385`.
  Modo completo con salida 0 en 193,76 segundos (presupuesto 1500 aplicado),
  20 de 20 afirmaciones CONFIRMADO; humo 206 de 206 con la defensa nueva que
  falla si se anade una funcion sin llamador real. A14 mide registro contra
  registro; funciones muertas conectadas (reproducir_liston,
  comparar_contra_el_liston, comprobar_presupuesto, avisar, comprobar);
  afirmaciones vacias convertidas en medidas; A16, A8 y la lista de siete sin
  STOP corregidos; auditoria AST con lista blanca cerrada. Listón de
  calibración reproducido a 1e-7; complementariedad medida; pregunta 1
  NEGATIVA. Aviso de P83 actualizado a EJECUTADO EN COMPLETO. F5 abierta.

- F5 COMPLETADA: commits `82aa067` y `2a94bb8`. Notebook de 451 celdas con la
  tabla de 23 sondas en cp1252, asserts robustos de la fase v9 y las secciones
  122 (auditoria F0-F4), 123 (P78 a P83), 124 (inventario y trazabilidad) y
  125 (correcciones pendientes). Se corrigieron dos errores de sintaxis
  preexistentes (celdas 270 y 433); el notebook compila entero.
- F6 COMPLETADA: `auditoria/sugerencias_080.md` (476 lineas, 206 con enie o
  tilde). Tres candidatos permitidos con expectativa honesta (SGNS corregido,
  DL-01 con protocolo corregido, EM con shrinkage), familias descartadas con
  motivo y referencias verificables (Saerens 2002, Alexandari 2020, Lipton
  2014, Forman 2008, Basile 2019). Conclusion honesta: ninguna via permitida
  promete 0.80 por si sola.
- F7 COMPLETADA: verificador de caracteres en 0 errores; notebook validado y
  sin errores de sintaxis; P83 con 20 de 20 CONFIRMADO en 193,8 s; inventario
  483 experimentos y 376 scripts; unica pieza sin commitear era la hoja de
  ruta, ya commiteada. GO documental.

- F8 COMPLETADA (sin commit): correcciones de la auditoria aplicadas.
  Registro experiments.csv reparado (43 malformadas a 0; 450 filas; 35 filas
  nuevas con P78/P82/P83 curadas y P79/P80/P81/P82-02); etiquetas P69 y P72
  corregidas; sexto uso del holdout registrado (6 de 5) con nota en config;
  leaderboard_log.csv convertido a UTF-8 y seis celdas del notebook pasadas a
  utf-8-sig; fes de erratas en los dos informes de P82; notebook con 122 y 124
  actualizadas, 125 reescrita como correcciones aplicadas y 126 nueva de
  erratas y limites (452 celdas, sin errores de sintaxis); documento
  auditoria/correcciones_aplicadas.md. Sin commit, a revision de Carlos.

- Reejecuciones del 2026-10-02 (sin commit): la cadena v4 de P69 se
  reentreno completa (27, 28 y 29) y los tres artefactos quedaron byte a byte
  identicos; p69_01 pasa 10 de 10 con nivel R3 tras corregir su puerta de
  cumplimiento (excluye el registro y la lista de identificadores del holdout,
  y el generador de submission p69_10, posterior a la corrida original).
  Errata anadida al addendum de P78 (E* real 8-17). El A7 de P82 se corrigio
  para comparar solo posiciones finitas y su corrida completa esta en curso.
  Escaner estatico e inventario re-ejecutados sobre el codigo corregido.

- P82 reejecutado y verificado (2026-10-02): corrida completa con el A7
  corregido, marca `20261002T132340Z_p563891`, EXIT 0 en ~96 minutos. 23 de 23
  CONFIRMADO; A7 compara 891 posiciones finitas con variacion 0.0 y patron de
  NaN identico; las medias de los cinco brazos coinciden con las almacenadas
  (determinismo entre corridas contrastado). Informe vigente publicado y el
  anterior conservado con fe de erratas. Registro, notebook y documento de
  correcciones actualizados. Sin commit.
