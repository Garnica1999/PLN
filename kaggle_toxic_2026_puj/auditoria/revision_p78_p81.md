# Revision adversarial P78 a P81 (parcial)

Revisor: ml-reviewer (sesión del 2026-10-02). La sesión agoto pasos antes de
llegar a P82; lo de P82 vive en `revision_p82.md`. No se modifico ningun
archivo existente y el artefacto congelado del Word2Vec quedo verificado
intacto (sha256 302c994d...).

## Veredictos

1. P78: CONFIRMADO en lo esencial. Gate de la rama 1 cumplido: P1=+0.024893,
   5/5 pliegues, EPS_GATE=0.003, coincidente con el pre-registro (seccion 7).
   Protocolo anidado real: `epoca_estrella` usa solo `grupos[ia]` e `ids[ia]`
   (`p75_01_complementariedad_textcnn.py:112-160`) y `entrenar_fijo` no
   selecciona sobre ib (`ib:211-215`); 60 unidades con una evaluación cada una.
   El addendum declara sin adornos el IC que contiene cero y la sd entre
   semillas de 0.0216.
   - Discrepancia menor: el JSON dice "E* entre 10 y 17" y el registro da 8 a
     17, con 12 casos por debajo de 10.
   - Matiz: `lr_de` esta diseñado para 40 epocas; la funcion es la misma en
     los cuatro brazos, pero E* difiere por brazo (C 12-17, D 8-13), de modo
     que la trayectoria efectiva no es identica. No sesga el delta pareado por
     construccion, pero no anula la varianza.
2. P79: CONFIRMADO CON SALVEDADES. El 1/K queda confirmado en
   `p71_04:214-216` (ratios 0.2/0.1/0.05 para K=5/10/20; cuatro K auditadas:
   1, 5, 10, 20). El recuento del scheduler NO queda confirmado:
   `range(0..4)` del plan contra `enumerate(start=1)` del entrenamiento
   (`p72_01:92-101`) da 1395 contra 1398 lotes, desvio 0.00215. Bug de
   instrumentacion en `p79_01:613-614`: clasifica ambos bucles como
   "entrenamiento" y publica `semillas_desalineadas=false`, contradiciendo su
   propio veredicto; P80 lo corrige explicitamente. La inicializacion a cero
   no alcanza al artefacto congelado porque `p72_01:77-78` sobrescribe la
   salida a uniforme.
3. P80: REFUTADO EN UNA PIEZA. "19 de 19 confirmadas" es impreciso: el
   informe real es 18 CONFIRMADO mas 1 NO APLICABLE POR CONSTRUCCION (A13).
   Confirmado que el npz solo tiene `W` y `palabras`: el par (E,S) con E=S=W
   nunca existio. Confirmado que v9_04 (`:105-155`) y v10_11 (`:240-275`)
   eligen epoca con el pliegue externo (no anidado). Pero "su pool de CV
   incluia el holdout" NO aplica a esos dos: ambos apartan el holdout
   (`v9_04:163-167`, `v10_11:303-307`); el pool de 1731 filas es el de la fase
   DL (`11_dl_desde_cero.py:446,491,1362`). La correccion de P80 se sostiene;
   la frase mezclaba fases.
4. P81: PARCIAL. El informe declara 18 CONFIRMADO mas 2 REFUTADO (A6 y A7) y
   el titular EL_TEST_NO_APOYA_A_LA_SUMA, con A6 publicado como información
   sin reinterpretar (`p81_cierre_huecos.md:32-33,51-64`). Las cifras de
   relleno y OOV (0.73091917 y 0.04124873) no se verificaron contra el script,
   ni el resto de bloques.

## NO_EVALUABLE

- P82 completo (cinco brazos, determinismo, MDE, GPU e hilos) y la sonda
  P82-02: se movieron a `revision_p82.md`.
- Transversal A (hiperparámetros P78-P81 contra pre-registros): parcial.
- Transversal B (early stopping sobre el pliegue evaluado en P75-P77): sin
  completar.
- Transversal C (fugas test/holdout con AST completa): sin completar.
- Transversal D: avance parcial; D1-ALTA en p80 y p81, D3 rama muerta en
  `p81:1778`, D1-MEDIA en `p78:642` (`_dos_pasos`, solo humo).

## Correcciones que deben sobrevivir

- Etiquetar "19/19" de P80 como "18 confirmadas mas 1 no aplicable".
- Documentar el bug de `semillas_desalineadas` de P79 (ya corregido por P80).
- Acotar la frase del pool de CV al experimento DL-01, no a v9_04/v10_11.
