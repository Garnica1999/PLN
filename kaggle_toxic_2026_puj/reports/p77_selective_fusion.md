# P77: operador selectivo

pre-registro sha256 `024b0288af5746baaa1b4acfbdade3c94fb2fbcf29601f41447c893706e7b0ca`, escrito y hasheado antes de ejecutar

## La pregunta

> La informacion complementaria de Word2Vec + TextCNN puede transferirse al punto de operacion de V03 mediante un operador selectivo que modifique la decision solo cuando exista evidencia suficiente de que la CNN es util?

## Los mecanismos

```text
A_V03                    (1483, 3) V03 intacto
B_CNN_sola               macro-F1 0.5838755350
C nivel 1, cuantil 80      154 conmutaciones
D nivel 2, cuantil 50      393 conmutaciones
E gate lineal anidado    543 conmutaciones
```

## Resultados

```text
A_V03                    cv 0.7772114525  delta +0.0000000000  folds 0/5  switches    0
B_CNN_sola               cv 0.5838755350  delta -0.1933359176  folds 0/5  switches 1081
C_determinista_nivel1    cv 0.7579728990  delta -0.0192385535  folds 0/5  switches  154
D_determinista_nivel2    cv 0.7259394272  delta -0.0512720253  folds 0/5  switches  393
E_gate_nested            cv 0.7215155047  delta -0.0556959478  folds 0/5  switches  543
```

## El objetivo real de P77

```text
A_V03                    rescue conservado   0/268  damage introducido   0/813  damage evitado 813  rescue perdido 268
C_determinista_nivel1    rescue conservado  56/268  damage introducido  98/813  damage evitado 715  rescue perdido 212
D_determinista_nivel2    rescue conservado 131/268  damage introducido 262/813  damage evitado 551  rescue perdido 137
E_gate_nested            rescue conservado 191/268  damage introducido 352/813  damage evitado 461  rescue perdido  77
```

## Interseccion del gate

```text
rescates_seleccionados             191
rescates_no_seleccionados          77
danos_seleccionados                352
danos_no_seleccionados             461
precision_del_selector             0.35175
precision_de_la_base               0.247919
```

## Bootstrap incondicional y McNemar

```text
C_determinista_nivel1    -0.019315 IC95 [-0.028184, -0.009886]  McNemar p 0.000954
D_determinista_nivel2    -0.051360 IC95 [-0.066903, -0.035310]  McNemar p 0.000000
E_gate_nested            -0.055807 IC95 [-0.072841, -0.039859]  McNemar p 0.000000
```

## Veredicto

**CASO_1_NINGUN_OPERADOR_TRANSFIERE**

Finalistas: ninguno

## Limitaciones

```text
- los cortes por cuantil se calculan con modelos internos mas debiles, luego el sesgo del gate es conservador y el de los cortes es a la alta, y no se cancelan de forma acotada
- las slices se solapan y no son pruebas independientes, y son solo diagnosticas
- las unidades estan agrupadas tambien por tweet, luego el numero efectivo es 1415 y no 1483
- el AUC del gate en outer se reporta con etiquetas de outer: es medicion, nunca ajuste
```
