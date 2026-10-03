# Fases V36 y V37 del notebook consolidado (fuente para `scripts/auditoria_v1v37_notebook.py`)

Este archivo existe para que el script sea **ASCII puro** (regla 8.4 del proyecto) y el texto
visible del notebook pueda llevar tildes. Los capitulos van separados por la linea de abajo, y
**todo lo anterior al primer separador es cabecera y se descarta**.

    ===CELDA===

# PHASE - V36 (VIA C1-A): CIERRE DE LA BRECHA L4

V36 responde a la ultima brecha de representacion que quedaba abierta: las **transiciones de clase de
caracter**. La brecha L4 se definia como "la forma en que un caracter sigue a otro", y era la unica
familia de la auditoria de brechas que no se habia medido con un banco propio.

## El pre-registro, y las dos correcciones que necesito

El protocolo se congelo antes de correr (`reports/v36_preregistro.md`,
`experiments/v36_00_preregistro.json`) y **C1-A es one-shot**: no se reintenta con otro alfabeto, otro
rango de n, otro mapa ni otro corte.

La enmienda pre-ejecucion 1 corrigio **dos controles degenerados** que la spec habia declarado y que
**no podian fallar**:

- **Permutar los nombres de las clases** es una renombracion biyectiva: produce una permutacion exacta
  de columnas con el mismo IDF, luego la matriz queda identica y el control no discrimina nada.
- **Barajar dentro del documento** deja la matriz **bit a bit identica**, porque el bloque es una bolsa
  de n-gramas de transicion y el orden interno no la altera.

Se sustituyeron por **remapeo aleatorio de caracteres** y **barajado entre documentos**, que si pueden
refutar la especificidad. Es el mismo patron de defecto que la auditoria encontro despues en otros
sitios: un control que no puede fallar es peor que no tener control, porque da confianza falsa.

## El banco: la senal existe

El banco de transiciones se mide sobre las 6 clases del alfabeto literal
(`S` espacio, `D` digitos, `P` puntuacion, `L` minuscula, `M` mayuscula, `O` resto) con 27 bigramas y
127 trigramas, **154 columnas observadas** de las 154 esperadas.

| Brazo | macro-F1 | folds |
|---|---|---|
| Referencia (sin transiciones) | 0.5408073 | — |
| **Banco C1-A** | **0.5440346** | 5 de 5 positivos |

El banco supera su hipotesis nula con un delta de **+0.0032273** y un IC95 pareado de
**[+0.0022338, +0.0042230]** que **excluye cero**. La senal existe y es especifica: el control por
remapeo aleatorio de caracteres rinde por debajo del mapa real.

## La integracion: y sin embargo no aporta

| Medida | Valor |
|---|---|
| delta integrado contra el campeon | **−0.0016013** |
| folds positivos | **0 de 5** |
| IC95 pareado contra el campeon | excluye cero (hacia abajo) |
| control favorable | si |

El nodo del arbol es **`SENAL_EXISTE_INTEGRACION_NEGATIVA`**: la senal es real, especifica y
reproducible, pero **el campeon ya la expresa mejor**. No es que la representacion no tenga
informacion: es que V30 la captura con menos coste.

## Decision y limites

**L4 se cierra por el nodo del arbol.** C1-A es one-shot y no se reintenta con otro alfabeto, otro
rango de n, otro mapa ni otro corte. Acta **D-088**.

El hallazgo transferible es de metodo, no de rendimiento: **una representacion puede tener senal
medible en aislamiento y aun asi no aportar nada al integrarse**, porque el modelo que ya existe la
absorbe. Es la misma leccion que la fase linguistica habia dado seis veces.

```python
import json, pathlib
RAIZ = pathlib.Path.cwd()
while not (RAIZ / "experiments.csv").exists() and RAIZ != RAIZ.parent:
    RAIZ = RAIZ.parent
v36 = json.loads((RAIZ / "experiments" / "v36_01_transiciones_clase.json").read_text(encoding="utf-8"))
print("fase              :", v36["fase"], "|", v36["fecha"])
print("baseline          :", v36["baseline_recomputado"])
print("columnas          :", v36["columnas_reales"]["total"], "observadas =", v36["columnas_observadas"]["total"], "esperadas")
for nombre, brazo in v36["banco"].items():
    if "macro_f1" not in brazo:      # el diccionario incluye una clave `delta` con otra forma
        continue
    print(f"  banco {nombre:20} macro-F1 {brazo['macro_f1']:.7f}  cambios {brazo['n_cambios_respecto_a_referencia']}")
ver = v36["veredicto"]
print("nodo              :", ver["nodo"])
print("delta integrado   :", round(ver["traza"]["delta_integrado"], 7))
print("sondas / especif. :", ver["traza"]["sondas_ok"], "/", ver["traza"]["especificidad_ok"])
print("banco supera H0   :", ver["traza"]["banco_supera_h0"], "| integracion supera H0:", ver["traza"]["integracion_supera_h0"])
print("cierra L4         :", ver["cierra_L4"][:96])
# un gate auxiliar (la variabilidad de referencia) no lleva `pasa`: se lee con get
print("gates             :", {k: v.get("pasa") for k, v in v36["gates"]["gates"].items()})
```

    ===CELDA===

# PHASE - V37: DIAGNOSTICO `P(POS|edad)` — LA FAMILIA POS CERRADA POR POTENCIA

V37 responde a una pregunta que **nunca se habia medido**, y que conviene no confundir con otra:

- `P(edad | palabra)` — la probabilidad de una clase dado un token. **Ya medida** en V13-P1a (D-041).
- `P(POS | edad)` — la distribucion condicional de **categorias gramaticales** por rango de edad.
  **Esta es la que V37 mide.**

No se entreno nada del campeon: V37 es un diagnostico, no un challenger.

## La cota de potencia: el numero que decide

Un clasificador **naive Bayes multinomial cerrado** (Laplace alpha = 1.0 fijo, 18 etiquetas UPOS,
leave-one-fold-out, sin ningun paso de seleccion y por tanto sin CV anidada) sobre **proporciones** POS:

| Referencia | macro-F1 | Que es |
|---|---|---|
| Azar uniforme (6 clases equiprobables) | 0.1666667 | suelo de azar |
| **A1 = NB sobre proporciones POS** | **0.1720532** | **la cota de este diagnostico** |
| A2 = NB sobre conteos crudos | 0.2052270 | sensibilidad a longitud |
| P2 (secuencia POS, n-gramas 1-3) | 0.2750921 | ya medido, integracion −0.0086 (D-077) |
| P1a (`P(clase \| token)` cross-fit) | 0.3997 | ya medido, integracion −0.0002 (D-041) |
| Campeon V30 + plug-in | 0.6159732 | intocado |

**La informacion POS marginal supera el azar por +0.0054**, por debajo del margen de 0.01 que el propio
diseno declaraba como minimo para tratarla como senal. El IC95 pareado de `A1 − A_perm` es
**[+0.0900353, +0.0942188]** y **excluye cero**, luego el modelo no es degenerado: **la senal es
simplemente diminuta**.

## El patron si existe: transiciones POS de primer orden

| Medida | Valor |
|---|---|
| `D` (divergencia entre clases) | **0.5695637** · IC95 [0.37976, 0.684608] |
| `R` (razon entre clases / entre folds) | **2.0645054** · IC95 [1.5748, 2.3354] |
| bigramas | 2.421.645 = esperado |

Los umbrales estan **bien calibrados**: bajo permutacion de etiquetas **0 celdas** sobreviven los floors
de V1..V4, frente a **20** en el dato real.

El gradiente por edad es coherente con la literatura: al crecer la edad suben `NOUN`, `DET` y `ADJ`, y
bajan `PRON`, `ADV`, `PUNCT` y `SPACE`. Este ultimo cae de **1,956 %** de los tokens en 13-17 a
**0,911 %** en 35-49 (factor 2,1), lo que confirma que es senal de **formato confundida con longitud**.

## El cierre: inestabilidad entre anotadores, no falta de senal

| Criterio | Resultado |
|---|---|
| V1 | cumple |
| V2 | cumple (20 de 20 celdas) |
| V3 | **no** (14 de 20) |
| V4 | **no** (16 de 20) |
| V5 | **no** (falla en md y en lg) |
| V6 | cumple |

De 27 celdas evaluables, las 27 son estables entre folds, pero solo **20 de 27** superan el umbral de
acuerdo md-lg. **Ese es el motivo formal de `DIAGNOSTICO_CERRADO`**, y lo produce el gate **G2**.

No invalida la conclusion: ambos anotadores proyectan sobre el **mismo inventario de 18 UPOS** por
construccion y coinciden solo en el **16,3 %** de las secuencias fila a fila, luego su desacuerdo es una
**limitacion del conjunto de datos**, no una propiedad del fenomeno.

## Lo que NO se ejecuto

El plan preveia que, si la auditoria de novedad daba "(i) misma familia ya cerrada" o "(ii) variante
disfrazada", el paso predictivo **no** se correria. Dio **(ii) variante disfrazada**: una matriz de
transiciones indexada por clase es una interaccion de orden tres que un modelo lineal sobre n-gramas no
expresa, pero el refinamiento por clase ya se habia medido en la direccion dual (P1a) y dos integraciones
de este tipo ya habian fallado (D-077, D-088).

**El canal queda deshabilitado** salvo autorizacion expresa con hipotesis nueva escrita por adelantado.

## Dos hallazgos de metodo que conviene conservar

1. **El control de fuga `A_perm` habria dado un `[STOP]` falso** sobre un pipeline limpio: el NB
   permutado degenera hacia la clase mayoritaria (159.653 de 169.336 filas), no hacia el azar uniforme.
   El criterio original estaba replicado en cinco sitios del codigo.
2. **El autoejercicio estructural mato 25 defectos reales** antes de gastar presupuesto, cuatro de ellos
   silenciosos. Uno corrompia la fase de transiciones **completa** sin dar ningun error visible. Es la
   leccion de D-086 pagada por segunda vez.

```python
import json, pathlib
RAIZ = pathlib.Path.cwd()
while not (RAIZ / "experiments.csv").exists() and RAIZ != RAIZ.parent:
    RAIZ = RAIZ.parent
v37 = json.loads((RAIZ / "experiments" / "age_linguistic_14_diagnostico_pos.json").read_text(encoding="utf-8"))
print("fase              :", v37["fase"], "|", v37["fecha"], "|", v37["segundos"], "s")
print("pre-registro      :", v37["pre_registro"]["sha256"][:16], "coincide:", v37["pre_registro"]["coincide_con_el_esperado"])
print()
print("cota de potencia (el numero que decide):")
for anot in ("md", "lg"):
    a1 = v37["fase_a"]["cota"][anot]["A1"]
    ic = a1["ic95"]["ic95"]
    print(f"  A1({anot}) = {a1['macro_f1_pooled']:.7f}  IC95 [{ic[0]:.6f}, {ic[1]:.6f}]"
          f"   azar 0.1666667   suelo estricto 0.0485421")
print()
d = v37["fase_d"]
print(f"transiciones: D = {d['D']:.7f} IC95 {[round(x,6) for x in d['D_ic95']]} excluye cero: {d['D_excluye_cero']}")
print(f"              R = {d['R']:.7f}  bigramas = {d['n_bigramas']} (esperado {d['n_bigramas_esperado']})")
print()
print("criterios V1..V6:")
for k, v in v37["fase_e"]["criterios"].items():
    if not (k.startswith("V") and isinstance(v, dict) and "cumple" in v):
        continue                     # hay claves auxiliares (lista de celdas, notas) sin `cumple`
    print(f"  {k}: cumple={str(v['cumple']):5} {str(v.get('resumen_valor'))[:88]}")
print()
print("gates:", {k: v.get("passed") for k, v in v37["fase_e"]["gates"].items()})
print("veredicto:", v37["fase_e"]["veredicto"])
print("rama     :", v37["fase_e"]["rama"])
```

    ===CELDA===
