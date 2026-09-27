"""Integra V28-03 (ambiguedad de etiqueta) en el notebook consolidado.

Anade la seccion 24 al final de `notebooks/10_final_pipeline.ipynb`. Las celdas de codigo leen
el JSON en tiempo de ejecucion y no llevan outputs embebidos, para que no puedan quedar
desincronizadas del informe.

Uso
---
    /home/carlos/venv/bin/python experiments/v28_04_integrar_seccion_ambiguedad.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "10_final_pipeline.ipynb"
INFORME = PROJECT_ROOT / "reports" / "v28_03_ambiguedad_etiquetas.json"

TITULO = """\
## 24. Fase v28: el margen de la region de error, es alcanzable? (2026-09-27)

La seccion 23 localizo **donde** falla V03. Esta seccion hace la pregunta que va despues, y que
es la unica que decide si seguir buscando informacion tiene sentido: si ese margen es **senal
reducible** que una fuente nueva podria capturar, o **ambiguedad irreducible** de las
etiquetas.
"""

PREGUNTA = """\
### El experimento natural ya existia

El proyecto tiene `grupos_fusionados` en `src/splits.py`, que une por union-find los
near-duplicados y los pseudo-autores. Los pliegues de V03 son `StratifiedGroupKFold` **sobre
esos grupos**, de modo que dos filas del mismo grupo nunca coexistieron en entrenamiento y
validacion. Ese grupo es ademas un experimento que los propios anotadores ya hicieron: si dos
textos casi identicos llevan etiquetas distintas, la discrepancia es de la anotacion, no del
modelo. No hace falta ningun modelo nuevo para medirlo.
"""

PODER = """\
### El limite, declarado antes de leer nada

| magnitud | valor |
|---|---:|
| filas del desarrollo | 1483 |
| grupos distintos | 1415 |
| grupos con 2 o mas filas | 105 |
| filas dentro de grupos con 2 o mas | 105 (**7,08 %**) |
| filas dentro de grupos **contradictorios** | 43 (**2,90 %**) |

**Con 43 filas informativas este experimento no puede responder la pregunta que se le hizo.**
Los doce tests de Fisher se calculan sobre bloques de 32 a 73 filas, donde la mitad de las
celdas no alcanza significacion aunque el efecto sea real. El limite no es de metodo, es de
datos: para medir ambiguedad haria falta un criterio de emparejamiento **mas laxo** que el
near-duplicate, que es ingenieria de datos, no modelado.
"""

TASA = """\
### Tasa de ambiguedad por region

La tasa se calcula **solo** sobre filas que viven en un grupo de 2 o mas, porque una fila sola
no puede contradecirse. junto a ella va el error de V03 en la region completa, para poder
comparar.

| region | filas | en grupos 2+ | HS amb. | TR amb. | AG amb. | HS err. V03 | TR err. V03 | AG err. V03 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| texto largo (>= 25 tokens) | 301 | 32 | 0.3438 | 0.1875 | 0.4688 | 0.2492 | 0.0963 | 0.2458 |
| texto no largo (< 25 tokens) | 1182 | 73 | 0.3562 | 0.2740 | 0.3562 | 0.2039 | 0.1142 | 0.1717 |
| con URL | 458 | 35 | 0.3714 | 0.1714 | 0.5429 | 0.2445 | 0.0852 | 0.2227 |
| sin URL | 1025 | 70 | 0.3429 | 0.2857 | 0.3143 | 0.1990 | 0.1220 | 0.1707 |

Las tasas de ambiguedad son **planas entre regiones**: 0.3438 frente a 0.3562 en HS, 0.4688
frente a 0.3562 en AG, 0.5429 frente a 0.3143 con URL. Con 32 a 73 filas por celda, esas
diferencias no son distinguibles del azar. **El diagnostico no encuentra ambiguedad
concentrada en la region de texto largo**, que es justo donde esta el margen.
"""

CONCENTRACION = """\
### Los errores de V03 en las filas ambiguas

| region | etiqueta | error si ambigua | error si no ambigua | razon | p Fisher |
|---|---|---:|---:|---:|---:|
| texto no largo | TR | 0.4000 | 0.0755 | 5,30 | 2,2e-03 |
| sin URL | TR | 0.3500 | 0.0800 | 4,38 | 9,5e-03 |
| texto no largo | AG | 0.3462 | 0.0851 | 4,07 | 9,2e-03 |
| sin URL | AG | 0.3182 | 0.0833 | 3,82 | 2,9e-02 |
| con URL | AG | 0.3684 | 0.1250 | 2,95 | 1,4e-01 |
| texto largo | AG | 0.3333 | 0.1176 | 2,83 | 2,1e-01 |
| sin URL | HS | 0.3333 | 0.1522 | 2,19 | 1,2e-01 |
| texto no largo | HS | 0.3077 | 0.1702 | 1,81 | 2,4e-01 |
| texto largo | HS | 0.1818 | 0.1429 | 1,27 | 1,0e+00 |
| con URL | HS | 0.1538 | 0.1818 | 0,85 | 1,0e+00 |
| texto largo | TR | 0.0000 | 0.0000 | n/d | 1,0e+00 |
| con URL | TR | 0.1667 | 0.0000 | n/d | 1,7e-01 |

Ordenado por p. `n/d` marca cociente sin contenido porque el error de las filas no ambiguas es
cero: un cociente sobre un denominador despreciable seria un numero sin significado, y se
declara en vez de imprimirse.

Cuatro de doce celdas alcanzan significancia, y las cuatro apuntan en la misma direccion: **donde
hay grupo contradictorio, V03 falla entre 3,8 y 5,3 veces mas**. Eso establece que la ambiguedad
de etiqueta es un multiplicador real del error.

Pero las cuatro celdas con significancia estan en las regiones **complementarias** (texto no
largo, sin URL), no en la region larga que concentra el margen. Con la potencia disponible, la
lectura correcta es: *la ambiguedad multiplica el error, y no se ha podido determinar si
explica el margen de la region larga*.
"""

COTA = """\
### Cota superior por voto mayoritario dentro del grupo

| etiqueta | cota de acierto |
|---|---:|
| HS | 0.9906 |
| TR | 0.9939 |
| AG | 0.9906 |
| macro | 0.9917 |

Una fila dentro de un grupo contradictorio no puede acertarse por ningun modelo que solo vea el
texto, asi que esta cifra acota el ruido de anotacion **visible en los grupos fusionados**.

**Es una cota, no un resultado alcanzable y no un modelo.** Usa la etiqueta, de modo que no
puede convertirse en submission ni en estimador de generalizacion.

Y su lectura es la mas util de la seccion: **0.9917 es practicamente 1,0**. Los grupos
fusionados no contienen la ambiguedad que limita a V03. El ceiling que importa **no esta ahi**,
y por eso V28-03 no puede decirnos si el techo de 1186 ejemplos es un techo real.
"""

CIERRE = """\
### Conclusion de V28-03

1. **Lo que SOSTIENE.** La ambiguedad de etiqueta es un multiplicador real del error de V03:
   entre 3,8 y 5,3 veces mas en filas cuyo grupo tiene anotaciones contradictorias, con
   significancia en 4 de 12 celdas y direccion uniforme.
2. **Lo que NO SOSTIENE.** Que el margen de la region de texto largo y con URL sea ambiguedad.
   Las tasas de ambiguedad son planas entre regiones y las celdas con significancia caen en las
   regiones complementarias. Con 43 filas informativas el experimento no tiene ese poder.
3. **Lo que REFUTA.** Que los grupos fusionados sean un proxy del ruido de anotacion. La cota
   de 0.9917 dice que alli no hay conflicto que valga la pena.
4. **Lo que NO SE PUEDE DECIRIR.** Si `N = 1186` es un techo. Nada de esta seccion lo toca.
5. **La fuente de informacion nueva que si queda.** No es un modelo: es un **criterio de
   emparejamiento mas laxo que el near-duplicate**, que daria muchas mas filas
   conflictivas y por tanto si podria medir la ambiguedad con potencia. Es trabajo de ingenieria de datos, no de
   modelado, y es la unica via que este diagnostico deja abierta.
"""

CODIGO_CARGA = """\
from pathlib import Path
import json

RAIZ = Path.cwd()
while not (RAIZ / "reports").exists() and RAIZ != RAIZ.parent:
    RAIZ = RAIZ.parent
INFORME = RAIZ / "reports" / "v28_03_ambiguedad_etiquetas.json"
datos = json.loads(INFORME.read_text(encoding="utf-8"))
print("familia:", datos["familia"], "| estado:", datos["estado"])
print("pregunta:", datos["pregunta"])
print()
t = datos["tamanos_de_grupo"]
print("PODER ESTADISTICO, que es el limite de todo lo que sigue")
print(f"{'magnitud':<42}{'valor':>12}")
for k, v in t.items():
    print(f"  {k:<40}{v:>12.4f}" if isinstance(v, float) else f"  {k:<40}{v:>12}")
"""

CODIGO_REGIONES = """\
print("TASA DE AMBIGUEDAD POR REGION, y el error de V03 para comparar")
print()
print(f"{'region':<28}{'filas':>7}{'en 2+':>7}{'HS amb':>9}{'TR amb':>9}{'AG amb':>9}"
      f"{'HS err':>9}{'TR err':>9}{'AG err':>9}")
for nombre, d in datos["desacuerdo_por_region"].items():
    pe = d["por_etiqueta"]
    amb = "".join(f"{pe[et]['tasa_de_ambiguedad']:>9.4f}" for et in ("HS", "TR", "AG"))
    err = "".join(f"{pe[et]['tasa_de_error_v03']:>9.4f}" for et in ("HS", "TR", "AG"))
    print(f"{nombre:<28}{d['n_filas']:>7}{d['n_filas_en_grupos_multimember']:>7}{amb}{err}")
print()
print("LECTURA: las tasas de ambiguedad son PLANAS entre regiones. El diagnostico no encuentra")
print("ambiguedad concentrada en la region de texto largo, que es donde esta el margen.")
"""

CODIGO_CONCENTRACION = """\
print("LOS ERRORES DE V03 EN LAS FILAS AMBIGUAS")
print()
con = datos["concentracion_del_error_en_filas_ambiguas"]
filas = []
for nombre, bloque in con.items():
    for etiqueta, d in bloque.items():
        filas.append((d["p_fisher"], nombre, etiqueta, d))
filas.sort()
def razon(d):
    r = d["razon_de_razones"]
    return "n/d" if r is None else f"{r:.2f}"

print(f"{'region':<28}{'etiqueta':>9}{'err ambigua':>13}{'err no ambigua':>16}{'razon':>8}{'p':>11}")
for _, nombre, etiqueta, d in filas:
    print(f"{nombre:<28}{etiqueta:>9}{d['error_si_ambigua']:>13.4f}"
          f"{d['error_si_no_ambigua']:>16.4f}{razon(d):>8}{d['p_fisher']:>11.2e}")
sig = [f for f in filas if f[0] < 0.05]
print()
print(f"celdas con p < 0.05: {len(sig)} de {len(filas)}")
for _, nombre, etiqueta, d in sig:
    print(f"  {nombre:<28}{etiqueta}  razon {d['razon_de_razones']:.2f}")
print()
print("LECTURA: donde hay grupo contradictorio, V03 falla 3.8 a 5.3 veces mas. La ambiguedad es")
print("un multiplicador real del error. Pero las 4 celdas con significancia estan en las")
print("regiones COMPLEMENTARIAS, no en la region larga que concentra el margen.")
"""

CODIGO_CIERRE = """\
c = datos["cota_por_voto_mayoritario"]
g = datos["gastos"]
print("COTA SUPERIOR POR VOTO MAYORITARIO DENTRO DEL GRUPO")
print("  (es una COTA, usa la etiqueta, no es un modelo ni un resultado alcanzable)")
print()
for et in ("HS", "TR", "AG"):
    print(f"  {et}  {c[et]:.4f}")
print(f"  macro  {c['macro']:.4f}")
print()
print("LECTURA: 0.9917 es practicamente 1.0. Los grupos fusionados NO contienen la ambiguedad")
print("que limita a V03, asi que el ceiling que importa no esta ahi. Por eso V28-03 no puede")
print("decir si N = 1186 es un techo real.")
print()
print("CONCLUSION")
print("  SOSTIENE:    la ambiguedad multiplica el error de V03 entre 3.8 y 5.3 veces.")
print("  NO SOSTIENE: que el margen de la region larga sea ambiguedad; las tasas son planas.")
print("  REFUTA:      que los grupos fusionados sean proxy del ruido de anotacion (cota 0.9917).")
print("  NO DICE NADA: sobre si N = 1186 es un techo.")
print("  ABRE:        un criterio de emparejamiento mas laxo que el near-duplicate, que es la")
print("               unica via que este diagnostico deja abierta, y es ingenieria de datos.")
print()
print(f"Coste: {g['segundos']} s, memoria {g['memoria_pico_mb']:.0f} MB, "
      f"{g['modelos_entrenados']} modelos, {g['submissions']} submissions.")
"""


def markdown(texto: str) -> dict[str, Any]:
    """Construye una celda markdown.

    Args:
        texto: fuente de la celda.

    Returns:
        La celda como diccionario de notebook.
    """
    return {"cell_type": "markdown", "metadata": {}, "source": texto.splitlines(keepends=True)}


def codigo(texto: str) -> dict[str, Any]:
    """Construye una celda de codigo sin outputs.

    Args:
        texto: fuente de la celda.

    Returns:
        La celda como diccionario de notebook.
    """
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": texto.splitlines(keepends=True),
    }


def main() -> None:
    """Anade las celdas de V28-03 al notebook consolidado.

    Returns:
        None.
    """
    if not INFORME.exists():
        raise SystemExit(f"PARADA: no existe {INFORME}, ejecuta antes v28_03")
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    antes = len(notebook["cells"])
    marcas = " ".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown"
    )
    if "el margen de la region de error" in marcas:
        print("  la seccion 24 ya existe, no se duplica")
        return
    notebook["cells"].extend(
        [
            markdown(TITULO),
            markdown(PREGUNTA),
            markdown(PODER),
            markdown(TASA),
            markdown(CONCENTRACION),
            markdown(COTA),
            markdown(CIERRE),
            codigo(CODIGO_CARGA),
            codigo(CODIGO_REGIONES),
            codigo(CODIGO_CONCENTRACION),
            codigo(CODIGO_CIERRE),
        ]
    )
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"  notebook: {antes} -> {len(notebook['cells'])} celdas")
    print("  seccion 24 integrada en el notebook consolidado")


if __name__ == "__main__":
    main()
