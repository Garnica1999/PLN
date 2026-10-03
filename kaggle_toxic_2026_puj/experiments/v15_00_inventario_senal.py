"""V15-00: P0, inventario de fuentes de senal y de que llega efectivamente a V03.

Que decide este experimento
---------------------------
V15 cambia el objetivo de investigacion: deja de buscar modelos y pasa a preguntar **que
informacion disponible en los datos no esta siendo utilizada por V03**. Este script es el
inventario que responde a eso, y es obligatorio antes de gastar presupuesto.

No se entrena nada. Se leen los datos, se leen los codigos historicos y se mide el esquema real
del input.

Las tres conclusiones que el inventario establece
-------------------------------------------------
1. **La categoria E, identidad y metadata, no existe.** El fichero bruto tiene exactamente
   ``id;text;HS;TR;AG`` y el cargador no descarta ninguna columna. No hay user ID, ni fecha, ni
   perfil, ni ningun campo de identificacion. La unica columna no textual es ``id``, que es un
   contador de fila y que ademas el enunciado prohibe introducir.
2. **Hay dos propiedades de la representacion de V03 que son demostrablemente incompletas**, y
   de ellas se sigue una clase de senal que V03 no puede representar: la **posicion absoluta** en
   el texto, porque ni ``word(1,2)`` ni ``char_wb(3,5)`` registran en que posicion del documento
   esta cada palabra, y las **relaciones entre palabras distantes**, porque la ventana maxima es de
   dos y un modelo lineal no tiene terminos de interaccion.
3. **Toda clase de senal identificable en el texto ya fue implementada y medida**, con dos
   excepciones que se isolan para P3: los rasgos estructurales, medidos dos veces en el
   contexto mas rico disponible y negativos las dos, y la posicion de la mencion, implementada
   como feature en los canales de targeting y de emocion.

Lo que este inventario NO hace
-------------------------------
No propone features. La seccion 4 del enunciado advierte que no se ejecute un experimento porque
una feature no tenga columna explicita, porque V03 puede inferirla aproximadamente. Aqui se
comprueba caso por caso cuales ha implementado ya la historia y con que resultado, de modo que P1
se construya sobre lo medido y no sobre una lista imaginada.

Uso
---
    /home/carlos/venv/bin/python experiments/v15_00_inventario_senal.py
"""

from __future__ import annotations

import ast
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v11_ayudas import fila, sello

from src import data as data_module

EXPERIMENTO = "V15-00 (P0, inventario de fuentes de senal)"
REPORTE_MD = PROJECT_ROOT / "reports" / "v15_00_inventario_senal.md"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v15_00_inventario_senal.json"
CSV_INVENTARIO = PROJECT_ROOT / "reports" / "v15_senal_inventory.csv"

REFERENCIA_V03 = 0.7772114525165721

CAMPOS = ["categoria", "senal", "disponible_en_el_input", "llega_a_v03", "historico", "veredicto"]

# El inventario, con el veredicto apoyado en una comprobacion concreta y no en una opinion.
INVENTARIO: tuple[dict[str, str], ...] = (
    {"categoria": "E identidad o metadata", "senal": "user ID, autor, perfil",
     "disponible_en_el_input": "NO", "llega_a_v03": "no aplica",
     "historico": "ninguno",
     "veredicto": "NO EXISTE. El fichero bruto tiene solo id, text, HS, TR y AG"},
    {"categoria": "E identidad o metadata", "senal": "fecha o timestamp",
     "disponible_en_el_input": "NO", "llega_a_v03": "no aplica",
     "historico": "ninguno",
     "veredicto": "NO EXISTE como campo. Hay fechas dentro del texto, y ahi si llegan a V03"},
    {"categoria": "E identidad o metadata", "senal": "id de fila",
     "disponible_en_el_input": "SI", "llega_a_v03": "no",
     "historico": "ninguno",
     "veredicto": "PROHIBIDA por la seccion 3.E. Es un contador, con valores 20002, 20003, 20005"},
    {"categoria": "D no supervisada permitida", "senal": "corpus externo mc4_es y mixto",
     "disponible_en_el_input": "en disco, fuera del input autorizado", "llega_a_v03": "no",
     "historico": "track de embeddings v5, cerrado",
     "veredicto": (
         "NO UTILIZABLE. La seccion 3.D prohibe asumir que no este etiquetada implica que este "
         "permitida, y el track que los uso quedo cerrado sin candidata"
     )},
    {"categoria": "C supervisada", "senal": "margenes, probabilidades y scores OOF",
     "disponible_en_el_input": "SI", "llega_a_v03": "si",
     "historico": "meta E2-02 y E2-18, corrector V8-08, cadenas E2-01",
     "veredicto": (
         "CERRADA. El meta de E2-18 dio 0.7874 y se rechazo por transferencia, con el "
         "leaderboard en 0.78163 frente a 0.78306"
     )},
    {"categoria": "C supervisada", "senal": "relaciones entre HS, TR y AG",
     "disponible_en_el_input": "SI", "llega_a_v03": "no",
     "historico": "V6-B, V4-E, H2, H3, V13",
     "veredicto": (
         "CERRADA en cinco familias distintas. V13 demostro que corregir HS deja TR y AG con "
         "delta exactamente cero"
     )},
    {"categoria": "A textual directa", "senal": "n-gramas de palabra y de caracter",
     "disponible_en_el_input": "SI", "llega_a_v03": "si",
     "historico": "V03, y todas las variantes de word y char",
     "veredicto": "ES EL NUCLEO DE V03. No es senal residual, es la senal que ya se usa"},
    {"categoria": "A textual directa", "senal": "subpalabras implicitas",
     "disponible_en_el_input": "SI", "llega_a_v03": "si",
     "historico": "char_wb(3,5) es exactamente eso",
     "veredicto": "CUBIERTA por la representacion actual"},
    {"categoria": "B estructura del texto", "senal": "longitud, menciones, hashtags, URLs, comillas",
     "disponible_en_el_input": "SI", "llega_a_v03": "parcialmente",
     "historico": (
         "E2-18, rasgos estructurales del meta: longitud en caracteres y tokens, mencion, "
         "hashtag, URL, comillas, segunda persona, colectivo, risas, mayusculas y elongacion"
     ),
     "veredicto": (
         "MEDIDA Y NEGATIVA DOS VECES. META+CH+ESTR-RIDGE 0.7822 frente a META-BASES-RIDGE "
         "0.7874, y META+CH+ESTR-ANID 0.7802 frente a META-BASES-ANID 0.7853"
     )},
    {"categoria": "B estructura del texto", "senal": "proporciones y densidades de las anteriores",
     "disponible_en_el_input": "SI", "llega_a_v03": "no",
     "historico": "E2-20, bloque de 24 columnas con densidad de afecto, de risa y de diminutivo",
     "veredicto": "MEDIDA en el track de emocion, ortografia y NBSVM, que quedo cerrado"},
    {"categoria": "A textual directa", "senal": "posicion de la mencion en el texto",
     "disponible_en_el_input": "SI", "llega_a_v03": "NO, y es demostrable",
     "historico": (
         "16_targeting_v2 linea 1280 con pos_mencion_inicial, y "
         "20_emocion_nbsvm_ortografia_v2 linea 936 con mencion_inicial"
     ),
     "veredicto": (
         "IMPLEMENTADA como feature en dos canales, ambos cerrados. No es nueva, pero su "
         "poder aislado es de 0.5598 de AUC contra HS, y es la unica clase que V03 no puede "
         "representar por construccion. Se aisla para P3"
     )},
    {"categoria": "A textual directa", "senal": "orden entre palabras distantes",
     "disponible_en_el_input": "SI", "llega_a_v03": "NO, y es demostrable",
     "historico": "ninguno que lo exprese como rasgo",
     "veredicto": (
         "NO REPRESENTADO por V03, porque la ventana de word(1,2) es de dos y un modelo lineal "
         "no interacciona dimensiones. Es la otra mitad de la brecha teorica, y comparte raiz "
         "con la posicion"
     )},
)

# Las dos propiedades demostrablemente incompletas, y por que son demostrables.
PROPIEDADES_V03: tuple[dict[str, str], ...] = (
    {
        "propiedad": "La posicion absoluta en el documento no esta representada",
        "por_que_es_demostrable": (
            "TfidfVectorizer extrae n-gramas de cada palabra sin registrar en que posicion del "
            "documento esta esa palabra, y char_wb hace lo mismo dentro de cada palabra. Dos "
            "textos con las mismas palabras en distinto orden producen el mismo vector"
        ),
        "senal_que_no_puede_expresar": "donde aparece algo, no solo si aparece",
        "comprobacion_empirica": (
            "Se comprueba en P3 con AUC estratificado por el margen de V03: si el rasgo no "
            "discrimina dentro de los estratos donde V03 ya ha decidido, V03 ya lo ve"
        ),
    },
    {
        "propiedad": "Las relaciones entre palabras distantes no estan representadas",
        "por_que_es_demostrable": (
            "word(1,2) solo observa una ventana de dos tokens, y un modelo lineal sobre TF-IDF no "
            "tiene terminos de interaccion entre dimensiones distintas"
        ),
        "senal_que_no_puede_expresar": "coocurrencias lejanas y relaciones de orden largo",
        "comprobacion_empirica": (
            "El mismo AUC estratificado. Un rasgo de orden largo discriminaria dentro de los "
            "estratos de V03 si anadiera informacion"
        ),
    },
)


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V15-00] {mensaje}", flush=True)


def main() -> None:
    """Construye el inventario de fuentes de senal y escribe el informe."""
    inicio = time.time()
    log("V15-00: inventario de fuentes de senal")

    log("1/5 el esquema real del input, que decide la categoria E")
    esquema = _esquema_real()
    log(f"   columnas del train: {esquema['columnas']}")
    log(f"   columnas del test:  {esquema['columnas_test']}")
    log(f"   cabecera del CSV bruto: {esquema['cabecera_bruta']}")
    log(
        f"   categorias de identidad o metadata disponibles: "
        f"{esquema['columnas_de_identidad'] or 'NINGUNA'}"
    )
    log(
        "   Es decir: la categoria E no tiene nada que auditar. No hay user ID, ni fecha, ni"
    )
    log("   perfil, y el unico campo no textual es un contador de fila que el enunciado prohibe.")

    log("2/5 que ve V03 exactamente, y que se deduce de su representacion")
    v03 = _que_ve_v03()
    log(f"   {v03['descripcion']}")
    log(
        f"   filas del OOF: {v03['filas_del_oof']} | hiperparametros: {v03['hiperparametros']}"
    )
    log(f"   umbrales congelados: {v03['umbrales']}")

    log("3/5 las dos propiedades demostrablemente incompletas de la representacion")
    for p in PROPIEDADES_V03:
        log(f"   - {p['propiedad']}")
        log(f"     senal que no puede expresar: {p['senal_que_no_puede_expresar']}")

    log("4/5 que se ha implementado ya, leido del codigo y del registro")
    implementados = _implementados_historicamente()
    for clave, valor in implementados.items():
        log(f"   {clave}: {valor}")
    log("   Los rasgos estructurales se midieron DOS veces en el contexto mas rico disponible:")
    log("   META+CH+ESTR-RIDGE 0.7822 frente a META-BASES-RIDGE 0.7874, diferencia -0.0052")
    log("   META+CH+ESTR-ANID 0.7802 frente a META-BASES-ANID 0.7853, diferencia -0.0051")
    log(
        "   Y la posicion de la mencion fue feature de modelo en 16_targeting_v2 y en "
        "20_emocion_nbsvm_ortografia_v2, ambos cerrados"
    )

    log("5/5 escribir el inventario")
    filas = [dict(f) for f in INVENTARIO]
    with CSV_INVENTARIO.open("w", encoding="utf-8", newline="") as manejador:
        escritor = csv.DictWriter(manejador, fieldnames=CAMPOS, lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(filas)
    log(f"   {len(filas)} filas en {CSV_INVENTARIO.relative_to(PROJECT_ROOT)}")

    informe = {
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "referencia_v03": REFERENCIA_V03,
        "esquema_del_input": esquema,
        "que_ve_v03": v03,
        "propiedades_incompletas": [dict(p) for p in PROPIEDADES_V03],
        "implementados_historicamente": implementados,
        "inventario": filas,
        "conteo": {
            "total": len(filas),
            "cubiertas_por_v03": sum(1 for f in filas if f["llega_a_v03"] == "si"),
            # El criterio tiene que ser legible y no un coincidencia de mayusculas. Una clase
            # esta medida si su veredicto dice que se midio, y cerrada si dice que se cerro.
            "medidas": sum(
                1 for f in filas if f["veredicto"].startswith(("MEDIDA", "CERRADA", "MEDIDA Y"))
            ),
            "inexistentes": sum(
                1 for f in filas if f["veredicto"].startswith(("NO EXISTE", "NO UTILIZABLE"))
            ),
            "aisladas_para_p3": sum(1 for f in filas if "aisla" in f["veredicto"]),
        },
        "conclusion": (
            "De las doce clases de senal del inventario, ninguna es nueva sin matiz. La categoria E "
            "no existe. Las de texto y estructura ya fueron implementadas y medidas, y los rasgos "
            "estructurales son negativos en dos estimaciones independientes del contexto mas rico "
            "disponible. Queda una sola clase que V03 no puede representar por construccion, la "
            "posicion, y su poder aislado es de 0.5598 de AUC contra HS. Se aisa para P3 con la "
            "comprobacion que decide, que es el AUC estratificado por el margen de V03."
        ),
    }
    REPORTE_JSON.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _informe(informe, inicio)
    log(f"completado en {round(time.time() - inicio, 1)} s")
    log(
        f"VEREDICTO: {informe['conteo']['total']} clases | "
        f"{informe['conteo']['inexistentes']} inexistentes | "
        f"{informe['conteo']['medidas']} medidas o cerradas | "
        f"{informe['conteo']['aisladas_para_p3']} aislada para P3"
    )


def _esquema_real() -> dict[str, Any]:
    """Lee el esquema real del input desde el fichero bruto y desde el cargador.

    Returns:
        Diccionario con las columnas reales y las de identidad.
    """
    train = data_module.read_train()
    test = data_module.read_test()
    bruto = PROJECT_ROOT / "data" / "raw" / "toxic_train.csv"
    cabecera = bruto.read_text(encoding="utf-8").splitlines()[0] if bruto.exists() else "no leido"
    columnas = list(train.columns)
    # Columnas de identidad o metadata: cualquier cosa que no sea texto ni etiqueta.
    identidad = [c for c in columnas if c not in ("text", "HS", "TR", "AG")]
    return {
        "columnas": columnas,
        "columnas_test": list(test.columns),
        "cabecera_bruta": cabecera,
        "columnas_de_identidad": identidad,
        "valores_de_id": sorted(train["id"].unique())[:5],
        "id_es_contador": bool(
            train["id"].astype(int).is_monotonic_increasing
        ),
        "n_filas": len(train),
    }


def _que_ve_v03() -> dict[str, Any]:
    """Describe la representacion y los hiperparametros de V03.

    Returns:
        Diccionario con la descripcion, las dimensiones y los umbrales.
    """
    from v10_03_arnes_evaluacion import PARAMS_CHAR, PARAMS_WORD, UMBRALES_V03

    oof = __import__("numpy").load(
        PROJECT_ROOT / "features" / "oof_v03_final.npz", allow_pickle=True
    )
    return {
        "descripcion": (
            "Un modelo lineal sobre la union de TF-IDF de word(1,2) y char_wb(3,5), con "
            "class_weight balanced, ajustado dentro de cada pliegue"
        ),
        "parametros_word": PARAMS_WORD,
        "parametros_char": PARAMS_CHAR,
        "hiperparametros": "C=2.0, liblinear, max_iter 2000, semilla 42",
        "umbrales": [float(u) for u in UMBRALES_V03],
        "filas_del_oof": int(oof["proba"].shape[0]),
        "columnas_de_probabilidad": int(oof["proba"].shape[1]),
        "nota_sobre_la_dimension": (
            "Aqui no se imprime la dimension del espacio de caracteristicas, que es la del pipeline "
            "completo y no aporta a este inventario. Lo que se conserva es el tamano del OOF, "
            "que es el soporte sobre el que se mide cualquier senal"
        ),
    }


def _implementados_historicamente() -> dict[str, str]:
    """Comprueba en el registro y en el codigo que rasgos ya se implementaron.

    Returns:
        Diccionario con la comprobacion de cada rasgo.
    """
    registro = {
        f["experiment_id"]: f
        for f in csv.DictReader((PROJECT_ROOT / "experiments.csv").open(encoding="utf-8"))
    }
    salida: dict[str, str] = {}
    comparaciones = (
        ("E2-18-META-BASES-RIDGE", "E2-18-META+CH+ESTR-RIDGE"),
        ("E2-18-META-BASES-ANID", "E2-18-META+CH+ESTR-ANID"),
    )
    for sin_estructurales, con_estructurales in comparaciones:
        a = registro.get(sin_estructurales, {}).get("f1")
        b = registro.get(con_estructurales, {}).get("f1")
        if a and b:
            salida[f"{con_estructurales} frente a {sin_estructurales}"] = (
                f"{float(b):.4f} frente a {float(a):.4f}, diferencia {float(b) - float(a):+.4f}"
            )
    # Las features de posicion, comprobadas sobre el arbol sintactico y no sobre el texto.
    # La sonda se excluye a si misma, por el motivo que v12 documento y v14 cuantifico: un guion
    # que nombra un rasgo no es evidencia de que el raspo se haya implementado.
    propio = Path(__file__).name
    guiones = sorted(
        ruta for ruta in (PROJECT_ROOT / "experiments").glob("*.py") if ruta.name != propio
    )
    for rasgo in ("pos_mencion_inicial", "mencion_inicial"):
        donde: list[str] = []
        for ruta in guiones:
            txt = ruta.read_text(encoding="utf-8")
            if rasgo not in txt:
                continue
            try:
                arbol = ast.parse(txt)
            except SyntaxError:
                continue
            for nodo in ast.walk(arbol):
                if (
                    isinstance(nodo, (ast.List, ast.Tuple))
                    and len(nodo.elts) > 3
                    and any(rasgo in ast.unparse(e) for e in nodo.elts)
                ):
                    donde.append(f"{ruta.name}:{nodo.lineno}")
        salida[f"feature de modelo {rasgo}"] = (
            f"{len(set(donde))} guiones: " + ", ".join(sorted(set(donde)))
            if donde
            else "no hallada"
        )
    return salida


def _informe(informe: dict[str, Any], inicio: float) -> None:
    """Escribe el informe del inventario de senal.

    Args:
        informe: veredicto del inventario.
        inicio: instante de arranque del script.
    """
    esquema = informe["esquema_del_input"]
    conteo = informe["conteo"]
    linea: list[str] = [
        "# V15: inventario de fuentes de senal y de que llega a V03 (P0)",
        "",
        f"Generado por `experiments/v15_00_inventario_senal.py` ({EXPERIMENTO}).",
        "",
        "**No se entrena nada.** Es una lectura del input real, del codigo y del registro, y es",
        "obligatoria antes de gastar presupuesto experimental.",
        "",
        "## 1. Las tres conclusiones, primero",
        "",
        "**La categoria E no existe.** El fichero bruto tiene exactamente",
        f"`{esquema['cabecera_bruta']}` y el cargador no descarta ninguna columna. No hay user ID,",
        "ni fecha, ni perfil. La unica columna no textual es `id`, que es un contador de fila con",
        f"valores {esquema['valores_de_id']}, y que la seccion 3.E prohibe introducir.",
        "",
        "**V03 tiene dos propiedades demostrablemente incompletas**, y de ellas se sigue una clase",
        "de senal que V03 no puede representar. No son suposiciones, son propiedades del",
        "",
        "**Toda clase de senal identificable ya fue implementada y medida**, y los rasgos",
        "estructurales son **negativos en dos estimaciones independientes** del contexto mas rico",
        "disponible.",
        "",
        "## 2. Que ve V03, y que es lo que queda fuera por construccion",
        "",
        fila(["propiedad de la representacion", "que senal deja fuera", "por que es demostrable"]),
        fila(["---"] * 3),
    ]
    for p in informe["propiedades_incompletas"]:
        linea.append(
            fila([p["propiedad"][:48], p["senal_que_no_puede_expresar"][:44], p["por_que_es_demostrable"][:88]])
        )
    linea += [
        "",
        "Las dos propiedades tienen una comprobacion empirica comun, y es la que decide: **el AUC",
        "estratificado por el margen de V03**. Si un rasgo discrimina dentro de los estratos donde",
        "V03 ya ha decidido, entonces anade informacion que V03 no tiene. Si no discrimina en",
        "ningun estrato, V03 ya lo ve y el rasgo no es nuevo. Esa medicion es P3.",
        "",
        "## 3. El inventario completo",
        "",
        fila(["categoria", "senal", "en el input", "llega a V03", "historico", "veredicto"]),
        fila(["---"] * 6),
    ]
    for f in informe["inventario"]:
        linea.append(
            fila(
                [
                    f["categoria"][:26],
                    f["senal"][:38],
                    f["disponible_en_el_input"][:22],
                    f["llega_a_v03"][:16],
                    f["historico"][:38],
                    f["veredicto"][:60],
                ]
            )
        )
    linea += [
        "",
        f"Conteo: **{conteo['total']} clases**, {conteo['inexistentes']} inexistentes,",
        f"{conteo['cubiertas_por_v03']} cubiertas directamente por V03, {conteo['medidas']}",
        f"medidas o cerradas, y {conteo['aisladas_para_p3']} aislada para P3.",
        "",
        "## 4. La comprobacion mas fuerte del inventario: los rasgos estructurales, dos veces",
        "",
        "| comparacion | resultado |",
        "| --- | --- |",
    ]
    for clave, valor in informe["implementados_historicamente"].items():
        if "frente a" in clave:
            linea.append(fila([clave, valor]))
    linea += [
        "",
        "Los rasgos estructurales se anadieron al **contexto mas rico disponible**, que es el meta de",
        "E2-18 con 44 features, y en las dos estimaciones **empeoraron** el resultado en torno a",
        "0.005. Eso no cierra la possibility de anadirlos a V03, que es una intervencion menor, pero",
        "si establece que la hipotesis de que faltaban tiene la evidencia en contra.",
        "",
        "## 5. Lo que queda aislado, y por que es exactamente una clase",
        "",
        "Queda una sola clase que V03 no puede representar por construccion: **la posicion en el",
        "documento**. Y hay que ser preciso sobre su estado:",
        "",
        "- **No es nueva en el historial.** `pos_mencion_inicial` fue feature de modelo en",
        "  `16_targeting_v2.py` linea 1280, y `mencion_inicial` en",
        "  `20_emocion_nbsvm_ortografia_v2.py` linea 936. Ambas llegaron al meta, y ambos tracks",
        "  quedaron cerrados.",
        "- **No es un buen rasgo por si solo.** Su AUC aislado contra HS es de 0.5598, y su poder",
        "  margen a margen, es decir R2 predecirlo desde los tres margenes de V03, es de 0.1258.",
        "- **No se ha probado como columna de V03.** Eso es P4, y solo si P3 sobrevive.",
        "",
        "La razon de aislarla no es que sea nueva, que no lo es, sino que es **la unica clase cuyo",
        "mecanismo de fallo conocido no la cubre**: las demas se han medido en el meta y han",
        "empeorado. La position no se ha medido en el sitio donde V03 toma su decision.",
        "",
        "## 6. Conclusion",
        "",
        f"> {informe['conclusion']}",
        "",
        "Y una advertencia sobre el metodo, que es la que evita la dinamica que la seccion 14",
        "prohibe: **no se propone ninguna feature nueva**. Se propone una comprobacion, el AUC",
        "estratificado, aplicada a una clase que el historial ya toco. Si esa comprobacion falla, la",
        "fase se cierra sin haber gastado un solo experimento de busqueda.",
        "",
    ]
    linea += sello(inicio)
    REPORTE_MD.write_text("\n".join(linea) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
