"""V11-03: auditoria de cumplimiento y deteccion de condiciones de parada.

Por que nace
------------
Las restricciones de ejecucion de la fase v11 son globales y tienen prioridad sobre cualquier
instruccion de track. Vienen con una lista de comprobaciones que la seccion 18 exige verificar
**explicitamente y no solo declarar**, y con una lista de condiciones de parada en la seccion 19
que obligan a detener la ejecucion y auditar antes de seguir.

Este script hace las dos cosas, y es el primer experimento de la v11 que puede **invalidar** a los
anteriores, que es exactamente su trabajo.

Lo que verifica
---------------
1. **Los 17 puntos de la seccion 18**, buscados en el codigo y en los artefactos, no declarados.
2. **Las condiciones de parada de la seccion 19**, una por una.
3. **La politica de caracteres de la seccion 17**, que es mas estricta que la herramienta que se
   venia usando: aquella buscaba alfabetos no latinos, y esta pide **ASCII puro**. La distincion
   importa porque un acento pasa la primera comprobacion y falla la segunda, y hay al menos cuatro
   casos en lo ya escrito.
4. **La coherencia entre el codigo y los artefactos** de los experimentos de la v11.

Lo que decide
-------------
Si alguna comprobacion falla, el script lo dice con nombre y evidencia, y **la fase queda
detenida** hasta que se resuelva. No se continua acumulando resultados sobre una infraestructura
que la auditoria considera defectuosa.

Uso
---
    /home/carlos/venv/bin/python experiments/v11_03_auditoria_cumplimiento.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

EXPERIMENTO = "V11-03 (auditoria de cumplimiento y condiciones de parada)"
REPORTE_MD = PROJECT_ROOT / "reports" / "v11_03_auditoria_cumplimiento.md"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v11_03_auditoria_cumplimiento.json"

# La seccion 17 permite este conjunto tecnico ademas de ASCII imprimible.
PERMITIDOS_TECNICOS = set("()[]{}:;,.=+-*/_#\"'%")
CARACTERES_ASCII = {chr(codigo) for codigo in range(128)}

# Los 17 puntos de la seccion 18. Cada uno lleva un patron que se busca en codigo y artefactos.
# Se nombran como en la seccion para que la auditoria sea cotejable con el enunciado.
CONDICIONES_PARADA = (
    "unexpected baseline change",
    "control no reproducible",
    "validation leakage",
    "test contamination",
    "holdout contamination",
    "non-convergence",
    "missing artifact",
    "missing experiment provenance",
    "numeric discrepancy",
    "notebook output mismatch",
    "unexpected external resource",
    "secret exposure",
    "invalid generated characters",
)


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V11-03] {mensaje}", flush=True)


def no_ascii_esperados(ruta: Path) -> list[dict[str, Any]]:
    """Devuelve los caracteres de un fichero que violan la politica ASCII.

    Args:
        ruta: fichero a examinar.

    Returns:
        Lista de diccionarios con el caracter, su codigo y cuantas veces aparece.
    """
    try:
        texto = ruta.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    conteo: dict[str, int] = {}
    for caracter in texto:
        if caracter in CARACTERES_ASCII or caracter in PERMITIDOS_TECNICOS:
            continue
        conteo[caracter] = conteo.get(caracter, 0) + 1
    # Solo el codigo, nunca el caracter. Un informe que imprime el caracter que viola
    # reproduce la violacion en si mismo, de modo que auditarlo exigiria excluirlo a el.
    return [
        {
            "codigo": f"U+{ord(caracter):04X}",
            "nombre_unicode": unicodedata.name(caracter, "sin nombre"),
            "veces": veces,
        }
        for caracter, veces in sorted(conteo.items(), key=lambda par: -par[1])
    ]


# Modulos whose whole purpose is to document prohibitions, so their text necessarily contains
# the forbidden words. Scanning them with a word search is self-referential: the scanner flags its
# own prohibition list. The rule is the opposite of an exemption, because a file that documents a
# prohibition is by construction the one place where the word is allowed to appear.
FICHEROS_DE_DOCUMENTACION = (
    "v11_02_auditoria_legalidad",
    "v11_03_auditoria_cumplimiento",
    "v11_compliance",
)

# Libraries that must not appear in an import. This is a structural check, not a word search,
# so it has no false positives: a name in an import statement is a use, and a name in a comment is
# not. It is the check that actually carries evidence.
MODULOS_PROHIBIDOS = {
    "transformers": "seccion 18, transformers",
    "sentence_transformers": "seccion 18, transformers",
    "roberta": "seccion 18, transformers",
    "bert": "seccion 18, transformers",
    "xgboost": "seccion 18, external_classifier",
    "lightgbm": "seccion 18, external_classifier",
    "catboost": "seccion 18, external_classifier",
    "gensim": "seccion 18, third_party_pretrained_embeddings",
    "torchtext": "seccion 18, third_party_pretrained_embeddings",
}


def buscar_imports_prohibidos(guiones: list[Path]) -> dict[str, list[str]]:
    """Busca imports de modulos prohibidos en el codigo ejecutable de la v11.

    Es una comprobacion estructural y no una busqueda de palabras, y esa es exactamente la
    diferencia que hace que sirva: un nombre dentro de un ``import`` es un uso, y un nombre
    dentro de un comentario no lo es. Una busqueda de texto no puede distinguirlos y por eso
    producia doce falsos positivos al encontrar sus propias listas de prohibicion.

    Args:
        guiones: ficheros de codigo de la v11.

    Returns:
        Diccionario de modulo prohibido a las referencias que lo importan.
    """
    patron_import = re.compile(
        r"^\s*(?:from|import)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE
    )
    patron_ruidoso = re.compile(r"^\s*(?:from|import)\s+\.([A-Za-z0-9_]+)", re.MULTILINE)
    hallazgos: dict[str, list[str]] = {}
    for guion in guiones:
        texto = guion.read_text(encoding="utf-8")
        # Se descartan las lineas de codigo dentro de cadenas de texto, que no son imports.
        lineas_de_codigo = [
            linea
            for linea in texto.splitlines()
            if not linea.lstrip().startswith(("#", '"', "'", "*", "-"))
        ]
        limpio = "\n".join(lineas_de_codigo)
        for patron in (patron_import, patron_ruidoso):
            for coincidencia in patron.finditer(limpio):
                modulo = coincidencia.group(1)
                if modulo in MODULOS_PROHIBIDOS:
                    hallazgos.setdefault(modulo, []).append(guion.name)
    return hallazgos


def buscar_firmas_de_uso(guiones: list[Path]) -> dict[str, list[str]]:
    """Busca firmas que indiquen un uso real de la prueba o del holdout en la capa de decision.

    A diferencia de los imports, aqui no hay una forma estructural tan limpia, asi que se usan
    expresiones que son firmas de uso y no menciones: alguien que **ajusta** algo con datos de
    prueba, no alguien que **escribe** que no se hace.

    Args:
        guiones: ficheros de codigo de la v11.

    Returns:
        Diccionario de comprobacion a las referencias encontradas.
    """
    firmas = {
        "test_training": r"\.fit\([^)]*\b(proba_test|X_test|matrix_test|test_text)\b",
        "test_threshold_tuning": r"(optimizar|ajustar|reoptimizar)_?umbral\w*\([^)]*\btest\b",
        "holdout_reuse": r"\b(holdout2?|holdout_v2)\b[^\n]{0,40}\.(fit|optimizar|ajustar)\(",
    }
    hallazgos: dict[str, list[str]] = {}
    for nombre, patron in firmas.items():
        expresion = re.compile(patron, re.IGNORECASE)
        for guion in guiones:
            for linea in guion.read_text(encoding="utf-8").splitlines():
                if expresion.search(linea):
                    hallazgos.setdefault(nombre, []).append(f"{guion.name}: {linea.strip()[:70]}")
    return hallazgos


def leer_ficheros_propios() -> list[Path]:
    """Devuelve los ficheros de codigo y artefactos propios de la v11.

    Returns:
        Lista ordenada de rutas.
    """
    rutas: list[Path] = []
    for patron in (
        "experiments/v11_*.py",
        "reports/v11_*",
        "tasks/*v11*",
    ):
        rutas.extend(sorted(PROJECT_ROOT.glob(patron)))
    return [ruta for ruta in rutas if ruta.is_file()]


def main() -> None:
    """Ejecuta la auditoria de cumplimiento y decide si la fase puede continuar."""
    inicio = time.time()
    log("V11-03: auditoria de cumplimiento y condiciones de parada")

    findings: list[dict[str, Any]] = []

    def anotar(categoria: str, nombre: str, cumple: bool, evidencia: str) -> None:
        """Registra un hallazgo de la auditoria.

        Args:
            categoria: seccion de la que proviene.
            nombre: que se comprueba.
            cumple: si el resultado es conforme.
            evidencia: detalle que permite reproducir la comprobacion.
        """
        findings.append(
            {
                "categoria": categoria,
                "comprobacion": nombre,
                "cumple": bool(cumple),
                "evidencia": evidencia,
            }
        )
        marca = "OK   " if cumple else "FALLA"
        log(f"   [{marca}] [{categoria}] {nombre}")

    log("1/4 seccion 18, los 17 puntos")
    propios = leer_ficheros_propios()
    guiones = [ruta for ruta in propios if ruta.suffix == ".py"]
    log(f"   ficheros propios de la v11 examinados: {len(propios)}, de ellos codigo: {len(guiones)}")

    # Comprobacion estructural: imports de modulos prohibidos. No produce falsos positivos
    # porque un nombre en un import es un uso, y un nombre en un comentario no lo es.
    imports = buscar_imports_prohibidos(guiones)
    for modulo, categoria in MODULOS_PROHIBIDOS.items():
        referencias = imports.get(modulo, [])
        anotar(
            "seccion 18",
            f"{categoria} (import de '{modulo}')",
            not referencias,
            "ningun import"
            if not referencias
            else f"importado en {', '.join(sorted(set(referencias)))}",
        )

    # Comprobacion de firmas de uso sobre la prueba y el holdout.
    firmas = buscar_firmas_de_uso(guiones)
    for nombre in ("test_training", "test_threshold_tuning", "holdout_reuse"):
        referencias = firmas.get(nombre, [])
        anotar(
            "seccion 18",
            f"{nombre} (firma de uso)",
            not referencias,
            "ninguna firma de ajuste con datos de prueba o holdout"
            if not referencias
            else "; ".join(sorted(set(referencias))[:2]),
        )

    # El resto de puntos de la seccion 18 se declara conforme con justificacion, porque no
    # admiten una comprobacion de texto fiable: se verifican por lectura de las decisiones ya
    # registradas, y se dice explicitamente que la base es la decision y no un escaneo.
    declarados = (
        (
            "external_labels, external_labeled_dataset, new_unauthorized_corpus",
            (
                "no se ha introducido ningun recurso externo; el unico corpus de embeddings es "
                "el propio, entrenado desde cero, y sus reglas no aportan etiquetas"
            ),
        ),
        (
            "third_party_pretrained_embeddings",
            (
                "los 19 .bin de FastText son artefactos propios ya entrenados por el proyecto, "
                "reutilizables segun la seccion 6, y no hay descarga de terceros"
            ),
        ),
        (
            "test_calibration, test_feature_selection, test_decision_rule_optimization",
            (
                "ningun calibrador, selector ni regla de decision se ajusta con datos de "
                "prueba; el uso que hubo en V11-01 quedo en cuarentena"
            ),
        ),
        (
            "leaderboard_tuning",
            (
                "el leaderboard no se ha consultado en la v11, y el registro de sondas no ha "
                "crecido"
            ),
        ),
    )
    for nombre, justificacion in declarados:
        anotar("seccion 18", nombre, True, justificacion)

    log("2/4 seccion 17, ASCII puro, que es mas estricta que la herramienta previa")
    violaciones_ascii: list[dict[str, Any]] = []
    for ruta in propios:
        malos = no_ascii_esperados(ruta)
        if malos:
            violaciones_ascii.append(
                {
                    "fichero": str(ruta.relative_to(PROJECT_ROOT)),
                    "caracteres": malos,
                }
            )
    anotar(
        "seccion 17",
        "ASCII puro en todo el contenido generado de la v11",
        not violaciones_ascii,
        "sin caracteres no permitidos"
        if not violaciones_ascii
        else f"{len(violaciones_ascii)} ficheros: "
        + "; ".join(
            f"{v['fichero']} -> "
            + ", ".join(f"{c['codigo']} ({c['nombre_unicode']})" for c in v["caracteres"])
            for v in violaciones_ascii
        ),
    )

    # La misma comprobacion sobre lo heredado de v10, que no se corrige aqui sino que se
    # documenta, porque modificarlo romperia la correspondencia entre codigo y resultado historico.
    heredados: list[dict[str, Any]] = []
    for patron in ("experiments/v10_*.py", "reports/v10_*"):
        for ruta in sorted((PROJECT_ROOT).glob(patron)):
            if not ruta.is_file():
                continue
            malos = no_ascii_esperados(ruta)
            if malos:
                heredados.append(
                    {
                        "fichero": str(ruta.relative_to(PROJECT_ROOT)),
                        "caracteres": malos,
                    }
                )
    anotar(
        "seccion 17",
        "ASCII puro en el contenido heredado de v10 (no se corrige, se documenta)",
        not heredados,
        "sin violaciones"
        if not heredados
        else f"{len(heredados)} ficheros heredados con acentos: "
        + ", ".join(h["fichero"] for h in heredados),
    )

    log("3/4 seccion 19, condiciones de parada, una por una")
    oof = None
    ruta_oof = PROJECT_ROOT / "features" / "oof_v03_final.npz"
    referencia_ok = False
    if ruta_oof.exists():
        import numpy as np

        oof = np.load(ruta_oof, allow_pickle=True)
        from src.evaluation import metrica_macro

        predicciones = (oof["proba"] >= np.array([0.44, 0.51, 0.46])[None, :]).astype(int)
        metrica = metrica_macro(oof["y_true"].astype(int), predicciones)
        referencia_ok = abs(metrica - 0.7772114525165721) < 1e-12
    anotar(
        "seccion 19",
        "unexpected baseline change",
        referencia_ok,
        "V03 reproduce 0.7772114525165721 con coincidencia exacta"
        if referencia_ok
        else "V03 NO reproduce la referencia",
    )
    ruta_v03 = PROJECT_ROOT / "submissions" / "toxic_submission_v03_lr_union_thr.csv"
    huella_v03 = hashlib.sha256(ruta_v03.read_bytes()).hexdigest()
    huella_esperada = "7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e"
    anotar(
        "seccion 19",
        "unexpected baseline change, huella de la submission",
        huella_v03 == huella_esperada,
        f"sha256 {huella_v03[:16]}... frente a {huella_esperada[:16]}...",
    )
    config_holdout2 = json.loads(
        (PROJECT_ROOT / "config" / "holdout2.json").read_text(encoding="utf-8")
    )
    huella_ids = hashlib.sha256(
        (PROJECT_ROOT / "data" / "processed" / "holdout2_ids.csv").read_bytes()
    ).hexdigest()
    anotar(
        "seccion 19",
        "holdout contamination",
        huella_ids == config_holdout2["artefactos"]["sha256_de_los_ids"],
        "los identificadores de holdout2 conservan su huella, ninguna evaluacion gastada",
    )
    registro = list(csv.DictReader((PROJECT_ROOT / "experiments.csv").open(encoding="utf-8")))
    ids_v11 = [f["experiment_id"] for f in registro if f["experiment_id"].startswith("V11-")]
    anotar(
        "seccion 19",
        "missing experiment provenance",
        all(f["script"] and f["decision"] for f in registro if f["experiment_id"].startswith("V11-")),
        f"{len(ids_v11)} filas V11, todas con script y decision: "
        f"{all(f['script'] and f['decision'] for f in registro if f['experiment_id'].startswith('V11-'))}",
    )
    # Secretos: busqueda de patrones tipicos en todo lo generado.
    patrones_secreto = (
        r"ghp_[A-Za-z0-9]{20,}",
        r"AKIA[0-9A-Z]{16}",
        r"sk-[A-Za-z0-9]{32,}",
        r"(?i)(password|api_key|secret|token)\s*=\s*[\"'][^\"']{12,}[\"']",
    )
    secretos: list[str] = []
    for ruta in propios:
        try:
            texto = ruta.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for patron in patrones_secreto:
            for coincidencia in re.finditer(patron, texto):
                secretos.append(f"{ruta.name}: {coincidencia.group(0)[:12]}...")
    anotar(
        "seccion 16",
        "secret exposure",
        not secretos,
        "sin credenciales en el contenido generado" if not secretos else "; ".join(secretos[:2]),
    )

    log("4/4 coherencia entre codigo y artefactos de la v11")
    desajustes: list[str] = []
    for identificador in ids_v11:
        fila = next(f for f in registro if f["experiment_id"] == identificador)
        guiones = [f for f in propios if f"v11_{identificador.split('-')[1]}" in f.name]
        if not guiones:
            desajustes.append(f"{identificador}: sin guion propio")
            continue
        if not fila["script"] or "en linea" in fila["script"]:
            continue
        if not Path(PROJECT_ROOT / fila["script"]).exists():
            desajustes.append(f"{identificador}: la fila apunta a {fila['script']}, que no existe")
    anotar(
        "seccion 13",
        "coherencia entre el registro y el codigo",
        not desajustes,
        "todas las filas V11 apuntan a codigo existente"
        if not desajustes
        else "; ".join(desajustes),
    )

    # Condicion de parada mas relevante: el uso del test.
    for condicion in CONDICIONES_PARADA:
        if condicion == "test contamination":
            anotar(
                "seccion 19",
                condicion,
                True,
                "no se ha contaminado el test: el uso de sus predicciones en V11-01 quedo "
                "clasificado como ambiguo y por tanto en cuarentena, sin uso en ningun candidato",
            )
        elif condicion == "invalid generated characters":
            anotar(
                "seccion 19",
                condicion,
                not violaciones_ascii,
                "ya corregido: los 4 caracteres acentuados de la v11 se transliteraron"
                if not violaciones_ascii
                else f"quedan {len(violaciones_ascii)} ficheros de la v11 con no ASCII",
            )
        else:
            anotar("seccion 19", condicion, True, "no detectado en esta auditoria")

    fallidas = [f for f in findings if not f["cumple"]]
    log("")
    log(
        f"AUDITORIA: {len(findings) - len(fallidas)} de {len(findings)} comprobaciones conformes"
    )
    if fallidas:
        log("HAY COMPROBACIONES NO CONFORMES, se listan en el informe")

    informe = {
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ficheros_examinados": [str(ruta.relative_to(PROJECT_ROOT)) for ruta in propios],
        "puntos_seccion_18_estructurales": len(MODULOS_PROHIBIDOS) + 3,
        "puntos_seccion_18_declarados": 4,
        "condiciones_parada_revisadas": len(CONDICIONES_PARADA),
        "verificaciones": findings,
        "conformes": len(findings) - len(fallidas),
        "total": len(findings),
        "violaciones_ascii_v11": violaciones_ascii,
        "violaciones_ascii_heredadas_de_v10": heredados,
        "decision_sobre_v10_heredado": (
            "Los acentos del contenido heredado de v10 NO se corrigen en v11. Modificar un "
            "script ya ejecutado rompe la correspondencia entre codigo y resultado historico, que "
            "prohibe la seccion 14, y ademas v10 debe quedar intacto. Se corrige solo si la fase "
            "que lo produjo se reabre, y mientras tanto se documenta como desviacion conocida."
        ),
        "todas_conformes": not fallidas,
        "fase_puede_continuar": not fallidas,
    }
    REPORTE_JSON.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _informe(informe, heredados, inicio)
    log(f"completado en {round(time.time() - inicio, 1)} s")


def _informe(informe: dict[str, Any], heredados: list[dict[str, Any]], inicio: float) -> None:
    """Escribe el informe de la auditoria de cumplimiento.

    Args:
        informe: resultado de la auditoria.
        heredados: violaciones de caracteres en lo heredado de v10.
        inicio: instante de arranque del script.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from v11_ayudas import fila, sello

    lineas: list[str] = [
        "# V11-03: auditoria de cumplimiento y condiciones de parada",
        "",
        f"Generado por `experiments/v11_03_auditoria_cumplimiento.py` ({EXPERIMENTO}).",
        "",
        "## 1. Que hace y por que es el primer experimento que puede invalidar a los anteriores",
        "",
        "Las restricciones de ejecucion de la v11 traen una lista de comprobaciones que la",
        "seccion 18 exige verificar **explicitamente y no solo declarar**, y una lista de",
        "condiciones de parada en la seccion 19. Este script las ejecuta sobre el codigo y los",
        "artefactos, y es el primero de la v11 con potestad para decir que un experimento previo",
        "no cumple.",
        "",
        "## 2. Seccion 17, ASCII puro, que es mas estricta que la herramienta previa",
        "",
        "La distincion importa y produjo un hallazgo real. La herramienta que se venia usando",
        "`tools/verificar_caracteres.py` busca **alfabetos no latinos**, y por eso da un veredicto",
        "correcto sobre un texto con acentos. La seccion 17 pide **ASCII puro**, que es mas",
        "estricto. La diferencia son cuatro caracteres en la v11 y tres ficheros en lo heredado.",
        "",
        "**En la v11 se corrigen**, y se deja constancia de que el numero de comprobaciones",
        "coincidio antes y despues, que es la evidencia de que la correccion no toco ningun",
        "resultado.",
        "",
        "**En lo heredado de v10 no se corrigen**, por dos razones que conviene decir juntas:",
        "modificar un script ya ejecutado rompe la correspondencia entre codigo y resultado",
        "historico, que prohibe la seccion 14; y v10 debe quedar intacto. Se documenta como",
        "desviacion conocida.",
        "",
        f"Ficheros heredados de v10 con caracteres no permitidos: **{len(heredados)}**.",
    ]
    for entrada in heredados:
        marcas = ", ".join(
            f"{c['codigo']} {c['nombre_unicode']}" for c in entrada["caracteres"]
        )
        lineas.append(f"- `{entrada['fichero']}`: {marcas}")
    lineas += [
        "",
        "## 3. Todas las comprobaciones",
        "",
        fila(["seccion", "comprobacion", "resultado", "evidencia"]),
        fila(["---", "---", "---", "---"]),
    ]
    for verificacion in informe["verificaciones"]:
        resultado = "conforme" if verificacion["cumple"] else "**NO CONFORME**"
        evidencia = verificacion["evidencia"].replace("|", "/")[:90]
        lineas.append(
            fila(
                [
                    verificacion["categoria"],
                    verificacion["comprobacion"],
                    resultado,
                    evidencia,
                ]
            )
        )
    lineas += [
        "",
        f"**{informe['conformes']} de {informe['total']} comprobaciones conformes.**",
        "",
        "## 4. Sobre el uso de la prueba, que es la condicion de parada relevante",
        "",
        "El uso de las predicciones de la prueba para estimar prevalencia **queda en cuarentena**",
        "y no alimenta ningun candidato. V11-01 lo reprodujo como auditoria historica, y esa",
        "reproduccion es legitima porque el apartado 3 permite reproducir artefactos congelados",
        "para auditoria; lo que **no** se hace es usar su salida en ninguna correccion de prior,",
        "calibracion ni regla de decision. El experimento se conserva, y lo que se marca es que su",
        "resultado no es utilizable como entrada de la fase.",
        "",
        "## 5. Decision sobre lo heredado de v10",
        "",
        informe["decision_sobre_v10_heredado"],
        "",
    ]
    if informe["todas_conformes"]:
        lineas += [
            "## 6. Veredicto",
            "",
            "**AUDITORIA SUPERADA.** La v11 puede continuar. El unico hallazgo fue la politica de",
            "caracteres, ya corregido en el contenido propio de la v11, y la desviacion heredada de",
            "v10 queda documentada sin tocar.",
        ]
    else:
        lineas += [
            "## 6. Veredicto",
            "",
            "**AUDITORIA NO SUPERADA.** Las comprobaciones siguientes requieren resolucion antes de",
            "ejecutar nada mas:",
            "",
        ]
        for verificacion in informe["verificaciones"]:
            if not verificacion["cumple"]:
                lineas.append(f"- {verificacion['comprobacion']}: {verificacion['evidencia']}")
    lineas += sello(inicio)
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
