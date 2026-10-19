"""Actualizacion del notebook consolidado: autocontenido, completo y limpio.

Genera la version consolidada de notebooks/10_final_pipeline.ipynb:

1. Autocontenido: las funciones que el notebook importaba de src/ (data.py,
   evaluation.py y embedding_utils.py) quedan embebidas en una celda nueva, y
   se eliminan los imports y las inserciones en sys.path.
2. Completo: se anade la seccion 127 (fase P84), el anexo A (seccion 128) con
   los identificadores de la union del registro y el inventario (483 filas),
   y la verificacion de completitud (seccion 129).
3. Limpieza: se sustituyen los simbolos no ASCII distintos de la ortografia
   espanola (guion largo, signo menos, comillas angulares, etc.) por sus
   equivalentes ASCII.

El script es idempotente: regenera las secciones nuevas si ya existen y no toca
el codigo de los experimentos cerrados.

Uso:
    /home/carlos/venv/bin/python experiments/p84_06_notebook_completo.py
"""

from __future__ import annotations

import csv
import json
import re
import uuid
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RUTA_NOTEBOOK = RAIZ / "notebooks" / "10_final_pipeline.ipynb"
RUTA_REGISTRO = RAIZ / "experiments.csv"
RUTA_INVENTARIO = RAIZ / "auditoria" / "inventario_experimentos.csv"

MARCA_DEFINICIONES = "Definiciones embebidas: el notebook no importa modulos"
MARCA_CONFIG = '"""Configuración común del notebook'
MARCA_VERIF_INVENTARIO = "Verificacion del inventario"
MARCA_P84 = "## 127. Fase P84"
MARCA_ANEXO = "## 128. Anexo A"
MARCA_VERIF_ANEXO = "## 129. Anexo B"
MARCA_CODIGO_ANEXO = "Verificacion de completitud del anexo A"

# Simbolos no ASCII que no son ortografia espanola. El orden importa: primero
# los contextos concretos y despues los caracteres sueltos.
REEMPLAZOS = [
    (
        "cero \u2014vectores que sí agrupan palabras por contexto\u2014 puede",
        "cero (vectores que sí agrupan palabras por contexto) puede",
    ),
    ("ortográficas \u2014el 66,6 %", "ortográficas, el 66,6 %"),
    ("`espaa`)\u2014 y esas", "`espaa`), y esas"),
    ("**estructura del texto** \u2014 123 features", "**estructura del texto**: 123 features"),
    ("\u2014", "-"),
    ("\u2212", "-"),
    ("\u00ab", '"'),
    ("\u00bb", '"'),
    ("\u2026", "..."),
    ("\u2260", "!="),
    ("\u2208", "en"),
    ("\u03b1\u2080", "alpha_0"),
    ("\u03b1", "alpha"),
    ("\u00a7", "seccion"),
]

# Ortografia espanola admitida: el resto de caracteres no ASCII es un error.
PERMITIDOS = set("áéíóúüñÁÉÍÓÚÜÑ¿¡")


def limpiar_texto(texto: str) -> str:
    """Sustituye los simbolos no ASCII por equivalentes ASCII."""
    for viejo, nuevo in REEMPLAZOS:
        texto = texto.replace(viejo, nuevo)
    return texto


def texto_de(celda: dict) -> str:
    """Devuelve el codigo o markdown de una celda como una sola cadena."""
    return "".join(celda.get("source", []))


def asignar_texto(celda: dict, texto: str) -> None:
    """Escribe una cadena como fuente de la celda, linea a linea."""
    lineas = texto.splitlines(keepends=True)
    celda["source"] = lineas if lineas else [""]


def nueva_celda_markdown(texto: str) -> dict:
    """Crea una celda de markdown con identificador unico."""
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": texto.splitlines(keepends=True),
    }


def nueva_celda_codigo(texto: str) -> dict:
    """Crea una celda de codigo vacia de salidas y sin ejecutar."""
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": texto.splitlines(keepends=True),
    }


def buscar(cells: list[dict], aguja: str) -> int:
    """Devuelve el indice de la primera celda que contiene la aguja."""
    for indice, celda in enumerate(cells):
        if aguja in texto_de(celda):
            return indice
    raise SystemExit(f"No se encontro la celda con la marca: {aguja!r}")


def reemplazar_en(cells: list[dict], aguja: str, viejo: str, nuevo: str) -> None:
    """Sustituye un texto exacto dentro de la celda que contiene la aguja.

    Es idempotente: si el texto viejo ya no está y el nuevo sí, no hace nada.
    """
    indice = buscar(cells, aguja)
    fuente = texto_de(cells[indice])
    if viejo not in fuente:
        if nuevo and nuevo in fuente:
            return
        raise SystemExit(f"No se encontro el texto a reemplazar en la celda {indice}: {viejo!r}")
    asignar_texto(cells[indice], fuente.replace(viejo, nuevo))


def quitar_de(cells: list[dict], aguja: str, texto: str) -> None:
    """Elimina un texto exacto dentro de la celda que contiene la aguja."""
    indice = buscar(cells, aguja)
    fuente = texto_de(cells[indice])
    if texto in fuente:
        asignar_texto(cells[indice], fuente.replace(texto, ""))


def truncar(texto: str, limite: int) -> str:
    """Recorta un texto a un limite de caracteres, en frontera de palabra."""
    limpio = " ".join(str(texto).replace("|", "/").split())
    if len(limpio) <= limite:
        return limpio
    recorte = limpio[:limite]
    if " " in recorte:
        recorte = recorte[: recorte.rfind(" ")]
    return recorte + "..."


CODIGO_EMBEBIDO = r'''"""Definiciones embebidas: el notebook no importa modulos del proyecto.

Estas funciones replican las de src/data.py, src/evaluation.py y
src/embedding_utils.py para que el notebook sea autocontenido. Las rutas por
defecto usan PROJECT_ROOT, definido en la celda anterior.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ETIQUETAS = ("HS", "TR", "AG")
LABEL_COLUMNS = ETIQUETAS
TRAIN_COLUMNS = ("id", "text", "HS", "TR", "AG")
TEST_COLUMNS = ("id", "text")
SUBMISSION_COLUMNS = ("id", "HS", "TR", "AG")

RUTA_TRAIN = PROJECT_ROOT / "data" / "raw" / "toxic_train.csv"
RUTA_TEST = PROJECT_ROOT / "data" / "raw" / "toxic_test.csv"
RUTA_SUBMISSION = PROJECT_ROOT / "data" / "raw" / "submission.csv"


def _decodificar_bytes(raw: bytes) -> tuple[str, str]:
    """Decodifica bytes CSV probando UTF-8 y cayendo a cp1252.

    Args:
        raw: contenido binario del archivo.

    Returns:
        Tupla ``(texto, encoding_usado)``.
    """
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace"), "cp1252"


def _leer_csv_robusto(path: Path, sep: str, columns: tuple[str, ...]) -> pd.DataFrame:
    """Lee un CSV oficial respetando el encoding verificado.

    Args:
        path: ruta del CSV.
        sep: separador de campos.
        columns: nombres de columna esperados, en orden.

    Returns:
        DataFrame con ``columns`` y todas las columnas en tipo ``str``.
    """
    raw = path.read_bytes()
    _, encoding = _decodificar_bytes(raw)
    frame = pd.read_csv(
        path,
        sep=sep,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8" if encoding == "utf-8" else "cp1252",
        encoding_errors="replace",
    )
    frame.columns = [column.strip() for column in frame.columns]
    if tuple(frame.columns) != columns:
        raise ValueError(f"Columnas inesperadas en {path.name}: {list(frame.columns)}")
    frame["id"] = frame["id"].str.strip().str.strip('"')
    return frame


def read_train(path: Path | None = None) -> pd.DataFrame:
    """Carga toxic_train.csv con etiquetas HS, TR y AG como enteros."""
    frame = _leer_csv_robusto(path or RUTA_TRAIN, sep=";", columns=TRAIN_COLUMNS)
    for column in LABEL_COLUMNS:
        frame[column] = frame[column].astype(int)
    return frame


def read_test(path: Path | None = None) -> pd.DataFrame:
    """Carga toxic_test.csv (sin etiquetas)."""
    return _leer_csv_robusto(path or RUTA_TEST, sep=";", columns=TEST_COLUMNS)


def read_submission(path: Path | None = None) -> pd.DataFrame:
    """Carga la plantilla submission.csv con etiquetas enteras."""
    frame = _leer_csv_robusto(path or RUTA_SUBMISSION, sep=",", columns=SUBMISSION_COLUMNS)
    for column in LABEL_COLUMNS:
        frame[column] = frame[column].astype(int)
    return frame


def validate_frame(frame: pd.DataFrame, columns: tuple[str, ...], n_rows: int) -> dict:
    """Valida integridad basica de un DataFrame oficial.

    Args:
        frame: DataFrame a validar.
        columns: columnas esperadas.
        n_rows: numero de filas esperado.

    Returns:
        Diccionario con el informe de validacion.
    """
    report = {
        "n_rows": len(frame),
        "n_rows_esperado": n_rows,
        "columnas": tuple(frame.columns),
        "sin_nulos": not frame.isna().any().any(),
        "ids_numericos": bool(frame["id"].str.fullmatch("[0-9]+").all()),
        "ids_unicos": bool(frame["id"].is_unique),
        "texto_no_vacio": bool((frame["text"].str.strip() != "").all()),
    }
    report["ok"] = bool(
        report["n_rows"] == report["n_rows_esperado"]
        and report["sin_nulos"]
        and report["ids_numericos"]
        and report["ids_unicos"]
        and report["texto_no_vacio"]
        and report["columnas"] == columns
    )
    return report


def metricas_multietiqueta(y_true, y_pred) -> dict:
    """Calcula las metricas multi-etiqueta del proyecto (f1_media_columnas, D-009).

    Args:
        y_true: matriz binaria verdadera de forma ``(n_filas, 3)`` (HS, TR, AG).
        y_pred: matriz binaria predicha, con la misma forma.

    Returns:
        Diccionario con las metricas globales y ``f1_por_etiqueta``.
    """
    verdad = np.asarray(y_true)
    prediccion = np.asarray(y_pred)
    if verdad.shape != prediccion.shape:
        raise ValueError(f"y_true {verdad.shape} e y_pred {prediccion.shape} deben coincidir")
    if verdad.ndim != 2 or verdad.shape[1] != len(ETIQUETAS):
        raise ValueError(f"Se esperaba una matriz (n, {len(ETIQUETAS)}); recibida {verdad.shape}")
    f1_positivas = [
        float(f1_score(verdad[:, indice], prediccion[:, indice], zero_division=0))
        for indice in range(len(ETIQUETAS))
    ]
    f1_negativas = [
        float(f1_score(1 - verdad[:, indice], 1 - prediccion[:, indice], zero_division=0))
        for indice in range(len(ETIQUETAS))
    ]
    return {
        "f1_media_columnas": float(np.mean(f1_positivas)),
        "f1_macro6": float(np.mean(f1_positivas + f1_negativas)),
        "f1_micro": float(f1_score(verdad.ravel(), prediccion.ravel(), zero_division=0)),
        "f1_samples": float(f1_score(verdad, prediccion, average="samples", zero_division=0)),
        "exact_match": float((verdad == prediccion).all(axis=1).mean()),
        "f1_por_etiqueta": {
            etiqueta: valor for etiqueta, valor in zip(ETIQUETAS, f1_positivas, strict=True)
        },
    }


def sha256_de(ruta: Path) -> str:
    """Calcula el sha256 de un archivo por bloques.

    Args:
        ruta: ruta del archivo.

    Returns:
        Hash hexadecimal.
    """
    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1 << 20), b""):
            digest.update(bloque)
    return digest.hexdigest()
'''

CODIGO_VERIF_INVENTARIO = r'''"""Verificacion del inventario y del registro.

Comprueba el estado del 2026-10-09: el inventario de la auditoria conserva 483
experimentos y 376 scripts clasificados; el registro tiene 459 filas validas y
unicas (P84-01 a P84-05, P85-01, P86-00, P87-00 y P88-00 se anadieron tras la auditoria); y no quedan
filas malformadas de P67 a P77. La cobertura completa del anexo A se verifica en la
seccion 129.
"""

import pandas as pd

inventario_experimentos = pd.read_csv(PROJECT_ROOT / "auditoria" / "inventario_experimentos.csv")
inventario_scripts = pd.read_csv(PROJECT_ROOT / "auditoria" / "inventario_scripts.csv")

assert len(inventario_experimentos) == 483, (
    f"el inventario debe tener 483 experimentos, tiene {len(inventario_experimentos)}"
)
assert len(inventario_scripts) == 376, (
    f"el inventario debe tener 376 scripts, tiene {len(inventario_scripts)}"
)

conteos = inventario_scripts["tipo"].value_counts().to_dict()
ESPERADOS = {
    "experimento": 251,
    "auditoria": 43,
    "integracion_notebook": 39,
    "cierre": 26,
    "submission": 9,
    "registro": 5,
    "scaffold": 2,
    "ayudas": 1,
}
for tipo, esperado in ESPERADOS.items():
    obtenido = int(conteos.get(tipo, 0))
    assert obtenido == esperado, f"{tipo}: se esperaban {esperado} scripts, hay {obtenido}"
assert sum(ESPERADOS.values()) == len(inventario_scripts)

malformadas = inventario_experimentos.loc[
    inventario_experimentos["id"].astype(str).str.match(r"^P(6[7-9]|7[0-7])")
    & (inventario_experimentos["fila_registro_valida"] == "NO")
]
assert len(malformadas) == 0, f"tras la reparacion no debe haber malformadas, hay {len(malformadas)}"

registro = pd.read_csv(PROJECT_ROOT / "experiments.csv")
assert len(registro) == 459, f"el registro debe tener 459 filas, tiene {len(registro)}"
assert registro["experiment_id"].is_unique, "los identificadores del registro deben ser unicos"

print("Inventario y registro (estado 2026-10-09):")
print(f"  inventario: {len(inventario_experimentos)} experimentos | {len(inventario_scripts)} scripts")
print(f"  registro experiments.csv: {len(registro)} filas validas y unicas (459 tras P88-00)")
print(f"  filas malformadas de P67 a P77: {len(malformadas)}")
print("Verificacion completada.")
'''

CODIGO_VERIF_ANEXO = r'''"""Verificacion de completitud del anexo A.

Comprueba que el notebook lista los identificadores de la union del registro
(experiments.csv) y del inventario (auditoria/inventario_experimentos.csv) que
no tienen fila propia.
"""

import json
import re

import pandas as pd

registro = pd.read_csv(PROJECT_ROOT / "experiments.csv")
inventario = pd.read_csv(PROJECT_ROOT / "auditoria" / "inventario_experimentos.csv")

ids = sorted(set(registro["experiment_id"].astype(str)) | set(inventario["id"].astype(str)))

notebook = json.loads(
    (PROJECT_ROOT / "notebooks" / "10_final_pipeline.ipynb").read_text(encoding="utf-8")
)
texto = "\n".join("".join(c.get("source", [])) for c in notebook["cells"])


def aparece(identificador: str) -> bool:
    """Comprueba si un identificador aparece como palabra completa en el texto."""
    patron = r"(?<![\w-])" + re.escape(identificador) + r"(?![\w-])"
    return re.search(patron, texto) is not None


faltan = [identificador for identificador in ids if not aparece(identificador)]
assert not faltan, f"faltan {len(faltan)} identificadores en el notebook: {faltan[:10]}"

print("Cobertura del anexo A:")
print(f"  registro: {len(registro)} | inventario: {len(inventario)} | union: {len(ids)}")
print(f"  identificadores presentes en el notebook: {len(ids) - len(faltan)} de {len(ids)}")
print("Completitud verificada: no falta ningun identificador.")
'''


def construir_seccion_p84() -> str:
    """Devuelve el markdown de la seccion 127 (fase P84)."""
    return """## 127. Fase P84: exploración dirigida y cierre (2026-10-02)

La fase P84 buscó, con pre-registro y gates, algún candidato capaz de superar a V03 antes de cerrar la exploración. El pre-registro está en `docs/p84_preregistro.md` (enmiendas 1 a 5 y 3bis), las especificaciones ejecutables en `docs/p84_spec.md` y el informe final en `reports/p84_informe_final.md`. El Gate 0 fijó los hashes congelados: train (`3d40e734`), OOF de V03 (`55926581`), grupos fusionados (`afabf9ae`), corpus wikipedia_es 10M (`00b839da`) y binario de fastText (`59c51cc5`). Reglas de la fase: desarrollo de 1483 filas, holdout y test intocables, un script nuevo por experimento, sin rejillas post-hoc y sin submissions.

| Experimento | Pregunta | Resultado principal | Decisión |
|---|---|---|---|
| P84-01 | DL-01 medido sin contaminación de pool ni de época | F1@0.5 0.758301; delta -0.018910 (IC95 [-0.033910, -0.004150]); 1/5 pliegues | Cerrar DL-01 como artefacto de protocolo |
| P84-02 | fastText supervisado entrenado desde cero | Principal degenerado 0.583398 (0/5); control sin subwords 0.737771 (delta -0.039441, 0/5) | Cerrar sin rejilla nueva |
| P84-03 | ¿Aporta ROS algo distinto del reweighting? | Equivalencia con reweighting; B1 +0.000266 (3/5), B2 -0.044992 (0/5), B3 +0.000210 (3/5) | Cerrar por equivalencia; no promovible |
| P84-04 | SGNS corregido (suma) anclado contra fastText | Ancla lograda; S 0.626091, M 0.622888, R 0.651825; S vs V03 -0.151120 | Cerrar la línea SGNS corregido |
| P84-05 | Corrección EM de prevalencia con shrinkage | Delta medio -0.008394 (IC95 [-0.015763, -0.001231]); 2/5 LOFO | Cerrar la familia EM/threshold |
| EXP-06 | Puerta de combinación de candidatos | Ningún candidato pasó el Gate 1 ni la vía de complementariedad | Cerrada sin ejecución |

**P84-01 (DL-01 corregido).** MLP sobre SVD(300) con 110.339 parámetros, época elegida en particiones internas (mediana de 3) y reajuste fijo. Brecha train-validación 0.209140 frente a 0.204721 de V03; `ensemble_gain` -0.001679; reproducible entre corridas (máx|delta proba| = 0.0). El 0.7949 histórico no es comparable: usó pool de 1731 filas y eligió la época sobre el pliegue puntuado.

**P84-02 (fastText supervisado).** Tres clasificadores binarios fold-local con dim 100, lr 0.1 y wordNgrams 2; el brazo principal quedó degenerado (loss ~ ln 2, probabilidades ~ 0.5) con los flags pre-registrados, y el control sin subwords alcanzó 0.737771 (IC95 [-0.054696, -0.024475]). La pregunta de subwords queda sin responder.

**P84-03 (ROS).** Micro-verificación de equivalencia: con liblinear L2, duplicar filas equivale a reasignar pesos por instancia. La refutación parcial encontró desviación en probabilidad (máx|delta proba| 7.19e-2 por el redondeo entero de las copias) pero |delta F1| 2.66e-4. Recuentos corregidos (Enmienda 3bis): B1 61 celdas (32 rescates / 29 daños), B2 446 (207/239) y B3 63 (34/29) sobre 4449.

**P84-04 (SGNS corregido).** Ancla lograda antes de entrenar (identidad algebraica 1.3e-16, gradiente 4.4e-08 y pureza 1.0 = 1.0 contra fastText oficial); corpus wikipedia_es 10M EXTERNAL-ONLY, dim 200 y 3 semillas. S vs M +0.003204 (3/5, IC95 [-0.012606, +0.018844], no significativo). Matiz de la refutación final: S vs R -0.025734 es efecto del promedio de probabilidades entre semillas; por semilla los deltas son -0.0032, -0.0035 y +0.0062 (empate).

**P84-05 (EM + shrinkage).** Simulación analítica sobre el OOF de V03 con 5 pliegues LOFO; la componente EM aporta +0.020019 pero el umbral reubicado resta -0.028414; la sanidad con rho = 1 da -0.005043 y el guardarraíl de tasa falla en AG.

**Coste.** La fase completa costó 15987.10 s (unas 4.44 h): P84-01 248.33 s en GPU, P84-02 206.36 s en CPU, P84-03 20.24 s en CPU, P84-04 15511.17 s en GPU y P84-05 1.0 s en CPU.

**Decisiones.** D-281 a D-287 registran los cinco cierres, la puerta EXP-06 y las correcciones de redacción posteriores a la refutación adversarial.

**Cierre.** Ningún candidato supera a V03 (`corrected_cv` 0.7772114525165721; LB 0.78306) y no se generaron submissions. V03 remains the best verified leaderboard result."""


def cargar_filas() -> tuple[list[dict], list[dict]]:
    """Carga el registro y el inventario como listas de diccionarios."""
    registro = list(csv.DictReader(RUTA_REGISTRO.read_text(encoding="utf-8-sig").splitlines()))
    inventario = list(csv.DictReader(RUTA_INVENTARIO.read_text(encoding="utf-8-sig").splitlines()))
    return registro, inventario


def construir_anexo(registro: list[dict], inventario: list[dict]) -> str:
    """Devuelve el markdown de la seccion 128 con los identificadores de la union."""
    ids_registro = {fila["experiment_id"].strip() for fila in registro}
    familia_por_id = {fila["id"].strip(): fila.get("familia", "").strip() for fila in inventario}

    filas_a1 = []
    for fila in registro:
        identificador = fila["experiment_id"].strip()
        familia = familia_por_id.get(identificador) or (
            "p84" if identificador.startswith("P84")
            else (
                "p85" if identificador.startswith("P85")
                else (
                    "p86" if identificador.startswith("P86")
                    else ("p87" if identificador.startswith("P87") else ("p88" if identificador.startswith("P88") else "-"))
                )
            )
        )
        fecha = (fila.get("date") or "").strip() or "-"
        f1 = (fila.get("f1") or "").strip() or "-"
        decision = truncar(limpiar_texto(fila.get("decision") or "-"), 100)
        filas_a1.append((fecha, identificador, familia, f1, decision))
    filas_a1.sort(key=lambda tupla: (tupla[0], tupla[1]))

    filas_a2 = []
    for fila in inventario:
        identificador = fila["id"].strip()
        if identificador in ids_registro:
            continue
        familia = fila.get("familia", "").strip() or "-"
        estado = fila.get("estado", "").strip() or "-"
        nota = truncar(limpiar_texto(fila.get("nota") or "-"), 100)
        filas_a2.append((identificador, familia, estado, nota))
    filas_a2.sort(key=lambda tupla: tupla[0])

    total = len(filas_a1) + len(filas_a2)
    lineas = [
        f"## 128. Anexo A. Lista completa de experimentos ({total})",
        "",
        f"El anexo reúne los {total} identificadores de la unión de las dos fuentes del proyecto: los {len(filas_a1)} del registro `experiments.csv` (sección A.1) y los {len(filas_a2)} ayudantes del inventario `auditoria/inventario_experimentos.csv` que no tienen fila propia (sección A.2). La columna `decisión` reproduce el inicio de la decisión registrada y la columna `F1` el valor local registrado cuando existe. La verificación de la sección 129 comprueba que no falta ningún identificador.",
        "",
        f"### A.1. Registro de experimentos ({len(filas_a1)})",
        "",
        "| id | familia | fecha | F1 | decisión |",
        "|---|---|---|---|---|",
    ]
    for fecha, identificador, familia, f1, decision in filas_a1:
        lineas.append(f"| {identificador} | {familia} | {fecha} | {f1} | {decision} |")
    lineas += [
        "",
        f"### A.2. Ayudantes del inventario sin fila propia ({len(filas_a2)})",
        "",
        "| id | familia | estado | nota |",
        "|---|---|---|---|",
    ]
    for identificador, familia, estado, nota in filas_a2:
        lineas.append(f"| {identificador} | {familia} | {estado} | {nota} |")
    lineas.append("")
    return "\n".join(lineas)


def construir_verificacion_anexo() -> str:
    """Devuelve el markdown de la seccion 129."""
    return """## 129. Anexo B. Verificación de completitud

La celda siguiente comprueba que este notebook lista los identificadores de la unión del registro y del inventario, y que el registro conserva sus filas válidas y únicas. Es la puerta que cierra el requisito de completitud del anexo A."""


def validar_caracteres(cells: list[dict]) -> None:
    """Falla si queda algún carácter no ASCII fuera de la ortografía española."""
    for indice, celda in enumerate(cells):
        for caracter in texto_de(celda):
            if ord(caracter) > 127 and caracter not in PERMITIDOS:
                raise SystemExit(
                    f"Caracter no permitido en la celda {indice}: {caracter!r} U+{ord(caracter):04X}"
                )


def validar_codigo(cells: list[dict]) -> None:
    """Compila todas las celdas de código para detectar errores de sintaxis."""
    for indice, celda in enumerate(cells):
        if celda["cell_type"] != "code":
            continue
        try:
            compile(texto_de(celda), f"<celda {indice}>", "exec")
        except SyntaxError as error:
            raise SystemExit(f"Error de sintaxis en la celda {indice}: {error}") from error


def main() -> None:
    """Aplica la actualización completa y valida el resultado."""
    notebook = json.loads(RUTA_NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    celdas_antes = len(cells)

    # 1. Celda de configuración: quitar sys.path y actualizar el docstring.
    reemplazar_en(
        cells,
        MARCA_CONFIG,
        "Define RUTAS (relativas a la raíz del proyecto), fija las semillas, añade la\n"
        "raíz al sys.path para importar src.* y expone los ayudantes `mostrar_figura` y\n"
        "`mostrar_tabla`. No descarga nada ni escribe archivos.",
        "Define RUTAS (relativas a la raíz del proyecto), fija las semillas y expone\n"
        "los ayudantes `mostrar_figura` y `mostrar_tabla`. El notebook es autocontenido:\n"
        "las funciones que antes venian de src/ estan embebidas en la celda siguiente.\n"
        "No descarga nada ni escribe archivos.",
    )
    quitar_de(cells, MARCA_CONFIG, "import sys\n")
    quitar_de(
        cells,
        MARCA_CONFIG,
        "if str(PROJECT_ROOT) not in sys.path:\n    sys.path.insert(0, str(PROJECT_ROOT))\n\n",
    )

    # 2. Celda de definiciones embebidas, justo despues de la configuracion.
    existentes = [i for i, c in enumerate(cells) if MARCA_DEFINICIONES in texto_de(c)]
    if existentes:
        asignar_texto(cells[existentes[0]], CODIGO_EMBEBIDO)
    else:
        cells.insert(buscar(cells, MARCA_CONFIG) + 1, nueva_celda_codigo(CODIGO_EMBEBIDO))

    # 3. Quitar los imports de src/ y ajustar los docstrings que los mencionan.
    for celda in cells:
        fuente = texto_de(celda)
        if "from src.data import (" in fuente:
            fuente = re.sub(r"from src\.data import \([^)]*\)\n\n", "", fuente)
            fuente = fuente.replace(
                "Carga oficial de los datos con src.data y auditoría de integridad.",
                "Carga oficial de los datos con las funciones embebidas y auditoría de integridad.",
            )
            asignar_texto(celda, fuente)
        if "from src.evaluation import metricas_multietiqueta" in fuente:
            fuente = fuente.replace("from src.evaluation import metricas_multietiqueta\n\n", "")
            fuente = fuente.replace(
                "Verificación de las métricas OOF del modelo final con src.evaluation.",
                "Verificación de las métricas OOF del modelo final con las funciones embebidas.",
            )
            asignar_texto(celda, fuente)
        if "(411 filas de registro, 23 sondas, 39 scripts)" in fuente:
            fuente = fuente.replace(
                "(411 filas de registro, 23 sondas, 39 scripts)",
                "(455 filas de registro, 23 sondas, 39 scripts)",
            )
            asignar_texto(celda, fuente)
        if "(455 filas de registro, 23 sondas, 39 scripts)" in fuente:
            fuente = fuente.replace(
                "(455 filas de registro, 23 sondas, 39 scripts)",
                "(457 filas de registro, 23 sondas, 39 scripts)",
            )
            asignar_texto(celda, fuente)
        if "(457 filas de registro, 23 sondas, 39 scripts)" in fuente:
            fuente = fuente.replace(
                "(457 filas de registro, 23 sondas, 39 scripts)",
                "(458 filas de registro, 23 sondas, 39 scripts)",
            )
            asignar_texto(celda, fuente)
        if "(458 filas de registro, 23 sondas, 39 scripts)" in fuente:
            fuente = fuente.replace(
                "(458 filas de registro, 23 sondas, 39 scripts)",
                "(459 filas de registro, 23 sondas, 39 scripts)",
            )
            asignar_texto(celda, fuente)
        if "from src.embedding_utils import sha256_de" in fuente:
            fuente = fuente.replace("from src.embedding_utils import sha256_de\n\n", "")
            asignar_texto(celda, fuente)
        if "sys.path.insert(0, str(R / 'experiments'))" in fuente:
            fuente = fuente.replace("import ast, json, subprocess, sys", "import ast, json, subprocess")
            fuente = fuente.replace("sys.path.insert(0, str(R / 'experiments'))\n", "")
            asignar_texto(celda, fuente)

    # 4. Textos desactualizados: resumen, secciones 122, 124 y 125.
    reemplazar_en(
        cells,
        "| **Fecha del notebook** |",
        "| **Fecha del notebook** | 2026-10-08 (trabajo experimental: 2026-09-22 a 2026-10-08) |",
        "| **Fecha del notebook** | 2026-10-09 (trabajo experimental: 2026-09-22 a 2026-10-09) |",
    )
    reemplazar_en(
        cells,
        "14. **Reproducibilidad**",
        "14. **Reproducibilidad**: 39 scripts de experimento (00-38), 458 filas en `experiments.csv`, 23 sondeos en `submissions/leaderboard_log.csv` y las figuras del proyecto.\n"
        "15. **Actualización del 2026-10-03**: el notebook es autocontenido (las funciones que antes se importaban de `src/` están embebidas), documenta la fase P84 en la sección 127 y lista los 491 identificadores de experimentos en el anexo A (sección 128); la verificación de completitud está en la sección 129.\n"
        "16. **Actualización del 2026-10-08**: la comprobación documental final explicita la hipótesis Twitter/X (sección 130, D-289) y la reapertura autorizada P85-01 (sección 131) mide SGNS corregido sobre el corpus de Twitter/X frente al control general; entregables en `reports/toxic_twitter_sgns_corrected.md`.\n"
        "17. **Actualización del 2026-10-09**: la auditoría de novedad P86-00 (sección 132) clasifica como DUPLICATE la Factorization Machine dispersa propuesta sobre la representación de V03: el precedente V25-01/V25-02 ya la ejecutó (control que reproduce V03 bit a bit, k agotado en 4, lambda_v óptimo en 1, mejor +0.005487 con neto -40) y la familia está cerrada; no se ejecuta nada.\n"
        "18. **Actualización del 2026-10-09 (segunda)**: la auditoría de gobernanza P87-00 (sección 133) clasifica la revisión humana de anotaciones en TR como LOW_EXPECTED_VALUE: la contradicción demostrable es escasa y no explica el error, el detector de ruido es marginal y el techo de cualquier intervención queda bajo la banda de detección; no se solicita autorización, no se revisa ni corrige nada y TOXIC permanece cerrado.",
        "14. **Reproducibilidad**: 39 scripts de experimento (00-38), 459 filas en `experiments.csv`, 23 sondeos en `submissions/leaderboard_log.csv` y las figuras del proyecto.\n"
        "15. **Actualización del 2026-10-03**: el notebook es autocontenido (las funciones que antes se importaban de `src/` están embebidas), documenta la fase P84 en la sección 127 y lista los 492 identificadores de experimentos en el anexo A (sección 128); la verificación de completitud está en la sección 129.\n"
        "16. **Actualización del 2026-10-08**: la comprobación documental final explicita la hipótesis Twitter/X (sección 130, D-289) y la reapertura autorizada P85-01 (sección 131) mide SGNS corregido sobre el corpus de Twitter/X frente al control general; entregables en `reports/toxic_twitter_sgns_corrected.md`.\n"
        "17. **Actualización del 2026-10-09**: la auditoría de novedad P86-00 (sección 132) clasifica como DUPLICATE la Factorization Machine dispersa propuesta sobre la representación de V03: el precedente V25-01/V25-02 ya la ejecutó (control que reproduce V03 bit a bit, k agotado en 4, lambda_v óptimo en 1, mejor +0.005487 con neto -40) y la familia está cerrada; no se ejecuta nada.\n"
        "18. **Actualización del 2026-10-09 (segunda)**: la auditoría de gobernanza P87-00 (sección 133) clasifica la revisión humana de anotaciones en TR como LOW_EXPECTED_VALUE: la contradicción demostrable es escasa y no explica el error, el detector de ruido es marginal y el techo de cualquier intervención queda bajo la banda de detección; no se solicita autorización, no se revisa ni corrige nada y TOXIC permanece cerrado.\n"
        "19. **Actualización del 2026-10-09 (tercera)**: la auditoría de novedad P88-00 (sección 134) clasifica como DUPLICATE la propuesta de LightGBM sobre la matriz TF-IDF de V03: el precedente P61-01 ya la ejecutó (mismos vectorizadores, pliegues y umbrales; brazos -0.0302 y -0.0258; 2/5 pliegues; neto -28) y la familia está cerrada; no se ejecuta nada.",
    )
    reemplazar_en(
        cells,
        "## 1. Configuración del notebook",
        "Ninguna celda del notebook reentrena modelos pesados: el contenido se reconstruye desde los artefactos guardados, que son la fuente de verdad del proyecto.",
        "Ninguna celda del notebook reentrena modelos pesados: el contenido se reconstruye desde los artefactos guardados, que son la fuente de verdad del proyecto. El notebook es autocontenido: las funciones auxiliares que antes venian de `src/` estan embebidas en la celda de definiciones, de modo que no importa modulos del proyecto.",
    )
    reemplazar_en(
        cells,
        "### F2. Inventario unificado",
        "(450 filas de registro más 33 ayudantes sin fila tras la corrección; 458 filas de registro y 491 identificadores en total tras P84-01 a P84-05, P85-01, P86-00 y P87-00, ver anexo A)",
        "(450 filas de registro más 33 ayudantes sin fila tras la corrección; 459 filas de registro y 492 identificadores en total tras P84-01 a P84-05, P85-01, P86-00, P87-00 y P88-00, ver anexo A)",
    )
    reemplazar_en(
        cells,
        "## 124. Inventario y trazabilidad",
        "el registro tiene 458 filas (450 tras la corrección del 2026-10-02, 5 más con P84-01 a P84-05, 1 más con P85-01, 1 más con P86-00 y 1 más con P87-00), todas de 23 columnas y sin identificadores duplicados",
        "el registro tiene 459 filas (450 tras la corrección del 2026-10-02, 5 más con P84-01 a P84-05, 1 más con P85-01, 1 más con P86-00, 1 más con P87-00 y 1 más con P88-00), todas de 23 columnas y sin identificadores duplicados",
    )
    reemplazar_en(
        cells,
        "## 124. Inventario y trazabilidad",
        "**Cobertura completa.** En el momento del inventario, 267 experimentos no aparecían mencionados en este notebook. El anexo A (sección 128) cierra ese hueco: lista los 491 identificadores de la unión (458 del registro y 33 ayudantes sin fila propia), y la verificación de la sección 129 comprueba que no falta ninguno. La columna `en_notebook` del inventario conserva el estado previo al anexo.",
        "**Cobertura completa.** En el momento del inventario, 267 experimentos no aparecían mencionados en este notebook. El anexo A (sección 128) cierra ese hueco: lista los 492 identificadores de la unión (459 del registro y 33 ayudantes sin fila propia), y la verificación de la sección 129 comprueba que no falta ninguno. La columna `en_notebook` del inventario conserva el estado previo al anexo.",
    )
    reemplazar_en(
        cells,
        "## 125. Correcciones aplicadas",
        "el registro pasa de 411 a 450 filas (455 tras añadir P84-01 a P84-05, 456 tras P85-01, 457 tras P86-00 y 458 tras P87-00), sin malformadas y sin identificadores duplicados.",
        "el registro pasa de 411 a 450 filas (455 tras añadir P84-01 a P84-05, 456 tras P85-01, 457 tras P86-00, 458 tras P87-00 y 459 tras P88-00), sin malformadas y sin identificadores duplicados.",
    )
    reemplazar_en(
        cells,
        "## 125. Correcciones aplicadas",
        "Los conteos del inventario se actualizaron al estado corregido y la celda de la sección 124 los verifica contra los CSV reales.",
        "Los conteos del inventario se actualizaron al estado corregido y la celda de la sección 124 los verifica contra los CSV reales; el anexo A (sección 128) y su verificación (sección 129) cubren la lista completa.",
    )
    reemplazar_en(
        cells,
        "Carga con las funciones embebidas",
        "Carga con `src.data` (módulo congelado en la Fase 0), validación de filas, nulos, ids y dominios;",
        "Carga con las funciones embebidas (replican `src.data`, módulo congelado en la Fase 0), validación de filas, nulos, ids y dominios;",
    )
    reemplazar_en(
        cells,
        "Verificación paralela de las métricas OOF",
        "Verificación paralela de las métricas OOF guardadas en `features/oof_mod03b.npz` con `src.evaluation`.",
        "Verificación paralela de las métricas OOF guardadas en `features/oof_mod03b.npz` con las funciones embebidas.",
    )
    reemplazar_en(
        cells,
        "Límite del entregable",
        "**Límite del entregable.** El notebook compila celda a celda, pero no se ha ejecutado end-to-end en este estado (es pesado y arrastra artefactos ausentes preexistentes, como `features/v2_meta_oof.npz`).",
        "**Límite del entregable.** El notebook compila celda a celda, pero no se ha ejecutado end-to-end en este estado (es pesado y arrastra artefactos ausentes preexistentes, como `features/v2_meta_oof.npz`). Actualización del 2026-10-03: las celdas clave (configuración, definiciones embebidas, datos, métricas, reproducibilidad y verificaciones de inventario y completitud) se ejecutaron en el smoke test; la ejecución end-to-end completa sigue pendiente por coste.",
    )

    # 5. Celda de verificacion de la seccion 124: estado nuevo.
    indice_verif = buscar(cells, MARCA_VERIF_INVENTARIO)
    asignar_texto(cells[indice_verif], CODIGO_VERIF_INVENTARIO)
    cells[indice_verif]["outputs"] = []
    cells[indice_verif]["execution_count"] = None

    # 6. Limpieza de simbolos en todas las celdas.
    for celda in cells:
        asignar_texto(celda, limpiar_texto(texto_de(celda)))

    # 7. Secciones nuevas: 127 (P84), 128 (anexo) y 129 (verificacion).
    def es_gestionada(celda: dict) -> bool:
        fuente = texto_de(celda)
        return any(fuente.startswith(marca) for marca in (MARCA_P84, MARCA_ANEXO, MARCA_VERIF_ANEXO)) or (
            MARCA_CODIGO_ANEXO in fuente
        )

    posiciones = [i for i, c in enumerate(cells) if es_gestionada(c)]
    indice_insercion = min(posiciones) if posiciones else len(cells)
    while any(es_gestionada(c) for c in cells):
        cells.pop(next(i for i, c in enumerate(cells) if es_gestionada(c)))

    registro, inventario = cargar_filas()
    nuevas = [
        nueva_celda_markdown(construir_seccion_p84()),
        nueva_celda_markdown(construir_anexo(registro, inventario)),
        nueva_celda_markdown(construir_verificacion_anexo()),
        nueva_celda_codigo(CODIGO_VERIF_ANEXO),
    ]
    cells[indice_insercion:indice_insercion] = nuevas

    # 8. Validaciones y guardado.
    validar_codigo(cells)
    validar_caracteres(cells)

    ids_union = sorted(
        {fila["experiment_id"].strip() for fila in registro}
        | {fila["id"].strip() for fila in inventario}
    )
    texto = "\n".join(texto_de(c) for c in cells)
    faltan = [
        identificador
        for identificador in ids_union
        if re.search(r"(?<![\w-])" + re.escape(identificador) + r"(?![\w-])", texto) is None
    ]
    if faltan:
        raise SystemExit(f"Faltan {len(faltan)} identificadores en el notebook: {faltan[:10]}")

    notebook["cells"] = cells
    RUTA_NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )

    print("Notebook actualizado:")
    print(f"  celdas antes: {celdas_antes} | despues: {len(cells)}")
    print(f"  registro: {len(registro)} | inventario: {len(inventario)} | union: {len(ids_union)}")
    print(f"  identificadores presentes: {len(ids_union)} de {len(ids_union)}")
    print("  celdas de codigo compiladas: OK | caracteres permitidos: OK")


if __name__ == "__main__":
    main()
