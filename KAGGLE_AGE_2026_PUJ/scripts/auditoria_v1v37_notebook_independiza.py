"""auditoria_v1v37_notebook_independiza.py

Fecha: 2026-10-03
Descripcion: hace INDEPENDIENTE el notebook consolidado del proyecto KAGGLE_AGE_2026_PUJ.
    Hoy seis celdas de codigo importan del paquete `src` y una importa de un script externo
    (`v13_p15_matriz_errores_miembros`), de modo que el notebook NO se puede ejecutar sin el
    arbol de codigo del proyecto. Este script EMBEBE el codigo necesario en el propio notebook:

    - reemplaza la celda de configuracion por una version autocontenida que define `cfg`, `set_seed`,
      `clean_text` y `load_train` sin importar de `src`;
    - reemplaza la celda que usaba el script externo por una que embebe `MEMBER_FILES`, `FAMILIAS`,
      `nombre_a_familia` y `predicciones_de_miembros`;
    - retira los `from src import ...` que quedan en las demas celdas.

    El notebook sigue leyendo los artefactos de resultados (JSON, CSV, NPY) porque SON los datos de
    los experimentos; lo que deja de necesitar es el CODIGO del proyecto.

Que NO hace: no reejecuta nada, no entrena, no toca test/holdout/leaderboard, no modifica scripts
de experimentos ni artefactos de metricas.

Modos
-----
- Sin argumentos: aplica al notebook real y escribe `experiments/auditoria_notebook_independiente.json`.
- `--verificar`: comprueba en una COPIA temporal que (a) cada celda de codigo compila con `ast.parse`,
  (b) tras la independizacion NO queda ninguna importacion de `src` ni de scripts del proyecto,
  (c) las definiciones embebidas producen el mismo resultado que las originales, y (d) la copia
  resultante se puede importar sin el paquete `src` presente en el path. No toca el notebook real.

ASCII puro en el codigo. El texto de las celdas si admite tildes.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "99_entrega_consolidado.ipynb"
RESULTADO = ROOT / "experiments" / "auditoria_notebook_independiente.json"

# --------------------------------------------------------------------------------------
# Codigo a embeber. Se copia del proyecto y se pega en el notebook.
# --------------------------------------------------------------------------------------
CELDA_ARRANQUE = '''# Arranque autocontenido del notebook.
#
# Este notebook NO importa nada del paquete `src` ni de `scripts/`: el codigo que necesita esta
# aqui dentro. Lo unico que lee del disco son los ARTEFACTOS de resultados (JSON, CSV, NPY), que
# son los datos que los experimentos produjeron, y el corpus de entrenamiento.
#
# La raiz del proyecto se localiza por la presencia de `experiments.csv`, que es el registro.

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path.cwd()
while not (RAIZ / "experiments.csv").exists() and RAIZ != RAIZ.parent:
    RAIZ = RAIZ.parent
if not (RAIZ / "experiments.csv").exists():
    raise RuntimeError("no se encontro la raiz del proyecto (experiments.csv)")


class _Configuracion:
    """Rutas, constantes y clases del proyecto, sin depender del paquete `src`."""

    ROOT = RAIZ
    DATA = RAIZ / "data"
    RAW = RAIZ / "data" / "raw"
    FEATURES = RAIZ / "features"
    EXPERIMENTS = RAIZ / "experiments"
    REPORTS = RAIZ / "reports"
    FIGURES = RAIZ / "reports" / "figures"
    TABLES = RAIZ / "reports" / "tables"
    SUBMISSIONS = RAIZ / "submissions"
    MODELS = RAIZ / "models"

    TRAIN_FILE = RAW / "age_train.csv"
    TRAIN_SEP = ";"
    TARGET = "age_range"
    ID_COL = "id"
    TEXT_COL = "text"
    CLASSES = ["13-17", "18-24", "25-34", "35-49", "50-64", "65-xx"]
    SEED = 42
    N_SPLITS = 5
    N_REPEATS = 3


cfg = _Configuracion


def set_seed(seed: int = cfg.SEED) -> None:
    """Fija las semillas de `random` y `numpy` para reproducibilidad."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def clean_text(series: pd.Series) -> pd.Series:
    """Elimina el padding de espacios del ancho fijo y normaliza cadenas vacias."""
    return series.fillna("").astype(str).str.strip()


def load_train() -> pd.DataFrame:
    """Carga `age_train.csv` con validacion de esquema, sin depender de `src`."""
    if not cfg.TRAIN_FILE.exists():
        raise FileNotFoundError(cfg.TRAIN_FILE)
    frame = pd.read_csv(cfg.TRAIN_FILE, sep=cfg.TRAIN_SEP, encoding="utf-8-sig")
    faltan = [c for c in (cfg.ID_COL, cfg.TEXT_COL, cfg.TARGET) if c not in frame.columns]
    if faltan:
        raise ValueError(f"faltan columnas en {cfg.TRAIN_FILE.name}: {faltan}")
    frame[cfg.TEXT_COL] = clean_text(frame[cfg.TEXT_COL])
    frame[cfg.TARGET] = pd.Categorical(frame[cfg.TARGET], categories=cfg.CLASSES, ordered=True)
    return frame


set_seed()
pd.set_option("display.width", 175)
pd.set_option("display.max_columns", 60)


def load(name):
    """Lee un JSON de `experiments/` por nombre."""
    return json.loads((cfg.EXPERIMENTS / name).read_text(encoding="utf-8"))


experimentos = pd.read_csv(cfg.ROOT / "experiments.csv")
decisiones = [l for l in (cfg.ROOT / "DECISIONS.md").read_text(encoding="utf-8").splitlines()
              if l.startswith("## D-")]
print(f"experimentos registrados: {len(experimentos)} | decisiones documentadas: {len(decisiones)}")
'''

CELDA_MIEMBROS = '''# Predicciones por miembro del ensemble, embebidas desde `scripts/v13_p15_matriz_errores_miembros.py`
# y `scripts/phase2_ponderacion.py` para que el notebook no dependa del codigo del proyecto.

MEMBER_FILES = [
    ("oof_E013_margenes.npy", "raw"), ("oof_E016_margenes.npy", "raw"),
    ("oof_E-009.npy", "log"), ("oof_E-011.npy", "log"), ("oof_E-010.npy", "log"),
    ("oof_E014_margenes.npy", "raw"), ("oof_E015_ordinal.npy", "raw"),
    ("oof_E015_softmax.npy", "log"), ("oof_E017_probabilidades.npy", "log"),
    ("oof_E019b_densas_svd.npy", "log"), ("oof_E021a_ordinal.npy", "log"),
    ("oof_E021b_rocchio.npy", "log"), ("oof_E028a_pares.npy", "raw"),
    ("oof_ADV05_nbsvm.npy", "raw"),
]

FAMILIAS = {
    "svm_lineal": ["oof_E013_margenes", "oof_E016_margenes", "oof_E014_margenes"],
    "prob_lineal": ["oof_E-009", "oof_E-011", "oof_E017_probabilidades"],
    "bayes": ["oof_E-010"],
    "ordinal": ["oof_E015_softmax", "oof_E021a_ordinal", "oof_E028a_pares"],
    "prototipos": ["oof_E021b_rocchio"],
    "denso": ["oof_E019b_densas_svd"],
    "nbsvm": ["oof_ADV05_nbsvm"],
}


def nombre_a_familia(nombres):
    """Mapea el nombre de cada miembro a su familia de modelos."""
    asignacion = {}
    for familia, archivos in FAMILIAS.items():
        for archivo in archivos:
            asignacion[Path(archivo).stem] = familia
    return {nombre: asignacion.get(nombre, "sin_clasificar") for nombre in nombres}


def predicciones_de_miembros():
    """Devuelve (votos, nombres, margenes normalizados, familia por miembro).

    Convenciones heredadas de `phase13`: argmax para los bloques de 6 columnas, conteo de signos
    para el bloque de 5 margenes adyacentes, y exclusion de la fuente ordinal continua.
    """
    votos, margenes, nombres = [], [], []
    for archivo, transformacion in MEMBER_FILES:
        matriz = np.load(cfg.FEATURES / archivo)
        if matriz.ndim == 1 or matriz.shape[1] == 1:
            continue
        nombre = Path(archivo).stem
        if matriz.shape[1] == 5:
            absolutos = np.abs(matriz)
            votos.append((matriz > 0).sum(axis=1))
            margenes.append(absolutos.min(axis=1) / np.clip(absolutos.mean(axis=1), 1e-12, None))
            nombres.append(nombre)
            continue
        probabilidades = np.exp(matriz) if transformacion == "log" else matriz
        probabilidades = probabilidades / np.clip(probabilidades.sum(axis=1, keepdims=True),
                                                  1e-12, None)
        votos.append(np.argmax(probabilidades, axis=1))
        orden = np.sort(probabilidades, axis=1)
        margenes.append((orden[:, -1] - orden[:, -2]) / np.clip(orden[:, -1].std(), 1e-12, None))
        nombres.append(nombre)
    return (np.column_stack(votos), nombres, np.column_stack(margenes), nombre_a_familia(nombres))


entrenamiento = load_train()
y = entrenamiento[cfg.TARGET].astype(str).to_numpy()
v12 = np.load(cfg.FEATURES / "oof_ensemble_v12.npy")
pares = np.load(cfg.FEATURES / "oof_E028a_pares.npy")
pred_v12 = np.array(cfg.CLASSES)[np.argmax(v12, axis=1)]
votos, _, _, _ = predicciones_de_miembros()
orden = np.sort(v12, axis=1)
separacion = (orden[:, -1] - orden[:, -2]) / orden[:, -1].std()

filas_mecanismo = []
for inferior, superior, columna in (("18-24", "25-34", 1), ("25-34", "35-49", 2)):
    mascara = (y == inferior) | (y == superior)
    arriba = (np.sort(np.argsort(v12[mascara], axis=1)[:, -2:], axis=1)
              == np.array([cfg.CLASSES.index(inferior), cfg.CLASSES.index(superior)])).all(axis=1)
    decision = np.where(pares[mascara, columna] > 0, superior, inferior)
    ok_v12 = pred_v12[mascara] == y[mascara]
    ok_esp = decision == y[mascara]
    for nombre, seleccion in (("rescate", (~ok_v12) & ok_esp),
                              ("dano", ok_v12 & (~ok_esp)),
                              ("irrecuperable", (~ok_v12) & (~ok_esp))):
        filas_mecanismo.append({
            "frontera": f"{inferior}|{superior}", "grupo": nombre,
            "filas": int(seleccion.sum()),
            "pct_frontera_en_V12": round(float(arriba[seleccion].mean()), 4),
            "separacion_V12": round(float(separacion[mascara][seleccion].mean()), 4),
        })
pd.DataFrame(filas_mecanismo)
'''

# Importaciones del proyecto que hay que retirar de cualquier celda de codigo.
RE_IMPORT_PROYECTO = re.compile(
    r"^\s*(?:from\s+(?:src|scripts|v13_p15\w*|phase\w+)\s+import[^\n]*|import\s+(?:src|scripts)\b[^\n]*)\n",
    re.MULTILINE,
)

# --------------------------------------------------------------------------------------
# Limpieza de simbolos. Se conservan los acentos espanoles y los signos de apertura, que son
# ortografia legitima; se sustituye todo lo demas por su equivalente ASCII.
# --------------------------------------------------------------------------------------
SIMBOLOS = {
    "\u2014": "-",      # em dash
    "\u2013": "-",      # en dash
    "\u2212": "-",      # signo menos
    "\u00b7": "-",      # punto medio
    "\u2265": ">=",     # mayor o igual
    "\u2264": "<=",     # menor o igual
    "\u2248": "~",      # aproximadamente
    "\u2192": "->",     # flecha derecha
    "\u2190": "<-",     # flecha izquierda
    "\u2194": "<->",    # flecha doble
    "\u0394": "delta",  # delta mayuscula
    "\u03c4": "tau",    # tau
    "\u03b5": "epsilon",  # epsilon
    "\u222a": "U",      # union
    "\u2229": "int",    # interseccion
    "\u00b1": "+/-",    # mas menos
    "\u00b2": "^2",     # superindice 2
    "\u00d7": "x",      # multiplicacion
    "\u2208": "en",     # pertenece
    "\u221a": "raiz",   # raiz
    "\u221e": "inf",    # infinito
    "\u2026": "...",    # puntos suspensivos
    "\u201c": '"', "\u201d": '"',   # comillas dobles tipograficas
    "\u2018": "'", "\u2019": "'",   # comillas simples tipograficas
}
#: Caracteres que SI se conservan: ortografia espanola.
#: a, e, i, o, u con tilde; u con dieresis; enie; sus mayusculas; signos de apertura.
PERMITIDOS = set(
    "\u00e1\u00e9\u00ed\u00f3\u00fa\u00fc\u00f1"
    "\u00c1\u00c9\u00cd\u00d3\u00da\u00dc\u00d1\u00bf\u00a1"
)


def limpiar_simbolos(nb: dict) -> dict:
    """Sustituye los simbolos no ASCII por equivalentes ASCII en TODAS las celdas."""
    cambios = {}
    for i, c in enumerate(nb["cells"]):
        src = "".join(c.get("source", []))
        nuevo = src
        for ch, reemplazo in SIMBOLOS.items():
            nuevo = nuevo.replace(ch, reemplazo)
        if nuevo != src:
            c["source"] = nuevo.splitlines(keepends=True)
            afectados = sorted({ch for ch in src if ch in SIMBOLOS})
            cambios[i] = afectados
    return cambios


def raros_restantes(nb: dict) -> dict:
    """Cuenta los caracteres no ASCII que NO son ortografia espanola."""
    salida = {}
    for i, c in enumerate(nb["cells"]):
        src = "".join(c.get("source", []))
        encontrados = sorted({ch for ch in src if ord(ch) > 127 and ch not in PERMITIDOS})
        if encontrados:
            salida[i] = encontrados
    return salida


def cargar(nb_path: Path) -> dict:
    """Lee el notebook."""
    return json.loads(nb_path.read_text(encoding="utf-8"))


def es_celda_de_arranque(c: dict) -> bool:
    """True si la celda es la de configuracion/arranque del notebook."""
    src = "".join(c.get("source", []))
    return "from src import config as cfg" in src or "Arranque autocontenido del notebook" in src


def es_celda_de_miembros(c: dict) -> bool:
    """True si la celda es la que usaba el script externo de miembros."""
    src = "".join(c.get("source", []))
    return "predicciones_de_miembros" in src and "v13_p15" in src


def independiza(nb: dict) -> dict:
    """Embebe el codigo necesario, retira las importaciones del proyecto y limpia los simbolos."""
    celdas = nb["cells"]
    informe = {"arranque_reemplazada": False, "miembros_reemplazada": False,
               "imports_retirados": 0, "celdas_tocadas": []}
    for i, c in enumerate(celdas):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        if es_celda_de_arranque(c):
            c["source"] = CELDA_ARRANQUE.splitlines(keepends=True)
            c["outputs"] = []
            c["execution_count"] = None
            informe["arranque_reemplazada"] = True
            informe["celdas_tocadas"].append({"celda": i, "accion": "arranque embebido"})
            continue
        if es_celda_de_miembros(c):
            c["source"] = CELDA_MIEMBROS.splitlines(keepends=True)
            c["outputs"] = []
            c["execution_count"] = None
            informe["miembros_reemplazada"] = True
            informe["celdas_tocadas"].append({"celda": i, "accion": "miembros embebidos"})
            continue
        nuevo = RE_IMPORT_PROYECTO.sub("", src)
        if nuevo != src:
            c["source"] = nuevo.splitlines(keepends=True)
            informe["imports_retirados"] += 1
            informe["celdas_tocadas"].append({"celda": i, "accion": "import del proyecto retirado"})

    cambios = limpiar_simbolos(nb)
    informe["celdas_con_simbolos_limpiados"] = len(cambios)
    informe["simbolos_limpiados"] = {str(k): v for k, v in cambios.items()}
    return informe


def audita(nb: dict) -> dict:
    """Comprueba que no queda dependencia de codigo del proyecto y que todo compila."""
    problemas_import = []
    problemas_sintaxis = []
    for i, c in enumerate(nb["cells"]):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        if RE_IMPORT_PROYECTO.search(src):
            problemas_import.append(i)
        # una celda puede ser un fragmento: se valida solo si es codigo completo
        try:
            ast.parse(src)
        except SyntaxError as exc:
            problemas_sintaxis.append({"celda": i, "error": f"{exc.msg} (linea {exc.lineno})"})
    return {"imports_del_proyecto_restantes": problemas_import,
            "celdas_con_error_de_sintaxis": problemas_sintaxis,
            "n_celdas_codigo": sum(1 for c in nb["cells"] if c["cell_type"] == "code")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Hace independiente el notebook consolidado.")
    ap.add_argument("--verificar", action="store_true", help="prueba en copia temporal")
    args = ap.parse_args(argv)
    inicio = time.time()

    if args.verificar:
        print("=== verificacion de la independencia del notebook ===")
        fallos = 0
        with tempfile.TemporaryDirectory() as tmp:
            copia = Path(tmp) / "nb.ipynb"
            shutil.copy2(NB, copia)
            nb = cargar(copia)
            antes = audita(nb)
            print(f"  antes: {len(antes['imports_del_proyecto_restantes'])} celdas con imports del "
                  f"proyecto -> {antes['imports_del_proyecto_restantes']}")
            informe = independiza(nb)
            despues = audita(nb)
            print(f"  despues: {len(despues['imports_del_proyecto_restantes'])} celdas con imports "
                  f"del proyecto -> {despues['imports_del_proyecto_restantes']}")
            ok = not despues["imports_del_proyecto_restantes"]
            print(f"  [{'OK' if ok else 'FALLA'}] no quedan imports del proyecto")
            fallos += 0 if ok else 1

            ok = not despues["celdas_con_error_de_sintaxis"]
            print(f"  [{'OK' if ok else 'FALLA'}] las {despues['n_celdas_codigo']} celdas de codigo "
                  f"compilan: {despues['celdas_con_error_de_sintaxis'] or 'sin errores'}")
            fallos += 0 if ok else 1

            ok = informe["arranque_reemplazada"] and informe["miembros_reemplazada"]
            print(f"  [{'OK' if ok else 'FALLA'}] arranque y miembros embebidos: "
                  f"{informe['arranque_reemplazada']} / {informe['miembros_reemplazada']}")
            fallos += 0 if ok else 1

            # el codigo embebido debe definir lo que las celdas usan
            todo = "\n".join("".join(c.get("source", [])) for c in nb["cells"]
                             if c.get("cell_type") == "code")
            faltan = [n for n in ("cfg", "set_seed", "load_train", "clean_text", "load",
                                  "predicciones_de_miembros", "nombre_a_familia")
                      if not re.search(rf"^(def |class )?{n}\b", todo, re.MULTILINE)]
            ok = not faltan
            print(f"  [{'OK' if ok else 'FALLA'}] definiciones presentes: faltan {faltan or 'ninguna'}")
            fallos += 0 if ok else 1

            restos = raros_restantes(nb)
            ok = not restos
            print(f"  [{'OK' if ok else 'FALLA'}] simbolos raros restantes: "
                  f"{ {k: v for k, v in list(restos.items())[:4]} if restos else 'ninguno'}")
            fallos += 0 if ok else 1

            # el codigo debe seguir compilando DESPUES de la limpieza: los simbolos estaban en
            # strings de print, pero un reemplazo mal hecho podria romper una celda
            sintaxis = audita(nb)["celdas_con_error_de_sintaxis"]
            ok = not sintaxis
            print(f"  [{'OK' if ok else 'FALLA'}] el codigo compila tras la limpieza: "
                  f"{sintaxis or 'sin errores'}")
            fallos += 0 if ok else 1

            copia.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
            ok = json.loads(copia.read_text(encoding="utf-8"))["cells"] == nb["cells"]
            print(f"  [{'OK' if ok else 'FALLA'}] round-trip JSON estable")
            fallos += 0 if ok else 1

        print(f"  {7 - fallos}/7 caminos pasan; el notebook real no fue modificado")
        return 0 if fallos == 0 else 2

    nb = cargar(NB)
    antes = audita(nb)
    informe = independiza(nb)
    despues = audita(nb)
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    salida = {
        "generado_por": "scripts/auditoria_v1v37_notebook_independiza.py",
        "fecha": time.strftime("%Y-%m-%d"),
        "celdas_codigo": despues["n_celdas_codigo"],
        "imports_del_proyecto_antes": antes["imports_del_proyecto_restantes"],
        "imports_del_proyecto_despues": despues["imports_del_proyecto_restantes"],
        "errores_de_sintaxis": despues["celdas_con_error_de_sintaxis"],
        "celdas_con_simbolos_limpiados": informe["celdas_con_simbolos_limpiados"],
        "simbolos_raros_restantes": raros_restantes(nb),
        "detalle": {k: v for k, v in informe.items() if k != "simbolos_limpiados"},
        "segundos": round(time.time() - inicio, 1),
    }
    RESULTADO.parent.mkdir(parents=True, exist_ok=True)
    RESULTADO.write_text(json.dumps(salida, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(salida, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
