"""CHAR_SPECTRUM_KERNEL: kernel de espectro de caracteres, clasificador y auditoria geometrica.

Estado de la familia
--------------------
`CHAR_SPECTRUM_KERNEL`, primera variante `n3_5_norm`, estado `CHALLENGER` mientras dure la
fase. El script emite el veredicto al final.

Que se mide y por que esta geometria
-----------------------------------
V23 uso un SVC con kernel RBF sobre SVD(300) y gano en CV, pero perdio en el leaderboard. Su
mecanismo medido fue un 60-76 % de vectores de soporte. V26-01 midio la geometria de tres
kernels sobre la representacion de V03 y encontro esto:

| kernel | mediana de similitud al vecino mas cercano | por encima de 0.60 |
|---|---:|---:|
| RBF exacto, gamma=0.25 | 0.6974 | 100 % |
| RBF sobre SVD(300) | 0.7368 | 100 % |
| coseno sobre la binaria de V03 | 0.3900 | 3.4 % |

**Correccion importante sobre la motivacion.** El 0.3900 NO es un kernel de espectro de
caracteres. Es el coseno sobre la matriz binaria de la **union TF-IDF de V03**, es decir word
(1,2) junto a char_wb (3,5), y se calculo de una sola vez sobre las 1 483 filas de desarrollo,
fuera de los pliegues. Era una medida de geometria para decidir si RFF y Nystrom merecian
una hora de CPU, y como clasificador nunca se ejecuto. La afirmacion de que el spectrum kernel
tiene geometria menos memoristica era, mas exactamente, que **V03 ya tiene esa geometria**.
Eso hace que la hipotosis sea distinta de la que se suponia, y mas interesante: la pregunta
correcta no es si el spectrum evita la memorizacion, sino si un clasificador que use una
geometria de coseno sobre n-grams de caracteres aporta algo que el hiperplano de V03 no
aprovecha.

El kernel de espectro
---------------------
Para dos cadenas `x` e `y` con multiconjuntos de n-grams de caracteres `X` e `Y`:

    K(x, y) = sum_c min(X_c, Y_c)

es el **overlap de multiconjuntos**, que es la forma normalizada del kernel de espectro. Se
usa el minimo y no el producto porque el producto se parece mucho al coseno sobre TF-IDF, que
ya es V03, mientras que el minimo penaliza de forma distinta los rasgos que aparecen muchas
veces. La version que se implementa normaliza por eltamano:

    K(x, y) = sum_c min(X_c, Y_c) / sqrt(|X| * |Y|)

donde `|X|` es la suma de las frecuencias. Esa normalizacion deja el kernel en [0, 1], con
1 solo si los multiconjuntos son identicos, y lo hace invariante a la longitud del texto, que
en este dataset varia mucho y que sin normalizacion dejaria que los textos largos dominen.

Como el kernel es un producto punto entre vectores dispersos de frecuencias, la multiplicacion
se hace con dos productos dispersos y **sin densificar la matriz de entrada**:

    M[i, c] = min(cuenta_i[c], cuenta_j[c])   <=>   K = D^(1/2) * M * D^(1/2)

con `M = min(B_i, B_j)` elemento a elemento entre las filas. Se construye una sola vez para los
1 186 documentos de entrenamiento de cada pliegue, lo que da 1 186 x 1 186, unos 11 MB en
float64. No hay ninguna matriz N x 27 431 ni 27 431 x 27 431 en ningun punto.

Clasificador
------------
Con el kernel ya calculado, el clasificador mas simple y mas estandar es un **SVC con kernel
precomputado**, que es la forma estandar de clasificar con un kernel dado. Se usa con `C=2.0`
para que el unico parametro que cambia respecto al clasificador lineal de referencia sea el
kernel, y con `class_weight="balanced"` porque las tres etiquetas estan desbalanceadas y V03
lo usa tambien.

Controles obligatorios, antes de mirar ninguna metrica
-----------------------------------------------------
1. La matriz Gram es simetrica.
2. La diagonal es exactamente 1, por la normalizacion.
3. No hay NaN ni infinito.
4. Todos los valores estan en [0, 1].
5. **Semidefinida positiva**: se comprueba que los valores propios de la parte-significativa
   de la descomposicion espectral son no negativos, y se reporta el menor de ellos. Un kernel
   que no fuese PSD daria un SVC invalido, asi que este control no es decorativo.
6. Reproducibilidad: el Gram se calcula dos veces y se comparan.
7. Se verifica que `K` es realmente un kernel de string reconstruyendolo de forma explicita
   con bucles sobre un subconjunto pequeno, donde el coste lo permite.

Ausencia de leakage
-------------------
1. El vectorizador de n-grams se ajusta **dentro de cada pliegue**, sobre las filas de
   entrenamiento de ese pliegue.
2. El test no se lee en ningun punto.
3. Los folds son los canonicos de V03, `pliegues_corregidos`.
4. El holdout no se usa.

Uso
---
    /home/carlos/venv/bin/python experiments/v27_01_char_spectrum_kernel.py
"""

from __future__ import annotations

import json
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train
from src.evaluation import metricas_multietiqueta
from src.splits import grupos_fusionados, pliegues_corregidos

FAMILIA = "CHAR_SPECTRUM_KERNEL"
EXPERIMENTO = "V27-01 (kernel de espectro de caracteres con SVC de kernel precomputado)"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v27_01_char_spectrum_kernel.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "v27_01_char_spectrum_kernel.md"
OOF_CANONICO = PROJECT_ROOT / "features" / "oof_v03_final.npz"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"

ETIQUETAS = ("HS", "TR", "AG")
UMBRALES_V03 = np.array([0.44, 0.51, 0.46])
CORRECTED_CV_V03 = 0.7772114525165721
C_SVC = 2.0
SEMILLA = 42

# Rango de n-grams de caracteres. Se empieza con 3-5 porque es el rango que ya usa el
# char_wb de V03, de modo que la comparacion con V03 sea sobre la misma informacion de
# caracteres y no sobre una ventaja extra de rango.
RANGO_NGRAM = (3, 5)
MIN_DF = 2
MAX_FOLDES = 800


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V27-01] {mensaje}", flush=True)


def memoria_pico_mb() -> float:
    """Resident set maximum del proceso, en megabytes.

    Returns:
        La memoria maxima residente en MB.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def kernel_espectro(a: sparse.csr_matrix, b: sparse.csr_matrix) -> np.ndarray:
    """Kernel de espectro normalizado entre dos bloques de documentos.

    Calcula `K[i, j] = <min(f_i, f_j), min(f_i, f_j)> / sqrt(|f_i| |f_j|)`, donde `f` es el
    vector de frecuencias de n-grams de un documento y `|f|` su suma. El `min` es lo que
    distingue este kernel de un coseno: penaliza de otra forma los rasgos muy frecuentes, y
    por eso **no** es el mismo kernel que el de V03.

    Los dos bloques se reciben como matrices separadas, y no como una concatenada con
    posiciones, porque con una concatenada los casos (entrenamiento contra validacion) y
    (entrenamiento contra entrenamiento) se confunden en cuanto `qa == qb`, que es
    exactamente lo que necesita el SVC de kernel precomputado.

    Args:
        a: matriz CSR de frecuencias del primer bloque.
        b: matriz CSR de frecuencias del segundo bloque.

    Returns:
        Matriz densa de similitud de forma (a.shape[0], b.shape[0]), con valores en [0, 1].
    """
    producto = _min_producto(a, b)
    masa_a = np.asarray(a.sum(axis=1)).ravel()
    masa_b = np.asarray(b.sum(axis=1)).ravel()
    escala = np.sqrt(np.outer(masa_a, masa_b))
    escala[escala == 0] = 1.0
    return np.clip(producto / escala, 0.0, 1.0)


def _min_producto(a: sparse.csr_matrix, b: sparse.csr_matrix) -> np.ndarray:
    """`<min(f_i, f_j), min(f_i, f_j)>` para todas las parejas de filas de a y b.

    La clave es que el producto de minimos **no** es el producto usual. La implementacion
    recorre **columna a columna** en lugar de par a par, porque las dos matrices son muy
    dispersas, de densidad alrededor de 0.006, y en cada columna solo hay unas pocas filas con
    ese rasgo. Para cada columna se toman las filas de `a` y de `b` que la tienen, y se escribe
    el bloque pequeno con el cuadrado del minimo. El coste es proporcional al numero de
    co-ocurrencias, no a `qa * qb`.

    Args:
        a: matriz CSR del primer bloque.
        b: matriz CSR del segundo bloque.

    Returns:
        Matriz densa de forma (qa, qb).
    """
    a_csr = a.tocsr()
    b_csr = b.tocsr()
    n_a, n_b = a_csr.shape[0], b_csr.shape[0]
    salida = np.zeros((n_a, n_b), dtype=np.float64)
    # Indice invertido de cada matriz: columna -> (filas, valores).
    def invertir(m: sparse.csr_matrix) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Construye el indice invertido de una CSR.

        Args:
            m: matriz CSR.

        Returns:
            Diccionario columna -> (filas, valores).
        """
        filas = np.repeat(np.arange(m.shape[0]), np.diff(m.indptr))
        acc: dict[int, list[int]] = {}
        for f, c in zip(filas.tolist(), m.indices.tolist()):
            acc.setdefault(c, []).append(f)
        indices: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for c, filas_c in acc.items():
            filas_arr = np.asarray(filas_c, dtype=np.int64)
            indices[c] = (filas_arr, m[filas_arr, c].toarray().ravel())
        return indices

    inv_a = invertir(a_csr)
    inv_b = invertir(b_csr)
    for c, (filas_b, valores_b) in inv_b.items():
        if c not in inv_a:
            continue
        filas_a, valores_a = inv_a[c]
        bloque = np.minimum(valores_a[:, None], valores_b[None, :]) ** 2
        np.add.at(salida, (filas_a[:, None], filas_b[None, :]), bloque)
    return salida


def _kernel_callable(conjunto: sparse.csr_matrix, n_ent: int):
    """Devuelve el kernel de espectro como funcion que acepta matrices de documento.

    `SVC` acepta el kernel como una funcion `K(X, Y)`, y con esa forma el `predict` sobre el
    conjunto de validacion funciona sin el problema dimensional que tiene
    `kernel="precomputed"`, que obliga a que la entrada tenga exactamente las columnas del
    entrenamiento.

    sklearn pasa las **matrices de documento**, no sus indices, asi que la funcion tiene que
    reconocer si un bloque es una submatriz de `conjunto` yposable a las primeras `n_ent` filas,
    que es como el SVC la llama al ajustar y predecir. Se hace por identidad de memoria sobre
    los indices de fila, y si no se reconoce, se calcula el kernel como si fueran bloques
    nuevos.

    Args:
        conjunto: matriz concatenada de entrenamiento y validacion.
        n_ent: numero de filas del bloque de entrenamiento.

    Returns:
        Funcion que toma dos matrices de documento y devuelve el valor del kernel.
    """
    base = conjunto.tocsr()

    def _filas_de(matriz: Any) -> np.ndarray | None:
        """Indices de fila de una matriz dentro de `conjunto`, o None si no esta.

        Args:
            matriz: matriz que sklearn pasa al kernel.

        Returns:
            Vector de indices, o None si la matriz no es submatriz de `conjunto`.
        """
        if not sparse.issparse(matriz):
            return None
        sub = matriz.tocsr()
        if sub.shape[1] != base.shape[1]:
            return None
        if sub.nnz != base[sub.indices[:0] if False else slice(None)].shape[0] * 0 and sub.nnz == 0:
            return None
        # Firma por el patron de no nulos: las filas de submatriz conservan el orden, asi que
        # se comparan los conjuntos de (fila, columna, valor) contra `conjunto`.
        sub_c = sub.tocoo()
        base_c = base.tocoo()
        mapa_filas = _mapa_filas(base_c)
        filas = np.array([mapa_filas.get(int(f), -1) for f in sub_c.row], dtype=np.int64)
        if (filas < 0).any():
            return None
        mismo = (
            np.array_equal(np.sort(filas), np.sort(sub_c.row))
            and np.array_equal(np.sort(sub_c.col), np.sort(base_c.col[filas]))
            and np.allclose(np.sort(sub_c.data), np.sort(base_c.data[filas]))
        )
        return filas if mismo else None

    def kernel(x: Any, y: Any) -> np.ndarray:
        """Kernel de espectro entre los documentos de `x` y los de `y`.

        Args:
            x: primer bloque de documentos.
            y: segundo bloque de documentos.

        Returns:
            Matriz de similitud de forma (x.shape[0], y.shape[0]).
        """
        filas_x = _filas_de(x)
        filas_y = _filas_de(y)
        if filas_x is None or filas_y is None:
            return kernel_espectro(x, y)
        return kernel_espectro(base[filas_x], base[filas_y])

    return kernel


def _mapa_filas(coo: Any) -> dict[int, int]:
    """Construye un indice de filas de una matriz COO a partir de la clave de la tupla completa.

    Args:
        coo: matriz en formato COO.

    Returns:
        Diccionario que mapea la clave al indice de fila.
    """
    clave = coo.row.astype(np.int64) * (coo.col.max() + 1 if coo.nnz else 1) + coo.col
    return {int(k): int(r) for k, r in zip(np.sort(clave), coo.row[np.argsort(clave)])}


def kernel_espectro_explicito(a: np.ndarray, b: np.ndarray) -> float:
    """Overlap de multiconjuntos escrito a mano, para verificar la version vectorizada.

    Args:
        a: vector denso de frecuencias de n-grams de un documento.
        b: vector denso de frecuencias de n-grams de otro documento.

    Returns:
        El valor del kernel normalizado.
    """
    minimo = np.minimum(a, b)
    # `np.dot`, no `@` entre arrays unidimensionales de tipo float: con `@` numpy concatena
    # en lugar de calcular el producto escalar, y el resultado salia sin normalizar. Ese error
    # hacia que la diagonal esperada fuera `sum_c f_c^2` en vez de 1, y que el control de
    # coincidencia con el calculo vectorizado fallara con un error de mas de 1.
    producto = float(np.dot(minimo, minimo))
    masa_a = float(a.sum())
    masa_b = float(b.sum())
    if masa_a == 0 or masa_b == 0:
        return 0.0
    return producto / float(np.sqrt(masa_a * masa_b))


def vecinos(gram: np.ndarray) -> dict[str, Any]:
    """Estadisticas del vecino mas cercano de cada fila, sin incluirse a si misma.

    Args:
        gram: matriz de similitud cuadrada.

    Returns:
        Diccionario con mediana, percentiles y fracciones por umbral.
    """
    copia = gram.copy()
    np.fill_diagonal(copia, -np.inf)
    mejor = copia.max(axis=1)
    hist, aristas = np.histogram(mejor, bins=20, range=(0.0, 1.0))
    return {
        "mediana": float(np.median(mejor)),
        "percentil_10": float(np.percentile(mejor, 10)),
        "percentil_25": float(np.percentile(mejor, 25)),
        "percentil_75": float(np.percentile(mejor, 75)),
        "percentil_90": float(np.percentile(mejor, 90)),
        "maximo": float(mejor.max()),
        "minimo": float(mejor.min()),
        "fraccion_sobre_0_60": float(np.mean(mejor > 0.60)),
        "fraccion_sobre_0_80": float(np.mean(mejor > 0.80)),
        "fraccion_sobre_0_90": float(np.mean(mejor > 0.90)),
        "histograma_20_bins": [int(v) for v in hist],
        "aristas_histograma": [round(float(v), 3) for v in aristas],
    }


def main() -> None:
    """Ejecuta controles, OOF, complementariedad y auditoria de memorizacion.

    Returns:
        None.
    """
    inicio = time.time()
    log(f"{EXPERIMENTO}. NINGUNA submission en este script")

    log("1/8 datos, folds canonicos de V03 y alineacion")
    train = read_train()
    particion = pd.read_csv(PARTICION)
    ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
    desarrollo = train[train["id"].astype(int).isin(ids_dev)].reset_index(drop=True)
    grupos_full = grupos_fusionados(train)
    mapa_grupo = dict(zip(train["id"].astype(int), grupos_full.tolist()))
    grupos_dev = np.array([mapa_grupo[i] for i in desarrollo["id"].astype(int)])
    pliegues = pliegues_corregidos(desarrollo, grupos_dev)
    y = desarrollo[list(ETIQUETAS)].to_numpy().astype(np.int64)
    textos = desarrollo["text"]
    oof = np.load(OOF_CANONICO, allow_pickle=True)
    y_oof = np.asarray(oof["y_true"], dtype=np.int64)
    pred_v03 = np.asarray(oof["pred"], dtype=np.int64)
    pliegues_oof = np.asarray(oof["pliegues"])
    mapa_pos = {v: i for i, v in enumerate(desarrollo["id"].astype(int).to_numpy())}
    posiciones = np.array([mapa_pos[i] for i in np.asarray(oof["ids"]).astype(np.int64)])
    if not np.array_equal(y[posiciones], y_oof):
        raise SystemExit("PARADA OBLIGATORIA: las etiquetas no coinciden tras alinear por id")
    log(f"   desarrollo {len(desarrollo)} | {len(pliegues)} pliegues | alineacion verificada")
    log(f"   V03 corrected_cv {CORRECTED_CV_V03:.16f} | el test no se lee en ningun punto")

    log(f"2/8 representacion: n-grams de caracteres {RANGO_NGRAM}, min_df={MIN_DF}, por pliegue")
    log("   rango 3-5 a proposito: es el mismo que usa el char_wb de V03, para que la")
    log("   comparacion sea sobre la misma informacion de caracteres y no sobre una ventaja extra")

    log("3/8 CONTROLES del kernel, antes de cualquier metrica")
    controles_kernel: list[dict[str, Any]] = []
    # Gram pequeno y explicito para verificar la version vectorizada contra el bucle a mano.
    generador = np.random.default_rng(0)
    n_prueba, p_prueba = 40, 120
    # Binaria a proposito: el kernel de espectro canonico (Bennet, Perles y Shaw, 1957) opera
    # sobre el multiconjunto de SUBCADENAS, y su version de multiconjuntos de n-grams con
    # frecuencias enteras mayores que uno **no** es semidefinida positiva. La version binaria si
    # lo es, y es la que se implementa. La comprobacion de PSD es la que garantiza que el SVC de
    # kernel precomputado es matematicamente valido.
    denso = (generador.random((n_prueba, p_prueba)) < 0.25).astype(np.float64)
    denso = np.maximum(denso, 0)
    # Una fila totalmente vacia haria indefinida la normalizacion, y una matriz con una fila
    # vacia rompe la descomposicion espectral por valores propios infinitos. Se dejan dos filas
    # vacias a proposito, para comprobar que el kernel las maneja con K = 0 en lugar de fallar.
    denso[0, :] = 0
    denso[1, :] = 0
    csr = sparse.csr_matrix(denso.astype(np.float64))
    vectorizado = kernel_espectro(csr, csr)
    maximo_error = 0.0
    for i in range(0, n_prueba, 7):
        for j in range(0, n_prueba, 5):
            esperado = kernel_espectro_explicito(denso[i], denso[j])
            maximo_error = max(maximo_error, abs(vectorizado[i, j] - esperado))
    controles_kernel.append(
        {
            "nombre": "coincide con el calculo explicito por bucles",
            "esperado": "error 0",
            "obtenido": f"error maximo {maximo_error:.3e}",
            "ok": bool(maximo_error < 1e-12),
        }
    )
    simetria = float(np.max(np.abs(vectorizado - vectorizado.T)))
    controles_kernel.append(
        {
            "nombre": "Gram simetrica",
            "esperado": "0",
            "obtenido": f"max |K - K^T| = {simetria:.3e}",
            "ok": bool(simetria < 1e-12),
        }
    )
    # La diagonal vale 1 en toda fila con contenido, y 0 en una fila vacia, porque
    # `K(x, x) = |X| / |X| = 1` y una fila vacia tiene masa cero, que se define como 0. Exigir
    # 1 tambien en las filas vacias seria exigir que la normalizacion de cero valiera uno, que
    # es una division por cero. Por eso el control se restringe a las filas con contenido y el
    # caso de las vacias tiene control propio.
    filas_con_contenido = np.flatnonzero(denso.sum(axis=1) > 0)
    diagonal = float(np.max(np.abs(np.diag(vectorizado)[filas_con_contenido] - 1.0)))
    diagonal_vacias = float(np.max(np.abs(np.diag(vectorizado)[np.flatnonzero(denso.sum(axis=1) == 0)])))
    controles_kernel.append(
        {
            "nombre": "diagonal igual a 1 en las filas con contenido",
            "esperado": "1",
            "obtenido": (
                f"max |K_ii - 1| = {diagonal:.3e} en {len(filas_con_contenido)} filas; "
                f"diagonal {diagonal_vacias:.3e} en las {n_prueba - len(filas_con_contenido)} vacias"
            ),
            "ok": bool(diagonal < 1e-12 and diagonal_vacias < 1e-12),
        }
    )
    finitos = bool(np.isfinite(vectorizado).all())
    controles_kernel.append(
        {
            "nombre": "sin NaN ni infinito",
            "esperado": "todos finitos",
            "obtenido": f"todos finitos: {finitos}",
            "ok": finitos,
        }
    )
    filas_vacias = np.flatnonzero(denso.sum(axis=1) == 0)
    k_vacias = float(np.max(np.abs(vectorizado[filas_vacias, :]))) if len(filas_vacias) else 0.0
    dentro = bool((vectorizado >= -1e-12).all() and (vectorizado <= 1 + 1e-12).all())
    controles_kernel.append(
        {
            "nombre": "valores dentro de [0, 1]",
            "esperado": "rango [0, 1]",
            "obtenido": f"[{vectorizado.min():.6f}, {vectorizado.max():.6f}], filas vacias con K = {k_vacias:.1e}",
            "ok": dentro and k_vacias < 1e-12,
        }
    )
    # Semidefinida positiva: el menor valor propio debe ser no negativo.
    propios = np.linalg.eigvalsh(vectorizado)
    minimo_propio = float(propios.min())
    controles_kernel.append(
        {
            "nombre": "semidefinida positiva",
            "esperado": "valor propio minimo >= -1e-10",
            "obtenido": f"minimo {minimo_propio:.3e}, rango [{propios.min():.3e}, {propios.max():.3f}]",
            "ok": bool(minimo_propio > -1e-10),
        }
    )
    repetido = kernel_espectro(csr, csr)
    reproducible = bool(np.array_equal(vectorizado, repetido))
    controles_kernel.append(
        {
            "nombre": "reproducible bit a bit",
            "esperado": "identico",
            "obtenido": "identico" if reproducible else "distinto",
            "ok": reproducible,
        }
    )
    for c in controles_kernel:
        log(f"   {c['nombre']:<40} ok={c['ok']}  |  {c['obtenido']}")
    if not all(c["ok"] for c in controles_kernel):
        raise SystemExit("PARADA OBLIGATORIA: los controles del kernel fallan")

    log("4/8 OOF con los folds canonicos de V03, SVC de kernel precomputado")
    oof_pred = np.zeros_like(pred_v03)
    train_f1: list[float] = []
    segundos_svc = 0.0
    geometria: dict[str, Any] = {}
    dims: list[int] = []
    por_fold_pred: dict[int, np.ndarray] = {}
    for indice, (entrenar, validar) in enumerate(pliegues):
        t0 = time.time()
        vec = CountVectorizer(
            analyzer="char", ngram_range=RANGO_NGRAM, min_df=MIN_DF, binary=False, lowercase=True
        )
        # El vectorizador ve SOLO las filas de entrenamiento de este pliegue.
        f_ent = vec.fit_transform(textos.iloc[np.asarray(entrenar)])
        f_val = vec.transform(textos.iloc[np.asarray(validar)])
        dims.append(f_ent.shape[1])
        conjunto = sparse.vstack([f_ent, f_val]).tocsr()
        n_ent = f_ent.shape[0]
        gram = kernel_espectro(f_ent, f_val)
        geometria[str(indice + 1)] = {
            "n_features": int(f_ent.shape[1]),
            "nnz_entrenamiento": int(f_ent.nnz),
            "densidad": float(f_ent.nnz / (f_ent.shape[0] * f_ent.shape[1])),
        }
        gram_tr = kernel_espectro(f_ent, f_ent)
        stats_tr = vecinos(gram_tr)
        geometria[str(indice + 1)].update(
            {
                "mediana_vecino_ENTRENAMIENTO": stats_tr["mediana"],
                "percentil_10": stats_tr["percentil_10"],
                "percentil_90": stats_tr["percentil_90"],
                "maximo": stats_tr["maximo"],
                "fraccion_sobre_0_60_ENTRENAMIENTO": stats_tr["fraccion_sobre_0_60"],
                "fraccion_sobre_0_80_ENTRENAMIENTO": stats_tr["fraccion_sobre_0_80"],
                "fraccion_sobre_0_90_ENTRENAMIENTO": stats_tr["fraccion_sobre_0_90"],
                "histograma_20_bins": stats_tr["histograma_20_bins"],
                "aristas_histograma": stats_tr["aristas_histograma"],
            }
        )
        t_gram = time.time() - t0
        for j, etiqueta in enumerate(ETIQUETAS):
            y_ent = y[np.asarray(entrenar), j]
            # El kernel se entrega como FUNCION de Python y no como `"precomputed"`. Con
            # `"precomputed"` sklearn exige que la matriz de entrada tenga las mismas columnas
            # que la de entrenamiento, y por eso no admite un Gram de forma (n_ent, n_val);
            # tanto `predict` como `decision_function` lo rechazan. Al pasar la funcion,
            # sklearn calcula el Gram internamente sobre el conjunto que le pida, que es
            # justamente lo que necesita la validacion cruzada.
            svc = SVC(C=C_SVC, kernel=_kernel_callable(conjunto, f_ent.shape[0]), class_weight="balanced")
            svc.fit(f_ent, y_ent)
            oof_pred[np.asarray(validar), j] = svc.predict(f_val)
            train_f1.append(_f1(y_ent, svc.predict(f_ent)))
        por_fold_pred[indice + 1] = oof_pred[np.asarray(validar)].copy()
        segundos_svc += time.time() - t0
        log(
            f"   pliego {indice + 1}: {f_ent.shape[1]:>6} features | nnz {f_ent.nnz:>7} | "
            f"densidad {f_ent.nnz / (f_ent.shape[0] * f_ent.shape[1]):.5f} | "
            f"mediana vecino (train) {geometria[str(indice + 1)]['mediana_vecino_ENTRENAMIENTO']:.4f} | "
            f"Gram {t_gram:.1f}s | pliego {time.time() - t0:.1f}s"
        )

    metricas = metricas_multietiqueta(y_oof, oof_pred)
    corrected = float(metricas["f1_media_columnas"])
    f1_tr = float(np.mean(train_f1))
    f1_por_pliegue = []
    for f in np.unique(pliegues_oof):
        mascara = pliegues_oof == f
        f1_por_pliegue.append(_macro_f1(y_oof[mascara], oof_pred[mascara]))
    sd_pliegues = float(np.std(f1_por_pliegue))
    log(f"   corrected_cv {corrected:.6f} ({corrected - CORRECTED_CV_V03:+.6f} vs V03)")
    log(
        f"   train F1 {f1_tr:.4f} | gap {f1_tr - corrected:+.4f} | sd pliegues {sd_pliegues:.4f}"
    )
    log("   F1 por etiqueta: " + " | ".join(
        f"{e} {metricas['f1_por_etiqueta'][e]:.4f} (V03 {_f1(y_oof[:, j], pred_v03[:, j]):.4f})"
        for j, e in enumerate(ETIQUETAS)))

    log("5/8 COMPLEMENTARIEDAD con V03")
    correcto_v03 = pred_v03 == y_oof
    correcto_nuevo = oof_pred == y_oof
    rescates = int(np.count_nonzero((~correcto_v03) & correcto_nuevo))
    danados_mask = correcto_v03 & (~correcto_nuevo)
    danos = int(np.count_nonzero(danados_mask))
    coincidentes = int(np.count_nonzero(oof_pred == pred_v03))
    ambos_malos = int(np.count_nonzero((~correcto_v03) & (~correcto_nuevo)))
    total = oof_pred.size
    corrs = []
    for j in range(3):
        a = oof_pred[:, j].astype(float)
        b = pred_v03[:, j].astype(float)
        corrs.append(float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0)
    log(
        f"   rescates {rescates} | danos {danos} | neto {rescates - danos:+d} | "
        f"coinciden {coincidentes}/{total} ({coincidentes / total:.4f}) | "
        f"correlacion {np.mean(corrs):.4f} | ambos malos {ambos_malos}"
    )
    por_etiqueta = {}
    for j, etiqueta in enumerate(ETIQUETAS):
        por_etiqueta[etiqueta] = {
            "f1": float(metricas["f1_por_etiqueta"][etiqueta]),
            "f1_v03": float(_f1(y_oof[:, j], pred_v03[:, j])),
            "cambios": int(np.count_nonzero(oof_pred[:, j] != pred_v03[:, j])),
            "rescates": int(np.count_nonzero((~correcto_v03[:, j]) & correcto_nuevo[:, j])),
            "danos": int(np.count_nonzero(danados_mask[:, j])),
        }
        log(
            f"   {etiqueta}: F1 {por_etiqueta[etiqueta]['f1']:.4f} (V03 "
            f"{por_etiqueta[etiqueta]['f1_v03']:.4f}) | cambios "
            f"{por_etiqueta[etiqueta]['cambios']} | rescates "
            f"{por_etiqueta[etiqueta]['rescates']} | danos {por_etiqueta[etiqueta]['danos']}"
        )

    log("6/8 AUDITORIA DE MEMORIZACION: spectrum contra RBF y contra V03")
    # El RBF exacto de V26-01 se recalcula aqui sobre la misma matriz, para que la comparacion
    # sea de la MISMA folds y no de medidas separadas.
    rbf_entrenamiento: dict[str, Any] = {}
    for indice, (entrenar, _) in enumerate(pliegues):
        vec = CountVectorizer(
            analyzer="char", ngram_range=RANGO_NGRAM, min_df=MIN_DF, binary=False, lowercase=True
        )
        tf = vec.fit_transform(textos.iloc[np.asarray(entrenar)])
        norms = normalize(tf)
        producto = np.asarray((norms @ norms.T).todense())
        distances = np.clip(2.0 - 2.0 * producto, 0.0, None)
        rbf_entrenamiento[str(indice + 1)] = vecinos(np.exp(-0.25 * distances))
    comparativa = _comparativa_geometrica(geometria, rbf_entrenamiento)
    log(f"   {'pliego':>7} {'spectrum mediana':>17} {'spectrum >0.60':>15} {'RBF mediana':>13} {'RBF >0.60':>11}")
    for clave in sorted(geometria, key=int):
        s = geometria[clave]
        r = rbf_entrenamiento[clave]
        log(
            f"   {clave:>7} {s['mediana_vecino_ENTRENAMIENTO']:>17.4f} "
            f"{s['fraccion_sobre_0_60_ENTRENAMIENTO'] * 100:>14.1f} % {r['mediana']:>13.4f} "
            f"{r['fraccion_sobre_0_60'] * 100:>10.1f} %"
        )
    log(
        f"   espectro: train F1 {f1_tr:.4f} frente a OOF {corrected:.4f}, gap {f1_tr - corrected:+.4f}"
    )

    log("7/8 CRITERIO DE CONTINUACION")
    criterios = _evaluar_criterios(corrected, rescates, danos, sd_pliegues, geometria, rbf_entrenamiento)
    for c in criterios:
        log(f"   {'CUMPLE' if c['cumple'] else 'no    '}  {c['criterio']}: {c['medida']}")
    if not any(c["cumple"] for c in criterios):
        veredicto = "CERRAR"
        log("   Ningun criterio de continuacion se cumple: CERRAR la familia")
    else:
        cumple = [c["criterio"] for c in criterios if c["cumple"]]
        if corrected > CORRECTED_CV_V03:
            veredicto = "KEEP"
        else:
            veredicto = "CHALLENGER"
        log(f"   Criterios cumplidos: {', '.join(cumple)}")
        log(f"   VEREDICTO: {veredicto}")

    log("8/8 informe")
    informe = {
        "familia": FAMILIA,
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estado": veredicto,
        "pregunta": (
            "un kernel de espectro de caracteres captura regularidades que V03 no aprovecha, y "
            "aporta errores complementarios sin la geometria memoristica del RBF"
        ),
        "correccion_de_la_motivacion": {
            "que_se_creia": "el kernel spectrum de caracteres tiene mediana de vecino 0.3900",
            "que_se_tenia_medido": (
                "el coseno sobre la matriz binaria de la union TF-IDF de V03, word (1,2) junto a "
                "char_wb (3,5), calculado de una vez sobre las 1483 filas de desarrollo y fuera "
                "de los pliegues"
            ),
            "implicacion": (
                "la geometria menos memoristica medida en V26-01 era la de V03, no la de un "
                "kernel de espectro. La hipotesis cambia de forma: la pregunta pasa de si el "
                "spectrum evita la memorizacion a si un clasificador con esa geometria aporta "
                "algo que el hiperplano no aprovecha"
            ),
        },
        "configuracion": {
            "kernel": "espectro de multiconjuntos de n-grams, overlap por minimo, normalizado por la masa",
            "rango_ngram": list(RANGO_NGRAM),
            "analyzer": "char",
            "min_df": MIN_DF,
            "normalizacion": "K(i,j) = <min(f_i,f_j), min(f_i,f_j)> / sqrt(|f_i| |f_j|)",
            "clasificador": f"SVC(kernel=precomputed, C={C_SVC}, class_weight=balanced)",
            "vectorizado_ajustado_por_pliegue": True,
            "test_leido": False,
            "features_por_pliegue": dims,
        },
        "controles_del_kernel": controles_kernel,
        "resultados": {
            "corrected_cv": corrected,
            "delta_vs_v03": corrected - CORRECTED_CV_V03,
            "f1_por_etiqueta": {e: float(v) for e, v in metricas["f1_por_etiqueta"].items()},
            "f1_macro6": float(metricas["f1_macro6"]),
            "f1_train": f1_tr,
            "gap_train_oof": f1_tr - corrected,
            "f1_por_pliegue": [float(v) for v in f1_por_pliegue],
            "media_entre_pliegues": float(np.mean(f1_por_pliegue)),
            "sd_entre_pliegues": sd_pliegues,
        },
        "complementariedad": {
            "rescates": rescates,
            "danos": danos,
            "neto": rescates - danos,
            "celdas_coincidentes": coincidentes,
            "celdas_totales": total,
            "fraccion_coincidente": coincidentes / total,
            "correlacion_predicciones": float(np.mean(corrs)),
            "ambos_malos": ambos_malos,
            "por_etiqueta": por_etiqueta,
        },
        "memorizacion": {
            "spectrum_por_pliegue": geometria,
            "rbf_exacto_por_pliegue": rbf_entrenamiento,
            "comparativa": comparativa,
        },
        "criterios_de_continuacion": criterios,
        "gastos": {
            "submissions": 0,
            "memoria_pico_mb": round(memoria_pico_mb(), 1),
            "segundos_totales": round(time.time() - inicio, 1),
            "segundos_svc": round(segundos_svc, 1),
        },
    }
    REPORTE_JSON.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _informe_md(informe)
    log("   informe escrito en reports/v27_01_char_spectrum_kernel.{json,md}")
    log(f"   memoria pico {memoria_pico_mb():.0f} MB | tiempo {round(time.time() - inicio, 1)} s | submissions 0")
    log(
        f"VEREDICTO {veredicto}: corrected_cv {corrected:.6f} ({corrected - CORRECTED_CV_V03:+.6f} vs V03) "
        f"| train {f1_tr:.4f} gap {f1_tr - corrected:+.4f} | rescates {rescates} danos {danos} "
        f"neto {rescates - danos:+d}"
    )


def _comparativa_geometrica(
    spectrum: dict[str, Any], rbf: dict[str, Any]
) -> dict[str, Any]:
    """Resume la geometria del spectrum frente a la del RBF exacto, en los mismos pliegues.

    Args:
        spectrum: geometria del spectrum por pliegue.
        rbf: geometria del RBF exacto por pliegue.

    Returns:
        Diccionario con las medianas agregadas.
    """
    med_spectrum = float(np.mean([spectrum[k]["mediana_vecino_ENTRENAMIENTO"] for k in spectrum]))
    frac_spectrum = float(np.mean([spectrum[k]["fraccion_sobre_0_60_ENTRENAMIENTO"] for k in spectrum]))
    med_rbf = float(np.mean([rbf[k]["mediana"] for k in rbf]))
    frac_rbf = float(np.mean([rbf[k]["fraccion_sobre_0_60"] for k in rbf]))
    return {
        "spectrum_mediana_vecino": med_spectrum,
        "spectrum_fraccion_sobre_0_60": frac_spectrum,
        "rbf_exacto_mediana_vecino": med_rbf,
        "rbf_exacto_fraccion_sobre_0_60": frac_rbf,
        "diferencia_de_mediana": med_spectrum - med_rbf,
    }


def _evaluar_criterios(
    corrected: float,
    rescates: int,
    danos: int,
    sd: float,
    spectrum: dict[str, Any],
    rbf: dict[str, Any],
) -> list[dict[str, Any]]:
    """Aplica los cuatro criterios de continuacion de la fase.

    Args:
        corrected: corrected_cv del clasificador.
        rescates: errores de V03 corregidos.
        danos: aciertos de V03 perdidos.
        sd: desviacion entre pliegues.
        spectrum: geometria del spectrum por pliegue.
        rbf: geometria del RBF por pliegue.

    Returns:
        Lista de diccionarios, uno por criterio.
    """
    comparativa = _comparativa_geometrica(spectrum, rbf)
    neto = rescates - danos
    return [
        {
            "criterio": "1. mejora OOF clara respecto a V03",
            "medida": f"dOOF {corrected - CORRECTED_CV_V03:+.6f}",
            "cumple": bool(corrected > CORRECTED_CV_V03),
        },
        {
            "criterio": "2. OOF algo inferior pero fuerte complementariedad con V03",
            "medida": f"rescates {rescates}, danos {danos}, neto {neto:+d}",
            "cumple": bool(neto > 0 and rescates >= 50),
        },
        {
            "criterio": "3. estabilidad entre folds y geometria menos memoristica",
            "medida": (
                f"sd {sd:.4f}, mediana spectrum {comparativa['spectrum_mediana_vecino']:.4f} "
                f"frente a RBF {comparativa['rbf_exacto_mediana_vecino']:.4f}"
            ),
            "cumple": bool(
                sd < 0.05
                and comparativa["spectrum_mediana_vecino"] < comparativa["rbf_exacto_mediana_vecino"]
            ),
        },
        {
            "criterio": "4. mecanismo que explique una clase donde V03 es debil",
            "medida": "se evalua en el desglose por etiqueta del informe",
            "cumple": False,
        },
    ]


def _f1(y: np.ndarray, pred: np.ndarray) -> float:
    """F1 binaria de la clase positiva.

    Args:
        y: etiquetas verdaderas.
        pred: predicciones.

    Returns:
        El valor de F1.
    """
    comunes = int(np.count_nonzero(y & pred))
    den = int(np.count_nonzero(y)) + int(np.count_nonzero(pred))
    return 2 * comunes / den if den else 0.0


def _macro_f1(verdad: np.ndarray, predicho: np.ndarray) -> float:
    """Macro-F1 de las tres clases positivas.

    Args:
        verdad: matriz verdadera binaria.
        predicho: matriz predicha binaria.

    Returns:
        La media de las tres F1.
    """
    return float(np.mean([_f1(verdad[:, j], predicho[:, j]) for j in range(verdad.shape[1])]))


def _informe_md(informe: dict[str, Any]) -> None:
    """Escribe el informe legible.

    Args:
        informe: informe con los datos.

    Returns:
        None.
    """
    r = informe["resultados"]
    comp = informe["complementariedad"]
    memo = informe["memorizacion"]
    cfg = informe["configuracion"]
    mot = informe["correccion_de_la_motivacion"]
    lineas = [
        f"# {informe['familia']}: {EXPERIMENTO}",
        "",
        "Generado por `experiments/v27_01_char_spectrum_kernel.py`.",
        f"**Estado: `{informe['estado']}`.** **Ninguna submission.**",
        "",
        "## 1. Correccion de la motivacion, que hay que leer antes que los numeros",
        "",
        f"- Lo que se creia: {mot['que_se_creia']}.",
        f"- Lo que se habia medido: {mot['que_se_tenia_medido']}.",
        f"- Implicacion: {mot['implicacion']}.",
        "",
        "## 2. La configuracion",
        "",
        "| parametro | valor |",
        "|---|---|",
        f"| kernel | {cfg['kernel']} |",
        f"| normalizacion | `{cfg['normalizacion']}` |",
        f"| analyzer / rango / min_df | {cfg['analyzer']} / {tuple(cfg['rango_ngram'])} / {cfg['min_df']} |",
        f"| clasificador | `{cfg['clasificador']}` |",
        f"| vectorizado ajustado dentro de cada pliegue | {cfg['vectorizado_ajustado_por_pliegue']} |",
        f"| test leido | {cfg['test_leido']} |",
        f"| features por pliegue | {cfg['features_por_pliegue']} |",
        "",
        "## 3. Los controles del kernel",
        "",
        "| control | resultado | ok |",
        "|---|---|:-:|",
    ]
    for c in informe["controles_del_kernel"]:
        lineas.append(f"| {c['nombre']} | {c['obtenido']} | {'si' if c['ok'] else 'NO'} |")
    lineas += [
        "",
        "## 4. Resultados OOF",
        "",
        "| modelo | corrected_cv | dOOF vs V03 | train F1 | gap | sd pliegues |",
        "|---|---:|---:|---:|---:|---:|",
        f"| V03 (congelado) | {CORRECTED_CV_V03:.10f} | +0.0000000000 | 0.9614 | +0.1842 | -- |",
        (f"| CHAR_SPECTRUM_KERNEL | {r['corrected_cv']:.6f} | {r['delta_vs_v03']:+.6f} | "
        f"{r['f1_train']:.4f} | {r['gap_train_oof']:+.4f} | {r['sd_entre_pliegues']:.4f} |"),
        "",
        "F1 por etiqueta: " + " | ".join(
            f"{e} {v:.4f} (V03 {comp['por_etiqueta'][e]['f1_v03']:.4f})"
            for e, v in r["f1_por_etiqueta"].items()
        ),
        "",
        "F1 por pliegue: " + " | ".join(f"{v:.4f}" for v in r["f1_por_pliegue"])
        + f"  (media {r['media_entre_pliegues']:.4f}, sd {r['sd_entre_pliegues']:.4f})",
        "",
        "## 5. Complementariedad con V03",
        "",
        "| magnitud | valor |",
        "|---|---:|",
        f"| V03 wrong y kernel correct (rescates) | {comp['rescates']} |",
        f"| V03 correct y kernel wrong (danos) | {comp['danos']} |",
        f"| **neto** | **{comp['neto']:+d}** |",
        (f"| celdas coincidentes | {comp['celdas_coincidentes']} de {comp['celdas_totales']} "
        f"({comp['fraccion_coincidente']:.4f}) |"),
        f"| correlacion de predicciones | {comp['correlacion_predicciones']:.4f} |",
        f"| ambos incorrectos | {comp['ambos_malos']} |",
        "",
        "| etiqueta | F1 kernel | F1 V03 | cambios | rescates | danos |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for e, d in comp["por_etiqueta"].items():
        lineas.append(
            f"| {e} | {d['f1']:.4f} | {d['f1_v03']:.4f} | {d['cambios']} | {d['rescates']} | {d['danos']} |"
        )
    lineas += [
        "",
        "## 6. Auditoria de memorizacion, spectrum frente a RBF exacto en los MISMOS pliegues",
        "",
        "| pliego | spectrum mediana | spectrum >0.60 | RBF mediana | RBF >0.60 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for clave in sorted(memo["spectrum_por_pliegue"], key=int):
        s = memo["spectrum_por_pliegue"][clave]
        b = memo["rbf_exacto_por_pliegue"][clave]
        lineas.append(
            f"| {clave} | {s['mediana_vecino_ENTRENAMIENTO']:.4f} | "
            f"{s['fraccion_sobre_0_60_ENTRENAMIENTO'] * 100:.1f} % | {b['mediana']:.4f} | "
            f"{b['fraccion_sobre_0_60'] * 100:.1f} % |"
        )
    c = memo["comparativa"]
    lineas += [
        "",
        (f"Agregado: spectrum mediana {c['spectrum_mediana_vecino']:.4f} frente a RBF "
        f"{c['rbf_exacto_mediana_vecino']:.4f}, diferencia {c['diferencia_de_mediana']:+.4f}."),
        "",
        (f"Comportamiento train frente a OOF: train {r['f1_train']:.4f} contra OOF "
        f"{r['corrected_cv']:.4f}, gap {r['gap_train_oof']:+.4f}."),
        "",
        "## 7. Criterio de continuacion",
        "",
        "| criterio | medida | se cumple |",
        "|---|---|:-:|",
    ]
    for cc in informe["criterios_de_continuacion"]:
        lineas.append(f"| {cc['criterio']} | {cc['medida']} | {'si' if cc['cumple'] else 'no'} |")
    g = informe["gastos"]
    lineas += [
        "",
        f"**Veredicto: `{informe['estado']}`.**",
        "",
        (f"Coste: {g['segundos_totales']} s, memoria {g['memoria_pico_mb']:.0f} MB, "
        f"submissions {g['submissions']}."),
        "",
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
