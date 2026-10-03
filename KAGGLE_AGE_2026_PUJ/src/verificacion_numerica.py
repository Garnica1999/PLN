"""Verificacion numerica del entorno antes de medir correlaciones o similaridades.

Motivo (D-040): en esta maquina, ``numpy 1.26.4`` sobre ``Python 3.14`` (combinacion no
soportada: esa version de numpy es de 2023 y Python 3.14 es de 2025) calcula **mal** el
producto matricial cuando ambos operandos son vistas de la **misma** matriz, es decir en el
patron ``X @ X.T``. El valor devuelto no es el producto: la diagonal del Gram sale ~0,44 en
lugar de 12,97, y con 5000x279 sale 298,6 en lugar de 4962,4. No es una carrera de hilos
(falla igual con ``OMP_NUM_THREADS=1``) ni un problema de orden de filas (copiar la matriz a
C-contiguo o a F-contiguo no lo arregla).

Lo que **si** funciona y por tanto puede usarse con confianza:

* ``A @ B.T`` con matrices **distintas** (verificado en cuatro tamanos). Los cinco sitios del
  proyecto que usan ese patron (centroides, vistas, verificacion de leakage, correccion
  probabilistica) son correctos.
* ``A.T @ A`` y ``Z.T @ z`` (matriz-vector con el primer operando transpuesto), que es
  exactamente el patron que uso la fase de lexicones para medir redundancia.
* ``np.einsum`` y las sumas explicitas, que son exactas por construccion.

Por eso :func:`autotest_numerico` se llama al inicio de cada experimento de V13: si el
entorno empeora, el experimento se detiene en el primer segundo en lugar de producir
correlaciones falsas. Y :func:`grama_exacta` ofrece la via fiable para calcular una matriz de
correlacion sin depender de BLAS.
"""

from __future__ import annotations

import numpy as np


class ErrorNumerico(RuntimeError):
    """Se lanza cuando el entorno numerico no supera la autocomprobacion."""


def grama_exacta(matriz: np.ndarray) -> np.ndarray:
    """Matriz de productos cruzados sin BLAS: ``M.T @ M`` calculado columna a columna.

    Es la via fiable en este entorno. Cada elemento es una suma explicita de productos, que
    NumPy hace con su propio bucle y no con la biblioteca BLAS.

    Parametros
    ----------
    matriz:
        Matriz de forma ``(filas, columnas)``.

    Devuelve
    -------
    numpy.ndarray
        Matriz de forma ``(columnas, columnas)`` con ``M.T @ M``.
    """
    columnas = matriz.shape[1]
    grama = np.empty((columnas, columnas), dtype=np.float64)
    for indice in range(columnas):
        grama[indice] = np.einsum("ij,i->j", matriz, matriz[:, indice])
    return grama


def autotest_numerico(verbose: bool = False) -> dict:
    """Comprueba el comportamiento del producto matricial del entorno.

    Devuelve un diccionario con el resultado de cada comprobacion. Si alguna falla se lanza
    :class:`ErrorNumerico` con el detalle, porque un fallo de este tipo invalida cualquier
    correlacion calculada con BLAS.
    """
    generador = np.random.default_rng(0)
    matriz = generador.normal(size=(500, 6))
    otra = generador.normal(size=(500, 6))
    vector = generador.normal(size=500)

    diagonal_real = float((matriz[:, 0] ** 2).sum())
    diagonal_blas = float((matriz @ matriz.T)[0, 0])
    cruz_real = float((matriz[:, 0] * matriz[:, 1]).sum())
    cruz_blas = float((matriz @ matriz.T)[0, 1])
    distintas_real = float(np.einsum("ij,ij->i", matriz[0:1], otra[0:1])[0])
    distintas_blas = float((matriz @ otra.T)[0, 0])
    vectorial_real = float((matriz[:, 0] * vector).sum())
    vectorial_blas = float((matriz.T @ vector)[0])
    einsum_grama = float(grama_exacta(matriz)[0, 0])
    einsum_real = float(np.einsum("ij,i->j", matriz, matriz[:, 0]).sum())

    comprobar = {
        "matrices_distintas": abs(distintas_blas - distintas_real),
        "matriz_vector": abs(vectorial_blas - vectorial_real),
        "einsum_suma": abs(einsum_grama - diagonal_real),
        "grama_exacta_diagonal": abs(einsum_grama - diagonal_real),
        "corrcoef_1d": abs(float(np.corrcoef(matriz[:, 0], matriz[:, 1])[0, 1])
                           - float(np.einsum("ij,ij->i", matriz[:, 0:1] - matriz[:, 0].mean(),
                                             matriz[:, 1:2] - matriz[:, 1].mean()).sum())
                           / float(np.sqrt(np.einsum("ij,ij->i", matriz[:, 0:1] - matriz[:, 0].mean(),
                                                    matriz[:, 0:1] - matriz[:, 0].mean()).sum()
                                           * np.einsum("ij,ij->i", matriz[:, 1:2] - matriz[:, 1].mean(),
                                                       matriz[:, 1:2] - matriz[:, 1].mean()).sum()))),
        "norma_vectorial": abs(float(np.linalg.norm(matriz[0] - matriz[1]))
                               - float(np.sqrt(((matriz[0] - matriz[1]) ** 2).sum()))),
    }
    conocidas_rotas = {
        "autotranspuesta_diagonal": abs(diagonal_blas - diagonal_real),
        "autotranspuesta_cruzado": abs(cruz_blas - cruz_real),
    }
    # ``np.corrcoef(matriz_2d)`` y ``np.cov(matriz_2d)`` con ``rowvar=True`` hacen internamente la
    # auto-transpuesta y devuelven una matriz (filas, filas) en vez de (variables, variables):
    # con 30 variables y 5000 filas devolvio 5000x5000. Con ``rowvar=False`` funcionan bien, y la
    # forma de 1D (la que usan phase5, phase13, phase16 y phase24 del proyecto) tambien.
    forma_corrcoef = np.corrcoef(matriz).shape
    if forma_corrcoef != (matriz.shape[1], matriz.shape[1]):
        conocidas_rotas["corrcoef_2d_rowvar_true"] = float(forma_corrcoef[0])
    fallos = {nombre: valor for nombre, valor in comprobar.items() if valor > 1e-8}
    resultado = {
        "numpy": np.__version__,
        "rutas_verificadas": {nombre: round(valor, 12) for nombre, valor in comprobar.items()},
        "rutas_prohibidas": {nombre: round(valor, 12)
                             for nombre, valor in conocidas_rotas.items() if valor > 1e-8},
    }
    if verbose:
        for nombre, valor in comprobar.items():
            print(f"  verificada  {nombre:24s} desviacion {valor:.3e} "
                  f"{'OK' if valor <= 1e-8 else 'FALLO'}")
        for nombre, valor in resultado["rutas_prohibidas"].items():
            print(f"  PROHIBIDA   {nombre:24s} desviacion {valor:.3e}  (D-040: no usar)")
    if fallos:
        raise ErrorNumerico(
            "una ruta numerica en la que se apoya el proyecto no es fiable: "
            + ", ".join(f"{nombre} ({valor:.2e})" for nombre, valor in sorted(fallos.items()))
            + ". Se detiene el experimento antes de producir metricas falsas."
        )
    return resultado
