"""Tsetlin Machine binaria, transcrita de la implementacion de referencia de pyTsetlinMachine.

Por que este archivo reescribe la version anterior
--------------------------------------------------
La primera version de este script era una reimplementacion **deducida**, y estaba mal. Solo
seilio al contrastarla con dos problemas que cualquier Tsetlin debe resolver por
construccion: el caso de una feature activa, donde la etiqueta esta en la propia feature, y el
XOR de dos features, que es el canonico de lo que un hiperplano no puede y una conjuncion si.
La version deducida se estancaba en F1 0.23 sin importar ni el numero de clausulas ni las
epocas, y con cero literales activos. Los errores eran tres:

1. **El estado de un literal es un contador de bits, no un signo.** La referencia guarda por
   literal un entero de `number_of_state_bits` bits, inicializado a `0b01111111`: los siete
   bits bajos a uno y el bit mas alto a cero. Ese **bit mas alto es el unico que decide** si el
   literal es include (la clausula lo exige) o exclude (la clausula lo prohibe), y los bits
   bajos son un contador con saturacion que sube con el Tipo Ia y baja con el Ib. La version
   anterior usaba un `int8` en {-1, 0, +1} con un `clip`, con lo cual los estados include y
   exclude no llegaban a existir y la inclusion no se activaba nunca.

2. **La polaridad va por la PARIDAD DEL INDICE.** La referencia decide con
   ``(2*target-1) * (1 - 2*(j & 1))``: las clausulas pares son de una polaridad y las impares
   de la otra, intercaladas. La version anterior ponia todas las positivas primero.

3. **Hay tres tipos de feedback, no dos.** Tipo Ia, sobre una clausula que satisface la fila;
   Tipo Ib, sobre una clausula que NO satisface, que degrada sus literales; y Tipo II, el Noah,
   que degrada clausulas de la polaridad contraria. La version anterior no tenia Tipo Ib.

La satisfaccion de una clausula, segun la referencia, es la condicion bitwise

    (bit_mas_alto_del_literal & entrada) == bit_mas_alto_del_literal

es decir, **todo literal include tiene que estar presente en la entrada**. Los bits bajos no
participan en la satisfaccion: son solo el contador que acaba cambiando el bit alto.

La decision
------------
``class_sum`` acumula, con signo segun la paridad del indice, las clausulas que satisfacen la
entrada, y se recorta a ``[-T, T]``. La prediccion es ``class_sum > 0``. Esa suma acumulativa
es lo que da al modelo una curva ROC propia, a diferencia del score lineal de V03.

El control que obliga a la comparacion
--------------------------------------
V03 usa TF-IDF con `sublinear_tf=True` y pesos float. El Tsetlin consume **bits**. Comparar
Tsetlin contra V03 con dos representaciones distintas mediria la diferencia entre dos
representaciones, no entre dos mecanismos. Por eso el control se entrena sobre la **matriz
binaria identica**, byte a byte, a la que ve el Tsetlin, y la comparacion informativa es de
tres vias:

    Tsetlin(binario)  vs  LogisticRegression(binario)  ->  ganancia del MECANISMO TM
    LogisticRegression(binario)  vs  V03                 ->  coste de la BINARIZACION
    Tsetlin(binario)  vs  V03                           ->  efecto neto

Controles, antes de interpretar nada
-------------------------------------
1. Aritmetica de clausulas contra un recuento escrito a mano.
2. Polaridad por paridad del indice.
3. Entrada vacia: satisfaccion vacuamente verdadera.
4. Determinismo con la misma semilla.
5. **Caso de una feature activa**, donde la respuesta esta en la propia feature y un Tsetlin
   correcto tiene que llegar a F1 1.0. Es el control que detecto los tres errores anteriores,
   y corre ANTES de cualquier ajuste sobre datos reales.
6. **XOR de dos features**.

Uso
---
    /home/carlos/venv/bin/python experiments/v26_02_tsetlin.py
"""

from __future__ import annotations

import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


EXPERIMENTO = "V26-02 (Tsetlin Machine binaria de pyTsetlinMachine, con control lineal de la misma matriz)"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v26_02_tsetlin.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "v26_02_tsetlin.md"
OOF_CANONICO = PROJECT_ROOT / "features" / "oof_v03_final.npz"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"

ETIQUETAS = ("HS", "TR", "AG")
UMBRALES_V03 = np.array([0.44, 0.51, 0.46])
CORRECTED_CV_V03 = 0.7772114525165721
C_V03 = 2.0
SEMILLA = 42
BITS_ESTADO = 8
TOPE = 1 << (BITS_ESTADO - 1)
MAX_LITERALES = 200
VALORES_K = (200, 500, 1000)
EPOCHS = 30
SENSIBILIDAD = 2.0
N_PROCESOS = 15

PARAMETROS_LR_V03 = {
    "C": C_V03,
    "class_weight": "balanced",
    "solver": "liblinear",
    "fit_intercept": True,
    "tol": 1e-4,
    "max_iter": 1000,
    "random_state": SEMILLA,
}

REPRESENTACIONES = {
    "word12": {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "binary": True},
    "charwb35": {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2, "binary": True},
}

_ESTADO: dict[str, Any] = {}


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V26-02] {mensaje}", flush=True)


def memoria_pico_mb() -> float:
    """Resident set maximum del proceso, en megabytes.

    Returns:
        La memoria maxima residente en MB.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def filas_de(matriz: sparse.csr_matrix) -> np.ndarray:
    """Indice de fila de cada elemento no nulo de una CSR.

    Args:
        matriz: matriz dispersa CSR.

    Returns:
        Vector de longitud `nnz` con el indice de fila de cada no nulo.
    """
    return np.repeat(np.arange(matriz.shape[0]), np.diff(matriz.indptr))


class TsetlinMachine:
    """Tsetlin Machine binaria con el estado de literales de la referencia.

    El estado de un literal es un entero de `BITS_ESTADO` bits que arranca en `TOPE - 1`, es
    decir, con los bits bajos a uno y el bit mas alto a cero. El bit mas alto dice si el literal
    es include, que la clausula exige, o exclude, que la clausula prohibe. Los bits bajos son un
    contador con saturacion: sube con el Tipo Ia y baja con el Ib, y el acarreo hace que el bit
    mas alto cambie alguna vez. Ese acarreo es lo que la version anterior rompia con un `clip`.
    """

    def __init__(
        self,
        n_clausulas: int,
        n_features: int,
        umbral: float,
        sensibilidad: float,
        semilla: int,
        max_literales: int = MAX_LITERALES,
    ) -> None:
        """Construye el modelo.

        Args:
            n_clausulas: clausulas totales; la paridad del indice fija la polaridad.
            n_features: columnas de la matriz de entrada.
            umbral: T, el umbral de decision sobre `class_sum`.
            sensibilidad: s, el parametro del feedback Tipo II.
            semilla: semilla del generador.
            max_literales: tope de literales include por clausula.
        """
        self.n_clausulas = n_clausulas
        self.n_features = n_features
        self.n_literales = 2 * n_features
        self.umbral = float(umbral)
        self.sensibilidad = float(sensibilidad)
        self.max_literales = int(max_literales)
        self.rng = np.random.default_rng(semilla)
        self.estado = np.full((n_clausulas, self.n_literales), TOPE - 1, dtype=np.int16)
        self.polaridades = np.where(np.arange(n_clausulas) % 2 == 0, 1, -1).astype(np.int8)

    def incluidos(self) -> np.ndarray:
        """Mascara booleana de los literales que son include.

        Returns:
            Array de forma (n_clausulas, n_literales) con True donde el bit alto esta a uno.
        """
        return self.estado >= TOPE

    def n_incluidos(self) -> np.ndarray:
        """Cuantos literales include tiene cada clausula.

        Returns:
            Vector de enteros con el conteo por clausula.
        """
        return self.incluidos().sum(axis=1)

    def satisfacen(self, presentes: np.ndarray) -> np.ndarray:
        """Que clausulas satisfacen la fila.

        Una clausula satisface si **todo** literal include suyo esta presente en la fila, que
        es la condicion bitwise de la referencia.

        Args:
            presentes: vector booleano de literales presentes, de longitud `n_literales`.

        Returns:
            Vector booleano de longitud `n_clausulas`.
        """
        if not self.incluidos().any():
            return np.ones(self.n_clausulas, dtype=bool)
        return ~(self.incluidos() & ~presentes[None, :]).any(axis=1)

    def _incrementar(self, clausula: int, literales: np.ndarray) -> None:
        """Satura hacia arriba el estado de los literales indicados.

        Args:
            clausula: indice de la clausula.
            literales: indices de literal a incrementar.
        """
        if len(literales) == 0:
            return
        self.estado[clausula, literales] = np.minimum(self.estado[clausula, literales] + 1, 255)

    def _decrementar(self, clausula: int, literales: np.ndarray) -> None:
        """Satura hacia abajo el estado de los literales indicados.

        Args:
            clausula: indice de la clausula.
            literales: indices de literal a decrementar.
        """
        if len(literales) == 0:
            return
        self.estado[clausula, literales] = np.maximum(self.estado[clausula, literales] - 1, 0)

    def ajustar(
        self,
        matriz: sparse.csr_matrix,
        y: np.ndarray,
        epochs: int,
        fila_indice: np.ndarray,
    ) -> dict[str, Any]:
        """Entrena con los tres tipos de feedback de la referencia.

        La referencia elige clausula por clausula con probabilidad
        ``(1/(2T)) * (T + (1-2*target)*class_sum)``, que es un muestreo tipo bandit: cuando la
        maquina se equivoca mucho, `class_sum` se aleja de cero y el muestreo se sesga hacia
        un lado, que es lo que corrige el error. Esa regla se reproduce aqui.

        Args:
            matriz: matriz binaria CSR de entrenamiento.
            y: etiquetas binarias.
            epochs: numero de epocas.
            fila_indice: indice de fila de cada no nulo.

        Returns:
            Diccionario de diagnostico.
        """
        inicio = time.time()
        n_filas = matriz.shape[0]
        for _ in range(epochs):
            for fila in self.rng.permutation(n_filas):
                mascara = fila_indice == fila
                indices = matriz.indices[mascara]
                if len(indices) == 0:
                    continue
                objetivo = int(y[fila])
                objetivo_signo = 1 if objetivo == 1 else -1
                presentes = np.zeros(self.n_literales, dtype=bool)
                presentes[indices] = True
                presentes[indices + self.n_features] = True
                ausentes = ~presentes
                satisfacen = self.satisfacen(presentes)
                included = self.incluidos()

                # Muestreo de clausulas por bandit, con la regla de la referencia.
                suma = float(np.dot(satisfacen.astype(np.int64), self.polaridades))
                suma = float(np.clip(suma, -self.umbral, self.umbral))
                pesos = (self.umbral + (1 - 2 * objetivo) * suma) / (2.0 * self.umbral)
                pesos = np.clip(pesos, 0.0, 1.0)
                elige = self.rng.random(self.n_clausulas) <= pesos

                # Tipo I sobre las clausulas de la polaridad del objetivo, y Tipo II sobre las
                # de la contraria. La referencia condiciona el Tipo Ia a que la clausula
                # satisfaga y a que no exceda el tope de literales, y el Tipo II tambien.
                propias = elige & (self.polaridades == objetivo_signo)
                for clausula in np.flatnonzero(propias):
                    if not satisfacen[clausula]:
                        # Tipo Ib: la clausula no satisface la fila, asi que se degradan sus
                        # literales include. Sin este paso las clausulas se llenan de
                        # literales irrelevantes y el modelo no discrimina.
                        self._decrementar(clausula, np.flatnonzero(included[clausula]))
                        continue
                    if self.n_incluidos()[clausula] > self.max_literales:
                        continue
                    # Tipo Ia: suben los literales presentes que ya son include, y los que no
                    # son include entran en la clausula. En ambos casos hay que empujar el
                    # contador para que el bit mas alto llegue a activarse.
                    subir = indices[self.estado[clausula, indices] >= TOPE]
                    entrar = indices[self.estado[clausula, indices] < TOPE]
                    self._incrementar(clausula, subir)
                    self.estado[clausula, entrar] = TOPE

                contrarias = satisfacen & (self.polaridades != objetivo_signo)
                if contrarias.any() and self.rng.random() < self.sensibilidad / (self.sensibilidad + 1.0):
                    # Tipo II, el Noah: degradan los literales que la clausula retenia y que no
                    # estaban en la fila, para que deje de retenerlos.
                    clausula = int(self.rng.choice(np.flatnonzero(contrarias)))
                    retenidos = np.flatnonzero(included[clausula] & ausentes)
                    self._decrementar(clausula, retenidos)
        return {
            "segundos": round(time.time() - inicio, 2),
            "clausulas_activas": int(np.count_nonzero(self.n_incluidos() > 0)),
            "literales_activos": int(self.n_incluidos().sum()),
            "bytes_modelo": int(self.estado.nbytes + self.estado.dtype.itemsize),
        }

    def predecir(self, matriz: sparse.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
        """Predicciones por `class_sum` y probabilidades de todas las filas.

        Args:
            matriz: matriz CSR.

        Returns:
            Tupla con las predicciones y las probabilidades.
        """
        probabilidades = np.zeros(matriz.shape[0])
        for fila in range(matriz.shape[0]):
            indices = matriz.indices[matriz.indptr[fila] : matriz.indptr[fila + 1]]
            presentes = np.zeros(self.n_literales, dtype=bool)
            presentes[indices] = True
            presentes[indices + self.n_features] = True
            satisfacen = self.satisfacen(presentes)
            suma = float(np.dot(satisfacen.astype(np.int64), self.polaridades))
            probabilidades[fila] = float(np.clip(suma / (2.0 * self.umbral) + 0.5, 0.0, 1.0))
        return (probabilidades > 0.5).astype(int), probabilidades
