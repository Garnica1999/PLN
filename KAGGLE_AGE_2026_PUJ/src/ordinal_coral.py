"""Cumulative-link ordinal (CORAL) con pesos compartidos y umbrales ordenados.

Este modulo implementa la formulacion ordinal acumulada **compartida** sobre un unico vector
de pesos, que es lo que distingue a CORAL de las dos formulaciones ya medidas en AGE:

- **V20 (Caso A, D-057)**: cinco discriminadores binarios independientes, cada uno con su
  propio vector de pesos `w_k`. No hay restriccion de orden entre ellos y la monotonia de las
  cinco CDF **no** esta garantizada: el 11,18 % de las filas la viola.
- **Modelo compartido de D-057**: un unico LightGBM multiclase del que las cinco CDF salen
  **por suma**. Comparte la representacion, pero no es un cumulative-link: no hay umbrales
  ordinales sino una salida multiclase descompuesta.
- **CORAL (este modulo)**: un unico vector de pesos `w` y cinco umbrales ordenados
  `theta_0 <= theta_1 <= ... <= theta_4`, con

      P(Y <= k | x) = sigmoid(theta_k - w' x)

  Los umbrales se parametrizan como `theta_k = gamma + suma_{j<k} softplus(delta_j)`, luego el
  orden es una **garantia de construccion** y no una restriccion que haya que imponer despues.

La verosimilitud de la clase observada es la **difercia** de dos CDF,

      P(Y = y) = sigmoid(theta_y - eta) - sigmoid(theta_{y-1} - eta)

que **no** se factoriza en producto de sigmoidos, y por eso se evalua con `logaddexp` y no con
`log(sigmoid(z_a)) + log(1 - sigmoid(z_b))`. El gradiente es analitico y esta verificado contra
diferencias finitas en `scripts/v23_00_tests_coral.py` (TEST D).

**Por que se penalizan tambien los separadores.** Sin penalizacion sobre `theta` el maximo
verosimilitud de un modelo ordinal con una direccion `w` rica sufre separacion: los umbrales
se van fuera del rango de los datos, `eta` satura y las cinco CDF se vuelven escalones. Medido
en el sintetico de TEST B, con penalty solo sobre `w` los umbrales llegaban a 12.07 con features
estandarizadas en [-3, 3] y el macro-F1 de entrenamiento caia a 0.213. Penalizar los **gaps**
entre umbrales acota la separacion **sin sesgar `gamma`**, que si debe ser grande y negativo
cuando la clase baja es rara, que es el caso de `13-17` en AGE.

**Alcance numerico**: el predictor es lineal, luego la escala de las columnas importa. Las 279
features de AGE incluyen agregados label-informed con magnitud de hasta 59.439 junto a
probabilidades en [0, 1]. La estandarizacion se aplica **dentro de cada fold de
entrenamiento** y es **afin y sin perdida**: no reduce dimensionalidad, no proyecta y no
cambia la clase de hipotesis de un modelo lineal, solo rescala cada eje.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

__all__ = ["CoralOrdinal", "logit_seguro", "softplus", "inversa_softplus", "cdf_a_clases"]


def softplus(x: np.ndarray | float) -> np.ndarray:
    """`log(1 + exp(x))` calculado de forma estable en los dos extremos."""
    return np.logaddexp(0.0, np.asarray(x, dtype=float))


def inversa_softplus(y: float) -> float:
    """Inversa de `softplus`, acotada para que un separador inicial muy pequeno no diverja.

    `softplus` es estrictamente creciente, luego la inversa existe para todo real. Se acota por
    abajo para que un gap pequeno no produzca un `delta_tilde` negativo, y por arriba para que
    un gap grande no desborde.
    """
    acotado = float(np.clip(y, 1e-6, 30.0))
    return float(np.log(np.expm1(acotado)))


def logit_seguro(p: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """`log(p / (1 - p))` sobre probabilidades recortadas."""
    recortado = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(recortado) - np.log1p(-recortado)


def cdf_a_clases(cdf: np.ndarray) -> np.ndarray:
    """Convierte `n` probabilidades acumuladas en `n + 1` probabilidades de clase.

    `P(Y = 0) = CDF_0`, `P(Y = k) = CDF_k - CDF_{k-1}` y la ultima clase se lleva el resto hasta
    1. Se recorta a no negativo y se renormaliza, que es exactamente la reconstruccion que
    aplica V20 a sus cinco umbrales.
    """
    cdf = np.asarray(cdf, dtype=float)
    n_clases = cdf.shape[1] + 1
    seis = np.zeros((len(cdf), n_clases))
    seis[:, 0] = cdf[:, 0]
    seis[:, 1:-1] = np.diff(cdf, axis=1)
    seis[:, -1] = 1.0 - cdf[:, -1]
    seis = np.clip(seis, 0.0, None)
    return seis / np.clip(seis.sum(axis=1, keepdims=True), 1e-9, None)


class CoralOrdinal:
    """Cumulative-link ordinal con pesos compartidos y `n_clases - 1` umbrales ordenados.

    Parameters
    ----------
    lambda_w : float
        Coeficiente L2 sobre `w`, normalizado por el numero de columnas.
    lambda_umbral : float
        Coeficiente L2 sobre los **gaps** entre umbrales, normalizado por su numero. Es lo que
        evita la separacion descrita en la documentacion del modulo; ver TEST B.
    max_iter : int
        Tope de iteraciones de L-BFGS-B.
    tol : float
        Tolerancia de gradiente de L-BFGS-B.
    estandarizar : bool
        Si `True` (por defecto) ajusta media y desviacion **con los datos de entrenamiento** y
        las reutiliza al predecir. Nunca ve la validacion.
    """

    def __init__(self, lambda_w: float = 1.0, lambda_umbral: float = 1.0, max_iter: int = 500,
                 tol: float = 1e-9, estandarizar: bool = True) -> None:
        self.lambda_w = float(lambda_w)
        self.lambda_umbral = float(lambda_umbral)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.estandarizar = bool(estandarizar)
        self.n_clases_: int = 0
        self.n_umbral_: int = 0
        self.media_: np.ndarray | None = None
        self.desviacion_: np.ndarray | None = None
        self.pesos_: np.ndarray | None = None
        self.umbrales_: np.ndarray | None = None
        self.n_iters_: int = 0
        self.gradiente_maximo_: float = float("nan")
        self.exito_: bool = False

    # ------------------------------------------------------------------ transformacion
    def _aplicar_escala(self, x: np.ndarray) -> np.ndarray:
        """Lleva las columnas a media cero y desviacion uno con los parametros del fold."""
        if not self.estandarizar or self.media_ is None:
            return x
        return (x - self.media_) / self.desviacion_

    # ------------------------------------------------------------------ perdida y gradiente
    def _objetivo(self, params: np.ndarray, x: np.ndarray, clase: np.ndarray) -> tuple[float, np.ndarray]:
        """Perdida media y su gradiente analitico respecto de `params`.

        `params` es `[w (d), gamma (1), delta_tilde (n_umbral - 1)]`. `n_umbral` se deduce de
        `clase` para que el objetivo no dependa de ningun argumento externo que pueda
        desincronizarse de la matriz.
        """
        d = x.shape[1]
        n_filas = len(clase)
        n_umbral = int(clase.max()) + 1 - 1
        w = params[:d]
        gamma = params[d]
        delta = softplus(params[d + 1:])
        # `theta_k = gamma + suma_{j<k} delta_j` con `k = 0..n_umbral-1`, luego hacen falta los
        # `n_umbral - 1` separadores **completos** en el cumsum, no recortados.
        theta = gamma + np.concatenate(([0.0], np.cumsum(delta)))
        z = theta[None, :] - (x @ w)[:, None]        # (n_filas, n_umbral)
        s = expit(z)                                  # s[:, k] = P(Y <= k)

        # `s_ext[:, k] = P(Y <= k - 1)`, luego `s_ext[:, k + 1] = P(Y <= k)`. Los dos extremos
        # son `P(Y <= -1) = 0` y `P(Y <= n_umbral) = 1`.
        s_ext = np.empty((n_filas, n_umbral + 2))
        s_ext[:, 0] = 0.0
        s_ext[:, 1:-1] = s
        s_ext[:, -1] = 1.0

        fila = np.arange(n_filas)
        s_alto = s_ext[fila, clase + 1]              # P(Y <= y); vale 1.0 en la clase extrema alta
        s_bajo = s_ext[fila, clase]                  # P(Y <= y - 1); vale 0.0 en la clase 0
        es_bajo = clase == 0
        es_alto = clase == n_umbral
        es_interior = ~(es_bajo | es_alto)

        log_prob = np.empty(n_filas)
        # Extremo bajo: P(Y = 0) = sigmoid(theta_0 - eta).
        log_prob[es_bajo] = -np.logaddexp(0.0, -z[es_bajo, 0])
        # Extremo alto: P(Y = n_umbral) = 1 - sigmoid(theta_{n_umbral-1} - eta).
        log_prob[es_alto] = -np.logaddexp(0.0, z[es_alto, n_umbral - 1])
        # Interior: diferencia de dos sigmoidos en forma estable.
        if es_interior.any():
            a = z[es_interior, clase[es_interior]]
            b = z[es_interior, clase[es_interior] - 1]
            # `s_b / s_a = (1 + e^{-a}) / (1 + e^{-b})`; como `a > b` la razon es menor que uno,
            # luego `log(1 - razon)` es un `log1p` de argumento negativo.
            razon = (1.0 + np.exp(-a)) / (1.0 + np.exp(-b))
            log_prob[es_interior] = (-np.logaddexp(0.0, -a)
                                     + np.log1p(-np.clip(razon, 0.0, 1.0 - 1e-15)))

        n_umbral_efectivo = max(n_umbral - 1, 1)
        n_d = max(d, 1)
        penal_w = 0.5 * self.lambda_w * float(w @ w) / n_d
        penal_gap = 0.5 * self.lambda_umbral * float(delta @ delta) / n_umbral_efectivo
        perdida = -float(log_prob.sum()) / n_filas + penal_w + penal_gap

        # La masa de la clase observada es la que se maximiza y hace falta para el gradiente. El
        # suelo de 1e-12 evita dividir por cero y es muy inferior a cualquier masa real.
        masa = np.clip(s_alto - s_bajo, 1e-12, None)

        # d(perdida)/d(eta) = 1 - s_y - s_{y-1}; la misma formula cubre los dos extremos porque
        # `s_alto` vale 1 en la clase alta y `s_bajo` vale 0 en la clase baja.
        d_eta = 1.0 - s_alto - s_bajo
        # d(perdida)/d(theta_k) = -s_k (1 - s_k) / P(Y = y) * 1[y = k]
        #                          + s_k (1 - s_k) / P(Y = y) * 1[y = k + 1]
        cociente = s * (1.0 - s) / masa[:, None]
        mascara = np.arange(n_umbral)[None, :]
        d_theta = -(cociente * (clase[:, None] == mascara)).sum(axis=0)
        d_theta += (cociente * (clase[:, None] == mascara + 1)).sum(axis=0)

        grad = np.empty_like(params)
        grad[:d] = x.T @ d_eta / n_filas + self.lambda_w * w / n_d
        # `gamma` entra en los cinco umbrales, luego recibe la suma completa.
        acum = np.cumsum(d_theta) / n_filas
        grad[d] = acum[-1]
        # El separador `j` entra solo en los umbrales `k > j`, luego recibe la cola de la suma.
        sigmoide = expit(params[d + 1:])
        grad[d + 1:] = ((acum[-1] - acum[:-1]) * sigmoide
                        + self.lambda_umbral * delta * sigmoide / n_umbral_efectivo)
        return perdida, grad

    # ------------------------------------------------------------------ ajuste
    def fit(self, x: np.ndarray, clase: np.ndarray) -> "CoralOrdinal":
        """Ajusta `w` y los umbrales ordenados con L-BFGS-B.

        La inicializacion pone los umbrales en los cuantiles marginales de la clase: con
        `w = 0` las cinco CDF inicializadas coinciden con la frecuencia observada de cada
        acumulado, que es el punto de partida natural.
        """
        x = np.asarray(x, dtype=float)
        clase = np.asarray(clase).astype(int)
        d = x.shape[1]
        self.n_clases_ = int(clase.max()) + 1
        self.n_umbral_ = self.n_clases_ - 1
        n_umbral = self.n_umbral_
        if self.estandarizar:
            self.media_ = x.mean(axis=0)
            desviacion = x.std(axis=0)
            self.desviacion_ = np.where(desviacion < 1e-8, 1.0, desviacion)
            x = self._aplicar_escala(x)

        marginal = np.array([(clase <= k).mean() for k in range(n_umbral)])
        theta0 = np.maximum.accumulate(logit_seguro(marginal)) + 1e-3 * np.arange(n_umbral)
        params = np.concatenate((np.zeros(d), [theta0[0]],
                                [inversa_softplus(g) for g in np.diff(theta0)]))

        resultado = minimize(self._objetivo, params, args=(x, clase), jac=True, method="L-BFGS-B",
                             options={"maxiter": self.max_iter, "ftol": 1e-15, "gtol": self.tol})
        self.pesos_ = resultado.x[:d]
        gaps = softplus(resultado.x[d + 1:])
        self.umbrales_ = resultado.x[d] + np.concatenate(([0.0], np.cumsum(gaps)))
        self.n_iters_ = int(resultado.nit)
        self.gradiente_maximo_ = (float(np.max(np.abs(resultado.jac)))
                                  if resultado.jac is not None else float("nan"))
        self.exito_ = bool(resultado.success)
        return self

    # ------------------------------------------------------------------ prediccion
    def predecir_cdf(self, x: np.ndarray) -> np.ndarray:
        """Devuelve las `n_clases - 1` CDF, es decir `P(Y <= k)` para `k = 0..n_clases-2`."""
        if self.pesos_ is None or self.umbrales_ is None:
            raise RuntimeError("el modelo no esta ajustado")
        x = self._aplicar_escala(np.asarray(x, dtype=float))
        return expit(self.umbrales_[None, :] - (x @ self.pesos_)[:, None])

    def predecir_clase(self, x: np.ndarray) -> np.ndarray:
        """Clase argmax sobre las seis reconstruidas a partir de las cinco CDF."""
        return np.argmax(cdf_a_clases(self.predecir_cdf(x)), axis=1)

    # ------------------------------------------------------------------ diagnosticos
    def norma_pesos(self) -> float:
        """Norma euclidea de `w`, como magnitud de la direccion ajustada."""
        if self.pesos_ is None:
            raise RuntimeError("el modelo no esta ajustado")
        return float(np.linalg.norm(self.pesos_))

    def violaciones_monotonia(self, x: np.ndarray) -> int:
        """Filas cuya serie de CDF decrece. Con umbrales ordenados debe ser **siempre** cero."""
        cdf = self.predecir_cdf(x)
        if cdf.shape[1] < 2:
            return 0
        return int((np.diff(cdf, axis=1) < 0).any(axis=1).sum())
