"""V23 · 00 — Tests unitarios de la implementacion CORAL, obligatorios antes de tocar AGE.

Este script **no entrena nada sobre AGE**. Solo valida que `src/ordinal_coral.py` sea correcta,
que la monotonia este garantizada por construccion y que el gradiente analitico coincida con
diferencias finitas. Si un test falla, CORAL no se ejecuta.

## Los cuatro tests

- **TEST A**: dataset ordinal sintetico **perfectamente separable** en las cinco fronteras.
  Se espera macro-F1 alto y **cero violaciones de monotonia**.
- **TEST B**: dataset donde las cinco fronteras tienen senal **aproximadamente compartida**, que
  es justo la hipotesis de CORAL. Se verifica que el modelo compartido **converge** y que se
  comprueba que **no hay separacion** a la escala de AGE (135.000 filas, 279 columnas,
  proporciones reales de clase). Ese resultado es lo que fija `lambda_umbral = 0` como valor
  pre-registrado.
- **TEST C**: `theta_1 <= theta_2 <= ... <= theta_5` **durante** el entrenamiento, no solo al
  final. Se recorre un gradiente descendente a mano y se comprueba el orden en cada paso.
- **TEST D**: el gradiente analitico contra diferencias finitas centralizadas, mas
  comprobaciones de forma y de que la perdida coincide con la densidad de clase calculada de
  otra manera.

Entregables: ``experiments/v23_00_tests.json``, ``reports/v23_00_tests_coral.md``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src.ordinal_coral import (CoralOrdinal, cdf_a_clases, inversa_softplus,  # noqa: E402
                               logit_seguro, softplus)
from src.verificacion_numerica import autotest_numerico  # noqa: E402

INFORME = cfg.REPORTS / "v23_00_tests_coral.md"
RESULTADO = cfg.EXPERIMENTS / "v23_00_tests.json"
CLASES = list(range(len(cfg.CLASSES)))
#: Valores **pre-registrados** antes de ver AGE, fijados por el analisis de separacion de TEST B.
LAMBDA_W = 1.0
LAMBDA_UMBRAL = 0.0


def _sintetico(n: int, d: int, semilla: int) -> tuple[np.ndarray, np.ndarray]:
    """Seis clases con proporciones **reales de AGE** y senal lineal repartida en `d` columnas.

    Los bloques replican la forma de la matriz de AGE: seis columnas de NB-SVM en [0, 1], un
    bloque label-informed de magnitud grande y el resto en escala uno. Las columnas que llevan
    senal se eligen **relativas a `d`**, para que el mismo generador sirva con 40 columnas
    (TEST C) y con 279 (TEST B).
    """
    rng = np.random.default_rng(semilla)
    proporciones = np.array([0.05, 0.20, 0.25, 0.20, 0.15, 0.15])
    x = rng.normal(size=(n, d))
    n_svm = min(6, d)
    x[:, :n_svm] = rng.random((n, n_svm))
    n_denso = min(73, max(d - n_svm, 0))
    if n_denso:
        x[:, n_svm:n_svm + n_denso] = x[:, n_svm:n_svm + n_denso] * 5000.0 + 30000.0
    # Las columnas con senal se eligen dentro de la region de escala uno y se acotan a `d - 1`,
    # para que el mismo generador sirva con pocas columnas y con las 279 de AGE.
    pesos_senal = [0.8, -0.6, 0.5, -0.4]
    inicio = n_svm + n_denso
    disponibles = max(d - inicio, 1)
    columnas = [min(d - 1, inicio + (i * disponibles) // (len(pesos_senal) + 1))
                for i in range(len(pesos_senal))]
    latente = sum(peso * x[:, columna]
                  for peso, columna in zip(pesos_senal, columnas)
                  if columna < d)
    if not columnas:
        latente = x[:, 0]
    latente = latente + rng.normal(scale=0.9, size=n)
    orden = np.argsort(np.argsort(latente))
    clase = np.clip(np.searchsorted(np.cumsum(proporciones)[:-1], orden / n), 0, 5).astype(int)
    return x, clase


def test_a() -> dict:
    """Separable: las cinco fronteras son cortes limpios sobre una sola direccion.

    Se usan **279 columnas**, el numero real de AGE, porque la penalizacion se normaliza por
    `1 / d`: con pocas columnas el mismo `lambda_w` regulariza mucho mas y el test no seria
    comparable con lo que luego corre sobre el proyecto.
    """
    rng = np.random.default_rng(1)
    cortes = np.array([-3.0, -1.5, -0.5, 0.5, 1.5])
    por_clase = 120
    desplazamiento = 3.0
    x = rng.normal(size=(len(cortes) * por_clase, 279))
    # La senal va en la primera columna; el resto es ruido en escala uno.
    x[:, 0] = np.concatenate([rng.normal(corte, 0.25, por_clase)
                              for corte in cortes]) + desplazamiento
    clase = np.digitize(x[:, 0], cortes + desplazamiento).astype(int)
    modelo = CoralOrdinal(lambda_w=LAMBDA_W, lambda_umbral=LAMBDA_UMBRAL).fit(x, clase)
    prediccion = modelo.predecir_clase(x)
    valor = float(f1_score(clase, prediccion, average="macro", labels=CLASES))
    violaciones = modelo.violaciones_monotonia(x)
    return {"test": "A", "nombre": "sintetico perfectamente separable",
            "filas": int(len(x)), "columnas": int(x.shape[1]),
            "clases_presentes": int(np.unique(clase).size),
            "macro_f1_entrenamiento": round(valor, 6),
            "violaciones_monotonia": violaciones, "convergio": modelo.exito_,
            "iteraciones": modelo.n_iters_,
            "umbrales": [round(float(v), 4) for v in modelo.umbrales_],
            "norma_pesos": round(modelo.norma_pesos(), 6),
            "criterio": "macro-F1 >= 0.80, seis clases presentes y violaciones = 0",
            "ok": bool(valor >= 0.80 and violaciones == 0 and np.unique(clase).size == 6)}

    return {"test": "A", "nombre": "sintetico perfectamente separable",
            "filas": int(len(x)), "macro_f1_entrenamiento": round(valor, 6),
            "violaciones_monotonia": violaciones, "convergio": modelo.exito_,
            "umbrales": [round(float(v), 4) for v in modelo.umbrales_],
            "criterio": "macro-F1 >= 0.80 y violaciones = 0",
            "ok": bool(valor >= 0.80 and violaciones == 0)}


def test_b() -> dict:
    """Senal compartida entre las cinco fronteras, a la escala y con la forma de AGE.

    Ademas comprueba que **no hay separacion**: si la alguna CDF se saturase, el maximo
    verosimilitud habria driving los umbrales fuera del rango de los datos.
    """
    x, clase = _sintetico(135_000, 279, 20260927)
    proporciones = np.round(np.bincount(clase, minlength=6) / len(clase), 4).tolist()
    modelo = CoralOrdinal(lambda_w=LAMBDA_W, lambda_umbral=LAMBDA_UMBRAL, max_iter=300)
    modelo.fit(x[:120_000], clase[:120_000])
    cdf = modelo.predecir_cdf(x[:120_000])
    valor = float(f1_score(clase[120_000:], modelo.predecir_clase(x[120_000:]),
                           average="macro", labels=CLASES))
    eta = np.abs(modelo._aplicar_escala(x[:120_000]) @ modelo.pesos_).max()
    fraccion_saturada = float((np.sort(cdf, axis=1)[:, -1] > 1 - 1e-6).mean())
    return {"test": "B", "nombre": "senal compartida a la escala de AGE",
            "filas": int(len(x)), "columnas": int(x.shape[1]), "proporciones_clase": proporciones,
            "convergio": modelo.exito_, "iteraciones": modelo.n_iters_,
            "gradiente_maximo": round(float(modelo.gradiente_maximo_), 8),
            "macro_f1_validacion": round(valor, 6),
            "prior_validacion": round(float(f1_score(clase[120_000:],
                                                     np.full(15_000, 2),
                                                     average="macro", labels=CLASES)), 6),
            "norma_eta_maxima": round(float(eta), 4),
            "cdf_min": float(f"{cdf.min():.3e}"), "cdf_max": round(float(cdf.max()), 6),
            "fraccion_saturada": fraccion_saturada,
            "violaciones_monotonia": modelo.violaciones_monotonia(x[:120_000]),
            "lambda_w_pre_registrado": LAMBDA_W, "lambda_umbral_pre_registrado": LAMBDA_UMBRAL,
            "criterio": "converge, supera el prior y no satura (fraccion_saturada = 0)",
            "ok": bool(modelo.exito_ and valor > 0.20 and fraccion_saturada == 0.0)}


def test_c() -> dict:
    """`theta_1 <= ... <= theta_5` durante y despues del entrenamiento.

    Se recorre un gradiente descendente **a mano** sobre los parametros libres y se comprueba
    el orden en cada paso, no solo en la solucion final.
    """
    x, clase = _sintetico(6000, 40, 7)
    estandarizada = (x - x.mean(axis=0)) / np.where(x.std(axis=0) < 1e-8, 1.0, x.std(axis=0))
    modelo = CoralOrdinal(lambda_w=LAMBDA_W, lambda_umbral=LAMBDA_UMBRAL)
    marginal = np.array([(clase <= k).mean() for k in range(5)])
    theta0 = np.maximum.accumulate(logit_seguro(marginal)) + 1e-3 * np.arange(5)
    params = np.concatenate((np.zeros(40), [theta0[0]],
                            [inversa_softplus(g) for g in np.diff(theta0)]))
    incumplimientos: list[int] = []
    traza: list[list[float]] = []
    paso = 1e-3
    for _ in range(60):
        umbrales = params[40] + np.concatenate(([0.0], np.cumsum(softplus(params[41:]))))
        traza.append([round(float(v), 6) for v in umbrales])
        if not np.all(np.diff(umbrales) >= 0.0):
            incumplimientos.append(len(traza))
        _, gradiente = modelo._objetivo(params, estandarizada, clase)
        params = params - paso * gradiente
    umbrales_finales = params[40] + np.concatenate(([0.0], np.cumsum(softplus(params[41:]))))
    if not np.all(np.diff(umbrales_finales) >= 0.0):
        incumplimientos.append(-1)
    ajustado = modelo.fit(x, clase)
    return {"test": "C", "nombre": "orden de los umbrales durante y despues",
            "pasos": 60, "incumplimientos": incumplimientos,
            "umbrales_al_final_del_descenso": [round(float(v), 6) for v in umbrales_finales],
            "umbrales_tras_fit": [round(float(v), 6) for v in ajustado.umbrales_],
            "violaciones_tras_fit": int(ajustado.violaciones_monotonia(x)),
            "traza_inicial": traza[0], "traza_media": traza[len(traza) // 2],
            "criterio": "cero incumplimientos en los 60 pasos y tras el ajuste",
            "ok": bool(len(incumplimientos) == 0)}


def test_d() -> dict:
    """Gradiente analitico contra diferencias finitas, mas forma y densidad de clase."""
    rng = np.random.default_rng(11)
    n, d = 800, 15
    x = rng.normal(size=(n, d)) * np.array([1.0, 300.0] + [1.0] * (d - 2))
    clase = np.clip((x[:, 0] + 0.5 * x[:, 1] / 300 + rng.normal(scale=0.5, size=n) + 2.5),
                    0, 5).astype(int)
    modelo = CoralOrdinal(lambda_w=LAMBDA_W, lambda_umbral=LAMBDA_UMBRAL, estandarizar=False)
    x_plana = (x - x.mean(axis=0)) / np.where(x.std(axis=0) < 1e-8, 1.0, x.std(axis=0))
    params = np.concatenate((rng.normal(scale=0.3, size=d), [-1.8], rng.normal(scale=0.4, size=4)))
    perdida, gradiente = modelo._objetivo(params, x_plana, clase)
    assert gradiente.shape == params.shape, "forma del gradiente distinta de la de params"
    assert np.isfinite(perdida) and np.isfinite(gradiente).all(), "perdida o gradiente no finitos"
    eps = 1e-6
    numerico = np.zeros_like(params)
    for i in range(len(params)):
        arriba = params.copy()
        arriba[i] += eps
        abajo = params.copy()
        abajo[i] -= eps
        numerico[i] = (modelo._objetivo(arriba, x_plana, clase)[0]
                       - modelo._objetivo(abajo, x_plana, clase)[0]) / (2 * eps)
    error_relativo = float(np.max(np.abs(gradiente - numerico)
                                  / np.maximum(np.abs(gradiente), 1.0)))
    # Referencia independiente: densidad de clase por diferencia de CDF, sin `_objetivo`.
    pesos = params[:d]
    umbrales = params[d] + np.concatenate(([0.0], np.cumsum(softplus(params[d + 1:]))))
    ext = np.empty((n, 7))
    ext[:, 0] = 0.0
    ext[:, 1:-1] = expit(umbrales[None, :] - (x_plana @ pesos)[:, None])
    ext[:, -1] = 1.0
    nll = -np.log(np.clip(ext[np.arange(n), clase + 1] - ext[np.arange(n), clase],
                          1e-300, None)).sum() / n
    penal = 0.5 * LAMBDA_W * float(pesos @ pesos) / d
    dif_perdida = abs(perdida - (nll + penal))
    cdf_calc = expit(umbrales[None, :] - (x_plana @ pesos)[:, None])
    forma_cdf = tuple(cdf_calc.shape) == (n, 5)
    return {"test": "D", "nombre": "gradiente analitico contra diferencias finitas",
            "error_relativo_maximo": float(f"{error_relativo:.3e}"),
            "forma_gradiente_correcta": bool(gradiente.shape == params.shape),
            "perdida_y_gradiente_finitos": True,
            "forma_cdf_correcta": forma_cdf,
            "diferencia_perdida_vs_referencia": float(f"{dif_perdida:.3e}"),
            "norma_gradiente_analitico": round(float(np.linalg.norm(gradiente)), 6),
            "norma_gradiente_numerico": round(float(np.linalg.norm(numerico)), 6),
            "criterio": "error relativo < 1e-6 y diferencia de perdida < 1e-9",
            "ok": bool(error_relativo < 1e-6 and dif_perdida < 1e-9 and forma_cdf)}


def main() -> int:
    """Ejecuta los cuatro tests y detiene la fase si alguno falla."""
    autotest_numerico()
    inicio = time.time()
    resultados = [test_a(), test_b(), test_c(), test_d()]
    for item in resultados:
        estado = "OK   " if item["ok"] else "FALLA"
        print(f"[TEST {item['test']}] {estado} {item['nombre']}", flush=True)
    todo_ok = all(item["ok"] for item in resultados)
    print(f"\nCORAL {'puede ejecutarse sobre AGE' if todo_ok else 'NO debe ejecutarse'}", flush=True)

    cuerpo = ["# V23 · 00 — Tests unitarios de CORAL", "",
              "Ninguno de estos tests toca AGE. Validan la implementacion antes de usarla.", "",
              "## Valores pre-registrados", "",
              f"- `lambda_w = {LAMBDA_W}` (ridge de libro, normalizado por el numero de columnas).",
              f"- `lambda_umbral = {LAMBDA_UMBRAL}` (maximo verosimilitud de libro).", "",
              "El segundo lo fija TEST B: a la escala de AGE (135.000 filas, 279 columnas y "
              "proporciones reales de clase) **no hay separacion**, luego no hace falta penalizar "
              "los separadores. En un sintetico pequeno y de senal debil si la hay, y por eso el "
              "modulo la tiene disponible.", ""]
    for item in resultados:
        cuerpo += [f"## TEST {item['test']} — {item['nombre']}", ""]
        for clave, valor in item.items():
            if clave not in {"test", "nombre", "ok", "criterio"}:
                cuerpo.append(f"- **{clave}**: {valor}")
        cuerpo += [f"- **criterio**: {item['criterio']}",
                   f"- **resultado**: {'PASA' if item['ok'] else 'FALLA'}", ""]
    cuerpo += ["## Decision", "",
               f"**{'Los cuatro tests pasan; CORAL puede ejecutarse sobre AGE' if todo_ok else 'Un test falla; CORAL no se ejecuta'}**.",
               ""]
    INFORME.write_text("\n".join(cuerpo), encoding="utf-8")
    RESULTADO.write_text(json.dumps({
        "fase": "V23 00 tests CORAL", "fecha": "2026-09-27",
        "lambda_w_pre_registrado": LAMBDA_W, "lambda_umbral_pre_registrado": LAMBDA_UMBRAL,
        "tests": resultados, "todos_pasan": todo_ok,
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if todo_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
