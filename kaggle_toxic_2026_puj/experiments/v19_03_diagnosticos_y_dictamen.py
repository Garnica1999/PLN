"""V19-03: diagnosticos de distribucion en test, curva historica score-LB, e informe final.

Que queda por mirar y por que
-----------------------------
P0 a P2 han establecido dos cosas: que la metrica esta verificada por dos vias, y que la decision
se separa en tres problemas de umbral porque la funcion objetivo no tiene terminos cruzados. Eso
deja fuera toda la maquinaria de decision conjunta. Quedan dos diagnosticos, ninguno de los dos es
un experimento, y los dos hacen falta para cerrar la fase con honestidad.

P8, distribucion de test
------------------------
El test se puede usar **solo** para transformarlo en inferencia, nunca para entrenar ni para inferir
etiquetas individuales. Medir su tasa de positivos y la divergencia respecto al train es
transformacion legitima, y dice si hay un desplazamiento que explique el hueco entre OOF y LB. Si lo
hubiera, seria la unica palanca que queda; si no, se descarta por medicion y no por opinion.
Convertir una diferencia de distribucion en una etiqueta artificial esta prohibido y este script no
lo hace: no escribe ninguna submission.

Lo que el proyecto **no** tiene, y por tanto aqui no se mide: las probabilidades de V03 sobre el
test. Solo esta la submission binaria congelada. Reconstruir los scores exigiria volver a ejecutar el
pipeline sobre el test, que es inference legitima pero no hace falta, porque la magnitud que importa
para el desplazamiento es la tasa positiva de las decisiones, y esa esta exacta en la submission.
Sustituir los scores por dos valores inventados, que es lo que se hizo en una primera version de este
script, seria fabricar una distribucion, y se descarto.

P9, curva historica score-LB
----------------------------
Con las submissions historicas se busca si alguna propiedad de la submission se relaciona con su LB:
score OOF, tasa positiva, numero de cambios respecto a V03 y distribucion por etiqueta. **No se
entrena nada sobre el leaderboard**, que es una evaluacion externa, y no se afirma causalidad. Se
miden correlaciones y se informa del tamano de muestra, que es pequeno y limita lo que puede
concluirse.

El criterio de la fase
----------------------
El champion solo cambia si el LB supera 0.78306. Este script no envia nada: su unico producto es el
dictamen, y el dictamen se apoya en la escala de cambios netos, que es lo que mueve el leaderboard, y
no en la mejora de la metrica en el papel.

Uso
---
    /home/carlos/venv/bin/python experiments/v19_03_diagnosticos_y_dictamen.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

EXPERIMENTO = "V19-03 (P8 diagnosticos de test, P9 curva historica score-LB e informe final)"
REPORTE_MD = PROJECT_ROOT / "reports" / "v19_03_diagnosticos_y_dictamen.md"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v19_03_diagnosticos_y_dictamen.json"
CSV_TEST = PROJECT_ROOT / "resultados" / "v19_07_distribucion_test.csv"
CSV_HISTORICA = PROJECT_ROOT / "resultados" / "v19_08_curva_historica.csv"
CSV_EXPERIMENTOS = PROJECT_ROOT / "resultados" / "v19_09_tabla_de_experimentos.csv"

ETIQUETAS = ("HS", "TR", "AG")
CORRECTED_CV_V03 = 0.7772114525165721
LB_V03 = 0.78306
UMBRALES_V03 = np.array([0.44, 0.51, 0.46])
CELDAS_POR_003_LB = 107
# 107 cambios netos mueven 0.003 de LB: esta es la sensibilidad con la que un efecto medido en el
# OOF se traduce a su equivalente esperado en el leaderboard.
LB_POR_CELDA_NETA = 0.003 / CELDAS_POR_003_LB


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V19-03] {mensaje}", flush=True)


def main() -> None:
    """Ejecuta los dos diagnosticos y emite el dictamen de la fase."""
    inicio = time.time()
    log("V19-03: diagnosticos de test, curva historica y dictamen")

    log("1/5 P8, la distribucion del test, medida y no supuesta")
    train = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "toxic_train.csv", sep=";",
                        encoding="utf-8")
    test = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "toxic_test.csv", sep=";",
                       encoding="latin-1")
    prevalencias = {c: float(train[c].mean()) for c in ETIQUETAS}
    log(f"   train {len(train)} filas | test {len(test)} filas")
    log(f"   prevalencias de train: {', '.join(f'{k} {v:.6f}' for k, v in prevalencias.items())}")
    if any(c in test.columns for c in ETIQUETAS):
        raise SystemExit(
            "PARADA OBLIGATORIA: el test trae columnas de etiqueta, luego no es el conjunto sin "
            "verdad y el diagnostico no seria legitimo"
        )
    log("   el test no trae ninguna columna de etiqueta, luego el diagnostico es legitimo")

    diagnostico = _distribucion_de_test(PROJECT_ROOT)
    for registro in diagnostico:
        log(
            f"   {registro['magnitud']:36s} OOF {registro['oof']:8.4f} | test "
            f"{registro['test']:8.4f} | diferencia {registro['diferencia']:+.4f}"
        )
    with CSV_TEST.open("w", encoding="utf-8", newline="") as manejador:
        campos = ["magnitud", "oof", "test", "diferencia"]
        escritor = csv.DictWriter(manejador, fieldnames=campos, lineterminator="\n")
        escritor.writeheader()
        for f in diagnostico:
            escritor.writerow(f)

    log("2/5 P8b, que significa ese desplazamiento en el leaderboard")
    media = diagnostico[-1]
    desplazamiento = media["diferencia"]
    implicito = desplazamiento / LB_POR_CELDA_NETA
    log(
        f"   la tasa positiva predicha se desplaza {desplazamiento:+.4f}, y con la sensibilidad de "
        f"{LB_POR_CELDA_NETA:.6f} de LB por celda neta eso equivale a {implicito:+.0f} cambios netos"
    )
    log("   **no se convierte en ninguna etiqueta**: se registra como diagnostico y no se usa para")
    log("   construir ninguna submission, porque el desplazamiento ya esta incorporado en el modelo")

    log("3/5 P9, la curva historica score-LB sobre las submissions ya enviadas")
    registro = pd.read_csv(PROJECT_ROOT / "submissions" / "leaderboard_log.csv")
    log(f"   registro con {len(registro)} submissions")
    # Los nombres reales de las columnas del registro, no los que se suponian.
    correlaciones: dict[str, Any] = {}
    if "kaggle_f1" in registro.columns:
        candidatas = [c for c in ("local_f1", "diferencia") if c in registro.columns]
        log(f"   propiedades disponibles para contrastar: {', '.join(candidatas) or 'ninguna'}")
        for candidata in candidatas:
            pareja = registro[[candidata, "kaggle_f1"]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(pareja) > 2:
                r = float(pareja[candidata].corr(pareja["kaggle_f1"]))
                correlaciones[candidata] = {"correlacion": r, "n": len(pareja)}
                log(
                    f"   {candidata:12s} correlacion con kaggle_f1 {r:+.4f} sobre "
                    f"{len(pareja):2d} submissions"
                )
        mejores = registro.apply(pd.to_numeric, errors="coerce").dropna(subset=["kaggle_f1"])
        top = mejores.nlargest(3, "kaggle_f1")[["archivo", "local_f1", "kaggle_f1"]]
        log("   los tres mejores scores del registro:")
        for _, fila in top.iterrows():
            log(
                f"     {str(fila['archivo'])[:40]:40s} local {fila['local_f1']:.5f} | "
                f"kaggle {fila['kaggle_f1']:.5f}"
            )
    log(f"   {len(registro)} puntos es una muestra pequena: una correlacion baja aqui no descarta")
    log("   nada, y una correlacion alta tampoco probaria causalidad")
    registro.to_csv(CSV_HISTORICA, index=False, lineterminator="\n")

    log("4/5 la tabla de experimentos de la fase")
    desde_v00 = _leer("v19_00_metrica_verificada.json")
    desde_v01 = _leer("v19_01_valor_de_cada_correccion.json")
    desde_v02 = _leer("v19_02_restriccion_tr_implica_hs.json")
    tabla = [
        {
            "experimento": "V19-00 metrica verificada",
            "mecanismo": "reconstruccion de la metrica y reduccion del average",
            "oof": CORRECTED_CV_V03,
            "delta_oof": 0.0,
            "cambios_netos": 0,
            "lb": "",
            "delta_lb": "",
            "estado": "diagnostico, fija el objetivo de la fase",
        },
        {
            "experimento": "V19-01 valor de cada correccion",
            "mecanismo": "formas cerradas del delta de F1 y estructura conjunta",
            "oof": desde_v01["base_reproducida"],
            "delta_oof": 0.0,
            "cambios_netos": 0,
            "lb": "",
            "delta_lb": "",
            "estado": "diagnostico, demuestra que la decision se separa",
        },
        {
            "experimento": "V19-02 restriccion TR implica HS",
            "mecanismo": "propagar HS=1 donde el score de TR supera el gatillo",
            "oof": CORRECTED_CV_V03 + desde_v02["delta_medio_fuera_de_muestra"],
            "delta_oof": desde_v02["delta_medio_fuera_de_muestra"],
            "cambios_netos": desde_v02["cambios_netos_de_celda"],
            "lb": "",
            "delta_lb": "",
            "estado": desde_v02["estado"],
        },
    ]
    with CSV_EXPERIMENTOS.open("w", encoding="utf-8", newline="") as manejador:
        campos = list(tabla[0].keys())
        escritor = csv.DictWriter(manejador, fieldnames=campos, lineterminator="\n")
        escritor.writeheader()
        for f in tabla:
            escritor.writerow(f)
    for f in tabla:
        log(
            f"   {f['experimento']:36s} dOOF {f['delta_oof']:+.6f} | netos "
            f"{f['cambios_netos']:+4d} | {f['estado']}"
        )
    log("   ninguna fila tiene LB porque **no se envio ninguna submission**")

    log("5/5 el dictamen")
    dictamen = _dictamen(desde_v00, desde_v01, desde_v02, diagnostico, correlaciones, inicio)
    _escribir(dictamen, desde_v00, desde_v01, desde_v02, diagnostico, correlaciones, tabla, inicio)
    log(f"completado en {round(time.time() - inicio, 1)} s")
    log(
        f"VEREDICTO: {dictamen['estado']} | champion {dictamen['champion']['version']} con LB "
        f"{LB_V03} | submissions gastadas 0 | conservadas 5"
    )


def _leer(nombre: str) -> dict[str, Any]:
    """Lee un informe JSON de la fase.

    Args:
        nombre: nombre del fichero dentro de reports.

    Returns:
        El contenido del informe.
    """
    ruta = PROJECT_ROOT / "reports" / nombre
    if not ruta.exists():
        raise SystemExit(f"PARADA OBLIGATORIA: falta {ruta.name}, no se puede cerrar la fase")
    return json.loads(ruta.read_text(encoding="utf-8"))


def _distribucion_de_test(raiz: Path) -> list[dict[str, Any]]:
    """Mide la tasa positiva predicha por V03 en OOF y en test, y su diferencia.

    La magnitud se toma de la submission binaria congelada, que contiene exactamente las decisiones
    de V03 sobre el test. No se usan probabilidades de test porque el proyecto no las guarda, y
    sustituirlas por valores inventados seria fabricar una distribucion.

    Args:
        raiz: raiz del proyecto.

    Returns:
        Una entrada por magnitud, con OOF, test y diferencia.
    """
    datos = np.load(raiz / "features" / "oof_v03_final.npz", allow_pickle=True)
    pred_oof = (np.asarray(datos["proba"], dtype=float) >= UMBRALES_V03[None, :]).astype(int)
    submission = pd.read_csv(raiz / "submissions" / "toxic_submission_v03_lr_union_thr.csv")
    pred_test = submission[list(ETIQUETAS)].to_numpy(dtype=int)
    salida = []
    for j, etiqueta in enumerate(ETIQUETAS):
        oof = float(pred_oof[:, j].mean())
        test = float(pred_test[:, j].mean())
        salida.append(
            {
                "magnitud": f"tasa positiva predicha en {etiqueta}",
                "oof": oof,
                "test": test,
                "diferencia": test - oof,
            }
        )
    oof_media = float(np.mean([pred_oof[:, j].mean() for j in range(3)]))
    test_media = float(np.mean([pred_test[:, j].mean() for j in range(3)]))
    salida.append(
        {
            "magnitud": "media de la tasa positiva sobre las tres",
            "oof": oof_media,
            "test": test_media,
            "diferencia": test_media - oof_media,
        }
    )
    return salida


def _dictamen(
    desde_v00: dict[str, Any],
    desde_v01: dict[str, Any],
    desde_v02: dict[str, Any],
    diagnostico: list[dict[str, Any]],
    correlaciones: dict[str, Any],
    inicio: float,
) -> dict[str, Any]:
    """Emite el dictamen de la fase.

    Args:
        desde_v00: veredicto de la metrica.
        desde_v01: veredicto del valor y la estructura.
        desde_v02: veredicto de la regla.
        diagnostico: distribucion de test medida.
        correlaciones: correlaciones historicas con el leaderboard.
        inicio: instante de arranque.

    Returns:
        El dictamen.
    """
    return {
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estado": "NO VALID SCORE IMPROVEMENT",
        "champion": {"version": "V03", "lb": LB_V03, "delta_lb": 0.0},
        "submissions_realizadas": 0,
        "submissions_conservadas": 5,
        "por_que_no_hay_mejora": [
            (
                "la funcion objetivo no tiene terminos cruzados entre etiquetas, luego la decision "
                "se separa en tres problemas de umbral y el optimo es un umbral por etiqueta. Toda "
                "la maquinaria de decision conjunta se queda sin justificacion"
            ),
            (
                "el unico hecho de estructura real es que TR implica HS, y la regla que de ahi sale "
                "es genuinamente distinta de un umbral, pero aporta -3 cambios netos de celda frente "
                "a los 27 que haria falta para mover el leaderboard, y solo mejora 3 de 5 pliegues"
            ),
            (
                "el valor de una correccion no es uniforme, pero eso no abre ninguna palanca nueva: "
                "solo decide que correccion merece la pena, y la decision sigue siendo un umbral"
            ),
            (
                "V16 cerro umbrales, tasa y capacidad; V17 cerro los ensembles historicos; V18 cerro "
                "la correccion selectiva. Ninguna de esas lineas se reabre"
            ),
        ],
        "lo_que_sigue_en_pie": (
            "V03 es el mejor score observado bajo los experimentos competitivos realizados. Eso no "
            "es una afirmacion de que 0.78306 sea el optimo matematico, y no se sostendria sin haber "
            "explorado la funcion objetivo entera, que es continua y tiene muchos puntos locales"
        ),
        "que_haria_falta_para_otra_fase": (
            "una fuente de informacion nueva, o un mecanismo con una diferencia algebraica "
            "identificable. La unica palanca que este analisis deja sin explorar del todo es la "
            "estructura de etiquetas, y de ella solo se ha encontrado una restriccion, que ya se ha "
            "probado y descartado. Una interaccion multiplicativa entre scores sigue sin ejecutar, "
            "pero la separacion del objetivo muestra que sin terminos cruzados no puede mejorar a "
            "un umbral por etiqueta, asi que solo tendria sentido si atacase informacion que el "
            "score de HS no contiene y que el score de TR si"
        ),
        "metric_verificada": {
            "contrastes_celda_a_celda_contra_sklearn": desde_v00["verificaciones"][
                "contrastes_contra_sklearn"
            ],
            "discrepancias": desde_v00["verificaciones"]["discrepancias"],
            "corrected_cv_reproducido": desde_v00["verificaciones"]["corrected_cv_reproducido"],
        },
        "estructura": {
            "estados_conjuntos_observados": desde_v01["estados_conjuntos_observados"],
            "valores_distintos_de_ganancia": desde_v01["valores_distintos_de_ganancia"],
        },
        "test_diagnostico": diagnostico,
        "correlaciones_historicas": correlaciones,
        "gastos": {"submissions": 0, "modelos_entrenados": 0, "meta_modelos": 0},
        "minutos": round((time.time() - inicio) / 60, 2),
    }


def _escribir(
    dictamen: dict[str, Any],
    desde_v00: dict[str, Any],
    desde_v01: dict[str, Any],
    desde_v02: dict[str, Any],
    diagnostico: list[dict[str, Any]],
    correlaciones: dict[str, Any],
    tabla: list[dict[str, Any]],
    inicio: float,
) -> None:
    """Escribe el informe final de la fase.

    Args:
        dictamen: dictamen emitido.
        desde_v00: veredicto de la metrica.
        desde_v01: veredicto del valor y la estructura.
        desde_v02: veredicto de la regla.
        diagnostico: distribucion de test.
        correlaciones: correlaciones historicas.
        tabla: tabla de experimentos.
        inicio: instante de arranque.
    """
    from v11_ayudas import fila, sello

    cuerpo: list[str] = [
        "# V19: analisis de metrica y score engineering",
        "",
        f"Generado por `experiments/v19_03_diagnosticos_y_dictamen.py` ({EXPERIMENTO}).",
        "",
        f"**{dictamen['estado']}**",
        "",
        "## 1. Champion",
        "",
        "| version | LB | delta LB |",
        "| --- | ---: | ---: |",
        fila(["V03", f"{LB_V03:.5f}", "+0.00000"]),
        "",
        "Sin cambios. Ninguna submission se ha enviado en esta fase.",
        "",
        "## 2. Analisis de la metrica",
        "",
        "### La formula, verificada por dos vias",
        "",
        f"- Scorer: {desde_v00['scorer']}.",
        f"- `zero_division`: {desde_v00['resuelto']['zero_division']}.",
        f"- `average=\"binary\"`: {desde_v00['resuelto']['average_binary']}.",
        "- La media de las tres F1 positivas coincide exactamente con `f1_score(average='macro')` de",
        "  sklearn sobre entrada con tres o mas columnas.",
        f"- Casos sinteticos: **{desde_v00['verificaciones']['casos_sinteticos']}** comprobaciones.",
        (
            f"- Contrastes contra sklearn: "
            f"**{desde_v00['verificaciones']['contrastes_contra_sklearn']}**, con "
            f"**{desde_v00['verificaciones']['discrepancias']} discrepancias**."
        ),
        (
            f"- `corrected_cv` de V03 reproducido: "
            f"**{desde_v00['verificaciones']['corrected_cv_reproducido']:.16f}**."
        ),
        "",
        "### La sensibilidad por celda",
        "",
        "Con D = 2*TP + FP + FN:",
        "",
        "```",
        "FN -> TP   Delta =  2 (TP + FP + FN) / (D (D + 1))",
        "FP -> TN   Delta =  2 TP             / (D (D - 1))",
        "```",
        "",
        f"En el OOF de V03 hay **{desde_v01['valores_distintos_de_ganancia']} valores distintos** de",
        "ganancia por celda, no uno. Corregir un falso negativo y eliminar un falso positivo valen",
        "distinto, luego contar errores no es medir su valor.",
        "",
        "### La estructura conjunta",
        "",
        "**TR=1 implica HS=1**: no hay ninguna de las 1483 filas con TR=1 y HS=0, cuando por",
        "independencia se esperarian 250.6. Y es de una sola direccion, porque AG=1 con HS=0 si",
        "ocurre, en 22 filas. Es un hecho medido, no una intuicion sobre el significado de las etiquetas.",
        "",
        "## 3. Experimentos",
        "",
        fila(["experimento", "mecanismo", "OOF", "delta OOF", "netos", "LB", "estado"]),
        fila(["---"] * 7),
    ]
    for f in tabla:
        cuerpo.append(
            fila(
                [
                    f["experimento"], f["mecanismo"][:36], f"{f['oof']:.6f}",
                    f"{f['delta_oof']:+.6f}", f"{f['cambios_netos']:+d}", f["lb"] or "no enviada",
                    f["estado"],
                ]
            )
        )
    cuerpo += [
        "",
        "Ninguna fila tiene LB porque no se envio ninguna submission.",
        "",
        f"La regla de V19-02 mejora el OOF en {desde_v02['delta_medio_fuera_de_muestra']:+.6f} de",
        f"media, pero con desviacion {desde_v02['delta_desviacion']:.6f},",
        f"{desde_v02['pliegues_que_mejoran']} de {desde_v02['pliegues_evaluados']} pliegues",
        f"mejorando, valor p empirico {desde_v02['p_empirico']:.4f} y",
        f"**{desde_v02['cambios_netos_de_celda']:+d} cambios netos** de celda frente a los",
        f"{desde_v02['cambios_netos_necesarios']:.0f} que haria falta para mover el leaderboard.",
        "Un cambio neto negativo no puede mover el LB con la sensibilidad del proyecto, y por eso no",
        "se gasto una submission en averiguarlo.",
        "",
        "## 4. Submissions",
        "",
        "**Ninguna.** La regla de la fase era no gastar sin evidencia competitiva, y la evidencia no",
        "llego: la regla nueva aporta un numero negativo de cambios netos de celda.",
        "",
        "## 5. Por que no hay mejora",
        "",
    ]
    for motivo in dictamen["por_que_no_hay_mejora"]:
        cuerpo.append(f"- {motivo}")
    cuerpo += [
        "",
        "## 6. Lo que sigue en pie, y lo que haria falta para otra fase",
        "",
        dictamen["lo_que_sigue_en_pie"] + ".",
        "",
        dictamen["que_haria_falta_para_otra_fase"] + ".",
        "",
        "## 7. Diagnostico de la distribucion de test",
        "",
        "El proyecto no guarda las probabilidades de V03 sobre el test, solo la submission binaria",
        "congelada, y aqui se mide la tasa positiva de esas decisiones, que es exacta. No se",
        "sustituyen los scores por valores inventados.",
        "",
        fila(["magnitud", "OOF", "test", "diferencia"]),
        fila(["---"] * 4),
    ]
    for f in diagnostico:
        cuerpo.append(
            fila([f["magnitud"][:40], f"{f['oof']:.4f}", f"{f['test']:.4f}",
                  f"{f['diferencia']:+.4f}"])
        )
    if correlaciones:
        cuerpo += [
            "",
            "## 8. Correlaciones historicas con el leaderboard",
            "",
            fila(["propiedad", "correlacion con LB", "n"]),
            fila(["---"] * 3),
        ]
        for clave, valor in correlaciones.items():
            cuerpo.append(fila([clave, f"{valor['correlacion']:+.4f}", str(valor["n"])]))
    cuerpo += [
        "",
        "El leaderboard es una evaluacion externa: no se ha entrenado nada sobre el, y una",
        "correlacion con veinte puntos no probaria causalidad en ningun caso.",
        "",
    ]
    cuerpo += sello(inicio)
    REPORTE_MD.write_text("\n".join(cuerpo) + "\n", encoding="utf-8")
    REPORTE_JSON.write_text(json.dumps(dictamen, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")


if __name__ == "__main__":
    main()
