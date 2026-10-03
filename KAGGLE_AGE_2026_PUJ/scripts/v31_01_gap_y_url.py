"""V31+ - 01 - P1 y P2: gap analysis de las familias y analisis de URLs sobre train.

Dos pasos en uno porque el segundo condiciona al primero: **no tiene sentido listar que el Track A
esta abierto sin haber mirado que dice el texto**, y no tiene sentido proponer una hipotesis de URL
sin saber cuanto hay.

## Lo que este script NO hace

No entrena, no lee el test para ninguna decision, no hace ninguna llamada de red y no usa
etiquetas fuera del train. El analisis de URL es **descriptivo sobre train** y su unico producto es
saber si la linea merece la pena.

## La distincion que el brief exige

```text
same hypothesis with different hyperparameter
```

frente a

```text
genuinely new mechanism
```

Una familia cuenta como cerrada si el historial la cerro **por equivalencia funcional**, no solo si
le cambio el nombre. Y al reves: una cuenta como abierta si el mecanismo no esta implementado en
ningun sitio, aunque el nombre suene a algo ya visto.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg  # noqa: E402
from src import data as datos  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

RESULTADO = cfg.EXPERIMENTS / "v31_01_gap_y_url.json"
GAP_MD = cfg.REPORTS / "v31_hypothesis_gap_analysis.md"
GAP_CSV = cfg.REPORTS / "v31_hypothesis_gap_analysis.csv"
URL_CSV = cfg.REPORTS / "v31_url_analysis.csv"
REGISTRY = cfg.REPORTS / "v31_hypothesis_registry.csv"
EDA_CSV = cfg.REPORTS / "v31_linguistic_eda.csv"
FECHA = "2026-09-30"
CV_V30 = 0.6134471  # baseline operativo, ver v31_00_auditoria.py
N_CLASES = 6
#: Regex de deteccion de URL. Deliberadamente simple y explicito: no se usa ningun parser externo.
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
SHORTENER = {"bit.ly", "t.co", "goo.gl", "tinyurl.com", "ow.ly", "is.gd", "buff.ly",
             "rebrand.ly", "cutt.ly", "trib.al", "shorturl.at", "rb.gy", "lnkd.in"}
DOMINIO_RE = re.compile(r"^(?:https?://)?(?:www\.)?([^/\s?#]+)", re.IGNORECASE)


# ============================================================================ P2: URLs
def _url_por_fila(texto: str) -> dict:
    """Taxonomia de URL de una fila, calculada **solo** con el texto del train.

    No se consulta nada externo, no se resuelve nada, no se hace ninguna llamada de red. Solo se
    cuenta lo que el propio texto dice.
    """
    urls = URL_RE.findall(texto or "")
    salida = {"n_urls": len(urls), "dominios": [], "categorias": []}
    for url in urls:
        m = DOMINIO_RE.match(url.strip())
        dominio = (m.group(1).lower() if m else "").rstrip("/")
        salida["dominios"].append(dominio)
        ruta = url.split("?")[0].lower()
        if not dominio:
            categoria = "sin_dominio"
        elif dominio in ("twitter.com", "x.com", "mobile.twitter.com", "t.co"):
            if "/status/" in ruta or "/i/web/status/" in ruta or "/statuses/" in ruta:
                categoria = "twitter_status"
            else:
                categoria = "twitter_otro"
        elif dominio in SHORTENER:
            categoria = "acortador"
        else:
            categoria = "url_externa"
        salida["categorias"].append(categoria)
    salida["tiene_url"] = len(urls) > 0
    # Twitter status incluye un identificador numerico largo: es la forma de una cita direta.
    salida["es_status"] = "twitter_status" in salida["categorias"]
    salida["es_acortador"] = "acortador" in salida["categorias"]
    salida["es_externa"] = "url_externa" in salida["categorias"]
    salida["solo_url"] = bool(urls) and len(re.sub(URL_RE, "", texto or "").strip()) < 3
    salida["texto_mas_url"] = bool(urls) and not salida["solo_url"]
    salida["multiples"] = len(urls) > 1
    salida["url_con_query"] = any("?" in u for u in urls)
    return salida


def analisis_url() -> tuple[pd.DataFrame, dict]:
    """Taxonomia de URL por clase, sobre **train** unicamente."""
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    textos = train["text"].fillna("").astype(str).to_numpy()
    registros = [_url_por_fila(t) for t in textos]
    tabla = pd.DataFrame({
        "clase": y,
        "n_urls": [r["n_urls"] for r in registros],
        "url": [r["tiene_url"] for r in registros],
        "status": [r["es_status"] for r in registros],
        "acortador": [r["es_acortador"] for r in registros],
        "externa": [r["es_externa"] for r in registros],
        "solo_url": [r["solo_url"] for r in registros],
        "texto_mas_url": [r["texto_mas_url"] for r in registros],
        "multiples": [r["multiples"] for r in registros],
        "con_query": [r["url_con_query"] for r in registros],
        "longitud_texto": [len(t) for t in textos],
        "longitud_url_max": [max((len(u) for u in URL_RE.findall(t)), default=0) for t in textos],
    })
    filas = []
    for clase in cfg.CLASSES:
        sub = tabla[tabla["clase"] == clase]
        base = tabla
        filas.append({
            "medida": "prevalencia de URL", "clase": clase,
            "valor": round(float(sub["url"].mean()), 6),
            "global": round(float(base["url"].mean()), 6),
            "enriquecimiento": round(float(sub["url"].mean() / max(base["url"].mean(), 1e-9)), 4),
        })
    for nombre, columna in (("twitter status", "status"), ("URL acortada", "acortador"),
                            ("URL externa", "externa"), ("solo URL", "solo_url"),
                            ("texto + URL", "texto_mas_url"), ("multiples URL", "multiples"),
                            ("URL con query string", "con_query")):
        for clase in cfg.CLASSES:
            sub = tabla[tabla["clase"] == clase]
            filas.append({
                "medida": nombre, "clase": clase,
                "valor": round(float(sub[columna].mean()), 6),
                "global": round(float(tabla[columna].mean()), 6),
                "enriquecimiento": round(float(sub[columna].mean()
                                               / max(float(tabla[columna].mean()), 1e-9)), 4),
            })
    for clase in cfg.CLASSES:
        sub = tabla[tabla["clase"] == clase]
        filas.append({"medida": "longitud media del texto", "clase": clase,
                      "valor": round(float(sub["longitud_texto"].mean()), 2),
                      "global": round(float(tabla["longitud_texto"].mean()), 2),
                      "enriquecimiento": round(float(sub["longitud_texto"].mean()
                                                     / max(float(tabla["longitud_texto"].mean()),
                                                           1e-9)), 4)})
        filas.append({"medida": "longitud media de la URL mas larga", "clase": clase,
                      "valor": round(float(sub["longitud_url_max"].mean()), 2),
                      "global": round(float(tabla["longitud_url_max"].mean()), 2),
                      "enriquecimiento": round(float(sub["longitud_url_max"].mean()
                                                     / max(float(tabla["longitud_url_max"].mean()),
                                                           1e-9)), 4)})
    # Solo filas con URL: longitud del texto que las rodea.
    con_url = tabla[tabla["url"]]
    filas.append({"medida": "longitud media del texto en filas CON url", "clase": "todas",
                  "valor": round(float(con_url["longitud_texto"].mean()), 2), "global": "",
                  "enriquecimiento": round(float(con_url["longitud_texto"].mean())
                                            / max(float(tabla["longitud_texto"].mean()), 1e-9), 4)})
    filas.append({"medida": "longitud media del texto en filas SIN url", "clase": "todas",
                  "valor": round(float(tabla[~tabla["url"]]["longitud_texto"].mean()), 2),
                  "global": "", "enriquecimiento": ""})
    salida = pd.DataFrame(filas)
    salida.to_csv(URL_CSV, index=False)

    resumen = {
        "filas_train": int(len(tabla)),
        "prevalencia_url": round(float(tabla["url"].mean()), 6),
        "filas_con_url": int(tabla["url"].sum()),
        "twitter_status": int(tabla["status"].sum()),
        "solo_url": int(tabla["solo_url"].sum()),
        "texto_mas_url": int(tabla["texto_mas_url"].sum()),
        "multiples_url": int(tabla["multiples"].sum()),
        "acortadores": int(tabla["acortador"].sum()),
        "url_externa": int(tabla["externa"].sum()),
        "url_con_query": int(tabla["con_query"].sum()),
        "longitud_media_texto": round(float(tabla["longitud_texto"].mean()), 2),
        "longitud_media_texto_con_url": round(float(con_url["longitud_texto"].mean()), 2),
        "longitud_media_texto_sin_url": round(float(tabla[~tabla["url"]]["longitud_texto"].mean()), 2),
        "rango_enriquecimiento_url_por_clase": [
            float(f["enriquecimiento"]) for f in filas
            if f["medida"] == "prevalencia de URL" and f["enriquecimiento"] != ""],
        "metodo": "regex explicito sobre el texto del train; ninguna llamada de red, ningun parser "
                  "externo, ningun modelo preentrenado",
    }
    return tabla, resumen


# ============================================================================ P1: gap analysis
FAMILIAS = [
    # (id, track, familia, ya_probada, resultado_historico, cerrada, por_que, nueva_formulacion,
    #  mecanismo_esperado, prioridad)
    ("F01", "A", "URL: presencia y estructura superficial", "A1-A3",
     "nunca medida: no hay columna de URL en el inventario de 279 features", "no",
     "", "taxonomia de URL como canal separado, y comoplaceholder estructural",
     "si el modelo usa la identidad textual de la URL en vez de su estructura, un canal de "
     "dominio y categoria aporta lo que la superficie no expresa", 1),
    ("F02", "A", "URL: recuperacion del contenido del tweet", "A5-A8",
     "nunca intentada", "si, por cuarentena",
     "las reglas de la compilacion no estan disponibles: sin el texto no se puede establecer que "
     "sea legal, y recuperarlo puede ser informacion externa o fuga de metadatos",
     "", "", 9),
    ("F03", "B", "EDA lexica basica", "advanced_* y fase03",
     "hay auditorias lexicales previas, pero orientadas a seleccionar features, no a caracter "
     "composicional por edad", "parcialmente",
     "", "perfil composicional por clase: ratios, cuantiles y enriquecimiento de familias "
     "funcionales", "la edad puede manifestarse en como se compone el texto mas que en que se dice",
     2),
    ("F04", "B", "palabras funcionales: pronombres", "v10 y familias de pronouns",
     "si, en v10 y en la bloque de label_informed", "si",
     "las familias de pronombres se cerraron al cerrar la seleccion de features, y lo que se "
     "cerro fue su uso como feature individual, no su perfil composicional", "", "", 6),
    ("F05", "C", "verbos y tiempo verbal", "no medida como composicion",
     "no", "no", "", "verbos auxiliares, imperativos, marcas de pasado/presente/futuro",
     "el uso del auxiliar y la marca temporal varia mucho entre generaciones y es determinista", 3),
    ("F06", "D", "ortografia y estilo de escritura", "features de estilo existentes",
     "si: hay 87 features estructuradas que incluyen longitud, mayusculas y ratios de puntuacion",
     "parcialmente",
     "el bloque estructurado ya captura longitud y densidad de puntuacion, luego solo lo que no "
     "este dentro de el sigue abierto", "elongacion, identidad del caracter repetido, bloques "
     "caps, repeticiones de puntuacion", "la ortografia de tecleo es una huella fuerte de edad y es "
     "totalmente determinista", 2),
    ("F07", "E", "discurso y pragmatica", "no medida",
     "no", "no", "", "comillas, direccion directa, preguntas, risas, intensificadores",
     "los marcadores discursivos cambian de forma sistematica entre generaciones", 3),
    ("F08", "E", "estructura de citas y mencion", "posicion de mencion",
     "parcial: hay features de posicion de mencion en el bloque estructurado", "parcialmente",
     "", "posicion de la cita y de la mencion como categoria combinatoria, no como coordenada",
     "la estructura del turno de habla es generacional", 4),
    ("F09", "F", "canales por posicion", "posicion de keywords",
     "si, en familias tempranas", "si",
     "se cerro el uso de posiciones de keywords como representacion densa; lo que no se probo es "
     "el canal de composicion por posicion", "", "", 5),
    ("F10", "H", "representaciones sparse nuevas", "word y char ngrams",
     "si: TF-IDF word y char, n-grams y skipgrams se exhausting", "si",
     "cualquier familia que solo cambie C, seed o max_features esta cerrada por V25", "",
     "solo se abre lo que cambie el MECANISMO: condicionar la palabra a su forma, su posicion o su "
     "categoria", 2),
    ("F11", "I", "familias de modelo nuevas", "LightGBM, MLP, lineal, NB-SVM, BM25, kNN, "
     "ExtraTrees, SGD",
     "V29 demostro que la capacidad del meta-modelo si es un cuello", "no para capacidad",
     "", "solo ejecutar un modelo nuevo si la representacion nueva aporta senal demostrablemente "
     "distinta", "primero controles lineales, despues no lineal", 1),
    ("F12", "J", "mineria linguistica guiada por errores", "error analysis de V12-V22",
     "si, hay analisis de error extensos", "parcialmente",
     "los analisis previos usaron las confusiones para entender, no para construir representacion",
     "perfil de los errores de alta confianza de V30 por dimension linguistica",
     "un error con alta confianza apunta a una limitacion de representacion, no de umbral", 2),
    ("F13", "N", "integracion en el meta-espacio de V30", "integraciones previas",
     "V28 mostro que anadir al espacio de V30 sin senal nueva no rinde", "no",
     "", "anadir solo la representacion que ya tenga evidencia independiente, con controles de "
     "atribucion", "complementariedad real frente a capacidad extra", 1),
]

CERRADAS_PREVIAS = {
    "feature selection supervisada": "V25, D-060, -0,000629",
    "escalado afil de la evidencia": "V26-01, D-061, -0,000001",
    "transformacion percentil": "V27-05, MARGINAL +0,000272",
    "fiabilidad por modelo y clase": "V27-01, H12, delta 0",
    "restricciones monotonas": "V27-02, H11, -0,000355",
    "reponder blends equivalentes": "V27, H14, equivalencia funcional",
    "calibracion": "H01, D-029",
    "ponderacion de modelos": "H04",
    "apilado de segundo nivel": "H06",
    "familias ordinales": "H10, CORAL, D-058",
    "bagging del meta-modelo": "V28-02, A, -0,000312",
    "descomposicion pairwise": "V28-03, C, -0,067553",
    "geometria del consenso": "V28, B, +0,0001 historico",
    "fusion de evidencia": "V28, D, equivalente a H04+H06",
    "incertidumbre explicita": "V28, E, duplicacion de B",
    "subespacio aleatorio": "V28, F, ya en colsample_bytree",
    "capacidad del meta-modelo": "V29 y V30, +0,002461 y luego equivalente",
    "diversidad de modelos historicos": "V29, 0 de 47 con rescate neto positivo",
}


def main() -> int:
    """Ejecuta el analisis de URL y escribe el gap analysis."""
    autotest_numerico()
    inicio = time.time()
    print("=== P2: taxonomia de URL sobre TRAIN (sin red, sin test) ===", flush=True)
    tabla_url, resumen = analisis_url()
    print(f"  filas con URL: {resumen['filas_con_url']:,} de {resumen['filas_train']:,} "
          f"({resumen['prevalencia_url']:.2%})", flush=True)
    print(f"  twitter status: {resumen['twitter_status']:,} | solo URL: "
          f"{resumen['solo_url']:,} | texto+URL: {resumen['texto_mas_url']:,} | multiples: "
          f"{resumen['multiples_url']:,}", flush=True)
    print(f"  acortadores: {resumen['acortadores']:,} | externas: {resumen['url_externa']:,} | "
          f"con query: {resumen['url_con_query']:,}", flush=True)
    print(f"  longitud media del texto: {resumen['longitud_media_texto']:.1f} | con URL: "
          f"{resumen['longitud_media_texto_con_url']:.1f} | sin URL: "
          f"{resumen['longitud_media_texto_sin_url']:.1f}", flush=True)
    enr = resumen["rango_enriquecimiento_url_por_clase"]
    print(f"  enriquecimiento de prevalencia de URL por clase: de {min(enr):.3f} a {max(enr):.3f}",
          flush=True)

    # ---------------------------------------------------------------- gap analysis
    print("\n=== P1: gap analysis de las familias ===", flush=True)
    filas = []
    for fid, track, familia, probada, resultado, cerrada, porque, nueva, mecanismo, prioridad in FAMILIAS:
        filas.append({
            "ID": fid, "track": track, "family": familia, "already_tested": probada,
            "historical_result": resultado, "closed": cerrada, "why_closed": porque,
            "new_formulation": nueva, "expected_mechanism": mecanismo,
            "priority": prioridad,
            "status": "CLOSED" if cerrada == "si" else (
                "QUARANTINED" if "cuarentena" in cerrada else "OPEN"),
        })
    gap = pd.DataFrame(filas).sort_values("priority")
    gap.to_csv(GAP_CSV, index=False)
    abiertos = gap[gap["status"] == "OPEN"]
    print(f"  {len(gap)} familias | {len(abiertos)} abiertas | "
          f"{int((gap['status'] == 'CLOSED').sum())} cerradas | "
          f"{int((gap['status'] == 'QUARANTINED').sum())} en cuarentena", flush=True)
    for _, f in gap.iterrows():
        print(f"    {f['ID']} [{f['track']}] p{f['priority']} {f['status']:<11} {f['family'][:56]}",
              flush=True)

    # ---------------------------------------------------------------- registro
    registro = [{"ID": f["ID"], "Track": f["track"], "Hypothesis": f["new_formulation"] or f["family"],
                 "Novelty": "nueva" if f["status"] == "OPEN" else f["status"],
                 "Historical_evidence": f["historical_result"],
                 "Expected_mechanism": f["expected_mechanism"],
                 "Implementation": "pendiente", "Status": f["status"], "Result": "",
                 "Decision": ""} for _, f in gap.iterrows()]
    pd.DataFrame(registro).to_csv(REGISTRY, index=False)

    _escribe_md(gap, resumen, CERRADAS_PREVIAS)

    RESULTADO.write_text(json.dumps({
        "fase": "V31+ 01 P1 gap analysis y P2 analisis de URL", "fecha": FECHA,
        "experimenta": False, "baseline_v30": CV_V30,
        "url": resumen,
        "familias": gap.to_dict("records"),
        "abiertas": int(len(abiertos)), "cerradas": int((gap["status"] == "CLOSED").sum()),
        "cuarentena": int((gap["status"] == "QUARANTINED").sum()),
        "lectura_del_track_A": (
            f"la prevalencia de URL es del {resumen['prevalencia_url']:.2%} y la clase mas "
            f"enriquecida es {max(enr):.3f} veces la media global, la menos {min(enr):.3f}. "
            "Un enriquecimiento de esa magnitud es diagnostico de que la URL **esta** "
            "distribuida por edad, pero no dice si el modelo la esta leyendo. Eso solo lo "
            "responde A4, el experimento de enmascarar la URL"),
        "familias_cerradas_previas": CERRADAS_PREVIAS,
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[fin] {time.time() - inicio:.0f} s")
    return 0


def _escribe_md(gap: pd.DataFrame, url: dict, cerradas: dict) -> None:
    """Redacta el gap analysis legible."""
    abiertos = gap[gap["status"] == "OPEN"]
    lineas = [
        "# V31+ - Gap analysis de las familias de representacion", "",
        f"**Baseline congelado:** V30 = {CV_V30} corrected_cv, LB 0,63302. "
        "El 0,6130051 que se publico en V30 no era reproducible; el operativo es 0,6134471 y es el "
        "que se envio y el que casa con el leaderboard.", "",
        "## La distincion que manda", "",
        "Una familia esta **cerrada** si el historial la cerro por equivalencia funcional, aunque el "
        "nombre suene nuevo. Y una esta **abierta** si el mecanismo no esta implementado en ningun "
        "sitio, aunque el nombre suene viejo.", "",
        "## Familias", "",
        "| ID | Track | Familia | Probada antes | Cerrada | Prioridad | Formulacion nueva |",
        "|---|---|---|---|---|---|---|"]
    for _, f in gap.iterrows():
        lineas.append(f"| {f['ID']} | {f['track']} | {f['family']} | {f['already_tested']} | "
                      f"**{f['status']}** | {f['priority']} | "
                      f"{(f['new_formulation'] or '-')[:110]} |")
    lineas += ["", "## Las que quedan abiertas, con su mecanismo esperado", ""]
    for _, f in abiertos.iterrows():
        lineas += [f"### {f['ID']} · {f['family']}", "",
                   f"- **Historial:** {f['historical_result']}",
                   f"- **Por que no es un cambio de hiperparametro:** {f['new_formulation']}",
                   f"- **Mecanismo esperado:** {f['expected_mechanism']}", ""]
    cuarentena = gap[gap["status"] == "QUARANTINED"]
    if not cuarentena.empty:
        f = cuarentena.iloc[0]
        lineas += ["## La familia en cuarentena", "",
                   f"**{f['ID']} · {f['family']}** queda en **QUARANTINED**.", "",
                   f"- {f['why_closed']}", "",
                   "No se implementa, no se hace ninguna llamada de red y ningun candidato que "
                   "dependa de contenido recuperado puede llegar a una submission. El resto del "
                   "Track A, que usa solo el texto del train, no se ve afectado.", ""]
    lineas += ["## Lo que ya no se puede reabrir", "",
               "| Familia | Decision que la cerro |", "|---|---|"]
    for nombre, motivo in cerradas.items():
        lineas.append(f"| {nombre} | {motivo} |")
    lineas += ["",
               "Y una regla de la propia fase: **cualquier familia que solo cambie `C`, `seed`, "
               "`threshold` o `max_features` esta cerrada por V25**, y no se reabre.", "",
               "## Lo que dice la URL, en diagnostico", "",
               "| Medida | Valor |", "|---|---|",
               f"| Prevalencia de URL en train | **{url['prevalencia_url']:.2%}** "
               f"({url['filas_con_url']:,} filas) |",
               f"| Twitter status | {url['twitter_status']:,} |",
               f"| Solo URL | {url['solo_url']:,} |",
               f"| Texto + URL | {url['texto_mas_url']:,} |",
               f"| Multiples URL | {url['multiples_url']:,} |",
               f"| Acortadores | {url['acortadores']:,} |",
               f"| URL externa | {url['url_externa']:,} |",
               f"| URL con query string | {url['url_con_query']:,} |",
               f"| Longitud media del texto | {url['longitud_media_texto']:.1f} |",
               f"| Longitud media con URL | {url['longitud_media_texto_con_url']:.1f} |",
               f"| Longitud media sin URL | {url['longitud_media_texto_sin_url']:.1f} |",
               f"| Rango de enriquecimiento por clase | de "
               f"{min(url['rango_enriquecimiento_url_por_clase']):.3f} a "
               f"{max(url['rango_enriquecimiento_url_por_clase']):.3f} |", "",
               "**Esto es diagnostico y no causal.** Que la URL este distribuida por edad no dice si "
               "el modelo la esta leyendo. La unica forma de saberlo es el experimento de "
               "enmascararla, que es A4.", ""]
    GAP_MD.write_text("\n".join(lineas), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
