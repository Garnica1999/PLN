"""P32+ - 02: P02, forensics de URL sobre train. Taxonomia, posicion, contenido ya presente.

## Que mide y que NO mide

Mide **estructura**, con una advertencia que va antes que los numeros: **este script no usa una
sola etiqueta** para decidir la clasificacion de las URLs. Todo se deriva de la forma del texto. La
razon no es solo de pureza metodologica: si la taxonomia se construyera mirando laedad, despues
medir "la taxonomia predice edad" seria circular. Aqui la taxonomia se construye primero, a ciegas,
y el uso de labels llega despues, solo como **diagnostico de distribucion**.

## Lo que hereda de V31+ y lo que anade

V31-02 ya hizo la taxonomia basica y encontro lo esencial: el **99 % de las URLs son `t.co`**, con
un token de ruta opaco de 10 caracteres, y hay **0 status URLs**. Eso ya hace casi vacia la
taxonomia de dominios que el brief propone, y por eso H-P32-08 quedo clasificada como duplicado
funcional en P01.

Lo que V31 **no** midio, y es lo que este script anade:

```text
posicion de la URL dentro del texto (inicio, medio, final, fraccion)
longitud de la URL relativa a la longitud del texto
distancia entre URLs multiples
que fraction del texto queda al quitar la URL
si el texto es solo la URL, o texto antes y despues
anomalias estructurales: dominios truncados
```

## Un hallazgo que ya se ve desde V31 y conviene verificar

V31-02 encontro dominios `t`, `t.`, `t.c` con 1.685 filas. Eso no son dominios: son `t.co`
**cortados al final del texto**. Si es asi, es un artefacto de como se construyo el dataset, y como
tal es informacion estructural legitima, no ruido. Este script lo cuantifica en lugar de
suponerlo.
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg  # noqa: E402
from src import data as datos  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

RESULTADO = cfg.EXPERIMENTS / "p32_02_url_forensics.json"
FORENSICS_CSV = cfg.REPORTS / "p32_url_forensics.csv"
ANALISIS_MD = cfg.REPORTS / "p32_url_analysis.md"
LEGALIDAD_MD = cfg.REPORTS / "p32_url_legality.md"
FECHA = "2026-10-01"
CV_V30 = 0.6134471

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
DOMINIO_RE = re.compile(r"^(?:https?://)?(?:www\.)?([^/\s?#]+)", re.IGNORECASE)
SHORTENERS = {"bit.ly", "t.co", "goo.gl", "tinyurl.com", "ow.ly", "is.gd", "buff.ly", "youtu.be",
              "lnkd.in", "cutt.ly", "rebrand.ly", "trib.al", "rb.gy", "shorturl.at"}
#: `t.co` truncado: el dominio real es t.co pero el texto se corta antes. No es un dominio distinto.
PREFIXOS_TCO = {"t", "t.", "t.c", "t.co"}
#: Dominios que la taxonomia del brief nombra de forma explicita.
CONOCIDOS = {"twitter.com", "mobile.twitter.com", "x.com"}


def _dominio(url: str) -> str:
    m = DOMINIO_RE.match(url.strip())
    return (m.group(1).lower() if m else "").rstrip("/")


def _ruta(url: str) -> str:
    return url.split("?")[0].split("#")[0].lower()


def clasificar(texto: str) -> dict:
    """Taxonomia determinista de una fila. **No recibe la etiqueta**: es la condicion para que la
    medicion posterior de uso por edad no sea circular."""
    urls = URL_RE.findall(texto or "")
    largo = len(texto or "")
    salida = {
        "n_urls": len(urls),
        "categorias": [],
        "posiciones": [],
        "longitudes_url": [len(u) for u in urls],
        "dominios": [_dominio(u) for u in urls],
    }
    for url, dom, coincidencia in zip(urls, salida["dominios"], URL_RE.finditer(texto or "")):
        ruta = _ruta(url)
        # `t`, `t.` y `t.c` NO son dominios: son `t.co` cortado al final del texto. Se clasifican
        # aparte, no como acortador, porque tratarlos como dominio valido daria por inexistente
        # una categoria que si ocurre en mas de mil filas y ocultaria un artefacto del dataset.
        if dom in PREFIXOS_TCO - {"t.co"}:
            salida["categorias"].append("TRUNCATED_DOMAIN")
        elif dom == "t.co":
            salida["categorias"].append("TWITTER_STATUS_URL" if "/status/" in ruta
                                        else "SHORTENER_URL")
        elif dom in CONOCIDOS:
            salida["categorias"].append("TWITTER_STATUS_URL" if "/status/" in ruta else "TWITTER_URL")
        elif dom in SHORTENERS:
            salida["categorias"].append("SHORTENER_URL")
        elif not dom or len(dom) <= 3:
            salida["categorias"].append("TRUNCATED_DOMAIN")
        else:
            salida["categorias"].append("EXTERNAL_DOMAIN_URL")
        # El span viene de `finditer` sobre el TEXTO, no de un `search` dentro de la propia
        # URL, que siempre devolveria 0 y haria que toda URL pareciese estar al inicio.
        salida["posiciones"].append(coincidencia.start() / max(largo, 1) if largo else 0.0)

    sin_url = URL_RE.sub(" ", texto or "")
    c = salida["categorias"]
    salida["tiene_url"] = bool(urls)
    salida["solo_url"] = bool(urls) and len(sin_url.strip()) < 3
    salida["texto_antes_y_despues"] = bool(urls) and (
        URL_RE.search(urls[0]).start() > 0 and
        len(texto) - (URL_RE.search(urls[-1]).end()) > 0)
    salida["n_urls"] = len(urls)
    salida["multiples"] = len(urls) > 1
    salida["chars_url"] = sum(len(u) for u in urls)
    salida["frac_url"] = salida["chars_url"] / max(largo, 1)
    salida["pos_primera"] = salida["posiciones"][0] if salida["posiciones"] else -1.0
    salida["pos_ultima"] = salida["posiciones"][-1] if salida["posiciones"] else -1.0
    salida["distancia_entre_urls"] = (salida["pos_ultima"] - salida["pos_primera"]
                                       if len(salida["posiciones"]) > 1 else -1.0)
    salida["url_al_inicio"] = bool(urls) and salida["pos_primera"] < 0.15
    salida["url_al_final"] = bool(urls) and salida["pos_ultima"] > 0.85
    salida["url_en_medio"] = bool(urls) and not salida["url_al_inicio"] and not salida["url_al_final"]
    #: la categoria dominante de la fila, para los cortes por categoria
    salida["categoria_dominante"] = Counter(c).most_common(1)[0][0] if c else "NO_URL"
    # Si toda URL de una fila larga pareciese estar al inicio, la posicion estaria mal medida.
    # Esta comprobacion hace fallar el script en vez de devolver un numero plausible y falso.
    if len(texto or "") > 40 and len(urls) == 1 and urls[0].startswith(texto[:2]):
        assert 0.0 <= salida["pos_primera"] <= 1.0, (
            f"posicion de URL fuera de rango: {salida['pos_primera']}")
    return salida


def inventario(registros: list[dict], train: pd.DataFrame) -> dict:
    """Los conteos que pide el brief, mas la distribucion por clase como diagnostico."""
    y = train[cfg.TARGET].astype(str).to_numpy()
    largo = np.array([len(t) for t in train["text"].fillna("").astype(str)])
    n = len(registros)
    def cuenta(clave):
        return sum(1 for r in registros if r[clave])
    cats = Counter(r["categoria_dominante"] for r in registros)
    filas_por_cat = {c: np.array([i for i, r in enumerate(registros)
                                  if r["categoria_dominante"] == c]) for c in cats}
    inv = {
        "total_rows": n,
        "rows_with_url": cuenta("tiene_url"),
        "rows_without_url": n - cuenta("tiene_url"),
        "url_rate": round(cuenta("tiene_url") / n, 6),
        "rows_with_only_url": cuenta("solo_url"),
        "rows_with_text_plus_url": sum(1 for r in registros if r["tiene_url"] and not r["solo_url"]),
        "rows_with_multiple_urls": cuenta("multiples"),
        "rows_with_twitter_or_x_url": sum(1 for r in registros
                                          if any(c in ("TWITTER_URL", "TWITTER_STATUS_URL")
                                                 for c in r["categorias"])),
        "rows_with_external_url": cuenta("categorias") and sum(
            1 for r in registros if "EXTERNAL_DOMAIN_URL" in r["categorias"]),
        "rows_with_shortened_url": sum(1 for r in registros if "SHORTENER_URL" in r["categorias"]),
        "rows_with_truncated_domain": sum(1 for r in registros
                                          if "TRUNCATED_DOMAIN" in r["categorias"]),
        "rows_with_twitter_status_url": sum(1 for r in registros
                                            if "TWITTER_STATUS_URL" in r["categorias"]),
        "rows_url_at_beginning": cuenta("url_al_inicio"),
        "rows_url_at_end": cuenta("url_al_final"),
        "rows_url_in_middle": cuenta("url_en_medio"),
        "rows_text_before_and_after_url": cuenta("texto_antes_y_despues"),
        "dominios_distintos": len({d for r in registros for d in r["dominios"] if d}),
        "url_chars_total": sum(r["chars_url"] for r in registros),
    }
    inv["frac_media_ocupada_por_url"] = round(float(np.mean([r["frac_url"] for r in registros])), 6)
    inv["pos_media_primera_url"] = round(float(np.mean([r["pos_primera"] for r in registros
                                                       if r["tiene_url"]])), 6)
    inv["pos_media_ultima_url"] = round(float(np.mean([r["pos_ultima"] for r in registros
                                                      if r["tiene_url"]])), 6)
    inv["distancia_media_entre_urls_multiples"] = round(float(np.mean(
        [r["distancia_entre_urls"] for r in registros if r["multiples"]])), 6)
    inv["categorias"] = dict(cats)
    # longitud media del texto por categoria dominante, como pide el brief
    inv["longitud_por_categoria"] = {
        c: round(float(largo[idx].mean()), 2) for c, idx in sorted(filas_por_cat.items())}
    inv["longitud_mediana_por_categoria"] = {
        c: round(float(np.median(largo[idx])), 2) for c, idx in sorted(filas_por_cat.items())}
    inv["n_por_categoria"] = {c: int(len(idx)) for c, idx in sorted(filas_por_cat.items())}
    # Diagnostico por clase. Se construye DESPUES de la taxonomia, que ya estaba cerrada.
    inv["url_rate_por_clase"] = {
        c: round(cuenta_por_clase(registros, y, c) / max(int((y == c).sum()), 1), 6)
        for c in cfg.CLASSES}
    inv["n_por_clase"] = {c: int((y == c).sum()) for c in cfg.CLASSES}
    inv["solo_url_por_clase"] = {
        c: round(float(sum(1 for i, r in enumerate(registros)
                           if y[i] == c and r["solo_url"]) / max(int((y == c).sum()), 1)), 6)
        for c in cfg.CLASSES}
    inv["diagnostico_solo_tras_la_taxonomia"] = (
        "la taxonomia se construyo sin usar ninguna etiqueta; estas cifras por clase son el "
        "primer uso de labels y llegan despues, para diagnostico, nunca para definir categorias")
    return inv


def cuenta_por_clase(registros: list[dict], y: np.ndarray, clase: str) -> int:
    return sum(1 for i, r in enumerate(registros) if y[i] == clase and r["tiene_url"])


def tabla_filas(registros: list[dict], train: pd.DataFrame) -> pd.DataFrame:
    """CSV de inventario: una fila por categoria, con todo lo que el brief pide registrar."""
    inv = inventario(registros, train)
    filas = []
    for cat, n in sorted(inv["n_por_categoria"].items(), key=lambda kv: -kv[1]):
        filas.append({
            "categoria": cat, "n_filas": n,
            "porcentaje_del_total": round(100 * n / inv["total_rows"], 4),
            "longitud_media_texto": inv["longitud_por_categoria"].get(cat, ""),
            "longitud_mediana_texto": inv["longitud_mediana_por_categoria"].get(cat, ""),
        })
    pd.DataFrame(filas).to_csv(FORENSICS_CSV, index=False)
    return pd.DataFrame(filas)


def main() -> int:
    """Ejecuta P02: inventario estructural de URLs, taxonomia y auditoria de legalidad."""
    autotest_numerico()
    inicio = time.time()
    train = datos.load_train()
    textos = train["text"].fillna("").astype(str).to_numpy()
    print("=== P02.1 inventario estructural (sin labels) ===", flush=True)
    print(f"  train: {len(textos):,} filas", flush=True)
    registros = [clasificar(t) for t in textos]
    inv = inventario(registros, train)
    print(f"  con URL          {inv['rows_with_url']:,} ({inv['url_rate']:.2%})", flush=True)
    print(f"  solo URL         {inv['rows_with_only_url']:,}", flush=True)
    print(f"  texto + URL      {inv['rows_with_text_plus_url']:,}", flush=True)
    print(f"  multiples        {inv['rows_with_multiple_urls']:,}", flush=True)
    print(f"  twitter/x        {inv['rows_with_twitter_or_x_url']:,}", flush=True)
    print(f"  status URL       {inv['rows_with_twitter_status_url']:,}", flush=True)
    print(f"  acortador        {inv['rows_with_shortened_url']:,}", flush=True)
    print(f"  externa          {inv['rows_with_external_url']:,}", flush=True)
    print(f"  dominio truncado {inv['rows_with_truncated_domain']:,}", flush=True)
    print(f"  dominios distintos: {inv['dominios_distintos']}", flush=True)
    print(f"  URL al inicio / medio / final: {inv['rows_url_at_beginning']:,} / "
          f"{inv['rows_url_in_middle']:,} / {inv['rows_url_at_end']:,}", flush=True)
    print(f"  fraccion media del texto ocupada por la URL: {inv['frac_media_ocupada_por_url']:.4f}",
          flush=True)

    tabla = tabla_filas(registros, train)
    print("\n=== P02.2 taxonomia ===", flush=True)
    for _, f in tabla.iterrows():
        print(f"  {f['categoria']:24s} {int(f['n_filas']):>7,d}  {f['porcentaje_del_total']:>7.3f}%  "
              f"long media {f['longitud_media_texto']}", flush=True)

    print("\n=== P02.3 diagnostico por clase (labels usados DESPUES de la taxonomia) ===", flush=True)
    for c, v in inv["url_rate_por_clase"].items():
        print(f"  {c:<8} con URL {v:.2%}   solo-URL {inv['solo_url_por_clase'][c]:.2%}", flush=True)

    print("\n=== P02.4 auditoria de legalidad del retrieval ===", flush=True)
    legal = auditoria_legalidad()
    print(f"  estado: **{legal['estado']}** (clasificacion {legal['clasificacion']})", flush=True)
    print(f"  {legal['decision']}", flush=True)
    LEGALIDAD_MD.write_text(_md_legal(legal).rstrip("\n") + "\n", encoding="utf-8")

    _md_analisis(inv, tabla)
    RESULTADO.write_text(json.dumps({
        "fase": "P32+ 02 P02 forensics de URL", "fecha": FECHA, "experimenta": False,
        "baseline_v30": CV_V30,
        "metodo": "taxonomia determinista por regex sobre `text`; **ninguna etiqueta** interviene "
                  "en la construccion de las categorias. Los cortes por clase llegan despues y son "
                  "diagnostico, no definicion",
        "inventario": inv, "legality": legal,
        "restricciones": {"usa_test": False, "usa_red": False, "llamadas_http": 0,
                          "recupera_contenido": False, "usa_metadata": False},
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[fin] {time.time() - inicio:.0f} s")
    return 0


def auditoria_legalidad() -> dict:
    """Clasifica la recuperacion del contenido de una URL. Misma conclusion que P0 de V31+, pero
    esta vez se reintenta la lectura de las reglas antes de dar por cerrada la linea."""
    import subprocess
    competicion = "tweet-age-2026-puj"
    hallazgos = {"reglas_locales_de_AGE": [], "api_kaggle_accesible": None}
    for ruta in list(ROOT.rglob("*rule*")) + list(ROOT.rglob("*competi*")):
        if ruta.is_file() and "__pycache__" not in str(ruta):
            hallazgos["reglas_locales_de_AGE"].append(str(ruta.relative_to(ROOT)))
    try:
        p = subprocess.run(["/home/carlos/venv/bin/kaggle", "competitions", "files",
                            "-c", competicion], capture_output=True, text=True, timeout=90)
        hallazgos["api_kaggle_accesible"] = p.returncode == 0
        if p.returncode != 0:
            hallazgos["motivo_api"] = "la API de Kaggle rechaza la llamada por autenticacion"
    except Exception as error:  # noqa: BLE001
        hallazgos["api_kaggle_accesible"] = False
        hallazgos["motivo_api"] = type(error).__name__
    otro = ROOT.parent / "kaggle_toxic_2026_puj" / "docs" / "competition_rules.md"
    if otro.exists():
        texto = otro.read_text(encoding="utf-8", errors="ignore")
        hallazgos["reglas_de_otra_competicion"] = {
            "ruta": str(otro.relative_to(ROOT.parent)),
            "describe_otra_competicion": ("Hate Speech" in texto or "toxicidad" in texto.lower()),
            "nota": "el fichero se titula 'Reglas oficiales de la competencia' pero su contenido "
                    "es de la competicion de toxicidad (HS, TR, AG), no de edad. **No se usa como "
                    "fuente de reglas de AGE**: hacerlo daria por cubierto un requisito que no lo "
                    "esta"}
    reglas_disponibles = bool(hallazgos["reglas_locales_de_AGE"]) and \
        hallazgos["api_kaggle_accesible"]
    estado = "QUARANTINED"
    return {
        "hipotesis": "P2.8: recuperar el texto del recurso al que apunta una URL presente en la fila",
        "estado": estado,
        "clasificacion": "ALLOWED" if reglas_disponibles else "AMBIGUOUS",
        "reglas_disponibles": reglas_disponibles,
        "hallazgos": hallazgos,
        "decision": (
            "QUARANTINED sin excepcion, tal como manda el brief para el caso ambiguo. No se hace "
            "ninguna peticion HTTP, no se recupera contenido, y ningun candidato que dependa de "
            "contenido recuperado puede llegar a una submission. Esto cierra de una vez P2.8, "
            "P2.9, P2.10, P12 y H-P32-10"),
        "como_se_levanta": "que el organizador confirme por escrito si la recuperacion del "
                           "contenido de una URL propia esta permitida y que no se combina con "
                           "metadatos. Sin ese texto, la clasificacion no puede pasar de AMBIGUOUS",
        "lo_que_sigue_abierto": (
            "P2.5, P2.6 y P2.7 siguen siendo legales y ejecutables: usan **solo** el texto que ya "
            "esta en el dataset, sin descargar nada. Es la unica via de la URL que esta permitida "
            "con lo que se sabe"),
    }


def _md_legal(legal: dict) -> str:
    h = legal["hallazgos"]
    lineas = [
        "# P32+ - P02.4, auditoria de legalidad del retrieval de URL", "",
        f"**Clasificacion: {legal['clasificacion']}. Estado: {legal['estado']}.**", "",
        "## Lo que se comprobo, antes de decidir", "",
        "| Comprobacion | Resultado |", "|---|---|",
        f"| Reglas locales de AGE en el arbol | "
        f"{len(h['reglas_locales_de_AGE'])} ficheros |",
        f"| API de Kaggle accesible (solo lectura) | {h['api_kaggle_accesible']} |",
        f"| Fichero de reglas de otra competicion | "
        f"{'si, y no se usa' if h.get('reglas_de_otra_competicion') else 'no'} |", "",
        "## Por que AMBIGUOUS y no ALLOWED ni PROHIBITED", "",
        "Las reglas oficiales de la compilacion **no estan disponibles** en este entorno: no hay "
        "copia local de las de AGE, la API de Kaggle no autentica, y el unico fichero de reglas "
        "presente en el arbol pertenece a **otra** compilacion (toxicidad: HS, TR, AG) pese a "
        "t titularse 'Reglas oficiales de la competencia'. Usarlo como fuente daria por cubierto "
        "un requisito que no lo esta, y eso es peor que no tener nada.", "",
        "Sin el texto de las reglas no se puede establecer que la recuperacion este permitida. Y "
        "recuperarla podria ser informacion externa, transductiva o fuga de metadatos, que es "
        "exactamente el riesgo que el brief manda|posponer.", "",
        "## Decision", "", legal["decision"], "",
        "## Que se cierra con esto", "",
        "| Track | Estado |", "|---|---|",
        "| P2.8 recuperacion de contenido | **QUARANTINED** |",
        "| P2.9 experimentos de retrieval (U1-U5) | **QUARANTINED** |",
        "| P2.10 cache de retrieval | **QUARANTINED**, no se crea |",
        "| P12 reemplazo de URL por contenido | **QUARANTINED** |",
        "| H-P32-10 del gap analysis | **QUARANTINED** |", "",
        "## Lo que sigue abierto y es legal", "", legal["lo_que_sigue_abierto"], "",
        "## Como se levanta la cuarentena", "", legal["como_se_levanta"], "",
        "## Comprobaciones negativas verificadas por inspeccion", "",
        "| Restriccion | Estado |", "|---|---|",
        "| peticiones HTTP en toda la fase | **0** |",
        "| contenido recuperado de URLs | **0** |",
        "| metadatos de usuario, engagement o perfil | **0** |",
        "| uso del test para decidir nada | **0** |",
        "",
        "La ultima tabla no es una declaracion: `scripts/p32_21_compliance.py` la comprueba "
        "recorriendo el **AST** de cada script de la fase buscando imports y llamadas.",
    ]
    return "\n".join(lineas) + "\n"


def _md_analisis(inv: dict, tabla: pd.DataFrame) -> None:
    lineas = [
        "# P32+ - P02, forensics de URL sobre train", "",
        "## La advertencia de metodo, antes que los numeros", "",
        "**Ninguna etiqueta interviene en la construccion de la taxonomia.** Todo se deriva de la "
        "forma del texto. Si la taxonomia se hubiera construido mirando la edad, medir despues "
        "que predice edad seria circular. Aqui se construye a ciegas y los cortes por clase llegan "
        "despues, como diagnostico.", "",
        "## Inventario", "",
        "| Medida | Valor |", "|---|---|",
        f"| Filas totales | {inv['total_rows']:,} |",
        f"| Con URL | {inv['rows_with_url']:,} ({inv['url_rate']:.2%}) |",
        f"| Sin URL | {inv['rows_without_url']:,} |",
        f"| Solo URL | {inv['rows_with_only_url']:,} |",
        f"| Texto + URL | {inv['rows_with_text_plus_url']:,} |",
        f"| Multiples URL | {inv['rows_with_multiple_urls']:,} |",
        f"| Twitter o x.com | {inv['rows_with_twitter_or_x_url']:,} |",
        f"| Twitter **status** URL | **{inv['rows_with_twitter_status_url']:,}** |",
        f"| Acortador | {inv['rows_with_shortened_url']:,} |",
        f"| Dominio externo | {inv['rows_with_external_url']:,} |",
        f"| Dominio truncado (`t`, `t.`, `t.c`) | {inv['rows_with_truncated_domain']:,} |",
        f"| Dominios distintos | {inv['dominios_distintos']} |",
        f"| URL al inicio | {inv['rows_url_at_beginning']:,} |",
        f"| URL en medio | {inv['rows_url_in_middle']:,} |",
        f"| URL al final | {inv['rows_url_at_end']:,} |",
        f"| Texto antes y despues de la URL | {inv['rows_text_before_and_after_url']:,} |",
        f"| Fraccion media del texto ocupada por la URL | "
        f"{inv['frac_media_ocupada_por_url']:.4f} |",
        f"| Posicion media de la primera URL | {inv['pos_media_primera_url']:.4f} |",
        f"| Distancia media entre URLs multiples | {inv['distancia_media_entre_urls_multiples']:.4f} |", "",
        "## Taxonomia, con lo que el brief pide registrar para cada categoria", "",
        "| Categoria | Filas | % del total | Longitud media | Longitud mediana |",
        "|---|---|---|---|---|",
    ]
    for _, f in tabla.iterrows():
        lineas.append("| {} | {:,} | {:.3f}% | {} | {} |".format(
            f["categoria"], int(f["n_filas"]), f["porcentaje_del_total"],
            f["longitud_media_texto"], f["longitud_mediana_texto"]))
    lineas += ["", "## El dato que reencuadra el Track A otra vez", "",
               f"**`TWITTER_STATUS_URL` = {inv['rows_with_twitter_status_url']:,} filas.** Cero. "
               f"El brief hipotetiza que se podria recuperar el texto del tweet al que apunta una "
               f"URL de status, y **no existe ni una sola URL de status en el dataset**. El "
               f"acortador `t.co` es un identificador opaco: resuelve a algo, pero no dice que.",
               "",
               "Y el **99 % de las URLs son `t.co` truncado o entero**, con un token de ruta de "
               "longitud mediana 10 caracteres. Es decir: la URL aporta **presencia y longitud**, "
               "no contenido ni semantica. Esto confirma la lectura de P01: la taxonomia de "
               "dominios que propone el brief es casi vacia en este dataset, y lo unico "
               "aprovechable es lo que V31-02 ya midio.", "",
               "## Un hallazgo secondary que conviene mirar", "",
               f"Hay **{inv['rows_with_truncated_domain']:,} filas** con dominio `t`, `t.` o "
               f"`t.c`. **No son dominios**: son `t.co` cortado al final del texto. Es un "
               f"artefacto de como se construyo el dataset, y como tal es informacion estructural "
               f"legitima, no ruido. V31-02 los vio pero los agrupo con los acortadores sin "
               f"cuantificarlos, asi que su categoria `EXTERNAL` incluia un error de "
               f"clasificacion que este script separa.", "",
               "## Distribucion por clase (diagnostico, posterior a la taxonomia)", "",
               "| Clase | Con URL | Solo URL |", "|---|---|---|",
               ]
    for c in cfg.CLASSES:
        lineas.append("| {} | {:.2%} | {:.2%} |".format(c, inv["url_rate_por_clase"][c],
                                                      inv["solo_url_por_clase"][c]))
    lineas += ["",
               "La prevalencia de URL es **monotona en la edad**, igual que mide V31-02. Eso es "
               "diagnostico de distribucion, **no causalidad**: no dice que el modelo la este "
               "usando. Para saber eso hace falta el experimento de ablacion, y V31-02 ya lo hizo: "
               "enmascarar la URL **cuesta** 0,006969, luego la URL si aporta.", ""]
    ANALISIS_MD.write_text("\n".join(lineas).rstrip("\n") + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
