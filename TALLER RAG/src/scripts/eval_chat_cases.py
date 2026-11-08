"""Evalúa el camino conversacional con casos esperados y tool calling."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

API = "http://localhost:8010"
MODEL = os.getenv("EVAL_MODEL", "gemma4:e4b")
K = int(os.getenv("EVAL_K", "5"))
THINK = os.getenv("EVAL_THINK", "0") == "1"
STRICT = os.getenv("EVAL_STRICT", "0") == "1"
CASES = [
    ("Hola", "charla"),
    ("Buenos días", "charla"),
    ("Gracias", "charla"),
    ("¿Quién eres?", "charla"),
    ("Hola, ¿cómo estás?", "charla"),
    ("¿Cómo estás?", "charla"),
    ("¿Cuántos municipios han reportado la presencia del Retamo Espinoso?", "dominio"),
    ("¿Cuándo se celebra el Global Big Day?", "dominio"),
    ("¿Entre qué años tuvo su primera circulación la moneda de 1.000 pesos?", "dominio"),
    ("¿Dónde se encuentra la Marianne North Gallery?", "dominio"),
    ("¿Cuál es la capital de Francia?", "fuera"),
    ("¿Cuál es la población de Tokio?", "fuera"),
    ("¿Quién escribió Cien años de soledad?", "fuera"),
    ("¿Quién ganó el mundial de fútbol de 2022?", "fuera"),
]


def ask(query: str, model: str = MODEL, think: bool = THINK, k: int = K, strict: bool = STRICT) -> dict:
    started = time.perf_counter()
    response = requests.post(f"{API}/chat", json={"query": query, "model": model, "k": k, "think": think, "strict": strict}, timeout=300)
    response.raise_for_status()
    data = response.json()
    data["ms"] = round((time.perf_counter() - started) * 1000)
    return data


def main() -> None:
    rows = []
    for query, kind in CASES:
        data = ask(query)
        if kind == "charla":
            correct = data["intent"] == "charla" and not data["abstained"]
        elif kind == "dominio":
            correct = data["intent"] == "dominio" and not data["abstained"] and bool(data["citations"])
        else:
            correct = data["abstained"] is True
        rows.append(
            {
                "query": query,
                "tipo": kind,
                "intent": data["intent"],
                "guardrail": data.get("guardrail"),
                "abstained": data["abstained"],
                "citas": len(data["citations"]),
                "correcto": correct,
                "ms": data["ms"],
            }
        )
    frame = pd.DataFrame(rows)
    Path("reports/tablas").mkdir(parents=True, exist_ok=True)
    frame.to_csv("reports/tablas/chat_eval.csv", index=False)
    print(frame.to_string(index=False))
    print(f"\nAciertos: {frame['correcto'].sum()} de {len(frame)}")

    tool = ask("¿Cuántos municipios han reportado la presencia del Retamo Espinoso?", "llama3.2:3b", think=False, strict=False)
    print(f"Tool calling: {tool['tool_calls']}")
    Path("reports/tablas/chat_eval_tool.json").write_text(json.dumps(tool, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
