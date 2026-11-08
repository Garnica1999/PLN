"""Pruebas de clasificación de intención y guardarraíles del camino conversacional."""
from __future__ import annotations

from src.rag import chat
from src.rag.chat import apply_guardrail, classify
from src.rag.pipeline import ABSTENTION_TEXT


def test_classify_charla() -> None:
    assert classify("Hola") == "charla"
    assert classify("Buenos días, gracias") == "charla"
    assert classify("¿Quién eres?") == "charla"
    assert classify("Hola, ¿cómo estás?") == "charla"
    assert classify("¿Cómo estás?") == "charla"
    assert classify("Buenas tardes, ¿cómo te va?") == "charla"
    assert classify("Hola, ¿me ayudas?") == "charla"


def test_classify_dominio() -> None:
    assert classify("¿Cuántos municipios han reportado la presencia del Retamo Espinoso?") == "dominio"
    assert classify("Hola, ¿cuántos municipios han reportado la presencia del Retamo Espinoso?") == "dominio"
    assert classify("¿Cuál es la capital de Francia?") == "dominio"
    assert classify("Hola, ¿qué pasó con la moneda de 1.000 pesos?") == "dominio"
    assert classify("Hola, ¿cómo está el precio del dólar?") == "dominio"


def test_guardrail_dominio_sin_citas() -> None:
    answer, citations, invalid, guardrail, abstained = apply_guardrail("París", "dominio", [])
    assert answer == ABSTENTION_TEXT
    assert citations == [] and invalid == []
    assert guardrail == "sin_citas" and abstained


def test_guardrail_cita_valida() -> None:
    answer, citations, invalid, guardrail, abstained = apply_guardrail("22 municipios [1]", "dominio", [{"id": 1}])
    assert answer.startswith("22")
    assert citations == [1] and invalid == []
    assert guardrail is None and not abstained


def test_guardrail_cita_invalida() -> None:
    answer, citations, invalid, guardrail, abstained = apply_guardrail("22 municipios [9]", "dominio", [{"id": 1}])
    assert answer == ABSTENTION_TEXT
    assert citations == [] and invalid == [9]
    assert guardrail == "citas_invalidas" and abstained


def test_guardrail_charla_sin_citas() -> None:
    answer, citations, invalid, guardrail, abstained = apply_guardrail("Hola, ¿qué quieres consultar?", "charla", [])
    assert answer.startswith("Hola")
    assert citations == [] and invalid == []
    assert guardrail is None and not abstained


def test_guardrail_sin_respaldo() -> None:
    answer, _, _, guardrail, abstained = apply_guardrail(
        "La capital de Francia es París [1]", "dominio", [{"id": 1}], "Francia es un país de Europa."
    )
    assert answer == ABSTENTION_TEXT
    assert guardrail and guardrail.startswith("sin_respaldo")
    assert abstained


def test_guardrail_con_respaldo() -> None:
    answer, citations, _, guardrail, abstained = apply_guardrail(
        "22 municipios [1]", "dominio", [{"id": 1}], "22 municipios reportaron la presencia del Retamo."
    )
    assert answer.startswith("22")
    assert citations == [1] and guardrail is None and not abstained


def test_guardrail_cobertura_baja() -> None:
    _, _, _, guardrail, abstained = apply_guardrail(
        "La capital de Francia es París [1]", "dominio", [{"text": "París acoge una cumbre"}], "París acoge una cumbre"
    )
    assert abstained and guardrail and guardrail.startswith("sin_respaldo")


def test_guardrail_cobertura_alta() -> None:
    _, _, _, guardrail, abstained = apply_guardrail(
        "22 municipios reportaron el retamo [1]", "dominio",
        [{"text": "22 municipios reportaron el retamo espinoso"}], "22 municipios reportaron el retamo espinoso"
    )
    assert not abstained and guardrail is None


def test_chat_charla_devuelve_campos_de_la_ui(monkeypatch) -> None:
    from src.rag import chat as chat_module

    monkeypatch.setattr(chat_module, "_chat_answer", lambda *args, **kwargs: ("Hola.", 1.0))
    result = chat_module.chat("Hola")
    assert result["intent"] == "charla"
    assert result["prompt_version"]
    assert result["prompt_hash"]
    assert result["context_tokens"] == 0
    assert result["context_chunks"] == 0


def test_judge_parsea_si_con_acento(monkeypatch) -> None:
    monkeypatch.setattr(chat, "call_llm", lambda *args, **kwargs: {"response": "Sí."})
    assert chat.judge("Boyacá [1]", "¿En qué departamento?", "Corpoboyacá, Boyacá", "qwen2.5:1.5b")


def test_judge_rechaza_no_y_texto_raro(monkeypatch) -> None:
    monkeypatch.setattr(chat, "call_llm", lambda *args, **kwargs: {"response": "NO."})
    assert not chat.judge("París [1]", "¿Capital?", "Bogotá", "qwen2.5:1.5b")
    monkeypatch.setattr(chat, "call_llm", lambda *args, **kwargs: {"response": "silencio"})
    assert not chat.judge("París [1]", "¿Capital?", "Bogotá", "qwen2.5:1.5b")


def test_unsupported_terms_ignora_citas_y_separadores() -> None:
    from src.rag.chat import unsupported_terms

    assert unsupported_terms("La moneda de 1.000 pesos [2, 3, 4]", "monedas de 1000 pesos") == []
    assert unsupported_terms("27 municipios [1]", "22 municipios") == ["27"]


def test_guardrail_exige_entidad_en_el_chunk_citado() -> None:
    hits = [{"text": "El proyecto fue aprobado por el Congreso."}]
    answer, citations, _invalid, guardrail, abstained = apply_guardrail(
        "El proyecto Vantara fue aprobado [1].",
        "dominio",
        hits,
        context="El proyecto fue aprobado por el Congreso. Vantara estudia el traslado.",
    )
    assert answer == ABSTENTION_TEXT and citations == []
    assert guardrail.startswith("sin_respaldo:Vantara") and abstained


def test_guardrail_ignora_arranque_de_frase() -> None:
    hits = [{"text": "hay mas de 1000 arboles maduros en el bosque"}]
    _answer, citations, _invalid, guardrail, abstained = apply_guardrail(
        "Hay más de 1.000 árboles maduros en el bosque [1].", "dominio", hits
    )
    assert citations == [1] and guardrail is None and not abstained


def test_guardrail_ignora_conectores_de_gemma4() -> None:
    hits = [{"text": "22 municipios reportaron el Retamo Espinoso."}]
    _answer, citations, _invalid, guardrail, abstained = apply_guardrail(
        "Adicionalmente, 22 municipios reportaron el Retamo Espinoso [1].", "dominio", hits
    )
    assert citations == [1] and guardrail is None and not abstained


def test_guardrail_ignora_terminos_de_la_pregunta() -> None:
    hits = [{"text": "NHSF invitó a hablar en el lanzamiento."}]
    _answer, citations, _invalid, guardrail, abstained = apply_guardrail(
        "NHSF invitó a hablar en el lanzamiento del Marco Estratégico [1].",
        "dominio",
        hits,
        question="¿Qué organización invitó al lanzamiento del Marco Estratégico?",
    )
    assert citations == [1] and guardrail is None and not abstained


def test_guardrail_cross_language_no_exige_entidades() -> None:
    hits = [{"text": "13% of urban trees in the United Kingdom are in large areas.", "language": "en"}]
    _answer, citations, _invalid, guardrail, abstained = apply_guardrail(
        "El 13% de los árboles urbanos del Reino Unido están en áreas grandes [1].",
        "dominio",
        hits,
    )
    assert citations == [1] and guardrail is None and not abstained
