from pathlib import Path
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from requests.exceptions import Timeout
from streamlit.testing.v1 import AppTest

import agent

WEATHER = {
    "city": "Tunis",
    "country": "Tunisia",
    "time": "2026-11-10T22:00",
    "temperature": 18.5,
    "humidity": 65,
    "wind_speed": None,
    "condition": "Light rain",
}


def test_graph_runs_three_nodes_and_uses_context(monkeypatch):
    events = []

    def fake_model(prompt):
        messages = prompt.to_messages()
        if "Last confirmed city:" in messages[-1].content:
            events.append("extract_city")
            assert "Last confirmed city: Tunis" in messages[-1].content
            return AIMessage(content="Tunis")
        events.append("generate_answer")
        assert messages[1].content == "Weather in Tunis?"
        assert "'condition': 'Light rain'" in messages[-1].content
        return AIMessage(content="Yes, there is light rain right now.")

    def fake_weather(city):
        events.append("get_weather")
        assert city == "Tunis"
        return WEATHER

    monkeypatch.setattr(agent, "fetch_weather", fake_weather)
    result = agent.build_graph(RunnableLambda(fake_model)).invoke(
        {"question": "Is it raining?", "previous_city": "Tunis", "history": [("user", "Weather in Tunis?")]}
    )
    assert events == ["extract_city", "get_weather", "generate_answer"]
    assert result["weather"] == WEATHER
    assert result["answer"] == "Yes, there is light rain right now."


@pytest.mark.parametrize(("content", "message"), [("NONE", "Which city"), ("", "did not return a city")])
def test_missing_city_does_not_call_weather(monkeypatch, content, message):
    fetch = Mock()
    monkeypatch.setattr(agent, "fetch_weather", fetch)
    model = RunnableLambda(lambda prompt: AIMessage(content=content))
    result = agent.build_graph(model).invoke({"question": "What's the weather?"})
    assert message in result["answer"]
    fetch.assert_not_called()


def test_unknown_city_skips_nvidia_explanation(monkeypatch):
    model = Mock(return_value=AIMessage(content="Unknown City"))
    monkeypatch.setattr(agent, "fetch_weather", Mock(side_effect=ValueError("City not found.")))
    result = agent.build_graph(RunnableLambda(model)).invoke({"question": "Weather in Unknown City?"})
    assert result["answer"] == "City not found."
    assert model.call_count == 1


@pytest.mark.parametrize("failure", [AIMessage(content=""), AIMessage(content="It is 99°C."), Timeout()])
def test_unusable_explanation_still_returns_weather(monkeypatch, failure):
    monkeypatch.setattr(agent, "fetch_weather", lambda city: WEATHER)
    model = Mock(side_effect=[AIMessage(content="Tunis"), failure])
    result = agent.build_graph(RunnableLambda(model)).invoke({"question": "Weather in Tunis?"})
    assert "Current conditions for Tunis" in result["answer"]
    assert result["weather"] == WEATHER


def test_streamlit_chat_metrics_and_clear(monkeypatch):
    def fake_model(prompt):
        if "Last confirmed city:" in prompt.to_messages()[-1].content:
            return AIMessage(content="Tunis")
        return AIMessage(content="There is light rain.")

    monkeypatch.setattr(agent, "create_model", lambda: RunnableLambda(fake_model))
    monkeypatch.setattr(agent, "fetch_weather", lambda city: WEATHER)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
    app.chat_input[0].set_value("Weather in Tunis?").run()

    assert not app.exception
    assert [message.name for message in app.chat_message] == ["user", "assistant"]
    assert [metric.value for metric in app.metric] == ["18.5 °C", "65 %", "Unavailable"]

    app.sidebar.button[0].click().run()
    assert not app.exception
    assert not app.chat_message
