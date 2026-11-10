import logging
import os
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph import END, START, StateGraph
from requests.exceptions import Timeout

from weather import fetch_weather

MODEL_NAME = "mistralai/mistral-large-4-1t-a52b"
logger = logging.getLogger(__name__)

CITY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Identify the city in the latest weather question. Return only its name, "
            "with no country, quotes, labels, or punctuation around it. For a weather "
            "follow-up with no new city, use the last confirmed city if available. "
            "Return NONE for a missing city or an unrelated question.",
        ),
        ("human", "Last confirmed city: {previous_city}\nQuestion: {question}"),
    ]
)
ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer in the user's language in at most two sentences. Use only the supplied "
            "current weather data. Use qualitative descriptions and leave all numeric values "
            "to the app's separate weather panel. Treat None as unavailable. If asked about "
            "a future or past time, say you only have current conditions. For clothing "
            "questions, give simple advice based on the current data.",
        ),
        MessagesPlaceholder("history"),
        ("human", "Question: {question}\nCurrent Open-Meteo data: {weather}"),
    ]
)


class WeatherState(TypedDict, total=False):
    question: str
    previous_city: str
    history: list[tuple[str, str]]
    city: str
    weather: dict
    answer: str
    error: str


def create_model():
    load_dotenv(Path(__file__).with_name(".env"))
    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Set NVIDIA_API_KEY in your .env file, then try again.")
    try:
        return ChatNVIDIA(
            model=MODEL_NAME,
            api_key=api_key,
            temperature=0,
            max_completion_tokens=2048,
            timeout=60,
        )
    except (Timeout, TimeoutError):
        raise RuntimeError("NVIDIA timed out during setup. Please try again.") from None
    except Exception as error:
        logger.warning("NVIDIA setup failed (%s)", type(error).__name__)
        raise RuntimeError("Could not connect to NVIDIA. Check your API key and connection.") from None


def build_graph(model):
    """Build the same three-node workflow for production and tests."""

    def extract_city(state: WeatherState) -> dict:
        try:
            response = (CITY_PROMPT | model).invoke(
                {"question": state["question"], "previous_city": state.get("previous_city") or "NONE"}
            )
        except (Timeout, TimeoutError):
            return {"error": "The NVIDIA request timed out while identifying the city."}
        except Exception as error:
            logger.warning("NVIDIA city extraction failed (%s)", type(error).__name__)
            return {"error": "NVIDIA could not identify the city. Check your API key and connection."}

        city = response.text.strip().strip("\"'`.")
        if not city or response.response_metadata.get("finish_reason") == "length":
            return {"error": "NVIDIA did not return a city. Try asking again with the city name."}
        if city.upper() == "NONE":
            return {"error": "Which city would you like the current weather for?"}
        return {"city": city}

    def get_weather(state: WeatherState) -> dict:
        if state.get("error"):
            return {}
        try:
            return {"weather": fetch_weather(state["city"])}
        except (ValueError, RuntimeError) as error:
            return {"error": str(error)}

    def generate_answer(state: WeatherState) -> dict:
        if state.get("error"):
            return {"answer": state["error"]}

        weather = state["weather"]
        fallback = f"Current conditions for {weather['city']} are shown below."
        try:
            response = (ANSWER_PROMPT | model).invoke(
                {
                    "question": state["question"],
                    "history": state.get("history", [])[-6:],
                    "weather": weather,
                }
            )
        except (Timeout, TimeoutError):
            return {"answer": f"NVIDIA took too long to explain the weather. {fallback}"}
        except Exception as error:
            logger.warning("NVIDIA weather explanation failed (%s)", type(error).__name__)
            return {"answer": f"NVIDIA could not explain the weather. {fallback}"}

        answer = response.text.strip()
        if not answer or response.response_metadata.get("finish_reason") == "length":
            return {"answer": f"NVIDIA did not provide a complete explanation. {fallback}"}
        if any(character.isdecimal() for character in answer):
            return {"answer": fallback}
        return {"answer": answer}

    graph = StateGraph(WeatherState)
    graph.add_node("extract_city", extract_city)
    graph.add_node("get_weather", get_weather)
    graph.add_node("generate_answer", generate_answer)
    graph.add_edge(START, "extract_city")
    graph.add_edge("extract_city", "get_weather")
    graph.add_edge("get_weather", "generate_answer")
    graph.add_edge("generate_answer", END)
    return graph.compile()
