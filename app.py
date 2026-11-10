import streamlit as st

from agent import build_graph, create_model


def show_weather(weather: dict):
    location = ", ".join(filter(None, (weather["city"], weather["country"])))
    st.text(f"{location} · {weather['condition'] or 'Conditions unavailable'}")
    if weather["time"]:
        st.caption(f"Open-Meteo · {weather['time'].replace('T', ' ')} local time")
    columns = st.columns(3)
    for column, (label, key, unit) in zip(
        columns,
        [
            ("Temperature", "temperature", "°C"),
            ("Humidity", "humidity", "%"),
            ("Wind speed", "wind_speed", "km/h"),
        ],
    ):
        value = weather[key]
        column.metric(label, f"{value:g} {unit}" if value is not None else "Unavailable")


st.set_page_config(page_title="AI Weather Assistant", page_icon="🌤️")
st.title("AI Weather Assistant")
st.caption("Ask about current weather in a city. Weather data comes from Open-Meteo.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if st.sidebar.button("Clear conversation"):
    st.session_state.messages.clear()
    st.rerun()

messages = st.session_state.messages
for message in messages:
    with st.chat_message(message["role"]):
        st.text(message["text"])
        if message.get("weather"):
            show_weather(message["weather"])

question = st.chat_input("What's the weather in Tunis?")
if question and question.strip():
    recent_history = [(message["role"], message["text"]) for message in messages[-6:]]
    previous_city = next(
        (message["weather"]["city"] for message in reversed(messages) if message.get("weather")),
        "",
    )
    messages.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.text(question)
    with st.chat_message("assistant"):
        with st.spinner("Checking the weather…"):
            try:
                result = build_graph(create_model()).invoke(
                    {"question": question, "history": recent_history, "previous_city": previous_city}
                )
            except (ValueError, RuntimeError) as error:
                result = {"answer": str(error)}
        answer = result["answer"]
        weather = result.get("weather")
        st.text(answer)
        if weather:
            show_weather(weather)
        messages.append({"role": "assistant", "text": answer, "weather": weather})
