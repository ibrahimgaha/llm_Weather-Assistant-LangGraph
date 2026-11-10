# Project 5 — AI Weather Assistant with LangGraph and NVIDIA AI

Ask about current weather in a city. Open-Meteo supplies the data and
NVIDIA Mistral Large 4 explains what the conditions mean.

## Run on Windows

With Python 3.14 installed, open PowerShell and run:

```powershell
cd "C:\Users\Aziz\Desktop\Agentic IA\05-Weather-Assistant-LangGraph"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

Set `NVIDIA_API_KEY` in `.env` and save. If `.env` already exists, skip the
copy command and edit the existing file. The Open-Meteo endpoints used here
require no separate key.

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Try “What's the weather in Tunis?” followed by “Is it raining?”.
The follow-up uses the last resolved city. The weather panel shows the
matched location, local data time, temperature, humidity, wind, and conditions.
**Clear conversation** resets the session's messages and city context.

The assistant uses current conditions for one city at a time. When multiple
places share a name, it uses the top geocoding match and displays the country
when available. Its weather panel marks missing measurements as unavailable.

## Workflow

`START → extract_city → get_weather → generate_answer → END`

1. `extract_city` asks NVIDIA to identify the city from the question and the last confirmed city.
2. `get_weather` finds the coordinates and retrieves the current conditions from Open-Meteo.
3. `generate_answer` gives NVIDIA the current data and recent chat messages so it can explain the conditions.

LangGraph passes the state dictionary between the nodes. If city extraction
or lookup fails, the graph returns a readable error. If the explanation fails,
the app still shows the Open-Meteo data. The model writes a qualitative answer
and the weather panel displays the actual measurements.

## Files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit chat, session history, and weather panel |
| `agent.py` | Three-node LangGraph workflow and NVIDIA prompts |
| `weather.py` | Geocoding, current weather requests, and condition codes |
| `requirements.txt` | Pinned Python dependencies |
| `.env.example` | Blank NVIDIA key template |
| `.gitignore` | Excludes credentials, environments, and Python caches |
| `tests/test_agent.py` | Offline graph and Streamlit tests |
| `tests/test_weather.py` | Offline Open-Meteo request and error tests |
| `rapport/rapport.md` | French technical report |

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The tests use simulated API responses and require no credentials.

[Open-Meteo Forecast API](https://open-meteo.com/en/docs) ·
[Open-Meteo Geocoding API](https://open-meteo.com/en/docs/geocoding-api) ·
[LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)
