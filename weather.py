import math

import requests

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
CURRENT_FIELDS = "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code"

# Open-Meteo uses WMO weather codes for current conditions.
CONDITIONS = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Light rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Light snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",
    77: "Snow grains",
    80: "Light rain showers",
    81: "Moderate rain showers",
    82: "Heavy rain showers",
    85: "Light snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with light hail",
    99: "Thunderstorm with heavy hail",
}


def request_data(url: str, params: dict) -> dict:
    """Read JSON from Open-Meteo and turn network failures into readable errors."""
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.Timeout:
        raise RuntimeError("Open-Meteo timed out. Please try again.") from None
    except (requests.exceptions.RequestException, ValueError):
        raise RuntimeError("Could not read data from Open-Meteo. Please try again.") from None
    if not isinstance(data, dict) or data.get("error"):
        raise RuntimeError("Open-Meteo returned an invalid response.")
    return data


def geocode_city(city: str) -> dict:
    """Find the top geocoding match and return its name and coordinates."""
    city = city.strip()
    if not city:
        raise ValueError("Please include a city name.")

    data = request_data(GEOCODING_URL, {"name": city, "count": 1, "language": "en"})
    results = data.get("results")
    if not results:
        raise ValueError(f"I couldn't find a city named {city}. Check the spelling and try again.")
    if not isinstance(results, list) or not isinstance(results[0], dict):
        raise RuntimeError("Open-Meteo returned an invalid city match.")
    location = results[0]
    if any(location.get(key) is None for key in ("name", "latitude", "longitude")):
        raise RuntimeError("Open-Meteo did not return a complete city match.")
    return location


def fetch_weather(city: str) -> dict:
    """Get the city's current conditions in Celsius and km/h."""
    location = geocode_city(city)
    data = request_data(
        FORECAST_URL,
        {
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "current": CURRENT_FIELDS,
            "temperature_unit": "celsius",
            "wind_speed_unit": "kmh",
            "timezone": "auto",
        },
    )
    current = data.get("current")
    if not isinstance(current, dict):
        raise RuntimeError("Open-Meteo has no current weather data for this city.")

    weather = {"city": location["name"], "country": location.get("country"), "time": current.get("time")}
    for label, field in (
        ("temperature", "temperature_2m"),
        ("humidity", "relative_humidity_2m"),
        ("wind_speed", "wind_speed_10m"),
    ):
        value = current.get(field)
        weather[label] = value if type(value) in (int, float) and math.isfinite(value) else None

    code = current.get("weather_code")
    weather["condition"] = CONDITIONS.get(code) if type(code) is int else None
    if all(weather[key] is None for key in ("temperature", "humidity", "wind_speed", "condition")):
        raise RuntimeError("Open-Meteo has no current weather data for this city.")
    return weather
