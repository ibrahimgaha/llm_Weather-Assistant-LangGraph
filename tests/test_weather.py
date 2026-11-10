from unittest.mock import Mock

import pytest
import requests

import weather


def response_with(data):
    response = Mock()
    response.json.return_value = data
    return response


@pytest.mark.parametrize("wind_speed", [0, None])
def test_weather_uses_city_coordinates_and_preserves_missing_values(monkeypatch, wind_speed):
    get = Mock(
        side_effect=[
            response_with(
                {"results": [{"name": "Tunis", "country": "Tunisia", "latitude": 36.8, "longitude": 10.2}]}
            ),
            response_with(
                {
                    "current": {
                        "time": "2026-11-10T22:00",
                        "temperature_2m": 18.5,
                        "relative_humidity_2m": 65,
                        "wind_speed_10m": wind_speed,
                        "weather_code": 61,
                    }
                }
            ),
        ]
    )
    monkeypatch.setattr(weather.requests, "get", get)

    assert weather.fetch_weather(" Tunis ") == {
        "city": "Tunis",
        "country": "Tunisia",
        "time": "2026-11-10T22:00",
        "temperature": 18.5,
        "humidity": 65,
        "wind_speed": wind_speed,
        "condition": "Light rain",
    }
    geocoding, forecast = get.call_args_list
    assert geocoding.args == (weather.GEOCODING_URL,)
    assert geocoding.kwargs["params"]["name"] == "Tunis"
    assert forecast.args == (weather.FORECAST_URL,)
    assert forecast.kwargs["params"]["latitude"] == 36.8
    assert forecast.kwargs["params"]["longitude"] == 10.2
    assert forecast.kwargs["timeout"] == 10


def test_unknown_city_skips_forecast(monkeypatch):
    get = Mock(return_value=response_with({}))
    monkeypatch.setattr(weather.requests, "get", get)
    with pytest.raises(ValueError, match="couldn't find"):
        weather.fetch_weather("Unknown City")
    assert get.call_count == 1


def test_no_current_data(monkeypatch):
    monkeypatch.setattr(weather, "geocode_city", lambda city: {"name": city, "latitude": 1, "longitude": 2})
    monkeypatch.setattr(weather.requests, "get", Mock(return_value=response_with({"current": {}})))
    with pytest.raises(RuntimeError, match="no current weather data"):
        weather.fetch_weather("Tunis")


@pytest.mark.parametrize(
    ("failure", "message"),
    [(requests.exceptions.Timeout(), "timed out"), (requests.exceptions.ConnectionError(), "Could not read")],
)
def test_weather_network_errors(monkeypatch, failure, message):
    monkeypatch.setattr(weather.requests, "get", Mock(side_effect=failure))
    with pytest.raises(RuntimeError, match=message):
        weather.fetch_weather("Tunis")
