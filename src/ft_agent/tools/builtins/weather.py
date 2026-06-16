def get_mock_weather(city: str) -> dict[str, str]:
    return {
        "city": city,
        "condition": "sunny",
        "temperature": "24C",
        "source": "mock",
    }
