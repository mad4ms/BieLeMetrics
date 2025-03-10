from typing import Optional, Any, Dict


class Venue:
    """
    Stores venue details such as name, capacity, location, and timezone.
    """

    def __init__(self, data: Dict[str, Any]) -> None:
        self.venue_id: str = data.get("id", "")
        self.name: str = data.get("name", "")
        self.capacity: Optional[int] = data.get("capacity", 0)
        self.city_name: str = data.get("city_name", "")
        self.country_name: str = data.get("country_name", "")
        self.map_coordinates: str = data.get("map_coordinates", "")
        self.country_code: str = data.get("country_code", "")
        self.timezone: str = data.get("timezone", "")

    def __repr__(self) -> str:
        return f"Venue ID: {self.venue_id}, Name: {self.name}, City: {self.city_name}, Country: {self.country_name}"
