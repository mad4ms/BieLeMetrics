from typing import Dict, Any


class Referee:
    """
    Stores referee information.
    """

    def __init__(self, data: Dict[str, Any]) -> None:
        self.referee_id: str = data.get("id", "")
        self.name: str = data.get("name", "")
        self.nationality: str = data.get("nationality", "")
        self.country_code: str = data.get("country_code", "")
        self.ref_type: str = data.get("type", "")

    def __repr__(self) -> str:
        return f"Referee ID: {self.referee_id}, Name: {self.name}, Nationality: {self.nationality}, Type: {self.ref_type}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "referee_id": self.referee_id,
            "name": self.name,
            "nationality": self.nationality,
            "country_code": self.country_code,
            "ref_type": self.ref_type,
        }
