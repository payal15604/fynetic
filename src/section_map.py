from pathlib import Path
from typing import Dict
from .utils import read_json

class SectionMap:
    def __init__(self, path: Path):
        self.map: Dict[str, str] = read_json(path)

    def describe(self, item_number: str) -> str:
        return self.map.get(item_number, "Unknown Section")
