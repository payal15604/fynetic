import json
import logging
from pathlib import Path
from typing import Any, Dict

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: Path, obj: Any):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def setup_logger(name: str, logs_dir: Path) -> logging.Logger:
    ensure_dir(logs_dir)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(logs_dir / "app.log", encoding="utf-8")
        ch = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
        fh.setFormatter(fmt); ch.setFormatter(fmt)
        logger.addHandler(fh); logger.addHandler(ch)
    return logger
