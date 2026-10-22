import hashlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load(path="config/pilot.yaml"):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    text = p.read_text(encoding="utf-8")
    cfg = yaml.safe_load(text)
    for name, m in cfg["methods"].items():
        if len(m["variations"]) != cfg["num_variations"]:
            raise ValueError(f"El método {name} define {len(m['variations'])} variaciones; deben ser {cfg['num_variations']}.")
    cfg["_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return cfg
