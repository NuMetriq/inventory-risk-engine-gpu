from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import json


def make_run_dir(root: str = "artifacts/runs", run_name: Optional[str] = None) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    name = run_name or ts
    p = Path(root) / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def to_jsonable(x: Any) -> Any:
    if is_dataclass(x):
        return asdict(x)
    if isinstance(x, Path):
        return str(x)
    return x


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps({k: to_jsonable(v) for k, v in obj.items()}, indent=2))