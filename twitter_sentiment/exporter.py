from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

from . import jsonio
import pandas as pd


def export_jsonl(records: Iterable[Dict], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for record in records:
            f.write(jsonio.dumps(record))
            f.write(b"\n")


def export_csv(records: Iterable[Dict], output_path: str) -> None:
    df = pd.DataFrame(list(records))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def export_parquet(records: Iterable[Dict], output_path: str, engine: Optional[str] = None) -> None:
    df = pd.DataFrame(list(records))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine=engine)


def export_records(records: Iterable[Dict], fmt: str, output_path: str) -> None:
    fmt_normalized = fmt.lower().strip()
    if fmt_normalized == "jsonl":
        export_jsonl(records, output_path)
    elif fmt_normalized == "csv":
        export_csv(records, output_path)
    elif fmt_normalized == "parquet":
        export_parquet(records, output_path)
    else:
        raise ValueError(f"Unsupported export format: {fmt}")

