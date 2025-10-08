from __future__ import annotations

from typing import Any

try:
    import orjson as _orjson

    def dumps(obj: Any) -> bytes:
        return _orjson.dumps(obj)

    def loads(data: str | bytes | bytearray) -> Any:
        return _orjson.loads(data)

except Exception:  # Fallback to stdlib json
    import json as _json

    def dumps(obj: Any) -> bytes:
        # Ensure UTF-8 bytes output to imitate orjson's bytes
        return _json.dumps(obj, ensure_ascii=False).encode("utf-8")

    def loads(data: str | bytes | bytearray) -> Any:
        if isinstance(data, (bytes, bytearray)):
            return _json.loads(data.decode("utf-8"))
        return _json.loads(data)

