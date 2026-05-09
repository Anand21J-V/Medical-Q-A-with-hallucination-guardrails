"""
src/utils/audit_logger.py

Append-only JSONL audit trail of every CRAG decision.
Each line is a self-contained JSON record — easy to parse, query, or stream
into an analytics dashboard.

Schema per record:
{
  "timestamp":    ISO-8601 string,
  "request_id":   UUID,
  "question":     str,
  "confidence":   "high" | "ambiguous" | "low",
  "avg_score":    float,
  "chunk_scores": [{text_preview, score}, ...],
  "web_triggered": bool,
  "web_query":    str | null,
  "sources_used": [str],
  "latency_ms":   int
}
"""

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AuditLogger:
    def __init__(self) -> None:
        settings = get_settings()
        self._path = Path(settings.audit_log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict[str, Any]) -> str:
        """Append a record to the JSONL audit file. Returns the request_id."""
        request_id = record.get("request_id") or str(uuid.uuid4())
        record["request_id"] = request_id
        record["timestamp"] = datetime.now(timezone.utc).isoformat()

        try:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("audit_write_failed", error=str(exc))

        return request_id

    def tail(self, n: int = 50) -> list[dict[str, Any]]:
        """Return the last n audit records."""
        if not self._path.exists():
            return []
        lines = self._path.read_text(encoding="utf-8").splitlines()
        records = []
        for line in lines[-n:]:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records


_audit_logger: AuditLogger | None = None


def get_audit_logger() -> AuditLogger:
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
