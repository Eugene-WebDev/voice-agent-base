from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from tools.registry import tool


@tool()
def get_current_time(timezone: str = "Europe/Warsaw") -> str:
    """Get the current date and time in a given IANA timezone.

    Useful when the user asks what time it is, or to timestamp records.
    """
    try:
        tz = ZoneInfo(timezone)
    except ZoneInfoNotFoundError:
        tz = ZoneInfo("UTC")
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
