"""מונה קריאות יומי שעוצר אותנו לפני שחורגים מהמכסה החינמית.

כל פעם שהקוד שלנו שולח בקשה ל-Gemini, קוראים ל-register("llm") או ל-register("embed").
המונה נשמר בקובץ usage_log.json ומתאפס כל יום.
"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config import DAILY_LIMITS, PROJECT_DIR

LOG_FILE = PROJECT_DIR / "usage_log.json"


def _quota_day() -> str:
    # המכסה של Google מתאפסת בחצות שעון קליפורניה (UTC-8 בחורף, UTC-7 בקיץ).
    # לוקחים UTC-8 כדי לא לאפס מוקדם מדי.
    return (datetime.now(timezone.utc) - timedelta(hours=8)).strftime("%Y-%m-%d")


def _load() -> dict:
    if LOG_FILE.exists():
        data = json.loads(LOG_FILE.read_text(encoding="utf-8"))
        if data.get("day") == _quota_day():
            return data
    return {"day": _quota_day(), "llm": 0, "embed": 0}


class QuotaExceeded(SystemExit):
    """נזרקת כשהמגבלה היומית נגמרה.

    יורשת מ-SystemExit: בשורת הפקודה התוכנית נעצרת עם ההודעה, כמו sys.exit.
    בממשק (app.py) תופסים אותה ומציגים הודעה יפה, בלי שהאפליקציה תיעצר.
    """


def register(kind: str, count: int = 1) -> None:
    """רושם קריאה. אם היא תחרוג מהמגבלה, עוצר עם הודעה ברורה."""
    data = _load()
    if data[kind] + count > DAILY_LIMITS[kind]:
        raise QuotaExceeded(
            f"עצירה: הגעת למגבלה היומית שהגדרנו לקריאות {kind} "
            f"({DAILY_LIMITS[kind]}). המכסה מתאפסת בערך ב-10:00 בבוקר שעון ישראל."
        )
    data[kind] += count
    LOG_FILE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def remaining(kind: str) -> int:
    """כמה קריאות מסוג kind נשארו לנו היום, לפי המגבלה שהגדרנו."""
    return DAILY_LIMITS[kind] - _load()[kind]


def report() -> str:
    data = _load()
    return (f"שימוש היום: מודל שפה {data['llm']}/{DAILY_LIMITS['llm']}, "
            f"Embeddings {data['embed']}/{DAILY_LIMITS['embed']}")
