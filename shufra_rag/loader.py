"""טעינת המידע הארגוני לאובייקטים של LlamaIndex (Document).

הקובץ הזה הוא "ספרייה" קטנה שלנו: הוא לא מדפיס כלום, רק מחזיר רשימת מסמכים.
בשלב 3 נשתמש בו כדי לבדוק את הטעינה, ובשלב 4 (בניית האינדקס) נשתמש בו שוב.

הטעינה לא פונה ל-Gemini בכלל, ולכן לא צורכת מכסה.
"""
import csv
from pathlib import Path

from llama_index.core import Document

from config import DATA_DIR

METADATA_FILE = DATA_DIR / "metadata.csv"

# קבצים שנמצאים בתיקיית המידע אבל אינם מידע ארגוני לתלמידות:
# metadata.csv הוא "כרטיס הזיהוי" של המסמכים, ו-questions.csv הן שאלות הבדיקה שלנו.
NOT_CONTENT = {"metadata.csv", "questions.csv"}

# שמות עמודות טבלת המחירים, בעברית, כדי שהמודל יבין כל שורה.
CSV_LABELS = {
    "course_id": "קוד קורס",
    "course_name": "שם הקורס",
    "field": "תחום",
    "hours": "שעות לימוד",
    "sessions": "מספר מפגשים",
    "days": "ימי לימוד",
    "time": "שעות המפגש",
    "start_date": "תאריך פתיחה",
    "estimated_end": "סיום משוער",
    "price_ils": "מחיר בשקלים",
    "certificate": "תעודה מטעם",
}

# שדות מטא-דאטה שלא נשלחים ל-Embedding ולמודל השפה.
# הם שימושיים לנו (לסינון ולהצגת מקורות), אבל לא עוזרים להבין את התוכן.
HIDDEN_FROM_MODELS = ["file_path", "access_scope", "document_id"]


def read_metadata() -> list[dict]:
    """קורא את metadata.csv ומחזיר שורה (מילון) לכל מסמך."""
    # utf-8-sig: מתעלם מסימן BOM שאקסל מוסיף לפעמים בתחילת הקובץ
    with open(METADATA_FILE, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def csv_to_text(path: Path) -> str:
    """הופך טבלת CSV לטקסט קריא: שורה אחת לכל קורס, עם שם לכל שדה.

    טבלה גולמית ("ACC-101,120,30,4800") קשה למודל להבין.
    "קוד קורס: ACC-101 | מחיר בשקלים: 4800" ברור הרבה יותר.
    """
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    lines = []
    for row in rows:
        parts = [f"{CSV_LABELS.get(key, key)}: {value}" for key, value in row.items()]
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def load_documents() -> list[Document]:
    """טוען את כל המסמכים שמופיעים ב-metadata.csv, כל אחד עם המטא-דאטה שלו."""
    documents = []
    for meta in read_metadata():
        path = DATA_DIR / meta["file_path"]
        if not path.exists():
            raise FileNotFoundError(f"הקובץ {meta['file_path']} מופיע ב-metadata.csv אבל לא נמצא בתיקייה.")

        if path.suffix == ".csv":
            text = csv_to_text(path)
        else:
            text = path.read_text(encoding="utf-8-sig")

        doc = Document(
            id_=meta["document_id"],   # מזהה קבוע: חשוב כשנעדכן מסמכים בעתיד
            text=text,
            metadata={
                "document_id": meta["document_id"],
                "title": meta["title"],
                "category": meta["category"],
                "version": int(meta["version"]),
                "updated_at": meta["updated_at"],
                "access_scope": meta["access_scope"],
                "file_path": meta["file_path"],
            },
            excluded_embed_metadata_keys=HIDDEN_FROM_MODELS,
            excluded_llm_metadata_keys=HIDDEN_FROM_MODELS,
        )
        documents.append(doc)
    return documents


def files_without_metadata() -> list[str]:
    """מחזיר קבצים שנמצאים בתיקייה אבל חסרים ב-metadata.csv (כדי שלא נשכח אף מסמך)."""
    listed = {m["file_path"] for m in read_metadata()}
    on_disk = {
        p.relative_to(DATA_DIR).as_posix()
        for p in DATA_DIR.rglob("*")
        if p.is_file() and p.suffix in (".md", ".csv") and p.name not in NOT_CONTENT
    }
    return sorted(on_disk - listed)
