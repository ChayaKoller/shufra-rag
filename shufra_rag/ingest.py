"""ingest.py: הכנת המידע. טעינת המסמכים וחיתוך שלהם לקטעים.

תחום האחריות של הקובץ: כל מה שקורה למידע *לפני* שהוא פוגש את Gemini.
1. load_documents(): קורא את קבצי המידע לפי metadata.csv והופך כל אחד ל-Document.
2. build_nodes():    מחלק את המסמכים לקטעים חופפים (Nodes), היחידה שהמערכת תחפש בה.

שום דבר כאן לא פונה ל-Gemini, ולכן לא נצרכת מכסה.
הרצה ישירה (python ingest.py) מדפיסה בדיקה מלאה: מה נטען, האם העברית תקינה, ואיך נראים הקטעים.
"""
import ssl_fix  # חייב להיות ראשון: מתקן שגיאת SSL ברשת מסוננת

import csv
import hashlib
import re
from pathlib import Path

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import NodeRelationship, TextNode

from config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR, FACTS

METADATA_FILE = DATA_DIR / "metadata.csv"

# קבצים בתיקיית המידע שאינם מידע לתלמידות: כרטיס הזיהוי של המסמכים, שאלות הבדיקה, וקובץ העובדות המשותפות.
NOT_CONTENT = {"metadata.csv", "questions.csv", "facts.csv"}

# שמות עמודות טבלת המחירים בעברית, כדי שהמודל יבין כל שורה.
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
    "registration_fee_ils": "דמי רישום בשקלים (בנוסף למחיר, לא מוחזרים)",
}

# מטא-דאטה ששמורה אצלנו (לסינון ולהצגת מקורות) אבל לא נשלחת ל-Embedding ולמודל השפה,
# כי היא לא עוזרת להבין את התוכן ורק מוסיפה "רעש".
HIDDEN_FROM_MODELS = ["file_path", "access_scope", "document_id", "doc_hash"]


# ---------------------------------------------------------------------------
# חלק 1: טעינת המסמכים
# ---------------------------------------------------------------------------

def _read_metadata() -> list[dict]:
    # utf-8-sig: קורא עברית נכון ומתעלם מסימן BOM שאקסל מוסיף בתחילת הקובץ
    with open(METADATA_FILE, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _csv_to_text(path: Path) -> str:
    """טבלה -> טקסט קריא: שורה לכל קורס, עם שם לכל שדה ("מחיר בשקלים: 4800")."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return "\n".join(
        " | ".join(f"{CSV_LABELS.get(k, k)}: {v}" for k, v in row.items()) for row in rows
    )


PLACEHOLDER = re.compile(r"\{\{([A-Z_]+)\}\}")


def fill_facts(text: str, file_path: str) -> str:
    """ממלא {{REG_FEE}} וכדומה בערכים מ-facts.csv (מקור אמת אחד לכל עובדה שחוזרת בכמה מסמכים).

    המילוי נעשה לפני חישוב טביעת האצבע, ולכן שינוי ערך ב-facts.csv מסמן כ"השתנה"
    בדיוק את המסמכים שמשתמשים בו, ו---update מטמיע מחדש רק אותם.
    """
    missing = sorted({m for m in PLACEHOLDER.findall(text) if m not in FACTS})
    if missing:
        raise KeyError(f"{file_path}: חסרים ב-facts.csv הערכים {missing}")
    return PLACEHOLDER.sub(lambda m: FACTS[m.group(1)], text)


def fingerprint(text: str, meta: dict) -> str:
    """טביעת אצבע (hash) של מסמך: מחרוזת קצרה שמשתנה אם משתנה אפילו אות אחת.

    כוללים גם את הגרסה, התאריך והכותרת, כי גם הם נכנסים ל-Embedding.
    rag.py משווה את טביעת האצבע לזו שנשמרה באינדקס, וכך יודע אילו מסמכים השתנו.
    """
    content = "|".join([text, meta["version"], meta["updated_at"], meta["title"]])
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def load_documents() -> list[Document]:
    """טוען כל מסמך שמופיע ב-metadata.csv, עם כרטיס הזיהוי שלו."""
    documents = []
    for meta in _read_metadata():
        path = DATA_DIR / meta["file_path"]
        if not path.exists():
            raise FileNotFoundError(f"{meta['file_path']} מופיע ב-metadata.csv אבל לא נמצא בתיקייה.")

        text = _csv_to_text(path) if path.suffix == ".csv" else path.read_text(encoding="utf-8-sig")
        text = fill_facts(text, meta["file_path"])
        documents.append(Document(
            id_=meta["document_id"],          # מזהה קבוע, חשוב לעדכון מסמכים בעתיד
            text=text,
            metadata={
                "document_id": meta["document_id"],
                "title": meta["title"],
                "category": meta["category"],
                "version": int(meta["version"]),
                "updated_at": meta["updated_at"],
                "access_scope": meta["access_scope"],
                "file_path": meta["file_path"],
                "doc_hash": fingerprint(text, meta),   # "טביעת אצבע": משתנה כשהמסמך משתנה
            },
            excluded_embed_metadata_keys=HIDDEN_FROM_MODELS,
            excluded_llm_metadata_keys=HIDDEN_FROM_MODELS,
        ))
    return documents


def files_without_metadata() -> list[str]:
    """קבצים שבתיקייה אבל חסרים ב-metadata.csv, כדי שלא נשכח אף מסמך."""
    listed = {m["file_path"] for m in _read_metadata()}
    on_disk = {
        p.relative_to(DATA_DIR).as_posix()
        for p in DATA_DIR.rglob("*")
        if p.is_file() and p.suffix in (".md", ".csv") and p.name not in NOT_CONTENT
    }
    return sorted(on_disk - listed)


# ---------------------------------------------------------------------------
# חלק 2: חיתוך לקטעים (Chunking)
# ---------------------------------------------------------------------------

def build_nodes(documents: list[Document]) -> list[TextNode]:
    """מחלק את המסמכים לקטעים חופפים.

    מסמכי טקסט: "חלון" של CHUNK_SIZE טוקנים שזז לאורך המסמך. כל קטע חדש מתחיל
    CHUNK_OVERLAP טוקנים לפני סוף הקטע הקודם, כך שבין כל שני קטעים צמודים יש חפיפה.
    החיתוך נעשה בגבול של משפט, לא באמצע מילה.

    טבלה: כל שורה (קורס אחד) היא קטע, בלי חפיפה. שורות הן רשומות נפרדות,
    וחפיפה הייתה רק מערבבת מחיר של קורס אחד עם קורס אחר.
    """
    text_docs = [d for d in documents if d.metadata["category"] != "table"]
    table_docs = [d for d in documents if d.metadata["category"] == "table"]

    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(text_docs)

    for doc in table_docs:
        for line in filter(str.strip, doc.text.splitlines()):
            node = TextNode(
                text=line,
                metadata=dict(doc.metadata),
                excluded_embed_metadata_keys=doc.excluded_embed_metadata_keys,
                excluded_llm_metadata_keys=doc.excluded_llm_metadata_keys,
            )
            node.relationships[NodeRelationship.SOURCE] = doc.as_related_node_info()
            nodes.append(node)
    return nodes


# ---------------------------------------------------------------------------
# בדיקה עצמית: python ingest.py  (0 קריאות ל-Gemini)
# ---------------------------------------------------------------------------

def _hebrew_problems(text: str) -> list[str]:
    hebrew = sum(1 for ch in text if "א" <= ch <= "ת")
    problems = []
    if "�" in text:
        problems.append("תווים שבורים: הקובץ נקרא בקידוד לא נכון")
    if text.startswith("﻿"):
        problems.append("סימן BOM נשאר בתחילת הטקסט")
    if hebrew < 50:
        problems.append("כמעט אין אותיות עבריות (אולי ג'יבריש)")
    return problems


if __name__ == "__main__":
    import sys
    from collections import Counter

    from llama_index.core.schema import MetadataMode
    from llama_index.core.utils import get_tokenizer

    from config import EMBED_BATCH_SIZE

    sys.stdout.reconfigure(encoding="utf-8")  # הדפסת עברית בטרמינל של Windows
    tokens = get_tokenizer()                   # סופר טוקנים (קירוב טוב גם ל-Gemini)

    # --- חלק 1: טעינה ---
    documents = load_documents()
    print(f"=== טעינה: {len(documents)} מסמכים ===")
    all_ok = True
    for doc in documents:
        m = doc.metadata
        problems = _hebrew_problems(doc.text)
        all_ok = all_ok and not problems
        status = "תקין" if not problems else "בעיה: " + "; ".join(problems)
        print(f"  [{status}] {m['document_id']:<12} {m['category']:<7} גרסה {m['version']}  "
              f"{m['updated_at']}  {len(doc.text):>5} תווים  {m['title']}")
    missing = files_without_metadata()
    if missing:
        all_ok = False
        print("  שימי לב: קבצים שלא נטענו כי חסרים ב-metadata.csv:", missing)

    # --- חלק 2: חיתוך ---
    nodes = build_nodes(documents)
    sizes = [len(tokens(n.get_content(metadata_mode=MetadataMode.EMBED))) for n in nodes]
    per_doc = Counter(n.metadata["document_id"] for n in nodes)
    print(f"\n=== חיתוך: {len(nodes)} קטעים (CHUNK_SIZE={CHUNK_SIZE}, CHUNK_OVERLAP={CHUNK_OVERLAP}) ===")
    print("  קטעים לכל מסמך:", ", ".join(f"{k} {v}" for k, v in per_doc.items()))
    print(f"  גודל קטע בטוקנים: הקטן {min(sizes)}, הגדול {max(sizes)}, ממוצע {sum(sizes) // len(sizes)}")

    print("\n=== כך נראה קטע כשהוא נשלח ל-Embedding (שתי דוגמאות) ===")
    for node in (next(n for n in nodes if "ביטול" in n.text and n.metadata["document_id"] == "POL-REG"),
                 next(n for n in nodes if n.metadata["category"] == "table")):
        print("-" * 60)
        print(node.get_content(metadata_mode=MetadataMode.EMBED)[:400])
    print("-" * 60)

    # --- החפיפה בפועל: מה משותף לקטע הראשון ולשני במסמך הרישום ---
    reg = [n for n in nodes if n.metadata["document_id"] == "POL-REG"]
    first, second = reg[0].text, reg[1].text
    shared = next((second[:k] for k in range(len(second), 0, -1) if second[:k] in first), "")
    print("\n=== חפיפה: הטקסט שמופיע גם בסוף קטע 1 וגם בתחילת קטע 2 (POL-REG) ===")
    print(shared if shared else "(אין חפיפה בין שני הקטעים האלה)")
    overlapping = sum(
        1 for a, b in zip(nodes, nodes[1:])
        if a.metadata["document_id"] == b.metadata["document_id"]
        and a.metadata["category"] != "table" and b.text[:40] in a.text
    )
    pairs = sum(
        1 for a, b in zip(nodes, nodes[1:])
        if a.metadata["document_id"] == b.metadata["document_id"] and a.metadata["category"] != "table"
    )
    print(f"זוגות קטעים צמודים עם חפיפה: {overlapping} מתוך {pairs}")

    requests = -(-len(nodes) // EMBED_BATCH_SIZE)
    print(f"\nבניית האינדקס תשלח {len(nodes)} קטעים ב-{requests} בקשות.")
    print("הכול תקין." if all_ok else "נמצאו בעיות, ראי למעלה.", "קריאות ל-Gemini: 0.")
