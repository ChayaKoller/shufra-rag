"""שלב 3: טעינת המסמכים ובדיקה שהעברית תקינה.

הסקריפט:
1. טוען את כל המסמכים לפי metadata.csv (בעזרת loader.py).
2. מדפיס טבלה: איזה מסמך נטען, מה הקטגוריה, הגרסה, התאריך והאורך.
3. בודק שהעברית לא השתבשה, ומדפיס את תחילת כל מסמך.
4. מראה מה בדיוק "יראה" מודל ה-Embedding (טקסט + מטא-דאטה).

לא נשלחת אף קריאה ל-Gemini: 0 קריאות מהמכסה.
"""
import ssl_fix  # חייב להיות ראשון (גם כאן, כדי לשמור על הרגל אחיד בכל הסקריפטים)

import sys

from llama_index.core.schema import MetadataMode

from loader import files_without_metadata, load_documents

# בטרמינל של Windows הדפסה בעברית עלולה להיכשל בלי השורה הזו
sys.stdout.reconfigure(encoding="utf-8")


def hebrew_letters(text: str) -> int:
    """סופר אותיות עבריות (א עד ת) בטקסט."""
    return sum(1 for ch in text if "א" <= ch <= "ת")


def check_hebrew(text: str) -> list[str]:
    """מחזיר רשימת בעיות קידוד שנמצאו (רשימה ריקה = הכול תקין)."""
    problems = []
    if "�" in text:
        problems.append("יש תווים שבורים (�): הקובץ נקרא בקידוד לא נכון")
    if text.startswith("﻿"):
        problems.append("סימן BOM נשאר בתחילת הטקסט")
    if "×" in text and hebrew_letters(text) == 0:
        problems.append("נראה כמו ג'יבריש (×...): עברית שנקראה כ-Latin-1")
    if hebrew_letters(text) < 50:
        problems.append("כמעט אין אותיות עבריות")
    return problems


documents = load_documents()
print(f"נטענו {len(documents)} מסמכים.\n")

# --- טבלת סיכום ---
print(f"{'מזהה':<12} {'קטגוריה':<8} {'גרסה':<5} {'עודכן':<11} {'תווים':>6} {'אותיות עבריות':>13}  כותרת")
print("-" * 95)
total_chars = 0
all_ok = True
for doc in documents:
    m = doc.metadata
    total_chars += len(doc.text)
    print(f"{m['document_id']:<12} {m['category']:<8} {m['version']:<5} {m['updated_at']:<11} "
          f"{len(doc.text):>6} {hebrew_letters(doc.text):>13}  {m['title']}")
print("-" * 95)
print(f"סה\"כ {total_chars:,} תווים.\n")

# --- בדיקת העברית ---
print("בדיקת עברית ותחילת כל מסמך:")
for doc in documents:
    problems = check_hebrew(doc.text)
    status = "תקין" if not problems else "בעיה: " + "; ".join(problems)
    all_ok = all_ok and not problems
    first_line = doc.text.strip().splitlines()[0][:70]
    print(f"  [{status}] {doc.metadata['document_id']}: {first_line}")

# --- קבצים שנשכחו ---
missing = files_without_metadata()
if missing:
    all_ok = False
    print("\nשימי לב: הקבצים האלה בתיקייה אבל לא ב-metadata.csv, ולכן לא נטענו:", missing)

# --- מה רואה מודל ה-Embedding? ---
example = next(d for d in documents if d.metadata["document_id"] == "TBL-PRICES")
print("\nכך ייראה מסמך טבלת המחירים כשיישלח ל-Embedding (300 תווים ראשונים):")
print(example.get_content(metadata_mode=MetadataMode.EMBED)[:300])

print("\nהטעינה הצליחה, הכול תקין." if all_ok else "\nנמצאו בעיות, ראי למעלה.")
print("קריאות ל-Gemini בשלב הזה: 0.")
