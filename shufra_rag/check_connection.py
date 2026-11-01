"""check_connection.py: בדיקת החיבור ל-Gemini (שלב 2). אפשר להריץ שוב בכל פעם שמשהו לא עובד.

הסקריפט עושה שלושה דברים:
1. מדפיס את המודלים שהמפתח שלך יכול להשתמש בהם (בלי לצרוך מכסה של מודל שפה).
2. שולח שאלה קצרה אחת למודל השפה (קריאה אחת מהמכסה).
3. ממיר משפט אחד לווקטור (קריאה אחת מהמכסה).

סה"כ: 2 קריאות בלבד.
"""
import ssl_fix  # חייב להיות ראשון: מתקן שגיאת SSL ברשת מסוננת

import os
import sys

from dotenv import load_dotenv
from google import genai

import usage_guard
from config import EMBED_MODEL, LLM_MODEL

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key or api_key == "paste-your-key-here":
    sys.exit("לא נמצא מפתח. צרי קובץ .env לפי .env.example והדביקי בו את המפתח.")

# --- 1. אילו מודלים זמינים? ---
client = genai.Client(api_key=api_key)
print("מודלים זמינים לך (Flash / Embedding בלבד):")
names = []
for m in client.models.list():
    name = m.name.replace("models/", "")
    names.append(name)
    if "flash" in name or "embedding" in name:
        print("  ", name)

for wanted in (LLM_MODEL, EMBED_MODEL):
    if wanted not in names:
        sys.exit(f"\nהמודל {wanted} לא נמצא ברשימה. בחרי שם מהרשימה ועדכני ב-config.py.")

# --- 2. קריאה אחת למודל השפה ---
from llama_index.llms.google_genai import GoogleGenAI

llm = GoogleGenAI(model=LLM_MODEL, api_key=api_key, max_retries=1)
usage_guard.register("llm")
answer = llm.complete("כתבי משפט אחד בעברית: שלום, החיבור עובד.")
print("\nתשובת מודל השפה:", answer.text.strip())

# --- 3. קריאה אחת למודל ה-Embeddings ---
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

embed = GoogleGenAIEmbedding(model_name=EMBED_MODEL, api_key=api_key, retries=1)
usage_guard.register("embed")
vector = embed.get_text_embedding("עד מתי אפשר לבטל הרשמה לקורס?")
print(f"\nהמשפט הומר לווקטור באורך {len(vector)}. חמשת המספרים הראשונים:", [round(x, 4) for x in vector[:5]])

print("\nהחיבור תקין.", usage_guard.report())
