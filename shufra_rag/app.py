"""app.py: ממשק הצ'אט של יועצת הלימודים (שלב 7).

תחום האחריות של הקובץ: רק התצוגה. כל ה-RAG (שליפה, תבנית, מודל שפה) נשאר ב-rag.py,
והממשק רק קורא ל-answer() ומציג את מה שחזר: תשובה + מקורות.

הרצה (מתוך התיקייה shufra_rag, כשה-venv פעיל):
    streamlit run app.py
הדפדפן ייפתח בכתובת http://localhost:8501

עלות: פתיחת הממשק = 0 קריאות. שאלה ראשונה = קריאת Embedding אחת + קריאת מודל שפה אחת, כמו --ask.
שאלת המשך (כשיש שאלות קודמות בשיחה) = עוד קריאת מודל שפה אחת, לניסוח השאלה מחדש.
"""
import ssl_fix  # חייב להיות ראשון: מתקן שגיאת SSL ברשת מסוננת

import csv
import re
import traceback

import streamlit as st

import usage_guard
from config import CONSULT_MESSAGE, DATA_DIR, HISTORY_TURNS
from rag import answer, build_query_engine

# ---------------------------------------------------------------------------
# הגדרות עמוד ועיצוב מימין לשמאל
# ---------------------------------------------------------------------------

# חייב להיות הפקודה הראשונה של Streamlit בקובץ
st.set_page_config(page_title="שופרא – יועצת לימודים", page_icon="🎓", layout="centered")

# Streamlit בנוי לשפות משמאל לימין. ה-CSS הזה הופך את הכיוון לעברית.
# data-testid הם השמות ש-Streamlit נותן לרכיבים שלו (רואים אותם ב"בדיקת רכיב" בדפדפן).
RTL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Heebo:wght@400;500;700&display=swap');

.stApp, [data-testid="stSidebar"] {
    direction: rtl;
    font-family: 'Heebo', Arial, sans-serif;
}
.stApp p, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp label,
[data-testid="stChatMessage"], [data-testid="stExpander"] {
    text-align: right;
}
[data-testid="stChatInput"] textarea {
    direction: rtl;
    text-align: right;
}
/* שמות קבצים באנגלית נשארים משמאל לימין, כדי שלא יתבלגנו */
.stApp code {
    direction: ltr;
    unicode-bidi: embed;
}
</style>
"""
st.markdown(RTL_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# מנוע השאלות: נבנה פעם אחת בלבד
# ---------------------------------------------------------------------------

# Streamlit מריץ את כל הקובץ מחדש בכל לחיצה ובכל שאלה.
# בלי cache_resource היינו פותחים את Chroma ובונים את המנוע מחדש בכל פעם.
# עם cache_resource המנוע נבנה בפעם הראשונה ונשמר בזיכרון. הבנייה עצמה לא שולחת קריאות ל-Gemini.
@st.cache_resource(show_spinner="טוענת את מאגר המידע...")
def get_engine():
    return build_query_engine()


# שאלות לדוגמה שמוצגות כשהצ'אט ריק (מתוך בנק השאלות)
EXAMPLES = [
    "כמה עולה קורס הנהלת חשבונות?",
    "איזו תעודה מקבלים בסוף קורס הגרפיקה?",
    "אפשר לדבר עם מישהי לפני שאני מחליטה על קורס?",
    "עד מתי אפשר לבטל הרשמה ולקבל את כל הכסף בחזרה?",
]

WELCOME = ("שלום וברוכה הבאה לשופרא – למידה שעובדת! 🌸  \n"
           "אני יועצת הלימודים של המרחב הלימודי. אפשר לשאול אותי על הקורסים, המחירים, "
           "המועדים וההרשמה, ואשמח לעזור לך למצוא את המסלול שמתאים לך.")


# ---------------------------------------------------------------------------
# תצוגה
# ---------------------------------------------------------------------------

# מספרים בתוך טקסט עברי: הדפדפן הופך לפעמים את הסדר, למשל "09:00–15:00" מוצג "15:00–09:00".
# הפתרון: לעטוף כל רצף של מספר בתווים בלתי נראים (LRI ו-PDI) שאומרים "את זה כתוב משמאל לימין".
NUMBER_RUN = re.compile(r"\d[\d:.,/\-–]*\d")


def fix_bidi(text: str) -> str:
    return NUMBER_RUN.sub(lambda m: "\u2066" + m.group() + "\u2069", text)


def show_sources(sources: list[dict], dev_mode: bool) -> None:
    """רשימת המסמכים שהתשובה מבוססת עליהם, בתוך תיבה מתקפלת."""
    if not sources:
        return
    with st.expander(f"📄 מקורות ({len(sources)})"):
        for s in sources:
            line = f"**{s['title']}** · עודכן {s['updated_at']}"
            if dev_mode:
                line += f" · ציון {s['score']:.3f}  \n`{s['file_path']}`"
            st.markdown(line)


def show_chunks(chunks: list[dict]) -> None:
    """מצב מפתחת: הקטעים שנשלחו למודל, כמו --chunks בשורת הפקודה."""
    with st.expander(f"🔍 הקטעים שנשלחו למודל ({len(chunks)})"):
        for c in chunks:
            st.markdown(f"`{c['document_id']}` · ציון {c['score']:.3f}")
            st.caption(c["text"][:300] + "...")


def show_message(msg: dict, dev_mode: bool) -> None:
    avatar = "🎓" if msg["role"] == "assistant" else "🙂"
    with st.chat_message(msg["role"], avatar=avatar):
        # ב-Markdown ירידת שורה בודדת נבלעת, והשורות מתחברות. "  \n" שומר אותה (חשוב בשורות החישוב)
        st.markdown(fix_bidi(msg["content"]).replace("\n", "  \n"))
        if msg["role"] == "assistant":
            if dev_mode and msg.get("standalone"):
                st.caption(f"🔁 השאלה נוסחה מחדש לחיפוש: {msg['standalone']}")
            show_sources(msg.get("sources", []), dev_mode)
            if dev_mode and msg.get("chunks"):
                show_chunks(msg["chunks"])
            if msg.get("expected"):           # שאלה מבנק הבדיקה: משווים לתשובה הצפויה
                st.success(f"**התשובה הצפויה ({msg['test_id']}):** {fix_bidi(msg['expected'])}", icon="🧪")


@st.cache_data
def load_test_questions() -> list[dict]:
    """בנק שאלות הבדיקה (questions.csv). נקרא פעם אחת, בלי קריאות ל-Gemini."""
    with open(DATA_DIR / "questions.csv", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def recent_history(messages: list[dict]) -> list[dict]:
    """היסטוריית הצ'אט -> HISTORY_TURNS הזוגות האחרונים של שאלה ותשובה, בשביל answer().

    הודעות התקלה לא נכנסות: אין בהן מידע שעוזר להבין את השאלה הבאה.
    """
    pairs = []
    for user_msg, bot_msg in zip(messages[::2], messages[1::2]):   # ההודעות מסודרות: שאלה, תשובה, שאלה...
        if not bot_msg.get("error"):
            pairs.append({"question": user_msg["content"], "answer": bot_msg["content"]})
    return pairs[-HISTORY_TURNS:]


def ask(question: str, history: list[dict]) -> dict:
    """שולח שאלה ל-answer() ומחזיר הודעה מוכנה לתצוגה. תקלות הופכות להודעה, לא לקריסה."""
    try:
        result = answer(question, get_engine(), history)
    except usage_guard.QuotaExceeded as e:       # המגבלה היומית שלנו נגמרה
        return {"role": "assistant", "content": f"⏳ {e}", "sources": [], "error": True}
    except Exception as e:                        # למשל 429 (יותר מדי בקשות בדקה) או בעיית רשת
        traceback.print_exc()                     # הפירוט המלא מודפס בטרמינל, בשביל אבחון
        return {"role": "assistant", "sources": [], "error": True,
                "content": ("מצטערת, הייתה תקלה בפנייה למודל. כדאי לחכות דקה ולנסות שוב.  \n"
                            f"(פרטים טכניים: {type(e).__name__}: {e})")}

    # שומרים את הקטעים כטקסט פשוט: את אובייקטי LlamaIndex לא צריך לשמור בזיכרון הצ'אט
    chunks = [{"document_id": r.node.metadata["document_id"], "score": r.score,
               "text": r.node.get_content().replace("\n", " ")} for r in result["retrieved"]]
    standalone = result["standalone"] if result["standalone"] != question else None
    return {"role": "assistant", "content": result["answer"], "standalone": standalone,
            "sources": result["sources"], "chunks": chunks}


# ---------------------------------------------------------------------------
# העמוד
# ---------------------------------------------------------------------------

# st.session_state = הזיכרון של העמוד בין הרצה להרצה. כאן נשמרת היסטוריית הצ'אט.
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("שופרא – למידה שעובדת")
    st.write("מרחב לימודי לקורסים מקצועיים. כל הלימודים פרונטליים.")
    st.info(fix_bidi(CONSULT_MESSAGE), icon="📞")
    dev_mode = st.toggle("מצב מפתחת (ציונים וקטעים)", value=False)
    st.caption(f"היועצת זוכרת את {HISTORY_TURNS} השאלות האחרונות בשיחה.")

    # בנק שאלות הבדיקה: מופיע רק במצב מפתחת
    test_row = None
    if dev_mode:
        with st.expander("🧪 בנק שאלות הבדיקה", expanded=True):
            bank = load_test_questions()
            types = ["הכול"] + sorted({q["question_type"] for q in bank})
            chosen_type = st.selectbox("סוג שאלה", types)
            shown = [q for q in bank if chosen_type == "הכול" or q["question_type"] == chosen_type]
            row = st.selectbox("שאלה", shown,
                               format_func=lambda q: f"{q['id']} · {q['question'][:45]}")
            st.caption(f"בודקת: {row['what_it_tests']}")
            if st.button("שאלי את השאלה", type="primary"):
                test_row = row

    if st.button("שיחה חדשה"):
        st.session_state.messages = []
        st.rerun()
    st.caption(usage_guard.report())

st.title("🎓 יועצת הלימודים של שופרא")

# בונים את המנוע כבר עכשיו, כדי שבעיה (אין מפתח, אינדקס ריק) תוצג מיד ולא אחרי השאלה הראשונה.
# הפונקציות ב-rag.py עוצרות במקרים האלה עם sys.exit, וכאן תופסים את זה ומציגים הודעה.
try:
    get_engine()
except SystemExit as e:
    st.error(str(e))
    st.stop()

with st.chat_message("assistant", avatar="🎓"):
    st.markdown(WELCOME)

# כפתורי דוגמה, רק כשהשיחה ריקה
clicked = None
if not st.session_state.messages:
    st.caption("אפשר להתחיל משאלה לדוגמה:")
    cols = st.columns(2)
    for i, example in enumerate(EXAMPLES):
        if cols[i % 2].button(example, use_container_width=True):
            clicked = example

for msg in st.session_state.messages:
    show_message(msg, dev_mode)

# שאלה חדשה: מתיבת הקלט או מכפתור דוגמה
question = st.chat_input("כתבי כאן את שאלתך...") or clicked or (test_row and test_row["question"])
if question:
    # ההיסטוריה נלקחת לפני שמוסיפים את השאלה החדשה.
    # שאלה מבנק הבדיקה נשאלת בלי היסטוריה: כל שאלה בבנק עומדת בפני עצמה, כמו ב---ask.
    history = [] if test_row else recent_history(st.session_state.messages)
    user_msg = {"role": "user", "content": question}
    st.session_state.messages.append(user_msg)
    show_message(user_msg, dev_mode)
    with st.spinner("מחפשת במסמכי המרחב הלימודי..."):
        reply = ask(question, history)
    if test_row:
        reply["test_id"], reply["expected"] = test_row["id"], test_row["expected_answer"]
    st.session_state.messages.append(reply)
    st.rerun()   # מציירים את העמוד מחדש: בלי כפתורי הדוגמה, ועם מונה השימוש המעודכן
