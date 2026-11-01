"""rag.py: מנוע ה-RAG. האינדקס, השליפה, ובהמשך גם התשובה.

תחום האחריות של הקובץ: כל העבודה מול Gemini ומול מסד הנתונים Chroma.
  שלב 4:        build_index()  הופך כל קטע לווקטור ושומר ב-Chroma.
                update_index() מחשב מחדש רק מסמכים שהשתנו.
                load_index()   פותח את האינדקס השמור, בלי לבנות מחדש.
  שלב 5:        retrieve()      שאלה -> הקטעים הקרובים ביותר עם ציון. בלי מודל שפה.
                evaluate_retrieval() מריץ את כל בנק השאלות ובודק אם המקור הנכון נשלף.
  שלב 6:        build_query_engine() מרכיב שליפה + תבנית (Prompt) + מודל שפה.
                answer()       שאלה -> תשובה בעברית + רשימת מקורות.
  שלב 7:        התבנית בסגנון יועצת לימודים. הממשק עצמו נמצא ב-app.py.
                condense_question() זיכרון שיחה: שאלת המשך -> שאלה שלמה, לפני החיפוש.

הרצה:
    python rag.py            מצב האינדקס (0 קריאות)
    python rag.py --build    בניית האינדקס, פעם אחת (קריאת Embedding לכל קטע)
    python rag.py --update   עדכון חלקי: רק מסמכים שהשתנו (למשל נוהל תשלומים חדש)
    python rag.py --rebuild  מחיקה ובנייה מלאה מחדש, אחרי שינוי בחיתוך או במודל
    python rag.py --retrieve "שאלה"   שליפה לשאלה אחת (קריאת Embedding אחת)
    python rag.py --eval              שליפה לכל 20 השאלות ב-questions.csv (20 קריאות)
    python rag.py --ask "שאלה"        תשובה לשאלה אחת (Embedding אחת + מודל שפה אחת)
    python rag.py --ask "שאלה" --chunks   כנ"ל, ומציג גם את הקטעים שנשלחו למודל
    python rag.py --demo              3 שאלות דוגמה: ישירה, מלכודת גרסה, בלי תשובה (3+3 קריאות)
    python rag.py --demo --style      ועוד 4 שאלות שבודקות את סגנון היועצת (7+7 קריאות)
    python rag.py --followup          שיחה של 3 שאלות המשך, לבדיקת זיכרון השיחה (3 Embedding + 5 מודל שפה)
    לכל אחת מ: --retrieve, --eval, --ask אפשר להוסיף --k 6 כדי לנסות מספר קטעים אחר,
    ו---per-doc 0 כדי לבטל את ההגבלה של קטעים לכל מסמך (להשוואה מול שלב 5)
"""
import ssl_fix  # חייב להיות ראשון: מתקן שגיאת SSL ברשת מסוננת

import csv
import os
import sys
import time

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from llama_index.core import PromptTemplate, VectorStoreIndex
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode
from llama_index.vector_stores.chroma import ChromaVectorStore

import usage_guard
from config import (BOT_INTRO, CHROMA_DIR, CONSULT_MESSAGE, HISTORY_ANSWER_CHARS, HISTORY_TURNS, COLLECTION_NAME, DATA_DIR, EMBED_BATCH_SIZE, EMBED_MODEL,
                    EMBED_PAUSE_SECONDS, CANDIDATE_K, LLM_CONTEXT_WINDOW, MAX_CHUNKS_PER_DOC, LLM_MAX_TOKENS, LLM_MODEL,
                    LLM_TEMPERATURE, NO_INFO_MESSAGE, PROJECT_DIR, QUERY_PAUSE_SECONDS, TOP_K)
from ingest import build_nodes, load_documents

load_dotenv()


# ---------------------------------------------------------------------------
# חיבורים: Chroma ומודל ה-Embedding
# ---------------------------------------------------------------------------

def get_collection(reset: bool = False):
    """פותח (או יוצר) את האוסף ב-Chroma, שנשמר כקבצים בתיקייה storage/chroma."""
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),  # בלי שליחת נתוני שימוש לאינטרנט
    )
    if reset and COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    # cosine: דמיון לפי "הכיוון" של הווקטורים, המדד המקובל לטקסט
    return client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def make_embed_model():
    """מודל ה-Embedding של Gemini. retries=2: מעט ניסיונות חוזרים, כדי לא לבזבז מכסה."""
    from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "paste-your-key-here":
        sys.exit("לא נמצא מפתח. ודאי שקובץ .env קיים ובו GOOGLE_API_KEY.")
    return GoogleGenAIEmbedding(model_name=EMBED_MODEL, api_key=api_key,
                                embed_batch_size=EMBED_BATCH_SIZE, retries=2)


# ---------------------------------------------------------------------------
# שלב 4: בניית האינדקס, עדכון חלקי ופתיחה
# ---------------------------------------------------------------------------

def _embed(nodes) -> None:
    """מחשב וקטור לכל קטע (ומצמיד אותו לקטע). לא שומר כלום: את זה עושים אחרי שהכול הצליח."""
    # בודקים מכסה לפני שמתחילים, כדי לא להיתקע באמצע
    if usage_guard.remaining("embed") < len(nodes):
        sys.exit(f"אין מספיק מכסה היום: צריך {len(nodes)}, נשארו {usage_guard.remaining('embed')}.")

    # קבוצה אחרי קבוצה, עם הפסקה כדי להישאר מתחת למגבלה לדקה
    embed_model = make_embed_model()
    for start in range(0, len(nodes), EMBED_BATCH_SIZE):
        batch = nodes[start:start + EMBED_BATCH_SIZE]
        texts = [n.get_content(metadata_mode=MetadataMode.EMBED) for n in batch]
        usage_guard.register("embed", len(batch))     # רושמים לפני השליחה
        for node, vector in zip(batch, embed_model.get_text_embedding_batch(texts)):
            node.embedding = vector                    # הווקטור "מוצמד" לקטע
        print(f"  {start + len(batch)}/{len(nodes)} קטעים הומרו לווקטורים")
        if start + len(batch) < len(nodes):
            time.sleep(EMBED_PAUSE_SECONDS)


def _store(nodes, collection) -> None:
    """שומר ב-Chroma קטעים שכבר יש להם וקטור. רק כתיבה לדיסק, בלי קריאות."""
    ChromaVectorStore(chroma_collection=collection).add(nodes)


def build_index(rebuild: bool = False) -> int:
    """בנייה מלאה של האינדקס. אם כבר קיים ו-rebuild=False, לא עושה כלום."""
    collection = get_collection()
    if collection.count() > 0 and not rebuild:
        print(f"האינדקס כבר קיים ({collection.count()} קטעים). לא נשלחה אף קריאה.")
        print("אחרי שינוי במסמכים: python rag.py --update")
        return collection.count()

    nodes = build_nodes(load_documents())
    _embed(nodes)                                  # קודם מחשבים...
    collection = get_collection(reset=rebuild)     # ...ורק אז מוחקים את הישן (ב-rebuild)
    _store(nodes, collection)
    return collection.count()


def _stored_fingerprints(collection) -> dict:
    """מה שמור עכשיו באינדקס: מזהה מסמך -> טביעת האצבע שלו בזמן שנשמר."""
    stored = {}
    for meta in collection.get(include=["metadatas"])["metadatas"]:
        stored[meta["document_id"]] = meta.get("doc_hash")  # באינדקס ישן אין doc_hash -> None
    return stored


def update_index() -> int:
    """עדכון חלקי: מחשב מחדש רק מסמכים שהשתנו, נוספו או נמחקו.

    1. משווים את טביעת האצבע של כל מסמך בתיקייה לזו ששמורה באינדקס.
    2. למסמך שהשתנה או נוסף: חותכים ומחשבים Embedding רק לקטעים שלו.
    3. מוחקים מ-Chroma את הקטעים הישנים של המסמכים האלה, ושומרים את החדשים.
    """
    collection = get_collection()
    if collection.count() == 0:
        print("האינדקס ריק, אז בונים אותו מאפס.")
        return build_index()

    documents = load_documents()
    current = {d.id_: d.metadata["doc_hash"] for d in documents}
    stored = _stored_fingerprints(collection)

    changed = [d for d in documents if stored.get(d.id_) != current[d.id_]]
    removed = [doc_id for doc_id in stored if doc_id not in current]

    if not changed and not removed:
        print("שום מסמך לא השתנה. לא נשלחה אף קריאה.")
        return collection.count()

    for d in changed:
        print(f"  {'חדש' if d.id_ not in stored else 'השתנה'}: {d.id_} ({d.metadata['title']})")
    for doc_id in removed:
        print(f"  נמחק: {doc_id}")

    new_nodes = build_nodes(changed)
    print(f"\nמחשבים Embedding ל-{len(new_nodes)} קטעים בלבד.")
    _embed(new_nodes)        # קודם מחשבים: אם משהו נכשל כאן, האינדקס הישן נשאר שלם

    old_ids = [d.id_ for d in changed] + removed
    collection.delete(where={"document_id": {"$in": old_ids}})   # הקטעים הישנים של המסמכים האלה
    _store(new_nodes, collection)                               # והחדשים במקומם
    return collection.count()


def load_index() -> VectorStoreIndex:
    """פותח את האינדקס השמור (בשביל שלב 5). לא שולח קריאות."""
    collection = get_collection()
    if collection.count() == 0:
        sys.exit("האינדקס ריק. הריצי קודם: python rag.py --build")
    return VectorStoreIndex.from_vector_store(ChromaVectorStore(chroma_collection=collection),
                                              embed_model=make_embed_model())


# ---------------------------------------------------------------------------
# שלב 5: שליפה (Retrieval), בלי מודל שפה
# ---------------------------------------------------------------------------

QUESTIONS_FILE = DATA_DIR / "questions.csv"
REPORT_FILE = PROJECT_DIR / "retrieval_report.csv"


class PerDocumentLimit(BaseNodePostprocessor):
    """מסננת אחרי השליפה: לכל היותר max_per_doc קטעים מכל מסמך, ובסך הכול top_k.

    הבעיה שהיא פותרת: כשמסמך אחד ארוך ודומה מאוד לשאלה, כל הקטעים מגיעים ממנו,
    ומסמך אחר שמחזיק חלק מהתשובה (למשל דמי רישום בנוהל) לא נכנס בכלל.
    הקטעים מגיעים כבר ממוינים מהציון הגבוה לנמוך, אז "דילוג" על קטע עודף
    פשוט מפנה מקום לקטע הבא בתור ממסמך אחר.
    """
    max_per_doc: int = MAX_CHUNKS_PER_DOC
    top_k: int = TOP_K

    def _postprocess_nodes(self, nodes, query_bundle=None):
        per_doc, kept = {}, []
        for n in nodes:
            doc_id = n.node.metadata["document_id"]
            if self.max_per_doc and per_doc.get(doc_id, 0) >= self.max_per_doc:
                continue                                  # כבר יש מספיק מהמסמך הזה
            per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
            kept.append(n)
            if len(kept) == self.top_k:
                break
        return kept


def _candidates(top_k: int, per_doc: int) -> int:
    """כמה מועמדים לשלוף לפני הסינון. בלי הגבלה למסמך אין צורך במועמדים נוספים."""
    return max(CANDIDATE_K, top_k * 3) if per_doc else top_k


def retrieve(question: str, top_k: int = TOP_K, index: VectorStoreIndex | None = None,
             per_doc: int = MAX_CHUNKS_PER_DOC):
    """שאלה -> top_k הקטעים הקרובים ביותר אליה, כל אחד עם ציון דמיון.

    מה קורה בפנים:
    1. השאלה נשלחת ל-Gemini והופכת לווקטור (קריאת Embedding אחת).
    2. Chroma משווה את הווקטור הזה לכל 58 הווקטורים השמורים (במחשב, בלי עלות)
       ומחזירה את 12 המועמדים הקרובים ביותר.
    3. PerDocumentLimit בוחרת מהם top_k, עם לכל היותר per_doc קטעים מכל מסמך.
    """
    index = index or load_index()
    usage_guard.register("embed")                        # שאלה = קריאת Embedding אחת
    retriever = index.as_retriever(similarity_top_k=_candidates(top_k, per_doc))
    candidates = retriever.retrieve(question)            # רשימת NodeWithScore: קטע + ציון
    return PerDocumentLimit(max_per_doc=per_doc, top_k=top_k).postprocess_nodes(candidates)


def print_results(question: str, results) -> None:
    """מדפיס את מה שנשלף: דירוג, ציון, מסמך מקור ותחילת הקטע."""
    print(f"\nשאלה: {question}")
    for rank, r in enumerate(results, start=1):
        meta = r.node.metadata
        snippet = r.node.get_content().replace("\n", " ")[:110]
        print(f"  {rank}. ציון {r.score:.3f}  [{meta['document_id']}] {meta['title']}")
        print(f"     {snippet}...")


def _expected_files(source_files: str) -> set[str]:
    """עמודת source_files בבנק השאלות -> קבוצת נתיבי קבצים.

    "courses/finance.md; tables/prices_schedule.csv" -> {"courses/finance.md", "tables/prices_schedule.csv"}
    "(אין)" -> קבוצה ריקה: שאלה שאין לה תשובה במאגר.
    """
    files = set()
    for part in source_files.split(";"):
        path = part.split("(")[0].strip()      # מורידים הערות כמו "(מידע חלקי בלבד)"
        if path.endswith((".md", ".csv")):
            files.add(path)
    return files


def evaluate_retrieval(top_k: int = TOP_K, per_doc: int = MAX_CHUNKS_PER_DOC) -> None:
    """מריץ שליפה על כל בנק השאלות ובודק לכל שאלה: האם מסמך המקור הנכון נשלף?

    "פגיעה" (hit) = לפחות אחד מהמסמכים הצפויים נמצא בין top_k הקטעים.
    "כיסוי מלא" = כל המסמכים הצפויים נמצאו (חשוב בשאלות שמשלבות שני מסמכים).
    התוצאות נשמרות גם ב-retrieval_report.csv, בשביל יומן השיפורים של שלב 8.
    """
    with open(QUESTIONS_FILE, encoding="utf-8-sig", newline="") as f:
        questions = list(csv.DictReader(f))

    if usage_guard.remaining("embed") < len(questions):
        sys.exit(f"אין מספיק מכסה היום: צריך {len(questions)}, נשארו {usage_guard.remaining('embed')}.")

    index = load_index()                     # פותחים פעם אחת, לא לכל שאלה
    rows, hits, full, answerable = [], 0, 0, 0
    for i, q in enumerate(questions):
        results = retrieve(q["question"], top_k, index, per_doc)
        print_results(f"{q['id']} ({q['question_type']}) {q['question']}", results)

        expected = _expected_files(q["source_files"])
        found = {r.node.metadata["file_path"] for r in results}
        retrieved_ids = [r.node.metadata["document_id"] for r in results]

        if expected:
            answerable += 1
            hit = bool(expected & found)
            hits += hit
            full += expected <= found
            missing = sorted(expected - found)
            verdict = "נשלף" if hit else "לא נשלף!"
            print(f"  -> המקור הצפוי {verdict}" + (f"  (חסר: {', '.join(missing)})" if hit and missing else ""))
        else:
            hit, missing = None, []
            print(f"  -> אין תשובה במאגר. הציון הגבוה ביותר: {results[0].score:.3f}")

        if "GEN-FAQ" in retrieved_ids and "general/faq_2025.md" not in expected:
            print("  שימי לב: נשלף גם מסמך השאלות הנפוצות הישן (2025)")

        rows.append({
            "id": q["id"], "question_type": q["question_type"], "question": q["question"],
            "hit": {True: "כן", False: "לא", None: "אין תשובה במאגר"}[hit],
            "missing": "; ".join(missing),
            "retrieved": "; ".join(f"{d} ({r.score:.3f})" for d, r in zip(retrieved_ids, results)),
        })
        if i < len(questions) - 1:
            time.sleep(QUERY_PAUSE_SECONDS)

    # utf-8-sig: כדי שאקסל יפתח את העברית נכון
    with open(REPORT_FILE, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 60)
    print(f"סיכום (top_k={top_k}, per_doc={per_doc or 'ללא הגבלה'}): המקור הנכון נשלף ב-{hits} מתוך {answerable} שאלות שיש להן תשובה")
    print(f"כל המקורות הצפויים נשלפו ב-{full} מתוך {answerable}")
    print(f"הדוח המלא נשמר ב-{REPORT_FILE.name}")


# ---------------------------------------------------------------------------
# שלב 6: יצירת תשובה (Generation) עם מקורות
# ---------------------------------------------------------------------------

# התבנית (Prompt) שנשלחת למודל השפה בכל שאלה.
# {context_str} ו-{query_str} הם "חורים" ש-LlamaIndex ממלא בעצמו:
#   context_str = הקטעים שנשלפו (עם הכותרת, הגרסה ותאריך העדכון של כל מסמך)
#   query_str   = השאלה של המשתמשת
# {no_info} ו-{consult} אנחנו ממלאים מראש (partial_format למטה), מתוך config.py.
# שלב 7: התבנית עברה לסגנון יועצת לימודים. כללי התוכן (1–10) קודמים, וכללי הסגנון (11–14)
# מוגבלים במפורש: בלי משפטי עידוד כלליים, יתרון רק אם הוא כתוב בקטעים, הזמנה רק בהתלבטות,
# ובלי הבטחות או לחץ. (גרסה 2, אחרי הרצת --demo --style: התשובות חזרו על אותו דפוס.)
QA_PROMPT_TMPL = """את יועצת הלימודים של "שופרא – למידה שעובדת", מרחב לימודי לקורסים מקצועיים.
תפקידך לענות לסטודנטיות ולמתעניינות על הקורסים, המחירים, המועדים, הנהלים והתקנון,
ולעזור להן להרגיש בטוחות בבחירה שלהן.

להלן קטעים מתוך מסמכי המרחב הלימודי. לפני כל קטע מופיעים פרטי המסמך שממנו נלקח.
---------------------
{context_str}
---------------------

כללי תוכן (הם קודמים לכל כלל אחר):
1. עני אך ורק לפי הקטעים שלמעלה. אל תוסיפי מידע מהידע הכללי שלך ואל תנחשי.
2. אם אין בקטעים שום מידע שעונה על השאלה, כתבי בדיוק: "{no_info}", ובשורה נוספת הציעי לפנות למזכירות המרחב הלימודי. בתשובה כזו אין משפט יתרון ואין הזמנה (כללים 12–13).
   אם יש בקטעים מידע שעונה על חלק מהשאלה (למשל הכתובת, בשאלה "איך מגיעים"), כתבי את המידע הזה, ואחריו משפט קצר שאומר מה אין במסמכים והפניה למזכירות. בתשובה חלקית אל תכתבי את המשפט "{no_info}".
   אם השאלה מבקשת הערכה או דעה (למשל "זה משתלם?", "כדאי לי?", "זה קשה?"), אל תכתבי "{no_info}" ואל תחווי דעה משלך. הציגי בקצרה את העובדות מהקטעים שעוזרות להחליט (למשל מחיר, הנחות ומלגות, היקף, תעודה, ליווי בהשמה), אמרי שההחלטה אישית, וציייני מה אין במסמכים (למשל שכר או סיכויי תעסוקה). בשאלה כזו מותר להזמין לשיחת ייעוץ.
3. אם יש סתירה בין מסמכים, הסתמכי על המסמך שתאריך העדכון שלו (updated_at) מאוחר יותר.
4. העתיקי מספרים, תאריכים ושעות בדיוק כפי שהם כתובים בקטעים. כתבי סכומי כסף עם פסיק וסימן ₪, למשל 4,800 ₪, ופעם אחת בלבד.
5. עני רק על מה שנשאל. אל תוסיפי פרטים שלא התבקשו, כמו אופן הגשת בקשה או אגרות חיצוניות. החריגים היחידים: בשאלה על מחיר קורס ציני גם את דמי הרישום אם הם מופיעים בקטעים, ומשפט היתרון וההזמנה לפי כללים 12–13.
6. בשאלה על מחיר שיש בה הנחה או כמה תשלומים: הראי את החישוב צעד אחר צעד, וסיימי בשורה "סה"כ לתשלום: ... ₪".
7. עני בעברית, בלשון פנייה לנקבה, בקצרה ובבהירות.
8. כני את המוסד "מרחב לימודי" ולא "מכללה".
9. אם בקטעים מופיעה רשימת הקורסים של המרחב הלימודי, ושואלים על קורס או תחום שלא נמצא בה, אל תכתבי "{no_info}". עני שהקורס הזה לא נלמד במרחב הלימודי, ופרטי את הקורסים שכן מופיעים ברשימה.
10. אם השאלה היא עלייך, היועצת (למשל "מה את יודעת לעשות?", "מי את?", "במה את יכולה לעזור?"), ולא על המרחב הלימודי, כתבי בדיוק: "{bot_intro}" ואל תוסיפי דבר.

כללי סגנון: יועצת לימודים
11. החום מגיע מהניסוח של התשובה עצמה: פנייה אישית ומשפטים נעימים. אל תכתבי משפטי עידוד או מחמאה כלליים, כמו "נפלא שאת בוחרת...", "אני כאן כדי לעזור לך" או "באהבה ובהצלחה".
12. משפט יתרון: רק בשאלה על קורס מסוים (תוכן, מחיר, תעודה, מועדים), ורק אם בקטעים כתוב יתרון מוחשי של הקורס או של המרחב הלימודי, למשל תעודה מקצועית, לימוד מעשי, תיק עבודות או ליווי בהשמה. משפט אחד, שמציין את העובדה עצמה, ושאינו חוזר על מה שכבר נכתב בתשובה. אם אין יתרון כזה בקטעים, או שהוא כבר נאמר בתשובה, אל תכתבי משפט כזה. בשאלות על נהלים (ביטול, נוכחות, ערעור, מלגה) אין משפט יתרון.
13. הזמנה לשיחת ייעוץ: רק כשהשאלה מראה התלבטות או בחירה בין קורסים, כששואלים אם קורס מתאים לה, או כשמבקשים לדבר עם מישהי. אז סיימי בשורה נפרדת עם ההזמנה הזו, מילה במילה: "{consult}". בכל שאר השאלות אל תוסיפי אותה.
14. אסור להבטיח מה שלא כתוב בקטעים (למשל עבודה, שכר, הנחה או מקום פנוי), ואסור ללחוץ ("מהרי", "מקומות אחרונים", "הזדמנות שלא תחזור").

שאלה: {query_str}
תשובה:"""

QA_PROMPT = PromptTemplate(QA_PROMPT_TMPL).partial_format(no_info=NO_INFO_MESSAGE, consult=CONSULT_MESSAGE,
                                                       bot_intro=BOT_INTRO)


def make_llm():
    """מודל השפה Gemini Flash-Lite, שכותב את התשובה מתוך הקטעים."""
    from llama_index.llms.google_genai import GoogleGenAI

    return GoogleGenAI(
        model=LLM_MODEL,
        api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=LLM_TEMPERATURE,        # נמוך: דיוק ועקביות, לא יצירתיות
        max_tokens=LLM_MAX_TOKENS,          # גבול לאורך התשובה
        context_window=LLM_CONTEXT_WINDOW,  # עם שני אלה LlamaIndex לא שואל את Google על המודל
        max_retries=2,                      # מעט ניסיונות חוזרים, כדי לא לבזבז מכסה
    )


def build_query_engine(index: VectorStoreIndex | None = None, top_k: int = TOP_K,
                       per_doc: int = MAX_CHUNKS_PER_DOC):
    """מרכיב את "מנוע השאלות" (Query Engine): שליפה + תבנית + מודל שפה, בצינור אחד.

    בכל שאלה המנוע עושה שלושה דברים:
    1. Retriever: שולף מועמדים, ו-PerDocumentLimit בוחרת מהם top_k (בדיוק כמו retrieve()).
    2. Response Synthesizer: מכניס אותם ל-QA_PROMPT יחד עם השאלה.
    3. LLM: שולח את הטקסט המלא ל-Gemini ומקבל תשובה.
    compact = כל הקטעים נדחסים לבקשה אחת, כלומר קריאת מודל שפה אחת לשאלה.
    """
    index = index or load_index()
    return index.as_query_engine(
        llm=make_llm(),
        similarity_top_k=_candidates(top_k, per_doc),
        node_postprocessors=[PerDocumentLimit(max_per_doc=per_doc, top_k=top_k)],
        text_qa_template=QA_PROMPT,
        response_mode="compact",
    )


def _sources(source_nodes) -> list[dict]:
    """הקטעים שנשלפו -> רשימת מקורות, מסמך אחד פעם אחת בלבד (עם הציון הגבוה שלו)."""
    best = {}
    for r in source_nodes:
        meta = r.node.metadata
        doc_id = meta["document_id"]
        if doc_id not in best or r.score > best[doc_id]["score"]:
            best[doc_id] = {"document_id": doc_id, "title": meta["title"],
                            "file_path": meta["file_path"], "updated_at": meta["updated_at"],
                            "score": r.score}
    return sorted(best.values(), key=lambda s: s["score"], reverse=True)


# ---------------------------------------------------------------------------
# זיכרון שיחה: ניסוח מחדש של שאלת המשך (Condense Question)
# ---------------------------------------------------------------------------

# הבעיה: החיפוש (Embedding) רואה רק את השאלה הנוכחית. בשאלה "וכמה הוא עולה?"
# אין שום מילה שמצביעה על קורס, ולכן נשלפים קטעים אקראיים על מחירים.
# הפתרון: לפני החיפוש מבקשים מהמודל לנסח את השאלה מחדש כשאלה שלמה,
# בעזרת השאלות הקודמות: "כמה עולה קורס עיצוב גרפי – יסודות?"
# מכאן הכול ממשיך בדיוק כמו קודם: אותה שליפה, אותה תבנית, אותם כללים.
CONDENSE_PROMPT_TMPL = """להלן שיחה בין מתעניינת לבין יועצת הלימודים של מרחב לימודי, ואחריה שאלה חדשה של המתעניינת.

השיחה הקודמת:
{history}

השאלה החדשה: {question}

נסחי את השאלה החדשה כשאלה אחת שלמה, שאפשר להבין בלי השיחה הקודמת.
- השלימי מהשיחה רק את מה שחסר כדי להבין את השאלה, למשל שם הקורס שעליו מדובר.
- אם השאלה החדשה כבר מובנת לבד, או שהיא על נושא חדש, החזירי אותה בדיוק כמו שהיא.
- אל תעני על השאלה ואל תוסיפי שום הסבר. החזירי רק את השאלה המנוסחת, בשורה אחת.
"""

_condense_llm = None   # נוצר בפעם הראשונה שצריך אותו


def _format_history(history: list[dict]) -> str:
    """רשימת שאלות ותשובות קודמות -> טקסט לתבנית. רק HISTORY_TURNS האחרונות, ותשובות מקוצרות."""
    lines = []
    for turn in history[-HISTORY_TURNS:]:
        short = turn["answer"].replace("\n", " ")[:HISTORY_ANSWER_CHARS]
        lines.append(f"מתעניינת: {turn['question']}\nיועצת: {short}")
    return "\n\n".join(lines)


def condense_question(question: str, history: list[dict]) -> str:
    """שאלת המשך -> שאלה עצמאית. בלי היסטוריה מחזיר את השאלה כמו שהיא, בלי קריאה."""
    if not history:
        return question
    global _condense_llm
    _condense_llm = _condense_llm or make_llm()
    usage_guard.register("llm")                          # הניסוח מחדש = קריאת מודל שפה אחת
    prompt = CONDENSE_PROMPT_TMPL.format(history=_format_history(history), question=question)
    text = _condense_llm.complete(prompt).text.strip()
    # ניקוי: לפעמים המודל מוסיף "השאלה:" או מירכאות
    text = text.splitlines()[0].removeprefix("השאלה המנוסחת:").removeprefix("השאלה:").strip().strip('"').strip()
    return text or question


def answer(question: str, engine=None, history: list[dict] | None = None) -> dict:
    """שאלה -> תשובה + מקורות. זו הפונקציה שהממשק (שלב 7) קורא לה.

    history = השאלות והתשובות הקודמות בשיחה: [{"question": ..., "answer": ...}, ...]
    אם יש היסטוריה, השאלה מנוסחת קודם מחדש כשאלה שלמה (condense_question).

    מחזירה מילון:
      answer  = טקסט התשובה
      found   = False אם המודל כתב "לא נמצא מידע", אחרת True
      sources = המסמכים שהתשובה מבוססת עליהם (ריק כשלא נמצא מידע)
    """
    # כל שאלה עולה שתי קריאות: Embedding לשאלה + מודל שפה לתשובה. שאלה עם היסטוריה: עוד קריאת מודל שפה.
    llm_needed = 2 if history else 1
    if usage_guard.remaining("embed") < 1 or usage_guard.remaining("llm") < llm_needed:
        # QuotaExceeded ולא sys.exit: בשורת הפקודה זה עוצר באותה צורה, ובממשק אפשר לתפוס את זה
        raise usage_guard.QuotaExceeded("אין מספיק מכסה היום. " + usage_guard.report())
    engine = engine or build_query_engine()
    standalone = condense_question(question, history or [])
    usage_guard.register("embed")
    usage_guard.register("llm")

    response = engine.query(standalone)     # מחפשים ועונים לפי השאלה השלמה
    text = (response.response or "").strip()
    # "לא נמצא" רק כשהתשובה *מתחילה* במשפט הקבוע. תשובה חלקית (כלל 2) מציגה מידע ומקורות.
    found = not text.startswith(NO_INFO_MESSAGE)
    # שאלה על היועצת עצמה: תשובה קבועה, לא מהמסמכים, ולכן גם בלי מקורות
    about_bot = text.startswith(BOT_INTRO[:30])
    return {
        "question": question,
        "standalone": standalone,             # השאלה כפי שנשלחה לחיפוש (שונה רק בשאלת המשך)
        "answer": text or NO_INFO_MESSAGE,
        "found": found and bool(text),
        # כשאין מידע לא מציגים מקורות: הם רק היו מטעים, כאילו התשובה מבוססת עליהם
        "sources": _sources(response.source_nodes) if found and text and not about_bot else [],
        "retrieved": response.source_nodes,   # כל מה שנשלף, לבדיקה ולשלב 8
    }


def print_answer(result: dict, show_chunks: bool = False) -> None:
    """מדפיס תשובה ומקורות בצורה קריאה."""
    print(f"\nשאלה: {result['question']}")
    if result.get("standalone") and result["standalone"] != result["question"]:
        print(f"(נוסחה מחדש: {result['standalone']})")
    print("-" * 60)
    print(result["answer"])
    if result["sources"]:
        print("\nמקורות:")
        for s in result["sources"]:
            print(f"  • {s['title']} ({s['file_path']}, עודכן {s['updated_at']})  ציון {s['score']:.3f}")
    else:
        print("\n(לא מוצגים מקורות: לא נמצא מידע במסמכים)")
    if show_chunks:
        print("\nהקטעים שנשלחו למודל:")
        for r in result["retrieved"]:
            print(f"  [{r.node.metadata['document_id']}] {r.score:.3f}  "
                  f"{r.node.get_content().replace(chr(10), ' ')[:90]}...")


# שלוש שאלות מייצגות מבנק השאלות: ישירה (משני מסמכים), מלכודת גרסה, ושאלה בלי תשובה
DEMO_IDS = ["Q01", "Q03", "Q19"]
# שלב 7: שאלות שבודקות את סגנון היועצת. האם התשובה נשארת עניינית ולא "מוכרת" יותר מדי?
#   Q11 (תעודה): שאלה על קורס מסוים. כאן מותר משפט יתרון אחד, עם עובדה מהקטעים.
#   Q12 (זום): התשובה היא "לא". היועצת צריכה לענות "לא" ברור, בלי לייפות.
#   Q14 (שיחת ייעוץ): כאן ההזמנה היא התשובה עצמה.
#   Q18 (קורס שלא קיים): מלכודת שכנוע. אסור "להמציא" קורס או ללחוץ לעבור לקורס אחר.
STYLE_DEMO_IDS = ["Q11", "Q12", "Q14", "Q18"]


def demo(ids: list[str] = DEMO_IDS) -> None:
    """מריץ את שאלות הדוגמה. כל שאלה = קריאת Embedding אחת + קריאת מודל שפה אחת."""
    with open(QUESTIONS_FILE, encoding="utf-8-sig", newline="") as f:
        questions = {q["id"]: q for q in csv.DictReader(f)}
    engine = build_query_engine()                 # בונים פעם אחת לכל השאלות
    for i, qid in enumerate(ids):
        q = questions[qid]
        result = answer(q["question"], engine)
        print(f"\n===== {qid} ({q['question_type']}) =====")
        print_answer(result)
        print(f"\nהתשובה הצפויה (מבנק השאלות): {q['expected_answer']}")
        if i < len(ids) - 1:
            time.sleep(QUERY_PAUSE_SECONDS)


# שיחה לבדיקת הזיכרון: השאלה השנייה והשלישית לא מובנות בלי הקודמות
FOLLOWUP_QUESTIONS = [
    "מתי מתחיל קורס עיצוב גרפי יסודות?",
    "וכמה הוא עולה?",
    "ואיזו תעודה מקבלים בסופו?",
]


def followup_demo() -> None:
    """מריץ שיחה של 3 שאלות המשך: 3 קריאות Embedding + 5 קריאות מודל שפה (2 ניסוחים מחדש)."""
    engine = build_query_engine()
    history = []
    for i, q in enumerate(FOLLOWUP_QUESTIONS):
        result = answer(q, engine, history)
        print(f"\n===== שאלה {i + 1} בשיחה =====")
        print_answer(result)
        history.append({"question": q, "answer": result["answer"]})
        if i < len(FOLLOWUP_QUESTIONS) - 1:
            time.sleep(QUERY_PAUSE_SECONDS)


# ---------------------------------------------------------------------------
# הרצה ישירה
# ---------------------------------------------------------------------------

def _arg_value(flag: str, default=None):
    """הערך שבא אחרי דגל בשורת הפקודה, למשל --k 6 -> "6"."""
    if flag in sys.argv and sys.argv.index(flag) + 1 < len(sys.argv):
        return sys.argv[sys.argv.index(flag) + 1]
    return default


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    k = int(_arg_value("--k", TOP_K))
    per_doc = int(_arg_value("--per-doc", MAX_CHUNKS_PER_DOC))

    if "--retrieve" in sys.argv:
        question = _arg_value("--retrieve")
        if not question:
            sys.exit('כתבי שאלה במירכאות, למשל: python rag.py --retrieve "כמה עולה קורס הנהלת חשבונות?"')
        print_results(question, retrieve(question, k, per_doc=per_doc))
        print("\n" + usage_guard.report())
        sys.exit()
    if "--ask" in sys.argv:
        question = _arg_value("--ask")
        if not question:
            sys.exit('כתבי שאלה במירכאות, למשל: python rag.py --ask "כמה עולים דמי הרישום?"')
        print_answer(answer(question, build_query_engine(top_k=k, per_doc=per_doc)), show_chunks="--chunks" in sys.argv)
        print("\n" + usage_guard.report())
        sys.exit()
    if "--demo" in sys.argv:
        demo(DEMO_IDS + STYLE_DEMO_IDS if "--style" in sys.argv else DEMO_IDS)
        print("\n" + usage_guard.report())
        sys.exit()
    if "--followup" in sys.argv:
        followup_demo()
        print("\n" + usage_guard.report())
        sys.exit()
    if "--eval" in sys.argv:
        evaluate_retrieval(k, per_doc)
        print(usage_guard.report())
        sys.exit()

    if "--update" in sys.argv:
        count = update_index()
        print(f"\nבאינדקס {count} קטעים, שמורים ב-{CHROMA_DIR}")
    elif "--build" in sys.argv or "--rebuild" in sys.argv:
        count = build_index(rebuild="--rebuild" in sys.argv)
        print(f"\nבאינדקס {count} קטעים, שמורים ב-{CHROMA_DIR}")
    else:
        count = get_collection().count()
        print(f"באינדקס {count} קטעים." if count else "האינדקס עוד לא נבנה. להרצה: python rag.py --build")

    # הצצה לרשומה אחת: כך נראה קטע בתוך Chroma
    collection = get_collection()
    if collection.count():
        sample = collection.get(limit=1, include=["embeddings", "metadatas", "documents"])
        meta = {k: v for k, v in sample["metadatas"][0].items() if not k.startswith("_")}
        print("\nרשומה לדוגמה מתוך Chroma:")
        print(f"  וקטור באורך {len(sample['embeddings'][0])}, מתחיל ב-",
              [round(float(x), 4) for x in sample["embeddings"][0][:4]])
        print("  מטא-דאטה:", meta)
        print("  טקסט:", sample["documents"][0][:100].replace("\n", " "), "...")
    print(usage_guard.report())
