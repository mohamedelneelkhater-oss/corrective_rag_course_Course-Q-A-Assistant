import re
from pathlib import Path

import pandas as pd

BASE_DIR = "/kaggle/working"
DATA_DIR = f"{BASE_DIR}/course_materials"
DB_DIR = f"{BASE_DIR}/chroma_db"
APP_DIR = f"{BASE_DIR}/rag_app"
SAVED_LLM_DIR = f"{BASE_DIR}/best_llm"
CONFIG_PATH = f"{APP_DIR}/best_model.json"
COLLECTION = "course_qa"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
OVERLAP_SENTENCES = 1
SUPPORTED = (".pdf", ".docx", ".csv", ".txt")

_EMBEDDER = None


def get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer(EMBED_MODEL, device="cpu")
    return _EMBEDDER


def get_collection(reset=False):
    import chromadb
    client = chromadb.PersistentClient(path=DB_DIR)
    if reset:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


# ---------------------------------------------------------------- 1) extract
def load_pdf(path):
    from pypdf import PdfReader
    return [{"text": page.extract_text() or "", "part": i + 1} for i, page in enumerate(PdfReader(path).pages)]


def load_docx(path):
    from docx import Document
    sections, title, body = [], None, []
    for para in Document(path).paragraphs:
        text = para.text.strip()
        if not text:
            continue
        if para.style.name.startswith("Heading") or para.style.name == "Title":
            if title and body:
                sections.append(f"{title}. {' '.join(body)}")
            title, body = text, []
        else:
            body.append(text)
    if title and body:
        sections.append(f"{title}. {' '.join(body)}")
    elif body:
        sections.append(" ".join(body))
    return [{"text": s, "part": i + 1} for i, s in enumerate(sections)]


def load_csv(path):
    df = pd.read_csv(path).fillna("")
    cols = {c.lower(): c for c in df.columns}
    if "term" in cols and "definition" in cols:
        ex = cols.get("example")
        texts = [f"{r[cols['term']]}. {r[cols['definition']]}" + (f" Example: {r[ex]}" if ex and r[ex] else "")
                 for _, r in df.iterrows()]
    else:
        texts = [". ".join(f"{c}: {v}" for c, v in r.items()) + "." for _, r in df.iterrows()]
    return [{"text": t, "part": i + 1} for i, t in enumerate(texts)]


def load_txt(path):
    return [{"text": Path(path).read_text(encoding="utf-8", errors="ignore"), "part": 1}]


LOADERS = {".pdf": load_pdf, ".docx": load_docx, ".csv": load_csv, ".txt": load_txt}


def extract_records(path):
    suffix = Path(path).suffix.lower()
    if suffix not in LOADERS:
        raise ValueError(f"Unsupported file type: {suffix}")
    return LOADERS[suffix](str(path))


# ---------------------------------------------------------------- 2) clean
def clean_text(text):
    text = text.replace("\u00a0", " ")
    text = re.sub(r"-\n(?=\w)", "", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    return re.sub(r"\s{2,}", " ", text).strip()


# ---------------------------------------------------------------- 3) split
def chunk_text(text, size=CHUNK_SIZE, overlap=OVERLAP_SENTENCES):
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    chunks, current = [], []
    for sentence in sentences:
        if current and len(" ".join(current + [sentence])) > size:
            chunks.append(" ".join(current))
            current = current[-overlap:] if overlap else []
        current.append(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks


def records_to_chunks(records):
    rows = []
    for rec in records:
        for j, chunk in enumerate(chunk_text(clean_text(rec["text"]))):
            rows.append({"id": f"{rec['course']}|{rec['file']}-{rec['part']}-{j}", "text": chunk, "course": rec["course"],
                         "file": rec["file"], "type": rec["type"], "page": rec["part"], "path": rec["path"]})
    return rows


# ---------------------------------------------------------------- 4) embed + store
def store_chunks(chunks, reset=False, show_progress=False):
    collection = get_collection(reset=reset)
    if not chunks:
        return 0
    texts = [c["text"] for c in chunks]
    vectors = get_embedder().encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=show_progress)
    collection.upsert(
        ids=[c["id"] for c in chunks],
        documents=texts,
        embeddings=vectors.tolist(),
        metadatas=[{k: c[k] for k in ("course", "file", "type", "page", "path")} for c in chunks],
    )
    return len(chunks)


def ingest_file(path, course):
    p = Path(path)
    records = [{**r, "course": course, "file": p.name, "type": p.suffix[1:].upper(), "path": str(p)} for r in extract_records(p)]
    return store_chunks(records_to_chunks(records))


def catalog():
    metas = get_collection().get(include=["metadatas"])["metadatas"]
    if not metas:
        return pd.DataFrame(columns=["course", "file", "type", "path", "chunks"])
    return pd.DataFrame(metas).groupby(["course", "file", "type", "path"]).size().reset_index(name="chunks")
