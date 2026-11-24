# Corrective RAG System for Course Q&A (Lightweight, CPU-Friendly)

An end-to-end, lightweight, CPU-friendly Corrective Retrieval-Augmented Generation (CRAG) assistant built to answer student questions across multi-course academic materials (PDF, DOCX, CSV, and TXT).

This repository merges two complete RAG architectures into one unified system:
1. **Standard Course Q&A RAG:** Multi-format ingestion, semantic embedding, persistent vector storage, and grounded question answering with source attribution.
2. **Corrective RAG (CRAG):** Pre-generation chunk relevance grading, automatic query rewriting with bounded retries when context is weak, verified-context-only answer generation, and strict refusal handling for out-of-scope questions.

**Kaggle Notebook & Dataset:** Run the full reproducible pipeline, inspect evaluation charts, or access the generated course materials dataset directly on Kaggle: [🔗 Open ](https://www.kaggle.com/code/mohamedelneel/corrective-rag-course-course-q-a-assistant)on Kaggle](#)

---

## System Architecture & Pipeline Flow

```text
User Question (+ Optional Course Filter)
  │
  ▼
[1. Retrieve Top-k Chunks] ──► ChromaDB (Cosine Similarity, all-MiniLM-L6-v2)
  │
  ▼
[2. Relevance Grader] ──► Deterministic LLM Grading (yes / no, temp = 0)
  │                       (Skips low-similarity hits < 0.20; stops after 3 accepted)
  ├─── Enough Verified Chunks? ──► YES ──► [4. Generate Answer (Verified Context Only)]
  │
  └─── Weak / Irrelevant? ──► NO ──► [3. Query Rewriter (Max 2 Retries)]
                                         │
  ┌──────────────────────────────────────┘
  ▼
Still No Verified Context? ──► Clear Refusal Message (Zero Hallucination)
```

---

## Key Features

- **Multi-Format Document Ingestion (`ingestion.py`):**
  - **PDF (`pypdf`):** Page-level text extraction preserving page metadata.
  - **DOCX (`python-docx`):** Heading- and section-aware extraction preserving logical document structure.
  - **CSV (`pandas`):** Schema-aware row serialization (supports `term`, `definition`, `example` glossaries as well as generic tabular CSVs).
  - **TXT (`pathlib`):** UTF-8 full-text extraction.
- **Text Cleaning & Sentence-Aware Chunking:** Normalizes non-breaking spaces, fixes hyphenated line breaks, and packs sentences into chunks of up to 500 characters with 1-sentence overlap.
- **Idempotent Storage (ChromaDB):** Encodes chunks into 384-dimensional normalized vectors using `sentence-transformers/all-MiniLM-L6-v2` (~22M parameters) and stores them via `upsert` to prevent duplicate entries.
- **Corrective RAG Engine (`rag_core.py`):**
  - Supports three lightweight CPU models (`flan-t5-small`, `flan-t5-base`, and `qwen2.5-0.5b-instruct`) with automatic local loading from `best_llm/` when saved.
  - Deterministic relevance grading (`temperature = 0`), similarity thresholding (`GRADE_MIN_SCORE = 0.20`), and capped acceptance (`MAX_ACCEPT = 3`).
  - LLM-powered query rewriting with a stopword-filtered keyword fallback and a hard cap of 2 rewrites to eliminate infinite loops.
  - Extractive sentence fallback if generative decoding produces malformed output, plus strict refusal enforcement (`UNANSWERED`) when no verified context exists.
  - Full step-by-step execution trace (`Retrieve`, `Grade chunks`, `Rewrite query`, `Decision`).
- **Interactive Chainlit Web App (`rag_app/app.py`):**
  - **Settings Gear Panel:** Switch LLM models on the fly, toggle Corrective RAG ON/OFF, adjust `max_rewrites` (0–2) and `top_k` (3–8), and filter queries by course.
  - **Live Document Upload:** Attach custom PDF, DOCX, CSV, or TXT files directly in the chat to ingest, preview chunks, and query immediately under `Uploaded Documents`.
  - **Collapsible Debug Trace & Source Panels:** Inspect every retrieval, grading, and rewrite decision inline, and click cited chunks to read full excerpts in side panels.
  - **Batch Evaluation Runner:** Run categorized question banks (`Machine Learning`, `Python`, `SQL`, `Vague`, `Unanswerable` — 61 questions total) from the UI and export accuracy charts and CSV reports.

---

## Knowledge Base Overview

The repository includes a reproducible multi-course dataset (`course_materials/`) spanning 3 courses and 4 file formats (26 raw records split into 29 chunks, averaging 235 characters/chunk):

| Course | File Name | Format | Records | Stored Chunks |
| :--- | :--- | :--- | :--- | :--- |
| Machine Learning | `ml_lecture_notes.pdf` | PDF | 3 pages (6 topics) | 6 |
| Machine Learning | `ml_syllabus.txt` | TXT | 1 file | 1 |
| Python Programming | `python_lectures.docx` | DOCX | 6 sections | 6 |
| Python Programming | `python_syllabus.txt` | TXT | 1 file | 1 |
| SQL and Databases | `sql_glossary.csv` | CSV | 14 rows | 14 |
| SQL and Databases | `sql_syllabus.txt` | TXT | 1 file | 1 |

---

## Benchmark & Evaluation Results

All three candidate models were evaluated on a curated 15-question test set (`test_set.csv`: 11 direct questions, 1 vague question requiring query rewriting, and 3 unanswerable questions testing hallucination control) across both Standard and Corrective modes on CPU:

| Model | Mode | Quality Score | Context Hit | Refusal Rate | Avg Latency (s) | Avg Rewrites | Final Score* |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `flan-t5-small` (~77M) | Corrective | 0.700 | 1.000 | 1.000 | 1.048 | 0.400 | **0.760** |
| `flan-t5-small` (~77M) | Standard | 0.533 | 1.000 | 0.333 | 1.061 | 0.000 | — |
| `flan-t5-base` (~250M) | Corrective | 0.800 | 0.917 | 1.000 | 3.287 | 0.533 | 0.704 |
| `flan-t5-base` (~250M) | Standard | 0.733 | 1.000 | 0.333 | 1.895 | 0.000 | — |
| `qwen2.5-0.5b-instruct` (~500M) | Corrective | 0.733 | 1.000 | 0.333 | 10.059 | 0.133 | 0.607 |
| `qwen2.5-0.5b-instruct` (~500M) | Standard | 0.733 | 1.000 | 0.333 | 5.103 | 0.000 | — |

*\*Final Score = `0.8 × Quality + 0.2 × Speed Score` (computed on Corrective runs).*

### Key Takeaways
- **100% Hallucination Control:** Enabling Corrective RAG raised the unanswerable-question refusal rate from **33.3% to 100%** for both `flan-t5-small` and `flan-t5-base`.
- **Quality Boost:** Corrective grading and query rewriting improved overall answer quality by **+16.7%** on `flan-t5-small` (`0.533 → 0.700`) and **+6.7%** on `flan-t5-base` (`0.733 → 0.800`).
- **Best Speed/Accuracy Trade-off:** `flan-t5-small` achieved the highest composite score (**0.760**) with ~1.05s average CPU latency and was selected as the default deployment model.

---

## Repository Structure

```text
corrective-rag-course-qa/
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
├── .gitignore                 # Ignored caches, local DBs, logs, and weights
├── ingestion.py               # Multi-format extraction, cleaning, chunking & ChromaDB storage
├── rag_core.py                # Retrieval, LLM chunk grading, query rewriting & generation
├── rag_app/
│   ├── app.py                 # Chainlit web interface, settings, upload & batch runner
│   └── best_model.json        # Selected default model & evaluation ranking metadata
├── notebooks/
│   └── corrective-rag-course-course-q-a-assistant.ipynb # End-to-end Kaggle notebook
└── data/
    ├── course_materials/      # Sample course files (PDF, DOCX, CSV, TXT)
    ├── test_set.csv           # 15-question benchmark dataset
    └── evaluation_details.csv # Full per-question evaluation logs (90 runs)
```

---

## Installation & Local Setup

### 1. Clone the Repository & Install Dependencies
```bash
git clone https://github.com/<your-username>/corrective-rag-course-qa.git
cd corrective-rag-course-qa
pip install -r requirements.txt
```

### 2. Ingest the Course Materials into ChromaDB
Run a quick initialization snippet to index the sample course files into `./chroma_db`:
```bash
python -c "
from pathlib import Path
import ingestion as ing

courses = {
    'machine_learning': 'Machine Learning',
    'python_programming': 'Python Programming',
    'sql_databases': 'SQL and Databases',
}
ing.get_collection(reset=True)
for folder, course_name in courses.items():
    for file_path in Path('data/course_materials', folder).glob('*.*'):
        n = ing.ingest_file(file_path, course_name)
        print(f'Ingested {file_path.name}: {n} chunks')
"
```

### 3. Launch the Chainlit Web Application
```bash
chainlit run rag_app/app.py -w
```
Open `http://localhost:8000` in your browser. On first launch, the embedding model (`all-MiniLM-L6-v2`) and default LLM (`flan-t5-small`) are downloaded automatically from Hugging Face and cached locally.

---

## Running on Kaggle
1. Open the notebook on Kaggle: **Corrective RAG Course Q&A Assistant**.
2. Set **Accelerator = None (CPU-only)** and **Internet = On** in the notebook settings.
3. Run all cells sequentially (`1` through `16`). Cell 16 launches Chainlit on port `8000` and exposes a public `https://*.trycloudflare.com` URL via `cloudflared`.

---

## License
This project is open-source and available under the **MIT License**.
