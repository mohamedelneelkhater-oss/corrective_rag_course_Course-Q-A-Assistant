import os
import re
from pathlib import Path
import pandas as pd

BASE_DIR = os.getenv("RAG_BASE_DIR", ".")
DATA_DIR = os.getenv("RAG_DATA_DIR", f"{BASE_DIR}/course_materials")
DB_DIR = os.getenv("RAG_DB_DIR", f"{BASE_DIR}/chroma_db")
APP_DIR = os.getenv("RAG_APP_DIR", f"{BASE_DIR}/rag_app")
SAVED_LLM_DIR = os.getenv("RAG_SAVED_LLM_DIR", f"{BASE_DIR}/best_llm")
CONFIG_PATH = os.getenv("RAG_CONFIG_PATH", f"{APP_DIR}/best_model.json")
