import gc
import re
import time

import ingestion as ing
from ingestion import CONFIG_PATH, SAVED_LLM_DIR

UNANSWERED = "Sorry, the course materials do not contain enough information to answer this question."
UNANSWERED_MARKERS = ("not contain enough information", "don't know")
MIN_SCORE = 0.25         # standard RAG: minimum similarity of the best chunk
GRADE_MIN_SCORE = 0.20   # corrective RAG: chunks below this similarity are rejected without calling the grader
MAX_ACCEPT = 3           # corrective RAG: stop grading once this many relevant chunks were found

LLM_REGISTRY = {
    "flan-t5-small": {"repo": "google/flan-t5-small", "type": "seq2seq"},
    "flan-t5-base": {"repo": "google/flan-t5-base", "type": "seq2seq"},
    "qwen2.5-0.5b-instruct": {"repo": "Qwen/Qwen2.5-0.5B-Instruct", "type": "causal"},
}

ANSWER_SYSTEM = (
    "You are a course assistant. Answer the question using ONLY the verified context. "
    "Be concise and answer in one or two complete sentences. "
    f"If the answer is not in the context, reply exactly: {UNANSWERED}"
)
GRADER_SYSTEM = (
    "You are a strict relevance grader. Decide whether the context contains information that helps answer the question. "
    "Reply with only one word: yes or no."
)
REWRITE_SYSTEM = (
    "You rewrite student questions into clear, specific search queries for a course knowledge base. "
    "Use precise technical keywords. Reply with the rewritten query only."
)
STOPWORDS = set("a an the is are was were be to of in on for and or what which who whom how why when where does do did can could "
                "should would i my me we you your it its this that these those with about from by at as into than then there "
                "stop use used using get make tell give".split())

_LLMS = {}


def load_llm(name):
    if name in _LLMS:
        return _LLMS[name]
    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
    cfg = LLM_REGISTRY[name]
    path = cfg["repo"]
    marker = f"{SAVED_LLM_DIR}/model_name.txt"
    if os.path.exists(marker) and open(marker).read().strip() == name:
        path = SAVED_LLM_DIR
    tok = AutoTokenizer.from_pretrained(path)
    model_cls = AutoModelForSeq2SeqLM if cfg["type"] == "seq2seq" else AutoModelForCausalLM
    model = model_cls.from_pretrained(path, dtype=torch.float32).eval()
    _LLMS[name] = (tok, model)
    return _LLMS[name]


def unload_llm(name):
    _LLMS.pop(name, None)
    gc.collect()


def chat(name, system, user, max_new_tokens=100, num_beams=1):
    import torch
    tok, model = load_llm(name)
    with torch.no_grad():
        if LLM_REGISTRY[name]["type"] == "seq2seq":
            inputs = tok(f"{system}\n\n{user}\nAnswer:", return_tensors="pt", truncation=True, max_length=1536)
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=num_beams, do_sample=False)
            return tok.decode(out[0], skip_special_tokens=True).strip()
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=1536)
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def retrieve(query, k=6, course=None):
    vec = ing.get_embedder().encode([query], normalize_embeddings=True).tolist()
    where = {"course": course} if course and course != "All" else None
    res = ing.get_collection().query(query_embeddings=vec, n_results=k, where=where)
    return [{"id": i, "text": t, "meta": m, "score": 1 - d}
            for i, t, m, d in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])]


def grade_chunk(question, chunk_text, name):
    verdict = chat(name, GRADER_SYSTEM, f"Context: {chunk_text}\n\nQuestion: {question}\n\nDoes the context help answer the question? Answer yes or no.", max_new_tokens=3)
    return verdict.strip().lower().startswith("yes")


def grade_hits(question, hits, name):
    accepted, rows = [], []
    for h in sorted(hits, key=lambda x: -x["score"]):
        if len(accepted) >= MAX_ACCEPT:
            verdict = "skipped (enough relevant chunks)"
        elif h["score"] < GRADE_MIN_SCORE:
            verdict = "no (similarity too low)"
        elif grade_chunk(question, h["text"], name):
            verdict = "yes"
            accepted.append(h)
        else:
            verdict = "no"
        rows.append(f"[{verdict}] {h['meta']['file']} (part {h['meta']['page']}, similarity {h['score']:.2f})")
    return accepted, rows


def keyword_query(question):
    tokens = re.findall(r"[a-zA-Z0-9%']+", question.lower())
    return " ".join(t for t in tokens if t not in STOPWORDS)


def rewrite_query(question, name, tried):
    user = f"Question: {question}\nQueries already tried (use different keywords): {'; '.join(tried)}\nRewritten query:"
    new = chat(name, REWRITE_SYSTEM, user, max_new_tokens=40).strip().strip('"')
    lowered = [t.lower() for t in tried]
    if len(new) < 8 or new.lower() in lowered:
        new = keyword_query(question)
        if new.lower() in lowered:
            new = f"{new} definition explanation"
    return new


def is_refusal(answer):
    a = answer.lower().replace("\u2019", "'")
    return any(m in a for m in UNANSWERED_MARKERS)


def source_labels(hits):
    labels, seen = [], set()
    for h in hits:
        m = h["meta"]
        label = f"{m['file']} (course: {m['course']}, {m['type']}, part {m['page']})"
        if label not in seen:
            seen.add(label)
            labels.append(label)
    return labels
def clean_answer(text):
    text = re.sub(r"\[\d+\]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return re.sub(r"\s+([.,;:!?])", r"\1", text).strip()


def is_bad_answer(text):
    return len(re.findall(r"[A-Za-z]", text)) < 3


def extractive_answer(question, hits):
    q_words = set(keyword_query(question).split())
    best, best_score = "", 0
    for h in hits:
        for sent in re.split(r"(?<=[.!?])\s+", h["text"]):
            score = len(q_words & set(keyword_query(sent).split()))
            if score > best_score:
                best, best_score = sent.strip(), score
    return best


def generate_answer(question, hits, model_name):
    context = "\n\n".join(h["text"] for h in hits)[:3000]
    user = f"Context:\n{context}\n\nQuestion: {question}"
    answer = clean_answer(chat(model_name, ANSWER_SYSTEM, user, max_new_tokens=120, num_beams=2))
    if is_bad_answer(answer):
        retry_system = ANSWER_SYSTEM + " Write a complete sentence that restates the question."
        answer = clean_answer(chat(model_name, retry_system, user, max_new_tokens=120, num_beams=4))
    if is_bad_answer(answer):
        answer = extractive_answer(question, hits)
    return answer    

def corrective_answer(question, model_name, k=6, course=None, corrective=True, max_rewrites=2):
    start = time.time()
    trace, accepted, seen, tried = [], [], set(), [question]
    query, rewrites, retrieved_total = question, 0, 0

    if not corrective:
        hits = retrieve(question, k=min(k, 4), course=course)
        retrieved_total = len(hits)
        trace.append({"step": "Retrieve", "detail": f'Standard RAG - query: "{question}" -> {len(hits)} chunks (no grading)'})
        accepted = hits if hits and hits[0]["score"] >= MIN_SCORE else []
    else:
        for attempt in range(max_rewrites + 1):
            hits = retrieve(query, k=k, course=course)
            retrieved_total += len(hits)
            trace.append({"step": f"Retrieve (query {attempt + 1})", "detail": f'Query: "{query}" -> {len(hits)} chunks'})
            fresh = [h for h in hits if h["id"] not in seen]
            seen.update(h["id"] for h in hits)
            good, rows = grade_hits(question, fresh, model_name)
            accepted += good
            trace.append({"step": f"Grade chunks (query {attempt + 1})",
                          "detail": "\n".join(rows) + f"\n=> {len(good)} relevant of {len(fresh)} new chunks"})
            if accepted or attempt == max_rewrites:
                break
            query = rewrite_query(question, model_name, tried)
            tried.append(query)
            rewrites += 1
            trace.append({"step": "Rewrite query", "detail": f'Context was weak. New query: "{query}"'})
        accepted = sorted(accepted, key=lambda h: -h["score"])[:MAX_ACCEPT]

    if not accepted:
        trace.append({"step": "Decision", "detail": "No verified context found -> the question is returned as unanswered."})
        answer = UNANSWERED
    else:
        trace.append({"step": "Decision", "detail": f"{len(accepted)} verified chunk(s) -> generate the answer from them only."})
        answer = generate_answer(question, accepted, model_name) or UNANSWERED
        if is_refusal(answer):
            answer, accepted = UNANSWERED, []

    return {"answer": answer, "sources": source_labels(accepted), "hits": accepted, "trace": trace, "rewrites": rewrites,
            "retrieved": retrieved_total, "accepted": len(accepted), "latency": time.time() - start, "corrective": corrective}


def format_trace(out):
    lines = [f"- {t['step']}: {t['detail']}" for t in out["trace"]]
    return "\n".join(lines)
