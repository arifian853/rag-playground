import os, io, time, uuid, numpy as np
from typing import List, Dict, Any
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Init Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -----------------------------
# KONFIGURASI INTI (5 TUNABLE PARAMETER)
# -----------------------------
# 1) Ukuran chunk 
CHUNK_SIZE = 300
# 2) Banyaknya kandidat awal dari vector search
TOP_K_INITIAL = 20
# 3) Jumlah konteks 
FINAL_TOP_M = 5
# 4) Reranker (Cross-Encoder) 
USE_RERANKER = True
# 5) Temperatur generasi 
TEMPERATURE = 0.2
# 6) Model yang tersedia di Groq ada banyak, coba ganti-ganti ke yang lain dan coba bandingkan hasilnya
GROQ_MODEL = "llama-3.1-8b-instant"  # model default di Groq

# Model lain yang bisa dicobain: gpt-oss-120b, llama-3.3-70b-versatile
# -----------------------------

# (non-tunable tapi penting)
CHUNK_OVERLAP = 60           # overlap kecil yg aman
SCORE_THRESHOLD = 0.30       # drop kandidat yg terlalu rendah

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

# Embedding ringan & multilingual
EMBED_NAME = "intfloat/multilingual-e5-small"
embedder = SentenceTransformer(EMBED_NAME)

# (Opsional) Reranker kecil yang cepat
# Diaktifkan hanya jika USE_RERANKER=True
_ce = None
def get_reranker():
    global _ce
    if _ce is None:
        from sentence_transformers import CrossEncoder
        _ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _ce

# In-memory store
DOCS: List[Dict[str, Any]] = []  # {id, text, meta, emb(np.ndarray)}

# ---------- Util ----------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    words = text.split()
    if not words: return []
    stride = max(1, chunk_size - overlap)
    chunks = []
    for i in range(0, len(words), stride):
        chunk = " ".join(words[i:i+chunk_size]).strip()
        if chunk: chunks.append(chunk)
    return chunks

def embed_texts(texts: List[str]) -> np.ndarray:
    embs = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    # normalize → cosine = dot
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
    return (embs / norms).astype(np.float32)

def extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = [p.extract_text() or "" for p in reader.pages]
    return "\n".join(pages)

def build_prompt(query: str, ctxs: List[Dict[str, Any]]) -> str:
    head = (
        "You are a grounded QA assistant.\n"
        "Answer ONLY using the CONTEXT. If not in context, say \"I don't know\".\n"
        "Give short citations like [source].\n\n"
        f"QUESTION: {query}\n\nCONTEXT:\n"
    )
    body = ""
    for i, c in enumerate(ctxs, 1):
        src = c.get("meta", {}).get("source", f"chunk-{i}")
        body += f"[{src}] {c['text']}\n\n"
    return head + body + "\nAnswer:\n"

def call_groq(prompt: str, model: str = GROQ_MODEL, temperature: float = TEMPERATURE, max_tokens: int = 512) -> str:
    completion = groq_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful, concise assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_completion_tokens=max_tokens
    )
    return completion.choices[0].message.content

# ---------- Web UI Endpoints ----------
@app.route("/")
def index():
    """Serve halaman utama"""
    return send_from_directory(".", "index.html")

@app.route("/<path:path>")
def serve_static(path):
    """Serve static files (css, js, images, dll)"""
    return send_from_directory(".", path)

@app.post("/clear")
def clear_knowledge():
    """Hapus semua dokumen dari knowledge base"""
    count = len(DOCS)
    DOCS.clear()
    return jsonify({"cleared": count, "total_store": 0})

# ---------- API Endpoints ----------
@app.get("/health")
def health():
    # Ambil unique sources dari DOCS
    sources = list(set(d.get("meta", {}).get("source", "unknown") for d in DOCS))
    return {"ok": True, "docs": len(DOCS), "sources": sources}

@app.post("/ingest")
def ingest():
    """
    Multipart form:
    - file: PDF (opsional, bisa lebih dari 1)
    - text: teks bebas (opsional)
    - source: string (opsional, nama sumber)
    - replace: "true" untuk hapus knowledge base lama (default: false)
    """
    t0 = time.time()
    source = request.form.get("source", "user")
    replace = request.form.get("replace", "false").lower() == "true"
    
    # Clear knowledge base jika replace=true
    if replace:
        DOCS.clear()
    
    chunks_all = []

    # teks bebas
    text = request.form.get("text", "").strip()
    if text:
        chunks_all += [{"text": ch, "meta": {"source": source}} for ch in chunk_text(text)]

    # PDF
    if "file" in request.files:
        files = request.files.getlist("file")
        for f in files:
            if not f.filename.lower().endswith(".pdf"): continue
            raw = f.read()
            txt = extract_pdf_text(raw)
            src = f.filename or source
            chunks = chunk_text(txt)
            chunks_all += [{"text": ch, "meta": {"source": src}} for ch in chunks]

    if not chunks_all:
        return jsonify({"error": "No content provided"}), 400

    embs = embed_texts([c["text"] for c in chunks_all])
    for c, e in zip(chunks_all, embs):
        DOCS.append({"id": str(uuid.uuid4()), "text": c["text"], "meta": c["meta"], "embedding": e})

    return {
        "ingested_chunks": len(chunks_all),
        "total_store": len(DOCS),
        "replaced": replace,
        "latency_ms": round((time.time() - t0) * 1000, 1)
    }

@app.post("/chat")
def chat():
    """
    JSON:
    { "query": "..." }
    """
    if not groq_client.api_key:
        return jsonify({"error": "GROQ_API_KEY not set"}), 400
    if not DOCS:
        return jsonify({"error": "Knowledge base empty. Ingest first."}), 400

    data = request.get_json(force=True)
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Missing query"}), 400

    t0 = time.time()
    q_vec = embed_texts([query])[0]

    M = np.stack([d["embedding"] for d in DOCS], axis=0)
    sims = (M @ q_vec)  # cosine (sudah normalized)
    idx = np.argsort(-sims)[:max(TOP_K_INITIAL, FINAL_TOP_M)]

    cands = []
    for i in idx:
        s = float(sims[i])
        if s < SCORE_THRESHOLD:  # filter rendah
            continue
        d = DOCS[i]
        cands.append({"id": d["id"], "text": d["text"], "meta": d["meta"], "_score": s})
    if not cands:  # kalau terlalu ketat, ambil TOP_K saja
        cands = [{"id": DOCS[i]["id"], "text": DOCS[i]["text"], "meta": DOCS[i]["meta"], "_score": float(sims[i])} for i in idx]

    # Reranker (optional, sangat menaikkan presisi)
    if USE_RERANKER and len(cands) > FINAL_TOP_M:
        ce = get_reranker()
        pairs = [[query, c["text"]] for c in cands]
        scores = ce.predict(pairs)
        for c, s in zip(cands, scores):
            c["_rerank"] = float(s)
        cands.sort(key=lambda x: x.get("_rerank", x["_score"]), reverse=True)
        cands = cands[:FINAL_TOP_M]
    else:
        cands = cands[:FINAL_TOP_M]

    retrieval_ms = round((time.time() - t0) * 1000, 1)

    prompt = build_prompt(query, cands)
    t1 = time.time()
    try:
        answer = call_groq(prompt, model=GROQ_MODEL, temperature=TEMPERATURE, max_tokens=512)
    except Exception as e:
        return jsonify({"error": f"LLM call failed: {e}"}), 500
    gen_ms = round((time.time() - t1) * 1000, 1)

    return jsonify({
        "answer": answer,
        "contexts": [{"text": c["text"], "meta": c["meta"], "score": round(c.get('_rerank', c['_score']), 4)} for c in cands],
        "metrics": {"retrieval_ms": retrieval_ms, "generation_ms": gen_ms}
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 7860)), debug=True)