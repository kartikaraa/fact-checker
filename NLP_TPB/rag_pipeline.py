import os
import re
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
import requests
import chromadb
from chromadb.utils import embedding_functions

try:
    import pypdf
except Exception:
    pypdf = None

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# 1. Inisialisasi Multilingual Embedding
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

# 2. Inisialisasi Persistent Database ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection_name = "uu_lalu_lintas"

# 3. Ambil atau Buat Koleksi Vektor
vector_collection = chroma_client.get_or_create_collection(
    name=collection_name, 
    embedding_function=sentence_transformer_ef
)

def is_placeholder_key(key: str) -> bool:
    if not key:
        return True
    key_lower = key.strip().lower()
    placeholders = ["your_google_gemini_api_key_here", "api_key_anda", "your_api_key", "dummy", "example"]
    return any(item in key_lower for item in placeholders)


def resolve_knowledge_path() -> str:
    candidates = [
        Path("UU Nomor 22 Tahun 2009.pdf"),
        Path("data/uu22_2009_knowledge.txt"),
        Path("data") / "uu22_2009_knowledge.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(Path("data/uu22_2009_knowledge.txt"))


def load_knowledge(path: str) -> str:
    path_obj = Path(path)
    if path_obj.suffix.lower() == ".pdf":
        if pypdf is None:
            return "PDF reader belum tersedia. Install pypdf untuk membaca dokumen PDF."
        try:
            reader = pypdf.PdfReader(str(path_obj))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            return text
        except Exception as exc:
            return f"Gagal membaca PDF: {exc}"

    with open(path_obj, "r", encoding="utf-8") as f:
        return f.read()


def split_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Memotong teks menggunakan metode sliding window untuk konteks yang lebih utuh."""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
        
    return chunks


def populate_vector_db(knowledge_text: str):
    """Memasukkan chunk ke ChromaDB hanya jika database masih kosong."""
    if vector_collection.count() > 0:
        return
        
    print("Memproses embedding dokumen perdana (mungkin memakan waktu beberapa menit)...")
    chunks = split_text(knowledge_text)
    
    # ChromaDB membutuhkan ID unik untuk setiap dokumen
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    
    vector_collection.add(
        documents=chunks,
        ids=ids
    )
    print("Embedding selesai dan berhasil disimpan ke disk lokal!")


def retrieve_context(claim: str, knowledge_text: str) -> str:
    """Mengambil konteks menggunakan Vector Search (Cosine Similarity)."""
    populate_vector_db(knowledge_text)
    
    # Mencari 7 chunk paling relevan berdasarkan semantic similarity
    results = vector_collection.query(
        query_texts=[claim],
        n_results=10
    )
    
    if results["documents"] and len(results["documents"][0]) > 0:
        top_chunks = results["documents"][0]
        return "\n\n".join(top_chunks)
    
    return knowledge_text

def build_prompt(claim: str, context: str) -> str:
    return f"""Anda adalah sistem fact-checking untuk peraturan lalu lintas berdasarkan dokumen UU Nomor 22 Tahun 2009.

Tugas Anda:
1. Tentukan apakah klaim berikut BENAR, SALAH, atau TIDAK CUKUP INFORMASI.
2. Berikan penjelasan singkat berdasarkan konteks yang tersedia.
3. Jika klaim tidak dapat dipastikan dari dokumen, gunakan TIDAK CUKUP INFORMASI.

Klaim: {claim}

Context dari dokumen:
{context}

Format jawaban:
Hasil: BENAR / SALAH / TIDAK CUKUP INFORMASI
Penjelasan: ...
"""


def call_gemini(prompt: str) -> str:
    if is_placeholder_key(API_KEY):
        return "API key Gemini belum valid atau belum diisi. Sistem akan menjalankan mode demo berbasis aturan sederhana."

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 250
        }
    }
    response = requests.post(url, json=payload, timeout=60)
    if response.status_code != 200:
        return f"Error API Gemini: {response.status_code} - {response.text}"

    data = response.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return str(data)


def fallback_rule_based_check(claim: str, context: str) -> str:
    return "Hasil: TIDAK CUKUP INFORMASI\nPenjelasan: API LLM sedang tidak tersedia. Sistem gagal melakukan verifikasi mendalam terhadap klaim ini."


def fact_check_claim(claim: str, knowledge_text: str) -> Dict[str, str]:
    context = retrieve_context(claim, knowledge_text)
    prompt = build_prompt(claim, context)
    answer = call_gemini(prompt)
    if "mode demo" in answer.lower() or "error api gemini" in answer.lower() or "api key" in answer.lower():
        answer = fallback_rule_based_check(claim, context)
    return {"context": context, "prompt": prompt, "answer": answer}