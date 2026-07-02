import json
from rag_pipeline import call_gemini

def evaluate_qualitative_metrics(claim: str, context: str, answer: str) -> dict:
    """Menggunakan LLM untuk mengevaluasi dengan skala 1-5 agar lebih realistis."""
    prompt = f"""Anda adalah juri akademik yang sangat kritis dan pelit nilai.
Tugas Anda: Evaluasi output sistem berdasarkan 4 kriteria menggunakan SKALA 1 sampai 5 (1=Sangat Buruk, 5=Sempurna).

Kriteria Penilaian:
1. relevansi: Seberapa langsung jawaban merespons klaim pengguna tanpa bertele-tele?
2. koherensi: Seberapa luwes, logis, dan rapi tata bahasa yang digunakan? (Kurangi nilai jika bahasanya kaku seperti robot).
3. factuality: Seberapa akurat fakta di jawaban dibandingkan dengan KONTEKS? (Jika tidak ada nomor pasal, jangan beri nilai 5).
4. hallucination: Berapa tingkat kebebasan dari halusinasi? (5 = 100% bebas halusinasi, 1 = banyak karangan bebas).

KLAIM PENGGUNA: {claim}
KONTEKS: {context}
JAWABAN: {answer}

Output WAJIB berupa format JSON murni yang HANYA berisi angka 1, 2, 3, 4, atau 5.
Contoh format:
{{
    "relevansi": 4,
    "koherensi": 3,
    "factuality": 5,
    "hallucination": 5
}}
"""
    response = call_gemini(prompt)
    
    try:
        clean_response = response.replace("```json", "").replace("```", "").strip()
        scores_1_to_5 = json.loads(clean_response)
        
        # Konversi skala 1-5 menjadi skala 0.0 - 1.0 (persentase)
        # Contoh: Nilai 4 menjadi 4/5 = 0.8 (80%)
        normalized_scores = {
            "relevansi": scores_1_to_5.get("relevansi", 0) / 5.0,
            "koherensi": scores_1_to_5.get("koherensi", 0) / 5.0,
            "factuality": scores_1_to_5.get("factuality", 0) / 5.0,
            "hallucination": scores_1_to_5.get("hallucination", 0) / 5.0
        }
        return normalized_scores
    except Exception as e:
        print(f"Gagal memparsing JSON: {e}")
        return {"relevansi": 0, "koherensi": 0, "factuality": 0, "hallucination": 0}