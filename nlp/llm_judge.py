import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def build_judge_prompt(
    claim: str,
    reference_explanation: str,
    model_answer: str
) -> str:

    return f"""
Anda adalah evaluator independen yang bertugas menilai kualitas jawaban sistem fact-checking.

Berikan skor 1 hingga 5 (1=Sangat Buruk, 5=Sangat Baik) untuk 4 metrik berikut: 
1. Relevansi: Seberapa relevan jawaban dengan klaim? 
2. Koherensi: Seberapa logis dan mudah dipahami kalimat jawabannya? 
3. Factuality: Seberapa akurat jawaban mengutip fakta dari konteks dokumen? 
4. Hallucination: Berikan skor 5 jika TIDAK ADA halusinasi (aman), dan skor 1 jika model mengarang fakta.

Data yang dievaluasi

Klaim:
{claim}

Jawaban Referensi:
{reference_explanation}

Jawaban Model:
{model_answer}

Keluarkan HANYA JSON berikut tanpa markdown dan tanpa penjelasan.

{{
    "relevansi": 0,
    "koherensi": 0,
    "factuality": 0,
    "hallucination": 0
}}
"""


def evaluate_qualitative_metrics(
    claim: str,
    reference_explanation: str,
    model_answer: str
) -> str:

    prompt = build_judge_prompt(
        claim,
        reference_explanation,
        model_answer
    )

    default_json = '{"relevansi":0,"koherensi":0,"factuality":0,"hallucination":0}'

    if not API_KEY or "dummy" in API_KEY.lower() or "example" in API_KEY.lower():
        return default_json

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={API_KEY}"

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 120
        }
    }

    try:
        response = requests.post(url, json=payload, timeout=60)

        if response.status_code != 200:
            print(response.text)
            return default_json

        data = response.json()

        return data["candidates"][0]["content"]["parts"][0]["text"]

    except Exception as e:
        print(e)
        return default_json