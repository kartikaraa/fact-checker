import csv
import os
import json
import time
from rag_pipeline import fact_check_claim, load_knowledge, resolve_knowledge_path

def run_evaluation():
    knowledge_path = resolve_knowledge_path()
    knowledge_text = load_knowledge(knowledge_path)
    
    file_path = os.path.join("data", "sample_claims.csv")
    if not os.path.exists(file_path):
        print(f"File dataset tidak ditemukan di {file_path}")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    evaluation_results = []

    for row in rows:
        claim = row["claim"]
        expected_label = row["label"]
        
        print(f"Memproses klaim: {claim}")
        result = fact_check_claim(claim, knowledge_text)
        
        # EKSTRAKSI YANG LEBIH ROBUST (Toleransi improvisasi LLM)
        answer_text = result["answer"]
        prediction = "TIDAK CUKUP INFORMASI"
        
        # 1. Pisahkan bagian jawaban awal dari teks penjelasan panjangnya
        header_text = answer_text.split("Penjelasan")[0] if "Penjelasan" in answer_text else answer_text
        
        # 2. Ubah ke huruf besar semua agar kebal terhadap perbedaan kapitalisasi (benar/BENAR/Benar)
        header_text = header_text.upper()
        
        # 3. Cek keberadaan kata kunci tanpa memedulikan karakter ekstra seperti ** atau :
        if "BENAR" in header_text:
            prediction = "BENAR"
        elif "SALAH" in header_text:
            prediction = "SALAH"
            
        evaluation_results.append({
            "claim": claim,
            "prediction": prediction,
            "label": expected_label,
            "explanation": answer_text
        })
        print(f"PREDIKSI: {prediction} | AKTUAL: {expected_label}")
        print("-" * 60)

        time.sleep(4)

    # Simpan hasil evaluasi ke JSON
    output_path = os.path.join("data", "evaluation_results.json")
    os.makedirs("data", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=4)
    print(f"Evaluasi selesai. Hasil disimpan di {output_path}")

if __name__ == "__main__":
    run_evaluation()