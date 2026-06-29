import csv
import os
import json
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
        
        # Mengekstrak label hasil prediksi (BENAR/SALAH/TIDAK CUKUP INFORMASI)
        answer_text = result["answer"]
        prediction = "TIDAK CUKUP INFORMASI"
        if "Label: BENAR" in answer_text:
            prediction = "BENAR"
        elif "Label: SALAH" in answer_text:
            prediction = "SALAH"
            
        evaluation_results.append({
            "claim": claim,
            "prediction": prediction,
            "label": expected_label,
            "explanation": answer_text
        })
        print(f"PREDIKSI: {prediction} | AKTUAL: {expected_label}")
        print("-" * 60)

    # Simpan hasil evaluasi ke JSON
    output_path = os.path.join("data", "evaluation_results.json")
    os.makedirs("data", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=4)
    print(f"Evaluasi selesai. Hasil disimpan di {output_path}")

if __name__ == "__main__":
    run_evaluation()