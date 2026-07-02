import csv
import os
import json
import time
from rag_pipeline import fact_check_claim, load_knowledge, resolve_knowledge_path
from llm_judge import evaluate_qualitative_metrics

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
        ref_context = row.get("reference_context", "")
        ref_explanation = row.get("reference_explanation", "")
        
        print(f"Memproses klaim: {claim}")
        result = fact_check_claim(claim, knowledge_text)
        
        answer_text = result["answer"]
        prediction = "TIDAK CUKUP INFORMASI"
        
        header_text = answer_text.split("Penjelasan")[0] if "Penjelasan" in answer_text else answer_text
        header_text = header_text.upper()
        
        if "BENAR" in header_text:
            prediction = "BENAR"
        elif "SALAH" in header_text:
            prediction = "SALAH"
            
        print("Menjalankan LLM-as-a-Judge untuk evaluasi kualitatif (Relevansi, Koherensi, Factuality, Hallucination)...")
        qualitative_scores = evaluate_qualitative_metrics(claim, result["context"], answer_text)
            
        evaluation_results.append({
            "claim": claim,
            "prediction": prediction,
            "label": expected_label,
            "explanation": answer_text,
            "retrieved_context": result["context"],
            "reference_context": ref_context,
            "reference_explanation": ref_explanation,
            "relevansi": qualitative_scores.get("relevansi", 0),
            "koherensi": qualitative_scores.get("koherensi", 0),
            "factuality": qualitative_scores.get("factuality", 0),
            "hallucination": qualitative_scores.get("hallucination", 0)
        })
        
        print(f"PREDIKSI: {prediction} | AKTUAL: {expected_label}")
        print(f"Skor Kualitatif: {qualitative_scores}")
        print("-" * 60)
        time.sleep(4)

    output_path = os.path.join("data", "evaluation_results.json")
    os.makedirs("data", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=4)
    print(f"Evaluasi selesai. Hasil disimpan di {output_path}")

if __name__ == "__main__":
    run_evaluation()