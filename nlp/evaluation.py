import csv
import os
import json
import time
import re

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
        reference_context = row.get("reference_context", "")
        reference_explanation = row.get("reference_explanation", "")
        
        print(f"Memproses klaim: {claim}")

        result = fact_check_claim(claim, knowledge_text)
        answer_text = result["answer"]
        retrieved_context = result["context"]
        
        prediction = "TIDAK CUKUP INFORMASI"
        header_text = answer_text.split("Penjelasan")[0] if "Penjelasan" in answer_text else answer_text
        header_text = header_text.upper()
        
        if "BENAR" in header_text:
            prediction = "BENAR"
        elif "SALAH" in header_text:
            prediction = "SALAH"
            
        print(f"PREDIKSI: {prediction} | AKTUAL: {expected_label}")
        
        print("Menjalankan LLM-as-a-Judge untuk evaluasi kualitatif...")
        
        llm_judge_output = evaluate_qualitative_metrics(
            claim,
            reference_explanation,
            answer_text
        )
        
        skor_kualitatif = {'relevansi': 0, 'koherensi': 0, 'factuality': 0, 'hallucination': 0}
        
        try:
            clean_text = re.sub(r"```(?:json)?\s*", "", llm_judge_output, flags=re.IGNORECASE)
            clean_text = re.sub(r"\s*```", "", clean_text)
            clean_text = clean_text.strip()

            json_match = re.search(r'\{.*\}', clean_text, re.DOTALL)
            if json_match:
                clean_text = json_match.group(0)

            skor_kualitatif = json.loads(clean_text)
        except Exception as e:
            print(f"Gagal memparsing JSON. Output asli LLM:\n{llm_judge_output}")
            print(f"Error detail: {e}")

        print(f"Skor Kualitatif: {skor_kualitatif}")
        print("-" * 60)
            
        norm_relevansi = skor_kualitatif.get("relevansi", 0) / 5.0
        norm_koherensi = skor_kualitatif.get("koherensi", 0) / 5.0
        norm_factuality = skor_kualitatif.get("factuality", 0) / 5.0
        norm_hallucination = skor_kualitatif.get("hallucination", 0) / 5.0

        evaluation_results.append({
            "claim": claim,
            "prediction": prediction,
            "label": expected_label,
            "explanation": answer_text,
            "retrieved_context": retrieved_context,
            "reference_context": reference_context,
            "reference_explanation": reference_explanation,
            "qualitative_metrics": skor_kualitatif, 
            
            "relevansi": norm_relevansi,
            "koherensi": norm_koherensi,
            "factuality": norm_factuality,
            "hallucination": norm_hallucination,
            "bebas_halusinasi": norm_hallucination,
        })
        
        time.sleep(5)

    output_path = os.path.join("data", "evaluation_results.json")
    os.makedirs("data", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=4)
    print(f"Evaluasi selesai. Hasil disimpan di {output_path}")

if __name__ == "__main__":
    run_evaluation()