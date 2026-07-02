import warnings
from typing import List, Dict
from collections import Counter
from rouge import Rouge
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_classification_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    # Tambahkan parameter average='weighted' dan zero_division=0 untuk Multiclass
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def compute_retrieval_metrics(results: List[Dict[str, str]]) -> Dict[str, float]:
    """Menghitung Hit Rate pada berbagai nilai K dan MRR"""
    k_values = [1, 3, 5, 10] 
    hits_at_k = {k: 0 for k in k_values}
    
    mrr_score = 0.0
    total = 0
    
    for item in results:
        ref_context = item.get("reference_context", "").strip().lower()
        if not ref_context:
            continue 
            
        total += 1
        retrieved_context = item.get("retrieved_context", "")
        chunks = retrieved_context.split("\n\n")
        
        # Bersihkan referensi dari spasi dan newline
        clean_ref = re.sub(r'\s+', '', ref_context)
        
        found = False
        for rank, chunk in enumerate(chunks, start=1):
            clean_chunk = re.sub(r'\s+', '', chunk.lower())
            
            if clean_ref in clean_chunk:
                if not found:
                    mrr_score += (1.0 / rank)
                    found = True
                    
                for k in k_values:
                    if rank <= k:
                        hits_at_k[k] += 1
                break # Hentikan pencarian untuk dokumen ini jika sudah ketemu
                    
    metrics = {"mrr": mrr_score / total if total > 0 else 0.0}
    for k, hits in hits_at_k.items():
        metrics[f"hit_rate_at_{k}"] = hits / total if total > 0 else 0.0
        
    return metrics

def compute_semantic_metrics(results: List[Dict[str, str]]) -> Dict[str, float]:
    """Menghitung metrik semantik NLG (ROUGE-L dan BLEU)"""
    rouge = Rouge()
    smoother = SmoothingFunction().method1
    total_rouge_l = 0
    total_bleu = 0
    valid_samples = 0
    
    for item in results:
        raw_answer = item.get("explanation", "")
        hypothesis = raw_answer.split("Penjelasan:")[-1].strip().lower()
        reference = item.get("reference_explanation", "").strip().lower()
        
        if hypothesis and reference:
            try:
                # ROUGE
                scores = rouge.get_scores(hypothesis, reference)
                total_rouge_l += scores[0]['rouge-l']['f']
                
                # BLEU
                ref_tokens = [reference.split()]
                hypo_tokens = hypothesis.split()
                total_bleu += sentence_bleu(ref_tokens, hypo_tokens, smoothing_function=smoother)
                
                valid_samples += 1
            except ValueError:
                pass
            
    return {
        "rouge_l": total_rouge_l / valid_samples if valid_samples > 0 else 0.0,
        "bleu": total_bleu / valid_samples if valid_samples > 0 else 0.0
    }

def evaluate_sample_results(results: List[Dict[str, str]]) -> Dict[str, object]:
    predictions = [item.get("prediction", "").upper() for item in results]
    labels = [item.get("label", "").upper() for item in results]

    metrics = calculate_classification_metrics(labels, predictions)
    metrics["support"] = len(labels) 
    
    retrieval_metrics = compute_retrieval_metrics(results)
    metrics.update(retrieval_metrics) 

    semantic_scores = compute_semantic_metrics(results)
    metrics["rouge_l"] = semantic_scores["rouge_l"]
    metrics["bleu"] = semantic_scores["bleu"]
    
    return metrics