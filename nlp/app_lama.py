import os
import json
import streamlit as st
from dotenv import load_dotenv
from rag_pipeline import fact_check_claim, is_placeholder_key, load_knowledge, resolve_knowledge_path
from evaluation_metrics import evaluate_sample_results
from evaluation_metrics import calculate_classification_metrics

load_dotenv()

st.set_page_config(page_title="Fact Checker Lalu Lintas", page_icon="🚦", layout="centered")

st.title("Fact Checker Peraturan Lalu Lintas")
st.write("Verifikasi klaim atau informasi media sosial berdasarkan dokumen pengetahuan UU Nomor 22 Tahun 2009.")

if is_placeholder_key(os.getenv("GEMINI_API_KEY", "")):
    st.info("API key Gemini belum aktif atau belum valid. Sistem saat ini berjalan dalam mode fallback terbatas.")

@st.cache_data
def get_cached_knowledge():
    path = resolve_knowledge_path()
    text = load_knowledge(path)
    return text

knowledge_text = get_cached_knowledge()
tab_checker, tab_eval = st.tabs(["Fact Checker", "Laporan Evaluasi Model"])

with tab_checker:
    st.subheader("Verifikasi Klaim")
    claim = st.text_area("Masukkan klaim yang ingin diverifikasi", height=120)

    if st.button("Cek Klaim", type="primary"):
        if not claim.strip():
            st.warning("Silakan masukkan klaim terlebih dahulu.")
        else:
            with st.spinner("Memproses klaim..."):
                result = fact_check_claim(claim, knowledge_text)
            
            st.subheader("Hasil Analisis")
            st.write(result["answer"])

            with st.expander("Lihat dokumen konteks (Vector Search Results)"):
                st.write(result["context"])

with tab_eval:
    st.subheader("Hasil Evaluasi Kuantitatif")
    st.write("Metrik di bawah ini dihasilkan dari pengujian *batch* terhadap dataset ground truth.")
    
    eval_file_path = os.path.join("data", "evaluation_results.json")
    
    if os.path.exists(eval_file_path):
        with open(eval_file_path, "r", encoding="utf-8") as f:
            sample_results = json.load(f)
            
        metrics = evaluate_sample_results(sample_results)
        
        st.write("##### 1. Kinerja Klasifikasi")
        
        y_true = [item.get("label", "").upper() for item in sample_results]
        y_pred = [item.get("prediction", "").upper() for item in sample_results]
        
        class_metrics = calculate_classification_metrics(y_true, y_pred)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{class_metrics['accuracy']:.2%}")
        col2.metric("Precision", f"{class_metrics['precision']:.2%}")
        col3.metric("Recall", f"{class_metrics['recall']:.2%}")
        col4.metric("F1 Score", f"{class_metrics['f1']:.2%}")
        
        st.write("##### 2. Kinerja Retrieval")
        col5, col6, col7, col8, col9 = st.columns(5)
        col5.metric("Hit Rate @1", f"{metrics.get('hit_rate_at_1', 0):.2%}")
        col6.metric("Hit Rate @3", f"{metrics.get('hit_rate_at_3', 0):.2%}")
        col7.metric("Hit Rate @5", f"{metrics.get('hit_rate_at_5', 0):.2%}")
        col8.metric("Hit Rate @10", f"{metrics.get('hit_rate_at_10', 0):.2%}")
        col9.metric("MRR", f"{metrics.get('mrr', 0):.2f}")
        
        st.write("##### 3. Kualitas Generasi")
        col7, col8, _ = st.columns([1, 1, 2])
        col7.metric("ROUGE-L Score", f"{metrics['rouge_l']:.2%}")
        col8.metric("BLEU Score", f"{metrics['bleu']:.2%}")
        
        st.write("##### 4. Evaluasi Kualitatif (LLM as a Judge)")
        total_samples = len(sample_results)
        avg_rel = sum(item.get("relevansi", 0) for item in sample_results) / total_samples if total_samples > 0 else 0
        avg_koh = sum(item.get("koherensi", 0) for item in sample_results) / total_samples if total_samples > 0 else 0
        avg_fac = sum(item.get("factuality", 0) for item in sample_results) / total_samples if total_samples > 0 else 0
        avg_hal = sum(item.get("hallucination", 0) for item in sample_results) / total_samples if total_samples > 0 else 0
        
        col9, col10, col11, col12 = st.columns(4)
        col9.metric("Relevansi", f"{avg_rel:.2%}")
        col10.metric("Koherensi", f"{avg_koh:.2%}")
        col11.metric("Factuality", f"{avg_fac:.2%}")
        col12.metric("Bebas Halusinasi", f"{avg_hal:.2%}")
        
        st.write(f"Total klaim diuji: **{metrics['support']} data**")
        
        with st.expander("Lihat Detail Prediksi & Referensi"):
            st.dataframe(sample_results)
    else:
        st.warning("File hasil evaluasi belum ditemukan. Silakan jalankan `python evaluation.py` terlebih dahulu untuk menghasilkan metrik.")