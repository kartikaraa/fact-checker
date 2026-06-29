import os
import json
import streamlit as st
from dotenv import load_dotenv
from rag_pipeline import fact_check_claim, is_placeholder_key, load_knowledge, resolve_knowledge_path
from evaluation_metrics import evaluate_sample_results

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
tab_checker, tab_eval = st.tabs(["Aplikasi Fact Checker", "Laporan Evaluasi Model"])

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
    st.subheader("Hasil Evaluasi Kuantitatif (Berdasarkan Dataset)")
    st.write("Metrik di bawah ini dihasilkan dari pengujian *batch* menggunakan `evaluation.py`.")
    
    eval_file_path = os.path.join("data", "evaluation_results.json")
    
    if os.path.exists(eval_file_path):
        with open(eval_file_path, "r", encoding="utf-8") as f:
            sample_results = json.load(f)
            
        metrics = evaluate_sample_results(sample_results)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        col2.metric("Precision", f"{metrics['precision']:.2%}")
        col3.metric("Recall", f"{metrics['recall']:.2%}")
        col4.metric("F1 Score", f"{metrics['f1_score']:.2%}")
        
        st.write(f"Total dataset diuji: **{metrics['support']} klaim**")
        
        with st.expander("Lihat Detail Prediksi"):
            st.dataframe(sample_results)
    else:
        st.warning("File hasil evaluasi belum ditemukan. Silakan jalankan `python evaluation.py` terlebih dahulu untuk menghasilkan metrik.")