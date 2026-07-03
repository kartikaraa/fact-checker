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
    eval_file_path = os.path.join("data", "evaluation_results.json")
    
    header_style = """
    <style>
    .dashboard-top {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        align-items: flex-start;
        gap: 16px;
        margin-bottom: 24px;
    }
    .dashboard-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0 0 10px;
    }
    .dashboard-subtitle {
        color: #475569;
        font-size: 0.96rem;
        line-height: 1.6;
        max-width: 680px;
    }
    .summary-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.98) 0%, rgba(239,246,255,0.95) 100%);
        border: 1px solid rgba(59,130,246,0.16);
        box-shadow: 0 18px 40px rgba(14, 47, 134, 0.08);
        border-radius: 22px;
        padding: 20px 24px;
        min-width: 220px;
        max-width: 280px;
    }
    .summary-label {
        color: #475569;
        font-size: 0.88rem;
        margin-bottom: 8px;
    }
    .summary-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1d4ed8;
    }
    .dashboard-card {
        background: #ffffff;
        border-radius: 26px;
        padding: 28px;
        border: 1px solid rgba(226,232,240,0.82);
        box-shadow: 0 24px 60px rgba(15, 23, 42, 0.06);
        margin-bottom: 24px;
    }
    .section-label {
        font-size: 1rem;
        color: #1d4ed8;
        font-weight: 700;
        margin-bottom: 8px;
        letter-spacing: 0.04em;
    }
    .section-title {
        font-size: 1.32rem;
        font-weight: 700;
        margin: 0 0 18px;
        color: #0f172a;
    }
    .section-note {
        color: #64748b;
        font-size: 0.95rem;
        line-height: 1.7;
        margin-bottom: 24px;
    }
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
        margin-top: 18px;
    }
    .metric-box {
        background: #f8fbff;
        border-radius: 20px;
        padding: 18px;
        border: 1px solid rgba(59, 130, 246, 0.12);
        min-height: 150px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .metric-label {
        color: #475569;
        font-size: 0.92rem;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 14px;
    }
    .metric-progress {
        width: 100%;
        height: 10px;
        background: #e2e8f0;
        border-radius: 999px;
        overflow: hidden;
    }
    .metric-progress-bar {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #2563eb 0%, #7c3aed 100%);
    }
    .metric-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 6px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        margin-top: 14px;
    }
    .badge-excellent { background: rgba(16, 185, 129, 0.12); color: #047857; }
    .badge-good { background: rgba(59, 130, 246, 0.12); color: #1d4ed8; }
    .badge-average { background: rgba(251, 191, 36, 0.14); color: #b45309; }
    .badge-warning { background: rgba(249, 115, 22, 0.14); color: #c2410c; }
    .badge-danger { background: rgba(239, 68, 68, 0.12); color: #b91c1c; }
    </style>
    """
    st.markdown(header_style, unsafe_allow_html=True)
    
    if os.path.exists(eval_file_path):
        with open(eval_file_path, "r", encoding="utf-8") as f:
            sample_results = json.load(f)
        
        metrics = evaluate_sample_results(sample_results)
        
        def pct(value):
            return float(value) * 100 if value is not None else 0.0
        
        def format_pct(value):
            return f"{value:.1f}%"
        
        def format_score(value):
            return f"{value:.1f}"
        
        def render_card(col, label, value, progress):
            col.markdown(f"**{label}**")
            col.metric(label="", value=value)
            col.progress(progress / 100 if progress <= 100 else 1.0)

        total_samples = len(sample_results)
        avg_rel = sum(item.get("relevansi", 0) for item in sample_results) / total_samples if total_samples > 0 else 0
        avg_koh = sum(item.get("koherensi", 0) for item in sample_results) / total_samples if total_samples > 0 else 0
        avg_fac = sum(item.get("factuality", 0) for item in sample_results) / total_samples if total_samples > 0 else 0
        avg_hal = sum(item.get("hallucination", 0) for item in sample_results) / total_samples if total_samples > 0 else 0
        
        acc = pct(metrics["accuracy"])
        prec = pct(metrics["precision"])
        rec = pct(metrics["recall"])
        f1 = pct(metrics["f1"])
        hr1 = pct(metrics.get("hit_rate_at_1", 0))
        hr3 = pct(metrics.get("hit_rate_at_3", 0))
        hr5 = pct(metrics.get("hit_rate_at_5", 0))
        hr10 = pct(metrics.get("hit_rate_at_10", 0))
        mrr = metrics.get("mrr", 0.0)
        rouge = pct(metrics.get("rouge_l", 0))
        bleu = pct(metrics.get("bleu", 0))
        
        st.header("Hasil Evaluasi Sistem")
        st.write("Ringkasan kinerja model AI dalam format dashboard profesional dengan komponen kuantitatif dan kualitatif.")

        col_title, col_total = st.columns([3, 1])
        with col_total:
            st.metric("Total Data Diuji", f"{total_samples} Data")

        st.subheader("Evaluasi Kuantitatif")
        st.write("Metrik di bawah ini menunjukkan performa klasifikasi, retrieval, dan generasi teks.")

        def render_card(col, label, value, progress):
            col.markdown(f"**{label}**")
            col.metric(label="", value=value)
            col.progress(progress / 100 if progress <= 100 else 1.0)

        with st.container():
            cols = st.columns(4)
            render_card(cols[0], "Accuracy", format_pct(acc), acc)
            render_card(cols[1], "Precision", format_pct(prec), prec)
            render_card(cols[2], "Recall", format_pct(rec), rec)
            render_card(cols[3], "F1 Score", format_pct(f1), f1)

        st.markdown("---")
        st.subheader("Evaluasi Retrieval (RAG)")
        with st.container():
            cols = st.columns(5)
            render_card(cols[0], "Hit Rate @1", format_pct(hr1), hr1)
            render_card(cols[1], "Hit Rate @3", format_pct(hr3), hr3)
            render_card(cols[2], "Hit Rate @5", format_pct(hr5), hr5)
            render_card(cols[3], "Hit Rate @10", format_pct(hr10), hr10)
            render_card(cols[4], "MRR", format_score(mrr), mrr * 100)

        st.markdown("---")
        st.subheader("Evaluasi Generasi Teks")
        with st.container():
            cols = st.columns(2)
            render_card(cols[0], "ROUGE-L", format_pct(rouge), rouge)
            render_card(cols[1], "BLEU", format_pct(bleu), bleu)

        st.markdown("---")
        st.subheader("Evaluasi Kualitatif (LLM as Judge)")
        with st.container():
            cols = st.columns(4)
            render_card(cols[0], "Relevansi", format_pct(avg_rel * 100), avg_rel * 100)
            render_card(cols[1], "Koherensi", format_pct(avg_koh * 100), avg_koh * 100)
            render_card(cols[2], "Faktualitas", format_pct(avg_fac * 100), avg_fac * 100)
            render_card(cols[3], "Bebas Halusinasi", format_pct(avg_hal * 100), avg_hal * 100)

        with st.expander("Lihat Detail Prediksi & Referensi"):
            st.dataframe(sample_results)
            
    else:
        st.warning("File hasil evaluasi belum ditemukan. Silakan jalankan `python evaluation.py` terlebih dahulu untuk menghasilkan metrik.")