# Fact Checker Lalu Lintas dengan RAG + Gemini 2.5 Flash

Proyek ini adalah prototype sistem fact-checking untuk klaim peraturan lalu lintas berbasis dokumen contoh UU Nomor 22 Tahun 2009.

## Tujuan
- Memverifikasi kebenaran klaim terkait aturan lalu lintas.
- Menyediakan output BENAR / SALAH / TIDAK CUKUP INFORMASI.
- Menunjukkan alur NLP yang sesuai syarat UAS: LLM, prompt engineering, RAG, evaluasi, dan analisis etika.

## Struktur folder
- data/uu22_2009_knowledge.txt : basis pengetahuan lokal.
- data/sample_claims.csv : contoh data evaluasi.
- rag_pipeline.py : pipeline retrieval + prompt + pemanggilan Gemini.
- app.py : aplikasi Streamlit interaktif.
- evaluation.py : script evaluasi sederhana.
- requirements.txt : dependensi Python.

## Urutan kerja
1. Instal dependensi.
2. Siapkan API key Gemini.
3. Jalankan aplikasi Streamlit.
4. Jalankan evaluasi contoh.

## Instalasi
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Konfigurasi API
Salin .env.example menjadi .env lalu isi nilai GEMINI_API_KEY.

## Jalankan aplikasi
```bash
streamlit run app.py
```

## Jalankan evaluasi
```bash
python evaluation.py
```

## Catatan penting
- Proyek ini bersifat demo edukatif.
- Saat API key Gemini belum diisi, sistem akan otomatis memakai mode demo berbasis aturan sederhana.
- Untuk penggunaan nyata, ganti data/uu22_2009_knowledge.txt dengan teks resmi UU yang lebih lengkap dan valid.
