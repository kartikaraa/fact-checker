# Laporan UAS NLP - Fact Checking Peraturan Lalu Lintas

## 1. Latar Belakang
Informasi terkait aturan lalu lintas sering beredar di media sosial, grup WhatsApp, TikTok, Instagram, dan forum online. Tidak semua informasi tersebut benar. Oleh karena itu, dibutuhkan sistem yang dapat memverifikasi klaim masyarakat dengan dasar dokumen resmi, yaitu UU Nomor 22 Tahun 2009.

## 2. Tujuan Sistem
Membuat sistem fact-checking yang mampu:
- menerima input klaim pengguna,
- mengambil konteks dari dokumen pengetahuan,
- memeriksa apakah klaim tersebut benar atau salah,
- memberikan penjelasan singkat dan hasil akhir.

## 3. Batasan Masalah
- Fokus pada klaim sederhana terkait aturan lalu lintas.
- Data pengetahuan masih berupa contoh teks ringkas.
- Sistem ini merupakan prototype, bukan sistem resmi pemeriksa hukum.

## 4. Alur Sistem
Input klaim → retrieval konteks → pemrosesan LLM → output hasil → evaluasi.

## 5. Arsitektur dan Teknologi
- Python
- Streamlit
- Gemini 2.5 Flash API
- Prompt engineering
- Retrieval-Augmented Generation (RAG)

## 6. Prompt Engineering
Prompt dibuat dengan struktur:
- instruction,
- context,
- constraint,
- format jawaban.

## 7. RAG
Sistem mengambil bagian dokumen yang relevan dari basis pengetahuan untuk mendukung keputusan model.

## 8. Evaluasi
Evaluasi dapat dilakukan dengan:
- akurasi untuk klaim benar/salah,
- precision/recall/F1 untuk eksperimen klasifikasi,
- analisis kualitas jawaban dari sisi relevansi dan koherensi.

## 9. Etika, Bias, dan Safety
Potensi masalah yang perlu dicatat:
- bias data,
- kesalahan karena konteks terbatas,
- risiko misinformasi jika sistem dipakai tanpa verifikasi.
Mitigasi:
- gunakan dokumen resmi,
- tampilkan sumber konteks,
- jangan gunakan hasil sebagai keputusan hukum final.

## 10. Kesimpulan
Sistem ini dapat menjadi prototipe awal untuk mendukung fact-checking informasi lalu lintas secara lebih cepat dan terstruktur.
