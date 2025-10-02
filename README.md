Repository ini berisi seluruh kode sumber, data mentah/terproses, hasil eksperimen, dan model klasifikasi dari penelitian skripsi berjudul: [Masukkan Judul Skripsi Lengkap Anda di Sini, Contoh: Sistem Kontrol Drone Berbasis Steady-State Visual Evoked Potential (SSVEP) Menggunakan OpenBCI dan Klasifikasi Machine Learning].

Proyek ini berfokus pada pengembangan sistem Brain-Computer Interface (BCI) non-invasif yang memungkinkan pengguna mengontrol drone simulasi menggunakan sinyal otak yang dibangkitkan oleh stimulus visual berkedip (SSVEP).

Struktur Proyek Utama
Proyek ini terbagi menjadi beberapa fase eksperimen dan komponen utama yang tercermin dalam struktur directory ini:

Hasil Percobaan Penelitian Pertama/: Data EEG mentah (6 subjek), pre-processing, ekstraksi fitur (CCA, FFT, Statistikal), dan hasil klasifikasi Machine Learning (LDA, Random Forest, XGBoost) menggunakan stimulus HTML/CSS (6 Frekuensi).

Hasil Percobaan Penelitian Kedua/ & Hasil Percobaan Penelitian Ketiga/: Pengembangan dan pengujian stimulus visual presisi berbasis BrainFlow/Python (5 Frekuensi) untuk optimalisasi sinyal SSVEP, termasuk analisis FFT dan SNR.

MODEL YANG DI HASILKAN/: Model klasifikasi terbaik (model_rf.joblib, scaler.joblib) yang digunakan untuk implementasi real-time.

Drone Simulation/: Kode (index.html, script.js) untuk demonstrasi Proof-of-Concept simulasi kontrol drone berbasis web yang menerima input dari hasil klasifikasi EEG.

Teknologi dan Tools
Akuisisi Data: OpenBCI Cyton & BrainFlow

Stimulus Visual: Python (PsychoPy/BrainFlow), HTML/CSS

Pemrosesan/Klasifikasi: Python, NumPy, Pandas, Scikit-learn (LDA, Random Forest), XGBoost, MNE

Kontributor
Penulis/Peneliti: Usamah Ikhwana Fadhlih

Pembimbing: Bpk. Eko
