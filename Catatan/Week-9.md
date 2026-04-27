
# Catatan Kelas NLP dan RNN, LSTM, GRU

## 1. Sequential Learning dan RNN
- **Sequential Learning** adalah proses belajar menggunakan data berurutan. Model yang digunakan harus bisa meng-handle urutan data, seperti kalimat, waktu, atau rangkaian data lainnya.
  
  - **RNN (Recurrent Neural Network)** adalah model yang cocok untuk data berurutan karena bisa mengingat informasi dari langkah sebelumnya dan memanfaatkannya untuk memproses data berikutnya.

## 2. Perbedaan RNN dengan CNN
- **CNN (Convolutional Neural Network)** digunakan untuk gambar. Inputnya berupa matriks pixel gambar.
- **RNN** digunakan untuk data urutan (sequential data). Inputnya adalah satu elemen dari sebuah sequence pada tiap langkah waktu.

## 3. Representasi Kata dalam RNN (Embedding Layer)
- Kata dalam RNN tidak langsung dimasukkan sebagai teks mentah, melainkan diubah menjadi **vektor representasi** (embedding).
  
  Contoh:
  
  | Kata | Vektor Representasi |
  |------|---------------------|
  | Saya | [0.2, -0.1, 0.7]    |
  | belajar | [0.5, 0.3, 0.1] |
  | AI    | [0.9, -0.4, 0.6]   |
  
  Vektor-vektor ini yang masuk ke RNN untuk diproses.

## 4. Macam-macam RNN
- **One-to-many:** Input satu, output banyak. Contoh: training gambar untuk menghasilkan deskripsi (misalnya gambar bunga, output "bunga mawar").
- **Many-to-one:** Banyak input, satu output. Contoh: analisis sentimen dari tweet (positif atau negatif).
- **Many-to-many (linear):** Banyak input, banyak output dengan hubungan satu per satu.
- **Many-to-many (seq2seq):** Seperti penerjemahan bahasa atau prediksi kata berikutnya.

## 5. Kelemahan RNN
- **Vanishing Gradient:** Ketika urutan data sangat panjang, informasi dari langkah sebelumnya semakin hilang. Misalnya pada kalimat panjang, informasi dari kata yang sangat awal bisa hilang.
  - Contoh: jika ada nilai yang lebih kecil dari 1, perkalian berulang-ulang akan menghasilkan nilai yang semakin kecil (mendekati nol).
  
- **Exploding Gradient:** Sebaliknya, ketika gradien menjadi terlalu besar dan membuat pembaruan bobot menjadi tidak stabil.

## 6. LSTM (Long Short-Term Memory)
- LSTM adalah modifikasi dari RNN yang dirancang untuk mengatasi masalah **Vanishing Gradient** dan **Exploding Gradient**.
  
  **Rumus LSTM**:
  
  - `ht = tanh(W_hh * ht-1 + W_xh * xt)`
  
  - LSTM menggunakan **tiga gate**:
    - **Forget gate:** Menentukan informasi yang harus dibuang.
    - **Input gate:** Menambahkan informasi baru ke memori jangka panjang.
    - **Output gate:** Mengeluarkan informasi yang akan masuk ke hidden state berikutnya.
  
  LSTM juga menggunakan dua jenis memori:
  - **Memori jangka panjang (Ct)**
  - **Memori jangka pendek (ht)**

  **Bi-LSTM (Bidirectional LSTM):** LSTM yang membaca data dari dua arah: maju dan mundur. Hal ini memungkinkan model untuk menangkap konteks dari kedua arah dalam urutan.

## 7. GRU (Gated Recurrent Unit)
- GRU adalah versi lebih efisien dari LSTM yang hanya menggunakan **dua gate**:
  - **Update gate (zt):** Mengontrol informasi yang perlu disimpan.
  - **Reset gate (rt):** Mengontrol informasi yang perlu dibuang.
  
  GRU tidak memiliki memori jangka panjang terpisah seperti LSTM, sehingga lebih sederhana dan lebih cepat.

## 8. Tahapan Pre-Processing Text
- **Stemming:** Mengubah kata ke bentuk dasarnya. Contoh: "dimakan" menjadi "makan".
- **Lemmatization:** Mirip dengan stemming, tetapi menggunakan teknik yang lebih kompleks dan memeriksa kata dalam konteks.
- **Tokenization:** Memotong teks menjadi unit terkecil seperti kata atau frasa.
- **Stopwords Filtering:** Menghilangkan kata-kata yang tidak memiliki banyak arti, seperti "di", "ke", "yang", dll.
- **Text Cleaning:** Menghapus noise atau karakter yang tidak perlu seperti tanda baca atau simbol yang tidak relevan.

## 9. Train and Test Split
- **Split Data:** Memisahkan data menjadi dua bagian, yaitu data untuk training dan data untuk testing.
- **Padding:** Menambahkan token ekstra pada input yang lebih pendek supaya semua input memiliki panjang yang seragam.

## 10. Evaluasi Model
- Penting untuk mengevaluasi perbandingan antara training dan validation untuk memeriksa overfitting atau underfitting.
- **Imbalance label:** Pastikan rasio label dalam training dan testing set seimbang untuk menghindari bias.

---

Notebook yang bisa dibuat belajar : [Kaggle - NLP Beginner: Text Classification using LSTM](https://www.kaggle.com/code/arunrk7/nlp-beginner-text-classification-using-lstm/notebook).

