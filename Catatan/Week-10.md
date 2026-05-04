# Week 10 — Catatan Deep Learning, Machine Learning, MLP, dan CNN

## 1. Perbedaan Fundamental ML dan DL

### Machine Learning (ML)

Pada **Machine Learning tradisional**, model biasanya membutuhkan **fitur yang sudah disiapkan atau diekstrak terlebih dahulu** oleh manusia.

Artinya, sebelum data masuk ke model, kita harus menentukan fitur apa yang penting.

Contoh pada data tabular:

| Data | Fitur |
|---|---|
| Data rumah | luas tanah, jumlah kamar, lokasi, harga |
| Data pasien | umur, tekanan darah, kolesterol, riwayat penyakit |
| Data transaksi | jumlah transaksi, waktu transaksi, lokasi, jenis barang |

Karena data tabular sudah berbentuk kolom-kolom fitur, model ML lebih mudah digunakan pada data seperti ini.

Contoh algoritma ML:

- Linear Regression
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- SVM
- KNN

### Deep Learning (DL)

Pada **Deep Learning**, model bisa melakukan **feature extraction** dan **prediction/classification** dalam satu proses.

Alurnya:

```text
Input data mentah → model mengekstrak fitur → model melakukan prediksi/classification → output
```

Contoh:

```text
Input gambar kucing → model mencari pola seperti tepi, bentuk mata, telinga, tekstur bulu → output: kucing
```

Jadi, DL cocok untuk data mentah yang kompleks seperti:

- Gambar
- Video
- Audio
- Teks
- Sinyal sensor

## 2. Kapan Menggunakan ML dan DL?

| Kondisi | Lebih Cocok ML | Lebih Cocok DL |
|---|---|---|
| Data kecil | Ya | Tidak selalu |
| Data besar | Bisa, tetapi terbatas | Ya |
| Data tabular/terstruktur | Ya | Bisa, tetapi tidak selalu lebih baik |
| Data gambar/video/audio/text mentah | Kurang cocok tanpa feature extraction manual | Sangat cocok |
| Butuh interpretasi model | Biasanya lebih mudah | Biasanya lebih sulit |
| Pola sangat kompleks | Bisa kurang kuat | Lebih kuat |
| Komputasi terbatas | Lebih ringan | Lebih berat |

Kesimpulan:

- **Data kecil + tabular → ML biasanya lebih direkomendasikan.**
- **Data besar + kompleks seperti image/text/audio → DL biasanya lebih unggul.**

Namun, jangan langsung menganggap DL selalu lebih baik. DL butuh data lebih banyak, komputasi lebih besar, dan tuning yang lebih rumit.

## 3. Cara Kerja Deep Learning Secara Umum

Deep Learning bekerja dengan proses berulang:

```text
Input → Forward Propagation → Prediksi → Hitung Loss/Error → Backpropagation → Update Bobot → Ulangi
```

### Penjelasan Singkat

1. **Forward propagation**
   - Data masuk dari input layer.
   - Data melewati hidden layer.
   - Model menghasilkan prediksi di output layer.

2. **Loss/Error**
   - Model membandingkan hasil prediksi dengan nilai asli/label asli.
   - Jika prediksi salah, loss akan besar.
   - Jika prediksi mendekati label asli, loss akan kecil.

3. **Backpropagation**
   - Error dari loss dikirim kembali ke layer sebelumnya.
   - Tujuannya untuk menghitung seberapa besar kontribusi setiap bobot terhadap error.
   - Dari sini model menghitung gradient.

4. **Parameter update**
   - Bobot dan bias diperbarui menggunakan optimizer.
   - Contoh optimizer: Gradient Descent, SGD, Adam, RMSprop.

5. **Epoch**
   - Satu epoch berarti model sudah melihat seluruh data training sebanyak satu kali.
   - Training dilakukan beberapa epoch supaya model belajar secara bertahap.

## 4. Pertanyaan: Kenapa Hasil Run Bisa Berbeda?

Kalimat yang sudah diperbaiki:

> Kadang hasil run bisa berbeda karena bobot awal model diinisialisasi secara acak, lalu diperbarui melalui proses forward propagation dan backpropagation.

Penjelasan tambahan:

Hasil training bisa berbeda antar-run karena beberapa hal:

- Bobot awal model biasanya random.
- Urutan data training bisa diacak, terutama jika menggunakan shuffle.
- Dropout, data augmentation, atau proses lain bisa memiliki unsur random.
- GPU computation kadang memiliki sedikit perbedaan numerik.

Agar hasil lebih konsisten, biasanya digunakan **random seed**.

Contoh:

```python
import torch
import random
import numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
```

Namun, meskipun memakai seed, hasil bisa tetap sedikit berbeda tergantung device, library, dan konfigurasi training.

## 5. Layer dalam Deep Learning

Secara umum, neural network terdiri dari:

```text
Input Layer → Hidden Layer → Output Layer
```

## 6. Input Layer

**Input layer** adalah tempat data pertama kali dimasukkan ke model.

Pada input layer belum terjadi proses learning yang kompleks. Layer ini lebih berfungsi sebagai tempat masuknya data.

Jumlah neuron input biasanya mengikuti jumlah fitur input.

### Contoh Input Tabular

Misalnya data rumah punya fitur:

```text
luas rumah, jumlah kamar, lokasi, umur bangunan
```

Maka jumlah input bisa menjadi 4 fitur.

### Contoh Input Gambar Grayscale

Gambar ukuran `6 × 6` grayscale:

```text
6 × 6 = 36 pixel
```

Jika dimasukkan ke MLP, bisa dianggap sebagai 36 input/neuron.

### Contoh Input Gambar RGB

Gambar ukuran `6 × 6` RGB:

```text
6 × 6 × 3 = 108 nilai input
```

Angka `3` berasal dari tiga channel warna:

- Red
- Green
- Blue

Jadi bukan “dikali 3 karena ada 3 channel” saja, tetapi karena setiap pixel punya 3 nilai warna.

## 7. Hidden Layer

**Hidden layer** adalah layer di antara input dan output.

Di sinilah proses learning utama terjadi.

Pada hidden layer terdapat parameter yang dipelajari model, yaitu:

- Weight/bobot
- Bias

Secara sederhana, neuron menghitung:

```text
output = input × weight + bias
```

Dalam bentuk lebih umum:

```text
z = w1x1 + w2x2 + w3x3 + ... + b
```

Setelah itu biasanya hasilnya masuk ke activation function.

Jumlah hidden layer tidak selalu tetap. Banyaknya layer dan neuron bergantung pada:

- Ukuran dataset
- Kompleksitas masalah
- Jenis data
- Risiko overfitting
- Kapasitas komputasi
- Referensi dari penelitian atau model dengan kasus serupa

Tips praktis:

> Untuk menentukan jumlah hidden layer, mulai dari arsitektur sederhana atau cari referensi dari kasus yang mirip, lalu lakukan eksperimen.

## 8. Activation Function

**Activation function** digunakan agar model tidak hanya belajar hubungan linear.

Tanpa activation function, banyak layer linear yang ditumpuk tetap hanya akan menjadi fungsi linear. Akibatnya, model sulit menangkap pola kompleks.

Activation function membantu model menangkap pola non-linear seperti:

- Lengkungan
- Batas keputusan yang kompleks
- Pola gambar
- Hubungan antarfitur yang tidak sederhana

## 9. ReLU

**ReLU** adalah activation function yang paling umum digunakan di hidden layer.

Rumus:

```text
ReLU(x) = max(0, x)
```

Artinya:

- Jika nilai input kurang dari 0, output menjadi 0.
- Jika nilai input lebih dari 0, output mengikuti nilai aslinya.

Contoh:

| Input | Output ReLU |
|---:|---:|
| -5 | 0 |
| -1 | 0 |
| 0 | 0 |
| 3 | 3 |
| 10 | 10 |

Kelebihan ReLU:

- Sederhana
- Cepat dihitung
- Umumnya bekerja baik pada banyak model DL

Kekurangan ReLU:

- Bisa mengalami **dying ReLU**, yaitu neuron menghasilkan 0 terus dan sulit belajar lagi.

## 10. Leaky ReLU

**Leaky ReLU** adalah variasi ReLU.

Bedanya, nilai negatif tidak langsung dibuat 0, tetapi dibuat menjadi nilai negatif kecil.

Rumus sederhana:

```text
LeakyReLU(x) = x, jika x > 0
LeakyReLU(x) = αx, jika x <= 0
```

Biasanya `α = 0.01`.

Contoh:

| Input | ReLU | Leaky ReLU, α=0.01 |
|---:|---:|---:|
| -5 | 0 | -0.05 |
| -1 | 0 | -0.01 |
| 0 | 0 | 0 |
| 3 | 3 | 3 |

Leaky ReLU membantu mengurangi risiko dying ReLU karena bagian negatif tetap punya gradient kecil.

## 11. Sigmoid

**Sigmoid** menghasilkan output antara 0 dan 1.

Rumus:

```text
Sigmoid(x) = 1 / (1 + e^(-x))
```

Biasanya digunakan untuk **binary classification**, terutama jika output hanya satu neuron.

Contoh kasus:

```text
0 = bukan spam
1 = spam
```

Output sigmoid bisa diinterpretasikan sebagai probabilitas.

Contoh:

```text
output = 0.87
```

Artinya model cukup yakin bahwa data masuk ke kelas 1.

## 12. Softmax

**Softmax** biasanya digunakan untuk **multi-class classification**, bukan hanya dua label.

Contoh klasifikasi 3 kelas:

```text
kucing, anjing, burung
```

Output Softmax:

| Kelas | Probabilitas |
|---|---:|
| Kucing | 0.70 |
| Anjing | 0.20 |
| Burung | 0.10 |

Total probabilitasnya = 1.

Softmax juga bisa digunakan untuk 2 kelas, tetapi untuk binary classification sederhana, sigmoid dengan 1 output neuron sering lebih praktis.

Koreksi penting:

> Sigmoid biasanya untuk binary classification. Softmax biasanya untuk multi-class classification.

## 13. Output Layer

**Output layer** menghasilkan prediksi akhir model.

Jumlah neuron output bergantung pada jenis masalah.

### Binary Classification

Contoh: spam atau bukan spam.

Biasanya:

```text
1 neuron output + sigmoid
```

Output:

```text
0.0 sampai 1.0
```

### Multi-class Classification

Contoh: kucing, anjing, burung.

Biasanya:

```text
jumlah neuron output = jumlah kelas
```

Jika ada 3 kelas, output layer punya 3 neuron.

Biasanya menggunakan Softmax.

### Regression

Contoh: prediksi harga rumah.

Biasanya:

```text
1 neuron output tanpa sigmoid/softmax
```

Output bisa berupa angka bebas, misalnya:

```text
750000000
```

## 14. MLP — Multi-Layer Perceptron

**MLP** bisa dianggap sebagai bentuk dasar atau vanilla deep learning.

Disebut vanilla karena tidak memiliki layer khusus seperti convolution layer pada CNN atau recurrent layer pada RNN.

Struktur MLP:

```text
Input Layer → Hidden Layer → Hidden Layer → Output Layer
```

MLP biasanya cocok untuk:

- Data tabular
- Data fitur yang sudah berbentuk vektor
- Masalah sederhana sampai menengah

MLP kurang ideal untuk gambar mentah karena gambar memiliki struktur spasial. Jika gambar langsung di-flatten ke MLP, informasi lokasi pixel bisa kurang dimanfaatkan.

Contoh:

```text
Gambar 28 × 28 → flatten jadi 784 angka → masuk ke MLP
```

Model masih bisa belajar, tetapi biasanya CNN lebih efektif untuk gambar.

## 15. Proses Training pada MLP

Alur training:

```text
Forward Propagation → Loss Calculation → Backpropagation → Parameter Update
```

### Forward Propagation

Data bergerak dari input sampai output.

```text
Input → Hidden Layer → Activation Function → Output → Prediction
```

### Loss Calculation

Loss mengukur seberapa jauh prediksi model dari label asli.

Contoh:

```text
Label asli: kucing
Prediksi model: anjing
Loss: besar
```

Jika prediksi benar dan yakin:

```text
Label asli: kucing
Prediksi model: kucing dengan probabilitas tinggi
Loss: kecil
```

### Backpropagation

Backpropagation menghitung gradient dari loss terhadap parameter model.

Gradient memberi tahu arah perubahan bobot agar loss mengecil.

### Parameter Update

Parameter seperti weight dan bias diperbarui menggunakan optimizer.

Contoh sederhana Gradient Descent:

```text
weight_baru = weight_lama - learning_rate × gradient
```

Jika learning rate terlalu besar, model bisa tidak stabil.
Jika learning rate terlalu kecil, training bisa sangat lambat.

## 16. Epoch, Batch, dan Iteration

Ini bagian yang sering tertukar.

### Epoch

Satu epoch berarti model sudah melihat seluruh data training satu kali.

Contoh:

```text
Dataset punya 1000 data
1 epoch = model memproses semua 1000 data tersebut satu kali
```

### Batch

Batch adalah sebagian data yang diproses dalam satu langkah training.

Contoh:

```text
Dataset = 1000 data
Batch size = 100
```

Maka setiap batch berisi 100 data.

### Iteration

Iteration adalah satu kali proses update parameter berdasarkan satu batch.

Dengan contoh di atas:

```text
1000 data / batch size 100 = 10 iteration per epoch
```

Jadi:

```text
1 epoch = 10 iteration
```

Koreksi penting:

> Epoch bukan sekadar “supaya loop tidak terus-terusan”. Epoch adalah jumlah putaran model melihat seluruh dataset. Training berhenti berdasarkan jumlah epoch, early stopping, atau kondisi tertentu.

## 17. Model yang Bagus

Model yang bagus bukan hanya model yang bagus di data training.

Model yang bagus adalah model yang:

- Loss training menurun.
- Akurasi atau metric utama membaik.
- Performa pada validation/test data juga bagus.
- Tidak hanya menghafal data training.
- Bisa melakukan generalisasi pada data baru.

Jika training accuracy tinggi tetapi validation accuracy rendah, model kemungkinan mengalami **overfitting**.

## 18. CNN — Convolutional Neural Network

**CNN** adalah jenis deep learning yang sangat sering digunakan untuk data gambar.

CNN cocok untuk:

- Image classification
- Object detection
- Face recognition
- Medical image analysis
- Computer vision task lainnya

Perbedaan penting dengan MLP:

- MLP biasanya membaca gambar sebagai vektor panjang.
- CNN membaca gambar dalam bentuk kelompok pixel menggunakan filter/kernel.

Karena itu, CNN lebih mampu menangkap struktur lokal pada gambar, seperti garis, sudut, tekstur, dan bentuk.

## 19. Kenapa CNN Lebih Cocok untuk Gambar?

Gambar memiliki struktur spasial.

Pixel yang berdekatan biasanya saling berhubungan.

Contoh:

```text
Pixel mata, hidung, dan mulut memiliki posisi tertentu pada wajah.
```

Jika gambar langsung di-flatten ke MLP, sebagian informasi spasial bisa hilang.

CNN mempertahankan struktur gambar dengan convolution layer.

## 20. Struktur Umum CNN

Struktur umum CNN:

```text
Input Image → Convolutional Layer → Activation Function → Pooling Layer → Flatten/GAP → Fully Connected Layer → Output
```

Contoh:

```text
Gambar kucing
→ convolution mencari garis dan tekstur
→ activation function memberi non-linearitas
→ pooling mengecilkan ukuran feature map
→ flatten/GAP mengubah fitur agar siap masuk classifier
→ fully connected layer membuat keputusan akhir
→ output: kucing
```

## 21. Convolutional Layer

**Convolutional layer** bertugas mengekstrak fitur dari gambar.

Layer ini menggunakan filter/kernel yang digeser ke seluruh bagian gambar.

Contoh fitur yang bisa dipelajari:

- Garis horizontal
- Garis vertikal
- Lengkungan
- Sudut
- Tekstur
- Pola bentuk

Pada layer awal, CNN biasanya belajar fitur sederhana seperti garis dan tepi.
Pada layer lebih dalam, CNN bisa belajar fitur yang lebih kompleks seperti mata, telinga, roda, wajah, atau bentuk objek.

## 22. Filter atau Kernel

**Filter/kernel** adalah matriks kecil yang digunakan untuk membaca bagian kecil dari gambar.

Contoh kernel 3 × 3:

```text
[ 1  0 -1 ]
[ 1  0 -1 ]
[ 1  0 -1 ]
```

Pada image processing tradisional atau ML klasik, filter bisa berupa template buatan manusia, misalnya filter sharpen, blur, atau edge detection.

Pada Deep Learning, nilai filter biasanya **dipelajari oleh model** selama training.

Jadi bukan kita yang menentukan semua isi filternya secara manual.

## 23. Stride

**Stride** adalah jumlah langkah pergeseran filter/kernel saat membaca gambar.

Jika stride = 1:

```text
Filter bergeser 1 pixel setiap langkah
```

Jika stride = 2:

```text
Filter bergeser 2 pixel setiap langkah
```

Efek stride:

- Stride kecil → feature map lebih besar, informasi lebih detail.
- Stride besar → feature map lebih kecil, proses lebih cepat, tetapi detail bisa berkurang.

Biasanya stride yang umum digunakan adalah 1 atau 2.

## 24. Padding

**Padding** adalah penambahan pixel di sekitar gambar.

Tujuannya:

- Menjaga ukuran feature map agar tidak terlalu cepat mengecil.
- Membantu informasi di bagian tepi/pojok gambar tidak cepat hilang.

Contoh tanpa padding:

```text
Input 5 × 5, kernel 3 × 3 → output 3 × 3
```

Contoh dengan padding:

```text
Input 5 × 5 + padding → output bisa tetap 5 × 5
```

Padding biasanya diisi dengan 0, sehingga sering disebut **zero padding**.

## 25. Feature Map

**Feature map** adalah hasil dari proses convolution.

Feature map menunjukkan lokasi fitur tertentu yang berhasil dideteksi oleh filter.

Contoh:

- Satu filter mendeteksi garis horizontal.
- Filter lain mendeteksi garis vertikal.
- Filter lain mendeteksi tekstur tertentu.

Jika ada banyak filter, maka akan ada banyak feature map.

Contoh:

```text
Input image → Conv layer dengan 32 filter → menghasilkan 32 feature map
```

Jika CNN punya banyak convolutional layer, output dari convolutional layer sebelumnya menjadi input untuk convolutional layer berikutnya.

## 26. Pooling Layer

**Pooling layer** melakukan downsampling pada feature map.

Tujuannya:

- Mengecilkan ukuran feature map.
- Mengurangi jumlah parameter/komputasi.
- Mengambil informasi paling penting.
- Membuat model lebih tahan terhadap sedikit pergeseran posisi objek.

Jenis pooling yang umum:

- Max Pooling
- Average Pooling
- Min Pooling, lebih jarang dipakai

## 27. Contoh Max Pooling

Misalnya ada feature map 4 × 4:

```text
[ 1  3  2  4 ]
[ 5  6  1  2 ]
[ 7  2  8  1 ]
[ 3  4  5  9 ]
```

Jika menggunakan max pooling 2 × 2 dengan stride 2, kita ambil nilai terbesar dari setiap area 2 × 2.

Area pertama:

```text
[ 1  3 ]
[ 5  6 ]
```

Nilai maksimum = 6.

Area kedua:

```text
[ 2  4 ]
[ 1  2 ]
```

Nilai maksimum = 4.

Area ketiga:

```text
[ 7  2 ]
[ 3  4 ]
```

Nilai maksimum = 7.

Area keempat:

```text
[ 8  1 ]
[ 5  9 ]
```

Nilai maksimum = 9.

Hasil max pooling:

```text
[ 6  4 ]
[ 7  9 ]
```

## 28. Contoh Average Pooling

Dengan feature map yang sama:

```text
[ 1  3  2  4 ]
[ 5  6  1  2 ]
[ 7  2  8  1 ]
[ 3  4  5  9 ]
```

Average pooling mengambil rata-rata dari setiap area 2 × 2.

Area pertama:

```text
[ 1  3 ]
[ 5  6 ]
```

Rata-rata:

```text
(1 + 3 + 5 + 6) / 4 = 3.75
```

Hasil average pooling:

```text
[ 3.75  2.25 ]
[ 4.00  5.75 ]
```

## 29. Contoh Min Pooling

Min pooling mengambil nilai terkecil dari setiap area.

Dengan feature map yang sama:

```text
[ 1  3  2  4 ]
[ 5  6  1  2 ]
[ 7  2  8  1 ]
[ 3  4  5  9 ]
```

Hasil min pooling 2 × 2 stride 2:

```text
[ 1  1 ]
[ 2  1 ]
```

Catatan:

> Dalam praktik CNN, max pooling lebih sering dipakai dibanding min pooling.

## 30. Flatten Layer

**Flatten layer** mengubah feature map multidimensi menjadi vektor satu dimensi.

Contoh:

```text
Feature map 4 × 4 × 32 → Flatten → vektor panjang 512
```

Karena:

```text
4 × 4 × 32 = 512
```

Flatten diperlukan jika fitur dari CNN akan dimasukkan ke fully connected layer.

## 31. Fully Connected Layer

**Fully connected layer** adalah layer di mana setiap neuron terhubung ke semua neuron dari layer sebelumnya.

Nama lain:

```text
Dense layer
```

Dalam CNN, fully connected layer biasanya digunakan di bagian akhir untuk menggabungkan fitur dan membuat keputusan klasifikasi.

Alur umum:

```text
Convolution + Pooling → Flatten → Fully Connected → Output
```

Contoh:

```text
Fitur garis + fitur mata + fitur telinga + fitur tekstur bulu → Fully Connected → Prediksi: kucing
```

## 32. GAP — Global Average Pooling

**GAP** adalah singkatan dari **Global Average Pooling**.

GAP mengambil rata-rata dari setiap feature map sehingga setiap feature map menjadi satu nilai.

Contoh:

Jika ada feature map 4 × 4:

```text
[ 1  3  2  4 ]
[ 5  6  1  2 ]
[ 7  2  8  1 ]
[ 3  4  5  9 ]
```

Maka GAP menghitung rata-rata semua nilainya:

```text
(1+3+2+4+5+6+1+2+7+2+8+1+3+4+5+9) / 16 = 3.94
```

Jadi satu feature map menjadi satu angka.

Jika ada 32 feature map:

```text
32 feature map → GAP → 32 nilai
```

GAP sering digunakan sebagai alternatif Flatten + Fully Connected yang terlalu besar.

Kelebihan GAP:

- Mengurangi jumlah parameter.
- Mengurangi risiko overfitting.
- Lebih ringan secara komputasi.

## 33. Flatten vs GAP

| Aspek | Flatten | GAP |
|---|---|---|
| Cara kerja | Mengubah semua nilai feature map menjadi vektor panjang | Merata-ratakan setiap feature map menjadi satu nilai |
| Jumlah fitur hasil akhir | Bisa sangat besar | Lebih kecil |
| Risiko overfitting | Lebih tinggi | Lebih rendah |
| Cocok untuk | Arsitektur sederhana atau klasik | Arsitektur CNN modern |

Contoh:

```text
Feature map 7 × 7 × 512
```

Jika Flatten:

```text
7 × 7 × 512 = 25088 fitur
```

Jika GAP:

```text
512 feature map → 512 fitur
```

Perbedaannya sangat besar.

## 34. Contoh Alur CNN Sederhana

Misalnya input gambar RGB ukuran 32 × 32 × 3.

```text
Input: 32 × 32 × 3
↓
Conv Layer 1: deteksi garis/tepi sederhana
↓
ReLU
↓
Max Pooling: mengecilkan feature map
↓
Conv Layer 2: deteksi pola lebih kompleks
↓
ReLU
↓
Max Pooling
↓
Flatten atau GAP
↓
Fully Connected Layer
↓
Output: kelas gambar
```

Contoh output:

```text
Kelas: kucing, anjing, mobil
```

Jika pakai Softmax:

```text
kucing: 0.82
anjing: 0.13
mobil: 0.05
```

Maka prediksi akhir adalah:

```text
kucing
```

## 35. Ringkasan MLP vs CNN

| Aspek | MLP | CNN |
|---|---|---|
| Bentuk input | Vektor/fitur tabular | Gambar atau data spasial |
| Cara membaca gambar | Di-flatten menjadi vektor | Dibaca per area pixel dengan kernel/filter |
| Feature extraction | Tidak sekuat CNN untuk gambar | Kuat untuk gambar |
| Layer khusus | Fully connected layer | Convolution, pooling, flatten/GAP, fully connected |
| Cocok untuk | Tabular, fitur sederhana | Image, object, visual pattern |

## 36. Catatan Koreksi dari Versi Awal

Beberapa bagian yang perlu diluruskan:

### 1. “ML butuh feature extraction, DL tidak”

Lebih tepat:

> ML tradisional sering membutuhkan feature extraction manual. DL tetap melakukan feature extraction, tetapi prosesnya dipelajari otomatis oleh model.

### 2. “Softmax biasanya untuk data 2 label saja”

Kurang tepat.

Yang benar:

> Sigmoid biasanya untuk binary classification. Softmax biasanya untuk multi-class classification, meskipun bisa juga dipakai untuk 2 kelas.

### 3. “Epoch agar loop tidak terus-terusan”

Kurang tepat.

Yang benar:

> Epoch adalah jumlah putaran model melihat seluruh dataset. Training bisa dihentikan berdasarkan jumlah epoch, early stopping, atau kondisi tertentu.

### 4. “Update bobot bisa pakai gradient”

Lebih tepat:

> Update bobot dilakukan menggunakan optimizer, dan optimizer menggunakan gradient untuk menentukan arah update parameter.

### 5. “Fully connected layer gabungin fitur-fitur”

Benar, tetapi perlu dilengkapi:

> Fully connected layer menggabungkan fitur yang sudah diekstrak dan mengubahnya menjadi keputusan akhir, misalnya klasifikasi kelas.

## 37. Contoh Mini: Binary Classification

Kasus:

```text
Prediksi apakah email spam atau bukan spam
```

Output layer:

```text
1 neuron + sigmoid
```

Output:

```text
0.91
```

Interpretasi:

```text
Model memprediksi email kemungkinan besar spam
```

Jika threshold = 0.5:

```text
0.91 > 0.5 → spam
```

## 38. Contoh Mini: Multi-class Classification

Kasus:

```text
Klasifikasi gambar: kucing, anjing, burung
```

Output layer:

```text
3 neuron + softmax
```

Output:

```text
kucing: 0.10
anjing: 0.80
burung: 0.10
```

Prediksi akhir:

```text
anjing
```

## 39. Contoh Mini: Regression

Kasus:

```text
Prediksi harga rumah
```

Output layer:

```text
1 neuron tanpa sigmoid/softmax
```

Output:

```text
850000000
```

Artinya model memprediksi harga rumah sekitar 850 juta.

## 40. Kesimpulan Utama

- ML tradisional biasanya butuh fitur yang sudah disiapkan.
- DL bisa belajar fitur langsung dari data mentah.
- ML cocok untuk data kecil dan tabular.
- DL cocok untuk data besar dan kompleks seperti gambar, teks, audio, dan video.
- MLP adalah arsitektur neural network dasar.
- CNN cocok untuk gambar karena membaca data per kelompok pixel menggunakan convolution.
- Activation function membuat model mampu menangkap pola non-linear.
- ReLU umum digunakan di hidden layer.
- Sigmoid umum digunakan untuk binary classification.
- Softmax umum digunakan untuk multi-class classification.
- Backpropagation menghitung gradient untuk memperbaiki bobot.
- Optimizer menggunakan gradient untuk update parameter.
- Epoch, batch, dan iteration adalah konsep berbeda yang penting dipahami.
- Model yang bagus bukan hanya bagus di training data, tetapi juga bagus pada data baru.

## 41. Stop, Start, Continue

### Stop

- Stop menghafal istilah tanpa memahami alurnya.
- Stop menyamakan epoch dengan iteration.
- Stop menganggap Softmax hanya untuk dua label.
- Stop menganggap DL selalu lebih baik dari ML.

### Start

- Start menggambar alur model dari input sampai output.
- Start membedakan binary classification, multi-class classification, dan regression.
- Start mencatat ukuran input dan output setiap layer.
- Start mencari referensi arsitektur dari kasus yang mirip sebelum menentukan jumlah layer.

### Continue

- Continue membandingkan ML dan DL berdasarkan jenis data.
- Continue memakai contoh konkret seperti image, tabular, dan klasifikasi.
- Continue bertanya kenapa suatu layer dipakai, bukan hanya apa namanya.
- Continue memperbaiki catatan dengan bahasa sendiri agar lebih paham.
