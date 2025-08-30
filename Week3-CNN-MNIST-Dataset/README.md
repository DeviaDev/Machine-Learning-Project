# Tugas Kelompok 3 – CNN pada Dataset MNIST

## Deskripsi Proyek
Proyek ini bertujuan untuk mengimplementasikan model Convolutional Neural Network (CNN) untuk mengklasifikasikan gambar angka tulisan tangan dari dataset MNIST.

## Dataset
Dataset yang digunakan adalah **MNIST** (Modified National Institute of Standards and Technology), yang berisi 60.000 data latih dan 10.000 data uji berupa gambar grayscale 28x28 piksel dari angka 0 hingga 9.

## Arsitektur Model CNN
Model CNN yang dibangun memiliki struktur sebagai berikut:

- **Input Layer**: Menerima gambar 28x28 piksel (grayscale).
- **Convolutional Layer**: Menggunakan filter untuk mengekstraksi fitur dari gambar.
- **Activation Function (ReLU)**: Menambahkan non-linearitas ke model.
- **Pooling Layer (MaxPooling)**: Mengurangi dimensi fitur untuk mengurangi kompleksitas komputasi.
- **Dropout Layer**: Mencegah overfitting dengan mengabaikan beberapa neuron secara acak selama pelatihan.
- **Flatten Layer**: Mengubah matriks fitur 2D menjadi vektor 1D.
- **Fully Connected Layer (Dense)**: Menghubungkan semua neuron untuk klasifikasi akhir.
- **Output Layer (Softmax)**: Menghasilkan probabilitas untuk setiap kelas digit (0-9).

## Proses Pelatihan
- **Data Preprocessing**: Normalisasi data agar nilai piksel berada dalam rentang 0 hingga 1.
- **Loss Function**: Menggunakan categorical cross-entropy.
- **Optimizer**: Menggunakan Adam optimizer untuk mempercepat konvergensi.
- **Epochs**: Model dilatih selama beberapa epoch untuk mencapai akurasi optimal.
- **Batch Size**: Ukuran batch yang digunakan selama pelatihan.

## Hasil dan Evaluasi
Model CNN berhasil mencapai akurasi tinggi dalam mengenali digit tangan pada dataset MNIST. Evaluasi dilakukan menggunakan data uji untuk memastikan model tidak overfitting.

## Visualisasi
Visualisasi hasil pelatihan dan evaluasi model, seperti grafik akurasi dan loss, serta contoh prediksi pada data uji, dapat dilihat pada notebook `CNN_MNIST.ipynb`.

## Anggota Kelompok
- Devianest Narendra
- Zainab Ahmad
- Adya Rusmalillah
- Naila Fatikhah
- Nurul Khoiriyah

## Catatan
Untuk detail implementasi dan hasil lebih lanjut, silakan lihat notebook `CNN_MNIST.ipynb` dalam folder `Week3-CNN-MNIST-Dataset`.
