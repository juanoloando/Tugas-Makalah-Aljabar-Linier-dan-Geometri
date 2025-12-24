# Analisis Matrix Kovarians Bursa Efek Indonesia (BEI) menggunakan PCA untuk Identifikasi Faktor Resiko dan Optimasi Portofolio
Proyek ini merupakan penerapan nilai eigen dan vektor eigen dalam analisis menggunakan metode PCA terhadap matriks kovarians Bursa Efek Indonesia guna mengidentifikasi faktor-fakotr risiko utama dan optimasi portofolio.

## Deskripsi Umum
Program ini bertujuan untuk menganalisis struktur risiko saham di Bursa Efek Indonesia(BEI) menggunakan metode PCA terhadap matriks kovarians return saham.  Pendekatan ini menggunakan konsep aljabar linier berupa nilai eigen, vektor eigen, dan kovarians. Seluruh dataset diambil dari website Bursa Efek Indonesia dan implementasi menggunakan Python dengan requirements:

* Python 3.x
* NumPy >=1.24.0
* Matplotlib >=3.7.0
* Seaborn >=0.12.0
* Scikit-learn >=1.3.0

## Struktur Data
Dataset yang digunakan berupa data historis saham dalam format CSV dengan struktur minimal sebagai berikut:

| Kolom  | Deskripsi              |
|-------|------------------------|
| Date  | Tanggal perdagangan    |
| Close | Harga penutupan saham  |

Program hanya menggunakan kolom Date dan Close. Data harga penutupan kemudian dikonversi menjadi return harian sebagai dasar analisis risiko.

## Output Program
Program menghasilkan beberapa output utama:
1. Scree plot komponen utama
2. Grafik loading PC1
3. Heatmap loading PC1–PC3
4. Grafik performa eigen-portfolio
