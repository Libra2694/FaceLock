# 🧑‍💼 FaceLock - Sistem Absensi Wajah

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

FaceLock adalah aplikasi absensi otomatis berbasis **pengenalan wajah** menggunakan **Python (OpenCV + Tkinter)** dan **MySQL Database**.  
Dengan FaceLock, proses absensi karyawan/mahasiswa dapat dilakukan **lebih cepat, akurat, dan modern** hanya dengan mendeteksi wajah.

---

## ✨ Fitur Utama
- 📷 Absensi Wajah **(Masuk & Keluar)** menggunakan kamera.
- 💾 Menyimpan data kehadiran ke database MySQL.
- 📊 Dashboard absensi interaktif:
  - Total user terdaftar.
  - Rekap masuk & keluar harian.
  - Tabel absensi (dengan filter nama, status, tanggal).
- 🔍 Filter data absensi berdasarkan nama, status, atau tanggal.
- 🚫 Tombol **Stop Scan** (ESC / tombol GUI) untuk menghentikan kamera.
- 🎨 Tampilan GUI sederhana dan user-friendly.

---

## 🛠️ Teknologi yang Digunakan
- **Python 3.x**
- **OpenCV** → pengolahan citra & pengenalan wajah
- **Tkinter** → Graphical User Interface (GUI)
- **MySQL** → database absensi
- **Numpy** → olah data numerik

---

## 📂 Struktur Folder
```bash
facelock/
│── dataset/              # Dataset wajah (ignored by git)
│── trainer/              # Model hasil training (ignored by git)
│── dataset_capture.py    # Script pengambilan dataset wajah
│── trainer.py            # Script training dataset
│── final.py              # Main program (GUI absensi)
│── names.txt             # List nama user (berurutan dengan ID training)
│── requirements.txt      # List dependencies
│── README.md             # Dokumentasi
│── LICENSE
│── .gitignore

```
---

## ⚙️ Instalasi & Setup
1. Clone Repository
```bash
git clone https://github.com/username/facelock.git
cd facelock
```
2. Instal Dependencies
```bash
pip install opencv-contrib-python mysql-connector-python numpy
```
atau cukup run:
```bash
pip install -r requirements.txt
```
3. Setup Database
Buat database absensi di MySQL:
```bash
CREATE DATABASE absensi;
USE absensi;

CREATE TABLE kehadiran (
    id INT AUTO_INCREMENT PRIMARY KEY,
    nama VARCHAR(100),
    waktu DATETIME,
    status ENUM('Masuk', 'Keluar')
);
```
4. Ambil Dataset Wajah
Run:
```bash
python dataset_capture.py
```
→ Masukkan Nama → Kamera aktif → Ambil beberapa sampel wajah.
5. Training Wajah
```bash
python trainer.py
```
→ Hasil model akan disimpan di trainer/face_trainer.yml.
6. Run Aplikasi
```bash
python final.py
```

## 📸 Preview
Dashboard
![Dashboard](https://res.cloudinary.com/dzgxqfnv9/image/upload/v1755324313/imgtourl/Screenshot_from_2025-08-16_13-51-19_fsiwvm.png)

## 👩‍💻 Kontributor
| Nama   |    Peran       |
|--------|----------------|
| Libra  | Developer Utama|
| Aisyah | AI assistant ✨|

---

## 📄 Lisensi

Proyek ini dirilis di bawah lisensi [MIT License](LICENSE).  
Bebas digunakan, dimodifikasi, dan didistribusikan selama mencantumkan copyright.

Semoga membantu! 😊

MIT License - © 2025 Libra

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## ✉️ Kontak
- 📧 Email: [Libra](mailto:libraproject26@gmail.com)
- 💬 Telegram: [Libra](https://t.me/libra_id26)