import cv2
import numpy as np
from PIL import Image
import os

# === Path ke dataset ===
dataset_path = 'data'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# === Fungsi baca semua gambar dari subfolder ===
def get_images_and_labels(path):
    face_samples = []
    ids = []

    for user_folder in os.listdir(path):
        user_path = os.path.join(path, user_folder)
        if not os.path.isdir(user_path):
            continue  # Lewati file selain folder

        for filename in os.listdir(user_path):
            if not filename.endswith('.jpg'):
                continue
            image_path = os.path.join(user_path, filename)
            img = Image.open(image_path).convert('L')  # Grayscale
            img_np = np.array(img, 'uint8')

            try:
                id = int(filename.split('.')[1])  # User.<id>.<count>.jpg
            except:
                print(f"[WARNING] Gagal ambil ID dari nama file: {filename}")
                continue

            faces = detector.detectMultiScale(img_np)
            for (x, y, w, h) in faces:
                face_samples.append(img_np[y:y+h, x:x+w])
                ids.append(id)

    return face_samples, ids

print("[INFO] Training wajah. Mohon tunggu...")
faces, ids = get_images_and_labels(dataset_path)

if len(faces) == 0 or len(ids) == 0:
    print("[ERROR] Tidak ditemukan data untuk training. Cek isi folder 'data/'.")
    exit()

recognizer.train(faces, np.array(ids))

# === Simpan hasil training ===
if not os.path.exists("trainer"):
    os.makedirs("trainer")
recognizer.write('trainer/face_trainer.yml')

print(f"[INFO] Training selesai. Total wajah dilatih: {len(np.unique(ids))}")
