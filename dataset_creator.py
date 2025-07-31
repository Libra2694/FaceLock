import cv2
import os

# === Setup ===
base_dir = "data"

# Input nama user
user_name = input("Masukkan nama user: ").strip().replace(" ", "_")

# Cek dan tambahkan nama ke names.txt
if not os.path.exists("names.txt"):
    with open("names.txt", "w") as f:
        f.write(user_name + "\n")
else:
    with open("names.txt", "r") as f:
        names = [line.strip() for line in f.readlines()]
    if user_name not in names:
        with open("names.txt", "a") as f:
            f.write(user_name + "\n")

# Hitung ID user dari urutan di names.txt
with open("names.txt", "r") as f:
    names = [line.strip() for line in f.readlines()]
    user_id = names.index(user_name) + 1  # ID dimulai dari 1

# Folder khusus untuk user
user_dir = os.path.join(base_dir, user_name)
os.makedirs(user_dir, exist_ok=True)

# === Inisialisasi Kamera dan Detektor ===
cam = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print(f"[INFO] Mulai ambil 50 gambar untuk: {user_name} (ID: {user_id})")

count = 0
while True:
    ret, frame = cam.read()
    if not ret:
        print("[ERROR] Kamera tidak tersedia.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        face_img = gray[y:y+h, x:x+w]
        filename = f"User.{user_id}.{count}.jpg"
        cv2.imwrite(os.path.join(user_dir, filename), face_img)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{count}/50", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imshow("Ambil Dataset Wajah", frame)

    k = cv2.waitKey(1)
    if k == 27 or count >= 50:
        break

print(f"[INFO] Selesai. Dataset {user_name} tersimpan di: {user_dir}")
cam.release()
cv2.destroyAllWindows()
