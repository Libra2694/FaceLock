import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
import mysql.connector
from datetime import datetime
import os

def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="absensi"
    )

def simpan_kehadiran(nama, status):
    db = connect_db()
    cursor = db.cursor()
    waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sql = "INSERT INTO kehadiran (nama, waktu, status) VALUES (%s, %s, %s)"
    val = (nama, waktu, status)
    cursor.execute(sql, val)
    db.commit()
    cursor.close()
    db.close()
    log = f"{nama} | {status} | {waktu}"
    print(log)
    return log

def load_names():
    with open("names.txt", "r") as f:
        return [line.strip() for line in f.readlines()]

# === Pengenalan Wajah ===
def mulai_scan(status_absen, result_text):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/face_trainer.yml')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    names = load_names()

    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)

    result_text.set("🔍 Scanning wajah... (Tekan ESC untuk batal)")
    recognized_ids = set()

    while True:
        ret, frame = cam.read()
        if not ret:
            messagebox.showerror("Error", "Kamera tidak terdeteksi.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            id_pred, confidence = recognizer.predict(face_img)

            if confidence < 50:
                name = names[id_pred - 1]
                if id_pred not in recognized_ids:
                    log = simpan_kehadiran(name, status_absen)
                    result_text.set("✅ " + log)
                    recognized_ids.add(id_pred)
            else:
                result_text.set("❌ Wajah tidak dikenali")

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.imshow("FaceLock Absensi", frame)
        if cv2.waitKey(1) == 27:
            result_text.set("🚫 Scan dibatalkan.")
            break

    cam.release()
    cv2.destroyAllWindows()

# === GUI Setup ===
# ... kode sebelumnya tetap ...

def start_gui():
    window = tk.Tk()
    window.title("FaceLock - Absensi Wajah")
    window.geometry("400x320")
    window.configure(bg="#f3f4f6")

    tk.Label(window, text="🧑‍💼 FaceLock Absensi", font=("Helvetica", 16, "bold"), bg="#f3f4f6").pack(pady=10)

    status_var = tk.StringVar(value="Masuk")
    tk.Radiobutton(window, text="Masuk", variable=status_var, value="Masuk", bg="#f3f4f6").pack()
    tk.Radiobutton(window, text="Keluar", variable=status_var, value="Keluar", bg="#f3f4f6").pack()

    result_text = tk.StringVar()
    result_label = tk.Label(window, textvariable=result_text, bg="#f3f4f6", fg="green", font=("Helvetica", 12))
    result_label.pack(pady=10)

    def on_scan():
        result_text.set("🕵️ Scanning wajah... Tekan [ESC] untuk keluar.")
        window.update()
        mulai_scan(status_var.get(), result_text)
        result_text.set("✔ Silakan pilih absensi lagi.")  # Setelah scan selesai

    def reset_gui():
        status_var.set("Masuk")
        result_text.set("")

    tk.Button(window, text="📷 Mulai Scan", command=on_scan, bg="#4ade80", fg="black", font=("Helvetica", 12)).pack(pady=5)
    tk.Button(window, text="❌ Batal / Reset", command=reset_gui, bg="#f87171", fg="white", font=("Helvetica", 11)).pack(pady=5)

    tk.Label(window, text="Tekan [ESC] saat kamera aktif untuk keluar dari scan.", font=("Helvetica", 9), bg="#f3f4f6", fg="#6b7280").pack(pady=10)

    window.mainloop()


# === Jalankan ===
if __name__ == "__main__":
    start_gui()
