import cv2
import numpy as np
import mysql.connector
from datetime import datetime
import os
import time
from colorama import Fore, Style, init

init(autoreset=True)

def line():
    print(Fore.CYAN + "-" * 40)

# === Load Nama ===
def load_names():
    with open("names.txt", "r") as file:
        return [line.strip() for line in file.readlines()]

# === Koneksi DB ===
def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="absensi"
    )

# === Simpan Absensi ===
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
    print(Fore.GREEN + f"[✔] Tercatat: {nama} | {status} | {waktu}")

# === UI Pilih Status ===
def pilih_status():
    line()
    print(Fore.YELLOW + " FACELOCK ABSENSI ".center(40, "="))
    line()
    print(Fore.BLUE + "[1] Absensi Masuk")
    print(Fore.BLUE + "[2] Absensi Keluar")
    line()
    choice = input(Fore.CYAN + "Pilih status [1/2]: ").strip()
    if choice == "1":
        return "Masuk"
    elif choice == "2":
        return "Keluar"
    else:
        print(Fore.RED + "[!] Pilihan tidak valid.")
        exit()

# === Load Model ===
def jalankan_absensi(status_absen):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/face_trainer.yml')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    names = load_names()

    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)

    print(Fore.MAGENTA + f"\n[•] Mode Absensi: {status_absen}")
    print(Fore.YELLOW + "[ESC] untuk keluar")
    print(Fore.CYAN + "Scanning wajah...\n")
    time.sleep(1)

    recognized_ids = set()

    while True:
        ret, frame = cam.read()
        if not ret:
            print(Fore.RED + "[!] Kamera tidak terdeteksi.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            id_pred, confidence = recognizer.predict(face_img)

            if confidence < 50:
                name = names[id_pred - 1]
                confidence_text = f"{round(100 - confidence)}%"
                color = (0, 255, 0)

                if id_pred not in recognized_ids:
                    simpan_kehadiran(name, status_absen)
                    recognized_ids.add(id_pred)

            else:
                name = "Unknown"
                confidence_text = f"{round(100 - confidence)}%"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, str(name), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, str(confidence_text), (x + 5, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        cv2.imshow('FaceLock Absensi', frame)

        k = cv2.waitKey(1)
        if k == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

# === Eksekusi ===
if __name__ == "__main__":
    status = pilih_status()
    jalankan_absensi(status)
