#!/usr/bin/env python3
# final.py (Dashboard Upgrade - camera cleanup fix)
# Fitur: Stats harian, search & filter tanggal, tabel realtime, export CSV (sesuai filter), threaded camera scan
# Fix: OpenCV window/kamera benar2 release setelah ESC/Stop agar bisa buka lagi

import tkinter as tk
from tkinter import messagebox, ttk
import threading
import cv2
import numpy as np
import mysql.connector
from datetime import datetime, date
import os
import csv
import time

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "absensi"
}

WINDOW_NAME = "FaceLock - Kamera (ESC untuk berhenti)"

# =========================
# DB Helpers
# =========================
def connect_db():
    return mysql.connector.connect(**DB_CONFIG)

def simpan_kehadiran(nama, status):
    try:
        db = connect_db()
        cursor = db.cursor()
        waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            "INSERT INTO kehadiran (nama, waktu, status) VALUES (%s, %s, %s)",
            (nama, waktu, status),
        )
        db.commit()
        cursor.close()
        db.close()
        return {"ok": True, "log": f"{nama} | {status} | {waktu}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def fetch_recent(limit=100, nama_filter=None, status_filter=None, tanggal_filter=None):
    try:
        db = connect_db()
        cursor = db.cursor()
        q = "SELECT id, nama, status, waktu FROM kehadiran"
        conds, params = [], []

        if nama_filter:
            conds.append("nama LIKE %s")
            params.append(f"%{nama_filter}%")
        if status_filter and status_filter in ("Masuk", "Keluar"):
            conds.append("status = %s")
            params.append(status_filter)
        if tanggal_filter:
            conds.append("DATE(waktu) = %s")
            params.append(tanggal_filter)

        if conds:
            q += " WHERE " + " AND ".join(conds)
        q += " ORDER BY waktu DESC LIMIT %s"
        params.append(limit)

        cursor.execute(q, tuple(params))
        rows = cursor.fetchall()
        cursor.close()
        db.close()
        return rows
    except Exception:
        return []

def fetch_stats_harian(tanggal_str=None):
    try:
        db = connect_db()
        cursor = db.cursor()

        total_user = len(load_names())

        if tanggal_str:
            cursor.execute("SELECT COUNT(*) FROM kehadiran WHERE status='Masuk' AND DATE(waktu)=%s", (tanggal_str,))
            masuk = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM kehadiran WHERE status='Keluar' AND DATE(waktu)=%s", (tanggal_str,))
            keluar = cursor.fetchone()[0]
            cursor.execute("SELECT nama, waktu, status FROM kehadiran WHERE DATE(waktu)=%s ORDER BY waktu DESC LIMIT 1", (tanggal_str,))
        else:
            cursor.execute("SELECT COUNT(*) FROM kehadiran WHERE status='Masuk' AND DATE(waktu)=CURDATE()")
            masuk = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM kehadiran WHERE status='Keluar' AND DATE(waktu)=CURDATE()")
            keluar = cursor.fetchone()[0]
            cursor.execute("SELECT nama, waktu, status FROM kehadiran WHERE DATE(waktu)=CURDATE() ORDER BY waktu DESC LIMIT 1")

        last = cursor.fetchone()
        if last:
            ts = last[1]
            jam = ts.strftime('%H:%M:%S') if hasattr(ts, 'strftime') else str(ts)
            terakhir = f"{last[0]} - {last[2]} @ {jam}"
        else:
            terakhir = "—"

        cursor.close()
        db.close()
        return {"total_user": total_user, "masuk": masuk, "keluar": keluar, "terakhir": terakhir}
    except Exception:
        return {"total_user": len(load_names()), "masuk": 0, "keluar": 0, "terakhir": "—"}

# =========================
# Names helper
# =========================
def load_names():
    if not os.path.exists("names.txt"):
        return []
    with open("names.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

# =========================
# Scanner (threaded)
# =========================
class FaceScanner(threading.Thread):
    def __init__(self, status_absen, result_var, table_refresher, stop_event):
        super().__init__(daemon=True)
        self.status_absen = status_absen
        self.result_var = result_var
        self.table_refresher = table_refresher
        self.stop_event = stop_event

        try:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.recognizer.read('trainer/face_trainer.yml')
        except Exception as e:
            self.recognizer = None
            self.result_var.set(f"[ERROR] Gagal muat model: {e}")

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.names = load_names()

    def run(self):
        if self.recognizer is None:
            return

        cam = None
        try:
            # Pastikan window lama (jika ada) ditutup sebelum mulai
            try:
                cv2.destroyWindow(WINDOW_NAME)
            except Exception:
                pass
            for _ in range(3):
                cv2.waitKey(1)

            cam = cv2.VideoCapture(0)
            if not cam.isOpened():
                self.result_var.set("[ERROR] Kamera tidak tersedia.")
                return

            cam.set(3, 640)
            cam.set(4, 480)

            recognized_ids = set()
            self.result_var.set("🔍 Scanning... Tekan ESC di jendela kamera atau ⛔ Stop di GUI.")

            while not self.stop_event.is_set():
                ret, frame = cam.read()
                if not ret:
                    self.result_var.set("[ERROR] Gagal baca frame kamera.")
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

                draw_name = "..."
                for (x, y, w, h) in faces:
                    roi = gray[y:y+h, x:x+w]
                    try:
                        id_pred, conf = self.recognizer.predict(roi)
                    except Exception:
                        continue

                    if conf < 50 and 1 <= id_pred <= len(self.names):
                        name = self.names[id_pred - 1]
                        draw_name = name
                        if id_pred not in recognized_ids:
                            res = simpan_kehadiran(name, self.status_absen)
                            if res.get("ok"):
                                self.result_var.set("✅ " + res.get("log"))
                            else:
                                self.result_var.set(f"[DB ERROR] {res.get('error')}")
                            recognized_ids.add(id_pred)
                            try:
                                self.table_refresher()
                            except Exception:
                                pass
                    else:
                        draw_name = "Unknown"
                        self.result_var.set("⚠️ Wajah tidak dikenali")

                    color = (0, 255, 0) if draw_name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, draw_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

                cv2.imshow(WINDOW_NAME, frame)
                key = (cv2.waitKey(1) & 0xFF)
                if key == 27:  # ESC
                    self.stop_event.set()
                    break

        finally:
            # RAPIKAN APAPUN YANG TERJADI
            try:
                if cam is not None and cam.isOpened():
                    cam.release()
            except Exception:
                pass

            try:
                # Tutup window spesifik + flush beberapa kali (fix HighGUI stuck)
                cv2.destroyWindow(WINDOW_NAME)
            except Exception:
                pass
            for _ in range(5):
                cv2.waitKey(1)
                time.sleep(0.01)

            try:
                # Tambahan jaga-jaga
                cv2.destroyAllWindows()
            except Exception:
                pass
            for _ in range(3):
                cv2.waitKey(1)

            # pastikan flag berhenti ON
            self.stop_event.set()

            if not self.result_var.get():
                self.result_var.set("🔁 Scan dihentikan.")

# =========================
# GUI
# =========================
def start_gui():
    window = tk.Tk()
    window.title("FaceLock - Absensi Wajah")
    window.geometry("900x640")
    window.configure(bg="#f3f4f6")

    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass
    style.configure("Treeview", rowheight=26)

    tk.Label(window, text="🧑‍💼 FaceLock Absensi", font=("Helvetica", 20, "bold"), bg="#f3f4f6").pack(pady=6)

    top_frame = tk.Frame(window, bg="#f3f4f6")
    top_frame.pack(pady=4, fill="x", padx=12)

    status_var = tk.StringVar(value="Masuk")
    tk.Radiobutton(top_frame, text="Masuk", variable=status_var, value="Masuk", bg="#f3f4f6").pack(side="left", padx=6)
    tk.Radiobutton(top_frame, text="Keluar", variable=status_var, value="Keluar", bg="#f3f4f6").pack(side="left", padx=6)

    result_var = tk.StringVar(value="")
    tk.Label(top_frame, textvariable=result_var, bg="#f3f4f6", fg="green", font=("Helvetica", 11)).pack(side="left", padx=12)

    # --- Stats Cards ---
    stats_frame = tk.Frame(window, bg="#f3f4f6")
    stats_frame.pack(fill="x", padx=12, pady=6)

    def card(parent, title, init_value, bg_color):
        frame = tk.Frame(parent, bg=bg_color, bd=0, highlightthickness=0)
        frame.pack(side="left", padx=8, pady=4, fill="x", expand=True)
        tk.Label(frame, text=title, bg=bg_color, font=("Helvetica", 11)).pack(anchor="w", padx=12, pady=(10,0))
        var = tk.StringVar(value=init_value)
        tk.Label(frame, textvariable=var, bg=bg_color, font=("Helvetica", 16, "bold")).pack(anchor="w", padx=12, pady=(0,10))
        return var

    stat_total_user = card(stats_frame, "👥 Total Terdaftar", "0", "#E0F2FE")
    stat_masuk = card(stats_frame, "🟢 Masuk (hari ini)", "0", "#DCFCE7")
    stat_keluar = card(stats_frame, "🔴 Keluar (hari ini)", "0", "#FEE2E2")
    stat_terakhir = card(stats_frame, "⏰ Terakhir Absen", "—", "#FEF9C3")

    # --- Controls ---
    control_frame = tk.Frame(window, bg="#f3f4f6")
    control_frame.pack(pady=4, fill="x", padx=12)

    tk.Label(control_frame, text="🔎 Cari Nama:", bg="#f3f4f6").pack(side="left", padx=(0,4))
    entry_search = tk.Entry(control_frame, width=18); entry_search.pack(side="left", padx=(0,10))

    tk.Label(control_frame, text="Status:", bg="#f3f4f6").pack(side="left")
    status_filter = ttk.Combobox(control_frame, values=["Semua", "Masuk", "Keluar"], width=8, state="readonly")
    status_filter.set("Semua"); status_filter.pack(side="left", padx=(4,10))

    tk.Label(control_frame, text="Tanggal (YYYY-MM-DD):", bg="#f3f4f6").pack(side="left")
    entry_date = tk.Entry(control_frame, width=12)
    entry_date.insert(0, date.today().strftime("%Y-%m-%d"))
    entry_date.pack(side="left", padx=(4,10))

    stop_event = threading.Event()
    scanner_thread = {"thread": None}

    def apply_filters_to_table():
        nama_f = entry_search.get().strip() or None
        st_f = status_filter.get(); st_f = None if st_f == "Semua" else st_f
        tgl = entry_date.get().strip(); tgl = None if tgl == "" else tgl
        rows = fetch_recent(200, nama_filter=nama_f, status_filter=st_f, tanggal_filter=tgl)
        for i in attendance_table.get_children():
            attendance_table.delete(i)
        for r in rows:
            attendance_table.insert("", tk.END, values=(r[0], r[1], r[2], r[3]))

    def fetch_stats_harian_wrapper():
        tgl = entry_date.get().strip() or None
        stats = fetch_stats_harian(tgl)
        stat_total_user.set(str(stats["total_user"]))
        stat_masuk.set(str(stats["masuk"]))
        stat_keluar.set(str(stats["keluar"]))
        stat_terakhir.set(stats["terakhir"])

    def refresh_all():
        apply_filters_to_table()
        fetch_stats_harian_wrapper()

    def wait_and_cleanup_windows():
        # jaga-jaga sebelum mulai sesi baru
        try:
            cv2.destroyWindow(WINDOW_NAME)
        except Exception:
            pass
        for _ in range(5):
            cv2.waitKey(1)
            time.sleep(0.01)
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        for _ in range(3):
            cv2.waitKey(1)

    def start_scan():
        # pastikan thread lama benar2 mati & semua window close
        if scanner_thread["thread"] and scanner_thread["thread"].is_alive():
            messagebox.showinfo("Info", "Scanner sedang berjalan.")
            return

        # cleanup windows yang tersisa (kalau ada)
        wait_and_cleanup_windows()

        # reset stop flag & start
        stop_event.clear()
        thr = FaceScanner(status_var.get(), result_var, lambda: window.after(100, refresh_all), stop_event)
        scanner_thread["thread"] = thr
        thr.start()
        result_var.set("🔍 Scanner berjalan...")

    def stop_scan():
        if scanner_thread["thread"] and scanner_thread["thread"].is_alive():
            if messagebox.askokcancel("Berhenti Scan", "Hentikan proses scanning sekarang?"):
                stop_event.set()
                result_var.set("⛔ Menghentikan scan...")
                # Tutup window juga dari GUI side (double safety)
                wait_and_cleanup_windows()
        else:
            result_var.set("Tidak ada proses scanning aktif.")

    def export_csv():
        nama_f = entry_search.get().strip() or None
        st_f = status_filter.get(); st_f = None if st_f == "Semua" else st_f
        tgl = entry_date.get().strip(); tgl = None if tgl == "" else tgl
        rows = fetch_recent(5000, nama_filter=nama_f, status_filter=st_f, tanggal_filter=tgl)
        if not rows:
            messagebox.showinfo("Export", "Tidak ada data untuk diexport.")
            return
        fname = f"absensi_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        try:
            with open(fname, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["id", "nama", "status", "waktu"])
                for r in reversed(rows):
                    writer.writerow(r)
            messagebox.showinfo("Export", f"Berhasil export ke {fname}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def confirm_exit():
        if scanner_thread["thread"] and scanner_thread["thread"].is_alive():
            if not messagebox.askyesno("Scanner aktif", "Scanner sedang berjalan. Hentikan dan keluar?"):
                return
            stop_event.set()
            wait_and_cleanup_windows()
            window.after(400, window.destroy)
        else:
            if messagebox.askokcancel("Konfirmasi Keluar", "Yakin ingin keluar dari FaceLock?"):
                wait_and_cleanup_windows()
                window.destroy()

    # Buttons
    tk.Button(control_frame, text="📷 Mulai Scan", command=start_scan, bg="#4ade80", font=("Helvetica", 11)).pack(side="left", padx=6)
    tk.Button(control_frame, text="⛔ Stop", command=stop_scan, bg="#f87171", font=("Helvetica", 11)).pack(side="left", padx=4)
    tk.Button(control_frame, text="🔄 Refresh", command=refresh_all, bg="#60a5fa", font=("Helvetica", 11)).pack(side="left", padx=4)
    tk.Button(control_frame, text="📤 Export CSV", command=export_csv, bg="#f59e0b", font=("Helvetica", 11)).pack(side="left", padx=4)
    tk.Button(control_frame, text="🚪 Keluar", command=confirm_exit, bg="#ef4444", fg="white", font=("Helvetica", 11)).pack(side="right", padx=6)

    # --- Table ---
    table_frame = tk.Frame(window, bg="#f3f4f6")
    table_frame.pack(fill="both", expand=True, padx=12, pady=(6,12))

    columns = ("ID", "Nama", "Status", "Waktu")
    attendance_table = ttk.Treeview(table_frame, columns=columns, show="headings", height=14)
    for col in columns:
        attendance_table.heading(col, text=col)
        attendance_table.column(col, anchor="center", stretch=True, width=150)

    vsb = ttk.Scrollbar(table_frame, orient="vertical", command=attendance_table.yview)
    attendance_table.configure(yscroll=vsb.set)
    attendance_table.pack(side="left", fill="both", expand=True)
    vsb.pack(side="right", fill="y")

    # Bind filter auto-apply
    entry_search.bind("<Return>", lambda e: apply_filters_to_table())
    status_filter.bind("<<ComboboxSelected>>", lambda e: apply_filters_to_table())
    entry_date.bind("<Return>", lambda e: refresh_all())

    # Initial load
    refresh_all()

    tk.Label(
        window,
        text="Tip: Isi tanggal (YYYY-MM-DD) lalu Enter untuk filter harian. Tekan ESC di jendela kamera untuk stop.",
        bg="#f3f4f6", fg="#6b7280", font=("Helvetica", 9)
    ).pack(pady=(0,8))

    window.protocol("WM_DELETE_WINDOW", confirm_exit)
    window.mainloop()

# =========================
# Main
# =========================
if __name__ == "__main__":
    start_gui()
