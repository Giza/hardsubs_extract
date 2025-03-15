import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser, ttk
import threading
from _hardsubs_extract import process_video, parse_region
import cv2
import numpy as np

class VideoProcessorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Видео обработчик")

        tk.Label(root, text="Путь к видеофайлу:").grid(row=0, column=0, sticky='e')
        self.video_path_entry = tk.Entry(root, width=50)
        self.video_path_entry.grid(row=0, column=1)
        tk.Button(root, text="Обзор", command=self.browse_file).grid(row=0, column=2)

        tk.Label(root, text="Область (x,y,w,h):").grid(row=1, column=0, sticky='e')
        self.region_entry = tk.Entry(root)
        self.region_entry.grid(row=1, column=1)
        tk.Button(root, text="Выбрать область", command=self.select_region).grid(row=1, column=2)

        tk.Label(root, text="Порог яркости (0-255):").grid(row=2, column=0, sticky='e')
        self.white_threshold_entry = tk.Entry(root)
        self.white_threshold_entry.insert(0, "200")
        self.white_threshold_entry.grid(row=2, column=1)

        tk.Label(root, text="Мин. процент белых пикселей:").grid(row=3, column=0, sticky='e')
        self.white_pixel_percent_entry = tk.Entry(root)
        self.white_pixel_percent_entry.insert(0, "0.01")
        self.white_pixel_percent_entry.grid(row=3, column=1)

        tk.Button(root, text="Выбрать цвет", command=self.choose_color).grid(row=3, column=2)
        tk.Button(root, text="Выбрать цвет в кадре", command=self.select_color_in_frame).grid(row=4, column=2)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = tk.ttk.Progressbar(root, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=6, column=0, columnspan=3, sticky='we')

        self.start_button = tk.Button(root, text="Запустить", command=self.start_processing)
        self.start_button.grid(row=5, column=1)

        self.stop_button = tk.Button(root, text="Остановить", command=self.stop_processing)
        self.stop_button.grid(row=7, column=1)
        self.stop_button.grid_remove()  # Скрываем кнопку 'Остановить' до начала обработки

        self.processing_thread = None
        self.stop_event = threading.Event()

        self.selected_color = (255, 255, 255)  # По умолчанию белый

    def browse_file(self):
        filename = filedialog.askopenfilename()
        self.video_path_entry.delete(0, tk.END)
        self.video_path_entry.insert(0, filename)

    def choose_color(self):
        color_code = colorchooser.askcolor(title="Выберите цвет")
        if color_code:
            self.selected_color = tuple(int(c) for c in color_code[0])

    def select_region(self):
        video_path = self.video_path_entry.get()
        if not video_path:
            messagebox.showerror("Ошибка", "Сначала выберите видеофайл.")
            return

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0
        frame_resized = None  # Инициализация переменной
        scale_x = 1
        scale_y = 1

        def on_trackbar(val):
            nonlocal current_frame, frame_resized, scale_x, scale_y
            current_frame = val
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if not ret:
                return
            height, width = frame.shape[:2]
            scale_x = 800 / width
            scale_y = 600 / height
            frame_resized = cv2.resize(frame, (800, 600))  # Изменяем размер для экрана
            cv2.imshow("Выберите область", frame_resized)
            cv2.waitKey(1)  # Обновляем окно после изменения кадра

        cv2.namedWindow("Выберите область")
        cv2.createTrackbar("Кадр", "Выберите область", 0, total_frames - 1, on_trackbar)

        on_trackbar(0)  # Инициализация первого кадра

        while True:
            if cv2.waitKey(1) & 0xFF == 27:  # Нажмите ESC для выхода
                break

        if frame_resized is not None:
            r = cv2.selectROI("Выберите область", frame_resized, False, False)
            if r != (0, 0, 0, 0):
                # Учитываем масштабирование
                real_x = int(r[0] / scale_x)
                real_y = int(r[1] / scale_y)
                real_w = int(r[2] / scale_x)
                real_h = int(r[3] / scale_y)
                self.region_entry.delete(0, tk.END)
                self.region_entry.insert(0, f"{real_x},{real_y},{real_w},{real_h}")

        cv2.destroyAllWindows()
        cap.release()

    def select_color_in_frame(self):
        video_path = self.video_path_entry.get()
        if not video_path:
            messagebox.showerror("Ошибка", "Сначала выберите видеофайл.")
            return

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0
        frame_resized = None  # Инициализация переменной

        def on_trackbar(val):
            nonlocal current_frame, frame_resized
            current_frame = val
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if not ret:
                return
            frame_resized = cv2.resize(frame, (800, 600))  # Изменяем размер для экрана
            cv2.imshow("Выберите цвет", frame_resized)
            cv2.waitKey(1)  # Обновляем окно после изменения кадра

        cv2.namedWindow("Выберите цвет")
        cv2.createTrackbar("Кадр", "Выберите цвет", 0, total_frames - 1, on_trackbar)

        on_trackbar(0)  # Инициализация первого кадра

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.selected_color = frame_resized[y, x].tolist()
                cv2.destroyAllWindows()
                messagebox.showinfo("Цвет выбран", f"Выбранный цвет: {self.selected_color}")

        cv2.setMouseCallback("Выберите цвет", click_event)

        while True:
            if cv2.waitKey(1) & 0xFF == 27:  # Нажмите ESC для выхода
                break

        cv2.destroyAllWindows()
        cap.release()

    def stop_processing(self):
        if self.processing_thread and self.processing_thread.is_alive():
            self.stop_event.set()
            self.processing_thread.join()
            self.stop_button.grid_remove()  # Скрываем кнопку 'Остановить' после остановки
            self.start_button.grid()  # Показываем кнопку 'Запустить' после остановки
            messagebox.showinfo("Остановлено", "Обработка остановлена.")

    def start_processing(self):
        video_path = self.video_path_entry.get()
        region_str = self.region_entry.get()
        white_threshold = int(self.white_threshold_entry.get())
        white_pixel_percent = float(self.white_pixel_percent_entry.get())

        region = parse_region(region_str) if region_str else None

        self.stop_event.clear()
        self.processing_thread = threading.Thread(target=self.run_processing, args=(video_path, region, white_threshold, white_pixel_percent))
        self.processing_thread.start()

        self.start_button.grid_remove()  # Скрываем кнопку 'Запустить' во время обработки
        self.stop_button.grid()  # Показываем кнопку 'Остановить' во время обработки

    def run_processing(self, video_path, region, white_threshold, white_pixel_percent):
        try:
            process_video(video_path, region, white_threshold, white_pixel_percent)
            messagebox.showinfo("Готово", "Обработка завершена успешно!")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoProcessorGUI(root)
    root.mainloop() 