import cv2
import numpy as np
from datetime import datetime
import os
import argparse

def format_timestamp(ms):
    """
    Преобразует миллисекунды в формат HH:MM:SS.mmm
    """
    seconds = int(ms / 1000)
    milliseconds = int(ms % 1000)
    minutes = int(seconds / 60)
    hours = int(minutes / 60)
    
    return f"{hours:02d}_{minutes%60:02d}_{seconds%60:02d}_{milliseconds:03d}"

def calculate_image_difference(img1, img2):
    """
    Вычисляет разницу между двумя изображениями.
    Возвращает True, если изображения достаточно различны.
    """
    if img1 is None or img2 is None:
        return True
    
    # Убеждаемся, что изображения одного размера
    if img1.shape != img2.shape:
        return True
    
    # Вычисляем разницу между изображениями
    diff = cv2.absdiff(img1, img2)
    diff_ratio = np.mean(diff) / 255.0
    
    # Если разница больше 2%, считаем что текст изменился
    return diff_ratio > 0.02

def process_video(video_path, region=None, white_threshold=200, white_pixel_percent=0.01):
    """
    Обрабатывает видео и делает скриншоты при появлении нового белого текста в указанной области.
    
    Args:
        video_path (str): Путь к видеофайлу
        region (tuple): Область интереса (x, y, width, height). Если None, используется всё изображение
        white_threshold (int): Порог яркости для определения белого цвета (0-255)
        white_pixel_percent (float): Минимальный процент белых пикселей для определения текста
    """
    # Создаем папку для скриншотов
    screenshots_dir = "screenshots"
    os.makedirs(screenshots_dir, exist_ok=True)
    
    # Открываем видео
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Ошибка при открытии видео")
        return
    
    # Получаем информацию о видео
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    duration = total_frames / fps
    print(f"Длительность видео: {duration:.1f} сек, Всего кадров: {total_frames}")
    
    # Читаем первый кадр
    ret, frame = cap.read()
    if not ret:
        print("Ошибка при чтении видео")
        return
    
    # Если область не указана, используем весь кадр
    if region is None:
        height, width = frame.shape[:2]
        region = (0, 0, width, height)
    
    x, y, w, h = region
    frame_count = 0
    last_saved_time = 0  # Для предотвращения сохранения слишком частых кадров
    last_saved_frame = None  # Последний сохраненный кадр для сравнения
    last_progress = -1  # Для отображения прогресса только при изменении процентов
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Вычисляем и показываем прогресс
        progress = (frame_count * 100) // total_frames
        if progress != last_progress:  # Показываем только когда процент изменился
            print(f"Прогресс: {progress}% (обработано {frame_count} из {total_frames} кадров)")
            last_progress = progress
            
        # Вырезаем интересующую область
        roi = frame[y:y+h, x:x+w]
        
        # Переводим в градации серого
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Находим белые пиксели
        white_pixels = cv2.threshold(gray, white_threshold, 255, cv2.THRESH_BINARY)[1]
        white_pixel_count = np.count_nonzero(white_pixels)
        total_pixels = white_pixels.size
        white_ratio = white_pixel_count / total_pixels
        
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC)
        
        # Проверяем наличие белого текста и его отличие от предыдущего кадра
        if (white_ratio >= white_pixel_percent and 
            current_time - last_saved_time >= 1000 and  # Минимум 1 секунда между кадрами
            calculate_image_difference(white_pixels, last_saved_frame)):
            
            # Форматируем время видео и текущее время
            video_time = format_timestamp(current_time)
            current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Создаем имя файла с обоими временными метками
            filename = f"{screenshots_dir}/screenshot_v{video_time}_t{current_datetime}.jpg"
            cv2.imwrite(filename, roi)
            print(f"Сохранен новый текст: {filename} (белых пикселей: {white_ratio:.2%})")
            
            last_saved_time = current_time
            last_saved_frame = white_pixels.copy()  # Сохраняем бинаризованное изображение для сравнения
        
        frame_count += 1
        
    cap.release()
    print(f"Обработка завершена. Всего кадров обработано: {frame_count}")

def parse_region(region_str):
    """Преобразует строку с координатами в кортеж чисел"""
    try:
        x, y, w, h = map(int, region_str.split(','))
        return (x, y, w, h)
    except:
        raise argparse.ArgumentTypeError("Регион должен быть указан в формате: x,y,width,height")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Обработка видео и создание скриншотов при появлении белого текста')
    parser.add_argument('video_path', type=str, help='Путь к видеофайлу')
    parser.add_argument('--region', type=parse_region, help='Область интереса в формате: x,y,width,height (например: 100,100,400,300)')
    parser.add_argument('--white-threshold', type=int, default=200, help='Порог яркости для определения белого цвета (0-255)')
    parser.add_argument('--white-pixel-percent', type=float, default=0.01, help='Минимальный процент белых пикселей (0.0-1.0)')
    
    args = parser.parse_args()
    
    process_video(
        args.video_path,
        region=args.region,
        white_threshold=args.white_threshold,
        white_pixel_percent=args.white_pixel_percent
    ) 