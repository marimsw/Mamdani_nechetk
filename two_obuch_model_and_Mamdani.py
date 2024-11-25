import os
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import cv2  # Добавляем библиотеку OpenCV для работы с изображениями

def detect_objects(model, image_path, target_class):
    """
    Детекция объектов на изображении.

    Args:
        model (YOLO): Модель детекции объектов.
        image_path (str): Путь к изображению.
        target_class (str): Класс объекта, который нужно детектировать.

    Returns:
        tuple: Достоверность детекции объекта и координаты (xmin, ymin, xmax, ymax).
    """
    try:
        results = model(image_path)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                xmin, ymin, xmax, ymax = box.xyxy[0]
                conf = box.conf[0]
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                if class_name == target_class:
                    return conf, (xmin, ymin, xmax, ymax)
        return 0, None
    except Exception as e:
        print(f"Ошибка детекции объектов: {e}")
        return 0, None

def plot_image_with_detections(image_path, detections):
    """
    Отображение изображения с детектированными объектами.

    Args:
        image_path (str): Путь к изображению.
        detections (list): Список координат обнаруженных объектов.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Преобразуем BGR в RGB для matplotlib

    for conf, (xmin, ymin, xmax, ymax) in detections:
        cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)
        plt.text(xmin, ymin, f'{conf:.2f}', color='white', fontsize=12, backgroundcolor='blue')

    plt.imshow(image)
    plt.axis('off')
    plt.title('Обнаруженные объекты')
    plt.show()

def main(input1, input2):
    """
    Главная функция, которая осуществляет нечеткую импликацию Мамдани.

    Args:
        input1 (float): Первый входной параметр.
        input2 (float): Второй входной параметр.

    Returns:
        float: Результат нечеткой импликации.
    """
    output = fuzzy_rule(input1, input2)
    aggregated_output = aggregation(output)
    result = defuzzification(aggregated_output)
    plot_memberships(input1, input2, aggregated_output)
    return result

if __name__ == "__main__":
    # Путь к файлам изображений
    image_path = '/content/drive/MyDrive/Rabota/Алгоритмы  Мамдани _от 18_11_24/YOLOv8_and_vesa_18_plus/66.jpg'

    # Проверка существования файла
    if not os.path.exists(image_path):
        print(f"Файл не найден: {image_path}")
    else:
        # Загрузка моделей с весами
        model1 = YOLO('/content/drive/MyDrive/Rabota/Алгоритмы  Мамдани _от 18_11_24/YOLOv8_and_vesa_18_plus/best.pt')
        model2 = YOLO('/content/drive/MyDrive/Rabota/Алгоритмы  Мамдани _от 18_11_24/YOLOv8_and_vesa_18_plus/best1.pt')

        # Детекция объектов на изображении
        input1, bbox1 = detect_objects(model1, image_path, "neck ass")
        input2, bbox2 = detect_objects(model2, image_path, "penis")

        # Проверяем, что достоверности не равны нулю
        if input1 > 0 and input2 > 0:
            print(f"Объект 'neck ass' обнаружен с достоверностью {input1:.2f}")
            print(f"Объект 'penis' обнаружен с достоверностью {input2:.2f}")
            result = main(input1, input2)
            print(f"Результат нечеткой импликации: {result:.2f}")

            # Отображаем изображение с детекцией
            detections = [(input1, bbox1), (input2, bbox2)]
            plot_image_with_detections(image_path, detections)
        else:
            print("Не удалось получить достоверности для объектов 'neck ass' и 'penis'.")
