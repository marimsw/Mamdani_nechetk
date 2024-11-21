# pip install ultralytics
# Берутся две переобученные модели(на частях мужского тела) и высчитывается Mamdani
import os
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt

# Путь к файлам изображений
image_path = '/content/drive/MyDrive/Rabota/Алгоритмы  Мамдани _от 18_11_24/YOLOv8_and_vesa_18_plus/66.jpg'

# Проверка существования файла
if not os.path.exists(image_path):
    print(f"Файл не найден: {image_path}")
else:
    # Загрузка моделей с весами
    model1 = YOLO('/content/drive/MyDrive/Rabota/Алгоритмы  Мамдани _от 18_11_24/YOLOv8_and_vesa_18_plus/best.pt')
    model2 = YOLO('/content/drive/MyDrive/Rabota/Алгоритмы  Мамдани _от 18_11_24/YOLOv8_and_vesa_18_plus/best1.pt')

    # Функция для обработки изображения
    def detect_objects(model, image_path, target_class):
        # Выполнение детекции на изображении
        results = model(image_path)

        # Инициализация переменной для хранения достоверности
        confidence = 0

        # Обработка результатов
        for result in results:
            boxes = result.boxes  # Получаем объект с детекциями
            for box in boxes:
                xmin, ymin, xmax, ymax = box.xyxy[0]  # Координаты
                conf = box.conf[0]  # Достоверность
                class_id = int(box.cls[0])  # ID класса
                class_name = model.names[class_id]  # Название класса

                # Выводим информацию о каждом найденном объекте
                print(f"Объект: {class_name}, Достоверность: {conf:.2f}, Координаты: ({xmin}, {ymin}, {xmax}, {ymax})")

                # Сохраняем достоверность для нужного объекта
                if class_name == target_class:
                    confidence = conf

        return confidence

    # Нечеткая логика Мамдани
    def fuzzy_and(x, y):
        return min(x, y)

    def fuzzy_or(x, y):
        return max(x, y)

    def membership_low(value):
        if value <= 0:
            return 1
        elif 0 < value < 0.3:
            return 1 - (value / 0.3)
        else:
            return 0

    def membership_medium(value):
        if 0 <= value <= 1:
            if value < 0.5:
                return 2 * value
            elif value <= 1:
                return 2 * (1 - value)
        return 0

    def membership_high(value):
        if value < 0.3:
            return 0
        elif 0.3 <= value < 1:
            return (value - 0.3) / 0.7
        else:
            return 1

    def fuzzy_rule(input1, input2):
        low1 = membership_low(input1)
        medium1 = membership_medium(input1)
        high1 = membership_high(input1)

        low2 = membership_low(input2)
        medium2 = membership_medium(input2)
        high2 = membership_high(input2)

        output_low = fuzzy_and(low1, high2)
        output_medium = fuzzy_and(medium1, medium2)
        output_high = fuzzy_and(high1, medium2)

        return (output_low, output_medium, output_high)

    def aggregation(outputs):
        aggregated_low = fuzzy_or(outputs[0], outputs[1])
        aggregated_medium = fuzzy_or(outputs[1], outputs[2])
        aggregated_high = fuzzy_or(outputs[2], outputs[0])
        return (aggregated_low, aggregated_medium, aggregated_high)

    def defuzzification(output):
        low, medium, high = output
        total_area = low + medium + high
        if total_area == 0:
            return 0
        defuzzified_value = (0 * low + 0.5 * medium + 1 * high) / total_area
        return defuzzified_value

    def plot_memberships(input1, input2, output):
        """Построение графиков степеней принадлежности"""
        x_values = np.linspace(0, 1, 100)

        # Степени принадлежности для входных переменных
        low1 = [membership_low(x) for x in x_values]
        medium1 = [membership_medium(x) for x in x_values]
        high1 = [membership_high(x) for x in x_values]

        low2 = [membership_low(x) for x in x_values]
        medium2 = [membership_medium(x) for x in x_values]
        high2 = [membership_high(x) for x in x_values]

        # Степени принадлежности для выходных переменных
        output_low = [min(membership_low(x), output[0]) for x in x_values]
        output_medium = [min(membership_medium(x), output[1]) for x in x_values]
        output_high = [min(membership_high(x), output[2]) for x in x_values]

        plt.figure(figsize=(12, 8))

        # График входной переменной 1
        plt.subplot(3, 2, 1)
        plt.title("Вход 1: Степени принадлежности")
        plt.plot(x_values, low1, label='Низкое', color='blue')
        plt.plot(x_values, medium1, label='Среднее', color='green')
        plt.plot(x_values, high1, label='Высокое', color='red')
        plt.axvline(input1, color='black', linestyle='--', label='Вход 1')
        plt.legend()

        # График входной переменной 2
        plt.subplot(3, 2, 2)
        plt.title("Вход 2: Степени принадлежности")
        plt.plot(x_values, low2, label='Низкое', color='blue')
        plt.plot(x_values, medium2, label='Среднее', color='green')
        plt.plot(x_values, high2, label='Высокое', color='red')
        plt.axvline(input2, color='black', linestyle='--', label='Вход 2')
        plt.legend()

        # График выходных переменных
        plt.subplot(3, 1, 2)
        plt.title("Выход: Степени принадлежности")
        plt.plot(x_values, output_low, label='Низкое', color='blue')
        plt.plot(x_values, output_medium, label='Среднее', color='green')
        plt.plot(x_values, output_high, label='Высокое', color='red')
        plt.axhline(0, color='black', lw=0.5)
        plt.axhline(1, color='black', lw=0.5)
        plt.fill_between(x_values, output_low, color='blue', alpha=0.1)
        plt.fill_between(x_values, output_medium, color='green', alpha=0.1)
        plt.fill_between(x_values, output_high, color='red', alpha=0.1)
        plt.legend()

        plt.tight_layout()
        plt.show()

    def main(input1, input2):
        """Главная функция, которая осуществляет нечеткую импликацию Мамдани"""
        output = fuzzy_rule(input1, input2)

        # Агрегация выходных значений
        aggregated_output = aggregation(output)

        # Дефузификация
        result = defuzzification(aggregated_output)

        # Построение графиков
        plot_memberships(input1, input2, aggregated_output)

        return result

    # Примеры использования
    if __name__ == "__main__":
        # Получаем достоверности для объектов "neck ass" из первой модели
        input1 = detect_objects(model1, image_path, "neck ass")
        # Получаем достоверности для объектов "penis" из второй модели
        input2 = detect_objects(model2, image_path, "penis")

        # Проверяем, что достоверности не равны нулю
        if input1 > 0 and input2 > 0:
            result = main(input1, input2)
            print(f"Результат нечеткой импликации: {result:.2f}")
        else:
            print("Не удалось получить достоверности для объектов 'neck ass' и 'penis'.")

