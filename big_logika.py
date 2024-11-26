
import os
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt

def detect_objects(model, image_path, target_class):
    """
    Детекция объектов на изображении.

    Args:
        model (YOLO): Модель детекции объектов.
        image_path (str): Путь к изображению.
        target_class (str): Класс объекта, который нужно детектировать.

    Returns:
        float: Достоверность детекции объекта.
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
                    return conf
        return 0
    except Exception as e:
        print(f"Ошибка детекции объектов: {e}")
        return 0

def fuzzy_and(x, y):
    """
    Нечеткая логика AND.

    Args:
        x (float): Первый входной параметр.
        y (float): Второй входной параметр.

    Returns:
        float: Результат нечеткой логике AND.
    """
    return min(x, y)

def fuzzy_or(x, y):
    """
    Нечеткая логика OR.

    Args:
        x (float): Первый входной параметр.
        y (float): Второй входной параметр.

    Returns:
        float: Результат нечеткой логике OR.
    """
    return max(x, y)

def fuzzy_not(x):
    """
    Нечеткая логика NOT.

    Args:
        x (float): Входной параметр.

    Returns:
        float: Результат нечеткой логике NOT.
    """
    return 1 - x

def membership_low(value):
    """
    Функция принадлежности для низких значений.

    Args:
        value (float): Входной параметр.

    Returns:
        float: Значение принадлежности.
    """
    if value <= 0:
        return 1
    elif 0 < value < 0.5:
        return 1 - (value / 0.5)
    else:
        return 0

def membership_medium(value):
    """
    Функция принадлежности для средних значений.

    Args:
        value (float): Входной параметр.

    Returns:
        float: Значение принадлежности.
    """
    if 0 <= value <= 1:
        if value < 0.5:
            return 2 * value
        elif value <= 1:
            return 2 * (1 - value)
    return 0

def membership_high(value):
    """
    Функция принадлежности для высоких значений.

    Args:
        value (float): Входной параметр.

    Returns:
        float: Значение принадлежности.
    """
    if value < 0.5:
        return 0
    elif 0.5 <= value < 1:
        return (value - 0.5) / 0.5
    else:
        return 1

def fuzzy_rule(input1, input2):
    """
    Нечеткая логика Мамдани.

    Args:
        input1 (float): Первый входной параметр.
        input2 (float): Второй входной параметр.

    Returns:
        tuple: Выходные значения нечеткой логике.
    """
    low1 = membership_low(input1)
    medium1 = membership_medium(input1)
    high1 = membership_high(input1)

    low2 = membership_low(input2)
    medium2 = membership_medium(input2)
    high2 = membership_high(input2)

    output_low = fuzzy_and(low1, high2)
    output_medium = fuzzy_and(medium1, medium2)
    output_high = fuzzy_and(high1, medium2)

    # Увеличиваем выходные значения, если один из входов больше 0.7
    if input1 > 0.7:
        output_high = max(output_high, high1)  # Увеличиваем выходное значение в зависимости от high1
    if input2 > 0.7:
        output_high = max(output_high, high2)  # Увеличиваем выходное значение в зависимости от high2

    return (output_low, output_medium, output_high)


def aggregation(outputs):
    """
    Агрегация выходных значений.

    Args:
        outputs (tuple): Выходные значения нечеткой логике.

    Returns:
        tuple: Агрегированные выходные значения.
    """
    aggregated_low = fuzzy_or(outputs[0], outputs[1])
    aggregated_medium = fuzzy_or(outputs[1], outputs[2])
    aggregated_high = fuzzy_or(outputs[2], outputs[0])
    return (aggregated_low, aggregated_medium, aggregated_high)

def defuzzification(output):
    """
    Дефазификация выходных значений.

    Args:
        output (tuple): Выходные значения нечеткой логике.

    Returns:
        float: Дефазифицированное значение.
    """
    low, medium, high = output
    total_area = low + medium + high
    if total_area == 0:
        return 0
    defuzzified_value = (0 * low + 0.5 * medium + 1 * high) / total_area
    return defuzzified_value

def plot_memberships(input1, input2, output):
    """
    Построение графиков степеней принадлежности.

    Args:
        input1 (float): Первый входной параметр.
        input2 (float): Второй входной параметр.
        output (tuple): Выходные значения нечеткой логике.
    """
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
    image_path = '/content/drive/MyDrive/Rabota/Алгоритмы  Мамдани _от 18_11_24/YOLOv8_and_vesa_18_plus/maket/images (7).jpg'

    # Проверка существования файла
    if not os.path.exists(image_path):
        print(f"Файл не найден: {image_path}")
    else:
        # Загрузка моделей с весами
        model1 = YOLO('/content/drive/MyDrive/Rabota/Алгоритмы  Мамдани _от 18_11_24/YOLOv8_and_vesa_18_plus/best.pt')
        model2 = YOLO('/content/drive/MyDrive/Rabota/Алгоритмы  Мамдани _от 18_11_24/YOLOv8_and_vesa_18_plus/best1.pt')

        # Детекция объектов на изображении
        input1 = detect_objects(model1, image_path, "neck ass")
        input2 = detect_objects(model2, image_path, "penis")

        # Проверяем, что хотя бы один объект обнаружен с достоверностью больше 0.7
        if (input1 > 0.7 and input2 == 0) or (input2 > 0.7 and input1 == 0):
            if input1 > 0:
                print(f"Объект 'neck ass' обнаружен с достоверностью {input1:.2f}")
            if input2 > 0:
                print(f"Объект 'penis' обнаружен с достоверностью {input2:.2f}")

            result = main(input1, input2)
            print(f"Результат нечеткой импликации: {result:.2f}")
        else:
            print("Не удалось получить достоверности для объектов 'neck ass' и 'penis' или оба объекта обнаружены.")



