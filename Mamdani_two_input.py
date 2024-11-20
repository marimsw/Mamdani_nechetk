import numpy as np
import matplotlib.pyplot as plt

def fuzzy_and(x, y):
    """Нечеткая операция AND (пересечение)"""
    return min(x, y)

def fuzzy_or(x, y):
    """Нечеткая операция OR (объединение)"""
    return max(x, y)

def fuzzy_not(x):
    """Нечеткая операция NOT (отрицание)"""
    return 1 - x

def membership_low(value):
    """Степень принадлежности к низкому множеству"""
    if value <= 0:
        return 1
    elif 0 < value < 0.5:
        return 1 - (value / 0.5)
    else:
        return 0

def membership_medium(value):
    """Степень принадлежности к среднему множеству"""
    if 0 <= value <= 1:
        if value < 0.5:
            return 2 * value  # Линейное увеличение
        elif value <= 1:
            return 2 * (1 - value)  # Линейное уменьшение
    return 0

def membership_high(value):
    """Степень принадлежности к высокому множеству"""
    if value < 0.5:
        return 0
    elif 0.5 <= value < 1:
        return (value - 0.5) / 0.5
    else:
        return 1

def fuzzy_rule(input1, input2):
    """Применяем нечеткие правила к входным значениям"""

    # Вычисляем степени принадлежности для входных значений
    low1 = membership_low(input1)
    medium1 = membership_medium(input1)
    high1 = membership_high(input1)

    low2 = membership_low(input2)
    medium2 = membership_medium(input2)
    high2 = membership_high(input2)

    # Применяем нечеткие правила
    output_low = fuzzy_and(low1, high2)  # Низкая температура и высокая влажность
    output_medium = fuzzy_and(medium1, medium2)  # Средняя температура и средняя влажность
    output_high = fuzzy_and(high1, low2)  # Высокая температура и низкая влажность

    return (output_low, output_medium, output_high)

def aggregation(outputs):
    """Агрегация выходных значений"""
    # Объединяем выходные значения с помощью операции OR
    aggregated_low = fuzzy_or(outputs[0], outputs[1])  # Низкое
    aggregated_medium = fuzzy_or(outputs[1], outputs[2])  # Среднее
    aggregated_high = fuzzy_or(outputs[2], outputs[0])  # Высокое
    return (aggregated_low, aggregated_medium, aggregated_high)


def defuzzification(output):
    """Дефузификация: вычисляет итоговое значение на основе нечетких значений"""
    low, medium, high = output

    total_area = low + medium + high
    if total_area == 0:
        return 0  # Избегаем деления на ноль

    # Применяем метод центра тяжести для дефузификации
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
    output_low = [min(membership_low(x), output[0]) for x in x_values]  # Низкое
    output_medium = [min(membership_medium(x), output[1]) for x in x_values]  # Среднее
    output_high = [min(membership_high(x), output[2]) for x in x_values]  # Высокое

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
    input1 = 0.4  # Пример входного значения 1
    input2 = 0.7  # Пример входного значения 2
    result = main(input1, input2)
    print(f"Результат нечеткой импликации: {result:.2f}")
