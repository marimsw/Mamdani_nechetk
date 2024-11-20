import numpy as np
import matplotlib.pyplot as plt

def fuzzy_and(x, y):
    """Нечеткая операция AND (пересечение)"""
    return min(x, y)

def fuzzy_or(x, y):
    """Нечеткая операция OR (объединение)"""
    return max(x, y)

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

def fuzzy_rule(inputs):
    """Применяем нечеткие правила к входным значениям"""
    memberships = [(
        membership_low(input_value),
        membership_medium(input_value),
        membership_high(input_value)
    ) for input_value in inputs]

    # Применяем нечеткие правила
    output_low = fuzzy_and(memberships[0][0], memberships[1][2])  # Низкая температура и высокая влажность
    output_medium = fuzzy_and(memberships[0][1], memberships[1][1])  # Средняя температура и средняя влажность
    output_high = fuzzy_and(memberships[0][2], memberships[1][0])  # Высокая температура и низкая влажность

    return (output_low, output_medium, output_high)

def aggregation(outputs):
    """Агрегация выходных значений"""
    return (outputs[0], outputs[1], outputs[2])

def defuzzification(output):
    """Дефузификация: вычисляет итоговое значение на основе нечетких значений"""
    low, medium, high = output

    total_area = low + medium + high
    if total_area == 0:
        return 0  # Избегаем деления на ноль

    # Применяем метод центра тяжести для дефузификации
    defuzzified_value = (0 * low + 0.5 * medium + 1 * high) / total_area
    return defuzzified_value

def plot_memberships(inputs, output):
    """Построение графиков степеней принадлежности"""
    x_values = np.linspace(0, 1, 100)

    # Степени принадлежности для входных переменных
    memberships = [(
        [membership_low(x) for x in x_values],
        [membership_medium(x) for x in x_values],
        [membership_high(x) for x in x_values]
    ) for input_value in inputs]

    # Степени принадлежности для выходных переменных
    output_low = [min(membership_low(x), output[0]) for x in x_values]
    output_medium = [min(membership_medium(x), output[1]) for x in x_values]
    output_high = [min(membership_high(x), output[2]) for x in x_values]

    plt.figure(figsize=(12, 8))

    # Графики входных переменных
    for i, (low, medium, high) in enumerate(memberships):
        plt.subplot(len(inputs), 2, i * 2 + 1)
        plt.title(f"Вход {i + 1}: Степени принадлежности")
        plt.plot(x_values, low, label='Низкое', color='blue')
        plt.plot(x_values, medium, label='Среднее', color='green')
        plt.plot(x_values, high, label='Высокое', color='red')
        plt.axvline(inputs[i], color='black', linestyle='--', label=f'Вход {i + 1}')
        plt.legend()

    # График выходных переменных
    plt.subplot(len(inputs), 2, len(inputs) * 2 - 1)
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

def main(inputs):
    """Главная функция, которая осуществляет нечеткую импликацию Мамдани"""
    output = fuzzy_rule(inputs)

    # Агрегация выходных значений
    aggregated_output = aggregation(output)

    # Дефузификация
    result = defuzzification(aggregated_output)

    # Построение графиков
    plot_memberships(inputs, aggregated_output)

    return result

# Примеры использования
if __name__ == "__main__":
    inputs = [0.4, 0.7, 0.8]  # Пример входных значений
    result = main(inputs)
    print(f"Результат нечеткой импликации: {result:.2f}")
