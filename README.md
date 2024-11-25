
Определение нечетких операций:     
Определяются три основные нечеткие операции:      
(функции)     
fuzzy_and: возвращает минимальное значение (логическое "И").     
fuzzy_or: возвращает максимальное значение (логическое "ИЛИ").     
fuzzy_not: возвращает отрицание (1 минус значение).     
Определение функций членства    
def membership_low(value):    
    ...

def membership_medium(value):     
    ...

def membership_high(value):      
    ...
Эти функции определяют степени принадлежности входного значения к трем нечетким множествам: низкому, среднему и высокому. Каждая функция возвращает значение от 0 до 1, в зависимости от входного значения.
Применение нечетких правил      
def fuzzy_rule(input1, input2):      
    ...
Эта функция вычисляет степени принадлежности для двух входных значений и применяет нечеткие правила для получения выходных значений (низкое, среднее, высокое) на основе входных значений.
Агрегация выходных значений      
def aggregation(outputs):       
    ...
Функция объединяет выходные значения с помощью операции "ИЛИ", чтобы получить агрегированные выходные значения для низкого, среднего и высокого уровней.      
Дефузификация       
Эта функция вычисляет итоговое значение на основе агрегированных выходных значений, используя метод центра тяжести. Она возвращает одно числовое значение, которое представляет результат нечеткой импликации.
Построение графиков      
def plot_memberships(input1, input2, output):       
    ...
Функция строит графики степеней принадлежности для входных и выходных значений. Она визуализирует, как входные значения соотносятся с нечеткими множествами и как выходные значения агрегируются
Главная функция      
def main(input1, input2):       
    ...
Эта функция объединяет все предыдущие шаги:      

Вычисляет выходные значения на основе входных.      
Агрегирует выходные значения.      
Выполняет дефузификацию.     
Строит графики для визуализации.       
# Mamdani_nechetk
![image](https://github.com/user-attachments/assets/9f49291b-e1a9-404d-8130-58b7183d4474)
![image](https://github.com/user-attachments/assets/151af3bc-f4ba-4803-99e1-81f02fe72e68)


