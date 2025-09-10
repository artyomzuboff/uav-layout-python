import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import json
import os
import time
from tqdm import tqdm
from datetime import datetime


class Equipment:
    """Класс для представления оборудования БПЛА"""

    def __init__(self, id, name, length, width, height, mass,
                 heat_emission=0, em_radiation=0, em_sensitivity=0, priority=1):
        self.id = id
        self.name = name
        self.dimensions = (length, width, height)
        self.mass = mass
        self.heat_emission = heat_emission
        self.em_radiation = em_radiation
        self.em_sensitivity = em_sensitivity
        self.priority = priority
        self.position = None

    def get_volume(self):
        """Возвращает объем оборудования"""
        return self.dimensions[0] * self.dimensions[1] * self.dimensions[2]

    def __str__(self):
        return f"ID: {self.id}, Название: {self.name}, Размеры: {self.dimensions}, Масса: {self.mass} кг"


class UAV:
    """Класс для представления БПЛА"""

    def __init__(self, id, name, length, width, height, empty_weight, max_payload,
                 cm_required=(None, None, None)):
        self.id = id
        self.name = name
        self.dimensions = (length, width, height)
        self.empty_weight = empty_weight
        self.max_payload = max_payload
        self.cm_required = cm_required
        self.equipment_list = []
        self.occupied_space = np.zeros((height, width, length), dtype=bool)  # z, y, x

    def add_equipment(self, equipment):
        """Добавляет оборудование в список"""
        if sum([eq.mass for eq in self.equipment_list]) + equipment.mass > self.max_payload:
            print(f"Превышена максимальная полезная нагрузка для {self.name}")
            return False
        self.equipment_list.append(equipment)
        return True

    def remove_equipment(self, equipment_id):
        """Удаляет оборудование из списка по ID"""
        for i, eq in enumerate(self.equipment_list):
            if eq.id == equipment_id:
                self.equipment_list.pop(i)
                return True
        return False

    def calculate_center_of_mass(self):
        """Рассчитывает центр масс БПЛА с учетом размещенного оборудования и массы корпуса"""
        if not self.equipment_list:
            return self.cm_required

        positioned_eq = [eq for eq in self.equipment_list if eq.position is not None]
        total_mass = self.empty_weight + sum(eq.mass for eq in positioned_eq)

        # Центр корпуса — это cm_required
        x_total = self.empty_weight * self.cm_required[0]
        y_total = self.empty_weight * self.cm_required[1]
        z_total = self.empty_weight * self.cm_required[2]

        for eq in positioned_eq:
            center_x = eq.position[0] + eq.dimensions[0] / 2
            center_y = eq.position[1] + eq.dimensions[1] / 2
            center_z = eq.position[2] + eq.dimensions[2] / 2

            x_total += eq.mass * center_x
            y_total += eq.mass * center_y
            z_total += eq.mass * center_z

        x_cm = x_total / total_mass
        y_cm = y_total / total_mass
        z_cm = z_total / total_mass

        return (x_cm, y_cm, z_cm)

    def get_cm_deviation(self):
        """Рассчитывает отклонение центра масс от требуемого положения"""
        if None in self.cm_required:
            return 0

        current_cm = self.calculate_center_of_mass()
        deviation = np.sqrt((current_cm[0] - self.cm_required[0]) ** 2 +
                            (current_cm[1] - self.cm_required[1]) ** 2 +
                            (current_cm[2] - self.cm_required[2]) ** 2)
        return deviation

    def __str__(self):
        return f"ID: {self.id}, Название: {self.name}, Размеры: {self.dimensions}, " \
               f"Масса: {self.empty_weight} кг, Макс. нагрузка: {self.max_payload} кг"


class LayoutOptimizer:
    """Класс для оптимизации компоновки оборудования на БПЛА"""

    def __init__(self, uav):
        self.uav = uav
        self.best_layout = None
        self.best_score = float('-inf')

    def can_place_equipment(self, equipment, position):
        """Проверяет возможность размещения оборудования в указанной позиции"""
        x, y, z = position
        length, width, height = equipment.dimensions

        # Проверка выхода за границы пространства БПЛА
        if (x + length > self.uav.dimensions[0] or
                y + width > self.uav.dimensions[1] or
                z + height > self.uav.dimensions[2]):
            return False

        # Проверка пересечения с другим оборудованием
        if np.any(self.uav.occupied_space[z:z + height, y:y + width, x:x + length]):
            return False

        return True

    def place_equipment(self, equipment, position):
        """Размещает оборудование в указанной позиции"""
        # Если модуль уже установлен — удалить его от старой позиции
        if equipment.position is not None:
            self.remove_equipment_from_position(equipment)

        x, y, z = position
        length, width, height = equipment.dimensions

        # Занимаем пространство
        self.uav.occupied_space[z:z + height, y:y + width, x:x + length] = True
        equipment.position = position

    def remove_equipment_from_position(self, equipment):
        """Удаляет оборудование с текущей позиции"""
        if equipment.position is None:
            return

        x, y, z = equipment.position
        length, width, height = equipment.dimensions

        self.uav.occupied_space[z:z + height, y:y + width, x:x + length] = False
        equipment.position = None

    def evaluate_layout(self, weights=(0.9, 0.03, 0.03, 0.02, 0.02)):
        """Оценивает текущий вариант компоновки по нескольким критериям"""
        if any(eq.position is None for eq in self.uav.equipment_list):
            return float('-inf')

        # Критерий центровки (экспоненциальный штраф)
        cm_deviation = self.uav.get_cm_deviation()
        normalized_deviation = cm_deviation / 150
        cm_score = - np.exp(normalized_deviation * 2.5)

        # Штраф за удалённость тяжёлых компонентов от требуемого ЦМ
        heavy_penalty = sum(
            eq.mass * np.linalg.norm(np.array(eq.position) - np.array(self.uav.cm_required))
            for eq in self.uav.equipment_list if eq.mass > 0.25
        )
        heavy_penalty = min(heavy_penalty, 150)
        cm_score -= 0.2 * heavy_penalty / 150

        # Дополнительный штраф за размещение вдали от центра
        edge_penalty = sum(
            np.linalg.norm(
                np.array(eq.position) + np.array(eq.dimensions) / 2 - np.array(self.uav.cm_required)
            )
            for eq in self.uav.equipment_list if eq.position is not None
        ) / len(self.uav.equipment_list)

        cm_score -= 0.01 * edge_penalty

        # Критерий электромагнитной совместимости (ограниченный вклад)
        em_score = 0
        for i, eq1 in enumerate(self.uav.equipment_list):
            for j, eq2 in enumerate(self.uav.equipment_list):
                if i != j:
                    dist = np.linalg.norm(np.array(eq1.position) - np.array(eq2.position))
                    if dist > 0:
                        penalty = (eq1.em_radiation * eq2.em_sensitivity) / (dist ** 2)
                        em_score -= min(penalty, 3)
        em_score = max(em_score, -10)

        # Критерий распределения массы
        md_score = 0
        for i, eq1 in enumerate(self.uav.equipment_list):
            for j, eq2 in enumerate(self.uav.equipment_list):
                if i != j:
                    dist = np.linalg.norm(np.array(eq1.position) - np.array(eq2.position))
                    if dist > 0:
                        md_score -= (eq1.mass * eq2.mass) / dist
        md_score = max(md_score, -5)

        # Критерий теплового режима
        th_score = 0
        for i, eq1 in enumerate(self.uav.equipment_list):
            for j, eq2 in enumerate(self.uav.equipment_list):
                if i != j:
                    dist = np.linalg.norm(np.array(eq1.position) - np.array(eq2.position))
                    if dist > 0:
                        th_score -= (eq1.heat_emission * eq2.heat_emission) / (dist ** 2)
        th_score = max(th_score, -5)

        # Интегральный критерий (итоговая оценка должна быть в разумных пределах)
        total_score = (weights[0] * cm_score +
                       weights[1] * em_score +
                       weights[2] * md_score +
                       weights[3] * th_score)

        total_score = max(total_score, -10)

        return total_score

    def combinatorial_algorithm(self, step=5, max_iterations=1000):
        """Комбинаторный алгоритм оптимизации размещения оборудования"""
        start_time = time.time()

        # Сортировка оборудования по приоритету
        sorted_equipment = sorted(self.uav.equipment_list,
                                  key=lambda eq: eq.priority, reverse=True)

        # Очистка текущего размещения
        for eq in self.uav.equipment_list:
            self.remove_equipment_from_position(eq)

        # Последовательное размещение компонентов
        for eq in sorted_equipment:
            best_pos = None
            best_pos_score = float('-inf')

            # Перебор возможных позиций
            for x in range(0, self.uav.dimensions[0] - eq.dimensions[0] + 1, step):
                for y in range(0, self.uav.dimensions[1] - eq.dimensions[1] + 1, step):
                    for z in range(0, self.uav.dimensions[2] - eq.dimensions[2] + 1, step):
                        if not self.can_place_equipment(eq, (x, y, z)):
                            print(f"❌ Нельзя разместить {eq.name} в позиции {(x, y, z)}")
                        else:
                            # Временное размещение для оценки
                            self.place_equipment(eq, (x, y, z))
                            score = self.evaluate_layout()
                            print(f"✅ {eq.name} МОЖНО разместить в {(x, y, z)} — оценка {score:.2f}")
                            self.remove_equipment_from_position(eq)

                            if best_pos is None or score > best_pos_score:
                                best_pos_score = score
                                best_pos = (x, y, z)

            # Размещение в лучшей позиции
            if best_pos is not None:
                self.place_equipment(eq, best_pos)
                print(f"✅ Установлен модуль: {eq.name} в {best_pos}")
                print(f"Суммарная занятость: {np.sum(self.uav.occupied_space)} ячеек")
            else:
                print(f"Не удалось разместить оборудование {eq.name}")
                return None

        # Локальная оптимизация
        iterations = 0
        while iterations < max_iterations:
            improved = False

            # Попытка обмена позициями
            for i, eq1 in enumerate(self.uav.equipment_list):
                for j, eq2 in enumerate(self.uav.equipment_list[i + 1:], i + 1):
                    # Временное сохранение позиций
                    pos1, pos2 = eq1.position, eq2.position

                    # Удаление с текущих позиций
                    self.remove_equipment_from_position(eq1)
                    self.remove_equipment_from_position(eq2)

                    # Попытка обмена позициями
                    can_swap = (self.can_place_equipment(eq1, pos2) and
                                self.can_place_equipment(eq2, pos1))

                    if can_swap:
                        # Временное размещение для оценки
                        self.place_equipment(eq1, pos2)
                        self.place_equipment(eq2, pos1)
                        new_score = self.evaluate_layout()

                        # Возврат на исходные позиции
                        self.remove_equipment_from_position(eq1)
                        self.remove_equipment_from_position(eq2)

                        # Если обмен улучшает оценку, принимаем его
                        if new_score > self.best_score:
                            self.place_equipment(eq1, pos2)
                            self.place_equipment(eq2, pos1)
                            self.best_score = new_score
                            improved = True
                        else:
                            self.place_equipment(eq1, pos1)
                            self.place_equipment(eq2, pos2)
                    else:
                        # Возврат на исходные позиции
                        self.place_equipment(eq1, pos1)
                        self.place_equipment(eq2, pos2)

            iterations += 1
            if not improved:
                break

        end_time = time.time()
        print(f"Оптимизация завершена за {end_time - start_time:.2f} секунд")

        # Сохранение лучшего варианта
        self.best_layout = {eq.id: eq.position for eq in self.uav.equipment_list}
        self.best_score = self.evaluate_layout()

        return self.best_layout

    def visualize_layout(self):
        """Визуализирует текущую компоновку оборудования"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Отображение границ БПЛА
        max_x, max_y, max_z = self.uav.dimensions
        ax.plot([0, max_x, max_x, 0, 0], [0, 0, max_y, max_y, 0], [0, 0, 0, 0, 0], 'k-')
        ax.plot([0, max_x, max_x, 0, 0], [0, 0, max_y, max_y, 0], [max_z, max_z, max_z, max_z, max_z], 'k-')
        ax.plot([0, 0], [0, 0], [0, max_z], 'k-')
        ax.plot([max_x, max_x], [0, 0], [0, max_z], 'k-')
        ax.plot([max_x, max_x], [max_y, max_y], [0, max_z], 'k-')
        ax.plot([0, 0], [max_y, max_y], [0, max_z], 'k-')

        # Отображение оборудования
        colors = plt.cm.jet(np.linspace(0, 1, len(self.uav.equipment_list)))
        for i, eq in enumerate(self.uav.equipment_list):
            if eq.position is not None:
                x, y, z = eq.position
                dx, dy, dz = eq.dimensions

                # Создание параллелепипеда
                xx, yy = np.meshgrid([x, x + dx], [y, y + dy])
                ax.plot_surface(xx, yy, z * np.ones(xx.shape), color=colors[i], alpha=0.3)
                ax.plot_surface(xx, yy, (z + dz) * np.ones(xx.shape), color=colors[i], alpha=0.3)

                yy, zz = np.meshgrid([y, y + dy], [z, z + dz])
                ax.plot_surface(x * np.ones(yy.shape), yy, zz, color=colors[i], alpha=0.3)
                ax.plot_surface((x + dx) * np.ones(yy.shape), yy, zz, color=colors[i], alpha=0.3)

                xx, zz = np.meshgrid([x, x + dx], [z, z + dz])
                ax.plot_surface(xx, y * np.ones(xx.shape), zz, color=colors[i], alpha=0.3)
                ax.plot_surface(xx, (y + dy) * np.ones(xx.shape), zz, color=colors[i], alpha=0.3)

                # Добавление метки
                ax.text(x + dx / 2, y + dy / 2, z + dz / 2, eq.name, color='black')

        # Отображение центра масс
        actual_cm = self.uav.calculate_center_of_mass()
        ax.scatter([actual_cm[0]], [actual_cm[1]], [actual_cm[2]], color='red', s=100, marker='*')
        ax.text(actual_cm[0], actual_cm[1], actual_cm[2], 'Центр масс', color='red')

        # Если задан требуемый центр масс, отображаем его
        if None not in self.uav.cm_required:
            ax.scatter([self.uav.cm_required[0]], [self.uav.cm_required[1]],
                       [self.uav.cm_required[2]], color='green', s=100, marker='*')
            ax.text(self.uav.cm_required[0], self.uav.cm_required[1],
                    self.uav.cm_required[2], 'Требуемый ЦМ', color='green')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Компоновка оборудования на БПЛА "{self.uav.name}"')

        plt.tight_layout()
        plt.show()


# Пример использования
if __name__ == "__main__":
    # Создание БПЛА
    uav = UAV(id=1, name="Квадрокоптер F-550", length=160, width=160, height=75,
              empty_weight=1.5, max_payload=2.5, cm_required=(80, 80, 37.5))

    # Создание оборудования
    eq1 = Equipment(id=1, name="Бортовой компьютер", length=100, width=80, height=20,
                    mass=0.3, heat_emission=3.0, em_radiation=0.6, em_sensitivity=0.7, priority=1)

    eq2 = Equipment(id=2, name="Аккумулятор", length=120, width=80, height=30,
                    mass=0.5, heat_emission=2.5, em_radiation=0.1, em_sensitivity=0.3, priority=3)

    eq3 = Equipment(id=3, name="GPS-приемник", length=50, width=30, height=15,
                    mass=0.1, heat_emission=0.5, em_radiation=0.2, em_sensitivity=0.8, priority=2)

    eq4 = Equipment(id=4, name="Радиопередатчик", length=60, width=50, height=20,
                    mass=0.15, heat_emission=5.0, em_radiation=8.0, em_sensitivity=1.0, priority=2)

    eq5 = Equipment(id=4, name="Видеокамера", length=70, width=40, height=30,
                    mass=0.1, heat_emission=2.5, em_radiation=3.0, em_sensitivity=4.0, priority=2)

    eq6 = Equipment(id=4, name="Модуль телеметрии", length=50, width=30, height=15,
                    mass=0.08, heat_emission=1.5, em_radiation=6.0, em_sensitivity=2.0, priority=3)

    eq7 = Equipment(id=7, name="Калибровочный модуль", length=60, width=40, height=20,
                    mass=0.06, heat_emission=0.3, em_radiation=0.1, em_sensitivity=0.1, priority=3)

    # Добавление оборудования в БПЛА
    uav.add_equipment(eq1)
    uav.add_equipment(eq2)
    uav.add_equipment(eq3)
    uav.add_equipment(eq4)
    uav.add_equipment(eq5)
    uav.add_equipment(eq6)
    uav.add_equipment(eq7)

    # Создание оптимизатора компоновки
    optimizer = LayoutOptimizer(uav)

    # Запуск комбинаторного алгоритма
    print("Запуск комбинаторного алгоритма оптимизации...")
    layout = optimizer.combinatorial_algorithm(step=10)

    if layout:
        print("\nРезультаты компоновки:")
        for eq in uav.equipment_list:
            print(f"{eq.name}: позиция {eq.position}")

        print(f"\nЦентр масс: {uav.calculate_center_of_mass()}")
        print(f"Отклонение от требуемого центра масс: {uav.get_cm_deviation():.2f}")
        print(f"Общая оценка компоновки: {optimizer.best_score:.2f}")

        # Визуализация результатов
        optimizer.visualize_layout()


class ElmanNetwork:
    """Класс для реализации рекуррентной нейронной сети Элмана"""

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Инициализация весов
        self.W_input_hidden = np.random.uniform(-0.5, 0.5, (hidden_size, input_size))
        self.W_context_hidden = np.random.uniform(-0.5, 0.5, (hidden_size, hidden_size))
        self.W_hidden_output = np.random.uniform(-0.5, 0.5, (output_size, hidden_size))

        # Инициализация смещений
        self.b_hidden = np.zeros((hidden_size, 1))
        self.b_output = np.zeros((output_size, 1))

        # Инициализация контекстного слоя
        self.context = np.zeros((hidden_size, 1))

        # Для хранения активаций
        self.hidden_activations = None
        self.output_activations = None

    def tanh(self, x):
        """Функция активации гиперболический тангенс"""
        return np.tanh(x)

    def tanh_derivative(self, x):
        """Производная функции активации гиперболический тангенс"""
        return 1.0 - np.tanh(x) ** 2

    def forward(self, x):
        """Прямое распространение сигнала через сеть"""
        # Преобразование входа в вектор-столбец
        x = np.array(x).reshape(-1, 1)

        # Вычисление активаций скрытого слоя
        hidden_input = np.dot(self.W_input_hidden, x) + np.dot(self.W_context_hidden, self.context) + self.b_hidden
        self.hidden_activations = self.tanh(hidden_input)

        # Обновление контекстного слоя
        self.context = self.hidden_activations.copy()

        # Вычисление выходного слоя
        output_input = np.dot(self.W_hidden_output, self.hidden_activations) + self.b_output
        self.output_activations = output_input  # Линейная активация для выходного слоя

        return self.output_activations.flatten()

    def backward(self, x, target, reset_context=False):
        """Обратное распространение ошибки и обновление весов"""
        # Преобразование входа и цели в векторы-столбцы
        x = np.array(x).reshape(-1, 1)
        target = np.array(target).reshape(-1, 1)

        # Вычисление ошибки выходного слоя
        output_error = target - self.output_activations

        # Вычисление градиента для весов между скрытым и выходным слоями
        dW_hidden_output = np.dot(output_error, self.hidden_activations.T)
        db_output = output_error

        # Вычисление ошибки скрытого слоя
        hidden_error = np.dot(self.W_hidden_output.T, output_error)
        hidden_delta = hidden_error * self.tanh_derivative(self.hidden_activations)

        # Вычисление градиента для весов между входным и скрытым слоями
        dW_input_hidden = np.dot(hidden_delta, x.T)
        dW_context_hidden = np.dot(hidden_delta, self.context.T)
        db_hidden = hidden_delta

        # Обновление весов
        self.W_hidden_output += self.learning_rate * dW_hidden_output
        self.b_output += self.learning_rate * db_output
        self.W_input_hidden += self.learning_rate * dW_input_hidden
        self.W_context_hidden += self.learning_rate * dW_context_hidden
        self.b_hidden += self.learning_rate * db_hidden

        # Опционально сбрасываем контекстный слой
        if reset_context:
            self.context = np.zeros((self.hidden_size, 1))

        # Возвращаем среднеквадратичную ошибку
        return np.mean(output_error ** 2)

    def train(self, training_data, targets, epochs=600, batch_size=32, validation_data=None):
        """Обучение нейронной сети"""
        print("Начало обучения нейронной сети...")
        training_errors = []
        validation_errors = []

        for epoch in tqdm(range(epochs), desc="Обучение нейросети"):
            # Перемешивание обучающих данных
            indices = np.random.permutation(len(training_data))
            training_data_shuffled = [training_data[i] for i in indices]
            targets_shuffled = [targets[i] for i in indices]

            epoch_errors = []

            # Обучение по мини-батчам
            for i in range(0, len(training_data_shuffled), batch_size):
                batch_data = training_data_shuffled[i:i + batch_size]
                batch_targets = targets_shuffled[i:i + batch_size]

                batch_error = 0
                for j in range(len(batch_data)):
                    # Прямое распространение
                    self.forward(batch_data[j])

                    # Обратное распространение
                    error = self.backward(batch_data[j], batch_targets[j], reset_context=(j == len(batch_data) - 1))
                    batch_error += error

                epoch_errors.append(batch_error / len(batch_data))

            # Средняя ошибка за эпоху
            avg_error = np.mean(epoch_errors)
            if np.isnan(avg_error):
                print(f"⚠️ Ошибка обучения = NaN на эпохе {epoch + 1}. Возможно, не хватает данных.")
                break
            training_errors.append(avg_error)

            # Валидация, если предоставлены данные
            if validation_data is not None:
                val_data, val_targets = validation_data
                val_errors = []

                for j in range(len(val_data)):
                    # Прямое распространение
                    pred = self.forward(val_data[j])

                    # Вычисление ошибки
                    error = np.mean((pred - val_targets[j]) ** 2)
                    val_errors.append(error)

                avg_val_error = np.mean(val_errors)
                validation_errors.append(avg_val_error)

                print(
                    f"Эпоха {epoch + 1}/{epochs}, Ошибка обучения: {avg_error:.6f}, Ошибка валидации: {avg_val_error:.6f}")
            else:
                print(f"Эпоха {epoch + 1}/{epochs}, Ошибка обучения: {avg_error:.6f}")

        print("Обучение завершено")
        return training_errors, validation_errors

    def predict(self, x):
        """Предсказание для входных данных"""
        return self.forward(x)

    def save_model(self, filename):
        """Сохранение модели в файл"""
        model_data = {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'learning_rate': self.learning_rate,
            'W_input_hidden': self.W_input_hidden.tolist(),
            'W_context_hidden': self.W_context_hidden.tolist(),
            'W_hidden_output': self.W_hidden_output.tolist(),
            'b_hidden': self.b_hidden.tolist(),
            'b_output': self.b_output.tolist()
        }

        with open(filename, 'w') as f:
            json.dump(model_data, f)

        print(f"Модель сохранена в файл {filename}")

    @classmethod
    def load_model(cls, filename):
        """Загрузка модели из файла"""
        with open(filename, 'r') as f:
            model_data = json.load(f)

        network = cls(
            model_data['input_size'],
            model_data['hidden_size'],
            model_data['output_size'],
            model_data['learning_rate']
        )

        network.W_input_hidden = np.array(model_data['W_input_hidden'])
        network.W_context_hidden = np.array(model_data['W_context_hidden'])
        network.W_hidden_output = np.array(model_data['W_hidden_output'])
        network.b_hidden = np.array(model_data['b_hidden'])
        network.b_output = np.array(model_data['b_output'])

        print(f"Модель загружена из файла {filename}")
        return network


class NeuralLayoutOptimizer:
    """Класс для оптимизации компоновки с использованием нейронной сети"""

    def __init__(self, uav, network=None):
        self.uav = uav
        self.network = network
        self.layout_optimizer = LayoutOptimizer(uav)

    def prepare_input_data(self, equipment):
        """Подготовка входных данных для нейронной сети"""
        # Нормализация габаритов оборудования
        normalized_dimensions = [
            equipment.dimensions[0] / self.uav.dimensions[0],
            equipment.dimensions[1] / self.uav.dimensions[1],
            equipment.dimensions[2] / self.uav.dimensions[2]
        ]

        # Нормализация массы
        normalized_mass = equipment.mass / self.uav.max_payload

        # Нормализация характеристик ЭМС и тепловыделения
        normalized_em_radiation = equipment.em_radiation / 10.0  # Предполагаем максимальное значение 10
        normalized_em_sensitivity = equipment.em_sensitivity / 10.0
        normalized_heat_emission = equipment.heat_emission / 10.0

        # Приоритет размещения
        normalized_priority = equipment.priority / 10.0  # Предполагаем максимальный приоритет 10

        # Матрица занятости (упрощенно - процент занятого пространства)
        occupied_percentage = np.sum(self.uav.occupied_space) / np.prod(self.uav.occupied_space.shape)

        # Текущий центр масс
        current_cm = self.uav.calculate_center_of_mass()
        normalized_cm = [
            current_cm[0] / self.uav.dimensions[0],
            current_cm[1] / self.uav.dimensions[1],
            current_cm[2] / self.uav.dimensions[2]
        ]

        # Формирование входного вектора
        input_data = normalized_dimensions + [normalized_mass, normalized_em_radiation,
                                              normalized_em_sensitivity, normalized_heat_emission,
                                              normalized_priority, occupied_percentage] + normalized_cm

        return input_data

    def denormalize_output(self, output):
        """Преобразование выхода нейронной сети в координаты"""
        x = int(min(max(output[0], 0.0), 1.0) * self.uav.dimensions[0])
        y = int(min(max(output[1], 0.0), 1.0) * self.uav.dimensions[1])
        z = int(min(max(output[2], 0.0), 1.0) * self.uav.dimensions[2])

        return (x, y, z)

    def generate_training_data(self, num_samples=100):
        """Генерация обучающих данных с использованием комбинаторного алгоритма"""
        print("Генерация обучающих данных...")
        training_inputs = []
        training_outputs = []
        print(f"Размеры БПЛА: {self.uav.dimensions}")
        for eq in self.uav.equipment_list:
            print(f"{eq.name}: размеры {eq.dimensions}, масса: {eq.mass}")

        for _ in tqdm(range(num_samples), desc="Генерация обучающих данных"):
            # Сброс состояния БПЛА
            self.uav.occupied_space.fill(False)
            for eq in self.uav.equipment_list:
                eq.position = None

            # Сортировка оборудования по приоритету
            sorted_equipment = sorted(self.uav.equipment_list,
                                      key=lambda eq: eq.priority, reverse=True)

            for eq in sorted_equipment:
                best_pos = None
                best_score = float('-inf')

                for x in range(0, self.uav.dimensions[0] - eq.dimensions[0] + 1, 10):
                    for y in range(0, self.uav.dimensions[1] - eq.dimensions[1] + 1, 10):
                        for z in range(0, self.uav.dimensions[2] - eq.dimensions[2] + 1, 10):
                            if self.layout_optimizer.can_place_equipment(eq, (x, y, z)):
                                self.layout_optimizer.place_equipment(eq, (x, y, z))
                                score = self.layout_optimizer.evaluate_layout()
                                self.layout_optimizer.remove_equipment_from_position(eq)

                                if best_pos is None or score > best_score:
                                    best_score = score
                                    best_pos = (x, y, z)

                if best_pos is not None:
                    input_data = self.prepare_input_data(eq)
                    self.layout_optimizer.place_equipment(eq, best_pos)

                    normalized_pos = [
                        best_pos[0] / self.uav.dimensions[0],
                        best_pos[1] / self.uav.dimensions[1],
                        best_pos[2] / self.uav.dimensions[2]
                    ]

                    training_inputs.append(input_data)
                    training_outputs.append(normalized_pos)

        print(f"✅ Сгенерировано {len(training_inputs)} обучающих примеров")
        if len(training_inputs) == 0:
            print("❌ Недостаточно данных для обучения. Нейросеть не будет обучена.")
            return None, None

        return training_inputs, training_outputs

    def train_network(self, hidden_size=75, epochs=600, batch_size=32):
        """Обучение нейронной сети"""
        # Генерация обучающих данных
        training_data, targets = self.generate_training_data()
        if training_data is None or len(training_data) == 0:
            print("❌ Не удалось сгенерировать обучающие данные. Проверь параметры оборудования или компоновку.")
            return None

        # Разделение на обучающую и валидационную выборки
        split_idx = int(0.85 * len(training_data))
        train_data, val_data = training_data[:split_idx], training_data[split_idx:]
        train_targets, val_targets = targets[:split_idx], targets[split_idx:]

        # Создание и обучение сети
        input_size = len(training_data[0])
        output_size = 3  # x, y, z координаты

        self.network = ElmanNetwork(input_size, hidden_size, output_size)
        training_errors, validation_errors = self.network.train(
            train_data, train_targets, epochs, batch_size, (val_data, val_targets)
        )

        # Визуализация процесса обучения
        plt.figure(figsize=(10, 6))
        plt.plot(training_errors, label='Ошибка обучения')
        plt.plot(validation_errors, label='Ошибка валидации')
        plt.xlabel('Эпоха')
        plt.ylabel('Среднеквадратичная ошибка')
        plt.title('Процесс обучения нейронной сети')
        plt.legend()
        plt.grid(True)
        plt.show()

        return self.network

    def optimize_layout(self):
        """Оптимизация компоновки с использованием обученной нейронной сети"""
        if self.network is None:
            print("Нейронная сеть не обучена. Используйте метод train_network() для обучения.")
            return None

        start_time = time.time()

        # Сброс состояния БПЛА
        self.uav.occupied_space.fill(False)
        for eq in self.uav.equipment_list:
            eq.position = None

        # Сортировка оборудования по приоритету
        sorted_equipment = sorted(self.uav.equipment_list,
                                  key=lambda eq: eq.priority, reverse=True)

        # Последовательное размещение оборудования
        for eq in sorted_equipment:
            # Подготовка входных данных для сети
            input_data = self.prepare_input_data(eq)

            # Получение предсказания от сети
            output = self.network.predict(input_data)

            # Преобразование выхода в координаты
            predicted_pos = self.denormalize_output(output)

            # Проверка возможности размещения
            if self.layout_optimizer.can_place_equipment(eq, predicted_pos):
                self.layout_optimizer.place_equipment(eq, predicted_pos)
            else:
                # Если предсказанная позиция недоступна, ищем ближайшую доступную
                best_pos = None
                min_distance = float('inf')

                for x in range(0, self.uav.dimensions[0] - eq.dimensions[0] + 1, 5):
                    for y in range(0, self.uav.dimensions[1] - eq.dimensions[1] + 1, 5):
                        for z in range(0, self.uav.dimensions[2] - eq.dimensions[2] + 1, 5):
                            if self.layout_optimizer.can_place_equipment(eq, (x, y, z)):
                                distance = np.sqrt((x - predicted_pos[0]) ** 2 +
                                                   (y - predicted_pos[1]) ** 2 +
                                                   (z - predicted_pos[2]) ** 2)

                                if distance < min_distance:
                                    min_distance = distance
                                    best_pos = (x, y, z)

                if best_pos is not None:
                    self.layout_optimizer.place_equipment(eq, best_pos)
                else:
                    print(f"Не удалось разместить оборудование {eq.name}")
                    return None

        end_time = time.time()
        print(f"Оптимизация с использованием нейронной сети завершена за {end_time - start_time:.2f} секунд")

        # Оценка полученной компоновки
        layout_score = self.layout_optimizer.evaluate_layout()
        print(f"Оценка компоновки: {layout_score:.2f}")

        return {eq.id: eq.position for eq in self.uav.equipment_list}

    def compare_methods(self):
        """Сравнение эффективности комбинаторного алгоритма и нейросетевого подхода"""
        print("Сравнение эффективности методов оптимизации компоновки:")

        # Сброс состояния БПЛА
        self.uav.occupied_space.fill(False)
        for eq in self.uav.equipment_list:
            eq.position = None

        # Измерение времени работы комбинаторного алгоритма
        start_time = time.time()
        combinatorial_layout = self.layout_optimizer.combinatorial_algorithm(step=10)
        combinatorial_time = time.time() - start_time
        combinatorial_score = self.layout_optimizer.evaluate_layout()

        # Сброс состояния БПЛА
        self.uav.occupied_space.fill(False)
        for eq in self.uav.equipment_list:
            eq.position = None

        # Измерение времени работы нейросетевого подхода
        start_time = time.time()
        neural_layout = self.optimize_layout()
        neural_time = time.time() - start_time
        neural_score = self.layout_optimizer.evaluate_layout()

        # Вывод результатов сравнения
        print("\nРезультаты сравнения:")
        print(f"Комбинаторный алгоритм: время = {combinatorial_time:.2f} с, оценка = {combinatorial_score:.2f}")
        print(f"Нейросетевой подход: время = {neural_time:.2f} с, оценка = {neural_score:.2f}")
        print(f"Ускорение: {combinatorial_time / neural_time:.1f}x")
        if combinatorial_score < 0 and neural_score < 0:
            if combinatorial_score < 0 and neural_score < 0:
                relative_accuracy = combinatorial_score / neural_score * 100
            elif combinatorial_score > 0 and neural_score > 0:
                relative_accuracy = combinatorial_score / neural_score * 100
            else:
                relative_accuracy = 0

            print(f"Относительная точность: {relative_accuracy:.1f}%")

        if relative_accuracy > 100:
            print("✅ Нейросеть показала лучшую компоновку, чем перебор.")
        elif relative_accuracy == 100:
            print("➖ Результаты нейросети и комбинаторного алгоритма идентичны.")
        else:
            print("⚠️ Комбинаторный алгоритм дал лучший результат.")

        return {
            'combinatorial': {'time': combinatorial_time, 'score': combinatorial_score},
            'neural': {'time': neural_time, 'score': neural_score}
        }


# Пример использования нейросетевого оптимизатора
if __name__ == "__main__":
    # Создание БПЛА
    uav = UAV(id=1, name="Квадрокоптер F-550", length=160, width=160, height=75,
              empty_weight=1.5, max_payload=2.5, cm_required=(80, 80, 37.5))

    # Создание оборудования
    eq1 = Equipment(id=1, name="Бортовой компьютер", length=100, width=80, height=20,
                   mass=0.3, heat_emission=3.0, em_radiation=0.6, em_sensitivity=0.7, priority=1)

    eq2 = Equipment(id=2, name="Аккумулятор", length=120, width=80, height=30,
                    mass=0.5, heat_emission=2.5, em_radiation=0.1, em_sensitivity=0.3, priority=3)

    eq3 = Equipment(id=3, name="GPS-приемник", length=50, width=30, height=15,
                    mass=0.1, heat_emission=0.5, em_radiation=0.2, em_sensitivity=0.8, priority=2)

    eq4 = Equipment(id=4, name="Радиопередатчик", length=60, width=50, height=20,
                    mass=0.15, heat_emission=5.0, em_radiation=8.0, em_sensitivity=1.0, priority=2)

    eq5 = Equipment(id=4, name="Видеокамера", length=70, width=40, height=30,
                    mass=0.1, heat_emission=2.5, em_radiation=3.0, em_sensitivity=4.0, priority=2)

    eq6 = Equipment(id=4, name="Модуль телеметрии", length=50, width=30, height=15,
                    mass=0.08, heat_emission=1.5, em_radiation=6.0, em_sensitivity=2.0, priority=3)

    eq7 = Equipment(id=7, name="Калибровочный модуль", length=60, width=40, height=20,
                    mass=0.06, heat_emission=0.3, em_radiation=0.1, em_sensitivity=0.1, priority=3)

    # Добавление оборудования в БПЛА
    uav.add_equipment(eq1)
    uav.add_equipment(eq2)
    uav.add_equipment(eq3)
    uav.add_equipment(eq4)
    uav.add_equipment(eq5)
    uav.add_equipment(eq6)
    uav.add_equipment(eq7)

    # Создание нейросетевого оптимизатора
    neural_optimizer = NeuralLayoutOptimizer(uav)

    # Обучение нейронной сети
    network = neural_optimizer.train_network(hidden_size=75, epochs=100, batch_size=32)

    # Сохранение обученной модели
    if network is not None:
        network.save_model("uav_layout_model.json")
    else:
        print("Обучение не состоялось, модель не сохранена.")

    # Оптимизация компоновки с использованием нейронной сети
    print("\nОптимизация компоновки с использованием нейронной сети:")
    neural_layout = neural_optimizer.optimize_layout()

    if neural_layout:
        print("\nРезультаты нейросетевой компоновки:")
        for eq in uav.equipment_list:
            print(f"{eq.name}: позиция {eq.position}")

        print(f"\nЦентр масс: {uav.calculate_center_of_mass()}")
        print(f"Отклонение от требуемого центра масс: {uav.get_cm_deviation():.2f}")

        # Визуализация результатов
        neural_optimizer.layout_optimizer.visualize_layout()

    # Сравнение эффективности методов
    comparison_results = neural_optimizer.compare_methods()

    # Визуализация сравнения времени выполнения
    methods = ['Комбинаторный алгоритм', 'Нейронная сеть']
    times = [comparison_results['combinatorial']['time'], comparison_results['neural']['time']]

    plt.figure(figsize=(10, 6))
    plt.bar(methods, times, color=['blue', 'orange'])
    plt.ylabel('Время выполнения (с)')
    plt.title('Сравнение времени выполнения методов оптимизации компоновки')
    plt.grid(axis='y')

    # Добавление значений времени над столбцами
    for i, v in enumerate(times):
        plt.text(i, v + 0.1, f"{v:.2f} с", ha='center')

    plt.tight_layout()
    plt.show()
