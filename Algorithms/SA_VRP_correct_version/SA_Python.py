import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import operator
import itertools

# Функция для вычисления расстояния между двумя городами
def distance(x1, y1, x2, y2):
    res = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return res

# Функция для вычисление длины всего тура
def totaldistancetour(tour, cities, depot):
    d = 0
    for i in range(1, len(tour)):
        x1 = cities[int(tour[i - 1])][0]
        y1 = cities[int(tour[i - 1])][1]
        x2 = cities[int(tour[i])][0]
        y2 = cities[int(tour[i])][1]
        d = d + distance(x1, y1, x2, y2)
    
    x1 = cities[int(tour[len(tour) - 1])][0]
    y1 = cities[int(tour[len(tour) - 1])][1]
    x2 = depot[0][0]
    y2 = depot[0][1]
    d = d + distance(x1, y1, x2, y2)
    
    x1 = cities[int(tour[0])][0]
    y1 = cities[int(tour[0])][1]
    x2 = depot[0][0]
    y2 = depot[0][1]
    d = d + distance(x1, y1, x2, y2)
    return d

def subtourslice(tour, vehicle, max_vehicle):
    #print("--------------------------------Начало работы функции--------------------------------")
    capacity_used = np.zeros(len(vehicle))
    k = 0
    slice = []
    mass = [] # список всех весов грузов для перевозки
    #print(len(vehicle))
    for i in range(len(vehicle)):
        # capacity_used[i] - начальный вес
        # vehicle[i][1] - это значение из колонки 8000, то есть максимальный объем
        # vehicle[i][0] - это первая колонка, то есть номер грузовика
        #print("Начальный суммарный вес: ", capacity_used[i])
        while((capacity_used[i] <= max_vehicle[i]) and (k <= (len(tour) - 1))):
            # cities[tour[k]][2] - кол-во груза, который грузовик везет в город k (третья колнка в первой таблице)
            # print("Номер города для входа: ", k)
            # print("Временный суммарный вес: ", capacity_used[i])
            # print("Вес груза, который нужно перевезти в текущий город: ", cities[tour[k]][2])
            capacity_used[i] += vehicle[int(tour[k])] # заполняем грузовик конкретным кол-вом груза для каждого города
            if(capacity_used[i] > max_vehicle[i]):
                # если кол-во груза, который нужно отвезти, больше чем вместимость грузовика, то вычитаем это кол-во груза
                capacity_used[i] -= vehicle[int(tour[k])]
                k -= 1 # это номер города после которого суммарный вес > объема грузовика и дальнейший вес груза не учитываем
                # print("Номер города, после которого суммарный вес > объема грузовика: ", k)
                # после такого города, нужно полностью очистить грузовик и дальше отправить его по городам
                slice.append(k) # запоминаем номер тура, чтобы повторно не проходить по такому же маршруту
                k += 1
                break
            k += 1
            #print(' ')
        #print(' ')
        mass.append(capacity_used[i])
    slice.append(k - 1)
    return (slice, mass)

# Что происходит в цикле while
# Мы на каждом шаге суммарный вес груза увеличиваем для каждого грузовика пока он не превысит максимальный объем вмесимости
# грузовика. Если это происходит, то мы запоминаем предыдущий номер города после которого общий вес превысиль обьем грузовика,
# возвращаемся к номеру города, на котором остановились и начинаем все заново.

# print(subtourslice(tour, vehicle))

def subtour(slice, tour):
    sub = []
    # добавляем номера городов, после которых суммарный вес груза превышал объем грузовика
    sub.append(tour[:(slice[0] + 1)]) # создаем список от 0 до 8
    for i in range(0, len(slice) - 1):
        # нарежем маршруты городов до момента, пока суммарный вес груза не превысел объем грузовика
        sub.append(tour[(slice[i] + 1):(slice[i + 1] + 1)]) # 1:5 -> 1, 2, 3, 4 || 1, 2, 3, 4, 5
        # создает список от 9 до 15, потом от 16 до 20
    #print(sub)
    return sub

# Вычисление общей длины всех туров

def all_vechile_distance(sub, cities, depot):
    all_distance = reduce(operator.add, (totaldistancetour(x, cities, depot) for x in sub))
    return all_distance

# Функция определения общего расстояния по данным туров и грузовиков
def tour_to_distance(tour, cities, vehicle, max_vehicle, depot):
    u = subtourslice(tour, vehicle, max_vehicle) # получаем спислок городов, после которых суммарный вес груза > объема грузовика
    v = subtour(u[0], tour) # получим тур из номеров городов, где u[0] это список из номеров городов, после которых суммарный вес груза > объема грузовика
    total = all_vechile_distance(v, cities, depot) # посчитаем суммарное расстояние между всеми городами в туре
    return total

def myShuffle(solution, n_customer):
    for i in range(0, 2 * n_customer):
        a = random.randint(0, len(solution)-1)
        b = random.randint(0, len(solution)-1)
        solution[a], solution[b] = solution[b], solution[a]
    return solution

def Initialize(count): # задаем список из count городов и случайно их перемешиваем
    solution = np.arange(count)
    random.shuffle(solution)#np.copy(myShuffle(solution, count))
    print(solution)
    return solution

def GenerateStateCandidate(current):
    new = current.copy()
    index_a = np.random.randint(len(current))
    index_b = np.random.randint(len(current))
    while index_b == index_a:
        index_b = np.random.randint(len(current))
    if(index_a > index_b):
        new[index_b:index_a] = np.flip(new[index_b:index_a])
    else:
        new[index_a:index_b] = np.flip(new[index_a:index_b])
    return new

def SA(cities, vehicle, max_vehicle, T_end, t_max, depot):
    current_solution = Initialize(len(cities))
    currentEnergy = tour_to_distance(current_solution, cities, vehicle, max_vehicle, depot) # вычисляем энергию для первого состояния
    best_tour = np.copy(current_solution)
    T = t_max
    best_Energy = worst_Energy = currentEnergy
    k = 1
    
    while(T > T_end):  # на всякий случай ограничеваем количество итераций
    # может быть полезно при тестировании сложных функций изменения температуры T       
        #print("Подшаг_шага:", k)
        new_solution = GenerateStateCandidate(current_solution) # получаем новое решение
        candidateEnergy = tour_to_distance(new_solution, cities, vehicle, max_vehicle, depot) # вычисляем его энергию
        #print("Текущая энергия: ", candidateEnergy)
        best_Energy = min(best_Energy, candidateEnergy)
        worst_Energy = max(worst_Energy, candidateEnergy)

        if(candidateEnergy < currentEnergy): # если кандидат обладает меньшей энергией
            currentEnergy = candidateEnergy # то оно становится текущим состоянием
            current_solution = np.copy(new_solution)
            if(currentEnergy < best_Energy):
                best_tour = np.copy(current_solution)
        else:
            p = np.exp((currentEnergy - candidateEnergy) / T) # иначе, считаем вероятность
            if (p > np.random.uniform()): # и смотрим, осуществится ли переход
                currentEnergy = candidateEnergy
                best_tour = np.copy(current_solution)
                current_solution = np.copy(new_solution)
        T = t_max / (k + 1) # уменьшаем температуру
        k += 1
        
    return best_Energy

# Для 20 вершин объем грузовика 500, для 50 вершин объем грузовика 750, для 50 вершин объем грузовика 1000

n_samples = 2
n_customer = 20
max_vehicle_car = 30

T_end = 1
t_max = 100000

rnds = np.random
rnds.seed(0)
# максимальная вместимость грузовика

lst = []
all_results = []

min_capasity = 1
max_capasity = 42
capacity = np.minimum(np.maximum(np.abs(np.random.normal(15, 10, size=[n_samples, n_customer])), min_capasity), max_capasity)

n = 2
max_vehicle_car *= n
while(max_vehicle_car < max_capasity):
    max_vehicle_car = max_vehicle_car / n
    n += 1
    max_vehicle_car *= n
print("Используется грузовичков: ", n)

max_vehicle = [max_vehicle_car] * len(capacity[0])
i = 0
# for k in range(31, 51):
while(i < n_samples):
    # Cоздаем одно депо, n_customer координат городов и n_customer весов
    depot = rnds.uniform(size=(1, 2))  # depot location
    cities = rnds.uniform(size=(n_customer, 2))  # node locations
    vehicle = capacity[i]
    res = SA(cities, vehicle, max_vehicle, T_end, t_max, depot)
    lst.append(res)
    i += 1
#     i = 0
#     all_results.append(lst)
# 23.620673440584618, 25.299787321688644
print(lst)