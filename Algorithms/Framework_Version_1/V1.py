from gurobipy import *
from gurobipy import Model, GRB, quicksum
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import operator
import itertools
import os.path

def distance(x1, y1, x2, y2):
    res = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return res

# Функция для вычисление длины всего тура
def totaldistancetour(cities, tour, depot):
    d = 0
    for i in range(1, len(tour)):
        x1 = cities[int(tour[i - 1])][0]
        y1 = cities[int(tour[i - 1])][1]
        x2 = cities[int(tour[i])][0]
        y2 = cities[int(tour[i])][1]
        d = d + distance(x1, y1, x2, y2)
    
    x1 = cities[int(tour[len(tour) - 1])][0]
    y1 = cities[int(tour[len(tour) - 1])][1]
    x2 = depot[0]
    y2 = depot[1]
    d = d + distance(x1, y1, x2, y2)
    
    x1 = cities[int(tour[0])][0]
    y1 = cities[int(tour[0])][1]
    x2 = depot[0]
    y2 = depot[1]
    d = d + distance(x1, y1, x2, y2)
    return d

def subtourslice(cities, tour, vehicle):
    #print("--------------------------------Начало работы функции--------------------------------")
    capacity_used = np.zeros(len(vehicle))
    k = 0
    slice = []
    mass = [] # список всех весов грузов для перевозки
    for i in range(len(vehicle)):
        # capacity_used[i] - начальный вес
        # vehicle[i][1] - это значение из колонки 8000, то есть максимальный объем
        # vehicle[i][0] - это первая колонка, то есть номер грузовика
        #print("Начальный суммарный вес: ", capacity_used[i])
        while((capacity_used[i] <= vehicle[i][1]) and (k <= (len(tour) - 1))):
            # cities[tour[k]][2] - кол-во груза, который грузовик везет в город k (третья колнка в первой таблице)
            # print("Номер города для входа: ", k)
            # print("Временный суммарный вес: ", capacity_used[i])
            # print("Вес груза, который нужно перевезти в текущий город: ", cities[tour[k]][2])
            capacity_used[i] += cities[int(tour[k])][2] # заполняем грузовик конкретным кол-вом груза для каждого города
            if(capacity_used[i] > vehicle[i][1]):
                # если кол-во груза, который нужно отвезти, больше чем вместимость грузовика, то вычитаем это кол-во груза
                capacity_used[i] -= cities[int(tour[k])][2]
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
    return(slice, mass)

# Что происходит в цикле while
# Мы на каждом шаге суммарный вес груза увеличиваем для каждого грузовика пока он не превысит максимальный объем вмесимости
# грузовика. Если это происходит, то мы запоминаем предыдущий номер города после которого общий вес превысиль обьем грузовика,
# возвращаемся к номеру города, на котором остановились и начинаем все заново.

# print(subtourslice(tour, vehicle))
# slice, mass = subtourslice(tour, vehicle)

def subtour(slice, tour):
    sub = []
    # добавляем номера городов, после которых суммарный вес груза превышал объем грузовика
    sub.append(tour[:(slice[0] + 1)]) # создаем список от 0 до 8
    for i in range(0, len(slice) - 1):
        # нарежем маршруты городов до момента, пока суммарный вес груза не превысел объем грузовика
        sub.append(tour[(slice[i] + 1):(slice[i + 1] + 1)])
        # создает список от 9 до 15, потом от 16 до 20
    return sub

#print(subtour(slice, tour))
# Вычисление общей длины всех туров

# sub = subtour(slice, tour)
def all_vechile_distance(cities, sub, depot):
    all_distance = reduce(operator.add, (totaldistancetour(cities, x, depot) for x in sub), 0)
    return all_distance

#all_vechile_distance(sub)
# Функция определения общего расстояния по данным туров и грузовиков
def tour_to_distance(cities, tour, vehicle, depot):
    u = subtourslice(cities, tour, vehicle) # получаем спислок городов, после которых суммарный вес груза > объема грузовика
    v = subtour(u[0], tour) # получим тур из номеров городов, где u[0] это список из номеров городов, после которых суммарный вес груза > объема грузовика
    total = all_vechile_distance(cities, v, depot) # посчитаем суммарное расстояние между всеми городами в туре
    return total

#tour_to_distance(tour, vehicle)

def Initialize(count): # задаем список из count городов и случайно их перемешиваем
    solution = np.arange(count)
    np.random.shuffle(solution)
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

def SA(cities, vehicle, depot):
    T_end = 1
    t_max = 100000

    current_solution = Initialize(len(cities))
    # print("Начальный маршрут: ", current_solution)
    # print(" ")
    currentEnergy = tour_to_distance(cities, current_solution, vehicle, depot) # вычисляем энергию для первого состояния
    best_tour = np.copy(current_solution)
    T = t_max
    best_Energy = worst_Energy = currentEnergy
    k = 0
    while(T > T_end):  # на всякий случай ограничеваем количество итераций
    # может быть полезно при тестировании сложных функций изменения температуры T       
        new_solution = GenerateStateCandidate(current_solution) # получаем новое решение
        candidateEnergy = tour_to_distance(cities, new_solution, vehicle, depot) # вычисляем его энергию
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
                best_tour = np. copy(current_solution)
                current_solution = np.copy(new_solution)
                
        T = t_max / (k + 1) # уменьшаем температуру
        k += 1
    print(" ")
    print("Худшее расстояние: ", worst_Energy)
    print("Лучшее расстояние: ", best_Energy)
    print("Лучший маршрут: ", best_tour)
    print(" ")

    best_slice = subtourslice(cities, best_tour, vehicle)
    best_subtour = subtour(best_slice[0], best_tour)
    print("Лучшее распределение объема груза по городам:", best_slice)
    print("Лучшее разделение городов по отдельным маршрутам: ", best_subtour)

def main_SA(name_file):
    # Читаем данные
    xlsx = pd.ExcelFile(name_file)
    #sheet1 = pd.read_excel(xlsx, 'Sheet20')
    #sheet2 = pd.read_excel(xlsx, 'Sheet50')
    sheet3 = pd.read_excel(xlsx, 'Sheet100')
    sheet_vehicle = pd.read_excel(xlsx, 'Sheet_vehicle')

    # Описываем данные
    #nodes = sheet1.values # множество координат и количество товара в каждой вершине, который нужно довезти
    #nodes = sheet2.values
    nodes = sheet3.values
    depot = nodes[len(nodes) - 1] # координаты и количество товара в депо
    cities = nodes[:(len(nodes) - 1)] # координаты и количество товара в остальных вершин(городов)
    vehicle = sheet_vehicle.values # первая колонка отвечает за номер траспортного средства
                            # вторая колонка это максимальная вместимость

    #tour = list(range(0, len(cities)))
    #print(tour)
    # Функция для вычисления расстояния между двумя городами
    return SA(cities, vehicle, depot)

def main_Gurobi(name_file):
        # this will try to fix the random number when seeded

    # Для 20 вершин объем грузовика 500, для 50 вершин объем грузовика 750, для 50 вершин объем грузовика 1000
#     TW_CAPACITIES = {
#         20: 30.,
#         50: 40.,
#         100: 50.,
#         500: 70.
#     }
    # Читаем данные
    xlsx = pd.ExcelFile(name_file)
    #sheet1 = pd.read_excel(xlsx, 'Sheet20')
    #sheet2 = pd.read_excel(xlsx, 'Sheet50')
    sheet3 = pd.read_excel(xlsx, 'Sheet100')
    sheet_vehicle = pd.read_excel(xlsx, 'Sheet_vehicle')

    # Описываем данные
    #nodes = sheet1.values # множество координат и количество товара в каждой вершине, который нужно довезти
    #nodes = sheet2.values
    nodes = sheet3.values
    depot = nodes[len(nodes) - 1] # координаты и количество товара в депо
    cities = nodes[0:(len(nodes) - 1)] # координаты и количество товара в остальных вершин(городов)
    vehicle = sheet_vehicle.values # первая колонка отвечает за номер траспортного средства
                            # вторая колонка это максимальная вместимость

    n_customer = len(nodes)
    xc = np.zeros(n_customer)
    yc = np.zeros(n_customer)
    for i in range(0, len(cities)):
        xc[i] = cities[i][0]
        yc[i] = cities[i][1]
    xc[len(cities)] = depot[0]
    yc[len(cities)] = depot[1]

    capacity = np.zeros(n_customer)
    for i in range(0, len(cities)):
        capacity[i] = cities[i][2]
    
    N = [i for i in range(1, n_customer)] # tour без депо
    V = [0] + N # tour с депо
    A = [(i,j) for i in V for j in V if i != j] # possible arcs
    c = {(i,j): np.hypot(xc[i]-xc[j], yc[i]-yc[j]) for (i,j) in A} # расстояния между соседними городами
    
    q = {i: capacity[i] for i in N} # number of VRUs at location i to be picked
    Q = vehicle[1][1]
    n = 1
    for i in range(len(capacity)):
        if(capacity[i] > Q):
            Q = Q / n
            n += 1
            Q *= n
    print("Используется грузовиков: ", n)
    time_limit = 5
    lst = []
    lst_time = []
#-----------------------------------------------------------------------------------------------------------------#        
    model = Model('CVRP')

    # Declaration of variables
    x = model.addVars(A, vtype= GRB.BINARY)
    y = model.addVars(N, vtype= GRB.CONTINUOUS)
    # setting the objective function
    model.modelSense = GRB.MINIMIZE
    model.setObjective(quicksum(x[i, j]*c[i, j] for i, j in A))

    # Adding constraints
    model.addConstrs(quicksum(x[i,j] for j in V if j!=i) == 1 for i in N)
    model.addConstrs(quicksum(x[i,j] for i in V if i!=j) == 1 for j in N)
    model.addConstrs((x[i,j] == 1) >> (y[i] + q[j] == y[j]) for (i,j) in A if i != 0 and j != 0)
    model.addConstrs(y[i] >= q[i] for i in N)
    model.addConstrs(y[k] <= Q for k in N)

    # Optimizing the model
#         model.Params.MIPGap = 0.1
    model.Params.TimeLimit = time_limit  # seconds
    model.Params.LogFile= "result.txt"
    model.optimize()
    if model.status == GRB.OPTIMAL:
        print('1.Optimal objective: %g' % model.objVal)
        print('Optimal cost: %g' % model.objVal)
        lst.append(model.objVal)
        lst_time.append(model.Runtime)
    elif model.status == GRB.INF_OR_UNBD:
        print('2.Model is infeasible or unbounded')
        res = -1
    elif model.status == GRB.INFEASIBLE:
        print('3.Model is infeasible')
        res = -1
    elif model.status == GRB.UNBOUNDED:
        print('4.Model is unbounded')
        res = -1
    else:
        print('5.Optimization ended with status %d' % model.status)
        print('Optimal cost: %g' % model.objVal)
        lst.append(model.objVal)
        lst_time.append(time_limit)
            
if __name__ == "__main__":
    while True:
        txt = input("Введите название одного из методов: SA или B&C, который вы хотите использовать для поиска решения:")
        if((txt == "SA") or (txt == "B&C")):
            file = input("Введите название файла вместе с расширением:")
            if(os.path.isfile(file)):
                #file = open(file, 'r')
                if(txt == "SA"):
                    main_SA(file)

                if(txt == "B&C"):
                    main_Gurobi(file)
                    
                break
            else:
                print('Файл не найден. Попробуйте еще раз!')
        else:
            print('Некорректно введено название метода. Попробуйте еще раз!')