from gurobipy import *
from gurobipy import Model, GRB, quicksum
import pandas as pd
from geopy.distance import geodesic
import random
from random import randint
import numpy as np
from functools import reduce
import operator
import os.path
import time
import copy

def distance(x1, y1, x2, y2):
    return geodesic((x1, y1), (x2, y2)).km

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

def subtourslice(tour, vehicle, max_vehicle):
    #print("--------------------------------Начало работы функции--------------------------------")
    capacity_used = np.zeros(len(vehicle))
    k = 0
    slice = []
    for i in range(len(vehicle)):
        # capacity_used[i] - начальный вес
        while((capacity_used[i] <= max_vehicle) and (k <= (len(tour) - 1))):
            capacity_used[i] += vehicle[k][1] # заполняем грузовик конкретным кол-вом груза для каждого города
            if(capacity_used[i] > max_vehicle):
                # если кол-во груза, который нужно отвезти, больше чем вместимость грузовика, то вычитаем это кол-во груза
                capacity_used[i] -= vehicle[k][1]
                k -= 1
                slice.append(vehicle[k][0]) # запоминаем номер тура, чтобы повторно не проходить по такому же маршруту
                k += 1
                break
            k += 1
    slice.append(vehicle[k-1][0])
    return(slice)

def subtour(slice, tour):
    sub = []
    # добавляем номера городов, после которых суммарный вес груза превышал объем грузовика
    sub.append(tour[:(slice[0] + 1)])
    for i in range(0, len(slice) - 1):
        # нарежем маршруты городов до момента, пока суммарный вес груза не превысел объем грузовика
        sub.append(tour[(slice[i] + 1):(slice[i + 1] + 1)])
    return sub

# Вычисление общей длины всех туров
def all_vechile_distance(cities, sub, depot):
    all_distance = reduce(operator.add, (totaldistancetour(cities, x, depot) for x in sub), 0)
    return all_distance

# Функция определения общего расстояния по данным туров и грузовиков
def tour_to_distance(cities, tour, vehicle, depot, max_vehicle):
    u = subtourslice(tour, vehicle, max_vehicle) # получаем спислок городов, после которых суммарный вес груза > объема грузовика
    v = subtour(u, tour) # получим тур из номеров городов, где u это список из номеров городов, после которых суммарный вес груза > объема грузовика
    total = all_vechile_distance(cities, v, depot) # посчитаем суммарное расстояние между всеми городами в туре
    return total

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

def SA(cities, vehicle, depot, max_vehicle):
    T_end = 1
    t_max = 100000

    current_solution = Initialize(len(cities))
    print("Начальный маршрут: ", current_solution)
    print(" ")
    currentEnergy = tour_to_distance(cities, current_solution, vehicle, depot, max_vehicle) # вычисляем энергию для первого состояния
    best_tour     = np.copy(current_solution)
    best_Energy   = worst_Energy = currentEnergy
    T = t_max
    k = 0
    while(T > T_end):  # на всякий случай ограничеваем количество итераций
    # может быть полезно при тестировании сложных функций изменения температуры T       
        new_solution    = GenerateStateCandidate(current_solution) # получаем новое решение
        candidateEnergy = tour_to_distance(cities, new_solution, vehicle, depot, max_vehicle) # вычисляем его энергию
        best_Energy     = min(best_Energy, candidateEnergy)
        worst_Energy    = max(worst_Energy, candidateEnergy)

        if(candidateEnergy < currentEnergy): # если кандидат обладает меньшей энергией
            currentEnergy    = candidateEnergy  # то оно становится текущим состоянием
            current_solution = np.copy(new_solution)
            if(currentEnergy < best_Energy):
                best_tour    = np.copy(current_solution)
        else:
            p = np.exp((currentEnergy - candidateEnergy) / T) # иначе, считаем вероятность
            if (p > np.random.uniform()): # и смотрим, осуществится ли переход
                currentEnergy    = candidateEnergy
                best_tour        = np. copy(current_solution)
                current_solution = np.copy(new_solution)
                
        T = t_max / (k + 1) # уменьшаем температуру
        k += 1
    print(" ")
    print("Худшее расстояние: ", worst_Energy)
    print("Лучшее расстояние: ", best_Energy)
    print("Лучший маршрут: ", best_tour)
    print(" ")

    best_slice = subtourslice(best_tour, vehicle, max_vehicle)
    best_subtour = subtour(best_slice, best_tour)
    print("Лучшее распределение объема груза по городам:", best_slice)
    print("Лучшее разделение городов по отдельным маршрутам: ", best_subtour)

def main_SA(name_file):
    sheet  = pd.read_csv(name_file, sep="\t")
    nodes  = sheet.values
    depot  = nodes[len(nodes) - 1]
    cities = nodes[:(len(nodes) - 1)]
    max_vehicle = 50
    n = len(cities) + 1
    q = []
    q.append((0, depot[2]))
    for i in range(n-1):
        q.append((i+1, cities[i][2])) # вес груза, который необходимо перевести в город
        
    p = 1
    max_vehicle *= p
    while(max_vehicle < max(q)[1]):
        max_vehicle = max_vehicle / p
        p += 1
        max_vehicle *= p
    print("Используется грузовиков: ", p)

    return SA(cities, q, depot, max_vehicle)

def main_Gurobi(name_file):
    # Читаем данные
    sheet  = pd.read_csv(name_file, sep="\t")
    nodes  = sheet.values
    depot  = nodes[len(nodes) - 1]
    cities = nodes[:(len(nodes) - 1)]

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
    c = {}
    for i,j in A:
        node_1 = (xc[i], yc[i])
        node_2 = (xc[j], yc[j])
        c[(i,j)] = geodesic(node_1, node_2).km
    q = {i: capacity[i] for i in N} # number of VRUs at location i to be picked
    Q = 50
    n = 1
    for i in range(len(capacity)):
        if(capacity[i] > Q):
            Q = Q / n
            n += 1
            Q *= n
    print("Используется грузовиков: ", n)
    time_limit = 100
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
        
def subtour_LKH(slice, tour):
    sub = []
    remember = 0
    # добавляем номера городов, после которых суммарный вес груза превышал объем грузовика
    p = tour[:(slice[0] + 1)]
    p.append(tour[0])
    if(0 not in p):
        h = randint(1, len(p)-1)
        p.insert(h, 0)
    else:
        remember = 0
    sub.append(p)
    for i in range(0, len(slice) - 1):
        # нарежем маршруты городов до момента, пока суммарный вес груза не превысел объем грузовика
        k = tour[(slice[i] + 1):(slice[i + 1] + 1)] 
        k.append(tour[(slice[i] + 1)])
        if(0 not in k):
            h = randint(1, len(k)-1)
            k.insert(h, 0)
        else:
            remember = i
        sub.append(k)
    return sub, remember

def make_valid_tour(tour, tour_edges, X, Y, distance, num_cities):
    if len(Y) - len(X) != 0:
        return [], False
    tour_edges_new = copy.deepcopy(tour_edges)
    for i in range(len(X)):
        Xii = list(X[i])
        tour_edges_new[Xii[0]].remove(Xii[1])
        tour_edges_new[Xii[1]].remove(Xii[0])
        Yii = list(Y[i])
        tour_edges_new[Yii[0]].add(Yii[1])
        tour_edges_new[Yii[1]].add(Yii[0])
    for i in tour_edges_new:
        if len(tour_edges_new[i]) != 2:
            return [], False
    new_tour = ['tour',0]
    i = list(tour_edges_new[0])[0]
    new_tour += [i]
    for foo in range(num_cities-1):
        node_connections = list(tour_edges_new[i]) # всевозможные ребра
        for j in [0,1]:
            if node_connections[j] != new_tour[-2]:
                new_tour += [node_connections[j]]
                if node_connections[j] == new_tour[1]:
                    new_tour.pop(0)
                    if len(new_tour) == num_cities+1:
                        return [new_tour, tour_edges_new], True
                    else:
                        return [], False
                i = node_connections[j]
                break
    print('Error')

    
def find_edge_to_remove(tour, tour_edges, delta, broken_edges, created_edges, initial_node, latest_node, distance, num_cities):
    if(latest_node in tour):
        j = tour.index(latest_node)
    else:
        return False, [], {}
    for t in neighbours(j, tour, num_cities):
        xi = {latest_node, t}
        if xi not in created_edges and xi not in broken_edges and t != initial_node:
            Xi = broken_edges[:]
            Xi += [xi]
            k = list(xi)
            reconnecting_edge = {t, initial_node}
            tempY = created_edges[:]
            tempY += [reconnecting_edge]
            new_tour_object, result = make_valid_tour(tour, tour_edges, Xi, tempY, distance, num_cities)
            if result == True:
                temp_delta = delta - distance[(k[0], k[1])]
                if temp_delta + distance[(t, initial_node)] < 0:
                    return True, new_tour_object[0], new_tour_object[1]
                else:
                    return find_edge_to_add(tour, tour_edges, temp_delta, Xi, created_edges, initial_node, t, distance, num_cities)
    return False, [], {}

def find_edge_to_add(tour, tour_edges, delta, broken_edges, created_edges, initial_node, latest_node, distance, num_cities):
    for t in range(num_cities):
        if t not in neighbours(latest_node, tour, num_cities) + [latest_node]:
            yi = {latest_node, t}
            temp_delta = delta + distance[(latest_node, t)]
            if yi not in broken_edges and temp_delta < 0:
                tempY = created_edges[:]
                tempY += [yi]
                return find_edge_to_remove(tour, tour_edges, temp_delta, broken_edges, tempY, initial_node, t, distance, num_cities)
    return False, [], {}

def neighbours(i, tour, num_cities):
    if i > 0 and i < num_cities:
        return [tour[i-1], tour[i+1]]
    else:
        return [tour[1], tour[-2]]

def apply_LK(tour, tour_edges, start, distance, num_cities, old_tour_length):
    min_tour_length = old_tour_length * 10
    min_tour = False
    for i in range(num_cities):
        if time.time() - start > 100:
            if min_tour:
                return False, min_tour, min_tour_edges
            else:
                return False, tour, tour_edges
        t1 = tour[i]
        for t2 in neighbours(i, tour, num_cities):
            x1 = {t1, t2}
            X  = [x1]
            for j in tour:
                if j not in x1:
                    t3 = j
                    y1 = {t2, t3}
                    delta1 = distance[(t2, t3)] - distance[(t1, t2)]
                    Y  = [y1]
                    if delta1 < 0:
                        result = find_edge_to_remove(tour, tour_edges, delta1, X, Y, t1, t3, distance, num_cities)
                        if result[0]:
                            temp_tour_length = 0
                            temp_tour = result[1]
                            for i in range(0,num_cities-1):
                                temp_tour_length = temp_tour_length + distance[(temp_tour[i], temp_tour[i+1])]
                            temp_tour_length = temp_tour_length + distance[(temp_tour[num_cities-1], temp_tour[0])]
                            if temp_tour_length < min_tour_length:
                                min_tour_length = temp_tour_length
                                min_tour = temp_tour
                                min_tour_edges  = result[2]
    if min_tour:
        return True, min_tour, min_tour_edges
    return False, tour, {}

def main_LKH(name_file):
    sheet       = pd.read_csv(name_file, sep="\t")
    nodes       = sheet.values
    depot       = nodes[len(nodes) - 1]
    cities_data = nodes[:(len(nodes) - 1)]
    
    #set up for the loop calling LK
    n = len(cities_data) + 1 # +1, так как учитываем депо

    roads = [(i, j) for i in range(n) for j in range(n) if(i != j)] # это попарные маршруты между всеми городами

    x = []
    y = []
    x.append(depot[0])
    y.append(depot[1])

    for i in range(n-1):
        x.append(cities_data[i][0])
        y.append(cities_data[i][1])
        
    distance = {}
    for i,j in roads:
        node_1 = (x[i], y[i])
        node_2 = (x[j], y[j])
        distance[(i,j)] = geodesic(node_1, node_2).km

    tour = list(range(n))
    random.shuffle(tour)

    max_vehicle = 50
    q = []
    q.append((0, depot[2]))
    for i in range(n-1):
        q.append((i+1, cities_data[i][2])) # вес груза, который необходимо перевести в город
        
    p = 1
    max_vehicle *= p
    while(max_vehicle < max(q)[1]):
        max_vehicle  = max_vehicle / p
        p += 1
        max_vehicle *= p
    print("Используется грузовиков: ", p)

    lst_subtour = subtourslice(tour, q, max_vehicle)
    all_subtour, rem_idx = subtour_LKH(lst_subtour, tour)
    print("all_subtour", all_subtour)

    dist_subtour = []
    s = 0
    for p in range(0, len(all_subtour)):
        if(len(all_subtour[p]) > 2):
            for i in range(len(all_subtour[p])-1):
                s += distance[(all_subtour[p][i], all_subtour[p][i+1])]
        dist_subtour.append(s)
        s = 0
    print("Old distance", sum(dist_subtour))

    all_tour = []
    for p in range(0, len(all_subtour)):
        success = True
        start=time.time()
        tour_edges = {}
        
        for i in range(1, len(all_subtour[p])-1):
            tour_edges[all_subtour[p][i]] = {all_subtour[p][i+1], all_subtour[p][i-1]}
        tour_edges[all_subtour[p][0]] = {all_subtour[p][1], all_subtour[p][-2]}

        if(len(all_subtour[p]) > 3):
            while success and time.time() - start < 100:
                success, all_subtour[p], tour_edges = apply_LK(all_subtour[p], tour_edges, start, distance, len(all_subtour[p])-1, dist_subtour[p])
        print("Tour", all_subtour[p])
        all_tour.append(all_subtour[p])

    s = 0
    for p in range(0, len(all_tour)):
        if(rem_idx != p):
            all_tour[p].remove(0)
        for i in range(len(all_tour[p])-1):
            if(len(all_tour[p]) > 3):
                s += distance[(all_tour[p][i], all_tour[p][i+1])]
        all_subtour[p].pop(-1)
    print("New distance", s)

    print(all_tour)

if __name__ == "__main__":
    while True:
        txt = input("Введите название одного из методов: SA или B&C или LKH, который вы хотите использовать для поиска решения:")
        if((txt == "SA") or (txt == "B&C") or (txt == "LKH")):
            file = input("Введите название файла вместе с расширением:")
            if(os.path.isfile(file)):
                if(txt   == "SA"):
                    main_SA(file)
                elif(txt == "B&C"):
                    main_Gurobi(file)
                elif(txt == "LKH"):
                    main_LKH(file)
                break
            else:
                print('Файл не найден. Попробуйте еще раз!')
        else:
            print('Некорректно введено название метода. Попробуйте еще раз!')

