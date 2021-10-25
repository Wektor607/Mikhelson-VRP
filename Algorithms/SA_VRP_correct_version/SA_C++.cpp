#include <iostream>
#include <math.h>
#include <fstream>
#include <random>
#include <stdlib.h>
#include <stdio.h>
#include <ctime>
using namespace std;

int n_samples = 10;  //Кол-во задач
const int n_customer = 50; //Кол-во городов
int max_vehicle_car = 40; //Макс. вместимоть грузовика

int minCapacity = 1;
int maxCapacity = 42;

int T_end = 1; //Конечная t
int t_max = 100000; //Начальная t

void printd(double *c, int max_size)
{
	printf("[");
	for(int i = 0; i < max_size; i++)
	{
		printf("%lf ", (double)c[i]);
	}
	printf("]\n");
}

void printi(int *c, int max_size)
{
	printf("[");
	for(int i = 0; i < max_size; i++)
	{
		printf("%d ", (int)c[i]);
	}
	printf("]\n");
}

void subtourslice(int u[], double current_solution[], double Capacity[], double max_vehicle[]) 
{
	double capacity_used[n_customer];
	for(int i = 0; i < n_customer; i++) 
	{
		capacity_used[i] = 0;
	}
	int k = 0;
	int i;
	int p = 0;
	for(i = 0; i < n_customer; i++) 
	{
		while((capacity_used[i] <= max_vehicle[i]) && (k <= (n_customer - 1))) 
		{
			capacity_used[i] += Capacity[(int)(current_solution[k])];
			if(capacity_used[i] > max_vehicle[i]) 
			{
				capacity_used[i] -= Capacity[(int)(current_solution[k])];
				
				u[p] = k - 1;
				p++;
				break;
			}
			k++;
		}
	}
	u[p] = k - 1;
	u[p + 1] = -1;
}


double distance(double x1, double y1, double x2, double y2){
    double res = sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
    return res;
}

double totaldistancetour(int sub[], double coords[][2], double depot[2])
{
    double d = 0;
    double x1, x2, y1, y2;
    int k;
    for(k = 0; sub[k] != -1; k++);
    for(int i = 1; i < k; i++)
	{
        x1 = coords[(int)(sub[i - 1])][0];
        y1 = coords[(int)(sub[i - 1])][1];
        x2 = coords[(int)(sub[i])][0];
        y2 = coords[(int)(sub[i])][1];
        d = d + distance(x1, y1, x2, y2);
    }


    x1 = coords[(int)(sub[k - 1])][0];
    y1 = coords[(int)(sub[k - 1])][1];
    x2 = depot[0];
    y2 = depot[1];
    d = d + distance(x1, y1, x2, y2);

    x1 = coords[(int)(sub[0])][0];
    y1 = coords[(int)(sub[0])][1];
    x2 = depot[0];
    y2 = depot[1];
    d = d + distance(x1, y1, x2, y2);

    return d;
}

double all_vechicle_distance(int sub[][n_customer], double coords[][2], double depot[2]) 
{
	double all_distance = 0;
	for(int i = 0; sub[i][0] != -1; i++) {
		all_distance += totaldistancetour(sub[i], coords, depot);
	}
	return all_distance;
}

double tour_to_distance(double current_solution[], double coords[][2], double Capacity[], double max_vehicle[], double depot[2]) 
{
	int u[n_customer];
	subtourslice(u, current_solution, Capacity, max_vehicle);
	int p = 0;
	int sub[n_customer][n_customer];
	
	int i;
	for(i = 0; i < (u[0] + 1); i++)
	{
		sub[p][i] = current_solution[i];
	}
	sub[p][i] = -1;
	p++;
	for(i = 0; u[i+1] != -1; i++) 
	{
		int j;
		for(j = u[i] + 1; j < u[i + 1] + 1; j++) 
		{
			sub[p][j - (u[i] + 1)] = current_solution[j];
		}
		sub[p][j - (u[i] + 1)] = -1;
		p++;
	}
	sub[p][0] = -1;

	double all_distance = 0;
	for(int i = 0; sub[i][0] != -1; i++) 
	{
		all_distance += totaldistancetour(sub[i], coords, depot);
	}
	return all_distance;
}

void GenerateStateCandidate(double r[], double current_solution[]) 
{
	int indexA = rand() % n_customer;
	int indexB = rand() % n_customer;
	while(indexA == indexB){
		indexB = rand() % n_customer;
	}

	for(int i = 0; i < min(indexA,indexB); i++) 
	{
		r[i] = current_solution[i];
	}

	for(int i = 0; i < max(indexA, indexB) - min(indexA, indexB) + 1; i++) 
	{ 
		r[min(indexA, indexB) + i] = current_solution[max(indexA, indexB) - i]; 
	}

	for(int i = max(indexA, indexB) + 1; i < n_customer; i++) 
	{
		r[i] = current_solution[i];
	}

}

double SA(double coords[][2], double Capacity[], double max_vehicle[], int T_end, int t_max, double depot[2])
{
	double current_solution[n_customer];
	// Список городов от 0 до n_customer
	for(int i = 0; i < n_customer; i++) 
	{ 
		current_solution[i] = i;
	}
	// Перемешиваем список городов, случайным образом
	for(int i = 0; i < n_customer; i++) 
	{
    	swap(current_solution[i], current_solution[rand() % n_customer]);
    }
    
    double currentEnergy = tour_to_distance(current_solution, coords, Capacity, max_vehicle, depot);
	double t = t_max;

	double r[n_customer];
	double bestEnergy = currentEnergy;
	double worstEnergy = currentEnergy;
	double candidateEnergy = 0;
	double p;
	int k = 1;

	while(t > T_end)
	{
		GenerateStateCandidate(r, current_solution);
		candidateEnergy = tour_to_distance(r, coords, Capacity, max_vehicle, depot);
		
		bestEnergy = min(bestEnergy, candidateEnergy);

		if(candidateEnergy < currentEnergy) 
		{
			currentEnergy = candidateEnergy;
			for(int i = 0; i < n_customer; i++) 
			{
				current_solution[i] = r[i];
			}
		} 
		else 
		{
			p = exp((currentEnergy - candidateEnergy) / t);
			if(p > (rand() % 10000 / 10000.0)) 
			{
				currentEnergy = candidateEnergy;
				for(int i = 0; i < n_customer; i++) 
				{
					current_solution[i] = r[i];
				}
			}
		}
		t = t_max * 0.1 / k;
		k++;
	}
	return bestEnergy;
}

int main(int argc, char** argv)
{
	srand((unsigned int)time(0));

	unsigned int start_time =  clock();

	double result[n_samples];

	double Capacity[n_customer];

	for(int i = 0; i < n_customer; i++) 
	{
		Capacity[i] = (double)(rand() % 41 + 1) + (rand() % 1000000 / 1000000.0);
	}

	int n = 2;
	max_vehicle_car = max_vehicle_car * n;
	while(max_vehicle_car < maxCapacity) 
	{
	    max_vehicle_car = max_vehicle_car / n;
	    n += 1;
	    max_vehicle_car *= n;
	}
	printf("Используется грузовичков: %d\n", n);

	double max_vehicle[n_customer];
	for(int i = 0; i < n_customer; i++)
	{
		max_vehicle[i] = max_vehicle_car;
	}

	int i = 0;
	double coords[n_customer][2]; 
	double depot[2];

	while(i < n_samples)
	{
		depot[0] = rand() % 10000 / 10000.0;
		depot[1] = rand() % 10000 / 10000.0;

	    for(int j = 0; j < n_customer; j++) 
		{
	    	coords[j][0] = rand() % 10000 / 10000.0;
	    	coords[j][1] = rand() % 10000 / 10000.0;
	    }
	    result[i] = SA(coords, Capacity, max_vehicle, T_end, t_max, depot);
	    i++;
	}
	printd(result, n_samples);
	double s = 0;
	for(int i = 0; i < n_samples; i++)
	{
		s += result[i];
	}
	printf("sum/n: %lf\n", s/n_samples);
	unsigned int end_time = clock();
	cout << "runtime = " << (end_time - start_time) / 1000000.0 << endl;
	return 0;
}






