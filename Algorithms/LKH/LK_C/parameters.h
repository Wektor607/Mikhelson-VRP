// for towns
#define countTowns 51

// for cars
//#define maxCapacity 30
#define maxCountCar 3
#define optimalRadius 0.37
#define countFiles 19

#define FILES \
	char *mfiles[] = { \
		"50(20 задач)/20201007_185322.csv", \
		"50(20 задач)/20201023_203450.csv", \
		"50(20 задач)/20201026_211708.csv", \
		"50(20 задач)/20201027_214904.csv", \
		"50(20 задач)/20201028_202211.csv", \
		"50(20 задач)/20201029_201947.csv", \
		"50(20 задач)/20201105_201910.csv", \
		"50(20 задач)/20201109_190333.csv", \
		"50(20 задач)/20201110_211510.csv", \
		"50(20 задач)/20201110_212146.csv", \
		"50(20 задач)/20201117_214239.csv", \
		"50(20 задач)/20201122_141441.csv", \
		"50(20 задач)/20201127_184250.csv", \
		"50(20 задач)/20201128_220622.csv", \
		"50(20 задач)/20201203_215226.csv", \
		"50(20 задач)/20201204_205445.csv", \
		"50(20 задач)/20201209_152236.csv", \
		"50(20 задач)/20201214_185310.csv", \
		"50(20 задач)/20200925_191607.csv", \
		"50(20 задач)/20200927_161843.csv" \
	}


#define STARTTOWNS \
	town towns[] = { \
		maketown(0, 0.77, 0.24, 0), \
		maketown(1, 0.1, 0.2, 30), \
		maketown(2, 0.15, 0.35, 23), \
		maketown(3, 0.43, 0.27, 10), \
		maketown(4, 0.9, 0.13, 2), \
		maketown(5, 0.81, 0.55, 73), \
		maketown(6, 0.28, 0.61, 48), \
		maketown(7, 0.37, 0.22, 51), \
		maketown(8, 0.21, 0.69, 7), \
		maketown(9, 0.7 , 0.9 , 24), \
		maketown(10,0.13, 0.91, 62), \
		maketown(11,0.89, 0.01, 27), \
		maketown(12,0.32, 0.51, 87), \
		maketown(13,0.48, 0.47, 41) \
	}

// a[0] == 'l' && a[1] == 'j'

// for lkh
#define ALGFOR(i) i < countUpdate
//(timeUpdate < ((clock() - runtime) / CLOCKS_PER_SEC))
// for lkh
#define countTasks 100000
#define countUpdate 7500
#define ALGFOR(i) i < countUpdate
//#define timeUpdate 1500000
#define LKH lkh3opt
#define fileout "lkh_50.txt"

//for sa
#define tmax 100000
#define tmin 20