
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <ctime>
#include <curand_kernel.h>

/***** default values of parameters ******************************************/
#define	TIMELIM	60	/* the time limit for the algorithm in seconds */
#define	GIVESOL	0	/* 1: input a solution; 0: do not give a solution */

#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )

typedef struct {
	int		timelim;	/* the time limit for the algorithm in secs. */
	int		givesol;	/* give a solution (1) or not (0) */
	/* Never modify the above two lines.  */
	/* You can add more components below. */
} Param;			/* parameters */

typedef struct {
	int	n;	/* number of jobs */
	int	m;	/* number of agents */
	int	**c;	/* cost matrix c_{ij} */
	int	**a;	/* resource requirement matrix a_{ij} */
	int	*b;	/* available amount b_i of resource for each agent i */
} GAPdata;	/* data of the generalized assignment problem */

typedef struct {
	double	timebrid;	/* the time before reading the instance data */
	double	starttime;	/* the time the search started */
	double	endtime;	/* the time the search ended */
	int		*bestsol;	/* the best solution found so far */
	/* Never modify the above four lines. */
	/* You can add more components below. */
} Vdata;		/* various data often necessary during the search */

/*************************** functions ***************************************/
void copy_parameters(int argc, char *arcv[], Param *param);
void read_instance(GAPdata *h_gapdata);
void prepare_memory(Vdata *vdata, GAPdata *h_gapdata);
void free_memory(Vdata *vdata, GAPdata *h_gapdata);
void read_sol(Vdata *vdata, GAPdata *h_gapdata);
void recompute_cost(Vdata *vdata, GAPdata *h_gapdata);
void *malloc_e(size_t size);

void greedy_init(int *sol, GAPdata *h_gapdata);
int calculate_cost(int *sol, GAPdata *h_gapdata);
bool is_feasible(int *rest_b, GAPdata *h_gapdata);

/***** check the feasibility and recompute the cost **************************/
/***** NEVER MODIFY THIS SUBROUTINE! *****************************************/
void recompute_cost(Vdata *vdata, GAPdata *h_gapdata)
{
	int	i, j;		/* indices of agents and jobs */
	int	*rest_b;	/* the amount of resource available at each agent */
	int	cost, penal;	/* the cost; the penalty = the total capacity excess */
	int	temp;		/* temporary variable */

	rest_b = (int *)malloc_e(h_gapdata->m * sizeof(int));
	cost = penal = 0;
	for (i = 0; i < h_gapdata->m; i++) { rest_b[i] = h_gapdata->b[i]; }
	for (j = 0; j < h_gapdata->n; j++) {
		// 计算剩余资源
		rest_b[vdata->bestsol[j]] -= h_gapdata->a[vdata->bestsol[j]][j];
		// 计算开销
		cost += h_gapdata->c[vdata->bestsol[j]][j];
	}
	for (i = 0; i < h_gapdata->m; i++) {
		// 计算资源是否够用，如果资源不够用，则temp<0，penal大于0
		temp = rest_b[i];
		if (temp < 0) { penal -= temp; }
	}
	printf("recomputed cost = %d\n", cost);
	// pental大于0，说明资源不够用
	if (penal > 0) {
		printf("INFEASIBLE!!\n");
		printf(" resource left:");
		for (i = 0; i < h_gapdata->m; i++) { printf(" %3d", rest_b[i]); }
		printf("\n");
	}
	printf("time for the search:       %7.2f seconds\n",
		(vdata->endtime - vdata->starttime) / CLOCKS_PER_SEC);
	printf("time to read the instance: %7.2f seconds\n",
		(vdata->starttime - vdata->timebrid) / CLOCKS_PER_SEC);

	free((void *)rest_b);
}

/***** read a solution from STDIN ********************************************/
void read_sol(Vdata *vdata, GAPdata *h_gapdata)
{
	int	j;		/* index of jobs */
	int	value_read;	/* the value read by fscanf */
	FILE	*fp = stdin;	/* set fp to the standard input */

	for (j = 0; j < h_gapdata->n; j++) {
		fscanf(fp, "%d", &value_read);
		/* change the range of agents from [1, m] to [0, m-1] */
		vdata->bestsol[j] = value_read - 1;
	}
}

/***** prepare memory space **************************************************/
/***** Feel free to modify this subroutine. **********************************/
void prepare_memory(Vdata *vdata, GAPdata *h_gapdata)
{
	int j;

	vdata->bestsol = (int *)malloc_e(h_gapdata->n * sizeof(int));
	/* the next line is just to avoid confusion */
	for (j = 0; j < h_gapdata->n; j++) { vdata->bestsol[j] = 0; }
}

/***** free memory space *****************************************************/
/***** Feel free to modify this subroutine. **********************************/
void free_memory(Vdata *vdata, GAPdata *h_gapdata)
{
	free((void *)vdata->bestsol);
	free((void *)h_gapdata->c[0]);
	free((void *)h_gapdata->c);
	free((void *)h_gapdata->a[0]);
	free((void *)h_gapdata->a);
	free((void *)h_gapdata->b);
}

/***** read the instance data ************************************************/
/***** NEVER MODIFY THIS SUBROUTINE! *****************************************/
void read_instance(GAPdata *h_gapdata)
{
	int	i, j;		/* indices of agents and jobs */
	int	value_read;	/* the value read by fscanf */
	// FILE	*fp = stdin;	/* set fp to the standard input */

	FILE *fp = fopen("data/c40400", "r");

	/* read the number of agents and jobs */
	fscanf(fp, "%d", &value_read);	/* number of agents */
	h_gapdata->m = value_read;
	fscanf(fp, "%d", &value_read);		/* number of jobs */
	h_gapdata->n = value_read;

	/* initialize memory */
	h_gapdata->c = (int **)malloc_e(h_gapdata->m * sizeof(int *));
	h_gapdata->c[0] = (int *)malloc_e(h_gapdata->m * h_gapdata->n * sizeof(int));
	for (i = 1; i < h_gapdata->m; i++) { h_gapdata->c[i] = h_gapdata->c[i - 1] + h_gapdata->n; }
	h_gapdata->a = (int **)malloc_e(h_gapdata->m * sizeof(int *));
	h_gapdata->a[0] = (int *)malloc_e(h_gapdata->m * h_gapdata->n * sizeof(int));
	for (i = 1; i < h_gapdata->m; i++) { h_gapdata->a[i] = h_gapdata->a[i - 1] + h_gapdata->n; }
	h_gapdata->b = (int *)malloc_e(h_gapdata->m * sizeof(int));

	/* read the cost coefficients */
	for (i = 0; i < h_gapdata->m; i++) {
		for (j = 0; j < h_gapdata->n; j++) {
			fscanf(fp, "%d", &value_read);
			h_gapdata->c[i][j] = value_read;
		}
	}

	/* read the resource consumption */
	for (i = 0; i < h_gapdata->m; i++) {
		for (j = 0; j < h_gapdata->n; j++) {
			fscanf(fp, "%d", &value_read);
			h_gapdata->a[i][j] = value_read;
		}
	}

	/* read the resource capacity */
	for (i = 0; i < h_gapdata->m; i++) {
		fscanf(fp, "%d", &value_read);
		h_gapdata->b[i] = value_read;
	}
}

/***** copy and read the parameters ******************************************/
/***** Feel free to modify this subroutine. **********************************/
void copy_parameters(int argc, char *argv[], Param *param)
{
	int i;

	/**** copy the parameters ****/
	param->timelim = TIMELIM;
	param->givesol = GIVESOL;
	/**** read the parameters ****/
	if (argc > 0 && (argc % 2) == 0) {
		printf("USAGE: ./gap [param_name, param_value] [name, value]...\n");
		exit(EXIT_FAILURE);
	}
	else {
		for (i = 1; i < argc; i += 2) {
			if (strcmp(argv[i], "timelim") == 0) param->timelim = atoi(argv[i + 1]);
			if (strcmp(argv[i], "givesol") == 0) param->givesol = atoi(argv[i + 1]);
		}
	}
}

/***** malloc with error check ***********************************************/
void *malloc_e(size_t size) {
	void *s;
	if ((s = malloc(size)) == NULL) {
		fprintf(stderr, "malloc : Not enough memory.\n");
		exit(EXIT_FAILURE);
	}
	return s;
}

/***** subroutines ***********************************************/
void greedy_init(int *sol, GAPdata *h_gapdata) {
	float sum, rnd;
	int *vals = (int *)malloc_e(h_gapdata->m * sizeof(int));
	int *rest_b = (int *)malloc_e(h_gapdata->m * sizeof(int));
	for (int i = 0; i < h_gapdata->m; i++) rest_b[i] = h_gapdata->b[i];

	for (int j = 0; j < h_gapdata->n; j++) {
		sum = 0;
		for (int i = 0; i < h_gapdata->m; i++) {
			// 计算vals和sum，
			// 如果出现资源已被用完的情况，则- min(0, rest_b[i])为正数，val变大，选择其的概率变小
			vals[i] = 3 * h_gapdata->c[i][j] + 2 * h_gapdata->a[i][j] - min(0, rest_b[i]);
			// 成本低且资源消耗小的，其val值小，则val的倒数就大
			sum += ((1.0 / vals[i]) * 2);
		}
		// rnd的值为0到sum，计算rnd落在哪个区间内，就将任务分配给该机器
		rnd = (float)rand() / (float)(RAND_MAX / sum);
		for (int i = 0; i < h_gapdata->m; i++) {
			rnd -= ((1.0 / vals[i]) * 2);
			// 减到rnd小于0就假设解，并break
			// 此处有bug，不能用rnd<0来判断，由于除法运算有误差，会导致rnd为一个很小的数，但大于0，于是导致sol[j]未初始化，于是会导致rest_b[sol[j]]内存访问错误
			if (rnd < 0) {
				sol[j] = i;
				break;
			}
		}
		// 防止sol[j]未初始化，导致rest_b[sol[j]]内存访问错误
		if (sol[j] < 0 || sol[j] >= h_gapdata->m) {
			sol[j] = h_gapdata->m - 1;
		}
		/*if (sol[j] < 0) {
			printf("%d, sol[j]:%d\n", j, sol[j]);
			printf("rdn:%10f\n", rnd);
		}*/
		// 减去资源
		rest_b[sol[j]] -= h_gapdata->a[sol[j]][j];
	}

	free((void *)vals);
	free((void *)rest_b);
}

// 计算开销
int calculate_cost(int *sol, GAPdata *h_gapdata) {
	int cost = 0;
	for (int i = 0; i < h_gapdata->n; i++) {
		cost += h_gapdata->c[sol[i]][i];
	}
	return cost;
}

// 计算解是否可行
bool is_feasible(int *rest_b, GAPdata *h_gapdata) {
	bool is_f = true;
	for (int i = 0; i < h_gapdata->m; i++) {
		if (rest_b[i] < 0) is_f = false;
	}
	return is_f;
}

// 生成随机数
__device__ int generateRand(curandState *globalState, int idx)
{
	curandState localState = globalState[idx];
	unsigned rand = abs(int(curand(&localState))); // 注意这里一定要加abs
	globalState[idx] = localState;
	return rand;
}

// 和生成随机数配合使用
__global__ void setup_kernal(curandState *state, long seed)
{
	// int id = blockDim.x * blockIdx.x + threadIdx.x;

	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int id = blockId * (blockDim.x * blockDim.y)
		+ (threadIdx.y * blockDim.x) + threadIdx.x;

	curand_init(seed, id, 0, &state[id]);
}

// sols[k][n]，为k个初始解，每一行为一个初始解，每行n列，代表有n个任务
// sols[i][j]，为第i个解，其第j个任务分配给值sols[i][j]表示的机器
__global__ void searchSol(int** sols, GAPdata* d_gapdata, int k, curandState* globalState, int** d_c, int** d_a, int* d_b) {

	// 计算线程号i，block的ID * block内线程个数 + 当前线程ID
	// int idx = blockDim.x * blockIdx.x + threadIdx.x;

	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int idx = blockId * (blockDim.x * blockDim.y)
		+ (threadIdx.y * blockDim.x) + threadIdx.x;

	// 对每一个初始解，gpu线程并行进行搜索
	if (idx < k) {
		// printf("idx:%d\n", idx);

		//for (int i = 0; i < d_gapdata->m; i++) {
		//	printf("b[i]:%d\n", d_b[i]);
		//}

		//for (int i = 0; i < k; i++) {
		//	for (int j = 0; j < d_gapdata->n; j++) {
		//		printf("sols[i][j]:%d\n", sols[i][j]);
		//	}
		//}
		//for (int i = 0; i < d_gapdata->m; i++) {
		//	for (int j = 0; j < d_gapdata->n; j++) {
		//		printf("d_c[i][j]:%d\n", d_c[i][j]);
		//	}
		//}
		//for (int i = 0; i < d_gapdata->m; i++) {
		//	for (int j = 0; j < d_gapdata->n; j++) {
		//		printf("d_a[i][j]:%d\n", d_a[i][j]);
		//	}
		//}
		//printf("d_gapdata->m:%d\n", d_gapdata->m);
		//printf("d_gapdata->n:%d\n", d_gapdata->n);

		// 剩余资源
		int *rest_b = new int[d_gapdata->m];

		const int INFEASIBLE_COST = 3;

		int swap, tmp;
		int impr = 0;
		//int impr_lim = d_gapdata-> n * 5;
		int impr_lim = 100;
		int pre_val, new_val;
		int swap_cost, cur_cost;

		// 可用资源
		for (int i = 0; i < d_gapdata->m; i++) {
			rest_b[i] = d_b[i];
		}

		// 计算剩余资源
		for (int i = 0; i < d_gapdata->n; i++) {
			rest_b[sols[idx][i]] -= d_a[sols[idx][i]][i];
		}
		/*printf("d_rest_b:\n");
		for (int j = 0; j < d_gapdata->m; j++) {
			printf("%d %d ", idx, rest_b[j]);
		}*/

		// 计算开销
		pre_val = 0;
		for (int i = 0; i < d_gapdata->n; i++) {
			pre_val += d_c[sols[idx][i]][i];
		}

		// 如果剩余资源小于0，则pre_val加上一个惩罚值
		for (int i = 0; i < d_gapdata->m; i++) {
			pre_val -= INFEASIBLE_COST * min(0, rest_b[i]);
		}
		new_val = pre_val;

		// 随机交换解的两个元素，尝试局部优化
		while (impr < impr_lim) {
			for (int j = 0; j < d_gapdata->n; j++) {
				// 计算一个随机值 swap
				unsigned rand = generateRand(globalState, idx);
				swap = rand % d_gapdata->n;
				// 交换两个数，并计算交换后的cost
				// 比如sol[3]=4, sol[5]=6，交换后为sol[3]=6,sol[5]=4
				swap_cost
					= d_c[sols[idx][j]][swap]
					+ d_c[sols[idx][swap]][j]
					+ INFEASIBLE_COST
					* (max(0, d_a[sols[idx][j]][swap] - rest_b[sols[idx][j]])
						+ max(0, d_a[sols[idx][swap]][j] - rest_b[sols[idx][swap]]));

				// 未交换前的cost
				cur_cost
					= d_c[sols[idx][j]][j]
					+ d_c[sols[idx][swap]][swap]
					+ INFEASIBLE_COST
					* (max(0, d_a[sols[idx][j]][j] - rest_b[sols[idx][j]])
						+ max(0, d_a[sols[idx][swap]][swap] - rest_b[sols[idx][swap]]));

				// 如果交换后的cost比交换前小，则更新最优解和剩余资源
				if (cur_cost > swap_cost) {
					tmp = sols[idx][j];

					rest_b[tmp] += (d_a[tmp][j] - d_a[tmp][swap]);
					rest_b[sols[idx][swap]] += (d_a[sols[idx][swap]][swap] - d_a[sols[idx][swap]][j]);

					sols[idx][j] = sols[idx][swap];
					sols[idx][swap] = tmp;
				}
			}

			new_val = 0;
			for (int j = 0; j < d_gapdata->n; j++) {
				new_val += d_c[sols[idx][j]][j];
			}
			for (int i = 0; i < d_gapdata->m; i++) {
				new_val -= INFEASIBLE_COST * min(0, rest_b[i]);
			}

			// 如果已经搜索到局部最优解了，则会跳出循环
			if (new_val >= pre_val) {
				impr++;
			}
			else {
				pre_val = new_val;
				impr = 0;
			}
		}
		
		// printf("pre_val:%d\n", pre_val);

	/*	printf("d__sols%d:\n", idx);
		for (int j = 0; j < d_gapdata->n; j++) {
			printf("%d,%d,%d ", idx, j, sols[idx][j]);
		}
		printf("\n");

		printf("d_rest_b_:\n");
		for (int j = 0; j < d_gapdata->m; j++) {
			printf("%d %d ", idx, rest_b[j]);
		}*/
	}

}

/***** main ******************************************************************/
int main(int argc, char *argv[])
{
	Param		param;		/* parameters */
	GAPdata		h_gapdata;	/* GAP instance data */
	Vdata		vdata;		/* various data often needed during search */

	cudaError_t err = cudaSuccess;

	vdata.timebrid = clock();
	copy_parameters(argc, argv, &param);
	read_instance(&h_gapdata);
	prepare_memory(&vdata, &h_gapdata);
	if (param.givesol == 1) { read_sol(&vdata, &h_gapdata); }
	vdata.starttime = clock();

	// 解的数量，即并行运行的gpu线程数
	int numSols = 1024;

	// 给cpu上的解分配内存
	// 分配行指针
	int** h_sols = (int **)malloc(numSols * sizeof(int *));
	// 行指针对应具体的数据
	int* h_sols_data = (int *)malloc(sizeof(int) * numSols * h_gapdata.n);
	int** d_sols;
	int* d_sols_data;
	cudaMalloc((void**)&d_sols, sizeof(int **) * numSols);
	cudaMalloc((void**)&d_sols_data, sizeof(int) * numSols * h_gapdata.n);
	for (int i = 0; i < numSols; i++) {
		h_sols[i] = h_sols_data + h_gapdata.n * i;
	}

	// 消耗
	int** h_c = (int **)malloc(h_gapdata.m * sizeof(int *));
	int* h_c_data = (int *)malloc(sizeof(int) * h_gapdata.m * h_gapdata.n);
	int** d_c;
	int* d_c_data;
	cudaMalloc((void**)&d_c, sizeof(int **) * h_gapdata.m);
	cudaMalloc((void**)&d_c_data, sizeof(int) * h_gapdata.m * h_gapdata.n);
	for (int i = 0; i < h_gapdata.m; i++) {
		h_c[i] = h_c_data + h_gapdata.n * i;
	}
	for (int i = 0; i < h_gapdata.m; i++) {
		for (int j = 0; j < h_gapdata.n; j++) {
			h_c[i][j] = h_gapdata.c[i][j];
		}
	}
	for (int i = 0; i < h_gapdata.m; i++) {
		h_c[i] = d_c_data + h_gapdata.n * i;
	}
	cudaMemcpy(d_c, h_c, sizeof(int*) * h_gapdata.m, cudaMemcpyHostToDevice);
	cudaMemcpy(d_c_data, h_c_data, sizeof(int) * h_gapdata.m * h_gapdata.n, cudaMemcpyHostToDevice);

	// 资源成本
	int** h_a = (int **)malloc(h_gapdata.m * sizeof(int *));
	int* h_a_data = (int *)malloc(sizeof(int) * h_gapdata.m * h_gapdata.n);
	int** d_a;
	int* d_a_data;
	cudaMalloc((void**)&d_a, sizeof(int **) * h_gapdata.m);
	cudaMalloc((void**)&d_a_data, sizeof(int) * h_gapdata.m * h_gapdata.n);
	for (int i = 0; i < h_gapdata.m; i++) {
		h_a[i] = h_a_data + h_gapdata.n * i;
	}
	for (int i = 0; i < h_gapdata.m; i++) {
		for (int j = 0; j < h_gapdata.n; j++) {
			h_a[i][j] = h_gapdata.a[i][j];
		}
	}
	for (int i = 0; i < h_gapdata.m; i++) {
		h_a[i] = d_a_data + h_gapdata.n * i;
	}
	cudaMemcpy(d_a, h_a, sizeof(int*) * h_gapdata.m, cudaMemcpyHostToDevice);
	cudaMemcpy(d_a_data, h_a_data, sizeof(int) * h_gapdata.m * h_gapdata.n, cudaMemcpyHostToDevice);

	// 可用资源
	int* d_b;
	cudaMalloc((void**)&d_b, sizeof(int) * h_gapdata.n);
	err = cudaMemcpy(d_b, h_gapdata.b, sizeof(int) * h_gapdata.n, cudaMemcpyHostToDevice);

	GAPdata *d_gapdata;
	cudaMalloc((void**)&d_gapdata, sizeof(GAPdata));

	err = cudaMemcpy(d_gapdata, &h_gapdata, sizeof(GAPdata), cudaMemcpyHostToDevice);

	int count = 0;
	int best_cost = INT_MAX;
	int *rest_b = (int *)malloc_e(h_gapdata.m * sizeof(int));

	// 设置grid和block
	dim3 grid(4, 4);
	//dim3 block(128, 16);
	dim3 block(128, 4);
	//dim3 block(1, 1);

	//int threadsPerBlock = 1;
	//// blocksPerGrid 有多少个block
	//int blocksPerGrid = (numSols + threadsPerBlock - 1) / threadsPerBlock;

	while (((clock() - vdata.starttime)) / CLOCKS_PER_SEC < param.timelim) {
		count++;

		// 初始化k个解
		for (int i = 0; i < numSols; i++)
		{
			greedy_init(h_sols[i], &h_gapdata);
		}

		/*	for (int i = 0; i < numSols; i++) {
				printf("\nh_init_sols:%d\n", i);
				for (int j = 0; j < h_gapdata.n; j++) {
					printf(" %d", h_sols[i][j]);
				}
			}*/

		for (int i = 0; i < numSols; i++)
		{
			h_sols[i] = d_sols_data + h_gapdata.n * i;
		}

		//// 将k个解拷贝到gpu空间
		cudaMemcpy(d_sols, h_sols, sizeof(int*) * numSols, cudaMemcpyHostToDevice);
		cudaMemcpy(d_sols_data, h_sols_data, sizeof(int) * numSols * h_gapdata.n, cudaMemcpyHostToDevice);

		// 调用gpu函数，在初始解中进行并行搜索
		// 设定block数、线程数；
		// 每个线程搜索一个初始解
		// 设定线程与block数的方法：先设定 threadsPerBlock 为某个值，然后使用公式计算blocksPerGrid
		// threadsPerBlock 每个block的线程数

		curandState *devStates;
		cudaMalloc((void**)&devStates, numSols * h_gapdata.n * sizeof(curandState));
		int seed = rand();

		setup_kernal << < grid, block >> > (devStates, seed);

		//cudaEvent_t start, stop;
		//cudaEventCreate(&start);
		//cudaEventCreate(&stop);
		//cudaEventRecord(start, 0);
		
		// 此函数耗时过多
		searchSol << < grid, block >> > (d_sols, d_gapdata, numSols, devStates, d_c, d_a, d_b);

		//cudaEventRecord(stop, 0);
		//cudaEventSynchronize(stop);

		// 将解拷贝回cpu
		err = cudaMemcpy(h_sols_data, d_sols_data, numSols * h_gapdata.n * sizeof(int), cudaMemcpyDeviceToHost);

		//float time;
		//cudaEventElapsedTime(&time, start, stop);
		//printf("gpu time = %3.1f ms\n", time);

		// 行指针
		for (int i = 0; i < numSols; i++)
		{
			h_sols[i] = h_sols_data + h_gapdata.n * i;
		}

		/*for (int i = 0; i < numSols; i++) {
			printf("\nh_sols:%d\n",i);
			for (int j = 0; j < h_gapdata.n; j++) {
				printf(" %d", h_sols[i][j]);
			}
		}*/

		int cost;
		// 计算k个解中的最优解
		for (int i = 0; i < numSols; i++)
		{
			for (int k = 0; k < h_gapdata.m; k++)
			{
				rest_b[k] = h_gapdata.b[k];
			}

			// 对每个解，计算其花销和计算剩余资源
			cost = 0;
			cost = calculate_cost(h_sols[i], &h_gapdata);
			// printf("idx:%d, cost:%d\n", i, cost);

			for (int k = 0; k < h_gapdata.n; k++)
			{
				rest_b[h_sols[i][k]] -= h_gapdata.a[h_sols[i][k]][k];
			}

			//printf("\nrest_b:");
			//for (int k = 0; k < h_gapdata.m; k++) {
			//	printf(" %d", rest_b[k]);
			//}

			//if (cost < best_cost)
			//{
			//	printf("cost:%d\n", cost);
			//	printf("restb:");
			//	for (int i = 0; i < h_gapdata.m; i++) {
			//		printf("%d ", rest_b[i]);
			//	}
			//	printf("\n");
			//}

			// 如果发现更优可行解则更新
			if (cost < best_cost && is_feasible(rest_b, &h_gapdata))
			{
				for (int j = 0; j < h_gapdata.n; j++) {
					vdata.bestsol[j] = h_sols[i][j];
				}
				best_cost = cost;
			}

		}

		printf("count:%d, best cost:%d, time: %f\n", count, best_cost, (clock() - vdata.starttime) / CLOCKS_PER_SEC);
	}

	printf("DONE! BestCost: %d Time: %f\n", best_cost, (clock() - vdata.starttime) / CLOCKS_PER_SEC);

	vdata.endtime = clock();
	recompute_cost(&vdata, &h_gapdata);
	free_memory(&vdata, &h_gapdata);
	free((void *)rest_b);
	free((void *)h_sols);

	cudaFree(d_sols);

	return EXIT_SUCCESS;
}