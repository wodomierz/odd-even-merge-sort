#include <cstdio>



extern "C" {

__device__
static int THREADS_IN_BLOCK = 1024;

__device__
void min_max(int* tab, int for_min, int for_max, int size) {
	if (for_min >= size || for_max >= size) {
		return;
	}
	int min = tab[for_min];
	int max = tab[for_max];
	if (max < min) {
		atomicExch(tab + for_max, min);
		atomicExch(tab + for_min, max);
	}
};


__global__ 
void odd_even_phase1(int* to_sort, int batch_size, int size) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int thid = x + y*gridDim.x*blockDim.x;
	if (thid >= size) {
		return;
	}
	int local_thid = thid % batch_size;
	int opposite = thid + batch_size / 2;

	if (local_thid < batch_size / 2) {
		min_max(to_sort, thid,  opposite, size);
	}

}

__global__ 
void odd_even_phase2(int* to_sort, int d, int batch_size, int size) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int thid = x + y*gridDim.x*blockDim.x;
	if (thid >= size) {
		return;
	}
	int local_thid = thid % batch_size;


	if (local_thid < d || local_thid + d >= batch_size - 1) {
		return;
	}

	int opposite = thid + d;
	if (local_thid % (2*d) < d ) {
		min_max(to_sort, thid,  opposite, size);
	}

}



}



