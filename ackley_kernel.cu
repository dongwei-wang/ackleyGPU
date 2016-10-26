#include "Benchmarks.h"

extern "C"
double ackley_GPU_impl(double *x, int dim);

__global__ void ackley_kernel(double *d_x, double *sum1, double *sum2, int dim){
	__shared__ double sm[BLOCK_SIZE];
	__shared__ double sm_cos[BLOCK_SIZE];

	int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;

	sm[tid] = d_x[global_tid];

	// initialization of sign, hat, c1, c2
	int sign;
	if( sm[tid] == 0 )
		sign = 0;
	else
		sign = sm[tid]>0 ? 1:-1;

	double hat;
	if(sm[tid] == 0)
		hat = 0;
	else
		hat = log(abs(sm[tid]));

	double c1;
	if(sm[tid]>0)
		c1 = 10;
	else
		c1 = 5.5;

	double c2;
	if( sm[tid]>0 )
		c2 = 7.9;
	else
		c2=3.1;

	// transform osz
	sm[tid] = sign * exp(hat + 0.049 * (sin(c1*hat) + sin(c2*hat)));

	// transform asy
	if(sm[tid]>0)
		sm[tid] = pow(sm[tid], 1+0.2* global_tid/(double)(dim-1) * sqrt(sm[tid]));

	// lambda
	sm[tid] = sm[tid] * pow( 10.0, 0.5* global_tid/((double)(dim-1)) );

	// cos(2.0 * pi * x[i])
	sm_cos[tid] = cos(2.0 * PI * sm[tid]);

	// x square
	sm[tid] = sm[tid]*sm[tid];

	__syncthreads();

	// reduction
	for( int i=BLOCK_SIZE/2; i>0; i>>=1 ){
		if(tid<i){
			sm[tid] += sm[tid+i];
			sm_cos[tid] += sm_cos[tid+i];
		}
	}
	__syncthreads();

	// get the value from first element of shared memory
	if(tid == 0){
		sum1[blockIdx.x] = sm[tid];
		sum2[blockIdx.x] = sm_cos[tid];
	}
}

double ackley_GPU_impl(double *x, int dim){
	int blk_cnt = (dim + BLOCK_SIZE - 1)/BLOCK_SIZE;
	double *h_sum1, *h_sum2;
	double *d_x, *d_sum1, *d_sum2;
	double ackley=0;
	double sum1=0;
	double sum2=0;

	h_sum1 = (double*)malloc(blk_cnt*sizeof(double));
	h_sum2 = (double*)malloc(blk_cnt*sizeof(double));

	cudaMalloc(&d_x, dim*sizeof(double));
	cudaMalloc(&d_sum1, blk_cnt * sizeof(double));
	cudaMalloc(&d_sum2, blk_cnt * sizeof(double));
	cudaMemcpy(d_x, x, dim*sizeof(double), cudaMemcpyHostToDevice);

	dim3 grid(blk_cnt);
	dim3 block(BLOCK_SIZE);

	/* printf("The size of dim is %d\n", dim); */
	/* printf("Kernel start\n"); */
	/* printf("Block cnt is %d\n", blk_cnt); */
	ackley_kernel<<< grid, block >>>(d_x, d_sum1, d_sum2, dim);
	//printf("kernel end\n");

	cudaMemcpy(h_sum1, d_sum1, blk_cnt*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_sum2, d_sum2, blk_cnt*sizeof(double), cudaMemcpyDeviceToHost);

	cudaError_t cudaError = cudaGetLastError();
	if(cudaError != cudaSuccess){
		fprintf(stderr, "cudaGetLastError() return %d: %s\n", cudaError, cudaGetErrorString(cudaError));
		exit(EXIT_FAILURE);
	}

	for(int i=0; i<blk_cnt; i++){
		sum1 += h_sum1[i];
		sum2 += h_sum2[i];
	}

	cudaFree(d_x);
	cudaFree(d_sum1);
	cudaFree(d_sum2);

	free(h_sum1);
	free(h_sum2);

	ackley =  -20.0 * exp(-0.2 * sqrt(sum1/dim)) - exp(sum2/dim) + 20.0 + E;
	return ackley;
}
