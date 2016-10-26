#include "Benchmarks.h"
//#define BLOCK_SIZE 1024

__global__ void ackley_kernel(double *x, double *sum1, double *sum2, int dim){

	__shared__ double sm[BLOCK_SIZE];
	__shared__ double sm_cos[BLOCK_SIZE];

	int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;

	sm[tid] = x[global_tid];

	// initialization of sign, hat, c1, c2
	// sign
	int sign;
	if( sm[tid] == 0 )
		sign = 0;
	else
		sign = sm[tid]>0?1:-1;

	// hat
	double hat;
	if(sm[tid] == 0)
		hat = 0;
	else
		hat = log(abs(sm[tid]));

	// c1
	double c1;
	if(sm[tid]>0)
		c1 = 10;
	else
		c1 = 5.5;

	// c2
	double c2;
	if( sm[tid]>0 )
		c2 = 7.9;
	else
		c2=3.1;

	// transform osz
	sm[tid] = sign * exp(hat + 0.049 * (sin(c1*hat) + sin(c2*hat)));
	// transform asy
	sm[tid] = pow(sm[tid], 1+0.2* tid/(double)(dim-1)) * sqrt(sm[tid]);
	// lambda
	sm[tid] = sm[tid] * pow( 10.0, 0.5* tid/(double)(dim-1) );

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

double ackley_GPU_kernel(double *x, int dim){

	int blk_cnt = (dim + BLOCK_SIZE - 1)/BLOCK_SIZE;
	double *h_sum1, *h_sum2;
	double *d_x, *d_sum1, *d_sum2;
	double ackley, sum1, sum2;

	h_sum1 = (double*)malloc(dim*sizeof(double));
	h_sum2 = (double*)malloc(dim*sizeof(double));

	cudaMalloc(&d_x, dim*sizeof(double));

	cudaMalloc(&d_sum1, blk_cnt * sizeof(double));
	cudaMalloc(&d_sum2, blk_cnt * sizeof(double));

	cudaMemcpy(d_x, x, dim*sizeof(double), cudaMemcpyHostToDevice);

	dim3 grid(blk_cnt);
	dim3 block(BLOCK_SIZE);

	ackley_kernel<<< grid, block >>>(d_x, d_sum1, d_sum2, dim);

	//cudaMalloc(h_sum1, d_sum1, blk_cnt * sizeof(double));
	//cudaMalloc(h_sum2, d_sum2, blk_cnt * sizeof(double));


	cudaMemcpy(h_sum1, d_sum1, dim*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(h_sum2, d_sum2, dim*sizeof(double), cudaMemcpyHostToDevice);

	for(int i=0; i<blk_cnt; i++){
		sum1 += h_sum1[i];
		sum2 += h_sum2[i];
	}

	cudaFree(d_x);
	cudaFree(d_sum1);
	cudaFree(d_sum2);

	free(h_sum1);
	free(h_sum2);

	ackley =  -20.0* exp(-0.2 * sqrt(sum1/dim)) - exp(sum2/dim) + 20.0 + E;
	return ackley;
}
