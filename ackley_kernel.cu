#include "Benchmarks.h"

extern "C"
double ackley_GPU_impl(double *x, int dim);
extern "C"
double ackley_CPU_impl(double *x, int dim);

int sign(double x)
{
	if (x > 0) return 1;
	if (x < 0) return -1;
	return 0;
}

double hat(double x) {
	if (x==0){
		return 0;
	}
	else{
		return log(abs(x));
	}
}

double c1(double x)
{
	if (x>0){
		return 10;
	}
	else{
		return 5.5;
	}
}

double c2(double x)
{
	if (x>0){
		return 7.9;
	}
	else{
		return 3.1;
	}
}

__global__ void ackley_kernel(double *d_x, double *sum1, double *sum2, int dim){

	// shared memory
	__shared__ double sm[BLOCK_SIZE];
	__shared__ double sm_cos[BLOCK_SIZE];

	// global thread index
	int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;

	if(global_tid < dim){
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
			__syncthreads();
		}

		// get the value from first element of shared memory
		if(tid == 0){
			sum1[blockIdx.x] = sm[tid];
			sum2[blockIdx.x] = sm_cos[tid];
		}
	}
}

double ackley_GPU_impl(double *x, int dim){
	int blk_cnt = (dim + BLOCK_SIZE - 1)/BLOCK_SIZE;
	double *h_sum1, *h_sum2;
	double *d_x, *d_sum1, *d_sum2;
	double ackley;
	double sum1=0;
	double sum2=0;

	h_sum1 = (double*)malloc(blk_cnt*sizeof(double));
	h_sum2 = (double*)malloc(blk_cnt*sizeof(double));

	cudaMalloc(&d_x, dim*sizeof(double));
	cudaMalloc(&d_sum1, blk_cnt * sizeof(double));
	cudaMalloc(&d_sum2, blk_cnt * sizeof(double));

	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

	dim3 grid(blk_cnt);
	dim3 block(BLOCK_SIZE);

	cudaEventRecord(start);
	cudaMemcpy(d_x, x, dim*sizeof(double), cudaMemcpyHostToDevice);
	ackley_kernel<<< grid, block >>>(d_x, d_sum1, d_sum2, dim);
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
	ackley = -20.0 * exp(-0.2 * sqrt(sum1 / dim)) - exp(sum2 / dim) + 20.0 + E;

	//ackley =  -20.0 * exp(-0.2 * sqrt(sum1/dim)) - exp(sum2/dim) + 20.0 + E;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("/***** GPU IMPLEMENTATION *****/\n");
	printf("GPU time for ackley shift is %1.20E, time is %f ms\n\n", ackley, milliseconds );

	cudaFree(d_x);
	cudaFree(d_sum1);
	cudaFree(d_sum2);

	free(h_sum1);
	free(h_sum2);
	return ackley;
}

double ackley_CPU_impl(double *z, int dim){
	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
	cudaEventRecord(start);

	double sum1 = 0.0;
	double sum2 = 0.0;
	double sum;
	int    i;

	// parallel start
	// T_{osz}
	//transform_osz(x,dim);
	for (int i = 0; i < dim; ++i){
		z[i] = sign(z[i]) * exp( hat(z[i]) + 0.049 * ( sin( c1(z[i]) * hat(z[i]) ) + sin( c2(z[i])* hat(z[i]) )  ) ) ;
	}

	// T_{asy}^{0.2}
	//transform_asy(x, 0.2, dim);
	for (int i = 0; i < dim; ++i){
		if (z[i]>0){
			z[i] = pow(z[i], 1 + 0.2 * i/((double) (dim-1)) * sqrt(z[i]) );
		}
	}

	// lambda
	// Lambda(x, 10, dim);
	for (int i = 0; i < dim; ++i){
		z[i] = z[i] * pow(10, 0.5 * i/((double) (dim-1)) );
	}

	for(i = dim - 1; i >= 0; i--) {
		sum1 += (z[i] * z[i]);
		sum2 += cos(2.0 * PI * z[i]);
	}
	// parallel end

	sum = -20.0 * exp(-0.2 * sqrt(sum1 / dim)) - exp(sum2 / dim) + 20.0 + E;

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("/***** CPU IMPLEMENTATION *****/\n");
	printf("CPU time for ackley shift is %1.20E, time is %f ms\n\n", sum, milliseconds );

	return sum;
}
