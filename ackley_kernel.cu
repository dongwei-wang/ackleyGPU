#include "Header.h"

// cuda error check macro definition
#define ErrorCheck(stmt) {											 \
	cudaError_t err = stmt;                                          \
	if (err != cudaSuccess)                                          \
	{                                                                \
		printf( "Failed to run stmt %d ", __LINE__);                 \
		printf( "Got CUDA error ...  %s ", cudaGetErrorString(err)); \
		return -1;                                                   \
	}                                                                \
}


extern "C"
double ackley_GPU_impl(double *x, int dim);
extern "C"
double ackley_CPU_impl(double *x, int dim);
extern "C"
double ackley_GPU_streams(double *x, int dim, int stream_cnt);


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
	// local thread index in a block
	int tid = threadIdx.x;

	if(global_tid < dim)
		sm[tid] = d_x[global_tid];
	else
		sm[tid] = 0.0f;

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

double ackley_GPU_impl(double *x, int dim){
	int blk_cnt = (dim + BLOCK_SIZE - 1)/BLOCK_SIZE;
	double *h_sum1, *h_sum2;
	double *d_x, *d_sum1, *d_sum2;
	double ackley;
	double sum1=0;
	double sum2=0;

	h_sum1 = (double*)malloc(blk_cnt*sizeof(double));
	h_sum2 = (double*)malloc(blk_cnt*sizeof(double));

	ErrorCheck(cudaMalloc(&d_x, dim*sizeof(double)));
	ErrorCheck(cudaMalloc(&d_sum1, blk_cnt * sizeof(double)));
	ErrorCheck(cudaMalloc(&d_sum2, blk_cnt * sizeof(double)));

	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

	dim3 grid(blk_cnt);
	dim3 block(BLOCK_SIZE);

	cudaEventRecord(start);
	ErrorCheck(cudaMemcpy(d_x, x, dim*sizeof(double), cudaMemcpyHostToDevice));
	ackley_kernel<<< grid, block >>>(d_x, d_sum1, d_sum2, dim);
	ErrorCheck(cudaMemcpy(h_sum1, d_sum1, blk_cnt*sizeof(double), cudaMemcpyDeviceToHost));
	ErrorCheck(cudaMemcpy(h_sum2, d_sum2, blk_cnt*sizeof(double), cudaMemcpyDeviceToHost));

	for(int i=0; i<blk_cnt; i++){
		sum1 += h_sum1[i];
		sum2 += h_sum2[i];
	}
	ackley = -20.0 * exp(-0.2 * sqrt(sum1 / dim)) - exp(sum2 / dim) + 20.0 + E;

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);

	printf("/***** GPU IMPLEMENTATION *****/\n");
	printf("GPU implementation for ackley shift is		%1.20E, time is %f ms\n", ackley, milliseconds );

	cudaFree(d_x);
	cudaFree(d_sum1);
	cudaFree(d_sum2);

	free(h_sum1);
	free(h_sum2);
	return ackley;
}

double ackley_GPU_streams(double *x, int dim, int stream_cnt){
	if( stream_cnt <= 0 ){
		printf("The number of streams should greater than 0 \n");
	}

	double *h_x, *h_sum1, *h_sum2;
	double *d_x, *d_sum1, *d_sum2;
	double ackley;
	double sum1=0;
	double sum2=0;

	int blk_cnt = ( dim + BLOCK_SIZE - 1 )/BLOCK_SIZE;
	ErrorCheck(cudaMallocHost(&h_x,		dim*sizeof(double)));
	ErrorCheck(cudaMallocHost(&h_sum1,	blk_cnt*sizeof(double)));
	ErrorCheck(cudaMallocHost(&h_sum2,	blk_cnt*sizeof(double)));

	memcpy(h_x, x, dim*sizeof(double));

	ErrorCheck(cudaMalloc(&d_x, dim*sizeof(double)));
	ErrorCheck(cudaMalloc(&d_sum1, blk_cnt * sizeof(double)));
	ErrorCheck(cudaMalloc(&d_sum2, blk_cnt * sizeof(double)));

	cudaStream_t *streams = (cudaStream_t*)malloc(stream_cnt*sizeof(cudaStream_t));
	for( int i=0; i<stream_cnt; i++ )
		ErrorCheck(cudaStreamCreate(&streams[i]));

	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

	cudaEventRecord(start);

	int instances_per_stream = dim/stream_cnt;
	int blk_cnt_per_stream = (instances_per_stream + BLOCK_SIZE -1 )/BLOCK_SIZE;

	dim3 grid(blk_cnt);
	dim3 block(BLOCK_SIZE);

	for( int i=0; i<stream_cnt; i++ ){
		ErrorCheck(cudaMemcpyAsync(&d_x[i*instances_per_stream],
					&h_x[i*instances_per_stream],
					instances_per_stream*sizeof(double),
					cudaMemcpyHostToDevice,
					streams[i]));

		// grid and block size is the same with nostreams version
		// add parameter three and four in multiple streams version
		ackley_kernel<<< grid, block, 0, streams[i] >>>(d_x, d_sum1, d_sum2, dim);

		ErrorCheck(cudaMemcpyAsync(&h_sum1[i*blk_cnt_per_stream],
					&d_sum1[i*blk_cnt_per_stream],
					blk_cnt_per_stream*sizeof(double),
					cudaMemcpyDeviceToHost,
					streams[i]));

		ErrorCheck(cudaMemcpyAsync(&h_sum2[i*blk_cnt_per_stream],
					&d_sum2[i*blk_cnt_per_stream],
					blk_cnt_per_stream*sizeof(double),
					cudaMemcpyDeviceToHost,
					streams[i]));
	}
	cudaDeviceSynchronize();

	for(int i=0; i<blk_cnt; i++){
		sum1 += h_sum1[i];
		sum2 += h_sum2[i];
	}
	ackley = -20.0 * exp(-0.2 * sqrt(sum1 / dim)) - exp(sum2 / dim) + 20.0 + E;

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);

	printf("/***** GPU STREAMS IMPLEMENTATION *****/\n");
	printf("GPU streams implementation for ackley shift is	%1.20E, time is %f ms\n", ackley, milliseconds );
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
	printf("CPU implementation for ackley shift is		%1.20E, time is %f ms\n", sum, milliseconds );
	return sum;
}
