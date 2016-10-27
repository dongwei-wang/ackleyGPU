#include "Benchmarks.h"

extern "C"
double ackley_GPU_impl(double *x, int dim);
extern "C"
double ackley_CPU_impl(double *x, int dim);

Benchmarks::Benchmarks(){
	dimension = 1024;
}

Benchmarks::~Benchmarks(){
	// delete[] anotherz;
	// delete[] anotherz1;
	// delete[] anotherz2;
}

void Benchmarks::transform_osz(double* z, int dim)
{
	// apply osz transformation to z
	for (int i = 0; i < dim; ++i)
	{
		z[i] = sign(z[i]) * exp( hat(z[i]) + 0.049 * ( sin( c1(z[i]) * hat(z[i]) ) + sin( c2(z[i])* hat(z[i]) )  ) ) ;
	}
}

void Benchmarks::transform_asy(double* z, double beta, int dim)
{
	for (int i = 0; i < dim; ++i){
		if (z[i]>0){
			z[i] = pow(z[i], 1 + beta * i/((double) (dim-1)) * sqrt(z[i]) );
		}
	}
}

void Benchmarks::Lambda(double* z, double alpha, int dim)
{
	for (int i = 0; i < dim; ++i){
		z[i] = z[i] * pow(alpha, 0.5 * i/((double) (dim-1)) );
	}
}

unsigned Benchmarks::getID(){
    return ID;
}

int Benchmarks::getDimension(){
	return dimension;
}

// ackley function for single group non-separable
double Benchmarks::ackley(double *z,int dim){
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
	return sum;
}


double Benchmarks::ackley_GPU(double *x, int dim){
	return ackley_GPU_impl(x, dim);
}

double Benchmarks::ackley_CPU(double *x, int dim){
	return ackley_CPU_impl(x, dim);
}

int Benchmarks::sign(double x)
{
	if (x > 0) return 1;
	if (x < 0) return -1;
	return 0;
}

double Benchmarks::hat(double x)
{
	if (x==0){
		return 0;
	}
	else{
		return log(abs(x));
	}
}

double Benchmarks::c1(double x)
{
	if (x>0){
		return 10;
	}
	else{
		return 5.5;
	}
}

double Benchmarks::c2(double x)
{
	if (x>0){
		return 7.9;
	}
	else{
		return 3.1;
	}
}
