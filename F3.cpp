/*
 Shifted Ackley's Function
*/

#include "F3.h"
F3::F3() : Benchmarks(){
	Ovector = NULL;
	minX = -32;
	maxX = 32;
	ID = 3;
	//anotherz = new double[dimension];
}

F3::~F3(){
	delete[] Ovector;
	//delete[] anotherz;
}

double F3::compute(double *x){
	return ackley(x, dimension);
}

double F3::compute_CPU(double *x){
	return ackley_CPU(x, dimension);
}

double F3::compute_GPU(double *x){
	return ackley_GPU(x, dimension);
}
