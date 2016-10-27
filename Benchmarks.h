#ifndef _BENCHMARKS_H
#define _BENCHMARKS_H

#include <sstream>
#include <vector>
#include <fstream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <ctime>
using namespace std;

#define PI (3.141592653589793238462643383279)
#define E  (2.718281828459045235360287471352)
#define BLOCK_SIZE (256)

#define L(i) ((int64_t)i)
#define D(i) ((double)i)

struct IndexMap{
	unsigned arrIndex1;
	unsigned arrIndex2;
};

class Benchmarks{
	protected:
		unsigned ID;

		double ackley(double*x,int dim);
		double ackley_GPU(double *x, int dim);
		double ackley_CPU(double *x, int dim);

		double *Ovector;

		double* anotherz;
		double* anotherz1;
		double* anotherz2;
		int dimension;

	public:
		Benchmarks();
		virtual ~Benchmarks();
		virtual double compute(double* x){return 0;};

		unsigned getID();
		int getDimension();

		void transform_osz(double* z, int dim);
		void transform_asy(double* z, double beta, int dim);
		void Lambda(double* z, double alpha, int dim);
		int sign(double x);
		double hat(double x);
		double c1(double x);
		double c2(double x);

};

#endif
