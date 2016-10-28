#ifndef _F3_H
#define _F3_H

#include "Header.h"

class F3 {

	public:
		F3();

		double compute(double* x, int dimensiont) ;
		double compute_CPU(double *x, int dimension);
		double compute_GPU(double *x, int dimension);
		double compute_GPU_steams(double *x, int dimension, int stream_cnt);

		void transform_osz(double *z, int dim);
		void transform_asy(double *z, double beta, int dim);
		void Lambda(double *z, double alpha, int dim);
		unsigned getID();
		double ackley(double *z, int dim);
		int sign(double x);
		double hat(double x);
		double c1(double x);
		double c2(double x);

		~F3();
	protected:

	public:
		double *Ovector;
		unsigned ID;
		int minX;
		int maxX;
};
#endif
