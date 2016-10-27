#ifndef _F3_H
#define _F3_H

#include "Benchmarks.h"

class F3 : public Benchmarks{
	protected:

	public:
		F3();

		double compute(double* x) ;
		double compute_CPU(double *x);
		double compute_GPU(double *x);
		~F3();
		int minX;
		int maxX;
};
#endif
