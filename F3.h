#ifndef _F3_H
#define _F3_H

#include "Benchmarks.h"

class F3 : public Benchmarks{
	protected:

	public:
		F3();
		double compute(double* x) ;
		//double compute() ;
//		double* readfile();

		/* double compute(vector<double> x) ; */
		~F3();
};
#endif
