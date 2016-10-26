/**
 * Shifted Ackley's Function
 *
 * as defined in "Benchmark Functions for the CEC'2010 Special Session
 * and Competition on Large-Scale Global Optimization" by Ke Tang,
 * Xiaodong Li, P. N. Suganthan, Zhenyu Yang, and Thomas Weise
 * published as technical report on January 8, 2010 at Nature Inspired
 * Computation and Applications Laboratory (NICAL), School of Computer
 * Science and Technology, University of Science and Technology of China,
 * Hefei, Anhui, China.
 */

#include "F3.h"

F3::F3() : Benchmarks(){
	Ovector = NULL;
	minX = -32;
	maxX = 32;
	ID = 3;
	anotherz = new double[dimension];
}

F3::~F3(){
	delete[] Ovector;
	delete[] anotherz;
}

double F3::compute(double*x){
	if(Ovector == NULL) {
		Ovector = readOvector();
	}

	for(int i = dimension - 1; i >= 0; i--) {
		anotherz[i] = 0 - Ovector[i];
	}

	// return ackley_GPU(anotherz, dimension);
	return ackley(x, dimension);
}

// double* F3::readfile(){
//     if(Ovector == NULL){
//         Ovector = readOvector();
//         for( int i = dimension -1; i>=0; i-- )
//             anotherz[i] = 0-Ovector[i];
//     }
//     return Ovector;
// }


