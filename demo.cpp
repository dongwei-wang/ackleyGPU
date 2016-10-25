#include "Header.h"
#include <sys/time.h>
#include <cstdio>
#include <unistd.h>

int main(){
	/*  Test the basic benchmark function */
	double* X;
	Benchmarks* fp=NULL;
	unsigned dim = 1000;
	unsigned run = 1;

	struct timeval start, end;
	long seconds, useconds;
	double mtime;
	double ackley;

	X = new double[dim];
	for (unsigned i=0; i<dim; i++){
		X[i]=0;
	}

	fp = new F3();
	gettimeofday(&start, NULL);
	for (unsigned j=0; j < run; j++){
		ackley = fp->compute(X);
	}
	gettimeofday(&end, NULL);

	seconds  = end.tv_sec  - start.tv_sec;
	useconds = end.tv_usec - start.tv_usec;

	mtime = (((seconds) * 1000 + useconds/1000.0) + 0.5)/1000;

	printf("F %d value = %1.20E\n", fp->getID(), ackley);
	printf("F %d Running Time = %f s\n\n", fp->getID(), mtime);

	delete fp;
	delete []X;
	return 0;
}

