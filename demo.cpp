#include "Header.h"
#include <sys/time.h>
#include <cstdio>
#include <unistd.h>

int main(){
	/*  Test the basic benchmark function */
	double *X_CPU, *X_GPU;
	//Benchmarks* fp=NULL;
	F3* fp = NULL;

	struct timeval start, end;
	long seconds, useconds;
	double mtime;
	double ackley_CPU, ackley_GPU;

	fp = new F3();
	X_CPU = fp->readOvector();
	X_GPU = fp->readOvector();
	gettimeofday(&start, NULL);
	ackley_CPU = fp->compute(X_CPU);
	ackley_GPU = fp->compute_GPU(X_GPU);
	gettimeofday(&end, NULL);

	seconds  = end.tv_sec  - start.tv_sec;
	useconds = end.tv_usec - start.tv_usec;
	mtime = (((seconds) * 1000 + useconds/1000.0) + 0.5)/1000;
	printf("********** CPU IMPLEMENTATION **********\n");
	printf("F %d value = %1.20E\n", fp->getID(), ackley_CPU);
	printf("F %d Running Time = %f s\n\n", fp->getID(), mtime);

	printf("********** GPU IMPLEMENTATION **********\n");
	printf("F %d value = %1.20E\n\n", fp->getID(), ackley_GPU);

	delete fp;
	delete []X_CPU;
	delete []X_GPU;
	return 0;
}

