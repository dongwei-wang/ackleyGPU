#include "Header.h"
#include <sys/time.h>
#include <cstdio>
#include <unistd.h>


double* readOvector(int dimension, int ID) {
	// read O vector from file in csv format
	double* d = new double[dimension];
	stringstream ss;
	ss<< "cdatafiles/" << "F" << ID << "-xopt.txt";
	ifstream file (ss.str());
	string value;
	string line;
	int c=0;

	if (file.is_open()){
		stringstream iss;
		while ( getline(file, line) ){
			iss<<line;
			while (getline(iss, value, ',')){
				d[c++] = stod(value);
			}
			iss.clear();
		}
		file.close();
	}
	else{
		cout<<"Cannot open datafiles"<<endl;
	}
	return d;
}

int main(){
	/*  Test the basic benchmark function */
	double *X_CPU, *X_GPU;
	//Benchmarks* fp=NULL;
	F3* fp = NULL;

	// struct timeval start, end;
	// long seconds, useconds;
	// double mtime;
	double ackley_CPU, ackley_GPU;

	fp = new F3();
	X_CPU = readOvector(fp->getDimension(), fp->getID());
	X_GPU = readOvector(fp->getDimension(), fp->getID());

	//gettimeofday(&start, NULL);
	ackley_CPU = fp->compute_CPU(X_CPU);
	//gettimeofday(&end, NULL);

	// seconds  = end.tv_sec  - start.tv_sec;
	// useconds = end.tv_usec - start.tv_usec;
	// mtime = (((seconds) * 1000 + useconds/1000.0) + 0.5)/1000;
	// printf("********** CPU IMPLEMENTATION **********\n");
	// printf("F %d value = %1.20E\n\n", fp->getID(), ackley_CPU);
	// printf("F %d Running Time = %f s\n\n", fp->getID(), mtime);

	ackley_GPU = fp->compute_GPU(X_GPU);


	delete fp;
	delete []X_CPU;
	delete []X_GPU;
	return 0;
}

