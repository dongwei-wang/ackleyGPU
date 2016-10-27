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
	double *X_CPU, *X_GPU;
	F3* fp = NULL;
	double ackley_CPU, ackley_GPU;

	fp = new F3();
	X_CPU = readOvector(fp->getDimension(), fp->getID());
	X_GPU = readOvector(fp->getDimension(), fp->getID());
	ackley_CPU = fp->compute_CPU(X_CPU);
	ackley_GPU = fp->compute_GPU(X_GPU);

	delete fp;
	delete []X_CPU;
	delete []X_GPU;
	return 0;
}

