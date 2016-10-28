#include "Header.h"
#include "F3.h"

unsigned linecnt(const char *filename){
	unsigned linecnt = 0;
	string line;
	ifstream myfile(filename);
	while(getline(myfile, line))
		linecnt++;
	return linecnt;
}

double* readOvector(int dimension, char* filename) {
	// read O vector from file in csv format
	double* d = new double[dimension];
	stringstream ss;
	//ss<< "cdatafiles/" << "F" << ID << "-xopt.txt";
	ss<<filename;
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

int main(int argc, char *argv[]){
	double *X_CPU, *X_GPU, *X_GPU_STREAM;
	F3* fp = NULL;
	fp = new F3();
	for (int i=1; i<argc; i++){
		cout<<"Processing file "<<argv[i]<<" ...... "<<endl;
		unsigned int instance_cnt = linecnt(argv[i]);
		cout<<"The size of input is : "<<instance_cnt<<endl;
		X_CPU = readOvector(instance_cnt, argv[i]);
		X_GPU = readOvector(instance_cnt, argv[i]);
		X_GPU_STREAM = readOvector(instance_cnt, argv[i]);
		fp->compute_CPU(X_CPU, instance_cnt);
		fp->compute_GPU(X_GPU, instance_cnt);
		fp->compute_GPU_steams(X_GPU_STREAM, instance_cnt, 4);
		cout<<"Processing END"<<endl<<endl;
		delete []X_CPU;
		delete []X_GPU;
	}
	delete fp;
	return 0;
}
