#pragma once
#include "Function.cuh"

class Convolution
{
public:

	int train_flg;
	int nodeNum;	//parameter for He method
	int dev;
	char txt;
	float buffer;
	string name;


	//size and dimension
	int pad, stride;
	int XN, XC, XH, XW;
	int FN, FC, FH, FW;
	int OH, OW;
	int dXH;
	int dXW;
	int output_size;
	int input_size;
	
	int size_w, size_b;
	int size_dw, size_db;

	float *w, *b;
	float *dw, *db;
	float *col;
	float* host_w, *host_b;



	//member function
	void init(string name, int* x_shape, int num_output, int kernel_size, const int pad, const int stride, int dev, 
		int train_flg, int previous_size=0);
	void init_size();
	void init_cpu();
	void init_gpu();
	void init_trained_parameter(ifstream& fin_w, ifstream& fin_b);
	void save_trained_parameter(ofstream& fout_w, ofstream& fout_b);


	float* forward_cpu(float* x);
	float* forward_gpu(float* x);
	float* backward_cpu(float* dout);
	float* backward_gpu(float* dout);
	Convolution();
	~Convolution();
};

