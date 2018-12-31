#pragma once
#include "Function.cuh"

class Pooling
{
public:

	string name;
	string type;
	char txt;
	int stride;
	int dev;
	int output_size;
	int input_size;
	int train_flg;


	//size and dimension
	int XN, XC, XH, XW;
	int FH, FW;
	int OH, OW;

	int *argMax;

	//member function
	Pooling();
	~Pooling();
	void init(string name, int* x_shape, int kernel_size, const int stride, int dev,
		string type, int train_flg);
	float* forward_cpu(float* x);
	float* forward_gpu(float* x);
	float* backward_cpu(float* dout);
	float* backward_gpu(float* dout);
};

