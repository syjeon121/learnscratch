#pragma once
#include "Function.cuh"

class Relu
{
public:
	int size;
	int* index;
	float negative_slope;
	int train_flg;
	string name;

	void init(string name, int dim, int* input_shape, int dev, int train_flg, float negative_slope=0);
	void forward_cpu(float* x);
	void forward_gpu(float* x);
	void backward_cpu(float* dout);
	void backward_gpu(float* dout);

	Relu();
	~Relu();
};

