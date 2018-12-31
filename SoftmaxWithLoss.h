#pragma once
#include "Function.cuh"
class SoftmaxWithLoss
{
public:

	string name;
	int dev;


	//size and dimension
	int XN, XC, XH, XW, IN;
	int size_dx;


	//variables(cpu)
	float* y;
	int* t;
	float* loss;



	//member function
	void init(string name, int dim, int* x_shape, int dev);
	float forward_cpu(float* x, int* t);
	float forward_gpu(float* x, int* t);
	float* backward_cpu(float dout=1);
	float* backward_gpu(float dout=1);

	SoftmaxWithLoss();
	~SoftmaxWithLoss();
};

