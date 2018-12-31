#pragma once
#include "Function.cuh"
class SGD
{
public:

	int the_number_of_layer;
	int* parameter_shape;
	float** parameter;
	float** gradient;

	void init(int* parameter_shape, int the_number_of_layer);
	void update_cpu(float lr, float** parameter, float** gradient);
	void update_gpu(float lr, float** parameter, float** gradient);

	SGD();
	~SGD();

};

