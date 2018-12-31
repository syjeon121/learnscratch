#include "SGD.h"

SGD::SGD()
{

}

void SGD::init(int* parameter_shape, int the_number_of_layer)
{
	this->the_number_of_layer = the_number_of_layer;
	this->parameter_shape = parameter_shape;
}

void SGD::update_cpu(float lr, float** parameter, float** gradient)
{
	for (int i = 0; i < the_number_of_layer; i++)
	{
		update_sgd_cpu(lr, parameter[i*2 + 0], gradient[i*2 + 0], parameter_shape[i*2 + 0]);	//weight
		update_sgd_cpu(lr, parameter[i*2 + 1], gradient[i*2 + 1], parameter_shape[i*2 + 1]);	//bias
	}
}

void SGD::update_gpu(float lr, float** parameter, float** gradient)
{
	for (int i = 0; i < the_number_of_layer; i++)
	{
		update_sgd_gpu(lr, parameter[i*2 + 0], gradient[i*2 + 0], parameter_shape[i*2 + 0]);	//weight
		update_sgd_gpu(lr, parameter[i*2 + 1], gradient[i*2 + 1], parameter_shape[i*2 + 1]);	//bias
	}
}

SGD::~SGD()
{

}
